// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 769) 
v_lshrrev_b32 v3, 4, v0 // (op 769) 

v_and_b32 v0, 1, v3 // (op 769) 

v_and_b32 v3, 15, v1 // (op 769) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 769) 


v_lshrrev_b32 v0, 6, v1 // (op 769) 
v_lshl_add_u32 v3, v0, 5, v4 // (op 769) 


v_and_b32 v0, 32, v1 // (op 769) 
v_lshl_add_u32 v4, v3, 6, v0 // (op 769) 


v_lshlrev_b32 v0, 2, v4 // (op 769) 

v_lshrrev_b32 v3, 3, v0 // (op 769) 

v_and_b32 v0, 63, v1 // (op 822) 
v_and_b32 v4, 48, v1 // (op 822) 
v_sub_u32 v5, v0, v4 // (op 822) 

v_lshrrev_b32 v4, 5, v0 // (op 822) 
v_lshl_add_u32 v6, v4, 5, v5 // (op 822) 


v_lshrrev_b32 v4, 4, v0 // (op 822) 

v_and_b32 v0, -2, v4 // (op 822) 
v_sub_u32 v5, v4, v0 // (op 822) 


v_lshlrev_b32 v0, 4, v5 // (op 822) 

v_lshl_add_u32 v4, v6, 5, v0 // (op 822) 


v_mul_lo_u32 v0, v4, 6 // (op 822) 
v_lshrrev_b32 v5, 3, v0 // (op 822) 

v_lshrrev_b32 v0, 4, v4 // (op 822) 

v_lshlrev_b32 v4, 2, v0 // (op 822) 

v_add_u32 v0, v5, v4 // (op 822) 


v_lshrrev_b32 v4, 6, v1 // (op 875) 
v_and_b32 v5, 31, v1 // (op 875) 
v_lshl_add_u32 v6, v4, 5, v5 // (op 875) 


v_and_b32 v4, 63, v1 // (op 875) 
v_lshrrev_b32 v5, 5, v4 // (op 875) 

v_lshl_add_u32 v4, v6, 1, v5 // (op 875) 


v_and_b32 v5, 32, v1 // (op 928) 
v_and_b32 v6, 31, v1 // (op 928) 
v_add_u32 v7, v5, v6 // (op 928) 


v_and_b32 v5, 63, v1 // (op 986) 
v_lshrrev_b32 v6, 4, v5 // (op 986) 

v_and_b32 v5, 1, v6 // (op 986) 

v_and_b32 v6, 15, v1 // (op 986) 
v_lshl_add_u32 v8, v5, 4, v6 // (op 986) 


v_lshrrev_b32 v5, 6, v1 // (op 986) 
v_lshl_add_u32 v6, v5, 5, v8 // (op 986) 


v_and_b32 v5, 32, v1 // (op 986) 
v_lshl_add_u32 v8, v6, 6, v5 // (op 986) 


v_lshlrev_b32 v5, 2, v8 // (op 986) 

v_lshrrev_b32 v6, 3, v5 // (op 986) 

v_and_b32 v5, 63, v1 // (op 1010) 
v_and_b32 v8, 48, v1 // (op 1010) 
v_sub_u32 v9, v5, v8 // (op 1010) 

v_lshrrev_b32 v8, 5, v5 // (op 1010) 
v_lshl_add_u32 v10, v8, 5, v9 // (op 1010) 


v_lshrrev_b32 v8, 4, v5 // (op 1010) 

v_and_b32 v5, -2, v8 // (op 1010) 
v_sub_u32 v9, v8, v5 // (op 1010) 


v_lshlrev_b32 v5, 4, v9 // (op 1010) 

v_lshl_add_u32 v8, v10, 5, v5 // (op 1010) 


v_mul_lo_u32 v5, v8, 6 // (op 1010) 
v_lshrrev_b32 v9, 3, v5 // (op 1010) 

v_lshrrev_b32 v5, 4, v8 // (op 1010) 

v_lshlrev_b32 v8, 2, v5 // (op 1010) 

v_add_u32 v5, v9, v8 // (op 1010) 


v_lshrrev_b32 v8, 6, v1 // (op 1034) 
v_and_b32 v9, 31, v1 // (op 1034) 
v_lshl_add_u32 v10, v8, 5, v9 // (op 1034) 


v_and_b32 v8, 63, v1 // (op 1034) 
v_lshrrev_b32 v9, 5, v8 // (op 1034) 

v_lshl_add_u32 v8, v10, 1, v9 // (op 1034) 


v_and_b32 v9, 32, v1 // (op 1058) 
v_and_b32 v10, 31, v1 // (op 1058) 
v_add_u32 v11, v9, v10 // (op 1058) 


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 65) call()
v_accvgpr_write a5, 0.00000 // (op 65) call()
v_accvgpr_write a6, 0.00000 // (op 65) call()
v_accvgpr_write a7, 0.00000 // (op 65) call()
v_accvgpr_write a8, 0.00000 // (op 65) call()
v_accvgpr_write a9, 0.00000 // (op 65) call()
v_accvgpr_write a10, 0.00000 // (op 65) call()
v_accvgpr_write a11, 0.00000 // (op 65) call()
v_accvgpr_write a12, 0.00000 // (op 65) call()
v_accvgpr_write a13, 0.00000 // (op 65) call()
v_accvgpr_write a14, 0.00000 // (op 65) call()
v_accvgpr_write a15, 0.00000 // (op 65) call()
v_lshrrev_b32 v12, 1, v1 // (op 479) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 479) 

v_and_b32 v12, -2, v1 // (op 479) 
v_sub_u32 v14, v1, v12 // (op 479) 

v_lshlrev_b32 v12, 5, v14 // (op 479) 

v_mov_b32 v15, 0 // (op 479) convert
v_mov_b32 v14, v12 // (op 479) convert

v_mul_lo_u32 v17, v13, s3 // (op 479) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 479) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 479) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 479) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 479) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 479) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 479) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 479) 

v_mov_b32 v12, v14 // (op 479) convert

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

v_lshrrev_b32 v13, 1, v1 // (op 498) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 498) 

v_and_b32 v13, -2, v1 // (op 498) 
v_sub_u32 v15, v1, v13 // (op 498) 

v_lshlrev_b32 v13, 5, v15 // (op 498) 

v_mov_b32 v17, 0 // (op 498) convert
v_mov_b32 v16, v13 // (op 498) convert

v_mul_lo_u32 v19, v14, s3 // (op 498) most significant: low of low * high
v_mul_hi_u32 v13, v14, s2 // (op 498) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 498) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 498) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 498) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 498) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 498) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 498) 

v_mov_b32 v13, v14 // (op 498) convert

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

s_lshl_b32 s0, s17, 5 // (op 543) 
s_mov_b32 s15, 0 // (op 543) convert
s_mov_b32 s14, s0 // (op 543) convert

v_mul_lo_u32 v15, v1, s9 // (op 543) most significant: low of low * high
v_mul_hi_u32 v16, v1, s8 // (op 543) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 543) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 543) most significant: sum

v_mov_b32 v16, s14 // (op 543) 
v_mov_b32 v17, s15 // (op 543) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 543) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 543) most significant half




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

s_lshl_b32 s0, s17, 5 // (op 562) 
s_mov_b32 s15, 0 // (op 562) convert
s_mov_b32 s14, s0 // (op 562) convert

v_mul_lo_u32 v15, v1, s9 // (op 562) most significant: low of low * high
v_mul_hi_u32 v17, v1, s8 // (op 562) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 562) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 562) most significant: sum

v_mov_b32 v17, s14 // (op 562) 
v_mov_b32 v18, s15 // (op 562) 
v_add_co_u32 v14, s[0:1], v14, v17 // (op 562) least significant half
v_addc_co_u32 v15, s[0:1], v15, v18, s[0:1] // (op 562) most significant half




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

v_lshrrev_b32 v14, 1, v1 // (op 607) 
v_lshl_add_u32 v15, s16, 5, v14 // (op 607) 

v_and_b32 v14, -2, v1 // (op 607) 
v_sub_u32 v18, v1, v14 // (op 607) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v14, v15, s20 // (op 607) 
v_add_u32 v14, v14, v18 // (op 607) 


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

v_lshrrev_b32 v15, 1, v1 // (op 626) 
v_lshl_add_u32 v18, s16, 5, v15 // (op 626) 

v_and_b32 v15, -2, v1 // (op 626) 
v_sub_u32 v19, v1, v15 // (op 626) 

v_mul_lo_u32 v15, v18, s20 // (op 626) 
v_add_u32 v15, v15, v19 // (op 626) 


s_mul_i32 s1, s20, s7 // (op 628) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 628) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 628) least significant
s_mul_hi_u32 s15, s20, s6 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 628) most significant: sum
s_add_u32 s1, s1, s15 // (op 628) most significant: sum


s_mov_b32 s48, s22 // (op 628) reg expression
s_mov_b32 s49, s23 // (op 628) reg expression
s_mov_b32 s50, s0 // (op 628) reg expression
s_mov_b32 s51, 131072 // (op 628) call()

v_and_b32 v18, -32, v1 // (op 671) 
v_sub_u32 v19, v1, v18 // (op 671) 

v_lshl_add_u32 v18, s17, 5, v19 // (op 671) 

v_lshrrev_b32 v19, 5, v1 // (op 671) 
v_mul_lo_u32 v20, v19, s24 // (op 671) 
v_add_u32 v20, v20, v18 // (op 671) 


s_mul_i32 s1, s24, s29 // (op 673) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 673) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 673) least significant
s_mul_hi_u32 s15, s24, s28 // (op 673) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 673) most significant: sum
s_add_u32 s1, s1, s15 // (op 673) most significant: sum


s_mov_b32 s52, s26 // (op 673) reg expression
s_mov_b32 s53, s27 // (op 673) reg expression
s_mov_b32 s54, s0 // (op 673) reg expression
s_mov_b32 s55, 131072 // (op 673) call()

v_and_b32 v18, -32, v1 // (op 690) 
v_sub_u32 v19, v1, v18 // (op 690) 

v_lshl_add_u32 v18, s17, 5, v19 // (op 690) 

v_lshrrev_b32 v19, 5, v1 // (op 690) 
v_mul_lo_u32 v21, v19, s24 // (op 690) 
v_add_u32 v21, v21, v18 // (op 690) 


s_mul_i32 s1, s24, s29 // (op 692) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 692) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 692) least significant
s_mul_hi_u32 s15, s24, s28 // (op 692) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 692) most significant: sum
s_add_u32 s1, s1, s15 // (op 692) most significant: sum


s_mov_b32 s52, s26 // (op 692) reg expression
s_mov_b32 s53, s27 // (op 692) reg expression
s_mov_b32 s54, s0 // (op 692) reg expression
s_mov_b32 s55, 131072 // (op 692) call()



v_mov_b32 v18, v16 // (op 370) reg expression

buffer_load_dwordx3 v[22:24], v18, s[44:47], 0 offen // (op 370) Load value
buffer_load_dwordx3 v[26:28], v18, s[44:47], 0 offen offset:12 // (op 370) Load value



v_mov_b32 v25, v20 // (op 373) reg expression

buffer_load_ubyte v18, v25, s[52:55], 0 offen offset: 0 // (op 373) Load value




buffer_load_dwordx4 v[30:33], v12, s[40:43], 0 offen // (op 376) Load value



buffer_load_ubyte v19, v14, s[48:51], 0 offen offset: 0 // (op 379) Load value


s_lshl_b64 s[0:1], s[8:9], 6 // (op 382) 
s_mul_i32 s15, s1, 6 // (op 382) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 382) least significant
s_mul_hi_u32 s18, s0, 6 // (op 382) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 382) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 382) 

v_add_u32 v25, v16, s0 // (op 382) 


buffer_load_dwordx3 v[34:36], v25, s[44:47], 0 offen // (op 382) Load value
buffer_load_dwordx3 v[38:40], v25, s[44:47], 0 offen offset:12 // (op 382) Load value




s_lshl_b32 s0, s24, 1 // (op 385) 
v_add_u32 v16, v20, s0 // (op 385) 


buffer_load_ubyte v25, v16, s[52:55], 0 offen offset: 0 // (op 385) Load value





buffer_load_dwordx4 v[42:45], v12, s[40:43], 0 offen offset:32 // (op 388) Load value




buffer_load_ubyte v12, v14, s[48:51], 0 offen offset:2 // (op 391) Load value

v_lshlrev_b32 v14, 5, v1 // (op 810) 
v_mul_lo_u32 v16, v14, 6 // (op 810) 
v_lshrrev_b32 v20, 3, v16 // (op 810) 

v_lshrrev_b32 v16, 4, v14 // (op 810) 

v_lshlrev_b32 v14, 2, v16 // (op 810) 

v_add_u32 v16, v20, v14 // (op 810) 





s_waitcnt vmcnt(9)
ds_write_b96 v16, v[22:24] // (op 393) Store local data 
s_waitcnt vmcnt(8)
ds_write_b96 v16, v[26:28] offset:16 // (op 393) Store local data 

v_and_b32 v14, -32, v1 // (op 916) 
v_sub_u32 v16, v1, v14 // (op 916) 

v_and_b32 v14, 4294967264, v1 // (op 916) 
v_add_u32 v20, v14, v16 // (op 916) 





s_waitcnt vmcnt(7)
ds_write_b8 v20, v18 offset:2048 // (op 394) Store local data 

v_and_b32 v14, -2, v1 // (op 757) 
v_sub_u32 v16, v1, v14 // (op 757) 

v_lshlrev_b32 v14, 5, v16 // (op 757) 

v_lshrrev_b32 v16, 1, v1 // (op 757) 
v_lshl_add_u32 v20, v16, 6, v14 // (op 757) 


v_lshlrev_b32 v14, 2, v20 // (op 757) 

v_lshrrev_b32 v16, 3, v14 // (op 757) 




s_waitcnt vmcnt(6)
ds_write_b128 v16, v[30:33] offset:2112 // (op 395) Store local data 

v_and_b32 v14, -2, v1 // (op 863) 
v_sub_u32 v16, v1, v14 // (op 863) 

v_and_b32 v14, 4294967294, v1 // (op 863) 
v_add_u32 v20, v14, v16 // (op 863) 





s_waitcnt vmcnt(5)
ds_write_b8 v20, v19 offset:3136 // (op 396) Store local data 

v_and_b32 v14, -2, v1 // (op 728) 
v_sub_u32 v16, v1, v14 // (op 728) 

v_lshlrev_b32 v14, 5, v16 // (op 728) 

v_lshrrev_b32 v16, 1, v1 // (op 728) 
v_lshl_add_u32 v20, v16, 6, v14 // (op 728) 


v_lshlrev_b32 v14, 2, v20 // (op 728) 

v_lshrrev_b32 v16, 3, v14 // (op 728) 

v_lshlrev_b32 v14, 5, v1 // (op 781) 
v_mul_lo_u32 v20, v14, 6 // (op 781) 
v_lshrrev_b32 v29, 3, v20 // (op 781) 

v_lshrrev_b32 v20, 4, v14 // (op 781) 

v_lshlrev_b32 v14, 2, v20 // (op 781) 

v_add_u32 v20, v29, v14 // (op 781) 


v_and_b32 v14, -2, v1 // (op 834) 
v_sub_u32 v29, v1, v14 // (op 834) 

v_and_b32 v14, 4294967294, v1 // (op 834) 
v_add_u32 v37, v14, v29 // (op 834) 


v_and_b32 v14, -32, v1 // (op 887) 
v_sub_u32 v29, v1, v14 // (op 887) 

v_and_b32 v14, 4294967264, v1 // (op 887) 
v_add_u32 v41, v14, v29 // (op 887) 


v_and_b32 v14, -2, v1 // (op 974) 
v_sub_u32 v29, v1, v14 // (op 974) 

v_lshlrev_b32 v14, 5, v29 // (op 974) 

v_lshrrev_b32 v29, 1, v1 // (op 974) 
v_lshl_add_u32 v46, v29, 6, v14 // (op 974) 


v_lshlrev_b32 v14, 2, v46 // (op 974) 

v_lshrrev_b32 v29, 3, v14 // (op 974) 

v_lshlrev_b32 v14, 5, v1 // (op 998) 
v_mul_lo_u32 v46, v14, 6 // (op 998) 
v_lshrrev_b32 v47, 3, v46 // (op 998) 

v_lshrrev_b32 v46, 4, v14 // (op 998) 

v_lshlrev_b32 v14, 2, v46 // (op 998) 

v_add_u32 v46, v47, v14 // (op 998) 


v_and_b32 v14, -2, v1 // (op 1022) 
v_sub_u32 v47, v1, v14 // (op 1022) 

v_and_b32 v14, 4294967294, v1 // (op 1022) 
v_add_u32 v48, v14, v47 // (op 1022) 


v_and_b32 v14, -32, v1 // (op 1046) 
v_sub_u32 v47, v1, v14 // (op 1046) 

v_and_b32 v14, 4294967264, v1 // (op 1046) 
v_add_u32 v49, v14, v47 // (op 1046) 


s_mov_b32 s0, 0 // (op 61) call()
s_ashr_i64 s[14:15], s[12:13], 6 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60:


s_add_u32 s0, s0, 2 // (op 62) 
v_mov_b32 v14, v17 // (op 10) 
s_lshl_b64 s[14:15], s[8:9], 6 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s1, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s1, s14, 1 // (op 10) 

v_add_u32 v47, v14, s1 // (op 10) 


buffer_load_dwordx3 v[22:24], v47, s[44:47], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[26:28], v47, s[44:47], 0 offen offset:12 // (op 10) Load value

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1095) 





v_mov_b32 v47, v21 // (op 25) 
s_lshl_b32 s1, s24, 2 // (op 25) 
v_add_u32 v50, v47, s1 // (op 25) 


ds_read_b128 v[52:55], v3 offset:2112 // (op 72) Load local data 
buffer_load_ubyte v18, v50, s[52:55], 0 offen offset: 0 // (op 25) Load value



s_barrier  // (op 1108) 
v_mov_b32 v50, v13 // (op 4) 




ds_read_b96_tr_b6 v[56:58], v0 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[59:61], v0 offset:512 // (op 77) Transpose load local data 
buffer_load_dwordx4 v[30:33], v50, s[40:43], 0 offen offset:64 // (op 4) Load value


v_mov_b32 v51, v15 // (op 17) 

buffer_load_ubyte v19, v51, s[48:51], 0 offen offset:4 // (op 17) Load value
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1122) 



s_barrier  // (op 1159) 
ds_read_u8 v62, v4 offset:3136 // (op 82) Load local data 



s_barrier  // (op 1136) 
s_waitcnt vmcnt(9)
ds_write_b96 v46, v[34:36] offset:3200 // (op 257) Store local data 


ds_read_u8 v63, v7 offset:2048 // (op 87) Load local data 
s_waitcnt vmcnt(8)
ds_write_b96 v46, v[38:40] offset:3216 // (op 257) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[52:55], v[56:61], a[0:15], v62, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 90) 




s_barrier  // (op 1171) 



s_waitcnt vmcnt(7)
ds_write_b8 v49, v25 offset:5248 // (op 259) Store local data 
s_barrier  // (op 1153) 



s_waitcnt vmcnt(6)
ds_write_b128 v29, v[42:45] offset:5312 // (op 247) Store local data 
s_barrier  // (op 1165) 



s_waitcnt vmcnt(5)
ds_write_b8 v48, v12 offset:6336 // (op 258) Store local data 


s_lshl_b64 s[14:15], s[8:9], 6 // (op 260) 
s_mul_i32 s19, s15, 6 // (op 260) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 260) least significant
s_mul_hi_u32 s1, s14, 6 // (op 260) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 260) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 260) 

s_mul_i32 s1, 3, s14 // (op 260) 

v_add_u32 v52, v14, s1 // (op 260) 


s_waitcnt lgkmcnt(1)
s_barrier  // (op 1150) 
buffer_load_dwordx3 v[34:36], v52, s[44:47], 0 offen // (op 260) Load value



ds_read_b128 v[54:57], v6 offset:5312 // (op 249) Load local data 
buffer_load_dwordx3 v[38:40], v52, s[44:47], 0 offen offset:12 // (op 260) Load value




s_barrier  // (op 1156) 
s_barrier  // (op 1230) 



s_lshl_b32 s1, s24, 1 // (op 262) 
s_mul_i32 s14, 3, s1 // (op 262) 

v_add_u32 v14, v47, s14 // (op 262) 


buffer_load_ubyte v25, v14, s[52:55], 0 offen offset: 0 // (op 262) Load value

s_lshl_b64 s[14:15], s[8:9], 6 // (op 554) 

s_mul_i32 s19, s15, 6 // (op 554) most significant: low of high * low



s_mul_i32 s18, s14, 6 // (op 554) least significant
s_mul_hi_u32 s1, s14, 6 // (op 554) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 554) most significant: sum


s_lshl_b32 s1, s24, 2 // (op 682) 
v_add_u32 v21, v21, s1 // (op 682) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 554) 

s_lshl_b32 s1, s14, 1 // (op 554) 

v_add_u32 v17, v17, s1 // (op 554) 

ds_read_b96_tr_b6 v[58:60], v5 offset:3200 // (op 251) Transpose load local data 
ds_read_b96_tr_b6 v[61:63], v5 offset:3712 // (op 251) Transpose load local data 
buffer_load_dwordx4 v[42:45], v50, s[40:43], 0 offen offset:96 // (op 246) Load value




s_barrier  // (op 1236) 
v_add_u32 v13, 64, v13 // (op 490) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1162) 
buffer_load_ubyte v12, v51, s[48:51], 0 offen offset:6 // (op 261) Load value




ds_read_u8 v14, v8 offset:6336 // (op 253) Load local data 
s_barrier  // (op 1111) 



s_barrier  // (op 1168) 
s_barrier  // (op 1242) 


v_add_u32 v15, 4, v15 // (op 618) 
ds_read_u8 v47, v11 offset:5248 // (op 255) Load local data 
s_waitcnt vmcnt(9)
ds_write_b96 v20, v[22:24] // (op 13) Store local data 
s_waitcnt vmcnt(8)
ds_write_b96 v20, v[26:28] offset:16 // (op 13) Store local data 
s_barrier  // (op 1248) 
s_barrier  // (op 1139) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[54:57], v[58:63], a[0:15], v14, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 256) 







s_waitcnt vmcnt(7)
ds_write_b8 v41, v18 offset:2048 // (op 30) Store local data 
s_barrier  // (op 1098) 



s_waitcnt vmcnt(6)
ds_write_b128 v16, v[30:33] offset:2112 // (op 6) Store local data 
s_barrier  // (op 1125) 



s_waitcnt vmcnt(5)
ds_write_b8 v37, v19 offset:3136 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[12:13], 6 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60:

















v_lshrrev_b32 v37, 1, v1 // (op 453) 
s_ashr_i64 s[0:1], s[12:13], 6 // (op 453) 
v_lshl_add_u32 v41, s16, 5, v37 // (op 453) 

s_lshr_b32 s14, s0, 1 // (op 453) 

v_and_b32 v37, -2, v1 // (op 453) 
v_sub_u32 v13, v1, v37 // (op 453) 

v_lshlrev_b32 v37, 5, v13 // (op 453) 

v_lshl_add_u32 v13, s14, 7, v37 // (op 453) 


v_mov_b32 v15, 0 // (op 453) convert
v_mov_b32 v14, v13 // (op 453) convert

v_mul_lo_u32 v17, v41, s3 // (op 453) most significant: low of low * high
v_mul_hi_u32 v13, v41, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v16, v41, s2 // (op 453) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 453) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 453) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 453) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 453) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 453) 

v_mov_b32 v37, v16 // (op 453) convert

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

s_ashr_i64 s[0:1], s[12:13], 6 // (op 520) 
s_lshr_b32 s14, s0, 1 // (op 520) 

v_lshl_add_u32 v41, s14, 7, v1 // (op 520) 

s_lshl_b32 s0, s17, 5 // (op 520) 
s_mov_b32 s15, 0 // (op 520) convert
s_mov_b32 s14, s0 // (op 520) convert

v_mul_lo_u32 v27, v41, s9 // (op 520) most significant: low of low * high
v_mul_hi_u32 v13, v41, s8 // (op 520) most significant: high of low * low
v_mul_lo_u32 v26, v41, s8 // (op 520) least significant: low of low * low
v_add_u32 v27, v27, v13 // (op 520) most significant: sum

v_mov_b32 v13, s14 // (op 520) 
v_mov_b32 v14, s15 // (op 520) 
v_add_co_u32 v26, s[0:1], v26, v13 // (op 520) least significant half
v_addc_co_u32 v27, s[0:1], v27, v14, s[0:1] // (op 520) most significant half





v_mul_lo_u32 v29, v27, 6 // (op 520) most significant: low of high * low
v_mul_hi_u32 v41, v26, 6 // (op 520) most significant: high of low * low
v_mul_lo_u32 v28, v26, 6 // (op 520) least significant: low of low * low
v_add_u32 v29, v29, v41 // (op 520) most significant: sum


v_lshrrev_b64 v[26:27], 3, v[28:29] // (op 520) 

v_mov_b32 v41, v26 // (op 520) convert

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

v_lshrrev_b32 v13, 1, v1 // (op 584) 
s_ashr_i64 s[0:1], s[12:13], 6 // (op 584) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 584) 

s_lshr_b32 s14, s0, 1 // (op 584) 

v_and_b32 v13, -2, v1 // (op 584) 
v_sub_u32 v15, v1, v13 // (op 584) 

v_lshl_add_u32 v13, s14, 2, v15 // (op 584) 


v_mul_lo_u32 v15, v14, s20 // (op 584) 
v_add_u32 v15, v15, v13 // (op 584) 


s_mul_i32 s1, s20, s7 // (op 586) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 586) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 586) least significant
s_mul_hi_u32 s15, s20, s6 // (op 586) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 586) most significant: sum
s_add_u32 s1, s1, s15 // (op 586) most significant: sum


s_mov_b32 s48, s22 // (op 586) reg expression
s_mov_b32 s49, s23 // (op 586) reg expression
s_mov_b32 s50, s0 // (op 586) reg expression
s_mov_b32 s51, 131072 // (op 586) call()

v_lshrrev_b32 v13, 5, v1 // (op 648) 
s_ashr_i64 s[0:1], s[12:13], 6 // (op 648) 
v_and_b32 v14, -32, v1 // (op 648) 
s_lshr_b32 s14, s0, 1 // (op 648) 

v_sub_u32 v16, v1, v14 // (op 648) 

v_lshl_add_u32 v14, s14, 2, v13 // (op 648) 


v_lshl_add_u32 v13, s17, 5, v16 // (op 648) 

v_mul_lo_u32 v16, v14, s24 // (op 648) 
v_add_u32 v16, v16, v13 // (op 648) 


s_mul_i32 s1, s24, s29 // (op 650) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 650) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 650) least significant
s_mul_hi_u32 s15, s24, s28 // (op 650) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 650) most significant: sum
s_add_u32 s1, s1, s15 // (op 650) most significant: sum


s_mov_b32 s52, s26 // (op 650) reg expression
s_mov_b32 s53, s27 // (op 650) reg expression
s_mov_b32 s54, s0 // (op 650) reg expression
s_mov_b32 s55, 131072 // (op 650) call()

v_and_b32 v13, -2, v1 // (op 741) 
v_sub_u32 v14, v1, v13 // (op 741) 

v_lshlrev_b32 v13, 5, v14 // (op 741) 

v_lshrrev_b32 v14, 1, v1 // (op 741) 
v_lshl_add_u32 v17, v14, 6, v13 // (op 741) 


v_lshlrev_b32 v13, 2, v17 // (op 741) 

v_lshrrev_b32 v14, 3, v13 // (op 741) 

v_lshlrev_b32 v13, 5, v1 // (op 794) 
v_mul_lo_u32 v17, v13, 6 // (op 794) 
v_lshrrev_b32 v18, 3, v17 // (op 794) 

v_lshrrev_b32 v17, 4, v13 // (op 794) 

v_lshlrev_b32 v13, 2, v17 // (op 794) 

v_add_u32 v17, v18, v13 // (op 794) 


v_and_b32 v13, -2, v1 // (op 847) 
v_sub_u32 v18, v1, v13 // (op 847) 

v_and_b32 v13, 4294967294, v1 // (op 847) 
v_add_u32 v19, v13, v18 // (op 847) 


v_and_b32 v13, -32, v1 // (op 900) 
v_sub_u32 v18, v1, v13 // (op 900) 

v_and_b32 v13, 4294967264, v1 // (op 900) 
v_add_u32 v20, v13, v18 // (op 900) 


s_ashr_i64 s[0:1], s[12:13], 6 // (op 176) 
s_and_b32 s14, s0, 4294967294 // (op 176) 

s_ashr_i64 s[0:1], s[12:13], 6 // (op 175) 
s_mov_b32 s15, s0 // (op 175) convert

s_cmp_lt_u32 s14, s15 // (op 175) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175:


s_add_u32 s14, s14, 1 // (op 177) 
v_mov_b32 v18, v37 // (op 180) 

buffer_load_dwordx4 v[22:25], v18, s[40:43], 0 offen // (op 180) Load value

v_add_u32 v37, 32, v37 // (op 443) 


s_barrier  // (op 1101) 
v_mov_b32 v18, v41 // (op 194) 



v_mov_b32 v21, v18 // (op 194) reg expression

buffer_load_dwordx3 v[26:28], v21, s[44:47], 0 offen // (op 194) Load value
buffer_load_dwordx3 v[30:32], v21, s[44:47], 0 offen offset:12 // (op 194) Load value


s_lshl_b64 s[0:1], s[8:9], 6 // (op 512) 
s_mul_i32 s19, s1, 6 // (op 512) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 512) least significant
s_mul_hi_u32 s15, s0, 6 // (op 512) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 512) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 512) 

v_add_u32 v41, v41, s0 // (op 512) 

s_waitcnt vmcnt(2)
ds_write_b128 v14, v[22:25] offset:2112 // (op 181) Store local data 



v_mov_b32 v18, v15 // (op 195) 

s_barrier  // (op 1114) 
buffer_load_ubyte v29, v18, s[48:51], 0 offen offset: 0 // (op 195) Load value




v_add_u32 v15, 2, v15 // (op 576) 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[26:28] // (op 191) Store local data 


v_mov_b32 v18, v16 // (op 196) 
v_mov_b32 v12, v18 // (op 196) reg expression

buffer_load_ubyte v13, v12, s[52:55], 0 offen offset: 0 // (op 196) Load value


s_lshl_b32 s15, s24, 1 // (op 640) 
v_add_u32 v16, v16, s15 // (op 640) 

s_waitcnt vmcnt(2)
ds_write_b96 v17, v[30:32] offset:16 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1104) 



s_barrier  // (op 1128) 
ds_read_b128 v[30:33], v3 offset:2112 // (op 183) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1118) 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v29 offset:3136 // (op 192) Store local data 




ds_read_b96_tr_b6 v[24:26], v0 // (op 185) Transpose load local data 
s_barrier  // (op 1142) 
ds_read_b96_tr_b6 v[27:29], v0 offset:512 // (op 185) Transpose load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 1132) 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v13 offset:2048 // (op 193) Store local data 




ds_read_u8 v18, v4 offset:3136 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 


ds_read_u8 v12, v7 offset:2048 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[30:33], v[24:29], a[0:15], v18, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 190) 




s_ashr_i64 s[0:1], s[12:13], 6 // (op 175) 
s_mov_b32 s15, s0 // (op 175) convert

s_cmp_lt_u32 s14, s15 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 if true)
RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175:
























s_barrier  // (op 1224) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135
s_nop 4
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
v_mul_f32 v28, v10, v12 // (op 50) 
v_mul_f32 v29, v10, v13 // (op 50) 
v_mul_f32 v30, v10, v14 // (op 50) 
v_mul_f32 v31, v10, v15 // (op 50) 
v_mul_f32 v32, v10, v16 // (op 50) 
v_mul_f32 v33, v10, v17 // (op 50) 
v_mul_f32 v34, v10, v18 // (op 50) 
v_mul_f32 v35, v10, v19 // (op 50) 
v_mul_f32 v36, v10, v20 // (op 50) 
v_mul_f32 v37, v10, v21 // (op 50) 
v_mul_f32 v38, v10, v22 // (op 50) 
v_mul_f32 v39, v10, v23 // (op 50) 
v_mul_f32 v40, v10, v24 // (op 50) 
v_mul_f32 v41, v10, v25 // (op 50) 
v_mul_f32 v42, v10, v26 // (op 50) 
v_mul_f32 v43, v10, v27 // (op 50) 

v_lshrrev_b32 v12, 16, v28 // (op 56) 
v_lshrrev_b32 v13, 16, v29 // (op 56) 
v_lshrrev_b32 v14, 16, v30 // (op 56) 
v_lshrrev_b32 v15, 16, v31 // (op 56) 
v_lshrrev_b32 v16, 16, v32 // (op 56) 
v_lshrrev_b32 v17, 16, v33 // (op 56) 
v_lshrrev_b32 v18, 16, v34 // (op 56) 
v_lshrrev_b32 v19, 16, v35 // (op 56) 
v_lshrrev_b32 v20, 16, v36 // (op 56) 
v_lshrrev_b32 v21, 16, v37 // (op 56) 
v_lshrrev_b32 v22, 16, v38 // (op 56) 
v_lshrrev_b32 v23, 16, v39 // (op 56) 
v_lshrrev_b32 v24, 16, v40 // (op 56) 
v_lshrrev_b32 v25, 16, v41 // (op 56) 
v_lshrrev_b32 v26, 16, v42 // (op 56) 
v_lshrrev_b32 v27, 16, v43 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 942) 
v_and_b32 v11, 63, v1 // (op 942) 
v_lshrrev_b32 v3, 5, v11 // (op 942) 
v_lshlrev_b32 v4, 2, v3 // (op 942) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 942) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 942) 

v_lshrrev_b32 v3, 2, v11 // (op 942) 

v_and_b32 v11, 7, v3 // (op 942) 

v_and_b32 v3, 3, v1 // (op 942) 
v_lshl_add_u32 v4, v11, 2, v3 // (op 942) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 942) 

v_mul_lo_u32 v11, v3, s30 // (op 942) 

v_add_lshl_u32 v3, v0, v11, 1 // (op 942) 


s_mul_i32 s1, s30, s35 // (op 944) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 944) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 944) least significant
s_mul_hi_u32 s3, s30, s34 // (op 944) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 944) most significant: sum
s_add_u32 s1, s1, s3 // (op 944) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 944) 

s_mov_b32 s4, s32 // (op 944) reg expression
s_mov_b32 s5, s33 // (op 944) reg expression
s_mov_b32 s6, s2 // (op 944) reg expression
s_mov_b32 s7, 131072 // (op 944) call()


v_pack_b32_f16 v28, v12, v13 // (op 58) 
v_pack_b32_f16 v29, v14, v15 // (op 58) 
v_pack_b32_f16 v30, v16, v17 // (op 58) 
v_pack_b32_f16 v31, v18, v19 // (op 58) 
v_pack_b32_f16 v32, v20, v21 // (op 58) 
v_pack_b32_f16 v33, v22, v23 // (op 58) 
v_pack_b32_f16 v34, v24, v25 // (op 58) 
v_pack_b32_f16 v35, v26, v27 // (op 58) 


buffer_store_dwordx2 v[28:29], v3, s[4:7], 0 offen // (op 58) Store value
buffer_store_dwordx2 v[30:31], v3, s[4:7], 0 offen offset:16 // (op 58) Store value
buffer_store_dwordx2 v[32:33], v3, s[4:7], 0 offen offset:32 // (op 58) Store value
buffer_store_dwordx2 v[34:35], v3, s[4:7], 0 offen offset:48 // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135:
v_lshrrev_b32 v0, 6, v1 // (op 711) 
v_and_b32 v3, 63, v1 // (op 711) 
v_lshrrev_b32 v4, 5, v3 // (op 711) 
v_lshlrev_b32 v5, 2, v4 // (op 711) 

v_lshl_add_u32 v4, v0, 5, v5 // (op 711) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 711) 

v_lshrrev_b32 v4, 2, v3 // (op 711) 

v_and_b32 v3, 7, v4 // (op 711) 

v_and_b32 v4, 3, v1 // (op 711) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 711) 


v_lshl_add_u32 v3, s17, 5, v5 // (op 711) 

v_mul_lo_u32 v4, v3, s36 // (op 711) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 711) 


s_mul_i32 s1, s36, s35 // (op 713) most significant: low of low * high
s_mul_i32 s2, s37, s34 // (op 713) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 713) least significant
s_mul_hi_u32 s3, s36, s34 // (op 713) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 713) most significant: sum
s_add_u32 s1, s1, s3 // (op 713) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 713) 

s_mov_b32 s8, s38 // (op 713) reg expression
s_mov_b32 s9, s39 // (op 713) reg expression
s_mov_b32 s10, s2 // (op 713) reg expression
s_mov_b32 s11, 131072 // (op 713) call()





buffer_load_dwordx2 v[12:13], v3, s[8:11], 0 offen // (op 148) Load value
buffer_load_dwordx2 v[14:15], v3, s[8:11], 0 offen offset:16 // (op 148) Load value
buffer_load_dwordx2 v[16:17], v3, s[8:11], 0 offen offset:32 // (op 148) Load value
buffer_load_dwordx2 v[18:19], v3, s[8:11], 0 offen offset:48 // (op 148) Load value



v_accvgpr_read v20, a0 // (op 144) 
v_accvgpr_read v21, a1 // (op 144) 
v_accvgpr_read v22, a2 // (op 144) 
v_accvgpr_read v23, a3 // (op 144) 
v_accvgpr_read v24, a4 // (op 144) 
v_accvgpr_read v25, a5 // (op 144) 
v_accvgpr_read v26, a6 // (op 144) 
v_accvgpr_read v27, a7 // (op 144) 
v_accvgpr_read v28, a8 // (op 144) 
v_accvgpr_read v29, a9 // (op 144) 
v_accvgpr_read v30, a10 // (op 144) 
v_accvgpr_read v31, a11 // (op 144) 
v_accvgpr_read v32, a12 // (op 144) 
v_accvgpr_read v33, a13 // (op 144) 
v_accvgpr_read v34, a14 // (op 144) 
v_accvgpr_read v35, a15 // (op 144) 
v_mul_f32 v36, v10, v20 // (op 144) 
v_mul_f32 v37, v10, v21 // (op 144) 
v_mul_f32 v38, v10, v22 // (op 144) 
v_mul_f32 v39, v10, v23 // (op 144) 
v_mul_f32 v40, v10, v24 // (op 144) 
v_mul_f32 v41, v10, v25 // (op 144) 
v_mul_f32 v42, v10, v26 // (op 144) 
v_mul_f32 v43, v10, v27 // (op 144) 
v_mul_f32 v44, v10, v28 // (op 144) 
v_mul_f32 v45, v10, v29 // (op 144) 
v_mul_f32 v46, v10, v30 // (op 144) 
v_mul_f32 v47, v10, v31 // (op 144) 
v_mul_f32 v48, v10, v32 // (op 144) 
v_mul_f32 v49, v10, v33 // (op 144) 
v_mul_f32 v50, v10, v34 // (op 144) 
v_mul_f32 v51, v10, v35 // (op 144) 

s_waitcnt vmcnt(3)
v_and_b32 v20, 65535, v12 // (op 149) 
v_lshrrev_b32 v21, 16, v12 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v52, v9, v20 // (op 149) 
v_mul_f32 v53, v9, v21 // (op 149) 
v_and_b32 v20, 65535, v13 // (op 149) 
v_lshrrev_b32 v21, 16, v13 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v54, v9, v20 // (op 149) 
v_mul_f32 v55, v9, v21 // (op 149) 
s_waitcnt vmcnt(2)
v_and_b32 v20, 65535, v14 // (op 149) 
v_lshrrev_b32 v21, 16, v14 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v56, v9, v20 // (op 149) 
v_mul_f32 v57, v9, v21 // (op 149) 
v_and_b32 v20, 65535, v15 // (op 149) 
v_lshrrev_b32 v21, 16, v15 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v58, v9, v20 // (op 149) 
v_mul_f32 v59, v9, v21 // (op 149) 
s_waitcnt vmcnt(1)
v_and_b32 v20, 65535, v16 // (op 149) 
v_lshrrev_b32 v21, 16, v16 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v60, v9, v20 // (op 149) 
v_mul_f32 v61, v9, v21 // (op 149) 
v_and_b32 v20, 65535, v17 // (op 149) 
v_lshrrev_b32 v21, 16, v17 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v62, v9, v20 // (op 149) 
v_mul_f32 v63, v9, v21 // (op 149) 
s_waitcnt vmcnt(0)
v_and_b32 v20, 65535, v18 // (op 149) 
v_lshrrev_b32 v21, 16, v18 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v64, v9, v20 // (op 149) 
v_mul_f32 v65, v9, v21 // (op 149) 
v_and_b32 v20, 65535, v19 // (op 149) 
v_lshrrev_b32 v21, 16, v19 // (op 149) 
v_lshlrev_b32 v20, 16, v20 // (op 149) 
v_lshlrev_b32 v21, 16, v21 // (op 149) 
v_mul_f32 v66, v9, v20 // (op 149) 
v_mul_f32 v67, v9, v21 // (op 149) 


v_add_f32 v20, v52, v36 // (op 145) 
v_add_f32 v21, v53, v37 // (op 145) 
v_add_f32 v22, v54, v38 // (op 145) 
v_add_f32 v23, v55, v39 // (op 145) 
v_add_f32 v24, v56, v40 // (op 145) 
v_add_f32 v25, v57, v41 // (op 145) 
v_add_f32 v26, v58, v42 // (op 145) 
v_add_f32 v27, v59, v43 // (op 145) 
v_add_f32 v28, v60, v44 // (op 145) 
v_add_f32 v29, v61, v45 // (op 145) 
v_add_f32 v30, v62, v46 // (op 145) 
v_add_f32 v31, v63, v47 // (op 145) 
v_add_f32 v32, v64, v48 // (op 145) 
v_add_f32 v33, v65, v49 // (op 145) 
v_add_f32 v34, v66, v50 // (op 145) 
v_add_f32 v35, v67, v51 // (op 145) 


v_lshrrev_b32 v36, 16, v20 // (op 146) 
v_lshrrev_b32 v37, 16, v21 // (op 146) 
v_lshrrev_b32 v38, 16, v22 // (op 146) 
v_lshrrev_b32 v39, 16, v23 // (op 146) 
v_lshrrev_b32 v40, 16, v24 // (op 146) 
v_lshrrev_b32 v41, 16, v25 // (op 146) 
v_lshrrev_b32 v42, 16, v26 // (op 146) 
v_lshrrev_b32 v43, 16, v27 // (op 146) 
v_lshrrev_b32 v44, 16, v28 // (op 146) 
v_lshrrev_b32 v45, 16, v29 // (op 146) 
v_lshrrev_b32 v46, 16, v30 // (op 146) 
v_lshrrev_b32 v47, 16, v31 // (op 146) 
v_lshrrev_b32 v48, 16, v32 // (op 146) 
v_lshrrev_b32 v49, 16, v33 // (op 146) 
v_lshrrev_b32 v50, 16, v34 // (op 146) 
v_lshrrev_b32 v51, 16, v35 // (op 146) 

v_lshrrev_b32 v0, 6, v1 // (op 959) 
v_and_b32 v3, 63, v1 // (op 959) 
v_lshrrev_b32 v4, 5, v3 // (op 959) 
v_lshlrev_b32 v5, 2, v4 // (op 959) 

v_lshl_add_u32 v4, v0, 5, v5 // (op 959) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 959) 

v_lshrrev_b32 v4, 2, v3 // (op 959) 

v_and_b32 v3, 7, v4 // (op 959) 

v_and_b32 v4, 3, v1 // (op 959) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 959) 


v_lshl_add_u32 v3, s17, 5, v5 // (op 959) 

v_mul_lo_u32 v4, v3, s30 // (op 959) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 959) 


s_mul_i32 s1, s30, s35 // (op 961) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 961) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 961) least significant
s_mul_hi_u32 s3, s30, s34 // (op 961) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 961) most significant: sum
s_add_u32 s1, s1, s3 // (op 961) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 961) 

s_mov_b32 s4, s32 // (op 961) reg expression
s_mov_b32 s5, s33 // (op 961) reg expression
s_mov_b32 s6, s2 // (op 961) reg expression
s_mov_b32 s7, 131072 // (op 961) call()


v_pack_b32_f16 v28, v36, v37 // (op 147) 
v_pack_b32_f16 v29, v38, v39 // (op 147) 
v_pack_b32_f16 v30, v40, v41 // (op 147) 
v_pack_b32_f16 v31, v42, v43 // (op 147) 
v_pack_b32_f16 v32, v44, v45 // (op 147) 
v_pack_b32_f16 v33, v46, v47 // (op 147) 
v_pack_b32_f16 v34, v48, v49 // (op 147) 
v_pack_b32_f16 v35, v50, v51 // (op 147) 


buffer_store_dwordx2 v[28:29], v3, s[4:7], 0 offen // (op 147) Store value
buffer_store_dwordx2 v[30:31], v3, s[4:7], 0 offen offset:16 // (op 147) Store value
buffer_store_dwordx2 v[32:33], v3, s[4:7], 0 offen offset:32 // (op 147) Store value
buffer_store_dwordx2 v[34:35], v3, s[4:7], 0 offen offset:48 // (op 147) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135:



s_endpgm  // End of RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 6400 // lds bytes
  .amdhsa_accum_offset 68
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
    DataTypeB: F6
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
    macrotile: [32, 32, 64]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TT_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 6400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 68
    .agpr_count: 16
    .max_flat_workgroup_size: 64
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

