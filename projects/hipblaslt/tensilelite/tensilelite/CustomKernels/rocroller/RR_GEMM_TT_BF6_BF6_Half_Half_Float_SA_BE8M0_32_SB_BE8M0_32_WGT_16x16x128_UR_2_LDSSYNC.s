// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC
.p2align 8
.type RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC,@function
RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 769) 
v_and_b32 v3, 15, v1 // (op 769) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 769) 


v_and_b32 v0, 63, v1 // (op 769) 
v_lshrrev_b32 v3, 4, v0 // (op 769) 

v_lshlrev_b32 v0, 5, v3 // (op 769) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 769) 


v_mul_lo_u32 v0, v3, 6 // (op 769) 

v_lshrrev_b32 v3, 3, v0 // (op 769) 

v_and_b32 v0, 63, v1 // (op 822) 
v_and_b32 v4, 48, v1 // (op 822) 
v_sub_u32 v5, v0, v4 // (op 822) 

v_lshrrev_b32 v4, 4, v0 // (op 822) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 822) 


v_lshlrev_b32 v4, 4, v0 // (op 822) 
v_mul_lo_u32 v5, v4, 6 // (op 822) 

v_lshrrev_b32 v4, 3, v5 // (op 822) 

v_and_b32 v5, 268435455, v0 // (op 822) 

v_lshlrev_b32 v0, 2, v5 // (op 822) 

v_add_u32 v5, v4, v0 // (op 822) 


v_lshrrev_b32 v0, 6, v1 // (op 875) 
v_and_b32 v4, 15, v1 // (op 875) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 875) 


v_and_b32 v0, 63, v1 // (op 875) 
v_lshrrev_b32 v4, 4, v0 // (op 875) 

v_lshl_add_u32 v0, v6, 2, v4 // (op 875) 


v_and_b32 v4, 48, v1 // (op 928) 
v_and_b32 v6, 15, v1 // (op 928) 
v_add_u32 v7, v4, v6 // (op 928) 


v_lshrrev_b32 v4, 6, v1 // (op 986) 
v_and_b32 v6, 15, v1 // (op 986) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 986) 


v_and_b32 v4, 63, v1 // (op 986) 
v_lshrrev_b32 v6, 4, v4 // (op 986) 

v_lshlrev_b32 v4, 5, v6 // (op 986) 

v_lshl_add_u32 v6, v8, 7, v4 // (op 986) 


v_mul_lo_u32 v4, v6, 6 // (op 986) 

v_lshrrev_b32 v6, 3, v4 // (op 986) 

v_and_b32 v4, 63, v1 // (op 1010) 
v_and_b32 v8, 48, v1 // (op 1010) 
v_sub_u32 v9, v4, v8 // (op 1010) 

v_lshrrev_b32 v8, 4, v4 // (op 1010) 

v_lshl_add_u32 v4, v8, 5, v9 // (op 1010) 


v_lshlrev_b32 v8, 4, v4 // (op 1010) 
v_mul_lo_u32 v9, v8, 6 // (op 1010) 

v_lshrrev_b32 v8, 3, v9 // (op 1010) 

v_and_b32 v9, 268435455, v4 // (op 1010) 

v_lshlrev_b32 v4, 2, v9 // (op 1010) 

v_add_u32 v9, v8, v4 // (op 1010) 


v_lshrrev_b32 v4, 6, v1 // (op 1034) 
v_and_b32 v8, 15, v1 // (op 1034) 
v_lshl_add_u32 v10, v4, 4, v8 // (op 1034) 


v_and_b32 v4, 63, v1 // (op 1034) 
v_lshrrev_b32 v8, 4, v4 // (op 1034) 

v_lshl_add_u32 v4, v10, 2, v8 // (op 1034) 


v_and_b32 v8, 48, v1 // (op 1058) 
v_and_b32 v10, 15, v1 // (op 1058) 
v_add_u32 v11, v8, v10 // (op 1058) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_lshrrev_b32 v12, 2, v1 // (op 479) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 479) 

v_and_b32 v12, -4, v1 // (op 479) 
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

v_lshrrev_b32 v13, 2, v1 // (op 498) 
v_lshl_add_u32 v14, s16, 4, v13 // (op 498) 

v_and_b32 v13, -4, v1 // (op 498) 
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

s_lshl_b32 s0, s17, 4 // (op 543) 
v_lshlrev_b32 v14, 1, v1 // (op 543) 
s_mov_b32 s15, 0 // (op 543) convert
s_mov_b32 s14, s0 // (op 543) convert

v_mul_lo_u32 v17, v14, s9 // (op 543) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 543) most significant: high of low * low
v_mul_lo_u32 v16, v14, s8 // (op 543) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 543) most significant: sum

v_mov_b32 v15, s14 // (op 543) 
v_mov_b32 v18, s15 // (op 543) 
v_add_co_u32 v16, s[0:1], v16, v15 // (op 543) least significant half
v_addc_co_u32 v17, s[0:1], v17, v18, s[0:1] // (op 543) most significant half





v_mul_lo_u32 v15, v17, 6 // (op 543) most significant: low of high * low
v_mul_hi_u32 v18, v16, 6 // (op 543) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 543) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 543) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 543) 

v_mov_b32 v14, v16 // (op 543) convert

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

s_lshl_b32 s0, s17, 4 // (op 562) 
v_lshlrev_b32 v15, 1, v1 // (op 562) 
s_mov_b32 s15, 0 // (op 562) convert
s_mov_b32 s14, s0 // (op 562) convert

v_mul_lo_u32 v17, v15, s9 // (op 562) most significant: low of low * high
v_mul_hi_u32 v18, v15, s8 // (op 562) most significant: high of low * low
v_mul_lo_u32 v16, v15, s8 // (op 562) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 562) most significant: sum

v_mov_b32 v18, s14 // (op 562) 
v_mov_b32 v19, s15 // (op 562) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 562) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 562) most significant half





v_mul_lo_u32 v19, v17, 6 // (op 562) most significant: low of high * low
v_mul_hi_u32 v15, v16, 6 // (op 562) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 562) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 562) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 562) 

v_mov_b32 v15, v16 // (op 562) convert

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

v_lshrrev_b32 v16, 2, v1 // (op 607) 
v_lshl_add_u32 v17, s16, 4, v16 // (op 607) 

v_and_b32 v16, -4, v1 // (op 607) 
v_sub_u32 v18, v1, v16 // (op 607) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v16, v17, s20 // (op 607) 
v_add_u32 v16, v16, v18 // (op 607) 


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

v_lshrrev_b32 v17, 2, v1 // (op 626) 
v_lshl_add_u32 v18, s16, 4, v17 // (op 626) 

v_and_b32 v17, -4, v1 // (op 626) 
v_sub_u32 v19, v1, v17 // (op 626) 

v_mul_lo_u32 v17, v18, s20 // (op 626) 
v_add_u32 v17, v17, v19 // (op 626) 


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

v_and_b32 v18, -16, v1 // (op 671) 
v_sub_u32 v19, v1, v18 // (op 671) 

v_lshl_add_u32 v18, s17, 4, v19 // (op 671) 

v_lshrrev_b32 v19, 4, v1 // (op 671) 
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

v_and_b32 v18, -16, v1 // (op 690) 
v_sub_u32 v19, v1, v18 // (op 690) 

v_lshl_add_u32 v18, s17, 4, v19 // (op 690) 

v_lshrrev_b32 v19, 4, v1 // (op 690) 
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


s_mul_i32 s1, s9, 6 // (op 370) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 370) least significant
s_mul_hi_u32 s14, s8, 6 // (op 370) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 370) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 370) 


v_mov_b32 v22, v14 // (op 370) reg expression
buffer_load_dwordx3 v[24:26], v22, s[44:47], 0 offen // (op 370) Load value
v_add_u32 v22, v22, v18 // (op 370) 
buffer_load_dwordx3 v[28:30], v22, s[44:47], 0 offen // (op 370) Load value




v_mov_b32 v27, v20 // (op 373) reg expression

buffer_load_ubyte v18, v27, s[52:55], 0 offen offset: 0 // (op 373) Load value




buffer_load_dwordx4 v[32:35], v12, s[40:43], 0 offen // (op 376) Load value
buffer_load_dwordx2 v[36:37], v12, s[40:43], 0 offen offset:16 // (op 376) Load value



buffer_load_ubyte v19, v16, s[48:51], 0 offen offset: 0 // (op 379) Load value

s_mul_i32 s1, s9, 6 // (op 382) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 382) least significant
s_mul_hi_u32 s14, s8, 6 // (op 382) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 382) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 382) 


s_lshl_b64 s[0:1], s[8:9], 7 // (op 382) 
s_mul_i32 s15, s1, 6 // (op 382) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 382) least significant
s_mul_hi_u32 s18, s0, 6 // (op 382) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 382) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 382) 

v_add_u32 v27, v14, s0 // (op 382) 

buffer_load_dwordx3 v[38:40], v27, s[44:47], 0 offen // (op 382) Load value
v_add_u32 v27, v27, v22 // (op 382) 
buffer_load_dwordx3 v[42:44], v27, s[44:47], 0 offen // (op 382) Load value





s_lshl_b32 s0, s24, 2 // (op 385) 
v_add_u32 v14, v20, s0 // (op 385) 


buffer_load_ubyte v27, v14, s[52:55], 0 offen offset: 0 // (op 385) Load value





buffer_load_dwordx4 v[46:49], v12, s[40:43], 0 offen offset:96 // (op 388) Load value
buffer_load_dwordx2 v[50:51], v12, s[40:43], 0 offen offset:112 // (op 388) Load value




buffer_load_ubyte v12, v16, s[48:51], 0 offen offset:4 // (op 391) Load value

v_lshlrev_b32 v14, 5, v1 // (op 810) 
v_mul_lo_u32 v16, v14, 6 // (op 810) 
v_lshrrev_b32 v20, 3, v16 // (op 810) 

v_lshrrev_b32 v16, 4, v14 // (op 810) 

v_lshlrev_b32 v14, 2, v16 // (op 810) 

v_add_u32 v16, v20, v14 // (op 810) 





s_waitcnt vmcnt(11)
ds_write_b96 v16, v[24:26] // (op 393) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v16, v[28:30] offset:16 // (op 393) Store local data 

v_and_b32 v14, -16, v1 // (op 916) 
v_sub_u32 v16, v1, v14 // (op 916) 

v_and_b32 v14, 4294967280, v1 // (op 916) 
v_add_u32 v20, v14, v16 // (op 916) 





s_waitcnt vmcnt(9)
ds_write_b8 v20, v18 offset:2048 // (op 394) Store local data 

v_and_b32 v14, -4, v1 // (op 757) 
v_sub_u32 v16, v1, v14 // (op 757) 

v_lshlrev_b32 v14, 5, v16 // (op 757) 

v_lshrrev_b32 v16, 2, v1 // (op 757) 
v_lshl_add_u32 v20, v16, 7, v14 // (op 757) 


v_mul_lo_u32 v14, v20, 6 // (op 757) 

v_lshrrev_b32 v16, 3, v14 // (op 757) 




s_waitcnt vmcnt(8)
ds_write_b128 v16, v[32:35] offset:2112 // (op 395) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v16, v[36:37] offset:2128 // (op 395) Store local data 

v_and_b32 v14, -4, v1 // (op 863) 
v_sub_u32 v16, v1, v14 // (op 863) 

v_and_b32 v14, 4294967292, v1 // (op 863) 
v_add_u32 v20, v14, v16 // (op 863) 





s_waitcnt vmcnt(6)
ds_write_b8 v20, v19 offset:3648 // (op 396) Store local data 

v_and_b32 v14, -4, v1 // (op 728) 
v_sub_u32 v16, v1, v14 // (op 728) 

v_lshlrev_b32 v14, 5, v16 // (op 728) 

v_lshrrev_b32 v16, 2, v1 // (op 728) 
v_lshl_add_u32 v20, v16, 7, v14 // (op 728) 


v_mul_lo_u32 v14, v20, 6 // (op 728) 

v_lshrrev_b32 v16, 3, v14 // (op 728) 

v_lshlrev_b32 v14, 5, v1 // (op 781) 
v_mul_lo_u32 v20, v14, 6 // (op 781) 
v_lshrrev_b32 v31, 3, v20 // (op 781) 

v_lshrrev_b32 v20, 4, v14 // (op 781) 

v_lshlrev_b32 v14, 2, v20 // (op 781) 

v_add_u32 v20, v31, v14 // (op 781) 


v_and_b32 v14, -4, v1 // (op 834) 
v_sub_u32 v31, v1, v14 // (op 834) 

v_and_b32 v14, 4294967292, v1 // (op 834) 
v_add_u32 v41, v14, v31 // (op 834) 


v_and_b32 v14, -16, v1 // (op 887) 
v_sub_u32 v31, v1, v14 // (op 887) 

v_and_b32 v14, 4294967280, v1 // (op 887) 
v_add_u32 v45, v14, v31 // (op 887) 


v_and_b32 v14, -4, v1 // (op 974) 
v_sub_u32 v31, v1, v14 // (op 974) 

v_lshlrev_b32 v14, 5, v31 // (op 974) 

v_lshrrev_b32 v31, 2, v1 // (op 974) 
v_lshl_add_u32 v22, v31, 7, v14 // (op 974) 


v_mul_lo_u32 v14, v22, 6 // (op 974) 

v_lshrrev_b32 v31, 3, v14 // (op 974) 

v_lshlrev_b32 v14, 5, v1 // (op 998) 
v_mul_lo_u32 v22, v14, 6 // (op 998) 
v_lshrrev_b32 v23, 3, v22 // (op 998) 

v_lshrrev_b32 v22, 4, v14 // (op 998) 

v_lshlrev_b32 v14, 2, v22 // (op 998) 

v_add_u32 v22, v23, v14 // (op 998) 


v_and_b32 v14, -4, v1 // (op 1022) 
v_sub_u32 v23, v1, v14 // (op 1022) 

v_and_b32 v14, 4294967292, v1 // (op 1022) 
v_add_u32 v52, v14, v23 // (op 1022) 


v_and_b32 v14, -16, v1 // (op 1046) 
v_sub_u32 v23, v1, v14 // (op 1046) 

v_and_b32 v14, 4294967280, v1 // (op 1046) 
v_add_u32 v53, v14, v23 // (op 1046) 


s_mov_b32 s0, 0 // (op 61) call()
s_ashr_i64 s[14:15], s[12:13], 7 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_60 if false)
RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopTop_KLoop_60:

s_barrier  // (op 1095) 
s_add_u32 s0, s0, 2 // (op 62) 



s_mul_i32 s15, s9, 6 // (op 10) most significant: low of high * low
s_mul_i32 s14, s8, 6 // (op 10) least significant
s_mul_hi_u32 s1, s8, 6 // (op 10) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 10) most significant: sum

v_lshrrev_b64 v[54:55], 3, s[14:15] // (op 10) 


v_mov_b32 v14, v15 // (op 10) 
s_lshl_b64 s[14:15], s[8:9], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s1, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s1, s14, 1 // (op 10) 

v_add_u32 v23, v14, s1 // (op 10) 

ds_read_b128 v[56:59], v3 offset:2112 // (op 72) Load local data 
ds_read_b64 v[60:61], v3 offset:2128 // (op 72) Load local data 
buffer_load_dwordx3 v[24:26], v23, s[44:47], 0 offen // (op 10) Load value
s_barrier  // (op 1108) 
v_add_u32 v23, v23, v54 // (op 10) 



buffer_load_dwordx3 v[28:30], v23, s[44:47], 0 offen // (op 10) Load value


ds_read_b96_tr_b6 v[62:64], v5 // (op 77) Transpose load local data 


ds_read_b96_tr_b6 v[65:67], v5 offset:256 // (op 77) Transpose load local data 
v_mov_b32 v23, v21 // (op 25) 
s_barrier  // (op 1122) 
s_lshl_b32 s1, s24, 3 // (op 25) 



ds_read_u8 v54, v0 offset:3648 // (op 82) Load local data 
v_add_u32 v55, v23, s1 // (op 25) 


s_barrier  // (op 1136) 
buffer_load_ubyte v18, v55, s[52:55], 0 offen offset: 0 // (op 25) Load value





v_mov_b32 v55, v13 // (op 4) 

ds_read_u8 v68, v7 offset:2048 // (op 87) Load local data 
buffer_load_dwordx4 v[32:35], v55, s[40:43], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[36:37], v55, s[40:43], 0 offen offset:208 // (op 4) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:61], v[62:67], a[0:3], v54, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 90) 






v_mov_b32 v54, v17 // (op 17) 

buffer_load_ubyte v19, v54, s[48:51], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1159) 



s_waitcnt vmcnt(11)
ds_write_b96 v22, v[38:40] offset:3712 // (op 257) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v22, v[42:44] offset:3728 // (op 257) Store local data 
s_barrier  // (op 1171) 



s_waitcnt vmcnt(9)
ds_write_b8 v53, v27 offset:5760 // (op 259) Store local data 
s_barrier  // (op 1153) 



s_waitcnt vmcnt(8)
ds_write_b128 v31, v[46:49] offset:5824 // (op 247) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v31, v[50:51] offset:5840 // (op 247) Store local data 
s_barrier  // (op 1165) 



s_waitcnt vmcnt(6)
ds_write_b8 v52, v12 offset:7360 // (op 258) Store local data 

s_mul_i32 s15, s9, 6 // (op 260) most significant: low of high * low
s_mul_i32 s14, s8, 6 // (op 260) least significant
s_mul_hi_u32 s1, s8, 6 // (op 260) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 260) most significant: sum

v_lshrrev_b64 v[56:57], 3, s[14:15] // (op 260) 


s_lshl_b64 s[14:15], s[8:9], 7 // (op 260) 
s_mul_i32 s19, s15, 6 // (op 260) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 260) least significant
s_mul_hi_u32 s1, s14, 6 // (op 260) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 260) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 260) 

s_mul_i32 s1, 3, s14 // (op 260) 

v_add_u32 v58, v14, s1 // (op 260) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1150) 
buffer_load_dwordx3 v[38:40], v58, s[44:47], 0 offen // (op 260) Load value



v_add_u32 v58, v58, v56 // (op 260) 
ds_read_b128 v[60:63], v6 offset:5824 // (op 249) Load local data 
ds_read_b64 v[64:65], v6 offset:5840 // (op 249) Load local data 
buffer_load_dwordx3 v[42:44], v58, s[44:47], 0 offen // (op 260) Load value





s_barrier  // (op 1156) 
s_barrier  // (op 1230) 



s_lshl_b32 s1, s24, 2 // (op 262) 
s_mul_i32 s14, 3, s1 // (op 262) 

v_add_u32 v14, v23, s14 // (op 262) 


s_lshl_b64 s[14:15], s[8:9], 7 // (op 554) 
s_mul_i32 s19, s15, 6 // (op 554) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 554) least significant
buffer_load_ubyte v27, v14, s[52:55], 0 offen offset: 0 // (op 262) Load value

s_mul_hi_u32 s1, s14, 6 // (op 554) most significant: high of low * low

s_add_u32 s19, s19, s1 // (op 554) most significant: sum





s_lshl_b32 s1, s24, 3 // (op 682) 
v_add_u32 v21, v21, s1 // (op 682) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 554) 

s_lshl_b32 s1, s14, 1 // (op 554) 

v_add_u32 v15, v15, s1 // (op 554) 

ds_read_b96_tr_b6 v[66:68], v9 offset:3712 // (op 251) Transpose load local data 
ds_read_b96_tr_b6 v[69:71], v9 offset:3968 // (op 251) Transpose load local data 
buffer_load_dwordx4 v[46:49], v55, s[40:43], 0 offen offset:288 // (op 246) Load value
buffer_load_dwordx2 v[50:51], v55, s[40:43], 0 offen offset:304 // (op 246) Load value

s_barrier  // (op 1236) 



v_add_u32 v13, 192, v13 // (op 490) 
buffer_load_ubyte v12, v54, s[48:51], 0 offen offset:12 // (op 261) Load value
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1162) 




s_barrier  // (op 1111) 
v_add_u32 v17, 8, v17 // (op 618) 



ds_read_u8 v14, v4 offset:7360 // (op 253) Load local data 
s_waitcnt vmcnt(11)
ds_write_b96 v20, v[24:26] // (op 13) Store local data 
s_waitcnt vmcnt(10)
ds_write_b96 v20, v[28:30] offset:16 // (op 13) Store local data 
s_barrier  // (op 1168) 


s_barrier  // (op 1242) 
s_barrier  // (op 1139) 
ds_read_u8 v23, v11 offset:5760 // (op 255) Load local data 



s_waitcnt vmcnt(9)
ds_write_b8 v45, v18 offset:2048 // (op 30) Store local data 
s_barrier  // (op 1248) 
s_barrier  // (op 1098) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:65], v[66:71], a[0:3], v14, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 256) 







s_waitcnt vmcnt(8)
ds_write_b128 v16, v[32:35] offset:2112 // (op 6) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v16, v[36:37] offset:2128 // (op 6) Store local data 
s_barrier  // (op 1125) 



s_waitcnt vmcnt(6)
ds_write_b8 v41, v19 offset:3648 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[12:13], 7 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopTop_KLoop_60 if true)
RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_60:

















v_lshrrev_b32 v41, 2, v1 // (op 453) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 453) 
v_lshl_add_u32 v45, s16, 4, v41 // (op 453) 

s_lshr_b32 s14, s0, 1 // (op 453) 

v_and_b32 v41, -4, v1 // (op 453) 
v_sub_u32 v13, v1, v41 // (op 453) 

v_lshlrev_b32 v41, 5, v13 // (op 453) 

v_lshl_add_u32 v13, s14, 8, v41 // (op 453) 


v_mov_b32 v15, 0 // (op 453) convert
v_mov_b32 v14, v13 // (op 453) convert

v_mul_lo_u32 v17, v45, s3 // (op 453) most significant: low of low * high
v_mul_hi_u32 v13, v45, s2 // (op 453) most significant: high of low * low
v_mul_lo_u32 v16, v45, s2 // (op 453) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 453) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 453) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 453) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 453) most significant: low of high * low
v_mul_hi_u32 v13, v16, 6 // (op 453) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 453) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 453) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 453) 

v_mov_b32 v41, v16 // (op 453) convert

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

v_lshlrev_b32 v45, 1, v1 // (op 520) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 520) 
s_lshr_b32 s14, s0, 1 // (op 520) 

v_lshl_add_u32 v13, s14, 8, v45 // (op 520) 


s_lshl_b32 s0, s17, 4 // (op 520) 
s_mov_b32 s15, 0 // (op 520) convert
s_mov_b32 s14, s0 // (op 520) convert

v_mul_lo_u32 v15, v13, s9 // (op 520) most significant: low of low * high
v_mul_hi_u32 v45, v13, s8 // (op 520) most significant: high of low * low
v_mul_lo_u32 v14, v13, s8 // (op 520) least significant: low of low * low
v_add_u32 v15, v15, v45 // (op 520) most significant: sum

v_mov_b32 v45, s14 // (op 520) 
v_mov_b32 v16, s15 // (op 520) 
v_add_co_u32 v14, s[0:1], v14, v45 // (op 520) least significant half
v_addc_co_u32 v15, s[0:1], v15, v16, s[0:1] // (op 520) most significant half





v_mul_lo_u32 v17, v15, 6 // (op 520) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 520) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 520) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 520) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 520) 

v_mov_b32 v13, v14 // (op 520) convert

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

v_lshrrev_b32 v45, 2, v1 // (op 584) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 584) 
v_lshl_add_u32 v14, s16, 4, v45 // (op 584) 

s_lshr_b32 s14, s0, 1 // (op 584) 

v_and_b32 v45, -4, v1 // (op 584) 
v_sub_u32 v15, v1, v45 // (op 584) 

v_lshl_add_u32 v45, s14, 3, v15 // (op 584) 


v_mul_lo_u32 v15, v14, s20 // (op 584) 
v_add_u32 v15, v15, v45 // (op 584) 


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

v_lshrrev_b32 v14, 4, v1 // (op 648) 
s_ashr_i64 s[0:1], s[12:13], 7 // (op 648) 
v_and_b32 v45, -16, v1 // (op 648) 
s_lshr_b32 s14, s0, 1 // (op 648) 

v_sub_u32 v16, v1, v45 // (op 648) 

v_lshl_add_u32 v45, s14, 3, v14 // (op 648) 


v_lshl_add_u32 v14, s17, 4, v16 // (op 648) 

v_mul_lo_u32 v16, v45, s24 // (op 648) 
v_add_u32 v16, v16, v14 // (op 648) 


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

v_and_b32 v14, -4, v1 // (op 741) 
v_sub_u32 v45, v1, v14 // (op 741) 

v_lshlrev_b32 v14, 5, v45 // (op 741) 

v_lshrrev_b32 v45, 2, v1 // (op 741) 
v_lshl_add_u32 v17, v45, 7, v14 // (op 741) 


v_mul_lo_u32 v14, v17, 6 // (op 741) 

v_lshrrev_b32 v45, 3, v14 // (op 741) 

v_lshlrev_b32 v14, 5, v1 // (op 794) 
v_mul_lo_u32 v17, v14, 6 // (op 794) 
v_lshrrev_b32 v18, 3, v17 // (op 794) 

v_lshrrev_b32 v17, 4, v14 // (op 794) 

v_lshlrev_b32 v14, 2, v17 // (op 794) 

v_add_u32 v17, v18, v14 // (op 794) 


v_and_b32 v14, -4, v1 // (op 847) 
v_sub_u32 v18, v1, v14 // (op 847) 

v_and_b32 v14, 4294967292, v1 // (op 847) 
v_add_u32 v19, v14, v18 // (op 847) 


v_and_b32 v14, -16, v1 // (op 900) 
v_sub_u32 v18, v1, v14 // (op 900) 

v_and_b32 v14, 4294967280, v1 // (op 900) 
v_add_u32 v20, v14, v18 // (op 900) 


s_ashr_i64 s[0:1], s[12:13], 7 // (op 176) 
s_and_b32 s14, s0, 4294967294 // (op 176) 

s_ashr_i64 s[0:1], s[12:13], 7 // (op 175) 
s_mov_b32 s15, s0 // (op 175) convert

s_cmp_lt_u32 s14, s15 // (op 175) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_175:


s_add_u32 s14, s14, 1 // (op 177) 
v_mov_b32 v12, v41 // (op 180) 

buffer_load_dwordx4 v[22:25], v12, s[40:43], 0 offen // (op 180) Load value
buffer_load_dwordx2 v[26:27], v12, s[40:43], 0 offen offset:16 // (op 180) Load value

v_add_u32 v41, 96, v41 // (op 443) 

s_barrier  // (op 1101) 
s_mul_i32 s1, s9, 6 // (op 194) most significant: low of high * low



s_mul_i32 s0, s8, 6 // (op 194) least significant
s_mul_hi_u32 s15, s8, 6 // (op 194) most significant: high of low * low
s_add_u32 s1, s1, s15 // (op 194) most significant: sum

v_lshrrev_b64 v[28:29], 3, s[0:1] // (op 194) 


v_mov_b32 v12, v13 // (op 194) 
v_mov_b32 v14, v12 // (op 194) reg expression
buffer_load_dwordx3 v[30:32], v14, s[44:47], 0 offen // (op 194) Load value
v_add_u32 v14, v14, v28 // (op 194) 
s_waitcnt vmcnt(2)
ds_write_b128 v45, v[22:25] offset:2112 // (op 181) Store local data 
buffer_load_dwordx3 v[34:36], v14, s[44:47], 0 offen // (op 194) Load value



s_lshl_b64 s[0:1], s[8:9], 7 // (op 512) 
s_mul_i32 s19, s1, 6 // (op 512) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 512) least significant
s_mul_hi_u32 s15, s0, 6 // (op 512) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 512) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 512) 

v_add_u32 v13, v13, s0 // (op 512) 

s_waitcnt vmcnt(2)
ds_write_b64 v45, v[26:27] offset:2128 // (op 181) Store local data 



v_mov_b32 v12, v15 // (op 195) 

s_barrier  // (op 1114) 
buffer_load_ubyte v14, v12, s[48:51], 0 offen offset: 0 // (op 195) Load value




v_add_u32 v15, 4, v15 // (op 576) 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[30:32] // (op 191) Store local data 


v_mov_b32 v12, v16 // (op 196) 
v_mov_b32 v18, v12 // (op 196) reg expression

buffer_load_ubyte v33, v18, s[52:55], 0 offen offset: 0 // (op 196) Load value


s_lshl_b32 s15, s24, 2 // (op 640) 
v_add_u32 v16, v16, s15 // (op 640) 

s_waitcnt vmcnt(2)
ds_write_b96 v17, v[34:36] offset:16 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1104) 



s_barrier  // (op 1128) 
ds_read_b128 v[34:37], v3 offset:2112 // (op 183) Load local data 



ds_read_b64 v[38:39], v3 offset:2128 // (op 183) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v14 offset:3648 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1118) 



s_barrier  // (op 1142) 
ds_read_b96_tr_b6 v[22:24], v5 // (op 185) Transpose load local data 



ds_read_b96_tr_b6 v[25:27], v5 offset:256 // (op 185) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v33 offset:2048 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1132) 



ds_read_u8 v12, v0 offset:3648 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 


ds_read_u8 v14, v7 offset:2048 // (op 189) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:39], v[22:27], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 190) 




s_ashr_i64 s[0:1], s[12:13], 7 // (op 175) 
s_mov_b32 s15, s0 // (op 175) convert

s_cmp_lt_u32 s14, s15 // (op 175) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_175 if true)
RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_175:
























s_barrier  // (op 1224) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

v_cvt_f16_f32 v4, v12 // (op 56) 
v_cvt_f16_f32 v5, v13 // (op 56) 
v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 942) 
v_and_b32 v3, 63, v1 // (op 942) 
v_lshrrev_b32 v9, 4, v3 // (op 942) 
v_lshlrev_b32 v11, 2, v9 // (op 942) 

v_lshl_add_u32 v9, v0, 4, v11 // (op 942) 


v_lshl_add_u32 v0, s16, 4, v9 // (op 942) 

v_lshrrev_b32 v9, 2, v3 // (op 942) 

v_and_b32 v3, 3, v9 // (op 942) 

v_and_b32 v9, 3, v1 // (op 942) 
v_lshl_add_u32 v11, v3, 2, v9 // (op 942) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 942) 

v_mul_lo_u32 v9, v3, s30 // (op 942) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 942) 


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


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[4:7], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135:
v_lshrrev_b32 v0, 6, v1 // (op 711) 
v_and_b32 v9, 63, v1 // (op 711) 
v_lshrrev_b32 v3, 4, v9 // (op 711) 
v_lshlrev_b32 v4, 2, v3 // (op 711) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 711) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 711) 

v_lshrrev_b32 v3, 2, v9 // (op 711) 

v_and_b32 v9, 3, v3 // (op 711) 

v_and_b32 v3, 3, v1 // (op 711) 
v_lshl_add_u32 v4, v9, 2, v3 // (op 711) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 711) 

v_mul_lo_u32 v9, v3, s36 // (op 711) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 711) 


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





buffer_load_dwordx2 v[4:5], v3, s[8:11], 0 offen // (op 148) Load value



v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v16, v10, v12 // (op 144) 
v_mul_f32 v17, v10, v13 // (op 144) 
v_mul_f32 v18, v10, v14 // (op 144) 
v_mul_f32 v19, v10, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 149) 
v_lshrrev_b32 v7, 16, v4 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v12, v8, v6 // (op 149) 
v_mul_f32 v13, v8, v7 // (op 149) 
v_cvt_f32_f16 v6, v5 // (op 149) 
v_lshrrev_b32 v7, 16, v5 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v14, v8, v6 // (op 149) 
v_mul_f32 v15, v8, v7 // (op 149) 


v_add_f32 v4, v12, v16 // (op 145) 
v_add_f32 v5, v13, v17 // (op 145) 
v_add_f32 v6, v14, v18 // (op 145) 
v_add_f32 v7, v15, v19 // (op 145) 


v_cvt_f16_f32 v12, v4 // (op 146) 
v_cvt_f16_f32 v13, v5 // (op 146) 
v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 

v_lshrrev_b32 v0, 6, v1 // (op 959) 
v_and_b32 v9, 63, v1 // (op 959) 
v_lshrrev_b32 v11, 4, v9 // (op 959) 
v_lshlrev_b32 v3, 2, v11 // (op 959) 

v_lshl_add_u32 v11, v0, 4, v3 // (op 959) 


v_lshl_add_u32 v0, s16, 4, v11 // (op 959) 

v_lshrrev_b32 v11, 2, v9 // (op 959) 

v_and_b32 v9, 3, v11 // (op 959) 

v_and_b32 v11, 3, v1 // (op 959) 
v_lshl_add_u32 v3, v9, 2, v11 // (op 959) 


v_lshl_add_u32 v9, s17, 4, v3 // (op 959) 

v_mul_lo_u32 v11, v9, s30 // (op 959) 

v_add_lshl_u32 v9, v0, v11, 1 // (op 959) 


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


v_pack_b32_f16 v6, v12, v13 // (op 147) 
v_pack_b32_f16 v7, v14, v15 // (op 147) 


buffer_store_dwordx2 v[6:7], v9, s[4:7], 0 offen // (op 147) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135:



s_endpgm  // End of RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC
.LRR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_end:
.size RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC, .LRR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC_end-RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC
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
    DataTypeB: B6
    DestDataType: h
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
  - .name: RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC
    .symbol: RR_GEMM_TT_BF6_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_LDSSYNC.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 7424
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 72
    .agpr_count: 4
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

