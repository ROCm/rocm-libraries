// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2,@function
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 765) 
v_and_b32 v3, 48, v1 // (op 765) 
v_sub_u32 v4, v0, v3 // (op 765) 

v_lshrrev_b32 v3, 4, v0 // (op 765) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 765) 


v_lshrrev_b32 v3, 6, v1 // (op 765) 
v_lshlrev_b32 v4, 4, v3 // (op 765) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 765) 


v_mul_lo_u32 v0, v3, 6 // (op 765) 
v_lshrrev_b32 v4, 3, v0 // (op 765) 

v_lshrrev_b32 v0, 4, v3 // (op 765) 

v_lshlrev_b32 v3, 2, v0 // (op 765) 

v_add_u32 v0, v4, v3 // (op 765) 


v_and_b32 v3, 15, v1 // (op 818) 
v_and_b32 v4, 63, v1 // (op 818) 
v_lshrrev_b32 v5, 4, v4 // (op 818) 

v_lshlrev_b32 v4, 5, v5 // (op 818) 

v_lshl_add_u32 v5, v3, 7, v4 // (op 818) 


v_mul_lo_u32 v3, v5, 6 // (op 818) 

v_lshrrev_b32 v4, 3, v3 // (op 818) 

v_lshrrev_b32 v3, 6, v1 // (op 871) 
v_and_b32 v5, 15, v1 // (op 871) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 871) 


v_and_b32 v3, 48, v1 // (op 871) 
v_add_u32 v5, v3, v6 // (op 871) 


v_and_b32 v3, 15, v1 // (op 924) 
v_and_b32 v6, 63, v1 // (op 924) 
v_lshrrev_b32 v7, 4, v6 // (op 924) 

v_lshl_add_u32 v6, v3, 2, v7 // (op 924) 


v_and_b32 v3, 63, v1 // (op 982) 
v_and_b32 v7, 48, v1 // (op 982) 
v_sub_u32 v8, v3, v7 // (op 982) 

v_lshrrev_b32 v7, 4, v3 // (op 982) 

v_lshl_add_u32 v3, v7, 5, v8 // (op 982) 


v_lshrrev_b32 v7, 6, v1 // (op 982) 
v_lshlrev_b32 v8, 4, v7 // (op 982) 

v_lshl_add_u32 v7, v3, 4, v8 // (op 982) 


v_mul_lo_u32 v3, v7, 6 // (op 982) 
v_lshrrev_b32 v8, 3, v3 // (op 982) 

v_lshrrev_b32 v3, 4, v7 // (op 982) 

v_lshlrev_b32 v7, 2, v3 // (op 982) 

v_add_u32 v3, v8, v7 // (op 982) 


v_and_b32 v7, 15, v1 // (op 1006) 
v_and_b32 v8, 63, v1 // (op 1006) 
v_lshrrev_b32 v9, 4, v8 // (op 1006) 

v_lshlrev_b32 v8, 5, v9 // (op 1006) 

v_lshl_add_u32 v9, v7, 7, v8 // (op 1006) 


v_mul_lo_u32 v7, v9, 6 // (op 1006) 

v_lshrrev_b32 v8, 3, v7 // (op 1006) 

v_lshrrev_b32 v7, 6, v1 // (op 1030) 
v_and_b32 v9, 15, v1 // (op 1030) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1030) 


v_and_b32 v7, 48, v1 // (op 1030) 
v_add_u32 v9, v7, v10 // (op 1030) 


v_and_b32 v7, 15, v1 // (op 1054) 
v_and_b32 v10, 63, v1 // (op 1054) 
v_lshrrev_b32 v11, 4, v10 // (op 1054) 

v_lshl_add_u32 v10, v7, 2, v11 // (op 1054) 


v_mov_b32 v7, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_lshl_b32 s0, s16, 4 // (op 475) 
v_lshlrev_b32 v12, 1, v1 // (op 475) 
s_mov_b32 s15, 0 // (op 475) convert
s_mov_b32 s14, s0 // (op 475) convert

v_mul_lo_u32 v15, v12, s3 // (op 475) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 475) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 475) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 475) most significant: sum


v_mov_b32 v12, s14 // (op 475) 
v_mov_b32 v13, s15 // (op 475) 
v_add_co_u32 v16, s[0:1], v12, v14 // (op 475) least significant half
v_addc_co_u32 v17, s[0:1], v13, v15, s[0:1] // (op 475) most significant half





v_mul_lo_u32 v13, v17, 6 // (op 475) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 475) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 475) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 475) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 475) 

v_mov_b32 v12, v14 // (op 475) convert

s_mul_i32 s1, s7, 6 // (op 477) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 477) least significant
s_mul_hi_u32 s14, s6, 6 // (op 477) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 477) most significant: sum

s_mul_i32 s15, s2, s1 // (op 477) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 477) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 477) least significant
s_mul_hi_u32 s19, s2, s0 // (op 477) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 477) most significant: sum
s_add_u32 s15, s15, s19 // (op 477) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 477) 

s_mov_b32 s40, s4 // (op 477) reg expression
s_mov_b32 s41, s5 // (op 477) reg expression
s_mov_b32 s42, s0 // (op 477) reg expression
s_mov_b32 s43, 131072 // (op 477) call()

s_lshl_b32 s0, s16, 4 // (op 494) 
v_lshlrev_b32 v13, 1, v1 // (op 494) 
s_mov_b32 s15, 0 // (op 494) convert
s_mov_b32 s14, s0 // (op 494) convert

v_mul_lo_u32 v15, v13, s3 // (op 494) most significant: low of low * high
v_mul_hi_u32 v16, v13, s2 // (op 494) most significant: high of low * low
v_mul_lo_u32 v14, v13, s2 // (op 494) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 494) most significant: sum


v_mov_b32 v13, s14 // (op 494) 
v_mov_b32 v16, s15 // (op 494) 
v_add_co_u32 v18, s[0:1], v13, v14 // (op 494) least significant half
v_addc_co_u32 v19, s[0:1], v16, v15, s[0:1] // (op 494) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 494) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 494) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 494) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 494) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 494) 

v_mov_b32 v13, v14 // (op 494) convert

s_mul_i32 s1, s7, 6 // (op 496) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 496) least significant
s_mul_hi_u32 s14, s6, 6 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 496) most significant: sum

s_mul_i32 s15, s2, s1 // (op 496) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 496) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 496) least significant
s_mul_hi_u32 s19, s2, s0 // (op 496) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 496) most significant: sum
s_add_u32 s15, s15, s19 // (op 496) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 496) 

s_mov_b32 s40, s4 // (op 496) reg expression
s_mov_b32 s41, s5 // (op 496) reg expression
s_mov_b32 s42, s0 // (op 496) reg expression
s_mov_b32 s43, 131072 // (op 496) call()

v_and_b32 v14, -4, v1 // (op 539) 
v_sub_u32 v15, v1, v14 // (op 539) 

v_lshlrev_b32 v14, 5, v15 // (op 539) 

v_mov_b32 v17, 0 // (op 539) convert
v_mov_b32 v16, v14 // (op 539) convert

v_lshrrev_b32 v14, 2, v1 // (op 539) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 539) 

v_mul_lo_u32 v19, v15, s9 // (op 539) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 539) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 539) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 539) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 539) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 539) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 539) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 539) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 539) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 539) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 539) 

v_mov_b32 v16, v14 // (op 539) convert

s_mul_i32 s1, s13, 6 // (op 541) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 541) least significant
s_mul_hi_u32 s14, s12, 6 // (op 541) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 541) most significant: sum

s_mul_i32 s15, s8, s1 // (op 541) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 541) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 541) least significant
s_mul_hi_u32 s19, s8, s0 // (op 541) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 541) most significant: sum
s_add_u32 s15, s15, s19 // (op 541) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 541) 

s_mov_b32 s44, s10 // (op 541) reg expression
s_mov_b32 s45, s11 // (op 541) reg expression
s_mov_b32 s46, s0 // (op 541) reg expression
s_mov_b32 s47, 131072 // (op 541) call()

v_and_b32 v14, -4, v1 // (op 558) 
v_sub_u32 v15, v1, v14 // (op 558) 

v_lshlrev_b32 v14, 5, v15 // (op 558) 

v_mov_b32 v19, 0 // (op 558) convert
v_mov_b32 v18, v14 // (op 558) convert

v_lshrrev_b32 v17, 2, v1 // (op 558) 
v_lshl_add_u32 v14, s17, 4, v17 // (op 558) 

v_mul_lo_u32 v21, v14, s9 // (op 558) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 558) most significant: high of low * low
v_mul_lo_u32 v20, v14, s8 // (op 558) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 558) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 558) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 558) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 558) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 558) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 558) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 558) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 558) 

v_mov_b32 v17, v14 // (op 558) convert

s_mul_i32 s1, s13, 6 // (op 560) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 560) least significant
s_mul_hi_u32 s14, s12, 6 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 560) most significant: sum

s_mul_i32 s15, s8, s1 // (op 560) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 560) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 560) least significant
s_mul_hi_u32 s19, s8, s0 // (op 560) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 560) most significant: sum
s_add_u32 s15, s15, s19 // (op 560) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 560) 

s_mov_b32 s44, s10 // (op 560) reg expression
s_mov_b32 s45, s11 // (op 560) reg expression
s_mov_b32 s46, s0 // (op 560) reg expression
s_mov_b32 s47, 131072 // (op 560) call()

v_and_b32 v14, -16, v1 // (op 603) 
v_sub_u32 v15, v1, v14 // (op 603) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 603) 

v_lshrrev_b32 v15, 4, v1 // (op 603) 

s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v18, v15, s20 // (op 603) 

v_add_u32 v15, v14, v18 // (op 603) 


s_mul_i32 s1, s20, s25 // (op 605) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 605) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 605) least significant
s_mul_hi_u32 s15, s20, s24 // (op 605) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 605) most significant: sum
s_add_u32 s1, s1, s15 // (op 605) most significant: sum


s_mov_b32 s48, s22 // (op 605) reg expression
s_mov_b32 s49, s23 // (op 605) reg expression
s_mov_b32 s50, s0 // (op 605) reg expression
s_mov_b32 s51, 131072 // (op 605) call()

v_and_b32 v14, -16, v1 // (op 622) 
v_sub_u32 v18, v1, v14 // (op 622) 

v_lshl_add_u32 v14, s16, 4, v18 // (op 622) 

v_lshrrev_b32 v18, 4, v1 // (op 622) 
v_mul_lo_u32 v19, v18, s20 // (op 622) 

v_add_u32 v18, v14, v19 // (op 622) 


s_mul_i32 s1, s20, s25 // (op 624) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 624) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 624) least significant
s_mul_hi_u32 s15, s20, s24 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 624) most significant: sum
s_add_u32 s1, s1, s15 // (op 624) most significant: sum


s_mov_b32 s48, s22 // (op 624) reg expression
s_mov_b32 s49, s23 // (op 624) reg expression
s_mov_b32 s50, s0 // (op 624) reg expression
s_mov_b32 s51, 131072 // (op 624) call()

v_and_b32 v14, -4, v1 // (op 667) 
v_sub_u32 v19, v1, v14 // (op 667) 

v_lshrrev_b32 v14, 2, v1 // (op 667) 
v_lshl_add_u32 v20, s17, 4, v14 // (op 667) 

v_mul_lo_u32 v14, v20, s26 // (op 667) 

v_add_u32 v20, v19, v14 // (op 667) 


s_mul_i32 s1, s26, s13 // (op 669) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 669) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 669) least significant
s_mul_hi_u32 s15, s26, s12 // (op 669) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 669) most significant: sum
s_add_u32 s1, s1, s15 // (op 669) most significant: sum


s_mov_b32 s52, s28 // (op 669) reg expression
s_mov_b32 s53, s29 // (op 669) reg expression
s_mov_b32 s54, s0 // (op 669) reg expression
s_mov_b32 s55, 131072 // (op 669) call()

v_and_b32 v14, -4, v1 // (op 686) 
v_sub_u32 v19, v1, v14 // (op 686) 

v_lshrrev_b32 v14, 2, v1 // (op 686) 
v_lshl_add_u32 v21, s17, 4, v14 // (op 686) 

v_mul_lo_u32 v14, v21, s26 // (op 686) 

v_add_u32 v21, v19, v14 // (op 686) 


s_mul_i32 s1, s26, s13 // (op 688) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 688) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 688) least significant
s_mul_hi_u32 s15, s26, s12 // (op 688) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 688) most significant: sum
s_add_u32 s1, s1, s15 // (op 688) most significant: sum


s_mov_b32 s52, s28 // (op 688) reg expression
s_mov_b32 s53, s29 // (op 688) reg expression
s_mov_b32 s54, s0 // (op 688) reg expression
s_mov_b32 s55, 131072 // (op 688) call()




buffer_load_dwordx4 v[22:25], v16, s[44:47], 0 offen // (op 366) Load value
buffer_load_dwordx2 v[26:27], v16, s[44:47], 0 offen offset:16 // (op 366) Load value



buffer_load_ubyte v14, v20, s[52:55], 0 offen offset: 0 // (op 369) Load value

s_mul_i32 s1, s3, 6 // (op 372) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 372) least significant
s_mul_hi_u32 s14, s2, 6 // (op 372) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 372) most significant: sum

v_lshrrev_b64 v[28:29], 3, s[0:1] // (op 372) 


v_mov_b32 v19, v12 // (op 372) reg expression
buffer_load_dwordx3 v[30:32], v19, s[40:43], 0 offen // (op 372) Load value
v_add_u32 v19, v19, v28 // (op 372) 
buffer_load_dwordx3 v[34:36], v19, s[40:43], 0 offen // (op 372) Load value




v_mov_b32 v19, v15 // (op 375) reg expression

buffer_load_ubyte v33, v19, s[48:51], 0 offen offset: 0 // (op 375) Load value




buffer_load_dwordx4 v[38:41], v16, s[44:47], 0 offen offset:96 // (op 378) Load value
buffer_load_dwordx2 v[42:43], v16, s[44:47], 0 offen offset:112 // (op 378) Load value




buffer_load_ubyte v16, v20, s[52:55], 0 offen offset:4 // (op 381) Load value


s_mul_i32 s1, s3, 6 // (op 384) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 384) least significant
s_mul_hi_u32 s14, s2, 6 // (op 384) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 384) most significant: sum

v_lshrrev_b64 v[28:29], 3, s[0:1] // (op 384) 


s_lshl_b64 s[0:1], s[2:3], 7 // (op 384) 
s_mul_i32 s15, s1, 6 // (op 384) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 384) least significant
s_mul_hi_u32 s18, s0, 6 // (op 384) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 384) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 384) 

v_add_u32 v37, v12, s0 // (op 384) 

buffer_load_dwordx3 v[44:46], v37, s[40:43], 0 offen // (op 384) Load value
v_add_u32 v37, v37, v28 // (op 384) 
buffer_load_dwordx3 v[48:50], v37, s[40:43], 0 offen // (op 384) Load value





s_lshl_b32 s0, s20, 2 // (op 387) 
v_add_u32 v12, v15, s0 // (op 387) 


buffer_load_ubyte v37, v12, s[48:51], 0 offen offset: 0 // (op 387) Load value


v_and_b32 v12, -4, v1 // (op 806) 
v_sub_u32 v15, v1, v12 // (op 806) 

v_lshlrev_b32 v12, 5, v15 // (op 806) 

v_lshrrev_b32 v15, 2, v1 // (op 806) 
v_lshl_add_u32 v47, v15, 7, v12 // (op 806) 


v_mul_lo_u32 v12, v47, 6 // (op 806) 

v_lshrrev_b32 v15, 3, v12 // (op 806) 




s_waitcnt vmcnt(11)
ds_write_b128 v15, v[22:25] // (op 389) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v15, v[26:27] offset:16 // (op 389) Store local data 

v_and_b32 v12, -4, v1 // (op 912) 
v_sub_u32 v15, v1, v12 // (op 912) 

v_and_b32 v12, 4294967292, v1 // (op 912) 
v_add_u32 v47, v12, v15 // (op 912) 





s_waitcnt vmcnt(9)
ds_write_b8 v47, v14 offset:1536 // (op 390) Store local data 

v_lshlrev_b32 v12, 5, v1 // (op 753) 
v_mul_lo_u32 v15, v12, 6 // (op 753) 
v_lshrrev_b32 v47, 3, v15 // (op 753) 

v_lshrrev_b32 v15, 4, v12 // (op 753) 

v_lshlrev_b32 v12, 2, v15 // (op 753) 

v_add_u32 v15, v47, v12 // (op 753) 





s_waitcnt vmcnt(8)
ds_write_b96 v15, v[30:32] offset:1600 // (op 391) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v15, v[34:36] offset:1616 // (op 391) Store local data 

v_and_b32 v12, -16, v1 // (op 859) 
v_sub_u32 v15, v1, v12 // (op 859) 

v_and_b32 v12, 4294967280, v1 // (op 859) 
v_add_u32 v47, v12, v15 // (op 859) 





s_waitcnt vmcnt(6)
ds_write_b8 v47, v33 offset:3648 // (op 392) Store local data 

v_lshlrev_b32 v12, 5, v1 // (op 724) 
v_mul_lo_u32 v15, v12, 6 // (op 724) 
v_lshrrev_b32 v47, 3, v15 // (op 724) 

v_lshrrev_b32 v15, 4, v12 // (op 724) 

v_lshlrev_b32 v12, 2, v15 // (op 724) 

v_add_u32 v15, v47, v12 // (op 724) 


v_and_b32 v12, -4, v1 // (op 777) 
v_sub_u32 v47, v1, v12 // (op 777) 

v_lshlrev_b32 v12, 5, v47 // (op 777) 

v_lshrrev_b32 v47, 2, v1 // (op 777) 
v_lshl_add_u32 v19, v47, 7, v12 // (op 777) 


v_mul_lo_u32 v12, v19, 6 // (op 777) 

v_lshrrev_b32 v47, 3, v12 // (op 777) 

v_and_b32 v12, -16, v1 // (op 830) 
v_sub_u32 v19, v1, v12 // (op 830) 

v_and_b32 v12, 4294967280, v1 // (op 830) 
v_add_u32 v20, v12, v19 // (op 830) 


v_and_b32 v12, -4, v1 // (op 883) 
v_sub_u32 v19, v1, v12 // (op 883) 

v_and_b32 v12, 4294967292, v1 // (op 883) 
v_add_u32 v28, v12, v19 // (op 883) 


v_lshlrev_b32 v12, 5, v1 // (op 970) 
v_mul_lo_u32 v19, v12, 6 // (op 970) 
v_lshrrev_b32 v29, 3, v19 // (op 970) 

v_lshrrev_b32 v19, 4, v12 // (op 970) 

v_lshlrev_b32 v12, 2, v19 // (op 970) 

v_add_u32 v19, v29, v12 // (op 970) 


v_and_b32 v12, -4, v1 // (op 994) 
v_sub_u32 v29, v1, v12 // (op 994) 

v_lshlrev_b32 v12, 5, v29 // (op 994) 

v_lshrrev_b32 v29, 2, v1 // (op 994) 
v_lshl_add_u32 v51, v29, 7, v12 // (op 994) 


v_mul_lo_u32 v12, v51, 6 // (op 994) 

v_lshrrev_b32 v29, 3, v12 // (op 994) 

v_and_b32 v12, -16, v1 // (op 1018) 
v_sub_u32 v51, v1, v12 // (op 1018) 

v_and_b32 v12, 4294967280, v1 // (op 1018) 
v_add_u32 v52, v12, v51 // (op 1018) 


v_and_b32 v12, -4, v1 // (op 1042) 
v_sub_u32 v51, v1, v12 // (op 1042) 

v_and_b32 v12, 4294967292, v1 // (op 1042) 
v_add_u32 v53, v12, v51 // (op 1042) 


s_mov_b32 s0, 0 // (op 59) call()
s_ashr_i64 s[14:15], s[6:7], 7 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_58:


s_add_u32 s0, s0, 2 // (op 60) 
v_mov_b32 v12, v17 // (op 10) 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1091) 
buffer_load_dwordx4 v[22:25], v12, s[44:47], 0 offen offset:192 // (op 10) Load value



ds_read_b96_tr_b6 v[54:56], v0 offset:1600 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[57:59], v0 offset:1856 // (op 70) Transpose load local data 
buffer_load_dwordx2 v[26:27], v12, s[44:47], 0 offen offset:208 // (op 10) Load value


s_barrier  // (op 1104) 
v_mov_b32 v51, v21 // (op 25) 




buffer_load_ubyte v14, v51, s[52:55], 0 offen offset:8 // (op 25) Load value
ds_read_b128 v[60:63], v4 // (op 75) Load local data 

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

ds_read_b64 v[64:65], v4 offset:16 // (op 75) Load local data 
buffer_load_dwordx3 v[30:32], v69, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v69, v69, v66 // (op 4) 
buffer_load_dwordx3 v[34:36], v69, s[40:43], 0 offen // (op 4) Load value


s_waitcnt lgkmcnt(4)
s_barrier  // (op 1118) 




ds_read_u8 v66, v5 offset:3648 // (op 80) Load local data 
v_mov_b32 v67, v18 // (op 17) 
s_barrier  // (op 1132) 
s_lshl_b32 s1, s20, 3 // (op 17) 



ds_read_u8 v69, v6 offset:1536 // (op 85) Load local data 
v_add_u32 v70, v67, s1 // (op 17) 


buffer_load_ubyte v33, v70, s[48:51], 0 offen offset: 0 // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:59], v[60:65], a[0:3], v66, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 88) 




s_barrier  // (op 1155) 



s_waitcnt vmcnt(11)
ds_write_b128 v29, v[38:41] offset:3712 // (op 253) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v29, v[42:43] offset:3728 // (op 253) Store local data 
s_barrier  // (op 1167) 



s_waitcnt vmcnt(9)
ds_write_b8 v53, v16 offset:5248 // (op 255) Store local data 
s_barrier  // (op 1149) 



s_waitcnt vmcnt(8)
ds_write_b96 v19, v[44:46] offset:5312 // (op 243) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v19, v[48:50] offset:5328 // (op 243) Store local data 
s_barrier  // (op 1161) 



s_waitcnt vmcnt(6)
ds_write_b8 v52, v37 offset:7360 // (op 254) Store local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 
buffer_load_dwordx4 v[38:41], v12, s[44:47], 0 offen offset:288 // (op 256) Load value



ds_read_b96_tr_b6 v[54:56], v3 offset:5312 // (op 245) Transpose load local data 
ds_read_b96_tr_b6 v[57:59], v3 offset:5568 // (op 245) Transpose load local data 
buffer_load_dwordx2 v[42:43], v12, s[44:47], 0 offen offset:304 // (op 256) Load value




s_barrier  // (op 1152) 
s_barrier  // (op 1222) 



v_add_u32 v17, 192, v17 // (op 550) 
ds_read_b128 v[60:63], v8 offset:3712 // (op 247) Load local data 
buffer_load_ubyte v16, v51, s[52:55], 0 offen offset:12 // (op 258) Load value
ds_read_b64 v[64:65], v8 offset:3728 // (op 247) Load local data 


s_barrier  // (op 1228) 
v_add_u32 v21, 8, v21 // (op 678) 
s_mul_i32 s15, s3, 6 // (op 242) most significant: low of high * low
s_mul_i32 s14, s2, 6 // (op 242) least significant
s_mul_hi_u32 s1, s2, 6 // (op 242) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 242) most significant: sum

v_lshrrev_b64 v[70:71], 3, s[14:15] // (op 242) 


s_lshl_b64 s[14:15], s[2:3], 7 // (op 242) 
s_mul_i32 s19, s15, 6 // (op 242) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 242) least significant
s_mul_hi_u32 s1, s14, 6 // (op 242) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 242) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 242) 

s_mul_i32 s1, 3, s14 // (op 242) 

v_add_u32 v12, v68, s1 // (op 242) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1158) 
buffer_load_dwordx3 v[44:46], v12, s[40:43], 0 offen // (op 242) Load value


v_add_u32 v12, v12, v70 // (op 242) 
ds_read_u8 v51, v9 offset:7360 // (op 249) Load local data 
buffer_load_dwordx3 v[48:50], v12, s[40:43], 0 offen // (op 242) Load value





s_barrier  // (op 1164) 
s_barrier  // (op 1234) 



ds_read_u8 v12, v10 offset:5248 // (op 251) Load local data 
s_lshl_b32 s1, s20, 2 // (op 257) 
s_mul_i32 s14, 3, s1 // (op 257) 

s_barrier  // (op 1240) 
v_add_u32 v66, v67, s14 // (op 257) 


s_lshl_b64 s[14:15], s[2:3], 7 // (op 486) 
buffer_load_ubyte v37, v66, s[48:51], 0 offen offset: 0 // (op 257) Load value

s_mul_i32 s19, s15, 6 // (op 486) most significant: low of high * low

s_mul_i32 s18, s14, 6 // (op 486) least significant
s_barrier  // (op 1107) 
s_mul_hi_u32 s1, s14, 6 // (op 486) most significant: high of low * low



s_add_u32 s19, s19, s1 // (op 486) most significant: sum


s_lshl_b32 s1, s20, 3 // (op 614) 
v_add_u32 v18, v18, s1 // (op 614) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 486) 

s_lshl_b32 s1, s14, 1 // (op 486) 

v_add_u32 v13, v13, s1 // (op 486) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:59], v[60:65], a[0:3], v51, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 252) 
s_waitcnt vmcnt(11)
ds_write_b128 v47, v[22:25] // (op 13) Store local data 




s_waitcnt vmcnt(10)
ds_write_b64 v47, v[26:27] offset:16 // (op 13) Store local data 
s_barrier  // (op 1135) 



s_waitcnt vmcnt(9)
ds_write_b8 v28, v14 offset:1536 // (op 30) Store local data 
s_barrier  // (op 1094) 



s_waitcnt vmcnt(8)
ds_write_b96 v15, v[30:32] offset:1600 // (op 6) Store local data 
s_waitcnt vmcnt(7)
ds_write_b96 v15, v[34:36] offset:1616 // (op 6) Store local data 
s_barrier  // (op 1121) 



s_waitcnt vmcnt(6)
ds_write_b8 v20, v33 offset:3648 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[6:7], 7 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoop_58:

















v_lshlrev_b32 v47, 1, v1 // (op 449) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 449) 
s_lshr_b32 s14, s0, 1 // (op 449) 

v_lshl_add_u32 v12, s14, 8, v47 // (op 449) 


s_lshl_b32 s0, s16, 4 // (op 449) 
v_mul_lo_u32 v15, v12, s3 // (op 449) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 449) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 449) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 449) most significant: sum


s_mov_b32 s15, 0 // (op 449) convert
s_mov_b32 s14, s0 // (op 449) convert

v_mov_b32 v47, s14 // (op 449) 
v_mov_b32 v12, s15 // (op 449) 
v_add_co_u32 v18, s[0:1], v47, v14 // (op 449) least significant half
v_addc_co_u32 v19, s[0:1], v12, v15, s[0:1] // (op 449) most significant half





v_mul_lo_u32 v13, v19, 6 // (op 449) most significant: low of high * low
v_mul_hi_u32 v17, v18, 6 // (op 449) most significant: high of low * low
v_mul_lo_u32 v12, v18, 6 // (op 449) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 449) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 449) 

v_mov_b32 v47, v14 // (op 449) convert

s_mul_i32 s1, s7, 6 // (op 451) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 451) least significant
s_mul_hi_u32 s14, s6, 6 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 451) most significant: sum

s_mul_i32 s15, s2, s1 // (op 451) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 451) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 451) least significant
s_mul_hi_u32 s19, s2, s0 // (op 451) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 451) most significant: sum
s_add_u32 s15, s15, s19 // (op 451) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 451) 

s_mov_b32 s40, s4 // (op 451) reg expression
s_mov_b32 s41, s5 // (op 451) reg expression
s_mov_b32 s42, s0 // (op 451) reg expression
s_mov_b32 s43, 131072 // (op 451) call()

v_and_b32 v12, -4, v1 // (op 516) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 516) 
v_sub_u32 v13, v1, v12 // (op 516) 

s_lshr_b32 s14, s0, 1 // (op 516) 

v_lshlrev_b32 v12, 5, v13 // (op 516) 

v_lshl_add_u32 v13, s14, 8, v12 // (op 516) 


v_mov_b32 v15, 0 // (op 516) convert
v_mov_b32 v14, v13 // (op 516) convert

v_lshrrev_b32 v12, 2, v1 // (op 516) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 516) 

v_mul_lo_u32 v19, v13, s9 // (op 516) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 516) most significant: high of low * low
v_mul_lo_u32 v18, v13, s8 // (op 516) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 516) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v18 // (op 516) least significant half
v_addc_co_u32 v13, s[0:1], v15, v19, s[0:1] // (op 516) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 516) most significant: low of high * low
v_mul_hi_u32 v17, v12, 6 // (op 516) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 516) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 516) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 516) 

v_mov_b32 v14, v12 // (op 516) convert

s_mul_i32 s1, s13, 6 // (op 518) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 518) least significant
s_mul_hi_u32 s14, s12, 6 // (op 518) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 518) most significant: sum

s_mul_i32 s15, s8, s1 // (op 518) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 518) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 518) least significant
s_mul_hi_u32 s19, s8, s0 // (op 518) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 518) most significant: sum
s_add_u32 s15, s15, s19 // (op 518) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 518) 

s_mov_b32 s44, s10 // (op 518) reg expression
s_mov_b32 s45, s11 // (op 518) reg expression
s_mov_b32 s46, s0 // (op 518) reg expression
s_mov_b32 s47, 131072 // (op 518) call()

v_and_b32 v15, -16, v1 // (op 580) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 580) 
v_sub_u32 v12, v1, v15 // (op 580) 

s_lshr_b32 s14, s0, 1 // (op 580) 

v_lshl_add_u32 v13, s16, 4, v12 // (op 580) 

v_lshrrev_b32 v12, 4, v1 // (op 580) 
v_lshl_add_u32 v15, s14, 3, v12 // (op 580) 


v_mul_lo_u32 v12, v15, s20 // (op 580) 

v_add_u32 v15, v13, v12 // (op 580) 


s_mul_i32 s1, s20, s25 // (op 582) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 582) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 582) least significant
s_mul_hi_u32 s15, s20, s24 // (op 582) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 582) most significant: sum
s_add_u32 s1, s1, s15 // (op 582) most significant: sum


s_mov_b32 s48, s22 // (op 582) reg expression
s_mov_b32 s49, s23 // (op 582) reg expression
s_mov_b32 s50, s0 // (op 582) reg expression
s_mov_b32 s51, 131072 // (op 582) call()

v_and_b32 v12, -4, v1 // (op 644) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 644) 
v_sub_u32 v13, v1, v12 // (op 644) 

s_lshr_b32 s14, s0, 1 // (op 644) 

v_lshrrev_b32 v12, 2, v1 // (op 644) 
v_lshl_add_u32 v17, s14, 3, v13 // (op 644) 


v_lshl_add_u32 v13, s17, 4, v12 // (op 644) 

v_mul_lo_u32 v12, v13, s26 // (op 644) 

v_add_u32 v13, v17, v12 // (op 644) 


s_mul_i32 s1, s26, s13 // (op 646) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 646) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 646) least significant
s_mul_hi_u32 s15, s26, s12 // (op 646) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 646) most significant: sum
s_add_u32 s1, s1, s15 // (op 646) most significant: sum


s_mov_b32 s52, s28 // (op 646) reg expression
s_mov_b32 s53, s29 // (op 646) reg expression
s_mov_b32 s54, s0 // (op 646) reg expression
s_mov_b32 s55, 131072 // (op 646) call()

v_lshlrev_b32 v12, 5, v1 // (op 737) 
v_mul_lo_u32 v17, v12, 6 // (op 737) 
v_lshrrev_b32 v18, 3, v17 // (op 737) 

v_lshrrev_b32 v17, 4, v12 // (op 737) 

v_lshlrev_b32 v12, 2, v17 // (op 737) 

v_add_u32 v17, v18, v12 // (op 737) 


v_and_b32 v12, -4, v1 // (op 790) 
v_sub_u32 v18, v1, v12 // (op 790) 

v_lshlrev_b32 v12, 5, v18 // (op 790) 

v_lshrrev_b32 v18, 2, v1 // (op 790) 
v_lshl_add_u32 v19, v18, 7, v12 // (op 790) 


v_mul_lo_u32 v12, v19, 6 // (op 790) 

v_lshrrev_b32 v18, 3, v12 // (op 790) 

v_and_b32 v12, -16, v1 // (op 843) 
v_sub_u32 v19, v1, v12 // (op 843) 

v_and_b32 v12, 4294967280, v1 // (op 843) 
v_add_u32 v20, v12, v19 // (op 843) 


v_and_b32 v12, -4, v1 // (op 896) 
v_sub_u32 v19, v1, v12 // (op 896) 

v_and_b32 v12, 4294967292, v1 // (op 896) 
v_add_u32 v21, v12, v19 // (op 896) 


s_ashr_i64 s[0:1], s[6:7], 7 // (op 172) 
s_and_b32 s14, s0, 4294967294 // (op 172) 

s_ashr_i64 s[0:1], s[6:7], 7 // (op 171) 
s_mov_b32 s15, s0 // (op 171) convert

s_cmp_lt_u32 s14, s15 // (op 171) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_171:

s_add_u32 s14, s14, 1 // (op 173) 
s_mul_i32 s1, s3, 6 // (op 176) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 176) least significant
s_mul_hi_u32 s15, s2, 6 // (op 176) most significant: high of low * low
s_add_u32 s1, s1, s15 // (op 176) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 176) 


v_mov_b32 v12, v47 // (op 176) 
v_mov_b32 v16, v12 // (op 176) reg expression
buffer_load_dwordx3 v[24:26], v16, s[40:43], 0 offen // (op 176) Load value
v_add_u32 v16, v16, v22 // (op 176) 
buffer_load_dwordx3 v[28:30], v16, s[40:43], 0 offen // (op 176) Load value



s_lshl_b64 s[0:1], s[2:3], 7 // (op 439) 
s_mul_i32 s19, s1, 6 // (op 439) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 439) least significant
s_mul_hi_u32 s15, s0, 6 // (op 439) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 439) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 439) 

v_add_u32 v47, v47, s0 // (op 439) 



s_barrier  // (op 1097) 
v_mov_b32 v12, v14 // (op 190) 




buffer_load_dwordx4 v[32:35], v12, s[44:47], 0 offen // (op 190) Load value
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[24:26] offset:1600 // (op 177) Store local data 
buffer_load_dwordx2 v[36:37], v12, s[44:47], 0 offen offset:16 // (op 190) Load value

v_add_u32 v14, 96, v14 // (op 508) 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[28:30] offset:1616 // (op 177) Store local data 



v_mov_b32 v12, v15 // (op 191) 
s_barrier  // (op 1110) 
v_mov_b32 v16, v12 // (op 191) reg expression




buffer_load_ubyte v19, v16, s[48:51], 0 offen offset: 0 // (op 191) Load value


s_lshl_b32 s15, s20, 2 // (op 572) 
v_add_u32 v15, v15, s15 // (op 572) 

s_waitcnt vmcnt(2)
ds_write_b128 v18, v[32:35] // (op 187) Store local data 


v_mov_b32 v12, v13 // (op 192) 

buffer_load_ubyte v16, v12, s[52:55], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v13, 4, v13 // (op 636) 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[36:37] offset:16 // (op 187) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1100) 



s_barrier  // (op 1124) 
ds_read_b96_tr_b6 v[22:24], v0 offset:1600 // (op 179) Transpose load local data 



ds_read_b96_tr_b6 v[25:27], v0 offset:1856 // (op 179) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v19 offset:3648 // (op 188) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1114) 



s_barrier  // (op 1138) 
ds_read_b128 v[28:31], v4 // (op 181) Load local data 



ds_read_b64 v[32:33], v4 offset:16 // (op 181) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v16 offset:1536 // (op 189) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1128) 


ds_read_u8 v12, v5 offset:3648 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1142) 



ds_read_u8 v16, v6 offset:1536 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[28:33], a[0:3], v12, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 186) 




s_ashr_i64 s[0:1], s[6:7], 7 // (op 171) 
s_mov_b32 s15, s0 // (op 171) convert

s_cmp_lt_u32 s14, s15 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopTop_KLoopTail_171 if true)
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ForLoopBottom_KLoopTail_171:























s_barrier  // (op 1216) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_133
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 938) 
v_and_b32 v3, 63, v1 // (op 938) 
v_lshrrev_b32 v4, 4, v3 // (op 938) 
v_lshlrev_b32 v5, 2, v4 // (op 938) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 938) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 938) 

v_lshrrev_b32 v4, 2, v3 // (op 938) 

v_and_b32 v3, 3, v4 // (op 938) 

v_and_b32 v4, 3, v1 // (op 938) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 938) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 938) 

v_mul_lo_u32 v3, v6, s30 // (op 938) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 938) 


s_mul_i32 s15, s30, s13 // (op 940) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 940) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 940) least significant
s_mul_hi_u32 s1, s30, s12 // (op 940) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 940) most significant: sum
s_add_u32 s15, s15, s1 // (op 940) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 940) 

s_mov_b32 s8, s32 // (op 940) reg expression
s_mov_b32 s9, s33 // (op 940) reg expression
s_mov_b32 s10, s0 // (op 940) reg expression
s_mov_b32 s11, 131072 // (op 940) call()




buffer_store_dwordx4 v[16:19], v4, s[8:11], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalFalse_0_beta_133:
v_lshrrev_b32 v0, 6, v1 // (op 707) 
v_and_b32 v3, 63, v1 // (op 707) 
v_lshrrev_b32 v4, 4, v3 // (op 707) 
v_lshlrev_b32 v5, 2, v4 // (op 707) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 707) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 707) 

v_lshrrev_b32 v4, 2, v3 // (op 707) 

v_and_b32 v3, 3, v4 // (op 707) 

v_and_b32 v4, 3, v1 // (op 707) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 707) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 707) 

v_mul_lo_u32 v3, v6, s34 // (op 707) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 707) 


s_mul_i32 s15, s34, s13 // (op 709) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 709) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 709) least significant
s_mul_hi_u32 s1, s34, s12 // (op 709) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 709) most significant: sum
s_add_u32 s15, s15, s1 // (op 709) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 709) 

s_mov_b32 s4, s36 // (op 709) reg expression
s_mov_b32 s5, s37 // (op 709) reg expression
s_mov_b32 s6, s0 // (op 709) reg expression
s_mov_b32 s7, 131072 // (op 709) call()





buffer_load_dwordx4 v[12:15], v4, s[4:7], 0 offen // (op 145) Load value



v_accvgpr_read v16, a0 // (op 142) 
v_accvgpr_read v17, a1 // (op 142) 
v_accvgpr_read v18, a2 // (op 142) 
v_accvgpr_read v19, a3 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 
v_mul_f32 v22, v11, v18 // (op 142) 
v_mul_f32 v23, v11, v19 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v7, v12 // (op 146) 
v_mul_f32 v17, v7, v13 // (op 146) 
v_mul_f32 v18, v7, v14 // (op 146) 
v_mul_f32 v19, v7, v15 // (op 146) 

v_add_f32 v12, v16, v20 // (op 143) 
v_add_f32 v13, v17, v21 // (op 143) 
v_add_f32 v14, v18, v22 // (op 143) 
v_add_f32 v15, v19, v23 // (op 143) 


v_lshrrev_b32 v0, 6, v1 // (op 955) 
v_and_b32 v3, 63, v1 // (op 955) 
v_lshrrev_b32 v4, 4, v3 // (op 955) 
v_lshlrev_b32 v5, 2, v4 // (op 955) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 955) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 955) 

v_lshrrev_b32 v4, 2, v3 // (op 955) 

v_and_b32 v3, 3, v4 // (op 955) 

v_and_b32 v4, 3, v1 // (op 955) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 955) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 955) 

v_mul_lo_u32 v3, v6, s30 // (op 955) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 955) 


s_mul_i32 s15, s30, s13 // (op 957) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 957) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 957) least significant
s_mul_hi_u32 s1, s30, s12 // (op 957) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 957) most significant: sum
s_add_u32 s15, s15, s1 // (op 957) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 957) 

s_mov_b32 s8, s32 // (op 957) reg expression
s_mov_b32 s9, s33 // (op 957) reg expression
s_mov_b32 s10, s0 // (op 957) reg expression
s_mov_b32 s11, 131072 // (op 957) call()




buffer_store_dwordx4 v[12:15], v4, s[8:11], 0 offen // (op 144) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_ConditionalBottom_0_beta_133:




s_endpgm  // End of RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.LRR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end:
.size RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2, .LRR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2_end-RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
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
    DestDataType: s
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
  - .name: RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2
    .symbol: RR_GEMM_NN_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x128_UR_2.kd
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

