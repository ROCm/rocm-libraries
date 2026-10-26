// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:


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
v_lshrrev_b32 v3, 4, v0 // (op 765) 

v_and_b32 v0, 1, v3 // (op 765) 

v_and_b32 v3, 15, v1 // (op 765) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 765) 


v_lshrrev_b32 v0, 6, v1 // (op 765) 
v_lshl_add_u32 v3, v0, 5, v4 // (op 765) 


v_and_b32 v0, 32, v1 // (op 765) 
v_lshl_add_u32 v4, v3, 6, v0 // (op 765) 


v_lshlrev_b32 v0, 2, v4 // (op 765) 

v_lshrrev_b32 v3, 3, v0 // (op 765) 

v_and_b32 v0, 63, v1 // (op 818) 
v_lshrrev_b32 v4, 4, v0 // (op 818) 

v_and_b32 v0, 1, v4 // (op 818) 

v_and_b32 v4, 15, v1 // (op 818) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 818) 


v_and_b32 v0, 32, v1 // (op 818) 
v_lshl_add_u32 v4, v5, 6, v0 // (op 818) 


v_mul_lo_u32 v0, v4, 6 // (op 818) 

v_lshrrev_b32 v4, 3, v0 // (op 818) 

v_lshrrev_b32 v0, 6, v1 // (op 871) 
v_and_b32 v5, 31, v1 // (op 871) 
v_lshl_add_u32 v6, v0, 5, v5 // (op 871) 


v_and_b32 v0, 63, v1 // (op 871) 
v_lshrrev_b32 v5, 5, v0 // (op 871) 

v_lshl_add_u32 v0, v6, 1, v5 // (op 871) 


v_and_b32 v5, 31, v1 // (op 924) 
v_and_b32 v6, 63, v1 // (op 924) 
v_lshrrev_b32 v7, 5, v6 // (op 924) 

v_lshl_add_u32 v6, v5, 1, v7 // (op 924) 


v_and_b32 v5, 63, v1 // (op 982) 
v_lshrrev_b32 v7, 4, v5 // (op 982) 

v_and_b32 v5, 1, v7 // (op 982) 

v_and_b32 v7, 15, v1 // (op 982) 
v_lshl_add_u32 v8, v5, 4, v7 // (op 982) 


v_lshrrev_b32 v5, 6, v1 // (op 982) 
v_lshl_add_u32 v7, v5, 5, v8 // (op 982) 


v_and_b32 v5, 32, v1 // (op 982) 
v_lshl_add_u32 v8, v7, 6, v5 // (op 982) 


v_lshlrev_b32 v5, 2, v8 // (op 982) 

v_lshrrev_b32 v7, 3, v5 // (op 982) 

v_and_b32 v5, 63, v1 // (op 1006) 
v_lshrrev_b32 v8, 4, v5 // (op 1006) 

v_and_b32 v5, 1, v8 // (op 1006) 

v_and_b32 v8, 15, v1 // (op 1006) 
v_lshl_add_u32 v9, v5, 4, v8 // (op 1006) 


v_and_b32 v5, 32, v1 // (op 1006) 
v_lshl_add_u32 v8, v9, 6, v5 // (op 1006) 


v_mul_lo_u32 v5, v8, 6 // (op 1006) 

v_lshrrev_b32 v8, 3, v5 // (op 1006) 

v_lshrrev_b32 v5, 6, v1 // (op 1030) 
v_and_b32 v9, 31, v1 // (op 1030) 
v_lshl_add_u32 v10, v5, 5, v9 // (op 1030) 


v_and_b32 v5, 63, v1 // (op 1030) 
v_lshrrev_b32 v9, 5, v5 // (op 1030) 

v_lshl_add_u32 v5, v10, 1, v9 // (op 1030) 


v_and_b32 v9, 31, v1 // (op 1054) 
v_and_b32 v10, 63, v1 // (op 1054) 
v_lshrrev_b32 v11, 5, v10 // (op 1054) 

v_lshl_add_u32 v10, v9, 1, v11 // (op 1054) 


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 63) call()
v_accvgpr_write a5, 0.00000 // (op 63) call()
v_accvgpr_write a6, 0.00000 // (op 63) call()
v_accvgpr_write a7, 0.00000 // (op 63) call()
v_accvgpr_write a8, 0.00000 // (op 63) call()
v_accvgpr_write a9, 0.00000 // (op 63) call()
v_accvgpr_write a10, 0.00000 // (op 63) call()
v_accvgpr_write a11, 0.00000 // (op 63) call()
v_accvgpr_write a12, 0.00000 // (op 63) call()
v_accvgpr_write a13, 0.00000 // (op 63) call()
v_accvgpr_write a14, 0.00000 // (op 63) call()
v_accvgpr_write a15, 0.00000 // (op 63) call()
v_lshrrev_b32 v12, 1, v1 // (op 475) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 475) 

v_and_b32 v12, -2, v1 // (op 475) 
v_sub_u32 v14, v1, v12 // (op 475) 

v_lshlrev_b32 v12, 5, v14 // (op 475) 

v_mov_b32 v15, 0 // (op 475) convert
v_mov_b32 v14, v12 // (op 475) convert

v_mul_lo_u32 v17, v13, s3 // (op 475) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 475) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 475) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 475) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 475) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 475) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 475) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 475) 

v_mov_b32 v12, v14 // (op 475) convert

s_mul_i32 s1, s2, s7 // (op 477) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 477) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 477) least significant
s_mul_hi_u32 s15, s2, s6 // (op 477) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 477) most significant: sum
s_add_u32 s1, s1, s15 // (op 477) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 477) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 477) 

s_mov_b32 s40, s4 // (op 477) reg expression
s_mov_b32 s41, s5 // (op 477) reg expression
s_mov_b32 s42, s0 // (op 477) reg expression
s_mov_b32 s43, 131072 // (op 477) call()

v_lshrrev_b32 v13, 1, v1 // (op 494) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 494) 

v_and_b32 v13, -2, v1 // (op 494) 
v_sub_u32 v15, v1, v13 // (op 494) 

v_lshlrev_b32 v13, 5, v15 // (op 494) 

v_mov_b32 v17, 0 // (op 494) convert
v_mov_b32 v16, v13 // (op 494) convert

v_mul_lo_u32 v19, v14, s3 // (op 494) most significant: low of low * high
v_mul_hi_u32 v13, v14, s2 // (op 494) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 494) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 494) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 494) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 494) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 494) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 494) 

v_mov_b32 v13, v14 // (op 494) convert

s_mul_i32 s1, s2, s7 // (op 496) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 496) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 496) least significant
s_mul_hi_u32 s15, s2, s6 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 496) most significant: sum
s_add_u32 s1, s1, s15 // (op 496) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 496) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 496) 

s_mov_b32 s40, s4 // (op 496) reg expression
s_mov_b32 s41, s5 // (op 496) reg expression
s_mov_b32 s42, s0 // (op 496) reg expression
s_mov_b32 s43, 131072 // (op 496) call()

v_and_b32 v14, -2, v1 // (op 539) 
v_sub_u32 v15, v1, v14 // (op 539) 

v_lshlrev_b32 v14, 5, v15 // (op 539) 

v_mov_b32 v17, 0 // (op 539) convert
v_mov_b32 v16, v14 // (op 539) convert

v_lshrrev_b32 v14, 1, v1 // (op 539) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 539) 

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

v_and_b32 v14, -2, v1 // (op 558) 
v_sub_u32 v15, v1, v14 // (op 558) 

v_lshlrev_b32 v14, 5, v15 // (op 558) 

v_mov_b32 v19, 0 // (op 558) convert
v_mov_b32 v18, v14 // (op 558) convert

v_lshrrev_b32 v17, 1, v1 // (op 558) 
v_lshl_add_u32 v14, s17, 5, v17 // (op 558) 

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

v_lshrrev_b32 v14, 1, v1 // (op 603) 
v_lshl_add_u32 v15, s16, 5, v14 // (op 603) 

v_and_b32 v14, -2, v1 // (op 603) 
v_sub_u32 v18, v1, v14 // (op 603) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v14, v15, s20 // (op 603) 
v_add_u32 v14, v14, v18 // (op 603) 


s_mul_i32 s1, s20, s7 // (op 605) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 605) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 605) least significant
s_mul_hi_u32 s15, s20, s6 // (op 605) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 605) most significant: sum
s_add_u32 s1, s1, s15 // (op 605) most significant: sum


s_mov_b32 s48, s22 // (op 605) reg expression
s_mov_b32 s49, s23 // (op 605) reg expression
s_mov_b32 s50, s0 // (op 605) reg expression
s_mov_b32 s51, 131072 // (op 605) call()

v_lshrrev_b32 v15, 1, v1 // (op 622) 
v_lshl_add_u32 v18, s16, 5, v15 // (op 622) 

v_and_b32 v15, -2, v1 // (op 622) 
v_sub_u32 v19, v1, v15 // (op 622) 

v_mul_lo_u32 v15, v18, s20 // (op 622) 
v_add_u32 v15, v15, v19 // (op 622) 


s_mul_i32 s1, s20, s7 // (op 624) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 624) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 624) least significant
s_mul_hi_u32 s15, s20, s6 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 624) most significant: sum
s_add_u32 s1, s1, s15 // (op 624) most significant: sum


s_mov_b32 s48, s22 // (op 624) reg expression
s_mov_b32 s49, s23 // (op 624) reg expression
s_mov_b32 s50, s0 // (op 624) reg expression
s_mov_b32 s51, 131072 // (op 624) call()

v_and_b32 v18, -2, v1 // (op 667) 
v_sub_u32 v19, v1, v18 // (op 667) 

v_lshrrev_b32 v18, 1, v1 // (op 667) 
v_lshl_add_u32 v20, s17, 5, v18 // (op 667) 

v_mul_lo_u32 v18, v20, s24 // (op 667) 

v_add_u32 v20, v19, v18 // (op 667) 


s_mul_i32 s1, s24, s13 // (op 669) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 669) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 669) least significant
s_mul_hi_u32 s15, s24, s12 // (op 669) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 669) most significant: sum
s_add_u32 s1, s1, s15 // (op 669) most significant: sum


s_mov_b32 s52, s26 // (op 669) reg expression
s_mov_b32 s53, s27 // (op 669) reg expression
s_mov_b32 s54, s0 // (op 669) reg expression
s_mov_b32 s55, 131072 // (op 669) call()

v_and_b32 v18, -2, v1 // (op 686) 
v_sub_u32 v19, v1, v18 // (op 686) 

v_lshrrev_b32 v18, 1, v1 // (op 686) 
v_lshl_add_u32 v21, s17, 5, v18 // (op 686) 

v_mul_lo_u32 v18, v21, s24 // (op 686) 

v_add_u32 v21, v19, v18 // (op 686) 


s_mul_i32 s1, s24, s13 // (op 688) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 688) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 688) least significant
s_mul_hi_u32 s15, s24, s12 // (op 688) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 688) most significant: sum
s_add_u32 s1, s1, s15 // (op 688) most significant: sum


s_mov_b32 s52, s26 // (op 688) reg expression
s_mov_b32 s53, s27 // (op 688) reg expression
s_mov_b32 s54, s0 // (op 688) reg expression
s_mov_b32 s55, 131072 // (op 688) call()




buffer_load_dwordx4 v[22:25], v16, s[44:47], 0 offen // (op 366) Load value
buffer_load_dwordx2 v[26:27], v16, s[44:47], 0 offen offset:16 // (op 366) Load value



buffer_load_ubyte v18, v20, s[52:55], 0 offen offset: 0 // (op 369) Load value



buffer_load_dwordx4 v[28:31], v12, s[40:43], 0 offen // (op 372) Load value



buffer_load_ubyte v19, v14, s[48:51], 0 offen offset: 0 // (op 375) Load value



buffer_load_dwordx4 v[32:35], v16, s[44:47], 0 offen offset:48 // (op 378) Load value
buffer_load_dwordx2 v[36:37], v16, s[44:47], 0 offen offset:64 // (op 378) Load value




buffer_load_ubyte v16, v20, s[52:55], 0 offen offset:2 // (op 381) Load value




buffer_load_dwordx4 v[38:41], v12, s[40:43], 0 offen offset:32 // (op 384) Load value




buffer_load_ubyte v12, v14, s[48:51], 0 offen offset:2 // (op 387) Load value

v_and_b32 v14, -2, v1 // (op 806) 
v_sub_u32 v20, v1, v14 // (op 806) 

v_lshlrev_b32 v14, 5, v20 // (op 806) 

v_lshrrev_b32 v20, 1, v1 // (op 806) 
v_lshl_add_u32 v42, v20, 6, v14 // (op 806) 


v_mul_lo_u32 v14, v42, 6 // (op 806) 

v_lshrrev_b32 v20, 3, v14 // (op 806) 




s_waitcnt vmcnt(9)
ds_write_b128 v20, v[22:25] // (op 389) Store local data 
s_waitcnt vmcnt(8)
ds_write_b64 v20, v[26:27] offset:16 // (op 389) Store local data 

v_and_b32 v14, -2, v1 // (op 912) 
v_sub_u32 v20, v1, v14 // (op 912) 

v_and_b32 v14, 4294967294, v1 // (op 912) 
v_add_u32 v42, v14, v20 // (op 912) 





s_waitcnt vmcnt(7)
ds_write_b8 v42, v18 offset:1536 // (op 390) Store local data 

v_and_b32 v14, -2, v1 // (op 753) 
v_sub_u32 v20, v1, v14 // (op 753) 

v_lshlrev_b32 v14, 5, v20 // (op 753) 

v_lshrrev_b32 v20, 1, v1 // (op 753) 
v_lshl_add_u32 v42, v20, 6, v14 // (op 753) 


v_lshlrev_b32 v14, 2, v42 // (op 753) 

v_lshrrev_b32 v20, 3, v14 // (op 753) 




s_waitcnt vmcnt(6)
ds_write_b128 v20, v[28:31] offset:1600 // (op 391) Store local data 

v_and_b32 v14, -2, v1 // (op 859) 
v_sub_u32 v20, v1, v14 // (op 859) 

v_and_b32 v14, 4294967294, v1 // (op 859) 
v_add_u32 v42, v14, v20 // (op 859) 





s_waitcnt vmcnt(5)
ds_write_b8 v42, v19 offset:2624 // (op 392) Store local data 

v_and_b32 v14, -2, v1 // (op 724) 
v_sub_u32 v20, v1, v14 // (op 724) 

v_lshlrev_b32 v14, 5, v20 // (op 724) 

v_lshrrev_b32 v20, 1, v1 // (op 724) 
v_lshl_add_u32 v42, v20, 6, v14 // (op 724) 


v_lshlrev_b32 v14, 2, v42 // (op 724) 

v_lshrrev_b32 v20, 3, v14 // (op 724) 

v_and_b32 v14, -2, v1 // (op 777) 
v_sub_u32 v42, v1, v14 // (op 777) 

v_lshlrev_b32 v14, 5, v42 // (op 777) 

v_lshrrev_b32 v42, 1, v1 // (op 777) 
v_lshl_add_u32 v43, v42, 6, v14 // (op 777) 


v_mul_lo_u32 v14, v43, 6 // (op 777) 

v_lshrrev_b32 v42, 3, v14 // (op 777) 

v_and_b32 v14, -2, v1 // (op 830) 
v_sub_u32 v43, v1, v14 // (op 830) 

v_and_b32 v14, 4294967294, v1 // (op 830) 
v_add_u32 v44, v14, v43 // (op 830) 


v_and_b32 v14, -2, v1 // (op 883) 
v_sub_u32 v43, v1, v14 // (op 883) 

v_and_b32 v14, 4294967294, v1 // (op 883) 
v_add_u32 v45, v14, v43 // (op 883) 


v_and_b32 v14, -2, v1 // (op 970) 
v_sub_u32 v43, v1, v14 // (op 970) 

v_lshlrev_b32 v14, 5, v43 // (op 970) 

v_lshrrev_b32 v43, 1, v1 // (op 970) 
v_lshl_add_u32 v46, v43, 6, v14 // (op 970) 


v_lshlrev_b32 v14, 2, v46 // (op 970) 

v_lshrrev_b32 v43, 3, v14 // (op 970) 

v_and_b32 v14, -2, v1 // (op 994) 
v_sub_u32 v46, v1, v14 // (op 994) 

v_lshlrev_b32 v14, 5, v46 // (op 994) 

v_lshrrev_b32 v46, 1, v1 // (op 994) 
v_lshl_add_u32 v47, v46, 6, v14 // (op 994) 


v_mul_lo_u32 v14, v47, 6 // (op 994) 

v_lshrrev_b32 v46, 3, v14 // (op 994) 

v_and_b32 v14, -2, v1 // (op 1018) 
v_sub_u32 v47, v1, v14 // (op 1018) 

v_and_b32 v14, 4294967294, v1 // (op 1018) 
v_add_u32 v48, v14, v47 // (op 1018) 


v_and_b32 v14, -2, v1 // (op 1042) 
v_sub_u32 v47, v1, v14 // (op 1042) 

v_and_b32 v14, 4294967294, v1 // (op 1042) 
v_add_u32 v49, v14, v47 // (op 1042) 


s_mov_b32 s0, 0 // (op 59) call()
s_ashr_i64 s[14:15], s[28:29], 6 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58:


s_add_u32 s0, s0, 2 // (op 60) 
v_mov_b32 v14, v17 // (op 10) 

buffer_load_dwordx4 v[22:25], v14, s[44:47], 0 offen offset:96 // (op 10) Load value
buffer_load_dwordx2 v[26:27], v14, s[44:47], 0 offen offset:112 // (op 10) Load value
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1091) 





v_mov_b32 v47, v21 // (op 25) 

ds_read_b128 v[50:53], v3 offset:1600 // (op 70) Load local data 
buffer_load_ubyte v18, v47, s[52:55], 0 offen offset:4 // (op 25) Load value


s_barrier  // (op 1104) 
v_mov_b32 v54, v13 // (op 4) 




ds_read_b128 v[56:59], v4 // (op 75) Load local data 
ds_read_b64 v[60:61], v4 offset:16 // (op 75) Load local data 
buffer_load_dwordx4 v[28:31], v54, s[40:43], 0 offen offset:64 // (op 4) Load value


v_mov_b32 v55, v15 // (op 17) 

buffer_load_ubyte v19, v55, s[48:51], 0 offen offset:4 // (op 17) Load value
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1118) 



s_barrier  // (op 1155) 
ds_read_u8 v62, v0 offset:2624 // (op 80) Load local data 



s_barrier  // (op 1132) 
s_waitcnt vmcnt(9)
ds_write_b128 v46, v[32:35] offset:2688 // (op 253) Store local data 



ds_read_u8 v63, v6 offset:1536 // (op 85) Load local data 
s_waitcnt vmcnt(8)
ds_write_b64 v46, v[36:37] offset:2704 // (op 253) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[50:53], v[56:61], a[0:15], v62, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 88) 




s_barrier  // (op 1167) 



s_waitcnt vmcnt(7)
ds_write_b8 v49, v16 offset:4224 // (op 255) Store local data 
s_barrier  // (op 1149) 



s_waitcnt vmcnt(6)
ds_write_b128 v43, v[38:41] offset:4288 // (op 243) Store local data 
s_barrier  // (op 1161) 



s_waitcnt vmcnt(5)
ds_write_b8 v48, v12 offset:5312 // (op 254) Store local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 
buffer_load_dwordx4 v[32:35], v14, s[44:47], 0 offen offset:144 // (op 256) Load value



ds_read_b128 v[50:53], v7 offset:4288 // (op 245) Load local data 
buffer_load_dwordx2 v[36:37], v14, s[44:47], 0 offen offset:160 // (op 256) Load value




s_barrier  // (op 1152) 
s_barrier  // (op 1222) 



buffer_load_ubyte v16, v47, s[52:55], 0 offen offset:6 // (op 258) Load value
v_add_u32 v17, 96, v17 // (op 550) 

ds_read_b128 v[56:59], v8 offset:2688 // (op 247) Load local data 



ds_read_b64 v[60:61], v8 offset:2704 // (op 247) Load local data 
v_add_u32 v21, 4, v21 // (op 678) 
buffer_load_dwordx4 v[38:41], v54, s[40:43], 0 offen offset:96 // (op 242) Load value

s_barrier  // (op 1228) 



v_add_u32 v13, 64, v13 // (op 486) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1158) 
buffer_load_ubyte v12, v55, s[48:51], 0 offen offset:6 // (op 257) Load value




ds_read_u8 v14, v5 offset:5312 // (op 249) Load local data 
s_barrier  // (op 1107) 



s_barrier  // (op 1164) 
s_barrier  // (op 1234) 



v_add_u32 v15, 4, v15 // (op 614) 
ds_read_u8 v47, v10 offset:4224 // (op 251) Load local data 
s_waitcnt vmcnt(9)
ds_write_b128 v42, v[22:25] // (op 13) Store local data 
s_waitcnt vmcnt(8)
ds_write_b64 v42, v[26:27] offset:16 // (op 13) Store local data 
s_barrier  // (op 1240) 
s_barrier  // (op 1135) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[50:53], v[56:61], a[0:15], v14, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 252) 







s_waitcnt vmcnt(7)
ds_write_b8 v45, v18 offset:1536 // (op 30) Store local data 
s_barrier  // (op 1094) 



s_waitcnt vmcnt(6)
ds_write_b128 v20, v[28:31] offset:1600 // (op 6) Store local data 
s_barrier  // (op 1121) 



s_waitcnt vmcnt(5)
ds_write_b8 v44, v19 offset:2624 // (op 21) Store local data 
s_ashr_i64 s[14:15], s[28:29], 6 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58:

















v_lshrrev_b32 v13, 1, v1 // (op 449) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 449) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 449) 

s_lshr_b32 s14, s0, 1 // (op 449) 

v_and_b32 v13, -2, v1 // (op 449) 
v_sub_u32 v15, v1, v13 // (op 449) 

v_lshlrev_b32 v13, 5, v15 // (op 449) 

v_lshl_add_u32 v15, s14, 7, v13 // (op 449) 


v_mov_b32 v31, 0 // (op 449) convert
v_mov_b32 v30, v15 // (op 449) convert

v_mul_lo_u32 v29, v14, s3 // (op 449) most significant: low of low * high
v_mul_hi_u32 v13, v14, s2 // (op 449) most significant: high of low * low
v_mul_lo_u32 v28, v14, s2 // (op 449) least significant: low of low * low
v_add_u32 v29, v29, v13 // (op 449) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v30 // (op 449) least significant half
v_addc_co_u32 v29, s[0:1], v29, v31, s[0:1] // (op 449) most significant half



v_lshlrev_b64 v[14:15], 2, v[28:29] // (op 449) 

v_lshrrev_b64 v[30:31], 3, v[14:15] // (op 449) 

v_mov_b32 v13, v30 // (op 449) convert

s_mul_i32 s1, s2, s7 // (op 451) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 451) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 451) least significant
s_mul_hi_u32 s15, s2, s6 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 451) most significant: sum
s_add_u32 s1, s1, s15 // (op 451) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 451) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 451) 

s_mov_b32 s40, s4 // (op 451) reg expression
s_mov_b32 s41, s5 // (op 451) reg expression
s_mov_b32 s42, s0 // (op 451) reg expression
s_mov_b32 s43, 131072 // (op 451) call()

v_and_b32 v14, -2, v1 // (op 516) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 516) 
v_sub_u32 v15, v1, v14 // (op 516) 

s_lshr_b32 s14, s0, 1 // (op 516) 

v_lshlrev_b32 v14, 5, v15 // (op 516) 

v_lshl_add_u32 v15, s14, 7, v14 // (op 516) 


v_mov_b32 v31, 0 // (op 516) convert
v_mov_b32 v30, v15 // (op 516) convert

v_lshrrev_b32 v14, 1, v1 // (op 516) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 516) 

v_mul_lo_u32 v29, v15, s9 // (op 516) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 516) most significant: high of low * low
v_mul_lo_u32 v28, v15, s8 // (op 516) least significant: low of low * low
v_add_u32 v29, v29, v14 // (op 516) most significant: sum


v_add_co_u32 v14, s[0:1], v30, v28 // (op 516) least significant half
v_addc_co_u32 v15, s[0:1], v31, v29, s[0:1] // (op 516) most significant half



v_mul_lo_u32 v31, v15, 6 // (op 516) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 516) most significant: high of low * low
v_mul_lo_u32 v30, v14, 6 // (op 516) least significant: low of low * low
v_add_u32 v31, v31, v17 // (op 516) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[30:31] // (op 516) 

v_mov_b32 v17, v14 // (op 516) convert

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

v_lshrrev_b32 v14, 1, v1 // (op 580) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 580) 
v_lshl_add_u32 v15, s16, 5, v14 // (op 580) 

s_lshr_b32 s14, s0, 1 // (op 580) 

v_and_b32 v14, -2, v1 // (op 580) 
v_sub_u32 v18, v1, v14 // (op 580) 

v_lshl_add_u32 v14, s14, 2, v18 // (op 580) 


v_mul_lo_u32 v18, v15, s20 // (op 580) 
v_add_u32 v18, v18, v14 // (op 580) 


s_mul_i32 s1, s20, s7 // (op 582) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 582) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 582) least significant
s_mul_hi_u32 s15, s20, s6 // (op 582) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 582) most significant: sum
s_add_u32 s1, s1, s15 // (op 582) most significant: sum


s_mov_b32 s48, s22 // (op 582) reg expression
s_mov_b32 s49, s23 // (op 582) reg expression
s_mov_b32 s50, s0 // (op 582) reg expression
s_mov_b32 s51, 131072 // (op 582) call()

v_and_b32 v14, -2, v1 // (op 644) 
s_ashr_i64 s[0:1], s[28:29], 6 // (op 644) 
v_sub_u32 v15, v1, v14 // (op 644) 

s_lshr_b32 s14, s0, 1 // (op 644) 

v_lshrrev_b32 v14, 1, v1 // (op 644) 
v_lshl_add_u32 v19, s14, 2, v15 // (op 644) 


v_lshl_add_u32 v15, s17, 5, v14 // (op 644) 

v_mul_lo_u32 v14, v15, s24 // (op 644) 

v_add_u32 v15, v19, v14 // (op 644) 


s_mul_i32 s1, s24, s13 // (op 646) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 646) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 646) least significant
s_mul_hi_u32 s15, s24, s12 // (op 646) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 646) most significant: sum
s_add_u32 s1, s1, s15 // (op 646) most significant: sum


s_mov_b32 s52, s26 // (op 646) reg expression
s_mov_b32 s53, s27 // (op 646) reg expression
s_mov_b32 s54, s0 // (op 646) reg expression
s_mov_b32 s55, 131072 // (op 646) call()

v_and_b32 v14, -2, v1 // (op 737) 
v_sub_u32 v19, v1, v14 // (op 737) 

v_lshlrev_b32 v14, 5, v19 // (op 737) 

v_lshrrev_b32 v19, 1, v1 // (op 737) 
v_lshl_add_u32 v20, v19, 6, v14 // (op 737) 


v_lshlrev_b32 v14, 2, v20 // (op 737) 

v_lshrrev_b32 v19, 3, v14 // (op 737) 

v_and_b32 v14, -2, v1 // (op 790) 
v_sub_u32 v20, v1, v14 // (op 790) 

v_lshlrev_b32 v14, 5, v20 // (op 790) 

v_lshrrev_b32 v20, 1, v1 // (op 790) 
v_lshl_add_u32 v21, v20, 6, v14 // (op 790) 


v_mul_lo_u32 v14, v21, 6 // (op 790) 

v_lshrrev_b32 v20, 3, v14 // (op 790) 

v_and_b32 v14, -2, v1 // (op 843) 
v_sub_u32 v21, v1, v14 // (op 843) 

v_and_b32 v14, 4294967294, v1 // (op 843) 
v_add_u32 v22, v14, v21 // (op 843) 


v_and_b32 v14, -2, v1 // (op 896) 
v_sub_u32 v21, v1, v14 // (op 896) 

v_and_b32 v14, 4294967294, v1 // (op 896) 
v_add_u32 v23, v14, v21 // (op 896) 


s_ashr_i64 s[0:1], s[28:29], 6 // (op 172) 
s_and_b32 s14, s0, 4294967294 // (op 172) 

s_ashr_i64 s[0:1], s[28:29], 6 // (op 171) 
s_mov_b32 s15, s0 // (op 171) convert

s_cmp_lt_u32 s14, s15 // (op 171) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171:


s_add_u32 s14, s14, 1 // (op 173) 
v_mov_b32 v12, v13 // (op 176) 

buffer_load_dwordx4 v[24:27], v12, s[40:43], 0 offen // (op 176) Load value

v_add_u32 v13, 32, v13 // (op 439) 


s_barrier  // (op 1097) 
v_mov_b32 v12, v17 // (op 190) 




buffer_load_dwordx4 v[28:31], v12, s[44:47], 0 offen // (op 190) Load value
buffer_load_dwordx2 v[32:33], v12, s[44:47], 0 offen offset:16 // (op 190) Load value

v_add_u32 v17, 48, v17 // (op 508) 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[24:27] offset:1600 // (op 177) Store local data 



v_mov_b32 v12, v18 // (op 191) 

s_barrier  // (op 1110) 
buffer_load_ubyte v14, v12, s[48:51], 0 offen offset: 0 // (op 191) Load value




v_add_u32 v18, 2, v18 // (op 572) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[28:31] // (op 187) Store local data 


v_mov_b32 v12, v15 // (op 192) 

buffer_load_ubyte v16, v12, s[52:55], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v15, 2, v15 // (op 636) 
s_waitcnt vmcnt(2)
ds_write_b64 v20, v[32:33] offset:16 // (op 187) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1100) 



s_barrier  // (op 1124) 
ds_read_b128 v[24:27], v3 offset:1600 // (op 179) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 1114) 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v14 offset:2624 // (op 188) Store local data 




ds_read_b128 v[28:31], v4 // (op 181) Load local data 
s_barrier  // (op 1138) 
ds_read_b64 v[32:33], v4 offset:16 // (op 181) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 1128) 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v16 offset:1536 // (op 189) Store local data 




ds_read_u8 v12, v0 offset:2624 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1142) 



ds_read_u8 v14, v6 offset:1536 // (op 185) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[24:27], v[28:33], a[0:15], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 186) 




s_ashr_i64 s[0:1], s[28:29], 6 // (op 171) 
s_mov_b32 s15, s0 // (op 171) convert

s_cmp_lt_u32 s14, s15 // (op 171) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171:























s_barrier  // (op 1216) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133
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
v_mul_f32 v28, v11, v12 // (op 50) 
v_mul_f32 v29, v11, v13 // (op 50) 
v_mul_f32 v30, v11, v14 // (op 50) 
v_mul_f32 v31, v11, v15 // (op 50) 
v_mul_f32 v32, v11, v16 // (op 50) 
v_mul_f32 v33, v11, v17 // (op 50) 
v_mul_f32 v34, v11, v18 // (op 50) 
v_mul_f32 v35, v11, v19 // (op 50) 
v_mul_f32 v36, v11, v20 // (op 50) 
v_mul_f32 v37, v11, v21 // (op 50) 
v_mul_f32 v38, v11, v22 // (op 50) 
v_mul_f32 v39, v11, v23 // (op 50) 
v_mul_f32 v40, v11, v24 // (op 50) 
v_mul_f32 v41, v11, v25 // (op 50) 
v_mul_f32 v42, v11, v26 // (op 50) 
v_mul_f32 v43, v11, v27 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 938) 
v_and_b32 v10, 63, v1 // (op 938) 
v_lshrrev_b32 v3, 5, v10 // (op 938) 
v_lshlrev_b32 v4, 2, v3 // (op 938) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 938) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 938) 

v_lshrrev_b32 v3, 2, v10 // (op 938) 

v_and_b32 v10, 7, v3 // (op 938) 

v_and_b32 v3, 3, v1 // (op 938) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 938) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 938) 

v_mul_lo_u32 v10, v3, s30 // (op 938) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 938) 


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




buffer_store_dwordx4 v[28:31], v3, s[8:11], 0 offen // (op 56) Store value
buffer_store_dwordx4 v[32:35], v3, s[8:11], 0 offen offset:32 // (op 56) Store value
buffer_store_dwordx4 v[36:39], v3, s[8:11], 0 offen offset:64 // (op 56) Store value
buffer_store_dwordx4 v[40:43], v3, s[8:11], 0 offen offset:96 // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133:
v_lshrrev_b32 v0, 6, v1 // (op 707) 
v_and_b32 v10, 63, v1 // (op 707) 
v_lshrrev_b32 v3, 5, v10 // (op 707) 
v_lshlrev_b32 v4, 2, v3 // (op 707) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 707) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 707) 

v_lshrrev_b32 v3, 2, v10 // (op 707) 

v_and_b32 v10, 7, v3 // (op 707) 

v_and_b32 v3, 3, v1 // (op 707) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 707) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 707) 

v_mul_lo_u32 v10, v3, s34 // (op 707) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 707) 


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





buffer_load_dwordx4 v[12:15], v3, s[4:7], 0 offen // (op 145) Load value
buffer_load_dwordx4 v[16:19], v3, s[4:7], 0 offen offset:32 // (op 145) Load value
buffer_load_dwordx4 v[20:23], v3, s[4:7], 0 offen offset:64 // (op 145) Load value
buffer_load_dwordx4 v[24:27], v3, s[4:7], 0 offen offset:96 // (op 145) Load value



v_accvgpr_read v28, a0 // (op 142) 
v_accvgpr_read v29, a1 // (op 142) 
v_accvgpr_read v30, a2 // (op 142) 
v_accvgpr_read v31, a3 // (op 142) 
v_accvgpr_read v32, a4 // (op 142) 
v_accvgpr_read v33, a5 // (op 142) 
v_accvgpr_read v34, a6 // (op 142) 
v_accvgpr_read v35, a7 // (op 142) 
v_accvgpr_read v36, a8 // (op 142) 
v_accvgpr_read v37, a9 // (op 142) 
v_accvgpr_read v38, a10 // (op 142) 
v_accvgpr_read v39, a11 // (op 142) 
v_accvgpr_read v40, a12 // (op 142) 
v_accvgpr_read v41, a13 // (op 142) 
v_accvgpr_read v42, a14 // (op 142) 
v_accvgpr_read v43, a15 // (op 142) 
v_mul_f32 v44, v11, v28 // (op 142) 
v_mul_f32 v45, v11, v29 // (op 142) 
v_mul_f32 v46, v11, v30 // (op 142) 
v_mul_f32 v47, v11, v31 // (op 142) 
v_mul_f32 v48, v11, v32 // (op 142) 
v_mul_f32 v49, v11, v33 // (op 142) 
v_mul_f32 v50, v11, v34 // (op 142) 
v_mul_f32 v51, v11, v35 // (op 142) 
v_mul_f32 v52, v11, v36 // (op 142) 
v_mul_f32 v53, v11, v37 // (op 142) 
v_mul_f32 v54, v11, v38 // (op 142) 
v_mul_f32 v55, v11, v39 // (op 142) 
v_mul_f32 v56, v11, v40 // (op 142) 
v_mul_f32 v57, v11, v41 // (op 142) 
v_mul_f32 v58, v11, v42 // (op 142) 
v_mul_f32 v59, v11, v43 // (op 142) 

s_waitcnt vmcnt(3)
v_mul_f32 v28, v9, v12 // (op 146) 
v_mul_f32 v29, v9, v13 // (op 146) 
v_mul_f32 v30, v9, v14 // (op 146) 
v_mul_f32 v31, v9, v15 // (op 146) 
s_waitcnt vmcnt(2)
v_mul_f32 v32, v9, v16 // (op 146) 
v_mul_f32 v33, v9, v17 // (op 146) 
v_mul_f32 v34, v9, v18 // (op 146) 
v_mul_f32 v35, v9, v19 // (op 146) 
s_waitcnt vmcnt(1)
v_mul_f32 v36, v9, v20 // (op 146) 
v_mul_f32 v37, v9, v21 // (op 146) 
v_mul_f32 v38, v9, v22 // (op 146) 
v_mul_f32 v39, v9, v23 // (op 146) 
s_waitcnt vmcnt(0)
v_mul_f32 v40, v9, v24 // (op 146) 
v_mul_f32 v41, v9, v25 // (op 146) 
v_mul_f32 v42, v9, v26 // (op 146) 
v_mul_f32 v43, v9, v27 // (op 146) 

v_add_f32 v12, v28, v44 // (op 143) 
v_add_f32 v13, v29, v45 // (op 143) 
v_add_f32 v14, v30, v46 // (op 143) 
v_add_f32 v15, v31, v47 // (op 143) 
v_add_f32 v16, v32, v48 // (op 143) 
v_add_f32 v17, v33, v49 // (op 143) 
v_add_f32 v18, v34, v50 // (op 143) 
v_add_f32 v19, v35, v51 // (op 143) 
v_add_f32 v20, v36, v52 // (op 143) 
v_add_f32 v21, v37, v53 // (op 143) 
v_add_f32 v22, v38, v54 // (op 143) 
v_add_f32 v23, v39, v55 // (op 143) 
v_add_f32 v24, v40, v56 // (op 143) 
v_add_f32 v25, v41, v57 // (op 143) 
v_add_f32 v26, v42, v58 // (op 143) 
v_add_f32 v27, v43, v59 // (op 143) 


v_lshrrev_b32 v0, 6, v1 // (op 955) 
v_and_b32 v10, 63, v1 // (op 955) 
v_lshrrev_b32 v3, 5, v10 // (op 955) 
v_lshlrev_b32 v4, 2, v3 // (op 955) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 955) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 955) 

v_lshrrev_b32 v3, 2, v10 // (op 955) 

v_and_b32 v10, 7, v3 // (op 955) 

v_and_b32 v3, 3, v1 // (op 955) 
v_lshl_add_u32 v4, v10, 2, v3 // (op 955) 


v_lshl_add_u32 v3, s17, 5, v4 // (op 955) 

v_mul_lo_u32 v10, v3, s30 // (op 955) 

v_add_lshl_u32 v3, v0, v10, 2 // (op 955) 


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




buffer_store_dwordx4 v[12:15], v3, s[8:11], 0 offen // (op 144) Store value
buffer_store_dwordx4 v[16:19], v3, s[8:11], 0 offen offset:32 // (op 144) Store value
buffer_store_dwordx4 v[20:23], v3, s[8:11], 0 offen offset:64 // (op 144) Store value
buffer_store_dwordx4 v[24:27], v3, s[8:11], 0 offen offset:96 // (op 144) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133:




s_endpgm  // End of RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
  .amdhsa_next_free_vgpr 80
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 5376 // lds bytes
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: F6
    DestDataType: s
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
    macrotile: [32, 32, 64]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TN_FP4_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 5376
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 64
    .agpr_count: 16
    .max_flat_workgroup_size: 64
    .workgroup_size: [64, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

