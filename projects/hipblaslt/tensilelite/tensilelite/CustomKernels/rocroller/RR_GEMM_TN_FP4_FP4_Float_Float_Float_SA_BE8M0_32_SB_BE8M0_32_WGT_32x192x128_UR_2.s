// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2857) 
v_and_b32 v3, 15, v1 // (op 2857) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2857) 


v_and_b32 v0, 63, v1 // (op 2857) 
v_lshrrev_b32 v3, 4, v0 // (op 2857) 

v_lshlrev_b32 v0, 5, v3 // (op 2857) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2857) 


v_lshlrev_b32 v0, 2, v3 // (op 2857) 

v_lshrrev_b32 v3, 3, v0 // (op 2857) 

v_lshrrev_b32 v0, 7, v1 // (op 3015) 
v_and_b32 v4, 15, v1 // (op 3015) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3015) 


v_and_b32 v0, 63, v1 // (op 3015) 
v_lshrrev_b32 v4, 4, v0 // (op 3015) 

v_lshlrev_b32 v0, 5, v4 // (op 3015) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 3015) 


v_lshlrev_b32 v0, 2, v4 // (op 3015) 

v_lshrrev_b32 v4, 3, v0 // (op 3015) 

v_lshrrev_b32 v0, 7, v1 // (op 2800) 
v_and_b32 v5, 63, v1 // (op 2800) 
v_lshrrev_b32 v6, 4, v5 // (op 2800) 
v_lshlrev_b32 v7, 2, v6 // (op 2800) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 2800) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 2800) 

v_lshrrev_b32 v6, 2, v5 // (op 2800) 

v_and_b32 v5, 3, v6 // (op 2800) 

v_and_b32 v6, 3, v1 // (op 2800) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 2800) 


v_lshrrev_b32 v5, 6, v1 // (op 2800) 
v_and_b32 v6, 1, v5 // (op 2800) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 2800) 


v_mov_b32 v6, 192 // (op 2800) 
v_mul_lo_u32 v7, s17, v6 // (op 2800) 

v_add_u32 v7, v7, v5 // (op 2800) 

v_mul_lo_u32 v5, v7, s2 // (op 2800) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 2800) 


v_lshrrev_b32 v0, 6, v1 // (op 2915) 
v_and_b32 v5, 1, v0 // (op 2915) 

v_and_b32 v0, 15, v1 // (op 2915) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 2915) 


v_and_b32 v0, 63, v1 // (op 2915) 
v_lshrrev_b32 v5, 4, v0 // (op 2915) 

v_lshlrev_b32 v0, 5, v5 // (op 2915) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 2915) 


v_lshlrev_b32 v0, 2, v5 // (op 2915) 

v_lshrrev_b32 v5, 3, v0 // (op 2915) 

v_lshrrev_b32 v0, 6, v1 // (op 2973) 
v_and_b32 v7, 1, v0 // (op 2973) 

v_and_b32 v0, 15, v1 // (op 2973) 
s_mul_i32 s1, s2, s9 // (op 2802) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2802) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2802) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2802) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2802) most significant: sum
s_add_u32 s1, s1, s19 // (op 2802) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2802) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 2973) 


v_and_b32 v0, 63, v1 // (op 2973) 
v_lshrrev_b32 v7, 4, v0 // (op 2973) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2973) 


v_lshrrev_b32 v7, 7, v1 // (op 2989) 
v_and_b32 v8, 63, v1 // (op 2989) 
v_lshrrev_b32 v9, 4, v8 // (op 2989) 
v_lshlrev_b32 v10, 2, v9 // (op 2989) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2989) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 2989) 

v_lshrrev_b32 v9, 2, v8 // (op 2989) 

v_and_b32 v8, 3, v9 // (op 2989) 

v_and_b32 v9, 3, v1 // (op 2989) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2989) 


v_lshrrev_b32 v8, 6, v1 // (op 2989) 
v_and_b32 v9, 1, v8 // (op 2989) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2989) 


v_mov_b32 v9, 192 // (op 2989) 
v_mul_lo_u32 v10, s17, v9 // (op 2989) 

v_add_u32 v10, v10, v8 // (op 2989) 

v_mul_lo_u32 v8, v10, s4 // (op 2989) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 2989) 


v_lshrrev_b32 v7, 6, v1 // (op 3044) 
v_and_b32 v8, 1, v7 // (op 3044) 

v_and_b32 v7, 15, v1 // (op 3044) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 3044) 


v_and_b32 v7, 63, v1 // (op 3044) 
v_lshrrev_b32 v8, 4, v7 // (op 3044) 

v_lshlrev_b32 v7, 5, v8 // (op 3044) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 3044) 


v_lshlrev_b32 v7, 2, v8 // (op 3044) 

v_lshrrev_b32 v8, 3, v7 // (op 3044) 

v_lshrrev_b32 v7, 6, v1 // (op 3073) 
v_and_b32 v10, 1, v7 // (op 3073) 

v_and_b32 v7, 15, v1 // (op 3073) 
s_mul_i32 s1, s4, s9 // (op 2991) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 3073) 


v_and_b32 v7, 63, v1 // (op 3073) 
v_lshrrev_b32 v10, 4, v7 // (op 3073) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 3073) 


s_mul_i32 s38, s5, s8 // (op 2991) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2991) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2991) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2991) most significant: sum
s_add_u32 s1, s1, s39 // (op 2991) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2991) 

s_mov_b32 s40, s6 // (op 2802) reg expression
s_mov_b32 s41, s7 // (op 2802) reg expression
s_mov_b32 s42, s18 // (op 2802) reg expression
s_mov_b32 s43, 131072 // (op 2802) call()

s_mov_b32 s44, s10 // (op 2991) reg expression

s_mov_b32 s45, s11 // (op 2991) reg expression
s_mov_b32 s46, s38 // (op 2991) reg expression
s_mov_b32 s47, 131072 // (op 2991) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 306) call()
v_accvgpr_write a5, 0.00000 // (op 306) call()
v_accvgpr_write a6, 0.00000 // (op 306) call()
v_accvgpr_write a7, 0.00000 // (op 306) call()
v_accvgpr_write a8, 0.00000 // (op 462) call()
v_accvgpr_write a9, 0.00000 // (op 462) call()
v_accvgpr_write a10, 0.00000 // (op 462) call()
v_accvgpr_write a11, 0.00000 // (op 462) call()
v_accvgpr_write a12, 0.00000 // (op 618) call()
v_accvgpr_write a13, 0.00000 // (op 618) call()
v_accvgpr_write a14, 0.00000 // (op 618) call()
v_accvgpr_write a15, 0.00000 // (op 618) call()
v_accvgpr_write a16, 0.00000 // (op 774) call()
v_accvgpr_write a17, 0.00000 // (op 774) call()
v_accvgpr_write a18, 0.00000 // (op 774) call()
v_accvgpr_write a19, 0.00000 // (op 774) call()
v_accvgpr_write a20, 0.00000 // (op 930) call()
v_accvgpr_write a21, 0.00000 // (op 930) call()
v_accvgpr_write a22, 0.00000 // (op 930) call()
v_accvgpr_write a23, 0.00000 // (op 930) call()
v_lshrrev_b32 v12, 3, v1 // (op 2567) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 2567) 

v_and_b32 v12, -8, v1 // (op 2567) 
v_sub_u32 v14, v1, v12 // (op 2567) 

v_lshlrev_b32 v12, 4, v14 // (op 2567) 

v_mov_b32 v15, 0 // (op 2567) convert
v_mov_b32 v14, v12 // (op 2567) convert

v_mul_lo_u32 v17, v13, s13 // (op 2567) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 2567) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 2567) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 2567) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 2567) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 2567) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 2567) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 2567) 

v_mov_b32 v12, v14 // (op 2567) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 2569) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2569) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2569) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2569) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2569) most significant: sum
s_add_u32 s1, s1, s7 // (op 2569) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2569) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2569) 

s_mov_b32 s48, s20 // (op 2569) reg expression
s_mov_b32 s49, s21 // (op 2569) reg expression
s_mov_b32 s50, s0 // (op 2569) reg expression
s_mov_b32 s51, 131072 // (op 2569) call()

v_lshrrev_b32 v13, 3, v1 // (op 2586) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 2586) 

v_and_b32 v13, -8, v1 // (op 2586) 
v_sub_u32 v15, v1, v13 // (op 2586) 

v_lshlrev_b32 v13, 4, v15 // (op 2586) 

v_mov_b32 v17, 0 // (op 2586) convert
v_mov_b32 v16, v13 // (op 2586) convert

v_mul_lo_u32 v19, v14, s13 // (op 2586) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 2586) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 2586) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 2586) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2586) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2586) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 2586) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 2586) 

v_mov_b32 v13, v14 // (op 2586) convert

s_mul_i32 s1, s12, s23 // (op 2588) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2588) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2588) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2588) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2588) most significant: sum
s_add_u32 s1, s1, s7 // (op 2588) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2588) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2588) 

s_mov_b32 s48, s20 // (op 2588) reg expression
s_mov_b32 s49, s21 // (op 2588) reg expression
s_mov_b32 s50, s0 // (op 2588) reg expression
s_mov_b32 s51, 131072 // (op 2588) call()

v_and_b32 v14, -4, v1 // (op 2631) 
v_sub_u32 v15, v1, v14 // (op 2631) 

v_lshlrev_b32 v14, 5, v15 // (op 2631) 

v_mov_b32 v17, 0 // (op 2631) convert
v_mov_b32 v16, v14 // (op 2631) convert

v_lshrrev_b32 v14, 2, v1 // (op 2631) 
v_mul_lo_u32 v15, v14, 3 // (op 2631) 

v_mov_b32 v14, 192 // (op 2631) 
v_mul_lo_u32 v18, s17, v14 // (op 2631) 

v_add_u32 v18, v18, v15 // (op 2631) 

v_mul_lo_u32 v15, v18, s25 // (op 2631) most significant: low of low * high
v_mul_hi_u32 v19, v18, s24 // (op 2631) most significant: high of low * low
v_mul_lo_u32 v14, v18, s24 // (op 2631) least significant: low of low * low
v_add_u32 v15, v15, v19 // (op 2631) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v14 // (op 2631) least significant half
v_addc_co_u32 v19, s[0:1], v17, v15, s[0:1] // (op 2631) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 2631) 


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 2631) 

v_mov_b32 v16, v14 // (op 2631) convert

s_mul_i32 s1, s24, s9 // (op 2633) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 2633) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2633) least significant
s_mul_hi_u32 s7, s24, s8 // (op 2633) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2633) most significant: sum
s_add_u32 s1, s1, s7 // (op 2633) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2633) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2633) 

s_mov_b32 s52, s26 // (op 2633) reg expression
s_mov_b32 s53, s27 // (op 2633) reg expression
s_mov_b32 s54, s0 // (op 2633) reg expression
s_mov_b32 s55, 131072 // (op 2633) call()

v_and_b32 v14, -4, v1 // (op 2650) 
v_sub_u32 v15, v1, v14 // (op 2650) 

v_lshlrev_b32 v14, 5, v15 // (op 2650) 

v_mov_b32 v19, 0 // (op 2650) convert
v_mov_b32 v18, v14 // (op 2650) convert

v_lshrrev_b32 v17, 2, v1 // (op 2650) 
v_mul_lo_u32 v14, v17, 3 // (op 2650) 

v_mov_b32 v15, 192 // (op 2650) 
v_mul_lo_u32 v17, s17, v15 // (op 2650) 

v_add_u32 v17, v17, v14 // (op 2650) 

v_mul_lo_u32 v15, v17, s25 // (op 2650) most significant: low of low * high
v_mul_hi_u32 v20, v17, s24 // (op 2650) most significant: high of low * low
v_mul_lo_u32 v14, v17, s24 // (op 2650) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 2650) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v14 // (op 2650) least significant half
v_addc_co_u32 v21, s[0:1], v19, v15, s[0:1] // (op 2650) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 2650) 


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 2650) 

v_mov_b32 v17, v14 // (op 2650) convert

s_mul_i32 s1, s24, s9 // (op 2652) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 2652) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2652) least significant
s_mul_hi_u32 s7, s24, s8 // (op 2652) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2652) most significant: sum
s_add_u32 s1, s1, s7 // (op 2652) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2652) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2652) 

s_mov_b32 s52, s26 // (op 2652) reg expression
s_mov_b32 s53, s27 // (op 2652) reg expression
s_mov_b32 s54, s0 // (op 2652) reg expression
s_mov_b32 s55, 131072 // (op 2652) call()

v_lshrrev_b32 v14, 7, v1 // (op 2695) 
v_and_b32 v15, 15, v1 // (op 2695) 
v_lshl_add_u32 v18, v14, 4, v15 // (op 2695) 


v_lshl_add_u32 v14, s16, 5, v18 // (op 2695) 

v_and_b32 v15, 63, v1 // (op 2695) 
v_lshrrev_b32 v18, 4, v15 // (op 2695) 

v_mul_lo_u32 v15, v14, s28 // (op 2695) 
v_add_u32 v15, v15, v18 // (op 2695) 


s_mul_i32 s1, s28, s23 // (op 2697) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2697) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2697) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2697) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2697) most significant: sum
s_add_u32 s1, s1, s7 // (op 2697) most significant: sum


s_mov_b32 s56, s30 // (op 2697) reg expression
s_mov_b32 s57, s31 // (op 2697) reg expression
s_mov_b32 s58, s0 // (op 2697) reg expression
s_mov_b32 s59, 131072 // (op 2697) call()

v_lshrrev_b32 v14, 7, v1 // (op 2714) 
v_and_b32 v18, 15, v1 // (op 2714) 
v_lshl_add_u32 v19, v14, 4, v18 // (op 2714) 


v_lshl_add_u32 v14, s16, 5, v19 // (op 2714) 

v_and_b32 v18, 63, v1 // (op 2714) 
v_lshrrev_b32 v19, 4, v18 // (op 2714) 

v_mul_lo_u32 v18, v14, s28 // (op 2714) 
v_add_u32 v18, v18, v19 // (op 2714) 


s_mul_i32 s1, s28, s23 // (op 2716) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2716) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2716) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2716) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2716) most significant: sum
s_add_u32 s1, s1, s7 // (op 2716) most significant: sum


s_mov_b32 s56, s30 // (op 2716) reg expression
s_mov_b32 s57, s31 // (op 2716) reg expression
s_mov_b32 s58, s0 // (op 2716) reg expression
s_mov_b32 s59, 131072 // (op 2716) call()

v_and_b32 v14, -4, v1 // (op 2759) 
v_sub_u32 v19, v1, v14 // (op 2759) 

v_lshrrev_b32 v14, 2, v1 // (op 2759) 
v_mul_lo_u32 v20, v14, 3 // (op 2759) 

v_mov_b32 v14, 192 // (op 2759) 
v_mul_lo_u32 v21, s17, v14 // (op 2759) 

v_add_u32 v21, v21, v20 // (op 2759) 

v_mul_lo_u32 v14, v21, s32 // (op 2759) 

v_add_u32 v20, v19, v14 // (op 2759) 


s_mul_i32 s1, s32, s9 // (op 2761) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2761) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2761) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2761) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2761) most significant: sum
s_add_u32 s1, s1, s7 // (op 2761) most significant: sum


s_mov_b32 s60, s34 // (op 2761) reg expression
s_mov_b32 s61, s35 // (op 2761) reg expression
s_mov_b32 s62, s0 // (op 2761) reg expression
s_mov_b32 s63, 131072 // (op 2761) call()

v_and_b32 v14, -4, v1 // (op 2778) 
v_sub_u32 v19, v1, v14 // (op 2778) 

v_lshrrev_b32 v14, 2, v1 // (op 2778) 
v_mul_lo_u32 v21, v14, 3 // (op 2778) 

v_mov_b32 v14, 192 // (op 2778) 
v_mul_lo_u32 v22, s17, v14 // (op 2778) 

v_add_u32 v22, v22, v21 // (op 2778) 

v_mul_lo_u32 v14, v22, s32 // (op 2778) 

v_add_u32 v21, v19, v14 // (op 2778) 


s_mul_i32 s1, s32, s9 // (op 2780) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2780) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2780) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2780) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2780) most significant: sum
s_add_u32 s1, s1, s7 // (op 2780) most significant: sum


s_mov_b32 s60, s34 // (op 2780) reg expression
s_mov_b32 s61, s35 // (op 2780) reg expression
s_mov_b32 s62, s0 // (op 2780) reg expression
s_mov_b32 s63, 131072 // (op 2780) call()




buffer_load_ubyte v14, v15, s[56:59], 0 offen offset: 0 // (op 2326) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 2330) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 2330) 


v_mov_b32 v19, v16 // (op 2330) 
buffer_load_dwordx4 v[24:27], v19, s[52:55], 0 offen // (op 2330) Load value
v_add_u32 v19, v19, v22 // (op 2330) 
buffer_load_dwordx4 v[28:31], v19, s[52:55], 0 offen // (op 2330) Load value
v_add_u32 v19, v19, v22 // (op 2330) 
buffer_load_dwordx4 v[32:35], v19, s[52:55], 0 offen // (op 2330) Load value



v_mov_b64 v[22:23], s[32:33] // (op 2333) convert

v_mov_b32 v19, v20 // (op 2333) 
buffer_load_ubyte v36, v19, s[60:63], 0 offen offset: 0 // (op 2333) Load value
v_add_u32 v19, v19, v22 // (op 2333) 
buffer_load_ubyte v37, v19, s[60:63], 0 offen offset: 0 // (op 2333) Load value
v_add_u32 v19, v19, v22 // (op 2333) 
buffer_load_ubyte v38, v19, s[60:63], 0 offen offset: 0 // (op 2333) Load value





buffer_load_dwordx2 v[22:23], v12, s[48:51], 0 offen // (op 2336) Load value



buffer_load_ubyte v19, v15, s[56:59], 0 offen offset:4 // (op 2340) Load value


s_lshl_b64 s[0:1], s[24:25], 2 // (op 2344) 
v_lshrrev_b64 v[40:41], 3, s[0:1] // (op 2344) 


v_mov_b32 v15, v16 // (op 2344) 
buffer_load_dwordx4 v[42:45], v15, s[52:55], 0 offen offset:64 // (op 2344) Load value
v_add_u32 v15, v15, v40 // (op 2344) 
buffer_load_dwordx4 v[46:49], v15, s[52:55], 0 offen offset:64 // (op 2344) Load value
v_add_u32 v15, v15, v40 // (op 2344) 
buffer_load_dwordx4 v[50:53], v15, s[52:55], 0 offen offset:64 // (op 2344) Load value




v_mov_b64 v[40:41], s[32:33] // (op 2347) convert

v_mov_b32 v39, v20 // (op 2347) 
buffer_load_ubyte v54, v39, s[60:63], 0 offen offset:4 // (op 2347) Load value
v_add_u32 v39, v39, v40 // (op 2347) 
buffer_load_ubyte v55, v39, s[60:63], 0 offen offset:4 // (op 2347) Load value
v_add_u32 v39, v39, v40 // (op 2347) 
buffer_load_ubyte v56, v39, s[60:63], 0 offen offset:4 // (op 2347) Load value






buffer_load_dwordx2 v[40:41], v12, s[48:51], 0 offen offset:64 // (op 2350) Load value

v_and_b32 v12, -4, v1 // (op 2898) 
v_sub_u32 v20, v1, v12 // (op 2898) 

v_lshlrev_b32 v12, 5, v20 // (op 2898) 

v_lshrrev_b32 v20, 2, v1 // (op 2898) 
v_mul_lo_u32 v39, v20, 3 // (op 2898) 

v_lshl_add_u32 v20, v39, 7, v12 // (op 2898) 


v_lshlrev_b32 v12, 2, v20 // (op 2898) 

v_lshrrev_b32 v20, 3, v12 // (op 2898) 




s_waitcnt vmcnt(14)
ds_write_b128 v20, v[24:27] // (op 2352) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v20, v[28:31] offset:64 // (op 2352) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v20, v[32:35] offset:128 // (op 2352) Store local data 

v_and_b32 v12, -4, v1 // (op 2956) 
v_sub_u32 v20, v1, v12 // (op 2956) 

v_lshrrev_b32 v12, 2, v1 // (op 2956) 
v_mul_lo_u32 v39, v12, 3 // (op 2956) 

v_lshl_add_u32 v12, v39, 2, v20 // (op 2956) 





s_waitcnt vmcnt(11)
ds_write_b8 v12, v36 offset:12288 // (op 2353) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v12, v37 offset:12292 // (op 2353) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v12, v38 offset:12296 // (op 2353) Store local data 

v_and_b32 v12, -8, v1 // (op 2845) 
v_sub_u32 v20, v1, v12 // (op 2845) 

v_lshlrev_b32 v12, 4, v20 // (op 2845) 

v_lshrrev_b32 v20, 3, v1 // (op 2845) 
v_lshl_add_u32 v39, v20, 7, v12 // (op 2845) 


v_lshlrev_b32 v12, 2, v39 // (op 2845) 

v_lshrrev_b32 v20, 3, v12 // (op 2845) 




s_waitcnt vmcnt(8)
ds_write_b64 v20, v[22:23] offset:13056 // (op 2354) Store local data 

v_and_b32 v12, -8, v1 // (op 2816) 
v_sub_u32 v20, v1, v12 // (op 2816) 

v_lshlrev_b32 v12, 4, v20 // (op 2816) 

v_lshrrev_b32 v20, 3, v1 // (op 2816) 
v_lshl_add_u32 v39, v20, 7, v12 // (op 2816) 


v_lshlrev_b32 v12, 2, v39 // (op 2816) 

v_lshrrev_b32 v20, 3, v12 // (op 2816) 

v_and_b32 v12, -4, v1 // (op 2869) 
v_sub_u32 v39, v1, v12 // (op 2869) 

v_lshlrev_b32 v12, 5, v39 // (op 2869) 

v_lshrrev_b32 v39, 2, v1 // (op 2869) 
v_mul_lo_u32 v15, v39, 3 // (op 2869) 

v_lshl_add_u32 v16, v15, 7, v12 // (op 2869) 


v_lshlrev_b32 v12, 2, v16 // (op 2869) 

v_lshrrev_b32 v39, 3, v12 // (op 2869) 

v_and_b32 v12, -4, v1 // (op 2927) 
v_sub_u32 v15, v1, v12 // (op 2927) 

v_lshrrev_b32 v12, 2, v1 // (op 2927) 
v_mul_lo_u32 v16, v12, 3 // (op 2927) 

v_lshl_add_u32 v12, v16, 2, v15 // (op 2927) 


v_and_b32 v15, -8, v1 // (op 3003) 
v_sub_u32 v16, v1, v15 // (op 3003) 

v_lshlrev_b32 v15, 4, v16 // (op 3003) 

v_lshrrev_b32 v16, 3, v1 // (op 3003) 
v_lshl_add_u32 v57, v16, 7, v15 // (op 3003) 


v_lshlrev_b32 v15, 2, v57 // (op 3003) 

v_lshrrev_b32 v16, 3, v15 // (op 3003) 

v_and_b32 v15, -4, v1 // (op 3027) 
v_sub_u32 v57, v1, v15 // (op 3027) 

v_lshlrev_b32 v15, 5, v57 // (op 3027) 

v_lshrrev_b32 v57, 2, v1 // (op 3027) 
v_mul_lo_u32 v58, v57, 3 // (op 3027) 

v_lshl_add_u32 v57, v58, 7, v15 // (op 3027) 


v_lshlrev_b32 v15, 2, v57 // (op 3027) 

v_lshrrev_b32 v57, 3, v15 // (op 3027) 

v_and_b32 v15, -4, v1 // (op 3056) 
v_sub_u32 v58, v1, v15 // (op 3056) 

v_lshrrev_b32 v15, 2, v1 // (op 3056) 
v_mul_lo_u32 v59, v15, 3 // (op 3056) 

v_lshl_add_u32 v15, v59, 2, v58 // (op 3056) 


s_mov_b32 s0, 0 // (op 932) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 931) 
s_mov_b32 s1, s6 // (op 931) convert

s_and_b32 s6, s1, 4294967294 // (op 931) 

s_cmp_lt_u32 s0, s6 // (op 931) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopBottom_KLoop_931 // (op 931) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopBottom_KLoop_931 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopTop_KLoop_931:
s_add_u32 s0, s0, 2 // (op 933) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3126) 



ds_read_b128 v[58:61], v3 offset:13056 // (op 65) Load local data 
s_barrier  // (op 3139) 



ds_read_b128 v[62:65], v5 // (op 70) Load local data 
s_barrier  // (op 3153) 



ds_read_u8 v66, v0 offset:12288 // (op 79) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:61], v[62:65], a[0:3], v14, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 82) 
ds_read_b128 v[68:71], v5 offset:2048 // (op 339) Load local data 





ds_read_u8 v62, v0 offset:12416 // (op 342) Load local data 



ds_read_b128 v[64:67], v5 offset:4096 // (op 495) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:61], v[68:71], a[4:7], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 343) 





ds_read_u8 v62, v0 offset:12544 // (op 498) Load local data 



ds_read_b128 v[68:71], v5 offset:6144 // (op 651) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[58:61], v[64:67], a[8:11], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 499) 





ds_read_u8 v62, v0 offset:12672 // (op 654) Load local data 



ds_read_b128 v[64:67], v5 offset:8192 // (op 807) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:61], v[68:71], a[12:15], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 655) 





ds_read_u8 v62, v0 offset:12800 // (op 810) Load local data 



ds_read_b128 v[68:71], v5 offset:10240 // (op 963) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:61], v[64:67], a[16:19], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 811) 





ds_read_u8 v62, v0 offset:12928 // (op 966) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:61], v[68:71], a[20:23], v14, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 967) 





v_mov_b32 v58, v18 // (op 75) 

buffer_load_ubyte v14, v58, s[56:59], 0 offen offset:8 // (op 75) Load value

s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[60:61], 3, s[6:7] // (op 10) 


v_mov_b32 v59, v17 // (op 10) 
v_mov_b32 v62, v59 // (op 10) 
buffer_load_dwordx4 v[24:27], v62, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v62, v62, v60 // (op 10) 
buffer_load_dwordx4 v[28:31], v62, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v62, v62, v60 // (op 10) 
buffer_load_dwordx4 v[32:35], v62, s[52:55], 0 offen offset:128 // (op 10) Load value



v_mov_b64 v[60:61], s[32:33] // (op 21) convert

v_mov_b32 v62, v21 // (op 21) 
v_mov_b32 v63, v62 // (op 21) 
buffer_load_ubyte v36, v63, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v63, v63, v60 // (op 21) 
buffer_load_ubyte v37, v63, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v63, v63, v60 // (op 21) 
buffer_load_ubyte v38, v63, s[60:63], 0 offen offset:8 // (op 21) Load value




v_mov_b32 v60, v13 // (op 4) 

buffer_load_dwordx2 v[22:23], v60, s[48:51], 0 offen offset:128 // (op 4) Load value
s_barrier  // (op 3176) 



s_waitcnt vmcnt(14)
ds_write_b128 v57, v[42:45] offset:15104 // (op 235) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v57, v[46:49] offset:15168 // (op 235) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v57, v[50:53] offset:15232 // (op 235) Store local data 
s_barrier  // (op 3182) 



s_waitcnt vmcnt(11)
ds_write_b8 v15, v54 offset:27392 // (op 236) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v15, v55 offset:27396 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v15, v56 offset:27400 // (op 236) Store local data 
s_barrier  // (op 3170) 



s_waitcnt vmcnt(8)
ds_write_b64 v16, v[40:41] offset:28160 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3167) 



ds_read_b128 v[64:67], v4 offset:28160 // (op 227) Load local data 
s_barrier  // (op 3173) 
s_barrier  // (op 3219) 



ds_read_b128 v[68:71], v8 offset:15104 // (op 229) Load local data 
s_barrier  // (op 3179) 



ds_read_u8 v61, v7 offset:27392 // (op 233) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:67], v[68:71], a[0:3], v19, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 234) 
ds_read_b128 v[72:75], v8 offset:17152 // (op 348) Load local data 





ds_read_u8 v61, v7 offset:27520 // (op 351) Load local data 



ds_read_b128 v[68:71], v8 offset:19200 // (op 504) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:67], v[72:75], a[4:7], v19, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 344) 





ds_read_u8 v61, v7 offset:27648 // (op 507) Load local data 



ds_read_b128 v[72:75], v8 offset:21248 // (op 660) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:67], v[68:71], a[8:11], v19, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 500) 





ds_read_u8 v61, v7 offset:27776 // (op 663) Load local data 



ds_read_b128 v[68:71], v8 offset:23296 // (op 816) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:67], v[72:75], a[12:15], v19, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 656) 





ds_read_u8 v61, v7 offset:27904 // (op 819) Load local data 



ds_read_b128 v[72:75], v8 offset:25344 // (op 972) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[64:67], v[68:71], a[16:19], v19, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 812) 





s_barrier  // (op 3277) 
ds_read_u8 v61, v7 offset:28032 // (op 975) Load local data 
s_barrier  // (op 3281) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:67], v[72:75], a[20:23], v19, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 968) 






buffer_load_ubyte v19, v58, s[56:59], 0 offen offset:12 // (op 231) Load value


v_add_u32 v18, 8, v18 // (op 2706) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 237) 
v_lshrrev_b64 v[64:65], 3, s[6:7] // (op 237) 


v_mov_b32 v58, v59 // (op 237) 
buffer_load_dwordx4 v[42:45], v58, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v58, v58, v64 // (op 237) 
buffer_load_dwordx4 v[46:49], v58, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v58, v58, v64 // (op 237) 
buffer_load_dwordx4 v[50:53], v58, s[52:55], 0 offen offset:192 // (op 237) Load value




v_add_u32 v17, 128, v17 // (op 2642) 
v_mov_b64 v[58:59], s[32:33] // (op 238) convert

v_mov_b32 v61, v62 // (op 238) 
buffer_load_ubyte v54, v61, s[60:63], 0 offen offset:12 // (op 238) Load value
v_add_u32 v61, v61, v58 // (op 238) 
buffer_load_ubyte v55, v61, s[60:63], 0 offen offset:12 // (op 238) Load value
v_add_u32 v61, v61, v58 // (op 238) 
buffer_load_ubyte v56, v61, s[60:63], 0 offen offset:12 // (op 238) Load value






v_add_u32 v21, 8, v21 // (op 2770) 
buffer_load_dwordx2 v[40:41], v60, s[48:51], 0 offen offset:192 // (op 224) Load value

s_barrier  // (op 3142) 
v_add_u32 v13, 128, v13 // (op 2578) 



s_waitcnt vmcnt(14)
ds_write_b128 v39, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v39, v[28:31] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v39, v[32:35] offset:128 // (op 13) Store local data 
s_barrier  // (op 3156) 



s_waitcnt vmcnt(11)
ds_write_b8 v12, v36 offset:12288 // (op 26) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v12, v37 offset:12292 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v12, v38 offset:12296 // (op 26) Store local data 
s_barrier  // (op 3129) 



s_waitcnt vmcnt(8)
ds_write_b64 v20, v[22:23] offset:13056 // (op 6) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 931) 
s_mov_b32 s1, s6 // (op 931) convert

s_and_b32 s6, s1, 4294967294 // (op 931) 

s_cmp_lt_u32 s0, s6 // (op 931) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopTop_KLoop_931 // (op 931) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopTop_KLoop_931 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopBottom_KLoop_931:















v_lshrrev_b32 v12, 3, v1 // (op 2536) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2536) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 2536) 

s_lshr_b32 s6, s0, 1 // (op 2536) 

v_and_b32 v12, -8, v1 // (op 2536) 
v_sub_u32 v14, v1, v12 // (op 2536) 

v_lshlrev_b32 v12, 4, v14 // (op 2536) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 2536) 


v_mov_b32 v21, 0 // (op 2536) convert
v_mov_b32 v20, v14 // (op 2536) convert

v_mul_lo_u32 v15, v13, s13 // (op 2536) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 2536) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 2536) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 2536) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v20 // (op 2536) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 2536) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 2536) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 2536) 

v_mov_b32 v12, v14 // (op 2536) convert

s_mul_i32 s1, s12, s23 // (op 2538) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2538) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2538) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2538) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2538) most significant: sum
s_add_u32 s1, s1, s7 // (op 2538) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2538) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2538) 

s_mov_b32 s48, s20 // (op 2538) reg expression
s_mov_b32 s49, s21 // (op 2538) reg expression
s_mov_b32 s50, s0 // (op 2538) reg expression
s_mov_b32 s51, 131072 // (op 2538) call()

v_and_b32 v13, -4, v1 // (op 2608) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2608) 
v_sub_u32 v14, v1, v13 // (op 2608) 

s_lshr_b32 s6, s0, 1 // (op 2608) 

v_lshlrev_b32 v13, 5, v14 // (op 2608) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 2608) 


v_mov_b32 v21, 0 // (op 2608) convert
v_mov_b32 v20, v14 // (op 2608) convert

v_lshrrev_b32 v13, 2, v1 // (op 2608) 
v_mul_lo_u32 v14, v13, 3 // (op 2608) 

v_mov_b32 v13, 192 // (op 2608) 
v_mul_lo_u32 v15, s17, v13 // (op 2608) 

v_add_u32 v15, v15, v14 // (op 2608) 

v_mul_lo_u32 v17, v15, s25 // (op 2608) most significant: low of low * high
v_mul_hi_u32 v18, v15, s24 // (op 2608) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 2608) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 2608) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v16 // (op 2608) least significant half
v_addc_co_u32 v15, s[0:1], v21, v17, s[0:1] // (op 2608) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 2608) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 2608) 

v_mov_b32 v18, v16 // (op 2608) convert

s_mul_i32 s1, s24, s9 // (op 2610) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 2610) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 2610) least significant
s_mul_hi_u32 s7, s24, s8 // (op 2610) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2610) most significant: sum
s_add_u32 s1, s1, s7 // (op 2610) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2610) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2610) 

s_mov_b32 s52, s26 // (op 2610) reg expression
s_mov_b32 s53, s27 // (op 2610) reg expression
s_mov_b32 s54, s0 // (op 2610) reg expression
s_mov_b32 s55, 131072 // (op 2610) call()

v_lshrrev_b32 v13, 7, v1 // (op 2672) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2672) 
v_and_b32 v14, 15, v1 // (op 2672) 
s_lshr_b32 s6, s0, 1 // (op 2672) 

v_lshl_add_u32 v15, v13, 4, v14 // (op 2672) 


v_lshl_add_u32 v13, s16, 5, v15 // (op 2672) 

v_and_b32 v14, 63, v1 // (op 2672) 
v_lshrrev_b32 v15, 4, v14 // (op 2672) 

v_lshl_add_u32 v14, s6, 3, v15 // (op 2672) 


v_mul_lo_u32 v15, v13, s28 // (op 2672) 
v_add_u32 v15, v15, v14 // (op 2672) 


s_mul_i32 s1, s28, s23 // (op 2674) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 2674) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 2674) least significant
s_mul_hi_u32 s7, s28, s22 // (op 2674) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2674) most significant: sum
s_add_u32 s1, s1, s7 // (op 2674) most significant: sum


s_mov_b32 s56, s30 // (op 2674) reg expression
s_mov_b32 s57, s31 // (op 2674) reg expression
s_mov_b32 s58, s0 // (op 2674) reg expression
s_mov_b32 s59, 131072 // (op 2674) call()

v_and_b32 v13, -4, v1 // (op 2736) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 2736) 
v_sub_u32 v14, v1, v13 // (op 2736) 

s_lshr_b32 s6, s0, 1 // (op 2736) 

v_lshrrev_b32 v13, 2, v1 // (op 2736) 
v_lshl_add_u32 v16, s6, 3, v14 // (op 2736) 


v_mul_lo_u32 v14, v13, 3 // (op 2736) 

v_mov_b32 v13, 192 // (op 2736) 
v_mul_lo_u32 v17, s17, v13 // (op 2736) 

v_add_u32 v17, v17, v14 // (op 2736) 

v_mul_lo_u32 v13, v17, s32 // (op 2736) 

v_add_u32 v14, v16, v13 // (op 2736) 


s_mul_i32 s1, s32, s9 // (op 2738) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 2738) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 2738) least significant
s_mul_hi_u32 s7, s32, s8 // (op 2738) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2738) most significant: sum
s_add_u32 s1, s1, s7 // (op 2738) most significant: sum


s_mov_b32 s60, s34 // (op 2738) reg expression
s_mov_b32 s61, s35 // (op 2738) reg expression
s_mov_b32 s62, s0 // (op 2738) reg expression
s_mov_b32 s63, 131072 // (op 2738) call()

v_and_b32 v13, -8, v1 // (op 2829) 
v_sub_u32 v16, v1, v13 // (op 2829) 

v_lshlrev_b32 v13, 4, v16 // (op 2829) 

v_lshrrev_b32 v16, 3, v1 // (op 2829) 
v_lshl_add_u32 v17, v16, 7, v13 // (op 2829) 


v_lshlrev_b32 v13, 2, v17 // (op 2829) 

v_lshrrev_b32 v16, 3, v13 // (op 2829) 

v_and_b32 v13, -4, v1 // (op 2882) 
v_sub_u32 v17, v1, v13 // (op 2882) 

v_lshlrev_b32 v13, 5, v17 // (op 2882) 

v_lshrrev_b32 v17, 2, v1 // (op 2882) 
v_mul_lo_u32 v20, v17, 3 // (op 2882) 

v_lshl_add_u32 v17, v20, 7, v13 // (op 2882) 


v_lshlrev_b32 v13, 2, v17 // (op 2882) 

v_lshrrev_b32 v17, 3, v13 // (op 2882) 

v_and_b32 v13, -4, v1 // (op 2940) 
v_sub_u32 v20, v1, v13 // (op 2940) 

v_lshrrev_b32 v13, 2, v1 // (op 2940) 
v_mul_lo_u32 v21, v13, 3 // (op 2940) 

v_lshl_add_u32 v13, v21, 2, v20 // (op 2940) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 937) 
s_and_b32 s6, s0, 4294967294 // (op 937) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 936) 
s_mov_b32 s7, s0 // (op 936) convert

s_cmp_lt_u32 s6, s7 // (op 936) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopBottom_KLoopTail_936 // (op 936) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopBottom_KLoopTail_936 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopTop_KLoopTail_936:


s_add_u32 s6, s6, 1 // (op 938) 
v_mov_b32 v19, v12 // (op 166) 

buffer_load_dwordx2 v[20:21], v19, s[48:51], 0 offen // (op 166) Load value

v_add_u32 v12, 64, v12 // (op 2526) 

s_barrier  // (op 3132) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 179) 



v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 179) 


v_mov_b32 v19, v18 // (op 179) 
v_mov_b32 v24, v19 // (op 179) 
buffer_load_dwordx4 v[26:29], v24, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v24, v24, v22 // (op 179) 
buffer_load_dwordx4 v[30:33], v24, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v24, v24, v22 // (op 179) 
s_waitcnt vmcnt(2)
ds_write_b64 v16, v[20:21] offset:13056 // (op 167) Store local data 
buffer_load_dwordx4 v[34:37], v24, s[52:55], 0 offen // (op 179) Load value




v_add_u32 v18, 64, v18 // (op 2600) 

s_barrier  // (op 3145) 
v_mov_b64 v[24:25], s[32:33] // (op 180) convert




v_mov_b32 v19, v14 // (op 180) 
v_mov_b32 v20, v19 // (op 180) 
buffer_load_ubyte v38, v20, s[60:63], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v20, v20, v24 // (op 180) 
buffer_load_ubyte v39, v20, s[60:63], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v20, v20, v24 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b128 v17, v[26:29] // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v17, v[30:33] offset:64 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[34:37] offset:128 // (op 177) Store local data 
buffer_load_ubyte v40, v20, s[60:63], 0 offen offset: 0 // (op 180) Load value




v_add_u32 v14, 4, v14 // (op 2728) 
s_barrier  // (op 3159) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3135) 






ds_read_b128 v[34:37], v3 offset:13056 // (op 169) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v13, v38 offset:12288 // (op 178) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3149) 
s_waitcnt vmcnt(1)
ds_write_b8 v13, v39 offset:12292 // (op 178) Store local data 



ds_read_b128 v[30:33], v5 // (op 171) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v13, v40 offset:12296 // (op 178) Store local data 



v_mov_b32 v19, v15 // (op 173) 

buffer_load_ubyte v20, v19, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v15, 4, v15 // (op 2664) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3163) 



ds_read_u8 v19, v0 offset:12288 // (op 175) Load local data 



ds_read_b128 v[26:29], v5 offset:2048 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[30:33], a[0:3], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 176) 





ds_read_u8 v19, v0 offset:12416 // (op 323) Load local data 



ds_read_b128 v[22:25], v5 offset:4096 // (op 476) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[26:29], a[4:7], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 324) 





ds_read_u8 v19, v0 offset:12544 // (op 479) Load local data 



ds_read_b128 v[26:29], v5 offset:6144 // (op 632) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[22:25], a[8:11], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 480) 





ds_read_u8 v19, v0 offset:12672 // (op 635) Load local data 



ds_read_b128 v[22:25], v5 offset:8192 // (op 788) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[26:29], a[12:15], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 636) 





ds_read_u8 v19, v0 offset:12800 // (op 791) Load local data 



ds_read_b128 v[26:29], v5 offset:10240 // (op 944) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[34:37], v[22:25], a[16:19], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 792) 





ds_read_u8 v19, v0 offset:12928 // (op 947) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:37], v[26:29], a[20:23], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 948) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 936) 
s_mov_b32 s7, s0 // (op 936) convert

s_cmp_lt_u32 s6, s7 // (op 936) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopTop_KLoopTail_936 // (op 936) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopTop_KLoopTail_936 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ForLoopBottom_KLoopTail_936:






















s_barrier  // (op 3265) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 52) Store value


v_accvgpr_read v12, a4 // (op 1216) 
v_accvgpr_read v13, a5 // (op 1216) 
v_accvgpr_read v14, a6 // (op 1216) 
v_accvgpr_read v15, a7 // (op 1216) 
v_mul_f32 v16, v11, v12 // (op 1216) 
v_mul_f32 v17, v11, v13 // (op 1216) 
v_mul_f32 v18, v11, v14 // (op 1216) 
v_mul_f32 v19, v11, v15 // (op 1216) 



s_lshl_b32 s0, s4, 7 // (op 1218) 
v_add_u32 v0, v9, s0 // (op 1218) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1218) Store value


v_accvgpr_read v12, a8 // (op 1228) 
v_accvgpr_read v13, a9 // (op 1228) 
v_accvgpr_read v14, a10 // (op 1228) 
v_accvgpr_read v15, a11 // (op 1228) 
v_mul_f32 v16, v11, v12 // (op 1228) 
v_mul_f32 v17, v11, v13 // (op 1228) 
v_mul_f32 v18, v11, v14 // (op 1228) 
v_mul_f32 v19, v11, v15 // (op 1228) 



s_lshl_b32 s0, s4, 8 // (op 1230) 
v_add_u32 v0, v9, s0 // (op 1230) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1230) Store value


v_accvgpr_read v12, a12 // (op 1240) 
v_accvgpr_read v13, a13 // (op 1240) 
v_accvgpr_read v14, a14 // (op 1240) 
v_accvgpr_read v15, a15 // (op 1240) 
v_mul_f32 v16, v11, v12 // (op 1240) 
v_mul_f32 v17, v11, v13 // (op 1240) 
v_mul_f32 v18, v11, v14 // (op 1240) 
v_mul_f32 v19, v11, v15 // (op 1240) 



s_lshl_b32 s0, s4, 7 // (op 1242) 
s_mul_i32 s1, 3, s0 // (op 1242) 

v_add_u32 v0, v9, s1 // (op 1242) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1242) Store value


v_accvgpr_read v12, a16 // (op 1252) 
v_accvgpr_read v13, a17 // (op 1252) 
v_accvgpr_read v14, a18 // (op 1252) 
v_accvgpr_read v15, a19 // (op 1252) 
v_mul_f32 v16, v11, v12 // (op 1252) 
v_mul_f32 v17, v11, v13 // (op 1252) 
v_mul_f32 v18, v11, v14 // (op 1252) 
v_mul_f32 v19, v11, v15 // (op 1252) 



s_lshl_b32 s0, s4, 9 // (op 1254) 
v_add_u32 v0, v9, s0 // (op 1254) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1254) Store value


v_accvgpr_read v12, a20 // (op 1264) 
v_accvgpr_read v13, a21 // (op 1264) 
v_accvgpr_read v14, a22 // (op 1264) 
v_accvgpr_read v15, a23 // (op 1264) 
v_mul_f32 v16, v11, v12 // (op 1264) 
v_mul_f32 v17, v11, v13 // (op 1264) 
v_mul_f32 v18, v11, v14 // (op 1264) 
v_mul_f32 v19, v11, v15 // (op 1264) 



s_lshl_b32 s0, s4, 7 // (op 1266) 
s_mul_i32 s1, 5, s0 // (op 1266) 

v_add_u32 v0, v9, s1 // (op 1266) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1266) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 135) Load value

v_accvgpr_read v16, a0 // (op 132) 
v_accvgpr_read v17, a1 // (op 132) 
v_accvgpr_read v18, a2 // (op 132) 
v_accvgpr_read v19, a3 // (op 132) 
v_mul_f32 v20, v11, v16 // (op 132) 
v_mul_f32 v21, v11, v17 // (op 132) 
v_mul_f32 v22, v11, v18 // (op 132) 
v_mul_f32 v23, v11, v19 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 136) 
v_mul_f32 v17, v10, v13 // (op 136) 
v_mul_f32 v18, v10, v14 // (op 136) 
v_mul_f32 v19, v10, v15 // (op 136) 

v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 
v_add_f32 v14, v18, v22 // (op 133) 
v_add_f32 v15, v19, v23 // (op 133) 




v_mov_b32 v0, v9 // (op 134) reg expression

buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 1308) 
v_add_u32 v0, v6, s0 // (op 1308) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1308) Load value

v_accvgpr_read v16, a4 // (op 1305) 
v_accvgpr_read v17, a5 // (op 1305) 
v_accvgpr_read v18, a6 // (op 1305) 
v_accvgpr_read v19, a7 // (op 1305) 
v_mul_f32 v20, v11, v16 // (op 1305) 
v_mul_f32 v21, v11, v17 // (op 1305) 
v_mul_f32 v22, v11, v18 // (op 1305) 
v_mul_f32 v23, v11, v19 // (op 1305) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1309) 
v_mul_f32 v17, v10, v13 // (op 1309) 
v_mul_f32 v18, v10, v14 // (op 1309) 
v_mul_f32 v19, v10, v15 // (op 1309) 

v_add_f32 v12, v16, v20 // (op 1306) 
v_add_f32 v13, v17, v21 // (op 1306) 
v_add_f32 v14, v18, v22 // (op 1306) 
v_add_f32 v15, v19, v23 // (op 1306) 




s_lshl_b32 s0, s4, 7 // (op 1307) 
v_add_u32 v0, v9, s0 // (op 1307) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1307) Store value




s_lshl_b32 s0, s2, 8 // (op 1320) 
v_add_u32 v0, v6, s0 // (op 1320) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1320) Load value

v_accvgpr_read v16, a8 // (op 1317) 
v_accvgpr_read v17, a9 // (op 1317) 
v_accvgpr_read v18, a10 // (op 1317) 
v_accvgpr_read v19, a11 // (op 1317) 
v_mul_f32 v20, v11, v16 // (op 1317) 
v_mul_f32 v21, v11, v17 // (op 1317) 
v_mul_f32 v22, v11, v18 // (op 1317) 
v_mul_f32 v23, v11, v19 // (op 1317) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1321) 
v_mul_f32 v17, v10, v13 // (op 1321) 
v_mul_f32 v18, v10, v14 // (op 1321) 
v_mul_f32 v19, v10, v15 // (op 1321) 

v_add_f32 v12, v16, v20 // (op 1318) 
v_add_f32 v13, v17, v21 // (op 1318) 
v_add_f32 v14, v18, v22 // (op 1318) 
v_add_f32 v15, v19, v23 // (op 1318) 




s_lshl_b32 s0, s4, 8 // (op 1319) 
v_add_u32 v0, v9, s0 // (op 1319) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1319) Store value




s_lshl_b32 s0, s2, 7 // (op 1332) 
s_mul_i32 s1, 3, s0 // (op 1332) 

v_add_u32 v0, v6, s1 // (op 1332) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1332) Load value

v_accvgpr_read v16, a12 // (op 1329) 
v_accvgpr_read v17, a13 // (op 1329) 
v_accvgpr_read v18, a14 // (op 1329) 
v_accvgpr_read v19, a15 // (op 1329) 
v_mul_f32 v20, v11, v16 // (op 1329) 
v_mul_f32 v21, v11, v17 // (op 1329) 
v_mul_f32 v22, v11, v18 // (op 1329) 
v_mul_f32 v23, v11, v19 // (op 1329) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1333) 
v_mul_f32 v17, v10, v13 // (op 1333) 
v_mul_f32 v18, v10, v14 // (op 1333) 
v_mul_f32 v19, v10, v15 // (op 1333) 

v_add_f32 v12, v16, v20 // (op 1330) 
v_add_f32 v13, v17, v21 // (op 1330) 
v_add_f32 v14, v18, v22 // (op 1330) 
v_add_f32 v15, v19, v23 // (op 1330) 




s_lshl_b32 s0, s4, 7 // (op 1331) 
s_mul_i32 s1, 3, s0 // (op 1331) 

v_add_u32 v0, v9, s1 // (op 1331) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1331) Store value




s_lshl_b32 s0, s2, 9 // (op 1344) 
v_add_u32 v0, v6, s0 // (op 1344) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1344) Load value

v_accvgpr_read v16, a16 // (op 1341) 
v_accvgpr_read v17, a17 // (op 1341) 
v_accvgpr_read v18, a18 // (op 1341) 
v_accvgpr_read v19, a19 // (op 1341) 
v_mul_f32 v20, v11, v16 // (op 1341) 
v_mul_f32 v21, v11, v17 // (op 1341) 
v_mul_f32 v22, v11, v18 // (op 1341) 
v_mul_f32 v23, v11, v19 // (op 1341) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1345) 
v_mul_f32 v17, v10, v13 // (op 1345) 
v_mul_f32 v18, v10, v14 // (op 1345) 
v_mul_f32 v19, v10, v15 // (op 1345) 

v_add_f32 v12, v16, v20 // (op 1342) 
v_add_f32 v13, v17, v21 // (op 1342) 
v_add_f32 v14, v18, v22 // (op 1342) 
v_add_f32 v15, v19, v23 // (op 1342) 




s_lshl_b32 s0, s4, 9 // (op 1343) 
v_add_u32 v0, v9, s0 // (op 1343) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1343) Store value




s_lshl_b32 s0, s2, 7 // (op 1356) 
s_mul_i32 s1, 5, s0 // (op 1356) 

v_add_u32 v0, v6, s1 // (op 1356) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1356) Load value

v_accvgpr_read v16, a20 // (op 1353) 
v_accvgpr_read v17, a21 // (op 1353) 
v_accvgpr_read v18, a22 // (op 1353) 
v_accvgpr_read v19, a23 // (op 1353) 
v_mul_f32 v20, v11, v16 // (op 1353) 
v_mul_f32 v21, v11, v17 // (op 1353) 
v_mul_f32 v22, v11, v18 // (op 1353) 
v_mul_f32 v23, v11, v19 // (op 1353) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1357) 
v_mul_f32 v17, v10, v13 // (op 1357) 
v_mul_f32 v18, v10, v14 // (op 1357) 
v_mul_f32 v19, v10, v15 // (op 1357) 

v_add_f32 v12, v16, v20 // (op 1354) 
v_add_f32 v13, v17, v21 // (op 1354) 
v_add_f32 v14, v18, v22 // (op 1354) 
v_add_f32 v15, v19, v23 // (op 1354) 




s_lshl_b32 s0, s4, 7 // (op 1355) 
s_mul_i32 s1, 5, s0 // (op 1355) 

v_add_u32 v0, v9, s1 // (op 1355) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1355) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_ConditionalBottom_0_beta_123:














s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 30208 // lds bytes
  .amdhsa_accum_offset 76
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
    DataTypeB: F4
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum } ]
    macrotile: [32, 192, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 6]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x192x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 30208
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 76
    .agpr_count: 24
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
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
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

