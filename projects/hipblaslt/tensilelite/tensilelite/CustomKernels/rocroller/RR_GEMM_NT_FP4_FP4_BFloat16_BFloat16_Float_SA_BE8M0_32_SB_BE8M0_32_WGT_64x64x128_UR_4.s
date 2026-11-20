// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
.p2align 8
.type RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4,@function
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 3575) 
v_and_b32 v3, 48, v1 // (op 3575) 
v_sub_u32 v4, v0, v3 // (op 3575) 

v_lshrrev_b32 v3, 4, v0 // (op 3575) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 3575) 


v_lshrrev_b32 v3, 7, v1 // (op 3575) 
v_lshlrev_b32 v4, 4, v3 // (op 3575) 

v_lshl_add_u32 v3, v0, 6, v4 // (op 3575) 


v_lshlrev_b32 v0, 2, v3 // (op 3575) 

v_lshrrev_b32 v3, 3, v0 // (op 3575) 

v_and_b32 v0, 63, v1 // (op 3633) 
v_and_b32 v4, 48, v1 // (op 3633) 
v_sub_u32 v5, v0, v4 // (op 3633) 

v_lshrrev_b32 v4, 4, v0 // (op 3633) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 3633) 


v_lshrrev_b32 v4, 6, v1 // (op 3633) 
v_and_b32 v5, 1, v4 // (op 3633) 

v_lshlrev_b32 v4, 4, v5 // (op 3633) 

v_lshl_add_u32 v5, v0, 6, v4 // (op 3633) 


v_lshlrev_b32 v0, 2, v5 // (op 3633) 

v_lshrrev_b32 v4, 3, v0 // (op 3633) 

v_lshrrev_b32 v0, 7, v1 // (op 3691) 
v_and_b32 v5, 15, v1 // (op 3691) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 3691) 


v_and_b32 v0, 63, v1 // (op 3691) 
v_lshrrev_b32 v5, 4, v0 // (op 3691) 

v_lshl_add_u32 v0, v5, 6, v6 // (op 3691) 


v_lshrrev_b32 v5, 6, v1 // (op 3749) 
v_and_b32 v6, 1, v5 // (op 3749) 

v_and_b32 v5, 15, v1 // (op 3749) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 3749) 


v_and_b32 v5, 63, v1 // (op 3749) 
v_lshrrev_b32 v6, 4, v5 // (op 3749) 

v_lshl_add_u32 v5, v6, 6, v7 // (op 3749) 


v_and_b32 v6, 63, v1 // (op 3801) 
v_and_b32 v7, 48, v1 // (op 3801) 
v_sub_u32 v8, v6, v7 // (op 3801) 

v_lshrrev_b32 v7, 4, v6 // (op 3801) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 3801) 


v_lshrrev_b32 v7, 7, v1 // (op 3801) 
v_lshlrev_b32 v8, 4, v7 // (op 3801) 

v_lshl_add_u32 v7, v6, 6, v8 // (op 3801) 


v_lshlrev_b32 v6, 2, v7 // (op 3801) 

v_lshrrev_b32 v7, 3, v6 // (op 3801) 

v_and_b32 v6, 63, v1 // (op 3830) 
v_and_b32 v8, 48, v1 // (op 3830) 
v_sub_u32 v9, v6, v8 // (op 3830) 

v_lshrrev_b32 v8, 4, v6 // (op 3830) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 3830) 


v_lshrrev_b32 v8, 6, v1 // (op 3830) 
v_and_b32 v9, 1, v8 // (op 3830) 

v_lshlrev_b32 v8, 4, v9 // (op 3830) 

v_lshl_add_u32 v9, v6, 6, v8 // (op 3830) 


v_lshlrev_b32 v6, 2, v9 // (op 3830) 

v_lshrrev_b32 v8, 3, v6 // (op 3830) 

v_lshrrev_b32 v6, 7, v1 // (op 3859) 
v_and_b32 v9, 15, v1 // (op 3859) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 3859) 


v_and_b32 v6, 63, v1 // (op 3859) 
v_lshrrev_b32 v9, 4, v6 // (op 3859) 

v_lshl_add_u32 v6, v9, 6, v10 // (op 3859) 


v_lshrrev_b32 v9, 6, v1 // (op 3888) 
v_and_b32 v10, 1, v9 // (op 3888) 

v_and_b32 v9, 15, v1 // (op 3888) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 3888) 


v_and_b32 v9, 63, v1 // (op 3888) 
v_lshrrev_b32 v10, 4, v9 // (op 3888) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 3888) 


v_and_b32 v10, 63, v1 // (op 3917) 
v_and_b32 v11, 48, v1 // (op 3917) 
v_sub_u32 v12, v10, v11 // (op 3917) 

v_lshrrev_b32 v11, 4, v10 // (op 3917) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 3917) 


v_lshrrev_b32 v11, 7, v1 // (op 3917) 
v_lshlrev_b32 v12, 4, v11 // (op 3917) 

v_lshl_add_u32 v11, v10, 6, v12 // (op 3917) 


v_lshlrev_b32 v10, 2, v11 // (op 3917) 

v_lshrrev_b32 v11, 3, v10 // (op 3917) 

v_and_b32 v10, 63, v1 // (op 3946) 
v_and_b32 v12, 48, v1 // (op 3946) 
v_sub_u32 v13, v10, v12 // (op 3946) 

v_lshrrev_b32 v12, 4, v10 // (op 3946) 

v_lshl_add_u32 v10, v12, 5, v13 // (op 3946) 


v_lshrrev_b32 v12, 6, v1 // (op 3946) 
v_and_b32 v13, 1, v12 // (op 3946) 

v_lshlrev_b32 v12, 4, v13 // (op 3946) 

v_lshl_add_u32 v13, v10, 6, v12 // (op 3946) 


v_lshlrev_b32 v10, 2, v13 // (op 3946) 

v_lshrrev_b32 v12, 3, v10 // (op 3946) 

v_lshrrev_b32 v10, 7, v1 // (op 3975) 
v_and_b32 v13, 15, v1 // (op 3975) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 3975) 


v_and_b32 v10, 63, v1 // (op 3975) 
v_lshrrev_b32 v13, 4, v10 // (op 3975) 

v_lshl_add_u32 v10, v13, 6, v14 // (op 3975) 


v_lshrrev_b32 v13, 6, v1 // (op 4004) 
v_and_b32 v14, 1, v13 // (op 4004) 

v_and_b32 v13, 15, v1 // (op 4004) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 4004) 


v_and_b32 v13, 63, v1 // (op 4004) 
v_lshrrev_b32 v14, 4, v13 // (op 4004) 

v_lshl_add_u32 v13, v14, 6, v15 // (op 4004) 


v_and_b32 v14, 63, v1 // (op 4033) 
v_and_b32 v15, 48, v1 // (op 4033) 
v_sub_u32 v16, v14, v15 // (op 4033) 

v_lshrrev_b32 v15, 4, v14 // (op 4033) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 4033) 


v_lshrrev_b32 v15, 7, v1 // (op 4033) 
v_lshlrev_b32 v16, 4, v15 // (op 4033) 

v_lshl_add_u32 v15, v14, 6, v16 // (op 4033) 


v_lshlrev_b32 v14, 2, v15 // (op 4033) 

v_lshrrev_b32 v15, 3, v14 // (op 4033) 

v_and_b32 v14, 63, v1 // (op 4062) 
v_and_b32 v16, 48, v1 // (op 4062) 
v_sub_u32 v17, v14, v16 // (op 4062) 

v_lshrrev_b32 v16, 4, v14 // (op 4062) 

v_lshl_add_u32 v14, v16, 5, v17 // (op 4062) 


v_lshrrev_b32 v16, 6, v1 // (op 4062) 
v_and_b32 v17, 1, v16 // (op 4062) 

v_lshlrev_b32 v16, 4, v17 // (op 4062) 

v_lshl_add_u32 v17, v14, 6, v16 // (op 4062) 


v_lshlrev_b32 v14, 2, v17 // (op 4062) 

v_lshrrev_b32 v16, 3, v14 // (op 4062) 

v_lshrrev_b32 v14, 7, v1 // (op 4091) 
v_and_b32 v17, 15, v1 // (op 4091) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 4091) 


v_and_b32 v14, 63, v1 // (op 4091) 
v_lshrrev_b32 v17, 4, v14 // (op 4091) 

v_lshl_add_u32 v14, v17, 6, v18 // (op 4091) 


v_lshrrev_b32 v17, 6, v1 // (op 4120) 
v_and_b32 v18, 1, v17 // (op 4120) 

v_and_b32 v17, 15, v1 // (op 4120) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 4120) 


v_and_b32 v17, 63, v1 // (op 4120) 
v_lshrrev_b32 v18, 4, v17 // (op 4120) 

v_lshl_add_u32 v17, v18, 6, v19 // (op 4120) 


v_lshrrev_b32 v18, 7, v1 // (op 3513) 
v_and_b32 v19, 63, v1 // (op 3513) 
v_lshrrev_b32 v20, 4, v19 // (op 3513) 
v_lshlrev_b32 v21, 2, v20 // (op 3513) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 3513) 


v_lshl_add_u32 v18, s16, 6, v20 // (op 3513) 

v_lshrrev_b32 v20, 2, v19 // (op 3513) 

v_and_b32 v19, 3, v20 // (op 3513) 

v_and_b32 v20, 3, v1 // (op 3513) 
v_lshl_add_u32 v21, v19, 2, v20 // (op 3513) 


v_lshrrev_b32 v19, 6, v1 // (op 3513) 
v_and_b32 v20, 1, v19 // (op 3513) 

v_lshl_add_u32 v19, v20, 4, v21 // (op 3513) 


v_lshl_add_u32 v20, s17, 6, v19 // (op 3513) 

v_mul_lo_u32 v19, v20, s2 // (op 3513) 

v_add_lshl_u32 v20, v18, v19, 1 // (op 3513) 


v_lshrrev_b32 v18, 7, v1 // (op 3770) 
v_and_b32 v19, 63, v1 // (op 3770) 
v_lshrrev_b32 v21, 4, v19 // (op 3770) 
s_mul_i32 s1, s2, s9 // (op 3515) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3515) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3515) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3515) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3515) most significant: sum
s_add_u32 s1, s1, s19 // (op 3515) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3515) 

v_lshlrev_b32 v22, 2, v21 // (op 3770) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 3770) 


v_lshl_add_u32 v18, s16, 6, v21 // (op 3770) 

v_lshrrev_b32 v21, 2, v19 // (op 3770) 

v_and_b32 v19, 3, v21 // (op 3770) 

v_and_b32 v21, 3, v1 // (op 3770) 
v_lshl_add_u32 v22, v19, 2, v21 // (op 3770) 


v_lshrrev_b32 v19, 6, v1 // (op 3770) 
v_and_b32 v21, 1, v19 // (op 3770) 

v_lshl_add_u32 v19, v21, 4, v22 // (op 3770) 


v_lshl_add_u32 v21, s17, 6, v19 // (op 3770) 

v_mul_lo_u32 v19, v21, s4 // (op 3770) 

v_add_lshl_u32 v21, v18, v19, 1 // (op 3770) 


s_mov_b32 s40, s6 // (op 3515) reg expression
s_mov_b32 s41, s7 // (op 3515) reg expression
s_mov_b32 s42, s18 // (op 3515) reg expression
s_mov_b32 s43, 131072 // (op 3515) call()

s_mul_i32 s1, s4, s9 // (op 3772) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 3772) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3772) least significant
s_mul_hi_u32 s7, s4, s8 // (op 3772) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3772) most significant: sum
s_add_u32 s1, s1, s7 // (op 3772) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 3772) 

s_mov_b32 s44, s10 // (op 3772) reg expression
s_mov_b32 s45, s11 // (op 3772) reg expression
s_mov_b32 s46, s6 // (op 3772) reg expression
s_mov_b32 s47, 131072 // (op 3772) call()



v_mov_b32 v18, s14 // (op 2) Move value
v_mov_b32 v19, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 486) call()
v_accvgpr_write a5, 0.00000 // (op 486) call()
v_accvgpr_write a6, 0.00000 // (op 486) call()
v_accvgpr_write a7, 0.00000 // (op 486) call()
v_accvgpr_write a8, 0.00000 // (op 852) call()
v_accvgpr_write a9, 0.00000 // (op 852) call()
v_accvgpr_write a10, 0.00000 // (op 852) call()
v_accvgpr_write a11, 0.00000 // (op 852) call()
v_accvgpr_write a12, 0.00000 // (op 1099) call()
v_accvgpr_write a13, 0.00000 // (op 1099) call()
v_accvgpr_write a14, 0.00000 // (op 1099) call()
v_accvgpr_write a15, 0.00000 // (op 1099) call()
v_and_b32 v22, -2, v1 // (op 3275) 
v_sub_u32 v23, v1, v22 // (op 3275) 

v_lshlrev_b32 v22, 5, v23 // (op 3275) 

v_lshl_add_u32 v23, s16, 6, v22 // (op 3275) 

v_mov_b32 v25, 0 // (op 3275) convert
v_mov_b32 v24, v23 // (op 3275) convert

v_lshrrev_b32 v22, 1, v1 // (op 3275) 
v_mul_lo_u32 v27, v22, s13 // (op 3275) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 3275) most significant: high of low * low
v_mul_lo_u32 v26, v22, s12 // (op 3275) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 3275) most significant: sum


v_add_co_u32 v22, s[0:1], v24, v26 // (op 3275) least significant half
v_addc_co_u32 v23, s[0:1], v25, v27, s[0:1] // (op 3275) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 3275) 


v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 3275) 

v_mov_b32 v22, v24 // (op 3275) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3277) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3277) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3277) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3277) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3277) most significant: sum
s_add_u32 s1, s1, s7 // (op 3277) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 3277) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 3277) 

s_mov_b32 s8, s20 // (op 3277) reg expression
s_mov_b32 s9, s21 // (op 3277) reg expression
s_mov_b32 s10, s0 // (op 3277) reg expression
s_mov_b32 s11, 131072 // (op 3277) call()

v_and_b32 v23, -2, v1 // (op 3294) 
v_sub_u32 v24, v1, v23 // (op 3294) 

v_lshlrev_b32 v23, 5, v24 // (op 3294) 

v_lshl_add_u32 v24, s16, 6, v23 // (op 3294) 

v_mov_b32 v27, 0 // (op 3294) convert
v_mov_b32 v26, v24 // (op 3294) convert

v_lshrrev_b32 v23, 1, v1 // (op 3294) 
v_mul_lo_u32 v25, v23, s13 // (op 3294) most significant: low of low * high
v_mul_hi_u32 v28, v23, s12 // (op 3294) most significant: high of low * low
v_mul_lo_u32 v24, v23, s12 // (op 3294) least significant: low of low * low
v_add_u32 v25, v25, v28 // (op 3294) most significant: sum


v_add_co_u32 v28, s[0:1], v26, v24 // (op 3294) least significant half
v_addc_co_u32 v29, s[0:1], v27, v25, s[0:1] // (op 3294) most significant half

v_lshlrev_b64 v[28:29], 2, v[28:29] // (op 3294) 


v_lshrrev_b64 v[26:27], 3, v[28:29] // (op 3294) 

v_mov_b32 v23, v26 // (op 3294) convert

s_mul_i32 s1, s12, s23 // (op 3296) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3296) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3296) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3296) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3296) most significant: sum
s_add_u32 s1, s1, s7 // (op 3296) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3296) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3296) 

s_mov_b32 s8, s20 // (op 3296) reg expression
s_mov_b32 s9, s21 // (op 3296) reg expression
s_mov_b32 s10, s0 // (op 3296) reg expression
s_mov_b32 s11, 131072 // (op 3296) call()

v_and_b32 v24, -2, v1 // (op 3339) 
v_sub_u32 v25, v1, v24 // (op 3339) 

v_lshlrev_b32 v24, 5, v25 // (op 3339) 

v_lshl_add_u32 v25, s17, 6, v24 // (op 3339) 

v_mov_b32 v27, 0 // (op 3339) convert
v_mov_b32 v26, v25 // (op 3339) convert

v_lshrrev_b32 v24, 1, v1 // (op 3339) 
v_mul_lo_u32 v29, v24, s25 // (op 3339) most significant: low of low * high
v_mul_hi_u32 v25, v24, s24 // (op 3339) most significant: high of low * low
v_mul_lo_u32 v28, v24, s24 // (op 3339) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 3339) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v26 // (op 3339) least significant half
v_addc_co_u32 v29, s[0:1], v29, v27, s[0:1] // (op 3339) most significant half



v_lshlrev_b64 v[24:25], 2, v[28:29] // (op 3339) 

v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 3339) 

v_mov_b32 v24, v26 // (op 3339) convert

s_mul_i32 s1, s24, s23 // (op 3341) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 3341) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 3341) least significant
s_mul_hi_u32 s7, s24, s22 // (op 3341) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3341) most significant: sum
s_add_u32 s1, s1, s7 // (op 3341) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3341) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3341) 

s_mov_b32 s48, s26 // (op 3341) reg expression
s_mov_b32 s49, s27 // (op 3341) reg expression
s_mov_b32 s50, s0 // (op 3341) reg expression
s_mov_b32 s51, 131072 // (op 3341) call()

v_and_b32 v25, -2, v1 // (op 3358) 
v_sub_u32 v26, v1, v25 // (op 3358) 

v_lshlrev_b32 v25, 5, v26 // (op 3358) 

v_lshl_add_u32 v26, s17, 6, v25 // (op 3358) 

v_mov_b32 v29, 0 // (op 3358) convert
v_mov_b32 v28, v26 // (op 3358) convert

v_lshrrev_b32 v25, 1, v1 // (op 3358) 
v_mul_lo_u32 v27, v25, s25 // (op 3358) most significant: low of low * high
v_mul_hi_u32 v30, v25, s24 // (op 3358) most significant: high of low * low
v_mul_lo_u32 v26, v25, s24 // (op 3358) least significant: low of low * low
v_add_u32 v27, v27, v30 // (op 3358) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v28 // (op 3358) least significant half
v_addc_co_u32 v27, s[0:1], v27, v29, s[0:1] // (op 3358) most significant half



v_lshlrev_b64 v[28:29], 2, v[26:27] // (op 3358) 

v_lshrrev_b64 v[26:27], 3, v[28:29] // (op 3358) 

v_mov_b32 v25, v26 // (op 3358) convert

s_mul_i32 s1, s24, s23 // (op 3360) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 3360) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 3360) least significant
s_mul_hi_u32 s7, s24, s22 // (op 3360) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3360) most significant: sum
s_add_u32 s1, s1, s7 // (op 3360) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3360) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3360) 

s_mov_b32 s48, s26 // (op 3360) reg expression
s_mov_b32 s49, s27 // (op 3360) reg expression
s_mov_b32 s50, s0 // (op 3360) reg expression
s_mov_b32 s51, 131072 // (op 3360) call()

v_and_b32 v26, -64, v1 // (op 3403) 
v_sub_u32 v27, v1, v26 // (op 3403) 

v_lshl_add_u32 v26, s16, 6, v27 // (op 3403) 

v_lshrrev_b32 v27, 6, v1 // (op 3403) 
v_mul_lo_u32 v28, v27, s28 // (op 3403) 

v_add_u32 v27, v26, v28 // (op 3403) 


s_mul_i32 s1, s28, s33 // (op 3405) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3405) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3405) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3405) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3405) most significant: sum
s_add_u32 s1, s1, s7 // (op 3405) most significant: sum


s_mov_b32 s52, s30 // (op 3405) reg expression
s_mov_b32 s53, s31 // (op 3405) reg expression
s_mov_b32 s54, s0 // (op 3405) reg expression
s_mov_b32 s55, 131072 // (op 3405) call()

v_and_b32 v26, -64, v1 // (op 3422) 
v_sub_u32 v28, v1, v26 // (op 3422) 

v_lshl_add_u32 v26, s16, 6, v28 // (op 3422) 

v_lshrrev_b32 v28, 6, v1 // (op 3422) 
v_mul_lo_u32 v29, v28, s28 // (op 3422) 

v_add_u32 v28, v26, v29 // (op 3422) 


s_mul_i32 s1, s28, s33 // (op 3424) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3424) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3424) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3424) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3424) most significant: sum
s_add_u32 s1, s1, s7 // (op 3424) most significant: sum


s_mov_b32 s52, s30 // (op 3424) reg expression
s_mov_b32 s53, s31 // (op 3424) reg expression
s_mov_b32 s54, s0 // (op 3424) reg expression
s_mov_b32 s55, 131072 // (op 3424) call()

v_and_b32 v26, -64, v1 // (op 3467) 
v_sub_u32 v29, v1, v26 // (op 3467) 

v_lshl_add_u32 v26, s17, 6, v29 // (op 3467) 

v_lshrrev_b32 v29, 6, v1 // (op 3467) 
v_mul_lo_u32 v30, v29, s34 // (op 3467) 
v_add_u32 v30, v30, v26 // (op 3467) 


s_mul_i32 s1, s34, s33 // (op 3469) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 3469) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 3469) least significant
s_mul_hi_u32 s7, s34, s32 // (op 3469) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3469) most significant: sum
s_add_u32 s1, s1, s7 // (op 3469) most significant: sum


s_mov_b32 s56, s36 // (op 3469) reg expression
s_mov_b32 s57, s37 // (op 3469) reg expression
s_mov_b32 s58, s0 // (op 3469) reg expression
s_mov_b32 s59, 131072 // (op 3469) call()

v_and_b32 v26, -64, v1 // (op 3486) 
v_sub_u32 v29, v1, v26 // (op 3486) 

v_lshl_add_u32 v26, s17, 6, v29 // (op 3486) 

v_lshrrev_b32 v29, 6, v1 // (op 3486) 
v_mul_lo_u32 v31, v29, s34 // (op 3486) 
v_add_u32 v31, v31, v26 // (op 3486) 


s_mul_i32 s1, s34, s33 // (op 3488) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 3488) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 3488) least significant
s_mul_hi_u32 s7, s34, s32 // (op 3488) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3488) most significant: sum
s_add_u32 s1, s1, s7 // (op 3488) most significant: sum


s_mov_b32 s56, s36 // (op 3488) reg expression
s_mov_b32 s57, s37 // (op 3488) reg expression
s_mov_b32 s58, s0 // (op 3488) reg expression
s_mov_b32 s59, 131072 // (op 3488) call()



v_mov_b32 v26, v24 // (op 2987) reg expression

buffer_load_dwordx4 v[32:35], v26, s[48:51], 0 offen // (op 2987) Load value



v_mov_b32 v26, v30 // (op 2990) reg expression

buffer_load_ubyte v29, v26, s[56:59], 0 offen offset: 0 // (op 2990) Load value



v_mov_b32 v26, v22 // (op 2993) reg expression

buffer_load_dwordx4 v[36:39], v26, s[8:11], 0 offen // (op 2993) Load value



v_mov_b32 v26, v27 // (op 2996) reg expression

buffer_load_ubyte v40, v26, s[52:55], 0 offen offset: 0 // (op 2996) Load value



s_lshl_b64 s[0:1], s[24:25], 7 // (op 2999) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 2999) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 2999) 

v_add_u32 v26, v24, s0 // (op 2999) 


buffer_load_dwordx4 v[42:45], v26, s[48:51], 0 offen // (op 2999) Load value



s_lshl_b32 s0, s34, 2 // (op 3002) 
v_add_u32 v26, v30, s0 // (op 3002) 


buffer_load_ubyte v41, v26, s[56:59], 0 offen offset: 0 // (op 3002) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 3005) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 3005) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 3005) 

v_add_u32 v26, v22, s0 // (op 3005) 


buffer_load_dwordx4 v[46:49], v26, s[8:11], 0 offen // (op 3005) Load value



s_lshl_b32 s0, s28, 2 // (op 3008) 
v_add_u32 v26, v27, s0 // (op 3008) 


buffer_load_ubyte v50, v26, s[52:55], 0 offen offset: 0 // (op 3008) Load value



s_lshl_b64 s[0:1], s[24:25], 7 // (op 3011) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 3011) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 3011) 

s_lshl_b32 s6, s0, 1 // (op 3011) 

v_add_u32 v26, v24, s6 // (op 3011) 


buffer_load_dwordx4 v[52:55], v26, s[48:51], 0 offen // (op 3011) Load value



s_lshl_b32 s0, s34, 3 // (op 3014) 
v_add_u32 v26, v30, s0 // (op 3014) 


buffer_load_ubyte v51, v26, s[56:59], 0 offen offset: 0 // (op 3014) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 3017) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 3017) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 3017) 

s_lshl_b32 s6, s0, 1 // (op 3017) 

v_add_u32 v26, v22, s6 // (op 3017) 


buffer_load_dwordx4 v[56:59], v26, s[8:11], 0 offen // (op 3017) Load value



s_lshl_b32 s0, s28, 3 // (op 3020) 
v_add_u32 v26, v27, s0 // (op 3020) 


buffer_load_ubyte v60, v26, s[52:55], 0 offen offset: 0 // (op 3020) Load value



s_lshl_b64 s[0:1], s[24:25], 7 // (op 3023) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 3023) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 3023) 

s_mul_i32 s6, 3, s0 // (op 3023) 

v_add_u32 v26, v24, s6 // (op 3023) 


buffer_load_dwordx4 v[62:65], v26, s[48:51], 0 offen // (op 3023) Load value




s_lshl_b32 s0, s34, 2 // (op 3026) 
s_mul_i32 s1, 3, s0 // (op 3026) 

v_add_u32 v24, v30, s1 // (op 3026) 


buffer_load_ubyte v26, v24, s[56:59], 0 offen offset: 0 // (op 3026) Load value




s_lshl_b64 s[0:1], s[12:13], 7 // (op 3029) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 3029) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 3029) 

s_mul_i32 s6, 3, s0 // (op 3029) 

v_add_u32 v24, v22, s6 // (op 3029) 


buffer_load_dwordx4 v[66:69], v24, s[8:11], 0 offen // (op 3029) Load value




s_lshl_b32 s0, s28, 2 // (op 3032) 
s_mul_i32 s1, 3, s0 // (op 3032) 

v_add_u32 v22, v27, s1 // (op 3032) 


buffer_load_ubyte v24, v22, s[52:55], 0 offen offset: 0 // (op 3032) Load value


v_and_b32 v22, -2, v1 // (op 3616) 
v_sub_u32 v27, v1, v22 // (op 3616) 

v_lshlrev_b32 v22, 5, v27 // (op 3616) 

v_lshrrev_b32 v27, 1, v1 // (op 3616) 
v_lshl_add_u32 v30, v27, 6, v22 // (op 3616) 


v_lshlrev_b32 v22, 2, v30 // (op 3616) 

v_lshrrev_b32 v27, 3, v22 // (op 3616) 




s_waitcnt vmcnt(15)
ds_write_b128 v27, v[32:35] // (op 3034) Store local data 

v_and_b32 v22, -64, v1 // (op 3732) 
v_sub_u32 v27, v1, v22 // (op 3732) 

v_and_b32 v22, 4294967232, v1 // (op 3732) 
v_add_u32 v30, v22, v27 // (op 3732) 





s_waitcnt vmcnt(14)
ds_write_b8 v30, v29 offset:4096 // (op 3035) Store local data 

v_and_b32 v22, -2, v1 // (op 3558) 
v_sub_u32 v27, v1, v22 // (op 3558) 

v_lshlrev_b32 v22, 5, v27 // (op 3558) 

v_lshrrev_b32 v27, 1, v1 // (op 3558) 
v_lshl_add_u32 v30, v27, 6, v22 // (op 3558) 


v_lshlrev_b32 v22, 2, v30 // (op 3558) 

v_lshrrev_b32 v27, 3, v22 // (op 3558) 




s_waitcnt vmcnt(13)
ds_write_b128 v27, v[36:39] offset:4352 // (op 3036) Store local data 

v_and_b32 v22, -64, v1 // (op 3674) 
v_sub_u32 v27, v1, v22 // (op 3674) 

v_and_b32 v22, 4294967232, v1 // (op 3674) 
v_add_u32 v30, v22, v27 // (op 3674) 





s_waitcnt vmcnt(12)
ds_write_b8 v30, v40 offset:8448 // (op 3037) Store local data 

v_and_b32 v22, -2, v1 // (op 3529) 
v_sub_u32 v27, v1, v22 // (op 3529) 

v_lshlrev_b32 v22, 5, v27 // (op 3529) 

v_lshrrev_b32 v27, 1, v1 // (op 3529) 
v_lshl_add_u32 v30, v27, 6, v22 // (op 3529) 


v_lshlrev_b32 v22, 2, v30 // (op 3529) 

v_lshrrev_b32 v27, 3, v22 // (op 3529) 

v_and_b32 v22, -2, v1 // (op 3587) 
v_sub_u32 v30, v1, v22 // (op 3587) 

v_lshlrev_b32 v22, 5, v30 // (op 3587) 

v_lshrrev_b32 v30, 1, v1 // (op 3587) 
v_lshl_add_u32 v61, v30, 6, v22 // (op 3587) 


v_lshlrev_b32 v22, 2, v61 // (op 3587) 

v_lshrrev_b32 v30, 3, v22 // (op 3587) 

v_and_b32 v22, -64, v1 // (op 3645) 
v_sub_u32 v61, v1, v22 // (op 3645) 

v_and_b32 v22, 4294967232, v1 // (op 3645) 
v_add_u32 v70, v22, v61 // (op 3645) 


v_and_b32 v22, -64, v1 // (op 3703) 
v_sub_u32 v61, v1, v22 // (op 3703) 

v_and_b32 v22, 4294967232, v1 // (op 3703) 
v_add_u32 v71, v22, v61 // (op 3703) 


v_and_b32 v22, -2, v1 // (op 3784) 
v_sub_u32 v61, v1, v22 // (op 3784) 

v_lshlrev_b32 v22, 5, v61 // (op 3784) 

v_lshrrev_b32 v61, 1, v1 // (op 3784) 
v_lshl_add_u32 v72, v61, 6, v22 // (op 3784) 


v_lshlrev_b32 v22, 2, v72 // (op 3784) 

v_lshrrev_b32 v61, 3, v22 // (op 3784) 

v_and_b32 v22, -2, v1 // (op 3813) 
v_sub_u32 v72, v1, v22 // (op 3813) 

v_lshlrev_b32 v22, 5, v72 // (op 3813) 

v_lshrrev_b32 v72, 1, v1 // (op 3813) 
v_lshl_add_u32 v73, v72, 6, v22 // (op 3813) 


v_lshlrev_b32 v22, 2, v73 // (op 3813) 

v_lshrrev_b32 v72, 3, v22 // (op 3813) 

v_and_b32 v22, -64, v1 // (op 3842) 
v_sub_u32 v73, v1, v22 // (op 3842) 

v_and_b32 v22, 4294967232, v1 // (op 3842) 
v_add_u32 v74, v22, v73 // (op 3842) 


v_and_b32 v22, -64, v1 // (op 3871) 
v_sub_u32 v73, v1, v22 // (op 3871) 

v_and_b32 v22, 4294967232, v1 // (op 3871) 
v_add_u32 v75, v22, v73 // (op 3871) 


v_and_b32 v22, -2, v1 // (op 3900) 
v_sub_u32 v73, v1, v22 // (op 3900) 

v_lshlrev_b32 v22, 5, v73 // (op 3900) 

v_lshrrev_b32 v73, 1, v1 // (op 3900) 
v_lshl_add_u32 v76, v73, 6, v22 // (op 3900) 


v_lshlrev_b32 v22, 2, v76 // (op 3900) 

v_lshrrev_b32 v73, 3, v22 // (op 3900) 

v_and_b32 v22, -2, v1 // (op 3929) 
v_sub_u32 v76, v1, v22 // (op 3929) 

v_lshlrev_b32 v22, 5, v76 // (op 3929) 

v_lshrrev_b32 v76, 1, v1 // (op 3929) 
v_lshl_add_u32 v77, v76, 6, v22 // (op 3929) 


v_lshlrev_b32 v22, 2, v77 // (op 3929) 

v_lshrrev_b32 v76, 3, v22 // (op 3929) 

v_and_b32 v22, -64, v1 // (op 3958) 
v_sub_u32 v77, v1, v22 // (op 3958) 

v_and_b32 v22, 4294967232, v1 // (op 3958) 
v_add_u32 v78, v22, v77 // (op 3958) 


v_and_b32 v22, -64, v1 // (op 3987) 
v_sub_u32 v77, v1, v22 // (op 3987) 

v_and_b32 v22, 4294967232, v1 // (op 3987) 
v_add_u32 v79, v22, v77 // (op 3987) 


v_and_b32 v22, -2, v1 // (op 4016) 
v_sub_u32 v77, v1, v22 // (op 4016) 

v_lshlrev_b32 v22, 5, v77 // (op 4016) 

v_lshrrev_b32 v77, 1, v1 // (op 4016) 
v_lshl_add_u32 v80, v77, 6, v22 // (op 4016) 


v_lshlrev_b32 v22, 2, v80 // (op 4016) 

v_lshrrev_b32 v77, 3, v22 // (op 4016) 

v_and_b32 v22, -2, v1 // (op 4045) 
v_sub_u32 v80, v1, v22 // (op 4045) 

v_lshlrev_b32 v22, 5, v80 // (op 4045) 

v_lshrrev_b32 v80, 1, v1 // (op 4045) 
v_lshl_add_u32 v81, v80, 6, v22 // (op 4045) 


v_lshlrev_b32 v22, 2, v81 // (op 4045) 

v_lshrrev_b32 v80, 3, v22 // (op 4045) 

v_and_b32 v22, -64, v1 // (op 4074) 
v_sub_u32 v81, v1, v22 // (op 4074) 

v_and_b32 v22, 4294967232, v1 // (op 4074) 
v_add_u32 v82, v22, v81 // (op 4074) 


v_and_b32 v22, -64, v1 // (op 4103) 
v_sub_u32 v81, v1, v22 // (op 4103) 

v_and_b32 v22, 4294967232, v1 // (op 4103) 
v_add_u32 v83, v22, v81 // (op 4103) 


s_mov_b32 s0, 0 // (op 885) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 884) 
s_mov_b32 s1, s6 // (op 884) convert

s_and_b32 s6, s1, 4294967292 // (op 884) 

s_cmp_lt_u32 s0, s6 // (op 884) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoop_884 // (op 884) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoop_884 if false)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoop_884:


s_add_u32 s0, s0, 4 // (op 886) 
v_mov_b32 v22, v25 // (op 10) 
s_lshl_b64 s[6:7], s[24:25], 7 // (op 10) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 10) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 10) 

s_lshl_b32 s1, s6, 2 // (op 10) 

v_add_u32 v81, v22, s1 // (op 10) 


buffer_load_dwordx4 v[32:35], v81, s[48:51], 0 offen // (op 10) Load value

s_waitcnt lgkmcnt(1)
s_barrier  // (op 4203) 





v_mov_b32 v81, v31 // (op 25) 
s_lshl_b32 s1, s34, 4 // (op 25) 
v_add_u32 v84, v81, s1 // (op 25) 


buffer_load_ubyte v29, v84, s[56:59], 0 offen offset: 0 // (op 25) Load value

ds_read_b64_tr_b4 v[84:85], v3 offset:4352 // (op 72) Transpose load local data 


ds_read_b64_tr_b4 v[86:87], v3 offset:4864 // (op 72) Transpose load local data 
v_mov_b32 v88, v23 // (op 4) 
s_barrier  // (op 4216) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 4) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 4) 

s_lshl_b32 s1, s6, 2 // (op 4) 

v_add_u32 v89, v88, s1 // (op 4) 


ds_read_b64_tr_b4 v[90:91], v4 // (op 77) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v4 offset:512 // (op 77) Transpose load local data 
buffer_load_dwordx4 v[36:39], v89, s[8:11], 0 offen // (op 4) Load value



v_mov_b32 v89, v28 // (op 17) 
s_lshl_b32 s1, s28, 4 // (op 17) 
v_add_u32 v94, v89, s1 // (op 17) 


s_waitcnt lgkmcnt(4)
s_barrier  // (op 4230) 
buffer_load_ubyte v40, v94, s[52:55], 0 offen offset: 0 // (op 17) Load value



s_barrier  // (op 4267) 
ds_read_u8 v94, v0 offset:8448 // (op 82) Load local data 



s_barrier  // (op 4244) 
s_waitcnt vmcnt(15)
ds_write_b128 v72, v[42:45] offset:8704 // (op 257) Store local data 


ds_read_u8 v95, v5 offset:4096 // (op 87) Load local data 



s_barrier  // (op 4279) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[90:93], a[0:3], v94, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 90) 



ds_read_b64_tr_b4 v[96:97], v4 offset:16 // (op 521) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v4 offset:528 // (op 521) Transpose load local data 
s_waitcnt vmcnt(14)
ds_write_b8 v75, v41 offset:12800 // (op 259) Store local data 


ds_read_u8 v100, v5 offset:4128 // (op 525) Load local data 



s_barrier  // (op 4261) 
ds_read_b64_tr_b4 v[102:103], v3 offset:4368 // (op 918) Transpose load local data 



ds_read_b64_tr_b4 v[104:105], v3 offset:4880 // (op 918) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:87], v[96:99], a[4:7], v94, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 526) 




ds_read_u8 v94, v0 offset:8480 // (op 921) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v61, v[46:49] offset:13056 // (op 247) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:105], v[90:93], a[8:11], v94, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1015) 


s_barrier  // (op 4273) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:105], v[96:99], a[12:15], v94, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 923) 
s_waitcnt vmcnt(12)
ds_write_b8 v74, v50 offset:17152 // (op 258) Store local data 






s_lshl_b64 s[6:7], s[24:25], 7 // (op 260) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 260) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 260) 

s_mul_i32 s1, 5, s6 // (op 260) 

v_add_u32 v84, v22, s1 // (op 260) 


buffer_load_dwordx4 v[42:45], v84, s[48:51], 0 offen // (op 260) Load value

s_waitcnt lgkmcnt(1)
s_barrier  // (op 4258) 





s_lshl_b32 s1, s34, 2 // (op 262) 
s_mul_i32 s6, 5, s1 // (op 262) 

v_add_u32 v84, v81, s6 // (op 262) 


ds_read_b64_tr_b4 v[90:91], v7 offset:13056 // (op 249) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v7 offset:13568 // (op 249) Transpose load local data 
buffer_load_ubyte v41, v84, s[56:59], 0 offen offset: 0 // (op 262) Load value



s_barrier  // (op 4264) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 246) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 246) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 246) 

s_mul_i32 s1, 5, s6 // (op 246) 

v_add_u32 v84, v88, s1 // (op 246) 


ds_read_b64_tr_b4 v[94:95], v8 offset:8704 // (op 251) Transpose load local data 
ds_read_b64_tr_b4 v[96:97], v8 offset:9216 // (op 251) Transpose load local data 
buffer_load_dwordx4 v[46:49], v84, s[8:11], 0 offen // (op 246) Load value



s_lshl_b32 s1, s28, 2 // (op 261) 
s_mul_i32 s6, 5, s1 // (op 261) 

v_add_u32 v84, v89, s6 // (op 261) 


s_waitcnt lgkmcnt(4)
s_barrier  // (op 4270) 
buffer_load_ubyte v50, v84, s[52:55], 0 offen offset: 0 // (op 261) Load value



s_barrier  // (op 4291) 
ds_read_u8 v84, v6 offset:17152 // (op 253) Load local data 



s_barrier  // (op 4276) 
s_waitcnt vmcnt(15)
ds_write_b128 v76, v[52:55] offset:17408 // (op 297) Store local data 


ds_read_u8 v85, v9 offset:12800 // (op 255) Load local data 



s_barrier  // (op 4303) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[94:97], a[0:3], v84, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 256) 



ds_read_b64_tr_b4 v[98:99], v8 offset:8720 // (op 533) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v8 offset:9232 // (op 533) Transpose load local data 
s_waitcnt vmcnt(14)
ds_write_b8 v79, v51 offset:21504 // (op 299) Store local data 


s_barrier  // (op 4408) 
s_barrier  // (op 4285) 
ds_read_u8 v86, v9 offset:12832 // (op 537) Load local data 



s_waitcnt vmcnt(13)
ds_write_b128 v73, v[56:59] offset:21760 // (op 287) Store local data 



s_barrier  // (op 4412) 
s_barrier  // (op 4297) 
ds_read_b64_tr_b4 v[102:103], v7 offset:13072 // (op 928) Transpose load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:93], v[98:101], a[4:7], v84, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 527) 
ds_read_b64_tr_b4 v[104:105], v7 offset:13584 // (op 928) Transpose load local data 


s_waitcnt vmcnt(12)
ds_write_b8 v78, v60 offset:25856 // (op 298) Store local data 


s_barrier  // (op 4452) 
ds_read_u8 v84, v6 offset:17184 // (op 931) Load local data 
s_barrier  // (op 4456) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:105], v[94:97], a[8:11], v84, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1016) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:105], v[98:101], a[12:15], v84, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 924) 






s_barrier  // (op 4282) 
s_lshl_b64 s[6:7], s[24:25], 7 // (op 300) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 300) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 300) 

s_mul_i32 s1, 6, s6 // (op 300) 

v_add_u32 v84, v22, s1 // (op 300) 


ds_read_b64_tr_b4 v[90:91], v11 offset:21760 // (op 289) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v11 offset:22272 // (op 289) Transpose load local data 
buffer_load_dwordx4 v[52:55], v84, s[48:51], 0 offen // (op 300) Load value



s_barrier  // (op 4288) 
s_lshl_b32 s1, s34, 2 // (op 302) 



s_mul_i32 s6, 6, s1 // (op 302) 

v_add_u32 v84, v81, s6 // (op 302) 


ds_read_b64_tr_b4 v[94:95], v12 offset:17408 // (op 291) Transpose load local data 
ds_read_b64_tr_b4 v[96:97], v12 offset:17920 // (op 291) Transpose load local data 
buffer_load_ubyte v51, v84, s[56:59], 0 offen offset: 0 // (op 302) Load value



s_barrier  // (op 4294) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 286) 


ds_read_u8 v84, v10 offset:25856 // (op 293) Load local data 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 286) 

s_barrier  // (op 4300) 
s_lshr_b64 s[6:7], s[14:15], 3 // (op 286) 



ds_read_u8 v85, v13 offset:21504 // (op 295) Load local data 
s_mul_i32 s1, 6, s6 // (op 286) 




v_add_u32 v86, v88, s1 // (op 286) 


ds_read_b64_tr_b4 v[98:99], v12 offset:17424 // (op 541) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[94:97], a[0:3], v84, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 296) 
ds_read_b64_tr_b4 v[100:101], v12 offset:17936 // (op 541) Transpose load local data 
buffer_load_dwordx4 v[56:59], v86, s[8:11], 0 offen // (op 286) Load value





s_barrier  // (op 4416) 
ds_read_u8 v86, v13 offset:21536 // (op 545) Load local data 
s_lshl_b32 s1, s28, 2 // (op 301) 
s_mul_i32 s6, 6, s1 // (op 301) 




s_barrier  // (op 4420) 
v_add_u32 v87, v89, s6 // (op 301) 


ds_read_b64_tr_b4 v[102:103], v11 offset:21776 // (op 934) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v11 offset:22288 // (op 934) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:93], v[98:101], a[4:7], v84, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 528) 




s_barrier  // (op 4460) 
ds_read_u8 v84, v10 offset:25888 // (op 937) Load local data 
buffer_load_ubyte v60, v87, s[52:55], 0 offen offset: 0 // (op 301) Load value

s_barrier  // (op 4315) 
s_barrier  // (op 4464) 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:105], v[94:97], a[8:11], v84, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1017) 
s_waitcnt vmcnt(15)
ds_write_b128 v80, v[62:65] offset:8704 // (op 337) Store local data 


s_barrier  // (op 4327) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:105], v[98:101], a[12:15], v84, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 925) 







s_waitcnt vmcnt(14)
ds_write_b8 v83, v26 offset:12800 // (op 339) Store local data 
s_barrier  // (op 4309) 



s_waitcnt vmcnt(13)
ds_write_b128 v77, v[66:69] offset:13056 // (op 327) Store local data 
s_barrier  // (op 4321) 



s_waitcnt vmcnt(12)
ds_write_b8 v82, v24 offset:17152 // (op 338) Store local data 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 340) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 340) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 340) 

s_mul_i32 s1, 7, s6 // (op 340) 

v_add_u32 v84, v22, s1 // (op 340) 


s_waitcnt lgkmcnt(1)
s_barrier  // (op 4306) 
buffer_load_dwordx4 v[62:65], v84, s[48:51], 0 offen // (op 340) Load value





ds_read_b64_tr_b4 v[84:85], v15 offset:13056 // (op 329) Transpose load local data 


ds_read_b64_tr_b4 v[86:87], v15 offset:13568 // (op 329) Transpose load local data 
s_lshl_b32 s1, s34, 2 // (op 342) 
s_barrier  // (op 4312) 
s_mul_i32 s6, 7, s1 // (op 342) 




v_add_u32 v22, v81, s6 // (op 342) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 3350) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 3350) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 3350) 

s_lshl_b32 s1, s6, 2 // (op 3350) 

v_add_u32 v25, v25, s1 // (op 3350) 

ds_read_b64_tr_b4 v[90:91], v16 offset:8704 // (op 331) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v16 offset:9216 // (op 331) Transpose load local data 
buffer_load_ubyte v26, v22, s[56:59], 0 offen offset: 0 // (op 342) Load value


s_waitcnt lgkmcnt(4)
s_barrier  // (op 4318) 




ds_read_u8 v22, v14 offset:17152 // (op 333) Load local data 
s_lshl_b32 s1, s34, 4 // (op 3478) 
s_barrier  // (op 4324) 
v_add_u32 v31, v31, s1 // (op 3478) 



ds_read_u8 v81, v17 offset:12800 // (op 335) Load local data 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 326) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 326) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 326) 

s_mul_i32 s1, 7, s6 // (op 326) 

v_add_u32 v94, v88, s1 // (op 326) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[90:93], a[0:3], v22, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 336) 
ds_read_b64_tr_b4 v[96:97], v16 offset:8720 // (op 549) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v16 offset:9232 // (op 549) Transpose load local data 
buffer_load_dwordx4 v[66:69], v94, s[8:11], 0 offen // (op 326) Load value






s_barrier  // (op 4424) 
ds_read_u8 v88, v17 offset:12832 // (op 553) Load local data 
s_lshl_b32 s1, s28, 2 // (op 341) 
s_mul_i32 s6, 7, s1 // (op 341) 




s_barrier  // (op 4428) 
v_add_u32 v94, v89, s6 // (op 341) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 3286) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 3286) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 3286) 

s_lshl_b32 s1, s6, 2 // (op 3286) 

v_add_u32 v23, v23, s1 // (op 3286) 

ds_read_b64_tr_b4 v[100:101], v15 offset:13072 // (op 940) Transpose load local data 
ds_read_b64_tr_b4 v[102:103], v15 offset:13584 // (op 940) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:87], v[96:99], a[4:7], v22, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 529) 




s_barrier  // (op 4468) 
ds_read_u8 v22, v14 offset:17184 // (op 943) Load local data 
buffer_load_ubyte v24, v94, s[52:55], 0 offen offset: 0 // (op 341) Load value


s_barrier  // (op 4472) 
s_barrier  // (op 4219) 
s_lshl_b32 s1, s28, 4 // (op 3414) 



v_add_u32 v28, v28, s1 // (op 3414) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:103], v[90:93], a[8:11], v22, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1018) 
s_waitcnt vmcnt(15)
ds_write_b128 v30, v[32:35] // (op 13) Store local data 


s_barrier  // (op 4247) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:103], v[96:99], a[12:15], v22, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 926) 







s_waitcnt vmcnt(14)
ds_write_b8 v71, v29 offset:4096 // (op 30) Store local data 
s_barrier  // (op 4206) 



s_waitcnt vmcnt(13)
ds_write_b128 v27, v[36:39] offset:4352 // (op 6) Store local data 
s_barrier  // (op 4233) 



s_waitcnt vmcnt(12)
ds_write_b8 v70, v40 offset:8448 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 884) 
s_mov_b32 s1, s6 // (op 884) convert

s_and_b32 s6, s1, 4294967292 // (op 884) 

s_cmp_lt_u32 s0, s6 // (op 884) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoop_884 // (op 884) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoop_884 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoop_884:

























v_and_b32 v25, -2, v1 // (op 3246) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3246) 
v_sub_u32 v61, v1, v25 // (op 3246) 

s_lshr_b32 s6, s0, 2 // (op 3246) 

v_lshlrev_b32 v25, 5, v61 // (op 3246) 

v_lshl_add_u32 v61, s16, 6, v25 // (op 3246) 

v_mov_b32 v23, 0 // (op 3246) convert
v_mov_b32 v22, v61 // (op 3246) convert

v_lshrrev_b32 v25, 1, v1 // (op 3246) 
v_lshl_add_u32 v61, s6, 9, v25 // (op 3246) 


v_mul_lo_u32 v29, v61, s13 // (op 3246) most significant: low of low * high
v_mul_hi_u32 v25, v61, s12 // (op 3246) most significant: high of low * low
v_mul_lo_u32 v28, v61, s12 // (op 3246) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 3246) most significant: sum


v_add_co_u32 v30, s[0:1], v22, v28 // (op 3246) least significant half
v_addc_co_u32 v31, s[0:1], v23, v29, s[0:1] // (op 3246) most significant half

v_lshlrev_b64 v[30:31], 2, v[30:31] // (op 3246) 


v_lshrrev_b64 v[22:23], 3, v[30:31] // (op 3246) 

v_mov_b32 v25, v22 // (op 3246) convert

s_mul_i32 s1, s12, s23 // (op 3248) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3248) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3248) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3248) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3248) most significant: sum
s_add_u32 s1, s1, s7 // (op 3248) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3248) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3248) 

s_mov_b32 s8, s20 // (op 3248) reg expression
s_mov_b32 s9, s21 // (op 3248) reg expression
s_mov_b32 s10, s0 // (op 3248) reg expression
s_mov_b32 s11, 131072 // (op 3248) call()

v_lshrrev_b32 v61, 1, v1 // (op 3316) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3316) 
v_and_b32 v22, -2, v1 // (op 3316) 
s_lshr_b32 s6, s0, 2 // (op 3316) 

v_sub_u32 v23, v1, v22 // (op 3316) 

v_lshl_add_u32 v22, s6, 9, v61 // (op 3316) 


v_lshlrev_b32 v61, 5, v23 // (op 3316) 

v_lshl_add_u32 v23, s17, 6, v61 // (op 3316) 

v_mov_b32 v29, 0 // (op 3316) convert
v_mov_b32 v28, v23 // (op 3316) convert

v_mul_lo_u32 v31, v22, s25 // (op 3316) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 3316) most significant: high of low * low
v_mul_lo_u32 v30, v22, s24 // (op 3316) least significant: low of low * low
v_add_u32 v31, v31, v23 // (op 3316) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v28 // (op 3316) least significant half
v_addc_co_u32 v31, s[0:1], v31, v29, s[0:1] // (op 3316) most significant half



v_lshlrev_b64 v[22:23], 2, v[30:31] // (op 3316) 

v_lshrrev_b64 v[28:29], 3, v[22:23] // (op 3316) 

v_mov_b32 v27, v28 // (op 3316) convert

s_mul_i32 s1, s24, s23 // (op 3318) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 3318) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 3318) least significant
s_mul_hi_u32 s7, s24, s22 // (op 3318) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3318) most significant: sum
s_add_u32 s1, s1, s7 // (op 3318) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3318) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3318) 

s_mov_b32 s48, s26 // (op 3318) reg expression
s_mov_b32 s49, s27 // (op 3318) reg expression
s_mov_b32 s50, s0 // (op 3318) reg expression
s_mov_b32 s51, 131072 // (op 3318) call()

v_and_b32 v61, -64, v1 // (op 3380) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3380) 
v_sub_u32 v22, v1, v61 // (op 3380) 

s_lshr_b32 s6, s0, 2 // (op 3380) 

v_lshl_add_u32 v23, s16, 6, v22 // (op 3380) 

v_lshrrev_b32 v22, 6, v1 // (op 3380) 
v_lshl_add_u32 v61, s6, 4, v22 // (op 3380) 


v_mul_lo_u32 v22, v61, s28 // (op 3380) 

v_add_u32 v61, v23, v22 // (op 3380) 


s_mul_i32 s1, s28, s33 // (op 3382) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3382) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3382) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3382) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3382) most significant: sum
s_add_u32 s1, s1, s7 // (op 3382) most significant: sum


s_mov_b32 s52, s30 // (op 3382) reg expression
s_mov_b32 s53, s31 // (op 3382) reg expression
s_mov_b32 s54, s0 // (op 3382) reg expression
s_mov_b32 s55, 131072 // (op 3382) call()

v_lshrrev_b32 v22, 6, v1 // (op 3444) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3444) 
v_and_b32 v23, -64, v1 // (op 3444) 
s_lshr_b32 s6, s0, 2 // (op 3444) 

v_sub_u32 v28, v1, v23 // (op 3444) 

v_lshl_add_u32 v23, s6, 4, v22 // (op 3444) 


v_lshl_add_u32 v22, s17, 6, v28 // (op 3444) 

v_mul_lo_u32 v28, v23, s34 // (op 3444) 
v_add_u32 v28, v28, v22 // (op 3444) 


s_mul_i32 s1, s34, s33 // (op 3446) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 3446) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 3446) least significant
s_mul_hi_u32 s7, s34, s32 // (op 3446) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3446) most significant: sum
s_add_u32 s1, s1, s7 // (op 3446) most significant: sum


s_mov_b32 s56, s36 // (op 3446) reg expression
s_mov_b32 s57, s37 // (op 3446) reg expression
s_mov_b32 s58, s0 // (op 3446) reg expression
s_mov_b32 s59, 131072 // (op 3446) call()

v_and_b32 v22, -2, v1 // (op 3542) 
v_sub_u32 v23, v1, v22 // (op 3542) 

v_lshlrev_b32 v22, 5, v23 // (op 3542) 

v_lshrrev_b32 v23, 1, v1 // (op 3542) 
v_lshl_add_u32 v29, v23, 6, v22 // (op 3542) 


v_lshlrev_b32 v22, 2, v29 // (op 3542) 

v_lshrrev_b32 v23, 3, v22 // (op 3542) 

v_and_b32 v22, -2, v1 // (op 3600) 
v_sub_u32 v29, v1, v22 // (op 3600) 

v_lshlrev_b32 v22, 5, v29 // (op 3600) 

v_lshrrev_b32 v29, 1, v1 // (op 3600) 
v_lshl_add_u32 v30, v29, 6, v22 // (op 3600) 


v_lshlrev_b32 v22, 2, v30 // (op 3600) 

v_lshrrev_b32 v29, 3, v22 // (op 3600) 

v_and_b32 v22, -64, v1 // (op 3658) 
v_sub_u32 v30, v1, v22 // (op 3658) 

v_and_b32 v22, 4294967232, v1 // (op 3658) 
v_add_u32 v31, v22, v30 // (op 3658) 


v_and_b32 v22, -64, v1 // (op 3716) 
v_sub_u32 v30, v1, v22 // (op 3716) 

v_and_b32 v22, 4294967232, v1 // (op 3716) 
v_add_u32 v32, v22, v30 // (op 3716) 


s_ashr_i64 s[0:1], s[22:23], 7 // (op 890) 
s_and_b32 s6, s0, 4294967292 // (op 890) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 889) 
s_mov_b32 s7, s0 // (op 889) convert

s_cmp_lt_u32 s6, s7 // (op 889) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoopTail_889 // (op 889) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoopTail_889 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoopTail_889:


s_add_u32 s6, s6, 1 // (op 891) 
v_mov_b32 v22, v25 // (op 180) 
v_mov_b32 v24, v22 // (op 180) reg expression

buffer_load_dwordx4 v[34:37], v24, s[8:11], 0 offen // (op 180) Load value


s_lshl_b64 s[0:1], s[12:13], 7 // (op 3236) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 3236) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 3236) 

v_add_u32 v25, v25, s0 // (op 3236) 



s_barrier  // (op 4209) 
v_mov_b32 v22, v27 // (op 194) 



v_mov_b32 v24, v22 // (op 194) reg expression

buffer_load_dwordx4 v[38:41], v24, s[48:51], 0 offen // (op 194) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 3308) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 3308) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 3308) 

v_add_u32 v27, v27, s0 // (op 3308) 

s_waitcnt vmcnt(1)
ds_write_b128 v23, v[34:37] offset:4352 // (op 181) Store local data 



v_mov_b32 v22, v61 // (op 195) 
s_barrier  // (op 4222) 
v_mov_b32 v24, v22 // (op 195) reg expression




buffer_load_ubyte v26, v24, s[52:55], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s7, s28, 2 // (op 3372) 
v_add_u32 v61, v61, s7 // (op 3372) 

s_waitcnt vmcnt(1)
ds_write_b128 v29, v[38:41] // (op 191) Store local data 



v_mov_b32 v22, v28 // (op 196) 
s_barrier  // (op 4236) 
v_mov_b32 v24, v22 // (op 196) reg expression




buffer_load_ubyte v30, v24, s[56:59], 0 offen offset: 0 // (op 196) Load value


s_lshl_b32 s7, s34, 2 // (op 3436) 
v_add_u32 v28, v28, s7 // (op 3436) 

s_waitcnt vmcnt(1)
ds_write_b8 v31, v26 offset:8448 // (op 192) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 4212) 



s_barrier  // (op 4250) 
ds_read_b64_tr_b4 v[34:35], v3 offset:4352 // (op 183) Transpose load local data 



ds_read_b64_tr_b4 v[36:37], v3 offset:4864 // (op 183) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v32, v30 offset:4096 // (op 193) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 4226) 



ds_read_b64_tr_b4 v[38:39], v4 // (op 185) Transpose load local data 
ds_read_b64_tr_b4 v[40:41], v4 offset:512 // (op 185) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 4240) 


ds_read_u8 v22, v0 offset:8448 // (op 187) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4254) 


ds_read_u8 v24, v5 offset:4096 // (op 189) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:41], a[0:3], v22, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 190) 
ds_read_b64_tr_b4 v[42:43], v4 offset:16 // (op 500) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v4 offset:528 // (op 500) Transpose load local data 


ds_read_u8 v26, v5 offset:4128 // (op 504) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[42:45], a[4:7], v22, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 505) 
ds_read_b64_tr_b4 v[46:47], v3 offset:4368 // (op 864) Transpose load local data 


ds_read_b64_tr_b4 v[48:49], v3 offset:4880 // (op 864) Transpose load local data 


ds_read_u8 v22, v0 offset:8480 // (op 867) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[38:41], a[8:11], v22, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 869) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[42:45], a[12:15], v22, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 900) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 889) 
s_mov_b32 s7, s0 // (op 889) convert

s_cmp_lt_u32 s6, s7 // (op 889) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoopTail_889 // (op 889) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoopTail_889 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoopTail_889:























s_barrier  // (op 4436) 
















v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalFalse_0_beta_135
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_mul_f32 v10, v19, v14 // (op 50) 
v_mul_f32 v11, v19, v15 // (op 50) 
v_mul_f32 v12, v19, v16 // (op 50) 
v_mul_f32 v13, v19, v17 // (op 50) 

v_lshrrev_b32 v14, 16, v10 // (op 56) 
v_lshrrev_b32 v15, 16, v11 // (op 56) 
v_lshrrev_b32 v16, 16, v12 // (op 56) 
v_lshrrev_b32 v17, 16, v13 // (op 56) 


v_pack_b32_f16 v12, v14, v15 // (op 58) 
v_pack_b32_f16 v13, v16, v17 // (op 58) 

v_mov_b32 v0, v21 // (op 58) reg expression

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v14, a4 // (op 1621) 
v_accvgpr_read v15, a5 // (op 1621) 
v_accvgpr_read v16, a6 // (op 1621) 
v_accvgpr_read v17, a7 // (op 1621) 
v_mul_f32 v10, v19, v14 // (op 1621) 
v_mul_f32 v11, v19, v15 // (op 1621) 
v_mul_f32 v12, v19, v16 // (op 1621) 
v_mul_f32 v13, v19, v17 // (op 1621) 

v_lshrrev_b32 v14, 16, v10 // (op 1623) 
v_lshrrev_b32 v15, 16, v11 // (op 1623) 
v_lshrrev_b32 v16, 16, v12 // (op 1623) 
v_lshrrev_b32 v17, 16, v13 // (op 1623) 


v_pack_b32_f16 v12, v14, v15 // (op 1624) 
v_pack_b32_f16 v13, v16, v17 // (op 1624) 

s_lshl_b32 s0, s4, 6 // (op 1624) 
v_add_u32 v0, v21, s0 // (op 1624) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 1624) Store value



v_accvgpr_read v14, a8 // (op 1647) 
v_accvgpr_read v15, a9 // (op 1647) 
v_accvgpr_read v16, a10 // (op 1647) 
v_accvgpr_read v17, a11 // (op 1647) 
v_mul_f32 v10, v19, v14 // (op 1647) 
v_mul_f32 v11, v19, v15 // (op 1647) 
v_mul_f32 v12, v19, v16 // (op 1647) 
v_mul_f32 v13, v19, v17 // (op 1647) 

v_lshrrev_b32 v14, 16, v10 // (op 1649) 
v_lshrrev_b32 v15, 16, v11 // (op 1649) 
v_lshrrev_b32 v16, 16, v12 // (op 1649) 
v_lshrrev_b32 v17, 16, v13 // (op 1649) 


v_pack_b32_f16 v12, v14, v15 // (op 1651) 
v_pack_b32_f16 v13, v16, v17 // (op 1651) 

v_add_u32 v0, 64, v21 // (op 1651) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 1651) Store value



v_accvgpr_read v14, a12 // (op 1657) 
v_accvgpr_read v15, a13 // (op 1657) 
v_accvgpr_read v16, a14 // (op 1657) 
v_accvgpr_read v17, a15 // (op 1657) 
v_mul_f32 v10, v19, v14 // (op 1657) 
v_mul_f32 v11, v19, v15 // (op 1657) 
v_mul_f32 v12, v19, v16 // (op 1657) 
v_mul_f32 v13, v19, v17 // (op 1657) 

v_lshrrev_b32 v14, 16, v10 // (op 1659) 
v_lshrrev_b32 v15, 16, v11 // (op 1659) 
v_lshrrev_b32 v16, 16, v12 // (op 1659) 
v_lshrrev_b32 v17, 16, v13 // (op 1659) 


v_pack_b32_f16 v12, v14, v15 // (op 1653) 
v_pack_b32_f16 v13, v16, v17 // (op 1653) 

s_lshl_b32 s0, s4, 6 // (op 1653) 
s_add_u32 s0, s0, 64 // (op 1653) 
v_add_u32 v0, v21, s0 // (op 1653) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 1653) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalBottom_0_beta_135
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v20 // (op 1743) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 1743) Load value

v_accvgpr_read v12, a8 // (op 1734) 
v_accvgpr_read v13, a9 // (op 1734) 
v_accvgpr_read v14, a10 // (op 1734) 
v_accvgpr_read v15, a11 // (op 1734) 
v_mul_f32 v8, v19, v12 // (op 1734) 
v_mul_f32 v9, v19, v13 // (op 1734) 
v_mul_f32 v10, v19, v14 // (op 1734) 
v_mul_f32 v11, v19, v15 // (op 1734) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 1742) 
v_lshrrev_b32 v7, 16, v16 // (op 1742) 
v_lshlrev_b32 v6, 16, v6 // (op 1742) 
v_lshlrev_b32 v7, 16, v7 // (op 1742) 
v_mul_f32 v12, v18, v6 // (op 1742) 
v_mul_f32 v13, v18, v7 // (op 1742) 
v_and_b32 v6, 65535, v17 // (op 1742) 
v_lshrrev_b32 v7, 16, v17 // (op 1742) 
v_lshlrev_b32 v6, 16, v6 // (op 1742) 
v_lshlrev_b32 v7, 16, v7 // (op 1742) 
v_mul_f32 v14, v18, v6 // (op 1742) 
v_mul_f32 v15, v18, v7 // (op 1742) 


v_add_f32 v4, v12, v8 // (op 1735) 
v_add_f32 v5, v13, v9 // (op 1735) 
v_add_f32 v6, v14, v10 // (op 1735) 
v_add_f32 v7, v15, v11 // (op 1735) 


v_lshrrev_b32 v8, 16, v4 // (op 1736) 
v_lshrrev_b32 v9, 16, v5 // (op 1736) 
v_lshrrev_b32 v10, 16, v6 // (op 1736) 
v_lshrrev_b32 v11, 16, v7 // (op 1736) 


v_pack_b32_f16 v6, v8, v9 // (op 1738) 
v_pack_b32_f16 v7, v10, v11 // (op 1738) 

v_add_u32 v0, 64, v21 // (op 1738) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 1738) Store value





s_lshl_b32 s0, s2, 6 // (op 1748) 
s_add_u32 s0, s0, 64 // (op 1748) 
v_add_u32 v0, v20, s0 // (op 1748) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 1748) Load value

v_accvgpr_read v12, a12 // (op 1750) 
v_accvgpr_read v13, a13 // (op 1750) 
v_accvgpr_read v14, a14 // (op 1750) 
v_accvgpr_read v15, a15 // (op 1750) 
v_mul_f32 v8, v19, v12 // (op 1750) 
v_mul_f32 v9, v19, v13 // (op 1750) 
v_mul_f32 v10, v19, v14 // (op 1750) 
v_mul_f32 v11, v19, v15 // (op 1750) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 1745) 
v_lshrrev_b32 v7, 16, v16 // (op 1745) 
v_lshlrev_b32 v6, 16, v6 // (op 1745) 
v_lshlrev_b32 v7, 16, v7 // (op 1745) 
v_mul_f32 v12, v18, v6 // (op 1745) 
v_mul_f32 v13, v18, v7 // (op 1745) 
v_and_b32 v6, 65535, v17 // (op 1745) 
v_lshrrev_b32 v7, 16, v17 // (op 1745) 
v_lshlrev_b32 v6, 16, v6 // (op 1745) 
v_lshlrev_b32 v7, 16, v7 // (op 1745) 
v_mul_f32 v14, v18, v6 // (op 1745) 
v_mul_f32 v15, v18, v7 // (op 1745) 


v_add_f32 v4, v12, v8 // (op 1746) 
v_add_f32 v5, v13, v9 // (op 1746) 
v_add_f32 v6, v14, v10 // (op 1746) 
v_add_f32 v7, v15, v11 // (op 1746) 


v_lshrrev_b32 v8, 16, v4 // (op 1747) 
v_lshrrev_b32 v9, 16, v5 // (op 1747) 
v_lshrrev_b32 v10, 16, v6 // (op 1747) 
v_lshrrev_b32 v11, 16, v7 // (op 1747) 


v_pack_b32_f16 v6, v8, v9 // (op 1740) 
v_pack_b32_f16 v7, v10, v11 // (op 1740) 

s_lshl_b32 s0, s4, 6 // (op 1740) 
s_add_u32 s0, s0, 64 // (op 1740) 
v_add_u32 v0, v21, s0 // (op 1740) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 1740) Store value





v_mov_b32 v0, v20 // (op 148) reg expression

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 148) Load value

v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v8, v19, v12 // (op 144) 
v_mul_f32 v9, v19, v13 // (op 144) 
v_mul_f32 v10, v19, v14 // (op 144) 
v_mul_f32 v11, v19, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 149) 
v_lshrrev_b32 v7, 16, v16 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v12, v18, v6 // (op 149) 
v_mul_f32 v13, v18, v7 // (op 149) 
v_and_b32 v6, 65535, v17 // (op 149) 
v_lshrrev_b32 v7, 16, v17 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v14, v18, v6 // (op 149) 
v_mul_f32 v15, v18, v7 // (op 149) 


v_add_f32 v4, v12, v8 // (op 145) 
v_add_f32 v5, v13, v9 // (op 145) 
v_add_f32 v6, v14, v10 // (op 145) 
v_add_f32 v7, v15, v11 // (op 145) 


v_lshrrev_b32 v8, 16, v4 // (op 146) 
v_lshrrev_b32 v9, 16, v5 // (op 146) 
v_lshrrev_b32 v10, 16, v6 // (op 146) 
v_lshrrev_b32 v11, 16, v7 // (op 146) 


v_pack_b32_f16 v6, v8, v9 // (op 147) 
v_pack_b32_f16 v7, v10, v11 // (op 147) 

v_mov_b32 v0, v21 // (op 147) reg expression

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 1703) 
v_add_u32 v0, v20, s0 // (op 1703) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 1703) Load value

v_accvgpr_read v12, a4 // (op 1699) 
v_accvgpr_read v13, a5 // (op 1699) 
v_accvgpr_read v14, a6 // (op 1699) 
v_accvgpr_read v15, a7 // (op 1699) 
v_mul_f32 v8, v19, v12 // (op 1699) 
v_mul_f32 v9, v19, v13 // (op 1699) 
v_mul_f32 v10, v19, v14 // (op 1699) 
v_mul_f32 v11, v19, v15 // (op 1699) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 1704) 
v_lshrrev_b32 v7, 16, v16 // (op 1704) 
v_lshlrev_b32 v6, 16, v6 // (op 1704) 
v_lshlrev_b32 v7, 16, v7 // (op 1704) 
v_mul_f32 v12, v18, v6 // (op 1704) 
v_mul_f32 v13, v18, v7 // (op 1704) 
v_and_b32 v6, 65535, v17 // (op 1704) 
v_lshrrev_b32 v7, 16, v17 // (op 1704) 
v_lshlrev_b32 v6, 16, v6 // (op 1704) 
v_lshlrev_b32 v7, 16, v7 // (op 1704) 
v_mul_f32 v14, v18, v6 // (op 1704) 
v_mul_f32 v15, v18, v7 // (op 1704) 


v_add_f32 v4, v12, v8 // (op 1700) 
v_add_f32 v5, v13, v9 // (op 1700) 
v_add_f32 v6, v14, v10 // (op 1700) 
v_add_f32 v7, v15, v11 // (op 1700) 


v_lshrrev_b32 v8, 16, v4 // (op 1701) 
v_lshrrev_b32 v9, 16, v5 // (op 1701) 
v_lshrrev_b32 v10, 16, v6 // (op 1701) 
v_lshrrev_b32 v11, 16, v7 // (op 1701) 


v_pack_b32_f16 v6, v8, v9 // (op 1702) 
v_pack_b32_f16 v7, v10, v11 // (op 1702) 

s_lshl_b32 s0, s4, 6 // (op 1702) 
v_add_u32 v0, v21, s0 // (op 1702) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 1702) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalBottom_0_beta_135:












s_endpgm  // End of RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
.LRR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_end:
.size RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4, .LRR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_end-RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
  .amdhsa_next_free_vgpr 124
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 26112 // lds bytes
  .amdhsa_accum_offset 108
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
    DataTypeB: F4
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
    TransposeB: True
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
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB } ]
    macrotile: [64, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 4
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
    .symbol: RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 26112
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 106
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_0
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
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

