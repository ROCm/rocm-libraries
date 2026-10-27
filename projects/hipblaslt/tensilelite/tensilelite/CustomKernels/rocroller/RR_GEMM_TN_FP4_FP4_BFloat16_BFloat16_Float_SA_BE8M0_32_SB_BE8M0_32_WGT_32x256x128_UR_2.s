// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 3749) 
v_and_b32 v3, 15, v1 // (op 3749) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3749) 


v_and_b32 v0, 63, v1 // (op 3749) 
v_lshrrev_b32 v3, 4, v0 // (op 3749) 

v_lshlrev_b32 v0, 5, v3 // (op 3749) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 3749) 


v_lshlrev_b32 v0, 2, v3 // (op 3749) 

v_lshrrev_b32 v3, 3, v0 // (op 3749) 

v_lshrrev_b32 v0, 7, v1 // (op 3907) 
v_and_b32 v4, 15, v1 // (op 3907) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3907) 


v_and_b32 v0, 63, v1 // (op 3907) 
v_lshrrev_b32 v4, 4, v0 // (op 3907) 

v_lshlrev_b32 v0, 5, v4 // (op 3907) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 3907) 


v_lshlrev_b32 v0, 2, v4 // (op 3907) 

v_lshrrev_b32 v4, 3, v0 // (op 3907) 

v_lshrrev_b32 v0, 7, v1 // (op 3692) 
v_and_b32 v5, 63, v1 // (op 3692) 
v_lshrrev_b32 v6, 4, v5 // (op 3692) 
v_lshlrev_b32 v7, 2, v6 // (op 3692) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3692) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 3692) 

v_lshrrev_b32 v6, 2, v5 // (op 3692) 

v_and_b32 v5, 3, v6 // (op 3692) 

v_and_b32 v6, 3, v1 // (op 3692) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3692) 


v_lshrrev_b32 v5, 6, v1 // (op 3692) 
v_and_b32 v6, 1, v5 // (op 3692) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3692) 


v_lshl_add_u32 v6, s17, 8, v5 // (op 3692) 

v_mul_lo_u32 v5, v6, s2 // (op 3692) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 3692) 


v_lshrrev_b32 v0, 6, v1 // (op 3807) 
v_and_b32 v5, 1, v0 // (op 3807) 

v_and_b32 v0, 15, v1 // (op 3807) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 3807) 


v_and_b32 v0, 63, v1 // (op 3807) 
v_lshrrev_b32 v5, 4, v0 // (op 3807) 

v_lshlrev_b32 v0, 5, v5 // (op 3807) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 3807) 


v_lshlrev_b32 v0, 2, v5 // (op 3807) 

v_lshrrev_b32 v5, 3, v0 // (op 3807) 

v_lshrrev_b32 v0, 6, v1 // (op 3865) 
v_and_b32 v7, 1, v0 // (op 3865) 

v_and_b32 v0, 15, v1 // (op 3865) 
s_mul_i32 s1, s2, s9 // (op 3694) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3694) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3694) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3694) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3694) most significant: sum
s_add_u32 s1, s1, s19 // (op 3694) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3694) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 3865) 


v_and_b32 v0, 63, v1 // (op 3865) 
v_lshrrev_b32 v7, 4, v0 // (op 3865) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 3865) 


v_lshrrev_b32 v7, 7, v1 // (op 3881) 
v_and_b32 v8, 63, v1 // (op 3881) 
v_lshrrev_b32 v9, 4, v8 // (op 3881) 
v_lshlrev_b32 v10, 2, v9 // (op 3881) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3881) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 3881) 

v_lshrrev_b32 v9, 2, v8 // (op 3881) 

v_and_b32 v8, 3, v9 // (op 3881) 

v_and_b32 v9, 3, v1 // (op 3881) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3881) 


v_lshrrev_b32 v8, 6, v1 // (op 3881) 
v_and_b32 v9, 1, v8 // (op 3881) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3881) 


v_lshl_add_u32 v9, s17, 8, v8 // (op 3881) 

v_mul_lo_u32 v8, v9, s4 // (op 3881) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 3881) 


v_lshrrev_b32 v7, 6, v1 // (op 3936) 
v_and_b32 v8, 1, v7 // (op 3936) 

v_and_b32 v7, 15, v1 // (op 3936) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 3936) 


v_and_b32 v7, 63, v1 // (op 3936) 
v_lshrrev_b32 v8, 4, v7 // (op 3936) 

v_lshlrev_b32 v7, 5, v8 // (op 3936) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 3936) 


v_lshlrev_b32 v7, 2, v8 // (op 3936) 

v_lshrrev_b32 v8, 3, v7 // (op 3936) 

v_lshrrev_b32 v7, 6, v1 // (op 3965) 
v_and_b32 v10, 1, v7 // (op 3965) 

v_and_b32 v7, 15, v1 // (op 3965) 
s_mul_i32 s1, s4, s9 // (op 3883) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 3965) 


v_and_b32 v7, 63, v1 // (op 3965) 
v_lshrrev_b32 v10, 4, v7 // (op 3965) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 3965) 


s_mul_i32 s38, s5, s8 // (op 3883) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3883) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3883) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3883) most significant: sum
s_add_u32 s1, s1, s39 // (op 3883) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 3883) 

s_mov_b32 s40, s6 // (op 3694) reg expression
s_mov_b32 s41, s7 // (op 3694) reg expression
s_mov_b32 s42, s18 // (op 3694) reg expression
s_mov_b32 s43, 131072 // (op 3694) call()

s_mov_b32 s44, s10 // (op 3883) reg expression

s_mov_b32 s45, s11 // (op 3883) reg expression
s_mov_b32 s46, s38 // (op 3883) reg expression
s_mov_b32 s47, 131072 // (op 3883) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
v_accvgpr_write a8, 0.00000 // (op 466) call()
v_accvgpr_write a9, 0.00000 // (op 466) call()
v_accvgpr_write a10, 0.00000 // (op 466) call()
v_accvgpr_write a11, 0.00000 // (op 466) call()
v_accvgpr_write a12, 0.00000 // (op 622) call()
v_accvgpr_write a13, 0.00000 // (op 622) call()
v_accvgpr_write a14, 0.00000 // (op 622) call()
v_accvgpr_write a15, 0.00000 // (op 622) call()
v_accvgpr_write a16, 0.00000 // (op 778) call()
v_accvgpr_write a17, 0.00000 // (op 778) call()
v_accvgpr_write a18, 0.00000 // (op 778) call()
v_accvgpr_write a19, 0.00000 // (op 778) call()
v_accvgpr_write a20, 0.00000 // (op 934) call()
v_accvgpr_write a21, 0.00000 // (op 934) call()
v_accvgpr_write a22, 0.00000 // (op 934) call()
v_accvgpr_write a23, 0.00000 // (op 934) call()
v_accvgpr_write a24, 0.00000 // (op 1090) call()
v_accvgpr_write a25, 0.00000 // (op 1090) call()
v_accvgpr_write a26, 0.00000 // (op 1090) call()
v_accvgpr_write a27, 0.00000 // (op 1090) call()
v_accvgpr_write a28, 0.00000 // (op 1246) call()
v_accvgpr_write a29, 0.00000 // (op 1246) call()
v_accvgpr_write a30, 0.00000 // (op 1246) call()
v_accvgpr_write a31, 0.00000 // (op 1246) call()
v_lshrrev_b32 v12, 3, v1 // (op 3459) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 3459) 

v_and_b32 v12, -8, v1 // (op 3459) 
v_sub_u32 v14, v1, v12 // (op 3459) 

v_lshlrev_b32 v12, 4, v14 // (op 3459) 

v_mov_b32 v15, 0 // (op 3459) convert
v_mov_b32 v14, v12 // (op 3459) convert

v_mul_lo_u32 v17, v13, s13 // (op 3459) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3459) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3459) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3459) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3459) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3459) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 3459) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3459) 

v_mov_b32 v12, v14 // (op 3459) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3461) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3461) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3461) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3461) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3461) most significant: sum
s_add_u32 s1, s1, s7 // (op 3461) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3461) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3461) 

s_mov_b32 s48, s20 // (op 3461) reg expression
s_mov_b32 s49, s21 // (op 3461) reg expression
s_mov_b32 s50, s0 // (op 3461) reg expression
s_mov_b32 s51, 131072 // (op 3461) call()

v_lshrrev_b32 v13, 3, v1 // (op 3478) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 3478) 

v_and_b32 v13, -8, v1 // (op 3478) 
v_sub_u32 v15, v1, v13 // (op 3478) 

v_lshlrev_b32 v13, 4, v15 // (op 3478) 

v_mov_b32 v17, 0 // (op 3478) convert
v_mov_b32 v16, v13 // (op 3478) convert

v_mul_lo_u32 v19, v14, s13 // (op 3478) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 3478) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 3478) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3478) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 3478) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 3478) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 3478) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3478) 

v_mov_b32 v13, v14 // (op 3478) convert

s_mul_i32 s1, s12, s23 // (op 3480) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3480) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3480) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3480) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3480) most significant: sum
s_add_u32 s1, s1, s7 // (op 3480) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3480) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3480) 

s_mov_b32 s48, s20 // (op 3480) reg expression
s_mov_b32 s49, s21 // (op 3480) reg expression
s_mov_b32 s50, s0 // (op 3480) reg expression
s_mov_b32 s51, 131072 // (op 3480) call()

v_and_b32 v14, -4, v1 // (op 3523) 
v_sub_u32 v15, v1, v14 // (op 3523) 

v_lshlrev_b32 v14, 5, v15 // (op 3523) 

v_mov_b32 v17, 0 // (op 3523) convert
v_mov_b32 v16, v14 // (op 3523) convert

v_and_b32 v14, 4294967292, v1 // (op 3523) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 3523) 

v_mul_lo_u32 v19, v15, s25 // (op 3523) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 3523) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 3523) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 3523) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3523) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3523) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 3523) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 3523) 

v_mov_b32 v14, v16 // (op 3523) convert

s_mul_i32 s1, s24, s9 // (op 3525) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3525) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3525) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3525) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3525) most significant: sum
s_add_u32 s1, s1, s7 // (op 3525) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3525) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3525) 

s_mov_b32 s52, s26 // (op 3525) reg expression
s_mov_b32 s53, s27 // (op 3525) reg expression
s_mov_b32 s54, s0 // (op 3525) reg expression
s_mov_b32 s55, 131072 // (op 3525) call()

v_and_b32 v15, -4, v1 // (op 3542) 
v_sub_u32 v16, v1, v15 // (op 3542) 

v_lshlrev_b32 v15, 5, v16 // (op 3542) 

v_mov_b32 v17, 0 // (op 3542) convert
v_mov_b32 v16, v15 // (op 3542) convert

v_and_b32 v15, 4294967292, v1 // (op 3542) 
v_lshl_add_u32 v18, s17, 8, v15 // (op 3542) 

v_mul_lo_u32 v21, v18, s25 // (op 3542) most significant: low of low * high
v_mul_hi_u32 v15, v18, s24 // (op 3542) most significant: high of low * low
v_mul_lo_u32 v20, v18, s24 // (op 3542) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3542) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v20 // (op 3542) least significant half
v_addc_co_u32 v19, s[0:1], v17, v21, s[0:1] // (op 3542) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 3542) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 3542) 

v_mov_b32 v15, v16 // (op 3542) convert

s_mul_i32 s1, s24, s9 // (op 3544) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3544) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3544) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3544) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3544) most significant: sum
s_add_u32 s1, s1, s7 // (op 3544) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3544) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3544) 

s_mov_b32 s52, s26 // (op 3544) reg expression
s_mov_b32 s53, s27 // (op 3544) reg expression
s_mov_b32 s54, s0 // (op 3544) reg expression
s_mov_b32 s55, 131072 // (op 3544) call()

v_lshrrev_b32 v16, 7, v1 // (op 3587) 
v_and_b32 v17, 15, v1 // (op 3587) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 3587) 


v_lshl_add_u32 v16, s16, 5, v18 // (op 3587) 

v_and_b32 v17, 63, v1 // (op 3587) 
v_lshrrev_b32 v18, 4, v17 // (op 3587) 

v_mul_lo_u32 v17, v16, s28 // (op 3587) 
v_add_u32 v17, v17, v18 // (op 3587) 


s_mul_i32 s1, s28, s23 // (op 3589) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3589) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3589) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3589) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3589) most significant: sum
s_add_u32 s1, s1, s7 // (op 3589) most significant: sum


s_mov_b32 s56, s30 // (op 3589) reg expression
s_mov_b32 s57, s31 // (op 3589) reg expression
s_mov_b32 s58, s0 // (op 3589) reg expression
s_mov_b32 s59, 131072 // (op 3589) call()

v_lshrrev_b32 v16, 7, v1 // (op 3606) 
v_and_b32 v18, 15, v1 // (op 3606) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 3606) 


v_lshl_add_u32 v16, s16, 5, v19 // (op 3606) 

v_and_b32 v18, 63, v1 // (op 3606) 
v_lshrrev_b32 v19, 4, v18 // (op 3606) 

v_mul_lo_u32 v18, v16, s28 // (op 3606) 
v_add_u32 v18, v18, v19 // (op 3606) 


s_mul_i32 s1, s28, s23 // (op 3608) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3608) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3608) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3608) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3608) most significant: sum
s_add_u32 s1, s1, s7 // (op 3608) most significant: sum


s_mov_b32 s56, s30 // (op 3608) reg expression
s_mov_b32 s57, s31 // (op 3608) reg expression
s_mov_b32 s58, s0 // (op 3608) reg expression
s_mov_b32 s59, 131072 // (op 3608) call()

v_lshl_add_u32 v16, s17, 8, v1 // (op 3651) 
v_mul_lo_u32 v19, v16, s32 // (op 3651) 

s_mul_i32 s1, s32, s9 // (op 3653) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3653) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3653) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3653) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3653) most significant: sum
s_add_u32 s1, s1, s7 // (op 3653) most significant: sum


s_mov_b32 s60, s34 // (op 3653) reg expression
s_mov_b32 s61, s35 // (op 3653) reg expression
s_mov_b32 s62, s0 // (op 3653) reg expression
s_mov_b32 s63, 131072 // (op 3653) call()

v_lshl_add_u32 v16, s17, 8, v1 // (op 3670) 
v_mul_lo_u32 v20, v16, s32 // (op 3670) 

s_mul_i32 s1, s32, s9 // (op 3672) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3672) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3672) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3672) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3672) most significant: sum
s_add_u32 s1, s1, s7 // (op 3672) most significant: sum


s_mov_b32 s60, s34 // (op 3672) reg expression
s_mov_b32 s61, s35 // (op 3672) reg expression
s_mov_b32 s62, s0 // (op 3672) reg expression
s_mov_b32 s63, 131072 // (op 3672) call()




buffer_load_ubyte v16, v17, s[56:59], 0 offen offset: 0 // (op 3140) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 3144) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 3144) 


v_mov_b32 v21, v14 // (op 3144) 
buffer_load_dwordx4 v[24:27], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[28:31], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[32:35], v21, s[52:55], 0 offen // (op 3144) Load value
v_add_u32 v21, v21, v22 // (op 3144) 
buffer_load_dwordx4 v[36:39], v21, s[52:55], 0 offen // (op 3144) Load value





buffer_load_dword v21, v19, s[60:63], 0 offen // (op 3147) Load value



buffer_load_dwordx2 v[22:23], v12, s[48:51], 0 offen // (op 3150) Load value



buffer_load_ubyte v40, v17, s[56:59], 0 offen offset:4 // (op 3154) Load value


s_lshl_b64 s[0:1], s[24:25], 2 // (op 3158) 
v_lshrrev_b64 v[42:43], 3, s[0:1] // (op 3158) 


v_mov_b32 v17, v14 // (op 3158) 
buffer_load_dwordx4 v[44:47], v17, s[52:55], 0 offen offset:64 // (op 3158) Load value
v_add_u32 v17, v17, v42 // (op 3158) 
buffer_load_dwordx4 v[48:51], v17, s[52:55], 0 offen offset:64 // (op 3158) Load value
v_add_u32 v17, v17, v42 // (op 3158) 
buffer_load_dwordx4 v[52:55], v17, s[52:55], 0 offen offset:64 // (op 3158) Load value
v_add_u32 v17, v17, v42 // (op 3158) 
buffer_load_dwordx4 v[56:59], v17, s[52:55], 0 offen offset:64 // (op 3158) Load value






buffer_load_dword v14, v19, s[60:63], 0 offen offset:4 // (op 3161) Load value




buffer_load_dwordx2 v[42:43], v12, s[48:51], 0 offen offset:64 // (op 3164) Load value

v_and_b32 v12, -4, v1 // (op 3790) 
v_sub_u32 v17, v1, v12 // (op 3790) 

v_lshlrev_b32 v12, 5, v17 // (op 3790) 

v_lshrrev_b32 v17, 2, v1 // (op 3790) 
v_lshl_add_u32 v19, v17, 9, v12 // (op 3790) 


v_lshlrev_b32 v12, 2, v19 // (op 3790) 

v_lshrrev_b32 v17, 3, v12 // (op 3790) 




s_waitcnt vmcnt(12)
ds_write_b128 v17, v[24:27] // (op 3166) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v17, v[28:31] offset:64 // (op 3166) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v17, v[32:35] offset:128 // (op 3166) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v17, v[36:39] offset:192 // (op 3166) Store local data 

v_lshlrev_b32 v12, 2, v1 // (op 3848) 



s_waitcnt vmcnt(8)
ds_write_b32 v12, v21 offset:16384 // (op 3167) Store local data 

v_and_b32 v12, -8, v1 // (op 3737) 
v_sub_u32 v17, v1, v12 // (op 3737) 

v_lshlrev_b32 v12, 4, v17 // (op 3737) 

v_lshrrev_b32 v17, 3, v1 // (op 3737) 
v_lshl_add_u32 v19, v17, 7, v12 // (op 3737) 


v_lshlrev_b32 v12, 2, v19 // (op 3737) 

v_lshrrev_b32 v17, 3, v12 // (op 3737) 




s_waitcnt vmcnt(7)
ds_write_b64 v17, v[22:23] offset:17408 // (op 3168) Store local data 

v_and_b32 v12, -8, v1 // (op 3708) 
v_sub_u32 v17, v1, v12 // (op 3708) 

v_lshlrev_b32 v12, 4, v17 // (op 3708) 

v_lshrrev_b32 v17, 3, v1 // (op 3708) 
v_lshl_add_u32 v19, v17, 7, v12 // (op 3708) 


v_lshlrev_b32 v12, 2, v19 // (op 3708) 

v_lshrrev_b32 v17, 3, v12 // (op 3708) 

v_and_b32 v12, -4, v1 // (op 3761) 
v_sub_u32 v19, v1, v12 // (op 3761) 

v_lshlrev_b32 v12, 5, v19 // (op 3761) 

v_lshrrev_b32 v19, 2, v1 // (op 3761) 
v_lshl_add_u32 v41, v19, 9, v12 // (op 3761) 


v_lshlrev_b32 v12, 2, v41 // (op 3761) 

v_lshrrev_b32 v19, 3, v12 // (op 3761) 

v_lshlrev_b32 v12, 2, v1 // (op 3819) 
v_and_b32 v41, -8, v1 // (op 3895) 
v_sub_u32 v60, v1, v41 // (op 3895) 

v_lshlrev_b32 v41, 4, v60 // (op 3895) 

v_lshrrev_b32 v60, 3, v1 // (op 3895) 
v_lshl_add_u32 v61, v60, 7, v41 // (op 3895) 


v_lshlrev_b32 v41, 2, v61 // (op 3895) 

v_lshrrev_b32 v60, 3, v41 // (op 3895) 

v_and_b32 v41, -4, v1 // (op 3919) 
v_sub_u32 v61, v1, v41 // (op 3919) 

v_lshlrev_b32 v41, 5, v61 // (op 3919) 

v_lshrrev_b32 v61, 2, v1 // (op 3919) 
v_lshl_add_u32 v62, v61, 9, v41 // (op 3919) 


v_lshlrev_b32 v41, 2, v62 // (op 3919) 

v_lshrrev_b32 v61, 3, v41 // (op 3919) 

v_lshlrev_b32 v41, 2, v1 // (op 3948) 
s_mov_b32 s0, 0 // (op 1248) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1247) 
s_mov_b32 s1, s6 // (op 1247) convert

s_and_b32 s6, s1, 4294967294 // (op 1247) 

s_cmp_lt_u32 s0, s6 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1247 // (op 1247) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1247 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1247:
s_add_u32 s0, s0, 2 // (op 1249) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4024) 



ds_read_b128 v[62:65], v3 offset:17408 // (op 67) Load local data 
s_barrier  // (op 4037) 



ds_read_b128 v[66:69], v5 // (op 72) Load local data 
s_barrier  // (op 4051) 



ds_read_u8 v70, v0 offset:16384 // (op 81) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[66:69], a[0:3], v16, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 84) 
ds_read_b128 v[72:75], v5 offset:2048 // (op 343) Load local data 





ds_read_u8 v66, v0 offset:16512 // (op 346) Load local data 



ds_read_b128 v[68:71], v5 offset:4096 // (op 499) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[72:75], a[4:7], v16, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 347) 





ds_read_u8 v66, v0 offset:16640 // (op 502) Load local data 



ds_read_b128 v[72:75], v5 offset:6144 // (op 655) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[68:71], a[8:11], v16, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 503) 





ds_read_u8 v66, v0 offset:16768 // (op 658) Load local data 



ds_read_b128 v[68:71], v5 offset:8192 // (op 811) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[72:75], a[12:15], v16, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 659) 





ds_read_u8 v66, v0 offset:16896 // (op 814) Load local data 



ds_read_b128 v[72:75], v5 offset:10240 // (op 967) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:65], v[68:71], a[16:19], v16, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 815) 





ds_read_u8 v66, v0 offset:17024 // (op 970) Load local data 



ds_read_b128 v[68:71], v5 offset:12288 // (op 1123) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[72:75], a[20:23], v16, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 971) 





ds_read_u8 v66, v0 offset:17152 // (op 1126) Load local data 



ds_read_b128 v[72:75], v5 offset:14336 // (op 1279) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[62:65], v[68:71], a[24:27], v16, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1127) 





ds_read_u8 v66, v0 offset:17280 // (op 1282) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[62:65], v[72:75], a[28:31], v16, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1283) 





v_mov_b32 v62, v18 // (op 77) 

buffer_load_ubyte v16, v62, s[56:59], 0 offen offset:8 // (op 77) Load value

s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[64:65], 3, s[6:7] // (op 10) 


v_mov_b32 v63, v15 // (op 10) 
v_mov_b32 v66, v63 // (op 10) 
buffer_load_dwordx4 v[24:27], v66, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v66, v66, v64 // (op 10) 
buffer_load_dwordx4 v[28:31], v66, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v66, v66, v64 // (op 10) 
buffer_load_dwordx4 v[32:35], v66, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v66, v66, v64 // (op 10) 
buffer_load_dwordx4 v[36:39], v66, s[52:55], 0 offen offset:128 // (op 10) Load value




v_mov_b32 v64, v20 // (op 21) 

buffer_load_dword v21, v64, s[60:63], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v65, v13 // (op 4) 

buffer_load_dwordx2 v[22:23], v65, s[48:51], 0 offen offset:128 // (op 4) Load value
s_barrier  // (op 4074) 



s_waitcnt vmcnt(12)
ds_write_b128 v61, v[44:47] offset:19456 // (op 239) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v61, v[48:51] offset:19520 // (op 239) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v61, v[52:55] offset:19584 // (op 239) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v61, v[56:59] offset:19648 // (op 239) Store local data 
s_barrier  // (op 4080) 



s_waitcnt vmcnt(8)
ds_write_b32 v41, v14 offset:35840 // (op 240) Store local data 
s_barrier  // (op 4068) 



s_waitcnt vmcnt(7)
ds_write_b64 v60, v[42:43] offset:36864 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4065) 



ds_read_b128 v[66:69], v4 offset:36864 // (op 231) Load local data 
s_barrier  // (op 4071) 
s_barrier  // (op 4121) 



ds_read_b128 v[70:73], v8 offset:19456 // (op 233) Load local data 
s_barrier  // (op 4077) 



ds_read_u8 v74, v7 offset:35840 // (op 237) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[70:73], a[0:3], v40, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 238) 
ds_read_b128 v[76:79], v8 offset:21504 // (op 352) Load local data 





ds_read_u8 v70, v7 offset:35968 // (op 355) Load local data 



ds_read_b128 v[72:75], v8 offset:23552 // (op 508) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[76:79], a[4:7], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 348) 





ds_read_u8 v70, v7 offset:36096 // (op 511) Load local data 



ds_read_b128 v[76:79], v8 offset:25600 // (op 664) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[72:75], a[8:11], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 504) 





ds_read_u8 v70, v7 offset:36224 // (op 667) Load local data 



ds_read_b128 v[72:75], v8 offset:27648 // (op 820) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[76:79], a[12:15], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 660) 





ds_read_u8 v70, v7 offset:36352 // (op 823) Load local data 



ds_read_b128 v[76:79], v8 offset:29696 // (op 976) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[66:69], v[72:75], a[16:19], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 816) 





ds_read_u8 v70, v7 offset:36480 // (op 979) Load local data 



ds_read_b128 v[72:75], v8 offset:31744 // (op 1132) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:69], v[76:79], a[20:23], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 972) 





ds_read_u8 v70, v7 offset:36608 // (op 1135) Load local data 



ds_read_b128 v[76:79], v8 offset:33792 // (op 1288) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:69], v[72:75], a[24:27], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1128) 





s_barrier  // (op 4191) 
ds_read_u8 v70, v7 offset:36736 // (op 1291) Load local data 
s_barrier  // (op 4195) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[66:69], v[76:79], a[28:31], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1284) 






buffer_load_ubyte v40, v62, s[56:59], 0 offen offset:12 // (op 235) Load value


v_add_u32 v18, 8, v18 // (op 3598) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 241) 
v_lshrrev_b64 v[66:67], 3, s[6:7] // (op 241) 


v_mov_b32 v62, v63 // (op 241) 
buffer_load_dwordx4 v[44:47], v62, s[52:55], 0 offen offset:192 // (op 241) Load value
v_add_u32 v62, v62, v66 // (op 241) 
buffer_load_dwordx4 v[48:51], v62, s[52:55], 0 offen offset:192 // (op 241) Load value
v_add_u32 v62, v62, v66 // (op 241) 
buffer_load_dwordx4 v[52:55], v62, s[52:55], 0 offen offset:192 // (op 241) Load value
v_add_u32 v62, v62, v66 // (op 241) 
buffer_load_dwordx4 v[56:59], v62, s[52:55], 0 offen offset:192 // (op 241) Load value






v_add_u32 v15, 128, v15 // (op 3534) 
buffer_load_dword v14, v64, s[60:63], 0 offen offset:12 // (op 242) Load value




v_add_u32 v20, 8, v20 // (op 3662) 
buffer_load_dwordx2 v[42:43], v65, s[48:51], 0 offen offset:192 // (op 228) Load value

s_barrier  // (op 4040) 
v_add_u32 v13, 128, v13 // (op 3470) 



s_waitcnt vmcnt(12)
ds_write_b128 v19, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v19, v[28:31] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v19, v[32:35] offset:128 // (op 13) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v19, v[36:39] offset:192 // (op 13) Store local data 
s_barrier  // (op 4054) 



s_waitcnt vmcnt(8)
ds_write_b32 v12, v21 offset:16384 // (op 26) Store local data 
s_barrier  // (op 4027) 



s_waitcnt vmcnt(7)
ds_write_b64 v17, v[22:23] offset:17408 // (op 6) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1247) 
s_mov_b32 s1, s6 // (op 1247) convert

s_and_b32 s6, s1, 4294967294 // (op 1247) 

s_cmp_lt_u32 s0, s6 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1247 // (op 1247) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1247 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1247:















v_lshrrev_b32 v41, 3, v1 // (op 3426) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3426) 
v_lshl_add_u32 v12, s16, 5, v41 // (op 3426) 

s_lshr_b32 s6, s0, 1 // (op 3426) 

v_and_b32 v13, -8, v1 // (op 3426) 
v_sub_u32 v41, v1, v13 // (op 3426) 

v_lshlrev_b32 v13, 4, v41 // (op 3426) 

v_lshl_add_u32 v41, s6, 8, v13 // (op 3426) 


v_mov_b32 v39, 0 // (op 3426) convert
v_mov_b32 v38, v41 // (op 3426) convert

v_mul_lo_u32 v37, v12, s13 // (op 3426) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 3426) most significant: high of low * low
v_mul_lo_u32 v36, v12, s12 // (op 3426) least significant: low of low * low
v_add_u32 v37, v37, v13 // (op 3426) most significant: sum

v_add_co_u32 v36, s[0:1], v36, v38 // (op 3426) least significant half
v_addc_co_u32 v37, s[0:1], v37, v39, s[0:1] // (op 3426) most significant half



v_lshlrev_b64 v[12:13], 2, v[36:37] // (op 3426) 

v_lshrrev_b64 v[38:39], 3, v[12:13] // (op 3426) 

v_mov_b32 v41, v38 // (op 3426) convert

s_mul_i32 s1, s12, s23 // (op 3428) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3428) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3428) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3428) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3428) most significant: sum
s_add_u32 s1, s1, s7 // (op 3428) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3428) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3428) 

s_mov_b32 s48, s20 // (op 3428) reg expression
s_mov_b32 s49, s21 // (op 3428) reg expression
s_mov_b32 s50, s0 // (op 3428) reg expression
s_mov_b32 s51, 131072 // (op 3428) call()

v_and_b32 v12, -4, v1 // (op 3500) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3500) 
v_sub_u32 v13, v1, v12 // (op 3500) 

s_lshr_b32 s6, s0, 1 // (op 3500) 

v_lshlrev_b32 v12, 5, v13 // (op 3500) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 3500) 


v_mov_b32 v39, 0 // (op 3500) convert
v_mov_b32 v38, v13 // (op 3500) convert

v_and_b32 v12, 4294967292, v1 // (op 3500) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3500) 

v_mul_lo_u32 v37, v13, s25 // (op 3500) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3500) most significant: high of low * low
v_mul_lo_u32 v36, v13, s24 // (op 3500) least significant: low of low * low
v_add_u32 v37, v37, v12 // (op 3500) most significant: sum


v_add_co_u32 v12, s[0:1], v38, v36 // (op 3500) least significant half
v_addc_co_u32 v13, s[0:1], v39, v37, s[0:1] // (op 3500) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 3500) 


v_lshrrev_b64 v[38:39], 3, v[12:13] // (op 3500) 

v_mov_b32 v12, v38 // (op 3500) convert

s_mul_i32 s1, s24, s9 // (op 3502) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3502) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3502) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3502) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3502) most significant: sum
s_add_u32 s1, s1, s7 // (op 3502) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3502) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3502) 

s_mov_b32 s52, s26 // (op 3502) reg expression
s_mov_b32 s53, s27 // (op 3502) reg expression
s_mov_b32 s54, s0 // (op 3502) reg expression
s_mov_b32 s55, 131072 // (op 3502) call()

v_lshrrev_b32 v13, 7, v1 // (op 3564) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3564) 
v_and_b32 v15, 15, v1 // (op 3564) 
s_lshr_b32 s6, s0, 1 // (op 3564) 

v_lshl_add_u32 v16, v13, 4, v15 // (op 3564) 


v_lshl_add_u32 v13, s16, 5, v16 // (op 3564) 

v_and_b32 v15, 63, v1 // (op 3564) 
v_lshrrev_b32 v16, 4, v15 // (op 3564) 

v_lshl_add_u32 v15, s6, 3, v16 // (op 3564) 


v_mul_lo_u32 v16, v13, s28 // (op 3564) 
v_add_u32 v16, v16, v15 // (op 3564) 


s_mul_i32 s1, s28, s23 // (op 3566) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3566) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3566) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3566) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3566) most significant: sum
s_add_u32 s1, s1, s7 // (op 3566) most significant: sum


s_mov_b32 s56, s30 // (op 3566) reg expression
s_mov_b32 s57, s31 // (op 3566) reg expression
s_mov_b32 s58, s0 // (op 3566) reg expression
s_mov_b32 s59, 131072 // (op 3566) call()

v_lshl_add_u32 v13, s17, 8, v1 // (op 3628) 
v_mul_lo_u32 v15, v13, s32 // (op 3628) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 3628) 
s_lshr_b32 s6, s0, 1 // (op 3628) 

v_lshl_add_u32 v13, s6, 3, v15 // (op 3628) 


s_mul_i32 s1, s32, s9 // (op 3630) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3630) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3630) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3630) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3630) most significant: sum
s_add_u32 s1, s1, s7 // (op 3630) most significant: sum


s_mov_b32 s60, s34 // (op 3630) reg expression
s_mov_b32 s61, s35 // (op 3630) reg expression
s_mov_b32 s62, s0 // (op 3630) reg expression
s_mov_b32 s63, 131072 // (op 3630) call()

v_and_b32 v15, -8, v1 // (op 3721) 
v_sub_u32 v17, v1, v15 // (op 3721) 

v_lshlrev_b32 v15, 4, v17 // (op 3721) 

v_lshrrev_b32 v17, 3, v1 // (op 3721) 
v_lshl_add_u32 v18, v17, 7, v15 // (op 3721) 


v_lshlrev_b32 v15, 2, v18 // (op 3721) 

v_lshrrev_b32 v17, 3, v15 // (op 3721) 

v_and_b32 v15, -4, v1 // (op 3774) 
v_sub_u32 v18, v1, v15 // (op 3774) 

v_lshlrev_b32 v15, 5, v18 // (op 3774) 

v_lshrrev_b32 v18, 2, v1 // (op 3774) 
v_lshl_add_u32 v19, v18, 9, v15 // (op 3774) 


v_lshlrev_b32 v15, 2, v19 // (op 3774) 

v_lshrrev_b32 v18, 3, v15 // (op 3774) 

v_lshlrev_b32 v15, 2, v1 // (op 3832) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1253) 
s_and_b32 s6, s0, 4294967294 // (op 1253) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 1252) 
s_mov_b32 s7, s0 // (op 1252) convert

s_cmp_lt_u32 s6, s7 // (op 1252) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1252 // (op 1252) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1252 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1252:


s_add_u32 s6, s6, 1 // (op 1254) 
v_mov_b32 v14, v41 // (op 170) 

buffer_load_dwordx2 v[20:21], v14, s[48:51], 0 offen // (op 170) Load value

v_add_u32 v41, 64, v41 // (op 3416) 

s_barrier  // (op 4030) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 183) 



v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 183) 


v_mov_b32 v14, v12 // (op 183) 
v_mov_b32 v19, v14 // (op 183) 
buffer_load_dwordx4 v[24:27], v19, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v19, v19, v22 // (op 183) 
buffer_load_dwordx4 v[28:31], v19, s[52:55], 0 offen // (op 183) Load value
v_add_u32 v19, v19, v22 // (op 183) 
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[20:21] offset:17408 // (op 171) Store local data 
buffer_load_dwordx4 v[32:35], v19, s[52:55], 0 offen // (op 183) Load value

v_add_u32 v19, v19, v22 // (op 183) 
buffer_load_dwordx4 v[36:39], v19, s[52:55], 0 offen // (op 183) Load value



v_add_u32 v12, 64, v12 // (op 3492) 


s_barrier  // (op 4043) 
v_mov_b32 v14, v13 // (op 184) 




buffer_load_dword v40, v14, s[60:63], 0 offen // (op 184) Load value

v_add_u32 v13, 4, v13 // (op 3620) 
s_waitcnt vmcnt(4)
ds_write_b128 v18, v[24:27] // (op 181) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4033) 
s_waitcnt vmcnt(3)
ds_write_b128 v18, v[28:31] offset:64 // (op 181) Store local data 



ds_read_b128 v[20:23], v3 offset:17408 // (op 173) Load local data 
s_waitcnt vmcnt(2)
ds_write_b128 v18, v[32:35] offset:128 // (op 181) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v18, v[36:39] offset:192 // (op 181) Store local data 

s_barrier  // (op 4057) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4047) 






ds_read_b128 v[24:27], v5 // (op 175) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v15, v40 offset:16384 // (op 182) Store local data 



v_mov_b32 v14, v16 // (op 177) 

buffer_load_ubyte v19, v14, s[56:59], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v16, 4, v16 // (op 3556) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4061) 



ds_read_u8 v14, v0 offset:16384 // (op 179) Load local data 



ds_read_b128 v[28:31], v5 offset:2048 // (op 324) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 180) 





ds_read_u8 v14, v0 offset:16512 // (op 327) Load local data 



ds_read_b128 v[24:27], v5 offset:4096 // (op 480) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[28:31], a[4:7], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 328) 





ds_read_u8 v14, v0 offset:16640 // (op 483) Load local data 



ds_read_b128 v[28:31], v5 offset:6144 // (op 636) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[24:27], a[8:11], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 484) 





ds_read_u8 v14, v0 offset:16768 // (op 639) Load local data 



ds_read_b128 v[24:27], v5 offset:8192 // (op 792) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[28:31], a[12:15], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 640) 





ds_read_u8 v14, v0 offset:16896 // (op 795) Load local data 



ds_read_b128 v[28:31], v5 offset:10240 // (op 948) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:23], v[24:27], a[16:19], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 796) 





ds_read_u8 v14, v0 offset:17024 // (op 951) Load local data 



ds_read_b128 v[24:27], v5 offset:12288 // (op 1104) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:23], v[28:31], a[20:23], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 952) 





ds_read_u8 v14, v0 offset:17152 // (op 1107) Load local data 



ds_read_b128 v[28:31], v5 offset:14336 // (op 1260) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:23], v[24:27], a[24:27], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1108) 





ds_read_u8 v14, v0 offset:17280 // (op 1263) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:23], v[28:31], a[28:31], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1264) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 1252) 
s_mov_b32 s7, s0 // (op 1252) convert

s_cmp_lt_u32 s6, s7 // (op 1252) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1252 // (op 1252) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1252 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1252:






















s_barrier  // (op 4179) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 

v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 
v_lshrrev_b32 v14, 16, v18 // (op 52) 
v_lshrrev_b32 v15, 16, v19 // (op 52) 


v_pack_b32_f16 v4, v12, v13 // (op 54) 
v_pack_b32_f16 v5, v14, v15 // (op 54) 

v_mov_b32 v0, v9 // (op 54) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 54) Store value



v_accvgpr_read v12, a4 // (op 1576) 
v_accvgpr_read v13, a5 // (op 1576) 
v_accvgpr_read v14, a6 // (op 1576) 
v_accvgpr_read v15, a7 // (op 1576) 
v_mul_f32 v16, v11, v12 // (op 1576) 
v_mul_f32 v17, v11, v13 // (op 1576) 
v_mul_f32 v18, v11, v14 // (op 1576) 
v_mul_f32 v19, v11, v15 // (op 1576) 

v_lshrrev_b32 v12, 16, v16 // (op 1578) 
v_lshrrev_b32 v13, 16, v17 // (op 1578) 
v_lshrrev_b32 v14, 16, v18 // (op 1578) 
v_lshrrev_b32 v15, 16, v19 // (op 1578) 


v_pack_b32_f16 v4, v12, v13 // (op 1579) 
v_pack_b32_f16 v5, v14, v15 // (op 1579) 

s_lshl_b32 s0, s4, 6 // (op 1579) 
v_add_u32 v0, v9, s0 // (op 1579) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1579) Store value



v_accvgpr_read v12, a8 // (op 1590) 
v_accvgpr_read v13, a9 // (op 1590) 
v_accvgpr_read v14, a10 // (op 1590) 
v_accvgpr_read v15, a11 // (op 1590) 
v_mul_f32 v16, v11, v12 // (op 1590) 
v_mul_f32 v17, v11, v13 // (op 1590) 
v_mul_f32 v18, v11, v14 // (op 1590) 
v_mul_f32 v19, v11, v15 // (op 1590) 

v_lshrrev_b32 v12, 16, v16 // (op 1592) 
v_lshrrev_b32 v13, 16, v17 // (op 1592) 
v_lshrrev_b32 v14, 16, v18 // (op 1592) 
v_lshrrev_b32 v15, 16, v19 // (op 1592) 


v_pack_b32_f16 v4, v12, v13 // (op 1593) 
v_pack_b32_f16 v5, v14, v15 // (op 1593) 

s_lshl_b32 s0, s4, 7 // (op 1593) 
v_add_u32 v0, v9, s0 // (op 1593) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1593) Store value



v_accvgpr_read v12, a12 // (op 1604) 
v_accvgpr_read v13, a13 // (op 1604) 
v_accvgpr_read v14, a14 // (op 1604) 
v_accvgpr_read v15, a15 // (op 1604) 
v_mul_f32 v16, v11, v12 // (op 1604) 
v_mul_f32 v17, v11, v13 // (op 1604) 
v_mul_f32 v18, v11, v14 // (op 1604) 
v_mul_f32 v19, v11, v15 // (op 1604) 

v_lshrrev_b32 v12, 16, v16 // (op 1606) 
v_lshrrev_b32 v13, 16, v17 // (op 1606) 
v_lshrrev_b32 v14, 16, v18 // (op 1606) 
v_lshrrev_b32 v15, 16, v19 // (op 1606) 


v_pack_b32_f16 v4, v12, v13 // (op 1607) 
v_pack_b32_f16 v5, v14, v15 // (op 1607) 

s_lshl_b32 s0, s4, 6 // (op 1607) 
s_mul_i32 s1, 3, s0 // (op 1607) 

v_add_u32 v0, v9, s1 // (op 1607) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1607) Store value



v_accvgpr_read v12, a16 // (op 1618) 
v_accvgpr_read v13, a17 // (op 1618) 
v_accvgpr_read v14, a18 // (op 1618) 
v_accvgpr_read v15, a19 // (op 1618) 
v_mul_f32 v16, v11, v12 // (op 1618) 
v_mul_f32 v17, v11, v13 // (op 1618) 
v_mul_f32 v18, v11, v14 // (op 1618) 
v_mul_f32 v19, v11, v15 // (op 1618) 

v_lshrrev_b32 v12, 16, v16 // (op 1620) 
v_lshrrev_b32 v13, 16, v17 // (op 1620) 
v_lshrrev_b32 v14, 16, v18 // (op 1620) 
v_lshrrev_b32 v15, 16, v19 // (op 1620) 


v_pack_b32_f16 v4, v12, v13 // (op 1621) 
v_pack_b32_f16 v5, v14, v15 // (op 1621) 

s_lshl_b32 s0, s4, 8 // (op 1621) 
v_add_u32 v0, v9, s0 // (op 1621) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1621) Store value



v_accvgpr_read v12, a20 // (op 1632) 
v_accvgpr_read v13, a21 // (op 1632) 
v_accvgpr_read v14, a22 // (op 1632) 
v_accvgpr_read v15, a23 // (op 1632) 
v_mul_f32 v16, v11, v12 // (op 1632) 
v_mul_f32 v17, v11, v13 // (op 1632) 
v_mul_f32 v18, v11, v14 // (op 1632) 
v_mul_f32 v19, v11, v15 // (op 1632) 

v_lshrrev_b32 v12, 16, v16 // (op 1634) 
v_lshrrev_b32 v13, 16, v17 // (op 1634) 
v_lshrrev_b32 v14, 16, v18 // (op 1634) 
v_lshrrev_b32 v15, 16, v19 // (op 1634) 


v_pack_b32_f16 v4, v12, v13 // (op 1635) 
v_pack_b32_f16 v5, v14, v15 // (op 1635) 

s_lshl_b32 s0, s4, 6 // (op 1635) 
s_mul_i32 s1, 5, s0 // (op 1635) 

v_add_u32 v0, v9, s1 // (op 1635) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1635) Store value



v_accvgpr_read v12, a24 // (op 1646) 
v_accvgpr_read v13, a25 // (op 1646) 
v_accvgpr_read v14, a26 // (op 1646) 
v_accvgpr_read v15, a27 // (op 1646) 
v_mul_f32 v16, v11, v12 // (op 1646) 
v_mul_f32 v17, v11, v13 // (op 1646) 
v_mul_f32 v18, v11, v14 // (op 1646) 
v_mul_f32 v19, v11, v15 // (op 1646) 

v_lshrrev_b32 v12, 16, v16 // (op 1648) 
v_lshrrev_b32 v13, 16, v17 // (op 1648) 
v_lshrrev_b32 v14, 16, v18 // (op 1648) 
v_lshrrev_b32 v15, 16, v19 // (op 1648) 


v_pack_b32_f16 v4, v12, v13 // (op 1649) 
v_pack_b32_f16 v5, v14, v15 // (op 1649) 

s_lshl_b32 s0, s4, 6 // (op 1649) 
s_mul_i32 s1, 6, s0 // (op 1649) 

v_add_u32 v0, v9, s1 // (op 1649) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1649) Store value



v_accvgpr_read v12, a28 // (op 1660) 
v_accvgpr_read v13, a29 // (op 1660) 
v_accvgpr_read v14, a30 // (op 1660) 
v_accvgpr_read v15, a31 // (op 1660) 
v_mul_f32 v16, v11, v12 // (op 1660) 
v_mul_f32 v17, v11, v13 // (op 1660) 
v_mul_f32 v18, v11, v14 // (op 1660) 
v_mul_f32 v19, v11, v15 // (op 1660) 

v_lshrrev_b32 v12, 16, v16 // (op 1662) 
v_lshrrev_b32 v13, 16, v17 // (op 1662) 
v_lshrrev_b32 v14, 16, v18 // (op 1662) 
v_lshrrev_b32 v15, 16, v19 // (op 1662) 


v_pack_b32_f16 v4, v12, v13 // (op 1663) 
v_pack_b32_f16 v5, v14, v15 // (op 1663) 

s_lshl_b32 s0, s4, 6 // (op 1663) 
s_mul_i32 s1, 7, s0 // (op 1663) 

v_add_u32 v0, v9, s1 // (op 1663) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1663) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 138) Load value

v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v11, v12 // (op 134) 
v_mul_f32 v17, v11, v13 // (op 134) 
v_mul_f32 v18, v11, v14 // (op 134) 
v_mul_f32 v19, v11, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 139) 
v_lshrrev_b32 v13, 16, v4 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v20, v10, v12 // (op 139) 
v_mul_f32 v21, v10, v13 // (op 139) 
v_and_b32 v12, 65535, v5 // (op 139) 
v_lshrrev_b32 v13, 16, v5 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v22, v10, v12 // (op 139) 
v_mul_f32 v23, v10, v13 // (op 139) 


v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 
v_add_f32 v14, v22, v18 // (op 135) 
v_add_f32 v15, v23, v19 // (op 135) 


v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 
v_lshrrev_b32 v18, 16, v14 // (op 136) 
v_lshrrev_b32 v19, 16, v15 // (op 136) 


v_pack_b32_f16 v4, v16, v17 // (op 137) 
v_pack_b32_f16 v5, v18, v19 // (op 137) 

v_mov_b32 v0, v9 // (op 137) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 1717) 
v_add_u32 v0, v6, s0 // (op 1717) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1717) Load value

v_accvgpr_read v12, a4 // (op 1713) 
v_accvgpr_read v13, a5 // (op 1713) 
v_accvgpr_read v14, a6 // (op 1713) 
v_accvgpr_read v15, a7 // (op 1713) 
v_mul_f32 v16, v11, v12 // (op 1713) 
v_mul_f32 v17, v11, v13 // (op 1713) 
v_mul_f32 v18, v11, v14 // (op 1713) 
v_mul_f32 v19, v11, v15 // (op 1713) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1718) 
v_lshrrev_b32 v13, 16, v4 // (op 1718) 
v_lshlrev_b32 v12, 16, v12 // (op 1718) 
v_lshlrev_b32 v13, 16, v13 // (op 1718) 
v_mul_f32 v20, v10, v12 // (op 1718) 
v_mul_f32 v21, v10, v13 // (op 1718) 
v_and_b32 v12, 65535, v5 // (op 1718) 
v_lshrrev_b32 v13, 16, v5 // (op 1718) 
v_lshlrev_b32 v12, 16, v12 // (op 1718) 
v_lshlrev_b32 v13, 16, v13 // (op 1718) 
v_mul_f32 v22, v10, v12 // (op 1718) 
v_mul_f32 v23, v10, v13 // (op 1718) 


v_add_f32 v12, v20, v16 // (op 1714) 
v_add_f32 v13, v21, v17 // (op 1714) 
v_add_f32 v14, v22, v18 // (op 1714) 
v_add_f32 v15, v23, v19 // (op 1714) 


v_lshrrev_b32 v16, 16, v12 // (op 1715) 
v_lshrrev_b32 v17, 16, v13 // (op 1715) 
v_lshrrev_b32 v18, 16, v14 // (op 1715) 
v_lshrrev_b32 v19, 16, v15 // (op 1715) 


v_pack_b32_f16 v4, v16, v17 // (op 1716) 
v_pack_b32_f16 v5, v18, v19 // (op 1716) 

s_lshl_b32 s0, s4, 6 // (op 1716) 
v_add_u32 v0, v9, s0 // (op 1716) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1716) Store value





s_lshl_b32 s0, s2, 7 // (op 1731) 
v_add_u32 v0, v6, s0 // (op 1731) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1731) Load value

v_accvgpr_read v12, a8 // (op 1727) 
v_accvgpr_read v13, a9 // (op 1727) 
v_accvgpr_read v14, a10 // (op 1727) 
v_accvgpr_read v15, a11 // (op 1727) 
v_mul_f32 v16, v11, v12 // (op 1727) 
v_mul_f32 v17, v11, v13 // (op 1727) 
v_mul_f32 v18, v11, v14 // (op 1727) 
v_mul_f32 v19, v11, v15 // (op 1727) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1732) 
v_lshrrev_b32 v13, 16, v4 // (op 1732) 
v_lshlrev_b32 v12, 16, v12 // (op 1732) 
v_lshlrev_b32 v13, 16, v13 // (op 1732) 
v_mul_f32 v20, v10, v12 // (op 1732) 
v_mul_f32 v21, v10, v13 // (op 1732) 
v_and_b32 v12, 65535, v5 // (op 1732) 
v_lshrrev_b32 v13, 16, v5 // (op 1732) 
v_lshlrev_b32 v12, 16, v12 // (op 1732) 
v_lshlrev_b32 v13, 16, v13 // (op 1732) 
v_mul_f32 v22, v10, v12 // (op 1732) 
v_mul_f32 v23, v10, v13 // (op 1732) 


v_add_f32 v12, v20, v16 // (op 1728) 
v_add_f32 v13, v21, v17 // (op 1728) 
v_add_f32 v14, v22, v18 // (op 1728) 
v_add_f32 v15, v23, v19 // (op 1728) 


v_lshrrev_b32 v16, 16, v12 // (op 1729) 
v_lshrrev_b32 v17, 16, v13 // (op 1729) 
v_lshrrev_b32 v18, 16, v14 // (op 1729) 
v_lshrrev_b32 v19, 16, v15 // (op 1729) 


v_pack_b32_f16 v4, v16, v17 // (op 1730) 
v_pack_b32_f16 v5, v18, v19 // (op 1730) 

s_lshl_b32 s0, s4, 7 // (op 1730) 
v_add_u32 v0, v9, s0 // (op 1730) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1730) Store value





s_lshl_b32 s0, s2, 6 // (op 1745) 
s_mul_i32 s1, 3, s0 // (op 1745) 

v_add_u32 v0, v6, s1 // (op 1745) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1745) Load value

v_accvgpr_read v12, a12 // (op 1741) 
v_accvgpr_read v13, a13 // (op 1741) 
v_accvgpr_read v14, a14 // (op 1741) 
v_accvgpr_read v15, a15 // (op 1741) 
v_mul_f32 v16, v11, v12 // (op 1741) 
v_mul_f32 v17, v11, v13 // (op 1741) 
v_mul_f32 v18, v11, v14 // (op 1741) 
v_mul_f32 v19, v11, v15 // (op 1741) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1746) 
v_lshrrev_b32 v13, 16, v4 // (op 1746) 
v_lshlrev_b32 v12, 16, v12 // (op 1746) 
v_lshlrev_b32 v13, 16, v13 // (op 1746) 
v_mul_f32 v20, v10, v12 // (op 1746) 
v_mul_f32 v21, v10, v13 // (op 1746) 
v_and_b32 v12, 65535, v5 // (op 1746) 
v_lshrrev_b32 v13, 16, v5 // (op 1746) 
v_lshlrev_b32 v12, 16, v12 // (op 1746) 
v_lshlrev_b32 v13, 16, v13 // (op 1746) 
v_mul_f32 v22, v10, v12 // (op 1746) 
v_mul_f32 v23, v10, v13 // (op 1746) 


v_add_f32 v12, v20, v16 // (op 1742) 
v_add_f32 v13, v21, v17 // (op 1742) 
v_add_f32 v14, v22, v18 // (op 1742) 
v_add_f32 v15, v23, v19 // (op 1742) 


v_lshrrev_b32 v16, 16, v12 // (op 1743) 
v_lshrrev_b32 v17, 16, v13 // (op 1743) 
v_lshrrev_b32 v18, 16, v14 // (op 1743) 
v_lshrrev_b32 v19, 16, v15 // (op 1743) 


v_pack_b32_f16 v4, v16, v17 // (op 1744) 
v_pack_b32_f16 v5, v18, v19 // (op 1744) 

s_lshl_b32 s0, s4, 6 // (op 1744) 
s_mul_i32 s1, 3, s0 // (op 1744) 

v_add_u32 v0, v9, s1 // (op 1744) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1744) Store value





s_lshl_b32 s0, s2, 8 // (op 1759) 
v_add_u32 v0, v6, s0 // (op 1759) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1759) Load value

v_accvgpr_read v12, a16 // (op 1755) 
v_accvgpr_read v13, a17 // (op 1755) 
v_accvgpr_read v14, a18 // (op 1755) 
v_accvgpr_read v15, a19 // (op 1755) 
v_mul_f32 v16, v11, v12 // (op 1755) 
v_mul_f32 v17, v11, v13 // (op 1755) 
v_mul_f32 v18, v11, v14 // (op 1755) 
v_mul_f32 v19, v11, v15 // (op 1755) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1760) 
v_lshrrev_b32 v13, 16, v4 // (op 1760) 
v_lshlrev_b32 v12, 16, v12 // (op 1760) 
v_lshlrev_b32 v13, 16, v13 // (op 1760) 
v_mul_f32 v20, v10, v12 // (op 1760) 
v_mul_f32 v21, v10, v13 // (op 1760) 
v_and_b32 v12, 65535, v5 // (op 1760) 
v_lshrrev_b32 v13, 16, v5 // (op 1760) 
v_lshlrev_b32 v12, 16, v12 // (op 1760) 
v_lshlrev_b32 v13, 16, v13 // (op 1760) 
v_mul_f32 v22, v10, v12 // (op 1760) 
v_mul_f32 v23, v10, v13 // (op 1760) 


v_add_f32 v12, v20, v16 // (op 1756) 
v_add_f32 v13, v21, v17 // (op 1756) 
v_add_f32 v14, v22, v18 // (op 1756) 
v_add_f32 v15, v23, v19 // (op 1756) 


v_lshrrev_b32 v16, 16, v12 // (op 1757) 
v_lshrrev_b32 v17, 16, v13 // (op 1757) 
v_lshrrev_b32 v18, 16, v14 // (op 1757) 
v_lshrrev_b32 v19, 16, v15 // (op 1757) 


v_pack_b32_f16 v4, v16, v17 // (op 1758) 
v_pack_b32_f16 v5, v18, v19 // (op 1758) 

s_lshl_b32 s0, s4, 8 // (op 1758) 
v_add_u32 v0, v9, s0 // (op 1758) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1758) Store value





s_lshl_b32 s0, s2, 6 // (op 1773) 
s_mul_i32 s1, 5, s0 // (op 1773) 

v_add_u32 v0, v6, s1 // (op 1773) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1773) Load value

v_accvgpr_read v12, a20 // (op 1769) 
v_accvgpr_read v13, a21 // (op 1769) 
v_accvgpr_read v14, a22 // (op 1769) 
v_accvgpr_read v15, a23 // (op 1769) 
v_mul_f32 v16, v11, v12 // (op 1769) 
v_mul_f32 v17, v11, v13 // (op 1769) 
v_mul_f32 v18, v11, v14 // (op 1769) 
v_mul_f32 v19, v11, v15 // (op 1769) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1774) 
v_lshrrev_b32 v13, 16, v4 // (op 1774) 
v_lshlrev_b32 v12, 16, v12 // (op 1774) 
v_lshlrev_b32 v13, 16, v13 // (op 1774) 
v_mul_f32 v20, v10, v12 // (op 1774) 
v_mul_f32 v21, v10, v13 // (op 1774) 
v_and_b32 v12, 65535, v5 // (op 1774) 
v_lshrrev_b32 v13, 16, v5 // (op 1774) 
v_lshlrev_b32 v12, 16, v12 // (op 1774) 
v_lshlrev_b32 v13, 16, v13 // (op 1774) 
v_mul_f32 v22, v10, v12 // (op 1774) 
v_mul_f32 v23, v10, v13 // (op 1774) 


v_add_f32 v12, v20, v16 // (op 1770) 
v_add_f32 v13, v21, v17 // (op 1770) 
v_add_f32 v14, v22, v18 // (op 1770) 
v_add_f32 v15, v23, v19 // (op 1770) 


v_lshrrev_b32 v16, 16, v12 // (op 1771) 
v_lshrrev_b32 v17, 16, v13 // (op 1771) 
v_lshrrev_b32 v18, 16, v14 // (op 1771) 
v_lshrrev_b32 v19, 16, v15 // (op 1771) 


v_pack_b32_f16 v4, v16, v17 // (op 1772) 
v_pack_b32_f16 v5, v18, v19 // (op 1772) 

s_lshl_b32 s0, s4, 6 // (op 1772) 
s_mul_i32 s1, 5, s0 // (op 1772) 

v_add_u32 v0, v9, s1 // (op 1772) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1772) Store value





s_lshl_b32 s0, s2, 6 // (op 1787) 
s_mul_i32 s1, 6, s0 // (op 1787) 

v_add_u32 v0, v6, s1 // (op 1787) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1787) Load value

v_accvgpr_read v12, a24 // (op 1783) 
v_accvgpr_read v13, a25 // (op 1783) 
v_accvgpr_read v14, a26 // (op 1783) 
v_accvgpr_read v15, a27 // (op 1783) 
v_mul_f32 v16, v11, v12 // (op 1783) 
v_mul_f32 v17, v11, v13 // (op 1783) 
v_mul_f32 v18, v11, v14 // (op 1783) 
v_mul_f32 v19, v11, v15 // (op 1783) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1788) 
v_lshrrev_b32 v13, 16, v4 // (op 1788) 
v_lshlrev_b32 v12, 16, v12 // (op 1788) 
v_lshlrev_b32 v13, 16, v13 // (op 1788) 
v_mul_f32 v20, v10, v12 // (op 1788) 
v_mul_f32 v21, v10, v13 // (op 1788) 
v_and_b32 v12, 65535, v5 // (op 1788) 
v_lshrrev_b32 v13, 16, v5 // (op 1788) 
v_lshlrev_b32 v12, 16, v12 // (op 1788) 
v_lshlrev_b32 v13, 16, v13 // (op 1788) 
v_mul_f32 v22, v10, v12 // (op 1788) 
v_mul_f32 v23, v10, v13 // (op 1788) 


v_add_f32 v12, v20, v16 // (op 1784) 
v_add_f32 v13, v21, v17 // (op 1784) 
v_add_f32 v14, v22, v18 // (op 1784) 
v_add_f32 v15, v23, v19 // (op 1784) 


v_lshrrev_b32 v16, 16, v12 // (op 1785) 
v_lshrrev_b32 v17, 16, v13 // (op 1785) 
v_lshrrev_b32 v18, 16, v14 // (op 1785) 
v_lshrrev_b32 v19, 16, v15 // (op 1785) 


v_pack_b32_f16 v4, v16, v17 // (op 1786) 
v_pack_b32_f16 v5, v18, v19 // (op 1786) 

s_lshl_b32 s0, s4, 6 // (op 1786) 
s_mul_i32 s1, 6, s0 // (op 1786) 

v_add_u32 v0, v9, s1 // (op 1786) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1786) Store value





s_lshl_b32 s0, s2, 6 // (op 1801) 
s_mul_i32 s1, 7, s0 // (op 1801) 

v_add_u32 v0, v6, s1 // (op 1801) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1801) Load value

v_accvgpr_read v12, a28 // (op 1797) 
v_accvgpr_read v13, a29 // (op 1797) 
v_accvgpr_read v14, a30 // (op 1797) 
v_accvgpr_read v15, a31 // (op 1797) 
v_mul_f32 v16, v11, v12 // (op 1797) 
v_mul_f32 v17, v11, v13 // (op 1797) 
v_mul_f32 v18, v11, v14 // (op 1797) 
v_mul_f32 v19, v11, v15 // (op 1797) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1802) 
v_lshrrev_b32 v13, 16, v4 // (op 1802) 
v_lshlrev_b32 v12, 16, v12 // (op 1802) 
v_lshlrev_b32 v13, 16, v13 // (op 1802) 
v_mul_f32 v20, v10, v12 // (op 1802) 
v_mul_f32 v21, v10, v13 // (op 1802) 
v_and_b32 v12, 65535, v5 // (op 1802) 
v_lshrrev_b32 v13, 16, v5 // (op 1802) 
v_lshlrev_b32 v12, 16, v12 // (op 1802) 
v_lshlrev_b32 v13, 16, v13 // (op 1802) 
v_mul_f32 v22, v10, v12 // (op 1802) 
v_mul_f32 v23, v10, v13 // (op 1802) 


v_add_f32 v12, v20, v16 // (op 1798) 
v_add_f32 v13, v21, v17 // (op 1798) 
v_add_f32 v14, v22, v18 // (op 1798) 
v_add_f32 v15, v23, v19 // (op 1798) 


v_lshrrev_b32 v16, 16, v12 // (op 1799) 
v_lshrrev_b32 v17, 16, v13 // (op 1799) 
v_lshrrev_b32 v18, 16, v14 // (op 1799) 
v_lshrrev_b32 v19, 16, v15 // (op 1799) 


v_pack_b32_f16 v4, v16, v17 // (op 1800) 
v_pack_b32_f16 v5, v18, v19 // (op 1800) 

s_lshl_b32 s0, s4, 6 // (op 1800) 
s_mul_i32 s1, 7, s0 // (op 1800) 

v_add_u32 v0, v9, s1 // (op 1800) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1800) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2, .LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end-RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
  .amdhsa_next_free_vgpr 112
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 38912 // lds bytes
  .amdhsa_accum_offset 80
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
    DestDataType: b
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
    macrotile: [32, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 38912
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 80
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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

