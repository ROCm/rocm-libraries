// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 3947) 
v_and_b32 v3, 1, v0 // (op 3947) 

v_and_b32 v0, 15, v1 // (op 3947) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3947) 


v_and_b32 v0, 63, v1 // (op 3947) 
v_lshrrev_b32 v3, 4, v0 // (op 3947) 

v_lshlrev_b32 v0, 5, v3 // (op 3947) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 3947) 


v_lshlrev_b32 v0, 2, v3 // (op 3947) 

v_lshrrev_b32 v3, 3, v0 // (op 3947) 

v_lshrrev_b32 v0, 6, v1 // (op 4076) 
v_and_b32 v4, 1, v0 // (op 4076) 

v_and_b32 v0, 15, v1 // (op 4076) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4076) 


v_and_b32 v0, 63, v1 // (op 4076) 
v_lshrrev_b32 v4, 4, v0 // (op 4076) 

v_lshlrev_b32 v0, 5, v4 // (op 4076) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 4076) 


v_lshlrev_b32 v0, 2, v4 // (op 4076) 

v_lshrrev_b32 v4, 3, v0 // (op 4076) 

v_lshrrev_b32 v0, 7, v1 // (op 3832) 
v_and_b32 v5, 63, v1 // (op 3832) 
v_lshrrev_b32 v6, 4, v5 // (op 3832) 
v_lshlrev_b32 v7, 2, v6 // (op 3832) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3832) 


v_lshl_add_u32 v0, s16, 8, v6 // (op 3832) 

v_lshrrev_b32 v6, 2, v5 // (op 3832) 

v_and_b32 v5, 3, v6 // (op 3832) 

v_and_b32 v6, 3, v1 // (op 3832) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3832) 


v_lshrrev_b32 v5, 6, v1 // (op 3832) 
v_and_b32 v6, 1, v5 // (op 3832) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3832) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 3832) 

v_mul_lo_u32 v5, v6, s2 // (op 3832) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 3832) 


v_lshrrev_b32 v0, 7, v1 // (op 3894) 
v_and_b32 v5, 15, v1 // (op 3894) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 3894) 


v_and_b32 v0, 63, v1 // (op 3894) 
v_lshrrev_b32 v5, 4, v0 // (op 3894) 

v_lshlrev_b32 v0, 5, v5 // (op 3894) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 3894) 


v_lshlrev_b32 v0, 2, v5 // (op 3894) 

v_lshrrev_b32 v5, 3, v0 // (op 3894) 

v_lshrrev_b32 v0, 7, v1 // (op 4005) 
v_and_b32 v7, 15, v1 // (op 4005) 
s_mul_i32 s1, s2, s9 // (op 3834) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3834) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3834) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3834) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3834) most significant: sum
s_add_u32 s1, s1, s19 // (op 3834) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3834) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 4005) 


v_and_b32 v0, 63, v1 // (op 4005) 
v_lshrrev_b32 v7, 4, v0 // (op 4005) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 4005) 


v_lshrrev_b32 v7, 7, v1 // (op 4021) 
v_and_b32 v8, 63, v1 // (op 4021) 
v_lshrrev_b32 v9, 4, v8 // (op 4021) 
v_lshlrev_b32 v10, 2, v9 // (op 4021) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 4021) 


v_lshl_add_u32 v7, s16, 8, v9 // (op 4021) 

v_lshrrev_b32 v9, 2, v8 // (op 4021) 

v_and_b32 v8, 3, v9 // (op 4021) 

v_and_b32 v9, 3, v1 // (op 4021) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 4021) 


v_lshrrev_b32 v8, 6, v1 // (op 4021) 
v_and_b32 v9, 1, v8 // (op 4021) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 4021) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 4021) 

v_mul_lo_u32 v8, v9, s4 // (op 4021) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 4021) 


v_lshrrev_b32 v7, 7, v1 // (op 4052) 
v_and_b32 v8, 15, v1 // (op 4052) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 4052) 


v_and_b32 v7, 63, v1 // (op 4052) 
v_lshrrev_b32 v8, 4, v7 // (op 4052) 

v_lshlrev_b32 v7, 5, v8 // (op 4052) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 4052) 


v_lshlrev_b32 v7, 2, v8 // (op 4052) 

v_lshrrev_b32 v8, 3, v7 // (op 4052) 

v_lshrrev_b32 v7, 7, v1 // (op 4105) 
v_and_b32 v10, 15, v1 // (op 4105) 
s_mul_i32 s1, s4, s9 // (op 4023) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 4105) 


v_and_b32 v7, 63, v1 // (op 4105) 
v_lshrrev_b32 v10, 4, v7 // (op 4105) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 4105) 


s_mul_i32 s38, s5, s8 // (op 4023) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 4023) least significant
s_mul_hi_u32 s39, s4, s8 // (op 4023) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 4023) most significant: sum
s_add_u32 s1, s1, s39 // (op 4023) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 4023) 

s_mov_b32 s40, s6 // (op 3834) reg expression
s_mov_b32 s41, s7 // (op 3834) reg expression
s_mov_b32 s42, s18 // (op 3834) reg expression
s_mov_b32 s43, 131072 // (op 3834) call()

s_mov_b32 s44, s10 // (op 4023) reg expression

s_mov_b32 s45, s11 // (op 4023) reg expression
s_mov_b32 s46, s38 // (op 4023) reg expression
s_mov_b32 s47, 131072 // (op 4023) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
v_accvgpr_write a8, 0.00000 // (op 475) call()
v_accvgpr_write a9, 0.00000 // (op 475) call()
v_accvgpr_write a10, 0.00000 // (op 475) call()
v_accvgpr_write a11, 0.00000 // (op 475) call()
v_accvgpr_write a12, 0.00000 // (op 637) call()
v_accvgpr_write a13, 0.00000 // (op 637) call()
v_accvgpr_write a14, 0.00000 // (op 637) call()
v_accvgpr_write a15, 0.00000 // (op 637) call()
v_accvgpr_write a16, 0.00000 // (op 799) call()
v_accvgpr_write a17, 0.00000 // (op 799) call()
v_accvgpr_write a18, 0.00000 // (op 799) call()
v_accvgpr_write a19, 0.00000 // (op 799) call()
v_accvgpr_write a20, 0.00000 // (op 961) call()
v_accvgpr_write a21, 0.00000 // (op 961) call()
v_accvgpr_write a22, 0.00000 // (op 961) call()
v_accvgpr_write a23, 0.00000 // (op 961) call()
v_accvgpr_write a24, 0.00000 // (op 1123) call()
v_accvgpr_write a25, 0.00000 // (op 1123) call()
v_accvgpr_write a26, 0.00000 // (op 1123) call()
v_accvgpr_write a27, 0.00000 // (op 1123) call()
v_accvgpr_write a28, 0.00000 // (op 1285) call()
v_accvgpr_write a29, 0.00000 // (op 1285) call()
v_accvgpr_write a30, 0.00000 // (op 1285) call()
v_accvgpr_write a31, 0.00000 // (op 1285) call()
v_and_b32 v12, 4294967292, v1 // (op 3599) 
v_lshl_add_u32 v13, s16, 8, v12 // (op 3599) 

v_and_b32 v12, -4, v1 // (op 3599) 
v_sub_u32 v14, v1, v12 // (op 3599) 

v_lshlrev_b32 v12, 5, v14 // (op 3599) 

v_mov_b32 v15, 0 // (op 3599) convert
v_mov_b32 v14, v12 // (op 3599) convert

v_mul_lo_u32 v17, v13, s13 // (op 3599) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3599) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3599) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3599) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3599) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3599) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 3599) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3599) 

v_mov_b32 v12, v14 // (op 3599) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3601) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3601) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3601) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3601) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3601) most significant: sum
s_add_u32 s1, s1, s7 // (op 3601) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3601) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3601) 

s_mov_b32 s48, s20 // (op 3601) reg expression
s_mov_b32 s49, s21 // (op 3601) reg expression
s_mov_b32 s50, s0 // (op 3601) reg expression
s_mov_b32 s51, 131072 // (op 3601) call()

v_and_b32 v13, 4294967292, v1 // (op 3618) 
v_lshl_add_u32 v14, s16, 8, v13 // (op 3618) 

v_and_b32 v13, -4, v1 // (op 3618) 
v_sub_u32 v15, v1, v13 // (op 3618) 

v_lshlrev_b32 v13, 5, v15 // (op 3618) 

v_mov_b32 v17, 0 // (op 3618) convert
v_mov_b32 v16, v13 // (op 3618) convert

v_mul_lo_u32 v19, v14, s13 // (op 3618) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 3618) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 3618) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3618) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 3618) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 3618) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 3618) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3618) 

v_mov_b32 v13, v14 // (op 3618) convert

s_mul_i32 s1, s12, s23 // (op 3620) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3620) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3620) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3620) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3620) most significant: sum
s_add_u32 s1, s1, s7 // (op 3620) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3620) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3620) 

s_mov_b32 s48, s20 // (op 3620) reg expression
s_mov_b32 s49, s21 // (op 3620) reg expression
s_mov_b32 s50, s0 // (op 3620) reg expression
s_mov_b32 s51, 131072 // (op 3620) call()

v_and_b32 v14, -8, v1 // (op 3663) 
v_sub_u32 v15, v1, v14 // (op 3663) 

v_lshlrev_b32 v14, 4, v15 // (op 3663) 

v_mov_b32 v17, 0 // (op 3663) convert
v_mov_b32 v16, v14 // (op 3663) convert

v_lshrrev_b32 v14, 3, v1 // (op 3663) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 3663) 

v_mul_lo_u32 v19, v15, s25 // (op 3663) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 3663) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 3663) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 3663) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3663) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3663) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 3663) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 3663) 

v_mov_b32 v14, v16 // (op 3663) convert

s_mul_i32 s1, s24, s9 // (op 3665) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3665) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3665) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3665) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3665) most significant: sum
s_add_u32 s1, s1, s7 // (op 3665) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3665) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3665) 

s_mov_b32 s52, s26 // (op 3665) reg expression
s_mov_b32 s53, s27 // (op 3665) reg expression
s_mov_b32 s54, s0 // (op 3665) reg expression
s_mov_b32 s55, 131072 // (op 3665) call()

v_and_b32 v15, -8, v1 // (op 3682) 
v_sub_u32 v16, v1, v15 // (op 3682) 

v_lshlrev_b32 v15, 4, v16 // (op 3682) 

v_mov_b32 v17, 0 // (op 3682) convert
v_mov_b32 v16, v15 // (op 3682) convert

v_lshrrev_b32 v15, 3, v1 // (op 3682) 
v_lshl_add_u32 v18, s17, 5, v15 // (op 3682) 

v_mul_lo_u32 v21, v18, s25 // (op 3682) most significant: low of low * high
v_mul_hi_u32 v15, v18, s24 // (op 3682) most significant: high of low * low
v_mul_lo_u32 v20, v18, s24 // (op 3682) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3682) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v20 // (op 3682) least significant half
v_addc_co_u32 v19, s[0:1], v17, v21, s[0:1] // (op 3682) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 3682) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 3682) 

v_mov_b32 v15, v16 // (op 3682) convert

s_mul_i32 s1, s24, s9 // (op 3684) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3684) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3684) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3684) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3684) most significant: sum
s_add_u32 s1, s1, s7 // (op 3684) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3684) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3684) 

s_mov_b32 s52, s26 // (op 3684) reg expression
s_mov_b32 s53, s27 // (op 3684) reg expression
s_mov_b32 s54, s0 // (op 3684) reg expression
s_mov_b32 s55, 131072 // (op 3684) call()

v_lshl_add_u32 v16, s16, 8, v1 // (op 3727) 
v_mul_lo_u32 v17, v16, s28 // (op 3727) 

s_mul_i32 s1, s28, s23 // (op 3729) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3729) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3729) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3729) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3729) most significant: sum
s_add_u32 s1, s1, s7 // (op 3729) most significant: sum


s_mov_b32 s56, s30 // (op 3729) reg expression
s_mov_b32 s57, s31 // (op 3729) reg expression
s_mov_b32 s58, s0 // (op 3729) reg expression
s_mov_b32 s59, 131072 // (op 3729) call()

v_lshl_add_u32 v16, s16, 8, v1 // (op 3746) 
v_mul_lo_u32 v18, v16, s28 // (op 3746) 

s_mul_i32 s1, s28, s23 // (op 3748) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3748) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3748) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3748) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3748) most significant: sum
s_add_u32 s1, s1, s7 // (op 3748) most significant: sum


s_mov_b32 s56, s30 // (op 3748) reg expression
s_mov_b32 s57, s31 // (op 3748) reg expression
s_mov_b32 s58, s0 // (op 3748) reg expression
s_mov_b32 s59, 131072 // (op 3748) call()

v_lshrrev_b32 v16, 6, v1 // (op 3791) 
v_and_b32 v19, 1, v16 // (op 3791) 

v_and_b32 v16, 15, v1 // (op 3791) 
v_lshl_add_u32 v20, v19, 4, v16 // (op 3791) 


v_lshl_add_u32 v16, s17, 5, v20 // (op 3791) 

v_mul_lo_u32 v19, v16, s32 // (op 3791) 

v_and_b32 v16, 63, v1 // (op 3791) 
v_lshrrev_b32 v20, 4, v16 // (op 3791) 

v_add_u32 v16, v20, v19 // (op 3791) 


s_mul_i32 s1, s32, s9 // (op 3793) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3793) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3793) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3793) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3793) most significant: sum
s_add_u32 s1, s1, s7 // (op 3793) most significant: sum


s_mov_b32 s60, s34 // (op 3793) reg expression
s_mov_b32 s61, s35 // (op 3793) reg expression
s_mov_b32 s62, s0 // (op 3793) reg expression
s_mov_b32 s63, 131072 // (op 3793) call()

v_lshrrev_b32 v19, 6, v1 // (op 3810) 
v_and_b32 v20, 1, v19 // (op 3810) 

v_and_b32 v19, 15, v1 // (op 3810) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 3810) 


v_lshl_add_u32 v19, s17, 5, v21 // (op 3810) 

v_mul_lo_u32 v20, v19, s32 // (op 3810) 

v_and_b32 v19, 63, v1 // (op 3810) 
v_lshrrev_b32 v21, 4, v19 // (op 3810) 

v_add_u32 v19, v21, v20 // (op 3810) 


s_mul_i32 s1, s32, s9 // (op 3812) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3812) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3812) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3812) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3812) most significant: sum
s_add_u32 s1, s1, s7 // (op 3812) most significant: sum


s_mov_b32 s60, s34 // (op 3812) reg expression
s_mov_b32 s61, s35 // (op 3812) reg expression
s_mov_b32 s62, s0 // (op 3812) reg expression
s_mov_b32 s63, 131072 // (op 3812) call()




buffer_load_ubyte v20, v16, s[60:63], 0 offen offset: 0 // (op 3284) Load value



buffer_load_dwordx2 v[22:23], v14, s[52:55], 0 offen // (op 3288) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 3291) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3291) 


v_mov_b32 v21, v12 // (op 3291) 
buffer_load_dwordx4 v[26:29], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[30:33], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[34:37], v21, s[48:51], 0 offen // (op 3291) Load value
v_add_u32 v21, v21, v24 // (op 3291) 
buffer_load_dwordx4 v[38:41], v21, s[48:51], 0 offen // (op 3291) Load value





buffer_load_dword v21, v17, s[56:59], 0 offen // (op 3294) Load value



buffer_load_ubyte v24, v16, s[60:63], 0 offen offset:4 // (op 3298) Load value




buffer_load_dwordx2 v[42:43], v14, s[52:55], 0 offen offset:64 // (op 3302) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 3305) 
v_lshrrev_b64 v[44:45], 3, s[0:1] // (op 3305) 


v_mov_b32 v14, v12 // (op 3305) 
buffer_load_dwordx4 v[46:49], v14, s[48:51], 0 offen offset:64 // (op 3305) Load value
v_add_u32 v14, v14, v44 // (op 3305) 
buffer_load_dwordx4 v[50:53], v14, s[48:51], 0 offen offset:64 // (op 3305) Load value
v_add_u32 v14, v14, v44 // (op 3305) 
buffer_load_dwordx4 v[54:57], v14, s[48:51], 0 offen offset:64 // (op 3305) Load value
v_add_u32 v14, v14, v44 // (op 3305) 
buffer_load_dwordx4 v[58:61], v14, s[48:51], 0 offen offset:64 // (op 3305) Load value






buffer_load_dword v12, v17, s[56:59], 0 offen offset:4 // (op 3308) Load value

v_and_b32 v14, -8, v1 // (op 3935) 
v_sub_u32 v25, v1, v14 // (op 3935) 

v_lshlrev_b32 v14, 4, v25 // (op 3935) 

v_lshrrev_b32 v25, 3, v1 // (op 3935) 
v_lshl_add_u32 v16, v25, 7, v14 // (op 3935) 


v_lshlrev_b32 v14, 2, v16 // (op 3935) 

v_lshrrev_b32 v25, 3, v14 // (op 3935) 




s_waitcnt vmcnt(12)
ds_write_b64 v25, v[22:23] // (op 3310) Store local data 

v_and_b32 v14, -4, v1 // (op 3877) 
v_sub_u32 v25, v1, v14 // (op 3877) 

v_lshlrev_b32 v14, 5, v25 // (op 3877) 

v_lshrrev_b32 v25, 2, v1 // (op 3877) 
v_lshl_add_u32 v16, v25, 9, v14 // (op 3877) 


v_lshlrev_b32 v14, 2, v16 // (op 3877) 

v_lshrrev_b32 v25, 3, v14 // (op 3877) 




s_waitcnt vmcnt(11)
ds_write_b128 v25, v[26:29] offset:2048 // (op 3311) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v25, v[30:33] offset:2112 // (op 3311) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v25, v[34:37] offset:2176 // (op 3311) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v25, v[38:41] offset:2240 // (op 3311) Store local data 

v_lshlrev_b32 v14, 2, v1 // (op 3988) 



s_waitcnt vmcnt(7)
ds_write_b32 v14, v21 offset:18432 // (op 3312) Store local data 

v_and_b32 v14, -4, v1 // (op 3848) 
v_sub_u32 v25, v1, v14 // (op 3848) 

v_lshlrev_b32 v14, 5, v25 // (op 3848) 

v_lshrrev_b32 v25, 2, v1 // (op 3848) 
v_lshl_add_u32 v16, v25, 9, v14 // (op 3848) 


v_lshlrev_b32 v14, 2, v16 // (op 3848) 

v_lshrrev_b32 v25, 3, v14 // (op 3848) 

v_and_b32 v14, -8, v1 // (op 3906) 
v_sub_u32 v16, v1, v14 // (op 3906) 

v_lshlrev_b32 v14, 4, v16 // (op 3906) 

v_lshrrev_b32 v16, 3, v1 // (op 3906) 
v_lshl_add_u32 v17, v16, 7, v14 // (op 3906) 


v_lshlrev_b32 v14, 2, v17 // (op 3906) 

v_lshrrev_b32 v16, 3, v14 // (op 3906) 

v_lshlrev_b32 v14, 2, v1 // (op 3959) 
v_and_b32 v17, -4, v1 // (op 4035) 
v_sub_u32 v44, v1, v17 // (op 4035) 

v_lshlrev_b32 v17, 5, v44 // (op 4035) 

v_lshrrev_b32 v44, 2, v1 // (op 4035) 
v_lshl_add_u32 v45, v44, 9, v17 // (op 4035) 


v_lshlrev_b32 v17, 2, v45 // (op 4035) 

v_lshrrev_b32 v44, 3, v17 // (op 4035) 

v_and_b32 v17, -8, v1 // (op 4064) 
v_sub_u32 v45, v1, v17 // (op 4064) 

v_lshlrev_b32 v17, 4, v45 // (op 4064) 

v_lshrrev_b32 v45, 3, v1 // (op 4064) 
v_lshl_add_u32 v62, v45, 7, v17 // (op 4064) 


v_lshlrev_b32 v17, 2, v62 // (op 4064) 

v_lshrrev_b32 v45, 3, v17 // (op 4064) 

v_lshlrev_b32 v17, 2, v1 // (op 4088) 
s_mov_b32 s0, 0 // (op 1287) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1286) 
s_mov_b32 s1, s6 // (op 1286) convert

s_and_b32 s6, s1, 4294967294 // (op 1286) 

s_cmp_lt_u32 s0, s6 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1286 // (op 1286) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1286 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1286:
s_add_u32 s0, s0, 2 // (op 1288) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4164) 



ds_read_b128 v[62:65], v5 offset:2048 // (op 67) Load local data 
s_barrier  // (op 4177) 



ds_read_b128 v[66:69], v3 // (op 72) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4191) 



ds_read_u8 v70, v0 offset:18432 // (op 77) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[66:69], a[0:3], v70, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 84) 
ds_read_b128 v[72:75], v5 offset:4096 // (op 344) Load local data 





ds_read_u8 v62, v0 offset:18560 // (op 348) Load local data 



ds_read_b128 v[76:79], v5 offset:6144 // (op 506) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[66:69], a[4:7], v62, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 350) 





ds_read_u8 v62, v0 offset:18688 // (op 510) Load local data 



ds_read_b128 v[70:73], v5 offset:8192 // (op 668) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[66:69], a[8:11], v62, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 512) 





ds_read_u8 v62, v0 offset:18816 // (op 672) Load local data 



ds_read_b128 v[74:77], v5 offset:10240 // (op 830) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[66:69], a[12:15], v62, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 674) 





ds_read_u8 v62, v0 offset:18944 // (op 834) Load local data 



ds_read_b128 v[70:73], v5 offset:12288 // (op 992) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[66:69], a[16:19], v62, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 836) 





ds_read_u8 v62, v0 offset:19072 // (op 996) Load local data 



ds_read_b128 v[74:77], v5 offset:14336 // (op 1154) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[66:69], a[20:23], v62, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 998) 





ds_read_u8 v62, v0 offset:19200 // (op 1158) Load local data 



ds_read_b128 v[70:73], v5 offset:16384 // (op 1316) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[66:69], a[24:27], v62, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1160) 





ds_read_u8 v62, v0 offset:19328 // (op 1320) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[66:69], a[28:31], v62, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1322) 





v_mov_b32 v62, v19 // (op 82) 

buffer_load_ubyte v20, v62, s[60:63], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v63, v15 // (op 10) 

buffer_load_dwordx2 v[22:23], v63, s[52:55], 0 offen offset:128 // (op 10) Load value

s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 
v_lshrrev_b64 v[64:65], 3, s[6:7] // (op 4) 


v_mov_b32 v66, v13 // (op 4) 
v_mov_b32 v67, v66 // (op 4) 
buffer_load_dwordx4 v[26:29], v67, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v67, v67, v64 // (op 4) 
buffer_load_dwordx4 v[30:33], v67, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v67, v67, v64 // (op 4) 
buffer_load_dwordx4 v[34:37], v67, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v67, v67, v64 // (op 4) 
buffer_load_dwordx4 v[38:41], v67, s[48:51], 0 offen offset:128 // (op 4) Load value




v_mov_b32 v64, v18 // (op 17) 

buffer_load_dword v21, v64, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 4214) 



s_waitcnt vmcnt(12)
ds_write_b64 v45, v[42:43] offset:19456 // (op 239) Store local data 
s_barrier  // (op 4208) 



s_waitcnt vmcnt(11)
ds_write_b128 v44, v[46:49] offset:21504 // (op 229) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v44, v[50:53] offset:21568 // (op 229) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v44, v[54:57] offset:21632 // (op 229) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v44, v[58:61] offset:21696 // (op 229) Store local data 
s_barrier  // (op 4220) 



s_waitcnt vmcnt(7)
ds_write_b32 v17, v12 offset:37888 // (op 240) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4205) 



ds_read_b128 v[68:71], v8 offset:21504 // (op 231) Load local data 
s_barrier  // (op 4211) 



ds_read_b128 v[72:75], v4 offset:19456 // (op 233) Load local data 
s_barrier  // (op 4261) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4217) 



ds_read_u8 v65, v7 offset:37888 // (op 235) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:71], v[72:75], a[0:3], v65, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 238) 
ds_read_b128 v[76:79], v8 offset:23552 // (op 357) Load local data 





ds_read_u8 v65, v7 offset:38016 // (op 361) Load local data 



ds_read_b128 v[68:71], v8 offset:25600 // (op 519) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[72:75], a[4:7], v65, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 351) 





ds_read_u8 v65, v7 offset:38144 // (op 523) Load local data 



ds_read_b128 v[76:79], v8 offset:27648 // (op 681) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:71], v[72:75], a[8:11], v65, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 513) 





ds_read_u8 v65, v7 offset:38272 // (op 685) Load local data 



ds_read_b128 v[68:71], v8 offset:29696 // (op 843) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[72:75], a[12:15], v65, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 675) 





ds_read_u8 v65, v7 offset:38400 // (op 847) Load local data 



ds_read_b128 v[76:79], v8 offset:31744 // (op 1005) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:71], v[72:75], a[16:19], v65, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 837) 





ds_read_u8 v65, v7 offset:38528 // (op 1009) Load local data 



ds_read_b128 v[68:71], v8 offset:33792 // (op 1167) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[72:75], a[20:23], v65, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 999) 





ds_read_u8 v65, v7 offset:38656 // (op 1171) Load local data 



ds_read_b128 v[76:79], v8 offset:35840 // (op 1329) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:71], v[72:75], a[24:27], v65, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1161) 





s_barrier  // (op 4331) 
ds_read_u8 v65, v7 offset:38784 // (op 1333) Load local data 
s_barrier  // (op 4335) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[72:75], a[28:31], v65, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1323) 






buffer_load_ubyte v24, v62, s[60:63], 0 offen offset:12 // (op 237) Load value




v_add_u32 v19, 8, v19 // (op 3802) 
buffer_load_dwordx2 v[42:43], v63, s[52:55], 0 offen offset:192 // (op 241) Load value


v_add_u32 v15, 128, v15 // (op 3674) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 228) 
v_lshrrev_b64 v[62:63], 3, s[6:7] // (op 228) 


v_mov_b32 v65, v66 // (op 228) 
buffer_load_dwordx4 v[46:49], v65, s[48:51], 0 offen offset:192 // (op 228) Load value
v_add_u32 v65, v65, v62 // (op 228) 
buffer_load_dwordx4 v[50:53], v65, s[48:51], 0 offen offset:192 // (op 228) Load value
v_add_u32 v65, v65, v62 // (op 228) 
buffer_load_dwordx4 v[54:57], v65, s[48:51], 0 offen offset:192 // (op 228) Load value
v_add_u32 v65, v65, v62 // (op 228) 
buffer_load_dwordx4 v[58:61], v65, s[48:51], 0 offen offset:192 // (op 228) Load value






v_add_u32 v13, 128, v13 // (op 3610) 
buffer_load_dword v12, v64, s[56:59], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 4180) 
v_add_u32 v18, 8, v18 // (op 3738) 



s_waitcnt vmcnt(12)
ds_write_b64 v16, v[22:23] // (op 13) Store local data 
s_barrier  // (op 4167) 



s_waitcnt vmcnt(11)
ds_write_b128 v25, v[26:29] offset:2048 // (op 6) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v25, v[30:33] offset:2112 // (op 6) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v25, v[34:37] offset:2176 // (op 6) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v25, v[38:41] offset:2240 // (op 6) Store local data 
s_barrier  // (op 4194) 



s_waitcnt vmcnt(7)
ds_write_b32 v14, v21 offset:18432 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1286) 
s_mov_b32 s1, s6 // (op 1286) convert

s_and_b32 s6, s1, 4294967294 // (op 1286) 

s_cmp_lt_u32 s0, s6 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1286 // (op 1286) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1286 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1286:















v_and_b32 v13, 4294967292, v1 // (op 3566) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3566) 
v_lshl_add_u32 v14, s16, 8, v13 // (op 3566) 

s_lshr_b32 s6, s0, 1 // (op 3566) 

v_and_b32 v13, -4, v1 // (op 3566) 
v_sub_u32 v15, v1, v13 // (op 3566) 

v_lshlrev_b32 v13, 5, v15 // (op 3566) 

v_lshl_add_u32 v15, s6, 8, v13 // (op 3566) 


v_mov_b32 v45, 0 // (op 3566) convert
v_mov_b32 v44, v15 // (op 3566) convert

v_mul_lo_u32 v23, v14, s13 // (op 3566) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 3566) most significant: high of low * low
v_mul_lo_u32 v22, v14, s12 // (op 3566) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 3566) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v44 // (op 3566) least significant half
v_addc_co_u32 v23, s[0:1], v23, v45, s[0:1] // (op 3566) most significant half



v_lshlrev_b64 v[44:45], 2, v[22:23] // (op 3566) 

v_lshrrev_b64 v[22:23], 3, v[44:45] // (op 3566) 

v_mov_b32 v13, v22 // (op 3566) convert

s_mul_i32 s1, s12, s23 // (op 3568) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3568) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3568) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3568) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3568) most significant: sum
s_add_u32 s1, s1, s7 // (op 3568) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3568) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3568) 

s_mov_b32 s48, s20 // (op 3568) reg expression
s_mov_b32 s49, s21 // (op 3568) reg expression
s_mov_b32 s50, s0 // (op 3568) reg expression
s_mov_b32 s51, 131072 // (op 3568) call()

v_and_b32 v14, -8, v1 // (op 3640) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3640) 
v_sub_u32 v15, v1, v14 // (op 3640) 

s_lshr_b32 s6, s0, 1 // (op 3640) 

v_lshlrev_b32 v14, 4, v15 // (op 3640) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 3640) 


v_mov_b32 v45, 0 // (op 3640) convert
v_mov_b32 v44, v15 // (op 3640) convert

v_lshrrev_b32 v14, 3, v1 // (op 3640) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 3640) 

v_mul_lo_u32 v17, v15, s25 // (op 3640) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 3640) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 3640) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 3640) most significant: sum


v_add_co_u32 v14, s[0:1], v44, v16 // (op 3640) least significant half
v_addc_co_u32 v15, s[0:1], v45, v17, s[0:1] // (op 3640) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 3640) 


v_lshrrev_b64 v[44:45], 3, v[14:15] // (op 3640) 

v_mov_b32 v14, v44 // (op 3640) convert

s_mul_i32 s1, s24, s9 // (op 3642) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3642) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3642) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3642) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3642) most significant: sum
s_add_u32 s1, s1, s7 // (op 3642) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3642) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3642) 

s_mov_b32 s52, s26 // (op 3642) reg expression
s_mov_b32 s53, s27 // (op 3642) reg expression
s_mov_b32 s54, s0 // (op 3642) reg expression
s_mov_b32 s55, 131072 // (op 3642) call()

v_lshl_add_u32 v15, s16, 8, v1 // (op 3704) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3704) 
s_lshr_b32 s6, s0, 1 // (op 3704) 

s_lshl_b32 s7, s6, 3 // (op 3704) 

v_mul_lo_u32 v16, v15, s28 // (op 3704) 
v_add_u32 v16, v16, s7 // (op 3704) 


s_mul_i32 s1, s28, s23 // (op 3706) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3706) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3706) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3706) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3706) most significant: sum
s_add_u32 s1, s1, s7 // (op 3706) most significant: sum


s_mov_b32 s56, s30 // (op 3706) reg expression
s_mov_b32 s57, s31 // (op 3706) reg expression
s_mov_b32 s58, s0 // (op 3706) reg expression
s_mov_b32 s59, 131072 // (op 3706) call()

v_and_b32 v15, 63, v1 // (op 3768) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3768) 
v_lshrrev_b32 v17, 4, v15 // (op 3768) 

s_lshr_b32 s6, s0, 1 // (op 3768) 

v_lshrrev_b32 v15, 6, v1 // (op 3768) 
v_lshl_add_u32 v18, s6, 3, v17 // (op 3768) 


v_and_b32 v17, 1, v15 // (op 3768) 

v_and_b32 v15, 15, v1 // (op 3768) 
v_lshl_add_u32 v19, v17, 4, v15 // (op 3768) 


v_lshl_add_u32 v15, s17, 5, v19 // (op 3768) 

v_mul_lo_u32 v17, v15, s32 // (op 3768) 

v_add_u32 v15, v18, v17 // (op 3768) 


s_mul_i32 s1, s32, s9 // (op 3770) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3770) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3770) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3770) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3770) most significant: sum
s_add_u32 s1, s1, s7 // (op 3770) most significant: sum


s_mov_b32 s60, s34 // (op 3770) reg expression
s_mov_b32 s61, s35 // (op 3770) reg expression
s_mov_b32 s62, s0 // (op 3770) reg expression
s_mov_b32 s63, 131072 // (op 3770) call()

v_and_b32 v17, -4, v1 // (op 3861) 
v_sub_u32 v18, v1, v17 // (op 3861) 

v_lshlrev_b32 v17, 5, v18 // (op 3861) 

v_lshrrev_b32 v18, 2, v1 // (op 3861) 
v_lshl_add_u32 v19, v18, 9, v17 // (op 3861) 


v_lshlrev_b32 v17, 2, v19 // (op 3861) 

v_lshrrev_b32 v18, 3, v17 // (op 3861) 

v_and_b32 v17, -8, v1 // (op 3919) 
v_sub_u32 v19, v1, v17 // (op 3919) 

v_lshlrev_b32 v17, 4, v19 // (op 3919) 

v_lshrrev_b32 v19, 3, v1 // (op 3919) 
v_lshl_add_u32 v20, v19, 7, v17 // (op 3919) 


v_lshlrev_b32 v17, 2, v20 // (op 3919) 

v_lshrrev_b32 v19, 3, v17 // (op 3919) 

v_lshlrev_b32 v17, 2, v1 // (op 3972) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1292) 
s_and_b32 s6, s0, 4294967294 // (op 1292) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 1291) 
s_mov_b32 s7, s0 // (op 1291) convert

s_cmp_lt_u32 s6, s7 // (op 1291) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1291 // (op 1291) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1291 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1291:

s_add_u32 s6, s6, 1 // (op 1293) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 170) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 170) 


v_mov_b32 v12, v13 // (op 170) 
v_mov_b32 v22, v12 // (op 170) 
buffer_load_dwordx4 v[24:27], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[28:31], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[32:35], v22, s[48:51], 0 offen // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[36:39], v22, s[48:51], 0 offen // (op 170) Load value



v_add_u32 v13, 64, v13 // (op 3556) 


s_barrier  // (op 4170) 
v_mov_b32 v12, v14 // (op 183) 




buffer_load_dwordx2 v[20:21], v12, s[52:55], 0 offen // (op 183) Load value

v_add_u32 v14, 64, v14 // (op 3632) 
s_waitcnt vmcnt(4)
ds_write_b128 v18, v[24:27] offset:2048 // (op 171) Store local data 


v_mov_b32 v12, v16 // (op 184) 

buffer_load_dword v22, v12, s[56:59], 0 offen // (op 184) Load value

v_add_u32 v16, 4, v16 // (op 3696) 
s_waitcnt vmcnt(4)
ds_write_b128 v18, v[28:31] offset:2112 // (op 171) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v18, v[32:35] offset:2176 // (op 171) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v18, v[36:39] offset:2240 // (op 171) Store local data 

s_barrier  // (op 4183) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4173) 






ds_read_b128 v[24:27], v5 offset:2048 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b64 v19, v[20:21] // (op 181) Store local data 

s_barrier  // (op 4197) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4187) 






ds_read_b128 v[28:31], v3 // (op 175) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v17, v22 offset:18432 // (op 182) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4201) 



ds_read_u8 v12, v0 offset:18432 // (op 177) Load local data 


v_mov_b32 v20, v15 // (op 179) 

buffer_load_ubyte v21, v20, s[60:63], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v15, 4, v15 // (op 3760) 



ds_read_b128 v[32:35], v5 offset:4096 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[28:31], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 180) 





ds_read_u8 v12, v0 offset:18560 // (op 329) Load local data 



ds_read_b128 v[22:25], v5 offset:6144 // (op 487) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:35], v[28:31], a[4:7], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 331) 





ds_read_u8 v12, v0 offset:18688 // (op 491) Load local data 



ds_read_b128 v[32:35], v5 offset:8192 // (op 649) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[28:31], a[8:11], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 493) 





ds_read_u8 v12, v0 offset:18816 // (op 653) Load local data 



ds_read_b128 v[22:25], v5 offset:10240 // (op 811) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:35], v[28:31], a[12:15], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 655) 





ds_read_u8 v12, v0 offset:18944 // (op 815) Load local data 



ds_read_b128 v[32:35], v5 offset:12288 // (op 973) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[28:31], a[16:19], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 817) 





ds_read_u8 v12, v0 offset:19072 // (op 977) Load local data 



ds_read_b128 v[22:25], v5 offset:14336 // (op 1135) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:35], v[28:31], a[20:23], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 979) 





ds_read_u8 v12, v0 offset:19200 // (op 1139) Load local data 



ds_read_b128 v[32:35], v5 offset:16384 // (op 1297) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[28:31], a[24:27], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1141) 





ds_read_u8 v12, v0 offset:19328 // (op 1301) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:35], v[28:31], a[28:31], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1303) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 1291) 
s_mov_b32 s7, s0 // (op 1291) convert

s_cmp_lt_u32 s6, s7 // (op 1291) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1291 // (op 1291) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1291 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1291:






















s_barrier  // (op 4319) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_125
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


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 54) Store value


v_accvgpr_read v12, a4 // (op 1621) 
v_accvgpr_read v13, a5 // (op 1621) 
v_accvgpr_read v14, a6 // (op 1621) 
v_accvgpr_read v15, a7 // (op 1621) 
v_mul_f32 v16, v11, v12 // (op 1621) 
v_mul_f32 v17, v11, v13 // (op 1621) 
v_mul_f32 v18, v11, v14 // (op 1621) 
v_mul_f32 v19, v11, v15 // (op 1621) 

v_lshrrev_b32 v12, 16, v16 // (op 1623) 
v_lshrrev_b32 v13, 16, v17 // (op 1623) 
v_lshrrev_b32 v14, 16, v18 // (op 1623) 
v_lshrrev_b32 v15, 16, v19 // (op 1623) 


v_pack_b32_f16 v4, v12, v13 // (op 1624) 
v_pack_b32_f16 v5, v14, v15 // (op 1624) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:64 // (op 1624) Store value


v_accvgpr_read v12, a8 // (op 1641) 
v_accvgpr_read v13, a9 // (op 1641) 
v_accvgpr_read v14, a10 // (op 1641) 
v_accvgpr_read v15, a11 // (op 1641) 
v_mul_f32 v16, v11, v12 // (op 1641) 
v_mul_f32 v17, v11, v13 // (op 1641) 
v_mul_f32 v18, v11, v14 // (op 1641) 
v_mul_f32 v19, v11, v15 // (op 1641) 

v_lshrrev_b32 v12, 16, v16 // (op 1643) 
v_lshrrev_b32 v13, 16, v17 // (op 1643) 
v_lshrrev_b32 v14, 16, v18 // (op 1643) 
v_lshrrev_b32 v15, 16, v19 // (op 1643) 


v_pack_b32_f16 v4, v12, v13 // (op 1644) 
v_pack_b32_f16 v5, v14, v15 // (op 1644) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:128 // (op 1644) Store value


v_accvgpr_read v12, a12 // (op 1661) 
v_accvgpr_read v13, a13 // (op 1661) 
v_accvgpr_read v14, a14 // (op 1661) 
v_accvgpr_read v15, a15 // (op 1661) 
v_mul_f32 v16, v11, v12 // (op 1661) 
v_mul_f32 v17, v11, v13 // (op 1661) 
v_mul_f32 v18, v11, v14 // (op 1661) 
v_mul_f32 v19, v11, v15 // (op 1661) 

v_lshrrev_b32 v12, 16, v16 // (op 1663) 
v_lshrrev_b32 v13, 16, v17 // (op 1663) 
v_lshrrev_b32 v14, 16, v18 // (op 1663) 
v_lshrrev_b32 v15, 16, v19 // (op 1663) 


v_pack_b32_f16 v4, v12, v13 // (op 1664) 
v_pack_b32_f16 v5, v14, v15 // (op 1664) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:192 // (op 1664) Store value


v_accvgpr_read v12, a16 // (op 1681) 
v_accvgpr_read v13, a17 // (op 1681) 
v_accvgpr_read v14, a18 // (op 1681) 
v_accvgpr_read v15, a19 // (op 1681) 
v_mul_f32 v16, v11, v12 // (op 1681) 
v_mul_f32 v17, v11, v13 // (op 1681) 
v_mul_f32 v18, v11, v14 // (op 1681) 
v_mul_f32 v19, v11, v15 // (op 1681) 

v_lshrrev_b32 v12, 16, v16 // (op 1683) 
v_lshrrev_b32 v13, 16, v17 // (op 1683) 
v_lshrrev_b32 v14, 16, v18 // (op 1683) 
v_lshrrev_b32 v15, 16, v19 // (op 1683) 


v_pack_b32_f16 v4, v12, v13 // (op 1684) 
v_pack_b32_f16 v5, v14, v15 // (op 1684) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:256 // (op 1684) Store value


v_accvgpr_read v12, a20 // (op 1701) 
v_accvgpr_read v13, a21 // (op 1701) 
v_accvgpr_read v14, a22 // (op 1701) 
v_accvgpr_read v15, a23 // (op 1701) 
v_mul_f32 v16, v11, v12 // (op 1701) 
v_mul_f32 v17, v11, v13 // (op 1701) 
v_mul_f32 v18, v11, v14 // (op 1701) 
v_mul_f32 v19, v11, v15 // (op 1701) 

v_lshrrev_b32 v12, 16, v16 // (op 1703) 
v_lshrrev_b32 v13, 16, v17 // (op 1703) 
v_lshrrev_b32 v14, 16, v18 // (op 1703) 
v_lshrrev_b32 v15, 16, v19 // (op 1703) 


v_pack_b32_f16 v4, v12, v13 // (op 1704) 
v_pack_b32_f16 v5, v14, v15 // (op 1704) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:320 // (op 1704) Store value


v_accvgpr_read v12, a24 // (op 1721) 
v_accvgpr_read v13, a25 // (op 1721) 
v_accvgpr_read v14, a26 // (op 1721) 
v_accvgpr_read v15, a27 // (op 1721) 
v_mul_f32 v16, v11, v12 // (op 1721) 
v_mul_f32 v17, v11, v13 // (op 1721) 
v_mul_f32 v18, v11, v14 // (op 1721) 
v_mul_f32 v19, v11, v15 // (op 1721) 

v_lshrrev_b32 v12, 16, v16 // (op 1723) 
v_lshrrev_b32 v13, 16, v17 // (op 1723) 
v_lshrrev_b32 v14, 16, v18 // (op 1723) 
v_lshrrev_b32 v15, 16, v19 // (op 1723) 


v_pack_b32_f16 v4, v12, v13 // (op 1724) 
v_pack_b32_f16 v5, v14, v15 // (op 1724) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:384 // (op 1724) Store value


v_accvgpr_read v12, a28 // (op 1741) 
v_accvgpr_read v13, a29 // (op 1741) 
v_accvgpr_read v14, a30 // (op 1741) 
v_accvgpr_read v15, a31 // (op 1741) 
v_mul_f32 v16, v11, v12 // (op 1741) 
v_mul_f32 v17, v11, v13 // (op 1741) 
v_mul_f32 v18, v11, v14 // (op 1741) 
v_mul_f32 v19, v11, v15 // (op 1741) 

v_lshrrev_b32 v12, 16, v16 // (op 1743) 
v_lshrrev_b32 v13, 16, v17 // (op 1743) 
v_lshrrev_b32 v14, 16, v18 // (op 1743) 
v_lshrrev_b32 v15, 16, v19 // (op 1743) 


v_pack_b32_f16 v4, v12, v13 // (op 1744) 
v_pack_b32_f16 v5, v14, v15 // (op 1744) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:448 // (op 1744) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:448 // (op 1924) Load value
v_accvgpr_read v12, a28 // (op 1920) 
v_accvgpr_read v13, a29 // (op 1920) 
v_accvgpr_read v14, a30 // (op 1920) 
v_accvgpr_read v15, a31 // (op 1920) 
v_mul_f32 v16, v11, v12 // (op 1920) 
v_mul_f32 v17, v11, v13 // (op 1920) 
v_mul_f32 v18, v11, v14 // (op 1920) 
v_mul_f32 v19, v11, v15 // (op 1920) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1925) 
v_lshrrev_b32 v13, 16, v4 // (op 1925) 
v_lshlrev_b32 v12, 16, v12 // (op 1925) 
v_lshlrev_b32 v13, 16, v13 // (op 1925) 
v_mul_f32 v20, v10, v12 // (op 1925) 
v_mul_f32 v21, v10, v13 // (op 1925) 
v_and_b32 v12, 65535, v5 // (op 1925) 
v_lshrrev_b32 v13, 16, v5 // (op 1925) 
v_lshlrev_b32 v12, 16, v12 // (op 1925) 
v_lshlrev_b32 v13, 16, v13 // (op 1925) 
v_mul_f32 v22, v10, v12 // (op 1925) 
v_mul_f32 v23, v10, v13 // (op 1925) 


v_add_f32 v12, v20, v16 // (op 1921) 
v_add_f32 v13, v21, v17 // (op 1921) 
v_add_f32 v14, v22, v18 // (op 1921) 
v_add_f32 v15, v23, v19 // (op 1921) 


v_lshrrev_b32 v16, 16, v12 // (op 1922) 
v_lshrrev_b32 v17, 16, v13 // (op 1922) 
v_lshrrev_b32 v18, 16, v14 // (op 1922) 
v_lshrrev_b32 v19, 16, v15 // (op 1922) 


v_pack_b32_f16 v4, v16, v17 // (op 1923) 
v_pack_b32_f16 v5, v18, v19 // (op 1923) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:448 // (op 1923) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:384 // (op 1904) Load value
v_accvgpr_read v12, a24 // (op 1900) 
v_accvgpr_read v13, a25 // (op 1900) 
v_accvgpr_read v14, a26 // (op 1900) 
v_accvgpr_read v15, a27 // (op 1900) 
v_mul_f32 v16, v11, v12 // (op 1900) 
v_mul_f32 v17, v11, v13 // (op 1900) 
v_mul_f32 v18, v11, v14 // (op 1900) 
v_mul_f32 v19, v11, v15 // (op 1900) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1905) 
v_lshrrev_b32 v13, 16, v4 // (op 1905) 
v_lshlrev_b32 v12, 16, v12 // (op 1905) 
v_lshlrev_b32 v13, 16, v13 // (op 1905) 
v_mul_f32 v20, v10, v12 // (op 1905) 
v_mul_f32 v21, v10, v13 // (op 1905) 
v_and_b32 v12, 65535, v5 // (op 1905) 
v_lshrrev_b32 v13, 16, v5 // (op 1905) 
v_lshlrev_b32 v12, 16, v12 // (op 1905) 
v_lshlrev_b32 v13, 16, v13 // (op 1905) 
v_mul_f32 v22, v10, v12 // (op 1905) 
v_mul_f32 v23, v10, v13 // (op 1905) 


v_add_f32 v12, v20, v16 // (op 1901) 
v_add_f32 v13, v21, v17 // (op 1901) 
v_add_f32 v14, v22, v18 // (op 1901) 
v_add_f32 v15, v23, v19 // (op 1901) 


v_lshrrev_b32 v16, 16, v12 // (op 1902) 
v_lshrrev_b32 v17, 16, v13 // (op 1902) 
v_lshrrev_b32 v18, 16, v14 // (op 1902) 
v_lshrrev_b32 v19, 16, v15 // (op 1902) 


v_pack_b32_f16 v4, v16, v17 // (op 1903) 
v_pack_b32_f16 v5, v18, v19 // (op 1903) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:384 // (op 1903) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:320 // (op 1884) Load value
v_accvgpr_read v12, a20 // (op 1880) 
v_accvgpr_read v13, a21 // (op 1880) 
v_accvgpr_read v14, a22 // (op 1880) 
v_accvgpr_read v15, a23 // (op 1880) 
v_mul_f32 v16, v11, v12 // (op 1880) 
v_mul_f32 v17, v11, v13 // (op 1880) 
v_mul_f32 v18, v11, v14 // (op 1880) 
v_mul_f32 v19, v11, v15 // (op 1880) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1885) 
v_lshrrev_b32 v13, 16, v4 // (op 1885) 
v_lshlrev_b32 v12, 16, v12 // (op 1885) 
v_lshlrev_b32 v13, 16, v13 // (op 1885) 
v_mul_f32 v20, v10, v12 // (op 1885) 
v_mul_f32 v21, v10, v13 // (op 1885) 
v_and_b32 v12, 65535, v5 // (op 1885) 
v_lshrrev_b32 v13, 16, v5 // (op 1885) 
v_lshlrev_b32 v12, 16, v12 // (op 1885) 
v_lshlrev_b32 v13, 16, v13 // (op 1885) 
v_mul_f32 v22, v10, v12 // (op 1885) 
v_mul_f32 v23, v10, v13 // (op 1885) 


v_add_f32 v12, v20, v16 // (op 1881) 
v_add_f32 v13, v21, v17 // (op 1881) 
v_add_f32 v14, v22, v18 // (op 1881) 
v_add_f32 v15, v23, v19 // (op 1881) 


v_lshrrev_b32 v16, 16, v12 // (op 1882) 
v_lshrrev_b32 v17, 16, v13 // (op 1882) 
v_lshrrev_b32 v18, 16, v14 // (op 1882) 
v_lshrrev_b32 v19, 16, v15 // (op 1882) 


v_pack_b32_f16 v4, v16, v17 // (op 1883) 
v_pack_b32_f16 v5, v18, v19 // (op 1883) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:320 // (op 1883) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:256 // (op 1864) Load value
v_accvgpr_read v12, a16 // (op 1860) 
v_accvgpr_read v13, a17 // (op 1860) 
v_accvgpr_read v14, a18 // (op 1860) 
v_accvgpr_read v15, a19 // (op 1860) 
v_mul_f32 v16, v11, v12 // (op 1860) 
v_mul_f32 v17, v11, v13 // (op 1860) 
v_mul_f32 v18, v11, v14 // (op 1860) 
v_mul_f32 v19, v11, v15 // (op 1860) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1865) 
v_lshrrev_b32 v13, 16, v4 // (op 1865) 
v_lshlrev_b32 v12, 16, v12 // (op 1865) 
v_lshlrev_b32 v13, 16, v13 // (op 1865) 
v_mul_f32 v20, v10, v12 // (op 1865) 
v_mul_f32 v21, v10, v13 // (op 1865) 
v_and_b32 v12, 65535, v5 // (op 1865) 
v_lshrrev_b32 v13, 16, v5 // (op 1865) 
v_lshlrev_b32 v12, 16, v12 // (op 1865) 
v_lshlrev_b32 v13, 16, v13 // (op 1865) 
v_mul_f32 v22, v10, v12 // (op 1865) 
v_mul_f32 v23, v10, v13 // (op 1865) 


v_add_f32 v12, v20, v16 // (op 1861) 
v_add_f32 v13, v21, v17 // (op 1861) 
v_add_f32 v14, v22, v18 // (op 1861) 
v_add_f32 v15, v23, v19 // (op 1861) 


v_lshrrev_b32 v16, 16, v12 // (op 1862) 
v_lshrrev_b32 v17, 16, v13 // (op 1862) 
v_lshrrev_b32 v18, 16, v14 // (op 1862) 
v_lshrrev_b32 v19, 16, v15 // (op 1862) 


v_pack_b32_f16 v4, v16, v17 // (op 1863) 
v_pack_b32_f16 v5, v18, v19 // (op 1863) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:256 // (op 1863) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:192 // (op 1844) Load value
v_accvgpr_read v12, a12 // (op 1840) 
v_accvgpr_read v13, a13 // (op 1840) 
v_accvgpr_read v14, a14 // (op 1840) 
v_accvgpr_read v15, a15 // (op 1840) 
v_mul_f32 v16, v11, v12 // (op 1840) 
v_mul_f32 v17, v11, v13 // (op 1840) 
v_mul_f32 v18, v11, v14 // (op 1840) 
v_mul_f32 v19, v11, v15 // (op 1840) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1845) 
v_lshrrev_b32 v13, 16, v4 // (op 1845) 
v_lshlrev_b32 v12, 16, v12 // (op 1845) 
v_lshlrev_b32 v13, 16, v13 // (op 1845) 
v_mul_f32 v20, v10, v12 // (op 1845) 
v_mul_f32 v21, v10, v13 // (op 1845) 
v_and_b32 v12, 65535, v5 // (op 1845) 
v_lshrrev_b32 v13, 16, v5 // (op 1845) 
v_lshlrev_b32 v12, 16, v12 // (op 1845) 
v_lshlrev_b32 v13, 16, v13 // (op 1845) 
v_mul_f32 v22, v10, v12 // (op 1845) 
v_mul_f32 v23, v10, v13 // (op 1845) 


v_add_f32 v12, v20, v16 // (op 1841) 
v_add_f32 v13, v21, v17 // (op 1841) 
v_add_f32 v14, v22, v18 // (op 1841) 
v_add_f32 v15, v23, v19 // (op 1841) 


v_lshrrev_b32 v16, 16, v12 // (op 1842) 
v_lshrrev_b32 v17, 16, v13 // (op 1842) 
v_lshrrev_b32 v18, 16, v14 // (op 1842) 
v_lshrrev_b32 v19, 16, v15 // (op 1842) 


v_pack_b32_f16 v4, v16, v17 // (op 1843) 
v_pack_b32_f16 v5, v18, v19 // (op 1843) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:192 // (op 1843) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:128 // (op 1824) Load value
v_accvgpr_read v12, a8 // (op 1820) 
v_accvgpr_read v13, a9 // (op 1820) 
v_accvgpr_read v14, a10 // (op 1820) 
v_accvgpr_read v15, a11 // (op 1820) 
v_mul_f32 v16, v11, v12 // (op 1820) 
v_mul_f32 v17, v11, v13 // (op 1820) 
v_mul_f32 v18, v11, v14 // (op 1820) 
v_mul_f32 v19, v11, v15 // (op 1820) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1825) 
v_lshrrev_b32 v13, 16, v4 // (op 1825) 
v_lshlrev_b32 v12, 16, v12 // (op 1825) 
v_lshlrev_b32 v13, 16, v13 // (op 1825) 
v_mul_f32 v20, v10, v12 // (op 1825) 
v_mul_f32 v21, v10, v13 // (op 1825) 
v_and_b32 v12, 65535, v5 // (op 1825) 
v_lshrrev_b32 v13, 16, v5 // (op 1825) 
v_lshlrev_b32 v12, 16, v12 // (op 1825) 
v_lshlrev_b32 v13, 16, v13 // (op 1825) 
v_mul_f32 v22, v10, v12 // (op 1825) 
v_mul_f32 v23, v10, v13 // (op 1825) 


v_add_f32 v12, v20, v16 // (op 1821) 
v_add_f32 v13, v21, v17 // (op 1821) 
v_add_f32 v14, v22, v18 // (op 1821) 
v_add_f32 v15, v23, v19 // (op 1821) 


v_lshrrev_b32 v16, 16, v12 // (op 1822) 
v_lshrrev_b32 v17, 16, v13 // (op 1822) 
v_lshrrev_b32 v18, 16, v14 // (op 1822) 
v_lshrrev_b32 v19, 16, v15 // (op 1822) 


v_pack_b32_f16 v4, v16, v17 // (op 1823) 
v_pack_b32_f16 v5, v18, v19 // (op 1823) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:128 // (op 1823) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:64 // (op 1804) Load value
v_accvgpr_read v12, a4 // (op 1800) 
v_accvgpr_read v13, a5 // (op 1800) 
v_accvgpr_read v14, a6 // (op 1800) 
v_accvgpr_read v15, a7 // (op 1800) 
v_mul_f32 v16, v11, v12 // (op 1800) 
v_mul_f32 v17, v11, v13 // (op 1800) 
v_mul_f32 v18, v11, v14 // (op 1800) 
v_mul_f32 v19, v11, v15 // (op 1800) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1805) 
v_lshrrev_b32 v13, 16, v4 // (op 1805) 
v_lshlrev_b32 v12, 16, v12 // (op 1805) 
v_lshlrev_b32 v13, 16, v13 // (op 1805) 
v_mul_f32 v20, v10, v12 // (op 1805) 
v_mul_f32 v21, v10, v13 // (op 1805) 
v_and_b32 v12, 65535, v5 // (op 1805) 
v_lshrrev_b32 v13, 16, v5 // (op 1805) 
v_lshlrev_b32 v12, 16, v12 // (op 1805) 
v_lshlrev_b32 v13, 16, v13 // (op 1805) 
v_mul_f32 v22, v10, v12 // (op 1805) 
v_mul_f32 v23, v10, v13 // (op 1805) 


v_add_f32 v12, v20, v16 // (op 1801) 
v_add_f32 v13, v21, v17 // (op 1801) 
v_add_f32 v14, v22, v18 // (op 1801) 
v_add_f32 v15, v23, v19 // (op 1801) 


v_lshrrev_b32 v16, 16, v12 // (op 1802) 
v_lshrrev_b32 v17, 16, v13 // (op 1802) 
v_lshrrev_b32 v18, 16, v14 // (op 1802) 
v_lshrrev_b32 v19, 16, v15 // (op 1802) 


v_pack_b32_f16 v4, v16, v17 // (op 1803) 
v_pack_b32_f16 v5, v18, v19 // (op 1803) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:64 // (op 1803) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen // (op 138) Load value
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


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2, .LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_end-RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
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
    macrotile: [256, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2.kd
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
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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

