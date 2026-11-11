// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
.p2align 8
.type RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC,@function
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 6986) 
v_and_b32 v3, 15, v1 // (op 6986) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 6986) 


v_and_b32 v0, 63, v1 // (op 6986) 
v_lshrrev_b32 v3, 4, v0 // (op 6986) 

v_lshlrev_b32 v0, 5, v3 // (op 6986) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 6986) 


v_lshlrev_b32 v0, 2, v3 // (op 6986) 

v_lshrrev_b32 v3, 3, v0 // (op 6986) 

v_lshrrev_b32 v0, 6, v1 // (op 7044) 
v_and_b32 v4, 1, v0 // (op 7044) 

v_and_b32 v0, 15, v1 // (op 7044) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 7044) 


v_and_b32 v0, 63, v1 // (op 7044) 
v_lshrrev_b32 v4, 4, v0 // (op 7044) 

v_lshlrev_b32 v0, 5, v4 // (op 7044) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 7044) 


v_lshlrev_b32 v0, 2, v4 // (op 7044) 

v_lshrrev_b32 v4, 3, v0 // (op 7044) 

v_lshrrev_b32 v0, 7, v1 // (op 7102) 
v_and_b32 v5, 15, v1 // (op 7102) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 7102) 


v_and_b32 v0, 63, v1 // (op 7102) 
v_lshrrev_b32 v5, 4, v0 // (op 7102) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 7102) 


v_lshrrev_b32 v5, 6, v1 // (op 7160) 
v_and_b32 v6, 1, v5 // (op 7160) 

v_and_b32 v5, 15, v1 // (op 7160) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 7160) 


v_and_b32 v5, 63, v1 // (op 7160) 
v_lshrrev_b32 v6, 4, v5 // (op 7160) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 7160) 


v_lshrrev_b32 v6, 7, v1 // (op 7212) 
v_and_b32 v7, 15, v1 // (op 7212) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 7212) 


v_and_b32 v6, 63, v1 // (op 7212) 
v_lshrrev_b32 v7, 4, v6 // (op 7212) 

v_lshlrev_b32 v6, 5, v7 // (op 7212) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 7212) 


v_lshlrev_b32 v6, 2, v7 // (op 7212) 

v_lshrrev_b32 v7, 3, v6 // (op 7212) 

v_lshrrev_b32 v6, 6, v1 // (op 7241) 
v_and_b32 v8, 1, v6 // (op 7241) 

v_and_b32 v6, 15, v1 // (op 7241) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 7241) 


v_and_b32 v6, 63, v1 // (op 7241) 
v_lshrrev_b32 v8, 4, v6 // (op 7241) 

v_lshlrev_b32 v6, 5, v8 // (op 7241) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 7241) 


v_lshlrev_b32 v6, 2, v8 // (op 7241) 

v_lshrrev_b32 v8, 3, v6 // (op 7241) 

v_lshrrev_b32 v6, 7, v1 // (op 7270) 
v_and_b32 v9, 15, v1 // (op 7270) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 7270) 


v_and_b32 v6, 63, v1 // (op 7270) 
v_lshrrev_b32 v9, 4, v6 // (op 7270) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 7270) 


v_lshrrev_b32 v9, 6, v1 // (op 7299) 
v_and_b32 v10, 1, v9 // (op 7299) 

v_and_b32 v9, 15, v1 // (op 7299) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 7299) 


v_and_b32 v9, 63, v1 // (op 7299) 
v_lshrrev_b32 v10, 4, v9 // (op 7299) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 7299) 


v_lshrrev_b32 v10, 7, v1 // (op 7328) 
v_and_b32 v11, 15, v1 // (op 7328) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 7328) 


v_and_b32 v10, 63, v1 // (op 7328) 
v_lshrrev_b32 v11, 4, v10 // (op 7328) 

v_lshlrev_b32 v10, 5, v11 // (op 7328) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 7328) 


v_lshlrev_b32 v10, 2, v11 // (op 7328) 

v_lshrrev_b32 v11, 3, v10 // (op 7328) 

v_lshrrev_b32 v10, 6, v1 // (op 7357) 
v_and_b32 v12, 1, v10 // (op 7357) 

v_and_b32 v10, 15, v1 // (op 7357) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 7357) 


v_and_b32 v10, 63, v1 // (op 7357) 
v_lshrrev_b32 v12, 4, v10 // (op 7357) 

v_lshlrev_b32 v10, 5, v12 // (op 7357) 

v_lshl_add_u32 v12, v13, 7, v10 // (op 7357) 


v_lshlrev_b32 v10, 2, v12 // (op 7357) 

v_lshrrev_b32 v12, 3, v10 // (op 7357) 

v_lshrrev_b32 v10, 7, v1 // (op 7386) 
v_and_b32 v13, 15, v1 // (op 7386) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 7386) 


v_and_b32 v10, 63, v1 // (op 7386) 
v_lshrrev_b32 v13, 4, v10 // (op 7386) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 7386) 


v_lshrrev_b32 v13, 6, v1 // (op 7415) 
v_and_b32 v14, 1, v13 // (op 7415) 

v_and_b32 v13, 15, v1 // (op 7415) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 7415) 


v_and_b32 v13, 63, v1 // (op 7415) 
v_lshrrev_b32 v14, 4, v13 // (op 7415) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 7415) 


v_lshrrev_b32 v14, 7, v1 // (op 7444) 
v_and_b32 v15, 15, v1 // (op 7444) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 7444) 


v_and_b32 v14, 63, v1 // (op 7444) 
v_lshrrev_b32 v15, 4, v14 // (op 7444) 

v_lshlrev_b32 v14, 5, v15 // (op 7444) 

v_lshl_add_u32 v15, v16, 7, v14 // (op 7444) 


v_lshlrev_b32 v14, 2, v15 // (op 7444) 

v_lshrrev_b32 v15, 3, v14 // (op 7444) 

v_lshrrev_b32 v14, 6, v1 // (op 7473) 
v_and_b32 v16, 1, v14 // (op 7473) 

v_and_b32 v14, 15, v1 // (op 7473) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 7473) 


v_and_b32 v14, 63, v1 // (op 7473) 
v_lshrrev_b32 v16, 4, v14 // (op 7473) 

v_lshlrev_b32 v14, 5, v16 // (op 7473) 

v_lshl_add_u32 v16, v17, 7, v14 // (op 7473) 


v_lshlrev_b32 v14, 2, v16 // (op 7473) 

v_lshrrev_b32 v16, 3, v14 // (op 7473) 

v_lshrrev_b32 v14, 7, v1 // (op 7502) 
v_and_b32 v17, 15, v1 // (op 7502) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 7502) 


v_and_b32 v14, 63, v1 // (op 7502) 
v_lshrrev_b32 v17, 4, v14 // (op 7502) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 7502) 


v_lshrrev_b32 v17, 6, v1 // (op 7531) 
v_and_b32 v18, 1, v17 // (op 7531) 

v_and_b32 v17, 15, v1 // (op 7531) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 7531) 


v_and_b32 v17, 63, v1 // (op 7531) 
v_lshrrev_b32 v18, 4, v17 // (op 7531) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 7531) 


v_lshrrev_b32 v18, 7, v1 // (op 6924) 
v_and_b32 v19, 63, v1 // (op 6924) 
v_lshrrev_b32 v20, 4, v19 // (op 6924) 
v_lshlrev_b32 v21, 2, v20 // (op 6924) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 6924) 


v_lshl_add_u32 v18, s16, 6, v20 // (op 6924) 

v_lshrrev_b32 v20, 2, v19 // (op 6924) 

v_and_b32 v19, 3, v20 // (op 6924) 

v_and_b32 v20, 3, v1 // (op 6924) 
v_lshl_add_u32 v21, v19, 2, v20 // (op 6924) 


v_lshrrev_b32 v19, 6, v1 // (op 6924) 
v_and_b32 v20, 1, v19 // (op 6924) 

v_lshl_add_u32 v19, v20, 4, v21 // (op 6924) 


v_lshl_add_u32 v20, s17, 7, v19 // (op 6924) 

v_mul_lo_u32 v19, v20, s2 // (op 6924) 

v_add_lshl_u32 v20, v18, v19, 1 // (op 6924) 


v_lshrrev_b32 v18, 7, v1 // (op 7181) 
v_and_b32 v19, 63, v1 // (op 7181) 
v_lshrrev_b32 v21, 4, v19 // (op 7181) 
s_mul_i32 s1, s2, s9 // (op 6926) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 6926) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 6926) least significant
s_mul_hi_u32 s19, s2, s8 // (op 6926) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6926) most significant: sum
s_add_u32 s1, s1, s19 // (op 6926) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 6926) 

v_lshlrev_b32 v22, 2, v21 // (op 7181) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 7181) 


v_lshl_add_u32 v18, s16, 6, v21 // (op 7181) 

v_lshrrev_b32 v21, 2, v19 // (op 7181) 

v_and_b32 v19, 3, v21 // (op 7181) 

v_and_b32 v21, 3, v1 // (op 7181) 
v_lshl_add_u32 v22, v19, 2, v21 // (op 7181) 


v_lshrrev_b32 v19, 6, v1 // (op 7181) 
v_and_b32 v21, 1, v19 // (op 7181) 

v_lshl_add_u32 v19, v21, 4, v22 // (op 7181) 


v_lshl_add_u32 v21, s17, 7, v19 // (op 7181) 

v_mul_lo_u32 v19, v21, s4 // (op 7181) 

v_add_lshl_u32 v21, v18, v19, 1 // (op 7181) 


s_mov_b32 s40, s6 // (op 6926) reg expression
s_mov_b32 s41, s7 // (op 6926) reg expression
s_mov_b32 s42, s18 // (op 6926) reg expression
s_mov_b32 s43, 131072 // (op 6926) call()

s_mul_i32 s1, s4, s9 // (op 7183) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 7183) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 7183) least significant
s_mul_hi_u32 s7, s4, s8 // (op 7183) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7183) most significant: sum
s_add_u32 s1, s1, s7 // (op 7183) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 7183) 

s_mov_b32 s44, s10 // (op 7183) reg expression
s_mov_b32 s45, s11 // (op 7183) reg expression
s_mov_b32 s46, s6 // (op 7183) reg expression
s_mov_b32 s47, 131072 // (op 7183) call()


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
v_accvgpr_write a8, 0.00000 // (op 784) call()
v_accvgpr_write a9, 0.00000 // (op 784) call()
v_accvgpr_write a10, 0.00000 // (op 784) call()
v_accvgpr_write a11, 0.00000 // (op 784) call()
v_accvgpr_write a12, 0.00000 // (op 1082) call()
v_accvgpr_write a13, 0.00000 // (op 1082) call()
v_accvgpr_write a14, 0.00000 // (op 1082) call()
v_accvgpr_write a15, 0.00000 // (op 1082) call()
v_accvgpr_write a16, 0.00000 // (op 1516) call()
v_accvgpr_write a17, 0.00000 // (op 1516) call()
v_accvgpr_write a18, 0.00000 // (op 1516) call()
v_accvgpr_write a19, 0.00000 // (op 1516) call()
v_accvgpr_write a20, 0.00000 // (op 2009) call()
v_accvgpr_write a21, 0.00000 // (op 2009) call()
v_accvgpr_write a22, 0.00000 // (op 2009) call()
v_accvgpr_write a23, 0.00000 // (op 2009) call()
v_accvgpr_write a24, 0.00000 // (op 2010) call()
v_accvgpr_write a25, 0.00000 // (op 2010) call()
v_accvgpr_write a26, 0.00000 // (op 2010) call()
v_accvgpr_write a27, 0.00000 // (op 2010) call()
v_accvgpr_write a28, 0.00000 // (op 2011) call()
v_accvgpr_write a29, 0.00000 // (op 2011) call()
v_accvgpr_write a30, 0.00000 // (op 2011) call()
v_accvgpr_write a31, 0.00000 // (op 2011) call()
v_lshrrev_b32 v22, 2, v1 // (op 6686) 
v_lshl_add_u32 v23, s16, 6, v22 // (op 6686) 

v_and_b32 v22, -4, v1 // (op 6686) 
v_sub_u32 v24, v1, v22 // (op 6686) 

v_lshlrev_b32 v22, 5, v24 // (op 6686) 

v_mov_b32 v25, 0 // (op 6686) convert
v_mov_b32 v24, v22 // (op 6686) convert

v_mul_lo_u32 v27, v23, s13 // (op 6686) most significant: low of low * high
v_mul_hi_u32 v22, v23, s12 // (op 6686) most significant: high of low * low
v_mul_lo_u32 v26, v23, s12 // (op 6686) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 6686) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v24 // (op 6686) least significant half
v_addc_co_u32 v27, s[0:1], v27, v25, s[0:1] // (op 6686) most significant half



v_lshlrev_b64 v[22:23], 2, v[26:27] // (op 6686) 

v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 6686) 

v_mov_b32 v22, v24 // (op 6686) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 6688) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6688) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6688) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6688) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6688) most significant: sum
s_add_u32 s1, s1, s7 // (op 6688) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6688) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6688) 

s_mov_b32 s48, s20 // (op 6688) reg expression
s_mov_b32 s49, s21 // (op 6688) reg expression
s_mov_b32 s50, s0 // (op 6688) reg expression
s_mov_b32 s51, 131072 // (op 6688) call()

v_lshrrev_b32 v23, 2, v1 // (op 6705) 
v_lshl_add_u32 v24, s16, 6, v23 // (op 6705) 

v_and_b32 v23, -4, v1 // (op 6705) 
v_sub_u32 v25, v1, v23 // (op 6705) 

v_lshlrev_b32 v23, 5, v25 // (op 6705) 

v_mov_b32 v27, 0 // (op 6705) convert
v_mov_b32 v26, v23 // (op 6705) convert

v_mul_lo_u32 v29, v24, s13 // (op 6705) most significant: low of low * high
v_mul_hi_u32 v23, v24, s12 // (op 6705) most significant: high of low * low
v_mul_lo_u32 v28, v24, s12 // (op 6705) least significant: low of low * low
v_add_u32 v29, v29, v23 // (op 6705) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v26 // (op 6705) least significant half
v_addc_co_u32 v29, s[0:1], v29, v27, s[0:1] // (op 6705) most significant half



v_lshlrev_b64 v[26:27], 2, v[28:29] // (op 6705) 

v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 6705) 

v_mov_b32 v23, v24 // (op 6705) convert

s_mul_i32 s1, s12, s23 // (op 6707) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6707) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6707) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6707) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6707) most significant: sum
s_add_u32 s1, s1, s7 // (op 6707) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6707) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6707) 

s_mov_b32 s48, s20 // (op 6707) reg expression
s_mov_b32 s49, s21 // (op 6707) reg expression
s_mov_b32 s50, s0 // (op 6707) reg expression
s_mov_b32 s51, 131072 // (op 6707) call()

v_and_b32 v24, -4, v1 // (op 6750) 
v_sub_u32 v25, v1, v24 // (op 6750) 

v_lshlrev_b32 v24, 5, v25 // (op 6750) 

v_mov_b32 v27, 0 // (op 6750) convert
v_mov_b32 v26, v24 // (op 6750) convert

v_lshrrev_b32 v24, 2, v1 // (op 6750) 
v_lshlrev_b32 v25, 1, v24 // (op 6750) 

v_lshl_add_u32 v24, s17, 7, v25 // (op 6750) 

v_mul_lo_u32 v29, v24, s25 // (op 6750) most significant: low of low * high
v_mul_hi_u32 v25, v24, s24 // (op 6750) most significant: high of low * low
v_mul_lo_u32 v28, v24, s24 // (op 6750) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 6750) most significant: sum


v_add_co_u32 v24, s[0:1], v26, v28 // (op 6750) least significant half
v_addc_co_u32 v25, s[0:1], v27, v29, s[0:1] // (op 6750) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 6750) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 6750) 

v_mov_b32 v24, v26 // (op 6750) convert

s_mul_i32 s1, s24, s9 // (op 6752) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6752) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6752) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6752) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6752) most significant: sum
s_add_u32 s1, s1, s7 // (op 6752) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6752) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6752) 

s_mov_b32 s52, s26 // (op 6752) reg expression
s_mov_b32 s53, s27 // (op 6752) reg expression
s_mov_b32 s54, s0 // (op 6752) reg expression
s_mov_b32 s55, 131072 // (op 6752) call()

v_and_b32 v25, -4, v1 // (op 6769) 
v_sub_u32 v26, v1, v25 // (op 6769) 

v_lshlrev_b32 v25, 5, v26 // (op 6769) 

v_mov_b32 v27, 0 // (op 6769) convert
v_mov_b32 v26, v25 // (op 6769) convert

v_lshrrev_b32 v25, 2, v1 // (op 6769) 
v_lshlrev_b32 v28, 1, v25 // (op 6769) 

v_lshl_add_u32 v25, s17, 7, v28 // (op 6769) 

v_mul_lo_u32 v29, v25, s25 // (op 6769) most significant: low of low * high
v_mul_hi_u32 v30, v25, s24 // (op 6769) most significant: high of low * low
v_mul_lo_u32 v28, v25, s24 // (op 6769) least significant: low of low * low
v_add_u32 v29, v29, v30 // (op 6769) most significant: sum


v_add_co_u32 v30, s[0:1], v26, v28 // (op 6769) least significant half
v_addc_co_u32 v31, s[0:1], v27, v29, s[0:1] // (op 6769) most significant half

v_lshlrev_b64 v[30:31], 2, v[30:31] // (op 6769) 


v_lshrrev_b64 v[28:29], 3, v[30:31] // (op 6769) 

v_mov_b32 v25, v28 // (op 6769) convert

s_mul_i32 s1, s24, s9 // (op 6771) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6771) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6771) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6771) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6771) most significant: sum
s_add_u32 s1, s1, s7 // (op 6771) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6771) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6771) 

s_mov_b32 s52, s26 // (op 6771) reg expression
s_mov_b32 s53, s27 // (op 6771) reg expression
s_mov_b32 s54, s0 // (op 6771) reg expression
s_mov_b32 s55, 131072 // (op 6771) call()

v_lshrrev_b32 v26, 2, v1 // (op 6814) 
v_lshl_add_u32 v27, s16, 6, v26 // (op 6814) 

v_and_b32 v26, -4, v1 // (op 6814) 
v_sub_u32 v28, v1, v26 // (op 6814) 

v_mul_lo_u32 v26, v27, s28 // (op 6814) 
v_add_u32 v26, v26, v28 // (op 6814) 


s_mul_i32 s1, s28, s23 // (op 6816) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6816) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6816) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6816) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6816) most significant: sum
s_add_u32 s1, s1, s7 // (op 6816) most significant: sum


s_mov_b32 s56, s30 // (op 6816) reg expression
s_mov_b32 s57, s31 // (op 6816) reg expression
s_mov_b32 s58, s0 // (op 6816) reg expression
s_mov_b32 s59, 131072 // (op 6816) call()

v_lshrrev_b32 v27, 2, v1 // (op 6833) 
v_lshl_add_u32 v28, s16, 6, v27 // (op 6833) 

v_and_b32 v27, -4, v1 // (op 6833) 
v_sub_u32 v29, v1, v27 // (op 6833) 

v_mul_lo_u32 v27, v28, s28 // (op 6833) 
v_add_u32 v27, v27, v29 // (op 6833) 


s_mul_i32 s1, s28, s23 // (op 6835) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6835) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6835) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6835) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6835) most significant: sum
s_add_u32 s1, s1, s7 // (op 6835) most significant: sum


s_mov_b32 s56, s30 // (op 6835) reg expression
s_mov_b32 s57, s31 // (op 6835) reg expression
s_mov_b32 s58, s0 // (op 6835) reg expression
s_mov_b32 s59, 131072 // (op 6835) call()

v_and_b32 v28, -4, v1 // (op 6878) 
v_sub_u32 v29, v1, v28 // (op 6878) 

v_lshrrev_b32 v28, 2, v1 // (op 6878) 
v_lshlrev_b32 v30, 1, v28 // (op 6878) 

v_lshl_add_u32 v28, s17, 7, v30 // (op 6878) 

v_mul_lo_u32 v30, v28, s32 // (op 6878) 

v_add_u32 v28, v29, v30 // (op 6878) 


s_mul_i32 s1, s32, s9 // (op 6880) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6880) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6880) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6880) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6880) most significant: sum
s_add_u32 s1, s1, s7 // (op 6880) most significant: sum


s_mov_b32 s60, s34 // (op 6880) reg expression
s_mov_b32 s61, s35 // (op 6880) reg expression
s_mov_b32 s62, s0 // (op 6880) reg expression
s_mov_b32 s63, 131072 // (op 6880) call()

v_and_b32 v29, -4, v1 // (op 6897) 
v_sub_u32 v30, v1, v29 // (op 6897) 

v_lshrrev_b32 v29, 2, v1 // (op 6897) 
v_lshlrev_b32 v31, 1, v29 // (op 6897) 

v_lshl_add_u32 v29, s17, 7, v31 // (op 6897) 

v_mul_lo_u32 v31, v29, s32 // (op 6897) 

v_add_u32 v29, v30, v31 // (op 6897) 


s_mul_i32 s1, s32, s9 // (op 6899) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6899) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6899) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6899) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6899) most significant: sum
s_add_u32 s1, s1, s7 // (op 6899) most significant: sum


s_mov_b32 s60, s34 // (op 6899) reg expression
s_mov_b32 s61, s35 // (op 6899) reg expression
s_mov_b32 s62, s0 // (op 6899) reg expression
s_mov_b32 s63, 131072 // (op 6899) call()


s_lshl_b64 s[0:1], s[24:25], 2 // (op 6226) 
v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 6226) 


v_mov_b32 v32, v24 // (op 6226) 
buffer_load_dwordx4 v[34:37], v32, s[52:55], 0 offen // (op 6226) Load value
v_add_u32 v32, v32, v30 // (op 6226) 
buffer_load_dwordx4 v[38:41], v32, s[52:55], 0 offen // (op 6226) Load value



v_mov_b64 v[30:31], s[32:33] // (op 6229) convert

v_mov_b32 v32, v28 // (op 6229) 
buffer_load_ubyte v42, v32, s[60:63], 0 offen offset: 0 // (op 6229) Load value
v_add_u32 v32, v32, v30 // (op 6229) 
buffer_load_ubyte v43, v32, s[60:63], 0 offen offset: 0 // (op 6229) Load value





buffer_load_dwordx4 v[30:33], v22, s[48:51], 0 offen // (op 6232) Load value



buffer_load_ubyte v44, v26, s[56:59], 0 offen offset: 0 // (op 6235) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 6238) 
v_lshrrev_b64 v[46:47], 3, s[0:1] // (op 6238) 


v_mov_b32 v45, v24 // (op 6238) 
buffer_load_dwordx4 v[48:51], v45, s[52:55], 0 offen offset:64 // (op 6238) Load value
v_add_u32 v45, v45, v46 // (op 6238) 
buffer_load_dwordx4 v[52:55], v45, s[52:55], 0 offen offset:64 // (op 6238) Load value



v_mov_b64 v[46:47], s[32:33] // (op 6241) convert

v_mov_b32 v45, v28 // (op 6241) 
buffer_load_ubyte v56, v45, s[60:63], 0 offen offset:4 // (op 6241) Load value
v_add_u32 v45, v45, v46 // (op 6241) 
buffer_load_ubyte v57, v45, s[60:63], 0 offen offset:4 // (op 6241) Load value





buffer_load_dwordx4 v[58:61], v22, s[48:51], 0 offen offset:64 // (op 6244) Load value



buffer_load_ubyte v45, v26, s[56:59], 0 offen offset:4 // (op 6247) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 6250) 
v_lshrrev_b64 v[46:47], 3, s[0:1] // (op 6250) 


v_mov_b32 v62, v24 // (op 6250) 
buffer_load_dwordx4 v[64:67], v62, s[52:55], 0 offen offset:128 // (op 6250) Load value
v_add_u32 v62, v62, v46 // (op 6250) 
buffer_load_dwordx4 v[68:71], v62, s[52:55], 0 offen offset:128 // (op 6250) Load value



v_mov_b64 v[46:47], s[32:33] // (op 6253) convert

v_mov_b32 v62, v28 // (op 6253) 
buffer_load_ubyte v72, v62, s[60:63], 0 offen offset:8 // (op 6253) Load value
v_add_u32 v62, v62, v46 // (op 6253) 
buffer_load_ubyte v73, v62, s[60:63], 0 offen offset:8 // (op 6253) Load value





buffer_load_dwordx4 v[74:77], v22, s[48:51], 0 offen offset:128 // (op 6256) Load value



buffer_load_ubyte v46, v26, s[56:59], 0 offen offset:8 // (op 6259) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 6262) 
v_lshrrev_b64 v[62:63], 3, s[0:1] // (op 6262) 


v_mov_b32 v47, v24 // (op 6262) 
buffer_load_dwordx4 v[78:81], v47, s[52:55], 0 offen offset:192 // (op 6262) Load value
v_add_u32 v47, v47, v62 // (op 6262) 
buffer_load_dwordx4 v[82:85], v47, s[52:55], 0 offen offset:192 // (op 6262) Load value




v_mov_b64 v[62:63], s[32:33] // (op 6265) convert

v_mov_b32 v24, v28 // (op 6265) 
buffer_load_ubyte v86, v24, s[60:63], 0 offen offset:12 // (op 6265) Load value
v_add_u32 v24, v24, v62 // (op 6265) 
buffer_load_ubyte v87, v24, s[60:63], 0 offen offset:12 // (op 6265) Load value






buffer_load_dwordx4 v[88:91], v22, s[48:51], 0 offen offset:192 // (op 6268) Load value




buffer_load_ubyte v22, v26, s[56:59], 0 offen offset:12 // (op 6271) Load value

v_and_b32 v24, -4, v1 // (op 7027) 
v_sub_u32 v26, v1, v24 // (op 7027) 

v_lshlrev_b32 v24, 5, v26 // (op 7027) 

v_lshrrev_b32 v26, 2, v1 // (op 7027) 
v_lshl_add_u32 v28, v26, 8, v24 // (op 7027) 


v_lshlrev_b32 v24, 2, v28 // (op 7027) 

v_lshrrev_b32 v26, 3, v24 // (op 7027) 




s_waitcnt vmcnt(23)
ds_write_b128 v26, v[34:37] // (op 6273) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v26, v[38:41] offset:64 // (op 6273) Store local data 

v_and_b32 v24, -4, v1 // (op 7143) 
v_sub_u32 v26, v1, v24 // (op 7143) 

v_lshrrev_b32 v24, 2, v1 // (op 7143) 
v_lshl_add_u32 v28, v24, 3, v26 // (op 7143) 





s_waitcnt vmcnt(21)
ds_write_b8 v28, v42 offset:8192 // (op 6274) Store local data 
s_waitcnt vmcnt(20)
ds_write_b8 v28, v43 offset:8196 // (op 6274) Store local data 

v_and_b32 v24, -4, v1 // (op 6969) 
v_sub_u32 v26, v1, v24 // (op 6969) 

v_lshlrev_b32 v24, 5, v26 // (op 6969) 

v_lshrrev_b32 v26, 2, v1 // (op 6969) 
v_lshl_add_u32 v28, v26, 7, v24 // (op 6969) 


v_lshlrev_b32 v24, 2, v28 // (op 6969) 

v_lshrrev_b32 v26, 3, v24 // (op 6969) 




s_waitcnt vmcnt(19)
ds_write_b128 v26, v[30:33] offset:8704 // (op 6275) Store local data 

v_and_b32 v24, -4, v1 // (op 7085) 
v_sub_u32 v26, v1, v24 // (op 7085) 

v_and_b32 v24, 4294967292, v1 // (op 7085) 
v_add_u32 v28, v24, v26 // (op 7085) 





s_waitcnt vmcnt(18)
ds_write_b8 v28, v44 offset:12800 // (op 6276) Store local data 

v_and_b32 v24, -4, v1 // (op 6940) 
v_sub_u32 v26, v1, v24 // (op 6940) 

v_lshlrev_b32 v24, 5, v26 // (op 6940) 

v_lshrrev_b32 v26, 2, v1 // (op 6940) 
v_lshl_add_u32 v28, v26, 7, v24 // (op 6940) 


v_lshlrev_b32 v24, 2, v28 // (op 6940) 

v_lshrrev_b32 v26, 3, v24 // (op 6940) 

v_and_b32 v24, -4, v1 // (op 6998) 
v_sub_u32 v28, v1, v24 // (op 6998) 

v_lshlrev_b32 v24, 5, v28 // (op 6998) 

v_lshrrev_b32 v28, 2, v1 // (op 6998) 
v_lshl_add_u32 v47, v28, 8, v24 // (op 6998) 


v_lshlrev_b32 v24, 2, v47 // (op 6998) 

v_lshrrev_b32 v28, 3, v24 // (op 6998) 

v_and_b32 v24, -4, v1 // (op 7056) 
v_sub_u32 v47, v1, v24 // (op 7056) 

v_and_b32 v24, 4294967292, v1 // (op 7056) 
v_add_u32 v62, v24, v47 // (op 7056) 


v_and_b32 v24, -4, v1 // (op 7114) 
v_sub_u32 v47, v1, v24 // (op 7114) 

v_lshrrev_b32 v24, 2, v1 // (op 7114) 
v_lshl_add_u32 v63, v24, 3, v47 // (op 7114) 


v_and_b32 v24, -4, v1 // (op 7195) 
v_sub_u32 v47, v1, v24 // (op 7195) 

v_lshlrev_b32 v24, 5, v47 // (op 7195) 

v_lshrrev_b32 v47, 2, v1 // (op 7195) 
v_lshl_add_u32 v92, v47, 7, v24 // (op 7195) 


v_lshlrev_b32 v24, 2, v92 // (op 7195) 

v_lshrrev_b32 v47, 3, v24 // (op 7195) 

v_and_b32 v24, -4, v1 // (op 7224) 
v_sub_u32 v92, v1, v24 // (op 7224) 

v_lshlrev_b32 v24, 5, v92 // (op 7224) 

v_lshrrev_b32 v92, 2, v1 // (op 7224) 
v_lshl_add_u32 v93, v92, 8, v24 // (op 7224) 


v_lshlrev_b32 v24, 2, v93 // (op 7224) 

v_lshrrev_b32 v92, 3, v24 // (op 7224) 

v_and_b32 v24, -4, v1 // (op 7253) 
v_sub_u32 v93, v1, v24 // (op 7253) 

v_and_b32 v24, 4294967292, v1 // (op 7253) 
v_add_u32 v94, v24, v93 // (op 7253) 


v_and_b32 v24, -4, v1 // (op 7282) 
v_sub_u32 v93, v1, v24 // (op 7282) 

v_lshrrev_b32 v24, 2, v1 // (op 7282) 
v_lshl_add_u32 v95, v24, 3, v93 // (op 7282) 


v_and_b32 v24, -4, v1 // (op 7311) 
v_sub_u32 v93, v1, v24 // (op 7311) 

v_lshlrev_b32 v24, 5, v93 // (op 7311) 

v_lshrrev_b32 v93, 2, v1 // (op 7311) 
v_lshl_add_u32 v96, v93, 7, v24 // (op 7311) 


v_lshlrev_b32 v24, 2, v96 // (op 7311) 

v_lshrrev_b32 v93, 3, v24 // (op 7311) 

v_and_b32 v24, -4, v1 // (op 7340) 
v_sub_u32 v96, v1, v24 // (op 7340) 

v_lshlrev_b32 v24, 5, v96 // (op 7340) 

v_lshrrev_b32 v96, 2, v1 // (op 7340) 
v_lshl_add_u32 v97, v96, 8, v24 // (op 7340) 


v_lshlrev_b32 v24, 2, v97 // (op 7340) 

v_lshrrev_b32 v96, 3, v24 // (op 7340) 

v_and_b32 v24, -4, v1 // (op 7369) 
v_sub_u32 v97, v1, v24 // (op 7369) 

v_and_b32 v24, 4294967292, v1 // (op 7369) 
v_add_u32 v98, v24, v97 // (op 7369) 


v_and_b32 v24, -4, v1 // (op 7398) 
v_sub_u32 v97, v1, v24 // (op 7398) 

v_lshrrev_b32 v24, 2, v1 // (op 7398) 
v_lshl_add_u32 v99, v24, 3, v97 // (op 7398) 


v_and_b32 v24, -4, v1 // (op 7427) 
v_sub_u32 v97, v1, v24 // (op 7427) 

v_lshlrev_b32 v24, 5, v97 // (op 7427) 

v_lshrrev_b32 v97, 2, v1 // (op 7427) 
v_lshl_add_u32 v100, v97, 7, v24 // (op 7427) 


v_lshlrev_b32 v24, 2, v100 // (op 7427) 

v_lshrrev_b32 v97, 3, v24 // (op 7427) 

v_and_b32 v24, -4, v1 // (op 7456) 
v_sub_u32 v100, v1, v24 // (op 7456) 

v_lshlrev_b32 v24, 5, v100 // (op 7456) 

v_lshrrev_b32 v100, 2, v1 // (op 7456) 
v_lshl_add_u32 v101, v100, 8, v24 // (op 7456) 


v_lshlrev_b32 v24, 2, v101 // (op 7456) 

v_lshrrev_b32 v100, 3, v24 // (op 7456) 

v_and_b32 v24, -4, v1 // (op 7485) 
v_sub_u32 v101, v1, v24 // (op 7485) 

v_and_b32 v24, 4294967292, v1 // (op 7485) 
v_add_u32 v102, v24, v101 // (op 7485) 


v_and_b32 v24, -4, v1 // (op 7514) 
v_sub_u32 v101, v1, v24 // (op 7514) 

v_lshrrev_b32 v24, 2, v1 // (op 7514) 
v_lshl_add_u32 v103, v24, 3, v101 // (op 7514) 


s_mov_b32 s0, 0 // (op 1611) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1610) 
s_mov_b32 s1, s6 // (op 1610) convert

s_and_b32 s6, s1, 4294967292 // (op 1610) 

s_cmp_lt_u32 s0, s6 // (op 1610) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_1610 // (op 1610) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_1610 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoop_1610:

s_barrier  // (op 7634) 
s_add_u32 s0, s0, 4 // (op 1612) 



s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[104:105], 3, s[6:7] // (op 10) 


v_mov_b32 v24, v25 // (op 10) 
v_mov_b32 v101, v24 // (op 10) 
ds_read_b128 v[106:109], v3 offset:8704 // (op 72) Load local data 
buffer_load_dwordx4 v[34:37], v101, s[52:55], 0 offen offset:256 // (op 10) Load value
s_barrier  // (op 7647) 
v_add_u32 v101, v101, v104 // (op 10) 



buffer_load_dwordx4 v[38:41], v101, s[52:55], 0 offen offset:256 // (op 10) Load value


ds_read_b128 v[110:113], v4 // (op 77) Load local data 

s_barrier  // (op 7661) 
v_mov_b64 v[104:105], s[32:33] // (op 25) convert




ds_read_u8 v101, v0 offset:12800 // (op 82) Load local data 
v_mov_b32 v114, v29 // (op 25) 
s_barrier  // (op 7675) 
v_mov_b32 v115, v114 // (op 25) 



buffer_load_ubyte v42, v115, s[60:63], 0 offen offset:16 // (op 25) Load value
v_add_u32 v115, v115, v104 // (op 25) 
ds_read_u8 v116, v5 offset:8192 // (op 87) Load local data 
buffer_load_ubyte v43, v115, s[60:63], 0 offen offset:16 // (op 25) Load value







v_mov_b32 v115, v23 // (op 4) 

ds_read_b128 v[118:121], v4 offset:2048 // (op 521) Load local data 
buffer_load_dwordx4 v[30:33], v115, s[48:51], 0 offen offset:256 // (op 4) Load value





ds_read_u8 v117, v5 offset:8320 // (op 525) Load local data 
v_mov_b32 v104, v27 // (op 17) 




ds_read_b128 v[122:125], v4 offset:4096 // (op 819) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[106:109], v[110:113], a[0:3], v101, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 90) 



buffer_load_ubyte v44, v104, s[56:59], 0 offen offset:16 // (op 17) Load value
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[106:109], v[118:121], a[4:7], v101, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 526) 
s_barrier  // (op 7698) 
ds_read_u8 v105, v5 offset:8448 // (op 823) Load local data 






ds_read_b128 v[126:129], v4 offset:6144 // (op 1117) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[106:109], v[122:125], a[8:11], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 824) 



ds_read_u8 v130, v5 offset:8576 // (op 1121) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v92, v[48:51] offset:13056 // (op 257) Store local data 



ds_read_b128 v[132:135], v3 offset:10752 // (op 1644) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[106:109], v[126:129], a[12:15], v101, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1122) 





ds_read_u8 v101, v0 offset:12928 // (op 1647) Load local data 
s_waitcnt vmcnt(22)
ds_write_b128 v92, v[52:55] offset:13120 // (op 257) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[110:113], a[16:19], v101, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1925) 


s_barrier  // (op 7710) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[118:121], a[20:23], v101, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1833) 
s_waitcnt vmcnt(21)
ds_write_b8 v95, v56 offset:21248 // (op 259) Store local data 


s_waitcnt vmcnt(20)
ds_write_b8 v95, v57 offset:21252 // (op 259) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[122:125], a[24:27], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1741) 


s_barrier  // (op 7692) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[126:129], a[28:31], v101, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1649) 
s_waitcnt vmcnt(19)
ds_write_b128 v47, v[58:61] offset:21760 // (op 247) Store local data 




s_barrier  // (op 7704) 



s_waitcnt vmcnt(18)
ds_write_b8 v94, v45 offset:25856 // (op 258) Store local data 

s_lshl_b64 s[6:7], s[24:25], 2 // (op 260) 
v_lshrrev_b64 v[112:113], 3, s[6:7] // (op 260) 


v_mov_b32 v101, v24 // (op 260) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 7689) 
buffer_load_dwordx4 v[48:51], v101, s[52:55], 0 offen offset:320 // (op 260) Load value



v_add_u32 v101, v101, v112 // (op 260) 
ds_read_b128 v[108:111], v7 offset:21760 // (op 249) Load local data 
buffer_load_dwordx4 v[52:55], v101, s[52:55], 0 offen offset:320 // (op 260) Load value



s_barrier  // (op 7695) 
v_mov_b64 v[106:107], s[32:33] // (op 262) convert




v_mov_b32 v101, v114 // (op 262) 
ds_read_b128 v[116:119], v8 offset:13056 // (op 251) Load local data 
buffer_load_ubyte v56, v101, s[60:63], 0 offen offset:20 // (op 262) Load value
v_add_u32 v101, v101, v106 // (op 262) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7701) 
buffer_load_ubyte v57, v101, s[60:63], 0 offen offset:20 // (op 262) Load value








ds_read_u8 v101, v6 offset:25856 // (op 253) Load local data 
buffer_load_dwordx4 v[58:61], v115, s[48:51], 0 offen offset:320 // (op 246) Load value



s_barrier  // (op 7707) 
buffer_load_ubyte v45, v104, s[56:59], 0 offen offset:20 // (op 261) Load value



s_barrier  // (op 7722) 
ds_read_u8 v105, v9 offset:21248 // (op 255) Load local data 






ds_read_b128 v[120:123], v8 offset:15104 // (op 533) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[116:119], a[0:3], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 256) 



ds_read_u8 v106, v9 offset:21376 // (op 537) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v96, v[64:67] offset:26112 // (op 297) Store local data 



ds_read_b128 v[124:127], v8 offset:17152 // (op 831) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[120:123], a[4:7], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 527) 



ds_read_u8 v107, v9 offset:21504 // (op 835) Load local data 
s_waitcnt vmcnt(22)
ds_write_b128 v96, v[68:71] offset:26176 // (op 297) Store local data 



ds_read_b128 v[128:131], v8 offset:19200 // (op 1129) Load local data 
s_barrier  // (op 7734) 






s_barrier  // (op 7839) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[124:127], a[8:11], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 825) 
ds_read_u8 v112, v9 offset:21632 // (op 1133) Load local data 
s_waitcnt vmcnt(21)
ds_write_b8 v99, v72 offset:34304 // (op 299) Store local data 
s_waitcnt vmcnt(20)
ds_write_b8 v99, v73 offset:34308 // (op 299) Store local data 



s_barrier  // (op 7843) 
s_barrier  // (op 7716) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[128:131], a[12:15], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1123) 





ds_read_b128 v[108:111], v7 offset:23808 // (op 1654) Load local data 
s_waitcnt vmcnt(19)
ds_write_b128 v93, v[74:77] offset:13056 // (op 287) Store local data 



s_barrier  // (op 7887) 
s_barrier  // (op 7728) 
ds_read_u8 v101, v6 offset:25984 // (op 1657) Load local data 



s_waitcnt vmcnt(18)
ds_write_b8 v98, v46 offset:17152 // (op 298) Store local data 
s_barrier  // (op 7891) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[116:119], a[16:19], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1926) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[120:123], a[20:23], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1834) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[124:127], a[24:27], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1742) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[128:131], a[28:31], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1650) 





s_barrier  // (op 7713) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 300) 



v_lshrrev_b64 v[112:113], 3, s[6:7] // (op 300) 


v_mov_b32 v101, v24 // (op 300) 
ds_read_b128 v[108:111], v11 offset:13056 // (op 289) Load local data 
buffer_load_dwordx4 v[64:67], v101, s[52:55], 0 offen offset:384 // (op 300) Load value
s_barrier  // (op 7719) 
v_add_u32 v101, v101, v112 // (op 300) 



buffer_load_dwordx4 v[68:71], v101, s[52:55], 0 offen offset:384 // (op 300) Load value


ds_read_b128 v[116:119], v12 offset:26112 // (op 291) Load local data 

v_mov_b64 v[106:107], s[32:33] // (op 302) convert

v_mov_b32 v101, v114 // (op 302) 
buffer_load_ubyte v72, v101, s[60:63], 0 offen offset:24 // (op 302) Load value
v_add_u32 v101, v101, v106 // (op 302) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7725) 
buffer_load_ubyte v73, v101, s[60:63], 0 offen offset:24 // (op 302) Load value








ds_read_u8 v101, v10 offset:17152 // (op 293) Load local data 
buffer_load_dwordx4 v[74:77], v115, s[48:51], 0 offen offset:384 // (op 286) Load value



s_barrier  // (op 7731) 
buffer_load_ubyte v46, v104, s[56:59], 0 offen offset:24 // (op 301) Load value



s_barrier  // (op 7746) 
ds_read_u8 v105, v13 offset:34304 // (op 295) Load local data 






ds_read_b128 v[120:123], v12 offset:28160 // (op 541) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[116:119], a[0:3], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 296) 



ds_read_u8 v106, v13 offset:34432 // (op 545) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v100, v[78:81] offset:17408 // (op 337) Store local data 



ds_read_b128 v[124:127], v12 offset:30208 // (op 839) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[120:123], a[4:7], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 528) 



ds_read_u8 v107, v13 offset:34560 // (op 843) Load local data 
s_waitcnt vmcnt(22)
ds_write_b128 v100, v[82:85] offset:17472 // (op 337) Store local data 



ds_read_b128 v[128:131], v12 offset:32256 // (op 1137) Load local data 
s_barrier  // (op 7758) 






s_barrier  // (op 7847) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[124:127], a[8:11], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 826) 
ds_read_u8 v112, v13 offset:34688 // (op 1141) Load local data 
s_waitcnt vmcnt(21)
ds_write_b8 v103, v86 offset:25600 // (op 339) Store local data 
s_waitcnt vmcnt(20)
ds_write_b8 v103, v87 offset:25604 // (op 339) Store local data 



s_barrier  // (op 7851) 
s_barrier  // (op 7740) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[128:131], a[12:15], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1124) 





ds_read_b128 v[108:111], v11 offset:15104 // (op 1660) Load local data 
s_waitcnt vmcnt(19)
ds_write_b128 v97, v[88:91] offset:26112 // (op 327) Store local data 



s_barrier  // (op 7895) 
s_barrier  // (op 7752) 
ds_read_u8 v101, v10 offset:17280 // (op 1663) Load local data 



s_waitcnt vmcnt(18)
ds_write_b8 v102, v22 offset:30208 // (op 338) Store local data 
s_barrier  // (op 7899) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[116:119], a[16:19], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1927) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[120:123], a[20:23], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1835) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[124:127], a[24:27], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1743) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[128:131], a[28:31], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1651) 





s_barrier  // (op 7737) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 340) 



v_lshrrev_b64 v[112:113], 3, s[6:7] // (op 340) 


v_mov_b32 v101, v24 // (op 340) 
ds_read_b128 v[108:111], v15 offset:26112 // (op 329) Load local data 
buffer_load_dwordx4 v[78:81], v101, s[52:55], 0 offen offset:448 // (op 340) Load value
s_barrier  // (op 7743) 
v_add_u32 v101, v101, v112 // (op 340) 



buffer_load_dwordx4 v[82:85], v101, s[52:55], 0 offen offset:448 // (op 340) Load value


ds_read_b128 v[116:119], v16 offset:17408 // (op 331) Load local data 


v_add_u32 v25, 256, v25 // (op 6761) 
v_mov_b64 v[106:107], s[32:33] // (op 342) convert

v_mov_b32 v24, v114 // (op 342) 
buffer_load_ubyte v86, v24, s[60:63], 0 offen offset:28 // (op 342) Load value
v_add_u32 v24, v24, v106 // (op 342) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7749) 
buffer_load_ubyte v87, v24, s[60:63], 0 offen offset:28 // (op 342) Load value






ds_read_u8 v24, v14 offset:30208 // (op 333) Load local data 



s_barrier  // (op 7755) 
buffer_load_dwordx4 v[88:91], v115, s[48:51], 0 offen offset:448 // (op 326) Load value




v_add_u32 v29, 16, v29 // (op 6889) 



v_add_u32 v23, 256, v23 // (op 6697) 
ds_read_u8 v101, v17 offset:25600 // (op 335) Load local data 
buffer_load_ubyte v22, v104, s[56:59], 0 offen offset:28 // (op 341) Load value




ds_read_b128 v[104:107], v16 offset:19456 // (op 549) Load local data 



s_barrier  // (op 7650) 
v_add_u32 v27, 16, v27 // (op 6825) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[116:119], a[0:3], v24, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 336) 
ds_read_u8 v112, v17 offset:25728 // (op 553) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v28, v[34:37] // (op 13) Store local data 



ds_read_b128 v[120:123], v16 offset:21504 // (op 847) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[104:107], a[4:7], v24, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 529) 



ds_read_u8 v113, v17 offset:25856 // (op 851) Load local data 
s_waitcnt vmcnt(22)
ds_write_b128 v28, v[38:41] offset:64 // (op 13) Store local data 



ds_read_b128 v[124:127], v16 offset:23552 // (op 1145) Load local data 
s_barrier  // (op 7678) 






s_barrier  // (op 7855) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[120:123], a[8:11], v24, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 827) 
ds_read_u8 v114, v17 offset:25984 // (op 1149) Load local data 
s_waitcnt vmcnt(21)
ds_write_b8 v63, v42 offset:8192 // (op 30) Store local data 
s_waitcnt vmcnt(20)
ds_write_b8 v63, v43 offset:8196 // (op 30) Store local data 



s_barrier  // (op 7859) 
s_barrier  // (op 7637) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[124:127], a[12:15], v24, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1125) 





ds_read_b128 v[108:111], v15 offset:28160 // (op 1666) Load local data 
s_waitcnt vmcnt(19)
ds_write_b128 v26, v[30:33] offset:8704 // (op 6) Store local data 



s_barrier  // (op 7903) 
s_barrier  // (op 7664) 
ds_read_u8 v24, v14 offset:30336 // (op 1669) Load local data 



s_waitcnt vmcnt(18)
ds_write_b8 v62, v44 offset:12800 // (op 21) Store local data 
s_barrier  // (op 7907) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[116:119], a[16:19], v24, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1928) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[104:107], a[20:23], v24, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1836) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[120:123], a[24:27], v24, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1744) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[124:127], a[28:31], v24, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1652) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1610) 
s_mov_b32 s1, s6 // (op 1610) convert

s_and_b32 s6, s1, 4294967292 // (op 1610) 

s_cmp_lt_u32 s0, s6 // (op 1610) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoop_1610 // (op 1610) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoop_1610 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_1610:

























v_lshrrev_b32 v47, 2, v1 // (op 6653) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6653) 
v_lshl_add_u32 v23, s16, 6, v47 // (op 6653) 

s_lshr_b32 s6, s0, 2 // (op 6653) 

v_and_b32 v47, -4, v1 // (op 6653) 
v_sub_u32 v24, v1, v47 // (op 6653) 

v_lshlrev_b32 v47, 5, v24 // (op 6653) 

v_lshl_add_u32 v24, s6, 9, v47 // (op 6653) 


v_mov_b32 v63, 0 // (op 6653) convert
v_mov_b32 v62, v24 // (op 6653) convert

v_mul_lo_u32 v25, v23, s13 // (op 6653) most significant: low of low * high
v_mul_hi_u32 v47, v23, s12 // (op 6653) most significant: high of low * low
v_mul_lo_u32 v24, v23, s12 // (op 6653) least significant: low of low * low
v_add_u32 v25, v25, v47 // (op 6653) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v62 // (op 6653) least significant half
v_addc_co_u32 v25, s[0:1], v25, v63, s[0:1] // (op 6653) most significant half



v_lshlrev_b64 v[62:63], 2, v[24:25] // (op 6653) 

v_lshrrev_b64 v[24:25], 3, v[62:63] // (op 6653) 

v_mov_b32 v23, v24 // (op 6653) convert

s_mul_i32 s1, s12, s23 // (op 6655) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6655) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6655) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6655) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6655) most significant: sum
s_add_u32 s1, s1, s7 // (op 6655) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6655) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6655) 

s_mov_b32 s48, s20 // (op 6655) reg expression
s_mov_b32 s49, s21 // (op 6655) reg expression
s_mov_b32 s50, s0 // (op 6655) reg expression
s_mov_b32 s51, 131072 // (op 6655) call()

v_and_b32 v47, -4, v1 // (op 6727) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6727) 
v_sub_u32 v24, v1, v47 // (op 6727) 

s_lshr_b32 s6, s0, 2 // (op 6727) 

v_lshlrev_b32 v47, 5, v24 // (op 6727) 

v_lshl_add_u32 v24, s6, 9, v47 // (op 6727) 


v_mov_b32 v63, 0 // (op 6727) convert
v_mov_b32 v62, v24 // (op 6727) convert

v_lshrrev_b32 v47, 2, v1 // (op 6727) 
v_lshlrev_b32 v24, 1, v47 // (op 6727) 

v_lshl_add_u32 v47, s17, 7, v24 // (op 6727) 

v_mul_lo_u32 v25, v47, s25 // (op 6727) most significant: low of low * high
v_mul_hi_u32 v26, v47, s24 // (op 6727) most significant: high of low * low
v_mul_lo_u32 v24, v47, s24 // (op 6727) least significant: low of low * low
v_add_u32 v25, v25, v26 // (op 6727) most significant: sum


v_add_co_u32 v26, s[0:1], v62, v24 // (op 6727) least significant half
v_addc_co_u32 v27, s[0:1], v63, v25, s[0:1] // (op 6727) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 6727) 


v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 6727) 

v_mov_b32 v47, v24 // (op 6727) convert

s_mul_i32 s1, s24, s9 // (op 6729) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6729) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6729) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6729) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6729) most significant: sum
s_add_u32 s1, s1, s7 // (op 6729) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6729) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6729) 

s_mov_b32 s52, s26 // (op 6729) reg expression
s_mov_b32 s53, s27 // (op 6729) reg expression
s_mov_b32 s54, s0 // (op 6729) reg expression
s_mov_b32 s55, 131072 // (op 6729) call()

v_lshrrev_b32 v24, 2, v1 // (op 6791) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6791) 
v_lshl_add_u32 v25, s16, 6, v24 // (op 6791) 

s_lshr_b32 s6, s0, 2 // (op 6791) 

v_and_b32 v24, -4, v1 // (op 6791) 
v_sub_u32 v26, v1, v24 // (op 6791) 

v_lshl_add_u32 v24, s6, 4, v26 // (op 6791) 


v_mul_lo_u32 v26, v25, s28 // (op 6791) 
v_add_u32 v26, v26, v24 // (op 6791) 


s_mul_i32 s1, s28, s23 // (op 6793) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6793) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6793) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6793) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6793) most significant: sum
s_add_u32 s1, s1, s7 // (op 6793) most significant: sum


s_mov_b32 s56, s30 // (op 6793) reg expression
s_mov_b32 s57, s31 // (op 6793) reg expression
s_mov_b32 s58, s0 // (op 6793) reg expression
s_mov_b32 s59, 131072 // (op 6793) call()

v_and_b32 v24, -4, v1 // (op 6855) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6855) 
v_sub_u32 v25, v1, v24 // (op 6855) 

s_lshr_b32 s6, s0, 2 // (op 6855) 

v_lshrrev_b32 v24, 2, v1 // (op 6855) 
v_lshl_add_u32 v27, s6, 4, v25 // (op 6855) 


v_lshlrev_b32 v25, 1, v24 // (op 6855) 

v_lshl_add_u32 v24, s17, 7, v25 // (op 6855) 

v_mul_lo_u32 v25, v24, s32 // (op 6855) 

v_add_u32 v24, v27, v25 // (op 6855) 


s_mul_i32 s1, s32, s9 // (op 6857) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6857) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6857) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6857) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6857) most significant: sum
s_add_u32 s1, s1, s7 // (op 6857) most significant: sum


s_mov_b32 s60, s34 // (op 6857) reg expression
s_mov_b32 s61, s35 // (op 6857) reg expression
s_mov_b32 s62, s0 // (op 6857) reg expression
s_mov_b32 s63, 131072 // (op 6857) call()

v_and_b32 v25, -4, v1 // (op 6953) 
v_sub_u32 v27, v1, v25 // (op 6953) 

v_lshlrev_b32 v25, 5, v27 // (op 6953) 

v_lshrrev_b32 v27, 2, v1 // (op 6953) 
v_lshl_add_u32 v28, v27, 7, v25 // (op 6953) 


v_lshlrev_b32 v25, 2, v28 // (op 6953) 

v_lshrrev_b32 v27, 3, v25 // (op 6953) 

v_and_b32 v25, -4, v1 // (op 7011) 
v_sub_u32 v28, v1, v25 // (op 7011) 

v_lshlrev_b32 v25, 5, v28 // (op 7011) 

v_lshrrev_b32 v28, 2, v1 // (op 7011) 
v_lshl_add_u32 v29, v28, 8, v25 // (op 7011) 


v_lshlrev_b32 v25, 2, v29 // (op 7011) 

v_lshrrev_b32 v28, 3, v25 // (op 7011) 

v_and_b32 v25, -4, v1 // (op 7069) 
v_sub_u32 v29, v1, v25 // (op 7069) 

v_and_b32 v25, 4294967292, v1 // (op 7069) 
v_add_u32 v30, v25, v29 // (op 7069) 


v_and_b32 v25, -4, v1 // (op 7127) 
v_sub_u32 v29, v1, v25 // (op 7127) 

v_lshrrev_b32 v25, 2, v1 // (op 7127) 
v_lshl_add_u32 v31, v25, 3, v29 // (op 7127) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 1616) 
s_and_b32 s6, s0, 4294967292 // (op 1616) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 1615) 
s_mov_b32 s7, s0 // (op 1615) convert

s_cmp_lt_u32 s6, s7 // (op 1615) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_1615 // (op 1615) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_1615 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_1615:


s_add_u32 s6, s6, 1 // (op 1617) 
v_mov_b32 v22, v23 // (op 180) 

buffer_load_dwordx4 v[32:35], v22, s[48:51], 0 offen // (op 180) Load value

v_add_u32 v23, 64, v23 // (op 6643) 

s_barrier  // (op 7640) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 194) 



v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 194) 


v_mov_b32 v22, v47 // (op 194) 
v_mov_b32 v25, v22 // (op 194) 
buffer_load_dwordx4 v[38:41], v25, s[52:55], 0 offen // (op 194) Load value
v_add_u32 v25, v25, v36 // (op 194) 
buffer_load_dwordx4 v[42:45], v25, s[52:55], 0 offen // (op 194) Load value



v_add_u32 v47, 64, v47 // (op 6719) 
s_waitcnt vmcnt(2)
ds_write_b128 v27, v[32:35] offset:8704 // (op 181) Store local data 



v_mov_b32 v22, v26 // (op 195) 

s_barrier  // (op 7653) 
buffer_load_ubyte v25, v22, s[56:59], 0 offen offset: 0 // (op 195) Load value




v_add_u32 v26, 4, v26 // (op 6783) 
s_waitcnt vmcnt(2)
ds_write_b128 v28, v[38:41] // (op 191) Store local data 

v_mov_b64 v[32:33], s[32:33] // (op 196) convert

v_mov_b32 v22, v24 // (op 196) 
v_mov_b32 v29, v22 // (op 196) 
buffer_load_ubyte v34, v29, s[60:63], 0 offen offset: 0 // (op 196) Load value
v_add_u32 v29, v29, v32 // (op 196) 
buffer_load_ubyte v35, v29, s[60:63], 0 offen offset: 0 // (op 196) Load value



v_add_u32 v24, 4, v24 // (op 6847) 
s_waitcnt vmcnt(3)
ds_write_b128 v28, v[42:45] offset:64 // (op 191) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 7643) 



s_barrier  // (op 7667) 
ds_read_b128 v[36:39], v3 offset:8704 // (op 183) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 7657) 
s_waitcnt vmcnt(2)
ds_write_b8 v30, v25 offset:12800 // (op 192) Store local data 




ds_read_b128 v[40:43], v4 // (op 185) Load local data 
s_barrier  // (op 7681) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 7671) 






ds_read_u8 v22, v0 offset:12800 // (op 187) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v31, v34 offset:8192 // (op 193) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v31, v35 offset:8196 // (op 193) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 7685) 



ds_read_u8 v25, v5 offset:8192 // (op 189) Load local data 



ds_read_b128 v[32:35], v4 offset:2048 // (op 500) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:39], v[40:43], a[0:3], v22, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 190) 



ds_read_u8 v29, v5 offset:8320 // (op 504) Load local data 



ds_read_b128 v[48:51], v4 offset:4096 // (op 798) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:39], v[32:35], a[4:7], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 505) 



ds_read_u8 v44, v5 offset:8448 // (op 802) Load local data 



ds_read_b128 v[52:55], v4 offset:6144 // (op 1096) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:39], v[48:51], a[8:11], v22, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 803) 



ds_read_u8 v45, v5 offset:8576 // (op 1100) Load local data 



ds_read_b128 v[56:59], v3 offset:10752 // (op 1528) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:39], v[52:55], a[12:15], v22, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1101) 





ds_read_u8 v22, v0 offset:12928 // (op 1531) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[40:43], a[16:19], v22, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1533) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[32:35], a[20:23], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1564) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v22, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1595) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[52:55], a[28:31], v22, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1626) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 1615) 
s_mov_b32 s7, s0 // (op 1615) convert

s_cmp_lt_u32 s6, s7 // (op 1615) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_1615 // (op 1615) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_1615 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_1615:























s_barrier  // (op 7871) 

















v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_135
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



v_accvgpr_read v14, a4 // (op 3049) 
v_accvgpr_read v15, a5 // (op 3049) 
v_accvgpr_read v16, a6 // (op 3049) 
v_accvgpr_read v17, a7 // (op 3049) 
v_mul_f32 v10, v19, v14 // (op 3049) 
v_mul_f32 v11, v19, v15 // (op 3049) 
v_mul_f32 v12, v19, v16 // (op 3049) 
v_mul_f32 v13, v19, v17 // (op 3049) 

v_lshrrev_b32 v14, 16, v10 // (op 3051) 
v_lshrrev_b32 v15, 16, v11 // (op 3051) 
v_lshrrev_b32 v16, 16, v12 // (op 3051) 
v_lshrrev_b32 v17, 16, v13 // (op 3051) 


v_pack_b32_f16 v12, v14, v15 // (op 3052) 
v_pack_b32_f16 v13, v16, v17 // (op 3052) 

s_lshl_b32 s0, s4, 6 // (op 3052) 
v_add_u32 v0, v21, s0 // (op 3052) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3052) Store value



v_accvgpr_read v14, a8 // (op 3063) 
v_accvgpr_read v15, a9 // (op 3063) 
v_accvgpr_read v16, a10 // (op 3063) 
v_accvgpr_read v17, a11 // (op 3063) 
v_mul_f32 v10, v19, v14 // (op 3063) 
v_mul_f32 v11, v19, v15 // (op 3063) 
v_mul_f32 v12, v19, v16 // (op 3063) 
v_mul_f32 v13, v19, v17 // (op 3063) 

v_lshrrev_b32 v14, 16, v10 // (op 3065) 
v_lshrrev_b32 v15, 16, v11 // (op 3065) 
v_lshrrev_b32 v16, 16, v12 // (op 3065) 
v_lshrrev_b32 v17, 16, v13 // (op 3065) 


v_pack_b32_f16 v12, v14, v15 // (op 3066) 
v_pack_b32_f16 v13, v16, v17 // (op 3066) 

s_lshl_b32 s0, s4, 7 // (op 3066) 
v_add_u32 v0, v21, s0 // (op 3066) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3066) Store value



v_accvgpr_read v14, a12 // (op 3077) 
v_accvgpr_read v15, a13 // (op 3077) 
v_accvgpr_read v16, a14 // (op 3077) 
v_accvgpr_read v17, a15 // (op 3077) 
v_mul_f32 v10, v19, v14 // (op 3077) 
v_mul_f32 v11, v19, v15 // (op 3077) 
v_mul_f32 v12, v19, v16 // (op 3077) 
v_mul_f32 v13, v19, v17 // (op 3077) 

v_lshrrev_b32 v14, 16, v10 // (op 3079) 
v_lshrrev_b32 v15, 16, v11 // (op 3079) 
v_lshrrev_b32 v16, 16, v12 // (op 3079) 
v_lshrrev_b32 v17, 16, v13 // (op 3079) 


v_pack_b32_f16 v12, v14, v15 // (op 3080) 
v_pack_b32_f16 v13, v16, v17 // (op 3080) 

s_lshl_b32 s0, s4, 6 // (op 3080) 
s_mul_i32 s1, 3, s0 // (op 3080) 

v_add_u32 v0, v21, s1 // (op 3080) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3080) Store value



v_accvgpr_read v14, a16 // (op 3113) 
v_accvgpr_read v15, a17 // (op 3113) 
v_accvgpr_read v16, a18 // (op 3113) 
v_accvgpr_read v17, a19 // (op 3113) 
v_mul_f32 v10, v19, v14 // (op 3113) 
v_mul_f32 v11, v19, v15 // (op 3113) 
v_mul_f32 v12, v19, v16 // (op 3113) 
v_mul_f32 v13, v19, v17 // (op 3113) 

v_lshrrev_b32 v14, 16, v10 // (op 3115) 
v_lshrrev_b32 v15, 16, v11 // (op 3115) 
v_lshrrev_b32 v16, 16, v12 // (op 3115) 
v_lshrrev_b32 v17, 16, v13 // (op 3115) 


v_pack_b32_f16 v12, v14, v15 // (op 3117) 
v_pack_b32_f16 v13, v16, v17 // (op 3117) 

v_add_u32 v0, 64, v21 // (op 3117) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3117) Store value



v_accvgpr_read v14, a20 // (op 3127) 
v_accvgpr_read v15, a21 // (op 3127) 
v_accvgpr_read v16, a22 // (op 3127) 
v_accvgpr_read v17, a23 // (op 3127) 
v_mul_f32 v10, v19, v14 // (op 3127) 
v_mul_f32 v11, v19, v15 // (op 3127) 
v_mul_f32 v12, v19, v16 // (op 3127) 
v_mul_f32 v13, v19, v17 // (op 3127) 

v_lshrrev_b32 v14, 16, v10 // (op 3129) 
v_lshrrev_b32 v15, 16, v11 // (op 3129) 
v_lshrrev_b32 v16, 16, v12 // (op 3129) 
v_lshrrev_b32 v17, 16, v13 // (op 3129) 


v_pack_b32_f16 v12, v14, v15 // (op 3119) 
v_pack_b32_f16 v13, v16, v17 // (op 3119) 

s_lshl_b32 s0, s4, 6 // (op 3119) 
s_add_u32 s0, s0, 64 // (op 3119) 
v_add_u32 v0, v21, s0 // (op 3119) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3119) Store value



v_accvgpr_read v14, a24 // (op 3133) 
v_accvgpr_read v15, a25 // (op 3133) 
v_accvgpr_read v16, a26 // (op 3133) 
v_accvgpr_read v17, a27 // (op 3133) 
v_mul_f32 v10, v19, v14 // (op 3133) 
v_mul_f32 v11, v19, v15 // (op 3133) 
v_mul_f32 v12, v19, v16 // (op 3133) 
v_mul_f32 v13, v19, v17 // (op 3133) 

v_lshrrev_b32 v14, 16, v10 // (op 3135) 
v_lshrrev_b32 v15, 16, v11 // (op 3135) 
v_lshrrev_b32 v16, 16, v12 // (op 3135) 
v_lshrrev_b32 v17, 16, v13 // (op 3135) 


v_pack_b32_f16 v12, v14, v15 // (op 3121) 
v_pack_b32_f16 v13, v16, v17 // (op 3121) 

s_lshl_b32 s0, s4, 7 // (op 3121) 
s_add_u32 s0, s0, 64 // (op 3121) 
v_add_u32 v0, v21, s0 // (op 3121) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3121) Store value



v_accvgpr_read v14, a28 // (op 3139) 
v_accvgpr_read v15, a29 // (op 3139) 
v_accvgpr_read v16, a30 // (op 3139) 
v_accvgpr_read v17, a31 // (op 3139) 
v_mul_f32 v10, v19, v14 // (op 3139) 
v_mul_f32 v11, v19, v15 // (op 3139) 
v_mul_f32 v12, v19, v16 // (op 3139) 
v_mul_f32 v13, v19, v17 // (op 3139) 

v_lshrrev_b32 v14, 16, v10 // (op 3141) 
v_lshrrev_b32 v15, 16, v11 // (op 3141) 
v_lshrrev_b32 v16, 16, v12 // (op 3141) 
v_lshrrev_b32 v17, 16, v13 // (op 3141) 


v_pack_b32_f16 v12, v14, v15 // (op 3123) 
v_pack_b32_f16 v13, v16, v17 // (op 3123) 

s_lshl_b32 s0, s4, 6 // (op 3123) 
s_mul_i32 s1, 3, s0 // (op 3123) 
s_add_u32 s1, s1, 64 // (op 3123) 

v_add_u32 v0, v21, s1 // (op 3123) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3123) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v20 // (op 3309) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3309) Load value

v_accvgpr_read v12, a16 // (op 3296) 
v_accvgpr_read v13, a17 // (op 3296) 
v_accvgpr_read v14, a18 // (op 3296) 
v_accvgpr_read v15, a19 // (op 3296) 
v_mul_f32 v8, v19, v12 // (op 3296) 
v_mul_f32 v9, v19, v13 // (op 3296) 
v_mul_f32 v10, v19, v14 // (op 3296) 
v_mul_f32 v11, v19, v15 // (op 3296) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 3308) 
v_lshrrev_b32 v7, 16, v16 // (op 3308) 
v_lshlrev_b32 v6, 16, v6 // (op 3308) 
v_lshlrev_b32 v7, 16, v7 // (op 3308) 
v_mul_f32 v12, v18, v6 // (op 3308) 
v_mul_f32 v13, v18, v7 // (op 3308) 
v_and_b32 v6, 65535, v17 // (op 3308) 
v_lshrrev_b32 v7, 16, v17 // (op 3308) 
v_lshlrev_b32 v6, 16, v6 // (op 3308) 
v_lshlrev_b32 v7, 16, v7 // (op 3308) 
v_mul_f32 v14, v18, v6 // (op 3308) 
v_mul_f32 v15, v18, v7 // (op 3308) 


v_add_f32 v4, v12, v8 // (op 3297) 
v_add_f32 v5, v13, v9 // (op 3297) 
v_add_f32 v6, v14, v10 // (op 3297) 
v_add_f32 v7, v15, v11 // (op 3297) 


v_lshrrev_b32 v8, 16, v4 // (op 3298) 
v_lshrrev_b32 v9, 16, v5 // (op 3298) 
v_lshrrev_b32 v10, 16, v6 // (op 3298) 
v_lshrrev_b32 v11, 16, v7 // (op 3298) 


v_pack_b32_f16 v6, v8, v9 // (op 3300) 
v_pack_b32_f16 v7, v10, v11 // (op 3300) 

v_add_u32 v0, 64, v21 // (op 3300) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3300) Store value





s_lshl_b32 s0, s2, 6 // (op 3314) 
s_add_u32 s0, s0, 64 // (op 3314) 
v_add_u32 v0, v20, s0 // (op 3314) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3314) Load value

v_accvgpr_read v12, a20 // (op 3326) 
v_accvgpr_read v13, a21 // (op 3326) 
v_accvgpr_read v14, a22 // (op 3326) 
v_accvgpr_read v15, a23 // (op 3326) 
v_mul_f32 v8, v19, v12 // (op 3326) 
v_mul_f32 v9, v19, v13 // (op 3326) 
v_mul_f32 v10, v19, v14 // (op 3326) 
v_mul_f32 v11, v19, v15 // (op 3326) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 3311) 
v_lshrrev_b32 v7, 16, v16 // (op 3311) 
v_lshlrev_b32 v6, 16, v6 // (op 3311) 
v_lshlrev_b32 v7, 16, v7 // (op 3311) 
v_mul_f32 v12, v18, v6 // (op 3311) 
v_mul_f32 v13, v18, v7 // (op 3311) 
v_and_b32 v6, 65535, v17 // (op 3311) 
v_lshrrev_b32 v7, 16, v17 // (op 3311) 
v_lshlrev_b32 v6, 16, v6 // (op 3311) 
v_lshlrev_b32 v7, 16, v7 // (op 3311) 
v_mul_f32 v14, v18, v6 // (op 3311) 
v_mul_f32 v15, v18, v7 // (op 3311) 


v_add_f32 v4, v12, v8 // (op 3312) 
v_add_f32 v5, v13, v9 // (op 3312) 
v_add_f32 v6, v14, v10 // (op 3312) 
v_add_f32 v7, v15, v11 // (op 3312) 


v_lshrrev_b32 v8, 16, v4 // (op 3313) 
v_lshrrev_b32 v9, 16, v5 // (op 3313) 
v_lshrrev_b32 v10, 16, v6 // (op 3313) 
v_lshrrev_b32 v11, 16, v7 // (op 3313) 


v_pack_b32_f16 v6, v8, v9 // (op 3302) 
v_pack_b32_f16 v7, v10, v11 // (op 3302) 

s_lshl_b32 s0, s4, 6 // (op 3302) 
s_add_u32 s0, s0, 64 // (op 3302) 
v_add_u32 v0, v21, s0 // (op 3302) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3302) Store value





s_lshl_b32 s0, s2, 7 // (op 3319) 
s_add_u32 s0, s0, 64 // (op 3319) 
v_add_u32 v0, v20, s0 // (op 3319) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3319) Load value

v_accvgpr_read v12, a24 // (op 3328) 
v_accvgpr_read v13, a25 // (op 3328) 
v_accvgpr_read v14, a26 // (op 3328) 
v_accvgpr_read v15, a27 // (op 3328) 
v_mul_f32 v8, v19, v12 // (op 3328) 
v_mul_f32 v9, v19, v13 // (op 3328) 
v_mul_f32 v10, v19, v14 // (op 3328) 
v_mul_f32 v11, v19, v15 // (op 3328) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 3316) 
v_lshrrev_b32 v7, 16, v16 // (op 3316) 
v_lshlrev_b32 v6, 16, v6 // (op 3316) 
v_lshlrev_b32 v7, 16, v7 // (op 3316) 
v_mul_f32 v12, v18, v6 // (op 3316) 
v_mul_f32 v13, v18, v7 // (op 3316) 
v_and_b32 v6, 65535, v17 // (op 3316) 
v_lshrrev_b32 v7, 16, v17 // (op 3316) 
v_lshlrev_b32 v6, 16, v6 // (op 3316) 
v_lshlrev_b32 v7, 16, v7 // (op 3316) 
v_mul_f32 v14, v18, v6 // (op 3316) 
v_mul_f32 v15, v18, v7 // (op 3316) 


v_add_f32 v4, v12, v8 // (op 3317) 
v_add_f32 v5, v13, v9 // (op 3317) 
v_add_f32 v6, v14, v10 // (op 3317) 
v_add_f32 v7, v15, v11 // (op 3317) 


v_lshrrev_b32 v8, 16, v4 // (op 3318) 
v_lshrrev_b32 v9, 16, v5 // (op 3318) 
v_lshrrev_b32 v10, 16, v6 // (op 3318) 
v_lshrrev_b32 v11, 16, v7 // (op 3318) 


v_pack_b32_f16 v6, v8, v9 // (op 3304) 
v_pack_b32_f16 v7, v10, v11 // (op 3304) 

s_lshl_b32 s0, s4, 7 // (op 3304) 
s_add_u32 s0, s0, 64 // (op 3304) 
v_add_u32 v0, v21, s0 // (op 3304) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3304) Store value





s_lshl_b32 s0, s2, 6 // (op 3324) 
s_mul_i32 s1, 3, s0 // (op 3324) 
s_add_u32 s1, s1, 64 // (op 3324) 

v_add_u32 v0, v20, s1 // (op 3324) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3324) Load value

v_accvgpr_read v12, a28 // (op 3330) 
v_accvgpr_read v13, a29 // (op 3330) 
v_accvgpr_read v14, a30 // (op 3330) 
v_accvgpr_read v15, a31 // (op 3330) 
v_mul_f32 v8, v19, v12 // (op 3330) 
v_mul_f32 v9, v19, v13 // (op 3330) 
v_mul_f32 v10, v19, v14 // (op 3330) 
v_mul_f32 v11, v19, v15 // (op 3330) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 3321) 
v_lshrrev_b32 v7, 16, v16 // (op 3321) 
v_lshlrev_b32 v6, 16, v6 // (op 3321) 
v_lshlrev_b32 v7, 16, v7 // (op 3321) 
v_mul_f32 v12, v18, v6 // (op 3321) 
v_mul_f32 v13, v18, v7 // (op 3321) 
v_and_b32 v6, 65535, v17 // (op 3321) 
v_lshrrev_b32 v7, 16, v17 // (op 3321) 
v_lshlrev_b32 v6, 16, v6 // (op 3321) 
v_lshlrev_b32 v7, 16, v7 // (op 3321) 
v_mul_f32 v14, v18, v6 // (op 3321) 
v_mul_f32 v15, v18, v7 // (op 3321) 


v_add_f32 v4, v12, v8 // (op 3322) 
v_add_f32 v5, v13, v9 // (op 3322) 
v_add_f32 v6, v14, v10 // (op 3322) 
v_add_f32 v7, v15, v11 // (op 3322) 


v_lshrrev_b32 v8, 16, v4 // (op 3323) 
v_lshrrev_b32 v9, 16, v5 // (op 3323) 
v_lshrrev_b32 v10, 16, v6 // (op 3323) 
v_lshrrev_b32 v11, 16, v7 // (op 3323) 


v_pack_b32_f16 v6, v8, v9 // (op 3306) 
v_pack_b32_f16 v7, v10, v11 // (op 3306) 

s_lshl_b32 s0, s4, 6 // (op 3306) 
s_mul_i32 s1, 3, s0 // (op 3306) 
s_add_u32 s1, s1, 64 // (op 3306) 

v_add_u32 v0, v21, s1 // (op 3306) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3306) Store value





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





s_lshl_b32 s0, s2, 6 // (op 3217) 
v_add_u32 v0, v20, s0 // (op 3217) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3217) Load value

v_accvgpr_read v12, a4 // (op 3213) 
v_accvgpr_read v13, a5 // (op 3213) 
v_accvgpr_read v14, a6 // (op 3213) 
v_accvgpr_read v15, a7 // (op 3213) 
v_mul_f32 v8, v19, v12 // (op 3213) 
v_mul_f32 v9, v19, v13 // (op 3213) 
v_mul_f32 v10, v19, v14 // (op 3213) 
v_mul_f32 v11, v19, v15 // (op 3213) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 3218) 
v_lshrrev_b32 v7, 16, v16 // (op 3218) 
v_lshlrev_b32 v6, 16, v6 // (op 3218) 
v_lshlrev_b32 v7, 16, v7 // (op 3218) 
v_mul_f32 v12, v18, v6 // (op 3218) 
v_mul_f32 v13, v18, v7 // (op 3218) 
v_and_b32 v6, 65535, v17 // (op 3218) 
v_lshrrev_b32 v7, 16, v17 // (op 3218) 
v_lshlrev_b32 v6, 16, v6 // (op 3218) 
v_lshlrev_b32 v7, 16, v7 // (op 3218) 
v_mul_f32 v14, v18, v6 // (op 3218) 
v_mul_f32 v15, v18, v7 // (op 3218) 


v_add_f32 v4, v12, v8 // (op 3214) 
v_add_f32 v5, v13, v9 // (op 3214) 
v_add_f32 v6, v14, v10 // (op 3214) 
v_add_f32 v7, v15, v11 // (op 3214) 


v_lshrrev_b32 v8, 16, v4 // (op 3215) 
v_lshrrev_b32 v9, 16, v5 // (op 3215) 
v_lshrrev_b32 v10, 16, v6 // (op 3215) 
v_lshrrev_b32 v11, 16, v7 // (op 3215) 


v_pack_b32_f16 v6, v8, v9 // (op 3216) 
v_pack_b32_f16 v7, v10, v11 // (op 3216) 

s_lshl_b32 s0, s4, 6 // (op 3216) 
v_add_u32 v0, v21, s0 // (op 3216) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3216) Store value





s_lshl_b32 s0, s2, 7 // (op 3231) 
v_add_u32 v0, v20, s0 // (op 3231) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3231) Load value

v_accvgpr_read v12, a8 // (op 3227) 
v_accvgpr_read v13, a9 // (op 3227) 
v_accvgpr_read v14, a10 // (op 3227) 
v_accvgpr_read v15, a11 // (op 3227) 
v_mul_f32 v8, v19, v12 // (op 3227) 
v_mul_f32 v9, v19, v13 // (op 3227) 
v_mul_f32 v10, v19, v14 // (op 3227) 
v_mul_f32 v11, v19, v15 // (op 3227) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 3232) 
v_lshrrev_b32 v7, 16, v16 // (op 3232) 
v_lshlrev_b32 v6, 16, v6 // (op 3232) 
v_lshlrev_b32 v7, 16, v7 // (op 3232) 
v_mul_f32 v12, v18, v6 // (op 3232) 
v_mul_f32 v13, v18, v7 // (op 3232) 
v_and_b32 v6, 65535, v17 // (op 3232) 
v_lshrrev_b32 v7, 16, v17 // (op 3232) 
v_lshlrev_b32 v6, 16, v6 // (op 3232) 
v_lshlrev_b32 v7, 16, v7 // (op 3232) 
v_mul_f32 v14, v18, v6 // (op 3232) 
v_mul_f32 v15, v18, v7 // (op 3232) 


v_add_f32 v4, v12, v8 // (op 3228) 
v_add_f32 v5, v13, v9 // (op 3228) 
v_add_f32 v6, v14, v10 // (op 3228) 
v_add_f32 v7, v15, v11 // (op 3228) 


v_lshrrev_b32 v8, 16, v4 // (op 3229) 
v_lshrrev_b32 v9, 16, v5 // (op 3229) 
v_lshrrev_b32 v10, 16, v6 // (op 3229) 
v_lshrrev_b32 v11, 16, v7 // (op 3229) 


v_pack_b32_f16 v6, v8, v9 // (op 3230) 
v_pack_b32_f16 v7, v10, v11 // (op 3230) 

s_lshl_b32 s0, s4, 7 // (op 3230) 
v_add_u32 v0, v21, s0 // (op 3230) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3230) Store value





s_lshl_b32 s0, s2, 6 // (op 3245) 
s_mul_i32 s1, 3, s0 // (op 3245) 

v_add_u32 v0, v20, s1 // (op 3245) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3245) Load value

v_accvgpr_read v12, a12 // (op 3241) 
v_accvgpr_read v13, a13 // (op 3241) 
v_accvgpr_read v14, a14 // (op 3241) 
v_accvgpr_read v15, a15 // (op 3241) 
v_mul_f32 v8, v19, v12 // (op 3241) 
v_mul_f32 v9, v19, v13 // (op 3241) 
v_mul_f32 v10, v19, v14 // (op 3241) 
v_mul_f32 v11, v19, v15 // (op 3241) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 3246) 
v_lshrrev_b32 v7, 16, v16 // (op 3246) 
v_lshlrev_b32 v6, 16, v6 // (op 3246) 
v_lshlrev_b32 v7, 16, v7 // (op 3246) 
v_mul_f32 v12, v18, v6 // (op 3246) 
v_mul_f32 v13, v18, v7 // (op 3246) 
v_and_b32 v6, 65535, v17 // (op 3246) 
v_lshrrev_b32 v7, 16, v17 // (op 3246) 
v_lshlrev_b32 v6, 16, v6 // (op 3246) 
v_lshlrev_b32 v7, 16, v7 // (op 3246) 
v_mul_f32 v14, v18, v6 // (op 3246) 
v_mul_f32 v15, v18, v7 // (op 3246) 


v_add_f32 v4, v12, v8 // (op 3242) 
v_add_f32 v5, v13, v9 // (op 3242) 
v_add_f32 v6, v14, v10 // (op 3242) 
v_add_f32 v7, v15, v11 // (op 3242) 


v_lshrrev_b32 v8, 16, v4 // (op 3243) 
v_lshrrev_b32 v9, 16, v5 // (op 3243) 
v_lshrrev_b32 v10, 16, v6 // (op 3243) 
v_lshrrev_b32 v11, 16, v7 // (op 3243) 


v_pack_b32_f16 v6, v8, v9 // (op 3244) 
v_pack_b32_f16 v7, v10, v11 // (op 3244) 

s_lshl_b32 s0, s4, 6 // (op 3244) 
s_mul_i32 s1, 3, s0 // (op 3244) 

v_add_u32 v0, v21, s1 // (op 3244) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3244) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
.LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_end:
.size RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC, .LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_end-RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
  .amdhsa_next_free_vgpr 168
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 34816 // lds bytes
  .amdhsa_accum_offset 136
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
    macrotile: [64, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 4
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
    .symbol: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 34816
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 136
    .agpr_count: 32
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
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
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
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .value_kind: by_value
...

.end_amdgpu_metadata

