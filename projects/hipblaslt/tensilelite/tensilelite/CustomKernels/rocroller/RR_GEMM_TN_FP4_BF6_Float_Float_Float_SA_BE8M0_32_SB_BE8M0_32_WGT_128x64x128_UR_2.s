// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2,@function
RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 4580) 
v_and_b32 v3, 15, v1 // (op 4580) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4580) 


v_and_b32 v0, 63, v1 // (op 4580) 
v_lshrrev_b32 v3, 4, v0 // (op 4580) 

v_lshlrev_b32 v0, 5, v3 // (op 4580) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 4580) 


v_lshlrev_b32 v0, 2, v3 // (op 4580) 

v_lshrrev_b32 v3, 3, v0 // (op 4580) 

v_lshrrev_b32 v0, 6, v1 // (op 4638) 
v_and_b32 v4, 1, v0 // (op 4638) 

v_and_b32 v0, 15, v1 // (op 4638) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4638) 


v_and_b32 v0, 63, v1 // (op 4638) 
v_lshrrev_b32 v4, 4, v0 // (op 4638) 

v_lshlrev_b32 v0, 5, v4 // (op 4638) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 4638) 


v_mul_lo_u32 v0, v4, 6 // (op 4638) 

v_lshrrev_b32 v4, 3, v0 // (op 4638) 

v_lshrrev_b32 v0, 7, v1 // (op 4696) 
v_and_b32 v5, 15, v1 // (op 4696) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4696) 


v_and_b32 v0, 63, v1 // (op 4696) 
v_lshrrev_b32 v5, 4, v0 // (op 4696) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 4696) 


v_lshrrev_b32 v5, 6, v1 // (op 4754) 
v_and_b32 v6, 1, v5 // (op 4754) 

v_and_b32 v5, 15, v1 // (op 4754) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4754) 


v_and_b32 v5, 63, v1 // (op 4754) 
v_lshrrev_b32 v6, 4, v5 // (op 4754) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 4754) 


v_lshrrev_b32 v6, 7, v1 // (op 4806) 
v_and_b32 v7, 15, v1 // (op 4806) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 4806) 


v_and_b32 v6, 63, v1 // (op 4806) 
v_lshrrev_b32 v7, 4, v6 // (op 4806) 

v_lshlrev_b32 v6, 5, v7 // (op 4806) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 4806) 


v_lshlrev_b32 v6, 2, v7 // (op 4806) 

v_lshrrev_b32 v7, 3, v6 // (op 4806) 

v_lshrrev_b32 v6, 6, v1 // (op 4835) 
v_and_b32 v8, 1, v6 // (op 4835) 

v_and_b32 v6, 15, v1 // (op 4835) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 4835) 


v_and_b32 v6, 63, v1 // (op 4835) 
v_lshrrev_b32 v8, 4, v6 // (op 4835) 

v_lshlrev_b32 v6, 5, v8 // (op 4835) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 4835) 


v_mul_lo_u32 v6, v8, 6 // (op 4835) 

v_lshrrev_b32 v8, 3, v6 // (op 4835) 

v_lshrrev_b32 v6, 7, v1 // (op 4864) 
v_and_b32 v9, 15, v1 // (op 4864) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 4864) 


v_and_b32 v6, 63, v1 // (op 4864) 
v_lshrrev_b32 v9, 4, v6 // (op 4864) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 4864) 


v_lshrrev_b32 v9, 6, v1 // (op 4893) 
v_and_b32 v10, 1, v9 // (op 4893) 

v_and_b32 v9, 15, v1 // (op 4893) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4893) 


v_and_b32 v9, 63, v1 // (op 4893) 
v_lshrrev_b32 v10, 4, v9 // (op 4893) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 4893) 


v_lshrrev_b32 v10, 7, v1 // (op 4518) 
v_and_b32 v11, 63, v1 // (op 4518) 
v_lshrrev_b32 v12, 4, v11 // (op 4518) 
v_lshlrev_b32 v13, 2, v12 // (op 4518) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4518) 


v_lshl_add_u32 v10, s16, 7, v12 // (op 4518) 

v_lshrrev_b32 v12, 2, v11 // (op 4518) 

v_and_b32 v11, 3, v12 // (op 4518) 

v_and_b32 v12, 3, v1 // (op 4518) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 4518) 


v_lshrrev_b32 v11, 6, v1 // (op 4518) 
v_and_b32 v12, 1, v11 // (op 4518) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 4518) 


v_lshl_add_u32 v12, s17, 6, v11 // (op 4518) 

v_mul_lo_u32 v11, v12, s2 // (op 4518) 

v_add_lshl_u32 v12, v10, v11, 2 // (op 4518) 


v_lshrrev_b32 v10, 7, v1 // (op 4775) 
v_and_b32 v11, 63, v1 // (op 4775) 
v_lshrrev_b32 v13, 4, v11 // (op 4775) 
s_mul_i32 s1, s2, s9 // (op 4520) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 4520) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 4520) least significant
s_mul_hi_u32 s19, s2, s8 // (op 4520) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4520) most significant: sum
s_add_u32 s1, s1, s19 // (op 4520) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4520) 

v_lshlrev_b32 v14, 2, v13 // (op 4775) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 4775) 


v_lshl_add_u32 v10, s16, 7, v13 // (op 4775) 

v_lshrrev_b32 v13, 2, v11 // (op 4775) 

v_and_b32 v11, 3, v13 // (op 4775) 

v_and_b32 v13, 3, v1 // (op 4775) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 4775) 


v_lshrrev_b32 v11, 6, v1 // (op 4775) 
v_and_b32 v13, 1, v11 // (op 4775) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 4775) 


v_lshl_add_u32 v13, s17, 6, v11 // (op 4775) 

v_mul_lo_u32 v11, v13, s4 // (op 4775) 

v_add_lshl_u32 v13, v10, v11, 2 // (op 4775) 


s_mov_b32 s40, s6 // (op 4520) reg expression
s_mov_b32 s41, s7 // (op 4520) reg expression
s_mov_b32 s42, s18 // (op 4520) reg expression
s_mov_b32 s43, 131072 // (op 4520) call()

s_mul_i32 s1, s4, s9 // (op 4777) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 4777) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 4777) least significant
s_mul_hi_u32 s7, s4, s8 // (op 4777) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4777) most significant: sum
s_add_u32 s1, s1, s7 // (op 4777) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4777) 

s_mov_b32 s44, s10 // (op 4777) reg expression
s_mov_b32 s45, s11 // (op 4777) reg expression
s_mov_b32 s46, s6 // (op 4777) reg expression
s_mov_b32 s47, 131072 // (op 4777) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 336) call()
v_accvgpr_write a5, 0.00000 // (op 336) call()
v_accvgpr_write a6, 0.00000 // (op 336) call()
v_accvgpr_write a7, 0.00000 // (op 336) call()
v_accvgpr_write a8, 0.00000 // (op 556) call()
v_accvgpr_write a9, 0.00000 // (op 556) call()
v_accvgpr_write a10, 0.00000 // (op 556) call()
v_accvgpr_write a11, 0.00000 // (op 556) call()
v_accvgpr_write a12, 0.00000 // (op 711) call()
v_accvgpr_write a13, 0.00000 // (op 711) call()
v_accvgpr_write a14, 0.00000 // (op 711) call()
v_accvgpr_write a15, 0.00000 // (op 711) call()
v_accvgpr_write a16, 0.00000 // (op 942) call()
v_accvgpr_write a17, 0.00000 // (op 942) call()
v_accvgpr_write a18, 0.00000 // (op 942) call()
v_accvgpr_write a19, 0.00000 // (op 942) call()
v_accvgpr_write a20, 0.00000 // (op 1097) call()
v_accvgpr_write a21, 0.00000 // (op 1097) call()
v_accvgpr_write a22, 0.00000 // (op 1097) call()
v_accvgpr_write a23, 0.00000 // (op 1097) call()
v_accvgpr_write a24, 0.00000 // (op 1328) call()
v_accvgpr_write a25, 0.00000 // (op 1328) call()
v_accvgpr_write a26, 0.00000 // (op 1328) call()
v_accvgpr_write a27, 0.00000 // (op 1328) call()
v_accvgpr_write a28, 0.00000 // (op 1483) call()
v_accvgpr_write a29, 0.00000 // (op 1483) call()
v_accvgpr_write a30, 0.00000 // (op 1483) call()
v_accvgpr_write a31, 0.00000 // (op 1483) call()
v_lshrrev_b32 v14, 2, v1 // (op 4280) 
v_lshlrev_b32 v15, 1, v14 // (op 4280) 

v_lshl_add_u32 v14, s16, 7, v15 // (op 4280) 

v_and_b32 v15, -4, v1 // (op 4280) 
v_sub_u32 v16, v1, v15 // (op 4280) 

v_lshlrev_b32 v15, 5, v16 // (op 4280) 

v_mov_b32 v17, 0 // (op 4280) convert
v_mov_b32 v16, v15 // (op 4280) convert

v_mul_lo_u32 v19, v14, s13 // (op 4280) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 4280) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 4280) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 4280) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 4280) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 4280) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 4280) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 4280) 

v_mov_b32 v14, v16 // (op 4280) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 4282) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4282) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4282) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4282) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4282) most significant: sum
s_add_u32 s1, s1, s7 // (op 4282) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4282) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 4282) 

s_mov_b32 s48, s20 // (op 4282) reg expression
s_mov_b32 s49, s21 // (op 4282) reg expression
s_mov_b32 s50, s0 // (op 4282) reg expression
s_mov_b32 s51, 131072 // (op 4282) call()

v_lshrrev_b32 v15, 2, v1 // (op 4299) 
v_lshlrev_b32 v16, 1, v15 // (op 4299) 

v_lshl_add_u32 v15, s16, 7, v16 // (op 4299) 

v_and_b32 v16, -4, v1 // (op 4299) 
v_sub_u32 v17, v1, v16 // (op 4299) 

v_lshlrev_b32 v16, 5, v17 // (op 4299) 

v_mov_b32 v19, 0 // (op 4299) convert
v_mov_b32 v18, v16 // (op 4299) convert

v_mul_lo_u32 v17, v15, s13 // (op 4299) most significant: low of low * high
v_mul_hi_u32 v20, v15, s12 // (op 4299) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 4299) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 4299) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 4299) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 4299) most significant half



v_lshlrev_b64 v[18:19], 2, v[16:17] // (op 4299) 

v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 4299) 

v_mov_b32 v15, v16 // (op 4299) convert

s_mul_i32 s1, s12, s23 // (op 4301) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4301) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4301) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4301) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4301) most significant: sum
s_add_u32 s1, s1, s7 // (op 4301) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4301) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 4301) 

s_mov_b32 s48, s20 // (op 4301) reg expression
s_mov_b32 s49, s21 // (op 4301) reg expression
s_mov_b32 s50, s0 // (op 4301) reg expression
s_mov_b32 s51, 131072 // (op 4301) call()

v_and_b32 v16, -4, v1 // (op 4344) 
v_sub_u32 v17, v1, v16 // (op 4344) 

v_lshlrev_b32 v16, 5, v17 // (op 4344) 

v_mov_b32 v19, 0 // (op 4344) convert
v_mov_b32 v18, v16 // (op 4344) convert

v_lshrrev_b32 v16, 2, v1 // (op 4344) 
v_lshl_add_u32 v17, s17, 6, v16 // (op 4344) 

v_mul_lo_u32 v21, v17, s25 // (op 4344) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 4344) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 4344) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 4344) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 4344) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 4344) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 4344) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 4344) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 4344) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 4344) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 4344) 

v_mov_b32 v18, v16 // (op 4344) convert

s_mul_i32 s1, s9, 6 // (op 4346) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 4346) least significant
s_mul_hi_u32 s6, s8, 6 // (op 4346) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4346) most significant: sum

s_mul_i32 s7, s24, s1 // (op 4346) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 4346) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 4346) least significant
s_mul_hi_u32 s11, s24, s0 // (op 4346) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 4346) most significant: sum
s_add_u32 s7, s7, s11 // (op 4346) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 4346) 

s_mov_b32 s52, s26 // (op 4346) reg expression
s_mov_b32 s53, s27 // (op 4346) reg expression
s_mov_b32 s54, s0 // (op 4346) reg expression
s_mov_b32 s55, 131072 // (op 4346) call()

v_and_b32 v16, -4, v1 // (op 4363) 
v_sub_u32 v17, v1, v16 // (op 4363) 

v_lshlrev_b32 v16, 5, v17 // (op 4363) 

v_mov_b32 v21, 0 // (op 4363) convert
v_mov_b32 v20, v16 // (op 4363) convert

v_lshrrev_b32 v19, 2, v1 // (op 4363) 
v_lshl_add_u32 v16, s17, 6, v19 // (op 4363) 

v_mul_lo_u32 v23, v16, s25 // (op 4363) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 4363) most significant: high of low * low
v_mul_lo_u32 v22, v16, s24 // (op 4363) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 4363) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 4363) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 4363) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 4363) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 4363) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 4363) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 4363) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 4363) 

v_mov_b32 v19, v16 // (op 4363) convert

s_mul_i32 s1, s9, 6 // (op 4365) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 4365) least significant
s_mul_hi_u32 s6, s8, 6 // (op 4365) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4365) most significant: sum

s_mul_i32 s7, s24, s1 // (op 4365) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 4365) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 4365) least significant
s_mul_hi_u32 s11, s24, s0 // (op 4365) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 4365) most significant: sum
s_add_u32 s7, s7, s11 // (op 4365) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 4365) 

s_mov_b32 s52, s26 // (op 4365) reg expression
s_mov_b32 s53, s27 // (op 4365) reg expression
s_mov_b32 s54, s0 // (op 4365) reg expression
s_mov_b32 s55, 131072 // (op 4365) call()

v_lshrrev_b32 v16, 2, v1 // (op 4408) 
v_lshlrev_b32 v17, 1, v16 // (op 4408) 

v_lshl_add_u32 v16, s16, 7, v17 // (op 4408) 

v_and_b32 v17, -4, v1 // (op 4408) 
v_sub_u32 v20, v1, v17 // (op 4408) 

v_mul_lo_u32 v17, v16, s28 // (op 4408) 
v_add_u32 v17, v17, v20 // (op 4408) 


s_mul_i32 s1, s28, s23 // (op 4410) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4410) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4410) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4410) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4410) most significant: sum
s_add_u32 s1, s1, s7 // (op 4410) most significant: sum


s_mov_b32 s56, s30 // (op 4410) reg expression
s_mov_b32 s57, s31 // (op 4410) reg expression
s_mov_b32 s58, s0 // (op 4410) reg expression
s_mov_b32 s59, 131072 // (op 4410) call()

v_lshrrev_b32 v16, 2, v1 // (op 4427) 
v_lshlrev_b32 v20, 1, v16 // (op 4427) 

v_lshl_add_u32 v16, s16, 7, v20 // (op 4427) 

v_and_b32 v20, -4, v1 // (op 4427) 
v_sub_u32 v21, v1, v20 // (op 4427) 

v_mul_lo_u32 v20, v16, s28 // (op 4427) 
v_add_u32 v20, v20, v21 // (op 4427) 


s_mul_i32 s1, s28, s23 // (op 4429) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4429) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4429) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4429) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4429) most significant: sum
s_add_u32 s1, s1, s7 // (op 4429) most significant: sum


s_mov_b32 s56, s30 // (op 4429) reg expression
s_mov_b32 s57, s31 // (op 4429) reg expression
s_mov_b32 s58, s0 // (op 4429) reg expression
s_mov_b32 s59, 131072 // (op 4429) call()

v_and_b32 v16, -4, v1 // (op 4472) 
v_sub_u32 v21, v1, v16 // (op 4472) 

v_lshrrev_b32 v16, 2, v1 // (op 4472) 
v_lshl_add_u32 v22, s17, 6, v16 // (op 4472) 

v_mul_lo_u32 v16, v22, s32 // (op 4472) 

v_add_u32 v22, v21, v16 // (op 4472) 


s_mul_i32 s1, s32, s9 // (op 4474) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4474) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4474) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4474) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4474) most significant: sum
s_add_u32 s1, s1, s7 // (op 4474) most significant: sum


s_mov_b32 s60, s34 // (op 4474) reg expression
s_mov_b32 s61, s35 // (op 4474) reg expression
s_mov_b32 s62, s0 // (op 4474) reg expression
s_mov_b32 s63, 131072 // (op 4474) call()

v_and_b32 v16, -4, v1 // (op 4491) 
v_sub_u32 v21, v1, v16 // (op 4491) 

v_lshrrev_b32 v16, 2, v1 // (op 4491) 
v_lshl_add_u32 v23, s17, 6, v16 // (op 4491) 

v_mul_lo_u32 v16, v23, s32 // (op 4491) 

v_add_u32 v23, v21, v16 // (op 4491) 


s_mul_i32 s1, s32, s9 // (op 4493) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4493) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4493) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4493) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4493) most significant: sum
s_add_u32 s1, s1, s7 // (op 4493) most significant: sum


s_mov_b32 s60, s34 // (op 4493) reg expression
s_mov_b32 s61, s35 // (op 4493) reg expression
s_mov_b32 s62, s0 // (op 4493) reg expression
s_mov_b32 s63, 131072 // (op 4493) call()




buffer_load_dwordx4 v[24:27], v18, s[52:55], 0 offen // (op 4024) Load value
buffer_load_dwordx2 v[28:29], v18, s[52:55], 0 offen offset:16 // (op 4024) Load value



buffer_load_ubyte v16, v22, s[60:63], 0 offen offset: 0 // (op 4027) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 4030) 
v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 4030) 


v_mov_b32 v21, v14 // (op 4030) 
buffer_load_dwordx4 v[32:35], v21, s[48:51], 0 offen // (op 4030) Load value
v_add_u32 v21, v21, v30 // (op 4030) 
buffer_load_dwordx4 v[36:39], v21, s[48:51], 0 offen // (op 4030) Load value



v_mov_b64 v[30:31], s[28:29] // (op 4033) convert

v_mov_b32 v21, v17 // (op 4033) 
buffer_load_ubyte v40, v21, s[56:59], 0 offen offset: 0 // (op 4033) Load value
v_add_u32 v21, v21, v30 // (op 4033) 
buffer_load_ubyte v41, v21, s[56:59], 0 offen offset: 0 // (op 4033) Load value





buffer_load_dwordx4 v[42:45], v18, s[52:55], 0 offen offset:96 // (op 4036) Load value
buffer_load_dwordx2 v[46:47], v18, s[52:55], 0 offen offset:112 // (op 4036) Load value




buffer_load_ubyte v18, v22, s[60:63], 0 offen offset:4 // (op 4039) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 4042) 
v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 4042) 


v_mov_b32 v21, v14 // (op 4042) 
buffer_load_dwordx4 v[48:51], v21, s[48:51], 0 offen offset:64 // (op 4042) Load value
v_add_u32 v21, v21, v30 // (op 4042) 
buffer_load_dwordx4 v[52:55], v21, s[48:51], 0 offen offset:64 // (op 4042) Load value




v_mov_b64 v[30:31], s[28:29] // (op 4045) convert

v_mov_b32 v14, v17 // (op 4045) 
buffer_load_ubyte v56, v14, s[56:59], 0 offen offset:4 // (op 4045) Load value
v_add_u32 v14, v14, v30 // (op 4045) 
buffer_load_ubyte v57, v14, s[56:59], 0 offen offset:4 // (op 4045) Load value



v_and_b32 v14, -4, v1 // (op 4621) 
v_sub_u32 v17, v1, v14 // (op 4621) 

v_lshlrev_b32 v14, 5, v17 // (op 4621) 

v_lshrrev_b32 v17, 2, v1 // (op 4621) 
v_lshl_add_u32 v21, v17, 7, v14 // (op 4621) 


v_mul_lo_u32 v14, v21, 6 // (op 4621) 

v_lshrrev_b32 v17, 3, v14 // (op 4621) 




s_waitcnt vmcnt(13)
ds_write_b128 v17, v[24:27] // (op 4047) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v17, v[28:29] offset:16 // (op 4047) Store local data 

v_and_b32 v14, -4, v1 // (op 4737) 
v_sub_u32 v17, v1, v14 // (op 4737) 

v_and_b32 v14, 4294967292, v1 // (op 4737) 
v_add_u32 v21, v14, v17 // (op 4737) 





s_waitcnt vmcnt(11)
ds_write_b8 v21, v16 offset:6144 // (op 4048) Store local data 

v_and_b32 v14, -4, v1 // (op 4563) 
v_sub_u32 v17, v1, v14 // (op 4563) 

v_lshlrev_b32 v14, 5, v17 // (op 4563) 

v_lshrrev_b32 v17, 2, v1 // (op 4563) 
v_lshl_add_u32 v21, v17, 8, v14 // (op 4563) 


v_lshlrev_b32 v14, 2, v21 // (op 4563) 

v_lshrrev_b32 v17, 3, v14 // (op 4563) 




s_waitcnt vmcnt(10)
ds_write_b128 v17, v[32:35] offset:6400 // (op 4049) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v17, v[36:39] offset:6464 // (op 4049) Store local data 

v_and_b32 v14, -4, v1 // (op 4679) 
v_sub_u32 v17, v1, v14 // (op 4679) 

v_lshrrev_b32 v14, 2, v1 // (op 4679) 
v_lshl_add_u32 v21, v14, 3, v17 // (op 4679) 





s_waitcnt vmcnt(8)
ds_write_b8 v21, v40 offset:14592 // (op 4050) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v21, v41 offset:14596 // (op 4050) Store local data 

v_and_b32 v14, -4, v1 // (op 4534) 
v_sub_u32 v17, v1, v14 // (op 4534) 

v_lshlrev_b32 v14, 5, v17 // (op 4534) 

v_lshrrev_b32 v17, 2, v1 // (op 4534) 
v_lshl_add_u32 v21, v17, 8, v14 // (op 4534) 


v_lshlrev_b32 v14, 2, v21 // (op 4534) 

v_lshrrev_b32 v17, 3, v14 // (op 4534) 

v_and_b32 v14, -4, v1 // (op 4592) 
v_sub_u32 v21, v1, v14 // (op 4592) 

v_lshlrev_b32 v14, 5, v21 // (op 4592) 

v_lshrrev_b32 v21, 2, v1 // (op 4592) 
v_lshl_add_u32 v22, v21, 7, v14 // (op 4592) 


v_mul_lo_u32 v14, v22, 6 // (op 4592) 

v_lshrrev_b32 v21, 3, v14 // (op 4592) 

v_and_b32 v14, -4, v1 // (op 4650) 
v_sub_u32 v22, v1, v14 // (op 4650) 

v_lshrrev_b32 v14, 2, v1 // (op 4650) 
v_lshl_add_u32 v30, v14, 3, v22 // (op 4650) 


v_and_b32 v14, -4, v1 // (op 4708) 
v_sub_u32 v22, v1, v14 // (op 4708) 

v_and_b32 v14, 4294967292, v1 // (op 4708) 
v_add_u32 v31, v14, v22 // (op 4708) 


v_and_b32 v14, -4, v1 // (op 4789) 
v_sub_u32 v22, v1, v14 // (op 4789) 

v_lshlrev_b32 v14, 5, v22 // (op 4789) 

v_lshrrev_b32 v22, 2, v1 // (op 4789) 
v_lshl_add_u32 v58, v22, 8, v14 // (op 4789) 


v_lshlrev_b32 v14, 2, v58 // (op 4789) 

v_lshrrev_b32 v22, 3, v14 // (op 4789) 

v_and_b32 v14, -4, v1 // (op 4818) 
v_sub_u32 v58, v1, v14 // (op 4818) 

v_lshlrev_b32 v14, 5, v58 // (op 4818) 

v_lshrrev_b32 v58, 2, v1 // (op 4818) 
v_lshl_add_u32 v59, v58, 7, v14 // (op 4818) 


v_mul_lo_u32 v14, v59, 6 // (op 4818) 

v_lshrrev_b32 v58, 3, v14 // (op 4818) 

v_and_b32 v14, -4, v1 // (op 4847) 
v_sub_u32 v59, v1, v14 // (op 4847) 

v_lshrrev_b32 v14, 2, v1 // (op 4847) 
v_lshl_add_u32 v60, v14, 3, v59 // (op 4847) 


v_and_b32 v14, -4, v1 // (op 4876) 
v_sub_u32 v59, v1, v14 // (op 4876) 

v_and_b32 v14, 4294967292, v1 // (op 4876) 
v_add_u32 v61, v14, v59 // (op 4876) 


s_mov_b32 s0, 0 // (op 1361) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1360) 
s_mov_b32 s1, s6 // (op 1360) convert

s_and_b32 s6, s1, 4294967294 // (op 1360) 

s_cmp_lt_u32 s0, s6 // (op 1360) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1360 // (op 1360) Condition: Top (jump to RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1360 if false)
RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1360:


s_add_u32 s0, s0, 2 // (op 1362) 
v_mov_b32 v14, v19 // (op 10) 

buffer_load_dwordx4 v[24:27], v14, s[52:55], 0 offen offset:192 // (op 10) Load value
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4956) 
buffer_load_dwordx2 v[28:29], v14, s[52:55], 0 offen offset:208 // (op 10) Load value





v_mov_b32 v59, v23 // (op 25) 

ds_read_b128 v[62:65], v3 offset:6400 // (op 70) Load local data 
buffer_load_ubyte v16, v59, s[60:63], 0 offen offset:8 // (op 25) Load value

s_barrier  // (op 4969) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 



v_lshrrev_b64 v[66:67], 3, s[6:7] // (op 4) 


v_mov_b32 v68, v15 // (op 4) 
v_mov_b32 v69, v68 // (op 4) 
ds_read_b128 v[70:73], v4 // (op 75) Load local data 
ds_read_b64 v[74:75], v4 offset:16 // (op 75) Load local data 
buffer_load_dwordx4 v[32:35], v69, s[48:51], 0 offen offset:128 // (op 4) Load value
v_add_u32 v69, v69, v66 // (op 4) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4983) 
buffer_load_dwordx4 v[36:39], v69, s[48:51], 0 offen offset:128 // (op 4) Load value






ds_read_u8 v69, v0 offset:14592 // (op 80) Load local data 
v_mov_b64 v[66:67], s[28:29] // (op 17) convert

s_barrier  // (op 4997) 
v_mov_b32 v76, v20 // (op 17) 



ds_read_u8 v77, v5 offset:6144 // (op 85) Load local data 
v_mov_b32 v78, v76 // (op 17) 



buffer_load_ubyte v40, v78, s[56:59], 0 offen offset:8 // (op 17) Load value
v_add_u32 v78, v78, v66 // (op 17) 
ds_read_b128 v[80:83], v4 offset:3072 // (op 371) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[70:75], a[0:3], v69, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 88) 
ds_read_b64 v[84:85], v4 offset:3088 // (op 371) Load local data 
buffer_load_ubyte v41, v78, s[56:59], 0 offen offset:8 // (op 17) Load value





s_barrier  // (op 5020) 
ds_read_u8 v66, v5 offset:6272 // (op 375) Load local data 






ds_read_b128 v[86:89], v3 offset:8448 // (op 622) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[80:85], a[4:7], v69, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 376) 





ds_read_u8 v67, v0 offset:14720 // (op 625) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v58, v[42:45] offset:15104 // (op 253) Store local data 



ds_read_b128 v[62:65], v3 offset:10496 // (op 1008) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[86:89], v[70:75], a[8:11], v67, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 673) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[80:85], a[12:15], v67, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 627) 
ds_read_u8 v69, v0 offset:14848 // (op 1011) Load local data 





ds_read_b128 v[86:89], v3 offset:12544 // (op 1394) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:65], v[70:75], a[16:19], v69, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1059) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[80:85], a[20:23], v69, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1013) 
ds_read_u8 v67, v0 offset:14976 // (op 1397) Load local data 


s_waitcnt vmcnt(12)
ds_write_b64 v58, v[46:47] offset:15120 // (op 253) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:89], v[70:75], a[24:27], v67, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1445) 


s_barrier  // (op 5032) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:89], v[80:85], a[28:31], v67, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1399) 
s_waitcnt vmcnt(11)
ds_write_b8 v61, v18 offset:21248 // (op 255) Store local data 




s_barrier  // (op 5014) 



s_waitcnt vmcnt(10)
ds_write_b128 v22, v[48:51] offset:21504 // (op 243) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v22, v[52:55] offset:21568 // (op 243) Store local data 
s_barrier  // (op 5026) 



s_waitcnt vmcnt(8)
ds_write_b8 v60, v56 offset:29696 // (op 254) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v60, v57 offset:29700 // (op 254) Store local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 5011) 
buffer_load_dwordx4 v[42:45], v14, s[52:55], 0 offen offset:288 // (op 256) Load value



ds_read_b128 v[62:65], v7 offset:21504 // (op 245) Load local data 
buffer_load_dwordx2 v[46:47], v14, s[52:55], 0 offen offset:304 // (op 256) Load value

s_barrier  // (op 5017) 






v_add_u32 v19, 192, v19 // (op 4355) 
ds_read_b128 v[70:73], v8 offset:15104 // (op 247) Load local data 
ds_read_b64 v[74:75], v8 offset:15120 // (op 247) Load local data 
buffer_load_ubyte v18, v59, s[60:63], 0 offen offset:12 // (op 258) Load value

s_waitcnt lgkmcnt(3)
s_barrier  // (op 5023) 




v_add_u32 v23, 8, v23 // (op 4483) 
ds_read_u8 v14, v6 offset:29696 // (op 249) Load local data 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 242) 
s_barrier  // (op 5029) 
v_lshrrev_b64 v[66:67], 3, s[6:7] // (op 242) 





ds_read_u8 v59, v9 offset:21248 // (op 251) Load local data 
v_mov_b32 v69, v68 // (op 242) 



ds_read_b128 v[78:81], v8 offset:18176 // (op 381) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[70:75], a[0:3], v14, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 252) 
ds_read_b64 v[82:83], v8 offset:18192 // (op 381) Load local data 
buffer_load_dwordx4 v[48:51], v69, s[48:51], 0 offen offset:192 // (op 242) Load value
v_add_u32 v69, v69, v66 // (op 242) 



s_barrier  // (op 5089) 
ds_read_u8 v77, v9 offset:21376 // (op 385) Load local data 
buffer_load_dwordx4 v[52:55], v69, s[48:51], 0 offen offset:192 // (op 242) Load value







s_barrier  // (op 5093) 
v_add_u32 v15, 128, v15 // (op 4291) 
v_mov_b64 v[66:67], s[28:29] // (op 257) convert

v_mov_b32 v68, v76 // (op 257) 
ds_read_b128 v[84:87], v7 offset:23552 // (op 630) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[78:83], a[4:7], v14, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 377) 





ds_read_u8 v14, v6 offset:29824 // (op 633) Load local data 
buffer_load_ubyte v56, v68, s[56:59], 0 offen offset:12 // (op 257) Load value



v_add_u32 v68, v68, v66 // (op 257) 
ds_read_b128 v[62:65], v7 offset:25600 // (op 1016) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[84:87], v[70:75], a[8:11], v14, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 674) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[84:87], v[78:83], a[12:15], v14, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 628) 
ds_read_u8 v69, v6 offset:29952 // (op 1019) Load local data 





ds_read_b128 v[84:87], v7 offset:27648 // (op 1402) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:65], v[70:75], a[16:19], v69, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1060) 



s_barrier  // (op 5125) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[78:83], a[20:23], v69, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1014) 
ds_read_u8 v14, v6 offset:30080 // (op 1405) Load local data 


buffer_load_ubyte v57, v68, s[56:59], 0 offen offset:12 // (op 257) Load value



s_barrier  // (op 5129) 
s_barrier  // (op 4972) 
v_add_u32 v20, 8, v20 // (op 4419) 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[84:87], v[70:75], a[24:27], v14, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1446) 
s_waitcnt vmcnt(13)
ds_write_b128 v21, v[24:27] // (op 13) Store local data 


s_waitcnt vmcnt(12)
ds_write_b64 v21, v[28:29] offset:16 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[84:87], v[78:83], a[28:31], v14, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1400) 




s_barrier  // (op 5000) 



s_waitcnt vmcnt(11)
ds_write_b8 v31, v16 offset:6144 // (op 30) Store local data 
s_barrier  // (op 4959) 



s_waitcnt vmcnt(10)
ds_write_b128 v17, v[32:35] offset:6400 // (op 6) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v17, v[36:39] offset:6464 // (op 6) Store local data 
s_barrier  // (op 4986) 



s_waitcnt vmcnt(8)
ds_write_b8 v30, v40 offset:14592 // (op 21) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v30, v41 offset:14596 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1360) 
s_mov_b32 s1, s6 // (op 1360) convert

s_and_b32 s6, s1, 4294967294 // (op 1360) 

s_cmp_lt_u32 s0, s6 // (op 1360) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1360 // (op 1360) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoop_1360 if true)
RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoop_1360:

















v_lshrrev_b32 v14, 2, v1 // (op 4247) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4247) 
v_lshlrev_b32 v15, 1, v14 // (op 4247) 

s_lshr_b32 s6, s0, 1 // (op 4247) 

v_lshl_add_u32 v14, s16, 7, v15 // (op 4247) 

v_and_b32 v15, -4, v1 // (op 4247) 
v_sub_u32 v16, v1, v15 // (op 4247) 

v_lshlrev_b32 v15, 5, v16 // (op 4247) 

v_lshl_add_u32 v16, s6, 8, v15 // (op 4247) 


v_mov_b32 v41, 0 // (op 4247) convert
v_mov_b32 v40, v16 // (op 4247) convert

v_mul_lo_u32 v17, v14, s13 // (op 4247) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 4247) most significant: high of low * low
v_mul_lo_u32 v16, v14, s12 // (op 4247) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 4247) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v40 // (op 4247) least significant half
v_addc_co_u32 v17, s[0:1], v17, v41, s[0:1] // (op 4247) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 4247) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 4247) 

v_mov_b32 v14, v16 // (op 4247) convert

s_mul_i32 s1, s12, s23 // (op 4249) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4249) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4249) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4249) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4249) most significant: sum
s_add_u32 s1, s1, s7 // (op 4249) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4249) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 4249) 

s_mov_b32 s48, s20 // (op 4249) reg expression
s_mov_b32 s49, s21 // (op 4249) reg expression
s_mov_b32 s50, s0 // (op 4249) reg expression
s_mov_b32 s51, 131072 // (op 4249) call()

v_and_b32 v15, -4, v1 // (op 4321) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4321) 
v_sub_u32 v16, v1, v15 // (op 4321) 

s_lshr_b32 s6, s0, 1 // (op 4321) 

v_lshlrev_b32 v15, 5, v16 // (op 4321) 

v_lshl_add_u32 v16, s6, 8, v15 // (op 4321) 


v_mov_b32 v41, 0 // (op 4321) convert
v_mov_b32 v40, v16 // (op 4321) convert

v_lshrrev_b32 v15, 2, v1 // (op 4321) 
v_lshl_add_u32 v16, s17, 6, v15 // (op 4321) 

v_mul_lo_u32 v39, v16, s25 // (op 4321) most significant: low of low * high
v_mul_hi_u32 v15, v16, s24 // (op 4321) most significant: high of low * low
v_mul_lo_u32 v38, v16, s24 // (op 4321) least significant: low of low * low
v_add_u32 v39, v39, v15 // (op 4321) most significant: sum


v_add_co_u32 v16, s[0:1], v40, v38 // (op 4321) least significant half
v_addc_co_u32 v17, s[0:1], v41, v39, s[0:1] // (op 4321) most significant half



v_mul_lo_u32 v41, v17, 6 // (op 4321) most significant: low of high * low
v_mul_hi_u32 v15, v16, 6 // (op 4321) most significant: high of low * low
v_mul_lo_u32 v40, v16, 6 // (op 4321) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 4321) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[40:41] // (op 4321) 

v_mov_b32 v15, v16 // (op 4321) convert

s_mul_i32 s1, s9, 6 // (op 4323) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 4323) least significant
s_mul_hi_u32 s6, s8, 6 // (op 4323) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4323) most significant: sum

s_mul_i32 s7, s24, s1 // (op 4323) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 4323) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 4323) least significant
s_mul_hi_u32 s11, s24, s0 // (op 4323) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 4323) most significant: sum
s_add_u32 s7, s7, s11 // (op 4323) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 4323) 

s_mov_b32 s52, s26 // (op 4323) reg expression
s_mov_b32 s53, s27 // (op 4323) reg expression
s_mov_b32 s54, s0 // (op 4323) reg expression
s_mov_b32 s55, 131072 // (op 4323) call()

v_lshrrev_b32 v16, 2, v1 // (op 4385) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4385) 
v_lshlrev_b32 v17, 1, v16 // (op 4385) 

s_lshr_b32 s6, s0, 1 // (op 4385) 

v_lshl_add_u32 v16, s16, 7, v17 // (op 4385) 

v_and_b32 v17, -4, v1 // (op 4385) 
v_sub_u32 v19, v1, v17 // (op 4385) 

v_lshl_add_u32 v17, s6, 3, v19 // (op 4385) 


v_mul_lo_u32 v19, v16, s28 // (op 4385) 
v_add_u32 v19, v19, v17 // (op 4385) 


s_mul_i32 s1, s28, s23 // (op 4387) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4387) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4387) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4387) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4387) most significant: sum
s_add_u32 s1, s1, s7 // (op 4387) most significant: sum


s_mov_b32 s56, s30 // (op 4387) reg expression
s_mov_b32 s57, s31 // (op 4387) reg expression
s_mov_b32 s58, s0 // (op 4387) reg expression
s_mov_b32 s59, 131072 // (op 4387) call()

v_and_b32 v16, -4, v1 // (op 4449) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4449) 
v_sub_u32 v17, v1, v16 // (op 4449) 

s_lshr_b32 s6, s0, 1 // (op 4449) 

v_lshrrev_b32 v16, 2, v1 // (op 4449) 
v_lshl_add_u32 v20, s6, 3, v17 // (op 4449) 


v_lshl_add_u32 v17, s17, 6, v16 // (op 4449) 

v_mul_lo_u32 v16, v17, s32 // (op 4449) 

v_add_u32 v17, v20, v16 // (op 4449) 


s_mul_i32 s1, s32, s9 // (op 4451) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4451) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4451) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4451) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4451) most significant: sum
s_add_u32 s1, s1, s7 // (op 4451) most significant: sum


s_mov_b32 s60, s34 // (op 4451) reg expression
s_mov_b32 s61, s35 // (op 4451) reg expression
s_mov_b32 s62, s0 // (op 4451) reg expression
s_mov_b32 s63, 131072 // (op 4451) call()

v_and_b32 v16, -4, v1 // (op 4547) 
v_sub_u32 v20, v1, v16 // (op 4547) 

v_lshlrev_b32 v16, 5, v20 // (op 4547) 

v_lshrrev_b32 v20, 2, v1 // (op 4547) 
v_lshl_add_u32 v21, v20, 8, v16 // (op 4547) 


v_lshlrev_b32 v16, 2, v21 // (op 4547) 

v_lshrrev_b32 v20, 3, v16 // (op 4547) 

v_and_b32 v16, -4, v1 // (op 4605) 
v_sub_u32 v21, v1, v16 // (op 4605) 

v_lshlrev_b32 v16, 5, v21 // (op 4605) 

v_lshrrev_b32 v21, 2, v1 // (op 4605) 
v_lshl_add_u32 v22, v21, 7, v16 // (op 4605) 


v_mul_lo_u32 v16, v22, 6 // (op 4605) 

v_lshrrev_b32 v21, 3, v16 // (op 4605) 

v_and_b32 v16, -4, v1 // (op 4663) 
v_sub_u32 v22, v1, v16 // (op 4663) 

v_lshrrev_b32 v16, 2, v1 // (op 4663) 
v_lshl_add_u32 v23, v16, 3, v22 // (op 4663) 


v_and_b32 v16, -4, v1 // (op 4721) 
v_sub_u32 v22, v1, v16 // (op 4721) 

v_and_b32 v16, 4294967292, v1 // (op 4721) 
v_add_u32 v24, v16, v22 // (op 4721) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 1366) 
s_and_b32 s6, s0, 4294967294 // (op 1366) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 1365) 
s_mov_b32 s7, s0 // (op 1365) convert

s_cmp_lt_u32 s6, s7 // (op 1365) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1365 // (op 1365) Condition: Top (jump to RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1365 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1365:

s_add_u32 s6, s6, 1 // (op 1367) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 176) 
v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 176) 


v_mov_b32 v16, v14 // (op 176) 
v_mov_b32 v18, v16 // (op 176) 
buffer_load_dwordx4 v[28:31], v18, s[48:51], 0 offen // (op 176) Load value
v_add_u32 v18, v18, v26 // (op 176) 
buffer_load_dwordx4 v[32:35], v18, s[48:51], 0 offen // (op 176) Load value



v_add_u32 v14, 64, v14 // (op 4237) 


s_barrier  // (op 4962) 
v_mov_b32 v16, v15 // (op 190) 




buffer_load_dwordx4 v[36:39], v16, s[52:55], 0 offen // (op 190) Load value
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[28:31] offset:6400 // (op 177) Store local data 
buffer_load_dwordx2 v[40:41], v16, s[52:55], 0 offen offset:16 // (op 190) Load value

v_add_u32 v15, 96, v15 // (op 4313) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[32:35] offset:6464 // (op 177) Store local data 


v_mov_b64 v[34:35], s[28:29] // (op 191) convert

s_barrier  // (op 4975) 
v_mov_b32 v16, v19 // (op 191) 



v_mov_b32 v18, v16 // (op 191) 
buffer_load_ubyte v32, v18, s[56:59], 0 offen offset: 0 // (op 191) Load value
v_add_u32 v18, v18, v34 // (op 191) 
buffer_load_ubyte v33, v18, s[56:59], 0 offen offset: 0 // (op 191) Load value



v_add_u32 v19, 4, v19 // (op 4377) 
s_waitcnt vmcnt(3)
ds_write_b128 v21, v[36:39] // (op 187) Store local data 


v_mov_b32 v16, v17 // (op 192) 

s_waitcnt vmcnt(2)
ds_write_b64 v21, v[40:41] offset:16 // (op 187) Store local data 
buffer_load_ubyte v18, v16, s[60:63], 0 offen offset: 0 // (op 192) Load value


v_add_u32 v17, 4, v17 // (op 4441) 
s_barrier  // (op 4989) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4965) 






ds_read_b128 v[28:31], v3 offset:6400 // (op 179) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v23, v32 offset:14592 // (op 188) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4979) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v33 offset:14596 // (op 188) Store local data 




ds_read_b128 v[32:35], v4 // (op 181) Load local data 
s_barrier  // (op 5003) 
ds_read_b64 v[36:37], v4 offset:16 // (op 181) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 4993) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v18 offset:6144 // (op 189) Store local data 




ds_read_u8 v16, v0 offset:14592 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 5007) 



ds_read_u8 v18, v5 offset:6144 // (op 185) Load local data 



ds_read_b128 v[38:41], v4 offset:3072 // (op 350) Load local data 
ds_read_b64 v[42:43], v4 offset:3088 // (op 350) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:31], v[32:37], a[0:3], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 186) 



ds_read_u8 v22, v5 offset:6272 // (op 354) Load local data 



ds_read_b128 v[44:47], v3 offset:8448 // (op 568) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:31], v[38:43], a[4:7], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 355) 





ds_read_u8 v16, v0 offset:14720 // (op 571) Load local data 



ds_read_b128 v[28:31], v3 offset:10496 // (op 954) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:47], v[32:37], a[8:11], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 573) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[44:47], v[38:43], a[12:15], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 604) 
ds_read_u8 v25, v0 offset:14848 // (op 957) Load local data 





ds_read_b128 v[44:47], v3 offset:12544 // (op 1340) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[28:31], v[32:37], a[16:19], v25, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 959) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[28:31], v[38:43], a[20:23], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 990) 
ds_read_u8 v16, v0 offset:14976 // (op 1343) Load local data 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[44:47], v[32:37], a[24:27], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1345) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[44:47], v[38:43], a[28:31], v16, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 1376) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 1365) 
s_mov_b32 s7, s0 // (op 1365) convert

s_cmp_lt_u32 s6, s7 // (op 1365) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1365 // (op 1365) Condition: Bottom (jump to RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopTop_KLoopTail_1365 if true)
RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ForLoopBottom_KLoopTail_1365:























s_barrier  // (op 5113) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_133
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 



v_mov_b32 v0, v13 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 56) Store value


v_accvgpr_read v6, a4 // (op 1869) 
v_accvgpr_read v7, a5 // (op 1869) 
v_accvgpr_read v8, a6 // (op 1869) 
v_accvgpr_read v9, a7 // (op 1869) 
v_mul_f32 v14, v11, v6 // (op 1869) 
v_mul_f32 v15, v11, v7 // (op 1869) 
v_mul_f32 v16, v11, v8 // (op 1869) 
v_mul_f32 v17, v11, v9 // (op 1869) 



s_lshl_b32 s0, s4, 7 // (op 1871) 
v_add_u32 v0, v13, s0 // (op 1871) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1871) Store value


v_accvgpr_read v6, a8 // (op 1893) 
v_accvgpr_read v7, a9 // (op 1893) 
v_accvgpr_read v8, a10 // (op 1893) 
v_accvgpr_read v9, a11 // (op 1893) 
v_mul_f32 v14, v11, v6 // (op 1893) 
v_mul_f32 v15, v11, v7 // (op 1893) 
v_mul_f32 v16, v11, v8 // (op 1893) 
v_mul_f32 v17, v11, v9 // (op 1893) 



v_add_u32 v0, 128, v13 // (op 1896) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1896) Store value


v_accvgpr_read v6, a12 // (op 1902) 
v_accvgpr_read v7, a13 // (op 1902) 
v_accvgpr_read v8, a14 // (op 1902) 
v_accvgpr_read v9, a15 // (op 1902) 
v_mul_f32 v14, v11, v6 // (op 1902) 
v_mul_f32 v15, v11, v7 // (op 1902) 
v_mul_f32 v16, v11, v8 // (op 1902) 
v_mul_f32 v17, v11, v9 // (op 1902) 



s_lshl_b32 s0, s4, 7 // (op 1898) 
s_add_u32 s0, s0, 128 // (op 1898) 
v_add_u32 v0, v13, s0 // (op 1898) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1898) Store value


v_accvgpr_read v6, a16 // (op 1929) 
v_accvgpr_read v7, a17 // (op 1929) 
v_accvgpr_read v8, a18 // (op 1929) 
v_accvgpr_read v9, a19 // (op 1929) 
v_mul_f32 v14, v11, v6 // (op 1929) 
v_mul_f32 v15, v11, v7 // (op 1929) 
v_mul_f32 v16, v11, v8 // (op 1929) 
v_mul_f32 v17, v11, v9 // (op 1929) 



v_add_u32 v0, 256, v13 // (op 1932) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1932) Store value


v_accvgpr_read v6, a20 // (op 1938) 
v_accvgpr_read v7, a21 // (op 1938) 
v_accvgpr_read v8, a22 // (op 1938) 
v_accvgpr_read v9, a23 // (op 1938) 
v_mul_f32 v14, v11, v6 // (op 1938) 
v_mul_f32 v15, v11, v7 // (op 1938) 
v_mul_f32 v16, v11, v8 // (op 1938) 
v_mul_f32 v17, v11, v9 // (op 1938) 



s_lshl_b32 s0, s4, 7 // (op 1934) 
s_add_u32 s0, s0, 256 // (op 1934) 
v_add_u32 v0, v13, s0 // (op 1934) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1934) Store value


v_accvgpr_read v6, a24 // (op 1965) 
v_accvgpr_read v7, a25 // (op 1965) 
v_accvgpr_read v8, a26 // (op 1965) 
v_accvgpr_read v9, a27 // (op 1965) 
v_mul_f32 v14, v11, v6 // (op 1965) 
v_mul_f32 v15, v11, v7 // (op 1965) 
v_mul_f32 v16, v11, v8 // (op 1965) 
v_mul_f32 v17, v11, v9 // (op 1965) 



v_add_u32 v0, 384, v13 // (op 1968) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1968) Store value


v_accvgpr_read v6, a28 // (op 1974) 
v_accvgpr_read v7, a29 // (op 1974) 
v_accvgpr_read v8, a30 // (op 1974) 
v_accvgpr_read v9, a31 // (op 1974) 
v_mul_f32 v14, v11, v6 // (op 1974) 
v_mul_f32 v15, v11, v7 // (op 1974) 
v_mul_f32 v16, v11, v8 // (op 1974) 
v_mul_f32 v17, v11, v9 // (op 1974) 



s_lshl_b32 s0, s4, 7 // (op 1970) 
s_add_u32 s0, s0, 384 // (op 1970) 
v_add_u32 v0, v13, s0 // (op 1970) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1970) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalFalse_0_beta_133:


v_add_u32 v0, 256, v12 // (op 2111) 

buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2111) Load value

v_accvgpr_read v14, a16 // (op 2103) 
v_accvgpr_read v15, a17 // (op 2103) 
v_accvgpr_read v16, a18 // (op 2103) 
v_accvgpr_read v17, a19 // (op 2103) 
v_mul_f32 v18, v11, v14 // (op 2103) 
v_mul_f32 v19, v11, v15 // (op 2103) 
v_mul_f32 v20, v11, v16 // (op 2103) 
v_mul_f32 v21, v11, v17 // (op 2103) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2110) 
v_mul_f32 v15, v10, v7 // (op 2110) 
v_mul_f32 v16, v10, v8 // (op 2110) 
v_mul_f32 v17, v10, v9 // (op 2110) 

v_add_f32 v6, v14, v18 // (op 2104) 
v_add_f32 v7, v15, v19 // (op 2104) 
v_add_f32 v8, v16, v20 // (op 2104) 
v_add_f32 v9, v17, v21 // (op 2104) 




v_add_u32 v0, 256, v13 // (op 2106) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2106) Store value




s_lshl_b32 s0, s2, 7 // (op 2115) 
s_add_u32 s0, s0, 256 // (op 2115) 
v_add_u32 v0, v12, s0 // (op 2115) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2115) Load value

v_accvgpr_read v14, a20 // (op 2117) 
v_accvgpr_read v15, a21 // (op 2117) 
v_accvgpr_read v16, a22 // (op 2117) 
v_accvgpr_read v17, a23 // (op 2117) 
v_mul_f32 v18, v11, v14 // (op 2117) 
v_mul_f32 v19, v11, v15 // (op 2117) 
v_mul_f32 v20, v11, v16 // (op 2117) 
v_mul_f32 v21, v11, v17 // (op 2117) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2113) 
v_mul_f32 v15, v10, v7 // (op 2113) 
v_mul_f32 v16, v10, v8 // (op 2113) 
v_mul_f32 v17, v10, v9 // (op 2113) 

v_add_f32 v6, v14, v18 // (op 2114) 
v_add_f32 v7, v15, v19 // (op 2114) 
v_add_f32 v8, v16, v20 // (op 2114) 
v_add_f32 v9, v17, v21 // (op 2114) 




s_lshl_b32 s0, s4, 7 // (op 2108) 
s_add_u32 s0, s0, 256 // (op 2108) 
v_add_u32 v0, v13, s0 // (op 2108) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2108) Store value




v_add_u32 v0, 128, v12 // (op 2070) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2070) Load value

v_accvgpr_read v14, a8 // (op 2062) 
v_accvgpr_read v15, a9 // (op 2062) 
v_accvgpr_read v16, a10 // (op 2062) 
v_accvgpr_read v17, a11 // (op 2062) 
v_mul_f32 v18, v11, v14 // (op 2062) 
v_mul_f32 v19, v11, v15 // (op 2062) 
v_mul_f32 v20, v11, v16 // (op 2062) 
v_mul_f32 v21, v11, v17 // (op 2062) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2069) 
v_mul_f32 v15, v10, v7 // (op 2069) 
v_mul_f32 v16, v10, v8 // (op 2069) 
v_mul_f32 v17, v10, v9 // (op 2069) 

v_add_f32 v6, v14, v18 // (op 2063) 
v_add_f32 v7, v15, v19 // (op 2063) 
v_add_f32 v8, v16, v20 // (op 2063) 
v_add_f32 v9, v17, v21 // (op 2063) 




v_add_u32 v0, 128, v13 // (op 2065) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2065) Store value




s_lshl_b32 s0, s2, 7 // (op 2074) 
s_add_u32 s0, s0, 128 // (op 2074) 
v_add_u32 v0, v12, s0 // (op 2074) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2074) Load value

v_accvgpr_read v14, a12 // (op 2076) 
v_accvgpr_read v15, a13 // (op 2076) 
v_accvgpr_read v16, a14 // (op 2076) 
v_accvgpr_read v17, a15 // (op 2076) 
v_mul_f32 v18, v11, v14 // (op 2076) 
v_mul_f32 v19, v11, v15 // (op 2076) 
v_mul_f32 v20, v11, v16 // (op 2076) 
v_mul_f32 v21, v11, v17 // (op 2076) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2072) 
v_mul_f32 v15, v10, v7 // (op 2072) 
v_mul_f32 v16, v10, v8 // (op 2072) 
v_mul_f32 v17, v10, v9 // (op 2072) 

v_add_f32 v6, v14, v18 // (op 2073) 
v_add_f32 v7, v15, v19 // (op 2073) 
v_add_f32 v8, v16, v20 // (op 2073) 
v_add_f32 v9, v17, v21 // (op 2073) 




s_lshl_b32 s0, s4, 7 // (op 2067) 
s_add_u32 s0, s0, 128 // (op 2067) 
v_add_u32 v0, v13, s0 // (op 2067) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2067) Store value




v_add_u32 v0, 384, v12 // (op 2152) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2152) Load value

v_accvgpr_read v14, a24 // (op 2144) 
v_accvgpr_read v15, a25 // (op 2144) 
v_accvgpr_read v16, a26 // (op 2144) 
v_accvgpr_read v17, a27 // (op 2144) 
v_mul_f32 v18, v11, v14 // (op 2144) 
v_mul_f32 v19, v11, v15 // (op 2144) 
v_mul_f32 v20, v11, v16 // (op 2144) 
v_mul_f32 v21, v11, v17 // (op 2144) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2151) 
v_mul_f32 v15, v10, v7 // (op 2151) 
v_mul_f32 v16, v10, v8 // (op 2151) 
v_mul_f32 v17, v10, v9 // (op 2151) 

v_add_f32 v6, v14, v18 // (op 2145) 
v_add_f32 v7, v15, v19 // (op 2145) 
v_add_f32 v8, v16, v20 // (op 2145) 
v_add_f32 v9, v17, v21 // (op 2145) 




v_add_u32 v0, 384, v13 // (op 2147) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2147) Store value




s_lshl_b32 s0, s2, 7 // (op 2156) 
s_add_u32 s0, s0, 384 // (op 2156) 
v_add_u32 v0, v12, s0 // (op 2156) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2156) Load value

v_accvgpr_read v14, a28 // (op 2158) 
v_accvgpr_read v15, a29 // (op 2158) 
v_accvgpr_read v16, a30 // (op 2158) 
v_accvgpr_read v17, a31 // (op 2158) 
v_mul_f32 v18, v11, v14 // (op 2158) 
v_mul_f32 v19, v11, v15 // (op 2158) 
v_mul_f32 v20, v11, v16 // (op 2158) 
v_mul_f32 v21, v11, v17 // (op 2158) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2154) 
v_mul_f32 v15, v10, v7 // (op 2154) 
v_mul_f32 v16, v10, v8 // (op 2154) 
v_mul_f32 v17, v10, v9 // (op 2154) 

v_add_f32 v6, v14, v18 // (op 2155) 
v_add_f32 v7, v15, v19 // (op 2155) 
v_add_f32 v8, v16, v20 // (op 2155) 
v_add_f32 v9, v17, v21 // (op 2155) 




s_lshl_b32 s0, s4, 7 // (op 2149) 
s_add_u32 s0, s0, 384 // (op 2149) 
v_add_u32 v0, v13, s0 // (op 2149) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2149) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 145) Load value

v_accvgpr_read v14, a0 // (op 142) 
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 146) 
v_mul_f32 v15, v10, v7 // (op 146) 
v_mul_f32 v16, v10, v8 // (op 146) 
v_mul_f32 v17, v10, v9 // (op 146) 

v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 
v_add_f32 v8, v16, v20 // (op 143) 
v_add_f32 v9, v17, v21 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 2032) 
v_add_u32 v0, v12, s0 // (op 2032) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2032) Load value

v_accvgpr_read v14, a4 // (op 2029) 
v_accvgpr_read v15, a5 // (op 2029) 
v_accvgpr_read v16, a6 // (op 2029) 
v_accvgpr_read v17, a7 // (op 2029) 
v_mul_f32 v18, v11, v14 // (op 2029) 
v_mul_f32 v19, v11, v15 // (op 2029) 
v_mul_f32 v20, v11, v16 // (op 2029) 
v_mul_f32 v21, v11, v17 // (op 2029) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2033) 
v_mul_f32 v15, v10, v7 // (op 2033) 
v_mul_f32 v16, v10, v8 // (op 2033) 
v_mul_f32 v17, v10, v9 // (op 2033) 

v_add_f32 v6, v14, v18 // (op 2030) 
v_add_f32 v7, v15, v19 // (op 2030) 
v_add_f32 v8, v16, v20 // (op 2030) 
v_add_f32 v9, v17, v21 // (op 2030) 




s_lshl_b32 s0, s4, 7 // (op 2031) 
v_add_u32 v0, v13, s0 // (op 2031) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2031) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
.LRR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_end:
.size RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2, .LRR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2_end-RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
  .amdhsa_next_free_vgpr 124
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 30208 // lds bytes
  .amdhsa_accum_offset 92
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
    DataTypeB: B6
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
    macrotile: [128, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2
    .symbol: RR_GEMM_TN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 30208
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 90
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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

