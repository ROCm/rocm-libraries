// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 6954) 
v_and_b32 v3, 15, v1 // (op 6954) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 6954) 


v_and_b32 v0, 63, v1 // (op 6954) 
v_lshrrev_b32 v3, 4, v0 // (op 6954) 

v_lshlrev_b32 v0, 5, v3 // (op 6954) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 6954) 


v_lshlrev_b32 v0, 2, v3 // (op 6954) 

v_lshrrev_b32 v3, 3, v0 // (op 6954) 

v_lshrrev_b32 v0, 6, v1 // (op 7012) 
v_and_b32 v4, 1, v0 // (op 7012) 

v_and_b32 v0, 15, v1 // (op 7012) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 7012) 


v_and_b32 v0, 63, v1 // (op 7012) 
v_lshrrev_b32 v4, 4, v0 // (op 7012) 

v_lshlrev_b32 v0, 5, v4 // (op 7012) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 7012) 


v_lshlrev_b32 v0, 2, v4 // (op 7012) 

v_lshrrev_b32 v4, 3, v0 // (op 7012) 

v_lshrrev_b32 v0, 7, v1 // (op 7070) 
v_and_b32 v5, 15, v1 // (op 7070) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 7070) 


v_and_b32 v0, 63, v1 // (op 7070) 
v_lshrrev_b32 v5, 4, v0 // (op 7070) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 7070) 


v_lshrrev_b32 v5, 6, v1 // (op 7128) 
v_and_b32 v6, 1, v5 // (op 7128) 

v_and_b32 v5, 15, v1 // (op 7128) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 7128) 


v_and_b32 v5, 63, v1 // (op 7128) 
v_lshrrev_b32 v6, 4, v5 // (op 7128) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 7128) 


v_lshrrev_b32 v6, 7, v1 // (op 7180) 
v_and_b32 v7, 15, v1 // (op 7180) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 7180) 


v_and_b32 v6, 63, v1 // (op 7180) 
v_lshrrev_b32 v7, 4, v6 // (op 7180) 

v_lshlrev_b32 v6, 5, v7 // (op 7180) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 7180) 


v_lshlrev_b32 v6, 2, v7 // (op 7180) 

v_lshrrev_b32 v7, 3, v6 // (op 7180) 

v_lshrrev_b32 v6, 6, v1 // (op 7209) 
v_and_b32 v8, 1, v6 // (op 7209) 

v_and_b32 v6, 15, v1 // (op 7209) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 7209) 


v_and_b32 v6, 63, v1 // (op 7209) 
v_lshrrev_b32 v8, 4, v6 // (op 7209) 

v_lshlrev_b32 v6, 5, v8 // (op 7209) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 7209) 


v_lshlrev_b32 v6, 2, v8 // (op 7209) 

v_lshrrev_b32 v8, 3, v6 // (op 7209) 

v_lshrrev_b32 v6, 7, v1 // (op 7238) 
v_and_b32 v9, 15, v1 // (op 7238) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 7238) 


v_and_b32 v6, 63, v1 // (op 7238) 
v_lshrrev_b32 v9, 4, v6 // (op 7238) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 7238) 


v_lshrrev_b32 v9, 6, v1 // (op 7267) 
v_and_b32 v10, 1, v9 // (op 7267) 

v_and_b32 v9, 15, v1 // (op 7267) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 7267) 


v_and_b32 v9, 63, v1 // (op 7267) 
v_lshrrev_b32 v10, 4, v9 // (op 7267) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 7267) 


v_lshrrev_b32 v10, 7, v1 // (op 7296) 
v_and_b32 v11, 15, v1 // (op 7296) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 7296) 


v_and_b32 v10, 63, v1 // (op 7296) 
v_lshrrev_b32 v11, 4, v10 // (op 7296) 

v_lshlrev_b32 v10, 5, v11 // (op 7296) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 7296) 


v_lshlrev_b32 v10, 2, v11 // (op 7296) 

v_lshrrev_b32 v11, 3, v10 // (op 7296) 

v_lshrrev_b32 v10, 6, v1 // (op 7325) 
v_and_b32 v12, 1, v10 // (op 7325) 

v_and_b32 v10, 15, v1 // (op 7325) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 7325) 


v_and_b32 v10, 63, v1 // (op 7325) 
v_lshrrev_b32 v12, 4, v10 // (op 7325) 

v_lshlrev_b32 v10, 5, v12 // (op 7325) 

v_lshl_add_u32 v12, v13, 7, v10 // (op 7325) 


v_lshlrev_b32 v10, 2, v12 // (op 7325) 

v_lshrrev_b32 v12, 3, v10 // (op 7325) 

v_lshrrev_b32 v10, 7, v1 // (op 7354) 
v_and_b32 v13, 15, v1 // (op 7354) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 7354) 


v_and_b32 v10, 63, v1 // (op 7354) 
v_lshrrev_b32 v13, 4, v10 // (op 7354) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 7354) 


v_lshrrev_b32 v13, 6, v1 // (op 7383) 
v_and_b32 v14, 1, v13 // (op 7383) 

v_and_b32 v13, 15, v1 // (op 7383) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 7383) 


v_and_b32 v13, 63, v1 // (op 7383) 
v_lshrrev_b32 v14, 4, v13 // (op 7383) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 7383) 


v_lshrrev_b32 v14, 7, v1 // (op 7412) 
v_and_b32 v15, 15, v1 // (op 7412) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 7412) 


v_and_b32 v14, 63, v1 // (op 7412) 
v_lshrrev_b32 v15, 4, v14 // (op 7412) 

v_lshlrev_b32 v14, 5, v15 // (op 7412) 

v_lshl_add_u32 v15, v16, 7, v14 // (op 7412) 


v_lshlrev_b32 v14, 2, v15 // (op 7412) 

v_lshrrev_b32 v15, 3, v14 // (op 7412) 

v_lshrrev_b32 v14, 6, v1 // (op 7441) 
v_and_b32 v16, 1, v14 // (op 7441) 

v_and_b32 v14, 15, v1 // (op 7441) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 7441) 


v_and_b32 v14, 63, v1 // (op 7441) 
v_lshrrev_b32 v16, 4, v14 // (op 7441) 

v_lshlrev_b32 v14, 5, v16 // (op 7441) 

v_lshl_add_u32 v16, v17, 7, v14 // (op 7441) 


v_lshlrev_b32 v14, 2, v16 // (op 7441) 

v_lshrrev_b32 v16, 3, v14 // (op 7441) 

v_lshrrev_b32 v14, 7, v1 // (op 7470) 
v_and_b32 v17, 15, v1 // (op 7470) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 7470) 


v_and_b32 v14, 63, v1 // (op 7470) 
v_lshrrev_b32 v17, 4, v14 // (op 7470) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 7470) 


v_lshrrev_b32 v17, 6, v1 // (op 7499) 
v_and_b32 v18, 1, v17 // (op 7499) 

v_and_b32 v17, 15, v1 // (op 7499) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 7499) 


v_and_b32 v17, 63, v1 // (op 7499) 
v_lshrrev_b32 v18, 4, v17 // (op 7499) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 7499) 


v_lshrrev_b32 v18, 7, v1 // (op 6892) 
v_and_b32 v19, 63, v1 // (op 6892) 
v_lshrrev_b32 v20, 4, v19 // (op 6892) 
v_lshlrev_b32 v21, 2, v20 // (op 6892) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 6892) 


v_lshl_add_u32 v18, s16, 6, v20 // (op 6892) 

v_lshrrev_b32 v20, 2, v19 // (op 6892) 

v_and_b32 v19, 3, v20 // (op 6892) 

v_and_b32 v20, 3, v1 // (op 6892) 
v_lshl_add_u32 v21, v19, 2, v20 // (op 6892) 


v_lshrrev_b32 v19, 6, v1 // (op 6892) 
v_and_b32 v20, 1, v19 // (op 6892) 

v_lshl_add_u32 v19, v20, 4, v21 // (op 6892) 


v_lshl_add_u32 v20, s17, 7, v19 // (op 6892) 

v_mul_lo_u32 v19, v20, s2 // (op 6892) 

v_add_lshl_u32 v20, v18, v19, 2 // (op 6892) 


v_lshrrev_b32 v18, 7, v1 // (op 7149) 
v_and_b32 v19, 63, v1 // (op 7149) 
v_lshrrev_b32 v21, 4, v19 // (op 7149) 
s_mul_i32 s1, s2, s9 // (op 6894) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 6894) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 6894) least significant
s_mul_hi_u32 s19, s2, s8 // (op 6894) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6894) most significant: sum
s_add_u32 s1, s1, s19 // (op 6894) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 6894) 

v_lshlrev_b32 v22, 2, v21 // (op 7149) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 7149) 


v_lshl_add_u32 v18, s16, 6, v21 // (op 7149) 

v_lshrrev_b32 v21, 2, v19 // (op 7149) 

v_and_b32 v19, 3, v21 // (op 7149) 

v_and_b32 v21, 3, v1 // (op 7149) 
v_lshl_add_u32 v22, v19, 2, v21 // (op 7149) 


v_lshrrev_b32 v19, 6, v1 // (op 7149) 
v_and_b32 v21, 1, v19 // (op 7149) 

v_lshl_add_u32 v19, v21, 4, v22 // (op 7149) 


v_lshl_add_u32 v21, s17, 7, v19 // (op 7149) 

v_mul_lo_u32 v19, v21, s4 // (op 7149) 

v_add_lshl_u32 v21, v18, v19, 2 // (op 7149) 


s_mov_b32 s40, s6 // (op 6894) reg expression
s_mov_b32 s41, s7 // (op 6894) reg expression
s_mov_b32 s42, s18 // (op 6894) reg expression
s_mov_b32 s43, 131072 // (op 6894) call()

s_mul_i32 s1, s4, s9 // (op 7151) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 7151) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 7151) least significant
s_mul_hi_u32 s7, s4, s8 // (op 7151) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7151) most significant: sum
s_add_u32 s1, s1, s7 // (op 7151) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 7151) 

s_mov_b32 s44, s10 // (op 7151) reg expression
s_mov_b32 s45, s11 // (op 7151) reg expression
s_mov_b32 s46, s6 // (op 7151) reg expression
s_mov_b32 s47, 131072 // (op 7151) call()


v_mov_b32 v18, s14 // (op 2) Move value
v_mov_b32 v19, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 482) call()
v_accvgpr_write a5, 0.00000 // (op 482) call()
v_accvgpr_write a6, 0.00000 // (op 482) call()
v_accvgpr_write a7, 0.00000 // (op 482) call()
v_accvgpr_write a8, 0.00000 // (op 780) call()
v_accvgpr_write a9, 0.00000 // (op 780) call()
v_accvgpr_write a10, 0.00000 // (op 780) call()
v_accvgpr_write a11, 0.00000 // (op 780) call()
v_accvgpr_write a12, 0.00000 // (op 1078) call()
v_accvgpr_write a13, 0.00000 // (op 1078) call()
v_accvgpr_write a14, 0.00000 // (op 1078) call()
v_accvgpr_write a15, 0.00000 // (op 1078) call()
v_accvgpr_write a16, 0.00000 // (op 1512) call()
v_accvgpr_write a17, 0.00000 // (op 1512) call()
v_accvgpr_write a18, 0.00000 // (op 1512) call()
v_accvgpr_write a19, 0.00000 // (op 1512) call()
v_accvgpr_write a20, 0.00000 // (op 2005) call()
v_accvgpr_write a21, 0.00000 // (op 2005) call()
v_accvgpr_write a22, 0.00000 // (op 2005) call()
v_accvgpr_write a23, 0.00000 // (op 2005) call()
v_accvgpr_write a24, 0.00000 // (op 2006) call()
v_accvgpr_write a25, 0.00000 // (op 2006) call()
v_accvgpr_write a26, 0.00000 // (op 2006) call()
v_accvgpr_write a27, 0.00000 // (op 2006) call()
v_accvgpr_write a28, 0.00000 // (op 2007) call()
v_accvgpr_write a29, 0.00000 // (op 2007) call()
v_accvgpr_write a30, 0.00000 // (op 2007) call()
v_accvgpr_write a31, 0.00000 // (op 2007) call()
v_lshrrev_b32 v22, 2, v1 // (op 6654) 
v_lshl_add_u32 v23, s16, 6, v22 // (op 6654) 

v_and_b32 v22, -4, v1 // (op 6654) 
v_sub_u32 v24, v1, v22 // (op 6654) 

v_lshlrev_b32 v22, 5, v24 // (op 6654) 

v_mov_b32 v25, 0 // (op 6654) convert
v_mov_b32 v24, v22 // (op 6654) convert

v_mul_lo_u32 v27, v23, s13 // (op 6654) most significant: low of low * high
v_mul_hi_u32 v22, v23, s12 // (op 6654) most significant: high of low * low
v_mul_lo_u32 v26, v23, s12 // (op 6654) least significant: low of low * low
v_add_u32 v27, v27, v22 // (op 6654) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v24 // (op 6654) least significant half
v_addc_co_u32 v27, s[0:1], v27, v25, s[0:1] // (op 6654) most significant half



v_lshlrev_b64 v[22:23], 2, v[26:27] // (op 6654) 

v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 6654) 

v_mov_b32 v22, v24 // (op 6654) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 6656) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6656) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6656) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6656) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6656) most significant: sum
s_add_u32 s1, s1, s7 // (op 6656) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6656) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6656) 

s_mov_b32 s48, s20 // (op 6656) reg expression
s_mov_b32 s49, s21 // (op 6656) reg expression
s_mov_b32 s50, s0 // (op 6656) reg expression
s_mov_b32 s51, 131072 // (op 6656) call()

v_lshrrev_b32 v23, 2, v1 // (op 6673) 
v_lshl_add_u32 v24, s16, 6, v23 // (op 6673) 

v_and_b32 v23, -4, v1 // (op 6673) 
v_sub_u32 v25, v1, v23 // (op 6673) 

v_lshlrev_b32 v23, 5, v25 // (op 6673) 

v_mov_b32 v27, 0 // (op 6673) convert
v_mov_b32 v26, v23 // (op 6673) convert

v_mul_lo_u32 v29, v24, s13 // (op 6673) most significant: low of low * high
v_mul_hi_u32 v23, v24, s12 // (op 6673) most significant: high of low * low
v_mul_lo_u32 v28, v24, s12 // (op 6673) least significant: low of low * low
v_add_u32 v29, v29, v23 // (op 6673) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v26 // (op 6673) least significant half
v_addc_co_u32 v29, s[0:1], v29, v27, s[0:1] // (op 6673) most significant half



v_lshlrev_b64 v[26:27], 2, v[28:29] // (op 6673) 

v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 6673) 

v_mov_b32 v23, v24 // (op 6673) convert

s_mul_i32 s1, s12, s23 // (op 6675) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6675) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6675) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6675) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6675) most significant: sum
s_add_u32 s1, s1, s7 // (op 6675) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6675) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6675) 

s_mov_b32 s48, s20 // (op 6675) reg expression
s_mov_b32 s49, s21 // (op 6675) reg expression
s_mov_b32 s50, s0 // (op 6675) reg expression
s_mov_b32 s51, 131072 // (op 6675) call()

v_and_b32 v24, -4, v1 // (op 6718) 
v_sub_u32 v25, v1, v24 // (op 6718) 

v_lshlrev_b32 v24, 5, v25 // (op 6718) 

v_mov_b32 v27, 0 // (op 6718) convert
v_mov_b32 v26, v24 // (op 6718) convert

v_lshrrev_b32 v24, 2, v1 // (op 6718) 
v_lshlrev_b32 v25, 1, v24 // (op 6718) 

v_lshl_add_u32 v24, s17, 7, v25 // (op 6718) 

v_mul_lo_u32 v29, v24, s25 // (op 6718) most significant: low of low * high
v_mul_hi_u32 v25, v24, s24 // (op 6718) most significant: high of low * low
v_mul_lo_u32 v28, v24, s24 // (op 6718) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 6718) most significant: sum


v_add_co_u32 v24, s[0:1], v26, v28 // (op 6718) least significant half
v_addc_co_u32 v25, s[0:1], v27, v29, s[0:1] // (op 6718) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 6718) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 6718) 

v_mov_b32 v24, v26 // (op 6718) convert

s_mul_i32 s1, s24, s9 // (op 6720) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6720) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6720) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6720) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6720) most significant: sum
s_add_u32 s1, s1, s7 // (op 6720) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6720) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6720) 

s_mov_b32 s52, s26 // (op 6720) reg expression
s_mov_b32 s53, s27 // (op 6720) reg expression
s_mov_b32 s54, s0 // (op 6720) reg expression
s_mov_b32 s55, 131072 // (op 6720) call()

v_and_b32 v25, -4, v1 // (op 6737) 
v_sub_u32 v26, v1, v25 // (op 6737) 

v_lshlrev_b32 v25, 5, v26 // (op 6737) 

v_mov_b32 v27, 0 // (op 6737) convert
v_mov_b32 v26, v25 // (op 6737) convert

v_lshrrev_b32 v25, 2, v1 // (op 6737) 
v_lshlrev_b32 v28, 1, v25 // (op 6737) 

v_lshl_add_u32 v25, s17, 7, v28 // (op 6737) 

v_mul_lo_u32 v29, v25, s25 // (op 6737) most significant: low of low * high
v_mul_hi_u32 v30, v25, s24 // (op 6737) most significant: high of low * low
v_mul_lo_u32 v28, v25, s24 // (op 6737) least significant: low of low * low
v_add_u32 v29, v29, v30 // (op 6737) most significant: sum


v_add_co_u32 v30, s[0:1], v26, v28 // (op 6737) least significant half
v_addc_co_u32 v31, s[0:1], v27, v29, s[0:1] // (op 6737) most significant half

v_lshlrev_b64 v[30:31], 2, v[30:31] // (op 6737) 


v_lshrrev_b64 v[28:29], 3, v[30:31] // (op 6737) 

v_mov_b32 v25, v28 // (op 6737) convert

s_mul_i32 s1, s24, s9 // (op 6739) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6739) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6739) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6739) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6739) most significant: sum
s_add_u32 s1, s1, s7 // (op 6739) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6739) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6739) 

s_mov_b32 s52, s26 // (op 6739) reg expression
s_mov_b32 s53, s27 // (op 6739) reg expression
s_mov_b32 s54, s0 // (op 6739) reg expression
s_mov_b32 s55, 131072 // (op 6739) call()

v_lshrrev_b32 v26, 2, v1 // (op 6782) 
v_lshl_add_u32 v27, s16, 6, v26 // (op 6782) 

v_and_b32 v26, -4, v1 // (op 6782) 
v_sub_u32 v28, v1, v26 // (op 6782) 

v_mul_lo_u32 v26, v27, s28 // (op 6782) 
v_add_u32 v26, v26, v28 // (op 6782) 


s_mul_i32 s1, s28, s23 // (op 6784) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6784) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6784) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6784) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6784) most significant: sum
s_add_u32 s1, s1, s7 // (op 6784) most significant: sum


s_mov_b32 s56, s30 // (op 6784) reg expression
s_mov_b32 s57, s31 // (op 6784) reg expression
s_mov_b32 s58, s0 // (op 6784) reg expression
s_mov_b32 s59, 131072 // (op 6784) call()

v_lshrrev_b32 v27, 2, v1 // (op 6801) 
v_lshl_add_u32 v28, s16, 6, v27 // (op 6801) 

v_and_b32 v27, -4, v1 // (op 6801) 
v_sub_u32 v29, v1, v27 // (op 6801) 

v_mul_lo_u32 v27, v28, s28 // (op 6801) 
v_add_u32 v27, v27, v29 // (op 6801) 


s_mul_i32 s1, s28, s23 // (op 6803) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6803) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6803) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6803) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6803) most significant: sum
s_add_u32 s1, s1, s7 // (op 6803) most significant: sum


s_mov_b32 s56, s30 // (op 6803) reg expression
s_mov_b32 s57, s31 // (op 6803) reg expression
s_mov_b32 s58, s0 // (op 6803) reg expression
s_mov_b32 s59, 131072 // (op 6803) call()

v_and_b32 v28, -4, v1 // (op 6846) 
v_sub_u32 v29, v1, v28 // (op 6846) 

v_lshrrev_b32 v28, 2, v1 // (op 6846) 
v_lshlrev_b32 v30, 1, v28 // (op 6846) 

v_lshl_add_u32 v28, s17, 7, v30 // (op 6846) 

v_mul_lo_u32 v30, v28, s32 // (op 6846) 

v_add_u32 v28, v29, v30 // (op 6846) 


s_mul_i32 s1, s32, s9 // (op 6848) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6848) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6848) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6848) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6848) most significant: sum
s_add_u32 s1, s1, s7 // (op 6848) most significant: sum


s_mov_b32 s60, s34 // (op 6848) reg expression
s_mov_b32 s61, s35 // (op 6848) reg expression
s_mov_b32 s62, s0 // (op 6848) reg expression
s_mov_b32 s63, 131072 // (op 6848) call()

v_and_b32 v29, -4, v1 // (op 6865) 
v_sub_u32 v30, v1, v29 // (op 6865) 

v_lshrrev_b32 v29, 2, v1 // (op 6865) 
v_lshlrev_b32 v31, 1, v29 // (op 6865) 

v_lshl_add_u32 v29, s17, 7, v31 // (op 6865) 

v_mul_lo_u32 v31, v29, s32 // (op 6865) 

v_add_u32 v29, v30, v31 // (op 6865) 


s_mul_i32 s1, s32, s9 // (op 6867) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6867) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6867) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6867) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6867) most significant: sum
s_add_u32 s1, s1, s7 // (op 6867) most significant: sum


s_mov_b32 s60, s34 // (op 6867) reg expression
s_mov_b32 s61, s35 // (op 6867) reg expression
s_mov_b32 s62, s0 // (op 6867) reg expression
s_mov_b32 s63, 131072 // (op 6867) call()


s_lshl_b64 s[0:1], s[24:25], 2 // (op 6194) 
v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 6194) 


v_mov_b32 v32, v24 // (op 6194) 
buffer_load_dwordx4 v[34:37], v32, s[52:55], 0 offen // (op 6194) Load value
v_add_u32 v32, v32, v30 // (op 6194) 
buffer_load_dwordx4 v[38:41], v32, s[52:55], 0 offen // (op 6194) Load value



v_mov_b64 v[30:31], s[32:33] // (op 6197) convert

v_mov_b32 v32, v28 // (op 6197) 
buffer_load_ubyte v42, v32, s[60:63], 0 offen offset: 0 // (op 6197) Load value
v_add_u32 v32, v32, v30 // (op 6197) 
buffer_load_ubyte v43, v32, s[60:63], 0 offen offset: 0 // (op 6197) Load value





buffer_load_dwordx4 v[30:33], v22, s[48:51], 0 offen // (op 6200) Load value



buffer_load_ubyte v44, v26, s[56:59], 0 offen offset: 0 // (op 6203) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 6206) 
v_lshrrev_b64 v[46:47], 3, s[0:1] // (op 6206) 


v_mov_b32 v45, v24 // (op 6206) 
buffer_load_dwordx4 v[48:51], v45, s[52:55], 0 offen offset:64 // (op 6206) Load value
v_add_u32 v45, v45, v46 // (op 6206) 
buffer_load_dwordx4 v[52:55], v45, s[52:55], 0 offen offset:64 // (op 6206) Load value



v_mov_b64 v[46:47], s[32:33] // (op 6209) convert

v_mov_b32 v45, v28 // (op 6209) 
buffer_load_ubyte v56, v45, s[60:63], 0 offen offset:4 // (op 6209) Load value
v_add_u32 v45, v45, v46 // (op 6209) 
buffer_load_ubyte v57, v45, s[60:63], 0 offen offset:4 // (op 6209) Load value





buffer_load_dwordx4 v[58:61], v22, s[48:51], 0 offen offset:64 // (op 6212) Load value



buffer_load_ubyte v45, v26, s[56:59], 0 offen offset:4 // (op 6215) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 6218) 
v_lshrrev_b64 v[46:47], 3, s[0:1] // (op 6218) 


v_mov_b32 v62, v24 // (op 6218) 
buffer_load_dwordx4 v[64:67], v62, s[52:55], 0 offen offset:128 // (op 6218) Load value
v_add_u32 v62, v62, v46 // (op 6218) 
buffer_load_dwordx4 v[68:71], v62, s[52:55], 0 offen offset:128 // (op 6218) Load value



v_mov_b64 v[46:47], s[32:33] // (op 6221) convert

v_mov_b32 v62, v28 // (op 6221) 
buffer_load_ubyte v72, v62, s[60:63], 0 offen offset:8 // (op 6221) Load value
v_add_u32 v62, v62, v46 // (op 6221) 
buffer_load_ubyte v73, v62, s[60:63], 0 offen offset:8 // (op 6221) Load value





buffer_load_dwordx4 v[74:77], v22, s[48:51], 0 offen offset:128 // (op 6224) Load value



buffer_load_ubyte v46, v26, s[56:59], 0 offen offset:8 // (op 6227) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 6230) 
v_lshrrev_b64 v[62:63], 3, s[0:1] // (op 6230) 


v_mov_b32 v47, v24 // (op 6230) 
buffer_load_dwordx4 v[78:81], v47, s[52:55], 0 offen offset:192 // (op 6230) Load value
v_add_u32 v47, v47, v62 // (op 6230) 
buffer_load_dwordx4 v[82:85], v47, s[52:55], 0 offen offset:192 // (op 6230) Load value




v_mov_b64 v[62:63], s[32:33] // (op 6233) convert

v_mov_b32 v24, v28 // (op 6233) 
buffer_load_ubyte v86, v24, s[60:63], 0 offen offset:12 // (op 6233) Load value
v_add_u32 v24, v24, v62 // (op 6233) 
buffer_load_ubyte v87, v24, s[60:63], 0 offen offset:12 // (op 6233) Load value






buffer_load_dwordx4 v[88:91], v22, s[48:51], 0 offen offset:192 // (op 6236) Load value




buffer_load_ubyte v22, v26, s[56:59], 0 offen offset:12 // (op 6239) Load value

v_and_b32 v24, -4, v1 // (op 6995) 
v_sub_u32 v26, v1, v24 // (op 6995) 

v_lshlrev_b32 v24, 5, v26 // (op 6995) 

v_lshrrev_b32 v26, 2, v1 // (op 6995) 
v_lshl_add_u32 v28, v26, 8, v24 // (op 6995) 


v_lshlrev_b32 v24, 2, v28 // (op 6995) 

v_lshrrev_b32 v26, 3, v24 // (op 6995) 




s_waitcnt vmcnt(23)
ds_write_b128 v26, v[34:37] // (op 6241) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v26, v[38:41] offset:64 // (op 6241) Store local data 

v_and_b32 v24, -4, v1 // (op 7111) 
v_sub_u32 v26, v1, v24 // (op 7111) 

v_lshrrev_b32 v24, 2, v1 // (op 7111) 
v_lshl_add_u32 v28, v24, 3, v26 // (op 7111) 





s_waitcnt vmcnt(21)
ds_write_b8 v28, v42 offset:8192 // (op 6242) Store local data 
s_waitcnt vmcnt(20)
ds_write_b8 v28, v43 offset:8196 // (op 6242) Store local data 

v_and_b32 v24, -4, v1 // (op 6937) 
v_sub_u32 v26, v1, v24 // (op 6937) 

v_lshlrev_b32 v24, 5, v26 // (op 6937) 

v_lshrrev_b32 v26, 2, v1 // (op 6937) 
v_lshl_add_u32 v28, v26, 7, v24 // (op 6937) 


v_lshlrev_b32 v24, 2, v28 // (op 6937) 

v_lshrrev_b32 v26, 3, v24 // (op 6937) 




s_waitcnt vmcnt(19)
ds_write_b128 v26, v[30:33] offset:8704 // (op 6243) Store local data 

v_and_b32 v24, -4, v1 // (op 7053) 
v_sub_u32 v26, v1, v24 // (op 7053) 

v_and_b32 v24, 4294967292, v1 // (op 7053) 
v_add_u32 v28, v24, v26 // (op 7053) 





s_waitcnt vmcnt(18)
ds_write_b8 v28, v44 offset:12800 // (op 6244) Store local data 

v_and_b32 v24, -4, v1 // (op 6908) 
v_sub_u32 v26, v1, v24 // (op 6908) 

v_lshlrev_b32 v24, 5, v26 // (op 6908) 

v_lshrrev_b32 v26, 2, v1 // (op 6908) 
v_lshl_add_u32 v28, v26, 7, v24 // (op 6908) 


v_lshlrev_b32 v24, 2, v28 // (op 6908) 

v_lshrrev_b32 v26, 3, v24 // (op 6908) 

v_and_b32 v24, -4, v1 // (op 6966) 
v_sub_u32 v28, v1, v24 // (op 6966) 

v_lshlrev_b32 v24, 5, v28 // (op 6966) 

v_lshrrev_b32 v28, 2, v1 // (op 6966) 
v_lshl_add_u32 v47, v28, 8, v24 // (op 6966) 


v_lshlrev_b32 v24, 2, v47 // (op 6966) 

v_lshrrev_b32 v28, 3, v24 // (op 6966) 

v_and_b32 v24, -4, v1 // (op 7024) 
v_sub_u32 v47, v1, v24 // (op 7024) 

v_and_b32 v24, 4294967292, v1 // (op 7024) 
v_add_u32 v62, v24, v47 // (op 7024) 


v_and_b32 v24, -4, v1 // (op 7082) 
v_sub_u32 v47, v1, v24 // (op 7082) 

v_lshrrev_b32 v24, 2, v1 // (op 7082) 
v_lshl_add_u32 v63, v24, 3, v47 // (op 7082) 


v_and_b32 v24, -4, v1 // (op 7163) 
v_sub_u32 v47, v1, v24 // (op 7163) 

v_lshlrev_b32 v24, 5, v47 // (op 7163) 

v_lshrrev_b32 v47, 2, v1 // (op 7163) 
v_lshl_add_u32 v92, v47, 7, v24 // (op 7163) 


v_lshlrev_b32 v24, 2, v92 // (op 7163) 

v_lshrrev_b32 v47, 3, v24 // (op 7163) 

v_and_b32 v24, -4, v1 // (op 7192) 
v_sub_u32 v92, v1, v24 // (op 7192) 

v_lshlrev_b32 v24, 5, v92 // (op 7192) 

v_lshrrev_b32 v92, 2, v1 // (op 7192) 
v_lshl_add_u32 v93, v92, 8, v24 // (op 7192) 


v_lshlrev_b32 v24, 2, v93 // (op 7192) 

v_lshrrev_b32 v92, 3, v24 // (op 7192) 

v_and_b32 v24, -4, v1 // (op 7221) 
v_sub_u32 v93, v1, v24 // (op 7221) 

v_and_b32 v24, 4294967292, v1 // (op 7221) 
v_add_u32 v94, v24, v93 // (op 7221) 


v_and_b32 v24, -4, v1 // (op 7250) 
v_sub_u32 v93, v1, v24 // (op 7250) 

v_lshrrev_b32 v24, 2, v1 // (op 7250) 
v_lshl_add_u32 v95, v24, 3, v93 // (op 7250) 


v_and_b32 v24, -4, v1 // (op 7279) 
v_sub_u32 v93, v1, v24 // (op 7279) 

v_lshlrev_b32 v24, 5, v93 // (op 7279) 

v_lshrrev_b32 v93, 2, v1 // (op 7279) 
v_lshl_add_u32 v96, v93, 7, v24 // (op 7279) 


v_lshlrev_b32 v24, 2, v96 // (op 7279) 

v_lshrrev_b32 v93, 3, v24 // (op 7279) 

v_and_b32 v24, -4, v1 // (op 7308) 
v_sub_u32 v96, v1, v24 // (op 7308) 

v_lshlrev_b32 v24, 5, v96 // (op 7308) 

v_lshrrev_b32 v96, 2, v1 // (op 7308) 
v_lshl_add_u32 v97, v96, 8, v24 // (op 7308) 


v_lshlrev_b32 v24, 2, v97 // (op 7308) 

v_lshrrev_b32 v96, 3, v24 // (op 7308) 

v_and_b32 v24, -4, v1 // (op 7337) 
v_sub_u32 v97, v1, v24 // (op 7337) 

v_and_b32 v24, 4294967292, v1 // (op 7337) 
v_add_u32 v98, v24, v97 // (op 7337) 


v_and_b32 v24, -4, v1 // (op 7366) 
v_sub_u32 v97, v1, v24 // (op 7366) 

v_lshrrev_b32 v24, 2, v1 // (op 7366) 
v_lshl_add_u32 v99, v24, 3, v97 // (op 7366) 


v_and_b32 v24, -4, v1 // (op 7395) 
v_sub_u32 v97, v1, v24 // (op 7395) 

v_lshlrev_b32 v24, 5, v97 // (op 7395) 

v_lshrrev_b32 v97, 2, v1 // (op 7395) 
v_lshl_add_u32 v100, v97, 7, v24 // (op 7395) 


v_lshlrev_b32 v24, 2, v100 // (op 7395) 

v_lshrrev_b32 v97, 3, v24 // (op 7395) 

v_and_b32 v24, -4, v1 // (op 7424) 
v_sub_u32 v100, v1, v24 // (op 7424) 

v_lshlrev_b32 v24, 5, v100 // (op 7424) 

v_lshrrev_b32 v100, 2, v1 // (op 7424) 
v_lshl_add_u32 v101, v100, 8, v24 // (op 7424) 


v_lshlrev_b32 v24, 2, v101 // (op 7424) 

v_lshrrev_b32 v100, 3, v24 // (op 7424) 

v_and_b32 v24, -4, v1 // (op 7453) 
v_sub_u32 v101, v1, v24 // (op 7453) 

v_and_b32 v24, 4294967292, v1 // (op 7453) 
v_add_u32 v102, v24, v101 // (op 7453) 


v_and_b32 v24, -4, v1 // (op 7482) 
v_sub_u32 v101, v1, v24 // (op 7482) 

v_lshrrev_b32 v24, 2, v1 // (op 7482) 
v_lshl_add_u32 v103, v24, 3, v101 // (op 7482) 


s_mov_b32 s0, 0 // (op 1607) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1606) 
s_mov_b32 s1, s6 // (op 1606) convert

s_and_b32 s6, s1, 4294967292 // (op 1606) 

s_cmp_lt_u32 s0, s6 // (op 1606) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_1606 // (op 1606) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_1606 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoop_1606:

s_barrier  // (op 7602) 
s_add_u32 s0, s0, 4 // (op 1608) 



s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[104:105], 3, s[6:7] // (op 10) 


v_mov_b32 v24, v25 // (op 10) 
v_mov_b32 v101, v24 // (op 10) 
ds_read_b128 v[106:109], v3 offset:8704 // (op 70) Load local data 
buffer_load_dwordx4 v[34:37], v101, s[52:55], 0 offen offset:256 // (op 10) Load value
s_barrier  // (op 7615) 
v_add_u32 v101, v101, v104 // (op 10) 



buffer_load_dwordx4 v[38:41], v101, s[52:55], 0 offen offset:256 // (op 10) Load value


ds_read_b128 v[110:113], v4 // (op 75) Load local data 

s_barrier  // (op 7629) 
v_mov_b64 v[104:105], s[32:33] // (op 25) convert




ds_read_u8 v101, v0 offset:12800 // (op 80) Load local data 
v_mov_b32 v114, v29 // (op 25) 
s_barrier  // (op 7643) 
v_mov_b32 v115, v114 // (op 25) 



buffer_load_ubyte v42, v115, s[60:63], 0 offen offset:16 // (op 25) Load value
v_add_u32 v115, v115, v104 // (op 25) 
ds_read_u8 v116, v5 offset:8192 // (op 85) Load local data 
buffer_load_ubyte v43, v115, s[60:63], 0 offen offset:16 // (op 25) Load value







v_mov_b32 v115, v23 // (op 4) 

ds_read_b128 v[118:121], v4 offset:2048 // (op 517) Load local data 
buffer_load_dwordx4 v[30:33], v115, s[48:51], 0 offen offset:256 // (op 4) Load value





ds_read_u8 v117, v5 offset:8320 // (op 521) Load local data 
v_mov_b32 v104, v27 // (op 17) 




ds_read_b128 v[122:125], v4 offset:4096 // (op 815) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[106:109], v[110:113], a[0:3], v101, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 88) 



buffer_load_ubyte v44, v104, s[56:59], 0 offen offset:16 // (op 17) Load value
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[106:109], v[118:121], a[4:7], v101, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 522) 
s_barrier  // (op 7666) 
ds_read_u8 v105, v5 offset:8448 // (op 819) Load local data 






ds_read_b128 v[126:129], v4 offset:6144 // (op 1113) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[106:109], v[122:125], a[8:11], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 820) 



ds_read_u8 v130, v5 offset:8576 // (op 1117) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v92, v[48:51] offset:13056 // (op 253) Store local data 



ds_read_b128 v[132:135], v3 offset:10752 // (op 1640) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[106:109], v[126:129], a[12:15], v101, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1118) 





ds_read_u8 v101, v0 offset:12928 // (op 1643) Load local data 
s_waitcnt vmcnt(22)
ds_write_b128 v92, v[52:55] offset:13120 // (op 253) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[110:113], a[16:19], v101, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1921) 


s_barrier  // (op 7678) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[118:121], a[20:23], v101, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1829) 
s_waitcnt vmcnt(21)
ds_write_b8 v95, v56 offset:21248 // (op 255) Store local data 


s_waitcnt vmcnt(20)
ds_write_b8 v95, v57 offset:21252 // (op 255) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[122:125], a[24:27], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1737) 


s_barrier  // (op 7660) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[126:129], a[28:31], v101, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1645) 
s_waitcnt vmcnt(19)
ds_write_b128 v47, v[58:61] offset:21760 // (op 243) Store local data 




s_barrier  // (op 7672) 



s_waitcnt vmcnt(18)
ds_write_b8 v94, v45 offset:25856 // (op 254) Store local data 

s_lshl_b64 s[6:7], s[24:25], 2 // (op 256) 
v_lshrrev_b64 v[112:113], 3, s[6:7] // (op 256) 


v_mov_b32 v101, v24 // (op 256) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 7657) 
buffer_load_dwordx4 v[48:51], v101, s[52:55], 0 offen offset:320 // (op 256) Load value



v_add_u32 v101, v101, v112 // (op 256) 
ds_read_b128 v[108:111], v7 offset:21760 // (op 245) Load local data 
buffer_load_dwordx4 v[52:55], v101, s[52:55], 0 offen offset:320 // (op 256) Load value



s_barrier  // (op 7663) 
v_mov_b64 v[106:107], s[32:33] // (op 258) convert




v_mov_b32 v101, v114 // (op 258) 
ds_read_b128 v[116:119], v8 offset:13056 // (op 247) Load local data 
buffer_load_ubyte v56, v101, s[60:63], 0 offen offset:20 // (op 258) Load value
v_add_u32 v101, v101, v106 // (op 258) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7669) 
buffer_load_ubyte v57, v101, s[60:63], 0 offen offset:20 // (op 258) Load value








ds_read_u8 v101, v6 offset:25856 // (op 249) Load local data 
buffer_load_dwordx4 v[58:61], v115, s[48:51], 0 offen offset:320 // (op 242) Load value



s_barrier  // (op 7675) 
buffer_load_ubyte v45, v104, s[56:59], 0 offen offset:20 // (op 257) Load value



s_barrier  // (op 7690) 
ds_read_u8 v105, v9 offset:21248 // (op 251) Load local data 






ds_read_b128 v[120:123], v8 offset:15104 // (op 529) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[116:119], a[0:3], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 252) 



ds_read_u8 v106, v9 offset:21376 // (op 533) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v96, v[64:67] offset:26112 // (op 293) Store local data 



ds_read_b128 v[124:127], v8 offset:17152 // (op 827) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[120:123], a[4:7], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 523) 



ds_read_u8 v107, v9 offset:21504 // (op 831) Load local data 
s_waitcnt vmcnt(22)
ds_write_b128 v96, v[68:71] offset:26176 // (op 293) Store local data 



ds_read_b128 v[128:131], v8 offset:19200 // (op 1125) Load local data 
s_barrier  // (op 7702) 






s_barrier  // (op 7803) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[124:127], a[8:11], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 821) 
ds_read_u8 v112, v9 offset:21632 // (op 1129) Load local data 
s_waitcnt vmcnt(21)
ds_write_b8 v99, v72 offset:34304 // (op 295) Store local data 
s_waitcnt vmcnt(20)
ds_write_b8 v99, v73 offset:34308 // (op 295) Store local data 



s_barrier  // (op 7807) 
s_barrier  // (op 7684) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[128:131], a[12:15], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1119) 





ds_read_b128 v[108:111], v7 offset:23808 // (op 1650) Load local data 
s_waitcnt vmcnt(19)
ds_write_b128 v93, v[74:77] offset:13056 // (op 283) Store local data 



s_barrier  // (op 7851) 
s_barrier  // (op 7696) 
ds_read_u8 v101, v6 offset:25984 // (op 1653) Load local data 



s_waitcnt vmcnt(18)
ds_write_b8 v98, v46 offset:17152 // (op 294) Store local data 
s_barrier  // (op 7855) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[116:119], a[16:19], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1922) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[120:123], a[20:23], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1830) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[124:127], a[24:27], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1738) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[128:131], a[28:31], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1646) 





s_barrier  // (op 7681) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 296) 



v_lshrrev_b64 v[112:113], 3, s[6:7] // (op 296) 


v_mov_b32 v101, v24 // (op 296) 
ds_read_b128 v[108:111], v11 offset:13056 // (op 285) Load local data 
buffer_load_dwordx4 v[64:67], v101, s[52:55], 0 offen offset:384 // (op 296) Load value
s_barrier  // (op 7687) 
v_add_u32 v101, v101, v112 // (op 296) 



buffer_load_dwordx4 v[68:71], v101, s[52:55], 0 offen offset:384 // (op 296) Load value


ds_read_b128 v[116:119], v12 offset:26112 // (op 287) Load local data 

v_mov_b64 v[106:107], s[32:33] // (op 298) convert

v_mov_b32 v101, v114 // (op 298) 
buffer_load_ubyte v72, v101, s[60:63], 0 offen offset:24 // (op 298) Load value
v_add_u32 v101, v101, v106 // (op 298) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7693) 
buffer_load_ubyte v73, v101, s[60:63], 0 offen offset:24 // (op 298) Load value








ds_read_u8 v101, v10 offset:17152 // (op 289) Load local data 
buffer_load_dwordx4 v[74:77], v115, s[48:51], 0 offen offset:384 // (op 282) Load value



s_barrier  // (op 7699) 
buffer_load_ubyte v46, v104, s[56:59], 0 offen offset:24 // (op 297) Load value



s_barrier  // (op 7714) 
ds_read_u8 v105, v13 offset:34304 // (op 291) Load local data 






ds_read_b128 v[120:123], v12 offset:28160 // (op 537) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[116:119], a[0:3], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 292) 



ds_read_u8 v106, v13 offset:34432 // (op 541) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v100, v[78:81] offset:17408 // (op 333) Store local data 



ds_read_b128 v[124:127], v12 offset:30208 // (op 835) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[120:123], a[4:7], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 524) 



ds_read_u8 v107, v13 offset:34560 // (op 839) Load local data 
s_waitcnt vmcnt(22)
ds_write_b128 v100, v[82:85] offset:17472 // (op 333) Store local data 



ds_read_b128 v[128:131], v12 offset:32256 // (op 1133) Load local data 
s_barrier  // (op 7726) 






s_barrier  // (op 7811) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[124:127], a[8:11], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 822) 
ds_read_u8 v112, v13 offset:34688 // (op 1137) Load local data 
s_waitcnt vmcnt(21)
ds_write_b8 v103, v86 offset:25600 // (op 335) Store local data 
s_waitcnt vmcnt(20)
ds_write_b8 v103, v87 offset:25604 // (op 335) Store local data 



s_barrier  // (op 7815) 
s_barrier  // (op 7708) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[128:131], a[12:15], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1120) 





ds_read_b128 v[108:111], v11 offset:15104 // (op 1656) Load local data 
s_waitcnt vmcnt(19)
ds_write_b128 v97, v[88:91] offset:26112 // (op 323) Store local data 



s_barrier  // (op 7859) 
s_barrier  // (op 7720) 
ds_read_u8 v101, v10 offset:17280 // (op 1659) Load local data 



s_waitcnt vmcnt(18)
ds_write_b8 v102, v22 offset:30208 // (op 334) Store local data 
s_barrier  // (op 7863) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[116:119], a[16:19], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1923) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[120:123], a[20:23], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1831) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[124:127], a[24:27], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1739) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[128:131], a[28:31], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1647) 





s_barrier  // (op 7705) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 336) 



v_lshrrev_b64 v[112:113], 3, s[6:7] // (op 336) 


v_mov_b32 v101, v24 // (op 336) 
ds_read_b128 v[108:111], v15 offset:26112 // (op 325) Load local data 
buffer_load_dwordx4 v[78:81], v101, s[52:55], 0 offen offset:448 // (op 336) Load value
s_barrier  // (op 7711) 
v_add_u32 v101, v101, v112 // (op 336) 



buffer_load_dwordx4 v[82:85], v101, s[52:55], 0 offen offset:448 // (op 336) Load value


ds_read_b128 v[116:119], v16 offset:17408 // (op 327) Load local data 


v_add_u32 v25, 256, v25 // (op 6729) 
v_mov_b64 v[106:107], s[32:33] // (op 338) convert

v_mov_b32 v24, v114 // (op 338) 
buffer_load_ubyte v86, v24, s[60:63], 0 offen offset:28 // (op 338) Load value
v_add_u32 v24, v24, v106 // (op 338) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7717) 
buffer_load_ubyte v87, v24, s[60:63], 0 offen offset:28 // (op 338) Load value






ds_read_u8 v24, v14 offset:30208 // (op 329) Load local data 



s_barrier  // (op 7723) 
buffer_load_dwordx4 v[88:91], v115, s[48:51], 0 offen offset:448 // (op 322) Load value




v_add_u32 v29, 16, v29 // (op 6857) 



v_add_u32 v23, 256, v23 // (op 6665) 
ds_read_u8 v101, v17 offset:25600 // (op 331) Load local data 
buffer_load_ubyte v22, v104, s[56:59], 0 offen offset:28 // (op 337) Load value




ds_read_b128 v[104:107], v16 offset:19456 // (op 545) Load local data 



s_barrier  // (op 7618) 
v_add_u32 v27, 16, v27 // (op 6793) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[116:119], a[0:3], v24, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 332) 
ds_read_u8 v112, v17 offset:25728 // (op 549) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v28, v[34:37] // (op 13) Store local data 



ds_read_b128 v[120:123], v16 offset:21504 // (op 843) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[104:107], a[4:7], v24, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 525) 



ds_read_u8 v113, v17 offset:25856 // (op 847) Load local data 
s_waitcnt vmcnt(22)
ds_write_b128 v28, v[38:41] offset:64 // (op 13) Store local data 



ds_read_b128 v[124:127], v16 offset:23552 // (op 1141) Load local data 
s_barrier  // (op 7646) 






s_barrier  // (op 7819) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[120:123], a[8:11], v24, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 823) 
ds_read_u8 v114, v17 offset:25984 // (op 1145) Load local data 
s_waitcnt vmcnt(21)
ds_write_b8 v63, v42 offset:8192 // (op 30) Store local data 
s_waitcnt vmcnt(20)
ds_write_b8 v63, v43 offset:8196 // (op 30) Store local data 



s_barrier  // (op 7823) 
s_barrier  // (op 7605) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[124:127], a[12:15], v24, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1121) 





ds_read_b128 v[108:111], v15 offset:28160 // (op 1662) Load local data 
s_waitcnt vmcnt(19)
ds_write_b128 v26, v[30:33] offset:8704 // (op 6) Store local data 



s_barrier  // (op 7867) 
s_barrier  // (op 7632) 
ds_read_u8 v24, v14 offset:30336 // (op 1665) Load local data 



s_waitcnt vmcnt(18)
ds_write_b8 v62, v44 offset:12800 // (op 21) Store local data 
s_barrier  // (op 7871) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[116:119], a[16:19], v24, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1924) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[104:107], a[20:23], v24, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1832) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[120:123], a[24:27], v24, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1740) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[124:127], a[28:31], v24, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1648) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1606) 
s_mov_b32 s1, s6 // (op 1606) convert

s_and_b32 s6, s1, 4294967292 // (op 1606) 

s_cmp_lt_u32 s0, s6 // (op 1606) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoop_1606 // (op 1606) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoop_1606 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_1606:

























v_lshrrev_b32 v47, 2, v1 // (op 6621) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6621) 
v_lshl_add_u32 v23, s16, 6, v47 // (op 6621) 

s_lshr_b32 s6, s0, 2 // (op 6621) 

v_and_b32 v47, -4, v1 // (op 6621) 
v_sub_u32 v24, v1, v47 // (op 6621) 

v_lshlrev_b32 v47, 5, v24 // (op 6621) 

v_lshl_add_u32 v24, s6, 9, v47 // (op 6621) 


v_mov_b32 v63, 0 // (op 6621) convert
v_mov_b32 v62, v24 // (op 6621) convert

v_mul_lo_u32 v25, v23, s13 // (op 6621) most significant: low of low * high
v_mul_hi_u32 v47, v23, s12 // (op 6621) most significant: high of low * low
v_mul_lo_u32 v24, v23, s12 // (op 6621) least significant: low of low * low
v_add_u32 v25, v25, v47 // (op 6621) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v62 // (op 6621) least significant half
v_addc_co_u32 v25, s[0:1], v25, v63, s[0:1] // (op 6621) most significant half



v_lshlrev_b64 v[62:63], 2, v[24:25] // (op 6621) 

v_lshrrev_b64 v[24:25], 3, v[62:63] // (op 6621) 

v_mov_b32 v23, v24 // (op 6621) convert

s_mul_i32 s1, s12, s23 // (op 6623) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6623) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6623) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6623) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6623) most significant: sum
s_add_u32 s1, s1, s7 // (op 6623) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6623) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6623) 

s_mov_b32 s48, s20 // (op 6623) reg expression
s_mov_b32 s49, s21 // (op 6623) reg expression
s_mov_b32 s50, s0 // (op 6623) reg expression
s_mov_b32 s51, 131072 // (op 6623) call()

v_and_b32 v47, -4, v1 // (op 6695) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6695) 
v_sub_u32 v24, v1, v47 // (op 6695) 

s_lshr_b32 s6, s0, 2 // (op 6695) 

v_lshlrev_b32 v47, 5, v24 // (op 6695) 

v_lshl_add_u32 v24, s6, 9, v47 // (op 6695) 


v_mov_b32 v63, 0 // (op 6695) convert
v_mov_b32 v62, v24 // (op 6695) convert

v_lshrrev_b32 v47, 2, v1 // (op 6695) 
v_lshlrev_b32 v24, 1, v47 // (op 6695) 

v_lshl_add_u32 v47, s17, 7, v24 // (op 6695) 

v_mul_lo_u32 v25, v47, s25 // (op 6695) most significant: low of low * high
v_mul_hi_u32 v26, v47, s24 // (op 6695) most significant: high of low * low
v_mul_lo_u32 v24, v47, s24 // (op 6695) least significant: low of low * low
v_add_u32 v25, v25, v26 // (op 6695) most significant: sum


v_add_co_u32 v26, s[0:1], v62, v24 // (op 6695) least significant half
v_addc_co_u32 v27, s[0:1], v63, v25, s[0:1] // (op 6695) most significant half

v_lshlrev_b64 v[26:27], 2, v[26:27] // (op 6695) 


v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 6695) 

v_mov_b32 v47, v24 // (op 6695) convert

s_mul_i32 s1, s24, s9 // (op 6697) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6697) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6697) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6697) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6697) most significant: sum
s_add_u32 s1, s1, s7 // (op 6697) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6697) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6697) 

s_mov_b32 s52, s26 // (op 6697) reg expression
s_mov_b32 s53, s27 // (op 6697) reg expression
s_mov_b32 s54, s0 // (op 6697) reg expression
s_mov_b32 s55, 131072 // (op 6697) call()

v_lshrrev_b32 v24, 2, v1 // (op 6759) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6759) 
v_lshl_add_u32 v25, s16, 6, v24 // (op 6759) 

s_lshr_b32 s6, s0, 2 // (op 6759) 

v_and_b32 v24, -4, v1 // (op 6759) 
v_sub_u32 v26, v1, v24 // (op 6759) 

v_lshl_add_u32 v24, s6, 4, v26 // (op 6759) 


v_mul_lo_u32 v26, v25, s28 // (op 6759) 
v_add_u32 v26, v26, v24 // (op 6759) 


s_mul_i32 s1, s28, s23 // (op 6761) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6761) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6761) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6761) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6761) most significant: sum
s_add_u32 s1, s1, s7 // (op 6761) most significant: sum


s_mov_b32 s56, s30 // (op 6761) reg expression
s_mov_b32 s57, s31 // (op 6761) reg expression
s_mov_b32 s58, s0 // (op 6761) reg expression
s_mov_b32 s59, 131072 // (op 6761) call()

v_and_b32 v24, -4, v1 // (op 6823) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6823) 
v_sub_u32 v25, v1, v24 // (op 6823) 

s_lshr_b32 s6, s0, 2 // (op 6823) 

v_lshrrev_b32 v24, 2, v1 // (op 6823) 
v_lshl_add_u32 v27, s6, 4, v25 // (op 6823) 


v_lshlrev_b32 v25, 1, v24 // (op 6823) 

v_lshl_add_u32 v24, s17, 7, v25 // (op 6823) 

v_mul_lo_u32 v25, v24, s32 // (op 6823) 

v_add_u32 v24, v27, v25 // (op 6823) 


s_mul_i32 s1, s32, s9 // (op 6825) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6825) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6825) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6825) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6825) most significant: sum
s_add_u32 s1, s1, s7 // (op 6825) most significant: sum


s_mov_b32 s60, s34 // (op 6825) reg expression
s_mov_b32 s61, s35 // (op 6825) reg expression
s_mov_b32 s62, s0 // (op 6825) reg expression
s_mov_b32 s63, 131072 // (op 6825) call()

v_and_b32 v25, -4, v1 // (op 6921) 
v_sub_u32 v27, v1, v25 // (op 6921) 

v_lshlrev_b32 v25, 5, v27 // (op 6921) 

v_lshrrev_b32 v27, 2, v1 // (op 6921) 
v_lshl_add_u32 v28, v27, 7, v25 // (op 6921) 


v_lshlrev_b32 v25, 2, v28 // (op 6921) 

v_lshrrev_b32 v27, 3, v25 // (op 6921) 

v_and_b32 v25, -4, v1 // (op 6979) 
v_sub_u32 v28, v1, v25 // (op 6979) 

v_lshlrev_b32 v25, 5, v28 // (op 6979) 

v_lshrrev_b32 v28, 2, v1 // (op 6979) 
v_lshl_add_u32 v29, v28, 8, v25 // (op 6979) 


v_lshlrev_b32 v25, 2, v29 // (op 6979) 

v_lshrrev_b32 v28, 3, v25 // (op 6979) 

v_and_b32 v25, -4, v1 // (op 7037) 
v_sub_u32 v29, v1, v25 // (op 7037) 

v_and_b32 v25, 4294967292, v1 // (op 7037) 
v_add_u32 v30, v25, v29 // (op 7037) 


v_and_b32 v25, -4, v1 // (op 7095) 
v_sub_u32 v29, v1, v25 // (op 7095) 

v_lshrrev_b32 v25, 2, v1 // (op 7095) 
v_lshl_add_u32 v31, v25, 3, v29 // (op 7095) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 1612) 
s_and_b32 s6, s0, 4294967292 // (op 1612) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 1611) 
s_mov_b32 s7, s0 // (op 1611) convert

s_cmp_lt_u32 s6, s7 // (op 1611) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_1611 // (op 1611) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_1611 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_1611:


s_add_u32 s6, s6, 1 // (op 1613) 
v_mov_b32 v22, v23 // (op 176) 

buffer_load_dwordx4 v[32:35], v22, s[48:51], 0 offen // (op 176) Load value

v_add_u32 v23, 64, v23 // (op 6611) 

s_barrier  // (op 7608) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 190) 



v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 190) 


v_mov_b32 v22, v47 // (op 190) 
v_mov_b32 v25, v22 // (op 190) 
buffer_load_dwordx4 v[38:41], v25, s[52:55], 0 offen // (op 190) Load value
v_add_u32 v25, v25, v36 // (op 190) 
buffer_load_dwordx4 v[42:45], v25, s[52:55], 0 offen // (op 190) Load value



v_add_u32 v47, 64, v47 // (op 6687) 
s_waitcnt vmcnt(2)
ds_write_b128 v27, v[32:35] offset:8704 // (op 177) Store local data 



v_mov_b32 v22, v26 // (op 191) 

s_barrier  // (op 7621) 
buffer_load_ubyte v25, v22, s[56:59], 0 offen offset: 0 // (op 191) Load value




v_add_u32 v26, 4, v26 // (op 6751) 
s_waitcnt vmcnt(2)
ds_write_b128 v28, v[38:41] // (op 187) Store local data 

v_mov_b64 v[32:33], s[32:33] // (op 192) convert

v_mov_b32 v22, v24 // (op 192) 
v_mov_b32 v29, v22 // (op 192) 
buffer_load_ubyte v34, v29, s[60:63], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v29, v29, v32 // (op 192) 
buffer_load_ubyte v35, v29, s[60:63], 0 offen offset: 0 // (op 192) Load value



v_add_u32 v24, 4, v24 // (op 6815) 
s_waitcnt vmcnt(3)
ds_write_b128 v28, v[42:45] offset:64 // (op 187) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 7611) 



s_barrier  // (op 7635) 
ds_read_b128 v[36:39], v3 offset:8704 // (op 179) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 7625) 
s_waitcnt vmcnt(2)
ds_write_b8 v30, v25 offset:12800 // (op 188) Store local data 




ds_read_b128 v[40:43], v4 // (op 181) Load local data 
s_barrier  // (op 7649) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 7639) 






ds_read_u8 v22, v0 offset:12800 // (op 183) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v31, v34 offset:8192 // (op 189) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v31, v35 offset:8196 // (op 189) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 7653) 



ds_read_u8 v25, v5 offset:8192 // (op 185) Load local data 



ds_read_b128 v[32:35], v4 offset:2048 // (op 496) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:39], v[40:43], a[0:3], v22, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 186) 



ds_read_u8 v29, v5 offset:8320 // (op 500) Load local data 



ds_read_b128 v[48:51], v4 offset:4096 // (op 794) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:39], v[32:35], a[4:7], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 501) 



ds_read_u8 v44, v5 offset:8448 // (op 798) Load local data 



ds_read_b128 v[52:55], v4 offset:6144 // (op 1092) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:39], v[48:51], a[8:11], v22, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 799) 



ds_read_u8 v45, v5 offset:8576 // (op 1096) Load local data 



ds_read_b128 v[56:59], v3 offset:10752 // (op 1524) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:39], v[52:55], a[12:15], v22, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1097) 





ds_read_u8 v22, v0 offset:12928 // (op 1527) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[40:43], a[16:19], v22, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1529) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[32:35], a[20:23], v22, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1560) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v22, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1591) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[52:55], a[28:31], v22, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1622) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 1611) 
s_mov_b32 s7, s0 // (op 1611) convert

s_cmp_lt_u32 s6, s7 // (op 1611) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_1611 // (op 1611) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_1611 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_1611:























s_barrier  // (op 7835) 

















v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_133
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_mul_f32 v10, v19, v14 // (op 50) 
v_mul_f32 v11, v19, v15 // (op 50) 
v_mul_f32 v12, v19, v16 // (op 50) 
v_mul_f32 v13, v19, v17 // (op 50) 



v_mov_b32 v0, v21 // (op 56) reg expression

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 56) Store value


v_accvgpr_read v14, a4 // (op 3045) 
v_accvgpr_read v15, a5 // (op 3045) 
v_accvgpr_read v16, a6 // (op 3045) 
v_accvgpr_read v17, a7 // (op 3045) 
v_mul_f32 v10, v19, v14 // (op 3045) 
v_mul_f32 v11, v19, v15 // (op 3045) 
v_mul_f32 v12, v19, v16 // (op 3045) 
v_mul_f32 v13, v19, v17 // (op 3045) 



s_lshl_b32 s0, s4, 7 // (op 3047) 
v_add_u32 v0, v21, s0 // (op 3047) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3047) Store value


v_accvgpr_read v14, a8 // (op 3057) 
v_accvgpr_read v15, a9 // (op 3057) 
v_accvgpr_read v16, a10 // (op 3057) 
v_accvgpr_read v17, a11 // (op 3057) 
v_mul_f32 v10, v19, v14 // (op 3057) 
v_mul_f32 v11, v19, v15 // (op 3057) 
v_mul_f32 v12, v19, v16 // (op 3057) 
v_mul_f32 v13, v19, v17 // (op 3057) 



s_lshl_b32 s0, s4, 8 // (op 3059) 
v_add_u32 v0, v21, s0 // (op 3059) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3059) Store value


v_accvgpr_read v14, a12 // (op 3069) 
v_accvgpr_read v15, a13 // (op 3069) 
v_accvgpr_read v16, a14 // (op 3069) 
v_accvgpr_read v17, a15 // (op 3069) 
v_mul_f32 v10, v19, v14 // (op 3069) 
v_mul_f32 v11, v19, v15 // (op 3069) 
v_mul_f32 v12, v19, v16 // (op 3069) 
v_mul_f32 v13, v19, v17 // (op 3069) 



s_lshl_b32 s0, s4, 7 // (op 3071) 
s_mul_i32 s1, 3, s0 // (op 3071) 

v_add_u32 v0, v21, s1 // (op 3071) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3071) Store value


v_accvgpr_read v14, a16 // (op 3103) 
v_accvgpr_read v15, a17 // (op 3103) 
v_accvgpr_read v16, a18 // (op 3103) 
v_accvgpr_read v17, a19 // (op 3103) 
v_mul_f32 v10, v19, v14 // (op 3103) 
v_mul_f32 v11, v19, v15 // (op 3103) 
v_mul_f32 v12, v19, v16 // (op 3103) 
v_mul_f32 v13, v19, v17 // (op 3103) 



v_add_u32 v0, 128, v21 // (op 3106) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3106) Store value


v_accvgpr_read v14, a20 // (op 3116) 
v_accvgpr_read v15, a21 // (op 3116) 
v_accvgpr_read v16, a22 // (op 3116) 
v_accvgpr_read v17, a23 // (op 3116) 
v_mul_f32 v10, v19, v14 // (op 3116) 
v_mul_f32 v11, v19, v15 // (op 3116) 
v_mul_f32 v12, v19, v16 // (op 3116) 
v_mul_f32 v13, v19, v17 // (op 3116) 



s_lshl_b32 s0, s4, 7 // (op 3108) 
s_add_u32 s0, s0, 128 // (op 3108) 
v_add_u32 v0, v21, s0 // (op 3108) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3108) Store value


v_accvgpr_read v14, a24 // (op 3121) 
v_accvgpr_read v15, a25 // (op 3121) 
v_accvgpr_read v16, a26 // (op 3121) 
v_accvgpr_read v17, a27 // (op 3121) 
v_mul_f32 v10, v19, v14 // (op 3121) 
v_mul_f32 v11, v19, v15 // (op 3121) 
v_mul_f32 v12, v19, v16 // (op 3121) 
v_mul_f32 v13, v19, v17 // (op 3121) 



s_lshl_b32 s0, s4, 8 // (op 3110) 
s_add_u32 s0, s0, 128 // (op 3110) 
v_add_u32 v0, v21, s0 // (op 3110) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3110) Store value


v_accvgpr_read v14, a28 // (op 3126) 
v_accvgpr_read v15, a29 // (op 3126) 
v_accvgpr_read v16, a30 // (op 3126) 
v_accvgpr_read v17, a31 // (op 3126) 
v_mul_f32 v10, v19, v14 // (op 3126) 
v_mul_f32 v11, v19, v15 // (op 3126) 
v_mul_f32 v12, v19, v16 // (op 3126) 
v_mul_f32 v13, v19, v17 // (op 3126) 



s_lshl_b32 s0, s4, 7 // (op 3112) 
s_mul_i32 s1, 3, s0 // (op 3112) 
s_add_u32 s1, s1, 128 // (op 3112) 

v_add_u32 v0, v21, s1 // (op 3112) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3112) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v20 // (op 3284) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3284) Load value

v_accvgpr_read v10, a16 // (op 3272) 
v_accvgpr_read v11, a17 // (op 3272) 
v_accvgpr_read v12, a18 // (op 3272) 
v_accvgpr_read v13, a19 // (op 3272) 
v_mul_f32 v6, v19, v10 // (op 3272) 
v_mul_f32 v7, v19, v11 // (op 3272) 
v_mul_f32 v8, v19, v12 // (op 3272) 
v_mul_f32 v9, v19, v13 // (op 3272) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3283) 
v_mul_f32 v11, v18, v15 // (op 3283) 
v_mul_f32 v12, v18, v16 // (op 3283) 
v_mul_f32 v13, v18, v17 // (op 3283) 

v_add_f32 v14, v10, v6 // (op 3273) 
v_add_f32 v15, v11, v7 // (op 3273) 
v_add_f32 v16, v12, v8 // (op 3273) 
v_add_f32 v17, v13, v9 // (op 3273) 




v_add_u32 v0, 128, v21 // (op 3275) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3275) Store value




s_lshl_b32 s0, s2, 7 // (op 3288) 
s_add_u32 s0, s0, 128 // (op 3288) 
v_add_u32 v0, v20, s0 // (op 3288) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3288) Load value

v_accvgpr_read v10, a20 // (op 3298) 
v_accvgpr_read v11, a21 // (op 3298) 
v_accvgpr_read v12, a22 // (op 3298) 
v_accvgpr_read v13, a23 // (op 3298) 
v_mul_f32 v6, v19, v10 // (op 3298) 
v_mul_f32 v7, v19, v11 // (op 3298) 
v_mul_f32 v8, v19, v12 // (op 3298) 
v_mul_f32 v9, v19, v13 // (op 3298) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3286) 
v_mul_f32 v11, v18, v15 // (op 3286) 
v_mul_f32 v12, v18, v16 // (op 3286) 
v_mul_f32 v13, v18, v17 // (op 3286) 

v_add_f32 v14, v10, v6 // (op 3287) 
v_add_f32 v15, v11, v7 // (op 3287) 
v_add_f32 v16, v12, v8 // (op 3287) 
v_add_f32 v17, v13, v9 // (op 3287) 




s_lshl_b32 s0, s4, 7 // (op 3277) 
s_add_u32 s0, s0, 128 // (op 3277) 
v_add_u32 v0, v21, s0 // (op 3277) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3277) Store value




s_lshl_b32 s0, s2, 8 // (op 3292) 
s_add_u32 s0, s0, 128 // (op 3292) 
v_add_u32 v0, v20, s0 // (op 3292) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3292) Load value

v_accvgpr_read v10, a24 // (op 3300) 
v_accvgpr_read v11, a25 // (op 3300) 
v_accvgpr_read v12, a26 // (op 3300) 
v_accvgpr_read v13, a27 // (op 3300) 
v_mul_f32 v6, v19, v10 // (op 3300) 
v_mul_f32 v7, v19, v11 // (op 3300) 
v_mul_f32 v8, v19, v12 // (op 3300) 
v_mul_f32 v9, v19, v13 // (op 3300) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3290) 
v_mul_f32 v11, v18, v15 // (op 3290) 
v_mul_f32 v12, v18, v16 // (op 3290) 
v_mul_f32 v13, v18, v17 // (op 3290) 

v_add_f32 v14, v10, v6 // (op 3291) 
v_add_f32 v15, v11, v7 // (op 3291) 
v_add_f32 v16, v12, v8 // (op 3291) 
v_add_f32 v17, v13, v9 // (op 3291) 




s_lshl_b32 s0, s4, 8 // (op 3279) 
s_add_u32 s0, s0, 128 // (op 3279) 
v_add_u32 v0, v21, s0 // (op 3279) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3279) Store value




s_lshl_b32 s0, s2, 7 // (op 3296) 
s_mul_i32 s1, 3, s0 // (op 3296) 
s_add_u32 s1, s1, 128 // (op 3296) 

v_add_u32 v0, v20, s1 // (op 3296) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3296) Load value

v_accvgpr_read v10, a28 // (op 3302) 
v_accvgpr_read v11, a29 // (op 3302) 
v_accvgpr_read v12, a30 // (op 3302) 
v_accvgpr_read v13, a31 // (op 3302) 
v_mul_f32 v6, v19, v10 // (op 3302) 
v_mul_f32 v7, v19, v11 // (op 3302) 
v_mul_f32 v8, v19, v12 // (op 3302) 
v_mul_f32 v9, v19, v13 // (op 3302) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3294) 
v_mul_f32 v11, v18, v15 // (op 3294) 
v_mul_f32 v12, v18, v16 // (op 3294) 
v_mul_f32 v13, v18, v17 // (op 3294) 

v_add_f32 v14, v10, v6 // (op 3295) 
v_add_f32 v15, v11, v7 // (op 3295) 
v_add_f32 v16, v12, v8 // (op 3295) 
v_add_f32 v17, v13, v9 // (op 3295) 




s_lshl_b32 s0, s4, 7 // (op 3281) 
s_mul_i32 s1, 3, s0 // (op 3281) 
s_add_u32 s1, s1, 128 // (op 3281) 

v_add_u32 v0, v21, s1 // (op 3281) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3281) Store value




v_mov_b32 v0, v20 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 145) Load value

v_accvgpr_read v10, a0 // (op 142) 
v_accvgpr_read v11, a1 // (op 142) 
v_accvgpr_read v12, a2 // (op 142) 
v_accvgpr_read v13, a3 // (op 142) 
v_mul_f32 v6, v19, v10 // (op 142) 
v_mul_f32 v7, v19, v11 // (op 142) 
v_mul_f32 v8, v19, v12 // (op 142) 
v_mul_f32 v9, v19, v13 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 146) 
v_mul_f32 v11, v18, v15 // (op 146) 
v_mul_f32 v12, v18, v16 // (op 146) 
v_mul_f32 v13, v18, v17 // (op 146) 

v_add_f32 v14, v10, v6 // (op 143) 
v_add_f32 v15, v11, v7 // (op 143) 
v_add_f32 v16, v12, v8 // (op 143) 
v_add_f32 v17, v13, v9 // (op 143) 




v_mov_b32 v0, v21 // (op 144) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 3198) 
v_add_u32 v0, v20, s0 // (op 3198) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3198) Load value

v_accvgpr_read v10, a4 // (op 3195) 
v_accvgpr_read v11, a5 // (op 3195) 
v_accvgpr_read v12, a6 // (op 3195) 
v_accvgpr_read v13, a7 // (op 3195) 
v_mul_f32 v6, v19, v10 // (op 3195) 
v_mul_f32 v7, v19, v11 // (op 3195) 
v_mul_f32 v8, v19, v12 // (op 3195) 
v_mul_f32 v9, v19, v13 // (op 3195) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3199) 
v_mul_f32 v11, v18, v15 // (op 3199) 
v_mul_f32 v12, v18, v16 // (op 3199) 
v_mul_f32 v13, v18, v17 // (op 3199) 

v_add_f32 v14, v10, v6 // (op 3196) 
v_add_f32 v15, v11, v7 // (op 3196) 
v_add_f32 v16, v12, v8 // (op 3196) 
v_add_f32 v17, v13, v9 // (op 3196) 




s_lshl_b32 s0, s4, 7 // (op 3197) 
v_add_u32 v0, v21, s0 // (op 3197) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3197) Store value




s_lshl_b32 s0, s2, 8 // (op 3210) 
v_add_u32 v0, v20, s0 // (op 3210) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3210) Load value

v_accvgpr_read v10, a8 // (op 3207) 
v_accvgpr_read v11, a9 // (op 3207) 
v_accvgpr_read v12, a10 // (op 3207) 
v_accvgpr_read v13, a11 // (op 3207) 
v_mul_f32 v6, v19, v10 // (op 3207) 
v_mul_f32 v7, v19, v11 // (op 3207) 
v_mul_f32 v8, v19, v12 // (op 3207) 
v_mul_f32 v9, v19, v13 // (op 3207) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3211) 
v_mul_f32 v11, v18, v15 // (op 3211) 
v_mul_f32 v12, v18, v16 // (op 3211) 
v_mul_f32 v13, v18, v17 // (op 3211) 

v_add_f32 v14, v10, v6 // (op 3208) 
v_add_f32 v15, v11, v7 // (op 3208) 
v_add_f32 v16, v12, v8 // (op 3208) 
v_add_f32 v17, v13, v9 // (op 3208) 




s_lshl_b32 s0, s4, 8 // (op 3209) 
v_add_u32 v0, v21, s0 // (op 3209) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3209) Store value




s_lshl_b32 s0, s2, 7 // (op 3222) 
s_mul_i32 s1, 3, s0 // (op 3222) 

v_add_u32 v0, v20, s1 // (op 3222) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3222) Load value

v_accvgpr_read v10, a12 // (op 3219) 
v_accvgpr_read v11, a13 // (op 3219) 
v_accvgpr_read v12, a14 // (op 3219) 
v_accvgpr_read v13, a15 // (op 3219) 
v_mul_f32 v6, v19, v10 // (op 3219) 
v_mul_f32 v7, v19, v11 // (op 3219) 
v_mul_f32 v8, v19, v12 // (op 3219) 
v_mul_f32 v9, v19, v13 // (op 3219) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3223) 
v_mul_f32 v11, v18, v15 // (op 3223) 
v_mul_f32 v12, v18, v16 // (op 3223) 
v_mul_f32 v13, v18, v17 // (op 3223) 

v_add_f32 v14, v10, v6 // (op 3220) 
v_add_f32 v15, v11, v7 // (op 3220) 
v_add_f32 v16, v12, v8 // (op 3220) 
v_add_f32 v17, v13, v9 // (op 3220) 




s_lshl_b32 s0, s4, 7 // (op 3221) 
s_mul_i32 s1, 3, s0 // (op 3221) 

v_add_u32 v0, v21, s1 // (op 3221) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3221) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
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
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_4_LDSSYNC.kd
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

