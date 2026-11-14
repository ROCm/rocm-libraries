// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 6979) 
v_and_b32 v3, 15, v1 // (op 6979) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 6979) 


v_and_b32 v0, 63, v1 // (op 6979) 
v_lshrrev_b32 v3, 4, v0 // (op 6979) 

v_lshlrev_b32 v0, 5, v3 // (op 6979) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 6979) 


v_lshlrev_b32 v0, 2, v3 // (op 6979) 

v_lshrrev_b32 v3, 3, v0 // (op 6979) 

v_lshrrev_b32 v0, 6, v1 // (op 7037) 
v_and_b32 v4, 1, v0 // (op 7037) 

v_and_b32 v0, 15, v1 // (op 7037) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 7037) 


v_and_b32 v0, 63, v1 // (op 7037) 
v_lshrrev_b32 v4, 4, v0 // (op 7037) 

v_lshlrev_b32 v0, 5, v4 // (op 7037) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 7037) 


v_lshlrev_b32 v0, 2, v4 // (op 7037) 

v_lshrrev_b32 v4, 3, v0 // (op 7037) 

v_lshrrev_b32 v0, 7, v1 // (op 7095) 
v_and_b32 v5, 15, v1 // (op 7095) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 7095) 


v_and_b32 v0, 63, v1 // (op 7095) 
v_lshrrev_b32 v5, 4, v0 // (op 7095) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 7095) 


v_lshrrev_b32 v5, 6, v1 // (op 7153) 
v_and_b32 v6, 1, v5 // (op 7153) 

v_and_b32 v5, 15, v1 // (op 7153) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 7153) 


v_and_b32 v5, 63, v1 // (op 7153) 
v_lshrrev_b32 v6, 4, v5 // (op 7153) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 7153) 


v_lshrrev_b32 v6, 7, v1 // (op 7205) 
v_and_b32 v7, 15, v1 // (op 7205) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 7205) 


v_and_b32 v6, 63, v1 // (op 7205) 
v_lshrrev_b32 v7, 4, v6 // (op 7205) 

v_lshlrev_b32 v6, 5, v7 // (op 7205) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 7205) 


v_lshlrev_b32 v6, 2, v7 // (op 7205) 

v_lshrrev_b32 v7, 3, v6 // (op 7205) 

v_lshrrev_b32 v6, 6, v1 // (op 7234) 
v_and_b32 v8, 1, v6 // (op 7234) 

v_and_b32 v6, 15, v1 // (op 7234) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 7234) 


v_and_b32 v6, 63, v1 // (op 7234) 
v_lshrrev_b32 v8, 4, v6 // (op 7234) 

v_lshlrev_b32 v6, 5, v8 // (op 7234) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 7234) 


v_lshlrev_b32 v6, 2, v8 // (op 7234) 

v_lshrrev_b32 v8, 3, v6 // (op 7234) 

v_lshrrev_b32 v6, 7, v1 // (op 7263) 
v_and_b32 v9, 15, v1 // (op 7263) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 7263) 


v_and_b32 v6, 63, v1 // (op 7263) 
v_lshrrev_b32 v9, 4, v6 // (op 7263) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 7263) 


v_lshrrev_b32 v9, 6, v1 // (op 7292) 
v_and_b32 v10, 1, v9 // (op 7292) 

v_and_b32 v9, 15, v1 // (op 7292) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 7292) 


v_and_b32 v9, 63, v1 // (op 7292) 
v_lshrrev_b32 v10, 4, v9 // (op 7292) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 7292) 


v_lshrrev_b32 v10, 7, v1 // (op 7321) 
v_and_b32 v11, 15, v1 // (op 7321) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 7321) 


v_and_b32 v10, 63, v1 // (op 7321) 
v_lshrrev_b32 v11, 4, v10 // (op 7321) 

v_lshlrev_b32 v10, 5, v11 // (op 7321) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 7321) 


v_lshlrev_b32 v10, 2, v11 // (op 7321) 

v_lshrrev_b32 v11, 3, v10 // (op 7321) 

v_lshrrev_b32 v10, 6, v1 // (op 7350) 
v_and_b32 v12, 1, v10 // (op 7350) 

v_and_b32 v10, 15, v1 // (op 7350) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 7350) 


v_and_b32 v10, 63, v1 // (op 7350) 
v_lshrrev_b32 v12, 4, v10 // (op 7350) 

v_lshlrev_b32 v10, 5, v12 // (op 7350) 

v_lshl_add_u32 v12, v13, 7, v10 // (op 7350) 


v_lshlrev_b32 v10, 2, v12 // (op 7350) 

v_lshrrev_b32 v12, 3, v10 // (op 7350) 

v_lshrrev_b32 v10, 7, v1 // (op 7379) 
v_and_b32 v13, 15, v1 // (op 7379) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 7379) 


v_and_b32 v10, 63, v1 // (op 7379) 
v_lshrrev_b32 v13, 4, v10 // (op 7379) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 7379) 


v_lshrrev_b32 v13, 6, v1 // (op 7408) 
v_and_b32 v14, 1, v13 // (op 7408) 

v_and_b32 v13, 15, v1 // (op 7408) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 7408) 


v_and_b32 v13, 63, v1 // (op 7408) 
v_lshrrev_b32 v14, 4, v13 // (op 7408) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 7408) 


v_lshrrev_b32 v14, 7, v1 // (op 7437) 
v_and_b32 v15, 15, v1 // (op 7437) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 7437) 


v_and_b32 v14, 63, v1 // (op 7437) 
v_lshrrev_b32 v15, 4, v14 // (op 7437) 

v_lshlrev_b32 v14, 5, v15 // (op 7437) 

v_lshl_add_u32 v15, v16, 7, v14 // (op 7437) 


v_lshlrev_b32 v14, 2, v15 // (op 7437) 

v_lshrrev_b32 v15, 3, v14 // (op 7437) 

v_lshrrev_b32 v14, 6, v1 // (op 7466) 
v_and_b32 v16, 1, v14 // (op 7466) 

v_and_b32 v14, 15, v1 // (op 7466) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 7466) 


v_and_b32 v14, 63, v1 // (op 7466) 
v_lshrrev_b32 v16, 4, v14 // (op 7466) 

v_lshlrev_b32 v14, 5, v16 // (op 7466) 

v_lshl_add_u32 v16, v17, 7, v14 // (op 7466) 


v_lshlrev_b32 v14, 2, v16 // (op 7466) 

v_lshrrev_b32 v16, 3, v14 // (op 7466) 

v_lshrrev_b32 v14, 7, v1 // (op 7495) 
v_and_b32 v17, 15, v1 // (op 7495) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 7495) 


v_and_b32 v14, 63, v1 // (op 7495) 
v_lshrrev_b32 v17, 4, v14 // (op 7495) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 7495) 


v_lshrrev_b32 v17, 6, v1 // (op 7524) 
v_and_b32 v18, 1, v17 // (op 7524) 

v_and_b32 v17, 15, v1 // (op 7524) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 7524) 


v_and_b32 v17, 63, v1 // (op 7524) 
v_lshrrev_b32 v18, 4, v17 // (op 7524) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 7524) 


v_lshrrev_b32 v18, 7, v1 // (op 6917) 
v_and_b32 v19, 63, v1 // (op 6917) 
v_lshrrev_b32 v20, 4, v19 // (op 6917) 
v_lshlrev_b32 v21, 2, v20 // (op 6917) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 6917) 


v_lshl_add_u32 v18, s16, 7, v20 // (op 6917) 

v_lshrrev_b32 v20, 2, v19 // (op 6917) 

v_and_b32 v19, 3, v20 // (op 6917) 

v_and_b32 v20, 3, v1 // (op 6917) 
v_lshl_add_u32 v21, v19, 2, v20 // (op 6917) 


v_lshrrev_b32 v19, 6, v1 // (op 6917) 
v_and_b32 v20, 1, v19 // (op 6917) 

v_lshl_add_u32 v19, v20, 4, v21 // (op 6917) 


v_lshl_add_u32 v20, s17, 6, v19 // (op 6917) 

v_mul_lo_u32 v19, v20, s2 // (op 6917) 

v_add_lshl_u32 v20, v18, v19, 2 // (op 6917) 


v_lshrrev_b32 v18, 7, v1 // (op 7174) 
v_and_b32 v19, 63, v1 // (op 7174) 
v_lshrrev_b32 v21, 4, v19 // (op 7174) 
s_mul_i32 s1, s2, s9 // (op 6919) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 6919) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 6919) least significant
s_mul_hi_u32 s19, s2, s8 // (op 6919) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6919) most significant: sum
s_add_u32 s1, s1, s19 // (op 6919) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 6919) 

v_lshlrev_b32 v22, 2, v21 // (op 7174) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 7174) 


v_lshl_add_u32 v18, s16, 7, v21 // (op 7174) 

v_lshrrev_b32 v21, 2, v19 // (op 7174) 

v_and_b32 v19, 3, v21 // (op 7174) 

v_and_b32 v21, 3, v1 // (op 7174) 
v_lshl_add_u32 v22, v19, 2, v21 // (op 7174) 


v_lshrrev_b32 v19, 6, v1 // (op 7174) 
v_and_b32 v21, 1, v19 // (op 7174) 

v_lshl_add_u32 v19, v21, 4, v22 // (op 7174) 


v_lshl_add_u32 v21, s17, 6, v19 // (op 7174) 

v_mul_lo_u32 v19, v21, s4 // (op 7174) 

v_add_lshl_u32 v21, v18, v19, 2 // (op 7174) 


s_mov_b32 s40, s6 // (op 6919) reg expression
s_mov_b32 s41, s7 // (op 6919) reg expression
s_mov_b32 s42, s18 // (op 6919) reg expression
s_mov_b32 s43, 131072 // (op 6919) call()

s_mul_i32 s1, s4, s9 // (op 7176) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 7176) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 7176) least significant
s_mul_hi_u32 s7, s4, s8 // (op 7176) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7176) most significant: sum
s_add_u32 s1, s1, s7 // (op 7176) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 7176) 

s_mov_b32 s44, s10 // (op 7176) reg expression
s_mov_b32 s45, s11 // (op 7176) reg expression
s_mov_b32 s46, s6 // (op 7176) reg expression
s_mov_b32 s47, 131072 // (op 7176) call()


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
v_accvgpr_write a8, 0.00000 // (op 848) call()
v_accvgpr_write a9, 0.00000 // (op 848) call()
v_accvgpr_write a10, 0.00000 // (op 848) call()
v_accvgpr_write a11, 0.00000 // (op 848) call()
v_accvgpr_write a12, 0.00000 // (op 1095) call()
v_accvgpr_write a13, 0.00000 // (op 1095) call()
v_accvgpr_write a14, 0.00000 // (op 1095) call()
v_accvgpr_write a15, 0.00000 // (op 1095) call()
v_accvgpr_write a16, 0.00000 // (op 1494) call()
v_accvgpr_write a17, 0.00000 // (op 1494) call()
v_accvgpr_write a18, 0.00000 // (op 1494) call()
v_accvgpr_write a19, 0.00000 // (op 1494) call()
v_accvgpr_write a20, 0.00000 // (op 1741) call()
v_accvgpr_write a21, 0.00000 // (op 1741) call()
v_accvgpr_write a22, 0.00000 // (op 1741) call()
v_accvgpr_write a23, 0.00000 // (op 1741) call()
v_accvgpr_write a24, 0.00000 // (op 2140) call()
v_accvgpr_write a25, 0.00000 // (op 2140) call()
v_accvgpr_write a26, 0.00000 // (op 2140) call()
v_accvgpr_write a27, 0.00000 // (op 2140) call()
v_accvgpr_write a28, 0.00000 // (op 2387) call()
v_accvgpr_write a29, 0.00000 // (op 2387) call()
v_accvgpr_write a30, 0.00000 // (op 2387) call()
v_accvgpr_write a31, 0.00000 // (op 2387) call()
v_lshrrev_b32 v22, 2, v1 // (op 6679) 
v_lshlrev_b32 v23, 1, v22 // (op 6679) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 6679) 

v_and_b32 v23, -4, v1 // (op 6679) 
v_sub_u32 v24, v1, v23 // (op 6679) 

v_lshlrev_b32 v23, 5, v24 // (op 6679) 

v_mov_b32 v25, 0 // (op 6679) convert
v_mov_b32 v24, v23 // (op 6679) convert

v_mul_lo_u32 v27, v22, s13 // (op 6679) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 6679) most significant: high of low * low
v_mul_lo_u32 v26, v22, s12 // (op 6679) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 6679) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v24 // (op 6679) least significant half
v_addc_co_u32 v27, s[0:1], v27, v25, s[0:1] // (op 6679) most significant half



v_lshlrev_b64 v[22:23], 2, v[26:27] // (op 6679) 

v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 6679) 

v_mov_b32 v22, v24 // (op 6679) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 6681) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6681) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6681) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6681) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6681) most significant: sum
s_add_u32 s1, s1, s7 // (op 6681) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6681) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6681) 

s_mov_b32 s48, s20 // (op 6681) reg expression
s_mov_b32 s49, s21 // (op 6681) reg expression
s_mov_b32 s50, s0 // (op 6681) reg expression
s_mov_b32 s51, 131072 // (op 6681) call()

v_lshrrev_b32 v23, 2, v1 // (op 6698) 
v_lshlrev_b32 v24, 1, v23 // (op 6698) 

v_lshl_add_u32 v23, s16, 7, v24 // (op 6698) 

v_and_b32 v24, -4, v1 // (op 6698) 
v_sub_u32 v25, v1, v24 // (op 6698) 

v_lshlrev_b32 v24, 5, v25 // (op 6698) 

v_mov_b32 v27, 0 // (op 6698) convert
v_mov_b32 v26, v24 // (op 6698) convert

v_mul_lo_u32 v25, v23, s13 // (op 6698) most significant: low of low * high
v_mul_hi_u32 v28, v23, s12 // (op 6698) most significant: high of low * low
v_mul_lo_u32 v24, v23, s12 // (op 6698) least significant: low of low * low
v_add_u32 v25, v25, v28 // (op 6698) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v26 // (op 6698) least significant half
v_addc_co_u32 v25, s[0:1], v25, v27, s[0:1] // (op 6698) most significant half



v_lshlrev_b64 v[26:27], 2, v[24:25] // (op 6698) 

v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 6698) 

v_mov_b32 v23, v24 // (op 6698) convert

s_mul_i32 s1, s12, s23 // (op 6700) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6700) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6700) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6700) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6700) most significant: sum
s_add_u32 s1, s1, s7 // (op 6700) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6700) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6700) 

s_mov_b32 s48, s20 // (op 6700) reg expression
s_mov_b32 s49, s21 // (op 6700) reg expression
s_mov_b32 s50, s0 // (op 6700) reg expression
s_mov_b32 s51, 131072 // (op 6700) call()

v_and_b32 v24, -4, v1 // (op 6743) 
v_sub_u32 v25, v1, v24 // (op 6743) 

v_lshlrev_b32 v24, 5, v25 // (op 6743) 

v_mov_b32 v27, 0 // (op 6743) convert
v_mov_b32 v26, v24 // (op 6743) convert

v_lshrrev_b32 v24, 2, v1 // (op 6743) 
v_lshl_add_u32 v25, s17, 6, v24 // (op 6743) 

v_mul_lo_u32 v29, v25, s25 // (op 6743) most significant: low of low * high
v_mul_hi_u32 v24, v25, s24 // (op 6743) most significant: high of low * low
v_mul_lo_u32 v28, v25, s24 // (op 6743) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 6743) most significant: sum


v_add_co_u32 v24, s[0:1], v26, v28 // (op 6743) least significant half
v_addc_co_u32 v25, s[0:1], v27, v29, s[0:1] // (op 6743) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 6743) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 6743) 

v_mov_b32 v24, v26 // (op 6743) convert

s_mul_i32 s1, s24, s9 // (op 6745) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6745) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6745) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6745) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6745) most significant: sum
s_add_u32 s1, s1, s7 // (op 6745) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6745) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6745) 

s_mov_b32 s52, s26 // (op 6745) reg expression
s_mov_b32 s53, s27 // (op 6745) reg expression
s_mov_b32 s54, s0 // (op 6745) reg expression
s_mov_b32 s55, 131072 // (op 6745) call()

v_and_b32 v25, -4, v1 // (op 6762) 
v_sub_u32 v26, v1, v25 // (op 6762) 

v_lshlrev_b32 v25, 5, v26 // (op 6762) 

v_mov_b32 v27, 0 // (op 6762) convert
v_mov_b32 v26, v25 // (op 6762) convert

v_lshrrev_b32 v25, 2, v1 // (op 6762) 
v_lshl_add_u32 v28, s17, 6, v25 // (op 6762) 

v_mul_lo_u32 v31, v28, s25 // (op 6762) most significant: low of low * high
v_mul_hi_u32 v25, v28, s24 // (op 6762) most significant: high of low * low
v_mul_lo_u32 v30, v28, s24 // (op 6762) least significant: low of low * low
v_add_u32 v31, v31, v25 // (op 6762) most significant: sum


v_add_co_u32 v28, s[0:1], v26, v30 // (op 6762) least significant half
v_addc_co_u32 v29, s[0:1], v27, v31, s[0:1] // (op 6762) most significant half

v_lshlrev_b64 v[28:29], 2, v[28:29] // (op 6762) 


v_lshrrev_b64 v[26:27], 3, v[28:29] // (op 6762) 

v_mov_b32 v25, v26 // (op 6762) convert

s_mul_i32 s1, s24, s9 // (op 6764) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6764) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6764) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6764) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6764) most significant: sum
s_add_u32 s1, s1, s7 // (op 6764) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6764) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6764) 

s_mov_b32 s52, s26 // (op 6764) reg expression
s_mov_b32 s53, s27 // (op 6764) reg expression
s_mov_b32 s54, s0 // (op 6764) reg expression
s_mov_b32 s55, 131072 // (op 6764) call()

v_lshrrev_b32 v26, 2, v1 // (op 6807) 
v_lshlrev_b32 v27, 1, v26 // (op 6807) 

v_lshl_add_u32 v26, s16, 7, v27 // (op 6807) 

v_and_b32 v27, -4, v1 // (op 6807) 
v_sub_u32 v28, v1, v27 // (op 6807) 

v_mul_lo_u32 v27, v26, s28 // (op 6807) 
v_add_u32 v27, v27, v28 // (op 6807) 


s_mul_i32 s1, s28, s23 // (op 6809) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6809) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6809) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6809) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6809) most significant: sum
s_add_u32 s1, s1, s7 // (op 6809) most significant: sum


s_mov_b32 s56, s30 // (op 6809) reg expression
s_mov_b32 s57, s31 // (op 6809) reg expression
s_mov_b32 s58, s0 // (op 6809) reg expression
s_mov_b32 s59, 131072 // (op 6809) call()

v_lshrrev_b32 v26, 2, v1 // (op 6826) 
v_lshlrev_b32 v28, 1, v26 // (op 6826) 

v_lshl_add_u32 v26, s16, 7, v28 // (op 6826) 

v_and_b32 v28, -4, v1 // (op 6826) 
v_sub_u32 v29, v1, v28 // (op 6826) 

v_mul_lo_u32 v28, v26, s28 // (op 6826) 
v_add_u32 v28, v28, v29 // (op 6826) 


s_mul_i32 s1, s28, s23 // (op 6828) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6828) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6828) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6828) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6828) most significant: sum
s_add_u32 s1, s1, s7 // (op 6828) most significant: sum


s_mov_b32 s56, s30 // (op 6828) reg expression
s_mov_b32 s57, s31 // (op 6828) reg expression
s_mov_b32 s58, s0 // (op 6828) reg expression
s_mov_b32 s59, 131072 // (op 6828) call()

v_and_b32 v26, -4, v1 // (op 6871) 
v_sub_u32 v29, v1, v26 // (op 6871) 

v_lshrrev_b32 v26, 2, v1 // (op 6871) 
v_lshl_add_u32 v30, s17, 6, v26 // (op 6871) 

v_mul_lo_u32 v26, v30, s32 // (op 6871) 

v_add_u32 v30, v29, v26 // (op 6871) 


s_mul_i32 s1, s32, s9 // (op 6873) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6873) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6873) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6873) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6873) most significant: sum
s_add_u32 s1, s1, s7 // (op 6873) most significant: sum


s_mov_b32 s60, s34 // (op 6873) reg expression
s_mov_b32 s61, s35 // (op 6873) reg expression
s_mov_b32 s62, s0 // (op 6873) reg expression
s_mov_b32 s63, 131072 // (op 6873) call()

v_and_b32 v26, -4, v1 // (op 6890) 
v_sub_u32 v29, v1, v26 // (op 6890) 

v_lshrrev_b32 v26, 2, v1 // (op 6890) 
v_lshl_add_u32 v31, s17, 6, v26 // (op 6890) 

v_mul_lo_u32 v26, v31, s32 // (op 6890) 

v_add_u32 v31, v29, v26 // (op 6890) 


s_mul_i32 s1, s32, s9 // (op 6892) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6892) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6892) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6892) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6892) most significant: sum
s_add_u32 s1, s1, s7 // (op 6892) most significant: sum


s_mov_b32 s60, s34 // (op 6892) reg expression
s_mov_b32 s61, s35 // (op 6892) reg expression
s_mov_b32 s62, s0 // (op 6892) reg expression
s_mov_b32 s63, 131072 // (op 6892) call()




buffer_load_dwordx4 v[32:35], v24, s[52:55], 0 offen // (op 6219) Load value



buffer_load_ubyte v26, v30, s[60:63], 0 offen offset: 0 // (op 6222) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 6225) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 6225) 


v_mov_b32 v29, v22 // (op 6225) 
buffer_load_dwordx4 v[38:41], v29, s[48:51], 0 offen // (op 6225) Load value
v_add_u32 v29, v29, v36 // (op 6225) 
buffer_load_dwordx4 v[42:45], v29, s[48:51], 0 offen // (op 6225) Load value



v_mov_b64 v[36:37], s[28:29] // (op 6228) convert

v_mov_b32 v29, v27 // (op 6228) 
buffer_load_ubyte v46, v29, s[56:59], 0 offen offset: 0 // (op 6228) Load value
v_add_u32 v29, v29, v36 // (op 6228) 
buffer_load_ubyte v47, v29, s[56:59], 0 offen offset: 0 // (op 6228) Load value





buffer_load_dwordx4 v[48:51], v24, s[52:55], 0 offen offset:64 // (op 6231) Load value



buffer_load_ubyte v29, v30, s[60:63], 0 offen offset:4 // (op 6234) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 6237) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 6237) 


v_mov_b32 v52, v22 // (op 6237) 
buffer_load_dwordx4 v[54:57], v52, s[48:51], 0 offen offset:64 // (op 6237) Load value
v_add_u32 v52, v52, v36 // (op 6237) 
buffer_load_dwordx4 v[58:61], v52, s[48:51], 0 offen offset:64 // (op 6237) Load value



v_mov_b64 v[36:37], s[28:29] // (op 6240) convert

v_mov_b32 v52, v27 // (op 6240) 
buffer_load_ubyte v62, v52, s[56:59], 0 offen offset:4 // (op 6240) Load value
v_add_u32 v52, v52, v36 // (op 6240) 
buffer_load_ubyte v63, v52, s[56:59], 0 offen offset:4 // (op 6240) Load value





buffer_load_dwordx4 v[64:67], v24, s[52:55], 0 offen offset:128 // (op 6243) Load value



buffer_load_ubyte v36, v30, s[60:63], 0 offen offset:8 // (op 6246) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 6249) 
v_lshrrev_b64 v[52:53], 3, s[0:1] // (op 6249) 


v_mov_b32 v37, v22 // (op 6249) 
buffer_load_dwordx4 v[68:71], v37, s[48:51], 0 offen offset:128 // (op 6249) Load value
v_add_u32 v37, v37, v52 // (op 6249) 
buffer_load_dwordx4 v[72:75], v37, s[48:51], 0 offen offset:128 // (op 6249) Load value



v_mov_b64 v[52:53], s[28:29] // (op 6252) convert

v_mov_b32 v37, v27 // (op 6252) 
buffer_load_ubyte v76, v37, s[56:59], 0 offen offset:8 // (op 6252) Load value
v_add_u32 v37, v37, v52 // (op 6252) 
buffer_load_ubyte v77, v37, s[56:59], 0 offen offset:8 // (op 6252) Load value





buffer_load_dwordx4 v[78:81], v24, s[52:55], 0 offen offset:192 // (op 6255) Load value




buffer_load_ubyte v24, v30, s[60:63], 0 offen offset:12 // (op 6258) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 6261) 
v_lshrrev_b64 v[52:53], 3, s[0:1] // (op 6261) 


v_mov_b32 v30, v22 // (op 6261) 
buffer_load_dwordx4 v[82:85], v30, s[48:51], 0 offen offset:192 // (op 6261) Load value
v_add_u32 v30, v30, v52 // (op 6261) 
buffer_load_dwordx4 v[86:89], v30, s[48:51], 0 offen offset:192 // (op 6261) Load value




v_mov_b64 v[52:53], s[28:29] // (op 6264) convert

v_mov_b32 v22, v27 // (op 6264) 
buffer_load_ubyte v90, v22, s[56:59], 0 offen offset:12 // (op 6264) Load value
v_add_u32 v22, v22, v52 // (op 6264) 
buffer_load_ubyte v91, v22, s[56:59], 0 offen offset:12 // (op 6264) Load value



v_and_b32 v22, -4, v1 // (op 7020) 
v_sub_u32 v27, v1, v22 // (op 7020) 

v_lshlrev_b32 v22, 5, v27 // (op 7020) 

v_lshrrev_b32 v27, 2, v1 // (op 7020) 
v_lshl_add_u32 v30, v27, 7, v22 // (op 7020) 


v_lshlrev_b32 v22, 2, v30 // (op 7020) 

v_lshrrev_b32 v27, 3, v22 // (op 7020) 




s_waitcnt vmcnt(23)
ds_write_b128 v27, v[32:35] // (op 6266) Store local data 

v_and_b32 v22, -4, v1 // (op 7136) 
v_sub_u32 v27, v1, v22 // (op 7136) 

v_and_b32 v22, 4294967292, v1 // (op 7136) 
v_add_u32 v30, v22, v27 // (op 7136) 





s_waitcnt vmcnt(22)
ds_write_b8 v30, v26 offset:4096 // (op 6267) Store local data 

v_and_b32 v22, -4, v1 // (op 6962) 
v_sub_u32 v27, v1, v22 // (op 6962) 

v_lshlrev_b32 v22, 5, v27 // (op 6962) 

v_lshrrev_b32 v27, 2, v1 // (op 6962) 
v_lshl_add_u32 v30, v27, 8, v22 // (op 6962) 


v_lshlrev_b32 v22, 2, v30 // (op 6962) 

v_lshrrev_b32 v27, 3, v22 // (op 6962) 




s_waitcnt vmcnt(21)
ds_write_b128 v27, v[38:41] offset:4352 // (op 6268) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v27, v[42:45] offset:4416 // (op 6268) Store local data 

v_and_b32 v22, -4, v1 // (op 7078) 
v_sub_u32 v27, v1, v22 // (op 7078) 

v_lshrrev_b32 v22, 2, v1 // (op 7078) 
v_lshl_add_u32 v30, v22, 3, v27 // (op 7078) 





s_waitcnt vmcnt(19)
ds_write_b8 v30, v46 offset:12544 // (op 6269) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v30, v47 offset:12548 // (op 6269) Store local data 

v_and_b32 v22, -4, v1 // (op 6933) 
v_sub_u32 v27, v1, v22 // (op 6933) 

v_lshlrev_b32 v22, 5, v27 // (op 6933) 

v_lshrrev_b32 v27, 2, v1 // (op 6933) 
v_lshl_add_u32 v30, v27, 8, v22 // (op 6933) 


v_lshlrev_b32 v22, 2, v30 // (op 6933) 

v_lshrrev_b32 v27, 3, v22 // (op 6933) 

v_and_b32 v22, -4, v1 // (op 6991) 
v_sub_u32 v30, v1, v22 // (op 6991) 

v_lshlrev_b32 v22, 5, v30 // (op 6991) 

v_lshrrev_b32 v30, 2, v1 // (op 6991) 
v_lshl_add_u32 v37, v30, 7, v22 // (op 6991) 


v_lshlrev_b32 v22, 2, v37 // (op 6991) 

v_lshrrev_b32 v30, 3, v22 // (op 6991) 

v_and_b32 v22, -4, v1 // (op 7049) 
v_sub_u32 v37, v1, v22 // (op 7049) 

v_lshrrev_b32 v22, 2, v1 // (op 7049) 
v_lshl_add_u32 v52, v22, 3, v37 // (op 7049) 


v_and_b32 v22, -4, v1 // (op 7107) 
v_sub_u32 v37, v1, v22 // (op 7107) 

v_and_b32 v22, 4294967292, v1 // (op 7107) 
v_add_u32 v53, v22, v37 // (op 7107) 


v_and_b32 v22, -4, v1 // (op 7188) 
v_sub_u32 v37, v1, v22 // (op 7188) 

v_lshlrev_b32 v22, 5, v37 // (op 7188) 

v_lshrrev_b32 v37, 2, v1 // (op 7188) 
v_lshl_add_u32 v92, v37, 8, v22 // (op 7188) 


v_lshlrev_b32 v22, 2, v92 // (op 7188) 

v_lshrrev_b32 v37, 3, v22 // (op 7188) 

v_and_b32 v22, -4, v1 // (op 7217) 
v_sub_u32 v92, v1, v22 // (op 7217) 

v_lshlrev_b32 v22, 5, v92 // (op 7217) 

v_lshrrev_b32 v92, 2, v1 // (op 7217) 
v_lshl_add_u32 v93, v92, 7, v22 // (op 7217) 


v_lshlrev_b32 v22, 2, v93 // (op 7217) 

v_lshrrev_b32 v92, 3, v22 // (op 7217) 

v_and_b32 v22, -4, v1 // (op 7246) 
v_sub_u32 v93, v1, v22 // (op 7246) 

v_lshrrev_b32 v22, 2, v1 // (op 7246) 
v_lshl_add_u32 v94, v22, 3, v93 // (op 7246) 


v_and_b32 v22, -4, v1 // (op 7275) 
v_sub_u32 v93, v1, v22 // (op 7275) 

v_and_b32 v22, 4294967292, v1 // (op 7275) 
v_add_u32 v95, v22, v93 // (op 7275) 


v_and_b32 v22, -4, v1 // (op 7304) 
v_sub_u32 v93, v1, v22 // (op 7304) 

v_lshlrev_b32 v22, 5, v93 // (op 7304) 

v_lshrrev_b32 v93, 2, v1 // (op 7304) 
v_lshl_add_u32 v96, v93, 8, v22 // (op 7304) 


v_lshlrev_b32 v22, 2, v96 // (op 7304) 

v_lshrrev_b32 v93, 3, v22 // (op 7304) 

v_and_b32 v22, -4, v1 // (op 7333) 
v_sub_u32 v96, v1, v22 // (op 7333) 

v_lshlrev_b32 v22, 5, v96 // (op 7333) 

v_lshrrev_b32 v96, 2, v1 // (op 7333) 
v_lshl_add_u32 v97, v96, 7, v22 // (op 7333) 


v_lshlrev_b32 v22, 2, v97 // (op 7333) 

v_lshrrev_b32 v96, 3, v22 // (op 7333) 

v_and_b32 v22, -4, v1 // (op 7362) 
v_sub_u32 v97, v1, v22 // (op 7362) 

v_lshrrev_b32 v22, 2, v1 // (op 7362) 
v_lshl_add_u32 v98, v22, 3, v97 // (op 7362) 


v_and_b32 v22, -4, v1 // (op 7391) 
v_sub_u32 v97, v1, v22 // (op 7391) 

v_and_b32 v22, 4294967292, v1 // (op 7391) 
v_add_u32 v99, v22, v97 // (op 7391) 


v_and_b32 v22, -4, v1 // (op 7420) 
v_sub_u32 v97, v1, v22 // (op 7420) 

v_lshlrev_b32 v22, 5, v97 // (op 7420) 

v_lshrrev_b32 v97, 2, v1 // (op 7420) 
v_lshl_add_u32 v100, v97, 8, v22 // (op 7420) 


v_lshlrev_b32 v22, 2, v100 // (op 7420) 

v_lshrrev_b32 v97, 3, v22 // (op 7420) 

v_and_b32 v22, -4, v1 // (op 7449) 
v_sub_u32 v100, v1, v22 // (op 7449) 

v_lshlrev_b32 v22, 5, v100 // (op 7449) 

v_lshrrev_b32 v100, 2, v1 // (op 7449) 
v_lshl_add_u32 v101, v100, 7, v22 // (op 7449) 


v_lshlrev_b32 v22, 2, v101 // (op 7449) 

v_lshrrev_b32 v100, 3, v22 // (op 7449) 

v_and_b32 v22, -4, v1 // (op 7478) 
v_sub_u32 v101, v1, v22 // (op 7478) 

v_lshrrev_b32 v22, 2, v1 // (op 7478) 
v_lshl_add_u32 v102, v22, 3, v101 // (op 7478) 


v_and_b32 v22, -4, v1 // (op 7507) 
v_sub_u32 v101, v1, v22 // (op 7507) 

v_and_b32 v22, 4294967292, v1 // (op 7507) 
v_add_u32 v103, v22, v101 // (op 7507) 


s_mov_b32 s0, 0 // (op 2173) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 2172) 
s_mov_b32 s1, s6 // (op 2172) convert

s_and_b32 s6, s1, 4294967292 // (op 2172) 

s_cmp_lt_u32 s0, s6 // (op 2172) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_2172 // (op 2172) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_2172 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoop_2172:


s_barrier  // (op 7627) 
s_add_u32 s0, s0, 4 // (op 2174) 



v_mov_b32 v22, v25 // (op 10) 

ds_read_b128 v[104:107], v3 offset:4352 // (op 70) Load local data 
buffer_load_dwordx4 v[32:35], v22, s[52:55], 0 offen offset:256 // (op 10) Load value


s_barrier  // (op 7640) 
v_mov_b32 v101, v31 // (op 25) 




buffer_load_ubyte v26, v101, s[60:63], 0 offen offset:16 // (op 25) Load value
ds_read_b128 v[108:111], v4 // (op 75) Load local data 

s_barrier  // (op 7654) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 



ds_read_u8 v112, v0 offset:12544 // (op 80) Load local data 
v_lshrrev_b64 v[114:115], 3, s[6:7] // (op 4) 


s_barrier  // (op 7668) 
v_mov_b32 v113, v23 // (op 4) 



ds_read_u8 v116, v5 offset:4096 // (op 85) Load local data 
v_mov_b32 v117, v113 // (op 4) 



buffer_load_dwordx4 v[38:41], v117, s[48:51], 0 offen offset:256 // (op 4) Load value
v_add_u32 v117, v117, v114 // (op 4) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[108:111], a[0:3], v112, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 88) 
ds_read_b128 v[118:121], v4 offset:2048 // (op 517) Load local data 
buffer_load_dwordx4 v[42:45], v117, s[48:51], 0 offen offset:256 // (op 4) Load value






ds_read_u8 v117, v5 offset:4224 // (op 521) Load local data 
v_mov_b64 v[114:115], s[28:29] // (op 17) convert




v_mov_b32 v122, v28 // (op 17) 
v_mov_b32 v123, v122 // (op 17) 
buffer_load_ubyte v46, v123, s[56:59], 0 offen offset:16 // (op 17) Load value
v_add_u32 v123, v123, v114 // (op 17) 
ds_read_b128 v[124:127], v3 offset:6400 // (op 914) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[118:121], a[4:7], v112, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 522) 





ds_read_u8 v112, v0 offset:12672 // (op 917) Load local data 
buffer_load_ubyte v47, v123, s[56:59], 0 offen offset:16 // (op 17) Load value





s_barrier  // (op 7691) 
ds_read_b128 v[104:107], v3 offset:8448 // (op 1560) Load local data 






ds_read_u8 v123, v0 offset:12800 // (op 1563) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[108:111], a[8:11], v112, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1011) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[118:121], a[12:15], v112, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 919) 
ds_read_b128 v[128:131], v3 offset:10496 // (op 2206) Load local data 





ds_read_u8 v112, v0 offset:12928 // (op 2209) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[108:111], a[16:19], v123, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1657) 
s_waitcnt vmcnt(23)
ds_write_b128 v92, v[48:51] offset:13056 // (op 253) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[118:121], a[20:23], v123, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1565) 


s_barrier  // (op 7703) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[108:111], a[24:27], v112, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2303) 
s_waitcnt vmcnt(22)
ds_write_b8 v95, v29 offset:17152 // (op 255) Store local data 


s_barrier  // (op 7685) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[118:121], a[28:31], v112, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2211) 







s_waitcnt vmcnt(21)
ds_write_b128 v37, v[54:57] offset:17408 // (op 243) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v37, v[58:61] offset:17472 // (op 243) Store local data 
s_barrier  // (op 7697) 



s_waitcnt vmcnt(19)
ds_write_b8 v94, v62 offset:25600 // (op 254) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v94, v63 offset:25604 // (op 254) Store local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 7682) 
buffer_load_dwordx4 v[48:51], v22, s[52:55], 0 offen offset:320 // (op 256) Load value






ds_read_b128 v[104:107], v7 offset:17408 // (op 245) Load local data 
buffer_load_ubyte v29, v101, s[60:63], 0 offen offset:20 // (op 258) Load value

s_barrier  // (op 7688) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 242) 



v_lshrrev_b64 v[108:109], 3, s[6:7] // (op 242) 


v_mov_b32 v110, v113 // (op 242) 
ds_read_b128 v[114:117], v8 offset:13056 // (op 247) Load local data 
buffer_load_dwordx4 v[54:57], v110, s[48:51], 0 offen offset:320 // (op 242) Load value
v_add_u32 v110, v110, v108 // (op 242) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7694) 
buffer_load_dwordx4 v[58:61], v110, s[48:51], 0 offen offset:320 // (op 242) Load value






ds_read_u8 v108, v6 offset:25600 // (op 249) Load local data 
v_mov_b64 v[118:119], s[28:29] // (op 257) convert

s_barrier  // (op 7700) 
v_mov_b32 v109, v122 // (op 257) 



buffer_load_ubyte v62, v109, s[56:59], 0 offen offset:20 // (op 257) Load value
v_add_u32 v109, v109, v118 // (op 257) 
ds_read_u8 v110, v9 offset:17152 // (op 251) Load local data 
buffer_load_ubyte v63, v109, s[56:59], 0 offen offset:20 // (op 257) Load value





s_barrier  // (op 7715) 
ds_read_b128 v[118:121], v8 offset:15104 // (op 529) Load local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[114:117], a[0:3], v108, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 252) 



s_barrier  // (op 7828) 
ds_read_u8 v109, v9 offset:17280 // (op 533) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v96, v[64:67] offset:26112 // (op 293) Store local data 



s_barrier  // (op 7832) 
s_barrier  // (op 7727) 
ds_read_b128 v[124:127], v7 offset:19456 // (op 924) Load local data 






s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[118:121], a[4:7], v108, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 523) 
ds_read_u8 v123, v6 offset:25728 // (op 927) Load local data 





ds_read_b128 v[104:107], v7 offset:21504 // (op 1570) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[114:117], a[8:11], v123, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1012) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[118:121], a[12:15], v123, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 920) 
ds_read_u8 v108, v6 offset:25856 // (op 1573) Load local data 





ds_read_b128 v[124:127], v7 offset:23552 // (op 2216) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[114:117], a[16:19], v108, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1658) 



s_barrier  // (op 7892) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[118:121], a[20:23], v108, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1566) 
ds_read_u8 v123, v6 offset:25984 // (op 2219) Load local data 


s_waitcnt vmcnt(22)
ds_write_b8 v99, v36 offset:30208 // (op 295) Store local data 
s_barrier  // (op 7896) 
s_barrier  // (op 7709) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[114:117], a[24:27], v123, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2304) 





s_waitcnt vmcnt(21)
ds_write_b128 v93, v[68:71] offset:13056 // (op 283) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[118:121], a[28:31], v123, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2212) 
s_waitcnt vmcnt(20)
ds_write_b128 v93, v[72:75] offset:13120 // (op 283) Store local data 




s_barrier  // (op 7721) 



s_waitcnt vmcnt(19)
ds_write_b8 v98, v76 offset:21248 // (op 294) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v98, v77 offset:21252 // (op 294) Store local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 7706) 
buffer_load_dwordx4 v[64:67], v22, s[52:55], 0 offen offset:384 // (op 296) Load value






ds_read_b128 v[104:107], v11 offset:13056 // (op 285) Load local data 
buffer_load_ubyte v36, v101, s[60:63], 0 offen offset:24 // (op 298) Load value

s_barrier  // (op 7712) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 282) 



v_lshrrev_b64 v[108:109], 3, s[6:7] // (op 282) 


v_mov_b32 v110, v113 // (op 282) 
ds_read_b128 v[114:117], v12 offset:26112 // (op 287) Load local data 
buffer_load_dwordx4 v[68:71], v110, s[48:51], 0 offen offset:384 // (op 282) Load value
v_add_u32 v110, v110, v108 // (op 282) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7718) 
buffer_load_dwordx4 v[72:75], v110, s[48:51], 0 offen offset:384 // (op 282) Load value






ds_read_u8 v108, v10 offset:21248 // (op 289) Load local data 
v_mov_b64 v[118:119], s[28:29] // (op 297) convert

s_barrier  // (op 7724) 
v_mov_b32 v109, v122 // (op 297) 



buffer_load_ubyte v76, v109, s[56:59], 0 offen offset:24 // (op 297) Load value
v_add_u32 v109, v109, v118 // (op 297) 
ds_read_u8 v110, v13 offset:30208 // (op 291) Load local data 
buffer_load_ubyte v77, v109, s[56:59], 0 offen offset:24 // (op 297) Load value





s_barrier  // (op 7739) 
ds_read_b128 v[118:121], v12 offset:28160 // (op 537) Load local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[114:117], a[0:3], v108, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 292) 



s_barrier  // (op 7836) 
ds_read_u8 v109, v13 offset:30336 // (op 541) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v100, v[78:81] offset:21760 // (op 333) Store local data 



s_barrier  // (op 7840) 
s_barrier  // (op 7751) 
ds_read_b128 v[124:127], v11 offset:15104 // (op 930) Load local data 






s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[118:121], a[4:7], v108, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 524) 
ds_read_u8 v123, v10 offset:21376 // (op 933) Load local data 





ds_read_b128 v[104:107], v11 offset:17152 // (op 1576) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[114:117], a[8:11], v123, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1013) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[118:121], a[12:15], v123, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 921) 
ds_read_u8 v108, v10 offset:21504 // (op 1579) Load local data 





ds_read_b128 v[124:127], v11 offset:19200 // (op 2222) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[114:117], a[16:19], v108, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1659) 



s_barrier  // (op 7900) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[118:121], a[20:23], v108, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1567) 
ds_read_u8 v123, v10 offset:21632 // (op 2225) Load local data 


s_waitcnt vmcnt(22)
ds_write_b8 v103, v24 offset:25856 // (op 335) Store local data 
s_barrier  // (op 7904) 
s_barrier  // (op 7733) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[114:117], a[24:27], v123, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2305) 





s_waitcnt vmcnt(21)
ds_write_b128 v97, v[82:85] offset:13056 // (op 323) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[118:121], a[28:31], v123, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2213) 
s_waitcnt vmcnt(20)
ds_write_b128 v97, v[86:89] offset:13120 // (op 323) Store local data 




s_barrier  // (op 7745) 



s_waitcnt vmcnt(19)
ds_write_b8 v102, v90 offset:21248 // (op 334) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v102, v91 offset:21252 // (op 334) Store local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 7730) 
buffer_load_dwordx4 v[78:81], v22, s[52:55], 0 offen offset:448 // (op 336) Load value




ds_read_b128 v[104:107], v15 offset:13056 // (op 325) Load local data 



s_barrier  // (op 7736) 
v_add_u32 v25, 256, v25 // (op 6754) 



ds_read_b128 v[108:111], v16 offset:21760 // (op 327) Load local data 
buffer_load_ubyte v24, v101, s[60:63], 0 offen offset:28 // (op 338) Load value

s_waitcnt lgkmcnt(2)
s_barrier  // (op 7742) 




v_add_u32 v31, 16, v31 // (op 6882) 
ds_read_u8 v22, v14 offset:21248 // (op 329) Load local data 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 322) 
s_barrier  // (op 7748) 
v_lshrrev_b64 v[114:115], 3, s[6:7] // (op 322) 





v_mov_b32 v101, v113 // (op 322) 
buffer_load_dwordx4 v[82:85], v101, s[48:51], 0 offen offset:448 // (op 322) Load value
v_add_u32 v101, v101, v114 // (op 322) 
ds_read_u8 v112, v17 offset:25856 // (op 331) Load local data 
buffer_load_dwordx4 v[86:89], v101, s[48:51], 0 offen offset:448 // (op 322) Load value






ds_read_b128 v[118:121], v16 offset:23808 // (op 545) Load local data 

v_add_u32 v23, 256, v23 // (op 6690) 



s_barrier  // (op 7844) 
ds_read_u8 v101, v17 offset:25984 // (op 549) Load local data 
v_mov_b64 v[116:117], s[28:29] // (op 337) convert

v_mov_b32 v113, v122 // (op 337) 



s_barrier  // (op 7848) 
buffer_load_ubyte v90, v113, s[56:59], 0 offen offset:28 // (op 337) Load value
v_add_u32 v113, v113, v116 // (op 337) 
ds_read_b128 v[124:127], v15 offset:15104 // (op 936) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[108:111], a[0:3], v22, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 332) 



buffer_load_ubyte v91, v113, s[56:59], 0 offen offset:28 // (op 337) Load value


s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[118:121], a[4:7], v22, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 525) 



s_barrier  // (op 7643) 
v_add_u32 v28, 16, v28 // (op 6818) 



ds_read_u8 v22, v14 offset:21376 // (op 939) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v30, v[32:35] // (op 13) Store local data 



ds_read_b128 v[104:107], v15 offset:17152 // (op 1582) Load local data 



s_barrier  // (op 7671) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[108:111], a[8:11], v22, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1014) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[118:121], a[12:15], v22, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 922) 
ds_read_u8 v113, v14 offset:21504 // (op 1585) Load local data 





ds_read_b128 v[114:117], v15 offset:19200 // (op 2228) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[108:111], a[16:19], v113, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1660) 



s_barrier  // (op 7908) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[118:121], a[20:23], v113, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1568) 
ds_read_u8 v22, v14 offset:21632 // (op 2231) Load local data 


s_waitcnt vmcnt(22)
ds_write_b8 v53, v26 offset:4096 // (op 30) Store local data 
s_barrier  // (op 7912) 
s_barrier  // (op 7630) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:117], v[108:111], a[24:27], v22, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2306) 





s_waitcnt vmcnt(21)
ds_write_b128 v27, v[38:41] offset:4352 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[114:117], v[118:121], a[28:31], v22, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2214) 
s_waitcnt vmcnt(20)
ds_write_b128 v27, v[42:45] offset:4416 // (op 6) Store local data 




s_barrier  // (op 7657) 



s_waitcnt vmcnt(19)
ds_write_b8 v52, v46 offset:12544 // (op 21) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v52, v47 offset:12548 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 2172) 
s_mov_b32 s1, s6 // (op 2172) convert

s_and_b32 s6, s1, 4294967292 // (op 2172) 

s_cmp_lt_u32 s0, s6 // (op 2172) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoop_2172 // (op 2172) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoop_2172 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_2172:

























v_lshrrev_b32 v22, 2, v1 // (op 6646) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6646) 
v_lshlrev_b32 v23, 1, v22 // (op 6646) 

s_lshr_b32 s6, s0, 2 // (op 6646) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 6646) 

v_and_b32 v23, -4, v1 // (op 6646) 
v_sub_u32 v25, v1, v23 // (op 6646) 

v_lshlrev_b32 v23, 5, v25 // (op 6646) 

v_lshl_add_u32 v25, s6, 9, v23 // (op 6646) 


v_mov_b32 v53, 0 // (op 6646) convert
v_mov_b32 v52, v25 // (op 6646) convert

v_mul_lo_u32 v31, v22, s13 // (op 6646) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 6646) most significant: high of low * low
v_mul_lo_u32 v30, v22, s12 // (op 6646) least significant: low of low * low
v_add_u32 v31, v31, v23 // (op 6646) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v52 // (op 6646) least significant half
v_addc_co_u32 v31, s[0:1], v31, v53, s[0:1] // (op 6646) most significant half



v_lshlrev_b64 v[22:23], 2, v[30:31] // (op 6646) 

v_lshrrev_b64 v[52:53], 3, v[22:23] // (op 6646) 

v_mov_b32 v22, v52 // (op 6646) convert

s_mul_i32 s1, s12, s23 // (op 6648) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6648) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6648) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6648) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6648) most significant: sum
s_add_u32 s1, s1, s7 // (op 6648) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6648) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6648) 

s_mov_b32 s48, s20 // (op 6648) reg expression
s_mov_b32 s49, s21 // (op 6648) reg expression
s_mov_b32 s50, s0 // (op 6648) reg expression
s_mov_b32 s51, 131072 // (op 6648) call()

v_and_b32 v23, -4, v1 // (op 6720) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6720) 
v_sub_u32 v25, v1, v23 // (op 6720) 

s_lshr_b32 s6, s0, 2 // (op 6720) 

v_lshlrev_b32 v23, 5, v25 // (op 6720) 

v_lshl_add_u32 v25, s6, 9, v23 // (op 6720) 


v_mov_b32 v53, 0 // (op 6720) convert
v_mov_b32 v52, v25 // (op 6720) convert

v_lshrrev_b32 v23, 2, v1 // (op 6720) 
v_lshl_add_u32 v25, s17, 6, v23 // (op 6720) 

v_mul_lo_u32 v27, v25, s25 // (op 6720) most significant: low of low * high
v_mul_hi_u32 v23, v25, s24 // (op 6720) most significant: high of low * low
v_mul_lo_u32 v26, v25, s24 // (op 6720) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 6720) most significant: sum


v_add_co_u32 v30, s[0:1], v52, v26 // (op 6720) least significant half
v_addc_co_u32 v31, s[0:1], v53, v27, s[0:1] // (op 6720) most significant half

v_lshlrev_b64 v[30:31], 2, v[30:31] // (op 6720) 


v_lshrrev_b64 v[52:53], 3, v[30:31] // (op 6720) 

v_mov_b32 v23, v52 // (op 6720) convert

s_mul_i32 s1, s24, s9 // (op 6722) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6722) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6722) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6722) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6722) most significant: sum
s_add_u32 s1, s1, s7 // (op 6722) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6722) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6722) 

s_mov_b32 s52, s26 // (op 6722) reg expression
s_mov_b32 s53, s27 // (op 6722) reg expression
s_mov_b32 s54, s0 // (op 6722) reg expression
s_mov_b32 s55, 131072 // (op 6722) call()

v_lshrrev_b32 v25, 2, v1 // (op 6784) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6784) 
v_lshlrev_b32 v26, 1, v25 // (op 6784) 

s_lshr_b32 s6, s0, 2 // (op 6784) 

v_lshl_add_u32 v25, s16, 7, v26 // (op 6784) 

v_and_b32 v26, -4, v1 // (op 6784) 
v_sub_u32 v27, v1, v26 // (op 6784) 

v_lshl_add_u32 v26, s6, 4, v27 // (op 6784) 


v_mul_lo_u32 v27, v25, s28 // (op 6784) 
v_add_u32 v27, v27, v26 // (op 6784) 


s_mul_i32 s1, s28, s23 // (op 6786) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6786) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6786) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6786) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6786) most significant: sum
s_add_u32 s1, s1, s7 // (op 6786) most significant: sum


s_mov_b32 s56, s30 // (op 6786) reg expression
s_mov_b32 s57, s31 // (op 6786) reg expression
s_mov_b32 s58, s0 // (op 6786) reg expression
s_mov_b32 s59, 131072 // (op 6786) call()

v_and_b32 v28, -4, v1 // (op 6848) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6848) 
v_sub_u32 v25, v1, v28 // (op 6848) 

s_lshr_b32 s6, s0, 2 // (op 6848) 

v_lshrrev_b32 v26, 2, v1 // (op 6848) 
v_lshl_add_u32 v28, s6, 4, v25 // (op 6848) 


v_lshl_add_u32 v25, s17, 6, v26 // (op 6848) 

v_mul_lo_u32 v26, v25, s32 // (op 6848) 

v_add_u32 v25, v28, v26 // (op 6848) 


s_mul_i32 s1, s32, s9 // (op 6850) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6850) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6850) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6850) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6850) most significant: sum
s_add_u32 s1, s1, s7 // (op 6850) most significant: sum


s_mov_b32 s60, s34 // (op 6850) reg expression
s_mov_b32 s61, s35 // (op 6850) reg expression
s_mov_b32 s62, s0 // (op 6850) reg expression
s_mov_b32 s63, 131072 // (op 6850) call()

v_and_b32 v26, -4, v1 // (op 6946) 
v_sub_u32 v28, v1, v26 // (op 6946) 

v_lshlrev_b32 v26, 5, v28 // (op 6946) 

v_lshrrev_b32 v28, 2, v1 // (op 6946) 
v_lshl_add_u32 v30, v28, 8, v26 // (op 6946) 


v_lshlrev_b32 v26, 2, v30 // (op 6946) 

v_lshrrev_b32 v28, 3, v26 // (op 6946) 

v_and_b32 v26, -4, v1 // (op 7004) 
v_sub_u32 v30, v1, v26 // (op 7004) 

v_lshlrev_b32 v26, 5, v30 // (op 7004) 

v_lshrrev_b32 v30, 2, v1 // (op 7004) 
v_lshl_add_u32 v31, v30, 7, v26 // (op 7004) 


v_lshlrev_b32 v26, 2, v31 // (op 7004) 

v_lshrrev_b32 v30, 3, v26 // (op 7004) 

v_and_b32 v26, -4, v1 // (op 7062) 
v_sub_u32 v31, v1, v26 // (op 7062) 

v_lshrrev_b32 v26, 2, v1 // (op 7062) 
v_lshl_add_u32 v32, v26, 3, v31 // (op 7062) 


v_and_b32 v26, -4, v1 // (op 7120) 
v_sub_u32 v31, v1, v26 // (op 7120) 

v_and_b32 v26, 4294967292, v1 // (op 7120) 
v_add_u32 v33, v26, v31 // (op 7120) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 2178) 
s_and_b32 s6, s0, 4294967292 // (op 2178) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 2177) 
s_mov_b32 s7, s0 // (op 2177) convert

s_cmp_lt_u32 s6, s7 // (op 2177) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_2177 // (op 2177) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_2177 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_2177:

s_add_u32 s6, s6, 1 // (op 2179) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 176) 
v_lshrrev_b64 v[34:35], 3, s[0:1] // (op 176) 


v_mov_b32 v24, v22 // (op 176) 
v_mov_b32 v26, v24 // (op 176) 
buffer_load_dwordx4 v[36:39], v26, s[48:51], 0 offen // (op 176) Load value
v_add_u32 v26, v26, v34 // (op 176) 
buffer_load_dwordx4 v[40:43], v26, s[48:51], 0 offen // (op 176) Load value



v_add_u32 v22, 64, v22 // (op 6636) 


s_barrier  // (op 7633) 
v_mov_b32 v24, v23 // (op 190) 




buffer_load_dwordx4 v[44:47], v24, s[52:55], 0 offen // (op 190) Load value

v_add_u32 v23, 64, v23 // (op 6712) 
s_waitcnt vmcnt(2)
ds_write_b128 v28, v[36:39] offset:4352 // (op 177) Store local data 

v_mov_b64 v[34:35], s[28:29] // (op 191) convert

v_mov_b32 v24, v27 // (op 191) 
v_mov_b32 v26, v24 // (op 191) 
buffer_load_ubyte v48, v26, s[56:59], 0 offen offset: 0 // (op 191) Load value
v_add_u32 v26, v26, v34 // (op 191) 
buffer_load_ubyte v49, v26, s[56:59], 0 offen offset: 0 // (op 191) Load value



v_add_u32 v27, 4, v27 // (op 6776) 
s_waitcnt vmcnt(3)
ds_write_b128 v28, v[40:43] offset:4416 // (op 177) Store local data 



v_mov_b32 v24, v25 // (op 192) 

s_barrier  // (op 7646) 
buffer_load_ubyte v26, v24, s[60:63], 0 offen offset: 0 // (op 192) Load value




v_add_u32 v25, 4, v25 // (op 6840) 
s_waitcnt vmcnt(3)
ds_write_b128 v30, v[44:47] // (op 187) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 7636) 



s_barrier  // (op 7660) 
ds_read_b128 v[34:37], v3 offset:4352 // (op 179) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 7650) 
s_waitcnt vmcnt(2)
ds_write_b8 v32, v48 offset:12544 // (op 188) Store local data 



ds_read_b128 v[38:41], v4 // (op 181) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v32, v49 offset:12548 // (op 188) Store local data 

s_barrier  // (op 7674) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7664) 






ds_read_u8 v24, v0 offset:12544 // (op 183) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v33, v26 offset:4096 // (op 189) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 7678) 



ds_read_u8 v26, v5 offset:4096 // (op 185) Load local data 



ds_read_b128 v[42:45], v4 offset:2048 // (op 496) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:41], a[0:3], v24, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 186) 



ds_read_u8 v29, v5 offset:4224 // (op 500) Load local data 



ds_read_b128 v[46:49], v3 offset:6400 // (op 860) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[42:45], a[4:7], v24, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 501) 





ds_read_u8 v24, v0 offset:12672 // (op 863) Load local data 



ds_read_b128 v[34:37], v3 offset:8448 // (op 1506) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[38:41], a[8:11], v24, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 865) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[42:45], a[12:15], v24, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 896) 
ds_read_u8 v31, v0 offset:12800 // (op 1509) Load local data 





ds_read_b128 v[46:49], v3 offset:10496 // (op 2152) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[34:37], v[38:41], a[16:19], v31, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1511) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:37], v[42:45], a[20:23], v31, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1542) 
ds_read_u8 v24, v0 offset:12928 // (op 2155) Load local data 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[46:49], v[38:41], a[24:27], v24, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2157) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[42:45], a[28:31], v24, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2188) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 2177) 
s_mov_b32 s7, s0 // (op 2177) convert

s_cmp_lt_u32 s6, s7 // (op 2177) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_2177 // (op 2177) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_2177 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_2177:























s_barrier  // (op 7876) 

















v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_133
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


v_accvgpr_read v14, a4 // (op 3037) 
v_accvgpr_read v15, a5 // (op 3037) 
v_accvgpr_read v16, a6 // (op 3037) 
v_accvgpr_read v17, a7 // (op 3037) 
v_mul_f32 v10, v19, v14 // (op 3037) 
v_mul_f32 v11, v19, v15 // (op 3037) 
v_mul_f32 v12, v19, v16 // (op 3037) 
v_mul_f32 v13, v19, v17 // (op 3037) 



s_lshl_b32 s0, s4, 7 // (op 3039) 
v_add_u32 v0, v21, s0 // (op 3039) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3039) Store value


v_accvgpr_read v14, a8 // (op 3061) 
v_accvgpr_read v15, a9 // (op 3061) 
v_accvgpr_read v16, a10 // (op 3061) 
v_accvgpr_read v17, a11 // (op 3061) 
v_mul_f32 v10, v19, v14 // (op 3061) 
v_mul_f32 v11, v19, v15 // (op 3061) 
v_mul_f32 v12, v19, v16 // (op 3061) 
v_mul_f32 v13, v19, v17 // (op 3061) 



v_add_u32 v0, 128, v21 // (op 3064) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3064) Store value


v_accvgpr_read v14, a12 // (op 3070) 
v_accvgpr_read v15, a13 // (op 3070) 
v_accvgpr_read v16, a14 // (op 3070) 
v_accvgpr_read v17, a15 // (op 3070) 
v_mul_f32 v10, v19, v14 // (op 3070) 
v_mul_f32 v11, v19, v15 // (op 3070) 
v_mul_f32 v12, v19, v16 // (op 3070) 
v_mul_f32 v13, v19, v17 // (op 3070) 



s_lshl_b32 s0, s4, 7 // (op 3066) 
s_add_u32 s0, s0, 128 // (op 3066) 
v_add_u32 v0, v21, s0 // (op 3066) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3066) Store value


v_accvgpr_read v14, a16 // (op 3097) 
v_accvgpr_read v15, a17 // (op 3097) 
v_accvgpr_read v16, a18 // (op 3097) 
v_accvgpr_read v17, a19 // (op 3097) 
v_mul_f32 v10, v19, v14 // (op 3097) 
v_mul_f32 v11, v19, v15 // (op 3097) 
v_mul_f32 v12, v19, v16 // (op 3097) 
v_mul_f32 v13, v19, v17 // (op 3097) 



v_add_u32 v0, 256, v21 // (op 3100) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3100) Store value


v_accvgpr_read v14, a20 // (op 3106) 
v_accvgpr_read v15, a21 // (op 3106) 
v_accvgpr_read v16, a22 // (op 3106) 
v_accvgpr_read v17, a23 // (op 3106) 
v_mul_f32 v10, v19, v14 // (op 3106) 
v_mul_f32 v11, v19, v15 // (op 3106) 
v_mul_f32 v12, v19, v16 // (op 3106) 
v_mul_f32 v13, v19, v17 // (op 3106) 



s_lshl_b32 s0, s4, 7 // (op 3102) 
s_add_u32 s0, s0, 256 // (op 3102) 
v_add_u32 v0, v21, s0 // (op 3102) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3102) Store value


v_accvgpr_read v14, a24 // (op 3133) 
v_accvgpr_read v15, a25 // (op 3133) 
v_accvgpr_read v16, a26 // (op 3133) 
v_accvgpr_read v17, a27 // (op 3133) 
v_mul_f32 v10, v19, v14 // (op 3133) 
v_mul_f32 v11, v19, v15 // (op 3133) 
v_mul_f32 v12, v19, v16 // (op 3133) 
v_mul_f32 v13, v19, v17 // (op 3133) 



v_add_u32 v0, 384, v21 // (op 3136) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3136) Store value


v_accvgpr_read v14, a28 // (op 3142) 
v_accvgpr_read v15, a29 // (op 3142) 
v_accvgpr_read v16, a30 // (op 3142) 
v_accvgpr_read v17, a31 // (op 3142) 
v_mul_f32 v10, v19, v14 // (op 3142) 
v_mul_f32 v11, v19, v15 // (op 3142) 
v_mul_f32 v12, v19, v16 // (op 3142) 
v_mul_f32 v13, v19, v17 // (op 3142) 



s_lshl_b32 s0, s4, 7 // (op 3138) 
s_add_u32 s0, s0, 384 // (op 3138) 
v_add_u32 v0, v21, s0 // (op 3138) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 3138) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 384, v20 // (op 3320) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3320) Load value

v_accvgpr_read v10, a24 // (op 3312) 
v_accvgpr_read v11, a25 // (op 3312) 
v_accvgpr_read v12, a26 // (op 3312) 
v_accvgpr_read v13, a27 // (op 3312) 
v_mul_f32 v6, v19, v10 // (op 3312) 
v_mul_f32 v7, v19, v11 // (op 3312) 
v_mul_f32 v8, v19, v12 // (op 3312) 
v_mul_f32 v9, v19, v13 // (op 3312) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3319) 
v_mul_f32 v11, v18, v15 // (op 3319) 
v_mul_f32 v12, v18, v16 // (op 3319) 
v_mul_f32 v13, v18, v17 // (op 3319) 

v_add_f32 v14, v10, v6 // (op 3313) 
v_add_f32 v15, v11, v7 // (op 3313) 
v_add_f32 v16, v12, v8 // (op 3313) 
v_add_f32 v17, v13, v9 // (op 3313) 




v_add_u32 v0, 384, v21 // (op 3315) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3315) Store value




s_lshl_b32 s0, s2, 7 // (op 3324) 
s_add_u32 s0, s0, 384 // (op 3324) 
v_add_u32 v0, v20, s0 // (op 3324) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3324) Load value

v_accvgpr_read v10, a28 // (op 3326) 
v_accvgpr_read v11, a29 // (op 3326) 
v_accvgpr_read v12, a30 // (op 3326) 
v_accvgpr_read v13, a31 // (op 3326) 
v_mul_f32 v6, v19, v10 // (op 3326) 
v_mul_f32 v7, v19, v11 // (op 3326) 
v_mul_f32 v8, v19, v12 // (op 3326) 
v_mul_f32 v9, v19, v13 // (op 3326) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3322) 
v_mul_f32 v11, v18, v15 // (op 3322) 
v_mul_f32 v12, v18, v16 // (op 3322) 
v_mul_f32 v13, v18, v17 // (op 3322) 

v_add_f32 v14, v10, v6 // (op 3323) 
v_add_f32 v15, v11, v7 // (op 3323) 
v_add_f32 v16, v12, v8 // (op 3323) 
v_add_f32 v17, v13, v9 // (op 3323) 




s_lshl_b32 s0, s4, 7 // (op 3317) 
s_add_u32 s0, s0, 384 // (op 3317) 
v_add_u32 v0, v21, s0 // (op 3317) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3317) Store value




v_add_u32 v0, 256, v20 // (op 3279) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3279) Load value

v_accvgpr_read v10, a16 // (op 3271) 
v_accvgpr_read v11, a17 // (op 3271) 
v_accvgpr_read v12, a18 // (op 3271) 
v_accvgpr_read v13, a19 // (op 3271) 
v_mul_f32 v6, v19, v10 // (op 3271) 
v_mul_f32 v7, v19, v11 // (op 3271) 
v_mul_f32 v8, v19, v12 // (op 3271) 
v_mul_f32 v9, v19, v13 // (op 3271) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3278) 
v_mul_f32 v11, v18, v15 // (op 3278) 
v_mul_f32 v12, v18, v16 // (op 3278) 
v_mul_f32 v13, v18, v17 // (op 3278) 

v_add_f32 v14, v10, v6 // (op 3272) 
v_add_f32 v15, v11, v7 // (op 3272) 
v_add_f32 v16, v12, v8 // (op 3272) 
v_add_f32 v17, v13, v9 // (op 3272) 




v_add_u32 v0, 256, v21 // (op 3274) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3274) Store value




s_lshl_b32 s0, s2, 7 // (op 3283) 
s_add_u32 s0, s0, 256 // (op 3283) 
v_add_u32 v0, v20, s0 // (op 3283) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3283) Load value

v_accvgpr_read v10, a20 // (op 3285) 
v_accvgpr_read v11, a21 // (op 3285) 
v_accvgpr_read v12, a22 // (op 3285) 
v_accvgpr_read v13, a23 // (op 3285) 
v_mul_f32 v6, v19, v10 // (op 3285) 
v_mul_f32 v7, v19, v11 // (op 3285) 
v_mul_f32 v8, v19, v12 // (op 3285) 
v_mul_f32 v9, v19, v13 // (op 3285) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3281) 
v_mul_f32 v11, v18, v15 // (op 3281) 
v_mul_f32 v12, v18, v16 // (op 3281) 
v_mul_f32 v13, v18, v17 // (op 3281) 

v_add_f32 v14, v10, v6 // (op 3282) 
v_add_f32 v15, v11, v7 // (op 3282) 
v_add_f32 v16, v12, v8 // (op 3282) 
v_add_f32 v17, v13, v9 // (op 3282) 




s_lshl_b32 s0, s4, 7 // (op 3276) 
s_add_u32 s0, s0, 256 // (op 3276) 
v_add_u32 v0, v21, s0 // (op 3276) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3276) Store value




v_add_u32 v0, 128, v20 // (op 3238) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3238) Load value

v_accvgpr_read v10, a8 // (op 3230) 
v_accvgpr_read v11, a9 // (op 3230) 
v_accvgpr_read v12, a10 // (op 3230) 
v_accvgpr_read v13, a11 // (op 3230) 
v_mul_f32 v6, v19, v10 // (op 3230) 
v_mul_f32 v7, v19, v11 // (op 3230) 
v_mul_f32 v8, v19, v12 // (op 3230) 
v_mul_f32 v9, v19, v13 // (op 3230) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3237) 
v_mul_f32 v11, v18, v15 // (op 3237) 
v_mul_f32 v12, v18, v16 // (op 3237) 
v_mul_f32 v13, v18, v17 // (op 3237) 

v_add_f32 v14, v10, v6 // (op 3231) 
v_add_f32 v15, v11, v7 // (op 3231) 
v_add_f32 v16, v12, v8 // (op 3231) 
v_add_f32 v17, v13, v9 // (op 3231) 




v_add_u32 v0, 128, v21 // (op 3233) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3233) Store value




s_lshl_b32 s0, s2, 7 // (op 3242) 
s_add_u32 s0, s0, 128 // (op 3242) 
v_add_u32 v0, v20, s0 // (op 3242) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3242) Load value

v_accvgpr_read v10, a12 // (op 3244) 
v_accvgpr_read v11, a13 // (op 3244) 
v_accvgpr_read v12, a14 // (op 3244) 
v_accvgpr_read v13, a15 // (op 3244) 
v_mul_f32 v6, v19, v10 // (op 3244) 
v_mul_f32 v7, v19, v11 // (op 3244) 
v_mul_f32 v8, v19, v12 // (op 3244) 
v_mul_f32 v9, v19, v13 // (op 3244) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3240) 
v_mul_f32 v11, v18, v15 // (op 3240) 
v_mul_f32 v12, v18, v16 // (op 3240) 
v_mul_f32 v13, v18, v17 // (op 3240) 

v_add_f32 v14, v10, v6 // (op 3241) 
v_add_f32 v15, v11, v7 // (op 3241) 
v_add_f32 v16, v12, v8 // (op 3241) 
v_add_f32 v17, v13, v9 // (op 3241) 




s_lshl_b32 s0, s4, 7 // (op 3235) 
s_add_u32 s0, s0, 128 // (op 3235) 
v_add_u32 v0, v21, s0 // (op 3235) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3235) Store value




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




s_lshl_b32 s0, s2, 7 // (op 3200) 
v_add_u32 v0, v20, s0 // (op 3200) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 3200) Load value

v_accvgpr_read v10, a4 // (op 3197) 
v_accvgpr_read v11, a5 // (op 3197) 
v_accvgpr_read v12, a6 // (op 3197) 
v_accvgpr_read v13, a7 // (op 3197) 
v_mul_f32 v6, v19, v10 // (op 3197) 
v_mul_f32 v7, v19, v11 // (op 3197) 
v_mul_f32 v8, v19, v12 // (op 3197) 
v_mul_f32 v9, v19, v13 // (op 3197) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 3201) 
v_mul_f32 v11, v18, v15 // (op 3201) 
v_mul_f32 v12, v18, v16 // (op 3201) 
v_mul_f32 v13, v18, v17 // (op 3201) 

v_add_f32 v14, v10, v6 // (op 3198) 
v_add_f32 v15, v11, v7 // (op 3198) 
v_add_f32 v16, v12, v8 // (op 3198) 
v_add_f32 v17, v13, v9 // (op 3198) 




s_lshl_b32 s0, s4, 7 // (op 3199) 
v_add_u32 v0, v21, s0 // (op 3199) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3199) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
  .amdhsa_next_free_vgpr 164
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 30464 // lds bytes
  .amdhsa_accum_offset 132
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
    macrotile: [128, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 4
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 30464
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 132
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

