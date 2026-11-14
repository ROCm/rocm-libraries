// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
.p2align 8
.type RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC,@function
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_exec_begin:


s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value
s_load_dwordx2 s[38:39], s[0:1], 128 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_lshrrev_b32 v0, 7, v1 // (op 9069) 
v_and_b32 v3, 15, v1 // (op 9069) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 9069) 


v_and_b32 v0, 48, v1 // (op 9069) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 9069) 


v_and_b32 v0, 63, v1 // (op 9127) 
v_and_b32 v4, 48, v1 // (op 9127) 
v_sub_u32 v5, v0, v4 // (op 9127) 

v_lshrrev_b32 v4, 4, v0 // (op 9127) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 9127) 


v_lshrrev_b32 v4, 6, v1 // (op 9127) 
v_and_b32 v5, 1, v4 // (op 9127) 

v_lshlrev_b32 v4, 4, v5 // (op 9127) 

v_lshl_add_u32 v5, v0, 6, v4 // (op 9127) 


v_lshlrev_b32 v0, 2, v5 // (op 9127) 

v_lshrrev_b32 v4, 3, v0 // (op 9127) 

v_lshrrev_b32 v0, 7, v1 // (op 9185) 
v_and_b32 v5, 15, v1 // (op 9185) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9185) 


v_and_b32 v0, 63, v1 // (op 9185) 
v_lshrrev_b32 v5, 4, v0 // (op 9185) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9185) 


v_lshrrev_b32 v5, 6, v1 // (op 9243) 
v_and_b32 v6, 1, v5 // (op 9243) 

v_and_b32 v5, 15, v1 // (op 9243) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9243) 


v_and_b32 v5, 63, v1 // (op 9243) 
v_lshrrev_b32 v6, 4, v5 // (op 9243) 

v_lshl_add_u32 v5, v6, 6, v7 // (op 9243) 


v_lshrrev_b32 v6, 7, v1 // (op 9295) 
v_and_b32 v7, 15, v1 // (op 9295) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9295) 


v_and_b32 v6, 48, v1 // (op 9295) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9295) 


v_and_b32 v6, 63, v1 // (op 9324) 
v_and_b32 v8, 48, v1 // (op 9324) 
v_sub_u32 v9, v6, v8 // (op 9324) 

v_lshrrev_b32 v8, 4, v6 // (op 9324) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 9324) 


v_lshrrev_b32 v8, 6, v1 // (op 9324) 
v_and_b32 v9, 1, v8 // (op 9324) 

v_lshlrev_b32 v8, 4, v9 // (op 9324) 

v_lshl_add_u32 v9, v6, 6, v8 // (op 9324) 


v_lshlrev_b32 v6, 2, v9 // (op 9324) 

v_lshrrev_b32 v8, 3, v6 // (op 9324) 

v_lshrrev_b32 v6, 7, v1 // (op 9353) 
v_and_b32 v9, 15, v1 // (op 9353) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9353) 


v_and_b32 v6, 63, v1 // (op 9353) 
v_lshrrev_b32 v9, 4, v6 // (op 9353) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 9353) 


v_lshrrev_b32 v9, 6, v1 // (op 9382) 
v_and_b32 v10, 1, v9 // (op 9382) 

v_and_b32 v9, 15, v1 // (op 9382) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9382) 


v_and_b32 v9, 63, v1 // (op 9382) 
v_lshrrev_b32 v10, 4, v9 // (op 9382) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 9382) 


v_lshrrev_b32 v10, 7, v1 // (op 9007) 
v_and_b32 v11, 63, v1 // (op 9007) 
v_lshrrev_b32 v12, 4, v11 // (op 9007) 
v_lshlrev_b32 v13, 2, v12 // (op 9007) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 9007) 


v_lshl_add_u32 v10, s16, 8, v12 // (op 9007) 

v_lshrrev_b32 v12, 2, v11 // (op 9007) 

v_and_b32 v11, 3, v12 // (op 9007) 

v_and_b32 v12, 3, v1 // (op 9007) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 9007) 


v_lshrrev_b32 v11, 6, v1 // (op 9007) 
v_and_b32 v12, 1, v11 // (op 9007) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 9007) 


v_lshl_add_u32 v12, s17, 6, v11 // (op 9007) 

v_mul_lo_u32 v11, v12, s2 // (op 9007) 

v_add_lshl_u32 v12, v10, v11, 1 // (op 9007) 


v_lshrrev_b32 v10, 7, v1 // (op 9264) 
v_and_b32 v11, 63, v1 // (op 9264) 
v_lshrrev_b32 v13, 4, v11 // (op 9264) 
s_mul_i32 s1, s2, s9 // (op 9009) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 9009) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 9009) least significant
s_mul_hi_u32 s19, s2, s8 // (op 9009) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 9009) most significant: sum
s_add_u32 s1, s1, s19 // (op 9009) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 9009) 

v_lshlrev_b32 v14, 2, v13 // (op 9264) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 9264) 


v_lshl_add_u32 v10, s16, 8, v13 // (op 9264) 

v_lshrrev_b32 v13, 2, v11 // (op 9264) 

v_and_b32 v11, 3, v13 // (op 9264) 

v_and_b32 v13, 3, v1 // (op 9264) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 9264) 


v_lshrrev_b32 v11, 6, v1 // (op 9264) 
v_and_b32 v13, 1, v11 // (op 9264) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 9264) 


v_lshl_add_u32 v13, s17, 6, v11 // (op 9264) 

v_mul_lo_u32 v11, v13, s4 // (op 9264) 

v_add_lshl_u32 v13, v10, v11, 1 // (op 9264) 


s_mov_b32 s40, s6 // (op 9009) reg expression
s_mov_b32 s41, s7 // (op 9009) reg expression
s_mov_b32 s42, s18 // (op 9009) reg expression
s_mov_b32 s43, 131072 // (op 9009) call()

s_mul_i32 s1, s4, s9 // (op 9266) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 9266) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 9266) least significant
s_mul_hi_u32 s7, s4, s8 // (op 9266) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 9266) most significant: sum
s_add_u32 s1, s1, s7 // (op 9266) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 9266) 

s_mov_b32 s44, s10 // (op 9266) reg expression
s_mov_b32 s45, s11 // (op 9266) reg expression
s_mov_b32 s46, s6 // (op 9266) reg expression
s_mov_b32 s47, 131072 // (op 9266) call()



v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 340) call()
v_accvgpr_write a5, 0.00000 // (op 340) call()
v_accvgpr_write a6, 0.00000 // (op 340) call()
v_accvgpr_write a7, 0.00000 // (op 340) call()
v_accvgpr_write a8, 0.00000 // (op 560) call()
v_accvgpr_write a9, 0.00000 // (op 560) call()
v_accvgpr_write a10, 0.00000 // (op 560) call()
v_accvgpr_write a11, 0.00000 // (op 560) call()
v_accvgpr_write a12, 0.00000 // (op 715) call()
v_accvgpr_write a13, 0.00000 // (op 715) call()
v_accvgpr_write a14, 0.00000 // (op 715) call()
v_accvgpr_write a15, 0.00000 // (op 715) call()
v_accvgpr_write a16, 0.00000 // (op 946) call()
v_accvgpr_write a17, 0.00000 // (op 946) call()
v_accvgpr_write a18, 0.00000 // (op 946) call()
v_accvgpr_write a19, 0.00000 // (op 946) call()
v_accvgpr_write a20, 0.00000 // (op 1101) call()
v_accvgpr_write a21, 0.00000 // (op 1101) call()
v_accvgpr_write a22, 0.00000 // (op 1101) call()
v_accvgpr_write a23, 0.00000 // (op 1101) call()
v_accvgpr_write a24, 0.00000 // (op 1332) call()
v_accvgpr_write a25, 0.00000 // (op 1332) call()
v_accvgpr_write a26, 0.00000 // (op 1332) call()
v_accvgpr_write a27, 0.00000 // (op 1332) call()
v_accvgpr_write a28, 0.00000 // (op 1487) call()
v_accvgpr_write a29, 0.00000 // (op 1487) call()
v_accvgpr_write a30, 0.00000 // (op 1487) call()
v_accvgpr_write a31, 0.00000 // (op 1487) call()
v_accvgpr_write a32, 0.00000 // (op 1718) call()
v_accvgpr_write a33, 0.00000 // (op 1718) call()
v_accvgpr_write a34, 0.00000 // (op 1718) call()
v_accvgpr_write a35, 0.00000 // (op 1718) call()
v_accvgpr_write a36, 0.00000 // (op 1873) call()
v_accvgpr_write a37, 0.00000 // (op 1873) call()
v_accvgpr_write a38, 0.00000 // (op 1873) call()
v_accvgpr_write a39, 0.00000 // (op 1873) call()
v_accvgpr_write a40, 0.00000 // (op 2104) call()
v_accvgpr_write a41, 0.00000 // (op 2104) call()
v_accvgpr_write a42, 0.00000 // (op 2104) call()
v_accvgpr_write a43, 0.00000 // (op 2104) call()
v_accvgpr_write a44, 0.00000 // (op 2259) call()
v_accvgpr_write a45, 0.00000 // (op 2259) call()
v_accvgpr_write a46, 0.00000 // (op 2259) call()
v_accvgpr_write a47, 0.00000 // (op 2259) call()
v_accvgpr_write a48, 0.00000 // (op 2490) call()
v_accvgpr_write a49, 0.00000 // (op 2490) call()
v_accvgpr_write a50, 0.00000 // (op 2490) call()
v_accvgpr_write a51, 0.00000 // (op 2490) call()
v_accvgpr_write a52, 0.00000 // (op 2645) call()
v_accvgpr_write a53, 0.00000 // (op 2645) call()
v_accvgpr_write a54, 0.00000 // (op 2645) call()
v_accvgpr_write a55, 0.00000 // (op 2645) call()
v_accvgpr_write a56, 0.00000 // (op 2876) call()
v_accvgpr_write a57, 0.00000 // (op 2876) call()
v_accvgpr_write a58, 0.00000 // (op 2876) call()
v_accvgpr_write a59, 0.00000 // (op 2876) call()
v_accvgpr_write a60, 0.00000 // (op 3031) call()
v_accvgpr_write a61, 0.00000 // (op 3031) call()
v_accvgpr_write a62, 0.00000 // (op 3031) call()
v_accvgpr_write a63, 0.00000 // (op 3031) call()
v_and_b32 v14, 4294967288, v1 // (op 8769) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 8769) 

v_and_b32 v14, -8, v1 // (op 8769) 
v_sub_u32 v16, v1, v14 // (op 8769) 

v_lshlrev_b32 v14, 4, v16 // (op 8769) 

v_mul_lo_u32 v16, v15, s12 // (op 8769) 
v_add_u32 v16, v16, v14 // (op 8769) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 8771) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8771) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8771) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8771) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8771) most significant: sum
s_add_u32 s1, s1, s7 // (op 8771) most significant: sum


s_mov_b32 s8, s20 // (op 8771) reg expression
s_mov_b32 s9, s21 // (op 8771) reg expression
s_mov_b32 s10, s0 // (op 8771) reg expression
s_mov_b32 s11, 131072 // (op 8771) call()

v_and_b32 v14, 4294967288, v1 // (op 8788) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 8788) 

v_and_b32 v14, -8, v1 // (op 8788) 
v_sub_u32 v17, v1, v14 // (op 8788) 

v_lshlrev_b32 v14, 4, v17 // (op 8788) 

v_mul_lo_u32 v17, v15, s12 // (op 8788) 
v_add_u32 v17, v17, v14 // (op 8788) 


s_mul_i32 s1, s12, s23 // (op 8790) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8790) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8790) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8790) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8790) most significant: sum
s_add_u32 s1, s1, s7 // (op 8790) most significant: sum


s_mov_b32 s8, s20 // (op 8790) reg expression
s_mov_b32 s9, s21 // (op 8790) reg expression
s_mov_b32 s10, s0 // (op 8790) reg expression
s_mov_b32 s11, 131072 // (op 8790) call()

v_and_b32 v14, -2, v1 // (op 8833) 
v_sub_u32 v15, v1, v14 // (op 8833) 

v_lshlrev_b32 v14, 5, v15 // (op 8833) 

v_lshl_add_u32 v15, s17, 6, v14 // (op 8833) 

v_mov_b32 v19, 0 // (op 8833) convert
v_mov_b32 v18, v15 // (op 8833) convert

v_lshrrev_b32 v14, 1, v1 // (op 8833) 
v_mul_lo_u32 v21, v14, s25 // (op 8833) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 8833) most significant: high of low * low
v_mul_lo_u32 v20, v14, s24 // (op 8833) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 8833) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 8833) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 8833) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 8833) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 8833) 

v_mov_b32 v14, v18 // (op 8833) convert

s_mul_i32 s1, s24, s29 // (op 8835) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 8835) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 8835) least significant
s_mul_hi_u32 s7, s24, s28 // (op 8835) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8835) most significant: sum
s_add_u32 s1, s1, s7 // (op 8835) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8835) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8835) 

s_mov_b32 s48, s26 // (op 8835) reg expression
s_mov_b32 s49, s27 // (op 8835) reg expression
s_mov_b32 s50, s0 // (op 8835) reg expression
s_mov_b32 s51, 131072 // (op 8835) call()

v_and_b32 v15, -2, v1 // (op 8852) 
v_sub_u32 v18, v1, v15 // (op 8852) 

v_lshlrev_b32 v15, 5, v18 // (op 8852) 

v_lshl_add_u32 v18, s17, 6, v15 // (op 8852) 

v_mov_b32 v21, 0 // (op 8852) convert
v_mov_b32 v20, v18 // (op 8852) convert

v_lshrrev_b32 v15, 1, v1 // (op 8852) 
v_mul_lo_u32 v19, v15, s25 // (op 8852) most significant: low of low * high
v_mul_hi_u32 v22, v15, s24 // (op 8852) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 8852) least significant: low of low * low
v_add_u32 v19, v19, v22 // (op 8852) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 8852) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 8852) most significant half



v_lshlrev_b64 v[20:21], 2, v[18:19] // (op 8852) 

v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 8852) 

v_mov_b32 v15, v18 // (op 8852) convert

s_mul_i32 s1, s24, s29 // (op 8854) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 8854) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 8854) least significant
s_mul_hi_u32 s7, s24, s28 // (op 8854) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8854) most significant: sum
s_add_u32 s1, s1, s7 // (op 8854) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8854) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8854) 

s_mov_b32 s48, s26 // (op 8854) reg expression
s_mov_b32 s49, s27 // (op 8854) reg expression
s_mov_b32 s50, s0 // (op 8854) reg expression
s_mov_b32 s51, 131072 // (op 8854) call()

v_lshl_add_u32 v18, s16, 8, v1 // (op 8897) 
v_mul_lo_u32 v19, v18, s30 // (op 8897) 

s_mul_i32 s1, s30, s23 // (op 8899) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 8899) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 8899) least significant
s_mul_hi_u32 s7, s30, s22 // (op 8899) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8899) most significant: sum
s_add_u32 s1, s1, s7 // (op 8899) most significant: sum


s_mov_b32 s52, s32 // (op 8899) reg expression
s_mov_b32 s53, s33 // (op 8899) reg expression
s_mov_b32 s54, s0 // (op 8899) reg expression
s_mov_b32 s55, 131072 // (op 8899) call()

v_lshl_add_u32 v18, s16, 8, v1 // (op 8916) 
v_mul_lo_u32 v20, v18, s30 // (op 8916) 

s_mul_i32 s1, s30, s23 // (op 8918) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 8918) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 8918) least significant
s_mul_hi_u32 s7, s30, s22 // (op 8918) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8918) most significant: sum
s_add_u32 s1, s1, s7 // (op 8918) most significant: sum


s_mov_b32 s52, s32 // (op 8918) reg expression
s_mov_b32 s53, s33 // (op 8918) reg expression
s_mov_b32 s54, s0 // (op 8918) reg expression
s_mov_b32 s55, 131072 // (op 8918) call()

v_and_b32 v18, -64, v1 // (op 8961) 
v_sub_u32 v21, v1, v18 // (op 8961) 

v_lshl_add_u32 v18, s17, 6, v21 // (op 8961) 

v_lshrrev_b32 v21, 6, v1 // (op 8961) 
v_mul_lo_u32 v22, v21, s34 // (op 8961) 
v_add_u32 v22, v22, v18 // (op 8961) 


s_mul_i32 s1, s34, s39 // (op 8963) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 8963) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 8963) least significant
s_mul_hi_u32 s7, s34, s38 // (op 8963) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8963) most significant: sum
s_add_u32 s1, s1, s7 // (op 8963) most significant: sum


s_mov_b32 s56, s36 // (op 8963) reg expression
s_mov_b32 s57, s37 // (op 8963) reg expression
s_mov_b32 s58, s0 // (op 8963) reg expression
s_mov_b32 s59, 131072 // (op 8963) call()

v_and_b32 v18, -64, v1 // (op 8980) 
v_sub_u32 v21, v1, v18 // (op 8980) 

v_lshl_add_u32 v18, s17, 6, v21 // (op 8980) 

v_lshrrev_b32 v21, 6, v1 // (op 8980) 
v_mul_lo_u32 v23, v21, s34 // (op 8980) 
v_add_u32 v23, v23, v18 // (op 8980) 


s_mul_i32 s1, s34, s39 // (op 8982) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 8982) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 8982) least significant
s_mul_hi_u32 s7, s34, s38 // (op 8982) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8982) most significant: sum
s_add_u32 s1, s1, s7 // (op 8982) most significant: sum


s_mov_b32 s56, s36 // (op 8982) reg expression
s_mov_b32 s57, s37 // (op 8982) reg expression
s_mov_b32 s58, s0 // (op 8982) reg expression
s_mov_b32 s59, 131072 // (op 8982) call()



v_mov_b32 v18, v14 // (op 8289) reg expression

buffer_load_dwordx4 v[24:27], v18, s[48:51], 0 offen // (op 8289) Load value



v_mov_b32 v18, v22 // (op 8292) reg expression

buffer_load_ubyte v21, v18, s[56:59], 0 offen offset: 0 // (op 8292) Load value


v_mov_b64 v[28:29], s[12:13] // (op 8295) convert

v_mov_b32 v18, v16 // (op 8295) 
buffer_load_dwordx4 v[30:33], v18, s[8:11], 0 offen // (op 8295) Load value
v_add_u32 v18, v18, v28 // (op 8295) 
buffer_load_dwordx4 v[34:37], v18, s[8:11], 0 offen // (op 8295) Load value
v_add_u32 v18, v18, v28 // (op 8295) 
buffer_load_dwordx4 v[38:41], v18, s[8:11], 0 offen // (op 8295) Load value
v_add_u32 v18, v18, v28 // (op 8295) 
buffer_load_dwordx4 v[42:45], v18, s[8:11], 0 offen // (op 8295) Load value
v_add_u32 v18, v18, v28 // (op 8295) 
buffer_load_dwordx4 v[46:49], v18, s[8:11], 0 offen // (op 8295) Load value
v_add_u32 v18, v18, v28 // (op 8295) 
buffer_load_dwordx4 v[50:53], v18, s[8:11], 0 offen // (op 8295) Load value
v_add_u32 v18, v18, v28 // (op 8295) 
buffer_load_dwordx4 v[54:57], v18, s[8:11], 0 offen // (op 8295) Load value
v_add_u32 v18, v18, v28 // (op 8295) 
buffer_load_dwordx4 v[58:61], v18, s[8:11], 0 offen // (op 8295) Load value





buffer_load_dword v18, v19, s[52:55], 0 offen // (op 8298) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 8301) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 8301) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 8301) 

v_add_u32 v28, v14, s0 // (op 8301) 


buffer_load_dwordx4 v[62:65], v28, s[48:51], 0 offen // (op 8301) Load value




s_lshl_b32 s0, s34, 2 // (op 8304) 
v_add_u32 v14, v22, s0 // (op 8304) 


buffer_load_ubyte v28, v14, s[56:59], 0 offen offset: 0 // (op 8304) Load value



v_mov_b64 v[66:67], s[12:13] // (op 8307) convert

v_mov_b32 v14, v16 // (op 8307) 
buffer_load_dwordx4 v[68:71], v14, s[8:11], 0 offen offset:128 // (op 8307) Load value
v_add_u32 v14, v14, v66 // (op 8307) 
buffer_load_dwordx4 v[72:75], v14, s[8:11], 0 offen offset:128 // (op 8307) Load value
v_add_u32 v14, v14, v66 // (op 8307) 
buffer_load_dwordx4 v[76:79], v14, s[8:11], 0 offen offset:128 // (op 8307) Load value
v_add_u32 v14, v14, v66 // (op 8307) 
buffer_load_dwordx4 v[80:83], v14, s[8:11], 0 offen offset:128 // (op 8307) Load value
v_add_u32 v14, v14, v66 // (op 8307) 
buffer_load_dwordx4 v[84:87], v14, s[8:11], 0 offen offset:128 // (op 8307) Load value
v_add_u32 v14, v14, v66 // (op 8307) 
buffer_load_dwordx4 v[88:91], v14, s[8:11], 0 offen offset:128 // (op 8307) Load value
v_add_u32 v14, v14, v66 // (op 8307) 
buffer_load_dwordx4 v[92:95], v14, s[8:11], 0 offen offset:128 // (op 8307) Load value
v_add_u32 v14, v14, v66 // (op 8307) 
buffer_load_dwordx4 v[96:99], v14, s[8:11], 0 offen offset:128 // (op 8307) Load value






buffer_load_dword v14, v19, s[52:55], 0 offen offset:4 // (op 8310) Load value

v_and_b32 v16, -2, v1 // (op 9110) 
v_sub_u32 v19, v1, v16 // (op 9110) 

v_lshlrev_b32 v16, 5, v19 // (op 9110) 

v_lshrrev_b32 v19, 1, v1 // (op 9110) 
v_lshl_add_u32 v22, v19, 6, v16 // (op 9110) 


v_lshlrev_b32 v16, 2, v22 // (op 9110) 

v_lshrrev_b32 v19, 3, v16 // (op 9110) 




s_waitcnt vmcnt(21)
ds_write_b128 v19, v[24:27] // (op 8312) Store local data 

v_and_b32 v16, -64, v1 // (op 9226) 
v_sub_u32 v19, v1, v16 // (op 9226) 

v_and_b32 v16, 4294967232, v1 // (op 9226) 
v_add_u32 v22, v16, v19 // (op 9226) 





s_waitcnt vmcnt(20)
ds_write_b8 v22, v21 offset:4096 // (op 8313) Store local data 

v_and_b32 v16, -8, v1 // (op 9052) 
v_sub_u32 v19, v1, v16 // (op 9052) 

v_lshlrev_b32 v16, 4, v19 // (op 9052) 

v_lshrrev_b32 v19, 3, v1 // (op 9052) 
v_lshl_add_u32 v22, v19, 10, v16 // (op 9052) 





s_waitcnt vmcnt(19)
ds_write_b128 v22, v[30:33] offset:4352 // (op 8314) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v22, v[34:37] offset:4480 // (op 8314) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v22, v[38:41] offset:4608 // (op 8314) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v22, v[42:45] offset:4736 // (op 8314) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v22, v[46:49] offset:4864 // (op 8314) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v22, v[50:53] offset:4992 // (op 8314) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v22, v[54:57] offset:5120 // (op 8314) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v22, v[58:61] offset:5248 // (op 8314) Store local data 

v_lshlrev_b32 v16, 2, v1 // (op 9168) 



s_waitcnt vmcnt(11)
ds_write_b32 v16, v18 offset:37120 // (op 8315) Store local data 

v_and_b32 v16, -8, v1 // (op 9036) 
v_sub_u32 v19, v1, v16 // (op 9036) 

v_lshlrev_b32 v16, 4, v19 // (op 9036) 

v_lshrrev_b32 v19, 3, v1 // (op 9036) 
v_lshl_add_u32 v22, v19, 10, v16 // (op 9036) 


v_and_b32 v16, -2, v1 // (op 9094) 
v_sub_u32 v19, v1, v16 // (op 9094) 

v_lshlrev_b32 v16, 5, v19 // (op 9094) 

v_lshrrev_b32 v19, 1, v1 // (op 9094) 
v_lshl_add_u32 v29, v19, 6, v16 // (op 9094) 


v_lshlrev_b32 v16, 2, v29 // (op 9094) 

v_lshrrev_b32 v19, 3, v16 // (op 9094) 

v_lshlrev_b32 v16, 2, v1 // (op 9152) 
v_and_b32 v29, -64, v1 // (op 9210) 
v_sub_u32 v66, v1, v29 // (op 9210) 

v_and_b32 v29, 4294967232, v1 // (op 9210) 
v_add_u32 v67, v29, v66 // (op 9210) 


v_and_b32 v29, -8, v1 // (op 9278) 
v_sub_u32 v66, v1, v29 // (op 9278) 

v_lshlrev_b32 v29, 4, v66 // (op 9278) 

v_lshrrev_b32 v66, 3, v1 // (op 9278) 
v_lshl_add_u32 v100, v66, 10, v29 // (op 9278) 


v_and_b32 v29, -2, v1 // (op 9307) 
v_sub_u32 v66, v1, v29 // (op 9307) 

v_lshlrev_b32 v29, 5, v66 // (op 9307) 

v_lshrrev_b32 v66, 1, v1 // (op 9307) 
v_lshl_add_u32 v101, v66, 6, v29 // (op 9307) 


v_lshlrev_b32 v29, 2, v101 // (op 9307) 

v_lshrrev_b32 v66, 3, v29 // (op 9307) 

v_lshlrev_b32 v29, 2, v1 // (op 9336) 
v_and_b32 v101, -64, v1 // (op 9365) 
v_sub_u32 v102, v1, v101 // (op 9365) 

v_and_b32 v101, 4294967232, v1 // (op 9365) 
v_add_u32 v103, v101, v102 // (op 9365) 


s_mov_b32 s0, 0 // (op 2909) call()
s_ashr_i64 s[6:7], s[28:29], 7 // (op 2908) 
s_mov_b32 s1, s6 // (op 2908) convert

s_and_b32 s6, s1, 4294967294 // (op 2908) 

s_cmp_lt_u32 s0, s6 // (op 2908) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_2908 // (op 2908) Condition: Top (jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_2908 if false)
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_2908:


s_barrier  // (op 9469) 
s_add_u32 s0, s0, 2 // (op 2910) 



v_mov_b32 v101, v15 // (op 10) 
s_lshl_b64 s[6:7], s[24:25], 7 // (op 10) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 10) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 10) 

s_lshl_b32 s1, s6, 1 // (op 10) 

v_add_u32 v102, v101, s1 // (op 10) 


buffer_load_dwordx4 v[24:27], v102, s[48:51], 0 offen // (op 10) Load value

ds_read_b128 v[104:107], v3 offset:4352 // (op 72) Load local data 


ds_read_b128 v[108:111], v3 offset:4416 // (op 72) Load local data 
v_mov_b32 v102, v23 // (op 25) 
s_barrier  // (op 9482) 
s_lshl_b32 s1, s34, 3 // (op 25) 



v_add_u32 v112, v102, s1 // (op 25) 


buffer_load_ubyte v21, v112, s[56:59], 0 offen offset: 0 // (op 25) Load value

ds_read_b64_tr_b4 v[112:113], v4 // (op 77) Transpose load local data 

v_mov_b64 v[116:117], s[12:13] // (op 4) convert

v_mov_b32 v118, v17 // (op 4) 
v_mov_b32 v119, v118 // (op 4) 
ds_read_b64_tr_b4 v[114:115], v4 offset:512 // (op 77) Transpose load local data 
buffer_load_dwordx4 v[30:33], v119, s[8:11], 0 offen offset:256 // (op 4) Load value
s_barrier  // (op 9496) 
v_add_u32 v119, v119, v116 // (op 4) 



ds_read_u8 v120, v0 offset:37120 // (op 82) Load local data 
buffer_load_dwordx4 v[34:37], v119, s[8:11], 0 offen offset:256 // (op 4) Load value
s_barrier  // (op 9510) 
v_add_u32 v119, v119, v116 // (op 4) 


ds_read_u8 v121, v5 offset:4096 // (op 87) Load local data 
buffer_load_dwordx4 v[38:41], v119, s[8:11], 0 offen offset:256 // (op 4) Load value



v_add_u32 v119, v119, v116 // (op 4) 
ds_read_b64_tr_b4 v[122:123], v4 offset:16 // (op 375) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v4 offset:528 // (op 375) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[112:115], a[0:3], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 90) 


ds_read_u8 v126, v5 offset:4128 // (op 379) Load local data 
buffer_load_dwordx4 v[42:45], v119, s[8:11], 0 offen offset:256 // (op 4) Load value



v_add_u32 v119, v119, v116 // (op 4) 
ds_read_b128 v[128:131], v3 offset:8448 // (op 626) Load local data 
ds_read_b128 v[132:135], v3 offset:8512 // (op 626) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[122:125], a[4:7], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 380) 





ds_read_u8 v120, v0 offset:37248 // (op 629) Load local data 
buffer_load_dwordx4 v[46:49], v119, s[8:11], 0 offen offset:256 // (op 4) Load value



v_add_u32 v119, v119, v116 // (op 4) 
ds_read_b128 v[104:107], v3 offset:12544 // (op 1012) Load local data 
ds_read_b128 v[108:111], v3 offset:12608 // (op 1012) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[128:135], v[112:115], a[8:11], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 677) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[128:135], v[122:125], a[12:15], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 631) 
buffer_load_dwordx4 v[50:53], v119, s[8:11], 0 offen offset:256 // (op 4) Load value


v_add_u32 v119, v119, v116 // (op 4) 
ds_read_u8 v120, v0 offset:37376 // (op 1015) Load local data 
buffer_load_dwordx4 v[54:57], v119, s[8:11], 0 offen offset:256 // (op 4) Load value



v_add_u32 v119, v119, v116 // (op 4) 
ds_read_b128 v[128:131], v3 offset:16640 // (op 1398) Load local data 
ds_read_b128 v[132:135], v3 offset:16704 // (op 1398) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:111], v[112:115], a[16:19], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1063) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:111], v[122:125], a[20:23], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1017) 
ds_read_u8 v127, v0 offset:37504 // (op 1401) Load local data 





s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:135], v[112:115], a[24:27], v127, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1449) 
ds_read_b128 v[104:107], v3 offset:20736 // (op 1784) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:135], v[122:125], a[28:31], v127, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1403) 
ds_read_b128 v[108:111], v3 offset:20800 // (op 1784) Load local data 





ds_read_u8 v120, v0 offset:37632 // (op 1787) Load local data 
buffer_load_dwordx4 v[58:61], v119, s[8:11], 0 offen offset:256 // (op 4) Load value







v_mov_b32 v119, v20 // (op 17) 

ds_read_b128 v[128:131], v3 offset:24832 // (op 2170) Load local data 
ds_read_b128 v[132:135], v3 offset:24896 // (op 2170) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:111], v[112:115], a[32:35], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1835) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:111], v[122:125], a[36:39], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1789) 
buffer_load_dword v18, v119, s[52:55], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 9533) 
ds_read_u8 v120, v0 offset:37760 // (op 2173) Load local data 






s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[128:135], v[112:115], a[40:43], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2221) 
ds_read_b128 v[104:107], v3 offset:28928 // (op 2556) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[128:135], v[122:125], a[44:47], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2175) 
ds_read_b128 v[108:111], v3 offset:28992 // (op 2556) Load local data 





ds_read_u8 v120, v0 offset:37888 // (op 2559) Load local data 
s_waitcnt vmcnt(21)
ds_write_b128 v66, v[62:65] offset:38144 // (op 257) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:111], v[112:115], a[48:51], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2607) 
s_barrier  // (op 9545) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:111], v[122:125], a[52:55], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2561) 





ds_read_b128 v[104:107], v3 offset:33024 // (op 2942) Load local data 
ds_read_b128 v[108:111], v3 offset:33088 // (op 2942) Load local data 
s_waitcnt vmcnt(20)
ds_write_b8 v103, v28 offset:42240 // (op 259) Store local data 



ds_read_u8 v120, v0 offset:38016 // (op 2945) Load local data 
s_barrier  // (op 9527) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:111], v[112:115], a[56:59], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2993) 





s_waitcnt vmcnt(19)
ds_write_b128 v100, v[68:71] offset:42496 // (op 247) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:111], v[122:125], a[60:63], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2947) 
s_waitcnt vmcnt(18)
ds_write_b128 v100, v[72:75] offset:42624 // (op 247) Store local data 




s_waitcnt vmcnt(17)
ds_write_b128 v100, v[76:79] offset:42752 // (op 247) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v100, v[80:83] offset:42880 // (op 247) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v100, v[84:87] offset:43008 // (op 247) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v100, v[88:91] offset:43136 // (op 247) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v100, v[92:95] offset:43264 // (op 247) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v100, v[96:99] offset:43392 // (op 247) Store local data 
s_barrier  // (op 9539) 



v_add_u32 v104, 75264, v29 // (op 258) 
s_waitcnt vmcnt(11)
ds_write_b32 v104, v14 // (op 258) Store local data 



s_lshl_b64 s[6:7], s[24:25], 7 // (op 260) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 260) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 260) 

s_mul_i32 s1, 3, s6 // (op 260) 

v_add_u32 v104, v101, s1 // (op 260) 


buffer_load_dwordx4 v[62:65], v104, s[48:51], 0 offen // (op 260) Load value

s_waitcnt lgkmcnt(1)
s_barrier  // (op 9524) 






s_lshl_b32 s1, s34, 2 // (op 262) 
s_mul_i32 s6, 3, s1 // (op 262) 

v_add_u32 v101, v102, s6 // (op 262) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 8844) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 8844) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 8844) 

s_lshl_b32 s1, s6, 1 // (op 8844) 

v_add_u32 v15, v15, s1 // (op 8844) 

ds_read_b128 v[104:107], v7 offset:42496 // (op 249) Load local data 
ds_read_b128 v[108:111], v7 offset:42560 // (op 249) Load local data 
buffer_load_ubyte v28, v101, s[56:59], 0 offen offset: 0 // (op 262) Load value


s_barrier  // (op 9530) 




s_lshl_b32 s1, s34, 3 // (op 8972) 
v_add_u32 v23, v23, s1 // (op 8972) 

v_mov_b64 v[112:113], s[12:13] // (op 246) convert

v_mov_b32 v101, v118 // (op 246) 
ds_read_b64_tr_b4 v[114:115], v8 offset:38144 // (op 251) Transpose load local data 
buffer_load_dwordx4 v[68:71], v101, s[8:11], 0 offen offset:384 // (op 246) Load value
v_add_u32 v101, v101, v112 // (op 246) 
ds_read_b64_tr_b4 v[116:117], v8 offset:38656 // (op 251) Transpose load local data 
buffer_load_dwordx4 v[72:75], v101, s[8:11], 0 offen offset:384 // (op 246) Load value
v_add_u32 v101, v101, v112 // (op 246) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 9536) 
buffer_load_dwordx4 v[76:79], v101, s[8:11], 0 offen offset:384 // (op 246) Load value



v_add_u32 v101, v101, v112 // (op 246) 
v_add_u32 v102, 75264, v6 // (op 253) 
ds_read_u8 v120, v102 // (op 253) Load local data 

buffer_load_dwordx4 v[80:83], v101, s[8:11], 0 offen offset:384 // (op 246) Load value
s_barrier  // (op 9542) 
v_add_u32 v101, v101, v112 // (op 246) 


ds_read_u8 v102, v9 offset:42240 // (op 255) Load local data 
buffer_load_dwordx4 v[84:87], v101, s[8:11], 0 offen offset:384 // (op 246) Load value



v_add_u32 v101, v101, v112 // (op 246) 
ds_read_b64_tr_b4 v[122:123], v8 offset:38160 // (op 385) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v8 offset:38672 // (op 385) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[114:117], a[0:3], v120, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 256) 
buffer_load_dwordx4 v[88:91], v101, s[8:11], 0 offen offset:384 // (op 246) Load value


s_barrier  // (op 9606) 
v_add_u32 v101, v101, v112 // (op 246) 
ds_read_u8 v121, v9 offset:42272 // (op 389) Load local data 
buffer_load_dwordx4 v[92:95], v101, s[8:11], 0 offen offset:384 // (op 246) Load value
v_add_u32 v101, v101, v112 // (op 246) 



s_barrier  // (op 9610) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[122:125], a[4:7], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 381) 
ds_read_b128 v[126:129], v7 offset:46592 // (op 634) Load local data 


ds_read_b128 v[130:133], v7 offset:46656 // (op 634) Load local data 
buffer_load_dwordx4 v[96:99], v101, s[8:11], 0 offen offset:384 // (op 246) Load value






v_add_u32 v101, 75392, v6 // (op 637) 



v_add_u32 v17, 256, v17 // (op 8780) 
ds_read_u8 v118, v101 // (op 637) Load local data 

buffer_load_dword v14, v119, s[52:55], 0 offen offset:12 // (op 261) Load value




ds_read_b128 v[104:107], v7 offset:50688 // (op 1020) Load local data 
s_barrier  // (op 9485) 
ds_read_b128 v[108:111], v7 offset:50752 // (op 1020) Load local data 






v_add_u32 v20, 8, v20 // (op 8908) 
v_add_u32 v101, 75520, v6 // (op 1023) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[126:133], v[114:117], a[8:11], v118, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 678) 
ds_read_u8 v112, v101 // (op 1023) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[126:133], v[122:125], a[12:15], v118, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 632) 
ds_read_b128 v[134:137], v7 offset:54784 // (op 1406) Load local data 


ds_read_b128 v[138:141], v7 offset:54848 // (op 1406) Load local data 



v_add_u32 v101, 75648, v6 // (op 1409) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:111], v[114:117], a[16:19], v112, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1064) 
ds_read_u8 v113, v101 // (op 1409) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:111], v[122:125], a[20:23], v112, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1018) 
ds_read_b128 v[126:129], v7 offset:58880 // (op 1792) Load local data 


ds_read_b128 v[130:133], v7 offset:58944 // (op 1792) Load local data 



v_add_u32 v101, 75776, v6 // (op 1795) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[134:141], v[114:117], a[24:27], v113, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1450) 
ds_read_u8 v104, v101 // (op 1795) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[134:141], v[122:125], a[28:31], v113, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1404) 
ds_read_b128 v[142:145], v7 offset:62976 // (op 2178) Load local data 


ds_read_b128 v[146:149], v7 offset:63040 // (op 2178) Load local data 



v_add_u32 v101, 75904, v6 // (op 2181) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[126:133], v[114:117], a[32:35], v104, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1836) 
ds_read_u8 v105, v101 // (op 2181) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[126:133], v[122:125], a[36:39], v104, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1790) 
v_add_u32 v101, 67072, v7 // (op 2564) 


ds_read_b128 v[106:109], v101 // (op 2564) Load local data 

v_add_u32 v101, 67136, v7 // (op 2564) 
ds_read_b128 v[110:113], v101 // (op 2564) Load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[142:149], v[114:117], a[40:43], v105, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2222) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[142:149], v[122:125], a[44:47], v105, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2176) 
v_add_u32 v101, 76032, v6 // (op 2567) 


ds_read_u8 v104, v101 // (op 2567) Load local data 

s_waitcnt vmcnt(21)
ds_write_b128 v19, v[24:27] // (op 13) Store local data 



v_add_u32 v101, 71168, v7 // (op 2950) 
s_barrier  // (op 9513) 
ds_read_b128 v[126:129], v101 // (op 2950) Load local data 




v_add_u32 v101, 71232, v7 // (op 2950) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[106:113], v[114:117], a[48:51], v104, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2608) 
ds_read_b128 v[130:133], v101 // (op 2950) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[106:113], v[122:125], a[52:55], v104, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2562) 





s_barrier  // (op 9666) 
v_add_u32 v101, 76160, v6 // (op 2953) 
ds_read_u8 v104, v101 // (op 2953) Load local data 

s_waitcnt vmcnt(20)
ds_write_b8 v67, v21 offset:4096 // (op 30) Store local data 
s_barrier  // (op 9670) 
s_barrier  // (op 9472) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[126:133], v[114:117], a[56:59], v104, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2994) 





s_waitcnt vmcnt(19)
ds_write_b128 v22, v[30:33] offset:4352 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[126:133], v[122:125], a[60:63], v104, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2948) 
s_waitcnt vmcnt(18)
ds_write_b128 v22, v[34:37] offset:4480 // (op 6) Store local data 




s_waitcnt vmcnt(17)
ds_write_b128 v22, v[38:41] offset:4608 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v22, v[42:45] offset:4736 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v22, v[46:49] offset:4864 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v22, v[50:53] offset:4992 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v22, v[54:57] offset:5120 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v22, v[58:61] offset:5248 // (op 6) Store local data 
s_barrier  // (op 9499) 



s_waitcnt vmcnt(11)
ds_write_b32 v16, v18 offset:37120 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[28:29], 7 // (op 2908) 
s_mov_b32 s1, s6 // (op 2908) convert

s_and_b32 s6, s1, 4294967294 // (op 2908) 

s_cmp_lt_u32 s0, s6 // (op 2908) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_2908 // (op 2908) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_2908 if true)
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_2908:

















v_and_b32 v15, 4294967288, v1 // (op 8728) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8728) 
v_lshl_add_u32 v16, s16, 8, v15 // (op 8728) 

s_lshr_b32 s6, s0, 1 // (op 8728) 

v_and_b32 v15, -8, v1 // (op 8728) 
v_sub_u32 v17, v1, v15 // (op 8728) 

v_lshlrev_b32 v15, 4, v17 // (op 8728) 

v_lshl_add_u32 v17, s6, 8, v15 // (op 8728) 


v_mul_lo_u32 v15, v16, s12 // (op 8728) 
v_add_u32 v15, v15, v17 // (op 8728) 


s_mul_i32 s1, s12, s23 // (op 8730) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8730) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8730) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8730) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8730) most significant: sum
s_add_u32 s1, s1, s7 // (op 8730) most significant: sum


s_mov_b32 s8, s20 // (op 8730) reg expression
s_mov_b32 s9, s21 // (op 8730) reg expression
s_mov_b32 s10, s0 // (op 8730) reg expression
s_mov_b32 s11, 131072 // (op 8730) call()

v_lshrrev_b32 v16, 1, v1 // (op 8810) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8810) 
v_and_b32 v17, -2, v1 // (op 8810) 
s_lshr_b32 s6, s0, 1 // (op 8810) 

v_sub_u32 v18, v1, v17 // (op 8810) 

v_lshl_add_u32 v17, s6, 8, v16 // (op 8810) 


v_lshlrev_b32 v16, 5, v18 // (op 8810) 

v_lshl_add_u32 v18, s17, 6, v16 // (op 8810) 

v_mov_b32 v67, 0 // (op 8810) convert
v_mov_b32 v66, v18 // (op 8810) convert

v_mul_lo_u32 v19, v17, s25 // (op 8810) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 8810) most significant: high of low * low
v_mul_lo_u32 v18, v17, s24 // (op 8810) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 8810) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v66 // (op 8810) least significant half
v_addc_co_u32 v19, s[0:1], v19, v67, s[0:1] // (op 8810) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 8810) 

v_lshrrev_b64 v[66:67], 3, v[16:17] // (op 8810) 

v_mov_b32 v16, v66 // (op 8810) convert

s_mul_i32 s1, s24, s29 // (op 8812) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 8812) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 8812) least significant
s_mul_hi_u32 s7, s24, s28 // (op 8812) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8812) most significant: sum
s_add_u32 s1, s1, s7 // (op 8812) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8812) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8812) 

s_mov_b32 s48, s26 // (op 8812) reg expression
s_mov_b32 s49, s27 // (op 8812) reg expression
s_mov_b32 s50, s0 // (op 8812) reg expression
s_mov_b32 s51, 131072 // (op 8812) call()

v_lshl_add_u32 v17, s16, 8, v1 // (op 8874) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8874) 
s_lshr_b32 s6, s0, 1 // (op 8874) 

s_lshl_b32 s7, s6, 3 // (op 8874) 

v_mul_lo_u32 v18, v17, s30 // (op 8874) 
v_add_u32 v18, v18, s7 // (op 8874) 


s_mul_i32 s1, s30, s23 // (op 8876) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 8876) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 8876) least significant
s_mul_hi_u32 s7, s30, s22 // (op 8876) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8876) most significant: sum
s_add_u32 s1, s1, s7 // (op 8876) most significant: sum


s_mov_b32 s52, s32 // (op 8876) reg expression
s_mov_b32 s53, s33 // (op 8876) reg expression
s_mov_b32 s54, s0 // (op 8876) reg expression
s_mov_b32 s55, 131072 // (op 8876) call()

v_lshrrev_b32 v17, 6, v1 // (op 8938) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8938) 
v_and_b32 v19, -64, v1 // (op 8938) 
s_lshr_b32 s6, s0, 1 // (op 8938) 

v_sub_u32 v20, v1, v19 // (op 8938) 

v_lshl_add_u32 v19, s6, 3, v17 // (op 8938) 


v_lshl_add_u32 v17, s17, 6, v20 // (op 8938) 

v_mul_lo_u32 v20, v19, s34 // (op 8938) 
v_add_u32 v20, v20, v17 // (op 8938) 


s_mul_i32 s1, s34, s39 // (op 8940) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 8940) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 8940) least significant
s_mul_hi_u32 s7, s34, s38 // (op 8940) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8940) most significant: sum
s_add_u32 s1, s1, s7 // (op 8940) most significant: sum


s_mov_b32 s56, s36 // (op 8940) reg expression
s_mov_b32 s57, s37 // (op 8940) reg expression
s_mov_b32 s58, s0 // (op 8940) reg expression
s_mov_b32 s59, 131072 // (op 8940) call()

v_and_b32 v17, -8, v1 // (op 9021) 
v_sub_u32 v19, v1, v17 // (op 9021) 

v_lshlrev_b32 v17, 4, v19 // (op 9021) 

v_lshrrev_b32 v19, 3, v1 // (op 9021) 
v_lshl_add_u32 v21, v19, 10, v17 // (op 9021) 


v_and_b32 v17, -2, v1 // (op 9081) 
v_sub_u32 v19, v1, v17 // (op 9081) 

v_lshlrev_b32 v17, 5, v19 // (op 9081) 

v_lshrrev_b32 v19, 1, v1 // (op 9081) 
v_lshl_add_u32 v22, v19, 6, v17 // (op 9081) 


v_lshlrev_b32 v17, 2, v22 // (op 9081) 

v_lshrrev_b32 v19, 3, v17 // (op 9081) 

v_lshlrev_b32 v17, 2, v1 // (op 9139) 
v_and_b32 v22, -64, v1 // (op 9197) 
v_sub_u32 v23, v1, v22 // (op 9197) 

v_and_b32 v22, 4294967232, v1 // (op 9197) 
v_add_u32 v24, v22, v23 // (op 9197) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2883) 
s_and_b32 s6, s0, 4294967294 // (op 2883) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 2882) 
s_mov_b32 s7, s0 // (op 2882) convert

s_cmp_lt_u32 s6, s7 // (op 2882) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_2882 // (op 2882) Condition: Top (jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_2882 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_2882:

s_add_u32 s6, s6, 1 // (op 2884) 
v_mov_b64 v[22:23], s[12:13] // (op 180) convert

v_mov_b32 v14, v15 // (op 180) 
v_mov_b32 v25, v14 // (op 180) 
buffer_load_dwordx4 v[26:29], v25, s[8:11], 0 offen // (op 180) Load value
v_add_u32 v25, v25, v22 // (op 180) 
buffer_load_dwordx4 v[30:33], v25, s[8:11], 0 offen // (op 180) Load value
v_add_u32 v25, v25, v22 // (op 180) 
buffer_load_dwordx4 v[34:37], v25, s[8:11], 0 offen // (op 180) Load value
v_add_u32 v25, v25, v22 // (op 180) 
buffer_load_dwordx4 v[38:41], v25, s[8:11], 0 offen // (op 180) Load value
v_add_u32 v25, v25, v22 // (op 180) 
buffer_load_dwordx4 v[42:45], v25, s[8:11], 0 offen // (op 180) Load value
v_add_u32 v25, v25, v22 // (op 180) 
buffer_load_dwordx4 v[46:49], v25, s[8:11], 0 offen // (op 180) Load value
v_add_u32 v25, v25, v22 // (op 180) 
buffer_load_dwordx4 v[50:53], v25, s[8:11], 0 offen // (op 180) Load value
v_add_u32 v25, v25, v22 // (op 180) 
buffer_load_dwordx4 v[54:57], v25, s[8:11], 0 offen // (op 180) Load value



v_add_u32 v15, 128, v15 // (op 8718) 


s_barrier  // (op 9475) 
v_mov_b32 v14, v16 // (op 194) 



v_mov_b32 v25, v14 // (op 194) reg expression

buffer_load_dwordx4 v[58:61], v25, s[48:51], 0 offen // (op 194) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 8802) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 8802) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 8802) 

v_add_u32 v16, v16, s0 // (op 8802) 

s_waitcnt vmcnt(8)
ds_write_b128 v21, v[26:29] offset:4352 // (op 181) Store local data 


v_mov_b32 v14, v18 // (op 195) 

s_waitcnt vmcnt(7)
ds_write_b128 v21, v[30:33] offset:4480 // (op 181) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v21, v[34:37] offset:4608 // (op 181) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v21, v[38:41] offset:4736 // (op 181) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v21, v[42:45] offset:4864 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v21, v[46:49] offset:4992 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[50:53] offset:5120 // (op 181) Store local data 
buffer_load_dword v25, v14, s[52:55], 0 offen // (op 195) Load value

v_add_u32 v18, 4, v18 // (op 8866) 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[54:57] offset:5248 // (op 181) Store local data 



v_mov_b32 v14, v20 // (op 196) 
s_barrier  // (op 9488) 
v_mov_b32 v22, v14 // (op 196) reg expression




buffer_load_ubyte v23, v22, s[56:59], 0 offen offset: 0 // (op 196) Load value


s_lshl_b32 s7, s34, 2 // (op 8930) 
v_add_u32 v20, v20, s7 // (op 8930) 

s_waitcnt vmcnt(2)
ds_write_b128 v19, v[58:61] // (op 191) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 9478) 



s_barrier  // (op 9502) 
ds_read_b128 v[26:29], v3 offset:4352 // (op 183) Load local data 



ds_read_b128 v[30:33], v3 offset:4416 // (op 183) Load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v17, v25 offset:37120 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9492) 



s_barrier  // (op 9516) 
ds_read_b64_tr_b4 v[34:35], v4 // (op 185) Transpose load local data 



ds_read_b64_tr_b4 v[36:37], v4 offset:512 // (op 185) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v23 offset:4096 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9506) 



ds_read_u8 v14, v0 offset:37120 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9520) 


ds_read_u8 v25, v5 offset:4096 // (op 189) Load local data 



ds_read_b64_tr_b4 v[38:39], v4 offset:16 // (op 354) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:37], a[0:3], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 190) 
ds_read_b64_tr_b4 v[40:41], v4 offset:528 // (op 354) Transpose load local data 


ds_read_u8 v22, v5 offset:4128 // (op 358) Load local data 



ds_read_b128 v[42:45], v3 offset:8448 // (op 572) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[38:41], a[4:7], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 359) 
ds_read_b128 v[46:49], v3 offset:8512 // (op 572) Load local data 





ds_read_u8 v14, v0 offset:37248 // (op 575) Load local data 



ds_read_b128 v[26:29], v3 offset:12544 // (op 958) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:49], v[34:37], a[8:11], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 577) 
ds_read_b128 v[30:33], v3 offset:12608 // (op 958) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[38:41], a[12:15], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 608) 
ds_read_u8 v23, v0 offset:37376 // (op 961) Load local data 





ds_read_b128 v[42:45], v3 offset:16640 // (op 1344) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:33], v[34:37], a[16:19], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 963) 
ds_read_b128 v[46:49], v3 offset:16704 // (op 1344) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:33], v[38:41], a[20:23], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 994) 
ds_read_u8 v14, v0 offset:37504 // (op 1347) Load local data 





ds_read_b128 v[26:29], v3 offset:20736 // (op 1730) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:49], v[34:37], a[24:27], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1349) 
ds_read_b128 v[30:33], v3 offset:20800 // (op 1730) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:49], v[38:41], a[28:31], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1380) 
ds_read_u8 v23, v0 offset:37632 // (op 1733) Load local data 





ds_read_b128 v[42:45], v3 offset:24832 // (op 2116) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:33], v[34:37], a[32:35], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1735) 
ds_read_b128 v[46:49], v3 offset:24896 // (op 2116) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[26:33], v[38:41], a[36:39], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1766) 
ds_read_u8 v14, v0 offset:37760 // (op 2119) Load local data 





ds_read_b128 v[26:29], v3 offset:28928 // (op 2502) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[42:49], v[34:37], a[40:43], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2121) 
ds_read_b128 v[30:33], v3 offset:28992 // (op 2502) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[42:49], v[38:41], a[44:47], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2152) 
ds_read_u8 v23, v0 offset:37888 // (op 2505) Load local data 





ds_read_b128 v[42:45], v3 offset:33024 // (op 2888) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[26:33], v[34:37], a[48:51], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2507) 
ds_read_b128 v[46:49], v3 offset:33088 // (op 2888) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[26:33], v[38:41], a[52:55], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2538) 
ds_read_u8 v14, v0 offset:38016 // (op 2891) Load local data 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[42:49], v[34:37], a[56:59], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2893) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[42:49], v[38:41], a[60:63], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2924) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 2882) 
s_mov_b32 s7, s0 // (op 2882) convert

s_cmp_lt_u32 s6, s7 // (op 2882) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_2882 // (op 2882) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_2882 if true)
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_2882:
























s_barrier  // (op 9650) 








v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_lshrrev_b32 v6, 16, v14 // (op 56) 
v_lshrrev_b32 v7, 16, v15 // (op 56) 
v_lshrrev_b32 v8, 16, v16 // (op 56) 
v_lshrrev_b32 v9, 16, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 3577) 
v_accvgpr_read v7, a5 // (op 3577) 
v_accvgpr_read v8, a6 // (op 3577) 
v_accvgpr_read v9, a7 // (op 3577) 
v_mul_f32 v14, v11, v6 // (op 3577) 
v_mul_f32 v15, v11, v7 // (op 3577) 
v_mul_f32 v16, v11, v8 // (op 3577) 
v_mul_f32 v17, v11, v9 // (op 3577) 

v_lshrrev_b32 v6, 16, v14 // (op 3579) 
v_lshrrev_b32 v7, 16, v15 // (op 3579) 
v_lshrrev_b32 v8, 16, v16 // (op 3579) 
v_lshrrev_b32 v9, 16, v17 // (op 3579) 


v_pack_b32_f16 v4, v6, v7 // (op 3580) 
v_pack_b32_f16 v5, v8, v9 // (op 3580) 

s_lshl_b32 s0, s4, 6 // (op 3580) 
v_add_u32 v0, v13, s0 // (op 3580) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3580) Store value



v_accvgpr_read v6, a8 // (op 3603) 
v_accvgpr_read v7, a9 // (op 3603) 
v_accvgpr_read v8, a10 // (op 3603) 
v_accvgpr_read v9, a11 // (op 3603) 
v_mul_f32 v14, v11, v6 // (op 3603) 
v_mul_f32 v15, v11, v7 // (op 3603) 
v_mul_f32 v16, v11, v8 // (op 3603) 
v_mul_f32 v17, v11, v9 // (op 3603) 

v_lshrrev_b32 v6, 16, v14 // (op 3605) 
v_lshrrev_b32 v7, 16, v15 // (op 3605) 
v_lshrrev_b32 v8, 16, v16 // (op 3605) 
v_lshrrev_b32 v9, 16, v17 // (op 3605) 


v_pack_b32_f16 v4, v6, v7 // (op 3607) 
v_pack_b32_f16 v5, v8, v9 // (op 3607) 

v_add_u32 v0, 64, v13 // (op 3607) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3607) Store value



v_accvgpr_read v6, a12 // (op 3613) 
v_accvgpr_read v7, a13 // (op 3613) 
v_accvgpr_read v8, a14 // (op 3613) 
v_accvgpr_read v9, a15 // (op 3613) 
v_mul_f32 v14, v11, v6 // (op 3613) 
v_mul_f32 v15, v11, v7 // (op 3613) 
v_mul_f32 v16, v11, v8 // (op 3613) 
v_mul_f32 v17, v11, v9 // (op 3613) 

v_lshrrev_b32 v6, 16, v14 // (op 3615) 
v_lshrrev_b32 v7, 16, v15 // (op 3615) 
v_lshrrev_b32 v8, 16, v16 // (op 3615) 
v_lshrrev_b32 v9, 16, v17 // (op 3615) 


v_pack_b32_f16 v4, v6, v7 // (op 3609) 
v_pack_b32_f16 v5, v8, v9 // (op 3609) 

s_lshl_b32 s0, s4, 6 // (op 3609) 
s_add_u32 s0, s0, 64 // (op 3609) 
v_add_u32 v0, v13, s0 // (op 3609) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3609) Store value



v_accvgpr_read v6, a16 // (op 3643) 
v_accvgpr_read v7, a17 // (op 3643) 
v_accvgpr_read v8, a18 // (op 3643) 
v_accvgpr_read v9, a19 // (op 3643) 
v_mul_f32 v14, v11, v6 // (op 3643) 
v_mul_f32 v15, v11, v7 // (op 3643) 
v_mul_f32 v16, v11, v8 // (op 3643) 
v_mul_f32 v17, v11, v9 // (op 3643) 

v_lshrrev_b32 v6, 16, v14 // (op 3645) 
v_lshrrev_b32 v7, 16, v15 // (op 3645) 
v_lshrrev_b32 v8, 16, v16 // (op 3645) 
v_lshrrev_b32 v9, 16, v17 // (op 3645) 


v_pack_b32_f16 v4, v6, v7 // (op 3647) 
v_pack_b32_f16 v5, v8, v9 // (op 3647) 

v_add_u32 v0, 128, v13 // (op 3647) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3647) Store value



v_accvgpr_read v6, a20 // (op 3653) 
v_accvgpr_read v7, a21 // (op 3653) 
v_accvgpr_read v8, a22 // (op 3653) 
v_accvgpr_read v9, a23 // (op 3653) 
v_mul_f32 v14, v11, v6 // (op 3653) 
v_mul_f32 v15, v11, v7 // (op 3653) 
v_mul_f32 v16, v11, v8 // (op 3653) 
v_mul_f32 v17, v11, v9 // (op 3653) 

v_lshrrev_b32 v6, 16, v14 // (op 3655) 
v_lshrrev_b32 v7, 16, v15 // (op 3655) 
v_lshrrev_b32 v8, 16, v16 // (op 3655) 
v_lshrrev_b32 v9, 16, v17 // (op 3655) 


v_pack_b32_f16 v4, v6, v7 // (op 3649) 
v_pack_b32_f16 v5, v8, v9 // (op 3649) 

s_lshl_b32 s0, s4, 6 // (op 3649) 
s_add_u32 s0, s0, 128 // (op 3649) 
v_add_u32 v0, v13, s0 // (op 3649) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3649) Store value



v_accvgpr_read v6, a24 // (op 3683) 
v_accvgpr_read v7, a25 // (op 3683) 
v_accvgpr_read v8, a26 // (op 3683) 
v_accvgpr_read v9, a27 // (op 3683) 
v_mul_f32 v14, v11, v6 // (op 3683) 
v_mul_f32 v15, v11, v7 // (op 3683) 
v_mul_f32 v16, v11, v8 // (op 3683) 
v_mul_f32 v17, v11, v9 // (op 3683) 

v_lshrrev_b32 v6, 16, v14 // (op 3685) 
v_lshrrev_b32 v7, 16, v15 // (op 3685) 
v_lshrrev_b32 v8, 16, v16 // (op 3685) 
v_lshrrev_b32 v9, 16, v17 // (op 3685) 


v_pack_b32_f16 v4, v6, v7 // (op 3687) 
v_pack_b32_f16 v5, v8, v9 // (op 3687) 

v_add_u32 v0, 192, v13 // (op 3687) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3687) Store value



v_accvgpr_read v6, a28 // (op 3693) 
v_accvgpr_read v7, a29 // (op 3693) 
v_accvgpr_read v8, a30 // (op 3693) 
v_accvgpr_read v9, a31 // (op 3693) 
v_mul_f32 v14, v11, v6 // (op 3693) 
v_mul_f32 v15, v11, v7 // (op 3693) 
v_mul_f32 v16, v11, v8 // (op 3693) 
v_mul_f32 v17, v11, v9 // (op 3693) 

v_lshrrev_b32 v6, 16, v14 // (op 3695) 
v_lshrrev_b32 v7, 16, v15 // (op 3695) 
v_lshrrev_b32 v8, 16, v16 // (op 3695) 
v_lshrrev_b32 v9, 16, v17 // (op 3695) 


v_pack_b32_f16 v4, v6, v7 // (op 3689) 
v_pack_b32_f16 v5, v8, v9 // (op 3689) 

s_lshl_b32 s0, s4, 6 // (op 3689) 
s_add_u32 s0, s0, 192 // (op 3689) 
v_add_u32 v0, v13, s0 // (op 3689) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3689) Store value



v_accvgpr_read v6, a32 // (op 3723) 
v_accvgpr_read v7, a33 // (op 3723) 
v_accvgpr_read v8, a34 // (op 3723) 
v_accvgpr_read v9, a35 // (op 3723) 
v_mul_f32 v14, v11, v6 // (op 3723) 
v_mul_f32 v15, v11, v7 // (op 3723) 
v_mul_f32 v16, v11, v8 // (op 3723) 
v_mul_f32 v17, v11, v9 // (op 3723) 

v_lshrrev_b32 v6, 16, v14 // (op 3725) 
v_lshrrev_b32 v7, 16, v15 // (op 3725) 
v_lshrrev_b32 v8, 16, v16 // (op 3725) 
v_lshrrev_b32 v9, 16, v17 // (op 3725) 


v_pack_b32_f16 v4, v6, v7 // (op 3727) 
v_pack_b32_f16 v5, v8, v9 // (op 3727) 

v_add_u32 v0, 256, v13 // (op 3727) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3727) Store value



v_accvgpr_read v6, a36 // (op 3733) 
v_accvgpr_read v7, a37 // (op 3733) 
v_accvgpr_read v8, a38 // (op 3733) 
v_accvgpr_read v9, a39 // (op 3733) 
v_mul_f32 v14, v11, v6 // (op 3733) 
v_mul_f32 v15, v11, v7 // (op 3733) 
v_mul_f32 v16, v11, v8 // (op 3733) 
v_mul_f32 v17, v11, v9 // (op 3733) 

v_lshrrev_b32 v6, 16, v14 // (op 3735) 
v_lshrrev_b32 v7, 16, v15 // (op 3735) 
v_lshrrev_b32 v8, 16, v16 // (op 3735) 
v_lshrrev_b32 v9, 16, v17 // (op 3735) 


v_pack_b32_f16 v4, v6, v7 // (op 3729) 
v_pack_b32_f16 v5, v8, v9 // (op 3729) 

s_lshl_b32 s0, s4, 6 // (op 3729) 
s_add_u32 s0, s0, 256 // (op 3729) 
v_add_u32 v0, v13, s0 // (op 3729) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3729) Store value



v_accvgpr_read v6, a40 // (op 3763) 
v_accvgpr_read v7, a41 // (op 3763) 
v_accvgpr_read v8, a42 // (op 3763) 
v_accvgpr_read v9, a43 // (op 3763) 
v_mul_f32 v14, v11, v6 // (op 3763) 
v_mul_f32 v15, v11, v7 // (op 3763) 
v_mul_f32 v16, v11, v8 // (op 3763) 
v_mul_f32 v17, v11, v9 // (op 3763) 

v_lshrrev_b32 v6, 16, v14 // (op 3765) 
v_lshrrev_b32 v7, 16, v15 // (op 3765) 
v_lshrrev_b32 v8, 16, v16 // (op 3765) 
v_lshrrev_b32 v9, 16, v17 // (op 3765) 


v_pack_b32_f16 v4, v6, v7 // (op 3767) 
v_pack_b32_f16 v5, v8, v9 // (op 3767) 

v_add_u32 v0, 320, v13 // (op 3767) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3767) Store value



v_accvgpr_read v6, a44 // (op 3773) 
v_accvgpr_read v7, a45 // (op 3773) 
v_accvgpr_read v8, a46 // (op 3773) 
v_accvgpr_read v9, a47 // (op 3773) 
v_mul_f32 v14, v11, v6 // (op 3773) 
v_mul_f32 v15, v11, v7 // (op 3773) 
v_mul_f32 v16, v11, v8 // (op 3773) 
v_mul_f32 v17, v11, v9 // (op 3773) 

v_lshrrev_b32 v6, 16, v14 // (op 3775) 
v_lshrrev_b32 v7, 16, v15 // (op 3775) 
v_lshrrev_b32 v8, 16, v16 // (op 3775) 
v_lshrrev_b32 v9, 16, v17 // (op 3775) 


v_pack_b32_f16 v4, v6, v7 // (op 3769) 
v_pack_b32_f16 v5, v8, v9 // (op 3769) 

s_lshl_b32 s0, s4, 6 // (op 3769) 
s_add_u32 s0, s0, 320 // (op 3769) 
v_add_u32 v0, v13, s0 // (op 3769) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3769) Store value



v_accvgpr_read v6, a48 // (op 3803) 
v_accvgpr_read v7, a49 // (op 3803) 
v_accvgpr_read v8, a50 // (op 3803) 
v_accvgpr_read v9, a51 // (op 3803) 
v_mul_f32 v14, v11, v6 // (op 3803) 
v_mul_f32 v15, v11, v7 // (op 3803) 
v_mul_f32 v16, v11, v8 // (op 3803) 
v_mul_f32 v17, v11, v9 // (op 3803) 

v_lshrrev_b32 v6, 16, v14 // (op 3805) 
v_lshrrev_b32 v7, 16, v15 // (op 3805) 
v_lshrrev_b32 v8, 16, v16 // (op 3805) 
v_lshrrev_b32 v9, 16, v17 // (op 3805) 


v_pack_b32_f16 v4, v6, v7 // (op 3807) 
v_pack_b32_f16 v5, v8, v9 // (op 3807) 

v_add_u32 v0, 384, v13 // (op 3807) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3807) Store value



v_accvgpr_read v6, a52 // (op 3813) 
v_accvgpr_read v7, a53 // (op 3813) 
v_accvgpr_read v8, a54 // (op 3813) 
v_accvgpr_read v9, a55 // (op 3813) 
v_mul_f32 v14, v11, v6 // (op 3813) 
v_mul_f32 v15, v11, v7 // (op 3813) 
v_mul_f32 v16, v11, v8 // (op 3813) 
v_mul_f32 v17, v11, v9 // (op 3813) 

v_lshrrev_b32 v6, 16, v14 // (op 3815) 
v_lshrrev_b32 v7, 16, v15 // (op 3815) 
v_lshrrev_b32 v8, 16, v16 // (op 3815) 
v_lshrrev_b32 v9, 16, v17 // (op 3815) 


v_pack_b32_f16 v4, v6, v7 // (op 3809) 
v_pack_b32_f16 v5, v8, v9 // (op 3809) 

s_lshl_b32 s0, s4, 6 // (op 3809) 
s_add_u32 s0, s0, 384 // (op 3809) 
v_add_u32 v0, v13, s0 // (op 3809) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3809) Store value



v_accvgpr_read v6, a56 // (op 3843) 
v_accvgpr_read v7, a57 // (op 3843) 
v_accvgpr_read v8, a58 // (op 3843) 
v_accvgpr_read v9, a59 // (op 3843) 
v_mul_f32 v14, v11, v6 // (op 3843) 
v_mul_f32 v15, v11, v7 // (op 3843) 
v_mul_f32 v16, v11, v8 // (op 3843) 
v_mul_f32 v17, v11, v9 // (op 3843) 

v_lshrrev_b32 v6, 16, v14 // (op 3845) 
v_lshrrev_b32 v7, 16, v15 // (op 3845) 
v_lshrrev_b32 v8, 16, v16 // (op 3845) 
v_lshrrev_b32 v9, 16, v17 // (op 3845) 


v_pack_b32_f16 v4, v6, v7 // (op 3847) 
v_pack_b32_f16 v5, v8, v9 // (op 3847) 

v_add_u32 v0, 448, v13 // (op 3847) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3847) Store value



v_accvgpr_read v6, a60 // (op 3853) 
v_accvgpr_read v7, a61 // (op 3853) 
v_accvgpr_read v8, a62 // (op 3853) 
v_accvgpr_read v9, a63 // (op 3853) 
v_mul_f32 v14, v11, v6 // (op 3853) 
v_mul_f32 v15, v11, v7 // (op 3853) 
v_mul_f32 v16, v11, v8 // (op 3853) 
v_mul_f32 v17, v11, v9 // (op 3853) 

v_lshrrev_b32 v6, 16, v14 // (op 3855) 
v_lshrrev_b32 v7, 16, v15 // (op 3855) 
v_lshrrev_b32 v8, 16, v16 // (op 3855) 
v_lshrrev_b32 v9, 16, v17 // (op 3855) 


v_pack_b32_f16 v4, v6, v7 // (op 3849) 
v_pack_b32_f16 v5, v8, v9 // (op 3849) 

s_lshl_b32 s0, s4, 6 // (op 3849) 
s_add_u32 s0, s0, 448 // (op 3849) 
v_add_u32 v0, v13, s0 // (op 3849) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3849) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 448, v12 // (op 4257) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4257) Load value

v_accvgpr_read v4, a56 // (op 4248) 
v_accvgpr_read v5, a57 // (op 4248) 
v_accvgpr_read v6, a58 // (op 4248) 
v_accvgpr_read v7, a59 // (op 4248) 
v_mul_f32 v14, v11, v4 // (op 4248) 
v_mul_f32 v15, v11, v5 // (op 4248) 
v_mul_f32 v16, v11, v6 // (op 4248) 
v_mul_f32 v17, v11, v7 // (op 4248) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4256) 
v_lshrrev_b32 v7, 16, v8 // (op 4256) 
v_lshlrev_b32 v6, 16, v6 // (op 4256) 
v_lshlrev_b32 v7, 16, v7 // (op 4256) 
v_mul_f32 v18, v10, v6 // (op 4256) 
v_mul_f32 v19, v10, v7 // (op 4256) 
v_and_b32 v6, 65535, v9 // (op 4256) 
v_lshrrev_b32 v7, 16, v9 // (op 4256) 
v_lshlrev_b32 v6, 16, v6 // (op 4256) 
v_lshlrev_b32 v7, 16, v7 // (op 4256) 
v_mul_f32 v20, v10, v6 // (op 4256) 
v_mul_f32 v21, v10, v7 // (op 4256) 


v_add_f32 v6, v18, v14 // (op 4249) 
v_add_f32 v7, v19, v15 // (op 4249) 
v_add_f32 v8, v20, v16 // (op 4249) 
v_add_f32 v9, v21, v17 // (op 4249) 


v_lshrrev_b32 v14, 16, v6 // (op 4250) 
v_lshrrev_b32 v15, 16, v7 // (op 4250) 
v_lshrrev_b32 v16, 16, v8 // (op 4250) 
v_lshrrev_b32 v17, 16, v9 // (op 4250) 


v_pack_b32_f16 v8, v14, v15 // (op 4252) 
v_pack_b32_f16 v9, v16, v17 // (op 4252) 

v_add_u32 v0, 448, v13 // (op 4252) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4252) Store value





s_lshl_b32 s0, s2, 6 // (op 4262) 
s_add_u32 s0, s0, 448 // (op 4262) 
v_add_u32 v0, v12, s0 // (op 4262) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4262) Load value

v_accvgpr_read v4, a60 // (op 4264) 
v_accvgpr_read v5, a61 // (op 4264) 
v_accvgpr_read v6, a62 // (op 4264) 
v_accvgpr_read v7, a63 // (op 4264) 
v_mul_f32 v14, v11, v4 // (op 4264) 
v_mul_f32 v15, v11, v5 // (op 4264) 
v_mul_f32 v16, v11, v6 // (op 4264) 
v_mul_f32 v17, v11, v7 // (op 4264) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4259) 
v_lshrrev_b32 v7, 16, v8 // (op 4259) 
v_lshlrev_b32 v6, 16, v6 // (op 4259) 
v_lshlrev_b32 v7, 16, v7 // (op 4259) 
v_mul_f32 v18, v10, v6 // (op 4259) 
v_mul_f32 v19, v10, v7 // (op 4259) 
v_and_b32 v6, 65535, v9 // (op 4259) 
v_lshrrev_b32 v7, 16, v9 // (op 4259) 
v_lshlrev_b32 v6, 16, v6 // (op 4259) 
v_lshlrev_b32 v7, 16, v7 // (op 4259) 
v_mul_f32 v20, v10, v6 // (op 4259) 
v_mul_f32 v21, v10, v7 // (op 4259) 


v_add_f32 v6, v18, v14 // (op 4260) 
v_add_f32 v7, v19, v15 // (op 4260) 
v_add_f32 v8, v20, v16 // (op 4260) 
v_add_f32 v9, v21, v17 // (op 4260) 


v_lshrrev_b32 v14, 16, v6 // (op 4261) 
v_lshrrev_b32 v15, 16, v7 // (op 4261) 
v_lshrrev_b32 v16, 16, v8 // (op 4261) 
v_lshrrev_b32 v17, 16, v9 // (op 4261) 


v_pack_b32_f16 v8, v14, v15 // (op 4254) 
v_pack_b32_f16 v9, v16, v17 // (op 4254) 

s_lshl_b32 s0, s4, 6 // (op 4254) 
s_add_u32 s0, s0, 448 // (op 4254) 
v_add_u32 v0, v13, s0 // (op 4254) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4254) Store value





v_add_u32 v0, 384, v12 // (op 4212) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4212) Load value

v_accvgpr_read v4, a48 // (op 4203) 
v_accvgpr_read v5, a49 // (op 4203) 
v_accvgpr_read v6, a50 // (op 4203) 
v_accvgpr_read v7, a51 // (op 4203) 
v_mul_f32 v14, v11, v4 // (op 4203) 
v_mul_f32 v15, v11, v5 // (op 4203) 
v_mul_f32 v16, v11, v6 // (op 4203) 
v_mul_f32 v17, v11, v7 // (op 4203) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4211) 
v_lshrrev_b32 v7, 16, v8 // (op 4211) 
v_lshlrev_b32 v6, 16, v6 // (op 4211) 
v_lshlrev_b32 v7, 16, v7 // (op 4211) 
v_mul_f32 v18, v10, v6 // (op 4211) 
v_mul_f32 v19, v10, v7 // (op 4211) 
v_and_b32 v6, 65535, v9 // (op 4211) 
v_lshrrev_b32 v7, 16, v9 // (op 4211) 
v_lshlrev_b32 v6, 16, v6 // (op 4211) 
v_lshlrev_b32 v7, 16, v7 // (op 4211) 
v_mul_f32 v20, v10, v6 // (op 4211) 
v_mul_f32 v21, v10, v7 // (op 4211) 


v_add_f32 v6, v18, v14 // (op 4204) 
v_add_f32 v7, v19, v15 // (op 4204) 
v_add_f32 v8, v20, v16 // (op 4204) 
v_add_f32 v9, v21, v17 // (op 4204) 


v_lshrrev_b32 v14, 16, v6 // (op 4205) 
v_lshrrev_b32 v15, 16, v7 // (op 4205) 
v_lshrrev_b32 v16, 16, v8 // (op 4205) 
v_lshrrev_b32 v17, 16, v9 // (op 4205) 


v_pack_b32_f16 v8, v14, v15 // (op 4207) 
v_pack_b32_f16 v9, v16, v17 // (op 4207) 

v_add_u32 v0, 384, v13 // (op 4207) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4207) Store value





s_lshl_b32 s0, s2, 6 // (op 4217) 
s_add_u32 s0, s0, 384 // (op 4217) 
v_add_u32 v0, v12, s0 // (op 4217) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4217) Load value

v_accvgpr_read v4, a52 // (op 4219) 
v_accvgpr_read v5, a53 // (op 4219) 
v_accvgpr_read v6, a54 // (op 4219) 
v_accvgpr_read v7, a55 // (op 4219) 
v_mul_f32 v14, v11, v4 // (op 4219) 
v_mul_f32 v15, v11, v5 // (op 4219) 
v_mul_f32 v16, v11, v6 // (op 4219) 
v_mul_f32 v17, v11, v7 // (op 4219) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4214) 
v_lshrrev_b32 v7, 16, v8 // (op 4214) 
v_lshlrev_b32 v6, 16, v6 // (op 4214) 
v_lshlrev_b32 v7, 16, v7 // (op 4214) 
v_mul_f32 v18, v10, v6 // (op 4214) 
v_mul_f32 v19, v10, v7 // (op 4214) 
v_and_b32 v6, 65535, v9 // (op 4214) 
v_lshrrev_b32 v7, 16, v9 // (op 4214) 
v_lshlrev_b32 v6, 16, v6 // (op 4214) 
v_lshlrev_b32 v7, 16, v7 // (op 4214) 
v_mul_f32 v20, v10, v6 // (op 4214) 
v_mul_f32 v21, v10, v7 // (op 4214) 


v_add_f32 v6, v18, v14 // (op 4215) 
v_add_f32 v7, v19, v15 // (op 4215) 
v_add_f32 v8, v20, v16 // (op 4215) 
v_add_f32 v9, v21, v17 // (op 4215) 


v_lshrrev_b32 v14, 16, v6 // (op 4216) 
v_lshrrev_b32 v15, 16, v7 // (op 4216) 
v_lshrrev_b32 v16, 16, v8 // (op 4216) 
v_lshrrev_b32 v17, 16, v9 // (op 4216) 


v_pack_b32_f16 v8, v14, v15 // (op 4209) 
v_pack_b32_f16 v9, v16, v17 // (op 4209) 

s_lshl_b32 s0, s4, 6 // (op 4209) 
s_add_u32 s0, s0, 384 // (op 4209) 
v_add_u32 v0, v13, s0 // (op 4209) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4209) Store value





v_add_u32 v0, 320, v12 // (op 4167) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4167) Load value

v_accvgpr_read v4, a40 // (op 4158) 
v_accvgpr_read v5, a41 // (op 4158) 
v_accvgpr_read v6, a42 // (op 4158) 
v_accvgpr_read v7, a43 // (op 4158) 
v_mul_f32 v14, v11, v4 // (op 4158) 
v_mul_f32 v15, v11, v5 // (op 4158) 
v_mul_f32 v16, v11, v6 // (op 4158) 
v_mul_f32 v17, v11, v7 // (op 4158) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4166) 
v_lshrrev_b32 v7, 16, v8 // (op 4166) 
v_lshlrev_b32 v6, 16, v6 // (op 4166) 
v_lshlrev_b32 v7, 16, v7 // (op 4166) 
v_mul_f32 v18, v10, v6 // (op 4166) 
v_mul_f32 v19, v10, v7 // (op 4166) 
v_and_b32 v6, 65535, v9 // (op 4166) 
v_lshrrev_b32 v7, 16, v9 // (op 4166) 
v_lshlrev_b32 v6, 16, v6 // (op 4166) 
v_lshlrev_b32 v7, 16, v7 // (op 4166) 
v_mul_f32 v20, v10, v6 // (op 4166) 
v_mul_f32 v21, v10, v7 // (op 4166) 


v_add_f32 v6, v18, v14 // (op 4159) 
v_add_f32 v7, v19, v15 // (op 4159) 
v_add_f32 v8, v20, v16 // (op 4159) 
v_add_f32 v9, v21, v17 // (op 4159) 


v_lshrrev_b32 v14, 16, v6 // (op 4160) 
v_lshrrev_b32 v15, 16, v7 // (op 4160) 
v_lshrrev_b32 v16, 16, v8 // (op 4160) 
v_lshrrev_b32 v17, 16, v9 // (op 4160) 


v_pack_b32_f16 v8, v14, v15 // (op 4162) 
v_pack_b32_f16 v9, v16, v17 // (op 4162) 

v_add_u32 v0, 320, v13 // (op 4162) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4162) Store value





s_lshl_b32 s0, s2, 6 // (op 4172) 
s_add_u32 s0, s0, 320 // (op 4172) 
v_add_u32 v0, v12, s0 // (op 4172) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4172) Load value

v_accvgpr_read v4, a44 // (op 4174) 
v_accvgpr_read v5, a45 // (op 4174) 
v_accvgpr_read v6, a46 // (op 4174) 
v_accvgpr_read v7, a47 // (op 4174) 
v_mul_f32 v14, v11, v4 // (op 4174) 
v_mul_f32 v15, v11, v5 // (op 4174) 
v_mul_f32 v16, v11, v6 // (op 4174) 
v_mul_f32 v17, v11, v7 // (op 4174) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4169) 
v_lshrrev_b32 v7, 16, v8 // (op 4169) 
v_lshlrev_b32 v6, 16, v6 // (op 4169) 
v_lshlrev_b32 v7, 16, v7 // (op 4169) 
v_mul_f32 v18, v10, v6 // (op 4169) 
v_mul_f32 v19, v10, v7 // (op 4169) 
v_and_b32 v6, 65535, v9 // (op 4169) 
v_lshrrev_b32 v7, 16, v9 // (op 4169) 
v_lshlrev_b32 v6, 16, v6 // (op 4169) 
v_lshlrev_b32 v7, 16, v7 // (op 4169) 
v_mul_f32 v20, v10, v6 // (op 4169) 
v_mul_f32 v21, v10, v7 // (op 4169) 


v_add_f32 v6, v18, v14 // (op 4170) 
v_add_f32 v7, v19, v15 // (op 4170) 
v_add_f32 v8, v20, v16 // (op 4170) 
v_add_f32 v9, v21, v17 // (op 4170) 


v_lshrrev_b32 v14, 16, v6 // (op 4171) 
v_lshrrev_b32 v15, 16, v7 // (op 4171) 
v_lshrrev_b32 v16, 16, v8 // (op 4171) 
v_lshrrev_b32 v17, 16, v9 // (op 4171) 


v_pack_b32_f16 v8, v14, v15 // (op 4164) 
v_pack_b32_f16 v9, v16, v17 // (op 4164) 

s_lshl_b32 s0, s4, 6 // (op 4164) 
s_add_u32 s0, s0, 320 // (op 4164) 
v_add_u32 v0, v13, s0 // (op 4164) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4164) Store value





v_add_u32 v0, 256, v12 // (op 4122) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4122) Load value

v_accvgpr_read v4, a32 // (op 4113) 
v_accvgpr_read v5, a33 // (op 4113) 
v_accvgpr_read v6, a34 // (op 4113) 
v_accvgpr_read v7, a35 // (op 4113) 
v_mul_f32 v14, v11, v4 // (op 4113) 
v_mul_f32 v15, v11, v5 // (op 4113) 
v_mul_f32 v16, v11, v6 // (op 4113) 
v_mul_f32 v17, v11, v7 // (op 4113) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4121) 
v_lshrrev_b32 v7, 16, v8 // (op 4121) 
v_lshlrev_b32 v6, 16, v6 // (op 4121) 
v_lshlrev_b32 v7, 16, v7 // (op 4121) 
v_mul_f32 v18, v10, v6 // (op 4121) 
v_mul_f32 v19, v10, v7 // (op 4121) 
v_and_b32 v6, 65535, v9 // (op 4121) 
v_lshrrev_b32 v7, 16, v9 // (op 4121) 
v_lshlrev_b32 v6, 16, v6 // (op 4121) 
v_lshlrev_b32 v7, 16, v7 // (op 4121) 
v_mul_f32 v20, v10, v6 // (op 4121) 
v_mul_f32 v21, v10, v7 // (op 4121) 


v_add_f32 v6, v18, v14 // (op 4114) 
v_add_f32 v7, v19, v15 // (op 4114) 
v_add_f32 v8, v20, v16 // (op 4114) 
v_add_f32 v9, v21, v17 // (op 4114) 


v_lshrrev_b32 v14, 16, v6 // (op 4115) 
v_lshrrev_b32 v15, 16, v7 // (op 4115) 
v_lshrrev_b32 v16, 16, v8 // (op 4115) 
v_lshrrev_b32 v17, 16, v9 // (op 4115) 


v_pack_b32_f16 v8, v14, v15 // (op 4117) 
v_pack_b32_f16 v9, v16, v17 // (op 4117) 

v_add_u32 v0, 256, v13 // (op 4117) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4117) Store value





s_lshl_b32 s0, s2, 6 // (op 4127) 
s_add_u32 s0, s0, 256 // (op 4127) 
v_add_u32 v0, v12, s0 // (op 4127) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4127) Load value

v_accvgpr_read v4, a36 // (op 4129) 
v_accvgpr_read v5, a37 // (op 4129) 
v_accvgpr_read v6, a38 // (op 4129) 
v_accvgpr_read v7, a39 // (op 4129) 
v_mul_f32 v14, v11, v4 // (op 4129) 
v_mul_f32 v15, v11, v5 // (op 4129) 
v_mul_f32 v16, v11, v6 // (op 4129) 
v_mul_f32 v17, v11, v7 // (op 4129) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4124) 
v_lshrrev_b32 v7, 16, v8 // (op 4124) 
v_lshlrev_b32 v6, 16, v6 // (op 4124) 
v_lshlrev_b32 v7, 16, v7 // (op 4124) 
v_mul_f32 v18, v10, v6 // (op 4124) 
v_mul_f32 v19, v10, v7 // (op 4124) 
v_and_b32 v6, 65535, v9 // (op 4124) 
v_lshrrev_b32 v7, 16, v9 // (op 4124) 
v_lshlrev_b32 v6, 16, v6 // (op 4124) 
v_lshlrev_b32 v7, 16, v7 // (op 4124) 
v_mul_f32 v20, v10, v6 // (op 4124) 
v_mul_f32 v21, v10, v7 // (op 4124) 


v_add_f32 v6, v18, v14 // (op 4125) 
v_add_f32 v7, v19, v15 // (op 4125) 
v_add_f32 v8, v20, v16 // (op 4125) 
v_add_f32 v9, v21, v17 // (op 4125) 


v_lshrrev_b32 v14, 16, v6 // (op 4126) 
v_lshrrev_b32 v15, 16, v7 // (op 4126) 
v_lshrrev_b32 v16, 16, v8 // (op 4126) 
v_lshrrev_b32 v17, 16, v9 // (op 4126) 


v_pack_b32_f16 v8, v14, v15 // (op 4119) 
v_pack_b32_f16 v9, v16, v17 // (op 4119) 

s_lshl_b32 s0, s4, 6 // (op 4119) 
s_add_u32 s0, s0, 256 // (op 4119) 
v_add_u32 v0, v13, s0 // (op 4119) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4119) Store value





v_add_u32 v0, 192, v12 // (op 4077) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4077) Load value

v_accvgpr_read v4, a24 // (op 4068) 
v_accvgpr_read v5, a25 // (op 4068) 
v_accvgpr_read v6, a26 // (op 4068) 
v_accvgpr_read v7, a27 // (op 4068) 
v_mul_f32 v14, v11, v4 // (op 4068) 
v_mul_f32 v15, v11, v5 // (op 4068) 
v_mul_f32 v16, v11, v6 // (op 4068) 
v_mul_f32 v17, v11, v7 // (op 4068) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4076) 
v_lshrrev_b32 v7, 16, v8 // (op 4076) 
v_lshlrev_b32 v6, 16, v6 // (op 4076) 
v_lshlrev_b32 v7, 16, v7 // (op 4076) 
v_mul_f32 v18, v10, v6 // (op 4076) 
v_mul_f32 v19, v10, v7 // (op 4076) 
v_and_b32 v6, 65535, v9 // (op 4076) 
v_lshrrev_b32 v7, 16, v9 // (op 4076) 
v_lshlrev_b32 v6, 16, v6 // (op 4076) 
v_lshlrev_b32 v7, 16, v7 // (op 4076) 
v_mul_f32 v20, v10, v6 // (op 4076) 
v_mul_f32 v21, v10, v7 // (op 4076) 


v_add_f32 v6, v18, v14 // (op 4069) 
v_add_f32 v7, v19, v15 // (op 4069) 
v_add_f32 v8, v20, v16 // (op 4069) 
v_add_f32 v9, v21, v17 // (op 4069) 


v_lshrrev_b32 v14, 16, v6 // (op 4070) 
v_lshrrev_b32 v15, 16, v7 // (op 4070) 
v_lshrrev_b32 v16, 16, v8 // (op 4070) 
v_lshrrev_b32 v17, 16, v9 // (op 4070) 


v_pack_b32_f16 v8, v14, v15 // (op 4072) 
v_pack_b32_f16 v9, v16, v17 // (op 4072) 

v_add_u32 v0, 192, v13 // (op 4072) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4072) Store value





s_lshl_b32 s0, s2, 6 // (op 4082) 
s_add_u32 s0, s0, 192 // (op 4082) 
v_add_u32 v0, v12, s0 // (op 4082) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4082) Load value

v_accvgpr_read v4, a28 // (op 4084) 
v_accvgpr_read v5, a29 // (op 4084) 
v_accvgpr_read v6, a30 // (op 4084) 
v_accvgpr_read v7, a31 // (op 4084) 
v_mul_f32 v14, v11, v4 // (op 4084) 
v_mul_f32 v15, v11, v5 // (op 4084) 
v_mul_f32 v16, v11, v6 // (op 4084) 
v_mul_f32 v17, v11, v7 // (op 4084) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4079) 
v_lshrrev_b32 v7, 16, v8 // (op 4079) 
v_lshlrev_b32 v6, 16, v6 // (op 4079) 
v_lshlrev_b32 v7, 16, v7 // (op 4079) 
v_mul_f32 v18, v10, v6 // (op 4079) 
v_mul_f32 v19, v10, v7 // (op 4079) 
v_and_b32 v6, 65535, v9 // (op 4079) 
v_lshrrev_b32 v7, 16, v9 // (op 4079) 
v_lshlrev_b32 v6, 16, v6 // (op 4079) 
v_lshlrev_b32 v7, 16, v7 // (op 4079) 
v_mul_f32 v20, v10, v6 // (op 4079) 
v_mul_f32 v21, v10, v7 // (op 4079) 


v_add_f32 v6, v18, v14 // (op 4080) 
v_add_f32 v7, v19, v15 // (op 4080) 
v_add_f32 v8, v20, v16 // (op 4080) 
v_add_f32 v9, v21, v17 // (op 4080) 


v_lshrrev_b32 v14, 16, v6 // (op 4081) 
v_lshrrev_b32 v15, 16, v7 // (op 4081) 
v_lshrrev_b32 v16, 16, v8 // (op 4081) 
v_lshrrev_b32 v17, 16, v9 // (op 4081) 


v_pack_b32_f16 v8, v14, v15 // (op 4074) 
v_pack_b32_f16 v9, v16, v17 // (op 4074) 

s_lshl_b32 s0, s4, 6 // (op 4074) 
s_add_u32 s0, s0, 192 // (op 4074) 
v_add_u32 v0, v13, s0 // (op 4074) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4074) Store value





v_add_u32 v0, 128, v12 // (op 4032) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4032) Load value

v_accvgpr_read v4, a16 // (op 4023) 
v_accvgpr_read v5, a17 // (op 4023) 
v_accvgpr_read v6, a18 // (op 4023) 
v_accvgpr_read v7, a19 // (op 4023) 
v_mul_f32 v14, v11, v4 // (op 4023) 
v_mul_f32 v15, v11, v5 // (op 4023) 
v_mul_f32 v16, v11, v6 // (op 4023) 
v_mul_f32 v17, v11, v7 // (op 4023) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4031) 
v_lshrrev_b32 v7, 16, v8 // (op 4031) 
v_lshlrev_b32 v6, 16, v6 // (op 4031) 
v_lshlrev_b32 v7, 16, v7 // (op 4031) 
v_mul_f32 v18, v10, v6 // (op 4031) 
v_mul_f32 v19, v10, v7 // (op 4031) 
v_and_b32 v6, 65535, v9 // (op 4031) 
v_lshrrev_b32 v7, 16, v9 // (op 4031) 
v_lshlrev_b32 v6, 16, v6 // (op 4031) 
v_lshlrev_b32 v7, 16, v7 // (op 4031) 
v_mul_f32 v20, v10, v6 // (op 4031) 
v_mul_f32 v21, v10, v7 // (op 4031) 


v_add_f32 v6, v18, v14 // (op 4024) 
v_add_f32 v7, v19, v15 // (op 4024) 
v_add_f32 v8, v20, v16 // (op 4024) 
v_add_f32 v9, v21, v17 // (op 4024) 


v_lshrrev_b32 v14, 16, v6 // (op 4025) 
v_lshrrev_b32 v15, 16, v7 // (op 4025) 
v_lshrrev_b32 v16, 16, v8 // (op 4025) 
v_lshrrev_b32 v17, 16, v9 // (op 4025) 


v_pack_b32_f16 v8, v14, v15 // (op 4027) 
v_pack_b32_f16 v9, v16, v17 // (op 4027) 

v_add_u32 v0, 128, v13 // (op 4027) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4027) Store value





s_lshl_b32 s0, s2, 6 // (op 4037) 
s_add_u32 s0, s0, 128 // (op 4037) 
v_add_u32 v0, v12, s0 // (op 4037) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4037) Load value

v_accvgpr_read v4, a20 // (op 4039) 
v_accvgpr_read v5, a21 // (op 4039) 
v_accvgpr_read v6, a22 // (op 4039) 
v_accvgpr_read v7, a23 // (op 4039) 
v_mul_f32 v14, v11, v4 // (op 4039) 
v_mul_f32 v15, v11, v5 // (op 4039) 
v_mul_f32 v16, v11, v6 // (op 4039) 
v_mul_f32 v17, v11, v7 // (op 4039) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4034) 
v_lshrrev_b32 v7, 16, v8 // (op 4034) 
v_lshlrev_b32 v6, 16, v6 // (op 4034) 
v_lshlrev_b32 v7, 16, v7 // (op 4034) 
v_mul_f32 v18, v10, v6 // (op 4034) 
v_mul_f32 v19, v10, v7 // (op 4034) 
v_and_b32 v6, 65535, v9 // (op 4034) 
v_lshrrev_b32 v7, 16, v9 // (op 4034) 
v_lshlrev_b32 v6, 16, v6 // (op 4034) 
v_lshlrev_b32 v7, 16, v7 // (op 4034) 
v_mul_f32 v20, v10, v6 // (op 4034) 
v_mul_f32 v21, v10, v7 // (op 4034) 


v_add_f32 v6, v18, v14 // (op 4035) 
v_add_f32 v7, v19, v15 // (op 4035) 
v_add_f32 v8, v20, v16 // (op 4035) 
v_add_f32 v9, v21, v17 // (op 4035) 


v_lshrrev_b32 v14, 16, v6 // (op 4036) 
v_lshrrev_b32 v15, 16, v7 // (op 4036) 
v_lshrrev_b32 v16, 16, v8 // (op 4036) 
v_lshrrev_b32 v17, 16, v9 // (op 4036) 


v_pack_b32_f16 v8, v14, v15 // (op 4029) 
v_pack_b32_f16 v9, v16, v17 // (op 4029) 

s_lshl_b32 s0, s4, 6 // (op 4029) 
s_add_u32 s0, s0, 128 // (op 4029) 
v_add_u32 v0, v13, s0 // (op 4029) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4029) Store value





v_add_u32 v0, 64, v12 // (op 3987) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3987) Load value

v_accvgpr_read v4, a8 // (op 3978) 
v_accvgpr_read v5, a9 // (op 3978) 
v_accvgpr_read v6, a10 // (op 3978) 
v_accvgpr_read v7, a11 // (op 3978) 
v_mul_f32 v14, v11, v4 // (op 3978) 
v_mul_f32 v15, v11, v5 // (op 3978) 
v_mul_f32 v16, v11, v6 // (op 3978) 
v_mul_f32 v17, v11, v7 // (op 3978) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3986) 
v_lshrrev_b32 v7, 16, v8 // (op 3986) 
v_lshlrev_b32 v6, 16, v6 // (op 3986) 
v_lshlrev_b32 v7, 16, v7 // (op 3986) 
v_mul_f32 v18, v10, v6 // (op 3986) 
v_mul_f32 v19, v10, v7 // (op 3986) 
v_and_b32 v6, 65535, v9 // (op 3986) 
v_lshrrev_b32 v7, 16, v9 // (op 3986) 
v_lshlrev_b32 v6, 16, v6 // (op 3986) 
v_lshlrev_b32 v7, 16, v7 // (op 3986) 
v_mul_f32 v20, v10, v6 // (op 3986) 
v_mul_f32 v21, v10, v7 // (op 3986) 


v_add_f32 v6, v18, v14 // (op 3979) 
v_add_f32 v7, v19, v15 // (op 3979) 
v_add_f32 v8, v20, v16 // (op 3979) 
v_add_f32 v9, v21, v17 // (op 3979) 


v_lshrrev_b32 v14, 16, v6 // (op 3980) 
v_lshrrev_b32 v15, 16, v7 // (op 3980) 
v_lshrrev_b32 v16, 16, v8 // (op 3980) 
v_lshrrev_b32 v17, 16, v9 // (op 3980) 


v_pack_b32_f16 v8, v14, v15 // (op 3982) 
v_pack_b32_f16 v9, v16, v17 // (op 3982) 

v_add_u32 v0, 64, v13 // (op 3982) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3982) Store value





s_lshl_b32 s0, s2, 6 // (op 3992) 
s_add_u32 s0, s0, 64 // (op 3992) 
v_add_u32 v0, v12, s0 // (op 3992) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3992) Load value

v_accvgpr_read v4, a12 // (op 3994) 
v_accvgpr_read v5, a13 // (op 3994) 
v_accvgpr_read v6, a14 // (op 3994) 
v_accvgpr_read v7, a15 // (op 3994) 
v_mul_f32 v14, v11, v4 // (op 3994) 
v_mul_f32 v15, v11, v5 // (op 3994) 
v_mul_f32 v16, v11, v6 // (op 3994) 
v_mul_f32 v17, v11, v7 // (op 3994) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3989) 
v_lshrrev_b32 v7, 16, v8 // (op 3989) 
v_lshlrev_b32 v6, 16, v6 // (op 3989) 
v_lshlrev_b32 v7, 16, v7 // (op 3989) 
v_mul_f32 v18, v10, v6 // (op 3989) 
v_mul_f32 v19, v10, v7 // (op 3989) 
v_and_b32 v6, 65535, v9 // (op 3989) 
v_lshrrev_b32 v7, 16, v9 // (op 3989) 
v_lshlrev_b32 v6, 16, v6 // (op 3989) 
v_lshlrev_b32 v7, 16, v7 // (op 3989) 
v_mul_f32 v20, v10, v6 // (op 3989) 
v_mul_f32 v21, v10, v7 // (op 3989) 


v_add_f32 v6, v18, v14 // (op 3990) 
v_add_f32 v7, v19, v15 // (op 3990) 
v_add_f32 v8, v20, v16 // (op 3990) 
v_add_f32 v9, v21, v17 // (op 3990) 


v_lshrrev_b32 v14, 16, v6 // (op 3991) 
v_lshrrev_b32 v15, 16, v7 // (op 3991) 
v_lshrrev_b32 v16, 16, v8 // (op 3991) 
v_lshrrev_b32 v17, 16, v9 // (op 3991) 


v_pack_b32_f16 v8, v14, v15 // (op 3984) 
v_pack_b32_f16 v9, v16, v17 // (op 3984) 

s_lshl_b32 s0, s4, 6 // (op 3984) 
s_add_u32 s0, s0, 64 // (op 3984) 
v_add_u32 v0, v13, s0 // (op 3984) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3984) Store value





v_mov_b32 v0, v12 // (op 148) reg expression

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 148) Load value

v_accvgpr_read v4, a0 // (op 144) 
v_accvgpr_read v5, a1 // (op 144) 
v_accvgpr_read v6, a2 // (op 144) 
v_accvgpr_read v7, a3 // (op 144) 
v_mul_f32 v14, v11, v4 // (op 144) 
v_mul_f32 v15, v11, v5 // (op 144) 
v_mul_f32 v16, v11, v6 // (op 144) 
v_mul_f32 v17, v11, v7 // (op 144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_and_b32 v6, 65535, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_lshrrev_b32 v14, 16, v6 // (op 146) 
v_lshrrev_b32 v15, 16, v7 // (op 146) 
v_lshrrev_b32 v16, 16, v8 // (op 146) 
v_lshrrev_b32 v17, 16, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 3947) 
v_add_u32 v0, v12, s0 // (op 3947) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3947) Load value

v_accvgpr_read v4, a4 // (op 3943) 
v_accvgpr_read v5, a5 // (op 3943) 
v_accvgpr_read v6, a6 // (op 3943) 
v_accvgpr_read v7, a7 // (op 3943) 
v_mul_f32 v14, v11, v4 // (op 3943) 
v_mul_f32 v15, v11, v5 // (op 3943) 
v_mul_f32 v16, v11, v6 // (op 3943) 
v_mul_f32 v17, v11, v7 // (op 3943) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3948) 
v_lshrrev_b32 v7, 16, v8 // (op 3948) 
v_lshlrev_b32 v6, 16, v6 // (op 3948) 
v_lshlrev_b32 v7, 16, v7 // (op 3948) 
v_mul_f32 v18, v10, v6 // (op 3948) 
v_mul_f32 v19, v10, v7 // (op 3948) 
v_and_b32 v6, 65535, v9 // (op 3948) 
v_lshrrev_b32 v7, 16, v9 // (op 3948) 
v_lshlrev_b32 v6, 16, v6 // (op 3948) 
v_lshlrev_b32 v7, 16, v7 // (op 3948) 
v_mul_f32 v20, v10, v6 // (op 3948) 
v_mul_f32 v21, v10, v7 // (op 3948) 


v_add_f32 v6, v18, v14 // (op 3944) 
v_add_f32 v7, v19, v15 // (op 3944) 
v_add_f32 v8, v20, v16 // (op 3944) 
v_add_f32 v9, v21, v17 // (op 3944) 


v_lshrrev_b32 v14, 16, v6 // (op 3945) 
v_lshrrev_b32 v15, 16, v7 // (op 3945) 
v_lshrrev_b32 v16, 16, v8 // (op 3945) 
v_lshrrev_b32 v17, 16, v9 // (op 3945) 


v_pack_b32_f16 v8, v14, v15 // (op 3946) 
v_pack_b32_f16 v9, v16, v17 // (op 3946) 

s_lshl_b32 s0, s4, 6 // (op 3946) 
v_add_u32 v0, v13, s0 // (op 3946) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3946) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
.LRR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_end:
.size RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC, .LRR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_end-RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
  .amdhsa_next_free_vgpr 216
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 76288 // lds bytes
  .amdhsa_accum_offset 152
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F4
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA } ]
    macrotile: [256, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
    .symbol: RR_GEMM_TT_FP8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 76288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 150
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
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
      - .name: Tensor_2_size_1_12
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
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 128
        .value_kind: by_value
...

.end_amdgpu_metadata

