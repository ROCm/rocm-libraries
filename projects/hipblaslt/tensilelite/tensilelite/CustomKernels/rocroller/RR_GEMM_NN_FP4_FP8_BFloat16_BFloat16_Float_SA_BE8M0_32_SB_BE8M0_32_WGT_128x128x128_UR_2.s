// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2,@function
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 8965) 
v_and_b32 v3, 48, v1 // (op 8965) 
v_sub_u32 v4, v0, v3 // (op 8965) 

v_lshrrev_b32 v3, 4, v0 // (op 8965) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 8965) 


v_lshrrev_b32 v3, 7, v1 // (op 8965) 
v_lshlrev_b32 v4, 4, v3 // (op 8965) 

v_lshl_add_u32 v3, v0, 7, v4 // (op 8965) 


v_lshlrev_b32 v0, 2, v3 // (op 8965) 

v_lshrrev_b32 v3, 3, v0 // (op 8965) 

v_lshrrev_b32 v0, 6, v1 // (op 9022) 
v_and_b32 v4, 1, v0 // (op 9022) 

v_and_b32 v0, 15, v1 // (op 9022) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 9022) 


v_and_b32 v0, 48, v1 // (op 9022) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 9022) 


v_lshrrev_b32 v0, 7, v1 // (op 9080) 
v_and_b32 v5, 15, v1 // (op 9080) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9080) 


v_and_b32 v0, 63, v1 // (op 9080) 
v_lshrrev_b32 v5, 4, v0 // (op 9080) 

v_lshl_add_u32 v0, v5, 7, v6 // (op 9080) 


v_lshrrev_b32 v5, 6, v1 // (op 9138) 
v_and_b32 v6, 1, v5 // (op 9138) 

v_and_b32 v5, 15, v1 // (op 9138) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9138) 


v_and_b32 v5, 63, v1 // (op 9138) 
v_lshrrev_b32 v6, 4, v5 // (op 9138) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9138) 


v_and_b32 v6, 63, v1 // (op 9205) 
v_and_b32 v7, 48, v1 // (op 9205) 
v_sub_u32 v8, v6, v7 // (op 9205) 

v_lshrrev_b32 v7, 4, v6 // (op 9205) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 9205) 


v_lshrrev_b32 v7, 7, v1 // (op 9205) 
v_lshlrev_b32 v8, 4, v7 // (op 9205) 

v_lshl_add_u32 v7, v6, 7, v8 // (op 9205) 


v_lshlrev_b32 v6, 2, v7 // (op 9205) 

v_lshrrev_b32 v7, 3, v6 // (op 9205) 

v_lshrrev_b32 v6, 6, v1 // (op 9249) 
v_and_b32 v8, 1, v6 // (op 9249) 

v_and_b32 v6, 15, v1 // (op 9249) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 9249) 


v_and_b32 v6, 48, v1 // (op 9249) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 9249) 


v_lshrrev_b32 v6, 7, v1 // (op 9278) 
v_and_b32 v9, 15, v1 // (op 9278) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9278) 


v_and_b32 v6, 63, v1 // (op 9278) 
v_lshrrev_b32 v9, 4, v6 // (op 9278) 

v_lshl_add_u32 v6, v9, 7, v10 // (op 9278) 


v_lshrrev_b32 v9, 6, v1 // (op 9307) 
v_and_b32 v10, 1, v9 // (op 9307) 

v_and_b32 v9, 15, v1 // (op 9307) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9307) 


v_and_b32 v9, 63, v1 // (op 9307) 
v_lshrrev_b32 v10, 4, v9 // (op 9307) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 9307) 


v_lshrrev_b32 v10, 7, v1 // (op 8904) 
v_and_b32 v11, 63, v1 // (op 8904) 
v_lshrrev_b32 v12, 4, v11 // (op 8904) 
v_lshlrev_b32 v13, 2, v12 // (op 8904) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8904) 


v_lshl_add_u32 v10, s16, 7, v12 // (op 8904) 

v_lshrrev_b32 v12, 2, v11 // (op 8904) 

v_and_b32 v11, 3, v12 // (op 8904) 

v_and_b32 v12, 3, v1 // (op 8904) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 8904) 


v_lshrrev_b32 v11, 6, v1 // (op 8904) 
v_and_b32 v12, 1, v11 // (op 8904) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 8904) 


v_lshl_add_u32 v12, s17, 7, v11 // (op 8904) 

v_mul_lo_u32 v11, v12, s2 // (op 8904) 

v_add_lshl_u32 v12, v10, v11, 1 // (op 8904) 


v_lshrrev_b32 v10, 7, v1 // (op 9159) 
v_and_b32 v11, 63, v1 // (op 9159) 
v_lshrrev_b32 v13, 4, v11 // (op 9159) 
s_mul_i32 s1, s2, s9 // (op 8906) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 8906) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 8906) least significant
s_mul_hi_u32 s19, s2, s8 // (op 8906) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8906) most significant: sum
s_add_u32 s1, s1, s19 // (op 8906) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 8906) 

v_lshlrev_b32 v14, 2, v13 // (op 9159) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 9159) 


v_lshl_add_u32 v10, s16, 7, v13 // (op 9159) 

v_lshrrev_b32 v13, 2, v11 // (op 9159) 

v_and_b32 v11, 3, v13 // (op 9159) 

v_and_b32 v13, 3, v1 // (op 9159) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 9159) 


v_lshrrev_b32 v11, 6, v1 // (op 9159) 
v_and_b32 v13, 1, v11 // (op 9159) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 9159) 


v_lshl_add_u32 v13, s17, 7, v11 // (op 9159) 

v_mul_lo_u32 v11, v13, s4 // (op 9159) 

v_add_lshl_u32 v13, v10, v11, 1 // (op 9159) 


s_mov_b32 s40, s6 // (op 8906) reg expression
s_mov_b32 s41, s7 // (op 8906) reg expression
s_mov_b32 s42, s18 // (op 8906) reg expression
s_mov_b32 s43, 131072 // (op 8906) call()

s_mul_i32 s1, s4, s9 // (op 9161) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 9161) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 9161) least significant
s_mul_hi_u32 s7, s4, s8 // (op 9161) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 9161) most significant: sum
s_add_u32 s1, s1, s7 // (op 9161) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 9161) 

s_mov_b32 s44, s10 // (op 9161) reg expression
s_mov_b32 s45, s11 // (op 9161) reg expression
s_mov_b32 s46, s6 // (op 9161) reg expression
s_mov_b32 s47, 131072 // (op 9161) call()


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
v_accvgpr_write a8, 0.00000 // (op 516) call()
v_accvgpr_write a9, 0.00000 // (op 516) call()
v_accvgpr_write a10, 0.00000 // (op 516) call()
v_accvgpr_write a11, 0.00000 // (op 516) call()
v_accvgpr_write a12, 0.00000 // (op 692) call()
v_accvgpr_write a13, 0.00000 // (op 692) call()
v_accvgpr_write a14, 0.00000 // (op 692) call()
v_accvgpr_write a15, 0.00000 // (op 692) call()
v_accvgpr_write a16, 0.00000 // (op 956) call()
v_accvgpr_write a17, 0.00000 // (op 956) call()
v_accvgpr_write a18, 0.00000 // (op 956) call()
v_accvgpr_write a19, 0.00000 // (op 956) call()
v_accvgpr_write a20, 0.00000 // (op 1265) call()
v_accvgpr_write a21, 0.00000 // (op 1265) call()
v_accvgpr_write a22, 0.00000 // (op 1265) call()
v_accvgpr_write a23, 0.00000 // (op 1265) call()
v_accvgpr_write a24, 0.00000 // (op 1266) call()
v_accvgpr_write a25, 0.00000 // (op 1266) call()
v_accvgpr_write a26, 0.00000 // (op 1266) call()
v_accvgpr_write a27, 0.00000 // (op 1266) call()
v_accvgpr_write a28, 0.00000 // (op 1267) call()
v_accvgpr_write a29, 0.00000 // (op 1267) call()
v_accvgpr_write a30, 0.00000 // (op 1267) call()
v_accvgpr_write a31, 0.00000 // (op 1267) call()
v_accvgpr_write a32, 0.00000 // (op 1726) call()
v_accvgpr_write a33, 0.00000 // (op 1726) call()
v_accvgpr_write a34, 0.00000 // (op 1726) call()
v_accvgpr_write a35, 0.00000 // (op 1726) call()
v_accvgpr_write a36, 0.00000 // (op 2035) call()
v_accvgpr_write a37, 0.00000 // (op 2035) call()
v_accvgpr_write a38, 0.00000 // (op 2035) call()
v_accvgpr_write a39, 0.00000 // (op 2035) call()
v_accvgpr_write a40, 0.00000 // (op 2036) call()
v_accvgpr_write a41, 0.00000 // (op 2036) call()
v_accvgpr_write a42, 0.00000 // (op 2036) call()
v_accvgpr_write a43, 0.00000 // (op 2036) call()
v_accvgpr_write a44, 0.00000 // (op 2037) call()
v_accvgpr_write a45, 0.00000 // (op 2037) call()
v_accvgpr_write a46, 0.00000 // (op 2037) call()
v_accvgpr_write a47, 0.00000 // (op 2037) call()
v_accvgpr_write a48, 0.00000 // (op 2496) call()
v_accvgpr_write a49, 0.00000 // (op 2496) call()
v_accvgpr_write a50, 0.00000 // (op 2496) call()
v_accvgpr_write a51, 0.00000 // (op 2496) call()
v_accvgpr_write a52, 0.00000 // (op 2805) call()
v_accvgpr_write a53, 0.00000 // (op 2805) call()
v_accvgpr_write a54, 0.00000 // (op 2805) call()
v_accvgpr_write a55, 0.00000 // (op 2805) call()
v_accvgpr_write a56, 0.00000 // (op 2806) call()
v_accvgpr_write a57, 0.00000 // (op 2806) call()
v_accvgpr_write a58, 0.00000 // (op 2806) call()
v_accvgpr_write a59, 0.00000 // (op 2806) call()
v_accvgpr_write a60, 0.00000 // (op 2807) call()
v_accvgpr_write a61, 0.00000 // (op 2807) call()
v_accvgpr_write a62, 0.00000 // (op 2807) call()
v_accvgpr_write a63, 0.00000 // (op 2807) call()
v_and_b32 v14, -4, v1 // (op 8666) 
v_sub_u32 v15, v1, v14 // (op 8666) 

v_lshlrev_b32 v14, 5, v15 // (op 8666) 

v_lshl_add_u32 v15, s16, 7, v14 // (op 8666) 

v_mov_b32 v17, 0 // (op 8666) convert
v_mov_b32 v16, v15 // (op 8666) convert

v_lshrrev_b32 v14, 2, v1 // (op 8666) 
v_mul_lo_u32 v19, v14, s13 // (op 8666) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 8666) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 8666) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 8666) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 8666) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 8666) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 8666) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8666) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 8668) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8668) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8668) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8668) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8668) most significant: sum
s_add_u32 s1, s1, s7 // (op 8668) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8668) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8668) 

s_mov_b32 s48, s20 // (op 8668) reg expression
s_mov_b32 s49, s21 // (op 8668) reg expression
s_mov_b32 s50, s0 // (op 8668) reg expression
s_mov_b32 s51, 131072 // (op 8668) call()

v_and_b32 v14, -4, v1 // (op 8685) 
v_sub_u32 v15, v1, v14 // (op 8685) 

v_lshlrev_b32 v14, 5, v15 // (op 8685) 

v_lshl_add_u32 v15, s16, 7, v14 // (op 8685) 

v_mov_b32 v19, 0 // (op 8685) convert
v_mov_b32 v18, v15 // (op 8685) convert

v_lshrrev_b32 v14, 2, v1 // (op 8685) 
v_mul_lo_u32 v21, v14, s13 // (op 8685) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 8685) most significant: high of low * low
v_mul_lo_u32 v20, v14, s12 // (op 8685) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 8685) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 8685) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 8685) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 8685) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 8685) 

s_mul_i32 s1, s12, s23 // (op 8687) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8687) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8687) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8687) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8687) most significant: sum
s_add_u32 s1, s1, s7 // (op 8687) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8687) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8687) 

s_mov_b32 s48, s20 // (op 8687) reg expression
s_mov_b32 s49, s21 // (op 8687) reg expression
s_mov_b32 s50, s0 // (op 8687) reg expression
s_mov_b32 s51, 131072 // (op 8687) call()

v_and_b32 v14, -8, v1 // (op 8730) 
v_sub_u32 v15, v1, v14 // (op 8730) 

v_lshlrev_b32 v14, 4, v15 // (op 8730) 

v_mov_b32 v21, 0 // (op 8730) convert
v_mov_b32 v20, v14 // (op 8730) convert

v_lshrrev_b32 v14, 3, v1 // (op 8730) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 8730) 

v_mul_lo_u32 v23, v15, s25 // (op 8730) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 8730) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 8730) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 8730) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 8730) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 8730) most significant half



s_mul_i32 s1, s24, s9 // (op 8732) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8732) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8732) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8732) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8732) most significant: sum
s_add_u32 s1, s1, s7 // (op 8732) most significant: sum


s_mov_b32 s52, s26 // (op 8732) reg expression
s_mov_b32 s53, s27 // (op 8732) reg expression
s_mov_b32 s54, s0 // (op 8732) reg expression
s_mov_b32 s55, 131072 // (op 8732) call()

v_and_b32 v20, -8, v1 // (op 8749) 
v_sub_u32 v21, v1, v20 // (op 8749) 

v_lshlrev_b32 v20, 4, v21 // (op 8749) 

v_mov_b32 v23, 0 // (op 8749) convert
v_mov_b32 v22, v20 // (op 8749) convert

v_lshrrev_b32 v20, 3, v1 // (op 8749) 
v_lshl_add_u32 v21, s17, 7, v20 // (op 8749) 

v_mul_lo_u32 v25, v21, s25 // (op 8749) most significant: low of low * high
v_mul_hi_u32 v20, v21, s24 // (op 8749) most significant: high of low * low
v_mul_lo_u32 v24, v21, s24 // (op 8749) least significant: low of low * low
v_add_u32 v25, v25, v20 // (op 8749) most significant: sum


v_add_co_u32 v20, s[0:1], v22, v24 // (op 8749) least significant half
v_addc_co_u32 v21, s[0:1], v23, v25, s[0:1] // (op 8749) most significant half



s_mul_i32 s1, s24, s9 // (op 8751) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8751) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8751) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8751) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8751) most significant: sum
s_add_u32 s1, s1, s7 // (op 8751) most significant: sum


s_mov_b32 s52, s26 // (op 8751) reg expression
s_mov_b32 s53, s27 // (op 8751) reg expression
s_mov_b32 s54, s0 // (op 8751) reg expression
s_mov_b32 s55, 131072 // (op 8751) call()

v_and_b32 v22, -128, v1 // (op 8794) 
v_sub_u32 v23, v1, v22 // (op 8794) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 8794) 

v_lshrrev_b32 v23, 7, v1 // (op 8794) 
v_lshlrev_b32 v24, 1, v23 // (op 8794) 

v_mul_lo_u32 v23, v24, s28 // (op 8794) 

v_add_u32 v24, v22, v23 // (op 8794) 


s_mul_i32 s1, s28, s33 // (op 8796) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 8796) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8796) least significant
s_mul_hi_u32 s7, s28, s32 // (op 8796) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8796) most significant: sum
s_add_u32 s1, s1, s7 // (op 8796) most significant: sum


s_mov_b32 s56, s30 // (op 8796) reg expression
s_mov_b32 s57, s31 // (op 8796) reg expression
s_mov_b32 s58, s0 // (op 8796) reg expression
s_mov_b32 s59, 131072 // (op 8796) call()

v_and_b32 v22, -128, v1 // (op 8813) 
v_sub_u32 v23, v1, v22 // (op 8813) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 8813) 

v_lshrrev_b32 v23, 7, v1 // (op 8813) 
v_lshlrev_b32 v25, 1, v23 // (op 8813) 

v_mul_lo_u32 v23, v25, s28 // (op 8813) 

v_add_u32 v25, v22, v23 // (op 8813) 


s_mul_i32 s1, s28, s33 // (op 8815) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 8815) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8815) least significant
s_mul_hi_u32 s7, s28, s32 // (op 8815) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8815) most significant: sum
s_add_u32 s1, s1, s7 // (op 8815) most significant: sum


s_mov_b32 s56, s30 // (op 8815) reg expression
s_mov_b32 s57, s31 // (op 8815) reg expression
s_mov_b32 s58, s0 // (op 8815) reg expression
s_mov_b32 s59, 131072 // (op 8815) call()

v_and_b32 v22, -4, v1 // (op 8858) 
v_sub_u32 v23, v1, v22 // (op 8858) 

v_lshrrev_b32 v22, 2, v1 // (op 8858) 
v_lshlrev_b32 v26, 1, v22 // (op 8858) 

v_lshl_add_u32 v22, s17, 7, v26 // (op 8858) 

v_mul_lo_u32 v26, v22, s34 // (op 8858) 

v_add_u32 v22, v23, v26 // (op 8858) 


s_mul_i32 s1, s34, s9 // (op 8860) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 8860) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8860) least significant
s_mul_hi_u32 s7, s34, s8 // (op 8860) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8860) most significant: sum
s_add_u32 s1, s1, s7 // (op 8860) most significant: sum


s_mov_b32 s60, s36 // (op 8860) reg expression
s_mov_b32 s61, s37 // (op 8860) reg expression
s_mov_b32 s62, s0 // (op 8860) reg expression
s_mov_b32 s63, 131072 // (op 8860) call()

v_and_b32 v23, -4, v1 // (op 8877) 
v_sub_u32 v26, v1, v23 // (op 8877) 

v_lshrrev_b32 v23, 2, v1 // (op 8877) 
v_lshlrev_b32 v27, 1, v23 // (op 8877) 

v_lshl_add_u32 v23, s17, 7, v27 // (op 8877) 

v_mul_lo_u32 v27, v23, s34 // (op 8877) 

v_add_u32 v23, v26, v27 // (op 8877) 


s_mul_i32 s1, s34, s9 // (op 8879) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 8879) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8879) least significant
s_mul_hi_u32 s7, s34, s8 // (op 8879) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8879) most significant: sum
s_add_u32 s1, s1, s7 // (op 8879) most significant: sum


s_mov_b32 s60, s36 // (op 8879) reg expression
s_mov_b32 s61, s37 // (op 8879) reg expression
s_mov_b32 s62, s0 // (op 8879) reg expression
s_mov_b32 s63, 131072 // (op 8879) call()

v_and_b32 v26, -8, v1 // (op 9005) 
v_sub_u32 v27, v1, v26 // (op 9005) 

v_lshlrev_b32 v26, 4, v27 // (op 9005) 

v_lshrrev_b32 v27, 3, v1 // (op 9005) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 9005) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 9005) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8600) 
v_mov_b64 v[26:27], s[6:7] // (op 8600) convert


v_mov_b64 v[28:29], v[14:15] // (op 8600) 
s_mov_b32 m0, s0 // (op 8600) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8600) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 8600) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8600) most significant half

s_add_u32 m0, m0, 4096 // (op 8600) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8600) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 8600) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8600) most significant half

s_add_u32 m0, m0, 4096 // (op 8600) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8600) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 8600) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8600) most significant half

s_add_u32 m0, m0, 4096 // (op 8600) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8600) Load value direct to lds




v_mov_b64 v[26:27], s[34:35] // (op 8318) convert

v_mov_b32 v28, v22 // (op 8318) 
buffer_load_ubyte v30, v28, s[60:63], 0 offen offset: 0 // (op 8318) Load value
v_add_u32 v28, v28, v26 // (op 8318) 
buffer_load_ubyte v31, v28, s[60:63], 0 offen offset: 0 // (op 8318) Load value


v_and_b32 v26, -4, v1 // (op 8948) 
v_sub_u32 v27, v1, v26 // (op 8948) 

v_lshlrev_b32 v26, 5, v27 // (op 8948) 

v_lshrrev_b32 v27, 2, v1 // (op 8948) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 8948) 


v_lshlrev_b32 v29, 2, v28 // (op 8948) 

v_lshrrev_b32 v26, 3, v29 // (op 8948) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 8948) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 8605) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 8605) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 8605) 


v_mov_b64 v[28:29], v[16:17] // (op 8605) reg expression
s_add_u32 s1, 16384, s0 // (op 8605) 
s_mov_b32 m0, s1 // (op 8605) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8605) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 8605) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8605) most significant half

s_add_u32 m0, m0, 4096 // (op 8605) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8605) Load value direct to lds





v_mov_b64 v[26:27], s[28:29] // (op 8324) convert

v_mov_b32 v28, v24 // (op 8324) reg expression
buffer_load_ubyte v32, v28, s[56:59], 0 offen offset: 0 // (op 8324) Load value
v_add_u32 v28, v28, v26 // (op 8324) 
buffer_load_ubyte v33, v28, s[56:59], 0 offen offset: 0 // (op 8324) Load value


v_and_b32 v26, -8, v1 // (op 9232) 
v_sub_u32 v27, v1, v26 // (op 9232) 

v_lshlrev_b32 v26, 4, v27 // (op 9232) 

v_lshrrev_b32 v27, 3, v1 // (op 9232) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 9232) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 9232) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8610) 
v_mov_b64 v[26:27], s[6:7] // (op 8610) convert


v_mov_b64 v[28:29], v[14:15] // (op 8610) 
s_add_u32 s1, 24576, s0 // (op 8610) 
s_mov_b32 m0, s1 // (op 8610) reg expression
s_mov_b32 s6, 128 // (op 8610) call()
buffer_load_dwordx4 v28, s[52:55], s6 offen offset: 0 lds // (op 8610) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v26 // (op 8610) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 8610) most significant half

s_add_u32 m0, m0, 4096 // (op 8610) 
buffer_load_dwordx4 v28, s[52:55], s6 offen offset: 0 lds // (op 8610) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v26 // (op 8610) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 8610) most significant half

s_add_u32 m0, m0, 4096 // (op 8610) 
buffer_load_dwordx4 v28, s[52:55], s6 offen offset: 0 lds // (op 8610) Load value direct to lds
v_add_co_u32 v28, s[10:11], v28, v26 // (op 8610) least significant half
v_addc_co_u32 v29, s[10:11], 0, v29, s[10:11] // (op 8610) most significant half

s_add_u32 m0, m0, 4096 // (op 8610) 
buffer_load_dwordx4 v28, s[52:55], s6 offen offset: 0 lds // (op 8610) Load value direct to lds







v_mov_b64 v[14:15], s[34:35] // (op 8330) convert

v_mov_b32 v26, v22 // (op 8330) 
buffer_load_ubyte v28, v26, s[60:63], 0 offen offset:4 // (op 8330) Load value
v_add_u32 v26, v26, v14 // (op 8330) 
buffer_load_ubyte v29, v26, s[60:63], 0 offen offset:4 // (op 8330) Load value



v_and_b32 v22, -4, v1 // (op 9188) 
v_sub_u32 v14, v1, v22 // (op 9188) 

v_lshlrev_b32 v15, 5, v14 // (op 9188) 

v_lshrrev_b32 v14, 2, v1 // (op 9188) 
v_lshl_add_u32 v22, v14, 7, v15 // (op 9188) 


v_lshlrev_b32 v14, 2, v22 // (op 9188) 

v_lshrrev_b32 v15, 3, v14 // (op 9188) 

s_nop 0
v_readfirstlane_b32 s0, v15 // (op 9188) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 8612) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 8612) 

v_lshrrev_b64 v[14:15], 3, s[10:11] // (op 8612) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 8612) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 8612) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 8612) 

v_mov_b32 v22, s6 // (op 8612) 
v_mov_b32 v26, s7 // (op 8612) 
v_add_co_u32 v34, s[10:11], v16, v22 // (op 8612) least significant half
v_addc_co_u32 v35, s[10:11], v17, v26, s[10:11] // (op 8612) most significant half




s_add_u32 s1, 40960, s0 // (op 8612) 
s_mov_b32 m0, s1 // (op 8612) reg expression
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 8612) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v14 // (op 8612) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 8612) most significant half

s_add_u32 m0, m0, 4096 // (op 8612) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 8612) Load value direct to lds






v_mov_b64 v[26:27], s[28:29] // (op 8336) convert

s_lshl_b32 s0, s28, 2 // (op 8336) 
v_add_u32 v22, v24, s0 // (op 8336) 

buffer_load_ubyte v14, v22, s[56:59], 0 offen offset: 0 // (op 8336) Load value
v_add_u32 v22, v22, v26 // (op 8336) 
buffer_load_ubyte v15, v22, s[56:59], 0 offen offset: 0 // (op 8336) Load value



v_and_b32 v22, -4, v1 // (op 9121) 
v_sub_u32 v24, v1, v22 // (op 9121) 

v_lshrrev_b32 v22, 2, v1 // (op 9121) 
v_lshl_add_u32 v16, v22, 3, v24 // (op 9121) 





s_waitcnt vmcnt(15)
ds_write_b8 v16, v30 offset:49152 // (op 8339) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v16, v31 offset:49156 // (op 8339) Store local data 

v_and_b32 v22, -128, v1 // (op 9063) 
v_sub_u32 v24, v1, v22 // (op 9063) 

v_lshrrev_b32 v22, 7, v1 // (op 9063) 
v_lshl_add_u32 v16, v22, 8, v24 // (op 9063) 





s_waitcnt vmcnt(11)
ds_write_b8 v16, v32 offset:49664 // (op 8341) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v16, v33 offset:49792 // (op 8341) Store local data 

s_barrier  // (op 9440) 



ds_read_b128 v[34:37], v4 // (op 8376) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 8376) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 9474) 



ds_read_u8 v22, v5 offset:49152 // (op 8416) Load local data 



ds_read_b128 v[42:45], v4 offset:4096 // (op 8381) Load local data 
ds_read_b128 v[46:49], v4 offset:4160 // (op 8381) Load local data 



ds_read_u8 v24, v5 offset:49280 // (op 8421) Load local data 



ds_read_b128 v[50:53], v4 offset:8192 // (op 8386) Load local data 
ds_read_b128 v[54:57], v4 offset:8256 // (op 8386) Load local data 



ds_read_u8 v16, v5 offset:49408 // (op 8426) Load local data 



ds_read_b128 v[58:61], v4 offset:12288 // (op 8391) Load local data 
ds_read_b128 v[62:65], v4 offset:12352 // (op 8391) Load local data 



ds_read_u8 v17, v5 offset:49536 // (op 8431) Load local data 
s_barrier  // (op 9424) 



ds_read_b64_tr_b4 v[66:67], v3 offset:16384 // (op 8356) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v3 offset:17408 // (op 8356) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 9457) 


ds_read_u8 v26, v0 offset:49664 // (op 8396) Load local data 



ds_read_b64_tr_b4 v[70:71], v3 offset:16400 // (op 8361) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v3 offset:17424 // (op 8361) Transpose load local data 


ds_read_u8 v27, v0 offset:49696 // (op 8401) Load local data 



ds_read_b64_tr_b4 v[74:75], v3 offset:16416 // (op 8366) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v3 offset:17440 // (op 8366) Transpose load local data 


ds_read_u8 v78, v0 offset:49728 // (op 8406) Load local data 



ds_read_b64_tr_b4 v[80:81], v3 offset:16432 // (op 8371) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v3 offset:17456 // (op 8371) Transpose load local data 


ds_read_u8 v79, v0 offset:49760 // (op 8411) Load local data 
v_and_b32 v84, -4, v1 // (op 8920) 
v_sub_u32 v85, v1, v84 // (op 8920) 

v_lshlrev_b32 v84, 5, v85 // (op 8920) 

v_lshrrev_b32 v85, 2, v1 // (op 8920) 
v_lshl_add_u32 v86, v85, 7, v84 // (op 8920) 


v_lshlrev_b32 v84, 2, v86 // (op 8920) 

v_lshrrev_b32 v85, 3, v84 // (op 8920) 

s_nop 0
v_readfirstlane_b32 s0, v85 // (op 8920) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 8977) 
v_sub_u32 v85, v1, v84 // (op 8977) 

v_lshlrev_b32 v84, 4, v85 // (op 8977) 

v_lshrrev_b32 v85, 3, v1 // (op 8977) 
v_lshl_add_u32 v86, v85, 7, v84 // (op 8977) 


s_nop 0
v_readfirstlane_b32 s1, v86 // (op 8977) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -128, v1 // (op 9034) 
v_sub_u32 v85, v1, v84 // (op 9034) 

v_lshrrev_b32 v84, 7, v1 // (op 9034) 
v_lshl_add_u32 v86, v84, 8, v85 // (op 9034) 


v_and_b32 v84, -4, v1 // (op 9092) 
v_sub_u32 v85, v1, v84 // (op 9092) 

v_lshrrev_b32 v84, 2, v1 // (op 9092) 
v_lshl_add_u32 v87, v84, 3, v85 // (op 9092) 


v_and_b32 v84, -4, v1 // (op 9173) 
v_sub_u32 v85, v1, v84 // (op 9173) 

v_lshlrev_b32 v84, 5, v85 // (op 9173) 

v_lshrrev_b32 v85, 2, v1 // (op 9173) 
v_lshl_add_u32 v88, v85, 7, v84 // (op 9173) 


v_lshlrev_b32 v84, 2, v88 // (op 9173) 

v_lshrrev_b32 v85, 3, v84 // (op 9173) 

s_nop 0
v_readfirstlane_b32 s6, v85 // (op 9173) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -8, v1 // (op 9217) 
v_sub_u32 v85, v1, v84 // (op 9217) 

v_lshlrev_b32 v84, 4, v85 // (op 9217) 

v_lshrrev_b32 v85, 3, v1 // (op 9217) 
v_lshl_add_u32 v88, v85, 7, v84 // (op 9217) 


s_nop 0
v_readfirstlane_b32 s7, v88 // (op 9217) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v84, -128, v1 // (op 9261) 
v_sub_u32 v85, v1, v84 // (op 9261) 

v_lshrrev_b32 v84, 7, v1 // (op 9261) 
v_lshl_add_u32 v88, v84, 8, v85 // (op 9261) 


v_and_b32 v84, -4, v1 // (op 9290) 
v_sub_u32 v85, v1, v84 // (op 9290) 

v_lshrrev_b32 v84, 2, v1 // (op 9290) 
v_lshl_add_u32 v89, v84, 3, v85 // (op 9290) 


s_mov_b32 s10, 0 // (op 2529) call()
s_ashr_i64 s[14:15], s[22:23], 7 // (op 2528) 
s_mov_b32 s11, s14 // (op 2528) convert

s_and_b32 s14, s11, 4294967294 // (op 2528) 

s_cmp_lt_u32 s10, s14 // (op 2528) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2528 // (op 2528) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2528 if false)
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2528:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:41], a[0:3], v26, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 90) 
s_barrier  // (op 9443) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:49], a[4:7], v26, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 380) 
s_add_u32 s10, s10, 2 // (op 2530) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:57], a[8:11], v26, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 556) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 8572) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:65], a[12:15], v26, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 732) 
v_mov_b64 v[84:85], s[14:15] // (op 8572) convert


v_mov_b64 v[90:91], v[20:21] // (op 8572) 
v_mov_b64 v[92:93], v[90:91] // (op 8572) 
s_mov_b32 m0, s1 // (op 8572) reg expression
s_mov_b32 s11, 256 // (op 8572) call()
buffer_load_dwordx4 v92, s[52:55], s11 offen offset: 0 lds // (op 8572) Load value direct to lds
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[34:41], a[16:19], v27, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1227) 
v_add_co_u32 v92, s[14:15], v92, v84 // (op 8572) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[42:49], a[20:23], v27, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1181) 
v_addc_co_u32 v93, s[14:15], 0, v93, s[14:15] // (op 8572) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[50:57], a[24:27], v27, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1135) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[58:65], a[28:31], v27, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1089) 
s_add_u32 m0, m0, 4096 // (op 8572) 
buffer_load_dwordx4 v92, s[52:55], s11 offen offset: 0 lds // (op 8572) Load value direct to lds
v_add_co_u32 v92, s[14:15], v92, v84 // (op 8572) least significant half
v_addc_co_u32 v93, s[14:15], 0, v93, s[14:15] // (op 8572) most significant half

s_add_u32 m0, m0, 4096 // (op 8572) 
buffer_load_dwordx4 v92, s[52:55], s11 offen offset: 0 lds // (op 8572) Load value direct to lds
v_add_co_u32 v92, s[14:15], v92, v84 // (op 8572) least significant half
v_addc_co_u32 v93, s[14:15], 0, v93, s[14:15] // (op 8572) most significant half

s_add_u32 m0, m0, 4096 // (op 8572) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[34:41], a[32:35], v78, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1997) 
buffer_load_dwordx4 v92, s[52:55], s11 offen offset: 0 lds // (op 8572) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[42:49], a[36:39], v78, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1951) 
v_mov_b64 v[84:85], s[34:35] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:77], v[50:57], a[40:43], v78, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1905) 
v_mov_b32 v92, v23 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:77], v[58:65], a[44:47], v78, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1859) 
v_mov_b32 v93, v92 // (op 25) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[34:41], a[48:51], v79, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2767) 
buffer_load_ubyte v30, v93, s[60:63], 0 offen offset:8 // (op 25) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[42:49], a[52:55], v79, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2721) 
v_add_u32 v93, v93, v84 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[50:57], a[56:59], v79, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2675) 
buffer_load_ubyte v31, v93, s[60:63], 0 offen offset:8 // (op 25) Load value


s_barrier  // (op 9427) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[58:65], a[60:63], v79, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2629) 

s_lshl_b64 s[14:15], s[12:13], 6 // (op 8567) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 8567) 

v_lshrrev_b64 v[84:85], 3, s[18:19] // (op 8567) 


v_mov_b64 v[94:95], v[18:19] // (op 8567) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 8567) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 8567) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 8567) 

s_lshl_b64 s[18:19], s[14:15], 1 // (op 8567) 

v_mov_b32 v93, s18 // (op 8567) 
v_mov_b32 v96, s19 // (op 8567) 
v_add_co_u32 v98, s[14:15], v94, v93 // (op 8567) least significant half
v_addc_co_u32 v99, s[14:15], v95, v96, s[14:15] // (op 8567) most significant half




s_add_u32 s11, 16384, s0 // (op 8567) 
s_mov_b32 m0, s11 // (op 8567) reg expression
buffer_load_dwordx4 v98, s[48:51], 0 offen offset: 0 lds // (op 8567) Load value direct to lds
v_add_co_u32 v98, s[14:15], v98, v84 // (op 8567) least significant half
v_addc_co_u32 v99, s[14:15], 0, v99, s[14:15] // (op 8567) most significant half

s_add_u32 m0, m0, 4096 // (op 8567) 
buffer_load_dwordx4 v98, s[48:51], 0 offen offset: 0 lds // (op 8567) Load value direct to lds




v_mov_b64 v[84:85], s[28:29] // (op 17) convert

v_mov_b32 v93, v25 // (op 17) 
s_lshl_b32 s11, s28, 3 // (op 17) 
v_add_u32 v96, v93, s11 // (op 17) 

buffer_load_ubyte v32, v96, s[56:59], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v96, v96, v84 // (op 17) 
buffer_load_ubyte v33, v96, s[56:59], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 9513) 



s_waitcnt vmcnt(15)
ds_write_b8 v89, v28 offset:50176 // (op 259) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v89, v29 offset:50180 // (op 259) Store local data 
s_barrier  // (op 9507) 



s_waitcnt vmcnt(11)
ds_write_b8 v88, v14 offset:50688 // (op 258) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v88, v15 offset:50816 // (op 258) Store local data 
s_barrier  // (op 9497) 



ds_read_b128 v[96:99], v8 offset:24576 // (op 251) Load local data 
ds_read_b128 v[100:103], v8 offset:24640 // (op 251) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 9510) 



ds_read_u8 v84, v9 offset:50176 // (op 255) Load local data 



ds_read_b128 v[104:107], v8 offset:28672 // (op 385) Load local data 
ds_read_b128 v[108:111], v8 offset:28736 // (op 385) Load local data 



ds_read_u8 v27, v9 offset:50304 // (op 389) Load local data 



ds_read_b128 v[112:115], v8 offset:32768 // (op 561) Load local data 
ds_read_b128 v[116:119], v8 offset:32832 // (op 561) Load local data 



ds_read_u8 v78, v9 offset:50432 // (op 565) Load local data 



ds_read_b128 v[120:123], v8 offset:36864 // (op 737) Load local data 
ds_read_b128 v[124:127], v8 offset:36928 // (op 737) Load local data 



ds_read_u8 v85, v9 offset:50560 // (op 741) Load local data 
s_barrier  // (op 9491) 
s_barrier  // (op 9558) 



ds_read_b64_tr_b4 v[66:67], v7 offset:40960 // (op 249) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v7 offset:41984 // (op 249) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 9504) 


ds_read_u8 v26, v6 offset:50688 // (op 253) Load local data 



ds_read_b64_tr_b4 v[70:71], v7 offset:40976 // (op 1092) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v7 offset:42000 // (op 1092) Transpose load local data 


ds_read_u8 v24, v6 offset:50720 // (op 1095) Load local data 



ds_read_b64_tr_b4 v[74:75], v7 offset:40992 // (op 1862) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v7 offset:42016 // (op 1862) Transpose load local data 


ds_read_u8 v16, v6 offset:50752 // (op 1865) Load local data 



ds_read_b64_tr_b4 v[80:81], v7 offset:41008 // (op 2632) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v7 offset:42032 // (op 2632) Transpose load local data 


ds_read_u8 v79, v6 offset:50784 // (op 2635) Load local data 
s_barrier  // (op 9592) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[96:103], a[0:3], v26, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 256) 
s_barrier  // (op 9501) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[104:111], a[4:7], v26, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 381) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 8595) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[112:119], a[8:11], v26, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 557) 
v_mov_b64 v[128:129], s[14:15] // (op 8595) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[120:127], a[12:15], v26, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 733) 
v_mov_b64 v[130:131], v[90:91] // (op 8595) 
s_add_u32 s11, 24576, s7 // (op 8595) 
s_mov_b32 m0, s11 // (op 8595) reg expression
s_mov_b32 s14, 384 // (op 8595) call()
buffer_load_dwordx4 v130, s[52:55], s14 offen offset: 0 lds // (op 8595) Load value direct to lds
v_add_co_u32 v130, s[18:19], v130, v128 // (op 8595) least significant half
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[96:103], a[16:19], v24, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1228) 
v_addc_co_u32 v131, s[18:19], 0, v131, s[18:19] // (op 8595) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[104:111], a[20:23], v24, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1182) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[112:119], a[24:27], v24, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1136) 
s_add_u32 m0, m0, 4096 // (op 8595) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[120:127], a[28:31], v24, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1090) 
buffer_load_dwordx4 v130, s[52:55], s14 offen offset: 0 lds // (op 8595) Load value direct to lds
v_add_co_u32 v130, s[18:19], v130, v128 // (op 8595) least significant half
v_addc_co_u32 v131, s[18:19], 0, v131, s[18:19] // (op 8595) most significant half

s_add_u32 m0, m0, 4096 // (op 8595) 
buffer_load_dwordx4 v130, s[52:55], s14 offen offset: 0 lds // (op 8595) Load value direct to lds
v_add_co_u32 v130, s[18:19], v130, v128 // (op 8595) least significant half
v_addc_co_u32 v131, s[18:19], 0, v131, s[18:19] // (op 8595) most significant half

s_add_u32 m0, m0, 4096 // (op 8595) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[96:103], a[32:35], v16, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1998) 
buffer_load_dwordx4 v130, s[52:55], s14 offen offset: 0 lds // (op 8595) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[104:111], a[36:39], v16, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1952) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:77], v[112:119], a[40:43], v16, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1906) 
v_add_co_u32 v20, vcc, 256, v20 // (op 8741) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:77], v[120:127], a[44:47], v16, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1860) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 8741) most significant half
v_mov_b64 v[90:91], s[34:35] // (op 262) convert

v_mov_b32 v128, v92 // (op 262) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[96:103], a[48:51], v79, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2768) 
buffer_load_ubyte v28, v128, s[60:63], 0 offen offset:12 // (op 262) Load value


v_add_u32 v128, v128, v90 // (op 262) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[104:111], a[52:55], v79, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2722) 
buffer_load_ubyte v29, v128, s[60:63], 0 offen offset:12 // (op 262) Load value




s_barrier  // (op 9494) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[112:119], a[56:59], v79, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2676) 


v_add_u32 v23, 8, v23 // (op 8869) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[120:127], a[60:63], v79, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2630) 
s_lshl_b64 s[14:15], s[12:13], 6 // (op 8590) 


s_lshl_b64 s[18:19], s[14:15], 2 // (op 8590) 

v_lshrrev_b64 v[84:85], 3, s[18:19] // (op 8590) 


s_lshl_b64 s[14:15], s[12:13], 7 // (op 8590) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 8590) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 8590) 

s_mul_i32 s19, 3, s15 // (op 8590) most significant: low of low * high
s_mul_i32 s18, 3, s14 // (op 8590) least significant
s_mul_hi_u32 s11, 3, s14 // (op 8590) most significant: high of low * low
s_add_u32 s19, s19, s11 // (op 8590) most significant: sum


v_mov_b32 v90, s18 // (op 8590) 
v_mov_b32 v91, s19 // (op 8590) 
v_add_co_u32 v96, s[14:15], v94, v90 // (op 8590) least significant half
v_addc_co_u32 v97, s[14:15], v95, v91, s[14:15] // (op 8590) most significant half




s_add_u32 s11, 40960, s6 // (op 8590) 
s_mov_b32 m0, s11 // (op 8590) reg expression
buffer_load_dwordx4 v96, s[48:51], 0 offen offset: 0 lds // (op 8590) Load value direct to lds
v_add_co_u32 v96, s[14:15], v96, v84 // (op 8590) least significant half
v_addc_co_u32 v97, s[14:15], 0, v97, s[14:15] // (op 8590) most significant half

s_add_u32 m0, m0, 4096 // (op 8590) 
buffer_load_dwordx4 v96, s[48:51], 0 offen offset: 0 lds // (op 8590) Load value direct to lds





v_mov_b64 v[84:85], s[28:29] // (op 261) convert

s_lshl_b32 s11, s28, 2 // (op 261) 
s_mul_i32 s14, 3, s11 // (op 261) 

v_add_u32 v90, v93, s14 // (op 261) 

s_lshl_b64 s[14:15], s[12:13], 7 // (op 8677) 
buffer_load_ubyte v14, v90, s[56:59], 0 offen offset: 0 // (op 261) Load value
v_add_u32 v90, v90, v84 // (op 261) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 8677) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 8677) 

s_lshl_b64 s[18:19], s[14:15], 1 // (op 8677) 

v_mov_b32 v91, s18 // (op 8677) 
v_mov_b32 v92, s19 // (op 8677) 
v_add_co_u32 v18, s[14:15], v18, v91 // (op 8677) least significant half
v_addc_co_u32 v19, s[14:15], v19, v92, s[14:15] // (op 8677) most significant half




buffer_load_ubyte v15, v90, s[56:59], 0 offen offset: 0 // (op 261) Load value



s_barrier  // (op 9477) 
s_lshl_b32 s11, s28, 3 // (op 8805) 



v_add_u32 v25, v25, s11 // (op 8805) 

s_waitcnt vmcnt(15)
ds_write_b8 v87, v30 offset:49152 // (op 30) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v87, v31 offset:49156 // (op 30) Store local data 
s_barrier  // (op 9460) 



s_waitcnt vmcnt(11)
ds_write_b8 v86, v32 offset:49664 // (op 21) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v86, v33 offset:49792 // (op 21) Store local data 
s_barrier  // (op 9449) 



ds_read_b128 v[34:37], v4 // (op 77) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 9484) 



ds_read_u8 v22, v5 offset:49152 // (op 87) Load local data 



ds_read_b128 v[42:45], v4 offset:4096 // (op 375) Load local data 
ds_read_b128 v[46:49], v4 offset:4160 // (op 375) Load local data 



ds_read_u8 v24, v5 offset:49280 // (op 379) Load local data 



ds_read_b128 v[50:53], v4 offset:8192 // (op 551) Load local data 
ds_read_b128 v[54:57], v4 offset:8256 // (op 551) Load local data 



ds_read_u8 v16, v5 offset:49408 // (op 555) Load local data 



ds_read_b128 v[58:61], v4 offset:12288 // (op 727) Load local data 
ds_read_b128 v[62:65], v4 offset:12352 // (op 727) Load local data 



ds_read_u8 v17, v5 offset:49536 // (op 731) Load local data 
s_barrier  // (op 9433) 



ds_read_b64_tr_b4 v[66:67], v3 offset:16384 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v3 offset:17408 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 9467) 


ds_read_u8 v26, v0 offset:49664 // (op 82) Load local data 



ds_read_b64_tr_b4 v[70:71], v3 offset:16400 // (op 1084) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v3 offset:17424 // (op 1084) Transpose load local data 


ds_read_u8 v27, v0 offset:49696 // (op 1087) Load local data 



ds_read_b64_tr_b4 v[74:75], v3 offset:16416 // (op 1854) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v3 offset:17440 // (op 1854) Transpose load local data 


ds_read_u8 v78, v0 offset:49728 // (op 1857) Load local data 



ds_read_b64_tr_b4 v[80:81], v3 offset:16432 // (op 2624) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v3 offset:17456 // (op 2624) Transpose load local data 


ds_read_u8 v79, v0 offset:49760 // (op 2627) Load local data 
s_ashr_i64 s[14:15], s[22:23], 7 // (op 2528) 
s_mov_b32 s11, s14 // (op 2528) convert

s_and_b32 s14, s11, 4294967294 // (op 2528) 

s_cmp_lt_u32 s10, s14 // (op 2528) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2528 // (op 2528) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2528 if true)
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2528:





















s_barrier  // (op 9572) 
v_and_b32 v16, -4, v1 // (op 8625) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8625) 
v_sub_u32 v18, v1, v16 // (op 8625) 

s_lshr_b32 s6, s0, 1 // (op 8625) 

v_lshlrev_b32 v16, 5, v18 // (op 8625) 

v_lshl_add_u32 v18, s16, 7, v16 // (op 8625) 

v_mov_b32 v25, 0 // (op 8625) convert
v_mov_b32 v24, v18 // (op 8625) convert

v_lshrrev_b32 v16, 2, v1 // (op 8625) 
v_lshl_add_u32 v18, s6, 8, v16 // (op 8625) 


v_mul_lo_u32 v23, v18, s13 // (op 8625) most significant: low of low * high
v_mul_hi_u32 v16, v18, s12 // (op 8625) most significant: high of low * low
v_mul_lo_u32 v22, v18, s12 // (op 8625) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 8625) most significant: sum


v_add_co_u32 v18, s[0:1], v24, v22 // (op 8625) least significant half
v_addc_co_u32 v19, s[0:1], v25, v23, s[0:1] // (op 8625) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 8625) 


v_lshrrev_b64 v[20:21], 3, v[18:19] // (op 8625) 

s_mul_i32 s1, s12, s23 // (op 8627) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8627) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8627) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8627) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8627) most significant: sum
s_add_u32 s1, s1, s7 // (op 8627) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8627) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8627) 

s_mov_b32 s48, s20 // (op 8627) reg expression
s_mov_b32 s49, s21 // (op 8627) reg expression
s_mov_b32 s50, s0 // (op 8627) reg expression
s_mov_b32 s51, 131072 // (op 8627) call()

v_and_b32 v16, -8, v1 // (op 8707) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8707) 
v_sub_u32 v18, v1, v16 // (op 8707) 

s_lshr_b32 s6, s0, 1 // (op 8707) 

v_lshlrev_b32 v16, 4, v18 // (op 8707) 

v_lshl_add_u32 v18, s6, 8, v16 // (op 8707) 


v_mov_b32 v23, 0 // (op 8707) convert
v_mov_b32 v22, v18 // (op 8707) convert

v_lshrrev_b32 v16, 3, v1 // (op 8707) 
v_lshl_add_u32 v18, s17, 7, v16 // (op 8707) 

v_mul_lo_u32 v25, v18, s25 // (op 8707) most significant: low of low * high
v_mul_hi_u32 v16, v18, s24 // (op 8707) most significant: high of low * low
v_mul_lo_u32 v24, v18, s24 // (op 8707) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 8707) most significant: sum


v_add_co_u32 v18, s[0:1], v22, v24 // (op 8707) least significant half
v_addc_co_u32 v19, s[0:1], v23, v25, s[0:1] // (op 8707) most significant half



s_mul_i32 s1, s24, s9 // (op 8709) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8709) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8709) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8709) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8709) most significant: sum
s_add_u32 s1, s1, s7 // (op 8709) most significant: sum


s_mov_b32 s52, s26 // (op 8709) reg expression
s_mov_b32 s53, s27 // (op 8709) reg expression
s_mov_b32 s54, s0 // (op 8709) reg expression
s_mov_b32 s55, 131072 // (op 8709) call()

v_and_b32 v16, -128, v1 // (op 8771) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8771) 
v_sub_u32 v22, v1, v16 // (op 8771) 

s_lshr_b32 s6, s0, 1 // (op 8771) 

v_lshl_add_u32 v16, s16, 7, v22 // (op 8771) 

v_lshrrev_b32 v22, 7, v1 // (op 8771) 
v_lshlrev_b32 v23, 1, v22 // (op 8771) 

v_lshl_add_u32 v22, s6, 3, v23 // (op 8771) 


v_mul_lo_u32 v23, v22, s28 // (op 8771) 

v_add_u32 v22, v16, v23 // (op 8771) 


s_mul_i32 s1, s28, s33 // (op 8773) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 8773) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8773) least significant
s_mul_hi_u32 s7, s28, s32 // (op 8773) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8773) most significant: sum
s_add_u32 s1, s1, s7 // (op 8773) most significant: sum


s_mov_b32 s56, s30 // (op 8773) reg expression
s_mov_b32 s57, s31 // (op 8773) reg expression
s_mov_b32 s58, s0 // (op 8773) reg expression
s_mov_b32 s59, 131072 // (op 8773) call()

v_and_b32 v16, -4, v1 // (op 8835) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8835) 
v_sub_u32 v23, v1, v16 // (op 8835) 

s_lshr_b32 s6, s0, 1 // (op 8835) 

v_lshrrev_b32 v16, 2, v1 // (op 8835) 
v_lshl_add_u32 v24, s6, 3, v23 // (op 8835) 


v_lshlrev_b32 v23, 1, v16 // (op 8835) 

v_lshl_add_u32 v16, s17, 7, v23 // (op 8835) 

v_mul_lo_u32 v23, v16, s34 // (op 8835) 

v_add_u32 v16, v24, v23 // (op 8835) 


s_mul_i32 s1, s34, s9 // (op 8837) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 8837) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8837) least significant
s_mul_hi_u32 s7, s34, s8 // (op 8837) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8837) most significant: sum
s_add_u32 s1, s1, s7 // (op 8837) most significant: sum


s_mov_b32 s60, s36 // (op 8837) reg expression
s_mov_b32 s61, s37 // (op 8837) reg expression
s_mov_b32 s62, s0 // (op 8837) reg expression
s_mov_b32 s63, 131072 // (op 8837) call()

v_and_b32 v23, -4, v1 // (op 8933) 
v_sub_u32 v24, v1, v23 // (op 8933) 

v_lshlrev_b32 v23, 5, v24 // (op 8933) 

v_lshrrev_b32 v24, 2, v1 // (op 8933) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 8933) 


v_lshlrev_b32 v23, 2, v25 // (op 8933) 

v_lshrrev_b32 v24, 3, v23 // (op 8933) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 8933) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -8, v1 // (op 8990) 
v_sub_u32 v24, v1, v23 // (op 8990) 

v_lshlrev_b32 v23, 4, v24 // (op 8990) 

v_lshrrev_b32 v24, 3, v1 // (op 8990) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 8990) 


s_nop 0
v_readfirstlane_b32 s1, v25 // (op 8990) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -128, v1 // (op 9047) 
v_sub_u32 v24, v1, v23 // (op 9047) 

v_lshrrev_b32 v23, 7, v1 // (op 9047) 
v_lshl_add_u32 v25, v23, 8, v24 // (op 9047) 


v_and_b32 v23, -4, v1 // (op 9105) 
v_sub_u32 v24, v1, v23 // (op 9105) 

v_lshrrev_b32 v23, 2, v1 // (op 9105) 
v_lshl_add_u32 v30, v23, 3, v24 // (op 9105) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 2596) 
s_and_b32 s10, s6, 4294967294 // (op 2596) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 2595) 
s_mov_b32 s11, s6 // (op 2595) convert

s_cmp_lt_u32 s10, s11 // (op 2595) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2595 // (op 2595) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2595 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2595:
s_barrier  // (op 9430) 
s_add_u32 s10, s10, 1 // (op 2597) 

s_lshl_b64 s[6:7], s[12:13], 6 // (op 8577) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 8577) 

v_lshrrev_b64 v[14:15], 3, s[14:15] // (op 8577) 


v_mov_b64 v[26:27], v[20:21] // (op 8577) 
v_mov_b64 v[28:29], v[26:27] // (op 8577) reg expression
s_add_u32 s11, 16384, s0 // (op 8577) 
s_mov_b32 m0, s11 // (op 8577) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8577) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8577) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8577) most significant half

s_add_u32 m0, m0, 4096 // (op 8577) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8577) Load value direct to lds




s_lshl_b64 s[6:7], s[12:13], 7 // (op 8615) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 8615) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 8615) 

v_mov_b32 v17, s6 // (op 8615) 
v_mov_b32 v14, s7 // (op 8615) 
v_add_co_u32 v20, s[14:15], v20, v17 // (op 8615) least significant half
v_addc_co_u32 v21, s[14:15], v21, v14, s[14:15] // (op 8615) most significant half




s_barrier  // (op 9446) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 8583) 
v_mov_b64 v[14:15], s[6:7] // (op 8583) convert


v_mov_b64 v[26:27], v[18:19] // (op 8583) 
v_mov_b64 v[28:29], v[26:27] // (op 8583) 
s_mov_b32 m0, s1 // (op 8583) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8583) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8583) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8583) most significant half

s_add_u32 m0, m0, 4096 // (op 8583) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8583) Load value direct to lds



v_add_co_u32 v18, vcc, 128, v18 // (op 8699) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 8699) most significant half

v_mov_b64 v[14:15], s[28:29] // (op 195) convert

v_mov_b32 v17, v22 // (op 195) 
v_mov_b32 v23, v17 // (op 195) reg expression
buffer_load_ubyte v26, v23, s[56:59], 0 offen offset: 0 // (op 195) Load value
v_add_u32 v23, v23, v14 // (op 195) 
buffer_load_ubyte v27, v23, s[56:59], 0 offen offset: 0 // (op 195) Load value



s_lshl_b32 s11, s28, 2 // (op 8763) 
v_add_u32 v22, v22, s11 // (op 8763) 


s_barrier  // (op 9463) 
v_mov_b64 v[14:15], s[34:35] // (op 196) convert




v_mov_b32 v17, v16 // (op 196) 
v_mov_b32 v23, v17 // (op 196) 
buffer_load_ubyte v28, v23, s[60:63], 0 offen offset: 0 // (op 196) Load value
v_add_u32 v23, v23, v14 // (op 196) 
s_waitcnt vmcnt(2)
ds_write_b8 v25, v26 offset:49664 // (op 192) Store local data 
buffer_load_ubyte v29, v23, s[60:63], 0 offen offset: 0 // (op 196) Load value



v_add_u32 v16, 4, v16 // (op 8827) 
s_waitcnt vmcnt(2)
ds_write_b8 v25, v27 offset:49792 // (op 192) Store local data 

s_barrier  // (op 9436) 



s_barrier  // (op 9480) 
ds_read_b64_tr_b4 v[32:33], v3 offset:16384 // (op 183) Transpose load local data 



ds_read_b64_tr_b4 v[34:35], v3 offset:17408 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v30, v28 offset:49152 // (op 193) Store local data 
s_barrier  // (op 9453) 
s_waitcnt vmcnt(0)
ds_write_b8 v30, v29 offset:49156 // (op 193) Store local data 




ds_read_b128 v[36:39], v4 // (op 185) Load local data 
ds_read_b128 v[40:43], v4 offset:64 // (op 185) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 9470) 


ds_read_u8 v17, v0 offset:49664 // (op 187) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9487) 



ds_read_u8 v31, v5 offset:49152 // (op 189) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:35], v[36:43], a[0:3], v17, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 190) 
ds_read_b128 v[44:47], v4 offset:4096 // (op 354) Load local data 
ds_read_b128 v[48:51], v4 offset:4160 // (op 354) Load local data 



ds_read_u8 v14, v5 offset:49280 // (op 358) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:35], v[44:51], a[4:7], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 359) 
ds_read_b128 v[52:55], v4 offset:8192 // (op 530) Load local data 
ds_read_b128 v[56:59], v4 offset:8256 // (op 530) Load local data 



ds_read_u8 v15, v5 offset:49408 // (op 534) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:35], v[52:59], a[8:11], v17, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 535) 
ds_read_b128 v[60:63], v4 offset:12288 // (op 706) Load local data 
ds_read_b128 v[64:67], v4 offset:12352 // (op 706) Load local data 



ds_read_u8 v23, v5 offset:49536 // (op 710) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:35], v[60:67], a[12:15], v17, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 711) 
ds_read_b64_tr_b4 v[26:27], v3 offset:16400 // (op 968) Transpose load local data 


ds_read_b64_tr_b4 v[28:29], v3 offset:17424 // (op 968) Transpose load local data 


ds_read_u8 v17, v0 offset:49696 // (op 971) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:29], v[36:43], a[16:19], v17, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 973) 
ds_read_b64_tr_b4 v[32:33], v3 offset:16416 // (op 1738) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:29], v[44:51], a[20:23], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1004) 
ds_read_b64_tr_b4 v[34:35], v3 offset:17440 // (op 1738) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[52:59], a[24:27], v17, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1035) 
ds_read_u8 v24, v0 offset:49728 // (op 1741) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:29], v[60:67], a[28:31], v17, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1066) 
ds_read_b64_tr_b4 v[68:69], v3 offset:16432 // (op 2508) Transpose load local data 


ds_read_b64_tr_b4 v[70:71], v3 offset:17456 // (op 2508) Transpose load local data 


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[32:35], v[36:43], a[32:35], v24, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1743) 
ds_read_u8 v17, v0 offset:49760 // (op 2511) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[32:35], v[44:51], a[36:39], v24, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1774) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[32:35], v[52:59], a[40:43], v24, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1805) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[32:35], v[60:67], a[44:47], v24, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1836) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[36:43], a[48:51], v17, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2513) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[44:51], a[52:55], v17, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2544) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[52:59], a[56:59], v17, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2575) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[60:67], a[60:63], v17, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2606) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 2595) 
s_mov_b32 s11, s6 // (op 2595) convert

s_cmp_lt_u32 s10, s11 // (op 2595) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2595 // (op 2595) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2595 if true)
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2595:























s_barrier  // (op 9580) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_135
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



v_accvgpr_read v6, a4 // (op 3565) 
v_accvgpr_read v7, a5 // (op 3565) 
v_accvgpr_read v8, a6 // (op 3565) 
v_accvgpr_read v9, a7 // (op 3565) 
v_mul_f32 v14, v11, v6 // (op 3565) 
v_mul_f32 v15, v11, v7 // (op 3565) 
v_mul_f32 v16, v11, v8 // (op 3565) 
v_mul_f32 v17, v11, v9 // (op 3565) 

v_lshrrev_b32 v6, 16, v14 // (op 3567) 
v_lshrrev_b32 v7, 16, v15 // (op 3567) 
v_lshrrev_b32 v8, 16, v16 // (op 3567) 
v_lshrrev_b32 v9, 16, v17 // (op 3567) 


v_pack_b32_f16 v4, v6, v7 // (op 3568) 
v_pack_b32_f16 v5, v8, v9 // (op 3568) 

s_lshl_b32 s0, s4, 6 // (op 3568) 
v_add_u32 v0, v13, s0 // (op 3568) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3568) Store value



v_accvgpr_read v6, a8 // (op 3579) 
v_accvgpr_read v7, a9 // (op 3579) 
v_accvgpr_read v8, a10 // (op 3579) 
v_accvgpr_read v9, a11 // (op 3579) 
v_mul_f32 v14, v11, v6 // (op 3579) 
v_mul_f32 v15, v11, v7 // (op 3579) 
v_mul_f32 v16, v11, v8 // (op 3579) 
v_mul_f32 v17, v11, v9 // (op 3579) 

v_lshrrev_b32 v6, 16, v14 // (op 3581) 
v_lshrrev_b32 v7, 16, v15 // (op 3581) 
v_lshrrev_b32 v8, 16, v16 // (op 3581) 
v_lshrrev_b32 v9, 16, v17 // (op 3581) 


v_pack_b32_f16 v4, v6, v7 // (op 3582) 
v_pack_b32_f16 v5, v8, v9 // (op 3582) 

s_lshl_b32 s0, s4, 7 // (op 3582) 
v_add_u32 v0, v13, s0 // (op 3582) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3582) Store value



v_accvgpr_read v6, a12 // (op 3593) 
v_accvgpr_read v7, a13 // (op 3593) 
v_accvgpr_read v8, a14 // (op 3593) 
v_accvgpr_read v9, a15 // (op 3593) 
v_mul_f32 v14, v11, v6 // (op 3593) 
v_mul_f32 v15, v11, v7 // (op 3593) 
v_mul_f32 v16, v11, v8 // (op 3593) 
v_mul_f32 v17, v11, v9 // (op 3593) 

v_lshrrev_b32 v6, 16, v14 // (op 3595) 
v_lshrrev_b32 v7, 16, v15 // (op 3595) 
v_lshrrev_b32 v8, 16, v16 // (op 3595) 
v_lshrrev_b32 v9, 16, v17 // (op 3595) 


v_pack_b32_f16 v4, v6, v7 // (op 3596) 
v_pack_b32_f16 v5, v8, v9 // (op 3596) 

s_lshl_b32 s0, s4, 6 // (op 3596) 
s_mul_i32 s1, 3, s0 // (op 3596) 

v_add_u32 v0, v13, s1 // (op 3596) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3596) Store value



v_accvgpr_read v6, a16 // (op 3629) 
v_accvgpr_read v7, a17 // (op 3629) 
v_accvgpr_read v8, a18 // (op 3629) 
v_accvgpr_read v9, a19 // (op 3629) 
v_mul_f32 v14, v11, v6 // (op 3629) 
v_mul_f32 v15, v11, v7 // (op 3629) 
v_mul_f32 v16, v11, v8 // (op 3629) 
v_mul_f32 v17, v11, v9 // (op 3629) 

v_lshrrev_b32 v6, 16, v14 // (op 3631) 
v_lshrrev_b32 v7, 16, v15 // (op 3631) 
v_lshrrev_b32 v8, 16, v16 // (op 3631) 
v_lshrrev_b32 v9, 16, v17 // (op 3631) 


v_pack_b32_f16 v4, v6, v7 // (op 3633) 
v_pack_b32_f16 v5, v8, v9 // (op 3633) 

v_add_u32 v0, 64, v13 // (op 3633) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3633) Store value



v_accvgpr_read v6, a20 // (op 3643) 
v_accvgpr_read v7, a21 // (op 3643) 
v_accvgpr_read v8, a22 // (op 3643) 
v_accvgpr_read v9, a23 // (op 3643) 
v_mul_f32 v14, v11, v6 // (op 3643) 
v_mul_f32 v15, v11, v7 // (op 3643) 
v_mul_f32 v16, v11, v8 // (op 3643) 
v_mul_f32 v17, v11, v9 // (op 3643) 

v_lshrrev_b32 v6, 16, v14 // (op 3645) 
v_lshrrev_b32 v7, 16, v15 // (op 3645) 
v_lshrrev_b32 v8, 16, v16 // (op 3645) 
v_lshrrev_b32 v9, 16, v17 // (op 3645) 


v_pack_b32_f16 v4, v6, v7 // (op 3635) 
v_pack_b32_f16 v5, v8, v9 // (op 3635) 

s_lshl_b32 s0, s4, 6 // (op 3635) 
s_add_u32 s0, s0, 64 // (op 3635) 
v_add_u32 v0, v13, s0 // (op 3635) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3635) Store value



v_accvgpr_read v6, a24 // (op 3649) 
v_accvgpr_read v7, a25 // (op 3649) 
v_accvgpr_read v8, a26 // (op 3649) 
v_accvgpr_read v9, a27 // (op 3649) 
v_mul_f32 v14, v11, v6 // (op 3649) 
v_mul_f32 v15, v11, v7 // (op 3649) 
v_mul_f32 v16, v11, v8 // (op 3649) 
v_mul_f32 v17, v11, v9 // (op 3649) 

v_lshrrev_b32 v6, 16, v14 // (op 3651) 
v_lshrrev_b32 v7, 16, v15 // (op 3651) 
v_lshrrev_b32 v8, 16, v16 // (op 3651) 
v_lshrrev_b32 v9, 16, v17 // (op 3651) 


v_pack_b32_f16 v4, v6, v7 // (op 3637) 
v_pack_b32_f16 v5, v8, v9 // (op 3637) 

s_lshl_b32 s0, s4, 7 // (op 3637) 
s_add_u32 s0, s0, 64 // (op 3637) 
v_add_u32 v0, v13, s0 // (op 3637) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3637) Store value



v_accvgpr_read v6, a28 // (op 3655) 
v_accvgpr_read v7, a29 // (op 3655) 
v_accvgpr_read v8, a30 // (op 3655) 
v_accvgpr_read v9, a31 // (op 3655) 
v_mul_f32 v14, v11, v6 // (op 3655) 
v_mul_f32 v15, v11, v7 // (op 3655) 
v_mul_f32 v16, v11, v8 // (op 3655) 
v_mul_f32 v17, v11, v9 // (op 3655) 

v_lshrrev_b32 v6, 16, v14 // (op 3657) 
v_lshrrev_b32 v7, 16, v15 // (op 3657) 
v_lshrrev_b32 v8, 16, v16 // (op 3657) 
v_lshrrev_b32 v9, 16, v17 // (op 3657) 


v_pack_b32_f16 v4, v6, v7 // (op 3639) 
v_pack_b32_f16 v5, v8, v9 // (op 3639) 

s_lshl_b32 s0, s4, 6 // (op 3639) 
s_mul_i32 s1, 3, s0 // (op 3639) 
s_add_u32 s1, s1, 64 // (op 3639) 

v_add_u32 v0, v13, s1 // (op 3639) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3639) Store value



v_accvgpr_read v6, a32 // (op 3705) 
v_accvgpr_read v7, a33 // (op 3705) 
v_accvgpr_read v8, a34 // (op 3705) 
v_accvgpr_read v9, a35 // (op 3705) 
v_mul_f32 v14, v11, v6 // (op 3705) 
v_mul_f32 v15, v11, v7 // (op 3705) 
v_mul_f32 v16, v11, v8 // (op 3705) 
v_mul_f32 v17, v11, v9 // (op 3705) 

v_lshrrev_b32 v6, 16, v14 // (op 3707) 
v_lshrrev_b32 v7, 16, v15 // (op 3707) 
v_lshrrev_b32 v8, 16, v16 // (op 3707) 
v_lshrrev_b32 v9, 16, v17 // (op 3707) 


v_pack_b32_f16 v4, v6, v7 // (op 3709) 
v_pack_b32_f16 v5, v8, v9 // (op 3709) 

v_add_u32 v0, 128, v13 // (op 3709) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3709) Store value



v_accvgpr_read v6, a36 // (op 3719) 
v_accvgpr_read v7, a37 // (op 3719) 
v_accvgpr_read v8, a38 // (op 3719) 
v_accvgpr_read v9, a39 // (op 3719) 
v_mul_f32 v14, v11, v6 // (op 3719) 
v_mul_f32 v15, v11, v7 // (op 3719) 
v_mul_f32 v16, v11, v8 // (op 3719) 
v_mul_f32 v17, v11, v9 // (op 3719) 

v_lshrrev_b32 v6, 16, v14 // (op 3721) 
v_lshrrev_b32 v7, 16, v15 // (op 3721) 
v_lshrrev_b32 v8, 16, v16 // (op 3721) 
v_lshrrev_b32 v9, 16, v17 // (op 3721) 


v_pack_b32_f16 v4, v6, v7 // (op 3711) 
v_pack_b32_f16 v5, v8, v9 // (op 3711) 

s_lshl_b32 s0, s4, 6 // (op 3711) 
s_add_u32 s0, s0, 128 // (op 3711) 
v_add_u32 v0, v13, s0 // (op 3711) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3711) Store value



v_accvgpr_read v6, a40 // (op 3725) 
v_accvgpr_read v7, a41 // (op 3725) 
v_accvgpr_read v8, a42 // (op 3725) 
v_accvgpr_read v9, a43 // (op 3725) 
v_mul_f32 v14, v11, v6 // (op 3725) 
v_mul_f32 v15, v11, v7 // (op 3725) 
v_mul_f32 v16, v11, v8 // (op 3725) 
v_mul_f32 v17, v11, v9 // (op 3725) 

v_lshrrev_b32 v6, 16, v14 // (op 3727) 
v_lshrrev_b32 v7, 16, v15 // (op 3727) 
v_lshrrev_b32 v8, 16, v16 // (op 3727) 
v_lshrrev_b32 v9, 16, v17 // (op 3727) 


v_pack_b32_f16 v4, v6, v7 // (op 3713) 
v_pack_b32_f16 v5, v8, v9 // (op 3713) 

s_lshl_b32 s0, s4, 7 // (op 3713) 
s_add_u32 s0, s0, 128 // (op 3713) 
v_add_u32 v0, v13, s0 // (op 3713) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3713) Store value



v_accvgpr_read v6, a44 // (op 3731) 
v_accvgpr_read v7, a45 // (op 3731) 
v_accvgpr_read v8, a46 // (op 3731) 
v_accvgpr_read v9, a47 // (op 3731) 
v_mul_f32 v14, v11, v6 // (op 3731) 
v_mul_f32 v15, v11, v7 // (op 3731) 
v_mul_f32 v16, v11, v8 // (op 3731) 
v_mul_f32 v17, v11, v9 // (op 3731) 

v_lshrrev_b32 v6, 16, v14 // (op 3733) 
v_lshrrev_b32 v7, 16, v15 // (op 3733) 
v_lshrrev_b32 v8, 16, v16 // (op 3733) 
v_lshrrev_b32 v9, 16, v17 // (op 3733) 


v_pack_b32_f16 v4, v6, v7 // (op 3715) 
v_pack_b32_f16 v5, v8, v9 // (op 3715) 

s_lshl_b32 s0, s4, 6 // (op 3715) 
s_mul_i32 s1, 3, s0 // (op 3715) 
s_add_u32 s1, s1, 128 // (op 3715) 

v_add_u32 v0, v13, s1 // (op 3715) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3715) Store value



v_accvgpr_read v6, a48 // (op 3781) 
v_accvgpr_read v7, a49 // (op 3781) 
v_accvgpr_read v8, a50 // (op 3781) 
v_accvgpr_read v9, a51 // (op 3781) 
v_mul_f32 v14, v11, v6 // (op 3781) 
v_mul_f32 v15, v11, v7 // (op 3781) 
v_mul_f32 v16, v11, v8 // (op 3781) 
v_mul_f32 v17, v11, v9 // (op 3781) 

v_lshrrev_b32 v6, 16, v14 // (op 3783) 
v_lshrrev_b32 v7, 16, v15 // (op 3783) 
v_lshrrev_b32 v8, 16, v16 // (op 3783) 
v_lshrrev_b32 v9, 16, v17 // (op 3783) 


v_pack_b32_f16 v4, v6, v7 // (op 3785) 
v_pack_b32_f16 v5, v8, v9 // (op 3785) 

v_add_u32 v0, 192, v13 // (op 3785) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3785) Store value



v_accvgpr_read v6, a52 // (op 3795) 
v_accvgpr_read v7, a53 // (op 3795) 
v_accvgpr_read v8, a54 // (op 3795) 
v_accvgpr_read v9, a55 // (op 3795) 
v_mul_f32 v14, v11, v6 // (op 3795) 
v_mul_f32 v15, v11, v7 // (op 3795) 
v_mul_f32 v16, v11, v8 // (op 3795) 
v_mul_f32 v17, v11, v9 // (op 3795) 

v_lshrrev_b32 v6, 16, v14 // (op 3797) 
v_lshrrev_b32 v7, 16, v15 // (op 3797) 
v_lshrrev_b32 v8, 16, v16 // (op 3797) 
v_lshrrev_b32 v9, 16, v17 // (op 3797) 


v_pack_b32_f16 v4, v6, v7 // (op 3787) 
v_pack_b32_f16 v5, v8, v9 // (op 3787) 

s_lshl_b32 s0, s4, 6 // (op 3787) 
s_add_u32 s0, s0, 192 // (op 3787) 
v_add_u32 v0, v13, s0 // (op 3787) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3787) Store value



v_accvgpr_read v6, a56 // (op 3801) 
v_accvgpr_read v7, a57 // (op 3801) 
v_accvgpr_read v8, a58 // (op 3801) 
v_accvgpr_read v9, a59 // (op 3801) 
v_mul_f32 v14, v11, v6 // (op 3801) 
v_mul_f32 v15, v11, v7 // (op 3801) 
v_mul_f32 v16, v11, v8 // (op 3801) 
v_mul_f32 v17, v11, v9 // (op 3801) 

v_lshrrev_b32 v6, 16, v14 // (op 3803) 
v_lshrrev_b32 v7, 16, v15 // (op 3803) 
v_lshrrev_b32 v8, 16, v16 // (op 3803) 
v_lshrrev_b32 v9, 16, v17 // (op 3803) 


v_pack_b32_f16 v4, v6, v7 // (op 3789) 
v_pack_b32_f16 v5, v8, v9 // (op 3789) 

s_lshl_b32 s0, s4, 7 // (op 3789) 
s_add_u32 s0, s0, 192 // (op 3789) 
v_add_u32 v0, v13, s0 // (op 3789) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3789) Store value



v_accvgpr_read v6, a60 // (op 3807) 
v_accvgpr_read v7, a61 // (op 3807) 
v_accvgpr_read v8, a62 // (op 3807) 
v_accvgpr_read v9, a63 // (op 3807) 
v_mul_f32 v14, v11, v6 // (op 3807) 
v_mul_f32 v15, v11, v7 // (op 3807) 
v_mul_f32 v16, v11, v8 // (op 3807) 
v_mul_f32 v17, v11, v9 // (op 3807) 

v_lshrrev_b32 v6, 16, v14 // (op 3809) 
v_lshrrev_b32 v7, 16, v15 // (op 3809) 
v_lshrrev_b32 v8, 16, v16 // (op 3809) 
v_lshrrev_b32 v9, 16, v17 // (op 3809) 


v_pack_b32_f16 v4, v6, v7 // (op 3791) 
v_pack_b32_f16 v5, v8, v9 // (op 3791) 

s_lshl_b32 s0, s4, 6 // (op 3791) 
s_mul_i32 s1, 3, s0 // (op 3791) 
s_add_u32 s1, s1, 192 // (op 3791) 

v_add_u32 v0, v13, s1 // (op 3791) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3791) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_135:


v_add_u32 v0, 192, v12 // (op 4179) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4179) Load value

v_accvgpr_read v4, a48 // (op 4166) 
v_accvgpr_read v5, a49 // (op 4166) 
v_accvgpr_read v6, a50 // (op 4166) 
v_accvgpr_read v7, a51 // (op 4166) 
v_mul_f32 v14, v11, v4 // (op 4166) 
v_mul_f32 v15, v11, v5 // (op 4166) 
v_mul_f32 v16, v11, v6 // (op 4166) 
v_mul_f32 v17, v11, v7 // (op 4166) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4178) 
v_lshrrev_b32 v7, 16, v8 // (op 4178) 
v_lshlrev_b32 v6, 16, v6 // (op 4178) 
v_lshlrev_b32 v7, 16, v7 // (op 4178) 
v_mul_f32 v18, v10, v6 // (op 4178) 
v_mul_f32 v19, v10, v7 // (op 4178) 
v_and_b32 v6, 65535, v9 // (op 4178) 
v_lshrrev_b32 v7, 16, v9 // (op 4178) 
v_lshlrev_b32 v6, 16, v6 // (op 4178) 
v_lshlrev_b32 v7, 16, v7 // (op 4178) 
v_mul_f32 v20, v10, v6 // (op 4178) 
v_mul_f32 v21, v10, v7 // (op 4178) 


v_add_f32 v6, v18, v14 // (op 4167) 
v_add_f32 v7, v19, v15 // (op 4167) 
v_add_f32 v8, v20, v16 // (op 4167) 
v_add_f32 v9, v21, v17 // (op 4167) 


v_lshrrev_b32 v14, 16, v6 // (op 4168) 
v_lshrrev_b32 v15, 16, v7 // (op 4168) 
v_lshrrev_b32 v16, 16, v8 // (op 4168) 
v_lshrrev_b32 v17, 16, v9 // (op 4168) 


v_pack_b32_f16 v8, v14, v15 // (op 4170) 
v_pack_b32_f16 v9, v16, v17 // (op 4170) 

v_add_u32 v0, 192, v13 // (op 4170) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4170) Store value





s_lshl_b32 s0, s2, 6 // (op 4184) 
s_add_u32 s0, s0, 192 // (op 4184) 
v_add_u32 v0, v12, s0 // (op 4184) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4184) Load value

v_accvgpr_read v4, a52 // (op 4196) 
v_accvgpr_read v5, a53 // (op 4196) 
v_accvgpr_read v6, a54 // (op 4196) 
v_accvgpr_read v7, a55 // (op 4196) 
v_mul_f32 v14, v11, v4 // (op 4196) 
v_mul_f32 v15, v11, v5 // (op 4196) 
v_mul_f32 v16, v11, v6 // (op 4196) 
v_mul_f32 v17, v11, v7 // (op 4196) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4181) 
v_lshrrev_b32 v7, 16, v8 // (op 4181) 
v_lshlrev_b32 v6, 16, v6 // (op 4181) 
v_lshlrev_b32 v7, 16, v7 // (op 4181) 
v_mul_f32 v18, v10, v6 // (op 4181) 
v_mul_f32 v19, v10, v7 // (op 4181) 
v_and_b32 v6, 65535, v9 // (op 4181) 
v_lshrrev_b32 v7, 16, v9 // (op 4181) 
v_lshlrev_b32 v6, 16, v6 // (op 4181) 
v_lshlrev_b32 v7, 16, v7 // (op 4181) 
v_mul_f32 v20, v10, v6 // (op 4181) 
v_mul_f32 v21, v10, v7 // (op 4181) 


v_add_f32 v6, v18, v14 // (op 4182) 
v_add_f32 v7, v19, v15 // (op 4182) 
v_add_f32 v8, v20, v16 // (op 4182) 
v_add_f32 v9, v21, v17 // (op 4182) 


v_lshrrev_b32 v14, 16, v6 // (op 4183) 
v_lshrrev_b32 v15, 16, v7 // (op 4183) 
v_lshrrev_b32 v16, 16, v8 // (op 4183) 
v_lshrrev_b32 v17, 16, v9 // (op 4183) 


v_pack_b32_f16 v8, v14, v15 // (op 4172) 
v_pack_b32_f16 v9, v16, v17 // (op 4172) 

s_lshl_b32 s0, s4, 6 // (op 4172) 
s_add_u32 s0, s0, 192 // (op 4172) 
v_add_u32 v0, v13, s0 // (op 4172) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4172) Store value





s_lshl_b32 s0, s2, 7 // (op 4189) 
s_add_u32 s0, s0, 192 // (op 4189) 
v_add_u32 v0, v12, s0 // (op 4189) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4189) Load value

v_accvgpr_read v4, a56 // (op 4198) 
v_accvgpr_read v5, a57 // (op 4198) 
v_accvgpr_read v6, a58 // (op 4198) 
v_accvgpr_read v7, a59 // (op 4198) 
v_mul_f32 v14, v11, v4 // (op 4198) 
v_mul_f32 v15, v11, v5 // (op 4198) 
v_mul_f32 v16, v11, v6 // (op 4198) 
v_mul_f32 v17, v11, v7 // (op 4198) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4186) 
v_lshrrev_b32 v7, 16, v8 // (op 4186) 
v_lshlrev_b32 v6, 16, v6 // (op 4186) 
v_lshlrev_b32 v7, 16, v7 // (op 4186) 
v_mul_f32 v18, v10, v6 // (op 4186) 
v_mul_f32 v19, v10, v7 // (op 4186) 
v_and_b32 v6, 65535, v9 // (op 4186) 
v_lshrrev_b32 v7, 16, v9 // (op 4186) 
v_lshlrev_b32 v6, 16, v6 // (op 4186) 
v_lshlrev_b32 v7, 16, v7 // (op 4186) 
v_mul_f32 v20, v10, v6 // (op 4186) 
v_mul_f32 v21, v10, v7 // (op 4186) 


v_add_f32 v6, v18, v14 // (op 4187) 
v_add_f32 v7, v19, v15 // (op 4187) 
v_add_f32 v8, v20, v16 // (op 4187) 
v_add_f32 v9, v21, v17 // (op 4187) 


v_lshrrev_b32 v14, 16, v6 // (op 4188) 
v_lshrrev_b32 v15, 16, v7 // (op 4188) 
v_lshrrev_b32 v16, 16, v8 // (op 4188) 
v_lshrrev_b32 v17, 16, v9 // (op 4188) 


v_pack_b32_f16 v8, v14, v15 // (op 4174) 
v_pack_b32_f16 v9, v16, v17 // (op 4174) 

s_lshl_b32 s0, s4, 7 // (op 4174) 
s_add_u32 s0, s0, 192 // (op 4174) 
v_add_u32 v0, v13, s0 // (op 4174) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4174) Store value





s_lshl_b32 s0, s2, 6 // (op 4194) 
s_mul_i32 s1, 3, s0 // (op 4194) 
s_add_u32 s1, s1, 192 // (op 4194) 

v_add_u32 v0, v12, s1 // (op 4194) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4194) Load value

v_accvgpr_read v4, a60 // (op 4200) 
v_accvgpr_read v5, a61 // (op 4200) 
v_accvgpr_read v6, a62 // (op 4200) 
v_accvgpr_read v7, a63 // (op 4200) 
v_mul_f32 v14, v11, v4 // (op 4200) 
v_mul_f32 v15, v11, v5 // (op 4200) 
v_mul_f32 v16, v11, v6 // (op 4200) 
v_mul_f32 v17, v11, v7 // (op 4200) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4191) 
v_lshrrev_b32 v7, 16, v8 // (op 4191) 
v_lshlrev_b32 v6, 16, v6 // (op 4191) 
v_lshlrev_b32 v7, 16, v7 // (op 4191) 
v_mul_f32 v18, v10, v6 // (op 4191) 
v_mul_f32 v19, v10, v7 // (op 4191) 
v_and_b32 v6, 65535, v9 // (op 4191) 
v_lshrrev_b32 v7, 16, v9 // (op 4191) 
v_lshlrev_b32 v6, 16, v6 // (op 4191) 
v_lshlrev_b32 v7, 16, v7 // (op 4191) 
v_mul_f32 v20, v10, v6 // (op 4191) 
v_mul_f32 v21, v10, v7 // (op 4191) 


v_add_f32 v6, v18, v14 // (op 4192) 
v_add_f32 v7, v19, v15 // (op 4192) 
v_add_f32 v8, v20, v16 // (op 4192) 
v_add_f32 v9, v21, v17 // (op 4192) 


v_lshrrev_b32 v14, 16, v6 // (op 4193) 
v_lshrrev_b32 v15, 16, v7 // (op 4193) 
v_lshrrev_b32 v16, 16, v8 // (op 4193) 
v_lshrrev_b32 v17, 16, v9 // (op 4193) 


v_pack_b32_f16 v8, v14, v15 // (op 4176) 
v_pack_b32_f16 v9, v16, v17 // (op 4176) 

s_lshl_b32 s0, s4, 6 // (op 4176) 
s_mul_i32 s1, 3, s0 // (op 4176) 
s_add_u32 s1, s1, 192 // (op 4176) 

v_add_u32 v0, v13, s1 // (op 4176) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4176) Store value





v_add_u32 v0, 128, v12 // (op 4092) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4092) Load value

v_accvgpr_read v4, a32 // (op 4079) 
v_accvgpr_read v5, a33 // (op 4079) 
v_accvgpr_read v6, a34 // (op 4079) 
v_accvgpr_read v7, a35 // (op 4079) 
v_mul_f32 v14, v11, v4 // (op 4079) 
v_mul_f32 v15, v11, v5 // (op 4079) 
v_mul_f32 v16, v11, v6 // (op 4079) 
v_mul_f32 v17, v11, v7 // (op 4079) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4091) 
v_lshrrev_b32 v7, 16, v8 // (op 4091) 
v_lshlrev_b32 v6, 16, v6 // (op 4091) 
v_lshlrev_b32 v7, 16, v7 // (op 4091) 
v_mul_f32 v18, v10, v6 // (op 4091) 
v_mul_f32 v19, v10, v7 // (op 4091) 
v_and_b32 v6, 65535, v9 // (op 4091) 
v_lshrrev_b32 v7, 16, v9 // (op 4091) 
v_lshlrev_b32 v6, 16, v6 // (op 4091) 
v_lshlrev_b32 v7, 16, v7 // (op 4091) 
v_mul_f32 v20, v10, v6 // (op 4091) 
v_mul_f32 v21, v10, v7 // (op 4091) 


v_add_f32 v6, v18, v14 // (op 4080) 
v_add_f32 v7, v19, v15 // (op 4080) 
v_add_f32 v8, v20, v16 // (op 4080) 
v_add_f32 v9, v21, v17 // (op 4080) 


v_lshrrev_b32 v14, 16, v6 // (op 4081) 
v_lshrrev_b32 v15, 16, v7 // (op 4081) 
v_lshrrev_b32 v16, 16, v8 // (op 4081) 
v_lshrrev_b32 v17, 16, v9 // (op 4081) 


v_pack_b32_f16 v8, v14, v15 // (op 4083) 
v_pack_b32_f16 v9, v16, v17 // (op 4083) 

v_add_u32 v0, 128, v13 // (op 4083) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4083) Store value





s_lshl_b32 s0, s2, 6 // (op 4097) 
s_add_u32 s0, s0, 128 // (op 4097) 
v_add_u32 v0, v12, s0 // (op 4097) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4097) Load value

v_accvgpr_read v4, a36 // (op 4109) 
v_accvgpr_read v5, a37 // (op 4109) 
v_accvgpr_read v6, a38 // (op 4109) 
v_accvgpr_read v7, a39 // (op 4109) 
v_mul_f32 v14, v11, v4 // (op 4109) 
v_mul_f32 v15, v11, v5 // (op 4109) 
v_mul_f32 v16, v11, v6 // (op 4109) 
v_mul_f32 v17, v11, v7 // (op 4109) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4094) 
v_lshrrev_b32 v7, 16, v8 // (op 4094) 
v_lshlrev_b32 v6, 16, v6 // (op 4094) 
v_lshlrev_b32 v7, 16, v7 // (op 4094) 
v_mul_f32 v18, v10, v6 // (op 4094) 
v_mul_f32 v19, v10, v7 // (op 4094) 
v_and_b32 v6, 65535, v9 // (op 4094) 
v_lshrrev_b32 v7, 16, v9 // (op 4094) 
v_lshlrev_b32 v6, 16, v6 // (op 4094) 
v_lshlrev_b32 v7, 16, v7 // (op 4094) 
v_mul_f32 v20, v10, v6 // (op 4094) 
v_mul_f32 v21, v10, v7 // (op 4094) 


v_add_f32 v6, v18, v14 // (op 4095) 
v_add_f32 v7, v19, v15 // (op 4095) 
v_add_f32 v8, v20, v16 // (op 4095) 
v_add_f32 v9, v21, v17 // (op 4095) 


v_lshrrev_b32 v14, 16, v6 // (op 4096) 
v_lshrrev_b32 v15, 16, v7 // (op 4096) 
v_lshrrev_b32 v16, 16, v8 // (op 4096) 
v_lshrrev_b32 v17, 16, v9 // (op 4096) 


v_pack_b32_f16 v8, v14, v15 // (op 4085) 
v_pack_b32_f16 v9, v16, v17 // (op 4085) 

s_lshl_b32 s0, s4, 6 // (op 4085) 
s_add_u32 s0, s0, 128 // (op 4085) 
v_add_u32 v0, v13, s0 // (op 4085) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4085) Store value





s_lshl_b32 s0, s2, 7 // (op 4102) 
s_add_u32 s0, s0, 128 // (op 4102) 
v_add_u32 v0, v12, s0 // (op 4102) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4102) Load value

v_accvgpr_read v4, a40 // (op 4111) 
v_accvgpr_read v5, a41 // (op 4111) 
v_accvgpr_read v6, a42 // (op 4111) 
v_accvgpr_read v7, a43 // (op 4111) 
v_mul_f32 v14, v11, v4 // (op 4111) 
v_mul_f32 v15, v11, v5 // (op 4111) 
v_mul_f32 v16, v11, v6 // (op 4111) 
v_mul_f32 v17, v11, v7 // (op 4111) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4099) 
v_lshrrev_b32 v7, 16, v8 // (op 4099) 
v_lshlrev_b32 v6, 16, v6 // (op 4099) 
v_lshlrev_b32 v7, 16, v7 // (op 4099) 
v_mul_f32 v18, v10, v6 // (op 4099) 
v_mul_f32 v19, v10, v7 // (op 4099) 
v_and_b32 v6, 65535, v9 // (op 4099) 
v_lshrrev_b32 v7, 16, v9 // (op 4099) 
v_lshlrev_b32 v6, 16, v6 // (op 4099) 
v_lshlrev_b32 v7, 16, v7 // (op 4099) 
v_mul_f32 v20, v10, v6 // (op 4099) 
v_mul_f32 v21, v10, v7 // (op 4099) 


v_add_f32 v6, v18, v14 // (op 4100) 
v_add_f32 v7, v19, v15 // (op 4100) 
v_add_f32 v8, v20, v16 // (op 4100) 
v_add_f32 v9, v21, v17 // (op 4100) 


v_lshrrev_b32 v14, 16, v6 // (op 4101) 
v_lshrrev_b32 v15, 16, v7 // (op 4101) 
v_lshrrev_b32 v16, 16, v8 // (op 4101) 
v_lshrrev_b32 v17, 16, v9 // (op 4101) 


v_pack_b32_f16 v8, v14, v15 // (op 4087) 
v_pack_b32_f16 v9, v16, v17 // (op 4087) 

s_lshl_b32 s0, s4, 7 // (op 4087) 
s_add_u32 s0, s0, 128 // (op 4087) 
v_add_u32 v0, v13, s0 // (op 4087) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4087) Store value





s_lshl_b32 s0, s2, 6 // (op 4107) 
s_mul_i32 s1, 3, s0 // (op 4107) 
s_add_u32 s1, s1, 128 // (op 4107) 

v_add_u32 v0, v12, s1 // (op 4107) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4107) Load value

v_accvgpr_read v4, a44 // (op 4113) 
v_accvgpr_read v5, a45 // (op 4113) 
v_accvgpr_read v6, a46 // (op 4113) 
v_accvgpr_read v7, a47 // (op 4113) 
v_mul_f32 v14, v11, v4 // (op 4113) 
v_mul_f32 v15, v11, v5 // (op 4113) 
v_mul_f32 v16, v11, v6 // (op 4113) 
v_mul_f32 v17, v11, v7 // (op 4113) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4104) 
v_lshrrev_b32 v7, 16, v8 // (op 4104) 
v_lshlrev_b32 v6, 16, v6 // (op 4104) 
v_lshlrev_b32 v7, 16, v7 // (op 4104) 
v_mul_f32 v18, v10, v6 // (op 4104) 
v_mul_f32 v19, v10, v7 // (op 4104) 
v_and_b32 v6, 65535, v9 // (op 4104) 
v_lshrrev_b32 v7, 16, v9 // (op 4104) 
v_lshlrev_b32 v6, 16, v6 // (op 4104) 
v_lshlrev_b32 v7, 16, v7 // (op 4104) 
v_mul_f32 v20, v10, v6 // (op 4104) 
v_mul_f32 v21, v10, v7 // (op 4104) 


v_add_f32 v6, v18, v14 // (op 4105) 
v_add_f32 v7, v19, v15 // (op 4105) 
v_add_f32 v8, v20, v16 // (op 4105) 
v_add_f32 v9, v21, v17 // (op 4105) 


v_lshrrev_b32 v14, 16, v6 // (op 4106) 
v_lshrrev_b32 v15, 16, v7 // (op 4106) 
v_lshrrev_b32 v16, 16, v8 // (op 4106) 
v_lshrrev_b32 v17, 16, v9 // (op 4106) 


v_pack_b32_f16 v8, v14, v15 // (op 4089) 
v_pack_b32_f16 v9, v16, v17 // (op 4089) 

s_lshl_b32 s0, s4, 6 // (op 4089) 
s_mul_i32 s1, 3, s0 // (op 4089) 
s_add_u32 s1, s1, 128 // (op 4089) 

v_add_u32 v0, v13, s1 // (op 4089) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4089) Store value





v_add_u32 v0, 64, v12 // (op 4005) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4005) Load value

v_accvgpr_read v4, a16 // (op 3992) 
v_accvgpr_read v5, a17 // (op 3992) 
v_accvgpr_read v6, a18 // (op 3992) 
v_accvgpr_read v7, a19 // (op 3992) 
v_mul_f32 v14, v11, v4 // (op 3992) 
v_mul_f32 v15, v11, v5 // (op 3992) 
v_mul_f32 v16, v11, v6 // (op 3992) 
v_mul_f32 v17, v11, v7 // (op 3992) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4004) 
v_lshrrev_b32 v7, 16, v8 // (op 4004) 
v_lshlrev_b32 v6, 16, v6 // (op 4004) 
v_lshlrev_b32 v7, 16, v7 // (op 4004) 
v_mul_f32 v18, v10, v6 // (op 4004) 
v_mul_f32 v19, v10, v7 // (op 4004) 
v_and_b32 v6, 65535, v9 // (op 4004) 
v_lshrrev_b32 v7, 16, v9 // (op 4004) 
v_lshlrev_b32 v6, 16, v6 // (op 4004) 
v_lshlrev_b32 v7, 16, v7 // (op 4004) 
v_mul_f32 v20, v10, v6 // (op 4004) 
v_mul_f32 v21, v10, v7 // (op 4004) 


v_add_f32 v6, v18, v14 // (op 3993) 
v_add_f32 v7, v19, v15 // (op 3993) 
v_add_f32 v8, v20, v16 // (op 3993) 
v_add_f32 v9, v21, v17 // (op 3993) 


v_lshrrev_b32 v14, 16, v6 // (op 3994) 
v_lshrrev_b32 v15, 16, v7 // (op 3994) 
v_lshrrev_b32 v16, 16, v8 // (op 3994) 
v_lshrrev_b32 v17, 16, v9 // (op 3994) 


v_pack_b32_f16 v8, v14, v15 // (op 3996) 
v_pack_b32_f16 v9, v16, v17 // (op 3996) 

v_add_u32 v0, 64, v13 // (op 3996) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3996) Store value





s_lshl_b32 s0, s2, 6 // (op 4010) 
s_add_u32 s0, s0, 64 // (op 4010) 
v_add_u32 v0, v12, s0 // (op 4010) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4010) Load value

v_accvgpr_read v4, a20 // (op 4022) 
v_accvgpr_read v5, a21 // (op 4022) 
v_accvgpr_read v6, a22 // (op 4022) 
v_accvgpr_read v7, a23 // (op 4022) 
v_mul_f32 v14, v11, v4 // (op 4022) 
v_mul_f32 v15, v11, v5 // (op 4022) 
v_mul_f32 v16, v11, v6 // (op 4022) 
v_mul_f32 v17, v11, v7 // (op 4022) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4007) 
v_lshrrev_b32 v7, 16, v8 // (op 4007) 
v_lshlrev_b32 v6, 16, v6 // (op 4007) 
v_lshlrev_b32 v7, 16, v7 // (op 4007) 
v_mul_f32 v18, v10, v6 // (op 4007) 
v_mul_f32 v19, v10, v7 // (op 4007) 
v_and_b32 v6, 65535, v9 // (op 4007) 
v_lshrrev_b32 v7, 16, v9 // (op 4007) 
v_lshlrev_b32 v6, 16, v6 // (op 4007) 
v_lshlrev_b32 v7, 16, v7 // (op 4007) 
v_mul_f32 v20, v10, v6 // (op 4007) 
v_mul_f32 v21, v10, v7 // (op 4007) 


v_add_f32 v6, v18, v14 // (op 4008) 
v_add_f32 v7, v19, v15 // (op 4008) 
v_add_f32 v8, v20, v16 // (op 4008) 
v_add_f32 v9, v21, v17 // (op 4008) 


v_lshrrev_b32 v14, 16, v6 // (op 4009) 
v_lshrrev_b32 v15, 16, v7 // (op 4009) 
v_lshrrev_b32 v16, 16, v8 // (op 4009) 
v_lshrrev_b32 v17, 16, v9 // (op 4009) 


v_pack_b32_f16 v8, v14, v15 // (op 3998) 
v_pack_b32_f16 v9, v16, v17 // (op 3998) 

s_lshl_b32 s0, s4, 6 // (op 3998) 
s_add_u32 s0, s0, 64 // (op 3998) 
v_add_u32 v0, v13, s0 // (op 3998) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3998) Store value





s_lshl_b32 s0, s2, 7 // (op 4015) 
s_add_u32 s0, s0, 64 // (op 4015) 
v_add_u32 v0, v12, s0 // (op 4015) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4015) Load value

v_accvgpr_read v4, a24 // (op 4024) 
v_accvgpr_read v5, a25 // (op 4024) 
v_accvgpr_read v6, a26 // (op 4024) 
v_accvgpr_read v7, a27 // (op 4024) 
v_mul_f32 v14, v11, v4 // (op 4024) 
v_mul_f32 v15, v11, v5 // (op 4024) 
v_mul_f32 v16, v11, v6 // (op 4024) 
v_mul_f32 v17, v11, v7 // (op 4024) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4012) 
v_lshrrev_b32 v7, 16, v8 // (op 4012) 
v_lshlrev_b32 v6, 16, v6 // (op 4012) 
v_lshlrev_b32 v7, 16, v7 // (op 4012) 
v_mul_f32 v18, v10, v6 // (op 4012) 
v_mul_f32 v19, v10, v7 // (op 4012) 
v_and_b32 v6, 65535, v9 // (op 4012) 
v_lshrrev_b32 v7, 16, v9 // (op 4012) 
v_lshlrev_b32 v6, 16, v6 // (op 4012) 
v_lshlrev_b32 v7, 16, v7 // (op 4012) 
v_mul_f32 v20, v10, v6 // (op 4012) 
v_mul_f32 v21, v10, v7 // (op 4012) 


v_add_f32 v6, v18, v14 // (op 4013) 
v_add_f32 v7, v19, v15 // (op 4013) 
v_add_f32 v8, v20, v16 // (op 4013) 
v_add_f32 v9, v21, v17 // (op 4013) 


v_lshrrev_b32 v14, 16, v6 // (op 4014) 
v_lshrrev_b32 v15, 16, v7 // (op 4014) 
v_lshrrev_b32 v16, 16, v8 // (op 4014) 
v_lshrrev_b32 v17, 16, v9 // (op 4014) 


v_pack_b32_f16 v8, v14, v15 // (op 4000) 
v_pack_b32_f16 v9, v16, v17 // (op 4000) 

s_lshl_b32 s0, s4, 7 // (op 4000) 
s_add_u32 s0, s0, 64 // (op 4000) 
v_add_u32 v0, v13, s0 // (op 4000) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4000) Store value





s_lshl_b32 s0, s2, 6 // (op 4020) 
s_mul_i32 s1, 3, s0 // (op 4020) 
s_add_u32 s1, s1, 64 // (op 4020) 

v_add_u32 v0, v12, s1 // (op 4020) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4020) Load value

v_accvgpr_read v4, a28 // (op 4026) 
v_accvgpr_read v5, a29 // (op 4026) 
v_accvgpr_read v6, a30 // (op 4026) 
v_accvgpr_read v7, a31 // (op 4026) 
v_mul_f32 v14, v11, v4 // (op 4026) 
v_mul_f32 v15, v11, v5 // (op 4026) 
v_mul_f32 v16, v11, v6 // (op 4026) 
v_mul_f32 v17, v11, v7 // (op 4026) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4017) 
v_lshrrev_b32 v7, 16, v8 // (op 4017) 
v_lshlrev_b32 v6, 16, v6 // (op 4017) 
v_lshlrev_b32 v7, 16, v7 // (op 4017) 
v_mul_f32 v18, v10, v6 // (op 4017) 
v_mul_f32 v19, v10, v7 // (op 4017) 
v_and_b32 v6, 65535, v9 // (op 4017) 
v_lshrrev_b32 v7, 16, v9 // (op 4017) 
v_lshlrev_b32 v6, 16, v6 // (op 4017) 
v_lshlrev_b32 v7, 16, v7 // (op 4017) 
v_mul_f32 v20, v10, v6 // (op 4017) 
v_mul_f32 v21, v10, v7 // (op 4017) 


v_add_f32 v6, v18, v14 // (op 4018) 
v_add_f32 v7, v19, v15 // (op 4018) 
v_add_f32 v8, v20, v16 // (op 4018) 
v_add_f32 v9, v21, v17 // (op 4018) 


v_lshrrev_b32 v14, 16, v6 // (op 4019) 
v_lshrrev_b32 v15, 16, v7 // (op 4019) 
v_lshrrev_b32 v16, 16, v8 // (op 4019) 
v_lshrrev_b32 v17, 16, v9 // (op 4019) 


v_pack_b32_f16 v8, v14, v15 // (op 4002) 
v_pack_b32_f16 v9, v16, v17 // (op 4002) 

s_lshl_b32 s0, s4, 6 // (op 4002) 
s_mul_i32 s1, 3, s0 // (op 4002) 
s_add_u32 s1, s1, 64 // (op 4002) 

v_add_u32 v0, v13, s1 // (op 4002) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4002) Store value





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





s_lshl_b32 s0, s2, 6 // (op 3913) 
v_add_u32 v0, v12, s0 // (op 3913) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3913) Load value

v_accvgpr_read v4, a4 // (op 3909) 
v_accvgpr_read v5, a5 // (op 3909) 
v_accvgpr_read v6, a6 // (op 3909) 
v_accvgpr_read v7, a7 // (op 3909) 
v_mul_f32 v14, v11, v4 // (op 3909) 
v_mul_f32 v15, v11, v5 // (op 3909) 
v_mul_f32 v16, v11, v6 // (op 3909) 
v_mul_f32 v17, v11, v7 // (op 3909) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3914) 
v_lshrrev_b32 v7, 16, v8 // (op 3914) 
v_lshlrev_b32 v6, 16, v6 // (op 3914) 
v_lshlrev_b32 v7, 16, v7 // (op 3914) 
v_mul_f32 v18, v10, v6 // (op 3914) 
v_mul_f32 v19, v10, v7 // (op 3914) 
v_and_b32 v6, 65535, v9 // (op 3914) 
v_lshrrev_b32 v7, 16, v9 // (op 3914) 
v_lshlrev_b32 v6, 16, v6 // (op 3914) 
v_lshlrev_b32 v7, 16, v7 // (op 3914) 
v_mul_f32 v20, v10, v6 // (op 3914) 
v_mul_f32 v21, v10, v7 // (op 3914) 


v_add_f32 v6, v18, v14 // (op 3910) 
v_add_f32 v7, v19, v15 // (op 3910) 
v_add_f32 v8, v20, v16 // (op 3910) 
v_add_f32 v9, v21, v17 // (op 3910) 


v_lshrrev_b32 v14, 16, v6 // (op 3911) 
v_lshrrev_b32 v15, 16, v7 // (op 3911) 
v_lshrrev_b32 v16, 16, v8 // (op 3911) 
v_lshrrev_b32 v17, 16, v9 // (op 3911) 


v_pack_b32_f16 v8, v14, v15 // (op 3912) 
v_pack_b32_f16 v9, v16, v17 // (op 3912) 

s_lshl_b32 s0, s4, 6 // (op 3912) 
v_add_u32 v0, v13, s0 // (op 3912) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3912) Store value





s_lshl_b32 s0, s2, 7 // (op 3927) 
v_add_u32 v0, v12, s0 // (op 3927) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3927) Load value

v_accvgpr_read v4, a8 // (op 3923) 
v_accvgpr_read v5, a9 // (op 3923) 
v_accvgpr_read v6, a10 // (op 3923) 
v_accvgpr_read v7, a11 // (op 3923) 
v_mul_f32 v14, v11, v4 // (op 3923) 
v_mul_f32 v15, v11, v5 // (op 3923) 
v_mul_f32 v16, v11, v6 // (op 3923) 
v_mul_f32 v17, v11, v7 // (op 3923) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3928) 
v_lshrrev_b32 v7, 16, v8 // (op 3928) 
v_lshlrev_b32 v6, 16, v6 // (op 3928) 
v_lshlrev_b32 v7, 16, v7 // (op 3928) 
v_mul_f32 v18, v10, v6 // (op 3928) 
v_mul_f32 v19, v10, v7 // (op 3928) 
v_and_b32 v6, 65535, v9 // (op 3928) 
v_lshrrev_b32 v7, 16, v9 // (op 3928) 
v_lshlrev_b32 v6, 16, v6 // (op 3928) 
v_lshlrev_b32 v7, 16, v7 // (op 3928) 
v_mul_f32 v20, v10, v6 // (op 3928) 
v_mul_f32 v21, v10, v7 // (op 3928) 


v_add_f32 v6, v18, v14 // (op 3924) 
v_add_f32 v7, v19, v15 // (op 3924) 
v_add_f32 v8, v20, v16 // (op 3924) 
v_add_f32 v9, v21, v17 // (op 3924) 


v_lshrrev_b32 v14, 16, v6 // (op 3925) 
v_lshrrev_b32 v15, 16, v7 // (op 3925) 
v_lshrrev_b32 v16, 16, v8 // (op 3925) 
v_lshrrev_b32 v17, 16, v9 // (op 3925) 


v_pack_b32_f16 v8, v14, v15 // (op 3926) 
v_pack_b32_f16 v9, v16, v17 // (op 3926) 

s_lshl_b32 s0, s4, 7 // (op 3926) 
v_add_u32 v0, v13, s0 // (op 3926) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3926) Store value





s_lshl_b32 s0, s2, 6 // (op 3941) 
s_mul_i32 s1, 3, s0 // (op 3941) 

v_add_u32 v0, v12, s1 // (op 3941) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3941) Load value

v_accvgpr_read v4, a12 // (op 3937) 
v_accvgpr_read v5, a13 // (op 3937) 
v_accvgpr_read v6, a14 // (op 3937) 
v_accvgpr_read v7, a15 // (op 3937) 
v_mul_f32 v14, v11, v4 // (op 3937) 
v_mul_f32 v15, v11, v5 // (op 3937) 
v_mul_f32 v16, v11, v6 // (op 3937) 
v_mul_f32 v17, v11, v7 // (op 3937) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3942) 
v_lshrrev_b32 v7, 16, v8 // (op 3942) 
v_lshlrev_b32 v6, 16, v6 // (op 3942) 
v_lshlrev_b32 v7, 16, v7 // (op 3942) 
v_mul_f32 v18, v10, v6 // (op 3942) 
v_mul_f32 v19, v10, v7 // (op 3942) 
v_and_b32 v6, 65535, v9 // (op 3942) 
v_lshrrev_b32 v7, 16, v9 // (op 3942) 
v_lshlrev_b32 v6, 16, v6 // (op 3942) 
v_lshlrev_b32 v7, 16, v7 // (op 3942) 
v_mul_f32 v20, v10, v6 // (op 3942) 
v_mul_f32 v21, v10, v7 // (op 3942) 


v_add_f32 v6, v18, v14 // (op 3938) 
v_add_f32 v7, v19, v15 // (op 3938) 
v_add_f32 v8, v20, v16 // (op 3938) 
v_add_f32 v9, v21, v17 // (op 3938) 


v_lshrrev_b32 v14, 16, v6 // (op 3939) 
v_lshrrev_b32 v15, 16, v7 // (op 3939) 
v_lshrrev_b32 v16, 16, v8 // (op 3939) 
v_lshrrev_b32 v17, 16, v9 // (op 3939) 


v_pack_b32_f16 v8, v14, v15 // (op 3940) 
v_pack_b32_f16 v9, v16, v17 // (op 3940) 

s_lshl_b32 s0, s4, 6 // (op 3940) 
s_mul_i32 s1, 3, s0 // (op 3940) 

v_add_u32 v0, v13, s1 // (op 3940) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3940) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.LRR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end:
.size RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2, .LRR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end-RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
  .amdhsa_next_free_vgpr 196
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 51200 // lds bytes
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
    DataTypeB: F8
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
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
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB } ]
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
    .symbol: RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 51200
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 132
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
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
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

