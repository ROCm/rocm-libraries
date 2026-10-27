// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2,@function
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 8882) 
v_and_b32 v3, 15, v1 // (op 8882) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 8882) 


v_and_b32 v0, 48, v1 // (op 8882) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 8882) 


v_lshrrev_b32 v0, 6, v1 // (op 8939) 
v_and_b32 v4, 1, v0 // (op 8939) 

v_and_b32 v0, 15, v1 // (op 8939) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 8939) 


v_and_b32 v0, 48, v1 // (op 8939) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 8939) 


v_lshrrev_b32 v0, 7, v1 // (op 8997) 
v_and_b32 v5, 15, v1 // (op 8997) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 8997) 


v_and_b32 v0, 63, v1 // (op 8997) 
v_lshrrev_b32 v5, 4, v0 // (op 8997) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 8997) 


v_lshrrev_b32 v5, 6, v1 // (op 9055) 
v_and_b32 v6, 1, v5 // (op 9055) 

v_and_b32 v5, 15, v1 // (op 9055) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9055) 


v_and_b32 v5, 63, v1 // (op 9055) 
v_lshrrev_b32 v6, 4, v5 // (op 9055) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9055) 


v_lshrrev_b32 v6, 7, v1 // (op 9122) 
v_and_b32 v7, 15, v1 // (op 9122) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9122) 


v_and_b32 v6, 48, v1 // (op 9122) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9122) 


v_lshrrev_b32 v6, 6, v1 // (op 9166) 
v_and_b32 v8, 1, v6 // (op 9166) 

v_and_b32 v6, 15, v1 // (op 9166) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 9166) 


v_and_b32 v6, 48, v1 // (op 9166) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 9166) 


v_lshrrev_b32 v6, 7, v1 // (op 9195) 
v_and_b32 v9, 15, v1 // (op 9195) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9195) 


v_and_b32 v6, 63, v1 // (op 9195) 
v_lshrrev_b32 v9, 4, v6 // (op 9195) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 9195) 


v_lshrrev_b32 v9, 6, v1 // (op 9224) 
v_and_b32 v10, 1, v9 // (op 9224) 

v_and_b32 v9, 15, v1 // (op 9224) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9224) 


v_and_b32 v9, 63, v1 // (op 9224) 
v_lshrrev_b32 v10, 4, v9 // (op 9224) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 9224) 


v_lshrrev_b32 v10, 7, v1 // (op 8821) 
v_and_b32 v11, 63, v1 // (op 8821) 
v_lshrrev_b32 v12, 4, v11 // (op 8821) 
v_lshlrev_b32 v13, 2, v12 // (op 8821) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8821) 


v_lshl_add_u32 v10, s16, 6, v12 // (op 8821) 

v_lshrrev_b32 v12, 2, v11 // (op 8821) 

v_and_b32 v11, 3, v12 // (op 8821) 

v_and_b32 v12, 3, v1 // (op 8821) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 8821) 


v_lshrrev_b32 v11, 6, v1 // (op 8821) 
v_and_b32 v12, 1, v11 // (op 8821) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 8821) 


v_lshl_add_u32 v12, s17, 8, v11 // (op 8821) 

v_mul_lo_u32 v11, v12, s2 // (op 8821) 

v_add_lshl_u32 v12, v10, v11, 1 // (op 8821) 


v_lshrrev_b32 v10, 7, v1 // (op 9076) 
v_and_b32 v11, 63, v1 // (op 9076) 
v_lshrrev_b32 v13, 4, v11 // (op 9076) 
s_mul_i32 s1, s2, s9 // (op 8823) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 8823) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 8823) least significant
s_mul_hi_u32 s19, s2, s8 // (op 8823) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8823) most significant: sum
s_add_u32 s1, s1, s19 // (op 8823) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 8823) 

v_lshlrev_b32 v14, 2, v13 // (op 9076) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 9076) 


v_lshl_add_u32 v10, s16, 6, v13 // (op 9076) 

v_lshrrev_b32 v13, 2, v11 // (op 9076) 

v_and_b32 v11, 3, v13 // (op 9076) 

v_and_b32 v13, 3, v1 // (op 9076) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 9076) 


v_lshrrev_b32 v11, 6, v1 // (op 9076) 
v_and_b32 v13, 1, v11 // (op 9076) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 9076) 


v_lshl_add_u32 v13, s17, 8, v11 // (op 9076) 

v_mul_lo_u32 v11, v13, s4 // (op 9076) 

v_add_lshl_u32 v13, v10, v11, 1 // (op 9076) 


s_mov_b32 s40, s6 // (op 8823) reg expression
s_mov_b32 s41, s7 // (op 8823) reg expression
s_mov_b32 s42, s18 // (op 8823) reg expression
s_mov_b32 s43, 131072 // (op 8823) call()

s_mul_i32 s1, s4, s9 // (op 9078) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 9078) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 9078) least significant
s_mul_hi_u32 s7, s4, s8 // (op 9078) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 9078) most significant: sum
s_add_u32 s1, s1, s7 // (op 9078) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 9078) 

s_mov_b32 s44, s10 // (op 9078) reg expression
s_mov_b32 s45, s11 // (op 9078) reg expression
s_mov_b32 s46, s6 // (op 9078) reg expression
s_mov_b32 s47, 131072 // (op 9078) call()


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
v_accvgpr_write a16, 0.00000 // (op 868) call()
v_accvgpr_write a17, 0.00000 // (op 868) call()
v_accvgpr_write a18, 0.00000 // (op 868) call()
v_accvgpr_write a19, 0.00000 // (op 868) call()
v_accvgpr_write a20, 0.00000 // (op 1044) call()
v_accvgpr_write a21, 0.00000 // (op 1044) call()
v_accvgpr_write a22, 0.00000 // (op 1044) call()
v_accvgpr_write a23, 0.00000 // (op 1044) call()
v_accvgpr_write a24, 0.00000 // (op 1220) call()
v_accvgpr_write a25, 0.00000 // (op 1220) call()
v_accvgpr_write a26, 0.00000 // (op 1220) call()
v_accvgpr_write a27, 0.00000 // (op 1220) call()
v_accvgpr_write a28, 0.00000 // (op 1396) call()
v_accvgpr_write a29, 0.00000 // (op 1396) call()
v_accvgpr_write a30, 0.00000 // (op 1396) call()
v_accvgpr_write a31, 0.00000 // (op 1396) call()
v_accvgpr_write a32, 0.00000 // (op 1748) call()
v_accvgpr_write a33, 0.00000 // (op 1748) call()
v_accvgpr_write a34, 0.00000 // (op 1748) call()
v_accvgpr_write a35, 0.00000 // (op 1748) call()
v_accvgpr_write a36, 0.00000 // (op 2365) call()
v_accvgpr_write a37, 0.00000 // (op 2365) call()
v_accvgpr_write a38, 0.00000 // (op 2365) call()
v_accvgpr_write a39, 0.00000 // (op 2365) call()
v_accvgpr_write a40, 0.00000 // (op 2366) call()
v_accvgpr_write a41, 0.00000 // (op 2366) call()
v_accvgpr_write a42, 0.00000 // (op 2366) call()
v_accvgpr_write a43, 0.00000 // (op 2366) call()
v_accvgpr_write a44, 0.00000 // (op 2367) call()
v_accvgpr_write a45, 0.00000 // (op 2367) call()
v_accvgpr_write a46, 0.00000 // (op 2367) call()
v_accvgpr_write a47, 0.00000 // (op 2367) call()
v_accvgpr_write a48, 0.00000 // (op 2368) call()
v_accvgpr_write a49, 0.00000 // (op 2368) call()
v_accvgpr_write a50, 0.00000 // (op 2368) call()
v_accvgpr_write a51, 0.00000 // (op 2368) call()
v_accvgpr_write a52, 0.00000 // (op 2369) call()
v_accvgpr_write a53, 0.00000 // (op 2369) call()
v_accvgpr_write a54, 0.00000 // (op 2369) call()
v_accvgpr_write a55, 0.00000 // (op 2369) call()
v_accvgpr_write a56, 0.00000 // (op 2370) call()
v_accvgpr_write a57, 0.00000 // (op 2370) call()
v_accvgpr_write a58, 0.00000 // (op 2370) call()
v_accvgpr_write a59, 0.00000 // (op 2370) call()
v_accvgpr_write a60, 0.00000 // (op 2371) call()
v_accvgpr_write a61, 0.00000 // (op 2371) call()
v_accvgpr_write a62, 0.00000 // (op 2371) call()
v_accvgpr_write a63, 0.00000 // (op 2371) call()
v_lshrrev_b32 v14, 3, v1 // (op 8583) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 8583) 

v_and_b32 v14, -8, v1 // (op 8583) 
v_sub_u32 v16, v1, v14 // (op 8583) 

v_lshlrev_b32 v14, 4, v16 // (op 8583) 

v_mov_b32 v17, 0 // (op 8583) convert
v_mov_b32 v16, v14 // (op 8583) convert

v_mul_lo_u32 v19, v15, s13 // (op 8583) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 8583) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 8583) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 8583) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 8583) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 8583) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 8585) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8585) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8585) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8585) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8585) most significant: sum
s_add_u32 s1, s1, s7 // (op 8585) most significant: sum


s_mov_b32 s48, s20 // (op 8585) reg expression
s_mov_b32 s49, s21 // (op 8585) reg expression
s_mov_b32 s50, s0 // (op 8585) reg expression
s_mov_b32 s51, 131072 // (op 8585) call()

v_lshrrev_b32 v14, 3, v1 // (op 8602) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 8602) 

v_and_b32 v14, -8, v1 // (op 8602) 
v_sub_u32 v16, v1, v14 // (op 8602) 

v_lshlrev_b32 v14, 4, v16 // (op 8602) 

v_mov_b32 v17, 0 // (op 8602) convert
v_mov_b32 v16, v14 // (op 8602) convert

v_mul_lo_u32 v21, v15, s13 // (op 8602) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 8602) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 8602) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8602) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 8602) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 8602) most significant half



s_mul_i32 s1, s12, s23 // (op 8604) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8604) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8604) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8604) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8604) most significant: sum
s_add_u32 s1, s1, s7 // (op 8604) most significant: sum


s_mov_b32 s48, s20 // (op 8604) reg expression
s_mov_b32 s49, s21 // (op 8604) reg expression
s_mov_b32 s50, s0 // (op 8604) reg expression
s_mov_b32 s51, 131072 // (op 8604) call()

v_and_b32 v14, -8, v1 // (op 8647) 
v_sub_u32 v15, v1, v14 // (op 8647) 

v_lshlrev_b32 v14, 4, v15 // (op 8647) 

v_mov_b32 v17, 0 // (op 8647) convert
v_mov_b32 v16, v14 // (op 8647) convert

v_lshrrev_b32 v14, 3, v1 // (op 8647) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 8647) 

v_mul_lo_u32 v23, v15, s25 // (op 8647) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 8647) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 8647) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 8647) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 8647) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 8647) most significant half



s_mul_i32 s1, s24, s9 // (op 8649) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8649) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8649) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8649) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8649) most significant: sum
s_add_u32 s1, s1, s7 // (op 8649) most significant: sum


s_mov_b32 s52, s26 // (op 8649) reg expression
s_mov_b32 s53, s27 // (op 8649) reg expression
s_mov_b32 s54, s0 // (op 8649) reg expression
s_mov_b32 s55, 131072 // (op 8649) call()

v_and_b32 v16, -8, v1 // (op 8666) 
v_sub_u32 v17, v1, v16 // (op 8666) 

v_lshlrev_b32 v16, 4, v17 // (op 8666) 

v_mov_b32 v23, 0 // (op 8666) convert
v_mov_b32 v22, v16 // (op 8666) convert

v_lshrrev_b32 v16, 3, v1 // (op 8666) 
v_lshl_add_u32 v17, s17, 8, v16 // (op 8666) 

v_mul_lo_u32 v25, v17, s25 // (op 8666) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 8666) most significant: high of low * low
v_mul_lo_u32 v24, v17, s24 // (op 8666) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 8666) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v24 // (op 8666) least significant half
v_addc_co_u32 v17, s[0:1], v23, v25, s[0:1] // (op 8666) most significant half



s_mul_i32 s1, s24, s9 // (op 8668) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8668) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8668) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8668) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8668) most significant: sum
s_add_u32 s1, s1, s7 // (op 8668) most significant: sum


s_mov_b32 s52, s26 // (op 8668) reg expression
s_mov_b32 s53, s27 // (op 8668) reg expression
s_mov_b32 s54, s0 // (op 8668) reg expression
s_mov_b32 s55, 131072 // (op 8668) call()

v_lshrrev_b32 v22, 2, v1 // (op 8711) 
v_lshl_add_u32 v23, s16, 6, v22 // (op 8711) 

v_and_b32 v22, -4, v1 // (op 8711) 
v_sub_u32 v24, v1, v22 // (op 8711) 

v_mul_lo_u32 v22, v23, s28 // (op 8711) 
v_add_u32 v22, v22, v24 // (op 8711) 


s_mul_i32 s1, s28, s23 // (op 8713) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 8713) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 8713) least significant
s_mul_hi_u32 s7, s28, s22 // (op 8713) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8713) most significant: sum
s_add_u32 s1, s1, s7 // (op 8713) most significant: sum


s_mov_b32 s56, s30 // (op 8713) reg expression
s_mov_b32 s57, s31 // (op 8713) reg expression
s_mov_b32 s58, s0 // (op 8713) reg expression
s_mov_b32 s59, 131072 // (op 8713) call()

v_lshrrev_b32 v23, 2, v1 // (op 8730) 
v_lshl_add_u32 v24, s16, 6, v23 // (op 8730) 

v_and_b32 v23, -4, v1 // (op 8730) 
v_sub_u32 v25, v1, v23 // (op 8730) 

v_mul_lo_u32 v23, v24, s28 // (op 8730) 
v_add_u32 v23, v23, v25 // (op 8730) 


s_mul_i32 s1, s28, s23 // (op 8732) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 8732) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 8732) least significant
s_mul_hi_u32 s7, s28, s22 // (op 8732) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8732) most significant: sum
s_add_u32 s1, s1, s7 // (op 8732) most significant: sum


s_mov_b32 s56, s30 // (op 8732) reg expression
s_mov_b32 s57, s31 // (op 8732) reg expression
s_mov_b32 s58, s0 // (op 8732) reg expression
s_mov_b32 s59, 131072 // (op 8732) call()

v_lshl_add_u32 v24, s17, 8, v1 // (op 8775) 
v_mul_lo_u32 v25, v24, s32 // (op 8775) 

s_mul_i32 s1, s32, s9 // (op 8777) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 8777) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 8777) least significant
s_mul_hi_u32 s7, s32, s8 // (op 8777) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8777) most significant: sum
s_add_u32 s1, s1, s7 // (op 8777) most significant: sum


s_mov_b32 s60, s34 // (op 8777) reg expression
s_mov_b32 s61, s35 // (op 8777) reg expression
s_mov_b32 s62, s0 // (op 8777) reg expression
s_mov_b32 s63, 131072 // (op 8777) call()

v_lshl_add_u32 v24, s17, 8, v1 // (op 8794) 
v_mul_lo_u32 v26, v24, s32 // (op 8794) 

s_mul_i32 s1, s32, s9 // (op 8796) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 8796) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 8796) least significant
s_mul_hi_u32 s7, s32, s8 // (op 8796) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8796) most significant: sum
s_add_u32 s1, s1, s7 // (op 8796) most significant: sum


s_mov_b32 s60, s34 // (op 8796) reg expression
s_mov_b32 s61, s35 // (op 8796) reg expression
s_mov_b32 s62, s0 // (op 8796) reg expression
s_mov_b32 s63, 131072 // (op 8796) call()

v_and_b32 v24, -8, v1 // (op 8922) 
v_sub_u32 v27, v1, v24 // (op 8922) 

v_lshlrev_b32 v24, 4, v27 // (op 8922) 

v_lshrrev_b32 v27, 3, v1 // (op 8922) 
v_lshl_add_u32 v28, v27, 7, v24 // (op 8922) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 8922) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8517) 
v_mov_b64 v[28:29], s[6:7] // (op 8517) convert


v_mov_b64 v[30:31], v[14:15] // (op 8517) 
s_mov_b32 m0, s0 // (op 8517) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8517) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8517) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8517) most significant half

s_add_u32 m0, m0, 4096 // (op 8517) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8517) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8517) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8517) most significant half

s_add_u32 m0, m0, 4096 // (op 8517) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8517) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8517) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8517) most significant half

s_add_u32 m0, m0, 4096 // (op 8517) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8517) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8517) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8517) most significant half

s_add_u32 m0, m0, 4096 // (op 8517) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8517) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8517) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8517) most significant half

s_add_u32 m0, m0, 4096 // (op 8517) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8517) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8517) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8517) most significant half

s_add_u32 m0, m0, 4096 // (op 8517) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8517) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8517) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8517) most significant half

s_add_u32 m0, m0, 4096 // (op 8517) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 8517) Load value direct to lds






buffer_load_dword v24, v25, s[60:63], 0 offen // (op 8203) Load value
v_and_b32 v27, -8, v1 // (op 8865) 
v_sub_u32 v28, v1, v27 // (op 8865) 

v_lshlrev_b32 v27, 4, v28 // (op 8865) 

v_lshrrev_b32 v28, 3, v1 // (op 8865) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 8865) 


s_nop 0
v_readfirstlane_b32 s0, v29 // (op 8865) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 8522) 
v_mov_b64 v[28:29], s[6:7] // (op 8522) convert


v_mov_b64 v[30:31], v[18:19] // (op 8522) 
s_add_u32 s1, 32768, s0 // (op 8522) 
s_mov_b32 m0, s1 // (op 8522) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8522) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8522) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8522) most significant half

s_add_u32 m0, m0, 4096 // (op 8522) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8522) Load value direct to lds







buffer_load_ubyte v27, v22, s[56:59], 0 offen offset: 0 // (op 8209) Load value
v_and_b32 v28, -8, v1 // (op 9149) 
v_sub_u32 v29, v1, v28 // (op 9149) 

v_lshlrev_b32 v28, 4, v29 // (op 9149) 

v_lshrrev_b32 v29, 3, v1 // (op 9149) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 9149) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 9149) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8527) 
v_mov_b64 v[28:29], s[6:7] // (op 8527) convert


v_mov_b64 v[30:31], v[14:15] // (op 8527) 
s_add_u32 s1, 40960, s0 // (op 8527) 
s_mov_b32 m0, s1 // (op 8527) reg expression
s_mov_b32 s6, 128 // (op 8527) call()
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8527) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8527) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8527) most significant half

s_add_u32 m0, m0, 4096 // (op 8527) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8527) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8527) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8527) most significant half

s_add_u32 m0, m0, 4096 // (op 8527) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8527) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8527) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8527) most significant half

s_add_u32 m0, m0, 4096 // (op 8527) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8527) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8527) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8527) most significant half

s_add_u32 m0, m0, 4096 // (op 8527) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8527) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8527) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8527) most significant half

s_add_u32 m0, m0, 4096 // (op 8527) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8527) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8527) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8527) most significant half

s_add_u32 m0, m0, 4096 // (op 8527) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8527) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8527) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8527) most significant half

s_add_u32 m0, m0, 4096 // (op 8527) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 8527) Load value direct to lds









buffer_load_dword v14, v25, s[60:63], 0 offen offset:4 // (op 8215) Load value

v_and_b32 v15, -8, v1 // (op 9105) 
v_sub_u32 v25, v1, v15 // (op 9105) 

v_lshlrev_b32 v15, 4, v25 // (op 9105) 

v_lshrrev_b32 v25, 3, v1 // (op 9105) 
v_lshl_add_u32 v28, v25, 7, v15 // (op 9105) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 9105) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 8529) 
v_mov_b64 v[28:29], s[6:7] // (op 8529) convert


v_mov_b64 v[30:31], v[18:19] // (op 8529) 
s_add_u32 s1, 73728, s0 // (op 8529) 
s_mov_b32 m0, s1 // (op 8529) reg expression
s_mov_b32 s6, 128 // (op 8529) call()
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8529) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v28 // (op 8529) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 8529) most significant half

s_add_u32 m0, m0, 4096 // (op 8529) 
buffer_load_dwordx4 v30, s[48:51], s6 offen offset: 0 lds // (op 8529) Load value direct to lds









buffer_load_ubyte v15, v22, s[56:59], 0 offen offset:4 // (op 8221) Load value

v_lshlrev_b32 v22, 2, v1 // (op 9038) 



v_add_u32 v25, 81920, v22 // (op 8224) 
s_waitcnt vmcnt(15)
ds_write_b32 v25, v24 // (op 8224) Store local data 


v_and_b32 v22, -4, v1 // (op 8980) 
v_sub_u32 v25, v1, v22 // (op 8980) 

v_and_b32 v22, 4294967292, v1 // (op 8980) 
v_add_u32 v18, v22, v25 // (op 8980) 





v_add_u32 v19, 82944, v18 // (op 8226) 
s_waitcnt vmcnt(12)
ds_write_b8 v19, v27 // (op 8226) Store local data 


s_barrier  // (op 9365) 



ds_read_b128 v[28:31], v4 // (op 8251) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 8251) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9400) 



v_add_u32 v22, 81920, v5 // (op 8301) 
ds_read_u8 v25, v22 // (op 8301) Load local data 




ds_read_b128 v[36:39], v4 offset:4096 // (op 8256) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 8256) Load local data 



v_add_u32 v22, 82048, v5 // (op 8306) 
ds_read_u8 v18, v22 // (op 8306) Load local data 




ds_read_b128 v[44:47], v4 offset:8192 // (op 8261) Load local data 
ds_read_b128 v[48:51], v4 offset:8256 // (op 8261) Load local data 



v_add_u32 v19, 82176, v5 // (op 8311) 
ds_read_u8 v22, v19 // (op 8311) Load local data 




ds_read_b128 v[52:55], v4 offset:12288 // (op 8266) Load local data 
ds_read_b128 v[56:59], v4 offset:12352 // (op 8266) Load local data 



v_add_u32 v19, 82304, v5 // (op 8316) 
ds_read_u8 v60, v19 // (op 8316) Load local data 




ds_read_b128 v[62:65], v4 offset:16384 // (op 8271) Load local data 
ds_read_b128 v[66:69], v4 offset:16448 // (op 8271) Load local data 



v_add_u32 v19, 82432, v5 // (op 8321) 
ds_read_u8 v61, v19 // (op 8321) Load local data 




ds_read_b128 v[70:73], v4 offset:20480 // (op 8276) Load local data 
ds_read_b128 v[74:77], v4 offset:20544 // (op 8276) Load local data 



v_add_u32 v19, 82560, v5 // (op 8326) 
ds_read_u8 v78, v19 // (op 8326) Load local data 




ds_read_b128 v[80:83], v4 offset:24576 // (op 8281) Load local data 
ds_read_b128 v[84:87], v4 offset:24640 // (op 8281) Load local data 



v_add_u32 v19, 82688, v5 // (op 8331) 
ds_read_u8 v79, v19 // (op 8331) Load local data 




ds_read_b128 v[88:91], v4 offset:28672 // (op 8286) Load local data 
ds_read_b128 v[92:95], v4 offset:28736 // (op 8286) Load local data 



v_add_u32 v19, 82816, v5 // (op 8336) 
ds_read_u8 v96, v19 // (op 8336) Load local data 

s_barrier  // (op 9349) 



ds_read_b128 v[98:101], v3 offset:32768 // (op 8241) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 8241) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9382) 



v_add_u32 v19, 82944, v0 // (op 8291) 
ds_read_u8 v97, v19 // (op 8291) Load local data 




ds_read_b128 v[106:109], v3 offset:36864 // (op 8246) Load local data 
ds_read_b128 v[110:113], v3 offset:36928 // (op 8246) Load local data 



v_add_u32 v19, 83072, v0 // (op 8296) 
ds_read_u8 v114, v19 // (op 8296) Load local data 

v_and_b32 v19, -8, v1 // (op 8837) 
v_sub_u32 v115, v1, v19 // (op 8837) 

v_lshlrev_b32 v19, 4, v115 // (op 8837) 

v_lshrrev_b32 v115, 3, v1 // (op 8837) 
v_lshl_add_u32 v116, v115, 7, v19 // (op 8837) 


s_nop 0
v_readfirstlane_b32 s0, v116 // (op 8837) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 8894) 
v_sub_u32 v115, v1, v19 // (op 8894) 

v_lshlrev_b32 v19, 4, v115 // (op 8894) 

v_lshrrev_b32 v115, 3, v1 // (op 8894) 
v_lshl_add_u32 v116, v115, 7, v19 // (op 8894) 


s_nop 0
v_readfirstlane_b32 s1, v116 // (op 8894) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -4, v1 // (op 8951) 
v_sub_u32 v115, v1, v19 // (op 8951) 

v_and_b32 v19, 4294967292, v1 // (op 8951) 
v_add_u32 v116, v19, v115 // (op 8951) 


v_lshlrev_b32 v19, 2, v1 // (op 9009) 
v_and_b32 v115, -8, v1 // (op 9090) 
v_sub_u32 v117, v1, v115 // (op 9090) 

v_lshlrev_b32 v115, 4, v117 // (op 9090) 

v_lshrrev_b32 v117, 3, v1 // (op 9090) 
v_lshl_add_u32 v118, v117, 7, v115 // (op 9090) 


s_nop 0
v_readfirstlane_b32 s6, v118 // (op 9090) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v115, -8, v1 // (op 9134) 
v_sub_u32 v117, v1, v115 // (op 9134) 

v_lshlrev_b32 v115, 4, v117 // (op 9134) 

v_lshrrev_b32 v117, 3, v1 // (op 9134) 
v_lshl_add_u32 v118, v117, 7, v115 // (op 9134) 


s_nop 0
v_readfirstlane_b32 s7, v118 // (op 9134) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v115, -4, v1 // (op 9178) 
v_sub_u32 v117, v1, v115 // (op 9178) 

v_and_b32 v115, 4294967292, v1 // (op 9178) 
v_add_u32 v118, v115, v117 // (op 9178) 


v_lshlrev_b32 v115, 2, v1 // (op 9207) 
s_mov_b32 s10, 0 // (op 1967) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1966) 
s_mov_b32 s11, s14 // (op 1966) convert

s_and_b32 s14, s11, 4294967294 // (op 1966) 

s_cmp_lt_u32 s10, s14 // (op 1966) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_1966 // (op 1966) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_1966 if false)
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_1966:
s_add_u32 s10, s10, 2 // (op 1968) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[28:35], a[0:3], v97, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 90) 
s_barrier  // (op 9368) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[36:43], a[4:7], v97, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 380) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 8489) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[44:51], a[8:11], v97, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 556) 
v_mov_b64 v[120:121], s[14:15] // (op 8489) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[52:59], a[12:15], v97, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 732) 
v_mov_b64 v[122:123], v[16:17] // (op 8489) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[62:69], a[16:19], v97, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 908) 
v_mov_b64 v[124:125], v[122:123] // (op 8489) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[70:77], a[20:23], v97, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1084) 
s_mov_b32 m0, s1 // (op 8489) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[80:87], a[24:27], v97, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1260) 
s_mov_b32 s11, 256 // (op 8489) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[88:95], a[28:31], v97, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1436) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8489) Load value direct to lds
v_add_co_u32 v124, s[14:15], v124, v120 // (op 8489) least significant half
v_addc_co_u32 v125, s[14:15], 0, v125, s[14:15] // (op 8489) most significant half

s_add_u32 m0, m0, 4096 // (op 8489) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8489) Load value direct to lds
v_add_co_u32 v124, s[14:15], v124, v120 // (op 8489) least significant half
v_addc_co_u32 v125, s[14:15], 0, v125, s[14:15] // (op 8489) most significant half

s_add_u32 m0, m0, 4096 // (op 8489) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8489) Load value direct to lds
v_add_co_u32 v124, s[14:15], v124, v120 // (op 8489) least significant half
v_addc_co_u32 v125, s[14:15], 0, v125, s[14:15] // (op 8489) most significant half

s_add_u32 m0, m0, 4096 // (op 8489) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[106:113], v[28:35], a[32:35], v114, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2327) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8489) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[106:113], v[36:43], a[36:39], v114, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2281) 
v_add_co_u32 v124, s[14:15], v124, v120 // (op 8489) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[106:113], v[44:51], a[40:43], v114, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2235) 
v_addc_co_u32 v125, s[14:15], 0, v125, s[14:15] // (op 8489) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[106:113], v[52:59], a[44:47], v114, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2189) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[106:113], v[62:69], a[48:51], v114, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2143) 
s_add_u32 m0, m0, 4096 // (op 8489) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[106:113], v[70:77], a[52:55], v114, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2097) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[80:87], a[56:59], v114, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2051) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8489) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[88:95], a[60:63], v114, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2005) 
v_add_co_u32 v124, s[14:15], v124, v120 // (op 8489) least significant half
v_addc_co_u32 v125, s[14:15], 0, v125, s[14:15] // (op 8489) most significant half

s_add_u32 m0, m0, 4096 // (op 8489) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8489) Load value direct to lds
v_add_co_u32 v124, s[14:15], v124, v120 // (op 8489) least significant half
v_addc_co_u32 v125, s[14:15], 0, v125, s[14:15] // (op 8489) most significant half

s_add_u32 m0, m0, 4096 // (op 8489) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8489) Load value direct to lds
v_add_co_u32 v124, s[14:15], v124, v120 // (op 8489) least significant half
v_addc_co_u32 v125, s[14:15], 0, v125, s[14:15] // (op 8489) most significant half

s_add_u32 m0, m0, 4096 // (op 8489) 
buffer_load_dwordx4 v124, s[52:55], s11 offen offset: 0 lds // (op 8489) Load value direct to lds





v_mov_b32 v117, v26 // (op 25) 

buffer_load_dword v24, v117, s[60:63], 0 offen offset:8 // (op 25) Load value
s_barrier  // (op 9352) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 8484) 
v_mov_b64 v[120:121], s[14:15] // (op 8484) convert


v_mov_b64 v[124:125], v[20:21] // (op 8484) 
v_mov_b64 v[126:127], v[124:125] // (op 8484) 
s_add_u32 s11, 32768, s0 // (op 8484) 
s_mov_b32 m0, s11 // (op 8484) reg expression
s_mov_b32 s14, 256 // (op 8484) call()
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8484) Load value direct to lds
v_add_co_u32 v126, s[18:19], v126, v120 // (op 8484) least significant half
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 8484) most significant half

s_add_u32 m0, m0, 4096 // (op 8484) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 8484) Load value direct to lds






v_mov_b32 v119, v23 // (op 17) 

buffer_load_ubyte v27, v119, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 9440) 



v_add_u32 v120, 83200, v115 // (op 259) 
s_waitcnt vmcnt(15)
ds_write_b32 v120, v14 // (op 259) Store local data 

s_barrier  // (op 9434) 



v_add_u32 v120, 84224, v118 // (op 258) 
s_waitcnt vmcnt(12)
ds_write_b8 v120, v15 // (op 258) Store local data 

s_barrier  // (op 9423) 



ds_read_b128 v[126:129], v8 offset:40960 // (op 251) Load local data 
ds_read_b128 v[130:133], v8 offset:41024 // (op 251) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9437) 



v_add_u32 v120, 83200, v9 // (op 255) 
ds_read_u8 v121, v120 // (op 255) Load local data 




ds_read_b128 v[134:137], v8 offset:45056 // (op 385) Load local data 
ds_read_b128 v[138:141], v8 offset:45120 // (op 385) Load local data 



v_add_u32 v120, 83328, v9 // (op 389) 
ds_read_u8 v142, v120 // (op 389) Load local data 




ds_read_b128 v[144:147], v8 offset:49152 // (op 561) Load local data 
ds_read_b128 v[148:151], v8 offset:49216 // (op 561) Load local data 



v_add_u32 v120, 83456, v9 // (op 565) 
ds_read_u8 v143, v120 // (op 565) Load local data 




ds_read_b128 v[152:155], v8 offset:53248 // (op 737) Load local data 
ds_read_b128 v[156:159], v8 offset:53312 // (op 737) Load local data 



v_add_u32 v120, 83584, v9 // (op 741) 
ds_read_u8 v160, v120 // (op 741) Load local data 




ds_read_b128 v[162:165], v8 offset:57344 // (op 913) Load local data 
ds_read_b128 v[166:169], v8 offset:57408 // (op 913) Load local data 



v_add_u32 v120, 83712, v9 // (op 917) 
ds_read_u8 v161, v120 // (op 917) Load local data 




ds_read_b128 v[170:173], v8 offset:61440 // (op 1089) Load local data 
ds_read_b128 v[174:177], v8 offset:61504 // (op 1089) Load local data 



v_add_u32 v120, 83840, v9 // (op 1093) 
ds_read_u8 v178, v120 // (op 1093) Load local data 




v_add_u32 v120, 65536, v8 // (op 1265) 
ds_read_b128 v[180:183], v120 // (op 1265) Load local data 

v_add_u32 v120, 65600, v8 // (op 1265) 
ds_read_b128 v[184:187], v120 // (op 1265) Load local data 




v_add_u32 v120, 83968, v9 // (op 1269) 
ds_read_u8 v179, v120 // (op 1269) Load local data 




v_add_u32 v120, 69632, v8 // (op 1441) 
ds_read_b128 v[188:191], v120 // (op 1441) Load local data 

v_add_u32 v120, 69696, v8 // (op 1441) 
ds_read_b128 v[192:195], v120 // (op 1441) Load local data 




v_add_u32 v120, 84096, v9 // (op 1445) 
ds_read_u8 v196, v120 // (op 1445) Load local data 

s_barrier  // (op 9417) 
s_barrier  // (op 9485) 



v_add_u32 v120, 73728, v7 // (op 249) 
ds_read_b128 v[98:101], v120 // (op 249) Load local data 

v_add_u32 v120, 73792, v7 // (op 249) 
ds_read_b128 v[102:105], v120 // (op 249) Load local data 

s_waitcnt lgkmcnt(15)
s_barrier  // (op 9430) 



v_add_u32 v120, 84224, v6 // (op 253) 
ds_read_u8 v97, v120 // (op 253) Load local data 




v_add_u32 v120, 77824, v7 // (op 2008) 
ds_read_b128 v[106:109], v120 // (op 2008) Load local data 

v_add_u32 v120, 77888, v7 // (op 2008) 
ds_read_b128 v[110:113], v120 // (op 2008) Load local data 




v_add_u32 v120, 84352, v6 // (op 2011) 
ds_read_u8 v114, v120 // (op 2011) Load local data 

s_barrier  // (op 9519) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[126:133], a[0:3], v97, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 256) 
s_barrier  // (op 9427) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[134:141], a[4:7], v97, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 381) 
s_lshl_b64 s[14:15], s[24:25], 5 // (op 8512) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[144:151], a[8:11], v97, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 557) 
v_mov_b64 v[198:199], s[14:15] // (op 8512) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[152:159], a[12:15], v97, v160 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 733) 
v_mov_b64 v[200:201], v[122:123] // (op 8512) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[162:169], a[16:19], v97, v161 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 909) 
s_add_u32 s11, 40960, s7 // (op 8512) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[170:177], a[20:23], v97, v178 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1085) 
s_mov_b32 m0, s11 // (op 8512) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[180:187], a[24:27], v97, v179 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1261) 
s_mov_b32 s14, 384 // (op 8512) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[188:195], a[28:31], v97, v196 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1437) 
buffer_load_dwordx4 v200, s[52:55], s14 offen offset: 0 lds // (op 8512) Load value direct to lds
v_add_co_u32 v200, s[18:19], v200, v198 // (op 8512) least significant half
v_addc_co_u32 v201, s[18:19], 0, v201, s[18:19] // (op 8512) most significant half

s_add_u32 m0, m0, 4096 // (op 8512) 
buffer_load_dwordx4 v200, s[52:55], s14 offen offset: 0 lds // (op 8512) Load value direct to lds
v_add_co_u32 v200, s[18:19], v200, v198 // (op 8512) least significant half
v_addc_co_u32 v201, s[18:19], 0, v201, s[18:19] // (op 8512) most significant half

s_add_u32 m0, m0, 4096 // (op 8512) 
buffer_load_dwordx4 v200, s[52:55], s14 offen offset: 0 lds // (op 8512) Load value direct to lds
v_add_co_u32 v200, s[18:19], v200, v198 // (op 8512) least significant half
v_addc_co_u32 v201, s[18:19], 0, v201, s[18:19] // (op 8512) most significant half

s_add_u32 m0, m0, 4096 // (op 8512) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[106:113], v[126:133], a[32:35], v114, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2328) 
buffer_load_dwordx4 v200, s[52:55], s14 offen offset: 0 lds // (op 8512) Load value direct to lds


v_add_co_u32 v200, s[18:19], v200, v198 // (op 8512) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[106:113], v[134:141], a[36:39], v114, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2282) 
v_addc_co_u32 v201, s[18:19], 0, v201, s[18:19] // (op 8512) most significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[106:113], v[144:151], a[40:43], v114, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2236) 
s_add_u32 m0, m0, 4096 // (op 8512) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[106:113], v[152:159], a[44:47], v114, v160 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2190) 
buffer_load_dwordx4 v200, s[52:55], s14 offen offset: 0 lds // (op 8512) Load value direct to lds


v_add_co_u32 v200, s[18:19], v200, v198 // (op 8512) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[106:113], v[162:169], a[48:51], v114, v161 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2144) 
v_addc_co_u32 v201, s[18:19], 0, v201, s[18:19] // (op 8512) most significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[106:113], v[170:177], a[52:55], v114, v178 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2098) 
s_add_u32 m0, m0, 4096 // (op 8512) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[180:187], a[56:59], v114, v179 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2052) 
buffer_load_dwordx4 v200, s[52:55], s14 offen offset: 0 lds // (op 8512) Load value direct to lds


v_add_co_u32 v200, s[18:19], v200, v198 // (op 8512) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[188:195], a[60:63], v114, v196 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 2006) 
v_addc_co_u32 v201, s[18:19], 0, v201, s[18:19] // (op 8512) most significant half



s_add_u32 m0, m0, 4096 // (op 8512) 
buffer_load_dwordx4 v200, s[52:55], s14 offen offset: 0 lds // (op 8512) Load value direct to lds
v_add_co_u32 v200, s[18:19], v200, v198 // (op 8512) least significant half
v_addc_co_u32 v201, s[18:19], 0, v201, s[18:19] // (op 8512) most significant half

s_add_u32 m0, m0, 4096 // (op 8512) 
buffer_load_dwordx4 v200, s[52:55], s14 offen offset: 0 lds // (op 8512) Load value direct to lds








v_add_co_u32 v16, vcc, 256, v16 // (op 8658) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 8658) most significant half
buffer_load_dword v14, v117, s[60:63], 0 offen offset:12 // (op 262) Load value

s_barrier  // (op 9420) 
v_add_u32 v26, 8, v26 // (op 8786) 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 8507) 
v_mov_b64 v[120:121], s[14:15] // (op 8507) convert


v_mov_b64 v[122:123], v[124:125] // (op 8507) 
s_add_u32 s11, 73728, s6 // (op 8507) 
s_mov_b32 m0, s11 // (op 8507) reg expression
s_mov_b32 s14, 384 // (op 8507) call()
buffer_load_dwordx4 v122, s[48:51], s14 offen offset: 0 lds // (op 8507) Load value direct to lds
v_add_co_u32 v122, s[18:19], v122, v120 // (op 8507) least significant half
v_addc_co_u32 v123, s[18:19], 0, v123, s[18:19] // (op 8507) most significant half

s_add_u32 m0, m0, 4096 // (op 8507) 
buffer_load_dwordx4 v122, s[48:51], s14 offen offset: 0 lds // (op 8507) Load value direct to lds








v_add_co_u32 v20, vcc, 256, v20 // (op 8594) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 8594) most significant half
buffer_load_ubyte v15, v119, s[56:59], 0 offen offset:12 // (op 261) Load value

s_barrier  // (op 9403) 
v_add_u32 v23, 8, v23 // (op 8722) 



v_add_u32 v117, 81920, v19 // (op 30) 
s_waitcnt vmcnt(15)
ds_write_b32 v117, v24 // (op 30) Store local data 

s_barrier  // (op 9385) 



v_add_u32 v117, 82944, v116 // (op 21) 
s_waitcnt vmcnt(12)
ds_write_b8 v117, v27 // (op 21) Store local data 

s_barrier  // (op 9374) 



ds_read_b128 v[28:31], v4 // (op 77) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9410) 



v_add_u32 v117, 81920, v5 // (op 87) 
ds_read_u8 v25, v117 // (op 87) Load local data 




ds_read_b128 v[36:39], v4 offset:4096 // (op 375) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 375) Load local data 



v_add_u32 v117, 82048, v5 // (op 379) 
ds_read_u8 v18, v117 // (op 379) Load local data 




ds_read_b128 v[44:47], v4 offset:8192 // (op 551) Load local data 
ds_read_b128 v[48:51], v4 offset:8256 // (op 551) Load local data 



v_add_u32 v117, 82176, v5 // (op 555) 
ds_read_u8 v22, v117 // (op 555) Load local data 




ds_read_b128 v[52:55], v4 offset:12288 // (op 727) Load local data 
ds_read_b128 v[56:59], v4 offset:12352 // (op 727) Load local data 



v_add_u32 v117, 82304, v5 // (op 731) 
ds_read_u8 v60, v117 // (op 731) Load local data 




ds_read_b128 v[62:65], v4 offset:16384 // (op 903) Load local data 
ds_read_b128 v[66:69], v4 offset:16448 // (op 903) Load local data 



v_add_u32 v117, 82432, v5 // (op 907) 
ds_read_u8 v61, v117 // (op 907) Load local data 




ds_read_b128 v[70:73], v4 offset:20480 // (op 1079) Load local data 
ds_read_b128 v[74:77], v4 offset:20544 // (op 1079) Load local data 



v_add_u32 v117, 82560, v5 // (op 1083) 
ds_read_u8 v78, v117 // (op 1083) Load local data 




ds_read_b128 v[80:83], v4 offset:24576 // (op 1255) Load local data 
ds_read_b128 v[84:87], v4 offset:24640 // (op 1255) Load local data 



v_add_u32 v117, 82688, v5 // (op 1259) 
ds_read_u8 v79, v117 // (op 1259) Load local data 




ds_read_b128 v[88:91], v4 offset:28672 // (op 1431) Load local data 
ds_read_b128 v[92:95], v4 offset:28736 // (op 1431) Load local data 



v_add_u32 v117, 82816, v5 // (op 1435) 
ds_read_u8 v96, v117 // (op 1435) Load local data 

s_barrier  // (op 9358) 



ds_read_b128 v[98:101], v3 offset:32768 // (op 72) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 72) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9392) 



v_add_u32 v117, 82944, v0 // (op 82) 
ds_read_u8 v97, v117 // (op 82) Load local data 




ds_read_b128 v[106:109], v3 offset:36864 // (op 2000) Load local data 
ds_read_b128 v[110:113], v3 offset:36928 // (op 2000) Load local data 



v_add_u32 v117, 83072, v0 // (op 2003) 
ds_read_u8 v114, v117 // (op 2003) Load local data 

s_ashr_i64 s[14:15], s[36:37], 7 // (op 1966) 
s_mov_b32 s11, s14 // (op 1966) convert

s_and_b32 s14, s11, 4294967294 // (op 1966) 

s_cmp_lt_u32 s10, s14 // (op 1966) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_1966 // (op 1966) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_1966 if true)
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_1966:

























s_barrier  // (op 9503) 
v_lshrrev_b32 v16, 3, v1 // (op 8542) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8542) 
v_lshl_add_u32 v17, s16, 6, v16 // (op 8542) 

s_lshr_b32 s6, s0, 1 // (op 8542) 

v_and_b32 v16, -8, v1 // (op 8542) 
v_sub_u32 v18, v1, v16 // (op 8542) 

v_lshlrev_b32 v16, 4, v18 // (op 8542) 

v_lshl_add_u32 v18, s6, 8, v16 // (op 8542) 


v_mov_b32 v69, 0 // (op 8542) convert
v_mov_b32 v68, v18 // (op 8542) convert

v_mul_lo_u32 v19, v17, s13 // (op 8542) most significant: low of low * high
v_mul_hi_u32 v16, v17, s12 // (op 8542) most significant: high of low * low
v_mul_lo_u32 v18, v17, s12 // (op 8542) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 8542) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v68 // (op 8542) least significant half
v_addc_co_u32 v19, s[0:1], v19, v69, s[0:1] // (op 8542) most significant half



s_mul_i32 s1, s12, s23 // (op 8544) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8544) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8544) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8544) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8544) most significant: sum
s_add_u32 s1, s1, s7 // (op 8544) most significant: sum


s_mov_b32 s48, s20 // (op 8544) reg expression
s_mov_b32 s49, s21 // (op 8544) reg expression
s_mov_b32 s50, s0 // (op 8544) reg expression
s_mov_b32 s51, 131072 // (op 8544) call()

v_and_b32 v16, -8, v1 // (op 8624) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8624) 
v_sub_u32 v17, v1, v16 // (op 8624) 

s_lshr_b32 s6, s0, 1 // (op 8624) 

v_lshlrev_b32 v16, 4, v17 // (op 8624) 

v_lshl_add_u32 v17, s6, 8, v16 // (op 8624) 


v_mov_b32 v21, 0 // (op 8624) convert
v_mov_b32 v20, v17 // (op 8624) convert

v_lshrrev_b32 v16, 3, v1 // (op 8624) 
v_lshl_add_u32 v17, s17, 8, v16 // (op 8624) 

v_mul_lo_u32 v23, v17, s25 // (op 8624) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 8624) most significant: high of low * low
v_mul_lo_u32 v22, v17, s24 // (op 8624) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 8624) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 8624) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 8624) most significant half



s_mul_i32 s1, s24, s9 // (op 8626) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8626) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8626) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8626) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8626) most significant: sum
s_add_u32 s1, s1, s7 // (op 8626) most significant: sum


s_mov_b32 s52, s26 // (op 8626) reg expression
s_mov_b32 s53, s27 // (op 8626) reg expression
s_mov_b32 s54, s0 // (op 8626) reg expression
s_mov_b32 s55, 131072 // (op 8626) call()

v_lshrrev_b32 v20, 2, v1 // (op 8688) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 8688) 
v_lshl_add_u32 v21, s16, 6, v20 // (op 8688) 

s_lshr_b32 s6, s0, 1 // (op 8688) 

v_and_b32 v20, -4, v1 // (op 8688) 
v_sub_u32 v22, v1, v20 // (op 8688) 

v_lshl_add_u32 v20, s6, 3, v22 // (op 8688) 


v_mul_lo_u32 v22, v21, s28 // (op 8688) 
v_add_u32 v22, v22, v20 // (op 8688) 


s_mul_i32 s1, s28, s23 // (op 8690) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 8690) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 8690) least significant
s_mul_hi_u32 s7, s28, s22 // (op 8690) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8690) most significant: sum
s_add_u32 s1, s1, s7 // (op 8690) most significant: sum


s_mov_b32 s56, s30 // (op 8690) reg expression
s_mov_b32 s57, s31 // (op 8690) reg expression
s_mov_b32 s58, s0 // (op 8690) reg expression
s_mov_b32 s59, 131072 // (op 8690) call()

v_lshl_add_u32 v20, s17, 8, v1 // (op 8752) 
v_mul_lo_u32 v21, v20, s32 // (op 8752) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 8752) 
s_lshr_b32 s6, s0, 1 // (op 8752) 

v_lshl_add_u32 v20, s6, 3, v21 // (op 8752) 


s_mul_i32 s1, s32, s9 // (op 8754) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 8754) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 8754) least significant
s_mul_hi_u32 s7, s32, s8 // (op 8754) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8754) most significant: sum
s_add_u32 s1, s1, s7 // (op 8754) most significant: sum


s_mov_b32 s60, s34 // (op 8754) reg expression
s_mov_b32 s61, s35 // (op 8754) reg expression
s_mov_b32 s62, s0 // (op 8754) reg expression
s_mov_b32 s63, 131072 // (op 8754) call()

v_and_b32 v21, -8, v1 // (op 8850) 
v_sub_u32 v23, v1, v21 // (op 8850) 

v_lshlrev_b32 v21, 4, v23 // (op 8850) 

v_lshrrev_b32 v23, 3, v1 // (op 8850) 
v_lshl_add_u32 v24, v23, 7, v21 // (op 8850) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 8850) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 8907) 
v_sub_u32 v23, v1, v21 // (op 8907) 

v_lshlrev_b32 v21, 4, v23 // (op 8907) 

v_lshrrev_b32 v23, 3, v1 // (op 8907) 
v_lshl_add_u32 v24, v23, 7, v21 // (op 8907) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 8907) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -4, v1 // (op 8964) 
v_sub_u32 v23, v1, v21 // (op 8964) 

v_and_b32 v21, 4294967292, v1 // (op 8964) 
v_add_u32 v24, v21, v23 // (op 8964) 


v_lshlrev_b32 v21, 2, v1 // (op 9022) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1972) 
s_and_b32 s10, s6, 4294967294 // (op 1972) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1971) 
s_mov_b32 s11, s6 // (op 1971) convert

s_cmp_lt_u32 s10, s11 // (op 1971) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_1971 // (op 1971) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_1971 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_1971:
s_barrier  // (op 9355) 
s_add_u32 s10, s10, 1 // (op 1973) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 8494) 
v_mov_b64 v[14:15], s[6:7] // (op 8494) convert


v_mov_b64 v[26:27], v[18:19] // (op 8494) 
v_mov_b64 v[28:29], v[26:27] // (op 8494) 
s_add_u32 s11, 32768, s0 // (op 8494) 
s_mov_b32 m0, s11 // (op 8494) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8494) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8494) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8494) most significant half

s_add_u32 m0, m0, 4096 // (op 8494) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8494) Load value direct to lds




v_add_co_u32 v18, vcc, 128, v18 // (op 8532) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 8532) most significant half
s_barrier  // (op 9371) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 8500) 
v_mov_b64 v[14:15], s[6:7] // (op 8500) convert


v_mov_b64 v[26:27], v[16:17] // (op 8500) 
v_mov_b64 v[28:29], v[26:27] // (op 8500) 
s_mov_b32 m0, s1 // (op 8500) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8500) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8500) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8500) most significant half

s_add_u32 m0, m0, 4096 // (op 8500) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8500) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8500) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8500) most significant half

s_add_u32 m0, m0, 4096 // (op 8500) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8500) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8500) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8500) most significant half

s_add_u32 m0, m0, 4096 // (op 8500) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8500) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8500) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8500) most significant half

s_add_u32 m0, m0, 4096 // (op 8500) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8500) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8500) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8500) most significant half

s_add_u32 m0, m0, 4096 // (op 8500) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8500) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8500) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8500) most significant half

s_add_u32 m0, m0, 4096 // (op 8500) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8500) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v14 // (op 8500) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8500) most significant half

s_add_u32 m0, m0, 4096 // (op 8500) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 8500) Load value direct to lds



v_add_co_u32 v16, vcc, 128, v16 // (op 8616) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 8616) most significant half


v_mov_b32 v23, v22 // (op 195) 

buffer_load_ubyte v14, v23, s[56:59], 0 offen offset: 0 // (op 195) Load value

v_add_u32 v22, 4, v22 // (op 8680) 


s_barrier  // (op 9388) 
v_mov_b32 v15, v20 // (op 196) 




v_add_u32 v23, 82944, v24 // (op 192) 
buffer_load_dword v25, v15, s[60:63], 0 offen // (op 196) Load value

v_add_u32 v20, 4, v20 // (op 8744) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v14 // (op 192) Store local data 


s_barrier  // (op 9361) 



s_barrier  // (op 9406) 
ds_read_b128 v[26:29], v3 offset:32768 // (op 183) Load local data 



ds_read_b128 v[30:33], v3 offset:32832 // (op 183) Load local data 
v_add_u32 v23, 81920, v21 // (op 193) 
s_barrier  // (op 9378) 
s_waitcnt vmcnt(0)
ds_write_b32 v23, v25 // (op 193) Store local data 





ds_read_b128 v[34:37], v4 // (op 185) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 185) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 9396) 



v_add_u32 v23, 82944, v0 // (op 187) 
ds_read_u8 v25, v23 // (op 187) Load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9413) 



v_add_u32 v23, 81920, v5 // (op 189) 
ds_read_u8 v14, v23 // (op 189) Load local data 




ds_read_b128 v[42:45], v4 offset:4096 // (op 354) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v25, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 190) 
ds_read_b128 v[46:49], v4 offset:4160 // (op 354) Load local data 



v_add_u32 v15, 82048, v5 // (op 358) 
ds_read_u8 v23, v15 // (op 358) Load local data 




ds_read_b128 v[50:53], v4 offset:8192 // (op 530) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[42:49], a[4:7], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 359) 
ds_read_b128 v[54:57], v4 offset:8256 // (op 530) Load local data 



v_add_u32 v15, 82176, v5 // (op 534) 
ds_read_u8 v58, v15 // (op 534) Load local data 




ds_read_b128 v[60:63], v4 offset:12288 // (op 706) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:33], v[50:57], a[8:11], v25, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 535) 
ds_read_b128 v[64:67], v4 offset:12352 // (op 706) Load local data 



v_add_u32 v15, 82304, v5 // (op 710) 
ds_read_u8 v59, v15 // (op 710) Load local data 




ds_read_b128 v[68:71], v4 offset:16384 // (op 882) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:33], v[60:67], a[12:15], v25, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 711) 
ds_read_b128 v[72:75], v4 offset:16448 // (op 882) Load local data 



v_add_u32 v15, 82432, v5 // (op 886) 
ds_read_u8 v76, v15 // (op 886) Load local data 




ds_read_b128 v[78:81], v4 offset:20480 // (op 1058) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:33], v[68:75], a[16:19], v25, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 887) 
ds_read_b128 v[82:85], v4 offset:20544 // (op 1058) Load local data 



v_add_u32 v15, 82560, v5 // (op 1062) 
ds_read_u8 v77, v15 // (op 1062) Load local data 




ds_read_b128 v[86:89], v4 offset:24576 // (op 1234) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:33], v[78:85], a[20:23], v25, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1063) 
ds_read_b128 v[90:93], v4 offset:24640 // (op 1234) Load local data 



v_add_u32 v15, 82688, v5 // (op 1238) 
ds_read_u8 v94, v15 // (op 1238) Load local data 




ds_read_b128 v[96:99], v4 offset:28672 // (op 1410) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:33], v[86:93], a[24:27], v25, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1239) 
ds_read_b128 v[100:103], v4 offset:28736 // (op 1410) Load local data 



v_add_u32 v15, 82816, v5 // (op 1414) 
ds_read_u8 v95, v15 // (op 1414) Load local data 




ds_read_b128 v[104:107], v3 offset:36864 // (op 1760) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:33], v[96:103], a[28:31], v25, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1415) 
ds_read_b128 v[108:111], v3 offset:36928 // (op 1760) Load local data 





v_add_u32 v15, 83072, v0 // (op 1763) 
ds_read_u8 v25, v15 // (op 1763) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:111], v[34:41], a[32:35], v25, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1765) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:111], v[42:49], a[36:39], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1796) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:111], v[50:57], a[40:43], v25, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1827) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:111], v[60:67], a[44:47], v25, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1858) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:111], v[68:75], a[48:51], v25, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1889) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:111], v[78:85], a[52:55], v25, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1920) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:111], v[86:93], a[56:59], v25, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1951) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:111], v[96:103], a[60:63], v25, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1982) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1971) 
s_mov_b32 s11, s6 // (op 1971) convert

s_cmp_lt_u32 s10, s11 // (op 1971) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_1971 // (op 1971) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_1971 if true)
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_1971:























s_barrier  // (op 9507) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_135
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



v_accvgpr_read v6, a8 // (op 3591) 
v_accvgpr_read v7, a9 // (op 3591) 
v_accvgpr_read v8, a10 // (op 3591) 
v_accvgpr_read v9, a11 // (op 3591) 
v_mul_f32 v14, v11, v6 // (op 3591) 
v_mul_f32 v15, v11, v7 // (op 3591) 
v_mul_f32 v16, v11, v8 // (op 3591) 
v_mul_f32 v17, v11, v9 // (op 3591) 

v_lshrrev_b32 v6, 16, v14 // (op 3593) 
v_lshrrev_b32 v7, 16, v15 // (op 3593) 
v_lshrrev_b32 v8, 16, v16 // (op 3593) 
v_lshrrev_b32 v9, 16, v17 // (op 3593) 


v_pack_b32_f16 v4, v6, v7 // (op 3594) 
v_pack_b32_f16 v5, v8, v9 // (op 3594) 

s_lshl_b32 s0, s4, 7 // (op 3594) 
v_add_u32 v0, v13, s0 // (op 3594) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3594) Store value



v_accvgpr_read v6, a12 // (op 3605) 
v_accvgpr_read v7, a13 // (op 3605) 
v_accvgpr_read v8, a14 // (op 3605) 
v_accvgpr_read v9, a15 // (op 3605) 
v_mul_f32 v14, v11, v6 // (op 3605) 
v_mul_f32 v15, v11, v7 // (op 3605) 
v_mul_f32 v16, v11, v8 // (op 3605) 
v_mul_f32 v17, v11, v9 // (op 3605) 

v_lshrrev_b32 v6, 16, v14 // (op 3607) 
v_lshrrev_b32 v7, 16, v15 // (op 3607) 
v_lshrrev_b32 v8, 16, v16 // (op 3607) 
v_lshrrev_b32 v9, 16, v17 // (op 3607) 


v_pack_b32_f16 v4, v6, v7 // (op 3608) 
v_pack_b32_f16 v5, v8, v9 // (op 3608) 

s_lshl_b32 s0, s4, 6 // (op 3608) 
s_mul_i32 s1, 3, s0 // (op 3608) 

v_add_u32 v0, v13, s1 // (op 3608) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3608) Store value



v_accvgpr_read v6, a16 // (op 3619) 
v_accvgpr_read v7, a17 // (op 3619) 
v_accvgpr_read v8, a18 // (op 3619) 
v_accvgpr_read v9, a19 // (op 3619) 
v_mul_f32 v14, v11, v6 // (op 3619) 
v_mul_f32 v15, v11, v7 // (op 3619) 
v_mul_f32 v16, v11, v8 // (op 3619) 
v_mul_f32 v17, v11, v9 // (op 3619) 

v_lshrrev_b32 v6, 16, v14 // (op 3621) 
v_lshrrev_b32 v7, 16, v15 // (op 3621) 
v_lshrrev_b32 v8, 16, v16 // (op 3621) 
v_lshrrev_b32 v9, 16, v17 // (op 3621) 


v_pack_b32_f16 v4, v6, v7 // (op 3622) 
v_pack_b32_f16 v5, v8, v9 // (op 3622) 

s_lshl_b32 s0, s4, 8 // (op 3622) 
v_add_u32 v0, v13, s0 // (op 3622) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3622) Store value



v_accvgpr_read v6, a20 // (op 3633) 
v_accvgpr_read v7, a21 // (op 3633) 
v_accvgpr_read v8, a22 // (op 3633) 
v_accvgpr_read v9, a23 // (op 3633) 
v_mul_f32 v14, v11, v6 // (op 3633) 
v_mul_f32 v15, v11, v7 // (op 3633) 
v_mul_f32 v16, v11, v8 // (op 3633) 
v_mul_f32 v17, v11, v9 // (op 3633) 

v_lshrrev_b32 v6, 16, v14 // (op 3635) 
v_lshrrev_b32 v7, 16, v15 // (op 3635) 
v_lshrrev_b32 v8, 16, v16 // (op 3635) 
v_lshrrev_b32 v9, 16, v17 // (op 3635) 


v_pack_b32_f16 v4, v6, v7 // (op 3636) 
v_pack_b32_f16 v5, v8, v9 // (op 3636) 

s_lshl_b32 s0, s4, 6 // (op 3636) 
s_mul_i32 s1, 5, s0 // (op 3636) 

v_add_u32 v0, v13, s1 // (op 3636) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3636) Store value



v_accvgpr_read v6, a24 // (op 3647) 
v_accvgpr_read v7, a25 // (op 3647) 
v_accvgpr_read v8, a26 // (op 3647) 
v_accvgpr_read v9, a27 // (op 3647) 
v_mul_f32 v14, v11, v6 // (op 3647) 
v_mul_f32 v15, v11, v7 // (op 3647) 
v_mul_f32 v16, v11, v8 // (op 3647) 
v_mul_f32 v17, v11, v9 // (op 3647) 

v_lshrrev_b32 v6, 16, v14 // (op 3649) 
v_lshrrev_b32 v7, 16, v15 // (op 3649) 
v_lshrrev_b32 v8, 16, v16 // (op 3649) 
v_lshrrev_b32 v9, 16, v17 // (op 3649) 


v_pack_b32_f16 v4, v6, v7 // (op 3650) 
v_pack_b32_f16 v5, v8, v9 // (op 3650) 

s_lshl_b32 s0, s4, 6 // (op 3650) 
s_mul_i32 s1, 6, s0 // (op 3650) 

v_add_u32 v0, v13, s1 // (op 3650) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3650) Store value



v_accvgpr_read v6, a28 // (op 3661) 
v_accvgpr_read v7, a29 // (op 3661) 
v_accvgpr_read v8, a30 // (op 3661) 
v_accvgpr_read v9, a31 // (op 3661) 
v_mul_f32 v14, v11, v6 // (op 3661) 
v_mul_f32 v15, v11, v7 // (op 3661) 
v_mul_f32 v16, v11, v8 // (op 3661) 
v_mul_f32 v17, v11, v9 // (op 3661) 

v_lshrrev_b32 v6, 16, v14 // (op 3663) 
v_lshrrev_b32 v7, 16, v15 // (op 3663) 
v_lshrrev_b32 v8, 16, v16 // (op 3663) 
v_lshrrev_b32 v9, 16, v17 // (op 3663) 


v_pack_b32_f16 v4, v6, v7 // (op 3664) 
v_pack_b32_f16 v5, v8, v9 // (op 3664) 

s_lshl_b32 s0, s4, 6 // (op 3664) 
s_mul_i32 s1, 7, s0 // (op 3664) 

v_add_u32 v0, v13, s1 // (op 3664) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3664) Store value



v_accvgpr_read v6, a32 // (op 3717) 
v_accvgpr_read v7, a33 // (op 3717) 
v_accvgpr_read v8, a34 // (op 3717) 
v_accvgpr_read v9, a35 // (op 3717) 
v_mul_f32 v14, v11, v6 // (op 3717) 
v_mul_f32 v15, v11, v7 // (op 3717) 
v_mul_f32 v16, v11, v8 // (op 3717) 
v_mul_f32 v17, v11, v9 // (op 3717) 

v_lshrrev_b32 v6, 16, v14 // (op 3719) 
v_lshrrev_b32 v7, 16, v15 // (op 3719) 
v_lshrrev_b32 v8, 16, v16 // (op 3719) 
v_lshrrev_b32 v9, 16, v17 // (op 3719) 


v_pack_b32_f16 v4, v6, v7 // (op 3721) 
v_pack_b32_f16 v5, v8, v9 // (op 3721) 

v_add_u32 v0, 64, v13 // (op 3721) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3721) Store value



v_accvgpr_read v6, a36 // (op 3739) 
v_accvgpr_read v7, a37 // (op 3739) 
v_accvgpr_read v8, a38 // (op 3739) 
v_accvgpr_read v9, a39 // (op 3739) 
v_mul_f32 v14, v11, v6 // (op 3739) 
v_mul_f32 v15, v11, v7 // (op 3739) 
v_mul_f32 v16, v11, v8 // (op 3739) 
v_mul_f32 v17, v11, v9 // (op 3739) 

v_lshrrev_b32 v6, 16, v14 // (op 3741) 
v_lshrrev_b32 v7, 16, v15 // (op 3741) 
v_lshrrev_b32 v8, 16, v16 // (op 3741) 
v_lshrrev_b32 v9, 16, v17 // (op 3741) 


v_pack_b32_f16 v4, v6, v7 // (op 3723) 
v_pack_b32_f16 v5, v8, v9 // (op 3723) 

s_lshl_b32 s0, s4, 6 // (op 3723) 
s_add_u32 s0, s0, 64 // (op 3723) 
v_add_u32 v0, v13, s0 // (op 3723) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3723) Store value



v_accvgpr_read v6, a40 // (op 3745) 
v_accvgpr_read v7, a41 // (op 3745) 
v_accvgpr_read v8, a42 // (op 3745) 
v_accvgpr_read v9, a43 // (op 3745) 
v_mul_f32 v14, v11, v6 // (op 3745) 
v_mul_f32 v15, v11, v7 // (op 3745) 
v_mul_f32 v16, v11, v8 // (op 3745) 
v_mul_f32 v17, v11, v9 // (op 3745) 

v_lshrrev_b32 v6, 16, v14 // (op 3747) 
v_lshrrev_b32 v7, 16, v15 // (op 3747) 
v_lshrrev_b32 v8, 16, v16 // (op 3747) 
v_lshrrev_b32 v9, 16, v17 // (op 3747) 


v_pack_b32_f16 v4, v6, v7 // (op 3725) 
v_pack_b32_f16 v5, v8, v9 // (op 3725) 

s_lshl_b32 s0, s4, 7 // (op 3725) 
s_add_u32 s0, s0, 64 // (op 3725) 
v_add_u32 v0, v13, s0 // (op 3725) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3725) Store value



v_accvgpr_read v6, a44 // (op 3751) 
v_accvgpr_read v7, a45 // (op 3751) 
v_accvgpr_read v8, a46 // (op 3751) 
v_accvgpr_read v9, a47 // (op 3751) 
v_mul_f32 v14, v11, v6 // (op 3751) 
v_mul_f32 v15, v11, v7 // (op 3751) 
v_mul_f32 v16, v11, v8 // (op 3751) 
v_mul_f32 v17, v11, v9 // (op 3751) 

v_lshrrev_b32 v6, 16, v14 // (op 3753) 
v_lshrrev_b32 v7, 16, v15 // (op 3753) 
v_lshrrev_b32 v8, 16, v16 // (op 3753) 
v_lshrrev_b32 v9, 16, v17 // (op 3753) 


v_pack_b32_f16 v4, v6, v7 // (op 3727) 
v_pack_b32_f16 v5, v8, v9 // (op 3727) 

s_lshl_b32 s0, s4, 6 // (op 3727) 
s_mul_i32 s1, 3, s0 // (op 3727) 
s_add_u32 s1, s1, 64 // (op 3727) 

v_add_u32 v0, v13, s1 // (op 3727) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3727) Store value



v_accvgpr_read v6, a48 // (op 3757) 
v_accvgpr_read v7, a49 // (op 3757) 
v_accvgpr_read v8, a50 // (op 3757) 
v_accvgpr_read v9, a51 // (op 3757) 
v_mul_f32 v14, v11, v6 // (op 3757) 
v_mul_f32 v15, v11, v7 // (op 3757) 
v_mul_f32 v16, v11, v8 // (op 3757) 
v_mul_f32 v17, v11, v9 // (op 3757) 

v_lshrrev_b32 v6, 16, v14 // (op 3759) 
v_lshrrev_b32 v7, 16, v15 // (op 3759) 
v_lshrrev_b32 v8, 16, v16 // (op 3759) 
v_lshrrev_b32 v9, 16, v17 // (op 3759) 


v_pack_b32_f16 v4, v6, v7 // (op 3729) 
v_pack_b32_f16 v5, v8, v9 // (op 3729) 

s_lshl_b32 s0, s4, 8 // (op 3729) 
s_add_u32 s0, s0, 64 // (op 3729) 
v_add_u32 v0, v13, s0 // (op 3729) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3729) Store value



v_accvgpr_read v6, a52 // (op 3763) 
v_accvgpr_read v7, a53 // (op 3763) 
v_accvgpr_read v8, a54 // (op 3763) 
v_accvgpr_read v9, a55 // (op 3763) 
v_mul_f32 v14, v11, v6 // (op 3763) 
v_mul_f32 v15, v11, v7 // (op 3763) 
v_mul_f32 v16, v11, v8 // (op 3763) 
v_mul_f32 v17, v11, v9 // (op 3763) 

v_lshrrev_b32 v6, 16, v14 // (op 3765) 
v_lshrrev_b32 v7, 16, v15 // (op 3765) 
v_lshrrev_b32 v8, 16, v16 // (op 3765) 
v_lshrrev_b32 v9, 16, v17 // (op 3765) 


v_pack_b32_f16 v4, v6, v7 // (op 3731) 
v_pack_b32_f16 v5, v8, v9 // (op 3731) 

s_lshl_b32 s0, s4, 6 // (op 3731) 
s_mul_i32 s1, 5, s0 // (op 3731) 
s_add_u32 s1, s1, 64 // (op 3731) 

v_add_u32 v0, v13, s1 // (op 3731) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3731) Store value



v_accvgpr_read v6, a56 // (op 3769) 
v_accvgpr_read v7, a57 // (op 3769) 
v_accvgpr_read v8, a58 // (op 3769) 
v_accvgpr_read v9, a59 // (op 3769) 
v_mul_f32 v14, v11, v6 // (op 3769) 
v_mul_f32 v15, v11, v7 // (op 3769) 
v_mul_f32 v16, v11, v8 // (op 3769) 
v_mul_f32 v17, v11, v9 // (op 3769) 

v_lshrrev_b32 v6, 16, v14 // (op 3771) 
v_lshrrev_b32 v7, 16, v15 // (op 3771) 
v_lshrrev_b32 v8, 16, v16 // (op 3771) 
v_lshrrev_b32 v9, 16, v17 // (op 3771) 


v_pack_b32_f16 v4, v6, v7 // (op 3733) 
v_pack_b32_f16 v5, v8, v9 // (op 3733) 

s_lshl_b32 s0, s4, 6 // (op 3733) 
s_mul_i32 s1, 6, s0 // (op 3733) 
s_add_u32 s1, s1, 64 // (op 3733) 

v_add_u32 v0, v13, s1 // (op 3733) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3733) Store value



v_accvgpr_read v6, a60 // (op 3775) 
v_accvgpr_read v7, a61 // (op 3775) 
v_accvgpr_read v8, a62 // (op 3775) 
v_accvgpr_read v9, a63 // (op 3775) 
v_mul_f32 v14, v11, v6 // (op 3775) 
v_mul_f32 v15, v11, v7 // (op 3775) 
v_mul_f32 v16, v11, v8 // (op 3775) 
v_mul_f32 v17, v11, v9 // (op 3775) 

v_lshrrev_b32 v6, 16, v14 // (op 3777) 
v_lshrrev_b32 v7, 16, v15 // (op 3777) 
v_lshrrev_b32 v8, 16, v16 // (op 3777) 
v_lshrrev_b32 v9, 16, v17 // (op 3777) 


v_pack_b32_f16 v4, v6, v7 // (op 3735) 
v_pack_b32_f16 v5, v8, v9 // (op 3735) 

s_lshl_b32 s0, s4, 6 // (op 3735) 
s_mul_i32 s1, 7, s0 // (op 3735) 
s_add_u32 s1, s1, 64 // (op 3735) 

v_add_u32 v0, v13, s1 // (op 3735) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3735) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v12 // (op 4113) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4113) Load value

v_accvgpr_read v4, a32 // (op 4092) 
v_accvgpr_read v5, a33 // (op 4092) 
v_accvgpr_read v6, a34 // (op 4092) 
v_accvgpr_read v7, a35 // (op 4092) 
v_mul_f32 v14, v11, v4 // (op 4092) 
v_mul_f32 v15, v11, v5 // (op 4092) 
v_mul_f32 v16, v11, v6 // (op 4092) 
v_mul_f32 v17, v11, v7 // (op 4092) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4112) 
v_lshrrev_b32 v7, 16, v8 // (op 4112) 
v_lshlrev_b32 v6, 16, v6 // (op 4112) 
v_lshlrev_b32 v7, 16, v7 // (op 4112) 
v_mul_f32 v18, v10, v6 // (op 4112) 
v_mul_f32 v19, v10, v7 // (op 4112) 
v_and_b32 v6, 65535, v9 // (op 4112) 
v_lshrrev_b32 v7, 16, v9 // (op 4112) 
v_lshlrev_b32 v6, 16, v6 // (op 4112) 
v_lshlrev_b32 v7, 16, v7 // (op 4112) 
v_mul_f32 v20, v10, v6 // (op 4112) 
v_mul_f32 v21, v10, v7 // (op 4112) 


v_add_f32 v6, v18, v14 // (op 4093) 
v_add_f32 v7, v19, v15 // (op 4093) 
v_add_f32 v8, v20, v16 // (op 4093) 
v_add_f32 v9, v21, v17 // (op 4093) 


v_lshrrev_b32 v14, 16, v6 // (op 4094) 
v_lshrrev_b32 v15, 16, v7 // (op 4094) 
v_lshrrev_b32 v16, 16, v8 // (op 4094) 
v_lshrrev_b32 v17, 16, v9 // (op 4094) 


v_pack_b32_f16 v8, v14, v15 // (op 4096) 
v_pack_b32_f16 v9, v16, v17 // (op 4096) 

v_add_u32 v0, 64, v13 // (op 4096) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4096) Store value





s_lshl_b32 s0, s2, 6 // (op 4118) 
s_add_u32 s0, s0, 64 // (op 4118) 
v_add_u32 v0, v12, s0 // (op 4118) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4118) Load value

v_accvgpr_read v4, a36 // (op 4150) 
v_accvgpr_read v5, a37 // (op 4150) 
v_accvgpr_read v6, a38 // (op 4150) 
v_accvgpr_read v7, a39 // (op 4150) 
v_mul_f32 v14, v11, v4 // (op 4150) 
v_mul_f32 v15, v11, v5 // (op 4150) 
v_mul_f32 v16, v11, v6 // (op 4150) 
v_mul_f32 v17, v11, v7 // (op 4150) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4115) 
v_lshrrev_b32 v7, 16, v8 // (op 4115) 
v_lshlrev_b32 v6, 16, v6 // (op 4115) 
v_lshlrev_b32 v7, 16, v7 // (op 4115) 
v_mul_f32 v18, v10, v6 // (op 4115) 
v_mul_f32 v19, v10, v7 // (op 4115) 
v_and_b32 v6, 65535, v9 // (op 4115) 
v_lshrrev_b32 v7, 16, v9 // (op 4115) 
v_lshlrev_b32 v6, 16, v6 // (op 4115) 
v_lshlrev_b32 v7, 16, v7 // (op 4115) 
v_mul_f32 v20, v10, v6 // (op 4115) 
v_mul_f32 v21, v10, v7 // (op 4115) 


v_add_f32 v6, v18, v14 // (op 4116) 
v_add_f32 v7, v19, v15 // (op 4116) 
v_add_f32 v8, v20, v16 // (op 4116) 
v_add_f32 v9, v21, v17 // (op 4116) 


v_lshrrev_b32 v14, 16, v6 // (op 4117) 
v_lshrrev_b32 v15, 16, v7 // (op 4117) 
v_lshrrev_b32 v16, 16, v8 // (op 4117) 
v_lshrrev_b32 v17, 16, v9 // (op 4117) 


v_pack_b32_f16 v8, v14, v15 // (op 4098) 
v_pack_b32_f16 v9, v16, v17 // (op 4098) 

s_lshl_b32 s0, s4, 6 // (op 4098) 
s_add_u32 s0, s0, 64 // (op 4098) 
v_add_u32 v0, v13, s0 // (op 4098) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4098) Store value





s_lshl_b32 s0, s2, 7 // (op 4123) 
s_add_u32 s0, s0, 64 // (op 4123) 
v_add_u32 v0, v12, s0 // (op 4123) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4123) Load value

v_accvgpr_read v4, a40 // (op 4152) 
v_accvgpr_read v5, a41 // (op 4152) 
v_accvgpr_read v6, a42 // (op 4152) 
v_accvgpr_read v7, a43 // (op 4152) 
v_mul_f32 v14, v11, v4 // (op 4152) 
v_mul_f32 v15, v11, v5 // (op 4152) 
v_mul_f32 v16, v11, v6 // (op 4152) 
v_mul_f32 v17, v11, v7 // (op 4152) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4120) 
v_lshrrev_b32 v7, 16, v8 // (op 4120) 
v_lshlrev_b32 v6, 16, v6 // (op 4120) 
v_lshlrev_b32 v7, 16, v7 // (op 4120) 
v_mul_f32 v18, v10, v6 // (op 4120) 
v_mul_f32 v19, v10, v7 // (op 4120) 
v_and_b32 v6, 65535, v9 // (op 4120) 
v_lshrrev_b32 v7, 16, v9 // (op 4120) 
v_lshlrev_b32 v6, 16, v6 // (op 4120) 
v_lshlrev_b32 v7, 16, v7 // (op 4120) 
v_mul_f32 v20, v10, v6 // (op 4120) 
v_mul_f32 v21, v10, v7 // (op 4120) 


v_add_f32 v6, v18, v14 // (op 4121) 
v_add_f32 v7, v19, v15 // (op 4121) 
v_add_f32 v8, v20, v16 // (op 4121) 
v_add_f32 v9, v21, v17 // (op 4121) 


v_lshrrev_b32 v14, 16, v6 // (op 4122) 
v_lshrrev_b32 v15, 16, v7 // (op 4122) 
v_lshrrev_b32 v16, 16, v8 // (op 4122) 
v_lshrrev_b32 v17, 16, v9 // (op 4122) 


v_pack_b32_f16 v8, v14, v15 // (op 4100) 
v_pack_b32_f16 v9, v16, v17 // (op 4100) 

s_lshl_b32 s0, s4, 7 // (op 4100) 
s_add_u32 s0, s0, 64 // (op 4100) 
v_add_u32 v0, v13, s0 // (op 4100) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4100) Store value





s_lshl_b32 s0, s2, 6 // (op 4128) 
s_mul_i32 s1, 3, s0 // (op 4128) 
s_add_u32 s1, s1, 64 // (op 4128) 

v_add_u32 v0, v12, s1 // (op 4128) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4128) Load value

v_accvgpr_read v4, a44 // (op 4154) 
v_accvgpr_read v5, a45 // (op 4154) 
v_accvgpr_read v6, a46 // (op 4154) 
v_accvgpr_read v7, a47 // (op 4154) 
v_mul_f32 v14, v11, v4 // (op 4154) 
v_mul_f32 v15, v11, v5 // (op 4154) 
v_mul_f32 v16, v11, v6 // (op 4154) 
v_mul_f32 v17, v11, v7 // (op 4154) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4125) 
v_lshrrev_b32 v7, 16, v8 // (op 4125) 
v_lshlrev_b32 v6, 16, v6 // (op 4125) 
v_lshlrev_b32 v7, 16, v7 // (op 4125) 
v_mul_f32 v18, v10, v6 // (op 4125) 
v_mul_f32 v19, v10, v7 // (op 4125) 
v_and_b32 v6, 65535, v9 // (op 4125) 
v_lshrrev_b32 v7, 16, v9 // (op 4125) 
v_lshlrev_b32 v6, 16, v6 // (op 4125) 
v_lshlrev_b32 v7, 16, v7 // (op 4125) 
v_mul_f32 v20, v10, v6 // (op 4125) 
v_mul_f32 v21, v10, v7 // (op 4125) 


v_add_f32 v6, v18, v14 // (op 4126) 
v_add_f32 v7, v19, v15 // (op 4126) 
v_add_f32 v8, v20, v16 // (op 4126) 
v_add_f32 v9, v21, v17 // (op 4126) 


v_lshrrev_b32 v14, 16, v6 // (op 4127) 
v_lshrrev_b32 v15, 16, v7 // (op 4127) 
v_lshrrev_b32 v16, 16, v8 // (op 4127) 
v_lshrrev_b32 v17, 16, v9 // (op 4127) 


v_pack_b32_f16 v8, v14, v15 // (op 4102) 
v_pack_b32_f16 v9, v16, v17 // (op 4102) 

s_lshl_b32 s0, s4, 6 // (op 4102) 
s_mul_i32 s1, 3, s0 // (op 4102) 
s_add_u32 s1, s1, 64 // (op 4102) 

v_add_u32 v0, v13, s1 // (op 4102) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4102) Store value





s_lshl_b32 s0, s2, 8 // (op 4133) 
s_add_u32 s0, s0, 64 // (op 4133) 
v_add_u32 v0, v12, s0 // (op 4133) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4133) Load value

v_accvgpr_read v4, a48 // (op 4156) 
v_accvgpr_read v5, a49 // (op 4156) 
v_accvgpr_read v6, a50 // (op 4156) 
v_accvgpr_read v7, a51 // (op 4156) 
v_mul_f32 v14, v11, v4 // (op 4156) 
v_mul_f32 v15, v11, v5 // (op 4156) 
v_mul_f32 v16, v11, v6 // (op 4156) 
v_mul_f32 v17, v11, v7 // (op 4156) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4130) 
v_lshrrev_b32 v7, 16, v8 // (op 4130) 
v_lshlrev_b32 v6, 16, v6 // (op 4130) 
v_lshlrev_b32 v7, 16, v7 // (op 4130) 
v_mul_f32 v18, v10, v6 // (op 4130) 
v_mul_f32 v19, v10, v7 // (op 4130) 
v_and_b32 v6, 65535, v9 // (op 4130) 
v_lshrrev_b32 v7, 16, v9 // (op 4130) 
v_lshlrev_b32 v6, 16, v6 // (op 4130) 
v_lshlrev_b32 v7, 16, v7 // (op 4130) 
v_mul_f32 v20, v10, v6 // (op 4130) 
v_mul_f32 v21, v10, v7 // (op 4130) 


v_add_f32 v6, v18, v14 // (op 4131) 
v_add_f32 v7, v19, v15 // (op 4131) 
v_add_f32 v8, v20, v16 // (op 4131) 
v_add_f32 v9, v21, v17 // (op 4131) 


v_lshrrev_b32 v14, 16, v6 // (op 4132) 
v_lshrrev_b32 v15, 16, v7 // (op 4132) 
v_lshrrev_b32 v16, 16, v8 // (op 4132) 
v_lshrrev_b32 v17, 16, v9 // (op 4132) 


v_pack_b32_f16 v8, v14, v15 // (op 4104) 
v_pack_b32_f16 v9, v16, v17 // (op 4104) 

s_lshl_b32 s0, s4, 8 // (op 4104) 
s_add_u32 s0, s0, 64 // (op 4104) 
v_add_u32 v0, v13, s0 // (op 4104) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4104) Store value





s_lshl_b32 s0, s2, 6 // (op 4138) 
s_mul_i32 s1, 5, s0 // (op 4138) 
s_add_u32 s1, s1, 64 // (op 4138) 

v_add_u32 v0, v12, s1 // (op 4138) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4138) Load value

v_accvgpr_read v4, a52 // (op 4158) 
v_accvgpr_read v5, a53 // (op 4158) 
v_accvgpr_read v6, a54 // (op 4158) 
v_accvgpr_read v7, a55 // (op 4158) 
v_mul_f32 v14, v11, v4 // (op 4158) 
v_mul_f32 v15, v11, v5 // (op 4158) 
v_mul_f32 v16, v11, v6 // (op 4158) 
v_mul_f32 v17, v11, v7 // (op 4158) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4135) 
v_lshrrev_b32 v7, 16, v8 // (op 4135) 
v_lshlrev_b32 v6, 16, v6 // (op 4135) 
v_lshlrev_b32 v7, 16, v7 // (op 4135) 
v_mul_f32 v18, v10, v6 // (op 4135) 
v_mul_f32 v19, v10, v7 // (op 4135) 
v_and_b32 v6, 65535, v9 // (op 4135) 
v_lshrrev_b32 v7, 16, v9 // (op 4135) 
v_lshlrev_b32 v6, 16, v6 // (op 4135) 
v_lshlrev_b32 v7, 16, v7 // (op 4135) 
v_mul_f32 v20, v10, v6 // (op 4135) 
v_mul_f32 v21, v10, v7 // (op 4135) 


v_add_f32 v6, v18, v14 // (op 4136) 
v_add_f32 v7, v19, v15 // (op 4136) 
v_add_f32 v8, v20, v16 // (op 4136) 
v_add_f32 v9, v21, v17 // (op 4136) 


v_lshrrev_b32 v14, 16, v6 // (op 4137) 
v_lshrrev_b32 v15, 16, v7 // (op 4137) 
v_lshrrev_b32 v16, 16, v8 // (op 4137) 
v_lshrrev_b32 v17, 16, v9 // (op 4137) 


v_pack_b32_f16 v8, v14, v15 // (op 4106) 
v_pack_b32_f16 v9, v16, v17 // (op 4106) 

s_lshl_b32 s0, s4, 6 // (op 4106) 
s_mul_i32 s1, 5, s0 // (op 4106) 
s_add_u32 s1, s1, 64 // (op 4106) 

v_add_u32 v0, v13, s1 // (op 4106) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4106) Store value





s_lshl_b32 s0, s2, 6 // (op 4143) 
s_mul_i32 s1, 6, s0 // (op 4143) 
s_add_u32 s1, s1, 64 // (op 4143) 

v_add_u32 v0, v12, s1 // (op 4143) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4143) Load value

v_accvgpr_read v4, a56 // (op 4160) 
v_accvgpr_read v5, a57 // (op 4160) 
v_accvgpr_read v6, a58 // (op 4160) 
v_accvgpr_read v7, a59 // (op 4160) 
v_mul_f32 v14, v11, v4 // (op 4160) 
v_mul_f32 v15, v11, v5 // (op 4160) 
v_mul_f32 v16, v11, v6 // (op 4160) 
v_mul_f32 v17, v11, v7 // (op 4160) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4140) 
v_lshrrev_b32 v7, 16, v8 // (op 4140) 
v_lshlrev_b32 v6, 16, v6 // (op 4140) 
v_lshlrev_b32 v7, 16, v7 // (op 4140) 
v_mul_f32 v18, v10, v6 // (op 4140) 
v_mul_f32 v19, v10, v7 // (op 4140) 
v_and_b32 v6, 65535, v9 // (op 4140) 
v_lshrrev_b32 v7, 16, v9 // (op 4140) 
v_lshlrev_b32 v6, 16, v6 // (op 4140) 
v_lshlrev_b32 v7, 16, v7 // (op 4140) 
v_mul_f32 v20, v10, v6 // (op 4140) 
v_mul_f32 v21, v10, v7 // (op 4140) 


v_add_f32 v6, v18, v14 // (op 4141) 
v_add_f32 v7, v19, v15 // (op 4141) 
v_add_f32 v8, v20, v16 // (op 4141) 
v_add_f32 v9, v21, v17 // (op 4141) 


v_lshrrev_b32 v14, 16, v6 // (op 4142) 
v_lshrrev_b32 v15, 16, v7 // (op 4142) 
v_lshrrev_b32 v16, 16, v8 // (op 4142) 
v_lshrrev_b32 v17, 16, v9 // (op 4142) 


v_pack_b32_f16 v8, v14, v15 // (op 4108) 
v_pack_b32_f16 v9, v16, v17 // (op 4108) 

s_lshl_b32 s0, s4, 6 // (op 4108) 
s_mul_i32 s1, 6, s0 // (op 4108) 
s_add_u32 s1, s1, 64 // (op 4108) 

v_add_u32 v0, v13, s1 // (op 4108) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4108) Store value





s_lshl_b32 s0, s2, 6 // (op 4148) 
s_mul_i32 s1, 7, s0 // (op 4148) 
s_add_u32 s1, s1, 64 // (op 4148) 

v_add_u32 v0, v12, s1 // (op 4148) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4148) Load value

v_accvgpr_read v4, a60 // (op 4162) 
v_accvgpr_read v5, a61 // (op 4162) 
v_accvgpr_read v6, a62 // (op 4162) 
v_accvgpr_read v7, a63 // (op 4162) 
v_mul_f32 v14, v11, v4 // (op 4162) 
v_mul_f32 v15, v11, v5 // (op 4162) 
v_mul_f32 v16, v11, v6 // (op 4162) 
v_mul_f32 v17, v11, v7 // (op 4162) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4145) 
v_lshrrev_b32 v7, 16, v8 // (op 4145) 
v_lshlrev_b32 v6, 16, v6 // (op 4145) 
v_lshlrev_b32 v7, 16, v7 // (op 4145) 
v_mul_f32 v18, v10, v6 // (op 4145) 
v_mul_f32 v19, v10, v7 // (op 4145) 
v_and_b32 v6, 65535, v9 // (op 4145) 
v_lshrrev_b32 v7, 16, v9 // (op 4145) 
v_lshlrev_b32 v6, 16, v6 // (op 4145) 
v_lshlrev_b32 v7, 16, v7 // (op 4145) 
v_mul_f32 v20, v10, v6 // (op 4145) 
v_mul_f32 v21, v10, v7 // (op 4145) 


v_add_f32 v6, v18, v14 // (op 4146) 
v_add_f32 v7, v19, v15 // (op 4146) 
v_add_f32 v8, v20, v16 // (op 4146) 
v_add_f32 v9, v21, v17 // (op 4146) 


v_lshrrev_b32 v14, 16, v6 // (op 4147) 
v_lshrrev_b32 v15, 16, v7 // (op 4147) 
v_lshrrev_b32 v16, 16, v8 // (op 4147) 
v_lshrrev_b32 v17, 16, v9 // (op 4147) 


v_pack_b32_f16 v8, v14, v15 // (op 4110) 
v_pack_b32_f16 v9, v16, v17 // (op 4110) 

s_lshl_b32 s0, s4, 6 // (op 4110) 
s_mul_i32 s1, 7, s0 // (op 4110) 
s_add_u32 s1, s1, 64 // (op 4110) 

v_add_u32 v0, v13, s1 // (op 4110) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4110) Store value





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





s_lshl_b32 s0, s2, 6 // (op 3917) 
v_add_u32 v0, v12, s0 // (op 3917) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3917) Load value

v_accvgpr_read v4, a4 // (op 3913) 
v_accvgpr_read v5, a5 // (op 3913) 
v_accvgpr_read v6, a6 // (op 3913) 
v_accvgpr_read v7, a7 // (op 3913) 
v_mul_f32 v14, v11, v4 // (op 3913) 
v_mul_f32 v15, v11, v5 // (op 3913) 
v_mul_f32 v16, v11, v6 // (op 3913) 
v_mul_f32 v17, v11, v7 // (op 3913) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3918) 
v_lshrrev_b32 v7, 16, v8 // (op 3918) 
v_lshlrev_b32 v6, 16, v6 // (op 3918) 
v_lshlrev_b32 v7, 16, v7 // (op 3918) 
v_mul_f32 v18, v10, v6 // (op 3918) 
v_mul_f32 v19, v10, v7 // (op 3918) 
v_and_b32 v6, 65535, v9 // (op 3918) 
v_lshrrev_b32 v7, 16, v9 // (op 3918) 
v_lshlrev_b32 v6, 16, v6 // (op 3918) 
v_lshlrev_b32 v7, 16, v7 // (op 3918) 
v_mul_f32 v20, v10, v6 // (op 3918) 
v_mul_f32 v21, v10, v7 // (op 3918) 


v_add_f32 v6, v18, v14 // (op 3914) 
v_add_f32 v7, v19, v15 // (op 3914) 
v_add_f32 v8, v20, v16 // (op 3914) 
v_add_f32 v9, v21, v17 // (op 3914) 


v_lshrrev_b32 v14, 16, v6 // (op 3915) 
v_lshrrev_b32 v15, 16, v7 // (op 3915) 
v_lshrrev_b32 v16, 16, v8 // (op 3915) 
v_lshrrev_b32 v17, 16, v9 // (op 3915) 


v_pack_b32_f16 v8, v14, v15 // (op 3916) 
v_pack_b32_f16 v9, v16, v17 // (op 3916) 

s_lshl_b32 s0, s4, 6 // (op 3916) 
v_add_u32 v0, v13, s0 // (op 3916) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3916) Store value





s_lshl_b32 s0, s2, 7 // (op 3931) 
v_add_u32 v0, v12, s0 // (op 3931) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3931) Load value

v_accvgpr_read v4, a8 // (op 3927) 
v_accvgpr_read v5, a9 // (op 3927) 
v_accvgpr_read v6, a10 // (op 3927) 
v_accvgpr_read v7, a11 // (op 3927) 
v_mul_f32 v14, v11, v4 // (op 3927) 
v_mul_f32 v15, v11, v5 // (op 3927) 
v_mul_f32 v16, v11, v6 // (op 3927) 
v_mul_f32 v17, v11, v7 // (op 3927) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3932) 
v_lshrrev_b32 v7, 16, v8 // (op 3932) 
v_lshlrev_b32 v6, 16, v6 // (op 3932) 
v_lshlrev_b32 v7, 16, v7 // (op 3932) 
v_mul_f32 v18, v10, v6 // (op 3932) 
v_mul_f32 v19, v10, v7 // (op 3932) 
v_and_b32 v6, 65535, v9 // (op 3932) 
v_lshrrev_b32 v7, 16, v9 // (op 3932) 
v_lshlrev_b32 v6, 16, v6 // (op 3932) 
v_lshlrev_b32 v7, 16, v7 // (op 3932) 
v_mul_f32 v20, v10, v6 // (op 3932) 
v_mul_f32 v21, v10, v7 // (op 3932) 


v_add_f32 v6, v18, v14 // (op 3928) 
v_add_f32 v7, v19, v15 // (op 3928) 
v_add_f32 v8, v20, v16 // (op 3928) 
v_add_f32 v9, v21, v17 // (op 3928) 


v_lshrrev_b32 v14, 16, v6 // (op 3929) 
v_lshrrev_b32 v15, 16, v7 // (op 3929) 
v_lshrrev_b32 v16, 16, v8 // (op 3929) 
v_lshrrev_b32 v17, 16, v9 // (op 3929) 


v_pack_b32_f16 v8, v14, v15 // (op 3930) 
v_pack_b32_f16 v9, v16, v17 // (op 3930) 

s_lshl_b32 s0, s4, 7 // (op 3930) 
v_add_u32 v0, v13, s0 // (op 3930) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3930) Store value





s_lshl_b32 s0, s2, 6 // (op 3945) 
s_mul_i32 s1, 3, s0 // (op 3945) 

v_add_u32 v0, v12, s1 // (op 3945) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3945) Load value

v_accvgpr_read v4, a12 // (op 3941) 
v_accvgpr_read v5, a13 // (op 3941) 
v_accvgpr_read v6, a14 // (op 3941) 
v_accvgpr_read v7, a15 // (op 3941) 
v_mul_f32 v14, v11, v4 // (op 3941) 
v_mul_f32 v15, v11, v5 // (op 3941) 
v_mul_f32 v16, v11, v6 // (op 3941) 
v_mul_f32 v17, v11, v7 // (op 3941) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3946) 
v_lshrrev_b32 v7, 16, v8 // (op 3946) 
v_lshlrev_b32 v6, 16, v6 // (op 3946) 
v_lshlrev_b32 v7, 16, v7 // (op 3946) 
v_mul_f32 v18, v10, v6 // (op 3946) 
v_mul_f32 v19, v10, v7 // (op 3946) 
v_and_b32 v6, 65535, v9 // (op 3946) 
v_lshrrev_b32 v7, 16, v9 // (op 3946) 
v_lshlrev_b32 v6, 16, v6 // (op 3946) 
v_lshlrev_b32 v7, 16, v7 // (op 3946) 
v_mul_f32 v20, v10, v6 // (op 3946) 
v_mul_f32 v21, v10, v7 // (op 3946) 


v_add_f32 v6, v18, v14 // (op 3942) 
v_add_f32 v7, v19, v15 // (op 3942) 
v_add_f32 v8, v20, v16 // (op 3942) 
v_add_f32 v9, v21, v17 // (op 3942) 


v_lshrrev_b32 v14, 16, v6 // (op 3943) 
v_lshrrev_b32 v15, 16, v7 // (op 3943) 
v_lshrrev_b32 v16, 16, v8 // (op 3943) 
v_lshrrev_b32 v17, 16, v9 // (op 3943) 


v_pack_b32_f16 v8, v14, v15 // (op 3944) 
v_pack_b32_f16 v9, v16, v17 // (op 3944) 

s_lshl_b32 s0, s4, 6 // (op 3944) 
s_mul_i32 s1, 3, s0 // (op 3944) 

v_add_u32 v0, v13, s1 // (op 3944) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3944) Store value





s_lshl_b32 s0, s2, 8 // (op 3959) 
v_add_u32 v0, v12, s0 // (op 3959) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3959) Load value

v_accvgpr_read v4, a16 // (op 3955) 
v_accvgpr_read v5, a17 // (op 3955) 
v_accvgpr_read v6, a18 // (op 3955) 
v_accvgpr_read v7, a19 // (op 3955) 
v_mul_f32 v14, v11, v4 // (op 3955) 
v_mul_f32 v15, v11, v5 // (op 3955) 
v_mul_f32 v16, v11, v6 // (op 3955) 
v_mul_f32 v17, v11, v7 // (op 3955) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3960) 
v_lshrrev_b32 v7, 16, v8 // (op 3960) 
v_lshlrev_b32 v6, 16, v6 // (op 3960) 
v_lshlrev_b32 v7, 16, v7 // (op 3960) 
v_mul_f32 v18, v10, v6 // (op 3960) 
v_mul_f32 v19, v10, v7 // (op 3960) 
v_and_b32 v6, 65535, v9 // (op 3960) 
v_lshrrev_b32 v7, 16, v9 // (op 3960) 
v_lshlrev_b32 v6, 16, v6 // (op 3960) 
v_lshlrev_b32 v7, 16, v7 // (op 3960) 
v_mul_f32 v20, v10, v6 // (op 3960) 
v_mul_f32 v21, v10, v7 // (op 3960) 


v_add_f32 v6, v18, v14 // (op 3956) 
v_add_f32 v7, v19, v15 // (op 3956) 
v_add_f32 v8, v20, v16 // (op 3956) 
v_add_f32 v9, v21, v17 // (op 3956) 


v_lshrrev_b32 v14, 16, v6 // (op 3957) 
v_lshrrev_b32 v15, 16, v7 // (op 3957) 
v_lshrrev_b32 v16, 16, v8 // (op 3957) 
v_lshrrev_b32 v17, 16, v9 // (op 3957) 


v_pack_b32_f16 v8, v14, v15 // (op 3958) 
v_pack_b32_f16 v9, v16, v17 // (op 3958) 

s_lshl_b32 s0, s4, 8 // (op 3958) 
v_add_u32 v0, v13, s0 // (op 3958) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3958) Store value





s_lshl_b32 s0, s2, 6 // (op 3973) 
s_mul_i32 s1, 5, s0 // (op 3973) 

v_add_u32 v0, v12, s1 // (op 3973) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3973) Load value

v_accvgpr_read v4, a20 // (op 3969) 
v_accvgpr_read v5, a21 // (op 3969) 
v_accvgpr_read v6, a22 // (op 3969) 
v_accvgpr_read v7, a23 // (op 3969) 
v_mul_f32 v14, v11, v4 // (op 3969) 
v_mul_f32 v15, v11, v5 // (op 3969) 
v_mul_f32 v16, v11, v6 // (op 3969) 
v_mul_f32 v17, v11, v7 // (op 3969) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3974) 
v_lshrrev_b32 v7, 16, v8 // (op 3974) 
v_lshlrev_b32 v6, 16, v6 // (op 3974) 
v_lshlrev_b32 v7, 16, v7 // (op 3974) 
v_mul_f32 v18, v10, v6 // (op 3974) 
v_mul_f32 v19, v10, v7 // (op 3974) 
v_and_b32 v6, 65535, v9 // (op 3974) 
v_lshrrev_b32 v7, 16, v9 // (op 3974) 
v_lshlrev_b32 v6, 16, v6 // (op 3974) 
v_lshlrev_b32 v7, 16, v7 // (op 3974) 
v_mul_f32 v20, v10, v6 // (op 3974) 
v_mul_f32 v21, v10, v7 // (op 3974) 


v_add_f32 v6, v18, v14 // (op 3970) 
v_add_f32 v7, v19, v15 // (op 3970) 
v_add_f32 v8, v20, v16 // (op 3970) 
v_add_f32 v9, v21, v17 // (op 3970) 


v_lshrrev_b32 v14, 16, v6 // (op 3971) 
v_lshrrev_b32 v15, 16, v7 // (op 3971) 
v_lshrrev_b32 v16, 16, v8 // (op 3971) 
v_lshrrev_b32 v17, 16, v9 // (op 3971) 


v_pack_b32_f16 v8, v14, v15 // (op 3972) 
v_pack_b32_f16 v9, v16, v17 // (op 3972) 

s_lshl_b32 s0, s4, 6 // (op 3972) 
s_mul_i32 s1, 5, s0 // (op 3972) 

v_add_u32 v0, v13, s1 // (op 3972) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3972) Store value





s_lshl_b32 s0, s2, 6 // (op 3987) 
s_mul_i32 s1, 6, s0 // (op 3987) 

v_add_u32 v0, v12, s1 // (op 3987) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3987) Load value

v_accvgpr_read v4, a24 // (op 3983) 
v_accvgpr_read v5, a25 // (op 3983) 
v_accvgpr_read v6, a26 // (op 3983) 
v_accvgpr_read v7, a27 // (op 3983) 
v_mul_f32 v14, v11, v4 // (op 3983) 
v_mul_f32 v15, v11, v5 // (op 3983) 
v_mul_f32 v16, v11, v6 // (op 3983) 
v_mul_f32 v17, v11, v7 // (op 3983) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3988) 
v_lshrrev_b32 v7, 16, v8 // (op 3988) 
v_lshlrev_b32 v6, 16, v6 // (op 3988) 
v_lshlrev_b32 v7, 16, v7 // (op 3988) 
v_mul_f32 v18, v10, v6 // (op 3988) 
v_mul_f32 v19, v10, v7 // (op 3988) 
v_and_b32 v6, 65535, v9 // (op 3988) 
v_lshrrev_b32 v7, 16, v9 // (op 3988) 
v_lshlrev_b32 v6, 16, v6 // (op 3988) 
v_lshlrev_b32 v7, 16, v7 // (op 3988) 
v_mul_f32 v20, v10, v6 // (op 3988) 
v_mul_f32 v21, v10, v7 // (op 3988) 


v_add_f32 v6, v18, v14 // (op 3984) 
v_add_f32 v7, v19, v15 // (op 3984) 
v_add_f32 v8, v20, v16 // (op 3984) 
v_add_f32 v9, v21, v17 // (op 3984) 


v_lshrrev_b32 v14, 16, v6 // (op 3985) 
v_lshrrev_b32 v15, 16, v7 // (op 3985) 
v_lshrrev_b32 v16, 16, v8 // (op 3985) 
v_lshrrev_b32 v17, 16, v9 // (op 3985) 


v_pack_b32_f16 v8, v14, v15 // (op 3986) 
v_pack_b32_f16 v9, v16, v17 // (op 3986) 

s_lshl_b32 s0, s4, 6 // (op 3986) 
s_mul_i32 s1, 6, s0 // (op 3986) 

v_add_u32 v0, v13, s1 // (op 3986) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3986) Store value





s_lshl_b32 s0, s2, 6 // (op 4001) 
s_mul_i32 s1, 7, s0 // (op 4001) 

v_add_u32 v0, v12, s1 // (op 4001) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4001) Load value

v_accvgpr_read v4, a28 // (op 3997) 
v_accvgpr_read v5, a29 // (op 3997) 
v_accvgpr_read v6, a30 // (op 3997) 
v_accvgpr_read v7, a31 // (op 3997) 
v_mul_f32 v14, v11, v4 // (op 3997) 
v_mul_f32 v15, v11, v5 // (op 3997) 
v_mul_f32 v16, v11, v6 // (op 3997) 
v_mul_f32 v17, v11, v7 // (op 3997) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4002) 
v_lshrrev_b32 v7, 16, v8 // (op 4002) 
v_lshlrev_b32 v6, 16, v6 // (op 4002) 
v_lshlrev_b32 v7, 16, v7 // (op 4002) 
v_mul_f32 v18, v10, v6 // (op 4002) 
v_mul_f32 v19, v10, v7 // (op 4002) 
v_and_b32 v6, 65535, v9 // (op 4002) 
v_lshrrev_b32 v7, 16, v9 // (op 4002) 
v_lshlrev_b32 v6, 16, v6 // (op 4002) 
v_lshlrev_b32 v7, 16, v7 // (op 4002) 
v_mul_f32 v20, v10, v6 // (op 4002) 
v_mul_f32 v21, v10, v7 // (op 4002) 


v_add_f32 v6, v18, v14 // (op 3998) 
v_add_f32 v7, v19, v15 // (op 3998) 
v_add_f32 v8, v20, v16 // (op 3998) 
v_add_f32 v9, v21, v17 // (op 3998) 


v_lshrrev_b32 v14, 16, v6 // (op 3999) 
v_lshrrev_b32 v15, 16, v7 // (op 3999) 
v_lshrrev_b32 v16, 16, v8 // (op 3999) 
v_lshrrev_b32 v17, 16, v9 // (op 3999) 


v_pack_b32_f16 v8, v14, v15 // (op 4000) 
v_pack_b32_f16 v9, v16, v17 // (op 4000) 

s_lshl_b32 s0, s4, 6 // (op 4000) 
s_mul_i32 s1, 7, s0 // (op 4000) 

v_add_u32 v0, v13, s1 // (op 4000) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4000) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.LRR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2, .LRR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_end-RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
  .amdhsa_next_free_vgpr 268
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 84480 // lds bytes
  .amdhsa_accum_offset 204
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 84480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 202
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

