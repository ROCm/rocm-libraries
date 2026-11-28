// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.p2align 8
.type RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2,@function
RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 2743) 
v_lshrrev_b32 v3, 4, v0 // (op 2743) 
v_and_b32 v4, 1, v3 // (op 2743) 

v_and_b32 v3, 15, v1 // (op 2743) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 2743) 


v_lshrrev_b32 v3, 7, v1 // (op 2743) 
v_lshl_add_u32 v4, v3, 5, v5 // (op 2743) 


v_lshrrev_b32 v3, 5, v0 // (op 2743) 

v_lshlrev_b32 v0, 4, v3 // (op 2743) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2743) 


v_lshrrev_b32 v0, 7, v1 // (op 2854) 
v_and_b32 v4, 31, v1 // (op 2854) 
v_lshl_add_u32 v5, v0, 5, v4 // (op 2854) 


v_and_b32 v0, 63, v1 // (op 2854) 
v_lshrrev_b32 v4, 5, v0 // (op 2854) 

v_lshl_add_u32 v0, v5, 2, v4 // (op 2854) 


v_and_b32 v4, 63, v1 // (op 2969) 
v_lshrrev_b32 v5, 4, v4 // (op 2969) 
v_and_b32 v6, 1, v5 // (op 2969) 

v_and_b32 v5, 15, v1 // (op 2969) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2969) 


v_lshrrev_b32 v5, 7, v1 // (op 2969) 
v_lshl_add_u32 v6, v5, 5, v7 // (op 2969) 


v_lshrrev_b32 v5, 5, v4 // (op 2969) 

v_lshlrev_b32 v4, 4, v5 // (op 2969) 

v_lshl_add_u32 v5, v6, 7, v4 // (op 2969) 


v_lshrrev_b32 v4, 7, v1 // (op 3022) 
v_and_b32 v6, 31, v1 // (op 3022) 
v_lshl_add_u32 v7, v4, 5, v6 // (op 3022) 


v_and_b32 v4, 63, v1 // (op 3022) 
v_lshrrev_b32 v6, 5, v4 // (op 3022) 

v_lshl_add_u32 v4, v7, 2, v6 // (op 3022) 


v_lshrrev_b32 v6, 7, v1 // (op 2687) 
v_and_b32 v7, 63, v1 // (op 2687) 
v_lshrrev_b32 v8, 5, v7 // (op 2687) 
v_lshlrev_b32 v9, 2, v8 // (op 2687) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2687) 


v_lshl_add_u32 v6, s16, 6, v8 // (op 2687) 

v_lshrrev_b32 v8, 2, v7 // (op 2687) 

v_and_b32 v7, 7, v8 // (op 2687) 

v_and_b32 v8, 3, v1 // (op 2687) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 2687) 


v_lshrrev_b32 v7, 6, v1 // (op 2687) 
v_and_b32 v8, 1, v7 // (op 2687) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 2687) 


v_lshl_add_u32 v8, s17, 8, v7 // (op 2687) 

v_mul_lo_u32 v7, v8, s2 // (op 2687) 

v_add_lshl_u32 v8, v6, v7, 1 // (op 2687) 


v_and_b32 v6, 63, v1 // (op 2801) 
v_and_b32 v7, 48, v1 // (op 2801) 
v_sub_u32 v9, v6, v7 // (op 2801) 

v_lshrrev_b32 v7, 5, v6 // (op 2801) 
s_mul_i32 s1, s2, s9 // (op 2689) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2689) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2689) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2689) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2689) most significant: sum
s_add_u32 s1, s1, s19 // (op 2689) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2689) 

v_lshl_add_u32 v10, v7, 5, v9 // (op 2801) 


v_lshrrev_b32 v7, 4, v6 // (op 2801) 

v_and_b32 v6, -2, v7 // (op 2801) 
v_sub_u32 v9, v7, v6 // (op 2801) 


v_lshlrev_b32 v6, 4, v9 // (op 2801) 

v_lshrrev_b32 v7, 6, v1 // (op 2801) 
v_and_b32 v9, 1, v7 // (op 2801) 

v_lshl_add_u32 v7, v9, 5, v6 // (op 2801) 


v_lshl_add_u32 v6, v10, 8, v7 // (op 2801) 


v_mul_lo_u32 v7, v6, 6 // (op 2801) 
v_lshrrev_b32 v9, 3, v7 // (op 2801) 

v_lshrrev_b32 v7, 4, v6 // (op 2801) 

v_lshlrev_b32 v6, 2, v7 // (op 2801) 

v_add_u32 v7, v9, v6 // (op 2801) 


v_lshrrev_b32 v6, 6, v1 // (op 2912) 
v_and_b32 v9, 1, v6 // (op 2912) 

v_and_b32 v6, 31, v1 // (op 2912) 
v_lshl_add_u32 v10, v9, 5, v6 // (op 2912) 


v_and_b32 v6, 63, v1 // (op 2912) 
v_lshrrev_b32 v9, 5, v6 // (op 2912) 

v_lshl_add_u32 v6, v9, 8, v10 // (op 2912) 


v_lshrrev_b32 v9, 7, v1 // (op 2928) 
v_and_b32 v10, 63, v1 // (op 2928) 
v_lshrrev_b32 v11, 5, v10 // (op 2928) 
v_lshlrev_b32 v12, 2, v11 // (op 2928) 

v_lshl_add_u32 v11, v9, 5, v12 // (op 2928) 


v_lshl_add_u32 v9, s16, 6, v11 // (op 2928) 

v_lshrrev_b32 v11, 2, v10 // (op 2928) 

v_and_b32 v10, 7, v11 // (op 2928) 

v_and_b32 v11, 3, v1 // (op 2928) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 2928) 


v_lshrrev_b32 v10, 6, v1 // (op 2928) 
v_and_b32 v11, 1, v10 // (op 2928) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 2928) 


v_lshl_add_u32 v11, s17, 8, v10 // (op 2928) 

v_mul_lo_u32 v10, v11, s4 // (op 2928) 

v_add_lshl_u32 v11, v9, v10, 1 // (op 2928) 


v_and_b32 v9, 63, v1 // (op 2998) 
v_and_b32 v10, 48, v1 // (op 2998) 
v_sub_u32 v12, v9, v10 // (op 2998) 

v_lshrrev_b32 v10, 5, v9 // (op 2998) 
s_mul_i32 s1, s4, s9 // (op 2930) most significant: low of low * high
v_lshl_add_u32 v13, v10, 5, v12 // (op 2998) 


v_lshrrev_b32 v10, 4, v9 // (op 2998) 

v_and_b32 v9, -2, v10 // (op 2998) 
v_sub_u32 v12, v10, v9 // (op 2998) 


v_lshlrev_b32 v9, 4, v12 // (op 2998) 

v_lshrrev_b32 v10, 6, v1 // (op 2998) 
v_and_b32 v12, 1, v10 // (op 2998) 

v_lshl_add_u32 v10, v12, 5, v9 // (op 2998) 


v_lshl_add_u32 v9, v13, 8, v10 // (op 2998) 


v_mul_lo_u32 v10, v9, 6 // (op 2998) 
v_lshrrev_b32 v12, 3, v10 // (op 2998) 

v_lshrrev_b32 v10, 4, v9 // (op 2998) 

v_lshlrev_b32 v9, 2, v10 // (op 2998) 

v_add_u32 v10, v12, v9 // (op 2998) 


v_lshrrev_b32 v9, 6, v1 // (op 3051) 
v_and_b32 v12, 1, v9 // (op 3051) 

v_and_b32 v9, 31, v1 // (op 3051) 
v_lshl_add_u32 v13, v12, 5, v9 // (op 3051) 


v_and_b32 v9, 63, v1 // (op 3051) 
v_lshrrev_b32 v12, 5, v9 // (op 3051) 

v_lshl_add_u32 v9, v12, 8, v13 // (op 3051) 


s_mul_i32 s40, s5, s8 // (op 2930) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2930) least significant
s_mul_hi_u32 s41, s4, s8 // (op 2930) most significant: high of low * low
s_add_u32 s1, s1, s40 // (op 2930) most significant: sum
s_add_u32 s1, s1, s41 // (op 2930) most significant: sum


s_lshl_b64 s[40:41], s[0:1], 1 // (op 2930) 

s_mov_b32 s44, s6 // (op 2689) reg expression
s_mov_b32 s45, s7 // (op 2689) reg expression
s_mov_b32 s46, s18 // (op 2689) reg expression
s_mov_b32 s47, 131072 // (op 2689) call()

s_mov_b32 s48, s10 // (op 2930) reg expression

s_mov_b32 s49, s11 // (op 2930) reg expression
s_mov_b32 s50, s40 // (op 2930) reg expression
s_mov_b32 s51, 131072 // (op 2930) call()



v_mov_b32 v12, s14 // (op 2) Move value
v_mov_b32 v13, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 65) call()
v_accvgpr_write a5, 0.00000 // (op 65) call()
v_accvgpr_write a6, 0.00000 // (op 65) call()
v_accvgpr_write a7, 0.00000 // (op 65) call()
v_accvgpr_write a8, 0.00000 // (op 65) call()
v_accvgpr_write a9, 0.00000 // (op 65) call()
v_accvgpr_write a10, 0.00000 // (op 65) call()
v_accvgpr_write a11, 0.00000 // (op 65) call()
v_accvgpr_write a12, 0.00000 // (op 65) call()
v_accvgpr_write a13, 0.00000 // (op 65) call()
v_accvgpr_write a14, 0.00000 // (op 65) call()
v_accvgpr_write a15, 0.00000 // (op 65) call()
v_accvgpr_write a16, 0.00000 // (op 402) call()
v_accvgpr_write a17, 0.00000 // (op 402) call()
v_accvgpr_write a18, 0.00000 // (op 402) call()
v_accvgpr_write a19, 0.00000 // (op 402) call()
v_accvgpr_write a20, 0.00000 // (op 402) call()
v_accvgpr_write a21, 0.00000 // (op 402) call()
v_accvgpr_write a22, 0.00000 // (op 402) call()
v_accvgpr_write a23, 0.00000 // (op 402) call()
v_accvgpr_write a24, 0.00000 // (op 402) call()
v_accvgpr_write a25, 0.00000 // (op 402) call()
v_accvgpr_write a26, 0.00000 // (op 402) call()
v_accvgpr_write a27, 0.00000 // (op 402) call()
v_accvgpr_write a28, 0.00000 // (op 402) call()
v_accvgpr_write a29, 0.00000 // (op 402) call()
v_accvgpr_write a30, 0.00000 // (op 402) call()
v_accvgpr_write a31, 0.00000 // (op 402) call()
v_accvgpr_write a32, 0.00000 // (op 636) call()
v_accvgpr_write a33, 0.00000 // (op 636) call()
v_accvgpr_write a34, 0.00000 // (op 636) call()
v_accvgpr_write a35, 0.00000 // (op 636) call()
v_accvgpr_write a36, 0.00000 // (op 636) call()
v_accvgpr_write a37, 0.00000 // (op 636) call()
v_accvgpr_write a38, 0.00000 // (op 636) call()
v_accvgpr_write a39, 0.00000 // (op 636) call()
v_accvgpr_write a40, 0.00000 // (op 636) call()
v_accvgpr_write a41, 0.00000 // (op 636) call()
v_accvgpr_write a42, 0.00000 // (op 636) call()
v_accvgpr_write a43, 0.00000 // (op 636) call()
v_accvgpr_write a44, 0.00000 // (op 636) call()
v_accvgpr_write a45, 0.00000 // (op 636) call()
v_accvgpr_write a46, 0.00000 // (op 636) call()
v_accvgpr_write a47, 0.00000 // (op 636) call()
v_accvgpr_write a48, 0.00000 // (op 870) call()
v_accvgpr_write a49, 0.00000 // (op 870) call()
v_accvgpr_write a50, 0.00000 // (op 870) call()
v_accvgpr_write a51, 0.00000 // (op 870) call()
v_accvgpr_write a52, 0.00000 // (op 870) call()
v_accvgpr_write a53, 0.00000 // (op 870) call()
v_accvgpr_write a54, 0.00000 // (op 870) call()
v_accvgpr_write a55, 0.00000 // (op 870) call()
v_accvgpr_write a56, 0.00000 // (op 870) call()
v_accvgpr_write a57, 0.00000 // (op 870) call()
v_accvgpr_write a58, 0.00000 // (op 870) call()
v_accvgpr_write a59, 0.00000 // (op 870) call()
v_accvgpr_write a60, 0.00000 // (op 870) call()
v_accvgpr_write a61, 0.00000 // (op 870) call()
v_accvgpr_write a62, 0.00000 // (op 870) call()
v_accvgpr_write a63, 0.00000 // (op 870) call()
v_lshrrev_b32 v14, 3, v1 // (op 2454) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 2454) 

v_and_b32 v14, -8, v1 // (op 2454) 
v_sub_u32 v16, v1, v14 // (op 2454) 

v_lshlrev_b32 v14, 4, v16 // (op 2454) 

v_mov_b32 v17, 0 // (op 2454) convert
v_mov_b32 v16, v14 // (op 2454) convert

v_mul_lo_u32 v19, v15, s13 // (op 2454) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 2454) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 2454) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2454) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2454) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2454) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 2456) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2456) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2456) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2456) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2456) most significant: sum
s_add_u32 s1, s1, s7 // (op 2456) most significant: sum


s_mov_b32 s8, s20 // (op 2456) reg expression
s_mov_b32 s9, s21 // (op 2456) reg expression
s_mov_b32 s10, s0 // (op 2456) reg expression
s_mov_b32 s11, 131072 // (op 2456) call()

v_lshrrev_b32 v14, 3, v1 // (op 2473) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 2473) 

v_and_b32 v14, -8, v1 // (op 2473) 
v_sub_u32 v16, v1, v14 // (op 2473) 

v_lshlrev_b32 v14, 4, v16 // (op 2473) 

v_mov_b32 v17, 0 // (op 2473) convert
v_mov_b32 v16, v14 // (op 2473) convert

v_mul_lo_u32 v21, v15, s13 // (op 2473) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 2473) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 2473) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 2473) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 2473) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 2473) most significant half



s_mul_i32 s1, s12, s23 // (op 2475) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2475) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2475) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2475) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2475) most significant: sum
s_add_u32 s1, s1, s7 // (op 2475) most significant: sum


s_mov_b32 s8, s20 // (op 2475) reg expression
s_mov_b32 s9, s21 // (op 2475) reg expression
s_mov_b32 s10, s0 // (op 2475) reg expression
s_mov_b32 s11, 131072 // (op 2475) call()

v_and_b32 v14, -4, v1 // (op 2518) 
v_sub_u32 v15, v1, v14 // (op 2518) 

v_lshlrev_b32 v14, 6, v15 // (op 2518) 

v_lshl_add_u32 v15, s17, 8, v14 // (op 2518) 

v_mov_b32 v17, 0 // (op 2518) convert
v_mov_b32 v16, v15 // (op 2518) convert

v_lshrrev_b32 v14, 2, v1 // (op 2518) 
v_lshlrev_b32 v15, 1, v14 // (op 2518) 

v_mul_lo_u32 v23, v15, s25 // (op 2518) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 2518) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 2518) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 2518) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v16 // (op 2518) least significant half
v_addc_co_u32 v23, s[0:1], v23, v17, s[0:1] // (op 2518) most significant half



v_mul_lo_u32 v15, v23, 6 // (op 2518) most significant: low of high * low
v_mul_hi_u32 v16, v22, 6 // (op 2518) most significant: high of low * low
v_mul_lo_u32 v14, v22, 6 // (op 2518) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 2518) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 2518) 

v_mov_b32 v14, v16 // (op 2518) convert

s_mul_i32 s1, s29, 6 // (op 2520) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 2520) least significant
s_mul_hi_u32 s6, s28, 6 // (op 2520) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2520) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2520) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 2520) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2520) least significant
s_mul_hi_u32 s15, s24, s0 // (op 2520) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 2520) most significant: sum
s_add_u32 s7, s7, s15 // (op 2520) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2520) 

s_mov_b32 s40, s26 // (op 2520) reg expression
s_mov_b32 s41, s27 // (op 2520) reg expression
s_mov_b32 s42, s0 // (op 2520) reg expression
s_mov_b32 s43, 131072 // (op 2520) call()

v_and_b32 v15, -4, v1 // (op 2537) 
v_sub_u32 v16, v1, v15 // (op 2537) 

v_lshlrev_b32 v15, 6, v16 // (op 2537) 

v_lshl_add_u32 v16, s17, 8, v15 // (op 2537) 

v_mov_b32 v23, 0 // (op 2537) convert
v_mov_b32 v22, v16 // (op 2537) convert

v_lshrrev_b32 v15, 2, v1 // (op 2537) 
v_lshlrev_b32 v16, 1, v15 // (op 2537) 

v_mul_lo_u32 v25, v16, s25 // (op 2537) most significant: low of low * high
v_mul_hi_u32 v15, v16, s24 // (op 2537) most significant: high of low * low
v_mul_lo_u32 v24, v16, s24 // (op 2537) least significant: low of low * low
v_add_u32 v25, v25, v15 // (op 2537) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v22 // (op 2537) least significant half
v_addc_co_u32 v25, s[0:1], v25, v23, s[0:1] // (op 2537) most significant half



v_mul_lo_u32 v17, v25, 6 // (op 2537) most significant: low of high * low
v_mul_hi_u32 v15, v24, 6 // (op 2537) most significant: high of low * low
v_mul_lo_u32 v16, v24, 6 // (op 2537) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 2537) most significant: sum


v_lshrrev_b64 v[22:23], 3, v[16:17] // (op 2537) 

v_mov_b32 v15, v22 // (op 2537) convert

s_mul_i32 s1, s29, 6 // (op 2539) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 2539) least significant
s_mul_hi_u32 s6, s28, 6 // (op 2539) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2539) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2539) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 2539) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2539) least significant
s_mul_hi_u32 s15, s24, s0 // (op 2539) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 2539) most significant: sum
s_add_u32 s7, s7, s15 // (op 2539) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2539) 

s_mov_b32 s40, s26 // (op 2539) reg expression
s_mov_b32 s41, s27 // (op 2539) reg expression
s_mov_b32 s42, s0 // (op 2539) reg expression
s_mov_b32 s43, 131072 // (op 2539) call()

v_lshrrev_b32 v16, 2, v1 // (op 2582) 
v_lshl_add_u32 v17, s16, 6, v16 // (op 2582) 

v_and_b32 v16, -4, v1 // (op 2582) 
v_sub_u32 v22, v1, v16 // (op 2582) 

v_mul_lo_u32 v16, v17, s30 // (op 2582) 
v_add_u32 v16, v16, v22 // (op 2582) 


s_mul_i32 s1, s30, s23 // (op 2584) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 2584) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 2584) least significant
s_mul_hi_u32 s7, s30, s22 // (op 2584) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2584) most significant: sum
s_add_u32 s1, s1, s7 // (op 2584) most significant: sum


s_mov_b32 s52, s32 // (op 2584) reg expression
s_mov_b32 s53, s33 // (op 2584) reg expression
s_mov_b32 s54, s0 // (op 2584) reg expression
s_mov_b32 s55, 131072 // (op 2584) call()

v_lshrrev_b32 v17, 2, v1 // (op 2601) 
v_lshl_add_u32 v22, s16, 6, v17 // (op 2601) 

v_and_b32 v17, -4, v1 // (op 2601) 
v_sub_u32 v23, v1, v17 // (op 2601) 

v_mul_lo_u32 v17, v22, s30 // (op 2601) 
v_add_u32 v17, v17, v23 // (op 2601) 


s_mul_i32 s1, s30, s23 // (op 2603) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 2603) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 2603) least significant
s_mul_hi_u32 s7, s30, s22 // (op 2603) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2603) most significant: sum
s_add_u32 s1, s1, s7 // (op 2603) most significant: sum


s_mov_b32 s52, s32 // (op 2603) reg expression
s_mov_b32 s53, s33 // (op 2603) reg expression
s_mov_b32 s54, s0 // (op 2603) reg expression
s_mov_b32 s55, 131072 // (op 2603) call()

v_and_b32 v22, -64, v1 // (op 2646) 
v_sub_u32 v23, v1, v22 // (op 2646) 

v_lshlrev_b32 v22, 2, v23 // (op 2646) 

v_lshl_add_u32 v23, s17, 8, v22 // (op 2646) 

v_lshrrev_b32 v22, 6, v1 // (op 2646) 
v_mul_lo_u32 v24, v22, s34 // (op 2646) 
v_add_u32 v24, v24, v23 // (op 2646) 


s_mul_i32 s1, s34, s39 // (op 2648) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 2648) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2648) least significant
s_mul_hi_u32 s7, s34, s38 // (op 2648) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2648) most significant: sum
s_add_u32 s1, s1, s7 // (op 2648) most significant: sum


s_mov_b32 s56, s36 // (op 2648) reg expression
s_mov_b32 s57, s37 // (op 2648) reg expression
s_mov_b32 s58, s0 // (op 2648) reg expression
s_mov_b32 s59, 131072 // (op 2648) call()

v_and_b32 v22, -64, v1 // (op 2665) 
v_sub_u32 v23, v1, v22 // (op 2665) 

v_lshlrev_b32 v22, 2, v23 // (op 2665) 

v_lshl_add_u32 v23, s17, 8, v22 // (op 2665) 

v_lshrrev_b32 v22, 6, v1 // (op 2665) 
v_mul_lo_u32 v25, v22, s34 // (op 2665) 
v_add_u32 v25, v25, v23 // (op 2665) 


s_mul_i32 s1, s34, s39 // (op 2667) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 2667) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2667) least significant
s_mul_hi_u32 s7, s34, s38 // (op 2667) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2667) most significant: sum
s_add_u32 s1, s1, s7 // (op 2667) most significant: sum


s_mov_b32 s56, s36 // (op 2667) reg expression
s_mov_b32 s57, s37 // (op 2667) reg expression
s_mov_b32 s58, s0 // (op 2667) reg expression
s_mov_b32 s59, 131072 // (op 2667) call()


s_mul_i32 s1, s25, 6 // (op 2169) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 2169) least significant
s_mul_hi_u32 s6, s24, 6 // (op 2169) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2169) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 2169) 


v_mov_b32 v26, v14 // (op 2169) reg expression
buffer_load_dwordx3 v[28:30], v26, s[40:43], 0 offen // (op 2169) Load value
buffer_load_dwordx3 v[32:34], v26, s[40:43], 0 offen offset:12 // (op 2169) Load value
buffer_load_dwordx3 v[36:38], v26, s[40:43], 0 offen offset:24 // (op 2169) Load value
buffer_load_dwordx3 v[40:42], v26, s[40:43], 0 offen offset:36 // (op 2169) Load value
v_add_u32 v26, v26, v22 // (op 2169) 
buffer_load_dwordx3 v[44:46], v26, s[40:43], 0 offen // (op 2169) Load value
buffer_load_dwordx3 v[48:50], v26, s[40:43], 0 offen offset:12 // (op 2169) Load value
buffer_load_dwordx3 v[52:54], v26, s[40:43], 0 offen offset:24 // (op 2169) Load value
buffer_load_dwordx3 v[56:58], v26, s[40:43], 0 offen offset:36 // (op 2169) Load value




v_mov_b32 v31, v24 // (op 2172) reg expression

buffer_load_dword v35, v31, s[56:59], 0 offen // (op 2172) Load value

v_and_b32 v31, -8, v1 // (op 2731) 
v_sub_u32 v39, v1, v31 // (op 2731) 

v_lshlrev_b32 v31, 4, v39 // (op 2731) 

v_lshrrev_b32 v39, 3, v1 // (op 2731) 
v_lshl_add_u32 v43, v39, 7, v31 // (op 2731) 


s_nop 0
v_readfirstlane_b32 s0, v43 // (op 2731) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 2407) 
v_mov_b64 v[22:23], s[6:7] // (op 2407) convert


v_mov_b64 v[26:27], v[18:19] // (op 2407) 
s_mov_b32 m0, s0 // (op 2407) reg expression
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 2407) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 2407) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 2407) most significant half

s_add_u32 m0, m0, 4096 // (op 2407) 
buffer_load_dwordx4 v26, s[8:11], 0 offen offset: 0 lds // (op 2407) Load value direct to lds






buffer_load_ubyte v31, v16, s[52:55], 0 offen offset: 0 // (op 2178) Load value

s_mul_i32 s1, s25, 6 // (op 2181) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 2181) least significant
s_mul_hi_u32 s6, s24, 6 // (op 2181) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2181) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 2181) 


s_lshl_b64 s[0:1], s[24:25], 7 // (op 2181) 
s_mul_i32 s7, s1, 6 // (op 2181) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 2181) least significant
s_mul_hi_u32 s14, s0, 6 // (op 2181) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 2181) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 2181) 

v_add_u32 v39, v14, s0 // (op 2181) 

buffer_load_dwordx3 v[60:62], v39, s[40:43], 0 offen // (op 2181) Load value
buffer_load_dwordx3 v[64:66], v39, s[40:43], 0 offen offset:12 // (op 2181) Load value
buffer_load_dwordx3 v[68:70], v39, s[40:43], 0 offen offset:24 // (op 2181) Load value
buffer_load_dwordx3 v[72:74], v39, s[40:43], 0 offen offset:36 // (op 2181) Load value
v_add_u32 v39, v39, v22 // (op 2181) 
buffer_load_dwordx3 v[76:78], v39, s[40:43], 0 offen // (op 2181) Load value
buffer_load_dwordx3 v[80:82], v39, s[40:43], 0 offen offset:12 // (op 2181) Load value
buffer_load_dwordx3 v[84:86], v39, s[40:43], 0 offen offset:24 // (op 2181) Load value
buffer_load_dwordx3 v[88:90], v39, s[40:43], 0 offen offset:36 // (op 2181) Load value





s_lshl_b32 s0, s34, 2 // (op 2184) 
v_add_u32 v14, v24, s0 // (op 2184) 


buffer_load_dword v39, v14, s[56:59], 0 offen // (op 2184) Load value


v_and_b32 v14, -8, v1 // (op 2957) 
v_sub_u32 v43, v1, v14 // (op 2957) 

v_lshlrev_b32 v14, 4, v43 // (op 2957) 

v_lshrrev_b32 v43, 3, v1 // (op 2957) 
v_lshl_add_u32 v47, v43, 7, v14 // (op 2957) 


s_nop 0
v_readfirstlane_b32 s0, v47 // (op 2957) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 2412) 
v_mov_b64 v[26:27], s[6:7] // (op 2412) convert


v_mov_b64 v[22:23], v[18:19] // (op 2412) 
s_add_u32 s1, 8192, s0 // (op 2412) 
s_mov_b32 m0, s1 // (op 2412) reg expression
s_mov_b32 s6, 128 // (op 2412) call()
buffer_load_dwordx4 v22, s[8:11], s6 offen offset: 0 lds // (op 2412) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v26 // (op 2412) least significant half
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 2412) most significant half

s_add_u32 m0, m0, 4096 // (op 2412) 
buffer_load_dwordx4 v22, s[8:11], s6 offen offset: 0 lds // (op 2412) Load value direct to lds









buffer_load_ubyte v14, v16, s[52:55], 0 offen offset:4 // (op 2190) Load value

v_and_b32 v16, -4, v1 // (op 2784) 
v_sub_u32 v43, v1, v16 // (op 2784) 

v_lshlrev_b32 v16, 6, v43 // (op 2784) 

v_lshrrev_b32 v43, 2, v1 // (op 2784) 
v_lshl_add_u32 v47, v43, 9, v16 // (op 2784) 


v_mul_lo_u32 v16, v47, 6 // (op 2784) 
v_lshrrev_b32 v43, 3, v16 // (op 2784) 

v_lshrrev_b32 v16, 4, v47 // (op 2784) 

v_lshlrev_b32 v47, 2, v16 // (op 2784) 

v_add_u32 v16, v43, v47 // (op 2784) 





s_waitcnt vmcnt(23)
ds_write_b96 v16, v[28:30] offset:16384 // (op 2192) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v16, v[32:34] offset:16400 // (op 2192) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v16, v[36:38] offset:16416 // (op 2192) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v16, v[40:42] offset:16432 // (op 2192) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v16, v[44:46] offset:16640 // (op 2192) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v16, v[48:50] offset:16656 // (op 2192) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v16, v[52:54] offset:16672 // (op 2192) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v16, v[56:58] offset:16688 // (op 2192) Store local data 

v_and_b32 v16, -64, v1 // (op 2895) 
v_sub_u32 v43, v1, v16 // (op 2895) 

v_lshlrev_b32 v16, 2, v43 // (op 2895) 

v_lshrrev_b32 v43, 6, v1 // (op 2895) 
v_lshl_add_u32 v47, v43, 8, v16 // (op 2895) 





s_waitcnt vmcnt(15)
ds_write_b32 v47, v35 offset:49152 // (op 2193) Store local data 

v_and_b32 v16, -4, v1 // (op 2842) 
v_sub_u32 v43, v1, v16 // (op 2842) 

v_and_b32 v16, 4294967292, v1 // (op 2842) 
v_add_u32 v47, v16, v43 // (op 2842) 





s_waitcnt vmcnt(12)
ds_write_b8 v47, v31 offset:50176 // (op 2195) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 3133) 



ds_read_b96_tr_b6 v[92:94], v7 offset:16384 // (op 2213) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v7 offset:20480 // (op 2213) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3174) 


ds_read_u8 v16, v6 offset:49152 // (op 2236) Load local data 



ds_read_b96_tr_b6 v[98:100], v7 offset:16448 // (op 2218) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v7 offset:20544 // (op 2218) Transpose load local data 


ds_read_u8 v43, v6 offset:49216 // (op 2241) Load local data 



ds_read_b96_tr_b6 v[104:106], v7 offset:16512 // (op 2223) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v7 offset:20608 // (op 2223) Transpose load local data 


ds_read_u8 v47, v6 offset:49280 // (op 2246) Load local data 



ds_read_b96_tr_b6 v[110:112], v7 offset:16576 // (op 2228) Transpose load local data 
ds_read_b96_tr_b6 v[113:115], v7 offset:20672 // (op 2228) Transpose load local data 


ds_read_u8 v51, v6 offset:49344 // (op 2251) Load local data 
s_barrier  // (op 3114) 



ds_read_b128 v[116:119], v3 // (op 2209) Load local data 
ds_read_b128 v[120:123], v3 offset:32 // (op 2209) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3154) 



ds_read_u8 v55, v0 offset:50176 // (op 2232) Load local data 
v_and_b32 v59, -8, v1 // (op 2703) 
v_sub_u32 v63, v1, v59 // (op 2703) 

v_lshlrev_b32 v59, 4, v63 // (op 2703) 

v_lshrrev_b32 v63, 3, v1 // (op 2703) 
v_lshl_add_u32 v67, v63, 7, v59 // (op 2703) 


s_nop 0
v_readfirstlane_b32 s0, v67 // (op 2703) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v59, -4, v1 // (op 2755) 
v_sub_u32 v63, v1, v59 // (op 2755) 

v_lshlrev_b32 v59, 6, v63 // (op 2755) 

v_lshrrev_b32 v63, 2, v1 // (op 2755) 
v_lshl_add_u32 v67, v63, 9, v59 // (op 2755) 


v_mul_lo_u32 v59, v67, 6 // (op 2755) 
v_lshrrev_b32 v63, 3, v59 // (op 2755) 

v_lshrrev_b32 v59, 4, v67 // (op 2755) 

v_lshlrev_b32 v67, 2, v59 // (op 2755) 

v_add_u32 v59, v63, v67 // (op 2755) 


v_and_b32 v63, -4, v1 // (op 2813) 
v_sub_u32 v67, v1, v63 // (op 2813) 

v_and_b32 v63, 4294967292, v1 // (op 2813) 
v_add_u32 v71, v63, v67 // (op 2813) 


v_and_b32 v63, -64, v1 // (op 2866) 
v_sub_u32 v67, v1, v63 // (op 2866) 

v_lshlrev_b32 v63, 2, v67 // (op 2866) 

v_lshrrev_b32 v67, 6, v1 // (op 2866) 
v_lshl_add_u32 v75, v67, 8, v63 // (op 2866) 


v_and_b32 v63, -8, v1 // (op 2942) 
v_sub_u32 v67, v1, v63 // (op 2942) 

v_lshlrev_b32 v63, 4, v67 // (op 2942) 

v_lshrrev_b32 v67, 3, v1 // (op 2942) 
v_lshl_add_u32 v79, v67, 7, v63 // (op 2942) 


s_nop 0
v_readfirstlane_b32 s1, v79 // (op 2942) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v63, -4, v1 // (op 2981) 
v_sub_u32 v67, v1, v63 // (op 2981) 

v_lshlrev_b32 v63, 6, v67 // (op 2981) 

v_lshrrev_b32 v67, 2, v1 // (op 2981) 
v_lshl_add_u32 v79, v67, 9, v63 // (op 2981) 


v_mul_lo_u32 v63, v79, 6 // (op 2981) 
v_lshrrev_b32 v67, 3, v63 // (op 2981) 

v_lshrrev_b32 v63, 4, v79 // (op 2981) 

v_lshlrev_b32 v79, 2, v63 // (op 2981) 

v_add_u32 v63, v67, v79 // (op 2981) 


v_and_b32 v67, -4, v1 // (op 3010) 
v_sub_u32 v79, v1, v67 // (op 3010) 

v_and_b32 v67, 4294967292, v1 // (op 3010) 
v_add_u32 v83, v67, v79 // (op 3010) 


v_and_b32 v67, -64, v1 // (op 3034) 
v_sub_u32 v79, v1, v67 // (op 3034) 

v_lshlrev_b32 v67, 2, v79 // (op 3034) 

v_lshrrev_b32 v79, 6, v1 // (op 3034) 
v_lshl_add_u32 v87, v79, 8, v67 // (op 3034) 


s_mov_b32 s6, 0 // (op 872) call()
s_ashr_i64 s[14:15], s[28:29], 7 // (op 871) 
s_mov_b32 s7, s14 // (op 871) convert

s_and_b32 s14, s7, 4294967294 // (op 871) 

s_cmp_lt_u32 s6, s14 // (op 871) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_871 // (op 871) Condition: Top (jump to RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_871 if false)
RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_871:

s_add_u32 s6, s6, 2 // (op 873) 
s_mul_i32 s15, s25, 6 // (op 10) most significant: low of high * low
s_mul_i32 s14, s24, 6 // (op 10) least significant
s_mul_hi_u32 s7, s24, 6 // (op 10) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 10) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[14:15] // (op 10) 


v_mov_b32 v67, v15 // (op 10) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s7, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s7, s14, 1 // (op 10) 

v_add_u32 v79, v67, s7 // (op 10) 

buffer_load_dwordx3 v[28:30], v79, s[40:43], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v79, s[40:43], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v79, s[40:43], 0 offen offset:24 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[116:123], v[92:97], a[0:15], v55, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 90) 
buffer_load_dwordx3 v[40:42], v79, s[40:43], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 3123) 
v_add_u32 v79, v79, v18 // (op 10) 



buffer_load_dwordx3 v[44:46], v79, s[40:43], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[48:50], v79, s[40:43], 0 offen offset:12 // (op 10) Load value
ds_read_b128 v[124:127], v3 offset:64 // (op 93) Load local data 
ds_read_b128 v[128:131], v3 offset:96 // (op 93) Load local data 
buffer_load_dwordx3 v[52:54], v79, s[40:43], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 3143) 
buffer_load_dwordx3 v[56:58], v79, s[40:43], 0 offen offset:36 // (op 10) Load value







ds_read_b96_tr_b6 v[92:94], v7 offset:32768 // (op 97) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v7 offset:36864 // (op 97) Transpose load local data 
v_mov_b32 v79, v25 // (op 25) 
s_barrier  // (op 3164) 
s_lshl_b32 s7, s34, 3 // (op 25) 



ds_read_u8 v91, v0 offset:50178 // (op 101) Load local data 
v_add_u32 v18, v79, s7 // (op 25) 


s_barrier  // (op 3184) 
buffer_load_dword v35, v18, s[56:59], 0 offen // (op 25) Load value



s_waitcnt lgkmcnt(3)
s_barrier  // (op 3117) 
ds_read_u8 v16, v6 offset:49664 // (op 105) Load local data 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 2391) 
v_mov_b64 v[18:19], s[14:15] // (op 2391) convert


v_mov_b64 v[26:27], v[20:21] // (op 2391) 
v_mov_b64 v[22:23], v[26:27] // (op 2391) 
s_mov_b32 m0, s0 // (op 2391) reg expression
s_mov_b32 s7, 256 // (op 2391) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[124:131], v[92:97], a[0:15], v91, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 107) 
buffer_load_dwordx4 v22, s[8:11], s7 offen offset: 0 lds // (op 2391) Load value direct to lds
v_add_co_u32 v22, s[14:15], v22, v18 // (op 2391) least significant half
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[116:123], v[98:103], a[16:31], v55, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 451) 
v_addc_co_u32 v23, s[14:15], 0, v23, s[14:15] // (op 2391) most significant half




s_add_u32 m0, m0, 4096 // (op 2391) 
ds_read_b96_tr_b6 v[92:94], v7 offset:32832 // (op 455) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v7 offset:36928 // (op 455) Transpose load local data 
buffer_load_dwordx4 v22, s[8:11], s7 offen offset: 0 lds // (op 2391) Load value direct to lds







ds_read_u8 v16, v6 offset:49728 // (op 459) Load local data 
v_mov_b32 v18, v17 // (op 17) 

buffer_load_ubyte v31, v18, s[52:55], 0 offen offset:8 // (op 17) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[124:131], v[92:97], a[16:31], v91, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 460) 
s_barrier  // (op 3204) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[116:123], v[104:109], a[32:47], v55, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 685) 
s_waitcnt vmcnt(23)
ds_write_b96 v63, v[60:62] offset:50432 // (op 306) Store local data 



ds_read_b96_tr_b6 v[92:94], v7 offset:32896 // (op 689) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v7 offset:36992 // (op 689) Transpose load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v63, v[64:66] offset:50448 // (op 306) Store local data 


ds_read_u8 v16, v6 offset:49792 // (op 693) Load local data 
s_waitcnt vmcnt(21)
ds_write_b96 v63, v[68:70] offset:50464 // (op 306) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[124:131], v[92:97], a[32:47], v91, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 694) 
s_waitcnt vmcnt(20)
ds_write_b96 v63, v[72:74] offset:50480 // (op 306) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v63, v[76:78] offset:50688 // (op 306) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[116:123], v[110:115], a[48:63], v55, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 919) 
s_waitcnt vmcnt(18)
ds_write_b96 v63, v[80:82] offset:50704 // (op 306) Store local data 



ds_read_b96_tr_b6 v[110:112], v7 offset:32960 // (op 923) Transpose load local data 
ds_read_b96_tr_b6 v[113:115], v7 offset:37056 // (op 923) Transpose load local data 
s_waitcnt vmcnt(17)
ds_write_b96 v63, v[84:86] offset:50720 // (op 306) Store local data 


ds_read_u8 v51, v6 offset:49856 // (op 927) Load local data 
s_waitcnt vmcnt(16)
ds_write_b96 v63, v[88:90] offset:50736 // (op 306) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[124:131], v[110:115], a[48:63], v91, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 928) 


s_barrier  // (op 3216) 



v_add_u32 v19, 83200, v87 // (op 308) 
s_waitcnt vmcnt(15)
ds_write_b32 v19, v39 // (op 308) Store local data 

s_barrier  // (op 3210) 



v_add_u32 v19, 84224, v83 // (op 307) 
s_waitcnt vmcnt(12)
ds_write_b8 v19, v14 // (op 307) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 3200) 



ds_read_b96_tr_b6 v[92:94], v10 offset:50432 // (op 291) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v10 offset:54528 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3213) 


v_add_u32 v19, 83200, v9 // (op 295) 
ds_read_u8 v16, v19 // (op 295) Load local data 




ds_read_b96_tr_b6 v[98:100], v10 offset:50496 // (op 474) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v10 offset:54592 // (op 474) Transpose load local data 


v_add_u32 v19, 83264, v9 // (op 478) 
ds_read_u8 v43, v19 // (op 478) Load local data 




ds_read_b96_tr_b6 v[104:106], v10 offset:50560 // (op 708) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v10 offset:54656 // (op 708) Transpose load local data 


v_add_u32 v19, 83328, v9 // (op 712) 
ds_read_u8 v47, v19 // (op 712) Load local data 




ds_read_b96_tr_b6 v[124:126], v10 offset:50624 // (op 942) Transpose load local data 
ds_read_b96_tr_b6 v[127:129], v10 offset:54720 // (op 942) Transpose load local data 


v_add_u32 v19, 83392, v9 // (op 946) 
ds_read_u8 v91, v19 // (op 946) Load local data 

s_barrier  // (op 3194) 



ds_read_b128 v[116:119], v5 offset:8192 // (op 289) Load local data 
ds_read_b128 v[120:123], v5 offset:8224 // (op 289) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3207) 



v_add_u32 v19, 84224, v4 // (op 293) 
ds_read_u8 v55, v19 // (op 293) Load local data 


s_mul_i32 s15, s25, 6 // (op 309) most significant: low of high * low
s_mul_i32 s14, s24, 6 // (op 309) least significant
s_mul_hi_u32 s7, s24, 6 // (op 309) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 309) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 309) 


s_lshl_b64 s[14:15], s[24:25], 7 // (op 309) 
s_mul_i32 s19, s15, 6 // (op 309) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 309) least significant
s_mul_hi_u32 s7, s14, 6 // (op 309) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 309) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 309) 

s_mul_i32 s7, 3, s14 // (op 309) 

v_add_u32 v19, v67, s7 // (op 309) 

buffer_load_dwordx3 v[60:62], v19, s[40:43], 0 offen // (op 309) Load value
buffer_load_dwordx3 v[64:66], v19, s[40:43], 0 offen offset:12 // (op 309) Load value
buffer_load_dwordx3 v[68:70], v19, s[40:43], 0 offen offset:24 // (op 309) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[116:123], v[92:97], a[0:15], v55, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 296) 
buffer_load_dwordx3 v[72:74], v19, s[40:43], 0 offen offset:36 // (op 309) Load value
v_add_u32 v19, v19, v22 // (op 309) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[116:123], v[98:103], a[16:31], v55, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 461) 
buffer_load_dwordx3 v[76:78], v19, s[40:43], 0 offen // (op 309) Load value
buffer_load_dwordx3 v[80:82], v19, s[40:43], 0 offen offset:12 // (op 309) Load value
ds_read_b128 v[130:133], v5 offset:8256 // (op 298) Load local data 
ds_read_b128 v[134:137], v5 offset:8288 // (op 298) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[116:123], v[104:109], a[32:47], v55, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 695) 



v_add_u32 v24, 66816, v10 // (op 300) 
ds_read_b96_tr_b6 v[138:140], v24 // (op 300) Transpose load local data 

v_add_u32 v24, 70912, v10 // (op 300) 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[116:123], v[124:129], a[48:63], v55, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 929) 
ds_read_b96_tr_b6 v[141:143], v24 // (op 300) Transpose load local data 






v_add_u32 v24, 84226, v4 // (op 302) 
ds_read_u8 v91, v24 // (op 302) Load local data 

buffer_load_dwordx3 v[84:86], v19, s[40:43], 0 offen offset:24 // (op 309) Load value
buffer_load_dwordx3 v[88:90], v19, s[40:43], 0 offen offset:36 // (op 309) Load value





s_barrier  // (op 3259) 


s_lshl_b32 s7, s34, 2 // (op 311) 
s_mul_i32 s14, 3, s7 // (op 311) 

v_add_u32 v19, 83712, v9 // (op 304) 
ds_read_u8 v67, v19 // (op 304) Load local data 

v_add_u32 v19, v79, s14 // (op 311) 





buffer_load_dword v39, v19, s[56:59], 0 offen // (op 311) Load value

v_add_u32 v19, 66880, v10 // (op 465) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 2529) 
s_lshl_b32 s7, s34, 3 // (op 2657) 
v_add_u32 v25, v25, s7 // (op 2657) 

s_mul_i32 s19, s15, 6 // (op 2529) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 2529) least significant
s_mul_hi_u32 s7, s14, 6 // (op 2529) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 2529) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 2529) 

s_lshl_b32 s7, s14, 1 // (op 2529) 

v_add_u32 v15, v15, s7 // (op 2529) 

ds_read_b96_tr_b6 v[124:126], v19 // (op 465) Transpose load local data 

v_add_u32 v19, 70976, v10 // (op 465) 
ds_read_b96_tr_b6 v[127:129], v19 // (op 465) Transpose load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[130:137], v[138:143], a[0:15], v91, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 305) 




s_barrier  // (op 3197) 
v_add_u32 v19, 83776, v9 // (op 469) 

ds_read_u8 v67, v19 // (op 469) Load local data 

s_lshl_b64 s[14:15], s[12:13], 5 // (op 2402) 



v_add_u32 v19, 66944, v10 // (op 699) 
v_mov_b64 v[22:23], s[14:15] // (op 2402) convert


v_mov_b64 v[138:139], v[26:27] // (op 2402) 
s_add_u32 s7, 8192, s1 // (op 2402) 
s_mov_b32 m0, s7 // (op 2402) reg expression
s_mov_b32 s14, 384 // (op 2402) call()
ds_read_b96_tr_b6 v[140:142], v19 // (op 699) Transpose load local data 

v_add_u32 v19, 71040, v10 // (op 699) 
ds_read_b96_tr_b6 v[143:145], v19 // (op 699) Transpose load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[130:137], v[124:129], a[16:31], v91, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 470) 




v_add_u32 v19, 83840, v9 // (op 703) 
ds_read_u8 v24, v19 // (op 703) Load local data 

buffer_load_dwordx4 v138, s[8:11], s14 offen offset: 0 lds // (op 2402) Load value direct to lds



v_add_u32 v19, 67008, v10 // (op 933) 
v_add_co_u32 v138, s[18:19], v138, v22 // (op 2402) least significant half
v_addc_co_u32 v139, s[18:19], 0, v139, s[18:19] // (op 2402) most significant half

s_add_u32 m0, m0, 4096 // (op 2402) 
buffer_load_dwordx4 v138, s[8:11], s14 offen offset: 0 lds // (op 2402) Load value direct to lds




ds_read_b96_tr_b6 v[124:126], v19 // (op 933) Transpose load local data 


v_add_u32 v19, 71104, v10 // (op 933) 



ds_read_b96_tr_b6 v[127:129], v19 // (op 933) Transpose load local data 

v_add_co_u32 v20, vcc, 256, v20 // (op 2465) least significant half
buffer_load_ubyte v14, v18, s[52:55], 0 offen offset:12 // (op 310) Load value



s_barrier  // (op 3317) 
s_barrier  // (op 3136) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 2465) most significant half



v_add_u32 v17, 8, v17 // (op 2593) 
v_add_u32 v67, 83904, v9 // (op 937) 
ds_read_u8 v79, v67 // (op 937) Load local data 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[130:137], v[140:145], a[32:47], v91, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 704) 


s_barrier  // (op 3321) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[130:137], v[124:129], a[48:63], v91, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 938) 
s_waitcnt vmcnt(23)
ds_write_b96 v59, v[28:30] offset:16384 // (op 13) Store local data 




s_waitcnt vmcnt(22)
ds_write_b96 v59, v[32:34] offset:16400 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v59, v[36:38] offset:16416 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v59, v[40:42] offset:16432 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v59, v[44:46] offset:16640 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v59, v[48:50] offset:16656 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v59, v[52:54] offset:16672 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v59, v[56:58] offset:16688 // (op 13) Store local data 
s_barrier  // (op 3177) 



s_waitcnt vmcnt(15)
ds_write_b32 v75, v35 offset:49152 // (op 30) Store local data 
s_barrier  // (op 3157) 



s_waitcnt vmcnt(12)
ds_write_b8 v71, v31 offset:50176 // (op 21) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3146) 



ds_read_b96_tr_b6 v[92:94], v7 offset:16384 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v7 offset:20480 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 3187) 


ds_read_u8 v16, v6 offset:49152 // (op 87) Load local data 



ds_read_b96_tr_b6 v[98:100], v7 offset:16448 // (op 446) Transpose load local data 
ds_read_b96_tr_b6 v[101:103], v7 offset:20544 // (op 446) Transpose load local data 


ds_read_u8 v43, v6 offset:49216 // (op 450) Load local data 



ds_read_b96_tr_b6 v[104:106], v7 offset:16512 // (op 680) Transpose load local data 
ds_read_b96_tr_b6 v[107:109], v7 offset:20608 // (op 680) Transpose load local data 


ds_read_u8 v47, v6 offset:49280 // (op 684) Load local data 



ds_read_b96_tr_b6 v[110:112], v7 offset:16576 // (op 914) Transpose load local data 
ds_read_b96_tr_b6 v[113:115], v7 offset:20672 // (op 914) Transpose load local data 


ds_read_u8 v51, v6 offset:49344 // (op 918) Load local data 
s_barrier  // (op 3126) 



ds_read_b128 v[116:119], v3 // (op 72) Load local data 
ds_read_b128 v[120:123], v3 offset:32 // (op 72) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3167) 



ds_read_u8 v55, v0 offset:50176 // (op 82) Load local data 
s_ashr_i64 s[14:15], s[28:29], 7 // (op 871) 
s_mov_b32 s7, s14 // (op 871) convert

s_and_b32 s14, s7, 4294967294 // (op 871) 

s_cmp_lt_u32 s6, s14 // (op 871) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_871 // (op 871) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoop_871 if true)
RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoop_871:
















s_barrier  // (op 3299) 
v_lshrrev_b32 v15, 3, v1 // (op 2425) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 2425) 
v_lshl_add_u32 v63, s16, 6, v15 // (op 2425) 

s_lshr_b32 s6, s0, 1 // (op 2425) 

v_and_b32 v15, -8, v1 // (op 2425) 
v_sub_u32 v67, v1, v15 // (op 2425) 

v_lshlrev_b32 v15, 4, v67 // (op 2425) 

v_lshl_add_u32 v67, s6, 8, v15 // (op 2425) 


v_mov_b32 v41, 0 // (op 2425) convert
v_mov_b32 v40, v67 // (op 2425) convert

v_mul_lo_u32 v45, v63, s13 // (op 2425) most significant: low of low * high
v_mul_hi_u32 v15, v63, s12 // (op 2425) most significant: high of low * low
v_mul_lo_u32 v44, v63, s12 // (op 2425) least significant: low of low * low
v_add_u32 v45, v45, v15 // (op 2425) most significant: sum

v_add_co_u32 v44, s[0:1], v44, v40 // (op 2425) least significant half
v_addc_co_u32 v45, s[0:1], v45, v41, s[0:1] // (op 2425) most significant half



s_mul_i32 s1, s12, s23 // (op 2427) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2427) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2427) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2427) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2427) most significant: sum
s_add_u32 s1, s1, s7 // (op 2427) most significant: sum


s_mov_b32 s8, s20 // (op 2427) reg expression
s_mov_b32 s9, s21 // (op 2427) reg expression
s_mov_b32 s10, s0 // (op 2427) reg expression
s_mov_b32 s11, 131072 // (op 2427) call()

v_lshrrev_b32 v15, 2, v1 // (op 2495) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 2495) 
v_lshlrev_b32 v46, 1, v15 // (op 2495) 

s_lshr_b32 s6, s0, 1 // (op 2495) 

v_and_b32 v15, -4, v1 // (op 2495) 
v_lshl_add_u32 v63, s6, 8, v46 // (op 2495) 


v_sub_u32 v46, v1, v15 // (op 2495) 

v_lshlrev_b32 v15, 6, v46 // (op 2495) 

v_lshl_add_u32 v46, s17, 8, v15 // (op 2495) 

v_mov_b32 v41, 0 // (op 2495) convert
v_mov_b32 v40, v46 // (op 2495) convert

v_mul_lo_u32 v49, v63, s25 // (op 2495) most significant: low of low * high
v_mul_hi_u32 v15, v63, s24 // (op 2495) most significant: high of low * low
v_mul_lo_u32 v48, v63, s24 // (op 2495) least significant: low of low * low
v_add_u32 v49, v49, v15 // (op 2495) most significant: sum

v_add_co_u32 v48, s[0:1], v48, v40 // (op 2495) least significant half
v_addc_co_u32 v49, s[0:1], v49, v41, s[0:1] // (op 2495) most significant half



v_mul_lo_u32 v41, v49, 6 // (op 2495) most significant: low of high * low
v_mul_hi_u32 v15, v48, 6 // (op 2495) most significant: high of low * low
v_mul_lo_u32 v40, v48, 6 // (op 2495) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 2495) most significant: sum


v_lshrrev_b64 v[48:49], 3, v[40:41] // (op 2495) 

v_mov_b32 v15, v48 // (op 2495) convert

s_mul_i32 s1, s29, 6 // (op 2497) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 2497) least significant
s_mul_hi_u32 s6, s28, 6 // (op 2497) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2497) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2497) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 2497) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2497) least significant
s_mul_hi_u32 s15, s24, s0 // (op 2497) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 2497) most significant: sum
s_add_u32 s7, s7, s15 // (op 2497) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2497) 

s_mov_b32 s40, s26 // (op 2497) reg expression
s_mov_b32 s41, s27 // (op 2497) reg expression
s_mov_b32 s42, s0 // (op 2497) reg expression
s_mov_b32 s43, 131072 // (op 2497) call()

v_lshrrev_b32 v46, 2, v1 // (op 2559) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 2559) 
v_lshl_add_u32 v63, s16, 6, v46 // (op 2559) 

s_lshr_b32 s6, s0, 1 // (op 2559) 

v_and_b32 v46, -4, v1 // (op 2559) 
v_sub_u32 v67, v1, v46 // (op 2559) 

v_lshl_add_u32 v46, s6, 3, v67 // (op 2559) 


v_mul_lo_u32 v67, v63, s30 // (op 2559) 
v_add_u32 v67, v67, v46 // (op 2559) 


s_mul_i32 s1, s30, s23 // (op 2561) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 2561) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 2561) least significant
s_mul_hi_u32 s7, s30, s22 // (op 2561) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2561) most significant: sum
s_add_u32 s1, s1, s7 // (op 2561) most significant: sum


s_mov_b32 s52, s32 // (op 2561) reg expression
s_mov_b32 s53, s33 // (op 2561) reg expression
s_mov_b32 s54, s0 // (op 2561) reg expression
s_mov_b32 s55, 131072 // (op 2561) call()

v_lshrrev_b32 v46, 6, v1 // (op 2623) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 2623) 
v_and_b32 v63, -64, v1 // (op 2623) 
s_lshr_b32 s6, s0, 1 // (op 2623) 

v_sub_u32 v71, v1, v63 // (op 2623) 

v_lshl_add_u32 v63, s6, 3, v46 // (op 2623) 


v_lshlrev_b32 v46, 2, v71 // (op 2623) 

v_lshl_add_u32 v71, s17, 8, v46 // (op 2623) 

v_mul_lo_u32 v46, v63, s34 // (op 2623) 
v_add_u32 v46, v46, v71 // (op 2623) 


s_mul_i32 s1, s34, s39 // (op 2625) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 2625) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 2625) least significant
s_mul_hi_u32 s7, s34, s38 // (op 2625) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2625) most significant: sum
s_add_u32 s1, s1, s7 // (op 2625) most significant: sum


s_mov_b32 s56, s36 // (op 2625) reg expression
s_mov_b32 s57, s37 // (op 2625) reg expression
s_mov_b32 s58, s0 // (op 2625) reg expression
s_mov_b32 s59, 131072 // (op 2625) call()

v_and_b32 v63, -8, v1 // (op 2716) 
v_sub_u32 v71, v1, v63 // (op 2716) 

v_lshlrev_b32 v63, 4, v71 // (op 2716) 

v_lshrrev_b32 v71, 3, v1 // (op 2716) 
v_lshl_add_u32 v75, v71, 7, v63 // (op 2716) 


s_nop 0
v_readfirstlane_b32 s0, v75 // (op 2716) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v63, -4, v1 // (op 2768) 
v_sub_u32 v71, v1, v63 // (op 2768) 

v_lshlrev_b32 v63, 6, v71 // (op 2768) 

v_lshrrev_b32 v71, 2, v1 // (op 2768) 
v_lshl_add_u32 v75, v71, 9, v63 // (op 2768) 


v_mul_lo_u32 v63, v75, 6 // (op 2768) 
v_lshrrev_b32 v71, 3, v63 // (op 2768) 

v_lshrrev_b32 v63, 4, v75 // (op 2768) 

v_lshlrev_b32 v75, 2, v63 // (op 2768) 

v_add_u32 v63, v71, v75 // (op 2768) 


v_and_b32 v71, -4, v1 // (op 2826) 
v_sub_u32 v75, v1, v71 // (op 2826) 

v_and_b32 v71, 4294967292, v1 // (op 2826) 
v_add_u32 v79, v71, v75 // (op 2826) 


v_and_b32 v71, -64, v1 // (op 2879) 
v_sub_u32 v75, v1, v71 // (op 2879) 

v_lshlrev_b32 v71, 2, v75 // (op 2879) 

v_lshrrev_b32 v75, 6, v1 // (op 2879) 
v_lshl_add_u32 v83, v75, 8, v71 // (op 2879) 


s_ashr_i64 s[6:7], s[28:29], 7 // (op 877) 
s_and_b32 s1, s6, 4294967294 // (op 877) 

s_ashr_i64 s[6:7], s[28:29], 7 // (op 876) 
s_mov_b32 s14, s6 // (op 876) convert

s_cmp_lt_u32 s1, s14 // (op 876) 














s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_876 // (op 876) Condition: Top (jump to RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_876 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_876:
s_barrier  // (op 3120) 
s_add_u32 s1, s1, 1 // (op 878) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 2396) 
v_mov_b64 v[16:17], s[6:7] // (op 2396) convert


v_mov_b64 v[18:19], v[44:45] // (op 2396) 
v_mov_b64 v[20:21], v[18:19] // (op 2396) 
s_mov_b32 m0, s0 // (op 2396) reg expression
buffer_load_dwordx4 v20, s[8:11], 0 offen offset: 0 lds // (op 2396) Load value direct to lds
v_add_co_u32 v20, s[6:7], v20, v16 // (op 2396) least significant half
v_addc_co_u32 v21, s[6:7], 0, v21, s[6:7] // (op 2396) most significant half

s_add_u32 m0, m0, 4096 // (op 2396) 
buffer_load_dwordx4 v20, s[8:11], 0 offen offset: 0 lds // (op 2396) Load value direct to lds



v_add_co_u32 v44, vcc, 128, v44 // (op 2415) least significant half
v_addc_co_u32 v45, vcc, 0, v45, vcc // (op 2415) most significant half

s_mul_i32 s7, s25, 6 // (op 225) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 225) least significant
s_mul_hi_u32 s14, s24, 6 // (op 225) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 225) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[6:7] // (op 225) 


v_mov_b32 v14, v15 // (op 225) 
v_mov_b32 v18, v14 // (op 225) reg expression
buffer_load_dwordx3 v[20:22], v18, s[40:43], 0 offen // (op 225) Load value
buffer_load_dwordx3 v[24:26], v18, s[40:43], 0 offen offset:12 // (op 225) Load value
buffer_load_dwordx3 v[28:30], v18, s[40:43], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx3 v[32:34], v18, s[40:43], 0 offen offset:36 // (op 225) Load value
v_add_u32 v18, v18, v16 // (op 225) 
buffer_load_dwordx3 v[36:38], v18, s[40:43], 0 offen // (op 225) Load value
buffer_load_dwordx3 v[40:42], v18, s[40:43], 0 offen offset:12 // (op 225) Load value
buffer_load_dwordx3 v[48:50], v18, s[40:43], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx3 v[52:54], v18, s[40:43], 0 offen offset:36 // (op 225) Load value



s_lshl_b64 s[6:7], s[24:25], 7 // (op 2487) 
s_mul_i32 s15, s7, 6 // (op 2487) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 2487) least significant
s_mul_hi_u32 s18, s6, 6 // (op 2487) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 2487) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 2487) 

v_add_u32 v15, v15, s6 // (op 2487) 



s_barrier  // (op 3139) 
v_mov_b32 v14, v67 // (op 226) 




buffer_load_ubyte v23, v14, s[52:55], 0 offen offset: 0 // (op 226) Load value

v_add_u32 v67, 4, v67 // (op 2551) 
s_waitcnt vmcnt(8)
ds_write_b96 v63, v[20:22] offset:16384 // (op 222) Store local data 


v_mov_b32 v14, v46 // (op 227) 
v_mov_b32 v27, v14 // (op 227) reg expression

buffer_load_dword v31, v27, s[56:59], 0 offen // (op 227) Load value


s_lshl_b32 s6, s34, 2 // (op 2615) 
v_add_u32 v46, v46, s6 // (op 2615) 

s_waitcnt vmcnt(8)
ds_write_b96 v63, v[24:26] offset:16400 // (op 222) Store local data 
s_barrier  // (op 3129) 
s_waitcnt vmcnt(7)
ds_write_b96 v63, v[28:30] offset:16416 // (op 222) Store local data 



ds_read_b128 v[68:71], v3 // (op 205) Load local data 
ds_read_b128 v[72:75], v3 offset:32 // (op 205) Load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v63, v[32:34] offset:16432 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v63, v[36:38] offset:16640 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v63, v[40:42] offset:16656 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v63, v[48:50] offset:16672 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v63, v[52:54] offset:16688 // (op 222) Store local data 

s_barrier  // (op 3160) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3150) 






ds_read_b96_tr_b6 v[16:18], v7 offset:16384 // (op 207) Transpose load local data 
ds_read_b96_tr_b6 v[19:21], v7 offset:20480 // (op 207) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v79, v23 offset:50176 // (op 223) Store local data 

s_barrier  // (op 3180) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3170) 






ds_read_u8 v14, v0 offset:50176 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v83, v31 offset:49152 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3190) 


ds_read_u8 v22, v6 offset:49152 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[68:75], v[16:21], a[0:15], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 212) 





ds_read_b128 v[16:19], v3 offset:64 // (op 214) Load local data 
ds_read_b128 v[20:23], v3 offset:96 // (op 214) Load local data 



ds_read_b96_tr_b6 v[24:26], v7 offset:32768 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v7 offset:36864 // (op 216) Transpose load local data 



ds_read_u8 v30, v0 offset:50178 // (op 218) Load local data 


ds_read_u8 v31, v6 offset:49664 // (op 220) Load local data 



ds_read_b96_tr_b6 v[32:34], v7 offset:16448 // (op 416) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v7 offset:20544 // (op 416) Transpose load local data 


ds_read_u8 v38, v6 offset:49216 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[68:75], v[32:37], a[16:31], v14, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 421) 





ds_read_b96_tr_b6 v[32:34], v7 offset:32832 // (op 425) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v7 offset:36928 // (op 425) Transpose load local data 


ds_read_u8 v38, v6 offset:49728 // (op 429) Load local data 



ds_read_b96_tr_b6 v[48:50], v7 offset:16512 // (op 650) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v7 offset:20608 // (op 650) Transpose load local data 


ds_read_u8 v47, v6 offset:49280 // (op 654) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[68:75], v[48:53], a[32:47], v14, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 655) 





ds_read_b96_tr_b6 v[48:50], v7 offset:32896 // (op 659) Transpose load local data 
ds_read_b96_tr_b6 v[51:53], v7 offset:36992 // (op 659) Transpose load local data 


ds_read_u8 v47, v6 offset:49792 // (op 663) Load local data 



ds_read_b96_tr_b6 v[54:56], v7 offset:16576 // (op 884) Transpose load local data 
ds_read_b96_tr_b6 v[57:59], v7 offset:20672 // (op 884) Transpose load local data 


ds_read_u8 v39, v6 offset:49344 // (op 888) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[68:75], v[54:59], a[48:63], v14, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 889) 







ds_read_b96_tr_b6 v[54:56], v7 offset:32960 // (op 893) Transpose load local data 
ds_read_b96_tr_b6 v[57:59], v7 offset:37056 // (op 893) Transpose load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[16:23], v[24:29], a[0:15], v30, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 221) 




ds_read_u8 v14, v6 offset:49856 // (op 897) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[16:23], v[32:37], a[16:31], v30, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 430) 


v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[16:23], v[48:53], a[32:47], v30, v47 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 664) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[16:23], v[54:59], a[48:63], v30, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b011 // (op 898) 




s_ashr_i64 s[6:7], s[28:29], 7 // (op 876) 
s_mov_b32 s14, s6 // (op 876) convert

s_cmp_lt_u32 s1, s14 // (op 876) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_876 // (op 876) Condition: Bottom (jump to RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopTop_KLoopTail_876 if true)
RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ForLoopBottom_KLoopTail_876:
























s_barrier  // (op 3303) 








v_cmp_eq_f32 vcc, 0.00000, v12 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v14, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_accvgpr_read v18, a4 // (op 50) 
v_accvgpr_read v19, a5 // (op 50) 
v_accvgpr_read v20, a6 // (op 50) 
v_accvgpr_read v21, a7 // (op 50) 
v_accvgpr_read v22, a8 // (op 50) 
v_accvgpr_read v23, a9 // (op 50) 
v_accvgpr_read v24, a10 // (op 50) 
v_accvgpr_read v25, a11 // (op 50) 
v_accvgpr_read v26, a12 // (op 50) 
v_accvgpr_read v27, a13 // (op 50) 
v_accvgpr_read v28, a14 // (op 50) 
v_accvgpr_read v29, a15 // (op 50) 
v_mul_f32 v30, v13, v14 // (op 50) 
v_mul_f32 v31, v13, v15 // (op 50) 
v_mul_f32 v32, v13, v16 // (op 50) 
v_mul_f32 v33, v13, v17 // (op 50) 
v_mul_f32 v34, v13, v18 // (op 50) 
v_mul_f32 v35, v13, v19 // (op 50) 
v_mul_f32 v36, v13, v20 // (op 50) 
v_mul_f32 v37, v13, v21 // (op 50) 
v_mul_f32 v38, v13, v22 // (op 50) 
v_mul_f32 v39, v13, v23 // (op 50) 
v_mul_f32 v40, v13, v24 // (op 50) 
v_mul_f32 v41, v13, v25 // (op 50) 
v_mul_f32 v42, v13, v26 // (op 50) 
v_mul_f32 v43, v13, v27 // (op 50) 
v_mul_f32 v44, v13, v28 // (op 50) 
v_mul_f32 v45, v13, v29 // (op 50) 

v_cvt_f16_f32 v14, v30 // (op 56) 
v_cvt_f16_f32 v15, v31 // (op 56) 
v_cvt_f16_f32 v16, v32 // (op 56) 
v_cvt_f16_f32 v17, v33 // (op 56) 
v_cvt_f16_f32 v18, v34 // (op 56) 
v_cvt_f16_f32 v19, v35 // (op 56) 
v_cvt_f16_f32 v20, v36 // (op 56) 
v_cvt_f16_f32 v21, v37 // (op 56) 
v_cvt_f16_f32 v22, v38 // (op 56) 
v_cvt_f16_f32 v23, v39 // (op 56) 
v_cvt_f16_f32 v24, v40 // (op 56) 
v_cvt_f16_f32 v25, v41 // (op 56) 
v_cvt_f16_f32 v26, v42 // (op 56) 
v_cvt_f16_f32 v27, v43 // (op 56) 
v_cvt_f16_f32 v28, v44 // (op 56) 
v_cvt_f16_f32 v29, v45 // (op 56) 


v_pack_b32_f16 v30, v14, v15 // (op 58) 
v_pack_b32_f16 v31, v16, v17 // (op 58) 
v_pack_b32_f16 v32, v18, v19 // (op 58) 
v_pack_b32_f16 v33, v20, v21 // (op 58) 
v_pack_b32_f16 v34, v22, v23 // (op 58) 
v_pack_b32_f16 v35, v24, v25 // (op 58) 
v_pack_b32_f16 v36, v26, v27 // (op 58) 
v_pack_b32_f16 v37, v28, v29 // (op 58) 

v_mov_b32 v0, v11 // (op 58) reg expression

buffer_store_dwordx2 v[30:31], v0, s[48:51], 0 offen // (op 58) Store value
buffer_store_dwordx2 v[32:33], v0, s[48:51], 0 offen offset:16 // (op 58) Store value
buffer_store_dwordx2 v[34:35], v0, s[48:51], 0 offen offset:32 // (op 58) Store value
buffer_store_dwordx2 v[36:37], v0, s[48:51], 0 offen offset:48 // (op 58) Store value



v_accvgpr_read v14, a16 // (op 1262) 
v_accvgpr_read v15, a17 // (op 1262) 
v_accvgpr_read v16, a18 // (op 1262) 
v_accvgpr_read v17, a19 // (op 1262) 
v_accvgpr_read v18, a20 // (op 1262) 
v_accvgpr_read v19, a21 // (op 1262) 
v_accvgpr_read v20, a22 // (op 1262) 
v_accvgpr_read v21, a23 // (op 1262) 
v_accvgpr_read v22, a24 // (op 1262) 
v_accvgpr_read v23, a25 // (op 1262) 
v_accvgpr_read v24, a26 // (op 1262) 
v_accvgpr_read v25, a27 // (op 1262) 
v_accvgpr_read v26, a28 // (op 1262) 
v_accvgpr_read v27, a29 // (op 1262) 
v_accvgpr_read v28, a30 // (op 1262) 
v_accvgpr_read v29, a31 // (op 1262) 
v_mul_f32 v30, v13, v14 // (op 1262) 
v_mul_f32 v31, v13, v15 // (op 1262) 
v_mul_f32 v32, v13, v16 // (op 1262) 
v_mul_f32 v33, v13, v17 // (op 1262) 
v_mul_f32 v34, v13, v18 // (op 1262) 
v_mul_f32 v35, v13, v19 // (op 1262) 
v_mul_f32 v36, v13, v20 // (op 1262) 
v_mul_f32 v37, v13, v21 // (op 1262) 
v_mul_f32 v38, v13, v22 // (op 1262) 
v_mul_f32 v39, v13, v23 // (op 1262) 
v_mul_f32 v40, v13, v24 // (op 1262) 
v_mul_f32 v41, v13, v25 // (op 1262) 
v_mul_f32 v42, v13, v26 // (op 1262) 
v_mul_f32 v43, v13, v27 // (op 1262) 
v_mul_f32 v44, v13, v28 // (op 1262) 
v_mul_f32 v45, v13, v29 // (op 1262) 

v_cvt_f16_f32 v14, v30 // (op 1264) 
v_cvt_f16_f32 v15, v31 // (op 1264) 
v_cvt_f16_f32 v16, v32 // (op 1264) 
v_cvt_f16_f32 v17, v33 // (op 1264) 
v_cvt_f16_f32 v18, v34 // (op 1264) 
v_cvt_f16_f32 v19, v35 // (op 1264) 
v_cvt_f16_f32 v20, v36 // (op 1264) 
v_cvt_f16_f32 v21, v37 // (op 1264) 
v_cvt_f16_f32 v22, v38 // (op 1264) 
v_cvt_f16_f32 v23, v39 // (op 1264) 
v_cvt_f16_f32 v24, v40 // (op 1264) 
v_cvt_f16_f32 v25, v41 // (op 1264) 
v_cvt_f16_f32 v26, v42 // (op 1264) 
v_cvt_f16_f32 v27, v43 // (op 1264) 
v_cvt_f16_f32 v28, v44 // (op 1264) 
v_cvt_f16_f32 v29, v45 // (op 1264) 


v_pack_b32_f16 v30, v14, v15 // (op 1265) 
v_pack_b32_f16 v31, v16, v17 // (op 1265) 
v_pack_b32_f16 v32, v18, v19 // (op 1265) 
v_pack_b32_f16 v33, v20, v21 // (op 1265) 
v_pack_b32_f16 v34, v22, v23 // (op 1265) 
v_pack_b32_f16 v35, v24, v25 // (op 1265) 
v_pack_b32_f16 v36, v26, v27 // (op 1265) 
v_pack_b32_f16 v37, v28, v29 // (op 1265) 

s_lshl_b32 s0, s4, 7 // (op 1265) 
v_add_u32 v0, v11, s0 // (op 1265) 


buffer_store_dwordx2 v[30:31], v0, s[48:51], 0 offen // (op 1265) Store value
buffer_store_dwordx2 v[32:33], v0, s[48:51], 0 offen offset:16 // (op 1265) Store value
buffer_store_dwordx2 v[34:35], v0, s[48:51], 0 offen offset:32 // (op 1265) Store value
buffer_store_dwordx2 v[36:37], v0, s[48:51], 0 offen offset:48 // (op 1265) Store value



v_accvgpr_read v14, a32 // (op 1276) 
v_accvgpr_read v15, a33 // (op 1276) 
v_accvgpr_read v16, a34 // (op 1276) 
v_accvgpr_read v17, a35 // (op 1276) 
v_accvgpr_read v18, a36 // (op 1276) 
v_accvgpr_read v19, a37 // (op 1276) 
v_accvgpr_read v20, a38 // (op 1276) 
v_accvgpr_read v21, a39 // (op 1276) 
v_accvgpr_read v22, a40 // (op 1276) 
v_accvgpr_read v23, a41 // (op 1276) 
v_accvgpr_read v24, a42 // (op 1276) 
v_accvgpr_read v25, a43 // (op 1276) 
v_accvgpr_read v26, a44 // (op 1276) 
v_accvgpr_read v27, a45 // (op 1276) 
v_accvgpr_read v28, a46 // (op 1276) 
v_accvgpr_read v29, a47 // (op 1276) 
v_mul_f32 v30, v13, v14 // (op 1276) 
v_mul_f32 v31, v13, v15 // (op 1276) 
v_mul_f32 v32, v13, v16 // (op 1276) 
v_mul_f32 v33, v13, v17 // (op 1276) 
v_mul_f32 v34, v13, v18 // (op 1276) 
v_mul_f32 v35, v13, v19 // (op 1276) 
v_mul_f32 v36, v13, v20 // (op 1276) 
v_mul_f32 v37, v13, v21 // (op 1276) 
v_mul_f32 v38, v13, v22 // (op 1276) 
v_mul_f32 v39, v13, v23 // (op 1276) 
v_mul_f32 v40, v13, v24 // (op 1276) 
v_mul_f32 v41, v13, v25 // (op 1276) 
v_mul_f32 v42, v13, v26 // (op 1276) 
v_mul_f32 v43, v13, v27 // (op 1276) 
v_mul_f32 v44, v13, v28 // (op 1276) 
v_mul_f32 v45, v13, v29 // (op 1276) 

v_cvt_f16_f32 v14, v30 // (op 1278) 
v_cvt_f16_f32 v15, v31 // (op 1278) 
v_cvt_f16_f32 v16, v32 // (op 1278) 
v_cvt_f16_f32 v17, v33 // (op 1278) 
v_cvt_f16_f32 v18, v34 // (op 1278) 
v_cvt_f16_f32 v19, v35 // (op 1278) 
v_cvt_f16_f32 v20, v36 // (op 1278) 
v_cvt_f16_f32 v21, v37 // (op 1278) 
v_cvt_f16_f32 v22, v38 // (op 1278) 
v_cvt_f16_f32 v23, v39 // (op 1278) 
v_cvt_f16_f32 v24, v40 // (op 1278) 
v_cvt_f16_f32 v25, v41 // (op 1278) 
v_cvt_f16_f32 v26, v42 // (op 1278) 
v_cvt_f16_f32 v27, v43 // (op 1278) 
v_cvt_f16_f32 v28, v44 // (op 1278) 
v_cvt_f16_f32 v29, v45 // (op 1278) 


v_pack_b32_f16 v30, v14, v15 // (op 1279) 
v_pack_b32_f16 v31, v16, v17 // (op 1279) 
v_pack_b32_f16 v32, v18, v19 // (op 1279) 
v_pack_b32_f16 v33, v20, v21 // (op 1279) 
v_pack_b32_f16 v34, v22, v23 // (op 1279) 
v_pack_b32_f16 v35, v24, v25 // (op 1279) 
v_pack_b32_f16 v36, v26, v27 // (op 1279) 
v_pack_b32_f16 v37, v28, v29 // (op 1279) 

s_lshl_b32 s0, s4, 8 // (op 1279) 
v_add_u32 v0, v11, s0 // (op 1279) 


buffer_store_dwordx2 v[30:31], v0, s[48:51], 0 offen // (op 1279) Store value
buffer_store_dwordx2 v[32:33], v0, s[48:51], 0 offen offset:16 // (op 1279) Store value
buffer_store_dwordx2 v[34:35], v0, s[48:51], 0 offen offset:32 // (op 1279) Store value
buffer_store_dwordx2 v[36:37], v0, s[48:51], 0 offen offset:48 // (op 1279) Store value



v_accvgpr_read v14, a48 // (op 1290) 
v_accvgpr_read v15, a49 // (op 1290) 
v_accvgpr_read v16, a50 // (op 1290) 
v_accvgpr_read v17, a51 // (op 1290) 
v_accvgpr_read v18, a52 // (op 1290) 
v_accvgpr_read v19, a53 // (op 1290) 
v_accvgpr_read v20, a54 // (op 1290) 
v_accvgpr_read v21, a55 // (op 1290) 
v_accvgpr_read v22, a56 // (op 1290) 
v_accvgpr_read v23, a57 // (op 1290) 
v_accvgpr_read v24, a58 // (op 1290) 
v_accvgpr_read v25, a59 // (op 1290) 
v_accvgpr_read v26, a60 // (op 1290) 
v_accvgpr_read v27, a61 // (op 1290) 
v_accvgpr_read v28, a62 // (op 1290) 
v_accvgpr_read v29, a63 // (op 1290) 
v_mul_f32 v30, v13, v14 // (op 1290) 
v_mul_f32 v31, v13, v15 // (op 1290) 
v_mul_f32 v32, v13, v16 // (op 1290) 
v_mul_f32 v33, v13, v17 // (op 1290) 
v_mul_f32 v34, v13, v18 // (op 1290) 
v_mul_f32 v35, v13, v19 // (op 1290) 
v_mul_f32 v36, v13, v20 // (op 1290) 
v_mul_f32 v37, v13, v21 // (op 1290) 
v_mul_f32 v38, v13, v22 // (op 1290) 
v_mul_f32 v39, v13, v23 // (op 1290) 
v_mul_f32 v40, v13, v24 // (op 1290) 
v_mul_f32 v41, v13, v25 // (op 1290) 
v_mul_f32 v42, v13, v26 // (op 1290) 
v_mul_f32 v43, v13, v27 // (op 1290) 
v_mul_f32 v44, v13, v28 // (op 1290) 
v_mul_f32 v45, v13, v29 // (op 1290) 

v_cvt_f16_f32 v14, v30 // (op 1292) 
v_cvt_f16_f32 v15, v31 // (op 1292) 
v_cvt_f16_f32 v16, v32 // (op 1292) 
v_cvt_f16_f32 v17, v33 // (op 1292) 
v_cvt_f16_f32 v18, v34 // (op 1292) 
v_cvt_f16_f32 v19, v35 // (op 1292) 
v_cvt_f16_f32 v20, v36 // (op 1292) 
v_cvt_f16_f32 v21, v37 // (op 1292) 
v_cvt_f16_f32 v22, v38 // (op 1292) 
v_cvt_f16_f32 v23, v39 // (op 1292) 
v_cvt_f16_f32 v24, v40 // (op 1292) 
v_cvt_f16_f32 v25, v41 // (op 1292) 
v_cvt_f16_f32 v26, v42 // (op 1292) 
v_cvt_f16_f32 v27, v43 // (op 1292) 
v_cvt_f16_f32 v28, v44 // (op 1292) 
v_cvt_f16_f32 v29, v45 // (op 1292) 


v_pack_b32_f16 v30, v14, v15 // (op 1293) 
v_pack_b32_f16 v31, v16, v17 // (op 1293) 
v_pack_b32_f16 v32, v18, v19 // (op 1293) 
v_pack_b32_f16 v33, v20, v21 // (op 1293) 
v_pack_b32_f16 v34, v22, v23 // (op 1293) 
v_pack_b32_f16 v35, v24, v25 // (op 1293) 
v_pack_b32_f16 v36, v26, v27 // (op 1293) 
v_pack_b32_f16 v37, v28, v29 // (op 1293) 

s_lshl_b32 s0, s4, 7 // (op 1293) 
s_mul_i32 s1, 3, s0 // (op 1293) 

v_add_u32 v0, v11, s1 // (op 1293) 


buffer_store_dwordx2 v[30:31], v0, s[48:51], 0 offen // (op 1293) Store value
buffer_store_dwordx2 v[32:33], v0, s[48:51], 0 offen offset:16 // (op 1293) Store value
buffer_store_dwordx2 v[34:35], v0, s[48:51], 0 offen offset:32 // (op 1293) Store value
buffer_store_dwordx2 v[36:37], v0, s[48:51], 0 offen offset:48 // (op 1293) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalFalse_0_beta_157:


v_mov_b32 v0, v8 // (op 170) reg expression

buffer_load_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[16:17], v0, s[44:47], 0 offen offset:16 // (op 170) Load value
buffer_load_dwordx2 v[18:19], v0, s[44:47], 0 offen offset:32 // (op 170) Load value
v_accvgpr_read v22, a0 // (op 166) 
v_accvgpr_read v23, a1 // (op 166) 
v_accvgpr_read v24, a2 // (op 166) 
v_accvgpr_read v25, a3 // (op 166) 
v_accvgpr_read v26, a4 // (op 166) 
v_accvgpr_read v27, a5 // (op 166) 
v_accvgpr_read v28, a6 // (op 166) 
v_accvgpr_read v29, a7 // (op 166) 
v_accvgpr_read v30, a8 // (op 166) 
v_accvgpr_read v31, a9 // (op 166) 
v_accvgpr_read v32, a10 // (op 166) 
v_accvgpr_read v33, a11 // (op 166) 
v_accvgpr_read v34, a12 // (op 166) 
v_accvgpr_read v35, a13 // (op 166) 
v_accvgpr_read v36, a14 // (op 166) 
v_accvgpr_read v37, a15 // (op 166) 
v_mul_f32 v38, v13, v22 // (op 166) 
v_mul_f32 v39, v13, v23 // (op 166) 
v_mul_f32 v40, v13, v24 // (op 166) 
v_mul_f32 v41, v13, v25 // (op 166) 
v_mul_f32 v42, v13, v26 // (op 166) 
v_mul_f32 v43, v13, v27 // (op 166) 
v_mul_f32 v44, v13, v28 // (op 166) 
v_mul_f32 v45, v13, v29 // (op 166) 
v_mul_f32 v46, v13, v30 // (op 166) 
v_mul_f32 v47, v13, v31 // (op 166) 
v_mul_f32 v48, v13, v32 // (op 166) 
v_mul_f32 v49, v13, v33 // (op 166) 
v_mul_f32 v50, v13, v34 // (op 166) 
v_mul_f32 v51, v13, v35 // (op 166) 
v_mul_f32 v52, v13, v36 // (op 166) 
v_mul_f32 v53, v13, v37 // (op 166) 

buffer_load_dwordx2 v[20:21], v0, s[44:47], 0 offen offset:48 // (op 170) Load value

s_waitcnt vmcnt(3)
v_cvt_f32_f16 v6, v14 // (op 171) 
v_lshrrev_b32 v7, 16, v14 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v22, v12, v6 // (op 171) 
v_mul_f32 v23, v12, v7 // (op 171) 
v_cvt_f32_f16 v6, v15 // (op 171) 
v_lshrrev_b32 v7, 16, v15 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v24, v12, v6 // (op 171) 
v_mul_f32 v25, v12, v7 // (op 171) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v6, v16 // (op 171) 
v_lshrrev_b32 v7, 16, v16 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v26, v12, v6 // (op 171) 
v_mul_f32 v27, v12, v7 // (op 171) 
v_cvt_f32_f16 v6, v17 // (op 171) 
v_lshrrev_b32 v7, 16, v17 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v28, v12, v6 // (op 171) 
v_mul_f32 v29, v12, v7 // (op 171) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v6, v18 // (op 171) 
v_lshrrev_b32 v7, 16, v18 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v30, v12, v6 // (op 171) 
v_mul_f32 v31, v12, v7 // (op 171) 
v_cvt_f32_f16 v6, v19 // (op 171) 
v_lshrrev_b32 v7, 16, v19 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v32, v12, v6 // (op 171) 
v_mul_f32 v33, v12, v7 // (op 171) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v20 // (op 171) 
v_lshrrev_b32 v7, 16, v20 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v34, v12, v6 // (op 171) 
v_mul_f32 v35, v12, v7 // (op 171) 
v_cvt_f32_f16 v6, v21 // (op 171) 
v_lshrrev_b32 v7, 16, v21 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v36, v12, v6 // (op 171) 
v_mul_f32 v37, v12, v7 // (op 171) 


v_add_f32 v54, v22, v38 // (op 167) 
v_add_f32 v55, v23, v39 // (op 167) 
v_add_f32 v56, v24, v40 // (op 167) 
v_add_f32 v57, v25, v41 // (op 167) 
v_add_f32 v58, v26, v42 // (op 167) 
v_add_f32 v59, v27, v43 // (op 167) 
v_add_f32 v60, v28, v44 // (op 167) 
v_add_f32 v61, v29, v45 // (op 167) 
v_add_f32 v62, v30, v46 // (op 167) 
v_add_f32 v63, v31, v47 // (op 167) 
v_add_f32 v64, v32, v48 // (op 167) 
v_add_f32 v65, v33, v49 // (op 167) 
v_add_f32 v66, v34, v50 // (op 167) 
v_add_f32 v67, v35, v51 // (op 167) 
v_add_f32 v68, v36, v52 // (op 167) 
v_add_f32 v69, v37, v53 // (op 167) 


v_cvt_f16_f32 v14, v54 // (op 168) 
v_cvt_f16_f32 v15, v55 // (op 168) 
v_cvt_f16_f32 v16, v56 // (op 168) 
v_cvt_f16_f32 v17, v57 // (op 168) 
v_cvt_f16_f32 v18, v58 // (op 168) 
v_cvt_f16_f32 v19, v59 // (op 168) 
v_cvt_f16_f32 v20, v60 // (op 168) 
v_cvt_f16_f32 v21, v61 // (op 168) 
v_cvt_f16_f32 v22, v62 // (op 168) 
v_cvt_f16_f32 v23, v63 // (op 168) 
v_cvt_f16_f32 v24, v64 // (op 168) 
v_cvt_f16_f32 v25, v65 // (op 168) 
v_cvt_f16_f32 v26, v66 // (op 168) 
v_cvt_f16_f32 v27, v67 // (op 168) 
v_cvt_f16_f32 v28, v68 // (op 168) 
v_cvt_f16_f32 v29, v69 // (op 168) 


v_pack_b32_f16 v30, v14, v15 // (op 169) 
v_pack_b32_f16 v31, v16, v17 // (op 169) 
v_pack_b32_f16 v32, v18, v19 // (op 169) 
v_pack_b32_f16 v33, v20, v21 // (op 169) 
v_pack_b32_f16 v34, v22, v23 // (op 169) 
v_pack_b32_f16 v35, v24, v25 // (op 169) 
v_pack_b32_f16 v36, v26, v27 // (op 169) 
v_pack_b32_f16 v37, v28, v29 // (op 169) 

v_mov_b32 v0, v11 // (op 169) reg expression

buffer_store_dwordx2 v[30:31], v0, s[48:51], 0 offen // (op 169) Store value
buffer_store_dwordx2 v[32:33], v0, s[48:51], 0 offen offset:16 // (op 169) Store value
buffer_store_dwordx2 v[34:35], v0, s[48:51], 0 offen offset:32 // (op 169) Store value
buffer_store_dwordx2 v[36:37], v0, s[48:51], 0 offen offset:48 // (op 169) Store value





s_lshl_b32 s0, s2, 7 // (op 1327) 
v_add_u32 v0, v8, s0 // (op 1327) 


buffer_load_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 1327) Load value
buffer_load_dwordx2 v[16:17], v0, s[44:47], 0 offen offset:16 // (op 1327) Load value
v_accvgpr_read v22, a16 // (op 1323) 
v_accvgpr_read v23, a17 // (op 1323) 
v_accvgpr_read v24, a18 // (op 1323) 
v_accvgpr_read v25, a19 // (op 1323) 
v_accvgpr_read v26, a20 // (op 1323) 
v_accvgpr_read v27, a21 // (op 1323) 
v_accvgpr_read v28, a22 // (op 1323) 
v_accvgpr_read v29, a23 // (op 1323) 
v_accvgpr_read v30, a24 // (op 1323) 
v_accvgpr_read v31, a25 // (op 1323) 
v_accvgpr_read v32, a26 // (op 1323) 
v_accvgpr_read v33, a27 // (op 1323) 
v_accvgpr_read v34, a28 // (op 1323) 
v_accvgpr_read v35, a29 // (op 1323) 
v_accvgpr_read v36, a30 // (op 1323) 
v_accvgpr_read v37, a31 // (op 1323) 
v_mul_f32 v38, v13, v22 // (op 1323) 
v_mul_f32 v39, v13, v23 // (op 1323) 
v_mul_f32 v40, v13, v24 // (op 1323) 
v_mul_f32 v41, v13, v25 // (op 1323) 
v_mul_f32 v42, v13, v26 // (op 1323) 
v_mul_f32 v43, v13, v27 // (op 1323) 
v_mul_f32 v44, v13, v28 // (op 1323) 
v_mul_f32 v45, v13, v29 // (op 1323) 
v_mul_f32 v46, v13, v30 // (op 1323) 
v_mul_f32 v47, v13, v31 // (op 1323) 
v_mul_f32 v48, v13, v32 // (op 1323) 
v_mul_f32 v49, v13, v33 // (op 1323) 
v_mul_f32 v50, v13, v34 // (op 1323) 
v_mul_f32 v51, v13, v35 // (op 1323) 
v_mul_f32 v52, v13, v36 // (op 1323) 
v_mul_f32 v53, v13, v37 // (op 1323) 

buffer_load_dwordx2 v[18:19], v0, s[44:47], 0 offen offset:32 // (op 1327) Load value
buffer_load_dwordx2 v[20:21], v0, s[44:47], 0 offen offset:48 // (op 1327) Load value

s_waitcnt vmcnt(3)
v_cvt_f32_f16 v6, v14 // (op 1328) 
v_lshrrev_b32 v7, 16, v14 // (op 1328) 
v_cvt_f32_f16 v7, v7 // (op 1328) 
v_mul_f32 v22, v12, v6 // (op 1328) 
v_mul_f32 v23, v12, v7 // (op 1328) 
v_cvt_f32_f16 v6, v15 // (op 1328) 
v_lshrrev_b32 v7, 16, v15 // (op 1328) 
v_cvt_f32_f16 v7, v7 // (op 1328) 
v_mul_f32 v24, v12, v6 // (op 1328) 
v_mul_f32 v25, v12, v7 // (op 1328) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v6, v16 // (op 1328) 
v_lshrrev_b32 v7, 16, v16 // (op 1328) 
v_cvt_f32_f16 v7, v7 // (op 1328) 
v_mul_f32 v26, v12, v6 // (op 1328) 
v_mul_f32 v27, v12, v7 // (op 1328) 
v_cvt_f32_f16 v6, v17 // (op 1328) 
v_lshrrev_b32 v7, 16, v17 // (op 1328) 
v_cvt_f32_f16 v7, v7 // (op 1328) 
v_mul_f32 v28, v12, v6 // (op 1328) 
v_mul_f32 v29, v12, v7 // (op 1328) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v6, v18 // (op 1328) 
v_lshrrev_b32 v7, 16, v18 // (op 1328) 
v_cvt_f32_f16 v7, v7 // (op 1328) 
v_mul_f32 v30, v12, v6 // (op 1328) 
v_mul_f32 v31, v12, v7 // (op 1328) 
v_cvt_f32_f16 v6, v19 // (op 1328) 
v_lshrrev_b32 v7, 16, v19 // (op 1328) 
v_cvt_f32_f16 v7, v7 // (op 1328) 
v_mul_f32 v32, v12, v6 // (op 1328) 
v_mul_f32 v33, v12, v7 // (op 1328) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v20 // (op 1328) 
v_lshrrev_b32 v7, 16, v20 // (op 1328) 
v_cvt_f32_f16 v7, v7 // (op 1328) 
v_mul_f32 v34, v12, v6 // (op 1328) 
v_mul_f32 v35, v12, v7 // (op 1328) 
v_cvt_f32_f16 v6, v21 // (op 1328) 
v_lshrrev_b32 v7, 16, v21 // (op 1328) 
v_cvt_f32_f16 v7, v7 // (op 1328) 
v_mul_f32 v36, v12, v6 // (op 1328) 
v_mul_f32 v37, v12, v7 // (op 1328) 


v_add_f32 v54, v22, v38 // (op 1324) 
v_add_f32 v55, v23, v39 // (op 1324) 
v_add_f32 v56, v24, v40 // (op 1324) 
v_add_f32 v57, v25, v41 // (op 1324) 
v_add_f32 v58, v26, v42 // (op 1324) 
v_add_f32 v59, v27, v43 // (op 1324) 
v_add_f32 v60, v28, v44 // (op 1324) 
v_add_f32 v61, v29, v45 // (op 1324) 
v_add_f32 v62, v30, v46 // (op 1324) 
v_add_f32 v63, v31, v47 // (op 1324) 
v_add_f32 v64, v32, v48 // (op 1324) 
v_add_f32 v65, v33, v49 // (op 1324) 
v_add_f32 v66, v34, v50 // (op 1324) 
v_add_f32 v67, v35, v51 // (op 1324) 
v_add_f32 v68, v36, v52 // (op 1324) 
v_add_f32 v69, v37, v53 // (op 1324) 


v_cvt_f16_f32 v14, v54 // (op 1325) 
v_cvt_f16_f32 v15, v55 // (op 1325) 
v_cvt_f16_f32 v16, v56 // (op 1325) 
v_cvt_f16_f32 v17, v57 // (op 1325) 
v_cvt_f16_f32 v18, v58 // (op 1325) 
v_cvt_f16_f32 v19, v59 // (op 1325) 
v_cvt_f16_f32 v20, v60 // (op 1325) 
v_cvt_f16_f32 v21, v61 // (op 1325) 
v_cvt_f16_f32 v22, v62 // (op 1325) 
v_cvt_f16_f32 v23, v63 // (op 1325) 
v_cvt_f16_f32 v24, v64 // (op 1325) 
v_cvt_f16_f32 v25, v65 // (op 1325) 
v_cvt_f16_f32 v26, v66 // (op 1325) 
v_cvt_f16_f32 v27, v67 // (op 1325) 
v_cvt_f16_f32 v28, v68 // (op 1325) 
v_cvt_f16_f32 v29, v69 // (op 1325) 


v_pack_b32_f16 v30, v14, v15 // (op 1326) 
v_pack_b32_f16 v31, v16, v17 // (op 1326) 
v_pack_b32_f16 v32, v18, v19 // (op 1326) 
v_pack_b32_f16 v33, v20, v21 // (op 1326) 
v_pack_b32_f16 v34, v22, v23 // (op 1326) 
v_pack_b32_f16 v35, v24, v25 // (op 1326) 
v_pack_b32_f16 v36, v26, v27 // (op 1326) 
v_pack_b32_f16 v37, v28, v29 // (op 1326) 

s_lshl_b32 s0, s4, 7 // (op 1326) 
v_add_u32 v0, v11, s0 // (op 1326) 


buffer_store_dwordx2 v[30:31], v0, s[48:51], 0 offen // (op 1326) Store value
buffer_store_dwordx2 v[32:33], v0, s[48:51], 0 offen offset:16 // (op 1326) Store value
buffer_store_dwordx2 v[34:35], v0, s[48:51], 0 offen offset:32 // (op 1326) Store value
buffer_store_dwordx2 v[36:37], v0, s[48:51], 0 offen offset:48 // (op 1326) Store value





s_lshl_b32 s0, s2, 8 // (op 1341) 
v_add_u32 v0, v8, s0 // (op 1341) 


buffer_load_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 1341) Load value
buffer_load_dwordx2 v[16:17], v0, s[44:47], 0 offen offset:16 // (op 1341) Load value
v_accvgpr_read v22, a32 // (op 1337) 
v_accvgpr_read v23, a33 // (op 1337) 
v_accvgpr_read v24, a34 // (op 1337) 
v_accvgpr_read v25, a35 // (op 1337) 
v_accvgpr_read v26, a36 // (op 1337) 
v_accvgpr_read v27, a37 // (op 1337) 
v_accvgpr_read v28, a38 // (op 1337) 
v_accvgpr_read v29, a39 // (op 1337) 
v_accvgpr_read v30, a40 // (op 1337) 
v_accvgpr_read v31, a41 // (op 1337) 
v_accvgpr_read v32, a42 // (op 1337) 
v_accvgpr_read v33, a43 // (op 1337) 
v_accvgpr_read v34, a44 // (op 1337) 
v_accvgpr_read v35, a45 // (op 1337) 
v_accvgpr_read v36, a46 // (op 1337) 
v_accvgpr_read v37, a47 // (op 1337) 
v_mul_f32 v38, v13, v22 // (op 1337) 
v_mul_f32 v39, v13, v23 // (op 1337) 
v_mul_f32 v40, v13, v24 // (op 1337) 
v_mul_f32 v41, v13, v25 // (op 1337) 
v_mul_f32 v42, v13, v26 // (op 1337) 
v_mul_f32 v43, v13, v27 // (op 1337) 
v_mul_f32 v44, v13, v28 // (op 1337) 
v_mul_f32 v45, v13, v29 // (op 1337) 
v_mul_f32 v46, v13, v30 // (op 1337) 
v_mul_f32 v47, v13, v31 // (op 1337) 
v_mul_f32 v48, v13, v32 // (op 1337) 
v_mul_f32 v49, v13, v33 // (op 1337) 
v_mul_f32 v50, v13, v34 // (op 1337) 
v_mul_f32 v51, v13, v35 // (op 1337) 
v_mul_f32 v52, v13, v36 // (op 1337) 
v_mul_f32 v53, v13, v37 // (op 1337) 

buffer_load_dwordx2 v[18:19], v0, s[44:47], 0 offen offset:32 // (op 1341) Load value
buffer_load_dwordx2 v[20:21], v0, s[44:47], 0 offen offset:48 // (op 1341) Load value

s_waitcnt vmcnt(3)
v_cvt_f32_f16 v6, v14 // (op 1342) 
v_lshrrev_b32 v7, 16, v14 // (op 1342) 
v_cvt_f32_f16 v7, v7 // (op 1342) 
v_mul_f32 v22, v12, v6 // (op 1342) 
v_mul_f32 v23, v12, v7 // (op 1342) 
v_cvt_f32_f16 v6, v15 // (op 1342) 
v_lshrrev_b32 v7, 16, v15 // (op 1342) 
v_cvt_f32_f16 v7, v7 // (op 1342) 
v_mul_f32 v24, v12, v6 // (op 1342) 
v_mul_f32 v25, v12, v7 // (op 1342) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v6, v16 // (op 1342) 
v_lshrrev_b32 v7, 16, v16 // (op 1342) 
v_cvt_f32_f16 v7, v7 // (op 1342) 
v_mul_f32 v26, v12, v6 // (op 1342) 
v_mul_f32 v27, v12, v7 // (op 1342) 
v_cvt_f32_f16 v6, v17 // (op 1342) 
v_lshrrev_b32 v7, 16, v17 // (op 1342) 
v_cvt_f32_f16 v7, v7 // (op 1342) 
v_mul_f32 v28, v12, v6 // (op 1342) 
v_mul_f32 v29, v12, v7 // (op 1342) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v6, v18 // (op 1342) 
v_lshrrev_b32 v7, 16, v18 // (op 1342) 
v_cvt_f32_f16 v7, v7 // (op 1342) 
v_mul_f32 v30, v12, v6 // (op 1342) 
v_mul_f32 v31, v12, v7 // (op 1342) 
v_cvt_f32_f16 v6, v19 // (op 1342) 
v_lshrrev_b32 v7, 16, v19 // (op 1342) 
v_cvt_f32_f16 v7, v7 // (op 1342) 
v_mul_f32 v32, v12, v6 // (op 1342) 
v_mul_f32 v33, v12, v7 // (op 1342) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v20 // (op 1342) 
v_lshrrev_b32 v7, 16, v20 // (op 1342) 
v_cvt_f32_f16 v7, v7 // (op 1342) 
v_mul_f32 v34, v12, v6 // (op 1342) 
v_mul_f32 v35, v12, v7 // (op 1342) 
v_cvt_f32_f16 v6, v21 // (op 1342) 
v_lshrrev_b32 v7, 16, v21 // (op 1342) 
v_cvt_f32_f16 v7, v7 // (op 1342) 
v_mul_f32 v36, v12, v6 // (op 1342) 
v_mul_f32 v37, v12, v7 // (op 1342) 


v_add_f32 v54, v22, v38 // (op 1338) 
v_add_f32 v55, v23, v39 // (op 1338) 
v_add_f32 v56, v24, v40 // (op 1338) 
v_add_f32 v57, v25, v41 // (op 1338) 
v_add_f32 v58, v26, v42 // (op 1338) 
v_add_f32 v59, v27, v43 // (op 1338) 
v_add_f32 v60, v28, v44 // (op 1338) 
v_add_f32 v61, v29, v45 // (op 1338) 
v_add_f32 v62, v30, v46 // (op 1338) 
v_add_f32 v63, v31, v47 // (op 1338) 
v_add_f32 v64, v32, v48 // (op 1338) 
v_add_f32 v65, v33, v49 // (op 1338) 
v_add_f32 v66, v34, v50 // (op 1338) 
v_add_f32 v67, v35, v51 // (op 1338) 
v_add_f32 v68, v36, v52 // (op 1338) 
v_add_f32 v69, v37, v53 // (op 1338) 


v_cvt_f16_f32 v14, v54 // (op 1339) 
v_cvt_f16_f32 v15, v55 // (op 1339) 
v_cvt_f16_f32 v16, v56 // (op 1339) 
v_cvt_f16_f32 v17, v57 // (op 1339) 
v_cvt_f16_f32 v18, v58 // (op 1339) 
v_cvt_f16_f32 v19, v59 // (op 1339) 
v_cvt_f16_f32 v20, v60 // (op 1339) 
v_cvt_f16_f32 v21, v61 // (op 1339) 
v_cvt_f16_f32 v22, v62 // (op 1339) 
v_cvt_f16_f32 v23, v63 // (op 1339) 
v_cvt_f16_f32 v24, v64 // (op 1339) 
v_cvt_f16_f32 v25, v65 // (op 1339) 
v_cvt_f16_f32 v26, v66 // (op 1339) 
v_cvt_f16_f32 v27, v67 // (op 1339) 
v_cvt_f16_f32 v28, v68 // (op 1339) 
v_cvt_f16_f32 v29, v69 // (op 1339) 


v_pack_b32_f16 v30, v14, v15 // (op 1340) 
v_pack_b32_f16 v31, v16, v17 // (op 1340) 
v_pack_b32_f16 v32, v18, v19 // (op 1340) 
v_pack_b32_f16 v33, v20, v21 // (op 1340) 
v_pack_b32_f16 v34, v22, v23 // (op 1340) 
v_pack_b32_f16 v35, v24, v25 // (op 1340) 
v_pack_b32_f16 v36, v26, v27 // (op 1340) 
v_pack_b32_f16 v37, v28, v29 // (op 1340) 

s_lshl_b32 s0, s4, 8 // (op 1340) 
v_add_u32 v0, v11, s0 // (op 1340) 


buffer_store_dwordx2 v[30:31], v0, s[48:51], 0 offen // (op 1340) Store value
buffer_store_dwordx2 v[32:33], v0, s[48:51], 0 offen offset:16 // (op 1340) Store value
buffer_store_dwordx2 v[34:35], v0, s[48:51], 0 offen offset:32 // (op 1340) Store value
buffer_store_dwordx2 v[36:37], v0, s[48:51], 0 offen offset:48 // (op 1340) Store value





s_lshl_b32 s0, s2, 7 // (op 1355) 
s_mul_i32 s1, 3, s0 // (op 1355) 

v_add_u32 v0, v8, s1 // (op 1355) 


buffer_load_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 1355) Load value
buffer_load_dwordx2 v[16:17], v0, s[44:47], 0 offen offset:16 // (op 1355) Load value
v_accvgpr_read v22, a48 // (op 1351) 
v_accvgpr_read v23, a49 // (op 1351) 
v_accvgpr_read v24, a50 // (op 1351) 
v_accvgpr_read v25, a51 // (op 1351) 
v_accvgpr_read v26, a52 // (op 1351) 
v_accvgpr_read v27, a53 // (op 1351) 
v_accvgpr_read v28, a54 // (op 1351) 
v_accvgpr_read v29, a55 // (op 1351) 
v_accvgpr_read v30, a56 // (op 1351) 
v_accvgpr_read v31, a57 // (op 1351) 
v_accvgpr_read v32, a58 // (op 1351) 
v_accvgpr_read v33, a59 // (op 1351) 
v_accvgpr_read v34, a60 // (op 1351) 
v_accvgpr_read v35, a61 // (op 1351) 
v_accvgpr_read v36, a62 // (op 1351) 
v_accvgpr_read v37, a63 // (op 1351) 
v_mul_f32 v38, v13, v22 // (op 1351) 
v_mul_f32 v39, v13, v23 // (op 1351) 
v_mul_f32 v40, v13, v24 // (op 1351) 
v_mul_f32 v41, v13, v25 // (op 1351) 
v_mul_f32 v42, v13, v26 // (op 1351) 
v_mul_f32 v43, v13, v27 // (op 1351) 
v_mul_f32 v44, v13, v28 // (op 1351) 
v_mul_f32 v45, v13, v29 // (op 1351) 
v_mul_f32 v46, v13, v30 // (op 1351) 
v_mul_f32 v47, v13, v31 // (op 1351) 
v_mul_f32 v48, v13, v32 // (op 1351) 
v_mul_f32 v49, v13, v33 // (op 1351) 
v_mul_f32 v50, v13, v34 // (op 1351) 
v_mul_f32 v51, v13, v35 // (op 1351) 
v_mul_f32 v52, v13, v36 // (op 1351) 
v_mul_f32 v53, v13, v37 // (op 1351) 

buffer_load_dwordx2 v[18:19], v0, s[44:47], 0 offen offset:32 // (op 1355) Load value
buffer_load_dwordx2 v[20:21], v0, s[44:47], 0 offen offset:48 // (op 1355) Load value

s_waitcnt vmcnt(3)
v_cvt_f32_f16 v6, v14 // (op 1356) 
v_lshrrev_b32 v7, 16, v14 // (op 1356) 
v_cvt_f32_f16 v7, v7 // (op 1356) 
v_mul_f32 v22, v12, v6 // (op 1356) 
v_mul_f32 v23, v12, v7 // (op 1356) 
v_cvt_f32_f16 v6, v15 // (op 1356) 
v_lshrrev_b32 v7, 16, v15 // (op 1356) 
v_cvt_f32_f16 v7, v7 // (op 1356) 
v_mul_f32 v24, v12, v6 // (op 1356) 
v_mul_f32 v25, v12, v7 // (op 1356) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v6, v16 // (op 1356) 
v_lshrrev_b32 v7, 16, v16 // (op 1356) 
v_cvt_f32_f16 v7, v7 // (op 1356) 
v_mul_f32 v26, v12, v6 // (op 1356) 
v_mul_f32 v27, v12, v7 // (op 1356) 
v_cvt_f32_f16 v6, v17 // (op 1356) 
v_lshrrev_b32 v7, 16, v17 // (op 1356) 
v_cvt_f32_f16 v7, v7 // (op 1356) 
v_mul_f32 v28, v12, v6 // (op 1356) 
v_mul_f32 v29, v12, v7 // (op 1356) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v6, v18 // (op 1356) 
v_lshrrev_b32 v7, 16, v18 // (op 1356) 
v_cvt_f32_f16 v7, v7 // (op 1356) 
v_mul_f32 v30, v12, v6 // (op 1356) 
v_mul_f32 v31, v12, v7 // (op 1356) 
v_cvt_f32_f16 v6, v19 // (op 1356) 
v_lshrrev_b32 v7, 16, v19 // (op 1356) 
v_cvt_f32_f16 v7, v7 // (op 1356) 
v_mul_f32 v32, v12, v6 // (op 1356) 
v_mul_f32 v33, v12, v7 // (op 1356) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v20 // (op 1356) 
v_lshrrev_b32 v7, 16, v20 // (op 1356) 
v_cvt_f32_f16 v7, v7 // (op 1356) 
v_mul_f32 v34, v12, v6 // (op 1356) 
v_mul_f32 v35, v12, v7 // (op 1356) 
v_cvt_f32_f16 v6, v21 // (op 1356) 
v_lshrrev_b32 v7, 16, v21 // (op 1356) 
v_cvt_f32_f16 v7, v7 // (op 1356) 
v_mul_f32 v36, v12, v6 // (op 1356) 
v_mul_f32 v37, v12, v7 // (op 1356) 


v_add_f32 v54, v22, v38 // (op 1352) 
v_add_f32 v55, v23, v39 // (op 1352) 
v_add_f32 v56, v24, v40 // (op 1352) 
v_add_f32 v57, v25, v41 // (op 1352) 
v_add_f32 v58, v26, v42 // (op 1352) 
v_add_f32 v59, v27, v43 // (op 1352) 
v_add_f32 v60, v28, v44 // (op 1352) 
v_add_f32 v61, v29, v45 // (op 1352) 
v_add_f32 v62, v30, v46 // (op 1352) 
v_add_f32 v63, v31, v47 // (op 1352) 
v_add_f32 v64, v32, v48 // (op 1352) 
v_add_f32 v65, v33, v49 // (op 1352) 
v_add_f32 v66, v34, v50 // (op 1352) 
v_add_f32 v67, v35, v51 // (op 1352) 
v_add_f32 v68, v36, v52 // (op 1352) 
v_add_f32 v69, v37, v53 // (op 1352) 


v_cvt_f16_f32 v14, v54 // (op 1353) 
v_cvt_f16_f32 v15, v55 // (op 1353) 
v_cvt_f16_f32 v16, v56 // (op 1353) 
v_cvt_f16_f32 v17, v57 // (op 1353) 
v_cvt_f16_f32 v18, v58 // (op 1353) 
v_cvt_f16_f32 v19, v59 // (op 1353) 
v_cvt_f16_f32 v20, v60 // (op 1353) 
v_cvt_f16_f32 v21, v61 // (op 1353) 
v_cvt_f16_f32 v22, v62 // (op 1353) 
v_cvt_f16_f32 v23, v63 // (op 1353) 
v_cvt_f16_f32 v24, v64 // (op 1353) 
v_cvt_f16_f32 v25, v65 // (op 1353) 
v_cvt_f16_f32 v26, v66 // (op 1353) 
v_cvt_f16_f32 v27, v67 // (op 1353) 
v_cvt_f16_f32 v28, v68 // (op 1353) 
v_cvt_f16_f32 v29, v69 // (op 1353) 


v_pack_b32_f16 v30, v14, v15 // (op 1354) 
v_pack_b32_f16 v31, v16, v17 // (op 1354) 
v_pack_b32_f16 v32, v18, v19 // (op 1354) 
v_pack_b32_f16 v33, v20, v21 // (op 1354) 
v_pack_b32_f16 v34, v22, v23 // (op 1354) 
v_pack_b32_f16 v35, v24, v25 // (op 1354) 
v_pack_b32_f16 v36, v26, v27 // (op 1354) 
v_pack_b32_f16 v37, v28, v29 // (op 1354) 

s_lshl_b32 s0, s4, 7 // (op 1354) 
s_mul_i32 s1, 3, s0 // (op 1354) 

v_add_u32 v0, v11, s1 // (op 1354) 


buffer_store_dwordx2 v[30:31], v0, s[48:51], 0 offen // (op 1354) Store value
buffer_store_dwordx2 v[32:33], v0, s[48:51], 0 offen offset:16 // (op 1354) Store value
buffer_store_dwordx2 v[34:35], v0, s[48:51], 0 offen offset:32 // (op 1354) Store value
buffer_store_dwordx2 v[36:37], v0, s[48:51], 0 offen offset:48 // (op 1354) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_ConditionalBottom_0_beta_157:












s_endpgm  // End of RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.LRR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_end:
.size RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2, .LRR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_end-RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
  .amdhsa_next_free_vgpr 212
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 84480 // lds bytes
  .amdhsa_accum_offset 148
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
    DataTypeB: B6
    DestDataType: h
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2
    .symbol: RR_GEMM_TT_FP8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 84480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 146
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

