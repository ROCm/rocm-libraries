// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.p2align 8
.type RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2,@function
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 2841) 
v_lshrrev_b32 v3, 4, v0 // (op 2841) 

v_and_b32 v0, 1, v3 // (op 2841) 

v_and_b32 v3, 15, v1 // (op 2841) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2841) 


v_lshrrev_b32 v0, 6, v1 // (op 2841) 
v_and_b32 v3, 1, v0 // (op 2841) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 2841) 


v_and_b32 v3, 32, v1 // (op 2841) 
v_lshl_add_u32 v4, v0, 7, v3 // (op 2841) 


v_mul_lo_u32 v0, v4, 6 // (op 2841) 

v_lshrrev_b32 v3, 3, v0 // (op 2841) 

v_lshrrev_b32 v0, 6, v1 // (op 2952) 
v_and_b32 v4, 1, v0 // (op 2952) 

v_and_b32 v0, 31, v1 // (op 2952) 
v_lshl_add_u32 v5, v4, 5, v0 // (op 2952) 


v_and_b32 v0, 63, v1 // (op 2952) 
v_lshrrev_b32 v4, 5, v0 // (op 2952) 

v_lshl_add_u32 v0, v5, 2, v4 // (op 2952) 


v_and_b32 v4, 63, v1 // (op 3023) 
v_lshrrev_b32 v5, 4, v4 // (op 3023) 

v_and_b32 v4, 1, v5 // (op 3023) 

v_and_b32 v5, 15, v1 // (op 3023) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 3023) 


v_lshrrev_b32 v4, 6, v1 // (op 3023) 
v_and_b32 v5, 1, v4 // (op 3023) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 3023) 


v_and_b32 v5, 32, v1 // (op 3023) 
v_lshl_add_u32 v6, v4, 7, v5 // (op 3023) 


v_mul_lo_u32 v4, v6, 6 // (op 3023) 

v_lshrrev_b32 v5, 3, v4 // (op 3023) 

v_lshrrev_b32 v4, 6, v1 // (op 3076) 
v_and_b32 v6, 1, v4 // (op 3076) 

v_and_b32 v4, 31, v1 // (op 3076) 
v_lshl_add_u32 v7, v6, 5, v4 // (op 3076) 


v_and_b32 v4, 63, v1 // (op 3076) 
v_lshrrev_b32 v6, 5, v4 // (op 3076) 

v_lshl_add_u32 v4, v7, 2, v6 // (op 3076) 


v_lshrrev_b32 v6, 7, v1 // (op 2726) 
v_and_b32 v7, 63, v1 // (op 2726) 
v_lshrrev_b32 v8, 5, v7 // (op 2726) 
v_lshlrev_b32 v9, 2, v8 // (op 2726) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2726) 


v_lshl_add_u32 v6, s16, 8, v8 // (op 2726) 

v_lshrrev_b32 v8, 2, v7 // (op 2726) 

v_and_b32 v7, 7, v8 // (op 2726) 

v_and_b32 v8, 3, v1 // (op 2726) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 2726) 


v_lshrrev_b32 v7, 6, v1 // (op 2726) 
v_and_b32 v8, 1, v7 // (op 2726) 

v_lshl_add_u32 v7, v8, 5, v9 // (op 2726) 


v_lshl_add_u32 v8, s17, 6, v7 // (op 2726) 

v_mul_lo_u32 v7, v8, s2 // (op 2726) 

v_add_lshl_u32 v8, v6, v7, 2 // (op 2726) 


v_and_b32 v6, 63, v1 // (op 2788) 
v_and_b32 v7, 48, v1 // (op 2788) 
v_sub_u32 v9, v6, v7 // (op 2788) 

v_lshrrev_b32 v7, 5, v6 // (op 2788) 
s_mul_i32 s1, s2, s9 // (op 2728) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 2728) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 2728) least significant
s_mul_hi_u32 s19, s2, s8 // (op 2728) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2728) most significant: sum
s_add_u32 s1, s1, s19 // (op 2728) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2728) 

v_lshl_add_u32 v10, v7, 5, v9 // (op 2788) 


v_lshrrev_b32 v7, 4, v6 // (op 2788) 

v_and_b32 v6, -2, v7 // (op 2788) 
v_sub_u32 v9, v7, v6 // (op 2788) 


v_lshlrev_b32 v6, 4, v9 // (op 2788) 

v_lshrrev_b32 v7, 7, v1 // (op 2788) 
v_lshl_add_u32 v9, v7, 5, v6 // (op 2788) 


v_lshl_add_u32 v6, v10, 8, v9 // (op 2788) 


v_lshlrev_b32 v7, 2, v6 // (op 2788) 

v_lshrrev_b32 v6, 3, v7 // (op 2788) 

v_lshrrev_b32 v7, 7, v1 // (op 2899) 
v_and_b32 v9, 31, v1 // (op 2899) 
v_lshl_add_u32 v10, v7, 5, v9 // (op 2899) 


v_and_b32 v7, 63, v1 // (op 2899) 
v_lshrrev_b32 v9, 5, v7 // (op 2899) 

v_lshl_add_u32 v7, v9, 8, v10 // (op 2899) 


v_lshrrev_b32 v9, 7, v1 // (op 2968) 
v_and_b32 v10, 63, v1 // (op 2968) 
v_lshrrev_b32 v11, 5, v10 // (op 2968) 
v_lshlrev_b32 v12, 2, v11 // (op 2968) 

v_lshl_add_u32 v11, v9, 5, v12 // (op 2968) 


v_lshl_add_u32 v9, s16, 8, v11 // (op 2968) 

v_lshrrev_b32 v11, 2, v10 // (op 2968) 

v_and_b32 v10, 7, v11 // (op 2968) 

v_and_b32 v11, 3, v1 // (op 2968) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 2968) 


v_lshrrev_b32 v10, 6, v1 // (op 2968) 
v_and_b32 v11, 1, v10 // (op 2968) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 2968) 


v_lshl_add_u32 v11, s17, 6, v10 // (op 2968) 

v_mul_lo_u32 v10, v11, s4 // (op 2968) 

v_add_lshl_u32 v11, v9, v10, 2 // (op 2968) 


v_and_b32 v9, 63, v1 // (op 2999) 
v_and_b32 v10, 48, v1 // (op 2999) 
v_sub_u32 v12, v9, v10 // (op 2999) 

v_lshrrev_b32 v10, 5, v9 // (op 2999) 
s_mul_i32 s1, s4, s9 // (op 2970) most significant: low of low * high
v_lshl_add_u32 v13, v10, 5, v12 // (op 2999) 


v_lshrrev_b32 v10, 4, v9 // (op 2999) 

v_and_b32 v9, -2, v10 // (op 2999) 
v_sub_u32 v12, v10, v9 // (op 2999) 


v_lshlrev_b32 v9, 4, v12 // (op 2999) 

v_lshrrev_b32 v10, 7, v1 // (op 2999) 
v_lshl_add_u32 v12, v10, 5, v9 // (op 2999) 


v_lshl_add_u32 v9, v13, 8, v12 // (op 2999) 


v_lshlrev_b32 v10, 2, v9 // (op 2999) 

v_lshrrev_b32 v9, 3, v10 // (op 2999) 

v_lshrrev_b32 v10, 7, v1 // (op 3052) 
v_and_b32 v12, 31, v1 // (op 3052) 
v_lshl_add_u32 v13, v10, 5, v12 // (op 3052) 


v_and_b32 v10, 63, v1 // (op 3052) 
v_lshrrev_b32 v12, 5, v10 // (op 3052) 

v_lshl_add_u32 v10, v12, 8, v13 // (op 3052) 


s_mul_i32 s38, s5, s8 // (op 2970) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2970) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2970) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2970) most significant: sum
s_add_u32 s1, s1, s39 // (op 2970) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2970) 

s_mov_b32 s40, s6 // (op 2728) reg expression
s_mov_b32 s41, s7 // (op 2728) reg expression
s_mov_b32 s42, s18 // (op 2728) reg expression
s_mov_b32 s43, 131072 // (op 2728) call()

s_mov_b32 s44, s10 // (op 2970) reg expression

s_mov_b32 s45, s11 // (op 2970) reg expression
s_mov_b32 s46, s38 // (op 2970) reg expression
s_mov_b32 s47, 131072 // (op 2970) call()


v_mov_b32 v12, s14 // (op 2) Move value
v_mov_b32 v13, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 63) call()
v_accvgpr_write a5, 0.00000 // (op 63) call()
v_accvgpr_write a6, 0.00000 // (op 63) call()
v_accvgpr_write a7, 0.00000 // (op 63) call()
v_accvgpr_write a8, 0.00000 // (op 63) call()
v_accvgpr_write a9, 0.00000 // (op 63) call()
v_accvgpr_write a10, 0.00000 // (op 63) call()
v_accvgpr_write a11, 0.00000 // (op 63) call()
v_accvgpr_write a12, 0.00000 // (op 63) call()
v_accvgpr_write a13, 0.00000 // (op 63) call()
v_accvgpr_write a14, 0.00000 // (op 63) call()
v_accvgpr_write a15, 0.00000 // (op 63) call()
v_accvgpr_write a16, 0.00000 // (op 401) call()
v_accvgpr_write a17, 0.00000 // (op 401) call()
v_accvgpr_write a18, 0.00000 // (op 401) call()
v_accvgpr_write a19, 0.00000 // (op 401) call()
v_accvgpr_write a20, 0.00000 // (op 401) call()
v_accvgpr_write a21, 0.00000 // (op 401) call()
v_accvgpr_write a22, 0.00000 // (op 401) call()
v_accvgpr_write a23, 0.00000 // (op 401) call()
v_accvgpr_write a24, 0.00000 // (op 401) call()
v_accvgpr_write a25, 0.00000 // (op 401) call()
v_accvgpr_write a26, 0.00000 // (op 401) call()
v_accvgpr_write a27, 0.00000 // (op 401) call()
v_accvgpr_write a28, 0.00000 // (op 401) call()
v_accvgpr_write a29, 0.00000 // (op 401) call()
v_accvgpr_write a30, 0.00000 // (op 401) call()
v_accvgpr_write a31, 0.00000 // (op 401) call()
v_accvgpr_write a32, 0.00000 // (op 641) call()
v_accvgpr_write a33, 0.00000 // (op 641) call()
v_accvgpr_write a34, 0.00000 // (op 641) call()
v_accvgpr_write a35, 0.00000 // (op 641) call()
v_accvgpr_write a36, 0.00000 // (op 641) call()
v_accvgpr_write a37, 0.00000 // (op 641) call()
v_accvgpr_write a38, 0.00000 // (op 641) call()
v_accvgpr_write a39, 0.00000 // (op 641) call()
v_accvgpr_write a40, 0.00000 // (op 641) call()
v_accvgpr_write a41, 0.00000 // (op 641) call()
v_accvgpr_write a42, 0.00000 // (op 641) call()
v_accvgpr_write a43, 0.00000 // (op 641) call()
v_accvgpr_write a44, 0.00000 // (op 641) call()
v_accvgpr_write a45, 0.00000 // (op 641) call()
v_accvgpr_write a46, 0.00000 // (op 641) call()
v_accvgpr_write a47, 0.00000 // (op 641) call()
v_accvgpr_write a48, 0.00000 // (op 881) call()
v_accvgpr_write a49, 0.00000 // (op 881) call()
v_accvgpr_write a50, 0.00000 // (op 881) call()
v_accvgpr_write a51, 0.00000 // (op 881) call()
v_accvgpr_write a52, 0.00000 // (op 881) call()
v_accvgpr_write a53, 0.00000 // (op 881) call()
v_accvgpr_write a54, 0.00000 // (op 881) call()
v_accvgpr_write a55, 0.00000 // (op 881) call()
v_accvgpr_write a56, 0.00000 // (op 881) call()
v_accvgpr_write a57, 0.00000 // (op 881) call()
v_accvgpr_write a58, 0.00000 // (op 881) call()
v_accvgpr_write a59, 0.00000 // (op 881) call()
v_accvgpr_write a60, 0.00000 // (op 881) call()
v_accvgpr_write a61, 0.00000 // (op 881) call()
v_accvgpr_write a62, 0.00000 // (op 881) call()
v_accvgpr_write a63, 0.00000 // (op 881) call()
v_and_b32 v14, -8, v1 // (op 2493) 
v_sub_u32 v15, v1, v14 // (op 2493) 

v_lshlrev_b32 v14, 5, v15 // (op 2493) 

v_lshl_add_u32 v15, s16, 8, v14 // (op 2493) 

v_mov_b32 v17, 0 // (op 2493) convert
v_mov_b32 v16, v15 // (op 2493) convert

v_lshrrev_b32 v14, 3, v1 // (op 2493) 
v_lshlrev_b32 v15, 2, v14 // (op 2493) 

v_mul_lo_u32 v19, v15, s13 // (op 2493) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 2493) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 2493) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2493) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 2493) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 2493) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 2493) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 2493) 

v_mov_b32 v14, v16 // (op 2493) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 2495) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2495) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2495) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2495) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2495) most significant: sum
s_add_u32 s1, s1, s7 // (op 2495) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2495) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2495) 

s_mov_b32 s48, s20 // (op 2495) reg expression
s_mov_b32 s49, s21 // (op 2495) reg expression
s_mov_b32 s50, s0 // (op 2495) reg expression
s_mov_b32 s51, 131072 // (op 2495) call()

v_and_b32 v15, -8, v1 // (op 2512) 
v_sub_u32 v16, v1, v15 // (op 2512) 

v_lshlrev_b32 v15, 5, v16 // (op 2512) 

v_lshl_add_u32 v16, s16, 8, v15 // (op 2512) 

v_mov_b32 v19, 0 // (op 2512) convert
v_mov_b32 v18, v16 // (op 2512) convert

v_lshrrev_b32 v15, 3, v1 // (op 2512) 
v_lshlrev_b32 v16, 2, v15 // (op 2512) 

v_mul_lo_u32 v21, v16, s13 // (op 2512) most significant: low of low * high
v_mul_hi_u32 v15, v16, s12 // (op 2512) most significant: high of low * low
v_mul_lo_u32 v20, v16, s12 // (op 2512) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 2512) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 2512) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 2512) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 2512) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 2512) 

v_mov_b32 v15, v18 // (op 2512) convert

s_mul_i32 s1, s12, s23 // (op 2514) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2514) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2514) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2514) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2514) most significant: sum
s_add_u32 s1, s1, s7 // (op 2514) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2514) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2514) 

s_mov_b32 s48, s20 // (op 2514) reg expression
s_mov_b32 s49, s21 // (op 2514) reg expression
s_mov_b32 s50, s0 // (op 2514) reg expression
s_mov_b32 s51, 131072 // (op 2514) call()

v_and_b32 v16, -4, v1 // (op 2557) 
v_sub_u32 v17, v1, v16 // (op 2557) 

v_lshlrev_b32 v16, 5, v17 // (op 2557) 

v_mov_b32 v19, 0 // (op 2557) convert
v_mov_b32 v18, v16 // (op 2557) convert

v_lshrrev_b32 v16, 2, v1 // (op 2557) 
v_lshl_add_u32 v17, s17, 6, v16 // (op 2557) 

v_mul_lo_u32 v21, v17, s25 // (op 2557) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 2557) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 2557) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2557) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 2557) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 2557) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 2557) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 2557) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 2557) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 2557) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 2557) 

v_mov_b32 v18, v16 // (op 2557) convert

s_mul_i32 s1, s9, 6 // (op 2559) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2559) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2559) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2559) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2559) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2559) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2559) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2559) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2559) most significant: sum
s_add_u32 s7, s7, s11 // (op 2559) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2559) 

s_mov_b32 s52, s26 // (op 2559) reg expression
s_mov_b32 s53, s27 // (op 2559) reg expression
s_mov_b32 s54, s0 // (op 2559) reg expression
s_mov_b32 s55, 131072 // (op 2559) call()

v_and_b32 v16, -4, v1 // (op 2576) 
v_sub_u32 v17, v1, v16 // (op 2576) 

v_lshlrev_b32 v16, 5, v17 // (op 2576) 

v_mov_b32 v21, 0 // (op 2576) convert
v_mov_b32 v20, v16 // (op 2576) convert

v_lshrrev_b32 v19, 2, v1 // (op 2576) 
v_lshl_add_u32 v16, s17, 6, v19 // (op 2576) 

v_mul_lo_u32 v23, v16, s25 // (op 2576) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 2576) most significant: high of low * low
v_mul_lo_u32 v22, v16, s24 // (op 2576) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 2576) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 2576) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 2576) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 2576) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 2576) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 2576) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 2576) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 2576) 

v_mov_b32 v19, v16 // (op 2576) convert

s_mul_i32 s1, s9, 6 // (op 2578) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2578) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2578) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2578) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2578) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2578) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2578) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2578) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2578) most significant: sum
s_add_u32 s7, s7, s11 // (op 2578) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2578) 

s_mov_b32 s52, s26 // (op 2578) reg expression
s_mov_b32 s53, s27 // (op 2578) reg expression
s_mov_b32 s54, s0 // (op 2578) reg expression
s_mov_b32 s55, 131072 // (op 2578) call()

v_and_b32 v16, -64, v1 // (op 2621) 
v_sub_u32 v17, v1, v16 // (op 2621) 

v_lshlrev_b32 v16, 2, v17 // (op 2621) 

v_lshl_add_u32 v17, s16, 8, v16 // (op 2621) 

v_lshrrev_b32 v16, 6, v1 // (op 2621) 
v_mul_lo_u32 v20, v16, s28 // (op 2621) 

v_add_u32 v16, v17, v20 // (op 2621) 


s_mul_i32 s1, s28, s33 // (op 2623) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2623) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2623) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2623) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2623) most significant: sum
s_add_u32 s1, s1, s7 // (op 2623) most significant: sum


s_mov_b32 s56, s30 // (op 2623) reg expression
s_mov_b32 s57, s31 // (op 2623) reg expression
s_mov_b32 s58, s0 // (op 2623) reg expression
s_mov_b32 s59, 131072 // (op 2623) call()

v_and_b32 v17, -64, v1 // (op 2640) 
v_sub_u32 v20, v1, v17 // (op 2640) 

v_lshlrev_b32 v17, 2, v20 // (op 2640) 

v_lshl_add_u32 v20, s16, 8, v17 // (op 2640) 

v_lshrrev_b32 v17, 6, v1 // (op 2640) 
v_mul_lo_u32 v21, v17, s28 // (op 2640) 

v_add_u32 v17, v20, v21 // (op 2640) 


s_mul_i32 s1, s28, s33 // (op 2642) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2642) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2642) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2642) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2642) most significant: sum
s_add_u32 s1, s1, s7 // (op 2642) most significant: sum


s_mov_b32 s56, s30 // (op 2642) reg expression
s_mov_b32 s57, s31 // (op 2642) reg expression
s_mov_b32 s58, s0 // (op 2642) reg expression
s_mov_b32 s59, 131072 // (op 2642) call()

v_and_b32 v20, -4, v1 // (op 2685) 
v_sub_u32 v21, v1, v20 // (op 2685) 

v_lshrrev_b32 v20, 2, v1 // (op 2685) 
v_lshl_add_u32 v22, s17, 6, v20 // (op 2685) 

v_mul_lo_u32 v20, v22, s34 // (op 2685) 

v_add_u32 v22, v21, v20 // (op 2685) 


s_mul_i32 s1, s34, s9 // (op 2687) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2687) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2687) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2687) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2687) most significant: sum
s_add_u32 s1, s1, s7 // (op 2687) most significant: sum


s_mov_b32 s60, s36 // (op 2687) reg expression
s_mov_b32 s61, s37 // (op 2687) reg expression
s_mov_b32 s62, s0 // (op 2687) reg expression
s_mov_b32 s63, 131072 // (op 2687) call()

v_and_b32 v20, -4, v1 // (op 2704) 
v_sub_u32 v21, v1, v20 // (op 2704) 

v_lshrrev_b32 v20, 2, v1 // (op 2704) 
v_lshl_add_u32 v23, s17, 6, v20 // (op 2704) 

v_mul_lo_u32 v20, v23, s34 // (op 2704) 

v_add_u32 v23, v21, v20 // (op 2704) 


s_mul_i32 s1, s34, s9 // (op 2706) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2706) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2706) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2706) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2706) most significant: sum
s_add_u32 s1, s1, s7 // (op 2706) most significant: sum


s_mov_b32 s60, s36 // (op 2706) reg expression
s_mov_b32 s61, s37 // (op 2706) reg expression
s_mov_b32 s62, s0 // (op 2706) reg expression
s_mov_b32 s63, 131072 // (op 2706) call()




buffer_load_dwordx4 v[24:27], v18, s[52:55], 0 offen // (op 2231) Load value
buffer_load_dwordx2 v[28:29], v18, s[52:55], 0 offen offset:16 // (op 2231) Load value



buffer_load_ubyte v20, v22, s[60:63], 0 offen offset: 0 // (op 2234) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 2237) 
v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 2237) 


v_mov_b32 v21, v14 // (op 2237) reg expression
buffer_load_dwordx4 v[32:35], v21, s[48:51], 0 offen // (op 2237) Load value
v_add_u32 v21, v21, v30 // (op 2237) 
buffer_load_dwordx4 v[36:39], v21, s[48:51], 0 offen // (op 2237) Load value
v_add_u32 v21, v21, v30 // (op 2237) 
buffer_load_dwordx4 v[40:43], v21, s[48:51], 0 offen // (op 2237) Load value
v_add_u32 v21, v21, v30 // (op 2237) 
buffer_load_dwordx4 v[44:47], v21, s[48:51], 0 offen // (op 2237) Load value




v_mov_b32 v21, v16 // (op 2240) reg expression

buffer_load_dword v30, v21, s[56:59], 0 offen // (op 2240) Load value




buffer_load_dwordx4 v[48:51], v18, s[52:55], 0 offen offset:96 // (op 2243) Load value
buffer_load_dwordx2 v[52:53], v18, s[52:55], 0 offen offset:112 // (op 2243) Load value




buffer_load_ubyte v18, v22, s[60:63], 0 offen offset:4 // (op 2246) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 2249) 
v_lshrrev_b64 v[54:55], 3, s[0:1] // (op 2249) 


s_lshl_b64 s[0:1], s[12:13], 7 // (op 2249) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 2249) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 2249) 

v_add_u32 v31, v14, s0 // (op 2249) 

buffer_load_dwordx4 v[56:59], v31, s[48:51], 0 offen // (op 2249) Load value
v_add_u32 v31, v31, v54 // (op 2249) 
buffer_load_dwordx4 v[60:63], v31, s[48:51], 0 offen // (op 2249) Load value
v_add_u32 v31, v31, v54 // (op 2249) 
buffer_load_dwordx4 v[64:67], v31, s[48:51], 0 offen // (op 2249) Load value
v_add_u32 v31, v31, v54 // (op 2249) 
buffer_load_dwordx4 v[68:71], v31, s[48:51], 0 offen // (op 2249) Load value





s_lshl_b32 s0, s28, 2 // (op 2252) 
v_add_u32 v14, v16, s0 // (op 2252) 


buffer_load_dword v31, v14, s[56:59], 0 offen // (op 2252) Load value


v_and_b32 v14, -4, v1 // (op 2829) 
v_sub_u32 v16, v1, v14 // (op 2829) 

v_lshlrev_b32 v14, 5, v16 // (op 2829) 

v_lshrrev_b32 v16, 2, v1 // (op 2829) 
v_lshl_add_u32 v21, v16, 7, v14 // (op 2829) 


v_mul_lo_u32 v14, v21, 6 // (op 2829) 

v_lshrrev_b32 v16, 3, v14 // (op 2829) 




s_waitcnt vmcnt(15)
ds_write_b128 v16, v[24:27] // (op 2254) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v16, v[28:29] offset:16 // (op 2254) Store local data 

v_and_b32 v14, -4, v1 // (op 2940) 
v_sub_u32 v16, v1, v14 // (op 2940) 

v_and_b32 v14, 4294967292, v1 // (op 2940) 
v_add_u32 v21, v14, v16 // (op 2940) 





s_waitcnt vmcnt(13)
ds_write_b8 v21, v20 offset:6144 // (op 2255) Store local data 

v_and_b32 v14, -8, v1 // (op 2771) 
v_sub_u32 v16, v1, v14 // (op 2771) 

v_lshlrev_b32 v14, 5, v16 // (op 2771) 

v_lshrrev_b32 v16, 3, v1 // (op 2771) 
v_lshl_add_u32 v21, v16, 10, v14 // (op 2771) 


v_lshlrev_b32 v14, 2, v21 // (op 2771) 

v_lshrrev_b32 v16, 3, v14 // (op 2771) 




s_waitcnt vmcnt(12)
ds_write_b128 v16, v[32:35] offset:6400 // (op 2256) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v16, v[36:39] offset:6528 // (op 2256) Store local data 
s_waitcnt vmcnt(10)
ds_write_b128 v16, v[40:43] offset:6656 // (op 2256) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v16, v[44:47] offset:6784 // (op 2256) Store local data 

v_and_b32 v14, -64, v1 // (op 2882) 
v_sub_u32 v16, v1, v14 // (op 2882) 

v_lshlrev_b32 v14, 2, v16 // (op 2882) 

v_lshrrev_b32 v16, 6, v1 // (op 2882) 
v_lshl_add_u32 v21, v16, 8, v14 // (op 2882) 





s_waitcnt vmcnt(8)
ds_write_b32 v21, v30 offset:22784 // (op 2257) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 3158) 



ds_read_b128 v[72:75], v3 // (op 2291) Load local data 
ds_read_b64 v[76:77], v3 offset:16 // (op 2291) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 3199) 



ds_read_u8 v14, v0 offset:6144 // (op 2314) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3139) 



ds_read_b64_tr_b4 v[78:79], v6 offset:6400 // (op 2272) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v6 offset:8448 // (op 2272) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 3179) 


ds_read_u8 v16, v7 offset:22784 // (op 2295) Load local data 



ds_read_b64_tr_b4 v[82:83], v6 offset:6432 // (op 2277) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v6 offset:8480 // (op 2277) Transpose load local data 


ds_read_u8 v21, v7 offset:22848 // (op 2300) Load local data 



ds_read_b64_tr_b4 v[86:87], v6 offset:6464 // (op 2282) Transpose load local data 
ds_read_b64_tr_b4 v[88:89], v6 offset:8512 // (op 2282) Transpose load local data 


ds_read_u8 v22, v7 offset:22912 // (op 2305) Load local data 



ds_read_b64_tr_b4 v[90:91], v6 offset:6496 // (op 2287) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v6 offset:8544 // (op 2287) Transpose load local data 


ds_read_u8 v54, v7 offset:22976 // (op 2310) Load local data 
v_and_b32 v55, -8, v1 // (op 2742) 
v_sub_u32 v94, v1, v55 // (op 2742) 

v_lshlrev_b32 v55, 5, v94 // (op 2742) 

v_lshrrev_b32 v94, 3, v1 // (op 2742) 
v_lshl_add_u32 v95, v94, 10, v55 // (op 2742) 


v_lshlrev_b32 v55, 2, v95 // (op 2742) 

v_lshrrev_b32 v94, 3, v55 // (op 2742) 

v_and_b32 v55, -4, v1 // (op 2800) 
v_sub_u32 v95, v1, v55 // (op 2800) 

v_lshlrev_b32 v55, 5, v95 // (op 2800) 

v_lshrrev_b32 v95, 2, v1 // (op 2800) 
v_lshl_add_u32 v96, v95, 7, v55 // (op 2800) 


v_mul_lo_u32 v55, v96, 6 // (op 2800) 

v_lshrrev_b32 v95, 3, v55 // (op 2800) 

v_and_b32 v55, -64, v1 // (op 2853) 
v_sub_u32 v96, v1, v55 // (op 2853) 

v_lshlrev_b32 v55, 2, v96 // (op 2853) 

v_lshrrev_b32 v96, 6, v1 // (op 2853) 
v_lshl_add_u32 v97, v96, 8, v55 // (op 2853) 


v_and_b32 v55, -4, v1 // (op 2911) 
v_sub_u32 v96, v1, v55 // (op 2911) 

v_and_b32 v55, 4294967292, v1 // (op 2911) 
v_add_u32 v98, v55, v96 // (op 2911) 


v_and_b32 v55, -8, v1 // (op 2982) 
v_sub_u32 v96, v1, v55 // (op 2982) 

v_lshlrev_b32 v55, 5, v96 // (op 2982) 

v_lshrrev_b32 v96, 3, v1 // (op 2982) 
v_lshl_add_u32 v99, v96, 10, v55 // (op 2982) 


v_lshlrev_b32 v55, 2, v99 // (op 2982) 

v_lshrrev_b32 v96, 3, v55 // (op 2982) 

v_and_b32 v55, -4, v1 // (op 3011) 
v_sub_u32 v99, v1, v55 // (op 3011) 

v_lshlrev_b32 v55, 5, v99 // (op 3011) 

v_lshrrev_b32 v99, 2, v1 // (op 3011) 
v_lshl_add_u32 v100, v99, 7, v55 // (op 3011) 


v_mul_lo_u32 v55, v100, 6 // (op 3011) 

v_lshrrev_b32 v99, 3, v55 // (op 3011) 

v_and_b32 v55, -64, v1 // (op 3035) 
v_sub_u32 v100, v1, v55 // (op 3035) 

v_lshlrev_b32 v55, 2, v100 // (op 3035) 

v_lshrrev_b32 v100, 6, v1 // (op 3035) 
v_lshl_add_u32 v101, v100, 8, v55 // (op 3035) 


v_and_b32 v55, -4, v1 // (op 3064) 
v_sub_u32 v100, v1, v55 // (op 3064) 

v_and_b32 v55, 4294967292, v1 // (op 3064) 
v_add_u32 v102, v55, v100 // (op 3064) 


s_mov_b32 s0, 0 // (op 883) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 882) 
s_mov_b32 s1, s6 // (op 882) convert

s_and_b32 s6, s1, 4294967294 // (op 882) 

s_cmp_lt_u32 s0, s6 // (op 882) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_882 // (op 882) Condition: Top (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_882 if false)
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_882:


s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[78:81], v[72:77], a[0:15], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 88) 
s_add_u32 s0, s0, 2 // (op 884) 
v_mov_b32 v55, v19 // (op 10) 

buffer_load_dwordx4 v[24:27], v55, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[28:29], v55, s[52:55], 0 offen offset:208 // (op 10) Load value
s_waitcnt lgkmcnt(1)
s_barrier  // (op 3148) 





ds_read_b64_tr_b4 v[78:79], v6 offset:14592 // (op 91) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v6 offset:16640 // (op 91) Transpose load local data 
v_mov_b32 v100, v23 // (op 25) 

s_barrier  // (op 3168) 
buffer_load_ubyte v20, v100, s[60:63], 0 offen offset:8 // (op 25) Load value




s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 
v_lshrrev_b64 v[104:105], 3, s[6:7] // (op 4) 


v_mov_b32 v103, v15 // (op 4) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 4) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v106, v103, s1 // (op 4) 

ds_read_b128 v[108:111], v3 offset:48 // (op 95) Load local data 
ds_read_b64 v[112:113], v3 offset:64 // (op 95) Load local data 
buffer_load_dwordx4 v[32:35], v106, s[48:51], 0 offen // (op 4) Load value
v_add_u32 v106, v106, v104 // (op 4) 
buffer_load_dwordx4 v[36:39], v106, s[48:51], 0 offen // (op 4) Load value
v_add_u32 v106, v106, v104 // (op 4) 
buffer_load_dwordx4 v[40:43], v106, s[48:51], 0 offen // (op 4) Load value
v_add_u32 v106, v106, v104 // (op 4) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3189) 
buffer_load_dwordx4 v[44:47], v106, s[48:51], 0 offen // (op 4) Load value






ds_read_u8 v16, v7 offset:23296 // (op 99) Load local data 
v_mov_b32 v104, v17 // (op 17) 
s_barrier  // (op 3209) 
s_lshl_b32 s1, s28, 3 // (op 17) 



ds_read_u8 v105, v0 offset:6146 // (op 103) Load local data 
v_add_u32 v106, v104, s1 // (op 17) 


buffer_load_dword v30, v106, s[56:59], 0 offen // (op 17) Load value

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[78:81], v[108:113], a[0:15], v16, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 105) 
s_barrier  // (op 3229) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[82:85], v[72:77], a[16:31], v21, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 450) 
s_waitcnt vmcnt(15)
ds_write_b128 v99, v[48:51] offset:23808 // (op 302) Store local data 



ds_read_b64_tr_b4 v[78:79], v6 offset:14624 // (op 452) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v6 offset:16672 // (op 452) Transpose load local data 
s_waitcnt vmcnt(14)
ds_write_b64 v99, v[52:53] offset:23824 // (op 302) Store local data 


ds_read_u8 v16, v7 offset:23360 // (op 456) Load local data 
s_barrier  // (op 3241) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[78:81], v[108:113], a[16:31], v16, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 459) 



s_waitcnt vmcnt(13)
ds_write_b8 v102, v18 offset:29952 // (op 304) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[86:89], v[72:77], a[32:47], v22, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 690) 



s_barrier  // (op 3222) 
ds_read_b64_tr_b4 v[78:79], v6 offset:14656 // (op 692) Transpose load local data 



ds_read_b64_tr_b4 v[80:81], v6 offset:16704 // (op 692) Transpose load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v96, v[56:59] offset:30208 // (op 283) Store local data 


ds_read_u8 v16, v7 offset:23424 // (op 696) Load local data 
s_waitcnt vmcnt(11)
ds_write_b128 v96, v[60:63] offset:30336 // (op 283) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[78:81], v[108:113], a[32:47], v16, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 699) 
s_waitcnt vmcnt(10)
ds_write_b128 v96, v[64:67] offset:30464 // (op 283) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v96, v[68:71] offset:30592 // (op 283) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[90:93], v[72:77], a[48:63], v54, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 930) 



s_barrier  // (op 3235) 
ds_read_b64_tr_b4 v[90:91], v6 offset:14688 // (op 932) Transpose load local data 



ds_read_b64_tr_b4 v[92:93], v6 offset:16736 // (op 932) Transpose load local data 
s_waitcnt vmcnt(8)
ds_write_b32 v101, v31 offset:46592 // (op 303) Store local data 


ds_read_u8 v54, v7 offset:23488 // (op 936) Load local data 
s_barrier  // (op 3225) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[90:93], v[108:113], a[48:63], v54, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 939) 





ds_read_b128 v[72:75], v5 offset:23808 // (op 287) Load local data 
ds_read_b64 v[76:77], v5 offset:23824 // (op 287) Load local data 
s_barrier  // (op 3238) 



ds_read_u8 v14, v4 offset:29952 // (op 291) Load local data 
s_barrier  // (op 3219) 



ds_read_b64_tr_b4 v[78:79], v9 offset:30208 // (op 285) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v9 offset:32256 // (op 285) Transpose load local data 
s_barrier  // (op 3232) 


ds_read_u8 v16, v10 offset:46592 // (op 289) Load local data 



ds_read_b64_tr_b4 v[82:83], v9 offset:30240 // (op 471) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v9 offset:32288 // (op 471) Transpose load local data 


ds_read_u8 v21, v10 offset:46656 // (op 475) Load local data 



ds_read_b64_tr_b4 v[86:87], v9 offset:30272 // (op 711) Transpose load local data 
ds_read_b64_tr_b4 v[88:89], v9 offset:32320 // (op 711) Transpose load local data 


ds_read_u8 v22, v10 offset:46720 // (op 715) Load local data 



ds_read_b64_tr_b4 v[106:107], v9 offset:30304 // (op 951) Transpose load local data 
ds_read_b64_tr_b4 v[108:109], v9 offset:32352 // (op 951) Transpose load local data 


ds_read_u8 v105, v10 offset:46784 // (op 955) Load local data 



s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[78:81], v[72:77], a[0:15], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 292) 
buffer_load_dwordx4 v[48:51], v55, s[52:55], 0 offen offset:288 // (op 305) Load value
buffer_load_dwordx2 v[52:53], v55, s[52:55], 0 offen offset:304 // (op 305) Load value




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[82:85], v[72:77], a[16:31], v21, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 460) 



buffer_load_ubyte v18, v100, s[60:63], 0 offen offset:12 // (op 307) Load value

v_add_u32 v19, 192, v19 // (op 2568) 

v_add_u32 v23, 8, v23 // (op 2696) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 282) 
v_lshrrev_b64 v[110:111], 3, s[6:7] // (op 282) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 282) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 282) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 282) 

s_mul_i32 s1, 3, s6 // (op 282) 

v_add_u32 v55, v103, s1 // (op 282) 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[86:89], v[72:77], a[32:47], v22, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 700) 
ds_read_b64_tr_b4 v[112:113], v9 offset:38400 // (op 294) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v9 offset:40448 // (op 294) Transpose load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[106:109], v[72:77], a[48:63], v105, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 940) 
ds_read_b128 v[116:119], v5 offset:23856 // (op 296) Load local data 


ds_read_b64 v[120:121], v5 offset:23872 // (op 296) Load local data 
buffer_load_dwordx4 v[56:59], v55, s[48:51], 0 offen // (op 282) Load value
v_add_u32 v55, v55, v110 // (op 282) 


s_barrier  // (op 3286) 
buffer_load_dwordx4 v[60:63], v55, s[48:51], 0 offen // (op 282) Load value
v_add_u32 v55, v55, v110 // (op 282) 
ds_read_u8 v100, v10 offset:47104 // (op 298) Load local data 
buffer_load_dwordx4 v[64:67], v55, s[48:51], 0 offen // (op 282) Load value



v_add_u32 v55, v55, v110 // (op 282) 
ds_read_u8 v105, v4 offset:29954 // (op 300) Load local data 
buffer_load_dwordx4 v[68:71], v55, s[48:51], 0 offen // (op 282) Load value








s_barrier  // (op 3290) 
s_lshl_b32 s1, s28, 2 // (op 306) 
s_mul_i32 s6, 3, s1 // (op 306) 

v_add_u32 v55, v104, s6 // (op 306) 


buffer_load_dword v31, v55, s[56:59], 0 offen // (op 306) Load value

s_lshl_b64 s[6:7], s[12:13], 7 // (op 2504) 

s_lshl_b64 s[10:11], s[6:7], 2 // (op 2504) 

s_barrier  // (op 3161) 
s_lshl_b32 s1, s28, 3 // (op 2632) 



v_add_u32 v17, v17, s1 // (op 2632) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 2504) 

s_lshl_b32 s1, s6, 1 // (op 2504) 

v_add_u32 v15, v15, s1 // (op 2504) 

ds_read_b64_tr_b4 v[106:107], v9 offset:38432 // (op 462) Transpose load local data 
ds_read_b64_tr_b4 v[108:109], v9 offset:40480 // (op 462) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[112:115], v[116:121], a[0:15], v100, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 301) 




ds_read_u8 v55, v10 offset:47168 // (op 466) Load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v95, v[24:27] // (op 13) Store local data 



ds_read_b64_tr_b4 v[110:111], v9 offset:38464 // (op 702) Transpose load local data 
ds_read_b64_tr_b4 v[112:113], v9 offset:40512 // (op 702) Transpose load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[106:109], v[116:121], a[16:31], v55, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 469) 




ds_read_u8 v55, v10 offset:47232 // (op 706) Load local data 
s_waitcnt vmcnt(14)
ds_write_b64 v95, v[28:29] offset:16 // (op 13) Store local data 



ds_read_b64_tr_b4 v[106:107], v9 offset:38496 // (op 942) Transpose load local data 
s_barrier  // (op 3202) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[110:113], v[116:121], a[32:47], v55, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 709) 





ds_read_b64_tr_b4 v[108:109], v9 offset:40544 // (op 942) Transpose load local data 
s_waitcnt vmcnt(13)
ds_write_b8 v98, v20 offset:6144 // (op 30) Store local data 


s_barrier  // (op 3346) 
s_barrier  // (op 3142) 
ds_read_u8 v55, v10 offset:47296 // (op 946) Load local data 



s_waitcnt vmcnt(12)
ds_write_b128 v94, v[32:35] offset:6400 // (op 6) Store local data 
s_barrier  // (op 3350) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[106:109], v[116:121], a[48:63], v55, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 949) 
s_waitcnt vmcnt(11)
ds_write_b128 v94, v[36:39] offset:6528 // (op 6) Store local data 




s_waitcnt vmcnt(10)
ds_write_b128 v94, v[40:43] offset:6656 // (op 6) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v94, v[44:47] offset:6784 // (op 6) Store local data 
s_barrier  // (op 3182) 



s_waitcnt vmcnt(8)
ds_write_b32 v97, v30 offset:22784 // (op 21) Store local data 
s_barrier  // (op 3171) 



ds_read_b128 v[72:75], v3 // (op 75) Load local data 
ds_read_b64 v[76:77], v3 offset:16 // (op 75) Load local data 
s_barrier  // (op 3212) 



ds_read_u8 v14, v0 offset:6144 // (op 85) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3151) 



ds_read_b64_tr_b4 v[78:79], v6 offset:6400 // (op 70) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v6 offset:8448 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 3192) 


ds_read_u8 v16, v7 offset:22784 // (op 80) Load local data 



ds_read_b64_tr_b4 v[82:83], v6 offset:6432 // (op 443) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v6 offset:8480 // (op 443) Transpose load local data 


ds_read_u8 v21, v7 offset:22848 // (op 447) Load local data 



ds_read_b64_tr_b4 v[86:87], v6 offset:6464 // (op 683) Transpose load local data 
ds_read_b64_tr_b4 v[88:89], v6 offset:8512 // (op 683) Transpose load local data 


ds_read_u8 v22, v7 offset:22912 // (op 687) Load local data 



ds_read_b64_tr_b4 v[90:91], v6 offset:6496 // (op 923) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v6 offset:8544 // (op 923) Transpose load local data 


ds_read_u8 v54, v7 offset:22976 // (op 927) Load local data 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 882) 
s_mov_b32 s1, s6 // (op 882) convert

s_and_b32 s6, s1, 4294967294 // (op 882) 

s_cmp_lt_u32 s0, s6 // (op 882) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_882 // (op 882) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoop_882 if true)
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoop_882:

















v_and_b32 v15, -8, v1 // (op 2464) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2464) 
v_sub_u32 v17, v1, v15 // (op 2464) 

s_lshr_b32 s6, s0, 1 // (op 2464) 

v_lshlrev_b32 v15, 5, v17 // (op 2464) 

v_lshl_add_u32 v17, s16, 8, v15 // (op 2464) 

v_mov_b32 v33, 0 // (op 2464) convert
v_mov_b32 v32, v17 // (op 2464) convert

v_lshrrev_b32 v15, 3, v1 // (op 2464) 
v_lshlrev_b32 v17, 2, v15 // (op 2464) 

v_lshl_add_u32 v15, s6, 8, v17 // (op 2464) 


v_mul_lo_u32 v35, v15, s13 // (op 2464) most significant: low of low * high
v_mul_hi_u32 v17, v15, s12 // (op 2464) most significant: high of low * low
v_mul_lo_u32 v34, v15, s12 // (op 2464) least significant: low of low * low
v_add_u32 v35, v35, v17 // (op 2464) most significant: sum


v_add_co_u32 v36, s[0:1], v32, v34 // (op 2464) least significant half
v_addc_co_u32 v37, s[0:1], v33, v35, s[0:1] // (op 2464) most significant half

v_lshlrev_b64 v[36:37], 2, v[36:37] // (op 2464) 


v_lshrrev_b64 v[32:33], 3, v[36:37] // (op 2464) 

v_mov_b32 v15, v32 // (op 2464) convert

s_mul_i32 s1, s12, s23 // (op 2466) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 2466) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 2466) least significant
s_mul_hi_u32 s7, s12, s22 // (op 2466) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2466) most significant: sum
s_add_u32 s1, s1, s7 // (op 2466) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 2466) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 2466) 

s_mov_b32 s48, s20 // (op 2466) reg expression
s_mov_b32 s49, s21 // (op 2466) reg expression
s_mov_b32 s50, s0 // (op 2466) reg expression
s_mov_b32 s51, 131072 // (op 2466) call()

v_and_b32 v17, -4, v1 // (op 2534) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2534) 
v_sub_u32 v55, v1, v17 // (op 2534) 

s_lshr_b32 s6, s0, 1 // (op 2534) 

v_lshlrev_b32 v17, 5, v55 // (op 2534) 

v_lshl_add_u32 v55, s6, 8, v17 // (op 2534) 


v_mov_b32 v33, 0 // (op 2534) convert
v_mov_b32 v32, v55 // (op 2534) convert

v_lshrrev_b32 v17, 2, v1 // (op 2534) 
v_lshl_add_u32 v55, s17, 6, v17 // (op 2534) 

v_mul_lo_u32 v35, v55, s25 // (op 2534) most significant: low of low * high
v_mul_hi_u32 v17, v55, s24 // (op 2534) most significant: high of low * low
v_mul_lo_u32 v34, v55, s24 // (op 2534) least significant: low of low * low
v_add_u32 v35, v35, v17 // (op 2534) most significant: sum


v_add_co_u32 v36, s[0:1], v32, v34 // (op 2534) least significant half
v_addc_co_u32 v37, s[0:1], v33, v35, s[0:1] // (op 2534) most significant half



v_mul_lo_u32 v33, v37, 6 // (op 2534) most significant: low of high * low
v_mul_hi_u32 v17, v36, 6 // (op 2534) most significant: high of low * low
v_mul_lo_u32 v32, v36, 6 // (op 2534) least significant: low of low * low
v_add_u32 v33, v33, v17 // (op 2534) most significant: sum


v_lshrrev_b64 v[34:35], 3, v[32:33] // (op 2534) 

v_mov_b32 v17, v34 // (op 2534) convert

s_mul_i32 s1, s9, 6 // (op 2536) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 2536) least significant
s_mul_hi_u32 s6, s8, 6 // (op 2536) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2536) most significant: sum

s_mul_i32 s7, s24, s1 // (op 2536) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 2536) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 2536) least significant
s_mul_hi_u32 s11, s24, s0 // (op 2536) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 2536) most significant: sum
s_add_u32 s7, s7, s11 // (op 2536) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 2536) 

s_mov_b32 s52, s26 // (op 2536) reg expression
s_mov_b32 s53, s27 // (op 2536) reg expression
s_mov_b32 s54, s0 // (op 2536) reg expression
s_mov_b32 s55, 131072 // (op 2536) call()

v_and_b32 v55, -64, v1 // (op 2598) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2598) 
v_sub_u32 v19, v1, v55 // (op 2598) 

s_lshr_b32 s6, s0, 1 // (op 2598) 

v_lshlrev_b32 v20, 2, v19 // (op 2598) 

v_lshl_add_u32 v19, s16, 8, v20 // (op 2598) 

v_lshrrev_b32 v20, 6, v1 // (op 2598) 
v_lshl_add_u32 v55, s6, 3, v20 // (op 2598) 


v_mul_lo_u32 v20, v55, s28 // (op 2598) 

v_add_u32 v55, v19, v20 // (op 2598) 


s_mul_i32 s1, s28, s33 // (op 2600) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 2600) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 2600) least significant
s_mul_hi_u32 s7, s28, s32 // (op 2600) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2600) most significant: sum
s_add_u32 s1, s1, s7 // (op 2600) most significant: sum


s_mov_b32 s56, s30 // (op 2600) reg expression
s_mov_b32 s57, s31 // (op 2600) reg expression
s_mov_b32 s58, s0 // (op 2600) reg expression
s_mov_b32 s59, 131072 // (op 2600) call()

v_and_b32 v19, -4, v1 // (op 2662) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 2662) 
v_sub_u32 v20, v1, v19 // (op 2662) 

s_lshr_b32 s6, s0, 1 // (op 2662) 

v_lshrrev_b32 v19, 2, v1 // (op 2662) 
v_lshl_add_u32 v23, s6, 3, v20 // (op 2662) 


v_lshl_add_u32 v20, s17, 6, v19 // (op 2662) 

v_mul_lo_u32 v19, v20, s34 // (op 2662) 

v_add_u32 v20, v23, v19 // (op 2662) 


s_mul_i32 s1, s34, s9 // (op 2664) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 2664) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 2664) least significant
s_mul_hi_u32 s7, s34, s8 // (op 2664) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 2664) most significant: sum
s_add_u32 s1, s1, s7 // (op 2664) most significant: sum


s_mov_b32 s60, s36 // (op 2664) reg expression
s_mov_b32 s61, s37 // (op 2664) reg expression
s_mov_b32 s62, s0 // (op 2664) reg expression
s_mov_b32 s63, 131072 // (op 2664) call()

v_and_b32 v19, -8, v1 // (op 2755) 
v_sub_u32 v23, v1, v19 // (op 2755) 

v_lshlrev_b32 v19, 5, v23 // (op 2755) 

v_lshrrev_b32 v23, 3, v1 // (op 2755) 
v_lshl_add_u32 v24, v23, 10, v19 // (op 2755) 


v_lshlrev_b32 v19, 2, v24 // (op 2755) 

v_lshrrev_b32 v23, 3, v19 // (op 2755) 

v_and_b32 v19, -4, v1 // (op 2813) 
v_sub_u32 v24, v1, v19 // (op 2813) 

v_lshlrev_b32 v19, 5, v24 // (op 2813) 

v_lshrrev_b32 v24, 2, v1 // (op 2813) 
v_lshl_add_u32 v25, v24, 7, v19 // (op 2813) 


v_mul_lo_u32 v19, v25, 6 // (op 2813) 

v_lshrrev_b32 v24, 3, v19 // (op 2813) 

v_and_b32 v19, -64, v1 // (op 2866) 
v_sub_u32 v25, v1, v19 // (op 2866) 

v_lshlrev_b32 v19, 2, v25 // (op 2866) 

v_lshrrev_b32 v25, 6, v1 // (op 2866) 
v_lshl_add_u32 v26, v25, 8, v19 // (op 2866) 


v_and_b32 v19, -4, v1 // (op 2924) 
v_sub_u32 v25, v1, v19 // (op 2924) 

v_and_b32 v19, 4294967292, v1 // (op 2924) 
v_add_u32 v27, v19, v25 // (op 2924) 


s_ashr_i64 s[0:1], s[22:23], 7 // (op 888) 
s_and_b32 s6, s0, 4294967294 // (op 888) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 887) 
s_mov_b32 s7, s0 // (op 887) convert

s_cmp_lt_u32 s6, s7 // (op 887) 















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_887 // (op 887) Condition: Top (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_887 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_887:

s_add_u32 s6, s6, 1 // (op 889) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 198) 
v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 198) 


v_mov_b32 v14, v15 // (op 198) 
v_mov_b32 v16, v14 // (op 198) reg expression
buffer_load_dwordx4 v[28:31], v16, s[48:51], 0 offen // (op 198) Load value
v_add_u32 v16, v16, v18 // (op 198) 
buffer_load_dwordx4 v[32:35], v16, s[48:51], 0 offen // (op 198) Load value
v_add_u32 v16, v16, v18 // (op 198) 
buffer_load_dwordx4 v[36:39], v16, s[48:51], 0 offen // (op 198) Load value
v_add_u32 v16, v16, v18 // (op 198) 
buffer_load_dwordx4 v[40:43], v16, s[48:51], 0 offen // (op 198) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 2454) 
s_lshl_b64 s[10:11], s[0:1], 2 // (op 2454) 

s_lshr_b64 s[0:1], s[10:11], 3 // (op 2454) 

v_add_u32 v15, v15, s0 // (op 2454) 



s_barrier  // (op 3145) 
v_mov_b32 v14, v17 // (op 221) 




buffer_load_dwordx4 v[44:47], v14, s[52:55], 0 offen // (op 221) Load value
buffer_load_dwordx2 v[48:49], v14, s[52:55], 0 offen offset:16 // (op 221) Load value

v_add_u32 v17, 96, v17 // (op 2526) 
s_waitcnt vmcnt(5)
ds_write_b128 v23, v[28:31] offset:6400 // (op 199) Store local data 


v_mov_b32 v14, v55 // (op 222) 
v_mov_b32 v16, v14 // (op 222) reg expression

s_waitcnt vmcnt(4)
ds_write_b128 v23, v[32:35] offset:6528 // (op 199) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v23, v[36:39] offset:6656 // (op 199) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v23, v[40:43] offset:6784 // (op 199) Store local data 
buffer_load_dword v25, v16, s[56:59], 0 offen // (op 222) Load value



s_lshl_b32 s7, s28, 2 // (op 2590) 
s_barrier  // (op 3164) 
v_add_u32 v55, v55, s7 // (op 2590) 






v_mov_b32 v14, v20 // (op 223) 

buffer_load_ubyte v16, v14, s[60:63], 0 offen offset: 0 // (op 223) Load value

v_add_u32 v20, 4, v20 // (op 2654) 
s_waitcnt vmcnt(3)
ds_write_b128 v24, v[44:47] // (op 218) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 3154) 
s_waitcnt vmcnt(2)
ds_write_b64 v24, v[48:49] offset:16 // (op 218) Store local data 




ds_read_b64_tr_b4 v[28:29], v6 offset:6400 // (op 201) Transpose load local data 
s_barrier  // (op 3185) 
ds_read_b64_tr_b4 v[30:31], v6 offset:8448 // (op 201) Transpose load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 3175) 
s_waitcnt vmcnt(1)
ds_write_b32 v26, v25 offset:22784 // (op 219) Store local data 




ds_read_b128 v[32:35], v3 // (op 203) Load local data 
s_barrier  // (op 3205) 
ds_read_b64 v[36:37], v3 offset:16 // (op 203) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 3195) 
s_waitcnt vmcnt(0)
ds_write_b8 v27, v16 offset:6144 // (op 220) Store local data 



ds_read_u8 v14, v7 offset:22784 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 3215) 



ds_read_u8 v16, v0 offset:6144 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[28:31], v[32:37], a[0:15], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 208) 





ds_read_b64_tr_b4 v[28:29], v6 offset:14592 // (op 210) Transpose load local data 
ds_read_b64_tr_b4 v[30:31], v6 offset:16640 // (op 210) Transpose load local data 



ds_read_b128 v[38:41], v3 offset:48 // (op 212) Load local data 
ds_read_b64 v[42:43], v3 offset:64 // (op 212) Load local data 


ds_read_u8 v14, v7 offset:23296 // (op 214) Load local data 



ds_read_u8 v25, v0 offset:6146 // (op 216) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[28:31], v[38:43], a[0:15], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 217) 
ds_read_b64_tr_b4 v[44:45], v6 offset:6432 // (op 413) Transpose load local data 


ds_read_b64_tr_b4 v[46:47], v6 offset:8480 // (op 413) Transpose load local data 


ds_read_u8 v14, v7 offset:22848 // (op 417) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[44:47], v[32:37], a[16:31], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 420) 





ds_read_b64_tr_b4 v[28:29], v6 offset:14624 // (op 422) Transpose load local data 
ds_read_b64_tr_b4 v[30:31], v6 offset:16672 // (op 422) Transpose load local data 


ds_read_u8 v14, v7 offset:23360 // (op 426) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[28:31], v[38:43], a[16:31], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 429) 
ds_read_b64_tr_b4 v[44:45], v6 offset:6464 // (op 653) Transpose load local data 


ds_read_b64_tr_b4 v[46:47], v6 offset:8512 // (op 653) Transpose load local data 


ds_read_u8 v14, v7 offset:22912 // (op 657) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[44:47], v[32:37], a[32:47], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 660) 





ds_read_b64_tr_b4 v[28:29], v6 offset:14656 // (op 662) Transpose load local data 
ds_read_b64_tr_b4 v[30:31], v6 offset:16704 // (op 662) Transpose load local data 


ds_read_u8 v14, v7 offset:23424 // (op 666) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[28:31], v[38:43], a[32:47], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 669) 
ds_read_b64_tr_b4 v[44:45], v6 offset:6496 // (op 893) Transpose load local data 


ds_read_b64_tr_b4 v[46:47], v6 offset:8544 // (op 893) Transpose load local data 


ds_read_u8 v14, v7 offset:22976 // (op 897) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[44:47], v[32:37], a[48:63], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 900) 







ds_read_b64_tr_b4 v[28:29], v6 offset:14688 // (op 902) Transpose load local data 
ds_read_b64_tr_b4 v[30:31], v6 offset:16736 // (op 902) Transpose load local data 


ds_read_u8 v14, v7 offset:23488 // (op 906) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[28:31], v[38:43], a[48:63], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 909) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 887) 
s_mov_b32 s7, s0 // (op 887) convert

s_cmp_lt_u32 s6, s7 // (op 887) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_887 // (op 887) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopTop_KLoopTail_887 if true)
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ForLoopBottom_KLoopTail_887:























s_barrier  // (op 3332) 









v_cmp_eq_f32 vcc, 0.00000, v12 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_155
v_accvgpr_read v14, a0 // (op 50) 
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




buffer_store_dwordx4 v[30:33], v11, s[44:47], 0 offen // (op 56) Store value
buffer_store_dwordx4 v[34:37], v11, s[44:47], 0 offen offset:32 // (op 56) Store value
buffer_store_dwordx4 v[38:41], v11, s[44:47], 0 offen offset:64 // (op 56) Store value
buffer_store_dwordx4 v[42:45], v11, s[44:47], 0 offen offset:96 // (op 56) Store value

v_accvgpr_read v14, a16 // (op 1279) 
v_accvgpr_read v15, a17 // (op 1279) 
v_accvgpr_read v16, a18 // (op 1279) 
v_accvgpr_read v17, a19 // (op 1279) 
v_accvgpr_read v18, a20 // (op 1279) 
v_accvgpr_read v19, a21 // (op 1279) 
v_accvgpr_read v20, a22 // (op 1279) 
v_accvgpr_read v21, a23 // (op 1279) 
v_accvgpr_read v22, a24 // (op 1279) 
v_accvgpr_read v23, a25 // (op 1279) 
v_accvgpr_read v24, a26 // (op 1279) 
v_accvgpr_read v25, a27 // (op 1279) 
v_accvgpr_read v26, a28 // (op 1279) 
v_accvgpr_read v27, a29 // (op 1279) 
v_accvgpr_read v28, a30 // (op 1279) 
v_accvgpr_read v29, a31 // (op 1279) 
v_mul_f32 v30, v13, v14 // (op 1279) 
v_mul_f32 v31, v13, v15 // (op 1279) 
v_mul_f32 v32, v13, v16 // (op 1279) 
v_mul_f32 v33, v13, v17 // (op 1279) 
v_mul_f32 v34, v13, v18 // (op 1279) 
v_mul_f32 v35, v13, v19 // (op 1279) 
v_mul_f32 v36, v13, v20 // (op 1279) 
v_mul_f32 v37, v13, v21 // (op 1279) 
v_mul_f32 v38, v13, v22 // (op 1279) 
v_mul_f32 v39, v13, v23 // (op 1279) 
v_mul_f32 v40, v13, v24 // (op 1279) 
v_mul_f32 v41, v13, v25 // (op 1279) 
v_mul_f32 v42, v13, v26 // (op 1279) 
v_mul_f32 v43, v13, v27 // (op 1279) 
v_mul_f32 v44, v13, v28 // (op 1279) 
v_mul_f32 v45, v13, v29 // (op 1279) 




buffer_store_dwordx4 v[30:33], v11, s[44:47], 0 offen offset:256 // (op 1281) Store value
buffer_store_dwordx4 v[34:37], v11, s[44:47], 0 offen offset:288 // (op 1281) Store value
buffer_store_dwordx4 v[38:41], v11, s[44:47], 0 offen offset:320 // (op 1281) Store value
buffer_store_dwordx4 v[42:45], v11, s[44:47], 0 offen offset:352 // (op 1281) Store value

v_accvgpr_read v14, a32 // (op 1297) 
v_accvgpr_read v15, a33 // (op 1297) 
v_accvgpr_read v16, a34 // (op 1297) 
v_accvgpr_read v17, a35 // (op 1297) 
v_accvgpr_read v18, a36 // (op 1297) 
v_accvgpr_read v19, a37 // (op 1297) 
v_accvgpr_read v20, a38 // (op 1297) 
v_accvgpr_read v21, a39 // (op 1297) 
v_accvgpr_read v22, a40 // (op 1297) 
v_accvgpr_read v23, a41 // (op 1297) 
v_accvgpr_read v24, a42 // (op 1297) 
v_accvgpr_read v25, a43 // (op 1297) 
v_accvgpr_read v26, a44 // (op 1297) 
v_accvgpr_read v27, a45 // (op 1297) 
v_accvgpr_read v28, a46 // (op 1297) 
v_accvgpr_read v29, a47 // (op 1297) 
v_mul_f32 v30, v13, v14 // (op 1297) 
v_mul_f32 v31, v13, v15 // (op 1297) 
v_mul_f32 v32, v13, v16 // (op 1297) 
v_mul_f32 v33, v13, v17 // (op 1297) 
v_mul_f32 v34, v13, v18 // (op 1297) 
v_mul_f32 v35, v13, v19 // (op 1297) 
v_mul_f32 v36, v13, v20 // (op 1297) 
v_mul_f32 v37, v13, v21 // (op 1297) 
v_mul_f32 v38, v13, v22 // (op 1297) 
v_mul_f32 v39, v13, v23 // (op 1297) 
v_mul_f32 v40, v13, v24 // (op 1297) 
v_mul_f32 v41, v13, v25 // (op 1297) 
v_mul_f32 v42, v13, v26 // (op 1297) 
v_mul_f32 v43, v13, v27 // (op 1297) 
v_mul_f32 v44, v13, v28 // (op 1297) 
v_mul_f32 v45, v13, v29 // (op 1297) 




buffer_store_dwordx4 v[30:33], v11, s[44:47], 0 offen offset:512 // (op 1299) Store value
buffer_store_dwordx4 v[34:37], v11, s[44:47], 0 offen offset:544 // (op 1299) Store value
buffer_store_dwordx4 v[38:41], v11, s[44:47], 0 offen offset:576 // (op 1299) Store value
buffer_store_dwordx4 v[42:45], v11, s[44:47], 0 offen offset:608 // (op 1299) Store value

v_accvgpr_read v14, a48 // (op 1315) 
v_accvgpr_read v15, a49 // (op 1315) 
v_accvgpr_read v16, a50 // (op 1315) 
v_accvgpr_read v17, a51 // (op 1315) 
v_accvgpr_read v18, a52 // (op 1315) 
v_accvgpr_read v19, a53 // (op 1315) 
v_accvgpr_read v20, a54 // (op 1315) 
v_accvgpr_read v21, a55 // (op 1315) 
v_accvgpr_read v22, a56 // (op 1315) 
v_accvgpr_read v23, a57 // (op 1315) 
v_accvgpr_read v24, a58 // (op 1315) 
v_accvgpr_read v25, a59 // (op 1315) 
v_accvgpr_read v26, a60 // (op 1315) 
v_accvgpr_read v27, a61 // (op 1315) 
v_accvgpr_read v28, a62 // (op 1315) 
v_accvgpr_read v29, a63 // (op 1315) 
v_mul_f32 v30, v13, v14 // (op 1315) 
v_mul_f32 v31, v13, v15 // (op 1315) 
v_mul_f32 v32, v13, v16 // (op 1315) 
v_mul_f32 v33, v13, v17 // (op 1315) 
v_mul_f32 v34, v13, v18 // (op 1315) 
v_mul_f32 v35, v13, v19 // (op 1315) 
v_mul_f32 v36, v13, v20 // (op 1315) 
v_mul_f32 v37, v13, v21 // (op 1315) 
v_mul_f32 v38, v13, v22 // (op 1315) 
v_mul_f32 v39, v13, v23 // (op 1315) 
v_mul_f32 v40, v13, v24 // (op 1315) 
v_mul_f32 v41, v13, v25 // (op 1315) 
v_mul_f32 v42, v13, v26 // (op 1315) 
v_mul_f32 v43, v13, v27 // (op 1315) 
v_mul_f32 v44, v13, v28 // (op 1315) 
v_mul_f32 v45, v13, v29 // (op 1315) 




buffer_store_dwordx4 v[30:33], v11, s[44:47], 0 offen offset:768 // (op 1317) Store value
buffer_store_dwordx4 v[34:37], v11, s[44:47], 0 offen offset:800 // (op 1317) Store value
buffer_store_dwordx4 v[38:41], v11, s[44:47], 0 offen offset:832 // (op 1317) Store value
buffer_store_dwordx4 v[42:45], v11, s[44:47], 0 offen offset:864 // (op 1317) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalFalse_0_beta_155:



buffer_load_dwordx4 v[14:17], v8, s[40:43], 0 offen offset:768 // (op 1391) Load value
buffer_load_dwordx4 v[18:21], v8, s[40:43], 0 offen offset:800 // (op 1391) Load value
buffer_load_dwordx4 v[22:25], v8, s[40:43], 0 offen offset:832 // (op 1391) Load value
v_accvgpr_read v30, a48 // (op 1388) 
v_accvgpr_read v31, a49 // (op 1388) 
v_accvgpr_read v32, a50 // (op 1388) 
v_accvgpr_read v33, a51 // (op 1388) 
v_accvgpr_read v34, a52 // (op 1388) 
v_accvgpr_read v35, a53 // (op 1388) 
v_accvgpr_read v36, a54 // (op 1388) 
v_accvgpr_read v37, a55 // (op 1388) 
v_accvgpr_read v38, a56 // (op 1388) 
v_accvgpr_read v39, a57 // (op 1388) 
v_accvgpr_read v40, a58 // (op 1388) 
v_accvgpr_read v41, a59 // (op 1388) 
v_accvgpr_read v42, a60 // (op 1388) 
v_accvgpr_read v43, a61 // (op 1388) 
v_accvgpr_read v44, a62 // (op 1388) 
v_accvgpr_read v45, a63 // (op 1388) 
v_mul_f32 v46, v13, v30 // (op 1388) 
v_mul_f32 v47, v13, v31 // (op 1388) 
v_mul_f32 v48, v13, v32 // (op 1388) 
v_mul_f32 v49, v13, v33 // (op 1388) 
v_mul_f32 v50, v13, v34 // (op 1388) 
v_mul_f32 v51, v13, v35 // (op 1388) 
v_mul_f32 v52, v13, v36 // (op 1388) 
v_mul_f32 v53, v13, v37 // (op 1388) 
v_mul_f32 v54, v13, v38 // (op 1388) 
v_mul_f32 v55, v13, v39 // (op 1388) 
v_mul_f32 v56, v13, v40 // (op 1388) 
v_mul_f32 v57, v13, v41 // (op 1388) 
v_mul_f32 v58, v13, v42 // (op 1388) 
v_mul_f32 v59, v13, v43 // (op 1388) 
v_mul_f32 v60, v13, v44 // (op 1388) 
v_mul_f32 v61, v13, v45 // (op 1388) 

buffer_load_dwordx4 v[26:29], v8, s[40:43], 0 offen offset:864 // (op 1391) Load value
s_waitcnt vmcnt(3)
v_mul_f32 v30, v12, v14 // (op 1392) 
v_mul_f32 v31, v12, v15 // (op 1392) 
v_mul_f32 v32, v12, v16 // (op 1392) 
v_mul_f32 v33, v12, v17 // (op 1392) 
s_waitcnt vmcnt(2)
v_mul_f32 v34, v12, v18 // (op 1392) 
v_mul_f32 v35, v12, v19 // (op 1392) 
v_mul_f32 v36, v12, v20 // (op 1392) 
v_mul_f32 v37, v12, v21 // (op 1392) 
s_waitcnt vmcnt(1)
v_mul_f32 v38, v12, v22 // (op 1392) 
v_mul_f32 v39, v12, v23 // (op 1392) 
v_mul_f32 v40, v12, v24 // (op 1392) 
v_mul_f32 v41, v12, v25 // (op 1392) 
s_waitcnt vmcnt(0)
v_mul_f32 v42, v12, v26 // (op 1392) 
v_mul_f32 v43, v12, v27 // (op 1392) 
v_mul_f32 v44, v12, v28 // (op 1392) 
v_mul_f32 v45, v12, v29 // (op 1392) 

v_add_f32 v14, v30, v46 // (op 1389) 
v_add_f32 v15, v31, v47 // (op 1389) 
v_add_f32 v16, v32, v48 // (op 1389) 
v_add_f32 v17, v33, v49 // (op 1389) 
v_add_f32 v18, v34, v50 // (op 1389) 
v_add_f32 v19, v35, v51 // (op 1389) 
v_add_f32 v20, v36, v52 // (op 1389) 
v_add_f32 v21, v37, v53 // (op 1389) 
v_add_f32 v22, v38, v54 // (op 1389) 
v_add_f32 v23, v39, v55 // (op 1389) 
v_add_f32 v24, v40, v56 // (op 1389) 
v_add_f32 v25, v41, v57 // (op 1389) 
v_add_f32 v26, v42, v58 // (op 1389) 
v_add_f32 v27, v43, v59 // (op 1389) 
v_add_f32 v28, v44, v60 // (op 1389) 
v_add_f32 v29, v45, v61 // (op 1389) 





buffer_store_dwordx4 v[14:17], v11, s[44:47], 0 offen offset:768 // (op 1390) Store value
buffer_store_dwordx4 v[18:21], v11, s[44:47], 0 offen offset:800 // (op 1390) Store value
buffer_store_dwordx4 v[22:25], v11, s[44:47], 0 offen offset:832 // (op 1390) Store value
buffer_store_dwordx4 v[26:29], v11, s[44:47], 0 offen offset:864 // (op 1390) Store value




buffer_load_dwordx4 v[14:17], v8, s[40:43], 0 offen offset:512 // (op 1373) Load value
buffer_load_dwordx4 v[18:21], v8, s[40:43], 0 offen offset:544 // (op 1373) Load value
v_accvgpr_read v30, a32 // (op 1370) 
v_accvgpr_read v31, a33 // (op 1370) 
v_accvgpr_read v32, a34 // (op 1370) 
v_accvgpr_read v33, a35 // (op 1370) 
v_accvgpr_read v34, a36 // (op 1370) 
v_accvgpr_read v35, a37 // (op 1370) 
v_accvgpr_read v36, a38 // (op 1370) 
v_accvgpr_read v37, a39 // (op 1370) 
v_accvgpr_read v38, a40 // (op 1370) 
v_accvgpr_read v39, a41 // (op 1370) 
v_accvgpr_read v40, a42 // (op 1370) 
v_accvgpr_read v41, a43 // (op 1370) 
v_accvgpr_read v42, a44 // (op 1370) 
v_accvgpr_read v43, a45 // (op 1370) 
v_accvgpr_read v44, a46 // (op 1370) 
v_accvgpr_read v45, a47 // (op 1370) 
v_mul_f32 v46, v13, v30 // (op 1370) 
v_mul_f32 v47, v13, v31 // (op 1370) 
v_mul_f32 v48, v13, v32 // (op 1370) 
v_mul_f32 v49, v13, v33 // (op 1370) 
v_mul_f32 v50, v13, v34 // (op 1370) 
v_mul_f32 v51, v13, v35 // (op 1370) 
v_mul_f32 v52, v13, v36 // (op 1370) 
v_mul_f32 v53, v13, v37 // (op 1370) 
v_mul_f32 v54, v13, v38 // (op 1370) 
v_mul_f32 v55, v13, v39 // (op 1370) 
v_mul_f32 v56, v13, v40 // (op 1370) 
v_mul_f32 v57, v13, v41 // (op 1370) 
v_mul_f32 v58, v13, v42 // (op 1370) 
v_mul_f32 v59, v13, v43 // (op 1370) 
v_mul_f32 v60, v13, v44 // (op 1370) 
v_mul_f32 v61, v13, v45 // (op 1370) 

buffer_load_dwordx4 v[22:25], v8, s[40:43], 0 offen offset:576 // (op 1373) Load value
buffer_load_dwordx4 v[26:29], v8, s[40:43], 0 offen offset:608 // (op 1373) Load value
s_waitcnt vmcnt(3)
v_mul_f32 v30, v12, v14 // (op 1374) 
v_mul_f32 v31, v12, v15 // (op 1374) 
v_mul_f32 v32, v12, v16 // (op 1374) 
v_mul_f32 v33, v12, v17 // (op 1374) 
s_waitcnt vmcnt(2)
v_mul_f32 v34, v12, v18 // (op 1374) 
v_mul_f32 v35, v12, v19 // (op 1374) 
v_mul_f32 v36, v12, v20 // (op 1374) 
v_mul_f32 v37, v12, v21 // (op 1374) 
s_waitcnt vmcnt(1)
v_mul_f32 v38, v12, v22 // (op 1374) 
v_mul_f32 v39, v12, v23 // (op 1374) 
v_mul_f32 v40, v12, v24 // (op 1374) 
v_mul_f32 v41, v12, v25 // (op 1374) 
s_waitcnt vmcnt(0)
v_mul_f32 v42, v12, v26 // (op 1374) 
v_mul_f32 v43, v12, v27 // (op 1374) 
v_mul_f32 v44, v12, v28 // (op 1374) 
v_mul_f32 v45, v12, v29 // (op 1374) 

v_add_f32 v14, v30, v46 // (op 1371) 
v_add_f32 v15, v31, v47 // (op 1371) 
v_add_f32 v16, v32, v48 // (op 1371) 
v_add_f32 v17, v33, v49 // (op 1371) 
v_add_f32 v18, v34, v50 // (op 1371) 
v_add_f32 v19, v35, v51 // (op 1371) 
v_add_f32 v20, v36, v52 // (op 1371) 
v_add_f32 v21, v37, v53 // (op 1371) 
v_add_f32 v22, v38, v54 // (op 1371) 
v_add_f32 v23, v39, v55 // (op 1371) 
v_add_f32 v24, v40, v56 // (op 1371) 
v_add_f32 v25, v41, v57 // (op 1371) 
v_add_f32 v26, v42, v58 // (op 1371) 
v_add_f32 v27, v43, v59 // (op 1371) 
v_add_f32 v28, v44, v60 // (op 1371) 
v_add_f32 v29, v45, v61 // (op 1371) 





buffer_store_dwordx4 v[14:17], v11, s[44:47], 0 offen offset:512 // (op 1372) Store value
buffer_store_dwordx4 v[18:21], v11, s[44:47], 0 offen offset:544 // (op 1372) Store value
buffer_store_dwordx4 v[22:25], v11, s[44:47], 0 offen offset:576 // (op 1372) Store value
buffer_store_dwordx4 v[26:29], v11, s[44:47], 0 offen offset:608 // (op 1372) Store value




buffer_load_dwordx4 v[14:17], v8, s[40:43], 0 offen offset:256 // (op 1355) Load value
buffer_load_dwordx4 v[18:21], v8, s[40:43], 0 offen offset:288 // (op 1355) Load value
v_accvgpr_read v30, a16 // (op 1352) 
v_accvgpr_read v31, a17 // (op 1352) 
v_accvgpr_read v32, a18 // (op 1352) 
v_accvgpr_read v33, a19 // (op 1352) 
v_accvgpr_read v34, a20 // (op 1352) 
v_accvgpr_read v35, a21 // (op 1352) 
v_accvgpr_read v36, a22 // (op 1352) 
v_accvgpr_read v37, a23 // (op 1352) 
v_accvgpr_read v38, a24 // (op 1352) 
v_accvgpr_read v39, a25 // (op 1352) 
v_accvgpr_read v40, a26 // (op 1352) 
v_accvgpr_read v41, a27 // (op 1352) 
v_accvgpr_read v42, a28 // (op 1352) 
v_accvgpr_read v43, a29 // (op 1352) 
v_accvgpr_read v44, a30 // (op 1352) 
v_accvgpr_read v45, a31 // (op 1352) 
v_mul_f32 v46, v13, v30 // (op 1352) 
v_mul_f32 v47, v13, v31 // (op 1352) 
v_mul_f32 v48, v13, v32 // (op 1352) 
v_mul_f32 v49, v13, v33 // (op 1352) 
v_mul_f32 v50, v13, v34 // (op 1352) 
v_mul_f32 v51, v13, v35 // (op 1352) 
v_mul_f32 v52, v13, v36 // (op 1352) 
v_mul_f32 v53, v13, v37 // (op 1352) 
v_mul_f32 v54, v13, v38 // (op 1352) 
v_mul_f32 v55, v13, v39 // (op 1352) 
v_mul_f32 v56, v13, v40 // (op 1352) 
v_mul_f32 v57, v13, v41 // (op 1352) 
v_mul_f32 v58, v13, v42 // (op 1352) 
v_mul_f32 v59, v13, v43 // (op 1352) 
v_mul_f32 v60, v13, v44 // (op 1352) 
v_mul_f32 v61, v13, v45 // (op 1352) 

buffer_load_dwordx4 v[22:25], v8, s[40:43], 0 offen offset:320 // (op 1355) Load value
buffer_load_dwordx4 v[26:29], v8, s[40:43], 0 offen offset:352 // (op 1355) Load value
s_waitcnt vmcnt(3)
v_mul_f32 v30, v12, v14 // (op 1356) 
v_mul_f32 v31, v12, v15 // (op 1356) 
v_mul_f32 v32, v12, v16 // (op 1356) 
v_mul_f32 v33, v12, v17 // (op 1356) 
s_waitcnt vmcnt(2)
v_mul_f32 v34, v12, v18 // (op 1356) 
v_mul_f32 v35, v12, v19 // (op 1356) 
v_mul_f32 v36, v12, v20 // (op 1356) 
v_mul_f32 v37, v12, v21 // (op 1356) 
s_waitcnt vmcnt(1)
v_mul_f32 v38, v12, v22 // (op 1356) 
v_mul_f32 v39, v12, v23 // (op 1356) 
v_mul_f32 v40, v12, v24 // (op 1356) 
v_mul_f32 v41, v12, v25 // (op 1356) 
s_waitcnt vmcnt(0)
v_mul_f32 v42, v12, v26 // (op 1356) 
v_mul_f32 v43, v12, v27 // (op 1356) 
v_mul_f32 v44, v12, v28 // (op 1356) 
v_mul_f32 v45, v12, v29 // (op 1356) 

v_add_f32 v14, v30, v46 // (op 1353) 
v_add_f32 v15, v31, v47 // (op 1353) 
v_add_f32 v16, v32, v48 // (op 1353) 
v_add_f32 v17, v33, v49 // (op 1353) 
v_add_f32 v18, v34, v50 // (op 1353) 
v_add_f32 v19, v35, v51 // (op 1353) 
v_add_f32 v20, v36, v52 // (op 1353) 
v_add_f32 v21, v37, v53 // (op 1353) 
v_add_f32 v22, v38, v54 // (op 1353) 
v_add_f32 v23, v39, v55 // (op 1353) 
v_add_f32 v24, v40, v56 // (op 1353) 
v_add_f32 v25, v41, v57 // (op 1353) 
v_add_f32 v26, v42, v58 // (op 1353) 
v_add_f32 v27, v43, v59 // (op 1353) 
v_add_f32 v28, v44, v60 // (op 1353) 
v_add_f32 v29, v45, v61 // (op 1353) 





buffer_store_dwordx4 v[14:17], v11, s[44:47], 0 offen offset:256 // (op 1354) Store value
buffer_store_dwordx4 v[18:21], v11, s[44:47], 0 offen offset:288 // (op 1354) Store value
buffer_store_dwordx4 v[22:25], v11, s[44:47], 0 offen offset:320 // (op 1354) Store value
buffer_store_dwordx4 v[26:29], v11, s[44:47], 0 offen offset:352 // (op 1354) Store value




buffer_load_dwordx4 v[14:17], v8, s[40:43], 0 offen // (op 167) Load value
buffer_load_dwordx4 v[18:21], v8, s[40:43], 0 offen offset:32 // (op 167) Load value
v_accvgpr_read v30, a0 // (op 164) 
v_accvgpr_read v31, a1 // (op 164) 
v_accvgpr_read v32, a2 // (op 164) 
v_accvgpr_read v33, a3 // (op 164) 
v_accvgpr_read v34, a4 // (op 164) 
v_accvgpr_read v35, a5 // (op 164) 
v_accvgpr_read v36, a6 // (op 164) 
v_accvgpr_read v37, a7 // (op 164) 
v_accvgpr_read v38, a8 // (op 164) 
v_accvgpr_read v39, a9 // (op 164) 
v_accvgpr_read v40, a10 // (op 164) 
v_accvgpr_read v41, a11 // (op 164) 
v_accvgpr_read v42, a12 // (op 164) 
v_accvgpr_read v43, a13 // (op 164) 
v_accvgpr_read v44, a14 // (op 164) 
v_accvgpr_read v45, a15 // (op 164) 
v_mul_f32 v46, v13, v30 // (op 164) 
v_mul_f32 v47, v13, v31 // (op 164) 
v_mul_f32 v48, v13, v32 // (op 164) 
v_mul_f32 v49, v13, v33 // (op 164) 
v_mul_f32 v50, v13, v34 // (op 164) 
v_mul_f32 v51, v13, v35 // (op 164) 
v_mul_f32 v52, v13, v36 // (op 164) 
v_mul_f32 v53, v13, v37 // (op 164) 
v_mul_f32 v54, v13, v38 // (op 164) 
v_mul_f32 v55, v13, v39 // (op 164) 
v_mul_f32 v56, v13, v40 // (op 164) 
v_mul_f32 v57, v13, v41 // (op 164) 
v_mul_f32 v58, v13, v42 // (op 164) 
v_mul_f32 v59, v13, v43 // (op 164) 
v_mul_f32 v60, v13, v44 // (op 164) 
v_mul_f32 v61, v13, v45 // (op 164) 

buffer_load_dwordx4 v[22:25], v8, s[40:43], 0 offen offset:64 // (op 167) Load value
buffer_load_dwordx4 v[26:29], v8, s[40:43], 0 offen offset:96 // (op 167) Load value
s_waitcnt vmcnt(3)
v_mul_f32 v30, v12, v14 // (op 168) 
v_mul_f32 v31, v12, v15 // (op 168) 
v_mul_f32 v32, v12, v16 // (op 168) 
v_mul_f32 v33, v12, v17 // (op 168) 
s_waitcnt vmcnt(2)
v_mul_f32 v34, v12, v18 // (op 168) 
v_mul_f32 v35, v12, v19 // (op 168) 
v_mul_f32 v36, v12, v20 // (op 168) 
v_mul_f32 v37, v12, v21 // (op 168) 
s_waitcnt vmcnt(1)
v_mul_f32 v38, v12, v22 // (op 168) 
v_mul_f32 v39, v12, v23 // (op 168) 
v_mul_f32 v40, v12, v24 // (op 168) 
v_mul_f32 v41, v12, v25 // (op 168) 
s_waitcnt vmcnt(0)
v_mul_f32 v42, v12, v26 // (op 168) 
v_mul_f32 v43, v12, v27 // (op 168) 
v_mul_f32 v44, v12, v28 // (op 168) 
v_mul_f32 v45, v12, v29 // (op 168) 

v_add_f32 v14, v30, v46 // (op 165) 
v_add_f32 v15, v31, v47 // (op 165) 
v_add_f32 v16, v32, v48 // (op 165) 
v_add_f32 v17, v33, v49 // (op 165) 
v_add_f32 v18, v34, v50 // (op 165) 
v_add_f32 v19, v35, v51 // (op 165) 
v_add_f32 v20, v36, v52 // (op 165) 
v_add_f32 v21, v37, v53 // (op 165) 
v_add_f32 v22, v38, v54 // (op 165) 
v_add_f32 v23, v39, v55 // (op 165) 
v_add_f32 v24, v40, v56 // (op 165) 
v_add_f32 v25, v41, v57 // (op 165) 
v_add_f32 v26, v42, v58 // (op 165) 
v_add_f32 v27, v43, v59 // (op 165) 
v_add_f32 v28, v44, v60 // (op 165) 
v_add_f32 v29, v45, v61 // (op 165) 





buffer_store_dwordx4 v[14:17], v11, s[44:47], 0 offen // (op 166) Store value
buffer_store_dwordx4 v[18:21], v11, s[44:47], 0 offen offset:32 // (op 166) Store value
buffer_store_dwordx4 v[22:25], v11, s[44:47], 0 offen offset:64 // (op 166) Store value
buffer_store_dwordx4 v[26:29], v11, s[44:47], 0 offen offset:96 // (op 166) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_ConditionalBottom_0_beta_155:












s_endpgm  // End of RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.LRR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_end:
.size RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2, .LRR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_end-RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
  .amdhsa_next_free_vgpr 188
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 47616 // lds bytes
  .amdhsa_accum_offset 124
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
    DataTypeB: B6
    DestDataType: s
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
    macrotile: [256, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2
    .symbol: RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 47616
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 122
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

