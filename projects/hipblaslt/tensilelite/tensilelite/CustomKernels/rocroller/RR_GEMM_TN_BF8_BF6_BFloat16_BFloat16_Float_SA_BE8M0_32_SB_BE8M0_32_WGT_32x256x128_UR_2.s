// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2,@function
RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 3771) 
v_and_b32 v3, 15, v1 // (op 3771) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3771) 


v_and_b32 v0, 48, v1 // (op 3771) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3771) 


v_lshrrev_b32 v0, 7, v1 // (op 3944) 
v_and_b32 v4, 15, v1 // (op 3944) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3944) 


v_and_b32 v0, 48, v1 // (op 3944) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 3944) 


v_lshrrev_b32 v0, 7, v1 // (op 3715) 
v_and_b32 v5, 63, v1 // (op 3715) 
v_lshrrev_b32 v6, 4, v5 // (op 3715) 
v_lshlrev_b32 v7, 2, v6 // (op 3715) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3715) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 3715) 

v_lshrrev_b32 v6, 2, v5 // (op 3715) 

v_and_b32 v5, 3, v6 // (op 3715) 

v_and_b32 v6, 3, v1 // (op 3715) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3715) 


v_lshrrev_b32 v5, 6, v1 // (op 3715) 
v_and_b32 v6, 1, v5 // (op 3715) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3715) 


v_lshl_add_u32 v6, s17, 8, v5 // (op 3715) 

v_mul_lo_u32 v5, v6, s2 // (op 3715) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 3715) 


v_lshrrev_b32 v0, 6, v1 // (op 3829) 
v_and_b32 v5, 1, v0 // (op 3829) 

v_and_b32 v0, 15, v1 // (op 3829) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 3829) 


v_and_b32 v0, 63, v1 // (op 3829) 
v_lshrrev_b32 v5, 4, v0 // (op 3829) 

v_lshlrev_b32 v0, 5, v5 // (op 3829) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 3829) 


v_mul_lo_u32 v0, v5, 6 // (op 3829) 

v_lshrrev_b32 v5, 3, v0 // (op 3829) 

v_lshrrev_b32 v0, 6, v1 // (op 3887) 
v_and_b32 v7, 1, v0 // (op 3887) 

v_and_b32 v0, 15, v1 // (op 3887) 
s_mul_i32 s1, s2, s9 // (op 3717) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3717) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3717) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3717) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3717) most significant: sum
s_add_u32 s1, s1, s19 // (op 3717) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3717) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 3887) 


v_and_b32 v0, 63, v1 // (op 3887) 
v_lshrrev_b32 v7, 4, v0 // (op 3887) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 3887) 


v_lshrrev_b32 v7, 7, v1 // (op 3903) 
v_and_b32 v8, 63, v1 // (op 3903) 
v_lshrrev_b32 v9, 4, v8 // (op 3903) 
v_lshlrev_b32 v10, 2, v9 // (op 3903) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3903) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 3903) 

v_lshrrev_b32 v9, 2, v8 // (op 3903) 

v_and_b32 v8, 3, v9 // (op 3903) 

v_and_b32 v9, 3, v1 // (op 3903) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3903) 


v_lshrrev_b32 v8, 6, v1 // (op 3903) 
v_and_b32 v9, 1, v8 // (op 3903) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3903) 


v_lshl_add_u32 v9, s17, 8, v8 // (op 3903) 

v_mul_lo_u32 v8, v9, s4 // (op 3903) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 3903) 


v_lshrrev_b32 v7, 6, v1 // (op 3973) 
v_and_b32 v8, 1, v7 // (op 3973) 

v_and_b32 v7, 15, v1 // (op 3973) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 3973) 


v_and_b32 v7, 63, v1 // (op 3973) 
v_lshrrev_b32 v8, 4, v7 // (op 3973) 

v_lshlrev_b32 v7, 5, v8 // (op 3973) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 3973) 


v_mul_lo_u32 v7, v8, 6 // (op 3973) 

v_lshrrev_b32 v8, 3, v7 // (op 3973) 

v_lshrrev_b32 v7, 6, v1 // (op 4002) 
v_and_b32 v10, 1, v7 // (op 4002) 

v_and_b32 v7, 15, v1 // (op 4002) 
s_mul_i32 s1, s4, s9 // (op 3905) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 4002) 


v_and_b32 v7, 63, v1 // (op 4002) 
v_lshrrev_b32 v10, 4, v7 // (op 4002) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 4002) 


s_mul_i32 s38, s5, s8 // (op 3905) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3905) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3905) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3905) most significant: sum
s_add_u32 s1, s1, s39 // (op 3905) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 3905) 

s_mov_b32 s40, s6 // (op 3717) reg expression
s_mov_b32 s41, s7 // (op 3717) reg expression
s_mov_b32 s42, s18 // (op 3717) reg expression
s_mov_b32 s43, 131072 // (op 3717) call()

s_mov_b32 s44, s10 // (op 3905) reg expression

s_mov_b32 s45, s11 // (op 3905) reg expression
s_mov_b32 s46, s38 // (op 3905) reg expression
s_mov_b32 s47, 131072 // (op 3905) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
v_accvgpr_write a8, 0.00000 // (op 466) call()
v_accvgpr_write a9, 0.00000 // (op 466) call()
v_accvgpr_write a10, 0.00000 // (op 466) call()
v_accvgpr_write a11, 0.00000 // (op 466) call()
v_accvgpr_write a12, 0.00000 // (op 622) call()
v_accvgpr_write a13, 0.00000 // (op 622) call()
v_accvgpr_write a14, 0.00000 // (op 622) call()
v_accvgpr_write a15, 0.00000 // (op 622) call()
v_accvgpr_write a16, 0.00000 // (op 778) call()
v_accvgpr_write a17, 0.00000 // (op 778) call()
v_accvgpr_write a18, 0.00000 // (op 778) call()
v_accvgpr_write a19, 0.00000 // (op 778) call()
v_accvgpr_write a20, 0.00000 // (op 934) call()
v_accvgpr_write a21, 0.00000 // (op 934) call()
v_accvgpr_write a22, 0.00000 // (op 934) call()
v_accvgpr_write a23, 0.00000 // (op 934) call()
v_accvgpr_write a24, 0.00000 // (op 1090) call()
v_accvgpr_write a25, 0.00000 // (op 1090) call()
v_accvgpr_write a26, 0.00000 // (op 1090) call()
v_accvgpr_write a27, 0.00000 // (op 1090) call()
v_accvgpr_write a28, 0.00000 // (op 1246) call()
v_accvgpr_write a29, 0.00000 // (op 1246) call()
v_accvgpr_write a30, 0.00000 // (op 1246) call()
v_accvgpr_write a31, 0.00000 // (op 1246) call()
v_lshrrev_b32 v12, 3, v1 // (op 3482) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 3482) 

v_and_b32 v12, -8, v1 // (op 3482) 
v_sub_u32 v14, v1, v12 // (op 3482) 

v_lshlrev_b32 v12, 4, v14 // (op 3482) 

v_mov_b32 v15, 0 // (op 3482) convert
v_mov_b32 v14, v12 // (op 3482) convert

v_mul_lo_u32 v17, v13, s13 // (op 3482) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3482) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3482) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3482) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3482) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3482) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3484) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3484) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3484) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3484) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3484) most significant: sum
s_add_u32 s1, s1, s7 // (op 3484) most significant: sum


s_mov_b32 s48, s20 // (op 3484) reg expression
s_mov_b32 s49, s21 // (op 3484) reg expression
s_mov_b32 s50, s0 // (op 3484) reg expression
s_mov_b32 s51, 131072 // (op 3484) call()

v_lshrrev_b32 v12, 3, v1 // (op 3501) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 3501) 

v_and_b32 v12, -8, v1 // (op 3501) 
v_sub_u32 v14, v1, v12 // (op 3501) 

v_lshlrev_b32 v12, 4, v14 // (op 3501) 

v_mov_b32 v15, 0 // (op 3501) convert
v_mov_b32 v14, v12 // (op 3501) convert

v_mul_lo_u32 v19, v13, s13 // (op 3501) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3501) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 3501) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3501) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 3501) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 3501) most significant half



s_mul_i32 s1, s12, s23 // (op 3503) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3503) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3503) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3503) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3503) most significant: sum
s_add_u32 s1, s1, s7 // (op 3503) most significant: sum


s_mov_b32 s48, s20 // (op 3503) reg expression
s_mov_b32 s49, s21 // (op 3503) reg expression
s_mov_b32 s50, s0 // (op 3503) reg expression
s_mov_b32 s51, 131072 // (op 3503) call()

v_and_b32 v12, -2, v1 // (op 3546) 
v_sub_u32 v13, v1, v12 // (op 3546) 

v_lshlrev_b32 v12, 6, v13 // (op 3546) 

v_mov_b32 v15, 0 // (op 3546) convert
v_mov_b32 v14, v12 // (op 3546) convert

v_and_b32 v12, 4294967294, v1 // (op 3546) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3546) 

v_mul_lo_u32 v21, v13, s25 // (op 3546) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3546) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 3546) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 3546) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 3546) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 3546) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 3546) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 3546) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 3546) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 3546) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 3546) 

v_mov_b32 v14, v12 // (op 3546) convert

s_mul_i32 s1, s9, 6 // (op 3548) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 3548) least significant
s_mul_hi_u32 s6, s8, 6 // (op 3548) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3548) most significant: sum

s_mul_i32 s7, s24, s1 // (op 3548) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 3548) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 3548) least significant
s_mul_hi_u32 s11, s24, s0 // (op 3548) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3548) most significant: sum
s_add_u32 s7, s7, s11 // (op 3548) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3548) 

s_mov_b32 s52, s26 // (op 3548) reg expression
s_mov_b32 s53, s27 // (op 3548) reg expression
s_mov_b32 s54, s0 // (op 3548) reg expression
s_mov_b32 s55, 131072 // (op 3548) call()

v_and_b32 v15, -2, v1 // (op 3565) 
v_sub_u32 v12, v1, v15 // (op 3565) 

v_lshlrev_b32 v13, 6, v12 // (op 3565) 

v_mov_b32 v21, 0 // (op 3565) convert
v_mov_b32 v20, v13 // (op 3565) convert

v_and_b32 v15, 4294967294, v1 // (op 3565) 
v_lshl_add_u32 v12, s17, 8, v15 // (op 3565) 

v_mul_lo_u32 v23, v12, s25 // (op 3565) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 3565) most significant: high of low * low
v_mul_lo_u32 v22, v12, s24 // (op 3565) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 3565) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v22 // (op 3565) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 3565) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 3565) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 3565) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 3565) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3565) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 3565) 

v_mov_b32 v15, v12 // (op 3565) convert

s_mul_i32 s1, s9, 6 // (op 3567) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 3567) least significant
s_mul_hi_u32 s6, s8, 6 // (op 3567) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3567) most significant: sum

s_mul_i32 s7, s24, s1 // (op 3567) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 3567) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 3567) least significant
s_mul_hi_u32 s11, s24, s0 // (op 3567) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3567) most significant: sum
s_add_u32 s7, s7, s11 // (op 3567) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3567) 

s_mov_b32 s52, s26 // (op 3567) reg expression
s_mov_b32 s53, s27 // (op 3567) reg expression
s_mov_b32 s54, s0 // (op 3567) reg expression
s_mov_b32 s55, 131072 // (op 3567) call()

v_lshrrev_b32 v12, 7, v1 // (op 3610) 
v_and_b32 v13, 15, v1 // (op 3610) 
v_lshl_add_u32 v20, v12, 4, v13 // (op 3610) 


v_lshl_add_u32 v12, s16, 5, v20 // (op 3610) 

v_and_b32 v13, 63, v1 // (op 3610) 
v_lshrrev_b32 v20, 4, v13 // (op 3610) 

v_mul_lo_u32 v13, v12, s28 // (op 3610) 
v_add_u32 v13, v13, v20 // (op 3610) 


s_mul_i32 s1, s28, s23 // (op 3612) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3612) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3612) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3612) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3612) most significant: sum
s_add_u32 s1, s1, s7 // (op 3612) most significant: sum


s_mov_b32 s56, s30 // (op 3612) reg expression
s_mov_b32 s57, s31 // (op 3612) reg expression
s_mov_b32 s58, s0 // (op 3612) reg expression
s_mov_b32 s59, 131072 // (op 3612) call()

v_lshrrev_b32 v12, 7, v1 // (op 3629) 
v_and_b32 v20, 15, v1 // (op 3629) 
v_lshl_add_u32 v21, v12, 4, v20 // (op 3629) 


v_lshl_add_u32 v12, s16, 5, v21 // (op 3629) 

v_and_b32 v20, 63, v1 // (op 3629) 
v_lshrrev_b32 v21, 4, v20 // (op 3629) 

v_mul_lo_u32 v20, v12, s28 // (op 3629) 
v_add_u32 v20, v20, v21 // (op 3629) 


s_mul_i32 s1, s28, s23 // (op 3631) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3631) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3631) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3631) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3631) most significant: sum
s_add_u32 s1, s1, s7 // (op 3631) most significant: sum


s_mov_b32 s56, s30 // (op 3631) reg expression
s_mov_b32 s57, s31 // (op 3631) reg expression
s_mov_b32 s58, s0 // (op 3631) reg expression
s_mov_b32 s59, 131072 // (op 3631) call()

v_lshl_add_u32 v12, s17, 8, v1 // (op 3674) 
v_mul_lo_u32 v21, v12, s32 // (op 3674) 

s_mul_i32 s1, s32, s9 // (op 3676) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3676) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3676) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3676) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3676) most significant: sum
s_add_u32 s1, s1, s7 // (op 3676) most significant: sum


s_mov_b32 s60, s34 // (op 3676) reg expression
s_mov_b32 s61, s35 // (op 3676) reg expression
s_mov_b32 s62, s0 // (op 3676) reg expression
s_mov_b32 s63, 131072 // (op 3676) call()

v_lshl_add_u32 v12, s17, 8, v1 // (op 3693) 
v_mul_lo_u32 v22, v12, s32 // (op 3693) 

s_mul_i32 s1, s32, s9 // (op 3695) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3695) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3695) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3695) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3695) most significant: sum
s_add_u32 s1, s1, s7 // (op 3695) most significant: sum


s_mov_b32 s60, s34 // (op 3695) reg expression
s_mov_b32 s61, s35 // (op 3695) reg expression
s_mov_b32 s62, s0 // (op 3695) reg expression
s_mov_b32 s63, 131072 // (op 3695) call()




buffer_load_ubyte v12, v13, s[56:59], 0 offen offset: 0 // (op 3140) Load value

s_mul_i32 s1, s25, 6 // (op 3144) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 3144) least significant
s_mul_hi_u32 s6, s24, 6 // (op 3144) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3144) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3144) 


v_mov_b32 v23, v14 // (op 3144) 
buffer_load_dwordx4 v[26:29], v23, s[52:55], 0 offen // (op 3144) Load value
buffer_load_dwordx2 v[30:31], v23, s[52:55], 0 offen offset:16 // (op 3144) Load value
buffer_load_dwordx4 v[32:35], v23, s[52:55], 0 offen offset:24 // (op 3144) Load value
buffer_load_dwordx2 v[36:37], v23, s[52:55], 0 offen offset:40 // (op 3144) Load value
v_add_u32 v23, v23, v24 // (op 3144) 
buffer_load_dwordx4 v[38:41], v23, s[52:55], 0 offen // (op 3144) Load value
buffer_load_dwordx2 v[42:43], v23, s[52:55], 0 offen offset:16 // (op 3144) Load value
buffer_load_dwordx4 v[44:47], v23, s[52:55], 0 offen offset:24 // (op 3144) Load value
buffer_load_dwordx2 v[48:49], v23, s[52:55], 0 offen offset:40 // (op 3144) Load value





buffer_load_dword v23, v21, s[60:63], 0 offen // (op 3147) Load value
v_and_b32 v24, -8, v1 // (op 3759) 
v_sub_u32 v25, v1, v24 // (op 3759) 

v_lshlrev_b32 v24, 4, v25 // (op 3759) 

v_lshrrev_b32 v25, 3, v1 // (op 3759) 
v_lshl_add_u32 v50, v25, 7, v24 // (op 3759) 


s_nop 0
v_readfirstlane_b32 s0, v50 // (op 3759) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 3431) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 3431) Load value direct to lds




buffer_load_ubyte v24, v13, s[56:59], 0 offen offset:4 // (op 3154) Load value


s_mul_i32 s1, s25, 6 // (op 3158) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 3158) least significant
s_mul_hi_u32 s6, s24, 6 // (op 3158) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3158) most significant: sum

v_lshrrev_b64 v[50:51], 3, s[0:1] // (op 3158) 


v_mov_b32 v13, v14 // (op 3158) 
buffer_load_dwordx4 v[52:55], v13, s[52:55], 0 offen offset:96 // (op 3158) Load value
buffer_load_dwordx2 v[56:57], v13, s[52:55], 0 offen offset:112 // (op 3158) Load value
buffer_load_dwordx4 v[58:61], v13, s[52:55], 0 offen offset:120 // (op 3158) Load value
buffer_load_dwordx2 v[62:63], v13, s[52:55], 0 offen offset:136 // (op 3158) Load value
v_add_u32 v13, v13, v50 // (op 3158) 
buffer_load_dwordx4 v[64:67], v13, s[52:55], 0 offen offset:96 // (op 3158) Load value
buffer_load_dwordx2 v[68:69], v13, s[52:55], 0 offen offset:112 // (op 3158) Load value
buffer_load_dwordx4 v[70:73], v13, s[52:55], 0 offen offset:120 // (op 3158) Load value
buffer_load_dwordx2 v[74:75], v13, s[52:55], 0 offen offset:136 // (op 3158) Load value






buffer_load_dword v25, v21, s[60:63], 0 offen offset:4 // (op 3161) Load value

v_and_b32 v21, -8, v1 // (op 3932) 
v_sub_u32 v13, v1, v21 // (op 3932) 

v_lshlrev_b32 v14, 4, v13 // (op 3932) 

v_lshrrev_b32 v13, 3, v1 // (op 3932) 
v_lshl_add_u32 v21, v13, 7, v14 // (op 3932) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 3932) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 3436) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3436) reg expression
s_mov_b32 s6, 128 // (op 3436) call()
buffer_load_dwordx4 v16, s[48:51], s6 offen offset: 0 lds // (op 3436) Load value direct to lds




v_and_b32 v21, -2, v1 // (op 3812) 
v_sub_u32 v13, v1, v21 // (op 3812) 

v_lshlrev_b32 v14, 6, v13 // (op 3812) 

v_lshrrev_b32 v13, 1, v1 // (op 3812) 
v_lshl_add_u32 v21, v13, 8, v14 // (op 3812) 


v_mul_lo_u32 v13, v21, 6 // (op 3812) 

v_lshrrev_b32 v14, 3, v13 // (op 3812) 




s_waitcnt vmcnt(20)
ds_write_b128 v14, v[26:29] offset:8192 // (op 3166) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v14, v[30:31] offset:8208 // (op 3166) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v14, v[32:35] offset:8216 // (op 3166) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v14, v[36:37] offset:8232 // (op 3166) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v14, v[38:41] offset:8288 // (op 3166) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v14, v[42:43] offset:8304 // (op 3166) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v14, v[44:47] offset:8312 // (op 3166) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v14, v[48:49] offset:8328 // (op 3166) Store local data 

v_lshlrev_b32 v21, 2, v1 // (op 3870) 



s_waitcnt vmcnt(12)
ds_write_b32 v21, v23 offset:32768 // (op 3167) Store local data 

v_and_b32 v21, -8, v1 // (op 3731) 
v_sub_u32 v13, v1, v21 // (op 3731) 

v_lshlrev_b32 v14, 4, v13 // (op 3731) 

v_lshrrev_b32 v13, 3, v1 // (op 3731) 
v_lshl_add_u32 v21, v13, 7, v14 // (op 3731) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 3731) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -2, v1 // (op 3783) 
v_sub_u32 v13, v1, v21 // (op 3783) 

v_lshlrev_b32 v14, 6, v13 // (op 3783) 

v_lshrrev_b32 v13, 1, v1 // (op 3783) 
v_lshl_add_u32 v21, v13, 8, v14 // (op 3783) 


v_mul_lo_u32 v13, v21, 6 // (op 3783) 

v_lshrrev_b32 v14, 3, v13 // (op 3783) 

v_lshlrev_b32 v13, 2, v1 // (op 3841) 
v_and_b32 v21, -8, v1 // (op 3917) 
v_sub_u32 v16, v1, v21 // (op 3917) 

v_lshlrev_b32 v17, 4, v16 // (op 3917) 

v_lshrrev_b32 v16, 3, v1 // (op 3917) 
v_lshl_add_u32 v21, v16, 7, v17 // (op 3917) 


s_nop 0
v_readfirstlane_b32 s1, v21 // (op 3917) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -2, v1 // (op 3956) 
v_sub_u32 v16, v1, v21 // (op 3956) 

v_lshlrev_b32 v17, 6, v16 // (op 3956) 

v_lshrrev_b32 v16, 1, v1 // (op 3956) 
v_lshl_add_u32 v21, v16, 8, v17 // (op 3956) 


v_mul_lo_u32 v16, v21, 6 // (op 3956) 

v_lshrrev_b32 v17, 3, v16 // (op 3956) 

v_lshlrev_b32 v16, 2, v1 // (op 3985) 
s_mov_b32 s6, 0 // (op 1248) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 1247) 
s_mov_b32 s7, s10 // (op 1247) convert

s_and_b32 s10, s7, 4294967294 // (op 1247) 

s_cmp_lt_u32 s6, s10 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1247 // (op 1247) Condition: Top (jump to RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1247 if false)
RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1247:
s_add_u32 s6, s6, 2 // (op 1249) 
s_waitcnt vmcnt(11)
s_barrier  // (op 4061) 



ds_read_b128 v[76:79], v3 // (op 67) Load local data 
ds_read_b128 v[80:83], v3 offset:64 // (op 67) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4074) 



ds_read_b128 v[84:87], v5 offset:8192 // (op 72) Load local data 
ds_read_b64 v[88:89], v5 offset:8208 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4088) 



ds_read_u8 v21, v0 offset:32768 // (op 81) Load local data 



ds_read_b128 v[90:93], v5 offset:11264 // (op 343) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:83], v[84:89], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 84) 
ds_read_b64 v[94:95], v5 offset:11280 // (op 343) Load local data 





ds_read_u8 v21, v0 offset:32896 // (op 346) Load local data 



ds_read_b128 v[84:87], v5 offset:14336 // (op 499) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:83], v[90:95], a[4:7], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 347) 
ds_read_b64 v[88:89], v5 offset:14352 // (op 499) Load local data 





ds_read_u8 v21, v0 offset:33024 // (op 502) Load local data 



ds_read_b128 v[90:93], v5 offset:17408 // (op 655) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:83], v[84:89], a[8:11], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 503) 
ds_read_b64 v[94:95], v5 offset:17424 // (op 655) Load local data 





ds_read_u8 v21, v0 offset:33152 // (op 658) Load local data 



ds_read_b128 v[84:87], v5 offset:20480 // (op 811) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:83], v[90:95], a[12:15], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 659) 
ds_read_b64 v[88:89], v5 offset:20496 // (op 811) Load local data 





ds_read_u8 v21, v0 offset:33280 // (op 814) Load local data 



ds_read_b128 v[90:93], v5 offset:23552 // (op 967) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:83], v[84:89], a[16:19], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 815) 
ds_read_b64 v[94:95], v5 offset:23568 // (op 967) Load local data 





ds_read_u8 v21, v0 offset:33408 // (op 970) Load local data 



ds_read_b128 v[84:87], v5 offset:26624 // (op 1123) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[90:95], a[20:23], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 971) 
ds_read_b64 v[88:89], v5 offset:26640 // (op 1123) Load local data 





ds_read_u8 v21, v0 offset:33536 // (op 1126) Load local data 



ds_read_b128 v[90:93], v5 offset:29696 // (op 1279) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[84:89], a[24:27], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1127) 
ds_read_b64 v[94:95], v5 offset:29712 // (op 1279) Load local data 





ds_read_u8 v21, v0 offset:33664 // (op 1282) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[90:95], a[28:31], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1283) 





v_mov_b32 v21, v20 // (op 77) 

buffer_load_ubyte v12, v21, s[56:59], 0 offen offset:8 // (op 77) Load value

s_mul_i32 s11, s25, 6 // (op 10) most significant: low of high * low
s_mul_i32 s10, s24, 6 // (op 10) least significant
s_mul_hi_u32 s7, s24, 6 // (op 10) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 10) most significant: sum

v_lshrrev_b64 v[50:51], 3, s[10:11] // (op 10) 


v_mov_b32 v76, v15 // (op 10) 
v_mov_b32 v77, v76 // (op 10) 
buffer_load_dwordx4 v[26:29], v77, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[30:31], v77, s[52:55], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[32:35], v77, s[52:55], 0 offen offset:216 // (op 10) Load value
buffer_load_dwordx2 v[36:37], v77, s[52:55], 0 offen offset:232 // (op 10) Load value
v_add_u32 v77, v77, v50 // (op 10) 
buffer_load_dwordx4 v[38:41], v77, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[42:43], v77, s[52:55], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[44:47], v77, s[52:55], 0 offen offset:216 // (op 10) Load value
buffer_load_dwordx2 v[48:49], v77, s[52:55], 0 offen offset:232 // (op 10) Load value




v_mov_b32 v50, v22 // (op 21) 

buffer_load_dword v23, v50, s[60:63], 0 offen offset:8 // (op 21) Load value
s_barrier  // (op 4064) 


v_mov_b64 v[78:79], v[18:19] // (op 3415) 

s_mov_b32 m0, s0 // (op 3415) reg expression
s_mov_b32 s7, 256 // (op 3415) call()
buffer_load_dwordx4 v78, s[48:51], s7 offen offset: 0 lds // (op 3415) Load value direct to lds

s_barrier  // (op 4111) 



s_waitcnt vmcnt(20)
ds_write_b128 v17, v[52:55] offset:33792 // (op 239) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v17, v[56:57] offset:33808 // (op 239) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v17, v[58:61] offset:33816 // (op 239) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v17, v[62:63] offset:33832 // (op 239) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v17, v[64:67] offset:33888 // (op 239) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v17, v[68:69] offset:33904 // (op 239) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v17, v[70:73] offset:33912 // (op 239) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v17, v[74:75] offset:33928 // (op 239) Store local data 
s_barrier  // (op 4117) 



s_waitcnt vmcnt(12)
ds_write_b32 v16, v25 offset:58368 // (op 240) Store local data 
s_waitcnt vmcnt(11)
s_barrier  // (op 4102) 



ds_read_b128 v[80:83], v4 offset:4096 // (op 231) Load local data 
ds_read_b128 v[84:87], v4 offset:4160 // (op 231) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4108) 



ds_read_b128 v[88:91], v8 offset:33792 // (op 233) Load local data 
ds_read_b64 v[92:93], v8 offset:33808 // (op 233) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4114) 



ds_read_u8 v51, v7 offset:58368 // (op 237) Load local data 



ds_read_b128 v[94:97], v8 offset:36864 // (op 352) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:87], v[88:93], a[0:3], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 238) 
ds_read_b64 v[98:99], v8 offset:36880 // (op 352) Load local data 





ds_read_u8 v51, v7 offset:58496 // (op 355) Load local data 



ds_read_b128 v[88:91], v8 offset:39936 // (op 508) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:87], v[94:99], a[4:7], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 348) 
ds_read_b64 v[92:93], v8 offset:39952 // (op 508) Load local data 





ds_read_u8 v51, v7 offset:58624 // (op 511) Load local data 



ds_read_b128 v[94:97], v8 offset:43008 // (op 664) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[80:87], v[88:93], a[8:11], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 504) 
ds_read_b64 v[98:99], v8 offset:43024 // (op 664) Load local data 





ds_read_u8 v51, v7 offset:58752 // (op 667) Load local data 



ds_read_b128 v[88:91], v8 offset:46080 // (op 820) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[80:87], v[94:99], a[12:15], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 660) 
ds_read_b64 v[92:93], v8 offset:46096 // (op 820) Load local data 





ds_read_u8 v51, v7 offset:58880 // (op 823) Load local data 



ds_read_b128 v[94:97], v8 offset:49152 // (op 976) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:87], v[88:93], a[16:19], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 816) 
ds_read_b64 v[98:99], v8 offset:49168 // (op 976) Load local data 





ds_read_u8 v51, v7 offset:59008 // (op 979) Load local data 



ds_read_b128 v[88:91], v8 offset:52224 // (op 1132) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:87], v[94:99], a[20:23], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 972) 
ds_read_b64 v[92:93], v8 offset:52240 // (op 1132) Load local data 





ds_read_u8 v51, v7 offset:59136 // (op 1135) Load local data 



ds_read_b128 v[94:97], v8 offset:55296 // (op 1288) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:87], v[88:93], a[24:27], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1128) 
ds_read_b64 v[98:99], v8 offset:55312 // (op 1288) Load local data 





s_barrier  // (op 4218) 
ds_read_u8 v51, v7 offset:59264 // (op 1291) Load local data 
s_barrier  // (op 4222) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:87], v[94:99], a[28:31], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1284) 






buffer_load_ubyte v24, v21, s[56:59], 0 offen offset:12 // (op 235) Load value


v_add_u32 v20, 8, v20 // (op 3621) 
s_mul_i32 s11, s25, 6 // (op 241) most significant: low of high * low
s_mul_i32 s10, s24, 6 // (op 241) least significant
s_mul_hi_u32 s7, s24, 6 // (op 241) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 241) most significant: sum

v_lshrrev_b64 v[80:81], 3, s[10:11] // (op 241) 


v_mov_b32 v21, v76 // (op 241) 
buffer_load_dwordx4 v[52:55], v21, s[52:55], 0 offen offset:288 // (op 241) Load value
buffer_load_dwordx2 v[56:57], v21, s[52:55], 0 offen offset:304 // (op 241) Load value
buffer_load_dwordx4 v[58:61], v21, s[52:55], 0 offen offset:312 // (op 241) Load value
buffer_load_dwordx2 v[62:63], v21, s[52:55], 0 offen offset:328 // (op 241) Load value
v_add_u32 v21, v21, v80 // (op 241) 
buffer_load_dwordx4 v[64:67], v21, s[52:55], 0 offen offset:288 // (op 241) Load value
buffer_load_dwordx2 v[68:69], v21, s[52:55], 0 offen offset:304 // (op 241) Load value
buffer_load_dwordx4 v[70:73], v21, s[52:55], 0 offen offset:312 // (op 241) Load value
buffer_load_dwordx2 v[74:75], v21, s[52:55], 0 offen offset:328 // (op 241) Load value






v_add_u32 v15, 192, v15 // (op 3557) 
buffer_load_dword v25, v50, s[60:63], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 4105) 
v_add_u32 v22, 8, v22 // (op 3685) 


s_add_u32 s7, 4096, s1 // (op 3426) 

s_mov_b32 m0, s7 // (op 3426) reg expression
s_mov_b32 s10, 384 // (op 3426) call()
buffer_load_dwordx4 v78, s[48:51], s10 offen offset: 0 lds // (op 3426) Load value direct to lds



s_barrier  // (op 4077) 
v_add_co_u32 v18, vcc, 256, v18 // (op 3493) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3493) most significant half
s_waitcnt vmcnt(20)
ds_write_b128 v14, v[26:29] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v14, v[30:31] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v14, v[32:35] offset:8216 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v14, v[36:37] offset:8232 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v14, v[38:41] offset:8288 // (op 13) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v14, v[42:43] offset:8304 // (op 13) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v14, v[44:47] offset:8312 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v14, v[48:49] offset:8328 // (op 13) Store local data 
s_barrier  // (op 4091) 



s_waitcnt vmcnt(12)
ds_write_b32 v13, v23 offset:32768 // (op 26) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 1247) 
s_mov_b32 s7, s10 // (op 1247) convert

s_and_b32 s10, s7, 4294967294 // (op 1247) 

s_cmp_lt_u32 s6, s10 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1247 // (op 1247) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1247 if true)
RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1247:














s_barrier  // (op 4202) 
v_lshrrev_b32 v12, 3, v1 // (op 3449) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3449) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 3449) 

s_lshr_b32 s6, s0, 1 // (op 3449) 

v_and_b32 v12, -8, v1 // (op 3449) 
v_sub_u32 v14, v1, v12 // (op 3449) 

v_lshlrev_b32 v12, 4, v14 // (op 3449) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 3449) 


v_mov_b32 v23, 0 // (op 3449) convert
v_mov_b32 v22, v14 // (op 3449) convert

v_mul_lo_u32 v15, v13, s13 // (op 3449) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3449) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 3449) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 3449) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v22 // (op 3449) least significant half
v_addc_co_u32 v15, s[0:1], v15, v23, s[0:1] // (op 3449) most significant half



s_mul_i32 s1, s12, s23 // (op 3451) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3451) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3451) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3451) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3451) most significant: sum
s_add_u32 s1, s1, s7 // (op 3451) most significant: sum


s_mov_b32 s48, s20 // (op 3451) reg expression
s_mov_b32 s49, s21 // (op 3451) reg expression
s_mov_b32 s50, s0 // (op 3451) reg expression
s_mov_b32 s51, 131072 // (op 3451) call()

v_and_b32 v12, -2, v1 // (op 3523) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3523) 
v_sub_u32 v13, v1, v12 // (op 3523) 

s_lshr_b32 s6, s0, 1 // (op 3523) 

v_lshlrev_b32 v12, 6, v13 // (op 3523) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 3523) 


v_mov_b32 v17, 0 // (op 3523) convert
v_mov_b32 v16, v13 // (op 3523) convert

v_and_b32 v12, 4294967294, v1 // (op 3523) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3523) 

v_mul_lo_u32 v19, v13, s25 // (op 3523) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3523) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 3523) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3523) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 3523) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 3523) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 3523) most significant: low of high * low
v_mul_hi_u32 v18, v12, 6 // (op 3523) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 3523) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 3523) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 3523) 

v_mov_b32 v16, v12 // (op 3523) convert

s_mul_i32 s1, s9, 6 // (op 3525) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 3525) least significant
s_mul_hi_u32 s6, s8, 6 // (op 3525) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3525) most significant: sum

s_mul_i32 s7, s24, s1 // (op 3525) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 3525) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 3525) least significant
s_mul_hi_u32 s11, s24, s0 // (op 3525) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3525) most significant: sum
s_add_u32 s7, s7, s11 // (op 3525) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3525) 

s_mov_b32 s52, s26 // (op 3525) reg expression
s_mov_b32 s53, s27 // (op 3525) reg expression
s_mov_b32 s54, s0 // (op 3525) reg expression
s_mov_b32 s55, 131072 // (op 3525) call()

v_lshrrev_b32 v12, 7, v1 // (op 3587) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3587) 
v_and_b32 v13, 15, v1 // (op 3587) 
s_lshr_b32 s6, s0, 1 // (op 3587) 

v_lshl_add_u32 v17, v12, 4, v13 // (op 3587) 


v_lshl_add_u32 v12, s16, 5, v17 // (op 3587) 

v_and_b32 v13, 63, v1 // (op 3587) 
v_lshrrev_b32 v17, 4, v13 // (op 3587) 

v_lshl_add_u32 v13, s6, 3, v17 // (op 3587) 


v_mul_lo_u32 v17, v12, s28 // (op 3587) 
v_add_u32 v17, v17, v13 // (op 3587) 


s_mul_i32 s1, s28, s23 // (op 3589) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3589) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3589) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3589) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3589) most significant: sum
s_add_u32 s1, s1, s7 // (op 3589) most significant: sum


s_mov_b32 s56, s30 // (op 3589) reg expression
s_mov_b32 s57, s31 // (op 3589) reg expression
s_mov_b32 s58, s0 // (op 3589) reg expression
s_mov_b32 s59, 131072 // (op 3589) call()

v_lshl_add_u32 v12, s17, 8, v1 // (op 3651) 
v_mul_lo_u32 v13, v12, s32 // (op 3651) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 3651) 
s_lshr_b32 s6, s0, 1 // (op 3651) 

v_lshl_add_u32 v12, s6, 3, v13 // (op 3651) 


s_mul_i32 s1, s32, s9 // (op 3653) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3653) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3653) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3653) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3653) most significant: sum
s_add_u32 s1, s1, s7 // (op 3653) most significant: sum


s_mov_b32 s60, s34 // (op 3653) reg expression
s_mov_b32 s61, s35 // (op 3653) reg expression
s_mov_b32 s62, s0 // (op 3653) reg expression
s_mov_b32 s63, 131072 // (op 3653) call()

v_and_b32 v13, -8, v1 // (op 3744) 
v_sub_u32 v18, v1, v13 // (op 3744) 

v_lshlrev_b32 v13, 4, v18 // (op 3744) 

v_lshrrev_b32 v18, 3, v1 // (op 3744) 
v_lshl_add_u32 v19, v18, 7, v13 // (op 3744) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 3744) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 3796) 
v_sub_u32 v18, v1, v13 // (op 3796) 

v_lshlrev_b32 v13, 6, v18 // (op 3796) 

v_lshrrev_b32 v18, 1, v1 // (op 3796) 
v_lshl_add_u32 v19, v18, 8, v13 // (op 3796) 


v_mul_lo_u32 v13, v19, 6 // (op 3796) 

v_lshrrev_b32 v18, 3, v13 // (op 3796) 

v_lshlrev_b32 v13, 2, v1 // (op 3854) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1253) 
s_and_b32 s1, s6, 4294967294 // (op 1253) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1252) 
s_mov_b32 s10, s6 // (op 1252) convert

s_cmp_lt_u32 s1, s10 // (op 1252) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1252 // (op 1252) Condition: Top (jump to RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1252 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1252:
s_barrier  // (op 4067) 
s_add_u32 s1, s1, 1 // (op 1254) 


v_mov_b64 v[20:21], v[14:15] // (op 3420) 

s_mov_b32 m0, s0 // (op 3420) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 3420) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 3439) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3439) most significant half

s_mul_i32 s7, s25, 6 // (op 183) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 183) least significant
s_mul_hi_u32 s10, s24, 6 // (op 183) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 183) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[6:7] // (op 183) 


v_mov_b32 v19, v16 // (op 183) 
v_mov_b32 v22, v19 // (op 183) 
buffer_load_dwordx4 v[24:27], v22, s[52:55], 0 offen // (op 183) Load value
buffer_load_dwordx2 v[28:29], v22, s[52:55], 0 offen offset:16 // (op 183) Load value
buffer_load_dwordx4 v[30:33], v22, s[52:55], 0 offen offset:24 // (op 183) Load value
buffer_load_dwordx2 v[34:35], v22, s[52:55], 0 offen offset:40 // (op 183) Load value
v_add_u32 v22, v22, v20 // (op 183) 
buffer_load_dwordx4 v[36:39], v22, s[52:55], 0 offen // (op 183) Load value
buffer_load_dwordx2 v[40:41], v22, s[52:55], 0 offen offset:16 // (op 183) Load value
buffer_load_dwordx4 v[42:45], v22, s[52:55], 0 offen offset:24 // (op 183) Load value
buffer_load_dwordx2 v[46:47], v22, s[52:55], 0 offen offset:40 // (op 183) Load value



v_add_u32 v16, 96, v16 // (op 3515) 


s_barrier  // (op 4080) 
v_mov_b32 v19, v12 // (op 184) 




s_waitcnt vmcnt(7)
ds_write_b128 v18, v[24:27] offset:8192 // (op 181) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v18, v[28:29] offset:8208 // (op 181) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v18, v[30:33] offset:8216 // (op 181) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v18, v[34:35] offset:8232 // (op 181) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v18, v[36:39] offset:8288 // (op 181) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[40:41] offset:8304 // (op 181) Store local data 
buffer_load_dword v20, v19, s[60:63], 0 offen // (op 184) Load value

v_add_u32 v12, 4, v12 // (op 3643) 
s_waitcnt vmcnt(2)
ds_write_b128 v18, v[42:45] offset:8312 // (op 181) Store local data 
s_barrier  // (op 4070) 
s_waitcnt vmcnt(1)
ds_write_b64 v18, v[46:47] offset:8328 // (op 181) Store local data 




ds_read_b128 v[22:25], v3 // (op 173) Load local data 
s_barrier  // (op 4094) 
ds_read_b128 v[26:29], v3 offset:64 // (op 173) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 4084) 
s_waitcnt vmcnt(0)
ds_write_b32 v13, v20 offset:32768 // (op 182) Store local data 




ds_read_b128 v[30:33], v5 offset:8192 // (op 175) Load local data 
ds_read_b64 v[34:35], v5 offset:8208 // (op 175) Load local data 


v_mov_b32 v19, v17 // (op 177) 

buffer_load_ubyte v20, v19, s[56:59], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v17, 4, v17 // (op 3579) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4098) 



ds_read_u8 v19, v0 offset:32768 // (op 179) Load local data 



ds_read_b128 v[36:39], v5 offset:11264 // (op 324) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 180) 
ds_read_b64 v[40:41], v5 offset:11280 // (op 324) Load local data 





ds_read_u8 v19, v0 offset:32896 // (op 327) Load local data 



ds_read_b128 v[30:33], v5 offset:14336 // (op 480) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[36:41], a[4:7], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 328) 
ds_read_b64 v[34:35], v5 offset:14352 // (op 480) Load local data 





ds_read_u8 v19, v0 offset:33024 // (op 483) Load local data 



ds_read_b128 v[36:39], v5 offset:17408 // (op 636) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:29], v[30:35], a[8:11], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 484) 
ds_read_b64 v[40:41], v5 offset:17424 // (op 636) Load local data 





ds_read_u8 v19, v0 offset:33152 // (op 639) Load local data 



ds_read_b128 v[30:33], v5 offset:20480 // (op 792) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:29], v[36:41], a[12:15], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 640) 
ds_read_b64 v[34:35], v5 offset:20496 // (op 792) Load local data 





ds_read_u8 v19, v0 offset:33280 // (op 795) Load local data 



ds_read_b128 v[36:39], v5 offset:23552 // (op 948) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:29], v[30:35], a[16:19], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 796) 
ds_read_b64 v[40:41], v5 offset:23568 // (op 948) Load local data 





ds_read_u8 v19, v0 offset:33408 // (op 951) Load local data 



ds_read_b128 v[30:33], v5 offset:26624 // (op 1104) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:29], v[36:41], a[20:23], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 952) 
ds_read_b64 v[34:35], v5 offset:26640 // (op 1104) Load local data 





ds_read_u8 v19, v0 offset:33536 // (op 1107) Load local data 



ds_read_b128 v[36:39], v5 offset:29696 // (op 1260) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:29], v[30:35], a[24:27], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1108) 
ds_read_b64 v[40:41], v5 offset:29712 // (op 1260) Load local data 





ds_read_u8 v19, v0 offset:33664 // (op 1263) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:29], v[36:41], a[28:31], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 1264) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1252) 
s_mov_b32 s10, s6 // (op 1252) convert

s_cmp_lt_u32 s1, s10 // (op 1252) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1252 // (op 1252) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1252 if true)
RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1252:






















s_barrier  // (op 4206) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_125
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

v_mov_b32 v0, v9 // (op 54) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 54) Store value



v_accvgpr_read v12, a4 // (op 1576) 
v_accvgpr_read v13, a5 // (op 1576) 
v_accvgpr_read v14, a6 // (op 1576) 
v_accvgpr_read v15, a7 // (op 1576) 
v_mul_f32 v16, v11, v12 // (op 1576) 
v_mul_f32 v17, v11, v13 // (op 1576) 
v_mul_f32 v18, v11, v14 // (op 1576) 
v_mul_f32 v19, v11, v15 // (op 1576) 

v_lshrrev_b32 v12, 16, v16 // (op 1578) 
v_lshrrev_b32 v13, 16, v17 // (op 1578) 
v_lshrrev_b32 v14, 16, v18 // (op 1578) 
v_lshrrev_b32 v15, 16, v19 // (op 1578) 


v_pack_b32_f16 v4, v12, v13 // (op 1579) 
v_pack_b32_f16 v5, v14, v15 // (op 1579) 

s_lshl_b32 s0, s4, 6 // (op 1579) 
v_add_u32 v0, v9, s0 // (op 1579) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1579) Store value



v_accvgpr_read v12, a8 // (op 1590) 
v_accvgpr_read v13, a9 // (op 1590) 
v_accvgpr_read v14, a10 // (op 1590) 
v_accvgpr_read v15, a11 // (op 1590) 
v_mul_f32 v16, v11, v12 // (op 1590) 
v_mul_f32 v17, v11, v13 // (op 1590) 
v_mul_f32 v18, v11, v14 // (op 1590) 
v_mul_f32 v19, v11, v15 // (op 1590) 

v_lshrrev_b32 v12, 16, v16 // (op 1592) 
v_lshrrev_b32 v13, 16, v17 // (op 1592) 
v_lshrrev_b32 v14, 16, v18 // (op 1592) 
v_lshrrev_b32 v15, 16, v19 // (op 1592) 


v_pack_b32_f16 v4, v12, v13 // (op 1593) 
v_pack_b32_f16 v5, v14, v15 // (op 1593) 

s_lshl_b32 s0, s4, 7 // (op 1593) 
v_add_u32 v0, v9, s0 // (op 1593) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1593) Store value



v_accvgpr_read v12, a12 // (op 1604) 
v_accvgpr_read v13, a13 // (op 1604) 
v_accvgpr_read v14, a14 // (op 1604) 
v_accvgpr_read v15, a15 // (op 1604) 
v_mul_f32 v16, v11, v12 // (op 1604) 
v_mul_f32 v17, v11, v13 // (op 1604) 
v_mul_f32 v18, v11, v14 // (op 1604) 
v_mul_f32 v19, v11, v15 // (op 1604) 

v_lshrrev_b32 v12, 16, v16 // (op 1606) 
v_lshrrev_b32 v13, 16, v17 // (op 1606) 
v_lshrrev_b32 v14, 16, v18 // (op 1606) 
v_lshrrev_b32 v15, 16, v19 // (op 1606) 


v_pack_b32_f16 v4, v12, v13 // (op 1607) 
v_pack_b32_f16 v5, v14, v15 // (op 1607) 

s_lshl_b32 s0, s4, 6 // (op 1607) 
s_mul_i32 s1, 3, s0 // (op 1607) 

v_add_u32 v0, v9, s1 // (op 1607) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1607) Store value



v_accvgpr_read v12, a16 // (op 1618) 
v_accvgpr_read v13, a17 // (op 1618) 
v_accvgpr_read v14, a18 // (op 1618) 
v_accvgpr_read v15, a19 // (op 1618) 
v_mul_f32 v16, v11, v12 // (op 1618) 
v_mul_f32 v17, v11, v13 // (op 1618) 
v_mul_f32 v18, v11, v14 // (op 1618) 
v_mul_f32 v19, v11, v15 // (op 1618) 

v_lshrrev_b32 v12, 16, v16 // (op 1620) 
v_lshrrev_b32 v13, 16, v17 // (op 1620) 
v_lshrrev_b32 v14, 16, v18 // (op 1620) 
v_lshrrev_b32 v15, 16, v19 // (op 1620) 


v_pack_b32_f16 v4, v12, v13 // (op 1621) 
v_pack_b32_f16 v5, v14, v15 // (op 1621) 

s_lshl_b32 s0, s4, 8 // (op 1621) 
v_add_u32 v0, v9, s0 // (op 1621) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1621) Store value



v_accvgpr_read v12, a20 // (op 1632) 
v_accvgpr_read v13, a21 // (op 1632) 
v_accvgpr_read v14, a22 // (op 1632) 
v_accvgpr_read v15, a23 // (op 1632) 
v_mul_f32 v16, v11, v12 // (op 1632) 
v_mul_f32 v17, v11, v13 // (op 1632) 
v_mul_f32 v18, v11, v14 // (op 1632) 
v_mul_f32 v19, v11, v15 // (op 1632) 

v_lshrrev_b32 v12, 16, v16 // (op 1634) 
v_lshrrev_b32 v13, 16, v17 // (op 1634) 
v_lshrrev_b32 v14, 16, v18 // (op 1634) 
v_lshrrev_b32 v15, 16, v19 // (op 1634) 


v_pack_b32_f16 v4, v12, v13 // (op 1635) 
v_pack_b32_f16 v5, v14, v15 // (op 1635) 

s_lshl_b32 s0, s4, 6 // (op 1635) 
s_mul_i32 s1, 5, s0 // (op 1635) 

v_add_u32 v0, v9, s1 // (op 1635) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1635) Store value



v_accvgpr_read v12, a24 // (op 1646) 
v_accvgpr_read v13, a25 // (op 1646) 
v_accvgpr_read v14, a26 // (op 1646) 
v_accvgpr_read v15, a27 // (op 1646) 
v_mul_f32 v16, v11, v12 // (op 1646) 
v_mul_f32 v17, v11, v13 // (op 1646) 
v_mul_f32 v18, v11, v14 // (op 1646) 
v_mul_f32 v19, v11, v15 // (op 1646) 

v_lshrrev_b32 v12, 16, v16 // (op 1648) 
v_lshrrev_b32 v13, 16, v17 // (op 1648) 
v_lshrrev_b32 v14, 16, v18 // (op 1648) 
v_lshrrev_b32 v15, 16, v19 // (op 1648) 


v_pack_b32_f16 v4, v12, v13 // (op 1649) 
v_pack_b32_f16 v5, v14, v15 // (op 1649) 

s_lshl_b32 s0, s4, 6 // (op 1649) 
s_mul_i32 s1, 6, s0 // (op 1649) 

v_add_u32 v0, v9, s1 // (op 1649) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1649) Store value



v_accvgpr_read v12, a28 // (op 1660) 
v_accvgpr_read v13, a29 // (op 1660) 
v_accvgpr_read v14, a30 // (op 1660) 
v_accvgpr_read v15, a31 // (op 1660) 
v_mul_f32 v16, v11, v12 // (op 1660) 
v_mul_f32 v17, v11, v13 // (op 1660) 
v_mul_f32 v18, v11, v14 // (op 1660) 
v_mul_f32 v19, v11, v15 // (op 1660) 

v_lshrrev_b32 v12, 16, v16 // (op 1662) 
v_lshrrev_b32 v13, 16, v17 // (op 1662) 
v_lshrrev_b32 v14, 16, v18 // (op 1662) 
v_lshrrev_b32 v15, 16, v19 // (op 1662) 


v_pack_b32_f16 v4, v12, v13 // (op 1663) 
v_pack_b32_f16 v5, v14, v15 // (op 1663) 

s_lshl_b32 s0, s4, 6 // (op 1663) 
s_mul_i32 s1, 7, s0 // (op 1663) 

v_add_u32 v0, v9, s1 // (op 1663) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1663) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 138) Load value

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

v_mov_b32 v0, v9 // (op 137) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 1717) 
v_add_u32 v0, v6, s0 // (op 1717) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1717) Load value

v_accvgpr_read v12, a4 // (op 1713) 
v_accvgpr_read v13, a5 // (op 1713) 
v_accvgpr_read v14, a6 // (op 1713) 
v_accvgpr_read v15, a7 // (op 1713) 
v_mul_f32 v16, v11, v12 // (op 1713) 
v_mul_f32 v17, v11, v13 // (op 1713) 
v_mul_f32 v18, v11, v14 // (op 1713) 
v_mul_f32 v19, v11, v15 // (op 1713) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1718) 
v_lshrrev_b32 v13, 16, v4 // (op 1718) 
v_lshlrev_b32 v12, 16, v12 // (op 1718) 
v_lshlrev_b32 v13, 16, v13 // (op 1718) 
v_mul_f32 v20, v10, v12 // (op 1718) 
v_mul_f32 v21, v10, v13 // (op 1718) 
v_and_b32 v12, 65535, v5 // (op 1718) 
v_lshrrev_b32 v13, 16, v5 // (op 1718) 
v_lshlrev_b32 v12, 16, v12 // (op 1718) 
v_lshlrev_b32 v13, 16, v13 // (op 1718) 
v_mul_f32 v22, v10, v12 // (op 1718) 
v_mul_f32 v23, v10, v13 // (op 1718) 


v_add_f32 v12, v20, v16 // (op 1714) 
v_add_f32 v13, v21, v17 // (op 1714) 
v_add_f32 v14, v22, v18 // (op 1714) 
v_add_f32 v15, v23, v19 // (op 1714) 


v_lshrrev_b32 v16, 16, v12 // (op 1715) 
v_lshrrev_b32 v17, 16, v13 // (op 1715) 
v_lshrrev_b32 v18, 16, v14 // (op 1715) 
v_lshrrev_b32 v19, 16, v15 // (op 1715) 


v_pack_b32_f16 v4, v16, v17 // (op 1716) 
v_pack_b32_f16 v5, v18, v19 // (op 1716) 

s_lshl_b32 s0, s4, 6 // (op 1716) 
v_add_u32 v0, v9, s0 // (op 1716) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1716) Store value





s_lshl_b32 s0, s2, 7 // (op 1731) 
v_add_u32 v0, v6, s0 // (op 1731) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1731) Load value

v_accvgpr_read v12, a8 // (op 1727) 
v_accvgpr_read v13, a9 // (op 1727) 
v_accvgpr_read v14, a10 // (op 1727) 
v_accvgpr_read v15, a11 // (op 1727) 
v_mul_f32 v16, v11, v12 // (op 1727) 
v_mul_f32 v17, v11, v13 // (op 1727) 
v_mul_f32 v18, v11, v14 // (op 1727) 
v_mul_f32 v19, v11, v15 // (op 1727) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1732) 
v_lshrrev_b32 v13, 16, v4 // (op 1732) 
v_lshlrev_b32 v12, 16, v12 // (op 1732) 
v_lshlrev_b32 v13, 16, v13 // (op 1732) 
v_mul_f32 v20, v10, v12 // (op 1732) 
v_mul_f32 v21, v10, v13 // (op 1732) 
v_and_b32 v12, 65535, v5 // (op 1732) 
v_lshrrev_b32 v13, 16, v5 // (op 1732) 
v_lshlrev_b32 v12, 16, v12 // (op 1732) 
v_lshlrev_b32 v13, 16, v13 // (op 1732) 
v_mul_f32 v22, v10, v12 // (op 1732) 
v_mul_f32 v23, v10, v13 // (op 1732) 


v_add_f32 v12, v20, v16 // (op 1728) 
v_add_f32 v13, v21, v17 // (op 1728) 
v_add_f32 v14, v22, v18 // (op 1728) 
v_add_f32 v15, v23, v19 // (op 1728) 


v_lshrrev_b32 v16, 16, v12 // (op 1729) 
v_lshrrev_b32 v17, 16, v13 // (op 1729) 
v_lshrrev_b32 v18, 16, v14 // (op 1729) 
v_lshrrev_b32 v19, 16, v15 // (op 1729) 


v_pack_b32_f16 v4, v16, v17 // (op 1730) 
v_pack_b32_f16 v5, v18, v19 // (op 1730) 

s_lshl_b32 s0, s4, 7 // (op 1730) 
v_add_u32 v0, v9, s0 // (op 1730) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1730) Store value





s_lshl_b32 s0, s2, 6 // (op 1745) 
s_mul_i32 s1, 3, s0 // (op 1745) 

v_add_u32 v0, v6, s1 // (op 1745) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1745) Load value

v_accvgpr_read v12, a12 // (op 1741) 
v_accvgpr_read v13, a13 // (op 1741) 
v_accvgpr_read v14, a14 // (op 1741) 
v_accvgpr_read v15, a15 // (op 1741) 
v_mul_f32 v16, v11, v12 // (op 1741) 
v_mul_f32 v17, v11, v13 // (op 1741) 
v_mul_f32 v18, v11, v14 // (op 1741) 
v_mul_f32 v19, v11, v15 // (op 1741) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1746) 
v_lshrrev_b32 v13, 16, v4 // (op 1746) 
v_lshlrev_b32 v12, 16, v12 // (op 1746) 
v_lshlrev_b32 v13, 16, v13 // (op 1746) 
v_mul_f32 v20, v10, v12 // (op 1746) 
v_mul_f32 v21, v10, v13 // (op 1746) 
v_and_b32 v12, 65535, v5 // (op 1746) 
v_lshrrev_b32 v13, 16, v5 // (op 1746) 
v_lshlrev_b32 v12, 16, v12 // (op 1746) 
v_lshlrev_b32 v13, 16, v13 // (op 1746) 
v_mul_f32 v22, v10, v12 // (op 1746) 
v_mul_f32 v23, v10, v13 // (op 1746) 


v_add_f32 v12, v20, v16 // (op 1742) 
v_add_f32 v13, v21, v17 // (op 1742) 
v_add_f32 v14, v22, v18 // (op 1742) 
v_add_f32 v15, v23, v19 // (op 1742) 


v_lshrrev_b32 v16, 16, v12 // (op 1743) 
v_lshrrev_b32 v17, 16, v13 // (op 1743) 
v_lshrrev_b32 v18, 16, v14 // (op 1743) 
v_lshrrev_b32 v19, 16, v15 // (op 1743) 


v_pack_b32_f16 v4, v16, v17 // (op 1744) 
v_pack_b32_f16 v5, v18, v19 // (op 1744) 

s_lshl_b32 s0, s4, 6 // (op 1744) 
s_mul_i32 s1, 3, s0 // (op 1744) 

v_add_u32 v0, v9, s1 // (op 1744) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1744) Store value





s_lshl_b32 s0, s2, 8 // (op 1759) 
v_add_u32 v0, v6, s0 // (op 1759) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1759) Load value

v_accvgpr_read v12, a16 // (op 1755) 
v_accvgpr_read v13, a17 // (op 1755) 
v_accvgpr_read v14, a18 // (op 1755) 
v_accvgpr_read v15, a19 // (op 1755) 
v_mul_f32 v16, v11, v12 // (op 1755) 
v_mul_f32 v17, v11, v13 // (op 1755) 
v_mul_f32 v18, v11, v14 // (op 1755) 
v_mul_f32 v19, v11, v15 // (op 1755) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1760) 
v_lshrrev_b32 v13, 16, v4 // (op 1760) 
v_lshlrev_b32 v12, 16, v12 // (op 1760) 
v_lshlrev_b32 v13, 16, v13 // (op 1760) 
v_mul_f32 v20, v10, v12 // (op 1760) 
v_mul_f32 v21, v10, v13 // (op 1760) 
v_and_b32 v12, 65535, v5 // (op 1760) 
v_lshrrev_b32 v13, 16, v5 // (op 1760) 
v_lshlrev_b32 v12, 16, v12 // (op 1760) 
v_lshlrev_b32 v13, 16, v13 // (op 1760) 
v_mul_f32 v22, v10, v12 // (op 1760) 
v_mul_f32 v23, v10, v13 // (op 1760) 


v_add_f32 v12, v20, v16 // (op 1756) 
v_add_f32 v13, v21, v17 // (op 1756) 
v_add_f32 v14, v22, v18 // (op 1756) 
v_add_f32 v15, v23, v19 // (op 1756) 


v_lshrrev_b32 v16, 16, v12 // (op 1757) 
v_lshrrev_b32 v17, 16, v13 // (op 1757) 
v_lshrrev_b32 v18, 16, v14 // (op 1757) 
v_lshrrev_b32 v19, 16, v15 // (op 1757) 


v_pack_b32_f16 v4, v16, v17 // (op 1758) 
v_pack_b32_f16 v5, v18, v19 // (op 1758) 

s_lshl_b32 s0, s4, 8 // (op 1758) 
v_add_u32 v0, v9, s0 // (op 1758) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1758) Store value





s_lshl_b32 s0, s2, 6 // (op 1773) 
s_mul_i32 s1, 5, s0 // (op 1773) 

v_add_u32 v0, v6, s1 // (op 1773) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1773) Load value

v_accvgpr_read v12, a20 // (op 1769) 
v_accvgpr_read v13, a21 // (op 1769) 
v_accvgpr_read v14, a22 // (op 1769) 
v_accvgpr_read v15, a23 // (op 1769) 
v_mul_f32 v16, v11, v12 // (op 1769) 
v_mul_f32 v17, v11, v13 // (op 1769) 
v_mul_f32 v18, v11, v14 // (op 1769) 
v_mul_f32 v19, v11, v15 // (op 1769) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1774) 
v_lshrrev_b32 v13, 16, v4 // (op 1774) 
v_lshlrev_b32 v12, 16, v12 // (op 1774) 
v_lshlrev_b32 v13, 16, v13 // (op 1774) 
v_mul_f32 v20, v10, v12 // (op 1774) 
v_mul_f32 v21, v10, v13 // (op 1774) 
v_and_b32 v12, 65535, v5 // (op 1774) 
v_lshrrev_b32 v13, 16, v5 // (op 1774) 
v_lshlrev_b32 v12, 16, v12 // (op 1774) 
v_lshlrev_b32 v13, 16, v13 // (op 1774) 
v_mul_f32 v22, v10, v12 // (op 1774) 
v_mul_f32 v23, v10, v13 // (op 1774) 


v_add_f32 v12, v20, v16 // (op 1770) 
v_add_f32 v13, v21, v17 // (op 1770) 
v_add_f32 v14, v22, v18 // (op 1770) 
v_add_f32 v15, v23, v19 // (op 1770) 


v_lshrrev_b32 v16, 16, v12 // (op 1771) 
v_lshrrev_b32 v17, 16, v13 // (op 1771) 
v_lshrrev_b32 v18, 16, v14 // (op 1771) 
v_lshrrev_b32 v19, 16, v15 // (op 1771) 


v_pack_b32_f16 v4, v16, v17 // (op 1772) 
v_pack_b32_f16 v5, v18, v19 // (op 1772) 

s_lshl_b32 s0, s4, 6 // (op 1772) 
s_mul_i32 s1, 5, s0 // (op 1772) 

v_add_u32 v0, v9, s1 // (op 1772) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1772) Store value





s_lshl_b32 s0, s2, 6 // (op 1787) 
s_mul_i32 s1, 6, s0 // (op 1787) 

v_add_u32 v0, v6, s1 // (op 1787) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1787) Load value

v_accvgpr_read v12, a24 // (op 1783) 
v_accvgpr_read v13, a25 // (op 1783) 
v_accvgpr_read v14, a26 // (op 1783) 
v_accvgpr_read v15, a27 // (op 1783) 
v_mul_f32 v16, v11, v12 // (op 1783) 
v_mul_f32 v17, v11, v13 // (op 1783) 
v_mul_f32 v18, v11, v14 // (op 1783) 
v_mul_f32 v19, v11, v15 // (op 1783) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1788) 
v_lshrrev_b32 v13, 16, v4 // (op 1788) 
v_lshlrev_b32 v12, 16, v12 // (op 1788) 
v_lshlrev_b32 v13, 16, v13 // (op 1788) 
v_mul_f32 v20, v10, v12 // (op 1788) 
v_mul_f32 v21, v10, v13 // (op 1788) 
v_and_b32 v12, 65535, v5 // (op 1788) 
v_lshrrev_b32 v13, 16, v5 // (op 1788) 
v_lshlrev_b32 v12, 16, v12 // (op 1788) 
v_lshlrev_b32 v13, 16, v13 // (op 1788) 
v_mul_f32 v22, v10, v12 // (op 1788) 
v_mul_f32 v23, v10, v13 // (op 1788) 


v_add_f32 v12, v20, v16 // (op 1784) 
v_add_f32 v13, v21, v17 // (op 1784) 
v_add_f32 v14, v22, v18 // (op 1784) 
v_add_f32 v15, v23, v19 // (op 1784) 


v_lshrrev_b32 v16, 16, v12 // (op 1785) 
v_lshrrev_b32 v17, 16, v13 // (op 1785) 
v_lshrrev_b32 v18, 16, v14 // (op 1785) 
v_lshrrev_b32 v19, 16, v15 // (op 1785) 


v_pack_b32_f16 v4, v16, v17 // (op 1786) 
v_pack_b32_f16 v5, v18, v19 // (op 1786) 

s_lshl_b32 s0, s4, 6 // (op 1786) 
s_mul_i32 s1, 6, s0 // (op 1786) 

v_add_u32 v0, v9, s1 // (op 1786) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1786) Store value





s_lshl_b32 s0, s2, 6 // (op 1801) 
s_mul_i32 s1, 7, s0 // (op 1801) 

v_add_u32 v0, v6, s1 // (op 1801) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 1801) Load value

v_accvgpr_read v12, a28 // (op 1797) 
v_accvgpr_read v13, a29 // (op 1797) 
v_accvgpr_read v14, a30 // (op 1797) 
v_accvgpr_read v15, a31 // (op 1797) 
v_mul_f32 v16, v11, v12 // (op 1797) 
v_mul_f32 v17, v11, v13 // (op 1797) 
v_mul_f32 v18, v11, v14 // (op 1797) 
v_mul_f32 v19, v11, v15 // (op 1797) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v4 // (op 1802) 
v_lshrrev_b32 v13, 16, v4 // (op 1802) 
v_lshlrev_b32 v12, 16, v12 // (op 1802) 
v_lshlrev_b32 v13, 16, v13 // (op 1802) 
v_mul_f32 v20, v10, v12 // (op 1802) 
v_mul_f32 v21, v10, v13 // (op 1802) 
v_and_b32 v12, 65535, v5 // (op 1802) 
v_lshrrev_b32 v13, 16, v5 // (op 1802) 
v_lshlrev_b32 v12, 16, v12 // (op 1802) 
v_lshlrev_b32 v13, 16, v13 // (op 1802) 
v_mul_f32 v22, v10, v12 // (op 1802) 
v_mul_f32 v23, v10, v13 // (op 1802) 


v_add_f32 v12, v20, v16 // (op 1798) 
v_add_f32 v13, v21, v17 // (op 1798) 
v_add_f32 v14, v22, v18 // (op 1798) 
v_add_f32 v15, v23, v19 // (op 1798) 


v_lshrrev_b32 v16, 16, v12 // (op 1799) 
v_lshrrev_b32 v17, 16, v13 // (op 1799) 
v_lshrrev_b32 v18, 16, v14 // (op 1799) 
v_lshrrev_b32 v19, 16, v15 // (op 1799) 


v_pack_b32_f16 v4, v16, v17 // (op 1800) 
v_pack_b32_f16 v5, v18, v19 // (op 1800) 

s_lshl_b32 s0, s4, 6 // (op 1800) 
s_mul_i32 s1, 7, s0 // (op 1800) 

v_add_u32 v0, v9, s1 // (op 1800) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1800) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.LRR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2, .LRR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end-RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
  .amdhsa_next_free_vgpr 132
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 59392 // lds bytes
  .amdhsa_accum_offset 100
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B6
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
    macrotile: [32, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 59392
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 100
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

