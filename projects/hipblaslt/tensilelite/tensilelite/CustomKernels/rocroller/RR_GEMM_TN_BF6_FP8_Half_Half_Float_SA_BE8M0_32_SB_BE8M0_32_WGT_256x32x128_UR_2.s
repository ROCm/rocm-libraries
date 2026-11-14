// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2,@function
RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 3970) 
v_and_b32 v3, 1, v0 // (op 3970) 

v_and_b32 v0, 15, v1 // (op 3970) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 3970) 


v_and_b32 v0, 48, v1 // (op 3970) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3970) 


v_lshrrev_b32 v0, 6, v1 // (op 4114) 
v_and_b32 v4, 1, v0 // (op 4114) 

v_and_b32 v0, 15, v1 // (op 4114) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4114) 


v_and_b32 v0, 48, v1 // (op 4114) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4114) 


v_lshrrev_b32 v0, 7, v1 // (op 3856) 
v_and_b32 v5, 63, v1 // (op 3856) 
v_lshrrev_b32 v6, 4, v5 // (op 3856) 
v_lshlrev_b32 v7, 2, v6 // (op 3856) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3856) 


v_lshl_add_u32 v0, s16, 8, v6 // (op 3856) 

v_lshrrev_b32 v6, 2, v5 // (op 3856) 

v_and_b32 v5, 3, v6 // (op 3856) 

v_and_b32 v6, 3, v1 // (op 3856) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3856) 


v_lshrrev_b32 v5, 6, v1 // (op 3856) 
v_and_b32 v6, 1, v5 // (op 3856) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3856) 


v_lshl_add_u32 v6, s17, 5, v5 // (op 3856) 

v_mul_lo_u32 v5, v6, s2 // (op 3856) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 3856) 


v_lshrrev_b32 v0, 7, v1 // (op 3918) 
v_and_b32 v5, 15, v1 // (op 3918) 
v_lshl_add_u32 v7, v0, 4, v5 // (op 3918) 


v_and_b32 v0, 63, v1 // (op 3918) 
v_lshrrev_b32 v5, 4, v0 // (op 3918) 

v_lshlrev_b32 v0, 5, v5 // (op 3918) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 3918) 


v_mul_lo_u32 v0, v5, 6 // (op 3918) 

v_lshrrev_b32 v5, 3, v0 // (op 3918) 

v_lshrrev_b32 v0, 7, v1 // (op 4028) 
v_and_b32 v7, 15, v1 // (op 4028) 
s_mul_i32 s1, s2, s9 // (op 3858) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3858) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3858) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3858) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3858) most significant: sum
s_add_u32 s1, s1, s19 // (op 3858) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3858) 

v_lshl_add_u32 v8, v0, 4, v7 // (op 4028) 


v_and_b32 v0, 63, v1 // (op 4028) 
v_lshrrev_b32 v7, 4, v0 // (op 4028) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 4028) 


v_lshrrev_b32 v7, 7, v1 // (op 4044) 
v_and_b32 v8, 63, v1 // (op 4044) 
v_lshrrev_b32 v9, 4, v8 // (op 4044) 
v_lshlrev_b32 v10, 2, v9 // (op 4044) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 4044) 


v_lshl_add_u32 v7, s16, 8, v9 // (op 4044) 

v_lshrrev_b32 v9, 2, v8 // (op 4044) 

v_and_b32 v8, 3, v9 // (op 4044) 

v_and_b32 v9, 3, v1 // (op 4044) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 4044) 


v_lshrrev_b32 v8, 6, v1 // (op 4044) 
v_and_b32 v9, 1, v8 // (op 4044) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 4044) 


v_lshl_add_u32 v9, s17, 5, v8 // (op 4044) 

v_mul_lo_u32 v8, v9, s4 // (op 4044) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 4044) 


v_lshrrev_b32 v7, 7, v1 // (op 4075) 
v_and_b32 v8, 15, v1 // (op 4075) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 4075) 


v_and_b32 v7, 63, v1 // (op 4075) 
v_lshrrev_b32 v8, 4, v7 // (op 4075) 

v_lshlrev_b32 v7, 5, v8 // (op 4075) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 4075) 


v_mul_lo_u32 v7, v8, 6 // (op 4075) 

v_lshrrev_b32 v8, 3, v7 // (op 4075) 

v_lshrrev_b32 v7, 7, v1 // (op 4143) 
v_and_b32 v10, 15, v1 // (op 4143) 
s_mul_i32 s1, s4, s9 // (op 4046) most significant: low of low * high
v_lshl_add_u32 v11, v7, 4, v10 // (op 4143) 


v_and_b32 v7, 63, v1 // (op 4143) 
v_lshrrev_b32 v10, 4, v7 // (op 4143) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 4143) 


s_mul_i32 s38, s5, s8 // (op 4046) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 4046) least significant
s_mul_hi_u32 s39, s4, s8 // (op 4046) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 4046) most significant: sum
s_add_u32 s1, s1, s39 // (op 4046) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 4046) 

s_mov_b32 s40, s6 // (op 3858) reg expression
s_mov_b32 s41, s7 // (op 3858) reg expression
s_mov_b32 s42, s18 // (op 3858) reg expression
s_mov_b32 s43, 131072 // (op 3858) call()

s_mov_b32 s44, s10 // (op 4046) reg expression

s_mov_b32 s45, s11 // (op 4046) reg expression
s_mov_b32 s46, s38 // (op 4046) reg expression
s_mov_b32 s47, 131072 // (op 4046) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
v_accvgpr_write a8, 0.00000 // (op 475) call()
v_accvgpr_write a9, 0.00000 // (op 475) call()
v_accvgpr_write a10, 0.00000 // (op 475) call()
v_accvgpr_write a11, 0.00000 // (op 475) call()
v_accvgpr_write a12, 0.00000 // (op 637) call()
v_accvgpr_write a13, 0.00000 // (op 637) call()
v_accvgpr_write a14, 0.00000 // (op 637) call()
v_accvgpr_write a15, 0.00000 // (op 637) call()
v_accvgpr_write a16, 0.00000 // (op 799) call()
v_accvgpr_write a17, 0.00000 // (op 799) call()
v_accvgpr_write a18, 0.00000 // (op 799) call()
v_accvgpr_write a19, 0.00000 // (op 799) call()
v_accvgpr_write a20, 0.00000 // (op 961) call()
v_accvgpr_write a21, 0.00000 // (op 961) call()
v_accvgpr_write a22, 0.00000 // (op 961) call()
v_accvgpr_write a23, 0.00000 // (op 961) call()
v_accvgpr_write a24, 0.00000 // (op 1123) call()
v_accvgpr_write a25, 0.00000 // (op 1123) call()
v_accvgpr_write a26, 0.00000 // (op 1123) call()
v_accvgpr_write a27, 0.00000 // (op 1123) call()
v_accvgpr_write a28, 0.00000 // (op 1285) call()
v_accvgpr_write a29, 0.00000 // (op 1285) call()
v_accvgpr_write a30, 0.00000 // (op 1285) call()
v_accvgpr_write a31, 0.00000 // (op 1285) call()
v_and_b32 v12, 4294967294, v1 // (op 3623) 
v_lshl_add_u32 v13, s16, 8, v12 // (op 3623) 

v_and_b32 v12, -2, v1 // (op 3623) 
v_sub_u32 v14, v1, v12 // (op 3623) 

v_lshlrev_b32 v12, 6, v14 // (op 3623) 

v_mov_b32 v15, 0 // (op 3623) convert
v_mov_b32 v14, v12 // (op 3623) convert

v_mul_lo_u32 v17, v13, s13 // (op 3623) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3623) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3623) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3623) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3623) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3623) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 3623) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 3623) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 3623) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 3623) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 3623) 

v_mov_b32 v12, v14 // (op 3623) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 3625) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3625) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3625) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3625) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3625) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3625) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3625) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3625) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3625) most significant: sum
s_add_u32 s7, s7, s11 // (op 3625) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3625) 

s_mov_b32 s48, s20 // (op 3625) reg expression
s_mov_b32 s49, s21 // (op 3625) reg expression
s_mov_b32 s50, s0 // (op 3625) reg expression
s_mov_b32 s51, 131072 // (op 3625) call()

v_and_b32 v13, 4294967294, v1 // (op 3642) 
v_lshl_add_u32 v14, s16, 8, v13 // (op 3642) 

v_and_b32 v13, -2, v1 // (op 3642) 
v_sub_u32 v15, v1, v13 // (op 3642) 

v_lshlrev_b32 v13, 6, v15 // (op 3642) 

v_mov_b32 v17, 0 // (op 3642) convert
v_mov_b32 v16, v13 // (op 3642) convert

v_mul_lo_u32 v19, v14, s13 // (op 3642) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 3642) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 3642) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3642) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 3642) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 3642) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 3642) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 3642) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 3642) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3642) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3642) 

v_mov_b32 v13, v14 // (op 3642) convert

s_mul_i32 s1, s23, 6 // (op 3644) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3644) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3644) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3644) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3644) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3644) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3644) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3644) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3644) most significant: sum
s_add_u32 s7, s7, s11 // (op 3644) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3644) 

s_mov_b32 s48, s20 // (op 3644) reg expression
s_mov_b32 s49, s21 // (op 3644) reg expression
s_mov_b32 s50, s0 // (op 3644) reg expression
s_mov_b32 s51, 131072 // (op 3644) call()

v_and_b32 v14, -8, v1 // (op 3687) 
v_sub_u32 v15, v1, v14 // (op 3687) 

v_lshlrev_b32 v14, 4, v15 // (op 3687) 

v_mov_b32 v17, 0 // (op 3687) convert
v_mov_b32 v16, v14 // (op 3687) convert

v_lshrrev_b32 v14, 3, v1 // (op 3687) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 3687) 

v_mul_lo_u32 v19, v15, s25 // (op 3687) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 3687) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 3687) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 3687) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3687) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3687) most significant half



s_mul_i32 s1, s24, s9 // (op 3689) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3689) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3689) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3689) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3689) most significant: sum
s_add_u32 s1, s1, s7 // (op 3689) most significant: sum


s_mov_b32 s52, s26 // (op 3689) reg expression
s_mov_b32 s53, s27 // (op 3689) reg expression
s_mov_b32 s54, s0 // (op 3689) reg expression
s_mov_b32 s55, 131072 // (op 3689) call()

v_and_b32 v16, -8, v1 // (op 3706) 
v_sub_u32 v17, v1, v16 // (op 3706) 

v_lshlrev_b32 v16, 4, v17 // (op 3706) 

v_mov_b32 v19, 0 // (op 3706) convert
v_mov_b32 v18, v16 // (op 3706) convert

v_lshrrev_b32 v16, 3, v1 // (op 3706) 
v_lshl_add_u32 v17, s17, 5, v16 // (op 3706) 

v_mul_lo_u32 v21, v17, s25 // (op 3706) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 3706) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 3706) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 3706) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 3706) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 3706) most significant half



s_mul_i32 s1, s24, s9 // (op 3708) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3708) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3708) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3708) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3708) most significant: sum
s_add_u32 s1, s1, s7 // (op 3708) most significant: sum


s_mov_b32 s52, s26 // (op 3708) reg expression
s_mov_b32 s53, s27 // (op 3708) reg expression
s_mov_b32 s54, s0 // (op 3708) reg expression
s_mov_b32 s55, 131072 // (op 3708) call()

v_lshl_add_u32 v18, s16, 8, v1 // (op 3751) 
v_mul_lo_u32 v19, v18, s28 // (op 3751) 

s_mul_i32 s1, s28, s23 // (op 3753) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3753) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3753) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3753) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3753) most significant: sum
s_add_u32 s1, s1, s7 // (op 3753) most significant: sum


s_mov_b32 s56, s30 // (op 3753) reg expression
s_mov_b32 s57, s31 // (op 3753) reg expression
s_mov_b32 s58, s0 // (op 3753) reg expression
s_mov_b32 s59, 131072 // (op 3753) call()

v_lshl_add_u32 v18, s16, 8, v1 // (op 3770) 
v_mul_lo_u32 v20, v18, s28 // (op 3770) 

s_mul_i32 s1, s28, s23 // (op 3772) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3772) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3772) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3772) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3772) most significant: sum
s_add_u32 s1, s1, s7 // (op 3772) most significant: sum


s_mov_b32 s56, s30 // (op 3772) reg expression
s_mov_b32 s57, s31 // (op 3772) reg expression
s_mov_b32 s58, s0 // (op 3772) reg expression
s_mov_b32 s59, 131072 // (op 3772) call()

v_lshrrev_b32 v18, 6, v1 // (op 3815) 
v_and_b32 v21, 1, v18 // (op 3815) 

v_and_b32 v18, 15, v1 // (op 3815) 
v_lshl_add_u32 v22, v21, 4, v18 // (op 3815) 


v_lshl_add_u32 v18, s17, 5, v22 // (op 3815) 

v_mul_lo_u32 v21, v18, s32 // (op 3815) 

v_and_b32 v18, 63, v1 // (op 3815) 
v_lshrrev_b32 v22, 4, v18 // (op 3815) 

v_add_u32 v18, v22, v21 // (op 3815) 


s_mul_i32 s1, s32, s9 // (op 3817) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3817) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3817) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3817) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3817) most significant: sum
s_add_u32 s1, s1, s7 // (op 3817) most significant: sum


s_mov_b32 s60, s34 // (op 3817) reg expression
s_mov_b32 s61, s35 // (op 3817) reg expression
s_mov_b32 s62, s0 // (op 3817) reg expression
s_mov_b32 s63, 131072 // (op 3817) call()

v_lshrrev_b32 v21, 6, v1 // (op 3834) 
v_and_b32 v22, 1, v21 // (op 3834) 

v_and_b32 v21, 15, v1 // (op 3834) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 3834) 


v_lshl_add_u32 v21, s17, 5, v23 // (op 3834) 

v_mul_lo_u32 v22, v21, s32 // (op 3834) 

v_and_b32 v21, 63, v1 // (op 3834) 
v_lshrrev_b32 v23, 4, v21 // (op 3834) 

v_add_u32 v21, v23, v22 // (op 3834) 


s_mul_i32 s1, s32, s9 // (op 3836) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3836) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3836) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3836) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3836) most significant: sum
s_add_u32 s1, s1, s7 // (op 3836) most significant: sum


s_mov_b32 s60, s34 // (op 3836) reg expression
s_mov_b32 s61, s35 // (op 3836) reg expression
s_mov_b32 s62, s0 // (op 3836) reg expression
s_mov_b32 s63, 131072 // (op 3836) call()




buffer_load_ubyte v22, v18, s[60:63], 0 offen offset: 0 // (op 3284) Load value
v_and_b32 v23, -8, v1 // (op 3958) 
v_sub_u32 v24, v1, v23 // (op 3958) 

v_lshlrev_b32 v23, 4, v24 // (op 3958) 

v_lshrrev_b32 v24, 3, v1 // (op 3958) 
v_lshl_add_u32 v25, v24, 7, v23 // (op 3958) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 3958) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 3572) reg expression
buffer_load_dwordx4 v14, s[52:55], 0 offen offset: 0 lds // (op 3572) Load value direct to lds


s_mul_i32 s1, s13, 6 // (op 3291) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 3291) least significant
s_mul_hi_u32 s6, s12, 6 // (op 3291) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3291) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3291) 


v_mov_b32 v23, v12 // (op 3291) 
buffer_load_dwordx4 v[26:29], v23, s[48:51], 0 offen // (op 3291) Load value
buffer_load_dwordx2 v[30:31], v23, s[48:51], 0 offen offset:16 // (op 3291) Load value
buffer_load_dwordx4 v[32:35], v23, s[48:51], 0 offen offset:24 // (op 3291) Load value
buffer_load_dwordx2 v[36:37], v23, s[48:51], 0 offen offset:40 // (op 3291) Load value
v_add_u32 v23, v23, v24 // (op 3291) 
buffer_load_dwordx4 v[38:41], v23, s[48:51], 0 offen // (op 3291) Load value
buffer_load_dwordx2 v[42:43], v23, s[48:51], 0 offen offset:16 // (op 3291) Load value
buffer_load_dwordx4 v[44:47], v23, s[48:51], 0 offen offset:24 // (op 3291) Load value
buffer_load_dwordx2 v[48:49], v23, s[48:51], 0 offen offset:40 // (op 3291) Load value





buffer_load_dword v23, v19, s[56:59], 0 offen // (op 3294) Load value



buffer_load_ubyte v24, v18, s[60:63], 0 offen offset:4 // (op 3298) Load value

v_and_b32 v18, -8, v1 // (op 4102) 
v_sub_u32 v25, v1, v18 // (op 4102) 

v_lshlrev_b32 v18, 4, v25 // (op 4102) 

v_lshrrev_b32 v25, 3, v1 // (op 4102) 
v_lshl_add_u32 v50, v25, 7, v18 // (op 4102) 


s_nop 0
v_readfirstlane_b32 s0, v50 // (op 4102) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 3577) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3577) reg expression
s_mov_b32 s6, 128 // (op 3577) call()
buffer_load_dwordx4 v14, s[52:55], s6 offen offset: 0 lds // (op 3577) Load value direct to lds





s_mul_i32 s1, s13, 6 // (op 3305) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 3305) least significant
s_mul_hi_u32 s6, s12, 6 // (op 3305) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3305) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[0:1] // (op 3305) 


v_mov_b32 v18, v12 // (op 3305) 
buffer_load_dwordx4 v[50:53], v18, s[48:51], 0 offen offset:96 // (op 3305) Load value
buffer_load_dwordx2 v[54:55], v18, s[48:51], 0 offen offset:112 // (op 3305) Load value
buffer_load_dwordx4 v[56:59], v18, s[48:51], 0 offen offset:120 // (op 3305) Load value
buffer_load_dwordx2 v[60:61], v18, s[48:51], 0 offen offset:136 // (op 3305) Load value
v_add_u32 v18, v18, v14 // (op 3305) 
buffer_load_dwordx4 v[62:65], v18, s[48:51], 0 offen offset:96 // (op 3305) Load value
buffer_load_dwordx2 v[66:67], v18, s[48:51], 0 offen offset:112 // (op 3305) Load value
buffer_load_dwordx4 v[68:71], v18, s[48:51], 0 offen offset:120 // (op 3305) Load value
buffer_load_dwordx2 v[72:73], v18, s[48:51], 0 offen offset:136 // (op 3305) Load value






buffer_load_dword v12, v19, s[56:59], 0 offen offset:4 // (op 3308) Load value

v_and_b32 v25, -2, v1 // (op 3901) 
v_sub_u32 v14, v1, v25 // (op 3901) 

v_lshlrev_b32 v15, 6, v14 // (op 3901) 

v_lshrrev_b32 v14, 1, v1 // (op 3901) 
v_lshl_add_u32 v25, v14, 8, v15 // (op 3901) 


v_mul_lo_u32 v14, v25, 6 // (op 3901) 

v_lshrrev_b32 v15, 3, v14 // (op 3901) 




s_waitcnt vmcnt(19)
ds_write_b128 v15, v[26:29] offset:8192 // (op 3311) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v15, v[30:31] offset:8208 // (op 3311) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v15, v[32:35] offset:8216 // (op 3311) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v15, v[36:37] offset:8232 // (op 3311) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v15, v[38:41] offset:8288 // (op 3311) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v15, v[42:43] offset:8304 // (op 3311) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v15, v[44:47] offset:8312 // (op 3311) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v15, v[48:49] offset:8328 // (op 3311) Store local data 

v_lshlrev_b32 v25, 2, v1 // (op 4011) 



s_waitcnt vmcnt(11)
ds_write_b32 v25, v23 offset:32768 // (op 3312) Store local data 

v_and_b32 v25, -2, v1 // (op 3872) 
v_sub_u32 v14, v1, v25 // (op 3872) 

v_lshlrev_b32 v15, 6, v14 // (op 3872) 

v_lshrrev_b32 v14, 1, v1 // (op 3872) 
v_lshl_add_u32 v25, v14, 8, v15 // (op 3872) 


v_mul_lo_u32 v14, v25, 6 // (op 3872) 

v_lshrrev_b32 v15, 3, v14 // (op 3872) 

v_and_b32 v14, -8, v1 // (op 3930) 
v_sub_u32 v25, v1, v14 // (op 3930) 

v_lshlrev_b32 v14, 4, v25 // (op 3930) 

v_lshrrev_b32 v25, 3, v1 // (op 3930) 
v_lshl_add_u32 v18, v25, 7, v14 // (op 3930) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 3930) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 2, v1 // (op 3982) 
v_and_b32 v25, -2, v1 // (op 4058) 
v_sub_u32 v18, v1, v25 // (op 4058) 

v_lshlrev_b32 v19, 6, v18 // (op 4058) 

v_lshrrev_b32 v18, 1, v1 // (op 4058) 
v_lshl_add_u32 v25, v18, 8, v19 // (op 4058) 


v_mul_lo_u32 v18, v25, 6 // (op 4058) 

v_lshrrev_b32 v19, 3, v18 // (op 4058) 

v_and_b32 v18, -8, v1 // (op 4087) 
v_sub_u32 v25, v1, v18 // (op 4087) 

v_lshlrev_b32 v18, 4, v25 // (op 4087) 

v_lshrrev_b32 v25, 3, v1 // (op 4087) 
v_lshl_add_u32 v74, v25, 7, v18 // (op 4087) 


s_nop 0
v_readfirstlane_b32 s1, v74 // (op 4087) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v18, 2, v1 // (op 4126) 
s_mov_b32 s6, 0 // (op 1287) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 1286) 
s_mov_b32 s7, s10 // (op 1286) convert

s_and_b32 s10, s7, 4294967294 // (op 1286) 

s_cmp_lt_u32 s6, s10 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1286 // (op 1286) Condition: Top (jump to RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1286 if false)
RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1286:
s_add_u32 s6, s6, 2 // (op 1288) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4202) 



ds_read_b128 v[74:77], v5 offset:8192 // (op 67) Load local data 
ds_read_b64 v[78:79], v5 offset:8208 // (op 67) Load local data 
s_barrier  // (op 4215) 



ds_read_b128 v[80:83], v3 // (op 72) Load local data 
ds_read_b128 v[84:87], v3 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4228) 



ds_read_u8 v25, v0 offset:32768 // (op 77) Load local data 



ds_read_b128 v[88:91], v5 offset:11264 // (op 344) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:79], v[80:87], a[0:3], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 84) 
ds_read_b64 v[92:93], v5 offset:11280 // (op 344) Load local data 





ds_read_u8 v25, v0 offset:32896 // (op 348) Load local data 



ds_read_b128 v[74:77], v5 offset:14336 // (op 506) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:93], v[80:87], a[4:7], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 350) 
ds_read_b64 v[78:79], v5 offset:14352 // (op 506) Load local data 





ds_read_u8 v25, v0 offset:33024 // (op 510) Load local data 



ds_read_b128 v[88:91], v5 offset:17408 // (op 668) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[74:79], v[80:87], a[8:11], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 512) 
ds_read_b64 v[92:93], v5 offset:17424 // (op 668) Load local data 





ds_read_u8 v25, v0 offset:33152 // (op 672) Load local data 



ds_read_b128 v[74:77], v5 offset:20480 // (op 830) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:93], v[80:87], a[12:15], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 674) 
ds_read_b64 v[78:79], v5 offset:20496 // (op 830) Load local data 





ds_read_u8 v25, v0 offset:33280 // (op 834) Load local data 



ds_read_b128 v[88:91], v5 offset:23552 // (op 992) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:79], v[80:87], a[16:19], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 836) 
ds_read_b64 v[92:93], v5 offset:23568 // (op 992) Load local data 





ds_read_u8 v25, v0 offset:33408 // (op 996) Load local data 



ds_read_b128 v[74:77], v5 offset:26624 // (op 1154) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:93], v[80:87], a[20:23], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 998) 
ds_read_b64 v[78:79], v5 offset:26640 // (op 1154) Load local data 





ds_read_u8 v25, v0 offset:33536 // (op 1158) Load local data 



ds_read_b128 v[88:91], v5 offset:29696 // (op 1316) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:79], v[80:87], a[24:27], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1160) 
ds_read_b64 v[92:93], v5 offset:29712 // (op 1316) Load local data 





ds_read_u8 v25, v0 offset:33664 // (op 1320) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:93], v[80:87], a[28:31], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1322) 





v_mov_b32 v25, v21 // (op 82) 

buffer_load_ubyte v22, v25, s[60:63], 0 offen offset:8 // (op 82) Load value
s_barrier  // (op 4218) 


v_mov_b64 v[74:75], v[16:17] // (op 3555) 

s_mov_b32 m0, s0 // (op 3555) reg expression
s_mov_b32 s7, 256 // (op 3555) call()
buffer_load_dwordx4 v74, s[52:55], s7 offen offset: 0 lds // (op 3555) Load value direct to lds


s_mul_i32 s11, s13, 6 // (op 4) most significant: low of high * low
s_mul_i32 s10, s12, 6 // (op 4) least significant
s_mul_hi_u32 s7, s12, 6 // (op 4) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 4) most significant: sum

v_lshrrev_b64 v[76:77], 3, s[10:11] // (op 4) 


v_mov_b32 v78, v13 // (op 4) 
v_mov_b32 v79, v78 // (op 4) 
buffer_load_dwordx4 v[26:29], v79, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[30:31], v79, s[48:51], 0 offen offset:208 // (op 4) Load value
buffer_load_dwordx4 v[32:35], v79, s[48:51], 0 offen offset:216 // (op 4) Load value
buffer_load_dwordx2 v[36:37], v79, s[48:51], 0 offen offset:232 // (op 4) Load value
v_add_u32 v79, v79, v76 // (op 4) 
buffer_load_dwordx4 v[38:41], v79, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[42:43], v79, s[48:51], 0 offen offset:208 // (op 4) Load value
buffer_load_dwordx4 v[44:47], v79, s[48:51], 0 offen offset:216 // (op 4) Load value
buffer_load_dwordx2 v[48:49], v79, s[48:51], 0 offen offset:232 // (op 4) Load value




v_mov_b32 v76, v20 // (op 17) 

buffer_load_dword v23, v76, s[56:59], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 4245) 



s_waitcnt vmcnt(19)
ds_write_b128 v19, v[50:53] offset:33792 // (op 229) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v19, v[54:55] offset:33808 // (op 229) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v19, v[56:59] offset:33816 // (op 229) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v19, v[60:61] offset:33832 // (op 229) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v19, v[62:65] offset:33888 // (op 229) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v19, v[66:67] offset:33904 // (op 229) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v19, v[68:71] offset:33912 // (op 229) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v19, v[72:73] offset:33928 // (op 229) Store local data 
s_barrier  // (op 4257) 



s_waitcnt vmcnt(11)
ds_write_b32 v18, v12 offset:58368 // (op 240) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4242) 



ds_read_b128 v[80:83], v8 offset:33792 // (op 231) Load local data 
ds_read_b64 v[84:85], v8 offset:33808 // (op 231) Load local data 
s_barrier  // (op 4248) 



ds_read_b128 v[86:89], v4 offset:4096 // (op 233) Load local data 
ds_read_b128 v[90:93], v4 offset:4160 // (op 233) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4254) 



ds_read_u8 v77, v7 offset:58368 // (op 235) Load local data 



ds_read_b128 v[94:97], v8 offset:36864 // (op 357) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:85], v[86:93], a[0:3], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 238) 
ds_read_b64 v[98:99], v8 offset:36880 // (op 357) Load local data 





ds_read_u8 v77, v7 offset:58496 // (op 361) Load local data 



ds_read_b128 v[80:83], v8 offset:39936 // (op 519) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[94:99], v[86:93], a[4:7], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 351) 
ds_read_b64 v[84:85], v8 offset:39952 // (op 519) Load local data 





ds_read_u8 v77, v7 offset:58624 // (op 523) Load local data 



ds_read_b128 v[94:97], v8 offset:43008 // (op 681) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[80:85], v[86:93], a[8:11], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 513) 
ds_read_b64 v[98:99], v8 offset:43024 // (op 681) Load local data 





ds_read_u8 v77, v7 offset:58752 // (op 685) Load local data 



ds_read_b128 v[80:83], v8 offset:46080 // (op 843) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[94:99], v[86:93], a[12:15], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 675) 
ds_read_b64 v[84:85], v8 offset:46096 // (op 843) Load local data 





ds_read_u8 v77, v7 offset:58880 // (op 847) Load local data 



ds_read_b128 v[94:97], v8 offset:49152 // (op 1005) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:85], v[86:93], a[16:19], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 837) 
ds_read_b64 v[98:99], v8 offset:49168 // (op 1005) Load local data 





ds_read_u8 v77, v7 offset:59008 // (op 1009) Load local data 



ds_read_b128 v[80:83], v8 offset:52224 // (op 1167) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[94:99], v[86:93], a[20:23], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 999) 
ds_read_b64 v[84:85], v8 offset:52240 // (op 1167) Load local data 





ds_read_u8 v77, v7 offset:59136 // (op 1171) Load local data 



ds_read_b128 v[94:97], v8 offset:55296 // (op 1329) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:85], v[86:93], a[24:27], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1161) 
ds_read_b64 v[98:99], v8 offset:55312 // (op 1329) Load local data 





s_barrier  // (op 4358) 
ds_read_u8 v77, v7 offset:59264 // (op 1333) Load local data 
s_barrier  // (op 4362) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[94:99], v[86:93], a[28:31], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1323) 






buffer_load_ubyte v24, v25, s[60:63], 0 offen offset:12 // (op 237) Load value

s_barrier  // (op 4251) 
v_add_u32 v21, 8, v21 // (op 3826) 


s_add_u32 s7, 4096, s1 // (op 3567) 

s_mov_b32 m0, s7 // (op 3567) reg expression
s_mov_b32 s10, 384 // (op 3567) call()
buffer_load_dwordx4 v74, s[52:55], s10 offen offset: 0 lds // (op 3567) Load value direct to lds




v_add_co_u32 v16, vcc, 256, v16 // (op 3698) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 3698) most significant half
s_mul_i32 s11, s13, 6 // (op 228) most significant: low of high * low
s_mul_i32 s10, s12, 6 // (op 228) least significant
s_mul_hi_u32 s7, s12, 6 // (op 228) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 228) most significant: sum

v_lshrrev_b64 v[74:75], 3, s[10:11] // (op 228) 


v_mov_b32 v25, v78 // (op 228) 
buffer_load_dwordx4 v[50:53], v25, s[48:51], 0 offen offset:288 // (op 228) Load value
buffer_load_dwordx2 v[54:55], v25, s[48:51], 0 offen offset:304 // (op 228) Load value
buffer_load_dwordx4 v[56:59], v25, s[48:51], 0 offen offset:312 // (op 228) Load value
buffer_load_dwordx2 v[60:61], v25, s[48:51], 0 offen offset:328 // (op 228) Load value
v_add_u32 v25, v25, v74 // (op 228) 
buffer_load_dwordx4 v[62:65], v25, s[48:51], 0 offen offset:288 // (op 228) Load value
buffer_load_dwordx2 v[66:67], v25, s[48:51], 0 offen offset:304 // (op 228) Load value
buffer_load_dwordx4 v[68:71], v25, s[48:51], 0 offen offset:312 // (op 228) Load value
buffer_load_dwordx2 v[72:73], v25, s[48:51], 0 offen offset:328 // (op 228) Load value






v_add_u32 v13, 192, v13 // (op 3634) 
buffer_load_dword v12, v76, s[56:59], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 4205) 
v_add_u32 v20, 8, v20 // (op 3762) 



s_waitcnt vmcnt(19)
ds_write_b128 v15, v[26:29] offset:8192 // (op 6) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v15, v[30:31] offset:8208 // (op 6) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v15, v[32:35] offset:8216 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v15, v[36:37] offset:8232 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v15, v[38:41] offset:8288 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v15, v[42:43] offset:8304 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v15, v[44:47] offset:8312 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v15, v[48:49] offset:8328 // (op 6) Store local data 
s_barrier  // (op 4231) 



s_waitcnt vmcnt(11)
ds_write_b32 v14, v23 offset:32768 // (op 21) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 1286) 
s_mov_b32 s7, s10 // (op 1286) convert

s_and_b32 s10, s7, 4294967294 // (op 1286) 

s_cmp_lt_u32 s6, s10 // (op 1286) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1286 // (op 1286) Condition: Bottom (jump to RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoop_1286 if true)
RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoop_1286:














s_barrier  // (op 4342) 
v_and_b32 v13, 4294967294, v1 // (op 3590) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3590) 
v_lshl_add_u32 v14, s16, 8, v13 // (op 3590) 

s_lshr_b32 s6, s0, 1 // (op 3590) 

v_and_b32 v13, -2, v1 // (op 3590) 
v_sub_u32 v15, v1, v13 // (op 3590) 

v_lshlrev_b32 v13, 6, v15 // (op 3590) 

v_lshl_add_u32 v15, s6, 8, v13 // (op 3590) 


v_mov_b32 v17, 0 // (op 3590) convert
v_mov_b32 v16, v15 // (op 3590) convert

v_mul_lo_u32 v19, v14, s13 // (op 3590) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 3590) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 3590) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3590) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 3590) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 3590) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 3590) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 3590) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 3590) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3590) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 3590) 

v_mov_b32 v13, v14 // (op 3590) convert

s_mul_i32 s1, s23, 6 // (op 3592) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3592) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3592) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3592) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3592) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3592) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3592) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3592) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3592) most significant: sum
s_add_u32 s7, s7, s11 // (op 3592) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3592) 

s_mov_b32 s48, s20 // (op 3592) reg expression
s_mov_b32 s49, s21 // (op 3592) reg expression
s_mov_b32 s50, s0 // (op 3592) reg expression
s_mov_b32 s51, 131072 // (op 3592) call()

v_and_b32 v14, -8, v1 // (op 3664) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3664) 
v_sub_u32 v15, v1, v14 // (op 3664) 

s_lshr_b32 s6, s0, 1 // (op 3664) 

v_lshlrev_b32 v14, 4, v15 // (op 3664) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 3664) 


v_mov_b32 v17, 0 // (op 3664) convert
v_mov_b32 v16, v15 // (op 3664) convert

v_lshrrev_b32 v14, 3, v1 // (op 3664) 
v_lshl_add_u32 v15, s17, 5, v14 // (op 3664) 

v_mul_lo_u32 v19, v15, s25 // (op 3664) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 3664) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 3664) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 3664) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 3664) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 3664) most significant half



s_mul_i32 s1, s24, s9 // (op 3666) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3666) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3666) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3666) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3666) most significant: sum
s_add_u32 s1, s1, s7 // (op 3666) most significant: sum


s_mov_b32 s52, s26 // (op 3666) reg expression
s_mov_b32 s53, s27 // (op 3666) reg expression
s_mov_b32 s54, s0 // (op 3666) reg expression
s_mov_b32 s55, 131072 // (op 3666) call()

v_lshl_add_u32 v16, s16, 8, v1 // (op 3728) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3728) 
s_lshr_b32 s6, s0, 1 // (op 3728) 

s_lshl_b32 s7, s6, 3 // (op 3728) 

v_mul_lo_u32 v17, v16, s28 // (op 3728) 
v_add_u32 v17, v17, s7 // (op 3728) 


s_mul_i32 s1, s28, s23 // (op 3730) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3730) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3730) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3730) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3730) most significant: sum
s_add_u32 s1, s1, s7 // (op 3730) most significant: sum


s_mov_b32 s56, s30 // (op 3730) reg expression
s_mov_b32 s57, s31 // (op 3730) reg expression
s_mov_b32 s58, s0 // (op 3730) reg expression
s_mov_b32 s59, 131072 // (op 3730) call()

v_and_b32 v16, 63, v1 // (op 3792) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3792) 
v_lshrrev_b32 v18, 4, v16 // (op 3792) 

s_lshr_b32 s6, s0, 1 // (op 3792) 

v_lshrrev_b32 v16, 6, v1 // (op 3792) 
v_lshl_add_u32 v19, s6, 3, v18 // (op 3792) 


v_and_b32 v18, 1, v16 // (op 3792) 

v_and_b32 v16, 15, v1 // (op 3792) 
v_lshl_add_u32 v20, v18, 4, v16 // (op 3792) 


v_lshl_add_u32 v16, s17, 5, v20 // (op 3792) 

v_mul_lo_u32 v18, v16, s32 // (op 3792) 

v_add_u32 v16, v19, v18 // (op 3792) 


s_mul_i32 s1, s32, s9 // (op 3794) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3794) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3794) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3794) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3794) most significant: sum
s_add_u32 s1, s1, s7 // (op 3794) most significant: sum


s_mov_b32 s60, s34 // (op 3794) reg expression
s_mov_b32 s61, s35 // (op 3794) reg expression
s_mov_b32 s62, s0 // (op 3794) reg expression
s_mov_b32 s63, 131072 // (op 3794) call()

v_and_b32 v18, -2, v1 // (op 3885) 
v_sub_u32 v19, v1, v18 // (op 3885) 

v_lshlrev_b32 v18, 6, v19 // (op 3885) 

v_lshrrev_b32 v19, 1, v1 // (op 3885) 
v_lshl_add_u32 v20, v19, 8, v18 // (op 3885) 


v_mul_lo_u32 v18, v20, 6 // (op 3885) 

v_lshrrev_b32 v19, 3, v18 // (op 3885) 

v_and_b32 v18, -8, v1 // (op 3943) 
v_sub_u32 v20, v1, v18 // (op 3943) 

v_lshlrev_b32 v18, 4, v20 // (op 3943) 

v_lshrrev_b32 v20, 3, v1 // (op 3943) 
v_lshl_add_u32 v21, v20, 7, v18 // (op 3943) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 3943) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v18, 2, v1 // (op 3995) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1292) 
s_and_b32 s1, s6, 4294967294 // (op 1292) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1291) 
s_mov_b32 s10, s6 // (op 1291) convert

s_cmp_lt_u32 s1, s10 // (op 1291) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1291 // (op 1291) Condition: Top (jump to RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1291 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1291:

s_add_u32 s1, s1, 1 // (op 1293) 
s_mul_i32 s7, s13, 6 // (op 170) most significant: low of high * low
s_mul_i32 s6, s12, 6 // (op 170) least significant
s_mul_hi_u32 s10, s12, 6 // (op 170) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 170) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[6:7] // (op 170) 


v_mov_b32 v12, v13 // (op 170) 
v_mov_b32 v22, v12 // (op 170) 
buffer_load_dwordx4 v[24:27], v22, s[48:51], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[28:29], v22, s[48:51], 0 offen offset:16 // (op 170) Load value
buffer_load_dwordx4 v[30:33], v22, s[48:51], 0 offen offset:24 // (op 170) Load value
buffer_load_dwordx2 v[34:35], v22, s[48:51], 0 offen offset:40 // (op 170) Load value
v_add_u32 v22, v22, v20 // (op 170) 
buffer_load_dwordx4 v[36:39], v22, s[48:51], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[40:41], v22, s[48:51], 0 offen offset:16 // (op 170) Load value
buffer_load_dwordx4 v[42:45], v22, s[48:51], 0 offen offset:24 // (op 170) Load value
buffer_load_dwordx2 v[46:47], v22, s[48:51], 0 offen offset:40 // (op 170) Load value



v_add_u32 v13, 96, v13 // (op 3580) 
s_barrier  // (op 4208) 
s_barrier  // (op 4221) 





v_mov_b64 v[20:21], v[14:15] // (op 3560) 

s_mov_b32 m0, s0 // (op 3560) reg expression
buffer_load_dwordx4 v20, s[52:55], 0 offen offset: 0 lds // (op 3560) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 3656) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3656) most significant half
s_waitcnt vmcnt(8)
ds_write_b128 v19, v[24:27] offset:8192 // (op 171) Store local data 


v_mov_b32 v12, v17 // (op 184) 

s_waitcnt vmcnt(7)
ds_write_b64 v19, v[28:29] offset:8208 // (op 171) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v19, v[30:33] offset:8216 // (op 171) Store local data 
s_waitcnt vmcnt(5)
ds_write_b64 v19, v[34:35] offset:8232 // (op 171) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v19, v[36:39] offset:8288 // (op 171) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v19, v[40:41] offset:8304 // (op 171) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[42:45] offset:8312 // (op 171) Store local data 
buffer_load_dword v20, v12, s[56:59], 0 offen // (op 184) Load value

v_add_u32 v17, 4, v17 // (op 3720) 
s_waitcnt vmcnt(2)
ds_write_b64 v19, v[46:47] offset:8328 // (op 171) Store local data 

s_barrier  // (op 4234) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4211) 






ds_read_b128 v[22:25], v5 offset:8192 // (op 173) Load local data 
ds_read_b64 v[26:27], v5 offset:8208 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v18, v20 offset:32768 // (op 182) Store local data 

s_barrier  // (op 4224) 



ds_read_b128 v[28:31], v3 // (op 175) Load local data 
ds_read_b128 v[32:35], v3 offset:64 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4238) 



ds_read_u8 v12, v0 offset:32768 // (op 177) Load local data 


v_mov_b32 v20, v16 // (op 179) 

buffer_load_ubyte v21, v20, s[60:63], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v16, 4, v16 // (op 3784) 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[28:35], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 180) 
ds_read_b128 v[36:39], v5 offset:11264 // (op 325) Load local data 


ds_read_b64 v[40:41], v5 offset:11280 // (op 325) Load local data 



ds_read_u8 v12, v0 offset:32896 // (op 329) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[28:35], a[4:7], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 331) 
ds_read_b128 v[22:25], v5 offset:14336 // (op 487) Load local data 


ds_read_b64 v[26:27], v5 offset:14352 // (op 487) Load local data 



ds_read_u8 v12, v0 offset:33024 // (op 491) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:27], v[28:35], a[8:11], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 493) 
ds_read_b128 v[36:39], v5 offset:17408 // (op 649) Load local data 


ds_read_b64 v[40:41], v5 offset:17424 // (op 649) Load local data 



ds_read_u8 v12, v0 offset:33152 // (op 653) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:41], v[28:35], a[12:15], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 655) 
ds_read_b128 v[22:25], v5 offset:20480 // (op 811) Load local data 


ds_read_b64 v[26:27], v5 offset:20496 // (op 811) Load local data 



ds_read_u8 v12, v0 offset:33280 // (op 815) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:27], v[28:35], a[16:19], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 817) 
ds_read_b128 v[36:39], v5 offset:23552 // (op 973) Load local data 


ds_read_b64 v[40:41], v5 offset:23568 // (op 973) Load local data 



ds_read_u8 v12, v0 offset:33408 // (op 977) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[28:35], a[20:23], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 979) 
ds_read_b128 v[22:25], v5 offset:26624 // (op 1135) Load local data 


ds_read_b64 v[26:27], v5 offset:26640 // (op 1135) Load local data 



ds_read_u8 v12, v0 offset:33536 // (op 1139) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:27], v[28:35], a[24:27], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1141) 
ds_read_b128 v[36:39], v5 offset:29696 // (op 1297) Load local data 


ds_read_b64 v[40:41], v5 offset:29712 // (op 1297) Load local data 



ds_read_u8 v12, v0 offset:33664 // (op 1301) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[36:41], v[28:35], a[28:31], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1303) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1291) 
s_mov_b32 s10, s6 // (op 1291) convert

s_cmp_lt_u32 s1, s10 // (op 1291) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1291 // (op 1291) Condition: Bottom (jump to RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopTop_KLoopTail_1291 if true)
RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ForLoopBottom_KLoopTail_1291:






















s_barrier  // (op 4346) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_125
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 

v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 
v_cvt_f16_f32 v14, v18 // (op 52) 
v_cvt_f16_f32 v15, v19 // (op 52) 


v_pack_b32_f16 v4, v12, v13 // (op 54) 
v_pack_b32_f16 v5, v14, v15 // (op 54) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 54) Store value


v_accvgpr_read v12, a4 // (op 1621) 
v_accvgpr_read v13, a5 // (op 1621) 
v_accvgpr_read v14, a6 // (op 1621) 
v_accvgpr_read v15, a7 // (op 1621) 
v_mul_f32 v16, v11, v12 // (op 1621) 
v_mul_f32 v17, v11, v13 // (op 1621) 
v_mul_f32 v18, v11, v14 // (op 1621) 
v_mul_f32 v19, v11, v15 // (op 1621) 

v_cvt_f16_f32 v12, v16 // (op 1623) 
v_cvt_f16_f32 v13, v17 // (op 1623) 
v_cvt_f16_f32 v14, v18 // (op 1623) 
v_cvt_f16_f32 v15, v19 // (op 1623) 


v_pack_b32_f16 v4, v12, v13 // (op 1624) 
v_pack_b32_f16 v5, v14, v15 // (op 1624) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:64 // (op 1624) Store value


v_accvgpr_read v12, a8 // (op 1641) 
v_accvgpr_read v13, a9 // (op 1641) 
v_accvgpr_read v14, a10 // (op 1641) 
v_accvgpr_read v15, a11 // (op 1641) 
v_mul_f32 v16, v11, v12 // (op 1641) 
v_mul_f32 v17, v11, v13 // (op 1641) 
v_mul_f32 v18, v11, v14 // (op 1641) 
v_mul_f32 v19, v11, v15 // (op 1641) 

v_cvt_f16_f32 v12, v16 // (op 1643) 
v_cvt_f16_f32 v13, v17 // (op 1643) 
v_cvt_f16_f32 v14, v18 // (op 1643) 
v_cvt_f16_f32 v15, v19 // (op 1643) 


v_pack_b32_f16 v4, v12, v13 // (op 1644) 
v_pack_b32_f16 v5, v14, v15 // (op 1644) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:128 // (op 1644) Store value


v_accvgpr_read v12, a12 // (op 1661) 
v_accvgpr_read v13, a13 // (op 1661) 
v_accvgpr_read v14, a14 // (op 1661) 
v_accvgpr_read v15, a15 // (op 1661) 
v_mul_f32 v16, v11, v12 // (op 1661) 
v_mul_f32 v17, v11, v13 // (op 1661) 
v_mul_f32 v18, v11, v14 // (op 1661) 
v_mul_f32 v19, v11, v15 // (op 1661) 

v_cvt_f16_f32 v12, v16 // (op 1663) 
v_cvt_f16_f32 v13, v17 // (op 1663) 
v_cvt_f16_f32 v14, v18 // (op 1663) 
v_cvt_f16_f32 v15, v19 // (op 1663) 


v_pack_b32_f16 v4, v12, v13 // (op 1664) 
v_pack_b32_f16 v5, v14, v15 // (op 1664) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:192 // (op 1664) Store value


v_accvgpr_read v12, a16 // (op 1681) 
v_accvgpr_read v13, a17 // (op 1681) 
v_accvgpr_read v14, a18 // (op 1681) 
v_accvgpr_read v15, a19 // (op 1681) 
v_mul_f32 v16, v11, v12 // (op 1681) 
v_mul_f32 v17, v11, v13 // (op 1681) 
v_mul_f32 v18, v11, v14 // (op 1681) 
v_mul_f32 v19, v11, v15 // (op 1681) 

v_cvt_f16_f32 v12, v16 // (op 1683) 
v_cvt_f16_f32 v13, v17 // (op 1683) 
v_cvt_f16_f32 v14, v18 // (op 1683) 
v_cvt_f16_f32 v15, v19 // (op 1683) 


v_pack_b32_f16 v4, v12, v13 // (op 1684) 
v_pack_b32_f16 v5, v14, v15 // (op 1684) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:256 // (op 1684) Store value


v_accvgpr_read v12, a20 // (op 1701) 
v_accvgpr_read v13, a21 // (op 1701) 
v_accvgpr_read v14, a22 // (op 1701) 
v_accvgpr_read v15, a23 // (op 1701) 
v_mul_f32 v16, v11, v12 // (op 1701) 
v_mul_f32 v17, v11, v13 // (op 1701) 
v_mul_f32 v18, v11, v14 // (op 1701) 
v_mul_f32 v19, v11, v15 // (op 1701) 

v_cvt_f16_f32 v12, v16 // (op 1703) 
v_cvt_f16_f32 v13, v17 // (op 1703) 
v_cvt_f16_f32 v14, v18 // (op 1703) 
v_cvt_f16_f32 v15, v19 // (op 1703) 


v_pack_b32_f16 v4, v12, v13 // (op 1704) 
v_pack_b32_f16 v5, v14, v15 // (op 1704) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:320 // (op 1704) Store value


v_accvgpr_read v12, a24 // (op 1721) 
v_accvgpr_read v13, a25 // (op 1721) 
v_accvgpr_read v14, a26 // (op 1721) 
v_accvgpr_read v15, a27 // (op 1721) 
v_mul_f32 v16, v11, v12 // (op 1721) 
v_mul_f32 v17, v11, v13 // (op 1721) 
v_mul_f32 v18, v11, v14 // (op 1721) 
v_mul_f32 v19, v11, v15 // (op 1721) 

v_cvt_f16_f32 v12, v16 // (op 1723) 
v_cvt_f16_f32 v13, v17 // (op 1723) 
v_cvt_f16_f32 v14, v18 // (op 1723) 
v_cvt_f16_f32 v15, v19 // (op 1723) 


v_pack_b32_f16 v4, v12, v13 // (op 1724) 
v_pack_b32_f16 v5, v14, v15 // (op 1724) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:384 // (op 1724) Store value


v_accvgpr_read v12, a28 // (op 1741) 
v_accvgpr_read v13, a29 // (op 1741) 
v_accvgpr_read v14, a30 // (op 1741) 
v_accvgpr_read v15, a31 // (op 1741) 
v_mul_f32 v16, v11, v12 // (op 1741) 
v_mul_f32 v17, v11, v13 // (op 1741) 
v_mul_f32 v18, v11, v14 // (op 1741) 
v_mul_f32 v19, v11, v15 // (op 1741) 

v_cvt_f16_f32 v12, v16 // (op 1743) 
v_cvt_f16_f32 v13, v17 // (op 1743) 
v_cvt_f16_f32 v14, v18 // (op 1743) 
v_cvt_f16_f32 v15, v19 // (op 1743) 


v_pack_b32_f16 v4, v12, v13 // (op 1744) 
v_pack_b32_f16 v5, v14, v15 // (op 1744) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:448 // (op 1744) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:448 // (op 1924) Load value
v_accvgpr_read v12, a28 // (op 1920) 
v_accvgpr_read v13, a29 // (op 1920) 
v_accvgpr_read v14, a30 // (op 1920) 
v_accvgpr_read v15, a31 // (op 1920) 
v_mul_f32 v16, v11, v12 // (op 1920) 
v_mul_f32 v17, v11, v13 // (op 1920) 
v_mul_f32 v18, v11, v14 // (op 1920) 
v_mul_f32 v19, v11, v15 // (op 1920) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 1925) 
v_lshrrev_b32 v13, 16, v4 // (op 1925) 
v_cvt_f32_f16 v13, v13 // (op 1925) 
v_mul_f32 v20, v10, v12 // (op 1925) 
v_mul_f32 v21, v10, v13 // (op 1925) 
v_cvt_f32_f16 v12, v5 // (op 1925) 
v_lshrrev_b32 v13, 16, v5 // (op 1925) 
v_cvt_f32_f16 v13, v13 // (op 1925) 
v_mul_f32 v22, v10, v12 // (op 1925) 
v_mul_f32 v23, v10, v13 // (op 1925) 


v_add_f32 v12, v20, v16 // (op 1921) 
v_add_f32 v13, v21, v17 // (op 1921) 
v_add_f32 v14, v22, v18 // (op 1921) 
v_add_f32 v15, v23, v19 // (op 1921) 


v_cvt_f16_f32 v16, v12 // (op 1922) 
v_cvt_f16_f32 v17, v13 // (op 1922) 
v_cvt_f16_f32 v18, v14 // (op 1922) 
v_cvt_f16_f32 v19, v15 // (op 1922) 


v_pack_b32_f16 v4, v16, v17 // (op 1923) 
v_pack_b32_f16 v5, v18, v19 // (op 1923) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:448 // (op 1923) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:384 // (op 1904) Load value
v_accvgpr_read v12, a24 // (op 1900) 
v_accvgpr_read v13, a25 // (op 1900) 
v_accvgpr_read v14, a26 // (op 1900) 
v_accvgpr_read v15, a27 // (op 1900) 
v_mul_f32 v16, v11, v12 // (op 1900) 
v_mul_f32 v17, v11, v13 // (op 1900) 
v_mul_f32 v18, v11, v14 // (op 1900) 
v_mul_f32 v19, v11, v15 // (op 1900) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 1905) 
v_lshrrev_b32 v13, 16, v4 // (op 1905) 
v_cvt_f32_f16 v13, v13 // (op 1905) 
v_mul_f32 v20, v10, v12 // (op 1905) 
v_mul_f32 v21, v10, v13 // (op 1905) 
v_cvt_f32_f16 v12, v5 // (op 1905) 
v_lshrrev_b32 v13, 16, v5 // (op 1905) 
v_cvt_f32_f16 v13, v13 // (op 1905) 
v_mul_f32 v22, v10, v12 // (op 1905) 
v_mul_f32 v23, v10, v13 // (op 1905) 


v_add_f32 v12, v20, v16 // (op 1901) 
v_add_f32 v13, v21, v17 // (op 1901) 
v_add_f32 v14, v22, v18 // (op 1901) 
v_add_f32 v15, v23, v19 // (op 1901) 


v_cvt_f16_f32 v16, v12 // (op 1902) 
v_cvt_f16_f32 v17, v13 // (op 1902) 
v_cvt_f16_f32 v18, v14 // (op 1902) 
v_cvt_f16_f32 v19, v15 // (op 1902) 


v_pack_b32_f16 v4, v16, v17 // (op 1903) 
v_pack_b32_f16 v5, v18, v19 // (op 1903) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:384 // (op 1903) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:320 // (op 1884) Load value
v_accvgpr_read v12, a20 // (op 1880) 
v_accvgpr_read v13, a21 // (op 1880) 
v_accvgpr_read v14, a22 // (op 1880) 
v_accvgpr_read v15, a23 // (op 1880) 
v_mul_f32 v16, v11, v12 // (op 1880) 
v_mul_f32 v17, v11, v13 // (op 1880) 
v_mul_f32 v18, v11, v14 // (op 1880) 
v_mul_f32 v19, v11, v15 // (op 1880) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 1885) 
v_lshrrev_b32 v13, 16, v4 // (op 1885) 
v_cvt_f32_f16 v13, v13 // (op 1885) 
v_mul_f32 v20, v10, v12 // (op 1885) 
v_mul_f32 v21, v10, v13 // (op 1885) 
v_cvt_f32_f16 v12, v5 // (op 1885) 
v_lshrrev_b32 v13, 16, v5 // (op 1885) 
v_cvt_f32_f16 v13, v13 // (op 1885) 
v_mul_f32 v22, v10, v12 // (op 1885) 
v_mul_f32 v23, v10, v13 // (op 1885) 


v_add_f32 v12, v20, v16 // (op 1881) 
v_add_f32 v13, v21, v17 // (op 1881) 
v_add_f32 v14, v22, v18 // (op 1881) 
v_add_f32 v15, v23, v19 // (op 1881) 


v_cvt_f16_f32 v16, v12 // (op 1882) 
v_cvt_f16_f32 v17, v13 // (op 1882) 
v_cvt_f16_f32 v18, v14 // (op 1882) 
v_cvt_f16_f32 v19, v15 // (op 1882) 


v_pack_b32_f16 v4, v16, v17 // (op 1883) 
v_pack_b32_f16 v5, v18, v19 // (op 1883) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:320 // (op 1883) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:256 // (op 1864) Load value
v_accvgpr_read v12, a16 // (op 1860) 
v_accvgpr_read v13, a17 // (op 1860) 
v_accvgpr_read v14, a18 // (op 1860) 
v_accvgpr_read v15, a19 // (op 1860) 
v_mul_f32 v16, v11, v12 // (op 1860) 
v_mul_f32 v17, v11, v13 // (op 1860) 
v_mul_f32 v18, v11, v14 // (op 1860) 
v_mul_f32 v19, v11, v15 // (op 1860) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 1865) 
v_lshrrev_b32 v13, 16, v4 // (op 1865) 
v_cvt_f32_f16 v13, v13 // (op 1865) 
v_mul_f32 v20, v10, v12 // (op 1865) 
v_mul_f32 v21, v10, v13 // (op 1865) 
v_cvt_f32_f16 v12, v5 // (op 1865) 
v_lshrrev_b32 v13, 16, v5 // (op 1865) 
v_cvt_f32_f16 v13, v13 // (op 1865) 
v_mul_f32 v22, v10, v12 // (op 1865) 
v_mul_f32 v23, v10, v13 // (op 1865) 


v_add_f32 v12, v20, v16 // (op 1861) 
v_add_f32 v13, v21, v17 // (op 1861) 
v_add_f32 v14, v22, v18 // (op 1861) 
v_add_f32 v15, v23, v19 // (op 1861) 


v_cvt_f16_f32 v16, v12 // (op 1862) 
v_cvt_f16_f32 v17, v13 // (op 1862) 
v_cvt_f16_f32 v18, v14 // (op 1862) 
v_cvt_f16_f32 v19, v15 // (op 1862) 


v_pack_b32_f16 v4, v16, v17 // (op 1863) 
v_pack_b32_f16 v5, v18, v19 // (op 1863) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:256 // (op 1863) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:192 // (op 1844) Load value
v_accvgpr_read v12, a12 // (op 1840) 
v_accvgpr_read v13, a13 // (op 1840) 
v_accvgpr_read v14, a14 // (op 1840) 
v_accvgpr_read v15, a15 // (op 1840) 
v_mul_f32 v16, v11, v12 // (op 1840) 
v_mul_f32 v17, v11, v13 // (op 1840) 
v_mul_f32 v18, v11, v14 // (op 1840) 
v_mul_f32 v19, v11, v15 // (op 1840) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 1845) 
v_lshrrev_b32 v13, 16, v4 // (op 1845) 
v_cvt_f32_f16 v13, v13 // (op 1845) 
v_mul_f32 v20, v10, v12 // (op 1845) 
v_mul_f32 v21, v10, v13 // (op 1845) 
v_cvt_f32_f16 v12, v5 // (op 1845) 
v_lshrrev_b32 v13, 16, v5 // (op 1845) 
v_cvt_f32_f16 v13, v13 // (op 1845) 
v_mul_f32 v22, v10, v12 // (op 1845) 
v_mul_f32 v23, v10, v13 // (op 1845) 


v_add_f32 v12, v20, v16 // (op 1841) 
v_add_f32 v13, v21, v17 // (op 1841) 
v_add_f32 v14, v22, v18 // (op 1841) 
v_add_f32 v15, v23, v19 // (op 1841) 


v_cvt_f16_f32 v16, v12 // (op 1842) 
v_cvt_f16_f32 v17, v13 // (op 1842) 
v_cvt_f16_f32 v18, v14 // (op 1842) 
v_cvt_f16_f32 v19, v15 // (op 1842) 


v_pack_b32_f16 v4, v16, v17 // (op 1843) 
v_pack_b32_f16 v5, v18, v19 // (op 1843) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:192 // (op 1843) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:128 // (op 1824) Load value
v_accvgpr_read v12, a8 // (op 1820) 
v_accvgpr_read v13, a9 // (op 1820) 
v_accvgpr_read v14, a10 // (op 1820) 
v_accvgpr_read v15, a11 // (op 1820) 
v_mul_f32 v16, v11, v12 // (op 1820) 
v_mul_f32 v17, v11, v13 // (op 1820) 
v_mul_f32 v18, v11, v14 // (op 1820) 
v_mul_f32 v19, v11, v15 // (op 1820) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 1825) 
v_lshrrev_b32 v13, 16, v4 // (op 1825) 
v_cvt_f32_f16 v13, v13 // (op 1825) 
v_mul_f32 v20, v10, v12 // (op 1825) 
v_mul_f32 v21, v10, v13 // (op 1825) 
v_cvt_f32_f16 v12, v5 // (op 1825) 
v_lshrrev_b32 v13, 16, v5 // (op 1825) 
v_cvt_f32_f16 v13, v13 // (op 1825) 
v_mul_f32 v22, v10, v12 // (op 1825) 
v_mul_f32 v23, v10, v13 // (op 1825) 


v_add_f32 v12, v20, v16 // (op 1821) 
v_add_f32 v13, v21, v17 // (op 1821) 
v_add_f32 v14, v22, v18 // (op 1821) 
v_add_f32 v15, v23, v19 // (op 1821) 


v_cvt_f16_f32 v16, v12 // (op 1822) 
v_cvt_f16_f32 v17, v13 // (op 1822) 
v_cvt_f16_f32 v18, v14 // (op 1822) 
v_cvt_f16_f32 v19, v15 // (op 1822) 


v_pack_b32_f16 v4, v16, v17 // (op 1823) 
v_pack_b32_f16 v5, v18, v19 // (op 1823) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:128 // (op 1823) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen offset:64 // (op 1804) Load value
v_accvgpr_read v12, a4 // (op 1800) 
v_accvgpr_read v13, a5 // (op 1800) 
v_accvgpr_read v14, a6 // (op 1800) 
v_accvgpr_read v15, a7 // (op 1800) 
v_mul_f32 v16, v11, v12 // (op 1800) 
v_mul_f32 v17, v11, v13 // (op 1800) 
v_mul_f32 v18, v11, v14 // (op 1800) 
v_mul_f32 v19, v11, v15 // (op 1800) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 1805) 
v_lshrrev_b32 v13, 16, v4 // (op 1805) 
v_cvt_f32_f16 v13, v13 // (op 1805) 
v_mul_f32 v20, v10, v12 // (op 1805) 
v_mul_f32 v21, v10, v13 // (op 1805) 
v_cvt_f32_f16 v12, v5 // (op 1805) 
v_lshrrev_b32 v13, 16, v5 // (op 1805) 
v_cvt_f32_f16 v13, v13 // (op 1805) 
v_mul_f32 v22, v10, v12 // (op 1805) 
v_mul_f32 v23, v10, v13 // (op 1805) 


v_add_f32 v12, v20, v16 // (op 1801) 
v_add_f32 v13, v21, v17 // (op 1801) 
v_add_f32 v14, v22, v18 // (op 1801) 
v_add_f32 v15, v23, v19 // (op 1801) 


v_cvt_f16_f32 v16, v12 // (op 1802) 
v_cvt_f16_f32 v17, v13 // (op 1802) 
v_cvt_f16_f32 v18, v14 // (op 1802) 
v_cvt_f16_f32 v19, v15 // (op 1802) 


v_pack_b32_f16 v4, v16, v17 // (op 1803) 
v_pack_b32_f16 v5, v18, v19 // (op 1803) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen offset:64 // (op 1803) Store value





buffer_load_dwordx2 v[4:5], v6, s[40:43], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v11, v12 // (op 134) 
v_mul_f32 v17, v11, v13 // (op 134) 
v_mul_f32 v18, v11, v14 // (op 134) 
v_mul_f32 v19, v11, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v4 // (op 139) 
v_lshrrev_b32 v13, 16, v4 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v10, v12 // (op 139) 
v_mul_f32 v21, v10, v13 // (op 139) 
v_cvt_f32_f16 v12, v5 // (op 139) 
v_lshrrev_b32 v13, 16, v5 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v10, v12 // (op 139) 
v_mul_f32 v23, v10, v13 // (op 139) 


v_add_f32 v12, v20, v16 // (op 135) 
v_add_f32 v13, v21, v17 // (op 135) 
v_add_f32 v14, v22, v18 // (op 135) 
v_add_f32 v15, v23, v19 // (op 135) 


v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 
v_cvt_f16_f32 v18, v14 // (op 136) 
v_cvt_f16_f32 v19, v15 // (op 136) 


v_pack_b32_f16 v4, v16, v17 // (op 137) 
v_pack_b32_f16 v5, v18, v19 // (op 137) 


buffer_store_dwordx2 v[4:5], v9, s[44:47], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.LRR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_end:
.size RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2, .LRR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2_end-RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F8
    DestDataType: h
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
    macrotile: [256, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2
    .symbol: RR_GEMM_TN_BF6_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x32x128_UR_2.kd
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

