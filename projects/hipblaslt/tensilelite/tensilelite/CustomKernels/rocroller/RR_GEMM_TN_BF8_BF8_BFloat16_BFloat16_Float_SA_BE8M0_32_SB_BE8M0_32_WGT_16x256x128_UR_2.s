// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2,@function
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 3749) 
v_and_b32 v3, 15, v1 // (op 3749) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3749) 


v_and_b32 v0, 48, v1 // (op 3749) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3749) 


v_lshrrev_b32 v0, 7, v1 // (op 3921) 
v_and_b32 v4, 15, v1 // (op 3921) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3921) 


v_and_b32 v0, 48, v1 // (op 3921) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 3921) 


v_lshrrev_b32 v0, 7, v1 // (op 3693) 
v_and_b32 v5, 63, v1 // (op 3693) 
v_lshrrev_b32 v6, 4, v5 // (op 3693) 
v_lshlrev_b32 v7, 2, v6 // (op 3693) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3693) 


v_lshl_add_u32 v0, s16, 4, v6 // (op 3693) 

v_lshrrev_b32 v6, 2, v5 // (op 3693) 

v_and_b32 v5, 3, v6 // (op 3693) 

v_and_b32 v6, 3, v1 // (op 3693) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3693) 


v_lshrrev_b32 v5, 6, v1 // (op 3693) 
v_and_b32 v6, 1, v5 // (op 3693) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3693) 


v_lshl_add_u32 v6, s17, 8, v5 // (op 3693) 

v_mul_lo_u32 v5, v6, s2 // (op 3693) 

v_add_lshl_u32 v6, v0, v5, 1 // (op 3693) 


v_lshrrev_b32 v0, 6, v1 // (op 3806) 
v_and_b32 v5, 1, v0 // (op 3806) 

v_and_b32 v0, 15, v1 // (op 3806) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 3806) 


v_and_b32 v0, 48, v1 // (op 3806) 
v_lshl_add_u32 v5, v7, 7, v0 // (op 3806) 


v_lshrrev_b32 v0, 6, v1 // (op 3864) 
v_and_b32 v7, 1, v0 // (op 3864) 

v_and_b32 v0, 15, v1 // (op 3864) 
s_mul_i32 s1, s2, s9 // (op 3695) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3695) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3695) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3695) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3695) most significant: sum
s_add_u32 s1, s1, s19 // (op 3695) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3695) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 3864) 


v_and_b32 v0, 63, v1 // (op 3864) 
v_lshrrev_b32 v7, 4, v0 // (op 3864) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 3864) 


v_lshrrev_b32 v7, 7, v1 // (op 3880) 
v_and_b32 v8, 63, v1 // (op 3880) 
v_lshrrev_b32 v9, 4, v8 // (op 3880) 
v_lshlrev_b32 v10, 2, v9 // (op 3880) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3880) 


v_lshl_add_u32 v7, s16, 4, v9 // (op 3880) 

v_lshrrev_b32 v9, 2, v8 // (op 3880) 

v_and_b32 v8, 3, v9 // (op 3880) 

v_and_b32 v9, 3, v1 // (op 3880) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3880) 


v_lshrrev_b32 v8, 6, v1 // (op 3880) 
v_and_b32 v9, 1, v8 // (op 3880) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3880) 


v_lshl_add_u32 v9, s17, 8, v8 // (op 3880) 

v_mul_lo_u32 v8, v9, s4 // (op 3880) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 3880) 


v_lshrrev_b32 v7, 6, v1 // (op 3965) 
v_and_b32 v8, 1, v7 // (op 3965) 

v_and_b32 v7, 15, v1 // (op 3965) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 3965) 


v_and_b32 v7, 48, v1 // (op 3965) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 3965) 


v_lshrrev_b32 v7, 6, v1 // (op 3994) 
v_and_b32 v10, 1, v7 // (op 3994) 

v_and_b32 v7, 15, v1 // (op 3994) 
s_mul_i32 s1, s4, s9 // (op 3882) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 3994) 


v_and_b32 v7, 63, v1 // (op 3994) 
v_lshrrev_b32 v10, 4, v7 // (op 3994) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 3994) 


s_mul_i32 s38, s5, s8 // (op 3882) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3882) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3882) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3882) most significant: sum
s_add_u32 s1, s1, s39 // (op 3882) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 3882) 

s_mov_b32 s40, s6 // (op 3695) reg expression
s_mov_b32 s41, s7 // (op 3695) reg expression
s_mov_b32 s42, s18 // (op 3695) reg expression
s_mov_b32 s43, 131072 // (op 3695) call()

s_mov_b32 s44, s10 // (op 3882) reg expression

s_mov_b32 s45, s11 // (op 3882) reg expression
s_mov_b32 s46, s38 // (op 3882) reg expression
s_mov_b32 s47, 131072 // (op 3882) call()


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
v_lshrrev_b32 v12, 3, v1 // (op 3460) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 3460) 

v_and_b32 v12, -8, v1 // (op 3460) 
v_sub_u32 v14, v1, v12 // (op 3460) 

v_lshlrev_b32 v12, 4, v14 // (op 3460) 

v_mov_b32 v15, 0 // (op 3460) convert
v_mov_b32 v14, v12 // (op 3460) convert

v_mul_lo_u32 v17, v13, s13 // (op 3460) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3460) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3460) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3460) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3460) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3460) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3462) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3462) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3462) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3462) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3462) most significant: sum
s_add_u32 s1, s1, s7 // (op 3462) most significant: sum


s_mov_b32 s48, s20 // (op 3462) reg expression
s_mov_b32 s49, s21 // (op 3462) reg expression
s_mov_b32 s50, s0 // (op 3462) reg expression
s_mov_b32 s51, 131072 // (op 3462) call()

v_lshrrev_b32 v12, 3, v1 // (op 3479) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 3479) 

v_and_b32 v12, -8, v1 // (op 3479) 
v_sub_u32 v14, v1, v12 // (op 3479) 

v_lshlrev_b32 v12, 4, v14 // (op 3479) 

v_mov_b32 v15, 0 // (op 3479) convert
v_mov_b32 v14, v12 // (op 3479) convert

v_mul_lo_u32 v19, v13, s13 // (op 3479) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3479) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 3479) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3479) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 3479) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 3479) most significant half



s_mul_i32 s1, s12, s23 // (op 3481) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3481) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3481) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3481) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3481) most significant: sum
s_add_u32 s1, s1, s7 // (op 3481) most significant: sum


s_mov_b32 s48, s20 // (op 3481) reg expression
s_mov_b32 s49, s21 // (op 3481) reg expression
s_mov_b32 s50, s0 // (op 3481) reg expression
s_mov_b32 s51, 131072 // (op 3481) call()

v_and_b32 v12, -8, v1 // (op 3524) 
v_sub_u32 v13, v1, v12 // (op 3524) 

v_lshlrev_b32 v12, 4, v13 // (op 3524) 

v_mov_b32 v15, 0 // (op 3524) convert
v_mov_b32 v14, v12 // (op 3524) convert

v_lshrrev_b32 v12, 3, v1 // (op 3524) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3524) 

v_mul_lo_u32 v21, v13, s25 // (op 3524) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3524) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 3524) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 3524) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 3524) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 3524) most significant half



s_mul_i32 s1, s24, s9 // (op 3526) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3526) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3526) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3526) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3526) most significant: sum
s_add_u32 s1, s1, s7 // (op 3526) most significant: sum


s_mov_b32 s52, s26 // (op 3526) reg expression
s_mov_b32 s53, s27 // (op 3526) reg expression
s_mov_b32 s54, s0 // (op 3526) reg expression
s_mov_b32 s55, 131072 // (op 3526) call()

v_and_b32 v14, -8, v1 // (op 3543) 
v_sub_u32 v15, v1, v14 // (op 3543) 

v_lshlrev_b32 v14, 4, v15 // (op 3543) 

v_mov_b32 v21, 0 // (op 3543) convert
v_mov_b32 v20, v14 // (op 3543) convert

v_lshrrev_b32 v14, 3, v1 // (op 3543) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 3543) 

v_mul_lo_u32 v23, v15, s25 // (op 3543) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 3543) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 3543) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 3543) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 3543) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 3543) most significant half



s_mul_i32 s1, s24, s9 // (op 3545) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3545) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3545) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3545) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3545) most significant: sum
s_add_u32 s1, s1, s7 // (op 3545) most significant: sum


s_mov_b32 s52, s26 // (op 3545) reg expression
s_mov_b32 s53, s27 // (op 3545) reg expression
s_mov_b32 s54, s0 // (op 3545) reg expression
s_mov_b32 s55, 131072 // (op 3545) call()

v_lshrrev_b32 v20, 7, v1 // (op 3588) 
v_and_b32 v21, 15, v1 // (op 3588) 
v_lshl_add_u32 v22, v20, 4, v21 // (op 3588) 


v_lshl_add_u32 v20, s16, 4, v22 // (op 3588) 

v_and_b32 v21, 63, v1 // (op 3588) 
v_lshrrev_b32 v22, 4, v21 // (op 3588) 

v_mul_lo_u32 v21, v20, s28 // (op 3588) 
v_add_u32 v21, v21, v22 // (op 3588) 


s_mul_i32 s1, s28, s23 // (op 3590) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3590) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3590) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3590) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3590) most significant: sum
s_add_u32 s1, s1, s7 // (op 3590) most significant: sum


s_mov_b32 s56, s30 // (op 3590) reg expression
s_mov_b32 s57, s31 // (op 3590) reg expression
s_mov_b32 s58, s0 // (op 3590) reg expression
s_mov_b32 s59, 131072 // (op 3590) call()

v_lshrrev_b32 v20, 7, v1 // (op 3607) 
v_and_b32 v22, 15, v1 // (op 3607) 
v_lshl_add_u32 v23, v20, 4, v22 // (op 3607) 


v_lshl_add_u32 v20, s16, 4, v23 // (op 3607) 

v_and_b32 v22, 63, v1 // (op 3607) 
v_lshrrev_b32 v23, 4, v22 // (op 3607) 

v_mul_lo_u32 v22, v20, s28 // (op 3607) 
v_add_u32 v22, v22, v23 // (op 3607) 


s_mul_i32 s1, s28, s23 // (op 3609) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3609) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3609) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3609) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3609) most significant: sum
s_add_u32 s1, s1, s7 // (op 3609) most significant: sum


s_mov_b32 s56, s30 // (op 3609) reg expression
s_mov_b32 s57, s31 // (op 3609) reg expression
s_mov_b32 s58, s0 // (op 3609) reg expression
s_mov_b32 s59, 131072 // (op 3609) call()

v_lshlrev_b32 v20, 1, v1 // (op 3652) 
v_lshl_add_u32 v23, s17, 8, v20 // (op 3652) 

v_mul_lo_u32 v20, v23, s32 // (op 3652) 

s_mul_i32 s1, s32, s9 // (op 3654) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3654) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3654) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3654) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3654) most significant: sum
s_add_u32 s1, s1, s7 // (op 3654) most significant: sum


s_mov_b32 s60, s34 // (op 3654) reg expression
s_mov_b32 s61, s35 // (op 3654) reg expression
s_mov_b32 s62, s0 // (op 3654) reg expression
s_mov_b32 s63, 131072 // (op 3654) call()

v_lshlrev_b32 v23, 1, v1 // (op 3671) 
v_lshl_add_u32 v24, s17, 8, v23 // (op 3671) 

v_mul_lo_u32 v23, v24, s32 // (op 3671) 

s_mul_i32 s1, s32, s9 // (op 3673) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3673) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3673) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3673) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3673) most significant: sum
s_add_u32 s1, s1, s7 // (op 3673) most significant: sum


s_mov_b32 s60, s34 // (op 3673) reg expression
s_mov_b32 s61, s35 // (op 3673) reg expression
s_mov_b32 s62, s0 // (op 3673) reg expression
s_mov_b32 s63, 131072 // (op 3673) call()




buffer_load_ubyte v24, v21, s[56:59], 0 offen offset: 0 // (op 3140) Load value
v_and_b32 v25, -8, v1 // (op 3789) 
v_sub_u32 v26, v1, v25 // (op 3789) 

v_lshlrev_b32 v25, 4, v26 // (op 3789) 

v_lshrrev_b32 v26, 3, v1 // (op 3789) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3789) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 3789) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 4 // (op 3402) 
v_mov_b64 v[26:27], s[6:7] // (op 3402) convert


v_mov_b64 v[28:29], v[12:13] // (op 3402) 
s_mov_b32 m0, s0 // (op 3402) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 3402) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3402) most significant half

s_add_u32 m0, m0, 2048 // (op 3402) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3402) Load value direct to lds




v_mov_b64 v[26:27], s[32:33] // (op 3147) convert

v_mov_b32 v25, v20 // (op 3147) 
buffer_load_dword v28, v25, s[60:63], 0 offen // (op 3147) Load value
v_add_u32 v25, v25, v26 // (op 3147) 
buffer_load_dword v29, v25, s[60:63], 0 offen // (op 3147) Load value


v_and_b32 v25, -8, v1 // (op 3737) 
v_sub_u32 v26, v1, v25 // (op 3737) 

v_lshlrev_b32 v25, 4, v26 // (op 3737) 

v_lshrrev_b32 v26, 3, v1 // (op 3737) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3737) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 3737) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 32768, s0 // (op 3407) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3407) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 3407) Load value direct to lds





buffer_load_ubyte v25, v21, s[56:59], 0 offen offset:4 // (op 3154) Load value

v_and_b32 v21, -8, v1 // (op 3948) 
v_sub_u32 v26, v1, v21 // (op 3948) 

v_lshlrev_b32 v21, 4, v26 // (op 3948) 

v_lshrrev_b32 v26, 3, v1 // (op 3948) 
v_lshl_add_u32 v27, v26, 7, v21 // (op 3948) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 3948) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 4 // (op 3412) 
v_mov_b64 v[26:27], s[6:7] // (op 3412) convert


v_mov_b64 v[30:31], v[12:13] // (op 3412) 
s_add_u32 s1, 34816, s0 // (op 3412) 
s_mov_b32 m0, s1 // (op 3412) reg expression
s_mov_b32 s6, 128 // (op 3412) call()
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds
v_add_co_u32 v30, s[10:11], v30, v26 // (op 3412) least significant half
v_addc_co_u32 v31, s[10:11], 0, v31, s[10:11] // (op 3412) most significant half

s_add_u32 m0, m0, 2048 // (op 3412) 
buffer_load_dwordx4 v30, s[52:55], s6 offen offset: 0 lds // (op 3412) Load value direct to lds







v_mov_b64 v[12:13], s[32:33] // (op 3161) convert

v_mov_b32 v21, v20 // (op 3161) 
buffer_load_dword v26, v21, s[60:63], 0 offen offset:4 // (op 3161) Load value
v_add_u32 v21, v21, v12 // (op 3161) 
buffer_load_dword v27, v21, s[60:63], 0 offen offset:4 // (op 3161) Load value



v_and_b32 v12, -8, v1 // (op 3909) 
v_sub_u32 v13, v1, v12 // (op 3909) 

v_lshlrev_b32 v12, 4, v13 // (op 3909) 

v_lshrrev_b32 v13, 3, v1 // (op 3909) 
v_lshl_add_u32 v20, v13, 7, v12 // (op 3909) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 3909) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 67584, s0 // (op 3414) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3414) reg expression
s_mov_b32 s6, 128 // (op 3414) call()
buffer_load_dwordx4 v16, s[48:51], s6 offen offset: 0 lds // (op 3414) Load value direct to lds




v_lshlrev_b32 v12, 3, v1 // (op 3847) 



v_add_u32 v13, 69632, v12 // (op 3167) 
s_waitcnt vmcnt(22)
ds_write_b32 v13, v28 // (op 3167) Store local data 

v_add_u32 v13, 69636, v12 // (op 3167) 
s_waitcnt vmcnt(21)
ds_write_b32 v13, v29 // (op 3167) Store local data 


s_barrier  // (op 4069) 



ds_read_b128 v[30:33], v5 // (op 3185) Load local data 
ds_read_b128 v[34:37], v5 offset:64 // (op 3185) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4085) 



v_add_u32 v12, 69632, v0 // (op 3225) 
ds_read_u8 v13, v12 // (op 3225) Load local data 




ds_read_b128 v[38:41], v5 offset:4096 // (op 3190) Load local data 
ds_read_b128 v[42:45], v5 offset:4160 // (op 3190) Load local data 



v_add_u32 v12, 69760, v0 // (op 3230) 
ds_read_u8 v16, v12 // (op 3230) Load local data 




ds_read_b128 v[46:49], v5 offset:8192 // (op 3195) Load local data 
ds_read_b128 v[50:53], v5 offset:8256 // (op 3195) Load local data 



v_add_u32 v12, 69888, v0 // (op 3235) 
ds_read_u8 v17, v12 // (op 3235) Load local data 




ds_read_b128 v[54:57], v5 offset:12288 // (op 3200) Load local data 
ds_read_b128 v[58:61], v5 offset:12352 // (op 3200) Load local data 



v_add_u32 v12, 70016, v0 // (op 3240) 
ds_read_u8 v20, v12 // (op 3240) Load local data 




ds_read_b128 v[62:65], v5 offset:16384 // (op 3205) Load local data 
ds_read_b128 v[66:69], v5 offset:16448 // (op 3205) Load local data 



v_add_u32 v12, 70144, v0 // (op 3245) 
ds_read_u8 v21, v12 // (op 3245) Load local data 




ds_read_b128 v[70:73], v5 offset:20480 // (op 3210) Load local data 
ds_read_b128 v[74:77], v5 offset:20544 // (op 3210) Load local data 



v_add_u32 v12, 70272, v0 // (op 3250) 
ds_read_u8 v78, v12 // (op 3250) Load local data 




ds_read_b128 v[80:83], v5 offset:24576 // (op 3215) Load local data 
ds_read_b128 v[84:87], v5 offset:24640 // (op 3215) Load local data 



v_add_u32 v12, 70400, v0 // (op 3255) 
ds_read_u8 v79, v12 // (op 3255) Load local data 




ds_read_b128 v[88:91], v5 offset:28672 // (op 3220) Load local data 
ds_read_b128 v[92:95], v5 offset:28736 // (op 3220) Load local data 



v_add_u32 v12, 70528, v0 // (op 3260) 
ds_read_u8 v96, v12 // (op 3260) Load local data 

s_waitcnt vmcnt(20)
s_barrier  // (op 4053) 



ds_read_b128 v[98:101], v3 offset:32768 // (op 3181) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 3181) Load local data 
v_and_b32 v12, -8, v1 // (op 3709) 
v_sub_u32 v97, v1, v12 // (op 3709) 

v_lshlrev_b32 v12, 4, v97 // (op 3709) 

v_lshrrev_b32 v97, 3, v1 // (op 3709) 
v_lshl_add_u32 v106, v97, 7, v12 // (op 3709) 


s_nop 0
v_readfirstlane_b32 s0, v106 // (op 3709) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 3761) 
v_sub_u32 v97, v1, v12 // (op 3761) 

v_lshlrev_b32 v12, 4, v97 // (op 3761) 

v_lshrrev_b32 v97, 3, v1 // (op 3761) 
v_lshl_add_u32 v106, v97, 7, v12 // (op 3761) 


s_nop 0
v_readfirstlane_b32 s1, v106 // (op 3761) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 3, v1 // (op 3818) 
v_and_b32 v97, -8, v1 // (op 3894) 
v_sub_u32 v106, v1, v97 // (op 3894) 

v_lshlrev_b32 v97, 4, v106 // (op 3894) 

v_lshrrev_b32 v106, 3, v1 // (op 3894) 
v_lshl_add_u32 v107, v106, 7, v97 // (op 3894) 


s_nop 0
v_readfirstlane_b32 s6, v107 // (op 3894) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v97, -8, v1 // (op 3933) 
v_sub_u32 v106, v1, v97 // (op 3933) 

v_lshlrev_b32 v97, 4, v106 // (op 3933) 

v_lshrrev_b32 v106, 3, v1 // (op 3933) 
v_lshl_add_u32 v107, v106, 7, v97 // (op 3933) 


s_nop 0
v_readfirstlane_b32 s7, v107 // (op 3933) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v97, 3, v1 // (op 3977) 
s_mov_b32 s10, 0 // (op 1248) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1247) 
s_mov_b32 s11, s14 // (op 1247) convert

s_and_b32 s14, s11, 4294967294 // (op 1247) 

s_cmp_lt_u32 s10, s14 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoop_1247 // (op 1247) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoop_1247 if false)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoop_1247:
s_add_u32 s10, s10, 2 // (op 1249) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[30:37], a[0:3], v24, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 84) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[38:45], a[4:7], v24, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 347) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[46:53], a[8:11], v24, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 503) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[54:61], a[12:15], v24, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 659) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[62:69], a[16:19], v24, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 815) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[70:77], a[20:23], v24, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 971) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[80:87], a[24:27], v24, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1127) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[88:95], a[28:31], v24, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1283) 


v_mov_b32 v106, v22 // (op 77) 

buffer_load_ubyte v24, v106, s[56:59], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 4072) 

s_lshl_b64 s[14:15], s[24:25], 4 // (op 3374) 
v_mov_b64 v[108:109], s[14:15] // (op 3374) convert


v_mov_b64 v[110:111], v[14:15] // (op 3374) 
v_mov_b64 v[112:113], v[110:111] // (op 3374) 
s_mov_b32 m0, s1 // (op 3374) reg expression
s_mov_b32 s11, 256 // (op 3374) call()
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds
v_add_co_u32 v112, s[14:15], v112, v108 // (op 3374) least significant half
v_addc_co_u32 v113, s[14:15], 0, v113, s[14:15] // (op 3374) most significant half

s_add_u32 m0, m0, 2048 // (op 3374) 
buffer_load_dwordx4 v112, s[52:55], s11 offen offset: 0 lds // (op 3374) Load value direct to lds




v_mov_b64 v[108:109], s[32:33] // (op 21) convert

v_mov_b32 v107, v23 // (op 21) 
v_mov_b32 v112, v107 // (op 21) 
buffer_load_dword v28, v112, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v112, v112, v108 // (op 21) 
buffer_load_dword v29, v112, s[60:63], 0 offen offset:8 // (op 21) Load value


s_barrier  // (op 4056) 


v_mov_b64 v[108:109], v[18:19] // (op 3369) 
s_add_u32 s11, 32768, s0 // (op 3369) 

s_mov_b32 m0, s11 // (op 3369) reg expression
s_mov_b32 s14, 256 // (op 3369) call()
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3369) Load value direct to lds


s_barrier  // (op 4117) 



v_add_u32 v112, 70656, v97 // (op 240) 
s_waitcnt vmcnt(22)
ds_write_b32 v112, v26 // (op 240) Store local data 

v_add_u32 v112, 70660, v97 // (op 240) 
s_waitcnt vmcnt(21)
ds_write_b32 v112, v27 // (op 240) Store local data 

s_barrier  // (op 4108) 



ds_read_b128 v[30:33], v8 offset:34816 // (op 233) Load local data 
ds_read_b128 v[34:37], v8 offset:34880 // (op 233) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4114) 



v_add_u32 v112, 70656, v7 // (op 237) 
ds_read_u8 v13, v112 // (op 237) Load local data 




ds_read_b128 v[38:41], v8 offset:38912 // (op 352) Load local data 
ds_read_b128 v[42:45], v8 offset:38976 // (op 352) Load local data 



v_add_u32 v112, 70784, v7 // (op 355) 
ds_read_u8 v16, v112 // (op 355) Load local data 




ds_read_b128 v[46:49], v8 offset:43008 // (op 508) Load local data 
ds_read_b128 v[50:53], v8 offset:43072 // (op 508) Load local data 



v_add_u32 v112, 70912, v7 // (op 511) 
ds_read_u8 v17, v112 // (op 511) Load local data 




ds_read_b128 v[54:57], v8 offset:47104 // (op 664) Load local data 
ds_read_b128 v[58:61], v8 offset:47168 // (op 664) Load local data 



v_add_u32 v112, 71040, v7 // (op 667) 
ds_read_u8 v20, v112 // (op 667) Load local data 




ds_read_b128 v[62:65], v8 offset:51200 // (op 820) Load local data 
ds_read_b128 v[66:69], v8 offset:51264 // (op 820) Load local data 



v_add_u32 v112, 71168, v7 // (op 823) 
ds_read_u8 v21, v112 // (op 823) Load local data 




ds_read_b128 v[70:73], v8 offset:55296 // (op 976) Load local data 
ds_read_b128 v[74:77], v8 offset:55360 // (op 976) Load local data 



v_add_u32 v112, 71296, v7 // (op 979) 
ds_read_u8 v78, v112 // (op 979) Load local data 




ds_read_b128 v[80:83], v8 offset:59392 // (op 1132) Load local data 
ds_read_b128 v[84:87], v8 offset:59456 // (op 1132) Load local data 



v_add_u32 v112, 71424, v7 // (op 1135) 
ds_read_u8 v79, v112 // (op 1135) Load local data 




ds_read_b128 v[88:91], v8 offset:63488 // (op 1288) Load local data 
ds_read_b128 v[92:95], v8 offset:63552 // (op 1288) Load local data 



v_add_u32 v112, 71552, v7 // (op 1291) 
ds_read_u8 v96, v112 // (op 1291) Load local data 

s_barrier  // (op 4196) 
s_waitcnt vmcnt(20)
s_barrier  // (op 4102) 



v_add_u32 v112, 67584, v4 // (op 231) 
ds_read_b128 v[98:101], v112 // (op 231) Load local data 

v_add_u32 v112, 67648, v4 // (op 231) 
ds_read_b128 v[102:105], v112 // (op 231) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[30:37], a[0:3], v25, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 238) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[38:45], a[4:7], v25, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 348) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[46:53], a[8:11], v25, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 504) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[54:61], a[12:15], v25, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 660) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[62:69], a[16:19], v25, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 816) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[70:77], a[20:23], v25, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 972) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[80:87], a[24:27], v25, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1128) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[88:95], a[28:31], v25, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1284) 



buffer_load_ubyte v25, v106, s[56:59], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 4111) 
v_add_u32 v22, 8, v22 // (op 3599) 

s_lshl_b64 s[14:15], s[24:25], 4 // (op 3397) 
v_mov_b64 v[112:113], s[14:15] // (op 3397) convert


v_mov_b64 v[114:115], v[110:111] // (op 3397) 
s_add_u32 s11, 34816, s7 // (op 3397) 
s_mov_b32 m0, s11 // (op 3397) reg expression
s_mov_b32 s14, 384 // (op 3397) call()
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v112 // (op 3397) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 3397) most significant half

s_add_u32 m0, m0, 2048 // (op 3397) 
buffer_load_dwordx4 v114, s[52:55], s14 offen offset: 0 lds // (op 3397) Load value direct to lds






v_add_co_u32 v14, vcc, 256, v14 // (op 3535) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3535) most significant half
v_mov_b64 v[110:111], s[32:33] // (op 242) convert

v_mov_b32 v106, v107 // (op 242) 
buffer_load_dword v26, v106, s[60:63], 0 offen offset:12 // (op 242) Load value
v_add_u32 v106, v106, v110 // (op 242) 
buffer_load_dword v27, v106, s[60:63], 0 offen offset:12 // (op 242) Load value



s_barrier  // (op 4105) 
v_add_u32 v23, 8, v23 // (op 3663) 


s_add_u32 s11, 67584, s6 // (op 3392) 

s_mov_b32 m0, s11 // (op 3392) reg expression
s_mov_b32 s14, 384 // (op 3392) call()
buffer_load_dwordx4 v108, s[48:51], s14 offen offset: 0 lds // (op 3392) Load value direct to lds



s_barrier  // (op 4088) 
v_add_co_u32 v18, vcc, 256, v18 // (op 3471) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3471) most significant half
v_add_u32 v106, 69632, v12 // (op 26) 
s_waitcnt vmcnt(22)
ds_write_b32 v106, v28 // (op 26) Store local data 

v_add_u32 v106, 69636, v12 // (op 26) 
s_waitcnt vmcnt(21)
ds_write_b32 v106, v29 // (op 26) Store local data 

s_barrier  // (op 4078) 



ds_read_b128 v[30:33], v5 // (op 72) Load local data 
ds_read_b128 v[34:37], v5 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4095) 



v_add_u32 v106, 69632, v0 // (op 81) 
ds_read_u8 v13, v106 // (op 81) Load local data 




ds_read_b128 v[38:41], v5 offset:4096 // (op 343) Load local data 
ds_read_b128 v[42:45], v5 offset:4160 // (op 343) Load local data 



v_add_u32 v106, 69760, v0 // (op 346) 
ds_read_u8 v16, v106 // (op 346) Load local data 




ds_read_b128 v[46:49], v5 offset:8192 // (op 499) Load local data 
ds_read_b128 v[50:53], v5 offset:8256 // (op 499) Load local data 



v_add_u32 v106, 69888, v0 // (op 502) 
ds_read_u8 v17, v106 // (op 502) Load local data 




ds_read_b128 v[54:57], v5 offset:12288 // (op 655) Load local data 
ds_read_b128 v[58:61], v5 offset:12352 // (op 655) Load local data 



v_add_u32 v106, 70016, v0 // (op 658) 
ds_read_u8 v20, v106 // (op 658) Load local data 




ds_read_b128 v[62:65], v5 offset:16384 // (op 811) Load local data 
ds_read_b128 v[66:69], v5 offset:16448 // (op 811) Load local data 



v_add_u32 v106, 70144, v0 // (op 814) 
ds_read_u8 v21, v106 // (op 814) Load local data 




ds_read_b128 v[70:73], v5 offset:20480 // (op 967) Load local data 
ds_read_b128 v[74:77], v5 offset:20544 // (op 967) Load local data 



v_add_u32 v106, 70272, v0 // (op 970) 
ds_read_u8 v78, v106 // (op 970) Load local data 




ds_read_b128 v[80:83], v5 offset:24576 // (op 1123) Load local data 
ds_read_b128 v[84:87], v5 offset:24640 // (op 1123) Load local data 



v_add_u32 v106, 70400, v0 // (op 1126) 
ds_read_u8 v79, v106 // (op 1126) Load local data 




ds_read_b128 v[88:91], v5 offset:28672 // (op 1279) Load local data 
ds_read_b128 v[92:95], v5 offset:28736 // (op 1279) Load local data 



v_add_u32 v106, 70528, v0 // (op 1282) 
ds_read_u8 v96, v106 // (op 1282) Load local data 

s_waitcnt vmcnt(20)
s_barrier  // (op 4062) 



ds_read_b128 v[98:101], v3 offset:32768 // (op 67) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 67) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 1247) 
s_mov_b32 s11, s14 // (op 1247) convert

s_and_b32 s14, s11, 4294967294 // (op 1247) 

s_cmp_lt_u32 s10, s14 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoop_1247 // (op 1247) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoop_1247 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoop_1247:




















s_barrier  // (op 4180) 
v_lshrrev_b32 v97, 3, v1 // (op 3427) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3427) 
v_lshl_add_u32 v12, s16, 4, v97 // (op 3427) 

s_lshr_b32 s6, s0, 1 // (op 3427) 

v_and_b32 v97, -8, v1 // (op 3427) 
v_sub_u32 v13, v1, v97 // (op 3427) 

v_lshlrev_b32 v97, 4, v13 // (op 3427) 

v_lshl_add_u32 v13, s6, 8, v97 // (op 3427) 


v_mov_b32 v15, 0 // (op 3427) convert
v_mov_b32 v14, v13 // (op 3427) convert

v_mul_lo_u32 v17, v12, s13 // (op 3427) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 3427) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 3427) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3427) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3427) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3427) most significant half



s_mul_i32 s1, s12, s23 // (op 3429) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3429) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3429) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3429) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3429) most significant: sum
s_add_u32 s1, s1, s7 // (op 3429) most significant: sum


s_mov_b32 s48, s20 // (op 3429) reg expression
s_mov_b32 s49, s21 // (op 3429) reg expression
s_mov_b32 s50, s0 // (op 3429) reg expression
s_mov_b32 s51, 131072 // (op 3429) call()

v_and_b32 v97, -8, v1 // (op 3501) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3501) 
v_sub_u32 v12, v1, v97 // (op 3501) 

s_lshr_b32 s6, s0, 1 // (op 3501) 

v_lshlrev_b32 v97, 4, v12 // (op 3501) 

v_lshl_add_u32 v12, s6, 8, v97 // (op 3501) 


v_mov_b32 v15, 0 // (op 3501) convert
v_mov_b32 v14, v12 // (op 3501) convert

v_lshrrev_b32 v97, 3, v1 // (op 3501) 
v_lshl_add_u32 v12, s17, 8, v97 // (op 3501) 

v_mul_lo_u32 v19, v12, s25 // (op 3501) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 3501) most significant: high of low * low
v_mul_lo_u32 v18, v12, s24 // (op 3501) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 3501) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v18 // (op 3501) least significant half
v_addc_co_u32 v13, s[0:1], v15, v19, s[0:1] // (op 3501) most significant half



s_mul_i32 s1, s24, s9 // (op 3503) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3503) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3503) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3503) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3503) most significant: sum
s_add_u32 s1, s1, s7 // (op 3503) most significant: sum


s_mov_b32 s52, s26 // (op 3503) reg expression
s_mov_b32 s53, s27 // (op 3503) reg expression
s_mov_b32 s54, s0 // (op 3503) reg expression
s_mov_b32 s55, 131072 // (op 3503) call()

v_lshrrev_b32 v97, 7, v1 // (op 3565) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3565) 
v_and_b32 v14, 15, v1 // (op 3565) 
s_lshr_b32 s6, s0, 1 // (op 3565) 

v_lshl_add_u32 v15, v97, 4, v14 // (op 3565) 


v_lshl_add_u32 v14, s16, 4, v15 // (op 3565) 

v_and_b32 v15, 63, v1 // (op 3565) 
v_lshrrev_b32 v97, 4, v15 // (op 3565) 

v_lshl_add_u32 v15, s6, 3, v97 // (op 3565) 


v_mul_lo_u32 v97, v14, s28 // (op 3565) 
v_add_u32 v97, v97, v15 // (op 3565) 


s_mul_i32 s1, s28, s23 // (op 3567) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3567) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3567) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3567) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3567) most significant: sum
s_add_u32 s1, s1, s7 // (op 3567) most significant: sum


s_mov_b32 s56, s30 // (op 3567) reg expression
s_mov_b32 s57, s31 // (op 3567) reg expression
s_mov_b32 s58, s0 // (op 3567) reg expression
s_mov_b32 s59, 131072 // (op 3567) call()

v_lshlrev_b32 v14, 1, v1 // (op 3629) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 3629) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 3629) 
v_mul_lo_u32 v14, v15, s32 // (op 3629) 

s_lshr_b32 s6, s0, 1 // (op 3629) 

v_lshl_add_u32 v15, s6, 3, v14 // (op 3629) 


s_mul_i32 s1, s32, s9 // (op 3631) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3631) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3631) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3631) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3631) most significant: sum
s_add_u32 s1, s1, s7 // (op 3631) most significant: sum


s_mov_b32 s60, s34 // (op 3631) reg expression
s_mov_b32 s61, s35 // (op 3631) reg expression
s_mov_b32 s62, s0 // (op 3631) reg expression
s_mov_b32 s63, 131072 // (op 3631) call()

v_and_b32 v14, -8, v1 // (op 3722) 
v_sub_u32 v18, v1, v14 // (op 3722) 

v_lshlrev_b32 v14, 4, v18 // (op 3722) 

v_lshrrev_b32 v18, 3, v1 // (op 3722) 
v_lshl_add_u32 v19, v18, 7, v14 // (op 3722) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 3722) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 3774) 
v_sub_u32 v18, v1, v14 // (op 3774) 

v_lshlrev_b32 v14, 4, v18 // (op 3774) 

v_lshrrev_b32 v18, 3, v1 // (op 3774) 
v_lshl_add_u32 v19, v18, 7, v14 // (op 3774) 


s_nop 0
v_readfirstlane_b32 s1, v19 // (op 3774) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 3, v1 // (op 3831) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1253) 
s_and_b32 s10, s6, 4294967294 // (op 1253) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1252) 
s_mov_b32 s11, s6 // (op 1252) convert

s_cmp_lt_u32 s10, s11 // (op 1252) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoopTail_1252 // (op 1252) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoopTail_1252 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoopTail_1252:
s_barrier  // (op 4059) 
s_add_u32 s10, s10, 1 // (op 1254) 


v_mov_b64 v[18:19], v[16:17] // (op 3379) 
s_add_u32 s11, 32768, s0 // (op 3379) 

s_mov_b32 m0, s11 // (op 3379) reg expression
buffer_load_dwordx4 v18, s[48:51], 0 offen offset: 0 lds // (op 3379) Load value direct to lds


v_add_co_u32 v16, vcc, 128, v16 // (op 3417) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 3417) most significant half
s_barrier  // (op 4075) 

s_lshl_b64 s[6:7], s[24:25], 4 // (op 3385) 
v_mov_b64 v[18:19], s[6:7] // (op 3385) convert


v_mov_b64 v[20:21], v[12:13] // (op 3385) 
v_mov_b64 v[22:23], v[20:21] // (op 3385) 
s_mov_b32 m0, s1 // (op 3385) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 3385) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 3385) most significant half

s_add_u32 m0, m0, 2048 // (op 3385) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 3385) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 3493) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 3493) most significant half

v_mov_b64 v[18:19], s[32:33] // (op 184) convert

v_mov_b32 v20, v15 // (op 184) 
v_mov_b32 v21, v20 // (op 184) 
buffer_load_dword v22, v21, s[60:63], 0 offen // (op 184) Load value
v_add_u32 v21, v21, v18 // (op 184) 
buffer_load_dword v23, v21, s[60:63], 0 offen // (op 184) Load value



v_add_u32 v15, 4, v15 // (op 3621) 
s_barrier  // (op 4091) 
s_waitcnt vmcnt(18)
s_barrier  // (op 4065) 






v_add_u32 v18, 69632, v14 // (op 182) 
ds_read_b128 v[24:27], v3 offset:32768 // (op 173) Load local data 
ds_read_b128 v[28:31], v3 offset:32832 // (op 173) Load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v18, v22 // (op 182) Store local data 

s_barrier  // (op 4081) 
v_add_u32 v18, 69636, v14 // (op 182) 



ds_read_b128 v[32:35], v5 // (op 175) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 175) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v18, v23 // (op 182) Store local data 




v_mov_b32 v18, v97 // (op 177) 

buffer_load_ubyte v19, v18, s[56:59], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v97, 4, v97 // (op 3557) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4098) 



v_add_u32 v18, 69632, v0 // (op 179) 
ds_read_u8 v20, v18 // (op 179) Load local data 




ds_read_b128 v[40:43], v5 offset:4096 // (op 324) Load local data 
ds_read_b128 v[44:47], v5 offset:4160 // (op 324) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:39], a[0:3], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 180) 





v_add_u32 v18, 69760, v0 // (op 327) 
ds_read_u8 v20, v18 // (op 327) Load local data 




ds_read_b128 v[32:35], v5 offset:8192 // (op 480) Load local data 
ds_read_b128 v[36:39], v5 offset:8256 // (op 480) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:31], v[40:47], a[4:7], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 328) 





v_add_u32 v18, 69888, v0 // (op 483) 
ds_read_u8 v20, v18 // (op 483) Load local data 




ds_read_b128 v[40:43], v5 offset:12288 // (op 636) Load local data 
ds_read_b128 v[44:47], v5 offset:12352 // (op 636) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[24:31], v[32:39], a[8:11], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 484) 





v_add_u32 v18, 70016, v0 // (op 639) 
ds_read_u8 v20, v18 // (op 639) Load local data 




ds_read_b128 v[32:35], v5 offset:16384 // (op 792) Load local data 
ds_read_b128 v[36:39], v5 offset:16448 // (op 792) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[24:31], v[40:47], a[12:15], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 640) 





v_add_u32 v18, 70144, v0 // (op 795) 
ds_read_u8 v20, v18 // (op 795) Load local data 




ds_read_b128 v[40:43], v5 offset:20480 // (op 948) Load local data 
ds_read_b128 v[44:47], v5 offset:20544 // (op 948) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:31], v[32:39], a[16:19], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 796) 





v_add_u32 v18, 70272, v0 // (op 951) 
ds_read_u8 v20, v18 // (op 951) Load local data 




ds_read_b128 v[32:35], v5 offset:24576 // (op 1104) Load local data 
ds_read_b128 v[36:39], v5 offset:24640 // (op 1104) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:31], v[40:47], a[20:23], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 952) 





v_add_u32 v18, 70400, v0 // (op 1107) 
ds_read_u8 v20, v18 // (op 1107) Load local data 




ds_read_b128 v[40:43], v5 offset:28672 // (op 1260) Load local data 
ds_read_b128 v[44:47], v5 offset:28736 // (op 1260) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:31], v[32:39], a[24:27], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1108) 





v_add_u32 v18, 70528, v0 // (op 1263) 
ds_read_u8 v20, v18 // (op 1263) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[24:31], v[40:47], a[28:31], v19, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1264) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1252) 
s_mov_b32 s11, s6 // (op 1252) convert

s_cmp_lt_u32 s10, s11 // (op 1252) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoopTail_1252 // (op 1252) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopTop_KLoopTail_1252 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ForLoopBottom_KLoopTail_1252:






















s_barrier  // (op 4184) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalFalse_0_beta_125
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
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalFalse_0_beta_125:


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
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
.LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_end:
.size RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2, .LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2_end-RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
  .amdhsa_next_free_vgpr 148
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 71680 // lds bytes
  .amdhsa_accum_offset 116
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
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
    macrotile: [16, 256, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2
    .symbol: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 71680
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 116
    .agpr_count: 32
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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

