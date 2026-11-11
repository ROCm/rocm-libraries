// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2,@function
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 3739) 
v_and_b32 v3, 15, v1 // (op 3739) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 3739) 


v_and_b32 v0, 48, v1 // (op 3739) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 3739) 


v_lshrrev_b32 v0, 7, v1 // (op 3912) 
v_and_b32 v4, 15, v1 // (op 3912) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 3912) 


v_and_b32 v0, 48, v1 // (op 3912) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 3912) 


v_lshrrev_b32 v0, 7, v1 // (op 3683) 
v_and_b32 v5, 63, v1 // (op 3683) 
v_lshrrev_b32 v6, 4, v5 // (op 3683) 
v_lshlrev_b32 v7, 2, v6 // (op 3683) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3683) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 3683) 

v_lshrrev_b32 v6, 2, v5 // (op 3683) 

v_and_b32 v5, 3, v6 // (op 3683) 

v_and_b32 v6, 3, v1 // (op 3683) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3683) 


v_lshrrev_b32 v5, 6, v1 // (op 3683) 
v_and_b32 v6, 1, v5 // (op 3683) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3683) 


v_lshl_add_u32 v6, s17, 8, v5 // (op 3683) 

v_mul_lo_u32 v5, v6, s2 // (op 3683) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 3683) 


v_lshrrev_b32 v0, 6, v1 // (op 3797) 
v_and_b32 v5, 1, v0 // (op 3797) 

v_and_b32 v0, 15, v1 // (op 3797) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 3797) 


v_and_b32 v0, 63, v1 // (op 3797) 
v_lshrrev_b32 v5, 4, v0 // (op 3797) 

v_lshlrev_b32 v0, 5, v5 // (op 3797) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 3797) 


v_mul_lo_u32 v0, v5, 6 // (op 3797) 

v_lshrrev_b32 v5, 3, v0 // (op 3797) 

v_lshrrev_b32 v0, 6, v1 // (op 3855) 
v_and_b32 v7, 1, v0 // (op 3855) 

v_and_b32 v0, 15, v1 // (op 3855) 
s_mul_i32 s1, s2, s9 // (op 3685) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3685) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3685) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3685) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3685) most significant: sum
s_add_u32 s1, s1, s19 // (op 3685) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3685) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 3855) 


v_and_b32 v0, 63, v1 // (op 3855) 
v_lshrrev_b32 v7, 4, v0 // (op 3855) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 3855) 


v_lshrrev_b32 v7, 7, v1 // (op 3871) 
v_and_b32 v8, 63, v1 // (op 3871) 
v_lshrrev_b32 v9, 4, v8 // (op 3871) 
v_lshlrev_b32 v10, 2, v9 // (op 3871) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3871) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 3871) 

v_lshrrev_b32 v9, 2, v8 // (op 3871) 

v_and_b32 v8, 3, v9 // (op 3871) 

v_and_b32 v9, 3, v1 // (op 3871) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3871) 


v_lshrrev_b32 v8, 6, v1 // (op 3871) 
v_and_b32 v9, 1, v8 // (op 3871) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3871) 


v_lshl_add_u32 v9, s17, 8, v8 // (op 3871) 

v_mul_lo_u32 v8, v9, s4 // (op 3871) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 3871) 


v_lshrrev_b32 v7, 6, v1 // (op 3941) 
v_and_b32 v8, 1, v7 // (op 3941) 

v_and_b32 v7, 15, v1 // (op 3941) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 3941) 


v_and_b32 v7, 63, v1 // (op 3941) 
v_lshrrev_b32 v8, 4, v7 // (op 3941) 

v_lshlrev_b32 v7, 5, v8 // (op 3941) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 3941) 


v_mul_lo_u32 v7, v8, 6 // (op 3941) 

v_lshrrev_b32 v8, 3, v7 // (op 3941) 

v_lshrrev_b32 v7, 6, v1 // (op 3970) 
v_and_b32 v10, 1, v7 // (op 3970) 

v_and_b32 v7, 15, v1 // (op 3970) 
s_mul_i32 s1, s4, s9 // (op 3873) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 3970) 


v_and_b32 v7, 63, v1 // (op 3970) 
v_lshrrev_b32 v10, 4, v7 // (op 3970) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 3970) 


s_mul_i32 s38, s5, s8 // (op 3873) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3873) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3873) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3873) most significant: sum
s_add_u32 s1, s1, s39 // (op 3873) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 3873) 

s_mov_b32 s40, s6 // (op 3685) reg expression
s_mov_b32 s41, s7 // (op 3685) reg expression
s_mov_b32 s42, s18 // (op 3685) reg expression
s_mov_b32 s43, 131072 // (op 3685) call()

s_mov_b32 s44, s10 // (op 3873) reg expression

s_mov_b32 s45, s11 // (op 3873) reg expression
s_mov_b32 s46, s38 // (op 3873) reg expression
s_mov_b32 s47, 131072 // (op 3873) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 306) call()
v_accvgpr_write a5, 0.00000 // (op 306) call()
v_accvgpr_write a6, 0.00000 // (op 306) call()
v_accvgpr_write a7, 0.00000 // (op 306) call()
v_accvgpr_write a8, 0.00000 // (op 462) call()
v_accvgpr_write a9, 0.00000 // (op 462) call()
v_accvgpr_write a10, 0.00000 // (op 462) call()
v_accvgpr_write a11, 0.00000 // (op 462) call()
v_accvgpr_write a12, 0.00000 // (op 618) call()
v_accvgpr_write a13, 0.00000 // (op 618) call()
v_accvgpr_write a14, 0.00000 // (op 618) call()
v_accvgpr_write a15, 0.00000 // (op 618) call()
v_accvgpr_write a16, 0.00000 // (op 774) call()
v_accvgpr_write a17, 0.00000 // (op 774) call()
v_accvgpr_write a18, 0.00000 // (op 774) call()
v_accvgpr_write a19, 0.00000 // (op 774) call()
v_accvgpr_write a20, 0.00000 // (op 930) call()
v_accvgpr_write a21, 0.00000 // (op 930) call()
v_accvgpr_write a22, 0.00000 // (op 930) call()
v_accvgpr_write a23, 0.00000 // (op 930) call()
v_accvgpr_write a24, 0.00000 // (op 1086) call()
v_accvgpr_write a25, 0.00000 // (op 1086) call()
v_accvgpr_write a26, 0.00000 // (op 1086) call()
v_accvgpr_write a27, 0.00000 // (op 1086) call()
v_accvgpr_write a28, 0.00000 // (op 1242) call()
v_accvgpr_write a29, 0.00000 // (op 1242) call()
v_accvgpr_write a30, 0.00000 // (op 1242) call()
v_accvgpr_write a31, 0.00000 // (op 1242) call()
v_lshrrev_b32 v12, 3, v1 // (op 3450) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 3450) 

v_and_b32 v12, -8, v1 // (op 3450) 
v_sub_u32 v14, v1, v12 // (op 3450) 

v_lshlrev_b32 v12, 4, v14 // (op 3450) 

v_mov_b32 v15, 0 // (op 3450) convert
v_mov_b32 v14, v12 // (op 3450) convert

v_mul_lo_u32 v17, v13, s13 // (op 3450) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3450) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3450) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3450) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3450) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3450) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3452) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3452) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3452) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3452) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3452) most significant: sum
s_add_u32 s1, s1, s7 // (op 3452) most significant: sum


s_mov_b32 s48, s20 // (op 3452) reg expression
s_mov_b32 s49, s21 // (op 3452) reg expression
s_mov_b32 s50, s0 // (op 3452) reg expression
s_mov_b32 s51, 131072 // (op 3452) call()

v_lshrrev_b32 v12, 3, v1 // (op 3469) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 3469) 

v_and_b32 v12, -8, v1 // (op 3469) 
v_sub_u32 v14, v1, v12 // (op 3469) 

v_lshlrev_b32 v12, 4, v14 // (op 3469) 

v_mov_b32 v15, 0 // (op 3469) convert
v_mov_b32 v14, v12 // (op 3469) convert

v_mul_lo_u32 v19, v13, s13 // (op 3469) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3469) most significant: high of low * low
v_mul_lo_u32 v18, v13, s12 // (op 3469) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3469) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 3469) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 3469) most significant half



s_mul_i32 s1, s12, s23 // (op 3471) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3471) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3471) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3471) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3471) most significant: sum
s_add_u32 s1, s1, s7 // (op 3471) most significant: sum


s_mov_b32 s48, s20 // (op 3471) reg expression
s_mov_b32 s49, s21 // (op 3471) reg expression
s_mov_b32 s50, s0 // (op 3471) reg expression
s_mov_b32 s51, 131072 // (op 3471) call()

v_and_b32 v12, -2, v1 // (op 3514) 
v_sub_u32 v13, v1, v12 // (op 3514) 

v_lshlrev_b32 v12, 6, v13 // (op 3514) 

v_mov_b32 v15, 0 // (op 3514) convert
v_mov_b32 v14, v12 // (op 3514) convert

v_and_b32 v12, 4294967294, v1 // (op 3514) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3514) 

v_mul_lo_u32 v21, v13, s25 // (op 3514) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3514) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 3514) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 3514) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 3514) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 3514) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 3514) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 3514) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 3514) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 3514) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 3514) 

v_mov_b32 v14, v12 // (op 3514) convert

s_mul_i32 s1, s9, 6 // (op 3516) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 3516) least significant
s_mul_hi_u32 s6, s8, 6 // (op 3516) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3516) most significant: sum

s_mul_i32 s7, s24, s1 // (op 3516) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 3516) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 3516) least significant
s_mul_hi_u32 s11, s24, s0 // (op 3516) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3516) most significant: sum
s_add_u32 s7, s7, s11 // (op 3516) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3516) 

s_mov_b32 s52, s26 // (op 3516) reg expression
s_mov_b32 s53, s27 // (op 3516) reg expression
s_mov_b32 s54, s0 // (op 3516) reg expression
s_mov_b32 s55, 131072 // (op 3516) call()

v_and_b32 v15, -2, v1 // (op 3533) 
v_sub_u32 v12, v1, v15 // (op 3533) 

v_lshlrev_b32 v13, 6, v12 // (op 3533) 

v_mov_b32 v21, 0 // (op 3533) convert
v_mov_b32 v20, v13 // (op 3533) convert

v_and_b32 v15, 4294967294, v1 // (op 3533) 
v_lshl_add_u32 v12, s17, 8, v15 // (op 3533) 

v_mul_lo_u32 v23, v12, s25 // (op 3533) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 3533) most significant: high of low * low
v_mul_lo_u32 v22, v12, s24 // (op 3533) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 3533) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v22 // (op 3533) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 3533) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 3533) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 3533) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 3533) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 3533) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 3533) 

v_mov_b32 v15, v12 // (op 3533) convert

s_mul_i32 s1, s9, 6 // (op 3535) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 3535) least significant
s_mul_hi_u32 s6, s8, 6 // (op 3535) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3535) most significant: sum

s_mul_i32 s7, s24, s1 // (op 3535) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 3535) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 3535) least significant
s_mul_hi_u32 s11, s24, s0 // (op 3535) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3535) most significant: sum
s_add_u32 s7, s7, s11 // (op 3535) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3535) 

s_mov_b32 s52, s26 // (op 3535) reg expression
s_mov_b32 s53, s27 // (op 3535) reg expression
s_mov_b32 s54, s0 // (op 3535) reg expression
s_mov_b32 s55, 131072 // (op 3535) call()

v_lshrrev_b32 v12, 7, v1 // (op 3578) 
v_and_b32 v13, 15, v1 // (op 3578) 
v_lshl_add_u32 v20, v12, 4, v13 // (op 3578) 


v_lshl_add_u32 v12, s16, 5, v20 // (op 3578) 

v_and_b32 v13, 63, v1 // (op 3578) 
v_lshrrev_b32 v20, 4, v13 // (op 3578) 

v_mul_lo_u32 v13, v12, s28 // (op 3578) 
v_add_u32 v13, v13, v20 // (op 3578) 


s_mul_i32 s1, s28, s23 // (op 3580) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3580) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3580) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3580) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3580) most significant: sum
s_add_u32 s1, s1, s7 // (op 3580) most significant: sum


s_mov_b32 s56, s30 // (op 3580) reg expression
s_mov_b32 s57, s31 // (op 3580) reg expression
s_mov_b32 s58, s0 // (op 3580) reg expression
s_mov_b32 s59, 131072 // (op 3580) call()

v_lshrrev_b32 v12, 7, v1 // (op 3597) 
v_and_b32 v20, 15, v1 // (op 3597) 
v_lshl_add_u32 v21, v12, 4, v20 // (op 3597) 


v_lshl_add_u32 v12, s16, 5, v21 // (op 3597) 

v_and_b32 v20, 63, v1 // (op 3597) 
v_lshrrev_b32 v21, 4, v20 // (op 3597) 

v_mul_lo_u32 v20, v12, s28 // (op 3597) 
v_add_u32 v20, v20, v21 // (op 3597) 


s_mul_i32 s1, s28, s23 // (op 3599) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3599) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3599) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3599) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3599) most significant: sum
s_add_u32 s1, s1, s7 // (op 3599) most significant: sum


s_mov_b32 s56, s30 // (op 3599) reg expression
s_mov_b32 s57, s31 // (op 3599) reg expression
s_mov_b32 s58, s0 // (op 3599) reg expression
s_mov_b32 s59, 131072 // (op 3599) call()

v_lshl_add_u32 v12, s17, 8, v1 // (op 3642) 
v_mul_lo_u32 v21, v12, s32 // (op 3642) 

s_mul_i32 s1, s32, s9 // (op 3644) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3644) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3644) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3644) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3644) most significant: sum
s_add_u32 s1, s1, s7 // (op 3644) most significant: sum


s_mov_b32 s60, s34 // (op 3644) reg expression
s_mov_b32 s61, s35 // (op 3644) reg expression
s_mov_b32 s62, s0 // (op 3644) reg expression
s_mov_b32 s63, 131072 // (op 3644) call()

v_lshl_add_u32 v12, s17, 8, v1 // (op 3661) 
v_mul_lo_u32 v22, v12, s32 // (op 3661) 

s_mul_i32 s1, s32, s9 // (op 3663) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3663) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3663) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3663) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3663) most significant: sum
s_add_u32 s1, s1, s7 // (op 3663) most significant: sum


s_mov_b32 s60, s34 // (op 3663) reg expression
s_mov_b32 s61, s35 // (op 3663) reg expression
s_mov_b32 s62, s0 // (op 3663) reg expression
s_mov_b32 s63, 131072 // (op 3663) call()




buffer_load_ubyte v12, v13, s[56:59], 0 offen offset: 0 // (op 3108) Load value

s_mul_i32 s1, s25, 6 // (op 3112) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 3112) least significant
s_mul_hi_u32 s6, s24, 6 // (op 3112) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3112) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3112) 


v_mov_b32 v23, v14 // (op 3112) 
buffer_load_dwordx4 v[26:29], v23, s[52:55], 0 offen // (op 3112) Load value
buffer_load_dwordx2 v[30:31], v23, s[52:55], 0 offen offset:16 // (op 3112) Load value
buffer_load_dwordx4 v[32:35], v23, s[52:55], 0 offen offset:24 // (op 3112) Load value
buffer_load_dwordx2 v[36:37], v23, s[52:55], 0 offen offset:40 // (op 3112) Load value
v_add_u32 v23, v23, v24 // (op 3112) 
buffer_load_dwordx4 v[38:41], v23, s[52:55], 0 offen // (op 3112) Load value
buffer_load_dwordx2 v[42:43], v23, s[52:55], 0 offen offset:16 // (op 3112) Load value
buffer_load_dwordx4 v[44:47], v23, s[52:55], 0 offen offset:24 // (op 3112) Load value
buffer_load_dwordx2 v[48:49], v23, s[52:55], 0 offen offset:40 // (op 3112) Load value





buffer_load_dword v23, v21, s[60:63], 0 offen // (op 3115) Load value
v_and_b32 v24, -8, v1 // (op 3727) 
v_sub_u32 v25, v1, v24 // (op 3727) 

v_lshlrev_b32 v24, 4, v25 // (op 3727) 

v_lshrrev_b32 v25, 3, v1 // (op 3727) 
v_lshl_add_u32 v50, v25, 7, v24 // (op 3727) 


s_nop 0
v_readfirstlane_b32 s0, v50 // (op 3727) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s0 // (op 3399) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 3399) Load value direct to lds




buffer_load_ubyte v24, v13, s[56:59], 0 offen offset:4 // (op 3122) Load value


s_mul_i32 s1, s25, 6 // (op 3126) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 3126) least significant
s_mul_hi_u32 s6, s24, 6 // (op 3126) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3126) most significant: sum

v_lshrrev_b64 v[50:51], 3, s[0:1] // (op 3126) 


v_mov_b32 v13, v14 // (op 3126) 
buffer_load_dwordx4 v[52:55], v13, s[52:55], 0 offen offset:96 // (op 3126) Load value
buffer_load_dwordx2 v[56:57], v13, s[52:55], 0 offen offset:112 // (op 3126) Load value
buffer_load_dwordx4 v[58:61], v13, s[52:55], 0 offen offset:120 // (op 3126) Load value
buffer_load_dwordx2 v[62:63], v13, s[52:55], 0 offen offset:136 // (op 3126) Load value
v_add_u32 v13, v13, v50 // (op 3126) 
buffer_load_dwordx4 v[64:67], v13, s[52:55], 0 offen offset:96 // (op 3126) Load value
buffer_load_dwordx2 v[68:69], v13, s[52:55], 0 offen offset:112 // (op 3126) Load value
buffer_load_dwordx4 v[70:73], v13, s[52:55], 0 offen offset:120 // (op 3126) Load value
buffer_load_dwordx2 v[74:75], v13, s[52:55], 0 offen offset:136 // (op 3126) Load value






buffer_load_dword v25, v21, s[60:63], 0 offen offset:4 // (op 3129) Load value

v_and_b32 v21, -8, v1 // (op 3900) 
v_sub_u32 v13, v1, v21 // (op 3900) 

v_lshlrev_b32 v14, 4, v13 // (op 3900) 

v_lshrrev_b32 v13, 3, v1 // (op 3900) 
v_lshl_add_u32 v21, v13, 7, v14 // (op 3900) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 3900) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 4096, s0 // (op 3404) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 3404) reg expression
s_mov_b32 s6, 128 // (op 3404) call()
buffer_load_dwordx4 v16, s[48:51], s6 offen offset: 0 lds // (op 3404) Load value direct to lds




v_and_b32 v21, -2, v1 // (op 3780) 
v_sub_u32 v13, v1, v21 // (op 3780) 

v_lshlrev_b32 v14, 6, v13 // (op 3780) 

v_lshrrev_b32 v13, 1, v1 // (op 3780) 
v_lshl_add_u32 v21, v13, 8, v14 // (op 3780) 


v_mul_lo_u32 v13, v21, 6 // (op 3780) 

v_lshrrev_b32 v14, 3, v13 // (op 3780) 




s_waitcnt vmcnt(20)
ds_write_b128 v14, v[26:29] offset:8192 // (op 3134) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v14, v[30:31] offset:8208 // (op 3134) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v14, v[32:35] offset:8216 // (op 3134) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v14, v[36:37] offset:8232 // (op 3134) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v14, v[38:41] offset:8288 // (op 3134) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v14, v[42:43] offset:8304 // (op 3134) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v14, v[44:47] offset:8312 // (op 3134) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v14, v[48:49] offset:8328 // (op 3134) Store local data 

v_lshlrev_b32 v21, 2, v1 // (op 3838) 



s_waitcnt vmcnt(12)
ds_write_b32 v21, v23 offset:32768 // (op 3135) Store local data 

v_and_b32 v21, -8, v1 // (op 3699) 
v_sub_u32 v13, v1, v21 // (op 3699) 

v_lshlrev_b32 v14, 4, v13 // (op 3699) 

v_lshrrev_b32 v13, 3, v1 // (op 3699) 
v_lshl_add_u32 v21, v13, 7, v14 // (op 3699) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 3699) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -2, v1 // (op 3751) 
v_sub_u32 v13, v1, v21 // (op 3751) 

v_lshlrev_b32 v14, 6, v13 // (op 3751) 

v_lshrrev_b32 v13, 1, v1 // (op 3751) 
v_lshl_add_u32 v21, v13, 8, v14 // (op 3751) 


v_mul_lo_u32 v13, v21, 6 // (op 3751) 

v_lshrrev_b32 v14, 3, v13 // (op 3751) 

v_lshlrev_b32 v13, 2, v1 // (op 3809) 
v_and_b32 v21, -8, v1 // (op 3885) 
v_sub_u32 v16, v1, v21 // (op 3885) 

v_lshlrev_b32 v17, 4, v16 // (op 3885) 

v_lshrrev_b32 v16, 3, v1 // (op 3885) 
v_lshl_add_u32 v21, v16, 7, v17 // (op 3885) 


s_nop 0
v_readfirstlane_b32 s1, v21 // (op 3885) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -2, v1 // (op 3924) 
v_sub_u32 v16, v1, v21 // (op 3924) 

v_lshlrev_b32 v17, 6, v16 // (op 3924) 

v_lshrrev_b32 v16, 1, v1 // (op 3924) 
v_lshl_add_u32 v21, v16, 8, v17 // (op 3924) 


v_mul_lo_u32 v16, v21, 6 // (op 3924) 

v_lshrrev_b32 v17, 3, v16 // (op 3924) 

v_lshlrev_b32 v16, 2, v1 // (op 3953) 
s_mov_b32 s6, 0 // (op 1244) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 1243) 
s_mov_b32 s7, s10 // (op 1243) convert

s_and_b32 s10, s7, 4294967294 // (op 1243) 

s_cmp_lt_u32 s6, s10 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1243 // (op 1243) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1243 if false)
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1243:
s_add_u32 s6, s6, 2 // (op 1245) 
s_waitcnt vmcnt(11)
s_barrier  // (op 4029) 



ds_read_b128 v[76:79], v3 // (op 65) Load local data 
ds_read_b128 v[80:83], v3 offset:64 // (op 65) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4042) 



ds_read_b128 v[84:87], v5 offset:8192 // (op 70) Load local data 
ds_read_b64 v[88:89], v5 offset:8208 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4056) 



ds_read_u8 v21, v0 offset:32768 // (op 79) Load local data 



ds_read_b128 v[90:93], v5 offset:11264 // (op 339) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:83], v[84:89], a[0:3], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 82) 
ds_read_b64 v[94:95], v5 offset:11280 // (op 339) Load local data 





ds_read_u8 v21, v0 offset:32896 // (op 342) Load local data 



ds_read_b128 v[84:87], v5 offset:14336 // (op 495) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:83], v[90:95], a[4:7], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 343) 
ds_read_b64 v[88:89], v5 offset:14352 // (op 495) Load local data 





ds_read_u8 v21, v0 offset:33024 // (op 498) Load local data 



ds_read_b128 v[90:93], v5 offset:17408 // (op 651) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:83], v[84:89], a[8:11], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 499) 
ds_read_b64 v[94:95], v5 offset:17424 // (op 651) Load local data 





ds_read_u8 v21, v0 offset:33152 // (op 654) Load local data 



ds_read_b128 v[84:87], v5 offset:20480 // (op 807) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:83], v[90:95], a[12:15], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 655) 
ds_read_b64 v[88:89], v5 offset:20496 // (op 807) Load local data 





ds_read_u8 v21, v0 offset:33280 // (op 810) Load local data 



ds_read_b128 v[90:93], v5 offset:23552 // (op 963) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:83], v[84:89], a[16:19], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 811) 
ds_read_b64 v[94:95], v5 offset:23568 // (op 963) Load local data 





ds_read_u8 v21, v0 offset:33408 // (op 966) Load local data 



ds_read_b128 v[84:87], v5 offset:26624 // (op 1119) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[90:95], a[20:23], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 967) 
ds_read_b64 v[88:89], v5 offset:26640 // (op 1119) Load local data 





ds_read_u8 v21, v0 offset:33536 // (op 1122) Load local data 



ds_read_b128 v[90:93], v5 offset:29696 // (op 1275) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[84:89], a[24:27], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1123) 
ds_read_b64 v[94:95], v5 offset:29712 // (op 1275) Load local data 





ds_read_u8 v21, v0 offset:33664 // (op 1278) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[90:95], a[28:31], v12, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1279) 





v_mov_b32 v21, v20 // (op 75) 

buffer_load_ubyte v12, v21, s[56:59], 0 offen offset:8 // (op 75) Load value

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
s_barrier  // (op 4032) 


v_mov_b64 v[78:79], v[18:19] // (op 3383) 

s_mov_b32 m0, s0 // (op 3383) reg expression
s_mov_b32 s7, 256 // (op 3383) call()
buffer_load_dwordx4 v78, s[48:51], s7 offen offset: 0 lds // (op 3383) Load value direct to lds

s_barrier  // (op 4079) 



s_waitcnt vmcnt(20)
ds_write_b128 v17, v[52:55] offset:33792 // (op 235) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v17, v[56:57] offset:33808 // (op 235) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v17, v[58:61] offset:33816 // (op 235) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v17, v[62:63] offset:33832 // (op 235) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v17, v[64:67] offset:33888 // (op 235) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v17, v[68:69] offset:33904 // (op 235) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v17, v[70:73] offset:33912 // (op 235) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v17, v[74:75] offset:33928 // (op 235) Store local data 
s_barrier  // (op 4085) 



s_waitcnt vmcnt(12)
ds_write_b32 v16, v25 offset:58368 // (op 236) Store local data 
s_waitcnt vmcnt(11)
s_barrier  // (op 4070) 



ds_read_b128 v[80:83], v4 offset:4096 // (op 227) Load local data 
ds_read_b128 v[84:87], v4 offset:4160 // (op 227) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4076) 



ds_read_b128 v[88:91], v8 offset:33792 // (op 229) Load local data 
ds_read_b64 v[92:93], v8 offset:33808 // (op 229) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4082) 



ds_read_u8 v51, v7 offset:58368 // (op 233) Load local data 



ds_read_b128 v[94:97], v8 offset:36864 // (op 348) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:87], v[88:93], a[0:3], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 234) 
ds_read_b64 v[98:99], v8 offset:36880 // (op 348) Load local data 





ds_read_u8 v51, v7 offset:58496 // (op 351) Load local data 



ds_read_b128 v[88:91], v8 offset:39936 // (op 504) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:87], v[94:99], a[4:7], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 344) 
ds_read_b64 v[92:93], v8 offset:39952 // (op 504) Load local data 





ds_read_u8 v51, v7 offset:58624 // (op 507) Load local data 



ds_read_b128 v[94:97], v8 offset:43008 // (op 660) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[80:87], v[88:93], a[8:11], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 500) 
ds_read_b64 v[98:99], v8 offset:43024 // (op 660) Load local data 





ds_read_u8 v51, v7 offset:58752 // (op 663) Load local data 



ds_read_b128 v[88:91], v8 offset:46080 // (op 816) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[80:87], v[94:99], a[12:15], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 656) 
ds_read_b64 v[92:93], v8 offset:46096 // (op 816) Load local data 





ds_read_u8 v51, v7 offset:58880 // (op 819) Load local data 



ds_read_b128 v[94:97], v8 offset:49152 // (op 972) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:87], v[88:93], a[16:19], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 812) 
ds_read_b64 v[98:99], v8 offset:49168 // (op 972) Load local data 





ds_read_u8 v51, v7 offset:59008 // (op 975) Load local data 



ds_read_b128 v[88:91], v8 offset:52224 // (op 1128) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:87], v[94:99], a[20:23], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 968) 
ds_read_b64 v[92:93], v8 offset:52240 // (op 1128) Load local data 





ds_read_u8 v51, v7 offset:59136 // (op 1131) Load local data 



ds_read_b128 v[94:97], v8 offset:55296 // (op 1284) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:87], v[88:93], a[24:27], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1124) 
ds_read_b64 v[98:99], v8 offset:55312 // (op 1284) Load local data 





s_barrier  // (op 4182) 
ds_read_u8 v51, v7 offset:59264 // (op 1287) Load local data 
s_barrier  // (op 4186) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:87], v[94:99], a[28:31], v24, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1280) 






buffer_load_ubyte v24, v21, s[56:59], 0 offen offset:12 // (op 231) Load value


v_add_u32 v20, 8, v20 // (op 3589) 
s_mul_i32 s11, s25, 6 // (op 237) most significant: low of high * low
s_mul_i32 s10, s24, 6 // (op 237) least significant
s_mul_hi_u32 s7, s24, 6 // (op 237) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 237) most significant: sum

v_lshrrev_b64 v[80:81], 3, s[10:11] // (op 237) 


v_mov_b32 v21, v76 // (op 237) 
buffer_load_dwordx4 v[52:55], v21, s[52:55], 0 offen offset:288 // (op 237) Load value
buffer_load_dwordx2 v[56:57], v21, s[52:55], 0 offen offset:304 // (op 237) Load value
buffer_load_dwordx4 v[58:61], v21, s[52:55], 0 offen offset:312 // (op 237) Load value
buffer_load_dwordx2 v[62:63], v21, s[52:55], 0 offen offset:328 // (op 237) Load value
v_add_u32 v21, v21, v80 // (op 237) 
buffer_load_dwordx4 v[64:67], v21, s[52:55], 0 offen offset:288 // (op 237) Load value
buffer_load_dwordx2 v[68:69], v21, s[52:55], 0 offen offset:304 // (op 237) Load value
buffer_load_dwordx4 v[70:73], v21, s[52:55], 0 offen offset:312 // (op 237) Load value
buffer_load_dwordx2 v[74:75], v21, s[52:55], 0 offen offset:328 // (op 237) Load value






v_add_u32 v15, 192, v15 // (op 3525) 
buffer_load_dword v25, v50, s[60:63], 0 offen offset:12 // (op 238) Load value

s_barrier  // (op 4073) 
v_add_u32 v22, 8, v22 // (op 3653) 


s_add_u32 s7, 4096, s1 // (op 3394) 

s_mov_b32 m0, s7 // (op 3394) reg expression
s_mov_b32 s10, 384 // (op 3394) call()
buffer_load_dwordx4 v78, s[48:51], s10 offen offset: 0 lds // (op 3394) Load value direct to lds



s_barrier  // (op 4045) 
v_add_co_u32 v18, vcc, 256, v18 // (op 3461) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3461) most significant half
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
s_barrier  // (op 4059) 



s_waitcnt vmcnt(12)
ds_write_b32 v13, v23 offset:32768 // (op 26) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 1243) 
s_mov_b32 s7, s10 // (op 1243) convert

s_and_b32 s10, s7, 4294967294 // (op 1243) 

s_cmp_lt_u32 s6, s10 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1243 // (op 1243) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1243 if true)
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1243:














s_barrier  // (op 4166) 
v_lshrrev_b32 v12, 3, v1 // (op 3417) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3417) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 3417) 

s_lshr_b32 s6, s0, 1 // (op 3417) 

v_and_b32 v12, -8, v1 // (op 3417) 
v_sub_u32 v14, v1, v12 // (op 3417) 

v_lshlrev_b32 v12, 4, v14 // (op 3417) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 3417) 


v_mov_b32 v23, 0 // (op 3417) convert
v_mov_b32 v22, v14 // (op 3417) convert

v_mul_lo_u32 v15, v13, s13 // (op 3417) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3417) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 3417) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 3417) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v22 // (op 3417) least significant half
v_addc_co_u32 v15, s[0:1], v15, v23, s[0:1] // (op 3417) most significant half



s_mul_i32 s1, s12, s23 // (op 3419) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3419) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3419) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3419) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3419) most significant: sum
s_add_u32 s1, s1, s7 // (op 3419) most significant: sum


s_mov_b32 s48, s20 // (op 3419) reg expression
s_mov_b32 s49, s21 // (op 3419) reg expression
s_mov_b32 s50, s0 // (op 3419) reg expression
s_mov_b32 s51, 131072 // (op 3419) call()

v_and_b32 v12, -2, v1 // (op 3491) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3491) 
v_sub_u32 v13, v1, v12 // (op 3491) 

s_lshr_b32 s6, s0, 1 // (op 3491) 

v_lshlrev_b32 v12, 6, v13 // (op 3491) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 3491) 


v_mov_b32 v17, 0 // (op 3491) convert
v_mov_b32 v16, v13 // (op 3491) convert

v_and_b32 v12, 4294967294, v1 // (op 3491) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3491) 

v_mul_lo_u32 v19, v13, s25 // (op 3491) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3491) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 3491) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3491) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 3491) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 3491) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 3491) most significant: low of high * low
v_mul_hi_u32 v18, v12, 6 // (op 3491) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 3491) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 3491) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 3491) 

v_mov_b32 v16, v12 // (op 3491) convert

s_mul_i32 s1, s9, 6 // (op 3493) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 3493) least significant
s_mul_hi_u32 s6, s8, 6 // (op 3493) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3493) most significant: sum

s_mul_i32 s7, s24, s1 // (op 3493) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 3493) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 3493) least significant
s_mul_hi_u32 s11, s24, s0 // (op 3493) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3493) most significant: sum
s_add_u32 s7, s7, s11 // (op 3493) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3493) 

s_mov_b32 s52, s26 // (op 3493) reg expression
s_mov_b32 s53, s27 // (op 3493) reg expression
s_mov_b32 s54, s0 // (op 3493) reg expression
s_mov_b32 s55, 131072 // (op 3493) call()

v_lshrrev_b32 v12, 7, v1 // (op 3555) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 3555) 
v_and_b32 v13, 15, v1 // (op 3555) 
s_lshr_b32 s6, s0, 1 // (op 3555) 

v_lshl_add_u32 v17, v12, 4, v13 // (op 3555) 


v_lshl_add_u32 v12, s16, 5, v17 // (op 3555) 

v_and_b32 v13, 63, v1 // (op 3555) 
v_lshrrev_b32 v17, 4, v13 // (op 3555) 

v_lshl_add_u32 v13, s6, 3, v17 // (op 3555) 


v_mul_lo_u32 v17, v12, s28 // (op 3555) 
v_add_u32 v17, v17, v13 // (op 3555) 


s_mul_i32 s1, s28, s23 // (op 3557) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 3557) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 3557) least significant
s_mul_hi_u32 s7, s28, s22 // (op 3557) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3557) most significant: sum
s_add_u32 s1, s1, s7 // (op 3557) most significant: sum


s_mov_b32 s56, s30 // (op 3557) reg expression
s_mov_b32 s57, s31 // (op 3557) reg expression
s_mov_b32 s58, s0 // (op 3557) reg expression
s_mov_b32 s59, 131072 // (op 3557) call()

v_lshl_add_u32 v12, s17, 8, v1 // (op 3619) 
v_mul_lo_u32 v13, v12, s32 // (op 3619) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 3619) 
s_lshr_b32 s6, s0, 1 // (op 3619) 

v_lshl_add_u32 v12, s6, 3, v13 // (op 3619) 


s_mul_i32 s1, s32, s9 // (op 3621) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 3621) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 3621) least significant
s_mul_hi_u32 s7, s32, s8 // (op 3621) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3621) most significant: sum
s_add_u32 s1, s1, s7 // (op 3621) most significant: sum


s_mov_b32 s60, s34 // (op 3621) reg expression
s_mov_b32 s61, s35 // (op 3621) reg expression
s_mov_b32 s62, s0 // (op 3621) reg expression
s_mov_b32 s63, 131072 // (op 3621) call()

v_and_b32 v13, -8, v1 // (op 3712) 
v_sub_u32 v18, v1, v13 // (op 3712) 

v_lshlrev_b32 v13, 4, v18 // (op 3712) 

v_lshrrev_b32 v18, 3, v1 // (op 3712) 
v_lshl_add_u32 v19, v18, 7, v13 // (op 3712) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 3712) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -2, v1 // (op 3764) 
v_sub_u32 v18, v1, v13 // (op 3764) 

v_lshlrev_b32 v13, 6, v18 // (op 3764) 

v_lshrrev_b32 v18, 1, v1 // (op 3764) 
v_lshl_add_u32 v19, v18, 8, v13 // (op 3764) 


v_mul_lo_u32 v13, v19, 6 // (op 3764) 

v_lshrrev_b32 v18, 3, v13 // (op 3764) 

v_lshlrev_b32 v13, 2, v1 // (op 3822) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1249) 
s_and_b32 s1, s6, 4294967294 // (op 1249) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 1248) 
s_mov_b32 s10, s6 // (op 1248) convert

s_cmp_lt_u32 s1, s10 // (op 1248) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1248 // (op 1248) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1248 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1248:
s_barrier  // (op 4035) 
s_add_u32 s1, s1, 1 // (op 1250) 


v_mov_b64 v[20:21], v[14:15] // (op 3388) 

s_mov_b32 m0, s0 // (op 3388) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 3388) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 3407) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3407) most significant half

s_mul_i32 s7, s25, 6 // (op 179) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 179) least significant
s_mul_hi_u32 s10, s24, 6 // (op 179) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 179) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[6:7] // (op 179) 


v_mov_b32 v19, v16 // (op 179) 
v_mov_b32 v22, v19 // (op 179) 
buffer_load_dwordx4 v[24:27], v22, s[52:55], 0 offen // (op 179) Load value
buffer_load_dwordx2 v[28:29], v22, s[52:55], 0 offen offset:16 // (op 179) Load value
buffer_load_dwordx4 v[30:33], v22, s[52:55], 0 offen offset:24 // (op 179) Load value
buffer_load_dwordx2 v[34:35], v22, s[52:55], 0 offen offset:40 // (op 179) Load value
v_add_u32 v22, v22, v20 // (op 179) 
buffer_load_dwordx4 v[36:39], v22, s[52:55], 0 offen // (op 179) Load value
buffer_load_dwordx2 v[40:41], v22, s[52:55], 0 offen offset:16 // (op 179) Load value
buffer_load_dwordx4 v[42:45], v22, s[52:55], 0 offen offset:24 // (op 179) Load value
buffer_load_dwordx2 v[46:47], v22, s[52:55], 0 offen offset:40 // (op 179) Load value



v_add_u32 v16, 96, v16 // (op 3483) 


s_barrier  // (op 4048) 
v_mov_b32 v19, v12 // (op 180) 




s_waitcnt vmcnt(7)
ds_write_b128 v18, v[24:27] offset:8192 // (op 177) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v18, v[28:29] offset:8208 // (op 177) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v18, v[30:33] offset:8216 // (op 177) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v18, v[34:35] offset:8232 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v18, v[36:39] offset:8288 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v18, v[40:41] offset:8304 // (op 177) Store local data 
buffer_load_dword v20, v19, s[60:63], 0 offen // (op 180) Load value

v_add_u32 v12, 4, v12 // (op 3611) 
s_waitcnt vmcnt(2)
ds_write_b128 v18, v[42:45] offset:8312 // (op 177) Store local data 
s_barrier  // (op 4038) 
s_waitcnt vmcnt(1)
ds_write_b64 v18, v[46:47] offset:8328 // (op 177) Store local data 




ds_read_b128 v[22:25], v3 // (op 169) Load local data 
s_barrier  // (op 4062) 
ds_read_b128 v[26:29], v3 offset:64 // (op 169) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 4052) 
s_waitcnt vmcnt(0)
ds_write_b32 v13, v20 offset:32768 // (op 178) Store local data 




ds_read_b128 v[30:33], v5 offset:8192 // (op 171) Load local data 
ds_read_b64 v[34:35], v5 offset:8208 // (op 171) Load local data 


v_mov_b32 v19, v17 // (op 173) 

buffer_load_ubyte v20, v19, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v17, 4, v17 // (op 3547) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4066) 



ds_read_u8 v19, v0 offset:32768 // (op 175) Load local data 



ds_read_b128 v[36:39], v5 offset:11264 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 176) 
ds_read_b64 v[40:41], v5 offset:11280 // (op 320) Load local data 





ds_read_u8 v19, v0 offset:32896 // (op 323) Load local data 



ds_read_b128 v[30:33], v5 offset:14336 // (op 476) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[36:41], a[4:7], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 324) 
ds_read_b64 v[34:35], v5 offset:14352 // (op 476) Load local data 





ds_read_u8 v19, v0 offset:33024 // (op 479) Load local data 



ds_read_b128 v[36:39], v5 offset:17408 // (op 632) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:29], v[30:35], a[8:11], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 480) 
ds_read_b64 v[40:41], v5 offset:17424 // (op 632) Load local data 





ds_read_u8 v19, v0 offset:33152 // (op 635) Load local data 



ds_read_b128 v[30:33], v5 offset:20480 // (op 788) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:29], v[36:41], a[12:15], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 636) 
ds_read_b64 v[34:35], v5 offset:20496 // (op 788) Load local data 





ds_read_u8 v19, v0 offset:33280 // (op 791) Load local data 



ds_read_b128 v[36:39], v5 offset:23552 // (op 944) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:29], v[30:35], a[16:19], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 792) 
ds_read_b64 v[40:41], v5 offset:23568 // (op 944) Load local data 





ds_read_u8 v19, v0 offset:33408 // (op 947) Load local data 



ds_read_b128 v[30:33], v5 offset:26624 // (op 1100) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:29], v[36:41], a[20:23], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 948) 
ds_read_b64 v[34:35], v5 offset:26640 // (op 1100) Load local data 





ds_read_u8 v19, v0 offset:33536 // (op 1103) Load local data 



ds_read_b128 v[36:39], v5 offset:29696 // (op 1256) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:29], v[30:35], a[24:27], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1104) 
ds_read_b64 v[40:41], v5 offset:29712 // (op 1256) Load local data 





ds_read_u8 v19, v0 offset:33664 // (op 1259) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:29], v[36:41], a[28:31], v20, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1260) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1248) 
s_mov_b32 s10, s6 // (op 1248) convert

s_cmp_lt_u32 s1, s10 // (op 1248) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1248 // (op 1248) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1248 if true)
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1248:






















s_barrier  // (op 4170) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 52) Store value


v_accvgpr_read v12, a4 // (op 1572) 
v_accvgpr_read v13, a5 // (op 1572) 
v_accvgpr_read v14, a6 // (op 1572) 
v_accvgpr_read v15, a7 // (op 1572) 
v_mul_f32 v16, v11, v12 // (op 1572) 
v_mul_f32 v17, v11, v13 // (op 1572) 
v_mul_f32 v18, v11, v14 // (op 1572) 
v_mul_f32 v19, v11, v15 // (op 1572) 



s_lshl_b32 s0, s4, 7 // (op 1574) 
v_add_u32 v0, v9, s0 // (op 1574) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1574) Store value


v_accvgpr_read v12, a8 // (op 1584) 
v_accvgpr_read v13, a9 // (op 1584) 
v_accvgpr_read v14, a10 // (op 1584) 
v_accvgpr_read v15, a11 // (op 1584) 
v_mul_f32 v16, v11, v12 // (op 1584) 
v_mul_f32 v17, v11, v13 // (op 1584) 
v_mul_f32 v18, v11, v14 // (op 1584) 
v_mul_f32 v19, v11, v15 // (op 1584) 



s_lshl_b32 s0, s4, 8 // (op 1586) 
v_add_u32 v0, v9, s0 // (op 1586) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1586) Store value


v_accvgpr_read v12, a12 // (op 1596) 
v_accvgpr_read v13, a13 // (op 1596) 
v_accvgpr_read v14, a14 // (op 1596) 
v_accvgpr_read v15, a15 // (op 1596) 
v_mul_f32 v16, v11, v12 // (op 1596) 
v_mul_f32 v17, v11, v13 // (op 1596) 
v_mul_f32 v18, v11, v14 // (op 1596) 
v_mul_f32 v19, v11, v15 // (op 1596) 



s_lshl_b32 s0, s4, 7 // (op 1598) 
s_mul_i32 s1, 3, s0 // (op 1598) 

v_add_u32 v0, v9, s1 // (op 1598) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1598) Store value


v_accvgpr_read v12, a16 // (op 1608) 
v_accvgpr_read v13, a17 // (op 1608) 
v_accvgpr_read v14, a18 // (op 1608) 
v_accvgpr_read v15, a19 // (op 1608) 
v_mul_f32 v16, v11, v12 // (op 1608) 
v_mul_f32 v17, v11, v13 // (op 1608) 
v_mul_f32 v18, v11, v14 // (op 1608) 
v_mul_f32 v19, v11, v15 // (op 1608) 



s_lshl_b32 s0, s4, 9 // (op 1610) 
v_add_u32 v0, v9, s0 // (op 1610) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1610) Store value


v_accvgpr_read v12, a20 // (op 1620) 
v_accvgpr_read v13, a21 // (op 1620) 
v_accvgpr_read v14, a22 // (op 1620) 
v_accvgpr_read v15, a23 // (op 1620) 
v_mul_f32 v16, v11, v12 // (op 1620) 
v_mul_f32 v17, v11, v13 // (op 1620) 
v_mul_f32 v18, v11, v14 // (op 1620) 
v_mul_f32 v19, v11, v15 // (op 1620) 



s_lshl_b32 s0, s4, 7 // (op 1622) 
s_mul_i32 s1, 5, s0 // (op 1622) 

v_add_u32 v0, v9, s1 // (op 1622) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1622) Store value


v_accvgpr_read v12, a24 // (op 1632) 
v_accvgpr_read v13, a25 // (op 1632) 
v_accvgpr_read v14, a26 // (op 1632) 
v_accvgpr_read v15, a27 // (op 1632) 
v_mul_f32 v16, v11, v12 // (op 1632) 
v_mul_f32 v17, v11, v13 // (op 1632) 
v_mul_f32 v18, v11, v14 // (op 1632) 
v_mul_f32 v19, v11, v15 // (op 1632) 



s_lshl_b32 s0, s4, 7 // (op 1634) 
s_mul_i32 s1, 6, s0 // (op 1634) 

v_add_u32 v0, v9, s1 // (op 1634) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1634) Store value


v_accvgpr_read v12, a28 // (op 1644) 
v_accvgpr_read v13, a29 // (op 1644) 
v_accvgpr_read v14, a30 // (op 1644) 
v_accvgpr_read v15, a31 // (op 1644) 
v_mul_f32 v16, v11, v12 // (op 1644) 
v_mul_f32 v17, v11, v13 // (op 1644) 
v_mul_f32 v18, v11, v14 // (op 1644) 
v_mul_f32 v19, v11, v15 // (op 1644) 



s_lshl_b32 s0, s4, 7 // (op 1646) 
s_mul_i32 s1, 7, s0 // (op 1646) 

v_add_u32 v0, v9, s1 // (op 1646) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 1646) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 135) Load value

v_accvgpr_read v16, a0 // (op 132) 
v_accvgpr_read v17, a1 // (op 132) 
v_accvgpr_read v18, a2 // (op 132) 
v_accvgpr_read v19, a3 // (op 132) 
v_mul_f32 v20, v11, v16 // (op 132) 
v_mul_f32 v21, v11, v17 // (op 132) 
v_mul_f32 v22, v11, v18 // (op 132) 
v_mul_f32 v23, v11, v19 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 136) 
v_mul_f32 v17, v10, v13 // (op 136) 
v_mul_f32 v18, v10, v14 // (op 136) 
v_mul_f32 v19, v10, v15 // (op 136) 

v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 
v_add_f32 v14, v18, v22 // (op 133) 
v_add_f32 v15, v19, v23 // (op 133) 




v_mov_b32 v0, v9 // (op 134) reg expression

buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 1698) 
v_add_u32 v0, v6, s0 // (op 1698) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1698) Load value

v_accvgpr_read v16, a4 // (op 1695) 
v_accvgpr_read v17, a5 // (op 1695) 
v_accvgpr_read v18, a6 // (op 1695) 
v_accvgpr_read v19, a7 // (op 1695) 
v_mul_f32 v20, v11, v16 // (op 1695) 
v_mul_f32 v21, v11, v17 // (op 1695) 
v_mul_f32 v22, v11, v18 // (op 1695) 
v_mul_f32 v23, v11, v19 // (op 1695) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1699) 
v_mul_f32 v17, v10, v13 // (op 1699) 
v_mul_f32 v18, v10, v14 // (op 1699) 
v_mul_f32 v19, v10, v15 // (op 1699) 

v_add_f32 v12, v16, v20 // (op 1696) 
v_add_f32 v13, v17, v21 // (op 1696) 
v_add_f32 v14, v18, v22 // (op 1696) 
v_add_f32 v15, v19, v23 // (op 1696) 




s_lshl_b32 s0, s4, 7 // (op 1697) 
v_add_u32 v0, v9, s0 // (op 1697) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1697) Store value




s_lshl_b32 s0, s2, 8 // (op 1710) 
v_add_u32 v0, v6, s0 // (op 1710) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1710) Load value

v_accvgpr_read v16, a8 // (op 1707) 
v_accvgpr_read v17, a9 // (op 1707) 
v_accvgpr_read v18, a10 // (op 1707) 
v_accvgpr_read v19, a11 // (op 1707) 
v_mul_f32 v20, v11, v16 // (op 1707) 
v_mul_f32 v21, v11, v17 // (op 1707) 
v_mul_f32 v22, v11, v18 // (op 1707) 
v_mul_f32 v23, v11, v19 // (op 1707) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1711) 
v_mul_f32 v17, v10, v13 // (op 1711) 
v_mul_f32 v18, v10, v14 // (op 1711) 
v_mul_f32 v19, v10, v15 // (op 1711) 

v_add_f32 v12, v16, v20 // (op 1708) 
v_add_f32 v13, v17, v21 // (op 1708) 
v_add_f32 v14, v18, v22 // (op 1708) 
v_add_f32 v15, v19, v23 // (op 1708) 




s_lshl_b32 s0, s4, 8 // (op 1709) 
v_add_u32 v0, v9, s0 // (op 1709) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1709) Store value




s_lshl_b32 s0, s2, 7 // (op 1722) 
s_mul_i32 s1, 3, s0 // (op 1722) 

v_add_u32 v0, v6, s1 // (op 1722) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1722) Load value

v_accvgpr_read v16, a12 // (op 1719) 
v_accvgpr_read v17, a13 // (op 1719) 
v_accvgpr_read v18, a14 // (op 1719) 
v_accvgpr_read v19, a15 // (op 1719) 
v_mul_f32 v20, v11, v16 // (op 1719) 
v_mul_f32 v21, v11, v17 // (op 1719) 
v_mul_f32 v22, v11, v18 // (op 1719) 
v_mul_f32 v23, v11, v19 // (op 1719) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1723) 
v_mul_f32 v17, v10, v13 // (op 1723) 
v_mul_f32 v18, v10, v14 // (op 1723) 
v_mul_f32 v19, v10, v15 // (op 1723) 

v_add_f32 v12, v16, v20 // (op 1720) 
v_add_f32 v13, v17, v21 // (op 1720) 
v_add_f32 v14, v18, v22 // (op 1720) 
v_add_f32 v15, v19, v23 // (op 1720) 




s_lshl_b32 s0, s4, 7 // (op 1721) 
s_mul_i32 s1, 3, s0 // (op 1721) 

v_add_u32 v0, v9, s1 // (op 1721) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1721) Store value




s_lshl_b32 s0, s2, 9 // (op 1734) 
v_add_u32 v0, v6, s0 // (op 1734) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1734) Load value

v_accvgpr_read v16, a16 // (op 1731) 
v_accvgpr_read v17, a17 // (op 1731) 
v_accvgpr_read v18, a18 // (op 1731) 
v_accvgpr_read v19, a19 // (op 1731) 
v_mul_f32 v20, v11, v16 // (op 1731) 
v_mul_f32 v21, v11, v17 // (op 1731) 
v_mul_f32 v22, v11, v18 // (op 1731) 
v_mul_f32 v23, v11, v19 // (op 1731) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1735) 
v_mul_f32 v17, v10, v13 // (op 1735) 
v_mul_f32 v18, v10, v14 // (op 1735) 
v_mul_f32 v19, v10, v15 // (op 1735) 

v_add_f32 v12, v16, v20 // (op 1732) 
v_add_f32 v13, v17, v21 // (op 1732) 
v_add_f32 v14, v18, v22 // (op 1732) 
v_add_f32 v15, v19, v23 // (op 1732) 




s_lshl_b32 s0, s4, 9 // (op 1733) 
v_add_u32 v0, v9, s0 // (op 1733) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1733) Store value




s_lshl_b32 s0, s2, 7 // (op 1746) 
s_mul_i32 s1, 5, s0 // (op 1746) 

v_add_u32 v0, v6, s1 // (op 1746) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1746) Load value

v_accvgpr_read v16, a20 // (op 1743) 
v_accvgpr_read v17, a21 // (op 1743) 
v_accvgpr_read v18, a22 // (op 1743) 
v_accvgpr_read v19, a23 // (op 1743) 
v_mul_f32 v20, v11, v16 // (op 1743) 
v_mul_f32 v21, v11, v17 // (op 1743) 
v_mul_f32 v22, v11, v18 // (op 1743) 
v_mul_f32 v23, v11, v19 // (op 1743) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1747) 
v_mul_f32 v17, v10, v13 // (op 1747) 
v_mul_f32 v18, v10, v14 // (op 1747) 
v_mul_f32 v19, v10, v15 // (op 1747) 

v_add_f32 v12, v16, v20 // (op 1744) 
v_add_f32 v13, v17, v21 // (op 1744) 
v_add_f32 v14, v18, v22 // (op 1744) 
v_add_f32 v15, v19, v23 // (op 1744) 




s_lshl_b32 s0, s4, 7 // (op 1745) 
s_mul_i32 s1, 5, s0 // (op 1745) 

v_add_u32 v0, v9, s1 // (op 1745) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1745) Store value




s_lshl_b32 s0, s2, 7 // (op 1758) 
s_mul_i32 s1, 6, s0 // (op 1758) 

v_add_u32 v0, v6, s1 // (op 1758) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1758) Load value

v_accvgpr_read v16, a24 // (op 1755) 
v_accvgpr_read v17, a25 // (op 1755) 
v_accvgpr_read v18, a26 // (op 1755) 
v_accvgpr_read v19, a27 // (op 1755) 
v_mul_f32 v20, v11, v16 // (op 1755) 
v_mul_f32 v21, v11, v17 // (op 1755) 
v_mul_f32 v22, v11, v18 // (op 1755) 
v_mul_f32 v23, v11, v19 // (op 1755) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1759) 
v_mul_f32 v17, v10, v13 // (op 1759) 
v_mul_f32 v18, v10, v14 // (op 1759) 
v_mul_f32 v19, v10, v15 // (op 1759) 

v_add_f32 v12, v16, v20 // (op 1756) 
v_add_f32 v13, v17, v21 // (op 1756) 
v_add_f32 v14, v18, v22 // (op 1756) 
v_add_f32 v15, v19, v23 // (op 1756) 




s_lshl_b32 s0, s4, 7 // (op 1757) 
s_mul_i32 s1, 6, s0 // (op 1757) 

v_add_u32 v0, v9, s1 // (op 1757) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1757) Store value




s_lshl_b32 s0, s2, 7 // (op 1770) 
s_mul_i32 s1, 7, s0 // (op 1770) 

v_add_u32 v0, v6, s1 // (op 1770) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 1770) Load value

v_accvgpr_read v16, a28 // (op 1767) 
v_accvgpr_read v17, a29 // (op 1767) 
v_accvgpr_read v18, a30 // (op 1767) 
v_accvgpr_read v19, a31 // (op 1767) 
v_mul_f32 v20, v11, v16 // (op 1767) 
v_mul_f32 v21, v11, v17 // (op 1767) 
v_mul_f32 v22, v11, v18 // (op 1767) 
v_mul_f32 v23, v11, v19 // (op 1767) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 1771) 
v_mul_f32 v17, v10, v13 // (op 1771) 
v_mul_f32 v18, v10, v14 // (op 1771) 
v_mul_f32 v19, v10, v15 // (op 1771) 

v_add_f32 v12, v16, v20 // (op 1768) 
v_add_f32 v13, v17, v21 // (op 1768) 
v_add_f32 v14, v18, v22 // (op 1768) 
v_add_f32 v15, v19, v23 // (op 1768) 




s_lshl_b32 s0, s4, 7 // (op 1769) 
s_mul_i32 s1, 7, s0 // (op 1769) 

v_add_u32 v0, v9, s1 // (op 1769) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 1769) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.LRR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end:
.size RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2, .LRR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end-RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
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
    DataTypeB: F6
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
  - .name: RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
    .symbol: RR_GEMM_TN_BF8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2.kd
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

