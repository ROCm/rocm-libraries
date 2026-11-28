// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2,@function
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 3741) 
v_and_b32 v3, 48, v1 // (op 3741) 
v_sub_u32 v4, v0, v3 // (op 3741) 

v_lshrrev_b32 v3, 4, v0 // (op 3741) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 3741) 


v_lshrrev_b32 v3, 7, v1 // (op 3741) 
v_lshlrev_b32 v4, 4, v3 // (op 3741) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 3741) 


v_mul_lo_u32 v0, v3, 6 // (op 3741) 
v_lshrrev_b32 v4, 3, v0 // (op 3741) 

v_lshrrev_b32 v0, 4, v3 // (op 3741) 

v_lshlrev_b32 v3, 2, v0 // (op 3741) 

v_add_u32 v0, v4, v3 // (op 3741) 


v_and_b32 v3, 63, v1 // (op 3898) 
v_and_b32 v4, 48, v1 // (op 3898) 
v_sub_u32 v5, v3, v4 // (op 3898) 

v_lshrrev_b32 v4, 4, v3 // (op 3898) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 3898) 


v_lshrrev_b32 v4, 7, v1 // (op 3898) 
v_lshlrev_b32 v5, 4, v4 // (op 3898) 

v_lshl_add_u32 v4, v3, 5, v5 // (op 3898) 


v_mul_lo_u32 v3, v4, 6 // (op 3898) 
v_lshrrev_b32 v5, 3, v3 // (op 3898) 

v_lshrrev_b32 v3, 4, v4 // (op 3898) 

v_lshlrev_b32 v4, 2, v3 // (op 3898) 

v_add_u32 v3, v5, v4 // (op 3898) 


v_lshrrev_b32 v4, 7, v1 // (op 3684) 
v_and_b32 v5, 63, v1 // (op 3684) 
v_lshrrev_b32 v6, 4, v5 // (op 3684) 
v_lshlrev_b32 v7, 2, v6 // (op 3684) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 3684) 


v_lshl_add_u32 v4, s16, 5, v6 // (op 3684) 

v_lshrrev_b32 v6, 2, v5 // (op 3684) 

v_and_b32 v5, 3, v6 // (op 3684) 

v_and_b32 v6, 3, v1 // (op 3684) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3684) 


v_lshrrev_b32 v5, 6, v1 // (op 3684) 
v_and_b32 v6, 1, v5 // (op 3684) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3684) 


v_lshl_add_u32 v6, s17, 8, v5 // (op 3684) 

v_mul_lo_u32 v5, v6, s2 // (op 3684) 

v_add_lshl_u32 v6, v4, v5, 2 // (op 3684) 


v_lshrrev_b32 v4, 6, v1 // (op 3798) 
v_and_b32 v5, 1, v4 // (op 3798) 

v_and_b32 v4, 15, v1 // (op 3798) 
v_lshl_add_u32 v7, v5, 4, v4 // (op 3798) 


v_and_b32 v4, 48, v1 // (op 3798) 
v_lshl_add_u32 v5, v7, 7, v4 // (op 3798) 


v_lshrrev_b32 v4, 6, v1 // (op 3856) 
v_and_b32 v7, 1, v4 // (op 3856) 

v_and_b32 v4, 15, v1 // (op 3856) 
s_mul_i32 s1, s2, s9 // (op 3686) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3686) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3686) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3686) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3686) most significant: sum
s_add_u32 s1, s1, s19 // (op 3686) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3686) 

v_lshl_add_u32 v8, v7, 4, v4 // (op 3856) 


v_and_b32 v4, 63, v1 // (op 3856) 
v_lshrrev_b32 v7, 4, v4 // (op 3856) 

v_lshl_add_u32 v4, v8, 2, v7 // (op 3856) 


v_lshrrev_b32 v7, 7, v1 // (op 3872) 
v_and_b32 v8, 63, v1 // (op 3872) 
v_lshrrev_b32 v9, 4, v8 // (op 3872) 
v_lshlrev_b32 v10, 2, v9 // (op 3872) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3872) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 3872) 

v_lshrrev_b32 v9, 2, v8 // (op 3872) 

v_and_b32 v8, 3, v9 // (op 3872) 

v_and_b32 v9, 3, v1 // (op 3872) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3872) 


v_lshrrev_b32 v8, 6, v1 // (op 3872) 
v_and_b32 v9, 1, v8 // (op 3872) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3872) 


v_lshl_add_u32 v9, s17, 8, v8 // (op 3872) 

v_mul_lo_u32 v8, v9, s4 // (op 3872) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 3872) 


v_lshrrev_b32 v7, 6, v1 // (op 3942) 
v_and_b32 v8, 1, v7 // (op 3942) 

v_and_b32 v7, 15, v1 // (op 3942) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 3942) 


v_and_b32 v7, 48, v1 // (op 3942) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 3942) 


v_lshrrev_b32 v7, 6, v1 // (op 3971) 
v_and_b32 v10, 1, v7 // (op 3971) 

v_and_b32 v7, 15, v1 // (op 3971) 
s_mul_i32 s1, s4, s9 // (op 3874) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 3971) 


v_and_b32 v7, 63, v1 // (op 3971) 
v_lshrrev_b32 v10, 4, v7 // (op 3971) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 3971) 


s_mul_i32 s38, s5, s8 // (op 3874) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3874) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3874) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3874) most significant: sum
s_add_u32 s1, s1, s39 // (op 3874) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 3874) 

s_mov_b32 s40, s6 // (op 3686) reg expression
s_mov_b32 s41, s7 // (op 3686) reg expression
s_mov_b32 s42, s18 // (op 3686) reg expression
s_mov_b32 s43, 131072 // (op 3686) call()

s_mov_b32 s44, s10 // (op 3874) reg expression

s_mov_b32 s45, s11 // (op 3874) reg expression
s_mov_b32 s46, s38 // (op 3874) reg expression
s_mov_b32 s47, 131072 // (op 3874) call()


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
v_and_b32 v12, -2, v1 // (op 3451) 
v_sub_u32 v13, v1, v12 // (op 3451) 

v_lshlrev_b32 v12, 4, v13 // (op 3451) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 3451) 

v_mov_b32 v15, 0 // (op 3451) convert
v_mov_b32 v14, v13 // (op 3451) convert

v_lshrrev_b32 v12, 1, v1 // (op 3451) 
v_mul_lo_u32 v17, v12, s13 // (op 3451) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 3451) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 3451) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3451) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 3451) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 3451) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 3451) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 3451) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 3451) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 3451) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 3451) 

v_mov_b32 v14, v12 // (op 3451) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 3453) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3453) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3453) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3453) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3453) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3453) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3453) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3453) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3453) most significant: sum
s_add_u32 s7, s7, s11 // (op 3453) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3453) 

s_mov_b32 s48, s20 // (op 3453) reg expression
s_mov_b32 s49, s21 // (op 3453) reg expression
s_mov_b32 s50, s0 // (op 3453) reg expression
s_mov_b32 s51, 131072 // (op 3453) call()

v_and_b32 v12, -2, v1 // (op 3470) 
v_sub_u32 v13, v1, v12 // (op 3470) 

v_lshlrev_b32 v12, 4, v13 // (op 3470) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 3470) 

v_mov_b32 v17, 0 // (op 3470) convert
v_mov_b32 v16, v13 // (op 3470) convert

v_lshrrev_b32 v15, 1, v1 // (op 3470) 
v_mul_lo_u32 v13, v15, s13 // (op 3470) most significant: low of low * high
v_mul_hi_u32 v18, v15, s12 // (op 3470) most significant: high of low * low
v_mul_lo_u32 v12, v15, s12 // (op 3470) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 3470) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v12 // (op 3470) least significant half
v_addc_co_u32 v19, s[0:1], v17, v13, s[0:1] // (op 3470) most significant half



v_mul_lo_u32 v13, v19, 6 // (op 3470) most significant: low of high * low
v_mul_hi_u32 v15, v18, 6 // (op 3470) most significant: high of low * low
v_mul_lo_u32 v12, v18, 6 // (op 3470) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 3470) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 3470) 

v_mov_b32 v15, v16 // (op 3470) convert

s_mul_i32 s1, s23, 6 // (op 3472) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3472) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3472) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3472) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3472) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3472) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3472) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3472) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3472) most significant: sum
s_add_u32 s7, s7, s11 // (op 3472) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3472) 

s_mov_b32 s48, s20 // (op 3472) reg expression
s_mov_b32 s49, s21 // (op 3472) reg expression
s_mov_b32 s50, s0 // (op 3472) reg expression
s_mov_b32 s51, 131072 // (op 3472) call()

v_and_b32 v12, -8, v1 // (op 3515) 
v_sub_u32 v13, v1, v12 // (op 3515) 

v_lshlrev_b32 v12, 4, v13 // (op 3515) 

v_mov_b32 v17, 0 // (op 3515) convert
v_mov_b32 v16, v12 // (op 3515) convert

v_lshrrev_b32 v12, 3, v1 // (op 3515) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3515) 

v_mul_lo_u32 v19, v13, s25 // (op 3515) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3515) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 3515) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3515) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 3515) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 3515) most significant half



s_mul_i32 s1, s24, s9 // (op 3517) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3517) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3517) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3517) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3517) most significant: sum
s_add_u32 s1, s1, s7 // (op 3517) most significant: sum


s_mov_b32 s52, s26 // (op 3517) reg expression
s_mov_b32 s53, s27 // (op 3517) reg expression
s_mov_b32 s54, s0 // (op 3517) reg expression
s_mov_b32 s55, 131072 // (op 3517) call()

v_and_b32 v16, -8, v1 // (op 3534) 
v_sub_u32 v17, v1, v16 // (op 3534) 

v_lshlrev_b32 v16, 4, v17 // (op 3534) 

v_mov_b32 v19, 0 // (op 3534) convert
v_mov_b32 v18, v16 // (op 3534) convert

v_lshrrev_b32 v16, 3, v1 // (op 3534) 
v_lshl_add_u32 v17, s17, 8, v16 // (op 3534) 

v_mul_lo_u32 v21, v17, s25 // (op 3534) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 3534) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 3534) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 3534) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 3534) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 3534) most significant half



s_mul_i32 s1, s24, s9 // (op 3536) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3536) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3536) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3536) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3536) most significant: sum
s_add_u32 s1, s1, s7 // (op 3536) most significant: sum


s_mov_b32 s52, s26 // (op 3536) reg expression
s_mov_b32 s53, s27 // (op 3536) reg expression
s_mov_b32 s54, s0 // (op 3536) reg expression
s_mov_b32 s55, 131072 // (op 3536) call()

v_lshrrev_b32 v18, 7, v1 // (op 3579) 
v_and_b32 v19, 15, v1 // (op 3579) 
v_lshl_add_u32 v20, v18, 4, v19 // (op 3579) 


v_lshl_add_u32 v18, s16, 5, v20 // (op 3579) 

v_and_b32 v19, 63, v1 // (op 3579) 
v_lshrrev_b32 v20, 4, v19 // (op 3579) 

v_mul_lo_u32 v19, v20, s28 // (op 3579) 

v_add_u32 v20, v18, v19 // (op 3579) 


s_mul_i32 s1, s28, s33 // (op 3581) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3581) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3581) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3581) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3581) most significant: sum
s_add_u32 s1, s1, s7 // (op 3581) most significant: sum


s_mov_b32 s56, s30 // (op 3581) reg expression
s_mov_b32 s57, s31 // (op 3581) reg expression
s_mov_b32 s58, s0 // (op 3581) reg expression
s_mov_b32 s59, 131072 // (op 3581) call()

v_lshrrev_b32 v18, 7, v1 // (op 3598) 
v_and_b32 v19, 15, v1 // (op 3598) 
v_lshl_add_u32 v21, v18, 4, v19 // (op 3598) 


v_lshl_add_u32 v18, s16, 5, v21 // (op 3598) 

v_and_b32 v19, 63, v1 // (op 3598) 
v_lshrrev_b32 v21, 4, v19 // (op 3598) 

v_mul_lo_u32 v19, v21, s28 // (op 3598) 

v_add_u32 v21, v18, v19 // (op 3598) 


s_mul_i32 s1, s28, s33 // (op 3600) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3600) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3600) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3600) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3600) most significant: sum
s_add_u32 s1, s1, s7 // (op 3600) most significant: sum


s_mov_b32 s56, s30 // (op 3600) reg expression
s_mov_b32 s57, s31 // (op 3600) reg expression
s_mov_b32 s58, s0 // (op 3600) reg expression
s_mov_b32 s59, 131072 // (op 3600) call()

v_lshl_add_u32 v18, s17, 8, v1 // (op 3643) 
v_mul_lo_u32 v19, v18, s34 // (op 3643) 

s_mul_i32 s1, s34, s9 // (op 3645) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 3645) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3645) least significant
s_mul_hi_u32 s7, s34, s8 // (op 3645) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3645) most significant: sum
s_add_u32 s1, s1, s7 // (op 3645) most significant: sum


s_mov_b32 s60, s36 // (op 3645) reg expression
s_mov_b32 s61, s37 // (op 3645) reg expression
s_mov_b32 s62, s0 // (op 3645) reg expression
s_mov_b32 s63, 131072 // (op 3645) call()

v_lshl_add_u32 v18, s17, 8, v1 // (op 3662) 
v_mul_lo_u32 v22, v18, s34 // (op 3662) 

s_mul_i32 s1, s34, s9 // (op 3664) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 3664) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3664) least significant
s_mul_hi_u32 s7, s34, s8 // (op 3664) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3664) most significant: sum
s_add_u32 s1, s1, s7 // (op 3664) most significant: sum


s_mov_b32 s60, s36 // (op 3664) reg expression
s_mov_b32 s61, s37 // (op 3664) reg expression
s_mov_b32 s62, s0 // (op 3664) reg expression
s_mov_b32 s63, 131072 // (op 3664) call()


v_mov_b64 v[24:25], s[28:29] // (op 3108) convert

v_mov_b32 v18, v20 // (op 3108) reg expression
v_mov_b32 v23, v18 // (op 3108) 
buffer_load_ubyte v26, v23, s[56:59], 0 offen offset: 0 // (op 3108) Load value



v_and_b32 v18, -8, v1 // (op 3781) 
v_sub_u32 v23, v1, v18 // (op 3781) 

v_lshlrev_b32 v18, 4, v23 // (op 3781) 

v_lshrrev_b32 v23, 3, v1 // (op 3781) 
v_lshl_add_u32 v24, v23, 7, v18 // (op 3781) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 3781) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 3400) 
v_mov_b64 v[24:25], s[6:7] // (op 3400) convert


v_mov_b64 v[28:29], v[12:13] // (op 3400) 
s_mov_b32 m0, s0 // (op 3400) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3400) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3400) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3400) most significant half

s_add_u32 m0, m0, 4096 // (op 3400) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3400) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3400) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3400) most significant half

s_add_u32 m0, m0, 4096 // (op 3400) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3400) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3400) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3400) most significant half

s_add_u32 m0, m0, 4096 // (op 3400) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3400) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3400) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3400) most significant half

s_add_u32 m0, m0, 4096 // (op 3400) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3400) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3400) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3400) most significant half

s_add_u32 m0, m0, 4096 // (op 3400) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3400) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3400) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3400) most significant half

s_add_u32 m0, m0, 4096 // (op 3400) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3400) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3400) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3400) most significant half

s_add_u32 m0, m0, 4096 // (op 3400) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3400) Load value direct to lds






buffer_load_dword v18, v19, s[60:63], 0 offen // (op 3115) Load value


v_mov_b32 v23, v14 // (op 3118) reg expression

buffer_load_dwordx3 v[28:30], v23, s[48:51], 0 offen // (op 3118) Load value


v_mov_b64 v[24:25], s[28:29] // (op 3122) convert

s_lshl_b32 s0, s28, 2 // (op 3122) 
v_add_u32 v23, v20, s0 // (op 3122) 

v_mov_b32 v27, v23 // (op 3122) 
buffer_load_ubyte v31, v27, s[56:59], 0 offen offset: 0 // (op 3122) Load value




v_and_b32 v20, -8, v1 // (op 3925) 
v_sub_u32 v27, v1, v20 // (op 3925) 

v_lshlrev_b32 v20, 4, v27 // (op 3925) 

v_lshrrev_b32 v27, 3, v1 // (op 3925) 
v_lshl_add_u32 v23, v27, 7, v20 // (op 3925) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 3925) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 3405) 
v_mov_b64 v[24:25], s[6:7] // (op 3405) convert


v_mov_b64 v[32:33], v[12:13] // (op 3405) 
s_add_u32 s1, 32768, s0 // (op 3405) 
s_mov_b32 m0, s1 // (op 3405) reg expression
s_mov_b32 s6, 128 // (op 3405) call()
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3405) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3405) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3405) most significant half

s_add_u32 m0, m0, 4096 // (op 3405) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3405) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3405) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3405) most significant half

s_add_u32 m0, m0, 4096 // (op 3405) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3405) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3405) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3405) most significant half

s_add_u32 m0, m0, 4096 // (op 3405) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3405) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3405) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3405) most significant half

s_add_u32 m0, m0, 4096 // (op 3405) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3405) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3405) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3405) most significant half

s_add_u32 m0, m0, 4096 // (op 3405) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3405) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3405) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3405) most significant half

s_add_u32 m0, m0, 4096 // (op 3405) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3405) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3405) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3405) most significant half

s_add_u32 m0, m0, 4096 // (op 3405) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3405) Load value direct to lds









buffer_load_dword v20, v19, s[60:63], 0 offen offset:4 // (op 3129) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 3132) 
s_mul_i32 s7, s1, 6 // (op 3132) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 3132) least significant
s_mul_hi_u32 s10, s0, 6 // (op 3132) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3132) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 3132) 

v_add_u32 v19, v14, s0 // (op 3132) 


buffer_load_dwordx3 v[32:34], v19, s[48:51], 0 offen // (op 3132) Load value


v_lshlrev_b32 v19, 2, v1 // (op 3839) 



v_add_u32 v27, 65536, v19 // (op 3135) 
s_waitcnt vmcnt(12)
ds_write_b32 v27, v18 // (op 3135) Store local data 


v_and_b32 v19, -2, v1 // (op 3729) 
v_sub_u32 v27, v1, v19 // (op 3729) 

v_lshlrev_b32 v19, 4, v27 // (op 3729) 

v_lshrrev_b32 v27, 1, v1 // (op 3729) 
v_lshl_add_u32 v12, v27, 5, v19 // (op 3729) 


v_mul_lo_u32 v19, v12, 6 // (op 3729) 
v_lshrrev_b32 v27, 3, v19 // (op 3729) 

v_lshrrev_b32 v19, 4, v12 // (op 3729) 

v_lshlrev_b32 v12, 2, v19 // (op 3729) 

v_add_u32 v19, v27, v12 // (op 3729) 





v_add_u32 v27, 66560, v19 // (op 3136) 
s_waitcnt vmcnt(11)
ds_write_b96 v27, v[28:30] // (op 3136) Store local data 


v_and_b32 v19, -2, v1 // (op 3700) 
v_sub_u32 v27, v1, v19 // (op 3700) 

v_lshlrev_b32 v19, 4, v27 // (op 3700) 

v_lshrrev_b32 v27, 1, v1 // (op 3700) 
v_lshl_add_u32 v12, v27, 5, v19 // (op 3700) 


v_mul_lo_u32 v19, v12, 6 // (op 3700) 
v_lshrrev_b32 v27, 3, v19 // (op 3700) 

v_lshrrev_b32 v19, 4, v12 // (op 3700) 

v_lshlrev_b32 v12, 2, v19 // (op 3700) 

v_add_u32 v19, v27, v12 // (op 3700) 


v_and_b32 v27, -8, v1 // (op 3753) 
v_sub_u32 v12, v1, v27 // (op 3753) 

v_lshlrev_b32 v27, 4, v12 // (op 3753) 

v_lshrrev_b32 v12, 3, v1 // (op 3753) 
v_lshl_add_u32 v13, v12, 7, v27 // (op 3753) 


s_nop 0
v_readfirstlane_b32 s0, v13 // (op 3753) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v27, 2, v1 // (op 3810) 
v_and_b32 v12, -2, v1 // (op 3886) 
v_sub_u32 v13, v1, v12 // (op 3886) 

v_lshlrev_b32 v12, 4, v13 // (op 3886) 

v_lshrrev_b32 v13, 1, v1 // (op 3886) 
v_lshl_add_u32 v14, v13, 5, v12 // (op 3886) 


v_mul_lo_u32 v12, v14, 6 // (op 3886) 
v_lshrrev_b32 v13, 3, v12 // (op 3886) 

v_lshrrev_b32 v12, 4, v14 // (op 3886) 

v_lshlrev_b32 v14, 2, v12 // (op 3886) 

v_add_u32 v12, v13, v14 // (op 3886) 


v_and_b32 v13, -8, v1 // (op 3910) 
v_sub_u32 v14, v1, v13 // (op 3910) 

v_lshlrev_b32 v13, 4, v14 // (op 3910) 

v_lshrrev_b32 v14, 3, v1 // (op 3910) 
v_lshl_add_u32 v23, v14, 7, v13 // (op 3910) 


s_nop 0
v_readfirstlane_b32 s1, v23 // (op 3910) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v13, 2, v1 // (op 3954) 
s_mov_b32 s6, 0 // (op 1244) call()
s_ashr_i64 s[10:11], s[22:23], 7 // (op 1243) 
s_mov_b32 s7, s10 // (op 1243) convert

s_and_b32 s10, s7, 4294967294 // (op 1243) 

s_cmp_lt_u32 s6, s10 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1243 // (op 1243) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1243 if false)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1243:
s_add_u32 s6, s6, 2 // (op 1245) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4030) 



v_add_u32 v14, 66560, v0 // (op 65) 
ds_read_b96_tr_b6 v[36:38], v14 // (op 65) Transpose load local data 

v_add_u32 v14, 67072, v0 // (op 65) 
ds_read_b96_tr_b6 v[39:41], v14 // (op 65) Transpose load local data 

s_barrier  // (op 4043) 



ds_read_b128 v[42:45], v5 // (op 70) Load local data 
ds_read_b128 v[46:49], v5 offset:64 // (op 70) Load local data 
s_barrier  // (op 4056) 



v_add_u32 v14, 65536, v4 // (op 79) 
ds_read_u8 v35, v14 // (op 79) Load local data 




ds_read_b128 v[50:53], v5 offset:4096 // (op 339) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 82) 
ds_read_b128 v[54:57], v5 offset:4160 // (op 339) Load local data 





v_add_u32 v14, 65664, v4 // (op 342) 
ds_read_u8 v35, v14 // (op 342) Load local data 




ds_read_b128 v[42:45], v5 offset:8192 // (op 495) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[50:57], a[4:7], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 343) 
ds_read_b128 v[46:49], v5 offset:8256 // (op 495) Load local data 





v_add_u32 v14, 65792, v4 // (op 498) 
ds_read_u8 v35, v14 // (op 498) Load local data 




ds_read_b128 v[50:53], v5 offset:12288 // (op 651) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:41], v[42:49], a[8:11], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 499) 
ds_read_b128 v[54:57], v5 offset:12352 // (op 651) Load local data 





v_add_u32 v14, 65920, v4 // (op 654) 
ds_read_u8 v35, v14 // (op 654) Load local data 




ds_read_b128 v[42:45], v5 offset:16384 // (op 807) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:41], v[50:57], a[12:15], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 655) 
ds_read_b128 v[46:49], v5 offset:16448 // (op 807) Load local data 





v_add_u32 v14, 66048, v4 // (op 810) 
ds_read_u8 v35, v14 // (op 810) Load local data 




ds_read_b128 v[50:53], v5 offset:20480 // (op 963) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[36:41], v[42:49], a[16:19], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 811) 
ds_read_b128 v[54:57], v5 offset:20544 // (op 963) Load local data 





v_add_u32 v14, 66176, v4 // (op 966) 
ds_read_u8 v35, v14 // (op 966) Load local data 




ds_read_b128 v[42:45], v5 offset:24576 // (op 1119) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[50:57], a[20:23], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 967) 
ds_read_b128 v[46:49], v5 offset:24640 // (op 1119) Load local data 





v_add_u32 v14, 66304, v4 // (op 1122) 
ds_read_u8 v35, v14 // (op 1122) Load local data 




ds_read_b128 v[50:53], v5 offset:28672 // (op 1275) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[36:41], v[42:49], a[24:27], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1123) 
ds_read_b128 v[54:57], v5 offset:28736 // (op 1275) Load local data 





v_add_u32 v14, 66432, v4 // (op 1278) 
ds_read_u8 v35, v14 // (op 1278) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[36:41], v[50:57], a[28:31], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1279) 




v_mov_b64 v[24:25], s[28:29] // (op 75) convert

v_mov_b32 v14, v21 // (op 75) 
s_lshl_b32 s7, s28, 3 // (op 75) 
v_add_u32 v23, v14, s7 // (op 75) 

v_mov_b32 v35, v23 // (op 75) 
buffer_load_ubyte v26, v35, s[56:59], 0 offen offset: 0 // (op 75) Load value



s_barrier  // (op 4046) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 3383) 
v_mov_b64 v[24:25], s[10:11] // (op 3383) convert


v_mov_b64 v[36:37], v[16:17] // (op 3383) 
v_mov_b64 v[38:39], v[36:37] // (op 3383) 
s_mov_b32 m0, s0 // (op 3383) reg expression
s_mov_b32 s7, 256 // (op 3383) call()
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3383) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3383) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3383) most significant half

s_add_u32 m0, m0, 4096 // (op 3383) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3383) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3383) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3383) most significant half

s_add_u32 m0, m0, 4096 // (op 3383) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3383) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3383) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3383) most significant half

s_add_u32 m0, m0, 4096 // (op 3383) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3383) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3383) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3383) most significant half

s_add_u32 m0, m0, 4096 // (op 3383) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3383) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3383) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3383) most significant half

s_add_u32 m0, m0, 4096 // (op 3383) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3383) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3383) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3383) most significant half

s_add_u32 m0, m0, 4096 // (op 3383) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3383) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3383) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3383) most significant half

s_add_u32 m0, m0, 4096 // (op 3383) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3383) Load value direct to lds





v_mov_b32 v35, v22 // (op 21) 

buffer_load_dword v18, v35, s[60:63], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v23, v15 // (op 4) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 4) 
s_mul_i32 s15, s11, 6 // (op 4) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 4) least significant
s_mul_hi_u32 s7, s10, 6 // (op 4) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 4) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 4) 

s_lshl_b32 s7, s10, 1 // (op 4) 

v_add_u32 v24, v23, s7 // (op 4) 


buffer_load_dwordx3 v[28:30], v24, s[48:51], 0 offen // (op 4) Load value

s_barrier  // (op 4085) 



v_add_u32 v24, 70656, v13 // (op 236) 
s_waitcnt vmcnt(12)
ds_write_b32 v24, v20 // (op 236) Store local data 

s_barrier  // (op 4073) 



v_add_u32 v24, 71680, v12 // (op 225) 
s_waitcnt vmcnt(11)
ds_write_b96 v24, v[32:34] // (op 225) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4070) 



v_add_u32 v24, 71680, v3 // (op 227) 
ds_read_b96_tr_b6 v[38:40], v24 // (op 227) Transpose load local data 

v_add_u32 v24, 72192, v3 // (op 227) 
ds_read_b96_tr_b6 v[41:43], v24 // (op 227) Transpose load local data 

s_barrier  // (op 4076) 
s_barrier  // (op 4116) 



ds_read_b128 v[44:47], v8 offset:32768 // (op 229) Load local data 
ds_read_b128 v[48:51], v8 offset:32832 // (op 229) Load local data 
s_barrier  // (op 4082) 



v_add_u32 v24, 70656, v7 // (op 233) 
ds_read_u8 v25, v24 // (op 233) Load local data 




ds_read_b128 v[52:55], v8 offset:36864 // (op 348) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[44:51], a[0:3], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 234) 
ds_read_b128 v[56:59], v8 offset:36928 // (op 348) Load local data 





v_add_u32 v24, 70784, v7 // (op 351) 
ds_read_u8 v25, v24 // (op 351) Load local data 




ds_read_b128 v[44:47], v8 offset:40960 // (op 504) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:43], v[52:59], a[4:7], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 344) 
ds_read_b128 v[48:51], v8 offset:41024 // (op 504) Load local data 





v_add_u32 v24, 70912, v7 // (op 507) 
ds_read_u8 v25, v24 // (op 507) Load local data 




ds_read_b128 v[52:55], v8 offset:45056 // (op 660) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:43], v[44:51], a[8:11], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 500) 
ds_read_b128 v[56:59], v8 offset:45120 // (op 660) Load local data 





v_add_u32 v24, 71040, v7 // (op 663) 
ds_read_u8 v25, v24 // (op 663) Load local data 




ds_read_b128 v[44:47], v8 offset:49152 // (op 816) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:43], v[52:59], a[12:15], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 656) 
ds_read_b128 v[48:51], v8 offset:49216 // (op 816) Load local data 





v_add_u32 v24, 71168, v7 // (op 819) 
ds_read_u8 v25, v24 // (op 819) Load local data 




ds_read_b128 v[52:55], v8 offset:53248 // (op 972) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[38:43], v[44:51], a[16:19], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 812) 
ds_read_b128 v[56:59], v8 offset:53312 // (op 972) Load local data 





v_add_u32 v24, 71296, v7 // (op 975) 
ds_read_u8 v25, v24 // (op 975) Load local data 




ds_read_b128 v[44:47], v8 offset:57344 // (op 1128) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[38:43], v[52:59], a[20:23], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 968) 
ds_read_b128 v[48:51], v8 offset:57408 // (op 1128) Load local data 





v_add_u32 v24, 71424, v7 // (op 1131) 
ds_read_u8 v25, v24 // (op 1131) Load local data 




ds_read_b128 v[52:55], v8 offset:61440 // (op 1284) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[38:43], v[44:51], a[24:27], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1124) 
ds_read_b128 v[56:59], v8 offset:61504 // (op 1284) Load local data 





v_add_u32 v24, 71552, v7 // (op 1287) 
ds_read_u8 v25, v24 // (op 1287) Load local data 

s_barrier  // (op 4188) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[38:43], v[52:59], a[28:31], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1280) 




v_mov_b64 v[24:25], s[28:29] // (op 231) convert

s_lshl_b32 s7, s28, 2 // (op 231) 
s_mul_i32 s10, 3, s7 // (op 231) 

v_add_u32 v38, v14, s10 // (op 231) 

v_mov_b32 v39, v38 // (op 231) 
buffer_load_ubyte v31, v39, s[56:59], 0 offen offset: 0 // (op 231) Load value




s_barrier  // (op 4079) 
s_lshl_b32 s7, s28, 3 // (op 3590) 

v_add_u32 v21, v21, s7 // (op 3590) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 3395) 
v_mov_b64 v[24:25], s[10:11] // (op 3395) convert


v_mov_b64 v[38:39], v[36:37] // (op 3395) 
s_add_u32 s7, 32768, s1 // (op 3395) 
s_mov_b32 m0, s7 // (op 3395) reg expression
s_mov_b32 s10, 384 // (op 3395) call()
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3395) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3395) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3395) most significant half

s_add_u32 m0, m0, 4096 // (op 3395) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3395) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3395) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3395) most significant half

s_add_u32 m0, m0, 4096 // (op 3395) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3395) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3395) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3395) most significant half

s_add_u32 m0, m0, 4096 // (op 3395) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3395) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3395) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3395) most significant half

s_add_u32 m0, m0, 4096 // (op 3395) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3395) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3395) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3395) most significant half

s_add_u32 m0, m0, 4096 // (op 3395) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3395) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3395) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3395) most significant half

s_add_u32 m0, m0, 4096 // (op 3395) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3395) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3395) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3395) most significant half

s_add_u32 m0, m0, 4096 // (op 3395) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3395) Load value direct to lds








v_add_co_u32 v16, vcc, 256, v16 // (op 3526) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 3526) most significant half
buffer_load_dword v20, v35, s[60:63], 0 offen offset:12 // (op 238) Load value



v_add_u32 v22, 8, v22 // (op 3654) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 224) 
s_mul_i32 s15, s11, 6 // (op 224) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 224) least significant
s_mul_hi_u32 s7, s10, 6 // (op 224) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 224) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 224) 

s_mul_i32 s7, 3, s10 // (op 224) 

v_add_u32 v14, v23, s7 // (op 224) 


buffer_load_dwordx3 v[32:34], v14, s[48:51], 0 offen // (op 224) Load value


s_barrier  // (op 4059) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 3462) 



v_add_u32 v14, 65536, v27 // (op 26) 
s_mul_i32 s15, s11, 6 // (op 3462) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 3462) least significant
s_mul_hi_u32 s7, s10, 6 // (op 3462) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 3462) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 3462) 

s_lshl_b32 s7, s10, 1 // (op 3462) 

v_add_u32 v15, v15, s7 // (op 3462) 

s_waitcnt vmcnt(12)
ds_write_b32 v14, v18 // (op 26) Store local data 

s_barrier  // (op 4033) 



v_add_u32 v14, 66560, v19 // (op 6) 
s_waitcnt vmcnt(11)
ds_write_b96 v14, v[28:30] // (op 6) Store local data 

s_ashr_i64 s[10:11], s[22:23], 7 // (op 1243) 
s_mov_b32 s7, s10 // (op 1243) convert

s_and_b32 s10, s7, 4294967294 // (op 1243) 

s_cmp_lt_u32 s6, s10 // (op 1243) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1243 // (op 1243) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1243 if true)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1243:














s_barrier  // (op 4170) 
v_and_b32 v12, -2, v1 // (op 3418) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3418) 
v_sub_u32 v13, v1, v12 // (op 3418) 

s_lshr_b32 s6, s0, 1 // (op 3418) 

v_lshlrev_b32 v12, 4, v13 // (op 3418) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 3418) 

v_mov_b32 v15, 0 // (op 3418) convert
v_mov_b32 v14, v13 // (op 3418) convert

v_lshrrev_b32 v12, 1, v1 // (op 3418) 
v_lshl_add_u32 v13, s6, 8, v12 // (op 3418) 


v_mul_lo_u32 v17, v13, s13 // (op 3418) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3418) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3418) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3418) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 3418) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 3418) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 3418) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 3418) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 3418) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 3418) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 3418) 

v_mov_b32 v14, v12 // (op 3418) convert

s_mul_i32 s1, s23, 6 // (op 3420) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3420) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3420) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3420) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3420) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3420) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3420) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3420) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3420) most significant: sum
s_add_u32 s7, s7, s11 // (op 3420) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3420) 

s_mov_b32 s48, s20 // (op 3420) reg expression
s_mov_b32 s49, s21 // (op 3420) reg expression
s_mov_b32 s50, s0 // (op 3420) reg expression
s_mov_b32 s51, 131072 // (op 3420) call()

v_and_b32 v12, -8, v1 // (op 3492) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3492) 
v_sub_u32 v13, v1, v12 // (op 3492) 

s_lshr_b32 s6, s0, 1 // (op 3492) 

v_lshlrev_b32 v12, 4, v13 // (op 3492) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 3492) 


v_mov_b32 v19, 0 // (op 3492) convert
v_mov_b32 v18, v13 // (op 3492) convert

v_lshrrev_b32 v12, 3, v1 // (op 3492) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3492) 

v_mul_lo_u32 v17, v13, s25 // (op 3492) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3492) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 3492) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3492) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v16 // (op 3492) least significant half
v_addc_co_u32 v13, s[0:1], v19, v17, s[0:1] // (op 3492) most significant half



s_mul_i32 s1, s24, s9 // (op 3494) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3494) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3494) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3494) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3494) most significant: sum
s_add_u32 s1, s1, s7 // (op 3494) most significant: sum


s_mov_b32 s52, s26 // (op 3494) reg expression
s_mov_b32 s53, s27 // (op 3494) reg expression
s_mov_b32 s54, s0 // (op 3494) reg expression
s_mov_b32 s55, 131072 // (op 3494) call()

v_lshrrev_b32 v15, 7, v1 // (op 3556) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3556) 
v_and_b32 v16, 15, v1 // (op 3556) 
s_lshr_b32 s6, s0, 1 // (op 3556) 

v_lshl_add_u32 v17, v15, 4, v16 // (op 3556) 


v_lshl_add_u32 v15, s16, 5, v17 // (op 3556) 

v_and_b32 v16, 63, v1 // (op 3556) 
v_lshrrev_b32 v17, 4, v16 // (op 3556) 

v_lshl_add_u32 v16, s6, 3, v17 // (op 3556) 


v_mul_lo_u32 v17, v16, s28 // (op 3556) 

v_add_u32 v16, v15, v17 // (op 3556) 


s_mul_i32 s1, s28, s33 // (op 3558) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3558) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3558) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3558) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3558) most significant: sum
s_add_u32 s1, s1, s7 // (op 3558) most significant: sum


s_mov_b32 s56, s30 // (op 3558) reg expression
s_mov_b32 s57, s31 // (op 3558) reg expression
s_mov_b32 s58, s0 // (op 3558) reg expression
s_mov_b32 s59, 131072 // (op 3558) call()

v_lshl_add_u32 v15, s17, 8, v1 // (op 3620) 
v_mul_lo_u32 v17, v15, s34 // (op 3620) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 3620) 
s_lshr_b32 s6, s0, 1 // (op 3620) 

v_lshl_add_u32 v15, s6, 3, v17 // (op 3620) 


s_mul_i32 s1, s34, s9 // (op 3622) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 3622) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3622) least significant
s_mul_hi_u32 s7, s34, s8 // (op 3622) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3622) most significant: sum
s_add_u32 s1, s1, s7 // (op 3622) most significant: sum


s_mov_b32 s60, s36 // (op 3622) reg expression
s_mov_b32 s61, s37 // (op 3622) reg expression
s_mov_b32 s62, s0 // (op 3622) reg expression
s_mov_b32 s63, 131072 // (op 3622) call()

v_and_b32 v17, -2, v1 // (op 3713) 
v_sub_u32 v18, v1, v17 // (op 3713) 

v_lshlrev_b32 v17, 4, v18 // (op 3713) 

v_lshrrev_b32 v18, 1, v1 // (op 3713) 
v_lshl_add_u32 v19, v18, 5, v17 // (op 3713) 


v_mul_lo_u32 v17, v19, 6 // (op 3713) 
v_lshrrev_b32 v18, 3, v17 // (op 3713) 

v_lshrrev_b32 v17, 4, v19 // (op 3713) 

v_lshlrev_b32 v19, 2, v17 // (op 3713) 

v_add_u32 v17, v18, v19 // (op 3713) 


v_and_b32 v18, -8, v1 // (op 3766) 
v_sub_u32 v19, v1, v18 // (op 3766) 

v_lshlrev_b32 v18, 4, v19 // (op 3766) 

v_lshrrev_b32 v19, 3, v1 // (op 3766) 
v_lshl_add_u32 v21, v19, 7, v18 // (op 3766) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 3766) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v18, 2, v1 // (op 3823) 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 1249) 
s_and_b32 s1, s6, 4294967294 // (op 1249) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 1248) 
s_mov_b32 s10, s6 // (op 1248) convert

s_cmp_lt_u32 s1, s10 // (op 1248) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1248 // (op 1248) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1248 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1248:


s_add_u32 s1, s1, 1 // (op 1250) 
v_mov_b32 v19, v14 // (op 166) 
v_mov_b32 v20, v19 // (op 166) reg expression

buffer_load_dwordx3 v[22:24], v20, s[48:51], 0 offen // (op 166) Load value


s_lshl_b64 s[6:7], s[12:13], 7 // (op 3408) 
s_mul_i32 s11, s7, 6 // (op 3408) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 3408) least significant
s_mul_hi_u32 s14, s6, 6 // (op 3408) most significant: high of low * low
s_add_u32 s11, s11, s14 // (op 3408) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 3408) 

v_add_u32 v14, v14, s6 // (op 3408) 

s_barrier  // (op 4036) 
s_barrier  // (op 4049) 




v_add_u32 v19, 66560, v17 // (op 167) 
s_lshl_b64 s[6:7], s[24:25], 5 // (op 3388) 
v_mov_b64 v[20:21], s[6:7] // (op 3388) convert


v_mov_b64 v[26:27], v[12:13] // (op 3388) 
v_mov_b64 v[28:29], v[26:27] // (op 3388) 
s_mov_b32 m0, s0 // (op 3388) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3388) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3388) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3388) most significant half

s_add_u32 m0, m0, 4096 // (op 3388) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3388) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3388) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3388) most significant half

s_add_u32 m0, m0, 4096 // (op 3388) 
s_waitcnt vmcnt(2)
ds_write_b96 v19, v[22:24] // (op 167) Store local data 

buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3388) Load value direct to lds

v_add_co_u32 v28, s[6:7], v28, v20 // (op 3388) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3388) most significant half

s_add_u32 m0, m0, 4096 // (op 3388) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3388) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3388) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3388) most significant half

s_add_u32 m0, m0, 4096 // (op 3388) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3388) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3388) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3388) most significant half

s_add_u32 m0, m0, 4096 // (op 3388) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3388) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3388) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3388) most significant half

s_add_u32 m0, m0, 4096 // (op 3388) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3388) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3388) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3388) most significant half

s_add_u32 m0, m0, 4096 // (op 3388) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3388) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 3484) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 3484) most significant half


v_mov_b32 v19, v15 // (op 180) 

buffer_load_dword v20, v19, s[60:63], 0 offen // (op 180) Load value

v_add_u32 v15, 4, v15 // (op 3612) 
s_barrier  // (op 4062) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4039) 






v_add_u32 v19, 65536, v18 // (op 178) 
v_add_u32 v21, 66560, v0 // (op 169) 
ds_read_b96_tr_b6 v[22:24], v21 // (op 169) Transpose load local data 

v_add_u32 v21, 67072, v0 // (op 169) 
ds_read_b96_tr_b6 v[25:27], v21 // (op 169) Transpose load local data 

s_waitcnt vmcnt(0)
ds_write_b32 v19, v20 // (op 178) Store local data 


s_barrier  // (op 4052) 



ds_read_b128 v[28:31], v5 // (op 171) Load local data 
ds_read_b128 v[32:35], v5 offset:64 // (op 171) Load local data 

v_mov_b64 v[20:21], s[28:29] // (op 173) convert

v_mov_b32 v19, v16 // (op 173) 
v_mov_b32 v36, v19 // (op 173) reg expression
v_mov_b32 v37, v36 // (op 173) 
buffer_load_ubyte v38, v37, s[56:59], 0 offen offset: 0 // (op 173) Load value




s_lshl_b32 s6, s28, 2 // (op 3548) 
v_add_u32 v16, v16, s6 // (op 3548) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 4066) 



v_add_u32 v19, 65536, v4 // (op 175) 
ds_read_u8 v20, v19 // (op 175) Load local data 




s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[28:35], a[0:3], v38, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 176) 
ds_read_b128 v[40:43], v5 offset:4096 // (op 320) Load local data 


ds_read_b128 v[44:47], v5 offset:4160 // (op 320) Load local data 



v_add_u32 v39, 65664, v4 // (op 323) 
ds_read_u8 v19, v39 // (op 323) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:27], v[40:47], a[4:7], v38, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 324) 
ds_read_b128 v[28:31], v5 offset:8192 // (op 476) Load local data 


ds_read_b128 v[32:35], v5 offset:8256 // (op 476) Load local data 



v_add_u32 v19, 65792, v4 // (op 479) 
ds_read_u8 v20, v19 // (op 479) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:27], v[28:35], a[8:11], v38, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 480) 
ds_read_b128 v[40:43], v5 offset:12288 // (op 632) Load local data 


ds_read_b128 v[44:47], v5 offset:12352 // (op 632) Load local data 



v_add_u32 v39, 65920, v4 // (op 635) 
ds_read_u8 v19, v39 // (op 635) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:27], v[40:47], a[12:15], v38, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 636) 
ds_read_b128 v[28:31], v5 offset:16384 // (op 788) Load local data 


ds_read_b128 v[32:35], v5 offset:16448 // (op 788) Load local data 



v_add_u32 v19, 66048, v4 // (op 791) 
ds_read_u8 v20, v19 // (op 791) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:27], v[28:35], a[16:19], v38, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 792) 
ds_read_b128 v[40:43], v5 offset:20480 // (op 944) Load local data 


ds_read_b128 v[44:47], v5 offset:20544 // (op 944) Load local data 



v_add_u32 v39, 66176, v4 // (op 947) 
ds_read_u8 v19, v39 // (op 947) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:27], v[40:47], a[20:23], v38, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 948) 
ds_read_b128 v[28:31], v5 offset:24576 // (op 1100) Load local data 


ds_read_b128 v[32:35], v5 offset:24640 // (op 1100) Load local data 



v_add_u32 v19, 66304, v4 // (op 1103) 
ds_read_u8 v20, v19 // (op 1103) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:27], v[28:35], a[24:27], v38, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1104) 
ds_read_b128 v[40:43], v5 offset:28672 // (op 1256) Load local data 


ds_read_b128 v[44:47], v5 offset:28736 // (op 1256) Load local data 



v_add_u32 v39, 66432, v4 // (op 1259) 
ds_read_u8 v19, v39 // (op 1259) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:27], v[40:47], a[28:31], v38, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1260) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 1248) 
s_mov_b32 s10, s6 // (op 1248) convert

s_cmp_lt_u32 s1, s10 // (op 1248) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1248 // (op 1248) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1248 if true)
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1248:






















s_barrier  // (op 4174) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_123:


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
RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.LRR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end:
.size RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2, .LRR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end-RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
  .amdhsa_next_free_vgpr 92
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 75776 // lds bytes
  .amdhsa_accum_offset 60
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
    DataTypeB: B8
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
  - .name: RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
    .symbol: RR_GEMM_NN_BF6_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 75776
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 60
    .agpr_count: 32
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

