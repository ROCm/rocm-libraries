// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2,@function
RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 3773) 
v_and_b32 v3, 48, v1 // (op 3773) 
v_sub_u32 v4, v0, v3 // (op 3773) 

v_lshrrev_b32 v3, 4, v0 // (op 3773) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 3773) 


v_lshrrev_b32 v3, 7, v1 // (op 3773) 
v_lshlrev_b32 v4, 4, v3 // (op 3773) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 3773) 


v_mul_lo_u32 v0, v3, 6 // (op 3773) 
v_lshrrev_b32 v4, 3, v0 // (op 3773) 

v_lshrrev_b32 v0, 4, v3 // (op 3773) 

v_lshlrev_b32 v3, 2, v0 // (op 3773) 

v_add_u32 v0, v4, v3 // (op 3773) 


v_and_b32 v3, 63, v1 // (op 3930) 
v_and_b32 v4, 48, v1 // (op 3930) 
v_sub_u32 v5, v3, v4 // (op 3930) 

v_lshrrev_b32 v4, 4, v3 // (op 3930) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 3930) 


v_lshrrev_b32 v4, 7, v1 // (op 3930) 
v_lshlrev_b32 v5, 4, v4 // (op 3930) 

v_lshl_add_u32 v4, v3, 5, v5 // (op 3930) 


v_mul_lo_u32 v3, v4, 6 // (op 3930) 
v_lshrrev_b32 v5, 3, v3 // (op 3930) 

v_lshrrev_b32 v3, 4, v4 // (op 3930) 

v_lshlrev_b32 v4, 2, v3 // (op 3930) 

v_add_u32 v3, v5, v4 // (op 3930) 


v_lshrrev_b32 v4, 7, v1 // (op 3716) 
v_and_b32 v5, 63, v1 // (op 3716) 
v_lshrrev_b32 v6, 4, v5 // (op 3716) 
v_lshlrev_b32 v7, 2, v6 // (op 3716) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 3716) 


v_lshl_add_u32 v4, s16, 5, v6 // (op 3716) 

v_lshrrev_b32 v6, 2, v5 // (op 3716) 

v_and_b32 v5, 3, v6 // (op 3716) 

v_and_b32 v6, 3, v1 // (op 3716) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 3716) 


v_lshrrev_b32 v5, 6, v1 // (op 3716) 
v_and_b32 v6, 1, v5 // (op 3716) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 3716) 


v_lshl_add_u32 v6, s17, 8, v5 // (op 3716) 

v_mul_lo_u32 v5, v6, s2 // (op 3716) 

v_add_lshl_u32 v6, v4, v5, 1 // (op 3716) 


v_lshrrev_b32 v4, 6, v1 // (op 3830) 
v_and_b32 v5, 1, v4 // (op 3830) 

v_and_b32 v4, 15, v1 // (op 3830) 
v_lshl_add_u32 v7, v5, 4, v4 // (op 3830) 


v_and_b32 v4, 48, v1 // (op 3830) 
v_lshl_add_u32 v5, v7, 7, v4 // (op 3830) 


v_lshrrev_b32 v4, 6, v1 // (op 3888) 
v_and_b32 v7, 1, v4 // (op 3888) 

v_and_b32 v4, 15, v1 // (op 3888) 
s_mul_i32 s1, s2, s9 // (op 3718) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3718) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3718) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3718) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3718) most significant: sum
s_add_u32 s1, s1, s19 // (op 3718) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 3718) 

v_lshl_add_u32 v8, v7, 4, v4 // (op 3888) 


v_and_b32 v4, 63, v1 // (op 3888) 
v_lshrrev_b32 v7, 4, v4 // (op 3888) 

v_lshl_add_u32 v4, v8, 2, v7 // (op 3888) 


v_lshrrev_b32 v7, 7, v1 // (op 3904) 
v_and_b32 v8, 63, v1 // (op 3904) 
v_lshrrev_b32 v9, 4, v8 // (op 3904) 
v_lshlrev_b32 v10, 2, v9 // (op 3904) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 3904) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 3904) 

v_lshrrev_b32 v9, 2, v8 // (op 3904) 

v_and_b32 v8, 3, v9 // (op 3904) 

v_and_b32 v9, 3, v1 // (op 3904) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 3904) 


v_lshrrev_b32 v8, 6, v1 // (op 3904) 
v_and_b32 v9, 1, v8 // (op 3904) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 3904) 


v_lshl_add_u32 v9, s17, 8, v8 // (op 3904) 

v_mul_lo_u32 v8, v9, s4 // (op 3904) 

v_add_lshl_u32 v9, v7, v8, 1 // (op 3904) 


v_lshrrev_b32 v7, 6, v1 // (op 3974) 
v_and_b32 v8, 1, v7 // (op 3974) 

v_and_b32 v7, 15, v1 // (op 3974) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 3974) 


v_and_b32 v7, 48, v1 // (op 3974) 
v_lshl_add_u32 v8, v10, 7, v7 // (op 3974) 


v_lshrrev_b32 v7, 6, v1 // (op 4003) 
v_and_b32 v10, 1, v7 // (op 4003) 

v_and_b32 v7, 15, v1 // (op 4003) 
s_mul_i32 s1, s4, s9 // (op 3906) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 4003) 


v_and_b32 v7, 63, v1 // (op 4003) 
v_lshrrev_b32 v10, 4, v7 // (op 4003) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 4003) 


s_mul_i32 s38, s5, s8 // (op 3906) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3906) least significant
s_mul_hi_u32 s39, s4, s8 // (op 3906) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 3906) most significant: sum
s_add_u32 s1, s1, s39 // (op 3906) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 3906) 

s_mov_b32 s40, s6 // (op 3718) reg expression
s_mov_b32 s41, s7 // (op 3718) reg expression
s_mov_b32 s42, s18 // (op 3718) reg expression
s_mov_b32 s43, 131072 // (op 3718) call()

s_mov_b32 s44, s10 // (op 3906) reg expression

s_mov_b32 s45, s11 // (op 3906) reg expression
s_mov_b32 s46, s38 // (op 3906) reg expression
s_mov_b32 s47, 131072 // (op 3906) call()


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
v_and_b32 v12, -2, v1 // (op 3483) 
v_sub_u32 v13, v1, v12 // (op 3483) 

v_lshlrev_b32 v12, 4, v13 // (op 3483) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 3483) 

v_mov_b32 v15, 0 // (op 3483) convert
v_mov_b32 v14, v13 // (op 3483) convert

v_lshrrev_b32 v12, 1, v1 // (op 3483) 
v_mul_lo_u32 v17, v12, s13 // (op 3483) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 3483) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 3483) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 3483) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 3483) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 3483) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 3483) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 3483) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 3483) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 3483) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 3483) 

v_mov_b32 v14, v12 // (op 3483) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 3485) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3485) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3485) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3485) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3485) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3485) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3485) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3485) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3485) most significant: sum
s_add_u32 s7, s7, s11 // (op 3485) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3485) 

s_mov_b32 s48, s20 // (op 3485) reg expression
s_mov_b32 s49, s21 // (op 3485) reg expression
s_mov_b32 s50, s0 // (op 3485) reg expression
s_mov_b32 s51, 131072 // (op 3485) call()

v_and_b32 v12, -2, v1 // (op 3502) 
v_sub_u32 v13, v1, v12 // (op 3502) 

v_lshlrev_b32 v12, 4, v13 // (op 3502) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 3502) 

v_mov_b32 v17, 0 // (op 3502) convert
v_mov_b32 v16, v13 // (op 3502) convert

v_lshrrev_b32 v15, 1, v1 // (op 3502) 
v_mul_lo_u32 v13, v15, s13 // (op 3502) most significant: low of low * high
v_mul_hi_u32 v18, v15, s12 // (op 3502) most significant: high of low * low
v_mul_lo_u32 v12, v15, s12 // (op 3502) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 3502) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v12 // (op 3502) least significant half
v_addc_co_u32 v19, s[0:1], v17, v13, s[0:1] // (op 3502) most significant half



v_mul_lo_u32 v13, v19, 6 // (op 3502) most significant: low of high * low
v_mul_hi_u32 v15, v18, 6 // (op 3502) most significant: high of low * low
v_mul_lo_u32 v12, v18, 6 // (op 3502) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 3502) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 3502) 

v_mov_b32 v15, v16 // (op 3502) convert

s_mul_i32 s1, s23, 6 // (op 3504) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3504) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3504) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3504) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3504) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3504) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3504) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3504) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3504) most significant: sum
s_add_u32 s7, s7, s11 // (op 3504) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3504) 

s_mov_b32 s48, s20 // (op 3504) reg expression
s_mov_b32 s49, s21 // (op 3504) reg expression
s_mov_b32 s50, s0 // (op 3504) reg expression
s_mov_b32 s51, 131072 // (op 3504) call()

v_and_b32 v12, -8, v1 // (op 3547) 
v_sub_u32 v13, v1, v12 // (op 3547) 

v_lshlrev_b32 v12, 4, v13 // (op 3547) 

v_mov_b32 v17, 0 // (op 3547) convert
v_mov_b32 v16, v12 // (op 3547) convert

v_lshrrev_b32 v12, 3, v1 // (op 3547) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3547) 

v_mul_lo_u32 v19, v13, s25 // (op 3547) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3547) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 3547) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3547) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 3547) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 3547) most significant half



s_mul_i32 s1, s24, s9 // (op 3549) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3549) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3549) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3549) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3549) most significant: sum
s_add_u32 s1, s1, s7 // (op 3549) most significant: sum


s_mov_b32 s52, s26 // (op 3549) reg expression
s_mov_b32 s53, s27 // (op 3549) reg expression
s_mov_b32 s54, s0 // (op 3549) reg expression
s_mov_b32 s55, 131072 // (op 3549) call()

v_and_b32 v16, -8, v1 // (op 3566) 
v_sub_u32 v17, v1, v16 // (op 3566) 

v_lshlrev_b32 v16, 4, v17 // (op 3566) 

v_mov_b32 v19, 0 // (op 3566) convert
v_mov_b32 v18, v16 // (op 3566) convert

v_lshrrev_b32 v16, 3, v1 // (op 3566) 
v_lshl_add_u32 v17, s17, 8, v16 // (op 3566) 

v_mul_lo_u32 v21, v17, s25 // (op 3566) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 3566) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 3566) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 3566) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 3566) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 3566) most significant half



s_mul_i32 s1, s24, s9 // (op 3568) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 3568) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 3568) least significant
s_mul_hi_u32 s7, s24, s8 // (op 3568) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3568) most significant: sum
s_add_u32 s1, s1, s7 // (op 3568) most significant: sum


s_mov_b32 s52, s26 // (op 3568) reg expression
s_mov_b32 s53, s27 // (op 3568) reg expression
s_mov_b32 s54, s0 // (op 3568) reg expression
s_mov_b32 s55, 131072 // (op 3568) call()

v_lshrrev_b32 v18, 7, v1 // (op 3611) 
v_and_b32 v19, 15, v1 // (op 3611) 
v_lshl_add_u32 v20, v18, 4, v19 // (op 3611) 


v_lshl_add_u32 v18, s16, 5, v20 // (op 3611) 

v_and_b32 v19, 63, v1 // (op 3611) 
v_lshrrev_b32 v20, 4, v19 // (op 3611) 

v_mul_lo_u32 v19, v20, s28 // (op 3611) 

v_add_u32 v20, v18, v19 // (op 3611) 


s_mul_i32 s1, s28, s33 // (op 3613) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3613) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3613) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3613) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3613) most significant: sum
s_add_u32 s1, s1, s7 // (op 3613) most significant: sum


s_mov_b32 s56, s30 // (op 3613) reg expression
s_mov_b32 s57, s31 // (op 3613) reg expression
s_mov_b32 s58, s0 // (op 3613) reg expression
s_mov_b32 s59, 131072 // (op 3613) call()

v_lshrrev_b32 v18, 7, v1 // (op 3630) 
v_and_b32 v19, 15, v1 // (op 3630) 
v_lshl_add_u32 v21, v18, 4, v19 // (op 3630) 


v_lshl_add_u32 v18, s16, 5, v21 // (op 3630) 

v_and_b32 v19, 63, v1 // (op 3630) 
v_lshrrev_b32 v21, 4, v19 // (op 3630) 

v_mul_lo_u32 v19, v21, s28 // (op 3630) 

v_add_u32 v21, v18, v19 // (op 3630) 


s_mul_i32 s1, s28, s33 // (op 3632) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3632) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3632) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3632) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3632) most significant: sum
s_add_u32 s1, s1, s7 // (op 3632) most significant: sum


s_mov_b32 s56, s30 // (op 3632) reg expression
s_mov_b32 s57, s31 // (op 3632) reg expression
s_mov_b32 s58, s0 // (op 3632) reg expression
s_mov_b32 s59, 131072 // (op 3632) call()

v_lshl_add_u32 v18, s17, 8, v1 // (op 3675) 
v_mul_lo_u32 v19, v18, s34 // (op 3675) 

s_mul_i32 s1, s34, s9 // (op 3677) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 3677) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3677) least significant
s_mul_hi_u32 s7, s34, s8 // (op 3677) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3677) most significant: sum
s_add_u32 s1, s1, s7 // (op 3677) most significant: sum


s_mov_b32 s60, s36 // (op 3677) reg expression
s_mov_b32 s61, s37 // (op 3677) reg expression
s_mov_b32 s62, s0 // (op 3677) reg expression
s_mov_b32 s63, 131072 // (op 3677) call()

v_lshl_add_u32 v18, s17, 8, v1 // (op 3694) 
v_mul_lo_u32 v22, v18, s34 // (op 3694) 

s_mul_i32 s1, s34, s9 // (op 3696) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 3696) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3696) least significant
s_mul_hi_u32 s7, s34, s8 // (op 3696) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3696) most significant: sum
s_add_u32 s1, s1, s7 // (op 3696) most significant: sum


s_mov_b32 s60, s36 // (op 3696) reg expression
s_mov_b32 s61, s37 // (op 3696) reg expression
s_mov_b32 s62, s0 // (op 3696) reg expression
s_mov_b32 s63, 131072 // (op 3696) call()


v_mov_b64 v[24:25], s[28:29] // (op 3140) convert

v_mov_b32 v18, v20 // (op 3140) reg expression
v_mov_b32 v23, v18 // (op 3140) 
buffer_load_ubyte v26, v23, s[56:59], 0 offen offset: 0 // (op 3140) Load value



v_and_b32 v18, -8, v1 // (op 3813) 
v_sub_u32 v23, v1, v18 // (op 3813) 

v_lshlrev_b32 v18, 4, v23 // (op 3813) 

v_lshrrev_b32 v23, 3, v1 // (op 3813) 
v_lshl_add_u32 v24, v23, 7, v18 // (op 3813) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 3813) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 3432) 
v_mov_b64 v[24:25], s[6:7] // (op 3432) convert


v_mov_b64 v[28:29], v[12:13] // (op 3432) 
s_mov_b32 m0, s0 // (op 3432) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3432) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3432) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3432) most significant half

s_add_u32 m0, m0, 4096 // (op 3432) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3432) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3432) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3432) most significant half

s_add_u32 m0, m0, 4096 // (op 3432) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3432) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3432) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3432) most significant half

s_add_u32 m0, m0, 4096 // (op 3432) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3432) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3432) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3432) most significant half

s_add_u32 m0, m0, 4096 // (op 3432) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3432) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3432) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3432) most significant half

s_add_u32 m0, m0, 4096 // (op 3432) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3432) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3432) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3432) most significant half

s_add_u32 m0, m0, 4096 // (op 3432) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3432) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 3432) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3432) most significant half

s_add_u32 m0, m0, 4096 // (op 3432) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3432) Load value direct to lds






buffer_load_dword v18, v19, s[60:63], 0 offen // (op 3147) Load value


v_mov_b32 v23, v14 // (op 3150) reg expression

buffer_load_dwordx3 v[28:30], v23, s[48:51], 0 offen // (op 3150) Load value


v_mov_b64 v[24:25], s[28:29] // (op 3154) convert

s_lshl_b32 s0, s28, 2 // (op 3154) 
v_add_u32 v23, v20, s0 // (op 3154) 

v_mov_b32 v27, v23 // (op 3154) 
buffer_load_ubyte v31, v27, s[56:59], 0 offen offset: 0 // (op 3154) Load value




v_and_b32 v20, -8, v1 // (op 3957) 
v_sub_u32 v27, v1, v20 // (op 3957) 

v_lshlrev_b32 v20, 4, v27 // (op 3957) 

v_lshrrev_b32 v27, 3, v1 // (op 3957) 
v_lshl_add_u32 v23, v27, 7, v20 // (op 3957) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 3957) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 3437) 
v_mov_b64 v[24:25], s[6:7] // (op 3437) convert


v_mov_b64 v[32:33], v[12:13] // (op 3437) 
s_add_u32 s1, 32768, s0 // (op 3437) 
s_mov_b32 m0, s1 // (op 3437) reg expression
s_mov_b32 s6, 128 // (op 3437) call()
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3437) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3437) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3437) most significant half

s_add_u32 m0, m0, 4096 // (op 3437) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3437) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3437) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3437) most significant half

s_add_u32 m0, m0, 4096 // (op 3437) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3437) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3437) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3437) most significant half

s_add_u32 m0, m0, 4096 // (op 3437) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3437) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3437) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3437) most significant half

s_add_u32 m0, m0, 4096 // (op 3437) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3437) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3437) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3437) most significant half

s_add_u32 m0, m0, 4096 // (op 3437) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3437) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3437) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3437) most significant half

s_add_u32 m0, m0, 4096 // (op 3437) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3437) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v24 // (op 3437) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 3437) most significant half

s_add_u32 m0, m0, 4096 // (op 3437) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 3437) Load value direct to lds









buffer_load_dword v20, v19, s[60:63], 0 offen offset:4 // (op 3161) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 3164) 
s_mul_i32 s7, s1, 6 // (op 3164) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 3164) least significant
s_mul_hi_u32 s10, s0, 6 // (op 3164) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3164) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 3164) 

v_add_u32 v19, v14, s0 // (op 3164) 


buffer_load_dwordx3 v[32:34], v19, s[48:51], 0 offen // (op 3164) Load value


v_lshlrev_b32 v19, 2, v1 // (op 3871) 



v_add_u32 v27, 65536, v19 // (op 3167) 
s_waitcnt vmcnt(12)
ds_write_b32 v27, v18 // (op 3167) Store local data 


v_and_b32 v19, -2, v1 // (op 3761) 
v_sub_u32 v27, v1, v19 // (op 3761) 

v_lshlrev_b32 v19, 4, v27 // (op 3761) 

v_lshrrev_b32 v27, 1, v1 // (op 3761) 
v_lshl_add_u32 v12, v27, 5, v19 // (op 3761) 


v_mul_lo_u32 v19, v12, 6 // (op 3761) 
v_lshrrev_b32 v27, 3, v19 // (op 3761) 

v_lshrrev_b32 v19, 4, v12 // (op 3761) 

v_lshlrev_b32 v12, 2, v19 // (op 3761) 

v_add_u32 v19, v27, v12 // (op 3761) 





v_add_u32 v27, 66560, v19 // (op 3168) 
s_waitcnt vmcnt(11)
ds_write_b96 v27, v[28:30] // (op 3168) Store local data 


v_and_b32 v19, -2, v1 // (op 3732) 
v_sub_u32 v27, v1, v19 // (op 3732) 

v_lshlrev_b32 v19, 4, v27 // (op 3732) 

v_lshrrev_b32 v27, 1, v1 // (op 3732) 
v_lshl_add_u32 v12, v27, 5, v19 // (op 3732) 


v_mul_lo_u32 v19, v12, 6 // (op 3732) 
v_lshrrev_b32 v27, 3, v19 // (op 3732) 

v_lshrrev_b32 v19, 4, v12 // (op 3732) 

v_lshlrev_b32 v12, 2, v19 // (op 3732) 

v_add_u32 v19, v27, v12 // (op 3732) 


v_and_b32 v27, -8, v1 // (op 3785) 
v_sub_u32 v12, v1, v27 // (op 3785) 

v_lshlrev_b32 v27, 4, v12 // (op 3785) 

v_lshrrev_b32 v12, 3, v1 // (op 3785) 
v_lshl_add_u32 v13, v12, 7, v27 // (op 3785) 


s_nop 0
v_readfirstlane_b32 s0, v13 // (op 3785) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v27, 2, v1 // (op 3842) 
v_and_b32 v12, -2, v1 // (op 3918) 
v_sub_u32 v13, v1, v12 // (op 3918) 

v_lshlrev_b32 v12, 4, v13 // (op 3918) 

v_lshrrev_b32 v13, 1, v1 // (op 3918) 
v_lshl_add_u32 v14, v13, 5, v12 // (op 3918) 


v_mul_lo_u32 v12, v14, 6 // (op 3918) 
v_lshrrev_b32 v13, 3, v12 // (op 3918) 

v_lshrrev_b32 v12, 4, v14 // (op 3918) 

v_lshlrev_b32 v14, 2, v12 // (op 3918) 

v_add_u32 v12, v13, v14 // (op 3918) 


v_and_b32 v13, -8, v1 // (op 3942) 
v_sub_u32 v14, v1, v13 // (op 3942) 

v_lshlrev_b32 v13, 4, v14 // (op 3942) 

v_lshrrev_b32 v14, 3, v1 // (op 3942) 
v_lshl_add_u32 v23, v14, 7, v13 // (op 3942) 


s_nop 0
v_readfirstlane_b32 s1, v23 // (op 3942) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v13, 2, v1 // (op 3986) 
s_mov_b32 s6, 0 // (op 1248) call()
s_ashr_i64 s[10:11], s[22:23], 7 // (op 1247) 
s_mov_b32 s7, s10 // (op 1247) convert

s_and_b32 s10, s7, 4294967294 // (op 1247) 

s_cmp_lt_u32 s6, s10 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1247 // (op 1247) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1247 if false)
RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1247:
s_add_u32 s6, s6, 2 // (op 1249) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4062) 



v_add_u32 v14, 66560, v0 // (op 67) 
ds_read_b96_tr_b6 v[36:38], v14 // (op 67) Transpose load local data 

v_add_u32 v14, 67072, v0 // (op 67) 
ds_read_b96_tr_b6 v[39:41], v14 // (op 67) Transpose load local data 

s_barrier  // (op 4075) 



ds_read_b128 v[42:45], v5 // (op 72) Load local data 
ds_read_b128 v[46:49], v5 offset:64 // (op 72) Load local data 
s_barrier  // (op 4088) 



v_add_u32 v14, 65536, v4 // (op 81) 
ds_read_u8 v35, v14 // (op 81) Load local data 




ds_read_b128 v[50:53], v5 offset:4096 // (op 343) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 84) 
ds_read_b128 v[54:57], v5 offset:4160 // (op 343) Load local data 





v_add_u32 v14, 65664, v4 // (op 346) 
ds_read_u8 v35, v14 // (op 346) Load local data 




ds_read_b128 v[42:45], v5 offset:8192 // (op 499) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[50:57], a[4:7], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 347) 
ds_read_b128 v[46:49], v5 offset:8256 // (op 499) Load local data 





v_add_u32 v14, 65792, v4 // (op 502) 
ds_read_u8 v35, v14 // (op 502) Load local data 




ds_read_b128 v[50:53], v5 offset:12288 // (op 655) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[36:41], v[42:49], a[8:11], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 503) 
ds_read_b128 v[54:57], v5 offset:12352 // (op 655) Load local data 





v_add_u32 v14, 65920, v4 // (op 658) 
ds_read_u8 v35, v14 // (op 658) Load local data 




ds_read_b128 v[42:45], v5 offset:16384 // (op 811) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[36:41], v[50:57], a[12:15], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 659) 
ds_read_b128 v[46:49], v5 offset:16448 // (op 811) Load local data 





v_add_u32 v14, 66048, v4 // (op 814) 
ds_read_u8 v35, v14 // (op 814) Load local data 




ds_read_b128 v[50:53], v5 offset:20480 // (op 967) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[36:41], v[42:49], a[16:19], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 815) 
ds_read_b128 v[54:57], v5 offset:20544 // (op 967) Load local data 





v_add_u32 v14, 66176, v4 // (op 970) 
ds_read_u8 v35, v14 // (op 970) Load local data 




ds_read_b128 v[42:45], v5 offset:24576 // (op 1123) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[36:41], v[50:57], a[20:23], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 971) 
ds_read_b128 v[46:49], v5 offset:24640 // (op 1123) Load local data 





v_add_u32 v14, 66304, v4 // (op 1126) 
ds_read_u8 v35, v14 // (op 1126) Load local data 




ds_read_b128 v[50:53], v5 offset:28672 // (op 1279) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[36:41], v[42:49], a[24:27], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1127) 
ds_read_b128 v[54:57], v5 offset:28736 // (op 1279) Load local data 





v_add_u32 v14, 66432, v4 // (op 1282) 
ds_read_u8 v35, v14 // (op 1282) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[36:41], v[50:57], a[28:31], v26, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1283) 




v_mov_b64 v[24:25], s[28:29] // (op 77) convert

v_mov_b32 v14, v21 // (op 77) 
s_lshl_b32 s7, s28, 3 // (op 77) 
v_add_u32 v23, v14, s7 // (op 77) 

v_mov_b32 v35, v23 // (op 77) 
buffer_load_ubyte v26, v35, s[56:59], 0 offen offset: 0 // (op 77) Load value



s_barrier  // (op 4078) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 3415) 
v_mov_b64 v[24:25], s[10:11] // (op 3415) convert


v_mov_b64 v[36:37], v[16:17] // (op 3415) 
v_mov_b64 v[38:39], v[36:37] // (op 3415) 
s_mov_b32 m0, s0 // (op 3415) reg expression
s_mov_b32 s7, 256 // (op 3415) call()
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3415) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3415) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3415) most significant half

s_add_u32 m0, m0, 4096 // (op 3415) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3415) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3415) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3415) most significant half

s_add_u32 m0, m0, 4096 // (op 3415) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3415) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3415) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3415) most significant half

s_add_u32 m0, m0, 4096 // (op 3415) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3415) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3415) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3415) most significant half

s_add_u32 m0, m0, 4096 // (op 3415) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3415) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3415) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3415) most significant half

s_add_u32 m0, m0, 4096 // (op 3415) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3415) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3415) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3415) most significant half

s_add_u32 m0, m0, 4096 // (op 3415) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3415) Load value direct to lds
v_add_co_u32 v38, s[10:11], v38, v24 // (op 3415) least significant half
v_addc_co_u32 v39, s[10:11], 0, v39, s[10:11] // (op 3415) most significant half

s_add_u32 m0, m0, 4096 // (op 3415) 
buffer_load_dwordx4 v38, s[52:55], s7 offen offset: 0 lds // (op 3415) Load value direct to lds





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

s_barrier  // (op 4117) 



v_add_u32 v24, 70656, v13 // (op 240) 
s_waitcnt vmcnt(12)
ds_write_b32 v24, v20 // (op 240) Store local data 

s_barrier  // (op 4105) 



v_add_u32 v24, 71680, v12 // (op 229) 
s_waitcnt vmcnt(11)
ds_write_b96 v24, v[32:34] // (op 229) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4102) 



v_add_u32 v24, 71680, v3 // (op 231) 
ds_read_b96_tr_b6 v[38:40], v24 // (op 231) Transpose load local data 

v_add_u32 v24, 72192, v3 // (op 231) 
ds_read_b96_tr_b6 v[41:43], v24 // (op 231) Transpose load local data 

s_barrier  // (op 4108) 
s_barrier  // (op 4152) 



ds_read_b128 v[44:47], v8 offset:32768 // (op 233) Load local data 
ds_read_b128 v[48:51], v8 offset:32832 // (op 233) Load local data 
s_barrier  // (op 4114) 



v_add_u32 v24, 70656, v7 // (op 237) 
ds_read_u8 v25, v24 // (op 237) Load local data 




ds_read_b128 v[52:55], v8 offset:36864 // (op 352) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:43], v[44:51], a[0:3], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 238) 
ds_read_b128 v[56:59], v8 offset:36928 // (op 352) Load local data 





v_add_u32 v24, 70784, v7 // (op 355) 
ds_read_u8 v25, v24 // (op 355) Load local data 




ds_read_b128 v[44:47], v8 offset:40960 // (op 508) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:43], v[52:59], a[4:7], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 348) 
ds_read_b128 v[48:51], v8 offset:41024 // (op 508) Load local data 





v_add_u32 v24, 70912, v7 // (op 511) 
ds_read_u8 v25, v24 // (op 511) Load local data 




ds_read_b128 v[52:55], v8 offset:45056 // (op 664) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:43], v[44:51], a[8:11], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 504) 
ds_read_b128 v[56:59], v8 offset:45120 // (op 664) Load local data 





v_add_u32 v24, 71040, v7 // (op 667) 
ds_read_u8 v25, v24 // (op 667) Load local data 




ds_read_b128 v[44:47], v8 offset:49152 // (op 820) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:43], v[52:59], a[12:15], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 660) 
ds_read_b128 v[48:51], v8 offset:49216 // (op 820) Load local data 





v_add_u32 v24, 71168, v7 // (op 823) 
ds_read_u8 v25, v24 // (op 823) Load local data 




ds_read_b128 v[52:55], v8 offset:53248 // (op 976) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[38:43], v[44:51], a[16:19], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 816) 
ds_read_b128 v[56:59], v8 offset:53312 // (op 976) Load local data 





v_add_u32 v24, 71296, v7 // (op 979) 
ds_read_u8 v25, v24 // (op 979) Load local data 




ds_read_b128 v[44:47], v8 offset:57344 // (op 1132) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[38:43], v[52:59], a[20:23], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 972) 
ds_read_b128 v[48:51], v8 offset:57408 // (op 1132) Load local data 





v_add_u32 v24, 71424, v7 // (op 1135) 
ds_read_u8 v25, v24 // (op 1135) Load local data 




ds_read_b128 v[52:55], v8 offset:61440 // (op 1288) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[38:43], v[44:51], a[24:27], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1128) 
ds_read_b128 v[56:59], v8 offset:61504 // (op 1288) Load local data 





v_add_u32 v24, 71552, v7 // (op 1291) 
ds_read_u8 v25, v24 // (op 1291) Load local data 

s_barrier  // (op 4224) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[38:43], v[52:59], a[28:31], v31, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1284) 




v_mov_b64 v[24:25], s[28:29] // (op 235) convert

s_lshl_b32 s7, s28, 2 // (op 235) 
s_mul_i32 s10, 3, s7 // (op 235) 

v_add_u32 v38, v14, s10 // (op 235) 

v_mov_b32 v39, v38 // (op 235) 
buffer_load_ubyte v31, v39, s[56:59], 0 offen offset: 0 // (op 235) Load value




s_barrier  // (op 4111) 
s_lshl_b32 s7, s28, 3 // (op 3622) 

v_add_u32 v21, v21, s7 // (op 3622) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 3427) 
v_mov_b64 v[24:25], s[10:11] // (op 3427) convert


v_mov_b64 v[38:39], v[36:37] // (op 3427) 
s_add_u32 s7, 32768, s1 // (op 3427) 
s_mov_b32 m0, s7 // (op 3427) reg expression
s_mov_b32 s10, 384 // (op 3427) call()
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3427) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3427) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3427) most significant half

s_add_u32 m0, m0, 4096 // (op 3427) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3427) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3427) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3427) most significant half

s_add_u32 m0, m0, 4096 // (op 3427) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3427) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3427) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3427) most significant half

s_add_u32 m0, m0, 4096 // (op 3427) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3427) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3427) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3427) most significant half

s_add_u32 m0, m0, 4096 // (op 3427) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3427) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3427) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3427) most significant half

s_add_u32 m0, m0, 4096 // (op 3427) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3427) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3427) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3427) most significant half

s_add_u32 m0, m0, 4096 // (op 3427) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3427) Load value direct to lds
v_add_co_u32 v38, s[14:15], v38, v24 // (op 3427) least significant half
v_addc_co_u32 v39, s[14:15], 0, v39, s[14:15] // (op 3427) most significant half

s_add_u32 m0, m0, 4096 // (op 3427) 
buffer_load_dwordx4 v38, s[52:55], s10 offen offset: 0 lds // (op 3427) Load value direct to lds








v_add_co_u32 v16, vcc, 256, v16 // (op 3558) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 3558) most significant half
buffer_load_dword v20, v35, s[60:63], 0 offen offset:12 // (op 242) Load value



v_add_u32 v22, 8, v22 // (op 3686) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 228) 
s_mul_i32 s15, s11, 6 // (op 228) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 228) least significant
s_mul_hi_u32 s7, s10, 6 // (op 228) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 228) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 228) 

s_mul_i32 s7, 3, s10 // (op 228) 

v_add_u32 v14, v23, s7 // (op 228) 


buffer_load_dwordx3 v[32:34], v14, s[48:51], 0 offen // (op 228) Load value


s_barrier  // (op 4091) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 3494) 



v_add_u32 v14, 65536, v27 // (op 26) 
s_mul_i32 s15, s11, 6 // (op 3494) most significant: low of high * low
s_mul_i32 s14, s10, 6 // (op 3494) least significant
s_mul_hi_u32 s7, s10, 6 // (op 3494) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 3494) most significant: sum


s_lshr_b64 s[10:11], s[14:15], 3 // (op 3494) 

s_lshl_b32 s7, s10, 1 // (op 3494) 

v_add_u32 v15, v15, s7 // (op 3494) 

s_waitcnt vmcnt(12)
ds_write_b32 v14, v18 // (op 26) Store local data 

s_barrier  // (op 4065) 



v_add_u32 v14, 66560, v19 // (op 6) 
s_waitcnt vmcnt(11)
ds_write_b96 v14, v[28:30] // (op 6) Store local data 

s_ashr_i64 s[10:11], s[22:23], 7 // (op 1247) 
s_mov_b32 s7, s10 // (op 1247) convert

s_and_b32 s10, s7, 4294967294 // (op 1247) 

s_cmp_lt_u32 s6, s10 // (op 1247) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1247 // (op 1247) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoop_1247 if true)
RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoop_1247:














s_barrier  // (op 4206) 
v_and_b32 v12, -2, v1 // (op 3450) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3450) 
v_sub_u32 v13, v1, v12 // (op 3450) 

s_lshr_b32 s6, s0, 1 // (op 3450) 

v_lshlrev_b32 v12, 4, v13 // (op 3450) 

v_lshl_add_u32 v13, s16, 5, v12 // (op 3450) 

v_mov_b32 v15, 0 // (op 3450) convert
v_mov_b32 v14, v13 // (op 3450) convert

v_lshrrev_b32 v12, 1, v1 // (op 3450) 
v_lshl_add_u32 v13, s6, 8, v12 // (op 3450) 


v_mul_lo_u32 v17, v13, s13 // (op 3450) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 3450) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 3450) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3450) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 3450) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 3450) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 3450) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 3450) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 3450) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 3450) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 3450) 

v_mov_b32 v14, v12 // (op 3450) convert

s_mul_i32 s1, s23, 6 // (op 3452) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 3452) least significant
s_mul_hi_u32 s6, s22, 6 // (op 3452) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3452) most significant: sum

s_mul_i32 s7, s12, s1 // (op 3452) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 3452) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 3452) least significant
s_mul_hi_u32 s11, s12, s0 // (op 3452) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 3452) most significant: sum
s_add_u32 s7, s7, s11 // (op 3452) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 3452) 

s_mov_b32 s48, s20 // (op 3452) reg expression
s_mov_b32 s49, s21 // (op 3452) reg expression
s_mov_b32 s50, s0 // (op 3452) reg expression
s_mov_b32 s51, 131072 // (op 3452) call()

v_and_b32 v12, -8, v1 // (op 3524) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3524) 
v_sub_u32 v13, v1, v12 // (op 3524) 

s_lshr_b32 s6, s0, 1 // (op 3524) 

v_lshlrev_b32 v12, 4, v13 // (op 3524) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 3524) 


v_mov_b32 v19, 0 // (op 3524) convert
v_mov_b32 v18, v13 // (op 3524) convert

v_lshrrev_b32 v12, 3, v1 // (op 3524) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 3524) 

v_mul_lo_u32 v17, v13, s25 // (op 3524) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 3524) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 3524) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3524) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v16 // (op 3524) least significant half
v_addc_co_u32 v13, s[0:1], v19, v17, s[0:1] // (op 3524) most significant half



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

v_lshrrev_b32 v15, 7, v1 // (op 3588) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3588) 
v_and_b32 v16, 15, v1 // (op 3588) 
s_lshr_b32 s6, s0, 1 // (op 3588) 

v_lshl_add_u32 v17, v15, 4, v16 // (op 3588) 


v_lshl_add_u32 v15, s16, 5, v17 // (op 3588) 

v_and_b32 v16, 63, v1 // (op 3588) 
v_lshrrev_b32 v17, 4, v16 // (op 3588) 

v_lshl_add_u32 v16, s6, 3, v17 // (op 3588) 


v_mul_lo_u32 v17, v16, s28 // (op 3588) 

v_add_u32 v16, v15, v17 // (op 3588) 


s_mul_i32 s1, s28, s33 // (op 3590) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3590) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3590) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3590) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3590) most significant: sum
s_add_u32 s1, s1, s7 // (op 3590) most significant: sum


s_mov_b32 s56, s30 // (op 3590) reg expression
s_mov_b32 s57, s31 // (op 3590) reg expression
s_mov_b32 s58, s0 // (op 3590) reg expression
s_mov_b32 s59, 131072 // (op 3590) call()

v_lshl_add_u32 v15, s17, 8, v1 // (op 3652) 
v_mul_lo_u32 v17, v15, s34 // (op 3652) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 3652) 
s_lshr_b32 s6, s0, 1 // (op 3652) 

v_lshl_add_u32 v15, s6, 3, v17 // (op 3652) 


s_mul_i32 s1, s34, s9 // (op 3654) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 3654) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 3654) least significant
s_mul_hi_u32 s7, s34, s8 // (op 3654) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3654) most significant: sum
s_add_u32 s1, s1, s7 // (op 3654) most significant: sum


s_mov_b32 s60, s36 // (op 3654) reg expression
s_mov_b32 s61, s37 // (op 3654) reg expression
s_mov_b32 s62, s0 // (op 3654) reg expression
s_mov_b32 s63, 131072 // (op 3654) call()

v_and_b32 v17, -2, v1 // (op 3745) 
v_sub_u32 v18, v1, v17 // (op 3745) 

v_lshlrev_b32 v17, 4, v18 // (op 3745) 

v_lshrrev_b32 v18, 1, v1 // (op 3745) 
v_lshl_add_u32 v19, v18, 5, v17 // (op 3745) 


v_mul_lo_u32 v17, v19, 6 // (op 3745) 
v_lshrrev_b32 v18, 3, v17 // (op 3745) 

v_lshrrev_b32 v17, 4, v19 // (op 3745) 

v_lshlrev_b32 v19, 2, v17 // (op 3745) 

v_add_u32 v17, v18, v19 // (op 3745) 


v_and_b32 v18, -8, v1 // (op 3798) 
v_sub_u32 v19, v1, v18 // (op 3798) 

v_lshlrev_b32 v18, 4, v19 // (op 3798) 

v_lshrrev_b32 v19, 3, v1 // (op 3798) 
v_lshl_add_u32 v21, v19, 7, v18 // (op 3798) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 3798) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v18, 2, v1 // (op 3855) 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 1253) 
s_and_b32 s1, s6, 4294967294 // (op 1253) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 1252) 
s_mov_b32 s10, s6 // (op 1252) convert

s_cmp_lt_u32 s1, s10 // (op 1252) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1252 // (op 1252) Condition: Top (jump to RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1252 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1252:


s_add_u32 s1, s1, 1 // (op 1254) 
v_mov_b32 v19, v14 // (op 170) 
v_mov_b32 v20, v19 // (op 170) reg expression

buffer_load_dwordx3 v[22:24], v20, s[48:51], 0 offen // (op 170) Load value


s_lshl_b64 s[6:7], s[12:13], 7 // (op 3440) 
s_mul_i32 s11, s7, 6 // (op 3440) most significant: low of high * low
s_mul_i32 s10, s6, 6 // (op 3440) least significant
s_mul_hi_u32 s14, s6, 6 // (op 3440) most significant: high of low * low
s_add_u32 s11, s11, s14 // (op 3440) most significant: sum


s_lshr_b64 s[6:7], s[10:11], 3 // (op 3440) 

v_add_u32 v14, v14, s6 // (op 3440) 

s_barrier  // (op 4068) 
s_barrier  // (op 4081) 




v_add_u32 v19, 66560, v17 // (op 171) 
s_lshl_b64 s[6:7], s[24:25], 5 // (op 3420) 
v_mov_b64 v[20:21], s[6:7] // (op 3420) convert


v_mov_b64 v[26:27], v[12:13] // (op 3420) 
v_mov_b64 v[28:29], v[26:27] // (op 3420) 
s_mov_b32 m0, s0 // (op 3420) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3420) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3420) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3420) most significant half

s_add_u32 m0, m0, 4096 // (op 3420) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3420) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3420) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3420) most significant half

s_add_u32 m0, m0, 4096 // (op 3420) 
s_waitcnt vmcnt(2)
ds_write_b96 v19, v[22:24] // (op 171) Store local data 

buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3420) Load value direct to lds

v_add_co_u32 v28, s[6:7], v28, v20 // (op 3420) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3420) most significant half

s_add_u32 m0, m0, 4096 // (op 3420) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3420) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3420) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3420) most significant half

s_add_u32 m0, m0, 4096 // (op 3420) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3420) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3420) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3420) most significant half

s_add_u32 m0, m0, 4096 // (op 3420) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3420) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3420) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3420) most significant half

s_add_u32 m0, m0, 4096 // (op 3420) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3420) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v20 // (op 3420) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 3420) most significant half

s_add_u32 m0, m0, 4096 // (op 3420) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 3420) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 3516) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 3516) most significant half


v_mov_b32 v19, v15 // (op 184) 

buffer_load_dword v20, v19, s[60:63], 0 offen // (op 184) Load value

v_add_u32 v15, 4, v15 // (op 3644) 
s_barrier  // (op 4094) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 4071) 






v_add_u32 v19, 65536, v18 // (op 182) 
v_add_u32 v21, 66560, v0 // (op 173) 
ds_read_b96_tr_b6 v[22:24], v21 // (op 173) Transpose load local data 

v_add_u32 v21, 67072, v0 // (op 173) 
ds_read_b96_tr_b6 v[25:27], v21 // (op 173) Transpose load local data 

s_waitcnt vmcnt(0)
ds_write_b32 v19, v20 // (op 182) Store local data 


s_barrier  // (op 4084) 



ds_read_b128 v[28:31], v5 // (op 175) Load local data 
ds_read_b128 v[32:35], v5 offset:64 // (op 175) Load local data 

v_mov_b64 v[20:21], s[28:29] // (op 177) convert

v_mov_b32 v19, v16 // (op 177) 
v_mov_b32 v36, v19 // (op 177) reg expression
v_mov_b32 v37, v36 // (op 177) 
buffer_load_ubyte v38, v37, s[56:59], 0 offen offset: 0 // (op 177) Load value




s_lshl_b32 s6, s28, 2 // (op 3580) 
v_add_u32 v16, v16, s6 // (op 3580) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 4098) 



v_add_u32 v19, 65536, v4 // (op 179) 
ds_read_u8 v20, v19 // (op 179) Load local data 




s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[28:35], a[0:3], v38, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 180) 
ds_read_b128 v[40:43], v5 offset:4096 // (op 324) Load local data 


ds_read_b128 v[44:47], v5 offset:4160 // (op 324) Load local data 



v_add_u32 v39, 65664, v4 // (op 327) 
ds_read_u8 v19, v39 // (op 327) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:27], v[40:47], a[4:7], v38, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 328) 
ds_read_b128 v[28:31], v5 offset:8192 // (op 480) Load local data 


ds_read_b128 v[32:35], v5 offset:8256 // (op 480) Load local data 



v_add_u32 v19, 65792, v4 // (op 483) 
ds_read_u8 v20, v19 // (op 483) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:27], v[28:35], a[8:11], v38, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 484) 
ds_read_b128 v[40:43], v5 offset:12288 // (op 636) Load local data 


ds_read_b128 v[44:47], v5 offset:12352 // (op 636) Load local data 



v_add_u32 v39, 65920, v4 // (op 639) 
ds_read_u8 v19, v39 // (op 639) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:27], v[40:47], a[12:15], v38, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 640) 
ds_read_b128 v[28:31], v5 offset:16384 // (op 792) Load local data 


ds_read_b128 v[32:35], v5 offset:16448 // (op 792) Load local data 



v_add_u32 v19, 66048, v4 // (op 795) 
ds_read_u8 v20, v19 // (op 795) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:27], v[28:35], a[16:19], v38, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 796) 
ds_read_b128 v[40:43], v5 offset:20480 // (op 948) Load local data 


ds_read_b128 v[44:47], v5 offset:20544 // (op 948) Load local data 



v_add_u32 v39, 66176, v4 // (op 951) 
ds_read_u8 v19, v39 // (op 951) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:27], v[40:47], a[20:23], v38, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 952) 
ds_read_b128 v[28:31], v5 offset:24576 // (op 1104) Load local data 


ds_read_b128 v[32:35], v5 offset:24640 // (op 1104) Load local data 



v_add_u32 v19, 66304, v4 // (op 1107) 
ds_read_u8 v20, v19 // (op 1107) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:27], v[28:35], a[24:27], v38, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1108) 
ds_read_b128 v[40:43], v5 offset:28672 // (op 1260) Load local data 


ds_read_b128 v[44:47], v5 offset:28736 // (op 1260) Load local data 



v_add_u32 v39, 66432, v4 // (op 1263) 
ds_read_u8 v19, v39 // (op 1263) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:27], v[40:47], a[28:31], v38, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 1264) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 1252) 
s_mov_b32 s10, s6 // (op 1252) convert

s_cmp_lt_u32 s1, s10 // (op 1252) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1252 // (op 1252) Condition: Bottom (jump to RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopTop_KLoopTail_1252 if true)
RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ForLoopBottom_KLoopTail_1252:






















s_barrier  // (op 4210) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_125
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

v_cvt_f16_f32 v12, v16 // (op 1578) 
v_cvt_f16_f32 v13, v17 // (op 1578) 
v_cvt_f16_f32 v14, v18 // (op 1578) 
v_cvt_f16_f32 v15, v19 // (op 1578) 


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

v_cvt_f16_f32 v12, v16 // (op 1592) 
v_cvt_f16_f32 v13, v17 // (op 1592) 
v_cvt_f16_f32 v14, v18 // (op 1592) 
v_cvt_f16_f32 v15, v19 // (op 1592) 


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

v_cvt_f16_f32 v12, v16 // (op 1606) 
v_cvt_f16_f32 v13, v17 // (op 1606) 
v_cvt_f16_f32 v14, v18 // (op 1606) 
v_cvt_f16_f32 v15, v19 // (op 1606) 


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

v_cvt_f16_f32 v12, v16 // (op 1620) 
v_cvt_f16_f32 v13, v17 // (op 1620) 
v_cvt_f16_f32 v14, v18 // (op 1620) 
v_cvt_f16_f32 v15, v19 // (op 1620) 


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

v_cvt_f16_f32 v12, v16 // (op 1634) 
v_cvt_f16_f32 v13, v17 // (op 1634) 
v_cvt_f16_f32 v14, v18 // (op 1634) 
v_cvt_f16_f32 v15, v19 // (op 1634) 


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

v_cvt_f16_f32 v12, v16 // (op 1648) 
v_cvt_f16_f32 v13, v17 // (op 1648) 
v_cvt_f16_f32 v14, v18 // (op 1648) 
v_cvt_f16_f32 v15, v19 // (op 1648) 


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

v_cvt_f16_f32 v12, v16 // (op 1662) 
v_cvt_f16_f32 v13, v17 // (op 1662) 
v_cvt_f16_f32 v14, v18 // (op 1662) 
v_cvt_f16_f32 v15, v19 // (op 1662) 


v_pack_b32_f16 v4, v12, v13 // (op 1663) 
v_pack_b32_f16 v5, v14, v15 // (op 1663) 

s_lshl_b32 s0, s4, 6 // (op 1663) 
s_mul_i32 s1, 7, s0 // (op 1663) 

v_add_u32 v0, v9, s1 // (op 1663) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1663) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalFalse_0_beta_125:


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
v_cvt_f32_f16 v12, v4 // (op 1718) 
v_lshrrev_b32 v13, 16, v4 // (op 1718) 
v_cvt_f32_f16 v13, v13 // (op 1718) 
v_mul_f32 v20, v10, v12 // (op 1718) 
v_mul_f32 v21, v10, v13 // (op 1718) 
v_cvt_f32_f16 v12, v5 // (op 1718) 
v_lshrrev_b32 v13, 16, v5 // (op 1718) 
v_cvt_f32_f16 v13, v13 // (op 1718) 
v_mul_f32 v22, v10, v12 // (op 1718) 
v_mul_f32 v23, v10, v13 // (op 1718) 


v_add_f32 v12, v20, v16 // (op 1714) 
v_add_f32 v13, v21, v17 // (op 1714) 
v_add_f32 v14, v22, v18 // (op 1714) 
v_add_f32 v15, v23, v19 // (op 1714) 


v_cvt_f16_f32 v16, v12 // (op 1715) 
v_cvt_f16_f32 v17, v13 // (op 1715) 
v_cvt_f16_f32 v18, v14 // (op 1715) 
v_cvt_f16_f32 v19, v15 // (op 1715) 


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
v_cvt_f32_f16 v12, v4 // (op 1732) 
v_lshrrev_b32 v13, 16, v4 // (op 1732) 
v_cvt_f32_f16 v13, v13 // (op 1732) 
v_mul_f32 v20, v10, v12 // (op 1732) 
v_mul_f32 v21, v10, v13 // (op 1732) 
v_cvt_f32_f16 v12, v5 // (op 1732) 
v_lshrrev_b32 v13, 16, v5 // (op 1732) 
v_cvt_f32_f16 v13, v13 // (op 1732) 
v_mul_f32 v22, v10, v12 // (op 1732) 
v_mul_f32 v23, v10, v13 // (op 1732) 


v_add_f32 v12, v20, v16 // (op 1728) 
v_add_f32 v13, v21, v17 // (op 1728) 
v_add_f32 v14, v22, v18 // (op 1728) 
v_add_f32 v15, v23, v19 // (op 1728) 


v_cvt_f16_f32 v16, v12 // (op 1729) 
v_cvt_f16_f32 v17, v13 // (op 1729) 
v_cvt_f16_f32 v18, v14 // (op 1729) 
v_cvt_f16_f32 v19, v15 // (op 1729) 


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
v_cvt_f32_f16 v12, v4 // (op 1746) 
v_lshrrev_b32 v13, 16, v4 // (op 1746) 
v_cvt_f32_f16 v13, v13 // (op 1746) 
v_mul_f32 v20, v10, v12 // (op 1746) 
v_mul_f32 v21, v10, v13 // (op 1746) 
v_cvt_f32_f16 v12, v5 // (op 1746) 
v_lshrrev_b32 v13, 16, v5 // (op 1746) 
v_cvt_f32_f16 v13, v13 // (op 1746) 
v_mul_f32 v22, v10, v12 // (op 1746) 
v_mul_f32 v23, v10, v13 // (op 1746) 


v_add_f32 v12, v20, v16 // (op 1742) 
v_add_f32 v13, v21, v17 // (op 1742) 
v_add_f32 v14, v22, v18 // (op 1742) 
v_add_f32 v15, v23, v19 // (op 1742) 


v_cvt_f16_f32 v16, v12 // (op 1743) 
v_cvt_f16_f32 v17, v13 // (op 1743) 
v_cvt_f16_f32 v18, v14 // (op 1743) 
v_cvt_f16_f32 v19, v15 // (op 1743) 


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
v_cvt_f32_f16 v12, v4 // (op 1760) 
v_lshrrev_b32 v13, 16, v4 // (op 1760) 
v_cvt_f32_f16 v13, v13 // (op 1760) 
v_mul_f32 v20, v10, v12 // (op 1760) 
v_mul_f32 v21, v10, v13 // (op 1760) 
v_cvt_f32_f16 v12, v5 // (op 1760) 
v_lshrrev_b32 v13, 16, v5 // (op 1760) 
v_cvt_f32_f16 v13, v13 // (op 1760) 
v_mul_f32 v22, v10, v12 // (op 1760) 
v_mul_f32 v23, v10, v13 // (op 1760) 


v_add_f32 v12, v20, v16 // (op 1756) 
v_add_f32 v13, v21, v17 // (op 1756) 
v_add_f32 v14, v22, v18 // (op 1756) 
v_add_f32 v15, v23, v19 // (op 1756) 


v_cvt_f16_f32 v16, v12 // (op 1757) 
v_cvt_f16_f32 v17, v13 // (op 1757) 
v_cvt_f16_f32 v18, v14 // (op 1757) 
v_cvt_f16_f32 v19, v15 // (op 1757) 


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
v_cvt_f32_f16 v12, v4 // (op 1774) 
v_lshrrev_b32 v13, 16, v4 // (op 1774) 
v_cvt_f32_f16 v13, v13 // (op 1774) 
v_mul_f32 v20, v10, v12 // (op 1774) 
v_mul_f32 v21, v10, v13 // (op 1774) 
v_cvt_f32_f16 v12, v5 // (op 1774) 
v_lshrrev_b32 v13, 16, v5 // (op 1774) 
v_cvt_f32_f16 v13, v13 // (op 1774) 
v_mul_f32 v22, v10, v12 // (op 1774) 
v_mul_f32 v23, v10, v13 // (op 1774) 


v_add_f32 v12, v20, v16 // (op 1770) 
v_add_f32 v13, v21, v17 // (op 1770) 
v_add_f32 v14, v22, v18 // (op 1770) 
v_add_f32 v15, v23, v19 // (op 1770) 


v_cvt_f16_f32 v16, v12 // (op 1771) 
v_cvt_f16_f32 v17, v13 // (op 1771) 
v_cvt_f16_f32 v18, v14 // (op 1771) 
v_cvt_f16_f32 v19, v15 // (op 1771) 


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
v_cvt_f32_f16 v12, v4 // (op 1788) 
v_lshrrev_b32 v13, 16, v4 // (op 1788) 
v_cvt_f32_f16 v13, v13 // (op 1788) 
v_mul_f32 v20, v10, v12 // (op 1788) 
v_mul_f32 v21, v10, v13 // (op 1788) 
v_cvt_f32_f16 v12, v5 // (op 1788) 
v_lshrrev_b32 v13, 16, v5 // (op 1788) 
v_cvt_f32_f16 v13, v13 // (op 1788) 
v_mul_f32 v22, v10, v12 // (op 1788) 
v_mul_f32 v23, v10, v13 // (op 1788) 


v_add_f32 v12, v20, v16 // (op 1784) 
v_add_f32 v13, v21, v17 // (op 1784) 
v_add_f32 v14, v22, v18 // (op 1784) 
v_add_f32 v15, v23, v19 // (op 1784) 


v_cvt_f16_f32 v16, v12 // (op 1785) 
v_cvt_f16_f32 v17, v13 // (op 1785) 
v_cvt_f16_f32 v18, v14 // (op 1785) 
v_cvt_f16_f32 v19, v15 // (op 1785) 


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
v_cvt_f32_f16 v12, v4 // (op 1802) 
v_lshrrev_b32 v13, 16, v4 // (op 1802) 
v_cvt_f32_f16 v13, v13 // (op 1802) 
v_mul_f32 v20, v10, v12 // (op 1802) 
v_mul_f32 v21, v10, v13 // (op 1802) 
v_cvt_f32_f16 v12, v5 // (op 1802) 
v_lshrrev_b32 v13, 16, v5 // (op 1802) 
v_cvt_f32_f16 v13, v13 // (op 1802) 
v_mul_f32 v22, v10, v12 // (op 1802) 
v_mul_f32 v23, v10, v13 // (op 1802) 


v_add_f32 v12, v20, v16 // (op 1798) 
v_add_f32 v13, v21, v17 // (op 1798) 
v_add_f32 v14, v22, v18 // (op 1798) 
v_add_f32 v15, v23, v19 // (op 1798) 


v_cvt_f16_f32 v16, v12 // (op 1799) 
v_cvt_f16_f32 v17, v13 // (op 1799) 
v_cvt_f16_f32 v18, v14 // (op 1799) 
v_cvt_f16_f32 v19, v15 // (op 1799) 


v_pack_b32_f16 v4, v16, v17 // (op 1800) 
v_pack_b32_f16 v5, v18, v19 // (op 1800) 

s_lshl_b32 s0, s4, 6 // (op 1800) 
s_mul_i32 s1, 7, s0 // (op 1800) 

v_add_u32 v0, v9, s1 // (op 1800) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1800) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_ConditionalBottom_0_beta_125:
















s_endpgm  // End of RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.LRR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end:
.size RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2, .LRR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2_end-RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
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
    DestDataType: h
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
  - .name: RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2
    .symbol: RR_GEMM_NN_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x256x128_UR_2.kd
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

