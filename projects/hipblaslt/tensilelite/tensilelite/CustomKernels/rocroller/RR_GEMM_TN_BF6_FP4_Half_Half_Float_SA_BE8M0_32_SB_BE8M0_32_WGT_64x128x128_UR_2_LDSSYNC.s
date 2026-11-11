// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
.p2align 8
.type RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC,@function
RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 4574) 
v_and_b32 v3, 15, v1 // (op 4574) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4574) 


v_and_b32 v0, 63, v1 // (op 4574) 
v_lshrrev_b32 v3, 4, v0 // (op 4574) 

v_lshlrev_b32 v0, 5, v3 // (op 4574) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 4574) 


v_mul_lo_u32 v0, v3, 6 // (op 4574) 

v_lshrrev_b32 v3, 3, v0 // (op 4574) 

v_lshrrev_b32 v0, 6, v1 // (op 4632) 
v_and_b32 v4, 1, v0 // (op 4632) 

v_and_b32 v0, 15, v1 // (op 4632) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4632) 


v_and_b32 v0, 63, v1 // (op 4632) 
v_lshrrev_b32 v4, 4, v0 // (op 4632) 

v_lshlrev_b32 v0, 5, v4 // (op 4632) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 4632) 


v_lshlrev_b32 v0, 2, v4 // (op 4632) 

v_lshrrev_b32 v4, 3, v0 // (op 4632) 

v_lshrrev_b32 v0, 7, v1 // (op 4690) 
v_and_b32 v5, 15, v1 // (op 4690) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4690) 


v_and_b32 v0, 63, v1 // (op 4690) 
v_lshrrev_b32 v5, 4, v0 // (op 4690) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 4690) 


v_lshrrev_b32 v5, 6, v1 // (op 4748) 
v_and_b32 v6, 1, v5 // (op 4748) 

v_and_b32 v5, 15, v1 // (op 4748) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4748) 


v_and_b32 v5, 63, v1 // (op 4748) 
v_lshrrev_b32 v6, 4, v5 // (op 4748) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 4748) 


v_lshrrev_b32 v6, 7, v1 // (op 4800) 
v_and_b32 v7, 15, v1 // (op 4800) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 4800) 


v_and_b32 v6, 63, v1 // (op 4800) 
v_lshrrev_b32 v7, 4, v6 // (op 4800) 

v_lshlrev_b32 v6, 5, v7 // (op 4800) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 4800) 


v_mul_lo_u32 v6, v7, 6 // (op 4800) 

v_lshrrev_b32 v7, 3, v6 // (op 4800) 

v_lshrrev_b32 v6, 6, v1 // (op 4829) 
v_and_b32 v8, 1, v6 // (op 4829) 

v_and_b32 v6, 15, v1 // (op 4829) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 4829) 


v_and_b32 v6, 63, v1 // (op 4829) 
v_lshrrev_b32 v8, 4, v6 // (op 4829) 

v_lshlrev_b32 v6, 5, v8 // (op 4829) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 4829) 


v_lshlrev_b32 v6, 2, v8 // (op 4829) 

v_lshrrev_b32 v8, 3, v6 // (op 4829) 

v_lshrrev_b32 v6, 7, v1 // (op 4858) 
v_and_b32 v9, 15, v1 // (op 4858) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 4858) 


v_and_b32 v6, 63, v1 // (op 4858) 
v_lshrrev_b32 v9, 4, v6 // (op 4858) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 4858) 


v_lshrrev_b32 v9, 6, v1 // (op 4887) 
v_and_b32 v10, 1, v9 // (op 4887) 

v_and_b32 v9, 15, v1 // (op 4887) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4887) 


v_and_b32 v9, 63, v1 // (op 4887) 
v_lshrrev_b32 v10, 4, v9 // (op 4887) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 4887) 


v_lshrrev_b32 v10, 7, v1 // (op 4512) 
v_and_b32 v11, 63, v1 // (op 4512) 
v_lshrrev_b32 v12, 4, v11 // (op 4512) 
v_lshlrev_b32 v13, 2, v12 // (op 4512) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4512) 


v_lshl_add_u32 v10, s16, 6, v12 // (op 4512) 

v_lshrrev_b32 v12, 2, v11 // (op 4512) 

v_and_b32 v11, 3, v12 // (op 4512) 

v_and_b32 v12, 3, v1 // (op 4512) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 4512) 


v_lshrrev_b32 v11, 6, v1 // (op 4512) 
v_and_b32 v12, 1, v11 // (op 4512) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 4512) 


v_lshl_add_u32 v12, s17, 7, v11 // (op 4512) 

v_mul_lo_u32 v11, v12, s2 // (op 4512) 

v_add_lshl_u32 v12, v10, v11, 1 // (op 4512) 


v_lshrrev_b32 v10, 7, v1 // (op 4769) 
v_and_b32 v11, 63, v1 // (op 4769) 
v_lshrrev_b32 v13, 4, v11 // (op 4769) 
s_mul_i32 s1, s2, s9 // (op 4514) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 4514) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 4514) least significant
s_mul_hi_u32 s19, s2, s8 // (op 4514) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4514) most significant: sum
s_add_u32 s1, s1, s19 // (op 4514) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 4514) 

v_lshlrev_b32 v14, 2, v13 // (op 4769) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 4769) 


v_lshl_add_u32 v10, s16, 6, v13 // (op 4769) 

v_lshrrev_b32 v13, 2, v11 // (op 4769) 

v_and_b32 v11, 3, v13 // (op 4769) 

v_and_b32 v13, 3, v1 // (op 4769) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 4769) 


v_lshrrev_b32 v11, 6, v1 // (op 4769) 
v_and_b32 v13, 1, v11 // (op 4769) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 4769) 


v_lshl_add_u32 v13, s17, 7, v11 // (op 4769) 

v_mul_lo_u32 v11, v13, s4 // (op 4769) 

v_add_lshl_u32 v13, v10, v11, 1 // (op 4769) 


s_mov_b32 s40, s6 // (op 4514) reg expression
s_mov_b32 s41, s7 // (op 4514) reg expression
s_mov_b32 s42, s18 // (op 4514) reg expression
s_mov_b32 s43, 131072 // (op 4514) call()

s_mul_i32 s1, s4, s9 // (op 4771) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 4771) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 4771) least significant
s_mul_hi_u32 s7, s4, s8 // (op 4771) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4771) most significant: sum
s_add_u32 s1, s1, s7 // (op 4771) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 4771) 

s_mov_b32 s44, s10 // (op 4771) reg expression
s_mov_b32 s45, s11 // (op 4771) reg expression
s_mov_b32 s46, s6 // (op 4771) reg expression
s_mov_b32 s47, 131072 // (op 4771) call()


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
v_lshrrev_b32 v14, 2, v1 // (op 4274) 
v_lshl_add_u32 v15, s16, 6, v14 // (op 4274) 

v_and_b32 v14, -4, v1 // (op 4274) 
v_sub_u32 v16, v1, v14 // (op 4274) 

v_lshlrev_b32 v14, 5, v16 // (op 4274) 

v_mov_b32 v17, 0 // (op 4274) convert
v_mov_b32 v16, v14 // (op 4274) convert

v_mul_lo_u32 v19, v15, s13 // (op 4274) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 4274) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 4274) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 4274) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 4274) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 4274) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 4274) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 4274) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 4274) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 4274) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 4274) 

v_mov_b32 v14, v16 // (op 4274) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 4276) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 4276) least significant
s_mul_hi_u32 s6, s22, 6 // (op 4276) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4276) most significant: sum

s_mul_i32 s7, s12, s1 // (op 4276) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 4276) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 4276) least significant
s_mul_hi_u32 s11, s12, s0 // (op 4276) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 4276) most significant: sum
s_add_u32 s7, s7, s11 // (op 4276) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 4276) 

s_mov_b32 s48, s20 // (op 4276) reg expression
s_mov_b32 s49, s21 // (op 4276) reg expression
s_mov_b32 s50, s0 // (op 4276) reg expression
s_mov_b32 s51, 131072 // (op 4276) call()

v_lshrrev_b32 v15, 2, v1 // (op 4293) 
v_lshl_add_u32 v16, s16, 6, v15 // (op 4293) 

v_and_b32 v15, -4, v1 // (op 4293) 
v_sub_u32 v17, v1, v15 // (op 4293) 

v_lshlrev_b32 v15, 5, v17 // (op 4293) 

v_mov_b32 v19, 0 // (op 4293) convert
v_mov_b32 v18, v15 // (op 4293) convert

v_mul_lo_u32 v21, v16, s13 // (op 4293) most significant: low of low * high
v_mul_hi_u32 v15, v16, s12 // (op 4293) most significant: high of low * low
v_mul_lo_u32 v20, v16, s12 // (op 4293) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 4293) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 4293) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 4293) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 4293) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 4293) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 4293) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 4293) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 4293) 

v_mov_b32 v15, v16 // (op 4293) convert

s_mul_i32 s1, s23, 6 // (op 4295) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 4295) least significant
s_mul_hi_u32 s6, s22, 6 // (op 4295) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4295) most significant: sum

s_mul_i32 s7, s12, s1 // (op 4295) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 4295) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 4295) least significant
s_mul_hi_u32 s11, s12, s0 // (op 4295) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 4295) most significant: sum
s_add_u32 s7, s7, s11 // (op 4295) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 4295) 

s_mov_b32 s48, s20 // (op 4295) reg expression
s_mov_b32 s49, s21 // (op 4295) reg expression
s_mov_b32 s50, s0 // (op 4295) reg expression
s_mov_b32 s51, 131072 // (op 4295) call()

v_and_b32 v16, -4, v1 // (op 4338) 
v_sub_u32 v17, v1, v16 // (op 4338) 

v_lshlrev_b32 v16, 5, v17 // (op 4338) 

v_mov_b32 v19, 0 // (op 4338) convert
v_mov_b32 v18, v16 // (op 4338) convert

v_lshrrev_b32 v16, 2, v1 // (op 4338) 
v_lshlrev_b32 v17, 1, v16 // (op 4338) 

v_lshl_add_u32 v16, s17, 7, v17 // (op 4338) 

v_mul_lo_u32 v21, v16, s25 // (op 4338) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 4338) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 4338) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 4338) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 4338) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 4338) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 4338) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 4338) 

v_mov_b32 v16, v18 // (op 4338) convert

s_mul_i32 s1, s24, s9 // (op 4340) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4340) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4340) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4340) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4340) most significant: sum
s_add_u32 s1, s1, s7 // (op 4340) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4340) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 4340) 

s_mov_b32 s52, s26 // (op 4340) reg expression
s_mov_b32 s53, s27 // (op 4340) reg expression
s_mov_b32 s54, s0 // (op 4340) reg expression
s_mov_b32 s55, 131072 // (op 4340) call()

v_and_b32 v17, -4, v1 // (op 4357) 
v_sub_u32 v18, v1, v17 // (op 4357) 

v_lshlrev_b32 v17, 5, v18 // (op 4357) 

v_mov_b32 v19, 0 // (op 4357) convert
v_mov_b32 v18, v17 // (op 4357) convert

v_lshrrev_b32 v17, 2, v1 // (op 4357) 
v_lshlrev_b32 v20, 1, v17 // (op 4357) 

v_lshl_add_u32 v17, s17, 7, v20 // (op 4357) 

v_mul_lo_u32 v21, v17, s25 // (op 4357) most significant: low of low * high
v_mul_hi_u32 v22, v17, s24 // (op 4357) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 4357) least significant: low of low * low
v_add_u32 v21, v21, v22 // (op 4357) most significant: sum


v_add_co_u32 v22, s[0:1], v18, v20 // (op 4357) least significant half
v_addc_co_u32 v23, s[0:1], v19, v21, s[0:1] // (op 4357) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 4357) 


v_lshrrev_b64 v[20:21], 3, v[22:23] // (op 4357) 

v_mov_b32 v17, v20 // (op 4357) convert

s_mul_i32 s1, s24, s9 // (op 4359) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4359) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4359) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4359) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4359) most significant: sum
s_add_u32 s1, s1, s7 // (op 4359) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4359) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 4359) 

s_mov_b32 s52, s26 // (op 4359) reg expression
s_mov_b32 s53, s27 // (op 4359) reg expression
s_mov_b32 s54, s0 // (op 4359) reg expression
s_mov_b32 s55, 131072 // (op 4359) call()

v_lshrrev_b32 v18, 2, v1 // (op 4402) 
v_lshl_add_u32 v19, s16, 6, v18 // (op 4402) 

v_and_b32 v18, -4, v1 // (op 4402) 
v_sub_u32 v20, v1, v18 // (op 4402) 

v_mul_lo_u32 v18, v19, s28 // (op 4402) 
v_add_u32 v18, v18, v20 // (op 4402) 


s_mul_i32 s1, s28, s23 // (op 4404) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4404) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4404) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4404) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4404) most significant: sum
s_add_u32 s1, s1, s7 // (op 4404) most significant: sum


s_mov_b32 s56, s30 // (op 4404) reg expression
s_mov_b32 s57, s31 // (op 4404) reg expression
s_mov_b32 s58, s0 // (op 4404) reg expression
s_mov_b32 s59, 131072 // (op 4404) call()

v_lshrrev_b32 v19, 2, v1 // (op 4421) 
v_lshl_add_u32 v20, s16, 6, v19 // (op 4421) 

v_and_b32 v19, -4, v1 // (op 4421) 
v_sub_u32 v21, v1, v19 // (op 4421) 

v_mul_lo_u32 v19, v20, s28 // (op 4421) 
v_add_u32 v19, v19, v21 // (op 4421) 


s_mul_i32 s1, s28, s23 // (op 4423) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4423) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4423) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4423) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4423) most significant: sum
s_add_u32 s1, s1, s7 // (op 4423) most significant: sum


s_mov_b32 s56, s30 // (op 4423) reg expression
s_mov_b32 s57, s31 // (op 4423) reg expression
s_mov_b32 s58, s0 // (op 4423) reg expression
s_mov_b32 s59, 131072 // (op 4423) call()

v_and_b32 v20, -4, v1 // (op 4466) 
v_sub_u32 v21, v1, v20 // (op 4466) 

v_lshrrev_b32 v20, 2, v1 // (op 4466) 
v_lshlrev_b32 v22, 1, v20 // (op 4466) 

v_lshl_add_u32 v20, s17, 7, v22 // (op 4466) 

v_mul_lo_u32 v22, v20, s32 // (op 4466) 

v_add_u32 v20, v21, v22 // (op 4466) 


s_mul_i32 s1, s32, s9 // (op 4468) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4468) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4468) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4468) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4468) most significant: sum
s_add_u32 s1, s1, s7 // (op 4468) most significant: sum


s_mov_b32 s60, s34 // (op 4468) reg expression
s_mov_b32 s61, s35 // (op 4468) reg expression
s_mov_b32 s62, s0 // (op 4468) reg expression
s_mov_b32 s63, 131072 // (op 4468) call()

v_and_b32 v21, -4, v1 // (op 4485) 
v_sub_u32 v22, v1, v21 // (op 4485) 

v_lshrrev_b32 v21, 2, v1 // (op 4485) 
v_lshlrev_b32 v23, 1, v21 // (op 4485) 

v_lshl_add_u32 v21, s17, 7, v23 // (op 4485) 

v_mul_lo_u32 v23, v21, s32 // (op 4485) 

v_add_u32 v21, v22, v23 // (op 4485) 


s_mul_i32 s1, s32, s9 // (op 4487) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4487) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4487) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4487) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4487) most significant: sum
s_add_u32 s1, s1, s7 // (op 4487) most significant: sum


s_mov_b32 s60, s34 // (op 4487) reg expression
s_mov_b32 s61, s35 // (op 4487) reg expression
s_mov_b32 s62, s0 // (op 4487) reg expression
s_mov_b32 s63, 131072 // (op 4487) call()


s_lshl_b64 s[0:1], s[24:25], 2 // (op 4018) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 4018) 


v_mov_b32 v24, v16 // (op 4018) 
buffer_load_dwordx4 v[26:29], v24, s[52:55], 0 offen // (op 4018) Load value
v_add_u32 v24, v24, v22 // (op 4018) 
buffer_load_dwordx4 v[30:33], v24, s[52:55], 0 offen // (op 4018) Load value



v_mov_b64 v[22:23], s[32:33] // (op 4021) convert

v_mov_b32 v24, v20 // (op 4021) 
buffer_load_ubyte v34, v24, s[60:63], 0 offen offset: 0 // (op 4021) Load value
v_add_u32 v24, v24, v22 // (op 4021) 
buffer_load_ubyte v35, v24, s[60:63], 0 offen offset: 0 // (op 4021) Load value





buffer_load_dwordx4 v[36:39], v14, s[48:51], 0 offen // (op 4024) Load value
buffer_load_dwordx2 v[40:41], v14, s[48:51], 0 offen offset:16 // (op 4024) Load value



buffer_load_ubyte v22, v18, s[56:59], 0 offen offset: 0 // (op 4027) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 4030) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 4030) 


v_mov_b32 v23, v16 // (op 4030) 
buffer_load_dwordx4 v[42:45], v23, s[52:55], 0 offen offset:64 // (op 4030) Load value
v_add_u32 v23, v23, v24 // (op 4030) 
buffer_load_dwordx4 v[46:49], v23, s[52:55], 0 offen offset:64 // (op 4030) Load value




v_mov_b64 v[24:25], s[32:33] // (op 4033) convert

v_mov_b32 v16, v20 // (op 4033) 
buffer_load_ubyte v50, v16, s[60:63], 0 offen offset:4 // (op 4033) Load value
v_add_u32 v16, v16, v24 // (op 4033) 
buffer_load_ubyte v51, v16, s[60:63], 0 offen offset:4 // (op 4033) Load value






buffer_load_dwordx4 v[52:55], v14, s[48:51], 0 offen offset:96 // (op 4036) Load value
buffer_load_dwordx2 v[56:57], v14, s[48:51], 0 offen offset:112 // (op 4036) Load value




buffer_load_ubyte v14, v18, s[56:59], 0 offen offset:4 // (op 4039) Load value

v_and_b32 v16, -4, v1 // (op 4615) 
v_sub_u32 v18, v1, v16 // (op 4615) 

v_lshlrev_b32 v16, 5, v18 // (op 4615) 

v_lshrrev_b32 v18, 2, v1 // (op 4615) 
v_lshl_add_u32 v20, v18, 8, v16 // (op 4615) 


v_lshlrev_b32 v16, 2, v20 // (op 4615) 

v_lshrrev_b32 v18, 3, v16 // (op 4615) 




s_waitcnt vmcnt(13)
ds_write_b128 v18, v[26:29] // (op 4041) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[30:33] offset:64 // (op 4041) Store local data 

v_and_b32 v16, -4, v1 // (op 4731) 
v_sub_u32 v18, v1, v16 // (op 4731) 

v_lshrrev_b32 v16, 2, v1 // (op 4731) 
v_lshl_add_u32 v20, v16, 3, v18 // (op 4731) 





s_waitcnt vmcnt(11)
ds_write_b8 v20, v34 offset:8192 // (op 4042) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v20, v35 offset:8196 // (op 4042) Store local data 

v_and_b32 v16, -4, v1 // (op 4557) 
v_sub_u32 v18, v1, v16 // (op 4557) 

v_lshlrev_b32 v16, 5, v18 // (op 4557) 

v_lshrrev_b32 v18, 2, v1 // (op 4557) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 4557) 


v_mul_lo_u32 v16, v20, 6 // (op 4557) 

v_lshrrev_b32 v18, 3, v16 // (op 4557) 




s_waitcnt vmcnt(9)
ds_write_b128 v18, v[36:39] offset:8704 // (op 4043) Store local data 
s_waitcnt vmcnt(8)
ds_write_b64 v18, v[40:41] offset:8720 // (op 4043) Store local data 

v_and_b32 v16, -4, v1 // (op 4673) 
v_sub_u32 v18, v1, v16 // (op 4673) 

v_and_b32 v16, 4294967292, v1 // (op 4673) 
v_add_u32 v20, v16, v18 // (op 4673) 





s_waitcnt vmcnt(7)
ds_write_b8 v20, v22 offset:14848 // (op 4044) Store local data 

v_and_b32 v16, -4, v1 // (op 4541) 
v_sub_u32 v18, v1, v16 // (op 4541) 

v_lshlrev_b32 v16, 5, v18 // (op 4541) 

v_lshrrev_b32 v18, 2, v1 // (op 4541) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 4541) 


v_mul_lo_u32 v16, v20, 6 // (op 4541) 

v_lshrrev_b32 v18, 3, v16 // (op 4541) 

v_and_b32 v16, -4, v1 // (op 4599) 
v_sub_u32 v20, v1, v16 // (op 4599) 

v_lshlrev_b32 v16, 5, v20 // (op 4599) 

v_lshrrev_b32 v20, 2, v1 // (op 4599) 
v_lshl_add_u32 v23, v20, 8, v16 // (op 4599) 


v_lshlrev_b32 v16, 2, v23 // (op 4599) 

v_lshrrev_b32 v20, 3, v16 // (op 4599) 

v_and_b32 v16, -4, v1 // (op 4657) 
v_sub_u32 v23, v1, v16 // (op 4657) 

v_and_b32 v16, 4294967292, v1 // (op 4657) 
v_add_u32 v24, v16, v23 // (op 4657) 


v_and_b32 v16, -4, v1 // (op 4715) 
v_sub_u32 v23, v1, v16 // (op 4715) 

v_lshrrev_b32 v16, 2, v1 // (op 4715) 
v_lshl_add_u32 v25, v16, 3, v23 // (op 4715) 


v_and_b32 v16, -4, v1 // (op 4783) 
v_sub_u32 v23, v1, v16 // (op 4783) 

v_lshlrev_b32 v16, 5, v23 // (op 4783) 

v_lshrrev_b32 v23, 2, v1 // (op 4783) 
v_lshl_add_u32 v58, v23, 7, v16 // (op 4783) 


v_mul_lo_u32 v16, v58, 6 // (op 4783) 

v_lshrrev_b32 v23, 3, v16 // (op 4783) 

v_and_b32 v16, -4, v1 // (op 4812) 
v_sub_u32 v58, v1, v16 // (op 4812) 

v_lshlrev_b32 v16, 5, v58 // (op 4812) 

v_lshrrev_b32 v58, 2, v1 // (op 4812) 
v_lshl_add_u32 v59, v58, 8, v16 // (op 4812) 


v_lshlrev_b32 v16, 2, v59 // (op 4812) 

v_lshrrev_b32 v58, 3, v16 // (op 4812) 

v_and_b32 v16, -4, v1 // (op 4841) 
v_sub_u32 v59, v1, v16 // (op 4841) 

v_and_b32 v16, 4294967292, v1 // (op 4841) 
v_add_u32 v60, v16, v59 // (op 4841) 


v_and_b32 v16, -4, v1 // (op 4870) 
v_sub_u32 v59, v1, v16 // (op 4870) 

v_lshrrev_b32 v16, 2, v1 // (op 4870) 
v_lshl_add_u32 v61, v16, 3, v59 // (op 4870) 


s_mov_b32 s0, 0 // (op 1051) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1050) 
s_mov_b32 s1, s6 // (op 1050) convert

s_and_b32 s6, s1, 4294967294 // (op 1050) 

s_cmp_lt_u32 s0, s6 // (op 1050) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_1050 // (op 1050) Condition: Top (jump to RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_1050 if false)
RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoop_1050:

s_barrier  // (op 4950) 
s_add_u32 s0, s0, 2 // (op 1052) 



s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[62:63], 3, s[6:7] // (op 10) 


v_mov_b32 v16, v17 // (op 10) 
v_mov_b32 v59, v16 // (op 10) 
buffer_load_dwordx4 v[26:29], v59, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v59, v59, v62 // (op 10) 
ds_read_b128 v[64:67], v3 offset:8704 // (op 72) Load local data 
ds_read_b64 v[68:69], v3 offset:8720 // (op 72) Load local data 
buffer_load_dwordx4 v[30:33], v59, s[52:55], 0 offen offset:128 // (op 10) Load value



s_barrier  // (op 4963) 
v_mov_b64 v[62:63], s[32:33] // (op 25) convert




v_mov_b32 v59, v21 // (op 25) 
v_mov_b32 v70, v59 // (op 25) 
buffer_load_ubyte v34, v70, s[60:63], 0 offen offset:8 // (op 25) Load value
v_add_u32 v70, v70, v62 // (op 25) 
ds_read_b128 v[72:75], v4 // (op 77) Load local data 
buffer_load_ubyte v35, v70, s[60:63], 0 offen offset:8 // (op 25) Load value




s_barrier  // (op 4977) 
v_mov_b32 v62, v15 // (op 4) 




ds_read_u8 v63, v0 offset:14848 // (op 82) Load local data 
buffer_load_dwordx4 v[36:39], v62, s[48:51], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 4991) 
buffer_load_dwordx2 v[40:41], v62, s[48:51], 0 offen offset:208 // (op 4) Load value





ds_read_u8 v70, v5 offset:8192 // (op 87) Load local data 
v_mov_b32 v71, v19 // (op 17) 




ds_read_b128 v[76:79], v4 offset:2048 // (op 375) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[72:75], a[0:3], v63, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 90) 



ds_read_u8 v80, v5 offset:8320 // (op 379) Load local data 
buffer_load_ubyte v22, v71, s[56:59], 0 offen offset:8 // (op 17) Load value



s_barrier  // (op 5014) 
ds_read_b128 v[82:85], v4 offset:4096 // (op 551) Load local data 






ds_read_u8 v81, v5 offset:8448 // (op 555) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:69], v[76:79], a[4:7], v63, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 380) 



ds_read_b128 v[86:89], v4 offset:6144 // (op 727) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:69], v[82:85], a[8:11], v63, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 556) 



ds_read_u8 v90, v5 offset:8576 // (op 731) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v58, v[42:45] offset:15104 // (op 257) Store local data 



ds_read_b128 v[92:95], v3 offset:11776 // (op 1084) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:69], v[86:89], a[12:15], v63, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 732) 
ds_read_b64 v[96:97], v3 offset:11792 // (op 1084) Load local data 





ds_read_u8 v91, v0 offset:14976 // (op 1087) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v58, v[46:49] offset:15168 // (op 257) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[92:97], v[72:75], a[16:19], v91, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1227) 


s_barrier  // (op 5026) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[92:97], v[76:79], a[20:23], v91, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1181) 
s_waitcnt vmcnt(11)
ds_write_b8 v61, v50 offset:23296 // (op 259) Store local data 


s_waitcnt vmcnt(10)
ds_write_b8 v61, v51 offset:23300 // (op 259) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[92:97], v[82:85], a[24:27], v91, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1135) 


s_barrier  // (op 5008) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[92:97], v[86:89], a[28:31], v91, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1089) 
s_waitcnt vmcnt(9)
ds_write_b128 v23, v[52:55] offset:23808 // (op 247) Store local data 




s_waitcnt vmcnt(8)
ds_write_b64 v23, v[56:57] offset:23824 // (op 247) Store local data 
s_barrier  // (op 5020) 



s_waitcnt vmcnt(7)
ds_write_b8 v60, v14 offset:29952 // (op 258) Store local data 

s_lshl_b64 s[6:7], s[24:25], 2 // (op 260) 
v_lshrrev_b64 v[64:65], 3, s[6:7] // (op 260) 


v_mov_b32 v63, v16 // (op 260) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 5005) 
buffer_load_dwordx4 v[42:45], v63, s[52:55], 0 offen offset:192 // (op 260) Load value



v_add_u32 v63, v63, v64 // (op 260) 
ds_read_b128 v[72:75], v7 offset:23808 // (op 249) Load local data 
ds_read_b64 v[76:77], v7 offset:23824 // (op 249) Load local data 
buffer_load_dwordx4 v[46:49], v63, s[52:55], 0 offen offset:192 // (op 260) Load value



s_barrier  // (op 5011) 




v_add_u32 v17, 128, v17 // (op 4349) 
v_mov_b64 v[64:65], s[32:33] // (op 262) convert

v_mov_b32 v16, v59 // (op 262) 
ds_read_b128 v[66:69], v8 offset:15104 // (op 251) Load local data 
buffer_load_ubyte v50, v16, s[60:63], 0 offen offset:12 // (op 262) Load value
v_add_u32 v16, v16, v64 // (op 262) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5017) 
buffer_load_ubyte v51, v16, s[60:63], 0 offen offset:12 // (op 262) Load value






ds_read_u8 v16, v6 offset:29952 // (op 253) Load local data 



s_barrier  // (op 5023) 
v_add_u32 v21, 8, v21 // (op 4477) 



ds_read_u8 v59, v9 offset:23296 // (op 255) Load local data 
buffer_load_dwordx4 v[52:55], v62, s[48:51], 0 offen offset:288 // (op 246) Load value



ds_read_b128 v[78:81], v8 offset:17152 // (op 385) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:77], v[66:69], a[0:3], v16, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 256) 



ds_read_u8 v70, v9 offset:23424 // (op 389) Load local data 
buffer_load_dwordx2 v[56:57], v62, s[48:51], 0 offen offset:304 // (op 246) Load value




ds_read_b128 v[62:65], v8 offset:19200 // (op 561) Load local data 






v_add_u32 v15, 192, v15 // (op 4285) 
ds_read_u8 v82, v9 offset:23552 // (op 565) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[78:81], a[4:7], v16, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 381) 



ds_read_b128 v[84:87], v8 offset:21248 // (op 737) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:77], v[62:65], a[8:11], v16, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 557) 
buffer_load_ubyte v14, v71, s[56:59], 0 offen offset:12 // (op 261) Load value




s_barrier  // (op 5087) 
s_barrier  // (op 4966) 
v_add_u32 v19, 8, v19 // (op 4413) 



ds_read_u8 v71, v9 offset:23680 // (op 741) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v20, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v20, v[30:33] offset:64 // (op 13) Store local data 



s_barrier  // (op 5091) 
s_barrier  // (op 4994) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[84:87], a[12:15], v16, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 733) 





ds_read_b128 v[72:75], v7 offset:26880 // (op 1092) Load local data 
ds_read_b64 v[76:77], v7 offset:26896 // (op 1092) Load local data 
s_waitcnt vmcnt(11)
ds_write_b8 v25, v34 offset:8192 // (op 30) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v25, v35 offset:8196 // (op 30) Store local data 



s_barrier  // (op 5115) 
s_barrier  // (op 4953) 
ds_read_u8 v16, v6 offset:30080 // (op 1095) Load local data 



s_waitcnt vmcnt(9)
ds_write_b128 v18, v[36:39] offset:8704 // (op 6) Store local data 
s_barrier  // (op 5119) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:77], v[66:69], a[16:19], v16, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1228) 
s_waitcnt vmcnt(8)
ds_write_b64 v18, v[40:41] offset:8720 // (op 6) Store local data 


s_barrier  // (op 4980) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:77], v[78:81], a[20:23], v16, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1182) 





s_waitcnt vmcnt(7)
ds_write_b8 v24, v22 offset:14848 // (op 21) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:77], v[62:65], a[24:27], v16, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1136) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:77], v[84:87], a[28:31], v16, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1090) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1050) 
s_mov_b32 s1, s6 // (op 1050) convert

s_and_b32 s6, s1, 4294967294 // (op 1050) 

s_cmp_lt_u32 s0, s6 // (op 1050) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoop_1050 // (op 1050) Condition: Bottom (jump to RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoop_1050 if true)
RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_1050:

















v_lshrrev_b32 v15, 2, v1 // (op 4241) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4241) 
v_lshl_add_u32 v16, s16, 6, v15 // (op 4241) 

s_lshr_b32 s6, s0, 1 // (op 4241) 

v_and_b32 v15, -4, v1 // (op 4241) 
v_sub_u32 v17, v1, v15 // (op 4241) 

v_lshlrev_b32 v15, 5, v17 // (op 4241) 

v_lshl_add_u32 v17, s6, 8, v15 // (op 4241) 


v_mov_b32 v19, 0 // (op 4241) convert
v_mov_b32 v18, v17 // (op 4241) convert

v_mul_lo_u32 v21, v16, s13 // (op 4241) most significant: low of low * high
v_mul_hi_u32 v15, v16, s12 // (op 4241) most significant: high of low * low
v_mul_lo_u32 v20, v16, s12 // (op 4241) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 4241) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 4241) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 4241) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 4241) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 4241) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 4241) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 4241) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 4241) 

v_mov_b32 v15, v16 // (op 4241) convert

s_mul_i32 s1, s23, 6 // (op 4243) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 4243) least significant
s_mul_hi_u32 s6, s22, 6 // (op 4243) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4243) most significant: sum

s_mul_i32 s7, s12, s1 // (op 4243) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 4243) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 4243) least significant
s_mul_hi_u32 s11, s12, s0 // (op 4243) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 4243) most significant: sum
s_add_u32 s7, s7, s11 // (op 4243) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 4243) 

s_mov_b32 s48, s20 // (op 4243) reg expression
s_mov_b32 s49, s21 // (op 4243) reg expression
s_mov_b32 s50, s0 // (op 4243) reg expression
s_mov_b32 s51, 131072 // (op 4243) call()

v_and_b32 v16, -4, v1 // (op 4315) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4315) 
v_sub_u32 v17, v1, v16 // (op 4315) 

s_lshr_b32 s6, s0, 1 // (op 4315) 

v_lshlrev_b32 v16, 5, v17 // (op 4315) 

v_lshl_add_u32 v17, s6, 8, v16 // (op 4315) 


v_mov_b32 v19, 0 // (op 4315) convert
v_mov_b32 v18, v17 // (op 4315) convert

v_lshrrev_b32 v16, 2, v1 // (op 4315) 
v_lshlrev_b32 v17, 1, v16 // (op 4315) 

v_lshl_add_u32 v16, s17, 7, v17 // (op 4315) 

v_mul_lo_u32 v21, v16, s25 // (op 4315) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 4315) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 4315) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 4315) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 4315) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 4315) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 4315) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 4315) 

v_mov_b32 v16, v18 // (op 4315) convert

s_mul_i32 s1, s24, s9 // (op 4317) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4317) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4317) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4317) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4317) most significant: sum
s_add_u32 s1, s1, s7 // (op 4317) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4317) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 4317) 

s_mov_b32 s52, s26 // (op 4317) reg expression
s_mov_b32 s53, s27 // (op 4317) reg expression
s_mov_b32 s54, s0 // (op 4317) reg expression
s_mov_b32 s55, 131072 // (op 4317) call()

v_lshrrev_b32 v17, 2, v1 // (op 4379) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4379) 
v_lshl_add_u32 v18, s16, 6, v17 // (op 4379) 

s_lshr_b32 s6, s0, 1 // (op 4379) 

v_and_b32 v17, -4, v1 // (op 4379) 
v_sub_u32 v19, v1, v17 // (op 4379) 

v_lshl_add_u32 v17, s6, 3, v19 // (op 4379) 


v_mul_lo_u32 v19, v18, s28 // (op 4379) 
v_add_u32 v19, v19, v17 // (op 4379) 


s_mul_i32 s1, s28, s23 // (op 4381) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4381) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4381) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4381) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4381) most significant: sum
s_add_u32 s1, s1, s7 // (op 4381) most significant: sum


s_mov_b32 s56, s30 // (op 4381) reg expression
s_mov_b32 s57, s31 // (op 4381) reg expression
s_mov_b32 s58, s0 // (op 4381) reg expression
s_mov_b32 s59, 131072 // (op 4381) call()

v_and_b32 v17, -4, v1 // (op 4443) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4443) 
v_sub_u32 v18, v1, v17 // (op 4443) 

s_lshr_b32 s6, s0, 1 // (op 4443) 

v_lshrrev_b32 v17, 2, v1 // (op 4443) 
v_lshl_add_u32 v20, s6, 3, v18 // (op 4443) 


v_lshlrev_b32 v18, 1, v17 // (op 4443) 

v_lshl_add_u32 v17, s17, 7, v18 // (op 4443) 

v_mul_lo_u32 v18, v17, s32 // (op 4443) 

v_add_u32 v17, v20, v18 // (op 4443) 


s_mul_i32 s1, s32, s9 // (op 4445) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4445) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4445) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4445) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4445) most significant: sum
s_add_u32 s1, s1, s7 // (op 4445) most significant: sum


s_mov_b32 s60, s34 // (op 4445) reg expression
s_mov_b32 s61, s35 // (op 4445) reg expression
s_mov_b32 s62, s0 // (op 4445) reg expression
s_mov_b32 s63, 131072 // (op 4445) call()

v_and_b32 v18, -4, v1 // (op 4526) 
v_sub_u32 v20, v1, v18 // (op 4526) 

v_lshlrev_b32 v18, 5, v20 // (op 4526) 

v_lshrrev_b32 v20, 2, v1 // (op 4526) 
v_lshl_add_u32 v21, v20, 7, v18 // (op 4526) 


v_mul_lo_u32 v18, v21, 6 // (op 4526) 

v_lshrrev_b32 v20, 3, v18 // (op 4526) 

v_and_b32 v18, -4, v1 // (op 4586) 
v_sub_u32 v21, v1, v18 // (op 4586) 

v_lshlrev_b32 v18, 5, v21 // (op 4586) 

v_lshrrev_b32 v21, 2, v1 // (op 4586) 
v_lshl_add_u32 v22, v21, 8, v18 // (op 4586) 


v_lshlrev_b32 v18, 2, v22 // (op 4586) 

v_lshrrev_b32 v21, 3, v18 // (op 4586) 

v_and_b32 v18, -4, v1 // (op 4644) 
v_sub_u32 v22, v1, v18 // (op 4644) 

v_and_b32 v18, 4294967292, v1 // (op 4644) 
v_add_u32 v23, v18, v22 // (op 4644) 


v_and_b32 v18, -4, v1 // (op 4702) 
v_sub_u32 v22, v1, v18 // (op 4702) 

v_lshrrev_b32 v18, 2, v1 // (op 4702) 
v_lshl_add_u32 v24, v18, 3, v22 // (op 4702) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 1025) 
s_and_b32 s6, s0, 4294967294 // (op 1025) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 1024) 
s_mov_b32 s7, s0 // (op 1024) convert

s_cmp_lt_u32 s6, s7 // (op 1024) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_1024 // (op 1024) Condition: Top (jump to RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_1024 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_1024:


s_add_u32 s6, s6, 1 // (op 1026) 
v_mov_b32 v14, v15 // (op 180) 

buffer_load_dwordx4 v[26:29], v14, s[48:51], 0 offen // (op 180) Load value
buffer_load_dwordx2 v[30:31], v14, s[48:51], 0 offen offset:16 // (op 180) Load value

v_add_u32 v15, 96, v15 // (op 4231) 

s_barrier  // (op 4956) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 194) 



v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 194) 


v_mov_b32 v14, v16 // (op 194) 
v_mov_b32 v18, v14 // (op 194) 
buffer_load_dwordx4 v[34:37], v18, s[52:55], 0 offen // (op 194) Load value
v_add_u32 v18, v18, v32 // (op 194) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[26:29] offset:8704 // (op 181) Store local data 
buffer_load_dwordx4 v[38:41], v18, s[52:55], 0 offen // (op 194) Load value



v_add_u32 v16, 64, v16 // (op 4307) 
s_waitcnt vmcnt(2)
ds_write_b64 v20, v[30:31] offset:8720 // (op 181) Store local data 



v_mov_b32 v14, v19 // (op 195) 

s_barrier  // (op 4969) 
buffer_load_ubyte v18, v14, s[56:59], 0 offen offset: 0 // (op 195) Load value




v_add_u32 v19, 4, v19 // (op 4371) 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[34:37] // (op 191) Store local data 

v_mov_b64 v[32:33], s[32:33] // (op 196) convert

v_mov_b32 v14, v17 // (op 196) 
v_mov_b32 v22, v14 // (op 196) 
buffer_load_ubyte v30, v22, s[60:63], 0 offen offset: 0 // (op 196) Load value
v_add_u32 v22, v22, v32 // (op 196) 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[38:41] offset:64 // (op 191) Store local data 
buffer_load_ubyte v31, v22, s[60:63], 0 offen offset: 0 // (op 196) Load value




v_add_u32 v17, 4, v17 // (op 4435) 
s_barrier  // (op 4983) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4959) 






ds_read_b128 v[32:35], v3 offset:8704 // (op 183) Load local data 
ds_read_b64 v[36:37], v3 offset:8720 // (op 183) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v23, v18 offset:14848 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 4973) 



s_barrier  // (op 4997) 
ds_read_b128 v[26:29], v4 // (op 185) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 4987) 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v30 offset:8192 // (op 193) Store local data 



ds_read_u8 v14, v0 offset:14848 // (op 187) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v31 offset:8196 // (op 193) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 5001) 



ds_read_u8 v18, v5 offset:8192 // (op 189) Load local data 



ds_read_b128 v[38:41], v4 offset:2048 // (op 354) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:37], v[26:29], a[0:3], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 190) 



ds_read_u8 v22, v5 offset:8320 // (op 358) Load local data 



ds_read_b128 v[42:45], v4 offset:4096 // (op 530) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:37], v[38:41], a[4:7], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 359) 



ds_read_u8 v25, v5 offset:8448 // (op 534) Load local data 



ds_read_b128 v[46:49], v4 offset:6144 // (op 706) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:37], v[42:45], a[8:11], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 535) 



ds_read_u8 v30, v5 offset:8576 // (op 710) Load local data 



ds_read_b128 v[50:53], v3 offset:11776 // (op 968) Load local data 
ds_read_b64 v[54:55], v3 offset:11792 // (op 968) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:37], v[46:49], a[12:15], v14, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 711) 





ds_read_u8 v14, v0 offset:14976 // (op 971) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[50:55], v[26:29], a[16:19], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 973) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:55], v[38:41], a[20:23], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1004) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[50:55], v[42:45], a[24:27], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1035) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[50:55], v[46:49], a[28:31], v14, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 1066) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 1024) 
s_mov_b32 s7, s0 // (op 1024) convert

s_cmp_lt_u32 s6, s7 // (op 1024) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_1024 // (op 1024) Condition: Bottom (jump to RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_1024 if true)
RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_1024:























s_barrier  // (op 5099) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 
v_cvt_f16_f32 v8, v16 // (op 56) 
v_cvt_f16_f32 v9, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 1873) 
v_accvgpr_read v7, a5 // (op 1873) 
v_accvgpr_read v8, a6 // (op 1873) 
v_accvgpr_read v9, a7 // (op 1873) 
v_mul_f32 v14, v11, v6 // (op 1873) 
v_mul_f32 v15, v11, v7 // (op 1873) 
v_mul_f32 v16, v11, v8 // (op 1873) 
v_mul_f32 v17, v11, v9 // (op 1873) 

v_cvt_f16_f32 v6, v14 // (op 1875) 
v_cvt_f16_f32 v7, v15 // (op 1875) 
v_cvt_f16_f32 v8, v16 // (op 1875) 
v_cvt_f16_f32 v9, v17 // (op 1875) 


v_pack_b32_f16 v4, v6, v7 // (op 1876) 
v_pack_b32_f16 v5, v8, v9 // (op 1876) 

s_lshl_b32 s0, s4, 6 // (op 1876) 
v_add_u32 v0, v13, s0 // (op 1876) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1876) Store value



v_accvgpr_read v6, a8 // (op 1887) 
v_accvgpr_read v7, a9 // (op 1887) 
v_accvgpr_read v8, a10 // (op 1887) 
v_accvgpr_read v9, a11 // (op 1887) 
v_mul_f32 v14, v11, v6 // (op 1887) 
v_mul_f32 v15, v11, v7 // (op 1887) 
v_mul_f32 v16, v11, v8 // (op 1887) 
v_mul_f32 v17, v11, v9 // (op 1887) 

v_cvt_f16_f32 v6, v14 // (op 1889) 
v_cvt_f16_f32 v7, v15 // (op 1889) 
v_cvt_f16_f32 v8, v16 // (op 1889) 
v_cvt_f16_f32 v9, v17 // (op 1889) 


v_pack_b32_f16 v4, v6, v7 // (op 1890) 
v_pack_b32_f16 v5, v8, v9 // (op 1890) 

s_lshl_b32 s0, s4, 7 // (op 1890) 
v_add_u32 v0, v13, s0 // (op 1890) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1890) Store value



v_accvgpr_read v6, a12 // (op 1901) 
v_accvgpr_read v7, a13 // (op 1901) 
v_accvgpr_read v8, a14 // (op 1901) 
v_accvgpr_read v9, a15 // (op 1901) 
v_mul_f32 v14, v11, v6 // (op 1901) 
v_mul_f32 v15, v11, v7 // (op 1901) 
v_mul_f32 v16, v11, v8 // (op 1901) 
v_mul_f32 v17, v11, v9 // (op 1901) 

v_cvt_f16_f32 v6, v14 // (op 1903) 
v_cvt_f16_f32 v7, v15 // (op 1903) 
v_cvt_f16_f32 v8, v16 // (op 1903) 
v_cvt_f16_f32 v9, v17 // (op 1903) 


v_pack_b32_f16 v4, v6, v7 // (op 1904) 
v_pack_b32_f16 v5, v8, v9 // (op 1904) 

s_lshl_b32 s0, s4, 6 // (op 1904) 
s_mul_i32 s1, 3, s0 // (op 1904) 

v_add_u32 v0, v13, s1 // (op 1904) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1904) Store value



v_accvgpr_read v6, a16 // (op 1937) 
v_accvgpr_read v7, a17 // (op 1937) 
v_accvgpr_read v8, a18 // (op 1937) 
v_accvgpr_read v9, a19 // (op 1937) 
v_mul_f32 v14, v11, v6 // (op 1937) 
v_mul_f32 v15, v11, v7 // (op 1937) 
v_mul_f32 v16, v11, v8 // (op 1937) 
v_mul_f32 v17, v11, v9 // (op 1937) 

v_cvt_f16_f32 v6, v14 // (op 1939) 
v_cvt_f16_f32 v7, v15 // (op 1939) 
v_cvt_f16_f32 v8, v16 // (op 1939) 
v_cvt_f16_f32 v9, v17 // (op 1939) 


v_pack_b32_f16 v4, v6, v7 // (op 1941) 
v_pack_b32_f16 v5, v8, v9 // (op 1941) 

v_add_u32 v0, 64, v13 // (op 1941) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1941) Store value



v_accvgpr_read v6, a20 // (op 1951) 
v_accvgpr_read v7, a21 // (op 1951) 
v_accvgpr_read v8, a22 // (op 1951) 
v_accvgpr_read v9, a23 // (op 1951) 
v_mul_f32 v14, v11, v6 // (op 1951) 
v_mul_f32 v15, v11, v7 // (op 1951) 
v_mul_f32 v16, v11, v8 // (op 1951) 
v_mul_f32 v17, v11, v9 // (op 1951) 

v_cvt_f16_f32 v6, v14 // (op 1953) 
v_cvt_f16_f32 v7, v15 // (op 1953) 
v_cvt_f16_f32 v8, v16 // (op 1953) 
v_cvt_f16_f32 v9, v17 // (op 1953) 


v_pack_b32_f16 v4, v6, v7 // (op 1943) 
v_pack_b32_f16 v5, v8, v9 // (op 1943) 

s_lshl_b32 s0, s4, 6 // (op 1943) 
s_add_u32 s0, s0, 64 // (op 1943) 
v_add_u32 v0, v13, s0 // (op 1943) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1943) Store value



v_accvgpr_read v6, a24 // (op 1957) 
v_accvgpr_read v7, a25 // (op 1957) 
v_accvgpr_read v8, a26 // (op 1957) 
v_accvgpr_read v9, a27 // (op 1957) 
v_mul_f32 v14, v11, v6 // (op 1957) 
v_mul_f32 v15, v11, v7 // (op 1957) 
v_mul_f32 v16, v11, v8 // (op 1957) 
v_mul_f32 v17, v11, v9 // (op 1957) 

v_cvt_f16_f32 v6, v14 // (op 1959) 
v_cvt_f16_f32 v7, v15 // (op 1959) 
v_cvt_f16_f32 v8, v16 // (op 1959) 
v_cvt_f16_f32 v9, v17 // (op 1959) 


v_pack_b32_f16 v4, v6, v7 // (op 1945) 
v_pack_b32_f16 v5, v8, v9 // (op 1945) 

s_lshl_b32 s0, s4, 7 // (op 1945) 
s_add_u32 s0, s0, 64 // (op 1945) 
v_add_u32 v0, v13, s0 // (op 1945) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1945) Store value



v_accvgpr_read v6, a28 // (op 1963) 
v_accvgpr_read v7, a29 // (op 1963) 
v_accvgpr_read v8, a30 // (op 1963) 
v_accvgpr_read v9, a31 // (op 1963) 
v_mul_f32 v14, v11, v6 // (op 1963) 
v_mul_f32 v15, v11, v7 // (op 1963) 
v_mul_f32 v16, v11, v8 // (op 1963) 
v_mul_f32 v17, v11, v9 // (op 1963) 

v_cvt_f16_f32 v6, v14 // (op 1965) 
v_cvt_f16_f32 v7, v15 // (op 1965) 
v_cvt_f16_f32 v8, v16 // (op 1965) 
v_cvt_f16_f32 v9, v17 // (op 1965) 


v_pack_b32_f16 v4, v6, v7 // (op 1947) 
v_pack_b32_f16 v5, v8, v9 // (op 1947) 

s_lshl_b32 s0, s4, 6 // (op 1947) 
s_mul_i32 s1, 3, s0 // (op 1947) 
s_add_u32 s1, s1, 64 // (op 1947) 

v_add_u32 v0, v13, s1 // (op 1947) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 1947) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v12 // (op 2133) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2133) Load value

v_accvgpr_read v4, a16 // (op 2120) 
v_accvgpr_read v5, a17 // (op 2120) 
v_accvgpr_read v6, a18 // (op 2120) 
v_accvgpr_read v7, a19 // (op 2120) 
v_mul_f32 v14, v11, v4 // (op 2120) 
v_mul_f32 v15, v11, v5 // (op 2120) 
v_mul_f32 v16, v11, v6 // (op 2120) 
v_mul_f32 v17, v11, v7 // (op 2120) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2132) 
v_lshrrev_b32 v7, 16, v8 // (op 2132) 
v_cvt_f32_f16 v7, v7 // (op 2132) 
v_mul_f32 v18, v10, v6 // (op 2132) 
v_mul_f32 v19, v10, v7 // (op 2132) 
v_cvt_f32_f16 v6, v9 // (op 2132) 
v_lshrrev_b32 v7, 16, v9 // (op 2132) 
v_cvt_f32_f16 v7, v7 // (op 2132) 
v_mul_f32 v20, v10, v6 // (op 2132) 
v_mul_f32 v21, v10, v7 // (op 2132) 


v_add_f32 v6, v18, v14 // (op 2121) 
v_add_f32 v7, v19, v15 // (op 2121) 
v_add_f32 v8, v20, v16 // (op 2121) 
v_add_f32 v9, v21, v17 // (op 2121) 


v_cvt_f16_f32 v14, v6 // (op 2122) 
v_cvt_f16_f32 v15, v7 // (op 2122) 
v_cvt_f16_f32 v16, v8 // (op 2122) 
v_cvt_f16_f32 v17, v9 // (op 2122) 


v_pack_b32_f16 v8, v14, v15 // (op 2124) 
v_pack_b32_f16 v9, v16, v17 // (op 2124) 

v_add_u32 v0, 64, v13 // (op 2124) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2124) Store value





s_lshl_b32 s0, s2, 6 // (op 2138) 
s_add_u32 s0, s0, 64 // (op 2138) 
v_add_u32 v0, v12, s0 // (op 2138) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2138) Load value

v_accvgpr_read v4, a20 // (op 2150) 
v_accvgpr_read v5, a21 // (op 2150) 
v_accvgpr_read v6, a22 // (op 2150) 
v_accvgpr_read v7, a23 // (op 2150) 
v_mul_f32 v14, v11, v4 // (op 2150) 
v_mul_f32 v15, v11, v5 // (op 2150) 
v_mul_f32 v16, v11, v6 // (op 2150) 
v_mul_f32 v17, v11, v7 // (op 2150) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2135) 
v_lshrrev_b32 v7, 16, v8 // (op 2135) 
v_cvt_f32_f16 v7, v7 // (op 2135) 
v_mul_f32 v18, v10, v6 // (op 2135) 
v_mul_f32 v19, v10, v7 // (op 2135) 
v_cvt_f32_f16 v6, v9 // (op 2135) 
v_lshrrev_b32 v7, 16, v9 // (op 2135) 
v_cvt_f32_f16 v7, v7 // (op 2135) 
v_mul_f32 v20, v10, v6 // (op 2135) 
v_mul_f32 v21, v10, v7 // (op 2135) 


v_add_f32 v6, v18, v14 // (op 2136) 
v_add_f32 v7, v19, v15 // (op 2136) 
v_add_f32 v8, v20, v16 // (op 2136) 
v_add_f32 v9, v21, v17 // (op 2136) 


v_cvt_f16_f32 v14, v6 // (op 2137) 
v_cvt_f16_f32 v15, v7 // (op 2137) 
v_cvt_f16_f32 v16, v8 // (op 2137) 
v_cvt_f16_f32 v17, v9 // (op 2137) 


v_pack_b32_f16 v8, v14, v15 // (op 2126) 
v_pack_b32_f16 v9, v16, v17 // (op 2126) 

s_lshl_b32 s0, s4, 6 // (op 2126) 
s_add_u32 s0, s0, 64 // (op 2126) 
v_add_u32 v0, v13, s0 // (op 2126) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2126) Store value





s_lshl_b32 s0, s2, 7 // (op 2143) 
s_add_u32 s0, s0, 64 // (op 2143) 
v_add_u32 v0, v12, s0 // (op 2143) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2143) Load value

v_accvgpr_read v4, a24 // (op 2152) 
v_accvgpr_read v5, a25 // (op 2152) 
v_accvgpr_read v6, a26 // (op 2152) 
v_accvgpr_read v7, a27 // (op 2152) 
v_mul_f32 v14, v11, v4 // (op 2152) 
v_mul_f32 v15, v11, v5 // (op 2152) 
v_mul_f32 v16, v11, v6 // (op 2152) 
v_mul_f32 v17, v11, v7 // (op 2152) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2140) 
v_lshrrev_b32 v7, 16, v8 // (op 2140) 
v_cvt_f32_f16 v7, v7 // (op 2140) 
v_mul_f32 v18, v10, v6 // (op 2140) 
v_mul_f32 v19, v10, v7 // (op 2140) 
v_cvt_f32_f16 v6, v9 // (op 2140) 
v_lshrrev_b32 v7, 16, v9 // (op 2140) 
v_cvt_f32_f16 v7, v7 // (op 2140) 
v_mul_f32 v20, v10, v6 // (op 2140) 
v_mul_f32 v21, v10, v7 // (op 2140) 


v_add_f32 v6, v18, v14 // (op 2141) 
v_add_f32 v7, v19, v15 // (op 2141) 
v_add_f32 v8, v20, v16 // (op 2141) 
v_add_f32 v9, v21, v17 // (op 2141) 


v_cvt_f16_f32 v14, v6 // (op 2142) 
v_cvt_f16_f32 v15, v7 // (op 2142) 
v_cvt_f16_f32 v16, v8 // (op 2142) 
v_cvt_f16_f32 v17, v9 // (op 2142) 


v_pack_b32_f16 v8, v14, v15 // (op 2128) 
v_pack_b32_f16 v9, v16, v17 // (op 2128) 

s_lshl_b32 s0, s4, 7 // (op 2128) 
s_add_u32 s0, s0, 64 // (op 2128) 
v_add_u32 v0, v13, s0 // (op 2128) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2128) Store value





s_lshl_b32 s0, s2, 6 // (op 2148) 
s_mul_i32 s1, 3, s0 // (op 2148) 
s_add_u32 s1, s1, 64 // (op 2148) 

v_add_u32 v0, v12, s1 // (op 2148) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2148) Load value

v_accvgpr_read v4, a28 // (op 2154) 
v_accvgpr_read v5, a29 // (op 2154) 
v_accvgpr_read v6, a30 // (op 2154) 
v_accvgpr_read v7, a31 // (op 2154) 
v_mul_f32 v14, v11, v4 // (op 2154) 
v_mul_f32 v15, v11, v5 // (op 2154) 
v_mul_f32 v16, v11, v6 // (op 2154) 
v_mul_f32 v17, v11, v7 // (op 2154) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2145) 
v_lshrrev_b32 v7, 16, v8 // (op 2145) 
v_cvt_f32_f16 v7, v7 // (op 2145) 
v_mul_f32 v18, v10, v6 // (op 2145) 
v_mul_f32 v19, v10, v7 // (op 2145) 
v_cvt_f32_f16 v6, v9 // (op 2145) 
v_lshrrev_b32 v7, 16, v9 // (op 2145) 
v_cvt_f32_f16 v7, v7 // (op 2145) 
v_mul_f32 v20, v10, v6 // (op 2145) 
v_mul_f32 v21, v10, v7 // (op 2145) 


v_add_f32 v6, v18, v14 // (op 2146) 
v_add_f32 v7, v19, v15 // (op 2146) 
v_add_f32 v8, v20, v16 // (op 2146) 
v_add_f32 v9, v21, v17 // (op 2146) 


v_cvt_f16_f32 v14, v6 // (op 2147) 
v_cvt_f16_f32 v15, v7 // (op 2147) 
v_cvt_f16_f32 v16, v8 // (op 2147) 
v_cvt_f16_f32 v17, v9 // (op 2147) 


v_pack_b32_f16 v8, v14, v15 // (op 2130) 
v_pack_b32_f16 v9, v16, v17 // (op 2130) 

s_lshl_b32 s0, s4, 6 // (op 2130) 
s_mul_i32 s1, 3, s0 // (op 2130) 
s_add_u32 s1, s1, 64 // (op 2130) 

v_add_u32 v0, v13, s1 // (op 2130) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2130) Store value





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
v_cvt_f32_f16 v6, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_cvt_f32_f16 v6, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_cvt_f16_f32 v14, v6 // (op 146) 
v_cvt_f16_f32 v15, v7 // (op 146) 
v_cvt_f16_f32 v16, v8 // (op 146) 
v_cvt_f16_f32 v17, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 2041) 
v_add_u32 v0, v12, s0 // (op 2041) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2041) Load value

v_accvgpr_read v4, a4 // (op 2037) 
v_accvgpr_read v5, a5 // (op 2037) 
v_accvgpr_read v6, a6 // (op 2037) 
v_accvgpr_read v7, a7 // (op 2037) 
v_mul_f32 v14, v11, v4 // (op 2037) 
v_mul_f32 v15, v11, v5 // (op 2037) 
v_mul_f32 v16, v11, v6 // (op 2037) 
v_mul_f32 v17, v11, v7 // (op 2037) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2042) 
v_lshrrev_b32 v7, 16, v8 // (op 2042) 
v_cvt_f32_f16 v7, v7 // (op 2042) 
v_mul_f32 v18, v10, v6 // (op 2042) 
v_mul_f32 v19, v10, v7 // (op 2042) 
v_cvt_f32_f16 v6, v9 // (op 2042) 
v_lshrrev_b32 v7, 16, v9 // (op 2042) 
v_cvt_f32_f16 v7, v7 // (op 2042) 
v_mul_f32 v20, v10, v6 // (op 2042) 
v_mul_f32 v21, v10, v7 // (op 2042) 


v_add_f32 v6, v18, v14 // (op 2038) 
v_add_f32 v7, v19, v15 // (op 2038) 
v_add_f32 v8, v20, v16 // (op 2038) 
v_add_f32 v9, v21, v17 // (op 2038) 


v_cvt_f16_f32 v14, v6 // (op 2039) 
v_cvt_f16_f32 v15, v7 // (op 2039) 
v_cvt_f16_f32 v16, v8 // (op 2039) 
v_cvt_f16_f32 v17, v9 // (op 2039) 


v_pack_b32_f16 v8, v14, v15 // (op 2040) 
v_pack_b32_f16 v9, v16, v17 // (op 2040) 

s_lshl_b32 s0, s4, 6 // (op 2040) 
v_add_u32 v0, v13, s0 // (op 2040) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2040) Store value





s_lshl_b32 s0, s2, 7 // (op 2055) 
v_add_u32 v0, v12, s0 // (op 2055) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2055) Load value

v_accvgpr_read v4, a8 // (op 2051) 
v_accvgpr_read v5, a9 // (op 2051) 
v_accvgpr_read v6, a10 // (op 2051) 
v_accvgpr_read v7, a11 // (op 2051) 
v_mul_f32 v14, v11, v4 // (op 2051) 
v_mul_f32 v15, v11, v5 // (op 2051) 
v_mul_f32 v16, v11, v6 // (op 2051) 
v_mul_f32 v17, v11, v7 // (op 2051) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2056) 
v_lshrrev_b32 v7, 16, v8 // (op 2056) 
v_cvt_f32_f16 v7, v7 // (op 2056) 
v_mul_f32 v18, v10, v6 // (op 2056) 
v_mul_f32 v19, v10, v7 // (op 2056) 
v_cvt_f32_f16 v6, v9 // (op 2056) 
v_lshrrev_b32 v7, 16, v9 // (op 2056) 
v_cvt_f32_f16 v7, v7 // (op 2056) 
v_mul_f32 v20, v10, v6 // (op 2056) 
v_mul_f32 v21, v10, v7 // (op 2056) 


v_add_f32 v6, v18, v14 // (op 2052) 
v_add_f32 v7, v19, v15 // (op 2052) 
v_add_f32 v8, v20, v16 // (op 2052) 
v_add_f32 v9, v21, v17 // (op 2052) 


v_cvt_f16_f32 v14, v6 // (op 2053) 
v_cvt_f16_f32 v15, v7 // (op 2053) 
v_cvt_f16_f32 v16, v8 // (op 2053) 
v_cvt_f16_f32 v17, v9 // (op 2053) 


v_pack_b32_f16 v8, v14, v15 // (op 2054) 
v_pack_b32_f16 v9, v16, v17 // (op 2054) 

s_lshl_b32 s0, s4, 7 // (op 2054) 
v_add_u32 v0, v13, s0 // (op 2054) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2054) Store value





s_lshl_b32 s0, s2, 6 // (op 2069) 
s_mul_i32 s1, 3, s0 // (op 2069) 

v_add_u32 v0, v12, s1 // (op 2069) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 2069) Load value

v_accvgpr_read v4, a12 // (op 2065) 
v_accvgpr_read v5, a13 // (op 2065) 
v_accvgpr_read v6, a14 // (op 2065) 
v_accvgpr_read v7, a15 // (op 2065) 
v_mul_f32 v14, v11, v4 // (op 2065) 
v_mul_f32 v15, v11, v5 // (op 2065) 
v_mul_f32 v16, v11, v6 // (op 2065) 
v_mul_f32 v17, v11, v7 // (op 2065) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v8 // (op 2070) 
v_lshrrev_b32 v7, 16, v8 // (op 2070) 
v_cvt_f32_f16 v7, v7 // (op 2070) 
v_mul_f32 v18, v10, v6 // (op 2070) 
v_mul_f32 v19, v10, v7 // (op 2070) 
v_cvt_f32_f16 v6, v9 // (op 2070) 
v_lshrrev_b32 v7, 16, v9 // (op 2070) 
v_cvt_f32_f16 v7, v7 // (op 2070) 
v_mul_f32 v20, v10, v6 // (op 2070) 
v_mul_f32 v21, v10, v7 // (op 2070) 


v_add_f32 v6, v18, v14 // (op 2066) 
v_add_f32 v7, v19, v15 // (op 2066) 
v_add_f32 v8, v20, v16 // (op 2066) 
v_add_f32 v9, v21, v17 // (op 2066) 


v_cvt_f16_f32 v14, v6 // (op 2067) 
v_cvt_f16_f32 v15, v7 // (op 2067) 
v_cvt_f16_f32 v16, v8 // (op 2067) 
v_cvt_f16_f32 v17, v9 // (op 2067) 


v_pack_b32_f16 v8, v14, v15 // (op 2068) 
v_pack_b32_f16 v9, v16, v17 // (op 2068) 

s_lshl_b32 s0, s4, 6 // (op 2068) 
s_mul_i32 s1, 3, s0 // (op 2068) 

v_add_u32 v0, v13, s1 // (op 2068) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 2068) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
.LRR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_end:
.size RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC, .LRR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_end-RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
  .amdhsa_next_free_vgpr 132
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 30208 // lds bytes
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
    DataTypeB: F4
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
    macrotile: [64, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
    .symbol: RR_GEMM_TN_BF6_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 30208
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 98
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

