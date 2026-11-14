// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC,@function
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 4540) 
v_and_b32 v3, 15, v1 // (op 4540) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4540) 


v_and_b32 v0, 48, v1 // (op 4540) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 4540) 


v_lshrrev_b32 v0, 6, v1 // (op 4598) 
v_and_b32 v4, 1, v0 // (op 4598) 

v_and_b32 v0, 15, v1 // (op 4598) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4598) 


v_and_b32 v0, 63, v1 // (op 4598) 
v_lshrrev_b32 v4, 4, v0 // (op 4598) 

v_lshlrev_b32 v0, 5, v4 // (op 4598) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 4598) 


v_lshlrev_b32 v0, 2, v4 // (op 4598) 

v_lshrrev_b32 v4, 3, v0 // (op 4598) 

v_lshrrev_b32 v0, 7, v1 // (op 4656) 
v_and_b32 v5, 15, v1 // (op 4656) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4656) 


v_and_b32 v0, 63, v1 // (op 4656) 
v_lshrrev_b32 v5, 4, v0 // (op 4656) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 4656) 


v_lshrrev_b32 v5, 6, v1 // (op 4714) 
v_and_b32 v6, 1, v5 // (op 4714) 

v_and_b32 v5, 15, v1 // (op 4714) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4714) 


v_and_b32 v5, 63, v1 // (op 4714) 
v_lshrrev_b32 v6, 4, v5 // (op 4714) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 4714) 


v_lshrrev_b32 v6, 7, v1 // (op 4766) 
v_and_b32 v7, 15, v1 // (op 4766) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 4766) 


v_and_b32 v6, 48, v1 // (op 4766) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 4766) 


v_lshrrev_b32 v6, 6, v1 // (op 4795) 
v_and_b32 v8, 1, v6 // (op 4795) 

v_and_b32 v6, 15, v1 // (op 4795) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 4795) 


v_and_b32 v6, 63, v1 // (op 4795) 
v_lshrrev_b32 v8, 4, v6 // (op 4795) 

v_lshlrev_b32 v6, 5, v8 // (op 4795) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 4795) 


v_lshlrev_b32 v6, 2, v8 // (op 4795) 

v_lshrrev_b32 v8, 3, v6 // (op 4795) 

v_lshrrev_b32 v6, 7, v1 // (op 4824) 
v_and_b32 v9, 15, v1 // (op 4824) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 4824) 


v_and_b32 v6, 63, v1 // (op 4824) 
v_lshrrev_b32 v9, 4, v6 // (op 4824) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 4824) 


v_lshrrev_b32 v9, 6, v1 // (op 4853) 
v_and_b32 v10, 1, v9 // (op 4853) 

v_and_b32 v9, 15, v1 // (op 4853) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4853) 


v_and_b32 v9, 63, v1 // (op 4853) 
v_lshrrev_b32 v10, 4, v9 // (op 4853) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 4853) 


v_lshrrev_b32 v10, 7, v1 // (op 4478) 
v_and_b32 v11, 63, v1 // (op 4478) 
v_lshrrev_b32 v12, 4, v11 // (op 4478) 
v_lshlrev_b32 v13, 2, v12 // (op 4478) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4478) 


v_lshl_add_u32 v10, s16, 6, v12 // (op 4478) 

v_lshrrev_b32 v12, 2, v11 // (op 4478) 

v_and_b32 v11, 3, v12 // (op 4478) 

v_and_b32 v12, 3, v1 // (op 4478) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 4478) 


v_lshrrev_b32 v11, 6, v1 // (op 4478) 
v_and_b32 v12, 1, v11 // (op 4478) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 4478) 


v_lshl_add_u32 v12, s17, 7, v11 // (op 4478) 

v_mul_lo_u32 v11, v12, s2 // (op 4478) 

v_add_lshl_u32 v12, v10, v11, 2 // (op 4478) 


v_lshrrev_b32 v10, 7, v1 // (op 4735) 
v_and_b32 v11, 63, v1 // (op 4735) 
v_lshrrev_b32 v13, 4, v11 // (op 4735) 
s_mul_i32 s1, s2, s9 // (op 4480) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 4480) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 4480) least significant
s_mul_hi_u32 s19, s2, s8 // (op 4480) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4480) most significant: sum
s_add_u32 s1, s1, s19 // (op 4480) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4480) 

v_lshlrev_b32 v14, 2, v13 // (op 4735) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 4735) 


v_lshl_add_u32 v10, s16, 6, v13 // (op 4735) 

v_lshrrev_b32 v13, 2, v11 // (op 4735) 

v_and_b32 v11, 3, v13 // (op 4735) 

v_and_b32 v13, 3, v1 // (op 4735) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 4735) 


v_lshrrev_b32 v11, 6, v1 // (op 4735) 
v_and_b32 v13, 1, v11 // (op 4735) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 4735) 


v_lshl_add_u32 v13, s17, 7, v11 // (op 4735) 

v_mul_lo_u32 v11, v13, s4 // (op 4735) 

v_add_lshl_u32 v13, v10, v11, 2 // (op 4735) 


s_mov_b32 s40, s6 // (op 4480) reg expression
s_mov_b32 s41, s7 // (op 4480) reg expression
s_mov_b32 s42, s18 // (op 4480) reg expression
s_mov_b32 s43, 131072 // (op 4480) call()

s_mul_i32 s1, s4, s9 // (op 4737) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 4737) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 4737) least significant
s_mul_hi_u32 s7, s4, s8 // (op 4737) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4737) most significant: sum
s_add_u32 s1, s1, s7 // (op 4737) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4737) 

s_mov_b32 s44, s10 // (op 4737) reg expression
s_mov_b32 s45, s11 // (op 4737) reg expression
s_mov_b32 s46, s6 // (op 4737) reg expression
s_mov_b32 s47, 131072 // (op 4737) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 336) call()
v_accvgpr_write a5, 0.00000 // (op 336) call()
v_accvgpr_write a6, 0.00000 // (op 336) call()
v_accvgpr_write a7, 0.00000 // (op 336) call()
v_accvgpr_write a8, 0.00000 // (op 512) call()
v_accvgpr_write a9, 0.00000 // (op 512) call()
v_accvgpr_write a10, 0.00000 // (op 512) call()
v_accvgpr_write a11, 0.00000 // (op 512) call()
v_accvgpr_write a12, 0.00000 // (op 688) call()
v_accvgpr_write a13, 0.00000 // (op 688) call()
v_accvgpr_write a14, 0.00000 // (op 688) call()
v_accvgpr_write a15, 0.00000 // (op 688) call()
v_accvgpr_write a16, 0.00000 // (op 952) call()
v_accvgpr_write a17, 0.00000 // (op 952) call()
v_accvgpr_write a18, 0.00000 // (op 952) call()
v_accvgpr_write a19, 0.00000 // (op 952) call()
v_accvgpr_write a20, 0.00000 // (op 1261) call()
v_accvgpr_write a21, 0.00000 // (op 1261) call()
v_accvgpr_write a22, 0.00000 // (op 1261) call()
v_accvgpr_write a23, 0.00000 // (op 1261) call()
v_accvgpr_write a24, 0.00000 // (op 1262) call()
v_accvgpr_write a25, 0.00000 // (op 1262) call()
v_accvgpr_write a26, 0.00000 // (op 1262) call()
v_accvgpr_write a27, 0.00000 // (op 1262) call()
v_accvgpr_write a28, 0.00000 // (op 1263) call()
v_accvgpr_write a29, 0.00000 // (op 1263) call()
v_accvgpr_write a30, 0.00000 // (op 1263) call()
v_accvgpr_write a31, 0.00000 // (op 1263) call()
v_lshrrev_b32 v14, 3, v1 // (op 4240) 
v_lshlrev_b32 v15, 1, v14 // (op 4240) 

v_lshl_add_u32 v14, s16, 6, v15 // (op 4240) 

v_and_b32 v15, -8, v1 // (op 4240) 
v_sub_u32 v16, v1, v15 // (op 4240) 

v_lshlrev_b32 v15, 4, v16 // (op 4240) 

v_mul_lo_u32 v16, v14, s12 // (op 4240) 
v_add_u32 v16, v16, v15 // (op 4240) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 4242) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4242) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4242) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4242) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4242) most significant: sum
s_add_u32 s1, s1, s7 // (op 4242) most significant: sum


s_mov_b32 s48, s20 // (op 4242) reg expression
s_mov_b32 s49, s21 // (op 4242) reg expression
s_mov_b32 s50, s0 // (op 4242) reg expression
s_mov_b32 s51, 131072 // (op 4242) call()

v_lshrrev_b32 v14, 3, v1 // (op 4259) 
v_lshlrev_b32 v15, 1, v14 // (op 4259) 

v_lshl_add_u32 v14, s16, 6, v15 // (op 4259) 

v_and_b32 v15, -8, v1 // (op 4259) 
v_sub_u32 v17, v1, v15 // (op 4259) 

v_lshlrev_b32 v15, 4, v17 // (op 4259) 

v_mul_lo_u32 v17, v14, s12 // (op 4259) 
v_add_u32 v17, v17, v15 // (op 4259) 


s_mul_i32 s1, s12, s23 // (op 4261) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4261) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4261) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4261) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4261) most significant: sum
s_add_u32 s1, s1, s7 // (op 4261) most significant: sum


s_mov_b32 s48, s20 // (op 4261) reg expression
s_mov_b32 s49, s21 // (op 4261) reg expression
s_mov_b32 s50, s0 // (op 4261) reg expression
s_mov_b32 s51, 131072 // (op 4261) call()

v_and_b32 v14, -4, v1 // (op 4304) 
v_sub_u32 v15, v1, v14 // (op 4304) 

v_lshlrev_b32 v14, 5, v15 // (op 4304) 

v_mov_b32 v19, 0 // (op 4304) convert
v_mov_b32 v18, v14 // (op 4304) convert

v_lshrrev_b32 v14, 2, v1 // (op 4304) 
v_lshlrev_b32 v15, 1, v14 // (op 4304) 

v_lshl_add_u32 v14, s17, 7, v15 // (op 4304) 

v_mul_lo_u32 v21, v14, s25 // (op 4304) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 4304) most significant: high of low * low
v_mul_lo_u32 v20, v14, s24 // (op 4304) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 4304) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 4304) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 4304) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 4304) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 4304) 

v_mov_b32 v14, v18 // (op 4304) convert

s_mul_i32 s1, s24, s9 // (op 4306) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4306) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4306) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4306) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4306) most significant: sum
s_add_u32 s1, s1, s7 // (op 4306) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4306) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 4306) 

s_mov_b32 s52, s26 // (op 4306) reg expression
s_mov_b32 s53, s27 // (op 4306) reg expression
s_mov_b32 s54, s0 // (op 4306) reg expression
s_mov_b32 s55, 131072 // (op 4306) call()

v_and_b32 v15, -4, v1 // (op 4323) 
v_sub_u32 v18, v1, v15 // (op 4323) 

v_lshlrev_b32 v15, 5, v18 // (op 4323) 

v_mov_b32 v19, 0 // (op 4323) convert
v_mov_b32 v18, v15 // (op 4323) convert

v_lshrrev_b32 v15, 2, v1 // (op 4323) 
v_lshlrev_b32 v20, 1, v15 // (op 4323) 

v_lshl_add_u32 v15, s17, 7, v20 // (op 4323) 

v_mul_lo_u32 v21, v15, s25 // (op 4323) most significant: low of low * high
v_mul_hi_u32 v22, v15, s24 // (op 4323) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 4323) least significant: low of low * low
v_add_u32 v21, v21, v22 // (op 4323) most significant: sum


v_add_co_u32 v22, s[0:1], v18, v20 // (op 4323) least significant half
v_addc_co_u32 v23, s[0:1], v19, v21, s[0:1] // (op 4323) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 4323) 


v_lshrrev_b64 v[18:19], 3, v[22:23] // (op 4323) 

v_mov_b32 v15, v18 // (op 4323) convert

s_mul_i32 s1, s24, s9 // (op 4325) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4325) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4325) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4325) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4325) most significant: sum
s_add_u32 s1, s1, s7 // (op 4325) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4325) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 4325) 

s_mov_b32 s52, s26 // (op 4325) reg expression
s_mov_b32 s53, s27 // (op 4325) reg expression
s_mov_b32 s54, s0 // (op 4325) reg expression
s_mov_b32 s55, 131072 // (op 4325) call()

v_lshrrev_b32 v18, 2, v1 // (op 4368) 
v_lshl_add_u32 v19, s16, 6, v18 // (op 4368) 

v_and_b32 v18, -4, v1 // (op 4368) 
v_sub_u32 v20, v1, v18 // (op 4368) 

v_mul_lo_u32 v18, v19, s28 // (op 4368) 
v_add_u32 v18, v18, v20 // (op 4368) 


s_mul_i32 s1, s28, s23 // (op 4370) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4370) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4370) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4370) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4370) most significant: sum
s_add_u32 s1, s1, s7 // (op 4370) most significant: sum


s_mov_b32 s56, s30 // (op 4370) reg expression
s_mov_b32 s57, s31 // (op 4370) reg expression
s_mov_b32 s58, s0 // (op 4370) reg expression
s_mov_b32 s59, 131072 // (op 4370) call()

v_lshrrev_b32 v19, 2, v1 // (op 4387) 
v_lshl_add_u32 v20, s16, 6, v19 // (op 4387) 

v_and_b32 v19, -4, v1 // (op 4387) 
v_sub_u32 v21, v1, v19 // (op 4387) 

v_mul_lo_u32 v19, v20, s28 // (op 4387) 
v_add_u32 v19, v19, v21 // (op 4387) 


s_mul_i32 s1, s28, s23 // (op 4389) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4389) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4389) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4389) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4389) most significant: sum
s_add_u32 s1, s1, s7 // (op 4389) most significant: sum


s_mov_b32 s56, s30 // (op 4389) reg expression
s_mov_b32 s57, s31 // (op 4389) reg expression
s_mov_b32 s58, s0 // (op 4389) reg expression
s_mov_b32 s59, 131072 // (op 4389) call()

v_and_b32 v20, -4, v1 // (op 4432) 
v_sub_u32 v21, v1, v20 // (op 4432) 

v_lshrrev_b32 v20, 2, v1 // (op 4432) 
v_lshlrev_b32 v22, 1, v20 // (op 4432) 

v_lshl_add_u32 v20, s17, 7, v22 // (op 4432) 

v_mul_lo_u32 v22, v20, s32 // (op 4432) 

v_add_u32 v20, v21, v22 // (op 4432) 


s_mul_i32 s1, s32, s9 // (op 4434) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4434) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4434) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4434) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4434) most significant: sum
s_add_u32 s1, s1, s7 // (op 4434) most significant: sum


s_mov_b32 s60, s34 // (op 4434) reg expression
s_mov_b32 s61, s35 // (op 4434) reg expression
s_mov_b32 s62, s0 // (op 4434) reg expression
s_mov_b32 s63, 131072 // (op 4434) call()

v_and_b32 v21, -4, v1 // (op 4451) 
v_sub_u32 v22, v1, v21 // (op 4451) 

v_lshrrev_b32 v21, 2, v1 // (op 4451) 
v_lshlrev_b32 v23, 1, v21 // (op 4451) 

v_lshl_add_u32 v21, s17, 7, v23 // (op 4451) 

v_mul_lo_u32 v23, v21, s32 // (op 4451) 

v_add_u32 v21, v22, v23 // (op 4451) 


s_mul_i32 s1, s32, s9 // (op 4453) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4453) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4453) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4453) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4453) most significant: sum
s_add_u32 s1, s1, s7 // (op 4453) most significant: sum


s_mov_b32 s60, s34 // (op 4453) reg expression
s_mov_b32 s61, s35 // (op 4453) reg expression
s_mov_b32 s62, s0 // (op 4453) reg expression
s_mov_b32 s63, 131072 // (op 4453) call()


s_lshl_b64 s[0:1], s[24:25], 2 // (op 3984) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 3984) 


v_mov_b32 v24, v14 // (op 3984) 
buffer_load_dwordx4 v[26:29], v24, s[52:55], 0 offen // (op 3984) Load value
v_add_u32 v24, v24, v22 // (op 3984) 
buffer_load_dwordx4 v[30:33], v24, s[52:55], 0 offen // (op 3984) Load value



v_mov_b64 v[22:23], s[32:33] // (op 3987) convert

v_mov_b32 v24, v20 // (op 3987) 
buffer_load_ubyte v34, v24, s[60:63], 0 offen offset: 0 // (op 3987) Load value
v_add_u32 v24, v24, v22 // (op 3987) 
buffer_load_ubyte v35, v24, s[60:63], 0 offen offset: 0 // (op 3987) Load value



v_mov_b64 v[22:23], s[12:13] // (op 3990) convert

v_mov_b32 v24, v16 // (op 3990) 
buffer_load_dwordx4 v[36:39], v24, s[48:51], 0 offen // (op 3990) Load value
v_add_u32 v24, v24, v22 // (op 3990) 
buffer_load_dwordx4 v[40:43], v24, s[48:51], 0 offen // (op 3990) Load value





buffer_load_ubyte v22, v18, s[56:59], 0 offen offset: 0 // (op 3993) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 3996) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 3996) 


v_mov_b32 v23, v14 // (op 3996) 
buffer_load_dwordx4 v[44:47], v23, s[52:55], 0 offen offset:64 // (op 3996) Load value
v_add_u32 v23, v23, v24 // (op 3996) 
buffer_load_dwordx4 v[48:51], v23, s[52:55], 0 offen offset:64 // (op 3996) Load value




v_mov_b64 v[24:25], s[32:33] // (op 3999) convert

v_mov_b32 v14, v20 // (op 3999) 
buffer_load_ubyte v52, v14, s[60:63], 0 offen offset:4 // (op 3999) Load value
v_add_u32 v14, v14, v24 // (op 3999) 
buffer_load_ubyte v53, v14, s[60:63], 0 offen offset:4 // (op 3999) Load value




v_mov_b64 v[24:25], s[12:13] // (op 4002) convert

v_mov_b32 v14, v16 // (op 4002) 
buffer_load_dwordx4 v[54:57], v14, s[48:51], 0 offen offset:128 // (op 4002) Load value
v_add_u32 v14, v14, v24 // (op 4002) 
buffer_load_dwordx4 v[58:61], v14, s[48:51], 0 offen offset:128 // (op 4002) Load value






buffer_load_ubyte v14, v18, s[56:59], 0 offen offset:4 // (op 4005) Load value

v_and_b32 v16, -4, v1 // (op 4581) 
v_sub_u32 v18, v1, v16 // (op 4581) 

v_lshlrev_b32 v16, 5, v18 // (op 4581) 

v_lshrrev_b32 v18, 2, v1 // (op 4581) 
v_lshl_add_u32 v20, v18, 8, v16 // (op 4581) 


v_lshlrev_b32 v16, 2, v20 // (op 4581) 

v_lshrrev_b32 v18, 3, v16 // (op 4581) 




s_waitcnt vmcnt(13)
ds_write_b128 v18, v[26:29] // (op 4007) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[30:33] offset:64 // (op 4007) Store local data 

v_and_b32 v16, -4, v1 // (op 4697) 
v_sub_u32 v18, v1, v16 // (op 4697) 

v_lshrrev_b32 v16, 2, v1 // (op 4697) 
v_lshl_add_u32 v20, v16, 3, v18 // (op 4697) 





s_waitcnt vmcnt(11)
ds_write_b8 v20, v34 offset:8192 // (op 4008) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v20, v35 offset:8196 // (op 4008) Store local data 

v_and_b32 v16, -8, v1 // (op 4523) 
v_sub_u32 v18, v1, v16 // (op 4523) 

v_lshlrev_b32 v16, 4, v18 // (op 4523) 

v_lshrrev_b32 v18, 3, v1 // (op 4523) 
v_lshl_add_u32 v20, v18, 8, v16 // (op 4523) 





s_waitcnt vmcnt(9)
ds_write_b128 v20, v[36:39] offset:8704 // (op 4009) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v20, v[40:43] offset:8832 // (op 4009) Store local data 

v_and_b32 v16, -4, v1 // (op 4639) 
v_sub_u32 v18, v1, v16 // (op 4639) 

v_and_b32 v16, 4294967292, v1 // (op 4639) 
v_add_u32 v20, v16, v18 // (op 4639) 





s_waitcnt vmcnt(7)
ds_write_b8 v20, v22 offset:16896 // (op 4010) Store local data 

v_and_b32 v16, -8, v1 // (op 4507) 
v_sub_u32 v18, v1, v16 // (op 4507) 

v_lshlrev_b32 v16, 4, v18 // (op 4507) 

v_lshrrev_b32 v18, 3, v1 // (op 4507) 
v_lshl_add_u32 v20, v18, 8, v16 // (op 4507) 


v_and_b32 v16, -4, v1 // (op 4565) 
v_sub_u32 v18, v1, v16 // (op 4565) 

v_lshlrev_b32 v16, 5, v18 // (op 4565) 

v_lshrrev_b32 v18, 2, v1 // (op 4565) 
v_lshl_add_u32 v23, v18, 8, v16 // (op 4565) 


v_lshlrev_b32 v16, 2, v23 // (op 4565) 

v_lshrrev_b32 v18, 3, v16 // (op 4565) 

v_and_b32 v16, -4, v1 // (op 4623) 
v_sub_u32 v23, v1, v16 // (op 4623) 

v_and_b32 v16, 4294967292, v1 // (op 4623) 
v_add_u32 v24, v16, v23 // (op 4623) 


v_and_b32 v16, -4, v1 // (op 4681) 
v_sub_u32 v23, v1, v16 // (op 4681) 

v_lshrrev_b32 v16, 2, v1 // (op 4681) 
v_lshl_add_u32 v25, v16, 3, v23 // (op 4681) 


v_and_b32 v16, -8, v1 // (op 4749) 
v_sub_u32 v23, v1, v16 // (op 4749) 

v_lshlrev_b32 v16, 4, v23 // (op 4749) 

v_lshrrev_b32 v23, 3, v1 // (op 4749) 
v_lshl_add_u32 v62, v23, 8, v16 // (op 4749) 


v_and_b32 v16, -4, v1 // (op 4778) 
v_sub_u32 v23, v1, v16 // (op 4778) 

v_lshlrev_b32 v16, 5, v23 // (op 4778) 

v_lshrrev_b32 v23, 2, v1 // (op 4778) 
v_lshl_add_u32 v63, v23, 8, v16 // (op 4778) 


v_lshlrev_b32 v16, 2, v63 // (op 4778) 

v_lshrrev_b32 v23, 3, v16 // (op 4778) 

v_and_b32 v16, -4, v1 // (op 4807) 
v_sub_u32 v63, v1, v16 // (op 4807) 

v_and_b32 v16, 4294967292, v1 // (op 4807) 
v_add_u32 v64, v16, v63 // (op 4807) 


v_and_b32 v16, -4, v1 // (op 4836) 
v_sub_u32 v63, v1, v16 // (op 4836) 

v_lshrrev_b32 v16, 2, v1 // (op 4836) 
v_lshl_add_u32 v65, v16, 3, v63 // (op 4836) 


s_mov_b32 s0, 0 // (op 1047) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 1046) 
s_mov_b32 s1, s6 // (op 1046) convert

s_and_b32 s6, s1, 4294967294 // (op 1046) 

s_cmp_lt_u32 s0, s6 // (op 1046) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_1046 // (op 1046) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_1046 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoop_1046:

s_barrier  // (op 4916) 
s_add_u32 s0, s0, 2 // (op 1048) 



s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[66:67], 3, s[6:7] // (op 10) 


v_mov_b32 v16, v15 // (op 10) 
v_mov_b32 v63, v16 // (op 10) 
buffer_load_dwordx4 v[26:29], v63, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v63, v63, v66 // (op 10) 
ds_read_b128 v[68:71], v3 offset:8704 // (op 70) Load local data 
ds_read_b128 v[72:75], v3 offset:8768 // (op 70) Load local data 
buffer_load_dwordx4 v[30:33], v63, s[52:55], 0 offen offset:128 // (op 10) Load value



s_barrier  // (op 4929) 
v_mov_b64 v[66:67], s[32:33] // (op 25) convert




v_mov_b32 v63, v21 // (op 25) 
v_mov_b32 v76, v63 // (op 25) 
buffer_load_ubyte v34, v76, s[60:63], 0 offen offset:8 // (op 25) Load value
v_add_u32 v76, v76, v66 // (op 25) 
ds_read_b128 v[78:81], v4 // (op 75) Load local data 
buffer_load_ubyte v35, v76, s[60:63], 0 offen offset:8 // (op 25) Load value



s_barrier  // (op 4943) 
v_mov_b64 v[66:67], s[12:13] // (op 4) convert




ds_read_u8 v76, v0 offset:16896 // (op 80) Load local data 
v_mov_b32 v77, v17 // (op 4) 
s_barrier  // (op 4957) 
v_mov_b32 v82, v77 // (op 4) 



ds_read_u8 v83, v5 offset:8192 // (op 85) Load local data 
buffer_load_dwordx4 v[36:39], v82, s[48:51], 0 offen offset:256 // (op 4) Load value



v_add_u32 v82, v82, v66 // (op 4) 
ds_read_b128 v[84:87], v4 offset:2048 // (op 371) Load local data 
buffer_load_dwordx4 v[40:43], v82, s[48:51], 0 offen offset:256 // (op 4) Load value







ds_read_u8 v82, v5 offset:8320 // (op 375) Load local data 
v_mov_b32 v66, v19 // (op 17) 




ds_read_b128 v[88:91], v4 offset:4096 // (op 547) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:75], v[78:81], a[0:3], v76, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 88) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[68:75], v[84:87], a[4:7], v76, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 376) 
ds_read_u8 v67, v5 offset:8448 // (op 551) Load local data 
buffer_load_ubyte v22, v66, s[56:59], 0 offen offset:8 // (op 17) Load value



s_barrier  // (op 4980) 
ds_read_b128 v[92:95], v4 offset:6144 // (op 723) Load local data 






ds_read_u8 v96, v5 offset:8576 // (op 727) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:75], v[88:91], a[8:11], v76, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 552) 



ds_read_b128 v[98:101], v3 offset:12800 // (op 1080) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:75], v[92:95], a[12:15], v76, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 728) 
ds_read_b128 v[102:105], v3 offset:12864 // (op 1080) Load local data 





ds_read_u8 v97, v0 offset:17024 // (op 1083) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v23, v[44:47] offset:17152 // (op 253) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[78:81], a[16:19], v97, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1223) 
s_waitcnt vmcnt(12)
ds_write_b128 v23, v[48:51] offset:17216 // (op 253) Store local data 


s_barrier  // (op 4992) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[84:87], a[20:23], v97, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1177) 





s_waitcnt vmcnt(11)
ds_write_b8 v65, v52 offset:25344 // (op 255) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[88:91], a[24:27], v97, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1131) 
s_waitcnt vmcnt(10)
ds_write_b8 v65, v53 offset:25348 // (op 255) Store local data 


s_barrier  // (op 4974) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[92:95], a[28:31], v97, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1085) 







s_waitcnt vmcnt(9)
ds_write_b128 v62, v[54:57] offset:25856 // (op 243) Store local data 
s_waitcnt vmcnt(8)
ds_write_b128 v62, v[58:61] offset:25984 // (op 243) Store local data 
s_barrier  // (op 4986) 



s_waitcnt vmcnt(7)
ds_write_b8 v64, v14 offset:34048 // (op 254) Store local data 

s_lshl_b64 s[6:7], s[24:25], 2 // (op 256) 
v_lshrrev_b64 v[68:69], 3, s[6:7] // (op 256) 


v_mov_b32 v67, v16 // (op 256) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 4971) 
buffer_load_dwordx4 v[44:47], v67, s[52:55], 0 offen offset:192 // (op 256) Load value



v_add_u32 v67, v67, v68 // (op 256) 
ds_read_b128 v[78:81], v7 offset:25856 // (op 245) Load local data 
ds_read_b128 v[82:85], v7 offset:25920 // (op 245) Load local data 
buffer_load_dwordx4 v[48:51], v67, s[52:55], 0 offen offset:192 // (op 256) Load value



s_barrier  // (op 4977) 




v_add_u32 v15, 128, v15 // (op 4315) 
v_mov_b64 v[68:69], s[32:33] // (op 258) convert

v_mov_b32 v16, v63 // (op 258) 
ds_read_b128 v[70:73], v8 offset:17152 // (op 247) Load local data 
buffer_load_ubyte v52, v16, s[60:63], 0 offen offset:12 // (op 258) Load value
v_add_u32 v16, v16, v68 // (op 258) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4983) 
buffer_load_ubyte v53, v16, s[60:63], 0 offen offset:12 // (op 258) Load value






ds_read_u8 v16, v6 offset:34048 // (op 249) Load local data 

s_barrier  // (op 4989) 
v_add_u32 v21, 8, v21 // (op 4443) 



ds_read_u8 v63, v9 offset:25344 // (op 251) Load local data 
v_mov_b64 v[68:69], s[12:13] // (op 242) convert




v_mov_b32 v67, v77 // (op 242) 
buffer_load_dwordx4 v[54:57], v67, s[48:51], 0 offen offset:384 // (op 242) Load value
v_add_u32 v67, v67, v68 // (op 242) 
ds_read_b128 v[86:89], v8 offset:19200 // (op 381) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[70:73], a[0:3], v16, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 252) 



ds_read_u8 v74, v9 offset:25472 // (op 385) Load local data 
buffer_load_dwordx4 v[58:61], v67, s[48:51], 0 offen offset:384 // (op 242) Load value






ds_read_b128 v[90:93], v8 offset:21248 // (op 557) Load local data 






v_add_u32 v17, 256, v17 // (op 4251) 
ds_read_u8 v67, v9 offset:25600 // (op 561) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:85], v[86:89], a[4:7], v16, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 377) 



ds_read_b128 v[94:97], v8 offset:23296 // (op 733) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[90:93], a[8:11], v16, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 553) 
buffer_load_ubyte v14, v66, s[56:59], 0 offen offset:12 // (op 257) Load value




s_barrier  // (op 5049) 
s_barrier  // (op 4932) 
v_add_u32 v19, 8, v19 // (op 4379) 



ds_read_u8 v66, v9 offset:25728 // (op 737) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[30:33] offset:64 // (op 13) Store local data 



s_barrier  // (op 5053) 
s_barrier  // (op 4960) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:85], v[94:97], a[12:15], v16, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 729) 





ds_read_b128 v[78:81], v7 offset:29952 // (op 1088) Load local data 
ds_read_b128 v[82:85], v7 offset:30016 // (op 1088) Load local data 
s_waitcnt vmcnt(11)
ds_write_b8 v25, v34 offset:8192 // (op 30) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v25, v35 offset:8196 // (op 30) Store local data 



s_barrier  // (op 5077) 
s_barrier  // (op 4919) 
ds_read_u8 v16, v6 offset:34176 // (op 1091) Load local data 



s_waitcnt vmcnt(9)
ds_write_b128 v20, v[36:39] offset:8704 // (op 6) Store local data 
s_barrier  // (op 5081) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:85], v[70:73], a[16:19], v16, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1224) 
s_waitcnt vmcnt(8)
ds_write_b128 v20, v[40:43] offset:8832 // (op 6) Store local data 


s_barrier  // (op 4946) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[86:89], a[20:23], v16, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1178) 





s_waitcnt vmcnt(7)
ds_write_b8 v24, v22 offset:16896 // (op 21) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:85], v[90:93], a[24:27], v16, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1132) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:85], v[94:97], a[28:31], v16, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1086) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 1046) 
s_mov_b32 s1, s6 // (op 1046) convert

s_and_b32 s6, s1, 4294967294 // (op 1046) 

s_cmp_lt_u32 s0, s6 // (op 1046) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoop_1046 // (op 1046) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoop_1046 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_1046:

















v_lshrrev_b32 v15, 3, v1 // (op 4207) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4207) 
v_lshlrev_b32 v16, 1, v15 // (op 4207) 

s_lshr_b32 s6, s0, 1 // (op 4207) 

v_lshl_add_u32 v15, s16, 6, v16 // (op 4207) 

v_and_b32 v16, -8, v1 // (op 4207) 
v_sub_u32 v17, v1, v16 // (op 4207) 

v_lshlrev_b32 v16, 4, v17 // (op 4207) 

v_lshl_add_u32 v17, s6, 8, v16 // (op 4207) 


v_mul_lo_u32 v16, v15, s12 // (op 4207) 
v_add_u32 v16, v16, v17 // (op 4207) 


s_mul_i32 s1, s12, s23 // (op 4209) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 4209) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 4209) least significant
s_mul_hi_u32 s7, s12, s22 // (op 4209) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4209) most significant: sum
s_add_u32 s1, s1, s7 // (op 4209) most significant: sum


s_mov_b32 s48, s20 // (op 4209) reg expression
s_mov_b32 s49, s21 // (op 4209) reg expression
s_mov_b32 s50, s0 // (op 4209) reg expression
s_mov_b32 s51, 131072 // (op 4209) call()

v_and_b32 v15, -4, v1 // (op 4281) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4281) 
v_sub_u32 v17, v1, v15 // (op 4281) 

s_lshr_b32 s6, s0, 1 // (op 4281) 

v_lshlrev_b32 v15, 5, v17 // (op 4281) 

v_lshl_add_u32 v17, s6, 8, v15 // (op 4281) 


v_mov_b32 v19, 0 // (op 4281) convert
v_mov_b32 v18, v17 // (op 4281) convert

v_lshrrev_b32 v15, 2, v1 // (op 4281) 
v_lshlrev_b32 v17, 1, v15 // (op 4281) 

v_lshl_add_u32 v15, s17, 7, v17 // (op 4281) 

v_mul_lo_u32 v21, v15, s25 // (op 4281) most significant: low of low * high
v_mul_hi_u32 v17, v15, s24 // (op 4281) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 4281) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 4281) most significant: sum


v_add_co_u32 v22, s[0:1], v18, v20 // (op 4281) least significant half
v_addc_co_u32 v23, s[0:1], v19, v21, s[0:1] // (op 4281) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 4281) 


v_lshrrev_b64 v[20:21], 3, v[22:23] // (op 4281) 

v_mov_b32 v15, v20 // (op 4281) convert

s_mul_i32 s1, s24, s9 // (op 4283) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 4283) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 4283) least significant
s_mul_hi_u32 s7, s24, s8 // (op 4283) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4283) most significant: sum
s_add_u32 s1, s1, s7 // (op 4283) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 4283) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 4283) 

s_mov_b32 s52, s26 // (op 4283) reg expression
s_mov_b32 s53, s27 // (op 4283) reg expression
s_mov_b32 s54, s0 // (op 4283) reg expression
s_mov_b32 s55, 131072 // (op 4283) call()

v_lshrrev_b32 v17, 2, v1 // (op 4345) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4345) 
v_lshl_add_u32 v18, s16, 6, v17 // (op 4345) 

s_lshr_b32 s6, s0, 1 // (op 4345) 

v_and_b32 v17, -4, v1 // (op 4345) 
v_sub_u32 v19, v1, v17 // (op 4345) 

v_lshl_add_u32 v17, s6, 3, v19 // (op 4345) 


v_mul_lo_u32 v19, v18, s28 // (op 4345) 
v_add_u32 v19, v19, v17 // (op 4345) 


s_mul_i32 s1, s28, s23 // (op 4347) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 4347) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 4347) least significant
s_mul_hi_u32 s7, s28, s22 // (op 4347) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4347) most significant: sum
s_add_u32 s1, s1, s7 // (op 4347) most significant: sum


s_mov_b32 s56, s30 // (op 4347) reg expression
s_mov_b32 s57, s31 // (op 4347) reg expression
s_mov_b32 s58, s0 // (op 4347) reg expression
s_mov_b32 s59, 131072 // (op 4347) call()

v_and_b32 v17, -4, v1 // (op 4409) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 4409) 
v_sub_u32 v18, v1, v17 // (op 4409) 

s_lshr_b32 s6, s0, 1 // (op 4409) 

v_lshrrev_b32 v17, 2, v1 // (op 4409) 
v_lshl_add_u32 v20, s6, 3, v18 // (op 4409) 


v_lshlrev_b32 v18, 1, v17 // (op 4409) 

v_lshl_add_u32 v17, s17, 7, v18 // (op 4409) 

v_mul_lo_u32 v18, v17, s32 // (op 4409) 

v_add_u32 v17, v20, v18 // (op 4409) 


s_mul_i32 s1, s32, s9 // (op 4411) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 4411) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 4411) least significant
s_mul_hi_u32 s7, s32, s8 // (op 4411) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 4411) most significant: sum
s_add_u32 s1, s1, s7 // (op 4411) most significant: sum


s_mov_b32 s60, s34 // (op 4411) reg expression
s_mov_b32 s61, s35 // (op 4411) reg expression
s_mov_b32 s62, s0 // (op 4411) reg expression
s_mov_b32 s63, 131072 // (op 4411) call()

v_and_b32 v18, -8, v1 // (op 4492) 
v_sub_u32 v20, v1, v18 // (op 4492) 

v_lshlrev_b32 v18, 4, v20 // (op 4492) 

v_lshrrev_b32 v20, 3, v1 // (op 4492) 
v_lshl_add_u32 v21, v20, 8, v18 // (op 4492) 


v_and_b32 v18, -4, v1 // (op 4552) 
v_sub_u32 v20, v1, v18 // (op 4552) 

v_lshlrev_b32 v18, 5, v20 // (op 4552) 

v_lshrrev_b32 v20, 2, v1 // (op 4552) 
v_lshl_add_u32 v22, v20, 8, v18 // (op 4552) 


v_lshlrev_b32 v18, 2, v22 // (op 4552) 

v_lshrrev_b32 v20, 3, v18 // (op 4552) 

v_and_b32 v18, -4, v1 // (op 4610) 
v_sub_u32 v22, v1, v18 // (op 4610) 

v_and_b32 v18, 4294967292, v1 // (op 4610) 
v_add_u32 v23, v18, v22 // (op 4610) 


v_and_b32 v18, -4, v1 // (op 4668) 
v_sub_u32 v22, v1, v18 // (op 4668) 

v_lshrrev_b32 v18, 2, v1 // (op 4668) 
v_lshl_add_u32 v24, v18, 3, v22 // (op 4668) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 1021) 
s_and_b32 s6, s0, 4294967294 // (op 1021) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 1020) 
s_mov_b32 s7, s0 // (op 1020) convert

s_cmp_lt_u32 s6, s7 // (op 1020) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_1020 // (op 1020) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_1020 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_1020:

s_add_u32 s6, s6, 1 // (op 1022) 
v_mov_b64 v[26:27], s[12:13] // (op 176) convert

v_mov_b32 v14, v16 // (op 176) 
v_mov_b32 v18, v14 // (op 176) 
buffer_load_dwordx4 v[28:31], v18, s[48:51], 0 offen // (op 176) Load value
v_add_u32 v18, v18, v26 // (op 176) 
buffer_load_dwordx4 v[32:35], v18, s[48:51], 0 offen // (op 176) Load value



v_add_u32 v16, 128, v16 // (op 4197) 

s_barrier  // (op 4922) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 190) 



v_lshrrev_b64 v[26:27], 3, s[0:1] // (op 190) 


v_mov_b32 v14, v15 // (op 190) 
v_mov_b32 v18, v14 // (op 190) 
buffer_load_dwordx4 v[36:39], v18, s[52:55], 0 offen // (op 190) Load value
v_add_u32 v18, v18, v26 // (op 190) 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[28:31] offset:8704 // (op 177) Store local data 
buffer_load_dwordx4 v[40:43], v18, s[52:55], 0 offen // (op 190) Load value



v_add_u32 v15, 64, v15 // (op 4273) 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[32:35] offset:8832 // (op 177) Store local data 



v_mov_b32 v14, v19 // (op 191) 

s_barrier  // (op 4935) 
buffer_load_ubyte v18, v14, s[56:59], 0 offen offset: 0 // (op 191) Load value




v_add_u32 v19, 4, v19 // (op 4337) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[36:39] // (op 187) Store local data 

v_mov_b64 v[34:35], s[32:33] // (op 192) convert

v_mov_b32 v14, v17 // (op 192) 
v_mov_b32 v22, v14 // (op 192) 
buffer_load_ubyte v32, v22, s[60:63], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v22, v22, v34 // (op 192) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[40:43] offset:64 // (op 187) Store local data 
buffer_load_ubyte v33, v22, s[60:63], 0 offen offset: 0 // (op 192) Load value




v_add_u32 v17, 4, v17 // (op 4401) 
s_barrier  // (op 4949) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 4925) 






ds_read_b128 v[34:37], v3 offset:8704 // (op 179) Load local data 
ds_read_b128 v[38:41], v3 offset:8768 // (op 179) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v23, v18 offset:16896 // (op 188) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 4939) 



s_barrier  // (op 4963) 
ds_read_b128 v[28:31], v4 // (op 181) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 4953) 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v32 offset:8192 // (op 189) Store local data 



ds_read_u8 v14, v0 offset:16896 // (op 183) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v33 offset:8196 // (op 189) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 4967) 



ds_read_u8 v18, v5 offset:8192 // (op 185) Load local data 



ds_read_b128 v[42:45], v4 offset:2048 // (op 350) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[28:31], a[0:3], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 186) 



ds_read_u8 v22, v5 offset:8320 // (op 354) Load local data 



ds_read_b128 v[46:49], v4 offset:4096 // (op 526) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[42:45], a[4:7], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 355) 



ds_read_u8 v25, v5 offset:8448 // (op 530) Load local data 



ds_read_b128 v[50:53], v4 offset:6144 // (op 702) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:41], v[46:49], a[8:11], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 531) 



ds_read_u8 v26, v5 offset:8576 // (op 706) Load local data 



ds_read_b128 v[54:57], v3 offset:12800 // (op 964) Load local data 
ds_read_b128 v[58:61], v3 offset:12864 // (op 964) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:41], v[50:53], a[12:15], v14, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 707) 





ds_read_u8 v14, v0 offset:17024 // (op 967) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:61], v[28:31], a[16:19], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 969) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:61], v[42:45], a[20:23], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1000) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:61], v[46:49], a[24:27], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1031) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[54:61], v[50:53], a[28:31], v14, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1062) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 1020) 
s_mov_b32 s7, s0 // (op 1020) convert

s_cmp_lt_u32 s6, s7 // (op 1020) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_1020 // (op 1020) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_1020 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_1020:























s_barrier  // (op 5061) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_133
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 



v_mov_b32 v0, v13 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 56) Store value


v_accvgpr_read v6, a4 // (op 1869) 
v_accvgpr_read v7, a5 // (op 1869) 
v_accvgpr_read v8, a6 // (op 1869) 
v_accvgpr_read v9, a7 // (op 1869) 
v_mul_f32 v14, v11, v6 // (op 1869) 
v_mul_f32 v15, v11, v7 // (op 1869) 
v_mul_f32 v16, v11, v8 // (op 1869) 
v_mul_f32 v17, v11, v9 // (op 1869) 



s_lshl_b32 s0, s4, 7 // (op 1871) 
v_add_u32 v0, v13, s0 // (op 1871) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1871) Store value


v_accvgpr_read v6, a8 // (op 1881) 
v_accvgpr_read v7, a9 // (op 1881) 
v_accvgpr_read v8, a10 // (op 1881) 
v_accvgpr_read v9, a11 // (op 1881) 
v_mul_f32 v14, v11, v6 // (op 1881) 
v_mul_f32 v15, v11, v7 // (op 1881) 
v_mul_f32 v16, v11, v8 // (op 1881) 
v_mul_f32 v17, v11, v9 // (op 1881) 



s_lshl_b32 s0, s4, 8 // (op 1883) 
v_add_u32 v0, v13, s0 // (op 1883) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1883) Store value


v_accvgpr_read v6, a12 // (op 1893) 
v_accvgpr_read v7, a13 // (op 1893) 
v_accvgpr_read v8, a14 // (op 1893) 
v_accvgpr_read v9, a15 // (op 1893) 
v_mul_f32 v14, v11, v6 // (op 1893) 
v_mul_f32 v15, v11, v7 // (op 1893) 
v_mul_f32 v16, v11, v8 // (op 1893) 
v_mul_f32 v17, v11, v9 // (op 1893) 



s_lshl_b32 s0, s4, 7 // (op 1895) 
s_mul_i32 s1, 3, s0 // (op 1895) 

v_add_u32 v0, v13, s1 // (op 1895) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1895) Store value


v_accvgpr_read v6, a16 // (op 1927) 
v_accvgpr_read v7, a17 // (op 1927) 
v_accvgpr_read v8, a18 // (op 1927) 
v_accvgpr_read v9, a19 // (op 1927) 
v_mul_f32 v14, v11, v6 // (op 1927) 
v_mul_f32 v15, v11, v7 // (op 1927) 
v_mul_f32 v16, v11, v8 // (op 1927) 
v_mul_f32 v17, v11, v9 // (op 1927) 



v_add_u32 v0, 128, v13 // (op 1930) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1930) Store value


v_accvgpr_read v6, a20 // (op 1940) 
v_accvgpr_read v7, a21 // (op 1940) 
v_accvgpr_read v8, a22 // (op 1940) 
v_accvgpr_read v9, a23 // (op 1940) 
v_mul_f32 v14, v11, v6 // (op 1940) 
v_mul_f32 v15, v11, v7 // (op 1940) 
v_mul_f32 v16, v11, v8 // (op 1940) 
v_mul_f32 v17, v11, v9 // (op 1940) 



s_lshl_b32 s0, s4, 7 // (op 1932) 
s_add_u32 s0, s0, 128 // (op 1932) 
v_add_u32 v0, v13, s0 // (op 1932) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1932) Store value


v_accvgpr_read v6, a24 // (op 1945) 
v_accvgpr_read v7, a25 // (op 1945) 
v_accvgpr_read v8, a26 // (op 1945) 
v_accvgpr_read v9, a27 // (op 1945) 
v_mul_f32 v14, v11, v6 // (op 1945) 
v_mul_f32 v15, v11, v7 // (op 1945) 
v_mul_f32 v16, v11, v8 // (op 1945) 
v_mul_f32 v17, v11, v9 // (op 1945) 



s_lshl_b32 s0, s4, 8 // (op 1934) 
s_add_u32 s0, s0, 128 // (op 1934) 
v_add_u32 v0, v13, s0 // (op 1934) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1934) Store value


v_accvgpr_read v6, a28 // (op 1950) 
v_accvgpr_read v7, a29 // (op 1950) 
v_accvgpr_read v8, a30 // (op 1950) 
v_accvgpr_read v9, a31 // (op 1950) 
v_mul_f32 v14, v11, v6 // (op 1950) 
v_mul_f32 v15, v11, v7 // (op 1950) 
v_mul_f32 v16, v11, v8 // (op 1950) 
v_mul_f32 v17, v11, v9 // (op 1950) 



s_lshl_b32 s0, s4, 7 // (op 1936) 
s_mul_i32 s1, 3, s0 // (op 1936) 
s_add_u32 s1, s1, 128 // (op 1936) 

v_add_u32 v0, v13, s1 // (op 1936) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1936) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v12 // (op 2108) 

buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2108) Load value

v_accvgpr_read v14, a16 // (op 2096) 
v_accvgpr_read v15, a17 // (op 2096) 
v_accvgpr_read v16, a18 // (op 2096) 
v_accvgpr_read v17, a19 // (op 2096) 
v_mul_f32 v18, v11, v14 // (op 2096) 
v_mul_f32 v19, v11, v15 // (op 2096) 
v_mul_f32 v20, v11, v16 // (op 2096) 
v_mul_f32 v21, v11, v17 // (op 2096) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2107) 
v_mul_f32 v15, v10, v7 // (op 2107) 
v_mul_f32 v16, v10, v8 // (op 2107) 
v_mul_f32 v17, v10, v9 // (op 2107) 

v_add_f32 v6, v14, v18 // (op 2097) 
v_add_f32 v7, v15, v19 // (op 2097) 
v_add_f32 v8, v16, v20 // (op 2097) 
v_add_f32 v9, v17, v21 // (op 2097) 




v_add_u32 v0, 128, v13 // (op 2099) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2099) Store value




s_lshl_b32 s0, s2, 7 // (op 2112) 
s_add_u32 s0, s0, 128 // (op 2112) 
v_add_u32 v0, v12, s0 // (op 2112) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2112) Load value

v_accvgpr_read v14, a20 // (op 2122) 
v_accvgpr_read v15, a21 // (op 2122) 
v_accvgpr_read v16, a22 // (op 2122) 
v_accvgpr_read v17, a23 // (op 2122) 
v_mul_f32 v18, v11, v14 // (op 2122) 
v_mul_f32 v19, v11, v15 // (op 2122) 
v_mul_f32 v20, v11, v16 // (op 2122) 
v_mul_f32 v21, v11, v17 // (op 2122) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2110) 
v_mul_f32 v15, v10, v7 // (op 2110) 
v_mul_f32 v16, v10, v8 // (op 2110) 
v_mul_f32 v17, v10, v9 // (op 2110) 

v_add_f32 v6, v14, v18 // (op 2111) 
v_add_f32 v7, v15, v19 // (op 2111) 
v_add_f32 v8, v16, v20 // (op 2111) 
v_add_f32 v9, v17, v21 // (op 2111) 




s_lshl_b32 s0, s4, 7 // (op 2101) 
s_add_u32 s0, s0, 128 // (op 2101) 
v_add_u32 v0, v13, s0 // (op 2101) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2101) Store value




s_lshl_b32 s0, s2, 8 // (op 2116) 
s_add_u32 s0, s0, 128 // (op 2116) 
v_add_u32 v0, v12, s0 // (op 2116) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2116) Load value

v_accvgpr_read v14, a24 // (op 2124) 
v_accvgpr_read v15, a25 // (op 2124) 
v_accvgpr_read v16, a26 // (op 2124) 
v_accvgpr_read v17, a27 // (op 2124) 
v_mul_f32 v18, v11, v14 // (op 2124) 
v_mul_f32 v19, v11, v15 // (op 2124) 
v_mul_f32 v20, v11, v16 // (op 2124) 
v_mul_f32 v21, v11, v17 // (op 2124) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2114) 
v_mul_f32 v15, v10, v7 // (op 2114) 
v_mul_f32 v16, v10, v8 // (op 2114) 
v_mul_f32 v17, v10, v9 // (op 2114) 

v_add_f32 v6, v14, v18 // (op 2115) 
v_add_f32 v7, v15, v19 // (op 2115) 
v_add_f32 v8, v16, v20 // (op 2115) 
v_add_f32 v9, v17, v21 // (op 2115) 




s_lshl_b32 s0, s4, 8 // (op 2103) 
s_add_u32 s0, s0, 128 // (op 2103) 
v_add_u32 v0, v13, s0 // (op 2103) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2103) Store value




s_lshl_b32 s0, s2, 7 // (op 2120) 
s_mul_i32 s1, 3, s0 // (op 2120) 
s_add_u32 s1, s1, 128 // (op 2120) 

v_add_u32 v0, v12, s1 // (op 2120) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2120) Load value

v_accvgpr_read v14, a28 // (op 2126) 
v_accvgpr_read v15, a29 // (op 2126) 
v_accvgpr_read v16, a30 // (op 2126) 
v_accvgpr_read v17, a31 // (op 2126) 
v_mul_f32 v18, v11, v14 // (op 2126) 
v_mul_f32 v19, v11, v15 // (op 2126) 
v_mul_f32 v20, v11, v16 // (op 2126) 
v_mul_f32 v21, v11, v17 // (op 2126) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2118) 
v_mul_f32 v15, v10, v7 // (op 2118) 
v_mul_f32 v16, v10, v8 // (op 2118) 
v_mul_f32 v17, v10, v9 // (op 2118) 

v_add_f32 v6, v14, v18 // (op 2119) 
v_add_f32 v7, v15, v19 // (op 2119) 
v_add_f32 v8, v16, v20 // (op 2119) 
v_add_f32 v9, v17, v21 // (op 2119) 




s_lshl_b32 s0, s4, 7 // (op 2105) 
s_mul_i32 s1, 3, s0 // (op 2105) 
s_add_u32 s1, s1, 128 // (op 2105) 

v_add_u32 v0, v13, s1 // (op 2105) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2105) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 145) Load value

v_accvgpr_read v14, a0 // (op 142) 
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 146) 
v_mul_f32 v15, v10, v7 // (op 146) 
v_mul_f32 v16, v10, v8 // (op 146) 
v_mul_f32 v17, v10, v9 // (op 146) 

v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 
v_add_f32 v8, v16, v20 // (op 143) 
v_add_f32 v9, v17, v21 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 2022) 
v_add_u32 v0, v12, s0 // (op 2022) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2022) Load value

v_accvgpr_read v14, a4 // (op 2019) 
v_accvgpr_read v15, a5 // (op 2019) 
v_accvgpr_read v16, a6 // (op 2019) 
v_accvgpr_read v17, a7 // (op 2019) 
v_mul_f32 v18, v11, v14 // (op 2019) 
v_mul_f32 v19, v11, v15 // (op 2019) 
v_mul_f32 v20, v11, v16 // (op 2019) 
v_mul_f32 v21, v11, v17 // (op 2019) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2023) 
v_mul_f32 v15, v10, v7 // (op 2023) 
v_mul_f32 v16, v10, v8 // (op 2023) 
v_mul_f32 v17, v10, v9 // (op 2023) 

v_add_f32 v6, v14, v18 // (op 2020) 
v_add_f32 v7, v15, v19 // (op 2020) 
v_add_f32 v8, v16, v20 // (op 2020) 
v_add_f32 v9, v17, v21 // (op 2020) 




s_lshl_b32 s0, s4, 7 // (op 2021) 
v_add_u32 v0, v13, s0 // (op 2021) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2021) Store value




s_lshl_b32 s0, s2, 8 // (op 2034) 
v_add_u32 v0, v12, s0 // (op 2034) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2034) Load value

v_accvgpr_read v14, a8 // (op 2031) 
v_accvgpr_read v15, a9 // (op 2031) 
v_accvgpr_read v16, a10 // (op 2031) 
v_accvgpr_read v17, a11 // (op 2031) 
v_mul_f32 v18, v11, v14 // (op 2031) 
v_mul_f32 v19, v11, v15 // (op 2031) 
v_mul_f32 v20, v11, v16 // (op 2031) 
v_mul_f32 v21, v11, v17 // (op 2031) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2035) 
v_mul_f32 v15, v10, v7 // (op 2035) 
v_mul_f32 v16, v10, v8 // (op 2035) 
v_mul_f32 v17, v10, v9 // (op 2035) 

v_add_f32 v6, v14, v18 // (op 2032) 
v_add_f32 v7, v15, v19 // (op 2032) 
v_add_f32 v8, v16, v20 // (op 2032) 
v_add_f32 v9, v17, v21 // (op 2032) 




s_lshl_b32 s0, s4, 8 // (op 2033) 
v_add_u32 v0, v13, s0 // (op 2033) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2033) Store value




s_lshl_b32 s0, s2, 7 // (op 2046) 
s_mul_i32 s1, 3, s0 // (op 2046) 

v_add_u32 v0, v12, s1 // (op 2046) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 2046) Load value

v_accvgpr_read v14, a12 // (op 2043) 
v_accvgpr_read v15, a13 // (op 2043) 
v_accvgpr_read v16, a14 // (op 2043) 
v_accvgpr_read v17, a15 // (op 2043) 
v_mul_f32 v18, v11, v14 // (op 2043) 
v_mul_f32 v19, v11, v15 // (op 2043) 
v_mul_f32 v20, v11, v16 // (op 2043) 
v_mul_f32 v21, v11, v17 // (op 2043) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2047) 
v_mul_f32 v15, v10, v7 // (op 2047) 
v_mul_f32 v16, v10, v8 // (op 2047) 
v_mul_f32 v17, v10, v9 // (op 2047) 

v_add_f32 v6, v14, v18 // (op 2044) 
v_add_f32 v7, v15, v19 // (op 2044) 
v_add_f32 v8, v16, v20 // (op 2044) 
v_add_f32 v9, v17, v21 // (op 2044) 




s_lshl_b32 s0, s4, 7 // (op 2045) 
s_mul_i32 s1, 3, s0 // (op 2045) 

v_add_u32 v0, v13, s1 // (op 2045) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 2045) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
.LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_end:
.size RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC, .LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC_end-RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
  .amdhsa_next_free_vgpr 140
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 34304 // lds bytes
  .amdhsa_accum_offset 108
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
    DataTypeB: F4
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
  - .name: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC
    .symbol: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_LDSSYNC.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 34304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 106
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

