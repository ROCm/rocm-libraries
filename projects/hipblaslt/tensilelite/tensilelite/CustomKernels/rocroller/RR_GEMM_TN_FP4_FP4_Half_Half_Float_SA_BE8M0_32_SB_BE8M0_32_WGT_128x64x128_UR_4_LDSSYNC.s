// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC,@function
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 7010) 
v_and_b32 v3, 15, v1 // (op 7010) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 7010) 


v_and_b32 v0, 63, v1 // (op 7010) 
v_lshrrev_b32 v3, 4, v0 // (op 7010) 

v_lshlrev_b32 v0, 5, v3 // (op 7010) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 7010) 


v_lshlrev_b32 v0, 2, v3 // (op 7010) 

v_lshrrev_b32 v3, 3, v0 // (op 7010) 

v_lshrrev_b32 v0, 6, v1 // (op 7068) 
v_and_b32 v4, 1, v0 // (op 7068) 

v_and_b32 v0, 15, v1 // (op 7068) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 7068) 


v_and_b32 v0, 63, v1 // (op 7068) 
v_lshrrev_b32 v4, 4, v0 // (op 7068) 

v_lshlrev_b32 v0, 5, v4 // (op 7068) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 7068) 


v_lshlrev_b32 v0, 2, v4 // (op 7068) 

v_lshrrev_b32 v4, 3, v0 // (op 7068) 

v_lshrrev_b32 v0, 7, v1 // (op 7126) 
v_and_b32 v5, 15, v1 // (op 7126) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 7126) 


v_and_b32 v0, 63, v1 // (op 7126) 
v_lshrrev_b32 v5, 4, v0 // (op 7126) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 7126) 


v_lshrrev_b32 v5, 6, v1 // (op 7184) 
v_and_b32 v6, 1, v5 // (op 7184) 

v_and_b32 v5, 15, v1 // (op 7184) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 7184) 


v_and_b32 v5, 63, v1 // (op 7184) 
v_lshrrev_b32 v6, 4, v5 // (op 7184) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 7184) 


v_lshrrev_b32 v6, 7, v1 // (op 7236) 
v_and_b32 v7, 15, v1 // (op 7236) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 7236) 


v_and_b32 v6, 63, v1 // (op 7236) 
v_lshrrev_b32 v7, 4, v6 // (op 7236) 

v_lshlrev_b32 v6, 5, v7 // (op 7236) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 7236) 


v_lshlrev_b32 v6, 2, v7 // (op 7236) 

v_lshrrev_b32 v7, 3, v6 // (op 7236) 

v_lshrrev_b32 v6, 6, v1 // (op 7265) 
v_and_b32 v8, 1, v6 // (op 7265) 

v_and_b32 v6, 15, v1 // (op 7265) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 7265) 


v_and_b32 v6, 63, v1 // (op 7265) 
v_lshrrev_b32 v8, 4, v6 // (op 7265) 

v_lshlrev_b32 v6, 5, v8 // (op 7265) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 7265) 


v_lshlrev_b32 v6, 2, v8 // (op 7265) 

v_lshrrev_b32 v8, 3, v6 // (op 7265) 

v_lshrrev_b32 v6, 7, v1 // (op 7294) 
v_and_b32 v9, 15, v1 // (op 7294) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 7294) 


v_and_b32 v6, 63, v1 // (op 7294) 
v_lshrrev_b32 v9, 4, v6 // (op 7294) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 7294) 


v_lshrrev_b32 v9, 6, v1 // (op 7323) 
v_and_b32 v10, 1, v9 // (op 7323) 

v_and_b32 v9, 15, v1 // (op 7323) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 7323) 


v_and_b32 v9, 63, v1 // (op 7323) 
v_lshrrev_b32 v10, 4, v9 // (op 7323) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 7323) 


v_lshrrev_b32 v10, 7, v1 // (op 7352) 
v_and_b32 v11, 15, v1 // (op 7352) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 7352) 


v_and_b32 v10, 63, v1 // (op 7352) 
v_lshrrev_b32 v11, 4, v10 // (op 7352) 

v_lshlrev_b32 v10, 5, v11 // (op 7352) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 7352) 


v_lshlrev_b32 v10, 2, v11 // (op 7352) 

v_lshrrev_b32 v11, 3, v10 // (op 7352) 

v_lshrrev_b32 v10, 6, v1 // (op 7381) 
v_and_b32 v12, 1, v10 // (op 7381) 

v_and_b32 v10, 15, v1 // (op 7381) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 7381) 


v_and_b32 v10, 63, v1 // (op 7381) 
v_lshrrev_b32 v12, 4, v10 // (op 7381) 

v_lshlrev_b32 v10, 5, v12 // (op 7381) 

v_lshl_add_u32 v12, v13, 7, v10 // (op 7381) 


v_lshlrev_b32 v10, 2, v12 // (op 7381) 

v_lshrrev_b32 v12, 3, v10 // (op 7381) 

v_lshrrev_b32 v10, 7, v1 // (op 7410) 
v_and_b32 v13, 15, v1 // (op 7410) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 7410) 


v_and_b32 v10, 63, v1 // (op 7410) 
v_lshrrev_b32 v13, 4, v10 // (op 7410) 

v_lshl_add_u32 v10, v14, 2, v13 // (op 7410) 


v_lshrrev_b32 v13, 6, v1 // (op 7439) 
v_and_b32 v14, 1, v13 // (op 7439) 

v_and_b32 v13, 15, v1 // (op 7439) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 7439) 


v_and_b32 v13, 63, v1 // (op 7439) 
v_lshrrev_b32 v14, 4, v13 // (op 7439) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 7439) 


v_lshrrev_b32 v14, 7, v1 // (op 7468) 
v_and_b32 v15, 15, v1 // (op 7468) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 7468) 


v_and_b32 v14, 63, v1 // (op 7468) 
v_lshrrev_b32 v15, 4, v14 // (op 7468) 

v_lshlrev_b32 v14, 5, v15 // (op 7468) 

v_lshl_add_u32 v15, v16, 7, v14 // (op 7468) 


v_lshlrev_b32 v14, 2, v15 // (op 7468) 

v_lshrrev_b32 v15, 3, v14 // (op 7468) 

v_lshrrev_b32 v14, 6, v1 // (op 7497) 
v_and_b32 v16, 1, v14 // (op 7497) 

v_and_b32 v14, 15, v1 // (op 7497) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 7497) 


v_and_b32 v14, 63, v1 // (op 7497) 
v_lshrrev_b32 v16, 4, v14 // (op 7497) 

v_lshlrev_b32 v14, 5, v16 // (op 7497) 

v_lshl_add_u32 v16, v17, 7, v14 // (op 7497) 


v_lshlrev_b32 v14, 2, v16 // (op 7497) 

v_lshrrev_b32 v16, 3, v14 // (op 7497) 

v_lshrrev_b32 v14, 7, v1 // (op 7526) 
v_and_b32 v17, 15, v1 // (op 7526) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 7526) 


v_and_b32 v14, 63, v1 // (op 7526) 
v_lshrrev_b32 v17, 4, v14 // (op 7526) 

v_lshl_add_u32 v14, v18, 2, v17 // (op 7526) 


v_lshrrev_b32 v17, 6, v1 // (op 7555) 
v_and_b32 v18, 1, v17 // (op 7555) 

v_and_b32 v17, 15, v1 // (op 7555) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 7555) 


v_and_b32 v17, 63, v1 // (op 7555) 
v_lshrrev_b32 v18, 4, v17 // (op 7555) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 7555) 


v_lshrrev_b32 v18, 7, v1 // (op 6948) 
v_and_b32 v19, 63, v1 // (op 6948) 
v_lshrrev_b32 v20, 4, v19 // (op 6948) 
v_lshlrev_b32 v21, 2, v20 // (op 6948) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 6948) 


v_lshl_add_u32 v18, s16, 7, v20 // (op 6948) 

v_lshrrev_b32 v20, 2, v19 // (op 6948) 

v_and_b32 v19, 3, v20 // (op 6948) 

v_and_b32 v20, 3, v1 // (op 6948) 
v_lshl_add_u32 v21, v19, 2, v20 // (op 6948) 


v_lshrrev_b32 v19, 6, v1 // (op 6948) 
v_and_b32 v20, 1, v19 // (op 6948) 

v_lshl_add_u32 v19, v20, 4, v21 // (op 6948) 


v_lshl_add_u32 v20, s17, 6, v19 // (op 6948) 

v_mul_lo_u32 v19, v20, s2 // (op 6948) 

v_add_lshl_u32 v20, v18, v19, 1 // (op 6948) 


v_lshrrev_b32 v18, 7, v1 // (op 7205) 
v_and_b32 v19, 63, v1 // (op 7205) 
v_lshrrev_b32 v21, 4, v19 // (op 7205) 
s_mul_i32 s1, s2, s9 // (op 6950) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 6950) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 6950) least significant
s_mul_hi_u32 s19, s2, s8 // (op 6950) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6950) most significant: sum
s_add_u32 s1, s1, s19 // (op 6950) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 6950) 

v_lshlrev_b32 v22, 2, v21 // (op 7205) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 7205) 


v_lshl_add_u32 v18, s16, 7, v21 // (op 7205) 

v_lshrrev_b32 v21, 2, v19 // (op 7205) 

v_and_b32 v19, 3, v21 // (op 7205) 

v_and_b32 v21, 3, v1 // (op 7205) 
v_lshl_add_u32 v22, v19, 2, v21 // (op 7205) 


v_lshrrev_b32 v19, 6, v1 // (op 7205) 
v_and_b32 v21, 1, v19 // (op 7205) 

v_lshl_add_u32 v19, v21, 4, v22 // (op 7205) 


v_lshl_add_u32 v21, s17, 6, v19 // (op 7205) 

v_mul_lo_u32 v19, v21, s4 // (op 7205) 

v_add_lshl_u32 v21, v18, v19, 1 // (op 7205) 


s_mov_b32 s40, s6 // (op 6950) reg expression
s_mov_b32 s41, s7 // (op 6950) reg expression
s_mov_b32 s42, s18 // (op 6950) reg expression
s_mov_b32 s43, 131072 // (op 6950) call()

s_mul_i32 s1, s4, s9 // (op 7207) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 7207) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 7207) least significant
s_mul_hi_u32 s7, s4, s8 // (op 7207) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7207) most significant: sum
s_add_u32 s1, s1, s7 // (op 7207) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 7207) 

s_mov_b32 s44, s10 // (op 7207) reg expression
s_mov_b32 s45, s11 // (op 7207) reg expression
s_mov_b32 s46, s6 // (op 7207) reg expression
s_mov_b32 s47, 131072 // (op 7207) call()


v_mov_b32 v18, s14 // (op 2) Move value
v_mov_b32 v19, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 486) call()
v_accvgpr_write a5, 0.00000 // (op 486) call()
v_accvgpr_write a6, 0.00000 // (op 486) call()
v_accvgpr_write a7, 0.00000 // (op 486) call()
v_accvgpr_write a8, 0.00000 // (op 852) call()
v_accvgpr_write a9, 0.00000 // (op 852) call()
v_accvgpr_write a10, 0.00000 // (op 852) call()
v_accvgpr_write a11, 0.00000 // (op 852) call()
v_accvgpr_write a12, 0.00000 // (op 1099) call()
v_accvgpr_write a13, 0.00000 // (op 1099) call()
v_accvgpr_write a14, 0.00000 // (op 1099) call()
v_accvgpr_write a15, 0.00000 // (op 1099) call()
v_accvgpr_write a16, 0.00000 // (op 1498) call()
v_accvgpr_write a17, 0.00000 // (op 1498) call()
v_accvgpr_write a18, 0.00000 // (op 1498) call()
v_accvgpr_write a19, 0.00000 // (op 1498) call()
v_accvgpr_write a20, 0.00000 // (op 1745) call()
v_accvgpr_write a21, 0.00000 // (op 1745) call()
v_accvgpr_write a22, 0.00000 // (op 1745) call()
v_accvgpr_write a23, 0.00000 // (op 1745) call()
v_accvgpr_write a24, 0.00000 // (op 2144) call()
v_accvgpr_write a25, 0.00000 // (op 2144) call()
v_accvgpr_write a26, 0.00000 // (op 2144) call()
v_accvgpr_write a27, 0.00000 // (op 2144) call()
v_accvgpr_write a28, 0.00000 // (op 2391) call()
v_accvgpr_write a29, 0.00000 // (op 2391) call()
v_accvgpr_write a30, 0.00000 // (op 2391) call()
v_accvgpr_write a31, 0.00000 // (op 2391) call()
v_lshrrev_b32 v22, 2, v1 // (op 6710) 
v_lshlrev_b32 v23, 1, v22 // (op 6710) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 6710) 

v_and_b32 v23, -4, v1 // (op 6710) 
v_sub_u32 v24, v1, v23 // (op 6710) 

v_lshlrev_b32 v23, 5, v24 // (op 6710) 

v_mov_b32 v25, 0 // (op 6710) convert
v_mov_b32 v24, v23 // (op 6710) convert

v_mul_lo_u32 v27, v22, s13 // (op 6710) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 6710) most significant: high of low * low
v_mul_lo_u32 v26, v22, s12 // (op 6710) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 6710) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v24 // (op 6710) least significant half
v_addc_co_u32 v27, s[0:1], v27, v25, s[0:1] // (op 6710) most significant half



v_lshlrev_b64 v[22:23], 2, v[26:27] // (op 6710) 

v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 6710) 

v_mov_b32 v22, v24 // (op 6710) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 6712) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6712) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6712) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6712) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6712) most significant: sum
s_add_u32 s1, s1, s7 // (op 6712) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6712) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6712) 

s_mov_b32 s48, s20 // (op 6712) reg expression
s_mov_b32 s49, s21 // (op 6712) reg expression
s_mov_b32 s50, s0 // (op 6712) reg expression
s_mov_b32 s51, 131072 // (op 6712) call()

v_lshrrev_b32 v23, 2, v1 // (op 6729) 
v_lshlrev_b32 v24, 1, v23 // (op 6729) 

v_lshl_add_u32 v23, s16, 7, v24 // (op 6729) 

v_and_b32 v24, -4, v1 // (op 6729) 
v_sub_u32 v25, v1, v24 // (op 6729) 

v_lshlrev_b32 v24, 5, v25 // (op 6729) 

v_mov_b32 v27, 0 // (op 6729) convert
v_mov_b32 v26, v24 // (op 6729) convert

v_mul_lo_u32 v25, v23, s13 // (op 6729) most significant: low of low * high
v_mul_hi_u32 v28, v23, s12 // (op 6729) most significant: high of low * low
v_mul_lo_u32 v24, v23, s12 // (op 6729) least significant: low of low * low
v_add_u32 v25, v25, v28 // (op 6729) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v26 // (op 6729) least significant half
v_addc_co_u32 v25, s[0:1], v25, v27, s[0:1] // (op 6729) most significant half



v_lshlrev_b64 v[26:27], 2, v[24:25] // (op 6729) 

v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 6729) 

v_mov_b32 v23, v24 // (op 6729) convert

s_mul_i32 s1, s12, s23 // (op 6731) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6731) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6731) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6731) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6731) most significant: sum
s_add_u32 s1, s1, s7 // (op 6731) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6731) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6731) 

s_mov_b32 s48, s20 // (op 6731) reg expression
s_mov_b32 s49, s21 // (op 6731) reg expression
s_mov_b32 s50, s0 // (op 6731) reg expression
s_mov_b32 s51, 131072 // (op 6731) call()

v_and_b32 v24, -4, v1 // (op 6774) 
v_sub_u32 v25, v1, v24 // (op 6774) 

v_lshlrev_b32 v24, 5, v25 // (op 6774) 

v_mov_b32 v27, 0 // (op 6774) convert
v_mov_b32 v26, v24 // (op 6774) convert

v_lshrrev_b32 v24, 2, v1 // (op 6774) 
v_lshl_add_u32 v25, s17, 6, v24 // (op 6774) 

v_mul_lo_u32 v29, v25, s25 // (op 6774) most significant: low of low * high
v_mul_hi_u32 v24, v25, s24 // (op 6774) most significant: high of low * low
v_mul_lo_u32 v28, v25, s24 // (op 6774) least significant: low of low * low
v_add_u32 v29, v29, v24 // (op 6774) most significant: sum


v_add_co_u32 v24, s[0:1], v26, v28 // (op 6774) least significant half
v_addc_co_u32 v25, s[0:1], v27, v29, s[0:1] // (op 6774) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 6774) 


v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 6774) 

v_mov_b32 v24, v26 // (op 6774) convert

s_mul_i32 s1, s24, s9 // (op 6776) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6776) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6776) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6776) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6776) most significant: sum
s_add_u32 s1, s1, s7 // (op 6776) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6776) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6776) 

s_mov_b32 s52, s26 // (op 6776) reg expression
s_mov_b32 s53, s27 // (op 6776) reg expression
s_mov_b32 s54, s0 // (op 6776) reg expression
s_mov_b32 s55, 131072 // (op 6776) call()

v_and_b32 v25, -4, v1 // (op 6793) 
v_sub_u32 v26, v1, v25 // (op 6793) 

v_lshlrev_b32 v25, 5, v26 // (op 6793) 

v_mov_b32 v27, 0 // (op 6793) convert
v_mov_b32 v26, v25 // (op 6793) convert

v_lshrrev_b32 v25, 2, v1 // (op 6793) 
v_lshl_add_u32 v28, s17, 6, v25 // (op 6793) 

v_mul_lo_u32 v31, v28, s25 // (op 6793) most significant: low of low * high
v_mul_hi_u32 v25, v28, s24 // (op 6793) most significant: high of low * low
v_mul_lo_u32 v30, v28, s24 // (op 6793) least significant: low of low * low
v_add_u32 v31, v31, v25 // (op 6793) most significant: sum


v_add_co_u32 v28, s[0:1], v26, v30 // (op 6793) least significant half
v_addc_co_u32 v29, s[0:1], v27, v31, s[0:1] // (op 6793) most significant half

v_lshlrev_b64 v[28:29], 2, v[28:29] // (op 6793) 


v_lshrrev_b64 v[26:27], 3, v[28:29] // (op 6793) 

v_mov_b32 v25, v26 // (op 6793) convert

s_mul_i32 s1, s24, s9 // (op 6795) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6795) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6795) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6795) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6795) most significant: sum
s_add_u32 s1, s1, s7 // (op 6795) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6795) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6795) 

s_mov_b32 s52, s26 // (op 6795) reg expression
s_mov_b32 s53, s27 // (op 6795) reg expression
s_mov_b32 s54, s0 // (op 6795) reg expression
s_mov_b32 s55, 131072 // (op 6795) call()

v_lshrrev_b32 v26, 2, v1 // (op 6838) 
v_lshlrev_b32 v27, 1, v26 // (op 6838) 

v_lshl_add_u32 v26, s16, 7, v27 // (op 6838) 

v_and_b32 v27, -4, v1 // (op 6838) 
v_sub_u32 v28, v1, v27 // (op 6838) 

v_mul_lo_u32 v27, v26, s28 // (op 6838) 
v_add_u32 v27, v27, v28 // (op 6838) 


s_mul_i32 s1, s28, s23 // (op 6840) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6840) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6840) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6840) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6840) most significant: sum
s_add_u32 s1, s1, s7 // (op 6840) most significant: sum


s_mov_b32 s56, s30 // (op 6840) reg expression
s_mov_b32 s57, s31 // (op 6840) reg expression
s_mov_b32 s58, s0 // (op 6840) reg expression
s_mov_b32 s59, 131072 // (op 6840) call()

v_lshrrev_b32 v26, 2, v1 // (op 6857) 
v_lshlrev_b32 v28, 1, v26 // (op 6857) 

v_lshl_add_u32 v26, s16, 7, v28 // (op 6857) 

v_and_b32 v28, -4, v1 // (op 6857) 
v_sub_u32 v29, v1, v28 // (op 6857) 

v_mul_lo_u32 v28, v26, s28 // (op 6857) 
v_add_u32 v28, v28, v29 // (op 6857) 


s_mul_i32 s1, s28, s23 // (op 6859) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6859) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6859) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6859) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6859) most significant: sum
s_add_u32 s1, s1, s7 // (op 6859) most significant: sum


s_mov_b32 s56, s30 // (op 6859) reg expression
s_mov_b32 s57, s31 // (op 6859) reg expression
s_mov_b32 s58, s0 // (op 6859) reg expression
s_mov_b32 s59, 131072 // (op 6859) call()

v_and_b32 v26, -4, v1 // (op 6902) 
v_sub_u32 v29, v1, v26 // (op 6902) 

v_lshrrev_b32 v26, 2, v1 // (op 6902) 
v_lshl_add_u32 v30, s17, 6, v26 // (op 6902) 

v_mul_lo_u32 v26, v30, s32 // (op 6902) 

v_add_u32 v30, v29, v26 // (op 6902) 


s_mul_i32 s1, s32, s9 // (op 6904) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6904) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6904) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6904) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6904) most significant: sum
s_add_u32 s1, s1, s7 // (op 6904) most significant: sum


s_mov_b32 s60, s34 // (op 6904) reg expression
s_mov_b32 s61, s35 // (op 6904) reg expression
s_mov_b32 s62, s0 // (op 6904) reg expression
s_mov_b32 s63, 131072 // (op 6904) call()

v_and_b32 v26, -4, v1 // (op 6921) 
v_sub_u32 v29, v1, v26 // (op 6921) 

v_lshrrev_b32 v26, 2, v1 // (op 6921) 
v_lshl_add_u32 v31, s17, 6, v26 // (op 6921) 

v_mul_lo_u32 v26, v31, s32 // (op 6921) 

v_add_u32 v31, v29, v26 // (op 6921) 


s_mul_i32 s1, s32, s9 // (op 6923) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6923) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6923) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6923) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6923) most significant: sum
s_add_u32 s1, s1, s7 // (op 6923) most significant: sum


s_mov_b32 s60, s34 // (op 6923) reg expression
s_mov_b32 s61, s35 // (op 6923) reg expression
s_mov_b32 s62, s0 // (op 6923) reg expression
s_mov_b32 s63, 131072 // (op 6923) call()




buffer_load_dwordx4 v[32:35], v24, s[52:55], 0 offen // (op 6250) Load value



buffer_load_ubyte v26, v30, s[60:63], 0 offen offset: 0 // (op 6253) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 6256) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 6256) 


v_mov_b32 v29, v22 // (op 6256) 
buffer_load_dwordx4 v[38:41], v29, s[48:51], 0 offen // (op 6256) Load value
v_add_u32 v29, v29, v36 // (op 6256) 
buffer_load_dwordx4 v[42:45], v29, s[48:51], 0 offen // (op 6256) Load value



v_mov_b64 v[36:37], s[28:29] // (op 6259) convert

v_mov_b32 v29, v27 // (op 6259) 
buffer_load_ubyte v46, v29, s[56:59], 0 offen offset: 0 // (op 6259) Load value
v_add_u32 v29, v29, v36 // (op 6259) 
buffer_load_ubyte v47, v29, s[56:59], 0 offen offset: 0 // (op 6259) Load value





buffer_load_dwordx4 v[48:51], v24, s[52:55], 0 offen offset:64 // (op 6262) Load value



buffer_load_ubyte v29, v30, s[60:63], 0 offen offset:4 // (op 6265) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 6268) 
v_lshrrev_b64 v[36:37], 3, s[0:1] // (op 6268) 


v_mov_b32 v52, v22 // (op 6268) 
buffer_load_dwordx4 v[54:57], v52, s[48:51], 0 offen offset:64 // (op 6268) Load value
v_add_u32 v52, v52, v36 // (op 6268) 
buffer_load_dwordx4 v[58:61], v52, s[48:51], 0 offen offset:64 // (op 6268) Load value



v_mov_b64 v[36:37], s[28:29] // (op 6271) convert

v_mov_b32 v52, v27 // (op 6271) 
buffer_load_ubyte v62, v52, s[56:59], 0 offen offset:4 // (op 6271) Load value
v_add_u32 v52, v52, v36 // (op 6271) 
buffer_load_ubyte v63, v52, s[56:59], 0 offen offset:4 // (op 6271) Load value





buffer_load_dwordx4 v[64:67], v24, s[52:55], 0 offen offset:128 // (op 6274) Load value



buffer_load_ubyte v36, v30, s[60:63], 0 offen offset:8 // (op 6277) Load value

s_lshl_b64 s[0:1], s[12:13], 2 // (op 6280) 
v_lshrrev_b64 v[52:53], 3, s[0:1] // (op 6280) 


v_mov_b32 v37, v22 // (op 6280) 
buffer_load_dwordx4 v[68:71], v37, s[48:51], 0 offen offset:128 // (op 6280) Load value
v_add_u32 v37, v37, v52 // (op 6280) 
buffer_load_dwordx4 v[72:75], v37, s[48:51], 0 offen offset:128 // (op 6280) Load value



v_mov_b64 v[52:53], s[28:29] // (op 6283) convert

v_mov_b32 v37, v27 // (op 6283) 
buffer_load_ubyte v76, v37, s[56:59], 0 offen offset:8 // (op 6283) Load value
v_add_u32 v37, v37, v52 // (op 6283) 
buffer_load_ubyte v77, v37, s[56:59], 0 offen offset:8 // (op 6283) Load value





buffer_load_dwordx4 v[78:81], v24, s[52:55], 0 offen offset:192 // (op 6286) Load value




buffer_load_ubyte v24, v30, s[60:63], 0 offen offset:12 // (op 6289) Load value


s_lshl_b64 s[0:1], s[12:13], 2 // (op 6292) 
v_lshrrev_b64 v[52:53], 3, s[0:1] // (op 6292) 


v_mov_b32 v30, v22 // (op 6292) 
buffer_load_dwordx4 v[82:85], v30, s[48:51], 0 offen offset:192 // (op 6292) Load value
v_add_u32 v30, v30, v52 // (op 6292) 
buffer_load_dwordx4 v[86:89], v30, s[48:51], 0 offen offset:192 // (op 6292) Load value




v_mov_b64 v[52:53], s[28:29] // (op 6295) convert

v_mov_b32 v22, v27 // (op 6295) 
buffer_load_ubyte v90, v22, s[56:59], 0 offen offset:12 // (op 6295) Load value
v_add_u32 v22, v22, v52 // (op 6295) 
buffer_load_ubyte v91, v22, s[56:59], 0 offen offset:12 // (op 6295) Load value



v_and_b32 v22, -4, v1 // (op 7051) 
v_sub_u32 v27, v1, v22 // (op 7051) 

v_lshlrev_b32 v22, 5, v27 // (op 7051) 

v_lshrrev_b32 v27, 2, v1 // (op 7051) 
v_lshl_add_u32 v30, v27, 7, v22 // (op 7051) 


v_lshlrev_b32 v22, 2, v30 // (op 7051) 

v_lshrrev_b32 v27, 3, v22 // (op 7051) 




s_waitcnt vmcnt(23)
ds_write_b128 v27, v[32:35] // (op 6297) Store local data 

v_and_b32 v22, -4, v1 // (op 7167) 
v_sub_u32 v27, v1, v22 // (op 7167) 

v_and_b32 v22, 4294967292, v1 // (op 7167) 
v_add_u32 v30, v22, v27 // (op 7167) 





s_waitcnt vmcnt(22)
ds_write_b8 v30, v26 offset:4096 // (op 6298) Store local data 

v_and_b32 v22, -4, v1 // (op 6993) 
v_sub_u32 v27, v1, v22 // (op 6993) 

v_lshlrev_b32 v22, 5, v27 // (op 6993) 

v_lshrrev_b32 v27, 2, v1 // (op 6993) 
v_lshl_add_u32 v30, v27, 8, v22 // (op 6993) 


v_lshlrev_b32 v22, 2, v30 // (op 6993) 

v_lshrrev_b32 v27, 3, v22 // (op 6993) 




s_waitcnt vmcnt(21)
ds_write_b128 v27, v[38:41] offset:4352 // (op 6299) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v27, v[42:45] offset:4416 // (op 6299) Store local data 

v_and_b32 v22, -4, v1 // (op 7109) 
v_sub_u32 v27, v1, v22 // (op 7109) 

v_lshrrev_b32 v22, 2, v1 // (op 7109) 
v_lshl_add_u32 v30, v22, 3, v27 // (op 7109) 





s_waitcnt vmcnt(19)
ds_write_b8 v30, v46 offset:12544 // (op 6300) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v30, v47 offset:12548 // (op 6300) Store local data 

v_and_b32 v22, -4, v1 // (op 6964) 
v_sub_u32 v27, v1, v22 // (op 6964) 

v_lshlrev_b32 v22, 5, v27 // (op 6964) 

v_lshrrev_b32 v27, 2, v1 // (op 6964) 
v_lshl_add_u32 v30, v27, 8, v22 // (op 6964) 


v_lshlrev_b32 v22, 2, v30 // (op 6964) 

v_lshrrev_b32 v27, 3, v22 // (op 6964) 

v_and_b32 v22, -4, v1 // (op 7022) 
v_sub_u32 v30, v1, v22 // (op 7022) 

v_lshlrev_b32 v22, 5, v30 // (op 7022) 

v_lshrrev_b32 v30, 2, v1 // (op 7022) 
v_lshl_add_u32 v37, v30, 7, v22 // (op 7022) 


v_lshlrev_b32 v22, 2, v37 // (op 7022) 

v_lshrrev_b32 v30, 3, v22 // (op 7022) 

v_and_b32 v22, -4, v1 // (op 7080) 
v_sub_u32 v37, v1, v22 // (op 7080) 

v_lshrrev_b32 v22, 2, v1 // (op 7080) 
v_lshl_add_u32 v52, v22, 3, v37 // (op 7080) 


v_and_b32 v22, -4, v1 // (op 7138) 
v_sub_u32 v37, v1, v22 // (op 7138) 

v_and_b32 v22, 4294967292, v1 // (op 7138) 
v_add_u32 v53, v22, v37 // (op 7138) 


v_and_b32 v22, -4, v1 // (op 7219) 
v_sub_u32 v37, v1, v22 // (op 7219) 

v_lshlrev_b32 v22, 5, v37 // (op 7219) 

v_lshrrev_b32 v37, 2, v1 // (op 7219) 
v_lshl_add_u32 v92, v37, 8, v22 // (op 7219) 


v_lshlrev_b32 v22, 2, v92 // (op 7219) 

v_lshrrev_b32 v37, 3, v22 // (op 7219) 

v_and_b32 v22, -4, v1 // (op 7248) 
v_sub_u32 v92, v1, v22 // (op 7248) 

v_lshlrev_b32 v22, 5, v92 // (op 7248) 

v_lshrrev_b32 v92, 2, v1 // (op 7248) 
v_lshl_add_u32 v93, v92, 7, v22 // (op 7248) 


v_lshlrev_b32 v22, 2, v93 // (op 7248) 

v_lshrrev_b32 v92, 3, v22 // (op 7248) 

v_and_b32 v22, -4, v1 // (op 7277) 
v_sub_u32 v93, v1, v22 // (op 7277) 

v_lshrrev_b32 v22, 2, v1 // (op 7277) 
v_lshl_add_u32 v94, v22, 3, v93 // (op 7277) 


v_and_b32 v22, -4, v1 // (op 7306) 
v_sub_u32 v93, v1, v22 // (op 7306) 

v_and_b32 v22, 4294967292, v1 // (op 7306) 
v_add_u32 v95, v22, v93 // (op 7306) 


v_and_b32 v22, -4, v1 // (op 7335) 
v_sub_u32 v93, v1, v22 // (op 7335) 

v_lshlrev_b32 v22, 5, v93 // (op 7335) 

v_lshrrev_b32 v93, 2, v1 // (op 7335) 
v_lshl_add_u32 v96, v93, 8, v22 // (op 7335) 


v_lshlrev_b32 v22, 2, v96 // (op 7335) 

v_lshrrev_b32 v93, 3, v22 // (op 7335) 

v_and_b32 v22, -4, v1 // (op 7364) 
v_sub_u32 v96, v1, v22 // (op 7364) 

v_lshlrev_b32 v22, 5, v96 // (op 7364) 

v_lshrrev_b32 v96, 2, v1 // (op 7364) 
v_lshl_add_u32 v97, v96, 7, v22 // (op 7364) 


v_lshlrev_b32 v22, 2, v97 // (op 7364) 

v_lshrrev_b32 v96, 3, v22 // (op 7364) 

v_and_b32 v22, -4, v1 // (op 7393) 
v_sub_u32 v97, v1, v22 // (op 7393) 

v_lshrrev_b32 v22, 2, v1 // (op 7393) 
v_lshl_add_u32 v98, v22, 3, v97 // (op 7393) 


v_and_b32 v22, -4, v1 // (op 7422) 
v_sub_u32 v97, v1, v22 // (op 7422) 

v_and_b32 v22, 4294967292, v1 // (op 7422) 
v_add_u32 v99, v22, v97 // (op 7422) 


v_and_b32 v22, -4, v1 // (op 7451) 
v_sub_u32 v97, v1, v22 // (op 7451) 

v_lshlrev_b32 v22, 5, v97 // (op 7451) 

v_lshrrev_b32 v97, 2, v1 // (op 7451) 
v_lshl_add_u32 v100, v97, 8, v22 // (op 7451) 


v_lshlrev_b32 v22, 2, v100 // (op 7451) 

v_lshrrev_b32 v97, 3, v22 // (op 7451) 

v_and_b32 v22, -4, v1 // (op 7480) 
v_sub_u32 v100, v1, v22 // (op 7480) 

v_lshlrev_b32 v22, 5, v100 // (op 7480) 

v_lshrrev_b32 v100, 2, v1 // (op 7480) 
v_lshl_add_u32 v101, v100, 7, v22 // (op 7480) 


v_lshlrev_b32 v22, 2, v101 // (op 7480) 

v_lshrrev_b32 v100, 3, v22 // (op 7480) 

v_and_b32 v22, -4, v1 // (op 7509) 
v_sub_u32 v101, v1, v22 // (op 7509) 

v_lshrrev_b32 v22, 2, v1 // (op 7509) 
v_lshl_add_u32 v102, v22, 3, v101 // (op 7509) 


v_and_b32 v22, -4, v1 // (op 7538) 
v_sub_u32 v101, v1, v22 // (op 7538) 

v_and_b32 v22, 4294967292, v1 // (op 7538) 
v_add_u32 v103, v22, v101 // (op 7538) 


s_mov_b32 s0, 0 // (op 2177) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 2176) 
s_mov_b32 s1, s6 // (op 2176) convert

s_and_b32 s6, s1, 4294967292 // (op 2176) 

s_cmp_lt_u32 s0, s6 // (op 2176) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_2176 // (op 2176) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_2176 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoop_2176:


s_barrier  // (op 7658) 
s_add_u32 s0, s0, 4 // (op 2178) 



v_mov_b32 v22, v25 // (op 10) 

ds_read_b128 v[104:107], v3 offset:4352 // (op 72) Load local data 
buffer_load_dwordx4 v[32:35], v22, s[52:55], 0 offen offset:256 // (op 10) Load value


s_barrier  // (op 7671) 
v_mov_b32 v101, v31 // (op 25) 




buffer_load_ubyte v26, v101, s[60:63], 0 offen offset:16 // (op 25) Load value
ds_read_b128 v[108:111], v4 // (op 77) Load local data 

s_barrier  // (op 7685) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 4) 



ds_read_u8 v112, v0 offset:12544 // (op 82) Load local data 
v_lshrrev_b64 v[114:115], 3, s[6:7] // (op 4) 


s_barrier  // (op 7699) 
v_mov_b32 v113, v23 // (op 4) 



ds_read_u8 v116, v5 offset:4096 // (op 87) Load local data 
v_mov_b32 v117, v113 // (op 4) 



buffer_load_dwordx4 v[38:41], v117, s[48:51], 0 offen offset:256 // (op 4) Load value
v_add_u32 v117, v117, v114 // (op 4) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[108:111], a[0:3], v112, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 90) 
ds_read_b128 v[118:121], v4 offset:2048 // (op 521) Load local data 
buffer_load_dwordx4 v[42:45], v117, s[48:51], 0 offen offset:256 // (op 4) Load value






ds_read_u8 v117, v5 offset:4224 // (op 525) Load local data 
v_mov_b64 v[114:115], s[28:29] // (op 17) convert




v_mov_b32 v122, v28 // (op 17) 
v_mov_b32 v123, v122 // (op 17) 
buffer_load_ubyte v46, v123, s[56:59], 0 offen offset:16 // (op 17) Load value
v_add_u32 v123, v123, v114 // (op 17) 
ds_read_b128 v[124:127], v3 offset:6400 // (op 918) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[118:121], a[4:7], v112, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 526) 





ds_read_u8 v112, v0 offset:12672 // (op 921) Load local data 
buffer_load_ubyte v47, v123, s[56:59], 0 offen offset:16 // (op 17) Load value





s_barrier  // (op 7722) 
ds_read_b128 v[104:107], v3 offset:8448 // (op 1564) Load local data 






ds_read_u8 v123, v0 offset:12800 // (op 1567) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[108:111], a[8:11], v112, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1015) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[118:121], a[12:15], v112, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 923) 
ds_read_b128 v[128:131], v3 offset:10496 // (op 2210) Load local data 





ds_read_u8 v112, v0 offset:12928 // (op 2213) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[108:111], a[16:19], v123, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1661) 
s_waitcnt vmcnt(23)
ds_write_b128 v92, v[48:51] offset:13056 // (op 257) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[118:121], a[20:23], v123, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1569) 


s_barrier  // (op 7734) 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[108:111], a[24:27], v112, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2307) 
s_waitcnt vmcnt(22)
ds_write_b8 v95, v29 offset:17152 // (op 259) Store local data 


s_barrier  // (op 7716) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[118:121], a[28:31], v112, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2215) 







s_waitcnt vmcnt(21)
ds_write_b128 v37, v[54:57] offset:17408 // (op 247) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v37, v[58:61] offset:17472 // (op 247) Store local data 
s_barrier  // (op 7728) 



s_waitcnt vmcnt(19)
ds_write_b8 v94, v62 offset:25600 // (op 258) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v94, v63 offset:25604 // (op 258) Store local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 7713) 
buffer_load_dwordx4 v[48:51], v22, s[52:55], 0 offen offset:320 // (op 260) Load value






ds_read_b128 v[104:107], v7 offset:17408 // (op 249) Load local data 
buffer_load_ubyte v29, v101, s[60:63], 0 offen offset:20 // (op 262) Load value

s_barrier  // (op 7719) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 246) 



v_lshrrev_b64 v[108:109], 3, s[6:7] // (op 246) 


v_mov_b32 v110, v113 // (op 246) 
ds_read_b128 v[114:117], v8 offset:13056 // (op 251) Load local data 
buffer_load_dwordx4 v[54:57], v110, s[48:51], 0 offen offset:320 // (op 246) Load value
v_add_u32 v110, v110, v108 // (op 246) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7725) 
buffer_load_dwordx4 v[58:61], v110, s[48:51], 0 offen offset:320 // (op 246) Load value






ds_read_u8 v108, v6 offset:25600 // (op 253) Load local data 
v_mov_b64 v[118:119], s[28:29] // (op 261) convert

s_barrier  // (op 7731) 
v_mov_b32 v109, v122 // (op 261) 



buffer_load_ubyte v62, v109, s[56:59], 0 offen offset:20 // (op 261) Load value
v_add_u32 v109, v109, v118 // (op 261) 
ds_read_u8 v110, v9 offset:17152 // (op 255) Load local data 
buffer_load_ubyte v63, v109, s[56:59], 0 offen offset:20 // (op 261) Load value





s_barrier  // (op 7746) 
ds_read_b128 v[118:121], v8 offset:15104 // (op 533) Load local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[114:117], a[0:3], v108, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 256) 



s_barrier  // (op 7863) 
ds_read_u8 v109, v9 offset:17280 // (op 537) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v96, v[64:67] offset:26112 // (op 297) Store local data 



s_barrier  // (op 7867) 
s_barrier  // (op 7758) 
ds_read_b128 v[124:127], v7 offset:19456 // (op 928) Load local data 






s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[118:121], a[4:7], v108, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 527) 
ds_read_u8 v123, v6 offset:25728 // (op 931) Load local data 





ds_read_b128 v[104:107], v7 offset:21504 // (op 1574) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[114:117], a[8:11], v123, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1016) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[118:121], a[12:15], v123, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 924) 
ds_read_u8 v108, v6 offset:25856 // (op 1577) Load local data 





ds_read_b128 v[124:127], v7 offset:23552 // (op 2220) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[114:117], a[16:19], v108, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1662) 



s_barrier  // (op 7927) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[118:121], a[20:23], v108, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1570) 
ds_read_u8 v123, v6 offset:25984 // (op 2223) Load local data 


s_waitcnt vmcnt(22)
ds_write_b8 v99, v36 offset:30208 // (op 299) Store local data 
s_barrier  // (op 7931) 
s_barrier  // (op 7740) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[114:117], a[24:27], v123, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2308) 





s_waitcnt vmcnt(21)
ds_write_b128 v93, v[68:71] offset:13056 // (op 287) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[118:121], a[28:31], v123, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2216) 
s_waitcnt vmcnt(20)
ds_write_b128 v93, v[72:75] offset:13120 // (op 287) Store local data 




s_barrier  // (op 7752) 



s_waitcnt vmcnt(19)
ds_write_b8 v98, v76 offset:21248 // (op 298) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v98, v77 offset:21252 // (op 298) Store local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 7737) 
buffer_load_dwordx4 v[64:67], v22, s[52:55], 0 offen offset:384 // (op 300) Load value






ds_read_b128 v[104:107], v11 offset:13056 // (op 289) Load local data 
buffer_load_ubyte v36, v101, s[60:63], 0 offen offset:24 // (op 302) Load value

s_barrier  // (op 7743) 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 286) 



v_lshrrev_b64 v[108:109], 3, s[6:7] // (op 286) 


v_mov_b32 v110, v113 // (op 286) 
ds_read_b128 v[114:117], v12 offset:26112 // (op 291) Load local data 
buffer_load_dwordx4 v[68:71], v110, s[48:51], 0 offen offset:384 // (op 286) Load value
v_add_u32 v110, v110, v108 // (op 286) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 7749) 
buffer_load_dwordx4 v[72:75], v110, s[48:51], 0 offen offset:384 // (op 286) Load value






ds_read_u8 v108, v10 offset:21248 // (op 293) Load local data 
v_mov_b64 v[118:119], s[28:29] // (op 301) convert

s_barrier  // (op 7755) 
v_mov_b32 v109, v122 // (op 301) 



buffer_load_ubyte v76, v109, s[56:59], 0 offen offset:24 // (op 301) Load value
v_add_u32 v109, v109, v118 // (op 301) 
ds_read_u8 v110, v13 offset:30208 // (op 295) Load local data 
buffer_load_ubyte v77, v109, s[56:59], 0 offen offset:24 // (op 301) Load value





s_barrier  // (op 7770) 
ds_read_b128 v[118:121], v12 offset:28160 // (op 541) Load local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[114:117], a[0:3], v108, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 296) 



s_barrier  // (op 7871) 
ds_read_u8 v109, v13 offset:30336 // (op 545) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v100, v[78:81] offset:21760 // (op 337) Store local data 



s_barrier  // (op 7875) 
s_barrier  // (op 7782) 
ds_read_b128 v[124:127], v11 offset:15104 // (op 934) Load local data 






s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[118:121], a[4:7], v108, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 528) 
ds_read_u8 v123, v10 offset:21376 // (op 937) Load local data 





ds_read_b128 v[104:107], v11 offset:17152 // (op 1580) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[114:117], a[8:11], v123, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1017) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[118:121], a[12:15], v123, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 925) 
ds_read_u8 v108, v10 offset:21504 // (op 1583) Load local data 





ds_read_b128 v[124:127], v11 offset:19200 // (op 2226) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[114:117], a[16:19], v108, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1663) 



s_barrier  // (op 7935) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[118:121], a[20:23], v108, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1571) 
ds_read_u8 v123, v10 offset:21632 // (op 2229) Load local data 


s_waitcnt vmcnt(22)
ds_write_b8 v103, v24 offset:25856 // (op 339) Store local data 
s_barrier  // (op 7939) 
s_barrier  // (op 7764) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[114:117], a[24:27], v123, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2309) 





s_waitcnt vmcnt(21)
ds_write_b128 v97, v[82:85] offset:13056 // (op 327) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[118:121], a[28:31], v123, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2217) 
s_waitcnt vmcnt(20)
ds_write_b128 v97, v[86:89] offset:13120 // (op 327) Store local data 




s_barrier  // (op 7776) 



s_waitcnt vmcnt(19)
ds_write_b8 v102, v90 offset:21248 // (op 338) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v102, v91 offset:21252 // (op 338) Store local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 7761) 
buffer_load_dwordx4 v[78:81], v22, s[52:55], 0 offen offset:448 // (op 340) Load value




ds_read_b128 v[104:107], v15 offset:13056 // (op 329) Load local data 



s_barrier  // (op 7767) 
v_add_u32 v25, 256, v25 // (op 6785) 



ds_read_b128 v[108:111], v16 offset:21760 // (op 331) Load local data 
buffer_load_ubyte v24, v101, s[60:63], 0 offen offset:28 // (op 342) Load value

s_waitcnt lgkmcnt(2)
s_barrier  // (op 7773) 




v_add_u32 v31, 16, v31 // (op 6913) 
ds_read_u8 v22, v14 offset:21248 // (op 333) Load local data 
s_lshl_b64 s[6:7], s[12:13], 2 // (op 326) 
s_barrier  // (op 7779) 
v_lshrrev_b64 v[114:115], 3, s[6:7] // (op 326) 





v_mov_b32 v101, v113 // (op 326) 
buffer_load_dwordx4 v[82:85], v101, s[48:51], 0 offen offset:448 // (op 326) Load value
v_add_u32 v101, v101, v114 // (op 326) 
ds_read_u8 v112, v17 offset:25856 // (op 335) Load local data 
buffer_load_dwordx4 v[86:89], v101, s[48:51], 0 offen offset:448 // (op 326) Load value






ds_read_b128 v[118:121], v16 offset:23808 // (op 549) Load local data 

v_add_u32 v23, 256, v23 // (op 6721) 



s_barrier  // (op 7879) 
ds_read_u8 v101, v17 offset:25984 // (op 553) Load local data 
v_mov_b64 v[116:117], s[28:29] // (op 341) convert

v_mov_b32 v113, v122 // (op 341) 



s_barrier  // (op 7883) 
buffer_load_ubyte v90, v113, s[56:59], 0 offen offset:28 // (op 341) Load value
v_add_u32 v113, v113, v116 // (op 341) 
ds_read_b128 v[124:127], v15 offset:15104 // (op 940) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[108:111], a[0:3], v22, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 336) 



buffer_load_ubyte v91, v113, s[56:59], 0 offen offset:28 // (op 341) Load value


s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[118:121], a[4:7], v22, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 529) 



s_barrier  // (op 7674) 
v_add_u32 v28, 16, v28 // (op 6849) 



ds_read_u8 v22, v14 offset:21376 // (op 943) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v30, v[32:35] // (op 13) Store local data 



ds_read_b128 v[104:107], v15 offset:17152 // (op 1586) Load local data 



s_barrier  // (op 7702) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[108:111], a[8:11], v22, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1018) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[118:121], a[12:15], v22, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 926) 
ds_read_u8 v113, v14 offset:21504 // (op 1589) Load local data 





ds_read_b128 v[114:117], v15 offset:19200 // (op 2232) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[108:111], a[16:19], v113, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1664) 



s_barrier  // (op 7943) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[118:121], a[20:23], v113, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1572) 
ds_read_u8 v22, v14 offset:21632 // (op 2235) Load local data 


s_waitcnt vmcnt(22)
ds_write_b8 v53, v26 offset:4096 // (op 30) Store local data 
s_barrier  // (op 7947) 
s_barrier  // (op 7661) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:117], v[108:111], a[24:27], v22, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2310) 





s_waitcnt vmcnt(21)
ds_write_b128 v27, v[38:41] offset:4352 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[114:117], v[118:121], a[28:31], v22, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2218) 
s_waitcnt vmcnt(20)
ds_write_b128 v27, v[42:45] offset:4416 // (op 6) Store local data 




s_barrier  // (op 7688) 



s_waitcnt vmcnt(19)
ds_write_b8 v52, v46 offset:12544 // (op 21) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v52, v47 offset:12548 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 2176) 
s_mov_b32 s1, s6 // (op 2176) convert

s_and_b32 s6, s1, 4294967292 // (op 2176) 

s_cmp_lt_u32 s0, s6 // (op 2176) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoop_2176 // (op 2176) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoop_2176 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoop_2176:

























v_lshrrev_b32 v22, 2, v1 // (op 6677) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6677) 
v_lshlrev_b32 v23, 1, v22 // (op 6677) 

s_lshr_b32 s6, s0, 2 // (op 6677) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 6677) 

v_and_b32 v23, -4, v1 // (op 6677) 
v_sub_u32 v25, v1, v23 // (op 6677) 

v_lshlrev_b32 v23, 5, v25 // (op 6677) 

v_lshl_add_u32 v25, s6, 9, v23 // (op 6677) 


v_mov_b32 v53, 0 // (op 6677) convert
v_mov_b32 v52, v25 // (op 6677) convert

v_mul_lo_u32 v31, v22, s13 // (op 6677) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 6677) most significant: high of low * low
v_mul_lo_u32 v30, v22, s12 // (op 6677) least significant: low of low * low
v_add_u32 v31, v31, v23 // (op 6677) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v52 // (op 6677) least significant half
v_addc_co_u32 v31, s[0:1], v31, v53, s[0:1] // (op 6677) most significant half



v_lshlrev_b64 v[22:23], 2, v[30:31] // (op 6677) 

v_lshrrev_b64 v[52:53], 3, v[22:23] // (op 6677) 

v_mov_b32 v22, v52 // (op 6677) convert

s_mul_i32 s1, s12, s23 // (op 6679) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6679) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6679) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6679) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6679) most significant: sum
s_add_u32 s1, s1, s7 // (op 6679) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6679) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6679) 

s_mov_b32 s48, s20 // (op 6679) reg expression
s_mov_b32 s49, s21 // (op 6679) reg expression
s_mov_b32 s50, s0 // (op 6679) reg expression
s_mov_b32 s51, 131072 // (op 6679) call()

v_and_b32 v23, -4, v1 // (op 6751) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6751) 
v_sub_u32 v25, v1, v23 // (op 6751) 

s_lshr_b32 s6, s0, 2 // (op 6751) 

v_lshlrev_b32 v23, 5, v25 // (op 6751) 

v_lshl_add_u32 v25, s6, 9, v23 // (op 6751) 


v_mov_b32 v53, 0 // (op 6751) convert
v_mov_b32 v52, v25 // (op 6751) convert

v_lshrrev_b32 v23, 2, v1 // (op 6751) 
v_lshl_add_u32 v25, s17, 6, v23 // (op 6751) 

v_mul_lo_u32 v27, v25, s25 // (op 6751) most significant: low of low * high
v_mul_hi_u32 v23, v25, s24 // (op 6751) most significant: high of low * low
v_mul_lo_u32 v26, v25, s24 // (op 6751) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 6751) most significant: sum


v_add_co_u32 v30, s[0:1], v52, v26 // (op 6751) least significant half
v_addc_co_u32 v31, s[0:1], v53, v27, s[0:1] // (op 6751) most significant half

v_lshlrev_b64 v[30:31], 2, v[30:31] // (op 6751) 


v_lshrrev_b64 v[52:53], 3, v[30:31] // (op 6751) 

v_mov_b32 v23, v52 // (op 6751) convert

s_mul_i32 s1, s24, s9 // (op 6753) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 6753) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 6753) least significant
s_mul_hi_u32 s7, s24, s8 // (op 6753) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6753) most significant: sum
s_add_u32 s1, s1, s7 // (op 6753) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6753) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6753) 

s_mov_b32 s52, s26 // (op 6753) reg expression
s_mov_b32 s53, s27 // (op 6753) reg expression
s_mov_b32 s54, s0 // (op 6753) reg expression
s_mov_b32 s55, 131072 // (op 6753) call()

v_lshrrev_b32 v25, 2, v1 // (op 6815) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6815) 
v_lshlrev_b32 v26, 1, v25 // (op 6815) 

s_lshr_b32 s6, s0, 2 // (op 6815) 

v_lshl_add_u32 v25, s16, 7, v26 // (op 6815) 

v_and_b32 v26, -4, v1 // (op 6815) 
v_sub_u32 v27, v1, v26 // (op 6815) 

v_lshl_add_u32 v26, s6, 4, v27 // (op 6815) 


v_mul_lo_u32 v27, v25, s28 // (op 6815) 
v_add_u32 v27, v27, v26 // (op 6815) 


s_mul_i32 s1, s28, s23 // (op 6817) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 6817) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 6817) least significant
s_mul_hi_u32 s7, s28, s22 // (op 6817) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6817) most significant: sum
s_add_u32 s1, s1, s7 // (op 6817) most significant: sum


s_mov_b32 s56, s30 // (op 6817) reg expression
s_mov_b32 s57, s31 // (op 6817) reg expression
s_mov_b32 s58, s0 // (op 6817) reg expression
s_mov_b32 s59, 131072 // (op 6817) call()

v_and_b32 v28, -4, v1 // (op 6879) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6879) 
v_sub_u32 v25, v1, v28 // (op 6879) 

s_lshr_b32 s6, s0, 2 // (op 6879) 

v_lshrrev_b32 v26, 2, v1 // (op 6879) 
v_lshl_add_u32 v28, s6, 4, v25 // (op 6879) 


v_lshl_add_u32 v25, s17, 6, v26 // (op 6879) 

v_mul_lo_u32 v26, v25, s32 // (op 6879) 

v_add_u32 v25, v28, v26 // (op 6879) 


s_mul_i32 s1, s32, s9 // (op 6881) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 6881) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 6881) least significant
s_mul_hi_u32 s7, s32, s8 // (op 6881) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6881) most significant: sum
s_add_u32 s1, s1, s7 // (op 6881) most significant: sum


s_mov_b32 s60, s34 // (op 6881) reg expression
s_mov_b32 s61, s35 // (op 6881) reg expression
s_mov_b32 s62, s0 // (op 6881) reg expression
s_mov_b32 s63, 131072 // (op 6881) call()

v_and_b32 v26, -4, v1 // (op 6977) 
v_sub_u32 v28, v1, v26 // (op 6977) 

v_lshlrev_b32 v26, 5, v28 // (op 6977) 

v_lshrrev_b32 v28, 2, v1 // (op 6977) 
v_lshl_add_u32 v30, v28, 8, v26 // (op 6977) 


v_lshlrev_b32 v26, 2, v30 // (op 6977) 

v_lshrrev_b32 v28, 3, v26 // (op 6977) 

v_and_b32 v26, -4, v1 // (op 7035) 
v_sub_u32 v30, v1, v26 // (op 7035) 

v_lshlrev_b32 v26, 5, v30 // (op 7035) 

v_lshrrev_b32 v30, 2, v1 // (op 7035) 
v_lshl_add_u32 v31, v30, 7, v26 // (op 7035) 


v_lshlrev_b32 v26, 2, v31 // (op 7035) 

v_lshrrev_b32 v30, 3, v26 // (op 7035) 

v_and_b32 v26, -4, v1 // (op 7093) 
v_sub_u32 v31, v1, v26 // (op 7093) 

v_lshrrev_b32 v26, 2, v1 // (op 7093) 
v_lshl_add_u32 v32, v26, 3, v31 // (op 7093) 


v_and_b32 v26, -4, v1 // (op 7151) 
v_sub_u32 v31, v1, v26 // (op 7151) 

v_and_b32 v26, 4294967292, v1 // (op 7151) 
v_add_u32 v33, v26, v31 // (op 7151) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 2182) 
s_and_b32 s6, s0, 4294967292 // (op 2182) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 2181) 
s_mov_b32 s7, s0 // (op 2181) convert

s_cmp_lt_u32 s6, s7 // (op 2181) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_2181 // (op 2181) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_2181 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_2181:

s_add_u32 s6, s6, 1 // (op 2183) 
s_lshl_b64 s[0:1], s[12:13], 2 // (op 180) 
v_lshrrev_b64 v[34:35], 3, s[0:1] // (op 180) 


v_mov_b32 v24, v22 // (op 180) 
v_mov_b32 v26, v24 // (op 180) 
buffer_load_dwordx4 v[36:39], v26, s[48:51], 0 offen // (op 180) Load value
v_add_u32 v26, v26, v34 // (op 180) 
buffer_load_dwordx4 v[40:43], v26, s[48:51], 0 offen // (op 180) Load value



v_add_u32 v22, 64, v22 // (op 6667) 


s_barrier  // (op 7664) 
v_mov_b32 v24, v23 // (op 194) 




buffer_load_dwordx4 v[44:47], v24, s[52:55], 0 offen // (op 194) Load value

v_add_u32 v23, 64, v23 // (op 6743) 
s_waitcnt vmcnt(2)
ds_write_b128 v28, v[36:39] offset:4352 // (op 181) Store local data 

v_mov_b64 v[34:35], s[28:29] // (op 195) convert

v_mov_b32 v24, v27 // (op 195) 
v_mov_b32 v26, v24 // (op 195) 
buffer_load_ubyte v48, v26, s[56:59], 0 offen offset: 0 // (op 195) Load value
v_add_u32 v26, v26, v34 // (op 195) 
buffer_load_ubyte v49, v26, s[56:59], 0 offen offset: 0 // (op 195) Load value



v_add_u32 v27, 4, v27 // (op 6807) 
s_waitcnt vmcnt(3)
ds_write_b128 v28, v[40:43] offset:4416 // (op 181) Store local data 



v_mov_b32 v24, v25 // (op 196) 

s_barrier  // (op 7677) 
buffer_load_ubyte v26, v24, s[60:63], 0 offen offset: 0 // (op 196) Load value




v_add_u32 v25, 4, v25 // (op 6871) 
s_waitcnt vmcnt(3)
ds_write_b128 v30, v[44:47] // (op 191) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 7667) 



s_barrier  // (op 7691) 
ds_read_b128 v[34:37], v3 offset:4352 // (op 183) Load local data 



s_waitcnt lgkmcnt(1)
s_barrier  // (op 7681) 
s_waitcnt vmcnt(2)
ds_write_b8 v32, v48 offset:12544 // (op 192) Store local data 



ds_read_b128 v[38:41], v4 // (op 185) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v32, v49 offset:12548 // (op 192) Store local data 

s_barrier  // (op 7705) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7695) 






ds_read_u8 v24, v0 offset:12544 // (op 187) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v33, v26 offset:4096 // (op 193) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 7709) 



ds_read_u8 v26, v5 offset:4096 // (op 189) Load local data 



ds_read_b128 v[42:45], v4 offset:2048 // (op 500) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:41], a[0:3], v24, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 190) 



ds_read_u8 v29, v5 offset:4224 // (op 504) Load local data 



ds_read_b128 v[46:49], v3 offset:6400 // (op 864) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[42:45], a[4:7], v24, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 505) 





ds_read_u8 v24, v0 offset:12672 // (op 867) Load local data 



ds_read_b128 v[34:37], v3 offset:8448 // (op 1510) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[38:41], a[8:11], v24, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 869) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[42:45], a[12:15], v24, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 900) 
ds_read_u8 v31, v0 offset:12800 // (op 1513) Load local data 





ds_read_b128 v[46:49], v3 offset:10496 // (op 2156) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[34:37], v[38:41], a[16:19], v31, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1515) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:37], v[42:45], a[20:23], v31, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1546) 
ds_read_u8 v24, v0 offset:12928 // (op 2159) Load local data 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[46:49], v[38:41], a[24:27], v24, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2161) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[42:45], a[28:31], v24, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2192) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 2181) 
s_mov_b32 s7, s0 // (op 2181) convert

s_cmp_lt_u32 s6, s7 // (op 2181) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_2181 // (op 2181) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopTop_KLoopTail_2181 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ForLoopBottom_KLoopTail_2181:























s_barrier  // (op 7911) 

















v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_135
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_mul_f32 v10, v19, v14 // (op 50) 
v_mul_f32 v11, v19, v15 // (op 50) 
v_mul_f32 v12, v19, v16 // (op 50) 
v_mul_f32 v13, v19, v17 // (op 50) 

v_cvt_f16_f32 v14, v10 // (op 56) 
v_cvt_f16_f32 v15, v11 // (op 56) 
v_cvt_f16_f32 v16, v12 // (op 56) 
v_cvt_f16_f32 v17, v13 // (op 56) 


v_pack_b32_f16 v12, v14, v15 // (op 58) 
v_pack_b32_f16 v13, v16, v17 // (op 58) 

v_mov_b32 v0, v21 // (op 58) reg expression

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v14, a4 // (op 3041) 
v_accvgpr_read v15, a5 // (op 3041) 
v_accvgpr_read v16, a6 // (op 3041) 
v_accvgpr_read v17, a7 // (op 3041) 
v_mul_f32 v10, v19, v14 // (op 3041) 
v_mul_f32 v11, v19, v15 // (op 3041) 
v_mul_f32 v12, v19, v16 // (op 3041) 
v_mul_f32 v13, v19, v17 // (op 3041) 

v_cvt_f16_f32 v14, v10 // (op 3043) 
v_cvt_f16_f32 v15, v11 // (op 3043) 
v_cvt_f16_f32 v16, v12 // (op 3043) 
v_cvt_f16_f32 v17, v13 // (op 3043) 


v_pack_b32_f16 v12, v14, v15 // (op 3044) 
v_pack_b32_f16 v13, v16, v17 // (op 3044) 

s_lshl_b32 s0, s4, 6 // (op 3044) 
v_add_u32 v0, v21, s0 // (op 3044) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3044) Store value



v_accvgpr_read v14, a8 // (op 3067) 
v_accvgpr_read v15, a9 // (op 3067) 
v_accvgpr_read v16, a10 // (op 3067) 
v_accvgpr_read v17, a11 // (op 3067) 
v_mul_f32 v10, v19, v14 // (op 3067) 
v_mul_f32 v11, v19, v15 // (op 3067) 
v_mul_f32 v12, v19, v16 // (op 3067) 
v_mul_f32 v13, v19, v17 // (op 3067) 

v_cvt_f16_f32 v14, v10 // (op 3069) 
v_cvt_f16_f32 v15, v11 // (op 3069) 
v_cvt_f16_f32 v16, v12 // (op 3069) 
v_cvt_f16_f32 v17, v13 // (op 3069) 


v_pack_b32_f16 v12, v14, v15 // (op 3071) 
v_pack_b32_f16 v13, v16, v17 // (op 3071) 

v_add_u32 v0, 64, v21 // (op 3071) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3071) Store value



v_accvgpr_read v14, a12 // (op 3077) 
v_accvgpr_read v15, a13 // (op 3077) 
v_accvgpr_read v16, a14 // (op 3077) 
v_accvgpr_read v17, a15 // (op 3077) 
v_mul_f32 v10, v19, v14 // (op 3077) 
v_mul_f32 v11, v19, v15 // (op 3077) 
v_mul_f32 v12, v19, v16 // (op 3077) 
v_mul_f32 v13, v19, v17 // (op 3077) 

v_cvt_f16_f32 v14, v10 // (op 3079) 
v_cvt_f16_f32 v15, v11 // (op 3079) 
v_cvt_f16_f32 v16, v12 // (op 3079) 
v_cvt_f16_f32 v17, v13 // (op 3079) 


v_pack_b32_f16 v12, v14, v15 // (op 3073) 
v_pack_b32_f16 v13, v16, v17 // (op 3073) 

s_lshl_b32 s0, s4, 6 // (op 3073) 
s_add_u32 s0, s0, 64 // (op 3073) 
v_add_u32 v0, v21, s0 // (op 3073) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3073) Store value



v_accvgpr_read v14, a16 // (op 3107) 
v_accvgpr_read v15, a17 // (op 3107) 
v_accvgpr_read v16, a18 // (op 3107) 
v_accvgpr_read v17, a19 // (op 3107) 
v_mul_f32 v10, v19, v14 // (op 3107) 
v_mul_f32 v11, v19, v15 // (op 3107) 
v_mul_f32 v12, v19, v16 // (op 3107) 
v_mul_f32 v13, v19, v17 // (op 3107) 

v_cvt_f16_f32 v14, v10 // (op 3109) 
v_cvt_f16_f32 v15, v11 // (op 3109) 
v_cvt_f16_f32 v16, v12 // (op 3109) 
v_cvt_f16_f32 v17, v13 // (op 3109) 


v_pack_b32_f16 v12, v14, v15 // (op 3111) 
v_pack_b32_f16 v13, v16, v17 // (op 3111) 

v_add_u32 v0, 128, v21 // (op 3111) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3111) Store value



v_accvgpr_read v14, a20 // (op 3117) 
v_accvgpr_read v15, a21 // (op 3117) 
v_accvgpr_read v16, a22 // (op 3117) 
v_accvgpr_read v17, a23 // (op 3117) 
v_mul_f32 v10, v19, v14 // (op 3117) 
v_mul_f32 v11, v19, v15 // (op 3117) 
v_mul_f32 v12, v19, v16 // (op 3117) 
v_mul_f32 v13, v19, v17 // (op 3117) 

v_cvt_f16_f32 v14, v10 // (op 3119) 
v_cvt_f16_f32 v15, v11 // (op 3119) 
v_cvt_f16_f32 v16, v12 // (op 3119) 
v_cvt_f16_f32 v17, v13 // (op 3119) 


v_pack_b32_f16 v12, v14, v15 // (op 3113) 
v_pack_b32_f16 v13, v16, v17 // (op 3113) 

s_lshl_b32 s0, s4, 6 // (op 3113) 
s_add_u32 s0, s0, 128 // (op 3113) 
v_add_u32 v0, v21, s0 // (op 3113) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3113) Store value



v_accvgpr_read v14, a24 // (op 3147) 
v_accvgpr_read v15, a25 // (op 3147) 
v_accvgpr_read v16, a26 // (op 3147) 
v_accvgpr_read v17, a27 // (op 3147) 
v_mul_f32 v10, v19, v14 // (op 3147) 
v_mul_f32 v11, v19, v15 // (op 3147) 
v_mul_f32 v12, v19, v16 // (op 3147) 
v_mul_f32 v13, v19, v17 // (op 3147) 

v_cvt_f16_f32 v14, v10 // (op 3149) 
v_cvt_f16_f32 v15, v11 // (op 3149) 
v_cvt_f16_f32 v16, v12 // (op 3149) 
v_cvt_f16_f32 v17, v13 // (op 3149) 


v_pack_b32_f16 v12, v14, v15 // (op 3151) 
v_pack_b32_f16 v13, v16, v17 // (op 3151) 

v_add_u32 v0, 192, v21 // (op 3151) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3151) Store value



v_accvgpr_read v14, a28 // (op 3157) 
v_accvgpr_read v15, a29 // (op 3157) 
v_accvgpr_read v16, a30 // (op 3157) 
v_accvgpr_read v17, a31 // (op 3157) 
v_mul_f32 v10, v19, v14 // (op 3157) 
v_mul_f32 v11, v19, v15 // (op 3157) 
v_mul_f32 v12, v19, v16 // (op 3157) 
v_mul_f32 v13, v19, v17 // (op 3157) 

v_cvt_f16_f32 v14, v10 // (op 3159) 
v_cvt_f16_f32 v15, v11 // (op 3159) 
v_cvt_f16_f32 v16, v12 // (op 3159) 
v_cvt_f16_f32 v17, v13 // (op 3159) 


v_pack_b32_f16 v12, v14, v15 // (op 3153) 
v_pack_b32_f16 v13, v16, v17 // (op 3153) 

s_lshl_b32 s0, s4, 6 // (op 3153) 
s_add_u32 s0, s0, 192 // (op 3153) 
v_add_u32 v0, v21, s0 // (op 3153) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 3153) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 192, v20 // (op 3349) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3349) Load value

v_accvgpr_read v12, a24 // (op 3340) 
v_accvgpr_read v13, a25 // (op 3340) 
v_accvgpr_read v14, a26 // (op 3340) 
v_accvgpr_read v15, a27 // (op 3340) 
v_mul_f32 v8, v19, v12 // (op 3340) 
v_mul_f32 v9, v19, v13 // (op 3340) 
v_mul_f32 v10, v19, v14 // (op 3340) 
v_mul_f32 v11, v19, v15 // (op 3340) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 3348) 
v_lshrrev_b32 v7, 16, v16 // (op 3348) 
v_cvt_f32_f16 v7, v7 // (op 3348) 
v_mul_f32 v12, v18, v6 // (op 3348) 
v_mul_f32 v13, v18, v7 // (op 3348) 
v_cvt_f32_f16 v6, v17 // (op 3348) 
v_lshrrev_b32 v7, 16, v17 // (op 3348) 
v_cvt_f32_f16 v7, v7 // (op 3348) 
v_mul_f32 v14, v18, v6 // (op 3348) 
v_mul_f32 v15, v18, v7 // (op 3348) 


v_add_f32 v4, v12, v8 // (op 3341) 
v_add_f32 v5, v13, v9 // (op 3341) 
v_add_f32 v6, v14, v10 // (op 3341) 
v_add_f32 v7, v15, v11 // (op 3341) 


v_cvt_f16_f32 v8, v4 // (op 3342) 
v_cvt_f16_f32 v9, v5 // (op 3342) 
v_cvt_f16_f32 v10, v6 // (op 3342) 
v_cvt_f16_f32 v11, v7 // (op 3342) 


v_pack_b32_f16 v6, v8, v9 // (op 3344) 
v_pack_b32_f16 v7, v10, v11 // (op 3344) 

v_add_u32 v0, 192, v21 // (op 3344) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3344) Store value





s_lshl_b32 s0, s2, 6 // (op 3354) 
s_add_u32 s0, s0, 192 // (op 3354) 
v_add_u32 v0, v20, s0 // (op 3354) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3354) Load value

v_accvgpr_read v12, a28 // (op 3356) 
v_accvgpr_read v13, a29 // (op 3356) 
v_accvgpr_read v14, a30 // (op 3356) 
v_accvgpr_read v15, a31 // (op 3356) 
v_mul_f32 v8, v19, v12 // (op 3356) 
v_mul_f32 v9, v19, v13 // (op 3356) 
v_mul_f32 v10, v19, v14 // (op 3356) 
v_mul_f32 v11, v19, v15 // (op 3356) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 3351) 
v_lshrrev_b32 v7, 16, v16 // (op 3351) 
v_cvt_f32_f16 v7, v7 // (op 3351) 
v_mul_f32 v12, v18, v6 // (op 3351) 
v_mul_f32 v13, v18, v7 // (op 3351) 
v_cvt_f32_f16 v6, v17 // (op 3351) 
v_lshrrev_b32 v7, 16, v17 // (op 3351) 
v_cvt_f32_f16 v7, v7 // (op 3351) 
v_mul_f32 v14, v18, v6 // (op 3351) 
v_mul_f32 v15, v18, v7 // (op 3351) 


v_add_f32 v4, v12, v8 // (op 3352) 
v_add_f32 v5, v13, v9 // (op 3352) 
v_add_f32 v6, v14, v10 // (op 3352) 
v_add_f32 v7, v15, v11 // (op 3352) 


v_cvt_f16_f32 v8, v4 // (op 3353) 
v_cvt_f16_f32 v9, v5 // (op 3353) 
v_cvt_f16_f32 v10, v6 // (op 3353) 
v_cvt_f16_f32 v11, v7 // (op 3353) 


v_pack_b32_f16 v6, v8, v9 // (op 3346) 
v_pack_b32_f16 v7, v10, v11 // (op 3346) 

s_lshl_b32 s0, s4, 6 // (op 3346) 
s_add_u32 s0, s0, 192 // (op 3346) 
v_add_u32 v0, v21, s0 // (op 3346) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3346) Store value





v_add_u32 v0, 128, v20 // (op 3304) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3304) Load value

v_accvgpr_read v12, a16 // (op 3295) 
v_accvgpr_read v13, a17 // (op 3295) 
v_accvgpr_read v14, a18 // (op 3295) 
v_accvgpr_read v15, a19 // (op 3295) 
v_mul_f32 v8, v19, v12 // (op 3295) 
v_mul_f32 v9, v19, v13 // (op 3295) 
v_mul_f32 v10, v19, v14 // (op 3295) 
v_mul_f32 v11, v19, v15 // (op 3295) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 3303) 
v_lshrrev_b32 v7, 16, v16 // (op 3303) 
v_cvt_f32_f16 v7, v7 // (op 3303) 
v_mul_f32 v12, v18, v6 // (op 3303) 
v_mul_f32 v13, v18, v7 // (op 3303) 
v_cvt_f32_f16 v6, v17 // (op 3303) 
v_lshrrev_b32 v7, 16, v17 // (op 3303) 
v_cvt_f32_f16 v7, v7 // (op 3303) 
v_mul_f32 v14, v18, v6 // (op 3303) 
v_mul_f32 v15, v18, v7 // (op 3303) 


v_add_f32 v4, v12, v8 // (op 3296) 
v_add_f32 v5, v13, v9 // (op 3296) 
v_add_f32 v6, v14, v10 // (op 3296) 
v_add_f32 v7, v15, v11 // (op 3296) 


v_cvt_f16_f32 v8, v4 // (op 3297) 
v_cvt_f16_f32 v9, v5 // (op 3297) 
v_cvt_f16_f32 v10, v6 // (op 3297) 
v_cvt_f16_f32 v11, v7 // (op 3297) 


v_pack_b32_f16 v6, v8, v9 // (op 3299) 
v_pack_b32_f16 v7, v10, v11 // (op 3299) 

v_add_u32 v0, 128, v21 // (op 3299) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3299) Store value





s_lshl_b32 s0, s2, 6 // (op 3309) 
s_add_u32 s0, s0, 128 // (op 3309) 
v_add_u32 v0, v20, s0 // (op 3309) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3309) Load value

v_accvgpr_read v12, a20 // (op 3311) 
v_accvgpr_read v13, a21 // (op 3311) 
v_accvgpr_read v14, a22 // (op 3311) 
v_accvgpr_read v15, a23 // (op 3311) 
v_mul_f32 v8, v19, v12 // (op 3311) 
v_mul_f32 v9, v19, v13 // (op 3311) 
v_mul_f32 v10, v19, v14 // (op 3311) 
v_mul_f32 v11, v19, v15 // (op 3311) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 3306) 
v_lshrrev_b32 v7, 16, v16 // (op 3306) 
v_cvt_f32_f16 v7, v7 // (op 3306) 
v_mul_f32 v12, v18, v6 // (op 3306) 
v_mul_f32 v13, v18, v7 // (op 3306) 
v_cvt_f32_f16 v6, v17 // (op 3306) 
v_lshrrev_b32 v7, 16, v17 // (op 3306) 
v_cvt_f32_f16 v7, v7 // (op 3306) 
v_mul_f32 v14, v18, v6 // (op 3306) 
v_mul_f32 v15, v18, v7 // (op 3306) 


v_add_f32 v4, v12, v8 // (op 3307) 
v_add_f32 v5, v13, v9 // (op 3307) 
v_add_f32 v6, v14, v10 // (op 3307) 
v_add_f32 v7, v15, v11 // (op 3307) 


v_cvt_f16_f32 v8, v4 // (op 3308) 
v_cvt_f16_f32 v9, v5 // (op 3308) 
v_cvt_f16_f32 v10, v6 // (op 3308) 
v_cvt_f16_f32 v11, v7 // (op 3308) 


v_pack_b32_f16 v6, v8, v9 // (op 3301) 
v_pack_b32_f16 v7, v10, v11 // (op 3301) 

s_lshl_b32 s0, s4, 6 // (op 3301) 
s_add_u32 s0, s0, 128 // (op 3301) 
v_add_u32 v0, v21, s0 // (op 3301) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3301) Store value





v_add_u32 v0, 64, v20 // (op 3259) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3259) Load value

v_accvgpr_read v12, a8 // (op 3250) 
v_accvgpr_read v13, a9 // (op 3250) 
v_accvgpr_read v14, a10 // (op 3250) 
v_accvgpr_read v15, a11 // (op 3250) 
v_mul_f32 v8, v19, v12 // (op 3250) 
v_mul_f32 v9, v19, v13 // (op 3250) 
v_mul_f32 v10, v19, v14 // (op 3250) 
v_mul_f32 v11, v19, v15 // (op 3250) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 3258) 
v_lshrrev_b32 v7, 16, v16 // (op 3258) 
v_cvt_f32_f16 v7, v7 // (op 3258) 
v_mul_f32 v12, v18, v6 // (op 3258) 
v_mul_f32 v13, v18, v7 // (op 3258) 
v_cvt_f32_f16 v6, v17 // (op 3258) 
v_lshrrev_b32 v7, 16, v17 // (op 3258) 
v_cvt_f32_f16 v7, v7 // (op 3258) 
v_mul_f32 v14, v18, v6 // (op 3258) 
v_mul_f32 v15, v18, v7 // (op 3258) 


v_add_f32 v4, v12, v8 // (op 3251) 
v_add_f32 v5, v13, v9 // (op 3251) 
v_add_f32 v6, v14, v10 // (op 3251) 
v_add_f32 v7, v15, v11 // (op 3251) 


v_cvt_f16_f32 v8, v4 // (op 3252) 
v_cvt_f16_f32 v9, v5 // (op 3252) 
v_cvt_f16_f32 v10, v6 // (op 3252) 
v_cvt_f16_f32 v11, v7 // (op 3252) 


v_pack_b32_f16 v6, v8, v9 // (op 3254) 
v_pack_b32_f16 v7, v10, v11 // (op 3254) 

v_add_u32 v0, 64, v21 // (op 3254) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3254) Store value





s_lshl_b32 s0, s2, 6 // (op 3264) 
s_add_u32 s0, s0, 64 // (op 3264) 
v_add_u32 v0, v20, s0 // (op 3264) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3264) Load value

v_accvgpr_read v12, a12 // (op 3266) 
v_accvgpr_read v13, a13 // (op 3266) 
v_accvgpr_read v14, a14 // (op 3266) 
v_accvgpr_read v15, a15 // (op 3266) 
v_mul_f32 v8, v19, v12 // (op 3266) 
v_mul_f32 v9, v19, v13 // (op 3266) 
v_mul_f32 v10, v19, v14 // (op 3266) 
v_mul_f32 v11, v19, v15 // (op 3266) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 3261) 
v_lshrrev_b32 v7, 16, v16 // (op 3261) 
v_cvt_f32_f16 v7, v7 // (op 3261) 
v_mul_f32 v12, v18, v6 // (op 3261) 
v_mul_f32 v13, v18, v7 // (op 3261) 
v_cvt_f32_f16 v6, v17 // (op 3261) 
v_lshrrev_b32 v7, 16, v17 // (op 3261) 
v_cvt_f32_f16 v7, v7 // (op 3261) 
v_mul_f32 v14, v18, v6 // (op 3261) 
v_mul_f32 v15, v18, v7 // (op 3261) 


v_add_f32 v4, v12, v8 // (op 3262) 
v_add_f32 v5, v13, v9 // (op 3262) 
v_add_f32 v6, v14, v10 // (op 3262) 
v_add_f32 v7, v15, v11 // (op 3262) 


v_cvt_f16_f32 v8, v4 // (op 3263) 
v_cvt_f16_f32 v9, v5 // (op 3263) 
v_cvt_f16_f32 v10, v6 // (op 3263) 
v_cvt_f16_f32 v11, v7 // (op 3263) 


v_pack_b32_f16 v6, v8, v9 // (op 3256) 
v_pack_b32_f16 v7, v10, v11 // (op 3256) 

s_lshl_b32 s0, s4, 6 // (op 3256) 
s_add_u32 s0, s0, 64 // (op 3256) 
v_add_u32 v0, v21, s0 // (op 3256) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3256) Store value





v_mov_b32 v0, v20 // (op 148) reg expression

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 148) Load value

v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v8, v19, v12 // (op 144) 
v_mul_f32 v9, v19, v13 // (op 144) 
v_mul_f32 v10, v19, v14 // (op 144) 
v_mul_f32 v11, v19, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 149) 
v_lshrrev_b32 v7, 16, v16 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v12, v18, v6 // (op 149) 
v_mul_f32 v13, v18, v7 // (op 149) 
v_cvt_f32_f16 v6, v17 // (op 149) 
v_lshrrev_b32 v7, 16, v17 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v14, v18, v6 // (op 149) 
v_mul_f32 v15, v18, v7 // (op 149) 


v_add_f32 v4, v12, v8 // (op 145) 
v_add_f32 v5, v13, v9 // (op 145) 
v_add_f32 v6, v14, v10 // (op 145) 
v_add_f32 v7, v15, v11 // (op 145) 


v_cvt_f16_f32 v8, v4 // (op 146) 
v_cvt_f16_f32 v9, v5 // (op 146) 
v_cvt_f16_f32 v10, v6 // (op 146) 
v_cvt_f16_f32 v11, v7 // (op 146) 


v_pack_b32_f16 v6, v8, v9 // (op 147) 
v_pack_b32_f16 v7, v10, v11 // (op 147) 

v_mov_b32 v0, v21 // (op 147) reg expression

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 3219) 
v_add_u32 v0, v20, s0 // (op 3219) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 3219) Load value

v_accvgpr_read v12, a4 // (op 3215) 
v_accvgpr_read v13, a5 // (op 3215) 
v_accvgpr_read v14, a6 // (op 3215) 
v_accvgpr_read v15, a7 // (op 3215) 
v_mul_f32 v8, v19, v12 // (op 3215) 
v_mul_f32 v9, v19, v13 // (op 3215) 
v_mul_f32 v10, v19, v14 // (op 3215) 
v_mul_f32 v11, v19, v15 // (op 3215) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 3220) 
v_lshrrev_b32 v7, 16, v16 // (op 3220) 
v_cvt_f32_f16 v7, v7 // (op 3220) 
v_mul_f32 v12, v18, v6 // (op 3220) 
v_mul_f32 v13, v18, v7 // (op 3220) 
v_cvt_f32_f16 v6, v17 // (op 3220) 
v_lshrrev_b32 v7, 16, v17 // (op 3220) 
v_cvt_f32_f16 v7, v7 // (op 3220) 
v_mul_f32 v14, v18, v6 // (op 3220) 
v_mul_f32 v15, v18, v7 // (op 3220) 


v_add_f32 v4, v12, v8 // (op 3216) 
v_add_f32 v5, v13, v9 // (op 3216) 
v_add_f32 v6, v14, v10 // (op 3216) 
v_add_f32 v7, v15, v11 // (op 3216) 


v_cvt_f16_f32 v8, v4 // (op 3217) 
v_cvt_f16_f32 v9, v5 // (op 3217) 
v_cvt_f16_f32 v10, v6 // (op 3217) 
v_cvt_f16_f32 v11, v7 // (op 3217) 


v_pack_b32_f16 v6, v8, v9 // (op 3218) 
v_pack_b32_f16 v7, v10, v11 // (op 3218) 

s_lshl_b32 s0, s4, 6 // (op 3218) 
v_add_u32 v0, v21, s0 // (op 3218) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 3218) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_ConditionalBottom_0_beta_135:
















s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
.LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_end:
.size RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC, .LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC_end-RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
  .amdhsa_next_free_vgpr 164
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 30464 // lds bytes
  .amdhsa_accum_offset 132
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
    macrotile: [128, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 4
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC
    .symbol: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x64x128_UR_4_LDSSYNC.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 30464
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 132
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

