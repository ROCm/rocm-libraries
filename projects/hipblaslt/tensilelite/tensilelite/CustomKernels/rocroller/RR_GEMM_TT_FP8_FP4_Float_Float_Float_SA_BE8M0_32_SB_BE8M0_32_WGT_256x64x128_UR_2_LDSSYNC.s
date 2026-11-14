// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
.p2align 8
.type RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC,@function
RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_exec_begin:


s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value
s_load_dwordx2 s[38:39], s[0:1], 128 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


v_lshrrev_b32 v0, 7, v1 // (op 9011) 
v_and_b32 v3, 15, v1 // (op 9011) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 9011) 


v_and_b32 v0, 48, v1 // (op 9011) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 9011) 


v_and_b32 v0, 63, v1 // (op 9069) 
v_and_b32 v4, 48, v1 // (op 9069) 
v_sub_u32 v5, v0, v4 // (op 9069) 

v_lshrrev_b32 v4, 4, v0 // (op 9069) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 9069) 


v_lshrrev_b32 v4, 6, v1 // (op 9069) 
v_and_b32 v5, 1, v4 // (op 9069) 

v_lshlrev_b32 v4, 4, v5 // (op 9069) 

v_lshl_add_u32 v5, v0, 6, v4 // (op 9069) 


v_lshlrev_b32 v0, 2, v5 // (op 9069) 

v_lshrrev_b32 v4, 3, v0 // (op 9069) 

v_lshrrev_b32 v0, 7, v1 // (op 9127) 
v_and_b32 v5, 15, v1 // (op 9127) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9127) 


v_and_b32 v0, 63, v1 // (op 9127) 
v_lshrrev_b32 v5, 4, v0 // (op 9127) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 9127) 


v_lshrrev_b32 v5, 6, v1 // (op 9185) 
v_and_b32 v6, 1, v5 // (op 9185) 

v_and_b32 v5, 15, v1 // (op 9185) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9185) 


v_and_b32 v5, 63, v1 // (op 9185) 
v_lshrrev_b32 v6, 4, v5 // (op 9185) 

v_lshl_add_u32 v5, v6, 6, v7 // (op 9185) 


v_lshrrev_b32 v6, 7, v1 // (op 9237) 
v_and_b32 v7, 15, v1 // (op 9237) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9237) 


v_and_b32 v6, 48, v1 // (op 9237) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9237) 


v_and_b32 v6, 63, v1 // (op 9266) 
v_and_b32 v8, 48, v1 // (op 9266) 
v_sub_u32 v9, v6, v8 // (op 9266) 

v_lshrrev_b32 v8, 4, v6 // (op 9266) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 9266) 


v_lshrrev_b32 v8, 6, v1 // (op 9266) 
v_and_b32 v9, 1, v8 // (op 9266) 

v_lshlrev_b32 v8, 4, v9 // (op 9266) 

v_lshl_add_u32 v9, v6, 6, v8 // (op 9266) 


v_lshlrev_b32 v6, 2, v9 // (op 9266) 

v_lshrrev_b32 v8, 3, v6 // (op 9266) 

v_lshrrev_b32 v6, 7, v1 // (op 9295) 
v_and_b32 v9, 15, v1 // (op 9295) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9295) 


v_and_b32 v6, 63, v1 // (op 9295) 
v_lshrrev_b32 v9, 4, v6 // (op 9295) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 9295) 


v_lshrrev_b32 v9, 6, v1 // (op 9324) 
v_and_b32 v10, 1, v9 // (op 9324) 

v_and_b32 v9, 15, v1 // (op 9324) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9324) 


v_and_b32 v9, 63, v1 // (op 9324) 
v_lshrrev_b32 v10, 4, v9 // (op 9324) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 9324) 


v_lshrrev_b32 v10, 7, v1 // (op 8949) 
v_and_b32 v11, 63, v1 // (op 8949) 
v_lshrrev_b32 v12, 4, v11 // (op 8949) 
v_lshlrev_b32 v13, 2, v12 // (op 8949) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8949) 


v_lshl_add_u32 v10, s16, 8, v12 // (op 8949) 

v_lshrrev_b32 v12, 2, v11 // (op 8949) 

v_and_b32 v11, 3, v12 // (op 8949) 

v_and_b32 v12, 3, v1 // (op 8949) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 8949) 


v_lshrrev_b32 v11, 6, v1 // (op 8949) 
v_and_b32 v12, 1, v11 // (op 8949) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 8949) 


v_lshl_add_u32 v12, s17, 6, v11 // (op 8949) 

v_mul_lo_u32 v11, v12, s2 // (op 8949) 

v_add_lshl_u32 v12, v10, v11, 2 // (op 8949) 


v_lshrrev_b32 v10, 7, v1 // (op 9206) 
v_and_b32 v11, 63, v1 // (op 9206) 
v_lshrrev_b32 v13, 4, v11 // (op 9206) 
s_mul_i32 s1, s2, s9 // (op 8951) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 8951) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 8951) least significant
s_mul_hi_u32 s19, s2, s8 // (op 8951) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8951) most significant: sum
s_add_u32 s1, s1, s19 // (op 8951) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8951) 

v_lshlrev_b32 v14, 2, v13 // (op 9206) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 9206) 


v_lshl_add_u32 v10, s16, 8, v13 // (op 9206) 

v_lshrrev_b32 v13, 2, v11 // (op 9206) 

v_and_b32 v11, 3, v13 // (op 9206) 

v_and_b32 v13, 3, v1 // (op 9206) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 9206) 


v_lshrrev_b32 v11, 6, v1 // (op 9206) 
v_and_b32 v13, 1, v11 // (op 9206) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 9206) 


v_lshl_add_u32 v13, s17, 6, v11 // (op 9206) 

v_mul_lo_u32 v11, v13, s4 // (op 9206) 

v_add_lshl_u32 v13, v10, v11, 2 // (op 9206) 


s_mov_b32 s40, s6 // (op 8951) reg expression
s_mov_b32 s41, s7 // (op 8951) reg expression
s_mov_b32 s42, s18 // (op 8951) reg expression
s_mov_b32 s43, 131072 // (op 8951) call()

s_mul_i32 s1, s4, s9 // (op 9208) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 9208) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 9208) least significant
s_mul_hi_u32 s7, s4, s8 // (op 9208) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 9208) most significant: sum
s_add_u32 s1, s1, s7 // (op 9208) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 9208) 

s_mov_b32 s44, s10 // (op 9208) reg expression
s_mov_b32 s45, s11 // (op 9208) reg expression
s_mov_b32 s46, s6 // (op 9208) reg expression
s_mov_b32 s47, 131072 // (op 9208) call()



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
v_accvgpr_write a8, 0.00000 // (op 556) call()
v_accvgpr_write a9, 0.00000 // (op 556) call()
v_accvgpr_write a10, 0.00000 // (op 556) call()
v_accvgpr_write a11, 0.00000 // (op 556) call()
v_accvgpr_write a12, 0.00000 // (op 711) call()
v_accvgpr_write a13, 0.00000 // (op 711) call()
v_accvgpr_write a14, 0.00000 // (op 711) call()
v_accvgpr_write a15, 0.00000 // (op 711) call()
v_accvgpr_write a16, 0.00000 // (op 942) call()
v_accvgpr_write a17, 0.00000 // (op 942) call()
v_accvgpr_write a18, 0.00000 // (op 942) call()
v_accvgpr_write a19, 0.00000 // (op 942) call()
v_accvgpr_write a20, 0.00000 // (op 1097) call()
v_accvgpr_write a21, 0.00000 // (op 1097) call()
v_accvgpr_write a22, 0.00000 // (op 1097) call()
v_accvgpr_write a23, 0.00000 // (op 1097) call()
v_accvgpr_write a24, 0.00000 // (op 1328) call()
v_accvgpr_write a25, 0.00000 // (op 1328) call()
v_accvgpr_write a26, 0.00000 // (op 1328) call()
v_accvgpr_write a27, 0.00000 // (op 1328) call()
v_accvgpr_write a28, 0.00000 // (op 1483) call()
v_accvgpr_write a29, 0.00000 // (op 1483) call()
v_accvgpr_write a30, 0.00000 // (op 1483) call()
v_accvgpr_write a31, 0.00000 // (op 1483) call()
v_accvgpr_write a32, 0.00000 // (op 1714) call()
v_accvgpr_write a33, 0.00000 // (op 1714) call()
v_accvgpr_write a34, 0.00000 // (op 1714) call()
v_accvgpr_write a35, 0.00000 // (op 1714) call()
v_accvgpr_write a36, 0.00000 // (op 1869) call()
v_accvgpr_write a37, 0.00000 // (op 1869) call()
v_accvgpr_write a38, 0.00000 // (op 1869) call()
v_accvgpr_write a39, 0.00000 // (op 1869) call()
v_accvgpr_write a40, 0.00000 // (op 2100) call()
v_accvgpr_write a41, 0.00000 // (op 2100) call()
v_accvgpr_write a42, 0.00000 // (op 2100) call()
v_accvgpr_write a43, 0.00000 // (op 2100) call()
v_accvgpr_write a44, 0.00000 // (op 2255) call()
v_accvgpr_write a45, 0.00000 // (op 2255) call()
v_accvgpr_write a46, 0.00000 // (op 2255) call()
v_accvgpr_write a47, 0.00000 // (op 2255) call()
v_accvgpr_write a48, 0.00000 // (op 2486) call()
v_accvgpr_write a49, 0.00000 // (op 2486) call()
v_accvgpr_write a50, 0.00000 // (op 2486) call()
v_accvgpr_write a51, 0.00000 // (op 2486) call()
v_accvgpr_write a52, 0.00000 // (op 2641) call()
v_accvgpr_write a53, 0.00000 // (op 2641) call()
v_accvgpr_write a54, 0.00000 // (op 2641) call()
v_accvgpr_write a55, 0.00000 // (op 2641) call()
v_accvgpr_write a56, 0.00000 // (op 2872) call()
v_accvgpr_write a57, 0.00000 // (op 2872) call()
v_accvgpr_write a58, 0.00000 // (op 2872) call()
v_accvgpr_write a59, 0.00000 // (op 2872) call()
v_accvgpr_write a60, 0.00000 // (op 3027) call()
v_accvgpr_write a61, 0.00000 // (op 3027) call()
v_accvgpr_write a62, 0.00000 // (op 3027) call()
v_accvgpr_write a63, 0.00000 // (op 3027) call()
v_and_b32 v14, 4294967288, v1 // (op 8711) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 8711) 

v_and_b32 v14, -8, v1 // (op 8711) 
v_sub_u32 v16, v1, v14 // (op 8711) 

v_lshlrev_b32 v14, 4, v16 // (op 8711) 

v_mul_lo_u32 v16, v15, s12 // (op 8711) 
v_add_u32 v16, v16, v14 // (op 8711) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 8713) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8713) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8713) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8713) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8713) most significant: sum
s_add_u32 s1, s1, s7 // (op 8713) most significant: sum


s_mov_b32 s8, s20 // (op 8713) reg expression
s_mov_b32 s9, s21 // (op 8713) reg expression
s_mov_b32 s10, s0 // (op 8713) reg expression
s_mov_b32 s11, 131072 // (op 8713) call()

v_and_b32 v14, 4294967288, v1 // (op 8730) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 8730) 

v_and_b32 v14, -8, v1 // (op 8730) 
v_sub_u32 v17, v1, v14 // (op 8730) 

v_lshlrev_b32 v14, 4, v17 // (op 8730) 

v_mul_lo_u32 v17, v15, s12 // (op 8730) 
v_add_u32 v17, v17, v14 // (op 8730) 


s_mul_i32 s1, s12, s23 // (op 8732) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8732) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8732) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8732) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8732) most significant: sum
s_add_u32 s1, s1, s7 // (op 8732) most significant: sum


s_mov_b32 s8, s20 // (op 8732) reg expression
s_mov_b32 s9, s21 // (op 8732) reg expression
s_mov_b32 s10, s0 // (op 8732) reg expression
s_mov_b32 s11, 131072 // (op 8732) call()

v_and_b32 v14, -2, v1 // (op 8775) 
v_sub_u32 v15, v1, v14 // (op 8775) 

v_lshlrev_b32 v14, 5, v15 // (op 8775) 

v_lshl_add_u32 v15, s17, 6, v14 // (op 8775) 

v_mov_b32 v19, 0 // (op 8775) convert
v_mov_b32 v18, v15 // (op 8775) convert

v_lshrrev_b32 v14, 1, v1 // (op 8775) 
v_mul_lo_u32 v21, v14, s25 // (op 8775) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 8775) most significant: high of low * low
v_mul_lo_u32 v20, v14, s24 // (op 8775) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 8775) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 8775) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 8775) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 8775) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 8775) 

v_mov_b32 v14, v18 // (op 8775) convert

s_mul_i32 s1, s24, s29 // (op 8777) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 8777) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 8777) least significant
s_mul_hi_u32 s7, s24, s28 // (op 8777) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8777) most significant: sum
s_add_u32 s1, s1, s7 // (op 8777) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8777) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8777) 

s_mov_b32 s48, s26 // (op 8777) reg expression
s_mov_b32 s49, s27 // (op 8777) reg expression
s_mov_b32 s50, s0 // (op 8777) reg expression
s_mov_b32 s51, 131072 // (op 8777) call()

v_and_b32 v15, -2, v1 // (op 8794) 
v_sub_u32 v18, v1, v15 // (op 8794) 

v_lshlrev_b32 v15, 5, v18 // (op 8794) 

v_lshl_add_u32 v18, s17, 6, v15 // (op 8794) 

v_mov_b32 v21, 0 // (op 8794) convert
v_mov_b32 v20, v18 // (op 8794) convert

v_lshrrev_b32 v15, 1, v1 // (op 8794) 
v_mul_lo_u32 v19, v15, s25 // (op 8794) most significant: low of low * high
v_mul_hi_u32 v22, v15, s24 // (op 8794) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 8794) least significant: low of low * low
v_add_u32 v19, v19, v22 // (op 8794) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 8794) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 8794) most significant half



v_lshlrev_b64 v[20:21], 2, v[18:19] // (op 8794) 

v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 8794) 

v_mov_b32 v15, v18 // (op 8794) convert

s_mul_i32 s1, s24, s29 // (op 8796) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 8796) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 8796) least significant
s_mul_hi_u32 s7, s24, s28 // (op 8796) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8796) most significant: sum
s_add_u32 s1, s1, s7 // (op 8796) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8796) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8796) 

s_mov_b32 s48, s26 // (op 8796) reg expression
s_mov_b32 s49, s27 // (op 8796) reg expression
s_mov_b32 s50, s0 // (op 8796) reg expression
s_mov_b32 s51, 131072 // (op 8796) call()

v_lshl_add_u32 v18, s16, 8, v1 // (op 8839) 
v_mul_lo_u32 v19, v18, s30 // (op 8839) 

s_mul_i32 s1, s30, s23 // (op 8841) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 8841) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 8841) least significant
s_mul_hi_u32 s7, s30, s22 // (op 8841) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8841) most significant: sum
s_add_u32 s1, s1, s7 // (op 8841) most significant: sum


s_mov_b32 s52, s32 // (op 8841) reg expression
s_mov_b32 s53, s33 // (op 8841) reg expression
s_mov_b32 s54, s0 // (op 8841) reg expression
s_mov_b32 s55, 131072 // (op 8841) call()

v_lshl_add_u32 v18, s16, 8, v1 // (op 8858) 
v_mul_lo_u32 v20, v18, s30 // (op 8858) 

s_mul_i32 s1, s30, s23 // (op 8860) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 8860) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 8860) least significant
s_mul_hi_u32 s7, s30, s22 // (op 8860) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8860) most significant: sum
s_add_u32 s1, s1, s7 // (op 8860) most significant: sum


s_mov_b32 s52, s32 // (op 8860) reg expression
s_mov_b32 s53, s33 // (op 8860) reg expression
s_mov_b32 s54, s0 // (op 8860) reg expression
s_mov_b32 s55, 131072 // (op 8860) call()

v_and_b32 v18, -64, v1 // (op 8903) 
v_sub_u32 v21, v1, v18 // (op 8903) 

v_lshl_add_u32 v18, s17, 6, v21 // (op 8903) 

v_lshrrev_b32 v21, 6, v1 // (op 8903) 
v_mul_lo_u32 v22, v21, s34 // (op 8903) 
v_add_u32 v22, v22, v18 // (op 8903) 


s_mul_i32 s1, s34, s39 // (op 8905) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 8905) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 8905) least significant
s_mul_hi_u32 s7, s34, s38 // (op 8905) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8905) most significant: sum
s_add_u32 s1, s1, s7 // (op 8905) most significant: sum


s_mov_b32 s56, s36 // (op 8905) reg expression
s_mov_b32 s57, s37 // (op 8905) reg expression
s_mov_b32 s58, s0 // (op 8905) reg expression
s_mov_b32 s59, 131072 // (op 8905) call()

v_and_b32 v18, -64, v1 // (op 8922) 
v_sub_u32 v21, v1, v18 // (op 8922) 

v_lshl_add_u32 v18, s17, 6, v21 // (op 8922) 

v_lshrrev_b32 v21, 6, v1 // (op 8922) 
v_mul_lo_u32 v23, v21, s34 // (op 8922) 
v_add_u32 v23, v23, v18 // (op 8922) 


s_mul_i32 s1, s34, s39 // (op 8924) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 8924) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 8924) least significant
s_mul_hi_u32 s7, s34, s38 // (op 8924) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8924) most significant: sum
s_add_u32 s1, s1, s7 // (op 8924) most significant: sum


s_mov_b32 s56, s36 // (op 8924) reg expression
s_mov_b32 s57, s37 // (op 8924) reg expression
s_mov_b32 s58, s0 // (op 8924) reg expression
s_mov_b32 s59, 131072 // (op 8924) call()



v_mov_b32 v18, v14 // (op 8231) reg expression

buffer_load_dwordx4 v[24:27], v18, s[48:51], 0 offen // (op 8231) Load value



v_mov_b32 v18, v22 // (op 8234) reg expression

buffer_load_ubyte v21, v18, s[56:59], 0 offen offset: 0 // (op 8234) Load value


v_mov_b64 v[28:29], s[12:13] // (op 8237) convert

v_mov_b32 v18, v16 // (op 8237) 
buffer_load_dwordx4 v[30:33], v18, s[8:11], 0 offen // (op 8237) Load value
v_add_u32 v18, v18, v28 // (op 8237) 
buffer_load_dwordx4 v[34:37], v18, s[8:11], 0 offen // (op 8237) Load value
v_add_u32 v18, v18, v28 // (op 8237) 
buffer_load_dwordx4 v[38:41], v18, s[8:11], 0 offen // (op 8237) Load value
v_add_u32 v18, v18, v28 // (op 8237) 
buffer_load_dwordx4 v[42:45], v18, s[8:11], 0 offen // (op 8237) Load value
v_add_u32 v18, v18, v28 // (op 8237) 
buffer_load_dwordx4 v[46:49], v18, s[8:11], 0 offen // (op 8237) Load value
v_add_u32 v18, v18, v28 // (op 8237) 
buffer_load_dwordx4 v[50:53], v18, s[8:11], 0 offen // (op 8237) Load value
v_add_u32 v18, v18, v28 // (op 8237) 
buffer_load_dwordx4 v[54:57], v18, s[8:11], 0 offen // (op 8237) Load value
v_add_u32 v18, v18, v28 // (op 8237) 
buffer_load_dwordx4 v[58:61], v18, s[8:11], 0 offen // (op 8237) Load value





buffer_load_dword v18, v19, s[52:55], 0 offen // (op 8240) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 8243) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 8243) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 8243) 

v_add_u32 v28, v14, s0 // (op 8243) 


buffer_load_dwordx4 v[62:65], v28, s[48:51], 0 offen // (op 8243) Load value




s_lshl_b32 s0, s34, 2 // (op 8246) 
v_add_u32 v14, v22, s0 // (op 8246) 


buffer_load_ubyte v28, v14, s[56:59], 0 offen offset: 0 // (op 8246) Load value



v_mov_b64 v[66:67], s[12:13] // (op 8249) convert

v_mov_b32 v14, v16 // (op 8249) 
buffer_load_dwordx4 v[68:71], v14, s[8:11], 0 offen offset:128 // (op 8249) Load value
v_add_u32 v14, v14, v66 // (op 8249) 
buffer_load_dwordx4 v[72:75], v14, s[8:11], 0 offen offset:128 // (op 8249) Load value
v_add_u32 v14, v14, v66 // (op 8249) 
buffer_load_dwordx4 v[76:79], v14, s[8:11], 0 offen offset:128 // (op 8249) Load value
v_add_u32 v14, v14, v66 // (op 8249) 
buffer_load_dwordx4 v[80:83], v14, s[8:11], 0 offen offset:128 // (op 8249) Load value
v_add_u32 v14, v14, v66 // (op 8249) 
buffer_load_dwordx4 v[84:87], v14, s[8:11], 0 offen offset:128 // (op 8249) Load value
v_add_u32 v14, v14, v66 // (op 8249) 
buffer_load_dwordx4 v[88:91], v14, s[8:11], 0 offen offset:128 // (op 8249) Load value
v_add_u32 v14, v14, v66 // (op 8249) 
buffer_load_dwordx4 v[92:95], v14, s[8:11], 0 offen offset:128 // (op 8249) Load value
v_add_u32 v14, v14, v66 // (op 8249) 
buffer_load_dwordx4 v[96:99], v14, s[8:11], 0 offen offset:128 // (op 8249) Load value






buffer_load_dword v14, v19, s[52:55], 0 offen offset:4 // (op 8252) Load value

v_and_b32 v16, -2, v1 // (op 9052) 
v_sub_u32 v19, v1, v16 // (op 9052) 

v_lshlrev_b32 v16, 5, v19 // (op 9052) 

v_lshrrev_b32 v19, 1, v1 // (op 9052) 
v_lshl_add_u32 v22, v19, 6, v16 // (op 9052) 


v_lshlrev_b32 v16, 2, v22 // (op 9052) 

v_lshrrev_b32 v19, 3, v16 // (op 9052) 




s_waitcnt vmcnt(21)
ds_write_b128 v19, v[24:27] // (op 8254) Store local data 

v_and_b32 v16, -64, v1 // (op 9168) 
v_sub_u32 v19, v1, v16 // (op 9168) 

v_and_b32 v16, 4294967232, v1 // (op 9168) 
v_add_u32 v22, v16, v19 // (op 9168) 





s_waitcnt vmcnt(20)
ds_write_b8 v22, v21 offset:4096 // (op 8255) Store local data 

v_and_b32 v16, -8, v1 // (op 8994) 
v_sub_u32 v19, v1, v16 // (op 8994) 

v_lshlrev_b32 v16, 4, v19 // (op 8994) 

v_lshrrev_b32 v19, 3, v1 // (op 8994) 
v_lshl_add_u32 v22, v19, 10, v16 // (op 8994) 





s_waitcnt vmcnt(19)
ds_write_b128 v22, v[30:33] offset:4352 // (op 8256) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v22, v[34:37] offset:4480 // (op 8256) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v22, v[38:41] offset:4608 // (op 8256) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v22, v[42:45] offset:4736 // (op 8256) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v22, v[46:49] offset:4864 // (op 8256) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v22, v[50:53] offset:4992 // (op 8256) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v22, v[54:57] offset:5120 // (op 8256) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v22, v[58:61] offset:5248 // (op 8256) Store local data 

v_lshlrev_b32 v16, 2, v1 // (op 9110) 



s_waitcnt vmcnt(11)
ds_write_b32 v16, v18 offset:37120 // (op 8257) Store local data 

v_and_b32 v16, -8, v1 // (op 8978) 
v_sub_u32 v19, v1, v16 // (op 8978) 

v_lshlrev_b32 v16, 4, v19 // (op 8978) 

v_lshrrev_b32 v19, 3, v1 // (op 8978) 
v_lshl_add_u32 v22, v19, 10, v16 // (op 8978) 


v_and_b32 v16, -2, v1 // (op 9036) 
v_sub_u32 v19, v1, v16 // (op 9036) 

v_lshlrev_b32 v16, 5, v19 // (op 9036) 

v_lshrrev_b32 v19, 1, v1 // (op 9036) 
v_lshl_add_u32 v29, v19, 6, v16 // (op 9036) 


v_lshlrev_b32 v16, 2, v29 // (op 9036) 

v_lshrrev_b32 v19, 3, v16 // (op 9036) 

v_lshlrev_b32 v16, 2, v1 // (op 9094) 
v_and_b32 v29, -64, v1 // (op 9152) 
v_sub_u32 v66, v1, v29 // (op 9152) 

v_and_b32 v29, 4294967232, v1 // (op 9152) 
v_add_u32 v67, v29, v66 // (op 9152) 


v_and_b32 v29, -8, v1 // (op 9220) 
v_sub_u32 v66, v1, v29 // (op 9220) 

v_lshlrev_b32 v29, 4, v66 // (op 9220) 

v_lshrrev_b32 v66, 3, v1 // (op 9220) 
v_lshl_add_u32 v100, v66, 10, v29 // (op 9220) 


v_and_b32 v29, -2, v1 // (op 9249) 
v_sub_u32 v66, v1, v29 // (op 9249) 

v_lshlrev_b32 v29, 5, v66 // (op 9249) 

v_lshrrev_b32 v66, 1, v1 // (op 9249) 
v_lshl_add_u32 v101, v66, 6, v29 // (op 9249) 


v_lshlrev_b32 v29, 2, v101 // (op 9249) 

v_lshrrev_b32 v66, 3, v29 // (op 9249) 

v_lshlrev_b32 v29, 2, v1 // (op 9278) 
v_and_b32 v101, -64, v1 // (op 9307) 
v_sub_u32 v102, v1, v101 // (op 9307) 

v_and_b32 v101, 4294967232, v1 // (op 9307) 
v_add_u32 v103, v101, v102 // (op 9307) 


s_mov_b32 s0, 0 // (op 2905) call()
s_ashr_i64 s[6:7], s[28:29], 7 // (op 2904) 
s_mov_b32 s1, s6 // (op 2904) convert

s_and_b32 s6, s1, 4294967294 // (op 2904) 

s_cmp_lt_u32 s0, s6 // (op 2904) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_2904 // (op 2904) Condition: Top (jump to RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_2904 if false)
RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_2904:


s_barrier  // (op 9411) 
s_add_u32 s0, s0, 2 // (op 2906) 



v_mov_b32 v101, v15 // (op 10) 
s_lshl_b64 s[6:7], s[24:25], 7 // (op 10) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 10) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 10) 

s_lshl_b32 s1, s6, 1 // (op 10) 

v_add_u32 v102, v101, s1 // (op 10) 


buffer_load_dwordx4 v[24:27], v102, s[48:51], 0 offen // (op 10) Load value

ds_read_b128 v[104:107], v3 offset:4352 // (op 70) Load local data 


ds_read_b128 v[108:111], v3 offset:4416 // (op 70) Load local data 
v_mov_b32 v102, v23 // (op 25) 
s_barrier  // (op 9424) 
s_lshl_b32 s1, s34, 3 // (op 25) 



v_add_u32 v112, v102, s1 // (op 25) 


buffer_load_ubyte v21, v112, s[56:59], 0 offen offset: 0 // (op 25) Load value

ds_read_b64_tr_b4 v[112:113], v4 // (op 75) Transpose load local data 

v_mov_b64 v[116:117], s[12:13] // (op 4) convert

v_mov_b32 v118, v17 // (op 4) 
v_mov_b32 v119, v118 // (op 4) 
ds_read_b64_tr_b4 v[114:115], v4 offset:512 // (op 75) Transpose load local data 
buffer_load_dwordx4 v[30:33], v119, s[8:11], 0 offen offset:256 // (op 4) Load value
s_barrier  // (op 9438) 
v_add_u32 v119, v119, v116 // (op 4) 



ds_read_u8 v120, v0 offset:37120 // (op 80) Load local data 
buffer_load_dwordx4 v[34:37], v119, s[8:11], 0 offen offset:256 // (op 4) Load value
s_barrier  // (op 9452) 
v_add_u32 v119, v119, v116 // (op 4) 


ds_read_u8 v121, v5 offset:4096 // (op 85) Load local data 
buffer_load_dwordx4 v[38:41], v119, s[8:11], 0 offen offset:256 // (op 4) Load value



v_add_u32 v119, v119, v116 // (op 4) 
ds_read_b64_tr_b4 v[122:123], v4 offset:16 // (op 371) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v4 offset:528 // (op 371) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[112:115], a[0:3], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 88) 


ds_read_u8 v126, v5 offset:4128 // (op 375) Load local data 
buffer_load_dwordx4 v[42:45], v119, s[8:11], 0 offen offset:256 // (op 4) Load value



v_add_u32 v119, v119, v116 // (op 4) 
ds_read_b128 v[128:131], v3 offset:8448 // (op 622) Load local data 
ds_read_b128 v[132:135], v3 offset:8512 // (op 622) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[122:125], a[4:7], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 376) 





ds_read_u8 v120, v0 offset:37248 // (op 625) Load local data 
buffer_load_dwordx4 v[46:49], v119, s[8:11], 0 offen offset:256 // (op 4) Load value



v_add_u32 v119, v119, v116 // (op 4) 
ds_read_b128 v[104:107], v3 offset:12544 // (op 1008) Load local data 
ds_read_b128 v[108:111], v3 offset:12608 // (op 1008) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[128:135], v[112:115], a[8:11], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 673) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[128:135], v[122:125], a[12:15], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 627) 
buffer_load_dwordx4 v[50:53], v119, s[8:11], 0 offen offset:256 // (op 4) Load value


v_add_u32 v119, v119, v116 // (op 4) 
ds_read_u8 v120, v0 offset:37376 // (op 1011) Load local data 
buffer_load_dwordx4 v[54:57], v119, s[8:11], 0 offen offset:256 // (op 4) Load value



v_add_u32 v119, v119, v116 // (op 4) 
ds_read_b128 v[128:131], v3 offset:16640 // (op 1394) Load local data 
ds_read_b128 v[132:135], v3 offset:16704 // (op 1394) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:111], v[112:115], a[16:19], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1059) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:111], v[122:125], a[20:23], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1013) 
ds_read_u8 v127, v0 offset:37504 // (op 1397) Load local data 





s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:135], v[112:115], a[24:27], v127, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1445) 
ds_read_b128 v[104:107], v3 offset:20736 // (op 1780) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:135], v[122:125], a[28:31], v127, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1399) 
ds_read_b128 v[108:111], v3 offset:20800 // (op 1780) Load local data 





ds_read_u8 v120, v0 offset:37632 // (op 1783) Load local data 
buffer_load_dwordx4 v[58:61], v119, s[8:11], 0 offen offset:256 // (op 4) Load value







v_mov_b32 v119, v20 // (op 17) 

ds_read_b128 v[128:131], v3 offset:24832 // (op 2166) Load local data 
ds_read_b128 v[132:135], v3 offset:24896 // (op 2166) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:111], v[112:115], a[32:35], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1831) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:111], v[122:125], a[36:39], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1785) 
buffer_load_dword v18, v119, s[52:55], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 9475) 
ds_read_u8 v120, v0 offset:37760 // (op 2169) Load local data 






s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[128:135], v[112:115], a[40:43], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2217) 
ds_read_b128 v[104:107], v3 offset:28928 // (op 2552) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[128:135], v[122:125], a[44:47], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2171) 
ds_read_b128 v[108:111], v3 offset:28992 // (op 2552) Load local data 





ds_read_u8 v120, v0 offset:37888 // (op 2555) Load local data 
s_waitcnt vmcnt(21)
ds_write_b128 v66, v[62:65] offset:38144 // (op 253) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:111], v[112:115], a[48:51], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2603) 
s_barrier  // (op 9487) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:111], v[122:125], a[52:55], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2557) 





ds_read_b128 v[104:107], v3 offset:33024 // (op 2938) Load local data 
ds_read_b128 v[108:111], v3 offset:33088 // (op 2938) Load local data 
s_waitcnt vmcnt(20)
ds_write_b8 v103, v28 offset:42240 // (op 255) Store local data 



ds_read_u8 v120, v0 offset:38016 // (op 2941) Load local data 
s_barrier  // (op 9469) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:111], v[112:115], a[56:59], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2989) 





s_waitcnt vmcnt(19)
ds_write_b128 v100, v[68:71] offset:42496 // (op 243) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:111], v[122:125], a[60:63], v120, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2943) 
s_waitcnt vmcnt(18)
ds_write_b128 v100, v[72:75] offset:42624 // (op 243) Store local data 




s_waitcnt vmcnt(17)
ds_write_b128 v100, v[76:79] offset:42752 // (op 243) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v100, v[80:83] offset:42880 // (op 243) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v100, v[84:87] offset:43008 // (op 243) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v100, v[88:91] offset:43136 // (op 243) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v100, v[92:95] offset:43264 // (op 243) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v100, v[96:99] offset:43392 // (op 243) Store local data 
s_barrier  // (op 9481) 



v_add_u32 v104, 75264, v29 // (op 254) 
s_waitcnt vmcnt(11)
ds_write_b32 v104, v14 // (op 254) Store local data 



s_lshl_b64 s[6:7], s[24:25], 7 // (op 256) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 256) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 256) 

s_mul_i32 s1, 3, s6 // (op 256) 

v_add_u32 v104, v101, s1 // (op 256) 


buffer_load_dwordx4 v[62:65], v104, s[48:51], 0 offen // (op 256) Load value

s_waitcnt lgkmcnt(1)
s_barrier  // (op 9466) 






s_lshl_b32 s1, s34, 2 // (op 258) 
s_mul_i32 s6, 3, s1 // (op 258) 

v_add_u32 v101, v102, s6 // (op 258) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 8786) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 8786) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 8786) 

s_lshl_b32 s1, s6, 1 // (op 8786) 

v_add_u32 v15, v15, s1 // (op 8786) 

ds_read_b128 v[104:107], v7 offset:42496 // (op 245) Load local data 
ds_read_b128 v[108:111], v7 offset:42560 // (op 245) Load local data 
buffer_load_ubyte v28, v101, s[56:59], 0 offen offset: 0 // (op 258) Load value


s_barrier  // (op 9472) 




s_lshl_b32 s1, s34, 3 // (op 8914) 
v_add_u32 v23, v23, s1 // (op 8914) 

v_mov_b64 v[112:113], s[12:13] // (op 242) convert

v_mov_b32 v101, v118 // (op 242) 
ds_read_b64_tr_b4 v[114:115], v8 offset:38144 // (op 247) Transpose load local data 
buffer_load_dwordx4 v[68:71], v101, s[8:11], 0 offen offset:384 // (op 242) Load value
v_add_u32 v101, v101, v112 // (op 242) 
ds_read_b64_tr_b4 v[116:117], v8 offset:38656 // (op 247) Transpose load local data 
buffer_load_dwordx4 v[72:75], v101, s[8:11], 0 offen offset:384 // (op 242) Load value
v_add_u32 v101, v101, v112 // (op 242) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 9478) 
buffer_load_dwordx4 v[76:79], v101, s[8:11], 0 offen offset:384 // (op 242) Load value



v_add_u32 v101, v101, v112 // (op 242) 
v_add_u32 v102, 75264, v6 // (op 249) 
ds_read_u8 v120, v102 // (op 249) Load local data 

buffer_load_dwordx4 v[80:83], v101, s[8:11], 0 offen offset:384 // (op 242) Load value
s_barrier  // (op 9484) 
v_add_u32 v101, v101, v112 // (op 242) 


ds_read_u8 v102, v9 offset:42240 // (op 251) Load local data 
buffer_load_dwordx4 v[84:87], v101, s[8:11], 0 offen offset:384 // (op 242) Load value



v_add_u32 v101, v101, v112 // (op 242) 
ds_read_b64_tr_b4 v[122:123], v8 offset:38160 // (op 381) Transpose load local data 
ds_read_b64_tr_b4 v[124:125], v8 offset:38672 // (op 381) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[114:117], a[0:3], v120, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 252) 
buffer_load_dwordx4 v[88:91], v101, s[8:11], 0 offen offset:384 // (op 242) Load value


s_barrier  // (op 9544) 
v_add_u32 v101, v101, v112 // (op 242) 
ds_read_u8 v121, v9 offset:42272 // (op 385) Load local data 
buffer_load_dwordx4 v[92:95], v101, s[8:11], 0 offen offset:384 // (op 242) Load value
v_add_u32 v101, v101, v112 // (op 242) 



s_barrier  // (op 9548) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:111], v[122:125], a[4:7], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 377) 
ds_read_b128 v[126:129], v7 offset:46592 // (op 630) Load local data 


ds_read_b128 v[130:133], v7 offset:46656 // (op 630) Load local data 
buffer_load_dwordx4 v[96:99], v101, s[8:11], 0 offen offset:384 // (op 242) Load value






v_add_u32 v101, 75392, v6 // (op 633) 



v_add_u32 v17, 256, v17 // (op 8722) 
ds_read_u8 v118, v101 // (op 633) Load local data 

buffer_load_dword v14, v119, s[52:55], 0 offen offset:12 // (op 257) Load value




ds_read_b128 v[104:107], v7 offset:50688 // (op 1016) Load local data 
s_barrier  // (op 9427) 
ds_read_b128 v[108:111], v7 offset:50752 // (op 1016) Load local data 






v_add_u32 v20, 8, v20 // (op 8850) 
v_add_u32 v101, 75520, v6 // (op 1019) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[126:133], v[114:117], a[8:11], v118, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 674) 
ds_read_u8 v112, v101 // (op 1019) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[126:133], v[122:125], a[12:15], v118, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 628) 
ds_read_b128 v[134:137], v7 offset:54784 // (op 1402) Load local data 


ds_read_b128 v[138:141], v7 offset:54848 // (op 1402) Load local data 



v_add_u32 v101, 75648, v6 // (op 1405) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:111], v[114:117], a[16:19], v112, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1060) 
ds_read_u8 v113, v101 // (op 1405) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:111], v[122:125], a[20:23], v112, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1014) 
ds_read_b128 v[126:129], v7 offset:58880 // (op 1788) Load local data 


ds_read_b128 v[130:133], v7 offset:58944 // (op 1788) Load local data 



v_add_u32 v101, 75776, v6 // (op 1791) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[134:141], v[114:117], a[24:27], v113, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1446) 
ds_read_u8 v104, v101 // (op 1791) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[134:141], v[122:125], a[28:31], v113, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1400) 
ds_read_b128 v[142:145], v7 offset:62976 // (op 2174) Load local data 


ds_read_b128 v[146:149], v7 offset:63040 // (op 2174) Load local data 



v_add_u32 v101, 75904, v6 // (op 2177) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[126:133], v[114:117], a[32:35], v104, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1832) 
ds_read_u8 v105, v101 // (op 2177) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[126:133], v[122:125], a[36:39], v104, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1786) 
v_add_u32 v101, 67072, v7 // (op 2560) 


ds_read_b128 v[106:109], v101 // (op 2560) Load local data 

v_add_u32 v101, 67136, v7 // (op 2560) 
ds_read_b128 v[110:113], v101 // (op 2560) Load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[142:149], v[114:117], a[40:43], v105, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2218) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[142:149], v[122:125], a[44:47], v105, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2172) 
v_add_u32 v101, 76032, v6 // (op 2563) 


ds_read_u8 v104, v101 // (op 2563) Load local data 

s_waitcnt vmcnt(21)
ds_write_b128 v19, v[24:27] // (op 13) Store local data 



v_add_u32 v101, 71168, v7 // (op 2946) 
s_barrier  // (op 9455) 
ds_read_b128 v[126:129], v101 // (op 2946) Load local data 




v_add_u32 v101, 71232, v7 // (op 2946) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[106:113], v[114:117], a[48:51], v104, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2604) 
ds_read_b128 v[130:133], v101 // (op 2946) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[106:113], v[122:125], a[52:55], v104, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2558) 





s_barrier  // (op 9604) 
v_add_u32 v101, 76160, v6 // (op 2949) 
ds_read_u8 v104, v101 // (op 2949) Load local data 

s_waitcnt vmcnt(20)
ds_write_b8 v67, v21 offset:4096 // (op 30) Store local data 
s_barrier  // (op 9608) 
s_barrier  // (op 9414) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[126:133], v[114:117], a[56:59], v104, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2990) 





s_waitcnt vmcnt(19)
ds_write_b128 v22, v[30:33] offset:4352 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[126:133], v[122:125], a[60:63], v104, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2944) 
s_waitcnt vmcnt(18)
ds_write_b128 v22, v[34:37] offset:4480 // (op 6) Store local data 




s_waitcnt vmcnt(17)
ds_write_b128 v22, v[38:41] offset:4608 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v22, v[42:45] offset:4736 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v22, v[46:49] offset:4864 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v22, v[50:53] offset:4992 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v22, v[54:57] offset:5120 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v22, v[58:61] offset:5248 // (op 6) Store local data 
s_barrier  // (op 9441) 



s_waitcnt vmcnt(11)
ds_write_b32 v16, v18 offset:37120 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[28:29], 7 // (op 2904) 
s_mov_b32 s1, s6 // (op 2904) convert

s_and_b32 s6, s1, 4294967294 // (op 2904) 

s_cmp_lt_u32 s0, s6 // (op 2904) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_2904 // (op 2904) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoop_2904 if true)
RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_2904:

















v_and_b32 v15, 4294967288, v1 // (op 8670) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8670) 
v_lshl_add_u32 v16, s16, 8, v15 // (op 8670) 

s_lshr_b32 s6, s0, 1 // (op 8670) 

v_and_b32 v15, -8, v1 // (op 8670) 
v_sub_u32 v17, v1, v15 // (op 8670) 

v_lshlrev_b32 v15, 4, v17 // (op 8670) 

v_lshl_add_u32 v17, s6, 8, v15 // (op 8670) 


v_mul_lo_u32 v15, v16, s12 // (op 8670) 
v_add_u32 v15, v15, v17 // (op 8670) 


s_mul_i32 s1, s12, s23 // (op 8672) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8672) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8672) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8672) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8672) most significant: sum
s_add_u32 s1, s1, s7 // (op 8672) most significant: sum


s_mov_b32 s8, s20 // (op 8672) reg expression
s_mov_b32 s9, s21 // (op 8672) reg expression
s_mov_b32 s10, s0 // (op 8672) reg expression
s_mov_b32 s11, 131072 // (op 8672) call()

v_lshrrev_b32 v16, 1, v1 // (op 8752) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8752) 
v_and_b32 v17, -2, v1 // (op 8752) 
s_lshr_b32 s6, s0, 1 // (op 8752) 

v_sub_u32 v18, v1, v17 // (op 8752) 

v_lshl_add_u32 v17, s6, 8, v16 // (op 8752) 


v_lshlrev_b32 v16, 5, v18 // (op 8752) 

v_lshl_add_u32 v18, s17, 6, v16 // (op 8752) 

v_mov_b32 v67, 0 // (op 8752) convert
v_mov_b32 v66, v18 // (op 8752) convert

v_mul_lo_u32 v19, v17, s25 // (op 8752) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 8752) most significant: high of low * low
v_mul_lo_u32 v18, v17, s24 // (op 8752) least significant: low of low * low
v_add_u32 v19, v19, v16 // (op 8752) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v66 // (op 8752) least significant half
v_addc_co_u32 v19, s[0:1], v19, v67, s[0:1] // (op 8752) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 8752) 

v_lshrrev_b64 v[66:67], 3, v[16:17] // (op 8752) 

v_mov_b32 v16, v66 // (op 8752) convert

s_mul_i32 s1, s24, s29 // (op 8754) most significant: low of low * high
s_mul_i32 s6, s25, s28 // (op 8754) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 8754) least significant
s_mul_hi_u32 s7, s24, s28 // (op 8754) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8754) most significant: sum
s_add_u32 s1, s1, s7 // (op 8754) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8754) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8754) 

s_mov_b32 s48, s26 // (op 8754) reg expression
s_mov_b32 s49, s27 // (op 8754) reg expression
s_mov_b32 s50, s0 // (op 8754) reg expression
s_mov_b32 s51, 131072 // (op 8754) call()

v_lshl_add_u32 v17, s16, 8, v1 // (op 8816) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8816) 
s_lshr_b32 s6, s0, 1 // (op 8816) 

s_lshl_b32 s7, s6, 3 // (op 8816) 

v_mul_lo_u32 v18, v17, s30 // (op 8816) 
v_add_u32 v18, v18, s7 // (op 8816) 


s_mul_i32 s1, s30, s23 // (op 8818) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 8818) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 8818) least significant
s_mul_hi_u32 s7, s30, s22 // (op 8818) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8818) most significant: sum
s_add_u32 s1, s1, s7 // (op 8818) most significant: sum


s_mov_b32 s52, s32 // (op 8818) reg expression
s_mov_b32 s53, s33 // (op 8818) reg expression
s_mov_b32 s54, s0 // (op 8818) reg expression
s_mov_b32 s55, 131072 // (op 8818) call()

v_lshrrev_b32 v17, 6, v1 // (op 8880) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8880) 
v_and_b32 v19, -64, v1 // (op 8880) 
s_lshr_b32 s6, s0, 1 // (op 8880) 

v_sub_u32 v20, v1, v19 // (op 8880) 

v_lshl_add_u32 v19, s6, 3, v17 // (op 8880) 


v_lshl_add_u32 v17, s17, 6, v20 // (op 8880) 

v_mul_lo_u32 v20, v19, s34 // (op 8880) 
v_add_u32 v20, v20, v17 // (op 8880) 


s_mul_i32 s1, s34, s39 // (op 8882) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 8882) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 8882) least significant
s_mul_hi_u32 s7, s34, s38 // (op 8882) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8882) most significant: sum
s_add_u32 s1, s1, s7 // (op 8882) most significant: sum


s_mov_b32 s56, s36 // (op 8882) reg expression
s_mov_b32 s57, s37 // (op 8882) reg expression
s_mov_b32 s58, s0 // (op 8882) reg expression
s_mov_b32 s59, 131072 // (op 8882) call()

v_and_b32 v17, -8, v1 // (op 8963) 
v_sub_u32 v19, v1, v17 // (op 8963) 

v_lshlrev_b32 v17, 4, v19 // (op 8963) 

v_lshrrev_b32 v19, 3, v1 // (op 8963) 
v_lshl_add_u32 v21, v19, 10, v17 // (op 8963) 


v_and_b32 v17, -2, v1 // (op 9023) 
v_sub_u32 v19, v1, v17 // (op 9023) 

v_lshlrev_b32 v17, 5, v19 // (op 9023) 

v_lshrrev_b32 v19, 1, v1 // (op 9023) 
v_lshl_add_u32 v22, v19, 6, v17 // (op 9023) 


v_lshlrev_b32 v17, 2, v22 // (op 9023) 

v_lshrrev_b32 v19, 3, v17 // (op 9023) 

v_lshlrev_b32 v17, 2, v1 // (op 9081) 
v_and_b32 v22, -64, v1 // (op 9139) 
v_sub_u32 v23, v1, v22 // (op 9139) 

v_and_b32 v22, 4294967232, v1 // (op 9139) 
v_add_u32 v24, v22, v23 // (op 9139) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2879) 
s_and_b32 s6, s0, 4294967294 // (op 2879) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 2878) 
s_mov_b32 s7, s0 // (op 2878) convert

s_cmp_lt_u32 s6, s7 // (op 2878) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_2878 // (op 2878) Condition: Top (jump to RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_2878 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_2878:

s_add_u32 s6, s6, 1 // (op 2880) 
v_mov_b64 v[22:23], s[12:13] // (op 176) convert

v_mov_b32 v14, v15 // (op 176) 
v_mov_b32 v25, v14 // (op 176) 
buffer_load_dwordx4 v[26:29], v25, s[8:11], 0 offen // (op 176) Load value
v_add_u32 v25, v25, v22 // (op 176) 
buffer_load_dwordx4 v[30:33], v25, s[8:11], 0 offen // (op 176) Load value
v_add_u32 v25, v25, v22 // (op 176) 
buffer_load_dwordx4 v[34:37], v25, s[8:11], 0 offen // (op 176) Load value
v_add_u32 v25, v25, v22 // (op 176) 
buffer_load_dwordx4 v[38:41], v25, s[8:11], 0 offen // (op 176) Load value
v_add_u32 v25, v25, v22 // (op 176) 
buffer_load_dwordx4 v[42:45], v25, s[8:11], 0 offen // (op 176) Load value
v_add_u32 v25, v25, v22 // (op 176) 
buffer_load_dwordx4 v[46:49], v25, s[8:11], 0 offen // (op 176) Load value
v_add_u32 v25, v25, v22 // (op 176) 
buffer_load_dwordx4 v[50:53], v25, s[8:11], 0 offen // (op 176) Load value
v_add_u32 v25, v25, v22 // (op 176) 
buffer_load_dwordx4 v[54:57], v25, s[8:11], 0 offen // (op 176) Load value



v_add_u32 v15, 128, v15 // (op 8660) 


s_barrier  // (op 9417) 
v_mov_b32 v14, v16 // (op 190) 



v_mov_b32 v25, v14 // (op 190) reg expression

buffer_load_dwordx4 v[58:61], v25, s[48:51], 0 offen // (op 190) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 8744) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 8744) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 8744) 

v_add_u32 v16, v16, s0 // (op 8744) 

s_waitcnt vmcnt(8)
ds_write_b128 v21, v[26:29] offset:4352 // (op 177) Store local data 


v_mov_b32 v14, v18 // (op 191) 

s_waitcnt vmcnt(7)
ds_write_b128 v21, v[30:33] offset:4480 // (op 177) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v21, v[34:37] offset:4608 // (op 177) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v21, v[38:41] offset:4736 // (op 177) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v21, v[42:45] offset:4864 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v21, v[46:49] offset:4992 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[50:53] offset:5120 // (op 177) Store local data 
buffer_load_dword v25, v14, s[52:55], 0 offen // (op 191) Load value

v_add_u32 v18, 4, v18 // (op 8808) 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[54:57] offset:5248 // (op 177) Store local data 



v_mov_b32 v14, v20 // (op 192) 
s_barrier  // (op 9430) 
v_mov_b32 v22, v14 // (op 192) reg expression




buffer_load_ubyte v23, v22, s[56:59], 0 offen offset: 0 // (op 192) Load value


s_lshl_b32 s7, s34, 2 // (op 8872) 
v_add_u32 v20, v20, s7 // (op 8872) 

s_waitcnt vmcnt(2)
ds_write_b128 v19, v[58:61] // (op 187) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 9420) 



s_barrier  // (op 9444) 
ds_read_b128 v[26:29], v3 offset:4352 // (op 179) Load local data 



ds_read_b128 v[30:33], v3 offset:4416 // (op 179) Load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v17, v25 offset:37120 // (op 188) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9434) 



s_barrier  // (op 9458) 
ds_read_b64_tr_b4 v[34:35], v4 // (op 181) Transpose load local data 



ds_read_b64_tr_b4 v[36:37], v4 offset:512 // (op 181) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v23 offset:4096 // (op 189) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9448) 



ds_read_u8 v14, v0 offset:37120 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9462) 


ds_read_u8 v25, v5 offset:4096 // (op 185) Load local data 



ds_read_b64_tr_b4 v[38:39], v4 offset:16 // (op 350) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:37], a[0:3], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 186) 
ds_read_b64_tr_b4 v[40:41], v4 offset:528 // (op 350) Transpose load local data 


ds_read_u8 v22, v5 offset:4128 // (op 354) Load local data 



ds_read_b128 v[42:45], v3 offset:8448 // (op 568) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[38:41], a[4:7], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 355) 
ds_read_b128 v[46:49], v3 offset:8512 // (op 568) Load local data 





ds_read_u8 v14, v0 offset:37248 // (op 571) Load local data 



ds_read_b128 v[26:29], v3 offset:12544 // (op 954) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:49], v[34:37], a[8:11], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 573) 
ds_read_b128 v[30:33], v3 offset:12608 // (op 954) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[38:41], a[12:15], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 604) 
ds_read_u8 v23, v0 offset:37376 // (op 957) Load local data 





ds_read_b128 v[42:45], v3 offset:16640 // (op 1340) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:33], v[34:37], a[16:19], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 959) 
ds_read_b128 v[46:49], v3 offset:16704 // (op 1340) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:33], v[38:41], a[20:23], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 990) 
ds_read_u8 v14, v0 offset:37504 // (op 1343) Load local data 





ds_read_b128 v[26:29], v3 offset:20736 // (op 1726) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:49], v[34:37], a[24:27], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1345) 
ds_read_b128 v[30:33], v3 offset:20800 // (op 1726) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:49], v[38:41], a[28:31], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1376) 
ds_read_u8 v23, v0 offset:37632 // (op 1729) Load local data 





ds_read_b128 v[42:45], v3 offset:24832 // (op 2112) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:33], v[34:37], a[32:35], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1731) 
ds_read_b128 v[46:49], v3 offset:24896 // (op 2112) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[26:33], v[38:41], a[36:39], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1762) 
ds_read_u8 v14, v0 offset:37760 // (op 2115) Load local data 





ds_read_b128 v[26:29], v3 offset:28928 // (op 2498) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[42:49], v[34:37], a[40:43], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2117) 
ds_read_b128 v[30:33], v3 offset:28992 // (op 2498) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[42:49], v[38:41], a[44:47], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2148) 
ds_read_u8 v23, v0 offset:37888 // (op 2501) Load local data 





ds_read_b128 v[42:45], v3 offset:33024 // (op 2884) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[26:33], v[34:37], a[48:51], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2503) 
ds_read_b128 v[46:49], v3 offset:33088 // (op 2884) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[26:33], v[38:41], a[52:55], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2534) 
ds_read_u8 v14, v0 offset:38016 // (op 2887) Load local data 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[42:49], v[34:37], a[56:59], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2889) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[42:49], v[38:41], a[60:63], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2920) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 2878) 
s_mov_b32 s7, s0 // (op 2878) convert

s_cmp_lt_u32 s6, s7 // (op 2878) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_2878 // (op 2878) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_2878 if true)
RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_2878:
























s_barrier  // (op 9588) 








v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_133
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


v_accvgpr_read v6, a4 // (op 3573) 
v_accvgpr_read v7, a5 // (op 3573) 
v_accvgpr_read v8, a6 // (op 3573) 
v_accvgpr_read v9, a7 // (op 3573) 
v_mul_f32 v14, v11, v6 // (op 3573) 
v_mul_f32 v15, v11, v7 // (op 3573) 
v_mul_f32 v16, v11, v8 // (op 3573) 
v_mul_f32 v17, v11, v9 // (op 3573) 



s_lshl_b32 s0, s4, 7 // (op 3575) 
v_add_u32 v0, v13, s0 // (op 3575) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3575) Store value


v_accvgpr_read v6, a8 // (op 3597) 
v_accvgpr_read v7, a9 // (op 3597) 
v_accvgpr_read v8, a10 // (op 3597) 
v_accvgpr_read v9, a11 // (op 3597) 
v_mul_f32 v14, v11, v6 // (op 3597) 
v_mul_f32 v15, v11, v7 // (op 3597) 
v_mul_f32 v16, v11, v8 // (op 3597) 
v_mul_f32 v17, v11, v9 // (op 3597) 



v_add_u32 v0, 128, v13 // (op 3600) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3600) Store value


v_accvgpr_read v6, a12 // (op 3606) 
v_accvgpr_read v7, a13 // (op 3606) 
v_accvgpr_read v8, a14 // (op 3606) 
v_accvgpr_read v9, a15 // (op 3606) 
v_mul_f32 v14, v11, v6 // (op 3606) 
v_mul_f32 v15, v11, v7 // (op 3606) 
v_mul_f32 v16, v11, v8 // (op 3606) 
v_mul_f32 v17, v11, v9 // (op 3606) 



s_lshl_b32 s0, s4, 7 // (op 3602) 
s_add_u32 s0, s0, 128 // (op 3602) 
v_add_u32 v0, v13, s0 // (op 3602) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3602) Store value


v_accvgpr_read v6, a16 // (op 3633) 
v_accvgpr_read v7, a17 // (op 3633) 
v_accvgpr_read v8, a18 // (op 3633) 
v_accvgpr_read v9, a19 // (op 3633) 
v_mul_f32 v14, v11, v6 // (op 3633) 
v_mul_f32 v15, v11, v7 // (op 3633) 
v_mul_f32 v16, v11, v8 // (op 3633) 
v_mul_f32 v17, v11, v9 // (op 3633) 



v_add_u32 v0, 256, v13 // (op 3636) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3636) Store value


v_accvgpr_read v6, a20 // (op 3642) 
v_accvgpr_read v7, a21 // (op 3642) 
v_accvgpr_read v8, a22 // (op 3642) 
v_accvgpr_read v9, a23 // (op 3642) 
v_mul_f32 v14, v11, v6 // (op 3642) 
v_mul_f32 v15, v11, v7 // (op 3642) 
v_mul_f32 v16, v11, v8 // (op 3642) 
v_mul_f32 v17, v11, v9 // (op 3642) 



s_lshl_b32 s0, s4, 7 // (op 3638) 
s_add_u32 s0, s0, 256 // (op 3638) 
v_add_u32 v0, v13, s0 // (op 3638) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3638) Store value


v_accvgpr_read v6, a24 // (op 3669) 
v_accvgpr_read v7, a25 // (op 3669) 
v_accvgpr_read v8, a26 // (op 3669) 
v_accvgpr_read v9, a27 // (op 3669) 
v_mul_f32 v14, v11, v6 // (op 3669) 
v_mul_f32 v15, v11, v7 // (op 3669) 
v_mul_f32 v16, v11, v8 // (op 3669) 
v_mul_f32 v17, v11, v9 // (op 3669) 



v_add_u32 v0, 384, v13 // (op 3672) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3672) Store value


v_accvgpr_read v6, a28 // (op 3678) 
v_accvgpr_read v7, a29 // (op 3678) 
v_accvgpr_read v8, a30 // (op 3678) 
v_accvgpr_read v9, a31 // (op 3678) 
v_mul_f32 v14, v11, v6 // (op 3678) 
v_mul_f32 v15, v11, v7 // (op 3678) 
v_mul_f32 v16, v11, v8 // (op 3678) 
v_mul_f32 v17, v11, v9 // (op 3678) 



s_lshl_b32 s0, s4, 7 // (op 3674) 
s_add_u32 s0, s0, 384 // (op 3674) 
v_add_u32 v0, v13, s0 // (op 3674) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3674) Store value


v_accvgpr_read v6, a32 // (op 3705) 
v_accvgpr_read v7, a33 // (op 3705) 
v_accvgpr_read v8, a34 // (op 3705) 
v_accvgpr_read v9, a35 // (op 3705) 
v_mul_f32 v14, v11, v6 // (op 3705) 
v_mul_f32 v15, v11, v7 // (op 3705) 
v_mul_f32 v16, v11, v8 // (op 3705) 
v_mul_f32 v17, v11, v9 // (op 3705) 



v_add_u32 v0, 512, v13 // (op 3708) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3708) Store value


v_accvgpr_read v6, a36 // (op 3714) 
v_accvgpr_read v7, a37 // (op 3714) 
v_accvgpr_read v8, a38 // (op 3714) 
v_accvgpr_read v9, a39 // (op 3714) 
v_mul_f32 v14, v11, v6 // (op 3714) 
v_mul_f32 v15, v11, v7 // (op 3714) 
v_mul_f32 v16, v11, v8 // (op 3714) 
v_mul_f32 v17, v11, v9 // (op 3714) 



s_lshl_b32 s0, s4, 7 // (op 3710) 
s_add_u32 s0, s0, 512 // (op 3710) 
v_add_u32 v0, v13, s0 // (op 3710) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3710) Store value


v_accvgpr_read v6, a40 // (op 3741) 
v_accvgpr_read v7, a41 // (op 3741) 
v_accvgpr_read v8, a42 // (op 3741) 
v_accvgpr_read v9, a43 // (op 3741) 
v_mul_f32 v14, v11, v6 // (op 3741) 
v_mul_f32 v15, v11, v7 // (op 3741) 
v_mul_f32 v16, v11, v8 // (op 3741) 
v_mul_f32 v17, v11, v9 // (op 3741) 



v_add_u32 v0, 640, v13 // (op 3744) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3744) Store value


v_accvgpr_read v6, a44 // (op 3750) 
v_accvgpr_read v7, a45 // (op 3750) 
v_accvgpr_read v8, a46 // (op 3750) 
v_accvgpr_read v9, a47 // (op 3750) 
v_mul_f32 v14, v11, v6 // (op 3750) 
v_mul_f32 v15, v11, v7 // (op 3750) 
v_mul_f32 v16, v11, v8 // (op 3750) 
v_mul_f32 v17, v11, v9 // (op 3750) 



s_lshl_b32 s0, s4, 7 // (op 3746) 
s_add_u32 s0, s0, 640 // (op 3746) 
v_add_u32 v0, v13, s0 // (op 3746) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3746) Store value


v_accvgpr_read v6, a48 // (op 3777) 
v_accvgpr_read v7, a49 // (op 3777) 
v_accvgpr_read v8, a50 // (op 3777) 
v_accvgpr_read v9, a51 // (op 3777) 
v_mul_f32 v14, v11, v6 // (op 3777) 
v_mul_f32 v15, v11, v7 // (op 3777) 
v_mul_f32 v16, v11, v8 // (op 3777) 
v_mul_f32 v17, v11, v9 // (op 3777) 



v_add_u32 v0, 768, v13 // (op 3780) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3780) Store value


v_accvgpr_read v6, a52 // (op 3786) 
v_accvgpr_read v7, a53 // (op 3786) 
v_accvgpr_read v8, a54 // (op 3786) 
v_accvgpr_read v9, a55 // (op 3786) 
v_mul_f32 v14, v11, v6 // (op 3786) 
v_mul_f32 v15, v11, v7 // (op 3786) 
v_mul_f32 v16, v11, v8 // (op 3786) 
v_mul_f32 v17, v11, v9 // (op 3786) 



s_lshl_b32 s0, s4, 7 // (op 3782) 
s_add_u32 s0, s0, 768 // (op 3782) 
v_add_u32 v0, v13, s0 // (op 3782) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3782) Store value


v_accvgpr_read v6, a56 // (op 3813) 
v_accvgpr_read v7, a57 // (op 3813) 
v_accvgpr_read v8, a58 // (op 3813) 
v_accvgpr_read v9, a59 // (op 3813) 
v_mul_f32 v14, v11, v6 // (op 3813) 
v_mul_f32 v15, v11, v7 // (op 3813) 
v_mul_f32 v16, v11, v8 // (op 3813) 
v_mul_f32 v17, v11, v9 // (op 3813) 



v_add_u32 v0, 896, v13 // (op 3816) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3816) Store value


v_accvgpr_read v6, a60 // (op 3822) 
v_accvgpr_read v7, a61 // (op 3822) 
v_accvgpr_read v8, a62 // (op 3822) 
v_accvgpr_read v9, a63 // (op 3822) 
v_mul_f32 v14, v11, v6 // (op 3822) 
v_mul_f32 v15, v11, v7 // (op 3822) 
v_mul_f32 v16, v11, v8 // (op 3822) 
v_mul_f32 v17, v11, v9 // (op 3822) 



s_lshl_b32 s0, s4, 7 // (op 3818) 
s_add_u32 s0, s0, 896 // (op 3818) 
v_add_u32 v0, v13, s0 // (op 3818) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3818) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 896, v12 // (op 4196) 

buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4196) Load value

v_accvgpr_read v14, a56 // (op 4188) 
v_accvgpr_read v15, a57 // (op 4188) 
v_accvgpr_read v16, a58 // (op 4188) 
v_accvgpr_read v17, a59 // (op 4188) 
v_mul_f32 v18, v11, v14 // (op 4188) 
v_mul_f32 v19, v11, v15 // (op 4188) 
v_mul_f32 v20, v11, v16 // (op 4188) 
v_mul_f32 v21, v11, v17 // (op 4188) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4195) 
v_mul_f32 v15, v10, v7 // (op 4195) 
v_mul_f32 v16, v10, v8 // (op 4195) 
v_mul_f32 v17, v10, v9 // (op 4195) 

v_add_f32 v6, v14, v18 // (op 4189) 
v_add_f32 v7, v15, v19 // (op 4189) 
v_add_f32 v8, v16, v20 // (op 4189) 
v_add_f32 v9, v17, v21 // (op 4189) 




v_add_u32 v0, 896, v13 // (op 4191) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4191) Store value




s_lshl_b32 s0, s2, 7 // (op 4200) 
s_add_u32 s0, s0, 896 // (op 4200) 
v_add_u32 v0, v12, s0 // (op 4200) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4200) Load value

v_accvgpr_read v14, a60 // (op 4202) 
v_accvgpr_read v15, a61 // (op 4202) 
v_accvgpr_read v16, a62 // (op 4202) 
v_accvgpr_read v17, a63 // (op 4202) 
v_mul_f32 v18, v11, v14 // (op 4202) 
v_mul_f32 v19, v11, v15 // (op 4202) 
v_mul_f32 v20, v11, v16 // (op 4202) 
v_mul_f32 v21, v11, v17 // (op 4202) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4198) 
v_mul_f32 v15, v10, v7 // (op 4198) 
v_mul_f32 v16, v10, v8 // (op 4198) 
v_mul_f32 v17, v10, v9 // (op 4198) 

v_add_f32 v6, v14, v18 // (op 4199) 
v_add_f32 v7, v15, v19 // (op 4199) 
v_add_f32 v8, v16, v20 // (op 4199) 
v_add_f32 v9, v17, v21 // (op 4199) 




s_lshl_b32 s0, s4, 7 // (op 4193) 
s_add_u32 s0, s0, 896 // (op 4193) 
v_add_u32 v0, v13, s0 // (op 4193) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4193) Store value




v_add_u32 v0, 768, v12 // (op 4155) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4155) Load value

v_accvgpr_read v14, a48 // (op 4147) 
v_accvgpr_read v15, a49 // (op 4147) 
v_accvgpr_read v16, a50 // (op 4147) 
v_accvgpr_read v17, a51 // (op 4147) 
v_mul_f32 v18, v11, v14 // (op 4147) 
v_mul_f32 v19, v11, v15 // (op 4147) 
v_mul_f32 v20, v11, v16 // (op 4147) 
v_mul_f32 v21, v11, v17 // (op 4147) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4154) 
v_mul_f32 v15, v10, v7 // (op 4154) 
v_mul_f32 v16, v10, v8 // (op 4154) 
v_mul_f32 v17, v10, v9 // (op 4154) 

v_add_f32 v6, v14, v18 // (op 4148) 
v_add_f32 v7, v15, v19 // (op 4148) 
v_add_f32 v8, v16, v20 // (op 4148) 
v_add_f32 v9, v17, v21 // (op 4148) 




v_add_u32 v0, 768, v13 // (op 4150) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4150) Store value




s_lshl_b32 s0, s2, 7 // (op 4159) 
s_add_u32 s0, s0, 768 // (op 4159) 
v_add_u32 v0, v12, s0 // (op 4159) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4159) Load value

v_accvgpr_read v14, a52 // (op 4161) 
v_accvgpr_read v15, a53 // (op 4161) 
v_accvgpr_read v16, a54 // (op 4161) 
v_accvgpr_read v17, a55 // (op 4161) 
v_mul_f32 v18, v11, v14 // (op 4161) 
v_mul_f32 v19, v11, v15 // (op 4161) 
v_mul_f32 v20, v11, v16 // (op 4161) 
v_mul_f32 v21, v11, v17 // (op 4161) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4157) 
v_mul_f32 v15, v10, v7 // (op 4157) 
v_mul_f32 v16, v10, v8 // (op 4157) 
v_mul_f32 v17, v10, v9 // (op 4157) 

v_add_f32 v6, v14, v18 // (op 4158) 
v_add_f32 v7, v15, v19 // (op 4158) 
v_add_f32 v8, v16, v20 // (op 4158) 
v_add_f32 v9, v17, v21 // (op 4158) 




s_lshl_b32 s0, s4, 7 // (op 4152) 
s_add_u32 s0, s0, 768 // (op 4152) 
v_add_u32 v0, v13, s0 // (op 4152) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4152) Store value




v_add_u32 v0, 640, v12 // (op 4114) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4114) Load value

v_accvgpr_read v14, a40 // (op 4106) 
v_accvgpr_read v15, a41 // (op 4106) 
v_accvgpr_read v16, a42 // (op 4106) 
v_accvgpr_read v17, a43 // (op 4106) 
v_mul_f32 v18, v11, v14 // (op 4106) 
v_mul_f32 v19, v11, v15 // (op 4106) 
v_mul_f32 v20, v11, v16 // (op 4106) 
v_mul_f32 v21, v11, v17 // (op 4106) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4113) 
v_mul_f32 v15, v10, v7 // (op 4113) 
v_mul_f32 v16, v10, v8 // (op 4113) 
v_mul_f32 v17, v10, v9 // (op 4113) 

v_add_f32 v6, v14, v18 // (op 4107) 
v_add_f32 v7, v15, v19 // (op 4107) 
v_add_f32 v8, v16, v20 // (op 4107) 
v_add_f32 v9, v17, v21 // (op 4107) 




v_add_u32 v0, 640, v13 // (op 4109) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4109) Store value




s_lshl_b32 s0, s2, 7 // (op 4118) 
s_add_u32 s0, s0, 640 // (op 4118) 
v_add_u32 v0, v12, s0 // (op 4118) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4118) Load value

v_accvgpr_read v14, a44 // (op 4120) 
v_accvgpr_read v15, a45 // (op 4120) 
v_accvgpr_read v16, a46 // (op 4120) 
v_accvgpr_read v17, a47 // (op 4120) 
v_mul_f32 v18, v11, v14 // (op 4120) 
v_mul_f32 v19, v11, v15 // (op 4120) 
v_mul_f32 v20, v11, v16 // (op 4120) 
v_mul_f32 v21, v11, v17 // (op 4120) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4116) 
v_mul_f32 v15, v10, v7 // (op 4116) 
v_mul_f32 v16, v10, v8 // (op 4116) 
v_mul_f32 v17, v10, v9 // (op 4116) 

v_add_f32 v6, v14, v18 // (op 4117) 
v_add_f32 v7, v15, v19 // (op 4117) 
v_add_f32 v8, v16, v20 // (op 4117) 
v_add_f32 v9, v17, v21 // (op 4117) 




s_lshl_b32 s0, s4, 7 // (op 4111) 
s_add_u32 s0, s0, 640 // (op 4111) 
v_add_u32 v0, v13, s0 // (op 4111) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4111) Store value




v_add_u32 v0, 512, v12 // (op 4073) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4073) Load value

v_accvgpr_read v14, a32 // (op 4065) 
v_accvgpr_read v15, a33 // (op 4065) 
v_accvgpr_read v16, a34 // (op 4065) 
v_accvgpr_read v17, a35 // (op 4065) 
v_mul_f32 v18, v11, v14 // (op 4065) 
v_mul_f32 v19, v11, v15 // (op 4065) 
v_mul_f32 v20, v11, v16 // (op 4065) 
v_mul_f32 v21, v11, v17 // (op 4065) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4072) 
v_mul_f32 v15, v10, v7 // (op 4072) 
v_mul_f32 v16, v10, v8 // (op 4072) 
v_mul_f32 v17, v10, v9 // (op 4072) 

v_add_f32 v6, v14, v18 // (op 4066) 
v_add_f32 v7, v15, v19 // (op 4066) 
v_add_f32 v8, v16, v20 // (op 4066) 
v_add_f32 v9, v17, v21 // (op 4066) 




v_add_u32 v0, 512, v13 // (op 4068) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4068) Store value




s_lshl_b32 s0, s2, 7 // (op 4077) 
s_add_u32 s0, s0, 512 // (op 4077) 
v_add_u32 v0, v12, s0 // (op 4077) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4077) Load value

v_accvgpr_read v14, a36 // (op 4079) 
v_accvgpr_read v15, a37 // (op 4079) 
v_accvgpr_read v16, a38 // (op 4079) 
v_accvgpr_read v17, a39 // (op 4079) 
v_mul_f32 v18, v11, v14 // (op 4079) 
v_mul_f32 v19, v11, v15 // (op 4079) 
v_mul_f32 v20, v11, v16 // (op 4079) 
v_mul_f32 v21, v11, v17 // (op 4079) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4075) 
v_mul_f32 v15, v10, v7 // (op 4075) 
v_mul_f32 v16, v10, v8 // (op 4075) 
v_mul_f32 v17, v10, v9 // (op 4075) 

v_add_f32 v6, v14, v18 // (op 4076) 
v_add_f32 v7, v15, v19 // (op 4076) 
v_add_f32 v8, v16, v20 // (op 4076) 
v_add_f32 v9, v17, v21 // (op 4076) 




s_lshl_b32 s0, s4, 7 // (op 4070) 
s_add_u32 s0, s0, 512 // (op 4070) 
v_add_u32 v0, v13, s0 // (op 4070) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4070) Store value




v_add_u32 v0, 384, v12 // (op 4032) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4032) Load value

v_accvgpr_read v14, a24 // (op 4024) 
v_accvgpr_read v15, a25 // (op 4024) 
v_accvgpr_read v16, a26 // (op 4024) 
v_accvgpr_read v17, a27 // (op 4024) 
v_mul_f32 v18, v11, v14 // (op 4024) 
v_mul_f32 v19, v11, v15 // (op 4024) 
v_mul_f32 v20, v11, v16 // (op 4024) 
v_mul_f32 v21, v11, v17 // (op 4024) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4031) 
v_mul_f32 v15, v10, v7 // (op 4031) 
v_mul_f32 v16, v10, v8 // (op 4031) 
v_mul_f32 v17, v10, v9 // (op 4031) 

v_add_f32 v6, v14, v18 // (op 4025) 
v_add_f32 v7, v15, v19 // (op 4025) 
v_add_f32 v8, v16, v20 // (op 4025) 
v_add_f32 v9, v17, v21 // (op 4025) 




v_add_u32 v0, 384, v13 // (op 4027) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4027) Store value




s_lshl_b32 s0, s2, 7 // (op 4036) 
s_add_u32 s0, s0, 384 // (op 4036) 
v_add_u32 v0, v12, s0 // (op 4036) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4036) Load value

v_accvgpr_read v14, a28 // (op 4038) 
v_accvgpr_read v15, a29 // (op 4038) 
v_accvgpr_read v16, a30 // (op 4038) 
v_accvgpr_read v17, a31 // (op 4038) 
v_mul_f32 v18, v11, v14 // (op 4038) 
v_mul_f32 v19, v11, v15 // (op 4038) 
v_mul_f32 v20, v11, v16 // (op 4038) 
v_mul_f32 v21, v11, v17 // (op 4038) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4034) 
v_mul_f32 v15, v10, v7 // (op 4034) 
v_mul_f32 v16, v10, v8 // (op 4034) 
v_mul_f32 v17, v10, v9 // (op 4034) 

v_add_f32 v6, v14, v18 // (op 4035) 
v_add_f32 v7, v15, v19 // (op 4035) 
v_add_f32 v8, v16, v20 // (op 4035) 
v_add_f32 v9, v17, v21 // (op 4035) 




s_lshl_b32 s0, s4, 7 // (op 4029) 
s_add_u32 s0, s0, 384 // (op 4029) 
v_add_u32 v0, v13, s0 // (op 4029) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4029) Store value




v_add_u32 v0, 256, v12 // (op 3991) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3991) Load value

v_accvgpr_read v14, a16 // (op 3983) 
v_accvgpr_read v15, a17 // (op 3983) 
v_accvgpr_read v16, a18 // (op 3983) 
v_accvgpr_read v17, a19 // (op 3983) 
v_mul_f32 v18, v11, v14 // (op 3983) 
v_mul_f32 v19, v11, v15 // (op 3983) 
v_mul_f32 v20, v11, v16 // (op 3983) 
v_mul_f32 v21, v11, v17 // (op 3983) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3990) 
v_mul_f32 v15, v10, v7 // (op 3990) 
v_mul_f32 v16, v10, v8 // (op 3990) 
v_mul_f32 v17, v10, v9 // (op 3990) 

v_add_f32 v6, v14, v18 // (op 3984) 
v_add_f32 v7, v15, v19 // (op 3984) 
v_add_f32 v8, v16, v20 // (op 3984) 
v_add_f32 v9, v17, v21 // (op 3984) 




v_add_u32 v0, 256, v13 // (op 3986) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3986) Store value




s_lshl_b32 s0, s2, 7 // (op 3995) 
s_add_u32 s0, s0, 256 // (op 3995) 
v_add_u32 v0, v12, s0 // (op 3995) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3995) Load value

v_accvgpr_read v14, a20 // (op 3997) 
v_accvgpr_read v15, a21 // (op 3997) 
v_accvgpr_read v16, a22 // (op 3997) 
v_accvgpr_read v17, a23 // (op 3997) 
v_mul_f32 v18, v11, v14 // (op 3997) 
v_mul_f32 v19, v11, v15 // (op 3997) 
v_mul_f32 v20, v11, v16 // (op 3997) 
v_mul_f32 v21, v11, v17 // (op 3997) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3993) 
v_mul_f32 v15, v10, v7 // (op 3993) 
v_mul_f32 v16, v10, v8 // (op 3993) 
v_mul_f32 v17, v10, v9 // (op 3993) 

v_add_f32 v6, v14, v18 // (op 3994) 
v_add_f32 v7, v15, v19 // (op 3994) 
v_add_f32 v8, v16, v20 // (op 3994) 
v_add_f32 v9, v17, v21 // (op 3994) 




s_lshl_b32 s0, s4, 7 // (op 3988) 
s_add_u32 s0, s0, 256 // (op 3988) 
v_add_u32 v0, v13, s0 // (op 3988) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3988) Store value




v_add_u32 v0, 128, v12 // (op 3950) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3950) Load value

v_accvgpr_read v14, a8 // (op 3942) 
v_accvgpr_read v15, a9 // (op 3942) 
v_accvgpr_read v16, a10 // (op 3942) 
v_accvgpr_read v17, a11 // (op 3942) 
v_mul_f32 v18, v11, v14 // (op 3942) 
v_mul_f32 v19, v11, v15 // (op 3942) 
v_mul_f32 v20, v11, v16 // (op 3942) 
v_mul_f32 v21, v11, v17 // (op 3942) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3949) 
v_mul_f32 v15, v10, v7 // (op 3949) 
v_mul_f32 v16, v10, v8 // (op 3949) 
v_mul_f32 v17, v10, v9 // (op 3949) 

v_add_f32 v6, v14, v18 // (op 3943) 
v_add_f32 v7, v15, v19 // (op 3943) 
v_add_f32 v8, v16, v20 // (op 3943) 
v_add_f32 v9, v17, v21 // (op 3943) 




v_add_u32 v0, 128, v13 // (op 3945) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3945) Store value




s_lshl_b32 s0, s2, 7 // (op 3954) 
s_add_u32 s0, s0, 128 // (op 3954) 
v_add_u32 v0, v12, s0 // (op 3954) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3954) Load value

v_accvgpr_read v14, a12 // (op 3956) 
v_accvgpr_read v15, a13 // (op 3956) 
v_accvgpr_read v16, a14 // (op 3956) 
v_accvgpr_read v17, a15 // (op 3956) 
v_mul_f32 v18, v11, v14 // (op 3956) 
v_mul_f32 v19, v11, v15 // (op 3956) 
v_mul_f32 v20, v11, v16 // (op 3956) 
v_mul_f32 v21, v11, v17 // (op 3956) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3952) 
v_mul_f32 v15, v10, v7 // (op 3952) 
v_mul_f32 v16, v10, v8 // (op 3952) 
v_mul_f32 v17, v10, v9 // (op 3952) 

v_add_f32 v6, v14, v18 // (op 3953) 
v_add_f32 v7, v15, v19 // (op 3953) 
v_add_f32 v8, v16, v20 // (op 3953) 
v_add_f32 v9, v17, v21 // (op 3953) 




s_lshl_b32 s0, s4, 7 // (op 3947) 
s_add_u32 s0, s0, 128 // (op 3947) 
v_add_u32 v0, v13, s0 // (op 3947) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3947) Store value




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




s_lshl_b32 s0, s2, 7 // (op 3912) 
v_add_u32 v0, v12, s0 // (op 3912) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3912) Load value

v_accvgpr_read v14, a4 // (op 3909) 
v_accvgpr_read v15, a5 // (op 3909) 
v_accvgpr_read v16, a6 // (op 3909) 
v_accvgpr_read v17, a7 // (op 3909) 
v_mul_f32 v18, v11, v14 // (op 3909) 
v_mul_f32 v19, v11, v15 // (op 3909) 
v_mul_f32 v20, v11, v16 // (op 3909) 
v_mul_f32 v21, v11, v17 // (op 3909) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3913) 
v_mul_f32 v15, v10, v7 // (op 3913) 
v_mul_f32 v16, v10, v8 // (op 3913) 
v_mul_f32 v17, v10, v9 // (op 3913) 

v_add_f32 v6, v14, v18 // (op 3910) 
v_add_f32 v7, v15, v19 // (op 3910) 
v_add_f32 v8, v16, v20 // (op 3910) 
v_add_f32 v9, v17, v21 // (op 3910) 




s_lshl_b32 s0, s4, 7 // (op 3911) 
v_add_u32 v0, v13, s0 // (op 3911) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3911) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_133:
























s_endpgm  // End of RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
.LRR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_end:
.size RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC, .LRR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC_end-RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
  .amdhsa_next_free_vgpr 216
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 76288 // lds bytes
  .amdhsa_accum_offset 152
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
    TransposeB: True
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
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA } ]
    macrotile: [256, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC
    .symbol: RR_GEMM_TT_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x64x128_UR_2_LDSSYNC.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 76288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 150
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
      - .name: Tensor_2_size_1_12
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
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 128
        .value_kind: by_value
...

.end_amdgpu_metadata

