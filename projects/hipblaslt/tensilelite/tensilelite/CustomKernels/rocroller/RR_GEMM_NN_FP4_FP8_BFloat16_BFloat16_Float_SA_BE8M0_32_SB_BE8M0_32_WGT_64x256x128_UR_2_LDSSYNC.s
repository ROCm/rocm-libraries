// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC
.p2align 8
.type RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC,@function
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 8980) 
v_and_b32 v3, 48, v1 // (op 8980) 
v_sub_u32 v4, v0, v3 // (op 8980) 

v_lshrrev_b32 v3, 4, v0 // (op 8980) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 8980) 


v_lshrrev_b32 v3, 7, v1 // (op 8980) 
v_lshlrev_b32 v4, 4, v3 // (op 8980) 

v_lshl_add_u32 v3, v0, 6, v4 // (op 8980) 


v_lshlrev_b32 v0, 2, v3 // (op 8980) 

v_lshrrev_b32 v3, 3, v0 // (op 8980) 

v_lshrrev_b32 v0, 6, v1 // (op 9038) 
v_and_b32 v4, 1, v0 // (op 9038) 

v_and_b32 v0, 15, v1 // (op 9038) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 9038) 


v_and_b32 v0, 48, v1 // (op 9038) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 9038) 


v_lshrrev_b32 v0, 7, v1 // (op 9096) 
v_and_b32 v5, 15, v1 // (op 9096) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 9096) 


v_and_b32 v0, 63, v1 // (op 9096) 
v_lshrrev_b32 v5, 4, v0 // (op 9096) 

v_lshl_add_u32 v0, v5, 6, v6 // (op 9096) 


v_lshrrev_b32 v5, 6, v1 // (op 9154) 
v_and_b32 v6, 1, v5 // (op 9154) 

v_and_b32 v5, 15, v1 // (op 9154) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9154) 


v_and_b32 v5, 63, v1 // (op 9154) 
v_lshrrev_b32 v6, 4, v5 // (op 9154) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 9154) 


v_and_b32 v6, 63, v1 // (op 9206) 
v_and_b32 v7, 48, v1 // (op 9206) 
v_sub_u32 v8, v6, v7 // (op 9206) 

v_lshrrev_b32 v7, 4, v6 // (op 9206) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 9206) 


v_lshrrev_b32 v7, 7, v1 // (op 9206) 
v_lshlrev_b32 v8, 4, v7 // (op 9206) 

v_lshl_add_u32 v7, v6, 6, v8 // (op 9206) 


v_lshlrev_b32 v6, 2, v7 // (op 9206) 

v_lshrrev_b32 v7, 3, v6 // (op 9206) 

v_lshrrev_b32 v6, 6, v1 // (op 9235) 
v_and_b32 v8, 1, v6 // (op 9235) 

v_and_b32 v6, 15, v1 // (op 9235) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 9235) 


v_and_b32 v6, 48, v1 // (op 9235) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 9235) 


v_lshrrev_b32 v6, 7, v1 // (op 9264) 
v_and_b32 v9, 15, v1 // (op 9264) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9264) 


v_and_b32 v6, 63, v1 // (op 9264) 
v_lshrrev_b32 v9, 4, v6 // (op 9264) 

v_lshl_add_u32 v6, v9, 6, v10 // (op 9264) 


v_lshrrev_b32 v9, 6, v1 // (op 9293) 
v_and_b32 v10, 1, v9 // (op 9293) 

v_and_b32 v9, 15, v1 // (op 9293) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9293) 


v_and_b32 v9, 63, v1 // (op 9293) 
v_lshrrev_b32 v10, 4, v9 // (op 9293) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 9293) 


v_lshrrev_b32 v10, 7, v1 // (op 8918) 
v_and_b32 v11, 63, v1 // (op 8918) 
v_lshrrev_b32 v12, 4, v11 // (op 8918) 
v_lshlrev_b32 v13, 2, v12 // (op 8918) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8918) 


v_lshl_add_u32 v10, s16, 6, v12 // (op 8918) 

v_lshrrev_b32 v12, 2, v11 // (op 8918) 

v_and_b32 v11, 3, v12 // (op 8918) 

v_and_b32 v12, 3, v1 // (op 8918) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 8918) 


v_lshrrev_b32 v11, 6, v1 // (op 8918) 
v_and_b32 v12, 1, v11 // (op 8918) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 8918) 


v_lshl_add_u32 v12, s17, 8, v11 // (op 8918) 

v_mul_lo_u32 v11, v12, s2 // (op 8918) 

v_add_lshl_u32 v12, v10, v11, 1 // (op 8918) 


v_lshrrev_b32 v10, 7, v1 // (op 9175) 
v_and_b32 v11, 63, v1 // (op 9175) 
v_lshrrev_b32 v13, 4, v11 // (op 9175) 
s_mul_i32 s1, s2, s9 // (op 8920) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 8920) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 8920) least significant
s_mul_hi_u32 s19, s2, s8 // (op 8920) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8920) most significant: sum
s_add_u32 s1, s1, s19 // (op 8920) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 8920) 

v_lshlrev_b32 v14, 2, v13 // (op 9175) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 9175) 


v_lshl_add_u32 v10, s16, 6, v13 // (op 9175) 

v_lshrrev_b32 v13, 2, v11 // (op 9175) 

v_and_b32 v11, 3, v13 // (op 9175) 

v_and_b32 v13, 3, v1 // (op 9175) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 9175) 


v_lshrrev_b32 v11, 6, v1 // (op 9175) 
v_and_b32 v13, 1, v11 // (op 9175) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 9175) 


v_lshl_add_u32 v13, s17, 8, v11 // (op 9175) 

v_mul_lo_u32 v11, v13, s4 // (op 9175) 

v_add_lshl_u32 v13, v10, v11, 1 // (op 9175) 


s_mov_b32 s40, s6 // (op 8920) reg expression
s_mov_b32 s41, s7 // (op 8920) reg expression
s_mov_b32 s42, s18 // (op 8920) reg expression
s_mov_b32 s43, 131072 // (op 8920) call()

s_mul_i32 s1, s4, s9 // (op 9177) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 9177) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 9177) least significant
s_mul_hi_u32 s7, s4, s8 // (op 9177) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 9177) most significant: sum
s_add_u32 s1, s1, s7 // (op 9177) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 9177) 

s_mov_b32 s44, s10 // (op 9177) reg expression
s_mov_b32 s45, s11 // (op 9177) reg expression
s_mov_b32 s46, s6 // (op 9177) reg expression
s_mov_b32 s47, 131072 // (op 9177) call()


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
v_accvgpr_write a16, 0.00000 // (op 868) call()
v_accvgpr_write a17, 0.00000 // (op 868) call()
v_accvgpr_write a18, 0.00000 // (op 868) call()
v_accvgpr_write a19, 0.00000 // (op 868) call()
v_accvgpr_write a20, 0.00000 // (op 1044) call()
v_accvgpr_write a21, 0.00000 // (op 1044) call()
v_accvgpr_write a22, 0.00000 // (op 1044) call()
v_accvgpr_write a23, 0.00000 // (op 1044) call()
v_accvgpr_write a24, 0.00000 // (op 1220) call()
v_accvgpr_write a25, 0.00000 // (op 1220) call()
v_accvgpr_write a26, 0.00000 // (op 1220) call()
v_accvgpr_write a27, 0.00000 // (op 1220) call()
v_accvgpr_write a28, 0.00000 // (op 1396) call()
v_accvgpr_write a29, 0.00000 // (op 1396) call()
v_accvgpr_write a30, 0.00000 // (op 1396) call()
v_accvgpr_write a31, 0.00000 // (op 1396) call()
v_accvgpr_write a32, 0.00000 // (op 1748) call()
v_accvgpr_write a33, 0.00000 // (op 1748) call()
v_accvgpr_write a34, 0.00000 // (op 1748) call()
v_accvgpr_write a35, 0.00000 // (op 1748) call()
v_accvgpr_write a36, 0.00000 // (op 2365) call()
v_accvgpr_write a37, 0.00000 // (op 2365) call()
v_accvgpr_write a38, 0.00000 // (op 2365) call()
v_accvgpr_write a39, 0.00000 // (op 2365) call()
v_accvgpr_write a40, 0.00000 // (op 2366) call()
v_accvgpr_write a41, 0.00000 // (op 2366) call()
v_accvgpr_write a42, 0.00000 // (op 2366) call()
v_accvgpr_write a43, 0.00000 // (op 2366) call()
v_accvgpr_write a44, 0.00000 // (op 2367) call()
v_accvgpr_write a45, 0.00000 // (op 2367) call()
v_accvgpr_write a46, 0.00000 // (op 2367) call()
v_accvgpr_write a47, 0.00000 // (op 2367) call()
v_accvgpr_write a48, 0.00000 // (op 2368) call()
v_accvgpr_write a49, 0.00000 // (op 2368) call()
v_accvgpr_write a50, 0.00000 // (op 2368) call()
v_accvgpr_write a51, 0.00000 // (op 2368) call()
v_accvgpr_write a52, 0.00000 // (op 2369) call()
v_accvgpr_write a53, 0.00000 // (op 2369) call()
v_accvgpr_write a54, 0.00000 // (op 2369) call()
v_accvgpr_write a55, 0.00000 // (op 2369) call()
v_accvgpr_write a56, 0.00000 // (op 2370) call()
v_accvgpr_write a57, 0.00000 // (op 2370) call()
v_accvgpr_write a58, 0.00000 // (op 2370) call()
v_accvgpr_write a59, 0.00000 // (op 2370) call()
v_accvgpr_write a60, 0.00000 // (op 2371) call()
v_accvgpr_write a61, 0.00000 // (op 2371) call()
v_accvgpr_write a62, 0.00000 // (op 2371) call()
v_accvgpr_write a63, 0.00000 // (op 2371) call()
v_and_b32 v14, -2, v1 // (op 8680) 
v_sub_u32 v15, v1, v14 // (op 8680) 

v_lshlrev_b32 v14, 5, v15 // (op 8680) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 8680) 

v_mov_b32 v17, 0 // (op 8680) convert
v_mov_b32 v16, v15 // (op 8680) convert

v_lshrrev_b32 v14, 1, v1 // (op 8680) 
v_mul_lo_u32 v19, v14, s13 // (op 8680) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 8680) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 8680) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 8680) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 8680) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 8680) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 8680) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8680) 

v_mov_b32 v14, v16 // (op 8680) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 8682) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8682) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8682) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8682) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8682) most significant: sum
s_add_u32 s1, s1, s7 // (op 8682) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8682) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8682) 

s_mov_b32 s48, s20 // (op 8682) reg expression
s_mov_b32 s49, s21 // (op 8682) reg expression
s_mov_b32 s50, s0 // (op 8682) reg expression
s_mov_b32 s51, 131072 // (op 8682) call()

v_and_b32 v15, -2, v1 // (op 8699) 
v_sub_u32 v16, v1, v15 // (op 8699) 

v_lshlrev_b32 v15, 5, v16 // (op 8699) 

v_lshl_add_u32 v16, s16, 6, v15 // (op 8699) 

v_mov_b32 v19, 0 // (op 8699) convert
v_mov_b32 v18, v16 // (op 8699) convert

v_lshrrev_b32 v15, 1, v1 // (op 8699) 
v_mul_lo_u32 v17, v15, s13 // (op 8699) most significant: low of low * high
v_mul_hi_u32 v20, v15, s12 // (op 8699) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 8699) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 8699) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v16 // (op 8699) least significant half
v_addc_co_u32 v21, s[0:1], v19, v17, s[0:1] // (op 8699) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 8699) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 8699) 

v_mov_b32 v15, v18 // (op 8699) convert

s_mul_i32 s1, s12, s23 // (op 8701) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8701) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8701) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8701) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8701) most significant: sum
s_add_u32 s1, s1, s7 // (op 8701) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8701) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8701) 

s_mov_b32 s48, s20 // (op 8701) reg expression
s_mov_b32 s49, s21 // (op 8701) reg expression
s_mov_b32 s50, s0 // (op 8701) reg expression
s_mov_b32 s51, 131072 // (op 8701) call()

v_and_b32 v16, -8, v1 // (op 8744) 
v_sub_u32 v17, v1, v16 // (op 8744) 

v_and_b32 v16, 4294967288, v1 // (op 8744) 
v_lshl_add_u32 v18, s17, 8, v16 // (op 8744) 

v_mul_lo_u32 v16, v18, s24 // (op 8744) 

v_lshl_add_u32 v18, v17, 4, v16 // (op 8744) 


s_mul_i32 s1, s24, s9 // (op 8746) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8746) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8746) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8746) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8746) most significant: sum
s_add_u32 s1, s1, s7 // (op 8746) most significant: sum


s_mov_b32 s52, s26 // (op 8746) reg expression
s_mov_b32 s53, s27 // (op 8746) reg expression
s_mov_b32 s54, s0 // (op 8746) reg expression
s_mov_b32 s55, 131072 // (op 8746) call()

v_and_b32 v16, -8, v1 // (op 8763) 
v_sub_u32 v17, v1, v16 // (op 8763) 

v_and_b32 v16, 4294967288, v1 // (op 8763) 
v_lshl_add_u32 v19, s17, 8, v16 // (op 8763) 

v_mul_lo_u32 v16, v19, s24 // (op 8763) 

v_lshl_add_u32 v19, v17, 4, v16 // (op 8763) 


s_mul_i32 s1, s24, s9 // (op 8765) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8765) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8765) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8765) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8765) most significant: sum
s_add_u32 s1, s1, s7 // (op 8765) most significant: sum


s_mov_b32 s52, s26 // (op 8765) reg expression
s_mov_b32 s53, s27 // (op 8765) reg expression
s_mov_b32 s54, s0 // (op 8765) reg expression
s_mov_b32 s55, 131072 // (op 8765) call()

v_and_b32 v16, -64, v1 // (op 8808) 
v_sub_u32 v17, v1, v16 // (op 8808) 

v_lshl_add_u32 v16, s16, 6, v17 // (op 8808) 

v_lshrrev_b32 v17, 6, v1 // (op 8808) 
v_mul_lo_u32 v20, v17, s28 // (op 8808) 

v_add_u32 v17, v16, v20 // (op 8808) 


s_mul_i32 s1, s28, s33 // (op 8810) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 8810) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8810) least significant
s_mul_hi_u32 s7, s28, s32 // (op 8810) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8810) most significant: sum
s_add_u32 s1, s1, s7 // (op 8810) most significant: sum


s_mov_b32 s56, s30 // (op 8810) reg expression
s_mov_b32 s57, s31 // (op 8810) reg expression
s_mov_b32 s58, s0 // (op 8810) reg expression
s_mov_b32 s59, 131072 // (op 8810) call()

v_and_b32 v16, -64, v1 // (op 8827) 
v_sub_u32 v20, v1, v16 // (op 8827) 

v_lshl_add_u32 v16, s16, 6, v20 // (op 8827) 

v_lshrrev_b32 v20, 6, v1 // (op 8827) 
v_mul_lo_u32 v21, v20, s28 // (op 8827) 

v_add_u32 v20, v16, v21 // (op 8827) 


s_mul_i32 s1, s28, s33 // (op 8829) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 8829) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8829) least significant
s_mul_hi_u32 s7, s28, s32 // (op 8829) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8829) most significant: sum
s_add_u32 s1, s1, s7 // (op 8829) most significant: sum


s_mov_b32 s56, s30 // (op 8829) reg expression
s_mov_b32 s57, s31 // (op 8829) reg expression
s_mov_b32 s58, s0 // (op 8829) reg expression
s_mov_b32 s59, 131072 // (op 8829) call()

v_lshl_add_u32 v16, s17, 8, v1 // (op 8872) 
v_mul_lo_u32 v21, v16, s34 // (op 8872) 

s_mul_i32 s1, s34, s9 // (op 8874) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 8874) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8874) least significant
s_mul_hi_u32 s7, s34, s8 // (op 8874) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8874) most significant: sum
s_add_u32 s1, s1, s7 // (op 8874) most significant: sum


s_mov_b32 s60, s36 // (op 8874) reg expression
s_mov_b32 s61, s37 // (op 8874) reg expression
s_mov_b32 s62, s0 // (op 8874) reg expression
s_mov_b32 s63, 131072 // (op 8874) call()

v_lshl_add_u32 v16, s17, 8, v1 // (op 8891) 
v_mul_lo_u32 v22, v16, s34 // (op 8891) 

s_mul_i32 s1, s34, s9 // (op 8893) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 8893) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8893) least significant
s_mul_hi_u32 s7, s34, s8 // (op 8893) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8893) most significant: sum
s_add_u32 s1, s1, s7 // (op 8893) most significant: sum


s_mov_b32 s60, s36 // (op 8893) reg expression
s_mov_b32 s61, s37 // (op 8893) reg expression
s_mov_b32 s62, s0 // (op 8893) reg expression
s_mov_b32 s63, 131072 // (op 8893) call()


v_mov_b64 v[24:25], s[24:25] // (op 8200) convert

v_mov_b32 v16, v18 // (op 8200) 
buffer_load_dwordx4 v[26:29], v16, s[52:55], 0 offen // (op 8200) Load value
v_add_u32 v16, v16, v24 // (op 8200) 
buffer_load_dwordx4 v[30:33], v16, s[52:55], 0 offen // (op 8200) Load value
v_add_u32 v16, v16, v24 // (op 8200) 
buffer_load_dwordx4 v[34:37], v16, s[52:55], 0 offen // (op 8200) Load value
v_add_u32 v16, v16, v24 // (op 8200) 
buffer_load_dwordx4 v[38:41], v16, s[52:55], 0 offen // (op 8200) Load value
v_add_u32 v16, v16, v24 // (op 8200) 
buffer_load_dwordx4 v[42:45], v16, s[52:55], 0 offen // (op 8200) Load value
v_add_u32 v16, v16, v24 // (op 8200) 
buffer_load_dwordx4 v[46:49], v16, s[52:55], 0 offen // (op 8200) Load value
v_add_u32 v16, v16, v24 // (op 8200) 
buffer_load_dwordx4 v[50:53], v16, s[52:55], 0 offen // (op 8200) Load value
v_add_u32 v16, v16, v24 // (op 8200) 
buffer_load_dwordx4 v[54:57], v16, s[52:55], 0 offen // (op 8200) Load value





buffer_load_dword v16, v21, s[60:63], 0 offen // (op 8203) Load value


v_mov_b32 v23, v14 // (op 8206) reg expression

buffer_load_dwordx4 v[58:61], v23, s[48:51], 0 offen // (op 8206) Load value



v_mov_b32 v23, v17 // (op 8209) reg expression

buffer_load_ubyte v24, v23, s[56:59], 0 offen offset: 0 // (op 8209) Load value


v_mov_b64 v[62:63], s[24:25] // (op 8212) convert

v_mov_b32 v23, v18 // (op 8212) 
buffer_load_dwordx4 v[64:67], v23, s[52:55], 0 offen offset:128 // (op 8212) Load value
v_add_u32 v23, v23, v62 // (op 8212) 
buffer_load_dwordx4 v[68:71], v23, s[52:55], 0 offen offset:128 // (op 8212) Load value
v_add_u32 v23, v23, v62 // (op 8212) 
buffer_load_dwordx4 v[72:75], v23, s[52:55], 0 offen offset:128 // (op 8212) Load value
v_add_u32 v23, v23, v62 // (op 8212) 
buffer_load_dwordx4 v[76:79], v23, s[52:55], 0 offen offset:128 // (op 8212) Load value
v_add_u32 v23, v23, v62 // (op 8212) 
buffer_load_dwordx4 v[80:83], v23, s[52:55], 0 offen offset:128 // (op 8212) Load value
v_add_u32 v23, v23, v62 // (op 8212) 
buffer_load_dwordx4 v[84:87], v23, s[52:55], 0 offen offset:128 // (op 8212) Load value
v_add_u32 v23, v23, v62 // (op 8212) 
buffer_load_dwordx4 v[88:91], v23, s[52:55], 0 offen offset:128 // (op 8212) Load value
v_add_u32 v23, v23, v62 // (op 8212) 
buffer_load_dwordx4 v[92:95], v23, s[52:55], 0 offen offset:128 // (op 8212) Load value






buffer_load_dword v18, v21, s[60:63], 0 offen offset:4 // (op 8215) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 8218) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 8218) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 8218) 

v_add_u32 v21, v14, s0 // (op 8218) 


buffer_load_dwordx4 v[96:99], v21, s[48:51], 0 offen // (op 8218) Load value




s_lshl_b32 s0, s28, 2 // (op 8221) 
v_add_u32 v14, v17, s0 // (op 8221) 


buffer_load_ubyte v21, v14, s[56:59], 0 offen offset: 0 // (op 8221) Load value


v_and_b32 v14, -8, v1 // (op 9021) 
v_sub_u32 v17, v1, v14 // (op 9021) 

v_lshlrev_b32 v14, 4, v17 // (op 9021) 

v_lshrrev_b32 v17, 3, v1 // (op 9021) 
v_lshl_add_u32 v23, v17, 10, v14 // (op 9021) 





s_waitcnt vmcnt(21)
ds_write_b128 v23, v[26:29] // (op 8223) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v23, v[30:33] offset:128 // (op 8223) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v23, v[34:37] offset:256 // (op 8223) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v23, v[38:41] offset:384 // (op 8223) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v23, v[42:45] offset:512 // (op 8223) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v23, v[46:49] offset:640 // (op 8223) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v23, v[50:53] offset:768 // (op 8223) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v23, v[54:57] offset:896 // (op 8223) Store local data 

v_lshlrev_b32 v14, 2, v1 // (op 9137) 



s_waitcnt vmcnt(13)
ds_write_b32 v14, v16 offset:32768 // (op 8224) Store local data 

v_and_b32 v14, -2, v1 // (op 8963) 
v_sub_u32 v17, v1, v14 // (op 8963) 

v_lshlrev_b32 v14, 5, v17 // (op 8963) 

v_lshrrev_b32 v17, 1, v1 // (op 8963) 
v_lshl_add_u32 v23, v17, 6, v14 // (op 8963) 


v_lshlrev_b32 v14, 2, v23 // (op 8963) 

v_lshrrev_b32 v17, 3, v14 // (op 8963) 




s_waitcnt vmcnt(12)
ds_write_b128 v17, v[58:61] offset:33792 // (op 8225) Store local data 

v_and_b32 v14, -64, v1 // (op 9079) 
v_sub_u32 v17, v1, v14 // (op 9079) 

v_and_b32 v14, 4294967232, v1 // (op 9079) 
v_add_u32 v23, v14, v17 // (op 9079) 





s_waitcnt vmcnt(11)
ds_write_b8 v23, v24 offset:37888 // (op 8226) Store local data 

v_and_b32 v14, -2, v1 // (op 8934) 
v_sub_u32 v17, v1, v14 // (op 8934) 

v_lshlrev_b32 v14, 5, v17 // (op 8934) 

v_lshrrev_b32 v17, 1, v1 // (op 8934) 
v_lshl_add_u32 v23, v17, 6, v14 // (op 8934) 


v_lshlrev_b32 v14, 2, v23 // (op 8934) 

v_lshrrev_b32 v17, 3, v14 // (op 8934) 

v_and_b32 v14, -8, v1 // (op 8992) 
v_sub_u32 v23, v1, v14 // (op 8992) 

v_lshlrev_b32 v14, 4, v23 // (op 8992) 

v_lshrrev_b32 v23, 3, v1 // (op 8992) 
v_lshl_add_u32 v25, v23, 10, v14 // (op 8992) 


v_and_b32 v14, -64, v1 // (op 9050) 
v_sub_u32 v23, v1, v14 // (op 9050) 

v_and_b32 v14, 4294967232, v1 // (op 9050) 
v_add_u32 v62, v14, v23 // (op 9050) 


v_lshlrev_b32 v14, 2, v1 // (op 9108) 
v_and_b32 v23, -2, v1 // (op 9189) 
v_sub_u32 v63, v1, v23 // (op 9189) 

v_lshlrev_b32 v23, 5, v63 // (op 9189) 

v_lshrrev_b32 v63, 1, v1 // (op 9189) 
v_lshl_add_u32 v100, v63, 6, v23 // (op 9189) 


v_lshlrev_b32 v23, 2, v100 // (op 9189) 

v_lshrrev_b32 v63, 3, v23 // (op 9189) 

v_and_b32 v23, -8, v1 // (op 9218) 
v_sub_u32 v100, v1, v23 // (op 9218) 

v_lshlrev_b32 v23, 4, v100 // (op 9218) 

v_lshrrev_b32 v100, 3, v1 // (op 9218) 
v_lshl_add_u32 v101, v100, 10, v23 // (op 9218) 


v_and_b32 v23, -64, v1 // (op 9247) 
v_sub_u32 v100, v1, v23 // (op 9247) 

v_and_b32 v23, 4294967232, v1 // (op 9247) 
v_add_u32 v102, v23, v100 // (op 9247) 


v_lshlrev_b32 v23, 2, v1 // (op 9276) 
s_mov_b32 s0, 0 // (op 1967) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 1966) 
s_mov_b32 s1, s6 // (op 1966) convert

s_and_b32 s6, s1, 4294967294 // (op 1966) 

s_cmp_lt_u32 s0, s6 // (op 1966) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_1966 // (op 1966) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_1966 if false)
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopTop_KLoop_1966:

s_barrier  // (op 9380) 
s_add_u32 s0, s0, 2 // (op 1968) 



v_mov_b64 v[104:105], s[24:25] // (op 10) convert

v_mov_b32 v100, v19 // (op 10) 
v_mov_b32 v103, v100 // (op 10) 
buffer_load_dwordx4 v[26:29], v103, s[52:55], 0 offen offset:256 // (op 10) Load value
v_add_u32 v103, v103, v104 // (op 10) 
buffer_load_dwordx4 v[30:33], v103, s[52:55], 0 offen offset:256 // (op 10) Load value
v_add_u32 v103, v103, v104 // (op 10) 
ds_read_b64_tr_b4 v[106:107], v3 offset:33792 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[108:109], v3 offset:34304 // (op 72) Transpose load local data 
buffer_load_dwordx4 v[34:37], v103, s[52:55], 0 offen offset:256 // (op 10) Load value
s_barrier  // (op 9393) 
v_add_u32 v103, v103, v104 // (op 10) 



ds_read_b128 v[110:113], v4 // (op 77) Load local data 
ds_read_b128 v[114:117], v4 offset:64 // (op 77) Load local data 
buffer_load_dwordx4 v[38:41], v103, s[52:55], 0 offen offset:256 // (op 10) Load value
s_barrier  // (op 9407) 
v_add_u32 v103, v103, v104 // (op 10) 


buffer_load_dwordx4 v[42:45], v103, s[52:55], 0 offen offset:256 // (op 10) Load value
v_add_u32 v103, v103, v104 // (op 10) 
ds_read_u8 v118, v0 offset:37888 // (op 82) Load local data 
buffer_load_dwordx4 v[46:49], v103, s[52:55], 0 offen offset:256 // (op 10) Load value
s_barrier  // (op 9421) 
v_add_u32 v103, v103, v104 // (op 10) 



ds_read_u8 v119, v5 offset:32768 // (op 87) Load local data 
buffer_load_dwordx4 v[50:53], v103, s[52:55], 0 offen offset:256 // (op 10) Load value



v_add_u32 v103, v103, v104 // (op 10) 
buffer_load_dwordx4 v[54:57], v103, s[52:55], 0 offen offset:256 // (op 10) Load value


ds_read_b128 v[120:123], v4 offset:4096 // (op 375) Load local data 


ds_read_b128 v[124:127], v4 offset:4160 // (op 375) Load local data 
v_mov_b32 v103, v22 // (op 25) 




ds_read_u8 v104, v5 offset:32896 // (op 379) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[106:109], v[110:117], a[0:3], v118, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 90) 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[106:109], v[120:127], a[4:7], v118, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 380) 
ds_read_b128 v[128:131], v4 offset:8192 // (op 551) Load local data 
ds_read_b128 v[132:135], v4 offset:8256 // (op 551) Load local data 
buffer_load_dword v16, v103, s[60:63], 0 offen offset:8 // (op 25) Load value





ds_read_u8 v105, v5 offset:33024 // (op 555) Load local data 
v_mov_b32 v136, v15 // (op 4) 



s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 4) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v137, v136, s1 // (op 4) 


ds_read_b128 v[138:141], v4 offset:12288 // (op 727) Load local data 
ds_read_b128 v[142:145], v4 offset:12352 // (op 727) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[106:109], v[128:135], a[8:11], v118, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 556) 



ds_read_u8 v146, v5 offset:33152 // (op 731) Load local data 
buffer_load_dwordx4 v[58:61], v137, s[48:51], 0 offen // (op 4) Load value






v_mov_b32 v137, v20 // (op 17) 
s_lshl_b32 s1, s28, 3 // (op 17) 
v_add_u32 v147, v137, s1 // (op 17) 


buffer_load_ubyte v24, v147, s[56:59], 0 offen offset: 0 // (op 17) Load value

ds_read_b128 v[148:151], v4 offset:16384 // (op 903) Load local data 
s_barrier  // (op 9444) 
ds_read_b128 v[152:155], v4 offset:16448 // (op 903) Load local data 






ds_read_u8 v147, v5 offset:33280 // (op 907) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[106:109], v[138:145], a[12:15], v118, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 732) 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[106:109], v[148:155], a[16:19], v118, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 908) 
ds_read_b128 v[156:159], v4 offset:20480 // (op 1079) Load local data 
ds_read_b128 v[160:163], v4 offset:20544 // (op 1079) Load local data 
s_waitcnt vmcnt(21)
ds_write_b128 v101, v[64:67] offset:38144 // (op 257) Store local data 



ds_read_u8 v164, v5 offset:33408 // (op 1083) Load local data 
s_waitcnt vmcnt(20)
ds_write_b128 v101, v[68:71] offset:38272 // (op 257) Store local data 



ds_read_b128 v[166:169], v4 offset:24576 // (op 1255) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[106:109], v[156:163], a[20:23], v118, v164 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1084) 
ds_read_b128 v[170:173], v4 offset:24640 // (op 1255) Load local data 
s_waitcnt vmcnt(19)
ds_write_b128 v101, v[72:75] offset:38400 // (op 257) Store local data 



ds_read_u8 v165, v5 offset:33536 // (op 1259) Load local data 
s_waitcnt vmcnt(18)
ds_write_b128 v101, v[76:79] offset:38528 // (op 257) Store local data 



ds_read_b128 v[174:177], v4 offset:28672 // (op 1431) Load local data 
ds_read_b128 v[178:181], v4 offset:28736 // (op 1431) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[106:109], v[166:173], a[24:27], v118, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1260) 



ds_read_u8 v182, v5 offset:33664 // (op 1435) Load local data 
s_waitcnt vmcnt(17)
ds_write_b128 v101, v[80:83] offset:38656 // (op 257) Store local data 



ds_read_b64_tr_b4 v[184:185], v3 offset:33808 // (op 2000) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[106:109], v[174:181], a[28:31], v118, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1436) 
ds_read_b64_tr_b4 v[186:187], v3 offset:34320 // (op 2000) Transpose load local data 




ds_read_u8 v118, v0 offset:37920 // (op 2003) Load local data 
s_waitcnt vmcnt(16)
ds_write_b128 v101, v[84:87] offset:38784 // (op 257) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[184:187], v[110:117], a[32:35], v118, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2327) 
s_waitcnt vmcnt(15)
ds_write_b128 v101, v[88:91] offset:38912 // (op 257) Store local data 


s_waitcnt vmcnt(14)
ds_write_b128 v101, v[92:95] offset:39040 // (op 257) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[184:187], v[120:127], a[36:39], v118, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2281) 


s_barrier  // (op 9456) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[184:187], v[128:135], a[40:43], v118, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2235) 
v_add_u32 v104, 70912, v23 // (op 259) 


s_waitcnt vmcnt(13)
ds_write_b32 v104, v18 // (op 259) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[184:187], v[138:145], a[44:47], v118, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2189) 


s_barrier  // (op 9438) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[184:187], v[148:155], a[48:51], v118, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2143) 
v_add_u32 v183, 71936, v63 // (op 247) 


s_waitcnt vmcnt(12)
ds_write_b128 v183, v[96:99] // (op 247) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[184:187], v[156:163], a[52:55], v118, v164 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2097) 


s_barrier  // (op 9450) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[184:187], v[166:173], a[56:59], v118, v165 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2051) 
v_add_u32 v183, 76032, v102 // (op 258) 


s_waitcnt vmcnt(11)
ds_write_b8 v183, v21 // (op 258) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[184:187], v[174:181], a[60:63], v118, v182 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2005) 





v_mov_b64 v[104:105], s[24:25] // (op 260) convert

v_mov_b32 v106, v100 // (op 260) 
buffer_load_dwordx4 v[64:67], v106, s[52:55], 0 offen offset:384 // (op 260) Load value
v_add_u32 v106, v106, v104 // (op 260) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9435) 
buffer_load_dwordx4 v[68:71], v106, s[52:55], 0 offen offset:384 // (op 260) Load value



v_add_u32 v106, v106, v104 // (op 260) 
v_add_u32 v107, 71936, v7 // (op 249) 
ds_read_b64_tr_b4 v[108:109], v107 // (op 249) Transpose load local data 

buffer_load_dwordx4 v[72:75], v106, s[52:55], 0 offen offset:384 // (op 260) Load value
v_add_u32 v106, v106, v104 // (op 260) 
v_add_u32 v107, 72448, v7 // (op 249) 
ds_read_b64_tr_b4 v[110:111], v107 // (op 249) Transpose load local data 

buffer_load_dwordx4 v[76:79], v106, s[52:55], 0 offen offset:384 // (op 260) Load value
s_barrier  // (op 9441) 
v_add_u32 v106, v106, v104 // (op 260) 



ds_read_b128 v[112:115], v8 offset:38144 // (op 251) Load local data 
ds_read_b128 v[116:119], v8 offset:38208 // (op 251) Load local data 
buffer_load_dwordx4 v[80:83], v106, s[52:55], 0 offen offset:384 // (op 260) Load value
v_add_u32 v106, v106, v104 // (op 260) 
buffer_load_dwordx4 v[84:87], v106, s[52:55], 0 offen offset:384 // (op 260) Load value
v_add_u32 v106, v106, v104 // (op 260) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 9447) 
buffer_load_dwordx4 v[88:91], v106, s[52:55], 0 offen offset:384 // (op 260) Load value


v_add_u32 v106, v106, v104 // (op 260) 
v_add_u32 v107, 76032, v6 // (op 253) 
ds_read_u8 v120, v107 // (op 253) Load local data 

buffer_load_dwordx4 v[92:95], v106, s[52:55], 0 offen offset:384 // (op 260) Load value



s_barrier  // (op 9453) 






v_add_u32 v19, 256, v19 // (op 8755) 
v_add_u32 v100, 70912, v9 // (op 255) 
buffer_load_dword v18, v103, s[60:63], 0 offen offset:12 // (op 262) Load value
ds_read_u8 v104, v100 // (op 255) Load local data 







v_add_u32 v22, 8, v22 // (op 8883) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 246) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 246) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 246) 

s_mul_i32 s1, 3, s6 // (op 246) 

v_add_u32 v100, v136, s1 // (op 246) 


ds_read_b128 v[128:131], v8 offset:42240 // (op 385) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[112:119], a[0:3], v120, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 256) 
ds_read_b128 v[132:135], v8 offset:42304 // (op 385) Load local data 
buffer_load_dwordx4 v[96:99], v100, s[48:51], 0 offen // (op 246) Load value





v_add_u32 v100, 71040, v9 // (op 389) 


ds_read_u8 v103, v100 // (op 389) Load local data 

s_lshl_b32 s1, s28, 2 // (op 261) 



s_mul_i32 s6, 3, s1 // (op 261) 

v_add_u32 v100, v137, s6 // (op 261) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 8691) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 8691) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 8691) 

s_lshl_b32 s1, s6, 1 // (op 8691) 

v_add_u32 v15, v15, s1 // (op 8691) 

ds_read_b128 v[138:141], v8 offset:46336 // (op 561) Load local data 
ds_read_b128 v[142:145], v8 offset:46400 // (op 561) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[128:135], a[4:7], v120, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 381) 



v_add_u32 v136, 71168, v9 // (op 565) 
ds_read_u8 v105, v136 // (op 565) Load local data 

buffer_load_ubyte v21, v100, s[56:59], 0 offen offset: 0 // (op 261) Load value





ds_read_b128 v[146:149], v8 offset:50432 // (op 737) Load local data 
s_barrier  // (op 9396) 
ds_read_b128 v[150:153], v8 offset:50496 // (op 737) Load local data 






s_lshl_b32 s1, s28, 3 // (op 8819) 
v_add_u32 v20, v20, s1 // (op 8819) 

v_add_u32 v100, 71296, v9 // (op 741) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[138:145], a[8:11], v120, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 557) 
ds_read_u8 v106, v100 // (op 741) Load local data 

s_waitcnt vmcnt(21)
ds_write_b128 v25, v[26:29] // (op 13) Store local data 



ds_read_b128 v[154:157], v8 offset:54528 // (op 913) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[146:153], a[12:15], v120, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 733) 
ds_read_b128 v[158:161], v8 offset:54592 // (op 913) Load local data 
s_waitcnt vmcnt(20)
ds_write_b128 v25, v[30:33] offset:128 // (op 13) Store local data 



v_add_u32 v100, 71424, v9 // (op 917) 
ds_read_u8 v107, v100 // (op 917) Load local data 

s_waitcnt vmcnt(19)
ds_write_b128 v25, v[34:37] offset:256 // (op 13) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[154:161], a[16:19], v120, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 909) 
ds_read_b128 v[162:165], v8 offset:58624 // (op 1089) Load local data 
ds_read_b128 v[166:169], v8 offset:58688 // (op 1089) Load local data 
s_waitcnt vmcnt(18)
ds_write_b128 v25, v[38:41] offset:384 // (op 13) Store local data 



v_add_u32 v100, 71552, v9 // (op 1093) 
ds_read_u8 v121, v100 // (op 1093) Load local data 

s_waitcnt vmcnt(17)
ds_write_b128 v25, v[42:45] offset:512 // (op 13) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[162:169], a[20:23], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1085) 
ds_read_b128 v[170:173], v8 offset:62720 // (op 1265) Load local data 
ds_read_b128 v[174:177], v8 offset:62784 // (op 1265) Load local data 
s_waitcnt vmcnt(16)
ds_write_b128 v25, v[46:49] offset:640 // (op 13) Store local data 



v_add_u32 v100, 71680, v9 // (op 1269) 
ds_read_u8 v122, v100 // (op 1269) Load local data 

s_waitcnt vmcnt(15)
ds_write_b128 v25, v[50:53] offset:768 // (op 13) Store local data 



v_add_u32 v100, 66816, v8 // (op 1441) 
ds_read_b128 v[178:181], v100 // (op 1441) Load local data 

v_add_u32 v100, 66880, v8 // (op 1441) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[170:177], a[24:27], v120, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1261) 
ds_read_b128 v[182:185], v100 // (op 1441) Load local data 

s_waitcnt vmcnt(14)
ds_write_b128 v25, v[54:57] offset:896 // (op 13) Store local data 



s_barrier  // (op 9517) 
s_barrier  // (op 9424) 
v_add_u32 v100, 71808, v9 // (op 1445) 



ds_read_u8 v123, v100 // (op 1445) Load local data 

s_waitcnt vmcnt(13)
ds_write_b32 v14, v16 offset:32768 // (op 30) Store local data 



s_barrier  // (op 9521) 
s_barrier  // (op 9383) 
v_add_u32 v100, 71952, v7 // (op 2008) 



ds_read_b64_tr_b4 v[124:125], v100 // (op 2008) Transpose load local data 

v_add_u32 v100, 72464, v7 // (op 2008) 
ds_read_b64_tr_b4 v[126:127], v100 // (op 2008) Transpose load local data 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[178:185], a[28:31], v120, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1437) 




s_barrier  // (op 9553) 
v_add_u32 v100, 76064, v6 // (op 2011) 
ds_read_u8 v120, v100 // (op 2011) Load local data 

s_waitcnt vmcnt(12)
ds_write_b128 v17, v[58:61] offset:33792 // (op 6) Store local data 
s_barrier  // (op 9557) 
s_barrier  // (op 9410) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[124:127], v[112:119], a[32:35], v120, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2328) 





s_waitcnt vmcnt(11)
ds_write_b8 v62, v24 offset:37888 // (op 21) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[124:127], v[128:135], a[36:39], v120, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2282) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[124:127], v[138:145], a[40:43], v120, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2236) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:127], v[146:153], a[44:47], v120, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2190) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:127], v[154:161], a[48:51], v120, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2144) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[162:169], a[52:55], v120, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2098) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[170:177], a[56:59], v120, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2052) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[178:185], a[60:63], v120, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2006) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 1966) 
s_mov_b32 s1, s6 // (op 1966) convert

s_and_b32 s6, s1, 4294967294 // (op 1966) 

s_cmp_lt_u32 s0, s6 // (op 1966) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopTop_KLoop_1966 // (op 1966) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopTop_KLoop_1966 if true)
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_1966:

















v_and_b32 v14, -2, v1 // (op 8639) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8639) 
v_sub_u32 v15, v1, v14 // (op 8639) 

s_lshr_b32 s6, s0, 1 // (op 8639) 

v_lshlrev_b32 v14, 5, v15 // (op 8639) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 8639) 

v_mov_b32 v17, 0 // (op 8639) convert
v_mov_b32 v16, v15 // (op 8639) convert

v_lshrrev_b32 v14, 1, v1 // (op 8639) 
v_lshl_add_u32 v15, s6, 8, v14 // (op 8639) 


v_mul_lo_u32 v23, v15, s13 // (op 8639) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 8639) most significant: high of low * low
v_mul_lo_u32 v22, v15, s12 // (op 8639) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 8639) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 8639) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 8639) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 8639) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8639) 

v_mov_b32 v14, v16 // (op 8639) convert

s_mul_i32 s1, s12, s23 // (op 8641) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8641) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8641) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8641) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8641) most significant: sum
s_add_u32 s1, s1, s7 // (op 8641) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 8641) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 8641) 

s_mov_b32 s48, s20 // (op 8641) reg expression
s_mov_b32 s49, s21 // (op 8641) reg expression
s_mov_b32 s50, s0 // (op 8641) reg expression
s_mov_b32 s51, 131072 // (op 8641) call()

v_and_b32 v15, -8, v1 // (op 8721) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8721) 
v_sub_u32 v16, v1, v15 // (op 8721) 

s_lshr_b32 s6, s0, 1 // (op 8721) 

v_lshlrev_b32 v15, 4, v16 // (op 8721) 

v_lshl_add_u32 v16, s6, 8, v15 // (op 8721) 


v_and_b32 v15, 4294967288, v1 // (op 8721) 
v_lshl_add_u32 v17, s17, 8, v15 // (op 8721) 

v_mul_lo_u32 v15, v17, s24 // (op 8721) 

v_add_u32 v17, v16, v15 // (op 8721) 


s_mul_i32 s1, s24, s9 // (op 8723) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 8723) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 8723) least significant
s_mul_hi_u32 s7, s24, s8 // (op 8723) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8723) most significant: sum
s_add_u32 s1, s1, s7 // (op 8723) most significant: sum


s_mov_b32 s52, s26 // (op 8723) reg expression
s_mov_b32 s53, s27 // (op 8723) reg expression
s_mov_b32 s54, s0 // (op 8723) reg expression
s_mov_b32 s55, 131072 // (op 8723) call()

v_and_b32 v15, -64, v1 // (op 8785) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8785) 
v_sub_u32 v16, v1, v15 // (op 8785) 

s_lshr_b32 s6, s0, 1 // (op 8785) 

v_lshl_add_u32 v15, s16, 6, v16 // (op 8785) 

v_lshrrev_b32 v16, 6, v1 // (op 8785) 
v_lshl_add_u32 v19, s6, 3, v16 // (op 8785) 


v_mul_lo_u32 v16, v19, s28 // (op 8785) 

v_add_u32 v19, v15, v16 // (op 8785) 


s_mul_i32 s1, s28, s33 // (op 8787) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 8787) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8787) least significant
s_mul_hi_u32 s7, s28, s32 // (op 8787) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8787) most significant: sum
s_add_u32 s1, s1, s7 // (op 8787) most significant: sum


s_mov_b32 s56, s30 // (op 8787) reg expression
s_mov_b32 s57, s31 // (op 8787) reg expression
s_mov_b32 s58, s0 // (op 8787) reg expression
s_mov_b32 s59, 131072 // (op 8787) call()

v_lshl_add_u32 v20, s17, 8, v1 // (op 8849) 
v_mul_lo_u32 v15, v20, s34 // (op 8849) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 8849) 
s_lshr_b32 s6, s0, 1 // (op 8849) 

v_lshl_add_u32 v16, s6, 3, v15 // (op 8849) 


s_mul_i32 s1, s34, s9 // (op 8851) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 8851) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 8851) least significant
s_mul_hi_u32 s7, s34, s8 // (op 8851) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8851) most significant: sum
s_add_u32 s1, s1, s7 // (op 8851) most significant: sum


s_mov_b32 s60, s36 // (op 8851) reg expression
s_mov_b32 s61, s37 // (op 8851) reg expression
s_mov_b32 s62, s0 // (op 8851) reg expression
s_mov_b32 s63, 131072 // (op 8851) call()

v_and_b32 v15, -2, v1 // (op 8947) 
v_sub_u32 v20, v1, v15 // (op 8947) 

v_lshlrev_b32 v15, 5, v20 // (op 8947) 

v_lshrrev_b32 v20, 1, v1 // (op 8947) 
v_lshl_add_u32 v22, v20, 6, v15 // (op 8947) 


v_lshlrev_b32 v15, 2, v22 // (op 8947) 

v_lshrrev_b32 v20, 3, v15 // (op 8947) 

v_and_b32 v15, -8, v1 // (op 9005) 
v_sub_u32 v22, v1, v15 // (op 9005) 

v_lshlrev_b32 v15, 4, v22 // (op 9005) 

v_lshrrev_b32 v22, 3, v1 // (op 9005) 
v_lshl_add_u32 v23, v22, 10, v15 // (op 9005) 


v_and_b32 v15, -64, v1 // (op 9063) 
v_sub_u32 v22, v1, v15 // (op 9063) 

v_and_b32 v15, 4294967232, v1 // (op 9063) 
v_add_u32 v24, v15, v22 // (op 9063) 


v_lshlrev_b32 v15, 2, v1 // (op 9121) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1972) 
s_and_b32 s6, s0, 4294967294 // (op 1972) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 1971) 
s_mov_b32 s7, s0 // (op 1971) convert

s_cmp_lt_u32 s6, s7 // (op 1971) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_1971 // (op 1971) Condition: Top (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_1971 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_1971:


s_add_u32 s6, s6, 1 // (op 1973) 
v_mov_b32 v18, v14 // (op 180) 
v_mov_b32 v21, v18 // (op 180) reg expression

buffer_load_dwordx4 v[26:29], v21, s[48:51], 0 offen // (op 180) Load value


s_lshl_b64 s[0:1], s[12:13], 7 // (op 8629) 
s_lshl_b64 s[10:11], s[0:1], 2 // (op 8629) 

s_lshr_b64 s[0:1], s[10:11], 3 // (op 8629) 

v_add_u32 v14, v14, s0 // (op 8629) 


s_barrier  // (op 9386) 
v_mov_b64 v[30:31], s[24:25] // (op 194) convert




v_mov_b32 v18, v17 // (op 194) 
v_mov_b32 v25, v18 // (op 194) 
buffer_load_dwordx4 v[32:35], v25, s[52:55], 0 offen // (op 194) Load value
v_add_u32 v25, v25, v30 // (op 194) 
buffer_load_dwordx4 v[36:39], v25, s[52:55], 0 offen // (op 194) Load value
v_add_u32 v25, v25, v30 // (op 194) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[26:29] offset:33792 // (op 181) Store local data 
buffer_load_dwordx4 v[40:43], v25, s[52:55], 0 offen // (op 194) Load value

v_add_u32 v25, v25, v30 // (op 194) 
buffer_load_dwordx4 v[44:47], v25, s[52:55], 0 offen // (op 194) Load value
v_add_u32 v25, v25, v30 // (op 194) 
buffer_load_dwordx4 v[48:51], v25, s[52:55], 0 offen // (op 194) Load value
v_add_u32 v25, v25, v30 // (op 194) 
buffer_load_dwordx4 v[52:55], v25, s[52:55], 0 offen // (op 194) Load value
v_add_u32 v25, v25, v30 // (op 194) 
buffer_load_dwordx4 v[56:59], v25, s[52:55], 0 offen // (op 194) Load value
v_add_u32 v25, v25, v30 // (op 194) 
buffer_load_dwordx4 v[60:63], v25, s[52:55], 0 offen // (op 194) Load value



v_add_u32 v17, 128, v17 // (op 8713) 


s_barrier  // (op 9399) 
v_mov_b32 v18, v19 // (op 195) 



v_mov_b32 v21, v18 // (op 195) reg expression

s_waitcnt vmcnt(7)
ds_write_b128 v23, v[32:35] // (op 191) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v23, v[36:39] offset:128 // (op 191) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v23, v[40:43] offset:256 // (op 191) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v23, v[44:47] offset:384 // (op 191) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v23, v[48:51] offset:512 // (op 191) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v23, v[52:55] offset:640 // (op 191) Store local data 
buffer_load_ubyte v22, v21, s[56:59], 0 offen offset: 0 // (op 195) Load value


s_lshl_b32 s7, s28, 2 // (op 8777) 
v_add_u32 v19, v19, s7 // (op 8777) 

s_waitcnt vmcnt(2)
ds_write_b128 v23, v[56:59] offset:768 // (op 191) Store local data 


v_mov_b32 v18, v16 // (op 196) 

buffer_load_dword v21, v18, s[60:63], 0 offen // (op 196) Load value

v_add_u32 v16, 4, v16 // (op 8841) 
s_waitcnt vmcnt(2)
ds_write_b128 v23, v[60:63] offset:896 // (op 191) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 9389) 



s_barrier  // (op 9413) 
ds_read_b64_tr_b4 v[26:27], v3 offset:33792 // (op 183) Transpose load local data 



ds_read_b64_tr_b4 v[28:29], v3 offset:34304 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v22 offset:37888 // (op 192) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9403) 



s_barrier  // (op 9427) 
ds_read_b128 v[30:33], v4 // (op 185) Load local data 



ds_read_b128 v[34:37], v4 offset:64 // (op 185) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v15, v21 offset:32768 // (op 193) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9417) 


ds_read_u8 v18, v0 offset:37888 // (op 187) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9431) 



ds_read_u8 v25, v5 offset:32768 // (op 189) Load local data 



ds_read_b128 v[38:41], v4 offset:4096 // (op 354) Load local data 
ds_read_b128 v[42:45], v4 offset:4160 // (op 354) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:37], a[0:3], v18, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 190) 



ds_read_u8 v21, v5 offset:32896 // (op 358) Load local data 



ds_read_b128 v[46:49], v4 offset:8192 // (op 530) Load local data 
ds_read_b128 v[50:53], v4 offset:8256 // (op 530) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[38:45], a[4:7], v18, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 359) 



ds_read_u8 v22, v5 offset:33024 // (op 534) Load local data 



ds_read_b128 v[54:57], v4 offset:12288 // (op 706) Load local data 
ds_read_b128 v[58:61], v4 offset:12352 // (op 706) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:29], v[46:53], a[8:11], v18, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 535) 



ds_read_u8 v62, v5 offset:33152 // (op 710) Load local data 



ds_read_b128 v[64:67], v4 offset:16384 // (op 882) Load local data 
ds_read_b128 v[68:71], v4 offset:16448 // (op 882) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:29], v[54:61], a[12:15], v18, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 711) 



ds_read_u8 v63, v5 offset:33280 // (op 886) Load local data 



ds_read_b128 v[72:75], v4 offset:20480 // (op 1058) Load local data 
ds_read_b128 v[76:79], v4 offset:20544 // (op 1058) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:29], v[64:71], a[16:19], v18, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 887) 



ds_read_u8 v80, v5 offset:33408 // (op 1062) Load local data 



ds_read_b128 v[82:85], v4 offset:24576 // (op 1234) Load local data 
ds_read_b128 v[86:89], v4 offset:24640 // (op 1234) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:29], v[72:79], a[20:23], v18, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1063) 



ds_read_u8 v81, v5 offset:33536 // (op 1238) Load local data 



ds_read_b128 v[90:93], v4 offset:28672 // (op 1410) Load local data 
ds_read_b128 v[94:97], v4 offset:28736 // (op 1410) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[82:89], a[24:27], v18, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1239) 



ds_read_u8 v98, v5 offset:33664 // (op 1414) Load local data 



ds_read_b64_tr_b4 v[100:101], v3 offset:33808 // (op 1760) Transpose load local data 
ds_read_b64_tr_b4 v[102:103], v3 offset:34320 // (op 1760) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:29], v[90:97], a[28:31], v18, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1415) 




ds_read_u8 v18, v0 offset:37920 // (op 1763) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[100:103], v[30:37], a[32:35], v18, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1765) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[100:103], v[38:45], a[36:39], v18, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1796) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:103], v[46:53], a[40:43], v18, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1827) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:103], v[54:61], a[44:47], v18, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1858) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[100:103], v[64:71], a[48:51], v18, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1889) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[100:103], v[72:79], a[52:55], v18, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1920) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[100:103], v[82:89], a[56:59], v18, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1951) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[100:103], v[90:97], a[60:63], v18, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1982) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 1971) 
s_mov_b32 s7, s0 // (op 1971) convert

s_cmp_lt_u32 s6, s7 // (op 1971) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_1971 // (op 1971) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_1971 if true)
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_1971:























s_barrier  // (op 9541) 









v_cmp_eq_f32 vcc, 0.00000, v10 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 

v_lshrrev_b32 v6, 16, v14 // (op 56) 
v_lshrrev_b32 v7, 16, v15 // (op 56) 
v_lshrrev_b32 v8, 16, v16 // (op 56) 
v_lshrrev_b32 v9, 16, v17 // (op 56) 


v_pack_b32_f16 v4, v6, v7 // (op 58) 
v_pack_b32_f16 v5, v8, v9 // (op 58) 

v_mov_b32 v0, v13 // (op 58) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v6, a4 // (op 3577) 
v_accvgpr_read v7, a5 // (op 3577) 
v_accvgpr_read v8, a6 // (op 3577) 
v_accvgpr_read v9, a7 // (op 3577) 
v_mul_f32 v14, v11, v6 // (op 3577) 
v_mul_f32 v15, v11, v7 // (op 3577) 
v_mul_f32 v16, v11, v8 // (op 3577) 
v_mul_f32 v17, v11, v9 // (op 3577) 

v_lshrrev_b32 v6, 16, v14 // (op 3579) 
v_lshrrev_b32 v7, 16, v15 // (op 3579) 
v_lshrrev_b32 v8, 16, v16 // (op 3579) 
v_lshrrev_b32 v9, 16, v17 // (op 3579) 


v_pack_b32_f16 v4, v6, v7 // (op 3580) 
v_pack_b32_f16 v5, v8, v9 // (op 3580) 

s_lshl_b32 s0, s4, 6 // (op 3580) 
v_add_u32 v0, v13, s0 // (op 3580) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3580) Store value



v_accvgpr_read v6, a8 // (op 3591) 
v_accvgpr_read v7, a9 // (op 3591) 
v_accvgpr_read v8, a10 // (op 3591) 
v_accvgpr_read v9, a11 // (op 3591) 
v_mul_f32 v14, v11, v6 // (op 3591) 
v_mul_f32 v15, v11, v7 // (op 3591) 
v_mul_f32 v16, v11, v8 // (op 3591) 
v_mul_f32 v17, v11, v9 // (op 3591) 

v_lshrrev_b32 v6, 16, v14 // (op 3593) 
v_lshrrev_b32 v7, 16, v15 // (op 3593) 
v_lshrrev_b32 v8, 16, v16 // (op 3593) 
v_lshrrev_b32 v9, 16, v17 // (op 3593) 


v_pack_b32_f16 v4, v6, v7 // (op 3594) 
v_pack_b32_f16 v5, v8, v9 // (op 3594) 

s_lshl_b32 s0, s4, 7 // (op 3594) 
v_add_u32 v0, v13, s0 // (op 3594) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3594) Store value



v_accvgpr_read v6, a12 // (op 3605) 
v_accvgpr_read v7, a13 // (op 3605) 
v_accvgpr_read v8, a14 // (op 3605) 
v_accvgpr_read v9, a15 // (op 3605) 
v_mul_f32 v14, v11, v6 // (op 3605) 
v_mul_f32 v15, v11, v7 // (op 3605) 
v_mul_f32 v16, v11, v8 // (op 3605) 
v_mul_f32 v17, v11, v9 // (op 3605) 

v_lshrrev_b32 v6, 16, v14 // (op 3607) 
v_lshrrev_b32 v7, 16, v15 // (op 3607) 
v_lshrrev_b32 v8, 16, v16 // (op 3607) 
v_lshrrev_b32 v9, 16, v17 // (op 3607) 


v_pack_b32_f16 v4, v6, v7 // (op 3608) 
v_pack_b32_f16 v5, v8, v9 // (op 3608) 

s_lshl_b32 s0, s4, 6 // (op 3608) 
s_mul_i32 s1, 3, s0 // (op 3608) 

v_add_u32 v0, v13, s1 // (op 3608) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3608) Store value



v_accvgpr_read v6, a16 // (op 3619) 
v_accvgpr_read v7, a17 // (op 3619) 
v_accvgpr_read v8, a18 // (op 3619) 
v_accvgpr_read v9, a19 // (op 3619) 
v_mul_f32 v14, v11, v6 // (op 3619) 
v_mul_f32 v15, v11, v7 // (op 3619) 
v_mul_f32 v16, v11, v8 // (op 3619) 
v_mul_f32 v17, v11, v9 // (op 3619) 

v_lshrrev_b32 v6, 16, v14 // (op 3621) 
v_lshrrev_b32 v7, 16, v15 // (op 3621) 
v_lshrrev_b32 v8, 16, v16 // (op 3621) 
v_lshrrev_b32 v9, 16, v17 // (op 3621) 


v_pack_b32_f16 v4, v6, v7 // (op 3622) 
v_pack_b32_f16 v5, v8, v9 // (op 3622) 

s_lshl_b32 s0, s4, 8 // (op 3622) 
v_add_u32 v0, v13, s0 // (op 3622) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3622) Store value



v_accvgpr_read v6, a20 // (op 3633) 
v_accvgpr_read v7, a21 // (op 3633) 
v_accvgpr_read v8, a22 // (op 3633) 
v_accvgpr_read v9, a23 // (op 3633) 
v_mul_f32 v14, v11, v6 // (op 3633) 
v_mul_f32 v15, v11, v7 // (op 3633) 
v_mul_f32 v16, v11, v8 // (op 3633) 
v_mul_f32 v17, v11, v9 // (op 3633) 

v_lshrrev_b32 v6, 16, v14 // (op 3635) 
v_lshrrev_b32 v7, 16, v15 // (op 3635) 
v_lshrrev_b32 v8, 16, v16 // (op 3635) 
v_lshrrev_b32 v9, 16, v17 // (op 3635) 


v_pack_b32_f16 v4, v6, v7 // (op 3636) 
v_pack_b32_f16 v5, v8, v9 // (op 3636) 

s_lshl_b32 s0, s4, 6 // (op 3636) 
s_mul_i32 s1, 5, s0 // (op 3636) 

v_add_u32 v0, v13, s1 // (op 3636) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3636) Store value



v_accvgpr_read v6, a24 // (op 3647) 
v_accvgpr_read v7, a25 // (op 3647) 
v_accvgpr_read v8, a26 // (op 3647) 
v_accvgpr_read v9, a27 // (op 3647) 
v_mul_f32 v14, v11, v6 // (op 3647) 
v_mul_f32 v15, v11, v7 // (op 3647) 
v_mul_f32 v16, v11, v8 // (op 3647) 
v_mul_f32 v17, v11, v9 // (op 3647) 

v_lshrrev_b32 v6, 16, v14 // (op 3649) 
v_lshrrev_b32 v7, 16, v15 // (op 3649) 
v_lshrrev_b32 v8, 16, v16 // (op 3649) 
v_lshrrev_b32 v9, 16, v17 // (op 3649) 


v_pack_b32_f16 v4, v6, v7 // (op 3650) 
v_pack_b32_f16 v5, v8, v9 // (op 3650) 

s_lshl_b32 s0, s4, 6 // (op 3650) 
s_mul_i32 s1, 6, s0 // (op 3650) 

v_add_u32 v0, v13, s1 // (op 3650) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3650) Store value



v_accvgpr_read v6, a28 // (op 3661) 
v_accvgpr_read v7, a29 // (op 3661) 
v_accvgpr_read v8, a30 // (op 3661) 
v_accvgpr_read v9, a31 // (op 3661) 
v_mul_f32 v14, v11, v6 // (op 3661) 
v_mul_f32 v15, v11, v7 // (op 3661) 
v_mul_f32 v16, v11, v8 // (op 3661) 
v_mul_f32 v17, v11, v9 // (op 3661) 

v_lshrrev_b32 v6, 16, v14 // (op 3663) 
v_lshrrev_b32 v7, 16, v15 // (op 3663) 
v_lshrrev_b32 v8, 16, v16 // (op 3663) 
v_lshrrev_b32 v9, 16, v17 // (op 3663) 


v_pack_b32_f16 v4, v6, v7 // (op 3664) 
v_pack_b32_f16 v5, v8, v9 // (op 3664) 

s_lshl_b32 s0, s4, 6 // (op 3664) 
s_mul_i32 s1, 7, s0 // (op 3664) 

v_add_u32 v0, v13, s1 // (op 3664) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3664) Store value



v_accvgpr_read v6, a32 // (op 3717) 
v_accvgpr_read v7, a33 // (op 3717) 
v_accvgpr_read v8, a34 // (op 3717) 
v_accvgpr_read v9, a35 // (op 3717) 
v_mul_f32 v14, v11, v6 // (op 3717) 
v_mul_f32 v15, v11, v7 // (op 3717) 
v_mul_f32 v16, v11, v8 // (op 3717) 
v_mul_f32 v17, v11, v9 // (op 3717) 

v_lshrrev_b32 v6, 16, v14 // (op 3719) 
v_lshrrev_b32 v7, 16, v15 // (op 3719) 
v_lshrrev_b32 v8, 16, v16 // (op 3719) 
v_lshrrev_b32 v9, 16, v17 // (op 3719) 


v_pack_b32_f16 v4, v6, v7 // (op 3721) 
v_pack_b32_f16 v5, v8, v9 // (op 3721) 

v_add_u32 v0, 64, v13 // (op 3721) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3721) Store value



v_accvgpr_read v6, a36 // (op 3739) 
v_accvgpr_read v7, a37 // (op 3739) 
v_accvgpr_read v8, a38 // (op 3739) 
v_accvgpr_read v9, a39 // (op 3739) 
v_mul_f32 v14, v11, v6 // (op 3739) 
v_mul_f32 v15, v11, v7 // (op 3739) 
v_mul_f32 v16, v11, v8 // (op 3739) 
v_mul_f32 v17, v11, v9 // (op 3739) 

v_lshrrev_b32 v6, 16, v14 // (op 3741) 
v_lshrrev_b32 v7, 16, v15 // (op 3741) 
v_lshrrev_b32 v8, 16, v16 // (op 3741) 
v_lshrrev_b32 v9, 16, v17 // (op 3741) 


v_pack_b32_f16 v4, v6, v7 // (op 3723) 
v_pack_b32_f16 v5, v8, v9 // (op 3723) 

s_lshl_b32 s0, s4, 6 // (op 3723) 
s_add_u32 s0, s0, 64 // (op 3723) 
v_add_u32 v0, v13, s0 // (op 3723) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3723) Store value



v_accvgpr_read v6, a40 // (op 3745) 
v_accvgpr_read v7, a41 // (op 3745) 
v_accvgpr_read v8, a42 // (op 3745) 
v_accvgpr_read v9, a43 // (op 3745) 
v_mul_f32 v14, v11, v6 // (op 3745) 
v_mul_f32 v15, v11, v7 // (op 3745) 
v_mul_f32 v16, v11, v8 // (op 3745) 
v_mul_f32 v17, v11, v9 // (op 3745) 

v_lshrrev_b32 v6, 16, v14 // (op 3747) 
v_lshrrev_b32 v7, 16, v15 // (op 3747) 
v_lshrrev_b32 v8, 16, v16 // (op 3747) 
v_lshrrev_b32 v9, 16, v17 // (op 3747) 


v_pack_b32_f16 v4, v6, v7 // (op 3725) 
v_pack_b32_f16 v5, v8, v9 // (op 3725) 

s_lshl_b32 s0, s4, 7 // (op 3725) 
s_add_u32 s0, s0, 64 // (op 3725) 
v_add_u32 v0, v13, s0 // (op 3725) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3725) Store value



v_accvgpr_read v6, a44 // (op 3751) 
v_accvgpr_read v7, a45 // (op 3751) 
v_accvgpr_read v8, a46 // (op 3751) 
v_accvgpr_read v9, a47 // (op 3751) 
v_mul_f32 v14, v11, v6 // (op 3751) 
v_mul_f32 v15, v11, v7 // (op 3751) 
v_mul_f32 v16, v11, v8 // (op 3751) 
v_mul_f32 v17, v11, v9 // (op 3751) 

v_lshrrev_b32 v6, 16, v14 // (op 3753) 
v_lshrrev_b32 v7, 16, v15 // (op 3753) 
v_lshrrev_b32 v8, 16, v16 // (op 3753) 
v_lshrrev_b32 v9, 16, v17 // (op 3753) 


v_pack_b32_f16 v4, v6, v7 // (op 3727) 
v_pack_b32_f16 v5, v8, v9 // (op 3727) 

s_lshl_b32 s0, s4, 6 // (op 3727) 
s_mul_i32 s1, 3, s0 // (op 3727) 
s_add_u32 s1, s1, 64 // (op 3727) 

v_add_u32 v0, v13, s1 // (op 3727) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3727) Store value



v_accvgpr_read v6, a48 // (op 3757) 
v_accvgpr_read v7, a49 // (op 3757) 
v_accvgpr_read v8, a50 // (op 3757) 
v_accvgpr_read v9, a51 // (op 3757) 
v_mul_f32 v14, v11, v6 // (op 3757) 
v_mul_f32 v15, v11, v7 // (op 3757) 
v_mul_f32 v16, v11, v8 // (op 3757) 
v_mul_f32 v17, v11, v9 // (op 3757) 

v_lshrrev_b32 v6, 16, v14 // (op 3759) 
v_lshrrev_b32 v7, 16, v15 // (op 3759) 
v_lshrrev_b32 v8, 16, v16 // (op 3759) 
v_lshrrev_b32 v9, 16, v17 // (op 3759) 


v_pack_b32_f16 v4, v6, v7 // (op 3729) 
v_pack_b32_f16 v5, v8, v9 // (op 3729) 

s_lshl_b32 s0, s4, 8 // (op 3729) 
s_add_u32 s0, s0, 64 // (op 3729) 
v_add_u32 v0, v13, s0 // (op 3729) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3729) Store value



v_accvgpr_read v6, a52 // (op 3763) 
v_accvgpr_read v7, a53 // (op 3763) 
v_accvgpr_read v8, a54 // (op 3763) 
v_accvgpr_read v9, a55 // (op 3763) 
v_mul_f32 v14, v11, v6 // (op 3763) 
v_mul_f32 v15, v11, v7 // (op 3763) 
v_mul_f32 v16, v11, v8 // (op 3763) 
v_mul_f32 v17, v11, v9 // (op 3763) 

v_lshrrev_b32 v6, 16, v14 // (op 3765) 
v_lshrrev_b32 v7, 16, v15 // (op 3765) 
v_lshrrev_b32 v8, 16, v16 // (op 3765) 
v_lshrrev_b32 v9, 16, v17 // (op 3765) 


v_pack_b32_f16 v4, v6, v7 // (op 3731) 
v_pack_b32_f16 v5, v8, v9 // (op 3731) 

s_lshl_b32 s0, s4, 6 // (op 3731) 
s_mul_i32 s1, 5, s0 // (op 3731) 
s_add_u32 s1, s1, 64 // (op 3731) 

v_add_u32 v0, v13, s1 // (op 3731) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3731) Store value



v_accvgpr_read v6, a56 // (op 3769) 
v_accvgpr_read v7, a57 // (op 3769) 
v_accvgpr_read v8, a58 // (op 3769) 
v_accvgpr_read v9, a59 // (op 3769) 
v_mul_f32 v14, v11, v6 // (op 3769) 
v_mul_f32 v15, v11, v7 // (op 3769) 
v_mul_f32 v16, v11, v8 // (op 3769) 
v_mul_f32 v17, v11, v9 // (op 3769) 

v_lshrrev_b32 v6, 16, v14 // (op 3771) 
v_lshrrev_b32 v7, 16, v15 // (op 3771) 
v_lshrrev_b32 v8, 16, v16 // (op 3771) 
v_lshrrev_b32 v9, 16, v17 // (op 3771) 


v_pack_b32_f16 v4, v6, v7 // (op 3733) 
v_pack_b32_f16 v5, v8, v9 // (op 3733) 

s_lshl_b32 s0, s4, 6 // (op 3733) 
s_mul_i32 s1, 6, s0 // (op 3733) 
s_add_u32 s1, s1, 64 // (op 3733) 

v_add_u32 v0, v13, s1 // (op 3733) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3733) Store value



v_accvgpr_read v6, a60 // (op 3775) 
v_accvgpr_read v7, a61 // (op 3775) 
v_accvgpr_read v8, a62 // (op 3775) 
v_accvgpr_read v9, a63 // (op 3775) 
v_mul_f32 v14, v11, v6 // (op 3775) 
v_mul_f32 v15, v11, v7 // (op 3775) 
v_mul_f32 v16, v11, v8 // (op 3775) 
v_mul_f32 v17, v11, v9 // (op 3775) 

v_lshrrev_b32 v6, 16, v14 // (op 3777) 
v_lshrrev_b32 v7, 16, v15 // (op 3777) 
v_lshrrev_b32 v8, 16, v16 // (op 3777) 
v_lshrrev_b32 v9, 16, v17 // (op 3777) 


v_pack_b32_f16 v4, v6, v7 // (op 3735) 
v_pack_b32_f16 v5, v8, v9 // (op 3735) 

s_lshl_b32 s0, s4, 6 // (op 3735) 
s_mul_i32 s1, 7, s0 // (op 3735) 
s_add_u32 s1, s1, 64 // (op 3735) 

v_add_u32 v0, v13, s1 // (op 3735) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 3735) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_135:


v_add_u32 v0, 64, v12 // (op 4113) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4113) Load value

v_accvgpr_read v4, a32 // (op 4092) 
v_accvgpr_read v5, a33 // (op 4092) 
v_accvgpr_read v6, a34 // (op 4092) 
v_accvgpr_read v7, a35 // (op 4092) 
v_mul_f32 v14, v11, v4 // (op 4092) 
v_mul_f32 v15, v11, v5 // (op 4092) 
v_mul_f32 v16, v11, v6 // (op 4092) 
v_mul_f32 v17, v11, v7 // (op 4092) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4112) 
v_lshrrev_b32 v7, 16, v8 // (op 4112) 
v_lshlrev_b32 v6, 16, v6 // (op 4112) 
v_lshlrev_b32 v7, 16, v7 // (op 4112) 
v_mul_f32 v18, v10, v6 // (op 4112) 
v_mul_f32 v19, v10, v7 // (op 4112) 
v_and_b32 v6, 65535, v9 // (op 4112) 
v_lshrrev_b32 v7, 16, v9 // (op 4112) 
v_lshlrev_b32 v6, 16, v6 // (op 4112) 
v_lshlrev_b32 v7, 16, v7 // (op 4112) 
v_mul_f32 v20, v10, v6 // (op 4112) 
v_mul_f32 v21, v10, v7 // (op 4112) 


v_add_f32 v6, v18, v14 // (op 4093) 
v_add_f32 v7, v19, v15 // (op 4093) 
v_add_f32 v8, v20, v16 // (op 4093) 
v_add_f32 v9, v21, v17 // (op 4093) 


v_lshrrev_b32 v14, 16, v6 // (op 4094) 
v_lshrrev_b32 v15, 16, v7 // (op 4094) 
v_lshrrev_b32 v16, 16, v8 // (op 4094) 
v_lshrrev_b32 v17, 16, v9 // (op 4094) 


v_pack_b32_f16 v8, v14, v15 // (op 4096) 
v_pack_b32_f16 v9, v16, v17 // (op 4096) 

v_add_u32 v0, 64, v13 // (op 4096) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4096) Store value





s_lshl_b32 s0, s2, 6 // (op 4118) 
s_add_u32 s0, s0, 64 // (op 4118) 
v_add_u32 v0, v12, s0 // (op 4118) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4118) Load value

v_accvgpr_read v4, a36 // (op 4150) 
v_accvgpr_read v5, a37 // (op 4150) 
v_accvgpr_read v6, a38 // (op 4150) 
v_accvgpr_read v7, a39 // (op 4150) 
v_mul_f32 v14, v11, v4 // (op 4150) 
v_mul_f32 v15, v11, v5 // (op 4150) 
v_mul_f32 v16, v11, v6 // (op 4150) 
v_mul_f32 v17, v11, v7 // (op 4150) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4115) 
v_lshrrev_b32 v7, 16, v8 // (op 4115) 
v_lshlrev_b32 v6, 16, v6 // (op 4115) 
v_lshlrev_b32 v7, 16, v7 // (op 4115) 
v_mul_f32 v18, v10, v6 // (op 4115) 
v_mul_f32 v19, v10, v7 // (op 4115) 
v_and_b32 v6, 65535, v9 // (op 4115) 
v_lshrrev_b32 v7, 16, v9 // (op 4115) 
v_lshlrev_b32 v6, 16, v6 // (op 4115) 
v_lshlrev_b32 v7, 16, v7 // (op 4115) 
v_mul_f32 v20, v10, v6 // (op 4115) 
v_mul_f32 v21, v10, v7 // (op 4115) 


v_add_f32 v6, v18, v14 // (op 4116) 
v_add_f32 v7, v19, v15 // (op 4116) 
v_add_f32 v8, v20, v16 // (op 4116) 
v_add_f32 v9, v21, v17 // (op 4116) 


v_lshrrev_b32 v14, 16, v6 // (op 4117) 
v_lshrrev_b32 v15, 16, v7 // (op 4117) 
v_lshrrev_b32 v16, 16, v8 // (op 4117) 
v_lshrrev_b32 v17, 16, v9 // (op 4117) 


v_pack_b32_f16 v8, v14, v15 // (op 4098) 
v_pack_b32_f16 v9, v16, v17 // (op 4098) 

s_lshl_b32 s0, s4, 6 // (op 4098) 
s_add_u32 s0, s0, 64 // (op 4098) 
v_add_u32 v0, v13, s0 // (op 4098) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4098) Store value





s_lshl_b32 s0, s2, 7 // (op 4123) 
s_add_u32 s0, s0, 64 // (op 4123) 
v_add_u32 v0, v12, s0 // (op 4123) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4123) Load value

v_accvgpr_read v4, a40 // (op 4152) 
v_accvgpr_read v5, a41 // (op 4152) 
v_accvgpr_read v6, a42 // (op 4152) 
v_accvgpr_read v7, a43 // (op 4152) 
v_mul_f32 v14, v11, v4 // (op 4152) 
v_mul_f32 v15, v11, v5 // (op 4152) 
v_mul_f32 v16, v11, v6 // (op 4152) 
v_mul_f32 v17, v11, v7 // (op 4152) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4120) 
v_lshrrev_b32 v7, 16, v8 // (op 4120) 
v_lshlrev_b32 v6, 16, v6 // (op 4120) 
v_lshlrev_b32 v7, 16, v7 // (op 4120) 
v_mul_f32 v18, v10, v6 // (op 4120) 
v_mul_f32 v19, v10, v7 // (op 4120) 
v_and_b32 v6, 65535, v9 // (op 4120) 
v_lshrrev_b32 v7, 16, v9 // (op 4120) 
v_lshlrev_b32 v6, 16, v6 // (op 4120) 
v_lshlrev_b32 v7, 16, v7 // (op 4120) 
v_mul_f32 v20, v10, v6 // (op 4120) 
v_mul_f32 v21, v10, v7 // (op 4120) 


v_add_f32 v6, v18, v14 // (op 4121) 
v_add_f32 v7, v19, v15 // (op 4121) 
v_add_f32 v8, v20, v16 // (op 4121) 
v_add_f32 v9, v21, v17 // (op 4121) 


v_lshrrev_b32 v14, 16, v6 // (op 4122) 
v_lshrrev_b32 v15, 16, v7 // (op 4122) 
v_lshrrev_b32 v16, 16, v8 // (op 4122) 
v_lshrrev_b32 v17, 16, v9 // (op 4122) 


v_pack_b32_f16 v8, v14, v15 // (op 4100) 
v_pack_b32_f16 v9, v16, v17 // (op 4100) 

s_lshl_b32 s0, s4, 7 // (op 4100) 
s_add_u32 s0, s0, 64 // (op 4100) 
v_add_u32 v0, v13, s0 // (op 4100) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4100) Store value





s_lshl_b32 s0, s2, 6 // (op 4128) 
s_mul_i32 s1, 3, s0 // (op 4128) 
s_add_u32 s1, s1, 64 // (op 4128) 

v_add_u32 v0, v12, s1 // (op 4128) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4128) Load value

v_accvgpr_read v4, a44 // (op 4154) 
v_accvgpr_read v5, a45 // (op 4154) 
v_accvgpr_read v6, a46 // (op 4154) 
v_accvgpr_read v7, a47 // (op 4154) 
v_mul_f32 v14, v11, v4 // (op 4154) 
v_mul_f32 v15, v11, v5 // (op 4154) 
v_mul_f32 v16, v11, v6 // (op 4154) 
v_mul_f32 v17, v11, v7 // (op 4154) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4125) 
v_lshrrev_b32 v7, 16, v8 // (op 4125) 
v_lshlrev_b32 v6, 16, v6 // (op 4125) 
v_lshlrev_b32 v7, 16, v7 // (op 4125) 
v_mul_f32 v18, v10, v6 // (op 4125) 
v_mul_f32 v19, v10, v7 // (op 4125) 
v_and_b32 v6, 65535, v9 // (op 4125) 
v_lshrrev_b32 v7, 16, v9 // (op 4125) 
v_lshlrev_b32 v6, 16, v6 // (op 4125) 
v_lshlrev_b32 v7, 16, v7 // (op 4125) 
v_mul_f32 v20, v10, v6 // (op 4125) 
v_mul_f32 v21, v10, v7 // (op 4125) 


v_add_f32 v6, v18, v14 // (op 4126) 
v_add_f32 v7, v19, v15 // (op 4126) 
v_add_f32 v8, v20, v16 // (op 4126) 
v_add_f32 v9, v21, v17 // (op 4126) 


v_lshrrev_b32 v14, 16, v6 // (op 4127) 
v_lshrrev_b32 v15, 16, v7 // (op 4127) 
v_lshrrev_b32 v16, 16, v8 // (op 4127) 
v_lshrrev_b32 v17, 16, v9 // (op 4127) 


v_pack_b32_f16 v8, v14, v15 // (op 4102) 
v_pack_b32_f16 v9, v16, v17 // (op 4102) 

s_lshl_b32 s0, s4, 6 // (op 4102) 
s_mul_i32 s1, 3, s0 // (op 4102) 
s_add_u32 s1, s1, 64 // (op 4102) 

v_add_u32 v0, v13, s1 // (op 4102) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4102) Store value





s_lshl_b32 s0, s2, 8 // (op 4133) 
s_add_u32 s0, s0, 64 // (op 4133) 
v_add_u32 v0, v12, s0 // (op 4133) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4133) Load value

v_accvgpr_read v4, a48 // (op 4156) 
v_accvgpr_read v5, a49 // (op 4156) 
v_accvgpr_read v6, a50 // (op 4156) 
v_accvgpr_read v7, a51 // (op 4156) 
v_mul_f32 v14, v11, v4 // (op 4156) 
v_mul_f32 v15, v11, v5 // (op 4156) 
v_mul_f32 v16, v11, v6 // (op 4156) 
v_mul_f32 v17, v11, v7 // (op 4156) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4130) 
v_lshrrev_b32 v7, 16, v8 // (op 4130) 
v_lshlrev_b32 v6, 16, v6 // (op 4130) 
v_lshlrev_b32 v7, 16, v7 // (op 4130) 
v_mul_f32 v18, v10, v6 // (op 4130) 
v_mul_f32 v19, v10, v7 // (op 4130) 
v_and_b32 v6, 65535, v9 // (op 4130) 
v_lshrrev_b32 v7, 16, v9 // (op 4130) 
v_lshlrev_b32 v6, 16, v6 // (op 4130) 
v_lshlrev_b32 v7, 16, v7 // (op 4130) 
v_mul_f32 v20, v10, v6 // (op 4130) 
v_mul_f32 v21, v10, v7 // (op 4130) 


v_add_f32 v6, v18, v14 // (op 4131) 
v_add_f32 v7, v19, v15 // (op 4131) 
v_add_f32 v8, v20, v16 // (op 4131) 
v_add_f32 v9, v21, v17 // (op 4131) 


v_lshrrev_b32 v14, 16, v6 // (op 4132) 
v_lshrrev_b32 v15, 16, v7 // (op 4132) 
v_lshrrev_b32 v16, 16, v8 // (op 4132) 
v_lshrrev_b32 v17, 16, v9 // (op 4132) 


v_pack_b32_f16 v8, v14, v15 // (op 4104) 
v_pack_b32_f16 v9, v16, v17 // (op 4104) 

s_lshl_b32 s0, s4, 8 // (op 4104) 
s_add_u32 s0, s0, 64 // (op 4104) 
v_add_u32 v0, v13, s0 // (op 4104) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4104) Store value





s_lshl_b32 s0, s2, 6 // (op 4138) 
s_mul_i32 s1, 5, s0 // (op 4138) 
s_add_u32 s1, s1, 64 // (op 4138) 

v_add_u32 v0, v12, s1 // (op 4138) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4138) Load value

v_accvgpr_read v4, a52 // (op 4158) 
v_accvgpr_read v5, a53 // (op 4158) 
v_accvgpr_read v6, a54 // (op 4158) 
v_accvgpr_read v7, a55 // (op 4158) 
v_mul_f32 v14, v11, v4 // (op 4158) 
v_mul_f32 v15, v11, v5 // (op 4158) 
v_mul_f32 v16, v11, v6 // (op 4158) 
v_mul_f32 v17, v11, v7 // (op 4158) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4135) 
v_lshrrev_b32 v7, 16, v8 // (op 4135) 
v_lshlrev_b32 v6, 16, v6 // (op 4135) 
v_lshlrev_b32 v7, 16, v7 // (op 4135) 
v_mul_f32 v18, v10, v6 // (op 4135) 
v_mul_f32 v19, v10, v7 // (op 4135) 
v_and_b32 v6, 65535, v9 // (op 4135) 
v_lshrrev_b32 v7, 16, v9 // (op 4135) 
v_lshlrev_b32 v6, 16, v6 // (op 4135) 
v_lshlrev_b32 v7, 16, v7 // (op 4135) 
v_mul_f32 v20, v10, v6 // (op 4135) 
v_mul_f32 v21, v10, v7 // (op 4135) 


v_add_f32 v6, v18, v14 // (op 4136) 
v_add_f32 v7, v19, v15 // (op 4136) 
v_add_f32 v8, v20, v16 // (op 4136) 
v_add_f32 v9, v21, v17 // (op 4136) 


v_lshrrev_b32 v14, 16, v6 // (op 4137) 
v_lshrrev_b32 v15, 16, v7 // (op 4137) 
v_lshrrev_b32 v16, 16, v8 // (op 4137) 
v_lshrrev_b32 v17, 16, v9 // (op 4137) 


v_pack_b32_f16 v8, v14, v15 // (op 4106) 
v_pack_b32_f16 v9, v16, v17 // (op 4106) 

s_lshl_b32 s0, s4, 6 // (op 4106) 
s_mul_i32 s1, 5, s0 // (op 4106) 
s_add_u32 s1, s1, 64 // (op 4106) 

v_add_u32 v0, v13, s1 // (op 4106) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4106) Store value





s_lshl_b32 s0, s2, 6 // (op 4143) 
s_mul_i32 s1, 6, s0 // (op 4143) 
s_add_u32 s1, s1, 64 // (op 4143) 

v_add_u32 v0, v12, s1 // (op 4143) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4143) Load value

v_accvgpr_read v4, a56 // (op 4160) 
v_accvgpr_read v5, a57 // (op 4160) 
v_accvgpr_read v6, a58 // (op 4160) 
v_accvgpr_read v7, a59 // (op 4160) 
v_mul_f32 v14, v11, v4 // (op 4160) 
v_mul_f32 v15, v11, v5 // (op 4160) 
v_mul_f32 v16, v11, v6 // (op 4160) 
v_mul_f32 v17, v11, v7 // (op 4160) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4140) 
v_lshrrev_b32 v7, 16, v8 // (op 4140) 
v_lshlrev_b32 v6, 16, v6 // (op 4140) 
v_lshlrev_b32 v7, 16, v7 // (op 4140) 
v_mul_f32 v18, v10, v6 // (op 4140) 
v_mul_f32 v19, v10, v7 // (op 4140) 
v_and_b32 v6, 65535, v9 // (op 4140) 
v_lshrrev_b32 v7, 16, v9 // (op 4140) 
v_lshlrev_b32 v6, 16, v6 // (op 4140) 
v_lshlrev_b32 v7, 16, v7 // (op 4140) 
v_mul_f32 v20, v10, v6 // (op 4140) 
v_mul_f32 v21, v10, v7 // (op 4140) 


v_add_f32 v6, v18, v14 // (op 4141) 
v_add_f32 v7, v19, v15 // (op 4141) 
v_add_f32 v8, v20, v16 // (op 4141) 
v_add_f32 v9, v21, v17 // (op 4141) 


v_lshrrev_b32 v14, 16, v6 // (op 4142) 
v_lshrrev_b32 v15, 16, v7 // (op 4142) 
v_lshrrev_b32 v16, 16, v8 // (op 4142) 
v_lshrrev_b32 v17, 16, v9 // (op 4142) 


v_pack_b32_f16 v8, v14, v15 // (op 4108) 
v_pack_b32_f16 v9, v16, v17 // (op 4108) 

s_lshl_b32 s0, s4, 6 // (op 4108) 
s_mul_i32 s1, 6, s0 // (op 4108) 
s_add_u32 s1, s1, 64 // (op 4108) 

v_add_u32 v0, v13, s1 // (op 4108) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4108) Store value





s_lshl_b32 s0, s2, 6 // (op 4148) 
s_mul_i32 s1, 7, s0 // (op 4148) 
s_add_u32 s1, s1, 64 // (op 4148) 

v_add_u32 v0, v12, s1 // (op 4148) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4148) Load value

v_accvgpr_read v4, a60 // (op 4162) 
v_accvgpr_read v5, a61 // (op 4162) 
v_accvgpr_read v6, a62 // (op 4162) 
v_accvgpr_read v7, a63 // (op 4162) 
v_mul_f32 v14, v11, v4 // (op 4162) 
v_mul_f32 v15, v11, v5 // (op 4162) 
v_mul_f32 v16, v11, v6 // (op 4162) 
v_mul_f32 v17, v11, v7 // (op 4162) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4145) 
v_lshrrev_b32 v7, 16, v8 // (op 4145) 
v_lshlrev_b32 v6, 16, v6 // (op 4145) 
v_lshlrev_b32 v7, 16, v7 // (op 4145) 
v_mul_f32 v18, v10, v6 // (op 4145) 
v_mul_f32 v19, v10, v7 // (op 4145) 
v_and_b32 v6, 65535, v9 // (op 4145) 
v_lshrrev_b32 v7, 16, v9 // (op 4145) 
v_lshlrev_b32 v6, 16, v6 // (op 4145) 
v_lshlrev_b32 v7, 16, v7 // (op 4145) 
v_mul_f32 v20, v10, v6 // (op 4145) 
v_mul_f32 v21, v10, v7 // (op 4145) 


v_add_f32 v6, v18, v14 // (op 4146) 
v_add_f32 v7, v19, v15 // (op 4146) 
v_add_f32 v8, v20, v16 // (op 4146) 
v_add_f32 v9, v21, v17 // (op 4146) 


v_lshrrev_b32 v14, 16, v6 // (op 4147) 
v_lshrrev_b32 v15, 16, v7 // (op 4147) 
v_lshrrev_b32 v16, 16, v8 // (op 4147) 
v_lshrrev_b32 v17, 16, v9 // (op 4147) 


v_pack_b32_f16 v8, v14, v15 // (op 4110) 
v_pack_b32_f16 v9, v16, v17 // (op 4110) 

s_lshl_b32 s0, s4, 6 // (op 4110) 
s_mul_i32 s1, 7, s0 // (op 4110) 
s_add_u32 s1, s1, 64 // (op 4110) 

v_add_u32 v0, v13, s1 // (op 4110) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4110) Store value





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
v_and_b32 v6, 65535, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v18, v10, v6 // (op 149) 
v_mul_f32 v19, v10, v7 // (op 149) 
v_and_b32 v6, 65535, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v20, v10, v6 // (op 149) 
v_mul_f32 v21, v10, v7 // (op 149) 


v_add_f32 v6, v18, v14 // (op 145) 
v_add_f32 v7, v19, v15 // (op 145) 
v_add_f32 v8, v20, v16 // (op 145) 
v_add_f32 v9, v21, v17 // (op 145) 


v_lshrrev_b32 v14, 16, v6 // (op 146) 
v_lshrrev_b32 v15, 16, v7 // (op 146) 
v_lshrrev_b32 v16, 16, v8 // (op 146) 
v_lshrrev_b32 v17, 16, v9 // (op 146) 


v_pack_b32_f16 v8, v14, v15 // (op 147) 
v_pack_b32_f16 v9, v16, v17 // (op 147) 

v_mov_b32 v0, v13 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 3917) 
v_add_u32 v0, v12, s0 // (op 3917) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3917) Load value

v_accvgpr_read v4, a4 // (op 3913) 
v_accvgpr_read v5, a5 // (op 3913) 
v_accvgpr_read v6, a6 // (op 3913) 
v_accvgpr_read v7, a7 // (op 3913) 
v_mul_f32 v14, v11, v4 // (op 3913) 
v_mul_f32 v15, v11, v5 // (op 3913) 
v_mul_f32 v16, v11, v6 // (op 3913) 
v_mul_f32 v17, v11, v7 // (op 3913) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3918) 
v_lshrrev_b32 v7, 16, v8 // (op 3918) 
v_lshlrev_b32 v6, 16, v6 // (op 3918) 
v_lshlrev_b32 v7, 16, v7 // (op 3918) 
v_mul_f32 v18, v10, v6 // (op 3918) 
v_mul_f32 v19, v10, v7 // (op 3918) 
v_and_b32 v6, 65535, v9 // (op 3918) 
v_lshrrev_b32 v7, 16, v9 // (op 3918) 
v_lshlrev_b32 v6, 16, v6 // (op 3918) 
v_lshlrev_b32 v7, 16, v7 // (op 3918) 
v_mul_f32 v20, v10, v6 // (op 3918) 
v_mul_f32 v21, v10, v7 // (op 3918) 


v_add_f32 v6, v18, v14 // (op 3914) 
v_add_f32 v7, v19, v15 // (op 3914) 
v_add_f32 v8, v20, v16 // (op 3914) 
v_add_f32 v9, v21, v17 // (op 3914) 


v_lshrrev_b32 v14, 16, v6 // (op 3915) 
v_lshrrev_b32 v15, 16, v7 // (op 3915) 
v_lshrrev_b32 v16, 16, v8 // (op 3915) 
v_lshrrev_b32 v17, 16, v9 // (op 3915) 


v_pack_b32_f16 v8, v14, v15 // (op 3916) 
v_pack_b32_f16 v9, v16, v17 // (op 3916) 

s_lshl_b32 s0, s4, 6 // (op 3916) 
v_add_u32 v0, v13, s0 // (op 3916) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3916) Store value





s_lshl_b32 s0, s2, 7 // (op 3931) 
v_add_u32 v0, v12, s0 // (op 3931) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3931) Load value

v_accvgpr_read v4, a8 // (op 3927) 
v_accvgpr_read v5, a9 // (op 3927) 
v_accvgpr_read v6, a10 // (op 3927) 
v_accvgpr_read v7, a11 // (op 3927) 
v_mul_f32 v14, v11, v4 // (op 3927) 
v_mul_f32 v15, v11, v5 // (op 3927) 
v_mul_f32 v16, v11, v6 // (op 3927) 
v_mul_f32 v17, v11, v7 // (op 3927) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3932) 
v_lshrrev_b32 v7, 16, v8 // (op 3932) 
v_lshlrev_b32 v6, 16, v6 // (op 3932) 
v_lshlrev_b32 v7, 16, v7 // (op 3932) 
v_mul_f32 v18, v10, v6 // (op 3932) 
v_mul_f32 v19, v10, v7 // (op 3932) 
v_and_b32 v6, 65535, v9 // (op 3932) 
v_lshrrev_b32 v7, 16, v9 // (op 3932) 
v_lshlrev_b32 v6, 16, v6 // (op 3932) 
v_lshlrev_b32 v7, 16, v7 // (op 3932) 
v_mul_f32 v20, v10, v6 // (op 3932) 
v_mul_f32 v21, v10, v7 // (op 3932) 


v_add_f32 v6, v18, v14 // (op 3928) 
v_add_f32 v7, v19, v15 // (op 3928) 
v_add_f32 v8, v20, v16 // (op 3928) 
v_add_f32 v9, v21, v17 // (op 3928) 


v_lshrrev_b32 v14, 16, v6 // (op 3929) 
v_lshrrev_b32 v15, 16, v7 // (op 3929) 
v_lshrrev_b32 v16, 16, v8 // (op 3929) 
v_lshrrev_b32 v17, 16, v9 // (op 3929) 


v_pack_b32_f16 v8, v14, v15 // (op 3930) 
v_pack_b32_f16 v9, v16, v17 // (op 3930) 

s_lshl_b32 s0, s4, 7 // (op 3930) 
v_add_u32 v0, v13, s0 // (op 3930) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3930) Store value





s_lshl_b32 s0, s2, 6 // (op 3945) 
s_mul_i32 s1, 3, s0 // (op 3945) 

v_add_u32 v0, v12, s1 // (op 3945) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3945) Load value

v_accvgpr_read v4, a12 // (op 3941) 
v_accvgpr_read v5, a13 // (op 3941) 
v_accvgpr_read v6, a14 // (op 3941) 
v_accvgpr_read v7, a15 // (op 3941) 
v_mul_f32 v14, v11, v4 // (op 3941) 
v_mul_f32 v15, v11, v5 // (op 3941) 
v_mul_f32 v16, v11, v6 // (op 3941) 
v_mul_f32 v17, v11, v7 // (op 3941) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3946) 
v_lshrrev_b32 v7, 16, v8 // (op 3946) 
v_lshlrev_b32 v6, 16, v6 // (op 3946) 
v_lshlrev_b32 v7, 16, v7 // (op 3946) 
v_mul_f32 v18, v10, v6 // (op 3946) 
v_mul_f32 v19, v10, v7 // (op 3946) 
v_and_b32 v6, 65535, v9 // (op 3946) 
v_lshrrev_b32 v7, 16, v9 // (op 3946) 
v_lshlrev_b32 v6, 16, v6 // (op 3946) 
v_lshlrev_b32 v7, 16, v7 // (op 3946) 
v_mul_f32 v20, v10, v6 // (op 3946) 
v_mul_f32 v21, v10, v7 // (op 3946) 


v_add_f32 v6, v18, v14 // (op 3942) 
v_add_f32 v7, v19, v15 // (op 3942) 
v_add_f32 v8, v20, v16 // (op 3942) 
v_add_f32 v9, v21, v17 // (op 3942) 


v_lshrrev_b32 v14, 16, v6 // (op 3943) 
v_lshrrev_b32 v15, 16, v7 // (op 3943) 
v_lshrrev_b32 v16, 16, v8 // (op 3943) 
v_lshrrev_b32 v17, 16, v9 // (op 3943) 


v_pack_b32_f16 v8, v14, v15 // (op 3944) 
v_pack_b32_f16 v9, v16, v17 // (op 3944) 

s_lshl_b32 s0, s4, 6 // (op 3944) 
s_mul_i32 s1, 3, s0 // (op 3944) 

v_add_u32 v0, v13, s1 // (op 3944) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3944) Store value





s_lshl_b32 s0, s2, 8 // (op 3959) 
v_add_u32 v0, v12, s0 // (op 3959) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3959) Load value

v_accvgpr_read v4, a16 // (op 3955) 
v_accvgpr_read v5, a17 // (op 3955) 
v_accvgpr_read v6, a18 // (op 3955) 
v_accvgpr_read v7, a19 // (op 3955) 
v_mul_f32 v14, v11, v4 // (op 3955) 
v_mul_f32 v15, v11, v5 // (op 3955) 
v_mul_f32 v16, v11, v6 // (op 3955) 
v_mul_f32 v17, v11, v7 // (op 3955) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3960) 
v_lshrrev_b32 v7, 16, v8 // (op 3960) 
v_lshlrev_b32 v6, 16, v6 // (op 3960) 
v_lshlrev_b32 v7, 16, v7 // (op 3960) 
v_mul_f32 v18, v10, v6 // (op 3960) 
v_mul_f32 v19, v10, v7 // (op 3960) 
v_and_b32 v6, 65535, v9 // (op 3960) 
v_lshrrev_b32 v7, 16, v9 // (op 3960) 
v_lshlrev_b32 v6, 16, v6 // (op 3960) 
v_lshlrev_b32 v7, 16, v7 // (op 3960) 
v_mul_f32 v20, v10, v6 // (op 3960) 
v_mul_f32 v21, v10, v7 // (op 3960) 


v_add_f32 v6, v18, v14 // (op 3956) 
v_add_f32 v7, v19, v15 // (op 3956) 
v_add_f32 v8, v20, v16 // (op 3956) 
v_add_f32 v9, v21, v17 // (op 3956) 


v_lshrrev_b32 v14, 16, v6 // (op 3957) 
v_lshrrev_b32 v15, 16, v7 // (op 3957) 
v_lshrrev_b32 v16, 16, v8 // (op 3957) 
v_lshrrev_b32 v17, 16, v9 // (op 3957) 


v_pack_b32_f16 v8, v14, v15 // (op 3958) 
v_pack_b32_f16 v9, v16, v17 // (op 3958) 

s_lshl_b32 s0, s4, 8 // (op 3958) 
v_add_u32 v0, v13, s0 // (op 3958) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3958) Store value





s_lshl_b32 s0, s2, 6 // (op 3973) 
s_mul_i32 s1, 5, s0 // (op 3973) 

v_add_u32 v0, v12, s1 // (op 3973) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3973) Load value

v_accvgpr_read v4, a20 // (op 3969) 
v_accvgpr_read v5, a21 // (op 3969) 
v_accvgpr_read v6, a22 // (op 3969) 
v_accvgpr_read v7, a23 // (op 3969) 
v_mul_f32 v14, v11, v4 // (op 3969) 
v_mul_f32 v15, v11, v5 // (op 3969) 
v_mul_f32 v16, v11, v6 // (op 3969) 
v_mul_f32 v17, v11, v7 // (op 3969) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3974) 
v_lshrrev_b32 v7, 16, v8 // (op 3974) 
v_lshlrev_b32 v6, 16, v6 // (op 3974) 
v_lshlrev_b32 v7, 16, v7 // (op 3974) 
v_mul_f32 v18, v10, v6 // (op 3974) 
v_mul_f32 v19, v10, v7 // (op 3974) 
v_and_b32 v6, 65535, v9 // (op 3974) 
v_lshrrev_b32 v7, 16, v9 // (op 3974) 
v_lshlrev_b32 v6, 16, v6 // (op 3974) 
v_lshlrev_b32 v7, 16, v7 // (op 3974) 
v_mul_f32 v20, v10, v6 // (op 3974) 
v_mul_f32 v21, v10, v7 // (op 3974) 


v_add_f32 v6, v18, v14 // (op 3970) 
v_add_f32 v7, v19, v15 // (op 3970) 
v_add_f32 v8, v20, v16 // (op 3970) 
v_add_f32 v9, v21, v17 // (op 3970) 


v_lshrrev_b32 v14, 16, v6 // (op 3971) 
v_lshrrev_b32 v15, 16, v7 // (op 3971) 
v_lshrrev_b32 v16, 16, v8 // (op 3971) 
v_lshrrev_b32 v17, 16, v9 // (op 3971) 


v_pack_b32_f16 v8, v14, v15 // (op 3972) 
v_pack_b32_f16 v9, v16, v17 // (op 3972) 

s_lshl_b32 s0, s4, 6 // (op 3972) 
s_mul_i32 s1, 5, s0 // (op 3972) 

v_add_u32 v0, v13, s1 // (op 3972) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3972) Store value





s_lshl_b32 s0, s2, 6 // (op 3987) 
s_mul_i32 s1, 6, s0 // (op 3987) 

v_add_u32 v0, v12, s1 // (op 3987) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 3987) Load value

v_accvgpr_read v4, a24 // (op 3983) 
v_accvgpr_read v5, a25 // (op 3983) 
v_accvgpr_read v6, a26 // (op 3983) 
v_accvgpr_read v7, a27 // (op 3983) 
v_mul_f32 v14, v11, v4 // (op 3983) 
v_mul_f32 v15, v11, v5 // (op 3983) 
v_mul_f32 v16, v11, v6 // (op 3983) 
v_mul_f32 v17, v11, v7 // (op 3983) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 3988) 
v_lshrrev_b32 v7, 16, v8 // (op 3988) 
v_lshlrev_b32 v6, 16, v6 // (op 3988) 
v_lshlrev_b32 v7, 16, v7 // (op 3988) 
v_mul_f32 v18, v10, v6 // (op 3988) 
v_mul_f32 v19, v10, v7 // (op 3988) 
v_and_b32 v6, 65535, v9 // (op 3988) 
v_lshrrev_b32 v7, 16, v9 // (op 3988) 
v_lshlrev_b32 v6, 16, v6 // (op 3988) 
v_lshlrev_b32 v7, 16, v7 // (op 3988) 
v_mul_f32 v20, v10, v6 // (op 3988) 
v_mul_f32 v21, v10, v7 // (op 3988) 


v_add_f32 v6, v18, v14 // (op 3984) 
v_add_f32 v7, v19, v15 // (op 3984) 
v_add_f32 v8, v20, v16 // (op 3984) 
v_add_f32 v9, v21, v17 // (op 3984) 


v_lshrrev_b32 v14, 16, v6 // (op 3985) 
v_lshrrev_b32 v15, 16, v7 // (op 3985) 
v_lshrrev_b32 v16, 16, v8 // (op 3985) 
v_lshrrev_b32 v17, 16, v9 // (op 3985) 


v_pack_b32_f16 v8, v14, v15 // (op 3986) 
v_pack_b32_f16 v9, v16, v17 // (op 3986) 

s_lshl_b32 s0, s4, 6 // (op 3986) 
s_mul_i32 s1, 6, s0 // (op 3986) 

v_add_u32 v0, v13, s1 // (op 3986) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 3986) Store value





s_lshl_b32 s0, s2, 6 // (op 4001) 
s_mul_i32 s1, 7, s0 // (op 4001) 

v_add_u32 v0, v12, s1 // (op 4001) 


buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 4001) Load value

v_accvgpr_read v4, a28 // (op 3997) 
v_accvgpr_read v5, a29 // (op 3997) 
v_accvgpr_read v6, a30 // (op 3997) 
v_accvgpr_read v7, a31 // (op 3997) 
v_mul_f32 v14, v11, v4 // (op 3997) 
v_mul_f32 v15, v11, v5 // (op 3997) 
v_mul_f32 v16, v11, v6 // (op 3997) 
v_mul_f32 v17, v11, v7 // (op 3997) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 4002) 
v_lshrrev_b32 v7, 16, v8 // (op 4002) 
v_lshlrev_b32 v6, 16, v6 // (op 4002) 
v_lshlrev_b32 v7, 16, v7 // (op 4002) 
v_mul_f32 v18, v10, v6 // (op 4002) 
v_mul_f32 v19, v10, v7 // (op 4002) 
v_and_b32 v6, 65535, v9 // (op 4002) 
v_lshrrev_b32 v7, 16, v9 // (op 4002) 
v_lshlrev_b32 v6, 16, v6 // (op 4002) 
v_lshlrev_b32 v7, 16, v7 // (op 4002) 
v_mul_f32 v20, v10, v6 // (op 4002) 
v_mul_f32 v21, v10, v7 // (op 4002) 


v_add_f32 v6, v18, v14 // (op 3998) 
v_add_f32 v7, v19, v15 // (op 3998) 
v_add_f32 v8, v20, v16 // (op 3998) 
v_add_f32 v9, v21, v17 // (op 3998) 


v_lshrrev_b32 v14, 16, v6 // (op 3999) 
v_lshrrev_b32 v15, 16, v7 // (op 3999) 
v_lshrrev_b32 v16, 16, v8 // (op 3999) 
v_lshrrev_b32 v17, 16, v9 // (op 3999) 


v_pack_b32_f16 v8, v14, v15 // (op 4000) 
v_pack_b32_f16 v9, v16, v17 // (op 4000) 

s_lshl_b32 s0, s4, 6 // (op 4000) 
s_mul_i32 s1, 7, s0 // (op 4000) 

v_add_u32 v0, v13, s1 // (op 4000) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 4000) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC
.LRR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_end:
.size RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC, .LRR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC_end-RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC
  .amdhsa_next_free_vgpr 252
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 76288 // lds bytes
  .amdhsa_accum_offset 188
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
    DataTypeB: F8
    DestDataType: b
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC
    .symbol: RR_GEMM_NN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_LDSSYNC.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 76288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 188
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

