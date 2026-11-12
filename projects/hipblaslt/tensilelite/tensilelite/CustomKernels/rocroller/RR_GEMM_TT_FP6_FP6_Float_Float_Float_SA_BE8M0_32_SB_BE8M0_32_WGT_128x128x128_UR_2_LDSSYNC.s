// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC
.p2align 8
.type RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC,@function
RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 8980) 
v_and_b32 v3, 15, v1 // (op 8980) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 8980) 


v_and_b32 v0, 63, v1 // (op 8980) 
v_lshrrev_b32 v3, 4, v0 // (op 8980) 

v_lshlrev_b32 v0, 5, v3 // (op 8980) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 8980) 


v_mul_lo_u32 v0, v3, 6 // (op 8980) 

v_lshrrev_b32 v3, 3, v0 // (op 8980) 

v_and_b32 v0, 63, v1 // (op 9038) 
v_and_b32 v4, 48, v1 // (op 9038) 
v_sub_u32 v5, v0, v4 // (op 9038) 

v_lshrrev_b32 v4, 4, v0 // (op 9038) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 9038) 


v_lshrrev_b32 v4, 6, v1 // (op 9038) 
v_and_b32 v5, 1, v4 // (op 9038) 

v_lshlrev_b32 v4, 4, v5 // (op 9038) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 9038) 


v_mul_lo_u32 v0, v5, 6 // (op 9038) 
v_lshrrev_b32 v4, 3, v0 // (op 9038) 

v_lshrrev_b32 v0, 4, v5 // (op 9038) 

v_lshlrev_b32 v5, 2, v0 // (op 9038) 

v_add_u32 v0, v4, v5 // (op 9038) 


v_lshrrev_b32 v4, 7, v1 // (op 9096) 
v_and_b32 v5, 15, v1 // (op 9096) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 9096) 


v_and_b32 v4, 63, v1 // (op 9096) 
v_lshrrev_b32 v5, 4, v4 // (op 9096) 

v_lshl_add_u32 v4, v6, 2, v5 // (op 9096) 


v_lshrrev_b32 v5, 6, v1 // (op 9154) 
v_and_b32 v6, 1, v5 // (op 9154) 

v_and_b32 v5, 15, v1 // (op 9154) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9154) 


v_and_b32 v5, 63, v1 // (op 9154) 
v_lshrrev_b32 v6, 4, v5 // (op 9154) 

v_lshl_add_u32 v5, v6, 7, v7 // (op 9154) 


v_lshrrev_b32 v6, 7, v1 // (op 9206) 
v_and_b32 v7, 15, v1 // (op 9206) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9206) 


v_and_b32 v6, 63, v1 // (op 9206) 
v_lshrrev_b32 v7, 4, v6 // (op 9206) 

v_lshlrev_b32 v6, 5, v7 // (op 9206) 

v_lshl_add_u32 v7, v8, 7, v6 // (op 9206) 


v_mul_lo_u32 v6, v7, 6 // (op 9206) 

v_lshrrev_b32 v7, 3, v6 // (op 9206) 

v_and_b32 v6, 63, v1 // (op 9235) 
v_and_b32 v8, 48, v1 // (op 9235) 
v_sub_u32 v9, v6, v8 // (op 9235) 

v_lshrrev_b32 v8, 4, v6 // (op 9235) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 9235) 


v_lshrrev_b32 v8, 6, v1 // (op 9235) 
v_and_b32 v9, 1, v8 // (op 9235) 

v_lshlrev_b32 v8, 4, v9 // (op 9235) 

v_lshl_add_u32 v9, v6, 7, v8 // (op 9235) 


v_mul_lo_u32 v6, v9, 6 // (op 9235) 
v_lshrrev_b32 v8, 3, v6 // (op 9235) 

v_lshrrev_b32 v6, 4, v9 // (op 9235) 

v_lshlrev_b32 v9, 2, v6 // (op 9235) 

v_add_u32 v6, v8, v9 // (op 9235) 


v_lshrrev_b32 v8, 7, v1 // (op 9264) 
v_and_b32 v9, 15, v1 // (op 9264) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 9264) 


v_and_b32 v8, 63, v1 // (op 9264) 
v_lshrrev_b32 v9, 4, v8 // (op 9264) 

v_lshl_add_u32 v8, v10, 2, v9 // (op 9264) 


v_lshrrev_b32 v9, 6, v1 // (op 9293) 
v_and_b32 v10, 1, v9 // (op 9293) 

v_and_b32 v9, 15, v1 // (op 9293) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9293) 


v_and_b32 v9, 63, v1 // (op 9293) 
v_lshrrev_b32 v10, 4, v9 // (op 9293) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 9293) 


v_lshrrev_b32 v10, 7, v1 // (op 8918) 
v_and_b32 v11, 63, v1 // (op 8918) 
v_lshrrev_b32 v12, 4, v11 // (op 8918) 
v_lshlrev_b32 v13, 2, v12 // (op 8918) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8918) 


v_lshl_add_u32 v10, s16, 7, v12 // (op 8918) 

v_lshrrev_b32 v12, 2, v11 // (op 8918) 

v_and_b32 v11, 3, v12 // (op 8918) 

v_and_b32 v12, 3, v1 // (op 8918) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 8918) 


v_lshrrev_b32 v11, 6, v1 // (op 8918) 
v_and_b32 v12, 1, v11 // (op 8918) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 8918) 


v_lshl_add_u32 v12, s17, 7, v11 // (op 8918) 

v_mul_lo_u32 v11, v12, s2 // (op 8918) 

v_add_lshl_u32 v12, v10, v11, 2 // (op 8918) 


v_lshrrev_b32 v10, 7, v1 // (op 9175) 
v_and_b32 v11, 63, v1 // (op 9175) 
v_lshrrev_b32 v13, 4, v11 // (op 9175) 
s_mul_i32 s1, s2, s9 // (op 8920) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 8920) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 8920) least significant
s_mul_hi_u32 s19, s2, s8 // (op 8920) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8920) most significant: sum
s_add_u32 s1, s1, s19 // (op 8920) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8920) 

v_lshlrev_b32 v14, 2, v13 // (op 9175) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 9175) 


v_lshl_add_u32 v10, s16, 7, v13 // (op 9175) 

v_lshrrev_b32 v13, 2, v11 // (op 9175) 

v_and_b32 v11, 3, v13 // (op 9175) 

v_and_b32 v13, 3, v1 // (op 9175) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 9175) 


v_lshrrev_b32 v11, 6, v1 // (op 9175) 
v_and_b32 v13, 1, v11 // (op 9175) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 9175) 


v_lshl_add_u32 v13, s17, 7, v11 // (op 9175) 

v_mul_lo_u32 v11, v13, s4 // (op 9175) 

v_add_lshl_u32 v13, v10, v11, 2 // (op 9175) 


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


s_lshl_b64 s[6:7], s[0:1], 2 // (op 9177) 

s_mov_b32 s44, s10 // (op 9177) reg expression
s_mov_b32 s45, s11 // (op 9177) reg expression
s_mov_b32 s46, s6 // (op 9177) reg expression
s_mov_b32 s47, 131072 // (op 9177) call()



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
v_accvgpr_write a32, 0.00000 // (op 1722) call()
v_accvgpr_write a33, 0.00000 // (op 1722) call()
v_accvgpr_write a34, 0.00000 // (op 1722) call()
v_accvgpr_write a35, 0.00000 // (op 1722) call()
v_accvgpr_write a36, 0.00000 // (op 2031) call()
v_accvgpr_write a37, 0.00000 // (op 2031) call()
v_accvgpr_write a38, 0.00000 // (op 2031) call()
v_accvgpr_write a39, 0.00000 // (op 2031) call()
v_accvgpr_write a40, 0.00000 // (op 2032) call()
v_accvgpr_write a41, 0.00000 // (op 2032) call()
v_accvgpr_write a42, 0.00000 // (op 2032) call()
v_accvgpr_write a43, 0.00000 // (op 2032) call()
v_accvgpr_write a44, 0.00000 // (op 2033) call()
v_accvgpr_write a45, 0.00000 // (op 2033) call()
v_accvgpr_write a46, 0.00000 // (op 2033) call()
v_accvgpr_write a47, 0.00000 // (op 2033) call()
v_accvgpr_write a48, 0.00000 // (op 2492) call()
v_accvgpr_write a49, 0.00000 // (op 2492) call()
v_accvgpr_write a50, 0.00000 // (op 2492) call()
v_accvgpr_write a51, 0.00000 // (op 2492) call()
v_accvgpr_write a52, 0.00000 // (op 2801) call()
v_accvgpr_write a53, 0.00000 // (op 2801) call()
v_accvgpr_write a54, 0.00000 // (op 2801) call()
v_accvgpr_write a55, 0.00000 // (op 2801) call()
v_accvgpr_write a56, 0.00000 // (op 2802) call()
v_accvgpr_write a57, 0.00000 // (op 2802) call()
v_accvgpr_write a58, 0.00000 // (op 2802) call()
v_accvgpr_write a59, 0.00000 // (op 2802) call()
v_accvgpr_write a60, 0.00000 // (op 2803) call()
v_accvgpr_write a61, 0.00000 // (op 2803) call()
v_accvgpr_write a62, 0.00000 // (op 2803) call()
v_accvgpr_write a63, 0.00000 // (op 2803) call()
v_lshrrev_b32 v14, 1, v1 // (op 8680) 
v_lshl_add_u32 v15, s16, 7, v14 // (op 8680) 

v_and_b32 v14, -2, v1 // (op 8680) 
v_sub_u32 v16, v1, v14 // (op 8680) 

v_lshlrev_b32 v14, 6, v16 // (op 8680) 

v_mov_b32 v17, 0 // (op 8680) convert
v_mov_b32 v16, v14 // (op 8680) convert

v_mul_lo_u32 v19, v15, s13 // (op 8680) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 8680) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 8680) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 8680) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 8680) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 8680) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 8680) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 8680) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 8680) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 8680) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8680) 

v_mov_b32 v14, v16 // (op 8680) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 8682) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 8682) least significant
s_mul_hi_u32 s6, s22, 6 // (op 8682) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8682) most significant: sum

s_mul_i32 s15, s12, s1 // (op 8682) most significant: low of low * high
s_mul_i32 s6, s13, s0 // (op 8682) most significant: low of high * low
s_mul_i32 s14, s12, s0 // (op 8682) least significant
s_mul_hi_u32 s7, s12, s0 // (op 8682) most significant: high of low * low
s_add_u32 s15, s15, s6 // (op 8682) most significant: sum
s_add_u32 s15, s15, s7 // (op 8682) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 8682) 

s_mov_b32 s8, s20 // (op 8682) reg expression
s_mov_b32 s9, s21 // (op 8682) reg expression
s_mov_b32 s10, s0 // (op 8682) reg expression
s_mov_b32 s11, 131072 // (op 8682) call()

v_lshrrev_b32 v15, 1, v1 // (op 8699) 
v_lshl_add_u32 v16, s16, 7, v15 // (op 8699) 

v_and_b32 v15, -2, v1 // (op 8699) 
v_sub_u32 v17, v1, v15 // (op 8699) 

v_lshlrev_b32 v15, 6, v17 // (op 8699) 

v_mov_b32 v19, 0 // (op 8699) convert
v_mov_b32 v18, v15 // (op 8699) convert

v_mul_lo_u32 v21, v16, s13 // (op 8699) most significant: low of low * high
v_mul_hi_u32 v15, v16, s12 // (op 8699) most significant: high of low * low
v_mul_lo_u32 v20, v16, s12 // (op 8699) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 8699) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 8699) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 8699) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 8699) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 8699) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 8699) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 8699) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 8699) 

v_mov_b32 v15, v16 // (op 8699) convert

s_mul_i32 s1, s23, 6 // (op 8701) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 8701) least significant
s_mul_hi_u32 s6, s22, 6 // (op 8701) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8701) most significant: sum

s_mul_i32 s7, s12, s1 // (op 8701) most significant: low of low * high
s_mul_i32 s14, s13, s0 // (op 8701) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 8701) least significant
s_mul_hi_u32 s15, s12, s0 // (op 8701) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 8701) most significant: sum
s_add_u32 s7, s7, s15 // (op 8701) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 8701) 

s_mov_b32 s8, s20 // (op 8701) reg expression
s_mov_b32 s9, s21 // (op 8701) reg expression
s_mov_b32 s10, s0 // (op 8701) reg expression
s_mov_b32 s11, 131072 // (op 8701) call()

v_and_b32 v16, -2, v1 // (op 8744) 
v_sub_u32 v17, v1, v16 // (op 8744) 

v_lshlrev_b32 v16, 6, v17 // (op 8744) 

v_lshl_add_u32 v17, s17, 7, v16 // (op 8744) 

v_mov_b32 v19, 0 // (op 8744) convert
v_mov_b32 v18, v17 // (op 8744) convert

v_lshrrev_b32 v16, 1, v1 // (op 8744) 
v_mul_lo_u32 v21, v16, s25 // (op 8744) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 8744) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 8744) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 8744) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 8744) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 8744) most significant half



v_mul_lo_u32 v17, v21, 6 // (op 8744) most significant: low of high * low
v_mul_hi_u32 v18, v20, 6 // (op 8744) most significant: high of low * low
v_mul_lo_u32 v16, v20, 6 // (op 8744) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 8744) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 8744) 

v_mov_b32 v16, v18 // (op 8744) convert

s_mul_i32 s1, s29, 6 // (op 8746) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 8746) least significant
s_mul_hi_u32 s6, s28, 6 // (op 8746) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8746) most significant: sum

s_mul_i32 s7, s24, s1 // (op 8746) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 8746) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 8746) least significant
s_mul_hi_u32 s15, s24, s0 // (op 8746) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 8746) most significant: sum
s_add_u32 s7, s7, s15 // (op 8746) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 8746) 

s_mov_b32 s48, s26 // (op 8746) reg expression
s_mov_b32 s49, s27 // (op 8746) reg expression
s_mov_b32 s50, s0 // (op 8746) reg expression
s_mov_b32 s51, 131072 // (op 8746) call()

v_and_b32 v17, -2, v1 // (op 8763) 
v_sub_u32 v18, v1, v17 // (op 8763) 

v_lshlrev_b32 v17, 6, v18 // (op 8763) 

v_lshl_add_u32 v18, s17, 7, v17 // (op 8763) 

v_mov_b32 v21, 0 // (op 8763) convert
v_mov_b32 v20, v18 // (op 8763) convert

v_lshrrev_b32 v17, 1, v1 // (op 8763) 
v_mul_lo_u32 v19, v17, s25 // (op 8763) most significant: low of low * high
v_mul_hi_u32 v22, v17, s24 // (op 8763) most significant: high of low * low
v_mul_lo_u32 v18, v17, s24 // (op 8763) least significant: low of low * low
v_add_u32 v19, v19, v22 // (op 8763) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v20 // (op 8763) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 8763) most significant half



v_mul_lo_u32 v21, v19, 6 // (op 8763) most significant: low of high * low
v_mul_hi_u32 v17, v18, 6 // (op 8763) most significant: high of low * low
v_mul_lo_u32 v20, v18, 6 // (op 8763) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 8763) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 8763) 

v_mov_b32 v17, v18 // (op 8763) convert

s_mul_i32 s1, s29, 6 // (op 8765) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 8765) least significant
s_mul_hi_u32 s6, s28, 6 // (op 8765) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8765) most significant: sum

s_mul_i32 s7, s24, s1 // (op 8765) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 8765) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 8765) least significant
s_mul_hi_u32 s15, s24, s0 // (op 8765) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 8765) most significant: sum
s_add_u32 s7, s7, s15 // (op 8765) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 8765) 

s_mov_b32 s48, s26 // (op 8765) reg expression
s_mov_b32 s49, s27 // (op 8765) reg expression
s_mov_b32 s50, s0 // (op 8765) reg expression
s_mov_b32 s51, 131072 // (op 8765) call()

v_lshrrev_b32 v18, 2, v1 // (op 8808) 
v_lshlrev_b32 v19, 1, v18 // (op 8808) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 8808) 

v_and_b32 v19, -4, v1 // (op 8808) 
v_sub_u32 v20, v1, v19 // (op 8808) 

v_mul_lo_u32 v19, v18, s30 // (op 8808) 
v_add_u32 v19, v19, v20 // (op 8808) 


s_mul_i32 s1, s30, s23 // (op 8810) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 8810) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 8810) least significant
s_mul_hi_u32 s7, s30, s22 // (op 8810) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8810) most significant: sum
s_add_u32 s1, s1, s7 // (op 8810) most significant: sum


s_mov_b32 s52, s32 // (op 8810) reg expression
s_mov_b32 s53, s33 // (op 8810) reg expression
s_mov_b32 s54, s0 // (op 8810) reg expression
s_mov_b32 s55, 131072 // (op 8810) call()

v_lshrrev_b32 v18, 2, v1 // (op 8827) 
v_lshlrev_b32 v20, 1, v18 // (op 8827) 

v_lshl_add_u32 v18, s16, 7, v20 // (op 8827) 

v_and_b32 v20, -4, v1 // (op 8827) 
v_sub_u32 v21, v1, v20 // (op 8827) 

v_mul_lo_u32 v20, v18, s30 // (op 8827) 
v_add_u32 v20, v20, v21 // (op 8827) 


s_mul_i32 s1, s30, s23 // (op 8829) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 8829) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 8829) least significant
s_mul_hi_u32 s7, s30, s22 // (op 8829) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8829) most significant: sum
s_add_u32 s1, s1, s7 // (op 8829) most significant: sum


s_mov_b32 s52, s32 // (op 8829) reg expression
s_mov_b32 s53, s33 // (op 8829) reg expression
s_mov_b32 s54, s0 // (op 8829) reg expression
s_mov_b32 s55, 131072 // (op 8829) call()

v_and_b32 v18, -128, v1 // (op 8872) 
v_sub_u32 v21, v1, v18 // (op 8872) 

v_lshl_add_u32 v18, s17, 7, v21 // (op 8872) 

v_lshrrev_b32 v21, 7, v1 // (op 8872) 
v_lshlrev_b32 v22, 1, v21 // (op 8872) 

v_mul_lo_u32 v21, v22, s34 // (op 8872) 
v_add_u32 v21, v21, v18 // (op 8872) 


s_mul_i32 s1, s34, s39 // (op 8874) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 8874) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 8874) least significant
s_mul_hi_u32 s7, s34, s38 // (op 8874) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8874) most significant: sum
s_add_u32 s1, s1, s7 // (op 8874) most significant: sum


s_mov_b32 s56, s36 // (op 8874) reg expression
s_mov_b32 s57, s37 // (op 8874) reg expression
s_mov_b32 s58, s0 // (op 8874) reg expression
s_mov_b32 s59, 131072 // (op 8874) call()

v_and_b32 v18, -128, v1 // (op 8891) 
v_sub_u32 v22, v1, v18 // (op 8891) 

v_lshl_add_u32 v18, s17, 7, v22 // (op 8891) 

v_lshrrev_b32 v22, 7, v1 // (op 8891) 
v_lshlrev_b32 v23, 1, v22 // (op 8891) 

v_mul_lo_u32 v22, v23, s34 // (op 8891) 
v_add_u32 v22, v22, v18 // (op 8891) 


s_mul_i32 s1, s34, s39 // (op 8893) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 8893) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 8893) least significant
s_mul_hi_u32 s7, s34, s38 // (op 8893) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8893) most significant: sum
s_add_u32 s1, s1, s7 // (op 8893) most significant: sum


s_mov_b32 s56, s36 // (op 8893) reg expression
s_mov_b32 s57, s37 // (op 8893) reg expression
s_mov_b32 s58, s0 // (op 8893) reg expression
s_mov_b32 s59, 131072 // (op 8893) call()



v_mov_b32 v18, v16 // (op 8268) reg expression

buffer_load_dwordx3 v[24:26], v18, s[48:51], 0 offen // (op 8268) Load value
buffer_load_dwordx3 v[28:30], v18, s[48:51], 0 offen offset:12 // (op 8268) Load value
buffer_load_dwordx3 v[32:34], v18, s[48:51], 0 offen offset:24 // (op 8268) Load value
buffer_load_dwordx3 v[36:38], v18, s[48:51], 0 offen offset:36 // (op 8268) Load value


v_mov_b64 v[40:41], s[34:35] // (op 8271) convert

v_mov_b32 v18, v21 // (op 8271) reg expression
buffer_load_ubyte v42, v18, s[56:59], 0 offen offset: 0 // (op 8271) Load value
v_add_u32 v18, v18, v40 // (op 8271) 
buffer_load_ubyte v43, v18, s[56:59], 0 offen offset: 0 // (op 8271) Load value





buffer_load_dwordx4 v[44:47], v14, s[8:11], 0 offen // (op 8274) Load value
buffer_load_dwordx2 v[48:49], v14, s[8:11], 0 offen offset:16 // (op 8274) Load value
buffer_load_dwordx4 v[50:53], v14, s[8:11], 0 offen offset:24 // (op 8274) Load value
buffer_load_dwordx2 v[54:55], v14, s[8:11], 0 offen offset:40 // (op 8274) Load value

v_mov_b64 v[40:41], s[30:31] // (op 8277) convert

v_mov_b32 v18, v19 // (op 8277) 
buffer_load_ubyte v56, v18, s[52:55], 0 offen offset: 0 // (op 8277) Load value
v_add_u32 v18, v18, v40 // (op 8277) 
buffer_load_ubyte v57, v18, s[52:55], 0 offen offset: 0 // (op 8277) Load value




s_lshl_b64 s[0:1], s[24:25], 7 // (op 8280) 
s_mul_i32 s7, s1, 6 // (op 8280) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 8280) least significant
s_mul_hi_u32 s14, s0, 6 // (op 8280) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 8280) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 8280) 

v_add_u32 v18, v16, s0 // (op 8280) 


buffer_load_dwordx3 v[58:60], v18, s[48:51], 0 offen // (op 8280) Load value
buffer_load_dwordx3 v[62:64], v18, s[48:51], 0 offen offset:12 // (op 8280) Load value
buffer_load_dwordx3 v[66:68], v18, s[48:51], 0 offen offset:24 // (op 8280) Load value
buffer_load_dwordx3 v[70:72], v18, s[48:51], 0 offen offset:36 // (op 8280) Load value



v_mov_b64 v[40:41], s[34:35] // (op 8283) convert

s_lshl_b32 s0, s34, 2 // (op 8283) 
v_add_u32 v16, v21, s0 // (op 8283) 

buffer_load_ubyte v74, v16, s[56:59], 0 offen offset: 0 // (op 8283) Load value
v_add_u32 v16, v16, v40 // (op 8283) 
buffer_load_ubyte v75, v16, s[56:59], 0 offen offset: 0 // (op 8283) Load value






buffer_load_dwordx4 v[76:79], v14, s[8:11], 0 offen offset:96 // (op 8286) Load value
buffer_load_dwordx2 v[80:81], v14, s[8:11], 0 offen offset:112 // (op 8286) Load value
buffer_load_dwordx4 v[82:85], v14, s[8:11], 0 offen offset:120 // (op 8286) Load value
buffer_load_dwordx2 v[86:87], v14, s[8:11], 0 offen offset:136 // (op 8286) Load value


v_mov_b64 v[40:41], s[30:31] // (op 8289) convert

v_mov_b32 v14, v19 // (op 8289) 
buffer_load_ubyte v88, v14, s[52:55], 0 offen offset:4 // (op 8289) Load value
v_add_u32 v14, v14, v40 // (op 8289) 
buffer_load_ubyte v89, v14, s[52:55], 0 offen offset:4 // (op 8289) Load value



v_and_b32 v14, -2, v1 // (op 9021) 
v_sub_u32 v16, v1, v14 // (op 9021) 

v_lshlrev_b32 v14, 6, v16 // (op 9021) 

v_lshrrev_b32 v16, 1, v1 // (op 9021) 
v_lshl_add_u32 v21, v16, 7, v14 // (op 9021) 


v_mul_lo_u32 v14, v21, 6 // (op 9021) 
v_lshrrev_b32 v16, 3, v14 // (op 9021) 

v_lshrrev_b32 v14, 4, v21 // (op 9021) 

v_lshlrev_b32 v21, 2, v14 // (op 9021) 

v_add_u32 v14, v16, v21 // (op 9021) 





s_waitcnt vmcnt(23)
ds_write_b96 v14, v[24:26] // (op 8291) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v14, v[28:30] offset:16 // (op 8291) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v14, v[32:34] offset:32 // (op 8291) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v14, v[36:38] offset:48 // (op 8291) Store local data 

v_and_b32 v14, -128, v1 // (op 9137) 
v_sub_u32 v16, v1, v14 // (op 9137) 

v_lshrrev_b32 v14, 7, v1 // (op 9137) 
v_lshl_add_u32 v21, v14, 8, v16 // (op 9137) 





s_waitcnt vmcnt(19)
ds_write_b8 v21, v42 offset:16384 // (op 8292) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v21, v43 offset:16512 // (op 8292) Store local data 

v_and_b32 v14, -2, v1 // (op 8963) 
v_sub_u32 v16, v1, v14 // (op 8963) 

v_lshlrev_b32 v14, 6, v16 // (op 8963) 

v_lshrrev_b32 v16, 1, v1 // (op 8963) 
v_lshl_add_u32 v21, v16, 7, v14 // (op 8963) 


v_mul_lo_u32 v14, v21, 6 // (op 8963) 

v_lshrrev_b32 v16, 3, v14 // (op 8963) 




s_waitcnt vmcnt(17)
ds_write_b128 v16, v[44:47] offset:16896 // (op 8293) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v16, v[48:49] offset:16912 // (op 8293) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v16, v[50:53] offset:16920 // (op 8293) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v16, v[54:55] offset:16936 // (op 8293) Store local data 

v_and_b32 v14, -4, v1 // (op 9079) 
v_sub_u32 v16, v1, v14 // (op 9079) 

v_lshrrev_b32 v14, 2, v1 // (op 9079) 
v_lshl_add_u32 v21, v14, 3, v16 // (op 9079) 





s_waitcnt vmcnt(13)
ds_write_b8 v21, v56 offset:29184 // (op 8294) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v21, v57 offset:29188 // (op 8294) Store local data 

v_and_b32 v14, -2, v1 // (op 8934) 
v_sub_u32 v16, v1, v14 // (op 8934) 

v_lshlrev_b32 v14, 6, v16 // (op 8934) 

v_lshrrev_b32 v16, 1, v1 // (op 8934) 
v_lshl_add_u32 v21, v16, 7, v14 // (op 8934) 


v_mul_lo_u32 v14, v21, 6 // (op 8934) 

v_lshrrev_b32 v16, 3, v14 // (op 8934) 

v_and_b32 v14, -2, v1 // (op 8992) 
v_sub_u32 v21, v1, v14 // (op 8992) 

v_lshlrev_b32 v14, 6, v21 // (op 8992) 

v_lshrrev_b32 v21, 1, v1 // (op 8992) 
v_lshl_add_u32 v23, v21, 7, v14 // (op 8992) 


v_mul_lo_u32 v14, v23, 6 // (op 8992) 
v_lshrrev_b32 v21, 3, v14 // (op 8992) 

v_lshrrev_b32 v14, 4, v23 // (op 8992) 

v_lshlrev_b32 v23, 2, v14 // (op 8992) 

v_add_u32 v14, v21, v23 // (op 8992) 


v_and_b32 v21, -4, v1 // (op 9050) 
v_sub_u32 v23, v1, v21 // (op 9050) 

v_lshrrev_b32 v21, 2, v1 // (op 9050) 
v_lshl_add_u32 v27, v21, 3, v23 // (op 9050) 


v_and_b32 v21, -128, v1 // (op 9108) 
v_sub_u32 v23, v1, v21 // (op 9108) 

v_lshrrev_b32 v21, 7, v1 // (op 9108) 
v_lshl_add_u32 v31, v21, 8, v23 // (op 9108) 


v_and_b32 v21, -2, v1 // (op 9189) 
v_sub_u32 v23, v1, v21 // (op 9189) 

v_lshlrev_b32 v21, 6, v23 // (op 9189) 

v_lshrrev_b32 v23, 1, v1 // (op 9189) 
v_lshl_add_u32 v35, v23, 7, v21 // (op 9189) 


v_mul_lo_u32 v21, v35, 6 // (op 9189) 

v_lshrrev_b32 v23, 3, v21 // (op 9189) 

v_and_b32 v21, -2, v1 // (op 9218) 
v_sub_u32 v35, v1, v21 // (op 9218) 

v_lshlrev_b32 v21, 6, v35 // (op 9218) 

v_lshrrev_b32 v35, 1, v1 // (op 9218) 
v_lshl_add_u32 v61, v35, 7, v21 // (op 9218) 


v_mul_lo_u32 v21, v61, 6 // (op 9218) 
v_lshrrev_b32 v35, 3, v21 // (op 9218) 

v_lshrrev_b32 v21, 4, v61 // (op 9218) 

v_lshlrev_b32 v61, 2, v21 // (op 9218) 

v_add_u32 v21, v35, v61 // (op 9218) 


v_and_b32 v35, -4, v1 // (op 9247) 
v_sub_u32 v61, v1, v35 // (op 9247) 

v_lshrrev_b32 v35, 2, v1 // (op 9247) 
v_lshl_add_u32 v65, v35, 3, v61 // (op 9247) 


v_and_b32 v35, -128, v1 // (op 9276) 
v_sub_u32 v61, v1, v35 // (op 9276) 

v_lshrrev_b32 v35, 7, v1 // (op 9276) 
v_lshl_add_u32 v69, v35, 8, v61 // (op 9276) 


s_mov_b32 s0, 0 // (op 2587) call()
s_ashr_i64 s[6:7], s[28:29], 7 // (op 2586) 
s_mov_b32 s1, s6 // (op 2586) convert

s_and_b32 s6, s1, 4294967294 // (op 2586) 

s_cmp_lt_u32 s0, s6 // (op 2586) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop entry
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_2586 // (op 2586) Condition: Top (jump to RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_2586 if false)
RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopTop_KLoop_2586:


s_barrier  // (op 9380) 
s_add_u32 s0, s0, 2 // (op 2588) 



v_mov_b32 v35, v17 // (op 10) 
s_lshl_b64 s[6:7], s[24:25], 7 // (op 10) 
s_mul_i32 s15, s7, 6 // (op 10) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 10) least significant
s_mul_hi_u32 s1, s6, 6 // (op 10) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 10) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 10) 

s_lshl_b32 s1, s6, 1 // (op 10) 

v_add_u32 v61, v35, s1 // (op 10) 


ds_read_b128 v[90:93], v3 offset:16896 // (op 70) Load local data 
ds_read_b64 v[94:95], v3 offset:16912 // (op 70) Load local data 
buffer_load_dwordx3 v[24:26], v61, s[48:51], 0 offen // (op 10) Load value
s_barrier  // (op 9393) 
buffer_load_dwordx3 v[28:30], v61, s[48:51], 0 offen offset:12 // (op 10) Load value



ds_read_b96_tr_b6 v[96:98], v0 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[99:101], v0 offset:2048 // (op 75) Transpose load local data 
buffer_load_dwordx3 v[32:34], v61, s[48:51], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 9407) 
buffer_load_dwordx3 v[36:38], v61, s[48:51], 0 offen offset:36 // (op 10) Load value





ds_read_u8 v61, v4 offset:29184 // (op 80) Load local data 
v_mov_b64 v[18:19], s[34:35] // (op 25) convert

s_barrier  // (op 9421) 
v_mov_b32 v73, v22 // (op 25) 


ds_read_u8 v39, v5 offset:16384 // (op 85) Load local data 
s_lshl_b32 s1, s34, 3 // (op 25) 



v_add_u32 v40, v73, s1 // (op 25) 

buffer_load_ubyte v42, v40, s[56:59], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v40, v40, v18 // (op 25) 
buffer_load_ubyte v43, v40, s[56:59], 0 offen offset: 0 // (op 25) Load value


ds_read_b96_tr_b6 v[102:104], v0 offset:32 // (op 371) Transpose load local data 


v_mov_b32 v18, v15 // (op 4) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[96:101], a[0:3], v61, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 88) 
ds_read_b96_tr_b6 v[105:107], v0 offset:2080 // (op 371) Transpose load local data 
buffer_load_dwordx4 v[44:47], v18, s[8:11], 0 offen offset:192 // (op 4) Load value


ds_read_u8 v19, v5 offset:16416 // (op 375) Load local data 
buffer_load_dwordx2 v[48:49], v18, s[8:11], 0 offen offset:208 // (op 4) Load value



ds_read_b96_tr_b6 v[108:110], v0 offset:64 // (op 547) Transpose load local data 
ds_read_b96_tr_b6 v[111:113], v0 offset:2112 // (op 547) Transpose load local data 
buffer_load_dwordx4 v[50:53], v18, s[8:11], 0 offen offset:216 // (op 4) Load value


s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:95], v[102:107], a[4:7], v61, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 376) 
ds_read_u8 v40, v5 offset:16448 // (op 551) Load local data 
buffer_load_dwordx2 v[54:55], v18, s[8:11], 0 offen offset:232 // (op 4) Load value




v_mov_b64 v[114:115], s[30:31] // (op 17) convert

v_mov_b32 v41, v20 // (op 17) 
v_mov_b32 v116, v41 // (op 17) 
ds_read_b96_tr_b6 v[118:120], v0 offset:96 // (op 723) Transpose load local data 
ds_read_b96_tr_b6 v[121:123], v0 offset:2144 // (op 723) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[108:113], a[8:11], v61, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 552) 


buffer_load_ubyte v56, v116, s[52:55], 0 offen offset:8 // (op 17) Load value
v_add_u32 v116, v116, v114 // (op 17) 
ds_read_u8 v117, v5 offset:16480 // (op 727) Load local data 
buffer_load_ubyte v57, v116, s[52:55], 0 offen offset:8 // (op 17) Load value





s_barrier  // (op 9444) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:95], v[118:123], a[12:15], v61, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 728) 





ds_read_b128 v[90:93], v3 offset:19968 // (op 1080) Load local data 
ds_read_b64 v[94:95], v3 offset:19984 // (op 1080) Load local data 
s_waitcnt vmcnt(23)
ds_write_b96 v21, v[58:60] offset:29696 // (op 253) Store local data 



ds_read_u8 v61, v4 offset:29312 // (op 1083) Load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v21, v[62:64] offset:29712 // (op 253) Store local data 



ds_read_b128 v[124:127], v3 offset:23040 // (op 1850) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:95], v[96:101], a[16:19], v61, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1223) 
ds_read_b64 v[128:129], v3 offset:23056 // (op 1850) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:95], v[102:107], a[20:23], v61, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1177) 
ds_read_u8 v114, v4 offset:29440 // (op 1853) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:95], v[108:113], a[24:27], v61, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1131) 
ds_read_b128 v[130:133], v3 offset:26112 // (op 2620) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[90:95], v[118:123], a[28:31], v61, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1085) 
ds_read_b64 v[134:135], v3 offset:26128 // (op 2620) Load local data 





s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[124:129], v[96:101], a[32:35], v114, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1993) 
ds_read_u8 v61, v4 offset:29568 // (op 2623) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[124:129], v[102:107], a[36:39], v114, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1947) 
s_waitcnt vmcnt(21)
ds_write_b96 v21, v[66:68] offset:29728 // (op 253) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[124:129], v[108:113], a[40:43], v114, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1901) 
s_waitcnt vmcnt(20)
ds_write_b96 v21, v[70:72] offset:29744 // (op 253) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:129], v[118:123], a[44:47], v114, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1855) 


s_barrier  // (op 9456) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[130:135], v[96:101], a[48:51], v61, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2763) 
s_waitcnt vmcnt(19)
ds_write_b8 v69, v74 offset:46080 // (op 255) Store local data 


s_waitcnt vmcnt(18)
ds_write_b8 v69, v75 offset:46208 // (op 255) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[130:135], v[102:107], a[52:55], v61, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2717) 


s_barrier  // (op 9438) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[130:135], v[108:113], a[56:59], v61, v40 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2671) 
s_waitcnt vmcnt(17)
ds_write_b128 v23, v[76:79] offset:46592 // (op 243) Store local data 


s_waitcnt vmcnt(16)
ds_write_b64 v23, v[80:81] offset:46608 // (op 243) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[130:135], v[118:123], a[60:63], v61, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2625) 
s_waitcnt vmcnt(15)
ds_write_b128 v23, v[82:85] offset:46616 // (op 243) Store local data 




s_waitcnt vmcnt(14)
ds_write_b64 v23, v[86:87] offset:46632 // (op 243) Store local data 
s_barrier  // (op 9450) 



s_waitcnt vmcnt(13)
ds_write_b8 v65, v88 offset:58880 // (op 254) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v65, v89 offset:58884 // (op 254) Store local data 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 256) 
s_mul_i32 s15, s7, 6 // (op 256) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 256) least significant
s_mul_hi_u32 s1, s6, 6 // (op 256) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 256) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 256) 

s_mul_i32 s1, 3, s6 // (op 256) 

v_add_u32 v19, v35, s1 // (op 256) 


buffer_load_dwordx3 v[58:60], v19, s[48:51], 0 offen // (op 256) Load value
buffer_load_dwordx3 v[62:64], v19, s[48:51], 0 offen offset:12 // (op 256) Load value
buffer_load_dwordx3 v[66:68], v19, s[48:51], 0 offen offset:24 // (op 256) Load value
s_waitcnt lgkmcnt(2)
s_barrier  // (op 9435) 
buffer_load_dwordx3 v[70:72], v19, s[48:51], 0 offen offset:36 // (op 256) Load value





ds_read_b128 v[90:93], v7 offset:46592 // (op 245) Load local data 

ds_read_b64 v[94:95], v7 offset:46608 // (op 245) Load local data 
v_mov_b64 v[96:97], s[34:35] // (op 258) convert

s_barrier  // (op 9441) 
s_lshl_b32 s1, s34, 2 // (op 258) 



s_mul_i32 s6, 3, s1 // (op 258) 

v_add_u32 v19, v73, s6 // (op 258) 

buffer_load_ubyte v74, v19, s[56:59], 0 offen offset: 0 // (op 258) Load value
v_add_u32 v19, v19, v96 // (op 258) 
buffer_load_ubyte v75, v19, s[56:59], 0 offen offset: 0 // (op 258) Load value


s_lshl_b64 s[6:7], s[24:25], 7 // (op 8755) 

s_mul_i32 s15, s7, 6 // (op 8755) most significant: low of high * low



s_mul_i32 s14, s6, 6 // (op 8755) least significant
s_mul_hi_u32 s1, s6, 6 // (op 8755) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 8755) most significant: sum


s_lshl_b32 s1, s34, 3 // (op 8883) 
v_add_u32 v22, v22, s1 // (op 8883) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 8755) 

s_lshl_b32 s1, s6, 1 // (op 8755) 

v_add_u32 v17, v17, s1 // (op 8755) 

ds_read_b96_tr_b6 v[96:98], v6 offset:29696 // (op 247) Transpose load local data 
ds_read_b96_tr_b6 v[99:101], v6 offset:31744 // (op 247) Transpose load local data 
buffer_load_dwordx4 v[76:79], v18, s[8:11], 0 offen offset:288 // (op 242) Load value
buffer_load_dwordx2 v[80:81], v18, s[8:11], 0 offen offset:304 // (op 242) Load value
buffer_load_dwordx4 v[82:85], v18, s[8:11], 0 offen offset:312 // (op 242) Load value
s_waitcnt lgkmcnt(4)
s_barrier  // (op 9447) 
buffer_load_dwordx2 v[86:87], v18, s[8:11], 0 offen offset:328 // (op 242) Load value




ds_read_u8 v35, v8 offset:58880 // (op 249) Load local data 

s_barrier  // (op 9453) 
v_add_u32 v15, 192, v15 // (op 8691) 


ds_read_u8 v61, v9 offset:46080 // (op 251) Load local data 
v_mov_b64 v[18:19], s[30:31] // (op 257) convert




v_mov_b32 v73, v41 // (op 257) 
ds_read_b96_tr_b6 v[102:104], v6 offset:29728 // (op 381) Transpose load local data 
buffer_load_ubyte v88, v73, s[52:55], 0 offen offset:12 // (op 257) Load value
v_add_u32 v73, v73, v18 // (op 257) 
buffer_load_ubyte v89, v73, s[52:55], 0 offen offset:12 // (op 257) Load value


s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:95], v[96:101], a[0:3], v35, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 252) 

ds_read_b96_tr_b6 v[105:107], v6 offset:31776 // (op 381) Transpose load local data 


s_barrier  // (op 9396) 
v_add_u32 v20, 8, v20 // (op 8819) 



ds_read_u8 v73, v9 offset:46112 // (op 385) Load local data 
s_waitcnt vmcnt(23)
ds_write_b96 v14, v[24:26] // (op 13) Store local data 



s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:95], v[102:107], a[4:7], v35, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 377) 
ds_read_b96_tr_b6 v[108:110], v6 offset:29760 // (op 557) Transpose load local data 
ds_read_b96_tr_b6 v[111:113], v6 offset:31808 // (op 557) Transpose load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v14, v[28:30] offset:16 // (op 13) Store local data 


ds_read_u8 v18, v9 offset:46144 // (op 561) Load local data 
s_waitcnt vmcnt(21)
ds_write_b96 v14, v[32:34] offset:32 // (op 13) Store local data 



ds_read_b96_tr_b6 v[114:116], v6 offset:29792 // (op 733) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:95], v[108:113], a[8:11], v35, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 553) 
ds_read_b96_tr_b6 v[117:119], v6 offset:31840 // (op 733) Transpose load local data 
s_waitcnt vmcnt(20)
ds_write_b96 v14, v[36:38] offset:48 // (op 13) Store local data 


s_barrier  // (op 9513) 
s_barrier  // (op 9424) 
ds_read_u8 v19, v9 offset:46176 // (op 737) Load local data 



s_waitcnt vmcnt(19)
ds_write_b8 v31, v42 offset:16384 // (op 30) Store local data 



s_barrier  // (op 9517) 
ds_read_b128 v[120:123], v7 offset:49664 // (op 1088) Load local data 
ds_read_b64 v[124:125], v7 offset:49680 // (op 1088) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:95], v[114:119], a[12:15], v35, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 729) 





ds_read_u8 v35, v8 offset:59008 // (op 1091) Load local data 
s_waitcnt vmcnt(18)
ds_write_b8 v31, v43 offset:16512 // (op 30) Store local data 



ds_read_b128 v[90:93], v7 offset:52736 // (op 1858) Load local data 
s_barrier  // (op 9383) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:125], v[96:101], a[16:19], v35, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1224) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:125], v[102:107], a[20:23], v35, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1178) 
ds_read_b64 v[94:95], v7 offset:52752 // (op 1858) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[120:125], v[108:113], a[24:27], v35, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1132) 
ds_read_u8 v39, v8 offset:59136 // (op 1861) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[120:125], v[114:119], a[28:31], v35, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1086) 
ds_read_b128 v[126:129], v7 offset:55808 // (op 2628) Load local data 


ds_read_b64 v[130:131], v7 offset:55824 // (op 2628) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:95], v[96:101], a[32:35], v39, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1994) 



s_barrier  // (op 9553) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:95], v[102:107], a[36:39], v39, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1948) 
ds_read_u8 v35, v8 offset:59264 // (op 2631) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:95], v[108:113], a[40:43], v39, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1902) 
s_barrier  // (op 9557) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:95], v[114:119], a[44:47], v39, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1856) 
s_waitcnt vmcnt(17)
ds_write_b128 v16, v[44:47] offset:16896 // (op 6) Store local data 


s_waitcnt vmcnt(16)
ds_write_b64 v16, v[48:49] offset:16912 // (op 6) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[126:131], v[96:101], a[48:51], v35, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2764) 
s_waitcnt vmcnt(15)
ds_write_b128 v16, v[50:53] offset:16920 // (op 6) Store local data 


s_waitcnt vmcnt(14)
ds_write_b64 v16, v[54:55] offset:16936 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[126:131], v[102:107], a[52:55], v35, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2718) 


s_barrier  // (op 9410) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[126:131], v[108:113], a[56:59], v35, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2672) 
s_waitcnt vmcnt(13)
ds_write_b8 v27, v56 offset:29184 // (op 21) Store local data 


s_waitcnt vmcnt(12)
ds_write_b8 v27, v57 offset:29188 // (op 21) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[126:131], v[114:119], a[60:63], v35, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2626) 




s_ashr_i64 s[6:7], s[28:29], 7 // (op 2586) 
s_mov_b32 s1, s6 // (op 2586) convert

s_and_b32 s6, s1, 4294967294 // (op 2586) 

s_cmp_lt_u32 s0, s6 // (op 2586) 

s_waitcnt lgkmcnt(0)// Sync LDS queue before KLoop back-edge
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopTop_KLoop_2586 // (op 2586) Condition: Bottom (jump to RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopTop_KLoop_2586 if true)
RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoop_2586:

















v_lshrrev_b32 v61, 1, v1 // (op 8639) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8639) 
v_lshl_add_u32 v65, s16, 7, v61 // (op 8639) 

s_lshr_b32 s6, s0, 1 // (op 8639) 

v_and_b32 v61, -2, v1 // (op 8639) 
v_sub_u32 v69, v1, v61 // (op 8639) 

v_lshlrev_b32 v61, 6, v69 // (op 8639) 

v_lshl_add_u32 v69, s6, 8, v61 // (op 8639) 


v_mov_b32 v15, 0 // (op 8639) convert
v_mov_b32 v14, v69 // (op 8639) convert

v_mul_lo_u32 v17, v65, s13 // (op 8639) most significant: low of low * high
v_mul_hi_u32 v61, v65, s12 // (op 8639) most significant: high of low * low
v_mul_lo_u32 v16, v65, s12 // (op 8639) least significant: low of low * low
v_add_u32 v17, v17, v61 // (op 8639) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 8639) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 8639) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 8639) most significant: low of high * low
v_mul_hi_u32 v61, v16, 6 // (op 8639) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 8639) least significant: low of low * low
v_add_u32 v15, v15, v61 // (op 8639) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8639) 

v_mov_b32 v61, v16 // (op 8639) convert

s_mul_i32 s1, s23, 6 // (op 8641) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 8641) least significant
s_mul_hi_u32 s6, s22, 6 // (op 8641) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8641) most significant: sum

s_mul_i32 s7, s12, s1 // (op 8641) most significant: low of low * high
s_mul_i32 s14, s13, s0 // (op 8641) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 8641) least significant
s_mul_hi_u32 s15, s12, s0 // (op 8641) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 8641) most significant: sum
s_add_u32 s7, s7, s15 // (op 8641) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 8641) 

s_mov_b32 s8, s20 // (op 8641) reg expression
s_mov_b32 s9, s21 // (op 8641) reg expression
s_mov_b32 s10, s0 // (op 8641) reg expression
s_mov_b32 s11, 131072 // (op 8641) call()

v_lshrrev_b32 v65, 1, v1 // (op 8721) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8721) 
v_and_b32 v69, -2, v1 // (op 8721) 
s_lshr_b32 s6, s0, 1 // (op 8721) 

v_sub_u32 v73, v1, v69 // (op 8721) 

v_lshl_add_u32 v69, s6, 8, v65 // (op 8721) 


v_lshlrev_b32 v65, 6, v73 // (op 8721) 

v_lshl_add_u32 v73, s17, 7, v65 // (op 8721) 

v_mov_b32 v15, 0 // (op 8721) convert
v_mov_b32 v14, v73 // (op 8721) convert

v_mul_lo_u32 v17, v69, s25 // (op 8721) most significant: low of low * high
v_mul_hi_u32 v65, v69, s24 // (op 8721) most significant: high of low * low
v_mul_lo_u32 v16, v69, s24 // (op 8721) least significant: low of low * low
v_add_u32 v17, v17, v65 // (op 8721) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 8721) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 8721) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 8721) most significant: low of high * low
v_mul_hi_u32 v65, v16, 6 // (op 8721) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 8721) least significant: low of low * low
v_add_u32 v15, v15, v65 // (op 8721) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 8721) 

v_mov_b32 v65, v16 // (op 8721) convert

s_mul_i32 s1, s29, 6 // (op 8723) most significant: low of high * low
s_mul_i32 s0, s28, 6 // (op 8723) least significant
s_mul_hi_u32 s6, s28, 6 // (op 8723) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8723) most significant: sum

s_mul_i32 s7, s24, s1 // (op 8723) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 8723) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 8723) least significant
s_mul_hi_u32 s15, s24, s0 // (op 8723) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 8723) most significant: sum
s_add_u32 s7, s7, s15 // (op 8723) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 8723) 

s_mov_b32 s48, s26 // (op 8723) reg expression
s_mov_b32 s49, s27 // (op 8723) reg expression
s_mov_b32 s50, s0 // (op 8723) reg expression
s_mov_b32 s51, 131072 // (op 8723) call()

v_lshrrev_b32 v69, 2, v1 // (op 8785) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8785) 
v_lshlrev_b32 v73, 1, v69 // (op 8785) 

s_lshr_b32 s6, s0, 1 // (op 8785) 

v_lshl_add_u32 v69, s16, 7, v73 // (op 8785) 

v_and_b32 v73, -4, v1 // (op 8785) 
v_sub_u32 v14, v1, v73 // (op 8785) 

v_lshl_add_u32 v73, s6, 3, v14 // (op 8785) 


v_mul_lo_u32 v14, v69, s30 // (op 8785) 
v_add_u32 v14, v14, v73 // (op 8785) 


s_mul_i32 s1, s30, s23 // (op 8787) most significant: low of low * high
s_mul_i32 s6, s31, s22 // (op 8787) most significant: low of high * low
s_mul_i32 s0, s30, s22 // (op 8787) least significant
s_mul_hi_u32 s7, s30, s22 // (op 8787) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8787) most significant: sum
s_add_u32 s1, s1, s7 // (op 8787) most significant: sum


s_mov_b32 s52, s32 // (op 8787) reg expression
s_mov_b32 s53, s33 // (op 8787) reg expression
s_mov_b32 s54, s0 // (op 8787) reg expression
s_mov_b32 s55, 131072 // (op 8787) call()

v_lshrrev_b32 v69, 7, v1 // (op 8849) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 8849) 
v_lshlrev_b32 v73, 1, v69 // (op 8849) 

s_lshr_b32 s6, s0, 1 // (op 8849) 

v_and_b32 v69, -128, v1 // (op 8849) 
v_lshl_add_u32 v15, s6, 3, v73 // (op 8849) 


v_sub_u32 v73, v1, v69 // (op 8849) 

v_lshl_add_u32 v69, s17, 7, v73 // (op 8849) 

v_mul_lo_u32 v73, v15, s34 // (op 8849) 
v_add_u32 v73, v73, v69 // (op 8849) 


s_mul_i32 s1, s34, s39 // (op 8851) most significant: low of low * high
s_mul_i32 s6, s35, s38 // (op 8851) most significant: low of high * low
s_mul_i32 s0, s34, s38 // (op 8851) least significant
s_mul_hi_u32 s7, s34, s38 // (op 8851) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8851) most significant: sum
s_add_u32 s1, s1, s7 // (op 8851) most significant: sum


s_mov_b32 s56, s36 // (op 8851) reg expression
s_mov_b32 s57, s37 // (op 8851) reg expression
s_mov_b32 s58, s0 // (op 8851) reg expression
s_mov_b32 s59, 131072 // (op 8851) call()

v_and_b32 v69, -2, v1 // (op 8947) 
v_sub_u32 v15, v1, v69 // (op 8947) 

v_lshlrev_b32 v69, 6, v15 // (op 8947) 

v_lshrrev_b32 v15, 1, v1 // (op 8947) 
v_lshl_add_u32 v16, v15, 7, v69 // (op 8947) 


v_mul_lo_u32 v15, v16, 6 // (op 8947) 

v_lshrrev_b32 v69, 3, v15 // (op 8947) 

v_and_b32 v15, -2, v1 // (op 9005) 
v_sub_u32 v16, v1, v15 // (op 9005) 

v_lshlrev_b32 v15, 6, v16 // (op 9005) 

v_lshrrev_b32 v16, 1, v1 // (op 9005) 
v_lshl_add_u32 v17, v16, 7, v15 // (op 9005) 


v_mul_lo_u32 v15, v17, 6 // (op 9005) 
v_lshrrev_b32 v16, 3, v15 // (op 9005) 

v_lshrrev_b32 v15, 4, v17 // (op 9005) 

v_lshlrev_b32 v17, 2, v15 // (op 9005) 

v_add_u32 v15, v16, v17 // (op 9005) 


v_and_b32 v16, -4, v1 // (op 9063) 
v_sub_u32 v17, v1, v16 // (op 9063) 

v_lshrrev_b32 v16, 2, v1 // (op 9063) 
v_lshl_add_u32 v18, v16, 3, v17 // (op 9063) 


v_and_b32 v16, -128, v1 // (op 9121) 
v_sub_u32 v17, v1, v16 // (op 9121) 

v_lshrrev_b32 v16, 7, v1 // (op 9121) 
v_lshl_add_u32 v19, v16, 8, v17 // (op 9121) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 2592) 
s_and_b32 s6, s0, 4294967294 // (op 2592) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 2591) 
s_mov_b32 s7, s0 // (op 2591) convert

s_cmp_lt_u32 s6, s7 // (op 2591) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_2591 // (op 2591) Condition: Top (jump to RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_2591 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_2591:


s_add_u32 s6, s6, 1 // (op 2593) 
v_mov_b32 v16, v61 // (op 176) 

buffer_load_dwordx4 v[20:23], v16, s[8:11], 0 offen // (op 176) Load value
buffer_load_dwordx2 v[24:25], v16, s[8:11], 0 offen offset:16 // (op 176) Load value
buffer_load_dwordx4 v[26:29], v16, s[8:11], 0 offen offset:24 // (op 176) Load value
buffer_load_dwordx2 v[30:31], v16, s[8:11], 0 offen offset:40 // (op 176) Load value

v_add_u32 v61, 96, v61 // (op 8629) 


s_barrier  // (op 9386) 
v_mov_b32 v16, v65 // (op 190) 



v_mov_b32 v17, v16 // (op 190) reg expression

buffer_load_dwordx3 v[32:34], v17, s[48:51], 0 offen // (op 190) Load value
buffer_load_dwordx3 v[36:38], v17, s[48:51], 0 offen offset:12 // (op 190) Load value
s_waitcnt vmcnt(5)
ds_write_b128 v69, v[20:23] offset:16896 // (op 177) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v69, v[24:25] offset:16912 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v69, v[26:29] offset:16920 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v69, v[30:31] offset:16936 // (op 177) Store local data 
buffer_load_dwordx3 v[40:42], v17, s[48:51], 0 offen offset:24 // (op 190) Load value

buffer_load_dwordx3 v[44:46], v17, s[48:51], 0 offen offset:36 // (op 190) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 8713) 
s_mul_i32 s15, s1, 6 // (op 8713) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 8713) least significant
s_mul_hi_u32 s7, s0, 6 // (op 8713) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 8713) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 8713) 

v_add_u32 v65, v65, s0 // (op 8713) 


s_barrier  // (op 9399) 
v_mov_b64 v[16:17], s[30:31] // (op 191) convert




v_mov_b32 v35, v14 // (op 191) 
v_mov_b32 v39, v35 // (op 191) 
buffer_load_ubyte v20, v39, s[52:55], 0 offen offset: 0 // (op 191) Load value
v_add_u32 v39, v39, v16 // (op 191) 
s_waitcnt vmcnt(4)
ds_write_b96 v15, v[32:34] // (op 187) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v15, v[36:38] offset:16 // (op 187) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[40:42] offset:32 // (op 187) Store local data 
buffer_load_ubyte v21, v39, s[52:55], 0 offen offset: 0 // (op 191) Load value



v_add_u32 v14, 4, v14 // (op 8777) 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[44:46] offset:48 // (op 187) Store local data 


v_mov_b64 v[16:17], s[34:35] // (op 192) convert

s_barrier  // (op 9413) 
v_mov_b32 v22, v73 // (op 192) 



v_mov_b32 v23, v22 // (op 192) reg expression
buffer_load_ubyte v24, v23, s[56:59], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v23, v23, v16 // (op 192) 
buffer_load_ubyte v25, v23, s[56:59], 0 offen offset: 0 // (op 192) Load value



s_lshl_b32 s7, s34, 2 // (op 8841) 
v_add_u32 v73, v73, s7 // (op 8841) 

s_waitcnt vmcnt(3)
ds_write_b8 v18, v20 offset:29184 // (op 188) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 9389) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v21 offset:29188 // (op 188) Store local data 




ds_read_b128 v[26:29], v3 offset:16896 // (op 179) Load local data 
s_barrier  // (op 9427) 
ds_read_b64 v[30:31], v3 offset:16912 // (op 179) Load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 9403) 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v24 offset:16384 // (op 189) Store local data 



ds_read_b96_tr_b6 v[32:34], v0 // (op 181) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v0 offset:2048 // (op 181) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v25 offset:16512 // (op 189) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 9417) 



ds_read_u8 v16, v4 offset:29184 // (op 183) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 9431) 


ds_read_u8 v17, v5 offset:16384 // (op 185) Load local data 



ds_read_b96_tr_b6 v[20:22], v0 offset:32 // (op 350) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[32:37], a[0:3], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 186) 
ds_read_b96_tr_b6 v[23:25], v0 offset:2080 // (op 350) Transpose load local data 


ds_read_u8 v38, v5 offset:16416 // (op 354) Load local data 



ds_read_b96_tr_b6 v[40:42], v0 offset:64 // (op 526) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:31], v[20:25], a[4:7], v16, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 355) 
ds_read_b96_tr_b6 v[43:45], v0 offset:2112 // (op 526) Transpose load local data 


ds_read_u8 v39, v5 offset:16448 // (op 530) Load local data 



ds_read_b96_tr_b6 v[46:48], v0 offset:96 // (op 702) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:31], v[40:45], a[8:11], v16, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 531) 
ds_read_b96_tr_b6 v[49:51], v0 offset:2144 // (op 702) Transpose load local data 


ds_read_u8 v52, v5 offset:16480 // (op 706) Load local data 



ds_read_b128 v[54:57], v3 offset:19968 // (op 964) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:31], v[46:51], a[12:15], v16, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 707) 
ds_read_b64 v[58:59], v3 offset:19984 // (op 964) Load local data 





ds_read_u8 v16, v4 offset:29312 // (op 967) Load local data 



ds_read_b128 v[26:29], v3 offset:23040 // (op 1734) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:59], v[32:37], a[16:19], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 969) 
ds_read_b64 v[30:31], v3 offset:23056 // (op 1734) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:59], v[20:25], a[20:23], v16, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1000) 
ds_read_u8 v53, v4 offset:29440 // (op 1737) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:59], v[40:45], a[24:27], v16, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1031) 
ds_read_b128 v[74:77], v3 offset:26112 // (op 2504) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[54:59], v[46:51], a[28:31], v16, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1062) 
ds_read_b64 v[78:79], v3 offset:26128 // (op 2504) Load local data 





s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:31], v[32:37], a[32:35], v53, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1739) 
ds_read_u8 v16, v4 offset:29568 // (op 2507) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[26:31], v[20:25], a[36:39], v53, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[26:31], v[40:45], a[40:43], v53, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1801) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[26:31], v[46:51], a[44:47], v53, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1832) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:79], v[32:37], a[48:51], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2509) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:79], v[20:25], a[52:55], v16, v38 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2540) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:79], v[40:45], a[56:59], v16, v39 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2571) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:79], v[46:51], a[60:63], v16, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2602) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 2591) 
s_mov_b32 s7, s0 // (op 2591) convert

s_cmp_lt_u32 s6, s7 // (op 2591) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_2591 // (op 2591) Condition: Bottom (jump to RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopTop_KLoopTail_2591 if true)
RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ForLoopBottom_KLoopTail_2591:
























s_barrier  // (op 9541) 








v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_133
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


v_accvgpr_read v6, a4 // (op 3561) 
v_accvgpr_read v7, a5 // (op 3561) 
v_accvgpr_read v8, a6 // (op 3561) 
v_accvgpr_read v9, a7 // (op 3561) 
v_mul_f32 v14, v11, v6 // (op 3561) 
v_mul_f32 v15, v11, v7 // (op 3561) 
v_mul_f32 v16, v11, v8 // (op 3561) 
v_mul_f32 v17, v11, v9 // (op 3561) 



s_lshl_b32 s0, s4, 7 // (op 3563) 
v_add_u32 v0, v13, s0 // (op 3563) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3563) Store value


v_accvgpr_read v6, a8 // (op 3573) 
v_accvgpr_read v7, a9 // (op 3573) 
v_accvgpr_read v8, a10 // (op 3573) 
v_accvgpr_read v9, a11 // (op 3573) 
v_mul_f32 v14, v11, v6 // (op 3573) 
v_mul_f32 v15, v11, v7 // (op 3573) 
v_mul_f32 v16, v11, v8 // (op 3573) 
v_mul_f32 v17, v11, v9 // (op 3573) 



s_lshl_b32 s0, s4, 8 // (op 3575) 
v_add_u32 v0, v13, s0 // (op 3575) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3575) Store value


v_accvgpr_read v6, a12 // (op 3585) 
v_accvgpr_read v7, a13 // (op 3585) 
v_accvgpr_read v8, a14 // (op 3585) 
v_accvgpr_read v9, a15 // (op 3585) 
v_mul_f32 v14, v11, v6 // (op 3585) 
v_mul_f32 v15, v11, v7 // (op 3585) 
v_mul_f32 v16, v11, v8 // (op 3585) 
v_mul_f32 v17, v11, v9 // (op 3585) 



s_lshl_b32 s0, s4, 7 // (op 3587) 
s_mul_i32 s1, 3, s0 // (op 3587) 

v_add_u32 v0, v13, s1 // (op 3587) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3587) Store value


v_accvgpr_read v6, a16 // (op 3619) 
v_accvgpr_read v7, a17 // (op 3619) 
v_accvgpr_read v8, a18 // (op 3619) 
v_accvgpr_read v9, a19 // (op 3619) 
v_mul_f32 v14, v11, v6 // (op 3619) 
v_mul_f32 v15, v11, v7 // (op 3619) 
v_mul_f32 v16, v11, v8 // (op 3619) 
v_mul_f32 v17, v11, v9 // (op 3619) 



v_add_u32 v0, 128, v13 // (op 3622) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3622) Store value


v_accvgpr_read v6, a20 // (op 3632) 
v_accvgpr_read v7, a21 // (op 3632) 
v_accvgpr_read v8, a22 // (op 3632) 
v_accvgpr_read v9, a23 // (op 3632) 
v_mul_f32 v14, v11, v6 // (op 3632) 
v_mul_f32 v15, v11, v7 // (op 3632) 
v_mul_f32 v16, v11, v8 // (op 3632) 
v_mul_f32 v17, v11, v9 // (op 3632) 



s_lshl_b32 s0, s4, 7 // (op 3624) 
s_add_u32 s0, s0, 128 // (op 3624) 
v_add_u32 v0, v13, s0 // (op 3624) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3624) Store value


v_accvgpr_read v6, a24 // (op 3637) 
v_accvgpr_read v7, a25 // (op 3637) 
v_accvgpr_read v8, a26 // (op 3637) 
v_accvgpr_read v9, a27 // (op 3637) 
v_mul_f32 v14, v11, v6 // (op 3637) 
v_mul_f32 v15, v11, v7 // (op 3637) 
v_mul_f32 v16, v11, v8 // (op 3637) 
v_mul_f32 v17, v11, v9 // (op 3637) 



s_lshl_b32 s0, s4, 8 // (op 3626) 
s_add_u32 s0, s0, 128 // (op 3626) 
v_add_u32 v0, v13, s0 // (op 3626) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3626) Store value


v_accvgpr_read v6, a28 // (op 3642) 
v_accvgpr_read v7, a29 // (op 3642) 
v_accvgpr_read v8, a30 // (op 3642) 
v_accvgpr_read v9, a31 // (op 3642) 
v_mul_f32 v14, v11, v6 // (op 3642) 
v_mul_f32 v15, v11, v7 // (op 3642) 
v_mul_f32 v16, v11, v8 // (op 3642) 
v_mul_f32 v17, v11, v9 // (op 3642) 



s_lshl_b32 s0, s4, 7 // (op 3628) 
s_mul_i32 s1, 3, s0 // (op 3628) 
s_add_u32 s1, s1, 128 // (op 3628) 

v_add_u32 v0, v13, s1 // (op 3628) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3628) Store value


v_accvgpr_read v6, a32 // (op 3687) 
v_accvgpr_read v7, a33 // (op 3687) 
v_accvgpr_read v8, a34 // (op 3687) 
v_accvgpr_read v9, a35 // (op 3687) 
v_mul_f32 v14, v11, v6 // (op 3687) 
v_mul_f32 v15, v11, v7 // (op 3687) 
v_mul_f32 v16, v11, v8 // (op 3687) 
v_mul_f32 v17, v11, v9 // (op 3687) 



v_add_u32 v0, 256, v13 // (op 3690) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3690) Store value


v_accvgpr_read v6, a36 // (op 3700) 
v_accvgpr_read v7, a37 // (op 3700) 
v_accvgpr_read v8, a38 // (op 3700) 
v_accvgpr_read v9, a39 // (op 3700) 
v_mul_f32 v14, v11, v6 // (op 3700) 
v_mul_f32 v15, v11, v7 // (op 3700) 
v_mul_f32 v16, v11, v8 // (op 3700) 
v_mul_f32 v17, v11, v9 // (op 3700) 



s_lshl_b32 s0, s4, 7 // (op 3692) 
s_add_u32 s0, s0, 256 // (op 3692) 
v_add_u32 v0, v13, s0 // (op 3692) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3692) Store value


v_accvgpr_read v6, a40 // (op 3705) 
v_accvgpr_read v7, a41 // (op 3705) 
v_accvgpr_read v8, a42 // (op 3705) 
v_accvgpr_read v9, a43 // (op 3705) 
v_mul_f32 v14, v11, v6 // (op 3705) 
v_mul_f32 v15, v11, v7 // (op 3705) 
v_mul_f32 v16, v11, v8 // (op 3705) 
v_mul_f32 v17, v11, v9 // (op 3705) 



s_lshl_b32 s0, s4, 8 // (op 3694) 
s_add_u32 s0, s0, 256 // (op 3694) 
v_add_u32 v0, v13, s0 // (op 3694) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3694) Store value


v_accvgpr_read v6, a44 // (op 3710) 
v_accvgpr_read v7, a45 // (op 3710) 
v_accvgpr_read v8, a46 // (op 3710) 
v_accvgpr_read v9, a47 // (op 3710) 
v_mul_f32 v14, v11, v6 // (op 3710) 
v_mul_f32 v15, v11, v7 // (op 3710) 
v_mul_f32 v16, v11, v8 // (op 3710) 
v_mul_f32 v17, v11, v9 // (op 3710) 



s_lshl_b32 s0, s4, 7 // (op 3696) 
s_mul_i32 s1, 3, s0 // (op 3696) 
s_add_u32 s1, s1, 256 // (op 3696) 

v_add_u32 v0, v13, s1 // (op 3696) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3696) Store value


v_accvgpr_read v6, a48 // (op 3755) 
v_accvgpr_read v7, a49 // (op 3755) 
v_accvgpr_read v8, a50 // (op 3755) 
v_accvgpr_read v9, a51 // (op 3755) 
v_mul_f32 v14, v11, v6 // (op 3755) 
v_mul_f32 v15, v11, v7 // (op 3755) 
v_mul_f32 v16, v11, v8 // (op 3755) 
v_mul_f32 v17, v11, v9 // (op 3755) 



v_add_u32 v0, 384, v13 // (op 3758) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3758) Store value


v_accvgpr_read v6, a52 // (op 3768) 
v_accvgpr_read v7, a53 // (op 3768) 
v_accvgpr_read v8, a54 // (op 3768) 
v_accvgpr_read v9, a55 // (op 3768) 
v_mul_f32 v14, v11, v6 // (op 3768) 
v_mul_f32 v15, v11, v7 // (op 3768) 
v_mul_f32 v16, v11, v8 // (op 3768) 
v_mul_f32 v17, v11, v9 // (op 3768) 



s_lshl_b32 s0, s4, 7 // (op 3760) 
s_add_u32 s0, s0, 384 // (op 3760) 
v_add_u32 v0, v13, s0 // (op 3760) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3760) Store value


v_accvgpr_read v6, a56 // (op 3773) 
v_accvgpr_read v7, a57 // (op 3773) 
v_accvgpr_read v8, a58 // (op 3773) 
v_accvgpr_read v9, a59 // (op 3773) 
v_mul_f32 v14, v11, v6 // (op 3773) 
v_mul_f32 v15, v11, v7 // (op 3773) 
v_mul_f32 v16, v11, v8 // (op 3773) 
v_mul_f32 v17, v11, v9 // (op 3773) 



s_lshl_b32 s0, s4, 8 // (op 3762) 
s_add_u32 s0, s0, 384 // (op 3762) 
v_add_u32 v0, v13, s0 // (op 3762) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3762) Store value


v_accvgpr_read v6, a60 // (op 3778) 
v_accvgpr_read v7, a61 // (op 3778) 
v_accvgpr_read v8, a62 // (op 3778) 
v_accvgpr_read v9, a63 // (op 3778) 
v_mul_f32 v14, v11, v6 // (op 3778) 
v_mul_f32 v15, v11, v7 // (op 3778) 
v_mul_f32 v16, v11, v8 // (op 3778) 
v_mul_f32 v17, v11, v9 // (op 3778) 



s_lshl_b32 s0, s4, 7 // (op 3764) 
s_mul_i32 s1, 3, s0 // (op 3764) 
s_add_u32 s1, s1, 384 // (op 3764) 

v_add_u32 v0, v13, s1 // (op 3764) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 3764) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_133
RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ConditionalFalse_0_beta_133:


v_add_u32 v0, 384, v12 // (op 4122) 

buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4122) Load value

v_accvgpr_read v14, a48 // (op 4110) 
v_accvgpr_read v15, a49 // (op 4110) 
v_accvgpr_read v16, a50 // (op 4110) 
v_accvgpr_read v17, a51 // (op 4110) 
v_mul_f32 v18, v11, v14 // (op 4110) 
v_mul_f32 v19, v11, v15 // (op 4110) 
v_mul_f32 v20, v11, v16 // (op 4110) 
v_mul_f32 v21, v11, v17 // (op 4110) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4121) 
v_mul_f32 v15, v10, v7 // (op 4121) 
v_mul_f32 v16, v10, v8 // (op 4121) 
v_mul_f32 v17, v10, v9 // (op 4121) 

v_add_f32 v6, v14, v18 // (op 4111) 
v_add_f32 v7, v15, v19 // (op 4111) 
v_add_f32 v8, v16, v20 // (op 4111) 
v_add_f32 v9, v17, v21 // (op 4111) 




v_add_u32 v0, 384, v13 // (op 4113) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4113) Store value




s_lshl_b32 s0, s2, 7 // (op 4126) 
s_add_u32 s0, s0, 384 // (op 4126) 
v_add_u32 v0, v12, s0 // (op 4126) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4126) Load value

v_accvgpr_read v14, a52 // (op 4136) 
v_accvgpr_read v15, a53 // (op 4136) 
v_accvgpr_read v16, a54 // (op 4136) 
v_accvgpr_read v17, a55 // (op 4136) 
v_mul_f32 v18, v11, v14 // (op 4136) 
v_mul_f32 v19, v11, v15 // (op 4136) 
v_mul_f32 v20, v11, v16 // (op 4136) 
v_mul_f32 v21, v11, v17 // (op 4136) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4124) 
v_mul_f32 v15, v10, v7 // (op 4124) 
v_mul_f32 v16, v10, v8 // (op 4124) 
v_mul_f32 v17, v10, v9 // (op 4124) 

v_add_f32 v6, v14, v18 // (op 4125) 
v_add_f32 v7, v15, v19 // (op 4125) 
v_add_f32 v8, v16, v20 // (op 4125) 
v_add_f32 v9, v17, v21 // (op 4125) 




s_lshl_b32 s0, s4, 7 // (op 4115) 
s_add_u32 s0, s0, 384 // (op 4115) 
v_add_u32 v0, v13, s0 // (op 4115) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4115) Store value




s_lshl_b32 s0, s2, 8 // (op 4130) 
s_add_u32 s0, s0, 384 // (op 4130) 
v_add_u32 v0, v12, s0 // (op 4130) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4130) Load value

v_accvgpr_read v14, a56 // (op 4138) 
v_accvgpr_read v15, a57 // (op 4138) 
v_accvgpr_read v16, a58 // (op 4138) 
v_accvgpr_read v17, a59 // (op 4138) 
v_mul_f32 v18, v11, v14 // (op 4138) 
v_mul_f32 v19, v11, v15 // (op 4138) 
v_mul_f32 v20, v11, v16 // (op 4138) 
v_mul_f32 v21, v11, v17 // (op 4138) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4128) 
v_mul_f32 v15, v10, v7 // (op 4128) 
v_mul_f32 v16, v10, v8 // (op 4128) 
v_mul_f32 v17, v10, v9 // (op 4128) 

v_add_f32 v6, v14, v18 // (op 4129) 
v_add_f32 v7, v15, v19 // (op 4129) 
v_add_f32 v8, v16, v20 // (op 4129) 
v_add_f32 v9, v17, v21 // (op 4129) 




s_lshl_b32 s0, s4, 8 // (op 4117) 
s_add_u32 s0, s0, 384 // (op 4117) 
v_add_u32 v0, v13, s0 // (op 4117) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4117) Store value




s_lshl_b32 s0, s2, 7 // (op 4134) 
s_mul_i32 s1, 3, s0 // (op 4134) 
s_add_u32 s1, s1, 384 // (op 4134) 

v_add_u32 v0, v12, s1 // (op 4134) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4134) Load value

v_accvgpr_read v14, a60 // (op 4140) 
v_accvgpr_read v15, a61 // (op 4140) 
v_accvgpr_read v16, a62 // (op 4140) 
v_accvgpr_read v17, a63 // (op 4140) 
v_mul_f32 v18, v11, v14 // (op 4140) 
v_mul_f32 v19, v11, v15 // (op 4140) 
v_mul_f32 v20, v11, v16 // (op 4140) 
v_mul_f32 v21, v11, v17 // (op 4140) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4132) 
v_mul_f32 v15, v10, v7 // (op 4132) 
v_mul_f32 v16, v10, v8 // (op 4132) 
v_mul_f32 v17, v10, v9 // (op 4132) 

v_add_f32 v6, v14, v18 // (op 4133) 
v_add_f32 v7, v15, v19 // (op 4133) 
v_add_f32 v8, v16, v20 // (op 4133) 
v_add_f32 v9, v17, v21 // (op 4133) 




s_lshl_b32 s0, s4, 7 // (op 4119) 
s_mul_i32 s1, 3, s0 // (op 4119) 
s_add_u32 s1, s1, 384 // (op 4119) 

v_add_u32 v0, v13, s1 // (op 4119) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4119) Store value




v_add_u32 v0, 256, v12 // (op 4043) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4043) Load value

v_accvgpr_read v14, a32 // (op 4031) 
v_accvgpr_read v15, a33 // (op 4031) 
v_accvgpr_read v16, a34 // (op 4031) 
v_accvgpr_read v17, a35 // (op 4031) 
v_mul_f32 v18, v11, v14 // (op 4031) 
v_mul_f32 v19, v11, v15 // (op 4031) 
v_mul_f32 v20, v11, v16 // (op 4031) 
v_mul_f32 v21, v11, v17 // (op 4031) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4042) 
v_mul_f32 v15, v10, v7 // (op 4042) 
v_mul_f32 v16, v10, v8 // (op 4042) 
v_mul_f32 v17, v10, v9 // (op 4042) 

v_add_f32 v6, v14, v18 // (op 4032) 
v_add_f32 v7, v15, v19 // (op 4032) 
v_add_f32 v8, v16, v20 // (op 4032) 
v_add_f32 v9, v17, v21 // (op 4032) 




v_add_u32 v0, 256, v13 // (op 4034) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4034) Store value




s_lshl_b32 s0, s2, 7 // (op 4047) 
s_add_u32 s0, s0, 256 // (op 4047) 
v_add_u32 v0, v12, s0 // (op 4047) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4047) Load value

v_accvgpr_read v14, a36 // (op 4057) 
v_accvgpr_read v15, a37 // (op 4057) 
v_accvgpr_read v16, a38 // (op 4057) 
v_accvgpr_read v17, a39 // (op 4057) 
v_mul_f32 v18, v11, v14 // (op 4057) 
v_mul_f32 v19, v11, v15 // (op 4057) 
v_mul_f32 v20, v11, v16 // (op 4057) 
v_mul_f32 v21, v11, v17 // (op 4057) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4045) 
v_mul_f32 v15, v10, v7 // (op 4045) 
v_mul_f32 v16, v10, v8 // (op 4045) 
v_mul_f32 v17, v10, v9 // (op 4045) 

v_add_f32 v6, v14, v18 // (op 4046) 
v_add_f32 v7, v15, v19 // (op 4046) 
v_add_f32 v8, v16, v20 // (op 4046) 
v_add_f32 v9, v17, v21 // (op 4046) 




s_lshl_b32 s0, s4, 7 // (op 4036) 
s_add_u32 s0, s0, 256 // (op 4036) 
v_add_u32 v0, v13, s0 // (op 4036) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4036) Store value




s_lshl_b32 s0, s2, 8 // (op 4051) 
s_add_u32 s0, s0, 256 // (op 4051) 
v_add_u32 v0, v12, s0 // (op 4051) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4051) Load value

v_accvgpr_read v14, a40 // (op 4059) 
v_accvgpr_read v15, a41 // (op 4059) 
v_accvgpr_read v16, a42 // (op 4059) 
v_accvgpr_read v17, a43 // (op 4059) 
v_mul_f32 v18, v11, v14 // (op 4059) 
v_mul_f32 v19, v11, v15 // (op 4059) 
v_mul_f32 v20, v11, v16 // (op 4059) 
v_mul_f32 v21, v11, v17 // (op 4059) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4049) 
v_mul_f32 v15, v10, v7 // (op 4049) 
v_mul_f32 v16, v10, v8 // (op 4049) 
v_mul_f32 v17, v10, v9 // (op 4049) 

v_add_f32 v6, v14, v18 // (op 4050) 
v_add_f32 v7, v15, v19 // (op 4050) 
v_add_f32 v8, v16, v20 // (op 4050) 
v_add_f32 v9, v17, v21 // (op 4050) 




s_lshl_b32 s0, s4, 8 // (op 4038) 
s_add_u32 s0, s0, 256 // (op 4038) 
v_add_u32 v0, v13, s0 // (op 4038) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4038) Store value




s_lshl_b32 s0, s2, 7 // (op 4055) 
s_mul_i32 s1, 3, s0 // (op 4055) 
s_add_u32 s1, s1, 256 // (op 4055) 

v_add_u32 v0, v12, s1 // (op 4055) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 4055) Load value

v_accvgpr_read v14, a44 // (op 4061) 
v_accvgpr_read v15, a45 // (op 4061) 
v_accvgpr_read v16, a46 // (op 4061) 
v_accvgpr_read v17, a47 // (op 4061) 
v_mul_f32 v18, v11, v14 // (op 4061) 
v_mul_f32 v19, v11, v15 // (op 4061) 
v_mul_f32 v20, v11, v16 // (op 4061) 
v_mul_f32 v21, v11, v17 // (op 4061) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4053) 
v_mul_f32 v15, v10, v7 // (op 4053) 
v_mul_f32 v16, v10, v8 // (op 4053) 
v_mul_f32 v17, v10, v9 // (op 4053) 

v_add_f32 v6, v14, v18 // (op 4054) 
v_add_f32 v7, v15, v19 // (op 4054) 
v_add_f32 v8, v16, v20 // (op 4054) 
v_add_f32 v9, v17, v21 // (op 4054) 




s_lshl_b32 s0, s4, 7 // (op 4040) 
s_mul_i32 s1, 3, s0 // (op 4040) 
s_add_u32 s1, s1, 256 // (op 4040) 

v_add_u32 v0, v13, s1 // (op 4040) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 4040) Store value




v_add_u32 v0, 128, v12 // (op 3964) 

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3964) Load value

v_accvgpr_read v14, a16 // (op 3952) 
v_accvgpr_read v15, a17 // (op 3952) 
v_accvgpr_read v16, a18 // (op 3952) 
v_accvgpr_read v17, a19 // (op 3952) 
v_mul_f32 v18, v11, v14 // (op 3952) 
v_mul_f32 v19, v11, v15 // (op 3952) 
v_mul_f32 v20, v11, v16 // (op 3952) 
v_mul_f32 v21, v11, v17 // (op 3952) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3963) 
v_mul_f32 v15, v10, v7 // (op 3963) 
v_mul_f32 v16, v10, v8 // (op 3963) 
v_mul_f32 v17, v10, v9 // (op 3963) 

v_add_f32 v6, v14, v18 // (op 3953) 
v_add_f32 v7, v15, v19 // (op 3953) 
v_add_f32 v8, v16, v20 // (op 3953) 
v_add_f32 v9, v17, v21 // (op 3953) 




v_add_u32 v0, 128, v13 // (op 3955) 

buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3955) Store value




s_lshl_b32 s0, s2, 7 // (op 3968) 
s_add_u32 s0, s0, 128 // (op 3968) 
v_add_u32 v0, v12, s0 // (op 3968) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3968) Load value

v_accvgpr_read v14, a20 // (op 3978) 
v_accvgpr_read v15, a21 // (op 3978) 
v_accvgpr_read v16, a22 // (op 3978) 
v_accvgpr_read v17, a23 // (op 3978) 
v_mul_f32 v18, v11, v14 // (op 3978) 
v_mul_f32 v19, v11, v15 // (op 3978) 
v_mul_f32 v20, v11, v16 // (op 3978) 
v_mul_f32 v21, v11, v17 // (op 3978) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3966) 
v_mul_f32 v15, v10, v7 // (op 3966) 
v_mul_f32 v16, v10, v8 // (op 3966) 
v_mul_f32 v17, v10, v9 // (op 3966) 

v_add_f32 v6, v14, v18 // (op 3967) 
v_add_f32 v7, v15, v19 // (op 3967) 
v_add_f32 v8, v16, v20 // (op 3967) 
v_add_f32 v9, v17, v21 // (op 3967) 




s_lshl_b32 s0, s4, 7 // (op 3957) 
s_add_u32 s0, s0, 128 // (op 3957) 
v_add_u32 v0, v13, s0 // (op 3957) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3957) Store value




s_lshl_b32 s0, s2, 8 // (op 3972) 
s_add_u32 s0, s0, 128 // (op 3972) 
v_add_u32 v0, v12, s0 // (op 3972) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3972) Load value

v_accvgpr_read v14, a24 // (op 3980) 
v_accvgpr_read v15, a25 // (op 3980) 
v_accvgpr_read v16, a26 // (op 3980) 
v_accvgpr_read v17, a27 // (op 3980) 
v_mul_f32 v18, v11, v14 // (op 3980) 
v_mul_f32 v19, v11, v15 // (op 3980) 
v_mul_f32 v20, v11, v16 // (op 3980) 
v_mul_f32 v21, v11, v17 // (op 3980) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3970) 
v_mul_f32 v15, v10, v7 // (op 3970) 
v_mul_f32 v16, v10, v8 // (op 3970) 
v_mul_f32 v17, v10, v9 // (op 3970) 

v_add_f32 v6, v14, v18 // (op 3971) 
v_add_f32 v7, v15, v19 // (op 3971) 
v_add_f32 v8, v16, v20 // (op 3971) 
v_add_f32 v9, v17, v21 // (op 3971) 




s_lshl_b32 s0, s4, 8 // (op 3959) 
s_add_u32 s0, s0, 128 // (op 3959) 
v_add_u32 v0, v13, s0 // (op 3959) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3959) Store value




s_lshl_b32 s0, s2, 7 // (op 3976) 
s_mul_i32 s1, 3, s0 // (op 3976) 
s_add_u32 s1, s1, 128 // (op 3976) 

v_add_u32 v0, v12, s1 // (op 3976) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3976) Load value

v_accvgpr_read v14, a28 // (op 3982) 
v_accvgpr_read v15, a29 // (op 3982) 
v_accvgpr_read v16, a30 // (op 3982) 
v_accvgpr_read v17, a31 // (op 3982) 
v_mul_f32 v18, v11, v14 // (op 3982) 
v_mul_f32 v19, v11, v15 // (op 3982) 
v_mul_f32 v20, v11, v16 // (op 3982) 
v_mul_f32 v21, v11, v17 // (op 3982) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3974) 
v_mul_f32 v15, v10, v7 // (op 3974) 
v_mul_f32 v16, v10, v8 // (op 3974) 
v_mul_f32 v17, v10, v9 // (op 3974) 

v_add_f32 v6, v14, v18 // (op 3975) 
v_add_f32 v7, v15, v19 // (op 3975) 
v_add_f32 v8, v16, v20 // (op 3975) 
v_add_f32 v9, v17, v21 // (op 3975) 




s_lshl_b32 s0, s4, 7 // (op 3961) 
s_mul_i32 s1, 3, s0 // (op 3961) 
s_add_u32 s1, s1, 128 // (op 3961) 

v_add_u32 v0, v13, s1 // (op 3961) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3961) Store value




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




s_lshl_b32 s0, s2, 7 // (op 3878) 
v_add_u32 v0, v12, s0 // (op 3878) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3878) Load value

v_accvgpr_read v14, a4 // (op 3875) 
v_accvgpr_read v15, a5 // (op 3875) 
v_accvgpr_read v16, a6 // (op 3875) 
v_accvgpr_read v17, a7 // (op 3875) 
v_mul_f32 v18, v11, v14 // (op 3875) 
v_mul_f32 v19, v11, v15 // (op 3875) 
v_mul_f32 v20, v11, v16 // (op 3875) 
v_mul_f32 v21, v11, v17 // (op 3875) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3879) 
v_mul_f32 v15, v10, v7 // (op 3879) 
v_mul_f32 v16, v10, v8 // (op 3879) 
v_mul_f32 v17, v10, v9 // (op 3879) 

v_add_f32 v6, v14, v18 // (op 3876) 
v_add_f32 v7, v15, v19 // (op 3876) 
v_add_f32 v8, v16, v20 // (op 3876) 
v_add_f32 v9, v17, v21 // (op 3876) 




s_lshl_b32 s0, s4, 7 // (op 3877) 
v_add_u32 v0, v13, s0 // (op 3877) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3877) Store value




s_lshl_b32 s0, s2, 8 // (op 3890) 
v_add_u32 v0, v12, s0 // (op 3890) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3890) Load value

v_accvgpr_read v14, a8 // (op 3887) 
v_accvgpr_read v15, a9 // (op 3887) 
v_accvgpr_read v16, a10 // (op 3887) 
v_accvgpr_read v17, a11 // (op 3887) 
v_mul_f32 v18, v11, v14 // (op 3887) 
v_mul_f32 v19, v11, v15 // (op 3887) 
v_mul_f32 v20, v11, v16 // (op 3887) 
v_mul_f32 v21, v11, v17 // (op 3887) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3891) 
v_mul_f32 v15, v10, v7 // (op 3891) 
v_mul_f32 v16, v10, v8 // (op 3891) 
v_mul_f32 v17, v10, v9 // (op 3891) 

v_add_f32 v6, v14, v18 // (op 3888) 
v_add_f32 v7, v15, v19 // (op 3888) 
v_add_f32 v8, v16, v20 // (op 3888) 
v_add_f32 v9, v17, v21 // (op 3888) 




s_lshl_b32 s0, s4, 8 // (op 3889) 
v_add_u32 v0, v13, s0 // (op 3889) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3889) Store value




s_lshl_b32 s0, s2, 7 // (op 3902) 
s_mul_i32 s1, 3, s0 // (op 3902) 

v_add_u32 v0, v12, s1 // (op 3902) 


buffer_load_dwordx4 v[6:9], v0, s[40:43], 0 offen // (op 3902) Load value

v_accvgpr_read v14, a12 // (op 3899) 
v_accvgpr_read v15, a13 // (op 3899) 
v_accvgpr_read v16, a14 // (op 3899) 
v_accvgpr_read v17, a15 // (op 3899) 
v_mul_f32 v18, v11, v14 // (op 3899) 
v_mul_f32 v19, v11, v15 // (op 3899) 
v_mul_f32 v20, v11, v16 // (op 3899) 
v_mul_f32 v21, v11, v17 // (op 3899) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3903) 
v_mul_f32 v15, v10, v7 // (op 3903) 
v_mul_f32 v16, v10, v8 // (op 3903) 
v_mul_f32 v17, v10, v9 // (op 3903) 

v_add_f32 v6, v14, v18 // (op 3900) 
v_add_f32 v7, v15, v19 // (op 3900) 
v_add_f32 v8, v16, v20 // (op 3900) 
v_add_f32 v9, v17, v21 // (op 3900) 




s_lshl_b32 s0, s4, 7 // (op 3901) 
s_mul_i32 s1, 3, s0 // (op 3901) 

v_add_u32 v0, v13, s1 // (op 3901) 


buffer_store_dwordx4 v[6:9], v0, s[44:47], 0 offen // (op 3901) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_ConditionalBottom_0_beta_133:
























s_endpgm  // End of RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC
.LRR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_end:
.size RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC, .LRR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC_end-RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC
  .amdhsa_next_free_vgpr 200
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 59392 // lds bytes
  .amdhsa_accum_offset 136
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F6
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
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC
    .symbol: RR_GEMM_TT_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_LDSSYNC.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 59392
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 136
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

