// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2,@function
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 8918) 
v_and_b32 v3, 63, v1 // (op 8918) 
v_sub_u32 v4, v3, v0 // (op 8918) 

v_lshrrev_b32 v3, 1, v4 // (op 8918) 
v_add_lshl_u32 v5, v0, v3, 7 // (op 8918) 


v_and_b32 v0, 1, v4 // (op 8918) 

v_lshlrev_b32 v3, 3, v0 // (op 8918) 

v_lshrrev_b32 v0, 7, v1 // (op 8918) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 8918) 


v_add_u32 v0, v5, v4 // (op 8918) 


v_and_b32 v3, 48, v1 // (op 8975) 
v_and_b32 v4, 63, v1 // (op 8975) 
v_sub_u32 v5, v4, v3 // (op 8975) 

v_lshrrev_b32 v4, 1, v5 // (op 8975) 
v_add_lshl_u32 v6, v3, v4, 7 // (op 8975) 


v_and_b32 v3, 1, v5 // (op 8975) 

v_lshlrev_b32 v4, 3, v3 // (op 8975) 

v_lshrrev_b32 v3, 6, v1 // (op 8975) 
v_and_b32 v5, 1, v3 // (op 8975) 

v_lshl_add_u32 v3, v5, 4, v4 // (op 8975) 


v_add_u32 v4, v6, v3 // (op 8975) 


v_lshrrev_b32 v3, 7, v1 // (op 9033) 
v_and_b32 v5, 15, v1 // (op 9033) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 9033) 


v_and_b32 v3, 63, v1 // (op 9033) 
v_lshrrev_b32 v5, 4, v3 // (op 9033) 

v_lshl_add_u32 v3, v5, 7, v6 // (op 9033) 


v_lshrrev_b32 v5, 6, v1 // (op 9091) 
v_and_b32 v6, 1, v5 // (op 9091) 

v_and_b32 v5, 15, v1 // (op 9091) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 9091) 


v_and_b32 v5, 63, v1 // (op 9091) 
v_lshrrev_b32 v6, 4, v5 // (op 9091) 

v_lshl_add_u32 v5, v6, 7, v7 // (op 9091) 


v_and_b32 v6, 48, v1 // (op 9158) 
v_and_b32 v7, 63, v1 // (op 9158) 
v_sub_u32 v8, v7, v6 // (op 9158) 

v_lshrrev_b32 v7, 1, v8 // (op 9158) 
v_add_lshl_u32 v9, v6, v7, 7 // (op 9158) 


v_and_b32 v6, 1, v8 // (op 9158) 

v_lshlrev_b32 v7, 3, v6 // (op 9158) 

v_lshrrev_b32 v6, 7, v1 // (op 9158) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9158) 


v_add_u32 v6, v9, v8 // (op 9158) 


v_and_b32 v7, 48, v1 // (op 9202) 
v_and_b32 v8, 63, v1 // (op 9202) 
v_sub_u32 v9, v8, v7 // (op 9202) 

v_lshrrev_b32 v8, 1, v9 // (op 9202) 
v_add_lshl_u32 v10, v7, v8, 7 // (op 9202) 


v_and_b32 v7, 1, v9 // (op 9202) 

v_lshlrev_b32 v8, 3, v7 // (op 9202) 

v_lshrrev_b32 v7, 6, v1 // (op 9202) 
v_and_b32 v9, 1, v7 // (op 9202) 

v_lshl_add_u32 v7, v9, 4, v8 // (op 9202) 


v_add_u32 v8, v10, v7 // (op 9202) 


v_lshrrev_b32 v7, 7, v1 // (op 9231) 
v_and_b32 v9, 15, v1 // (op 9231) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 9231) 


v_and_b32 v7, 63, v1 // (op 9231) 
v_lshrrev_b32 v9, 4, v7 // (op 9231) 

v_lshl_add_u32 v7, v9, 7, v10 // (op 9231) 


v_lshrrev_b32 v9, 6, v1 // (op 9260) 
v_and_b32 v10, 1, v9 // (op 9260) 

v_and_b32 v9, 15, v1 // (op 9260) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9260) 


v_and_b32 v9, 63, v1 // (op 9260) 
v_lshrrev_b32 v10, 4, v9 // (op 9260) 

v_lshl_add_u32 v9, v10, 7, v11 // (op 9260) 


v_lshrrev_b32 v10, 7, v1 // (op 8857) 
v_and_b32 v11, 63, v1 // (op 8857) 
v_lshrrev_b32 v12, 4, v11 // (op 8857) 
v_lshlrev_b32 v13, 2, v12 // (op 8857) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8857) 


v_lshl_add_u32 v10, s16, 7, v12 // (op 8857) 

v_lshrrev_b32 v12, 2, v11 // (op 8857) 

v_and_b32 v11, 3, v12 // (op 8857) 

v_and_b32 v12, 3, v1 // (op 8857) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 8857) 


v_lshrrev_b32 v11, 6, v1 // (op 8857) 
v_and_b32 v12, 1, v11 // (op 8857) 

v_lshl_add_u32 v11, v12, 4, v13 // (op 8857) 


v_lshl_add_u32 v12, s17, 7, v11 // (op 8857) 

v_mul_lo_u32 v11, v12, s2 // (op 8857) 

v_add_lshl_u32 v12, v10, v11, 2 // (op 8857) 


v_lshrrev_b32 v10, 7, v1 // (op 9112) 
v_and_b32 v11, 63, v1 // (op 9112) 
v_lshrrev_b32 v13, 4, v11 // (op 9112) 
s_mul_i32 s1, s2, s9 // (op 8859) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 8859) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 8859) least significant
s_mul_hi_u32 s19, s2, s8 // (op 8859) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8859) most significant: sum
s_add_u32 s1, s1, s19 // (op 8859) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8859) 

v_lshlrev_b32 v14, 2, v13 // (op 9112) 

v_lshl_add_u32 v13, v10, 4, v14 // (op 9112) 


v_lshl_add_u32 v10, s16, 7, v13 // (op 9112) 

v_lshrrev_b32 v13, 2, v11 // (op 9112) 

v_and_b32 v11, 3, v13 // (op 9112) 

v_and_b32 v13, 3, v1 // (op 9112) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 9112) 


v_lshrrev_b32 v11, 6, v1 // (op 9112) 
v_and_b32 v13, 1, v11 // (op 9112) 

v_lshl_add_u32 v11, v13, 4, v14 // (op 9112) 


v_lshl_add_u32 v13, s17, 7, v11 // (op 9112) 

v_mul_lo_u32 v11, v13, s4 // (op 9112) 

v_add_lshl_u32 v13, v10, v11, 2 // (op 9112) 


s_mov_b32 s40, s6 // (op 8859) reg expression
s_mov_b32 s41, s7 // (op 8859) reg expression
s_mov_b32 s42, s18 // (op 8859) reg expression
s_mov_b32 s43, 131072 // (op 8859) call()

s_mul_i32 s1, s4, s9 // (op 9114) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 9114) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 9114) least significant
s_mul_hi_u32 s7, s4, s8 // (op 9114) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 9114) most significant: sum
s_add_u32 s1, s1, s7 // (op 9114) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 9114) 

s_mov_b32 s44, s10 // (op 9114) reg expression
s_mov_b32 s45, s11 // (op 9114) reg expression
s_mov_b32 s46, s6 // (op 9114) reg expression
s_mov_b32 s47, 131072 // (op 9114) call()



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
v_and_b32 v14, -8, v1 // (op 8619) 
v_sub_u32 v15, v1, v14 // (op 8619) 

v_lshlrev_b32 v14, 4, v15 // (op 8619) 

v_lshl_add_u32 v15, s16, 7, v14 // (op 8619) 

v_mov_b32 v17, 0 // (op 8619) convert
v_mov_b32 v16, v15 // (op 8619) convert

v_lshrrev_b32 v14, 3, v1 // (op 8619) 
v_mul_lo_u32 v19, v14, s13 // (op 8619) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 8619) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 8619) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 8619) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 8619) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 8619) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 8621) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8621) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8621) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8621) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8621) most significant: sum
s_add_u32 s1, s1, s7 // (op 8621) most significant: sum


s_mov_b32 s8, s20 // (op 8621) reg expression
s_mov_b32 s9, s21 // (op 8621) reg expression
s_mov_b32 s10, s0 // (op 8621) reg expression
s_mov_b32 s11, 131072 // (op 8621) call()

v_and_b32 v16, -8, v1 // (op 8638) 
v_sub_u32 v17, v1, v16 // (op 8638) 

v_lshlrev_b32 v16, 4, v17 // (op 8638) 

v_lshl_add_u32 v17, s16, 7, v16 // (op 8638) 

v_mov_b32 v19, 0 // (op 8638) convert
v_mov_b32 v18, v17 // (op 8638) convert

v_lshrrev_b32 v16, 3, v1 // (op 8638) 
v_mul_lo_u32 v21, v16, s13 // (op 8638) most significant: low of low * high
v_mul_hi_u32 v17, v16, s12 // (op 8638) most significant: high of low * low
v_mul_lo_u32 v20, v16, s12 // (op 8638) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 8638) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 8638) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 8638) most significant half



s_mul_i32 s1, s12, s23 // (op 8640) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8640) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8640) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8640) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8640) most significant: sum
s_add_u32 s1, s1, s7 // (op 8640) most significant: sum


s_mov_b32 s8, s20 // (op 8640) reg expression
s_mov_b32 s9, s21 // (op 8640) reg expression
s_mov_b32 s10, s0 // (op 8640) reg expression
s_mov_b32 s11, 131072 // (op 8640) call()

v_and_b32 v18, -8, v1 // (op 8683) 
v_sub_u32 v19, v1, v18 // (op 8683) 

v_lshlrev_b32 v18, 4, v19 // (op 8683) 

v_lshl_add_u32 v19, s17, 7, v18 // (op 8683) 

v_mov_b32 v21, 0 // (op 8683) convert
v_mov_b32 v20, v19 // (op 8683) convert

v_lshrrev_b32 v18, 3, v1 // (op 8683) 
v_mul_lo_u32 v23, v18, s25 // (op 8683) most significant: low of low * high
v_mul_hi_u32 v19, v18, s24 // (op 8683) most significant: high of low * low
v_mul_lo_u32 v22, v18, s24 // (op 8683) least significant: low of low * low
v_add_u32 v23, v23, v19 // (op 8683) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v20 // (op 8683) least significant half
v_addc_co_u32 v23, s[0:1], v23, v21, s[0:1] // (op 8683) most significant half



s_mul_i32 s1, s24, s23 // (op 8685) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 8685) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 8685) least significant
s_mul_hi_u32 s7, s24, s22 // (op 8685) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8685) most significant: sum
s_add_u32 s1, s1, s7 // (op 8685) most significant: sum


s_mov_b32 s48, s26 // (op 8685) reg expression
s_mov_b32 s49, s27 // (op 8685) reg expression
s_mov_b32 s50, s0 // (op 8685) reg expression
s_mov_b32 s51, 131072 // (op 8685) call()

v_and_b32 v18, -8, v1 // (op 8702) 
v_sub_u32 v19, v1, v18 // (op 8702) 

v_lshlrev_b32 v18, 4, v19 // (op 8702) 

v_lshl_add_u32 v19, s17, 7, v18 // (op 8702) 

v_mov_b32 v21, 0 // (op 8702) convert
v_mov_b32 v20, v19 // (op 8702) convert

v_lshrrev_b32 v18, 3, v1 // (op 8702) 
v_mul_lo_u32 v25, v18, s25 // (op 8702) most significant: low of low * high
v_mul_hi_u32 v19, v18, s24 // (op 8702) most significant: high of low * low
v_mul_lo_u32 v24, v18, s24 // (op 8702) least significant: low of low * low
v_add_u32 v25, v25, v19 // (op 8702) most significant: sum

v_add_co_u32 v24, s[0:1], v24, v20 // (op 8702) least significant half
v_addc_co_u32 v25, s[0:1], v25, v21, s[0:1] // (op 8702) most significant half



s_mul_i32 s1, s24, s23 // (op 8704) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 8704) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 8704) least significant
s_mul_hi_u32 s7, s24, s22 // (op 8704) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8704) most significant: sum
s_add_u32 s1, s1, s7 // (op 8704) most significant: sum


s_mov_b32 s48, s26 // (op 8704) reg expression
s_mov_b32 s49, s27 // (op 8704) reg expression
s_mov_b32 s50, s0 // (op 8704) reg expression
s_mov_b32 s51, 131072 // (op 8704) call()

v_and_b32 v18, -128, v1 // (op 8747) 
v_sub_u32 v19, v1, v18 // (op 8747) 

v_lshl_add_u32 v18, s16, 7, v19 // (op 8747) 

v_lshrrev_b32 v19, 7, v1 // (op 8747) 
v_lshlrev_b32 v20, 1, v19 // (op 8747) 

v_mul_lo_u32 v19, v20, s28 // (op 8747) 

v_add_u32 v20, v18, v19 // (op 8747) 


s_mul_i32 s1, s28, s33 // (op 8749) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 8749) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8749) least significant
s_mul_hi_u32 s7, s28, s32 // (op 8749) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8749) most significant: sum
s_add_u32 s1, s1, s7 // (op 8749) most significant: sum


s_mov_b32 s52, s30 // (op 8749) reg expression
s_mov_b32 s53, s31 // (op 8749) reg expression
s_mov_b32 s54, s0 // (op 8749) reg expression
s_mov_b32 s55, 131072 // (op 8749) call()

v_and_b32 v21, -128, v1 // (op 8766) 
v_sub_u32 v18, v1, v21 // (op 8766) 

v_lshl_add_u32 v19, s16, 7, v18 // (op 8766) 

v_lshrrev_b32 v18, 7, v1 // (op 8766) 
v_lshlrev_b32 v21, 1, v18 // (op 8766) 

v_mul_lo_u32 v18, v21, s28 // (op 8766) 

v_add_u32 v21, v19, v18 // (op 8766) 


s_mul_i32 s1, s28, s33 // (op 8768) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 8768) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8768) least significant
s_mul_hi_u32 s7, s28, s32 // (op 8768) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8768) most significant: sum
s_add_u32 s1, s1, s7 // (op 8768) most significant: sum


s_mov_b32 s52, s30 // (op 8768) reg expression
s_mov_b32 s53, s31 // (op 8768) reg expression
s_mov_b32 s54, s0 // (op 8768) reg expression
s_mov_b32 s55, 131072 // (op 8768) call()

v_and_b32 v18, -128, v1 // (op 8811) 
v_sub_u32 v19, v1, v18 // (op 8811) 

v_lshl_add_u32 v18, s17, 7, v19 // (op 8811) 

v_lshrrev_b32 v19, 7, v1 // (op 8811) 
v_lshlrev_b32 v26, 1, v19 // (op 8811) 

v_mul_lo_u32 v19, v26, s34 // (op 8811) 
v_add_u32 v19, v19, v18 // (op 8811) 


s_mul_i32 s1, s34, s33 // (op 8813) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 8813) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 8813) least significant
s_mul_hi_u32 s7, s34, s32 // (op 8813) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8813) most significant: sum
s_add_u32 s1, s1, s7 // (op 8813) most significant: sum


s_mov_b32 s56, s36 // (op 8813) reg expression
s_mov_b32 s57, s37 // (op 8813) reg expression
s_mov_b32 s58, s0 // (op 8813) reg expression
s_mov_b32 s59, 131072 // (op 8813) call()

v_and_b32 v18, -128, v1 // (op 8830) 
v_sub_u32 v26, v1, v18 // (op 8830) 

v_lshl_add_u32 v18, s17, 7, v26 // (op 8830) 

v_lshrrev_b32 v26, 7, v1 // (op 8830) 
v_lshlrev_b32 v27, 1, v26 // (op 8830) 

v_mul_lo_u32 v26, v27, s34 // (op 8830) 
v_add_u32 v26, v26, v18 // (op 8830) 


s_mul_i32 s1, s34, s33 // (op 8832) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 8832) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 8832) least significant
s_mul_hi_u32 s7, s34, s32 // (op 8832) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8832) most significant: sum
s_add_u32 s1, s1, s7 // (op 8832) most significant: sum


s_mov_b32 s56, s36 // (op 8832) reg expression
s_mov_b32 s57, s37 // (op 8832) reg expression
s_mov_b32 s58, s0 // (op 8832) reg expression
s_mov_b32 s59, 131072 // (op 8832) call()

v_and_b32 v18, -8, v1 // (op 8958) 
v_sub_u32 v27, v1, v18 // (op 8958) 

v_lshlrev_b32 v18, 4, v27 // (op 8958) 

v_lshrrev_b32 v27, 3, v1 // (op 8958) 
v_lshl_add_u32 v28, v27, 7, v18 // (op 8958) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 8958) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8553) 
v_mov_b64 v[28:29], s[6:7] // (op 8553) convert


v_mov_b64 v[30:31], v[22:23] // (op 8553) reg expression
s_mov_b32 m0, s0 // (op 8553) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 8553) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 8553) most significant half

s_add_u32 m0, m0, 4096 // (op 8553) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8553) Load value direct to lds




v_mov_b64 v[28:29], s[34:35] // (op 8271) convert

v_mov_b32 v18, v19 // (op 8271) reg expression
buffer_load_ubyte v30, v18, s[56:59], 0 offen offset: 0 // (op 8271) Load value
v_add_u32 v18, v18, v28 // (op 8271) 
buffer_load_ubyte v31, v18, s[56:59], 0 offen offset: 0 // (op 8271) Load value


v_and_b32 v18, -8, v1 // (op 8901) 
v_sub_u32 v27, v1, v18 // (op 8901) 

v_lshlrev_b32 v18, 4, v27 // (op 8901) 

v_lshrrev_b32 v27, 3, v1 // (op 8901) 
v_lshl_add_u32 v28, v27, 7, v18 // (op 8901) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 8901) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 8558) 
v_mov_b64 v[28:29], s[6:7] // (op 8558) convert


v_mov_b64 v[32:33], v[14:15] // (op 8558) reg expression
s_add_u32 s1, 16384, s0 // (op 8558) 
s_mov_b32 m0, s1 // (op 8558) reg expression
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 8558) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 8558) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 8558) most significant half

s_add_u32 m0, m0, 4096 // (op 8558) 
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 8558) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 8558) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 8558) most significant half

s_add_u32 m0, m0, 4096 // (op 8558) 
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 8558) Load value direct to lds
v_add_co_u32 v32, s[6:7], v32, v28 // (op 8558) least significant half
v_addc_co_u32 v33, s[6:7], 0, v33, s[6:7] // (op 8558) most significant half

s_add_u32 m0, m0, 4096 // (op 8558) 
buffer_load_dwordx4 v32, s[8:11], 0 offen offset: 0 lds // (op 8558) Load value direct to lds





v_mov_b64 v[28:29], s[28:29] // (op 8277) convert

v_mov_b32 v18, v20 // (op 8277) reg expression
buffer_load_ubyte v32, v18, s[52:55], 0 offen offset: 0 // (op 8277) Load value
v_add_u32 v18, v18, v28 // (op 8277) 
buffer_load_ubyte v33, v18, s[52:55], 0 offen offset: 0 // (op 8277) Load value


v_and_b32 v18, -8, v1 // (op 9185) 
v_sub_u32 v27, v1, v18 // (op 9185) 

v_lshlrev_b32 v18, 4, v27 // (op 9185) 

v_lshrrev_b32 v27, 3, v1 // (op 9185) 
v_lshl_add_u32 v28, v27, 7, v18 // (op 9185) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 9185) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 8563) 
v_mov_b64 v[28:29], s[6:7] // (op 8563) convert


s_lshl_b64 s[6:7], s[24:25], 7 // (op 8563) 
v_mov_b32 v18, s6 // (op 8563) 
v_mov_b32 v27, s7 // (op 8563) 
v_add_co_u32 v34, s[14:15], v22, v18 // (op 8563) least significant half
v_addc_co_u32 v35, s[14:15], v23, v27, s[14:15] // (op 8563) most significant half




s_add_u32 s1, 32768, s0 // (op 8563) 
s_mov_b32 m0, s1 // (op 8563) reg expression
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 8563) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v28 // (op 8563) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 8563) most significant half

s_add_u32 m0, m0, 4096 // (op 8563) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 8563) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v28 // (op 8563) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 8563) most significant half

s_add_u32 m0, m0, 4096 // (op 8563) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 8563) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v28 // (op 8563) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 8563) most significant half

s_add_u32 m0, m0, 4096 // (op 8563) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 8563) Load value direct to lds






v_mov_b64 v[22:23], s[34:35] // (op 8283) convert

s_lshl_b32 s0, s34, 2 // (op 8283) 
v_add_u32 v18, v19, s0 // (op 8283) 

buffer_load_ubyte v28, v18, s[56:59], 0 offen offset: 0 // (op 8283) Load value
v_add_u32 v18, v18, v22 // (op 8283) 
buffer_load_ubyte v29, v18, s[56:59], 0 offen offset: 0 // (op 8283) Load value



v_and_b32 v27, -8, v1 // (op 9141) 
v_sub_u32 v18, v1, v27 // (op 9141) 

v_lshlrev_b32 v19, 4, v18 // (op 9141) 

v_lshrrev_b32 v18, 3, v1 // (op 9141) 
v_lshl_add_u32 v27, v18, 7, v19 // (op 9141) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 9141) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 8565) 
v_mov_b64 v[18:19], s[6:7] // (op 8565) convert


s_lshl_b64 s[6:7], s[12:13], 7 // (op 8565) 
v_mov_b32 v27, s6 // (op 8565) 
v_mov_b32 v22, s7 // (op 8565) 
v_add_co_u32 v34, s[14:15], v14, v27 // (op 8565) least significant half
v_addc_co_u32 v35, s[14:15], v15, v22, s[14:15] // (op 8565) most significant half




s_add_u32 s1, 49152, s0 // (op 8565) 
s_mov_b32 m0, s1 // (op 8565) reg expression
buffer_load_dwordx4 v34, s[8:11], 0 offen offset: 0 lds // (op 8565) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v18 // (op 8565) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 8565) most significant half

s_add_u32 m0, m0, 4096 // (op 8565) 
buffer_load_dwordx4 v34, s[8:11], 0 offen offset: 0 lds // (op 8565) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v18 // (op 8565) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 8565) most significant half

s_add_u32 m0, m0, 4096 // (op 8565) 
buffer_load_dwordx4 v34, s[8:11], 0 offen offset: 0 lds // (op 8565) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v18 // (op 8565) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 8565) most significant half

s_add_u32 m0, m0, 4096 // (op 8565) 
buffer_load_dwordx4 v34, s[8:11], 0 offen offset: 0 lds // (op 8565) Load value direct to lds






v_mov_b64 v[14:15], s[28:29] // (op 8289) convert

s_lshl_b32 s0, s28, 2 // (op 8289) 
v_add_u32 v27, v20, s0 // (op 8289) 

buffer_load_ubyte v18, v27, s[52:55], 0 offen offset: 0 // (op 8289) Load value
v_add_u32 v27, v27, v14 // (op 8289) 
buffer_load_ubyte v19, v27, s[52:55], 0 offen offset: 0 // (op 8289) Load value



v_and_b32 v20, -128, v1 // (op 9074) 
v_sub_u32 v27, v1, v20 // (op 9074) 

v_lshrrev_b32 v20, 7, v1 // (op 9074) 
v_lshl_add_u32 v14, v20, 8, v27 // (op 9074) 





v_add_u32 v15, 65536, v14 // (op 8292) 
s_waitcnt vmcnt(19)
ds_write_b8 v15, v30 // (op 8292) Store local data 

v_add_u32 v15, 65664, v14 // (op 8292) 
s_waitcnt vmcnt(18)
ds_write_b8 v15, v31 // (op 8292) Store local data 


v_and_b32 v20, -128, v1 // (op 9016) 
v_sub_u32 v27, v1, v20 // (op 9016) 

v_lshrrev_b32 v20, 7, v1 // (op 9016) 
v_lshl_add_u32 v14, v20, 8, v27 // (op 9016) 





v_add_u32 v15, 66048, v14 // (op 8294) 
s_waitcnt vmcnt(13)
ds_write_b8 v15, v32 // (op 8294) Store local data 

v_add_u32 v15, 66176, v14 // (op 8294) 
s_waitcnt vmcnt(12)
ds_write_b8 v15, v33 // (op 8294) Store local data 


s_barrier  // (op 9393) 



ds_read_b64_tr_b8 v[34:35], v4 // (op 8329) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:1024 // (op 8329) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:8192 // (op 8329) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:9216 // (op 8329) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 9428) 


v_add_u32 v20, 65536, v5 // (op 8369) 
ds_read_u8 v27, v20 // (op 8369) Load local data 




ds_read_b64_tr_b8 v[42:43], v4 offset:32 // (op 8334) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:1056 // (op 8334) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:8224 // (op 8334) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v4 offset:9248 // (op 8334) Transpose load local data 


v_add_u32 v20, 65568, v5 // (op 8374) 
ds_read_u8 v14, v20 // (op 8374) Load local data 




ds_read_b64_tr_b8 v[50:51], v4 offset:64 // (op 8339) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:1088 // (op 8339) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:8256 // (op 8339) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:9280 // (op 8339) Transpose load local data 


v_add_u32 v15, 65600, v5 // (op 8379) 
ds_read_u8 v20, v15 // (op 8379) Load local data 




ds_read_b64_tr_b8 v[58:59], v4 offset:96 // (op 8344) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:1120 // (op 8344) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:8288 // (op 8344) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v4 offset:9312 // (op 8344) Transpose load local data 


v_add_u32 v15, 65632, v5 // (op 8384) 
ds_read_u8 v22, v15 // (op 8384) Load local data 

s_barrier  // (op 9377) 



ds_read_b64_tr_b8 v[66:67], v0 offset:16384 // (op 8309) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:17408 // (op 8309) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:24576 // (op 8309) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:25600 // (op 8309) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9410) 


v_add_u32 v15, 66048, v3 // (op 8349) 
ds_read_u8 v23, v15 // (op 8349) Load local data 




ds_read_b64_tr_b8 v[74:75], v0 offset:16416 // (op 8314) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:17440 // (op 8314) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v0 offset:24608 // (op 8314) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:25632 // (op 8314) Transpose load local data 


v_add_u32 v15, 66080, v3 // (op 8354) 
ds_read_u8 v82, v15 // (op 8354) Load local data 




ds_read_b64_tr_b8 v[84:85], v0 offset:16448 // (op 8319) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:17472 // (op 8319) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:24640 // (op 8319) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v0 offset:25664 // (op 8319) Transpose load local data 


v_add_u32 v15, 66112, v3 // (op 8359) 
ds_read_u8 v83, v15 // (op 8359) Load local data 




ds_read_b64_tr_b8 v[92:93], v0 offset:16480 // (op 8324) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v0 offset:17504 // (op 8324) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v0 offset:24672 // (op 8324) Transpose load local data 
ds_read_b64_tr_b8 v[98:99], v0 offset:25696 // (op 8324) Transpose load local data 


v_add_u32 v15, 66144, v3 // (op 8364) 
ds_read_u8 v100, v15 // (op 8364) Load local data 

v_and_b32 v15, -8, v1 // (op 8873) 
v_sub_u32 v101, v1, v15 // (op 8873) 

v_lshlrev_b32 v15, 4, v101 // (op 8873) 

v_lshrrev_b32 v101, 3, v1 // (op 8873) 
v_lshl_add_u32 v102, v101, 7, v15 // (op 8873) 


s_nop 0
v_readfirstlane_b32 s0, v102 // (op 8873) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 8930) 
v_sub_u32 v101, v1, v15 // (op 8930) 

v_lshlrev_b32 v15, 4, v101 // (op 8930) 

v_lshrrev_b32 v101, 3, v1 // (op 8930) 
v_lshl_add_u32 v102, v101, 7, v15 // (op 8930) 


s_nop 0
v_readfirstlane_b32 s1, v102 // (op 8930) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -128, v1 // (op 8987) 
v_sub_u32 v101, v1, v15 // (op 8987) 

v_lshrrev_b32 v15, 7, v1 // (op 8987) 
v_lshl_add_u32 v102, v15, 8, v101 // (op 8987) 


v_and_b32 v15, -128, v1 // (op 9045) 
v_sub_u32 v101, v1, v15 // (op 9045) 

v_lshrrev_b32 v15, 7, v1 // (op 9045) 
v_lshl_add_u32 v103, v15, 8, v101 // (op 9045) 


v_and_b32 v15, -8, v1 // (op 9126) 
v_sub_u32 v101, v1, v15 // (op 9126) 

v_lshlrev_b32 v15, 4, v101 // (op 9126) 

v_lshrrev_b32 v101, 3, v1 // (op 9126) 
v_lshl_add_u32 v104, v101, 7, v15 // (op 9126) 


s_nop 0
v_readfirstlane_b32 s6, v104 // (op 9126) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -8, v1 // (op 9170) 
v_sub_u32 v101, v1, v15 // (op 9170) 

v_lshlrev_b32 v15, 4, v101 // (op 9170) 

v_lshrrev_b32 v101, 3, v1 // (op 9170) 
v_lshl_add_u32 v104, v101, 7, v15 // (op 9170) 


s_nop 0
v_readfirstlane_b32 s7, v104 // (op 9170) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -128, v1 // (op 9214) 
v_sub_u32 v101, v1, v15 // (op 9214) 

v_lshrrev_b32 v15, 7, v1 // (op 9214) 
v_lshl_add_u32 v104, v15, 8, v101 // (op 9214) 


v_and_b32 v15, -128, v1 // (op 9243) 
v_sub_u32 v101, v1, v15 // (op 9243) 

v_lshrrev_b32 v15, 7, v1 // (op 9243) 
v_lshl_add_u32 v105, v15, 8, v101 // (op 9243) 


s_mov_b32 s14, 0 // (op 2587) call()
s_ashr_i64 s[18:19], s[22:23], 7 // (op 2586) 
s_mov_b32 s15, s18 // (op 2586) convert

s_and_b32 s18, s15, 4294967294 // (op 2586) 

s_cmp_lt_u32 s14, s18 // (op 2586) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2586 // (op 2586) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2586 if false)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2586:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[34:41], a[0:3], v23, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 88) 
s_barrier  // (op 9396) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[42:49], a[4:7], v23, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 376) 
s_add_u32 s14, s14, 2 // (op 2588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[50:57], a[8:11], v23, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 552) 
s_lshl_b64 s[18:19], s[24:25], 5 // (op 8525) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[58:65], a[12:15], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 728) 
v_mov_b64 v[106:107], s[18:19] // (op 8525) convert


v_mov_b64 v[108:109], v[24:25] // (op 8525) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 8525) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:81], v[34:41], a[16:19], v82, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1223) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 8525) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:81], v[42:49], a[20:23], v82, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1177) 
v_mov_b32 v15, s38 // (op 8525) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:81], v[50:57], a[24:27], v82, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1131) 
v_mov_b32 v101, s39 // (op 8525) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:81], v[58:65], a[28:31], v82, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1085) 
v_add_co_u32 v110, s[18:19], v108, v15 // (op 8525) least significant half
v_addc_co_u32 v111, s[18:19], v109, v101, s[18:19] // (op 8525) most significant half




s_mov_b32 m0, s1 // (op 8525) reg expression
buffer_load_dwordx4 v110, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v110, s[18:19], v110, v106 // (op 8525) least significant half
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:91], v[34:41], a[32:35], v83, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1993) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:91], v[42:49], a[36:39], v83, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1947) 
buffer_load_dwordx4 v110, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:91], v[50:57], a[40:43], v83, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1901) 
v_add_co_u32 v110, s[18:19], v110, v106 // (op 8525) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:91], v[58:65], a[44:47], v83, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1855) 
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
buffer_load_dwordx4 v110, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds
v_add_co_u32 v110, s[18:19], v110, v106 // (op 8525) least significant half
v_addc_co_u32 v111, s[18:19], 0, v111, s[18:19] // (op 8525) most significant half

s_add_u32 m0, m0, 4096 // (op 8525) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:99], v[34:41], a[48:51], v100, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2763) 
buffer_load_dwordx4 v110, s[48:51], 0 offen offset: 0 lds // (op 8525) Load value direct to lds



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:99], v[42:49], a[52:55], v100, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2717) 
v_mov_b64 v[106:107], s[34:35] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:99], v[50:57], a[56:59], v100, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2671) 
v_mov_b32 v15, v26 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:99], v[58:65], a[60:63], v100, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2625) 
s_lshl_b32 s15, s34, 3 // (op 25) 
v_add_u32 v101, v15, s15 // (op 25) 

buffer_load_ubyte v30, v101, s[56:59], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v101, v101, v106 // (op 25) 
buffer_load_ubyte v31, v101, s[56:59], 0 offen offset: 0 // (op 25) Load value


s_barrier  // (op 9380) 

s_lshl_b64 s[18:19], s[12:13], 5 // (op 8520) 
v_mov_b64 v[106:107], s[18:19] // (op 8520) convert


v_mov_b64 v[110:111], v[16:17] // (op 8520) 
s_lshl_b64 s[18:19], s[12:13], 7 // (op 8520) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 8520) 

v_mov_b32 v101, s38 // (op 8520) 
v_mov_b32 v112, s39 // (op 8520) 
v_add_co_u32 v114, s[18:19], v110, v101 // (op 8520) least significant half
v_addc_co_u32 v115, s[18:19], v111, v112, s[18:19] // (op 8520) most significant half




s_add_u32 s15, 16384, s0 // (op 8520) 
s_mov_b32 m0, s15 // (op 8520) reg expression
buffer_load_dwordx4 v114, s[8:11], 0 offen offset: 0 lds // (op 8520) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v106 // (op 8520) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 8520) most significant half

s_add_u32 m0, m0, 4096 // (op 8520) 
buffer_load_dwordx4 v114, s[8:11], 0 offen offset: 0 lds // (op 8520) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v106 // (op 8520) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 8520) most significant half

s_add_u32 m0, m0, 4096 // (op 8520) 
buffer_load_dwordx4 v114, s[8:11], 0 offen offset: 0 lds // (op 8520) Load value direct to lds
v_add_co_u32 v114, s[18:19], v114, v106 // (op 8520) least significant half
v_addc_co_u32 v115, s[18:19], 0, v115, s[18:19] // (op 8520) most significant half

s_add_u32 m0, m0, 4096 // (op 8520) 
buffer_load_dwordx4 v114, s[8:11], 0 offen offset: 0 lds // (op 8520) Load value direct to lds




v_mov_b64 v[106:107], s[28:29] // (op 17) convert

v_mov_b32 v101, v21 // (op 17) 
s_lshl_b32 s15, s28, 3 // (op 17) 
v_add_u32 v112, v101, s15 // (op 17) 

buffer_load_ubyte v32, v112, s[52:55], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v112, v112, v106 // (op 17) 
buffer_load_ubyte v33, v112, s[52:55], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 9468) 



v_add_u32 v106, 66560, v105 // (op 255) 
s_waitcnt vmcnt(19)
ds_write_b8 v106, v28 // (op 255) Store local data 

v_add_u32 v106, 66688, v105 // (op 255) 
s_waitcnt vmcnt(18)
ds_write_b8 v106, v29 // (op 255) Store local data 

s_barrier  // (op 9462) 



v_add_u32 v106, 67072, v104 // (op 254) 
s_waitcnt vmcnt(13)
ds_write_b8 v106, v18 // (op 254) Store local data 

v_add_u32 v106, 67200, v104 // (op 254) 
s_waitcnt vmcnt(12)
ds_write_b8 v106, v19 // (op 254) Store local data 

s_barrier  // (op 9451) 



ds_read_b64_tr_b8 v[112:113], v8 offset:32768 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[114:115], v8 offset:33792 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[116:117], v8 offset:40960 // (op 247) Transpose load local data 
ds_read_b64_tr_b8 v[118:119], v8 offset:41984 // (op 247) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 9465) 


v_add_u32 v106, 66560, v9 // (op 251) 
ds_read_u8 v107, v106 // (op 251) Load local data 




ds_read_b64_tr_b8 v[120:121], v8 offset:32800 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[122:123], v8 offset:33824 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[124:125], v8 offset:40992 // (op 381) Transpose load local data 
ds_read_b64_tr_b8 v[126:127], v8 offset:42016 // (op 381) Transpose load local data 


v_add_u32 v106, 66592, v9 // (op 385) 
ds_read_u8 v82, v106 // (op 385) Load local data 




ds_read_b64_tr_b8 v[128:129], v8 offset:32832 // (op 557) Transpose load local data 
ds_read_b64_tr_b8 v[130:131], v8 offset:33856 // (op 557) Transpose load local data 
ds_read_b64_tr_b8 v[132:133], v8 offset:41024 // (op 557) Transpose load local data 
ds_read_b64_tr_b8 v[134:135], v8 offset:42048 // (op 557) Transpose load local data 


v_add_u32 v106, 66624, v9 // (op 561) 
ds_read_u8 v83, v106 // (op 561) Load local data 




ds_read_b64_tr_b8 v[136:137], v8 offset:32864 // (op 733) Transpose load local data 
ds_read_b64_tr_b8 v[138:139], v8 offset:33888 // (op 733) Transpose load local data 
ds_read_b64_tr_b8 v[140:141], v8 offset:41056 // (op 733) Transpose load local data 
ds_read_b64_tr_b8 v[142:143], v8 offset:42080 // (op 733) Transpose load local data 


v_add_u32 v106, 66656, v9 // (op 737) 
ds_read_u8 v144, v106 // (op 737) Load local data 

s_barrier  // (op 9445) 
s_barrier  // (op 9509) 



ds_read_b64_tr_b8 v[66:67], v6 offset:49152 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v6 offset:50176 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v6 offset:57344 // (op 245) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v6 offset:58368 // (op 245) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9458) 


v_add_u32 v106, 67072, v7 // (op 249) 
ds_read_u8 v23, v106 // (op 249) Load local data 




ds_read_b64_tr_b8 v[74:75], v6 offset:49184 // (op 1088) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v6 offset:50208 // (op 1088) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v6 offset:57376 // (op 1088) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v6 offset:58400 // (op 1088) Transpose load local data 


v_add_u32 v106, 67104, v7 // (op 1091) 
ds_read_u8 v14, v106 // (op 1091) Load local data 




ds_read_b64_tr_b8 v[84:85], v6 offset:49216 // (op 1858) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v6 offset:50240 // (op 1858) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v6 offset:57408 // (op 1858) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v6 offset:58432 // (op 1858) Transpose load local data 


v_add_u32 v106, 67136, v7 // (op 1861) 
ds_read_u8 v20, v106 // (op 1861) Load local data 




ds_read_b64_tr_b8 v[92:93], v6 offset:49248 // (op 2628) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v6 offset:50272 // (op 2628) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v6 offset:57440 // (op 2628) Transpose load local data 
ds_read_b64_tr_b8 v[98:99], v6 offset:58464 // (op 2628) Transpose load local data 


v_add_u32 v106, 67168, v7 // (op 2631) 
ds_read_u8 v100, v106 // (op 2631) Load local data 

s_barrier  // (op 9543) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:73], v[112:119], a[0:3], v23, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 252) 
s_barrier  // (op 9455) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:73], v[120:127], a[4:7], v23, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 377) 
s_lshl_b64 s[18:19], s[24:25], 5 // (op 8548) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:73], v[128:135], a[8:11], v23, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 553) 
v_mov_b64 v[146:147], s[18:19] // (op 8548) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:73], v[136:143], a[12:15], v23, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 729) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 8548) 
s_mul_i32 s39, 3, s19 // (op 8548) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 8548) least significant
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:81], v[112:119], a[16:19], v14, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1224) 
s_mul_hi_u32 s15, 3, s18 // (op 8548) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:81], v[120:127], a[20:23], v14, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1178) 
s_add_u32 s39, s39, s15 // (op 8548) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:81], v[128:135], a[24:27], v14, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1132) 
v_mov_b32 v106, s38 // (op 8548) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:81], v[136:143], a[28:31], v14, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1086) 
v_mov_b32 v145, s39 // (op 8548) 
v_add_co_u32 v148, s[18:19], v108, v106 // (op 8548) least significant half
v_addc_co_u32 v149, s[18:19], v109, v145, s[18:19] // (op 8548) most significant half




s_add_u32 s15, 32768, s7 // (op 8548) 
s_mov_b32 m0, s15 // (op 8548) reg expression
buffer_load_dwordx4 v148, s[48:51], 0 offen offset: 0 lds // (op 8548) Load value direct to lds
v_add_co_u32 v148, s[18:19], v148, v146 // (op 8548) least significant half
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:91], v[112:119], a[32:35], v20, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1994) 
v_addc_co_u32 v149, s[18:19], 0, v149, s[18:19] // (op 8548) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:91], v[120:127], a[36:39], v20, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1948) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:91], v[128:135], a[40:43], v20, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1902) 
s_add_u32 m0, m0, 4096 // (op 8548) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:91], v[136:143], a[44:47], v20, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1856) 
buffer_load_dwordx4 v148, s[48:51], 0 offen offset: 0 lds // (op 8548) Load value direct to lds
v_add_co_u32 v148, s[18:19], v148, v146 // (op 8548) least significant half
v_addc_co_u32 v149, s[18:19], 0, v149, s[18:19] // (op 8548) most significant half

s_add_u32 m0, m0, 4096 // (op 8548) 
buffer_load_dwordx4 v148, s[48:51], 0 offen offset: 0 lds // (op 8548) Load value direct to lds
v_add_co_u32 v148, s[18:19], v148, v146 // (op 8548) least significant half
v_addc_co_u32 v149, s[18:19], 0, v149, s[18:19] // (op 8548) most significant half

s_add_u32 m0, m0, 4096 // (op 8548) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:99], v[112:119], a[48:51], v100, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2764) 
buffer_load_dwordx4 v148, s[48:51], 0 offen offset: 0 lds // (op 8548) Load value direct to lds







v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:99], v[120:127], a[52:55], v100, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2718) 
v_mov_b64 v[106:107], s[34:35] // (op 258) convert


s_lshl_b32 s15, s34, 2 // (op 258) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:99], v[128:135], a[56:59], v100, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2672) 
s_mul_i32 s18, 3, s15 // (op 258) 


v_add_u32 v108, v15, s18 // (op 258) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:99], v[136:143], a[60:63], v100, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2626) 
s_lshl_b64 s[18:19], s[24:25], 7 // (op 8694) 


s_lshl_b64 s[38:39], s[18:19], 1 // (op 8694) 

v_mov_b32 v109, s38 // (op 8694) 
v_mov_b32 v112, s39 // (op 8694) 
v_add_co_u32 v24, s[18:19], v24, v109 // (op 8694) least significant half
v_addc_co_u32 v25, s[18:19], v25, v112, s[18:19] // (op 8694) most significant half




buffer_load_ubyte v28, v108, s[56:59], 0 offen offset: 0 // (op 258) Load value
v_add_u32 v108, v108, v106 // (op 258) 
buffer_load_ubyte v29, v108, s[56:59], 0 offen offset: 0 // (op 258) Load value



s_barrier  // (op 9448) 
s_lshl_b32 s15, s34, 3 // (op 8822) 

v_add_u32 v26, v26, s15 // (op 8822) 

s_lshl_b64 s[18:19], s[12:13], 5 // (op 8543) 
v_mov_b64 v[106:107], s[18:19] // (op 8543) convert


s_lshl_b64 s[18:19], s[12:13], 7 // (op 8543) 
s_mul_i32 s39, 3, s19 // (op 8543) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 8543) least significant
s_mul_hi_u32 s15, 3, s18 // (op 8543) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 8543) most significant: sum


v_mov_b32 v15, s38 // (op 8543) 
v_mov_b32 v108, s39 // (op 8543) 
v_add_co_u32 v112, s[18:19], v110, v15 // (op 8543) least significant half
v_addc_co_u32 v113, s[18:19], v111, v108, s[18:19] // (op 8543) most significant half




s_add_u32 s15, 49152, s6 // (op 8543) 
s_mov_b32 m0, s15 // (op 8543) reg expression
buffer_load_dwordx4 v112, s[8:11], 0 offen offset: 0 lds // (op 8543) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v106 // (op 8543) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 8543) most significant half

s_add_u32 m0, m0, 4096 // (op 8543) 
buffer_load_dwordx4 v112, s[8:11], 0 offen offset: 0 lds // (op 8543) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v106 // (op 8543) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 8543) most significant half

s_add_u32 m0, m0, 4096 // (op 8543) 
buffer_load_dwordx4 v112, s[8:11], 0 offen offset: 0 lds // (op 8543) Load value direct to lds
v_add_co_u32 v112, s[18:19], v112, v106 // (op 8543) least significant half
v_addc_co_u32 v113, s[18:19], 0, v113, s[18:19] // (op 8543) most significant half

s_add_u32 m0, m0, 4096 // (op 8543) 
buffer_load_dwordx4 v112, s[8:11], 0 offen offset: 0 lds // (op 8543) Load value direct to lds





v_mov_b64 v[106:107], s[28:29] // (op 257) convert

s_lshl_b32 s15, s28, 2 // (op 257) 
s_mul_i32 s18, 3, s15 // (op 257) 

v_add_u32 v15, v101, s18 // (op 257) 

s_lshl_b64 s[18:19], s[12:13], 7 // (op 8630) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 8630) 

v_mov_b32 v108, s38 // (op 8630) 
v_mov_b32 v109, s39 // (op 8630) 
v_add_co_u32 v16, s[18:19], v16, v108 // (op 8630) least significant half
v_addc_co_u32 v17, s[18:19], v17, v109, s[18:19] // (op 8630) most significant half




buffer_load_ubyte v18, v15, s[52:55], 0 offen offset: 0 // (op 257) Load value
v_add_u32 v15, v15, v106 // (op 257) 
buffer_load_ubyte v19, v15, s[52:55], 0 offen offset: 0 // (op 257) Load value



s_barrier  // (op 9431) 
s_lshl_b32 s15, s28, 3 // (op 8758) 



v_add_u32 v21, v21, s15 // (op 8758) 

v_add_u32 v15, 65536, v103 // (op 30) 
s_waitcnt vmcnt(19)
ds_write_b8 v15, v30 // (op 30) Store local data 

v_add_u32 v15, 65664, v103 // (op 30) 
s_waitcnt vmcnt(18)
ds_write_b8 v15, v31 // (op 30) Store local data 

s_barrier  // (op 9413) 



v_add_u32 v15, 66048, v102 // (op 21) 
s_waitcnt vmcnt(13)
ds_write_b8 v15, v32 // (op 21) Store local data 

v_add_u32 v15, 66176, v102 // (op 21) 
s_waitcnt vmcnt(12)
ds_write_b8 v15, v33 // (op 21) Store local data 

s_barrier  // (op 9402) 



ds_read_b64_tr_b8 v[34:35], v4 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v4 offset:1024 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v4 offset:8192 // (op 75) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v4 offset:9216 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 9438) 


v_add_u32 v15, 65536, v5 // (op 85) 
ds_read_u8 v27, v15 // (op 85) Load local data 




ds_read_b64_tr_b8 v[42:43], v4 offset:32 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:1056 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:8224 // (op 371) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v4 offset:9248 // (op 371) Transpose load local data 


v_add_u32 v15, 65568, v5 // (op 375) 
ds_read_u8 v14, v15 // (op 375) Load local data 




ds_read_b64_tr_b8 v[50:51], v4 offset:64 // (op 547) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:1088 // (op 547) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:8256 // (op 547) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v4 offset:9280 // (op 547) Transpose load local data 


v_add_u32 v15, 65600, v5 // (op 551) 
ds_read_u8 v20, v15 // (op 551) Load local data 




ds_read_b64_tr_b8 v[58:59], v4 offset:96 // (op 723) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:1120 // (op 723) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:8288 // (op 723) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v4 offset:9312 // (op 723) Transpose load local data 


v_add_u32 v15, 65632, v5 // (op 727) 
ds_read_u8 v22, v15 // (op 727) Load local data 

s_barrier  // (op 9386) 



ds_read_b64_tr_b8 v[66:67], v0 offset:16384 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:17408 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:24576 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:25600 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9420) 


v_add_u32 v15, 66048, v3 // (op 80) 
ds_read_u8 v23, v15 // (op 80) Load local data 




ds_read_b64_tr_b8 v[74:75], v0 offset:16416 // (op 1080) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:17440 // (op 1080) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v0 offset:24608 // (op 1080) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v0 offset:25632 // (op 1080) Transpose load local data 


v_add_u32 v15, 66080, v3 // (op 1083) 
ds_read_u8 v82, v15 // (op 1083) Load local data 




ds_read_b64_tr_b8 v[84:85], v0 offset:16448 // (op 1850) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:17472 // (op 1850) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:24640 // (op 1850) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v0 offset:25664 // (op 1850) Transpose load local data 


v_add_u32 v15, 66112, v3 // (op 1853) 
ds_read_u8 v83, v15 // (op 1853) Load local data 




ds_read_b64_tr_b8 v[92:93], v0 offset:16480 // (op 2620) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v0 offset:17504 // (op 2620) Transpose load local data 
ds_read_b64_tr_b8 v[96:97], v0 offset:24672 // (op 2620) Transpose load local data 
ds_read_b64_tr_b8 v[98:99], v0 offset:25696 // (op 2620) Transpose load local data 


v_add_u32 v15, 66144, v3 // (op 2623) 
ds_read_u8 v100, v15 // (op 2623) Load local data 

s_ashr_i64 s[18:19], s[22:23], 7 // (op 2586) 
s_mov_b32 s15, s18 // (op 2586) convert

s_and_b32 s18, s15, 4294967294 // (op 2586) 

s_cmp_lt_u32 s14, s18 // (op 2586) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2586 // (op 2586) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2586 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2586:

























s_barrier  // (op 9527) 
v_and_b32 v14, -8, v1 // (op 8578) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8578) 
v_sub_u32 v15, v1, v14 // (op 8578) 

s_lshr_b32 s6, s0, 1 // (op 8578) 

v_lshlrev_b32 v14, 4, v15 // (op 8578) 

v_lshl_add_u32 v15, s16, 7, v14 // (op 8578) 

v_mov_b32 v17, 0 // (op 8578) convert
v_mov_b32 v16, v15 // (op 8578) convert

v_lshrrev_b32 v14, 3, v1 // (op 8578) 
v_lshl_add_u32 v15, s6, 8, v14 // (op 8578) 


v_mul_lo_u32 v21, v15, s13 // (op 8578) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 8578) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 8578) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8578) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v20 // (op 8578) least significant half
v_addc_co_u32 v15, s[0:1], v17, v21, s[0:1] // (op 8578) most significant half



s_mul_i32 s1, s12, s23 // (op 8580) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 8580) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 8580) least significant
s_mul_hi_u32 s7, s12, s22 // (op 8580) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8580) most significant: sum
s_add_u32 s1, s1, s7 // (op 8580) most significant: sum


s_mov_b32 s8, s20 // (op 8580) reg expression
s_mov_b32 s9, s21 // (op 8580) reg expression
s_mov_b32 s10, s0 // (op 8580) reg expression
s_mov_b32 s11, 131072 // (op 8580) call()

v_lshrrev_b32 v16, 3, v1 // (op 8660) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8660) 
v_and_b32 v17, -8, v1 // (op 8660) 
s_lshr_b32 s6, s0, 1 // (op 8660) 

v_sub_u32 v20, v1, v17 // (op 8660) 

v_lshl_add_u32 v17, s6, 8, v16 // (op 8660) 


v_lshlrev_b32 v16, 4, v20 // (op 8660) 

v_lshl_add_u32 v20, s17, 7, v16 // (op 8660) 

v_mov_b32 v27, 0 // (op 8660) convert
v_mov_b32 v26, v20 // (op 8660) convert

v_mul_lo_u32 v21, v17, s25 // (op 8660) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 8660) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 8660) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 8660) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v26 // (op 8660) least significant half
v_addc_co_u32 v21, s[0:1], v21, v27, s[0:1] // (op 8660) most significant half



s_mul_i32 s1, s24, s23 // (op 8662) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 8662) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 8662) least significant
s_mul_hi_u32 s7, s24, s22 // (op 8662) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8662) most significant: sum
s_add_u32 s1, s1, s7 // (op 8662) most significant: sum


s_mov_b32 s48, s26 // (op 8662) reg expression
s_mov_b32 s49, s27 // (op 8662) reg expression
s_mov_b32 s50, s0 // (op 8662) reg expression
s_mov_b32 s51, 131072 // (op 8662) call()

v_and_b32 v16, -128, v1 // (op 8724) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8724) 
v_sub_u32 v17, v1, v16 // (op 8724) 

s_lshr_b32 s6, s0, 1 // (op 8724) 

v_lshl_add_u32 v16, s16, 7, v17 // (op 8724) 

v_lshrrev_b32 v17, 7, v1 // (op 8724) 
v_lshlrev_b32 v22, 1, v17 // (op 8724) 

v_lshl_add_u32 v17, s6, 3, v22 // (op 8724) 


v_mul_lo_u32 v22, v17, s28 // (op 8724) 

v_add_u32 v17, v16, v22 // (op 8724) 


s_mul_i32 s1, s28, s33 // (op 8726) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 8726) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 8726) least significant
s_mul_hi_u32 s7, s28, s32 // (op 8726) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8726) most significant: sum
s_add_u32 s1, s1, s7 // (op 8726) most significant: sum


s_mov_b32 s52, s30 // (op 8726) reg expression
s_mov_b32 s53, s31 // (op 8726) reg expression
s_mov_b32 s54, s0 // (op 8726) reg expression
s_mov_b32 s55, 131072 // (op 8726) call()

v_lshrrev_b32 v16, 7, v1 // (op 8788) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 8788) 
v_lshlrev_b32 v22, 1, v16 // (op 8788) 

s_lshr_b32 s6, s0, 1 // (op 8788) 

v_and_b32 v16, -128, v1 // (op 8788) 
v_lshl_add_u32 v23, s6, 3, v22 // (op 8788) 


v_sub_u32 v22, v1, v16 // (op 8788) 

v_lshl_add_u32 v16, s17, 7, v22 // (op 8788) 

v_mul_lo_u32 v22, v23, s34 // (op 8788) 
v_add_u32 v22, v22, v16 // (op 8788) 


s_mul_i32 s1, s34, s33 // (op 8790) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 8790) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 8790) least significant
s_mul_hi_u32 s7, s34, s32 // (op 8790) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 8790) most significant: sum
s_add_u32 s1, s1, s7 // (op 8790) most significant: sum


s_mov_b32 s56, s36 // (op 8790) reg expression
s_mov_b32 s57, s37 // (op 8790) reg expression
s_mov_b32 s58, s0 // (op 8790) reg expression
s_mov_b32 s59, 131072 // (op 8790) call()

v_and_b32 v16, -8, v1 // (op 8886) 
v_sub_u32 v23, v1, v16 // (op 8886) 

v_lshlrev_b32 v16, 4, v23 // (op 8886) 

v_lshrrev_b32 v23, 3, v1 // (op 8886) 
v_lshl_add_u32 v24, v23, 7, v16 // (op 8886) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 8886) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 8943) 
v_sub_u32 v23, v1, v16 // (op 8943) 

v_lshlrev_b32 v16, 4, v23 // (op 8943) 

v_lshrrev_b32 v23, 3, v1 // (op 8943) 
v_lshl_add_u32 v24, v23, 7, v16 // (op 8943) 


s_nop 0
v_readfirstlane_b32 s1, v24 // (op 8943) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -128, v1 // (op 9000) 
v_sub_u32 v23, v1, v16 // (op 9000) 

v_lshrrev_b32 v16, 7, v1 // (op 9000) 
v_lshl_add_u32 v24, v16, 8, v23 // (op 9000) 


v_and_b32 v16, -128, v1 // (op 9058) 
v_sub_u32 v23, v1, v16 // (op 9058) 

v_lshrrev_b32 v16, 7, v1 // (op 9058) 
v_lshl_add_u32 v25, v16, 8, v23 // (op 9058) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 2592) 
s_and_b32 s14, s6, 4294967294 // (op 2592) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 2591) 
s_mov_b32 s15, s6 // (op 2591) convert

s_cmp_lt_u32 s14, s15 // (op 2591) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2591 // (op 2591) Condition: Top (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2591 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2591:
s_barrier  // (op 9383) 
s_add_u32 s14, s14, 1 // (op 2593) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 8530) 
v_mov_b64 v[18:19], s[6:7] // (op 8530) convert


v_mov_b64 v[26:27], v[14:15] // (op 8530) 
v_mov_b64 v[28:29], v[26:27] // (op 8530) reg expression
s_add_u32 s15, 16384, s0 // (op 8530) 
s_mov_b32 m0, s15 // (op 8530) reg expression
buffer_load_dwordx4 v28, s[8:11], 0 offen offset: 0 lds // (op 8530) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 8530) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8530) most significant half

s_add_u32 m0, m0, 4096 // (op 8530) 
buffer_load_dwordx4 v28, s[8:11], 0 offen offset: 0 lds // (op 8530) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 8530) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8530) most significant half

s_add_u32 m0, m0, 4096 // (op 8530) 
buffer_load_dwordx4 v28, s[8:11], 0 offen offset: 0 lds // (op 8530) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 8530) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8530) most significant half

s_add_u32 m0, m0, 4096 // (op 8530) 
buffer_load_dwordx4 v28, s[8:11], 0 offen offset: 0 lds // (op 8530) Load value direct to lds




s_lshl_b64 s[6:7], s[12:13], 7 // (op 8568) 
v_mov_b32 v16, s6 // (op 8568) 
v_mov_b32 v23, s7 // (op 8568) 
v_add_co_u32 v14, s[18:19], v14, v16 // (op 8568) least significant half
v_addc_co_u32 v15, s[18:19], v15, v23, s[18:19] // (op 8568) most significant half




s_barrier  // (op 9399) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 8536) 
v_mov_b64 v[18:19], s[6:7] // (op 8536) convert


v_mov_b64 v[26:27], v[20:21] // (op 8536) 
v_mov_b64 v[28:29], v[26:27] // (op 8536) reg expression
s_mov_b32 m0, s1 // (op 8536) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8536) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 8536) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8536) most significant half

s_add_u32 m0, m0, 4096 // (op 8536) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8536) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 8536) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8536) most significant half

s_add_u32 m0, m0, 4096 // (op 8536) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8536) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v18 // (op 8536) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 8536) most significant half

s_add_u32 m0, m0, 4096 // (op 8536) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8536) Load value direct to lds



s_lshl_b64 s[6:7], s[24:25], 7 // (op 8652) 
v_mov_b32 v16, s6 // (op 8652) 
v_mov_b32 v23, s7 // (op 8652) 
v_add_co_u32 v20, s[18:19], v20, v16 // (op 8652) least significant half
v_addc_co_u32 v21, s[18:19], v21, v23, s[18:19] // (op 8652) most significant half





v_mov_b64 v[18:19], s[28:29] // (op 191) convert

v_mov_b32 v16, v17 // (op 191) 
v_mov_b32 v23, v16 // (op 191) reg expression
buffer_load_ubyte v26, v23, s[52:55], 0 offen offset: 0 // (op 191) Load value
v_add_u32 v23, v23, v18 // (op 191) 
buffer_load_ubyte v27, v23, s[52:55], 0 offen offset: 0 // (op 191) Load value



s_lshl_b32 s15, s28, 2 // (op 8716) 
v_add_u32 v17, v17, s15 // (op 8716) 


s_barrier  // (op 9416) 
v_mov_b64 v[18:19], s[34:35] // (op 192) convert




v_mov_b32 v16, v22 // (op 192) 
v_mov_b32 v23, v16 // (op 192) reg expression
v_add_u32 v28, 66048, v24 // (op 188) 
buffer_load_ubyte v30, v23, s[56:59], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v23, v23, v18 // (op 192) 
buffer_load_ubyte v31, v23, s[56:59], 0 offen offset: 0 // (op 192) Load value



s_lshl_b32 s15, s34, 2 // (op 8780) 
v_add_u32 v22, v22, s15 // (op 8780) 

s_waitcnt vmcnt(3)
ds_write_b8 v28, v26 // (op 188) Store local data 

s_barrier  // (op 9389) 
v_add_u32 v16, 66176, v24 // (op 188) 



ds_read_b64_tr_b8 v[32:33], v0 offset:16384 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:17408 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:24576 // (op 179) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:25600 // (op 179) Transpose load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v16, v27 // (op 188) Store local data 


s_barrier  // (op 9406) 



s_barrier  // (op 9434) 
ds_read_b64_tr_b8 v[40:41], v4 // (op 181) Transpose load local data 



v_add_u32 v16, 65536, v25 // (op 189) 
ds_read_b64_tr_b8 v[42:43], v4 offset:1024 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v4 offset:8192 // (op 181) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v4 offset:9216 // (op 181) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v30 // (op 189) Store local data 

v_add_u32 v16, 65664, v25 // (op 189) 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 9424) 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v31 // (op 189) Store local data 




v_add_u32 v16, 66048, v3 // (op 183) 
ds_read_u8 v23, v16 // (op 183) Load local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 9441) 


v_add_u32 v16, 65536, v5 // (op 185) 
ds_read_u8 v18, v16 // (op 185) Load local data 




ds_read_b64_tr_b8 v[48:49], v4 offset:32 // (op 350) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v23, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 186) 
ds_read_b64_tr_b8 v[50:51], v4 offset:1056 // (op 350) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:8224 // (op 350) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:9248 // (op 350) Transpose load local data 


v_add_u32 v16, 65568, v5 // (op 354) 
ds_read_u8 v19, v16 // (op 354) Load local data 




ds_read_b64_tr_b8 v[56:57], v4 offset:64 // (op 526) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v4 offset:1088 // (op 526) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[48:55], a[4:7], v23, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 355) 
ds_read_b64_tr_b8 v[60:61], v4 offset:8256 // (op 526) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:9280 // (op 526) Transpose load local data 


v_add_u32 v16, 65600, v5 // (op 530) 
ds_read_u8 v26, v16 // (op 530) Load local data 




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:39], v[56:63], a[8:11], v23, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 531) 
ds_read_b64_tr_b8 v[64:65], v4 offset:96 // (op 702) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v4 offset:1120 // (op 702) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v4 offset:8288 // (op 702) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v4 offset:9312 // (op 702) Transpose load local data 


v_add_u32 v16, 65632, v5 // (op 706) 
ds_read_u8 v27, v16 // (op 706) Load local data 




ds_read_b64_tr_b8 v[72:73], v0 offset:16416 // (op 964) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:39], v[64:71], a[12:15], v23, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 707) 
ds_read_b64_tr_b8 v[74:75], v0 offset:17440 // (op 964) Transpose load local data 


ds_read_b64_tr_b8 v[76:77], v0 offset:24608 // (op 964) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v0 offset:25632 // (op 964) Transpose load local data 


v_add_u32 v16, 66080, v3 // (op 967) 
ds_read_u8 v23, v16 // (op 967) Load local data 




ds_read_b64_tr_b8 v[28:29], v0 offset:16448 // (op 1734) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:17472 // (op 1734) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:79], v[40:47], a[16:19], v23, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 969) 
ds_read_b64_tr_b8 v[32:33], v0 offset:24640 // (op 1734) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:79], v[48:55], a[20:23], v23, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1000) 
ds_read_b64_tr_b8 v[34:35], v0 offset:25664 // (op 1734) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[56:63], a[24:27], v23, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1031) 
v_add_u32 v16, 66112, v3 // (op 1737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[64:71], a[28:31], v23, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1062) 
ds_read_u8 v36, v16 // (op 1737) Load local data 






s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[28:35], v[40:47], a[32:35], v36, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1739) 
ds_read_b64_tr_b8 v[72:73], v0 offset:16480 // (op 2504) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[28:35], v[48:55], a[36:39], v36, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1770) 
ds_read_b64_tr_b8 v[74:75], v0 offset:17504 // (op 2504) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[28:35], v[56:63], a[40:43], v36, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1801) 
ds_read_b64_tr_b8 v[76:77], v0 offset:24672 // (op 2504) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[28:35], v[64:71], a[44:47], v36, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1832) 
ds_read_b64_tr_b8 v[78:79], v0 offset:25696 // (op 2504) Transpose load local data 




v_add_u32 v16, 66144, v3 // (op 2507) 
ds_read_u8 v23, v16 // (op 2507) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:79], v[40:47], a[48:51], v23, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2509) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[72:79], v[48:55], a[52:55], v23, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2540) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:79], v[56:63], a[56:59], v23, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2571) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[72:79], v[64:71], a[60:63], v23, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2602) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 2591) 
s_mov_b32 s15, s6 // (op 2591) convert

s_cmp_lt_u32 s14, s15 // (op 2591) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2591 // (op 2591) Condition: Bottom (jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_2591 if true)
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_2591:























s_barrier  // (op 9531) 








v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_133
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
s_branch RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_133:


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
RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_133:
























s_endpgm  // End of RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.LRR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end:
.size RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2, .LRR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end-RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
  .amdhsa_next_free_vgpr 216
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 67584 // lds bytes
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: False
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
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB } ]
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
  - .name: RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
    .symbol: RR_GEMM_NT_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 67584
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 150
    .agpr_count: 64
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
      - .name: Tensor_2_size_1_11
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
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 104
        .value_kind: by_value
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
...

.end_amdgpu_metadata

