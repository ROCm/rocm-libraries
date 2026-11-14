// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2,@function
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 1206) 
v_and_b32 v3, 63, v1 // (op 1206) 
v_sub_u32 v4, v3, v0 // (op 1206) 

v_lshrrev_b32 v3, 1, v4 // (op 1206) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 1206) 


v_and_b32 v0, 1, v4 // (op 1206) 

v_lshlrev_b32 v3, 3, v0 // (op 1206) 

v_lshrrev_b32 v0, 7, v1 // (op 1206) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1206) 


v_add_u32 v0, v5, v4 // (op 1206) 


v_and_b32 v3, 48, v1 // (op 1379) 
v_and_b32 v4, 63, v1 // (op 1379) 
v_sub_u32 v5, v4, v3 // (op 1379) 

v_lshrrev_b32 v4, 1, v5 // (op 1379) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 1379) 


v_and_b32 v3, 1, v5 // (op 1379) 

v_lshlrev_b32 v4, 3, v3 // (op 1379) 

v_lshrrev_b32 v3, 7, v1 // (op 1379) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1379) 


v_add_u32 v3, v6, v5 // (op 1379) 


v_lshrrev_b32 v4, 7, v1 // (op 1150) 
v_and_b32 v5, 63, v1 // (op 1150) 
v_lshrrev_b32 v6, 4, v5 // (op 1150) 
v_lshlrev_b32 v7, 2, v6 // (op 1150) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1150) 


v_lshl_add_u32 v4, s16, 4, v6 // (op 1150) 

v_lshrrev_b32 v6, 2, v5 // (op 1150) 

v_and_b32 v5, 3, v6 // (op 1150) 

v_and_b32 v6, 3, v1 // (op 1150) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1150) 


v_lshrrev_b32 v5, 6, v1 // (op 1150) 
v_and_b32 v6, 1, v5 // (op 1150) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1150) 


v_lshl_add_u32 v6, s17, 6, v5 // (op 1150) 

v_mul_lo_u32 v5, v6, s2 // (op 1150) 

v_add_lshl_u32 v6, v4, v5, 2 // (op 1150) 


v_and_b32 v4, 63, v1 // (op 1264) 
v_and_b32 v5, 48, v1 // (op 1264) 
v_sub_u32 v7, v4, v5 // (op 1264) 

v_lshrrev_b32 v5, 4, v4 // (op 1264) 

v_lshl_add_u32 v4, v5, 5, v7 // (op 1264) 


v_lshrrev_b32 v5, 6, v1 // (op 1264) 
v_and_b32 v7, 1, v5 // (op 1264) 

v_lshlrev_b32 v5, 4, v7 // (op 1264) 

v_lshl_add_u32 v7, v4, 6, v5 // (op 1264) 


v_mul_lo_u32 v4, v7, 6 // (op 1264) 
v_lshrrev_b32 v5, 3, v4 // (op 1264) 

v_lshrrev_b32 v4, 4, v7 // (op 1264) 

v_lshlrev_b32 v7, 2, v4 // (op 1264) 

v_add_u32 v4, v5, v7 // (op 1264) 


v_lshrrev_b32 v5, 6, v1 // (op 1322) 
v_and_b32 v7, 1, v5 // (op 1322) 

v_and_b32 v5, 15, v1 // (op 1322) 
s_mul_i32 s1, s2, s9 // (op 1152) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1152) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1152) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1152) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1152) most significant: sum
s_add_u32 s1, s1, s19 // (op 1152) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1152) 

v_lshl_add_u32 v8, v7, 4, v5 // (op 1322) 


v_and_b32 v5, 63, v1 // (op 1322) 
v_lshrrev_b32 v7, 4, v5 // (op 1322) 

v_lshl_add_u32 v5, v7, 6, v8 // (op 1322) 


v_lshrrev_b32 v7, 7, v1 // (op 1338) 
v_and_b32 v8, 63, v1 // (op 1338) 
v_lshrrev_b32 v9, 4, v8 // (op 1338) 
v_lshlrev_b32 v10, 2, v9 // (op 1338) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1338) 


v_lshl_add_u32 v7, s16, 4, v9 // (op 1338) 

v_lshrrev_b32 v9, 2, v8 // (op 1338) 

v_and_b32 v8, 3, v9 // (op 1338) 

v_and_b32 v9, 3, v1 // (op 1338) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1338) 


v_lshrrev_b32 v8, 6, v1 // (op 1338) 
v_and_b32 v9, 1, v8 // (op 1338) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1338) 


v_lshl_add_u32 v9, s17, 6, v8 // (op 1338) 

v_mul_lo_u32 v8, v9, s4 // (op 1338) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 1338) 


v_and_b32 v7, 63, v1 // (op 1408) 
v_and_b32 v8, 48, v1 // (op 1408) 
v_sub_u32 v10, v7, v8 // (op 1408) 

v_lshrrev_b32 v8, 4, v7 // (op 1408) 

v_lshl_add_u32 v7, v8, 5, v10 // (op 1408) 


v_lshrrev_b32 v8, 6, v1 // (op 1408) 
v_and_b32 v10, 1, v8 // (op 1408) 

v_lshlrev_b32 v8, 4, v10 // (op 1408) 

v_lshl_add_u32 v10, v7, 6, v8 // (op 1408) 


v_mul_lo_u32 v7, v10, 6 // (op 1408) 
v_lshrrev_b32 v8, 3, v7 // (op 1408) 

v_lshrrev_b32 v7, 4, v10 // (op 1408) 

v_lshlrev_b32 v10, 2, v7 // (op 1408) 

v_add_u32 v7, v8, v10 // (op 1408) 


v_lshrrev_b32 v8, 6, v1 // (op 1437) 
v_and_b32 v10, 1, v8 // (op 1437) 

v_and_b32 v8, 15, v1 // (op 1437) 
s_mul_i32 s1, s4, s9 // (op 1340) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v8 // (op 1437) 


v_and_b32 v8, 63, v1 // (op 1437) 
v_lshrrev_b32 v10, 4, v8 // (op 1437) 

v_lshl_add_u32 v8, v10, 6, v11 // (op 1437) 


s_mul_i32 s38, s5, s8 // (op 1340) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1340) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1340) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1340) most significant: sum
s_add_u32 s1, s1, s39 // (op 1340) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 1340) 

s_mov_b32 s40, s6 // (op 1152) reg expression
s_mov_b32 s41, s7 // (op 1152) reg expression
s_mov_b32 s42, s18 // (op 1152) reg expression
s_mov_b32 s43, 131072 // (op 1152) call()

s_mov_b32 s44, s10 // (op 1340) reg expression

s_mov_b32 s45, s11 // (op 1340) reg expression
s_mov_b32 s46, s38 // (op 1340) reg expression
s_mov_b32 s47, 131072 // (op 1340) call()



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
s_lshl_b32 s0, s16, 4 // (op 917) 
v_mul_lo_u32 v13, v1, s13 // (op 917) most significant: low of low * high
s_mov_b32 s15, 0 // (op 917) convert
s_mov_b32 s14, s0 // (op 917) convert

v_mul_hi_u32 v14, v1, s12 // (op 917) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 917) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 917) most significant: sum

v_mov_b32 v14, s14 // (op 917) 
v_mov_b32 v15, s15 // (op 917) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 917) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 917) most significant half






s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 919) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 919) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 919) least significant
s_mul_hi_u32 s7, s12, s22 // (op 919) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 919) most significant: sum
s_add_u32 s1, s1, s7 // (op 919) most significant: sum


s_mov_b32 s8, s20 // (op 919) reg expression
s_mov_b32 s9, s21 // (op 919) reg expression
s_mov_b32 s10, s0 // (op 919) reg expression
s_mov_b32 s11, 131072 // (op 919) call()

s_lshl_b32 s0, s16, 4 // (op 936) 
v_mul_lo_u32 v13, v1, s13 // (op 936) most significant: low of low * high
v_mul_hi_u32 v14, v1, s12 // (op 936) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 936) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 936) most significant: sum

s_mov_b32 s7, 0 // (op 936) convert
s_mov_b32 s6, s0 // (op 936) convert

v_mov_b32 v14, s6 // (op 936) 
v_mov_b32 v15, s7 // (op 936) 
v_add_co_u32 v18, s[0:1], v14, v12 // (op 936) least significant half
v_addc_co_u32 v19, s[0:1], v15, v13, s[0:1] // (op 936) most significant half





s_mul_i32 s1, s12, s23 // (op 938) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 938) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 938) least significant
s_mul_hi_u32 s7, s12, s22 // (op 938) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 938) most significant: sum
s_add_u32 s1, s1, s7 // (op 938) most significant: sum


s_mov_b32 s8, s20 // (op 938) reg expression
s_mov_b32 s9, s21 // (op 938) reg expression
s_mov_b32 s10, s0 // (op 938) reg expression
s_mov_b32 s11, 131072 // (op 938) call()

s_lshl_b32 s0, s17, 6 // (op 981) 
s_mov_b32 s7, 0 // (op 981) convert
s_mov_b32 s6, s0 // (op 981) convert

v_mul_lo_u32 v13, v1, s25 // (op 981) most significant: low of low * high
v_mul_hi_u32 v14, v1, s24 // (op 981) most significant: high of low * low
v_mul_lo_u32 v12, v1, s24 // (op 981) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 981) most significant: sum

v_mov_b32 v14, s6 // (op 981) 
v_mov_b32 v15, s7 // (op 981) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 981) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 981) most significant half




v_mul_lo_u32 v15, v13, 6 // (op 981) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 981) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 981) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 981) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 981) 

v_mov_b32 v14, v12 // (op 981) convert

s_mul_i32 s1, s23, 6 // (op 983) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 983) least significant
s_mul_hi_u32 s6, s22, 6 // (op 983) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 983) most significant: sum

s_mul_i32 s7, s24, s1 // (op 983) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 983) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 983) least significant
s_mul_hi_u32 s15, s24, s0 // (op 983) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 983) most significant: sum
s_add_u32 s7, s7, s15 // (op 983) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 983) 

s_mov_b32 s48, s26 // (op 983) reg expression
s_mov_b32 s49, s27 // (op 983) reg expression
s_mov_b32 s50, s0 // (op 983) reg expression
s_mov_b32 s51, 131072 // (op 983) call()

s_lshl_b32 s0, s17, 6 // (op 1000) 
s_mov_b32 s7, 0 // (op 1000) convert
s_mov_b32 s6, s0 // (op 1000) convert

v_mul_lo_u32 v13, v1, s25 // (op 1000) most significant: low of low * high
v_mul_hi_u32 v15, v1, s24 // (op 1000) most significant: high of low * low
v_mul_lo_u32 v12, v1, s24 // (op 1000) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 1000) most significant: sum

v_mov_b32 v15, s6 // (op 1000) 
v_mov_b32 v20, s7 // (op 1000) 
v_add_co_u32 v12, s[0:1], v12, v15 // (op 1000) least significant half
v_addc_co_u32 v13, s[0:1], v13, v20, s[0:1] // (op 1000) most significant half




v_mul_lo_u32 v21, v13, 6 // (op 1000) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 1000) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 1000) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1000) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 1000) 

v_mov_b32 v15, v12 // (op 1000) convert

s_mul_i32 s1, s23, 6 // (op 1002) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1002) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1002) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1002) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1002) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1002) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1002) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1002) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1002) most significant: sum
s_add_u32 s7, s7, s15 // (op 1002) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1002) 

s_mov_b32 s48, s26 // (op 1002) reg expression
s_mov_b32 s49, s27 // (op 1002) reg expression
s_mov_b32 s50, s0 // (op 1002) reg expression
s_mov_b32 s51, 131072 // (op 1002) call()

v_lshrrev_b32 v12, 7, v1 // (op 1045) 
v_and_b32 v13, 15, v1 // (op 1045) 
v_lshl_add_u32 v20, v12, 4, v13 // (op 1045) 


v_lshl_add_u32 v12, s16, 4, v20 // (op 1045) 

v_and_b32 v13, 63, v1 // (op 1045) 
v_lshrrev_b32 v20, 4, v13 // (op 1045) 

v_mul_lo_u32 v13, v20, s28 // (op 1045) 

v_add_u32 v20, v12, v13 // (op 1045) 


s_mul_i32 s1, s28, s33 // (op 1047) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1047) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1047) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1047) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1047) most significant: sum
s_add_u32 s1, s1, s7 // (op 1047) most significant: sum


s_mov_b32 s52, s30 // (op 1047) reg expression
s_mov_b32 s53, s31 // (op 1047) reg expression
s_mov_b32 s54, s0 // (op 1047) reg expression
s_mov_b32 s55, 131072 // (op 1047) call()

v_lshrrev_b32 v12, 7, v1 // (op 1064) 
v_and_b32 v13, 15, v1 // (op 1064) 
v_lshl_add_u32 v21, v12, 4, v13 // (op 1064) 


v_lshl_add_u32 v12, s16, 4, v21 // (op 1064) 

v_and_b32 v13, 63, v1 // (op 1064) 
v_lshrrev_b32 v21, 4, v13 // (op 1064) 

v_mul_lo_u32 v13, v21, s28 // (op 1064) 

v_add_u32 v21, v12, v13 // (op 1064) 


s_mul_i32 s1, s28, s33 // (op 1066) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1066) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1066) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1066) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1066) most significant: sum
s_add_u32 s1, s1, s7 // (op 1066) most significant: sum


s_mov_b32 s52, s30 // (op 1066) reg expression
s_mov_b32 s53, s31 // (op 1066) reg expression
s_mov_b32 s54, s0 // (op 1066) reg expression
s_mov_b32 s55, 131072 // (op 1066) call()

v_and_b32 v12, -64, v1 // (op 1109) 
v_sub_u32 v13, v1, v12 // (op 1109) 

v_lshl_add_u32 v12, s17, 6, v13 // (op 1109) 

v_lshrrev_b32 v13, 6, v1 // (op 1109) 
v_lshlrev_b32 v22, 1, v13 // (op 1109) 

v_mul_lo_u32 v13, v22, s34 // (op 1109) 
v_add_u32 v13, v13, v12 // (op 1109) 


s_mul_i32 s1, s34, s33 // (op 1111) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1111) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1111) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1111) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1111) most significant: sum
s_add_u32 s1, s1, s7 // (op 1111) most significant: sum


s_mov_b32 s56, s36 // (op 1111) reg expression
s_mov_b32 s57, s37 // (op 1111) reg expression
s_mov_b32 s58, s0 // (op 1111) reg expression
s_mov_b32 s59, 131072 // (op 1111) call()

v_and_b32 v12, -64, v1 // (op 1128) 
v_sub_u32 v22, v1, v12 // (op 1128) 

v_lshl_add_u32 v12, s17, 6, v22 // (op 1128) 

v_lshrrev_b32 v22, 6, v1 // (op 1128) 
v_lshlrev_b32 v23, 1, v22 // (op 1128) 

v_mul_lo_u32 v22, v23, s34 // (op 1128) 
v_add_u32 v22, v22, v12 // (op 1128) 


s_mul_i32 s1, s34, s33 // (op 1130) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1130) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1130) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1130) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1130) most significant: sum
s_add_u32 s1, s1, s7 // (op 1130) most significant: sum


s_mov_b32 s56, s36 // (op 1130) reg expression
s_mov_b32 s57, s37 // (op 1130) reg expression
s_mov_b32 s58, s0 // (op 1130) reg expression
s_mov_b32 s59, 131072 // (op 1130) call()


v_mov_b64 v[24:25], s[28:29] // (op 761) convert

v_mov_b32 v12, v20 // (op 761) reg expression
v_mov_b32 v23, v12 // (op 761) 
buffer_load_ubyte v26, v23, s[52:55], 0 offen offset: 0 // (op 761) Load value





v_mov_b32 v12, v14 // (op 765) reg expression

buffer_load_dwordx3 v[28:30], v12, s[48:51], 0 offen // (op 765) Load value
buffer_load_dwordx3 v[32:34], v12, s[48:51], 0 offen offset:12 // (op 765) Load value
buffer_load_dwordx3 v[36:38], v12, s[48:51], 0 offen offset:24 // (op 765) Load value
buffer_load_dwordx3 v[40:42], v12, s[48:51], 0 offen offset:36 // (op 765) Load value


v_mov_b64 v[24:25], s[34:35] // (op 768) convert

v_mov_b32 v12, v13 // (op 768) reg expression
buffer_load_ubyte v44, v12, s[56:59], 0 offen offset: 0 // (op 768) Load value
v_add_u32 v12, v12, v24 // (op 768) 
buffer_load_ubyte v45, v12, s[56:59], 0 offen offset: 0 // (op 768) Load value


v_lshlrev_b32 v12, 4, v1 // (op 1194) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1194) Wait state hazard: VALU Write Readlane Hazard



v_mov_b64 v[24:25], v[16:17] // (op 872) reg expression

s_nop 0
s_mov_b32 m0, s0 // (op 872) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 872) Load value direct to lds



v_mov_b64 v[24:25], s[28:29] // (op 775) convert

s_lshl_b32 s0, s28, 2 // (op 775) 
v_add_u32 v12, v20, s0 // (op 775) 

v_mov_b32 v23, v12 // (op 775) 
buffer_load_ubyte v27, v23, s[52:55], 0 offen offset: 0 // (op 775) Load value






s_lshl_b64 s[0:1], s[24:25], 7 // (op 779) 
s_mul_i32 s7, s1, 6 // (op 779) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 779) least significant
s_mul_hi_u32 s14, s0, 6 // (op 779) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 779) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 779) 

v_add_u32 v12, v14, s0 // (op 779) 


buffer_load_dwordx3 v[46:48], v12, s[48:51], 0 offen // (op 779) Load value
buffer_load_dwordx3 v[50:52], v12, s[48:51], 0 offen offset:12 // (op 779) Load value
buffer_load_dwordx3 v[54:56], v12, s[48:51], 0 offen offset:24 // (op 779) Load value
buffer_load_dwordx3 v[58:60], v12, s[48:51], 0 offen offset:36 // (op 779) Load value



v_mov_b64 v[24:25], s[34:35] // (op 782) convert

s_lshl_b32 s0, s34, 2 // (op 782) 
v_add_u32 v12, v13, s0 // (op 782) 

buffer_load_ubyte v62, v12, s[56:59], 0 offen offset: 0 // (op 782) Load value
v_add_u32 v12, v12, v24 // (op 782) 
buffer_load_ubyte v63, v12, s[56:59], 0 offen offset: 0 // (op 782) Load value



v_lshlrev_b32 v20, 4, v1 // (op 1367) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1367) Wait state hazard: VALU Write Readlane Hazard



s_lshl_b64 s[6:7], s[12:13], 7 // (op 877) 
v_mov_b32 v20, s6 // (op 877) 
v_mov_b32 v31, s7 // (op 877) 
v_add_co_u32 v24, s[14:15], v16, v20 // (op 877) least significant half
v_addc_co_u32 v25, s[14:15], v17, v31, s[14:15] // (op 877) most significant half




s_add_u32 s1, 2048, s0 // (op 877) 

s_mov_b32 m0, s1 // (op 877) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 877) Load value direct to lds




v_lshlrev_b32 v20, 6, v1 // (op 1247) 
v_mul_lo_u32 v31, v20, 6 // (op 1247) 
v_lshrrev_b32 v35, 3, v31 // (op 1247) 

v_lshrrev_b32 v31, 4, v20 // (op 1247) 

v_lshlrev_b32 v20, 2, v31 // (op 1247) 

v_add_u32 v31, v35, v20 // (op 1247) 





s_waitcnt vmcnt(14)
ds_write_b96 v31, v[28:30] offset:4096 // (op 787) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v31, v[32:34] offset:4112 // (op 787) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v31, v[36:38] offset:4128 // (op 787) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v31, v[40:42] offset:4144 // (op 787) Store local data 

v_and_b32 v20, -64, v1 // (op 1305) 
v_sub_u32 v31, v1, v20 // (op 1305) 

v_lshrrev_b32 v20, 6, v1 // (op 1305) 
v_lshl_add_u32 v35, v20, 7, v31 // (op 1305) 





s_waitcnt vmcnt(10)
ds_write_b8 v35, v44 offset:12288 // (op 788) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v35, v45 offset:12352 // (op 788) Store local data 

v_lshlrev_b32 v20, 4, v1 // (op 1166) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1166) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 6, v1 // (op 1218) 
v_mul_lo_u32 v31, v20, 6 // (op 1218) 
v_lshrrev_b32 v35, 3, v31 // (op 1218) 

v_lshrrev_b32 v31, 4, v20 // (op 1218) 

v_lshlrev_b32 v20, 2, v31 // (op 1218) 

v_add_u32 v31, v35, v20 // (op 1218) 


v_and_b32 v20, -64, v1 // (op 1276) 
v_sub_u32 v35, v1, v20 // (op 1276) 

v_lshrrev_b32 v20, 6, v1 // (op 1276) 
v_lshl_add_u32 v39, v20, 7, v35 // (op 1276) 


v_lshlrev_b32 v20, 4, v1 // (op 1352) 
s_nop 0
v_readfirstlane_b32 s1, v20 // (op 1352) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 6, v1 // (op 1391) 
v_mul_lo_u32 v35, v20, 6 // (op 1391) 
v_lshrrev_b32 v43, 3, v35 // (op 1391) 

v_lshrrev_b32 v35, 4, v20 // (op 1391) 

v_lshlrev_b32 v20, 2, v35 // (op 1391) 

v_add_u32 v35, v43, v20 // (op 1391) 


v_and_b32 v20, -64, v1 // (op 1420) 
v_sub_u32 v43, v1, v20 // (op 1420) 

v_lshrrev_b32 v20, 6, v1 // (op 1420) 
v_lshl_add_u32 v49, v20, 7, v43 // (op 1420) 


s_mov_b32 s6, 0 // (op 308) call()
s_ashr_i64 s[14:15], s[22:23], 7 // (op 307) 
s_mov_b32 s7, s14 // (op 307) convert

s_and_b32 s14, s7, 4294967294 // (op 307) 

s_cmp_lt_u32 s6, s14 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_307 // (op 307) Condition: Top (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_307 if false)
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_307:
s_add_u32 s6, s6, 2 // (op 309) 
s_waitcnt vmcnt(8)
s_barrier  // (op 1478) 



ds_read_b64_tr_b8 v[64:65], v0 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v0 offset:128 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v0 offset:1024 // (op 65) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v0 offset:1152 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1491) 



ds_read_b96_tr_b6 v[72:74], v4 offset:4096 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[75:77], v4 offset:5120 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1505) 


ds_read_u8 v20, v5 offset:12288 // (op 79) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:71], v[72:77], a[0:3], v26, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 82) 
ds_read_b96_tr_b6 v[78:80], v4 offset:4128 // (op 339) Transpose load local data 


ds_read_b96_tr_b6 v[81:83], v4 offset:5152 // (op 339) Transpose load local data 


ds_read_u8 v20, v5 offset:12320 // (op 342) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:71], v[78:83], a[4:7], v26, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 343) 




v_mov_b64 v[16:17], s[28:29] // (op 75) convert

v_mov_b32 v20, v21 // (op 75) 
s_lshl_b32 s7, s28, 3 // (op 75) 
v_add_u32 v43, v20, s7 // (op 75) 

v_mov_b32 v53, v43 // (op 75) 
buffer_load_ubyte v26, v53, s[52:55], 0 offen offset: 0 // (op 75) Load value





v_mov_b32 v43, v15 // (op 10) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s7, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s7, s14, 1 // (op 10) 

v_add_u32 v53, v43, s7 // (op 10) 


buffer_load_dwordx3 v[28:30], v53, s[48:51], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[32:34], v53, s[48:51], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[36:38], v53, s[48:51], 0 offen offset:24 // (op 10) Load value
buffer_load_dwordx3 v[40:42], v53, s[48:51], 0 offen offset:36 // (op 10) Load value


v_mov_b64 v[16:17], s[34:35] // (op 21) convert

v_mov_b32 v53, v22 // (op 21) 
s_lshl_b32 s7, s34, 3 // (op 21) 
v_add_u32 v57, v53, s7 // (op 21) 

buffer_load_ubyte v44, v57, s[56:59], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v57, v57, v16 // (op 21) 
buffer_load_ubyte v45, v57, s[56:59], 0 offen offset: 0 // (op 21) Load value


s_barrier  // (op 1481) 


v_mov_b64 v[16:17], v[18:19] // (op 856) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 856) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 856) 

v_mov_b32 v57, s18 // (op 856) 
v_mov_b32 v61, s19 // (op 856) 
v_add_co_u32 v24, s[14:15], v16, v57 // (op 856) least significant half
v_addc_co_u32 v25, s[14:15], v17, v61, s[14:15] // (op 856) most significant half





s_mov_b32 m0, s0 // (op 856) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 856) Load value direct to lds

s_barrier  // (op 1528) 



s_waitcnt vmcnt(14)
ds_write_b96 v35, v[46:48] offset:12544 // (op 235) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v35, v[50:52] offset:12560 // (op 235) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v35, v[54:56] offset:12576 // (op 235) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v35, v[58:60] offset:12592 // (op 235) Store local data 
s_barrier  // (op 1534) 



s_waitcnt vmcnt(10)
ds_write_b8 v49, v62 offset:20736 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v49, v63 offset:20800 // (op 236) Store local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 1519) 



ds_read_b64_tr_b8 v[64:65], v3 offset:2048 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v3 offset:2176 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[68:69], v3 offset:3072 // (op 227) Transpose load local data 
ds_read_b64_tr_b8 v[70:71], v3 offset:3200 // (op 227) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1525) 



ds_read_b96_tr_b6 v[72:74], v7 offset:12544 // (op 229) Transpose load local data 
ds_read_b96_tr_b6 v[75:77], v7 offset:13568 // (op 229) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1531) 


ds_read_u8 v57, v8 offset:20736 // (op 233) Load local data 



ds_read_b96_tr_b6 v[78:80], v7 offset:12576 // (op 348) Transpose load local data 
ds_read_b96_tr_b6 v[81:83], v7 offset:13600 // (op 348) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:71], v[72:77], a[0:3], v27, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 234) 




s_barrier  // (op 1595) 
ds_read_u8 v57, v8 offset:20768 // (op 351) Load local data 
s_barrier  // (op 1599) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:71], v[78:83], a[4:7], v27, v57 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 344) 




v_mov_b64 v[24:25], s[28:29] // (op 231) convert

s_lshl_b32 s7, s28, 2 // (op 231) 
s_mul_i32 s14, 3, s7 // (op 231) 

v_add_u32 v23, v20, s14 // (op 231) 

v_mov_b32 v57, v23 // (op 231) 
buffer_load_ubyte v27, v57, s[52:55], 0 offen offset: 0 // (op 231) Load value






s_lshl_b32 s7, s28, 3 // (op 1056) 
v_add_u32 v21, v21, s7 // (op 1056) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 237) 
s_mul_i32 s19, s15, 6 // (op 237) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 237) least significant
s_mul_hi_u32 s7, s14, 6 // (op 237) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 237) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 237) 

s_mul_i32 s7, 3, s14 // (op 237) 

v_add_u32 v20, v43, s7 // (op 237) 


buffer_load_dwordx3 v[46:48], v20, s[48:51], 0 offen // (op 237) Load value
buffer_load_dwordx3 v[50:52], v20, s[48:51], 0 offen offset:12 // (op 237) Load value
buffer_load_dwordx3 v[54:56], v20, s[48:51], 0 offen offset:24 // (op 237) Load value
buffer_load_dwordx3 v[58:60], v20, s[48:51], 0 offen offset:36 // (op 237) Load value



v_mov_b64 v[24:25], s[34:35] // (op 238) convert

s_lshl_b32 s7, s34, 2 // (op 238) 
s_mul_i32 s14, 3, s7 // (op 238) 

v_add_u32 v20, v53, s14 // (op 238) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 992) 
s_mul_i32 s19, s15, 6 // (op 992) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 992) least significant
s_mul_hi_u32 s7, s14, 6 // (op 992) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 992) most significant: sum


buffer_load_ubyte v62, v20, s[56:59], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v20, v20, v24 // (op 238) 
s_lshr_b64 s[14:15], s[18:19], 3 // (op 992) 

s_lshl_b32 s7, s14, 1 // (op 992) 

v_add_u32 v15, v15, s7 // (op 992) 

buffer_load_ubyte v63, v20, s[56:59], 0 offen offset: 0 // (op 238) Load value



s_barrier  // (op 1522) 
s_lshl_b32 s7, s34, 3 // (op 1120) 


v_add_u32 v22, v22, s7 // (op 1120) 

s_lshl_b64 s[14:15], s[12:13], 7 // (op 867) 
s_mul_i32 s19, 3, s15 // (op 867) most significant: low of low * high
s_mul_i32 s18, 3, s14 // (op 867) least significant
s_mul_hi_u32 s7, 3, s14 // (op 867) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 867) most significant: sum


v_mov_b32 v20, s18 // (op 867) 
v_mov_b32 v43, s19 // (op 867) 
v_add_co_u32 v24, s[14:15], v16, v20 // (op 867) least significant half
v_addc_co_u32 v25, s[14:15], v17, v43, s[14:15] // (op 867) most significant half




s_add_u32 s7, 2048, s1 // (op 867) 

s_mov_b32 m0, s7 // (op 867) reg expression
buffer_load_dwordx4 v24, s[8:11], 0 offen offset: 0 lds // (op 867) Load value direct to lds



s_barrier  // (op 1494) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 928) 



s_lshl_b64 s[18:19], s[14:15], 1 // (op 928) 

v_mov_b32 v20, s18 // (op 928) 
v_mov_b32 v43, s19 // (op 928) 
v_add_co_u32 v18, s[14:15], v18, v20 // (op 928) least significant half
v_addc_co_u32 v19, s[14:15], v19, v43, s[14:15] // (op 928) most significant half




s_waitcnt vmcnt(14)
ds_write_b96 v31, v[28:30] offset:4096 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v31, v[32:34] offset:4112 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b96 v31, v[36:38] offset:4128 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b96 v31, v[40:42] offset:4144 // (op 13) Store local data 
s_barrier  // (op 1508) 



s_waitcnt vmcnt(10)
ds_write_b8 v39, v44 offset:12288 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v39, v45 offset:12352 // (op 26) Store local data 
s_ashr_i64 s[14:15], s[22:23], 7 // (op 307) 
s_mov_b32 s7, s14 // (op 307) convert

s_and_b32 s14, s7, 4294967294 // (op 307) 

s_cmp_lt_u32 s6, s14 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_307 // (op 307) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_307 if true)
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_307:














s_barrier  // (op 1579) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 890) 
s_lshr_b32 s6, s0, 1 // (op 890) 

v_lshl_add_u32 v49, s6, 8, v1 // (op 890) 

s_lshl_b32 s0, s16, 4 // (op 890) 
v_mul_lo_u32 v13, v49, s13 // (op 890) most significant: low of low * high
v_mul_hi_u32 v53, v49, s12 // (op 890) most significant: high of low * low
v_mul_lo_u32 v12, v49, s12 // (op 890) least significant: low of low * low
v_add_u32 v13, v13, v53 // (op 890) most significant: sum


s_mov_b32 s7, 0 // (op 890) convert
s_mov_b32 s6, s0 // (op 890) convert

v_mov_b32 v49, s6 // (op 890) 
v_mov_b32 v53, s7 // (op 890) 
v_add_co_u32 v14, s[0:1], v49, v12 // (op 890) least significant half
v_addc_co_u32 v15, s[0:1], v53, v13, s[0:1] // (op 890) most significant half





s_mul_i32 s1, s12, s23 // (op 892) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 892) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 892) least significant
s_mul_hi_u32 s7, s12, s22 // (op 892) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 892) most significant: sum
s_add_u32 s1, s1, s7 // (op 892) most significant: sum


s_mov_b32 s8, s20 // (op 892) reg expression
s_mov_b32 s9, s21 // (op 892) reg expression
s_mov_b32 s10, s0 // (op 892) reg expression
s_mov_b32 s11, 131072 // (op 892) call()

s_ashr_i64 s[0:1], s[22:23], 7 // (op 958) 
s_lshr_b32 s6, s0, 1 // (op 958) 

v_lshl_add_u32 v49, s6, 8, v1 // (op 958) 

s_lshl_b32 s0, s17, 6 // (op 958) 
s_mov_b32 s7, 0 // (op 958) convert
s_mov_b32 s6, s0 // (op 958) convert

v_mul_lo_u32 v13, v49, s25 // (op 958) most significant: low of low * high
v_mul_hi_u32 v53, v49, s24 // (op 958) most significant: high of low * low
v_mul_lo_u32 v12, v49, s24 // (op 958) least significant: low of low * low
v_add_u32 v13, v13, v53 // (op 958) most significant: sum

v_mov_b32 v53, s6 // (op 958) 
v_mov_b32 v57, s7 // (op 958) 
v_add_co_u32 v12, s[0:1], v12, v53 // (op 958) least significant half
v_addc_co_u32 v13, s[0:1], v13, v57, s[0:1] // (op 958) most significant half





v_mul_lo_u32 v17, v13, 6 // (op 958) most significant: low of high * low
v_mul_hi_u32 v49, v12, 6 // (op 958) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 958) least significant: low of low * low
v_add_u32 v17, v17, v49 // (op 958) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 958) 

v_mov_b32 v49, v12 // (op 958) convert

s_mul_i32 s1, s23, 6 // (op 960) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 960) least significant
s_mul_hi_u32 s6, s22, 6 // (op 960) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 960) most significant: sum

s_mul_i32 s7, s24, s1 // (op 960) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 960) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 960) least significant
s_mul_hi_u32 s15, s24, s0 // (op 960) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 960) most significant: sum
s_add_u32 s7, s7, s15 // (op 960) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 960) 

s_mov_b32 s48, s26 // (op 960) reg expression
s_mov_b32 s49, s27 // (op 960) reg expression
s_mov_b32 s50, s0 // (op 960) reg expression
s_mov_b32 s51, 131072 // (op 960) call()

v_lshrrev_b32 v53, 7, v1 // (op 1022) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1022) 
v_and_b32 v57, 15, v1 // (op 1022) 
s_lshr_b32 s6, s0, 1 // (op 1022) 

v_lshl_add_u32 v61, v53, 4, v57 // (op 1022) 


v_lshl_add_u32 v53, s16, 4, v61 // (op 1022) 

v_and_b32 v57, 63, v1 // (op 1022) 
v_lshrrev_b32 v61, 4, v57 // (op 1022) 

v_lshl_add_u32 v57, s6, 3, v61 // (op 1022) 


v_mul_lo_u32 v61, v57, s28 // (op 1022) 

v_add_u32 v57, v53, v61 // (op 1022) 


s_mul_i32 s1, s28, s33 // (op 1024) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1024) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1024) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1024) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1024) most significant: sum
s_add_u32 s1, s1, s7 // (op 1024) most significant: sum


s_mov_b32 s52, s30 // (op 1024) reg expression
s_mov_b32 s53, s31 // (op 1024) reg expression
s_mov_b32 s54, s0 // (op 1024) reg expression
s_mov_b32 s55, 131072 // (op 1024) call()

v_lshrrev_b32 v53, 6, v1 // (op 1086) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1086) 
v_lshlrev_b32 v61, 1, v53 // (op 1086) 

s_lshr_b32 s6, s0, 1 // (op 1086) 

v_and_b32 v53, -64, v1 // (op 1086) 
v_lshl_add_u32 v12, s6, 3, v61 // (op 1086) 


v_sub_u32 v13, v1, v53 // (op 1086) 

v_lshl_add_u32 v53, s17, 6, v13 // (op 1086) 

v_mul_lo_u32 v13, v12, s34 // (op 1086) 
v_add_u32 v13, v13, v53 // (op 1086) 


s_mul_i32 s1, s34, s33 // (op 1088) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1088) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1088) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1088) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1088) most significant: sum
s_add_u32 s1, s1, s7 // (op 1088) most significant: sum


s_mov_b32 s56, s36 // (op 1088) reg expression
s_mov_b32 s57, s37 // (op 1088) reg expression
s_mov_b32 s58, s0 // (op 1088) reg expression
s_mov_b32 s59, 131072 // (op 1088) call()

v_lshlrev_b32 v12, 4, v1 // (op 1179) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1179) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v12, 6, v1 // (op 1231) 
v_mul_lo_u32 v53, v12, 6 // (op 1231) 
v_lshrrev_b32 v61, 3, v53 // (op 1231) 

v_lshrrev_b32 v53, 4, v12 // (op 1231) 

v_lshlrev_b32 v12, 2, v53 // (op 1231) 

v_add_u32 v53, v61, v12 // (op 1231) 


v_and_b32 v12, -64, v1 // (op 1289) 
v_sub_u32 v61, v1, v12 // (op 1289) 

v_lshrrev_b32 v12, 6, v1 // (op 1289) 
v_lshl_add_u32 v16, v12, 7, v61 // (op 1289) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 313) 
s_and_b32 s1, s6, 4294967294 // (op 313) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 312) 
s_mov_b32 s14, s6 // (op 312) convert

s_cmp_lt_u32 s1, s14 // (op 312) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_312 // (op 312) Condition: Top (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_312 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_312:
s_barrier  // (op 1484) 
s_add_u32 s1, s1, 1 // (op 314) 


v_mov_b64 v[50:51], v[14:15] // (op 861) 
v_mov_b64 v[54:55], v[50:51] // (op 861) reg expression

s_mov_b32 m0, s0 // (op 861) reg expression
buffer_load_dwordx4 v54, s[8:11], 0 offen offset: 0 lds // (op 861) Load value direct to lds


s_lshl_b64 s[6:7], s[12:13], 7 // (op 880) 
v_mov_b32 v12, s6 // (op 880) 
v_mov_b32 v17, s7 // (op 880) 
v_add_co_u32 v14, s[14:15], v14, v12 // (op 880) least significant half
v_addc_co_u32 v15, s[14:15], v15, v17, s[14:15] // (op 880) most significant half






v_mov_b32 v12, v49 // (op 179) 
v_mov_b32 v17, v12 // (op 179) reg expression

buffer_load_dwordx3 v[18:20], v17, s[48:51], 0 offen // (op 179) Load value
buffer_load_dwordx3 v[22:24], v17, s[48:51], 0 offen offset:12 // (op 179) Load value
buffer_load_dwordx3 v[26:28], v17, s[48:51], 0 offen offset:24 // (op 179) Load value
buffer_load_dwordx3 v[30:32], v17, s[48:51], 0 offen offset:36 // (op 179) Load value


s_lshl_b64 s[6:7], s[24:25], 7 // (op 950) 
s_mul_i32 s15, s7, 6 // (op 950) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 950) least significant
s_mul_hi_u32 s18, s6, 6 // (op 950) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 950) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 950) 

v_add_u32 v49, v49, s6 // (op 950) 


s_barrier  // (op 1497) 
v_mov_b64 v[50:51], s[34:35] // (op 180) convert




v_mov_b32 v12, v13 // (op 180) 
v_mov_b32 v17, v12 // (op 180) reg expression
buffer_load_ubyte v54, v17, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v17, v17, v50 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b96 v53, v[18:20] offset:4096 // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v53, v[22:24] offset:4112 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v53, v[26:28] offset:4128 // (op 177) Store local data 
buffer_load_ubyte v55, v17, s[56:59], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s6, s34, 2 // (op 1078) 
v_add_u32 v13, v13, s6 // (op 1078) 

s_waitcnt vmcnt(2)
ds_write_b96 v53, v[30:32] offset:4144 // (op 177) Store local data 

s_barrier  // (op 1487) 



s_barrier  // (op 1511) 
ds_read_b64_tr_b8 v[18:19], v0 // (op 169) Transpose load local data 



ds_read_b64_tr_b8 v[20:21], v0 offset:128 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:1024 // (op 169) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:1152 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v54 offset:12288 // (op 178) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1501) 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v55 offset:12352 // (op 178) Store local data 




ds_read_b96_tr_b6 v[26:28], v4 offset:4096 // (op 171) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v4 offset:5120 // (op 171) Transpose load local data 

v_mov_b64 v[32:33], s[28:29] // (op 173) convert

v_mov_b32 v12, v57 // (op 173) 
v_mov_b32 v17, v12 // (op 173) reg expression
v_mov_b32 v34, v17 // (op 173) 
buffer_load_ubyte v35, v34, s[52:55], 0 offen offset: 0 // (op 173) Load value




s_lshl_b32 s6, s28, 2 // (op 1014) 
v_add_u32 v57, v57, s6 // (op 1014) 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 1515) 


ds_read_u8 v12, v5 offset:12288 // (op 175) Load local data 



ds_read_b96_tr_b6 v[36:38], v4 offset:4128 // (op 320) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v4 offset:5152 // (op 320) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[26:31], a[0:3], v35, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 176) 




ds_read_u8 v12, v5 offset:12320 // (op 323) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:25], v[36:41], a[4:7], v35, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 324) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 312) 
s_mov_b32 s14, s6 // (op 312) convert

s_cmp_lt_u32 s1, s14 // (op 312) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_312 // (op 312) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_312 if true)
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_312:






















s_barrier  // (op 1583) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123
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


v_accvgpr_read v12, a4 // (op 504) 
v_accvgpr_read v13, a5 // (op 504) 
v_accvgpr_read v14, a6 // (op 504) 
v_accvgpr_read v15, a7 // (op 504) 
v_mul_f32 v16, v11, v12 // (op 504) 
v_mul_f32 v17, v11, v13 // (op 504) 
v_mul_f32 v18, v11, v14 // (op 504) 
v_mul_f32 v19, v11, v15 // (op 504) 



s_lshl_b32 s0, s4, 7 // (op 506) 
v_add_u32 v0, v9, s0 // (op 506) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 506) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123:


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




s_lshl_b32 s0, s2, 7 // (op 528) 
v_add_u32 v0, v6, s0 // (op 528) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 528) Load value

v_accvgpr_read v16, a4 // (op 525) 
v_accvgpr_read v17, a5 // (op 525) 
v_accvgpr_read v18, a6 // (op 525) 
v_accvgpr_read v19, a7 // (op 525) 
v_mul_f32 v20, v11, v16 // (op 525) 
v_mul_f32 v21, v11, v17 // (op 525) 
v_mul_f32 v22, v11, v18 // (op 525) 
v_mul_f32 v23, v11, v19 // (op 525) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 529) 
v_mul_f32 v17, v10, v13 // (op 529) 
v_mul_f32 v18, v10, v14 // (op 529) 
v_mul_f32 v19, v10, v15 // (op 529) 

v_add_f32 v12, v16, v20 // (op 526) 
v_add_f32 v13, v17, v21 // (op 526) 
v_add_f32 v14, v18, v22 // (op 526) 
v_add_f32 v15, v19, v23 // (op 526) 




s_lshl_b32 s0, s4, 7 // (op 527) 
v_add_u32 v0, v9, s0 // (op 527) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 527) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.LRR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end:
.size RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2, .LRR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end-RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
  .amdhsa_next_free_vgpr 92
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
  .amdhsa_accum_offset 84
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
    DataTypeB: F6
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
    macrotile: [16, 64, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
    .symbol: RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 84
    .agpr_count: 8
    .max_flat_workgroup_size: 128
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

