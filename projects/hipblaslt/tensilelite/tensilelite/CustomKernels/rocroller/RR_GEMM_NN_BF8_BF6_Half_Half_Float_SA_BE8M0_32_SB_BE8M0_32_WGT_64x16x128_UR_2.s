// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2,@function
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_exec_begin:


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


v_and_b32 v0, 15, v1 // (op 1281) 
v_and_b32 v3, 63, v1 // (op 1281) 
v_lshrrev_b32 v4, 4, v3 // (op 1281) 

v_lshlrev_b32 v3, 5, v4 // (op 1281) 

v_lshl_add_u32 v4, v0, 7, v3 // (op 1281) 


v_mul_lo_u32 v0, v4, 6 // (op 1281) 

v_lshrrev_b32 v3, 3, v0 // (op 1281) 

v_and_b32 v0, 15, v1 // (op 1425) 
v_and_b32 v4, 63, v1 // (op 1425) 
v_lshrrev_b32 v5, 4, v4 // (op 1425) 

v_lshlrev_b32 v4, 5, v5 // (op 1425) 

v_lshl_add_u32 v5, v0, 7, v4 // (op 1425) 


v_mul_lo_u32 v0, v5, 6 // (op 1425) 

v_lshrrev_b32 v4, 3, v0 // (op 1425) 

v_lshrrev_b32 v0, 6, v1 // (op 1167) 
v_and_b32 v5, 63, v1 // (op 1167) 
v_lshrrev_b32 v6, 4, v5 // (op 1167) 
v_lshlrev_b32 v7, 2, v6 // (op 1167) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1167) 


v_lshl_add_u32 v0, s16, 6, v6 // (op 1167) 

v_lshrrev_b32 v6, 2, v5 // (op 1167) 

v_and_b32 v5, 3, v6 // (op 1167) 

v_and_b32 v6, 3, v1 // (op 1167) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1167) 


v_lshl_add_u32 v5, s17, 4, v7 // (op 1167) 

v_mul_lo_u32 v6, v5, s2 // (op 1167) 

v_add_lshl_u32 v5, v0, v6, 1 // (op 1167) 


v_and_b32 v0, 48, v1 // (op 1228) 
v_and_b32 v6, 63, v1 // (op 1228) 
v_sub_u32 v7, v6, v0 // (op 1228) 

v_lshrrev_b32 v6, 1, v7 // (op 1228) 
s_mul_i32 s1, s2, s9 // (op 1169) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1169) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1169) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1169) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1169) most significant: sum
s_add_u32 s1, s1, s19 // (op 1169) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1169) 

v_add_lshl_u32 v8, v0, v6, 6 // (op 1228) 


v_and_b32 v0, 1, v7 // (op 1228) 

v_lshlrev_b32 v6, 3, v0 // (op 1228) 

v_lshrrev_b32 v0, 6, v1 // (op 1228) 
v_lshl_add_u32 v7, v0, 4, v6 // (op 1228) 


v_add_u32 v0, v8, v7 // (op 1228) 


v_lshrrev_b32 v6, 6, v1 // (op 1339) 
v_and_b32 v7, 15, v1 // (op 1339) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 1339) 


v_and_b32 v6, 63, v1 // (op 1339) 
v_lshrrev_b32 v7, 4, v6 // (op 1339) 

v_lshl_add_u32 v6, v7, 6, v8 // (op 1339) 


v_lshrrev_b32 v7, 6, v1 // (op 1355) 
v_and_b32 v8, 63, v1 // (op 1355) 
v_lshrrev_b32 v9, 4, v8 // (op 1355) 
v_lshlrev_b32 v10, 2, v9 // (op 1355) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1355) 


v_lshl_add_u32 v7, s16, 6, v9 // (op 1355) 

v_lshrrev_b32 v9, 2, v8 // (op 1355) 

v_and_b32 v8, 3, v9 // (op 1355) 

v_and_b32 v9, 3, v1 // (op 1355) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1355) 


v_lshl_add_u32 v8, s17, 4, v10 // (op 1355) 

v_mul_lo_u32 v9, v8, s4 // (op 1355) 

v_add_lshl_u32 v8, v7, v9, 1 // (op 1355) 


v_and_b32 v7, 48, v1 // (op 1401) 
v_and_b32 v9, 63, v1 // (op 1401) 
v_sub_u32 v10, v9, v7 // (op 1401) 

v_lshrrev_b32 v9, 1, v10 // (op 1401) 
s_mul_i32 s1, s4, s9 // (op 1357) most significant: low of low * high
v_add_lshl_u32 v11, v7, v9, 6 // (op 1401) 


v_and_b32 v7, 1, v10 // (op 1401) 

v_lshlrev_b32 v9, 3, v7 // (op 1401) 

v_lshrrev_b32 v7, 6, v1 // (op 1401) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1401) 


v_add_u32 v7, v11, v10 // (op 1401) 


v_lshrrev_b32 v9, 6, v1 // (op 1454) 
v_and_b32 v10, 15, v1 // (op 1454) 
v_lshl_add_u32 v11, v9, 4, v10 // (op 1454) 


v_and_b32 v9, 63, v1 // (op 1454) 
v_lshrrev_b32 v10, 4, v9 // (op 1454) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 1454) 


s_mul_i32 s38, s5, s8 // (op 1357) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1357) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1357) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1357) most significant: sum
s_add_u32 s1, s1, s39 // (op 1357) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 1 // (op 1357) 

s_mov_b32 s40, s6 // (op 1169) reg expression
s_mov_b32 s41, s7 // (op 1169) reg expression
s_mov_b32 s42, s18 // (op 1169) reg expression
s_mov_b32 s43, 131072 // (op 1169) call()

s_mov_b32 s44, s10 // (op 1357) reg expression

s_mov_b32 s45, s11 // (op 1357) reg expression
s_mov_b32 s46, s38 // (op 1357) reg expression
s_mov_b32 s47, 131072 // (op 1357) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
v_and_b32 v12, -4, v1 // (op 934) 
v_sub_u32 v13, v1, v12 // (op 934) 

v_lshlrev_b32 v12, 4, v13 // (op 934) 

v_lshl_add_u32 v13, s16, 6, v12 // (op 934) 

v_mov_b32 v15, 0 // (op 934) convert
v_mov_b32 v14, v13 // (op 934) convert

v_lshrrev_b32 v12, 2, v1 // (op 934) 
v_mul_lo_u32 v17, v12, s13 // (op 934) most significant: low of low * high
v_mul_hi_u32 v13, v12, s12 // (op 934) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 934) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 934) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 934) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 934) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 936) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 936) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 936) least significant
s_mul_hi_u32 s7, s12, s22 // (op 936) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 936) most significant: sum
s_add_u32 s1, s1, s7 // (op 936) most significant: sum


s_mov_b32 s48, s20 // (op 936) reg expression
s_mov_b32 s49, s21 // (op 936) reg expression
s_mov_b32 s50, s0 // (op 936) reg expression
s_mov_b32 s51, 131072 // (op 936) call()

v_and_b32 v14, -4, v1 // (op 953) 
v_sub_u32 v15, v1, v14 // (op 953) 

v_lshlrev_b32 v14, 4, v15 // (op 953) 

v_lshl_add_u32 v15, s16, 6, v14 // (op 953) 

v_mov_b32 v17, 0 // (op 953) convert
v_mov_b32 v16, v15 // (op 953) convert

v_lshrrev_b32 v14, 2, v1 // (op 953) 
v_mul_lo_u32 v19, v14, s13 // (op 953) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 953) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 953) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 953) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 953) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 953) most significant half



s_mul_i32 s1, s12, s23 // (op 955) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 955) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 955) least significant
s_mul_hi_u32 s7, s12, s22 // (op 955) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 955) most significant: sum
s_add_u32 s1, s1, s7 // (op 955) most significant: sum


s_mov_b32 s48, s20 // (op 955) reg expression
s_mov_b32 s49, s21 // (op 955) reg expression
s_mov_b32 s50, s0 // (op 955) reg expression
s_mov_b32 s51, 131072 // (op 955) call()

v_and_b32 v16, -8, v1 // (op 998) 
v_sub_u32 v17, v1, v16 // (op 998) 

v_lshlrev_b32 v16, 4, v17 // (op 998) 

v_mov_b32 v19, 0 // (op 998) convert
v_mov_b32 v18, v16 // (op 998) convert

v_lshrrev_b32 v16, 3, v1 // (op 998) 
v_lshl_add_u32 v17, s17, 4, v16 // (op 998) 

v_mul_lo_u32 v21, v17, s25 // (op 998) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 998) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 998) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 998) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 998) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 998) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 998) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 998) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 998) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 998) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 998) 

v_mov_b32 v18, v16 // (op 998) convert

s_mul_i32 s1, s9, 6 // (op 1000) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1000) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1000) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1000) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1000) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1000) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1000) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1000) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1000) most significant: sum
s_add_u32 s7, s7, s11 // (op 1000) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1000) 

s_mov_b32 s52, s26 // (op 1000) reg expression
s_mov_b32 s53, s27 // (op 1000) reg expression
s_mov_b32 s54, s0 // (op 1000) reg expression
s_mov_b32 s55, 131072 // (op 1000) call()

v_and_b32 v16, -8, v1 // (op 1017) 
v_sub_u32 v17, v1, v16 // (op 1017) 

v_lshlrev_b32 v16, 4, v17 // (op 1017) 

v_mov_b32 v21, 0 // (op 1017) convert
v_mov_b32 v20, v16 // (op 1017) convert

v_lshrrev_b32 v19, 3, v1 // (op 1017) 
v_lshl_add_u32 v16, s17, 4, v19 // (op 1017) 

v_mul_lo_u32 v23, v16, s25 // (op 1017) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 1017) most significant: high of low * low
v_mul_lo_u32 v22, v16, s24 // (op 1017) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 1017) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 1017) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 1017) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 1017) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 1017) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 1017) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 1017) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 1017) 

v_mov_b32 v19, v16 // (op 1017) convert

s_mul_i32 s1, s9, 6 // (op 1019) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 1019) least significant
s_mul_hi_u32 s6, s8, 6 // (op 1019) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1019) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1019) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 1019) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1019) least significant
s_mul_hi_u32 s11, s24, s0 // (op 1019) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 1019) most significant: sum
s_add_u32 s7, s7, s11 // (op 1019) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1019) 

s_mov_b32 s52, s26 // (op 1019) reg expression
s_mov_b32 s53, s27 // (op 1019) reg expression
s_mov_b32 s54, s0 // (op 1019) reg expression
s_mov_b32 s55, 131072 // (op 1019) call()

v_and_b32 v16, -64, v1 // (op 1062) 
v_sub_u32 v17, v1, v16 // (op 1062) 

v_lshl_add_u32 v16, s16, 6, v17 // (op 1062) 

v_lshrrev_b32 v17, 6, v1 // (op 1062) 
v_lshlrev_b32 v20, 1, v17 // (op 1062) 

v_mul_lo_u32 v17, v20, s28 // (op 1062) 

v_add_u32 v20, v16, v17 // (op 1062) 


s_mul_i32 s1, s28, s33 // (op 1064) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1064) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1064) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1064) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1064) most significant: sum
s_add_u32 s1, s1, s7 // (op 1064) most significant: sum


s_mov_b32 s56, s30 // (op 1064) reg expression
s_mov_b32 s57, s31 // (op 1064) reg expression
s_mov_b32 s58, s0 // (op 1064) reg expression
s_mov_b32 s59, 131072 // (op 1064) call()

v_and_b32 v16, -64, v1 // (op 1081) 
v_sub_u32 v17, v1, v16 // (op 1081) 

v_lshl_add_u32 v16, s16, 6, v17 // (op 1081) 

v_lshrrev_b32 v17, 6, v1 // (op 1081) 
v_lshlrev_b32 v21, 1, v17 // (op 1081) 

v_mul_lo_u32 v17, v21, s28 // (op 1081) 

v_add_u32 v21, v16, v17 // (op 1081) 


s_mul_i32 s1, s28, s33 // (op 1083) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1083) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1083) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1083) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1083) most significant: sum
s_add_u32 s1, s1, s7 // (op 1083) most significant: sum


s_mov_b32 s56, s30 // (op 1083) reg expression
s_mov_b32 s57, s31 // (op 1083) reg expression
s_mov_b32 s58, s0 // (op 1083) reg expression
s_mov_b32 s59, 131072 // (op 1083) call()

v_and_b32 v16, 15, v1 // (op 1126) 
v_lshl_add_u32 v17, s17, 4, v16 // (op 1126) 

v_mul_lo_u32 v16, v17, s34 // (op 1126) 

v_and_b32 v17, 63, v1 // (op 1126) 
v_lshrrev_b32 v22, 4, v17 // (op 1126) 

v_add_u32 v17, v22, v16 // (op 1126) 


s_mul_i32 s1, s34, s9 // (op 1128) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1128) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1128) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1128) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1128) most significant: sum
s_add_u32 s1, s1, s7 // (op 1128) most significant: sum


s_mov_b32 s60, s36 // (op 1128) reg expression
s_mov_b32 s61, s37 // (op 1128) reg expression
s_mov_b32 s62, s0 // (op 1128) reg expression
s_mov_b32 s63, 131072 // (op 1128) call()

v_and_b32 v16, 15, v1 // (op 1145) 
v_lshl_add_u32 v22, s17, 4, v16 // (op 1145) 

v_mul_lo_u32 v16, v22, s34 // (op 1145) 

v_and_b32 v22, 63, v1 // (op 1145) 
v_lshrrev_b32 v23, 4, v22 // (op 1145) 

v_add_u32 v22, v23, v16 // (op 1145) 


s_mul_i32 s1, s34, s9 // (op 1147) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1147) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1147) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1147) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1147) most significant: sum
s_add_u32 s1, s1, s7 // (op 1147) most significant: sum


s_mov_b32 s60, s36 // (op 1147) reg expression
s_mov_b32 s61, s37 // (op 1147) reg expression
s_mov_b32 s62, s0 // (op 1147) reg expression
s_mov_b32 s63, 131072 // (op 1147) call()




buffer_load_ubyte v16, v17, s[60:63], 0 offen offset: 0 // (op 782) Load value



buffer_load_dwordx3 v[24:26], v18, s[52:55], 0 offen // (op 786) Load value
v_and_b32 v23, -4, v1 // (op 1211) 
v_sub_u32 v27, v1, v23 // (op 1211) 

v_lshlrev_b32 v23, 4, v27 // (op 1211) 

v_lshrrev_b32 v27, 2, v1 // (op 1211) 
v_lshl_add_u32 v28, v27, 6, v23 // (op 1211) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1211) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 889) 
v_mov_b64 v[28:29], s[6:7] // (op 889) convert


v_mov_b64 v[30:31], v[12:13] // (op 889) reg expression
s_mov_b32 m0, s0 // (op 889) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 889) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 889) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 889) most significant half

s_add_u32 m0, m0, 2048 // (op 889) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 889) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 889) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 889) most significant half

s_add_u32 m0, m0, 2048 // (op 889) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 889) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v28 // (op 889) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 889) most significant half

s_add_u32 m0, m0, 2048 // (op 889) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 889) Load value direct to lds




v_mov_b64 v[28:29], s[28:29] // (op 792) convert

v_mov_b32 v23, v20 // (op 792) reg expression
buffer_load_ubyte v30, v23, s[56:59], 0 offen offset: 0 // (op 792) Load value
v_add_u32 v23, v23, v28 // (op 792) 
buffer_load_ubyte v31, v23, s[56:59], 0 offen offset: 0 // (op 792) Load value





buffer_load_ubyte v23, v17, s[60:63], 0 offen offset:4 // (op 796) Load value




buffer_load_dwordx3 v[32:34], v18, s[52:55], 0 offen offset:96 // (op 800) Load value

v_and_b32 v17, -4, v1 // (op 1384) 
v_sub_u32 v18, v1, v17 // (op 1384) 

v_lshlrev_b32 v17, 4, v18 // (op 1384) 

v_lshrrev_b32 v18, 2, v1 // (op 1384) 
v_lshl_add_u32 v27, v18, 6, v17 // (op 1384) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1384) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 894) 
v_mov_b64 v[28:29], s[6:7] // (op 894) convert


s_lshl_b64 s[6:7], s[12:13], 7 // (op 894) 
v_mov_b32 v27, s6 // (op 894) 
v_mov_b32 v17, s7 // (op 894) 
v_add_co_u32 v36, s[10:11], v12, v27 // (op 894) least significant half
v_addc_co_u32 v37, s[10:11], v13, v17, s[10:11] // (op 894) most significant half




s_add_u32 s1, 8192, s0 // (op 894) 
s_mov_b32 m0, s1 // (op 894) reg expression
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v28 // (op 894) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 894) most significant half

s_add_u32 m0, m0, 2048 // (op 894) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v28 // (op 894) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 894) most significant half

s_add_u32 m0, m0, 2048 // (op 894) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v28 // (op 894) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 894) most significant half

s_add_u32 m0, m0, 2048 // (op 894) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 894) Load value direct to lds






v_mov_b64 v[12:13], s[28:29] // (op 806) convert

s_lshl_b32 s0, s28, 2 // (op 806) 
v_add_u32 v17, v20, s0 // (op 806) 

buffer_load_ubyte v28, v17, s[56:59], 0 offen offset: 0 // (op 806) Load value
v_add_u32 v17, v17, v12 // (op 806) 
buffer_load_ubyte v29, v17, s[56:59], 0 offen offset: 0 // (op 806) Load value



v_and_b32 v20, -8, v1 // (op 1269) 
v_sub_u32 v27, v1, v20 // (op 1269) 

v_lshlrev_b32 v20, 4, v27 // (op 1269) 

v_lshrrev_b32 v27, 3, v1 // (op 1269) 
v_lshl_add_u32 v12, v27, 7, v20 // (op 1269) 


v_mul_lo_u32 v13, v12, 6 // (op 1269) 

v_lshrrev_b32 v12, 3, v13 // (op 1269) 




s_waitcnt vmcnt(14)
ds_write_b96 v12, v[24:26] offset:16384 // (op 808) Store local data 

v_and_b32 v20, -64, v1 // (op 1322) 
v_sub_u32 v27, v1, v20 // (op 1322) 

v_lshrrev_b32 v20, 6, v1 // (op 1322) 
v_lshl_add_u32 v12, v20, 7, v27 // (op 1322) 





s_waitcnt vmcnt(9)
ds_write_b8 v12, v30 offset:17920 // (op 810) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v12, v31 offset:17984 // (op 810) Store local data 

v_and_b32 v20, -4, v1 // (op 1183) 
v_sub_u32 v27, v1, v20 // (op 1183) 

v_lshlrev_b32 v20, 4, v27 // (op 1183) 

v_lshrrev_b32 v27, 2, v1 // (op 1183) 
v_lshl_add_u32 v12, v27, 6, v20 // (op 1183) 


s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1183) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 1240) 
v_sub_u32 v27, v1, v20 // (op 1240) 

v_lshlrev_b32 v20, 4, v27 // (op 1240) 

v_lshrrev_b32 v27, 3, v1 // (op 1240) 
v_lshl_add_u32 v12, v27, 7, v20 // (op 1240) 


v_mul_lo_u32 v13, v12, 6 // (op 1240) 

v_lshrrev_b32 v12, 3, v13 // (op 1240) 

v_and_b32 v13, -64, v1 // (op 1293) 
v_sub_u32 v20, v1, v13 // (op 1293) 

v_lshrrev_b32 v13, 6, v1 // (op 1293) 
v_lshl_add_u32 v27, v13, 7, v20 // (op 1293) 


v_and_b32 v13, -4, v1 // (op 1369) 
v_sub_u32 v20, v1, v13 // (op 1369) 

v_lshlrev_b32 v13, 4, v20 // (op 1369) 

v_lshrrev_b32 v20, 2, v1 // (op 1369) 
v_lshl_add_u32 v17, v20, 6, v13 // (op 1369) 


s_nop 0
v_readfirstlane_b32 s1, v17 // (op 1369) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 1413) 
v_sub_u32 v20, v1, v13 // (op 1413) 

v_lshlrev_b32 v13, 4, v20 // (op 1413) 

v_lshrrev_b32 v20, 3, v1 // (op 1413) 
v_lshl_add_u32 v17, v20, 7, v13 // (op 1413) 


v_mul_lo_u32 v13, v17, 6 // (op 1413) 

v_lshrrev_b32 v20, 3, v13 // (op 1413) 

v_and_b32 v13, -64, v1 // (op 1437) 
v_sub_u32 v17, v1, v13 // (op 1437) 

v_lshrrev_b32 v13, 6, v1 // (op 1437) 
v_lshl_add_u32 v18, v13, 7, v17 // (op 1437) 


s_mov_b32 s6, 0 // (op 315) call()
s_ashr_i64 s[10:11], s[22:23], 7 // (op 314) 
s_mov_b32 s7, s10 // (op 314) convert

s_and_b32 s10, s7, 4294967294 // (op 314) 

s_cmp_lt_u32 s6, s10 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314 if false)
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314:
s_barrier  // (op 1495) 
s_add_u32 s6, s6, 2 // (op 316) 



ds_read_b64_tr_b8 v[36:37], v0 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:512 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:4096 // (op 67) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:4608 // (op 67) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1508) 



ds_read_b128 v[44:47], v3 offset:16384 // (op 72) Load local data 
ds_read_b64 v[48:49], v3 offset:16400 // (op 72) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1522) 


ds_read_u8 v13, v6 offset:17920 // (op 77) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:49], a[0:3], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 84) 
ds_read_b64_tr_b8 v[50:51], v0 offset:32 // (op 344) Transpose load local data 


ds_read_b64_tr_b8 v[52:53], v0 offset:544 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:4128 // (op 344) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v0 offset:4640 // (op 344) Transpose load local data 


ds_read_u8 v13, v6 offset:17952 // (op 348) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[44:49], a[4:7], v13, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 350) 





v_mov_b32 v13, v22 // (op 82) 

buffer_load_ubyte v16, v13, s[60:63], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v17, v19 // (op 10) 

buffer_load_dwordx3 v[24:26], v17, s[52:55], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 1498) 

s_lshl_b64 s[10:11], s[12:13], 5 // (op 873) 
v_mov_b64 v[36:37], s[10:11] // (op 873) convert


v_mov_b64 v[38:39], v[14:15] // (op 873) 
s_lshl_b64 s[10:11], s[12:13], 7 // (op 873) 
s_lshl_b64 s[14:15], s[10:11], 1 // (op 873) 

v_mov_b32 v35, s14 // (op 873) 
v_mov_b32 v40, s15 // (op 873) 
v_add_co_u32 v42, s[10:11], v38, v35 // (op 873) least significant half
v_addc_co_u32 v43, s[10:11], v39, v40, s[10:11] // (op 873) most significant half




s_mov_b32 m0, s0 // (op 873) reg expression
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v42, s[10:11], v42, v36 // (op 873) least significant half
v_addc_co_u32 v43, s[10:11], 0, v43, s[10:11] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v42, s[10:11], v42, v36 // (op 873) least significant half
v_addc_co_u32 v43, s[10:11], 0, v43, s[10:11] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v42, s[10:11], v42, v36 // (op 873) least significant half
v_addc_co_u32 v43, s[10:11], 0, v43, s[10:11] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 873) Load value direct to lds



v_mov_b64 v[36:37], s[28:29] // (op 17) convert

v_mov_b32 v35, v21 // (op 17) 
s_lshl_b32 s7, s28, 3 // (op 17) 
v_add_u32 v40, v35, s7 // (op 17) 

buffer_load_ubyte v30, v40, s[56:59], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v40, v40, v36 // (op 17) 
buffer_load_ubyte v31, v40, s[56:59], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1545) 



s_waitcnt vmcnt(14)
ds_write_b96 v20, v[32:34] offset:18176 // (op 239) Store local data 
s_barrier  // (op 1551) 



s_waitcnt vmcnt(9)
ds_write_b8 v18, v28 offset:19712 // (op 240) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v18, v29 offset:19776 // (op 240) Store local data 
s_barrier  // (op 1536) 



ds_read_b64_tr_b8 v[40:41], v7 offset:8192 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:8704 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v7 offset:12288 // (op 231) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:12800 // (op 231) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1542) 



ds_read_b128 v[48:51], v4 offset:18176 // (op 233) Load local data 
ds_read_b64 v[52:53], v4 offset:18192 // (op 233) Load local data 
s_barrier  // (op 1584) 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1548) 


ds_read_u8 v36, v9 offset:19712 // (op 235) Load local data 



ds_read_b64_tr_b8 v[54:55], v7 offset:8224 // (op 357) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v7 offset:8736 // (op 357) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:47], v[48:53], a[0:3], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 238) 
ds_read_b64_tr_b8 v[58:59], v7 offset:12320 // (op 357) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v7 offset:12832 // (op 357) Transpose load local data 


ds_read_u8 v36, v9 offset:19744 // (op 361) Load local data 
s_barrier  // (op 1622) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:61], v[48:53], a[4:7], v36, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 351) 






buffer_load_ubyte v23, v13, s[60:63], 0 offen offset:12 // (op 237) Load value




v_add_u32 v22, 8, v22 // (op 1137) 
buffer_load_dwordx3 v[32:34], v17, s[52:55], 0 offen offset:288 // (op 241) Load value

s_barrier  // (op 1539) 
v_add_u32 v19, 192, v19 // (op 1009) 

s_lshl_b64 s[10:11], s[12:13], 5 // (op 884) 
v_mov_b64 v[36:37], s[10:11] // (op 884) convert


s_lshl_b64 s[10:11], s[12:13], 7 // (op 884) 
s_mul_i32 s15, 3, s11 // (op 884) most significant: low of low * high
s_mul_i32 s14, 3, s10 // (op 884) least significant
s_mul_hi_u32 s7, 3, s10 // (op 884) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 884) most significant: sum


v_mov_b32 v13, s14 // (op 884) 
v_mov_b32 v17, s15 // (op 884) 
v_add_co_u32 v40, s[10:11], v38, v13 // (op 884) least significant half
v_addc_co_u32 v41, s[10:11], v39, v17, s[10:11] // (op 884) most significant half




s_add_u32 s7, 8192, s1 // (op 884) 
s_mov_b32 m0, s7 // (op 884) reg expression
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 884) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v36 // (op 884) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 884) most significant half

s_add_u32 m0, m0, 2048 // (op 884) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 884) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v36 // (op 884) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 884) most significant half

s_add_u32 m0, m0, 2048 // (op 884) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 884) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v36 // (op 884) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 884) most significant half

s_add_u32 m0, m0, 2048 // (op 884) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 884) Load value direct to lds





v_mov_b64 v[36:37], s[28:29] // (op 242) convert

s_lshl_b32 s7, s28, 2 // (op 242) 
s_mul_i32 s10, 3, s7 // (op 242) 

v_add_u32 v13, v35, s10 // (op 242) 

s_lshl_b64 s[10:11], s[12:13], 7 // (op 945) 
s_lshl_b64 s[14:15], s[10:11], 1 // (op 945) 

v_mov_b32 v17, s14 // (op 945) 
v_mov_b32 v38, s15 // (op 945) 
v_add_co_u32 v14, s[10:11], v14, v17 // (op 945) least significant half
v_addc_co_u32 v15, s[10:11], v15, v38, s[10:11] // (op 945) most significant half




buffer_load_ubyte v28, v13, s[56:59], 0 offen offset: 0 // (op 242) Load value
v_add_u32 v13, v13, v36 // (op 242) 
buffer_load_ubyte v29, v13, s[56:59], 0 offen offset: 0 // (op 242) Load value



s_barrier  // (op 1511) 
s_lshl_b32 s7, s28, 3 // (op 1073) 



v_add_u32 v21, v21, s7 // (op 1073) 

s_waitcnt vmcnt(14)
ds_write_b96 v12, v[24:26] offset:16384 // (op 13) Store local data 
s_barrier  // (op 1525) 



s_waitcnt vmcnt(9)
ds_write_b8 v27, v30 offset:17920 // (op 21) Store local data 
s_waitcnt vmcnt(8)
ds_write_b8 v27, v31 offset:17984 // (op 21) Store local data 
s_ashr_i64 s[10:11], s[22:23], 7 // (op 314) 
s_mov_b32 s7, s10 // (op 314) convert

s_and_b32 s10, s7, 4294967294 // (op 314) 

s_cmp_lt_u32 s6, s10 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoop_314 if true)
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoop_314:














s_barrier  // (op 1604) 
v_and_b32 v12, -4, v1 // (op 907) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 907) 
v_sub_u32 v13, v1, v12 // (op 907) 

s_lshr_b32 s6, s0, 1 // (op 907) 

v_lshlrev_b32 v12, 4, v13 // (op 907) 

v_lshl_add_u32 v13, s16, 6, v12 // (op 907) 

v_mov_b32 v31, 0 // (op 907) convert
v_mov_b32 v30, v13 // (op 907) convert

v_lshrrev_b32 v12, 2, v1 // (op 907) 
v_lshl_add_u32 v13, s6, 8, v12 // (op 907) 


v_mul_lo_u32 v15, v13, s13 // (op 907) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 907) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 907) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 907) most significant: sum


v_add_co_u32 v12, s[0:1], v30, v14 // (op 907) least significant half
v_addc_co_u32 v13, s[0:1], v31, v15, s[0:1] // (op 907) most significant half



s_mul_i32 s1, s12, s23 // (op 909) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 909) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 909) least significant
s_mul_hi_u32 s7, s12, s22 // (op 909) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 909) most significant: sum
s_add_u32 s1, s1, s7 // (op 909) most significant: sum


s_mov_b32 s48, s20 // (op 909) reg expression
s_mov_b32 s49, s21 // (op 909) reg expression
s_mov_b32 s50, s0 // (op 909) reg expression
s_mov_b32 s51, 131072 // (op 909) call()

v_and_b32 v14, -8, v1 // (op 975) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 975) 
v_sub_u32 v15, v1, v14 // (op 975) 

s_lshr_b32 s6, s0, 1 // (op 975) 

v_lshlrev_b32 v14, 4, v15 // (op 975) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 975) 


v_mov_b32 v31, 0 // (op 975) convert
v_mov_b32 v30, v15 // (op 975) convert

v_lshrrev_b32 v14, 3, v1 // (op 975) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 975) 

v_mul_lo_u32 v17, v15, s25 // (op 975) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 975) most significant: high of low * low
v_mul_lo_u32 v16, v15, s24 // (op 975) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 975) most significant: sum


v_add_co_u32 v14, s[0:1], v30, v16 // (op 975) least significant half
v_addc_co_u32 v15, s[0:1], v31, v17, s[0:1] // (op 975) most significant half



v_mul_lo_u32 v31, v15, 6 // (op 975) most significant: low of high * low
v_mul_hi_u32 v16, v14, 6 // (op 975) most significant: high of low * low
v_mul_lo_u32 v30, v14, 6 // (op 975) least significant: low of low * low
v_add_u32 v31, v31, v16 // (op 975) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[30:31] // (op 975) 

v_mov_b32 v16, v14 // (op 975) convert

s_mul_i32 s1, s9, 6 // (op 977) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 977) least significant
s_mul_hi_u32 s6, s8, 6 // (op 977) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 977) most significant: sum

s_mul_i32 s7, s24, s1 // (op 977) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 977) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 977) least significant
s_mul_hi_u32 s11, s24, s0 // (op 977) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 977) most significant: sum
s_add_u32 s7, s7, s11 // (op 977) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 977) 

s_mov_b32 s52, s26 // (op 977) reg expression
s_mov_b32 s53, s27 // (op 977) reg expression
s_mov_b32 s54, s0 // (op 977) reg expression
s_mov_b32 s55, 131072 // (op 977) call()

v_and_b32 v14, -64, v1 // (op 1039) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1039) 
v_sub_u32 v15, v1, v14 // (op 1039) 

s_lshr_b32 s6, s0, 1 // (op 1039) 

v_lshl_add_u32 v14, s16, 6, v15 // (op 1039) 

v_lshrrev_b32 v15, 6, v1 // (op 1039) 
v_lshlrev_b32 v17, 1, v15 // (op 1039) 

v_lshl_add_u32 v15, s6, 3, v17 // (op 1039) 


v_mul_lo_u32 v17, v15, s28 // (op 1039) 

v_add_u32 v15, v14, v17 // (op 1039) 


s_mul_i32 s1, s28, s33 // (op 1041) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1041) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1041) most significant: sum
s_add_u32 s1, s1, s7 // (op 1041) most significant: sum


s_mov_b32 s56, s30 // (op 1041) reg expression
s_mov_b32 s57, s31 // (op 1041) reg expression
s_mov_b32 s58, s0 // (op 1041) reg expression
s_mov_b32 s59, 131072 // (op 1041) call()

v_and_b32 v14, 63, v1 // (op 1103) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1103) 
v_lshrrev_b32 v17, 4, v14 // (op 1103) 

s_lshr_b32 s6, s0, 1 // (op 1103) 

v_and_b32 v14, 15, v1 // (op 1103) 
v_lshl_add_u32 v18, s6, 3, v17 // (op 1103) 


v_lshl_add_u32 v17, s17, 4, v14 // (op 1103) 

v_mul_lo_u32 v14, v17, s34 // (op 1103) 

v_add_u32 v17, v18, v14 // (op 1103) 


s_mul_i32 s1, s34, s9 // (op 1105) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 1105) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1105) least significant
s_mul_hi_u32 s7, s34, s8 // (op 1105) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1105) most significant: sum
s_add_u32 s1, s1, s7 // (op 1105) most significant: sum


s_mov_b32 s60, s36 // (op 1105) reg expression
s_mov_b32 s61, s37 // (op 1105) reg expression
s_mov_b32 s62, s0 // (op 1105) reg expression
s_mov_b32 s63, 131072 // (op 1105) call()

v_and_b32 v14, -4, v1 // (op 1196) 
v_sub_u32 v18, v1, v14 // (op 1196) 

v_lshlrev_b32 v14, 4, v18 // (op 1196) 

v_lshrrev_b32 v18, 2, v1 // (op 1196) 
v_lshl_add_u32 v19, v18, 6, v14 // (op 1196) 


s_nop 0
v_readfirstlane_b32 s0, v19 // (op 1196) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 1253) 
v_sub_u32 v18, v1, v14 // (op 1253) 

v_lshlrev_b32 v14, 4, v18 // (op 1253) 

v_lshrrev_b32 v18, 3, v1 // (op 1253) 
v_lshl_add_u32 v19, v18, 7, v14 // (op 1253) 


v_mul_lo_u32 v14, v19, 6 // (op 1253) 

v_lshrrev_b32 v18, 3, v14 // (op 1253) 

v_and_b32 v14, -64, v1 // (op 1306) 
v_sub_u32 v19, v1, v14 // (op 1306) 

v_lshrrev_b32 v14, 6, v1 // (op 1306) 
v_lshl_add_u32 v20, v14, 7, v19 // (op 1306) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 320) 
s_and_b32 s1, s6, 4294967294 // (op 320) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 319) 
s_mov_b32 s10, s6 // (op 319) convert

s_cmp_lt_u32 s1, s10 // (op 319) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319:
s_barrier  // (op 1501) 
s_add_u32 s1, s1, 1 // (op 321) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 878) 
v_mov_b64 v[22:23], s[6:7] // (op 878) convert


v_mov_b64 v[24:25], v[12:13] // (op 878) 
v_mov_b64 v[26:27], v[24:25] // (op 878) reg expression
s_mov_b32 m0, s0 // (op 878) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 878) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 878) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 878) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds



s_lshl_b64 s[6:7], s[12:13], 7 // (op 897) 
v_mov_b32 v14, s6 // (op 897) 
v_mov_b32 v19, s7 // (op 897) 
v_add_co_u32 v12, s[10:11], v12, v14 // (op 897) least significant half
v_addc_co_u32 v13, s[10:11], v13, v19, s[10:11] // (op 897) most significant half






v_mov_b32 v14, v16 // (op 183) 

buffer_load_dwordx3 v[22:24], v14, s[52:55], 0 offen // (op 183) Load value

v_add_u32 v16, 96, v16 // (op 967) 

s_barrier  // (op 1514) 
v_mov_b64 v[26:27], s[28:29] // (op 184) convert




v_mov_b32 v14, v15 // (op 184) 
v_mov_b32 v19, v14 // (op 184) reg expression
buffer_load_ubyte v28, v19, s[56:59], 0 offen offset: 0 // (op 184) Load value
v_add_u32 v19, v19, v26 // (op 184) 
s_waitcnt vmcnt(1)
ds_write_b96 v18, v[22:24] offset:16384 // (op 181) Store local data 
buffer_load_ubyte v29, v19, s[56:59], 0 offen offset: 0 // (op 184) Load value




s_lshl_b32 s6, s28, 2 // (op 1031) 
v_add_u32 v15, v15, s6 // (op 1031) 

s_barrier  // (op 1504) 
s_barrier  // (op 1528) 






ds_read_b64_tr_b8 v[30:31], v0 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:512 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:4096 // (op 173) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:4608 // (op 173) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v28 offset:17920 // (op 182) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1518) 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v29 offset:17984 // (op 182) Store local data 




ds_read_b128 v[24:27], v3 offset:16384 // (op 175) Load local data 
ds_read_b64 v[28:29], v3 offset:16400 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1532) 


ds_read_u8 v14, v6 offset:17920 // (op 177) Load local data 


v_mov_b32 v19, v17 // (op 179) 

buffer_load_ubyte v21, v19, s[60:63], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v17, 4, v17 // (op 1095) 



ds_read_b64_tr_b8 v[38:39], v0 offset:32 // (op 325) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:544 // (op 325) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[24:29], a[0:3], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 180) 
ds_read_b64_tr_b8 v[42:43], v0 offset:4128 // (op 325) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v0 offset:4640 // (op 325) Transpose load local data 


ds_read_u8 v14, v6 offset:17952 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[24:29], a[4:7], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 331) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 319) 
s_mov_b32 s10, s6 // (op 319) convert

s_cmp_lt_u32 s1, s10 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopTop_KLoopTail_319 if true)
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ForLoopBottom_KLoopTail_319:






















s_barrier  // (op 1608) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125
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


v_pack_b32_f16 v6, v12, v13 // (op 54) 
v_pack_b32_f16 v7, v14, v15 // (op 54) 


buffer_store_dwordx2 v[6:7], v8, s[44:47], 0 offen // (op 54) Store value


v_accvgpr_read v12, a4 // (op 517) 
v_accvgpr_read v13, a5 // (op 517) 
v_accvgpr_read v14, a6 // (op 517) 
v_accvgpr_read v15, a7 // (op 517) 
v_mul_f32 v16, v11, v12 // (op 517) 
v_mul_f32 v17, v11, v13 // (op 517) 
v_mul_f32 v18, v11, v14 // (op 517) 
v_mul_f32 v19, v11, v15 // (op 517) 

v_cvt_f16_f32 v12, v16 // (op 519) 
v_cvt_f16_f32 v13, v17 // (op 519) 
v_cvt_f16_f32 v14, v18 // (op 519) 
v_cvt_f16_f32 v15, v19 // (op 519) 


v_pack_b32_f16 v6, v12, v13 // (op 520) 
v_pack_b32_f16 v7, v14, v15 // (op 520) 


buffer_store_dwordx2 v[6:7], v8, s[44:47], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[6:7], v5, s[40:43], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v12, a4 // (op 546) 
v_accvgpr_read v13, a5 // (op 546) 
v_accvgpr_read v14, a6 // (op 546) 
v_accvgpr_read v15, a7 // (op 546) 
v_mul_f32 v16, v11, v12 // (op 546) 
v_mul_f32 v17, v11, v13 // (op 546) 
v_mul_f32 v18, v11, v14 // (op 546) 
v_mul_f32 v19, v11, v15 // (op 546) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v6 // (op 551) 
v_lshrrev_b32 v13, 16, v6 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v20, v10, v12 // (op 551) 
v_mul_f32 v21, v10, v13 // (op 551) 
v_cvt_f32_f16 v12, v7 // (op 551) 
v_lshrrev_b32 v13, 16, v7 // (op 551) 
v_cvt_f32_f16 v13, v13 // (op 551) 
v_mul_f32 v22, v10, v12 // (op 551) 
v_mul_f32 v23, v10, v13 // (op 551) 


v_add_f32 v12, v20, v16 // (op 547) 
v_add_f32 v13, v21, v17 // (op 547) 
v_add_f32 v14, v22, v18 // (op 547) 
v_add_f32 v15, v23, v19 // (op 547) 


v_cvt_f16_f32 v16, v12 // (op 548) 
v_cvt_f16_f32 v17, v13 // (op 548) 
v_cvt_f16_f32 v18, v14 // (op 548) 
v_cvt_f16_f32 v19, v15 // (op 548) 


v_pack_b32_f16 v6, v16, v17 // (op 549) 
v_pack_b32_f16 v7, v18, v19 // (op 549) 


buffer_store_dwordx2 v[6:7], v8, s[44:47], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[6:7], v5, s[40:43], 0 offen // (op 138) Load value
v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v11, v12 // (op 134) 
v_mul_f32 v17, v11, v13 // (op 134) 
v_mul_f32 v18, v11, v14 // (op 134) 
v_mul_f32 v19, v11, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v6 // (op 139) 
v_lshrrev_b32 v13, 16, v6 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v10, v12 // (op 139) 
v_mul_f32 v21, v10, v13 // (op 139) 
v_cvt_f32_f16 v12, v7 // (op 139) 
v_lshrrev_b32 v13, 16, v7 // (op 139) 
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


v_pack_b32_f16 v6, v16, v17 // (op 137) 
v_pack_b32_f16 v7, v18, v19 // (op 137) 


buffer_store_dwordx2 v[6:7], v8, s[44:47], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.LRR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end:
.size RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2, .LRR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2_end-RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 19968 // lds bytes
  .amdhsa_accum_offset 64
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
    DataTypeB: B6
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
    macrotile: [64, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2
    .symbol: RR_GEMM_NN_BF8_BF6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x16x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 19968
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 62
    .agpr_count: 8
    .max_flat_workgroup_size: 128
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

