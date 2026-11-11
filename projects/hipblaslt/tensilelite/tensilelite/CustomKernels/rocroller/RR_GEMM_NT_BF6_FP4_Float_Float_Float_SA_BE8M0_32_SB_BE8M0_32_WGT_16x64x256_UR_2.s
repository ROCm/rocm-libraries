// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.p2align 8
.type RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2,@function
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 1495) 
v_and_b32 v3, 48, v1 // (op 1495) 
v_sub_u32 v4, v0, v3 // (op 1495) 

v_lshrrev_b32 v3, 4, v0 // (op 1495) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1495) 


v_lshrrev_b32 v3, 7, v1 // (op 1495) 
v_lshlrev_b32 v4, 4, v3 // (op 1495) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1495) 


v_mul_lo_u32 v0, v3, 6 // (op 1495) 
v_lshrrev_b32 v4, 3, v0 // (op 1495) 

v_lshrrev_b32 v0, 4, v3 // (op 1495) 

v_lshlrev_b32 v3, 2, v0 // (op 1495) 

v_add_u32 v0, v4, v3 // (op 1495) 


v_lshrrev_b32 v3, 7, v1 // (op 1606) 
v_and_b32 v4, 15, v1 // (op 1606) 
v_lshl_add_u32 v5, v3, 4, v4 // (op 1606) 


v_and_b32 v3, 48, v1 // (op 1606) 
v_add_u32 v4, v3, v5 // (op 1606) 


v_and_b32 v3, 63, v1 // (op 1706) 
v_and_b32 v5, 48, v1 // (op 1706) 
v_sub_u32 v6, v3, v5 // (op 1706) 

v_lshrrev_b32 v5, 4, v3 // (op 1706) 

v_lshl_add_u32 v3, v5, 5, v6 // (op 1706) 


v_lshrrev_b32 v5, 7, v1 // (op 1706) 
v_lshlrev_b32 v6, 4, v5 // (op 1706) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 1706) 


v_mul_lo_u32 v3, v5, 6 // (op 1706) 
v_lshrrev_b32 v6, 3, v3 // (op 1706) 

v_lshrrev_b32 v3, 4, v5 // (op 1706) 

v_lshlrev_b32 v5, 2, v3 // (op 1706) 

v_add_u32 v3, v6, v5 // (op 1706) 


v_lshrrev_b32 v5, 7, v1 // (op 1759) 
v_and_b32 v6, 15, v1 // (op 1759) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 1759) 


v_and_b32 v5, 48, v1 // (op 1759) 
v_add_u32 v6, v5, v7 // (op 1759) 


v_lshrrev_b32 v5, 7, v1 // (op 1438) 
v_and_b32 v7, 63, v1 // (op 1438) 
v_lshrrev_b32 v8, 4, v7 // (op 1438) 
v_lshlrev_b32 v9, 2, v8 // (op 1438) 

v_lshl_add_u32 v8, v5, 4, v9 // (op 1438) 


v_lshl_add_u32 v5, s16, 4, v8 // (op 1438) 

v_lshrrev_b32 v8, 2, v7 // (op 1438) 

v_and_b32 v7, 3, v8 // (op 1438) 

v_and_b32 v8, 3, v1 // (op 1438) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 1438) 


v_lshrrev_b32 v7, 6, v1 // (op 1438) 
v_and_b32 v8, 1, v7 // (op 1438) 

v_lshl_add_u32 v7, v8, 4, v9 // (op 1438) 


v_lshl_add_u32 v8, s17, 6, v7 // (op 1438) 

v_mul_lo_u32 v7, v8, s2 // (op 1438) 

v_add_lshl_u32 v8, v5, v7, 2 // (op 1438) 


v_and_b32 v5, 63, v1 // (op 1553) 
v_and_b32 v7, 48, v1 // (op 1553) 
v_sub_u32 v9, v5, v7 // (op 1553) 

v_lshrrev_b32 v7, 4, v5 // (op 1553) 

v_lshl_add_u32 v5, v7, 5, v9 // (op 1553) 


v_lshrrev_b32 v7, 6, v1 // (op 1553) 
v_and_b32 v9, 1, v7 // (op 1553) 

v_lshlrev_b32 v7, 4, v9 // (op 1553) 

v_lshl_add_u32 v9, v5, 6, v7 // (op 1553) 


v_lshlrev_b32 v5, 2, v9 // (op 1553) 

v_lshrrev_b32 v7, 3, v5 // (op 1553) 

v_lshrrev_b32 v5, 6, v1 // (op 1664) 
v_and_b32 v9, 1, v5 // (op 1664) 

v_and_b32 v5, 15, v1 // (op 1664) 
s_mul_i32 s1, s2, s9 // (op 1440) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1440) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1440) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1440) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1440) most significant: sum
s_add_u32 s1, s1, s19 // (op 1440) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1440) 

v_lshl_add_u32 v10, v9, 4, v5 // (op 1664) 


v_and_b32 v5, 63, v1 // (op 1664) 
v_lshrrev_b32 v9, 4, v5 // (op 1664) 

v_lshl_add_u32 v5, v9, 6, v10 // (op 1664) 


v_lshrrev_b32 v9, 7, v1 // (op 1680) 
v_and_b32 v10, 63, v1 // (op 1680) 
v_lshrrev_b32 v11, 4, v10 // (op 1680) 
v_lshlrev_b32 v12, 2, v11 // (op 1680) 

v_lshl_add_u32 v11, v9, 4, v12 // (op 1680) 


v_lshl_add_u32 v9, s16, 4, v11 // (op 1680) 

v_lshrrev_b32 v11, 2, v10 // (op 1680) 

v_and_b32 v10, 3, v11 // (op 1680) 

v_and_b32 v11, 3, v1 // (op 1680) 
v_lshl_add_u32 v12, v10, 2, v11 // (op 1680) 


v_lshrrev_b32 v10, 6, v1 // (op 1680) 
v_and_b32 v11, 1, v10 // (op 1680) 

v_lshl_add_u32 v10, v11, 4, v12 // (op 1680) 


v_lshl_add_u32 v11, s17, 6, v10 // (op 1680) 

v_mul_lo_u32 v10, v11, s4 // (op 1680) 

v_add_lshl_u32 v11, v9, v10, 2 // (op 1680) 


v_and_b32 v9, 63, v1 // (op 1735) 
v_and_b32 v10, 48, v1 // (op 1735) 
v_sub_u32 v12, v9, v10 // (op 1735) 

v_lshrrev_b32 v10, 4, v9 // (op 1735) 

v_lshl_add_u32 v9, v10, 5, v12 // (op 1735) 


v_lshrrev_b32 v10, 6, v1 // (op 1735) 
v_and_b32 v12, 1, v10 // (op 1735) 

v_lshlrev_b32 v10, 4, v12 // (op 1735) 

v_lshl_add_u32 v12, v9, 6, v10 // (op 1735) 


v_lshlrev_b32 v9, 2, v12 // (op 1735) 

v_lshrrev_b32 v10, 3, v9 // (op 1735) 

v_lshrrev_b32 v9, 6, v1 // (op 1788) 
v_and_b32 v12, 1, v9 // (op 1788) 

v_and_b32 v9, 15, v1 // (op 1788) 
s_mul_i32 s1, s4, s9 // (op 1682) most significant: low of low * high
v_lshl_add_u32 v13, v12, 4, v9 // (op 1788) 


v_and_b32 v9, 63, v1 // (op 1788) 
v_lshrrev_b32 v12, 4, v9 // (op 1788) 

v_lshl_add_u32 v9, v12, 6, v13 // (op 1788) 


s_mul_i32 s38, s5, s8 // (op 1682) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1682) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1682) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1682) most significant: sum
s_add_u32 s1, s1, s39 // (op 1682) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 1682) 

s_mov_b32 s40, s6 // (op 1440) reg expression
s_mov_b32 s41, s7 // (op 1440) reg expression
s_mov_b32 s42, s18 // (op 1440) reg expression
s_mov_b32 s43, 131072 // (op 1440) call()

s_mov_b32 s44, s10 // (op 1682) reg expression

s_mov_b32 s45, s11 // (op 1682) reg expression
s_mov_b32 s46, s38 // (op 1682) reg expression
s_mov_b32 s47, 131072 // (op 1682) call()



v_mov_b32 v12, s14 // (op 2) Move value
v_mov_b32 v13, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 398) call()
v_accvgpr_write a5, 0.00000 // (op 398) call()
v_accvgpr_write a6, 0.00000 // (op 398) call()
v_accvgpr_write a7, 0.00000 // (op 398) call()
s_lshl_b32 s0, s16, 4 // (op 1205) 
v_lshlrev_b32 v14, 1, v1 // (op 1205) 
s_mov_b32 s15, 0 // (op 1205) convert
s_mov_b32 s14, s0 // (op 1205) convert

v_mul_lo_u32 v17, v14, s13 // (op 1205) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 1205) most significant: high of low * low
v_mul_lo_u32 v16, v14, s12 // (op 1205) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1205) most significant: sum


v_mov_b32 v14, s14 // (op 1205) 
v_mov_b32 v15, s15 // (op 1205) 
v_add_co_u32 v18, s[0:1], v14, v16 // (op 1205) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 1205) most significant half





v_mul_lo_u32 v15, v19, 6 // (op 1205) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 1205) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 1205) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 1205) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1205) 

v_mov_b32 v14, v16 // (op 1205) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 1207) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1207) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1207) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1207) most significant: sum

s_mul_i32 s15, s12, s1 // (op 1207) most significant: low of low * high
s_mul_i32 s6, s13, s0 // (op 1207) most significant: low of high * low
s_mul_i32 s14, s12, s0 // (op 1207) least significant
s_mul_hi_u32 s7, s12, s0 // (op 1207) most significant: high of low * low
s_add_u32 s15, s15, s6 // (op 1207) most significant: sum
s_add_u32 s15, s15, s7 // (op 1207) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 1207) 

s_mov_b32 s8, s20 // (op 1207) reg expression
s_mov_b32 s9, s21 // (op 1207) reg expression
s_mov_b32 s10, s0 // (op 1207) reg expression
s_mov_b32 s11, 131072 // (op 1207) call()

s_lshl_b32 s0, s16, 4 // (op 1224) 
v_lshlrev_b32 v15, 1, v1 // (op 1224) 
s_mov_b32 s7, 0 // (op 1224) convert
s_mov_b32 s6, s0 // (op 1224) convert

v_mul_lo_u32 v17, v15, s13 // (op 1224) most significant: low of low * high
v_mul_hi_u32 v18, v15, s12 // (op 1224) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 1224) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1224) most significant: sum


v_mov_b32 v15, s6 // (op 1224) 
v_mov_b32 v18, s7 // (op 1224) 
v_add_co_u32 v20, s[0:1], v15, v16 // (op 1224) least significant half
v_addc_co_u32 v21, s[0:1], v18, v17, s[0:1] // (op 1224) most significant half





v_mul_lo_u32 v19, v21, 6 // (op 1224) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 1224) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 1224) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1224) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 1224) 

v_mov_b32 v15, v16 // (op 1224) convert

s_mul_i32 s1, s23, 6 // (op 1226) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1226) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1226) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1226) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1226) most significant: low of low * high
s_mul_i32 s14, s13, s0 // (op 1226) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1226) least significant
s_mul_hi_u32 s15, s12, s0 // (op 1226) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1226) most significant: sum
s_add_u32 s7, s7, s15 // (op 1226) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1226) 

s_mov_b32 s8, s20 // (op 1226) reg expression
s_mov_b32 s9, s21 // (op 1226) reg expression
s_mov_b32 s10, s0 // (op 1226) reg expression
s_mov_b32 s11, 131072 // (op 1226) call()

v_and_b32 v16, -2, v1 // (op 1269) 
v_sub_u32 v17, v1, v16 // (op 1269) 

v_lshlrev_b32 v16, 5, v17 // (op 1269) 

v_lshl_add_u32 v17, s17, 6, v16 // (op 1269) 

v_mov_b32 v19, 0 // (op 1269) convert
v_mov_b32 v18, v17 // (op 1269) convert

v_lshrrev_b32 v16, 1, v1 // (op 1269) 
v_lshlrev_b32 v17, 2, v16 // (op 1269) 

v_mul_lo_u32 v21, v17, s25 // (op 1269) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 1269) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 1269) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 1269) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 1269) most significant half



v_lshlrev_b64 v[16:17], 2, v[20:21] // (op 1269) 

v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 1269) 

v_mov_b32 v16, v18 // (op 1269) convert

s_mul_i32 s1, s24, s23 // (op 1271) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1271) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1271) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1271) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1271) most significant: sum
s_add_u32 s1, s1, s7 // (op 1271) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1271) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1271) 

s_mov_b32 s48, s26 // (op 1271) reg expression
s_mov_b32 s49, s27 // (op 1271) reg expression
s_mov_b32 s50, s0 // (op 1271) reg expression
s_mov_b32 s51, 131072 // (op 1271) call()

v_and_b32 v17, -2, v1 // (op 1288) 
v_sub_u32 v18, v1, v17 // (op 1288) 

v_lshlrev_b32 v17, 5, v18 // (op 1288) 

v_lshl_add_u32 v18, s17, 6, v17 // (op 1288) 

v_mov_b32 v21, 0 // (op 1288) convert
v_mov_b32 v20, v18 // (op 1288) convert

v_lshrrev_b32 v17, 1, v1 // (op 1288) 
v_lshlrev_b32 v18, 2, v17 // (op 1288) 

v_mul_lo_u32 v23, v18, s25 // (op 1288) most significant: low of low * high
v_mul_hi_u32 v17, v18, s24 // (op 1288) most significant: high of low * low
v_mul_lo_u32 v22, v18, s24 // (op 1288) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 1288) most significant: sum

v_add_co_u32 v22, s[0:1], v22, v20 // (op 1288) least significant half
v_addc_co_u32 v23, s[0:1], v23, v21, s[0:1] // (op 1288) most significant half



v_lshlrev_b64 v[20:21], 2, v[22:23] // (op 1288) 

v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 1288) 

v_mov_b32 v17, v18 // (op 1288) convert

s_mul_i32 s1, s24, s23 // (op 1290) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1290) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1290) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1290) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1290) most significant: sum
s_add_u32 s1, s1, s7 // (op 1290) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1290) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1290) 

s_mov_b32 s48, s26 // (op 1290) reg expression
s_mov_b32 s49, s27 // (op 1290) reg expression
s_mov_b32 s50, s0 // (op 1290) reg expression
s_mov_b32 s51, 131072 // (op 1290) call()

v_and_b32 v18, -16, v1 // (op 1333) 
v_sub_u32 v19, v1, v18 // (op 1333) 

v_lshl_add_u32 v18, s16, 4, v19 // (op 1333) 

v_lshrrev_b32 v19, 4, v1 // (op 1333) 
v_mul_lo_u32 v20, v19, s28 // (op 1333) 

v_add_u32 v19, v18, v20 // (op 1333) 


s_mul_i32 s1, s28, s33 // (op 1335) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1335) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1335) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1335) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1335) most significant: sum
s_add_u32 s1, s1, s7 // (op 1335) most significant: sum


s_mov_b32 s52, s30 // (op 1335) reg expression
s_mov_b32 s53, s31 // (op 1335) reg expression
s_mov_b32 s54, s0 // (op 1335) reg expression
s_mov_b32 s55, 131072 // (op 1335) call()

v_and_b32 v18, -16, v1 // (op 1352) 
v_sub_u32 v20, v1, v18 // (op 1352) 

v_lshl_add_u32 v18, s16, 4, v20 // (op 1352) 

v_lshrrev_b32 v20, 4, v1 // (op 1352) 
v_mul_lo_u32 v21, v20, s28 // (op 1352) 

v_add_u32 v20, v18, v21 // (op 1352) 


s_mul_i32 s1, s28, s33 // (op 1354) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1354) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1354) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1354) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1354) most significant: sum
s_add_u32 s1, s1, s7 // (op 1354) most significant: sum


s_mov_b32 s52, s30 // (op 1354) reg expression
s_mov_b32 s53, s31 // (op 1354) reg expression
s_mov_b32 s54, s0 // (op 1354) reg expression
s_mov_b32 s55, 131072 // (op 1354) call()

v_and_b32 v18, -16, v1 // (op 1397) 
v_sub_u32 v21, v1, v18 // (op 1397) 

v_lshlrev_b32 v18, 2, v21 // (op 1397) 

v_lshl_add_u32 v21, s17, 6, v18 // (op 1397) 

v_lshrrev_b32 v18, 4, v1 // (op 1397) 
v_mul_lo_u32 v22, v18, s34 // (op 1397) 
v_add_u32 v22, v22, v21 // (op 1397) 


s_mul_i32 s1, s34, s33 // (op 1399) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1399) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1399) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1399) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1399) most significant: sum
s_add_u32 s1, s1, s7 // (op 1399) most significant: sum


s_mov_b32 s56, s36 // (op 1399) reg expression
s_mov_b32 s57, s37 // (op 1399) reg expression
s_mov_b32 s58, s0 // (op 1399) reg expression
s_mov_b32 s59, 131072 // (op 1399) call()

v_and_b32 v18, -16, v1 // (op 1416) 
v_sub_u32 v21, v1, v18 // (op 1416) 

v_lshlrev_b32 v18, 2, v21 // (op 1416) 

v_lshl_add_u32 v21, s17, 6, v18 // (op 1416) 

v_lshrrev_b32 v18, 4, v1 // (op 1416) 
v_mul_lo_u32 v23, v18, s34 // (op 1416) 
v_add_u32 v23, v23, v21 // (op 1416) 


s_mul_i32 s1, s34, s33 // (op 1418) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1418) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1418) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1418) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1418) most significant: sum
s_add_u32 s1, s1, s7 // (op 1418) most significant: sum


s_mov_b32 s56, s36 // (op 1418) reg expression
s_mov_b32 s57, s37 // (op 1418) reg expression
s_mov_b32 s58, s0 // (op 1418) reg expression
s_mov_b32 s59, 131072 // (op 1418) call()


s_lshl_b64 s[0:1], s[24:25], 2 // (op 1029) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1029) 


v_mov_b32 v18, v16 // (op 1029) reg expression
buffer_load_dwordx4 v[26:29], v18, s[48:51], 0 offen // (op 1029) Load value
v_add_u32 v18, v18, v24 // (op 1029) 
buffer_load_dwordx4 v[30:33], v18, s[48:51], 0 offen // (op 1029) Load value
v_add_u32 v18, v18, v24 // (op 1029) 
buffer_load_dwordx4 v[34:37], v18, s[48:51], 0 offen // (op 1029) Load value
v_add_u32 v18, v18, v24 // (op 1029) 
buffer_load_dwordx4 v[38:41], v18, s[48:51], 0 offen // (op 1029) Load value




v_mov_b32 v18, v22 // (op 1032) reg expression

buffer_load_dword v21, v18, s[56:59], 0 offen // (op 1032) Load value


s_mul_i32 s1, s13, 6 // (op 1035) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 1035) least significant
s_mul_hi_u32 s6, s12, 6 // (op 1035) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1035) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1035) 


v_mov_b32 v18, v14 // (op 1035) reg expression
buffer_load_dwordx3 v[42:44], v18, s[8:11], 0 offen // (op 1035) Load value
v_add_u32 v18, v18, v24 // (op 1035) 
buffer_load_dwordx3 v[46:48], v18, s[8:11], 0 offen // (op 1035) Load value




v_mov_b32 v18, v19 // (op 1038) reg expression

buffer_load_ubyte v45, v18, s[52:55], 0 offen offset: 0 // (op 1038) Load value


s_lshl_b64 s[0:1], s[24:25], 2 // (op 1041) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1041) 


s_lshl_b64 s[0:1], s[24:25], 8 // (op 1041) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 1041) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 1041) 

v_add_u32 v18, v16, s0 // (op 1041) 

buffer_load_dwordx4 v[50:53], v18, s[48:51], 0 offen // (op 1041) Load value
v_add_u32 v18, v18, v24 // (op 1041) 
buffer_load_dwordx4 v[54:57], v18, s[48:51], 0 offen // (op 1041) Load value
v_add_u32 v18, v18, v24 // (op 1041) 
buffer_load_dwordx4 v[58:61], v18, s[48:51], 0 offen // (op 1041) Load value
v_add_u32 v18, v18, v24 // (op 1041) 
buffer_load_dwordx4 v[62:65], v18, s[48:51], 0 offen // (op 1041) Load value





s_lshl_b32 s0, s34, 3 // (op 1044) 
v_add_u32 v16, v22, s0 // (op 1044) 


buffer_load_dword v18, v16, s[56:59], 0 offen // (op 1044) Load value



s_mul_i32 s1, s13, 6 // (op 1047) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 1047) least significant
s_mul_hi_u32 s6, s12, 6 // (op 1047) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1047) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1047) 


s_lshl_b64 s[0:1], s[12:13], 8 // (op 1047) 
s_mul_i32 s7, s1, 6 // (op 1047) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 1047) least significant
s_mul_hi_u32 s14, s0, 6 // (op 1047) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1047) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 1047) 

v_add_u32 v16, v14, s0 // (op 1047) 

buffer_load_dwordx3 v[66:68], v16, s[8:11], 0 offen // (op 1047) Load value
v_add_u32 v16, v16, v24 // (op 1047) 
buffer_load_dwordx3 v[70:72], v16, s[8:11], 0 offen // (op 1047) Load value





s_lshl_b32 s0, s28, 3 // (op 1050) 
v_add_u32 v14, v19, s0 // (op 1050) 


buffer_load_ubyte v16, v14, s[52:55], 0 offen offset: 0 // (op 1050) Load value


v_and_b32 v14, -2, v1 // (op 1536) 
v_sub_u32 v19, v1, v14 // (op 1536) 

v_lshlrev_b32 v14, 5, v19 // (op 1536) 

v_lshrrev_b32 v19, 1, v1 // (op 1536) 
v_lshl_add_u32 v22, v19, 8, v14 // (op 1536) 


v_lshlrev_b32 v14, 2, v22 // (op 1536) 

v_lshrrev_b32 v19, 3, v14 // (op 1536) 




s_waitcnt vmcnt(15)
ds_write_b128 v19, v[26:29] // (op 1052) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v19, v[30:33] offset:32 // (op 1052) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v19, v[34:37] offset:64 // (op 1052) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v19, v[38:41] offset:96 // (op 1052) Store local data 

v_and_b32 v14, -16, v1 // (op 1647) 
v_sub_u32 v19, v1, v14 // (op 1647) 

v_lshlrev_b32 v14, 2, v19 // (op 1647) 

v_lshrrev_b32 v19, 4, v1 // (op 1647) 
v_lshl_add_u32 v22, v19, 6, v14 // (op 1647) 





s_waitcnt vmcnt(11)
ds_write_b32 v22, v21 offset:8192 // (op 1053) Store local data 

v_lshlrev_b32 v14, 5, v1 // (op 1483) 
v_mul_lo_u32 v19, v14, 6 // (op 1483) 
v_lshrrev_b32 v22, 3, v19 // (op 1483) 

v_lshrrev_b32 v19, 4, v14 // (op 1483) 

v_lshlrev_b32 v14, 2, v19 // (op 1483) 

v_add_u32 v19, v22, v14 // (op 1483) 





s_waitcnt vmcnt(10)
ds_write_b96 v19, v[42:44] offset:8704 // (op 1054) Store local data 
s_waitcnt vmcnt(9)
ds_write_b96 v19, v[46:48] offset:8720 // (op 1054) Store local data 

v_and_b32 v14, -16, v1 // (op 1594) 
v_sub_u32 v19, v1, v14 // (op 1594) 

v_and_b32 v14, 4294967280, v1 // (op 1594) 
v_add_u32 v22, v14, v19 // (op 1594) 





s_waitcnt vmcnt(8)
ds_write_b8 v22, v45 offset:12800 // (op 1055) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1858) 



ds_read_b64_tr_b4 v[74:75], v7 // (op 1073) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v7 offset:512 // (op 1073) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1899) 


ds_read_u8 v14, v5 offset:8192 // (op 1086) Load local data 



ds_read_b64_tr_b4 v[78:79], v7 offset:16 // (op 1078) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v7 offset:528 // (op 1078) Transpose load local data 


ds_read_u8 v19, v5 offset:8224 // (op 1091) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1839) 



ds_read_b96_tr_b6 v[82:84], v0 offset:8704 // (op 1069) Transpose load local data 
ds_read_b96_tr_b6 v[85:87], v0 offset:8960 // (op 1069) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1879) 


ds_read_u8 v22, v4 offset:12800 // (op 1082) Load local data 
v_lshlrev_b32 v49, 5, v1 // (op 1454) 
v_mul_lo_u32 v69, v49, 6 // (op 1454) 
v_lshrrev_b32 v73, 3, v69 // (op 1454) 

v_lshrrev_b32 v69, 4, v49 // (op 1454) 

v_lshlrev_b32 v49, 2, v69 // (op 1454) 

v_add_u32 v69, v73, v49 // (op 1454) 


v_and_b32 v49, -2, v1 // (op 1507) 
v_sub_u32 v73, v1, v49 // (op 1507) 

v_lshlrev_b32 v49, 5, v73 // (op 1507) 

v_lshrrev_b32 v73, 1, v1 // (op 1507) 
v_lshl_add_u32 v24, v73, 8, v49 // (op 1507) 


v_lshlrev_b32 v25, 2, v24 // (op 1507) 

v_lshrrev_b32 v24, 3, v25 // (op 1507) 

v_and_b32 v25, -16, v1 // (op 1565) 
v_sub_u32 v49, v1, v25 // (op 1565) 

v_and_b32 v25, 4294967280, v1 // (op 1565) 
v_add_u32 v73, v25, v49 // (op 1565) 


v_and_b32 v25, -16, v1 // (op 1618) 
v_sub_u32 v49, v1, v25 // (op 1618) 

v_lshlrev_b32 v25, 2, v49 // (op 1618) 

v_lshrrev_b32 v49, 4, v1 // (op 1618) 
v_lshl_add_u32 v88, v49, 6, v25 // (op 1618) 


v_lshlrev_b32 v25, 5, v1 // (op 1694) 
v_mul_lo_u32 v49, v25, 6 // (op 1694) 
v_lshrrev_b32 v89, 3, v49 // (op 1694) 

v_lshrrev_b32 v49, 4, v25 // (op 1694) 

v_lshlrev_b32 v25, 2, v49 // (op 1694) 

v_add_u32 v49, v89, v25 // (op 1694) 


v_and_b32 v25, -2, v1 // (op 1718) 
v_sub_u32 v89, v1, v25 // (op 1718) 

v_lshlrev_b32 v25, 5, v89 // (op 1718) 

v_lshrrev_b32 v89, 1, v1 // (op 1718) 
v_lshl_add_u32 v90, v89, 8, v25 // (op 1718) 


v_lshlrev_b32 v25, 2, v90 // (op 1718) 

v_lshrrev_b32 v89, 3, v25 // (op 1718) 

v_and_b32 v25, -16, v1 // (op 1747) 
v_sub_u32 v90, v1, v25 // (op 1747) 

v_and_b32 v25, 4294967280, v1 // (op 1747) 
v_add_u32 v91, v25, v90 // (op 1747) 


v_and_b32 v25, -16, v1 // (op 1771) 
v_sub_u32 v90, v1, v25 // (op 1771) 

v_lshlrev_b32 v25, 2, v90 // (op 1771) 

v_lshrrev_b32 v90, 4, v1 // (op 1771) 
v_lshl_add_u32 v92, v90, 6, v25 // (op 1771) 


s_mov_b32 s0, 0 // (op 400) call()
s_ashr_i64 s[6:7], s[22:23], 8 // (op 399) 
s_mov_b32 s1, s6 // (op 399) convert

s_and_b32 s6, s1, 4294967294 // (op 399) 

s_cmp_lt_u32 s0, s6 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 // (op 399) Condition: Top (jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 if false)
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399:

s_add_u32 s0, s0, 2 // (op 401) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[94:95], 3, s[6:7] // (op 10) 


v_mov_b32 v25, v17 // (op 10) 
s_lshl_b64 s[6:7], s[24:25], 8 // (op 10) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 10) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 10) 

s_lshl_b32 s1, s6, 1 // (op 10) 

v_add_u32 v90, v25, s1 // (op 10) 

buffer_load_dwordx4 v[26:29], v90, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v90, v90, v94 // (op 10) 
buffer_load_dwordx4 v[30:33], v90, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v90, v90, v94 // (op 10) 
buffer_load_dwordx4 v[34:37], v90, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v90, v90, v94 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:87], v[74:77], a[0:3], v22, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 88) 
buffer_load_dwordx4 v[38:41], v90, s[48:51], 0 offen // (op 10) Load value




s_barrier  // (op 1848) 
v_mov_b32 v90, v23 // (op 25) 



s_lshl_b32 s1, s34, 4 // (op 25) 
v_add_u32 v93, v90, s1 // (op 25) 


buffer_load_dword v21, v93, s[56:59], 0 offen // (op 25) Load value

ds_read_b96_tr_b6 v[94:96], v0 offset:10752 // (op 91) Transpose load local data 

ds_read_b96_tr_b6 v[97:99], v0 offset:11008 // (op 91) Transpose load local data 
s_mul_i32 s7, s13, 6 // (op 4) most significant: low of high * low
s_barrier  // (op 1868) 
s_mul_i32 s6, s12, 6 // (op 4) least significant



ds_read_b64_tr_b4 v[74:75], v7 offset:4096 // (op 95) Transpose load local data 
s_mul_hi_u32 s1, s12, 6 // (op 4) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 4) most significant: sum

v_lshrrev_b64 v[100:101], 3, s[6:7] // (op 4) 


v_mov_b32 v93, v15 // (op 4) 
s_lshl_b64 s[6:7], s[12:13], 8 // (op 4) 
s_mul_i32 s15, s7, 6 // (op 4) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 4) least significant
s_mul_hi_u32 s1, s6, 6 // (op 4) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 4) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v102, v93, s1 // (op 4) 

buffer_load_dwordx3 v[42:44], v102, s[8:11], 0 offen // (op 4) Load value
v_add_u32 v102, v102, v100 // (op 4) 
ds_read_b64_tr_b4 v[76:77], v7 offset:4608 // (op 95) Transpose load local data 
buffer_load_dwordx3 v[46:48], v102, s[8:11], 0 offen // (op 4) Load value




s_barrier  // (op 1889) 
v_mov_b32 v100, v20 // (op 17) 


ds_read_u8 v101, v4 offset:12864 // (op 99) Load local data 
s_lshl_b32 s1, s28, 4 // (op 17) 
s_barrier  // (op 1909) 
v_add_u32 v102, v100, s1 // (op 17) 




ds_read_u8 v14, v5 offset:8448 // (op 103) Load local data 
buffer_load_ubyte v45, v102, s[52:55], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1929) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[94:99], v[74:77], a[0:3], v101, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 105) 



s_waitcnt vmcnt(15)
ds_write_b128 v89, v[50:53] offset:12928 // (op 302) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:87], v[78:81], a[4:7], v22, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 447) 
s_waitcnt vmcnt(14)
ds_write_b128 v89, v[54:57] offset:12960 // (op 302) Store local data 



ds_read_b64_tr_b4 v[78:79], v7 offset:4112 // (op 451) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v7 offset:4624 // (op 451) Transpose load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v89, v[58:61] offset:12992 // (op 302) Store local data 


ds_read_u8 v19, v5 offset:8480 // (op 455) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v89, v[62:65] offset:13024 // (op 302) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[94:99], v[78:81], a[4:7], v101, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 456) 


s_barrier  // (op 1941) 



s_waitcnt vmcnt(11)
ds_write_b32 v92, v18 offset:21120 // (op 304) Store local data 
s_barrier  // (op 1922) 



s_waitcnt vmcnt(10)
ds_write_b96 v49, v[66:68] offset:21632 // (op 283) Store local data 
s_waitcnt vmcnt(9)
ds_write_b96 v49, v[70:72] offset:21648 // (op 283) Store local data 
s_barrier  // (op 1935) 



s_waitcnt vmcnt(8)
ds_write_b8 v91, v16 offset:25728 // (op 303) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1925) 



ds_read_b64_tr_b4 v[74:75], v10 offset:12928 // (op 287) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v10 offset:13440 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1938) 


ds_read_u8 v14, v9 offset:21120 // (op 291) Load local data 



ds_read_b64_tr_b4 v[94:95], v10 offset:12944 // (op 470) Transpose load local data 
ds_read_b64_tr_b4 v[96:97], v10 offset:13456 // (op 470) Transpose load local data 


ds_read_u8 v98, v9 offset:21152 // (op 474) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1919) 



ds_read_b96_tr_b6 v[82:84], v3 offset:21632 // (op 285) Transpose load local data 
ds_read_b96_tr_b6 v[85:87], v3 offset:21888 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1932) 


ds_read_u8 v22, v6 offset:25728 // (op 289) Load local data 

s_lshl_b64 s[6:7], s[24:25], 2 // (op 305) 
v_lshrrev_b64 v[102:103], 3, s[6:7] // (op 305) 


s_lshl_b64 s[6:7], s[24:25], 8 // (op 305) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 305) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 305) 

s_mul_i32 s1, 3, s6 // (op 305) 

v_add_u32 v99, v25, s1 // (op 305) 

buffer_load_dwordx4 v[50:53], v99, s[48:51], 0 offen // (op 305) Load value
v_add_u32 v99, v99, v102 // (op 305) 
buffer_load_dwordx4 v[54:57], v99, s[48:51], 0 offen // (op 305) Load value
v_add_u32 v99, v99, v102 // (op 305) 
buffer_load_dwordx4 v[58:61], v99, s[48:51], 0 offen // (op 305) Load value
v_add_u32 v99, v99, v102 // (op 305) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:87], v[74:77], a[0:3], v22, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 292) 
buffer_load_dwordx4 v[62:65], v99, s[48:51], 0 offen // (op 305) Load value








v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:87], v[94:97], a[4:7], v22, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 457) 
s_lshl_b32 s1, s34, 3 // (op 307) 


s_mul_i32 s6, 3, s1 // (op 307) 

v_add_u32 v25, v90, s6 // (op 307) 


buffer_load_dword v18, v25, s[56:59], 0 offen // (op 307) Load value

s_lshl_b64 s[6:7], s[24:25], 8 // (op 1280) 

s_lshl_b64 s[14:15], s[6:7], 2 // (op 1280) 


s_lshl_b32 s1, s34, 4 // (op 1408) 
v_add_u32 v23, v23, s1 // (op 1408) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 1280) 

s_lshl_b32 s1, s6, 1 // (op 1280) 

v_add_u32 v17, v17, s1 // (op 1280) 

s_mul_i32 s7, s13, 6 // (op 282) most significant: low of high * low
s_mul_i32 s6, s12, 6 // (op 282) least significant
s_mul_hi_u32 s1, s12, 6 // (op 282) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 282) most significant: sum

v_lshrrev_b64 v[94:95], 3, s[6:7] // (op 282) 


s_lshl_b64 s[6:7], s[12:13], 8 // (op 282) 
s_mul_i32 s15, s7, 6 // (op 282) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 282) least significant
s_mul_hi_u32 s1, s6, 6 // (op 282) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 282) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 282) 

s_mul_i32 s1, 3, s6 // (op 282) 

v_add_u32 v25, v93, s1 // (op 282) 

buffer_load_dwordx3 v[66:68], v25, s[8:11], 0 offen // (op 282) Load value
v_add_u32 v25, v25, v94 // (op 282) 
ds_read_b96_tr_b6 v[102:104], v3 offset:23680 // (op 294) Transpose load local data 
ds_read_b96_tr_b6 v[105:107], v3 offset:23936 // (op 294) Transpose load local data 
buffer_load_dwordx3 v[70:72], v25, s[8:11], 0 offen // (op 282) Load value








s_barrier  // (op 1986) 
s_lshl_b32 s1, s28, 3 // (op 306) 
s_mul_i32 s6, 3, s1 // (op 306) 

v_add_u32 v25, v100, s6 // (op 306) 


buffer_load_ubyte v16, v25, s[52:55], 0 offen offset: 0 // (op 306) Load value

s_lshl_b64 s[6:7], s[12:13], 8 // (op 1216) 

s_mul_i32 s15, s7, 6 // (op 1216) most significant: low of high * low
s_barrier  // (op 1861) 
s_mul_i32 s14, s6, 6 // (op 1216) least significant



s_mul_hi_u32 s1, s6, 6 // (op 1216) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 1216) most significant: sum


s_lshl_b32 s1, s28, 4 // (op 1344) 
v_add_u32 v20, v20, s1 // (op 1344) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 1216) 

s_lshl_b32 s1, s6, 1 // (op 1216) 

v_add_u32 v15, v15, s1 // (op 1216) 

ds_read_b64_tr_b4 v[98:99], v10 offset:17024 // (op 296) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v10 offset:17536 // (op 296) Transpose load local data 
s_waitcnt vmcnt(15)
ds_write_b128 v24, v[26:29] // (op 13) Store local data 


ds_read_u8 v25, v6 offset:25792 // (op 298) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v24, v[30:33] offset:32 // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v24, v[34:37] offset:64 // (op 13) Store local data 


s_barrier  // (op 1990) 
ds_read_u8 v90, v9 offset:21376 // (op 300) Load local data 
s_waitcnt vmcnt(12)
ds_write_b128 v24, v[38:41] offset:96 // (op 13) Store local data 



ds_read_b64_tr_b4 v[94:95], v10 offset:17040 // (op 461) Transpose load local data 
s_barrier  // (op 1902) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[102:107], v[98:101], a[0:3], v25, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 301) 





ds_read_b64_tr_b4 v[96:97], v10 offset:17552 // (op 461) Transpose load local data 
s_waitcnt vmcnt(11)
ds_write_b32 v88, v21 offset:8192 // (op 30) Store local data 


s_barrier  // (op 2026) 
s_barrier  // (op 1842) 
ds_read_u8 v90, v9 offset:21408 // (op 465) Load local data 



s_waitcnt vmcnt(10)
ds_write_b96 v69, v[42:44] offset:8704 // (op 6) Store local data 
s_barrier  // (op 2030) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:107], v[94:97], a[4:7], v25, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 466) 
s_waitcnt vmcnt(9)
ds_write_b96 v69, v[46:48] offset:8720 // (op 6) Store local data 




s_barrier  // (op 1882) 



s_waitcnt vmcnt(8)
ds_write_b8 v73, v45 offset:12800 // (op 21) Store local data 
s_barrier  // (op 1871) 



ds_read_b64_tr_b4 v[74:75], v7 // (op 75) Transpose load local data 
ds_read_b64_tr_b4 v[76:77], v7 offset:512 // (op 75) Transpose load local data 
s_barrier  // (op 1912) 


ds_read_u8 v14, v5 offset:8192 // (op 85) Load local data 



ds_read_b64_tr_b4 v[78:79], v7 offset:16 // (op 442) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v7 offset:528 // (op 442) Transpose load local data 


ds_read_u8 v19, v5 offset:8224 // (op 446) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1851) 



ds_read_b96_tr_b6 v[82:84], v0 offset:8704 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[85:87], v0 offset:8960 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1892) 


ds_read_u8 v22, v4 offset:12800 // (op 80) Load local data 
s_ashr_i64 s[6:7], s[22:23], 8 // (op 399) 
s_mov_b32 s1, s6 // (op 399) convert

s_and_b32 s6, s1, 4294967294 // (op 399) 

s_cmp_lt_u32 s0, s6 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 // (op 399) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 if true)
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399:

















v_lshlrev_b32 v15, 1, v1 // (op 1178) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1178) 
s_lshr_b32 s6, s0, 1 // (op 1178) 

v_lshl_add_u32 v17, s6, 9, v15 // (op 1178) 


s_lshl_b32 s0, s16, 4 // (op 1178) 
v_mul_lo_u32 v21, v17, s13 // (op 1178) most significant: low of low * high
v_mul_hi_u32 v15, v17, s12 // (op 1178) most significant: high of low * low
v_mul_lo_u32 v20, v17, s12 // (op 1178) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1178) most significant: sum


s_mov_b32 s7, 0 // (op 1178) convert
s_mov_b32 s6, s0 // (op 1178) convert

v_mov_b32 v15, s6 // (op 1178) 
v_mov_b32 v17, s7 // (op 1178) 
v_add_co_u32 v48, s[0:1], v15, v20 // (op 1178) least significant half
v_addc_co_u32 v49, s[0:1], v17, v21, s[0:1] // (op 1178) most significant half





v_mul_lo_u32 v21, v49, 6 // (op 1178) most significant: low of high * low
v_mul_hi_u32 v15, v48, 6 // (op 1178) most significant: high of low * low
v_mul_lo_u32 v20, v48, 6 // (op 1178) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1178) most significant: sum


v_lshrrev_b64 v[48:49], 3, v[20:21] // (op 1178) 

v_mov_b32 v15, v48 // (op 1178) convert

s_mul_i32 s1, s23, 6 // (op 1180) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1180) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1180) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1180) most significant: sum

s_mul_i32 s7, s12, s1 // (op 1180) most significant: low of low * high
s_mul_i32 s14, s13, s0 // (op 1180) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 1180) least significant
s_mul_hi_u32 s15, s12, s0 // (op 1180) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1180) most significant: sum
s_add_u32 s7, s7, s15 // (op 1180) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1180) 

s_mov_b32 s8, s20 // (op 1180) reg expression
s_mov_b32 s9, s21 // (op 1180) reg expression
s_mov_b32 s10, s0 // (op 1180) reg expression
s_mov_b32 s11, 131072 // (op 1180) call()

v_lshrrev_b32 v17, 1, v1 // (op 1246) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1246) 
v_lshlrev_b32 v69, 2, v17 // (op 1246) 

s_lshr_b32 s6, s0, 1 // (op 1246) 

v_and_b32 v17, -2, v1 // (op 1246) 
v_lshl_add_u32 v73, s6, 9, v69 // (op 1246) 


v_sub_u32 v69, v1, v17 // (op 1246) 

v_lshlrev_b32 v17, 5, v69 // (op 1246) 

v_lshl_add_u32 v69, s17, 6, v17 // (op 1246) 

v_mov_b32 v21, 0 // (op 1246) convert
v_mov_b32 v20, v69 // (op 1246) convert

v_mul_lo_u32 v49, v73, s25 // (op 1246) most significant: low of low * high
v_mul_hi_u32 v17, v73, s24 // (op 1246) most significant: high of low * low
v_mul_lo_u32 v48, v73, s24 // (op 1246) least significant: low of low * low
v_add_u32 v49, v49, v17 // (op 1246) most significant: sum

v_add_co_u32 v48, s[0:1], v48, v20 // (op 1246) least significant half
v_addc_co_u32 v49, s[0:1], v49, v21, s[0:1] // (op 1246) most significant half



v_lshlrev_b64 v[20:21], 2, v[48:49] // (op 1246) 

v_lshrrev_b64 v[48:49], 3, v[20:21] // (op 1246) 

v_mov_b32 v17, v48 // (op 1246) convert

s_mul_i32 s1, s24, s23 // (op 1248) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1248) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1248) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1248) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1248) most significant: sum
s_add_u32 s1, s1, s7 // (op 1248) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1248) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1248) 

s_mov_b32 s48, s26 // (op 1248) reg expression
s_mov_b32 s49, s27 // (op 1248) reg expression
s_mov_b32 s50, s0 // (op 1248) reg expression
s_mov_b32 s51, 131072 // (op 1248) call()

v_and_b32 v69, -16, v1 // (op 1310) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1310) 
v_sub_u32 v73, v1, v69 // (op 1310) 

s_lshr_b32 s6, s0, 1 // (op 1310) 

v_lshl_add_u32 v69, s16, 4, v73 // (op 1310) 

v_lshrrev_b32 v73, 4, v1 // (op 1310) 
v_lshl_add_u32 v20, s6, 4, v73 // (op 1310) 


v_mul_lo_u32 v21, v20, s28 // (op 1310) 

v_add_u32 v20, v69, v21 // (op 1310) 


s_mul_i32 s1, s28, s33 // (op 1312) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1312) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1312) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1312) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1312) most significant: sum
s_add_u32 s1, s1, s7 // (op 1312) most significant: sum


s_mov_b32 s52, s30 // (op 1312) reg expression
s_mov_b32 s53, s31 // (op 1312) reg expression
s_mov_b32 s54, s0 // (op 1312) reg expression
s_mov_b32 s55, 131072 // (op 1312) call()

v_lshrrev_b32 v21, 4, v1 // (op 1374) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1374) 
v_and_b32 v69, -16, v1 // (op 1374) 
s_lshr_b32 s6, s0, 1 // (op 1374) 

v_sub_u32 v73, v1, v69 // (op 1374) 

v_lshl_add_u32 v69, s6, 4, v21 // (op 1374) 


v_lshlrev_b32 v21, 2, v73 // (op 1374) 

v_lshl_add_u32 v73, s17, 6, v21 // (op 1374) 

v_mul_lo_u32 v21, v69, s34 // (op 1374) 
v_add_u32 v21, v21, v73 // (op 1374) 


s_mul_i32 s1, s34, s33 // (op 1376) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1376) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1376) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1376) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1376) most significant: sum
s_add_u32 s1, s1, s7 // (op 1376) most significant: sum


s_mov_b32 s56, s36 // (op 1376) reg expression
s_mov_b32 s57, s37 // (op 1376) reg expression
s_mov_b32 s58, s0 // (op 1376) reg expression
s_mov_b32 s59, 131072 // (op 1376) call()

v_lshlrev_b32 v69, 5, v1 // (op 1467) 
v_mul_lo_u32 v73, v69, 6 // (op 1467) 
v_lshrrev_b32 v23, 3, v73 // (op 1467) 

v_lshrrev_b32 v73, 4, v69 // (op 1467) 

v_lshlrev_b32 v69, 2, v73 // (op 1467) 

v_add_u32 v73, v23, v69 // (op 1467) 


v_and_b32 v69, -2, v1 // (op 1520) 
v_sub_u32 v23, v1, v69 // (op 1520) 

v_lshlrev_b32 v69, 5, v23 // (op 1520) 

v_lshrrev_b32 v23, 1, v1 // (op 1520) 
v_lshl_add_u32 v24, v23, 8, v69 // (op 1520) 


v_lshlrev_b32 v23, 2, v24 // (op 1520) 

v_lshrrev_b32 v69, 3, v23 // (op 1520) 

v_and_b32 v23, -16, v1 // (op 1578) 
v_sub_u32 v24, v1, v23 // (op 1578) 

v_and_b32 v23, 4294967280, v1 // (op 1578) 
v_add_u32 v25, v23, v24 // (op 1578) 


v_and_b32 v23, -16, v1 // (op 1631) 
v_sub_u32 v24, v1, v23 // (op 1631) 

v_lshlrev_b32 v23, 2, v24 // (op 1631) 

v_lshrrev_b32 v24, 4, v1 // (op 1631) 
v_lshl_add_u32 v26, v24, 6, v23 // (op 1631) 


s_ashr_i64 s[0:1], s[22:23], 8 // (op 405) 
s_and_b32 s6, s0, 4294967294 // (op 405) 

s_ashr_i64 s[0:1], s[22:23], 8 // (op 404) 
s_mov_b32 s7, s0 // (op 404) convert

s_cmp_lt_u32 s6, s7 // (op 404) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404:

s_add_u32 s6, s6, 1 // (op 406) 
s_mul_i32 s1, s13, 6 // (op 198) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 198) least significant
s_mul_hi_u32 s7, s12, 6 // (op 198) most significant: high of low * low
s_add_u32 s1, s1, s7 // (op 198) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 198) 


v_mov_b32 v14, v15 // (op 198) 
v_mov_b32 v16, v14 // (op 198) reg expression
buffer_load_dwordx3 v[22:24], v16, s[8:11], 0 offen // (op 198) Load value
v_add_u32 v16, v16, v18 // (op 198) 
buffer_load_dwordx3 v[28:30], v16, s[8:11], 0 offen // (op 198) Load value



s_lshl_b64 s[0:1], s[12:13], 8 // (op 1168) 
s_mul_i32 s15, s1, 6 // (op 1168) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 1168) least significant
s_mul_hi_u32 s7, s0, 6 // (op 1168) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 1168) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 1168) 

v_add_u32 v15, v15, s0 // (op 1168) 


s_barrier  // (op 1845) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 221) 



v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 221) 


v_mov_b32 v14, v17 // (op 221) 
v_mov_b32 v16, v14 // (op 221) reg expression
buffer_load_dwordx4 v[32:35], v16, s[48:51], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v18 // (op 221) 
s_waitcnt vmcnt(2)
ds_write_b96 v73, v[22:24] offset:8704 // (op 199) Store local data 
buffer_load_dwordx4 v[36:39], v16, s[48:51], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v18 // (op 221) 
buffer_load_dwordx4 v[40:43], v16, s[48:51], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v18 // (op 221) 
buffer_load_dwordx4 v[44:47], v16, s[48:51], 0 offen // (op 221) Load value



s_lshl_b64 s[0:1], s[24:25], 8 // (op 1238) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 1238) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 1238) 

v_add_u32 v17, v17, s0 // (op 1238) 

s_waitcnt vmcnt(4)
ds_write_b96 v73, v[28:30] offset:8720 // (op 199) Store local data 



v_mov_b32 v14, v20 // (op 222) 
s_barrier  // (op 1864) 
v_mov_b32 v16, v14 // (op 222) reg expression




s_waitcnt vmcnt(3)
ds_write_b128 v69, v[32:35] // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v69, v[36:39] offset:32 // (op 218) Store local data 
buffer_load_ubyte v18, v16, s[52:55], 0 offen offset: 0 // (op 222) Load value


s_lshl_b32 s7, s28, 3 // (op 1302) 
v_add_u32 v20, v20, s7 // (op 1302) 

s_waitcnt vmcnt(2)
ds_write_b128 v69, v[40:43] offset:64 // (op 218) Store local data 


v_mov_b32 v14, v21 // (op 223) 
v_mov_b32 v16, v14 // (op 223) reg expression

buffer_load_dword v19, v16, s[56:59], 0 offen // (op 223) Load value


s_lshl_b32 s7, s34, 3 // (op 1366) 
v_add_u32 v21, v21, s7 // (op 1366) 

s_waitcnt vmcnt(2)
ds_write_b128 v69, v[44:47] offset:96 // (op 218) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1854) 



s_barrier  // (op 1885) 
ds_read_b96_tr_b6 v[28:30], v0 offset:8704 // (op 201) Transpose load local data 



ds_read_b96_tr_b6 v[31:33], v0 offset:8960 // (op 201) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v25, v18 offset:12800 // (op 219) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1875) 



s_barrier  // (op 1905) 
ds_read_b64_tr_b4 v[34:35], v7 // (op 203) Transpose load local data 



ds_read_b64_tr_b4 v[36:37], v7 offset:512 // (op 203) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v26, v19 offset:8192 // (op 220) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1895) 


ds_read_u8 v14, v4 offset:12800 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1915) 


ds_read_u8 v16, v5 offset:8192 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[34:37], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 208) 





ds_read_b96_tr_b6 v[34:36], v0 offset:10752 // (op 210) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:11008 // (op 210) Transpose load local data 



ds_read_b64_tr_b4 v[40:41], v7 offset:4096 // (op 212) Transpose load local data 
ds_read_b64_tr_b4 v[42:43], v7 offset:4608 // (op 212) Transpose load local data 


ds_read_u8 v16, v4 offset:12864 // (op 214) Load local data 


ds_read_u8 v27, v5 offset:8448 // (op 216) Load local data 



ds_read_b64_tr_b4 v[44:45], v7 offset:16 // (op 412) Transpose load local data 
ds_read_b64_tr_b4 v[46:47], v7 offset:528 // (op 412) Transpose load local data 


ds_read_u8 v18, v5 offset:8224 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:33], v[44:47], a[4:7], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 417) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:39], v[40:43], a[0:3], v16, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 217) 
ds_read_b64_tr_b4 v[28:29], v7 offset:4112 // (op 421) Transpose load local data 


ds_read_b64_tr_b4 v[30:31], v7 offset:4624 // (op 421) Transpose load local data 


ds_read_u8 v14, v5 offset:8480 // (op 425) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:39], v[28:31], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b100 // (op 426) 




s_ashr_i64 s[0:1], s[22:23], 8 // (op 404) 
s_mov_b32 s7, s0 // (op 404) convert

s_cmp_lt_u32 s6, s7 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 if true)
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404:























s_barrier  // (op 2012) 








v_cmp_eq_f32 vcc, 0.00000, v12 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155
v_accvgpr_read v4, a0 // (op 50) 
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v14, v13, v4 // (op 50) 
v_mul_f32 v15, v13, v5 // (op 50) 
v_mul_f32 v16, v13, v6 // (op 50) 
v_mul_f32 v17, v13, v7 // (op 50) 



v_mov_b32 v0, v11 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 56) Store value


v_accvgpr_read v4, a4 // (op 710) 
v_accvgpr_read v5, a5 // (op 710) 
v_accvgpr_read v6, a6 // (op 710) 
v_accvgpr_read v7, a7 // (op 710) 
v_mul_f32 v14, v13, v4 // (op 710) 
v_mul_f32 v15, v13, v5 // (op 710) 
v_mul_f32 v16, v13, v6 // (op 710) 
v_mul_f32 v17, v13, v7 // (op 710) 



s_lshl_b32 s0, s4, 7 // (op 712) 
v_add_u32 v0, v11, s0 // (op 712) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 712) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155:


v_mov_b32 v0, v8 // (op 167) reg expression

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 167) Load value

v_accvgpr_read v14, a0 // (op 164) 
v_accvgpr_read v15, a1 // (op 164) 
v_accvgpr_read v16, a2 // (op 164) 
v_accvgpr_read v17, a3 // (op 164) 
v_mul_f32 v18, v13, v14 // (op 164) 
v_mul_f32 v19, v13, v15 // (op 164) 
v_mul_f32 v20, v13, v16 // (op 164) 
v_mul_f32 v21, v13, v17 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v12, v4 // (op 168) 
v_mul_f32 v15, v12, v5 // (op 168) 
v_mul_f32 v16, v12, v6 // (op 168) 
v_mul_f32 v17, v12, v7 // (op 168) 

v_add_f32 v4, v14, v18 // (op 165) 
v_add_f32 v5, v15, v19 // (op 165) 
v_add_f32 v6, v16, v20 // (op 165) 
v_add_f32 v7, v17, v21 // (op 165) 




v_mov_b32 v0, v11 // (op 166) reg expression

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 166) Store value




s_lshl_b32 s0, s2, 7 // (op 734) 
v_add_u32 v0, v8, s0 // (op 734) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 734) Load value

v_accvgpr_read v14, a4 // (op 731) 
v_accvgpr_read v15, a5 // (op 731) 
v_accvgpr_read v16, a6 // (op 731) 
v_accvgpr_read v17, a7 // (op 731) 
v_mul_f32 v18, v13, v14 // (op 731) 
v_mul_f32 v19, v13, v15 // (op 731) 
v_mul_f32 v20, v13, v16 // (op 731) 
v_mul_f32 v21, v13, v17 // (op 731) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v12, v4 // (op 735) 
v_mul_f32 v15, v12, v5 // (op 735) 
v_mul_f32 v16, v12, v6 // (op 735) 
v_mul_f32 v17, v12, v7 // (op 735) 

v_add_f32 v4, v14, v18 // (op 732) 
v_add_f32 v5, v15, v19 // (op 732) 
v_add_f32 v6, v16, v20 // (op 732) 
v_add_f32 v7, v17, v21 // (op 732) 




s_lshl_b32 s0, s4, 7 // (op 733) 
v_add_u32 v0, v11, s0 // (op 733) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 733) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155:










s_endpgm  // End of RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.LRR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end:
.size RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2, .LRR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end-RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
  .amdhsa_next_free_vgpr 116
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25856 // lds bytes
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: F4
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
    macrotile: [16, 64, 256]
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
  - .name: RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
    .symbol: RR_GEMM_NT_BF6_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 25856
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 108
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

