// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.p2align 8
.type RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2,@function
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_exec_begin:


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


v_mul_lo_u32 v5, v9, 6 // (op 1553) 
v_lshrrev_b32 v7, 3, v5 // (op 1553) 

v_lshrrev_b32 v5, 4, v9 // (op 1553) 

v_lshlrev_b32 v9, 2, v5 // (op 1553) 

v_add_u32 v5, v7, v9 // (op 1553) 


v_lshrrev_b32 v7, 6, v1 // (op 1664) 
v_and_b32 v9, 1, v7 // (op 1664) 

v_and_b32 v7, 15, v1 // (op 1664) 
s_mul_i32 s1, s2, s9 // (op 1440) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1440) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1440) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1440) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1440) most significant: sum
s_add_u32 s1, s1, s19 // (op 1440) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1440) 

v_lshl_add_u32 v10, v9, 4, v7 // (op 1664) 


v_and_b32 v7, 63, v1 // (op 1664) 
v_lshrrev_b32 v9, 4, v7 // (op 1664) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 1664) 


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


v_mul_lo_u32 v9, v12, 6 // (op 1735) 
v_lshrrev_b32 v10, 3, v9 // (op 1735) 

v_lshrrev_b32 v9, 4, v12 // (op 1735) 

v_lshlrev_b32 v12, 2, v9 // (op 1735) 

v_add_u32 v9, v10, v12 // (op 1735) 


v_lshrrev_b32 v10, 6, v1 // (op 1788) 
v_and_b32 v12, 1, v10 // (op 1788) 

v_and_b32 v10, 15, v1 // (op 1788) 
s_mul_i32 s1, s4, s9 // (op 1682) most significant: low of low * high
v_lshl_add_u32 v13, v12, 4, v10 // (op 1788) 


v_and_b32 v10, 63, v1 // (op 1788) 
v_lshrrev_b32 v12, 4, v10 // (op 1788) 

v_lshl_add_u32 v10, v12, 6, v13 // (op 1788) 


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

s_lshl_b32 s0, s17, 6 // (op 1269) 
v_lshlrev_b32 v16, 1, v1 // (op 1269) 
s_mov_b32 s7, 0 // (op 1269) convert
s_mov_b32 s6, s0 // (op 1269) convert

v_mul_lo_u32 v19, v16, s25 // (op 1269) most significant: low of low * high
v_mul_hi_u32 v17, v16, s24 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v18, v16, s24 // (op 1269) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 1269) most significant: sum

v_mov_b32 v17, s6 // (op 1269) 
v_mov_b32 v20, s7 // (op 1269) 
v_add_co_u32 v18, s[0:1], v18, v17 // (op 1269) least significant half
v_addc_co_u32 v19, s[0:1], v19, v20, s[0:1] // (op 1269) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 1269) most significant: low of high * low
v_mul_hi_u32 v20, v18, 6 // (op 1269) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 1269) least significant: low of low * low
v_add_u32 v17, v17, v20 // (op 1269) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 1269) 

v_mov_b32 v16, v18 // (op 1269) convert

s_mul_i32 s1, s23, 6 // (op 1271) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1271) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1271) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1271) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1271) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1271) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1271) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1271) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1271) most significant: sum
s_add_u32 s7, s7, s15 // (op 1271) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 1271) 

s_mov_b32 s48, s26 // (op 1271) reg expression
s_mov_b32 s49, s27 // (op 1271) reg expression
s_mov_b32 s50, s0 // (op 1271) reg expression
s_mov_b32 s51, 131072 // (op 1271) call()

s_lshl_b32 s0, s17, 6 // (op 1288) 
v_lshlrev_b32 v17, 1, v1 // (op 1288) 
s_mov_b32 s7, 0 // (op 1288) convert
s_mov_b32 s6, s0 // (op 1288) convert

v_mul_lo_u32 v19, v17, s25 // (op 1288) most significant: low of low * high
v_mul_hi_u32 v20, v17, s24 // (op 1288) most significant: high of low * low
v_mul_lo_u32 v18, v17, s24 // (op 1288) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1288) most significant: sum

v_mov_b32 v20, s6 // (op 1288) 
v_mov_b32 v21, s7 // (op 1288) 
v_add_co_u32 v18, s[0:1], v18, v20 // (op 1288) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 1288) most significant half





v_mul_lo_u32 v21, v19, 6 // (op 1288) most significant: low of high * low
v_mul_hi_u32 v17, v18, 6 // (op 1288) most significant: high of low * low
v_mul_lo_u32 v20, v18, 6 // (op 1288) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1288) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 1288) 

v_mov_b32 v17, v18 // (op 1288) convert

s_mul_i32 s1, s23, 6 // (op 1290) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1290) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1290) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1290) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1290) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1290) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1290) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1290) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1290) most significant: sum
s_add_u32 s7, s7, s15 // (op 1290) most significant: sum



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


s_mul_i32 s1, s25, 6 // (op 1029) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 1029) least significant
s_mul_hi_u32 s6, s24, 6 // (op 1029) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1029) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1029) 


v_mov_b32 v18, v16 // (op 1029) reg expression
buffer_load_dwordx3 v[26:28], v18, s[48:51], 0 offen // (op 1029) Load value
buffer_load_dwordx3 v[30:32], v18, s[48:51], 0 offen offset:12 // (op 1029) Load value
buffer_load_dwordx3 v[34:36], v18, s[48:51], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx3 v[38:40], v18, s[48:51], 0 offen offset:36 // (op 1029) Load value
v_add_u32 v18, v18, v24 // (op 1029) 
buffer_load_dwordx3 v[42:44], v18, s[48:51], 0 offen // (op 1029) Load value
buffer_load_dwordx3 v[46:48], v18, s[48:51], 0 offen offset:12 // (op 1029) Load value
buffer_load_dwordx3 v[50:52], v18, s[48:51], 0 offen offset:24 // (op 1029) Load value
buffer_load_dwordx3 v[54:56], v18, s[48:51], 0 offen offset:36 // (op 1029) Load value




v_mov_b32 v18, v22 // (op 1032) reg expression

buffer_load_dword v21, v18, s[56:59], 0 offen // (op 1032) Load value


s_mul_i32 s1, s13, 6 // (op 1035) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 1035) least significant
s_mul_hi_u32 s6, s12, 6 // (op 1035) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1035) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1035) 


v_mov_b32 v18, v14 // (op 1035) reg expression
buffer_load_dwordx3 v[58:60], v18, s[8:11], 0 offen // (op 1035) Load value
v_add_u32 v18, v18, v24 // (op 1035) 
buffer_load_dwordx3 v[62:64], v18, s[8:11], 0 offen // (op 1035) Load value




v_mov_b32 v18, v19 // (op 1038) reg expression

buffer_load_ubyte v29, v18, s[52:55], 0 offen offset: 0 // (op 1038) Load value


s_mul_i32 s1, s25, 6 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 1041) least significant
s_mul_hi_u32 s6, s24, 6 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1041) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 1041) 


s_lshl_b64 s[0:1], s[24:25], 8 // (op 1041) 
s_mul_i32 s7, s1, 6 // (op 1041) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 1041) least significant
s_mul_hi_u32 s14, s0, 6 // (op 1041) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1041) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 1041) 

v_add_u32 v18, v16, s0 // (op 1041) 

buffer_load_dwordx3 v[66:68], v18, s[48:51], 0 offen // (op 1041) Load value
buffer_load_dwordx3 v[70:72], v18, s[48:51], 0 offen offset:12 // (op 1041) Load value
buffer_load_dwordx3 v[74:76], v18, s[48:51], 0 offen offset:24 // (op 1041) Load value
buffer_load_dwordx3 v[78:80], v18, s[48:51], 0 offen offset:36 // (op 1041) Load value
v_add_u32 v18, v18, v24 // (op 1041) 
buffer_load_dwordx3 v[82:84], v18, s[48:51], 0 offen // (op 1041) Load value
buffer_load_dwordx3 v[86:88], v18, s[48:51], 0 offen offset:12 // (op 1041) Load value
buffer_load_dwordx3 v[90:92], v18, s[48:51], 0 offen offset:24 // (op 1041) Load value
buffer_load_dwordx3 v[94:96], v18, s[48:51], 0 offen offset:36 // (op 1041) Load value





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

buffer_load_dwordx3 v[98:100], v16, s[8:11], 0 offen // (op 1047) Load value
v_add_u32 v16, v16, v24 // (op 1047) 
buffer_load_dwordx3 v[102:104], v16, s[8:11], 0 offen // (op 1047) Load value





s_lshl_b32 s0, s28, 3 // (op 1050) 
v_add_u32 v14, v19, s0 // (op 1050) 


buffer_load_ubyte v16, v14, s[52:55], 0 offen offset: 0 // (op 1050) Load value


v_lshlrev_b32 v14, 7, v1 // (op 1536) 
v_mul_lo_u32 v19, v14, 6 // (op 1536) 
v_lshrrev_b32 v22, 3, v19 // (op 1536) 

v_lshrrev_b32 v19, 4, v14 // (op 1536) 

v_lshlrev_b32 v14, 2, v19 // (op 1536) 

v_add_u32 v19, v22, v14 // (op 1536) 





s_waitcnt vmcnt(23)
ds_write_b96 v19, v[26:28] // (op 1052) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v19, v[30:32] offset:16 // (op 1052) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v19, v[34:36] offset:32 // (op 1052) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v19, v[38:40] offset:48 // (op 1052) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v19, v[42:44] offset:64 // (op 1052) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v19, v[46:48] offset:80 // (op 1052) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v19, v[50:52] offset:96 // (op 1052) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v19, v[54:56] offset:112 // (op 1052) Store local data 

v_and_b32 v14, -16, v1 // (op 1647) 
v_sub_u32 v19, v1, v14 // (op 1647) 

v_lshlrev_b32 v14, 2, v19 // (op 1647) 

v_lshrrev_b32 v19, 4, v1 // (op 1647) 
v_lshl_add_u32 v22, v19, 6, v14 // (op 1647) 





s_waitcnt vmcnt(15)
ds_write_b32 v22, v21 offset:16384 // (op 1053) Store local data 

v_lshlrev_b32 v14, 5, v1 // (op 1483) 
v_mul_lo_u32 v19, v14, 6 // (op 1483) 
v_lshrrev_b32 v22, 3, v19 // (op 1483) 

v_lshrrev_b32 v19, 4, v14 // (op 1483) 

v_lshlrev_b32 v14, 2, v19 // (op 1483) 

v_add_u32 v19, v22, v14 // (op 1483) 





s_waitcnt vmcnt(14)
ds_write_b96 v19, v[58:60] offset:16896 // (op 1054) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v19, v[62:64] offset:16912 // (op 1054) Store local data 

v_and_b32 v14, -16, v1 // (op 1594) 
v_sub_u32 v19, v1, v14 // (op 1594) 

v_and_b32 v14, 4294967280, v1 // (op 1594) 
v_add_u32 v22, v14, v19 // (op 1594) 





s_waitcnt vmcnt(12)
ds_write_b8 v22, v29 offset:20992 // (op 1055) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1858) 



ds_read_b96_tr_b6 v[106:108], v5 // (op 1073) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v5 offset:1024 // (op 1073) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1899) 


ds_read_u8 v14, v7 offset:16384 // (op 1086) Load local data 



ds_read_b96_tr_b6 v[112:114], v5 offset:32 // (op 1078) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v5 offset:1056 // (op 1078) Transpose load local data 


ds_read_u8 v19, v7 offset:16416 // (op 1091) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1839) 



ds_read_b96_tr_b6 v[118:120], v0 offset:16896 // (op 1069) Transpose load local data 
ds_read_b96_tr_b6 v[121:123], v0 offset:17152 // (op 1069) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1879) 


ds_read_u8 v22, v4 offset:20992 // (op 1082) Load local data 
v_lshlrev_b32 v33, 5, v1 // (op 1454) 
v_mul_lo_u32 v37, v33, 6 // (op 1454) 
v_lshrrev_b32 v41, 3, v37 // (op 1454) 

v_lshrrev_b32 v37, 4, v33 // (op 1454) 

v_lshlrev_b32 v33, 2, v37 // (op 1454) 

v_add_u32 v37, v41, v33 // (op 1454) 


v_lshlrev_b32 v33, 7, v1 // (op 1507) 
v_mul_lo_u32 v41, v33, 6 // (op 1507) 
v_lshrrev_b32 v45, 3, v41 // (op 1507) 

v_lshrrev_b32 v41, 4, v33 // (op 1507) 

v_lshlrev_b32 v33, 2, v41 // (op 1507) 

v_add_u32 v41, v45, v33 // (op 1507) 


v_and_b32 v33, -16, v1 // (op 1565) 
v_sub_u32 v45, v1, v33 // (op 1565) 

v_and_b32 v33, 4294967280, v1 // (op 1565) 
v_add_u32 v49, v33, v45 // (op 1565) 


v_and_b32 v33, -16, v1 // (op 1618) 
v_sub_u32 v45, v1, v33 // (op 1618) 

v_lshlrev_b32 v33, 2, v45 // (op 1618) 

v_lshrrev_b32 v45, 4, v1 // (op 1618) 
v_lshl_add_u32 v53, v45, 6, v33 // (op 1618) 


v_lshlrev_b32 v33, 5, v1 // (op 1694) 
v_mul_lo_u32 v45, v33, 6 // (op 1694) 
v_lshrrev_b32 v57, 3, v45 // (op 1694) 

v_lshrrev_b32 v45, 4, v33 // (op 1694) 

v_lshlrev_b32 v33, 2, v45 // (op 1694) 

v_add_u32 v45, v57, v33 // (op 1694) 


v_lshlrev_b32 v33, 7, v1 // (op 1718) 
v_mul_lo_u32 v57, v33, 6 // (op 1718) 
v_lshrrev_b32 v61, 3, v57 // (op 1718) 

v_lshrrev_b32 v57, 4, v33 // (op 1718) 

v_lshlrev_b32 v33, 2, v57 // (op 1718) 

v_add_u32 v57, v61, v33 // (op 1718) 


v_and_b32 v33, -16, v1 // (op 1747) 
v_sub_u32 v61, v1, v33 // (op 1747) 

v_and_b32 v33, 4294967280, v1 // (op 1747) 
v_add_u32 v65, v33, v61 // (op 1747) 


v_and_b32 v33, -16, v1 // (op 1771) 
v_sub_u32 v61, v1, v33 // (op 1771) 

v_lshlrev_b32 v33, 2, v61 // (op 1771) 

v_lshrrev_b32 v61, 4, v1 // (op 1771) 
v_lshl_add_u32 v69, v61, 6, v33 // (op 1771) 


s_mov_b32 s0, 0 // (op 400) call()
s_ashr_i64 s[6:7], s[22:23], 8 // (op 399) 
s_mov_b32 s1, s6 // (op 399) convert

s_and_b32 s6, s1, 4294967294 // (op 399) 

s_cmp_lt_u32 s0, s6 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 // (op 399) Condition: Top (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399 if false)
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399:

s_add_u32 s0, s0, 2 // (op 401) 
s_mul_i32 s7, s25, 6 // (op 10) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 10) least significant
s_mul_hi_u32 s1, s24, 6 // (op 10) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 10) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[6:7] // (op 10) 


v_mov_b32 v33, v17 // (op 10) 
s_lshl_b64 s[6:7], s[24:25], 8 // (op 10) 
s_mul_i32 s15, s7, 6 // (op 10) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 10) least significant
s_mul_hi_u32 s1, s6, 6 // (op 10) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 10) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 10) 

s_lshl_b32 s1, s6, 1 // (op 10) 

v_add_u32 v61, v33, s1 // (op 10) 

buffer_load_dwordx3 v[26:28], v61, s[48:51], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[30:32], v61, s[48:51], 0 offen offset:12 // (op 10) Load value
buffer_load_dwordx3 v[34:36], v61, s[48:51], 0 offen offset:24 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[118:123], v[106:111], a[0:3], v22, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 88) 
buffer_load_dwordx3 v[38:40], v61, s[48:51], 0 offen offset:36 // (op 10) Load value
s_barrier  // (op 1848) 
v_add_u32 v61, v61, v24 // (op 10) 



buffer_load_dwordx3 v[42:44], v61, s[48:51], 0 offen // (op 10) Load value
buffer_load_dwordx3 v[46:48], v61, s[48:51], 0 offen offset:12 // (op 10) Load value
ds_read_b96_tr_b6 v[124:126], v0 offset:18944 // (op 91) Transpose load local data 
ds_read_b96_tr_b6 v[127:129], v0 offset:19200 // (op 91) Transpose load local data 
buffer_load_dwordx3 v[50:52], v61, s[48:51], 0 offen offset:24 // (op 10) Load value
s_barrier  // (op 1868) 
buffer_load_dwordx3 v[54:56], v61, s[48:51], 0 offen offset:36 // (op 10) Load value







ds_read_b96_tr_b6 v[106:108], v5 offset:8192 // (op 95) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v5 offset:9216 // (op 95) Transpose load local data 
v_mov_b32 v61, v23 // (op 25) 
s_barrier  // (op 1889) 
s_lshl_b32 s1, s34, 4 // (op 25) 


ds_read_u8 v73, v4 offset:21056 // (op 99) Load local data 
v_add_u32 v77, v61, s1 // (op 25) 


s_barrier  // (op 1909) 
buffer_load_dword v21, v77, s[56:59], 0 offen // (op 25) Load value




s_mul_i32 s7, s13, 6 // (op 4) most significant: low of high * low
s_mul_i32 s6, s12, 6 // (op 4) least significant
s_mul_hi_u32 s1, s12, 6 // (op 4) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 4) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[6:7] // (op 4) 


v_mov_b32 v77, v15 // (op 4) 
s_lshl_b64 s[6:7], s[12:13], 8 // (op 4) 
s_mul_i32 s15, s7, 6 // (op 4) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 4) least significant
s_mul_hi_u32 s1, s6, 6 // (op 4) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 4) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v81, v77, s1 // (op 4) 

ds_read_u8 v14, v7 offset:16640 // (op 103) Load local data 
buffer_load_dwordx3 v[58:60], v81, s[8:11], 0 offen // (op 4) Load value
v_add_u32 v81, v81, v24 // (op 4) 
buffer_load_dwordx3 v[62:64], v81, s[8:11], 0 offen // (op 4) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:129], v[106:111], a[0:3], v73, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 105) 


v_mov_b32 v81, v20 // (op 17) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[112:117], a[4:7], v22, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 447) 
s_lshl_b32 s1, s28, 4 // (op 17) 



ds_read_b96_tr_b6 v[112:114], v5 offset:8224 // (op 451) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v5 offset:9248 // (op 451) Transpose load local data 
v_add_u32 v85, v81, s1 // (op 17) 




ds_read_u8 v19, v7 offset:16672 // (op 455) Load local data 
buffer_load_ubyte v29, v85, s[52:55], 0 offen offset: 0 // (op 17) Load value

s_barrier  // (op 1929) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:129], v[112:117], a[4:7], v73, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 456) 





s_waitcnt vmcnt(23)
ds_write_b96 v57, v[66:68] offset:21120 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v57, v[70:72] offset:21136 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v57, v[74:76] offset:21152 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v57, v[78:80] offset:21168 // (op 302) Store local data 
s_waitcnt vmcnt(19)
ds_write_b96 v57, v[82:84] offset:21184 // (op 302) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v57, v[86:88] offset:21200 // (op 302) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v57, v[90:92] offset:21216 // (op 302) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v57, v[94:96] offset:21232 // (op 302) Store local data 
s_barrier  // (op 1941) 



s_waitcnt vmcnt(15)
ds_write_b32 v69, v18 offset:37504 // (op 304) Store local data 
s_barrier  // (op 1922) 



s_waitcnt vmcnt(14)
ds_write_b96 v45, v[98:100] offset:38016 // (op 283) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v45, v[102:104] offset:38032 // (op 283) Store local data 
s_barrier  // (op 1935) 



s_waitcnt vmcnt(12)
ds_write_b8 v65, v16 offset:42112 // (op 303) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1925) 



ds_read_b96_tr_b6 v[106:108], v9 offset:21120 // (op 287) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v9 offset:22144 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1938) 


ds_read_u8 v14, v10 offset:37504 // (op 291) Load local data 



ds_read_b96_tr_b6 v[124:126], v9 offset:21152 // (op 470) Transpose load local data 
ds_read_b96_tr_b6 v[127:129], v9 offset:22176 // (op 470) Transpose load local data 


ds_read_u8 v73, v10 offset:37536 // (op 474) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1919) 



ds_read_b96_tr_b6 v[118:120], v3 offset:38016 // (op 285) Transpose load local data 
ds_read_b96_tr_b6 v[121:123], v3 offset:38272 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1932) 


ds_read_u8 v22, v6 offset:42112 // (op 289) Load local data 

s_mul_i32 s7, s25, 6 // (op 305) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 305) least significant
s_mul_hi_u32 s1, s24, 6 // (op 305) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 305) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[6:7] // (op 305) 


s_lshl_b64 s[6:7], s[24:25], 8 // (op 305) 
s_mul_i32 s15, s7, 6 // (op 305) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 305) least significant
s_mul_hi_u32 s1, s6, 6 // (op 305) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 305) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 305) 

s_mul_i32 s1, 3, s6 // (op 305) 

v_add_u32 v85, v33, s1 // (op 305) 

buffer_load_dwordx3 v[66:68], v85, s[48:51], 0 offen // (op 305) Load value
buffer_load_dwordx3 v[70:72], v85, s[48:51], 0 offen offset:12 // (op 305) Load value
buffer_load_dwordx3 v[74:76], v85, s[48:51], 0 offen offset:24 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[118:123], v[106:111], a[0:3], v22, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 292) 
buffer_load_dwordx3 v[78:80], v85, s[48:51], 0 offen offset:36 // (op 305) Load value
v_add_u32 v85, v85, v24 // (op 305) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:123], v[124:129], a[4:7], v22, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 457) 
buffer_load_dwordx3 v[82:84], v85, s[48:51], 0 offen // (op 305) Load value


buffer_load_dwordx3 v[86:88], v85, s[48:51], 0 offen offset:12 // (op 305) Load value
ds_read_b96_tr_b6 v[124:126], v3 offset:40064 // (op 294) Transpose load local data 
ds_read_b96_tr_b6 v[127:129], v3 offset:40320 // (op 294) Transpose load local data 
buffer_load_dwordx3 v[90:92], v85, s[48:51], 0 offen offset:24 // (op 305) Load value
buffer_load_dwordx3 v[94:96], v85, s[48:51], 0 offen offset:36 // (op 305) Load value






s_barrier  // (op 1986) 


s_lshl_b32 s1, s34, 3 // (op 307) 
s_mul_i32 s6, 3, s1 // (op 307) 

v_add_u32 v33, v61, s6 // (op 307) 


buffer_load_dword v18, v33, s[56:59], 0 offen // (op 307) Load value

s_lshl_b64 s[6:7], s[24:25], 8 // (op 1280) 

s_mul_i32 s15, s7, 6 // (op 1280) most significant: low of high * low

s_mul_i32 s14, s6, 6 // (op 1280) least significant
s_mul_hi_u32 s1, s6, 6 // (op 1280) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 1280) most significant: sum


s_lshl_b32 s1, s34, 4 // (op 1408) 
v_add_u32 v23, v23, s1 // (op 1408) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 1280) 

s_lshl_b32 s1, s6, 1 // (op 1280) 

v_add_u32 v17, v17, s1 // (op 1280) 

s_mul_i32 s7, s13, 6 // (op 282) most significant: low of high * low
s_mul_i32 s6, s12, 6 // (op 282) least significant
s_mul_hi_u32 s1, s12, 6 // (op 282) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 282) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[6:7] // (op 282) 


s_lshl_b64 s[6:7], s[12:13], 8 // (op 282) 
s_mul_i32 s15, s7, 6 // (op 282) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 282) least significant
s_mul_hi_u32 s1, s6, 6 // (op 282) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 282) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 282) 

s_mul_i32 s1, 3, s6 // (op 282) 

v_add_u32 v33, v77, s1 // (op 282) 

ds_read_b96_tr_b6 v[130:132], v9 offset:29312 // (op 296) Transpose load local data 
ds_read_b96_tr_b6 v[133:135], v9 offset:30336 // (op 296) Transpose load local data 
buffer_load_dwordx3 v[98:100], v33, s[8:11], 0 offen // (op 282) Load value


v_add_u32 v33, v33, v24 // (op 282) 
ds_read_u8 v61, v6 offset:42176 // (op 298) Load local data 
buffer_load_dwordx3 v[102:104], v33, s[8:11], 0 offen // (op 282) Load value







s_barrier  // (op 1990) 
ds_read_u8 v33, v10 offset:37760 // (op 300) Load local data 
s_lshl_b32 s1, s28, 3 // (op 306) 



s_mul_i32 s6, 3, s1 // (op 306) 

v_add_u32 v73, v81, s6 // (op 306) 


buffer_load_ubyte v16, v73, s[52:55], 0 offen offset: 0 // (op 306) Load value

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

ds_read_b96_tr_b6 v[136:138], v9 offset:29344 // (op 461) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:129], v[130:135], a[0:3], v61, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 301) 
ds_read_b96_tr_b6 v[139:141], v9 offset:30368 // (op 461) Transpose load local data 


s_waitcnt vmcnt(23)
ds_write_b96 v41, v[26:28] // (op 13) Store local data 


s_barrier  // (op 2026) 
ds_read_u8 v33, v10 offset:37792 // (op 465) Load local data 
s_waitcnt vmcnt(22)
ds_write_b96 v41, v[30:32] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v41, v[34:36] offset:32 // (op 13) Store local data 
s_barrier  // (op 2030) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:129], v[136:141], a[4:7], v61, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 466) 
s_waitcnt vmcnt(20)
ds_write_b96 v41, v[38:40] offset:48 // (op 13) Store local data 




s_waitcnt vmcnt(19)
ds_write_b96 v41, v[42:44] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b96 v41, v[46:48] offset:80 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b96 v41, v[50:52] offset:96 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v41, v[54:56] offset:112 // (op 13) Store local data 
s_barrier  // (op 1902) 



s_waitcnt vmcnt(15)
ds_write_b32 v53, v21 offset:16384 // (op 30) Store local data 
s_barrier  // (op 1842) 



s_waitcnt vmcnt(14)
ds_write_b96 v37, v[58:60] offset:16896 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b96 v37, v[62:64] offset:16912 // (op 6) Store local data 
s_barrier  // (op 1882) 



s_waitcnt vmcnt(12)
ds_write_b8 v49, v29 offset:20992 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1871) 



ds_read_b96_tr_b6 v[106:108], v5 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v5 offset:1024 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1912) 


ds_read_u8 v14, v7 offset:16384 // (op 85) Load local data 



ds_read_b96_tr_b6 v[112:114], v5 offset:32 // (op 442) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v5 offset:1056 // (op 442) Transpose load local data 


ds_read_u8 v19, v7 offset:16416 // (op 446) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1851) 



ds_read_b96_tr_b6 v[118:120], v0 offset:16896 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[121:123], v0 offset:17152 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1892) 


ds_read_u8 v22, v4 offset:20992 // (op 80) Load local data 
s_ashr_i64 s[6:7], s[22:23], 8 // (op 399) 
s_mov_b32 s1, s6 // (op 399) convert

s_and_b32 s6, s1, 4294967294 // (op 399) 

s_cmp_lt_u32 s0, s6 // (op 399) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 // (op 399) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoop_399 if true)
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoop_399:

















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
v_add_co_u32 v64, s[0:1], v15, v20 // (op 1178) least significant half
v_addc_co_u32 v65, s[0:1], v17, v21, s[0:1] // (op 1178) most significant half





v_mul_lo_u32 v21, v65, 6 // (op 1178) most significant: low of high * low
v_mul_hi_u32 v15, v64, 6 // (op 1178) most significant: high of low * low
v_mul_lo_u32 v20, v64, 6 // (op 1178) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1178) most significant: sum


v_lshrrev_b64 v[64:65], 3, v[20:21] // (op 1178) 

v_mov_b32 v15, v64 // (op 1178) convert

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

v_lshlrev_b32 v17, 1, v1 // (op 1246) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1246) 
s_lshr_b32 s6, s0, 1 // (op 1246) 

v_lshl_add_u32 v69, s6, 9, v17 // (op 1246) 


s_lshl_b32 s0, s17, 6 // (op 1246) 
s_mov_b32 s7, 0 // (op 1246) convert
s_mov_b32 s6, s0 // (op 1246) convert

v_mul_lo_u32 v21, v69, s25 // (op 1246) most significant: low of low * high
v_mul_hi_u32 v17, v69, s24 // (op 1246) most significant: high of low * low
v_mul_lo_u32 v20, v69, s24 // (op 1246) least significant: low of low * low
v_add_u32 v21, v21, v17 // (op 1246) most significant: sum

v_mov_b32 v17, s6 // (op 1246) 
v_mov_b32 v73, s7 // (op 1246) 
v_add_co_u32 v20, s[0:1], v20, v17 // (op 1246) least significant half
v_addc_co_u32 v21, s[0:1], v21, v73, s[0:1] // (op 1246) most significant half





v_mul_lo_u32 v65, v21, 6 // (op 1246) most significant: low of high * low
v_mul_hi_u32 v17, v20, 6 // (op 1246) most significant: high of low * low
v_mul_lo_u32 v64, v20, 6 // (op 1246) least significant: low of low * low
v_add_u32 v65, v65, v17 // (op 1246) most significant: sum


v_lshrrev_b64 v[20:21], 3, v[64:65] // (op 1246) 

v_mov_b32 v17, v20 // (op 1246) convert

s_mul_i32 s1, s23, 6 // (op 1248) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 1248) least significant
s_mul_hi_u32 s6, s22, 6 // (op 1248) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1248) most significant: sum

s_mul_i32 s7, s24, s1 // (op 1248) most significant: low of low * high
s_mul_i32 s14, s25, s0 // (op 1248) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 1248) least significant
s_mul_hi_u32 s15, s24, s0 // (op 1248) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 1248) most significant: sum
s_add_u32 s7, s7, s15 // (op 1248) most significant: sum



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
v_lshl_add_u32 v77, s6, 4, v73 // (op 1310) 


v_mul_lo_u32 v73, v77, s28 // (op 1310) 

v_add_u32 v77, v69, v73 // (op 1310) 


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

v_lshrrev_b32 v69, 4, v1 // (op 1374) 
s_ashr_i64 s[0:1], s[22:23], 8 // (op 1374) 
v_and_b32 v73, -16, v1 // (op 1374) 
s_lshr_b32 s6, s0, 1 // (op 1374) 

v_sub_u32 v81, v1, v73 // (op 1374) 

v_lshl_add_u32 v73, s6, 4, v69 // (op 1374) 


v_lshlrev_b32 v69, 2, v81 // (op 1374) 

v_lshl_add_u32 v81, s17, 6, v69 // (op 1374) 

v_mul_lo_u32 v69, v73, s34 // (op 1374) 
v_add_u32 v69, v69, v81 // (op 1374) 


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

v_lshlrev_b32 v73, 5, v1 // (op 1467) 
v_mul_lo_u32 v81, v73, 6 // (op 1467) 
v_lshrrev_b32 v85, 3, v81 // (op 1467) 

v_lshrrev_b32 v81, 4, v73 // (op 1467) 

v_lshlrev_b32 v73, 2, v81 // (op 1467) 

v_add_u32 v81, v85, v73 // (op 1467) 


v_lshlrev_b32 v73, 7, v1 // (op 1520) 
v_mul_lo_u32 v85, v73, 6 // (op 1520) 
v_lshrrev_b32 v89, 3, v85 // (op 1520) 

v_lshrrev_b32 v85, 4, v73 // (op 1520) 

v_lshlrev_b32 v73, 2, v85 // (op 1520) 

v_add_u32 v85, v89, v73 // (op 1520) 


v_and_b32 v73, -16, v1 // (op 1578) 
v_sub_u32 v89, v1, v73 // (op 1578) 

v_and_b32 v73, 4294967280, v1 // (op 1578) 
v_add_u32 v93, v73, v89 // (op 1578) 


v_and_b32 v73, -16, v1 // (op 1631) 
v_sub_u32 v89, v1, v73 // (op 1631) 

v_lshlrev_b32 v73, 2, v89 // (op 1631) 

v_lshrrev_b32 v89, 4, v1 // (op 1631) 
v_lshl_add_u32 v97, v89, 6, v73 // (op 1631) 


s_ashr_i64 s[0:1], s[22:23], 8 // (op 405) 
s_and_b32 s6, s0, 4294967294 // (op 405) 

s_ashr_i64 s[0:1], s[22:23], 8 // (op 404) 
s_mov_b32 s7, s0 // (op 404) convert

s_cmp_lt_u32 s6, s7 // (op 404) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 // (op 404) Condition: Top (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404:

s_add_u32 s6, s6, 1 // (op 406) 
s_mul_i32 s1, s13, 6 // (op 198) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 198) least significant
s_mul_hi_u32 s7, s12, 6 // (op 198) most significant: high of low * low
s_add_u32 s1, s1, s7 // (op 198) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 198) 


v_mov_b32 v14, v15 // (op 198) 
v_mov_b32 v16, v14 // (op 198) reg expression
buffer_load_dwordx3 v[20:22], v16, s[8:11], 0 offen // (op 198) Load value
v_add_u32 v16, v16, v18 // (op 198) 
buffer_load_dwordx3 v[24:26], v16, s[8:11], 0 offen // (op 198) Load value



s_lshl_b64 s[0:1], s[12:13], 8 // (op 1168) 
s_mul_i32 s15, s1, 6 // (op 1168) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 1168) least significant
s_mul_hi_u32 s7, s0, 6 // (op 1168) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 1168) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 1168) 

v_add_u32 v15, v15, s0 // (op 1168) 


s_barrier  // (op 1845) 
s_mul_i32 s1, s25, 6 // (op 221) most significant: low of high * low



s_mul_i32 s0, s24, 6 // (op 221) least significant
s_mul_hi_u32 s7, s24, 6 // (op 221) most significant: high of low * low
s_add_u32 s1, s1, s7 // (op 221) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 221) 


v_mov_b32 v14, v17 // (op 221) 
v_mov_b32 v16, v14 // (op 221) reg expression
buffer_load_dwordx3 v[28:30], v16, s[48:51], 0 offen // (op 221) Load value
s_waitcnt vmcnt(2)
ds_write_b96 v81, v[20:22] offset:16896 // (op 199) Store local data 
buffer_load_dwordx3 v[32:34], v16, s[48:51], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[36:38], v16, s[48:51], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[40:42], v16, s[48:51], 0 offen offset:36 // (op 221) Load value
v_add_u32 v16, v16, v18 // (op 221) 
s_waitcnt vmcnt(4)
ds_write_b96 v81, v[24:26] offset:16912 // (op 199) Store local data 
buffer_load_dwordx3 v[44:46], v16, s[48:51], 0 offen // (op 221) Load value

buffer_load_dwordx3 v[48:50], v16, s[48:51], 0 offen offset:12 // (op 221) Load value
buffer_load_dwordx3 v[52:54], v16, s[48:51], 0 offen offset:24 // (op 221) Load value
buffer_load_dwordx3 v[56:58], v16, s[48:51], 0 offen offset:36 // (op 221) Load value



s_lshl_b64 s[0:1], s[24:25], 8 // (op 1238) 
s_mul_i32 s15, s1, 6 // (op 1238) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 1238) least significant
s_mul_hi_u32 s7, s0, 6 // (op 1238) most significant: high of low * low
s_add_u32 s15, s15, s7 // (op 1238) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 1238) 

v_add_u32 v17, v17, s0 // (op 1238) 



s_barrier  // (op 1864) 
v_mov_b32 v14, v77 // (op 222) 



v_mov_b32 v16, v14 // (op 222) reg expression

buffer_load_ubyte v31, v16, s[52:55], 0 offen offset: 0 // (op 222) Load value


s_lshl_b32 s7, s28, 3 // (op 1302) 
v_add_u32 v77, v77, s7 // (op 1302) 

s_waitcnt vmcnt(8)
ds_write_b96 v85, v[28:30] // (op 218) Store local data 


v_mov_b32 v14, v69 // (op 223) 
v_mov_b32 v16, v14 // (op 223) reg expression

buffer_load_dword v35, v16, s[56:59], 0 offen // (op 223) Load value


s_lshl_b32 s7, s34, 3 // (op 1366) 
v_add_u32 v69, v69, s7 // (op 1366) 

s_waitcnt vmcnt(8)
ds_write_b96 v85, v[32:34] offset:16 // (op 218) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1854) 
s_waitcnt vmcnt(7)
ds_write_b96 v85, v[36:38] offset:32 // (op 218) Store local data 



ds_read_b96_tr_b6 v[18:20], v0 offset:16896 // (op 201) Transpose load local data 
ds_read_b96_tr_b6 v[21:23], v0 offset:17152 // (op 201) Transpose load local data 
s_waitcnt vmcnt(6)
ds_write_b96 v85, v[40:42] offset:48 // (op 218) Store local data 
s_waitcnt vmcnt(5)
ds_write_b96 v85, v[44:46] offset:64 // (op 218) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v85, v[48:50] offset:80 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v85, v[52:54] offset:96 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v85, v[56:58] offset:112 // (op 218) Store local data 

s_barrier  // (op 1885) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1875) 






ds_read_b96_tr_b6 v[24:26], v5 // (op 203) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v5 offset:1024 // (op 203) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v93, v31 offset:20992 // (op 219) Store local data 

s_barrier  // (op 1905) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1895) 





ds_read_u8 v14, v4 offset:20992 // (op 205) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v97, v35 offset:16384 // (op 220) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1915) 


ds_read_u8 v16, v7 offset:16384 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:29], a[0:3], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 208) 





ds_read_b96_tr_b6 v[24:26], v0 offset:18944 // (op 210) Transpose load local data 
ds_read_b96_tr_b6 v[27:29], v0 offset:19200 // (op 210) Transpose load local data 



ds_read_b96_tr_b6 v[30:32], v5 offset:8192 // (op 212) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v5 offset:9216 // (op 212) Transpose load local data 


ds_read_u8 v16, v4 offset:21056 // (op 214) Load local data 


ds_read_u8 v36, v7 offset:16640 // (op 216) Load local data 



ds_read_b96_tr_b6 v[70:72], v5 offset:32 // (op 412) Transpose load local data 
ds_read_b96_tr_b6 v[73:75], v5 offset:1056 // (op 412) Transpose load local data 


ds_read_u8 v76, v7 offset:16416 // (op 416) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:23], v[70:75], a[4:7], v14, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 417) 







v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], a[0:3], v16, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 217) 
ds_read_b96_tr_b6 v[18:20], v5 offset:8224 // (op 421) Transpose load local data 


ds_read_b96_tr_b6 v[21:23], v5 offset:9248 // (op 421) Transpose load local data 


ds_read_u8 v14, v7 offset:16672 // (op 425) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:29], v[18:23], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b010 // (op 426) 




s_ashr_i64 s[0:1], s[22:23], 8 // (op 404) 
s_mov_b32 s7, s0 // (op 404) convert

s_cmp_lt_u32 s6, s7 // (op 404) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 // (op 404) Condition: Bottom (jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopTop_KLoopTail_404 if true)
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ForLoopBottom_KLoopTail_404:























s_barrier  // (op 2012) 








v_cmp_eq_f32 vcc, 0.00000, v12 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155
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
s_branch RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalFalse_0_beta_155:


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
RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_ConditionalBottom_0_beta_155:










s_endpgm  // End of RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.LRR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end:
.size RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2, .LRR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2_end-RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
  .amdhsa_next_free_vgpr 152
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 42240 // lds bytes
  .amdhsa_accum_offset 144
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
  - .name: RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2
    .symbol: RR_GEMM_NT_BF6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 42240
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 142
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

