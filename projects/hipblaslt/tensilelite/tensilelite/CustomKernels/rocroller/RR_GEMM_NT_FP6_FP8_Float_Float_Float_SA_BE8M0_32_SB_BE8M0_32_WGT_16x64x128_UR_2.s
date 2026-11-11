// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2,@function
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 1208) 
v_and_b32 v3, 48, v1 // (op 1208) 
v_sub_u32 v4, v0, v3 // (op 1208) 

v_lshrrev_b32 v3, 4, v0 // (op 1208) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1208) 


v_lshrrev_b32 v3, 7, v1 // (op 1208) 
v_lshlrev_b32 v4, 4, v3 // (op 1208) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1208) 


v_mul_lo_u32 v0, v3, 6 // (op 1208) 
v_lshrrev_b32 v4, 3, v0 // (op 1208) 

v_lshrrev_b32 v0, 4, v3 // (op 1208) 

v_lshlrev_b32 v3, 2, v0 // (op 1208) 

v_add_u32 v0, v4, v3 // (op 1208) 


v_and_b32 v3, 63, v1 // (op 1365) 
v_and_b32 v4, 48, v1 // (op 1365) 
v_sub_u32 v5, v3, v4 // (op 1365) 

v_lshrrev_b32 v4, 4, v3 // (op 1365) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 1365) 


v_lshrrev_b32 v4, 7, v1 // (op 1365) 
v_lshlrev_b32 v5, 4, v4 // (op 1365) 

v_lshl_add_u32 v4, v3, 4, v5 // (op 1365) 


v_mul_lo_u32 v3, v4, 6 // (op 1365) 
v_lshrrev_b32 v5, 3, v3 // (op 1365) 

v_lshrrev_b32 v3, 4, v4 // (op 1365) 

v_lshlrev_b32 v4, 2, v3 // (op 1365) 

v_add_u32 v3, v5, v4 // (op 1365) 


v_lshrrev_b32 v4, 7, v1 // (op 1151) 
v_and_b32 v5, 63, v1 // (op 1151) 
v_lshrrev_b32 v6, 4, v5 // (op 1151) 
v_lshlrev_b32 v7, 2, v6 // (op 1151) 

v_lshl_add_u32 v6, v4, 4, v7 // (op 1151) 


v_lshl_add_u32 v4, s16, 4, v6 // (op 1151) 

v_lshrrev_b32 v6, 2, v5 // (op 1151) 

v_and_b32 v5, 3, v6 // (op 1151) 

v_and_b32 v6, 3, v1 // (op 1151) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1151) 


v_lshrrev_b32 v5, 6, v1 // (op 1151) 
v_and_b32 v6, 1, v5 // (op 1151) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1151) 


v_lshl_add_u32 v6, s17, 6, v5 // (op 1151) 

v_mul_lo_u32 v5, v6, s2 // (op 1151) 

v_add_lshl_u32 v6, v4, v5, 2 // (op 1151) 


v_and_b32 v4, 48, v1 // (op 1265) 
v_and_b32 v5, 63, v1 // (op 1265) 
v_sub_u32 v7, v5, v4 // (op 1265) 

v_lshrrev_b32 v5, 1, v7 // (op 1265) 
s_mul_i32 s1, s2, s9 // (op 1153) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1153) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1153) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1153) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1153) most significant: sum
s_add_u32 s1, s1, s19 // (op 1153) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1153) 

v_add_lshl_u32 v8, v4, v5, 6 // (op 1265) 


v_and_b32 v4, 1, v7 // (op 1265) 

v_lshlrev_b32 v5, 3, v4 // (op 1265) 

v_lshrrev_b32 v4, 6, v1 // (op 1265) 
v_and_b32 v7, 1, v4 // (op 1265) 

v_lshl_add_u32 v4, v7, 4, v5 // (op 1265) 


v_add_u32 v5, v8, v4 // (op 1265) 


v_lshrrev_b32 v4, 6, v1 // (op 1323) 
v_and_b32 v7, 1, v4 // (op 1323) 

v_and_b32 v4, 15, v1 // (op 1323) 
v_lshl_add_u32 v8, v7, 4, v4 // (op 1323) 


v_and_b32 v4, 63, v1 // (op 1323) 
v_lshrrev_b32 v7, 4, v4 // (op 1323) 

v_lshl_add_u32 v4, v7, 6, v8 // (op 1323) 


v_lshrrev_b32 v7, 7, v1 // (op 1339) 
v_and_b32 v8, 63, v1 // (op 1339) 
v_lshrrev_b32 v9, 4, v8 // (op 1339) 
v_lshlrev_b32 v10, 2, v9 // (op 1339) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1339) 


v_lshl_add_u32 v7, s16, 4, v9 // (op 1339) 

v_lshrrev_b32 v9, 2, v8 // (op 1339) 

v_and_b32 v8, 3, v9 // (op 1339) 

v_and_b32 v9, 3, v1 // (op 1339) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1339) 


v_lshrrev_b32 v8, 6, v1 // (op 1339) 
v_and_b32 v9, 1, v8 // (op 1339) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1339) 


v_lshl_add_u32 v9, s17, 6, v8 // (op 1339) 

v_mul_lo_u32 v8, v9, s4 // (op 1339) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 1339) 


v_and_b32 v7, 48, v1 // (op 1409) 
v_and_b32 v8, 63, v1 // (op 1409) 
v_sub_u32 v10, v8, v7 // (op 1409) 

v_lshrrev_b32 v8, 1, v10 // (op 1409) 
s_mul_i32 s1, s4, s9 // (op 1341) most significant: low of low * high
v_add_lshl_u32 v11, v7, v8, 6 // (op 1409) 


v_and_b32 v7, 1, v10 // (op 1409) 

v_lshlrev_b32 v8, 3, v7 // (op 1409) 

v_lshrrev_b32 v7, 6, v1 // (op 1409) 
v_and_b32 v10, 1, v7 // (op 1409) 

v_lshl_add_u32 v7, v10, 4, v8 // (op 1409) 


v_add_u32 v8, v11, v7 // (op 1409) 


v_lshrrev_b32 v7, 6, v1 // (op 1438) 
v_and_b32 v10, 1, v7 // (op 1438) 

v_and_b32 v7, 15, v1 // (op 1438) 
v_lshl_add_u32 v11, v10, 4, v7 // (op 1438) 


v_and_b32 v7, 63, v1 // (op 1438) 
v_lshrrev_b32 v10, 4, v7 // (op 1438) 

v_lshl_add_u32 v7, v10, 6, v11 // (op 1438) 


s_mul_i32 s38, s5, s8 // (op 1341) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1341) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1341) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1341) most significant: sum
s_add_u32 s1, s1, s39 // (op 1341) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 1341) 

s_mov_b32 s40, s6 // (op 1153) reg expression
s_mov_b32 s41, s7 // (op 1153) reg expression
s_mov_b32 s42, s18 // (op 1153) reg expression
s_mov_b32 s43, 131072 // (op 1153) call()

s_mov_b32 s44, s10 // (op 1341) reg expression

s_mov_b32 s45, s11 // (op 1341) reg expression
s_mov_b32 s46, s38 // (op 1341) reg expression
s_mov_b32 s47, 131072 // (op 1341) call()



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
s_lshl_b32 s0, s16, 4 // (op 918) 
v_mul_lo_u32 v13, v1, s13 // (op 918) most significant: low of low * high
s_mov_b32 s15, 0 // (op 918) convert
s_mov_b32 s14, s0 // (op 918) convert

v_mul_hi_u32 v14, v1, s12 // (op 918) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 918) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 918) most significant: sum

v_mov_b32 v14, s14 // (op 918) 
v_mov_b32 v15, s15 // (op 918) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 918) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 918) most significant half





v_mul_lo_u32 v13, v17, 6 // (op 918) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 918) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 918) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 918) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 918) 

v_mov_b32 v12, v14 // (op 918) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 920) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 920) least significant
s_mul_hi_u32 s6, s22, 6 // (op 920) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 920) most significant: sum

s_mul_i32 s15, s12, s1 // (op 920) most significant: low of low * high
s_mul_i32 s6, s13, s0 // (op 920) most significant: low of high * low
s_mul_i32 s14, s12, s0 // (op 920) least significant
s_mul_hi_u32 s7, s12, s0 // (op 920) most significant: high of low * low
s_add_u32 s15, s15, s6 // (op 920) most significant: sum
s_add_u32 s15, s15, s7 // (op 920) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 920) 

s_mov_b32 s8, s20 // (op 920) reg expression
s_mov_b32 s9, s21 // (op 920) reg expression
s_mov_b32 s10, s0 // (op 920) reg expression
s_mov_b32 s11, 131072 // (op 920) call()

s_lshl_b32 s0, s16, 4 // (op 937) 
v_mul_lo_u32 v15, v1, s13 // (op 937) most significant: low of low * high
v_mul_hi_u32 v13, v1, s12 // (op 937) most significant: high of low * low
v_mul_lo_u32 v14, v1, s12 // (op 937) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 937) most significant: sum

s_mov_b32 s7, 0 // (op 937) convert
s_mov_b32 s6, s0 // (op 937) convert

v_mov_b32 v13, s6 // (op 937) 
v_mov_b32 v16, s7 // (op 937) 
v_add_co_u32 v18, s[0:1], v13, v14 // (op 937) least significant half
v_addc_co_u32 v19, s[0:1], v16, v15, s[0:1] // (op 937) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 937) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 937) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 937) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 937) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 937) 

v_mov_b32 v13, v14 // (op 937) convert

s_mul_i32 s1, s23, 6 // (op 939) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 939) least significant
s_mul_hi_u32 s6, s22, 6 // (op 939) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 939) most significant: sum

s_mul_i32 s7, s12, s1 // (op 939) most significant: low of low * high
s_mul_i32 s14, s13, s0 // (op 939) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 939) least significant
s_mul_hi_u32 s15, s12, s0 // (op 939) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 939) most significant: sum
s_add_u32 s7, s7, s15 // (op 939) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 939) 

s_mov_b32 s8, s20 // (op 939) reg expression
s_mov_b32 s9, s21 // (op 939) reg expression
s_mov_b32 s10, s0 // (op 939) reg expression
s_mov_b32 s11, 131072 // (op 939) call()

v_and_b32 v14, -4, v1 // (op 982) 
v_sub_u32 v15, v1, v14 // (op 982) 

v_lshlrev_b32 v14, 4, v15 // (op 982) 

v_lshl_add_u32 v15, s17, 6, v14 // (op 982) 

v_mov_b32 v17, 0 // (op 982) convert
v_mov_b32 v16, v15 // (op 982) convert

v_lshrrev_b32 v14, 2, v1 // (op 982) 
v_mul_lo_u32 v19, v14, s25 // (op 982) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 982) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 982) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 982) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 982) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 982) most significant half



s_mul_i32 s1, s24, s23 // (op 984) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 984) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 984) least significant
s_mul_hi_u32 s7, s24, s22 // (op 984) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 984) most significant: sum
s_add_u32 s1, s1, s7 // (op 984) most significant: sum


s_mov_b32 s48, s26 // (op 984) reg expression
s_mov_b32 s49, s27 // (op 984) reg expression
s_mov_b32 s50, s0 // (op 984) reg expression
s_mov_b32 s51, 131072 // (op 984) call()

v_and_b32 v14, -4, v1 // (op 1001) 
v_sub_u32 v15, v1, v14 // (op 1001) 

v_lshlrev_b32 v14, 4, v15 // (op 1001) 

v_lshl_add_u32 v15, s17, 6, v14 // (op 1001) 

v_mov_b32 v17, 0 // (op 1001) convert
v_mov_b32 v16, v15 // (op 1001) convert

v_lshrrev_b32 v14, 2, v1 // (op 1001) 
v_mul_lo_u32 v21, v14, s25 // (op 1001) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1001) most significant: high of low * low
v_mul_lo_u32 v20, v14, s24 // (op 1001) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1001) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 1001) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 1001) most significant half



s_mul_i32 s1, s24, s23 // (op 1003) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 1003) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 1003) least significant
s_mul_hi_u32 s7, s24, s22 // (op 1003) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1003) most significant: sum
s_add_u32 s1, s1, s7 // (op 1003) most significant: sum


s_mov_b32 s48, s26 // (op 1003) reg expression
s_mov_b32 s49, s27 // (op 1003) reg expression
s_mov_b32 s50, s0 // (op 1003) reg expression
s_mov_b32 s51, 131072 // (op 1003) call()

v_lshrrev_b32 v14, 7, v1 // (op 1046) 
v_and_b32 v15, 15, v1 // (op 1046) 
v_lshl_add_u32 v16, v14, 4, v15 // (op 1046) 


v_lshl_add_u32 v17, s16, 4, v16 // (op 1046) 

v_and_b32 v14, 63, v1 // (op 1046) 
v_lshrrev_b32 v15, 4, v14 // (op 1046) 

v_mul_lo_u32 v14, v15, s28 // (op 1046) 

v_add_u32 v15, v17, v14 // (op 1046) 


s_mul_i32 s1, s28, s33 // (op 1048) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1048) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1048) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1048) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1048) most significant: sum
s_add_u32 s1, s1, s7 // (op 1048) most significant: sum


s_mov_b32 s52, s30 // (op 1048) reg expression
s_mov_b32 s53, s31 // (op 1048) reg expression
s_mov_b32 s54, s0 // (op 1048) reg expression
s_mov_b32 s55, 131072 // (op 1048) call()

v_lshrrev_b32 v14, 7, v1 // (op 1065) 
v_and_b32 v16, 15, v1 // (op 1065) 
v_lshl_add_u32 v17, v14, 4, v16 // (op 1065) 


v_lshl_add_u32 v14, s16, 4, v17 // (op 1065) 

v_and_b32 v16, 63, v1 // (op 1065) 
v_lshrrev_b32 v17, 4, v16 // (op 1065) 

v_mul_lo_u32 v16, v17, s28 // (op 1065) 

v_add_u32 v17, v14, v16 // (op 1065) 


s_mul_i32 s1, s28, s33 // (op 1067) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1067) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1067) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1067) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1067) most significant: sum
s_add_u32 s1, s1, s7 // (op 1067) most significant: sum


s_mov_b32 s52, s30 // (op 1067) reg expression
s_mov_b32 s53, s31 // (op 1067) reg expression
s_mov_b32 s54, s0 // (op 1067) reg expression
s_mov_b32 s55, 131072 // (op 1067) call()

v_and_b32 v14, -64, v1 // (op 1110) 
v_sub_u32 v16, v1, v14 // (op 1110) 

v_lshl_add_u32 v14, s17, 6, v16 // (op 1110) 

v_lshrrev_b32 v16, 6, v1 // (op 1110) 
v_lshlrev_b32 v22, 1, v16 // (op 1110) 

v_mul_lo_u32 v16, v22, s34 // (op 1110) 
v_add_u32 v16, v16, v14 // (op 1110) 


s_mul_i32 s1, s34, s33 // (op 1112) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1112) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1112) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1112) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1112) most significant: sum
s_add_u32 s1, s1, s7 // (op 1112) most significant: sum


s_mov_b32 s56, s36 // (op 1112) reg expression
s_mov_b32 s57, s37 // (op 1112) reg expression
s_mov_b32 s58, s0 // (op 1112) reg expression
s_mov_b32 s59, 131072 // (op 1112) call()

v_and_b32 v14, -64, v1 // (op 1129) 
v_sub_u32 v22, v1, v14 // (op 1129) 

v_lshl_add_u32 v14, s17, 6, v22 // (op 1129) 

v_lshrrev_b32 v22, 6, v1 // (op 1129) 
v_lshlrev_b32 v23, 1, v22 // (op 1129) 

v_mul_lo_u32 v22, v23, s34 // (op 1129) 
v_add_u32 v22, v22, v14 // (op 1129) 


s_mul_i32 s1, s34, s33 // (op 1131) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1131) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1131) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1131) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1131) most significant: sum
s_add_u32 s1, s1, s7 // (op 1131) most significant: sum


s_mov_b32 s56, s36 // (op 1131) reg expression
s_mov_b32 s57, s37 // (op 1131) reg expression
s_mov_b32 s58, s0 // (op 1131) reg expression
s_mov_b32 s59, 131072 // (op 1131) call()


v_mov_b64 v[24:25], s[28:29] // (op 761) convert

v_mov_b32 v14, v15 // (op 761) reg expression
v_mov_b32 v23, v14 // (op 761) 
buffer_load_ubyte v26, v23, s[52:55], 0 offen offset: 0 // (op 761) Load value



v_and_b32 v14, -4, v1 // (op 1248) 
v_sub_u32 v23, v1, v14 // (op 1248) 

v_lshlrev_b32 v14, 4, v23 // (op 1248) 

v_lshrrev_b32 v23, 2, v1 // (op 1248) 
v_lshl_add_u32 v24, v23, 6, v14 // (op 1248) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 1248) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 873) 
v_mov_b64 v[24:25], s[6:7] // (op 873) convert


v_mov_b64 v[28:29], v[18:19] // (op 873) reg expression
s_mov_b32 m0, s0 // (op 873) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 873) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 873) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v24 // (op 873) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 873) most significant half

s_add_u32 m0, m0, 2048 // (op 873) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 873) Load value direct to lds




v_mov_b64 v[24:25], s[34:35] // (op 768) convert

v_mov_b32 v14, v16 // (op 768) reg expression
buffer_load_ubyte v28, v14, s[56:59], 0 offen offset: 0 // (op 768) Load value
v_add_u32 v14, v14, v24 // (op 768) 
buffer_load_ubyte v29, v14, s[56:59], 0 offen offset: 0 // (op 768) Load value




v_mov_b32 v14, v12 // (op 771) reg expression

buffer_load_dwordx3 v[30:32], v14, s[8:11], 0 offen // (op 771) Load value


v_mov_b64 v[24:25], s[28:29] // (op 775) convert

s_lshl_b32 s0, s28, 2 // (op 775) 
v_add_u32 v14, v15, s0 // (op 775) 

v_mov_b32 v23, v14 // (op 775) 
buffer_load_ubyte v27, v23, s[52:55], 0 offen offset: 0 // (op 775) Load value




v_and_b32 v14, -4, v1 // (op 1392) 
v_sub_u32 v15, v1, v14 // (op 1392) 

v_lshlrev_b32 v14, 4, v15 // (op 1392) 

v_lshrrev_b32 v15, 2, v1 // (op 1392) 
v_lshl_add_u32 v23, v15, 6, v14 // (op 1392) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1392) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 878) 
v_mov_b64 v[14:15], s[6:7] // (op 878) convert


s_lshl_b64 s[6:7], s[24:25], 7 // (op 878) 
v_mov_b32 v23, s6 // (op 878) 
v_mov_b32 v24, s7 // (op 878) 
v_add_co_u32 v34, s[14:15], v18, v23 // (op 878) least significant half
v_addc_co_u32 v35, s[14:15], v19, v24, s[14:15] // (op 878) most significant half




s_add_u32 s1, 8192, s0 // (op 878) 
s_mov_b32 m0, s1 // (op 878) reg expression
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v14 // (op 878) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v14 // (op 878) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v14 // (op 878) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 878) most significant half

s_add_u32 m0, m0, 2048 // (op 878) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds






v_mov_b64 v[14:15], s[34:35] // (op 782) convert

s_lshl_b32 s0, s34, 2 // (op 782) 
v_add_u32 v18, v16, s0 // (op 782) 

buffer_load_ubyte v24, v18, s[56:59], 0 offen offset: 0 // (op 782) Load value
v_add_u32 v18, v18, v14 // (op 782) 
buffer_load_ubyte v25, v18, s[56:59], 0 offen offset: 0 // (op 782) Load value





s_lshl_b64 s[0:1], s[12:13], 7 // (op 785) 
s_mul_i32 s7, s1, 6 // (op 785) most significant: low of high * low
s_mul_i32 s6, s0, 6 // (op 785) least significant
s_mul_hi_u32 s14, s0, 6 // (op 785) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 785) most significant: sum


s_lshr_b64 s[0:1], s[6:7], 3 // (op 785) 

v_add_u32 v23, v12, s0 // (op 785) 


buffer_load_dwordx3 v[14:16], v23, s[8:11], 0 offen // (op 785) Load value


v_and_b32 v12, -64, v1 // (op 1306) 
v_sub_u32 v23, v1, v12 // (op 1306) 

v_lshrrev_b32 v12, 6, v1 // (op 1306) 
v_lshl_add_u32 v18, v12, 7, v23 // (op 1306) 





s_waitcnt vmcnt(10)
ds_write_b8 v18, v28 offset:16384 // (op 788) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v18, v29 offset:16448 // (op 788) Store local data 

v_lshlrev_b32 v12, 4, v1 // (op 1196) 
v_mul_lo_u32 v23, v12, 6 // (op 1196) 

v_lshrrev_b32 v12, 3, v23 // (op 1196) 

v_and_b32 v23, 268435455, v1 // (op 1196) 
v_lshlrev_b32 v18, 2, v23 // (op 1196) 

v_add_u32 v19, v12, v18 // (op 1196) 





s_waitcnt vmcnt(8)
ds_write_b96 v19, v[30:32] offset:16640 // (op 789) Store local data 

v_lshlrev_b32 v12, 4, v1 // (op 1167) 
v_mul_lo_u32 v23, v12, 6 // (op 1167) 

v_lshrrev_b32 v12, 3, v23 // (op 1167) 

v_and_b32 v23, 268435455, v1 // (op 1167) 
v_lshlrev_b32 v18, 2, v23 // (op 1167) 

v_add_u32 v19, v12, v18 // (op 1167) 


v_and_b32 v12, -4, v1 // (op 1220) 
v_sub_u32 v18, v1, v12 // (op 1220) 

v_lshlrev_b32 v12, 4, v18 // (op 1220) 

v_lshrrev_b32 v18, 2, v1 // (op 1220) 
v_lshl_add_u32 v23, v18, 6, v12 // (op 1220) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1220) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1277) 
v_sub_u32 v18, v1, v12 // (op 1277) 

v_lshrrev_b32 v12, 6, v1 // (op 1277) 
v_lshl_add_u32 v23, v12, 7, v18 // (op 1277) 


v_lshlrev_b32 v12, 4, v1 // (op 1353) 
v_mul_lo_u32 v18, v12, 6 // (op 1353) 

v_lshrrev_b32 v12, 3, v18 // (op 1353) 

v_and_b32 v18, 268435455, v1 // (op 1353) 
v_lshlrev_b32 v33, 2, v18 // (op 1353) 

v_add_u32 v18, v12, v33 // (op 1353) 


v_and_b32 v12, -4, v1 // (op 1377) 
v_sub_u32 v33, v1, v12 // (op 1377) 

v_lshlrev_b32 v12, 4, v33 // (op 1377) 

v_lshrrev_b32 v33, 2, v1 // (op 1377) 
v_lshl_add_u32 v34, v33, 6, v12 // (op 1377) 


s_nop 0
v_readfirstlane_b32 s1, v34 // (op 1377) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1421) 
v_sub_u32 v33, v1, v12 // (op 1421) 

v_lshrrev_b32 v12, 6, v1 // (op 1421) 
v_lshl_add_u32 v34, v12, 7, v33 // (op 1421) 


s_mov_b32 s6, 0 // (op 308) call()
s_ashr_i64 s[14:15], s[22:23], 7 // (op 307) 
s_mov_b32 s7, s14 // (op 307) convert

s_and_b32 s14, s7, 4294967294 // (op 307) 

s_cmp_lt_u32 s6, s14 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_307 // (op 307) Condition: Top (jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_307 if false)
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_307:
s_add_u32 s6, s6, 2 // (op 309) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1479) 



ds_read_b96_tr_b6 v[36:38], v0 offset:16640 // (op 65) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v0 offset:16896 // (op 65) Transpose load local data 
s_barrier  // (op 1492) 



ds_read_b64_tr_b8 v[42:43], v5 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:512 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v5 offset:4096 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v5 offset:4608 // (op 70) Transpose load local data 
s_barrier  // (op 1505) 


ds_read_u8 v12, v4 offset:16384 // (op 79) Load local data 



ds_read_b64_tr_b8 v[50:51], v5 offset:32 // (op 339) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v5 offset:544 // (op 339) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:49], a[0:3], v26, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 82) 
ds_read_b64_tr_b8 v[54:55], v5 offset:4128 // (op 339) Transpose load local data 


ds_read_b64_tr_b8 v[56:57], v5 offset:4640 // (op 339) Transpose load local data 


ds_read_u8 v12, v4 offset:16416 // (op 342) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:41], v[50:57], a[4:7], v26, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 343) 




v_mov_b64 v[36:37], s[28:29] // (op 75) convert

v_mov_b32 v12, v17 // (op 75) 
s_lshl_b32 s7, s28, 3 // (op 75) 
v_add_u32 v33, v12, s7 // (op 75) 

v_mov_b32 v35, v33 // (op 75) 
buffer_load_ubyte v26, v35, s[52:55], 0 offen offset: 0 // (op 75) Load value



s_barrier  // (op 1495) 

s_lshl_b64 s[14:15], s[24:25], 5 // (op 856) 
v_mov_b64 v[36:37], s[14:15] // (op 856) convert


v_mov_b64 v[38:39], v[20:21] // (op 856) 
s_lshl_b64 s[14:15], s[24:25], 7 // (op 856) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 856) 

v_mov_b32 v33, s18 // (op 856) 
v_mov_b32 v35, s19 // (op 856) 
v_add_co_u32 v40, s[14:15], v38, v33 // (op 856) least significant half
v_addc_co_u32 v41, s[14:15], v39, v35, s[14:15] // (op 856) most significant half




s_mov_b32 m0, s0 // (op 856) reg expression
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 856) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 856) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 856) most significant half

s_add_u32 m0, m0, 2048 // (op 856) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 856) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 856) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 856) most significant half

s_add_u32 m0, m0, 2048 // (op 856) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 856) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 856) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 856) most significant half

s_add_u32 m0, m0, 2048 // (op 856) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 856) Load value direct to lds



v_mov_b64 v[36:37], s[34:35] // (op 21) convert

v_mov_b32 v33, v22 // (op 21) 
s_lshl_b32 s7, s34, 3 // (op 21) 
v_add_u32 v35, v33, s7 // (op 21) 

buffer_load_ubyte v28, v35, s[56:59], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v35, v35, v36 // (op 21) 
buffer_load_ubyte v29, v35, s[56:59], 0 offen offset: 0 // (op 21) Load value




v_mov_b32 v35, v13 // (op 4) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 4) 
s_mul_i32 s19, s15, 6 // (op 4) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 4) least significant
s_mul_hi_u32 s7, s14, 6 // (op 4) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 4) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s7, s14, 1 // (op 4) 

v_add_u32 v36, v35, s7 // (op 4) 


buffer_load_dwordx3 v[30:32], v36, s[8:11], 0 offen // (op 4) Load value

s_barrier  // (op 1534) 



s_waitcnt vmcnt(10)
ds_write_b8 v34, v24 offset:18688 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v34, v25 offset:18752 // (op 236) Store local data 
s_barrier  // (op 1522) 



s_waitcnt vmcnt(8)
ds_write_b96 v18, v[14:16] offset:18944 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1519) 



ds_read_b96_tr_b6 v[40:42], v3 offset:18944 // (op 227) Transpose load local data 
ds_read_b96_tr_b6 v[43:45], v3 offset:19200 // (op 227) Transpose load local data 
s_barrier  // (op 1525) 
s_barrier  // (op 1565) 



ds_read_b64_tr_b8 v[46:47], v8 offset:8192 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v8 offset:8704 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v8 offset:12288 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v8 offset:12800 // (op 229) Transpose load local data 
s_barrier  // (op 1531) 


ds_read_u8 v36, v7 offset:18688 // (op 233) Load local data 



ds_read_b64_tr_b8 v[54:55], v8 offset:8224 // (op 348) Transpose load local data 
ds_read_b64_tr_b8 v[56:57], v8 offset:8736 // (op 348) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:45], v[46:53], a[0:3], v27, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 234) 
ds_read_b64_tr_b8 v[58:59], v8 offset:12320 // (op 348) Transpose load local data 


ds_read_b64_tr_b8 v[60:61], v8 offset:12832 // (op 348) Transpose load local data 


ds_read_u8 v36, v7 offset:18720 // (op 351) Load local data 
s_barrier  // (op 1601) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:45], v[54:61], a[4:7], v27, v36 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 344) 




v_mov_b64 v[36:37], s[28:29] // (op 231) convert

s_lshl_b32 s7, s28, 2 // (op 231) 
s_mul_i32 s14, 3, s7 // (op 231) 

v_add_u32 v40, v12, s14 // (op 231) 

v_mov_b32 v41, v40 // (op 231) 
buffer_load_ubyte v27, v41, s[52:55], 0 offen offset: 0 // (op 231) Load value




s_barrier  // (op 1528) 
s_lshl_b32 s7, s28, 3 // (op 1057) 

v_add_u32 v17, v17, s7 // (op 1057) 

s_lshl_b64 s[14:15], s[24:25], 5 // (op 868) 
v_mov_b64 v[36:37], s[14:15] // (op 868) convert


s_lshl_b64 s[14:15], s[24:25], 7 // (op 868) 
s_mul_i32 s19, 3, s15 // (op 868) most significant: low of low * high
s_mul_i32 s18, 3, s14 // (op 868) least significant
s_mul_hi_u32 s7, 3, s14 // (op 868) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 868) most significant: sum


v_mov_b32 v12, s18 // (op 868) 
v_mov_b32 v40, s19 // (op 868) 
v_add_co_u32 v42, s[14:15], v38, v12 // (op 868) least significant half
v_addc_co_u32 v43, s[14:15], v39, v40, s[14:15] // (op 868) most significant half




s_add_u32 s7, 8192, s1 // (op 868) 
s_mov_b32 m0, s7 // (op 868) reg expression
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 868) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v36 // (op 868) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 868) most significant half

s_add_u32 m0, m0, 2048 // (op 868) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 868) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v36 // (op 868) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 868) most significant half

s_add_u32 m0, m0, 2048 // (op 868) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 868) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v36 // (op 868) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 868) most significant half

s_add_u32 m0, m0, 2048 // (op 868) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 868) Load value direct to lds





v_mov_b64 v[36:37], s[34:35] // (op 238) convert

s_lshl_b32 s7, s34, 2 // (op 238) 
s_mul_i32 s14, 3, s7 // (op 238) 

v_add_u32 v12, v33, s14 // (op 238) 

s_lshl_b64 s[14:15], s[24:25], 7 // (op 993) 
s_lshl_b64 s[18:19], s[14:15], 1 // (op 993) 

v_mov_b32 v38, s18 // (op 993) 
v_mov_b32 v39, s19 // (op 993) 
v_add_co_u32 v20, s[14:15], v20, v38 // (op 993) least significant half
v_addc_co_u32 v21, s[14:15], v21, v39, s[14:15] // (op 993) most significant half




buffer_load_ubyte v24, v12, s[56:59], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v12, v12, v36 // (op 238) 
buffer_load_ubyte v25, v12, s[56:59], 0 offen offset: 0 // (op 238) Load value





s_lshl_b32 s7, s34, 3 // (op 1121) 
v_add_u32 v22, v22, s7 // (op 1121) 

s_lshl_b64 s[14:15], s[12:13], 7 // (op 224) 
s_mul_i32 s19, s15, 6 // (op 224) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 224) least significant
s_mul_hi_u32 s7, s14, 6 // (op 224) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 224) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 224) 

s_mul_i32 s7, 3, s14 // (op 224) 

v_add_u32 v12, v35, s7 // (op 224) 


buffer_load_dwordx3 v[14:16], v12, s[8:11], 0 offen // (op 224) Load value


s_barrier  // (op 1508) 
s_lshl_b64 s[14:15], s[12:13], 7 // (op 929) 



s_mul_i32 s19, s15, 6 // (op 929) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 929) least significant
s_mul_hi_u32 s7, s14, 6 // (op 929) most significant: high of low * low
s_add_u32 s19, s19, s7 // (op 929) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 929) 

s_lshl_b32 s7, s14, 1 // (op 929) 

v_add_u32 v13, v13, s7 // (op 929) 

s_waitcnt vmcnt(10)
ds_write_b8 v23, v28 offset:16384 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v23, v29 offset:16448 // (op 26) Store local data 
s_barrier  // (op 1482) 



s_waitcnt vmcnt(8)
ds_write_b96 v19, v[30:32] offset:16640 // (op 6) Store local data 
s_ashr_i64 s[14:15], s[22:23], 7 // (op 307) 
s_mov_b32 s7, s14 // (op 307) convert

s_and_b32 s14, s7, 4294967294 // (op 307) 

s_cmp_lt_u32 s6, s14 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_307 // (op 307) Condition: Bottom (jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_307 if true)
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_307:














s_barrier  // (op 1583) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 891) 
s_lshr_b32 s6, s0, 1 // (op 891) 

v_lshl_add_u32 v26, s6, 8, v1 // (op 891) 

s_lshl_b32 s0, s16, 4 // (op 891) 
v_mul_lo_u32 v13, v26, s13 // (op 891) most significant: low of low * high
v_mul_hi_u32 v17, v26, s12 // (op 891) most significant: high of low * low
v_mul_lo_u32 v12, v26, s12 // (op 891) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 891) most significant: sum


s_mov_b32 s7, 0 // (op 891) convert
s_mov_b32 s6, s0 // (op 891) convert

v_mov_b32 v26, s6 // (op 891) 
v_mov_b32 v17, s7 // (op 891) 
v_add_co_u32 v18, s[0:1], v26, v12 // (op 891) least significant half
v_addc_co_u32 v19, s[0:1], v17, v13, s[0:1] // (op 891) most significant half





v_mul_lo_u32 v13, v19, 6 // (op 891) most significant: low of high * low
v_mul_hi_u32 v17, v18, 6 // (op 891) most significant: high of low * low
v_mul_lo_u32 v12, v18, 6 // (op 891) least significant: low of low * low
v_add_u32 v13, v13, v17 // (op 891) most significant: sum


v_lshrrev_b64 v[22:23], 3, v[12:13] // (op 891) 

v_mov_b32 v26, v22 // (op 891) convert

s_mul_i32 s1, s23, 6 // (op 893) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 893) least significant
s_mul_hi_u32 s6, s22, 6 // (op 893) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 893) most significant: sum

s_mul_i32 s7, s12, s1 // (op 893) most significant: low of low * high
s_mul_i32 s14, s13, s0 // (op 893) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 893) least significant
s_mul_hi_u32 s15, s12, s0 // (op 893) most significant: high of low * low
s_add_u32 s7, s7, s14 // (op 893) most significant: sum
s_add_u32 s7, s7, s15 // (op 893) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 893) 

s_mov_b32 s8, s20 // (op 893) reg expression
s_mov_b32 s9, s21 // (op 893) reg expression
s_mov_b32 s10, s0 // (op 893) reg expression
s_mov_b32 s11, 131072 // (op 893) call()

v_lshrrev_b32 v12, 2, v1 // (op 959) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 959) 
v_and_b32 v13, -4, v1 // (op 959) 
s_lshr_b32 s6, s0, 1 // (op 959) 

v_sub_u32 v17, v1, v13 // (op 959) 

v_lshl_add_u32 v13, s6, 8, v12 // (op 959) 


v_lshlrev_b32 v12, 4, v17 // (op 959) 

v_lshl_add_u32 v17, s17, 6, v12 // (op 959) 

v_mov_b32 v19, 0 // (op 959) convert
v_mov_b32 v18, v17 // (op 959) convert

v_mul_lo_u32 v21, v13, s25 // (op 959) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 959) most significant: high of low * low
v_mul_lo_u32 v20, v13, s24 // (op 959) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 959) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 959) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 959) most significant half



s_mul_i32 s1, s24, s23 // (op 961) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 961) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 961) least significant
s_mul_hi_u32 s7, s24, s22 // (op 961) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 961) most significant: sum
s_add_u32 s1, s1, s7 // (op 961) most significant: sum


s_mov_b32 s48, s26 // (op 961) reg expression
s_mov_b32 s49, s27 // (op 961) reg expression
s_mov_b32 s50, s0 // (op 961) reg expression
s_mov_b32 s51, 131072 // (op 961) call()

v_lshrrev_b32 v12, 7, v1 // (op 1023) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1023) 
v_and_b32 v13, 15, v1 // (op 1023) 
s_lshr_b32 s6, s0, 1 // (op 1023) 

v_lshl_add_u32 v17, v12, 4, v13 // (op 1023) 


v_lshl_add_u32 v12, s16, 4, v17 // (op 1023) 

v_and_b32 v13, 63, v1 // (op 1023) 
v_lshrrev_b32 v17, 4, v13 // (op 1023) 

v_lshl_add_u32 v13, s6, 3, v17 // (op 1023) 


v_mul_lo_u32 v17, v13, s28 // (op 1023) 

v_add_u32 v13, v12, v17 // (op 1023) 


s_mul_i32 s1, s28, s33 // (op 1025) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1025) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1025) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1025) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1025) most significant: sum
s_add_u32 s1, s1, s7 // (op 1025) most significant: sum


s_mov_b32 s52, s30 // (op 1025) reg expression
s_mov_b32 s53, s31 // (op 1025) reg expression
s_mov_b32 s54, s0 // (op 1025) reg expression
s_mov_b32 s55, 131072 // (op 1025) call()

v_lshrrev_b32 v12, 6, v1 // (op 1087) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1087) 
v_lshlrev_b32 v17, 1, v12 // (op 1087) 

s_lshr_b32 s6, s0, 1 // (op 1087) 

v_and_b32 v12, -64, v1 // (op 1087) 
v_lshl_add_u32 v18, s6, 3, v17 // (op 1087) 


v_sub_u32 v17, v1, v12 // (op 1087) 

v_lshl_add_u32 v12, s17, 6, v17 // (op 1087) 

v_mul_lo_u32 v17, v18, s34 // (op 1087) 
v_add_u32 v17, v17, v12 // (op 1087) 


s_mul_i32 s1, s34, s33 // (op 1089) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1089) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1089) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1089) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1089) most significant: sum
s_add_u32 s1, s1, s7 // (op 1089) most significant: sum


s_mov_b32 s56, s36 // (op 1089) reg expression
s_mov_b32 s57, s37 // (op 1089) reg expression
s_mov_b32 s58, s0 // (op 1089) reg expression
s_mov_b32 s59, 131072 // (op 1089) call()

v_lshlrev_b32 v12, 4, v1 // (op 1180) 
v_mul_lo_u32 v18, v12, 6 // (op 1180) 

v_lshrrev_b32 v12, 3, v18 // (op 1180) 

v_and_b32 v18, 268435455, v1 // (op 1180) 
v_lshlrev_b32 v19, 2, v18 // (op 1180) 

v_add_u32 v18, v12, v19 // (op 1180) 


v_and_b32 v12, -4, v1 // (op 1233) 
v_sub_u32 v19, v1, v12 // (op 1233) 

v_lshlrev_b32 v12, 4, v19 // (op 1233) 

v_lshrrev_b32 v19, 2, v1 // (op 1233) 
v_lshl_add_u32 v22, v19, 6, v12 // (op 1233) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 1233) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -64, v1 // (op 1290) 
v_sub_u32 v19, v1, v12 // (op 1290) 

v_lshrrev_b32 v12, 6, v1 // (op 1290) 
v_lshl_add_u32 v22, v12, 7, v19 // (op 1290) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 313) 
s_and_b32 s1, s6, 4294967294 // (op 313) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 312) 
s_mov_b32 s14, s6 // (op 312) convert

s_cmp_lt_u32 s1, s14 // (op 312) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_312 // (op 312) Condition: Top (jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_312 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_312:


s_add_u32 s1, s1, 1 // (op 314) 
v_mov_b32 v12, v26 // (op 166) 
v_mov_b32 v19, v12 // (op 166) reg expression

buffer_load_dwordx3 v[14:16], v19, s[8:11], 0 offen // (op 166) Load value


s_lshl_b64 s[6:7], s[12:13], 7 // (op 881) 
s_mul_i32 s15, s7, 6 // (op 881) most significant: low of high * low
s_mul_i32 s14, s6, 6 // (op 881) least significant
s_mul_hi_u32 s18, s6, 6 // (op 881) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 881) most significant: sum


s_lshr_b64 s[6:7], s[14:15], 3 // (op 881) 

v_add_u32 v26, v26, s6 // (op 881) 

s_barrier  // (op 1485) 
s_barrier  // (op 1498) 




s_lshl_b64 s[6:7], s[24:25], 5 // (op 861) 
v_mov_b64 v[24:25], s[6:7] // (op 861) convert


v_mov_b64 v[28:29], v[20:21] // (op 861) 
v_mov_b64 v[30:31], v[28:29] // (op 861) reg expression
s_mov_b32 m0, s0 // (op 861) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 861) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v24 // (op 861) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 861) most significant half

s_add_u32 m0, m0, 2048 // (op 861) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 861) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v24 // (op 861) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 861) most significant half

s_add_u32 m0, m0, 2048 // (op 861) 
s_waitcnt vmcnt(2)
ds_write_b96 v18, v[14:16] offset:16640 // (op 167) Store local data 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 861) Load value direct to lds

v_add_co_u32 v30, s[6:7], v30, v24 // (op 861) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 861) most significant half

s_add_u32 m0, m0, 2048 // (op 861) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 861) Load value direct to lds



s_lshl_b64 s[6:7], s[24:25], 7 // (op 951) 
v_mov_b32 v12, s6 // (op 951) 
v_mov_b32 v19, s7 // (op 951) 
v_add_co_u32 v20, s[14:15], v20, v12 // (op 951) least significant half
v_addc_co_u32 v21, s[14:15], v21, v19, s[14:15] // (op 951) most significant half





v_mov_b64 v[24:25], s[34:35] // (op 180) convert

v_mov_b32 v12, v17 // (op 180) 
v_mov_b32 v19, v12 // (op 180) reg expression
buffer_load_ubyte v14, v19, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v19, v19, v24 // (op 180) 
buffer_load_ubyte v15, v19, s[56:59], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s6, s34, 2 // (op 1079) 
v_add_u32 v17, v17, s6 // (op 1079) 

s_barrier  // (op 1511) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1488) 






ds_read_b96_tr_b6 v[28:30], v0 offset:16640 // (op 169) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v0 offset:16896 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v14 offset:16384 // (op 178) Store local data 
s_barrier  // (op 1501) 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v15 offset:16448 // (op 178) Store local data 




ds_read_b64_tr_b8 v[34:35], v5 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v5 offset:512 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:4096 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v5 offset:4608 // (op 171) Transpose load local data 

v_mov_b64 v[24:25], s[28:29] // (op 173) convert

v_mov_b32 v12, v13 // (op 173) 
v_mov_b32 v19, v12 // (op 173) reg expression
v_mov_b32 v23, v19 // (op 173) 
buffer_load_ubyte v27, v23, s[52:55], 0 offen offset: 0 // (op 173) Load value




s_lshl_b32 s6, s28, 2 // (op 1015) 
v_add_u32 v13, v13, s6 // (op 1015) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1515) 


ds_read_u8 v12, v4 offset:16384 // (op 175) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:33], v[34:41], a[0:3], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 176) 
ds_read_b64_tr_b8 v[42:43], v5 offset:32 // (op 320) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v5 offset:544 // (op 320) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v5 offset:4128 // (op 320) Transpose load local data 
ds_read_b64_tr_b8 v[48:49], v5 offset:4640 // (op 320) Transpose load local data 


ds_read_u8 v12, v4 offset:16416 // (op 323) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[28:33], v[42:49], a[4:7], v27, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b000 // (op 324) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 312) 
s_mov_b32 s14, s6 // (op 312) convert

s_cmp_lt_u32 s1, s14 // (op 312) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_312 // (op 312) Condition: Bottom (jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_312 if true)
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_312:






















s_barrier  // (op 1587) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123:


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
RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.LRR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end:
.size RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2, .LRR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end-RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 20992 // lds bytes
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F8
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
  - .name: RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
    .symbol: RR_GEMM_NT_FP6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 20992
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 62
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

