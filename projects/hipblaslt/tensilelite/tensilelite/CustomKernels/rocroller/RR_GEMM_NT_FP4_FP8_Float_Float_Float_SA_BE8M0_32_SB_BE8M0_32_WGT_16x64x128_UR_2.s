// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.p2align 8
.type RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2,@function
RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 1184) 
v_and_b32 v3, 48, v1 // (op 1184) 
v_sub_u32 v4, v0, v3 // (op 1184) 

v_lshrrev_b32 v3, 4, v0 // (op 1184) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1184) 


v_lshrrev_b32 v3, 7, v1 // (op 1184) 
v_lshlrev_b32 v4, 4, v3 // (op 1184) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1184) 


v_lshlrev_b32 v0, 2, v3 // (op 1184) 

v_lshrrev_b32 v3, 3, v0 // (op 1184) 

v_and_b32 v0, 63, v1 // (op 1342) 
v_and_b32 v4, 48, v1 // (op 1342) 
v_sub_u32 v5, v0, v4 // (op 1342) 

v_lshrrev_b32 v4, 4, v0 // (op 1342) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 1342) 


v_lshrrev_b32 v4, 7, v1 // (op 1342) 
v_lshlrev_b32 v5, 4, v4 // (op 1342) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1342) 


v_lshlrev_b32 v0, 2, v4 // (op 1342) 

v_lshrrev_b32 v4, 3, v0 // (op 1342) 

v_lshrrev_b32 v0, 7, v1 // (op 1127) 
v_and_b32 v5, 63, v1 // (op 1127) 
v_lshrrev_b32 v6, 4, v5 // (op 1127) 
v_lshlrev_b32 v7, 2, v6 // (op 1127) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1127) 


v_lshl_add_u32 v0, s16, 4, v6 // (op 1127) 

v_lshrrev_b32 v6, 2, v5 // (op 1127) 

v_and_b32 v5, 3, v6 // (op 1127) 

v_and_b32 v6, 3, v1 // (op 1127) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1127) 


v_lshrrev_b32 v5, 6, v1 // (op 1127) 
v_and_b32 v6, 1, v5 // (op 1127) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1127) 


v_lshl_add_u32 v6, s17, 6, v5 // (op 1127) 

v_mul_lo_u32 v5, v6, s2 // (op 1127) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 1127) 


v_and_b32 v0, 48, v1 // (op 1242) 
v_and_b32 v5, 63, v1 // (op 1242) 
v_sub_u32 v7, v5, v0 // (op 1242) 

v_lshrrev_b32 v5, 1, v7 // (op 1242) 
s_mul_i32 s1, s2, s9 // (op 1129) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1129) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1129) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1129) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1129) most significant: sum
s_add_u32 s1, s1, s19 // (op 1129) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1129) 

v_add_lshl_u32 v8, v0, v5, 6 // (op 1242) 


v_and_b32 v0, 1, v7 // (op 1242) 

v_lshlrev_b32 v5, 3, v0 // (op 1242) 

v_lshrrev_b32 v0, 6, v1 // (op 1242) 
v_and_b32 v7, 1, v0 // (op 1242) 

v_lshl_add_u32 v0, v7, 4, v5 // (op 1242) 


v_add_u32 v5, v8, v0 // (op 1242) 


v_lshrrev_b32 v0, 6, v1 // (op 1300) 
v_and_b32 v7, 1, v0 // (op 1300) 

v_and_b32 v0, 15, v1 // (op 1300) 
v_lshl_add_u32 v8, v7, 4, v0 // (op 1300) 


v_and_b32 v0, 63, v1 // (op 1300) 
v_lshrrev_b32 v7, 4, v0 // (op 1300) 

v_lshl_add_u32 v0, v7, 6, v8 // (op 1300) 


v_lshrrev_b32 v7, 7, v1 // (op 1316) 
v_and_b32 v8, 63, v1 // (op 1316) 
v_lshrrev_b32 v9, 4, v8 // (op 1316) 
v_lshlrev_b32 v10, 2, v9 // (op 1316) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1316) 


v_lshl_add_u32 v7, s16, 4, v9 // (op 1316) 

v_lshrrev_b32 v9, 2, v8 // (op 1316) 

v_and_b32 v8, 3, v9 // (op 1316) 

v_and_b32 v9, 3, v1 // (op 1316) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1316) 


v_lshrrev_b32 v8, 6, v1 // (op 1316) 
v_and_b32 v9, 1, v8 // (op 1316) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 1316) 


v_lshl_add_u32 v9, s17, 6, v8 // (op 1316) 

v_mul_lo_u32 v8, v9, s4 // (op 1316) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 1316) 


v_and_b32 v7, 48, v1 // (op 1371) 
v_and_b32 v8, 63, v1 // (op 1371) 
v_sub_u32 v10, v8, v7 // (op 1371) 

v_lshrrev_b32 v8, 1, v10 // (op 1371) 
s_mul_i32 s1, s4, s9 // (op 1318) most significant: low of low * high
v_add_lshl_u32 v11, v7, v8, 6 // (op 1371) 


v_and_b32 v7, 1, v10 // (op 1371) 

v_lshlrev_b32 v8, 3, v7 // (op 1371) 

v_lshrrev_b32 v7, 6, v1 // (op 1371) 
v_and_b32 v10, 1, v7 // (op 1371) 

v_lshl_add_u32 v7, v10, 4, v8 // (op 1371) 


v_add_u32 v8, v11, v7 // (op 1371) 


v_lshrrev_b32 v7, 6, v1 // (op 1400) 
v_and_b32 v10, 1, v7 // (op 1400) 

v_and_b32 v7, 15, v1 // (op 1400) 
v_lshl_add_u32 v11, v10, 4, v7 // (op 1400) 


v_and_b32 v7, 63, v1 // (op 1400) 
v_lshrrev_b32 v10, 4, v7 // (op 1400) 

v_lshl_add_u32 v7, v10, 6, v11 // (op 1400) 


s_mul_i32 s38, s5, s8 // (op 1318) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 1318) least significant
s_mul_hi_u32 s39, s4, s8 // (op 1318) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 1318) most significant: sum
s_add_u32 s1, s1, s39 // (op 1318) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 1318) 

s_mov_b32 s40, s6 // (op 1129) reg expression
s_mov_b32 s41, s7 // (op 1129) reg expression
s_mov_b32 s42, s18 // (op 1129) reg expression
s_mov_b32 s43, 131072 // (op 1129) call()

s_mov_b32 s44, s10 // (op 1318) reg expression

s_mov_b32 s45, s11 // (op 1318) reg expression
s_mov_b32 s46, s38 // (op 1318) reg expression
s_mov_b32 s47, 131072 // (op 1318) call()



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
s_lshl_b32 s0, s16, 4 // (op 894) 
v_mul_lo_u32 v13, v1, s13 // (op 894) most significant: low of low * high
s_mov_b32 s15, 0 // (op 894) convert
s_mov_b32 s14, s0 // (op 894) convert

v_mul_hi_u32 v14, v1, s12 // (op 894) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 894) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 894) most significant: sum

v_mov_b32 v14, s14 // (op 894) 
v_mov_b32 v15, s15 // (op 894) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 894) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 894) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 894) 


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 894) 

v_mov_b32 v14, v12 // (op 894) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 896) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 896) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 896) least significant
s_mul_hi_u32 s7, s12, s22 // (op 896) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 896) most significant: sum
s_add_u32 s1, s1, s7 // (op 896) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 896) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 896) 

s_mov_b32 s8, s20 // (op 896) reg expression
s_mov_b32 s9, s21 // (op 896) reg expression
s_mov_b32 s10, s0 // (op 896) reg expression
s_mov_b32 s11, 131072 // (op 896) call()

s_lshl_b32 s0, s16, 4 // (op 913) 
v_mul_lo_u32 v13, v1, s13 // (op 913) most significant: low of low * high
v_mul_hi_u32 v15, v1, s12 // (op 913) most significant: high of low * low
v_mul_lo_u32 v12, v1, s12 // (op 913) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 913) most significant: sum

s_mov_b32 s7, 0 // (op 913) convert
s_mov_b32 s6, s0 // (op 913) convert

v_mov_b32 v15, s6 // (op 913) 
v_mov_b32 v16, s7 // (op 913) 
v_add_co_u32 v18, s[0:1], v15, v12 // (op 913) least significant half
v_addc_co_u32 v19, s[0:1], v16, v13, s[0:1] // (op 913) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 913) 


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 913) 

v_mov_b32 v15, v12 // (op 913) convert

s_mul_i32 s1, s12, s23 // (op 915) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 915) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 915) least significant
s_mul_hi_u32 s7, s12, s22 // (op 915) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 915) most significant: sum
s_add_u32 s1, s1, s7 // (op 915) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 915) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 915) 

s_mov_b32 s8, s20 // (op 915) reg expression
s_mov_b32 s9, s21 // (op 915) reg expression
s_mov_b32 s10, s0 // (op 915) reg expression
s_mov_b32 s11, 131072 // (op 915) call()

v_and_b32 v12, -4, v1 // (op 958) 
v_sub_u32 v13, v1, v12 // (op 958) 

v_lshlrev_b32 v12, 4, v13 // (op 958) 

v_lshl_add_u32 v13, s17, 6, v12 // (op 958) 

v_and_b32 v12, 4294967292, v1 // (op 958) 
v_mul_lo_u32 v16, v12, s24 // (op 958) 
v_add_u32 v16, v16, v13 // (op 958) 


s_mul_i32 s1, s24, s23 // (op 960) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 960) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 960) least significant
s_mul_hi_u32 s7, s24, s22 // (op 960) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 960) most significant: sum
s_add_u32 s1, s1, s7 // (op 960) most significant: sum


s_mov_b32 s48, s26 // (op 960) reg expression
s_mov_b32 s49, s27 // (op 960) reg expression
s_mov_b32 s50, s0 // (op 960) reg expression
s_mov_b32 s51, 131072 // (op 960) call()

v_and_b32 v12, -4, v1 // (op 977) 
v_sub_u32 v13, v1, v12 // (op 977) 

v_lshlrev_b32 v12, 4, v13 // (op 977) 

v_lshl_add_u32 v13, s17, 6, v12 // (op 977) 

v_and_b32 v12, 4294967292, v1 // (op 977) 
v_mul_lo_u32 v17, v12, s24 // (op 977) 
v_add_u32 v17, v17, v13 // (op 977) 


s_mul_i32 s1, s24, s23 // (op 979) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 979) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 979) least significant
s_mul_hi_u32 s7, s24, s22 // (op 979) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 979) most significant: sum
s_add_u32 s1, s1, s7 // (op 979) most significant: sum


s_mov_b32 s48, s26 // (op 979) reg expression
s_mov_b32 s49, s27 // (op 979) reg expression
s_mov_b32 s50, s0 // (op 979) reg expression
s_mov_b32 s51, 131072 // (op 979) call()

v_lshrrev_b32 v12, 7, v1 // (op 1022) 
v_and_b32 v13, 15, v1 // (op 1022) 
v_lshl_add_u32 v18, v12, 4, v13 // (op 1022) 


v_lshl_add_u32 v12, s16, 4, v18 // (op 1022) 

v_and_b32 v13, 63, v1 // (op 1022) 
v_lshrrev_b32 v18, 4, v13 // (op 1022) 

v_mul_lo_u32 v13, v18, s28 // (op 1022) 

v_add_u32 v18, v12, v13 // (op 1022) 


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

v_lshrrev_b32 v12, 7, v1 // (op 1041) 
v_and_b32 v13, 15, v1 // (op 1041) 
v_lshl_add_u32 v19, v12, 4, v13 // (op 1041) 


v_lshl_add_u32 v12, s16, 4, v19 // (op 1041) 

v_and_b32 v13, 63, v1 // (op 1041) 
v_lshrrev_b32 v19, 4, v13 // (op 1041) 

v_mul_lo_u32 v13, v19, s28 // (op 1041) 

v_add_u32 v19, v12, v13 // (op 1041) 


s_mul_i32 s1, s28, s33 // (op 1043) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1043) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1043) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1043) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1043) most significant: sum
s_add_u32 s1, s1, s7 // (op 1043) most significant: sum


s_mov_b32 s52, s30 // (op 1043) reg expression
s_mov_b32 s53, s31 // (op 1043) reg expression
s_mov_b32 s54, s0 // (op 1043) reg expression
s_mov_b32 s55, 131072 // (op 1043) call()

v_and_b32 v12, -64, v1 // (op 1086) 
v_sub_u32 v13, v1, v12 // (op 1086) 

v_lshl_add_u32 v12, s17, 6, v13 // (op 1086) 

v_lshrrev_b32 v13, 6, v1 // (op 1086) 
v_lshlrev_b32 v20, 1, v13 // (op 1086) 

v_mul_lo_u32 v13, v20, s34 // (op 1086) 
v_add_u32 v13, v13, v12 // (op 1086) 


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

v_and_b32 v12, -64, v1 // (op 1105) 
v_sub_u32 v20, v1, v12 // (op 1105) 

v_lshl_add_u32 v12, s17, 6, v20 // (op 1105) 

v_lshrrev_b32 v20, 6, v1 // (op 1105) 
v_lshlrev_b32 v21, 1, v20 // (op 1105) 

v_mul_lo_u32 v20, v21, s34 // (op 1105) 
v_add_u32 v20, v20, v12 // (op 1105) 


s_mul_i32 s1, s34, s33 // (op 1107) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1107) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1107) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1107) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1107) most significant: sum
s_add_u32 s1, s1, s7 // (op 1107) most significant: sum


s_mov_b32 s56, s36 // (op 1107) reg expression
s_mov_b32 s57, s37 // (op 1107) reg expression
s_mov_b32 s58, s0 // (op 1107) reg expression
s_mov_b32 s59, 131072 // (op 1107) call()


v_mov_b64 v[22:23], s[28:29] // (op 761) convert

v_mov_b32 v12, v18 // (op 761) reg expression
v_mov_b32 v21, v12 // (op 761) 
buffer_load_ubyte v24, v21, s[52:55], 0 offen offset: 0 // (op 761) Load value




v_mov_b64 v[22:23], s[24:25] // (op 765) convert

v_mov_b32 v12, v16 // (op 765) reg expression
buffer_load_dwordx4 v[26:29], v12, s[48:51], 0 offen // (op 765) Load value
v_add_u32 v12, v12, v22 // (op 765) 
buffer_load_dwordx4 v[30:33], v12, s[48:51], 0 offen // (op 765) Load value
v_add_u32 v12, v12, v22 // (op 765) 
buffer_load_dwordx4 v[34:37], v12, s[48:51], 0 offen // (op 765) Load value
v_add_u32 v12, v12, v22 // (op 765) 
buffer_load_dwordx4 v[38:41], v12, s[48:51], 0 offen // (op 765) Load value



v_mov_b64 v[22:23], s[34:35] // (op 768) convert

v_mov_b32 v12, v13 // (op 768) reg expression
buffer_load_ubyte v42, v12, s[56:59], 0 offen offset: 0 // (op 768) Load value
v_add_u32 v12, v12, v22 // (op 768) 
buffer_load_ubyte v43, v12, s[56:59], 0 offen offset: 0 // (op 768) Load value




v_mov_b32 v12, v14 // (op 771) reg expression

buffer_load_dwordx2 v[22:23], v12, s[8:11], 0 offen // (op 771) Load value


v_mov_b64 v[44:45], s[28:29] // (op 775) convert

s_lshl_b32 s0, s28, 2 // (op 775) 
v_add_u32 v12, v18, s0 // (op 775) 

v_mov_b32 v21, v12 // (op 775) 
buffer_load_ubyte v25, v21, s[52:55], 0 offen offset: 0 // (op 775) Load value





v_mov_b64 v[44:45], s[24:25] // (op 779) convert

s_lshl_b32 s0, s24, 7 // (op 779) 
v_add_u32 v12, v16, s0 // (op 779) 

buffer_load_dwordx4 v[46:49], v12, s[48:51], 0 offen // (op 779) Load value
v_add_u32 v12, v12, v44 // (op 779) 
buffer_load_dwordx4 v[50:53], v12, s[48:51], 0 offen // (op 779) Load value
v_add_u32 v12, v12, v44 // (op 779) 
buffer_load_dwordx4 v[54:57], v12, s[48:51], 0 offen // (op 779) Load value
v_add_u32 v12, v12, v44 // (op 779) 
buffer_load_dwordx4 v[58:61], v12, s[48:51], 0 offen // (op 779) Load value




v_mov_b64 v[44:45], s[34:35] // (op 782) convert

s_lshl_b32 s0, s34, 2 // (op 782) 
v_add_u32 v12, v13, s0 // (op 782) 

buffer_load_ubyte v62, v12, s[56:59], 0 offen offset: 0 // (op 782) Load value
v_add_u32 v12, v12, v44 // (op 782) 
buffer_load_ubyte v63, v12, s[56:59], 0 offen offset: 0 // (op 782) Load value





s_lshl_b64 s[0:1], s[12:13], 7 // (op 785) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 785) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 785) 

v_add_u32 v16, v14, s0 // (op 785) 


buffer_load_dwordx2 v[12:13], v16, s[8:11], 0 offen // (op 785) Load value


v_and_b32 v14, -4, v1 // (op 1225) 
v_sub_u32 v16, v1, v14 // (op 1225) 

v_lshlrev_b32 v14, 4, v16 // (op 1225) 

v_lshrrev_b32 v16, 2, v1 // (op 1225) 
v_lshl_add_u32 v18, v16, 8, v14 // (op 1225) 





s_waitcnt vmcnt(14)
ds_write_b128 v18, v[26:29] // (op 787) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[30:33] offset:64 // (op 787) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[34:37] offset:128 // (op 787) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v18, v[38:41] offset:192 // (op 787) Store local data 

v_and_b32 v14, -64, v1 // (op 1283) 
v_sub_u32 v16, v1, v14 // (op 1283) 

v_lshrrev_b32 v14, 6, v1 // (op 1283) 
v_lshl_add_u32 v18, v14, 7, v16 // (op 1283) 





s_waitcnt vmcnt(10)
ds_write_b8 v18, v42 offset:8192 // (op 788) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v18, v43 offset:8256 // (op 788) Store local data 

v_lshlrev_b32 v14, 6, v1 // (op 1172) 
v_lshrrev_b32 v16, 3, v14 // (op 1172) 




s_waitcnt vmcnt(8)
ds_write_b64 v16, v[22:23] offset:8448 // (op 789) Store local data 

v_lshlrev_b32 v14, 6, v1 // (op 1143) 
v_lshrrev_b32 v16, 3, v14 // (op 1143) 

v_and_b32 v14, -4, v1 // (op 1196) 
v_sub_u32 v18, v1, v14 // (op 1196) 

v_lshlrev_b32 v14, 4, v18 // (op 1196) 

v_lshrrev_b32 v18, 2, v1 // (op 1196) 
v_lshl_add_u32 v21, v18, 8, v14 // (op 1196) 


v_and_b32 v14, -64, v1 // (op 1254) 
v_sub_u32 v18, v1, v14 // (op 1254) 

v_lshrrev_b32 v14, 6, v1 // (op 1254) 
v_lshl_add_u32 v44, v14, 7, v18 // (op 1254) 


v_lshlrev_b32 v14, 6, v1 // (op 1330) 
v_lshrrev_b32 v18, 3, v14 // (op 1330) 

v_and_b32 v14, -4, v1 // (op 1354) 
v_sub_u32 v45, v1, v14 // (op 1354) 

v_lshlrev_b32 v14, 4, v45 // (op 1354) 

v_lshrrev_b32 v45, 2, v1 // (op 1354) 
v_lshl_add_u32 v64, v45, 8, v14 // (op 1354) 


v_and_b32 v14, -64, v1 // (op 1383) 
v_sub_u32 v45, v1, v14 // (op 1383) 

v_lshrrev_b32 v14, 6, v1 // (op 1383) 
v_lshl_add_u32 v65, v14, 7, v45 // (op 1383) 


s_mov_b32 s0, 0 // (op 308) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 307) 
s_mov_b32 s1, s6 // (op 307) convert

s_and_b32 s6, s1, 4294967294 // (op 307) 

s_cmp_lt_u32 s0, s6 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_307 // (op 307) Condition: Top (jump to RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_307 if false)
RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_307:
s_add_u32 s0, s0, 2 // (op 309) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1441) 



ds_read_b64_tr_b4 v[66:67], v3 offset:8448 // (op 65) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v3 offset:8576 // (op 65) Transpose load local data 
s_barrier  // (op 1454) 



ds_read_b64_tr_b8 v[70:71], v5 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v5 offset:512 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v5 offset:4096 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v5 offset:4608 // (op 70) Transpose load local data 
s_barrier  // (op 1468) 


ds_read_u8 v14, v0 offset:8192 // (op 79) Load local data 



ds_read_b64_tr_b8 v[78:79], v5 offset:32 // (op 339) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v5 offset:544 // (op 339) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[70:77], a[0:3], v24, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 82) 
ds_read_b64_tr_b8 v[82:83], v5 offset:4128 // (op 339) Transpose load local data 


ds_read_b64_tr_b8 v[84:85], v5 offset:4640 // (op 339) Transpose load local data 


ds_read_u8 v14, v0 offset:8224 // (op 342) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[78:85], a[4:7], v24, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 343) 




v_mov_b64 v[66:67], s[28:29] // (op 75) convert

v_mov_b32 v14, v19 // (op 75) 
s_lshl_b32 s1, s28, 3 // (op 75) 
v_add_u32 v45, v14, s1 // (op 75) 

v_mov_b32 v68, v45 // (op 75) 
buffer_load_ubyte v24, v68, s[52:55], 0 offen offset: 0 // (op 75) Load value




v_mov_b64 v[66:67], s[24:25] // (op 10) convert

v_mov_b32 v45, v17 // (op 10) 
s_lshl_b32 s1, s24, 8 // (op 10) 
v_add_u32 v68, v45, s1 // (op 10) 

buffer_load_dwordx4 v[26:29], v68, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v68, v68, v66 // (op 10) 
buffer_load_dwordx4 v[30:33], v68, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v68, v68, v66 // (op 10) 
buffer_load_dwordx4 v[34:37], v68, s[48:51], 0 offen // (op 10) Load value
v_add_u32 v68, v68, v66 // (op 10) 
buffer_load_dwordx4 v[38:41], v68, s[48:51], 0 offen // (op 10) Load value



v_mov_b64 v[66:67], s[34:35] // (op 21) convert

v_mov_b32 v68, v20 // (op 21) 
s_lshl_b32 s1, s34, 3 // (op 21) 
v_add_u32 v69, v68, s1 // (op 21) 

buffer_load_ubyte v42, v69, s[56:59], 0 offen offset: 0 // (op 21) Load value
v_add_u32 v69, v69, v66 // (op 21) 
buffer_load_ubyte v43, v69, s[56:59], 0 offen offset: 0 // (op 21) Load value




v_mov_b32 v66, v15 // (op 4) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 4) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 4) 

s_lshl_b32 s1, s6, 1 // (op 4) 

v_add_u32 v67, v66, s1 // (op 4) 


buffer_load_dwordx2 v[22:23], v67, s[8:11], 0 offen // (op 4) Load value

s_barrier  // (op 1491) 



s_waitcnt vmcnt(14)
ds_write_b128 v64, v[46:49] offset:9472 // (op 235) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v64, v[50:53] offset:9536 // (op 235) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v64, v[54:57] offset:9600 // (op 235) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v64, v[58:61] offset:9664 // (op 235) Store local data 
s_barrier  // (op 1497) 



s_waitcnt vmcnt(10)
ds_write_b8 v65, v62 offset:17664 // (op 236) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v65, v63 offset:17728 // (op 236) Store local data 
s_barrier  // (op 1485) 



s_waitcnt vmcnt(8)
ds_write_b64 v18, v[12:13] offset:17920 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1482) 



ds_read_b64_tr_b4 v[70:71], v4 offset:17920 // (op 227) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v4 offset:18048 // (op 227) Transpose load local data 
s_barrier  // (op 1488) 
s_barrier  // (op 1534) 



ds_read_b64_tr_b8 v[74:75], v8 offset:9472 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v8 offset:9984 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v8 offset:13568 // (op 229) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v8 offset:14080 // (op 229) Transpose load local data 
s_barrier  // (op 1494) 


ds_read_u8 v67, v7 offset:17664 // (op 233) Load local data 



ds_read_b64_tr_b8 v[82:83], v8 offset:9504 // (op 348) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v8 offset:10016 // (op 348) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[74:81], a[0:3], v25, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 234) 
ds_read_b64_tr_b8 v[86:87], v8 offset:13600 // (op 348) Transpose load local data 


ds_read_b64_tr_b8 v[88:89], v8 offset:14112 // (op 348) Transpose load local data 


s_barrier  // (op 1568) 
ds_read_u8 v67, v7 offset:17696 // (op 351) Load local data 
s_barrier  // (op 1572) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[82:89], a[4:7], v25, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 344) 




v_mov_b64 v[70:71], s[28:29] // (op 231) convert

s_lshl_b32 s1, s28, 2 // (op 231) 
s_mul_i32 s6, 3, s1 // (op 231) 

v_add_u32 v67, v14, s6 // (op 231) 

v_mov_b32 v69, v67 // (op 231) 
buffer_load_ubyte v25, v69, s[52:55], 0 offen offset: 0 // (op 231) Load value





s_lshl_b32 s1, s28, 3 // (op 1033) 
v_add_u32 v19, v19, s1 // (op 1033) 

v_mov_b64 v[70:71], s[24:25] // (op 237) convert

s_lshl_b32 s1, s24, 7 // (op 237) 
s_mul_i32 s6, 3, s1 // (op 237) 

v_add_u32 v14, v45, s6 // (op 237) 

buffer_load_dwordx4 v[46:49], v14, s[48:51], 0 offen // (op 237) Load value
v_add_u32 v14, v14, v70 // (op 237) 
buffer_load_dwordx4 v[50:53], v14, s[48:51], 0 offen // (op 237) Load value
v_add_u32 v14, v14, v70 // (op 237) 
buffer_load_dwordx4 v[54:57], v14, s[48:51], 0 offen // (op 237) Load value
v_add_u32 v14, v14, v70 // (op 237) 
buffer_load_dwordx4 v[58:61], v14, s[48:51], 0 offen // (op 237) Load value




s_lshl_b32 s1, s24, 8 // (op 969) 
v_add_u32 v17, v17, s1 // (op 969) 

v_mov_b64 v[70:71], s[34:35] // (op 238) convert

s_lshl_b32 s1, s34, 2 // (op 238) 
s_mul_i32 s6, 3, s1 // (op 238) 

v_add_u32 v14, v68, s6 // (op 238) 

buffer_load_ubyte v62, v14, s[56:59], 0 offen offset: 0 // (op 238) Load value
v_add_u32 v14, v14, v70 // (op 238) 
buffer_load_ubyte v63, v14, s[56:59], 0 offen offset: 0 // (op 238) Load value





s_lshl_b32 s1, s34, 3 // (op 1097) 
v_add_u32 v20, v20, s1 // (op 1097) 

s_lshl_b64 s[6:7], s[12:13], 7 // (op 224) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 224) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 224) 

s_mul_i32 s1, 3, s6 // (op 224) 

v_add_u32 v14, v66, s1 // (op 224) 


buffer_load_dwordx2 v[12:13], v14, s[8:11], 0 offen // (op 224) Load value


s_barrier  // (op 1457) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 905) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 905) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 905) 

s_lshl_b32 s1, s6, 1 // (op 905) 

v_add_u32 v15, v15, s1 // (op 905) 

s_waitcnt vmcnt(14)
ds_write_b128 v21, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v21, v[30:33] offset:64 // (op 13) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v21, v[34:37] offset:128 // (op 13) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v21, v[38:41] offset:192 // (op 13) Store local data 
s_barrier  // (op 1471) 



s_waitcnt vmcnt(10)
ds_write_b8 v44, v42 offset:8192 // (op 26) Store local data 
s_waitcnt vmcnt(9)
ds_write_b8 v44, v43 offset:8256 // (op 26) Store local data 
s_barrier  // (op 1444) 



s_waitcnt vmcnt(8)
ds_write_b64 v16, v[22:23] offset:8448 // (op 6) Store local data 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 307) 
s_mov_b32 s1, s6 // (op 307) convert

s_and_b32 s6, s1, 4294967294 // (op 307) 

s_cmp_lt_u32 s0, s6 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_307 // (op 307) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoop_307 if true)
RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoop_307:















s_ashr_i64 s[0:1], s[22:23], 7 // (op 867) 
s_lshr_b32 s6, s0, 1 // (op 867) 

v_lshl_add_u32 v14, s6, 8, v1 // (op 867) 

s_lshl_b32 s0, s16, 4 // (op 867) 
v_mul_lo_u32 v27, v14, s13 // (op 867) most significant: low of low * high
v_mul_hi_u32 v15, v14, s12 // (op 867) most significant: high of low * low
v_mul_lo_u32 v26, v14, s12 // (op 867) least significant: low of low * low
v_add_u32 v27, v27, v15 // (op 867) most significant: sum


s_mov_b32 s7, 0 // (op 867) convert
s_mov_b32 s6, s0 // (op 867) convert

v_mov_b32 v14, s6 // (op 867) 
v_mov_b32 v15, s7 // (op 867) 
v_add_co_u32 v16, s[0:1], v14, v26 // (op 867) least significant half
v_addc_co_u32 v17, s[0:1], v15, v27, s[0:1] // (op 867) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 867) 


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 867) 

v_mov_b32 v16, v14 // (op 867) convert

s_mul_i32 s1, s12, s23 // (op 869) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 869) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 869) least significant
s_mul_hi_u32 s7, s12, s22 // (op 869) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 869) most significant: sum
s_add_u32 s1, s1, s7 // (op 869) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 869) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 869) 

s_mov_b32 s8, s20 // (op 869) reg expression
s_mov_b32 s9, s21 // (op 869) reg expression
s_mov_b32 s10, s0 // (op 869) reg expression
s_mov_b32 s11, 131072 // (op 869) call()

v_and_b32 v14, 4294967292, v1 // (op 935) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 935) 
v_and_b32 v15, -4, v1 // (op 935) 
s_lshr_b32 s6, s0, 1 // (op 935) 

v_sub_u32 v17, v1, v15 // (op 935) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 935) 


v_lshlrev_b32 v14, 4, v17 // (op 935) 

v_lshl_add_u32 v17, s17, 6, v14 // (op 935) 

v_mul_lo_u32 v14, v15, s24 // (op 935) 
v_add_u32 v14, v14, v17 // (op 935) 


s_mul_i32 s1, s24, s23 // (op 937) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 937) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 937) least significant
s_mul_hi_u32 s7, s24, s22 // (op 937) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 937) most significant: sum
s_add_u32 s1, s1, s7 // (op 937) most significant: sum


s_mov_b32 s48, s26 // (op 937) reg expression
s_mov_b32 s49, s27 // (op 937) reg expression
s_mov_b32 s50, s0 // (op 937) reg expression
s_mov_b32 s51, 131072 // (op 937) call()

v_lshrrev_b32 v15, 7, v1 // (op 999) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 999) 
v_and_b32 v17, 15, v1 // (op 999) 
s_lshr_b32 s6, s0, 1 // (op 999) 

v_lshl_add_u32 v18, v15, 4, v17 // (op 999) 


v_lshl_add_u32 v15, s16, 4, v18 // (op 999) 

v_and_b32 v17, 63, v1 // (op 999) 
v_lshrrev_b32 v18, 4, v17 // (op 999) 

v_lshl_add_u32 v17, s6, 3, v18 // (op 999) 


v_mul_lo_u32 v18, v17, s28 // (op 999) 

v_add_u32 v17, v15, v18 // (op 999) 


s_mul_i32 s1, s28, s33 // (op 1001) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 1001) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1001) least significant
s_mul_hi_u32 s7, s28, s32 // (op 1001) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1001) most significant: sum
s_add_u32 s1, s1, s7 // (op 1001) most significant: sum


s_mov_b32 s52, s30 // (op 1001) reg expression
s_mov_b32 s53, s31 // (op 1001) reg expression
s_mov_b32 s54, s0 // (op 1001) reg expression
s_mov_b32 s55, 131072 // (op 1001) call()

v_lshrrev_b32 v15, 6, v1 // (op 1063) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 1063) 
v_lshlrev_b32 v18, 1, v15 // (op 1063) 

s_lshr_b32 s6, s0, 1 // (op 1063) 

v_and_b32 v15, -64, v1 // (op 1063) 
v_lshl_add_u32 v19, s6, 3, v18 // (op 1063) 


v_sub_u32 v18, v1, v15 // (op 1063) 

v_lshl_add_u32 v15, s17, 6, v18 // (op 1063) 

v_mul_lo_u32 v18, v19, s34 // (op 1063) 
v_add_u32 v18, v18, v15 // (op 1063) 


s_mul_i32 s1, s34, s33 // (op 1065) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 1065) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 1065) least significant
s_mul_hi_u32 s7, s34, s32 // (op 1065) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1065) most significant: sum
s_add_u32 s1, s1, s7 // (op 1065) most significant: sum


s_mov_b32 s56, s36 // (op 1065) reg expression
s_mov_b32 s57, s37 // (op 1065) reg expression
s_mov_b32 s58, s0 // (op 1065) reg expression
s_mov_b32 s59, 131072 // (op 1065) call()

v_lshlrev_b32 v15, 6, v1 // (op 1156) 
v_lshrrev_b32 v19, 3, v15 // (op 1156) 

v_and_b32 v15, -4, v1 // (op 1209) 
v_sub_u32 v20, v1, v15 // (op 1209) 

v_lshlrev_b32 v15, 4, v20 // (op 1209) 

v_lshrrev_b32 v20, 2, v1 // (op 1209) 
v_lshl_add_u32 v21, v20, 8, v15 // (op 1209) 


v_and_b32 v15, -64, v1 // (op 1267) 
v_sub_u32 v20, v1, v15 // (op 1267) 

v_lshrrev_b32 v15, 6, v1 // (op 1267) 
v_lshl_add_u32 v22, v15, 7, v20 // (op 1267) 


s_ashr_i64 s[0:1], s[22:23], 7 // (op 313) 
s_and_b32 s6, s0, 4294967294 // (op 313) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 312) 
s_mov_b32 s7, s0 // (op 312) convert

s_cmp_lt_u32 s6, s7 // (op 312) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_312 // (op 312) Condition: Top (jump to RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_312 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_312:


s_add_u32 s6, s6, 1 // (op 314) 
v_mov_b32 v15, v16 // (op 166) 
v_mov_b32 v20, v15 // (op 166) reg expression

buffer_load_dwordx2 v[12:13], v20, s[8:11], 0 offen // (op 166) Load value


s_lshl_b64 s[0:1], s[12:13], 7 // (op 857) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 857) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 857) 

v_add_u32 v16, v16, s0 // (op 857) 


s_barrier  // (op 1447) 
v_mov_b64 v[24:25], s[24:25] // (op 179) convert




v_mov_b32 v15, v14 // (op 179) 
v_mov_b32 v20, v15 // (op 179) reg expression
buffer_load_dwordx4 v[26:29], v20, s[48:51], 0 offen // (op 179) Load value
v_add_u32 v20, v20, v24 // (op 179) 
buffer_load_dwordx4 v[30:33], v20, s[48:51], 0 offen // (op 179) Load value
v_add_u32 v20, v20, v24 // (op 179) 
s_waitcnt vmcnt(2)
ds_write_b64 v19, v[12:13] offset:8448 // (op 167) Store local data 
buffer_load_dwordx4 v[34:37], v20, s[48:51], 0 offen // (op 179) Load value

v_add_u32 v20, v20, v24 // (op 179) 
buffer_load_dwordx4 v[38:41], v20, s[48:51], 0 offen // (op 179) Load value



s_lshl_b32 s7, s24, 7 // (op 927) 
v_add_u32 v14, v14, s7 // (op 927) 


s_barrier  // (op 1460) 
v_mov_b64 v[12:13], s[34:35] // (op 180) convert




v_mov_b32 v15, v18 // (op 180) 
v_mov_b32 v20, v15 // (op 180) reg expression
buffer_load_ubyte v24, v20, s[56:59], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v20, v20, v12 // (op 180) 
s_waitcnt vmcnt(4)
ds_write_b128 v21, v[26:29] // (op 177) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v21, v[30:33] offset:64 // (op 177) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v21, v[34:37] offset:128 // (op 177) Store local data 
buffer_load_ubyte v25, v20, s[56:59], 0 offen offset: 0 // (op 180) Load value



s_lshl_b32 s7, s34, 2 // (op 1055) 
v_add_u32 v18, v18, s7 // (op 1055) 

s_waitcnt vmcnt(2)
ds_write_b128 v21, v[38:41] offset:192 // (op 177) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1450) 



s_barrier  // (op 1474) 
ds_read_b64_tr_b4 v[26:27], v3 offset:8448 // (op 169) Transpose load local data 



ds_read_b64_tr_b4 v[28:29], v3 offset:8576 // (op 169) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v22, v24 offset:8192 // (op 178) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1464) 
s_waitcnt vmcnt(0)
ds_write_b8 v22, v25 offset:8256 // (op 178) Store local data 




ds_read_b64_tr_b8 v[30:31], v5 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v5 offset:512 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v5 offset:4096 // (op 171) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v5 offset:4608 // (op 171) Transpose load local data 

v_mov_b64 v[12:13], s[28:29] // (op 173) convert

v_mov_b32 v15, v17 // (op 173) 
v_mov_b32 v20, v15 // (op 173) reg expression
v_mov_b32 v23, v20 // (op 173) 
buffer_load_ubyte v24, v23, s[52:55], 0 offen offset: 0 // (op 173) Load value




s_lshl_b32 s7, s28, 2 // (op 991) 
v_add_u32 v17, v17, s7 // (op 991) 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1478) 


ds_read_u8 v15, v0 offset:8192 // (op 175) Load local data 



ds_read_b64_tr_b8 v[38:39], v5 offset:32 // (op 320) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:37], a[0:3], v24, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 176) 
ds_read_b64_tr_b8 v[40:41], v5 offset:544 // (op 320) Transpose load local data 


ds_read_b64_tr_b8 v[42:43], v5 offset:4128 // (op 320) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v5 offset:4640 // (op 320) Transpose load local data 


ds_read_u8 v15, v0 offset:8224 // (op 323) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[38:45], a[4:7], v24, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 324) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 312) 
s_mov_b32 s7, s0 // (op 312) convert

s_cmp_lt_u32 s6, s7 // (op 312) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_312 // (op 312) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopTop_KLoopTail_312 if true)
RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ForLoopBottom_KLoopTail_312:






















s_barrier  // (op 1556) 






v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalFalse_0_beta_123:


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
RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.LRR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end:
.size RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2, .LRR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2_end-RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 18944 // lds bytes
  .amdhsa_accum_offset 92
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
  - .name: RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2
    .symbol: RR_GEMM_NT_FP4_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 18944
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 90
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

