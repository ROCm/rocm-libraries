// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2,@function
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 1184) 
v_and_b32 v3, 15, v1 // (op 1184) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1184) 


v_and_b32 v0, 48, v1 // (op 1184) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1184) 


v_lshrrev_b32 v0, 7, v1 // (op 1342) 
v_and_b32 v4, 15, v1 // (op 1342) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1342) 


v_and_b32 v0, 48, v1 // (op 1342) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1342) 


v_lshrrev_b32 v0, 7, v1 // (op 1127) 
v_and_b32 v5, 63, v1 // (op 1127) 
v_lshrrev_b32 v6, 4, v5 // (op 1127) 
v_lshlrev_b32 v7, 2, v6 // (op 1127) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1127) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 1127) 

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


v_lshrrev_b32 v0, 6, v1 // (op 1242) 
v_and_b32 v5, 1, v0 // (op 1242) 

v_and_b32 v0, 15, v1 // (op 1242) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 1242) 


v_and_b32 v0, 63, v1 // (op 1242) 
v_lshrrev_b32 v5, 4, v0 // (op 1242) 

v_lshlrev_b32 v0, 5, v5 // (op 1242) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 1242) 


v_lshlrev_b32 v0, 2, v5 // (op 1242) 

v_lshrrev_b32 v5, 3, v0 // (op 1242) 

v_lshrrev_b32 v0, 6, v1 // (op 1300) 
v_and_b32 v7, 1, v0 // (op 1300) 

v_and_b32 v0, 15, v1 // (op 1300) 
s_mul_i32 s1, s2, s9 // (op 1129) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1129) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1129) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1129) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1129) most significant: sum
s_add_u32 s1, s1, s19 // (op 1129) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1129) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 1300) 


v_and_b32 v0, 63, v1 // (op 1300) 
v_lshrrev_b32 v7, 4, v0 // (op 1300) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 1300) 


v_lshrrev_b32 v7, 7, v1 // (op 1316) 
v_and_b32 v8, 63, v1 // (op 1316) 
v_lshrrev_b32 v9, 4, v8 // (op 1316) 
v_lshlrev_b32 v10, 2, v9 // (op 1316) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 1316) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 1316) 

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


v_lshrrev_b32 v7, 6, v1 // (op 1371) 
v_and_b32 v8, 1, v7 // (op 1371) 

v_and_b32 v7, 15, v1 // (op 1371) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 1371) 


v_and_b32 v7, 63, v1 // (op 1371) 
v_lshrrev_b32 v8, 4, v7 // (op 1371) 

v_lshlrev_b32 v7, 5, v8 // (op 1371) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 1371) 


v_lshlrev_b32 v7, 2, v8 // (op 1371) 

v_lshrrev_b32 v8, 3, v7 // (op 1371) 

v_lshrrev_b32 v7, 6, v1 // (op 1400) 
v_and_b32 v10, 1, v7 // (op 1400) 

v_and_b32 v7, 15, v1 // (op 1400) 
s_mul_i32 s1, s4, s9 // (op 1318) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 1400) 


v_and_b32 v7, 63, v1 // (op 1400) 
v_lshrrev_b32 v10, 4, v7 // (op 1400) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 1400) 


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
v_lshrrev_b32 v12, 3, v1 // (op 894) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 894) 

v_and_b32 v12, -8, v1 // (op 894) 
v_sub_u32 v14, v1, v12 // (op 894) 

v_lshlrev_b32 v12, 4, v14 // (op 894) 

v_mul_lo_u32 v14, v13, s12 // (op 894) 
v_add_u32 v14, v14, v12 // (op 894) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 896) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 896) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 896) least significant
s_mul_hi_u32 s7, s12, s22 // (op 896) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 896) most significant: sum
s_add_u32 s1, s1, s7 // (op 896) most significant: sum


s_mov_b32 s48, s20 // (op 896) reg expression
s_mov_b32 s49, s21 // (op 896) reg expression
s_mov_b32 s50, s0 // (op 896) reg expression
s_mov_b32 s51, 131072 // (op 896) call()

v_lshrrev_b32 v12, 3, v1 // (op 913) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 913) 

v_and_b32 v12, -8, v1 // (op 913) 
v_sub_u32 v15, v1, v12 // (op 913) 

v_lshlrev_b32 v12, 4, v15 // (op 913) 

v_mul_lo_u32 v15, v13, s12 // (op 913) 
v_add_u32 v15, v15, v12 // (op 913) 


s_mul_i32 s1, s12, s23 // (op 915) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 915) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 915) least significant
s_mul_hi_u32 s7, s12, s22 // (op 915) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 915) most significant: sum
s_add_u32 s1, s1, s7 // (op 915) most significant: sum


s_mov_b32 s48, s20 // (op 915) reg expression
s_mov_b32 s49, s21 // (op 915) reg expression
s_mov_b32 s50, s0 // (op 915) reg expression
s_mov_b32 s51, 131072 // (op 915) call()

v_and_b32 v12, -4, v1 // (op 958) 
v_sub_u32 v13, v1, v12 // (op 958) 

v_lshlrev_b32 v12, 5, v13 // (op 958) 

v_mov_b32 v17, 0 // (op 958) convert
v_mov_b32 v16, v12 // (op 958) convert

v_lshrrev_b32 v12, 2, v1 // (op 958) 
v_lshl_add_u32 v13, s17, 6, v12 // (op 958) 

v_mul_lo_u32 v19, v13, s25 // (op 958) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 958) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 958) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 958) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 958) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 958) most significant half

v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 958) 


v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 958) 

v_mov_b32 v12, v16 // (op 958) convert

s_mul_i32 s1, s24, s9 // (op 960) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 960) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 960) least significant
s_mul_hi_u32 s7, s24, s8 // (op 960) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 960) most significant: sum
s_add_u32 s1, s1, s7 // (op 960) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 960) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 960) 

s_mov_b32 s52, s26 // (op 960) reg expression
s_mov_b32 s53, s27 // (op 960) reg expression
s_mov_b32 s54, s0 // (op 960) reg expression
s_mov_b32 s55, 131072 // (op 960) call()

v_and_b32 v13, -4, v1 // (op 977) 
v_sub_u32 v16, v1, v13 // (op 977) 

v_lshlrev_b32 v13, 5, v16 // (op 977) 

v_mov_b32 v17, 0 // (op 977) convert
v_mov_b32 v16, v13 // (op 977) convert

v_lshrrev_b32 v13, 2, v1 // (op 977) 
v_lshl_add_u32 v18, s17, 6, v13 // (op 977) 

v_mul_lo_u32 v21, v18, s25 // (op 977) most significant: low of low * high
v_mul_hi_u32 v13, v18, s24 // (op 977) most significant: high of low * low
v_mul_lo_u32 v20, v18, s24 // (op 977) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 977) most significant: sum


v_add_co_u32 v18, s[0:1], v16, v20 // (op 977) least significant half
v_addc_co_u32 v19, s[0:1], v17, v21, s[0:1] // (op 977) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 977) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 977) 

v_mov_b32 v13, v16 // (op 977) convert

s_mul_i32 s1, s24, s9 // (op 979) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 979) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 979) least significant
s_mul_hi_u32 s7, s24, s8 // (op 979) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 979) most significant: sum
s_add_u32 s1, s1, s7 // (op 979) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 979) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 979) 

s_mov_b32 s52, s26 // (op 979) reg expression
s_mov_b32 s53, s27 // (op 979) reg expression
s_mov_b32 s54, s0 // (op 979) reg expression
s_mov_b32 s55, 131072 // (op 979) call()

v_lshrrev_b32 v16, 7, v1 // (op 1022) 
v_and_b32 v17, 15, v1 // (op 1022) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 1022) 


v_lshl_add_u32 v16, s16, 5, v18 // (op 1022) 

v_and_b32 v17, 63, v1 // (op 1022) 
v_lshrrev_b32 v18, 4, v17 // (op 1022) 

v_mul_lo_u32 v17, v16, s28 // (op 1022) 
v_add_u32 v17, v17, v18 // (op 1022) 


s_mul_i32 s1, s28, s23 // (op 1024) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1024) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1024) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1024) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1024) most significant: sum
s_add_u32 s1, s1, s7 // (op 1024) most significant: sum


s_mov_b32 s56, s30 // (op 1024) reg expression
s_mov_b32 s57, s31 // (op 1024) reg expression
s_mov_b32 s58, s0 // (op 1024) reg expression
s_mov_b32 s59, 131072 // (op 1024) call()

v_lshrrev_b32 v16, 7, v1 // (op 1041) 
v_and_b32 v18, 15, v1 // (op 1041) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 1041) 


v_lshl_add_u32 v16, s16, 5, v19 // (op 1041) 

v_and_b32 v18, 63, v1 // (op 1041) 
v_lshrrev_b32 v19, 4, v18 // (op 1041) 

v_mul_lo_u32 v18, v16, s28 // (op 1041) 
v_add_u32 v18, v18, v19 // (op 1041) 


s_mul_i32 s1, s28, s23 // (op 1043) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1043) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1043) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1043) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1043) most significant: sum
s_add_u32 s1, s1, s7 // (op 1043) most significant: sum


s_mov_b32 s56, s30 // (op 1043) reg expression
s_mov_b32 s57, s31 // (op 1043) reg expression
s_mov_b32 s58, s0 // (op 1043) reg expression
s_mov_b32 s59, 131072 // (op 1043) call()

v_and_b32 v16, -4, v1 // (op 1086) 
v_sub_u32 v19, v1, v16 // (op 1086) 

v_lshrrev_b32 v16, 2, v1 // (op 1086) 
v_lshl_add_u32 v20, s17, 6, v16 // (op 1086) 

v_mul_lo_u32 v16, v20, s32 // (op 1086) 

v_add_u32 v20, v19, v16 // (op 1086) 


s_mul_i32 s1, s32, s9 // (op 1088) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1088) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1088) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1088) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1088) most significant: sum
s_add_u32 s1, s1, s7 // (op 1088) most significant: sum


s_mov_b32 s60, s34 // (op 1088) reg expression
s_mov_b32 s61, s35 // (op 1088) reg expression
s_mov_b32 s62, s0 // (op 1088) reg expression
s_mov_b32 s63, 131072 // (op 1088) call()

v_and_b32 v16, -4, v1 // (op 1105) 
v_sub_u32 v19, v1, v16 // (op 1105) 

v_lshrrev_b32 v16, 2, v1 // (op 1105) 
v_lshl_add_u32 v21, s17, 6, v16 // (op 1105) 

v_mul_lo_u32 v16, v21, s32 // (op 1105) 

v_add_u32 v21, v19, v16 // (op 1105) 


s_mul_i32 s1, s32, s9 // (op 1107) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1107) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1107) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1107) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1107) most significant: sum
s_add_u32 s1, s1, s7 // (op 1107) most significant: sum


s_mov_b32 s60, s34 // (op 1107) reg expression
s_mov_b32 s61, s35 // (op 1107) reg expression
s_mov_b32 s62, s0 // (op 1107) reg expression
s_mov_b32 s63, 131072 // (op 1107) call()




buffer_load_ubyte v16, v17, s[56:59], 0 offen offset: 0 // (op 761) Load value



buffer_load_dwordx4 v[22:25], v12, s[52:55], 0 offen // (op 765) Load value



buffer_load_ubyte v19, v20, s[60:63], 0 offen offset: 0 // (op 768) Load value



buffer_load_dwordx4 v[26:29], v14, s[48:51], 0 offen // (op 771) Load value



buffer_load_ubyte v30, v17, s[56:59], 0 offen offset:4 // (op 775) Load value




buffer_load_dwordx4 v[32:35], v12, s[52:55], 0 offen offset:64 // (op 779) Load value




buffer_load_ubyte v12, v20, s[60:63], 0 offen offset:4 // (op 782) Load value




buffer_load_dwordx4 v[36:39], v14, s[48:51], 0 offen offset:128 // (op 785) Load value

v_and_b32 v14, -4, v1 // (op 1225) 
v_sub_u32 v17, v1, v14 // (op 1225) 

v_lshlrev_b32 v14, 5, v17 // (op 1225) 

v_lshrrev_b32 v17, 2, v1 // (op 1225) 
v_lshl_add_u32 v20, v17, 7, v14 // (op 1225) 


v_lshlrev_b32 v14, 2, v20 // (op 1225) 

v_lshrrev_b32 v17, 3, v14 // (op 1225) 




s_waitcnt vmcnt(6)
ds_write_b128 v17, v[22:25] // (op 787) Store local data 

v_and_b32 v14, -4, v1 // (op 1283) 
v_sub_u32 v17, v1, v14 // (op 1283) 

v_and_b32 v14, 4294967292, v1 // (op 1283) 
v_add_u32 v20, v14, v17 // (op 1283) 





s_waitcnt vmcnt(5)
ds_write_b8 v20, v19 offset:4096 // (op 788) Store local data 

v_and_b32 v14, -8, v1 // (op 1172) 
v_sub_u32 v17, v1, v14 // (op 1172) 

v_lshlrev_b32 v14, 4, v17 // (op 1172) 

v_lshrrev_b32 v17, 3, v1 // (op 1172) 
v_lshl_add_u32 v20, v17, 7, v14 // (op 1172) 





s_waitcnt vmcnt(4)
ds_write_b128 v20, v[26:29] offset:4352 // (op 789) Store local data 

v_and_b32 v14, -8, v1 // (op 1143) 
v_sub_u32 v17, v1, v14 // (op 1143) 

v_lshlrev_b32 v14, 4, v17 // (op 1143) 

v_lshrrev_b32 v17, 3, v1 // (op 1143) 
v_lshl_add_u32 v20, v17, 7, v14 // (op 1143) 


v_and_b32 v14, -4, v1 // (op 1196) 
v_sub_u32 v17, v1, v14 // (op 1196) 

v_lshlrev_b32 v14, 5, v17 // (op 1196) 

v_lshrrev_b32 v17, 2, v1 // (op 1196) 
v_lshl_add_u32 v31, v17, 7, v14 // (op 1196) 


v_lshlrev_b32 v14, 2, v31 // (op 1196) 

v_lshrrev_b32 v17, 3, v14 // (op 1196) 

v_and_b32 v14, -4, v1 // (op 1254) 
v_sub_u32 v31, v1, v14 // (op 1254) 

v_and_b32 v14, 4294967292, v1 // (op 1254) 
v_add_u32 v40, v14, v31 // (op 1254) 


v_and_b32 v14, -8, v1 // (op 1330) 
v_sub_u32 v31, v1, v14 // (op 1330) 

v_lshlrev_b32 v14, 4, v31 // (op 1330) 

v_lshrrev_b32 v31, 3, v1 // (op 1330) 
v_lshl_add_u32 v41, v31, 7, v14 // (op 1330) 


v_and_b32 v14, -4, v1 // (op 1354) 
v_sub_u32 v31, v1, v14 // (op 1354) 

v_lshlrev_b32 v14, 5, v31 // (op 1354) 

v_lshrrev_b32 v31, 2, v1 // (op 1354) 
v_lshl_add_u32 v42, v31, 7, v14 // (op 1354) 


v_lshlrev_b32 v14, 2, v42 // (op 1354) 

v_lshrrev_b32 v31, 3, v14 // (op 1354) 

v_and_b32 v14, -4, v1 // (op 1383) 
v_sub_u32 v42, v1, v14 // (op 1383) 

v_and_b32 v14, 4294967292, v1 // (op 1383) 
v_add_u32 v43, v14, v42 // (op 1383) 


s_mov_b32 s0, 0 // (op 308) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 307) 
s_mov_b32 s1, s6 // (op 307) convert

s_and_b32 s6, s1, 4294967294 // (op 307) 

s_cmp_lt_u32 s0, s6 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_307 // (op 307) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_307 if false)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoop_307:
s_add_u32 s0, s0, 2 // (op 309) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1441) 



ds_read_b128 v[44:47], v3 offset:4352 // (op 65) Load local data 
ds_read_b128 v[48:51], v3 offset:4416 // (op 65) Load local data 
s_barrier  // (op 1454) 



ds_read_b128 v[52:55], v5 // (op 70) Load local data 
s_barrier  // (op 1468) 



ds_read_u8 v14, v0 offset:4096 // (op 79) Load local data 



ds_read_b128 v[56:59], v5 offset:2048 // (op 339) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[52:55], a[0:3], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 82) 





ds_read_u8 v14, v0 offset:4224 // (op 342) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[56:59], a[4:7], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 343) 





v_mov_b32 v14, v18 // (op 75) 

buffer_load_ubyte v16, v14, s[56:59], 0 offen offset:8 // (op 75) Load value


v_mov_b32 v42, v13 // (op 10) 

buffer_load_dwordx4 v[22:25], v42, s[52:55], 0 offen offset:128 // (op 10) Load value


v_mov_b32 v44, v21 // (op 21) 

buffer_load_ubyte v19, v44, s[60:63], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v45, v15 // (op 4) 

buffer_load_dwordx4 v[26:29], v45, s[48:51], 0 offen offset:256 // (op 4) Load value
s_barrier  // (op 1491) 



s_waitcnt vmcnt(6)
ds_write_b128 v31, v[32:35] offset:8448 // (op 235) Store local data 
s_barrier  // (op 1497) 



s_waitcnt vmcnt(5)
ds_write_b8 v43, v12 offset:12544 // (op 236) Store local data 
s_barrier  // (op 1485) 



s_waitcnt vmcnt(4)
ds_write_b128 v41, v[36:39] offset:12800 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1482) 



ds_read_b128 v[46:49], v4 offset:12800 // (op 227) Load local data 
ds_read_b128 v[50:53], v4 offset:12864 // (op 227) Load local data 
s_barrier  // (op 1488) 
s_barrier  // (op 1534) 



ds_read_b128 v[54:57], v8 offset:8448 // (op 229) Load local data 
s_barrier  // (op 1494) 



ds_read_u8 v58, v7 offset:12544 // (op 233) Load local data 



ds_read_b128 v[60:63], v8 offset:10496 // (op 348) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[54:57], a[0:3], v30, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 234) 





s_barrier  // (op 1568) 
ds_read_u8 v54, v7 offset:12672 // (op 351) Load local data 
s_barrier  // (op 1572) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[60:63], a[4:7], v30, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 344) 






buffer_load_ubyte v30, v14, s[56:59], 0 offen offset:12 // (op 231) Load value




v_add_u32 v18, 8, v18 // (op 1033) 
buffer_load_dwordx4 v[32:35], v42, s[52:55], 0 offen offset:192 // (op 237) Load value




v_add_u32 v13, 128, v13 // (op 969) 
buffer_load_ubyte v12, v44, s[60:63], 0 offen offset:12 // (op 238) Load value




v_add_u32 v21, 8, v21 // (op 1097) 
buffer_load_dwordx4 v[36:39], v45, s[48:51], 0 offen offset:384 // (op 224) Load value

s_barrier  // (op 1457) 
v_add_u32 v15, 256, v15 // (op 905) 



s_waitcnt vmcnt(6)
ds_write_b128 v17, v[22:25] // (op 13) Store local data 
s_barrier  // (op 1471) 



s_waitcnt vmcnt(5)
ds_write_b8 v40, v19 offset:4096 // (op 26) Store local data 
s_barrier  // (op 1444) 



s_waitcnt vmcnt(4)
ds_write_b128 v20, v[26:29] offset:4352 // (op 6) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 307) 
s_mov_b32 s1, s6 // (op 307) convert

s_and_b32 s6, s1, 4294967294 // (op 307) 

s_cmp_lt_u32 s0, s6 // (op 307) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoop_307 // (op 307) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoop_307 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoop_307:















v_lshrrev_b32 v31, 3, v1 // (op 867) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 867) 
v_lshl_add_u32 v13, s16, 5, v31 // (op 867) 

s_lshr_b32 s6, s0, 1 // (op 867) 

v_and_b32 v31, -8, v1 // (op 867) 
v_sub_u32 v14, v1, v31 // (op 867) 

v_lshlrev_b32 v31, 4, v14 // (op 867) 

v_lshl_add_u32 v14, s6, 8, v31 // (op 867) 


v_mul_lo_u32 v31, v13, s12 // (op 867) 
v_add_u32 v31, v31, v14 // (op 867) 


s_mul_i32 s1, s12, s23 // (op 869) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 869) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 869) least significant
s_mul_hi_u32 s7, s12, s22 // (op 869) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 869) most significant: sum
s_add_u32 s1, s1, s7 // (op 869) most significant: sum


s_mov_b32 s48, s20 // (op 869) reg expression
s_mov_b32 s49, s21 // (op 869) reg expression
s_mov_b32 s50, s0 // (op 869) reg expression
s_mov_b32 s51, 131072 // (op 869) call()

v_and_b32 v13, -4, v1 // (op 935) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 935) 
v_sub_u32 v14, v1, v13 // (op 935) 

s_lshr_b32 s6, s0, 1 // (op 935) 

v_lshlrev_b32 v13, 5, v14 // (op 935) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 935) 


v_mov_b32 v29, 0 // (op 935) convert
v_mov_b32 v28, v14 // (op 935) convert

v_lshrrev_b32 v13, 2, v1 // (op 935) 
v_lshl_add_u32 v14, s17, 6, v13 // (op 935) 

v_mul_lo_u32 v27, v14, s25 // (op 935) most significant: low of low * high
v_mul_hi_u32 v13, v14, s24 // (op 935) most significant: high of low * low
v_mul_lo_u32 v26, v14, s24 // (op 935) least significant: low of low * low
v_add_u32 v27, v27, v13 // (op 935) most significant: sum


v_add_co_u32 v24, s[0:1], v28, v26 // (op 935) least significant half
v_addc_co_u32 v25, s[0:1], v29, v27, s[0:1] // (op 935) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 935) 


v_lshrrev_b64 v[22:23], 3, v[24:25] // (op 935) 

v_mov_b32 v13, v22 // (op 935) convert

s_mul_i32 s1, s24, s9 // (op 937) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 937) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 937) least significant
s_mul_hi_u32 s7, s24, s8 // (op 937) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 937) most significant: sum
s_add_u32 s1, s1, s7 // (op 937) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 937) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 937) 

s_mov_b32 s52, s26 // (op 937) reg expression
s_mov_b32 s53, s27 // (op 937) reg expression
s_mov_b32 s54, s0 // (op 937) reg expression
s_mov_b32 s55, 131072 // (op 937) call()

v_lshrrev_b32 v14, 7, v1 // (op 999) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 999) 
v_and_b32 v15, 15, v1 // (op 999) 
s_lshr_b32 s6, s0, 1 // (op 999) 

v_lshl_add_u32 v16, v14, 4, v15 // (op 999) 


v_lshl_add_u32 v14, s16, 5, v16 // (op 999) 

v_and_b32 v15, 63, v1 // (op 999) 
v_lshrrev_b32 v16, 4, v15 // (op 999) 

v_lshl_add_u32 v15, s6, 3, v16 // (op 999) 


v_mul_lo_u32 v16, v14, s28 // (op 999) 
v_add_u32 v16, v16, v15 // (op 999) 


s_mul_i32 s1, s28, s23 // (op 1001) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1001) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1001) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1001) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1001) most significant: sum
s_add_u32 s1, s1, s7 // (op 1001) most significant: sum


s_mov_b32 s56, s30 // (op 1001) reg expression
s_mov_b32 s57, s31 // (op 1001) reg expression
s_mov_b32 s58, s0 // (op 1001) reg expression
s_mov_b32 s59, 131072 // (op 1001) call()

v_and_b32 v14, -4, v1 // (op 1063) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1063) 
v_sub_u32 v15, v1, v14 // (op 1063) 

s_lshr_b32 s6, s0, 1 // (op 1063) 

v_lshrrev_b32 v14, 2, v1 // (op 1063) 
v_lshl_add_u32 v17, s6, 3, v15 // (op 1063) 


v_lshl_add_u32 v15, s17, 6, v14 // (op 1063) 

v_mul_lo_u32 v14, v15, s32 // (op 1063) 

v_add_u32 v15, v17, v14 // (op 1063) 


s_mul_i32 s1, s32, s9 // (op 1065) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1065) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1065) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1065) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1065) most significant: sum
s_add_u32 s1, s1, s7 // (op 1065) most significant: sum


s_mov_b32 s60, s34 // (op 1065) reg expression
s_mov_b32 s61, s35 // (op 1065) reg expression
s_mov_b32 s62, s0 // (op 1065) reg expression
s_mov_b32 s63, 131072 // (op 1065) call()

v_and_b32 v14, -8, v1 // (op 1156) 
v_sub_u32 v17, v1, v14 // (op 1156) 

v_lshlrev_b32 v14, 4, v17 // (op 1156) 

v_lshrrev_b32 v17, 3, v1 // (op 1156) 
v_lshl_add_u32 v18, v17, 7, v14 // (op 1156) 


v_and_b32 v14, -4, v1 // (op 1209) 
v_sub_u32 v17, v1, v14 // (op 1209) 

v_lshlrev_b32 v14, 5, v17 // (op 1209) 

v_lshrrev_b32 v17, 2, v1 // (op 1209) 
v_lshl_add_u32 v19, v17, 7, v14 // (op 1209) 


v_lshlrev_b32 v14, 2, v19 // (op 1209) 

v_lshrrev_b32 v17, 3, v14 // (op 1209) 

v_and_b32 v14, -4, v1 // (op 1267) 
v_sub_u32 v19, v1, v14 // (op 1267) 

v_and_b32 v14, 4294967292, v1 // (op 1267) 
v_add_u32 v20, v14, v19 // (op 1267) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 313) 
s_and_b32 s6, s0, 4294967294 // (op 313) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 312) 
s_mov_b32 s7, s0 // (op 312) convert

s_cmp_lt_u32 s6, s7 // (op 312) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_312 // (op 312) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_312 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_312:


s_add_u32 s6, s6, 1 // (op 314) 
v_mov_b32 v12, v31 // (op 166) 

buffer_load_dwordx4 v[22:25], v12, s[48:51], 0 offen // (op 166) Load value

v_add_u32 v31, 128, v31 // (op 857) 


s_barrier  // (op 1447) 
v_mov_b32 v12, v13 // (op 179) 




buffer_load_dwordx4 v[26:29], v12, s[52:55], 0 offen // (op 179) Load value

v_add_u32 v13, 64, v13 // (op 927) 
s_waitcnt vmcnt(1)
ds_write_b128 v18, v[22:25] offset:4352 // (op 167) Store local data 



v_mov_b32 v12, v15 // (op 180) 

s_barrier  // (op 1460) 
buffer_load_ubyte v14, v12, s[60:63], 0 offen offset: 0 // (op 180) Load value




v_add_u32 v15, 4, v15 // (op 1055) 
s_waitcnt vmcnt(1)
ds_write_b128 v17, v[26:29] // (op 177) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1450) 



s_barrier  // (op 1474) 
ds_read_b128 v[22:25], v3 offset:4352 // (op 169) Load local data 



ds_read_b128 v[26:29], v3 offset:4416 // (op 169) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v14 offset:4096 // (op 178) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1464) 



ds_read_b128 v[32:35], v5 // (op 171) Load local data 


v_mov_b32 v12, v16 // (op 173) 

buffer_load_ubyte v14, v12, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v16, 4, v16 // (op 991) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1478) 



ds_read_u8 v12, v0 offset:4096 // (op 175) Load local data 



ds_read_b128 v[36:39], v5 offset:2048 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[32:35], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 176) 





ds_read_u8 v12, v0 offset:4224 // (op 323) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[36:39], a[4:7], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 324) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 312) 
s_mov_b32 s7, s0 // (op 312) convert

s_cmp_lt_u32 s6, s7 // (op 312) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_312 // (op 312) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopTop_KLoopTail_312 if true)
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ForLoopBottom_KLoopTail_312:






















s_barrier  // (op 1556) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_123
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
s_branch RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalFalse_0_beta_123:


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
RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_ConditionalBottom_0_beta_123:










s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
.LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_end:
.size RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2, .LRR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_end-RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F4
    DestDataType: s
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum } ]
    macrotile: [32, 64, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2
    .symbol: RR_GEMM_TN_FP8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 64
    .agpr_count: 8
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
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .value_kind: by_value
...

.end_amdgpu_metadata

