// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 917) 
v_and_b32 v3, 15, v1 // (op 917) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 917) 


v_and_b32 v0, 48, v1 // (op 917) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 917) 


v_and_b32 v0, 48, v1 // (op 969) 
v_and_b32 v4, 63, v1 // (op 969) 
v_sub_u32 v5, v4, v0 // (op 969) 

v_lshrrev_b32 v4, 1, v5 // (op 969) 
v_add_lshl_u32 v6, v0, v4, 4 // (op 969) 


v_and_b32 v0, 1, v5 // (op 969) 

v_lshlrev_b32 v4, 3, v0 // (op 969) 

v_add_u32 v0, v6, v4 // (op 969) 


v_lshrrev_b32 v4, 6, v1 // (op 1022) 
v_and_b32 v5, 15, v1 // (op 1022) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1022) 


v_and_b32 v4, 63, v1 // (op 1022) 
v_lshrrev_b32 v5, 4, v4 // (op 1022) 

v_lshl_add_u32 v4, v6, 3, v5 // (op 1022) 


v_and_b32 v5, 48, v1 // (op 1075) 
v_and_b32 v6, 15, v1 // (op 1075) 
v_add_u32 v7, v5, v6 // (op 1075) 


v_lshrrev_b32 v5, 6, v1 // (op 1148) 
v_and_b32 v6, 15, v1 // (op 1148) 
v_lshl_add_u32 v8, v5, 4, v6 // (op 1148) 


v_and_b32 v5, 48, v1 // (op 1148) 
v_lshl_add_u32 v6, v8, 8, v5 // (op 1148) 


v_and_b32 v5, 48, v1 // (op 1187) 
v_and_b32 v8, 63, v1 // (op 1187) 
v_sub_u32 v9, v8, v5 // (op 1187) 

v_lshrrev_b32 v8, 1, v9 // (op 1187) 
v_add_lshl_u32 v10, v5, v8, 4 // (op 1187) 


v_and_b32 v5, 1, v9 // (op 1187) 

v_lshlrev_b32 v8, 3, v5 // (op 1187) 

v_add_u32 v5, v10, v8 // (op 1187) 


v_lshrrev_b32 v8, 6, v1 // (op 1211) 
v_and_b32 v9, 15, v1 // (op 1211) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 1211) 


v_and_b32 v8, 63, v1 // (op 1211) 
v_lshrrev_b32 v9, 4, v8 // (op 1211) 

v_lshl_add_u32 v8, v10, 3, v9 // (op 1211) 


v_and_b32 v9, 48, v1 // (op 1235) 
v_and_b32 v10, 15, v1 // (op 1235) 
v_add_u32 v11, v9, v10 // (op 1235) 


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_lshrrev_b32 v12, 4, v1 // (op 628) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 628) 

v_and_b32 v12, -16, v1 // (op 628) 
v_sub_u32 v14, v1, v12 // (op 628) 

v_lshlrev_b32 v12, 4, v14 // (op 628) 

v_mov_b32 v15, 0 // (op 628) convert
v_mov_b32 v14, v12 // (op 628) convert

v_mul_lo_u32 v17, v13, s3 // (op 628) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 628) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 628) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 628) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 628) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 628) most significant half



s_mul_i32 s1, s2, s7 // (op 630) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 630) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 630) least significant
s_mul_hi_u32 s15, s2, s6 // (op 630) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 630) most significant: sum
s_add_u32 s1, s1, s15 // (op 630) most significant: sum


s_mov_b32 s40, s4 // (op 630) reg expression
s_mov_b32 s41, s5 // (op 630) reg expression
s_mov_b32 s42, s0 // (op 630) reg expression
s_mov_b32 s43, 131072 // (op 630) call()

v_lshrrev_b32 v12, 4, v1 // (op 647) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 647) 

v_and_b32 v12, -16, v1 // (op 647) 
v_sub_u32 v14, v1, v12 // (op 647) 

v_lshlrev_b32 v12, 4, v14 // (op 647) 

v_mov_b32 v15, 0 // (op 647) convert
v_mov_b32 v14, v12 // (op 647) convert

v_mul_lo_u32 v19, v13, s3 // (op 647) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 647) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 647) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 647) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 647) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 647) most significant half



s_mul_i32 s1, s2, s7 // (op 649) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 649) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 649) least significant
s_mul_hi_u32 s15, s2, s6 // (op 649) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 649) most significant: sum
s_add_u32 s1, s1, s15 // (op 649) most significant: sum


s_mov_b32 s40, s4 // (op 649) reg expression
s_mov_b32 s41, s5 // (op 649) reg expression
s_mov_b32 s42, s0 // (op 649) reg expression
s_mov_b32 s43, 131072 // (op 649) call()

s_lshl_b32 s0, s17, 4 // (op 692) 
s_mov_b32 s15, 0 // (op 692) convert
s_mov_b32 s14, s0 // (op 692) convert

v_mul_lo_u32 v13, v1, s9 // (op 692) most significant: low of low * high
v_mul_hi_u32 v14, v1, s8 // (op 692) most significant: high of low * low
v_mul_lo_u32 v12, v1, s8 // (op 692) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 692) most significant: sum

v_mov_b32 v14, s14 // (op 692) 
v_mov_b32 v15, s15 // (op 692) 
v_add_co_u32 v12, s[0:1], v12, v14 // (op 692) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 692) most significant half




s_mul_i32 s1, s8, s13 // (op 694) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 694) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 694) least significant
s_mul_hi_u32 s15, s8, s12 // (op 694) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 694) most significant: sum
s_add_u32 s1, s1, s15 // (op 694) most significant: sum


s_mov_b32 s44, s10 // (op 694) reg expression
s_mov_b32 s45, s11 // (op 694) reg expression
s_mov_b32 s46, s0 // (op 694) reg expression
s_mov_b32 s47, 131072 // (op 694) call()

s_lshl_b32 s0, s17, 4 // (op 711) 
s_mov_b32 s15, 0 // (op 711) convert
s_mov_b32 s14, s0 // (op 711) convert

v_mul_lo_u32 v15, v1, s9 // (op 711) most significant: low of low * high
v_mul_hi_u32 v20, v1, s8 // (op 711) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 711) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 711) most significant: sum

v_mov_b32 v20, s14 // (op 711) 
v_mov_b32 v21, s15 // (op 711) 
v_add_co_u32 v14, s[0:1], v14, v20 // (op 711) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 711) most significant half




s_mul_i32 s1, s8, s13 // (op 713) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 713) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 713) least significant
s_mul_hi_u32 s15, s8, s12 // (op 713) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 713) most significant: sum
s_add_u32 s1, s1, s15 // (op 713) most significant: sum


s_mov_b32 s44, s10 // (op 713) reg expression
s_mov_b32 s45, s11 // (op 713) reg expression
s_mov_b32 s46, s0 // (op 713) reg expression
s_mov_b32 s47, 131072 // (op 713) call()

v_lshrrev_b32 v20, 3, v1 // (op 756) 
v_lshlrev_b32 v21, 1, v20 // (op 756) 

v_lshl_add_u32 v20, s16, 4, v21 // (op 756) 

v_and_b32 v21, -8, v1 // (op 756) 
v_sub_u32 v22, v1, v21 // (op 756) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v21, v20, s20 // (op 756) 
v_add_u32 v21, v21, v22 // (op 756) 


s_mul_i32 s1, s20, s7 // (op 758) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 758) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 758) least significant
s_mul_hi_u32 s15, s20, s6 // (op 758) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 758) most significant: sum
s_add_u32 s1, s1, s15 // (op 758) most significant: sum


s_mov_b32 s48, s22 // (op 758) reg expression
s_mov_b32 s49, s23 // (op 758) reg expression
s_mov_b32 s50, s0 // (op 758) reg expression
s_mov_b32 s51, 131072 // (op 758) call()

v_lshrrev_b32 v20, 3, v1 // (op 775) 
v_lshlrev_b32 v22, 1, v20 // (op 775) 

v_lshl_add_u32 v20, s16, 4, v22 // (op 775) 

v_and_b32 v22, -8, v1 // (op 775) 
v_sub_u32 v23, v1, v22 // (op 775) 

v_mul_lo_u32 v22, v20, s20 // (op 775) 
v_add_u32 v22, v22, v23 // (op 775) 


s_mul_i32 s1, s20, s7 // (op 777) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 777) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 777) least significant
s_mul_hi_u32 s15, s20, s6 // (op 777) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 777) most significant: sum
s_add_u32 s1, s1, s15 // (op 777) most significant: sum


s_mov_b32 s48, s22 // (op 777) reg expression
s_mov_b32 s49, s23 // (op 777) reg expression
s_mov_b32 s50, s0 // (op 777) reg expression
s_mov_b32 s51, 131072 // (op 777) call()

v_and_b32 v20, -16, v1 // (op 820) 
v_sub_u32 v23, v1, v20 // (op 820) 

v_lshl_add_u32 v20, s17, 4, v23 // (op 820) 

v_lshrrev_b32 v23, 4, v1 // (op 820) 
v_lshlrev_b32 v24, 1, v23 // (op 820) 

v_mul_lo_u32 v23, v24, s24 // (op 820) 
v_add_u32 v23, v23, v20 // (op 820) 


s_mul_i32 s1, s24, s29 // (op 822) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 822) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 822) least significant
s_mul_hi_u32 s15, s24, s28 // (op 822) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 822) most significant: sum
s_add_u32 s1, s1, s15 // (op 822) most significant: sum


s_mov_b32 s52, s26 // (op 822) reg expression
s_mov_b32 s53, s27 // (op 822) reg expression
s_mov_b32 s54, s0 // (op 822) reg expression
s_mov_b32 s55, 131072 // (op 822) call()

v_and_b32 v20, -16, v1 // (op 839) 
v_sub_u32 v24, v1, v20 // (op 839) 

v_lshl_add_u32 v20, s17, 4, v24 // (op 839) 

v_lshrrev_b32 v24, 4, v1 // (op 839) 
v_lshlrev_b32 v25, 1, v24 // (op 839) 

v_mul_lo_u32 v24, v25, s24 // (op 839) 
v_add_u32 v24, v24, v20 // (op 839) 


s_mul_i32 s1, s24, s29 // (op 841) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 841) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 841) least significant
s_mul_hi_u32 s15, s24, s28 // (op 841) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 841) most significant: sum
s_add_u32 s1, s1, s15 // (op 841) most significant: sum


s_mov_b32 s52, s26 // (op 841) reg expression
s_mov_b32 s53, s27 // (op 841) reg expression
s_mov_b32 s54, s0 // (op 841) reg expression
s_mov_b32 s55, 131072 // (op 841) call()

v_lshlrev_b32 v20, 4, v1 // (op 957) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 957) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 6 // (op 577) 
v_mov_b64 v[26:27], s[14:15] // (op 577) convert


v_mov_b64 v[28:29], v[12:13] // (op 577) reg expression
s_mov_b32 m0, s0 // (op 577) reg expression
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 577) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 577) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 577) most significant half

s_add_u32 m0, m0, 1024 // (op 577) 
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 577) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 577) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 577) most significant half

s_add_u32 m0, m0, 1024 // (op 577) 
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 577) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v26 // (op 577) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 577) most significant half

s_add_u32 m0, m0, 1024 // (op 577) 
buffer_load_dwordx4 v28, s[44:47], 0 offen offset: 0 lds // (op 577) Load value direct to lds




v_mov_b64 v[26:27], s[24:25] // (op 431) convert

v_mov_b32 v20, v23 // (op 431) reg expression
buffer_load_ubyte v28, v20, s[52:55], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v20, v20, v26 // (op 431) 
buffer_load_ubyte v29, v20, s[52:55], 0 offen offset: 0 // (op 431) Load value


v_and_b32 v20, -16, v1 // (op 905) 
v_sub_u32 v25, v1, v20 // (op 905) 

v_lshlrev_b32 v20, 4, v25 // (op 905) 

v_lshrrev_b32 v25, 4, v1 // (op 905) 
v_lshl_add_u32 v26, v25, 8, v20 // (op 905) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 905) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 2 // (op 582) 
v_mov_b64 v[26:27], s[14:15] // (op 582) convert


v_mov_b64 v[30:31], v[16:17] // (op 582) 
s_add_u32 s1, 4096, s0 // (op 582) 
s_mov_b32 m0, s1 // (op 582) reg expression
buffer_load_dwordx4 v30, s[40:43], 0 offen offset: 0 lds // (op 582) Load value direct to lds
v_add_co_u32 v30, s[14:15], v30, v26 // (op 582) least significant half
v_addc_co_u32 v31, s[14:15], 0, v31, s[14:15] // (op 582) most significant half

s_add_u32 m0, m0, 1024 // (op 582) 
buffer_load_dwordx4 v30, s[40:43], 0 offen offset: 0 lds // (op 582) Load value direct to lds
v_add_co_u32 v30, s[14:15], v30, v26 // (op 582) least significant half
v_addc_co_u32 v31, s[14:15], 0, v31, s[14:15] // (op 582) most significant half

s_add_u32 m0, m0, 1024 // (op 582) 
buffer_load_dwordx4 v30, s[40:43], 0 offen offset: 0 lds // (op 582) Load value direct to lds
v_add_co_u32 v30, s[14:15], v30, v26 // (op 582) least significant half
v_addc_co_u32 v31, s[14:15], 0, v31, s[14:15] // (op 582) most significant half

s_add_u32 m0, m0, 1024 // (op 582) 
buffer_load_dwordx4 v30, s[40:43], 0 offen offset: 0 lds // (op 582) Load value direct to lds





v_mov_b64 v[26:27], s[20:21] // (op 437) convert

v_mov_b32 v20, v21 // (op 437) 
buffer_load_ubyte v30, v20, s[48:51], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v20, v20, v26 // (op 437) 
buffer_load_ubyte v31, v20, s[48:51], 0 offen offset: 0 // (op 437) Load value


v_lshlrev_b32 v20, 4, v1 // (op 1175) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1175) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 6 // (op 587) 
v_mov_b64 v[26:27], s[14:15] // (op 587) convert


s_lshl_b64 s[14:15], s[8:9], 8 // (op 587) 
v_mov_b32 v20, s14 // (op 587) 
v_mov_b32 v25, s15 // (op 587) 
v_add_co_u32 v32, s[18:19], v12, v20 // (op 587) least significant half
v_addc_co_u32 v33, s[18:19], v13, v25, s[18:19] // (op 587) most significant half




s_add_u32 s1, 8192, s0 // (op 587) 
s_mov_b32 m0, s1 // (op 587) reg expression
buffer_load_dwordx4 v32, s[44:47], 0 offen offset: 0 lds // (op 587) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v26 // (op 587) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 587) most significant half

s_add_u32 m0, m0, 1024 // (op 587) 
buffer_load_dwordx4 v32, s[44:47], 0 offen offset: 0 lds // (op 587) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v26 // (op 587) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 587) most significant half

s_add_u32 m0, m0, 1024 // (op 587) 
buffer_load_dwordx4 v32, s[44:47], 0 offen offset: 0 lds // (op 587) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v26 // (op 587) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 587) most significant half

s_add_u32 m0, m0, 1024 // (op 587) 
buffer_load_dwordx4 v32, s[44:47], 0 offen offset: 0 lds // (op 587) Load value direct to lds






v_mov_b64 v[12:13], s[24:25] // (op 443) convert

s_lshl_b32 s0, s24, 3 // (op 443) 
v_add_u32 v20, v23, s0 // (op 443) 

buffer_load_ubyte v26, v20, s[52:55], 0 offen offset: 0 // (op 443) Load value
v_add_u32 v20, v20, v12 // (op 443) 
buffer_load_ubyte v27, v20, s[52:55], 0 offen offset: 0 // (op 443) Load value



v_and_b32 v20, -16, v1 // (op 1136) 
v_sub_u32 v23, v1, v20 // (op 1136) 

v_lshlrev_b32 v20, 4, v23 // (op 1136) 

v_lshrrev_b32 v23, 4, v1 // (op 1136) 
v_lshl_add_u32 v25, v23, 8, v20 // (op 1136) 


s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1136) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 2 // (op 589) 
v_mov_b64 v[12:13], s[14:15] // (op 589) convert


v_mov_b64 v[32:33], v[16:17] // (op 589) 
s_add_u32 s1, 12288, s0 // (op 589) 
s_mov_b32 m0, s1 // (op 589) reg expression
s_mov_b32 s14, 256 // (op 589) call()
buffer_load_dwordx4 v32, s[40:43], s14 offen offset: 0 lds // (op 589) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v12 // (op 589) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 589) most significant half

s_add_u32 m0, m0, 1024 // (op 589) 
buffer_load_dwordx4 v32, s[40:43], s14 offen offset: 0 lds // (op 589) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v12 // (op 589) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 589) most significant half

s_add_u32 m0, m0, 1024 // (op 589) 
buffer_load_dwordx4 v32, s[40:43], s14 offen offset: 0 lds // (op 589) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v12 // (op 589) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 589) most significant half

s_add_u32 m0, m0, 1024 // (op 589) 
buffer_load_dwordx4 v32, s[40:43], s14 offen offset: 0 lds // (op 589) Load value direct to lds







v_mov_b64 v[12:13], s[20:21] // (op 449) convert

v_mov_b32 v20, v21 // (op 449) 
buffer_load_ubyte v16, v20, s[48:51], 0 offen offset:8 // (op 449) Load value
v_add_u32 v20, v20, v12 // (op 449) 
buffer_load_ubyte v17, v20, s[48:51], 0 offen offset:8 // (op 449) Load value



v_and_b32 v23, -16, v1 // (op 1063) 
v_sub_u32 v25, v1, v23 // (op 1063) 

v_lshrrev_b32 v23, 4, v1 // (op 1063) 
v_lshl_add_u32 v12, v23, 5, v25 // (op 1063) 





s_waitcnt vmcnt(19)
ds_write_b8 v12, v28 offset:16384 // (op 452) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v12, v29 offset:16400 // (op 452) Store local data 

v_and_b32 v23, -8, v1 // (op 1010) 
v_sub_u32 v25, v1, v23 // (op 1010) 

v_lshrrev_b32 v23, 3, v1 // (op 1010) 
v_lshl_add_u32 v12, v23, 4, v25 // (op 1010) 





s_waitcnt vmcnt(13)
ds_write_b8 v12, v30 offset:16512 // (op 454) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v12, v31 offset:16520 // (op 454) Store local data 

s_barrier  // (op 1306) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:128 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:1024 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1152 // (op 474) Transpose load local data 



ds_read_b64_tr_b8 v[40:41], v0 offset:2048 // (op 477) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:2176 // (op 477) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3072 // (op 477) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:3200 // (op 477) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1340) 


ds_read_u8 v23, v7 offset:16384 // (op 486) Load local data 


ds_read_u8 v25, v7 offset:16448 // (op 489) Load local data 
s_barrier  // (op 1290) 



ds_read_b128 v[48:51], v3 offset:4096 // (op 468) Load local data 
ds_read_b128 v[52:55], v3 offset:4160 // (op 468) Load local data 



ds_read_b128 v[56:59], v3 offset:4224 // (op 471) Load local data 
ds_read_b128 v[60:63], v3 offset:4288 // (op 471) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1323) 



ds_read_u8 v12, v4 offset:16512 // (op 480) Load local data 



ds_read_u8 v13, v4 offset:16516 // (op 483) Load local data 
v_and_b32 v20, -16, v1 // (op 877) 
v_sub_u32 v21, v1, v20 // (op 877) 

v_lshlrev_b32 v20, 4, v21 // (op 877) 

v_lshrrev_b32 v21, 4, v1 // (op 877) 
v_lshl_add_u32 v64, v21, 8, v20 // (op 877) 


s_nop 0
v_readfirstlane_b32 s0, v64 // (op 877) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 4, v1 // (op 929) 
s_nop 0
v_readfirstlane_b32 s1, v20 // (op 929) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 981) 
v_sub_u32 v21, v1, v20 // (op 981) 

v_lshrrev_b32 v20, 3, v1 // (op 981) 
v_lshl_add_u32 v64, v20, 4, v21 // (op 981) 


v_and_b32 v20, -16, v1 // (op 1034) 
v_sub_u32 v21, v1, v20 // (op 1034) 

v_lshrrev_b32 v20, 4, v1 // (op 1034) 
v_lshl_add_u32 v65, v20, 5, v21 // (op 1034) 


v_and_b32 v20, -16, v1 // (op 1121) 
v_sub_u32 v21, v1, v20 // (op 1121) 

v_lshlrev_b32 v20, 4, v21 // (op 1121) 

v_lshrrev_b32 v21, 4, v1 // (op 1121) 
v_lshl_add_u32 v66, v21, 8, v20 // (op 1121) 


s_nop 0
v_readfirstlane_b32 s14, v66 // (op 1121) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v20, 4, v1 // (op 1160) 
s_nop 0
v_readfirstlane_b32 s15, v20 // (op 1160) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -8, v1 // (op 1199) 
v_sub_u32 v21, v1, v20 // (op 1199) 

v_lshrrev_b32 v20, 3, v1 // (op 1199) 
v_lshl_add_u32 v66, v20, 4, v21 // (op 1199) 


v_and_b32 v20, -16, v1 // (op 1223) 
v_sub_u32 v21, v1, v20 // (op 1223) 

v_lshrrev_b32 v20, 4, v1 // (op 1223) 
v_lshl_add_u32 v67, v20, 5, v21 // (op 1223) 


s_mov_b32 s18, 0 // (op 61) call()
s_ashr_i64 s[56:57], s[12:13], 8 // (op 60) 
s_mov_b32 s19, s56 // (op 60) convert

s_and_b32 s56, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s56 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60:
s_add_u32 s18, s18, 2 // (op 62) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 1309) 

s_lshl_b64 s[56:57], s[8:9], 6 // (op 549) 
v_mov_b64 v[20:21], s[56:57] // (op 549) convert


v_mov_b64 v[68:69], v[14:15] // (op 549) 
s_lshl_b64 s[56:57], s[8:9], 8 // (op 549) 
s_lshl_b64 s[58:59], s[56:57], 1 // (op 549) 

v_mov_b32 v70, s58 // (op 549) 
v_mov_b32 v71, s59 // (op 549) 
v_add_co_u32 v72, s[56:57], v68, v70 // (op 549) least significant half
v_addc_co_u32 v73, s[56:57], v69, v71, s[56:57] // (op 549) most significant half




s_mov_b32 m0, s1 // (op 549) reg expression
buffer_load_dwordx4 v72, s[44:47], 0 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v72, s[56:57], v72, v20 // (op 549) least significant half
v_addc_co_u32 v73, s[56:57], 0, v73, s[56:57] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
buffer_load_dwordx4 v72, s[44:47], 0 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v72, s[56:57], v72, v20 // (op 549) least significant half
v_addc_co_u32 v73, s[56:57], 0, v73, s[56:57] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
buffer_load_dwordx4 v72, s[44:47], 0 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v72, s[56:57], v72, v20 // (op 549) least significant half
v_addc_co_u32 v73, s[56:57], 0, v73, s[56:57] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v13, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 107) 
buffer_load_dwordx4 v72, s[44:47], 0 offen offset: 0 lds // (op 549) Load value direct to lds



v_mov_b64 v[20:21], s[24:25] // (op 25) convert

v_mov_b32 v70, v24 // (op 25) 
s_lshl_b32 s19, s24, 4 // (op 25) 
v_add_u32 v71, v70, s19 // (op 25) 

buffer_load_ubyte v28, v71, s[52:55], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v71, v71, v20 // (op 25) 
buffer_load_ubyte v29, v71, s[52:55], 0 offen offset: 0 // (op 25) Load value


s_barrier  // (op 1293) 

s_lshl_b64 s[56:57], s[2:3], 2 // (op 544) 
v_mov_b64 v[20:21], s[56:57] // (op 544) convert


v_mov_b64 v[72:73], v[18:19] // (op 544) 
v_mov_b64 v[74:75], v[72:73] // (op 544) 
s_add_u32 s19, 4096, s0 // (op 544) 
s_mov_b32 m0, s19 // (op 544) reg expression
s_mov_b32 s56, 512 // (op 544) call()
buffer_load_dwordx4 v74, s[40:43], s56 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v74, s[58:59], v74, v20 // (op 544) least significant half
v_addc_co_u32 v75, s[58:59], 0, v75, s[58:59] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v74, s[40:43], s56 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v74, s[58:59], v74, v20 // (op 544) least significant half
v_addc_co_u32 v75, s[58:59], 0, v75, s[58:59] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v74, s[40:43], s56 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v74, s[58:59], v74, v20 // (op 544) least significant half
v_addc_co_u32 v75, s[58:59], 0, v75, s[58:59] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v74, s[40:43], s56 offen offset: 0 lds // (op 544) Load value direct to lds





v_mov_b64 v[20:21], s[20:21] // (op 17) convert

v_mov_b32 v71, v22 // (op 17) 
v_mov_b32 v74, v71 // (op 17) 
buffer_load_ubyte v30, v74, s[48:51], 0 offen offset:16 // (op 17) Load value
v_add_u32 v74, v74, v20 // (op 17) 
buffer_load_ubyte v31, v74, s[48:51], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 1379) 



s_waitcnt vmcnt(19)
ds_write_b8 v67, v26 offset:16640 // (op 308) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v67, v27 offset:16656 // (op 308) Store local data 
s_barrier  // (op 1373) 



s_waitcnt vmcnt(13)
ds_write_b8 v66, v16 offset:16768 // (op 307) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v66, v17 offset:16776 // (op 307) Store local data 
s_barrier  // (op 1363) 



ds_read_b64_tr_b8 v[32:33], v5 offset:8192 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v5 offset:8320 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v5 offset:9216 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v5 offset:9344 // (op 291) Transpose load local data 



ds_read_b64_tr_b8 v[74:75], v5 offset:10240 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v5 offset:10368 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[78:79], v5 offset:11264 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v5 offset:11392 // (op 300) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1376) 


ds_read_u8 v23, v11 offset:16640 // (op 295) Load local data 


ds_read_u8 v20, v11 offset:16704 // (op 304) Load local data 
s_barrier  // (op 1357) 
s_barrier  // (op 1436) 



ds_read_b128 v[48:51], v6 offset:12288 // (op 289) Load local data 
ds_read_b128 v[52:55], v6 offset:12352 // (op 289) Load local data 



ds_read_b128 v[56:59], v6 offset:12416 // (op 298) Load local data 
ds_read_b128 v[60:63], v6 offset:12480 // (op 298) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1370) 



ds_read_u8 v12, v8 offset:16768 // (op 293) Load local data 



ds_read_u8 v13, v8 offset:16772 // (op 302) Load local data 
s_barrier  // (op 1432) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 296) 
s_barrier  // (op 1367) 

s_lshl_b64 s[56:57], s[8:9], 6 // (op 572) 
v_mov_b64 v[82:83], s[56:57] // (op 572) convert


s_lshl_b64 s[56:57], s[8:9], 8 // (op 572) 
s_mul_i32 s59, 3, s57 // (op 572) most significant: low of low * high
s_mul_i32 s58, 3, s56 // (op 572) least significant
s_mul_hi_u32 s19, 3, s56 // (op 572) most significant: high of low * low
s_add_u32 s59, s59, s19 // (op 572) most significant: sum


v_mov_b32 v21, s58 // (op 572) 
v_mov_b32 v84, s59 // (op 572) 
v_add_co_u32 v86, s[56:57], v68, v21 // (op 572) least significant half
v_addc_co_u32 v87, s[56:57], v69, v84, s[56:57] // (op 572) most significant half




s_add_u32 s19, 8192, s15 // (op 572) 
s_mov_b32 m0, s19 // (op 572) reg expression
buffer_load_dwordx4 v86, s[44:47], 0 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v86, s[56:57], v86, v82 // (op 572) least significant half
v_addc_co_u32 v87, s[56:57], 0, v87, s[56:57] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
buffer_load_dwordx4 v86, s[44:47], 0 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v86, s[56:57], v86, v82 // (op 572) least significant half
v_addc_co_u32 v87, s[56:57], 0, v87, s[56:57] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
buffer_load_dwordx4 v86, s[44:47], 0 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v86, s[56:57], v86, v82 // (op 572) least significant half
v_addc_co_u32 v87, s[56:57], 0, v87, s[56:57] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[74:81], a[0:3], v13, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 305) 
buffer_load_dwordx4 v86, s[44:47], 0 offen offset: 0 lds // (op 572) Load value direct to lds







v_mov_b64 v[20:21], s[24:25] // (op 311) convert

s_lshl_b32 s19, s24, 3 // (op 311) 
s_mul_i32 s56, 3, s19 // (op 311) 

v_add_u32 v68, v70, s56 // (op 311) 

buffer_load_ubyte v26, v68, s[52:55], 0 offen offset: 0 // (op 311) Load value
v_add_u32 v68, v68, v20 // (op 311) 
s_lshl_b64 s[56:57], s[8:9], 8 // (op 703) 
buffer_load_ubyte v27, v68, s[52:55], 0 offen offset: 0 // (op 311) Load value


s_lshl_b64 s[58:59], s[56:57], 1 // (op 703) 


v_mov_b32 v20, s58 // (op 703) 
s_barrier  // (op 1360) 
s_lshl_b32 s19, s24, 4 // (op 831) 

v_add_u32 v24, v24, s19 // (op 831) 

v_mov_b32 v21, s59 // (op 703) 
s_lshl_b64 s[56:57], s[2:3], 2 // (op 567) 
v_mov_b64 v[68:69], s[56:57] // (op 567) convert


v_mov_b64 v[74:75], v[72:73] // (op 567) 
s_add_u32 s19, 12288, s14 // (op 567) 
s_mov_b32 m0, s19 // (op 567) reg expression
s_mov_b32 s56, 768 // (op 567) call()
v_add_co_u32 v14, s[60:61], v14, v20 // (op 703) least significant half
v_addc_co_u32 v15, s[60:61], v15, v21, s[60:61] // (op 703) most significant half




buffer_load_dwordx4 v74, s[40:43], s56 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v74, s[58:59], v74, v68 // (op 567) least significant half
v_addc_co_u32 v75, s[58:59], 0, v75, s[58:59] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v74, s[40:43], s56 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v74, s[58:59], v74, v68 // (op 567) least significant half
v_addc_co_u32 v75, s[58:59], 0, v75, s[58:59] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v74, s[40:43], s56 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v74, s[58:59], v74, v68 // (op 567) least significant half
v_addc_co_u32 v75, s[58:59], 0, v75, s[58:59] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v74, s[40:43], s56 offen offset: 0 lds // (op 567) Load value direct to lds






v_add_co_u32 v18, vcc, 512, v18 // (op 639) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 639) most significant half
v_mov_b64 v[20:21], s[20:21] // (op 310) convert

v_mov_b32 v68, v71 // (op 310) 
buffer_load_ubyte v16, v68, s[48:51], 0 offen offset:24 // (op 310) Load value
v_add_u32 v68, v68, v20 // (op 310) 
buffer_load_ubyte v17, v68, s[48:51], 0 offen offset:24 // (op 310) Load value



s_barrier  // (op 1343) 
v_add_u32 v22, 16, v22 // (op 767) 



s_waitcnt vmcnt(19)
ds_write_b8 v65, v28 offset:16384 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v65, v29 offset:16400 // (op 30) Store local data 
s_barrier  // (op 1326) 



s_waitcnt vmcnt(13)
ds_write_b8 v64, v30 offset:16512 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v64, v31 offset:16520 // (op 21) Store local data 
s_barrier  // (op 1315) 



ds_read_b64_tr_b8 v[32:33], v0 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:128 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:1024 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:1152 // (op 77) Transpose load local data 



ds_read_b64_tr_b8 v[40:41], v0 offset:2048 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:2176 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3072 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v0 offset:3200 // (op 97) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1350) 


ds_read_u8 v23, v7 offset:16384 // (op 87) Load local data 


ds_read_u8 v25, v7 offset:16448 // (op 105) Load local data 
s_barrier  // (op 1299) 



ds_read_b128 v[48:51], v3 offset:4096 // (op 72) Load local data 
ds_read_b128 v[52:55], v3 offset:4160 // (op 72) Load local data 



ds_read_b128 v[56:59], v3 offset:4224 // (op 93) Load local data 
ds_read_b128 v[60:63], v3 offset:4288 // (op 93) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1333) 



ds_read_u8 v12, v4 offset:16512 // (op 82) Load local data 



ds_read_u8 v13, v4 offset:16516 // (op 101) Load local data 
s_ashr_i64 s[56:57], s[12:13], 8 // (op 60) 
s_mov_b32 s19, s56 // (op 60) convert

s_and_b32 s56, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s56 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60:















s_barrier  // (op 1392) 
v_lshrrev_b32 v24, 4, v1 // (op 602) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 602) 
v_lshl_add_u32 v14, s16, 4, v24 // (op 602) 

s_lshr_b32 s14, s0, 1 // (op 602) 

v_and_b32 v15, -16, v1 // (op 602) 
v_sub_u32 v24, v1, v15 // (op 602) 

v_lshlrev_b32 v15, 4, v24 // (op 602) 

v_lshl_add_u32 v24, s14, 9, v15 // (op 602) 


v_mov_b32 v19, 0 // (op 602) convert
v_mov_b32 v18, v24 // (op 602) convert

v_mul_lo_u32 v21, v14, s3 // (op 602) most significant: low of low * high
v_mul_hi_u32 v15, v14, s2 // (op 602) most significant: high of low * low
v_mul_lo_u32 v20, v14, s2 // (op 602) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 602) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 602) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 602) most significant half



s_mul_i32 s1, s2, s7 // (op 604) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 604) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 604) least significant
s_mul_hi_u32 s15, s2, s6 // (op 604) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 604) most significant: sum
s_add_u32 s1, s1, s15 // (op 604) most significant: sum


s_mov_b32 s40, s4 // (op 604) reg expression
s_mov_b32 s41, s5 // (op 604) reg expression
s_mov_b32 s42, s0 // (op 604) reg expression
s_mov_b32 s43, 131072 // (op 604) call()

s_ashr_i64 s[0:1], s[12:13], 8 // (op 669) 
s_lshr_b32 s14, s0, 1 // (op 669) 

v_lshl_add_u32 v22, s14, 9, v1 // (op 669) 

s_lshl_b32 s0, s17, 4 // (op 669) 
s_mov_b32 s15, 0 // (op 669) convert
s_mov_b32 s14, s0 // (op 669) convert

v_mul_lo_u32 v15, v22, s9 // (op 669) most significant: low of low * high
v_mul_hi_u32 v24, v22, s8 // (op 669) most significant: high of low * low
v_mul_lo_u32 v14, v22, s8 // (op 669) least significant: low of low * low
v_add_u32 v15, v15, v24 // (op 669) most significant: sum

v_mov_b32 v24, s14 // (op 669) 
v_mov_b32 v18, s15 // (op 669) 
v_add_co_u32 v14, s[0:1], v14, v24 // (op 669) least significant half
v_addc_co_u32 v15, s[0:1], v15, v18, s[0:1] // (op 669) most significant half





s_mul_i32 s1, s8, s13 // (op 671) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 671) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 671) least significant
s_mul_hi_u32 s15, s8, s12 // (op 671) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 671) most significant: sum
s_add_u32 s1, s1, s15 // (op 671) most significant: sum


s_mov_b32 s44, s10 // (op 671) reg expression
s_mov_b32 s45, s11 // (op 671) reg expression
s_mov_b32 s46, s0 // (op 671) reg expression
s_mov_b32 s47, 131072 // (op 671) call()

v_lshrrev_b32 v22, 3, v1 // (op 733) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 733) 
v_lshlrev_b32 v24, 1, v22 // (op 733) 

s_lshr_b32 s14, s0, 1 // (op 733) 

v_lshl_add_u32 v22, s16, 4, v24 // (op 733) 

v_and_b32 v24, -8, v1 // (op 733) 
v_sub_u32 v18, v1, v24 // (op 733) 

v_lshl_add_u32 v19, s14, 4, v18 // (op 733) 


v_mul_lo_u32 v18, v22, s20 // (op 733) 
v_add_u32 v18, v18, v19 // (op 733) 


s_mul_i32 s1, s20, s7 // (op 735) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 735) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 735) least significant
s_mul_hi_u32 s15, s20, s6 // (op 735) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 735) most significant: sum
s_add_u32 s1, s1, s15 // (op 735) most significant: sum


s_mov_b32 s48, s22 // (op 735) reg expression
s_mov_b32 s49, s23 // (op 735) reg expression
s_mov_b32 s50, s0 // (op 735) reg expression
s_mov_b32 s51, 131072 // (op 735) call()

v_lshrrev_b32 v19, 4, v1 // (op 797) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 797) 
v_lshlrev_b32 v22, 1, v19 // (op 797) 

s_lshr_b32 s14, s0, 1 // (op 797) 

v_and_b32 v19, -16, v1 // (op 797) 
v_lshl_add_u32 v24, s14, 4, v22 // (op 797) 


v_sub_u32 v22, v1, v19 // (op 797) 

v_lshl_add_u32 v19, s17, 4, v22 // (op 797) 

v_mul_lo_u32 v22, v24, s24 // (op 797) 
v_add_u32 v22, v22, v19 // (op 797) 


s_mul_i32 s1, s24, s29 // (op 799) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 799) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 799) least significant
s_mul_hi_u32 s15, s24, s28 // (op 799) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 799) most significant: sum
s_add_u32 s1, s1, s15 // (op 799) most significant: sum


s_mov_b32 s52, s26 // (op 799) reg expression
s_mov_b32 s53, s27 // (op 799) reg expression
s_mov_b32 s54, s0 // (op 799) reg expression
s_mov_b32 s55, 131072 // (op 799) call()

v_and_b32 v19, -16, v1 // (op 890) 
v_sub_u32 v24, v1, v19 // (op 890) 

v_lshlrev_b32 v19, 4, v24 // (op 890) 

v_lshrrev_b32 v24, 4, v1 // (op 890) 
v_lshl_add_u32 v28, v24, 8, v19 // (op 890) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 890) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 4, v1 // (op 942) 
s_nop 0
v_readfirstlane_b32 s1, v19 // (op 942) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 994) 
v_sub_u32 v24, v1, v19 // (op 994) 

v_lshrrev_b32 v19, 3, v1 // (op 994) 
v_lshl_add_u32 v28, v19, 4, v24 // (op 994) 


v_and_b32 v19, -16, v1 // (op 1047) 
v_sub_u32 v24, v1, v19 // (op 1047) 

v_lshrrev_b32 v19, 4, v1 // (op 1047) 
v_lshl_add_u32 v29, v19, 5, v24 // (op 1047) 


s_ashr_i64 s[14:15], s[12:13], 8 // (op 198) 
s_and_b32 s18, s14, 4294967294 // (op 198) 

s_ashr_i64 s[14:15], s[12:13], 8 // (op 197) 
s_mov_b32 s19, s14 // (op 197) convert

s_cmp_lt_u32 s18, s19 // (op 197) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197:
s_barrier  // (op 1296) 
s_add_u32 s18, s18, 1 // (op 199) 

s_lshl_b64 s[14:15], s[2:3], 2 // (op 554) 
v_mov_b64 v[12:13], s[14:15] // (op 554) convert


v_mov_b64 v[16:17], v[20:21] // (op 554) 
v_mov_b64 v[26:27], v[16:17] // (op 554) 
s_add_u32 s19, 4096, s0 // (op 554) 
s_mov_b32 m0, s19 // (op 554) reg expression
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 554) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 554) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 554) most significant half

s_add_u32 m0, m0, 1024 // (op 554) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 554) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 554) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 554) most significant half

s_add_u32 m0, m0, 1024 // (op 554) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 554) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 554) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 554) most significant half

s_add_u32 m0, m0, 1024 // (op 554) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 554) Load value direct to lds




v_add_co_u32 v20, vcc, 256, v20 // (op 592) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 592) most significant half
s_barrier  // (op 1312) 

s_lshl_b64 s[14:15], s[8:9], 6 // (op 560) 
v_mov_b64 v[12:13], s[14:15] // (op 560) convert


v_mov_b64 v[16:17], v[14:15] // (op 560) 
v_mov_b64 v[26:27], v[16:17] // (op 560) reg expression
s_mov_b32 m0, s1 // (op 560) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 560) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 560) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 560) most significant half

s_add_u32 m0, m0, 1024 // (op 560) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 560) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 560) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 560) most significant half

s_add_u32 m0, m0, 1024 // (op 560) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 560) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 560) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 560) most significant half

s_add_u32 m0, m0, 1024 // (op 560) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 560) Load value direct to lds



s_lshl_b64 s[14:15], s[8:9], 8 // (op 661) 
v_mov_b32 v19, s14 // (op 661) 
v_mov_b32 v12, s15 // (op 661) 
v_add_co_u32 v14, s[56:57], v14, v19 // (op 661) least significant half
v_addc_co_u32 v15, s[56:57], v15, v12, s[56:57] // (op 661) most significant half





v_mov_b64 v[12:13], s[20:21] // (op 226) convert

v_mov_b32 v19, v18 // (op 226) 
v_mov_b32 v16, v19 // (op 226) 
buffer_load_ubyte v26, v16, s[48:51], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v16, v16, v12 // (op 226) 
buffer_load_ubyte v27, v16, s[48:51], 0 offen offset: 0 // (op 226) Load value



v_add_u32 v18, 8, v18 // (op 725) 

s_barrier  // (op 1329) 
v_mov_b64 v[12:13], s[24:25] // (op 227) convert




v_mov_b32 v19, v22 // (op 227) 
v_mov_b32 v16, v19 // (op 227) reg expression
buffer_load_ubyte v24, v16, s[52:55], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v16, v16, v12 // (op 227) 
buffer_load_ubyte v25, v16, s[52:55], 0 offen offset: 0 // (op 227) Load value



s_lshl_b32 s19, s24, 3 // (op 789) 
v_add_u32 v22, v22, s19 // (op 789) 

s_waitcnt vmcnt(3)
ds_write_b8 v28, v26 offset:16512 // (op 223) Store local data 
s_barrier  // (op 1302) 
s_waitcnt vmcnt(2)
ds_write_b8 v28, v27 offset:16520 // (op 223) Store local data 




ds_read_b128 v[30:33], v3 offset:4096 // (op 205) Load local data 
s_barrier  // (op 1346) 
ds_read_b128 v[34:37], v3 offset:4160 // (op 205) Load local data 



s_barrier  // (op 1319) 
s_waitcnt vmcnt(1)
ds_write_b8 v29, v24 offset:16384 // (op 224) Store local data 



ds_read_b64_tr_b8 v[38:39], v0 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:128 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:1024 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:1152 // (op 207) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v29, v25 offset:16400 // (op 224) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1336) 



ds_read_u8 v19, v4 offset:16512 // (op 209) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1353) 


ds_read_u8 v12, v7 offset:16384 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v19, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 212) 







ds_read_b128 v[30:33], v3 offset:4224 // (op 214) Load local data 
ds_read_b128 v[34:37], v3 offset:4288 // (op 214) Load local data 



ds_read_b64_tr_b8 v[38:39], v0 offset:2048 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:2176 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:3072 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v0 offset:3200 // (op 216) Transpose load local data 



ds_read_u8 v19, v4 offset:16516 // (op 218) Load local data 


ds_read_u8 v12, v7 offset:16448 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v19, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 221) 




s_ashr_i64 s[14:15], s[12:13], 8 // (op 197) 
s_mov_b32 s19, s14 // (op 197) convert

s_cmp_lt_u32 s18, s19 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 if true)
RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197:
























s_barrier  // (op 1422) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

v_lshrrev_b32 v4, 16, v12 // (op 56) 
v_lshrrev_b32 v5, 16, v13 // (op 56) 
v_lshrrev_b32 v6, 16, v14 // (op 56) 
v_lshrrev_b32 v7, 16, v15 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 1089) 
v_and_b32 v3, 63, v1 // (op 1089) 
v_lshrrev_b32 v8, 4, v3 // (op 1089) 
v_lshlrev_b32 v11, 2, v8 // (op 1089) 

v_lshl_add_u32 v8, v0, 4, v11 // (op 1089) 


v_lshl_add_u32 v0, s16, 4, v8 // (op 1089) 

v_lshrrev_b32 v8, 2, v3 // (op 1089) 

v_and_b32 v3, 3, v8 // (op 1089) 

v_and_b32 v8, 3, v1 // (op 1089) 
v_lshl_add_u32 v11, v3, 2, v8 // (op 1089) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 1089) 

v_mul_lo_u32 v8, v3, s30 // (op 1089) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 1089) 


s_mul_i32 s1, s30, s35 // (op 1091) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 1091) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 1091) least significant
s_mul_hi_u32 s3, s30, s34 // (op 1091) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1091) most significant: sum
s_add_u32 s1, s1, s3 // (op 1091) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1091) 

s_mov_b32 s4, s32 // (op 1091) reg expression
s_mov_b32 s5, s33 // (op 1091) reg expression
s_mov_b32 s6, s2 // (op 1091) reg expression
s_mov_b32 s7, 131072 // (op 1091) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[4:7], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157:
v_lshrrev_b32 v0, 6, v1 // (op 860) 
v_and_b32 v3, 63, v1 // (op 860) 
v_lshrrev_b32 v4, 4, v3 // (op 860) 
v_lshlrev_b32 v5, 2, v4 // (op 860) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 860) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 860) 

v_lshrrev_b32 v4, 2, v3 // (op 860) 

v_and_b32 v3, 3, v4 // (op 860) 

v_and_b32 v4, 3, v1 // (op 860) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 860) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 860) 

v_mul_lo_u32 v4, v3, s36 // (op 860) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 860) 


s_mul_i32 s1, s36, s35 // (op 862) most significant: low of low * high
s_mul_i32 s2, s37, s34 // (op 862) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 862) least significant
s_mul_hi_u32 s3, s36, s34 // (op 862) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 862) most significant: sum
s_add_u32 s1, s1, s3 // (op 862) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 862) 

s_mov_b32 s8, s38 // (op 862) reg expression
s_mov_b32 s9, s39 // (op 862) reg expression
s_mov_b32 s10, s2 // (op 862) reg expression
s_mov_b32 s11, 131072 // (op 862) call()





buffer_load_dwordx2 v[4:5], v3, s[8:11], 0 offen // (op 170) Load value



v_accvgpr_read v12, a0 // (op 166) 
v_accvgpr_read v13, a1 // (op 166) 
v_accvgpr_read v14, a2 // (op 166) 
v_accvgpr_read v15, a3 // (op 166) 
v_mul_f32 v16, v10, v12 // (op 166) 
v_mul_f32 v17, v10, v13 // (op 166) 
v_mul_f32 v18, v10, v14 // (op 166) 
v_mul_f32 v19, v10, v15 // (op 166) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 171) 
v_lshrrev_b32 v7, 16, v4 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v12, v9, v6 // (op 171) 
v_mul_f32 v13, v9, v7 // (op 171) 
v_and_b32 v6, 65535, v5 // (op 171) 
v_lshrrev_b32 v7, 16, v5 // (op 171) 
v_lshlrev_b32 v6, 16, v6 // (op 171) 
v_lshlrev_b32 v7, 16, v7 // (op 171) 
v_mul_f32 v14, v9, v6 // (op 171) 
v_mul_f32 v15, v9, v7 // (op 171) 


v_add_f32 v4, v12, v16 // (op 167) 
v_add_f32 v5, v13, v17 // (op 167) 
v_add_f32 v6, v14, v18 // (op 167) 
v_add_f32 v7, v15, v19 // (op 167) 


v_lshrrev_b32 v12, 16, v4 // (op 168) 
v_lshrrev_b32 v13, 16, v5 // (op 168) 
v_lshrrev_b32 v14, 16, v6 // (op 168) 
v_lshrrev_b32 v15, 16, v7 // (op 168) 

v_lshrrev_b32 v0, 6, v1 // (op 1106) 
v_and_b32 v11, 63, v1 // (op 1106) 
v_lshrrev_b32 v3, 4, v11 // (op 1106) 
v_lshlrev_b32 v4, 2, v3 // (op 1106) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1106) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 1106) 

v_lshrrev_b32 v3, 2, v11 // (op 1106) 

v_and_b32 v11, 3, v3 // (op 1106) 

v_and_b32 v3, 3, v1 // (op 1106) 
v_lshl_add_u32 v4, v11, 2, v3 // (op 1106) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 1106) 

v_mul_lo_u32 v11, v3, s30 // (op 1106) 

v_add_lshl_u32 v3, v0, v11, 1 // (op 1106) 


s_mul_i32 s1, s30, s35 // (op 1108) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 1108) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 1108) least significant
s_mul_hi_u32 s3, s30, s34 // (op 1108) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1108) most significant: sum
s_add_u32 s1, s1, s3 // (op 1108) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1108) 

s_mov_b32 s4, s32 // (op 1108) reg expression
s_mov_b32 s5, s33 // (op 1108) reg expression
s_mov_b32 s6, s2 // (op 1108) reg expression
s_mov_b32 s7, 131072 // (op 1108) call()


v_pack_b32_f16 v4, v12, v13 // (op 169) 
v_pack_b32_f16 v5, v14, v15 // (op 169) 


buffer_store_dwordx2 v[4:5], v3, s[4:7], 0 offen // (op 169) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157:



s_endpgm  // End of RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 92
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
  .amdhsa_accum_offset 88
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
    DestDataType: b
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
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [16, 16, 256]
    threads: [64, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_TT_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 62
    .vgpr_count: 88
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_4
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
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 120
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 128
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

