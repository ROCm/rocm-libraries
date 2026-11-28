// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 917) 
v_and_b32 v3, 63, v1 // (op 917) 
v_sub_u32 v4, v3, v0 // (op 917) 

v_lshrrev_b32 v3, 1, v4 // (op 917) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 917) 


v_and_b32 v0, 1, v4 // (op 917) 

v_lshlrev_b32 v3, 3, v0 // (op 917) 

v_lshrrev_b32 v0, 6, v1 // (op 917) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 917) 


v_add_u32 v0, v5, v4 // (op 917) 


v_and_b32 v3, 15, v1 // (op 969) 
v_and_b32 v4, 48, v1 // (op 969) 
v_lshl_add_u32 v5, v3, 8, v4 // (op 969) 


v_lshrrev_b32 v3, 6, v1 // (op 1022) 
v_and_b32 v4, 15, v1 // (op 1022) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 1022) 


v_and_b32 v3, 48, v1 // (op 1022) 
v_add_u32 v4, v3, v6 // (op 1022) 


v_and_b32 v3, 15, v1 // (op 1075) 
v_and_b32 v6, 63, v1 // (op 1075) 
v_lshrrev_b32 v7, 4, v6 // (op 1075) 

v_lshl_add_u32 v6, v3, 3, v7 // (op 1075) 


v_and_b32 v3, 48, v1 // (op 1148) 
v_and_b32 v7, 63, v1 // (op 1148) 
v_sub_u32 v8, v7, v3 // (op 1148) 

v_lshrrev_b32 v7, 1, v8 // (op 1148) 
v_add_lshl_u32 v9, v3, v7, 4 // (op 1148) 


v_and_b32 v3, 1, v8 // (op 1148) 

v_lshlrev_b32 v7, 3, v3 // (op 1148) 

v_lshrrev_b32 v3, 6, v1 // (op 1148) 
v_lshl_add_u32 v8, v3, 4, v7 // (op 1148) 


v_add_u32 v3, v9, v8 // (op 1148) 


v_and_b32 v7, 15, v1 // (op 1187) 
v_and_b32 v8, 48, v1 // (op 1187) 
v_lshl_add_u32 v9, v7, 8, v8 // (op 1187) 


v_lshrrev_b32 v7, 6, v1 // (op 1211) 
v_and_b32 v8, 15, v1 // (op 1211) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 1211) 


v_and_b32 v7, 48, v1 // (op 1211) 
v_add_u32 v8, v7, v10 // (op 1211) 


v_and_b32 v7, 15, v1 // (op 1235) 
v_and_b32 v10, 63, v1 // (op 1235) 
v_lshrrev_b32 v11, 4, v10 // (op 1235) 

v_lshl_add_u32 v10, v7, 3, v11 // (op 1235) 


v_mov_b32 v7, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_lshl_b32 s0, s16, 4 // (op 628) 
v_mul_lo_u32 v13, v1, s3 // (op 628) most significant: low of low * high
s_mov_b32 s15, 0 // (op 628) convert
s_mov_b32 s14, s0 // (op 628) convert

v_mul_hi_u32 v14, v1, s2 // (op 628) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 628) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 628) most significant: sum

v_mov_b32 v14, s14 // (op 628) 
v_mov_b32 v15, s15 // (op 628) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 628) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 628) most significant half





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

s_lshl_b32 s0, s16, 4 // (op 647) 
v_mul_lo_u32 v13, v1, s3 // (op 647) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 647) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 647) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 647) most significant: sum

s_mov_b32 s15, 0 // (op 647) convert
s_mov_b32 s14, s0 // (op 647) convert

v_mov_b32 v14, s14 // (op 647) 
v_mov_b32 v15, s15 // (op 647) 
v_add_co_u32 v18, s[0:1], v14, v12 // (op 647) least significant half
v_addc_co_u32 v19, s[0:1], v15, v13, s[0:1] // (op 647) most significant half





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

v_and_b32 v12, -16, v1 // (op 692) 
v_sub_u32 v13, v1, v12 // (op 692) 

v_lshlrev_b32 v12, 4, v13 // (op 692) 

v_mov_b32 v15, 0 // (op 692) convert
v_mov_b32 v14, v12 // (op 692) convert

v_lshrrev_b32 v12, 4, v1 // (op 692) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 692) 

v_mul_lo_u32 v21, v13, s9 // (op 692) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 692) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 692) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 692) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 692) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 692) most significant half



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

v_and_b32 v14, -16, v1 // (op 711) 
v_sub_u32 v15, v1, v14 // (op 711) 

v_lshlrev_b32 v14, 4, v15 // (op 711) 

v_mov_b32 v21, 0 // (op 711) convert
v_mov_b32 v20, v14 // (op 711) convert

v_lshrrev_b32 v14, 4, v1 // (op 711) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 711) 

v_mul_lo_u32 v23, v15, s9 // (op 711) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 711) most significant: high of low * low
v_mul_lo_u32 v22, v15, s8 // (op 711) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 711) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 711) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 711) most significant half



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

v_and_b32 v20, -16, v1 // (op 756) 
v_sub_u32 v21, v1, v20 // (op 756) 

v_lshl_add_u32 v20, s16, 4, v21 // (op 756) 

v_lshrrev_b32 v21, 4, v1 // (op 756) 
v_lshlrev_b32 v22, 1, v21 // (op 756) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v21, v22, s20 // (op 756) 

v_add_u32 v22, v20, v21 // (op 756) 


s_mul_i32 s1, s20, s25 // (op 758) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 758) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 758) least significant
s_mul_hi_u32 s15, s20, s24 // (op 758) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 758) most significant: sum
s_add_u32 s1, s1, s15 // (op 758) most significant: sum


s_mov_b32 s48, s22 // (op 758) reg expression
s_mov_b32 s49, s23 // (op 758) reg expression
s_mov_b32 s50, s0 // (op 758) reg expression
s_mov_b32 s51, 131072 // (op 758) call()

v_and_b32 v20, -16, v1 // (op 775) 
v_sub_u32 v21, v1, v20 // (op 775) 

v_lshl_add_u32 v20, s16, 4, v21 // (op 775) 

v_lshrrev_b32 v21, 4, v1 // (op 775) 
v_lshlrev_b32 v23, 1, v21 // (op 775) 

v_mul_lo_u32 v21, v23, s20 // (op 775) 

v_add_u32 v23, v20, v21 // (op 775) 


s_mul_i32 s1, s20, s25 // (op 777) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 777) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 777) least significant
s_mul_hi_u32 s15, s20, s24 // (op 777) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 777) most significant: sum
s_add_u32 s1, s1, s15 // (op 777) most significant: sum


s_mov_b32 s48, s22 // (op 777) reg expression
s_mov_b32 s49, s23 // (op 777) reg expression
s_mov_b32 s50, s0 // (op 777) reg expression
s_mov_b32 s51, 131072 // (op 777) call()

v_and_b32 v20, -8, v1 // (op 820) 
v_sub_u32 v21, v1, v20 // (op 820) 

v_lshrrev_b32 v20, 3, v1 // (op 820) 
v_lshlrev_b32 v24, 1, v20 // (op 820) 

v_lshl_add_u32 v20, s17, 4, v24 // (op 820) 

v_mul_lo_u32 v24, v20, s26 // (op 820) 

v_add_u32 v20, v21, v24 // (op 820) 


s_mul_i32 s1, s26, s13 // (op 822) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 822) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 822) least significant
s_mul_hi_u32 s15, s26, s12 // (op 822) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 822) most significant: sum
s_add_u32 s1, s1, s15 // (op 822) most significant: sum


s_mov_b32 s52, s28 // (op 822) reg expression
s_mov_b32 s53, s29 // (op 822) reg expression
s_mov_b32 s54, s0 // (op 822) reg expression
s_mov_b32 s55, 131072 // (op 822) call()

v_and_b32 v21, -8, v1 // (op 839) 
v_sub_u32 v24, v1, v21 // (op 839) 

v_lshrrev_b32 v21, 3, v1 // (op 839) 
v_lshlrev_b32 v25, 1, v21 // (op 839) 

v_lshl_add_u32 v21, s17, 4, v25 // (op 839) 

v_mul_lo_u32 v25, v21, s26 // (op 839) 

v_add_u32 v21, v24, v25 // (op 839) 


s_mul_i32 s1, s26, s13 // (op 841) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 841) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 841) least significant
s_mul_hi_u32 s15, s26, s12 // (op 841) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 841) most significant: sum
s_add_u32 s1, s1, s15 // (op 841) most significant: sum


s_mov_b32 s52, s28 // (op 841) reg expression
s_mov_b32 s53, s29 // (op 841) reg expression
s_mov_b32 s54, s0 // (op 841) reg expression
s_mov_b32 s55, 131072 // (op 841) call()

v_and_b32 v24, -16, v1 // (op 957) 
v_sub_u32 v25, v1, v24 // (op 957) 

v_lshlrev_b32 v24, 4, v25 // (op 957) 

v_lshrrev_b32 v25, 4, v1 // (op 957) 
v_lshl_add_u32 v26, v25, 8, v24 // (op 957) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 957) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 2 // (op 577) 
v_mov_b64 v[24:25], s[14:15] // (op 577) convert


v_mov_b64 v[26:27], v[12:13] // (op 577) 
s_mov_b32 m0, s0 // (op 577) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 577) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 577) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 577) most significant half

s_add_u32 m0, m0, 1024 // (op 577) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 577) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 577) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 577) most significant half

s_add_u32 m0, m0, 1024 // (op 577) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 577) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 577) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 577) most significant half

s_add_u32 m0, m0, 1024 // (op 577) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 577) Load value direct to lds




v_mov_b64 v[24:25], s[26:27] // (op 431) convert

v_mov_b32 v26, v20 // (op 431) 
buffer_load_ubyte v28, v26, s[52:55], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v26, v26, v24 // (op 431) 
buffer_load_ubyte v29, v26, s[52:55], 0 offen offset: 0 // (op 431) Load value


v_lshlrev_b32 v24, 4, v1 // (op 905) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 905) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 582) 
v_mov_b64 v[24:25], s[14:15] // (op 582) convert


v_mov_b64 v[26:27], v[16:17] // (op 582) reg expression
s_add_u32 s1, 4096, s0 // (op 582) 
s_mov_b32 m0, s1 // (op 582) reg expression
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 582) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 582) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 582) most significant half

s_add_u32 m0, m0, 1024 // (op 582) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 582) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 582) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 582) most significant half

s_add_u32 m0, m0, 1024 // (op 582) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 582) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 582) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 582) most significant half

s_add_u32 m0, m0, 1024 // (op 582) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 582) Load value direct to lds





v_mov_b64 v[24:25], s[20:21] // (op 437) convert

v_mov_b32 v26, v22 // (op 437) reg expression
buffer_load_ubyte v30, v26, s[48:51], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v26, v26, v24 // (op 437) 
buffer_load_ubyte v31, v26, s[48:51], 0 offen offset: 0 // (op 437) Load value


v_and_b32 v24, -16, v1 // (op 1175) 
v_sub_u32 v25, v1, v24 // (op 1175) 

v_lshlrev_b32 v24, 4, v25 // (op 1175) 

v_lshrrev_b32 v25, 4, v1 // (op 1175) 
v_lshl_add_u32 v26, v25, 8, v24 // (op 1175) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1175) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 2 // (op 587) 
v_mov_b64 v[24:25], s[14:15] // (op 587) convert


v_mov_b64 v[26:27], v[12:13] // (op 587) 
s_add_u32 s1, 8192, s0 // (op 587) 
s_mov_b32 m0, s1 // (op 587) reg expression
s_mov_b32 s14, 256 // (op 587) call()
buffer_load_dwordx4 v26, s[44:47], s14 offen offset: 0 lds // (op 587) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 587) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 587) most significant half

s_add_u32 m0, m0, 1024 // (op 587) 
buffer_load_dwordx4 v26, s[44:47], s14 offen offset: 0 lds // (op 587) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 587) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 587) most significant half

s_add_u32 m0, m0, 1024 // (op 587) 
buffer_load_dwordx4 v26, s[44:47], s14 offen offset: 0 lds // (op 587) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 587) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 587) most significant half

s_add_u32 m0, m0, 1024 // (op 587) 
buffer_load_dwordx4 v26, s[44:47], s14 offen offset: 0 lds // (op 587) Load value direct to lds







v_mov_b64 v[12:13], s[26:27] // (op 443) convert

v_mov_b32 v24, v20 // (op 443) 
buffer_load_ubyte v26, v24, s[52:55], 0 offen offset:8 // (op 443) Load value
v_add_u32 v24, v24, v12 // (op 443) 
buffer_load_ubyte v27, v24, s[52:55], 0 offen offset:8 // (op 443) Load value



v_lshlrev_b32 v20, 4, v1 // (op 1136) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1136) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 589) 
v_mov_b64 v[12:13], s[14:15] // (op 589) convert


s_lshl_b64 s[14:15], s[2:3], 8 // (op 589) 
v_mov_b32 v20, s14 // (op 589) 
v_mov_b32 v24, s15 // (op 589) 
v_add_co_u32 v32, s[18:19], v16, v20 // (op 589) least significant half
v_addc_co_u32 v33, s[18:19], v17, v24, s[18:19] // (op 589) most significant half




s_add_u32 s1, 12288, s0 // (op 589) 
s_mov_b32 m0, s1 // (op 589) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 589) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v12 // (op 589) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 589) most significant half

s_add_u32 m0, m0, 1024 // (op 589) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 589) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v12 // (op 589) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 589) most significant half

s_add_u32 m0, m0, 1024 // (op 589) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 589) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v12 // (op 589) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 589) most significant half

s_add_u32 m0, m0, 1024 // (op 589) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 589) Load value direct to lds






v_mov_b64 v[12:13], s[20:21] // (op 449) convert

s_lshl_b32 s0, s20, 3 // (op 449) 
v_add_u32 v20, v22, s0 // (op 449) 

buffer_load_ubyte v16, v20, s[48:51], 0 offen offset: 0 // (op 449) Load value
v_add_u32 v20, v20, v12 // (op 449) 
buffer_load_ubyte v17, v20, s[48:51], 0 offen offset: 0 // (op 449) Load value



v_and_b32 v20, -8, v1 // (op 1063) 
v_sub_u32 v22, v1, v20 // (op 1063) 

v_lshrrev_b32 v20, 3, v1 // (op 1063) 
v_lshl_add_u32 v12, v20, 4, v22 // (op 1063) 





s_waitcnt vmcnt(19)
ds_write_b8 v12, v28 offset:16384 // (op 452) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v12, v29 offset:16392 // (op 452) Store local data 

v_and_b32 v20, -16, v1 // (op 1010) 
v_sub_u32 v22, v1, v20 // (op 1010) 

v_lshrrev_b32 v20, 4, v1 // (op 1010) 
v_lshl_add_u32 v12, v20, 5, v22 // (op 1010) 





s_waitcnt vmcnt(13)
ds_write_b8 v12, v30 offset:16512 // (op 454) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v12, v31 offset:16528 // (op 454) Store local data 

s_barrier  // (op 1306) 



ds_read_b128 v[32:35], v5 // (op 474) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 474) Load local data 



ds_read_b128 v[40:43], v5 offset:128 // (op 477) Load local data 
ds_read_b128 v[44:47], v5 offset:192 // (op 477) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1340) 



ds_read_u8 v20, v6 offset:16384 // (op 486) Load local data 



ds_read_u8 v22, v6 offset:16388 // (op 489) Load local data 
s_barrier  // (op 1290) 



ds_read_b64_tr_b8 v[48:49], v0 offset:4096 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:4224 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5120 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:5248 // (op 468) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:6144 // (op 471) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:6272 // (op 471) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:7168 // (op 471) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:7296 // (op 471) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1323) 


ds_read_u8 v12, v4 offset:16512 // (op 480) Load local data 


ds_read_u8 v13, v4 offset:16576 // (op 483) Load local data 
v_lshlrev_b32 v24, 4, v1 // (op 877) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 877) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 929) 
v_sub_u32 v25, v1, v24 // (op 929) 

v_lshlrev_b32 v24, 4, v25 // (op 929) 

v_lshrrev_b32 v25, 4, v1 // (op 929) 
v_lshl_add_u32 v64, v25, 8, v24 // (op 929) 


s_nop 0
v_readfirstlane_b32 s1, v64 // (op 929) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 981) 
v_sub_u32 v25, v1, v24 // (op 981) 

v_lshrrev_b32 v24, 4, v1 // (op 981) 
v_lshl_add_u32 v64, v24, 5, v25 // (op 981) 


v_and_b32 v24, -8, v1 // (op 1034) 
v_sub_u32 v25, v1, v24 // (op 1034) 

v_lshrrev_b32 v24, 3, v1 // (op 1034) 
v_lshl_add_u32 v65, v24, 4, v25 // (op 1034) 


v_lshlrev_b32 v24, 4, v1 // (op 1121) 
s_nop 0
v_readfirstlane_b32 s14, v24 // (op 1121) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 1160) 
v_sub_u32 v25, v1, v24 // (op 1160) 

v_lshlrev_b32 v24, 4, v25 // (op 1160) 

v_lshrrev_b32 v25, 4, v1 // (op 1160) 
v_lshl_add_u32 v66, v25, 8, v24 // (op 1160) 


s_nop 0
v_readfirstlane_b32 s15, v66 // (op 1160) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 1199) 
v_sub_u32 v25, v1, v24 // (op 1199) 

v_lshrrev_b32 v24, 4, v1 // (op 1199) 
v_lshl_add_u32 v66, v24, 5, v25 // (op 1199) 


v_and_b32 v24, -8, v1 // (op 1223) 
v_sub_u32 v25, v1, v24 // (op 1223) 

v_lshrrev_b32 v24, 3, v1 // (op 1223) 
v_lshl_add_u32 v67, v24, 4, v25 // (op 1223) 


s_mov_b32 s18, 0 // (op 61) call()
s_ashr_i64 s[38:39], s[6:7], 8 // (op 60) 
s_mov_b32 s19, s38 // (op 60) convert

s_and_b32 s38, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s38 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60:
s_add_u32 s18, s18, 2 // (op 62) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 90) 
s_barrier  // (op 1309) 

s_lshl_b64 s[38:39], s[8:9], 2 // (op 549) 
v_mov_b64 v[24:25], s[38:39] // (op 549) convert


v_mov_b64 v[68:69], v[14:15] // (op 549) 
v_mov_b64 v[70:71], v[68:69] // (op 549) 
s_mov_b32 m0, s1 // (op 549) reg expression
s_mov_b32 s19, 512 // (op 549) call()
buffer_load_dwordx4 v70, s[44:47], s19 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v70, s[38:39], v70, v24 // (op 549) least significant half
v_addc_co_u32 v71, s[38:39], 0, v71, s[38:39] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
buffer_load_dwordx4 v70, s[44:47], s19 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v70, s[38:39], v70, v24 // (op 549) least significant half
v_addc_co_u32 v71, s[38:39], 0, v71, s[38:39] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
buffer_load_dwordx4 v70, s[44:47], s19 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v70, s[38:39], v70, v24 // (op 549) least significant half
v_addc_co_u32 v71, s[38:39], 0, v71, s[38:39] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 107) 
buffer_load_dwordx4 v70, s[44:47], s19 offen offset: 0 lds // (op 549) Load value direct to lds




v_mov_b64 v[24:25], s[26:27] // (op 25) convert

v_mov_b32 v70, v21 // (op 25) 
v_mov_b32 v71, v70 // (op 25) 
buffer_load_ubyte v28, v71, s[52:55], 0 offen offset:16 // (op 25) Load value
v_add_u32 v71, v71, v24 // (op 25) 
buffer_load_ubyte v29, v71, s[52:55], 0 offen offset:16 // (op 25) Load value


s_barrier  // (op 1293) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 544) 
v_mov_b64 v[24:25], s[38:39] // (op 544) convert


v_mov_b64 v[72:73], v[18:19] // (op 544) 
s_lshl_b64 s[38:39], s[2:3], 8 // (op 544) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 544) 

v_mov_b32 v71, s56 // (op 544) 
v_mov_b32 v74, s57 // (op 544) 
v_add_co_u32 v76, s[38:39], v72, v71 // (op 544) least significant half
v_addc_co_u32 v77, s[38:39], v73, v74, s[38:39] // (op 544) most significant half




s_add_u32 s19, 4096, s0 // (op 544) 
s_mov_b32 m0, s19 // (op 544) reg expression
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v24 // (op 544) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v24 // (op 544) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v24 // (op 544) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 544) Load value direct to lds




v_mov_b64 v[24:25], s[20:21] // (op 17) convert

v_mov_b32 v71, v23 // (op 17) 
s_lshl_b32 s19, s20, 4 // (op 17) 
v_add_u32 v74, v71, s19 // (op 17) 

buffer_load_ubyte v30, v74, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v74, v74, v24 // (op 17) 
buffer_load_ubyte v31, v74, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1379) 



s_waitcnt vmcnt(19)
ds_write_b8 v67, v26 offset:16640 // (op 308) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v67, v27 offset:16648 // (op 308) Store local data 
s_barrier  // (op 1373) 



s_waitcnt vmcnt(13)
ds_write_b8 v66, v16 offset:16768 // (op 307) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v66, v17 offset:16784 // (op 307) Store local data 
s_barrier  // (op 1363) 



ds_read_b128 v[32:35], v9 offset:8192 // (op 291) Load local data 
ds_read_b128 v[36:39], v9 offset:8256 // (op 291) Load local data 



ds_read_b128 v[74:77], v9 offset:8320 // (op 300) Load local data 
ds_read_b128 v[78:81], v9 offset:8384 // (op 300) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1376) 



ds_read_u8 v20, v10 offset:16640 // (op 295) Load local data 



ds_read_u8 v24, v10 offset:16644 // (op 304) Load local data 
s_barrier  // (op 1357) 
s_barrier  // (op 1436) 



ds_read_b64_tr_b8 v[48:49], v3 offset:12288 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v3 offset:12416 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v3 offset:13312 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v3 offset:13440 // (op 289) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v3 offset:14336 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v3 offset:14464 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v3 offset:15360 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v3 offset:15488 // (op 298) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1370) 


ds_read_u8 v12, v8 offset:16768 // (op 293) Load local data 


ds_read_u8 v13, v8 offset:16832 // (op 302) Load local data 
s_barrier  // (op 1432) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 296) 
s_barrier  // (op 1367) 

s_lshl_b64 s[38:39], s[8:9], 2 // (op 572) 
v_mov_b64 v[82:83], s[38:39] // (op 572) convert


v_mov_b64 v[84:85], v[68:69] // (op 572) 
s_add_u32 s19, 8192, s15 // (op 572) 
s_mov_b32 m0, s19 // (op 572) reg expression
s_mov_b32 s38, 768 // (op 572) call()
buffer_load_dwordx4 v84, s[44:47], s38 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v84, s[56:57], v84, v82 // (op 572) least significant half
v_addc_co_u32 v85, s[56:57], 0, v85, s[56:57] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
buffer_load_dwordx4 v84, s[44:47], s38 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v84, s[56:57], v84, v82 // (op 572) least significant half
v_addc_co_u32 v85, s[56:57], 0, v85, s[56:57] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
buffer_load_dwordx4 v84, s[44:47], s38 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v84, s[56:57], v84, v82 // (op 572) least significant half
v_addc_co_u32 v85, s[56:57], 0, v85, s[56:57] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[74:81], a[0:3], v13, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 305) 
buffer_load_dwordx4 v84, s[44:47], s38 offen offset: 0 lds // (op 572) Load value direct to lds








v_add_co_u32 v14, vcc, 512, v14 // (op 703) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 703) most significant half
v_mov_b64 v[24:25], s[26:27] // (op 311) convert

v_mov_b32 v68, v70 // (op 311) 
buffer_load_ubyte v26, v68, s[52:55], 0 offen offset:24 // (op 311) Load value
v_add_u32 v68, v68, v24 // (op 311) 
buffer_load_ubyte v27, v68, s[52:55], 0 offen offset:24 // (op 311) Load value



s_barrier  // (op 1360) 
v_add_u32 v21, 16, v21 // (op 831) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 567) 
v_mov_b64 v[24:25], s[38:39] // (op 567) convert


s_lshl_b64 s[38:39], s[2:3], 8 // (op 567) 
s_mul_i32 s57, 3, s39 // (op 567) most significant: low of low * high
s_mul_i32 s56, 3, s38 // (op 567) least significant
s_mul_hi_u32 s19, 3, s38 // (op 567) most significant: high of low * low
s_add_u32 s57, s57, s19 // (op 567) most significant: sum


v_mov_b32 v68, s56 // (op 567) 
v_mov_b32 v69, s57 // (op 567) 
v_add_co_u32 v74, s[38:39], v72, v68 // (op 567) least significant half
v_addc_co_u32 v75, s[38:39], v73, v69, s[38:39] // (op 567) most significant half




s_add_u32 s19, 12288, s14 // (op 567) 
s_mov_b32 m0, s19 // (op 567) reg expression
buffer_load_dwordx4 v74, s[40:43], 0 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v74, s[38:39], v74, v24 // (op 567) least significant half
v_addc_co_u32 v75, s[38:39], 0, v75, s[38:39] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v74, s[40:43], 0 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v74, s[38:39], v74, v24 // (op 567) least significant half
v_addc_co_u32 v75, s[38:39], 0, v75, s[38:39] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v74, s[40:43], 0 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v74, s[38:39], v74, v24 // (op 567) least significant half
v_addc_co_u32 v75, s[38:39], 0, v75, s[38:39] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v74, s[40:43], 0 offen offset: 0 lds // (op 567) Load value direct to lds





v_mov_b64 v[24:25], s[20:21] // (op 310) convert

s_lshl_b32 s19, s20, 3 // (op 310) 
s_mul_i32 s38, 3, s19 // (op 310) 

v_add_u32 v68, v71, s38 // (op 310) 

buffer_load_ubyte v16, v68, s[48:51], 0 offen offset: 0 // (op 310) Load value
v_add_u32 v68, v68, v24 // (op 310) 
s_lshl_b64 s[38:39], s[2:3], 8 // (op 639) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 639) 

buffer_load_ubyte v17, v68, s[48:51], 0 offen offset: 0 // (op 310) Load value


v_mov_b32 v24, s56 // (op 639) 

v_mov_b32 v25, s57 // (op 639) 
s_barrier  // (op 1343) 
s_lshl_b32 s19, s20, 4 // (op 767) 



v_add_u32 v23, v23, s19 // (op 767) 

v_add_co_u32 v18, s[38:39], v18, v24 // (op 639) least significant half
v_addc_co_u32 v19, s[38:39], v19, v25, s[38:39] // (op 639) most significant half




s_waitcnt vmcnt(19)
ds_write_b8 v65, v28 offset:16384 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v65, v29 offset:16392 // (op 30) Store local data 
s_barrier  // (op 1326) 



s_waitcnt vmcnt(13)
ds_write_b8 v64, v30 offset:16512 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v64, v31 offset:16528 // (op 21) Store local data 
s_barrier  // (op 1315) 



ds_read_b128 v[32:35], v5 // (op 77) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 77) Load local data 



ds_read_b128 v[40:43], v5 offset:128 // (op 97) Load local data 
ds_read_b128 v[44:47], v5 offset:192 // (op 97) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1350) 



ds_read_u8 v20, v6 offset:16384 // (op 87) Load local data 



ds_read_u8 v22, v6 offset:16388 // (op 105) Load local data 
s_barrier  // (op 1299) 



ds_read_b64_tr_b8 v[48:49], v0 offset:4096 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:4224 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5120 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:5248 // (op 72) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:6144 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:6272 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:7168 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:7296 // (op 93) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1333) 


ds_read_u8 v12, v4 offset:16512 // (op 82) Load local data 


ds_read_u8 v13, v4 offset:16576 // (op 101) Load local data 
s_ashr_i64 s[38:39], s[6:7], 8 // (op 60) 
s_mov_b32 s19, s38 // (op 60) convert

s_and_b32 s38, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s38 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60:















s_barrier  // (op 1392) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 602) 
s_lshr_b32 s14, s0, 1 // (op 602) 

v_lshl_add_u32 v21, s14, 9, v1 // (op 602) 

s_lshl_b32 s0, s16, 4 // (op 602) 
v_mul_lo_u32 v15, v21, s3 // (op 602) most significant: low of low * high
v_mul_hi_u32 v18, v21, s2 // (op 602) most significant: high of low * low
v_mul_lo_u32 v14, v21, s2 // (op 602) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 602) most significant: sum


s_mov_b32 s15, 0 // (op 602) convert
s_mov_b32 s14, s0 // (op 602) convert

v_mov_b32 v21, s14 // (op 602) 
v_mov_b32 v18, s15 // (op 602) 
v_add_co_u32 v24, s[0:1], v21, v14 // (op 602) least significant half
v_addc_co_u32 v25, s[0:1], v18, v15, s[0:1] // (op 602) most significant half





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

v_and_b32 v21, -16, v1 // (op 669) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 669) 
v_sub_u32 v23, v1, v21 // (op 669) 

s_lshr_b32 s14, s0, 1 // (op 669) 

v_lshlrev_b32 v21, 4, v23 // (op 669) 

v_lshl_add_u32 v23, s14, 9, v21 // (op 669) 


v_mov_b32 v15, 0 // (op 669) convert
v_mov_b32 v14, v23 // (op 669) convert

v_lshrrev_b32 v21, 4, v1 // (op 669) 
v_lshl_add_u32 v23, s17, 4, v21 // (op 669) 

v_mul_lo_u32 v19, v23, s9 // (op 669) most significant: low of low * high
v_mul_hi_u32 v21, v23, s8 // (op 669) most significant: high of low * low
v_mul_lo_u32 v18, v23, s8 // (op 669) least significant: low of low * low
v_add_u32 v19, v19, v21 // (op 669) most significant: sum


v_add_co_u32 v28, s[0:1], v14, v18 // (op 669) least significant half
v_addc_co_u32 v29, s[0:1], v15, v19, s[0:1] // (op 669) most significant half



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

v_and_b32 v21, -16, v1 // (op 733) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 733) 
v_sub_u32 v23, v1, v21 // (op 733) 

s_lshr_b32 s14, s0, 1 // (op 733) 

v_lshl_add_u32 v21, s16, 4, v23 // (op 733) 

v_lshrrev_b32 v23, 4, v1 // (op 733) 
v_lshlrev_b32 v14, 1, v23 // (op 733) 

v_lshl_add_u32 v15, s14, 4, v14 // (op 733) 


v_mul_lo_u32 v14, v15, s20 // (op 733) 

v_add_u32 v15, v21, v14 // (op 733) 


s_mul_i32 s1, s20, s25 // (op 735) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 735) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 735) least significant
s_mul_hi_u32 s15, s20, s24 // (op 735) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 735) most significant: sum
s_add_u32 s1, s1, s15 // (op 735) most significant: sum


s_mov_b32 s48, s22 // (op 735) reg expression
s_mov_b32 s49, s23 // (op 735) reg expression
s_mov_b32 s50, s0 // (op 735) reg expression
s_mov_b32 s51, 131072 // (op 735) call()

v_and_b32 v14, -8, v1 // (op 797) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 797) 
v_sub_u32 v21, v1, v14 // (op 797) 

s_lshr_b32 s14, s0, 1 // (op 797) 

v_lshrrev_b32 v14, 3, v1 // (op 797) 
v_lshl_add_u32 v23, s14, 4, v21 // (op 797) 


v_lshlrev_b32 v21, 1, v14 // (op 797) 

v_lshl_add_u32 v14, s17, 4, v21 // (op 797) 

v_mul_lo_u32 v21, v14, s26 // (op 797) 

v_add_u32 v14, v23, v21 // (op 797) 


s_mul_i32 s1, s26, s13 // (op 799) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 799) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 799) least significant
s_mul_hi_u32 s15, s26, s12 // (op 799) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 799) most significant: sum
s_add_u32 s1, s1, s15 // (op 799) most significant: sum


s_mov_b32 s52, s28 // (op 799) reg expression
s_mov_b32 s53, s29 // (op 799) reg expression
s_mov_b32 s54, s0 // (op 799) reg expression
s_mov_b32 s55, 131072 // (op 799) call()

v_lshlrev_b32 v21, 4, v1 // (op 890) 
s_nop 0
v_readfirstlane_b32 s0, v21 // (op 890) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 942) 
v_sub_u32 v23, v1, v21 // (op 942) 

v_lshlrev_b32 v21, 4, v23 // (op 942) 

v_lshrrev_b32 v23, 4, v1 // (op 942) 
v_lshl_add_u32 v18, v23, 8, v21 // (op 942) 


s_nop 0
v_readfirstlane_b32 s1, v18 // (op 942) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 994) 
v_sub_u32 v23, v1, v21 // (op 994) 

v_lshrrev_b32 v21, 4, v1 // (op 994) 
v_lshl_add_u32 v18, v21, 5, v23 // (op 994) 


v_and_b32 v19, -8, v1 // (op 1047) 
v_sub_u32 v21, v1, v19 // (op 1047) 

v_lshrrev_b32 v19, 3, v1 // (op 1047) 
v_lshl_add_u32 v23, v19, 4, v21 // (op 1047) 


s_ashr_i64 s[14:15], s[6:7], 8 // (op 198) 
s_and_b32 s18, s14, 4294967294 // (op 198) 

s_ashr_i64 s[14:15], s[6:7], 8 // (op 197) 
s_mov_b32 s19, s14 // (op 197) convert

s_cmp_lt_u32 s18, s19 // (op 197) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197:
s_barrier  // (op 1296) 
s_add_u32 s18, s18, 1 // (op 199) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 554) 
v_mov_b64 v[12:13], s[14:15] // (op 554) convert


v_mov_b64 v[16:17], v[24:25] // (op 554) 
v_mov_b64 v[26:27], v[16:17] // (op 554) reg expression
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




s_lshl_b64 s[14:15], s[2:3], 8 // (op 592) 
v_mov_b32 v12, s14 // (op 592) 
v_mov_b32 v13, s15 // (op 592) 
v_add_co_u32 v24, s[38:39], v24, v12 // (op 592) least significant half
v_addc_co_u32 v25, s[38:39], v25, v13, s[38:39] // (op 592) most significant half




s_barrier  // (op 1312) 

s_lshl_b64 s[14:15], s[8:9], 2 // (op 560) 
v_mov_b64 v[12:13], s[14:15] // (op 560) convert


v_mov_b64 v[16:17], v[28:29] // (op 560) 
v_mov_b64 v[26:27], v[16:17] // (op 560) 
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



v_add_co_u32 v28, vcc, 256, v28 // (op 661) least significant half
v_addc_co_u32 v29, vcc, 0, v29, vcc // (op 661) most significant half

v_mov_b64 v[12:13], s[20:21] // (op 226) convert

v_mov_b32 v16, v15 // (op 226) 
v_mov_b32 v17, v16 // (op 226) reg expression
buffer_load_ubyte v26, v17, s[48:51], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v17, v17, v12 // (op 226) 
buffer_load_ubyte v27, v17, s[48:51], 0 offen offset: 0 // (op 226) Load value



s_lshl_b32 s19, s20, 3 // (op 725) 
v_add_u32 v15, v15, s19 // (op 725) 


s_barrier  // (op 1329) 
v_mov_b64 v[12:13], s[26:27] // (op 227) convert




v_mov_b32 v16, v14 // (op 227) 
v_mov_b32 v17, v16 // (op 227) 
buffer_load_ubyte v20, v17, s[52:55], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v17, v17, v12 // (op 227) 
buffer_load_ubyte v21, v17, s[52:55], 0 offen offset: 0 // (op 227) Load value



v_add_u32 v14, 8, v14 // (op 789) 
s_waitcnt vmcnt(3)
ds_write_b8 v18, v26 offset:16512 // (op 223) Store local data 
s_barrier  // (op 1302) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v27 offset:16528 // (op 223) Store local data 




ds_read_b64_tr_b8 v[30:31], v0 offset:4096 // (op 205) Transpose load local data 
s_barrier  // (op 1346) 
ds_read_b64_tr_b8 v[32:33], v0 offset:4224 // (op 205) Transpose load local data 



ds_read_b64_tr_b8 v[34:35], v0 offset:5120 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:5248 // (op 205) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v20 offset:16384 // (op 224) Store local data 
s_barrier  // (op 1319) 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v21 offset:16392 // (op 224) Store local data 




ds_read_b128 v[38:41], v5 // (op 207) Load local data 
ds_read_b128 v[42:45], v5 offset:64 // (op 207) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1336) 


ds_read_u8 v12, v4 offset:16512 // (op 209) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1353) 



ds_read_u8 v13, v6 offset:16384 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 212) 







ds_read_b64_tr_b8 v[30:31], v0 offset:6144 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:6272 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:7168 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:7296 // (op 214) Transpose load local data 



ds_read_b128 v[38:41], v5 offset:128 // (op 216) Load local data 
ds_read_b128 v[42:45], v5 offset:192 // (op 216) Load local data 


ds_read_u8 v12, v4 offset:16576 // (op 218) Load local data 



ds_read_u8 v13, v6 offset:16388 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 221) 




s_ashr_i64 s[14:15], s[6:7], 8 // (op 197) 
s_mov_b32 s19, s14 // (op 197) convert

s_cmp_lt_u32 s18, s19 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 if true)
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197:























s_barrier  // (op 1422) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

v_cvt_f16_f32 v12, v16 // (op 56) 
v_cvt_f16_f32 v13, v17 // (op 56) 
v_cvt_f16_f32 v14, v18 // (op 56) 
v_cvt_f16_f32 v15, v19 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 1089) 
v_and_b32 v3, 63, v1 // (op 1089) 
v_lshrrev_b32 v4, 4, v3 // (op 1089) 
v_lshlrev_b32 v5, 2, v4 // (op 1089) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1089) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1089) 

v_lshrrev_b32 v4, 2, v3 // (op 1089) 

v_and_b32 v3, 3, v4 // (op 1089) 

v_and_b32 v4, 3, v1 // (op 1089) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1089) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1089) 

v_mul_lo_u32 v3, v6, s30 // (op 1089) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 1089) 


s_mul_i32 s15, s30, s13 // (op 1091) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1091) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1091) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1091) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1091) most significant: sum
s_add_u32 s15, s15, s1 // (op 1091) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 1091) 

s_mov_b32 s8, s32 // (op 1091) reg expression
s_mov_b32 s9, s33 // (op 1091) reg expression
s_mov_b32 s10, s0 // (op 1091) reg expression
s_mov_b32 s11, 131072 // (op 1091) call()


v_pack_b32_f16 v8, v12, v13 // (op 58) 
v_pack_b32_f16 v9, v14, v15 // (op 58) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157:
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


v_lshl_add_u32 v6, s17, 4, v5 // (op 860) 

v_mul_lo_u32 v3, v6, s34 // (op 860) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 860) 


s_mul_i32 s15, s34, s13 // (op 862) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 862) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 862) least significant
s_mul_hi_u32 s1, s34, s12 // (op 862) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 862) most significant: sum
s_add_u32 s15, s15, s1 // (op 862) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 862) 

s_mov_b32 s4, s36 // (op 862) reg expression
s_mov_b32 s5, s37 // (op 862) reg expression
s_mov_b32 s6, s0 // (op 862) reg expression
s_mov_b32 s7, 131072 // (op 862) call()





buffer_load_dwordx2 v[8:9], v4, s[4:7], 0 offen // (op 170) Load value



v_accvgpr_read v12, a0 // (op 166) 
v_accvgpr_read v13, a1 // (op 166) 
v_accvgpr_read v14, a2 // (op 166) 
v_accvgpr_read v15, a3 // (op 166) 
v_mul_f32 v16, v11, v12 // (op 166) 
v_mul_f32 v17, v11, v13 // (op 166) 
v_mul_f32 v18, v11, v14 // (op 166) 
v_mul_f32 v19, v11, v15 // (op 166) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 171) 
v_lshrrev_b32 v13, 16, v8 // (op 171) 
v_cvt_f32_f16 v13, v13 // (op 171) 
v_mul_f32 v20, v7, v12 // (op 171) 
v_mul_f32 v21, v7, v13 // (op 171) 
v_cvt_f32_f16 v12, v9 // (op 171) 
v_lshrrev_b32 v13, 16, v9 // (op 171) 
v_cvt_f32_f16 v13, v13 // (op 171) 
v_mul_f32 v22, v7, v12 // (op 171) 
v_mul_f32 v23, v7, v13 // (op 171) 


v_add_f32 v12, v20, v16 // (op 167) 
v_add_f32 v13, v21, v17 // (op 167) 
v_add_f32 v14, v22, v18 // (op 167) 
v_add_f32 v15, v23, v19 // (op 167) 


v_cvt_f16_f32 v16, v12 // (op 168) 
v_cvt_f16_f32 v17, v13 // (op 168) 
v_cvt_f16_f32 v18, v14 // (op 168) 
v_cvt_f16_f32 v19, v15 // (op 168) 

v_lshrrev_b32 v0, 6, v1 // (op 1106) 
v_and_b32 v3, 63, v1 // (op 1106) 
v_lshrrev_b32 v4, 4, v3 // (op 1106) 
v_lshlrev_b32 v5, 2, v4 // (op 1106) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1106) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1106) 

v_lshrrev_b32 v4, 2, v3 // (op 1106) 

v_and_b32 v3, 3, v4 // (op 1106) 

v_and_b32 v4, 3, v1 // (op 1106) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1106) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1106) 

v_mul_lo_u32 v3, v6, s30 // (op 1106) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 1106) 


s_mul_i32 s15, s30, s13 // (op 1108) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1108) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1108) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1108) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1108) most significant: sum
s_add_u32 s15, s15, s1 // (op 1108) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 1108) 

s_mov_b32 s8, s32 // (op 1108) reg expression
s_mov_b32 s9, s33 // (op 1108) reg expression
s_mov_b32 s10, s0 // (op 1108) reg expression
s_mov_b32 s11, 131072 // (op 1108) call()


v_pack_b32_f16 v8, v16, v17 // (op 169) 
v_pack_b32_f16 v9, v18, v19 // (op 169) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 169) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157:




s_endpgm  // End of RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
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
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
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
  - .name: RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NN_BF8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 58
    .vgpr_count: 86
    .agpr_count: 4
    .max_flat_workgroup_size: 64
    .wavefront_size: 64
    .args:
      - .name: Tensor_0_stride_1_0
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
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
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

