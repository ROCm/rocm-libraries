// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_and_b32 v3, 48, v1 // (op 969) 
v_and_b32 v4, 63, v1 // (op 969) 
v_sub_u32 v5, v4, v3 // (op 969) 

v_lshrrev_b32 v4, 1, v5 // (op 969) 
v_add_lshl_u32 v6, v3, v4, 4 // (op 969) 


v_and_b32 v3, 1, v5 // (op 969) 

v_lshlrev_b32 v4, 3, v3 // (op 969) 

v_add_u32 v3, v6, v4 // (op 969) 


v_lshrrev_b32 v4, 6, v1 // (op 1022) 
v_and_b32 v5, 15, v1 // (op 1022) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 1022) 


v_and_b32 v4, 48, v1 // (op 1022) 
v_add_u32 v5, v4, v6 // (op 1022) 


v_and_b32 v4, 48, v1 // (op 1075) 
v_and_b32 v6, 15, v1 // (op 1075) 
v_add_u32 v7, v4, v6 // (op 1075) 


v_and_b32 v4, 48, v1 // (op 1148) 
v_and_b32 v6, 63, v1 // (op 1148) 
v_sub_u32 v8, v6, v4 // (op 1148) 

v_lshrrev_b32 v6, 1, v8 // (op 1148) 
v_add_lshl_u32 v9, v4, v6, 4 // (op 1148) 


v_and_b32 v4, 1, v8 // (op 1148) 

v_lshlrev_b32 v6, 3, v4 // (op 1148) 

v_lshrrev_b32 v4, 6, v1 // (op 1148) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 1148) 


v_add_u32 v4, v9, v8 // (op 1148) 


v_and_b32 v6, 48, v1 // (op 1187) 
v_and_b32 v8, 63, v1 // (op 1187) 
v_sub_u32 v9, v8, v6 // (op 1187) 

v_lshrrev_b32 v8, 1, v9 // (op 1187) 
v_add_lshl_u32 v10, v6, v8, 4 // (op 1187) 


v_and_b32 v6, 1, v9 // (op 1187) 

v_lshlrev_b32 v8, 3, v6 // (op 1187) 

v_add_u32 v6, v10, v8 // (op 1187) 


v_lshrrev_b32 v8, 6, v1 // (op 1211) 
v_and_b32 v9, 15, v1 // (op 1211) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 1211) 


v_and_b32 v8, 48, v1 // (op 1211) 
v_add_u32 v9, v8, v10 // (op 1211) 


v_and_b32 v8, 48, v1 // (op 1235) 
v_and_b32 v10, 15, v1 // (op 1235) 
v_add_u32 v11, v8, v10 // (op 1235) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


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




s_mul_i32 s1, s8, s7 // (op 694) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 694) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 694) least significant
s_mul_hi_u32 s15, s8, s6 // (op 694) most significant: high of low * low
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




s_mul_i32 s1, s8, s7 // (op 713) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 713) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 713) least significant
s_mul_hi_u32 s15, s8, s6 // (op 713) most significant: high of low * low
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

v_mul_lo_u32 v21, v22, s12 // (op 756) 

v_add_u32 v22, v20, v21 // (op 756) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 758) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 758) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 758) least significant
s_mul_hi_u32 s15, s12, s22 // (op 758) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 758) most significant: sum
s_add_u32 s1, s1, s15 // (op 758) most significant: sum


s_mov_b32 s48, s20 // (op 758) reg expression
s_mov_b32 s49, s21 // (op 758) reg expression
s_mov_b32 s50, s0 // (op 758) reg expression
s_mov_b32 s51, 131072 // (op 758) call()

v_and_b32 v20, -16, v1 // (op 775) 
v_sub_u32 v21, v1, v20 // (op 775) 

v_lshl_add_u32 v20, s16, 4, v21 // (op 775) 

v_lshrrev_b32 v21, 4, v1 // (op 775) 
v_lshlrev_b32 v23, 1, v21 // (op 775) 

v_mul_lo_u32 v21, v23, s12 // (op 775) 

v_add_u32 v23, v20, v21 // (op 775) 


s_mul_i32 s1, s12, s23 // (op 777) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 777) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 777) least significant
s_mul_hi_u32 s15, s12, s22 // (op 777) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 777) most significant: sum
s_add_u32 s1, s1, s15 // (op 777) most significant: sum


s_mov_b32 s48, s20 // (op 777) reg expression
s_mov_b32 s49, s21 // (op 777) reg expression
s_mov_b32 s50, s0 // (op 777) reg expression
s_mov_b32 s51, 131072 // (op 777) call()

v_and_b32 v20, -16, v1 // (op 820) 
v_sub_u32 v21, v1, v20 // (op 820) 

v_lshl_add_u32 v20, s17, 4, v21 // (op 820) 

v_lshrrev_b32 v21, 4, v1 // (op 820) 
v_lshlrev_b32 v24, 1, v21 // (op 820) 

v_mul_lo_u32 v21, v24, s24 // (op 820) 
v_add_u32 v21, v21, v20 // (op 820) 


s_mul_i32 s1, s24, s23 // (op 822) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 822) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 822) least significant
s_mul_hi_u32 s15, s24, s22 // (op 822) most significant: high of low * low
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


s_mul_i32 s1, s24, s23 // (op 841) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 841) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 841) least significant
s_mul_hi_u32 s15, s24, s22 // (op 841) most significant: high of low * low
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

v_mov_b32 v20, v21 // (op 431) reg expression
buffer_load_ubyte v28, v20, s[52:55], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v20, v20, v26 // (op 431) 
buffer_load_ubyte v29, v20, s[52:55], 0 offen offset: 0 // (op 431) Load value


v_lshlrev_b32 v20, 4, v1 // (op 905) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 905) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 582) 
v_mov_b64 v[26:27], s[14:15] // (op 582) convert


v_mov_b64 v[30:31], v[16:17] // (op 582) reg expression
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





v_mov_b64 v[26:27], s[12:13] // (op 437) convert

v_mov_b32 v20, v22 // (op 437) reg expression
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
v_add_u32 v20, v21, s0 // (op 443) 

buffer_load_ubyte v26, v20, s[52:55], 0 offen offset: 0 // (op 443) Load value
v_add_u32 v20, v20, v12 // (op 443) 
buffer_load_ubyte v27, v20, s[52:55], 0 offen offset: 0 // (op 443) Load value



v_lshlrev_b32 v25, 4, v1 // (op 1136) 
s_nop 0
v_readfirstlane_b32 s0, v25 // (op 1136) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 589) 
v_mov_b64 v[12:13], s[14:15] // (op 589) convert


s_lshl_b64 s[14:15], s[2:3], 8 // (op 589) 
v_mov_b32 v25, s14 // (op 589) 
v_mov_b32 v20, s15 // (op 589) 
v_add_co_u32 v32, s[18:19], v16, v25 // (op 589) least significant half
v_addc_co_u32 v33, s[18:19], v17, v20, s[18:19] // (op 589) most significant half




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






v_mov_b64 v[12:13], s[12:13] // (op 449) convert

s_lshl_b32 s0, s12, 3 // (op 449) 
v_add_u32 v25, v22, s0 // (op 449) 

buffer_load_ubyte v16, v25, s[48:51], 0 offen offset: 0 // (op 449) Load value
v_add_u32 v25, v25, v12 // (op 449) 
buffer_load_ubyte v17, v25, s[48:51], 0 offen offset: 0 // (op 449) Load value



v_and_b32 v25, -16, v1 // (op 1063) 
v_sub_u32 v12, v1, v25 // (op 1063) 

v_lshrrev_b32 v13, 4, v1 // (op 1063) 
v_lshl_add_u32 v25, v13, 5, v12 // (op 1063) 





s_waitcnt vmcnt(19)
ds_write_b8 v25, v28 offset:16384 // (op 452) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v25, v29 offset:16400 // (op 452) Store local data 

v_and_b32 v25, -16, v1 // (op 1010) 
v_sub_u32 v12, v1, v25 // (op 1010) 

v_lshrrev_b32 v13, 4, v1 // (op 1010) 
v_lshl_add_u32 v25, v13, 5, v12 // (op 1010) 





s_waitcnt vmcnt(13)
ds_write_b8 v25, v30 offset:16512 // (op 454) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v25, v31 offset:16528 // (op 454) Store local data 

s_barrier  // (op 1302) 



ds_read_b64_tr_b8 v[32:33], v3 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:128 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:1024 // (op 474) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:1152 // (op 474) Transpose load local data 



ds_read_b64_tr_b8 v[40:41], v3 offset:2048 // (op 477) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:2176 // (op 477) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:3072 // (op 477) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v3 offset:3200 // (op 477) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1336) 


ds_read_u8 v25, v7 offset:16384 // (op 486) Load local data 


ds_read_u8 v12, v7 offset:16448 // (op 489) Load local data 
s_barrier  // (op 1286) 



ds_read_b64_tr_b8 v[48:49], v0 offset:4096 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:4224 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5120 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:5248 // (op 468) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:6144 // (op 471) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:6272 // (op 471) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:7168 // (op 471) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:7296 // (op 471) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 1319) 


ds_read_u8 v13, v5 offset:16512 // (op 480) Load local data 


ds_read_u8 v20, v5 offset:16576 // (op 483) Load local data 
v_lshlrev_b32 v21, 4, v1 // (op 877) 
s_nop 0
v_readfirstlane_b32 s0, v21 // (op 877) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 4, v1 // (op 929) 
s_nop 0
v_readfirstlane_b32 s1, v21 // (op 929) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 981) 
v_sub_u32 v22, v1, v21 // (op 981) 

v_lshrrev_b32 v21, 4, v1 // (op 981) 
v_lshl_add_u32 v64, v21, 5, v22 // (op 981) 


v_and_b32 v21, -16, v1 // (op 1034) 
v_sub_u32 v22, v1, v21 // (op 1034) 

v_lshrrev_b32 v21, 4, v1 // (op 1034) 
v_lshl_add_u32 v65, v21, 5, v22 // (op 1034) 


v_lshlrev_b32 v21, 4, v1 // (op 1121) 
s_nop 0
v_readfirstlane_b32 s14, v21 // (op 1121) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v21, 4, v1 // (op 1160) 
s_nop 0
v_readfirstlane_b32 s15, v21 // (op 1160) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 1199) 
v_sub_u32 v22, v1, v21 // (op 1199) 

v_lshrrev_b32 v21, 4, v1 // (op 1199) 
v_lshl_add_u32 v66, v21, 5, v22 // (op 1199) 


v_and_b32 v21, -16, v1 // (op 1223) 
v_sub_u32 v22, v1, v21 // (op 1223) 

v_lshrrev_b32 v21, 4, v1 // (op 1223) 
v_lshl_add_u32 v67, v21, 5, v22 // (op 1223) 


s_mov_b32 s18, 0 // (op 61) call()
s_ashr_i64 s[38:39], s[6:7], 8 // (op 60) 
s_mov_b32 s19, s38 // (op 60) convert

s_and_b32 s38, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s38 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60:
s_add_u32 s18, s18, 2 // (op 62) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v13, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 90) 
s_barrier  // (op 1305) 

s_lshl_b64 s[38:39], s[8:9], 6 // (op 549) 
v_mov_b64 v[68:69], s[38:39] // (op 549) convert


v_mov_b64 v[70:71], v[14:15] // (op 549) 
s_lshl_b64 s[38:39], s[8:9], 8 // (op 549) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 549) 

v_mov_b32 v21, s56 // (op 549) 
v_mov_b32 v22, s57 // (op 549) 
v_add_co_u32 v72, s[38:39], v70, v21 // (op 549) least significant half
v_addc_co_u32 v73, s[38:39], v71, v22, s[38:39] // (op 549) most significant half




s_mov_b32 m0, s1 // (op 549) reg expression
buffer_load_dwordx4 v72, s[44:47], 0 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v72, s[38:39], v72, v68 // (op 549) least significant half
v_addc_co_u32 v73, s[38:39], 0, v73, s[38:39] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
buffer_load_dwordx4 v72, s[44:47], 0 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v72, s[38:39], v72, v68 // (op 549) least significant half
v_addc_co_u32 v73, s[38:39], 0, v73, s[38:39] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
buffer_load_dwordx4 v72, s[44:47], 0 offen offset: 0 lds // (op 549) Load value direct to lds
v_add_co_u32 v72, s[38:39], v72, v68 // (op 549) least significant half
v_addc_co_u32 v73, s[38:39], 0, v73, s[38:39] // (op 549) most significant half

s_add_u32 m0, m0, 1024 // (op 549) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 107) 
buffer_load_dwordx4 v72, s[44:47], 0 offen offset: 0 lds // (op 549) Load value direct to lds



v_mov_b64 v[68:69], s[24:25] // (op 25) convert

v_mov_b32 v21, v24 // (op 25) 
s_lshl_b32 s19, s24, 4 // (op 25) 
v_add_u32 v22, v21, s19 // (op 25) 

buffer_load_ubyte v28, v22, s[52:55], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v22, v22, v68 // (op 25) 
buffer_load_ubyte v29, v22, s[52:55], 0 offen offset: 0 // (op 25) Load value


s_barrier  // (op 1289) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 544) 
v_mov_b64 v[68:69], s[38:39] // (op 544) convert


v_mov_b64 v[72:73], v[18:19] // (op 544) 
s_lshl_b64 s[38:39], s[2:3], 8 // (op 544) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 544) 

v_mov_b32 v22, s56 // (op 544) 
v_mov_b32 v74, s57 // (op 544) 
v_add_co_u32 v76, s[38:39], v72, v22 // (op 544) least significant half
v_addc_co_u32 v77, s[38:39], v73, v74, s[38:39] // (op 544) most significant half




s_add_u32 s19, 4096, s0 // (op 544) 
s_mov_b32 m0, s19 // (op 544) reg expression
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v68 // (op 544) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v68 // (op 544) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 544) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v68 // (op 544) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 544) most significant half

s_add_u32 m0, m0, 1024 // (op 544) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 544) Load value direct to lds




v_mov_b64 v[68:69], s[12:13] // (op 17) convert

v_mov_b32 v22, v23 // (op 17) 
s_lshl_b32 s19, s12, 4 // (op 17) 
v_add_u32 v74, v22, s19 // (op 17) 

buffer_load_ubyte v30, v74, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v74, v74, v68 // (op 17) 
buffer_load_ubyte v31, v74, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1375) 



s_waitcnt vmcnt(19)
ds_write_b8 v67, v26 offset:16640 // (op 308) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v67, v27 offset:16656 // (op 308) Store local data 
s_barrier  // (op 1369) 



s_waitcnt vmcnt(13)
ds_write_b8 v66, v16 offset:16768 // (op 307) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v66, v17 offset:16784 // (op 307) Store local data 
s_barrier  // (op 1359) 



ds_read_b64_tr_b8 v[32:33], v6 offset:8192 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v6 offset:8320 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v6 offset:9216 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v6 offset:9344 // (op 291) Transpose load local data 



ds_read_b64_tr_b8 v[40:41], v6 offset:10240 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v6 offset:10368 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v6 offset:11264 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v6 offset:11392 // (op 300) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1372) 


ds_read_u8 v25, v11 offset:16640 // (op 295) Load local data 


ds_read_u8 v12, v11 offset:16704 // (op 304) Load local data 
s_barrier  // (op 1353) 
s_barrier  // (op 1432) 



ds_read_b64_tr_b8 v[48:49], v4 offset:12288 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v4 offset:12416 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v4 offset:13312 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v4 offset:13440 // (op 289) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v4 offset:14336 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v4 offset:14464 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v4 offset:15360 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v4 offset:15488 // (op 298) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 1366) 


ds_read_u8 v13, v9 offset:16768 // (op 293) Load local data 


ds_read_u8 v20, v9 offset:16832 // (op 302) Load local data 
s_barrier  // (op 1428) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v13, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 296) 
s_barrier  // (op 1363) 

s_lshl_b64 s[38:39], s[8:9], 6 // (op 572) 
v_mov_b64 v[68:69], s[38:39] // (op 572) convert


s_lshl_b64 s[38:39], s[8:9], 8 // (op 572) 
s_mul_i32 s57, 3, s39 // (op 572) most significant: low of low * high
s_mul_i32 s56, 3, s38 // (op 572) least significant
s_mul_hi_u32 s19, 3, s38 // (op 572) most significant: high of low * low
s_add_u32 s57, s57, s19 // (op 572) most significant: sum


v_mov_b32 v74, s56 // (op 572) 
v_mov_b32 v75, s57 // (op 572) 
v_add_co_u32 v76, s[38:39], v70, v74 // (op 572) least significant half
v_addc_co_u32 v77, s[38:39], v71, v75, s[38:39] // (op 572) most significant half




s_add_u32 s19, 8192, s15 // (op 572) 
s_mov_b32 m0, s19 // (op 572) reg expression
buffer_load_dwordx4 v76, s[44:47], 0 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v68 // (op 572) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
buffer_load_dwordx4 v76, s[44:47], 0 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v68 // (op 572) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
buffer_load_dwordx4 v76, s[44:47], 0 offen offset: 0 lds // (op 572) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v68 // (op 572) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 572) most significant half

s_add_u32 m0, m0, 1024 // (op 572) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 305) 
buffer_load_dwordx4 v76, s[44:47], 0 offen offset: 0 lds // (op 572) Load value direct to lds





v_mov_b64 v[68:69], s[24:25] // (op 311) convert

s_lshl_b32 s19, s24, 3 // (op 311) 
s_mul_i32 s38, 3, s19 // (op 311) 

v_add_u32 v70, v21, s38 // (op 311) 

buffer_load_ubyte v26, v70, s[52:55], 0 offen offset: 0 // (op 311) Load value
v_add_u32 v70, v70, v68 // (op 311) 
s_lshl_b64 s[38:39], s[8:9], 8 // (op 703) 
buffer_load_ubyte v27, v70, s[52:55], 0 offen offset: 0 // (op 311) Load value


s_lshl_b64 s[56:57], s[38:39], 1 // (op 703) 


v_mov_b32 v21, s56 // (op 703) 
s_barrier  // (op 1356) 
s_lshl_b32 s19, s24, 4 // (op 831) 

v_add_u32 v24, v24, s19 // (op 831) 

v_mov_b32 v68, s57 // (op 703) 
s_lshl_b64 s[38:39], s[2:3], 6 // (op 567) 
v_mov_b64 v[70:71], s[38:39] // (op 567) convert


s_lshl_b64 s[38:39], s[2:3], 8 // (op 567) 
s_mul_i32 s59, 3, s39 // (op 567) most significant: low of low * high
s_mul_i32 s58, 3, s38 // (op 567) least significant
s_mul_hi_u32 s19, 3, s38 // (op 567) most significant: high of low * low
s_add_u32 s59, s59, s19 // (op 567) most significant: sum


v_mov_b32 v69, s58 // (op 567) 
v_mov_b32 v74, s59 // (op 567) 
v_add_co_u32 v14, s[38:39], v14, v21 // (op 703) least significant half
v_addc_co_u32 v15, s[38:39], v15, v68, s[38:39] // (op 703) most significant half




v_add_co_u32 v76, s[38:39], v72, v69 // (op 567) least significant half
v_addc_co_u32 v77, s[38:39], v73, v74, s[38:39] // (op 567) most significant half




s_add_u32 s19, 12288, s14 // (op 567) 
s_mov_b32 m0, s19 // (op 567) reg expression
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v70 // (op 567) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v70 // (op 567) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 567) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v70 // (op 567) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 567) most significant half

s_add_u32 m0, m0, 1024 // (op 567) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 567) Load value direct to lds





v_mov_b64 v[68:69], s[12:13] // (op 310) convert

s_lshl_b32 s19, s12, 3 // (op 310) 
s_mul_i32 s38, 3, s19 // (op 310) 

v_add_u32 v21, v22, s38 // (op 310) 

buffer_load_ubyte v16, v21, s[48:51], 0 offen offset: 0 // (op 310) Load value
v_add_u32 v21, v21, v68 // (op 310) 
s_lshl_b64 s[38:39], s[2:3], 8 // (op 639) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 639) 

buffer_load_ubyte v17, v21, s[48:51], 0 offen offset: 0 // (op 310) Load value


v_mov_b32 v21, s56 // (op 639) 

v_mov_b32 v22, s57 // (op 639) 
s_barrier  // (op 1339) 
s_lshl_b32 s19, s12, 4 // (op 767) 



v_add_u32 v23, v23, s19 // (op 767) 

v_add_co_u32 v18, s[38:39], v18, v21 // (op 639) least significant half
v_addc_co_u32 v19, s[38:39], v19, v22, s[38:39] // (op 639) most significant half




s_waitcnt vmcnt(19)
ds_write_b8 v65, v28 offset:16384 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v65, v29 offset:16400 // (op 30) Store local data 
s_barrier  // (op 1322) 



s_waitcnt vmcnt(13)
ds_write_b8 v64, v30 offset:16512 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v64, v31 offset:16528 // (op 21) Store local data 
s_barrier  // (op 1311) 



ds_read_b64_tr_b8 v[32:33], v3 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v3 offset:128 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v3 offset:1024 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:1152 // (op 77) Transpose load local data 



ds_read_b64_tr_b8 v[40:41], v3 offset:2048 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:2176 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:3072 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v3 offset:3200 // (op 97) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1346) 


ds_read_u8 v25, v7 offset:16384 // (op 87) Load local data 


ds_read_u8 v12, v7 offset:16448 // (op 105) Load local data 
s_barrier  // (op 1295) 



ds_read_b64_tr_b8 v[48:49], v0 offset:4096 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:4224 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5120 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:5248 // (op 72) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:6144 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:6272 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:7168 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:7296 // (op 93) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 1329) 


ds_read_u8 v13, v5 offset:16512 // (op 82) Load local data 


ds_read_u8 v20, v5 offset:16576 // (op 101) Load local data 
s_ashr_i64 s[38:39], s[6:7], 8 // (op 60) 
s_mov_b32 s19, s38 // (op 60) convert

s_and_b32 s38, s19, 4294967294 // (op 60) 

s_cmp_lt_u32 s18, s38 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60:
















s_barrier  // (op 1388) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 602) 
s_lshr_b32 s14, s0, 1 // (op 602) 

v_lshl_add_u32 v14, s14, 9, v1 // (op 602) 

s_lshl_b32 s0, s16, 4 // (op 602) 
v_mul_lo_u32 v19, v14, s3 // (op 602) most significant: low of low * high
v_mul_hi_u32 v15, v14, s2 // (op 602) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 602) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 602) most significant: sum


s_mov_b32 s15, 0 // (op 602) convert
s_mov_b32 s14, s0 // (op 602) convert

v_mov_b32 v14, s14 // (op 602) 
v_mov_b32 v15, s15 // (op 602) 
v_add_co_u32 v28, s[0:1], v14, v18 // (op 602) least significant half
v_addc_co_u32 v29, s[0:1], v15, v19, s[0:1] // (op 602) most significant half





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

s_ashr_i64 s[0:1], s[6:7], 8 // (op 669) 
s_lshr_b32 s14, s0, 1 // (op 669) 

v_lshl_add_u32 v14, s14, 9, v1 // (op 669) 

s_lshl_b32 s0, s17, 4 // (op 669) 
s_mov_b32 s15, 0 // (op 669) convert
s_mov_b32 s14, s0 // (op 669) convert

v_mul_lo_u32 v19, v14, s9 // (op 669) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 669) most significant: high of low * low
v_mul_lo_u32 v18, v14, s8 // (op 669) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 669) most significant: sum

v_mov_b32 v15, s14 // (op 669) 
v_mov_b32 v21, s15 // (op 669) 
v_add_co_u32 v18, s[0:1], v18, v15 // (op 669) least significant half
v_addc_co_u32 v19, s[0:1], v19, v21, s[0:1] // (op 669) most significant half





s_mul_i32 s1, s8, s7 // (op 671) most significant: low of low * high
s_mul_i32 s14, s9, s6 // (op 671) most significant: low of high * low
s_mul_i32 s0, s8, s6 // (op 671) least significant
s_mul_hi_u32 s15, s8, s6 // (op 671) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 671) most significant: sum
s_add_u32 s1, s1, s15 // (op 671) most significant: sum


s_mov_b32 s44, s10 // (op 671) reg expression
s_mov_b32 s45, s11 // (op 671) reg expression
s_mov_b32 s46, s0 // (op 671) reg expression
s_mov_b32 s47, 131072 // (op 671) call()

v_and_b32 v14, -16, v1 // (op 733) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 733) 
v_sub_u32 v15, v1, v14 // (op 733) 

s_lshr_b32 s14, s0, 1 // (op 733) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 733) 

v_lshrrev_b32 v15, 4, v1 // (op 733) 
v_lshlrev_b32 v21, 1, v15 // (op 733) 

v_lshl_add_u32 v15, s14, 4, v21 // (op 733) 


v_mul_lo_u32 v21, v15, s12 // (op 733) 

v_add_u32 v15, v14, v21 // (op 733) 


s_mul_i32 s1, s12, s23 // (op 735) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 735) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 735) least significant
s_mul_hi_u32 s15, s12, s22 // (op 735) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 735) most significant: sum
s_add_u32 s1, s1, s15 // (op 735) most significant: sum


s_mov_b32 s48, s20 // (op 735) reg expression
s_mov_b32 s49, s21 // (op 735) reg expression
s_mov_b32 s50, s0 // (op 735) reg expression
s_mov_b32 s51, 131072 // (op 735) call()

v_lshrrev_b32 v14, 4, v1 // (op 797) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 797) 
v_lshlrev_b32 v21, 1, v14 // (op 797) 

s_lshr_b32 s14, s0, 1 // (op 797) 

v_and_b32 v14, -16, v1 // (op 797) 
v_lshl_add_u32 v22, s14, 4, v21 // (op 797) 


v_sub_u32 v21, v1, v14 // (op 797) 

v_lshl_add_u32 v14, s17, 4, v21 // (op 797) 

v_mul_lo_u32 v21, v22, s24 // (op 797) 
v_add_u32 v21, v21, v14 // (op 797) 


s_mul_i32 s1, s24, s23 // (op 799) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 799) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 799) least significant
s_mul_hi_u32 s15, s24, s22 // (op 799) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 799) most significant: sum
s_add_u32 s1, s1, s15 // (op 799) most significant: sum


s_mov_b32 s52, s26 // (op 799) reg expression
s_mov_b32 s53, s27 // (op 799) reg expression
s_mov_b32 s54, s0 // (op 799) reg expression
s_mov_b32 s55, 131072 // (op 799) call()

v_lshlrev_b32 v14, 4, v1 // (op 890) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 890) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v14, 4, v1 // (op 942) 
s_nop 0
v_readfirstlane_b32 s1, v14 // (op 942) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -16, v1 // (op 994) 
v_sub_u32 v22, v1, v14 // (op 994) 

v_lshrrev_b32 v14, 4, v1 // (op 994) 
v_lshl_add_u32 v23, v14, 5, v22 // (op 994) 


v_and_b32 v14, -16, v1 // (op 1047) 
v_sub_u32 v22, v1, v14 // (op 1047) 

v_lshrrev_b32 v14, 4, v1 // (op 1047) 
v_lshl_add_u32 v24, v14, 5, v22 // (op 1047) 


s_ashr_i64 s[14:15], s[6:7], 8 // (op 198) 
s_and_b32 s18, s14, 4294967294 // (op 198) 

s_ashr_i64 s[14:15], s[6:7], 8 // (op 197) 
s_mov_b32 s19, s14 // (op 197) convert

s_cmp_lt_u32 s18, s19 // (op 197) 










s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197:
s_barrier  // (op 1292) 
s_add_u32 s18, s18, 1 // (op 199) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 554) 
v_mov_b64 v[16:17], s[14:15] // (op 554) convert


v_mov_b64 v[26:27], v[28:29] // (op 554) 
v_mov_b64 v[12:13], v[26:27] // (op 554) reg expression
s_add_u32 s19, 4096, s0 // (op 554) 
s_mov_b32 m0, s19 // (op 554) reg expression
buffer_load_dwordx4 v12, s[40:43], 0 offen offset: 0 lds // (op 554) Load value direct to lds
v_add_co_u32 v12, s[14:15], v12, v16 // (op 554) least significant half
v_addc_co_u32 v13, s[14:15], 0, v13, s[14:15] // (op 554) most significant half

s_add_u32 m0, m0, 1024 // (op 554) 
buffer_load_dwordx4 v12, s[40:43], 0 offen offset: 0 lds // (op 554) Load value direct to lds
v_add_co_u32 v12, s[14:15], v12, v16 // (op 554) least significant half
v_addc_co_u32 v13, s[14:15], 0, v13, s[14:15] // (op 554) most significant half

s_add_u32 m0, m0, 1024 // (op 554) 
buffer_load_dwordx4 v12, s[40:43], 0 offen offset: 0 lds // (op 554) Load value direct to lds
v_add_co_u32 v12, s[14:15], v12, v16 // (op 554) least significant half
v_addc_co_u32 v13, s[14:15], 0, v13, s[14:15] // (op 554) most significant half

s_add_u32 m0, m0, 1024 // (op 554) 
buffer_load_dwordx4 v12, s[40:43], 0 offen offset: 0 lds // (op 554) Load value direct to lds




s_lshl_b64 s[14:15], s[2:3], 8 // (op 592) 
v_mov_b32 v20, s14 // (op 592) 
v_mov_b32 v22, s15 // (op 592) 
v_add_co_u32 v28, s[38:39], v28, v20 // (op 592) least significant half
v_addc_co_u32 v29, s[38:39], v29, v22, s[38:39] // (op 592) most significant half




s_barrier  // (op 1308) 

s_lshl_b64 s[14:15], s[8:9], 6 // (op 560) 
v_mov_b64 v[16:17], s[14:15] // (op 560) convert


v_mov_b64 v[26:27], v[18:19] // (op 560) 
v_mov_b64 v[12:13], v[26:27] // (op 560) reg expression
s_mov_b32 m0, s1 // (op 560) reg expression
buffer_load_dwordx4 v12, s[44:47], 0 offen offset: 0 lds // (op 560) Load value direct to lds
v_add_co_u32 v12, s[14:15], v12, v16 // (op 560) least significant half
v_addc_co_u32 v13, s[14:15], 0, v13, s[14:15] // (op 560) most significant half

s_add_u32 m0, m0, 1024 // (op 560) 
buffer_load_dwordx4 v12, s[44:47], 0 offen offset: 0 lds // (op 560) Load value direct to lds
v_add_co_u32 v12, s[14:15], v12, v16 // (op 560) least significant half
v_addc_co_u32 v13, s[14:15], 0, v13, s[14:15] // (op 560) most significant half

s_add_u32 m0, m0, 1024 // (op 560) 
buffer_load_dwordx4 v12, s[44:47], 0 offen offset: 0 lds // (op 560) Load value direct to lds
v_add_co_u32 v12, s[14:15], v12, v16 // (op 560) least significant half
v_addc_co_u32 v13, s[14:15], 0, v13, s[14:15] // (op 560) most significant half

s_add_u32 m0, m0, 1024 // (op 560) 
buffer_load_dwordx4 v12, s[44:47], 0 offen offset: 0 lds // (op 560) Load value direct to lds



s_lshl_b64 s[14:15], s[8:9], 8 // (op 661) 
v_mov_b32 v20, s14 // (op 661) 
v_mov_b32 v22, s15 // (op 661) 
v_add_co_u32 v18, s[38:39], v18, v20 // (op 661) least significant half
v_addc_co_u32 v19, s[38:39], v19, v22, s[38:39] // (op 661) most significant half





v_mov_b64 v[16:17], s[12:13] // (op 226) convert

v_mov_b32 v20, v15 // (op 226) 
v_mov_b32 v22, v20 // (op 226) reg expression
buffer_load_ubyte v26, v22, s[48:51], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v22, v22, v16 // (op 226) 
buffer_load_ubyte v27, v22, s[48:51], 0 offen offset: 0 // (op 226) Load value



s_lshl_b32 s19, s12, 3 // (op 725) 
v_add_u32 v15, v15, s19 // (op 725) 


s_barrier  // (op 1325) 
v_mov_b64 v[16:17], s[24:25] // (op 227) convert




v_mov_b32 v20, v21 // (op 227) 
v_mov_b32 v22, v20 // (op 227) reg expression
buffer_load_ubyte v12, v22, s[52:55], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v22, v22, v16 // (op 227) 
buffer_load_ubyte v13, v22, s[52:55], 0 offen offset: 0 // (op 227) Load value



s_lshl_b32 s19, s24, 3 // (op 789) 
v_add_u32 v21, v21, s19 // (op 789) 

s_waitcnt vmcnt(3)
ds_write_b8 v23, v26 offset:16512 // (op 223) Store local data 
s_barrier  // (op 1298) 
s_waitcnt vmcnt(2)
ds_write_b8 v23, v27 offset:16528 // (op 223) Store local data 




ds_read_b64_tr_b8 v[30:31], v0 offset:4096 // (op 205) Transpose load local data 
s_barrier  // (op 1342) 
ds_read_b64_tr_b8 v[32:33], v0 offset:4224 // (op 205) Transpose load local data 



ds_read_b64_tr_b8 v[34:35], v0 offset:5120 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:5248 // (op 205) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v24, v12 offset:16384 // (op 224) Store local data 
s_barrier  // (op 1315) 
s_waitcnt vmcnt(0)
ds_write_b8 v24, v13 offset:16400 // (op 224) Store local data 




ds_read_b64_tr_b8 v[38:39], v3 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:128 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:1024 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:1152 // (op 207) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1332) 


ds_read_u8 v20, v5 offset:16512 // (op 209) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1349) 


ds_read_u8 v22, v7 offset:16384 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 212) 







ds_read_b64_tr_b8 v[30:31], v0 offset:6144 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:6272 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:7168 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:7296 // (op 214) Transpose load local data 



ds_read_b64_tr_b8 v[38:39], v3 offset:2048 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:2176 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:3072 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[44:45], v3 offset:3200 // (op 216) Transpose load local data 


ds_read_u8 v20, v5 offset:16576 // (op 218) Load local data 


ds_read_u8 v22, v7 offset:16448 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 221) 




s_ashr_i64 s[14:15], s[6:7], 8 // (op 197) 
s_mov_b32 s19, s14 // (op 197) convert

s_cmp_lt_u32 s18, s19 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 if true)
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197:























s_barrier  // (op 1418) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

v_cvt_f16_f32 v4, v12 // (op 56) 
v_cvt_f16_f32 v5, v13 // (op 56) 
v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 1089) 
v_and_b32 v3, 63, v1 // (op 1089) 
v_lshrrev_b32 v9, 4, v3 // (op 1089) 
v_lshlrev_b32 v11, 2, v9 // (op 1089) 

v_lshl_add_u32 v9, v0, 4, v11 // (op 1089) 


v_lshl_add_u32 v0, s16, 4, v9 // (op 1089) 

v_lshrrev_b32 v9, 2, v3 // (op 1089) 

v_and_b32 v3, 3, v9 // (op 1089) 

v_and_b32 v9, 3, v1 // (op 1089) 
v_lshl_add_u32 v11, v3, 2, v9 // (op 1089) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 1089) 

v_mul_lo_u32 v9, v3, s28 // (op 1089) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 1089) 


s_mul_i32 s1, s28, s33 // (op 1091) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1091) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1091) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1091) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1091) most significant: sum
s_add_u32 s1, s1, s3 // (op 1091) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1091) 

s_mov_b32 s4, s30 // (op 1091) reg expression
s_mov_b32 s5, s31 // (op 1091) reg expression
s_mov_b32 s6, s2 // (op 1091) reg expression
s_mov_b32 s7, 131072 // (op 1091) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[4:7], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157:
v_lshrrev_b32 v0, 6, v1 // (op 860) 
v_and_b32 v9, 63, v1 // (op 860) 
v_lshrrev_b32 v3, 4, v9 // (op 860) 
v_lshlrev_b32 v4, 2, v3 // (op 860) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 860) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 860) 

v_lshrrev_b32 v3, 2, v9 // (op 860) 

v_and_b32 v9, 3, v3 // (op 860) 

v_and_b32 v3, 3, v1 // (op 860) 
v_lshl_add_u32 v4, v9, 2, v3 // (op 860) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 860) 

v_mul_lo_u32 v9, v3, s34 // (op 860) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 860) 


s_mul_i32 s1, s34, s33 // (op 862) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 862) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 862) least significant
s_mul_hi_u32 s3, s34, s32 // (op 862) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 862) most significant: sum
s_add_u32 s1, s1, s3 // (op 862) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 862) 

s_mov_b32 s8, s36 // (op 862) reg expression
s_mov_b32 s9, s37 // (op 862) reg expression
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
v_cvt_f32_f16 v6, v4 // (op 171) 
v_lshrrev_b32 v7, 16, v4 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v12, v8, v6 // (op 171) 
v_mul_f32 v13, v8, v7 // (op 171) 
v_cvt_f32_f16 v6, v5 // (op 171) 
v_lshrrev_b32 v7, 16, v5 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v14, v8, v6 // (op 171) 
v_mul_f32 v15, v8, v7 // (op 171) 


v_add_f32 v4, v12, v16 // (op 167) 
v_add_f32 v5, v13, v17 // (op 167) 
v_add_f32 v6, v14, v18 // (op 167) 
v_add_f32 v7, v15, v19 // (op 167) 


v_cvt_f16_f32 v12, v4 // (op 168) 
v_cvt_f16_f32 v13, v5 // (op 168) 
v_cvt_f16_f32 v14, v6 // (op 168) 
v_cvt_f16_f32 v15, v7 // (op 168) 

v_lshrrev_b32 v0, 6, v1 // (op 1106) 
v_and_b32 v9, 63, v1 // (op 1106) 
v_lshrrev_b32 v11, 4, v9 // (op 1106) 
v_lshlrev_b32 v3, 2, v11 // (op 1106) 

v_lshl_add_u32 v11, v0, 4, v3 // (op 1106) 


v_lshl_add_u32 v0, s16, 4, v11 // (op 1106) 

v_lshrrev_b32 v11, 2, v9 // (op 1106) 

v_and_b32 v9, 3, v11 // (op 1106) 

v_and_b32 v11, 3, v1 // (op 1106) 
v_lshl_add_u32 v3, v9, 2, v11 // (op 1106) 


v_lshl_add_u32 v9, s17, 4, v3 // (op 1106) 

v_mul_lo_u32 v11, v9, s28 // (op 1106) 

v_add_lshl_u32 v9, v0, v11, 1 // (op 1106) 


s_mul_i32 s1, s28, s33 // (op 1108) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1108) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1108) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1108) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1108) most significant: sum
s_add_u32 s1, s1, s3 // (op 1108) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1108) 

s_mov_b32 s4, s30 // (op 1108) reg expression
s_mov_b32 s5, s31 // (op 1108) reg expression
s_mov_b32 s6, s2 // (op 1108) reg expression
s_mov_b32 s7, 131072 // (op 1108) call()


v_pack_b32_f16 v6, v12, v13 // (op 169) 
v_pack_b32_f16 v7, v14, v15 // (op 169) 


buffer_store_dwordx2 v[6:7], v9, s[4:7], 0 offen // (op 169) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157:



s_endpgm  // End of RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
  .amdhsa_accum_offset 80
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
    DataTypeB: B8
    DestDataType: h
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
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
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
  - .name: RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NT_FP8_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 78
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
      - .name: Tensor_4_stride_1_5
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
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 64
        .value_kind: by_value
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
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_10_stride_1_10
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

