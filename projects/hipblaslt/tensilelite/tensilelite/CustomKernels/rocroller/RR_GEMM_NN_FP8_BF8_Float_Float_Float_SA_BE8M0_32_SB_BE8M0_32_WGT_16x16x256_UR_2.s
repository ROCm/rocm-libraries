// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 913) 
v_and_b32 v3, 63, v1 // (op 913) 
v_sub_u32 v4, v3, v0 // (op 913) 

v_lshrrev_b32 v3, 1, v4 // (op 913) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 913) 


v_and_b32 v0, 1, v4 // (op 913) 

v_lshlrev_b32 v3, 3, v0 // (op 913) 

v_lshrrev_b32 v0, 6, v1 // (op 913) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 913) 


v_add_u32 v0, v5, v4 // (op 913) 


v_and_b32 v3, 15, v1 // (op 965) 
v_and_b32 v4, 48, v1 // (op 965) 
v_lshl_add_u32 v5, v3, 8, v4 // (op 965) 


v_lshrrev_b32 v3, 6, v1 // (op 1018) 
v_and_b32 v4, 15, v1 // (op 1018) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 1018) 


v_and_b32 v3, 48, v1 // (op 1018) 
v_add_u32 v4, v3, v6 // (op 1018) 


v_and_b32 v3, 15, v1 // (op 1071) 
v_and_b32 v6, 63, v1 // (op 1071) 
v_lshrrev_b32 v7, 4, v6 // (op 1071) 

v_lshl_add_u32 v6, v3, 3, v7 // (op 1071) 


v_and_b32 v3, 48, v1 // (op 1144) 
v_and_b32 v7, 63, v1 // (op 1144) 
v_sub_u32 v8, v7, v3 // (op 1144) 

v_lshrrev_b32 v7, 1, v8 // (op 1144) 
v_add_lshl_u32 v9, v3, v7, 4 // (op 1144) 


v_and_b32 v3, 1, v8 // (op 1144) 

v_lshlrev_b32 v7, 3, v3 // (op 1144) 

v_lshrrev_b32 v3, 6, v1 // (op 1144) 
v_lshl_add_u32 v8, v3, 4, v7 // (op 1144) 


v_add_u32 v3, v9, v8 // (op 1144) 


v_and_b32 v7, 15, v1 // (op 1183) 
v_and_b32 v8, 48, v1 // (op 1183) 
v_lshl_add_u32 v9, v7, 8, v8 // (op 1183) 


v_lshrrev_b32 v7, 6, v1 // (op 1207) 
v_and_b32 v8, 15, v1 // (op 1207) 
v_lshl_add_u32 v10, v7, 4, v8 // (op 1207) 


v_and_b32 v7, 48, v1 // (op 1207) 
v_add_u32 v8, v7, v10 // (op 1207) 


v_and_b32 v7, 15, v1 // (op 1231) 
v_and_b32 v10, 63, v1 // (op 1231) 
v_lshrrev_b32 v11, 4, v10 // (op 1231) 

v_lshl_add_u32 v10, v7, 3, v11 // (op 1231) 


v_mov_b32 v7, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_lshl_b32 s0, s16, 4 // (op 624) 
v_mul_lo_u32 v13, v1, s3 // (op 624) most significant: low of low * high
s_mov_b32 s15, 0 // (op 624) convert
s_mov_b32 s14, s0 // (op 624) convert

v_mul_hi_u32 v14, v1, s2 // (op 624) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 624) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 624) most significant: sum

v_mov_b32 v14, s14 // (op 624) 
v_mov_b32 v15, s15 // (op 624) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 624) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 624) most significant half





s_mul_i32 s1, s2, s7 // (op 626) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 626) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 626) least significant
s_mul_hi_u32 s15, s2, s6 // (op 626) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 626) most significant: sum
s_add_u32 s1, s1, s15 // (op 626) most significant: sum


s_mov_b32 s40, s4 // (op 626) reg expression
s_mov_b32 s41, s5 // (op 626) reg expression
s_mov_b32 s42, s0 // (op 626) reg expression
s_mov_b32 s43, 131072 // (op 626) call()

s_lshl_b32 s0, s16, 4 // (op 643) 
v_mul_lo_u32 v13, v1, s3 // (op 643) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 643) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 643) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 643) most significant: sum

s_mov_b32 s15, 0 // (op 643) convert
s_mov_b32 s14, s0 // (op 643) convert

v_mov_b32 v14, s14 // (op 643) 
v_mov_b32 v15, s15 // (op 643) 
v_add_co_u32 v18, s[0:1], v14, v12 // (op 643) least significant half
v_addc_co_u32 v19, s[0:1], v15, v13, s[0:1] // (op 643) most significant half





s_mul_i32 s1, s2, s7 // (op 645) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 645) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 645) least significant
s_mul_hi_u32 s15, s2, s6 // (op 645) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 645) most significant: sum
s_add_u32 s1, s1, s15 // (op 645) most significant: sum


s_mov_b32 s40, s4 // (op 645) reg expression
s_mov_b32 s41, s5 // (op 645) reg expression
s_mov_b32 s42, s0 // (op 645) reg expression
s_mov_b32 s43, 131072 // (op 645) call()

v_and_b32 v12, -16, v1 // (op 688) 
v_sub_u32 v13, v1, v12 // (op 688) 

v_lshlrev_b32 v12, 4, v13 // (op 688) 

v_mov_b32 v15, 0 // (op 688) convert
v_mov_b32 v14, v12 // (op 688) convert

v_lshrrev_b32 v12, 4, v1 // (op 688) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 688) 

v_mul_lo_u32 v21, v13, s9 // (op 688) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 688) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 688) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 688) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 688) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 688) most significant half



s_mul_i32 s1, s8, s13 // (op 690) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 690) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 690) least significant
s_mul_hi_u32 s15, s8, s12 // (op 690) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 690) most significant: sum
s_add_u32 s1, s1, s15 // (op 690) most significant: sum


s_mov_b32 s44, s10 // (op 690) reg expression
s_mov_b32 s45, s11 // (op 690) reg expression
s_mov_b32 s46, s0 // (op 690) reg expression
s_mov_b32 s47, 131072 // (op 690) call()

v_and_b32 v14, -16, v1 // (op 707) 
v_sub_u32 v15, v1, v14 // (op 707) 

v_lshlrev_b32 v14, 4, v15 // (op 707) 

v_mov_b32 v21, 0 // (op 707) convert
v_mov_b32 v20, v14 // (op 707) convert

v_lshrrev_b32 v14, 4, v1 // (op 707) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 707) 

v_mul_lo_u32 v23, v15, s9 // (op 707) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 707) most significant: high of low * low
v_mul_lo_u32 v22, v15, s8 // (op 707) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 707) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 707) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 707) most significant half



s_mul_i32 s1, s8, s13 // (op 709) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 709) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 709) least significant
s_mul_hi_u32 s15, s8, s12 // (op 709) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 709) most significant: sum
s_add_u32 s1, s1, s15 // (op 709) most significant: sum


s_mov_b32 s44, s10 // (op 709) reg expression
s_mov_b32 s45, s11 // (op 709) reg expression
s_mov_b32 s46, s0 // (op 709) reg expression
s_mov_b32 s47, 131072 // (op 709) call()

v_and_b32 v20, -16, v1 // (op 752) 
v_sub_u32 v21, v1, v20 // (op 752) 

v_lshl_add_u32 v20, s16, 4, v21 // (op 752) 

v_lshrrev_b32 v21, 4, v1 // (op 752) 
v_lshlrev_b32 v22, 1, v21 // (op 752) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v21, v22, s20 // (op 752) 

v_add_u32 v22, v20, v21 // (op 752) 


s_mul_i32 s1, s20, s25 // (op 754) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 754) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 754) least significant
s_mul_hi_u32 s15, s20, s24 // (op 754) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 754) most significant: sum
s_add_u32 s1, s1, s15 // (op 754) most significant: sum


s_mov_b32 s48, s22 // (op 754) reg expression
s_mov_b32 s49, s23 // (op 754) reg expression
s_mov_b32 s50, s0 // (op 754) reg expression
s_mov_b32 s51, 131072 // (op 754) call()

v_and_b32 v20, -16, v1 // (op 771) 
v_sub_u32 v21, v1, v20 // (op 771) 

v_lshl_add_u32 v20, s16, 4, v21 // (op 771) 

v_lshrrev_b32 v21, 4, v1 // (op 771) 
v_lshlrev_b32 v23, 1, v21 // (op 771) 

v_mul_lo_u32 v21, v23, s20 // (op 771) 

v_add_u32 v23, v20, v21 // (op 771) 


s_mul_i32 s1, s20, s25 // (op 773) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 773) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 773) least significant
s_mul_hi_u32 s15, s20, s24 // (op 773) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 773) most significant: sum
s_add_u32 s1, s1, s15 // (op 773) most significant: sum


s_mov_b32 s48, s22 // (op 773) reg expression
s_mov_b32 s49, s23 // (op 773) reg expression
s_mov_b32 s50, s0 // (op 773) reg expression
s_mov_b32 s51, 131072 // (op 773) call()

v_and_b32 v20, -8, v1 // (op 816) 
v_sub_u32 v21, v1, v20 // (op 816) 

v_lshrrev_b32 v20, 3, v1 // (op 816) 
v_lshlrev_b32 v24, 1, v20 // (op 816) 

v_lshl_add_u32 v20, s17, 4, v24 // (op 816) 

v_mul_lo_u32 v24, v20, s26 // (op 816) 

v_add_u32 v20, v21, v24 // (op 816) 


s_mul_i32 s1, s26, s13 // (op 818) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 818) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 818) least significant
s_mul_hi_u32 s15, s26, s12 // (op 818) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 818) most significant: sum
s_add_u32 s1, s1, s15 // (op 818) most significant: sum


s_mov_b32 s52, s28 // (op 818) reg expression
s_mov_b32 s53, s29 // (op 818) reg expression
s_mov_b32 s54, s0 // (op 818) reg expression
s_mov_b32 s55, 131072 // (op 818) call()

v_and_b32 v21, -8, v1 // (op 835) 
v_sub_u32 v24, v1, v21 // (op 835) 

v_lshrrev_b32 v21, 3, v1 // (op 835) 
v_lshlrev_b32 v25, 1, v21 // (op 835) 

v_lshl_add_u32 v21, s17, 4, v25 // (op 835) 

v_mul_lo_u32 v25, v21, s26 // (op 835) 

v_add_u32 v21, v24, v25 // (op 835) 


s_mul_i32 s1, s26, s13 // (op 837) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 837) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 837) least significant
s_mul_hi_u32 s15, s26, s12 // (op 837) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 837) most significant: sum
s_add_u32 s1, s1, s15 // (op 837) most significant: sum


s_mov_b32 s52, s28 // (op 837) reg expression
s_mov_b32 s53, s29 // (op 837) reg expression
s_mov_b32 s54, s0 // (op 837) reg expression
s_mov_b32 s55, 131072 // (op 837) call()

v_and_b32 v24, -16, v1 // (op 953) 
v_sub_u32 v25, v1, v24 // (op 953) 

v_lshlrev_b32 v24, 4, v25 // (op 953) 

v_lshrrev_b32 v25, 4, v1 // (op 953) 
v_lshl_add_u32 v26, v25, 8, v24 // (op 953) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 953) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 2 // (op 573) 
v_mov_b64 v[24:25], s[14:15] // (op 573) convert


v_mov_b64 v[26:27], v[12:13] // (op 573) 
s_mov_b32 m0, s0 // (op 573) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 573) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 573) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 573) most significant half

s_add_u32 m0, m0, 1024 // (op 573) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 573) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 573) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 573) most significant half

s_add_u32 m0, m0, 1024 // (op 573) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 573) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 573) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 573) most significant half

s_add_u32 m0, m0, 1024 // (op 573) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 573) Load value direct to lds




v_mov_b64 v[24:25], s[26:27] // (op 427) convert

v_mov_b32 v26, v20 // (op 427) 
buffer_load_ubyte v28, v26, s[52:55], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v26, v26, v24 // (op 427) 
buffer_load_ubyte v29, v26, s[52:55], 0 offen offset: 0 // (op 427) Load value


v_lshlrev_b32 v24, 4, v1 // (op 901) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 901) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 578) 
v_mov_b64 v[24:25], s[14:15] // (op 578) convert


v_mov_b64 v[26:27], v[16:17] // (op 578) reg expression
s_add_u32 s1, 4096, s0 // (op 578) 
s_mov_b32 m0, s1 // (op 578) reg expression
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 578) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 578) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 578) most significant half

s_add_u32 m0, m0, 1024 // (op 578) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 578) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 578) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 578) most significant half

s_add_u32 m0, m0, 1024 // (op 578) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 578) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 578) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 578) most significant half

s_add_u32 m0, m0, 1024 // (op 578) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 578) Load value direct to lds





v_mov_b64 v[24:25], s[20:21] // (op 433) convert

v_mov_b32 v26, v22 // (op 433) reg expression
buffer_load_ubyte v30, v26, s[48:51], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v26, v26, v24 // (op 433) 
buffer_load_ubyte v31, v26, s[48:51], 0 offen offset: 0 // (op 433) Load value


v_and_b32 v24, -16, v1 // (op 1171) 
v_sub_u32 v25, v1, v24 // (op 1171) 

v_lshlrev_b32 v24, 4, v25 // (op 1171) 

v_lshrrev_b32 v25, 4, v1 // (op 1171) 
v_lshl_add_u32 v26, v25, 8, v24 // (op 1171) 


s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1171) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 2 // (op 583) 
v_mov_b64 v[24:25], s[14:15] // (op 583) convert


v_mov_b64 v[26:27], v[12:13] // (op 583) 
s_add_u32 s1, 8192, s0 // (op 583) 
s_mov_b32 m0, s1 // (op 583) reg expression
s_mov_b32 s14, 256 // (op 583) call()
buffer_load_dwordx4 v26, s[44:47], s14 offen offset: 0 lds // (op 583) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 583) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 583) most significant half

s_add_u32 m0, m0, 1024 // (op 583) 
buffer_load_dwordx4 v26, s[44:47], s14 offen offset: 0 lds // (op 583) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 583) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 583) most significant half

s_add_u32 m0, m0, 1024 // (op 583) 
buffer_load_dwordx4 v26, s[44:47], s14 offen offset: 0 lds // (op 583) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v24 // (op 583) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 583) most significant half

s_add_u32 m0, m0, 1024 // (op 583) 
buffer_load_dwordx4 v26, s[44:47], s14 offen offset: 0 lds // (op 583) Load value direct to lds







v_mov_b64 v[12:13], s[26:27] // (op 439) convert

v_mov_b32 v24, v20 // (op 439) 
buffer_load_ubyte v26, v24, s[52:55], 0 offen offset:8 // (op 439) Load value
v_add_u32 v24, v24, v12 // (op 439) 
buffer_load_ubyte v27, v24, s[52:55], 0 offen offset:8 // (op 439) Load value



v_lshlrev_b32 v20, 4, v1 // (op 1132) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1132) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 585) 
v_mov_b64 v[12:13], s[14:15] // (op 585) convert


s_lshl_b64 s[14:15], s[2:3], 8 // (op 585) 
v_mov_b32 v20, s14 // (op 585) 
v_mov_b32 v24, s15 // (op 585) 
v_add_co_u32 v32, s[18:19], v16, v20 // (op 585) least significant half
v_addc_co_u32 v33, s[18:19], v17, v24, s[18:19] // (op 585) most significant half




s_add_u32 s1, 12288, s0 // (op 585) 
s_mov_b32 m0, s1 // (op 585) reg expression
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 585) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v12 // (op 585) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 585) most significant half

s_add_u32 m0, m0, 1024 // (op 585) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 585) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v12 // (op 585) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 585) most significant half

s_add_u32 m0, m0, 1024 // (op 585) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 585) Load value direct to lds
v_add_co_u32 v32, s[14:15], v32, v12 // (op 585) least significant half
v_addc_co_u32 v33, s[14:15], 0, v33, s[14:15] // (op 585) most significant half

s_add_u32 m0, m0, 1024 // (op 585) 
buffer_load_dwordx4 v32, s[40:43], 0 offen offset: 0 lds // (op 585) Load value direct to lds






v_mov_b64 v[12:13], s[20:21] // (op 445) convert

s_lshl_b32 s0, s20, 3 // (op 445) 
v_add_u32 v20, v22, s0 // (op 445) 

buffer_load_ubyte v16, v20, s[48:51], 0 offen offset: 0 // (op 445) Load value
v_add_u32 v20, v20, v12 // (op 445) 
buffer_load_ubyte v17, v20, s[48:51], 0 offen offset: 0 // (op 445) Load value



v_and_b32 v20, -8, v1 // (op 1059) 
v_sub_u32 v22, v1, v20 // (op 1059) 

v_lshrrev_b32 v20, 3, v1 // (op 1059) 
v_lshl_add_u32 v12, v20, 4, v22 // (op 1059) 





s_waitcnt vmcnt(19)
ds_write_b8 v12, v28 offset:16384 // (op 448) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v12, v29 offset:16392 // (op 448) Store local data 

v_and_b32 v20, -16, v1 // (op 1006) 
v_sub_u32 v22, v1, v20 // (op 1006) 

v_lshrrev_b32 v20, 4, v1 // (op 1006) 
v_lshl_add_u32 v12, v20, 5, v22 // (op 1006) 





s_waitcnt vmcnt(13)
ds_write_b8 v12, v30 offset:16512 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v12, v31 offset:16528 // (op 450) Store local data 

s_barrier  // (op 1298) 



ds_read_b128 v[32:35], v5 // (op 470) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 470) Load local data 



ds_read_b128 v[40:43], v5 offset:128 // (op 473) Load local data 
ds_read_b128 v[44:47], v5 offset:192 // (op 473) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1332) 



ds_read_u8 v20, v6 offset:16384 // (op 482) Load local data 



ds_read_u8 v22, v6 offset:16388 // (op 485) Load local data 
s_barrier  // (op 1282) 



ds_read_b64_tr_b8 v[48:49], v0 offset:4096 // (op 464) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:4224 // (op 464) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5120 // (op 464) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:5248 // (op 464) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:6144 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:6272 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:7168 // (op 467) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:7296 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1315) 


ds_read_u8 v12, v4 offset:16512 // (op 476) Load local data 


ds_read_u8 v13, v4 offset:16576 // (op 479) Load local data 
v_lshlrev_b32 v24, 4, v1 // (op 873) 
s_nop 0
v_readfirstlane_b32 s0, v24 // (op 873) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 925) 
v_sub_u32 v25, v1, v24 // (op 925) 

v_lshlrev_b32 v24, 4, v25 // (op 925) 

v_lshrrev_b32 v25, 4, v1 // (op 925) 
v_lshl_add_u32 v64, v25, 8, v24 // (op 925) 


s_nop 0
v_readfirstlane_b32 s1, v64 // (op 925) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 977) 
v_sub_u32 v25, v1, v24 // (op 977) 

v_lshrrev_b32 v24, 4, v1 // (op 977) 
v_lshl_add_u32 v64, v24, 5, v25 // (op 977) 


v_and_b32 v24, -8, v1 // (op 1030) 
v_sub_u32 v25, v1, v24 // (op 1030) 

v_lshrrev_b32 v24, 3, v1 // (op 1030) 
v_lshl_add_u32 v65, v24, 4, v25 // (op 1030) 


v_lshlrev_b32 v24, 4, v1 // (op 1117) 
s_nop 0
v_readfirstlane_b32 s14, v24 // (op 1117) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 1156) 
v_sub_u32 v25, v1, v24 // (op 1156) 

v_lshlrev_b32 v24, 4, v25 // (op 1156) 

v_lshrrev_b32 v25, 4, v1 // (op 1156) 
v_lshl_add_u32 v66, v25, 8, v24 // (op 1156) 


s_nop 0
v_readfirstlane_b32 s15, v66 // (op 1156) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -16, v1 // (op 1195) 
v_sub_u32 v25, v1, v24 // (op 1195) 

v_lshrrev_b32 v24, 4, v1 // (op 1195) 
v_lshl_add_u32 v66, v24, 5, v25 // (op 1195) 


v_and_b32 v24, -8, v1 // (op 1219) 
v_sub_u32 v25, v1, v24 // (op 1219) 

v_lshrrev_b32 v24, 3, v1 // (op 1219) 
v_lshl_add_u32 v67, v24, 4, v25 // (op 1219) 


s_mov_b32 s18, 0 // (op 59) call()
s_ashr_i64 s[38:39], s[6:7], 8 // (op 58) 
s_mov_b32 s19, s38 // (op 58) convert

s_and_b32 s38, s19, 4294967294 // (op 58) 

s_cmp_lt_u32 s18, s38 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58:
s_add_u32 s18, s18, 2 // (op 60) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 88) 
s_barrier  // (op 1301) 

s_lshl_b64 s[38:39], s[8:9], 2 // (op 545) 
v_mov_b64 v[24:25], s[38:39] // (op 545) convert


v_mov_b64 v[68:69], v[14:15] // (op 545) 
v_mov_b64 v[70:71], v[68:69] // (op 545) 
s_mov_b32 m0, s1 // (op 545) reg expression
s_mov_b32 s19, 512 // (op 545) call()
buffer_load_dwordx4 v70, s[44:47], s19 offen offset: 0 lds // (op 545) Load value direct to lds
v_add_co_u32 v70, s[38:39], v70, v24 // (op 545) least significant half
v_addc_co_u32 v71, s[38:39], 0, v71, s[38:39] // (op 545) most significant half

s_add_u32 m0, m0, 1024 // (op 545) 
buffer_load_dwordx4 v70, s[44:47], s19 offen offset: 0 lds // (op 545) Load value direct to lds
v_add_co_u32 v70, s[38:39], v70, v24 // (op 545) least significant half
v_addc_co_u32 v71, s[38:39], 0, v71, s[38:39] // (op 545) most significant half

s_add_u32 m0, m0, 1024 // (op 545) 
buffer_load_dwordx4 v70, s[44:47], s19 offen offset: 0 lds // (op 545) Load value direct to lds
v_add_co_u32 v70, s[38:39], v70, v24 // (op 545) least significant half
v_addc_co_u32 v71, s[38:39], 0, v71, s[38:39] // (op 545) most significant half

s_add_u32 m0, m0, 1024 // (op 545) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 105) 
buffer_load_dwordx4 v70, s[44:47], s19 offen offset: 0 lds // (op 545) Load value direct to lds




v_mov_b64 v[24:25], s[26:27] // (op 25) convert

v_mov_b32 v70, v21 // (op 25) 
v_mov_b32 v71, v70 // (op 25) 
buffer_load_ubyte v28, v71, s[52:55], 0 offen offset:16 // (op 25) Load value
v_add_u32 v71, v71, v24 // (op 25) 
buffer_load_ubyte v29, v71, s[52:55], 0 offen offset:16 // (op 25) Load value


s_barrier  // (op 1285) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 540) 
v_mov_b64 v[24:25], s[38:39] // (op 540) convert


v_mov_b64 v[72:73], v[18:19] // (op 540) 
s_lshl_b64 s[38:39], s[2:3], 8 // (op 540) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 540) 

v_mov_b32 v71, s56 // (op 540) 
v_mov_b32 v74, s57 // (op 540) 
v_add_co_u32 v76, s[38:39], v72, v71 // (op 540) least significant half
v_addc_co_u32 v77, s[38:39], v73, v74, s[38:39] // (op 540) most significant half




s_add_u32 s19, 4096, s0 // (op 540) 
s_mov_b32 m0, s19 // (op 540) reg expression
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 540) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v24 // (op 540) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 540) most significant half

s_add_u32 m0, m0, 1024 // (op 540) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 540) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v24 // (op 540) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 540) most significant half

s_add_u32 m0, m0, 1024 // (op 540) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 540) Load value direct to lds
v_add_co_u32 v76, s[38:39], v76, v24 // (op 540) least significant half
v_addc_co_u32 v77, s[38:39], 0, v77, s[38:39] // (op 540) most significant half

s_add_u32 m0, m0, 1024 // (op 540) 
buffer_load_dwordx4 v76, s[40:43], 0 offen offset: 0 lds // (op 540) Load value direct to lds




v_mov_b64 v[24:25], s[20:21] // (op 17) convert

v_mov_b32 v71, v23 // (op 17) 
s_lshl_b32 s19, s20, 4 // (op 17) 
v_add_u32 v74, v71, s19 // (op 17) 

buffer_load_ubyte v30, v74, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v74, v74, v24 // (op 17) 
buffer_load_ubyte v31, v74, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1371) 



s_waitcnt vmcnt(19)
ds_write_b8 v67, v26 offset:16640 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v67, v27 offset:16648 // (op 304) Store local data 
s_barrier  // (op 1365) 



s_waitcnt vmcnt(13)
ds_write_b8 v66, v16 offset:16768 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v66, v17 offset:16784 // (op 303) Store local data 
s_barrier  // (op 1355) 



ds_read_b128 v[32:35], v9 offset:8192 // (op 287) Load local data 
ds_read_b128 v[36:39], v9 offset:8256 // (op 287) Load local data 



ds_read_b128 v[40:43], v9 offset:8320 // (op 296) Load local data 
ds_read_b128 v[44:47], v9 offset:8384 // (op 296) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1368) 



ds_read_u8 v20, v10 offset:16640 // (op 291) Load local data 



ds_read_u8 v22, v10 offset:16644 // (op 300) Load local data 
s_barrier  // (op 1349) 
s_barrier  // (op 1424) 



ds_read_b64_tr_b8 v[48:49], v3 offset:12288 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v3 offset:12416 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v3 offset:13312 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v3 offset:13440 // (op 285) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v3 offset:14336 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v3 offset:14464 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v3 offset:15360 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v3 offset:15488 // (op 294) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1362) 


ds_read_u8 v12, v8 offset:16768 // (op 289) Load local data 


ds_read_u8 v13, v8 offset:16832 // (op 298) Load local data 
s_barrier  // (op 1420) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:55], v[32:39], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 292) 
s_barrier  // (op 1359) 

s_lshl_b64 s[38:39], s[8:9], 2 // (op 568) 
v_mov_b64 v[24:25], s[38:39] // (op 568) convert


v_mov_b64 v[74:75], v[68:69] // (op 568) 
s_add_u32 s19, 8192, s15 // (op 568) 
s_mov_b32 m0, s19 // (op 568) reg expression
s_mov_b32 s38, 768 // (op 568) call()
buffer_load_dwordx4 v74, s[44:47], s38 offen offset: 0 lds // (op 568) Load value direct to lds
v_add_co_u32 v74, s[56:57], v74, v24 // (op 568) least significant half
v_addc_co_u32 v75, s[56:57], 0, v75, s[56:57] // (op 568) most significant half

s_add_u32 m0, m0, 1024 // (op 568) 
buffer_load_dwordx4 v74, s[44:47], s38 offen offset: 0 lds // (op 568) Load value direct to lds
v_add_co_u32 v74, s[56:57], v74, v24 // (op 568) least significant half
v_addc_co_u32 v75, s[56:57], 0, v75, s[56:57] // (op 568) most significant half

s_add_u32 m0, m0, 1024 // (op 568) 
buffer_load_dwordx4 v74, s[44:47], s38 offen offset: 0 lds // (op 568) Load value direct to lds
v_add_co_u32 v74, s[56:57], v74, v24 // (op 568) least significant half
v_addc_co_u32 v75, s[56:57], 0, v75, s[56:57] // (op 568) most significant half

s_add_u32 m0, m0, 1024 // (op 568) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:63], v[40:47], a[0:3], v13, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 301) 
buffer_load_dwordx4 v74, s[44:47], s38 offen offset: 0 lds // (op 568) Load value direct to lds






v_add_co_u32 v14, vcc, 512, v14 // (op 699) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 699) most significant half
v_mov_b64 v[24:25], s[26:27] // (op 307) convert

v_mov_b32 v68, v70 // (op 307) 
buffer_load_ubyte v26, v68, s[52:55], 0 offen offset:24 // (op 307) Load value
v_add_u32 v68, v68, v24 // (op 307) 
buffer_load_ubyte v27, v68, s[52:55], 0 offen offset:24 // (op 307) Load value



s_barrier  // (op 1352) 
v_add_u32 v21, 16, v21 // (op 827) 

s_lshl_b64 s[38:39], s[2:3], 6 // (op 563) 
v_mov_b64 v[24:25], s[38:39] // (op 563) convert


s_lshl_b64 s[38:39], s[2:3], 8 // (op 563) 
s_mul_i32 s57, 3, s39 // (op 563) most significant: low of low * high
s_mul_i32 s56, 3, s38 // (op 563) least significant
s_mul_hi_u32 s19, 3, s38 // (op 563) most significant: high of low * low
s_add_u32 s57, s57, s19 // (op 563) most significant: sum


v_mov_b32 v68, s56 // (op 563) 
v_mov_b32 v69, s57 // (op 563) 
v_add_co_u32 v74, s[38:39], v72, v68 // (op 563) least significant half
v_addc_co_u32 v75, s[38:39], v73, v69, s[38:39] // (op 563) most significant half




s_add_u32 s19, 12288, s14 // (op 563) 
s_mov_b32 m0, s19 // (op 563) reg expression
buffer_load_dwordx4 v74, s[40:43], 0 offen offset: 0 lds // (op 563) Load value direct to lds
v_add_co_u32 v74, s[38:39], v74, v24 // (op 563) least significant half
v_addc_co_u32 v75, s[38:39], 0, v75, s[38:39] // (op 563) most significant half

s_add_u32 m0, m0, 1024 // (op 563) 
buffer_load_dwordx4 v74, s[40:43], 0 offen offset: 0 lds // (op 563) Load value direct to lds
v_add_co_u32 v74, s[38:39], v74, v24 // (op 563) least significant half
v_addc_co_u32 v75, s[38:39], 0, v75, s[38:39] // (op 563) most significant half

s_add_u32 m0, m0, 1024 // (op 563) 
buffer_load_dwordx4 v74, s[40:43], 0 offen offset: 0 lds // (op 563) Load value direct to lds
v_add_co_u32 v74, s[38:39], v74, v24 // (op 563) least significant half
v_addc_co_u32 v75, s[38:39], 0, v75, s[38:39] // (op 563) most significant half

s_add_u32 m0, m0, 1024 // (op 563) 
buffer_load_dwordx4 v74, s[40:43], 0 offen offset: 0 lds // (op 563) Load value direct to lds





v_mov_b64 v[24:25], s[20:21] // (op 306) convert

s_lshl_b32 s19, s20, 3 // (op 306) 
s_mul_i32 s38, 3, s19 // (op 306) 

v_add_u32 v68, v71, s38 // (op 306) 

buffer_load_ubyte v16, v68, s[48:51], 0 offen offset: 0 // (op 306) Load value
v_add_u32 v68, v68, v24 // (op 306) 
s_lshl_b64 s[38:39], s[2:3], 8 // (op 635) 
s_lshl_b64 s[56:57], s[38:39], 1 // (op 635) 

buffer_load_ubyte v17, v68, s[48:51], 0 offen offset: 0 // (op 306) Load value


v_mov_b32 v24, s56 // (op 635) 

v_mov_b32 v25, s57 // (op 635) 
s_barrier  // (op 1335) 
s_lshl_b32 s19, s20, 4 // (op 763) 



v_add_u32 v23, v23, s19 // (op 763) 

v_add_co_u32 v18, s[38:39], v18, v24 // (op 635) least significant half
v_addc_co_u32 v19, s[38:39], v19, v25, s[38:39] // (op 635) most significant half




s_waitcnt vmcnt(19)
ds_write_b8 v65, v28 offset:16384 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v65, v29 offset:16392 // (op 30) Store local data 
s_barrier  // (op 1318) 



s_waitcnt vmcnt(13)
ds_write_b8 v64, v30 offset:16512 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v64, v31 offset:16528 // (op 21) Store local data 
s_barrier  // (op 1307) 



ds_read_b128 v[32:35], v5 // (op 75) Load local data 
ds_read_b128 v[36:39], v5 offset:64 // (op 75) Load local data 



ds_read_b128 v[40:43], v5 offset:128 // (op 95) Load local data 
ds_read_b128 v[44:47], v5 offset:192 // (op 95) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1342) 



ds_read_u8 v20, v6 offset:16384 // (op 85) Load local data 



ds_read_u8 v22, v6 offset:16388 // (op 103) Load local data 
s_barrier  // (op 1291) 



ds_read_b64_tr_b8 v[48:49], v0 offset:4096 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[50:51], v0 offset:4224 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[52:53], v0 offset:5120 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[54:55], v0 offset:5248 // (op 70) Transpose load local data 



ds_read_b64_tr_b8 v[56:57], v0 offset:6144 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:6272 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:7168 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:7296 // (op 91) Transpose load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 1325) 


ds_read_u8 v12, v4 offset:16512 // (op 80) Load local data 


ds_read_u8 v13, v4 offset:16576 // (op 99) Load local data 
s_ashr_i64 s[38:39], s[6:7], 8 // (op 58) 
s_mov_b32 s19, s38 // (op 58) convert

s_and_b32 s38, s19, 4294967294 // (op 58) 

s_cmp_lt_u32 s18, s38 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58:















s_barrier  // (op 1382) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 598) 
s_lshr_b32 s14, s0, 1 // (op 598) 

v_lshl_add_u32 v21, s14, 9, v1 // (op 598) 

s_lshl_b32 s0, s16, 4 // (op 598) 
v_mul_lo_u32 v15, v21, s3 // (op 598) most significant: low of low * high
v_mul_hi_u32 v18, v21, s2 // (op 598) most significant: high of low * low
v_mul_lo_u32 v14, v21, s2 // (op 598) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 598) most significant: sum


s_mov_b32 s15, 0 // (op 598) convert
s_mov_b32 s14, s0 // (op 598) convert

v_mov_b32 v21, s14 // (op 598) 
v_mov_b32 v18, s15 // (op 598) 
v_add_co_u32 v24, s[0:1], v21, v14 // (op 598) least significant half
v_addc_co_u32 v25, s[0:1], v18, v15, s[0:1] // (op 598) most significant half





s_mul_i32 s1, s2, s7 // (op 600) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 600) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 600) least significant
s_mul_hi_u32 s15, s2, s6 // (op 600) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 600) most significant: sum
s_add_u32 s1, s1, s15 // (op 600) most significant: sum


s_mov_b32 s40, s4 // (op 600) reg expression
s_mov_b32 s41, s5 // (op 600) reg expression
s_mov_b32 s42, s0 // (op 600) reg expression
s_mov_b32 s43, 131072 // (op 600) call()

v_and_b32 v21, -16, v1 // (op 665) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 665) 
v_sub_u32 v23, v1, v21 // (op 665) 

s_lshr_b32 s14, s0, 1 // (op 665) 

v_lshlrev_b32 v21, 4, v23 // (op 665) 

v_lshl_add_u32 v23, s14, 9, v21 // (op 665) 


v_mov_b32 v15, 0 // (op 665) convert
v_mov_b32 v14, v23 // (op 665) convert

v_lshrrev_b32 v21, 4, v1 // (op 665) 
v_lshl_add_u32 v23, s17, 4, v21 // (op 665) 

v_mul_lo_u32 v19, v23, s9 // (op 665) most significant: low of low * high
v_mul_hi_u32 v21, v23, s8 // (op 665) most significant: high of low * low
v_mul_lo_u32 v18, v23, s8 // (op 665) least significant: low of low * low
v_add_u32 v19, v19, v21 // (op 665) most significant: sum


v_add_co_u32 v28, s[0:1], v14, v18 // (op 665) least significant half
v_addc_co_u32 v29, s[0:1], v15, v19, s[0:1] // (op 665) most significant half



s_mul_i32 s1, s8, s13 // (op 667) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 667) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 667) least significant
s_mul_hi_u32 s15, s8, s12 // (op 667) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 667) most significant: sum
s_add_u32 s1, s1, s15 // (op 667) most significant: sum


s_mov_b32 s44, s10 // (op 667) reg expression
s_mov_b32 s45, s11 // (op 667) reg expression
s_mov_b32 s46, s0 // (op 667) reg expression
s_mov_b32 s47, 131072 // (op 667) call()

v_and_b32 v21, -16, v1 // (op 729) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 729) 
v_sub_u32 v23, v1, v21 // (op 729) 

s_lshr_b32 s14, s0, 1 // (op 729) 

v_lshl_add_u32 v21, s16, 4, v23 // (op 729) 

v_lshrrev_b32 v23, 4, v1 // (op 729) 
v_lshlrev_b32 v14, 1, v23 // (op 729) 

v_lshl_add_u32 v15, s14, 4, v14 // (op 729) 


v_mul_lo_u32 v14, v15, s20 // (op 729) 

v_add_u32 v15, v21, v14 // (op 729) 


s_mul_i32 s1, s20, s25 // (op 731) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 731) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 731) least significant
s_mul_hi_u32 s15, s20, s24 // (op 731) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 731) most significant: sum
s_add_u32 s1, s1, s15 // (op 731) most significant: sum


s_mov_b32 s48, s22 // (op 731) reg expression
s_mov_b32 s49, s23 // (op 731) reg expression
s_mov_b32 s50, s0 // (op 731) reg expression
s_mov_b32 s51, 131072 // (op 731) call()

v_and_b32 v14, -8, v1 // (op 793) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 793) 
v_sub_u32 v21, v1, v14 // (op 793) 

s_lshr_b32 s14, s0, 1 // (op 793) 

v_lshrrev_b32 v14, 3, v1 // (op 793) 
v_lshl_add_u32 v23, s14, 4, v21 // (op 793) 


v_lshlrev_b32 v21, 1, v14 // (op 793) 

v_lshl_add_u32 v14, s17, 4, v21 // (op 793) 

v_mul_lo_u32 v21, v14, s26 // (op 793) 

v_add_u32 v14, v23, v21 // (op 793) 


s_mul_i32 s1, s26, s13 // (op 795) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 795) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 795) least significant
s_mul_hi_u32 s15, s26, s12 // (op 795) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 795) most significant: sum
s_add_u32 s1, s1, s15 // (op 795) most significant: sum


s_mov_b32 s52, s28 // (op 795) reg expression
s_mov_b32 s53, s29 // (op 795) reg expression
s_mov_b32 s54, s0 // (op 795) reg expression
s_mov_b32 s55, 131072 // (op 795) call()

v_lshlrev_b32 v21, 4, v1 // (op 886) 
s_nop 0
v_readfirstlane_b32 s0, v21 // (op 886) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 938) 
v_sub_u32 v23, v1, v21 // (op 938) 

v_lshlrev_b32 v21, 4, v23 // (op 938) 

v_lshrrev_b32 v23, 4, v1 // (op 938) 
v_lshl_add_u32 v18, v23, 8, v21 // (op 938) 


s_nop 0
v_readfirstlane_b32 s1, v18 // (op 938) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -16, v1 // (op 990) 
v_sub_u32 v23, v1, v21 // (op 990) 

v_lshrrev_b32 v21, 4, v1 // (op 990) 
v_lshl_add_u32 v18, v21, 5, v23 // (op 990) 


v_and_b32 v19, -8, v1 // (op 1043) 
v_sub_u32 v21, v1, v19 // (op 1043) 

v_lshrrev_b32 v19, 3, v1 // (op 1043) 
v_lshl_add_u32 v23, v19, 4, v21 // (op 1043) 


s_ashr_i64 s[14:15], s[6:7], 8 // (op 194) 
s_and_b32 s18, s14, 4294967294 // (op 194) 

s_ashr_i64 s[14:15], s[6:7], 8 // (op 193) 
s_mov_b32 s19, s14 // (op 193) convert

s_cmp_lt_u32 s18, s19 // (op 193) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193:
s_barrier  // (op 1288) 
s_add_u32 s18, s18, 1 // (op 195) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 550) 
v_mov_b64 v[12:13], s[14:15] // (op 550) convert


v_mov_b64 v[16:17], v[24:25] // (op 550) 
v_mov_b64 v[26:27], v[16:17] // (op 550) reg expression
s_add_u32 s19, 4096, s0 // (op 550) 
s_mov_b32 m0, s19 // (op 550) reg expression
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 550) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 550) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 550) most significant half

s_add_u32 m0, m0, 1024 // (op 550) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 550) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 550) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 550) most significant half

s_add_u32 m0, m0, 1024 // (op 550) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 550) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 550) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 550) most significant half

s_add_u32 m0, m0, 1024 // (op 550) 
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 550) Load value direct to lds




s_lshl_b64 s[14:15], s[2:3], 8 // (op 588) 
v_mov_b32 v12, s14 // (op 588) 
v_mov_b32 v13, s15 // (op 588) 
v_add_co_u32 v24, s[38:39], v24, v12 // (op 588) least significant half
v_addc_co_u32 v25, s[38:39], v25, v13, s[38:39] // (op 588) most significant half




s_barrier  // (op 1304) 

s_lshl_b64 s[14:15], s[8:9], 2 // (op 556) 
v_mov_b64 v[12:13], s[14:15] // (op 556) convert


v_mov_b64 v[16:17], v[28:29] // (op 556) 
v_mov_b64 v[26:27], v[16:17] // (op 556) 
s_mov_b32 m0, s1 // (op 556) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 556) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 556) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 556) most significant half

s_add_u32 m0, m0, 1024 // (op 556) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 556) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 556) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 556) most significant half

s_add_u32 m0, m0, 1024 // (op 556) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 556) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v12 // (op 556) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 556) most significant half

s_add_u32 m0, m0, 1024 // (op 556) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 556) Load value direct to lds



v_add_co_u32 v28, vcc, 256, v28 // (op 657) least significant half
v_addc_co_u32 v29, vcc, 0, v29, vcc // (op 657) most significant half

v_mov_b64 v[12:13], s[20:21] // (op 222) convert

v_mov_b32 v16, v15 // (op 222) 
v_mov_b32 v17, v16 // (op 222) reg expression
buffer_load_ubyte v26, v17, s[48:51], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v17, v17, v12 // (op 222) 
buffer_load_ubyte v27, v17, s[48:51], 0 offen offset: 0 // (op 222) Load value



s_lshl_b32 s19, s20, 3 // (op 721) 
v_add_u32 v15, v15, s19 // (op 721) 


s_barrier  // (op 1321) 
v_mov_b64 v[12:13], s[26:27] // (op 223) convert




v_mov_b32 v16, v14 // (op 223) 
v_mov_b32 v17, v16 // (op 223) 
buffer_load_ubyte v20, v17, s[52:55], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v17, v17, v12 // (op 223) 
buffer_load_ubyte v21, v17, s[52:55], 0 offen offset: 0 // (op 223) Load value



v_add_u32 v14, 8, v14 // (op 785) 
s_waitcnt vmcnt(3)
ds_write_b8 v18, v26 offset:16512 // (op 219) Store local data 
s_barrier  // (op 1294) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v27 offset:16528 // (op 219) Store local data 




ds_read_b64_tr_b8 v[30:31], v0 offset:4096 // (op 201) Transpose load local data 
s_barrier  // (op 1338) 
ds_read_b64_tr_b8 v[32:33], v0 offset:4224 // (op 201) Transpose load local data 



ds_read_b64_tr_b8 v[34:35], v0 offset:5120 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:5248 // (op 201) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v20 offset:16384 // (op 220) Store local data 
s_barrier  // (op 1311) 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v21 offset:16392 // (op 220) Store local data 




ds_read_b128 v[38:41], v5 // (op 203) Load local data 
ds_read_b128 v[42:45], v5 offset:64 // (op 203) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1328) 


ds_read_u8 v12, v4 offset:16512 // (op 205) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1345) 



ds_read_u8 v13, v6 offset:16384 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 208) 







ds_read_b64_tr_b8 v[30:31], v0 offset:6144 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:6272 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:7168 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:7296 // (op 210) Transpose load local data 



ds_read_b128 v[38:41], v5 offset:128 // (op 212) Load local data 
ds_read_b128 v[42:45], v5 offset:192 // (op 212) Load local data 


ds_read_u8 v12, v4 offset:16576 // (op 214) Load local data 



ds_read_u8 v13, v6 offset:16388 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:37], v[38:45], a[0:3], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b001 // (op 217) 




s_ashr_i64 s[14:15], s[6:7], 8 // (op 193) 
s_mov_b32 s19, s14 // (op 193) convert

s_cmp_lt_u32 s18, s19 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 if true)
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1410) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 1085) 
v_and_b32 v3, 63, v1 // (op 1085) 
v_lshrrev_b32 v4, 4, v3 // (op 1085) 
v_lshlrev_b32 v5, 2, v4 // (op 1085) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1085) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1085) 

v_lshrrev_b32 v4, 2, v3 // (op 1085) 

v_and_b32 v3, 3, v4 // (op 1085) 

v_and_b32 v4, 3, v1 // (op 1085) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1085) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1085) 

v_mul_lo_u32 v3, v6, s30 // (op 1085) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 1085) 


s_mul_i32 s15, s30, s13 // (op 1087) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1087) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1087) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1087) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1087) most significant: sum
s_add_u32 s15, s15, s1 // (op 1087) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 1087) 

s_mov_b32 s8, s32 // (op 1087) reg expression
s_mov_b32 s9, s33 // (op 1087) reg expression
s_mov_b32 s10, s0 // (op 1087) reg expression
s_mov_b32 s11, 131072 // (op 1087) call()




buffer_store_dwordx4 v[16:19], v4, s[8:11], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155:
v_lshrrev_b32 v0, 6, v1 // (op 856) 
v_and_b32 v3, 63, v1 // (op 856) 
v_lshrrev_b32 v4, 4, v3 // (op 856) 
v_lshlrev_b32 v5, 2, v4 // (op 856) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 856) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 856) 

v_lshrrev_b32 v4, 2, v3 // (op 856) 

v_and_b32 v3, 3, v4 // (op 856) 

v_and_b32 v4, 3, v1 // (op 856) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 856) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 856) 

v_mul_lo_u32 v3, v6, s34 // (op 856) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 856) 


s_mul_i32 s15, s34, s13 // (op 858) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 858) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 858) least significant
s_mul_hi_u32 s1, s34, s12 // (op 858) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 858) most significant: sum
s_add_u32 s15, s15, s1 // (op 858) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 858) 

s_mov_b32 s4, s36 // (op 858) reg expression
s_mov_b32 s5, s37 // (op 858) reg expression
s_mov_b32 s6, s0 // (op 858) reg expression
s_mov_b32 s7, 131072 // (op 858) call()





buffer_load_dwordx4 v[12:15], v4, s[4:7], 0 offen // (op 167) Load value



v_accvgpr_read v16, a0 // (op 164) 
v_accvgpr_read v17, a1 // (op 164) 
v_accvgpr_read v18, a2 // (op 164) 
v_accvgpr_read v19, a3 // (op 164) 
v_mul_f32 v20, v11, v16 // (op 164) 
v_mul_f32 v21, v11, v17 // (op 164) 
v_mul_f32 v22, v11, v18 // (op 164) 
v_mul_f32 v23, v11, v19 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v7, v12 // (op 168) 
v_mul_f32 v17, v7, v13 // (op 168) 
v_mul_f32 v18, v7, v14 // (op 168) 
v_mul_f32 v19, v7, v15 // (op 168) 

v_add_f32 v12, v16, v20 // (op 165) 
v_add_f32 v13, v17, v21 // (op 165) 
v_add_f32 v14, v18, v22 // (op 165) 
v_add_f32 v15, v19, v23 // (op 165) 


v_lshrrev_b32 v0, 6, v1 // (op 1102) 
v_and_b32 v3, 63, v1 // (op 1102) 
v_lshrrev_b32 v4, 4, v3 // (op 1102) 
v_lshlrev_b32 v5, 2, v4 // (op 1102) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1102) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1102) 

v_lshrrev_b32 v4, 2, v3 // (op 1102) 

v_and_b32 v3, 3, v4 // (op 1102) 

v_and_b32 v4, 3, v1 // (op 1102) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1102) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1102) 

v_mul_lo_u32 v3, v6, s30 // (op 1102) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 1102) 


s_mul_i32 s15, s30, s13 // (op 1104) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1104) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1104) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1104) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1104) most significant: sum
s_add_u32 s15, s15, s1 // (op 1104) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 1104) 

s_mov_b32 s8, s32 // (op 1104) reg expression
s_mov_b32 s9, s33 // (op 1104) reg expression
s_mov_b32 s10, s0 // (op 1104) reg expression
s_mov_b32 s11, 131072 // (op 1104) call()




buffer_store_dwordx4 v[12:15], v4, s[8:11], 0 offen // (op 166) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155:




s_endpgm  // End of RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
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
    DestDataType: s
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
  - .name: RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NN_FP8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 58
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

