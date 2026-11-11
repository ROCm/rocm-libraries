// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 875) 
v_and_b32 v3, 63, v1 // (op 875) 
v_sub_u32 v4, v3, v0 // (op 875) 

v_lshrrev_b32 v3, 1, v4 // (op 875) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 875) 


v_and_b32 v0, 1, v4 // (op 875) 

v_lshlrev_b32 v3, 3, v0 // (op 875) 

v_lshrrev_b32 v0, 6, v1 // (op 875) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 875) 


v_add_u32 v0, v5, v4 // (op 875) 


v_and_b32 v3, 15, v1 // (op 928) 
v_and_b32 v4, 63, v1 // (op 928) 
v_lshrrev_b32 v5, 4, v4 // (op 928) 

v_lshlrev_b32 v4, 5, v5 // (op 928) 

v_lshl_add_u32 v5, v3, 8, v4 // (op 928) 


v_mul_lo_u32 v3, v5, 6 // (op 928) 

v_lshrrev_b32 v4, 3, v3 // (op 928) 

v_lshrrev_b32 v3, 6, v1 // (op 981) 
v_and_b32 v5, 15, v1 // (op 981) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 981) 


v_and_b32 v3, 48, v1 // (op 981) 
v_add_u32 v5, v3, v6 // (op 981) 


v_and_b32 v3, 15, v1 // (op 1034) 
v_and_b32 v6, 63, v1 // (op 1034) 
v_lshrrev_b32 v7, 4, v6 // (op 1034) 

v_lshl_add_u32 v6, v3, 3, v7 // (op 1034) 


v_and_b32 v3, 48, v1 // (op 1107) 
v_and_b32 v7, 63, v1 // (op 1107) 
v_sub_u32 v8, v7, v3 // (op 1107) 

v_lshrrev_b32 v7, 1, v8 // (op 1107) 
v_add_lshl_u32 v9, v3, v7, 4 // (op 1107) 


v_and_b32 v3, 1, v8 // (op 1107) 

v_lshlrev_b32 v7, 3, v3 // (op 1107) 

v_lshrrev_b32 v3, 6, v1 // (op 1107) 
v_lshl_add_u32 v8, v3, 4, v7 // (op 1107) 


v_add_u32 v3, v9, v8 // (op 1107) 


v_and_b32 v7, 15, v1 // (op 1131) 
v_and_b32 v8, 63, v1 // (op 1131) 
v_lshrrev_b32 v9, 4, v8 // (op 1131) 

v_lshlrev_b32 v8, 5, v9 // (op 1131) 

v_lshl_add_u32 v9, v7, 8, v8 // (op 1131) 


v_mul_lo_u32 v7, v9, 6 // (op 1131) 

v_lshrrev_b32 v8, 3, v7 // (op 1131) 

v_lshrrev_b32 v7, 6, v1 // (op 1155) 
v_and_b32 v9, 15, v1 // (op 1155) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1155) 


v_and_b32 v7, 48, v1 // (op 1155) 
v_add_u32 v9, v7, v10 // (op 1155) 


v_and_b32 v7, 15, v1 // (op 1179) 
v_and_b32 v10, 63, v1 // (op 1179) 
v_lshrrev_b32 v11, 4, v10 // (op 1179) 

v_lshl_add_u32 v10, v7, 3, v11 // (op 1179) 


v_mov_b32 v7, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
s_lshl_b32 s0, s16, 4 // (op 586) 
v_mul_lo_u32 v13, v1, s3 // (op 586) most significant: low of low * high
s_mov_b32 s15, 0 // (op 586) convert
s_mov_b32 s14, s0 // (op 586) convert

v_mul_hi_u32 v14, v1, s2 // (op 586) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 586) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 586) most significant: sum

v_mov_b32 v14, s14 // (op 586) 
v_mov_b32 v15, s15 // (op 586) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 586) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 586) most significant half





s_mul_i32 s1, s2, s7 // (op 588) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 588) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 588) least significant
s_mul_hi_u32 s15, s2, s6 // (op 588) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 588) most significant: sum
s_add_u32 s1, s1, s15 // (op 588) most significant: sum


s_mov_b32 s40, s4 // (op 588) reg expression
s_mov_b32 s41, s5 // (op 588) reg expression
s_mov_b32 s42, s0 // (op 588) reg expression
s_mov_b32 s43, 131072 // (op 588) call()

s_lshl_b32 s0, s16, 4 // (op 605) 
v_mul_lo_u32 v13, v1, s3 // (op 605) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 605) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 605) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 605) most significant: sum

s_mov_b32 s15, 0 // (op 605) convert
s_mov_b32 s14, s0 // (op 605) convert

v_mov_b32 v14, s14 // (op 605) 
v_mov_b32 v15, s15 // (op 605) 
v_add_co_u32 v18, s[0:1], v14, v12 // (op 605) least significant half
v_addc_co_u32 v19, s[0:1], v15, v13, s[0:1] // (op 605) most significant half





s_mul_i32 s1, s2, s7 // (op 607) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 607) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 607) least significant
s_mul_hi_u32 s15, s2, s6 // (op 607) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 607) most significant: sum
s_add_u32 s1, s1, s15 // (op 607) most significant: sum


s_mov_b32 s40, s4 // (op 607) reg expression
s_mov_b32 s41, s5 // (op 607) reg expression
s_mov_b32 s42, s0 // (op 607) reg expression
s_mov_b32 s43, 131072 // (op 607) call()

v_and_b32 v12, -4, v1 // (op 650) 
v_sub_u32 v13, v1, v12 // (op 650) 

v_lshlrev_b32 v12, 6, v13 // (op 650) 

v_mov_b32 v15, 0 // (op 650) convert
v_mov_b32 v14, v12 // (op 650) convert

v_lshrrev_b32 v12, 2, v1 // (op 650) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 650) 

v_mul_lo_u32 v21, v13, s9 // (op 650) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 650) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 650) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 650) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 650) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 650) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 650) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 650) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 650) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 650) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 650) 

v_mov_b32 v14, v12 // (op 650) convert

s_mul_i32 s1, s13, 6 // (op 652) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 652) least significant
s_mul_hi_u32 s14, s12, 6 // (op 652) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 652) most significant: sum

s_mul_i32 s15, s8, s1 // (op 652) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 652) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 652) least significant
s_mul_hi_u32 s19, s8, s0 // (op 652) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 652) most significant: sum
s_add_u32 s15, s15, s19 // (op 652) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 652) 

s_mov_b32 s44, s10 // (op 652) reg expression
s_mov_b32 s45, s11 // (op 652) reg expression
s_mov_b32 s46, s0 // (op 652) reg expression
s_mov_b32 s47, 131072 // (op 652) call()

v_and_b32 v15, -4, v1 // (op 669) 
v_sub_u32 v12, v1, v15 // (op 669) 

v_lshlrev_b32 v13, 6, v12 // (op 669) 

v_mov_b32 v21, 0 // (op 669) convert
v_mov_b32 v20, v13 // (op 669) convert

v_lshrrev_b32 v15, 2, v1 // (op 669) 
v_lshl_add_u32 v12, s17, 4, v15 // (op 669) 

v_mul_lo_u32 v23, v12, s9 // (op 669) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 669) most significant: high of low * low
v_mul_lo_u32 v22, v12, s8 // (op 669) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 669) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v22 // (op 669) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 669) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 669) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 669) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 669) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 669) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 669) 

v_mov_b32 v15, v12 // (op 669) convert

s_mul_i32 s1, s13, 6 // (op 671) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 671) least significant
s_mul_hi_u32 s14, s12, 6 // (op 671) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 671) most significant: sum

s_mul_i32 s15, s8, s1 // (op 671) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 671) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 671) least significant
s_mul_hi_u32 s19, s8, s0 // (op 671) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 671) most significant: sum
s_add_u32 s15, s15, s19 // (op 671) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 671) 

s_mov_b32 s44, s10 // (op 671) reg expression
s_mov_b32 s45, s11 // (op 671) reg expression
s_mov_b32 s46, s0 // (op 671) reg expression
s_mov_b32 s47, 131072 // (op 671) call()

v_and_b32 v12, -16, v1 // (op 714) 
v_sub_u32 v13, v1, v12 // (op 714) 

v_lshl_add_u32 v12, s16, 4, v13 // (op 714) 

v_lshrrev_b32 v13, 4, v1 // (op 714) 
v_lshlrev_b32 v20, 1, v13 // (op 714) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v13, v20, s20 // (op 714) 

v_add_u32 v20, v12, v13 // (op 714) 


s_mul_i32 s1, s20, s25 // (op 716) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 716) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 716) least significant
s_mul_hi_u32 s15, s20, s24 // (op 716) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 716) most significant: sum
s_add_u32 s1, s1, s15 // (op 716) most significant: sum


s_mov_b32 s48, s22 // (op 716) reg expression
s_mov_b32 s49, s23 // (op 716) reg expression
s_mov_b32 s50, s0 // (op 716) reg expression
s_mov_b32 s51, 131072 // (op 716) call()

v_and_b32 v12, -16, v1 // (op 733) 
v_sub_u32 v13, v1, v12 // (op 733) 

v_lshl_add_u32 v12, s16, 4, v13 // (op 733) 

v_lshrrev_b32 v13, 4, v1 // (op 733) 
v_lshlrev_b32 v21, 1, v13 // (op 733) 

v_mul_lo_u32 v13, v21, s20 // (op 733) 

v_add_u32 v21, v12, v13 // (op 733) 


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

v_and_b32 v12, -8, v1 // (op 778) 
v_sub_u32 v13, v1, v12 // (op 778) 

v_lshrrev_b32 v12, 3, v1 // (op 778) 
v_lshlrev_b32 v22, 1, v12 // (op 778) 

v_lshl_add_u32 v12, s17, 4, v22 // (op 778) 

v_mul_lo_u32 v22, v12, s26 // (op 778) 

v_add_u32 v12, v13, v22 // (op 778) 


s_mul_i32 s1, s26, s13 // (op 780) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 780) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 780) least significant
s_mul_hi_u32 s15, s26, s12 // (op 780) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 780) most significant: sum
s_add_u32 s1, s1, s15 // (op 780) most significant: sum


s_mov_b32 s52, s28 // (op 780) reg expression
s_mov_b32 s53, s29 // (op 780) reg expression
s_mov_b32 s54, s0 // (op 780) reg expression
s_mov_b32 s55, 131072 // (op 780) call()

v_and_b32 v13, -8, v1 // (op 797) 
v_sub_u32 v22, v1, v13 // (op 797) 

v_lshrrev_b32 v13, 3, v1 // (op 797) 
v_lshlrev_b32 v23, 1, v13 // (op 797) 

v_lshl_add_u32 v13, s17, 4, v23 // (op 797) 

v_mul_lo_u32 v23, v13, s26 // (op 797) 

v_add_u32 v13, v22, v23 // (op 797) 


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




buffer_load_dwordx4 v[22:25], v14, s[44:47], 0 offen // (op 428) Load value
buffer_load_dwordx2 v[26:27], v14, s[44:47], 0 offen offset:16 // (op 428) Load value
buffer_load_dwordx4 v[28:31], v14, s[44:47], 0 offen offset:24 // (op 428) Load value
buffer_load_dwordx2 v[32:33], v14, s[44:47], 0 offen offset:40 // (op 428) Load value

v_mov_b64 v[34:35], s[26:27] // (op 431) convert

v_mov_b32 v36, v12 // (op 431) 
buffer_load_ubyte v38, v36, s[52:55], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v36, v36, v34 // (op 431) 
buffer_load_ubyte v39, v36, s[52:55], 0 offen offset: 0 // (op 431) Load value


v_lshlrev_b32 v34, 4, v1 // (op 863) 
s_nop 0
v_readfirstlane_b32 s0, v34 // (op 863) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 542) 
v_mov_b64 v[34:35], s[14:15] // (op 542) convert


v_mov_b64 v[36:37], v[16:17] // (op 542) reg expression
s_mov_b32 m0, s0 // (op 542) reg expression
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 542) Load value direct to lds
v_add_co_u32 v36, s[14:15], v36, v34 // (op 542) least significant half
v_addc_co_u32 v37, s[14:15], 0, v37, s[14:15] // (op 542) most significant half

s_add_u32 m0, m0, 1024 // (op 542) 
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 542) Load value direct to lds
v_add_co_u32 v36, s[14:15], v36, v34 // (op 542) least significant half
v_addc_co_u32 v37, s[14:15], 0, v37, s[14:15] // (op 542) most significant half

s_add_u32 m0, m0, 1024 // (op 542) 
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 542) Load value direct to lds
v_add_co_u32 v36, s[14:15], v36, v34 // (op 542) least significant half
v_addc_co_u32 v37, s[14:15], 0, v37, s[14:15] // (op 542) most significant half

s_add_u32 m0, m0, 1024 // (op 542) 
buffer_load_dwordx4 v36, s[40:43], 0 offen offset: 0 lds // (op 542) Load value direct to lds




v_mov_b64 v[34:35], s[20:21] // (op 437) convert

v_mov_b32 v36, v20 // (op 437) reg expression
buffer_load_ubyte v40, v36, s[48:51], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v36, v36, v34 // (op 437) 
buffer_load_ubyte v41, v36, s[48:51], 0 offen offset: 0 // (op 437) Load value





buffer_load_dwordx4 v[42:45], v14, s[44:47], 0 offen offset:192 // (op 440) Load value
buffer_load_dwordx2 v[46:47], v14, s[44:47], 0 offen offset:208 // (op 440) Load value
buffer_load_dwordx4 v[48:51], v14, s[44:47], 0 offen offset:216 // (op 440) Load value
buffer_load_dwordx2 v[52:53], v14, s[44:47], 0 offen offset:232 // (op 440) Load value


v_mov_b64 v[34:35], s[26:27] // (op 443) convert

v_mov_b32 v14, v12 // (op 443) 
buffer_load_ubyte v36, v14, s[52:55], 0 offen offset:8 // (op 443) Load value
v_add_u32 v14, v14, v34 // (op 443) 
buffer_load_ubyte v37, v14, s[52:55], 0 offen offset:8 // (op 443) Load value



v_lshlrev_b32 v12, 4, v1 // (op 1095) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1095) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 547) 
v_mov_b64 v[34:35], s[14:15] // (op 547) convert


s_lshl_b64 s[14:15], s[2:3], 8 // (op 547) 
v_mov_b32 v12, s14 // (op 547) 
v_mov_b32 v14, s15 // (op 547) 
v_add_co_u32 v54, s[18:19], v16, v12 // (op 547) least significant half
v_addc_co_u32 v55, s[18:19], v17, v14, s[18:19] // (op 547) most significant half




s_add_u32 s1, 4096, s0 // (op 547) 
s_mov_b32 m0, s1 // (op 547) reg expression
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v54, s[14:15], v54, v34 // (op 547) least significant half
v_addc_co_u32 v55, s[14:15], 0, v55, s[14:15] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v54, s[14:15], v54, v34 // (op 547) least significant half
v_addc_co_u32 v55, s[14:15], 0, v55, s[14:15] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v54, s[14:15], v54, v34 // (op 547) least significant half
v_addc_co_u32 v55, s[14:15], 0, v55, s[14:15] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v54, s[40:43], 0 offen offset: 0 lds // (op 547) Load value direct to lds






v_mov_b64 v[16:17], s[20:21] // (op 449) convert

s_lshl_b32 s0, s20, 3 // (op 449) 
v_add_u32 v12, v20, s0 // (op 449) 

buffer_load_ubyte v34, v12, s[48:51], 0 offen offset: 0 // (op 449) Load value
v_add_u32 v12, v12, v16 // (op 449) 
buffer_load_ubyte v35, v12, s[48:51], 0 offen offset: 0 // (op 449) Load value



v_and_b32 v12, -4, v1 // (op 916) 
v_sub_u32 v14, v1, v12 // (op 916) 

v_lshlrev_b32 v12, 6, v14 // (op 916) 

v_lshrrev_b32 v14, 2, v1 // (op 916) 
v_lshl_add_u32 v20, v14, 8, v12 // (op 916) 


v_mul_lo_u32 v12, v20, 6 // (op 916) 

v_lshrrev_b32 v14, 3, v12 // (op 916) 




s_waitcnt vmcnt(23)
ds_write_b128 v14, v[22:25] offset:8192 // (op 451) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v14, v[26:27] offset:8208 // (op 451) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v14, v[28:31] offset:8216 // (op 451) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v14, v[32:33] offset:8232 // (op 451) Store local data 

v_and_b32 v12, -8, v1 // (op 1022) 
v_sub_u32 v14, v1, v12 // (op 1022) 

v_lshrrev_b32 v12, 3, v1 // (op 1022) 
v_lshl_add_u32 v20, v12, 4, v14 // (op 1022) 





s_waitcnt vmcnt(19)
ds_write_b8 v20, v38 offset:11264 // (op 452) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v20, v39 offset:11272 // (op 452) Store local data 

v_and_b32 v12, -16, v1 // (op 969) 
v_sub_u32 v14, v1, v12 // (op 969) 

v_lshrrev_b32 v12, 4, v1 // (op 969) 
v_lshl_add_u32 v20, v12, 5, v14 // (op 969) 





s_waitcnt vmcnt(13)
ds_write_b8 v20, v40 offset:11392 // (op 454) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v20, v41 offset:11408 // (op 454) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1235) 



ds_read_b128 v[54:57], v4 offset:8192 // (op 471) Load local data 
ds_read_b64 v[58:59], v4 offset:8208 // (op 471) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1276) 



ds_read_u8 v12, v6 offset:11264 // (op 477) Load local data 
s_barrier  // (op 1216) 



ds_read_b64_tr_b8 v[60:61], v0 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:128 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:1024 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v0 offset:1152 // (op 468) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1256) 


ds_read_u8 v14, v5 offset:11392 // (op 474) Load local data 
v_lshlrev_b32 v20, 4, v1 // (op 835) 
s_nop 0
v_readfirstlane_b32 s0, v20 // (op 835) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v20, -4, v1 // (op 887) 
v_sub_u32 v16, v1, v20 // (op 887) 

v_lshlrev_b32 v17, 6, v16 // (op 887) 

v_lshrrev_b32 v16, 2, v1 // (op 887) 
v_lshl_add_u32 v20, v16, 8, v17 // (op 887) 


v_mul_lo_u32 v16, v20, 6 // (op 887) 

v_lshrrev_b32 v17, 3, v16 // (op 887) 

v_and_b32 v16, -16, v1 // (op 940) 
v_sub_u32 v20, v1, v16 // (op 940) 

v_lshrrev_b32 v16, 4, v1 // (op 940) 
v_lshl_add_u32 v68, v16, 5, v20 // (op 940) 


v_and_b32 v16, -8, v1 // (op 993) 
v_sub_u32 v20, v1, v16 // (op 993) 

v_lshrrev_b32 v16, 3, v1 // (op 993) 
v_lshl_add_u32 v69, v16, 4, v20 // (op 993) 


v_lshlrev_b32 v16, 4, v1 // (op 1080) 
s_nop 0
v_readfirstlane_b32 s1, v16 // (op 1080) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 1119) 
v_sub_u32 v20, v1, v16 // (op 1119) 

v_lshlrev_b32 v16, 6, v20 // (op 1119) 

v_lshrrev_b32 v20, 2, v1 // (op 1119) 
v_lshl_add_u32 v70, v20, 8, v16 // (op 1119) 


v_mul_lo_u32 v16, v70, 6 // (op 1119) 

v_lshrrev_b32 v20, 3, v16 // (op 1119) 

v_and_b32 v16, -16, v1 // (op 1143) 
v_sub_u32 v70, v1, v16 // (op 1143) 

v_lshrrev_b32 v16, 4, v1 // (op 1143) 
v_lshl_add_u32 v71, v16, 5, v70 // (op 1143) 


v_and_b32 v16, -8, v1 // (op 1167) 
v_sub_u32 v70, v1, v16 // (op 1167) 

v_lshrrev_b32 v16, 3, v1 // (op 1167) 
v_lshl_add_u32 v72, v16, 4, v70 // (op 1167) 


s_mov_b32 s14, 0 // (op 61) call()
s_ashr_i64 s[18:19], s[6:7], 8 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60:


s_add_u32 s14, s14, 2 // (op 62) 
v_mov_b32 v16, v15 // (op 10) 

buffer_load_dwordx4 v[22:25], v16, s[44:47], 0 offen offset:384 // (op 10) Load value
buffer_load_dwordx2 v[26:27], v16, s[44:47], 0 offen offset:400 // (op 10) Load value
buffer_load_dwordx4 v[28:31], v16, s[44:47], 0 offen offset:408 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:67], v[54:59], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 90) 
buffer_load_dwordx2 v[32:33], v16, s[44:47], 0 offen offset:424 // (op 10) Load value

s_barrier  // (op 1225) 
v_mov_b64 v[74:75], s[26:27] // (op 25) convert




ds_read_b64_tr_b8 v[60:61], v0 offset:2048 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:2176 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:3072 // (op 93) Transpose load local data 
v_mov_b32 v70, v13 // (op 25) 
v_mov_b32 v73, v70 // (op 25) 
buffer_load_ubyte v38, v73, s[52:55], 0 offen offset:16 // (op 25) Load value
v_add_u32 v73, v73, v74 // (op 25) 
buffer_load_ubyte v39, v73, s[52:55], 0 offen offset:16 // (op 25) Load value


ds_read_b64_tr_b8 v[66:67], v0 offset:3200 // (op 93) Transpose load local data 
s_barrier  // (op 1245) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1219) 




ds_read_b128 v[54:57], v4 offset:8288 // (op 97) Load local data 
ds_read_b64 v[58:59], v4 offset:8304 // (op 97) Load local data 
s_lshl_b64 s[18:19], s[2:3], 6 // (op 526) 
s_barrier  // (op 1266) 
v_mov_b64 v[74:75], s[18:19] // (op 526) convert




ds_read_u8 v14, v5 offset:11456 // (op 101) Load local data 
v_mov_b64 v[76:77], v[18:19] // (op 526) 
s_barrier  // (op 1286) 
s_lshl_b64 s[18:19], s[2:3], 8 // (op 526) 



s_lshl_b64 s[38:39], s[18:19], 1 // (op 526) 

v_mov_b32 v73, s38 // (op 526) 
v_mov_b32 v78, s39 // (op 526) 
v_add_co_u32 v80, s[18:19], v76, v73 // (op 526) least significant half
v_addc_co_u32 v81, s[18:19], v77, v78, s[18:19] // (op 526) most significant half




s_mov_b32 m0, s0 // (op 526) reg expression
ds_read_u8 v12, v6 offset:11268 // (op 105) Load local data 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v74 // (op 526) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[60:67], v[54:59], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 107) 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v74 // (op 526) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v74 // (op 526) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 526) Load value direct to lds



v_mov_b64 v[74:75], s[20:21] // (op 17) convert

v_mov_b32 v73, v21 // (op 17) 
s_lshl_b32 s15, s20, 4 // (op 17) 
v_add_u32 v78, v73, s15 // (op 17) 

buffer_load_ubyte v40, v78, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v78, v78, v74 // (op 17) 
buffer_load_ubyte v41, v78, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1306) 



s_waitcnt vmcnt(23)
ds_write_b128 v20, v[42:45] offset:11520 // (op 306) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v20, v[46:47] offset:11536 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v20, v[48:51] offset:11544 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v20, v[52:53] offset:11560 // (op 306) Store local data 
s_barrier  // (op 1318) 



s_waitcnt vmcnt(19)
ds_write_b8 v72, v36 offset:14592 // (op 308) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v72, v37 offset:14600 // (op 308) Store local data 
s_barrier  // (op 1312) 



s_waitcnt vmcnt(13)
ds_write_b8 v71, v34 offset:14720 // (op 307) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v71, v35 offset:14736 // (op 307) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1302) 



ds_read_b128 v[78:81], v8 offset:11520 // (op 291) Load local data 
ds_read_b64 v[82:83], v8 offset:11536 // (op 291) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1315) 



ds_read_u8 v74, v10 offset:14592 // (op 295) Load local data 
s_barrier  // (op 1296) 



ds_read_b64_tr_b8 v[84:85], v3 offset:4096 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v3 offset:4224 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v3 offset:5120 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v3 offset:5248 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1309) 


ds_read_u8 v75, v9 offset:14720 // (op 293) Load local data 



buffer_load_dwordx4 v[42:45], v16, s[44:47], 0 offen offset:576 // (op 309) Load value
buffer_load_dwordx2 v[46:47], v16, s[44:47], 0 offen offset:592 // (op 309) Load value
buffer_load_dwordx4 v[48:51], v16, s[44:47], 0 offen offset:600 // (op 309) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:91], v[78:83], a[0:3], v75, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 296) 
buffer_load_dwordx2 v[52:53], v16, s[44:47], 0 offen offset:616 // (op 309) Load value









v_add_u32 v15, 384, v15 // (op 661) 
v_mov_b64 v[74:75], s[26:27] // (op 311) convert

v_mov_b32 v16, v70 // (op 311) 
buffer_load_ubyte v36, v16, s[52:55], 0 offen offset:24 // (op 311) Load value
v_add_u32 v16, v16, v74 // (op 311) 
buffer_load_ubyte v37, v16, s[52:55], 0 offen offset:24 // (op 311) Load value


ds_read_b64_tr_b8 v[78:79], v3 offset:6144 // (op 298) Transpose load local data 

ds_read_b64_tr_b8 v[80:81], v3 offset:6272 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v3 offset:7168 // (op 298) Transpose load local data 
v_add_u32 v13, 16, v13 // (op 789) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1299) 
ds_read_b64_tr_b8 v[84:85], v3 offset:7296 // (op 298) Transpose load local data 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 537) 



v_mov_b64 v[74:75], s[18:19] // (op 537) convert


s_lshl_b64 s[18:19], s[2:3], 8 // (op 537) 
s_mul_i32 s39, 3, s19 // (op 537) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 537) least significant
s_mul_hi_u32 s15, 3, s18 // (op 537) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 537) most significant: sum


v_mov_b32 v16, s38 // (op 537) 
v_mov_b32 v70, s39 // (op 537) 
ds_read_b128 v[86:89], v8 offset:11616 // (op 300) Load local data 
ds_read_b64 v[90:91], v8 offset:11632 // (op 300) Load local data 
v_add_co_u32 v92, s[18:19], v76, v16 // (op 537) least significant half
v_addc_co_u32 v93, s[18:19], v77, v70, s[18:19] // (op 537) most significant half






s_barrier  // (op 1377) 
s_add_u32 s15, 4096, s1 // (op 537) 
s_mov_b32 m0, s15 // (op 537) reg expression
ds_read_u8 v16, v9 offset:14784 // (op 302) Load local data 
buffer_load_dwordx4 v92, s[40:43], 0 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v74 // (op 537) least significant half



s_barrier  // (op 1381) 
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
buffer_load_dwordx4 v92, s[40:43], 0 offen offset: 0 lds // (op 537) Load value direct to lds
ds_read_u8 v70, v10 offset:14596 // (op 304) Load local data 
v_add_co_u32 v92, s[18:19], v92, v74 // (op 537) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 537) most significant half

s_barrier  // (op 1385) 
s_add_u32 m0, m0, 1024 // (op 537) 
buffer_load_dwordx4 v92, s[40:43], 0 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v92, s[18:19], v92, v74 // (op 537) least significant half
v_addc_co_u32 v93, s[18:19], 0, v93, s[18:19] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[86:91], a[0:3], v16, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 305) 
buffer_load_dwordx4 v92, s[40:43], 0 offen offset: 0 lds // (op 537) Load value direct to lds









v_mov_b64 v[74:75], s[20:21] // (op 310) convert

s_lshl_b32 s15, s20, 3 // (op 310) 
s_mul_i32 s18, 3, s15 // (op 310) 

v_add_u32 v16, v73, s18 // (op 310) 

buffer_load_ubyte v34, v16, s[48:51], 0 offen offset: 0 // (op 310) Load value
v_add_u32 v16, v16, v74 // (op 310) 
s_lshl_b64 s[18:19], s[2:3], 8 // (op 597) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 597) 

v_mov_b32 v70, s38 // (op 597) 
v_mov_b32 v76, s39 // (op 597) 
buffer_load_ubyte v35, v16, s[48:51], 0 offen offset: 0 // (op 310) Load value


v_add_co_u32 v18, s[18:19], v18, v70 // (op 597) least significant half

v_addc_co_u32 v19, s[18:19], v19, v76, s[18:19] // (op 597) most significant half




s_barrier  // (op 1238) 
s_lshl_b32 s15, s20, 4 // (op 725) 



v_add_u32 v21, v21, s15 // (op 725) 

s_waitcnt vmcnt(23)
ds_write_b128 v17, v[22:25] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v17, v[26:27] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v17, v[28:31] offset:8216 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v17, v[32:33] offset:8232 // (op 13) Store local data 
s_barrier  // (op 1279) 



s_waitcnt vmcnt(19)
ds_write_b8 v69, v38 offset:11264 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v69, v39 offset:11272 // (op 30) Store local data 
s_barrier  // (op 1259) 



s_waitcnt vmcnt(13)
ds_write_b8 v68, v40 offset:11392 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v68, v41 offset:11408 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1248) 



ds_read_b128 v[54:57], v4 offset:8192 // (op 77) Load local data 
ds_read_b64 v[58:59], v4 offset:8208 // (op 77) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1289) 



ds_read_u8 v12, v6 offset:11264 // (op 87) Load local data 
s_barrier  // (op 1228) 



ds_read_b64_tr_b8 v[60:61], v0 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:128 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[64:65], v0 offset:1024 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[66:67], v0 offset:1152 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1269) 


ds_read_u8 v14, v5 offset:11392 // (op 82) Load local data 
s_ashr_i64 s[18:19], s[6:7], 8 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60:
















s_barrier  // (op 1333) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 560) 
s_lshr_b32 s14, s0, 1 // (op 560) 

v_lshl_add_u32 v13, s14, 9, v1 // (op 560) 

s_lshl_b32 s0, s16, 4 // (op 560) 
v_mul_lo_u32 v33, v13, s3 // (op 560) most significant: low of low * high
v_mul_hi_u32 v15, v13, s2 // (op 560) most significant: high of low * low
v_mul_lo_u32 v32, v13, s2 // (op 560) least significant: low of low * low
v_add_u32 v33, v33, v15 // (op 560) most significant: sum


s_mov_b32 s15, 0 // (op 560) convert
s_mov_b32 s14, s0 // (op 560) convert

v_mov_b32 v13, s14 // (op 560) 
v_mov_b32 v15, s15 // (op 560) 
v_add_co_u32 v16, s[0:1], v13, v32 // (op 560) least significant half
v_addc_co_u32 v17, s[0:1], v15, v33, s[0:1] // (op 560) most significant half





s_mul_i32 s1, s2, s7 // (op 562) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 562) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 562) least significant
s_mul_hi_u32 s15, s2, s6 // (op 562) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 562) most significant: sum
s_add_u32 s1, s1, s15 // (op 562) most significant: sum


s_mov_b32 s40, s4 // (op 562) reg expression
s_mov_b32 s41, s5 // (op 562) reg expression
s_mov_b32 s42, s0 // (op 562) reg expression
s_mov_b32 s43, 131072 // (op 562) call()

v_and_b32 v13, -4, v1 // (op 627) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 627) 
v_sub_u32 v15, v1, v13 // (op 627) 

s_lshr_b32 s14, s0, 1 // (op 627) 

v_lshlrev_b32 v13, 6, v15 // (op 627) 

v_lshl_add_u32 v15, s14, 9, v13 // (op 627) 


v_mov_b32 v19, 0 // (op 627) convert
v_mov_b32 v18, v15 // (op 627) convert

v_lshrrev_b32 v13, 2, v1 // (op 627) 
v_lshl_add_u32 v15, s17, 4, v13 // (op 627) 

v_mul_lo_u32 v21, v15, s9 // (op 627) most significant: low of low * high
v_mul_hi_u32 v13, v15, s8 // (op 627) most significant: high of low * low
v_mul_lo_u32 v20, v15, s8 // (op 627) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 627) most significant: sum


v_add_co_u32 v22, s[0:1], v18, v20 // (op 627) least significant half
v_addc_co_u32 v23, s[0:1], v19, v21, s[0:1] // (op 627) most significant half



v_mul_lo_u32 v19, v23, 6 // (op 627) most significant: low of high * low
v_mul_hi_u32 v13, v22, 6 // (op 627) most significant: high of low * low
v_mul_lo_u32 v18, v22, 6 // (op 627) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 627) most significant: sum


v_lshrrev_b64 v[20:21], 3, v[18:19] // (op 627) 

v_mov_b32 v13, v20 // (op 627) convert

s_mul_i32 s1, s13, 6 // (op 629) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 629) least significant
s_mul_hi_u32 s14, s12, 6 // (op 629) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 629) most significant: sum

s_mul_i32 s15, s8, s1 // (op 629) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 629) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 629) least significant
s_mul_hi_u32 s19, s8, s0 // (op 629) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 629) most significant: sum
s_add_u32 s15, s15, s19 // (op 629) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 629) 

s_mov_b32 s44, s10 // (op 629) reg expression
s_mov_b32 s45, s11 // (op 629) reg expression
s_mov_b32 s46, s0 // (op 629) reg expression
s_mov_b32 s47, 131072 // (op 629) call()

v_and_b32 v15, -16, v1 // (op 691) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 691) 
v_sub_u32 v18, v1, v15 // (op 691) 

s_lshr_b32 s14, s0, 1 // (op 691) 

v_lshl_add_u32 v15, s16, 4, v18 // (op 691) 

v_lshrrev_b32 v18, 4, v1 // (op 691) 
v_lshlrev_b32 v19, 1, v18 // (op 691) 

v_lshl_add_u32 v18, s14, 4, v19 // (op 691) 


v_mul_lo_u32 v19, v18, s20 // (op 691) 

v_add_u32 v18, v15, v19 // (op 691) 


s_mul_i32 s1, s20, s25 // (op 693) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 693) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 693) least significant
s_mul_hi_u32 s15, s20, s24 // (op 693) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 693) most significant: sum
s_add_u32 s1, s1, s15 // (op 693) most significant: sum


s_mov_b32 s48, s22 // (op 693) reg expression
s_mov_b32 s49, s23 // (op 693) reg expression
s_mov_b32 s50, s0 // (op 693) reg expression
s_mov_b32 s51, 131072 // (op 693) call()

v_and_b32 v15, -8, v1 // (op 755) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 755) 
v_sub_u32 v19, v1, v15 // (op 755) 

s_lshr_b32 s14, s0, 1 // (op 755) 

v_lshrrev_b32 v15, 3, v1 // (op 755) 
v_lshl_add_u32 v20, s14, 4, v19 // (op 755) 


v_lshlrev_b32 v19, 1, v15 // (op 755) 

v_lshl_add_u32 v15, s17, 4, v19 // (op 755) 

v_mul_lo_u32 v19, v15, s26 // (op 755) 

v_add_u32 v15, v20, v19 // (op 755) 


s_mul_i32 s1, s26, s13 // (op 757) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 757) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 757) least significant
s_mul_hi_u32 s15, s26, s12 // (op 757) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 757) most significant: sum
s_add_u32 s1, s1, s15 // (op 757) most significant: sum


s_mov_b32 s52, s28 // (op 757) reg expression
s_mov_b32 s53, s29 // (op 757) reg expression
s_mov_b32 s54, s0 // (op 757) reg expression
s_mov_b32 s55, 131072 // (op 757) call()

v_lshlrev_b32 v19, 4, v1 // (op 848) 
s_nop 0
v_readfirstlane_b32 s0, v19 // (op 848) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -4, v1 // (op 900) 
v_sub_u32 v20, v1, v19 // (op 900) 

v_lshlrev_b32 v19, 6, v20 // (op 900) 

v_lshrrev_b32 v20, 2, v1 // (op 900) 
v_lshl_add_u32 v21, v20, 8, v19 // (op 900) 


v_mul_lo_u32 v19, v21, 6 // (op 900) 

v_lshrrev_b32 v20, 3, v19 // (op 900) 

v_and_b32 v19, -16, v1 // (op 953) 
v_sub_u32 v21, v1, v19 // (op 953) 

v_lshrrev_b32 v19, 4, v1 // (op 953) 
v_lshl_add_u32 v22, v19, 5, v21 // (op 953) 


v_and_b32 v19, -8, v1 // (op 1006) 
v_sub_u32 v21, v1, v19 // (op 1006) 

v_lshrrev_b32 v19, 3, v1 // (op 1006) 
v_lshl_add_u32 v23, v19, 4, v21 // (op 1006) 


s_ashr_i64 s[14:15], s[6:7], 8 // (op 198) 
s_and_b32 s1, s14, 4294967294 // (op 198) 

s_ashr_i64 s[14:15], s[6:7], 8 // (op 197) 
s_mov_b32 s18, s14 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197:
s_barrier  // (op 1222) 
s_add_u32 s1, s1, 1 // (op 199) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 531) 
v_mov_b64 v[24:25], s[14:15] // (op 531) convert


v_mov_b64 v[26:27], v[16:17] // (op 531) 
v_mov_b64 v[28:29], v[26:27] // (op 531) reg expression
s_mov_b32 m0, s0 // (op 531) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v24 // (op 531) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v24 // (op 531) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v28, s[14:15], v28, v24 // (op 531) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds



s_lshl_b64 s[14:15], s[2:3], 8 // (op 550) 
v_mov_b32 v12, s14 // (op 550) 
v_mov_b32 v14, s15 // (op 550) 
v_add_co_u32 v16, s[18:19], v16, v12 // (op 550) least significant half
v_addc_co_u32 v17, s[18:19], v17, v14, s[18:19] // (op 550) most significant half






v_mov_b32 v12, v13 // (op 225) 

buffer_load_dwordx4 v[24:27], v12, s[44:47], 0 offen // (op 225) Load value
buffer_load_dwordx2 v[28:29], v12, s[44:47], 0 offen offset:16 // (op 225) Load value
buffer_load_dwordx4 v[30:33], v12, s[44:47], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx2 v[34:35], v12, s[44:47], 0 offen offset:40 // (op 225) Load value

v_add_u32 v13, 192, v13 // (op 619) 

s_barrier  // (op 1241) 
v_mov_b64 v[36:37], s[20:21] // (op 226) convert




v_mov_b32 v12, v18 // (op 226) 
v_mov_b32 v14, v12 // (op 226) reg expression
buffer_load_ubyte v38, v14, s[48:51], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v14, v14, v36 // (op 226) 
s_waitcnt vmcnt(4)
ds_write_b128 v20, v[24:27] offset:8192 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v20, v[28:29] offset:8208 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[30:33] offset:8216 // (op 222) Store local data 
buffer_load_ubyte v39, v14, s[48:51], 0 offen offset: 0 // (op 226) Load value



s_lshl_b32 s14, s20, 3 // (op 683) 
v_add_u32 v18, v18, s14 // (op 683) 

s_waitcnt vmcnt(2)
ds_write_b64 v20, v[34:35] offset:8232 // (op 222) Store local data 


v_mov_b64 v[24:25], s[26:27] // (op 227) convert

s_barrier  // (op 1262) 
v_mov_b32 v12, v15 // (op 227) 



v_mov_b32 v14, v12 // (op 227) 
buffer_load_ubyte v26, v14, s[52:55], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v14, v14, v24 // (op 227) 
buffer_load_ubyte v27, v14, s[52:55], 0 offen offset: 0 // (op 227) Load value



v_add_u32 v15, 8, v15 // (op 747) 
s_waitcnt vmcnt(3)
ds_write_b8 v22, v38 offset:11392 // (op 223) Store local data 
s_barrier  // (op 1231) 
s_waitcnt vmcnt(2)
ds_write_b8 v22, v39 offset:11408 // (op 223) Store local data 




ds_read_b64_tr_b8 v[28:29], v0 // (op 205) Transpose load local data 
s_barrier  // (op 1282) 
ds_read_b64_tr_b8 v[30:31], v0 offset:128 // (op 205) Transpose load local data 



ds_read_b64_tr_b8 v[32:33], v0 offset:1024 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:1152 // (op 205) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v26 offset:11264 // (op 224) Store local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1252) 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v27 offset:11272 // (op 224) Store local data 




ds_read_b128 v[36:39], v4 offset:8192 // (op 207) Load local data 
ds_read_b64 v[40:41], v4 offset:8208 // (op 207) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1272) 


ds_read_u8 v12, v5 offset:11392 // (op 209) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1292) 



ds_read_u8 v14, v6 offset:11264 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:41], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 212) 







ds_read_b64_tr_b8 v[24:25], v0 offset:2048 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:2176 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:3072 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:3200 // (op 214) Transpose load local data 



ds_read_b128 v[32:35], v4 offset:8288 // (op 216) Load local data 
ds_read_b64 v[36:37], v4 offset:8304 // (op 216) Load local data 


ds_read_u8 v12, v5 offset:11456 // (op 218) Load local data 



ds_read_u8 v14, v6 offset:11268 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:37], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b011 // (op 221) 




s_ashr_i64 s[14:15], s[6:7], 8 // (op 197) 
s_mov_b32 s18, s14 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 if true)
RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197:























s_barrier  // (op 1369) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

v_lshrrev_b32 v12, 16, v16 // (op 56) 
v_lshrrev_b32 v13, 16, v17 // (op 56) 
v_lshrrev_b32 v14, 16, v18 // (op 56) 
v_lshrrev_b32 v15, 16, v19 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 1048) 
v_and_b32 v3, 63, v1 // (op 1048) 
v_lshrrev_b32 v4, 4, v3 // (op 1048) 
v_lshlrev_b32 v5, 2, v4 // (op 1048) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1048) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1048) 

v_lshrrev_b32 v4, 2, v3 // (op 1048) 

v_and_b32 v3, 3, v4 // (op 1048) 

v_and_b32 v4, 3, v1 // (op 1048) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1048) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1048) 

v_mul_lo_u32 v3, v6, s30 // (op 1048) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 1048) 


s_mul_i32 s15, s30, s13 // (op 1050) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1050) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1050) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1050) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1050) most significant: sum
s_add_u32 s15, s15, s1 // (op 1050) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 1050) 

s_mov_b32 s8, s32 // (op 1050) reg expression
s_mov_b32 s9, s33 // (op 1050) reg expression
s_mov_b32 s10, s0 // (op 1050) reg expression
s_mov_b32 s11, 131072 // (op 1050) call()


v_pack_b32_f16 v8, v12, v13 // (op 58) 
v_pack_b32_f16 v9, v14, v15 // (op 58) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157:
v_lshrrev_b32 v0, 6, v1 // (op 818) 
v_and_b32 v3, 63, v1 // (op 818) 
v_lshrrev_b32 v4, 4, v3 // (op 818) 
v_lshlrev_b32 v5, 2, v4 // (op 818) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 818) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 818) 

v_lshrrev_b32 v4, 2, v3 // (op 818) 

v_and_b32 v3, 3, v4 // (op 818) 

v_and_b32 v4, 3, v1 // (op 818) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 818) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 818) 

v_mul_lo_u32 v3, v6, s34 // (op 818) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 818) 


s_mul_i32 s15, s34, s13 // (op 820) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 820) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 820) least significant
s_mul_hi_u32 s1, s34, s12 // (op 820) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 820) most significant: sum
s_add_u32 s15, s15, s1 // (op 820) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 820) 

s_mov_b32 s4, s36 // (op 820) reg expression
s_mov_b32 s5, s37 // (op 820) reg expression
s_mov_b32 s6, s0 // (op 820) reg expression
s_mov_b32 s7, 131072 // (op 820) call()





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
v_and_b32 v12, 65535, v8 // (op 171) 
v_lshrrev_b32 v13, 16, v8 // (op 171) 
v_lshlrev_b32 v12, 16, v12 // (op 171) 
v_lshlrev_b32 v13, 16, v13 // (op 171) 
v_mul_f32 v20, v7, v12 // (op 171) 
v_mul_f32 v21, v7, v13 // (op 171) 
v_and_b32 v12, 65535, v9 // (op 171) 
v_lshrrev_b32 v13, 16, v9 // (op 171) 
v_lshlrev_b32 v12, 16, v12 // (op 171) 
v_lshlrev_b32 v13, 16, v13 // (op 171) 
v_mul_f32 v22, v7, v12 // (op 171) 
v_mul_f32 v23, v7, v13 // (op 171) 


v_add_f32 v12, v20, v16 // (op 167) 
v_add_f32 v13, v21, v17 // (op 167) 
v_add_f32 v14, v22, v18 // (op 167) 
v_add_f32 v15, v23, v19 // (op 167) 


v_lshrrev_b32 v16, 16, v12 // (op 168) 
v_lshrrev_b32 v17, 16, v13 // (op 168) 
v_lshrrev_b32 v18, 16, v14 // (op 168) 
v_lshrrev_b32 v19, 16, v15 // (op 168) 

v_lshrrev_b32 v0, 6, v1 // (op 1065) 
v_and_b32 v3, 63, v1 // (op 1065) 
v_lshrrev_b32 v4, 4, v3 // (op 1065) 
v_lshlrev_b32 v5, 2, v4 // (op 1065) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1065) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1065) 

v_lshrrev_b32 v4, 2, v3 // (op 1065) 

v_and_b32 v3, 3, v4 // (op 1065) 

v_and_b32 v4, 3, v1 // (op 1065) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1065) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1065) 

v_mul_lo_u32 v3, v6, s30 // (op 1065) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 1065) 


s_mul_i32 s15, s30, s13 // (op 1067) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1067) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1067) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1067) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1067) most significant: sum
s_add_u32 s15, s15, s1 // (op 1067) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 1067) 

s_mov_b32 s8, s32 // (op 1067) reg expression
s_mov_b32 s9, s33 // (op 1067) reg expression
s_mov_b32 s10, s0 // (op 1067) reg expression
s_mov_b32 s11, 131072 // (op 1067) call()


v_pack_b32_f16 v8, v16, v17 // (op 169) 
v_pack_b32_f16 v9, v18, v19 // (op 169) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 169) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157:




s_endpgm  // End of RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 14848 // lds bytes
  .amdhsa_accum_offset 96
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
    DestDataType: b
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
  - .name: RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NN_BF8_BF6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 14848
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 94
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

