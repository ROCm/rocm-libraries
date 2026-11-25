// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_and_b32 v3, 63, v1 // (op 928) 
v_and_b32 v4, 48, v1 // (op 928) 
v_sub_u32 v5, v3, v4 // (op 928) 

v_lshrrev_b32 v4, 4, v3 // (op 928) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 928) 


v_lshlrev_b32 v4, 4, v3 // (op 928) 
v_mul_lo_u32 v5, v4, 6 // (op 928) 

v_lshrrev_b32 v4, 3, v5 // (op 928) 

v_and_b32 v5, 268435455, v3 // (op 928) 

v_lshlrev_b32 v3, 2, v5 // (op 928) 

v_add_u32 v5, v4, v3 // (op 928) 


v_lshrrev_b32 v3, 6, v1 // (op 981) 
v_and_b32 v4, 15, v1 // (op 981) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 981) 


v_and_b32 v3, 48, v1 // (op 981) 
v_add_u32 v4, v3, v6 // (op 981) 


v_and_b32 v3, 48, v1 // (op 1034) 
v_and_b32 v6, 15, v1 // (op 1034) 
v_add_u32 v7, v3, v6 // (op 1034) 


v_and_b32 v3, 48, v1 // (op 1107) 
v_and_b32 v6, 63, v1 // (op 1107) 
v_sub_u32 v8, v6, v3 // (op 1107) 

v_lshrrev_b32 v6, 1, v8 // (op 1107) 
v_add_lshl_u32 v9, v3, v6, 4 // (op 1107) 


v_and_b32 v3, 1, v8 // (op 1107) 

v_lshlrev_b32 v6, 3, v3 // (op 1107) 

v_lshrrev_b32 v3, 6, v1 // (op 1107) 
v_lshl_add_u32 v8, v3, 4, v6 // (op 1107) 


v_add_u32 v3, v9, v8 // (op 1107) 


v_and_b32 v6, 63, v1 // (op 1131) 
v_and_b32 v8, 48, v1 // (op 1131) 
v_sub_u32 v9, v6, v8 // (op 1131) 

v_lshrrev_b32 v8, 4, v6 // (op 1131) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 1131) 


v_lshlrev_b32 v8, 4, v6 // (op 1131) 
v_mul_lo_u32 v9, v8, 6 // (op 1131) 

v_lshrrev_b32 v8, 3, v9 // (op 1131) 

v_and_b32 v9, 268435455, v6 // (op 1131) 

v_lshlrev_b32 v6, 2, v9 // (op 1131) 

v_add_u32 v9, v8, v6 // (op 1131) 


v_lshrrev_b32 v6, 6, v1 // (op 1155) 
v_and_b32 v8, 15, v1 // (op 1155) 
v_lshl_add_u32 v10, v6, 4, v8 // (op 1155) 


v_and_b32 v6, 48, v1 // (op 1155) 
v_add_u32 v8, v6, v10 // (op 1155) 


v_and_b32 v6, 48, v1 // (op 1179) 
v_and_b32 v10, 15, v1 // (op 1179) 
v_add_u32 v11, v6, v10 // (op 1179) 


v_mov_b32 v6, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


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

s_lshl_b32 s0, s17, 4 // (op 650) 
v_lshlrev_b32 v12, 2, v1 // (op 650) 
s_mov_b32 s15, 0 // (op 650) convert
s_mov_b32 s14, s0 // (op 650) convert

v_mul_lo_u32 v15, v12, s9 // (op 650) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 650) most significant: high of low * low
v_mul_lo_u32 v14, v12, s8 // (op 650) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 650) most significant: sum

v_mov_b32 v13, s14 // (op 650) 
v_mov_b32 v20, s15 // (op 650) 
v_add_co_u32 v14, s[0:1], v14, v13 // (op 650) least significant half
v_addc_co_u32 v15, s[0:1], v15, v20, s[0:1] // (op 650) most significant half





v_mul_lo_u32 v13, v15, 6 // (op 650) most significant: low of high * low
v_mul_hi_u32 v20, v14, 6 // (op 650) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 650) least significant: low of low * low
v_add_u32 v13, v13, v20 // (op 650) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 650) 

v_mov_b32 v12, v14 // (op 650) convert

s_mul_i32 s1, s7, 6 // (op 652) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 652) least significant
s_mul_hi_u32 s14, s6, 6 // (op 652) most significant: high of low * low
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

s_lshl_b32 s0, s17, 4 // (op 669) 
v_lshlrev_b32 v13, 2, v1 // (op 669) 
s_mov_b32 s15, 0 // (op 669) convert
s_mov_b32 s14, s0 // (op 669) convert

v_mul_lo_u32 v15, v13, s9 // (op 669) most significant: low of low * high
v_mul_hi_u32 v20, v13, s8 // (op 669) most significant: high of low * low
v_mul_lo_u32 v14, v13, s8 // (op 669) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 669) most significant: sum

v_mov_b32 v20, s14 // (op 669) 
v_mov_b32 v21, s15 // (op 669) 
v_add_co_u32 v14, s[0:1], v14, v20 // (op 669) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 669) most significant half





v_mul_lo_u32 v21, v15, 6 // (op 669) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 669) most significant: high of low * low
v_mul_lo_u32 v20, v14, 6 // (op 669) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 669) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 669) 

v_mov_b32 v13, v14 // (op 669) convert

s_mul_i32 s1, s7, 6 // (op 671) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 671) least significant
s_mul_hi_u32 s14, s6, 6 // (op 671) most significant: high of low * low
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

v_and_b32 v14, -16, v1 // (op 714) 
v_sub_u32 v15, v1, v14 // (op 714) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 714) 

v_lshrrev_b32 v15, 4, v1 // (op 714) 
v_lshlrev_b32 v20, 1, v15 // (op 714) 

v_mul_lo_u32 v15, v20, s12 // (op 714) 

v_add_u32 v20, v14, v15 // (op 714) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 716) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 716) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 716) least significant
s_mul_hi_u32 s15, s12, s22 // (op 716) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 716) most significant: sum
s_add_u32 s1, s1, s15 // (op 716) most significant: sum


s_mov_b32 s48, s20 // (op 716) reg expression
s_mov_b32 s49, s21 // (op 716) reg expression
s_mov_b32 s50, s0 // (op 716) reg expression
s_mov_b32 s51, 131072 // (op 716) call()

v_and_b32 v14, -16, v1 // (op 733) 
v_sub_u32 v15, v1, v14 // (op 733) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 733) 

v_lshrrev_b32 v15, 4, v1 // (op 733) 
v_lshlrev_b32 v21, 1, v15 // (op 733) 

v_mul_lo_u32 v15, v21, s12 // (op 733) 

v_add_u32 v21, v14, v15 // (op 733) 


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

v_and_b32 v14, -16, v1 // (op 778) 
v_sub_u32 v15, v1, v14 // (op 778) 

v_lshl_add_u32 v14, s17, 4, v15 // (op 778) 

v_lshrrev_b32 v15, 4, v1 // (op 778) 
v_lshlrev_b32 v22, 1, v15 // (op 778) 

v_mul_lo_u32 v15, v22, s24 // (op 778) 
v_add_u32 v15, v15, v14 // (op 778) 


s_mul_i32 s1, s24, s23 // (op 780) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 780) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 780) least significant
s_mul_hi_u32 s15, s24, s22 // (op 780) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 780) most significant: sum
s_add_u32 s1, s1, s15 // (op 780) most significant: sum


s_mov_b32 s52, s26 // (op 780) reg expression
s_mov_b32 s53, s27 // (op 780) reg expression
s_mov_b32 s54, s0 // (op 780) reg expression
s_mov_b32 s55, 131072 // (op 780) call()

v_and_b32 v14, -16, v1 // (op 797) 
v_sub_u32 v22, v1, v14 // (op 797) 

v_lshl_add_u32 v14, s17, 4, v22 // (op 797) 

v_lshrrev_b32 v22, 4, v1 // (op 797) 
v_lshlrev_b32 v23, 1, v22 // (op 797) 

v_mul_lo_u32 v22, v23, s24 // (op 797) 
v_add_u32 v22, v22, v14 // (op 797) 


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


s_mul_i32 s1, s9, 6 // (op 428) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 428) least significant
s_mul_hi_u32 s14, s8, 6 // (op 428) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 428) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 428) 


v_mov_b32 v14, v12 // (op 428) reg expression
buffer_load_dwordx3 v[26:28], v14, s[44:47], 0 offen // (op 428) Load value
v_add_u32 v14, v14, v24 // (op 428) 
buffer_load_dwordx3 v[30:32], v14, s[44:47], 0 offen // (op 428) Load value
v_add_u32 v14, v14, v24 // (op 428) 
buffer_load_dwordx3 v[34:36], v14, s[44:47], 0 offen // (op 428) Load value
v_add_u32 v14, v14, v24 // (op 428) 
buffer_load_dwordx3 v[38:40], v14, s[44:47], 0 offen // (op 428) Load value



v_mov_b64 v[24:25], s[24:25] // (op 431) convert

v_mov_b32 v14, v15 // (op 431) reg expression
buffer_load_ubyte v42, v14, s[52:55], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v14, v14, v24 // (op 431) 
buffer_load_ubyte v43, v14, s[52:55], 0 offen offset: 0 // (op 431) Load value


v_lshlrev_b32 v14, 4, v1 // (op 863) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 863) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 542) 
v_mov_b64 v[24:25], s[14:15] // (op 542) convert


v_mov_b64 v[44:45], v[16:17] // (op 542) reg expression
s_mov_b32 m0, s0 // (op 542) reg expression
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 542) Load value direct to lds
v_add_co_u32 v44, s[14:15], v44, v24 // (op 542) least significant half
v_addc_co_u32 v45, s[14:15], 0, v45, s[14:15] // (op 542) most significant half

s_add_u32 m0, m0, 1024 // (op 542) 
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 542) Load value direct to lds
v_add_co_u32 v44, s[14:15], v44, v24 // (op 542) least significant half
v_addc_co_u32 v45, s[14:15], 0, v45, s[14:15] // (op 542) most significant half

s_add_u32 m0, m0, 1024 // (op 542) 
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 542) Load value direct to lds
v_add_co_u32 v44, s[14:15], v44, v24 // (op 542) least significant half
v_addc_co_u32 v45, s[14:15], 0, v45, s[14:15] // (op 542) most significant half

s_add_u32 m0, m0, 1024 // (op 542) 
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 542) Load value direct to lds




v_mov_b64 v[24:25], s[12:13] // (op 437) convert

v_mov_b32 v14, v20 // (op 437) reg expression
buffer_load_ubyte v44, v14, s[48:51], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v14, v14, v24 // (op 437) 
buffer_load_ubyte v45, v14, s[48:51], 0 offen offset: 0 // (op 437) Load value



s_mul_i32 s1, s9, 6 // (op 440) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 440) least significant
s_mul_hi_u32 s14, s8, 6 // (op 440) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 440) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 440) 


s_lshl_b64 s[0:1], s[8:9], 8 // (op 440) 
s_mul_i32 s15, s1, 6 // (op 440) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 440) least significant
s_mul_hi_u32 s18, s0, 6 // (op 440) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 440) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 440) 

v_add_u32 v14, v12, s0 // (op 440) 

buffer_load_dwordx3 v[46:48], v14, s[44:47], 0 offen // (op 440) Load value
v_add_u32 v14, v14, v24 // (op 440) 
buffer_load_dwordx3 v[50:52], v14, s[44:47], 0 offen // (op 440) Load value
v_add_u32 v14, v14, v24 // (op 440) 
buffer_load_dwordx3 v[54:56], v14, s[44:47], 0 offen // (op 440) Load value
v_add_u32 v14, v14, v24 // (op 440) 
buffer_load_dwordx3 v[58:60], v14, s[44:47], 0 offen // (op 440) Load value




v_mov_b64 v[24:25], s[24:25] // (op 443) convert

s_lshl_b32 s0, s24, 3 // (op 443) 
v_add_u32 v12, v15, s0 // (op 443) 

buffer_load_ubyte v62, v12, s[52:55], 0 offen offset: 0 // (op 443) Load value
v_add_u32 v12, v12, v24 // (op 443) 
buffer_load_ubyte v63, v12, s[52:55], 0 offen offset: 0 // (op 443) Load value



v_lshlrev_b32 v12, 4, v1 // (op 1095) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1095) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 547) 
v_mov_b64 v[14:15], s[14:15] // (op 547) convert


s_lshl_b64 s[14:15], s[2:3], 8 // (op 547) 
v_mov_b32 v12, s14 // (op 547) 
v_mov_b32 v29, s15 // (op 547) 
v_add_co_u32 v24, s[18:19], v16, v12 // (op 547) least significant half
v_addc_co_u32 v25, s[18:19], v17, v29, s[18:19] // (op 547) most significant half




s_add_u32 s1, 4096, s0 // (op 547) 
s_mov_b32 m0, s1 // (op 547) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 547) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 547) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 547) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 547) Load value direct to lds






v_mov_b64 v[24:25], s[12:13] // (op 449) convert

s_lshl_b32 s0, s12, 3 // (op 449) 
v_add_u32 v12, v20, s0 // (op 449) 

buffer_load_ubyte v14, v12, s[48:51], 0 offen offset: 0 // (op 449) Load value
v_add_u32 v12, v12, v24 // (op 449) 
buffer_load_ubyte v15, v12, s[48:51], 0 offen offset: 0 // (op 449) Load value



v_lshlrev_b32 v12, 6, v1 // (op 916) 
v_mul_lo_u32 v20, v12, 6 // (op 916) 
v_lshrrev_b32 v29, 3, v20 // (op 916) 

v_lshrrev_b32 v20, 4, v12 // (op 916) 

v_lshlrev_b32 v12, 2, v20 // (op 916) 

v_add_u32 v20, v29, v12 // (op 916) 





s_waitcnt vmcnt(23)
ds_write_b96 v20, v[26:28] offset:8192 // (op 451) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v20, v[30:32] offset:8208 // (op 451) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v20, v[34:36] offset:8224 // (op 451) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v20, v[38:40] offset:8240 // (op 451) Store local data 

v_and_b32 v12, -16, v1 // (op 1022) 
v_sub_u32 v20, v1, v12 // (op 1022) 

v_lshrrev_b32 v12, 4, v1 // (op 1022) 
v_lshl_add_u32 v29, v12, 5, v20 // (op 1022) 





s_waitcnt vmcnt(19)
ds_write_b8 v29, v42 offset:12288 // (op 452) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v29, v43 offset:12304 // (op 452) Store local data 

v_and_b32 v12, -16, v1 // (op 969) 
v_sub_u32 v20, v1, v12 // (op 969) 

v_lshrrev_b32 v12, 4, v1 // (op 969) 
v_lshl_add_u32 v29, v12, 5, v20 // (op 969) 





s_waitcnt vmcnt(13)
ds_write_b8 v29, v44 offset:12416 // (op 454) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v29, v45 offset:12432 // (op 454) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1235) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 471) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 471) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1276) 


ds_read_u8 v12, v7 offset:12288 // (op 477) Load local data 
s_barrier  // (op 1216) 



ds_read_b64_tr_b8 v[70:71], v0 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:128 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:1024 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:1152 // (op 468) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1256) 


ds_read_u8 v20, v4 offset:12416 // (op 474) Load local data 
v_lshlrev_b32 v29, 4, v1 // (op 835) 
s_nop 0
v_readfirstlane_b32 s0, v29 // (op 835) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 6, v1 // (op 887) 
v_mul_lo_u32 v33, v29, 6 // (op 887) 
v_lshrrev_b32 v37, 3, v33 // (op 887) 

v_lshrrev_b32 v33, 4, v29 // (op 887) 

v_lshlrev_b32 v29, 2, v33 // (op 887) 

v_add_u32 v33, v37, v29 // (op 887) 


v_and_b32 v29, -16, v1 // (op 940) 
v_sub_u32 v37, v1, v29 // (op 940) 

v_lshrrev_b32 v29, 4, v1 // (op 940) 
v_lshl_add_u32 v41, v29, 5, v37 // (op 940) 


v_and_b32 v29, -16, v1 // (op 993) 
v_sub_u32 v37, v1, v29 // (op 993) 

v_lshrrev_b32 v29, 4, v1 // (op 993) 
v_lshl_add_u32 v49, v29, 5, v37 // (op 993) 


v_lshlrev_b32 v29, 4, v1 // (op 1080) 
s_nop 0
v_readfirstlane_b32 s1, v29 // (op 1080) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 6, v1 // (op 1119) 
v_mul_lo_u32 v37, v29, 6 // (op 1119) 
v_lshrrev_b32 v53, 3, v37 // (op 1119) 

v_lshrrev_b32 v37, 4, v29 // (op 1119) 

v_lshlrev_b32 v29, 2, v37 // (op 1119) 

v_add_u32 v37, v53, v29 // (op 1119) 


v_and_b32 v29, -16, v1 // (op 1143) 
v_sub_u32 v53, v1, v29 // (op 1143) 

v_lshrrev_b32 v29, 4, v1 // (op 1143) 
v_lshl_add_u32 v57, v29, 5, v53 // (op 1143) 


v_and_b32 v29, -16, v1 // (op 1167) 
v_sub_u32 v53, v1, v29 // (op 1167) 

v_lshrrev_b32 v29, 4, v1 // (op 1167) 
v_lshl_add_u32 v61, v29, 5, v53 // (op 1167) 


s_mov_b32 s14, 0 // (op 61) call()
s_ashr_i64 s[18:19], s[6:7], 8 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60:

s_add_u32 s14, s14, 2 // (op 62) 
s_mul_i32 s19, s9, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 10) least significant
s_mul_hi_u32 s15, s8, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 10) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[18:19] // (op 10) 


v_mov_b32 v29, v13 // (op 10) 
s_lshl_b64 s[18:19], s[8:9], 8 // (op 10) 
s_mul_i32 s39, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 10) least significant
s_mul_hi_u32 s15, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 10) 

s_lshl_b32 s15, s18, 1 // (op 10) 

v_add_u32 v53, v29, s15 // (op 10) 

buffer_load_dwordx3 v[26:28], v53, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v53, v53, v16 // (op 10) 
buffer_load_dwordx3 v[30:32], v53, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v53, v53, v16 // (op 10) 
buffer_load_dwordx3 v[34:36], v53, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v53, v53, v16 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 90) 
buffer_load_dwordx3 v[38:40], v53, s[44:47], 0 offen // (op 10) Load value



s_barrier  // (op 1225) 
v_mov_b64 v[16:17], s[24:25] // (op 25) convert




ds_read_b64_tr_b8 v[70:71], v0 offset:2048 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:2176 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:3072 // (op 93) Transpose load local data 
v_mov_b32 v53, v22 // (op 25) 
s_lshl_b32 s15, s24, 4 // (op 25) 
v_add_u32 v23, v53, s15 // (op 25) 

buffer_load_ubyte v42, v23, s[52:55], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v23, v23, v16 // (op 25) 
buffer_load_ubyte v43, v23, s[52:55], 0 offen offset: 0 // (op 25) Load value


ds_read_b64_tr_b8 v[76:77], v0 offset:3200 // (op 93) Transpose load local data 
s_barrier  // (op 1245) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1219) 




ds_read_b96_tr_b6 v[64:66], v5 offset:10240 // (op 97) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:10496 // (op 97) Transpose load local data 
s_lshl_b64 s[18:19], s[2:3], 6 // (op 526) 
s_barrier  // (op 1266) 
v_mov_b64 v[16:17], s[18:19] // (op 526) convert




ds_read_u8 v20, v4 offset:12480 // (op 101) Load local data 
v_mov_b64 v[24:25], v[18:19] // (op 526) 
s_barrier  // (op 1286) 
s_lshl_b64 s[18:19], s[2:3], 8 // (op 526) 


s_lshl_b64 s[38:39], s[18:19], 1 // (op 526) 

v_mov_b32 v23, s38 // (op 526) 
v_mov_b32 v78, s39 // (op 526) 
v_add_co_u32 v80, s[18:19], v24, v23 // (op 526) least significant half
v_addc_co_u32 v81, s[18:19], v25, v78, s[18:19] // (op 526) most significant half




s_mov_b32 m0, s0 // (op 526) reg expression
ds_read_u8 v12, v7 offset:12352 // (op 105) Load local data 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v16 // (op 526) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 107) 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v16 // (op 526) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v16 // (op 526) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 526) Load value direct to lds



v_mov_b64 v[16:17], s[12:13] // (op 17) convert

v_mov_b32 v23, v21 // (op 17) 
s_lshl_b32 s15, s12, 4 // (op 17) 
v_add_u32 v78, v23, s15 // (op 17) 

buffer_load_ubyte v44, v78, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v78, v78, v16 // (op 17) 
buffer_load_ubyte v45, v78, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1306) 



s_waitcnt vmcnt(23)
ds_write_b96 v37, v[46:48] offset:12544 // (op 306) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v37, v[50:52] offset:12560 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v37, v[54:56] offset:12576 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v37, v[58:60] offset:12592 // (op 306) Store local data 
s_barrier  // (op 1318) 



s_waitcnt vmcnt(19)
ds_write_b8 v61, v62 offset:16640 // (op 308) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v61, v63 offset:16656 // (op 308) Store local data 
s_barrier  // (op 1312) 



s_waitcnt vmcnt(13)
ds_write_b8 v57, v14 offset:16768 // (op 307) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v57, v15 offset:16784 // (op 307) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1302) 



ds_read_b96_tr_b6 v[78:80], v9 offset:12544 // (op 291) Transpose load local data 
ds_read_b96_tr_b6 v[81:83], v9 offset:12800 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1315) 


ds_read_u8 v16, v11 offset:16640 // (op 295) Load local data 
s_barrier  // (op 1296) 



ds_read_b64_tr_b8 v[84:85], v3 offset:4096 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v3 offset:4224 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v3 offset:5120 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v3 offset:5248 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1309) 


ds_read_u8 v17, v8 offset:16768 // (op 293) Load local data 

s_mul_i32 s19, s9, 6 // (op 309) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 309) least significant
s_mul_hi_u32 s15, s8, 6 // (op 309) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 309) most significant: sum

v_lshrrev_b64 v[92:93], 3, s[18:19] // (op 309) 


s_lshl_b64 s[18:19], s[8:9], 8 // (op 309) 
s_mul_i32 s39, s19, 6 // (op 309) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 309) least significant
s_mul_hi_u32 s15, s18, 6 // (op 309) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 309) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 309) 

s_mul_i32 s15, 3, s18 // (op 309) 

v_add_u32 v94, v29, s15 // (op 309) 

buffer_load_dwordx3 v[46:48], v94, s[44:47], 0 offen // (op 309) Load value
v_add_u32 v94, v94, v92 // (op 309) 
buffer_load_dwordx3 v[50:52], v94, s[44:47], 0 offen // (op 309) Load value
v_add_u32 v94, v94, v92 // (op 309) 
buffer_load_dwordx3 v[54:56], v94, s[44:47], 0 offen // (op 309) Load value
v_add_u32 v94, v94, v92 // (op 309) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:91], v[78:83], a[0:3], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 296) 
buffer_load_dwordx3 v[58:60], v94, s[44:47], 0 offen // (op 309) Load value











v_mov_b64 v[16:17], s[24:25] // (op 311) convert

s_lshl_b32 s15, s24, 3 // (op 311) 
s_mul_i32 s18, 3, s15 // (op 311) 

v_add_u32 v29, v53, s18 // (op 311) 

buffer_load_ubyte v62, v29, s[52:55], 0 offen offset: 0 // (op 311) Load value
v_add_u32 v29, v29, v16 // (op 311) 
buffer_load_ubyte v63, v29, s[52:55], 0 offen offset: 0 // (op 311) Load value


s_lshl_b64 s[18:19], s[8:9], 8 // (op 661) 

s_mul_i32 s39, s19, 6 // (op 661) most significant: low of high * low
s_barrier  // (op 1299) 
s_mul_i32 s38, s18, 6 // (op 661) least significant

s_mul_hi_u32 s15, s18, 6 // (op 661) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 661) most significant: sum


s_lshl_b32 s15, s24, 4 // (op 789) 
v_add_u32 v22, v22, s15 // (op 789) 

s_lshr_b64 s[18:19], s[38:39], 3 // (op 661) 

s_lshl_b32 s15, s18, 1 // (op 661) 

v_add_u32 v13, v13, s15 // (op 661) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 537) 
v_mov_b64 v[16:17], s[18:19] // (op 537) convert


s_lshl_b64 s[18:19], s[2:3], 8 // (op 537) 
s_mul_i32 s39, 3, s19 // (op 537) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 537) least significant
s_mul_hi_u32 s15, 3, s18 // (op 537) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 537) most significant: sum


v_mov_b32 v29, s38 // (op 537) 
v_mov_b32 v53, s39 // (op 537) 
v_add_co_u32 v78, s[18:19], v24, v29 // (op 537) least significant half
v_addc_co_u32 v79, s[18:19], v25, v53, s[18:19] // (op 537) most significant half




s_add_u32 s15, 4096, s1 // (op 537) 
s_mov_b32 m0, s15 // (op 537) reg expression
ds_read_b64_tr_b8 v[80:81], v3 offset:6144 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v3 offset:6272 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v3 offset:7168 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v3 offset:7296 // (op 298) Transpose load local data 
buffer_load_dwordx4 v78, s[40:43], 0 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v16 // (op 537) least significant half



v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
ds_read_b96_tr_b6 v[88:90], v9 offset:14592 // (op 300) Transpose load local data 
ds_read_b96_tr_b6 v[91:93], v9 offset:14848 // (op 300) Transpose load local data 
buffer_load_dwordx4 v78, s[40:43], 0 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v16 // (op 537) least significant half


s_barrier  // (op 1377) 
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
buffer_load_dwordx4 v78, s[40:43], 0 offen offset: 0 lds // (op 537) Load value direct to lds
ds_read_u8 v29, v8 offset:16832 // (op 302) Load local data 
v_add_co_u32 v78, s[18:19], v78, v16 // (op 537) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 537) most significant half



s_barrier  // (op 1381) 
s_add_u32 m0, m0, 1024 // (op 537) 
ds_read_u8 v53, v11 offset:16704 // (op 304) Load local data 
buffer_load_dwordx4 v78, s[40:43], 0 offen offset: 0 lds // (op 537) Load value direct to lds





s_barrier  // (op 1385) 
v_mov_b64 v[16:17], s[12:13] // (op 310) convert

s_lshl_b32 s15, s12, 3 // (op 310) 
s_mul_i32 s18, 3, s15 // (op 310) 

v_add_u32 v24, v23, s18 // (op 310) 

buffer_load_ubyte v14, v24, s[48:51], 0 offen offset: 0 // (op 310) Load value
v_add_u32 v24, v24, v16 // (op 310) 
s_lshl_b64 s[18:19], s[2:3], 8 // (op 597) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 597) 

v_mov_b32 v25, s38 // (op 597) 
buffer_load_ubyte v15, v24, s[48:51], 0 offen offset: 0 // (op 310) Load value


v_mov_b32 v24, s39 // (op 597) 

v_add_co_u32 v18, s[18:19], v18, v25 // (op 597) least significant half
s_barrier  // (op 1238) 
v_addc_co_u32 v19, s[18:19], v19, v24, s[18:19] // (op 597) most significant half







s_lshl_b32 s15, s12, 4 // (op 725) 
v_add_u32 v21, v21, s15 // (op 725) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:87], v[88:93], a[0:3], v29, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 305) 
s_waitcnt vmcnt(23)
ds_write_b96 v33, v[26:28] offset:8192 // (op 13) Store local data 




s_waitcnt vmcnt(22)
ds_write_b96 v33, v[30:32] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v33, v[34:36] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v33, v[38:40] offset:8240 // (op 13) Store local data 
s_barrier  // (op 1279) 



s_waitcnt vmcnt(19)
ds_write_b8 v49, v42 offset:12288 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v49, v43 offset:12304 // (op 30) Store local data 
s_barrier  // (op 1259) 



s_waitcnt vmcnt(13)
ds_write_b8 v41, v44 offset:12416 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v41, v45 offset:12432 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1248) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1289) 


ds_read_u8 v12, v7 offset:12288 // (op 87) Load local data 
s_barrier  // (op 1228) 



ds_read_b64_tr_b8 v[70:71], v0 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:128 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:1024 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:1152 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1269) 


ds_read_u8 v20, v4 offset:12416 // (op 82) Load local data 
s_ashr_i64 s[18:19], s[6:7], 8 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60:
















s_barrier  // (op 1333) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 560) 
s_lshr_b32 s14, s0, 1 // (op 560) 

v_lshl_add_u32 v13, s14, 9, v1 // (op 560) 

s_lshl_b32 s0, s16, 4 // (op 560) 
v_mul_lo_u32 v17, v13, s3 // (op 560) most significant: low of low * high
v_mul_hi_u32 v49, v13, s2 // (op 560) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 560) least significant: low of low * low
v_add_u32 v17, v17, v49 // (op 560) most significant: sum


s_mov_b32 s15, 0 // (op 560) convert
s_mov_b32 s14, s0 // (op 560) convert

v_mov_b32 v13, s14 // (op 560) 
v_mov_b32 v49, s15 // (op 560) 
v_add_co_u32 v18, s[0:1], v13, v16 // (op 560) least significant half
v_addc_co_u32 v19, s[0:1], v49, v17, s[0:1] // (op 560) most significant half





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

v_lshlrev_b32 v13, 2, v1 // (op 627) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 627) 
s_lshr_b32 s14, s0, 1 // (op 627) 

v_lshl_add_u32 v49, s14, 9, v13 // (op 627) 


s_lshl_b32 s0, s17, 4 // (op 627) 
s_mov_b32 s15, 0 // (op 627) convert
s_mov_b32 s14, s0 // (op 627) convert

v_mul_lo_u32 v17, v49, s9 // (op 627) most significant: low of low * high
v_mul_hi_u32 v13, v49, s8 // (op 627) most significant: high of low * low
v_mul_lo_u32 v16, v49, s8 // (op 627) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 627) most significant: sum

v_mov_b32 v13, s14 // (op 627) 
v_mov_b32 v53, s15 // (op 627) 
v_add_co_u32 v16, s[0:1], v16, v13 // (op 627) least significant half
v_addc_co_u32 v17, s[0:1], v17, v53, s[0:1] // (op 627) most significant half





v_mul_lo_u32 v45, v17, 6 // (op 627) most significant: low of high * low
v_mul_hi_u32 v13, v16, 6 // (op 627) most significant: high of low * low
v_mul_lo_u32 v44, v16, 6 // (op 627) least significant: low of low * low
v_add_u32 v45, v45, v13 // (op 627) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[44:45] // (op 627) 

v_mov_b32 v13, v16 // (op 627) convert

s_mul_i32 s1, s7, 6 // (op 629) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 629) least significant
s_mul_hi_u32 s14, s6, 6 // (op 629) most significant: high of low * low
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

v_and_b32 v49, -16, v1 // (op 691) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 691) 
v_sub_u32 v53, v1, v49 // (op 691) 

s_lshr_b32 s14, s0, 1 // (op 691) 

v_lshl_add_u32 v49, s16, 4, v53 // (op 691) 

v_lshrrev_b32 v53, 4, v1 // (op 691) 
v_lshlrev_b32 v57, 1, v53 // (op 691) 

v_lshl_add_u32 v53, s14, 4, v57 // (op 691) 


v_mul_lo_u32 v57, v53, s12 // (op 691) 

v_add_u32 v53, v49, v57 // (op 691) 


s_mul_i32 s1, s12, s23 // (op 693) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 693) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 693) least significant
s_mul_hi_u32 s15, s12, s22 // (op 693) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 693) most significant: sum
s_add_u32 s1, s1, s15 // (op 693) most significant: sum


s_mov_b32 s48, s20 // (op 693) reg expression
s_mov_b32 s49, s21 // (op 693) reg expression
s_mov_b32 s50, s0 // (op 693) reg expression
s_mov_b32 s51, 131072 // (op 693) call()

v_lshrrev_b32 v49, 4, v1 // (op 755) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 755) 
v_lshlrev_b32 v57, 1, v49 // (op 755) 

s_lshr_b32 s14, s0, 1 // (op 755) 

v_and_b32 v49, -16, v1 // (op 755) 
v_lshl_add_u32 v61, s14, 4, v57 // (op 755) 


v_sub_u32 v57, v1, v49 // (op 755) 

v_lshl_add_u32 v49, s17, 4, v57 // (op 755) 

v_mul_lo_u32 v57, v61, s24 // (op 755) 
v_add_u32 v57, v57, v49 // (op 755) 


s_mul_i32 s1, s24, s23 // (op 757) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 757) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 757) least significant
s_mul_hi_u32 s15, s24, s22 // (op 757) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 757) most significant: sum
s_add_u32 s1, s1, s15 // (op 757) most significant: sum


s_mov_b32 s52, s26 // (op 757) reg expression
s_mov_b32 s53, s27 // (op 757) reg expression
s_mov_b32 s54, s0 // (op 757) reg expression
s_mov_b32 s55, 131072 // (op 757) call()

v_lshlrev_b32 v49, 4, v1 // (op 848) 
s_nop 0
v_readfirstlane_b32 s0, v49 // (op 848) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v49, 6, v1 // (op 900) 
v_mul_lo_u32 v61, v49, 6 // (op 900) 
v_lshrrev_b32 v16, 3, v61 // (op 900) 

v_lshrrev_b32 v17, 4, v49 // (op 900) 

v_lshlrev_b32 v49, 2, v17 // (op 900) 

v_add_u32 v17, v16, v49 // (op 900) 


v_and_b32 v16, -16, v1 // (op 953) 
v_sub_u32 v49, v1, v16 // (op 953) 

v_lshrrev_b32 v16, 4, v1 // (op 953) 
v_lshl_add_u32 v61, v16, 5, v49 // (op 953) 


v_and_b32 v16, -16, v1 // (op 1006) 
v_sub_u32 v49, v1, v16 // (op 1006) 

v_lshrrev_b32 v16, 4, v1 // (op 1006) 
v_lshl_add_u32 v21, v16, 5, v49 // (op 1006) 


s_ashr_i64 s[14:15], s[6:7], 8 // (op 198) 
s_and_b32 s1, s14, 4294967294 // (op 198) 

s_ashr_i64 s[14:15], s[6:7], 8 // (op 197) 
s_mov_b32 s18, s14 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197:
s_barrier  // (op 1222) 
s_add_u32 s1, s1, 1 // (op 199) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 531) 
v_mov_b64 v[14:15], s[14:15] // (op 531) convert


v_mov_b64 v[22:23], v[18:19] // (op 531) 
v_mov_b64 v[24:25], v[22:23] // (op 531) reg expression
s_mov_b32 m0, s0 // (op 531) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 531) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 531) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 531) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds



s_lshl_b64 s[14:15], s[2:3], 8 // (op 550) 
v_mov_b32 v12, s14 // (op 550) 
v_mov_b32 v20, s15 // (op 550) 
v_add_co_u32 v18, s[18:19], v18, v12 // (op 550) least significant half
v_addc_co_u32 v19, s[18:19], v19, v20, s[18:19] // (op 550) most significant half





s_mul_i32 s15, s9, 6 // (op 225) most significant: low of high * low
s_mul_i32 s14, s8, 6 // (op 225) least significant
s_mul_hi_u32 s18, s8, 6 // (op 225) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 225) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[14:15] // (op 225) 


v_mov_b32 v12, v13 // (op 225) 
v_mov_b32 v16, v12 // (op 225) reg expression
buffer_load_dwordx3 v[22:24], v16, s[44:47], 0 offen // (op 225) Load value
v_add_u32 v16, v16, v14 // (op 225) 
buffer_load_dwordx3 v[26:28], v16, s[44:47], 0 offen // (op 225) Load value
v_add_u32 v16, v16, v14 // (op 225) 
buffer_load_dwordx3 v[30:32], v16, s[44:47], 0 offen // (op 225) Load value
v_add_u32 v16, v16, v14 // (op 225) 
buffer_load_dwordx3 v[34:36], v16, s[44:47], 0 offen // (op 225) Load value



s_lshl_b64 s[14:15], s[8:9], 8 // (op 619) 
s_mul_i32 s19, s15, 6 // (op 619) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 619) least significant
s_mul_hi_u32 s38, s14, 6 // (op 619) most significant: high of low * low
s_add_u32 s19, s19, s38 // (op 619) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 619) 

v_add_u32 v13, v13, s14 // (op 619) 


s_barrier  // (op 1241) 
v_mov_b64 v[14:15], s[12:13] // (op 226) convert




v_mov_b32 v12, v53 // (op 226) 
v_mov_b32 v16, v12 // (op 226) reg expression
buffer_load_ubyte v54, v16, s[48:51], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v16, v16, v14 // (op 226) 
s_waitcnt vmcnt(4)
ds_write_b96 v17, v[22:24] offset:8192 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v17, v[26:28] offset:8208 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[30:32] offset:8224 // (op 222) Store local data 
buffer_load_ubyte v55, v16, s[48:51], 0 offen offset: 0 // (op 226) Load value



s_lshl_b32 s14, s12, 3 // (op 683) 
v_add_u32 v53, v53, s14 // (op 683) 

s_waitcnt vmcnt(2)
ds_write_b96 v17, v[34:36] offset:8240 // (op 222) Store local data 


v_mov_b64 v[14:15], s[24:25] // (op 227) convert

s_barrier  // (op 1262) 
v_mov_b32 v12, v57 // (op 227) 



v_mov_b32 v16, v12 // (op 227) reg expression
buffer_load_ubyte v22, v16, s[52:55], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v16, v16, v14 // (op 227) 
buffer_load_ubyte v23, v16, s[52:55], 0 offen offset: 0 // (op 227) Load value



s_lshl_b32 s14, s24, 3 // (op 747) 
v_add_u32 v57, v57, s14 // (op 747) 

s_waitcnt vmcnt(3)
ds_write_b8 v61, v54 offset:12416 // (op 223) Store local data 
s_barrier  // (op 1231) 
s_waitcnt vmcnt(2)
ds_write_b8 v61, v55 offset:12432 // (op 223) Store local data 




ds_read_b64_tr_b8 v[24:25], v0 // (op 205) Transpose load local data 
s_barrier  // (op 1282) 
ds_read_b64_tr_b8 v[26:27], v0 offset:128 // (op 205) Transpose load local data 



ds_read_b64_tr_b8 v[28:29], v0 offset:1024 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:1152 // (op 205) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v21, v22 offset:12288 // (op 224) Store local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1252) 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v23 offset:12304 // (op 224) Store local data 




ds_read_b96_tr_b6 v[32:34], v5 offset:8192 // (op 207) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v5 offset:8448 // (op 207) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1272) 


ds_read_u8 v12, v4 offset:12416 // (op 209) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1292) 


ds_read_u8 v20, v7 offset:12288 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:37], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 212) 







ds_read_b64_tr_b8 v[22:23], v0 offset:2048 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:2176 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:3072 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:3200 // (op 214) Transpose load local data 



ds_read_b96_tr_b6 v[30:32], v5 offset:10240 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v5 offset:10496 // (op 216) Transpose load local data 


ds_read_u8 v12, v4 offset:12480 // (op 218) Load local data 


ds_read_u8 v20, v7 offset:12352 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 221) 




s_ashr_i64 s[14:15], s[6:7], 8 // (op 197) 
s_mov_b32 s18, s14 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 if true)
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197:























s_barrier  // (op 1369) 








v_cmp_eq_f32 vcc, 0.00000, v6 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v10, v12 // (op 50) 
v_mul_f32 v17, v10, v13 // (op 50) 
v_mul_f32 v18, v10, v14 // (op 50) 
v_mul_f32 v19, v10, v15 // (op 50) 

v_lshrrev_b32 v12, 16, v16 // (op 56) 
v_lshrrev_b32 v13, 16, v17 // (op 56) 
v_lshrrev_b32 v14, 16, v18 // (op 56) 
v_lshrrev_b32 v15, 16, v19 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 1048) 
v_and_b32 v11, 63, v1 // (op 1048) 
v_lshrrev_b32 v3, 4, v11 // (op 1048) 
v_lshlrev_b32 v4, 2, v3 // (op 1048) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1048) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 1048) 

v_lshrrev_b32 v3, 2, v11 // (op 1048) 

v_and_b32 v11, 3, v3 // (op 1048) 

v_and_b32 v3, 3, v1 // (op 1048) 
v_lshl_add_u32 v4, v11, 2, v3 // (op 1048) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 1048) 

v_mul_lo_u32 v11, v3, s28 // (op 1048) 

v_add_lshl_u32 v3, v0, v11, 1 // (op 1048) 


s_mul_i32 s1, s28, s33 // (op 1050) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1050) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1050) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1050) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1050) most significant: sum
s_add_u32 s1, s1, s3 // (op 1050) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1050) 

s_mov_b32 s4, s30 // (op 1050) reg expression
s_mov_b32 s5, s31 // (op 1050) reg expression
s_mov_b32 s6, s2 // (op 1050) reg expression
s_mov_b32 s7, 131072 // (op 1050) call()


v_pack_b32_f16 v4, v12, v13 // (op 58) 
v_pack_b32_f16 v5, v14, v15 // (op 58) 


buffer_store_dwordx2 v[4:5], v3, s[4:7], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157:
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


v_lshl_add_u32 v3, s17, 4, v5 // (op 818) 

v_mul_lo_u32 v4, v3, s34 // (op 818) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 818) 


s_mul_i32 s1, s34, s33 // (op 820) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 820) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 820) least significant
s_mul_hi_u32 s3, s34, s32 // (op 820) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 820) most significant: sum
s_add_u32 s1, s1, s3 // (op 820) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 820) 

s_mov_b32 s8, s36 // (op 820) reg expression
s_mov_b32 s9, s37 // (op 820) reg expression
s_mov_b32 s10, s2 // (op 820) reg expression
s_mov_b32 s11, 131072 // (op 820) call()





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
v_and_b32 v8, 65535, v4 // (op 171) 
v_lshrrev_b32 v9, 16, v4 // (op 171) 
v_lshlrev_b32 v8, 16, v8 // (op 171) 
v_lshlrev_b32 v9, 16, v9 // (op 171) 
v_mul_f32 v12, v6, v8 // (op 171) 
v_mul_f32 v13, v6, v9 // (op 171) 
v_and_b32 v8, 65535, v5 // (op 171) 
v_lshrrev_b32 v9, 16, v5 // (op 171) 
v_lshlrev_b32 v8, 16, v8 // (op 171) 
v_lshlrev_b32 v9, 16, v9 // (op 171) 
v_mul_f32 v14, v6, v8 // (op 171) 
v_mul_f32 v15, v6, v9 // (op 171) 


v_add_f32 v20, v12, v16 // (op 167) 
v_add_f32 v21, v13, v17 // (op 167) 
v_add_f32 v22, v14, v18 // (op 167) 
v_add_f32 v23, v15, v19 // (op 167) 


v_lshrrev_b32 v16, 16, v20 // (op 168) 
v_lshrrev_b32 v17, 16, v21 // (op 168) 
v_lshrrev_b32 v18, 16, v22 // (op 168) 
v_lshrrev_b32 v19, 16, v23 // (op 168) 

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


v_lshl_add_u32 v3, s17, 4, v5 // (op 1065) 

v_mul_lo_u32 v4, v3, s28 // (op 1065) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 1065) 


s_mul_i32 s1, s28, s33 // (op 1067) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1067) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1067) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1067) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1067) most significant: sum
s_add_u32 s1, s1, s3 // (op 1067) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1067) 

s_mov_b32 s4, s30 // (op 1067) reg expression
s_mov_b32 s5, s31 // (op 1067) reg expression
s_mov_b32 s6, s2 // (op 1067) reg expression
s_mov_b32 s7, 131072 // (op 1067) call()


v_pack_b32_f16 v4, v16, v17 // (op 169) 
v_pack_b32_f16 v5, v18, v19 // (op 169) 


buffer_store_dwordx2 v[4:5], v3, s[4:7], 0 offen // (op 169) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157:



s_endpgm  // End of RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
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
    DataTypeB: F6
    DestDataType: b
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
  - .name: RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NT_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 95
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

