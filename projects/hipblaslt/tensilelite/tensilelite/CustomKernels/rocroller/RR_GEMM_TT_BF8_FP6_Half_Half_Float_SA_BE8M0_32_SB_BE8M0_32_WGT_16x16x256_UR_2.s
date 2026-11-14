// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 875) 
v_and_b32 v3, 15, v1 // (op 875) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 875) 


v_and_b32 v0, 48, v1 // (op 875) 
v_lshl_add_u32 v3, v4, 8, v0 // (op 875) 


v_and_b32 v0, 63, v1 // (op 928) 
v_and_b32 v4, 48, v1 // (op 928) 
v_sub_u32 v5, v0, v4 // (op 928) 

v_lshrrev_b32 v4, 4, v0 // (op 928) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 928) 


v_lshlrev_b32 v4, 4, v0 // (op 928) 
v_mul_lo_u32 v5, v4, 6 // (op 928) 

v_lshrrev_b32 v4, 3, v5 // (op 928) 

v_and_b32 v5, 268435455, v0 // (op 928) 

v_lshlrev_b32 v0, 2, v5 // (op 928) 

v_add_u32 v5, v4, v0 // (op 928) 


v_lshrrev_b32 v0, 6, v1 // (op 981) 
v_and_b32 v4, 15, v1 // (op 981) 
v_lshl_add_u32 v6, v0, 4, v4 // (op 981) 


v_and_b32 v0, 63, v1 // (op 981) 
v_lshrrev_b32 v4, 4, v0 // (op 981) 

v_lshl_add_u32 v0, v6, 3, v4 // (op 981) 


v_and_b32 v4, 48, v1 // (op 1034) 
v_and_b32 v6, 15, v1 // (op 1034) 
v_add_u32 v7, v4, v6 // (op 1034) 


v_lshrrev_b32 v4, 6, v1 // (op 1107) 
v_and_b32 v6, 15, v1 // (op 1107) 
v_lshl_add_u32 v8, v4, 4, v6 // (op 1107) 


v_and_b32 v4, 48, v1 // (op 1107) 
v_lshl_add_u32 v6, v8, 8, v4 // (op 1107) 


v_and_b32 v4, 63, v1 // (op 1131) 
v_and_b32 v8, 48, v1 // (op 1131) 
v_sub_u32 v9, v4, v8 // (op 1131) 

v_lshrrev_b32 v8, 4, v4 // (op 1131) 

v_lshl_add_u32 v4, v8, 5, v9 // (op 1131) 


v_lshlrev_b32 v8, 4, v4 // (op 1131) 
v_mul_lo_u32 v9, v8, 6 // (op 1131) 

v_lshrrev_b32 v8, 3, v9 // (op 1131) 

v_and_b32 v9, 268435455, v4 // (op 1131) 

v_lshlrev_b32 v4, 2, v9 // (op 1131) 

v_add_u32 v9, v8, v4 // (op 1131) 


v_lshrrev_b32 v4, 6, v1 // (op 1155) 
v_and_b32 v8, 15, v1 // (op 1155) 
v_lshl_add_u32 v10, v4, 4, v8 // (op 1155) 


v_and_b32 v4, 63, v1 // (op 1155) 
v_lshrrev_b32 v8, 4, v4 // (op 1155) 

v_lshl_add_u32 v4, v10, 3, v8 // (op 1155) 


v_and_b32 v8, 48, v1 // (op 1179) 
v_and_b32 v10, 15, v1 // (op 1179) 
v_add_u32 v11, v8, v10 // (op 1179) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_lshrrev_b32 v12, 4, v1 // (op 586) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 586) 

v_and_b32 v12, -16, v1 // (op 586) 
v_sub_u32 v14, v1, v12 // (op 586) 

v_lshlrev_b32 v12, 4, v14 // (op 586) 

v_mov_b32 v15, 0 // (op 586) convert
v_mov_b32 v14, v12 // (op 586) convert

v_mul_lo_u32 v17, v13, s3 // (op 586) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 586) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 586) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 586) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 586) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 586) most significant half



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

v_lshrrev_b32 v12, 4, v1 // (op 605) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 605) 

v_and_b32 v12, -16, v1 // (op 605) 
v_sub_u32 v14, v1, v12 // (op 605) 

v_lshlrev_b32 v12, 4, v14 // (op 605) 

v_mov_b32 v15, 0 // (op 605) convert
v_mov_b32 v14, v12 // (op 605) convert

v_mul_lo_u32 v19, v13, s3 // (op 605) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 605) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 605) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 605) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 605) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 605) most significant half



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

v_lshrrev_b32 v14, 3, v1 // (op 714) 
v_lshlrev_b32 v15, 1, v14 // (op 714) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 714) 

v_and_b32 v15, -8, v1 // (op 714) 
v_sub_u32 v20, v1, v15 // (op 714) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v15, v14, s20 // (op 714) 
v_add_u32 v15, v15, v20 // (op 714) 


s_mul_i32 s1, s20, s7 // (op 716) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 716) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 716) least significant
s_mul_hi_u32 s15, s20, s6 // (op 716) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 716) most significant: sum
s_add_u32 s1, s1, s15 // (op 716) most significant: sum


s_mov_b32 s48, s22 // (op 716) reg expression
s_mov_b32 s49, s23 // (op 716) reg expression
s_mov_b32 s50, s0 // (op 716) reg expression
s_mov_b32 s51, 131072 // (op 716) call()

v_lshrrev_b32 v14, 3, v1 // (op 733) 
v_lshlrev_b32 v20, 1, v14 // (op 733) 

v_lshl_add_u32 v14, s16, 4, v20 // (op 733) 

v_and_b32 v20, -8, v1 // (op 733) 
v_sub_u32 v21, v1, v20 // (op 733) 

v_mul_lo_u32 v20, v14, s20 // (op 733) 
v_add_u32 v20, v20, v21 // (op 733) 


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

v_and_b32 v14, -16, v1 // (op 778) 
v_sub_u32 v21, v1, v14 // (op 778) 

v_lshl_add_u32 v14, s17, 4, v21 // (op 778) 

v_lshrrev_b32 v21, 4, v1 // (op 778) 
v_lshlrev_b32 v22, 1, v21 // (op 778) 

v_mul_lo_u32 v21, v22, s24 // (op 778) 
v_add_u32 v21, v21, v14 // (op 778) 


s_mul_i32 s1, s24, s29 // (op 780) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 780) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 780) least significant
s_mul_hi_u32 s15, s24, s28 // (op 780) most significant: high of low * low
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

v_mov_b32 v14, v21 // (op 431) reg expression
buffer_load_ubyte v42, v14, s[52:55], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v14, v14, v24 // (op 431) 
buffer_load_ubyte v43, v14, s[52:55], 0 offen offset: 0 // (op 431) Load value


v_and_b32 v14, -16, v1 // (op 863) 
v_sub_u32 v29, v1, v14 // (op 863) 

v_lshlrev_b32 v14, 4, v29 // (op 863) 

v_lshrrev_b32 v29, 4, v1 // (op 863) 
v_lshl_add_u32 v33, v29, 8, v14 // (op 863) 


s_nop 0
v_readfirstlane_b32 s0, v33 // (op 863) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 2 // (op 542) 
v_mov_b64 v[24:25], s[14:15] // (op 542) convert


v_mov_b64 v[44:45], v[16:17] // (op 542) 
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




v_mov_b64 v[24:25], s[20:21] // (op 437) convert

v_mov_b32 v14, v15 // (op 437) 
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
v_add_u32 v12, v21, s0 // (op 443) 

buffer_load_ubyte v62, v12, s[52:55], 0 offen offset: 0 // (op 443) Load value
v_add_u32 v12, v12, v24 // (op 443) 
buffer_load_ubyte v63, v12, s[52:55], 0 offen offset: 0 // (op 443) Load value



v_and_b32 v12, -16, v1 // (op 1095) 
v_sub_u32 v14, v1, v12 // (op 1095) 

v_lshlrev_b32 v12, 4, v14 // (op 1095) 

v_lshrrev_b32 v14, 4, v1 // (op 1095) 
v_lshl_add_u32 v21, v14, 8, v12 // (op 1095) 


s_nop 0
v_readfirstlane_b32 s0, v21 // (op 1095) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 2 // (op 547) 
v_mov_b64 v[24:25], s[14:15] // (op 547) convert


v_mov_b64 v[64:65], v[16:17] // (op 547) 
s_add_u32 s1, 4096, s0 // (op 547) 
s_mov_b32 m0, s1 // (op 547) reg expression
s_mov_b32 s14, 256 // (op 547) call()
buffer_load_dwordx4 v64, s[40:43], s14 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v64, s[18:19], v64, v24 // (op 547) least significant half
v_addc_co_u32 v65, s[18:19], 0, v65, s[18:19] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v64, s[40:43], s14 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v64, s[18:19], v64, v24 // (op 547) least significant half
v_addc_co_u32 v65, s[18:19], 0, v65, s[18:19] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v64, s[40:43], s14 offen offset: 0 lds // (op 547) Load value direct to lds
v_add_co_u32 v64, s[18:19], v64, v24 // (op 547) least significant half
v_addc_co_u32 v65, s[18:19], 0, v65, s[18:19] // (op 547) most significant half

s_add_u32 m0, m0, 1024 // (op 547) 
buffer_load_dwordx4 v64, s[40:43], s14 offen offset: 0 lds // (op 547) Load value direct to lds







v_mov_b64 v[16:17], s[20:21] // (op 449) convert

v_mov_b32 v12, v15 // (op 449) 
buffer_load_ubyte v24, v12, s[48:51], 0 offen offset:8 // (op 449) Load value
v_add_u32 v12, v12, v16 // (op 449) 
buffer_load_ubyte v25, v12, s[48:51], 0 offen offset:8 // (op 449) Load value



v_lshlrev_b32 v12, 6, v1 // (op 916) 
v_mul_lo_u32 v21, v12, 6 // (op 916) 
v_lshrrev_b32 v23, 3, v21 // (op 916) 

v_lshrrev_b32 v21, 4, v12 // (op 916) 

v_lshlrev_b32 v12, 2, v21 // (op 916) 

v_add_u32 v21, v23, v12 // (op 916) 





s_waitcnt vmcnt(23)
ds_write_b96 v21, v[26:28] offset:8192 // (op 451) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v21, v[30:32] offset:8208 // (op 451) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v21, v[34:36] offset:8224 // (op 451) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v21, v[38:40] offset:8240 // (op 451) Store local data 

v_and_b32 v12, -16, v1 // (op 1022) 
v_sub_u32 v21, v1, v12 // (op 1022) 

v_lshrrev_b32 v12, 4, v1 // (op 1022) 
v_lshl_add_u32 v23, v12, 5, v21 // (op 1022) 





s_waitcnt vmcnt(19)
ds_write_b8 v23, v42 offset:12288 // (op 452) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v23, v43 offset:12304 // (op 452) Store local data 

v_and_b32 v12, -8, v1 // (op 969) 
v_sub_u32 v21, v1, v12 // (op 969) 

v_lshrrev_b32 v12, 3, v1 // (op 969) 
v_lshl_add_u32 v23, v12, 4, v21 // (op 969) 





s_waitcnt vmcnt(13)
ds_write_b8 v23, v44 offset:12416 // (op 454) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v23, v45 offset:12424 // (op 454) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1235) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 471) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 471) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1276) 


ds_read_u8 v12, v7 offset:12288 // (op 477) Load local data 
s_barrier  // (op 1216) 



ds_read_b128 v[70:73], v3 // (op 468) Load local data 
ds_read_b128 v[74:77], v3 offset:64 // (op 468) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1256) 



ds_read_u8 v21, v0 offset:12416 // (op 474) Load local data 
v_and_b32 v23, -16, v1 // (op 835) 
v_sub_u32 v29, v1, v23 // (op 835) 

v_lshlrev_b32 v23, 4, v29 // (op 835) 

v_lshrrev_b32 v29, 4, v1 // (op 835) 
v_lshl_add_u32 v33, v29, 8, v23 // (op 835) 


s_nop 0
v_readfirstlane_b32 s0, v33 // (op 835) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v23, 6, v1 // (op 887) 
v_mul_lo_u32 v29, v23, 6 // (op 887) 
v_lshrrev_b32 v33, 3, v29 // (op 887) 

v_lshrrev_b32 v29, 4, v23 // (op 887) 

v_lshlrev_b32 v23, 2, v29 // (op 887) 

v_add_u32 v29, v33, v23 // (op 887) 


v_and_b32 v23, -8, v1 // (op 940) 
v_sub_u32 v33, v1, v23 // (op 940) 

v_lshrrev_b32 v23, 3, v1 // (op 940) 
v_lshl_add_u32 v37, v23, 4, v33 // (op 940) 


v_and_b32 v23, -16, v1 // (op 993) 
v_sub_u32 v33, v1, v23 // (op 993) 

v_lshrrev_b32 v23, 4, v1 // (op 993) 
v_lshl_add_u32 v41, v23, 5, v33 // (op 993) 


v_and_b32 v23, -16, v1 // (op 1080) 
v_sub_u32 v33, v1, v23 // (op 1080) 

v_lshlrev_b32 v23, 4, v33 // (op 1080) 

v_lshrrev_b32 v33, 4, v1 // (op 1080) 
v_lshl_add_u32 v49, v33, 8, v23 // (op 1080) 


s_nop 0
v_readfirstlane_b32 s1, v49 // (op 1080) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v23, 6, v1 // (op 1119) 
v_mul_lo_u32 v33, v23, 6 // (op 1119) 
v_lshrrev_b32 v49, 3, v33 // (op 1119) 

v_lshrrev_b32 v33, 4, v23 // (op 1119) 

v_lshlrev_b32 v23, 2, v33 // (op 1119) 

v_add_u32 v33, v49, v23 // (op 1119) 


v_and_b32 v23, -8, v1 // (op 1143) 
v_sub_u32 v49, v1, v23 // (op 1143) 

v_lshrrev_b32 v23, 3, v1 // (op 1143) 
v_lshl_add_u32 v53, v23, 4, v49 // (op 1143) 


v_and_b32 v23, -16, v1 // (op 1167) 
v_sub_u32 v49, v1, v23 // (op 1167) 

v_lshrrev_b32 v23, 4, v1 // (op 1167) 
v_lshl_add_u32 v57, v23, 5, v49 // (op 1167) 


s_mov_b32 s14, 0 // (op 61) call()
s_ashr_i64 s[18:19], s[12:13], 8 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60:

s_add_u32 s14, s14, 2 // (op 62) 
s_mul_i32 s19, s9, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 10) least significant
s_mul_hi_u32 s15, s8, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 10) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[18:19] // (op 10) 


v_mov_b32 v23, v13 // (op 10) 
s_lshl_b64 s[18:19], s[8:9], 8 // (op 10) 
s_mul_i32 s57, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s56, s18, 6 // (op 10) least significant
s_mul_hi_u32 s15, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s57, s57, s15 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[56:57], 3 // (op 10) 

s_lshl_b32 s15, s18, 1 // (op 10) 

v_add_u32 v49, v23, s15 // (op 10) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 90) 
buffer_load_dwordx3 v[26:28], v49, s[44:47], 0 offen // (op 10) Load value
s_barrier  // (op 1225) 
v_add_u32 v49, v49, v14 // (op 10) 



ds_read_b128 v[70:73], v3 offset:128 // (op 93) Load local data 
ds_read_b128 v[74:77], v3 offset:192 // (op 93) Load local data 
buffer_load_dwordx3 v[30:32], v49, s[44:47], 0 offen // (op 10) Load value
s_barrier  // (op 1245) 
v_add_u32 v49, v49, v14 // (op 10) 



ds_read_b96_tr_b6 v[64:66], v5 offset:10240 // (op 97) Transpose load local data 
buffer_load_dwordx3 v[34:36], v49, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v49, v49, v14 // (op 10) 
ds_read_b96_tr_b6 v[67:69], v5 offset:10496 // (op 97) Transpose load local data 
buffer_load_dwordx3 v[38:40], v49, s[44:47], 0 offen // (op 10) Load value



s_barrier  // (op 1266) 
v_mov_b64 v[14:15], s[24:25] // (op 25) convert




ds_read_u8 v21, v0 offset:12420 // (op 101) Load local data 
v_mov_b32 v49, v22 // (op 25) 
s_barrier  // (op 1286) 
s_lshl_b32 s15, s24, 4 // (op 25) 


ds_read_u8 v12, v7 offset:12352 // (op 105) Load local data 
v_add_u32 v61, v49, s15 // (op 25) 

buffer_load_ubyte v42, v61, s[52:55], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v61, v61, v14 // (op 25) 
buffer_load_ubyte v43, v61, s[52:55], 0 offen offset: 0 // (op 25) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 107) 
s_barrier  // (op 1219) 

s_lshl_b64 s[18:19], s[2:3], 2 // (op 526) 
v_mov_b64 v[14:15], s[18:19] // (op 526) convert


v_mov_b64 v[16:17], v[18:19] // (op 526) 
v_mov_b64 v[78:79], v[16:17] // (op 526) 
s_mov_b32 m0, s0 // (op 526) reg expression
s_mov_b32 s15, 512 // (op 526) call()
buffer_load_dwordx4 v78, s[40:43], s15 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v78, s[40:43], s15 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v78, s[40:43], s15 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v14 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v78, s[40:43], s15 offen offset: 0 lds // (op 526) Load value direct to lds




v_mov_b64 v[14:15], s[20:21] // (op 17) convert

v_mov_b32 v61, v20 // (op 17) 
v_mov_b32 v78, v61 // (op 17) 
buffer_load_ubyte v44, v78, s[48:51], 0 offen offset:16 // (op 17) Load value
v_add_u32 v78, v78, v14 // (op 17) 
buffer_load_ubyte v45, v78, s[48:51], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 1306) 



s_waitcnt vmcnt(23)
ds_write_b96 v33, v[46:48] offset:12544 // (op 306) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v33, v[50:52] offset:12560 // (op 306) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v33, v[54:56] offset:12576 // (op 306) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v33, v[58:60] offset:12592 // (op 306) Store local data 
s_barrier  // (op 1318) 



s_waitcnt vmcnt(19)
ds_write_b8 v57, v62 offset:16640 // (op 308) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v57, v63 offset:16656 // (op 308) Store local data 
s_barrier  // (op 1312) 



s_waitcnt vmcnt(13)
ds_write_b8 v53, v24 offset:16768 // (op 307) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v53, v25 offset:16776 // (op 307) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1302) 



ds_read_b96_tr_b6 v[78:80], v9 offset:12544 // (op 291) Transpose load local data 
ds_read_b96_tr_b6 v[81:83], v9 offset:12800 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1315) 


ds_read_u8 v14, v11 offset:16640 // (op 295) Load local data 
s_barrier  // (op 1296) 



ds_read_b128 v[84:87], v6 offset:4096 // (op 289) Load local data 
ds_read_b128 v[88:91], v6 offset:4160 // (op 289) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1309) 



ds_read_u8 v15, v4 offset:16768 // (op 293) Load local data 

s_mul_i32 s19, s9, 6 // (op 309) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 309) least significant
s_mul_hi_u32 s15, s8, 6 // (op 309) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 309) most significant: sum

v_lshrrev_b64 v[92:93], 3, s[18:19] // (op 309) 


s_lshl_b64 s[18:19], s[8:9], 8 // (op 309) 
s_mul_i32 s57, s19, 6 // (op 309) most significant: low of high * low
s_mul_i32 s56, s18, 6 // (op 309) least significant
s_mul_hi_u32 s15, s18, 6 // (op 309) most significant: high of low * low
s_add_u32 s57, s57, s15 // (op 309) most significant: sum


s_lshr_b64 s[18:19], s[56:57], 3 // (op 309) 

s_mul_i32 s15, 3, s18 // (op 309) 

v_add_u32 v94, v23, s15 // (op 309) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:91], v[78:83], a[0:3], v15, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 296) 
buffer_load_dwordx3 v[46:48], v94, s[44:47], 0 offen // (op 309) Load value




v_add_u32 v94, v94, v92 // (op 309) 



ds_read_b128 v[78:81], v6 offset:4224 // (op 298) Load local data 
ds_read_b128 v[82:85], v6 offset:4288 // (op 298) Load local data 
buffer_load_dwordx3 v[50:52], v94, s[44:47], 0 offen // (op 309) Load value
v_add_u32 v94, v94, v92 // (op 309) 



ds_read_b96_tr_b6 v[86:88], v9 offset:14592 // (op 300) Transpose load local data 
buffer_load_dwordx3 v[54:56], v94, s[44:47], 0 offen // (op 309) Load value
v_add_u32 v94, v94, v92 // (op 309) 
ds_read_b96_tr_b6 v[89:91], v9 offset:14848 // (op 300) Transpose load local data 
buffer_load_dwordx3 v[58:60], v94, s[44:47], 0 offen // (op 309) Load value







s_barrier  // (op 1377) 
ds_read_u8 v23, v4 offset:16772 // (op 302) Load local data 
v_mov_b64 v[14:15], s[24:25] // (op 311) convert

s_lshl_b32 s15, s24, 3 // (op 311) 


s_barrier  // (op 1381) 
ds_read_u8 v92, v11 offset:16704 // (op 304) Load local data 
s_mul_i32 s18, 3, s15 // (op 311) 

v_add_u32 v93, v49, s18 // (op 311) 

s_barrier  // (op 1385) 
buffer_load_ubyte v62, v93, s[52:55], 0 offen offset: 0 // (op 311) Load value
v_add_u32 v93, v93, v14 // (op 311) 
s_lshl_b64 s[18:19], s[8:9], 8 // (op 661) 
buffer_load_ubyte v63, v93, s[52:55], 0 offen offset: 0 // (op 311) Load value


s_mul_i32 s57, s19, 6 // (op 661) most significant: low of high * low

s_mul_i32 s56, s18, 6 // (op 661) least significant
s_mul_hi_u32 s15, s18, 6 // (op 661) most significant: high of low * low
s_add_u32 s57, s57, s15 // (op 661) most significant: sum


s_lshl_b32 s15, s24, 4 // (op 789) 
v_add_u32 v22, v22, s15 // (op 789) 

s_lshr_b64 s[18:19], s[56:57], 3 // (op 661) 

s_lshl_b32 s15, s18, 1 // (op 661) 

v_add_u32 v13, v13, s15 // (op 661) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[86:91], a[0:3], v23, v92 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 305) 
s_barrier  // (op 1299) 





s_lshl_b64 s[18:19], s[2:3], 2 // (op 537) 
v_mov_b64 v[14:15], s[18:19] // (op 537) convert


v_mov_b64 v[78:79], v[16:17] // (op 537) 
s_add_u32 s15, 4096, s1 // (op 537) 
s_mov_b32 m0, s15 // (op 537) reg expression
s_mov_b32 s18, 768 // (op 537) call()
buffer_load_dwordx4 v78, s[40:43], s18 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v78, s[56:57], v78, v14 // (op 537) least significant half
v_addc_co_u32 v79, s[56:57], 0, v79, s[56:57] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
buffer_load_dwordx4 v78, s[40:43], s18 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v78, s[56:57], v78, v14 // (op 537) least significant half
v_addc_co_u32 v79, s[56:57], 0, v79, s[56:57] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
buffer_load_dwordx4 v78, s[40:43], s18 offen offset: 0 lds // (op 537) Load value direct to lds
v_add_co_u32 v78, s[56:57], v78, v14 // (op 537) least significant half
v_addc_co_u32 v79, s[56:57], 0, v79, s[56:57] // (op 537) most significant half

s_add_u32 m0, m0, 1024 // (op 537) 
buffer_load_dwordx4 v78, s[40:43], s18 offen offset: 0 lds // (op 537) Load value direct to lds






v_add_co_u32 v18, vcc, 512, v18 // (op 597) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 597) most significant half
v_mov_b64 v[14:15], s[20:21] // (op 310) convert

v_mov_b32 v23, v61 // (op 310) 
buffer_load_ubyte v24, v23, s[48:51], 0 offen offset:24 // (op 310) Load value
v_add_u32 v23, v23, v14 // (op 310) 
buffer_load_ubyte v25, v23, s[48:51], 0 offen offset:24 // (op 310) Load value



s_barrier  // (op 1238) 
v_add_u32 v20, 16, v20 // (op 725) 



s_waitcnt vmcnt(23)
ds_write_b96 v29, v[26:28] offset:8192 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v29, v[30:32] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v29, v[34:36] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v29, v[38:40] offset:8240 // (op 13) Store local data 
s_barrier  // (op 1279) 



s_waitcnt vmcnt(19)
ds_write_b8 v41, v42 offset:12288 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v41, v43 offset:12304 // (op 30) Store local data 
s_barrier  // (op 1259) 



s_waitcnt vmcnt(13)
ds_write_b8 v37, v44 offset:12416 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v37, v45 offset:12424 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1248) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 77) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1289) 


ds_read_u8 v12, v7 offset:12288 // (op 87) Load local data 
s_barrier  // (op 1228) 



ds_read_b128 v[70:73], v3 // (op 72) Load local data 
ds_read_b128 v[74:77], v3 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1269) 



ds_read_u8 v21, v0 offset:12416 // (op 82) Load local data 
s_ashr_i64 s[18:19], s[12:13], 8 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60:
















s_barrier  // (op 1333) 
v_lshrrev_b32 v49, 4, v1 // (op 560) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 560) 
v_lshl_add_u32 v53, s16, 4, v49 // (op 560) 

s_lshr_b32 s14, s0, 1 // (op 560) 

v_and_b32 v49, -16, v1 // (op 560) 
v_sub_u32 v57, v1, v49 // (op 560) 

v_lshlrev_b32 v49, 4, v57 // (op 560) 

v_lshl_add_u32 v57, s14, 9, v49 // (op 560) 


v_mov_b32 v23, 0 // (op 560) convert
v_mov_b32 v22, v57 // (op 560) convert

v_mul_lo_u32 v27, v53, s3 // (op 560) most significant: low of low * high
v_mul_hi_u32 v49, v53, s2 // (op 560) most significant: high of low * low
v_mul_lo_u32 v26, v53, s2 // (op 560) least significant: low of low * low
v_add_u32 v27, v27, v49 // (op 560) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v22 // (op 560) least significant half
v_addc_co_u32 v27, s[0:1], v27, v23, s[0:1] // (op 560) most significant half



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

v_lshlrev_b32 v49, 2, v1 // (op 627) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 627) 
s_lshr_b32 s14, s0, 1 // (op 627) 

v_lshl_add_u32 v53, s14, 9, v49 // (op 627) 


s_lshl_b32 s0, s17, 4 // (op 627) 
s_mov_b32 s15, 0 // (op 627) convert
s_mov_b32 s14, s0 // (op 627) convert

v_mul_lo_u32 v23, v53, s9 // (op 627) most significant: low of low * high
v_mul_hi_u32 v49, v53, s8 // (op 627) most significant: high of low * low
v_mul_lo_u32 v22, v53, s8 // (op 627) least significant: low of low * low
v_add_u32 v23, v23, v49 // (op 627) most significant: sum

v_mov_b32 v49, s14 // (op 627) 
v_mov_b32 v57, s15 // (op 627) 
v_add_co_u32 v22, s[0:1], v22, v49 // (op 627) least significant half
v_addc_co_u32 v23, s[0:1], v23, v57, s[0:1] // (op 627) most significant half





v_mul_lo_u32 v29, v23, 6 // (op 627) most significant: low of high * low
v_mul_hi_u32 v49, v22, 6 // (op 627) most significant: high of low * low
v_mul_lo_u32 v28, v22, 6 // (op 627) least significant: low of low * low
v_add_u32 v29, v29, v49 // (op 627) most significant: sum


v_lshrrev_b64 v[22:23], 3, v[28:29] // (op 627) 

v_mov_b32 v49, v22 // (op 627) convert

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

v_lshrrev_b32 v53, 3, v1 // (op 691) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 691) 
v_lshlrev_b32 v57, 1, v53 // (op 691) 

s_lshr_b32 s14, s0, 1 // (op 691) 

v_lshl_add_u32 v53, s16, 4, v57 // (op 691) 

v_and_b32 v57, -8, v1 // (op 691) 
v_sub_u32 v61, v1, v57 // (op 691) 

v_lshl_add_u32 v57, s14, 4, v61 // (op 691) 


v_mul_lo_u32 v61, v53, s20 // (op 691) 
v_add_u32 v61, v61, v57 // (op 691) 


s_mul_i32 s1, s20, s7 // (op 693) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 693) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 693) least significant
s_mul_hi_u32 s15, s20, s6 // (op 693) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 693) most significant: sum
s_add_u32 s1, s1, s15 // (op 693) most significant: sum


s_mov_b32 s48, s22 // (op 693) reg expression
s_mov_b32 s49, s23 // (op 693) reg expression
s_mov_b32 s50, s0 // (op 693) reg expression
s_mov_b32 s51, 131072 // (op 693) call()

v_lshrrev_b32 v53, 4, v1 // (op 755) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 755) 
v_lshlrev_b32 v57, 1, v53 // (op 755) 

s_lshr_b32 s14, s0, 1 // (op 755) 

v_and_b32 v53, -16, v1 // (op 755) 
v_lshl_add_u32 v13, s14, 4, v57 // (op 755) 


v_sub_u32 v57, v1, v53 // (op 755) 

v_lshl_add_u32 v53, s17, 4, v57 // (op 755) 

v_mul_lo_u32 v57, v13, s24 // (op 755) 
v_add_u32 v57, v57, v53 // (op 755) 


s_mul_i32 s1, s24, s29 // (op 757) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 757) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 757) least significant
s_mul_hi_u32 s15, s24, s28 // (op 757) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 757) most significant: sum
s_add_u32 s1, s1, s15 // (op 757) most significant: sum


s_mov_b32 s52, s26 // (op 757) reg expression
s_mov_b32 s53, s27 // (op 757) reg expression
s_mov_b32 s54, s0 // (op 757) reg expression
s_mov_b32 s55, 131072 // (op 757) call()

v_and_b32 v53, -16, v1 // (op 848) 
v_sub_u32 v13, v1, v53 // (op 848) 

v_lshlrev_b32 v53, 4, v13 // (op 848) 

v_lshrrev_b32 v13, 4, v1 // (op 848) 
v_lshl_add_u32 v14, v13, 8, v53 // (op 848) 


s_nop 0
v_readfirstlane_b32 s0, v14 // (op 848) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v53, 6, v1 // (op 900) 
v_mul_lo_u32 v13, v53, 6 // (op 900) 
v_lshrrev_b32 v14, 3, v13 // (op 900) 

v_lshrrev_b32 v13, 4, v53 // (op 900) 

v_lshlrev_b32 v53, 2, v13 // (op 900) 

v_add_u32 v13, v14, v53 // (op 900) 


v_and_b32 v53, -8, v1 // (op 953) 
v_sub_u32 v14, v1, v53 // (op 953) 

v_lshrrev_b32 v53, 3, v1 // (op 953) 
v_lshl_add_u32 v15, v53, 4, v14 // (op 953) 


v_and_b32 v14, -16, v1 // (op 1006) 
v_sub_u32 v53, v1, v14 // (op 1006) 

v_lshrrev_b32 v14, 4, v1 // (op 1006) 
v_lshl_add_u32 v16, v14, 5, v53 // (op 1006) 


s_ashr_i64 s[14:15], s[12:13], 8 // (op 198) 
s_and_b32 s1, s14, 4294967294 // (op 198) 

s_ashr_i64 s[14:15], s[12:13], 8 // (op 197) 
s_mov_b32 s18, s14 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197:
s_barrier  // (op 1222) 
s_add_u32 s1, s1, 1 // (op 199) 

s_lshl_b64 s[14:15], s[2:3], 2 // (op 531) 
v_mov_b64 v[24:25], s[14:15] // (op 531) convert


v_mov_b64 v[22:23], v[26:27] // (op 531) 
v_mov_b64 v[20:21], v[22:23] // (op 531) 
s_mov_b32 m0, s0 // (op 531) reg expression
buffer_load_dwordx4 v20, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v20, s[14:15], v20, v24 // (op 531) least significant half
v_addc_co_u32 v21, s[14:15], 0, v21, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v20, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v20, s[14:15], v20, v24 // (op 531) least significant half
v_addc_co_u32 v21, s[14:15], 0, v21, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v20, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v20, s[14:15], v20, v24 // (op 531) least significant half
v_addc_co_u32 v21, s[14:15], 0, v21, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v20, s[40:43], 0 offen offset: 0 lds // (op 531) Load value direct to lds



v_add_co_u32 v26, vcc, 256, v26 // (op 550) least significant half
v_addc_co_u32 v27, vcc, 0, v27, vcc // (op 550) most significant half

s_mul_i32 s15, s9, 6 // (op 225) most significant: low of high * low
s_mul_i32 s14, s8, 6 // (op 225) least significant
s_mul_hi_u32 s18, s8, 6 // (op 225) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 225) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[14:15] // (op 225) 


v_mov_b32 v12, v49 // (op 225) 
v_mov_b32 v14, v12 // (op 225) reg expression
buffer_load_dwordx3 v[18:20], v14, s[44:47], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v24 // (op 225) 
buffer_load_dwordx3 v[28:30], v14, s[44:47], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v24 // (op 225) 
buffer_load_dwordx3 v[32:34], v14, s[44:47], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v24 // (op 225) 
buffer_load_dwordx3 v[36:38], v14, s[44:47], 0 offen // (op 225) Load value



s_lshl_b64 s[14:15], s[8:9], 8 // (op 619) 
s_mul_i32 s19, s15, 6 // (op 619) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 619) least significant
s_mul_hi_u32 s56, s14, 6 // (op 619) most significant: high of low * low
s_add_u32 s19, s19, s56 // (op 619) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 619) 

v_add_u32 v49, v49, s14 // (op 619) 


s_barrier  // (op 1241) 
v_mov_b64 v[24:25], s[20:21] // (op 226) convert




v_mov_b32 v12, v61 // (op 226) 
v_mov_b32 v14, v12 // (op 226) 
buffer_load_ubyte v22, v14, s[48:51], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v14, v14, v24 // (op 226) 
s_waitcnt vmcnt(4)
ds_write_b96 v13, v[18:20] offset:8192 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v13, v[28:30] offset:8208 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v13, v[32:34] offset:8224 // (op 222) Store local data 
buffer_load_ubyte v23, v14, s[48:51], 0 offen offset: 0 // (op 226) Load value



v_add_u32 v61, 8, v61 // (op 683) 
s_waitcnt vmcnt(2)
ds_write_b96 v13, v[36:38] offset:8240 // (op 222) Store local data 


v_mov_b64 v[24:25], s[24:25] // (op 227) convert

s_barrier  // (op 1262) 
v_mov_b32 v12, v57 // (op 227) 



v_mov_b32 v14, v12 // (op 227) reg expression
buffer_load_ubyte v20, v14, s[52:55], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v14, v14, v24 // (op 227) 
buffer_load_ubyte v21, v14, s[52:55], 0 offen offset: 0 // (op 227) Load value



s_lshl_b32 s14, s24, 3 // (op 747) 
v_add_u32 v57, v57, s14 // (op 747) 

s_waitcnt vmcnt(3)
ds_write_b8 v15, v22 offset:12416 // (op 223) Store local data 
s_barrier  // (op 1231) 
s_waitcnt vmcnt(2)
ds_write_b8 v15, v23 offset:12424 // (op 223) Store local data 




ds_read_b128 v[28:31], v3 // (op 205) Load local data 
s_barrier  // (op 1282) 
ds_read_b128 v[32:35], v3 offset:64 // (op 205) Load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 1252) 
s_waitcnt vmcnt(1)
ds_write_b8 v16, v20 offset:12288 // (op 224) Store local data 



ds_read_b96_tr_b6 v[36:38], v5 offset:8192 // (op 207) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v5 offset:8448 // (op 207) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v16, v21 offset:12304 // (op 224) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1272) 



ds_read_u8 v12, v0 offset:12416 // (op 209) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1292) 


ds_read_u8 v14, v7 offset:12288 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:41], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 212) 







ds_read_b128 v[18:21], v3 offset:128 // (op 214) Load local data 
ds_read_b128 v[22:25], v3 offset:192 // (op 214) Load local data 



ds_read_b96_tr_b6 v[28:30], v5 offset:10240 // (op 216) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v5 offset:10496 // (op 216) Transpose load local data 



ds_read_u8 v12, v0 offset:12420 // (op 218) Load local data 


ds_read_u8 v14, v7 offset:12352 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:25], v[28:33], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 221) 




s_ashr_i64 s[14:15], s[12:13], 8 // (op 197) 
s_mov_b32 s18, s14 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 if true)
RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197:
























s_barrier  // (op 1369) 








v_cmp_eq_f32 vcc, 0.00000, v8 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157
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

v_lshrrev_b32 v0, 6, v1 // (op 1048) 
v_and_b32 v3, 63, v1 // (op 1048) 
v_lshrrev_b32 v9, 4, v3 // (op 1048) 
v_lshlrev_b32 v11, 2, v9 // (op 1048) 

v_lshl_add_u32 v9, v0, 4, v11 // (op 1048) 


v_lshl_add_u32 v0, s16, 4, v9 // (op 1048) 

v_lshrrev_b32 v9, 2, v3 // (op 1048) 

v_and_b32 v3, 3, v9 // (op 1048) 

v_and_b32 v9, 3, v1 // (op 1048) 
v_lshl_add_u32 v11, v3, 2, v9 // (op 1048) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 1048) 

v_mul_lo_u32 v9, v3, s30 // (op 1048) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 1048) 


s_mul_i32 s1, s30, s35 // (op 1050) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 1050) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 1050) least significant
s_mul_hi_u32 s3, s30, s34 // (op 1050) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1050) most significant: sum
s_add_u32 s1, s1, s3 // (op 1050) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1050) 

s_mov_b32 s4, s32 // (op 1050) reg expression
s_mov_b32 s5, s33 // (op 1050) reg expression
s_mov_b32 s6, s2 // (op 1050) reg expression
s_mov_b32 s7, 131072 // (op 1050) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[4:7], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157:
v_lshrrev_b32 v0, 6, v1 // (op 818) 
v_and_b32 v9, 63, v1 // (op 818) 
v_lshrrev_b32 v3, 4, v9 // (op 818) 
v_lshlrev_b32 v4, 2, v3 // (op 818) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 818) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 818) 

v_lshrrev_b32 v3, 2, v9 // (op 818) 

v_and_b32 v9, 3, v3 // (op 818) 

v_and_b32 v3, 3, v1 // (op 818) 
v_lshl_add_u32 v4, v9, 2, v3 // (op 818) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 818) 

v_mul_lo_u32 v9, v3, s36 // (op 818) 

v_add_lshl_u32 v3, v0, v9, 1 // (op 818) 


s_mul_i32 s1, s36, s35 // (op 820) most significant: low of low * high
s_mul_i32 s2, s37, s34 // (op 820) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 820) least significant
s_mul_hi_u32 s3, s36, s34 // (op 820) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 820) most significant: sum
s_add_u32 s1, s1, s3 // (op 820) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 820) 

s_mov_b32 s8, s38 // (op 820) reg expression
s_mov_b32 s9, s39 // (op 820) reg expression
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

v_lshrrev_b32 v0, 6, v1 // (op 1065) 
v_and_b32 v9, 63, v1 // (op 1065) 
v_lshrrev_b32 v11, 4, v9 // (op 1065) 
v_lshlrev_b32 v3, 2, v11 // (op 1065) 

v_lshl_add_u32 v11, v0, 4, v3 // (op 1065) 


v_lshl_add_u32 v0, s16, 4, v11 // (op 1065) 

v_lshrrev_b32 v11, 2, v9 // (op 1065) 

v_and_b32 v9, 3, v11 // (op 1065) 

v_and_b32 v11, 3, v1 // (op 1065) 
v_lshl_add_u32 v3, v9, 2, v11 // (op 1065) 


v_lshl_add_u32 v9, s17, 4, v3 // (op 1065) 

v_mul_lo_u32 v11, v9, s30 // (op 1065) 

v_add_lshl_u32 v9, v0, v11, 1 // (op 1065) 


s_mul_i32 s1, s30, s35 // (op 1067) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 1067) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 1067) least significant
s_mul_hi_u32 s3, s30, s34 // (op 1067) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1067) most significant: sum
s_add_u32 s1, s1, s3 // (op 1067) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1067) 

s_mov_b32 s4, s32 // (op 1067) reg expression
s_mov_b32 s5, s33 // (op 1067) reg expression
s_mov_b32 s6, s2 // (op 1067) reg expression
s_mov_b32 s7, 131072 // (op 1067) call()


v_pack_b32_f16 v6, v12, v13 // (op 169) 
v_pack_b32_f16 v7, v14, v15 // (op 169) 


buffer_store_dwordx2 v[6:7], v9, s[4:7], 0 offen // (op 169) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157:



s_endpgm  // End of RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
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
    DestDataType: h
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
  - .name: RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_TT_BF8_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 58
    .vgpr_count: 95
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

