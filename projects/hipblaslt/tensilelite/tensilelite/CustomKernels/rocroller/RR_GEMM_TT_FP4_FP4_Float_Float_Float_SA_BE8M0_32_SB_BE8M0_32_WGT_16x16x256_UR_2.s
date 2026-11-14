// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 849) 
v_and_b32 v3, 15, v1 // (op 849) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 849) 


v_and_b32 v0, 63, v1 // (op 849) 
v_lshrrev_b32 v3, 4, v0 // (op 849) 

v_lshlrev_b32 v0, 5, v3 // (op 849) 

v_lshl_add_u32 v3, v4, 8, v0 // (op 849) 


v_lshlrev_b32 v0, 2, v3 // (op 849) 

v_lshrrev_b32 v3, 3, v0 // (op 849) 

v_and_b32 v0, 63, v1 // (op 902) 
v_and_b32 v4, 48, v1 // (op 902) 
v_sub_u32 v5, v0, v4 // (op 902) 

v_lshrrev_b32 v4, 4, v0 // (op 902) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 902) 


v_lshlrev_b32 v4, 6, v0 // (op 902) 

v_lshrrev_b32 v0, 3, v4 // (op 902) 

v_lshrrev_b32 v4, 6, v1 // (op 955) 
v_and_b32 v5, 15, v1 // (op 955) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 955) 


v_and_b32 v4, 63, v1 // (op 955) 
v_lshrrev_b32 v5, 4, v4 // (op 955) 

v_lshl_add_u32 v4, v6, 3, v5 // (op 955) 


v_and_b32 v5, 48, v1 // (op 1008) 
v_and_b32 v6, 15, v1 // (op 1008) 
v_add_u32 v7, v5, v6 // (op 1008) 


v_lshrrev_b32 v5, 6, v1 // (op 1066) 
v_and_b32 v6, 15, v1 // (op 1066) 
v_lshl_add_u32 v8, v5, 4, v6 // (op 1066) 


v_and_b32 v5, 63, v1 // (op 1066) 
v_lshrrev_b32 v6, 4, v5 // (op 1066) 

v_lshlrev_b32 v5, 5, v6 // (op 1066) 

v_lshl_add_u32 v6, v8, 8, v5 // (op 1066) 


v_lshlrev_b32 v5, 2, v6 // (op 1066) 

v_lshrrev_b32 v6, 3, v5 // (op 1066) 

v_and_b32 v5, 63, v1 // (op 1090) 
v_and_b32 v8, 48, v1 // (op 1090) 
v_sub_u32 v9, v5, v8 // (op 1090) 

v_lshrrev_b32 v8, 4, v5 // (op 1090) 

v_lshl_add_u32 v5, v8, 5, v9 // (op 1090) 


v_lshlrev_b32 v8, 6, v5 // (op 1090) 

v_lshrrev_b32 v5, 3, v8 // (op 1090) 

v_lshrrev_b32 v8, 6, v1 // (op 1114) 
v_and_b32 v9, 15, v1 // (op 1114) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 1114) 


v_and_b32 v8, 63, v1 // (op 1114) 
v_lshrrev_b32 v9, 4, v8 // (op 1114) 

v_lshl_add_u32 v8, v10, 3, v9 // (op 1114) 


v_and_b32 v9, 48, v1 // (op 1138) 
v_and_b32 v10, 15, v1 // (op 1138) 
v_add_u32 v11, v9, v10 // (op 1138) 


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_lshrrev_b32 v12, 3, v1 // (op 559) 
v_lshlrev_b32 v13, 1, v12 // (op 559) 

v_lshl_add_u32 v12, s16, 4, v13 // (op 559) 

v_and_b32 v13, -8, v1 // (op 559) 
v_sub_u32 v14, v1, v13 // (op 559) 

v_lshlrev_b32 v13, 5, v14 // (op 559) 

v_mov_b32 v15, 0 // (op 559) convert
v_mov_b32 v14, v13 // (op 559) convert

v_mul_lo_u32 v17, v12, s3 // (op 559) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 559) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 559) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 559) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 559) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 559) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 559) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 559) 

v_mov_b32 v12, v14 // (op 559) convert

s_mul_i32 s1, s2, s7 // (op 561) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 561) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 561) least significant
s_mul_hi_u32 s15, s2, s6 // (op 561) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 561) most significant: sum
s_add_u32 s1, s1, s15 // (op 561) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 561) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 561) 

s_mov_b32 s40, s4 // (op 561) reg expression
s_mov_b32 s41, s5 // (op 561) reg expression
s_mov_b32 s42, s0 // (op 561) reg expression
s_mov_b32 s43, 131072 // (op 561) call()

v_lshrrev_b32 v13, 3, v1 // (op 578) 
v_lshlrev_b32 v14, 1, v13 // (op 578) 

v_lshl_add_u32 v13, s16, 4, v14 // (op 578) 

v_and_b32 v14, -8, v1 // (op 578) 
v_sub_u32 v15, v1, v14 // (op 578) 

v_lshlrev_b32 v14, 5, v15 // (op 578) 

v_mov_b32 v17, 0 // (op 578) convert
v_mov_b32 v16, v14 // (op 578) convert

v_mul_lo_u32 v15, v13, s3 // (op 578) most significant: low of low * high
v_mul_hi_u32 v18, v13, s2 // (op 578) most significant: high of low * low
v_mul_lo_u32 v14, v13, s2 // (op 578) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 578) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v16 // (op 578) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 578) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 578) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 578) 

v_mov_b32 v13, v14 // (op 578) convert

s_mul_i32 s1, s2, s7 // (op 580) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 580) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 580) least significant
s_mul_hi_u32 s15, s2, s6 // (op 580) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 580) most significant: sum
s_add_u32 s1, s1, s15 // (op 580) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 580) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 580) 

s_mov_b32 s40, s4 // (op 580) reg expression
s_mov_b32 s41, s5 // (op 580) reg expression
s_mov_b32 s42, s0 // (op 580) reg expression
s_mov_b32 s43, 131072 // (op 580) call()

s_lshl_b32 s0, s17, 4 // (op 623) 
v_lshlrev_b32 v14, 2, v1 // (op 623) 
s_mov_b32 s15, 0 // (op 623) convert
s_mov_b32 s14, s0 // (op 623) convert

v_mul_lo_u32 v17, v14, s9 // (op 623) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 623) most significant: high of low * low
v_mul_lo_u32 v16, v14, s8 // (op 623) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 623) most significant: sum

v_mov_b32 v15, s14 // (op 623) 
v_mov_b32 v18, s15 // (op 623) 
v_add_co_u32 v16, s[0:1], v16, v15 // (op 623) least significant half
v_addc_co_u32 v17, s[0:1], v17, v18, s[0:1] // (op 623) most significant half





v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 623) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 623) 

v_mov_b32 v14, v16 // (op 623) convert

s_mul_i32 s1, s8, s13 // (op 625) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 625) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 625) least significant
s_mul_hi_u32 s15, s8, s12 // (op 625) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 625) most significant: sum
s_add_u32 s1, s1, s15 // (op 625) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 625) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 625) 

s_mov_b32 s44, s10 // (op 625) reg expression
s_mov_b32 s45, s11 // (op 625) reg expression
s_mov_b32 s46, s0 // (op 625) reg expression
s_mov_b32 s47, 131072 // (op 625) call()

s_lshl_b32 s0, s17, 4 // (op 642) 
v_lshlrev_b32 v15, 2, v1 // (op 642) 
s_mov_b32 s15, 0 // (op 642) convert
s_mov_b32 s14, s0 // (op 642) convert

v_mul_lo_u32 v17, v15, s9 // (op 642) most significant: low of low * high
v_mul_hi_u32 v18, v15, s8 // (op 642) most significant: high of low * low
v_mul_lo_u32 v16, v15, s8 // (op 642) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 642) most significant: sum

v_mov_b32 v18, s14 // (op 642) 
v_mov_b32 v19, s15 // (op 642) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 642) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 642) most significant half





v_lshlrev_b64 v[18:19], 2, v[16:17] // (op 642) 

v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 642) 

v_mov_b32 v15, v16 // (op 642) convert

s_mul_i32 s1, s8, s13 // (op 644) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 644) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 644) least significant
s_mul_hi_u32 s15, s8, s12 // (op 644) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 644) most significant: sum
s_add_u32 s1, s1, s15 // (op 644) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 644) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 644) 

s_mov_b32 s44, s10 // (op 644) reg expression
s_mov_b32 s45, s11 // (op 644) reg expression
s_mov_b32 s46, s0 // (op 644) reg expression
s_mov_b32 s47, 131072 // (op 644) call()

v_lshrrev_b32 v16, 3, v1 // (op 687) 
v_lshlrev_b32 v17, 1, v16 // (op 687) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 687) 

v_and_b32 v17, -8, v1 // (op 687) 
v_sub_u32 v18, v1, v17 // (op 687) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v17, v16, s20 // (op 687) 
v_add_u32 v17, v17, v18 // (op 687) 


s_mul_i32 s1, s20, s7 // (op 689) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 689) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 689) least significant
s_mul_hi_u32 s15, s20, s6 // (op 689) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 689) most significant: sum
s_add_u32 s1, s1, s15 // (op 689) most significant: sum


s_mov_b32 s48, s22 // (op 689) reg expression
s_mov_b32 s49, s23 // (op 689) reg expression
s_mov_b32 s50, s0 // (op 689) reg expression
s_mov_b32 s51, 131072 // (op 689) call()

v_lshrrev_b32 v16, 3, v1 // (op 706) 
v_lshlrev_b32 v18, 1, v16 // (op 706) 

v_lshl_add_u32 v16, s16, 4, v18 // (op 706) 

v_and_b32 v18, -8, v1 // (op 706) 
v_sub_u32 v19, v1, v18 // (op 706) 

v_mul_lo_u32 v18, v16, s20 // (op 706) 
v_add_u32 v18, v18, v19 // (op 706) 


s_mul_i32 s1, s20, s7 // (op 708) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 708) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 708) least significant
s_mul_hi_u32 s15, s20, s6 // (op 708) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 708) most significant: sum
s_add_u32 s1, s1, s15 // (op 708) most significant: sum


s_mov_b32 s48, s22 // (op 708) reg expression
s_mov_b32 s49, s23 // (op 708) reg expression
s_mov_b32 s50, s0 // (op 708) reg expression
s_mov_b32 s51, 131072 // (op 708) call()

v_and_b32 v16, -16, v1 // (op 751) 
v_sub_u32 v19, v1, v16 // (op 751) 

v_lshl_add_u32 v16, s17, 4, v19 // (op 751) 

v_lshrrev_b32 v19, 4, v1 // (op 751) 
v_lshlrev_b32 v20, 1, v19 // (op 751) 

v_mul_lo_u32 v19, v20, s24 // (op 751) 
v_add_u32 v19, v19, v16 // (op 751) 


s_mul_i32 s1, s24, s29 // (op 753) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 753) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 753) least significant
s_mul_hi_u32 s15, s24, s28 // (op 753) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 753) most significant: sum
s_add_u32 s1, s1, s15 // (op 753) most significant: sum


s_mov_b32 s52, s26 // (op 753) reg expression
s_mov_b32 s53, s27 // (op 753) reg expression
s_mov_b32 s54, s0 // (op 753) reg expression
s_mov_b32 s55, 131072 // (op 753) call()

v_and_b32 v16, -16, v1 // (op 770) 
v_sub_u32 v20, v1, v16 // (op 770) 

v_lshl_add_u32 v16, s17, 4, v20 // (op 770) 

v_lshrrev_b32 v20, 4, v1 // (op 770) 
v_lshlrev_b32 v21, 1, v20 // (op 770) 

v_mul_lo_u32 v20, v21, s24 // (op 770) 
v_add_u32 v20, v20, v16 // (op 770) 


s_mul_i32 s1, s24, s29 // (op 772) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 772) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 772) least significant
s_mul_hi_u32 s15, s24, s28 // (op 772) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 772) most significant: sum
s_add_u32 s1, s1, s15 // (op 772) most significant: sum


s_mov_b32 s52, s26 // (op 772) reg expression
s_mov_b32 s53, s27 // (op 772) reg expression
s_mov_b32 s54, s0 // (op 772) reg expression
s_mov_b32 s55, 131072 // (op 772) call()


s_lshl_b64 s[0:1], s[8:9], 2 // (op 424) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 424) 


v_mov_b32 v16, v14 // (op 424) reg expression
buffer_load_dwordx2 v[24:25], v16, s[44:47], 0 offen // (op 424) Load value
v_add_u32 v16, v16, v22 // (op 424) 
buffer_load_dwordx2 v[26:27], v16, s[44:47], 0 offen // (op 424) Load value
v_add_u32 v16, v16, v22 // (op 424) 
buffer_load_dwordx2 v[28:29], v16, s[44:47], 0 offen // (op 424) Load value
v_add_u32 v16, v16, v22 // (op 424) 
buffer_load_dwordx2 v[30:31], v16, s[44:47], 0 offen // (op 424) Load value



v_mov_b64 v[22:23], s[24:25] // (op 427) convert

v_mov_b32 v16, v19 // (op 427) reg expression
buffer_load_ubyte v32, v16, s[52:55], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v16, v16, v22 // (op 427) 
buffer_load_ubyte v33, v16, s[52:55], 0 offen offset: 0 // (op 427) Load value



s_lshl_b64 s[0:1], s[2:3], 2 // (op 430) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 430) 


v_mov_b32 v16, v12 // (op 430) 
buffer_load_dwordx4 v[34:37], v16, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v16, v16, v22 // (op 430) 
buffer_load_dwordx4 v[38:41], v16, s[40:43], 0 offen // (op 430) Load value



v_mov_b64 v[22:23], s[20:21] // (op 433) convert

v_mov_b32 v16, v17 // (op 433) 
buffer_load_ubyte v42, v16, s[48:51], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v16, v16, v22 // (op 433) 
buffer_load_ubyte v43, v16, s[48:51], 0 offen offset: 0 // (op 433) Load value



s_lshl_b64 s[0:1], s[8:9], 2 // (op 436) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 436) 


s_lshl_b64 s[0:1], s[8:9], 8 // (op 436) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 436) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 436) 

v_add_u32 v16, v14, s0 // (op 436) 

buffer_load_dwordx2 v[44:45], v16, s[44:47], 0 offen // (op 436) Load value
v_add_u32 v16, v16, v22 // (op 436) 
buffer_load_dwordx2 v[46:47], v16, s[44:47], 0 offen // (op 436) Load value
v_add_u32 v16, v16, v22 // (op 436) 
buffer_load_dwordx2 v[48:49], v16, s[44:47], 0 offen // (op 436) Load value
v_add_u32 v16, v16, v22 // (op 436) 
buffer_load_dwordx2 v[50:51], v16, s[44:47], 0 offen // (op 436) Load value




v_mov_b64 v[22:23], s[24:25] // (op 439) convert

s_lshl_b32 s0, s24, 3 // (op 439) 
v_add_u32 v14, v19, s0 // (op 439) 

buffer_load_ubyte v52, v14, s[52:55], 0 offen offset: 0 // (op 439) Load value
v_add_u32 v14, v14, v22 // (op 439) 
buffer_load_ubyte v53, v14, s[52:55], 0 offen offset: 0 // (op 439) Load value




s_lshl_b64 s[0:1], s[2:3], 2 // (op 442) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 442) 


v_mov_b32 v14, v12 // (op 442) 
buffer_load_dwordx4 v[54:57], v14, s[40:43], 0 offen offset:128 // (op 442) Load value
v_add_u32 v14, v14, v22 // (op 442) 
buffer_load_dwordx4 v[58:61], v14, s[40:43], 0 offen offset:128 // (op 442) Load value




v_mov_b64 v[22:23], s[20:21] // (op 445) convert

v_mov_b32 v12, v17 // (op 445) 
buffer_load_ubyte v62, v12, s[48:51], 0 offen offset:8 // (op 445) Load value
v_add_u32 v12, v12, v22 // (op 445) 
buffer_load_ubyte v63, v12, s[48:51], 0 offen offset:8 // (op 445) Load value



v_lshlrev_b32 v12, 8, v1 // (op 890) 
v_lshrrev_b32 v14, 3, v12 // (op 890) 




s_waitcnt vmcnt(19)
ds_write_b64 v14, v[24:25] // (op 447) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v14, v[26:27] offset:8 // (op 447) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v14, v[28:29] offset:16 // (op 447) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v14, v[30:31] offset:24 // (op 447) Store local data 

v_and_b32 v12, -16, v1 // (op 996) 
v_sub_u32 v14, v1, v12 // (op 996) 

v_lshrrev_b32 v12, 4, v1 // (op 996) 
v_lshl_add_u32 v19, v12, 5, v14 // (op 996) 





s_waitcnt vmcnt(15)
ds_write_b8 v19, v32 offset:2048 // (op 448) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v19, v33 offset:2064 // (op 448) Store local data 

v_and_b32 v12, -8, v1 // (op 837) 
v_sub_u32 v14, v1, v12 // (op 837) 

v_lshlrev_b32 v12, 5, v14 // (op 837) 

v_lshrrev_b32 v14, 3, v1 // (op 837) 
v_lshl_add_u32 v19, v14, 9, v12 // (op 837) 


v_lshlrev_b32 v12, 2, v19 // (op 837) 

v_lshrrev_b32 v14, 3, v12 // (op 837) 




s_waitcnt vmcnt(13)
ds_write_b128 v14, v[34:37] offset:2176 // (op 449) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v14, v[38:41] offset:2304 // (op 449) Store local data 

v_and_b32 v12, -8, v1 // (op 943) 
v_sub_u32 v14, v1, v12 // (op 943) 

v_lshrrev_b32 v12, 3, v1 // (op 943) 
v_lshl_add_u32 v19, v12, 4, v14 // (op 943) 





s_waitcnt vmcnt(11)
ds_write_b8 v19, v42 offset:4224 // (op 450) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v19, v43 offset:4232 // (op 450) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1194) 



ds_read_b64_tr_b4 v[64:65], v0 // (op 467) Transpose load local data 
ds_read_b64_tr_b4 v[66:67], v0 offset:128 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1235) 


ds_read_u8 v12, v7 offset:2048 // (op 473) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1175) 



ds_read_b128 v[68:71], v3 offset:2176 // (op 464) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1215) 



ds_read_u8 v14, v4 offset:4224 // (op 470) Load local data 
v_and_b32 v19, -8, v1 // (op 808) 
v_sub_u32 v16, v1, v19 // (op 808) 

v_lshlrev_b32 v17, 5, v16 // (op 808) 

v_lshrrev_b32 v16, 3, v1 // (op 808) 
v_lshl_add_u32 v19, v16, 9, v17 // (op 808) 


v_lshlrev_b32 v16, 2, v19 // (op 808) 

v_lshrrev_b32 v17, 3, v16 // (op 808) 

v_lshlrev_b32 v16, 8, v1 // (op 861) 
v_lshrrev_b32 v19, 3, v16 // (op 861) 

v_and_b32 v16, -8, v1 // (op 914) 
v_sub_u32 v21, v1, v16 // (op 914) 

v_lshrrev_b32 v16, 3, v1 // (op 914) 
v_lshl_add_u32 v22, v16, 4, v21 // (op 914) 


v_and_b32 v16, -16, v1 // (op 967) 
v_sub_u32 v21, v1, v16 // (op 967) 

v_lshrrev_b32 v16, 4, v1 // (op 967) 
v_lshl_add_u32 v23, v16, 5, v21 // (op 967) 


v_and_b32 v16, -8, v1 // (op 1054) 
v_sub_u32 v21, v1, v16 // (op 1054) 

v_lshlrev_b32 v16, 5, v21 // (op 1054) 

v_lshrrev_b32 v21, 3, v1 // (op 1054) 
v_lshl_add_u32 v72, v21, 9, v16 // (op 1054) 


v_lshlrev_b32 v16, 2, v72 // (op 1054) 

v_lshrrev_b32 v21, 3, v16 // (op 1054) 

v_lshlrev_b32 v16, 8, v1 // (op 1078) 
v_lshrrev_b32 v72, 3, v16 // (op 1078) 

v_and_b32 v16, -8, v1 // (op 1102) 
v_sub_u32 v73, v1, v16 // (op 1102) 

v_lshrrev_b32 v16, 3, v1 // (op 1102) 
v_lshl_add_u32 v74, v16, 4, v73 // (op 1102) 


v_and_b32 v16, -16, v1 // (op 1126) 
v_sub_u32 v73, v1, v16 // (op 1126) 

v_lshrrev_b32 v16, 4, v1 // (op 1126) 
v_lshl_add_u32 v75, v16, 5, v73 // (op 1126) 


s_mov_b32 s0, 0 // (op 59) call()
s_ashr_i64 s[14:15], s[12:13], 8 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58:

s_add_u32 s0, s0, 2 // (op 60) 
s_lshl_b64 s[14:15], s[8:9], 2 // (op 10) 
v_lshrrev_b64 v[76:77], 3, s[14:15] // (op 10) 


v_mov_b32 v16, v15 // (op 10) 
s_lshl_b64 s[14:15], s[8:9], 8 // (op 10) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 10) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s1, s14, 1 // (op 10) 

v_add_u32 v73, v16, s1 // (op 10) 

buffer_load_dwordx2 v[24:25], v73, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v73, v73, v76 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:71], v[64:67], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 88) 
buffer_load_dwordx2 v[26:27], v73, s[44:47], 0 offen // (op 10) Load value
s_barrier  // (op 1184) 
v_add_u32 v73, v73, v76 // (op 10) 



ds_read_b128 v[68:71], v3 offset:2240 // (op 91) Load local data 
buffer_load_dwordx2 v[28:29], v73, s[44:47], 0 offen // (op 10) Load value
s_barrier  // (op 1204) 
v_add_u32 v73, v73, v76 // (op 10) 



ds_read_b64_tr_b4 v[64:65], v0 offset:1024 // (op 95) Transpose load local data 
ds_read_b64_tr_b4 v[66:67], v0 offset:1152 // (op 95) Transpose load local data 
buffer_load_dwordx2 v[30:31], v73, s[44:47], 0 offen // (op 10) Load value



s_barrier  // (op 1225) 
v_mov_b64 v[76:77], s[24:25] // (op 25) convert




ds_read_u8 v14, v4 offset:4228 // (op 99) Load local data 
v_mov_b32 v73, v20 // (op 25) 
s_barrier  // (op 1245) 
s_lshl_b32 s1, s24, 4 // (op 25) 


ds_read_u8 v12, v7 offset:2112 // (op 103) Load local data 
v_add_u32 v78, v73, s1 // (op 25) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[68:71], v[64:67], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 105) 
buffer_load_ubyte v32, v78, s[52:55], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v78, v78, v76 // (op 25) 
buffer_load_ubyte v33, v78, s[52:55], 0 offen offset: 0 // (op 25) Load value



s_lshl_b64 s[14:15], s[2:3], 2 // (op 4) 
v_lshrrev_b64 v[76:77], 3, s[14:15] // (op 4) 


v_mov_b32 v78, v13 // (op 4) 
v_mov_b32 v79, v78 // (op 4) 
buffer_load_dwordx4 v[34:37], v79, s[40:43], 0 offen offset:256 // (op 4) Load value
v_add_u32 v79, v79, v76 // (op 4) 
buffer_load_dwordx4 v[38:41], v79, s[40:43], 0 offen offset:256 // (op 4) Load value



v_mov_b64 v[76:77], s[20:21] // (op 17) convert

v_mov_b32 v79, v18 // (op 17) 
v_mov_b32 v80, v79 // (op 17) 
buffer_load_ubyte v42, v80, s[48:51], 0 offen offset:16 // (op 17) Load value
v_add_u32 v80, v80, v76 // (op 17) 
buffer_load_ubyte v43, v80, s[48:51], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 1265) 



s_waitcnt vmcnt(19)
ds_write_b64 v72, v[44:45] offset:4352 // (op 302) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v72, v[46:47] offset:4360 // (op 302) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v72, v[48:49] offset:4368 // (op 302) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v72, v[50:51] offset:4376 // (op 302) Store local data 
s_barrier  // (op 1277) 



s_waitcnt vmcnt(15)
ds_write_b8 v75, v52 offset:6400 // (op 304) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v75, v53 offset:6416 // (op 304) Store local data 
s_barrier  // (op 1258) 



s_waitcnt vmcnt(13)
ds_write_b128 v21, v[54:57] offset:6528 // (op 283) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v21, v[58:61] offset:6656 // (op 283) Store local data 
s_barrier  // (op 1271) 



s_waitcnt vmcnt(11)
ds_write_b8 v74, v62 offset:8576 // (op 303) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v74, v63 offset:8584 // (op 303) Store local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1261) 



ds_read_b64_tr_b4 v[80:81], v5 offset:4352 // (op 287) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v5 offset:4480 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1274) 


ds_read_u8 v76, v11 offset:6400 // (op 291) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1255) 



ds_read_b128 v[84:87], v6 offset:6528 // (op 285) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1268) 



ds_read_u8 v77, v8 offset:8576 // (op 289) Load local data 

s_lshl_b64 s[14:15], s[8:9], 2 // (op 305) 
v_lshrrev_b64 v[88:89], 3, s[14:15] // (op 305) 


s_lshl_b64 s[14:15], s[8:9], 8 // (op 305) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 305) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 305) 

s_mul_i32 s1, 3, s14 // (op 305) 

v_add_u32 v90, v16, s1 // (op 305) 

buffer_load_dwordx2 v[44:45], v90, s[44:47], 0 offen // (op 305) Load value
v_add_u32 v90, v90, v88 // (op 305) 
buffer_load_dwordx2 v[46:47], v90, s[44:47], 0 offen // (op 305) Load value
v_add_u32 v90, v90, v88 // (op 305) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[80:83], a[0:3], v77, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 292) 
buffer_load_dwordx2 v[48:49], v90, s[44:47], 0 offen // (op 305) Load value




v_add_u32 v90, v90, v88 // (op 305) 



ds_read_b128 v[80:83], v6 offset:6592 // (op 294) Load local data 
buffer_load_dwordx2 v[50:51], v90, s[44:47], 0 offen // (op 305) Load value







s_barrier  // (op 1336) 
v_mov_b64 v[76:77], s[24:25] // (op 307) convert

s_lshl_b32 s1, s24, 3 // (op 307) 
s_mul_i32 s14, 3, s1 // (op 307) 

v_add_u32 v16, v73, s14 // (op 307) 

buffer_load_ubyte v52, v16, s[52:55], 0 offen offset: 0 // (op 307) Load value
v_add_u32 v16, v16, v76 // (op 307) 
s_lshl_b64 s[14:15], s[8:9], 8 // (op 634) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 634) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 634) 

s_lshl_b32 s1, s14, 1 // (op 634) 

v_add_u32 v15, v15, s1 // (op 634) 

ds_read_b64_tr_b4 v[84:85], v5 offset:5376 // (op 296) Transpose load local data 
ds_read_b64_tr_b4 v[86:87], v5 offset:5504 // (op 296) Transpose load local data 
buffer_load_ubyte v53, v16, s[52:55], 0 offen offset: 0 // (op 307) Load value







s_barrier  // (op 1340) 
ds_read_u8 v16, v8 offset:8580 // (op 298) Load local data 
s_lshl_b32 s1, s24, 4 // (op 762) 
v_add_u32 v20, v20, s1 // (op 762) 



s_barrier  // (op 1344) 
s_lshl_b64 s[14:15], s[2:3], 2 // (op 282) 
v_lshrrev_b64 v[76:77], 3, s[14:15] // (op 282) 


v_mov_b32 v73, v78 // (op 282) 
ds_read_u8 v88, v11 offset:6464 // (op 300) Load local data 
buffer_load_dwordx4 v[54:57], v73, s[40:43], 0 offen offset:384 // (op 282) Load value
v_add_u32 v73, v73, v76 // (op 282) 
s_barrier  // (op 1348) 
buffer_load_dwordx4 v[58:61], v73, s[40:43], 0 offen offset:384 // (op 282) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:83], v[84:87], a[0:3], v16, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 301) 






v_add_u32 v13, 256, v13 // (op 570) 
v_mov_b64 v[76:77], s[20:21] // (op 306) convert

v_mov_b32 v16, v79 // (op 306) 
buffer_load_ubyte v62, v16, s[48:51], 0 offen offset:24 // (op 306) Load value
v_add_u32 v16, v16, v76 // (op 306) 
buffer_load_ubyte v63, v16, s[48:51], 0 offen offset:24 // (op 306) Load value



s_barrier  // (op 1197) 
v_add_u32 v18, 16, v18 // (op 698) 



s_waitcnt vmcnt(19)
ds_write_b64 v19, v[24:25] // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v19, v[26:27] offset:8 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v19, v[28:29] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v19, v[30:31] offset:24 // (op 13) Store local data 
s_barrier  // (op 1238) 



s_waitcnt vmcnt(15)
ds_write_b8 v23, v32 offset:2048 // (op 30) Store local data 
s_waitcnt vmcnt(14)
ds_write_b8 v23, v33 offset:2064 // (op 30) Store local data 
s_barrier  // (op 1178) 



s_waitcnt vmcnt(13)
ds_write_b128 v17, v[34:37] offset:2176 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v17, v[38:41] offset:2304 // (op 6) Store local data 
s_barrier  // (op 1218) 



s_waitcnt vmcnt(11)
ds_write_b8 v22, v42 offset:4224 // (op 21) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v22, v43 offset:4232 // (op 21) Store local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1207) 



ds_read_b64_tr_b4 v[64:65], v0 // (op 75) Transpose load local data 
ds_read_b64_tr_b4 v[66:67], v0 offset:128 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1248) 


ds_read_u8 v12, v7 offset:2048 // (op 85) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1187) 



ds_read_b128 v[68:71], v3 offset:2176 // (op 70) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1228) 



ds_read_u8 v14, v4 offset:4224 // (op 80) Load local data 
s_ashr_i64 s[14:15], s[12:13], 8 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58:

















v_lshrrev_b32 v13, 3, v1 // (op 533) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 533) 
v_lshlrev_b32 v15, 1, v13 // (op 533) 

s_lshr_b32 s14, s0, 1 // (op 533) 

v_lshl_add_u32 v13, s16, 4, v15 // (op 533) 

v_and_b32 v15, -8, v1 // (op 533) 
v_sub_u32 v16, v1, v15 // (op 533) 

v_lshlrev_b32 v15, 5, v16 // (op 533) 

v_lshl_add_u32 v16, s14, 9, v15 // (op 533) 


v_mov_b32 v43, 0 // (op 533) convert
v_mov_b32 v42, v16 // (op 533) convert

v_mul_lo_u32 v41, v13, s3 // (op 533) most significant: low of low * high
v_mul_hi_u32 v15, v13, s2 // (op 533) most significant: high of low * low
v_mul_lo_u32 v40, v13, s2 // (op 533) least significant: low of low * low
v_add_u32 v41, v41, v15 // (op 533) most significant: sum

v_add_co_u32 v40, s[0:1], v40, v42 // (op 533) least significant half
v_addc_co_u32 v41, s[0:1], v41, v43, s[0:1] // (op 533) most significant half



v_lshlrev_b64 v[42:43], 2, v[40:41] // (op 533) 

v_lshrrev_b64 v[40:41], 3, v[42:43] // (op 533) 

v_mov_b32 v13, v40 // (op 533) convert

s_mul_i32 s1, s2, s7 // (op 535) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 535) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 535) least significant
s_mul_hi_u32 s15, s2, s6 // (op 535) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 535) most significant: sum
s_add_u32 s1, s1, s15 // (op 535) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 535) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 535) 

s_mov_b32 s40, s4 // (op 535) reg expression
s_mov_b32 s41, s5 // (op 535) reg expression
s_mov_b32 s42, s0 // (op 535) reg expression
s_mov_b32 s43, 131072 // (op 535) call()

v_lshlrev_b32 v15, 2, v1 // (op 600) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 600) 
s_lshr_b32 s14, s0, 1 // (op 600) 

v_lshl_add_u32 v16, s14, 9, v15 // (op 600) 


s_lshl_b32 s0, s17, 4 // (op 600) 
s_mov_b32 s15, 0 // (op 600) convert
s_mov_b32 s14, s0 // (op 600) convert

v_mul_lo_u32 v43, v16, s9 // (op 600) most significant: low of low * high
v_mul_hi_u32 v15, v16, s8 // (op 600) most significant: high of low * low
v_mul_lo_u32 v42, v16, s8 // (op 600) least significant: low of low * low
v_add_u32 v43, v43, v15 // (op 600) most significant: sum

v_mov_b32 v15, s14 // (op 600) 
v_mov_b32 v17, s15 // (op 600) 
v_add_co_u32 v42, s[0:1], v42, v15 // (op 600) least significant half
v_addc_co_u32 v43, s[0:1], v43, v17, s[0:1] // (op 600) most significant half





v_lshlrev_b64 v[40:41], 2, v[42:43] // (op 600) 

v_lshrrev_b64 v[42:43], 3, v[40:41] // (op 600) 

v_mov_b32 v15, v42 // (op 600) convert

s_mul_i32 s1, s8, s13 // (op 602) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 602) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 602) least significant
s_mul_hi_u32 s15, s8, s12 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 602) most significant: sum
s_add_u32 s1, s1, s15 // (op 602) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 602) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 602) 

s_mov_b32 s44, s10 // (op 602) reg expression
s_mov_b32 s45, s11 // (op 602) reg expression
s_mov_b32 s46, s0 // (op 602) reg expression
s_mov_b32 s47, 131072 // (op 602) call()

v_lshrrev_b32 v16, 3, v1 // (op 664) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 664) 
v_lshlrev_b32 v17, 1, v16 // (op 664) 

s_lshr_b32 s14, s0, 1 // (op 664) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 664) 

v_and_b32 v17, -8, v1 // (op 664) 
v_sub_u32 v18, v1, v17 // (op 664) 

v_lshl_add_u32 v17, s14, 4, v18 // (op 664) 


v_mul_lo_u32 v18, v16, s20 // (op 664) 
v_add_u32 v18, v18, v17 // (op 664) 


s_mul_i32 s1, s20, s7 // (op 666) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 666) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 666) least significant
s_mul_hi_u32 s15, s20, s6 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 666) most significant: sum
s_add_u32 s1, s1, s15 // (op 666) most significant: sum


s_mov_b32 s48, s22 // (op 666) reg expression
s_mov_b32 s49, s23 // (op 666) reg expression
s_mov_b32 s50, s0 // (op 666) reg expression
s_mov_b32 s51, 131072 // (op 666) call()

v_lshrrev_b32 v16, 4, v1 // (op 728) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 728) 
v_lshlrev_b32 v17, 1, v16 // (op 728) 

s_lshr_b32 s14, s0, 1 // (op 728) 

v_and_b32 v16, -16, v1 // (op 728) 
v_lshl_add_u32 v19, s14, 4, v17 // (op 728) 


v_sub_u32 v17, v1, v16 // (op 728) 

v_lshl_add_u32 v16, s17, 4, v17 // (op 728) 

v_mul_lo_u32 v17, v19, s24 // (op 728) 
v_add_u32 v17, v17, v16 // (op 728) 


s_mul_i32 s1, s24, s29 // (op 730) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 730) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 730) least significant
s_mul_hi_u32 s15, s24, s28 // (op 730) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 730) most significant: sum
s_add_u32 s1, s1, s15 // (op 730) most significant: sum


s_mov_b32 s52, s26 // (op 730) reg expression
s_mov_b32 s53, s27 // (op 730) reg expression
s_mov_b32 s54, s0 // (op 730) reg expression
s_mov_b32 s55, 131072 // (op 730) call()

v_and_b32 v16, -8, v1 // (op 821) 
v_sub_u32 v19, v1, v16 // (op 821) 

v_lshlrev_b32 v16, 5, v19 // (op 821) 

v_lshrrev_b32 v19, 3, v1 // (op 821) 
v_lshl_add_u32 v20, v19, 9, v16 // (op 821) 


v_lshlrev_b32 v16, 2, v20 // (op 821) 

v_lshrrev_b32 v19, 3, v16 // (op 821) 

v_lshlrev_b32 v16, 8, v1 // (op 874) 
v_lshrrev_b32 v20, 3, v16 // (op 874) 

v_and_b32 v16, -8, v1 // (op 927) 
v_sub_u32 v21, v1, v16 // (op 927) 

v_lshrrev_b32 v16, 3, v1 // (op 927) 
v_lshl_add_u32 v22, v16, 4, v21 // (op 927) 


v_and_b32 v16, -16, v1 // (op 980) 
v_sub_u32 v21, v1, v16 // (op 980) 

v_lshrrev_b32 v16, 4, v1 // (op 980) 
v_lshl_add_u32 v23, v16, 5, v21 // (op 980) 


s_ashr_i64 s[0:1], s[12:13], 8 // (op 194) 
s_and_b32 s14, s0, 4294967294 // (op 194) 

s_ashr_i64 s[0:1], s[12:13], 8 // (op 193) 
s_mov_b32 s15, s0 // (op 193) convert

s_cmp_lt_u32 s14, s15 // (op 193) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193:

s_add_u32 s14, s14, 1 // (op 195) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 198) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 198) 


v_mov_b32 v12, v13 // (op 198) 
v_mov_b32 v14, v12 // (op 198) 
buffer_load_dwordx4 v[26:29], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v24 // (op 198) 
buffer_load_dwordx4 v[30:33], v14, s[40:43], 0 offen // (op 198) Load value



v_add_u32 v13, 128, v13 // (op 523) 

s_barrier  // (op 1181) 
s_lshl_b64 s[0:1], s[8:9], 2 // (op 221) 



v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 221) 


v_mov_b32 v12, v15 // (op 221) 
v_mov_b32 v14, v12 // (op 221) reg expression
buffer_load_dwordx2 v[34:35], v14, s[44:47], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v24 // (op 221) 
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[26:29] offset:2176 // (op 199) Store local data 
buffer_load_dwordx2 v[36:37], v14, s[44:47], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v24 // (op 221) 
buffer_load_dwordx2 v[38:39], v14, s[44:47], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v24 // (op 221) 
buffer_load_dwordx2 v[40:41], v14, s[44:47], 0 offen // (op 221) Load value



s_lshl_b64 s[0:1], s[8:9], 8 // (op 592) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 592) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 592) 

v_add_u32 v15, v15, s0 // (op 592) 

s_waitcnt vmcnt(4)
ds_write_b128 v19, v[30:33] offset:2304 // (op 199) Store local data 


v_mov_b64 v[24:25], s[20:21] // (op 222) convert

s_barrier  // (op 1200) 
v_mov_b32 v12, v18 // (op 222) 



v_mov_b32 v14, v12 // (op 222) 
s_waitcnt vmcnt(3)
ds_write_b64 v20, v[34:35] // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v20, v[36:37] offset:8 // (op 218) Store local data 
buffer_load_ubyte v26, v14, s[48:51], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v14, v14, v24 // (op 222) 
buffer_load_ubyte v27, v14, s[48:51], 0 offen offset: 0 // (op 222) Load value



v_add_u32 v18, 8, v18 // (op 656) 
s_waitcnt vmcnt(3)
ds_write_b64 v20, v[38:39] offset:16 // (op 218) Store local data 

v_mov_b64 v[24:25], s[24:25] // (op 223) convert

v_mov_b32 v12, v17 // (op 223) 
v_mov_b32 v14, v12 // (op 223) reg expression
buffer_load_ubyte v28, v14, s[52:55], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v14, v14, v24 // (op 223) 
s_waitcnt vmcnt(3)
ds_write_b64 v20, v[40:41] offset:24 // (op 218) Store local data 
buffer_load_ubyte v29, v14, s[52:55], 0 offen offset: 0 // (op 223) Load value




s_lshl_b32 s15, s24, 3 // (op 720) 
s_barrier  // (op 1221) 
v_add_u32 v17, v17, s15 // (op 720) 




s_waitcnt lgkmcnt(4)
s_barrier  // (op 1190) 
s_waitcnt vmcnt(3)
ds_write_b8 v22, v26 offset:4224 // (op 219) Store local data 



ds_read_b128 v[30:33], v3 offset:2176 // (op 201) Load local data 
s_waitcnt vmcnt(2)
ds_write_b8 v22, v27 offset:4232 // (op 219) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1211) 



s_barrier  // (op 1241) 
ds_read_b64_tr_b4 v[24:25], v0 // (op 203) Transpose load local data 



ds_read_b64_tr_b4 v[26:27], v0 offset:128 // (op 203) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v28 offset:2048 // (op 220) Store local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1231) 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v29 offset:2064 // (op 220) Store local data 




ds_read_u8 v12, v4 offset:4224 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1251) 


ds_read_u8 v14, v7 offset:2048 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[24:27], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 208) 







ds_read_b128 v[24:27], v3 offset:2240 // (op 210) Load local data 



ds_read_b64_tr_b4 v[28:29], v0 offset:1024 // (op 212) Transpose load local data 
ds_read_b64_tr_b4 v[30:31], v0 offset:1152 // (op 212) Transpose load local data 



ds_read_u8 v12, v4 offset:4228 // (op 214) Load local data 


ds_read_u8 v14, v7 offset:2112 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[28:31], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 217) 




s_ashr_i64 s[0:1], s[12:13], 8 // (op 193) 
s_mov_b32 s15, s0 // (op 193) convert

s_cmp_lt_u32 s14, s15 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 if true)
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193:
























s_barrier  // (op 1328) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 1022) 
v_and_b32 v11, 63, v1 // (op 1022) 
v_lshrrev_b32 v3, 4, v11 // (op 1022) 
v_lshlrev_b32 v4, 2, v3 // (op 1022) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1022) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 1022) 

v_lshrrev_b32 v3, 2, v11 // (op 1022) 

v_and_b32 v11, 3, v3 // (op 1022) 

v_and_b32 v3, 3, v1 // (op 1022) 
v_lshl_add_u32 v4, v11, 2, v3 // (op 1022) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 1022) 

v_mul_lo_u32 v11, v3, s30 // (op 1022) 

v_add_lshl_u32 v3, v0, v11, 2 // (op 1022) 


s_mul_i32 s1, s30, s35 // (op 1024) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 1024) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 1024) least significant
s_mul_hi_u32 s3, s30, s34 // (op 1024) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1024) most significant: sum
s_add_u32 s1, s1, s3 // (op 1024) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 1024) 

s_mov_b32 s4, s32 // (op 1024) reg expression
s_mov_b32 s5, s33 // (op 1024) reg expression
s_mov_b32 s6, s2 // (op 1024) reg expression
s_mov_b32 s7, 131072 // (op 1024) call()




buffer_store_dwordx4 v[12:15], v3, s[4:7], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155:
v_lshrrev_b32 v0, 6, v1 // (op 791) 
v_and_b32 v3, 63, v1 // (op 791) 
v_lshrrev_b32 v4, 4, v3 // (op 791) 
v_lshlrev_b32 v5, 2, v4 // (op 791) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 791) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 791) 

v_lshrrev_b32 v4, 2, v3 // (op 791) 

v_and_b32 v3, 3, v4 // (op 791) 

v_and_b32 v4, 3, v1 // (op 791) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 791) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 791) 

v_mul_lo_u32 v4, v3, s36 // (op 791) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 791) 


s_mul_i32 s1, s36, s35 // (op 793) most significant: low of low * high
s_mul_i32 s2, s37, s34 // (op 793) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 793) least significant
s_mul_hi_u32 s3, s36, s34 // (op 793) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 793) most significant: sum
s_add_u32 s1, s1, s3 // (op 793) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 793) 

s_mov_b32 s8, s38 // (op 793) reg expression
s_mov_b32 s9, s39 // (op 793) reg expression
s_mov_b32 s10, s2 // (op 793) reg expression
s_mov_b32 s11, 131072 // (op 793) call()





buffer_load_dwordx4 v[4:7], v3, s[8:11], 0 offen // (op 167) Load value



v_accvgpr_read v12, a0 // (op 164) 
v_accvgpr_read v13, a1 // (op 164) 
v_accvgpr_read v14, a2 // (op 164) 
v_accvgpr_read v15, a3 // (op 164) 
v_mul_f32 v16, v10, v12 // (op 164) 
v_mul_f32 v17, v10, v13 // (op 164) 
v_mul_f32 v18, v10, v14 // (op 164) 
v_mul_f32 v19, v10, v15 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v9, v4 // (op 168) 
v_mul_f32 v13, v9, v5 // (op 168) 
v_mul_f32 v14, v9, v6 // (op 168) 
v_mul_f32 v15, v9, v7 // (op 168) 

v_add_f32 v4, v12, v16 // (op 165) 
v_add_f32 v5, v13, v17 // (op 165) 
v_add_f32 v6, v14, v18 // (op 165) 
v_add_f32 v7, v15, v19 // (op 165) 


v_lshrrev_b32 v0, 6, v1 // (op 1039) 
v_and_b32 v3, 63, v1 // (op 1039) 
v_lshrrev_b32 v8, 4, v3 // (op 1039) 
v_lshlrev_b32 v11, 2, v8 // (op 1039) 

v_lshl_add_u32 v8, v0, 4, v11 // (op 1039) 


v_lshl_add_u32 v0, s16, 4, v8 // (op 1039) 

v_lshrrev_b32 v8, 2, v3 // (op 1039) 

v_and_b32 v3, 3, v8 // (op 1039) 

v_and_b32 v8, 3, v1 // (op 1039) 
v_lshl_add_u32 v11, v3, 2, v8 // (op 1039) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 1039) 

v_mul_lo_u32 v8, v3, s30 // (op 1039) 

v_add_lshl_u32 v3, v0, v8, 2 // (op 1039) 


s_mul_i32 s1, s30, s35 // (op 1041) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 1041) least significant
s_mul_hi_u32 s3, s30, s34 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1041) most significant: sum
s_add_u32 s1, s1, s3 // (op 1041) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 1041) 

s_mov_b32 s4, s32 // (op 1041) reg expression
s_mov_b32 s5, s33 // (op 1041) reg expression
s_mov_b32 s6, s2 // (op 1041) reg expression
s_mov_b32 s7, 131072 // (op 1041) call()




buffer_store_dwordx4 v[4:7], v3, s[4:7], 0 offen // (op 166) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155:



s_endpgm  // End of RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8704 // lds bytes
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
    DataTypeB: F4
    DestDataType: s
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
  - .name: RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_TT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 8704
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 91
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

