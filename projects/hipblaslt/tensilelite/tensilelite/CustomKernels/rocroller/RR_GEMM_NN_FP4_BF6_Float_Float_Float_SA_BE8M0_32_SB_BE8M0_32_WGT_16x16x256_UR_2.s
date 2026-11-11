// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 849) 
v_and_b32 v3, 48, v1 // (op 849) 
v_sub_u32 v4, v0, v3 // (op 849) 

v_lshrrev_b32 v3, 4, v0 // (op 849) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 849) 


v_lshrrev_b32 v3, 6, v1 // (op 849) 
v_lshlrev_b32 v4, 4, v3 // (op 849) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 849) 


v_lshlrev_b32 v0, 2, v3 // (op 849) 

v_lshrrev_b32 v3, 3, v0 // (op 849) 

v_and_b32 v0, 15, v1 // (op 902) 
v_and_b32 v4, 63, v1 // (op 902) 
v_lshrrev_b32 v5, 4, v4 // (op 902) 

v_lshlrev_b32 v4, 5, v5 // (op 902) 

v_lshl_add_u32 v5, v0, 8, v4 // (op 902) 


v_mul_lo_u32 v0, v5, 6 // (op 902) 

v_lshrrev_b32 v4, 3, v0 // (op 902) 

v_lshrrev_b32 v0, 6, v1 // (op 955) 
v_and_b32 v5, 15, v1 // (op 955) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 955) 


v_and_b32 v0, 48, v1 // (op 955) 
v_add_u32 v5, v0, v6 // (op 955) 


v_and_b32 v0, 15, v1 // (op 1008) 
v_and_b32 v6, 63, v1 // (op 1008) 
v_lshrrev_b32 v7, 4, v6 // (op 1008) 

v_lshl_add_u32 v6, v0, 3, v7 // (op 1008) 


v_and_b32 v0, 63, v1 // (op 1066) 
v_and_b32 v7, 48, v1 // (op 1066) 
v_sub_u32 v8, v0, v7 // (op 1066) 

v_lshrrev_b32 v7, 4, v0 // (op 1066) 

v_lshl_add_u32 v0, v7, 5, v8 // (op 1066) 


v_lshrrev_b32 v7, 6, v1 // (op 1066) 
v_lshlrev_b32 v8, 4, v7 // (op 1066) 

v_lshl_add_u32 v7, v0, 4, v8 // (op 1066) 


v_lshlrev_b32 v0, 2, v7 // (op 1066) 

v_lshrrev_b32 v7, 3, v0 // (op 1066) 

v_and_b32 v0, 15, v1 // (op 1090) 
v_and_b32 v8, 63, v1 // (op 1090) 
v_lshrrev_b32 v9, 4, v8 // (op 1090) 

v_lshlrev_b32 v8, 5, v9 // (op 1090) 

v_lshl_add_u32 v9, v0, 8, v8 // (op 1090) 


v_mul_lo_u32 v0, v9, 6 // (op 1090) 

v_lshrrev_b32 v8, 3, v0 // (op 1090) 

v_lshrrev_b32 v0, 6, v1 // (op 1114) 
v_and_b32 v9, 15, v1 // (op 1114) 
v_lshl_add_u32 v10, v0, 4, v9 // (op 1114) 


v_and_b32 v0, 48, v1 // (op 1114) 
v_add_u32 v9, v0, v10 // (op 1114) 


v_and_b32 v0, 15, v1 // (op 1138) 
v_and_b32 v10, 63, v1 // (op 1138) 
v_lshrrev_b32 v11, 4, v10 // (op 1138) 

v_lshl_add_u32 v10, v0, 3, v11 // (op 1138) 


v_mov_b32 v0, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_lshl_b32 s0, s16, 4 // (op 559) 
v_lshlrev_b32 v12, 2, v1 // (op 559) 
s_mov_b32 s15, 0 // (op 559) convert
s_mov_b32 s14, s0 // (op 559) convert

v_mul_lo_u32 v15, v12, s3 // (op 559) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 559) most significant: high of low * low
v_mul_lo_u32 v14, v12, s2 // (op 559) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 559) most significant: sum


v_mov_b32 v12, s14 // (op 559) 
v_mov_b32 v13, s15 // (op 559) 
v_add_co_u32 v16, s[0:1], v12, v14 // (op 559) least significant half
v_addc_co_u32 v17, s[0:1], v13, v15, s[0:1] // (op 559) most significant half



v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 559) 


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 559) 

v_mov_b32 v14, v12 // (op 559) convert

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

s_lshl_b32 s0, s16, 4 // (op 578) 
v_lshlrev_b32 v12, 2, v1 // (op 578) 
s_mov_b32 s15, 0 // (op 578) convert
s_mov_b32 s14, s0 // (op 578) convert

v_mul_lo_u32 v17, v12, s3 // (op 578) most significant: low of low * high
v_mul_hi_u32 v13, v12, s2 // (op 578) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 578) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 578) most significant: sum


v_mov_b32 v15, s14 // (op 578) 
v_mov_b32 v12, s15 // (op 578) 
v_add_co_u32 v18, s[0:1], v15, v16 // (op 578) least significant half
v_addc_co_u32 v19, s[0:1], v12, v17, s[0:1] // (op 578) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 578) 


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 578) 

v_mov_b32 v15, v12 // (op 578) convert

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

v_and_b32 v12, -4, v1 // (op 623) 
v_sub_u32 v13, v1, v12 // (op 623) 

v_lshlrev_b32 v12, 6, v13 // (op 623) 

v_mov_b32 v17, 0 // (op 623) convert
v_mov_b32 v16, v12 // (op 623) convert

v_lshrrev_b32 v12, 2, v1 // (op 623) 
v_lshl_add_u32 v13, s17, 4, v12 // (op 623) 

v_mul_lo_u32 v19, v13, s9 // (op 623) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 623) most significant: high of low * low
v_mul_lo_u32 v18, v13, s8 // (op 623) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 623) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 623) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 623) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 623) most significant: low of high * low
v_mul_hi_u32 v18, v12, 6 // (op 623) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 623) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 623) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 623) 

v_mov_b32 v16, v12 // (op 623) convert

s_mul_i32 s1, s13, 6 // (op 625) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 625) least significant
s_mul_hi_u32 s14, s12, 6 // (op 625) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 625) most significant: sum

s_mul_i32 s15, s8, s1 // (op 625) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 625) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 625) least significant
s_mul_hi_u32 s19, s8, s0 // (op 625) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 625) most significant: sum
s_add_u32 s15, s15, s19 // (op 625) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 625) 

s_mov_b32 s44, s10 // (op 625) reg expression
s_mov_b32 s45, s11 // (op 625) reg expression
s_mov_b32 s46, s0 // (op 625) reg expression
s_mov_b32 s47, 131072 // (op 625) call()

v_and_b32 v12, -4, v1 // (op 642) 
v_sub_u32 v13, v1, v12 // (op 642) 

v_lshlrev_b32 v12, 6, v13 // (op 642) 

v_mov_b32 v19, 0 // (op 642) convert
v_mov_b32 v18, v12 // (op 642) convert

v_lshrrev_b32 v17, 2, v1 // (op 642) 
v_lshl_add_u32 v12, s17, 4, v17 // (op 642) 

v_mul_lo_u32 v21, v12, s9 // (op 642) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 642) most significant: high of low * low
v_mul_lo_u32 v20, v12, s8 // (op 642) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 642) most significant: sum


v_add_co_u32 v12, s[0:1], v18, v20 // (op 642) least significant half
v_addc_co_u32 v13, s[0:1], v19, v21, s[0:1] // (op 642) most significant half



v_mul_lo_u32 v19, v13, 6 // (op 642) most significant: low of high * low
v_mul_hi_u32 v17, v12, 6 // (op 642) most significant: high of low * low
v_mul_lo_u32 v18, v12, 6 // (op 642) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 642) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[18:19] // (op 642) 

v_mov_b32 v17, v12 // (op 642) convert

s_mul_i32 s1, s13, 6 // (op 644) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 644) least significant
s_mul_hi_u32 s14, s12, 6 // (op 644) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 644) most significant: sum

s_mul_i32 s15, s8, s1 // (op 644) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 644) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 644) least significant
s_mul_hi_u32 s19, s8, s0 // (op 644) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 644) most significant: sum
s_add_u32 s15, s15, s19 // (op 644) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 644) 

s_mov_b32 s44, s10 // (op 644) reg expression
s_mov_b32 s45, s11 // (op 644) reg expression
s_mov_b32 s46, s0 // (op 644) reg expression
s_mov_b32 s47, 131072 // (op 644) call()

v_and_b32 v12, -16, v1 // (op 687) 
v_sub_u32 v13, v1, v12 // (op 687) 

v_lshl_add_u32 v12, s16, 4, v13 // (op 687) 

v_lshrrev_b32 v13, 4, v1 // (op 687) 
v_lshlrev_b32 v18, 1, v13 // (op 687) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v13, v18, s20 // (op 687) 

v_add_u32 v18, v12, v13 // (op 687) 


s_mul_i32 s1, s20, s25 // (op 689) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 689) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 689) least significant
s_mul_hi_u32 s15, s20, s24 // (op 689) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 689) most significant: sum
s_add_u32 s1, s1, s15 // (op 689) most significant: sum


s_mov_b32 s48, s22 // (op 689) reg expression
s_mov_b32 s49, s23 // (op 689) reg expression
s_mov_b32 s50, s0 // (op 689) reg expression
s_mov_b32 s51, 131072 // (op 689) call()

v_and_b32 v12, -16, v1 // (op 706) 
v_sub_u32 v13, v1, v12 // (op 706) 

v_lshl_add_u32 v12, s16, 4, v13 // (op 706) 

v_lshrrev_b32 v13, 4, v1 // (op 706) 
v_lshlrev_b32 v19, 1, v13 // (op 706) 

v_mul_lo_u32 v13, v19, s20 // (op 706) 

v_add_u32 v19, v12, v13 // (op 706) 


s_mul_i32 s1, s20, s25 // (op 708) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 708) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 708) least significant
s_mul_hi_u32 s15, s20, s24 // (op 708) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 708) most significant: sum
s_add_u32 s1, s1, s15 // (op 708) most significant: sum


s_mov_b32 s48, s22 // (op 708) reg expression
s_mov_b32 s49, s23 // (op 708) reg expression
s_mov_b32 s50, s0 // (op 708) reg expression
s_mov_b32 s51, 131072 // (op 708) call()

v_and_b32 v12, -8, v1 // (op 751) 
v_sub_u32 v13, v1, v12 // (op 751) 

v_lshrrev_b32 v12, 3, v1 // (op 751) 
v_lshlrev_b32 v20, 1, v12 // (op 751) 

v_lshl_add_u32 v12, s17, 4, v20 // (op 751) 

v_mul_lo_u32 v20, v12, s26 // (op 751) 

v_add_u32 v12, v13, v20 // (op 751) 


s_mul_i32 s1, s26, s13 // (op 753) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 753) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 753) least significant
s_mul_hi_u32 s15, s26, s12 // (op 753) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 753) most significant: sum
s_add_u32 s1, s1, s15 // (op 753) most significant: sum


s_mov_b32 s52, s28 // (op 753) reg expression
s_mov_b32 s53, s29 // (op 753) reg expression
s_mov_b32 s54, s0 // (op 753) reg expression
s_mov_b32 s55, 131072 // (op 753) call()

v_and_b32 v13, -8, v1 // (op 770) 
v_sub_u32 v20, v1, v13 // (op 770) 

v_lshrrev_b32 v13, 3, v1 // (op 770) 
v_lshlrev_b32 v21, 1, v13 // (op 770) 

v_lshl_add_u32 v13, s17, 4, v21 // (op 770) 

v_mul_lo_u32 v21, v13, s26 // (op 770) 

v_add_u32 v13, v20, v21 // (op 770) 


s_mul_i32 s1, s26, s13 // (op 772) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 772) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 772) least significant
s_mul_hi_u32 s15, s26, s12 // (op 772) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 772) most significant: sum
s_add_u32 s1, s1, s15 // (op 772) most significant: sum


s_mov_b32 s52, s28 // (op 772) reg expression
s_mov_b32 s53, s29 // (op 772) reg expression
s_mov_b32 s54, s0 // (op 772) reg expression
s_mov_b32 s55, 131072 // (op 772) call()




buffer_load_dwordx4 v[20:23], v16, s[44:47], 0 offen // (op 424) Load value
buffer_load_dwordx2 v[24:25], v16, s[44:47], 0 offen offset:16 // (op 424) Load value
buffer_load_dwordx4 v[26:29], v16, s[44:47], 0 offen offset:24 // (op 424) Load value
buffer_load_dwordx2 v[30:31], v16, s[44:47], 0 offen offset:40 // (op 424) Load value

v_mov_b64 v[32:33], s[26:27] // (op 427) convert

v_mov_b32 v34, v12 // (op 427) 
buffer_load_ubyte v36, v34, s[52:55], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v34, v34, v32 // (op 427) 
buffer_load_ubyte v37, v34, s[52:55], 0 offen offset: 0 // (op 427) Load value



s_lshl_b64 s[0:1], s[2:3], 2 // (op 430) 
v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 430) 


v_mov_b32 v34, v14 // (op 430) reg expression
buffer_load_dwordx2 v[38:39], v34, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v34, v34, v32 // (op 430) 
buffer_load_dwordx2 v[40:41], v34, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v34, v34, v32 // (op 430) 
buffer_load_dwordx2 v[42:43], v34, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v34, v34, v32 // (op 430) 
buffer_load_dwordx2 v[44:45], v34, s[40:43], 0 offen // (op 430) Load value



v_mov_b64 v[32:33], s[20:21] // (op 433) convert

v_mov_b32 v34, v18 // (op 433) reg expression
buffer_load_ubyte v46, v34, s[48:51], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v34, v34, v32 // (op 433) 
buffer_load_ubyte v47, v34, s[48:51], 0 offen offset: 0 // (op 433) Load value





buffer_load_dwordx4 v[48:51], v16, s[44:47], 0 offen offset:192 // (op 436) Load value
buffer_load_dwordx2 v[52:53], v16, s[44:47], 0 offen offset:208 // (op 436) Load value
buffer_load_dwordx4 v[54:57], v16, s[44:47], 0 offen offset:216 // (op 436) Load value
buffer_load_dwordx2 v[58:59], v16, s[44:47], 0 offen offset:232 // (op 436) Load value


v_mov_b64 v[32:33], s[26:27] // (op 439) convert

v_mov_b32 v16, v12 // (op 439) 
buffer_load_ubyte v34, v16, s[52:55], 0 offen offset:8 // (op 439) Load value
v_add_u32 v16, v16, v32 // (op 439) 
buffer_load_ubyte v35, v16, s[52:55], 0 offen offset:8 // (op 439) Load value




s_lshl_b64 s[0:1], s[2:3], 2 // (op 442) 
v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 442) 


s_lshl_b64 s[0:1], s[2:3], 8 // (op 442) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 442) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 442) 

v_add_u32 v12, v14, s0 // (op 442) 

buffer_load_dwordx2 v[60:61], v12, s[40:43], 0 offen // (op 442) Load value
v_add_u32 v12, v12, v32 // (op 442) 
buffer_load_dwordx2 v[62:63], v12, s[40:43], 0 offen // (op 442) Load value
v_add_u32 v12, v12, v32 // (op 442) 
buffer_load_dwordx2 v[64:65], v12, s[40:43], 0 offen // (op 442) Load value
v_add_u32 v12, v12, v32 // (op 442) 
buffer_load_dwordx2 v[66:67], v12, s[40:43], 0 offen // (op 442) Load value




v_mov_b64 v[32:33], s[20:21] // (op 445) convert

s_lshl_b32 s0, s20, 3 // (op 445) 
v_add_u32 v12, v18, s0 // (op 445) 

buffer_load_ubyte v68, v12, s[48:51], 0 offen offset: 0 // (op 445) Load value
v_add_u32 v12, v12, v32 // (op 445) 
buffer_load_ubyte v69, v12, s[48:51], 0 offen offset: 0 // (op 445) Load value



v_and_b32 v12, -4, v1 // (op 890) 
v_sub_u32 v14, v1, v12 // (op 890) 

v_lshlrev_b32 v12, 6, v14 // (op 890) 

v_lshrrev_b32 v14, 2, v1 // (op 890) 
v_lshl_add_u32 v16, v14, 8, v12 // (op 890) 


v_mul_lo_u32 v12, v16, 6 // (op 890) 

v_lshrrev_b32 v14, 3, v12 // (op 890) 




s_waitcnt vmcnt(23)
ds_write_b128 v14, v[20:23] // (op 447) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v14, v[24:25] offset:16 // (op 447) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v14, v[26:29] offset:24 // (op 447) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v14, v[30:31] offset:40 // (op 447) Store local data 

v_and_b32 v12, -8, v1 // (op 996) 
v_sub_u32 v14, v1, v12 // (op 996) 

v_lshrrev_b32 v12, 3, v1 // (op 996) 
v_lshl_add_u32 v16, v12, 4, v14 // (op 996) 





s_waitcnt vmcnt(19)
ds_write_b8 v16, v36 offset:3072 // (op 448) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v16, v37 offset:3080 // (op 448) Store local data 

v_lshlrev_b32 v12, 8, v1 // (op 837) 
v_lshrrev_b32 v14, 3, v12 // (op 837) 




s_waitcnt vmcnt(17)
ds_write_b64 v14, v[38:39] offset:3200 // (op 449) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v14, v[40:41] offset:3208 // (op 449) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v14, v[42:43] offset:3216 // (op 449) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v14, v[44:45] offset:3224 // (op 449) Store local data 

v_and_b32 v12, -16, v1 // (op 943) 
v_sub_u32 v14, v1, v12 // (op 943) 

v_lshrrev_b32 v12, 4, v1 // (op 943) 
v_lshl_add_u32 v16, v12, 5, v14 // (op 943) 





s_waitcnt vmcnt(13)
ds_write_b8 v16, v46 offset:5248 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v16, v47 offset:5264 // (op 450) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1194) 



ds_read_b128 v[70:73], v4 // (op 467) Load local data 
ds_read_b64 v[74:75], v4 offset:16 // (op 467) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1235) 



ds_read_u8 v12, v6 offset:3072 // (op 473) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1175) 



ds_read_b64_tr_b4 v[76:77], v3 offset:3200 // (op 464) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v3 offset:3328 // (op 464) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1215) 


ds_read_u8 v14, v5 offset:5248 // (op 470) Load local data 
v_lshlrev_b32 v16, 8, v1 // (op 808) 
v_lshrrev_b32 v18, 3, v16 // (op 808) 

v_and_b32 v16, -4, v1 // (op 861) 
v_sub_u32 v32, v1, v16 // (op 861) 

v_lshlrev_b32 v16, 6, v32 // (op 861) 

v_lshrrev_b32 v32, 2, v1 // (op 861) 
v_lshl_add_u32 v33, v32, 8, v16 // (op 861) 


v_mul_lo_u32 v16, v33, 6 // (op 861) 

v_lshrrev_b32 v32, 3, v16 // (op 861) 

v_and_b32 v16, -16, v1 // (op 914) 
v_sub_u32 v33, v1, v16 // (op 914) 

v_lshrrev_b32 v16, 4, v1 // (op 914) 
v_lshl_add_u32 v80, v16, 5, v33 // (op 914) 


v_and_b32 v16, -8, v1 // (op 967) 
v_sub_u32 v33, v1, v16 // (op 967) 

v_lshrrev_b32 v16, 3, v1 // (op 967) 
v_lshl_add_u32 v81, v16, 4, v33 // (op 967) 


v_lshlrev_b32 v16, 8, v1 // (op 1054) 
v_lshrrev_b32 v33, 3, v16 // (op 1054) 

v_and_b32 v16, -4, v1 // (op 1078) 
v_sub_u32 v82, v1, v16 // (op 1078) 

v_lshlrev_b32 v16, 6, v82 // (op 1078) 

v_lshrrev_b32 v82, 2, v1 // (op 1078) 
v_lshl_add_u32 v83, v82, 8, v16 // (op 1078) 


v_mul_lo_u32 v16, v83, 6 // (op 1078) 

v_lshrrev_b32 v82, 3, v16 // (op 1078) 

v_and_b32 v16, -16, v1 // (op 1102) 
v_sub_u32 v83, v1, v16 // (op 1102) 

v_lshrrev_b32 v16, 4, v1 // (op 1102) 
v_lshl_add_u32 v84, v16, 5, v83 // (op 1102) 


v_and_b32 v16, -8, v1 // (op 1126) 
v_sub_u32 v83, v1, v16 // (op 1126) 

v_lshrrev_b32 v16, 3, v1 // (op 1126) 
v_lshl_add_u32 v85, v16, 4, v83 // (op 1126) 


s_mov_b32 s0, 0 // (op 59) call()
s_ashr_i64 s[14:15], s[6:7], 8 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58:


s_add_u32 s0, s0, 2 // (op 60) 
v_mov_b32 v16, v17 // (op 10) 

buffer_load_dwordx4 v[20:23], v16, s[44:47], 0 offen offset:384 // (op 10) Load value
buffer_load_dwordx2 v[24:25], v16, s[44:47], 0 offen offset:400 // (op 10) Load value
buffer_load_dwordx4 v[26:29], v16, s[44:47], 0 offen offset:408 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[70:75], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 88) 
buffer_load_dwordx2 v[30:31], v16, s[44:47], 0 offen offset:424 // (op 10) Load value

s_barrier  // (op 1184) 
v_mov_b64 v[86:87], s[26:27] // (op 25) convert




ds_read_b64_tr_b4 v[76:77], v3 offset:4224 // (op 91) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v3 offset:4352 // (op 91) Transpose load local data 
v_mov_b32 v83, v13 // (op 25) 
s_barrier  // (op 1204) 
v_mov_b32 v88, v83 // (op 25) 



ds_read_b128 v[70:73], v4 offset:96 // (op 95) Load local data 
buffer_load_ubyte v36, v88, s[52:55], 0 offen offset:16 // (op 25) Load value
v_add_u32 v88, v88, v86 // (op 25) 
buffer_load_ubyte v37, v88, s[52:55], 0 offen offset:16 // (op 25) Load value


ds_read_b64 v[74:75], v4 offset:112 // (op 95) Load local data 

s_barrier  // (op 1225) 
s_lshl_b64 s[14:15], s[2:3], 2 // (op 4) 


ds_read_u8 v14, v5 offset:5312 // (op 99) Load local data 
v_lshrrev_b64 v[86:87], 3, s[14:15] // (op 4) 


s_barrier  // (op 1245) 
v_mov_b32 v88, v15 // (op 4) 



ds_read_u8 v12, v6 offset:3076 // (op 103) Load local data 
s_lshl_b64 s[14:15], s[2:3], 8 // (op 4) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 4) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s1, s14, 1 // (op 4) 

v_add_u32 v89, v88, s1 // (op 4) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[70:75], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 105) 
buffer_load_dwordx2 v[38:39], v89, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v89, v89, v86 // (op 4) 
buffer_load_dwordx2 v[40:41], v89, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v89, v89, v86 // (op 4) 
buffer_load_dwordx2 v[42:43], v89, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v89, v89, v86 // (op 4) 
buffer_load_dwordx2 v[44:45], v89, s[40:43], 0 offen // (op 4) Load value



v_mov_b64 v[86:87], s[20:21] // (op 17) convert

v_mov_b32 v89, v19 // (op 17) 
s_lshl_b32 s1, s20, 4 // (op 17) 
v_add_u32 v90, v89, s1 // (op 17) 

buffer_load_ubyte v46, v90, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v90, v90, v86 // (op 17) 
buffer_load_ubyte v47, v90, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1265) 



s_waitcnt vmcnt(23)
ds_write_b128 v82, v[48:51] offset:5376 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v82, v[52:53] offset:5392 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v82, v[54:57] offset:5400 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v82, v[58:59] offset:5416 // (op 302) Store local data 
s_barrier  // (op 1277) 



s_waitcnt vmcnt(19)
ds_write_b8 v85, v34 offset:8448 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v85, v35 offset:8456 // (op 304) Store local data 
s_barrier  // (op 1258) 



s_waitcnt vmcnt(17)
ds_write_b64 v33, v[60:61] offset:8576 // (op 283) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v33, v[62:63] offset:8584 // (op 283) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v33, v[64:65] offset:8592 // (op 283) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v33, v[66:67] offset:8600 // (op 283) Store local data 
s_barrier  // (op 1271) 



s_waitcnt vmcnt(13)
ds_write_b8 v84, v68 offset:10624 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v84, v69 offset:10640 // (op 303) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1261) 



ds_read_b128 v[90:93], v8 offset:5376 // (op 287) Load local data 
ds_read_b64 v[94:95], v8 offset:5392 // (op 287) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1274) 



ds_read_u8 v86, v10 offset:8448 // (op 291) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1255) 



ds_read_b64_tr_b4 v[96:97], v7 offset:8576 // (op 285) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v7 offset:8704 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1268) 


ds_read_u8 v87, v9 offset:10624 // (op 289) Load local data 



buffer_load_dwordx4 v[48:51], v16, s[44:47], 0 offen offset:576 // (op 305) Load value
buffer_load_dwordx2 v[52:53], v16, s[44:47], 0 offen offset:592 // (op 305) Load value
buffer_load_dwordx4 v[54:57], v16, s[44:47], 0 offen offset:600 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[96:99], v[90:95], a[0:3], v87, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 292) 
buffer_load_dwordx2 v[58:59], v16, s[44:47], 0 offen offset:616 // (op 305) Load value









v_add_u32 v17, 384, v17 // (op 634) 
v_mov_b64 v[86:87], s[26:27] // (op 307) convert

v_mov_b32 v16, v83 // (op 307) 
buffer_load_ubyte v34, v16, s[52:55], 0 offen offset:24 // (op 307) Load value
v_add_u32 v16, v16, v86 // (op 307) 
buffer_load_ubyte v35, v16, s[52:55], 0 offen offset:24 // (op 307) Load value


ds_read_b64_tr_b4 v[90:91], v7 offset:9600 // (op 294) Transpose load local data 

ds_read_b64_tr_b4 v[92:93], v7 offset:9728 // (op 294) Transpose load local data 

v_add_u32 v13, 16, v13 // (op 762) 



s_barrier  // (op 1336) 
s_lshl_b64 s[14:15], s[2:3], 2 // (op 282) 
v_lshrrev_b64 v[86:87], 3, s[14:15] // (op 282) 


s_lshl_b64 s[14:15], s[2:3], 8 // (op 282) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 282) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 282) 

s_mul_i32 s1, 3, s14 // (op 282) 

v_add_u32 v16, v88, s1 // (op 282) 

ds_read_b128 v[94:97], v8 offset:5472 // (op 296) Load local data 
ds_read_b64 v[98:99], v8 offset:5488 // (op 296) Load local data 
buffer_load_dwordx2 v[60:61], v16, s[40:43], 0 offen // (op 282) Load value
v_add_u32 v16, v16, v86 // (op 282) 


s_barrier  // (op 1340) 
ds_read_u8 v83, v9 offset:10688 // (op 298) Load local data 
buffer_load_dwordx2 v[62:63], v16, s[40:43], 0 offen // (op 282) Load value
v_add_u32 v16, v16, v86 // (op 282) 



s_barrier  // (op 1344) 
buffer_load_dwordx2 v[64:65], v16, s[40:43], 0 offen // (op 282) Load value
v_add_u32 v16, v16, v86 // (op 282) 
ds_read_u8 v100, v10 offset:8452 // (op 300) Load local data 
buffer_load_dwordx2 v[66:67], v16, s[40:43], 0 offen // (op 282) Load value




s_barrier  // (op 1348) 
v_mov_b64 v[86:87], s[20:21] // (op 306) convert

s_lshl_b32 s1, s20, 3 // (op 306) 
s_mul_i32 s14, 3, s1 // (op 306) 

v_add_u32 v16, v89, s14 // (op 306) 

buffer_load_ubyte v68, v16, s[48:51], 0 offen offset: 0 // (op 306) Load value
v_add_u32 v16, v16, v86 // (op 306) 
buffer_load_ubyte v69, v16, s[48:51], 0 offen offset: 0 // (op 306) Load value


s_lshl_b64 s[14:15], s[2:3], 8 // (op 570) 

s_lshl_b64 s[18:19], s[14:15], 2 // (op 570) 

s_barrier  // (op 1197) 
s_lshl_b32 s1, s20, 4 // (op 698) 



v_add_u32 v19, v19, s1 // (op 698) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 570) 

s_lshl_b32 s1, s14, 1 // (op 570) 

v_add_u32 v15, v15, s1 // (op 570) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[94:99], a[0:3], v83, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 301) 
s_waitcnt vmcnt(23)
ds_write_b128 v32, v[20:23] // (op 13) Store local data 




s_waitcnt vmcnt(22)
ds_write_b64 v32, v[24:25] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v32, v[26:29] offset:24 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v32, v[30:31] offset:40 // (op 13) Store local data 
s_barrier  // (op 1238) 



s_waitcnt vmcnt(19)
ds_write_b8 v81, v36 offset:3072 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v81, v37 offset:3080 // (op 30) Store local data 
s_barrier  // (op 1178) 



s_waitcnt vmcnt(17)
ds_write_b64 v18, v[38:39] offset:3200 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v18, v[40:41] offset:3208 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v18, v[42:43] offset:3216 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v18, v[44:45] offset:3224 // (op 6) Store local data 
s_barrier  // (op 1218) 



s_waitcnt vmcnt(13)
ds_write_b8 v80, v46 offset:5248 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v80, v47 offset:5264 // (op 21) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1207) 



ds_read_b128 v[70:73], v4 // (op 75) Load local data 
ds_read_b64 v[74:75], v4 offset:16 // (op 75) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1248) 



ds_read_u8 v12, v6 offset:3072 // (op 85) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1187) 



ds_read_b64_tr_b4 v[76:77], v3 offset:3200 // (op 70) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v3 offset:3328 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1228) 


ds_read_u8 v14, v5 offset:5248 // (op 80) Load local data 
s_ashr_i64 s[14:15], s[6:7], 8 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58:

















v_lshlrev_b32 v13, 2, v1 // (op 533) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 533) 
s_lshr_b32 s14, s0, 1 // (op 533) 

v_lshl_add_u32 v15, s14, 9, v13 // (op 533) 


s_lshl_b32 s0, s16, 4 // (op 533) 
v_mul_lo_u32 v17, v15, s3 // (op 533) most significant: low of low * high
v_mul_hi_u32 v13, v15, s2 // (op 533) most significant: high of low * low
v_mul_lo_u32 v16, v15, s2 // (op 533) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 533) most significant: sum


s_mov_b32 s15, 0 // (op 533) convert
s_mov_b32 s14, s0 // (op 533) convert

v_mov_b32 v13, s14 // (op 533) 
v_mov_b32 v15, s15 // (op 533) 
v_add_co_u32 v18, s[0:1], v13, v16 // (op 533) least significant half
v_addc_co_u32 v19, s[0:1], v15, v17, s[0:1] // (op 533) most significant half



v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 533) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 533) 

v_mov_b32 v13, v16 // (op 533) convert

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

v_and_b32 v15, -4, v1 // (op 600) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 600) 
v_sub_u32 v16, v1, v15 // (op 600) 

s_lshr_b32 s14, s0, 1 // (op 600) 

v_lshlrev_b32 v15, 6, v16 // (op 600) 

v_lshl_add_u32 v16, s14, 9, v15 // (op 600) 


v_mov_b32 v33, 0 // (op 600) convert
v_mov_b32 v32, v16 // (op 600) convert

v_lshrrev_b32 v15, 2, v1 // (op 600) 
v_lshl_add_u32 v16, s17, 4, v15 // (op 600) 

v_mul_lo_u32 v31, v16, s9 // (op 600) most significant: low of low * high
v_mul_hi_u32 v15, v16, s8 // (op 600) most significant: high of low * low
v_mul_lo_u32 v30, v16, s8 // (op 600) least significant: low of low * low
v_add_u32 v31, v31, v15 // (op 600) most significant: sum


v_add_co_u32 v28, s[0:1], v32, v30 // (op 600) least significant half
v_addc_co_u32 v29, s[0:1], v33, v31, s[0:1] // (op 600) most significant half



v_mul_lo_u32 v27, v29, 6 // (op 600) most significant: low of high * low
v_mul_hi_u32 v15, v28, 6 // (op 600) most significant: high of low * low
v_mul_lo_u32 v26, v28, 6 // (op 600) least significant: low of low * low
v_add_u32 v27, v27, v15 // (op 600) most significant: sum


v_lshrrev_b64 v[24:25], 3, v[26:27] // (op 600) 

v_mov_b32 v15, v24 // (op 600) convert

s_mul_i32 s1, s13, 6 // (op 602) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 602) least significant
s_mul_hi_u32 s14, s12, 6 // (op 602) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 602) most significant: sum

s_mul_i32 s15, s8, s1 // (op 602) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 602) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 602) least significant
s_mul_hi_u32 s19, s8, s0 // (op 602) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 602) most significant: sum
s_add_u32 s15, s15, s19 // (op 602) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 602) 

s_mov_b32 s44, s10 // (op 602) reg expression
s_mov_b32 s45, s11 // (op 602) reg expression
s_mov_b32 s46, s0 // (op 602) reg expression
s_mov_b32 s47, 131072 // (op 602) call()

v_and_b32 v16, -16, v1 // (op 664) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 664) 
v_sub_u32 v17, v1, v16 // (op 664) 

s_lshr_b32 s14, s0, 1 // (op 664) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 664) 

v_lshrrev_b32 v17, 4, v1 // (op 664) 
v_lshlrev_b32 v18, 1, v17 // (op 664) 

v_lshl_add_u32 v17, s14, 4, v18 // (op 664) 


v_mul_lo_u32 v18, v17, s20 // (op 664) 

v_add_u32 v17, v16, v18 // (op 664) 


s_mul_i32 s1, s20, s25 // (op 666) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 666) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 666) least significant
s_mul_hi_u32 s15, s20, s24 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 666) most significant: sum
s_add_u32 s1, s1, s15 // (op 666) most significant: sum


s_mov_b32 s48, s22 // (op 666) reg expression
s_mov_b32 s49, s23 // (op 666) reg expression
s_mov_b32 s50, s0 // (op 666) reg expression
s_mov_b32 s51, 131072 // (op 666) call()

v_and_b32 v16, -8, v1 // (op 728) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 728) 
v_sub_u32 v18, v1, v16 // (op 728) 

s_lshr_b32 s14, s0, 1 // (op 728) 

v_lshrrev_b32 v16, 3, v1 // (op 728) 
v_lshl_add_u32 v19, s14, 4, v18 // (op 728) 


v_lshlrev_b32 v18, 1, v16 // (op 728) 

v_lshl_add_u32 v16, s17, 4, v18 // (op 728) 

v_mul_lo_u32 v18, v16, s26 // (op 728) 

v_add_u32 v16, v19, v18 // (op 728) 


s_mul_i32 s1, s26, s13 // (op 730) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 730) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 730) least significant
s_mul_hi_u32 s15, s26, s12 // (op 730) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 730) most significant: sum
s_add_u32 s1, s1, s15 // (op 730) most significant: sum


s_mov_b32 s52, s28 // (op 730) reg expression
s_mov_b32 s53, s29 // (op 730) reg expression
s_mov_b32 s54, s0 // (op 730) reg expression
s_mov_b32 s55, 131072 // (op 730) call()

v_lshlrev_b32 v18, 8, v1 // (op 821) 
v_lshrrev_b32 v19, 3, v18 // (op 821) 

v_and_b32 v18, -4, v1 // (op 874) 
v_sub_u32 v20, v1, v18 // (op 874) 

v_lshlrev_b32 v18, 6, v20 // (op 874) 

v_lshrrev_b32 v20, 2, v1 // (op 874) 
v_lshl_add_u32 v21, v20, 8, v18 // (op 874) 


v_mul_lo_u32 v18, v21, 6 // (op 874) 

v_lshrrev_b32 v20, 3, v18 // (op 874) 

v_and_b32 v18, -16, v1 // (op 927) 
v_sub_u32 v21, v1, v18 // (op 927) 

v_lshrrev_b32 v18, 4, v1 // (op 927) 
v_lshl_add_u32 v22, v18, 5, v21 // (op 927) 


v_and_b32 v18, -8, v1 // (op 980) 
v_sub_u32 v21, v1, v18 // (op 980) 

v_lshrrev_b32 v18, 3, v1 // (op 980) 
v_lshl_add_u32 v23, v18, 4, v21 // (op 980) 


s_ashr_i64 s[0:1], s[6:7], 8 // (op 194) 
s_and_b32 s14, s0, 4294967294 // (op 194) 

s_ashr_i64 s[0:1], s[6:7], 8 // (op 193) 
s_mov_b32 s15, s0 // (op 193) convert

s_cmp_lt_u32 s14, s15 // (op 193) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193:

s_add_u32 s14, s14, 1 // (op 195) 
s_lshl_b64 s[0:1], s[2:3], 2 // (op 198) 
v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 198) 


v_mov_b32 v12, v13 // (op 198) 
v_mov_b32 v14, v12 // (op 198) reg expression
buffer_load_dwordx2 v[26:27], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v24 // (op 198) 
buffer_load_dwordx2 v[28:29], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v24 // (op 198) 
buffer_load_dwordx2 v[30:31], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v24 // (op 198) 
buffer_load_dwordx2 v[32:33], v14, s[40:43], 0 offen // (op 198) Load value



s_lshl_b64 s[0:1], s[2:3], 8 // (op 523) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 523) 

s_lshr_b64 s[0:1], s[18:19], 3 // (op 523) 

v_add_u32 v13, v13, s0 // (op 523) 



s_barrier  // (op 1181) 
v_mov_b32 v12, v15 // (op 221) 




buffer_load_dwordx4 v[34:37], v12, s[44:47], 0 offen // (op 221) Load value
buffer_load_dwordx2 v[38:39], v12, s[44:47], 0 offen offset:16 // (op 221) Load value
s_waitcnt vmcnt(5)
ds_write_b64 v19, v[26:27] offset:3200 // (op 199) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v19, v[28:29] offset:3208 // (op 199) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v19, v[30:31] offset:3216 // (op 199) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v19, v[32:33] offset:3224 // (op 199) Store local data 
buffer_load_dwordx4 v[40:43], v12, s[44:47], 0 offen offset:24 // (op 221) Load value

buffer_load_dwordx2 v[44:45], v12, s[44:47], 0 offen offset:40 // (op 221) Load value

v_add_u32 v15, 192, v15 // (op 592) 

s_barrier  // (op 1200) 
v_mov_b64 v[24:25], s[20:21] // (op 222) convert




v_mov_b32 v12, v17 // (op 222) 
v_mov_b32 v14, v12 // (op 222) reg expression
buffer_load_ubyte v26, v14, s[48:51], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v14, v14, v24 // (op 222) 
s_waitcnt vmcnt(4)
ds_write_b128 v20, v[34:37] // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v20, v[38:39] offset:16 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[40:43] offset:24 // (op 218) Store local data 
buffer_load_ubyte v27, v14, s[48:51], 0 offen offset: 0 // (op 222) Load value



s_lshl_b32 s15, s20, 3 // (op 656) 
v_add_u32 v17, v17, s15 // (op 656) 

s_waitcnt vmcnt(2)
ds_write_b64 v20, v[44:45] offset:40 // (op 218) Store local data 


v_mov_b64 v[24:25], s[26:27] // (op 223) convert

s_barrier  // (op 1221) 
v_mov_b32 v12, v16 // (op 223) 



v_mov_b32 v14, v12 // (op 223) 
buffer_load_ubyte v28, v14, s[52:55], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v14, v14, v24 // (op 223) 
buffer_load_ubyte v29, v14, s[52:55], 0 offen offset: 0 // (op 223) Load value



v_add_u32 v16, 8, v16 // (op 720) 
s_waitcnt vmcnt(3)
ds_write_b8 v22, v26 offset:5248 // (op 219) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1190) 
s_waitcnt vmcnt(2)
ds_write_b8 v22, v27 offset:5264 // (op 219) Store local data 




ds_read_b64_tr_b4 v[24:25], v3 offset:3200 // (op 201) Transpose load local data 
s_barrier  // (op 1241) 
ds_read_b64_tr_b4 v[26:27], v3 offset:3328 // (op 201) Transpose load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 1211) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v28 offset:3072 // (op 220) Store local data 



ds_read_b128 v[30:33], v4 // (op 203) Load local data 
ds_read_b64 v[34:35], v4 offset:16 // (op 203) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v29 offset:3080 // (op 220) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1231) 


ds_read_u8 v12, v5 offset:5248 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1251) 



ds_read_u8 v14, v6 offset:3072 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[30:35], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 208) 







ds_read_b64_tr_b4 v[24:25], v3 offset:4224 // (op 210) Transpose load local data 
ds_read_b64_tr_b4 v[26:27], v3 offset:4352 // (op 210) Transpose load local data 



ds_read_b128 v[28:31], v4 offset:96 // (op 212) Load local data 
ds_read_b64 v[32:33], v4 offset:112 // (op 212) Load local data 


ds_read_u8 v12, v5 offset:5312 // (op 214) Load local data 



ds_read_u8 v14, v6 offset:3076 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:27], v[28:33], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b011 // (op 217) 




s_ashr_i64 s[0:1], s[6:7], 8 // (op 193) 
s_mov_b32 s15, s0 // (op 193) convert

s_cmp_lt_u32 s14, s15 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 if true)
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1328) 








v_cmp_eq_f32 vcc, 0.00000, v0 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v11, v4 // (op 50) 
v_mul_f32 v13, v11, v5 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 

v_lshrrev_b32 v3, 6, v1 // (op 1022) 
v_and_b32 v4, 63, v1 // (op 1022) 
v_lshrrev_b32 v5, 4, v4 // (op 1022) 
v_lshlrev_b32 v6, 2, v5 // (op 1022) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 1022) 


v_lshl_add_u32 v3, s16, 4, v5 // (op 1022) 

v_lshrrev_b32 v5, 2, v4 // (op 1022) 

v_and_b32 v4, 3, v5 // (op 1022) 

v_and_b32 v5, 3, v1 // (op 1022) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 1022) 


v_lshl_add_u32 v4, s17, 4, v6 // (op 1022) 

v_mul_lo_u32 v5, v4, s30 // (op 1022) 

v_add_lshl_u32 v4, v3, v5, 2 // (op 1022) 


s_mul_i32 s15, s30, s13 // (op 1024) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1024) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1024) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1024) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1024) most significant: sum
s_add_u32 s15, s15, s1 // (op 1024) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 1024) 

s_mov_b32 s8, s32 // (op 1024) reg expression
s_mov_b32 s9, s33 // (op 1024) reg expression
s_mov_b32 s10, s0 // (op 1024) reg expression
s_mov_b32 s11, 131072 // (op 1024) call()




buffer_store_dwordx4 v[12:15], v4, s[8:11], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155:
v_lshrrev_b32 v3, 6, v1 // (op 791) 
v_and_b32 v4, 63, v1 // (op 791) 
v_lshrrev_b32 v5, 4, v4 // (op 791) 
v_lshlrev_b32 v6, 2, v5 // (op 791) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 791) 


v_lshl_add_u32 v3, s16, 4, v5 // (op 791) 

v_lshrrev_b32 v5, 2, v4 // (op 791) 

v_and_b32 v4, 3, v5 // (op 791) 

v_and_b32 v5, 3, v1 // (op 791) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 791) 


v_lshl_add_u32 v4, s17, 4, v6 // (op 791) 

v_mul_lo_u32 v5, v4, s34 // (op 791) 

v_add_lshl_u32 v4, v3, v5, 2 // (op 791) 


s_mul_i32 s15, s34, s13 // (op 793) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 793) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 793) least significant
s_mul_hi_u32 s1, s34, s12 // (op 793) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 793) most significant: sum
s_add_u32 s15, s15, s1 // (op 793) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 793) 

s_mov_b32 s4, s36 // (op 793) reg expression
s_mov_b32 s5, s37 // (op 793) reg expression
s_mov_b32 s6, s0 // (op 793) reg expression
s_mov_b32 s7, 131072 // (op 793) call()





buffer_load_dwordx4 v[6:9], v4, s[4:7], 0 offen // (op 167) Load value



v_accvgpr_read v12, a0 // (op 164) 
v_accvgpr_read v13, a1 // (op 164) 
v_accvgpr_read v14, a2 // (op 164) 
v_accvgpr_read v15, a3 // (op 164) 
v_mul_f32 v16, v11, v12 // (op 164) 
v_mul_f32 v17, v11, v13 // (op 164) 
v_mul_f32 v18, v11, v14 // (op 164) 
v_mul_f32 v19, v11, v15 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v0, v6 // (op 168) 
v_mul_f32 v13, v0, v7 // (op 168) 
v_mul_f32 v14, v0, v8 // (op 168) 
v_mul_f32 v15, v0, v9 // (op 168) 

v_add_f32 v4, v12, v16 // (op 165) 
v_add_f32 v5, v13, v17 // (op 165) 
v_add_f32 v6, v14, v18 // (op 165) 
v_add_f32 v7, v15, v19 // (op 165) 


v_lshrrev_b32 v3, 6, v1 // (op 1039) 
v_and_b32 v8, 63, v1 // (op 1039) 
v_lshrrev_b32 v9, 4, v8 // (op 1039) 
v_lshlrev_b32 v10, 2, v9 // (op 1039) 

v_lshl_add_u32 v9, v3, 4, v10 // (op 1039) 


v_lshl_add_u32 v3, s16, 4, v9 // (op 1039) 

v_lshrrev_b32 v9, 2, v8 // (op 1039) 

v_and_b32 v8, 3, v9 // (op 1039) 

v_and_b32 v9, 3, v1 // (op 1039) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 1039) 


v_lshl_add_u32 v8, s17, 4, v10 // (op 1039) 

v_mul_lo_u32 v9, v8, s30 // (op 1039) 

v_add_lshl_u32 v8, v3, v9, 2 // (op 1039) 


s_mul_i32 s15, s30, s13 // (op 1041) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1041) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1041) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1041) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1041) most significant: sum
s_add_u32 s15, s15, s1 // (op 1041) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 1041) 

s_mov_b32 s8, s32 // (op 1041) reg expression
s_mov_b32 s9, s33 // (op 1041) reg expression
s_mov_b32 s10, s0 // (op 1041) reg expression
s_mov_b32 s11, 131072 // (op 1041) call()




buffer_store_dwordx4 v[4:7], v8, s[8:11], 0 offen // (op 166) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155:




s_endpgm  // End of RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 108
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 10752 // lds bytes
  .amdhsa_accum_offset 104
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
    DataTypeB: B6
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
  - .name: RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NN_FP4_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 10752
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 101
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

