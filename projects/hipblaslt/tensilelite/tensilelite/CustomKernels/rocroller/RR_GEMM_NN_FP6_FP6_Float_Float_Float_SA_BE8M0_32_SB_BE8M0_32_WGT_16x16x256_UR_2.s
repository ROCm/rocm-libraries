// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_mul_lo_u32 v0, v3, 6 // (op 849) 
v_lshrrev_b32 v4, 3, v0 // (op 849) 

v_lshrrev_b32 v0, 4, v3 // (op 849) 

v_lshlrev_b32 v3, 2, v0 // (op 849) 

v_add_u32 v0, v4, v3 // (op 849) 


v_and_b32 v3, 15, v1 // (op 902) 
v_and_b32 v4, 63, v1 // (op 902) 
v_lshrrev_b32 v5, 4, v4 // (op 902) 

v_lshlrev_b32 v4, 5, v5 // (op 902) 

v_lshl_add_u32 v5, v3, 8, v4 // (op 902) 


v_mul_lo_u32 v3, v5, 6 // (op 902) 

v_lshrrev_b32 v4, 3, v3 // (op 902) 

v_lshrrev_b32 v3, 6, v1 // (op 955) 
v_and_b32 v5, 15, v1 // (op 955) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 955) 


v_and_b32 v3, 48, v1 // (op 955) 
v_add_u32 v5, v3, v6 // (op 955) 


v_and_b32 v3, 15, v1 // (op 1008) 
v_and_b32 v6, 63, v1 // (op 1008) 
v_lshrrev_b32 v7, 4, v6 // (op 1008) 

v_lshl_add_u32 v6, v3, 3, v7 // (op 1008) 


v_and_b32 v3, 63, v1 // (op 1066) 
v_and_b32 v7, 48, v1 // (op 1066) 
v_sub_u32 v8, v3, v7 // (op 1066) 

v_lshrrev_b32 v7, 4, v3 // (op 1066) 

v_lshl_add_u32 v3, v7, 5, v8 // (op 1066) 


v_lshrrev_b32 v7, 6, v1 // (op 1066) 
v_lshlrev_b32 v8, 4, v7 // (op 1066) 

v_lshl_add_u32 v7, v3, 4, v8 // (op 1066) 


v_mul_lo_u32 v3, v7, 6 // (op 1066) 
v_lshrrev_b32 v8, 3, v3 // (op 1066) 

v_lshrrev_b32 v3, 4, v7 // (op 1066) 

v_lshlrev_b32 v7, 2, v3 // (op 1066) 

v_add_u32 v3, v8, v7 // (op 1066) 


v_and_b32 v7, 15, v1 // (op 1090) 
v_and_b32 v8, 63, v1 // (op 1090) 
v_lshrrev_b32 v9, 4, v8 // (op 1090) 

v_lshlrev_b32 v8, 5, v9 // (op 1090) 

v_lshl_add_u32 v9, v7, 8, v8 // (op 1090) 


v_mul_lo_u32 v7, v9, 6 // (op 1090) 

v_lshrrev_b32 v8, 3, v7 // (op 1090) 

v_lshrrev_b32 v7, 6, v1 // (op 1114) 
v_and_b32 v9, 15, v1 // (op 1114) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1114) 


v_and_b32 v7, 48, v1 // (op 1114) 
v_add_u32 v9, v7, v10 // (op 1114) 


v_and_b32 v7, 15, v1 // (op 1138) 
v_and_b32 v10, 63, v1 // (op 1138) 
v_lshrrev_b32 v11, 4, v10 // (op 1138) 

v_lshl_add_u32 v10, v7, 3, v11 // (op 1138) 


v_mov_b32 v7, s14 // (op 2) Move value
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





v_mul_lo_u32 v13, v17, 6 // (op 559) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 559) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 559) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 559) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 559) 

v_mov_b32 v12, v14 // (op 559) convert

s_mul_i32 s1, s7, 6 // (op 561) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 561) least significant
s_mul_hi_u32 s14, s6, 6 // (op 561) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 561) most significant: sum

s_mul_i32 s15, s2, s1 // (op 561) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 561) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 561) least significant
s_mul_hi_u32 s19, s2, s0 // (op 561) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 561) most significant: sum
s_add_u32 s15, s15, s19 // (op 561) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 561) 

s_mov_b32 s40, s4 // (op 561) reg expression
s_mov_b32 s41, s5 // (op 561) reg expression
s_mov_b32 s42, s0 // (op 561) reg expression
s_mov_b32 s43, 131072 // (op 561) call()

s_lshl_b32 s0, s16, 4 // (op 578) 
v_lshlrev_b32 v13, 2, v1 // (op 578) 
s_mov_b32 s15, 0 // (op 578) convert
s_mov_b32 s14, s0 // (op 578) convert

v_mul_lo_u32 v15, v13, s3 // (op 578) most significant: low of low * high
v_mul_hi_u32 v16, v13, s2 // (op 578) most significant: high of low * low
v_mul_lo_u32 v14, v13, s2 // (op 578) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 578) most significant: sum


v_mov_b32 v13, s14 // (op 578) 
v_mov_b32 v16, s15 // (op 578) 
v_add_co_u32 v18, s[0:1], v13, v14 // (op 578) least significant half
v_addc_co_u32 v19, s[0:1], v16, v15, s[0:1] // (op 578) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 578) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 578) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 578) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 578) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 578) 

v_mov_b32 v13, v14 // (op 578) convert

s_mul_i32 s1, s7, 6 // (op 580) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 580) least significant
s_mul_hi_u32 s14, s6, 6 // (op 580) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 580) most significant: sum

s_mul_i32 s15, s2, s1 // (op 580) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 580) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 580) least significant
s_mul_hi_u32 s19, s2, s0 // (op 580) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 580) most significant: sum
s_add_u32 s15, s15, s19 // (op 580) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 580) 

s_mov_b32 s40, s4 // (op 580) reg expression
s_mov_b32 s41, s5 // (op 580) reg expression
s_mov_b32 s42, s0 // (op 580) reg expression
s_mov_b32 s43, 131072 // (op 580) call()

v_and_b32 v14, -4, v1 // (op 623) 
v_sub_u32 v15, v1, v14 // (op 623) 

v_lshlrev_b32 v14, 6, v15 // (op 623) 

v_mov_b32 v17, 0 // (op 623) convert
v_mov_b32 v16, v14 // (op 623) convert

v_lshrrev_b32 v14, 2, v1 // (op 623) 
v_lshl_add_u32 v15, s17, 4, v14 // (op 623) 

v_mul_lo_u32 v19, v15, s9 // (op 623) most significant: low of low * high
v_mul_hi_u32 v14, v15, s8 // (op 623) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 623) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 623) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 623) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 623) most significant half



v_mul_lo_u32 v17, v15, 6 // (op 623) most significant: low of high * low
v_mul_hi_u32 v18, v14, 6 // (op 623) most significant: high of low * low
v_mul_lo_u32 v16, v14, 6 // (op 623) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 623) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 623) 

v_mov_b32 v16, v14 // (op 623) convert

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

v_and_b32 v14, -4, v1 // (op 642) 
v_sub_u32 v15, v1, v14 // (op 642) 

v_lshlrev_b32 v14, 6, v15 // (op 642) 

v_mov_b32 v19, 0 // (op 642) convert
v_mov_b32 v18, v14 // (op 642) convert

v_lshrrev_b32 v17, 2, v1 // (op 642) 
v_lshl_add_u32 v14, s17, 4, v17 // (op 642) 

v_mul_lo_u32 v21, v14, s9 // (op 642) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 642) most significant: high of low * low
v_mul_lo_u32 v20, v14, s8 // (op 642) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 642) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 642) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 642) most significant half



v_mul_lo_u32 v19, v15, 6 // (op 642) most significant: low of high * low
v_mul_hi_u32 v17, v14, 6 // (op 642) most significant: high of low * low
v_mul_lo_u32 v18, v14, 6 // (op 642) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 642) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[18:19] // (op 642) 

v_mov_b32 v17, v14 // (op 642) convert

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

v_and_b32 v14, -16, v1 // (op 687) 
v_sub_u32 v15, v1, v14 // (op 687) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 687) 

v_lshrrev_b32 v15, 4, v1 // (op 687) 
v_lshlrev_b32 v18, 1, v15 // (op 687) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v15, v18, s20 // (op 687) 

v_add_u32 v18, v14, v15 // (op 687) 


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

v_and_b32 v14, -16, v1 // (op 706) 
v_sub_u32 v15, v1, v14 // (op 706) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 706) 

v_lshrrev_b32 v15, 4, v1 // (op 706) 
v_lshlrev_b32 v19, 1, v15 // (op 706) 

v_mul_lo_u32 v15, v19, s20 // (op 706) 

v_add_u32 v19, v14, v15 // (op 706) 


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

v_and_b32 v14, -8, v1 // (op 751) 
v_sub_u32 v15, v1, v14 // (op 751) 

v_lshrrev_b32 v14, 3, v1 // (op 751) 
v_lshlrev_b32 v20, 1, v14 // (op 751) 

v_lshl_add_u32 v14, s17, 4, v20 // (op 751) 

v_mul_lo_u32 v20, v14, s26 // (op 751) 

v_add_u32 v14, v15, v20 // (op 751) 


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

v_and_b32 v15, -8, v1 // (op 770) 
v_sub_u32 v20, v1, v15 // (op 770) 

v_lshrrev_b32 v15, 3, v1 // (op 770) 
v_lshlrev_b32 v21, 1, v15 // (op 770) 

v_lshl_add_u32 v15, s17, 4, v21 // (op 770) 

v_mul_lo_u32 v21, v15, s26 // (op 770) 

v_add_u32 v15, v20, v21 // (op 770) 


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

v_mov_b32 v34, v14 // (op 427) 
buffer_load_ubyte v36, v34, s[52:55], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v34, v34, v32 // (op 427) 
buffer_load_ubyte v37, v34, s[52:55], 0 offen offset: 0 // (op 427) Load value



s_mul_i32 s1, s3, 6 // (op 430) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 430) least significant
s_mul_hi_u32 s14, s2, 6 // (op 430) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 430) most significant: sum

v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 430) 


v_mov_b32 v34, v12 // (op 430) reg expression
buffer_load_dwordx3 v[38:40], v34, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v34, v34, v32 // (op 430) 
buffer_load_dwordx3 v[42:44], v34, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v34, v34, v32 // (op 430) 
buffer_load_dwordx3 v[46:48], v34, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v34, v34, v32 // (op 430) 
buffer_load_dwordx3 v[50:52], v34, s[40:43], 0 offen // (op 430) Load value



v_mov_b64 v[32:33], s[20:21] // (op 433) convert

v_mov_b32 v41, v18 // (op 433) reg expression
buffer_load_ubyte v34, v41, s[48:51], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v41, v41, v32 // (op 433) 
buffer_load_ubyte v35, v41, s[48:51], 0 offen offset: 0 // (op 433) Load value





buffer_load_dwordx4 v[54:57], v16, s[44:47], 0 offen offset:192 // (op 436) Load value
buffer_load_dwordx2 v[58:59], v16, s[44:47], 0 offen offset:208 // (op 436) Load value
buffer_load_dwordx4 v[60:63], v16, s[44:47], 0 offen offset:216 // (op 436) Load value
buffer_load_dwordx2 v[64:65], v16, s[44:47], 0 offen offset:232 // (op 436) Load value


v_mov_b64 v[32:33], s[26:27] // (op 439) convert

v_mov_b32 v16, v14 // (op 439) 
buffer_load_ubyte v66, v16, s[52:55], 0 offen offset:8 // (op 439) Load value
v_add_u32 v16, v16, v32 // (op 439) 
buffer_load_ubyte v67, v16, s[52:55], 0 offen offset:8 // (op 439) Load value




s_mul_i32 s1, s3, 6 // (op 442) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 442) least significant
s_mul_hi_u32 s14, s2, 6 // (op 442) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 442) most significant: sum

v_lshrrev_b64 v[32:33], 3, s[0:1] // (op 442) 


s_lshl_b64 s[0:1], s[2:3], 8 // (op 442) 
s_mul_i32 s15, s1, 6 // (op 442) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 442) least significant
s_mul_hi_u32 s18, s0, 6 // (op 442) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 442) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 442) 

v_add_u32 v14, v12, s0 // (op 442) 

buffer_load_dwordx3 v[68:70], v14, s[40:43], 0 offen // (op 442) Load value
v_add_u32 v14, v14, v32 // (op 442) 
buffer_load_dwordx3 v[72:74], v14, s[40:43], 0 offen // (op 442) Load value
v_add_u32 v14, v14, v32 // (op 442) 
buffer_load_dwordx3 v[76:78], v14, s[40:43], 0 offen // (op 442) Load value
v_add_u32 v14, v14, v32 // (op 442) 
buffer_load_dwordx3 v[80:82], v14, s[40:43], 0 offen // (op 442) Load value




v_mov_b64 v[32:33], s[20:21] // (op 445) convert

s_lshl_b32 s0, s20, 3 // (op 445) 
v_add_u32 v12, v18, s0 // (op 445) 

buffer_load_ubyte v84, v12, s[48:51], 0 offen offset: 0 // (op 445) Load value
v_add_u32 v12, v12, v32 // (op 445) 
buffer_load_ubyte v85, v12, s[48:51], 0 offen offset: 0 // (op 445) Load value



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

v_lshlrev_b32 v12, 6, v1 // (op 837) 
v_mul_lo_u32 v14, v12, 6 // (op 837) 
v_lshrrev_b32 v16, 3, v14 // (op 837) 

v_lshrrev_b32 v14, 4, v12 // (op 837) 

v_lshlrev_b32 v12, 2, v14 // (op 837) 

v_add_u32 v14, v16, v12 // (op 837) 





s_waitcnt vmcnt(17)
ds_write_b96 v14, v[38:40] offset:3200 // (op 449) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v14, v[42:44] offset:3216 // (op 449) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v14, v[46:48] offset:3232 // (op 449) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v14, v[50:52] offset:3248 // (op 449) Store local data 

v_and_b32 v12, -16, v1 // (op 943) 
v_sub_u32 v14, v1, v12 // (op 943) 

v_lshrrev_b32 v12, 4, v1 // (op 943) 
v_lshl_add_u32 v16, v12, 5, v14 // (op 943) 





s_waitcnt vmcnt(13)
ds_write_b8 v16, v34 offset:7296 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v16, v35 offset:7312 // (op 450) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1194) 



ds_read_b128 v[86:89], v4 // (op 467) Load local data 
ds_read_b64 v[90:91], v4 offset:16 // (op 467) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1235) 



ds_read_u8 v12, v6 offset:3072 // (op 473) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1175) 



ds_read_b96_tr_b6 v[92:94], v0 offset:3200 // (op 464) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v0 offset:3456 // (op 464) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1215) 


ds_read_u8 v14, v5 offset:7296 // (op 470) Load local data 
v_lshlrev_b32 v16, 6, v1 // (op 808) 
v_mul_lo_u32 v18, v16, 6 // (op 808) 
v_lshrrev_b32 v41, 3, v18 // (op 808) 

v_lshrrev_b32 v18, 4, v16 // (op 808) 

v_lshlrev_b32 v16, 2, v18 // (op 808) 

v_add_u32 v18, v41, v16 // (op 808) 


v_and_b32 v16, -4, v1 // (op 861) 
v_sub_u32 v41, v1, v16 // (op 861) 

v_lshlrev_b32 v16, 6, v41 // (op 861) 

v_lshrrev_b32 v41, 2, v1 // (op 861) 
v_lshl_add_u32 v45, v41, 8, v16 // (op 861) 


v_mul_lo_u32 v16, v45, 6 // (op 861) 

v_lshrrev_b32 v41, 3, v16 // (op 861) 

v_and_b32 v16, -16, v1 // (op 914) 
v_sub_u32 v45, v1, v16 // (op 914) 

v_lshrrev_b32 v16, 4, v1 // (op 914) 
v_lshl_add_u32 v49, v16, 5, v45 // (op 914) 


v_and_b32 v16, -8, v1 // (op 967) 
v_sub_u32 v45, v1, v16 // (op 967) 

v_lshrrev_b32 v16, 3, v1 // (op 967) 
v_lshl_add_u32 v53, v16, 4, v45 // (op 967) 


v_lshlrev_b32 v16, 6, v1 // (op 1054) 
v_mul_lo_u32 v45, v16, 6 // (op 1054) 
v_lshrrev_b32 v71, 3, v45 // (op 1054) 

v_lshrrev_b32 v45, 4, v16 // (op 1054) 

v_lshlrev_b32 v16, 2, v45 // (op 1054) 

v_add_u32 v45, v71, v16 // (op 1054) 


v_and_b32 v16, -4, v1 // (op 1078) 
v_sub_u32 v71, v1, v16 // (op 1078) 

v_lshlrev_b32 v16, 6, v71 // (op 1078) 

v_lshrrev_b32 v71, 2, v1 // (op 1078) 
v_lshl_add_u32 v75, v71, 8, v16 // (op 1078) 


v_mul_lo_u32 v16, v75, 6 // (op 1078) 

v_lshrrev_b32 v71, 3, v16 // (op 1078) 

v_and_b32 v16, -16, v1 // (op 1102) 
v_sub_u32 v75, v1, v16 // (op 1102) 

v_lshrrev_b32 v16, 4, v1 // (op 1102) 
v_lshl_add_u32 v79, v16, 5, v75 // (op 1102) 


v_and_b32 v16, -8, v1 // (op 1126) 
v_sub_u32 v75, v1, v16 // (op 1126) 

v_lshrrev_b32 v16, 3, v1 // (op 1126) 
v_lshl_add_u32 v83, v16, 4, v75 // (op 1126) 


s_mov_b32 s0, 0 // (op 59) call()
s_ashr_i64 s[14:15], s[6:7], 8 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58:


s_add_u32 s0, s0, 2 // (op 60) 
v_mov_b32 v16, v17 // (op 10) 

buffer_load_dwordx4 v[20:23], v16, s[44:47], 0 offen offset:384 // (op 10) Load value
buffer_load_dwordx2 v[24:25], v16, s[44:47], 0 offen offset:400 // (op 10) Load value
buffer_load_dwordx4 v[26:29], v16, s[44:47], 0 offen offset:408 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:97], v[86:91], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 88) 
buffer_load_dwordx2 v[30:31], v16, s[44:47], 0 offen offset:424 // (op 10) Load value

s_barrier  // (op 1184) 
v_mov_b64 v[32:33], s[26:27] // (op 25) convert




ds_read_b96_tr_b6 v[92:94], v0 offset:5248 // (op 91) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v0 offset:5504 // (op 91) Transpose load local data 
v_mov_b32 v75, v15 // (op 25) 
s_barrier  // (op 1204) 
v_mov_b32 v98, v75 // (op 25) 



ds_read_b128 v[86:89], v4 offset:96 // (op 95) Load local data 
buffer_load_ubyte v36, v98, s[52:55], 0 offen offset:16 // (op 25) Load value
v_add_u32 v98, v98, v32 // (op 25) 
buffer_load_ubyte v37, v98, s[52:55], 0 offen offset:16 // (op 25) Load value


ds_read_b64 v[90:91], v4 offset:112 // (op 95) Load local data 

s_barrier  // (op 1225) 
s_mul_i32 s15, s3, 6 // (op 4) most significant: low of high * low


s_mul_i32 s14, s2, 6 // (op 4) least significant
ds_read_u8 v14, v5 offset:7360 // (op 99) Load local data 
s_mul_hi_u32 s1, s2, 6 // (op 4) most significant: high of low * low
s_barrier  // (op 1245) 
s_add_u32 s15, s15, s1 // (op 4) most significant: sum




ds_read_u8 v12, v6 offset:3076 // (op 103) Load local data 
v_lshrrev_b64 v[32:33], 3, s[14:15] // (op 4) 


v_mov_b32 v98, v13 // (op 4) 
s_lshl_b64 s[14:15], s[2:3], 8 // (op 4) 
s_mul_i32 s19, s15, 6 // (op 4) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 4) least significant
s_mul_hi_u32 s1, s14, 6 // (op 4) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 4) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s1, s14, 1 // (op 4) 

v_add_u32 v99, v98, s1 // (op 4) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[92:97], v[86:91], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 105) 
buffer_load_dwordx3 v[38:40], v99, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v99, v99, v32 // (op 4) 
buffer_load_dwordx3 v[42:44], v99, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v99, v99, v32 // (op 4) 
buffer_load_dwordx3 v[46:48], v99, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v99, v99, v32 // (op 4) 
buffer_load_dwordx3 v[50:52], v99, s[40:43], 0 offen // (op 4) Load value



v_mov_b64 v[32:33], s[20:21] // (op 17) convert

v_mov_b32 v99, v19 // (op 17) 
s_lshl_b32 s1, s20, 4 // (op 17) 
v_add_u32 v100, v99, s1 // (op 17) 

buffer_load_ubyte v34, v100, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v100, v100, v32 // (op 17) 
buffer_load_ubyte v35, v100, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1265) 



s_waitcnt vmcnt(23)
ds_write_b128 v71, v[54:57] offset:7424 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v71, v[58:59] offset:7440 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v71, v[60:63] offset:7448 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v71, v[64:65] offset:7464 // (op 302) Store local data 
s_barrier  // (op 1277) 



s_waitcnt vmcnt(19)
ds_write_b8 v83, v66 offset:10496 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v83, v67 offset:10504 // (op 304) Store local data 
s_barrier  // (op 1258) 



s_waitcnt vmcnt(17)
ds_write_b96 v45, v[68:70] offset:10624 // (op 283) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v45, v[72:74] offset:10640 // (op 283) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v45, v[76:78] offset:10656 // (op 283) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v45, v[80:82] offset:10672 // (op 283) Store local data 
s_barrier  // (op 1271) 



s_waitcnt vmcnt(13)
ds_write_b8 v79, v84 offset:14720 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v79, v85 offset:14736 // (op 303) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1261) 



ds_read_b128 v[100:103], v8 offset:7424 // (op 287) Load local data 
ds_read_b64 v[104:105], v8 offset:7440 // (op 287) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1274) 



ds_read_u8 v32, v10 offset:10496 // (op 291) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1255) 



ds_read_b96_tr_b6 v[106:108], v3 offset:10624 // (op 285) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v3 offset:10880 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1268) 


ds_read_u8 v33, v9 offset:14720 // (op 289) Load local data 



buffer_load_dwordx4 v[54:57], v16, s[44:47], 0 offen offset:576 // (op 305) Load value
buffer_load_dwordx2 v[58:59], v16, s[44:47], 0 offen offset:592 // (op 305) Load value
buffer_load_dwordx4 v[60:63], v16, s[44:47], 0 offen offset:600 // (op 305) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[106:111], v[100:105], a[0:3], v33, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 292) 
buffer_load_dwordx2 v[64:65], v16, s[44:47], 0 offen offset:616 // (op 305) Load value









v_add_u32 v17, 384, v17 // (op 634) 
v_mov_b64 v[32:33], s[26:27] // (op 307) convert

v_mov_b32 v16, v75 // (op 307) 
buffer_load_ubyte v66, v16, s[52:55], 0 offen offset:24 // (op 307) Load value
v_add_u32 v16, v16, v32 // (op 307) 
buffer_load_ubyte v67, v16, s[52:55], 0 offen offset:24 // (op 307) Load value


ds_read_b96_tr_b6 v[100:102], v3 offset:12672 // (op 294) Transpose load local data 

ds_read_b96_tr_b6 v[103:105], v3 offset:12928 // (op 294) Transpose load local data 

v_add_u32 v15, 16, v15 // (op 762) 



s_barrier  // (op 1336) 
s_mul_i32 s15, s3, 6 // (op 282) most significant: low of high * low
s_mul_i32 s14, s2, 6 // (op 282) least significant
s_mul_hi_u32 s1, s2, 6 // (op 282) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 282) most significant: sum

v_lshrrev_b64 v[32:33], 3, s[14:15] // (op 282) 


s_lshl_b64 s[14:15], s[2:3], 8 // (op 282) 
s_mul_i32 s19, s15, 6 // (op 282) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 282) least significant
s_mul_hi_u32 s1, s14, 6 // (op 282) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 282) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 282) 

s_mul_i32 s1, 3, s14 // (op 282) 

v_add_u32 v16, v98, s1 // (op 282) 

ds_read_b128 v[106:109], v8 offset:7520 // (op 296) Load local data 
ds_read_b64 v[110:111], v8 offset:7536 // (op 296) Load local data 
buffer_load_dwordx3 v[68:70], v16, s[40:43], 0 offen // (op 282) Load value
v_add_u32 v16, v16, v32 // (op 282) 


s_barrier  // (op 1340) 
ds_read_u8 v75, v9 offset:14784 // (op 298) Load local data 
buffer_load_dwordx3 v[72:74], v16, s[40:43], 0 offen // (op 282) Load value
v_add_u32 v16, v16, v32 // (op 282) 



s_barrier  // (op 1344) 
buffer_load_dwordx3 v[76:78], v16, s[40:43], 0 offen // (op 282) Load value
v_add_u32 v16, v16, v32 // (op 282) 
ds_read_u8 v112, v10 offset:10500 // (op 300) Load local data 
buffer_load_dwordx3 v[80:82], v16, s[40:43], 0 offen // (op 282) Load value




s_barrier  // (op 1348) 
v_mov_b64 v[32:33], s[20:21] // (op 306) convert

s_lshl_b32 s1, s20, 3 // (op 306) 
s_mul_i32 s14, 3, s1 // (op 306) 

v_add_u32 v16, v99, s14 // (op 306) 

buffer_load_ubyte v84, v16, s[48:51], 0 offen offset: 0 // (op 306) Load value
v_add_u32 v16, v16, v32 // (op 306) 
buffer_load_ubyte v85, v16, s[48:51], 0 offen offset: 0 // (op 306) Load value


s_lshl_b64 s[14:15], s[2:3], 8 // (op 570) 

s_mul_i32 s19, s15, 6 // (op 570) most significant: low of high * low
s_barrier  // (op 1197) 
s_mul_i32 s18, s14, 6 // (op 570) least significant



s_mul_hi_u32 s1, s14, 6 // (op 570) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 570) most significant: sum


s_lshl_b32 s1, s20, 4 // (op 698) 
v_add_u32 v19, v19, s1 // (op 698) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 570) 

s_lshl_b32 s1, s14, 1 // (op 570) 

v_add_u32 v13, v13, s1 // (op 570) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:105], v[106:111], a[0:3], v75, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 301) 
s_waitcnt vmcnt(23)
ds_write_b128 v41, v[20:23] // (op 13) Store local data 




s_waitcnt vmcnt(22)
ds_write_b64 v41, v[24:25] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v41, v[26:29] offset:24 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v41, v[30:31] offset:40 // (op 13) Store local data 
s_barrier  // (op 1238) 



s_waitcnt vmcnt(19)
ds_write_b8 v53, v36 offset:3072 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v53, v37 offset:3080 // (op 30) Store local data 
s_barrier  // (op 1178) 



s_waitcnt vmcnt(17)
ds_write_b96 v18, v[38:40] offset:3200 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v18, v[42:44] offset:3216 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v18, v[46:48] offset:3232 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v18, v[50:52] offset:3248 // (op 6) Store local data 
s_barrier  // (op 1218) 



s_waitcnt vmcnt(13)
ds_write_b8 v49, v34 offset:7296 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v49, v35 offset:7312 // (op 21) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1207) 



ds_read_b128 v[86:89], v4 // (op 75) Load local data 
ds_read_b64 v[90:91], v4 offset:16 // (op 75) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1248) 



ds_read_u8 v12, v6 offset:3072 // (op 85) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1187) 



ds_read_b96_tr_b6 v[92:94], v0 offset:3200 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[95:97], v0 offset:3456 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1228) 


ds_read_u8 v14, v5 offset:7296 // (op 80) Load local data 
s_ashr_i64 s[14:15], s[6:7], 8 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58:

















v_lshlrev_b32 v13, 2, v1 // (op 533) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 533) 
s_lshr_b32 s14, s0, 1 // (op 533) 

v_lshl_add_u32 v71, s14, 9, v13 // (op 533) 


s_lshl_b32 s0, s16, 4 // (op 533) 
v_mul_lo_u32 v53, v71, s3 // (op 533) most significant: low of low * high
v_mul_hi_u32 v13, v71, s2 // (op 533) most significant: high of low * low
v_mul_lo_u32 v52, v71, s2 // (op 533) least significant: low of low * low
v_add_u32 v53, v53, v13 // (op 533) most significant: sum


s_mov_b32 s15, 0 // (op 533) convert
s_mov_b32 s14, s0 // (op 533) convert

v_mov_b32 v13, s14 // (op 533) 
v_mov_b32 v71, s15 // (op 533) 
v_add_co_u32 v50, s[0:1], v13, v52 // (op 533) least significant half
v_addc_co_u32 v51, s[0:1], v71, v53, s[0:1] // (op 533) most significant half





v_mul_lo_u32 v53, v51, 6 // (op 533) most significant: low of high * low
v_mul_hi_u32 v13, v50, 6 // (op 533) most significant: high of low * low
v_mul_lo_u32 v52, v50, 6 // (op 533) least significant: low of low * low
v_add_u32 v53, v53, v13 // (op 533) most significant: sum


v_lshrrev_b64 v[50:51], 3, v[52:53] // (op 533) 

v_mov_b32 v13, v50 // (op 533) convert

s_mul_i32 s1, s7, 6 // (op 535) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 535) least significant
s_mul_hi_u32 s14, s6, 6 // (op 535) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 535) most significant: sum

s_mul_i32 s15, s2, s1 // (op 535) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 535) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 535) least significant
s_mul_hi_u32 s19, s2, s0 // (op 535) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 535) most significant: sum
s_add_u32 s15, s15, s19 // (op 535) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 535) 

s_mov_b32 s40, s4 // (op 535) reg expression
s_mov_b32 s41, s5 // (op 535) reg expression
s_mov_b32 s42, s0 // (op 535) reg expression
s_mov_b32 s43, 131072 // (op 535) call()

v_and_b32 v71, -4, v1 // (op 600) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 600) 
v_sub_u32 v75, v1, v71 // (op 600) 

s_lshr_b32 s14, s0, 1 // (op 600) 

v_lshlrev_b32 v71, 6, v75 // (op 600) 

v_lshl_add_u32 v75, s14, 9, v71 // (op 600) 


v_mov_b32 v53, 0 // (op 600) convert
v_mov_b32 v52, v75 // (op 600) convert

v_lshrrev_b32 v71, 2, v1 // (op 600) 
v_lshl_add_u32 v75, s17, 4, v71 // (op 600) 

v_mul_lo_u32 v51, v75, s9 // (op 600) most significant: low of low * high
v_mul_hi_u32 v71, v75, s8 // (op 600) most significant: high of low * low
v_mul_lo_u32 v50, v75, s8 // (op 600) least significant: low of low * low
v_add_u32 v51, v51, v71 // (op 600) most significant: sum


v_add_co_u32 v48, s[0:1], v52, v50 // (op 600) least significant half
v_addc_co_u32 v49, s[0:1], v53, v51, s[0:1] // (op 600) most significant half



v_mul_lo_u32 v47, v49, 6 // (op 600) most significant: low of high * low
v_mul_hi_u32 v71, v48, 6 // (op 600) most significant: high of low * low
v_mul_lo_u32 v46, v48, 6 // (op 600) least significant: low of low * low
v_add_u32 v47, v47, v71 // (op 600) most significant: sum


v_lshrrev_b64 v[44:45], 3, v[46:47] // (op 600) 

v_mov_b32 v71, v44 // (op 600) convert

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

v_and_b32 v75, -16, v1 // (op 664) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 664) 
v_sub_u32 v79, v1, v75 // (op 664) 

s_lshr_b32 s14, s0, 1 // (op 664) 

v_lshl_add_u32 v75, s16, 4, v79 // (op 664) 

v_lshrrev_b32 v79, 4, v1 // (op 664) 
v_lshlrev_b32 v83, 1, v79 // (op 664) 

v_lshl_add_u32 v79, s14, 4, v83 // (op 664) 


v_mul_lo_u32 v83, v79, s20 // (op 664) 

v_add_u32 v79, v75, v83 // (op 664) 


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

v_and_b32 v75, -8, v1 // (op 728) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 728) 
v_sub_u32 v83, v1, v75 // (op 728) 

s_lshr_b32 s14, s0, 1 // (op 728) 

v_lshrrev_b32 v75, 3, v1 // (op 728) 
v_lshl_add_u32 v15, s14, 4, v83 // (op 728) 


v_lshlrev_b32 v83, 1, v75 // (op 728) 

v_lshl_add_u32 v75, s17, 4, v83 // (op 728) 

v_mul_lo_u32 v83, v75, s26 // (op 728) 

v_add_u32 v75, v15, v83 // (op 728) 


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

v_lshlrev_b32 v83, 6, v1 // (op 821) 
v_mul_lo_u32 v15, v83, 6 // (op 821) 
v_lshrrev_b32 v16, 3, v15 // (op 821) 

v_lshrrev_b32 v15, 4, v83 // (op 821) 

v_lshlrev_b32 v83, 2, v15 // (op 821) 

v_add_u32 v15, v16, v83 // (op 821) 


v_and_b32 v83, -4, v1 // (op 874) 
v_sub_u32 v16, v1, v83 // (op 874) 

v_lshlrev_b32 v83, 6, v16 // (op 874) 

v_lshrrev_b32 v16, 2, v1 // (op 874) 
v_lshl_add_u32 v17, v16, 8, v83 // (op 874) 


v_mul_lo_u32 v16, v17, 6 // (op 874) 

v_lshrrev_b32 v83, 3, v16 // (op 874) 

v_and_b32 v16, -16, v1 // (op 927) 
v_sub_u32 v17, v1, v16 // (op 927) 

v_lshrrev_b32 v16, 4, v1 // (op 927) 
v_lshl_add_u32 v18, v16, 5, v17 // (op 927) 


v_and_b32 v16, -8, v1 // (op 980) 
v_sub_u32 v17, v1, v16 // (op 980) 

v_lshrrev_b32 v16, 3, v1 // (op 980) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 980) 


s_ashr_i64 s[0:1], s[6:7], 8 // (op 194) 
s_and_b32 s14, s0, 4294967294 // (op 194) 

s_ashr_i64 s[0:1], s[6:7], 8 // (op 193) 
s_mov_b32 s15, s0 // (op 193) convert

s_cmp_lt_u32 s14, s15 // (op 193) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193:

s_add_u32 s14, s14, 1 // (op 195) 
s_mul_i32 s1, s3, 6 // (op 198) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 198) least significant
s_mul_hi_u32 s15, s2, 6 // (op 198) most significant: high of low * low
s_add_u32 s1, s1, s15 // (op 198) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 198) 


v_mov_b32 v12, v13 // (op 198) 
v_mov_b32 v14, v12 // (op 198) reg expression
buffer_load_dwordx3 v[20:22], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v16 // (op 198) 
buffer_load_dwordx3 v[24:26], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v16 // (op 198) 
buffer_load_dwordx3 v[28:30], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v16 // (op 198) 
buffer_load_dwordx3 v[32:34], v14, s[40:43], 0 offen // (op 198) Load value



s_lshl_b64 s[0:1], s[2:3], 8 // (op 523) 
s_mul_i32 s19, s1, 6 // (op 523) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 523) least significant
s_mul_hi_u32 s15, s0, 6 // (op 523) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 523) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 523) 

v_add_u32 v13, v13, s0 // (op 523) 



s_barrier  // (op 1181) 
v_mov_b32 v12, v71 // (op 221) 




buffer_load_dwordx4 v[36:39], v12, s[44:47], 0 offen // (op 221) Load value
buffer_load_dwordx2 v[40:41], v12, s[44:47], 0 offen offset:16 // (op 221) Load value
s_waitcnt vmcnt(5)
ds_write_b96 v15, v[20:22] offset:3200 // (op 199) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v15, v[24:26] offset:3216 // (op 199) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v15, v[28:30] offset:3232 // (op 199) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v15, v[32:34] offset:3248 // (op 199) Store local data 
buffer_load_dwordx4 v[42:45], v12, s[44:47], 0 offen offset:24 // (op 221) Load value

buffer_load_dwordx2 v[46:47], v12, s[44:47], 0 offen offset:40 // (op 221) Load value

v_add_u32 v71, 192, v71 // (op 592) 

s_barrier  // (op 1200) 
v_mov_b64 v[16:17], s[20:21] // (op 222) convert




v_mov_b32 v12, v79 // (op 222) 
v_mov_b32 v14, v12 // (op 222) reg expression
buffer_load_ubyte v20, v14, s[48:51], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v14, v14, v16 // (op 222) 
s_waitcnt vmcnt(4)
ds_write_b128 v83, v[36:39] // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v83, v[40:41] offset:16 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v83, v[42:45] offset:24 // (op 218) Store local data 
buffer_load_ubyte v21, v14, s[48:51], 0 offen offset: 0 // (op 222) Load value



s_lshl_b32 s15, s20, 3 // (op 656) 
v_add_u32 v79, v79, s15 // (op 656) 

s_waitcnt vmcnt(2)
ds_write_b64 v83, v[46:47] offset:40 // (op 218) Store local data 


v_mov_b64 v[16:17], s[26:27] // (op 223) convert

s_barrier  // (op 1221) 
v_mov_b32 v12, v75 // (op 223) 



v_mov_b32 v14, v12 // (op 223) 
buffer_load_ubyte v22, v14, s[52:55], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v14, v14, v16 // (op 223) 
buffer_load_ubyte v23, v14, s[52:55], 0 offen offset: 0 // (op 223) Load value



v_add_u32 v75, 8, v75 // (op 720) 
s_waitcnt vmcnt(3)
ds_write_b8 v18, v20 offset:7296 // (op 219) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1190) 
s_waitcnt vmcnt(2)
ds_write_b8 v18, v21 offset:7312 // (op 219) Store local data 




ds_read_b96_tr_b6 v[24:26], v0 offset:3200 // (op 201) Transpose load local data 
s_barrier  // (op 1241) 
ds_read_b96_tr_b6 v[27:29], v0 offset:3456 // (op 201) Transpose load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 1211) 
s_waitcnt vmcnt(1)
ds_write_b8 v19, v22 offset:3072 // (op 220) Store local data 



ds_read_b128 v[30:33], v4 // (op 203) Load local data 
ds_read_b64 v[34:35], v4 offset:16 // (op 203) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v23 offset:3080 // (op 220) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1231) 


ds_read_u8 v12, v5 offset:7296 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1251) 



ds_read_u8 v14, v6 offset:3072 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:35], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 208) 







ds_read_b96_tr_b6 v[20:22], v0 offset:5248 // (op 210) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v0 offset:5504 // (op 210) Transpose load local data 



ds_read_b128 v[26:29], v4 offset:96 // (op 212) Load local data 
ds_read_b64 v[30:31], v4 offset:112 // (op 212) Load local data 


ds_read_u8 v12, v5 offset:7360 // (op 214) Load local data 



ds_read_u8 v14, v6 offset:3076 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 217) 




s_ashr_i64 s[0:1], s[6:7], 8 // (op 193) 
s_mov_b32 s15, s0 // (op 193) convert

s_cmp_lt_u32 s14, s15 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 if true)
RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1328) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 1022) 
v_and_b32 v3, 63, v1 // (op 1022) 
v_lshrrev_b32 v4, 4, v3 // (op 1022) 
v_lshlrev_b32 v5, 2, v4 // (op 1022) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1022) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1022) 

v_lshrrev_b32 v4, 2, v3 // (op 1022) 

v_and_b32 v3, 3, v4 // (op 1022) 

v_and_b32 v4, 3, v1 // (op 1022) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1022) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1022) 

v_mul_lo_u32 v3, v6, s30 // (op 1022) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 1022) 


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




buffer_store_dwordx4 v[16:19], v4, s[8:11], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155:
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


v_lshl_add_u32 v6, s17, 4, v5 // (op 791) 

v_mul_lo_u32 v3, v6, s34 // (op 791) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 791) 


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


v_lshrrev_b32 v0, 6, v1 // (op 1039) 
v_and_b32 v3, 63, v1 // (op 1039) 
v_lshrrev_b32 v4, 4, v3 // (op 1039) 
v_lshlrev_b32 v5, 2, v4 // (op 1039) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1039) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1039) 

v_lshrrev_b32 v4, 2, v3 // (op 1039) 

v_and_b32 v3, 3, v4 // (op 1039) 

v_and_b32 v4, 3, v1 // (op 1039) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1039) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1039) 

v_mul_lo_u32 v3, v6, s30 // (op 1039) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 1039) 


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




buffer_store_dwordx4 v[12:15], v4, s[8:11], 0 offen // (op 166) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155:




s_endpgm  // End of RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 120
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 14848 // lds bytes
  .amdhsa_accum_offset 116
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F6
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
  - .name: RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NN_FP6_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 14848
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 113
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

