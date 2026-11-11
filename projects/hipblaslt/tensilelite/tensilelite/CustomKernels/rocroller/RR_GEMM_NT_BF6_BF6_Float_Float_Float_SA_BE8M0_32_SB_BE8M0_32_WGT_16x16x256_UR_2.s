// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_and_b32 v3, 63, v1 // (op 902) 
v_and_b32 v4, 48, v1 // (op 902) 
v_sub_u32 v5, v3, v4 // (op 902) 

v_lshrrev_b32 v4, 4, v3 // (op 902) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 902) 


v_lshlrev_b32 v4, 4, v3 // (op 902) 
v_mul_lo_u32 v5, v4, 6 // (op 902) 

v_lshrrev_b32 v4, 3, v5 // (op 902) 

v_and_b32 v5, 268435455, v3 // (op 902) 

v_lshlrev_b32 v3, 2, v5 // (op 902) 

v_add_u32 v5, v4, v3 // (op 902) 


v_lshrrev_b32 v3, 6, v1 // (op 955) 
v_and_b32 v4, 15, v1 // (op 955) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 955) 


v_and_b32 v3, 48, v1 // (op 955) 
v_add_u32 v4, v3, v6 // (op 955) 


v_and_b32 v3, 48, v1 // (op 1008) 
v_and_b32 v6, 15, v1 // (op 1008) 
v_add_u32 v7, v3, v6 // (op 1008) 


v_and_b32 v3, 63, v1 // (op 1066) 
v_and_b32 v6, 48, v1 // (op 1066) 
v_sub_u32 v8, v3, v6 // (op 1066) 

v_lshrrev_b32 v6, 4, v3 // (op 1066) 

v_lshl_add_u32 v3, v6, 5, v8 // (op 1066) 


v_lshrrev_b32 v6, 6, v1 // (op 1066) 
v_lshlrev_b32 v8, 4, v6 // (op 1066) 

v_lshl_add_u32 v6, v3, 4, v8 // (op 1066) 


v_mul_lo_u32 v3, v6, 6 // (op 1066) 
v_lshrrev_b32 v8, 3, v3 // (op 1066) 

v_lshrrev_b32 v3, 4, v6 // (op 1066) 

v_lshlrev_b32 v6, 2, v3 // (op 1066) 

v_add_u32 v3, v8, v6 // (op 1066) 


v_and_b32 v6, 63, v1 // (op 1090) 
v_and_b32 v8, 48, v1 // (op 1090) 
v_sub_u32 v9, v6, v8 // (op 1090) 

v_lshrrev_b32 v8, 4, v6 // (op 1090) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 1090) 


v_lshlrev_b32 v8, 4, v6 // (op 1090) 
v_mul_lo_u32 v9, v8, 6 // (op 1090) 

v_lshrrev_b32 v8, 3, v9 // (op 1090) 

v_and_b32 v9, 268435455, v6 // (op 1090) 

v_lshlrev_b32 v6, 2, v9 // (op 1090) 

v_add_u32 v9, v8, v6 // (op 1090) 


v_lshrrev_b32 v6, 6, v1 // (op 1114) 
v_and_b32 v8, 15, v1 // (op 1114) 
v_lshl_add_u32 v10, v6, 4, v8 // (op 1114) 


v_and_b32 v6, 48, v1 // (op 1114) 
v_add_u32 v8, v6, v10 // (op 1114) 


v_and_b32 v6, 48, v1 // (op 1138) 
v_and_b32 v10, 15, v1 // (op 1138) 
v_add_u32 v11, v6, v10 // (op 1138) 


v_mov_b32 v6, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


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





v_mul_lo_u32 v15, v17, 6 // (op 623) most significant: low of high * low
v_mul_hi_u32 v18, v16, 6 // (op 623) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 623) least significant: low of low * low
v_add_u32 v15, v15, v18 // (op 623) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 623) 

v_mov_b32 v14, v16 // (op 623) convert

s_mul_i32 s1, s7, 6 // (op 625) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 625) least significant
s_mul_hi_u32 s14, s6, 6 // (op 625) most significant: high of low * low
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





v_mul_lo_u32 v19, v17, 6 // (op 642) most significant: low of high * low
v_mul_hi_u32 v15, v16, 6 // (op 642) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 642) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 642) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 642) 

v_mov_b32 v15, v16 // (op 642) convert

s_mul_i32 s1, s7, 6 // (op 644) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 644) least significant
s_mul_hi_u32 s14, s6, 6 // (op 644) most significant: high of low * low
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

v_and_b32 v16, -16, v1 // (op 687) 
v_sub_u32 v17, v1, v16 // (op 687) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 687) 

v_lshrrev_b32 v17, 4, v1 // (op 687) 
v_lshlrev_b32 v18, 1, v17 // (op 687) 

v_mul_lo_u32 v17, v18, s12 // (op 687) 

v_add_u32 v18, v16, v17 // (op 687) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 689) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 689) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 689) least significant
s_mul_hi_u32 s15, s12, s22 // (op 689) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 689) most significant: sum
s_add_u32 s1, s1, s15 // (op 689) most significant: sum


s_mov_b32 s48, s20 // (op 689) reg expression
s_mov_b32 s49, s21 // (op 689) reg expression
s_mov_b32 s50, s0 // (op 689) reg expression
s_mov_b32 s51, 131072 // (op 689) call()

v_and_b32 v16, -16, v1 // (op 706) 
v_sub_u32 v17, v1, v16 // (op 706) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 706) 

v_lshrrev_b32 v17, 4, v1 // (op 706) 
v_lshlrev_b32 v19, 1, v17 // (op 706) 

v_mul_lo_u32 v17, v19, s12 // (op 706) 

v_add_u32 v19, v16, v17 // (op 706) 


s_mul_i32 s1, s12, s23 // (op 708) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 708) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 708) least significant
s_mul_hi_u32 s15, s12, s22 // (op 708) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 708) most significant: sum
s_add_u32 s1, s1, s15 // (op 708) most significant: sum


s_mov_b32 s48, s20 // (op 708) reg expression
s_mov_b32 s49, s21 // (op 708) reg expression
s_mov_b32 s50, s0 // (op 708) reg expression
s_mov_b32 s51, 131072 // (op 708) call()

v_and_b32 v16, -16, v1 // (op 751) 
v_sub_u32 v17, v1, v16 // (op 751) 

v_lshl_add_u32 v16, s17, 4, v17 // (op 751) 

v_lshrrev_b32 v17, 4, v1 // (op 751) 
v_lshlrev_b32 v20, 1, v17 // (op 751) 

v_mul_lo_u32 v17, v20, s24 // (op 751) 
v_add_u32 v17, v17, v16 // (op 751) 


s_mul_i32 s1, s24, s23 // (op 753) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 753) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 753) least significant
s_mul_hi_u32 s15, s24, s22 // (op 753) most significant: high of low * low
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


s_mul_i32 s1, s24, s23 // (op 772) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 772) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 772) least significant
s_mul_hi_u32 s15, s24, s22 // (op 772) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 772) most significant: sum
s_add_u32 s1, s1, s15 // (op 772) most significant: sum


s_mov_b32 s52, s26 // (op 772) reg expression
s_mov_b32 s53, s27 // (op 772) reg expression
s_mov_b32 s54, s0 // (op 772) reg expression
s_mov_b32 s55, 131072 // (op 772) call()


s_mul_i32 s1, s9, 6 // (op 424) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 424) least significant
s_mul_hi_u32 s14, s8, 6 // (op 424) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 424) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 424) 


v_mov_b32 v16, v14 // (op 424) reg expression
buffer_load_dwordx3 v[24:26], v16, s[44:47], 0 offen // (op 424) Load value
v_add_u32 v16, v16, v22 // (op 424) 
buffer_load_dwordx3 v[28:30], v16, s[44:47], 0 offen // (op 424) Load value
v_add_u32 v16, v16, v22 // (op 424) 
buffer_load_dwordx3 v[32:34], v16, s[44:47], 0 offen // (op 424) Load value
v_add_u32 v16, v16, v22 // (op 424) 
buffer_load_dwordx3 v[36:38], v16, s[44:47], 0 offen // (op 424) Load value



v_mov_b64 v[22:23], s[24:25] // (op 427) convert

v_mov_b32 v16, v17 // (op 427) reg expression
buffer_load_ubyte v40, v16, s[52:55], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v16, v16, v22 // (op 427) 
buffer_load_ubyte v41, v16, s[52:55], 0 offen offset: 0 // (op 427) Load value



s_mul_i32 s1, s3, 6 // (op 430) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 430) least significant
s_mul_hi_u32 s14, s2, 6 // (op 430) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 430) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 430) 


v_mov_b32 v16, v12 // (op 430) reg expression
buffer_load_dwordx3 v[42:44], v16, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v16, v16, v22 // (op 430) 
buffer_load_dwordx3 v[46:48], v16, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v16, v16, v22 // (op 430) 
buffer_load_dwordx3 v[50:52], v16, s[40:43], 0 offen // (op 430) Load value
v_add_u32 v16, v16, v22 // (op 430) 
buffer_load_dwordx3 v[54:56], v16, s[40:43], 0 offen // (op 430) Load value



v_mov_b64 v[22:23], s[12:13] // (op 433) convert

v_mov_b32 v16, v18 // (op 433) reg expression
buffer_load_ubyte v58, v16, s[48:51], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v16, v16, v22 // (op 433) 
buffer_load_ubyte v59, v16, s[48:51], 0 offen offset: 0 // (op 433) Load value



s_mul_i32 s1, s9, 6 // (op 436) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 436) least significant
s_mul_hi_u32 s14, s8, 6 // (op 436) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 436) most significant: sum

v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 436) 


s_lshl_b64 s[0:1], s[8:9], 8 // (op 436) 
s_mul_i32 s15, s1, 6 // (op 436) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 436) least significant
s_mul_hi_u32 s18, s0, 6 // (op 436) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 436) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 436) 

v_add_u32 v16, v14, s0 // (op 436) 

buffer_load_dwordx3 v[60:62], v16, s[44:47], 0 offen // (op 436) Load value
v_add_u32 v16, v16, v22 // (op 436) 
buffer_load_dwordx3 v[64:66], v16, s[44:47], 0 offen // (op 436) Load value
v_add_u32 v16, v16, v22 // (op 436) 
buffer_load_dwordx3 v[68:70], v16, s[44:47], 0 offen // (op 436) Load value
v_add_u32 v16, v16, v22 // (op 436) 
buffer_load_dwordx3 v[72:74], v16, s[44:47], 0 offen // (op 436) Load value




v_mov_b64 v[22:23], s[24:25] // (op 439) convert

s_lshl_b32 s0, s24, 3 // (op 439) 
v_add_u32 v14, v17, s0 // (op 439) 

buffer_load_ubyte v76, v14, s[52:55], 0 offen offset: 0 // (op 439) Load value
v_add_u32 v14, v14, v22 // (op 439) 
buffer_load_ubyte v77, v14, s[52:55], 0 offen offset: 0 // (op 439) Load value




s_mul_i32 s1, s3, 6 // (op 442) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 442) least significant
s_mul_hi_u32 s14, s2, 6 // (op 442) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 442) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 442) 


s_lshl_b64 s[0:1], s[2:3], 8 // (op 442) 
s_mul_i32 s15, s1, 6 // (op 442) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 442) least significant
s_mul_hi_u32 s18, s0, 6 // (op 442) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 442) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 442) 

v_add_u32 v14, v12, s0 // (op 442) 

buffer_load_dwordx3 v[78:80], v14, s[40:43], 0 offen // (op 442) Load value
v_add_u32 v14, v14, v16 // (op 442) 
buffer_load_dwordx3 v[82:84], v14, s[40:43], 0 offen // (op 442) Load value
v_add_u32 v14, v14, v16 // (op 442) 
buffer_load_dwordx3 v[86:88], v14, s[40:43], 0 offen // (op 442) Load value
v_add_u32 v14, v14, v16 // (op 442) 
buffer_load_dwordx3 v[90:92], v14, s[40:43], 0 offen // (op 442) Load value




v_mov_b64 v[16:17], s[12:13] // (op 445) convert

s_lshl_b32 s0, s12, 3 // (op 445) 
v_add_u32 v12, v18, s0 // (op 445) 

buffer_load_ubyte v22, v12, s[48:51], 0 offen offset: 0 // (op 445) Load value
v_add_u32 v12, v12, v16 // (op 445) 
buffer_load_ubyte v23, v12, s[48:51], 0 offen offset: 0 // (op 445) Load value



v_lshlrev_b32 v12, 6, v1 // (op 890) 
v_mul_lo_u32 v14, v12, 6 // (op 890) 
v_lshrrev_b32 v21, 3, v14 // (op 890) 

v_lshrrev_b32 v14, 4, v12 // (op 890) 

v_lshlrev_b32 v12, 2, v14 // (op 890) 

v_add_u32 v14, v21, v12 // (op 890) 





s_waitcnt vmcnt(23)
ds_write_b96 v14, v[24:26] // (op 447) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v14, v[28:30] offset:16 // (op 447) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v14, v[32:34] offset:32 // (op 447) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v14, v[36:38] offset:48 // (op 447) Store local data 

v_and_b32 v12, -16, v1 // (op 996) 
v_sub_u32 v14, v1, v12 // (op 996) 

v_lshrrev_b32 v12, 4, v1 // (op 996) 
v_lshl_add_u32 v21, v12, 5, v14 // (op 996) 





s_waitcnt vmcnt(19)
ds_write_b8 v21, v40 offset:4096 // (op 448) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v21, v41 offset:4112 // (op 448) Store local data 

v_lshlrev_b32 v12, 6, v1 // (op 837) 
v_mul_lo_u32 v14, v12, 6 // (op 837) 
v_lshrrev_b32 v21, 3, v14 // (op 837) 

v_lshrrev_b32 v14, 4, v12 // (op 837) 

v_lshlrev_b32 v12, 2, v14 // (op 837) 

v_add_u32 v14, v21, v12 // (op 837) 





s_waitcnt vmcnt(17)
ds_write_b96 v14, v[42:44] offset:4224 // (op 449) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v14, v[46:48] offset:4240 // (op 449) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v14, v[50:52] offset:4256 // (op 449) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v14, v[54:56] offset:4272 // (op 449) Store local data 

v_and_b32 v12, -16, v1 // (op 943) 
v_sub_u32 v14, v1, v12 // (op 943) 

v_lshrrev_b32 v12, 4, v1 // (op 943) 
v_lshl_add_u32 v21, v12, 5, v14 // (op 943) 





s_waitcnt vmcnt(13)
ds_write_b8 v21, v58 offset:8320 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v21, v59 offset:8336 // (op 450) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1194) 



ds_read_b96_tr_b6 v[94:96], v5 // (op 467) Transpose load local data 
ds_read_b96_tr_b6 v[97:99], v5 offset:256 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1235) 


ds_read_u8 v12, v7 offset:4096 // (op 473) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1175) 



ds_read_b96_tr_b6 v[100:102], v0 offset:4224 // (op 464) Transpose load local data 
ds_read_b96_tr_b6 v[103:105], v0 offset:4480 // (op 464) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1215) 


ds_read_u8 v14, v4 offset:8320 // (op 470) Load local data 
v_lshlrev_b32 v21, 6, v1 // (op 808) 
v_mul_lo_u32 v27, v21, 6 // (op 808) 
v_lshrrev_b32 v31, 3, v27 // (op 808) 

v_lshrrev_b32 v27, 4, v21 // (op 808) 

v_lshlrev_b32 v21, 2, v27 // (op 808) 

v_add_u32 v27, v31, v21 // (op 808) 


v_lshlrev_b32 v21, 6, v1 // (op 861) 
v_mul_lo_u32 v31, v21, 6 // (op 861) 
v_lshrrev_b32 v35, 3, v31 // (op 861) 

v_lshrrev_b32 v31, 4, v21 // (op 861) 

v_lshlrev_b32 v21, 2, v31 // (op 861) 

v_add_u32 v31, v35, v21 // (op 861) 


v_and_b32 v21, -16, v1 // (op 914) 
v_sub_u32 v35, v1, v21 // (op 914) 

v_lshrrev_b32 v21, 4, v1 // (op 914) 
v_lshl_add_u32 v39, v21, 5, v35 // (op 914) 


v_and_b32 v21, -16, v1 // (op 967) 
v_sub_u32 v35, v1, v21 // (op 967) 

v_lshrrev_b32 v21, 4, v1 // (op 967) 
v_lshl_add_u32 v45, v21, 5, v35 // (op 967) 


v_lshlrev_b32 v21, 6, v1 // (op 1054) 
v_mul_lo_u32 v35, v21, 6 // (op 1054) 
v_lshrrev_b32 v49, 3, v35 // (op 1054) 

v_lshrrev_b32 v35, 4, v21 // (op 1054) 

v_lshlrev_b32 v21, 2, v35 // (op 1054) 

v_add_u32 v35, v49, v21 // (op 1054) 


v_lshlrev_b32 v21, 6, v1 // (op 1078) 
v_mul_lo_u32 v49, v21, 6 // (op 1078) 
v_lshrrev_b32 v53, 3, v49 // (op 1078) 

v_lshrrev_b32 v49, 4, v21 // (op 1078) 

v_lshlrev_b32 v21, 2, v49 // (op 1078) 

v_add_u32 v49, v53, v21 // (op 1078) 


v_and_b32 v21, -16, v1 // (op 1102) 
v_sub_u32 v53, v1, v21 // (op 1102) 

v_lshrrev_b32 v21, 4, v1 // (op 1102) 
v_lshl_add_u32 v57, v21, 5, v53 // (op 1102) 


v_and_b32 v21, -16, v1 // (op 1126) 
v_sub_u32 v53, v1, v21 // (op 1126) 

v_lshrrev_b32 v21, 4, v1 // (op 1126) 
v_lshl_add_u32 v63, v21, 5, v53 // (op 1126) 


s_mov_b32 s0, 0 // (op 59) call()
s_ashr_i64 s[14:15], s[6:7], 8 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58:

s_add_u32 s0, s0, 2 // (op 60) 
s_mul_i32 s15, s9, 6 // (op 10) most significant: low of high * low
s_mul_i32 s14, s8, 6 // (op 10) least significant
s_mul_hi_u32 s1, s8, 6 // (op 10) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 10) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[14:15] // (op 10) 


v_mov_b32 v18, v15 // (op 10) 
s_lshl_b64 s[14:15], s[8:9], 8 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s1, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s1, s14, 1 // (op 10) 

v_add_u32 v21, v18, s1 // (op 10) 

buffer_load_dwordx3 v[24:26], v21, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v21, v21, v16 // (op 10) 
buffer_load_dwordx3 v[28:30], v21, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v21, v21, v16 // (op 10) 
buffer_load_dwordx3 v[32:34], v21, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v21, v21, v16 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:105], v[94:99], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 88) 
buffer_load_dwordx3 v[36:38], v21, s[44:47], 0 offen // (op 10) Load value



s_barrier  // (op 1184) 
v_mov_b64 v[16:17], s[24:25] // (op 25) convert




ds_read_b96_tr_b6 v[100:102], v0 offset:6272 // (op 91) Transpose load local data 
ds_read_b96_tr_b6 v[103:105], v0 offset:6528 // (op 91) Transpose load local data 
v_mov_b32 v21, v20 // (op 25) 
s_barrier  // (op 1204) 
s_lshl_b32 s1, s24, 4 // (op 25) 



ds_read_b96_tr_b6 v[94:96], v5 offset:2048 // (op 95) Transpose load local data 
v_add_u32 v53, v21, s1 // (op 25) 

buffer_load_ubyte v40, v53, s[52:55], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v53, v53, v16 // (op 25) 
buffer_load_ubyte v41, v53, s[52:55], 0 offen offset: 0 // (op 25) Load value


ds_read_b96_tr_b6 v[97:99], v5 offset:2304 // (op 95) Transpose load local data 

s_barrier  // (op 1225) 
s_mul_i32 s15, s3, 6 // (op 4) most significant: low of high * low


s_mul_i32 s14, s2, 6 // (op 4) least significant
ds_read_u8 v14, v4 offset:8384 // (op 99) Load local data 
s_mul_hi_u32 s1, s2, 6 // (op 4) most significant: high of low * low
s_barrier  // (op 1245) 
s_add_u32 s15, s15, s1 // (op 4) most significant: sum



ds_read_u8 v12, v7 offset:4160 // (op 103) Load local data 
v_lshrrev_b64 v[16:17], 3, s[14:15] // (op 4) 


v_mov_b32 v53, v13 // (op 4) 
s_lshl_b64 s[14:15], s[2:3], 8 // (op 4) 
s_mul_i32 s19, s15, 6 // (op 4) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 4) least significant
s_mul_hi_u32 s1, s14, 6 // (op 4) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 4) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s1, s14, 1 // (op 4) 

v_add_u32 v67, v53, s1 // (op 4) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:105], v[94:99], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 105) 
buffer_load_dwordx3 v[42:44], v67, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v67, v67, v16 // (op 4) 
buffer_load_dwordx3 v[46:48], v67, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v67, v67, v16 // (op 4) 
buffer_load_dwordx3 v[50:52], v67, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v67, v67, v16 // (op 4) 
buffer_load_dwordx3 v[54:56], v67, s[40:43], 0 offen // (op 4) Load value



v_mov_b64 v[16:17], s[12:13] // (op 17) convert

v_mov_b32 v67, v19 // (op 17) 
s_lshl_b32 s1, s12, 4 // (op 17) 
v_add_u32 v71, v67, s1 // (op 17) 

buffer_load_ubyte v58, v71, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v71, v71, v16 // (op 17) 
buffer_load_ubyte v59, v71, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1265) 



s_waitcnt vmcnt(23)
ds_write_b96 v49, v[60:62] offset:8448 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v49, v[64:66] offset:8464 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v49, v[68:70] offset:8480 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v49, v[72:74] offset:8496 // (op 302) Store local data 
s_barrier  // (op 1277) 



s_waitcnt vmcnt(19)
ds_write_b8 v63, v76 offset:12544 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v63, v77 offset:12560 // (op 304) Store local data 
s_barrier  // (op 1258) 



s_waitcnt vmcnt(17)
ds_write_b96 v35, v[78:80] offset:12672 // (op 283) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v35, v[82:84] offset:12688 // (op 283) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v35, v[86:88] offset:12704 // (op 283) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v35, v[90:92] offset:12720 // (op 283) Store local data 
s_barrier  // (op 1271) 



s_waitcnt vmcnt(13)
ds_write_b8 v57, v22 offset:16768 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v57, v23 offset:16784 // (op 303) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1261) 



ds_read_b96_tr_b6 v[106:108], v9 offset:8448 // (op 287) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v9 offset:8704 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1274) 


ds_read_u8 v71, v11 offset:12544 // (op 291) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1255) 



ds_read_b96_tr_b6 v[112:114], v3 offset:12672 // (op 285) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v3 offset:12928 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1268) 


ds_read_u8 v75, v8 offset:16768 // (op 289) Load local data 

s_mul_i32 s15, s9, 6 // (op 305) most significant: low of high * low
s_mul_i32 s14, s8, 6 // (op 305) least significant
s_mul_hi_u32 s1, s8, 6 // (op 305) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 305) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[14:15] // (op 305) 


s_lshl_b64 s[14:15], s[8:9], 8 // (op 305) 
s_mul_i32 s19, s15, 6 // (op 305) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 305) least significant
s_mul_hi_u32 s1, s14, 6 // (op 305) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 305) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 305) 

s_mul_i32 s1, 3, s14 // (op 305) 

v_add_u32 v81, v18, s1 // (op 305) 

buffer_load_dwordx3 v[60:62], v81, s[44:47], 0 offen // (op 305) Load value
v_add_u32 v81, v81, v16 // (op 305) 
buffer_load_dwordx3 v[64:66], v81, s[44:47], 0 offen // (op 305) Load value
v_add_u32 v81, v81, v16 // (op 305) 
buffer_load_dwordx3 v[68:70], v81, s[44:47], 0 offen // (op 305) Load value
v_add_u32 v81, v81, v16 // (op 305) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:117], v[106:111], a[0:3], v75, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 292) 
buffer_load_dwordx3 v[72:74], v81, s[44:47], 0 offen // (op 305) Load value











v_mov_b64 v[16:17], s[24:25] // (op 307) convert

s_lshl_b32 s1, s24, 3 // (op 307) 
s_mul_i32 s14, 3, s1 // (op 307) 

v_add_u32 v18, v21, s14 // (op 307) 

buffer_load_ubyte v76, v18, s[52:55], 0 offen offset: 0 // (op 307) Load value
v_add_u32 v18, v18, v16 // (op 307) 
buffer_load_ubyte v77, v18, s[52:55], 0 offen offset: 0 // (op 307) Load value


s_lshl_b64 s[14:15], s[8:9], 8 // (op 634) 

s_mul_i32 s19, s15, 6 // (op 634) most significant: low of high * low

s_mul_i32 s18, s14, 6 // (op 634) least significant
s_mul_hi_u32 s1, s14, 6 // (op 634) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 634) most significant: sum


s_lshl_b32 s1, s24, 4 // (op 762) 
v_add_u32 v20, v20, s1 // (op 762) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 634) 

s_lshl_b32 s1, s14, 1 // (op 634) 

v_add_u32 v15, v15, s1 // (op 634) 

s_mul_i32 s15, s3, 6 // (op 282) most significant: low of high * low
s_mul_i32 s14, s2, 6 // (op 282) least significant
s_mul_hi_u32 s1, s2, 6 // (op 282) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 282) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[14:15] // (op 282) 


s_lshl_b64 s[14:15], s[2:3], 8 // (op 282) 
s_mul_i32 s19, s15, 6 // (op 282) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 282) least significant
s_mul_hi_u32 s1, s14, 6 // (op 282) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 282) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 282) 

s_mul_i32 s1, 3, s14 // (op 282) 

v_add_u32 v18, v53, s1 // (op 282) 

ds_read_b96_tr_b6 v[106:108], v3 offset:14720 // (op 294) Transpose load local data 
ds_read_b96_tr_b6 v[109:111], v3 offset:14976 // (op 294) Transpose load local data 
buffer_load_dwordx3 v[78:80], v18, s[40:43], 0 offen // (op 282) Load value
v_add_u32 v18, v18, v16 // (op 282) 



s_barrier  // (op 1336) 
ds_read_b96_tr_b6 v[112:114], v9 offset:10496 // (op 296) Transpose load local data 
ds_read_b96_tr_b6 v[115:117], v9 offset:10752 // (op 296) Transpose load local data 
buffer_load_dwordx3 v[82:84], v18, s[40:43], 0 offen // (op 282) Load value
v_add_u32 v18, v18, v16 // (op 282) 


s_barrier  // (op 1340) 
buffer_load_dwordx3 v[86:88], v18, s[40:43], 0 offen // (op 282) Load value
v_add_u32 v18, v18, v16 // (op 282) 
ds_read_u8 v21, v8 offset:16832 // (op 298) Load local data 
buffer_load_dwordx3 v[90:92], v18, s[40:43], 0 offen // (op 282) Load value






s_barrier  // (op 1344) 
ds_read_u8 v53, v11 offset:12608 // (op 300) Load local data 
v_mov_b64 v[16:17], s[12:13] // (op 306) convert

s_lshl_b32 s1, s12, 3 // (op 306) 
s_barrier  // (op 1348) 
s_mul_i32 s14, 3, s1 // (op 306) 

v_add_u32 v18, v67, s14 // (op 306) 

buffer_load_ubyte v22, v18, s[48:51], 0 offen offset: 0 // (op 306) Load value
v_add_u32 v18, v18, v16 // (op 306) 
buffer_load_ubyte v23, v18, s[48:51], 0 offen offset: 0 // (op 306) Load value


s_lshl_b64 s[14:15], s[2:3], 8 // (op 570) 

s_mul_i32 s19, s15, 6 // (op 570) most significant: low of high * low
s_barrier  // (op 1197) 
s_mul_i32 s18, s14, 6 // (op 570) least significant



s_mul_hi_u32 s1, s14, 6 // (op 570) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 570) most significant: sum


s_lshl_b32 s1, s12, 4 // (op 698) 
v_add_u32 v19, v19, s1 // (op 698) 

s_lshr_b64 s[14:15], s[18:19], 3 // (op 570) 

s_lshl_b32 s1, s14, 1 // (op 570) 

v_add_u32 v13, v13, s1 // (op 570) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[106:111], v[112:117], a[0:3], v21, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 301) 
s_waitcnt vmcnt(23)
ds_write_b96 v31, v[24:26] // (op 13) Store local data 




s_waitcnt vmcnt(22)
ds_write_b96 v31, v[28:30] offset:16 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v31, v[32:34] offset:32 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v31, v[36:38] offset:48 // (op 13) Store local data 
s_barrier  // (op 1238) 



s_waitcnt vmcnt(19)
ds_write_b8 v45, v40 offset:4096 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v45, v41 offset:4112 // (op 30) Store local data 
s_barrier  // (op 1178) 



s_waitcnt vmcnt(17)
ds_write_b96 v27, v[42:44] offset:4224 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b96 v27, v[46:48] offset:4240 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b96 v27, v[50:52] offset:4256 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b96 v27, v[54:56] offset:4272 // (op 6) Store local data 
s_barrier  // (op 1218) 



s_waitcnt vmcnt(13)
ds_write_b8 v39, v58 offset:8320 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v39, v59 offset:8336 // (op 21) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1207) 



ds_read_b96_tr_b6 v[94:96], v5 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[97:99], v5 offset:256 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1248) 


ds_read_u8 v12, v7 offset:4096 // (op 85) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1187) 



ds_read_b96_tr_b6 v[100:102], v0 offset:4224 // (op 70) Transpose load local data 
ds_read_b96_tr_b6 v[103:105], v0 offset:4480 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1228) 


ds_read_u8 v14, v4 offset:8320 // (op 80) Load local data 
s_ashr_i64 s[14:15], s[6:7], 8 // (op 58) 
s_mov_b32 s1, s14 // (op 58) convert

s_and_b32 s14, s1, 4294967294 // (op 58) 

s_cmp_lt_u32 s0, s14 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58:

















v_lshlrev_b32 v13, 2, v1 // (op 533) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 533) 
s_lshr_b32 s14, s0, 1 // (op 533) 

v_lshl_add_u32 v63, s14, 9, v13 // (op 533) 


s_lshl_b32 s0, s16, 4 // (op 533) 
v_mul_lo_u32 v21, v63, s3 // (op 533) most significant: low of low * high
v_mul_hi_u32 v13, v63, s2 // (op 533) most significant: high of low * low
v_mul_lo_u32 v20, v63, s2 // (op 533) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 533) most significant: sum


s_mov_b32 s15, 0 // (op 533) convert
s_mov_b32 s14, s0 // (op 533) convert

v_mov_b32 v13, s14 // (op 533) 
v_mov_b32 v63, s15 // (op 533) 
v_add_co_u32 v18, s[0:1], v13, v20 // (op 533) least significant half
v_addc_co_u32 v19, s[0:1], v63, v21, s[0:1] // (op 533) most significant half





v_mul_lo_u32 v17, v19, 6 // (op 533) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 533) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 533) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 533) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 533) 

v_mov_b32 v13, v18 // (op 533) convert

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

v_lshlrev_b32 v63, 2, v1 // (op 600) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 600) 
s_lshr_b32 s14, s0, 1 // (op 600) 

v_lshl_add_u32 v67, s14, 9, v63 // (op 600) 


s_lshl_b32 s0, s17, 4 // (op 600) 
s_mov_b32 s15, 0 // (op 600) convert
s_mov_b32 s14, s0 // (op 600) convert

v_mul_lo_u32 v21, v67, s9 // (op 600) most significant: low of low * high
v_mul_hi_u32 v63, v67, s8 // (op 600) most significant: high of low * low
v_mul_lo_u32 v20, v67, s8 // (op 600) least significant: low of low * low
v_add_u32 v21, v21, v63 // (op 600) most significant: sum

v_mov_b32 v63, s14 // (op 600) 
v_mov_b32 v71, s15 // (op 600) 
v_add_co_u32 v20, s[0:1], v20, v63 // (op 600) least significant half
v_addc_co_u32 v21, s[0:1], v21, v71, s[0:1] // (op 600) most significant half





v_mul_lo_u32 v19, v21, 6 // (op 600) most significant: low of high * low
v_mul_hi_u32 v63, v20, 6 // (op 600) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 600) least significant: low of low * low
v_add_u32 v19, v19, v63 // (op 600) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 600) 

v_mov_b32 v15, v16 // (op 600) convert

s_mul_i32 s1, s7, 6 // (op 602) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 602) least significant
s_mul_hi_u32 s14, s6, 6 // (op 602) most significant: high of low * low
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

v_and_b32 v63, -16, v1 // (op 664) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 664) 
v_sub_u32 v67, v1, v63 // (op 664) 

s_lshr_b32 s14, s0, 1 // (op 664) 

v_lshl_add_u32 v63, s16, 4, v67 // (op 664) 

v_lshrrev_b32 v67, 4, v1 // (op 664) 
v_lshlrev_b32 v71, 1, v67 // (op 664) 

v_lshl_add_u32 v67, s14, 4, v71 // (op 664) 


v_mul_lo_u32 v71, v67, s12 // (op 664) 

v_add_u32 v67, v63, v71 // (op 664) 


s_mul_i32 s1, s12, s23 // (op 666) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 666) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 666) least significant
s_mul_hi_u32 s15, s12, s22 // (op 666) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 666) most significant: sum
s_add_u32 s1, s1, s15 // (op 666) most significant: sum


s_mov_b32 s48, s20 // (op 666) reg expression
s_mov_b32 s49, s21 // (op 666) reg expression
s_mov_b32 s50, s0 // (op 666) reg expression
s_mov_b32 s51, 131072 // (op 666) call()

v_lshrrev_b32 v63, 4, v1 // (op 728) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 728) 
v_lshlrev_b32 v71, 1, v63 // (op 728) 

s_lshr_b32 s14, s0, 1 // (op 728) 

v_and_b32 v63, -16, v1 // (op 728) 
v_lshl_add_u32 v75, s14, 4, v71 // (op 728) 


v_sub_u32 v71, v1, v63 // (op 728) 

v_lshl_add_u32 v63, s17, 4, v71 // (op 728) 

v_mul_lo_u32 v71, v75, s24 // (op 728) 
v_add_u32 v71, v71, v63 // (op 728) 


s_mul_i32 s1, s24, s23 // (op 730) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 730) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 730) least significant
s_mul_hi_u32 s15, s24, s22 // (op 730) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 730) most significant: sum
s_add_u32 s1, s1, s15 // (op 730) most significant: sum


s_mov_b32 s52, s26 // (op 730) reg expression
s_mov_b32 s53, s27 // (op 730) reg expression
s_mov_b32 s54, s0 // (op 730) reg expression
s_mov_b32 s55, 131072 // (op 730) call()

v_lshlrev_b32 v63, 6, v1 // (op 821) 
v_mul_lo_u32 v75, v63, 6 // (op 821) 
v_lshrrev_b32 v81, 3, v75 // (op 821) 

v_lshrrev_b32 v75, 4, v63 // (op 821) 

v_lshlrev_b32 v63, 2, v75 // (op 821) 

v_add_u32 v75, v81, v63 // (op 821) 


v_lshlrev_b32 v63, 6, v1 // (op 874) 
v_mul_lo_u32 v81, v63, 6 // (op 874) 
v_lshrrev_b32 v85, 3, v81 // (op 874) 

v_lshrrev_b32 v81, 4, v63 // (op 874) 

v_lshlrev_b32 v63, 2, v81 // (op 874) 

v_add_u32 v81, v85, v63 // (op 874) 


v_and_b32 v63, -16, v1 // (op 927) 
v_sub_u32 v85, v1, v63 // (op 927) 

v_lshrrev_b32 v63, 4, v1 // (op 927) 
v_lshl_add_u32 v89, v63, 5, v85 // (op 927) 


v_and_b32 v63, -16, v1 // (op 980) 
v_sub_u32 v85, v1, v63 // (op 980) 

v_lshrrev_b32 v63, 4, v1 // (op 980) 
v_lshl_add_u32 v93, v63, 5, v85 // (op 980) 


s_ashr_i64 s[0:1], s[6:7], 8 // (op 194) 
s_and_b32 s14, s0, 4294967294 // (op 194) 

s_ashr_i64 s[0:1], s[6:7], 8 // (op 193) 
s_mov_b32 s15, s0 // (op 193) convert

s_cmp_lt_u32 s14, s15 // (op 193) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193:

s_add_u32 s14, s14, 1 // (op 195) 
s_mul_i32 s1, s3, 6 // (op 198) most significant: low of high * low
s_mul_i32 s0, s2, 6 // (op 198) least significant
s_mul_hi_u32 s15, s2, 6 // (op 198) most significant: high of low * low
s_add_u32 s1, s1, s15 // (op 198) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 198) 


v_mov_b32 v12, v13 // (op 198) 
v_mov_b32 v14, v12 // (op 198) reg expression
buffer_load_dwordx3 v[18:20], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v16 // (op 198) 
buffer_load_dwordx3 v[22:24], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v16 // (op 198) 
buffer_load_dwordx3 v[26:28], v14, s[40:43], 0 offen // (op 198) Load value
v_add_u32 v14, v14, v16 // (op 198) 
buffer_load_dwordx3 v[30:32], v14, s[40:43], 0 offen // (op 198) Load value



s_lshl_b64 s[0:1], s[2:3], 8 // (op 523) 
s_mul_i32 s19, s1, 6 // (op 523) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 523) least significant
s_mul_hi_u32 s15, s0, 6 // (op 523) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 523) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 523) 

v_add_u32 v13, v13, s0 // (op 523) 


s_barrier  // (op 1181) 
s_mul_i32 s1, s9, 6 // (op 221) most significant: low of high * low



s_mul_i32 s0, s8, 6 // (op 221) least significant
s_mul_hi_u32 s15, s8, 6 // (op 221) most significant: high of low * low
s_add_u32 s1, s1, s15 // (op 221) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 221) 


v_mov_b32 v12, v15 // (op 221) 
v_mov_b32 v14, v12 // (op 221) reg expression
buffer_load_dwordx3 v[34:36], v14, s[44:47], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v16 // (op 221) 
buffer_load_dwordx3 v[38:40], v14, s[44:47], 0 offen // (op 221) Load value
v_add_u32 v14, v14, v16 // (op 221) 
s_waitcnt vmcnt(5)
ds_write_b96 v75, v[18:20] offset:4224 // (op 199) Store local data 
s_waitcnt vmcnt(4)
ds_write_b96 v75, v[22:24] offset:4240 // (op 199) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v75, v[26:28] offset:4256 // (op 199) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v75, v[30:32] offset:4272 // (op 199) Store local data 
buffer_load_dwordx3 v[42:44], v14, s[44:47], 0 offen // (op 221) Load value

v_add_u32 v14, v14, v16 // (op 221) 
buffer_load_dwordx3 v[46:48], v14, s[44:47], 0 offen // (op 221) Load value



s_lshl_b64 s[0:1], s[8:9], 8 // (op 592) 
s_mul_i32 s19, s1, 6 // (op 592) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 592) least significant
s_mul_hi_u32 s15, s0, 6 // (op 592) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 592) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 592) 

v_add_u32 v15, v15, s0 // (op 592) 


s_barrier  // (op 1200) 
v_mov_b64 v[16:17], s[12:13] // (op 222) convert




v_mov_b32 v12, v67 // (op 222) 
v_mov_b32 v14, v12 // (op 222) reg expression
buffer_load_ubyte v18, v14, s[48:51], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v14, v14, v16 // (op 222) 
s_waitcnt vmcnt(4)
ds_write_b96 v81, v[34:36] // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v81, v[38:40] offset:16 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v81, v[42:44] offset:32 // (op 218) Store local data 
buffer_load_ubyte v19, v14, s[48:51], 0 offen offset: 0 // (op 222) Load value



s_lshl_b32 s15, s12, 3 // (op 656) 
v_add_u32 v67, v67, s15 // (op 656) 

s_waitcnt vmcnt(2)
ds_write_b96 v81, v[46:48] offset:48 // (op 218) Store local data 


v_mov_b64 v[16:17], s[24:25] // (op 223) convert

s_barrier  // (op 1221) 
v_mov_b32 v12, v71 // (op 223) 



v_mov_b32 v14, v12 // (op 223) reg expression
buffer_load_ubyte v20, v14, s[52:55], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v14, v14, v16 // (op 223) 
buffer_load_ubyte v21, v14, s[52:55], 0 offen offset: 0 // (op 223) Load value



s_lshl_b32 s15, s24, 3 // (op 720) 
v_add_u32 v71, v71, s15 // (op 720) 

s_waitcnt vmcnt(3)
ds_write_b8 v89, v18 offset:8320 // (op 219) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1190) 
s_waitcnt vmcnt(2)
ds_write_b8 v89, v19 offset:8336 // (op 219) Store local data 




ds_read_b96_tr_b6 v[22:24], v0 offset:4224 // (op 201) Transpose load local data 
s_barrier  // (op 1241) 
ds_read_b96_tr_b6 v[25:27], v0 offset:4480 // (op 201) Transpose load local data 



s_waitcnt lgkmcnt(4)
s_barrier  // (op 1211) 
s_waitcnt vmcnt(1)
ds_write_b8 v93, v20 offset:4096 // (op 220) Store local data 



ds_read_b96_tr_b6 v[28:30], v5 // (op 203) Transpose load local data 
ds_read_b96_tr_b6 v[31:33], v5 offset:256 // (op 203) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v93, v21 offset:4112 // (op 220) Store local data 

s_waitcnt lgkmcnt(6)
s_barrier  // (op 1231) 


ds_read_u8 v12, v4 offset:8320 // (op 205) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1251) 


ds_read_u8 v14, v7 offset:4096 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:27], v[28:33], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 208) 







ds_read_b96_tr_b6 v[16:18], v0 offset:6272 // (op 210) Transpose load local data 
ds_read_b96_tr_b6 v[19:21], v0 offset:6528 // (op 210) Transpose load local data 



ds_read_b96_tr_b6 v[22:24], v5 offset:2048 // (op 212) Transpose load local data 
ds_read_b96_tr_b6 v[25:27], v5 offset:2304 // (op 212) Transpose load local data 


ds_read_u8 v12, v4 offset:8384 // (op 214) Load local data 


ds_read_u8 v14, v7 offset:4160 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[22:27], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 217) 




s_ashr_i64 s[0:1], s[6:7], 8 // (op 193) 
s_mov_b32 s15, s0 // (op 193) convert

s_cmp_lt_u32 s14, s15 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 if true)
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1328) 








v_cmp_eq_f32 vcc, 0.00000, v6 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v10, v12 // (op 50) 
v_mul_f32 v17, v10, v13 // (op 50) 
v_mul_f32 v18, v10, v14 // (op 50) 
v_mul_f32 v19, v10, v15 // (op 50) 

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


v_lshl_add_u32 v3, s17, 4, v5 // (op 1022) 

v_mul_lo_u32 v4, v3, s28 // (op 1022) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 1022) 


s_mul_i32 s1, s28, s33 // (op 1024) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1024) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1024) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1024) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1024) most significant: sum
s_add_u32 s1, s1, s3 // (op 1024) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 1024) 

s_mov_b32 s4, s30 // (op 1024) reg expression
s_mov_b32 s5, s31 // (op 1024) reg expression
s_mov_b32 s6, s2 // (op 1024) reg expression
s_mov_b32 s7, 131072 // (op 1024) call()




buffer_store_dwordx4 v[16:19], v3, s[4:7], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155:
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

v_mul_lo_u32 v4, v3, s34 // (op 791) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 791) 


s_mul_i32 s1, s34, s33 // (op 793) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 793) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 793) least significant
s_mul_hi_u32 s3, s34, s32 // (op 793) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 793) most significant: sum
s_add_u32 s1, s1, s3 // (op 793) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 793) 

s_mov_b32 s8, s36 // (op 793) reg expression
s_mov_b32 s9, s37 // (op 793) reg expression
s_mov_b32 s10, s2 // (op 793) reg expression
s_mov_b32 s11, 131072 // (op 793) call()





buffer_load_dwordx4 v[12:15], v3, s[8:11], 0 offen // (op 167) Load value



v_accvgpr_read v16, a0 // (op 164) 
v_accvgpr_read v17, a1 // (op 164) 
v_accvgpr_read v18, a2 // (op 164) 
v_accvgpr_read v19, a3 // (op 164) 
v_mul_f32 v20, v10, v16 // (op 164) 
v_mul_f32 v21, v10, v17 // (op 164) 
v_mul_f32 v22, v10, v18 // (op 164) 
v_mul_f32 v23, v10, v19 // (op 164) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v6, v12 // (op 168) 
v_mul_f32 v17, v6, v13 // (op 168) 
v_mul_f32 v18, v6, v14 // (op 168) 
v_mul_f32 v19, v6, v15 // (op 168) 

v_add_f32 v12, v16, v20 // (op 165) 
v_add_f32 v13, v17, v21 // (op 165) 
v_add_f32 v14, v18, v22 // (op 165) 
v_add_f32 v15, v19, v23 // (op 165) 


v_lshrrev_b32 v0, 6, v1 // (op 1039) 
v_and_b32 v11, 63, v1 // (op 1039) 
v_lshrrev_b32 v3, 4, v11 // (op 1039) 
v_lshlrev_b32 v4, 2, v3 // (op 1039) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1039) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 1039) 

v_lshrrev_b32 v3, 2, v11 // (op 1039) 

v_and_b32 v11, 3, v3 // (op 1039) 

v_and_b32 v3, 3, v1 // (op 1039) 
v_lshl_add_u32 v4, v11, 2, v3 // (op 1039) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 1039) 

v_mul_lo_u32 v11, v3, s28 // (op 1039) 

v_add_lshl_u32 v3, v0, v11, 2 // (op 1039) 


s_mul_i32 s1, s28, s33 // (op 1041) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1041) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1041) most significant: sum
s_add_u32 s1, s1, s3 // (op 1041) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 1041) 

s_mov_b32 s4, s30 // (op 1041) reg expression
s_mov_b32 s5, s31 // (op 1041) reg expression
s_mov_b32 s6, s2 // (op 1041) reg expression
s_mov_b32 s7, 131072 // (op 1041) call()




buffer_store_dwordx4 v[12:15], v3, s[4:7], 0 offen // (op 166) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155:



s_endpgm  // End of RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 124
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
  .amdhsa_accum_offset 120
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: B6
    DestDataType: s
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
  - .name: RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 118
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

