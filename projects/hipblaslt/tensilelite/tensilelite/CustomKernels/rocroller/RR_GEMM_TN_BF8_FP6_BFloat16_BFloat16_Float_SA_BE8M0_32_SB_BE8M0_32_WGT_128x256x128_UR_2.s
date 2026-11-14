// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2,@function
RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 13761) 
v_and_b32 v3, 15, v1 // (op 13761) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 13761) 


v_and_b32 v0, 48, v1 // (op 13761) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 13761) 


v_lshrrev_b32 v0, 6, v1 // (op 13819) 
v_and_b32 v4, 1, v0 // (op 13819) 

v_and_b32 v0, 15, v1 // (op 13819) 
v_lshl_add_u32 v5, v4, 7, v0 // (op 13819) 


v_and_b32 v0, 63, v1 // (op 13819) 
v_lshrrev_b32 v4, 4, v0 // (op 13819) 

v_lshlrev_b32 v0, 5, v4 // (op 13819) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 13819) 


v_mul_lo_u32 v0, v4, 6 // (op 13819) 

v_lshrrev_b32 v4, 3, v0 // (op 13819) 

v_lshrrev_b32 v0, 7, v1 // (op 13886) 
v_and_b32 v5, 15, v1 // (op 13886) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 13886) 


v_and_b32 v0, 48, v1 // (op 13886) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 13886) 


v_lshrrev_b32 v0, 6, v1 // (op 13915) 
v_and_b32 v6, 1, v0 // (op 13915) 

v_and_b32 v0, 15, v1 // (op 13915) 
v_lshl_add_u32 v7, v6, 7, v0 // (op 13915) 


v_and_b32 v0, 63, v1 // (op 13915) 
v_lshrrev_b32 v6, 4, v0 // (op 13915) 

v_lshlrev_b32 v0, 5, v6 // (op 13915) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 13915) 


v_mul_lo_u32 v0, v6, 6 // (op 13915) 

v_lshrrev_b32 v6, 3, v0 // (op 13915) 

v_lshrrev_b32 v0, 7, v1 // (op 13700) 
v_and_b32 v7, 63, v1 // (op 13700) 
v_lshrrev_b32 v8, 4, v7 // (op 13700) 
v_lshlrev_b32 v9, 2, v8 // (op 13700) 

v_lshl_add_u32 v8, v0, 6, v9 // (op 13700) 


v_lshl_add_u32 v0, s16, 7, v8 // (op 13700) 

v_lshrrev_b32 v8, 2, v7 // (op 13700) 

v_and_b32 v7, 3, v8 // (op 13700) 

v_and_b32 v8, 3, v1 // (op 13700) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 13700) 


v_lshrrev_b32 v7, 6, v1 // (op 13700) 
v_and_b32 v8, 1, v7 // (op 13700) 

v_lshl_add_u32 v7, v8, 7, v9 // (op 13700) 


v_lshl_add_u32 v8, s17, 8, v7 // (op 13700) 

v_mul_lo_u32 v7, v8, s2 // (op 13700) 

v_add_lshl_u32 v8, v0, v7, 1 // (op 13700) 


v_lshrrev_b32 v0, 7, v1 // (op 13840) 
v_and_b32 v7, 63, v1 // (op 13840) 
v_lshrrev_b32 v9, 4, v7 // (op 13840) 
s_mul_i32 s1, s2, s9 // (op 13702) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 13702) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 13702) least significant
s_mul_hi_u32 s19, s2, s8 // (op 13702) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13702) most significant: sum
s_add_u32 s1, s1, s19 // (op 13702) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 13702) 

v_lshlrev_b32 v10, 2, v9 // (op 13840) 

v_lshl_add_u32 v9, v0, 6, v10 // (op 13840) 


v_lshl_add_u32 v0, s16, 7, v9 // (op 13840) 

v_lshrrev_b32 v9, 2, v7 // (op 13840) 

v_and_b32 v7, 3, v9 // (op 13840) 

v_and_b32 v9, 3, v1 // (op 13840) 
v_lshl_add_u32 v10, v7, 2, v9 // (op 13840) 


v_lshrrev_b32 v7, 6, v1 // (op 13840) 
v_and_b32 v9, 1, v7 // (op 13840) 

v_lshl_add_u32 v7, v9, 7, v10 // (op 13840) 


v_lshl_add_u32 v9, s17, 8, v7 // (op 13840) 

v_mul_lo_u32 v7, v9, s4 // (op 13840) 

v_add_lshl_u32 v9, v0, v7, 1 // (op 13840) 


s_mov_b32 s40, s6 // (op 13702) reg expression
s_mov_b32 s41, s7 // (op 13702) reg expression
s_mov_b32 s42, s18 // (op 13702) reg expression
s_mov_b32 s43, 131072 // (op 13702) call()

s_mul_i32 s1, s4, s9 // (op 13842) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 13842) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 13842) least significant
s_mul_hi_u32 s7, s4, s8 // (op 13842) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13842) most significant: sum
s_add_u32 s1, s1, s7 // (op 13842) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 13842) 

s_mov_b32 s44, s10 // (op 13842) reg expression
s_mov_b32 s45, s11 // (op 13842) reg expression
s_mov_b32 s46, s6 // (op 13842) reg expression
s_mov_b32 s47, 131072 // (op 13842) call()


v_mov_b32 v0, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_accvgpr_write a4, 0.00000 // (op 281) call()
v_accvgpr_write a5, 0.00000 // (op 281) call()
v_accvgpr_write a6, 0.00000 // (op 281) call()
v_accvgpr_write a7, 0.00000 // (op 281) call()
v_accvgpr_write a8, 0.00000 // (op 417) call()
v_accvgpr_write a9, 0.00000 // (op 417) call()
v_accvgpr_write a10, 0.00000 // (op 417) call()
v_accvgpr_write a11, 0.00000 // (op 417) call()
v_accvgpr_write a12, 0.00000 // (op 553) call()
v_accvgpr_write a13, 0.00000 // (op 553) call()
v_accvgpr_write a14, 0.00000 // (op 553) call()
v_accvgpr_write a15, 0.00000 // (op 553) call()
v_accvgpr_write a16, 0.00000 // (op 689) call()
v_accvgpr_write a17, 0.00000 // (op 689) call()
v_accvgpr_write a18, 0.00000 // (op 689) call()
v_accvgpr_write a19, 0.00000 // (op 689) call()
v_accvgpr_write a20, 0.00000 // (op 825) call()
v_accvgpr_write a21, 0.00000 // (op 825) call()
v_accvgpr_write a22, 0.00000 // (op 825) call()
v_accvgpr_write a23, 0.00000 // (op 825) call()
v_accvgpr_write a24, 0.00000 // (op 961) call()
v_accvgpr_write a25, 0.00000 // (op 961) call()
v_accvgpr_write a26, 0.00000 // (op 961) call()
v_accvgpr_write a27, 0.00000 // (op 961) call()
v_accvgpr_write a28, 0.00000 // (op 1097) call()
v_accvgpr_write a29, 0.00000 // (op 1097) call()
v_accvgpr_write a30, 0.00000 // (op 1097) call()
v_accvgpr_write a31, 0.00000 // (op 1097) call()
v_accvgpr_write a32, 0.00000 // (op 1409) call()
v_accvgpr_write a33, 0.00000 // (op 1409) call()
v_accvgpr_write a34, 0.00000 // (op 1409) call()
v_accvgpr_write a35, 0.00000 // (op 1409) call()
v_accvgpr_write a36, 0.00000 // (op 1930) call()
v_accvgpr_write a37, 0.00000 // (op 1930) call()
v_accvgpr_write a38, 0.00000 // (op 1930) call()
v_accvgpr_write a39, 0.00000 // (op 1930) call()
v_accvgpr_write a40, 0.00000 // (op 1931) call()
v_accvgpr_write a41, 0.00000 // (op 1931) call()
v_accvgpr_write a42, 0.00000 // (op 1931) call()
v_accvgpr_write a43, 0.00000 // (op 1931) call()
v_accvgpr_write a44, 0.00000 // (op 1932) call()
v_accvgpr_write a45, 0.00000 // (op 1932) call()
v_accvgpr_write a46, 0.00000 // (op 1932) call()
v_accvgpr_write a47, 0.00000 // (op 1932) call()
v_accvgpr_write a48, 0.00000 // (op 1933) call()
v_accvgpr_write a49, 0.00000 // (op 1933) call()
v_accvgpr_write a50, 0.00000 // (op 1933) call()
v_accvgpr_write a51, 0.00000 // (op 1933) call()
v_accvgpr_write a52, 0.00000 // (op 1934) call()
v_accvgpr_write a53, 0.00000 // (op 1934) call()
v_accvgpr_write a54, 0.00000 // (op 1934) call()
v_accvgpr_write a55, 0.00000 // (op 1934) call()
v_accvgpr_write a56, 0.00000 // (op 1935) call()
v_accvgpr_write a57, 0.00000 // (op 1935) call()
v_accvgpr_write a58, 0.00000 // (op 1935) call()
v_accvgpr_write a59, 0.00000 // (op 1935) call()
v_accvgpr_write a60, 0.00000 // (op 1936) call()
v_accvgpr_write a61, 0.00000 // (op 1936) call()
v_accvgpr_write a62, 0.00000 // (op 1936) call()
v_accvgpr_write a63, 0.00000 // (op 1936) call()
v_accvgpr_write a64, 0.00000 // (op 2627) call()
v_accvgpr_write a65, 0.00000 // (op 2627) call()
v_accvgpr_write a66, 0.00000 // (op 2627) call()
v_accvgpr_write a67, 0.00000 // (op 2627) call()
v_accvgpr_write a68, 0.00000 // (op 3148) call()
v_accvgpr_write a69, 0.00000 // (op 3148) call()
v_accvgpr_write a70, 0.00000 // (op 3148) call()
v_accvgpr_write a71, 0.00000 // (op 3148) call()
v_accvgpr_write a72, 0.00000 // (op 3149) call()
v_accvgpr_write a73, 0.00000 // (op 3149) call()
v_accvgpr_write a74, 0.00000 // (op 3149) call()
v_accvgpr_write a75, 0.00000 // (op 3149) call()
v_accvgpr_write a76, 0.00000 // (op 3150) call()
v_accvgpr_write a77, 0.00000 // (op 3150) call()
v_accvgpr_write a78, 0.00000 // (op 3150) call()
v_accvgpr_write a79, 0.00000 // (op 3150) call()
v_accvgpr_write a80, 0.00000 // (op 3151) call()
v_accvgpr_write a81, 0.00000 // (op 3151) call()
v_accvgpr_write a82, 0.00000 // (op 3151) call()
v_accvgpr_write a83, 0.00000 // (op 3151) call()
v_accvgpr_write a84, 0.00000 // (op 3152) call()
v_accvgpr_write a85, 0.00000 // (op 3152) call()
v_accvgpr_write a86, 0.00000 // (op 3152) call()
v_accvgpr_write a87, 0.00000 // (op 3152) call()
v_accvgpr_write a88, 0.00000 // (op 3153) call()
v_accvgpr_write a89, 0.00000 // (op 3153) call()
v_accvgpr_write a90, 0.00000 // (op 3153) call()
v_accvgpr_write a91, 0.00000 // (op 3153) call()
v_accvgpr_write a92, 0.00000 // (op 3154) call()
v_accvgpr_write a93, 0.00000 // (op 3154) call()
v_accvgpr_write a94, 0.00000 // (op 3154) call()
v_accvgpr_write a95, 0.00000 // (op 3154) call()
v_accvgpr_write a96, 0.00000 // (op 3845) call()
v_accvgpr_write a97, 0.00000 // (op 3845) call()
v_accvgpr_write a98, 0.00000 // (op 3845) call()
v_accvgpr_write a99, 0.00000 // (op 3845) call()
v_accvgpr_write a100, 0.00000 // (op 4366) call()
v_accvgpr_write a101, 0.00000 // (op 4366) call()
v_accvgpr_write a102, 0.00000 // (op 4366) call()
v_accvgpr_write a103, 0.00000 // (op 4366) call()
v_accvgpr_write a104, 0.00000 // (op 4367) call()
v_accvgpr_write a105, 0.00000 // (op 4367) call()
v_accvgpr_write a106, 0.00000 // (op 4367) call()
v_accvgpr_write a107, 0.00000 // (op 4367) call()
v_accvgpr_write a108, 0.00000 // (op 4368) call()
v_accvgpr_write a109, 0.00000 // (op 4368) call()
v_accvgpr_write a110, 0.00000 // (op 4368) call()
v_accvgpr_write a111, 0.00000 // (op 4368) call()
v_accvgpr_write a112, 0.00000 // (op 4369) call()
v_accvgpr_write a113, 0.00000 // (op 4369) call()
v_accvgpr_write a114, 0.00000 // (op 4369) call()
v_accvgpr_write a115, 0.00000 // (op 4369) call()
v_accvgpr_write a116, 0.00000 // (op 4370) call()
v_accvgpr_write a117, 0.00000 // (op 4370) call()
v_accvgpr_write a118, 0.00000 // (op 4370) call()
v_accvgpr_write a119, 0.00000 // (op 4370) call()
v_accvgpr_write a120, 0.00000 // (op 4371) call()
v_accvgpr_write a121, 0.00000 // (op 4371) call()
v_accvgpr_write a122, 0.00000 // (op 4371) call()
v_accvgpr_write a123, 0.00000 // (op 4371) call()
v_accvgpr_write a124, 0.00000 // (op 4372) call()
v_accvgpr_write a125, 0.00000 // (op 4372) call()
v_accvgpr_write a126, 0.00000 // (op 4372) call()
v_accvgpr_write a127, 0.00000 // (op 4372) call()
v_lshrrev_b32 v10, 3, v1 // (op 13590) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 13590) 

v_and_b32 v10, -8, v1 // (op 13590) 
v_sub_u32 v12, v1, v10 // (op 13590) 

v_lshlrev_b32 v10, 4, v12 // (op 13590) 

v_mov_b32 v13, 0 // (op 13590) convert
v_mov_b32 v12, v10 // (op 13590) convert

v_mul_lo_u32 v15, v11, s13 // (op 13590) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 13590) most significant: high of low * low
v_mul_lo_u32 v14, v11, s12 // (op 13590) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 13590) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 13590) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 13590) most significant half




s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 13592) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 13592) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 13592) least significant
s_mul_hi_u32 s7, s12, s22 // (op 13592) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13592) most significant: sum
s_add_u32 s1, s1, s7 // (op 13592) most significant: sum


s_mov_b32 s48, s20 // (op 13592) reg expression
s_mov_b32 s49, s21 // (op 13592) reg expression
s_mov_b32 s50, s0 // (op 13592) reg expression
s_mov_b32 s51, 131072 // (op 13592) call()

v_lshrrev_b32 v10, 3, v1 // (op 13609) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 13609) 

v_and_b32 v10, -8, v1 // (op 13609) 
v_sub_u32 v12, v1, v10 // (op 13609) 

v_lshlrev_b32 v10, 4, v12 // (op 13609) 

v_mov_b32 v13, 0 // (op 13609) convert
v_mov_b32 v12, v10 // (op 13609) convert

v_mul_lo_u32 v17, v11, s13 // (op 13609) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 13609) most significant: high of low * low
v_mul_lo_u32 v16, v11, s12 // (op 13609) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 13609) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 13609) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 13609) most significant half



s_mul_i32 s1, s12, s23 // (op 13611) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 13611) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 13611) least significant
s_mul_hi_u32 s7, s12, s22 // (op 13611) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13611) most significant: sum
s_add_u32 s1, s1, s7 // (op 13611) most significant: sum


s_mov_b32 s48, s20 // (op 13611) reg expression
s_mov_b32 s49, s21 // (op 13611) reg expression
s_mov_b32 s50, s0 // (op 13611) reg expression
s_mov_b32 s51, 131072 // (op 13611) call()

v_and_b32 v10, -2, v1 // (op 13654) 
v_sub_u32 v11, v1, v10 // (op 13654) 

v_lshlrev_b32 v10, 6, v11 // (op 13654) 

v_mov_b32 v13, 0 // (op 13654) convert
v_mov_b32 v12, v10 // (op 13654) convert

v_and_b32 v10, 4294967294, v1 // (op 13654) 
v_lshl_add_u32 v11, s17, 8, v10 // (op 13654) 

v_mul_lo_u32 v19, v11, s25 // (op 13654) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 13654) most significant: high of low * low
v_mul_lo_u32 v18, v11, s24 // (op 13654) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 13654) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 13654) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 13654) most significant half



v_mul_lo_u32 v13, v11, 6 // (op 13654) most significant: low of high * low
v_mul_hi_u32 v18, v10, 6 // (op 13654) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 13654) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 13654) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 13654) 

v_mov_b32 v12, v10 // (op 13654) convert

s_mul_i32 s1, s9, 6 // (op 13656) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 13656) least significant
s_mul_hi_u32 s6, s8, 6 // (op 13656) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13656) most significant: sum

s_mul_i32 s7, s24, s1 // (op 13656) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 13656) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 13656) least significant
s_mul_hi_u32 s11, s24, s0 // (op 13656) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13656) most significant: sum
s_add_u32 s7, s7, s11 // (op 13656) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13656) 

s_mov_b32 s52, s26 // (op 13656) reg expression
s_mov_b32 s53, s27 // (op 13656) reg expression
s_mov_b32 s54, s0 // (op 13656) reg expression
s_mov_b32 s55, 131072 // (op 13656) call()

v_and_b32 v13, -2, v1 // (op 13673) 
v_sub_u32 v10, v1, v13 // (op 13673) 

v_lshlrev_b32 v11, 6, v10 // (op 13673) 

v_mov_b32 v19, 0 // (op 13673) convert
v_mov_b32 v18, v11 // (op 13673) convert

v_and_b32 v13, 4294967294, v1 // (op 13673) 
v_lshl_add_u32 v10, s17, 8, v13 // (op 13673) 

v_mul_lo_u32 v21, v10, s25 // (op 13673) most significant: low of low * high
v_mul_hi_u32 v11, v10, s24 // (op 13673) most significant: high of low * low
v_mul_lo_u32 v20, v10, s24 // (op 13673) least significant: low of low * low
v_add_u32 v21, v21, v11 // (op 13673) most significant: sum


v_add_co_u32 v10, s[0:1], v18, v20 // (op 13673) least significant half
v_addc_co_u32 v11, s[0:1], v19, v21, s[0:1] // (op 13673) most significant half



v_mul_lo_u32 v19, v11, 6 // (op 13673) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 13673) most significant: high of low * low
v_mul_lo_u32 v18, v10, 6 // (op 13673) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 13673) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[18:19] // (op 13673) 

v_mov_b32 v13, v10 // (op 13673) convert

s_mul_i32 s1, s9, 6 // (op 13675) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 13675) least significant
s_mul_hi_u32 s6, s8, 6 // (op 13675) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13675) most significant: sum

s_mul_i32 s7, s24, s1 // (op 13675) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 13675) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 13675) least significant
s_mul_hi_u32 s11, s24, s0 // (op 13675) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13675) most significant: sum
s_add_u32 s7, s7, s11 // (op 13675) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13675) 

s_mov_b32 s52, s26 // (op 13675) reg expression
s_mov_b32 s53, s27 // (op 13675) reg expression
s_mov_b32 s54, s0 // (op 13675) reg expression
s_mov_b32 s55, 131072 // (op 13675) call()

v_and_b32 v10, 63, v1 // (op 13937) 
v_lshrrev_b32 v11, 4, v10 // (op 13937) 
v_and_b32 v18, 3, v11 // (op 13937) 

v_and_b32 v11, 15, v1 // (op 13937) 
v_lshl_add_u32 v19, v18, 4, v11 // (op 13937) 


v_lshrrev_b32 v11, 7, v1 // (op 13937) 
v_lshl_add_u32 v18, v11, 6, v19 // (op 13937) 


v_lshl_add_u32 v11, s16, 7, v18 // (op 13937) 

v_lshrrev_b32 v18, 6, v10 // (op 13937) 

v_lshlrev_b32 v10, 2, v18 // (op 13937) 

v_mul_lo_u32 v18, v11, s28 // (op 13937) 
v_add_u32 v18, v18, v10 // (op 13937) 


s_mul_i32 s1, s28, s23 // (op 13939) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 13939) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 13939) least significant
s_mul_hi_u32 s7, s28, s22 // (op 13939) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13939) most significant: sum
s_add_u32 s1, s1, s7 // (op 13939) most significant: sum


s_mov_b32 s56, s30 // (op 13939) reg expression
s_mov_b32 s57, s31 // (op 13939) reg expression
s_mov_b32 s58, s0 // (op 13939) reg expression
s_mov_b32 s59, 131072 // (op 13939) call()

v_and_b32 v10, 63, v1 // (op 13958) 
v_lshrrev_b32 v11, 4, v10 // (op 13958) 
v_and_b32 v19, 3, v11 // (op 13958) 

v_and_b32 v11, 15, v1 // (op 13958) 
v_lshl_add_u32 v20, v19, 4, v11 // (op 13958) 


v_lshrrev_b32 v11, 7, v1 // (op 13958) 
v_lshl_add_u32 v19, v11, 6, v20 // (op 13958) 


v_lshl_add_u32 v11, s16, 7, v19 // (op 13958) 

v_lshrrev_b32 v19, 6, v10 // (op 13958) 

v_lshlrev_b32 v10, 2, v19 // (op 13958) 

v_mul_lo_u32 v19, v11, s28 // (op 13958) 
v_add_u32 v19, v19, v10 // (op 13958) 


s_mul_i32 s1, s28, s23 // (op 13960) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 13960) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 13960) least significant
s_mul_hi_u32 s7, s28, s22 // (op 13960) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13960) most significant: sum
s_add_u32 s1, s1, s7 // (op 13960) most significant: sum


s_mov_b32 s56, s30 // (op 13960) reg expression
s_mov_b32 s57, s31 // (op 13960) reg expression
s_mov_b32 s58, s0 // (op 13960) reg expression
s_mov_b32 s59, 131072 // (op 13960) call()

v_and_b32 v10, 63, v1 // (op 13989) 
v_lshrrev_b32 v11, 4, v10 // (op 13989) 
v_and_b32 v20, 3, v11 // (op 13989) 

v_and_b32 v11, 15, v1 // (op 13989) 
v_lshl_add_u32 v21, v20, 4, v11 // (op 13989) 


v_lshrrev_b32 v11, 6, v1 // (op 13989) 
v_and_b32 v20, 1, v11 // (op 13989) 

v_lshl_add_u32 v11, v20, 7, v21 // (op 13989) 


v_lshl_add_u32 v20, s17, 8, v11 // (op 13989) 

v_mul_lo_u32 v11, v20, s32 // (op 13989) 

v_lshrrev_b32 v20, 6, v10 // (op 13989) 

v_lshl_add_u32 v10, v20, 2, v11 // (op 13989) 


s_mul_i32 s1, s32, s9 // (op 13991) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 13991) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 13991) least significant
s_mul_hi_u32 s7, s32, s8 // (op 13991) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13991) most significant: sum
s_add_u32 s1, s1, s7 // (op 13991) most significant: sum


s_mov_b32 s60, s34 // (op 13991) reg expression
s_mov_b32 s61, s35 // (op 13991) reg expression
s_mov_b32 s62, s0 // (op 13991) reg expression
s_mov_b32 s63, 131072 // (op 13991) call()

v_and_b32 v11, 63, v1 // (op 14010) 
v_lshrrev_b32 v20, 4, v11 // (op 14010) 
v_and_b32 v21, 3, v20 // (op 14010) 

v_and_b32 v20, 15, v1 // (op 14010) 
v_lshl_add_u32 v22, v21, 4, v20 // (op 14010) 


v_lshrrev_b32 v20, 6, v1 // (op 14010) 
v_and_b32 v21, 1, v20 // (op 14010) 

v_lshl_add_u32 v20, v21, 7, v22 // (op 14010) 


v_lshl_add_u32 v21, s17, 8, v20 // (op 14010) 

v_mul_lo_u32 v20, v21, s32 // (op 14010) 

v_lshrrev_b32 v21, 6, v11 // (op 14010) 

v_lshl_add_u32 v11, v21, 2, v20 // (op 14010) 


s_mul_i32 s1, s32, s9 // (op 14012) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 14012) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 14012) least significant
s_mul_hi_u32 s7, s32, s8 // (op 14012) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 14012) most significant: sum
s_add_u32 s1, s1, s7 // (op 14012) most significant: sum


s_mov_b32 s60, s34 // (op 14012) reg expression
s_mov_b32 s61, s35 // (op 14012) reg expression
s_mov_b32 s62, s0 // (op 14012) reg expression
s_mov_b32 s63, 131072 // (op 14012) call()


s_mul_i32 s1, s25, 6 // (op 12920) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 12920) least significant
s_mul_hi_u32 s6, s24, 6 // (op 12920) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 12920) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 12920) 


v_mov_b32 v22, v12 // (op 12920) 
buffer_load_dwordx4 v[24:27], v22, s[52:55], 0 offen // (op 12920) Load value
buffer_load_dwordx2 v[28:29], v22, s[52:55], 0 offen offset:16 // (op 12920) Load value
buffer_load_dwordx4 v[30:33], v22, s[52:55], 0 offen offset:24 // (op 12920) Load value
buffer_load_dwordx2 v[34:35], v22, s[52:55], 0 offen offset:40 // (op 12920) Load value
v_add_u32 v22, v22, v20 // (op 12920) 
buffer_load_dwordx4 v[36:39], v22, s[52:55], 0 offen // (op 12920) Load value
buffer_load_dwordx2 v[40:41], v22, s[52:55], 0 offen offset:16 // (op 12920) Load value
buffer_load_dwordx4 v[42:45], v22, s[52:55], 0 offen offset:24 // (op 12920) Load value
buffer_load_dwordx2 v[46:47], v22, s[52:55], 0 offen offset:40 // (op 12920) Load value


v_and_b32 v20, -8, v1 // (op 13744) 
v_sub_u32 v21, v1, v20 // (op 13744) 

v_lshlrev_b32 v20, 4, v21 // (op 13744) 

v_lshrrev_b32 v21, 3, v1 // (op 13744) 
v_lshl_add_u32 v22, v21, 7, v20 // (op 13744) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 13744) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 13515) 
v_mov_b64 v[20:21], s[6:7] // (op 13515) convert


v_mov_b64 v[22:23], v[14:15] // (op 13515) 
s_mov_b32 m0, s0 // (op 13515) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 13515) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v20 // (op 13515) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 13515) most significant half

s_add_u32 m0, m0, 4096 // (op 13515) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 13515) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v20 // (op 13515) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 13515) most significant half

s_add_u32 m0, m0, 4096 // (op 13515) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 13515) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v20 // (op 13515) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 13515) most significant half

s_add_u32 m0, m0, 4096 // (op 13515) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 13515) Load value direct to lds





v_mov_b32 v20, v10 // (op 76) reg expression

buffer_load_dwordx2 v[22:23], v20, s[60:63], 0 offen // (op 76) Load value



s_lshl_b32 s0, s32, 6 // (op 726) 
v_add_u32 v20, v10, s0 // (op 726) 


buffer_load_dwordx2 v[48:49], v20, s[60:63], 0 offen // (op 726) Load value




v_mov_b32 v10, v18 // (op 72) reg expression

buffer_load_dwordx2 v[20:21], v10, s[56:59], 0 offen // (op 72) Load value



s_mul_i32 s1, s25, 6 // (op 12926) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 12926) least significant
s_mul_hi_u32 s6, s24, 6 // (op 12926) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 12926) most significant: sum

v_lshrrev_b64 v[50:51], 3, s[0:1] // (op 12926) 


v_mov_b32 v10, v12 // (op 12926) 
buffer_load_dwordx4 v[52:55], v10, s[52:55], 0 offen offset:96 // (op 12926) Load value
buffer_load_dwordx2 v[56:57], v10, s[52:55], 0 offen offset:112 // (op 12926) Load value
buffer_load_dwordx4 v[58:61], v10, s[52:55], 0 offen offset:120 // (op 12926) Load value
buffer_load_dwordx2 v[62:63], v10, s[52:55], 0 offen offset:136 // (op 12926) Load value
v_add_u32 v10, v10, v50 // (op 12926) 
buffer_load_dwordx4 v[64:67], v10, s[52:55], 0 offen offset:96 // (op 12926) Load value
buffer_load_dwordx2 v[68:69], v10, s[52:55], 0 offen offset:112 // (op 12926) Load value
buffer_load_dwordx4 v[70:73], v10, s[52:55], 0 offen offset:120 // (op 12926) Load value
buffer_load_dwordx2 v[74:75], v10, s[52:55], 0 offen offset:136 // (op 12926) Load value



v_and_b32 v10, -8, v1 // (op 13869) 
v_sub_u32 v12, v1, v10 // (op 13869) 

v_lshlrev_b32 v10, 4, v12 // (op 13869) 

v_lshrrev_b32 v12, 3, v1 // (op 13869) 
v_lshl_add_u32 v18, v12, 7, v10 // (op 13869) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 13869) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 13520) 
v_mov_b64 v[50:51], s[6:7] // (op 13520) convert


v_mov_b64 v[76:77], v[14:15] // (op 13520) 
s_add_u32 s1, 16384, s0 // (op 13520) 
s_mov_b32 m0, s1 // (op 13520) reg expression
s_mov_b32 s6, 128 // (op 13520) call()
buffer_load_dwordx4 v76, s[48:51], s6 offen offset: 0 lds // (op 13520) Load value direct to lds
v_add_co_u32 v76, s[10:11], v76, v50 // (op 13520) least significant half
v_addc_co_u32 v77, s[10:11], 0, v77, s[10:11] // (op 13520) most significant half

s_add_u32 m0, m0, 4096 // (op 13520) 
buffer_load_dwordx4 v76, s[48:51], s6 offen offset: 0 lds // (op 13520) Load value direct to lds
v_add_co_u32 v76, s[10:11], v76, v50 // (op 13520) least significant half
v_addc_co_u32 v77, s[10:11], 0, v77, s[10:11] // (op 13520) most significant half

s_add_u32 m0, m0, 4096 // (op 13520) 
buffer_load_dwordx4 v76, s[48:51], s6 offen offset: 0 lds // (op 13520) Load value direct to lds
v_add_co_u32 v76, s[10:11], v76, v50 // (op 13520) least significant half
v_addc_co_u32 v77, s[10:11], 0, v77, s[10:11] // (op 13520) most significant half

s_add_u32 m0, m0, 4096 // (op 13520) 
buffer_load_dwordx4 v76, s[48:51], s6 offen offset: 0 lds // (op 13520) Load value direct to lds






v_and_b32 v10, -2, v1 // (op 13802) 
v_sub_u32 v12, v1, v10 // (op 13802) 

v_lshlrev_b32 v10, 6, v12 // (op 13802) 

v_lshrrev_b32 v12, 1, v1 // (op 13802) 
v_lshl_add_u32 v18, v12, 8, v10 // (op 13802) 


v_mul_lo_u32 v10, v18, 6 // (op 13802) 

v_lshrrev_b32 v12, 3, v10 // (op 13802) 




s_waitcnt vmcnt(26)
ds_write_b128 v12, v[24:27] offset:32768 // (op 12931) Store local data 
s_waitcnt vmcnt(25)
ds_write_b64 v12, v[28:29] offset:32784 // (op 12931) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v12, v[30:33] offset:32792 // (op 12931) Store local data 
s_waitcnt vmcnt(23)
ds_write_b64 v12, v[34:35] offset:32808 // (op 12931) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v12, v[36:39] offset:32864 // (op 12931) Store local data 
s_waitcnt vmcnt(21)
ds_write_b64 v12, v[40:41] offset:32880 // (op 12931) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v12, v[42:45] offset:32888 // (op 12931) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v12, v[46:47] offset:32904 // (op 12931) Store local data 

v_and_b32 v10, -8, v1 // (op 13729) 
v_sub_u32 v12, v1, v10 // (op 13729) 

v_lshlrev_b32 v10, 4, v12 // (op 13729) 

v_lshrrev_b32 v12, 3, v1 // (op 13729) 
v_lshl_add_u32 v18, v12, 7, v10 // (op 13729) 


s_nop 0
v_readfirstlane_b32 s0, v18 // (op 13729) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 13786) 
v_sub_u32 v12, v1, v10 // (op 13786) 

v_lshlrev_b32 v10, 6, v12 // (op 13786) 

v_lshrrev_b32 v12, 1, v1 // (op 13786) 
v_lshl_add_u32 v18, v12, 8, v10 // (op 13786) 


v_mul_lo_u32 v10, v18, 6 // (op 13786) 

v_lshrrev_b32 v12, 3, v10 // (op 13786) 

v_and_b32 v10, -8, v1 // (op 13854) 
v_sub_u32 v18, v1, v10 // (op 13854) 

v_lshlrev_b32 v10, 4, v18 // (op 13854) 

v_lshrrev_b32 v18, 3, v1 // (op 13854) 
v_lshl_add_u32 v14, v18, 7, v10 // (op 13854) 


s_nop 0
v_readfirstlane_b32 s1, v14 // (op 13854) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 13898) 
v_sub_u32 v18, v1, v10 // (op 13898) 

v_lshlrev_b32 v10, 6, v18 // (op 13898) 

v_lshrrev_b32 v18, 1, v1 // (op 13898) 
v_lshl_add_u32 v14, v18, 8, v10 // (op 13898) 


v_mul_lo_u32 v10, v14, 6 // (op 13898) 

v_lshrrev_b32 v18, 3, v10 // (op 13898) 

s_mov_b32 s6, 0 // (op 4036) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 4035) 
s_mov_b32 s7, s10 // (op 4035) convert

s_and_b32 s10, s7, 4294967294 // (op 4035) 

s_cmp_lt_u32 s6, s10 // (op 4035) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoop_4035 // (op 4035) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoop_4035 if false)
RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoop_4035:
s_add_u32 s6, s6, 2 // (op 4037) 
s_waitcnt vmcnt(15)
s_barrier  // (op 14210) 



ds_read_b128 v[76:79], v3 // (op 62) Load local data 
ds_read_b128 v[80:83], v3 offset:64 // (op 62) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 14226) 



ds_read_b128 v[84:87], v4 offset:32768 // (op 67) Load local data 
ds_read_b64 v[88:89], v4 offset:32784 // (op 67) Load local data 



ds_read_b128 v[90:93], v4 offset:34304 // (op 312) Load local data 
ds_read_b64 v[94:95], v4 offset:34320 // (op 312) Load local data 
s_waitcnt vmcnt(14)
v_mov_b32 v14, v22 // (op 13435) reg expression



v_mov_b32 v15, v23 // (op 13435) reg expression
s_waitcnt vmcnt(12)
v_mov_b32 v50, v20 // (op 13462) reg expression
v_mov_b32 v51, v21 // (op 13462) reg expression
ds_read_b128 v[96:99], v4 offset:35840 // (op 448) Load local data 
v_bfe_u32 v102, v50, 0, 8 // (op 12241) 
v_bfe_u32 v103, v50, 8, 8 // (op 12241) 
v_bfe_u32 v104, v50, 16, 8 // (op 12241) 
v_bfe_u32 v105, v50, 24, 8 // (op 12241) 
v_permlane16_swap_b32 v102, v103 // (op 12241) 
s_nop 0
v_permlane16_swap_b32 v104, v105 // (op 12241) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v102, v104 // (op 12241) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v103, v105 // (op 12241) 
ds_read_b64 v[100:101], v4 offset:35856 // (op 448) Load local data 



v_bfe_u32 v106, v14, 0, 8 // (op 12336) 
v_bfe_u32 v107, v14, 8, 8 // (op 12336) 
v_bfe_u32 v108, v14, 16, 8 // (op 12336) 
v_bfe_u32 v109, v14, 24, 8 // (op 12336) 
v_permlane16_swap_b32 v106, v107 // (op 12336) 
ds_read_b128 v[110:113], v4 offset:37376 // (op 584) Load local data 
v_permlane16_swap_b32 v108, v109 // (op 12336) 
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 12336) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 12336) 
ds_read_b64 v[114:115], v4 offset:37392 // (op 584) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:83], v[84:89], a[0:3], v102, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 78) 
ds_read_b128 v[116:119], v4 offset:38912 // (op 720) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:83], v[90:95], a[4:7], v102, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 315) 
ds_read_b64 v[120:121], v4 offset:38928 // (op 720) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:83], v[96:101], a[8:11], v102, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 451) 

s_mul_i32 s11, s25, 6 // (op 10) most significant: low of high * low
s_mul_i32 s10, s24, 6 // (op 10) least significant
s_mul_hi_u32 s7, s24, 6 // (op 10) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 10) most significant: sum

v_lshrrev_b64 v[122:123], 3, s[10:11] // (op 10) 


v_mov_b32 v10, v13 // (op 10) 
v_mov_b32 v124, v10 // (op 10) 
buffer_load_dwordx4 v[24:27], v124, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[28:29], v124, s[52:55], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[30:33], v124, s[52:55], 0 offen offset:216 // (op 10) Load value
v_mov_b32 v126, v48 // (op 13448) reg expression
v_mov_b32 v127, v49 // (op 13448) reg expression
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:83], v[110:115], a[12:15], v102, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 587) 
v_bfe_u32 v128, v126, 0, 8 // (op 12368) 
v_bfe_u32 v129, v126, 8, 8 // (op 12368) 
v_bfe_u32 v130, v126, 16, 8 // (op 12368) 
v_bfe_u32 v131, v126, 24, 8 // (op 12368) 
v_permlane16_swap_b32 v128, v129 // (op 12368) 
ds_read_b128 v[132:135], v4 offset:40448 // (op 856) Load local data 
v_permlane16_swap_b32 v130, v131 // (op 12368) 
s_nop 1
v_permlane32_swap_b32 v128, v130 // (op 12368) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v129, v131 // (op 12368) 
ds_read_b64 v[136:137], v4 offset:40464 // (op 856) Load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:83], v[116:121], a[16:19], v102, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 723) 
ds_read_b128 v[138:141], v4 offset:41984 // (op 992) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:83], v[132:137], a[20:23], v102, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 859) 
ds_read_b64 v[142:143], v4 offset:42000 // (op 992) Load local data 
buffer_load_dwordx2 v[34:35], v124, s[52:55], 0 offen offset:232 // (op 10) Load value



v_add_u32 v124, v124, v122 // (op 10) 
buffer_load_dwordx4 v[36:39], v124, s[52:55], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[40:41], v124, s[52:55], 0 offen offset:208 // (op 10) Load value
ds_read_b128 v[144:147], v4 offset:43520 // (op 1128) Load local data 
ds_read_b64 v[148:149], v4 offset:43536 // (op 1128) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:83], v[138:143], a[24:27], v102, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 995) 
s_barrier  // (op 14219) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:83], v[144:149], a[28:31], v102, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1131) 




ds_read_b128 v[76:79], v3 offset:2048 // (op 1629) Load local data 
ds_read_b128 v[80:83], v3 offset:2112 // (op 1629) Load local data 
buffer_load_dwordx4 v[42:45], v124, s[52:55], 0 offen offset:216 // (op 10) Load value



buffer_load_dwordx2 v[46:47], v124, s[52:55], 0 offen offset:232 // (op 10) Load value


ds_read_b128 v[150:153], v3 offset:4096 // (op 2847) Load local data 
ds_read_b128 v[154:157], v3 offset:4160 // (op 2847) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:83], v[84:89], a[32:35], v103, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1899) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:83], v[90:95], a[36:39], v103, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1861) 
ds_read_b128 v[158:161], v3 offset:6144 // (op 4065) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:83], v[96:101], a[40:43], v103, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1823) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14213) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:83], v[110:115], a[44:47], v103, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1785) 
ds_read_b128 v[162:165], v3 offset:6208 // (op 4065) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:83], v[116:121], a[48:51], v103, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1747) 
s_lshl_b64 s[10:11], s[12:13], 5 // (op 13499) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:83], v[132:137], a[52:55], v103, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1709) 
v_mov_b64 v[122:123], s[10:11] // (op 13499) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:83], v[138:143], a[56:59], v103, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1671) 
v_mov_b64 v[124:125], v[16:17] // (op 13499) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:83], v[144:149], a[60:63], v103, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1633) 
v_mov_b64 v[166:167], v[124:125] // (op 13499) 

s_mov_b32 m0, s0 // (op 13499) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[150:157], v[84:89], a[64:67], v104, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3117) 
s_mov_b32 s7, 256 // (op 13499) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[150:157], v[90:95], a[68:71], v104, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3079) 
buffer_load_dwordx4 v166, s[48:51], s7 offen offset: 0 lds // (op 13499) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[150:157], v[96:101], a[72:75], v104, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3041) 
v_add_co_u32 v166, s[10:11], v166, v122 // (op 13499) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[150:157], v[110:115], a[76:79], v104, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3003) 
v_addc_co_u32 v167, s[10:11], 0, v167, s[10:11] // (op 13499) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[150:157], v[116:121], a[80:83], v104, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2965) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[150:157], v[132:137], a[84:87], v104, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2927) 
s_add_u32 m0, m0, 4096 // (op 13499) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[150:157], v[138:143], a[88:91], v104, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2889) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[150:157], v[144:149], a[92:95], v104, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2851) 
buffer_load_dwordx4 v166, s[48:51], s7 offen offset: 0 lds // (op 13499) Load value direct to lds

v_add_co_u32 v166, s[10:11], v166, v122 // (op 13499) least significant half
v_addc_co_u32 v167, s[10:11], 0, v167, s[10:11] // (op 13499) most significant half

s_add_u32 m0, m0, 4096 // (op 13499) 
buffer_load_dwordx4 v166, s[48:51], s7 offen offset: 0 lds // (op 13499) Load value direct to lds
v_add_co_u32 v166, s[10:11], v166, v122 // (op 13499) least significant half
v_addc_co_u32 v167, s[10:11], 0, v167, s[10:11] // (op 13499) most significant half

s_add_u32 m0, m0, 4096 // (op 13499) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[158:165], v[84:89], a[96:99], v105, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4335) 
buffer_load_dwordx4 v166, s[48:51], s7 offen offset: 0 lds // (op 13499) Load value direct to lds






s_barrier  // (op 14252) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[158:165], v[90:95], a[100:103], v105, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4297) 




v_mov_b32 v76, v11 // (op 13443) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[158:165], v[96:101], a[104:107], v105, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4259) 
v_add_u32 v77, 8, v76 // (op 13443) 


s_waitcnt vmcnt(23)
ds_write_b128 v18, v[52:55] offset:57344 // (op 222) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[158:165], v[110:115], a[108:111], v105, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4221) 
s_waitcnt vmcnt(22)
ds_write_b64 v18, v[56:57] offset:57360 // (op 222) Store local data 


s_waitcnt vmcnt(21)
ds_write_b128 v18, v[58:61] offset:57368 // (op 222) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[158:165], v[116:121], a[112:115], v105, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4183) 
s_waitcnt vmcnt(20)
ds_write_b64 v18, v[62:63] offset:57384 // (op 222) Store local data 

s_waitcnt vmcnt(19)
ds_write_b128 v18, v[64:67] offset:57440 // (op 222) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[158:165], v[132:137], a[116:119], v105, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4145) 
s_waitcnt vmcnt(18)
ds_write_b64 v18, v[68:69] offset:57456 // (op 222) Store local data 

s_waitcnt vmcnt(17)
ds_write_b128 v18, v[70:73] offset:57464 // (op 222) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[158:165], v[138:143], a[120:123], v105, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4107) 
s_waitcnt vmcnt(16)
ds_write_b64 v18, v[74:75] offset:57480 // (op 222) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[158:165], v[144:149], a[124:127], v105, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4069) 
buffer_load_dwordx2 v[22:23], v77, s[60:63], 0 offen // (op 13443) Load value







s_lshl_b32 s7, s32, 6 // (op 13457) 
s_add_u32 s7, s7, 8 // (op 13457) 
v_add_u32 v77, v76, s7 // (op 13457) 


buffer_load_dwordx2 v[48:49], v77, s[60:63], 0 offen // (op 13457) Load value

s_waitcnt vmcnt(14)
s_barrier  // (op 14240) 






v_add_u32 v11, 8, v11 // (op 14000) 
v_mov_b32 v76, v19 // (op 13470) 
v_add_u32 v77, 8, v76 // (op 13470) 

ds_read_b128 v[78:81], v5 offset:16384 // (op 214) Load local data 
ds_read_b128 v[82:85], v5 offset:16448 // (op 214) Load local data 
buffer_load_dwordx2 v[20:21], v77, s[56:59], 0 offen // (op 13470) Load value


v_add_u32 v19, 8, v19 // (op 13948) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 14249) 




s_mul_i32 s11, s25, 6 // (op 223) most significant: low of high * low
s_mul_i32 s10, s24, 6 // (op 223) least significant
s_mul_hi_u32 s7, s24, 6 // (op 223) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 223) most significant: sum

v_lshrrev_b64 v[76:77], 3, s[10:11] // (op 223) 


v_mov_b32 v86, v10 // (op 223) 
ds_read_b128 v[118:121], v6 offset:57344 // (op 216) Load local data 
ds_read_b64 v[122:123], v6 offset:57360 // (op 216) Load local data 
buffer_load_dwordx4 v[52:55], v86, s[52:55], 0 offen offset:288 // (op 223) Load value



v_bfe_u32 v114, v51, 0, 8 // (op 12256) 
v_bfe_u32 v115, v51, 8, 8 // (op 12256) 
v_bfe_u32 v116, v51, 16, 8 // (op 12256) 
buffer_load_dwordx2 v[56:57], v86, s[52:55], 0 offen offset:304 // (op 223) Load value
v_bfe_u32 v117, v51, 24, 8 // (op 12256) 
v_permlane16_swap_b32 v114, v115 // (op 12256) 
ds_read_b128 v[108:111], v6 offset:58880 // (op 324) Load local data 
ds_read_b64 v[112:113], v6 offset:58896 // (op 324) Load local data 
v_permlane16_swap_b32 v116, v117 // (op 12256) 



ds_read_b128 v[102:105], v6 offset:60416 // (op 460) Load local data 
s_nop 0
v_permlane32_swap_b32 v114, v116 // (op 12256) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v115, v117 // (op 12256) 
buffer_load_dwordx4 v[58:61], v86, s[52:55], 0 offen offset:312 // (op 223) Load value

ds_read_b64 v[106:107], v6 offset:60432 // (op 460) Load local data 



v_bfe_u32 v98, v15, 0, 8 // (op 12351) 
v_bfe_u32 v99, v15, 8, 8 // (op 12351) 
v_bfe_u32 v100, v15, 16, 8 // (op 12351) 
v_bfe_u32 v101, v15, 24, 8 // (op 12351) 
v_permlane16_swap_b32 v98, v99 // (op 12351) 
ds_read_b128 v[92:95], v6 offset:61952 // (op 596) Load local data 
ds_read_b64 v[96:97], v6 offset:61968 // (op 596) Load local data 
v_permlane16_swap_b32 v100, v101 // (op 12351) 



s_nop 1
v_permlane32_swap_b32 v98, v100 // (op 12351) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v99, v101 // (op 12351) 
ds_read_b128 v[128:131], v6 offset:63488 // (op 732) Load local data 

ds_read_b64 v[132:133], v6 offset:63504 // (op 732) Load local data 



s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:85], v[118:123], a[0:3], v114, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 221) 
ds_read_b128 v[134:137], v6 offset:65024 // (op 868) Load local data 
v_bfe_u32 v88, v127, 0, 8 // (op 12383) 
v_bfe_u32 v89, v127, 8, 8 // (op 12383) 
v_bfe_u32 v90, v127, 16, 8 // (op 12383) 
v_bfe_u32 v91, v127, 24, 8 // (op 12383) 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:85], v[108:113], a[4:7], v114, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 316) 
v_permlane16_swap_b32 v88, v89 // (op 12383) 
v_permlane16_swap_b32 v90, v91 // (op 12383) 
s_nop 1
v_permlane32_swap_b32 v88, v90 // (op 12383) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v89, v91 // (op 12383) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[102:107], a[8:11], v114, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 452) 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:85], v[92:97], a[12:15], v114, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 588) 
ds_read_b64 v[138:139], v6 offset:65040 // (op 868) Load local data 



v_add_u32 v87, 66560, v6 // (op 1004) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:85], v[128:133], a[16:19], v114, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 724) 
ds_read_b128 v[140:143], v87 // (op 1004) Load local data 
ds_read_b64 v[144:145], v87 offset:16 // (op 1004) Load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[134:139], a[20:23], v114, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 860) 



v_add_u32 v87, 68096, v6 // (op 1140) 
ds_read_b128 v[146:149], v87 // (op 1140) Load local data 
ds_read_b64 v[150:151], v87 offset:16 // (op 1140) Load local data 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:85], v[140:145], a[24:27], v114, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 996) 
buffer_load_dwordx2 v[62:63], v86, s[52:55], 0 offen offset:328 // (op 223) Load value
s_barrier  // (op 14246) 
s_barrier  // (op 14295) 



v_add_u32 v86, v86, v76 // (op 223) 
buffer_load_dwordx4 v[64:67], v86, s[52:55], 0 offen offset:288 // (op 223) Load value
ds_read_b128 v[152:155], v5 offset:18432 // (op 1641) Load local data 
ds_read_b128 v[156:159], v5 offset:18496 // (op 1641) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:85], v[146:151], a[28:31], v114, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1132) 




ds_read_b128 v[78:81], v5 offset:20480 // (op 2859) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[152:159], v[118:123], a[32:35], v115, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1900) 
ds_read_b128 v[82:85], v5 offset:20544 // (op 2859) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[152:159], v[102:107], a[40:43], v115, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1824) 
ds_read_b128 v[160:163], v5 offset:22528 // (op 4077) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[152:159], v[92:97], a[44:47], v115, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1786) 
ds_read_b128 v[164:167], v5 offset:22592 // (op 4077) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[152:159], v[108:113], a[36:39], v115, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1862) 
buffer_load_dwordx2 v[68:69], v86, s[52:55], 0 offen offset:304 // (op 223) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:159], v[128:133], a[48:51], v115, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1748) 
buffer_load_dwordx4 v[70:73], v86, s[52:55], 0 offen offset:312 // (op 223) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[152:159], v[140:145], a[56:59], v115, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1672) 
buffer_load_dwordx2 v[74:75], v86, s[52:55], 0 offen offset:328 // (op 223) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[152:159], v[146:151], a[60:63], v115, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1634) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:159], v[134:139], a[52:55], v115, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1710) 
v_add_u32 v13, 192, v13 // (op 13665) 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14243) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:85], v[118:123], a[64:67], v116, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3118) 
s_lshl_b64 s[10:11], s[12:13], 5 // (op 13510) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:85], v[102:107], a[72:75], v116, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3042) 
v_mov_b64 v[14:15], s[10:11] // (op 13510) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[78:85], v[92:97], a[76:79], v116, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3004) 
v_mov_b64 v[50:51], v[124:125] // (op 13510) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:85], v[108:113], a[68:71], v116, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3080) 
s_add_u32 s7, 16384, s1 // (op 13510) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[78:85], v[128:133], a[80:83], v116, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2966) 
s_mov_b32 m0, s7 // (op 13510) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[78:85], v[140:145], a[88:91], v116, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2890) 
s_mov_b32 s10, 384 // (op 13510) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[78:85], v[146:151], a[92:95], v116, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2852) 
buffer_load_dwordx4 v50, s[48:51], s10 offen offset: 0 lds // (op 13510) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:85], v[134:139], a[84:87], v116, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2928) 
v_add_co_u32 v50, s[14:15], v50, v14 // (op 13510) least significant half

v_addc_co_u32 v51, s[14:15], 0, v51, s[14:15] // (op 13510) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[160:167], v[118:123], a[96:99], v117, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4336) 

s_add_u32 m0, m0, 4096 // (op 13510) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[160:167], v[102:107], a[104:107], v117, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4260) 

buffer_load_dwordx4 v50, s[48:51], s10 offen offset: 0 lds // (op 13510) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[160:167], v[92:97], a[108:111], v117, v101 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4222) 
v_add_co_u32 v50, s[14:15], v50, v14 // (op 13510) least significant half

v_addc_co_u32 v51, s[14:15], 0, v51, s[14:15] // (op 13510) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[160:167], v[108:113], a[100:103], v117, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4298) 


s_add_u32 m0, m0, 4096 // (op 13510) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[160:167], v[128:133], a[112:115], v117, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4184) 

buffer_load_dwordx4 v50, s[48:51], s10 offen offset: 0 lds // (op 13510) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[160:167], v[140:145], a[120:123], v117, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4108) 
v_add_co_u32 v50, s[14:15], v50, v14 // (op 13510) least significant half

v_addc_co_u32 v51, s[14:15], 0, v51, s[14:15] // (op 13510) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[160:167], v[146:151], a[124:127], v117, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4070) 

s_add_u32 m0, m0, 4096 // (op 13510) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[160:167], v[134:139], a[116:119], v117, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4146) 




buffer_load_dwordx4 v50, s[48:51], s10 offen offset: 0 lds // (op 13510) Load value direct to lds





s_barrier  // (op 14229) 
v_add_co_u32 v16, vcc, 256, v16 // (op 13601) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 13601) most significant half
s_waitcnt vmcnt(26)
ds_write_b128 v12, v[24:27] offset:32768 // (op 13) Store local data 
s_waitcnt vmcnt(25)
ds_write_b64 v12, v[28:29] offset:32784 // (op 13) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v12, v[30:33] offset:32792 // (op 13) Store local data 
s_waitcnt vmcnt(23)
ds_write_b64 v12, v[34:35] offset:32808 // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v12, v[36:39] offset:32864 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b64 v12, v[40:41] offset:32880 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v12, v[42:45] offset:32888 // (op 13) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v12, v[46:47] offset:32904 // (op 13) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 4035) 
s_mov_b32 s7, s10 // (op 4035) convert

s_and_b32 s10, s7, 4294967294 // (op 4035) 

s_cmp_lt_u32 s6, s10 // (op 4035) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoop_4035 // (op 4035) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoop_4035 if true)
RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoop_4035:












s_barrier  // (op 14331) 
v_lshrrev_b32 v10, 3, v1 // (op 13533) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 13533) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 13533) 

s_lshr_b32 s6, s0, 1 // (op 13533) 

v_and_b32 v10, -8, v1 // (op 13533) 
v_sub_u32 v12, v1, v10 // (op 13533) 

v_lshlrev_b32 v10, 4, v12 // (op 13533) 

v_lshl_add_u32 v12, s6, 8, v10 // (op 13533) 


v_mov_b32 v51, 0 // (op 13533) convert
v_mov_b32 v50, v12 // (op 13533) convert

v_mul_lo_u32 v13, v11, s13 // (op 13533) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 13533) most significant: high of low * low
v_mul_lo_u32 v12, v11, s12 // (op 13533) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 13533) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v50 // (op 13533) least significant half
v_addc_co_u32 v13, s[0:1], v13, v51, s[0:1] // (op 13533) most significant half



s_mul_i32 s1, s12, s23 // (op 13535) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 13535) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 13535) least significant
s_mul_hi_u32 s7, s12, s22 // (op 13535) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13535) most significant: sum
s_add_u32 s1, s1, s7 // (op 13535) most significant: sum


s_mov_b32 s48, s20 // (op 13535) reg expression
s_mov_b32 s49, s21 // (op 13535) reg expression
s_mov_b32 s50, s0 // (op 13535) reg expression
s_mov_b32 s51, 131072 // (op 13535) call()

v_and_b32 v10, -2, v1 // (op 13631) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 13631) 
v_sub_u32 v11, v1, v10 // (op 13631) 

s_lshr_b32 s6, s0, 1 // (op 13631) 

v_lshlrev_b32 v10, 6, v11 // (op 13631) 

v_lshl_add_u32 v11, s6, 8, v10 // (op 13631) 


v_mov_b32 v51, 0 // (op 13631) convert
v_mov_b32 v50, v11 // (op 13631) convert

v_and_b32 v10, 4294967294, v1 // (op 13631) 
v_lshl_add_u32 v11, s17, 8, v10 // (op 13631) 

v_mul_lo_u32 v15, v11, s25 // (op 13631) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 13631) most significant: high of low * low
v_mul_lo_u32 v14, v11, s24 // (op 13631) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 13631) most significant: sum


v_add_co_u32 v10, s[0:1], v50, v14 // (op 13631) least significant half
v_addc_co_u32 v11, s[0:1], v51, v15, s[0:1] // (op 13631) most significant half



v_mul_lo_u32 v51, v11, 6 // (op 13631) most significant: low of high * low
v_mul_hi_u32 v14, v10, 6 // (op 13631) most significant: high of low * low
v_mul_lo_u32 v50, v10, 6 // (op 13631) least significant: low of low * low
v_add_u32 v51, v51, v14 // (op 13631) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[50:51] // (op 13631) 

v_mov_b32 v14, v10 // (op 13631) convert

s_mul_i32 s1, s9, 6 // (op 13633) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 13633) least significant
s_mul_hi_u32 s6, s8, 6 // (op 13633) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13633) most significant: sum

s_mul_i32 s7, s24, s1 // (op 13633) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 13633) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 13633) least significant
s_mul_hi_u32 s11, s24, s0 // (op 13633) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13633) most significant: sum
s_add_u32 s7, s7, s11 // (op 13633) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13633) 

s_mov_b32 s52, s26 // (op 13633) reg expression
s_mov_b32 s53, s27 // (op 13633) reg expression
s_mov_b32 s54, s0 // (op 13633) reg expression
s_mov_b32 s55, 131072 // (op 13633) call()

v_and_b32 v10, -8, v1 // (op 13714) 
v_sub_u32 v11, v1, v10 // (op 13714) 

v_lshlrev_b32 v10, 4, v11 // (op 13714) 

v_lshrrev_b32 v11, 3, v1 // (op 13714) 
v_lshl_add_u32 v15, v11, 7, v10 // (op 13714) 


s_nop 0
v_readfirstlane_b32 s0, v15 // (op 13714) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 13773) 
v_sub_u32 v11, v1, v10 // (op 13773) 

v_lshlrev_b32 v10, 6, v11 // (op 13773) 

v_lshrrev_b32 v11, 1, v1 // (op 13773) 
v_lshl_add_u32 v15, v11, 8, v10 // (op 13773) 


v_mul_lo_u32 v10, v15, 6 // (op 13773) 

v_lshrrev_b32 v11, 3, v10 // (op 13773) 

v_and_b32 v10, 63, v1 // (op 14037) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 14037) 
v_lshrrev_b32 v15, 4, v10 // (op 14037) 
s_lshr_b32 s1, s6, 1 // (op 14037) 

v_and_b32 v16, 3, v15 // (op 14037) 

v_and_b32 v15, 15, v1 // (op 14037) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 14037) 


v_lshrrev_b32 v15, 7, v1 // (op 14037) 
v_lshl_add_u32 v16, v15, 6, v17 // (op 14037) 


v_lshl_add_u32 v15, s16, 7, v16 // (op 14037) 

v_lshrrev_b32 v16, 6, v10 // (op 14037) 

v_lshlrev_b32 v10, 2, v16 // (op 14037) 

v_lshl_add_u32 v16, s1, 3, v10 // (op 14037) 


v_mul_lo_u32 v10, v15, s28 // (op 14037) 
v_add_u32 v10, v10, v16 // (op 14037) 


s_mul_i32 s7, s28, s23 // (op 14039) most significant: low of low * high
s_mul_i32 s1, s29, s22 // (op 14039) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 14039) least significant
s_mul_hi_u32 s10, s28, s22 // (op 14039) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 14039) most significant: sum
s_add_u32 s7, s7, s10 // (op 14039) most significant: sum


s_mov_b32 s56, s30 // (op 14039) reg expression
s_mov_b32 s57, s31 // (op 14039) reg expression
s_mov_b32 s58, s6 // (op 14039) reg expression
s_mov_b32 s59, 131072 // (op 14039) call()

v_and_b32 v15, 63, v1 // (op 14064) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 14064) 
v_lshrrev_b32 v16, 6, v15 // (op 14064) 
s_lshr_b32 s1, s6, 1 // (op 14064) 

v_lshlrev_b32 v17, 2, v16 // (op 14064) 

v_lshl_add_u32 v16, s1, 3, v17 // (op 14064) 


v_lshrrev_b32 v17, 4, v15 // (op 14064) 

v_and_b32 v15, 3, v17 // (op 14064) 

v_and_b32 v17, 15, v1 // (op 14064) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 14064) 


v_lshrrev_b32 v15, 6, v1 // (op 14064) 
v_and_b32 v17, 1, v15 // (op 14064) 

v_lshl_add_u32 v15, v17, 7, v18 // (op 14064) 


v_lshl_add_u32 v17, s17, 8, v15 // (op 14064) 

v_mul_lo_u32 v15, v17, s32 // (op 14064) 

v_add_u32 v17, v16, v15 // (op 14064) 


s_mul_i32 s7, s32, s9 // (op 14066) most significant: low of low * high
s_mul_i32 s1, s33, s8 // (op 14066) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 14066) least significant
s_mul_hi_u32 s10, s32, s8 // (op 14066) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 14066) most significant: sum
s_add_u32 s7, s7, s10 // (op 14066) most significant: sum


s_mov_b32 s60, s34 // (op 14066) reg expression
s_mov_b32 s61, s35 // (op 14066) reg expression
s_mov_b32 s62, s6 // (op 14066) reg expression
s_mov_b32 s63, 131072 // (op 14066) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 4014) 
s_and_b32 s1, s6, 4294967294 // (op 4014) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 4013) 
s_mov_b32 s10, s6 // (op 4013) convert

s_cmp_lt_u32 s1, s10 // (op 4013) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoopTail_4013 // (op 4013) Condition: Top (jump to RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoopTail_4013 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoopTail_4013:
s_barrier  // (op 14216) 
s_add_u32 s1, s1, 1 // (op 4015) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 13504) 
v_mov_b64 v[18:19], s[6:7] // (op 13504) convert


v_mov_b64 v[20:21], v[12:13] // (op 13504) 
v_mov_b64 v[22:23], v[20:21] // (op 13504) 
s_mov_b32 m0, s0 // (op 13504) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 13504) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 13504) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 13504) most significant half

s_add_u32 m0, m0, 4096 // (op 13504) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 13504) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 13504) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 13504) most significant half

s_add_u32 m0, m0, 4096 // (op 13504) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 13504) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 13504) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 13504) most significant half

s_add_u32 m0, m0, 4096 // (op 13504) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 13504) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 13523) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 13523) most significant half

s_mul_i32 s7, s25, 6 // (op 173) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 173) least significant
s_mul_hi_u32 s10, s24, 6 // (op 173) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 173) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[6:7] // (op 173) 


v_mov_b32 v15, v14 // (op 173) 
v_mov_b32 v16, v15 // (op 173) 
buffer_load_dwordx4 v[20:23], v16, s[52:55], 0 offen // (op 173) Load value
buffer_load_dwordx2 v[24:25], v16, s[52:55], 0 offen offset:16 // (op 173) Load value
buffer_load_dwordx4 v[26:29], v16, s[52:55], 0 offen offset:24 // (op 173) Load value
buffer_load_dwordx2 v[30:31], v16, s[52:55], 0 offen offset:40 // (op 173) Load value
v_add_u32 v16, v16, v18 // (op 173) 
buffer_load_dwordx4 v[32:35], v16, s[52:55], 0 offen // (op 173) Load value
buffer_load_dwordx2 v[36:37], v16, s[52:55], 0 offen offset:16 // (op 173) Load value
buffer_load_dwordx4 v[38:41], v16, s[52:55], 0 offen offset:24 // (op 173) Load value
buffer_load_dwordx2 v[42:43], v16, s[52:55], 0 offen offset:40 // (op 173) Load value



v_add_u32 v14, 96, v14 // (op 13623) 
s_barrier  // (op 14232) 
s_waitcnt vmcnt(8)
s_barrier  // (op 14222) 






ds_read_b128 v[44:47], v3 // (op 164) Load local data 
ds_read_b128 v[48:51], v3 offset:64 // (op 164) Load local data 
s_waitcnt vmcnt(7)
ds_write_b128 v11, v[20:23] offset:32768 // (op 172) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v11, v[24:25] offset:32784 // (op 172) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v11, v[26:29] offset:32792 // (op 172) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v11, v[30:31] offset:32808 // (op 172) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v11, v[32:35] offset:32864 // (op 172) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v11, v[36:37] offset:32880 // (op 172) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v11, v[38:41] offset:32888 // (op 172) Store local data 
s_waitcnt vmcnt(0)
ds_write_b64 v11, v[42:43] offset:32904 // (op 172) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14236) 



ds_read_b128 v[18:21], v4 offset:32768 // (op 166) Load local data 
ds_read_b64 v[22:23], v4 offset:32784 // (op 166) Load local data 


v_mov_b32 v15, v10 // (op 168) 
v_mov_b32 v16, v15 // (op 168) reg expression

buffer_load_dwordx2 v[24:25], v16, s[56:59], 0 offen // (op 168) Load value


v_add_u32 v10, 4, v10 // (op 14027) 


v_mov_b32 v15, v17 // (op 170) 
v_mov_b32 v16, v15 // (op 170) reg expression

buffer_load_dwordx2 v[26:27], v16, s[60:63], 0 offen // (op 170) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v28, v24, 0, 8 // (op 12464) 



v_bfe_u32 v29, v24, 8, 8 // (op 12464) 
v_bfe_u32 v30, v24, 16, 8 // (op 12464) 
v_bfe_u32 v31, v24, 24, 8 // (op 12464) 
v_permlane16_swap_b32 v28, v29 // (op 12464) 
ds_read_b128 v[32:35], v4 offset:34304 // (op 295) Load local data 
v_permlane16_swap_b32 v30, v31 // (op 12464) 
s_nop 1
v_permlane32_swap_b32 v28, v30 // (op 12464) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v29, v31 // (op 12464) 
ds_read_b64 v[36:37], v4 offset:34320 // (op 295) Load local data 




ds_read_b128 v[38:41], v4 offset:35840 // (op 431) Load local data 
ds_read_b64 v[42:43], v4 offset:35856 // (op 431) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v52, v26, 0, 8 // (op 12510) 



v_bfe_u32 v53, v26, 8, 8 // (op 12510) 
v_bfe_u32 v54, v26, 16, 8 // (op 12510) 
v_bfe_u32 v55, v26, 24, 8 // (op 12510) 
v_permlane16_swap_b32 v52, v53 // (op 12510) 
s_nop 0
v_permlane16_swap_b32 v54, v55 // (op 12510) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 12510) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 12510) 
ds_read_b128 v[56:59], v4 offset:37376 // (op 567) Load local data 

ds_read_b64 v[60:61], v4 offset:37392 // (op 567) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[18:23], a[0:3], v28, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 171) 
ds_read_b128 v[62:65], v4 offset:38912 // (op 703) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[32:37], a[4:7], v28, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 298) 
ds_read_b64 v[66:67], v4 offset:38928 // (op 703) Load local data 


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[44:51], v[38:43], a[8:11], v28, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 434) 
s_lshl_b32 s6, s32, 6 // (op 708) 
v_add_u32 v16, v15, s6 // (op 708) 


buffer_load_dwordx2 v[24:25], v16, s[60:63], 0 offen // (op 708) Load value

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[44:51], v[56:61], a[12:15], v28, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 570) 




v_add_u32 v17, 4, v17 // (op 14054) 
ds_read_b128 v[68:71], v4 offset:40448 // (op 839) Load local data 
ds_read_b64 v[72:73], v4 offset:40464 // (op 839) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v74, v24, 0, 8 // (op 12524) 



v_bfe_u32 v75, v24, 8, 8 // (op 12524) 
v_bfe_u32 v76, v24, 16, 8 // (op 12524) 
v_bfe_u32 v77, v24, 24, 8 // (op 12524) 
v_permlane16_swap_b32 v74, v75 // (op 12524) 
s_nop 0
v_permlane16_swap_b32 v76, v77 // (op 12524) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v74, v76 // (op 12524) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v75, v77 // (op 12524) 
ds_read_b128 v[78:81], v4 offset:41984 // (op 975) Load local data 

ds_read_b64 v[82:83], v4 offset:42000 // (op 975) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[44:51], v[62:67], a[16:19], v28, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 706) 
ds_read_b128 v[84:87], v4 offset:43520 // (op 1111) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[44:51], v[68:73], a[20:23], v28, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 842) 
ds_read_b64 v[88:89], v4 offset:43536 // (op 1111) Load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[44:51], v[78:83], a[24:27], v28, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 978) 
ds_read_b128 v[90:93], v3 offset:2048 // (op 1421) Load local data 
ds_read_b128 v[94:97], v3 offset:2112 // (op 1421) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[44:51], v[84:89], a[28:31], v28, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1114) 




ds_read_b128 v[44:47], v3 offset:4096 // (op 2639) Load local data 
ds_read_b128 v[48:51], v3 offset:4160 // (op 2639) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:97], v[18:23], a[32:35], v29, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1425) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:97], v[38:43], a[40:43], v29, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1479) 
ds_read_b128 v[98:101], v3 offset:6144 // (op 3857) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:97], v[56:61], a[44:47], v29, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1506) 
ds_read_b128 v[102:105], v3 offset:6208 // (op 3857) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:97], v[32:37], a[36:39], v29, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1452) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:97], v[62:67], a[48:51], v29, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1533) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:97], v[78:83], a[56:59], v29, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1587) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:97], v[84:89], a[60:63], v29, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1614) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:97], v[68:73], a[52:55], v29, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 1560) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[44:51], v[18:23], a[64:67], v30, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2643) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[44:51], v[38:43], a[72:75], v30, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2697) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[44:51], v[56:61], a[76:79], v30, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2724) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[44:51], v[32:37], a[68:71], v30, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2670) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[44:51], v[62:67], a[80:83], v30, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2751) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[44:51], v[78:83], a[88:91], v30, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2805) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[44:51], v[84:89], a[92:95], v30, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2832) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[44:51], v[68:73], a[84:87], v30, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 2778) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[98:105], v[18:23], a[96:99], v31, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3861) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[98:105], v[38:43], a[104:107], v31, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3915) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[98:105], v[56:61], a[108:111], v31, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3942) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[98:105], v[32:37], a[100:103], v31, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3888) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[98:105], v[62:67], a[112:115], v31, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3969) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[98:105], v[78:83], a[120:123], v31, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4023) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[98:105], v[84:89], a[124:127], v31, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 4050) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[98:105], v[68:73], a[116:119], v31, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b010 // (op 3996) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 4013) 
s_mov_b32 s10, s6 // (op 4013) convert

s_cmp_lt_u32 s1, s10 // (op 4013) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoopTail_4013 // (op 4013) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoopTail_4013 if true)
RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoopTail_4013:





















s_barrier  // (op 14323) 





v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalFalse_0_beta_116
v_accvgpr_read v10, a0 // (op 42) 
v_accvgpr_read v11, a1 // (op 42) 
v_accvgpr_read v12, a2 // (op 42) 
v_accvgpr_read v13, a3 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 
v_mul_f32 v16, v7, v12 // (op 42) 
v_mul_f32 v17, v7, v13 // (op 42) 

v_lshrrev_b32 v10, 16, v14 // (op 48) 
v_lshrrev_b32 v11, 16, v15 // (op 48) 
v_lshrrev_b32 v12, 16, v16 // (op 48) 
v_lshrrev_b32 v13, 16, v17 // (op 48) 


v_pack_b32_f16 v4, v10, v11 // (op 50) 
v_pack_b32_f16 v5, v12, v13 // (op 50) 

v_mov_b32 v3, v9 // (op 50) reg expression

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 50) Store value



v_accvgpr_read v10, a4 // (op 5650) 
v_accvgpr_read v11, a5 // (op 5650) 
v_accvgpr_read v12, a6 // (op 5650) 
v_accvgpr_read v13, a7 // (op 5650) 
v_mul_f32 v14, v7, v10 // (op 5650) 
v_mul_f32 v15, v7, v11 // (op 5650) 
v_mul_f32 v16, v7, v12 // (op 5650) 
v_mul_f32 v17, v7, v13 // (op 5650) 

v_lshrrev_b32 v10, 16, v14 // (op 5652) 
v_lshrrev_b32 v11, 16, v15 // (op 5652) 
v_lshrrev_b32 v12, 16, v16 // (op 5652) 
v_lshrrev_b32 v13, 16, v17 // (op 5652) 


v_pack_b32_f16 v4, v10, v11 // (op 5653) 
v_pack_b32_f16 v5, v12, v13 // (op 5653) 

s_lshl_b32 s0, s4, 5 // (op 5653) 
v_add_u32 v3, v9, s0 // (op 5653) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5653) Store value



v_accvgpr_read v10, a8 // (op 5664) 
v_accvgpr_read v11, a9 // (op 5664) 
v_accvgpr_read v12, a10 // (op 5664) 
v_accvgpr_read v13, a11 // (op 5664) 
v_mul_f32 v14, v7, v10 // (op 5664) 
v_mul_f32 v15, v7, v11 // (op 5664) 
v_mul_f32 v16, v7, v12 // (op 5664) 
v_mul_f32 v17, v7, v13 // (op 5664) 

v_lshrrev_b32 v10, 16, v14 // (op 5666) 
v_lshrrev_b32 v11, 16, v15 // (op 5666) 
v_lshrrev_b32 v12, 16, v16 // (op 5666) 
v_lshrrev_b32 v13, 16, v17 // (op 5666) 


v_pack_b32_f16 v4, v10, v11 // (op 5667) 
v_pack_b32_f16 v5, v12, v13 // (op 5667) 

s_lshl_b32 s0, s4, 6 // (op 5667) 
v_add_u32 v3, v9, s0 // (op 5667) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5667) Store value



v_accvgpr_read v10, a12 // (op 5678) 
v_accvgpr_read v11, a13 // (op 5678) 
v_accvgpr_read v12, a14 // (op 5678) 
v_accvgpr_read v13, a15 // (op 5678) 
v_mul_f32 v14, v7, v10 // (op 5678) 
v_mul_f32 v15, v7, v11 // (op 5678) 
v_mul_f32 v16, v7, v12 // (op 5678) 
v_mul_f32 v17, v7, v13 // (op 5678) 

v_lshrrev_b32 v10, 16, v14 // (op 5680) 
v_lshrrev_b32 v11, 16, v15 // (op 5680) 
v_lshrrev_b32 v12, 16, v16 // (op 5680) 
v_lshrrev_b32 v13, 16, v17 // (op 5680) 


v_pack_b32_f16 v4, v10, v11 // (op 5681) 
v_pack_b32_f16 v5, v12, v13 // (op 5681) 

s_lshl_b32 s0, s4, 5 // (op 5681) 
s_mul_i32 s1, 3, s0 // (op 5681) 

v_add_u32 v3, v9, s1 // (op 5681) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5681) Store value



v_accvgpr_read v10, a16 // (op 5692) 
v_accvgpr_read v11, a17 // (op 5692) 
v_accvgpr_read v12, a18 // (op 5692) 
v_accvgpr_read v13, a19 // (op 5692) 
v_mul_f32 v14, v7, v10 // (op 5692) 
v_mul_f32 v15, v7, v11 // (op 5692) 
v_mul_f32 v16, v7, v12 // (op 5692) 
v_mul_f32 v17, v7, v13 // (op 5692) 

v_lshrrev_b32 v10, 16, v14 // (op 5694) 
v_lshrrev_b32 v11, 16, v15 // (op 5694) 
v_lshrrev_b32 v12, 16, v16 // (op 5694) 
v_lshrrev_b32 v13, 16, v17 // (op 5694) 


v_pack_b32_f16 v4, v10, v11 // (op 5695) 
v_pack_b32_f16 v5, v12, v13 // (op 5695) 

s_lshl_b32 s0, s4, 7 // (op 5695) 
v_add_u32 v3, v9, s0 // (op 5695) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5695) Store value



v_accvgpr_read v10, a20 // (op 5706) 
v_accvgpr_read v11, a21 // (op 5706) 
v_accvgpr_read v12, a22 // (op 5706) 
v_accvgpr_read v13, a23 // (op 5706) 
v_mul_f32 v14, v7, v10 // (op 5706) 
v_mul_f32 v15, v7, v11 // (op 5706) 
v_mul_f32 v16, v7, v12 // (op 5706) 
v_mul_f32 v17, v7, v13 // (op 5706) 

v_lshrrev_b32 v10, 16, v14 // (op 5708) 
v_lshrrev_b32 v11, 16, v15 // (op 5708) 
v_lshrrev_b32 v12, 16, v16 // (op 5708) 
v_lshrrev_b32 v13, 16, v17 // (op 5708) 


v_pack_b32_f16 v4, v10, v11 // (op 5709) 
v_pack_b32_f16 v5, v12, v13 // (op 5709) 

s_lshl_b32 s0, s4, 5 // (op 5709) 
s_mul_i32 s1, 5, s0 // (op 5709) 

v_add_u32 v3, v9, s1 // (op 5709) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5709) Store value



v_accvgpr_read v10, a24 // (op 5720) 
v_accvgpr_read v11, a25 // (op 5720) 
v_accvgpr_read v12, a26 // (op 5720) 
v_accvgpr_read v13, a27 // (op 5720) 
v_mul_f32 v14, v7, v10 // (op 5720) 
v_mul_f32 v15, v7, v11 // (op 5720) 
v_mul_f32 v16, v7, v12 // (op 5720) 
v_mul_f32 v17, v7, v13 // (op 5720) 

v_lshrrev_b32 v10, 16, v14 // (op 5722) 
v_lshrrev_b32 v11, 16, v15 // (op 5722) 
v_lshrrev_b32 v12, 16, v16 // (op 5722) 
v_lshrrev_b32 v13, 16, v17 // (op 5722) 


v_pack_b32_f16 v4, v10, v11 // (op 5723) 
v_pack_b32_f16 v5, v12, v13 // (op 5723) 

s_lshl_b32 s0, s4, 5 // (op 5723) 
s_mul_i32 s1, 6, s0 // (op 5723) 

v_add_u32 v3, v9, s1 // (op 5723) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5723) Store value



v_accvgpr_read v10, a28 // (op 5734) 
v_accvgpr_read v11, a29 // (op 5734) 
v_accvgpr_read v12, a30 // (op 5734) 
v_accvgpr_read v13, a31 // (op 5734) 
v_mul_f32 v14, v7, v10 // (op 5734) 
v_mul_f32 v15, v7, v11 // (op 5734) 
v_mul_f32 v16, v7, v12 // (op 5734) 
v_mul_f32 v17, v7, v13 // (op 5734) 

v_lshrrev_b32 v10, 16, v14 // (op 5736) 
v_lshrrev_b32 v11, 16, v15 // (op 5736) 
v_lshrrev_b32 v12, 16, v16 // (op 5736) 
v_lshrrev_b32 v13, 16, v17 // (op 5736) 


v_pack_b32_f16 v4, v10, v11 // (op 5737) 
v_pack_b32_f16 v5, v12, v13 // (op 5737) 

s_lshl_b32 s0, s4, 5 // (op 5737) 
s_mul_i32 s1, 7, s0 // (op 5737) 

v_add_u32 v3, v9, s1 // (op 5737) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5737) Store value



v_accvgpr_read v10, a32 // (op 5790) 
v_accvgpr_read v11, a33 // (op 5790) 
v_accvgpr_read v12, a34 // (op 5790) 
v_accvgpr_read v13, a35 // (op 5790) 
v_mul_f32 v14, v7, v10 // (op 5790) 
v_mul_f32 v15, v7, v11 // (op 5790) 
v_mul_f32 v16, v7, v12 // (op 5790) 
v_mul_f32 v17, v7, v13 // (op 5790) 

v_lshrrev_b32 v10, 16, v14 // (op 5792) 
v_lshrrev_b32 v11, 16, v15 // (op 5792) 
v_lshrrev_b32 v12, 16, v16 // (op 5792) 
v_lshrrev_b32 v13, 16, v17 // (op 5792) 


v_pack_b32_f16 v4, v10, v11 // (op 5794) 
v_pack_b32_f16 v5, v12, v13 // (op 5794) 

v_add_u32 v3, 32, v9 // (op 5794) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5794) Store value



v_accvgpr_read v10, a36 // (op 5812) 
v_accvgpr_read v11, a37 // (op 5812) 
v_accvgpr_read v12, a38 // (op 5812) 
v_accvgpr_read v13, a39 // (op 5812) 
v_mul_f32 v14, v7, v10 // (op 5812) 
v_mul_f32 v15, v7, v11 // (op 5812) 
v_mul_f32 v16, v7, v12 // (op 5812) 
v_mul_f32 v17, v7, v13 // (op 5812) 

v_lshrrev_b32 v10, 16, v14 // (op 5814) 
v_lshrrev_b32 v11, 16, v15 // (op 5814) 
v_lshrrev_b32 v12, 16, v16 // (op 5814) 
v_lshrrev_b32 v13, 16, v17 // (op 5814) 


v_pack_b32_f16 v4, v10, v11 // (op 5796) 
v_pack_b32_f16 v5, v12, v13 // (op 5796) 

s_lshl_b32 s0, s4, 5 // (op 5796) 
s_add_u32 s0, s0, 32 // (op 5796) 
v_add_u32 v3, v9, s0 // (op 5796) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5796) Store value



v_accvgpr_read v10, a40 // (op 5818) 
v_accvgpr_read v11, a41 // (op 5818) 
v_accvgpr_read v12, a42 // (op 5818) 
v_accvgpr_read v13, a43 // (op 5818) 
v_mul_f32 v14, v7, v10 // (op 5818) 
v_mul_f32 v15, v7, v11 // (op 5818) 
v_mul_f32 v16, v7, v12 // (op 5818) 
v_mul_f32 v17, v7, v13 // (op 5818) 

v_lshrrev_b32 v10, 16, v14 // (op 5820) 
v_lshrrev_b32 v11, 16, v15 // (op 5820) 
v_lshrrev_b32 v12, 16, v16 // (op 5820) 
v_lshrrev_b32 v13, 16, v17 // (op 5820) 


v_pack_b32_f16 v4, v10, v11 // (op 5798) 
v_pack_b32_f16 v5, v12, v13 // (op 5798) 

s_lshl_b32 s0, s4, 6 // (op 5798) 
s_add_u32 s0, s0, 32 // (op 5798) 
v_add_u32 v3, v9, s0 // (op 5798) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5798) Store value



v_accvgpr_read v10, a44 // (op 5824) 
v_accvgpr_read v11, a45 // (op 5824) 
v_accvgpr_read v12, a46 // (op 5824) 
v_accvgpr_read v13, a47 // (op 5824) 
v_mul_f32 v14, v7, v10 // (op 5824) 
v_mul_f32 v15, v7, v11 // (op 5824) 
v_mul_f32 v16, v7, v12 // (op 5824) 
v_mul_f32 v17, v7, v13 // (op 5824) 

v_lshrrev_b32 v10, 16, v14 // (op 5826) 
v_lshrrev_b32 v11, 16, v15 // (op 5826) 
v_lshrrev_b32 v12, 16, v16 // (op 5826) 
v_lshrrev_b32 v13, 16, v17 // (op 5826) 


v_pack_b32_f16 v4, v10, v11 // (op 5800) 
v_pack_b32_f16 v5, v12, v13 // (op 5800) 

s_lshl_b32 s0, s4, 5 // (op 5800) 
s_mul_i32 s1, 3, s0 // (op 5800) 
s_add_u32 s1, s1, 32 // (op 5800) 

v_add_u32 v3, v9, s1 // (op 5800) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5800) Store value



v_accvgpr_read v10, a48 // (op 5830) 
v_accvgpr_read v11, a49 // (op 5830) 
v_accvgpr_read v12, a50 // (op 5830) 
v_accvgpr_read v13, a51 // (op 5830) 
v_mul_f32 v14, v7, v10 // (op 5830) 
v_mul_f32 v15, v7, v11 // (op 5830) 
v_mul_f32 v16, v7, v12 // (op 5830) 
v_mul_f32 v17, v7, v13 // (op 5830) 

v_lshrrev_b32 v10, 16, v14 // (op 5832) 
v_lshrrev_b32 v11, 16, v15 // (op 5832) 
v_lshrrev_b32 v12, 16, v16 // (op 5832) 
v_lshrrev_b32 v13, 16, v17 // (op 5832) 


v_pack_b32_f16 v4, v10, v11 // (op 5802) 
v_pack_b32_f16 v5, v12, v13 // (op 5802) 

s_lshl_b32 s0, s4, 7 // (op 5802) 
s_add_u32 s0, s0, 32 // (op 5802) 
v_add_u32 v3, v9, s0 // (op 5802) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5802) Store value



v_accvgpr_read v10, a52 // (op 5836) 
v_accvgpr_read v11, a53 // (op 5836) 
v_accvgpr_read v12, a54 // (op 5836) 
v_accvgpr_read v13, a55 // (op 5836) 
v_mul_f32 v14, v7, v10 // (op 5836) 
v_mul_f32 v15, v7, v11 // (op 5836) 
v_mul_f32 v16, v7, v12 // (op 5836) 
v_mul_f32 v17, v7, v13 // (op 5836) 

v_lshrrev_b32 v10, 16, v14 // (op 5838) 
v_lshrrev_b32 v11, 16, v15 // (op 5838) 
v_lshrrev_b32 v12, 16, v16 // (op 5838) 
v_lshrrev_b32 v13, 16, v17 // (op 5838) 


v_pack_b32_f16 v4, v10, v11 // (op 5804) 
v_pack_b32_f16 v5, v12, v13 // (op 5804) 

s_lshl_b32 s0, s4, 5 // (op 5804) 
s_mul_i32 s1, 5, s0 // (op 5804) 
s_add_u32 s1, s1, 32 // (op 5804) 

v_add_u32 v3, v9, s1 // (op 5804) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5804) Store value



v_accvgpr_read v10, a56 // (op 5842) 
v_accvgpr_read v11, a57 // (op 5842) 
v_accvgpr_read v12, a58 // (op 5842) 
v_accvgpr_read v13, a59 // (op 5842) 
v_mul_f32 v14, v7, v10 // (op 5842) 
v_mul_f32 v15, v7, v11 // (op 5842) 
v_mul_f32 v16, v7, v12 // (op 5842) 
v_mul_f32 v17, v7, v13 // (op 5842) 

v_lshrrev_b32 v10, 16, v14 // (op 5844) 
v_lshrrev_b32 v11, 16, v15 // (op 5844) 
v_lshrrev_b32 v12, 16, v16 // (op 5844) 
v_lshrrev_b32 v13, 16, v17 // (op 5844) 


v_pack_b32_f16 v4, v10, v11 // (op 5806) 
v_pack_b32_f16 v5, v12, v13 // (op 5806) 

s_lshl_b32 s0, s4, 5 // (op 5806) 
s_mul_i32 s1, 6, s0 // (op 5806) 
s_add_u32 s1, s1, 32 // (op 5806) 

v_add_u32 v3, v9, s1 // (op 5806) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5806) Store value



v_accvgpr_read v10, a60 // (op 5848) 
v_accvgpr_read v11, a61 // (op 5848) 
v_accvgpr_read v12, a62 // (op 5848) 
v_accvgpr_read v13, a63 // (op 5848) 
v_mul_f32 v14, v7, v10 // (op 5848) 
v_mul_f32 v15, v7, v11 // (op 5848) 
v_mul_f32 v16, v7, v12 // (op 5848) 
v_mul_f32 v17, v7, v13 // (op 5848) 

v_lshrrev_b32 v10, 16, v14 // (op 5850) 
v_lshrrev_b32 v11, 16, v15 // (op 5850) 
v_lshrrev_b32 v12, 16, v16 // (op 5850) 
v_lshrrev_b32 v13, 16, v17 // (op 5850) 


v_pack_b32_f16 v4, v10, v11 // (op 5808) 
v_pack_b32_f16 v5, v12, v13 // (op 5808) 

s_lshl_b32 s0, s4, 5 // (op 5808) 
s_mul_i32 s1, 7, s0 // (op 5808) 
s_add_u32 s1, s1, 32 // (op 5808) 

v_add_u32 v3, v9, s1 // (op 5808) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5808) Store value



v_accvgpr_read v10, a64 // (op 5938) 
v_accvgpr_read v11, a65 // (op 5938) 
v_accvgpr_read v12, a66 // (op 5938) 
v_accvgpr_read v13, a67 // (op 5938) 
v_mul_f32 v14, v7, v10 // (op 5938) 
v_mul_f32 v15, v7, v11 // (op 5938) 
v_mul_f32 v16, v7, v12 // (op 5938) 
v_mul_f32 v17, v7, v13 // (op 5938) 

v_lshrrev_b32 v10, 16, v14 // (op 5940) 
v_lshrrev_b32 v11, 16, v15 // (op 5940) 
v_lshrrev_b32 v12, 16, v16 // (op 5940) 
v_lshrrev_b32 v13, 16, v17 // (op 5940) 


v_pack_b32_f16 v4, v10, v11 // (op 5942) 
v_pack_b32_f16 v5, v12, v13 // (op 5942) 

v_add_u32 v3, 64, v9 // (op 5942) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5942) Store value



v_accvgpr_read v10, a68 // (op 5960) 
v_accvgpr_read v11, a69 // (op 5960) 
v_accvgpr_read v12, a70 // (op 5960) 
v_accvgpr_read v13, a71 // (op 5960) 
v_mul_f32 v14, v7, v10 // (op 5960) 
v_mul_f32 v15, v7, v11 // (op 5960) 
v_mul_f32 v16, v7, v12 // (op 5960) 
v_mul_f32 v17, v7, v13 // (op 5960) 

v_lshrrev_b32 v10, 16, v14 // (op 5962) 
v_lshrrev_b32 v11, 16, v15 // (op 5962) 
v_lshrrev_b32 v12, 16, v16 // (op 5962) 
v_lshrrev_b32 v13, 16, v17 // (op 5962) 


v_pack_b32_f16 v4, v10, v11 // (op 5944) 
v_pack_b32_f16 v5, v12, v13 // (op 5944) 

s_lshl_b32 s0, s4, 5 // (op 5944) 
s_add_u32 s0, s0, 64 // (op 5944) 
v_add_u32 v3, v9, s0 // (op 5944) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5944) Store value



v_accvgpr_read v10, a72 // (op 5966) 
v_accvgpr_read v11, a73 // (op 5966) 
v_accvgpr_read v12, a74 // (op 5966) 
v_accvgpr_read v13, a75 // (op 5966) 
v_mul_f32 v14, v7, v10 // (op 5966) 
v_mul_f32 v15, v7, v11 // (op 5966) 
v_mul_f32 v16, v7, v12 // (op 5966) 
v_mul_f32 v17, v7, v13 // (op 5966) 

v_lshrrev_b32 v10, 16, v14 // (op 5968) 
v_lshrrev_b32 v11, 16, v15 // (op 5968) 
v_lshrrev_b32 v12, 16, v16 // (op 5968) 
v_lshrrev_b32 v13, 16, v17 // (op 5968) 


v_pack_b32_f16 v4, v10, v11 // (op 5946) 
v_pack_b32_f16 v5, v12, v13 // (op 5946) 

s_lshl_b32 s0, s4, 6 // (op 5946) 
s_add_u32 s0, s0, 64 // (op 5946) 
v_add_u32 v3, v9, s0 // (op 5946) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5946) Store value



v_accvgpr_read v10, a76 // (op 5972) 
v_accvgpr_read v11, a77 // (op 5972) 
v_accvgpr_read v12, a78 // (op 5972) 
v_accvgpr_read v13, a79 // (op 5972) 
v_mul_f32 v14, v7, v10 // (op 5972) 
v_mul_f32 v15, v7, v11 // (op 5972) 
v_mul_f32 v16, v7, v12 // (op 5972) 
v_mul_f32 v17, v7, v13 // (op 5972) 

v_lshrrev_b32 v10, 16, v14 // (op 5974) 
v_lshrrev_b32 v11, 16, v15 // (op 5974) 
v_lshrrev_b32 v12, 16, v16 // (op 5974) 
v_lshrrev_b32 v13, 16, v17 // (op 5974) 


v_pack_b32_f16 v4, v10, v11 // (op 5948) 
v_pack_b32_f16 v5, v12, v13 // (op 5948) 

s_lshl_b32 s0, s4, 5 // (op 5948) 
s_mul_i32 s1, 3, s0 // (op 5948) 
s_add_u32 s1, s1, 64 // (op 5948) 

v_add_u32 v3, v9, s1 // (op 5948) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5948) Store value



v_accvgpr_read v10, a80 // (op 5978) 
v_accvgpr_read v11, a81 // (op 5978) 
v_accvgpr_read v12, a82 // (op 5978) 
v_accvgpr_read v13, a83 // (op 5978) 
v_mul_f32 v14, v7, v10 // (op 5978) 
v_mul_f32 v15, v7, v11 // (op 5978) 
v_mul_f32 v16, v7, v12 // (op 5978) 
v_mul_f32 v17, v7, v13 // (op 5978) 

v_lshrrev_b32 v10, 16, v14 // (op 5980) 
v_lshrrev_b32 v11, 16, v15 // (op 5980) 
v_lshrrev_b32 v12, 16, v16 // (op 5980) 
v_lshrrev_b32 v13, 16, v17 // (op 5980) 


v_pack_b32_f16 v4, v10, v11 // (op 5950) 
v_pack_b32_f16 v5, v12, v13 // (op 5950) 

s_lshl_b32 s0, s4, 7 // (op 5950) 
s_add_u32 s0, s0, 64 // (op 5950) 
v_add_u32 v3, v9, s0 // (op 5950) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5950) Store value



v_accvgpr_read v10, a84 // (op 5984) 
v_accvgpr_read v11, a85 // (op 5984) 
v_accvgpr_read v12, a86 // (op 5984) 
v_accvgpr_read v13, a87 // (op 5984) 
v_mul_f32 v14, v7, v10 // (op 5984) 
v_mul_f32 v15, v7, v11 // (op 5984) 
v_mul_f32 v16, v7, v12 // (op 5984) 
v_mul_f32 v17, v7, v13 // (op 5984) 

v_lshrrev_b32 v10, 16, v14 // (op 5986) 
v_lshrrev_b32 v11, 16, v15 // (op 5986) 
v_lshrrev_b32 v12, 16, v16 // (op 5986) 
v_lshrrev_b32 v13, 16, v17 // (op 5986) 


v_pack_b32_f16 v4, v10, v11 // (op 5952) 
v_pack_b32_f16 v5, v12, v13 // (op 5952) 

s_lshl_b32 s0, s4, 5 // (op 5952) 
s_mul_i32 s1, 5, s0 // (op 5952) 
s_add_u32 s1, s1, 64 // (op 5952) 

v_add_u32 v3, v9, s1 // (op 5952) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5952) Store value



v_accvgpr_read v10, a88 // (op 5990) 
v_accvgpr_read v11, a89 // (op 5990) 
v_accvgpr_read v12, a90 // (op 5990) 
v_accvgpr_read v13, a91 // (op 5990) 
v_mul_f32 v14, v7, v10 // (op 5990) 
v_mul_f32 v15, v7, v11 // (op 5990) 
v_mul_f32 v16, v7, v12 // (op 5990) 
v_mul_f32 v17, v7, v13 // (op 5990) 

v_lshrrev_b32 v10, 16, v14 // (op 5992) 
v_lshrrev_b32 v11, 16, v15 // (op 5992) 
v_lshrrev_b32 v12, 16, v16 // (op 5992) 
v_lshrrev_b32 v13, 16, v17 // (op 5992) 


v_pack_b32_f16 v4, v10, v11 // (op 5954) 
v_pack_b32_f16 v5, v12, v13 // (op 5954) 

s_lshl_b32 s0, s4, 5 // (op 5954) 
s_mul_i32 s1, 6, s0 // (op 5954) 
s_add_u32 s1, s1, 64 // (op 5954) 

v_add_u32 v3, v9, s1 // (op 5954) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5954) Store value



v_accvgpr_read v10, a92 // (op 5996) 
v_accvgpr_read v11, a93 // (op 5996) 
v_accvgpr_read v12, a94 // (op 5996) 
v_accvgpr_read v13, a95 // (op 5996) 
v_mul_f32 v14, v7, v10 // (op 5996) 
v_mul_f32 v15, v7, v11 // (op 5996) 
v_mul_f32 v16, v7, v12 // (op 5996) 
v_mul_f32 v17, v7, v13 // (op 5996) 

v_lshrrev_b32 v10, 16, v14 // (op 5998) 
v_lshrrev_b32 v11, 16, v15 // (op 5998) 
v_lshrrev_b32 v12, 16, v16 // (op 5998) 
v_lshrrev_b32 v13, 16, v17 // (op 5998) 


v_pack_b32_f16 v4, v10, v11 // (op 5956) 
v_pack_b32_f16 v5, v12, v13 // (op 5956) 

s_lshl_b32 s0, s4, 5 // (op 5956) 
s_mul_i32 s1, 7, s0 // (op 5956) 
s_add_u32 s1, s1, 64 // (op 5956) 

v_add_u32 v3, v9, s1 // (op 5956) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 5956) Store value



v_accvgpr_read v10, a96 // (op 6086) 
v_accvgpr_read v11, a97 // (op 6086) 
v_accvgpr_read v12, a98 // (op 6086) 
v_accvgpr_read v13, a99 // (op 6086) 
v_mul_f32 v14, v7, v10 // (op 6086) 
v_mul_f32 v15, v7, v11 // (op 6086) 
v_mul_f32 v16, v7, v12 // (op 6086) 
v_mul_f32 v17, v7, v13 // (op 6086) 

v_lshrrev_b32 v10, 16, v14 // (op 6088) 
v_lshrrev_b32 v11, 16, v15 // (op 6088) 
v_lshrrev_b32 v12, 16, v16 // (op 6088) 
v_lshrrev_b32 v13, 16, v17 // (op 6088) 


v_pack_b32_f16 v4, v10, v11 // (op 6090) 
v_pack_b32_f16 v5, v12, v13 // (op 6090) 

v_add_u32 v3, 96, v9 // (op 6090) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 6090) Store value



v_accvgpr_read v10, a100 // (op 6108) 
v_accvgpr_read v11, a101 // (op 6108) 
v_accvgpr_read v12, a102 // (op 6108) 
v_accvgpr_read v13, a103 // (op 6108) 
v_mul_f32 v14, v7, v10 // (op 6108) 
v_mul_f32 v15, v7, v11 // (op 6108) 
v_mul_f32 v16, v7, v12 // (op 6108) 
v_mul_f32 v17, v7, v13 // (op 6108) 

v_lshrrev_b32 v10, 16, v14 // (op 6110) 
v_lshrrev_b32 v11, 16, v15 // (op 6110) 
v_lshrrev_b32 v12, 16, v16 // (op 6110) 
v_lshrrev_b32 v13, 16, v17 // (op 6110) 


v_pack_b32_f16 v4, v10, v11 // (op 6092) 
v_pack_b32_f16 v5, v12, v13 // (op 6092) 

s_lshl_b32 s0, s4, 5 // (op 6092) 
s_add_u32 s0, s0, 96 // (op 6092) 
v_add_u32 v3, v9, s0 // (op 6092) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 6092) Store value



v_accvgpr_read v10, a104 // (op 6114) 
v_accvgpr_read v11, a105 // (op 6114) 
v_accvgpr_read v12, a106 // (op 6114) 
v_accvgpr_read v13, a107 // (op 6114) 
v_mul_f32 v14, v7, v10 // (op 6114) 
v_mul_f32 v15, v7, v11 // (op 6114) 
v_mul_f32 v16, v7, v12 // (op 6114) 
v_mul_f32 v17, v7, v13 // (op 6114) 

v_lshrrev_b32 v10, 16, v14 // (op 6116) 
v_lshrrev_b32 v11, 16, v15 // (op 6116) 
v_lshrrev_b32 v12, 16, v16 // (op 6116) 
v_lshrrev_b32 v13, 16, v17 // (op 6116) 


v_pack_b32_f16 v4, v10, v11 // (op 6094) 
v_pack_b32_f16 v5, v12, v13 // (op 6094) 

s_lshl_b32 s0, s4, 6 // (op 6094) 
s_add_u32 s0, s0, 96 // (op 6094) 
v_add_u32 v3, v9, s0 // (op 6094) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 6094) Store value



v_accvgpr_read v10, a108 // (op 6120) 
v_accvgpr_read v11, a109 // (op 6120) 
v_accvgpr_read v12, a110 // (op 6120) 
v_accvgpr_read v13, a111 // (op 6120) 
v_mul_f32 v14, v7, v10 // (op 6120) 
v_mul_f32 v15, v7, v11 // (op 6120) 
v_mul_f32 v16, v7, v12 // (op 6120) 
v_mul_f32 v17, v7, v13 // (op 6120) 

v_lshrrev_b32 v10, 16, v14 // (op 6122) 
v_lshrrev_b32 v11, 16, v15 // (op 6122) 
v_lshrrev_b32 v12, 16, v16 // (op 6122) 
v_lshrrev_b32 v13, 16, v17 // (op 6122) 


v_pack_b32_f16 v4, v10, v11 // (op 6096) 
v_pack_b32_f16 v5, v12, v13 // (op 6096) 

s_lshl_b32 s0, s4, 5 // (op 6096) 
s_mul_i32 s1, 3, s0 // (op 6096) 
s_add_u32 s1, s1, 96 // (op 6096) 

v_add_u32 v3, v9, s1 // (op 6096) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 6096) Store value



v_accvgpr_read v10, a112 // (op 6126) 
v_accvgpr_read v11, a113 // (op 6126) 
v_accvgpr_read v12, a114 // (op 6126) 
v_accvgpr_read v13, a115 // (op 6126) 
v_mul_f32 v14, v7, v10 // (op 6126) 
v_mul_f32 v15, v7, v11 // (op 6126) 
v_mul_f32 v16, v7, v12 // (op 6126) 
v_mul_f32 v17, v7, v13 // (op 6126) 

v_lshrrev_b32 v10, 16, v14 // (op 6128) 
v_lshrrev_b32 v11, 16, v15 // (op 6128) 
v_lshrrev_b32 v12, 16, v16 // (op 6128) 
v_lshrrev_b32 v13, 16, v17 // (op 6128) 


v_pack_b32_f16 v4, v10, v11 // (op 6098) 
v_pack_b32_f16 v5, v12, v13 // (op 6098) 

s_lshl_b32 s0, s4, 7 // (op 6098) 
s_add_u32 s0, s0, 96 // (op 6098) 
v_add_u32 v3, v9, s0 // (op 6098) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 6098) Store value



v_accvgpr_read v10, a116 // (op 6132) 
v_accvgpr_read v11, a117 // (op 6132) 
v_accvgpr_read v12, a118 // (op 6132) 
v_accvgpr_read v13, a119 // (op 6132) 
v_mul_f32 v14, v7, v10 // (op 6132) 
v_mul_f32 v15, v7, v11 // (op 6132) 
v_mul_f32 v16, v7, v12 // (op 6132) 
v_mul_f32 v17, v7, v13 // (op 6132) 

v_lshrrev_b32 v10, 16, v14 // (op 6134) 
v_lshrrev_b32 v11, 16, v15 // (op 6134) 
v_lshrrev_b32 v12, 16, v16 // (op 6134) 
v_lshrrev_b32 v13, 16, v17 // (op 6134) 


v_pack_b32_f16 v4, v10, v11 // (op 6100) 
v_pack_b32_f16 v5, v12, v13 // (op 6100) 

s_lshl_b32 s0, s4, 5 // (op 6100) 
s_mul_i32 s1, 5, s0 // (op 6100) 
s_add_u32 s1, s1, 96 // (op 6100) 

v_add_u32 v3, v9, s1 // (op 6100) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 6100) Store value



v_accvgpr_read v10, a120 // (op 6138) 
v_accvgpr_read v11, a121 // (op 6138) 
v_accvgpr_read v12, a122 // (op 6138) 
v_accvgpr_read v13, a123 // (op 6138) 
v_mul_f32 v14, v7, v10 // (op 6138) 
v_mul_f32 v15, v7, v11 // (op 6138) 
v_mul_f32 v16, v7, v12 // (op 6138) 
v_mul_f32 v17, v7, v13 // (op 6138) 

v_lshrrev_b32 v10, 16, v14 // (op 6140) 
v_lshrrev_b32 v11, 16, v15 // (op 6140) 
v_lshrrev_b32 v12, 16, v16 // (op 6140) 
v_lshrrev_b32 v13, 16, v17 // (op 6140) 


v_pack_b32_f16 v4, v10, v11 // (op 6102) 
v_pack_b32_f16 v5, v12, v13 // (op 6102) 

s_lshl_b32 s0, s4, 5 // (op 6102) 
s_mul_i32 s1, 6, s0 // (op 6102) 
s_add_u32 s1, s1, 96 // (op 6102) 

v_add_u32 v3, v9, s1 // (op 6102) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 6102) Store value



v_accvgpr_read v10, a124 // (op 6144) 
v_accvgpr_read v11, a125 // (op 6144) 
v_accvgpr_read v12, a126 // (op 6144) 
v_accvgpr_read v13, a127 // (op 6144) 
v_mul_f32 v14, v7, v10 // (op 6144) 
v_mul_f32 v15, v7, v11 // (op 6144) 
v_mul_f32 v16, v7, v12 // (op 6144) 
v_mul_f32 v17, v7, v13 // (op 6144) 

v_lshrrev_b32 v10, 16, v14 // (op 6146) 
v_lshrrev_b32 v11, 16, v15 // (op 6146) 
v_lshrrev_b32 v12, 16, v16 // (op 6146) 
v_lshrrev_b32 v13, 16, v17 // (op 6146) 


v_pack_b32_f16 v4, v10, v11 // (op 6104) 
v_pack_b32_f16 v5, v12, v13 // (op 6104) 

s_lshl_b32 s0, s4, 5 // (op 6104) 
s_mul_i32 s1, 7, s0 // (op 6104) 
s_add_u32 s1, s1, 96 // (op 6104) 

v_add_u32 v3, v9, s1 // (op 6104) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 6104) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalFalse_0_beta_116:


v_add_u32 v3, 96, v8 // (op 6876) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6876) Load value

v_accvgpr_read v10, a96 // (op 6855) 
v_accvgpr_read v11, a97 // (op 6855) 
v_accvgpr_read v12, a98 // (op 6855) 
v_accvgpr_read v13, a99 // (op 6855) 
v_mul_f32 v14, v7, v10 // (op 6855) 
v_mul_f32 v15, v7, v11 // (op 6855) 
v_mul_f32 v16, v7, v12 // (op 6855) 
v_mul_f32 v17, v7, v13 // (op 6855) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6875) 
v_lshrrev_b32 v11, 16, v4 // (op 6875) 
v_lshlrev_b32 v10, 16, v10 // (op 6875) 
v_lshlrev_b32 v11, 16, v11 // (op 6875) 
v_mul_f32 v18, v0, v10 // (op 6875) 
v_mul_f32 v19, v0, v11 // (op 6875) 
v_and_b32 v10, 65535, v5 // (op 6875) 
v_lshrrev_b32 v11, 16, v5 // (op 6875) 
v_lshlrev_b32 v10, 16, v10 // (op 6875) 
v_lshlrev_b32 v11, 16, v11 // (op 6875) 
v_mul_f32 v20, v0, v10 // (op 6875) 
v_mul_f32 v21, v0, v11 // (op 6875) 


v_add_f32 v10, v18, v14 // (op 6856) 
v_add_f32 v11, v19, v15 // (op 6856) 
v_add_f32 v12, v20, v16 // (op 6856) 
v_add_f32 v13, v21, v17 // (op 6856) 


v_lshrrev_b32 v14, 16, v10 // (op 6857) 
v_lshrrev_b32 v15, 16, v11 // (op 6857) 
v_lshrrev_b32 v16, 16, v12 // (op 6857) 
v_lshrrev_b32 v17, 16, v13 // (op 6857) 


v_pack_b32_f16 v10, v14, v15 // (op 6859) 
v_pack_b32_f16 v11, v16, v17 // (op 6859) 

v_add_u32 v3, 96, v9 // (op 6859) 

buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6859) Store value





s_lshl_b32 s0, s2, 5 // (op 6881) 
s_add_u32 s0, s0, 96 // (op 6881) 
v_add_u32 v3, v8, s0 // (op 6881) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6881) Load value

v_accvgpr_read v10, a100 // (op 6913) 
v_accvgpr_read v11, a101 // (op 6913) 
v_accvgpr_read v12, a102 // (op 6913) 
v_accvgpr_read v13, a103 // (op 6913) 
v_mul_f32 v14, v7, v10 // (op 6913) 
v_mul_f32 v15, v7, v11 // (op 6913) 
v_mul_f32 v16, v7, v12 // (op 6913) 
v_mul_f32 v17, v7, v13 // (op 6913) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6878) 
v_lshrrev_b32 v11, 16, v4 // (op 6878) 
v_lshlrev_b32 v10, 16, v10 // (op 6878) 
v_lshlrev_b32 v11, 16, v11 // (op 6878) 
v_mul_f32 v18, v0, v10 // (op 6878) 
v_mul_f32 v19, v0, v11 // (op 6878) 
v_and_b32 v10, 65535, v5 // (op 6878) 
v_lshrrev_b32 v11, 16, v5 // (op 6878) 
v_lshlrev_b32 v10, 16, v10 // (op 6878) 
v_lshlrev_b32 v11, 16, v11 // (op 6878) 
v_mul_f32 v20, v0, v10 // (op 6878) 
v_mul_f32 v21, v0, v11 // (op 6878) 


v_add_f32 v10, v18, v14 // (op 6879) 
v_add_f32 v11, v19, v15 // (op 6879) 
v_add_f32 v12, v20, v16 // (op 6879) 
v_add_f32 v13, v21, v17 // (op 6879) 


v_lshrrev_b32 v14, 16, v10 // (op 6880) 
v_lshrrev_b32 v15, 16, v11 // (op 6880) 
v_lshrrev_b32 v16, 16, v12 // (op 6880) 
v_lshrrev_b32 v17, 16, v13 // (op 6880) 


v_pack_b32_f16 v10, v14, v15 // (op 6861) 
v_pack_b32_f16 v11, v16, v17 // (op 6861) 

s_lshl_b32 s0, s4, 5 // (op 6861) 
s_add_u32 s0, s0, 96 // (op 6861) 
v_add_u32 v3, v9, s0 // (op 6861) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6861) Store value





s_lshl_b32 s0, s2, 6 // (op 6886) 
s_add_u32 s0, s0, 96 // (op 6886) 
v_add_u32 v3, v8, s0 // (op 6886) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6886) Load value

v_accvgpr_read v10, a104 // (op 6915) 
v_accvgpr_read v11, a105 // (op 6915) 
v_accvgpr_read v12, a106 // (op 6915) 
v_accvgpr_read v13, a107 // (op 6915) 
v_mul_f32 v14, v7, v10 // (op 6915) 
v_mul_f32 v15, v7, v11 // (op 6915) 
v_mul_f32 v16, v7, v12 // (op 6915) 
v_mul_f32 v17, v7, v13 // (op 6915) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6883) 
v_lshrrev_b32 v11, 16, v4 // (op 6883) 
v_lshlrev_b32 v10, 16, v10 // (op 6883) 
v_lshlrev_b32 v11, 16, v11 // (op 6883) 
v_mul_f32 v18, v0, v10 // (op 6883) 
v_mul_f32 v19, v0, v11 // (op 6883) 
v_and_b32 v10, 65535, v5 // (op 6883) 
v_lshrrev_b32 v11, 16, v5 // (op 6883) 
v_lshlrev_b32 v10, 16, v10 // (op 6883) 
v_lshlrev_b32 v11, 16, v11 // (op 6883) 
v_mul_f32 v20, v0, v10 // (op 6883) 
v_mul_f32 v21, v0, v11 // (op 6883) 


v_add_f32 v10, v18, v14 // (op 6884) 
v_add_f32 v11, v19, v15 // (op 6884) 
v_add_f32 v12, v20, v16 // (op 6884) 
v_add_f32 v13, v21, v17 // (op 6884) 


v_lshrrev_b32 v14, 16, v10 // (op 6885) 
v_lshrrev_b32 v15, 16, v11 // (op 6885) 
v_lshrrev_b32 v16, 16, v12 // (op 6885) 
v_lshrrev_b32 v17, 16, v13 // (op 6885) 


v_pack_b32_f16 v10, v14, v15 // (op 6863) 
v_pack_b32_f16 v11, v16, v17 // (op 6863) 

s_lshl_b32 s0, s4, 6 // (op 6863) 
s_add_u32 s0, s0, 96 // (op 6863) 
v_add_u32 v3, v9, s0 // (op 6863) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6863) Store value





s_lshl_b32 s0, s2, 5 // (op 6891) 
s_mul_i32 s1, 3, s0 // (op 6891) 
s_add_u32 s1, s1, 96 // (op 6891) 

v_add_u32 v3, v8, s1 // (op 6891) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6891) Load value

v_accvgpr_read v10, a108 // (op 6917) 
v_accvgpr_read v11, a109 // (op 6917) 
v_accvgpr_read v12, a110 // (op 6917) 
v_accvgpr_read v13, a111 // (op 6917) 
v_mul_f32 v14, v7, v10 // (op 6917) 
v_mul_f32 v15, v7, v11 // (op 6917) 
v_mul_f32 v16, v7, v12 // (op 6917) 
v_mul_f32 v17, v7, v13 // (op 6917) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6888) 
v_lshrrev_b32 v11, 16, v4 // (op 6888) 
v_lshlrev_b32 v10, 16, v10 // (op 6888) 
v_lshlrev_b32 v11, 16, v11 // (op 6888) 
v_mul_f32 v18, v0, v10 // (op 6888) 
v_mul_f32 v19, v0, v11 // (op 6888) 
v_and_b32 v10, 65535, v5 // (op 6888) 
v_lshrrev_b32 v11, 16, v5 // (op 6888) 
v_lshlrev_b32 v10, 16, v10 // (op 6888) 
v_lshlrev_b32 v11, 16, v11 // (op 6888) 
v_mul_f32 v20, v0, v10 // (op 6888) 
v_mul_f32 v21, v0, v11 // (op 6888) 


v_add_f32 v10, v18, v14 // (op 6889) 
v_add_f32 v11, v19, v15 // (op 6889) 
v_add_f32 v12, v20, v16 // (op 6889) 
v_add_f32 v13, v21, v17 // (op 6889) 


v_lshrrev_b32 v14, 16, v10 // (op 6890) 
v_lshrrev_b32 v15, 16, v11 // (op 6890) 
v_lshrrev_b32 v16, 16, v12 // (op 6890) 
v_lshrrev_b32 v17, 16, v13 // (op 6890) 


v_pack_b32_f16 v10, v14, v15 // (op 6865) 
v_pack_b32_f16 v11, v16, v17 // (op 6865) 

s_lshl_b32 s0, s4, 5 // (op 6865) 
s_mul_i32 s1, 3, s0 // (op 6865) 
s_add_u32 s1, s1, 96 // (op 6865) 

v_add_u32 v3, v9, s1 // (op 6865) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6865) Store value





s_lshl_b32 s0, s2, 7 // (op 6896) 
s_add_u32 s0, s0, 96 // (op 6896) 
v_add_u32 v3, v8, s0 // (op 6896) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6896) Load value

v_accvgpr_read v10, a112 // (op 6919) 
v_accvgpr_read v11, a113 // (op 6919) 
v_accvgpr_read v12, a114 // (op 6919) 
v_accvgpr_read v13, a115 // (op 6919) 
v_mul_f32 v14, v7, v10 // (op 6919) 
v_mul_f32 v15, v7, v11 // (op 6919) 
v_mul_f32 v16, v7, v12 // (op 6919) 
v_mul_f32 v17, v7, v13 // (op 6919) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6893) 
v_lshrrev_b32 v11, 16, v4 // (op 6893) 
v_lshlrev_b32 v10, 16, v10 // (op 6893) 
v_lshlrev_b32 v11, 16, v11 // (op 6893) 
v_mul_f32 v18, v0, v10 // (op 6893) 
v_mul_f32 v19, v0, v11 // (op 6893) 
v_and_b32 v10, 65535, v5 // (op 6893) 
v_lshrrev_b32 v11, 16, v5 // (op 6893) 
v_lshlrev_b32 v10, 16, v10 // (op 6893) 
v_lshlrev_b32 v11, 16, v11 // (op 6893) 
v_mul_f32 v20, v0, v10 // (op 6893) 
v_mul_f32 v21, v0, v11 // (op 6893) 


v_add_f32 v10, v18, v14 // (op 6894) 
v_add_f32 v11, v19, v15 // (op 6894) 
v_add_f32 v12, v20, v16 // (op 6894) 
v_add_f32 v13, v21, v17 // (op 6894) 


v_lshrrev_b32 v14, 16, v10 // (op 6895) 
v_lshrrev_b32 v15, 16, v11 // (op 6895) 
v_lshrrev_b32 v16, 16, v12 // (op 6895) 
v_lshrrev_b32 v17, 16, v13 // (op 6895) 


v_pack_b32_f16 v10, v14, v15 // (op 6867) 
v_pack_b32_f16 v11, v16, v17 // (op 6867) 

s_lshl_b32 s0, s4, 7 // (op 6867) 
s_add_u32 s0, s0, 96 // (op 6867) 
v_add_u32 v3, v9, s0 // (op 6867) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6867) Store value





s_lshl_b32 s0, s2, 5 // (op 6901) 
s_mul_i32 s1, 5, s0 // (op 6901) 
s_add_u32 s1, s1, 96 // (op 6901) 

v_add_u32 v3, v8, s1 // (op 6901) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6901) Load value

v_accvgpr_read v10, a116 // (op 6921) 
v_accvgpr_read v11, a117 // (op 6921) 
v_accvgpr_read v12, a118 // (op 6921) 
v_accvgpr_read v13, a119 // (op 6921) 
v_mul_f32 v14, v7, v10 // (op 6921) 
v_mul_f32 v15, v7, v11 // (op 6921) 
v_mul_f32 v16, v7, v12 // (op 6921) 
v_mul_f32 v17, v7, v13 // (op 6921) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6898) 
v_lshrrev_b32 v11, 16, v4 // (op 6898) 
v_lshlrev_b32 v10, 16, v10 // (op 6898) 
v_lshlrev_b32 v11, 16, v11 // (op 6898) 
v_mul_f32 v18, v0, v10 // (op 6898) 
v_mul_f32 v19, v0, v11 // (op 6898) 
v_and_b32 v10, 65535, v5 // (op 6898) 
v_lshrrev_b32 v11, 16, v5 // (op 6898) 
v_lshlrev_b32 v10, 16, v10 // (op 6898) 
v_lshlrev_b32 v11, 16, v11 // (op 6898) 
v_mul_f32 v20, v0, v10 // (op 6898) 
v_mul_f32 v21, v0, v11 // (op 6898) 


v_add_f32 v10, v18, v14 // (op 6899) 
v_add_f32 v11, v19, v15 // (op 6899) 
v_add_f32 v12, v20, v16 // (op 6899) 
v_add_f32 v13, v21, v17 // (op 6899) 


v_lshrrev_b32 v14, 16, v10 // (op 6900) 
v_lshrrev_b32 v15, 16, v11 // (op 6900) 
v_lshrrev_b32 v16, 16, v12 // (op 6900) 
v_lshrrev_b32 v17, 16, v13 // (op 6900) 


v_pack_b32_f16 v10, v14, v15 // (op 6869) 
v_pack_b32_f16 v11, v16, v17 // (op 6869) 

s_lshl_b32 s0, s4, 5 // (op 6869) 
s_mul_i32 s1, 5, s0 // (op 6869) 
s_add_u32 s1, s1, 96 // (op 6869) 

v_add_u32 v3, v9, s1 // (op 6869) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6869) Store value





s_lshl_b32 s0, s2, 5 // (op 6906) 
s_mul_i32 s1, 6, s0 // (op 6906) 
s_add_u32 s1, s1, 96 // (op 6906) 

v_add_u32 v3, v8, s1 // (op 6906) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6906) Load value

v_accvgpr_read v10, a120 // (op 6923) 
v_accvgpr_read v11, a121 // (op 6923) 
v_accvgpr_read v12, a122 // (op 6923) 
v_accvgpr_read v13, a123 // (op 6923) 
v_mul_f32 v14, v7, v10 // (op 6923) 
v_mul_f32 v15, v7, v11 // (op 6923) 
v_mul_f32 v16, v7, v12 // (op 6923) 
v_mul_f32 v17, v7, v13 // (op 6923) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6903) 
v_lshrrev_b32 v11, 16, v4 // (op 6903) 
v_lshlrev_b32 v10, 16, v10 // (op 6903) 
v_lshlrev_b32 v11, 16, v11 // (op 6903) 
v_mul_f32 v18, v0, v10 // (op 6903) 
v_mul_f32 v19, v0, v11 // (op 6903) 
v_and_b32 v10, 65535, v5 // (op 6903) 
v_lshrrev_b32 v11, 16, v5 // (op 6903) 
v_lshlrev_b32 v10, 16, v10 // (op 6903) 
v_lshlrev_b32 v11, 16, v11 // (op 6903) 
v_mul_f32 v20, v0, v10 // (op 6903) 
v_mul_f32 v21, v0, v11 // (op 6903) 


v_add_f32 v10, v18, v14 // (op 6904) 
v_add_f32 v11, v19, v15 // (op 6904) 
v_add_f32 v12, v20, v16 // (op 6904) 
v_add_f32 v13, v21, v17 // (op 6904) 


v_lshrrev_b32 v14, 16, v10 // (op 6905) 
v_lshrrev_b32 v15, 16, v11 // (op 6905) 
v_lshrrev_b32 v16, 16, v12 // (op 6905) 
v_lshrrev_b32 v17, 16, v13 // (op 6905) 


v_pack_b32_f16 v10, v14, v15 // (op 6871) 
v_pack_b32_f16 v11, v16, v17 // (op 6871) 

s_lshl_b32 s0, s4, 5 // (op 6871) 
s_mul_i32 s1, 6, s0 // (op 6871) 
s_add_u32 s1, s1, 96 // (op 6871) 

v_add_u32 v3, v9, s1 // (op 6871) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6871) Store value





s_lshl_b32 s0, s2, 5 // (op 6911) 
s_mul_i32 s1, 7, s0 // (op 6911) 
s_add_u32 s1, s1, 96 // (op 6911) 

v_add_u32 v3, v8, s1 // (op 6911) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6911) Load value

v_accvgpr_read v10, a124 // (op 6925) 
v_accvgpr_read v11, a125 // (op 6925) 
v_accvgpr_read v12, a126 // (op 6925) 
v_accvgpr_read v13, a127 // (op 6925) 
v_mul_f32 v14, v7, v10 // (op 6925) 
v_mul_f32 v15, v7, v11 // (op 6925) 
v_mul_f32 v16, v7, v12 // (op 6925) 
v_mul_f32 v17, v7, v13 // (op 6925) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6908) 
v_lshrrev_b32 v11, 16, v4 // (op 6908) 
v_lshlrev_b32 v10, 16, v10 // (op 6908) 
v_lshlrev_b32 v11, 16, v11 // (op 6908) 
v_mul_f32 v18, v0, v10 // (op 6908) 
v_mul_f32 v19, v0, v11 // (op 6908) 
v_and_b32 v10, 65535, v5 // (op 6908) 
v_lshrrev_b32 v11, 16, v5 // (op 6908) 
v_lshlrev_b32 v10, 16, v10 // (op 6908) 
v_lshlrev_b32 v11, 16, v11 // (op 6908) 
v_mul_f32 v20, v0, v10 // (op 6908) 
v_mul_f32 v21, v0, v11 // (op 6908) 


v_add_f32 v10, v18, v14 // (op 6909) 
v_add_f32 v11, v19, v15 // (op 6909) 
v_add_f32 v12, v20, v16 // (op 6909) 
v_add_f32 v13, v21, v17 // (op 6909) 


v_lshrrev_b32 v14, 16, v10 // (op 6910) 
v_lshrrev_b32 v15, 16, v11 // (op 6910) 
v_lshrrev_b32 v16, 16, v12 // (op 6910) 
v_lshrrev_b32 v17, 16, v13 // (op 6910) 


v_pack_b32_f16 v10, v14, v15 // (op 6873) 
v_pack_b32_f16 v11, v16, v17 // (op 6873) 

s_lshl_b32 s0, s4, 5 // (op 6873) 
s_mul_i32 s1, 7, s0 // (op 6873) 
s_add_u32 s1, s1, 96 // (op 6873) 

v_add_u32 v3, v9, s1 // (op 6873) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6873) Store value





v_add_u32 v3, 64, v8 // (op 6705) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6705) Load value

v_accvgpr_read v10, a64 // (op 6684) 
v_accvgpr_read v11, a65 // (op 6684) 
v_accvgpr_read v12, a66 // (op 6684) 
v_accvgpr_read v13, a67 // (op 6684) 
v_mul_f32 v14, v7, v10 // (op 6684) 
v_mul_f32 v15, v7, v11 // (op 6684) 
v_mul_f32 v16, v7, v12 // (op 6684) 
v_mul_f32 v17, v7, v13 // (op 6684) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6704) 
v_lshrrev_b32 v11, 16, v4 // (op 6704) 
v_lshlrev_b32 v10, 16, v10 // (op 6704) 
v_lshlrev_b32 v11, 16, v11 // (op 6704) 
v_mul_f32 v18, v0, v10 // (op 6704) 
v_mul_f32 v19, v0, v11 // (op 6704) 
v_and_b32 v10, 65535, v5 // (op 6704) 
v_lshrrev_b32 v11, 16, v5 // (op 6704) 
v_lshlrev_b32 v10, 16, v10 // (op 6704) 
v_lshlrev_b32 v11, 16, v11 // (op 6704) 
v_mul_f32 v20, v0, v10 // (op 6704) 
v_mul_f32 v21, v0, v11 // (op 6704) 


v_add_f32 v10, v18, v14 // (op 6685) 
v_add_f32 v11, v19, v15 // (op 6685) 
v_add_f32 v12, v20, v16 // (op 6685) 
v_add_f32 v13, v21, v17 // (op 6685) 


v_lshrrev_b32 v14, 16, v10 // (op 6686) 
v_lshrrev_b32 v15, 16, v11 // (op 6686) 
v_lshrrev_b32 v16, 16, v12 // (op 6686) 
v_lshrrev_b32 v17, 16, v13 // (op 6686) 


v_pack_b32_f16 v10, v14, v15 // (op 6688) 
v_pack_b32_f16 v11, v16, v17 // (op 6688) 

v_add_u32 v3, 64, v9 // (op 6688) 

buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6688) Store value





s_lshl_b32 s0, s2, 5 // (op 6710) 
s_add_u32 s0, s0, 64 // (op 6710) 
v_add_u32 v3, v8, s0 // (op 6710) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6710) Load value

v_accvgpr_read v10, a68 // (op 6742) 
v_accvgpr_read v11, a69 // (op 6742) 
v_accvgpr_read v12, a70 // (op 6742) 
v_accvgpr_read v13, a71 // (op 6742) 
v_mul_f32 v14, v7, v10 // (op 6742) 
v_mul_f32 v15, v7, v11 // (op 6742) 
v_mul_f32 v16, v7, v12 // (op 6742) 
v_mul_f32 v17, v7, v13 // (op 6742) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6707) 
v_lshrrev_b32 v11, 16, v4 // (op 6707) 
v_lshlrev_b32 v10, 16, v10 // (op 6707) 
v_lshlrev_b32 v11, 16, v11 // (op 6707) 
v_mul_f32 v18, v0, v10 // (op 6707) 
v_mul_f32 v19, v0, v11 // (op 6707) 
v_and_b32 v10, 65535, v5 // (op 6707) 
v_lshrrev_b32 v11, 16, v5 // (op 6707) 
v_lshlrev_b32 v10, 16, v10 // (op 6707) 
v_lshlrev_b32 v11, 16, v11 // (op 6707) 
v_mul_f32 v20, v0, v10 // (op 6707) 
v_mul_f32 v21, v0, v11 // (op 6707) 


v_add_f32 v10, v18, v14 // (op 6708) 
v_add_f32 v11, v19, v15 // (op 6708) 
v_add_f32 v12, v20, v16 // (op 6708) 
v_add_f32 v13, v21, v17 // (op 6708) 


v_lshrrev_b32 v14, 16, v10 // (op 6709) 
v_lshrrev_b32 v15, 16, v11 // (op 6709) 
v_lshrrev_b32 v16, 16, v12 // (op 6709) 
v_lshrrev_b32 v17, 16, v13 // (op 6709) 


v_pack_b32_f16 v10, v14, v15 // (op 6690) 
v_pack_b32_f16 v11, v16, v17 // (op 6690) 

s_lshl_b32 s0, s4, 5 // (op 6690) 
s_add_u32 s0, s0, 64 // (op 6690) 
v_add_u32 v3, v9, s0 // (op 6690) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6690) Store value





s_lshl_b32 s0, s2, 6 // (op 6715) 
s_add_u32 s0, s0, 64 // (op 6715) 
v_add_u32 v3, v8, s0 // (op 6715) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6715) Load value

v_accvgpr_read v10, a72 // (op 6744) 
v_accvgpr_read v11, a73 // (op 6744) 
v_accvgpr_read v12, a74 // (op 6744) 
v_accvgpr_read v13, a75 // (op 6744) 
v_mul_f32 v14, v7, v10 // (op 6744) 
v_mul_f32 v15, v7, v11 // (op 6744) 
v_mul_f32 v16, v7, v12 // (op 6744) 
v_mul_f32 v17, v7, v13 // (op 6744) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6712) 
v_lshrrev_b32 v11, 16, v4 // (op 6712) 
v_lshlrev_b32 v10, 16, v10 // (op 6712) 
v_lshlrev_b32 v11, 16, v11 // (op 6712) 
v_mul_f32 v18, v0, v10 // (op 6712) 
v_mul_f32 v19, v0, v11 // (op 6712) 
v_and_b32 v10, 65535, v5 // (op 6712) 
v_lshrrev_b32 v11, 16, v5 // (op 6712) 
v_lshlrev_b32 v10, 16, v10 // (op 6712) 
v_lshlrev_b32 v11, 16, v11 // (op 6712) 
v_mul_f32 v20, v0, v10 // (op 6712) 
v_mul_f32 v21, v0, v11 // (op 6712) 


v_add_f32 v10, v18, v14 // (op 6713) 
v_add_f32 v11, v19, v15 // (op 6713) 
v_add_f32 v12, v20, v16 // (op 6713) 
v_add_f32 v13, v21, v17 // (op 6713) 


v_lshrrev_b32 v14, 16, v10 // (op 6714) 
v_lshrrev_b32 v15, 16, v11 // (op 6714) 
v_lshrrev_b32 v16, 16, v12 // (op 6714) 
v_lshrrev_b32 v17, 16, v13 // (op 6714) 


v_pack_b32_f16 v10, v14, v15 // (op 6692) 
v_pack_b32_f16 v11, v16, v17 // (op 6692) 

s_lshl_b32 s0, s4, 6 // (op 6692) 
s_add_u32 s0, s0, 64 // (op 6692) 
v_add_u32 v3, v9, s0 // (op 6692) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6692) Store value





s_lshl_b32 s0, s2, 5 // (op 6720) 
s_mul_i32 s1, 3, s0 // (op 6720) 
s_add_u32 s1, s1, 64 // (op 6720) 

v_add_u32 v3, v8, s1 // (op 6720) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6720) Load value

v_accvgpr_read v10, a76 // (op 6746) 
v_accvgpr_read v11, a77 // (op 6746) 
v_accvgpr_read v12, a78 // (op 6746) 
v_accvgpr_read v13, a79 // (op 6746) 
v_mul_f32 v14, v7, v10 // (op 6746) 
v_mul_f32 v15, v7, v11 // (op 6746) 
v_mul_f32 v16, v7, v12 // (op 6746) 
v_mul_f32 v17, v7, v13 // (op 6746) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6717) 
v_lshrrev_b32 v11, 16, v4 // (op 6717) 
v_lshlrev_b32 v10, 16, v10 // (op 6717) 
v_lshlrev_b32 v11, 16, v11 // (op 6717) 
v_mul_f32 v18, v0, v10 // (op 6717) 
v_mul_f32 v19, v0, v11 // (op 6717) 
v_and_b32 v10, 65535, v5 // (op 6717) 
v_lshrrev_b32 v11, 16, v5 // (op 6717) 
v_lshlrev_b32 v10, 16, v10 // (op 6717) 
v_lshlrev_b32 v11, 16, v11 // (op 6717) 
v_mul_f32 v20, v0, v10 // (op 6717) 
v_mul_f32 v21, v0, v11 // (op 6717) 


v_add_f32 v10, v18, v14 // (op 6718) 
v_add_f32 v11, v19, v15 // (op 6718) 
v_add_f32 v12, v20, v16 // (op 6718) 
v_add_f32 v13, v21, v17 // (op 6718) 


v_lshrrev_b32 v14, 16, v10 // (op 6719) 
v_lshrrev_b32 v15, 16, v11 // (op 6719) 
v_lshrrev_b32 v16, 16, v12 // (op 6719) 
v_lshrrev_b32 v17, 16, v13 // (op 6719) 


v_pack_b32_f16 v10, v14, v15 // (op 6694) 
v_pack_b32_f16 v11, v16, v17 // (op 6694) 

s_lshl_b32 s0, s4, 5 // (op 6694) 
s_mul_i32 s1, 3, s0 // (op 6694) 
s_add_u32 s1, s1, 64 // (op 6694) 

v_add_u32 v3, v9, s1 // (op 6694) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6694) Store value





s_lshl_b32 s0, s2, 7 // (op 6725) 
s_add_u32 s0, s0, 64 // (op 6725) 
v_add_u32 v3, v8, s0 // (op 6725) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6725) Load value

v_accvgpr_read v10, a80 // (op 6748) 
v_accvgpr_read v11, a81 // (op 6748) 
v_accvgpr_read v12, a82 // (op 6748) 
v_accvgpr_read v13, a83 // (op 6748) 
v_mul_f32 v14, v7, v10 // (op 6748) 
v_mul_f32 v15, v7, v11 // (op 6748) 
v_mul_f32 v16, v7, v12 // (op 6748) 
v_mul_f32 v17, v7, v13 // (op 6748) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6722) 
v_lshrrev_b32 v11, 16, v4 // (op 6722) 
v_lshlrev_b32 v10, 16, v10 // (op 6722) 
v_lshlrev_b32 v11, 16, v11 // (op 6722) 
v_mul_f32 v18, v0, v10 // (op 6722) 
v_mul_f32 v19, v0, v11 // (op 6722) 
v_and_b32 v10, 65535, v5 // (op 6722) 
v_lshrrev_b32 v11, 16, v5 // (op 6722) 
v_lshlrev_b32 v10, 16, v10 // (op 6722) 
v_lshlrev_b32 v11, 16, v11 // (op 6722) 
v_mul_f32 v20, v0, v10 // (op 6722) 
v_mul_f32 v21, v0, v11 // (op 6722) 


v_add_f32 v10, v18, v14 // (op 6723) 
v_add_f32 v11, v19, v15 // (op 6723) 
v_add_f32 v12, v20, v16 // (op 6723) 
v_add_f32 v13, v21, v17 // (op 6723) 


v_lshrrev_b32 v14, 16, v10 // (op 6724) 
v_lshrrev_b32 v15, 16, v11 // (op 6724) 
v_lshrrev_b32 v16, 16, v12 // (op 6724) 
v_lshrrev_b32 v17, 16, v13 // (op 6724) 


v_pack_b32_f16 v10, v14, v15 // (op 6696) 
v_pack_b32_f16 v11, v16, v17 // (op 6696) 

s_lshl_b32 s0, s4, 7 // (op 6696) 
s_add_u32 s0, s0, 64 // (op 6696) 
v_add_u32 v3, v9, s0 // (op 6696) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6696) Store value





s_lshl_b32 s0, s2, 5 // (op 6730) 
s_mul_i32 s1, 5, s0 // (op 6730) 
s_add_u32 s1, s1, 64 // (op 6730) 

v_add_u32 v3, v8, s1 // (op 6730) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6730) Load value

v_accvgpr_read v10, a84 // (op 6750) 
v_accvgpr_read v11, a85 // (op 6750) 
v_accvgpr_read v12, a86 // (op 6750) 
v_accvgpr_read v13, a87 // (op 6750) 
v_mul_f32 v14, v7, v10 // (op 6750) 
v_mul_f32 v15, v7, v11 // (op 6750) 
v_mul_f32 v16, v7, v12 // (op 6750) 
v_mul_f32 v17, v7, v13 // (op 6750) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6727) 
v_lshrrev_b32 v11, 16, v4 // (op 6727) 
v_lshlrev_b32 v10, 16, v10 // (op 6727) 
v_lshlrev_b32 v11, 16, v11 // (op 6727) 
v_mul_f32 v18, v0, v10 // (op 6727) 
v_mul_f32 v19, v0, v11 // (op 6727) 
v_and_b32 v10, 65535, v5 // (op 6727) 
v_lshrrev_b32 v11, 16, v5 // (op 6727) 
v_lshlrev_b32 v10, 16, v10 // (op 6727) 
v_lshlrev_b32 v11, 16, v11 // (op 6727) 
v_mul_f32 v20, v0, v10 // (op 6727) 
v_mul_f32 v21, v0, v11 // (op 6727) 


v_add_f32 v10, v18, v14 // (op 6728) 
v_add_f32 v11, v19, v15 // (op 6728) 
v_add_f32 v12, v20, v16 // (op 6728) 
v_add_f32 v13, v21, v17 // (op 6728) 


v_lshrrev_b32 v14, 16, v10 // (op 6729) 
v_lshrrev_b32 v15, 16, v11 // (op 6729) 
v_lshrrev_b32 v16, 16, v12 // (op 6729) 
v_lshrrev_b32 v17, 16, v13 // (op 6729) 


v_pack_b32_f16 v10, v14, v15 // (op 6698) 
v_pack_b32_f16 v11, v16, v17 // (op 6698) 

s_lshl_b32 s0, s4, 5 // (op 6698) 
s_mul_i32 s1, 5, s0 // (op 6698) 
s_add_u32 s1, s1, 64 // (op 6698) 

v_add_u32 v3, v9, s1 // (op 6698) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6698) Store value





s_lshl_b32 s0, s2, 5 // (op 6735) 
s_mul_i32 s1, 6, s0 // (op 6735) 
s_add_u32 s1, s1, 64 // (op 6735) 

v_add_u32 v3, v8, s1 // (op 6735) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6735) Load value

v_accvgpr_read v10, a88 // (op 6752) 
v_accvgpr_read v11, a89 // (op 6752) 
v_accvgpr_read v12, a90 // (op 6752) 
v_accvgpr_read v13, a91 // (op 6752) 
v_mul_f32 v14, v7, v10 // (op 6752) 
v_mul_f32 v15, v7, v11 // (op 6752) 
v_mul_f32 v16, v7, v12 // (op 6752) 
v_mul_f32 v17, v7, v13 // (op 6752) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6732) 
v_lshrrev_b32 v11, 16, v4 // (op 6732) 
v_lshlrev_b32 v10, 16, v10 // (op 6732) 
v_lshlrev_b32 v11, 16, v11 // (op 6732) 
v_mul_f32 v18, v0, v10 // (op 6732) 
v_mul_f32 v19, v0, v11 // (op 6732) 
v_and_b32 v10, 65535, v5 // (op 6732) 
v_lshrrev_b32 v11, 16, v5 // (op 6732) 
v_lshlrev_b32 v10, 16, v10 // (op 6732) 
v_lshlrev_b32 v11, 16, v11 // (op 6732) 
v_mul_f32 v20, v0, v10 // (op 6732) 
v_mul_f32 v21, v0, v11 // (op 6732) 


v_add_f32 v10, v18, v14 // (op 6733) 
v_add_f32 v11, v19, v15 // (op 6733) 
v_add_f32 v12, v20, v16 // (op 6733) 
v_add_f32 v13, v21, v17 // (op 6733) 


v_lshrrev_b32 v14, 16, v10 // (op 6734) 
v_lshrrev_b32 v15, 16, v11 // (op 6734) 
v_lshrrev_b32 v16, 16, v12 // (op 6734) 
v_lshrrev_b32 v17, 16, v13 // (op 6734) 


v_pack_b32_f16 v10, v14, v15 // (op 6700) 
v_pack_b32_f16 v11, v16, v17 // (op 6700) 

s_lshl_b32 s0, s4, 5 // (op 6700) 
s_mul_i32 s1, 6, s0 // (op 6700) 
s_add_u32 s1, s1, 64 // (op 6700) 

v_add_u32 v3, v9, s1 // (op 6700) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6700) Store value





s_lshl_b32 s0, s2, 5 // (op 6740) 
s_mul_i32 s1, 7, s0 // (op 6740) 
s_add_u32 s1, s1, 64 // (op 6740) 

v_add_u32 v3, v8, s1 // (op 6740) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6740) Load value

v_accvgpr_read v10, a92 // (op 6754) 
v_accvgpr_read v11, a93 // (op 6754) 
v_accvgpr_read v12, a94 // (op 6754) 
v_accvgpr_read v13, a95 // (op 6754) 
v_mul_f32 v14, v7, v10 // (op 6754) 
v_mul_f32 v15, v7, v11 // (op 6754) 
v_mul_f32 v16, v7, v12 // (op 6754) 
v_mul_f32 v17, v7, v13 // (op 6754) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6737) 
v_lshrrev_b32 v11, 16, v4 // (op 6737) 
v_lshlrev_b32 v10, 16, v10 // (op 6737) 
v_lshlrev_b32 v11, 16, v11 // (op 6737) 
v_mul_f32 v18, v0, v10 // (op 6737) 
v_mul_f32 v19, v0, v11 // (op 6737) 
v_and_b32 v10, 65535, v5 // (op 6737) 
v_lshrrev_b32 v11, 16, v5 // (op 6737) 
v_lshlrev_b32 v10, 16, v10 // (op 6737) 
v_lshlrev_b32 v11, 16, v11 // (op 6737) 
v_mul_f32 v20, v0, v10 // (op 6737) 
v_mul_f32 v21, v0, v11 // (op 6737) 


v_add_f32 v10, v18, v14 // (op 6738) 
v_add_f32 v11, v19, v15 // (op 6738) 
v_add_f32 v12, v20, v16 // (op 6738) 
v_add_f32 v13, v21, v17 // (op 6738) 


v_lshrrev_b32 v14, 16, v10 // (op 6739) 
v_lshrrev_b32 v15, 16, v11 // (op 6739) 
v_lshrrev_b32 v16, 16, v12 // (op 6739) 
v_lshrrev_b32 v17, 16, v13 // (op 6739) 


v_pack_b32_f16 v10, v14, v15 // (op 6702) 
v_pack_b32_f16 v11, v16, v17 // (op 6702) 

s_lshl_b32 s0, s4, 5 // (op 6702) 
s_mul_i32 s1, 7, s0 // (op 6702) 
s_add_u32 s1, s1, 64 // (op 6702) 

v_add_u32 v3, v9, s1 // (op 6702) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6702) Store value





v_add_u32 v3, 32, v8 // (op 6534) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6534) Load value

v_accvgpr_read v10, a32 // (op 6513) 
v_accvgpr_read v11, a33 // (op 6513) 
v_accvgpr_read v12, a34 // (op 6513) 
v_accvgpr_read v13, a35 // (op 6513) 
v_mul_f32 v14, v7, v10 // (op 6513) 
v_mul_f32 v15, v7, v11 // (op 6513) 
v_mul_f32 v16, v7, v12 // (op 6513) 
v_mul_f32 v17, v7, v13 // (op 6513) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6533) 
v_lshrrev_b32 v11, 16, v4 // (op 6533) 
v_lshlrev_b32 v10, 16, v10 // (op 6533) 
v_lshlrev_b32 v11, 16, v11 // (op 6533) 
v_mul_f32 v18, v0, v10 // (op 6533) 
v_mul_f32 v19, v0, v11 // (op 6533) 
v_and_b32 v10, 65535, v5 // (op 6533) 
v_lshrrev_b32 v11, 16, v5 // (op 6533) 
v_lshlrev_b32 v10, 16, v10 // (op 6533) 
v_lshlrev_b32 v11, 16, v11 // (op 6533) 
v_mul_f32 v20, v0, v10 // (op 6533) 
v_mul_f32 v21, v0, v11 // (op 6533) 


v_add_f32 v10, v18, v14 // (op 6514) 
v_add_f32 v11, v19, v15 // (op 6514) 
v_add_f32 v12, v20, v16 // (op 6514) 
v_add_f32 v13, v21, v17 // (op 6514) 


v_lshrrev_b32 v14, 16, v10 // (op 6515) 
v_lshrrev_b32 v15, 16, v11 // (op 6515) 
v_lshrrev_b32 v16, 16, v12 // (op 6515) 
v_lshrrev_b32 v17, 16, v13 // (op 6515) 


v_pack_b32_f16 v10, v14, v15 // (op 6517) 
v_pack_b32_f16 v11, v16, v17 // (op 6517) 

v_add_u32 v3, 32, v9 // (op 6517) 

buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6517) Store value





s_lshl_b32 s0, s2, 5 // (op 6539) 
s_add_u32 s0, s0, 32 // (op 6539) 
v_add_u32 v3, v8, s0 // (op 6539) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6539) Load value

v_accvgpr_read v10, a36 // (op 6571) 
v_accvgpr_read v11, a37 // (op 6571) 
v_accvgpr_read v12, a38 // (op 6571) 
v_accvgpr_read v13, a39 // (op 6571) 
v_mul_f32 v14, v7, v10 // (op 6571) 
v_mul_f32 v15, v7, v11 // (op 6571) 
v_mul_f32 v16, v7, v12 // (op 6571) 
v_mul_f32 v17, v7, v13 // (op 6571) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6536) 
v_lshrrev_b32 v11, 16, v4 // (op 6536) 
v_lshlrev_b32 v10, 16, v10 // (op 6536) 
v_lshlrev_b32 v11, 16, v11 // (op 6536) 
v_mul_f32 v18, v0, v10 // (op 6536) 
v_mul_f32 v19, v0, v11 // (op 6536) 
v_and_b32 v10, 65535, v5 // (op 6536) 
v_lshrrev_b32 v11, 16, v5 // (op 6536) 
v_lshlrev_b32 v10, 16, v10 // (op 6536) 
v_lshlrev_b32 v11, 16, v11 // (op 6536) 
v_mul_f32 v20, v0, v10 // (op 6536) 
v_mul_f32 v21, v0, v11 // (op 6536) 


v_add_f32 v10, v18, v14 // (op 6537) 
v_add_f32 v11, v19, v15 // (op 6537) 
v_add_f32 v12, v20, v16 // (op 6537) 
v_add_f32 v13, v21, v17 // (op 6537) 


v_lshrrev_b32 v14, 16, v10 // (op 6538) 
v_lshrrev_b32 v15, 16, v11 // (op 6538) 
v_lshrrev_b32 v16, 16, v12 // (op 6538) 
v_lshrrev_b32 v17, 16, v13 // (op 6538) 


v_pack_b32_f16 v10, v14, v15 // (op 6519) 
v_pack_b32_f16 v11, v16, v17 // (op 6519) 

s_lshl_b32 s0, s4, 5 // (op 6519) 
s_add_u32 s0, s0, 32 // (op 6519) 
v_add_u32 v3, v9, s0 // (op 6519) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6519) Store value





s_lshl_b32 s0, s2, 6 // (op 6544) 
s_add_u32 s0, s0, 32 // (op 6544) 
v_add_u32 v3, v8, s0 // (op 6544) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6544) Load value

v_accvgpr_read v10, a40 // (op 6573) 
v_accvgpr_read v11, a41 // (op 6573) 
v_accvgpr_read v12, a42 // (op 6573) 
v_accvgpr_read v13, a43 // (op 6573) 
v_mul_f32 v14, v7, v10 // (op 6573) 
v_mul_f32 v15, v7, v11 // (op 6573) 
v_mul_f32 v16, v7, v12 // (op 6573) 
v_mul_f32 v17, v7, v13 // (op 6573) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6541) 
v_lshrrev_b32 v11, 16, v4 // (op 6541) 
v_lshlrev_b32 v10, 16, v10 // (op 6541) 
v_lshlrev_b32 v11, 16, v11 // (op 6541) 
v_mul_f32 v18, v0, v10 // (op 6541) 
v_mul_f32 v19, v0, v11 // (op 6541) 
v_and_b32 v10, 65535, v5 // (op 6541) 
v_lshrrev_b32 v11, 16, v5 // (op 6541) 
v_lshlrev_b32 v10, 16, v10 // (op 6541) 
v_lshlrev_b32 v11, 16, v11 // (op 6541) 
v_mul_f32 v20, v0, v10 // (op 6541) 
v_mul_f32 v21, v0, v11 // (op 6541) 


v_add_f32 v10, v18, v14 // (op 6542) 
v_add_f32 v11, v19, v15 // (op 6542) 
v_add_f32 v12, v20, v16 // (op 6542) 
v_add_f32 v13, v21, v17 // (op 6542) 


v_lshrrev_b32 v14, 16, v10 // (op 6543) 
v_lshrrev_b32 v15, 16, v11 // (op 6543) 
v_lshrrev_b32 v16, 16, v12 // (op 6543) 
v_lshrrev_b32 v17, 16, v13 // (op 6543) 


v_pack_b32_f16 v10, v14, v15 // (op 6521) 
v_pack_b32_f16 v11, v16, v17 // (op 6521) 

s_lshl_b32 s0, s4, 6 // (op 6521) 
s_add_u32 s0, s0, 32 // (op 6521) 
v_add_u32 v3, v9, s0 // (op 6521) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6521) Store value





s_lshl_b32 s0, s2, 5 // (op 6549) 
s_mul_i32 s1, 3, s0 // (op 6549) 
s_add_u32 s1, s1, 32 // (op 6549) 

v_add_u32 v3, v8, s1 // (op 6549) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6549) Load value

v_accvgpr_read v10, a44 // (op 6575) 
v_accvgpr_read v11, a45 // (op 6575) 
v_accvgpr_read v12, a46 // (op 6575) 
v_accvgpr_read v13, a47 // (op 6575) 
v_mul_f32 v14, v7, v10 // (op 6575) 
v_mul_f32 v15, v7, v11 // (op 6575) 
v_mul_f32 v16, v7, v12 // (op 6575) 
v_mul_f32 v17, v7, v13 // (op 6575) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6546) 
v_lshrrev_b32 v11, 16, v4 // (op 6546) 
v_lshlrev_b32 v10, 16, v10 // (op 6546) 
v_lshlrev_b32 v11, 16, v11 // (op 6546) 
v_mul_f32 v18, v0, v10 // (op 6546) 
v_mul_f32 v19, v0, v11 // (op 6546) 
v_and_b32 v10, 65535, v5 // (op 6546) 
v_lshrrev_b32 v11, 16, v5 // (op 6546) 
v_lshlrev_b32 v10, 16, v10 // (op 6546) 
v_lshlrev_b32 v11, 16, v11 // (op 6546) 
v_mul_f32 v20, v0, v10 // (op 6546) 
v_mul_f32 v21, v0, v11 // (op 6546) 


v_add_f32 v10, v18, v14 // (op 6547) 
v_add_f32 v11, v19, v15 // (op 6547) 
v_add_f32 v12, v20, v16 // (op 6547) 
v_add_f32 v13, v21, v17 // (op 6547) 


v_lshrrev_b32 v14, 16, v10 // (op 6548) 
v_lshrrev_b32 v15, 16, v11 // (op 6548) 
v_lshrrev_b32 v16, 16, v12 // (op 6548) 
v_lshrrev_b32 v17, 16, v13 // (op 6548) 


v_pack_b32_f16 v10, v14, v15 // (op 6523) 
v_pack_b32_f16 v11, v16, v17 // (op 6523) 

s_lshl_b32 s0, s4, 5 // (op 6523) 
s_mul_i32 s1, 3, s0 // (op 6523) 
s_add_u32 s1, s1, 32 // (op 6523) 

v_add_u32 v3, v9, s1 // (op 6523) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6523) Store value





s_lshl_b32 s0, s2, 7 // (op 6554) 
s_add_u32 s0, s0, 32 // (op 6554) 
v_add_u32 v3, v8, s0 // (op 6554) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6554) Load value

v_accvgpr_read v10, a48 // (op 6577) 
v_accvgpr_read v11, a49 // (op 6577) 
v_accvgpr_read v12, a50 // (op 6577) 
v_accvgpr_read v13, a51 // (op 6577) 
v_mul_f32 v14, v7, v10 // (op 6577) 
v_mul_f32 v15, v7, v11 // (op 6577) 
v_mul_f32 v16, v7, v12 // (op 6577) 
v_mul_f32 v17, v7, v13 // (op 6577) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6551) 
v_lshrrev_b32 v11, 16, v4 // (op 6551) 
v_lshlrev_b32 v10, 16, v10 // (op 6551) 
v_lshlrev_b32 v11, 16, v11 // (op 6551) 
v_mul_f32 v18, v0, v10 // (op 6551) 
v_mul_f32 v19, v0, v11 // (op 6551) 
v_and_b32 v10, 65535, v5 // (op 6551) 
v_lshrrev_b32 v11, 16, v5 // (op 6551) 
v_lshlrev_b32 v10, 16, v10 // (op 6551) 
v_lshlrev_b32 v11, 16, v11 // (op 6551) 
v_mul_f32 v20, v0, v10 // (op 6551) 
v_mul_f32 v21, v0, v11 // (op 6551) 


v_add_f32 v10, v18, v14 // (op 6552) 
v_add_f32 v11, v19, v15 // (op 6552) 
v_add_f32 v12, v20, v16 // (op 6552) 
v_add_f32 v13, v21, v17 // (op 6552) 


v_lshrrev_b32 v14, 16, v10 // (op 6553) 
v_lshrrev_b32 v15, 16, v11 // (op 6553) 
v_lshrrev_b32 v16, 16, v12 // (op 6553) 
v_lshrrev_b32 v17, 16, v13 // (op 6553) 


v_pack_b32_f16 v10, v14, v15 // (op 6525) 
v_pack_b32_f16 v11, v16, v17 // (op 6525) 

s_lshl_b32 s0, s4, 7 // (op 6525) 
s_add_u32 s0, s0, 32 // (op 6525) 
v_add_u32 v3, v9, s0 // (op 6525) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6525) Store value





s_lshl_b32 s0, s2, 5 // (op 6559) 
s_mul_i32 s1, 5, s0 // (op 6559) 
s_add_u32 s1, s1, 32 // (op 6559) 

v_add_u32 v3, v8, s1 // (op 6559) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6559) Load value

v_accvgpr_read v10, a52 // (op 6579) 
v_accvgpr_read v11, a53 // (op 6579) 
v_accvgpr_read v12, a54 // (op 6579) 
v_accvgpr_read v13, a55 // (op 6579) 
v_mul_f32 v14, v7, v10 // (op 6579) 
v_mul_f32 v15, v7, v11 // (op 6579) 
v_mul_f32 v16, v7, v12 // (op 6579) 
v_mul_f32 v17, v7, v13 // (op 6579) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6556) 
v_lshrrev_b32 v11, 16, v4 // (op 6556) 
v_lshlrev_b32 v10, 16, v10 // (op 6556) 
v_lshlrev_b32 v11, 16, v11 // (op 6556) 
v_mul_f32 v18, v0, v10 // (op 6556) 
v_mul_f32 v19, v0, v11 // (op 6556) 
v_and_b32 v10, 65535, v5 // (op 6556) 
v_lshrrev_b32 v11, 16, v5 // (op 6556) 
v_lshlrev_b32 v10, 16, v10 // (op 6556) 
v_lshlrev_b32 v11, 16, v11 // (op 6556) 
v_mul_f32 v20, v0, v10 // (op 6556) 
v_mul_f32 v21, v0, v11 // (op 6556) 


v_add_f32 v10, v18, v14 // (op 6557) 
v_add_f32 v11, v19, v15 // (op 6557) 
v_add_f32 v12, v20, v16 // (op 6557) 
v_add_f32 v13, v21, v17 // (op 6557) 


v_lshrrev_b32 v14, 16, v10 // (op 6558) 
v_lshrrev_b32 v15, 16, v11 // (op 6558) 
v_lshrrev_b32 v16, 16, v12 // (op 6558) 
v_lshrrev_b32 v17, 16, v13 // (op 6558) 


v_pack_b32_f16 v10, v14, v15 // (op 6527) 
v_pack_b32_f16 v11, v16, v17 // (op 6527) 

s_lshl_b32 s0, s4, 5 // (op 6527) 
s_mul_i32 s1, 5, s0 // (op 6527) 
s_add_u32 s1, s1, 32 // (op 6527) 

v_add_u32 v3, v9, s1 // (op 6527) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6527) Store value





s_lshl_b32 s0, s2, 5 // (op 6564) 
s_mul_i32 s1, 6, s0 // (op 6564) 
s_add_u32 s1, s1, 32 // (op 6564) 

v_add_u32 v3, v8, s1 // (op 6564) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6564) Load value

v_accvgpr_read v10, a56 // (op 6581) 
v_accvgpr_read v11, a57 // (op 6581) 
v_accvgpr_read v12, a58 // (op 6581) 
v_accvgpr_read v13, a59 // (op 6581) 
v_mul_f32 v14, v7, v10 // (op 6581) 
v_mul_f32 v15, v7, v11 // (op 6581) 
v_mul_f32 v16, v7, v12 // (op 6581) 
v_mul_f32 v17, v7, v13 // (op 6581) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6561) 
v_lshrrev_b32 v11, 16, v4 // (op 6561) 
v_lshlrev_b32 v10, 16, v10 // (op 6561) 
v_lshlrev_b32 v11, 16, v11 // (op 6561) 
v_mul_f32 v18, v0, v10 // (op 6561) 
v_mul_f32 v19, v0, v11 // (op 6561) 
v_and_b32 v10, 65535, v5 // (op 6561) 
v_lshrrev_b32 v11, 16, v5 // (op 6561) 
v_lshlrev_b32 v10, 16, v10 // (op 6561) 
v_lshlrev_b32 v11, 16, v11 // (op 6561) 
v_mul_f32 v20, v0, v10 // (op 6561) 
v_mul_f32 v21, v0, v11 // (op 6561) 


v_add_f32 v10, v18, v14 // (op 6562) 
v_add_f32 v11, v19, v15 // (op 6562) 
v_add_f32 v12, v20, v16 // (op 6562) 
v_add_f32 v13, v21, v17 // (op 6562) 


v_lshrrev_b32 v14, 16, v10 // (op 6563) 
v_lshrrev_b32 v15, 16, v11 // (op 6563) 
v_lshrrev_b32 v16, 16, v12 // (op 6563) 
v_lshrrev_b32 v17, 16, v13 // (op 6563) 


v_pack_b32_f16 v10, v14, v15 // (op 6529) 
v_pack_b32_f16 v11, v16, v17 // (op 6529) 

s_lshl_b32 s0, s4, 5 // (op 6529) 
s_mul_i32 s1, 6, s0 // (op 6529) 
s_add_u32 s1, s1, 32 // (op 6529) 

v_add_u32 v3, v9, s1 // (op 6529) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6529) Store value





s_lshl_b32 s0, s2, 5 // (op 6569) 
s_mul_i32 s1, 7, s0 // (op 6569) 
s_add_u32 s1, s1, 32 // (op 6569) 

v_add_u32 v3, v8, s1 // (op 6569) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6569) Load value

v_accvgpr_read v10, a60 // (op 6583) 
v_accvgpr_read v11, a61 // (op 6583) 
v_accvgpr_read v12, a62 // (op 6583) 
v_accvgpr_read v13, a63 // (op 6583) 
v_mul_f32 v14, v7, v10 // (op 6583) 
v_mul_f32 v15, v7, v11 // (op 6583) 
v_mul_f32 v16, v7, v12 // (op 6583) 
v_mul_f32 v17, v7, v13 // (op 6583) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6566) 
v_lshrrev_b32 v11, 16, v4 // (op 6566) 
v_lshlrev_b32 v10, 16, v10 // (op 6566) 
v_lshlrev_b32 v11, 16, v11 // (op 6566) 
v_mul_f32 v18, v0, v10 // (op 6566) 
v_mul_f32 v19, v0, v11 // (op 6566) 
v_and_b32 v10, 65535, v5 // (op 6566) 
v_lshrrev_b32 v11, 16, v5 // (op 6566) 
v_lshlrev_b32 v10, 16, v10 // (op 6566) 
v_lshlrev_b32 v11, 16, v11 // (op 6566) 
v_mul_f32 v20, v0, v10 // (op 6566) 
v_mul_f32 v21, v0, v11 // (op 6566) 


v_add_f32 v10, v18, v14 // (op 6567) 
v_add_f32 v11, v19, v15 // (op 6567) 
v_add_f32 v12, v20, v16 // (op 6567) 
v_add_f32 v13, v21, v17 // (op 6567) 


v_lshrrev_b32 v14, 16, v10 // (op 6568) 
v_lshrrev_b32 v15, 16, v11 // (op 6568) 
v_lshrrev_b32 v16, 16, v12 // (op 6568) 
v_lshrrev_b32 v17, 16, v13 // (op 6568) 


v_pack_b32_f16 v10, v14, v15 // (op 6531) 
v_pack_b32_f16 v11, v16, v17 // (op 6531) 

s_lshl_b32 s0, s4, 5 // (op 6531) 
s_mul_i32 s1, 7, s0 // (op 6531) 
s_add_u32 s1, s1, 32 // (op 6531) 

v_add_u32 v3, v9, s1 // (op 6531) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6531) Store value





v_mov_b32 v3, v8 // (op 129) reg expression

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 129) Load value

v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 130) 
v_lshrrev_b32 v11, 16, v4 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v18, v0, v10 // (op 130) 
v_mul_f32 v19, v0, v11 // (op 130) 
v_and_b32 v10, 65535, v5 // (op 130) 
v_lshrrev_b32 v11, 16, v5 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v20, v0, v10 // (op 130) 
v_mul_f32 v21, v0, v11 // (op 130) 


v_add_f32 v10, v18, v14 // (op 126) 
v_add_f32 v11, v19, v15 // (op 126) 
v_add_f32 v12, v20, v16 // (op 126) 
v_add_f32 v13, v21, v17 // (op 126) 


v_lshrrev_b32 v14, 16, v10 // (op 127) 
v_lshrrev_b32 v15, 16, v11 // (op 127) 
v_lshrrev_b32 v16, 16, v12 // (op 127) 
v_lshrrev_b32 v17, 16, v13 // (op 127) 


v_pack_b32_f16 v10, v14, v15 // (op 128) 
v_pack_b32_f16 v11, v16, v17 // (op 128) 

v_mov_b32 v3, v9 // (op 128) reg expression

buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 6338) 
v_add_u32 v3, v8, s0 // (op 6338) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6338) Load value

v_accvgpr_read v10, a4 // (op 6334) 
v_accvgpr_read v11, a5 // (op 6334) 
v_accvgpr_read v12, a6 // (op 6334) 
v_accvgpr_read v13, a7 // (op 6334) 
v_mul_f32 v14, v7, v10 // (op 6334) 
v_mul_f32 v15, v7, v11 // (op 6334) 
v_mul_f32 v16, v7, v12 // (op 6334) 
v_mul_f32 v17, v7, v13 // (op 6334) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6339) 
v_lshrrev_b32 v11, 16, v4 // (op 6339) 
v_lshlrev_b32 v10, 16, v10 // (op 6339) 
v_lshlrev_b32 v11, 16, v11 // (op 6339) 
v_mul_f32 v18, v0, v10 // (op 6339) 
v_mul_f32 v19, v0, v11 // (op 6339) 
v_and_b32 v10, 65535, v5 // (op 6339) 
v_lshrrev_b32 v11, 16, v5 // (op 6339) 
v_lshlrev_b32 v10, 16, v10 // (op 6339) 
v_lshlrev_b32 v11, 16, v11 // (op 6339) 
v_mul_f32 v20, v0, v10 // (op 6339) 
v_mul_f32 v21, v0, v11 // (op 6339) 


v_add_f32 v10, v18, v14 // (op 6335) 
v_add_f32 v11, v19, v15 // (op 6335) 
v_add_f32 v12, v20, v16 // (op 6335) 
v_add_f32 v13, v21, v17 // (op 6335) 


v_lshrrev_b32 v14, 16, v10 // (op 6336) 
v_lshrrev_b32 v15, 16, v11 // (op 6336) 
v_lshrrev_b32 v16, 16, v12 // (op 6336) 
v_lshrrev_b32 v17, 16, v13 // (op 6336) 


v_pack_b32_f16 v10, v14, v15 // (op 6337) 
v_pack_b32_f16 v11, v16, v17 // (op 6337) 

s_lshl_b32 s0, s4, 5 // (op 6337) 
v_add_u32 v3, v9, s0 // (op 6337) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6337) Store value





s_lshl_b32 s0, s2, 6 // (op 6352) 
v_add_u32 v3, v8, s0 // (op 6352) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6352) Load value

v_accvgpr_read v10, a8 // (op 6348) 
v_accvgpr_read v11, a9 // (op 6348) 
v_accvgpr_read v12, a10 // (op 6348) 
v_accvgpr_read v13, a11 // (op 6348) 
v_mul_f32 v14, v7, v10 // (op 6348) 
v_mul_f32 v15, v7, v11 // (op 6348) 
v_mul_f32 v16, v7, v12 // (op 6348) 
v_mul_f32 v17, v7, v13 // (op 6348) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6353) 
v_lshrrev_b32 v11, 16, v4 // (op 6353) 
v_lshlrev_b32 v10, 16, v10 // (op 6353) 
v_lshlrev_b32 v11, 16, v11 // (op 6353) 
v_mul_f32 v18, v0, v10 // (op 6353) 
v_mul_f32 v19, v0, v11 // (op 6353) 
v_and_b32 v10, 65535, v5 // (op 6353) 
v_lshrrev_b32 v11, 16, v5 // (op 6353) 
v_lshlrev_b32 v10, 16, v10 // (op 6353) 
v_lshlrev_b32 v11, 16, v11 // (op 6353) 
v_mul_f32 v20, v0, v10 // (op 6353) 
v_mul_f32 v21, v0, v11 // (op 6353) 


v_add_f32 v10, v18, v14 // (op 6349) 
v_add_f32 v11, v19, v15 // (op 6349) 
v_add_f32 v12, v20, v16 // (op 6349) 
v_add_f32 v13, v21, v17 // (op 6349) 


v_lshrrev_b32 v14, 16, v10 // (op 6350) 
v_lshrrev_b32 v15, 16, v11 // (op 6350) 
v_lshrrev_b32 v16, 16, v12 // (op 6350) 
v_lshrrev_b32 v17, 16, v13 // (op 6350) 


v_pack_b32_f16 v10, v14, v15 // (op 6351) 
v_pack_b32_f16 v11, v16, v17 // (op 6351) 

s_lshl_b32 s0, s4, 6 // (op 6351) 
v_add_u32 v3, v9, s0 // (op 6351) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6351) Store value





s_lshl_b32 s0, s2, 5 // (op 6366) 
s_mul_i32 s1, 3, s0 // (op 6366) 

v_add_u32 v3, v8, s1 // (op 6366) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6366) Load value

v_accvgpr_read v10, a12 // (op 6362) 
v_accvgpr_read v11, a13 // (op 6362) 
v_accvgpr_read v12, a14 // (op 6362) 
v_accvgpr_read v13, a15 // (op 6362) 
v_mul_f32 v14, v7, v10 // (op 6362) 
v_mul_f32 v15, v7, v11 // (op 6362) 
v_mul_f32 v16, v7, v12 // (op 6362) 
v_mul_f32 v17, v7, v13 // (op 6362) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6367) 
v_lshrrev_b32 v11, 16, v4 // (op 6367) 
v_lshlrev_b32 v10, 16, v10 // (op 6367) 
v_lshlrev_b32 v11, 16, v11 // (op 6367) 
v_mul_f32 v18, v0, v10 // (op 6367) 
v_mul_f32 v19, v0, v11 // (op 6367) 
v_and_b32 v10, 65535, v5 // (op 6367) 
v_lshrrev_b32 v11, 16, v5 // (op 6367) 
v_lshlrev_b32 v10, 16, v10 // (op 6367) 
v_lshlrev_b32 v11, 16, v11 // (op 6367) 
v_mul_f32 v20, v0, v10 // (op 6367) 
v_mul_f32 v21, v0, v11 // (op 6367) 


v_add_f32 v10, v18, v14 // (op 6363) 
v_add_f32 v11, v19, v15 // (op 6363) 
v_add_f32 v12, v20, v16 // (op 6363) 
v_add_f32 v13, v21, v17 // (op 6363) 


v_lshrrev_b32 v14, 16, v10 // (op 6364) 
v_lshrrev_b32 v15, 16, v11 // (op 6364) 
v_lshrrev_b32 v16, 16, v12 // (op 6364) 
v_lshrrev_b32 v17, 16, v13 // (op 6364) 


v_pack_b32_f16 v10, v14, v15 // (op 6365) 
v_pack_b32_f16 v11, v16, v17 // (op 6365) 

s_lshl_b32 s0, s4, 5 // (op 6365) 
s_mul_i32 s1, 3, s0 // (op 6365) 

v_add_u32 v3, v9, s1 // (op 6365) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6365) Store value





s_lshl_b32 s0, s2, 7 // (op 6380) 
v_add_u32 v3, v8, s0 // (op 6380) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6380) Load value

v_accvgpr_read v10, a16 // (op 6376) 
v_accvgpr_read v11, a17 // (op 6376) 
v_accvgpr_read v12, a18 // (op 6376) 
v_accvgpr_read v13, a19 // (op 6376) 
v_mul_f32 v14, v7, v10 // (op 6376) 
v_mul_f32 v15, v7, v11 // (op 6376) 
v_mul_f32 v16, v7, v12 // (op 6376) 
v_mul_f32 v17, v7, v13 // (op 6376) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6381) 
v_lshrrev_b32 v11, 16, v4 // (op 6381) 
v_lshlrev_b32 v10, 16, v10 // (op 6381) 
v_lshlrev_b32 v11, 16, v11 // (op 6381) 
v_mul_f32 v18, v0, v10 // (op 6381) 
v_mul_f32 v19, v0, v11 // (op 6381) 
v_and_b32 v10, 65535, v5 // (op 6381) 
v_lshrrev_b32 v11, 16, v5 // (op 6381) 
v_lshlrev_b32 v10, 16, v10 // (op 6381) 
v_lshlrev_b32 v11, 16, v11 // (op 6381) 
v_mul_f32 v20, v0, v10 // (op 6381) 
v_mul_f32 v21, v0, v11 // (op 6381) 


v_add_f32 v10, v18, v14 // (op 6377) 
v_add_f32 v11, v19, v15 // (op 6377) 
v_add_f32 v12, v20, v16 // (op 6377) 
v_add_f32 v13, v21, v17 // (op 6377) 


v_lshrrev_b32 v14, 16, v10 // (op 6378) 
v_lshrrev_b32 v15, 16, v11 // (op 6378) 
v_lshrrev_b32 v16, 16, v12 // (op 6378) 
v_lshrrev_b32 v17, 16, v13 // (op 6378) 


v_pack_b32_f16 v10, v14, v15 // (op 6379) 
v_pack_b32_f16 v11, v16, v17 // (op 6379) 

s_lshl_b32 s0, s4, 7 // (op 6379) 
v_add_u32 v3, v9, s0 // (op 6379) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6379) Store value





s_lshl_b32 s0, s2, 5 // (op 6394) 
s_mul_i32 s1, 5, s0 // (op 6394) 

v_add_u32 v3, v8, s1 // (op 6394) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6394) Load value

v_accvgpr_read v10, a20 // (op 6390) 
v_accvgpr_read v11, a21 // (op 6390) 
v_accvgpr_read v12, a22 // (op 6390) 
v_accvgpr_read v13, a23 // (op 6390) 
v_mul_f32 v14, v7, v10 // (op 6390) 
v_mul_f32 v15, v7, v11 // (op 6390) 
v_mul_f32 v16, v7, v12 // (op 6390) 
v_mul_f32 v17, v7, v13 // (op 6390) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6395) 
v_lshrrev_b32 v11, 16, v4 // (op 6395) 
v_lshlrev_b32 v10, 16, v10 // (op 6395) 
v_lshlrev_b32 v11, 16, v11 // (op 6395) 
v_mul_f32 v18, v0, v10 // (op 6395) 
v_mul_f32 v19, v0, v11 // (op 6395) 
v_and_b32 v10, 65535, v5 // (op 6395) 
v_lshrrev_b32 v11, 16, v5 // (op 6395) 
v_lshlrev_b32 v10, 16, v10 // (op 6395) 
v_lshlrev_b32 v11, 16, v11 // (op 6395) 
v_mul_f32 v20, v0, v10 // (op 6395) 
v_mul_f32 v21, v0, v11 // (op 6395) 


v_add_f32 v10, v18, v14 // (op 6391) 
v_add_f32 v11, v19, v15 // (op 6391) 
v_add_f32 v12, v20, v16 // (op 6391) 
v_add_f32 v13, v21, v17 // (op 6391) 


v_lshrrev_b32 v14, 16, v10 // (op 6392) 
v_lshrrev_b32 v15, 16, v11 // (op 6392) 
v_lshrrev_b32 v16, 16, v12 // (op 6392) 
v_lshrrev_b32 v17, 16, v13 // (op 6392) 


v_pack_b32_f16 v10, v14, v15 // (op 6393) 
v_pack_b32_f16 v11, v16, v17 // (op 6393) 

s_lshl_b32 s0, s4, 5 // (op 6393) 
s_mul_i32 s1, 5, s0 // (op 6393) 

v_add_u32 v3, v9, s1 // (op 6393) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6393) Store value





s_lshl_b32 s0, s2, 5 // (op 6408) 
s_mul_i32 s1, 6, s0 // (op 6408) 

v_add_u32 v3, v8, s1 // (op 6408) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6408) Load value

v_accvgpr_read v10, a24 // (op 6404) 
v_accvgpr_read v11, a25 // (op 6404) 
v_accvgpr_read v12, a26 // (op 6404) 
v_accvgpr_read v13, a27 // (op 6404) 
v_mul_f32 v14, v7, v10 // (op 6404) 
v_mul_f32 v15, v7, v11 // (op 6404) 
v_mul_f32 v16, v7, v12 // (op 6404) 
v_mul_f32 v17, v7, v13 // (op 6404) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6409) 
v_lshrrev_b32 v11, 16, v4 // (op 6409) 
v_lshlrev_b32 v10, 16, v10 // (op 6409) 
v_lshlrev_b32 v11, 16, v11 // (op 6409) 
v_mul_f32 v18, v0, v10 // (op 6409) 
v_mul_f32 v19, v0, v11 // (op 6409) 
v_and_b32 v10, 65535, v5 // (op 6409) 
v_lshrrev_b32 v11, 16, v5 // (op 6409) 
v_lshlrev_b32 v10, 16, v10 // (op 6409) 
v_lshlrev_b32 v11, 16, v11 // (op 6409) 
v_mul_f32 v20, v0, v10 // (op 6409) 
v_mul_f32 v21, v0, v11 // (op 6409) 


v_add_f32 v10, v18, v14 // (op 6405) 
v_add_f32 v11, v19, v15 // (op 6405) 
v_add_f32 v12, v20, v16 // (op 6405) 
v_add_f32 v13, v21, v17 // (op 6405) 


v_lshrrev_b32 v14, 16, v10 // (op 6406) 
v_lshrrev_b32 v15, 16, v11 // (op 6406) 
v_lshrrev_b32 v16, 16, v12 // (op 6406) 
v_lshrrev_b32 v17, 16, v13 // (op 6406) 


v_pack_b32_f16 v10, v14, v15 // (op 6407) 
v_pack_b32_f16 v11, v16, v17 // (op 6407) 

s_lshl_b32 s0, s4, 5 // (op 6407) 
s_mul_i32 s1, 6, s0 // (op 6407) 

v_add_u32 v3, v9, s1 // (op 6407) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6407) Store value





s_lshl_b32 s0, s2, 5 // (op 6422) 
s_mul_i32 s1, 7, s0 // (op 6422) 

v_add_u32 v3, v8, s1 // (op 6422) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 6422) Load value

v_accvgpr_read v10, a28 // (op 6418) 
v_accvgpr_read v11, a29 // (op 6418) 
v_accvgpr_read v12, a30 // (op 6418) 
v_accvgpr_read v13, a31 // (op 6418) 
v_mul_f32 v14, v7, v10 // (op 6418) 
v_mul_f32 v15, v7, v11 // (op 6418) 
v_mul_f32 v16, v7, v12 // (op 6418) 
v_mul_f32 v17, v7, v13 // (op 6418) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 6423) 
v_lshrrev_b32 v11, 16, v4 // (op 6423) 
v_lshlrev_b32 v10, 16, v10 // (op 6423) 
v_lshlrev_b32 v11, 16, v11 // (op 6423) 
v_mul_f32 v18, v0, v10 // (op 6423) 
v_mul_f32 v19, v0, v11 // (op 6423) 
v_and_b32 v10, 65535, v5 // (op 6423) 
v_lshrrev_b32 v11, 16, v5 // (op 6423) 
v_lshlrev_b32 v10, 16, v10 // (op 6423) 
v_lshlrev_b32 v11, 16, v11 // (op 6423) 
v_mul_f32 v20, v0, v10 // (op 6423) 
v_mul_f32 v21, v0, v11 // (op 6423) 


v_add_f32 v10, v18, v14 // (op 6419) 
v_add_f32 v11, v19, v15 // (op 6419) 
v_add_f32 v12, v20, v16 // (op 6419) 
v_add_f32 v13, v21, v17 // (op 6419) 


v_lshrrev_b32 v14, 16, v10 // (op 6420) 
v_lshrrev_b32 v15, 16, v11 // (op 6420) 
v_lshrrev_b32 v16, 16, v12 // (op 6420) 
v_lshrrev_b32 v17, 16, v13 // (op 6420) 


v_pack_b32_f16 v10, v14, v15 // (op 6421) 
v_pack_b32_f16 v11, v16, v17 // (op 6421) 

s_lshl_b32 s0, s4, 5 // (op 6421) 
s_mul_i32 s1, 7, s0 // (op 6421) 

v_add_u32 v3, v9, s1 // (op 6421) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 6421) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalBottom_0_beta_116:








































s_endpgm  // End of RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
.LRR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_end:
.size RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2, .LRR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_end-RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
  .amdhsa_next_free_vgpr 296
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 81920 // lds bytes
  .amdhsa_accum_offset 168
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
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    MXBlockA: 32
    MXBlockB: 32
  CustomKernel:
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum } ]
    macrotile: [128, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
    .symbol: RR_GEMM_TN_BF8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 81920
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 168
    .agpr_count: 128
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
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
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .value_kind: by_value
...

.end_amdgpu_metadata

