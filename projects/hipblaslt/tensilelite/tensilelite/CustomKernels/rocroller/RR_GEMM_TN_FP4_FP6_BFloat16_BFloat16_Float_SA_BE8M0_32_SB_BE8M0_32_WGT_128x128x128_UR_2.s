

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2,@function
RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 7029) 
v_and_b32 v3, 15, v1 // (op 7029) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 7029) 


v_and_b32 v0, 63, v1 // (op 7029) 
v_lshrrev_b32 v3, 4, v0 // (op 7029) 

v_lshlrev_b32 v0, 5, v3 // (op 7029) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 7029) 


v_lshlrev_b32 v0, 2, v3 // (op 7029) 

v_lshrrev_b32 v3, 3, v0 // (op 7029) 

v_lshrrev_b32 v0, 6, v1 // (op 7087) 
v_and_b32 v4, 1, v0 // (op 7087) 

v_and_b32 v0, 15, v1 // (op 7087) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 7087) 


v_and_b32 v0, 63, v1 // (op 7087) 
v_lshrrev_b32 v4, 4, v0 // (op 7087) 

v_lshlrev_b32 v0, 5, v4 // (op 7087) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 7087) 


v_mul_lo_u32 v0, v4, 6 // (op 7087) 

v_lshrrev_b32 v4, 3, v0 // (op 7087) 

v_lshrrev_b32 v0, 7, v1 // (op 7154) 
v_and_b32 v5, 15, v1 // (op 7154) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 7154) 


v_and_b32 v0, 63, v1 // (op 7154) 
v_lshrrev_b32 v5, 4, v0 // (op 7154) 

v_lshlrev_b32 v0, 5, v5 // (op 7154) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 7154) 


v_lshlrev_b32 v0, 2, v5 // (op 7154) 

v_lshrrev_b32 v5, 3, v0 // (op 7154) 

v_lshrrev_b32 v0, 6, v1 // (op 7183) 
v_and_b32 v6, 1, v0 // (op 7183) 

v_and_b32 v0, 15, v1 // (op 7183) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 7183) 


v_and_b32 v0, 63, v1 // (op 7183) 
v_lshrrev_b32 v6, 4, v0 // (op 7183) 

v_lshlrev_b32 v0, 5, v6 // (op 7183) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 7183) 


v_mul_lo_u32 v0, v6, 6 // (op 7183) 

v_lshrrev_b32 v6, 3, v0 // (op 7183) 

v_lshrrev_b32 v0, 7, v1 // (op 6968) 
v_and_b32 v7, 63, v1 // (op 6968) 
v_lshrrev_b32 v8, 4, v7 // (op 6968) 
v_lshlrev_b32 v9, 2, v8 // (op 6968) 

v_lshl_add_u32 v8, v0, 6, v9 // (op 6968) 


v_lshl_add_u32 v0, s16, 7, v8 // (op 6968) 

v_lshrrev_b32 v8, 2, v7 // (op 6968) 

v_and_b32 v7, 3, v8 // (op 6968) 

v_and_b32 v8, 3, v1 // (op 6968) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 6968) 


v_lshrrev_b32 v7, 6, v1 // (op 6968) 
v_and_b32 v8, 1, v7 // (op 6968) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 6968) 


v_lshl_add_u32 v8, s17, 7, v7 // (op 6968) 

v_mul_lo_u32 v7, v8, s2 // (op 6968) 

v_add_lshl_u32 v8, v0, v7, 1 // (op 6968) 


v_lshrrev_b32 v0, 7, v1 // (op 7108) 
v_and_b32 v7, 63, v1 // (op 7108) 
v_lshrrev_b32 v9, 4, v7 // (op 7108) 
s_mul_i32 s1, s2, s9 // (op 6970) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 6970) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 6970) least significant
s_mul_hi_u32 s19, s2, s8 // (op 6970) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6970) most significant: sum
s_add_u32 s1, s1, s19 // (op 6970) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 6970) 

v_lshlrev_b32 v10, 2, v9 // (op 7108) 

v_lshl_add_u32 v9, v0, 6, v10 // (op 7108) 


v_lshl_add_u32 v0, s16, 7, v9 // (op 7108) 

v_lshrrev_b32 v9, 2, v7 // (op 7108) 

v_and_b32 v7, 3, v9 // (op 7108) 

v_and_b32 v9, 3, v1 // (op 7108) 
v_lshl_add_u32 v10, v7, 2, v9 // (op 7108) 


v_lshrrev_b32 v7, 6, v1 // (op 7108) 
v_and_b32 v9, 1, v7 // (op 7108) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 7108) 


v_lshl_add_u32 v9, s17, 7, v7 // (op 7108) 

v_mul_lo_u32 v7, v9, s4 // (op 7108) 

v_add_lshl_u32 v9, v0, v7, 1 // (op 7108) 


s_mov_b32 s40, s6 // (op 6970) reg expression
s_mov_b32 s41, s7 // (op 6970) reg expression
s_mov_b32 s42, s18 // (op 6970) reg expression
s_mov_b32 s43, 131072 // (op 6970) call()

s_mul_i32 s1, s4, s9 // (op 7110) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 7110) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 7110) least significant
s_mul_hi_u32 s7, s4, s8 // (op 7110) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7110) most significant: sum
s_add_u32 s1, s1, s7 // (op 7110) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 7110) 

s_mov_b32 s44, s10 // (op 7110) reg expression
s_mov_b32 s45, s11 // (op 7110) reg expression
s_mov_b32 s46, s6 // (op 7110) reg expression
s_mov_b32 s47, 131072 // (op 7110) call()


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
v_accvgpr_write a16, 0.00000 // (op 777) call()
v_accvgpr_write a17, 0.00000 // (op 777) call()
v_accvgpr_write a18, 0.00000 // (op 777) call()
v_accvgpr_write a19, 0.00000 // (op 777) call()
v_accvgpr_write a20, 0.00000 // (op 1038) call()
v_accvgpr_write a21, 0.00000 // (op 1038) call()
v_accvgpr_write a22, 0.00000 // (op 1038) call()
v_accvgpr_write a23, 0.00000 // (op 1038) call()
v_accvgpr_write a24, 0.00000 // (op 1039) call()
v_accvgpr_write a25, 0.00000 // (op 1039) call()
v_accvgpr_write a26, 0.00000 // (op 1039) call()
v_accvgpr_write a27, 0.00000 // (op 1039) call()
v_accvgpr_write a28, 0.00000 // (op 1040) call()
v_accvgpr_write a29, 0.00000 // (op 1040) call()
v_accvgpr_write a30, 0.00000 // (op 1040) call()
v_accvgpr_write a31, 0.00000 // (op 1040) call()
v_accvgpr_write a32, 0.00000 // (op 1387) call()
v_accvgpr_write a33, 0.00000 // (op 1387) call()
v_accvgpr_write a34, 0.00000 // (op 1387) call()
v_accvgpr_write a35, 0.00000 // (op 1387) call()
v_accvgpr_write a36, 0.00000 // (op 1648) call()
v_accvgpr_write a37, 0.00000 // (op 1648) call()
v_accvgpr_write a38, 0.00000 // (op 1648) call()
v_accvgpr_write a39, 0.00000 // (op 1648) call()
v_accvgpr_write a40, 0.00000 // (op 1649) call()
v_accvgpr_write a41, 0.00000 // (op 1649) call()
v_accvgpr_write a42, 0.00000 // (op 1649) call()
v_accvgpr_write a43, 0.00000 // (op 1649) call()
v_accvgpr_write a44, 0.00000 // (op 1650) call()
v_accvgpr_write a45, 0.00000 // (op 1650) call()
v_accvgpr_write a46, 0.00000 // (op 1650) call()
v_accvgpr_write a47, 0.00000 // (op 1650) call()
v_accvgpr_write a48, 0.00000 // (op 1997) call()
v_accvgpr_write a49, 0.00000 // (op 1997) call()
v_accvgpr_write a50, 0.00000 // (op 1997) call()
v_accvgpr_write a51, 0.00000 // (op 1997) call()
v_accvgpr_write a52, 0.00000 // (op 2258) call()
v_accvgpr_write a53, 0.00000 // (op 2258) call()
v_accvgpr_write a54, 0.00000 // (op 2258) call()
v_accvgpr_write a55, 0.00000 // (op 2258) call()
v_accvgpr_write a56, 0.00000 // (op 2259) call()
v_accvgpr_write a57, 0.00000 // (op 2259) call()
v_accvgpr_write a58, 0.00000 // (op 2259) call()
v_accvgpr_write a59, 0.00000 // (op 2259) call()
v_accvgpr_write a60, 0.00000 // (op 2260) call()
v_accvgpr_write a61, 0.00000 // (op 2260) call()
v_accvgpr_write a62, 0.00000 // (op 2260) call()
v_accvgpr_write a63, 0.00000 // (op 2260) call()
v_lshrrev_b32 v10, 2, v1 // (op 6858) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 6858) 

v_and_b32 v10, -4, v1 // (op 6858) 
v_sub_u32 v12, v1, v10 // (op 6858) 

v_lshlrev_b32 v10, 5, v12 // (op 6858) 

v_mov_b32 v13, 0 // (op 6858) convert
v_mov_b32 v12, v10 // (op 6858) convert

v_mul_lo_u32 v15, v11, s13 // (op 6858) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 6858) most significant: high of low * low
v_mul_lo_u32 v14, v11, s12 // (op 6858) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 6858) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 6858) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 6858) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 6858) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 6858) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 6860) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6860) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6860) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6860) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6860) most significant: sum
s_add_u32 s1, s1, s7 // (op 6860) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6860) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6860) 

s_mov_b32 s48, s20 // (op 6860) reg expression
s_mov_b32 s49, s21 // (op 6860) reg expression
s_mov_b32 s50, s0 // (op 6860) reg expression
s_mov_b32 s51, 131072 // (op 6860) call()

v_lshrrev_b32 v10, 2, v1 // (op 6877) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 6877) 

v_and_b32 v10, -4, v1 // (op 6877) 
v_sub_u32 v14, v1, v10 // (op 6877) 

v_lshlrev_b32 v10, 5, v14 // (op 6877) 

v_mov_b32 v15, 0 // (op 6877) convert
v_mov_b32 v14, v10 // (op 6877) convert

v_mul_lo_u32 v17, v11, s13 // (op 6877) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 6877) most significant: high of low * low
v_mul_lo_u32 v16, v11, s12 // (op 6877) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 6877) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 6877) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 6877) most significant half



v_lshlrev_b64 v[10:11], 2, v[16:17] // (op 6877) 

v_lshrrev_b64 v[14:15], 3, v[10:11] // (op 6877) 

s_mul_i32 s1, s12, s23 // (op 6879) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6879) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6879) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6879) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6879) most significant: sum
s_add_u32 s1, s1, s7 // (op 6879) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6879) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6879) 

s_mov_b32 s48, s20 // (op 6879) reg expression
s_mov_b32 s49, s21 // (op 6879) reg expression
s_mov_b32 s50, s0 // (op 6879) reg expression
s_mov_b32 s51, 131072 // (op 6879) call()

v_and_b32 v10, -2, v1 // (op 6922) 
v_sub_u32 v11, v1, v10 // (op 6922) 

v_lshlrev_b32 v10, 6, v11 // (op 6922) 

v_mov_b32 v17, 0 // (op 6922) convert
v_mov_b32 v16, v10 // (op 6922) convert

v_lshrrev_b32 v10, 1, v1 // (op 6922) 
v_lshl_add_u32 v11, s17, 7, v10 // (op 6922) 

v_mul_lo_u32 v19, v11, s25 // (op 6922) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 6922) most significant: high of low * low
v_mul_lo_u32 v18, v11, s24 // (op 6922) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 6922) most significant: sum


v_add_co_u32 v10, s[0:1], v16, v18 // (op 6922) least significant half
v_addc_co_u32 v11, s[0:1], v17, v19, s[0:1] // (op 6922) most significant half



v_mul_lo_u32 v17, v11, 6 // (op 6922) most significant: low of high * low
v_mul_hi_u32 v18, v10, 6 // (op 6922) most significant: high of low * low
v_mul_lo_u32 v16, v10, 6 // (op 6922) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 6922) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[16:17] // (op 6922) 

v_mov_b32 v16, v10 // (op 6922) convert

s_mul_i32 s1, s9, 6 // (op 6924) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 6924) least significant
s_mul_hi_u32 s6, s8, 6 // (op 6924) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6924) most significant: sum

s_mul_i32 s7, s24, s1 // (op 6924) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 6924) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 6924) least significant
s_mul_hi_u32 s11, s24, s0 // (op 6924) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 6924) most significant: sum
s_add_u32 s7, s7, s11 // (op 6924) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 6924) 

s_mov_b32 s52, s26 // (op 6924) reg expression
s_mov_b32 s53, s27 // (op 6924) reg expression
s_mov_b32 s54, s0 // (op 6924) reg expression
s_mov_b32 s55, 131072 // (op 6924) call()

v_and_b32 v10, -2, v1 // (op 6941) 
v_sub_u32 v11, v1, v10 // (op 6941) 

v_lshlrev_b32 v10, 6, v11 // (op 6941) 

v_mov_b32 v19, 0 // (op 6941) convert
v_mov_b32 v18, v10 // (op 6941) convert

v_lshrrev_b32 v17, 1, v1 // (op 6941) 
v_lshl_add_u32 v10, s17, 7, v17 // (op 6941) 

v_mul_lo_u32 v21, v10, s25 // (op 6941) most significant: low of low * high
v_mul_hi_u32 v11, v10, s24 // (op 6941) most significant: high of low * low
v_mul_lo_u32 v20, v10, s24 // (op 6941) least significant: low of low * low
v_add_u32 v21, v21, v11 // (op 6941) most significant: sum


v_add_co_u32 v10, s[0:1], v18, v20 // (op 6941) least significant half
v_addc_co_u32 v11, s[0:1], v19, v21, s[0:1] // (op 6941) most significant half



v_mul_lo_u32 v19, v11, 6 // (op 6941) most significant: low of high * low
v_mul_hi_u32 v17, v10, 6 // (op 6941) most significant: high of low * low
v_mul_lo_u32 v18, v10, 6 // (op 6941) least significant: low of low * low
v_add_u32 v19, v19, v17 // (op 6941) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[18:19] // (op 6941) 

v_mov_b32 v17, v10 // (op 6941) convert

s_mul_i32 s1, s9, 6 // (op 6943) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 6943) least significant
s_mul_hi_u32 s6, s8, 6 // (op 6943) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6943) most significant: sum

s_mul_i32 s7, s24, s1 // (op 6943) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 6943) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 6943) least significant
s_mul_hi_u32 s11, s24, s0 // (op 6943) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 6943) most significant: sum
s_add_u32 s7, s7, s11 // (op 6943) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 6943) 

s_mov_b32 s52, s26 // (op 6943) reg expression
s_mov_b32 s53, s27 // (op 6943) reg expression
s_mov_b32 s54, s0 // (op 6943) reg expression
s_mov_b32 s55, 131072 // (op 6943) call()

v_and_b32 v10, 63, v1 // (op 7205) 
v_lshrrev_b32 v11, 4, v10 // (op 7205) 
v_and_b32 v18, 3, v11 // (op 7205) 

v_and_b32 v11, 15, v1 // (op 7205) 
v_lshl_add_u32 v19, v18, 4, v11 // (op 7205) 


v_lshrrev_b32 v11, 7, v1 // (op 7205) 
v_lshl_add_u32 v18, v11, 6, v19 // (op 7205) 


v_lshl_add_u32 v11, s16, 7, v18 // (op 7205) 

v_lshrrev_b32 v18, 6, v10 // (op 7205) 

v_lshlrev_b32 v10, 2, v18 // (op 7205) 

v_mul_lo_u32 v18, v11, s28 // (op 7205) 
v_add_u32 v18, v18, v10 // (op 7205) 


s_mul_i32 s1, s28, s23 // (op 7207) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 7207) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 7207) least significant
s_mul_hi_u32 s7, s28, s22 // (op 7207) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7207) most significant: sum
s_add_u32 s1, s1, s7 // (op 7207) most significant: sum


s_mov_b32 s56, s30 // (op 7207) reg expression
s_mov_b32 s57, s31 // (op 7207) reg expression
s_mov_b32 s58, s0 // (op 7207) reg expression
s_mov_b32 s59, 131072 // (op 7207) call()

v_and_b32 v10, 63, v1 // (op 7226) 
v_lshrrev_b32 v11, 4, v10 // (op 7226) 
v_and_b32 v19, 3, v11 // (op 7226) 

v_and_b32 v11, 15, v1 // (op 7226) 
v_lshl_add_u32 v20, v19, 4, v11 // (op 7226) 


v_lshrrev_b32 v11, 7, v1 // (op 7226) 
v_lshl_add_u32 v19, v11, 6, v20 // (op 7226) 


v_lshl_add_u32 v11, s16, 7, v19 // (op 7226) 

v_lshrrev_b32 v19, 6, v10 // (op 7226) 

v_lshlrev_b32 v10, 2, v19 // (op 7226) 

v_mul_lo_u32 v19, v11, s28 // (op 7226) 
v_add_u32 v19, v19, v10 // (op 7226) 


s_mul_i32 s1, s28, s23 // (op 7228) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 7228) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 7228) least significant
s_mul_hi_u32 s7, s28, s22 // (op 7228) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7228) most significant: sum
s_add_u32 s1, s1, s7 // (op 7228) most significant: sum


s_mov_b32 s56, s30 // (op 7228) reg expression
s_mov_b32 s57, s31 // (op 7228) reg expression
s_mov_b32 s58, s0 // (op 7228) reg expression
s_mov_b32 s59, 131072 // (op 7228) call()

v_and_b32 v10, 63, v1 // (op 7257) 
v_lshrrev_b32 v11, 4, v10 // (op 7257) 
v_and_b32 v20, 3, v11 // (op 7257) 

v_and_b32 v11, 15, v1 // (op 7257) 
v_lshl_add_u32 v21, v20, 4, v11 // (op 7257) 


v_lshrrev_b32 v11, 6, v1 // (op 7257) 
v_and_b32 v20, 1, v11 // (op 7257) 

v_lshl_add_u32 v11, v20, 6, v21 // (op 7257) 


v_lshl_add_u32 v20, s17, 7, v11 // (op 7257) 

v_mul_lo_u32 v11, v20, s32 // (op 7257) 

v_lshrrev_b32 v20, 6, v10 // (op 7257) 

v_lshl_add_u32 v10, v20, 2, v11 // (op 7257) 


s_mul_i32 s1, s32, s9 // (op 7259) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 7259) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 7259) least significant
s_mul_hi_u32 s7, s32, s8 // (op 7259) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7259) most significant: sum
s_add_u32 s1, s1, s7 // (op 7259) most significant: sum


s_mov_b32 s60, s34 // (op 7259) reg expression
s_mov_b32 s61, s35 // (op 7259) reg expression
s_mov_b32 s62, s0 // (op 7259) reg expression
s_mov_b32 s63, 131072 // (op 7259) call()

v_and_b32 v11, 63, v1 // (op 7278) 
v_lshrrev_b32 v20, 4, v11 // (op 7278) 
v_and_b32 v21, 3, v20 // (op 7278) 

v_and_b32 v20, 15, v1 // (op 7278) 
v_lshl_add_u32 v22, v21, 4, v20 // (op 7278) 


v_lshrrev_b32 v20, 6, v1 // (op 7278) 
v_and_b32 v21, 1, v20 // (op 7278) 

v_lshl_add_u32 v20, v21, 6, v22 // (op 7278) 


v_lshl_add_u32 v21, s17, 7, v20 // (op 7278) 

v_mul_lo_u32 v20, v21, s32 // (op 7278) 

v_lshrrev_b32 v21, 6, v11 // (op 7278) 

v_lshl_add_u32 v11, v21, 2, v20 // (op 7278) 


s_mul_i32 s1, s32, s9 // (op 7280) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 7280) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 7280) least significant
s_mul_hi_u32 s7, s32, s8 // (op 7280) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 7280) most significant: sum
s_add_u32 s1, s1, s7 // (op 7280) most significant: sum


s_mov_b32 s60, s34 // (op 7280) reg expression
s_mov_b32 s61, s35 // (op 7280) reg expression
s_mov_b32 s62, s0 // (op 7280) reg expression
s_mov_b32 s63, 131072 // (op 7280) call()




buffer_load_dwordx4 v[20:23], v16, s[52:55], 0 offen // (op 6473) Load value
buffer_load_dwordx2 v[24:25], v16, s[52:55], 0 offen offset:16 // (op 6473) Load value
buffer_load_dwordx4 v[26:29], v16, s[52:55], 0 offen offset:24 // (op 6473) Load value
buffer_load_dwordx2 v[30:31], v16, s[52:55], 0 offen offset:40 // (op 6473) Load value
v_and_b32 v32, -4, v1 // (op 7012) 
v_sub_u32 v33, v1, v32 // (op 7012) 

v_lshlrev_b32 v32, 5, v33 // (op 7012) 

v_lshrrev_b32 v33, 2, v1 // (op 7012) 
v_lshl_add_u32 v34, v33, 7, v32 // (op 7012) 


v_lshlrev_b32 v32, 2, v34 // (op 7012) 

v_lshrrev_b32 v33, 3, v32 // (op 7012) 

s_nop 0
v_readfirstlane_b32 s0, v33 // (op 7012) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 6799) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 6799) 

v_lshrrev_b64 v[32:33], 3, s[10:11] // (op 6799) 


v_mov_b64 v[34:35], v[12:13] // (op 6799) 
s_mov_b32 m0, s0 // (op 6799) reg expression
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 6799) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v32 // (op 6799) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 6799) most significant half

s_add_u32 m0, m0, 4096 // (op 6799) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 6799) Load value direct to lds





v_mov_b32 v32, v10 // (op 76) reg expression

buffer_load_dwordx2 v[34:35], v32, s[60:63], 0 offen // (op 76) Load value




v_mov_b32 v10, v18 // (op 72) reg expression

buffer_load_dwordx2 v[32:33], v10, s[56:59], 0 offen // (op 72) Load value





buffer_load_dwordx4 v[36:39], v16, s[52:55], 0 offen offset:96 // (op 6479) Load value
buffer_load_dwordx2 v[40:41], v16, s[52:55], 0 offen offset:112 // (op 6479) Load value
buffer_load_dwordx4 v[42:45], v16, s[52:55], 0 offen offset:120 // (op 6479) Load value
buffer_load_dwordx2 v[46:47], v16, s[52:55], 0 offen offset:136 // (op 6479) Load value

v_and_b32 v10, -4, v1 // (op 7137) 
v_sub_u32 v16, v1, v10 // (op 7137) 

v_lshlrev_b32 v10, 5, v16 // (op 7137) 

v_lshrrev_b32 v16, 2, v1 // (op 7137) 
v_lshl_add_u32 v18, v16, 7, v10 // (op 7137) 


v_lshlrev_b32 v10, 2, v18 // (op 7137) 

v_lshrrev_b32 v16, 3, v10 // (op 7137) 

s_nop 0
v_readfirstlane_b32 s0, v16 // (op 7137) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 6804) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 6804) 

v_lshrrev_b64 v[48:49], 3, s[10:11] // (op 6804) 


v_mov_b64 v[50:51], v[12:13] // (op 6804) 
s_add_u32 s1, 8192, s0 // (op 6804) 
s_mov_b32 m0, s1 // (op 6804) reg expression
buffer_load_dwordx4 v50, s[48:51], 64 offen offset: 0 lds // (op 6804) Load value direct to lds
v_add_co_u32 v50, s[6:7], v50, v48 // (op 6804) least significant half
v_addc_co_u32 v51, s[6:7], 0, v51, s[6:7] // (op 6804) most significant half

s_add_u32 m0, m0, 4096 // (op 6804) 
buffer_load_dwordx4 v50, s[48:51], 64 offen offset: 0 lds // (op 6804) Load value direct to lds





v_and_b32 v10, -2, v1 // (op 7070) 
v_sub_u32 v16, v1, v10 // (op 7070) 

v_lshlrev_b32 v10, 6, v16 // (op 7070) 

v_lshrrev_b32 v16, 1, v1 // (op 7070) 
v_lshl_add_u32 v18, v16, 7, v10 // (op 7070) 


v_mul_lo_u32 v10, v18, 6 // (op 7070) 

v_lshrrev_b32 v16, 3, v10 // (op 7070) 




s_waitcnt vmcnt(13)
ds_write_b128 v16, v[20:23] offset:16384 // (op 6484) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v16, v[24:25] offset:16400 // (op 6484) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v16, v[26:29] offset:16408 // (op 6484) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[30:31] offset:16424 // (op 6484) Store local data 

v_and_b32 v10, -4, v1 // (op 6997) 
v_sub_u32 v16, v1, v10 // (op 6997) 

v_lshlrev_b32 v10, 5, v16 // (op 6997) 

v_lshrrev_b32 v16, 2, v1 // (op 6997) 
v_lshl_add_u32 v18, v16, 7, v10 // (op 6997) 


v_lshlrev_b32 v10, 2, v18 // (op 6997) 

v_lshrrev_b32 v16, 3, v10 // (op 6997) 

s_nop 0
v_readfirstlane_b32 s0, v16 // (op 6997) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 7054) 
v_sub_u32 v16, v1, v10 // (op 7054) 

v_lshlrev_b32 v10, 6, v16 // (op 7054) 

v_lshrrev_b32 v16, 1, v1 // (op 7054) 
v_lshl_add_u32 v18, v16, 7, v10 // (op 7054) 


v_mul_lo_u32 v10, v18, 6 // (op 7054) 

v_lshrrev_b32 v16, 3, v10 // (op 7054) 

v_and_b32 v10, -4, v1 // (op 7122) 
v_sub_u32 v18, v1, v10 // (op 7122) 

v_lshlrev_b32 v10, 5, v18 // (op 7122) 

v_lshrrev_b32 v18, 2, v1 // (op 7122) 
v_lshl_add_u32 v12, v18, 7, v10 // (op 7122) 


v_lshlrev_b32 v10, 2, v12 // (op 7122) 

v_lshrrev_b32 v18, 3, v10 // (op 7122) 

s_nop 0
v_readfirstlane_b32 s1, v18 // (op 7122) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 7166) 
v_sub_u32 v18, v1, v10 // (op 7166) 

v_lshlrev_b32 v10, 6, v18 // (op 7166) 

v_lshrrev_b32 v18, 1, v1 // (op 7166) 
v_lshl_add_u32 v12, v18, 7, v10 // (op 7166) 


v_mul_lo_u32 v10, v12, 6 // (op 7166) 

v_lshrrev_b32 v18, 3, v10 // (op 7166) 

s_mov_b32 s6, 0 // (op 2080) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 2079) 
s_mov_b32 s7, s10 // (op 2079) convert

s_and_b32 s10, s7, 4294967294 // (op 2079) 

s_cmp_lt_u32 s6, s10 // (op 2079) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2079 // (op 2079) Condition: Top (jump to RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2079 if false)
RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2079:
s_add_u32 s6, s6, 2 // (op 2081) 
s_waitcnt vmcnt(8)
s_barrier  // (op 7422) 



ds_read_b128 v[48:51], v3 // (op 62) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 7438) 



ds_read_b128 v[52:55], v4 offset:16384 // (op 67) Load local data 
ds_read_b64 v[56:57], v4 offset:16400 // (op 67) Load local data 



ds_read_b128 v[58:61], v4 offset:17920 // (op 312) Load local data 


v_mov_b32 v10, v17 // (op 10) 

buffer_load_dwordx4 v[20:23], v10, s[52:55], 0 offen offset:192 // (op 10) Load value
s_waitcnt vmcnt(8)
v_mov_b32 v12, v34 // (op 6739) reg expression
v_mov_b32 v13, v35 // (op 6739) reg expression
s_waitcnt vmcnt(7)
v_mov_b32 v64, v32 // (op 6752) reg expression
v_mov_b32 v65, v33 // (op 6752) reg expression
ds_read_b64 v[62:63], v4 offset:17936 // (op 312) Load local data 



v_bfe_u32 v66, v64, 0, 8 // (op 6114) 
v_bfe_u32 v67, v64, 8, 8 // (op 6114) 
v_bfe_u32 v68, v64, 16, 8 // (op 6114) 
v_bfe_u32 v69, v64, 24, 8 // (op 6114) 
v_permlane16_swap_b32 v66, v67 // (op 6114) 
ds_read_b128 v[70:73], v4 offset:19456 // (op 448) Load local data 
v_permlane16_swap_b32 v68, v69 // (op 6114) 
s_nop 1
v_permlane32_swap_b32 v66, v68 // (op 6114) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 6114) 
ds_read_b64 v[74:75], v4 offset:19472 // (op 448) Load local data 



v_bfe_u32 v76, v12, 0, 8 // (op 6177) 
v_bfe_u32 v77, v12, 8, 8 // (op 6177) 
v_bfe_u32 v78, v12, 16, 8 // (op 6177) 
v_bfe_u32 v79, v12, 24, 8 // (op 6177) 
v_permlane16_swap_b32 v76, v77 // (op 6177) 
s_nop 0
v_permlane16_swap_b32 v78, v79 // (op 6177) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v76, v78 // (op 6177) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v77, v79 // (op 6177) 
ds_read_b128 v[80:83], v4 offset:20992 // (op 584) Load local data 
ds_read_b64 v[84:85], v4 offset:21008 // (op 584) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[48:51], v[52:57], a[0:3], v66, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 78) 
s_barrier  // (op 7431) 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:51], v[58:63], a[4:7], v66, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 315) 
ds_read_b128 v[86:89], v3 offset:1024 // (op 889) Load local data 



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[48:51], v[70:75], a[8:11], v66, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 451) 
ds_read_b128 v[90:93], v3 offset:2048 // (op 1499) Load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[48:51], v[80:85], a[12:15], v66, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 587) 
ds_read_b128 v[94:97], v3 offset:3072 // (op 2109) Load local data 

buffer_load_dwordx2 v[24:25], v10, s[52:55], 0 offen offset:208 // (op 10) Load value
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[86:89], v[58:63], a[20:23], v67, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 969) 
buffer_load_dwordx4 v[26:29], v10, s[52:55], 0 offen offset:216 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[86:89], v[70:75], a[24:27], v67, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 931) 
buffer_load_dwordx2 v[30:31], v10, s[52:55], 0 offen offset:232 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[86:89], v[80:85], a[28:31], v67, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 893) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7425) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[86:89], v[52:57], a[16:19], v67, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1007) 
s_lshl_b64 s[10:11], s[12:13], 6 // (op 6783) 

s_lshl_b64 s[14:15], s[10:11], 2 // (op 6783) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:93], v[52:57], a[32:35], v68, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1617) 
v_lshrrev_b64 v[48:49], 3, s[14:15] // (op 6783) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:93], v[58:63], a[36:39], v68, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1579) 
v_mov_b64 v[50:51], v[14:15] // (op 6783) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[70:75], a[40:43], v68, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1541) 
v_mov_b64 v[86:87], v[50:51] // (op 6783) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:93], v[80:85], a[44:47], v68, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1503) 
s_mov_b32 m0, s0 // (op 6783) reg expression

s_mov_b32 s7, 128 // (op 6783) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[94:97], v[52:57], a[48:51], v69, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2227) 
buffer_load_dwordx4 v86, s[48:51], s7 offen offset: 0 lds // (op 6783) Load value direct to lds

v_add_co_u32 v86, s[10:11], v86, v48 // (op 6783) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[94:97], v[58:63], a[52:55], v69, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2189) 
v_addc_co_u32 v87, s[10:11], 0, v87, s[10:11] // (op 6783) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[94:97], v[70:75], a[56:59], v69, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2151) 
s_add_u32 m0, m0, 4096 // (op 6783) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[94:97], v[80:85], a[60:63], v69, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2113) 
buffer_load_dwordx4 v86, s[48:51], s7 offen offset: 0 lds // (op 6783) Load value direct to lds









s_barrier  // (op 7464) 
v_mov_b32 v48, v11 // (op 6747) 



v_add_u32 v49, 8, v48 // (op 6747) 

s_waitcnt vmcnt(11)
ds_write_b128 v18, v[36:39] offset:28672 // (op 222) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v18, v[40:41] offset:28688 // (op 222) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v18, v[42:45] offset:28696 // (op 222) Store local data 
s_waitcnt vmcnt(8)
ds_write_b64 v18, v[46:47] offset:28712 // (op 222) Store local data 
buffer_load_dwordx2 v[34:35], v49, s[60:63], 0 offen // (op 6747) Load value


v_add_u32 v11, 8, v11 // (op 7268) 


v_mov_b32 v48, v19 // (op 6760) 
v_add_u32 v49, 8, v48 // (op 6760) 

s_waitcnt vmcnt(7)
s_barrier  // (op 7452) 
buffer_load_dwordx2 v[32:33], v49, s[56:59], 0 offen // (op 6760) Load value





v_add_u32 v19, 8, v19 // (op 7216) 
ds_read_b128 v[52:55], v5 offset:8192 // (op 214) Load local data 



buffer_load_dwordx4 v[36:39], v10, s[52:55], 0 offen offset:288 // (op 223) Load value
s_waitcnt lgkmcnt(1)
s_barrier  // (op 7461) 
buffer_load_dwordx2 v[40:41], v10, s[52:55], 0 offen offset:304 // (op 223) Load value



ds_read_b128 v[56:59], v6 offset:28672 // (op 216) Load local data 
ds_read_b64 v[60:61], v6 offset:28688 // (op 216) Load local data 
buffer_load_dwordx4 v[42:45], v10, s[52:55], 0 offen offset:312 // (op 223) Load value



v_bfe_u32 v66, v65, 0, 8 // (op 6129) 
v_bfe_u32 v67, v65, 8, 8 // (op 6129) 
v_bfe_u32 v68, v65, 16, 8 // (op 6129) 
v_bfe_u32 v69, v65, 24, 8 // (op 6129) 
v_permlane16_swap_b32 v66, v67 // (op 6129) 
ds_read_b128 v[70:73], v6 offset:30208 // (op 324) Load local data 
ds_read_b64 v[74:75], v6 offset:30224 // (op 324) Load local data 
v_permlane16_swap_b32 v68, v69 // (op 6129) 



ds_read_b128 v[76:79], v6 offset:31744 // (op 460) Load local data 
s_nop 0
v_permlane32_swap_b32 v66, v68 // (op 6129) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 6129) 
buffer_load_dwordx2 v[46:47], v10, s[52:55], 0 offen offset:328 // (op 223) Load value


v_add_u32 v17, 192, v17 // (op 6933) 
v_bfe_u32 v62, v13, 0, 8 // (op 6192) 
v_bfe_u32 v63, v13, 8, 8 // (op 6192) 
v_bfe_u32 v64, v13, 16, 8 // (op 6192) 
v_bfe_u32 v65, v13, 24, 8 // (op 6192) 
v_permlane16_swap_b32 v62, v63 // (op 6192) 
s_nop 0
v_permlane16_swap_b32 v64, v65 // (op 6192) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v62, v64 // (op 6192) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v63, v65 // (op 6192) 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 7455) 


s_lshl_b64 s[10:11], s[12:13], 6 // (op 6794) 
s_lshl_b64 s[14:15], s[10:11], 2 // (op 6794) 

v_lshrrev_b64 v[12:13], 3, s[14:15] // (op 6794) 


v_mov_b64 v[48:49], v[50:51] // (op 6794) 
s_add_u32 s7, 8192, s1 // (op 6794) 
s_mov_b32 m0, s7 // (op 6794) reg expression
s_mov_b32 s10, 192 // (op 6794) call()
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[56:61], a[0:3], v66, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 221) 
ds_read_b64 v[80:81], v6 offset:31760 // (op 460) Load local data 



ds_read_b128 v[82:85], v6 offset:33280 // (op 596) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[70:75], a[4:7], v66, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 316) 
ds_read_b64 v[86:87], v6 offset:33296 // (op 596) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[76:81], a[8:11], v66, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 452) 
s_barrier  // (op 7458) 
s_barrier  // (op 7515) 



ds_read_b128 v[88:91], v5 offset:9216 // (op 901) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[82:87], a[12:15], v66, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 588) 




ds_read_b128 v[52:55], v5 offset:10240 // (op 1511) Load local data 



ds_read_b128 v[92:95], v5 offset:11264 // (op 2121) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[56:61], a[16:19], v67, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1008) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[76:81], a[24:27], v67, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 932) 
buffer_load_dwordx4 v48, s[48:51], s10 offen offset: 0 lds // (op 6794) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[82:87], a[28:31], v67, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 894) 
v_add_co_u32 v48, s[14:15], v48, v12 // (op 6794) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[70:75], a[20:23], v67, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 970) 
v_addc_co_u32 v49, s[14:15], 0, v49, s[14:15] // (op 6794) most significant half


s_add_u32 m0, m0, 4096 // (op 6794) 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[52:55], v[56:61], a[32:35], v68, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1618) 
buffer_load_dwordx4 v48, s[48:51], s10 offen offset: 0 lds // (op 6794) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[52:55], v[76:81], a[40:43], v68, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1542) 
s_barrier  // (op 7441) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[52:55], v[82:87], a[44:47], v68, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1504) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[52:55], v[70:75], a[36:39], v68, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1580) 
v_add_co_u32 v14, vcc, 128, v14 // (op 6869) least significant half

v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 6869) most significant half
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[56:61], a[48:51], v69, v62 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2228) 
s_waitcnt vmcnt(13)
ds_write_b128 v16, v[20:23] offset:16384 // (op 13) Store local data 

s_waitcnt vmcnt(12)
ds_write_b64 v16, v[24:25] offset:16400 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[76:81], a[56:59], v69, v64 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2152) 
s_waitcnt vmcnt(11)
ds_write_b128 v16, v[26:29] offset:16408 // (op 13) Store local data 

s_waitcnt vmcnt(10)
ds_write_b64 v16, v[30:31] offset:16424 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[82:87], a[60:63], v69, v65 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2114) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[70:75], a[52:55], v69, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2190) 




s_ashr_i64 s[10:11], s[36:37], 7 // (op 2079) 
s_mov_b32 s7, s10 // (op 2079) convert

s_and_b32 s10, s7, 4294967294 // (op 2079) 

s_cmp_lt_u32 s6, s10 // (op 2079) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2079 // (op 2079) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoop_2079 if true)
RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoop_2079:












s_barrier  // (op 7537) 
v_lshrrev_b32 v10, 2, v1 // (op 6817) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6817) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 6817) 

s_lshr_b32 s6, s0, 1 // (op 6817) 

v_and_b32 v10, -4, v1 // (op 6817) 
v_sub_u32 v12, v1, v10 // (op 6817) 

v_lshlrev_b32 v10, 5, v12 // (op 6817) 

v_lshl_add_u32 v12, s6, 8, v10 // (op 6817) 


v_mov_b32 v31, 0 // (op 6817) convert
v_mov_b32 v30, v12 // (op 6817) convert

v_mul_lo_u32 v13, v11, s13 // (op 6817) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 6817) most significant: high of low * low
v_mul_lo_u32 v12, v11, s12 // (op 6817) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 6817) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v30 // (op 6817) least significant half
v_addc_co_u32 v13, s[0:1], v13, v31, s[0:1] // (op 6817) most significant half



v_lshlrev_b64 v[10:11], 2, v[12:13] // (op 6817) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 6817) 

s_mul_i32 s1, s12, s23 // (op 6819) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 6819) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 6819) least significant
s_mul_hi_u32 s7, s12, s22 // (op 6819) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6819) most significant: sum
s_add_u32 s1, s1, s7 // (op 6819) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 6819) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 6819) 

s_mov_b32 s48, s20 // (op 6819) reg expression
s_mov_b32 s49, s21 // (op 6819) reg expression
s_mov_b32 s50, s0 // (op 6819) reg expression
s_mov_b32 s51, 131072 // (op 6819) call()

v_and_b32 v10, -2, v1 // (op 6899) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 6899) 
v_sub_u32 v11, v1, v10 // (op 6899) 

s_lshr_b32 s6, s0, 1 // (op 6899) 

v_lshlrev_b32 v10, 6, v11 // (op 6899) 

v_lshl_add_u32 v11, s6, 8, v10 // (op 6899) 


v_mov_b32 v15, 0 // (op 6899) convert
v_mov_b32 v14, v11 // (op 6899) convert

v_lshrrev_b32 v10, 1, v1 // (op 6899) 
v_lshl_add_u32 v11, s17, 7, v10 // (op 6899) 

v_mul_lo_u32 v17, v11, s25 // (op 6899) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 6899) most significant: high of low * low
v_mul_lo_u32 v16, v11, s24 // (op 6899) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 6899) most significant: sum


v_add_co_u32 v10, s[0:1], v14, v16 // (op 6899) least significant half
v_addc_co_u32 v11, s[0:1], v15, v17, s[0:1] // (op 6899) most significant half



v_mul_lo_u32 v15, v11, 6 // (op 6899) most significant: low of high * low
v_mul_hi_u32 v16, v10, 6 // (op 6899) most significant: high of low * low
v_mul_lo_u32 v14, v10, 6 // (op 6899) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 6899) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 6899) 

v_mov_b32 v14, v10 // (op 6899) convert

s_mul_i32 s1, s9, 6 // (op 6901) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 6901) least significant
s_mul_hi_u32 s6, s8, 6 // (op 6901) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 6901) most significant: sum

s_mul_i32 s7, s24, s1 // (op 6901) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 6901) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 6901) least significant
s_mul_hi_u32 s11, s24, s0 // (op 6901) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 6901) most significant: sum
s_add_u32 s7, s7, s11 // (op 6901) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 6901) 

s_mov_b32 s52, s26 // (op 6901) reg expression
s_mov_b32 s53, s27 // (op 6901) reg expression
s_mov_b32 s54, s0 // (op 6901) reg expression
s_mov_b32 s55, 131072 // (op 6901) call()

v_and_b32 v10, -4, v1 // (op 6982) 
v_sub_u32 v11, v1, v10 // (op 6982) 

v_lshlrev_b32 v10, 5, v11 // (op 6982) 

v_lshrrev_b32 v11, 2, v1 // (op 6982) 
v_lshl_add_u32 v15, v11, 7, v10 // (op 6982) 


v_lshlrev_b32 v10, 2, v15 // (op 6982) 

v_lshrrev_b32 v11, 3, v10 // (op 6982) 

s_nop 0
v_readfirstlane_b32 s0, v11 // (op 6982) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 7041) 
v_sub_u32 v11, v1, v10 // (op 7041) 

v_lshlrev_b32 v10, 6, v11 // (op 7041) 

v_lshrrev_b32 v11, 1, v1 // (op 7041) 
v_lshl_add_u32 v15, v11, 7, v10 // (op 7041) 


v_mul_lo_u32 v10, v15, 6 // (op 7041) 

v_lshrrev_b32 v11, 3, v10 // (op 7041) 

v_and_b32 v10, 63, v1 // (op 7305) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 7305) 
v_lshrrev_b32 v15, 4, v10 // (op 7305) 
s_lshr_b32 s1, s6, 1 // (op 7305) 

v_and_b32 v16, 3, v15 // (op 7305) 

v_and_b32 v15, 15, v1 // (op 7305) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 7305) 


v_lshrrev_b32 v15, 7, v1 // (op 7305) 
v_lshl_add_u32 v16, v15, 6, v17 // (op 7305) 


v_lshl_add_u32 v15, s16, 7, v16 // (op 7305) 

v_lshrrev_b32 v16, 6, v10 // (op 7305) 

v_lshlrev_b32 v10, 2, v16 // (op 7305) 

v_lshl_add_u32 v16, s1, 3, v10 // (op 7305) 


v_mul_lo_u32 v10, v15, s28 // (op 7305) 
v_add_u32 v10, v10, v16 // (op 7305) 


s_mul_i32 s7, s28, s23 // (op 7307) most significant: low of low * high
s_mul_i32 s1, s29, s22 // (op 7307) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 7307) least significant
s_mul_hi_u32 s10, s28, s22 // (op 7307) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 7307) most significant: sum
s_add_u32 s7, s7, s10 // (op 7307) most significant: sum


s_mov_b32 s56, s30 // (op 7307) reg expression
s_mov_b32 s57, s31 // (op 7307) reg expression
s_mov_b32 s58, s6 // (op 7307) reg expression
s_mov_b32 s59, 131072 // (op 7307) call()

v_and_b32 v15, 63, v1 // (op 7332) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 7332) 
v_lshrrev_b32 v16, 6, v15 // (op 7332) 
s_lshr_b32 s1, s6, 1 // (op 7332) 

v_lshlrev_b32 v17, 2, v16 // (op 7332) 

v_lshl_add_u32 v16, s1, 3, v17 // (op 7332) 


v_lshrrev_b32 v17, 4, v15 // (op 7332) 

v_and_b32 v15, 3, v17 // (op 7332) 

v_and_b32 v17, 15, v1 // (op 7332) 
v_lshl_add_u32 v18, v15, 4, v17 // (op 7332) 


v_lshrrev_b32 v15, 6, v1 // (op 7332) 
v_and_b32 v17, 1, v15 // (op 7332) 

v_lshl_add_u32 v15, v17, 6, v18 // (op 7332) 


v_lshl_add_u32 v17, s17, 7, v15 // (op 7332) 

v_mul_lo_u32 v15, v17, s32 // (op 7332) 

v_add_u32 v17, v16, v15 // (op 7332) 


s_mul_i32 s7, s32, s9 // (op 7334) most significant: low of low * high
s_mul_i32 s1, s33, s8 // (op 7334) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 7334) least significant
s_mul_hi_u32 s10, s32, s8 // (op 7334) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 7334) most significant: sum
s_add_u32 s7, s7, s10 // (op 7334) most significant: sum


s_mov_b32 s60, s34 // (op 7334) reg expression
s_mov_b32 s61, s35 // (op 7334) reg expression
s_mov_b32 s62, s6 // (op 7334) reg expression
s_mov_b32 s63, 131072 // (op 7334) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 157) 
s_and_b32 s1, s6, 4294967294 // (op 157) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 156) 
s_mov_b32 s10, s6 // (op 156) convert

s_cmp_lt_u32 s1, s10 // (op 156) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_156:
s_barrier  // (op 7428) 
s_add_u32 s1, s1, 1 // (op 158) 

s_lshl_b64 s[6:7], s[12:13], 6 // (op 6788) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 6788) 

v_lshrrev_b64 v[18:19], 3, s[10:11] // (op 6788) 


v_mov_b64 v[20:21], v[12:13] // (op 6788) 
v_mov_b64 v[22:23], v[20:21] // (op 6788) 
s_mov_b32 m0, s0 // (op 6788) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 6788) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v18 // (op 6788) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 6788) most significant half

s_add_u32 m0, m0, 4096 // (op 6788) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 6788) Load value direct to lds



v_add_co_u32 v12, s[6:7], 64, v12 // (op 6807) least significant half
v_addc_co_u32 v13, s[6:7], 0, v13, s[6:7] // (op 6807) most significant half



v_mov_b32 v15, v14 // (op 173) 

buffer_load_dwordx4 v[18:21], v15, s[52:55], 0 offen // (op 173) Load value
buffer_load_dwordx2 v[22:23], v15, s[52:55], 0 offen offset:16 // (op 173) Load value
buffer_load_dwordx4 v[24:27], v15, s[52:55], 0 offen offset:24 // (op 173) Load value
buffer_load_dwordx2 v[28:29], v15, s[52:55], 0 offen offset:40 // (op 173) Load value

v_add_u32 v14, 96, v14 // (op 6891) 
s_barrier  // (op 7444) 
s_waitcnt vmcnt(4)
s_barrier  // (op 7434) 






ds_read_b128 v[30:33], v3 // (op 164) Load local data 
s_waitcnt vmcnt(3)
ds_write_b128 v11, v[18:21] offset:16384 // (op 172) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v11, v[22:23] offset:16400 // (op 172) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v11, v[24:27] offset:16408 // (op 172) Store local data 
s_waitcnt vmcnt(0)
ds_write_b64 v11, v[28:29] offset:16424 // (op 172) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 7448) 



ds_read_b128 v[18:21], v4 offset:16384 // (op 166) Load local data 
ds_read_b64 v[22:23], v4 offset:16400 // (op 166) Load local data 


v_mov_b32 v15, v10 // (op 168) 
v_mov_b32 v16, v15 // (op 168) reg expression

buffer_load_dwordx2 v[24:25], v16, s[56:59], 0 offen // (op 168) Load value


v_add_u32 v10, 4, v10 // (op 7295) 


v_mov_b32 v15, v17 // (op 170) 
v_mov_b32 v16, v15 // (op 170) reg expression

buffer_load_dwordx2 v[26:27], v16, s[60:63], 0 offen // (op 170) Load value


v_add_u32 v17, 4, v17 // (op 7322) 
s_waitcnt vmcnt(1)
v_bfe_u32 v34, v24, 0, 8 // (op 6241) 



v_bfe_u32 v35, v24, 8, 8 // (op 6241) 
v_bfe_u32 v36, v24, 16, 8 // (op 6241) 
v_bfe_u32 v37, v24, 24, 8 // (op 6241) 
v_permlane16_swap_b32 v34, v35 // (op 6241) 
ds_read_b128 v[38:41], v4 offset:17920 // (op 295) Load local data 
v_permlane16_swap_b32 v36, v37 // (op 6241) 
s_nop 1
v_permlane32_swap_b32 v34, v36 // (op 6241) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v35, v37 // (op 6241) 
ds_read_b64 v[42:43], v4 offset:17936 // (op 295) Load local data 




ds_read_b128 v[44:47], v4 offset:19456 // (op 431) Load local data 
ds_read_b64 v[48:49], v4 offset:19472 // (op 431) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v26, 0, 8 // (op 6271) 



v_bfe_u32 v51, v26, 8, 8 // (op 6271) 
v_bfe_u32 v52, v26, 16, 8 // (op 6271) 
v_bfe_u32 v53, v26, 24, 8 // (op 6271) 
v_permlane16_swap_b32 v50, v51 // (op 6271) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 6271) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 6271) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 6271) 
ds_read_b128 v[54:57], v4 offset:20992 // (op 567) Load local data 

ds_read_b64 v[58:59], v4 offset:21008 // (op 567) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[18:23], a[0:3], v34, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 171) 
ds_read_b128 v[24:27], v3 offset:1024 // (op 789) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[38:43], a[4:7], v34, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 298) 
ds_read_b128 v[60:63], v3 offset:2048 // (op 1399) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:33], v[44:49], a[8:11], v34, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 434) 
ds_read_b128 v[64:67], v3 offset:3072 // (op 2009) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:33], v[54:59], a[12:15], v34, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 570) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[24:27], v[18:23], a[16:19], v35, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 793) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[24:27], v[44:49], a[24:27], v35, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 847) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[24:27], v[54:59], a[28:31], v35, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 874) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[24:27], v[38:43], a[20:23], v35, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 820) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:63], v[18:23], a[32:35], v36, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1403) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:63], v[44:49], a[40:43], v36, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1457) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:63], v[54:59], a[44:47], v36, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1484) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[38:43], a[36:39], v36, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 1430) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[64:67], v[18:23], a[48:51], v37, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2013) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:67], v[44:49], a[56:59], v37, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2067) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[64:67], v[54:59], a[60:63], v37, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2094) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[64:67], v[38:43], a[52:55], v37, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b010 // (op 2040) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 156) 
s_mov_b32 s10, s6 // (op 156) convert

s_cmp_lt_u32 s1, s10 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopTop_KLoopTail_156 if true)
RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ForLoopBottom_KLoopTail_156:





















s_barrier  // (op 7499) 





v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_116
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



v_accvgpr_read v10, a4 // (op 2906) 
v_accvgpr_read v11, a5 // (op 2906) 
v_accvgpr_read v12, a6 // (op 2906) 
v_accvgpr_read v13, a7 // (op 2906) 
v_mul_f32 v14, v7, v10 // (op 2906) 
v_mul_f32 v15, v7, v11 // (op 2906) 
v_mul_f32 v16, v7, v12 // (op 2906) 
v_mul_f32 v17, v7, v13 // (op 2906) 

v_lshrrev_b32 v10, 16, v14 // (op 2908) 
v_lshrrev_b32 v11, 16, v15 // (op 2908) 
v_lshrrev_b32 v12, 16, v16 // (op 2908) 
v_lshrrev_b32 v13, 16, v17 // (op 2908) 


v_pack_b32_f16 v4, v10, v11 // (op 2909) 
v_pack_b32_f16 v5, v12, v13 // (op 2909) 

s_lshl_b32 s0, s4, 5 // (op 2909) 
v_add_u32 v3, v9, s0 // (op 2909) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 2909) Store value



v_accvgpr_read v10, a8 // (op 2920) 
v_accvgpr_read v11, a9 // (op 2920) 
v_accvgpr_read v12, a10 // (op 2920) 
v_accvgpr_read v13, a11 // (op 2920) 
v_mul_f32 v14, v7, v10 // (op 2920) 
v_mul_f32 v15, v7, v11 // (op 2920) 
v_mul_f32 v16, v7, v12 // (op 2920) 
v_mul_f32 v17, v7, v13 // (op 2920) 

v_lshrrev_b32 v10, 16, v14 // (op 2922) 
v_lshrrev_b32 v11, 16, v15 // (op 2922) 
v_lshrrev_b32 v12, 16, v16 // (op 2922) 
v_lshrrev_b32 v13, 16, v17 // (op 2922) 


v_pack_b32_f16 v4, v10, v11 // (op 2923) 
v_pack_b32_f16 v5, v12, v13 // (op 2923) 

s_lshl_b32 s0, s4, 6 // (op 2923) 
v_add_u32 v3, v9, s0 // (op 2923) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 2923) Store value



v_accvgpr_read v10, a12 // (op 2934) 
v_accvgpr_read v11, a13 // (op 2934) 
v_accvgpr_read v12, a14 // (op 2934) 
v_accvgpr_read v13, a15 // (op 2934) 
v_mul_f32 v14, v7, v10 // (op 2934) 
v_mul_f32 v15, v7, v11 // (op 2934) 
v_mul_f32 v16, v7, v12 // (op 2934) 
v_mul_f32 v17, v7, v13 // (op 2934) 

v_lshrrev_b32 v10, 16, v14 // (op 2936) 
v_lshrrev_b32 v11, 16, v15 // (op 2936) 
v_lshrrev_b32 v12, 16, v16 // (op 2936) 
v_lshrrev_b32 v13, 16, v17 // (op 2936) 


v_pack_b32_f16 v4, v10, v11 // (op 2937) 
v_pack_b32_f16 v5, v12, v13 // (op 2937) 

s_lshl_b32 s0, s4, 5 // (op 2937) 
s_mul_i32 s1, 3, s0 // (op 2937) 

v_add_u32 v3, v9, s1 // (op 2937) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 2937) Store value



v_accvgpr_read v10, a16 // (op 2970) 
v_accvgpr_read v11, a17 // (op 2970) 
v_accvgpr_read v12, a18 // (op 2970) 
v_accvgpr_read v13, a19 // (op 2970) 
v_mul_f32 v14, v7, v10 // (op 2970) 
v_mul_f32 v15, v7, v11 // (op 2970) 
v_mul_f32 v16, v7, v12 // (op 2970) 
v_mul_f32 v17, v7, v13 // (op 2970) 

v_lshrrev_b32 v10, 16, v14 // (op 2972) 
v_lshrrev_b32 v11, 16, v15 // (op 2972) 
v_lshrrev_b32 v12, 16, v16 // (op 2972) 
v_lshrrev_b32 v13, 16, v17 // (op 2972) 


v_pack_b32_f16 v4, v10, v11 // (op 2974) 
v_pack_b32_f16 v5, v12, v13 // (op 2974) 

v_add_u32 v3, 32, v9 // (op 2974) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 2974) Store value



v_accvgpr_read v10, a20 // (op 2984) 
v_accvgpr_read v11, a21 // (op 2984) 
v_accvgpr_read v12, a22 // (op 2984) 
v_accvgpr_read v13, a23 // (op 2984) 
v_mul_f32 v14, v7, v10 // (op 2984) 
v_mul_f32 v15, v7, v11 // (op 2984) 
v_mul_f32 v16, v7, v12 // (op 2984) 
v_mul_f32 v17, v7, v13 // (op 2984) 

v_lshrrev_b32 v10, 16, v14 // (op 2986) 
v_lshrrev_b32 v11, 16, v15 // (op 2986) 
v_lshrrev_b32 v12, 16, v16 // (op 2986) 
v_lshrrev_b32 v13, 16, v17 // (op 2986) 


v_pack_b32_f16 v4, v10, v11 // (op 2976) 
v_pack_b32_f16 v5, v12, v13 // (op 2976) 

s_lshl_b32 s0, s4, 5 // (op 2976) 
s_add_u32 s0, s0, 32 // (op 2976) 
v_add_u32 v3, v9, s0 // (op 2976) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 2976) Store value



v_accvgpr_read v10, a24 // (op 2990) 
v_accvgpr_read v11, a25 // (op 2990) 
v_accvgpr_read v12, a26 // (op 2990) 
v_accvgpr_read v13, a27 // (op 2990) 
v_mul_f32 v14, v7, v10 // (op 2990) 
v_mul_f32 v15, v7, v11 // (op 2990) 
v_mul_f32 v16, v7, v12 // (op 2990) 
v_mul_f32 v17, v7, v13 // (op 2990) 

v_lshrrev_b32 v10, 16, v14 // (op 2992) 
v_lshrrev_b32 v11, 16, v15 // (op 2992) 
v_lshrrev_b32 v12, 16, v16 // (op 2992) 
v_lshrrev_b32 v13, 16, v17 // (op 2992) 


v_pack_b32_f16 v4, v10, v11 // (op 2978) 
v_pack_b32_f16 v5, v12, v13 // (op 2978) 

s_lshl_b32 s0, s4, 6 // (op 2978) 
s_add_u32 s0, s0, 32 // (op 2978) 
v_add_u32 v3, v9, s0 // (op 2978) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 2978) Store value



v_accvgpr_read v10, a28 // (op 2996) 
v_accvgpr_read v11, a29 // (op 2996) 
v_accvgpr_read v12, a30 // (op 2996) 
v_accvgpr_read v13, a31 // (op 2996) 
v_mul_f32 v14, v7, v10 // (op 2996) 
v_mul_f32 v15, v7, v11 // (op 2996) 
v_mul_f32 v16, v7, v12 // (op 2996) 
v_mul_f32 v17, v7, v13 // (op 2996) 

v_lshrrev_b32 v10, 16, v14 // (op 2998) 
v_lshrrev_b32 v11, 16, v15 // (op 2998) 
v_lshrrev_b32 v12, 16, v16 // (op 2998) 
v_lshrrev_b32 v13, 16, v17 // (op 2998) 


v_pack_b32_f16 v4, v10, v11 // (op 2980) 
v_pack_b32_f16 v5, v12, v13 // (op 2980) 

s_lshl_b32 s0, s4, 5 // (op 2980) 
s_mul_i32 s1, 3, s0 // (op 2980) 
s_add_u32 s1, s1, 32 // (op 2980) 

v_add_u32 v3, v9, s1 // (op 2980) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 2980) Store value



v_accvgpr_read v10, a32 // (op 3046) 
v_accvgpr_read v11, a33 // (op 3046) 
v_accvgpr_read v12, a34 // (op 3046) 
v_accvgpr_read v13, a35 // (op 3046) 
v_mul_f32 v14, v7, v10 // (op 3046) 
v_mul_f32 v15, v7, v11 // (op 3046) 
v_mul_f32 v16, v7, v12 // (op 3046) 
v_mul_f32 v17, v7, v13 // (op 3046) 

v_lshrrev_b32 v10, 16, v14 // (op 3048) 
v_lshrrev_b32 v11, 16, v15 // (op 3048) 
v_lshrrev_b32 v12, 16, v16 // (op 3048) 
v_lshrrev_b32 v13, 16, v17 // (op 3048) 


v_pack_b32_f16 v4, v10, v11 // (op 3050) 
v_pack_b32_f16 v5, v12, v13 // (op 3050) 

v_add_u32 v3, 64, v9 // (op 3050) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 3050) Store value



v_accvgpr_read v10, a36 // (op 3060) 
v_accvgpr_read v11, a37 // (op 3060) 
v_accvgpr_read v12, a38 // (op 3060) 
v_accvgpr_read v13, a39 // (op 3060) 
v_mul_f32 v14, v7, v10 // (op 3060) 
v_mul_f32 v15, v7, v11 // (op 3060) 
v_mul_f32 v16, v7, v12 // (op 3060) 
v_mul_f32 v17, v7, v13 // (op 3060) 

v_lshrrev_b32 v10, 16, v14 // (op 3062) 
v_lshrrev_b32 v11, 16, v15 // (op 3062) 
v_lshrrev_b32 v12, 16, v16 // (op 3062) 
v_lshrrev_b32 v13, 16, v17 // (op 3062) 


v_pack_b32_f16 v4, v10, v11 // (op 3052) 
v_pack_b32_f16 v5, v12, v13 // (op 3052) 

s_lshl_b32 s0, s4, 5 // (op 3052) 
s_add_u32 s0, s0, 64 // (op 3052) 
v_add_u32 v3, v9, s0 // (op 3052) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 3052) Store value



v_accvgpr_read v10, a40 // (op 3066) 
v_accvgpr_read v11, a41 // (op 3066) 
v_accvgpr_read v12, a42 // (op 3066) 
v_accvgpr_read v13, a43 // (op 3066) 
v_mul_f32 v14, v7, v10 // (op 3066) 
v_mul_f32 v15, v7, v11 // (op 3066) 
v_mul_f32 v16, v7, v12 // (op 3066) 
v_mul_f32 v17, v7, v13 // (op 3066) 

v_lshrrev_b32 v10, 16, v14 // (op 3068) 
v_lshrrev_b32 v11, 16, v15 // (op 3068) 
v_lshrrev_b32 v12, 16, v16 // (op 3068) 
v_lshrrev_b32 v13, 16, v17 // (op 3068) 


v_pack_b32_f16 v4, v10, v11 // (op 3054) 
v_pack_b32_f16 v5, v12, v13 // (op 3054) 

s_lshl_b32 s0, s4, 6 // (op 3054) 
s_add_u32 s0, s0, 64 // (op 3054) 
v_add_u32 v3, v9, s0 // (op 3054) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 3054) Store value



v_accvgpr_read v10, a44 // (op 3072) 
v_accvgpr_read v11, a45 // (op 3072) 
v_accvgpr_read v12, a46 // (op 3072) 
v_accvgpr_read v13, a47 // (op 3072) 
v_mul_f32 v14, v7, v10 // (op 3072) 
v_mul_f32 v15, v7, v11 // (op 3072) 
v_mul_f32 v16, v7, v12 // (op 3072) 
v_mul_f32 v17, v7, v13 // (op 3072) 

v_lshrrev_b32 v10, 16, v14 // (op 3074) 
v_lshrrev_b32 v11, 16, v15 // (op 3074) 
v_lshrrev_b32 v12, 16, v16 // (op 3074) 
v_lshrrev_b32 v13, 16, v17 // (op 3074) 


v_pack_b32_f16 v4, v10, v11 // (op 3056) 
v_pack_b32_f16 v5, v12, v13 // (op 3056) 

s_lshl_b32 s0, s4, 5 // (op 3056) 
s_mul_i32 s1, 3, s0 // (op 3056) 
s_add_u32 s1, s1, 64 // (op 3056) 

v_add_u32 v3, v9, s1 // (op 3056) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 3056) Store value



v_accvgpr_read v10, a48 // (op 3122) 
v_accvgpr_read v11, a49 // (op 3122) 
v_accvgpr_read v12, a50 // (op 3122) 
v_accvgpr_read v13, a51 // (op 3122) 
v_mul_f32 v14, v7, v10 // (op 3122) 
v_mul_f32 v15, v7, v11 // (op 3122) 
v_mul_f32 v16, v7, v12 // (op 3122) 
v_mul_f32 v17, v7, v13 // (op 3122) 

v_lshrrev_b32 v10, 16, v14 // (op 3124) 
v_lshrrev_b32 v11, 16, v15 // (op 3124) 
v_lshrrev_b32 v12, 16, v16 // (op 3124) 
v_lshrrev_b32 v13, 16, v17 // (op 3124) 


v_pack_b32_f16 v4, v10, v11 // (op 3126) 
v_pack_b32_f16 v5, v12, v13 // (op 3126) 

v_add_u32 v3, 96, v9 // (op 3126) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 3126) Store value



v_accvgpr_read v10, a52 // (op 3136) 
v_accvgpr_read v11, a53 // (op 3136) 
v_accvgpr_read v12, a54 // (op 3136) 
v_accvgpr_read v13, a55 // (op 3136) 
v_mul_f32 v14, v7, v10 // (op 3136) 
v_mul_f32 v15, v7, v11 // (op 3136) 
v_mul_f32 v16, v7, v12 // (op 3136) 
v_mul_f32 v17, v7, v13 // (op 3136) 

v_lshrrev_b32 v10, 16, v14 // (op 3138) 
v_lshrrev_b32 v11, 16, v15 // (op 3138) 
v_lshrrev_b32 v12, 16, v16 // (op 3138) 
v_lshrrev_b32 v13, 16, v17 // (op 3138) 


v_pack_b32_f16 v4, v10, v11 // (op 3128) 
v_pack_b32_f16 v5, v12, v13 // (op 3128) 

s_lshl_b32 s0, s4, 5 // (op 3128) 
s_add_u32 s0, s0, 96 // (op 3128) 
v_add_u32 v3, v9, s0 // (op 3128) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 3128) Store value



v_accvgpr_read v10, a56 // (op 3142) 
v_accvgpr_read v11, a57 // (op 3142) 
v_accvgpr_read v12, a58 // (op 3142) 
v_accvgpr_read v13, a59 // (op 3142) 
v_mul_f32 v14, v7, v10 // (op 3142) 
v_mul_f32 v15, v7, v11 // (op 3142) 
v_mul_f32 v16, v7, v12 // (op 3142) 
v_mul_f32 v17, v7, v13 // (op 3142) 

v_lshrrev_b32 v10, 16, v14 // (op 3144) 
v_lshrrev_b32 v11, 16, v15 // (op 3144) 
v_lshrrev_b32 v12, 16, v16 // (op 3144) 
v_lshrrev_b32 v13, 16, v17 // (op 3144) 


v_pack_b32_f16 v4, v10, v11 // (op 3130) 
v_pack_b32_f16 v5, v12, v13 // (op 3130) 

s_lshl_b32 s0, s4, 6 // (op 3130) 
s_add_u32 s0, s0, 96 // (op 3130) 
v_add_u32 v3, v9, s0 // (op 3130) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 3130) Store value



v_accvgpr_read v10, a60 // (op 3148) 
v_accvgpr_read v11, a61 // (op 3148) 
v_accvgpr_read v12, a62 // (op 3148) 
v_accvgpr_read v13, a63 // (op 3148) 
v_mul_f32 v14, v7, v10 // (op 3148) 
v_mul_f32 v15, v7, v11 // (op 3148) 
v_mul_f32 v16, v7, v12 // (op 3148) 
v_mul_f32 v17, v7, v13 // (op 3148) 

v_lshrrev_b32 v10, 16, v14 // (op 3150) 
v_lshrrev_b32 v11, 16, v15 // (op 3150) 
v_lshrrev_b32 v12, 16, v16 // (op 3150) 
v_lshrrev_b32 v13, 16, v17 // (op 3150) 


v_pack_b32_f16 v4, v10, v11 // (op 3132) 
v_pack_b32_f16 v5, v12, v13 // (op 3132) 

s_lshl_b32 s0, s4, 5 // (op 3132) 
s_mul_i32 s1, 3, s0 // (op 3132) 
s_add_u32 s1, s1, 96 // (op 3132) 

v_add_u32 v3, v9, s1 // (op 3132) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 3132) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalFalse_0_beta_116:


v_add_u32 v3, 96, v8 // (op 3520) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3520) Load value

v_accvgpr_read v10, a48 // (op 3507) 
v_accvgpr_read v11, a49 // (op 3507) 
v_accvgpr_read v12, a50 // (op 3507) 
v_accvgpr_read v13, a51 // (op 3507) 
v_mul_f32 v14, v7, v10 // (op 3507) 
v_mul_f32 v15, v7, v11 // (op 3507) 
v_mul_f32 v16, v7, v12 // (op 3507) 
v_mul_f32 v17, v7, v13 // (op 3507) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3519) 
v_lshrrev_b32 v11, 16, v4 // (op 3519) 
v_lshlrev_b32 v10, 16, v10 // (op 3519) 
v_lshlrev_b32 v11, 16, v11 // (op 3519) 
v_mul_f32 v18, v0, v10 // (op 3519) 
v_mul_f32 v19, v0, v11 // (op 3519) 
v_and_b32 v10, 65535, v5 // (op 3519) 
v_lshrrev_b32 v11, 16, v5 // (op 3519) 
v_lshlrev_b32 v10, 16, v10 // (op 3519) 
v_lshlrev_b32 v11, 16, v11 // (op 3519) 
v_mul_f32 v20, v0, v10 // (op 3519) 
v_mul_f32 v21, v0, v11 // (op 3519) 


v_add_f32 v10, v18, v14 // (op 3508) 
v_add_f32 v11, v19, v15 // (op 3508) 
v_add_f32 v12, v20, v16 // (op 3508) 
v_add_f32 v13, v21, v17 // (op 3508) 


v_lshrrev_b32 v14, 16, v10 // (op 3509) 
v_lshrrev_b32 v15, 16, v11 // (op 3509) 
v_lshrrev_b32 v16, 16, v12 // (op 3509) 
v_lshrrev_b32 v17, 16, v13 // (op 3509) 


v_pack_b32_f16 v10, v14, v15 // (op 3511) 
v_pack_b32_f16 v11, v16, v17 // (op 3511) 

v_add_u32 v3, 96, v9 // (op 3511) 

buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3511) Store value





s_lshl_b32 s0, s2, 5 // (op 3525) 
s_add_u32 s0, s0, 96 // (op 3525) 
v_add_u32 v3, v8, s0 // (op 3525) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3525) Load value

v_accvgpr_read v10, a52 // (op 3537) 
v_accvgpr_read v11, a53 // (op 3537) 
v_accvgpr_read v12, a54 // (op 3537) 
v_accvgpr_read v13, a55 // (op 3537) 
v_mul_f32 v14, v7, v10 // (op 3537) 
v_mul_f32 v15, v7, v11 // (op 3537) 
v_mul_f32 v16, v7, v12 // (op 3537) 
v_mul_f32 v17, v7, v13 // (op 3537) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3522) 
v_lshrrev_b32 v11, 16, v4 // (op 3522) 
v_lshlrev_b32 v10, 16, v10 // (op 3522) 
v_lshlrev_b32 v11, 16, v11 // (op 3522) 
v_mul_f32 v18, v0, v10 // (op 3522) 
v_mul_f32 v19, v0, v11 // (op 3522) 
v_and_b32 v10, 65535, v5 // (op 3522) 
v_lshrrev_b32 v11, 16, v5 // (op 3522) 
v_lshlrev_b32 v10, 16, v10 // (op 3522) 
v_lshlrev_b32 v11, 16, v11 // (op 3522) 
v_mul_f32 v20, v0, v10 // (op 3522) 
v_mul_f32 v21, v0, v11 // (op 3522) 


v_add_f32 v10, v18, v14 // (op 3523) 
v_add_f32 v11, v19, v15 // (op 3523) 
v_add_f32 v12, v20, v16 // (op 3523) 
v_add_f32 v13, v21, v17 // (op 3523) 


v_lshrrev_b32 v14, 16, v10 // (op 3524) 
v_lshrrev_b32 v15, 16, v11 // (op 3524) 
v_lshrrev_b32 v16, 16, v12 // (op 3524) 
v_lshrrev_b32 v17, 16, v13 // (op 3524) 


v_pack_b32_f16 v10, v14, v15 // (op 3513) 
v_pack_b32_f16 v11, v16, v17 // (op 3513) 

s_lshl_b32 s0, s4, 5 // (op 3513) 
s_add_u32 s0, s0, 96 // (op 3513) 
v_add_u32 v3, v9, s0 // (op 3513) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3513) Store value





s_lshl_b32 s0, s2, 6 // (op 3530) 
s_add_u32 s0, s0, 96 // (op 3530) 
v_add_u32 v3, v8, s0 // (op 3530) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3530) Load value

v_accvgpr_read v10, a56 // (op 3539) 
v_accvgpr_read v11, a57 // (op 3539) 
v_accvgpr_read v12, a58 // (op 3539) 
v_accvgpr_read v13, a59 // (op 3539) 
v_mul_f32 v14, v7, v10 // (op 3539) 
v_mul_f32 v15, v7, v11 // (op 3539) 
v_mul_f32 v16, v7, v12 // (op 3539) 
v_mul_f32 v17, v7, v13 // (op 3539) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3527) 
v_lshrrev_b32 v11, 16, v4 // (op 3527) 
v_lshlrev_b32 v10, 16, v10 // (op 3527) 
v_lshlrev_b32 v11, 16, v11 // (op 3527) 
v_mul_f32 v18, v0, v10 // (op 3527) 
v_mul_f32 v19, v0, v11 // (op 3527) 
v_and_b32 v10, 65535, v5 // (op 3527) 
v_lshrrev_b32 v11, 16, v5 // (op 3527) 
v_lshlrev_b32 v10, 16, v10 // (op 3527) 
v_lshlrev_b32 v11, 16, v11 // (op 3527) 
v_mul_f32 v20, v0, v10 // (op 3527) 
v_mul_f32 v21, v0, v11 // (op 3527) 


v_add_f32 v10, v18, v14 // (op 3528) 
v_add_f32 v11, v19, v15 // (op 3528) 
v_add_f32 v12, v20, v16 // (op 3528) 
v_add_f32 v13, v21, v17 // (op 3528) 


v_lshrrev_b32 v14, 16, v10 // (op 3529) 
v_lshrrev_b32 v15, 16, v11 // (op 3529) 
v_lshrrev_b32 v16, 16, v12 // (op 3529) 
v_lshrrev_b32 v17, 16, v13 // (op 3529) 


v_pack_b32_f16 v10, v14, v15 // (op 3515) 
v_pack_b32_f16 v11, v16, v17 // (op 3515) 

s_lshl_b32 s0, s4, 6 // (op 3515) 
s_add_u32 s0, s0, 96 // (op 3515) 
v_add_u32 v3, v9, s0 // (op 3515) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3515) Store value





s_lshl_b32 s0, s2, 5 // (op 3535) 
s_mul_i32 s1, 3, s0 // (op 3535) 
s_add_u32 s1, s1, 96 // (op 3535) 

v_add_u32 v3, v8, s1 // (op 3535) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3535) Load value

v_accvgpr_read v10, a60 // (op 3541) 
v_accvgpr_read v11, a61 // (op 3541) 
v_accvgpr_read v12, a62 // (op 3541) 
v_accvgpr_read v13, a63 // (op 3541) 
v_mul_f32 v14, v7, v10 // (op 3541) 
v_mul_f32 v15, v7, v11 // (op 3541) 
v_mul_f32 v16, v7, v12 // (op 3541) 
v_mul_f32 v17, v7, v13 // (op 3541) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3532) 
v_lshrrev_b32 v11, 16, v4 // (op 3532) 
v_lshlrev_b32 v10, 16, v10 // (op 3532) 
v_lshlrev_b32 v11, 16, v11 // (op 3532) 
v_mul_f32 v18, v0, v10 // (op 3532) 
v_mul_f32 v19, v0, v11 // (op 3532) 
v_and_b32 v10, 65535, v5 // (op 3532) 
v_lshrrev_b32 v11, 16, v5 // (op 3532) 
v_lshlrev_b32 v10, 16, v10 // (op 3532) 
v_lshlrev_b32 v11, 16, v11 // (op 3532) 
v_mul_f32 v20, v0, v10 // (op 3532) 
v_mul_f32 v21, v0, v11 // (op 3532) 


v_add_f32 v10, v18, v14 // (op 3533) 
v_add_f32 v11, v19, v15 // (op 3533) 
v_add_f32 v12, v20, v16 // (op 3533) 
v_add_f32 v13, v21, v17 // (op 3533) 


v_lshrrev_b32 v14, 16, v10 // (op 3534) 
v_lshrrev_b32 v15, 16, v11 // (op 3534) 
v_lshrrev_b32 v16, 16, v12 // (op 3534) 
v_lshrrev_b32 v17, 16, v13 // (op 3534) 


v_pack_b32_f16 v10, v14, v15 // (op 3517) 
v_pack_b32_f16 v11, v16, v17 // (op 3517) 

s_lshl_b32 s0, s4, 5 // (op 3517) 
s_mul_i32 s1, 3, s0 // (op 3517) 
s_add_u32 s1, s1, 96 // (op 3517) 

v_add_u32 v3, v9, s1 // (op 3517) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3517) Store value





v_add_u32 v3, 64, v8 // (op 3433) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3433) Load value

v_accvgpr_read v10, a32 // (op 3420) 
v_accvgpr_read v11, a33 // (op 3420) 
v_accvgpr_read v12, a34 // (op 3420) 
v_accvgpr_read v13, a35 // (op 3420) 
v_mul_f32 v14, v7, v10 // (op 3420) 
v_mul_f32 v15, v7, v11 // (op 3420) 
v_mul_f32 v16, v7, v12 // (op 3420) 
v_mul_f32 v17, v7, v13 // (op 3420) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3432) 
v_lshrrev_b32 v11, 16, v4 // (op 3432) 
v_lshlrev_b32 v10, 16, v10 // (op 3432) 
v_lshlrev_b32 v11, 16, v11 // (op 3432) 
v_mul_f32 v18, v0, v10 // (op 3432) 
v_mul_f32 v19, v0, v11 // (op 3432) 
v_and_b32 v10, 65535, v5 // (op 3432) 
v_lshrrev_b32 v11, 16, v5 // (op 3432) 
v_lshlrev_b32 v10, 16, v10 // (op 3432) 
v_lshlrev_b32 v11, 16, v11 // (op 3432) 
v_mul_f32 v20, v0, v10 // (op 3432) 
v_mul_f32 v21, v0, v11 // (op 3432) 


v_add_f32 v10, v18, v14 // (op 3421) 
v_add_f32 v11, v19, v15 // (op 3421) 
v_add_f32 v12, v20, v16 // (op 3421) 
v_add_f32 v13, v21, v17 // (op 3421) 


v_lshrrev_b32 v14, 16, v10 // (op 3422) 
v_lshrrev_b32 v15, 16, v11 // (op 3422) 
v_lshrrev_b32 v16, 16, v12 // (op 3422) 
v_lshrrev_b32 v17, 16, v13 // (op 3422) 


v_pack_b32_f16 v10, v14, v15 // (op 3424) 
v_pack_b32_f16 v11, v16, v17 // (op 3424) 

v_add_u32 v3, 64, v9 // (op 3424) 

buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3424) Store value





s_lshl_b32 s0, s2, 5 // (op 3438) 
s_add_u32 s0, s0, 64 // (op 3438) 
v_add_u32 v3, v8, s0 // (op 3438) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3438) Load value

v_accvgpr_read v10, a36 // (op 3450) 
v_accvgpr_read v11, a37 // (op 3450) 
v_accvgpr_read v12, a38 // (op 3450) 
v_accvgpr_read v13, a39 // (op 3450) 
v_mul_f32 v14, v7, v10 // (op 3450) 
v_mul_f32 v15, v7, v11 // (op 3450) 
v_mul_f32 v16, v7, v12 // (op 3450) 
v_mul_f32 v17, v7, v13 // (op 3450) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3435) 
v_lshrrev_b32 v11, 16, v4 // (op 3435) 
v_lshlrev_b32 v10, 16, v10 // (op 3435) 
v_lshlrev_b32 v11, 16, v11 // (op 3435) 
v_mul_f32 v18, v0, v10 // (op 3435) 
v_mul_f32 v19, v0, v11 // (op 3435) 
v_and_b32 v10, 65535, v5 // (op 3435) 
v_lshrrev_b32 v11, 16, v5 // (op 3435) 
v_lshlrev_b32 v10, 16, v10 // (op 3435) 
v_lshlrev_b32 v11, 16, v11 // (op 3435) 
v_mul_f32 v20, v0, v10 // (op 3435) 
v_mul_f32 v21, v0, v11 // (op 3435) 


v_add_f32 v10, v18, v14 // (op 3436) 
v_add_f32 v11, v19, v15 // (op 3436) 
v_add_f32 v12, v20, v16 // (op 3436) 
v_add_f32 v13, v21, v17 // (op 3436) 


v_lshrrev_b32 v14, 16, v10 // (op 3437) 
v_lshrrev_b32 v15, 16, v11 // (op 3437) 
v_lshrrev_b32 v16, 16, v12 // (op 3437) 
v_lshrrev_b32 v17, 16, v13 // (op 3437) 


v_pack_b32_f16 v10, v14, v15 // (op 3426) 
v_pack_b32_f16 v11, v16, v17 // (op 3426) 

s_lshl_b32 s0, s4, 5 // (op 3426) 
s_add_u32 s0, s0, 64 // (op 3426) 
v_add_u32 v3, v9, s0 // (op 3426) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3426) Store value





s_lshl_b32 s0, s2, 6 // (op 3443) 
s_add_u32 s0, s0, 64 // (op 3443) 
v_add_u32 v3, v8, s0 // (op 3443) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3443) Load value

v_accvgpr_read v10, a40 // (op 3452) 
v_accvgpr_read v11, a41 // (op 3452) 
v_accvgpr_read v12, a42 // (op 3452) 
v_accvgpr_read v13, a43 // (op 3452) 
v_mul_f32 v14, v7, v10 // (op 3452) 
v_mul_f32 v15, v7, v11 // (op 3452) 
v_mul_f32 v16, v7, v12 // (op 3452) 
v_mul_f32 v17, v7, v13 // (op 3452) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3440) 
v_lshrrev_b32 v11, 16, v4 // (op 3440) 
v_lshlrev_b32 v10, 16, v10 // (op 3440) 
v_lshlrev_b32 v11, 16, v11 // (op 3440) 
v_mul_f32 v18, v0, v10 // (op 3440) 
v_mul_f32 v19, v0, v11 // (op 3440) 
v_and_b32 v10, 65535, v5 // (op 3440) 
v_lshrrev_b32 v11, 16, v5 // (op 3440) 
v_lshlrev_b32 v10, 16, v10 // (op 3440) 
v_lshlrev_b32 v11, 16, v11 // (op 3440) 
v_mul_f32 v20, v0, v10 // (op 3440) 
v_mul_f32 v21, v0, v11 // (op 3440) 


v_add_f32 v10, v18, v14 // (op 3441) 
v_add_f32 v11, v19, v15 // (op 3441) 
v_add_f32 v12, v20, v16 // (op 3441) 
v_add_f32 v13, v21, v17 // (op 3441) 


v_lshrrev_b32 v14, 16, v10 // (op 3442) 
v_lshrrev_b32 v15, 16, v11 // (op 3442) 
v_lshrrev_b32 v16, 16, v12 // (op 3442) 
v_lshrrev_b32 v17, 16, v13 // (op 3442) 


v_pack_b32_f16 v10, v14, v15 // (op 3428) 
v_pack_b32_f16 v11, v16, v17 // (op 3428) 

s_lshl_b32 s0, s4, 6 // (op 3428) 
s_add_u32 s0, s0, 64 // (op 3428) 
v_add_u32 v3, v9, s0 // (op 3428) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3428) Store value





s_lshl_b32 s0, s2, 5 // (op 3448) 
s_mul_i32 s1, 3, s0 // (op 3448) 
s_add_u32 s1, s1, 64 // (op 3448) 

v_add_u32 v3, v8, s1 // (op 3448) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3448) Load value

v_accvgpr_read v10, a44 // (op 3454) 
v_accvgpr_read v11, a45 // (op 3454) 
v_accvgpr_read v12, a46 // (op 3454) 
v_accvgpr_read v13, a47 // (op 3454) 
v_mul_f32 v14, v7, v10 // (op 3454) 
v_mul_f32 v15, v7, v11 // (op 3454) 
v_mul_f32 v16, v7, v12 // (op 3454) 
v_mul_f32 v17, v7, v13 // (op 3454) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3445) 
v_lshrrev_b32 v11, 16, v4 // (op 3445) 
v_lshlrev_b32 v10, 16, v10 // (op 3445) 
v_lshlrev_b32 v11, 16, v11 // (op 3445) 
v_mul_f32 v18, v0, v10 // (op 3445) 
v_mul_f32 v19, v0, v11 // (op 3445) 
v_and_b32 v10, 65535, v5 // (op 3445) 
v_lshrrev_b32 v11, 16, v5 // (op 3445) 
v_lshlrev_b32 v10, 16, v10 // (op 3445) 
v_lshlrev_b32 v11, 16, v11 // (op 3445) 
v_mul_f32 v20, v0, v10 // (op 3445) 
v_mul_f32 v21, v0, v11 // (op 3445) 


v_add_f32 v10, v18, v14 // (op 3446) 
v_add_f32 v11, v19, v15 // (op 3446) 
v_add_f32 v12, v20, v16 // (op 3446) 
v_add_f32 v13, v21, v17 // (op 3446) 


v_lshrrev_b32 v14, 16, v10 // (op 3447) 
v_lshrrev_b32 v15, 16, v11 // (op 3447) 
v_lshrrev_b32 v16, 16, v12 // (op 3447) 
v_lshrrev_b32 v17, 16, v13 // (op 3447) 


v_pack_b32_f16 v10, v14, v15 // (op 3430) 
v_pack_b32_f16 v11, v16, v17 // (op 3430) 

s_lshl_b32 s0, s4, 5 // (op 3430) 
s_mul_i32 s1, 3, s0 // (op 3430) 
s_add_u32 s1, s1, 64 // (op 3430) 

v_add_u32 v3, v9, s1 // (op 3430) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3430) Store value





v_add_u32 v3, 32, v8 // (op 3346) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3346) Load value

v_accvgpr_read v10, a16 // (op 3333) 
v_accvgpr_read v11, a17 // (op 3333) 
v_accvgpr_read v12, a18 // (op 3333) 
v_accvgpr_read v13, a19 // (op 3333) 
v_mul_f32 v14, v7, v10 // (op 3333) 
v_mul_f32 v15, v7, v11 // (op 3333) 
v_mul_f32 v16, v7, v12 // (op 3333) 
v_mul_f32 v17, v7, v13 // (op 3333) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3345) 
v_lshrrev_b32 v11, 16, v4 // (op 3345) 
v_lshlrev_b32 v10, 16, v10 // (op 3345) 
v_lshlrev_b32 v11, 16, v11 // (op 3345) 
v_mul_f32 v18, v0, v10 // (op 3345) 
v_mul_f32 v19, v0, v11 // (op 3345) 
v_and_b32 v10, 65535, v5 // (op 3345) 
v_lshrrev_b32 v11, 16, v5 // (op 3345) 
v_lshlrev_b32 v10, 16, v10 // (op 3345) 
v_lshlrev_b32 v11, 16, v11 // (op 3345) 
v_mul_f32 v20, v0, v10 // (op 3345) 
v_mul_f32 v21, v0, v11 // (op 3345) 


v_add_f32 v10, v18, v14 // (op 3334) 
v_add_f32 v11, v19, v15 // (op 3334) 
v_add_f32 v12, v20, v16 // (op 3334) 
v_add_f32 v13, v21, v17 // (op 3334) 


v_lshrrev_b32 v14, 16, v10 // (op 3335) 
v_lshrrev_b32 v15, 16, v11 // (op 3335) 
v_lshrrev_b32 v16, 16, v12 // (op 3335) 
v_lshrrev_b32 v17, 16, v13 // (op 3335) 


v_pack_b32_f16 v10, v14, v15 // (op 3337) 
v_pack_b32_f16 v11, v16, v17 // (op 3337) 

v_add_u32 v3, 32, v9 // (op 3337) 

buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3337) Store value





s_lshl_b32 s0, s2, 5 // (op 3351) 
s_add_u32 s0, s0, 32 // (op 3351) 
v_add_u32 v3, v8, s0 // (op 3351) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3351) Load value

v_accvgpr_read v10, a20 // (op 3363) 
v_accvgpr_read v11, a21 // (op 3363) 
v_accvgpr_read v12, a22 // (op 3363) 
v_accvgpr_read v13, a23 // (op 3363) 
v_mul_f32 v14, v7, v10 // (op 3363) 
v_mul_f32 v15, v7, v11 // (op 3363) 
v_mul_f32 v16, v7, v12 // (op 3363) 
v_mul_f32 v17, v7, v13 // (op 3363) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3348) 
v_lshrrev_b32 v11, 16, v4 // (op 3348) 
v_lshlrev_b32 v10, 16, v10 // (op 3348) 
v_lshlrev_b32 v11, 16, v11 // (op 3348) 
v_mul_f32 v18, v0, v10 // (op 3348) 
v_mul_f32 v19, v0, v11 // (op 3348) 
v_and_b32 v10, 65535, v5 // (op 3348) 
v_lshrrev_b32 v11, 16, v5 // (op 3348) 
v_lshlrev_b32 v10, 16, v10 // (op 3348) 
v_lshlrev_b32 v11, 16, v11 // (op 3348) 
v_mul_f32 v20, v0, v10 // (op 3348) 
v_mul_f32 v21, v0, v11 // (op 3348) 


v_add_f32 v10, v18, v14 // (op 3349) 
v_add_f32 v11, v19, v15 // (op 3349) 
v_add_f32 v12, v20, v16 // (op 3349) 
v_add_f32 v13, v21, v17 // (op 3349) 


v_lshrrev_b32 v14, 16, v10 // (op 3350) 
v_lshrrev_b32 v15, 16, v11 // (op 3350) 
v_lshrrev_b32 v16, 16, v12 // (op 3350) 
v_lshrrev_b32 v17, 16, v13 // (op 3350) 


v_pack_b32_f16 v10, v14, v15 // (op 3339) 
v_pack_b32_f16 v11, v16, v17 // (op 3339) 

s_lshl_b32 s0, s4, 5 // (op 3339) 
s_add_u32 s0, s0, 32 // (op 3339) 
v_add_u32 v3, v9, s0 // (op 3339) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3339) Store value





s_lshl_b32 s0, s2, 6 // (op 3356) 
s_add_u32 s0, s0, 32 // (op 3356) 
v_add_u32 v3, v8, s0 // (op 3356) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3356) Load value

v_accvgpr_read v10, a24 // (op 3365) 
v_accvgpr_read v11, a25 // (op 3365) 
v_accvgpr_read v12, a26 // (op 3365) 
v_accvgpr_read v13, a27 // (op 3365) 
v_mul_f32 v14, v7, v10 // (op 3365) 
v_mul_f32 v15, v7, v11 // (op 3365) 
v_mul_f32 v16, v7, v12 // (op 3365) 
v_mul_f32 v17, v7, v13 // (op 3365) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3353) 
v_lshrrev_b32 v11, 16, v4 // (op 3353) 
v_lshlrev_b32 v10, 16, v10 // (op 3353) 
v_lshlrev_b32 v11, 16, v11 // (op 3353) 
v_mul_f32 v18, v0, v10 // (op 3353) 
v_mul_f32 v19, v0, v11 // (op 3353) 
v_and_b32 v10, 65535, v5 // (op 3353) 
v_lshrrev_b32 v11, 16, v5 // (op 3353) 
v_lshlrev_b32 v10, 16, v10 // (op 3353) 
v_lshlrev_b32 v11, 16, v11 // (op 3353) 
v_mul_f32 v20, v0, v10 // (op 3353) 
v_mul_f32 v21, v0, v11 // (op 3353) 


v_add_f32 v10, v18, v14 // (op 3354) 
v_add_f32 v11, v19, v15 // (op 3354) 
v_add_f32 v12, v20, v16 // (op 3354) 
v_add_f32 v13, v21, v17 // (op 3354) 


v_lshrrev_b32 v14, 16, v10 // (op 3355) 
v_lshrrev_b32 v15, 16, v11 // (op 3355) 
v_lshrrev_b32 v16, 16, v12 // (op 3355) 
v_lshrrev_b32 v17, 16, v13 // (op 3355) 


v_pack_b32_f16 v10, v14, v15 // (op 3341) 
v_pack_b32_f16 v11, v16, v17 // (op 3341) 

s_lshl_b32 s0, s4, 6 // (op 3341) 
s_add_u32 s0, s0, 32 // (op 3341) 
v_add_u32 v3, v9, s0 // (op 3341) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3341) Store value





s_lshl_b32 s0, s2, 5 // (op 3361) 
s_mul_i32 s1, 3, s0 // (op 3361) 
s_add_u32 s1, s1, 32 // (op 3361) 

v_add_u32 v3, v8, s1 // (op 3361) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3361) Load value

v_accvgpr_read v10, a28 // (op 3367) 
v_accvgpr_read v11, a29 // (op 3367) 
v_accvgpr_read v12, a30 // (op 3367) 
v_accvgpr_read v13, a31 // (op 3367) 
v_mul_f32 v14, v7, v10 // (op 3367) 
v_mul_f32 v15, v7, v11 // (op 3367) 
v_mul_f32 v16, v7, v12 // (op 3367) 
v_mul_f32 v17, v7, v13 // (op 3367) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3358) 
v_lshrrev_b32 v11, 16, v4 // (op 3358) 
v_lshlrev_b32 v10, 16, v10 // (op 3358) 
v_lshlrev_b32 v11, 16, v11 // (op 3358) 
v_mul_f32 v18, v0, v10 // (op 3358) 
v_mul_f32 v19, v0, v11 // (op 3358) 
v_and_b32 v10, 65535, v5 // (op 3358) 
v_lshrrev_b32 v11, 16, v5 // (op 3358) 
v_lshlrev_b32 v10, 16, v10 // (op 3358) 
v_lshlrev_b32 v11, 16, v11 // (op 3358) 
v_mul_f32 v20, v0, v10 // (op 3358) 
v_mul_f32 v21, v0, v11 // (op 3358) 


v_add_f32 v10, v18, v14 // (op 3359) 
v_add_f32 v11, v19, v15 // (op 3359) 
v_add_f32 v12, v20, v16 // (op 3359) 
v_add_f32 v13, v21, v17 // (op 3359) 


v_lshrrev_b32 v14, 16, v10 // (op 3360) 
v_lshrrev_b32 v15, 16, v11 // (op 3360) 
v_lshrrev_b32 v16, 16, v12 // (op 3360) 
v_lshrrev_b32 v17, 16, v13 // (op 3360) 


v_pack_b32_f16 v10, v14, v15 // (op 3343) 
v_pack_b32_f16 v11, v16, v17 // (op 3343) 

s_lshl_b32 s0, s4, 5 // (op 3343) 
s_mul_i32 s1, 3, s0 // (op 3343) 
s_add_u32 s1, s1, 32 // (op 3343) 

v_add_u32 v3, v9, s1 // (op 3343) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3343) Store value





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





s_lshl_b32 s0, s2, 5 // (op 3254) 
v_add_u32 v3, v8, s0 // (op 3254) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3254) Load value

v_accvgpr_read v10, a4 // (op 3250) 
v_accvgpr_read v11, a5 // (op 3250) 
v_accvgpr_read v12, a6 // (op 3250) 
v_accvgpr_read v13, a7 // (op 3250) 
v_mul_f32 v14, v7, v10 // (op 3250) 
v_mul_f32 v15, v7, v11 // (op 3250) 
v_mul_f32 v16, v7, v12 // (op 3250) 
v_mul_f32 v17, v7, v13 // (op 3250) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3255) 
v_lshrrev_b32 v11, 16, v4 // (op 3255) 
v_lshlrev_b32 v10, 16, v10 // (op 3255) 
v_lshlrev_b32 v11, 16, v11 // (op 3255) 
v_mul_f32 v18, v0, v10 // (op 3255) 
v_mul_f32 v19, v0, v11 // (op 3255) 
v_and_b32 v10, 65535, v5 // (op 3255) 
v_lshrrev_b32 v11, 16, v5 // (op 3255) 
v_lshlrev_b32 v10, 16, v10 // (op 3255) 
v_lshlrev_b32 v11, 16, v11 // (op 3255) 
v_mul_f32 v20, v0, v10 // (op 3255) 
v_mul_f32 v21, v0, v11 // (op 3255) 


v_add_f32 v10, v18, v14 // (op 3251) 
v_add_f32 v11, v19, v15 // (op 3251) 
v_add_f32 v12, v20, v16 // (op 3251) 
v_add_f32 v13, v21, v17 // (op 3251) 


v_lshrrev_b32 v14, 16, v10 // (op 3252) 
v_lshrrev_b32 v15, 16, v11 // (op 3252) 
v_lshrrev_b32 v16, 16, v12 // (op 3252) 
v_lshrrev_b32 v17, 16, v13 // (op 3252) 


v_pack_b32_f16 v10, v14, v15 // (op 3253) 
v_pack_b32_f16 v11, v16, v17 // (op 3253) 

s_lshl_b32 s0, s4, 5 // (op 3253) 
v_add_u32 v3, v9, s0 // (op 3253) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3253) Store value





s_lshl_b32 s0, s2, 6 // (op 3268) 
v_add_u32 v3, v8, s0 // (op 3268) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3268) Load value

v_accvgpr_read v10, a8 // (op 3264) 
v_accvgpr_read v11, a9 // (op 3264) 
v_accvgpr_read v12, a10 // (op 3264) 
v_accvgpr_read v13, a11 // (op 3264) 
v_mul_f32 v14, v7, v10 // (op 3264) 
v_mul_f32 v15, v7, v11 // (op 3264) 
v_mul_f32 v16, v7, v12 // (op 3264) 
v_mul_f32 v17, v7, v13 // (op 3264) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3269) 
v_lshrrev_b32 v11, 16, v4 // (op 3269) 
v_lshlrev_b32 v10, 16, v10 // (op 3269) 
v_lshlrev_b32 v11, 16, v11 // (op 3269) 
v_mul_f32 v18, v0, v10 // (op 3269) 
v_mul_f32 v19, v0, v11 // (op 3269) 
v_and_b32 v10, 65535, v5 // (op 3269) 
v_lshrrev_b32 v11, 16, v5 // (op 3269) 
v_lshlrev_b32 v10, 16, v10 // (op 3269) 
v_lshlrev_b32 v11, 16, v11 // (op 3269) 
v_mul_f32 v20, v0, v10 // (op 3269) 
v_mul_f32 v21, v0, v11 // (op 3269) 


v_add_f32 v10, v18, v14 // (op 3265) 
v_add_f32 v11, v19, v15 // (op 3265) 
v_add_f32 v12, v20, v16 // (op 3265) 
v_add_f32 v13, v21, v17 // (op 3265) 


v_lshrrev_b32 v14, 16, v10 // (op 3266) 
v_lshrrev_b32 v15, 16, v11 // (op 3266) 
v_lshrrev_b32 v16, 16, v12 // (op 3266) 
v_lshrrev_b32 v17, 16, v13 // (op 3266) 


v_pack_b32_f16 v10, v14, v15 // (op 3267) 
v_pack_b32_f16 v11, v16, v17 // (op 3267) 

s_lshl_b32 s0, s4, 6 // (op 3267) 
v_add_u32 v3, v9, s0 // (op 3267) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3267) Store value





s_lshl_b32 s0, s2, 5 // (op 3282) 
s_mul_i32 s1, 3, s0 // (op 3282) 

v_add_u32 v3, v8, s1 // (op 3282) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 3282) Load value

v_accvgpr_read v10, a12 // (op 3278) 
v_accvgpr_read v11, a13 // (op 3278) 
v_accvgpr_read v12, a14 // (op 3278) 
v_accvgpr_read v13, a15 // (op 3278) 
v_mul_f32 v14, v7, v10 // (op 3278) 
v_mul_f32 v15, v7, v11 // (op 3278) 
v_mul_f32 v16, v7, v12 // (op 3278) 
v_mul_f32 v17, v7, v13 // (op 3278) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v4 // (op 3283) 
v_lshrrev_b32 v11, 16, v4 // (op 3283) 
v_lshlrev_b32 v10, 16, v10 // (op 3283) 
v_lshlrev_b32 v11, 16, v11 // (op 3283) 
v_mul_f32 v18, v0, v10 // (op 3283) 
v_mul_f32 v19, v0, v11 // (op 3283) 
v_and_b32 v10, 65535, v5 // (op 3283) 
v_lshrrev_b32 v11, 16, v5 // (op 3283) 
v_lshlrev_b32 v10, 16, v10 // (op 3283) 
v_lshlrev_b32 v11, 16, v11 // (op 3283) 
v_mul_f32 v20, v0, v10 // (op 3283) 
v_mul_f32 v21, v0, v11 // (op 3283) 


v_add_f32 v10, v18, v14 // (op 3279) 
v_add_f32 v11, v19, v15 // (op 3279) 
v_add_f32 v12, v20, v16 // (op 3279) 
v_add_f32 v13, v21, v17 // (op 3279) 


v_lshrrev_b32 v14, 16, v10 // (op 3280) 
v_lshrrev_b32 v15, 16, v11 // (op 3280) 
v_lshrrev_b32 v16, 16, v12 // (op 3280) 
v_lshrrev_b32 v17, 16, v13 // (op 3280) 


v_pack_b32_f16 v10, v14, v15 // (op 3281) 
v_pack_b32_f16 v11, v16, v17 // (op 3281) 

s_lshl_b32 s0, s4, 5 // (op 3281) 
s_mul_i32 s1, 3, s0 // (op 3281) 

v_add_u32 v3, v9, s1 // (op 3281) 


buffer_store_dwordx2 v[10:11], v3, s[44:47], 0 offen // (op 3281) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_ConditionalBottom_0_beta_116:
























s_endpgm  // End of RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.LRR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2, .LRR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_end-RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
  .amdhsa_next_free_vgpr 164
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 40960 // lds bytes
  .amdhsa_accum_offset 100
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
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DataTypeA: F4
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
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 40960
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 98
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

