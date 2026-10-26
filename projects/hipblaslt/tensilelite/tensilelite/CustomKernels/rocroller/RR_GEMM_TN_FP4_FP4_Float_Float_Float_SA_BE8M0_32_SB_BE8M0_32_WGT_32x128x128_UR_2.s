

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 2013) 
v_and_b32 v3, 15, v1 // (op 2013) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2013) 


v_and_b32 v0, 63, v1 // (op 2013) 
v_lshrrev_b32 v3, 4, v0 // (op 2013) 

v_lshlrev_b32 v0, 5, v3 // (op 2013) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 2013) 


v_lshlrev_b32 v0, 2, v3 // (op 2013) 

v_lshrrev_b32 v3, 3, v0 // (op 2013) 

v_lshrrev_b32 v0, 7, v1 // (op 2171) 
v_and_b32 v4, 15, v1 // (op 2171) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 2171) 


v_and_b32 v0, 63, v1 // (op 2171) 
v_lshrrev_b32 v4, 4, v0 // (op 2171) 

v_lshlrev_b32 v0, 5, v4 // (op 2171) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 2171) 


v_lshlrev_b32 v0, 2, v4 // (op 2171) 

v_lshrrev_b32 v4, 3, v0 // (op 2171) 

v_lshrrev_b32 v0, 7, v1 // (op 1956) 
v_and_b32 v5, 63, v1 // (op 1956) 
v_lshrrev_b32 v6, 4, v5 // (op 1956) 
v_lshlrev_b32 v7, 2, v6 // (op 1956) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1956) 


v_lshl_add_u32 v0, s16, 5, v6 // (op 1956) 

v_lshrrev_b32 v6, 2, v5 // (op 1956) 

v_and_b32 v5, 3, v6 // (op 1956) 

v_and_b32 v6, 3, v1 // (op 1956) 
v_lshl_add_u32 v7, v5, 2, v6 // (op 1956) 


v_lshrrev_b32 v5, 6, v1 // (op 1956) 
v_and_b32 v6, 1, v5 // (op 1956) 

v_lshl_add_u32 v5, v6, 4, v7 // (op 1956) 


v_lshl_add_u32 v6, s17, 7, v5 // (op 1956) 

v_mul_lo_u32 v5, v6, s2 // (op 1956) 

v_add_lshl_u32 v6, v0, v5, 2 // (op 1956) 


v_lshrrev_b32 v0, 6, v1 // (op 2071) 
v_and_b32 v5, 1, v0 // (op 2071) 

v_and_b32 v0, 15, v1 // (op 2071) 
v_lshl_add_u32 v7, v5, 4, v0 // (op 2071) 


v_and_b32 v0, 63, v1 // (op 2071) 
v_lshrrev_b32 v5, 4, v0 // (op 2071) 

v_lshlrev_b32 v0, 5, v5 // (op 2071) 

v_lshl_add_u32 v5, v7, 7, v0 // (op 2071) 


v_lshlrev_b32 v0, 2, v5 // (op 2071) 

v_lshrrev_b32 v5, 3, v0 // (op 2071) 

v_lshrrev_b32 v0, 6, v1 // (op 2129) 
v_and_b32 v7, 1, v0 // (op 2129) 

v_and_b32 v0, 15, v1 // (op 2129) 
s_mul_i32 s1, s2, s9 // (op 1958) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 1958) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 1958) least significant
s_mul_hi_u32 s19, s2, s8 // (op 1958) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1958) most significant: sum
s_add_u32 s1, s1, s19 // (op 1958) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 1958) 

v_lshl_add_u32 v8, v7, 4, v0 // (op 2129) 


v_and_b32 v0, 63, v1 // (op 2129) 
v_lshrrev_b32 v7, 4, v0 // (op 2129) 

v_lshl_add_u32 v0, v8, 2, v7 // (op 2129) 


v_lshrrev_b32 v7, 7, v1 // (op 2145) 
v_and_b32 v8, 63, v1 // (op 2145) 
v_lshrrev_b32 v9, 4, v8 // (op 2145) 
v_lshlrev_b32 v10, 2, v9 // (op 2145) 

v_lshl_add_u32 v9, v7, 4, v10 // (op 2145) 


v_lshl_add_u32 v7, s16, 5, v9 // (op 2145) 

v_lshrrev_b32 v9, 2, v8 // (op 2145) 

v_and_b32 v8, 3, v9 // (op 2145) 

v_and_b32 v9, 3, v1 // (op 2145) 
v_lshl_add_u32 v10, v8, 2, v9 // (op 2145) 


v_lshrrev_b32 v8, 6, v1 // (op 2145) 
v_and_b32 v9, 1, v8 // (op 2145) 

v_lshl_add_u32 v8, v9, 4, v10 // (op 2145) 


v_lshl_add_u32 v9, s17, 7, v8 // (op 2145) 

v_mul_lo_u32 v8, v9, s4 // (op 2145) 

v_add_lshl_u32 v9, v7, v8, 2 // (op 2145) 


v_lshrrev_b32 v7, 6, v1 // (op 2200) 
v_and_b32 v8, 1, v7 // (op 2200) 

v_and_b32 v7, 15, v1 // (op 2200) 
v_lshl_add_u32 v10, v8, 4, v7 // (op 2200) 


v_and_b32 v7, 63, v1 // (op 2200) 
v_lshrrev_b32 v8, 4, v7 // (op 2200) 

v_lshlrev_b32 v7, 5, v8 // (op 2200) 

v_lshl_add_u32 v8, v10, 7, v7 // (op 2200) 


v_lshlrev_b32 v7, 2, v8 // (op 2200) 

v_lshrrev_b32 v8, 3, v7 // (op 2200) 

v_lshrrev_b32 v7, 6, v1 // (op 2229) 
v_and_b32 v10, 1, v7 // (op 2229) 

v_and_b32 v7, 15, v1 // (op 2229) 
s_mul_i32 s1, s4, s9 // (op 2147) most significant: low of low * high
v_lshl_add_u32 v11, v10, 4, v7 // (op 2229) 


v_and_b32 v7, 63, v1 // (op 2229) 
v_lshrrev_b32 v10, 4, v7 // (op 2229) 

v_lshl_add_u32 v7, v11, 2, v10 // (op 2229) 


s_mul_i32 s38, s5, s8 // (op 2147) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 2147) least significant
s_mul_hi_u32 s39, s4, s8 // (op 2147) most significant: high of low * low
s_add_u32 s1, s1, s38 // (op 2147) most significant: sum
s_add_u32 s1, s1, s39 // (op 2147) most significant: sum


s_lshl_b64 s[38:39], s[0:1], 2 // (op 2147) 

s_mov_b32 s40, s6 // (op 1958) reg expression
s_mov_b32 s41, s7 // (op 1958) reg expression
s_mov_b32 s42, s18 // (op 1958) reg expression
s_mov_b32 s43, 131072 // (op 1958) call()

s_mov_b32 s44, s10 // (op 2147) reg expression

s_mov_b32 s45, s11 // (op 2147) reg expression
s_mov_b32 s46, s38 // (op 2147) reg expression
s_mov_b32 s47, 131072 // (op 2147) call()


v_mov_b32 v10, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 306) call()
v_accvgpr_write a5, 0.00000 // (op 306) call()
v_accvgpr_write a6, 0.00000 // (op 306) call()
v_accvgpr_write a7, 0.00000 // (op 306) call()
v_accvgpr_write a8, 0.00000 // (op 462) call()
v_accvgpr_write a9, 0.00000 // (op 462) call()
v_accvgpr_write a10, 0.00000 // (op 462) call()
v_accvgpr_write a11, 0.00000 // (op 462) call()
v_accvgpr_write a12, 0.00000 // (op 618) call()
v_accvgpr_write a13, 0.00000 // (op 618) call()
v_accvgpr_write a14, 0.00000 // (op 618) call()
v_accvgpr_write a15, 0.00000 // (op 618) call()
v_lshrrev_b32 v12, 3, v1 // (op 1723) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1723) 

v_and_b32 v12, -8, v1 // (op 1723) 
v_sub_u32 v14, v1, v12 // (op 1723) 

v_lshlrev_b32 v12, 4, v14 // (op 1723) 

v_mov_b32 v15, 0 // (op 1723) convert
v_mov_b32 v14, v12 // (op 1723) convert

v_mul_lo_u32 v17, v13, s13 // (op 1723) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1723) most significant: high of low * low
v_mul_lo_u32 v16, v13, s12 // (op 1723) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 1723) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1723) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1723) most significant half



v_lshlrev_b64 v[12:13], 2, v[16:17] // (op 1723) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1723) 

v_mov_b32 v12, v14 // (op 1723) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 1725) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1725) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1725) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1725) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1725) most significant: sum
s_add_u32 s1, s1, s7 // (op 1725) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1725) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1725) 

s_mov_b32 s48, s20 // (op 1725) reg expression
s_mov_b32 s49, s21 // (op 1725) reg expression
s_mov_b32 s50, s0 // (op 1725) reg expression
s_mov_b32 s51, 131072 // (op 1725) call()

v_lshrrev_b32 v13, 3, v1 // (op 1742) 
v_lshl_add_u32 v14, s16, 5, v13 // (op 1742) 

v_and_b32 v13, -8, v1 // (op 1742) 
v_sub_u32 v15, v1, v13 // (op 1742) 

v_lshlrev_b32 v13, 4, v15 // (op 1742) 

v_mov_b32 v17, 0 // (op 1742) convert
v_mov_b32 v16, v13 // (op 1742) convert

v_mul_lo_u32 v19, v14, s13 // (op 1742) most significant: low of low * high
v_mul_hi_u32 v13, v14, s12 // (op 1742) most significant: high of low * low
v_mul_lo_u32 v18, v14, s12 // (op 1742) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 1742) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1742) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1742) most significant half



v_lshlrev_b64 v[16:17], 2, v[18:19] // (op 1742) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 1742) 

v_mov_b32 v13, v14 // (op 1742) convert

s_mul_i32 s1, s12, s23 // (op 1744) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1744) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1744) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1744) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1744) most significant: sum
s_add_u32 s1, s1, s7 // (op 1744) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1744) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1744) 

s_mov_b32 s48, s20 // (op 1744) reg expression
s_mov_b32 s49, s21 // (op 1744) reg expression
s_mov_b32 s50, s0 // (op 1744) reg expression
s_mov_b32 s51, 131072 // (op 1744) call()

v_and_b32 v14, -4, v1 // (op 1787) 
v_sub_u32 v15, v1, v14 // (op 1787) 

v_lshlrev_b32 v14, 5, v15 // (op 1787) 

v_mov_b32 v17, 0 // (op 1787) convert
v_mov_b32 v16, v14 // (op 1787) convert

v_lshrrev_b32 v14, 2, v1 // (op 1787) 
v_lshlrev_b32 v15, 1, v14 // (op 1787) 

v_lshl_add_u32 v14, s17, 7, v15 // (op 1787) 

v_mul_lo_u32 v19, v14, s25 // (op 1787) most significant: low of low * high
v_mul_hi_u32 v15, v14, s24 // (op 1787) most significant: high of low * low
v_mul_lo_u32 v18, v14, s24 // (op 1787) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 1787) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 1787) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 1787) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 1787) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 1787) 

v_mov_b32 v14, v16 // (op 1787) convert

s_mul_i32 s1, s24, s9 // (op 1789) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1789) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1789) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1789) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1789) most significant: sum
s_add_u32 s1, s1, s7 // (op 1789) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1789) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1789) 

s_mov_b32 s52, s26 // (op 1789) reg expression
s_mov_b32 s53, s27 // (op 1789) reg expression
s_mov_b32 s54, s0 // (op 1789) reg expression
s_mov_b32 s55, 131072 // (op 1789) call()

v_and_b32 v15, -4, v1 // (op 1806) 
v_sub_u32 v16, v1, v15 // (op 1806) 

v_lshlrev_b32 v15, 5, v16 // (op 1806) 

v_mov_b32 v17, 0 // (op 1806) convert
v_mov_b32 v16, v15 // (op 1806) convert

v_lshrrev_b32 v15, 2, v1 // (op 1806) 
v_lshlrev_b32 v18, 1, v15 // (op 1806) 

v_lshl_add_u32 v15, s17, 7, v18 // (op 1806) 

v_mul_lo_u32 v19, v15, s25 // (op 1806) most significant: low of low * high
v_mul_hi_u32 v20, v15, s24 // (op 1806) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 1806) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1806) most significant: sum


v_add_co_u32 v20, s[0:1], v16, v18 // (op 1806) least significant half
v_addc_co_u32 v21, s[0:1], v17, v19, s[0:1] // (op 1806) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 1806) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 1806) 

v_mov_b32 v15, v18 // (op 1806) convert

s_mul_i32 s1, s24, s9 // (op 1808) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1808) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1808) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1808) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1808) most significant: sum
s_add_u32 s1, s1, s7 // (op 1808) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1808) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1808) 

s_mov_b32 s52, s26 // (op 1808) reg expression
s_mov_b32 s53, s27 // (op 1808) reg expression
s_mov_b32 s54, s0 // (op 1808) reg expression
s_mov_b32 s55, 131072 // (op 1808) call()

v_lshrrev_b32 v16, 7, v1 // (op 1851) 
v_and_b32 v17, 15, v1 // (op 1851) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 1851) 


v_lshl_add_u32 v16, s16, 5, v18 // (op 1851) 

v_and_b32 v17, 63, v1 // (op 1851) 
v_lshrrev_b32 v18, 4, v17 // (op 1851) 

v_mul_lo_u32 v17, v16, s28 // (op 1851) 
v_add_u32 v17, v17, v18 // (op 1851) 


s_mul_i32 s1, s28, s23 // (op 1853) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1853) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1853) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1853) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1853) most significant: sum
s_add_u32 s1, s1, s7 // (op 1853) most significant: sum


s_mov_b32 s56, s30 // (op 1853) reg expression
s_mov_b32 s57, s31 // (op 1853) reg expression
s_mov_b32 s58, s0 // (op 1853) reg expression
s_mov_b32 s59, 131072 // (op 1853) call()

v_lshrrev_b32 v16, 7, v1 // (op 1870) 
v_and_b32 v18, 15, v1 // (op 1870) 
v_lshl_add_u32 v19, v16, 4, v18 // (op 1870) 


v_lshl_add_u32 v16, s16, 5, v19 // (op 1870) 

v_and_b32 v18, 63, v1 // (op 1870) 
v_lshrrev_b32 v19, 4, v18 // (op 1870) 

v_mul_lo_u32 v18, v16, s28 // (op 1870) 
v_add_u32 v18, v18, v19 // (op 1870) 


s_mul_i32 s1, s28, s23 // (op 1872) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1872) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1872) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1872) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1872) most significant: sum
s_add_u32 s1, s1, s7 // (op 1872) most significant: sum


s_mov_b32 s56, s30 // (op 1872) reg expression
s_mov_b32 s57, s31 // (op 1872) reg expression
s_mov_b32 s58, s0 // (op 1872) reg expression
s_mov_b32 s59, 131072 // (op 1872) call()

v_and_b32 v16, -4, v1 // (op 1915) 
v_sub_u32 v19, v1, v16 // (op 1915) 

v_lshrrev_b32 v16, 2, v1 // (op 1915) 
v_lshlrev_b32 v20, 1, v16 // (op 1915) 

v_lshl_add_u32 v16, s17, 7, v20 // (op 1915) 

v_mul_lo_u32 v20, v16, s32 // (op 1915) 

v_add_u32 v16, v19, v20 // (op 1915) 


s_mul_i32 s1, s32, s9 // (op 1917) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1917) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1917) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1917) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1917) most significant: sum
s_add_u32 s1, s1, s7 // (op 1917) most significant: sum


s_mov_b32 s60, s34 // (op 1917) reg expression
s_mov_b32 s61, s35 // (op 1917) reg expression
s_mov_b32 s62, s0 // (op 1917) reg expression
s_mov_b32 s63, 131072 // (op 1917) call()

v_and_b32 v19, -4, v1 // (op 1934) 
v_sub_u32 v20, v1, v19 // (op 1934) 

v_lshrrev_b32 v19, 2, v1 // (op 1934) 
v_lshlrev_b32 v21, 1, v19 // (op 1934) 

v_lshl_add_u32 v19, s17, 7, v21 // (op 1934) 

v_mul_lo_u32 v21, v19, s32 // (op 1934) 

v_add_u32 v19, v20, v21 // (op 1934) 


s_mul_i32 s1, s32, s9 // (op 1936) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1936) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1936) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1936) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1936) most significant: sum
s_add_u32 s1, s1, s7 // (op 1936) most significant: sum


s_mov_b32 s60, s34 // (op 1936) reg expression
s_mov_b32 s61, s35 // (op 1936) reg expression
s_mov_b32 s62, s0 // (op 1936) reg expression
s_mov_b32 s63, 131072 // (op 1936) call()




buffer_load_ubyte v20, v17, s[56:59], 0 offen offset: 0 // (op 1544) Load value

s_lshl_b64 s[0:1], s[24:25], 2 // (op 1548) 
v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 1548) 


v_mov_b32 v21, v14 // (op 1548) 
buffer_load_dwordx4 v[24:27], v21, s[52:55], 0 offen // (op 1548) Load value
v_add_u32 v21, v21, v22 // (op 1548) 
buffer_load_dwordx4 v[28:31], v21, s[52:55], 0 offen // (op 1548) Load value



v_mov_b64 v[22:23], s[32:33] // (op 1551) convert

v_mov_b32 v21, v16 // (op 1551) 
buffer_load_ubyte v32, v21, s[60:63], 0 offen offset: 0 // (op 1551) Load value
v_add_u32 v21, v21, v22 // (op 1551) 
buffer_load_ubyte v33, v21, s[60:63], 0 offen offset: 0 // (op 1551) Load value





buffer_load_dwordx2 v[22:23], v12, s[48:51], 0 offen // (op 1554) Load value



buffer_load_ubyte v21, v17, s[56:59], 0 offen offset:4 // (op 1558) Load value


s_lshl_b64 s[0:1], s[24:25], 2 // (op 1562) 
v_lshrrev_b64 v[34:35], 3, s[0:1] // (op 1562) 


v_mov_b32 v17, v14 // (op 1562) 
buffer_load_dwordx4 v[36:39], v17, s[52:55], 0 offen offset:64 // (op 1562) Load value
v_add_u32 v17, v17, v34 // (op 1562) 
buffer_load_dwordx4 v[40:43], v17, s[52:55], 0 offen offset:64 // (op 1562) Load value




v_mov_b64 v[34:35], s[32:33] // (op 1565) convert

v_mov_b32 v14, v16 // (op 1565) 
buffer_load_ubyte v44, v14, s[60:63], 0 offen offset:4 // (op 1565) Load value
v_add_u32 v14, v14, v34 // (op 1565) 
buffer_load_ubyte v45, v14, s[60:63], 0 offen offset:4 // (op 1565) Load value






buffer_load_dwordx2 v[16:17], v12, s[48:51], 0 offen offset:64 // (op 1568) Load value

v_and_b32 v12, -4, v1 // (op 2054) 
v_sub_u32 v14, v1, v12 // (op 2054) 

v_lshlrev_b32 v12, 5, v14 // (op 2054) 

v_lshrrev_b32 v14, 2, v1 // (op 2054) 
v_lshl_add_u32 v34, v14, 8, v12 // (op 2054) 


v_lshlrev_b32 v12, 2, v34 // (op 2054) 

v_lshrrev_b32 v14, 3, v12 // (op 2054) 




s_waitcnt vmcnt(10)
ds_write_b128 v14, v[24:27] // (op 1570) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v14, v[28:31] offset:64 // (op 1570) Store local data 

v_and_b32 v12, -4, v1 // (op 2112) 
v_sub_u32 v14, v1, v12 // (op 2112) 

v_lshrrev_b32 v12, 2, v1 // (op 2112) 
v_lshl_add_u32 v34, v12, 3, v14 // (op 2112) 





s_waitcnt vmcnt(8)
ds_write_b8 v34, v32 offset:8192 // (op 1571) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v34, v33 offset:8196 // (op 1571) Store local data 

v_and_b32 v12, -8, v1 // (op 2001) 
v_sub_u32 v14, v1, v12 // (op 2001) 

v_lshlrev_b32 v12, 4, v14 // (op 2001) 

v_lshrrev_b32 v14, 3, v1 // (op 2001) 
v_lshl_add_u32 v34, v14, 7, v12 // (op 2001) 


v_lshlrev_b32 v12, 2, v34 // (op 2001) 

v_lshrrev_b32 v14, 3, v12 // (op 2001) 




s_waitcnt vmcnt(6)
ds_write_b64 v14, v[22:23] offset:8704 // (op 1572) Store local data 

v_and_b32 v12, -8, v1 // (op 1972) 
v_sub_u32 v14, v1, v12 // (op 1972) 

v_lshlrev_b32 v12, 4, v14 // (op 1972) 

v_lshrrev_b32 v14, 3, v1 // (op 1972) 
v_lshl_add_u32 v34, v14, 7, v12 // (op 1972) 


v_lshlrev_b32 v12, 2, v34 // (op 1972) 

v_lshrrev_b32 v14, 3, v12 // (op 1972) 

v_and_b32 v12, -4, v1 // (op 2025) 
v_sub_u32 v34, v1, v12 // (op 2025) 

v_lshlrev_b32 v12, 5, v34 // (op 2025) 

v_lshrrev_b32 v34, 2, v1 // (op 2025) 
v_lshl_add_u32 v35, v34, 8, v12 // (op 2025) 


v_lshlrev_b32 v12, 2, v35 // (op 2025) 

v_lshrrev_b32 v34, 3, v12 // (op 2025) 

v_and_b32 v12, -4, v1 // (op 2083) 
v_sub_u32 v35, v1, v12 // (op 2083) 

v_lshrrev_b32 v12, 2, v1 // (op 2083) 
v_lshl_add_u32 v46, v12, 3, v35 // (op 2083) 


v_and_b32 v12, -8, v1 // (op 2159) 
v_sub_u32 v35, v1, v12 // (op 2159) 

v_lshlrev_b32 v12, 4, v35 // (op 2159) 

v_lshrrev_b32 v35, 3, v1 // (op 2159) 
v_lshl_add_u32 v47, v35, 7, v12 // (op 2159) 


v_lshlrev_b32 v12, 2, v47 // (op 2159) 

v_lshrrev_b32 v35, 3, v12 // (op 2159) 

v_and_b32 v12, -4, v1 // (op 2183) 
v_sub_u32 v47, v1, v12 // (op 2183) 

v_lshlrev_b32 v12, 5, v47 // (op 2183) 

v_lshrrev_b32 v47, 2, v1 // (op 2183) 
v_lshl_add_u32 v48, v47, 8, v12 // (op 2183) 


v_lshlrev_b32 v12, 2, v48 // (op 2183) 

v_lshrrev_b32 v47, 3, v12 // (op 2183) 

v_and_b32 v12, -4, v1 // (op 2212) 
v_sub_u32 v48, v1, v12 // (op 2212) 

v_lshrrev_b32 v12, 2, v1 // (op 2212) 
v_lshl_add_u32 v49, v12, 3, v48 // (op 2212) 


s_mov_b32 s0, 0 // (op 620) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 619) 
s_mov_b32 s1, s6 // (op 619) convert

s_and_b32 s6, s1, 4294967294 // (op 619) 

s_cmp_lt_u32 s0, s6 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 // (op 619) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619:
s_add_u32 s0, s0, 2 // (op 621) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2276) 



ds_read_b128 v[50:53], v3 offset:8704 // (op 65) Load local data 
s_barrier  // (op 2289) 



ds_read_b128 v[54:57], v5 // (op 70) Load local data 
s_barrier  // (op 2303) 



ds_read_u8 v12, v0 offset:8192 // (op 79) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[54:57], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 82) 
ds_read_b128 v[58:61], v5 offset:2048 // (op 339) Load local data 





ds_read_u8 v12, v0 offset:8320 // (op 342) Load local data 



ds_read_b128 v[54:57], v5 offset:4096 // (op 495) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:53], v[58:61], a[4:7], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 343) 





ds_read_u8 v12, v0 offset:8448 // (op 498) Load local data 



ds_read_b128 v[58:61], v5 offset:6144 // (op 651) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:53], v[54:57], a[8:11], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 499) 





ds_read_u8 v12, v0 offset:8576 // (op 654) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:53], v[58:61], a[12:15], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 655) 





v_mov_b32 v12, v18 // (op 75) 

buffer_load_ubyte v20, v12, s[56:59], 0 offen offset:8 // (op 75) Load value

s_lshl_b64 s[6:7], s[24:25], 2 // (op 10) 
v_lshrrev_b64 v[50:51], 3, s[6:7] // (op 10) 


v_mov_b32 v48, v15 // (op 10) 
v_mov_b32 v52, v48 // (op 10) 
buffer_load_dwordx4 v[24:27], v52, s[52:55], 0 offen offset:128 // (op 10) Load value
v_add_u32 v52, v52, v50 // (op 10) 
buffer_load_dwordx4 v[28:31], v52, s[52:55], 0 offen offset:128 // (op 10) Load value



v_mov_b64 v[50:51], s[32:33] // (op 21) convert

v_mov_b32 v52, v19 // (op 21) 
v_mov_b32 v53, v52 // (op 21) 
buffer_load_ubyte v32, v53, s[60:63], 0 offen offset:8 // (op 21) Load value
v_add_u32 v53, v53, v50 // (op 21) 
buffer_load_ubyte v33, v53, s[60:63], 0 offen offset:8 // (op 21) Load value




v_mov_b32 v50, v13 // (op 4) 

buffer_load_dwordx2 v[22:23], v50, s[48:51], 0 offen offset:128 // (op 4) Load value
s_barrier  // (op 2326) 



s_waitcnt vmcnt(10)
ds_write_b128 v47, v[36:39] offset:10752 // (op 235) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v47, v[40:43] offset:10816 // (op 235) Store local data 
s_barrier  // (op 2332) 



s_waitcnt vmcnt(8)
ds_write_b8 v49, v44 offset:18944 // (op 236) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v49, v45 offset:18948 // (op 236) Store local data 
s_barrier  // (op 2320) 



s_waitcnt vmcnt(6)
ds_write_b64 v35, v[16:17] offset:19456 // (op 225) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 2317) 



ds_read_b128 v[54:57], v4 offset:19456 // (op 227) Load local data 
s_barrier  // (op 2323) 
s_barrier  // (op 2369) 



ds_read_b128 v[58:61], v8 offset:10752 // (op 229) Load local data 
s_barrier  // (op 2329) 



ds_read_u8 v51, v7 offset:18944 // (op 233) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[54:57], v[58:61], a[0:3], v21, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 234) 
ds_read_b128 v[62:65], v8 offset:12800 // (op 348) Load local data 





ds_read_u8 v51, v7 offset:19072 // (op 351) Load local data 



ds_read_b128 v[58:61], v8 offset:14848 // (op 504) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[62:65], a[4:7], v21, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 344) 





ds_read_u8 v51, v7 offset:19200 // (op 507) Load local data 



ds_read_b128 v[62:65], v8 offset:16896 // (op 660) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[58:61], a[8:11], v21, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 500) 





s_barrier  // (op 2415) 
ds_read_u8 v51, v7 offset:19328 // (op 663) Load local data 
s_barrier  // (op 2419) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[62:65], a[12:15], v21, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 656) 






buffer_load_ubyte v21, v12, s[56:59], 0 offen offset:12 // (op 231) Load value


v_add_u32 v18, 8, v18 // (op 1862) 
s_lshl_b64 s[6:7], s[24:25], 2 // (op 237) 
v_lshrrev_b64 v[54:55], 3, s[6:7] // (op 237) 


v_mov_b32 v12, v48 // (op 237) 
buffer_load_dwordx4 v[36:39], v12, s[52:55], 0 offen offset:192 // (op 237) Load value
v_add_u32 v12, v12, v54 // (op 237) 
buffer_load_dwordx4 v[40:43], v12, s[52:55], 0 offen offset:192 // (op 237) Load value




v_add_u32 v15, 128, v15 // (op 1798) 
v_mov_b64 v[54:55], s[32:33] // (op 238) convert

v_mov_b32 v12, v52 // (op 238) 
buffer_load_ubyte v44, v12, s[60:63], 0 offen offset:12 // (op 238) Load value
v_add_u32 v12, v12, v54 // (op 238) 
buffer_load_ubyte v45, v12, s[60:63], 0 offen offset:12 // (op 238) Load value






v_add_u32 v19, 8, v19 // (op 1926) 
buffer_load_dwordx2 v[16:17], v50, s[48:51], 0 offen offset:192 // (op 224) Load value

s_barrier  // (op 2292) 
v_add_u32 v13, 128, v13 // (op 1734) 



s_waitcnt vmcnt(10)
ds_write_b128 v34, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(9)
ds_write_b128 v34, v[28:31] offset:64 // (op 13) Store local data 
s_barrier  // (op 2306) 



s_waitcnt vmcnt(8)
ds_write_b8 v46, v32 offset:8192 // (op 26) Store local data 
s_waitcnt vmcnt(7)
ds_write_b8 v46, v33 offset:8196 // (op 26) Store local data 
s_barrier  // (op 2279) 



s_waitcnt vmcnt(6)
ds_write_b64 v14, v[22:23] offset:8704 // (op 6) Store local data 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 619) 
s_mov_b32 s1, s6 // (op 619) convert

s_and_b32 s6, s1, 4294967294 // (op 619) 

s_cmp_lt_u32 s0, s6 // (op 619) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 // (op 619) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoop_619 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoop_619:















v_lshrrev_b32 v12, 3, v1 // (op 1694) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1694) 
v_lshl_add_u32 v13, s16, 5, v12 // (op 1694) 

s_lshr_b32 s6, s0, 1 // (op 1694) 

v_and_b32 v12, -8, v1 // (op 1694) 
v_sub_u32 v14, v1, v12 // (op 1694) 

v_lshlrev_b32 v12, 4, v14 // (op 1694) 

v_lshl_add_u32 v14, s6, 8, v12 // (op 1694) 


v_mov_b32 v19, 0 // (op 1694) convert
v_mov_b32 v18, v14 // (op 1694) convert

v_mul_lo_u32 v15, v13, s13 // (op 1694) most significant: low of low * high
v_mul_hi_u32 v12, v13, s12 // (op 1694) most significant: high of low * low
v_mul_lo_u32 v14, v13, s12 // (op 1694) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1694) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v18 // (op 1694) least significant half
v_addc_co_u32 v15, s[0:1], v15, v19, s[0:1] // (op 1694) most significant half



v_lshlrev_b64 v[12:13], 2, v[14:15] // (op 1694) 

v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 1694) 

v_mov_b32 v12, v14 // (op 1694) convert

s_mul_i32 s1, s12, s23 // (op 1696) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 1696) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 1696) least significant
s_mul_hi_u32 s7, s12, s22 // (op 1696) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1696) most significant: sum
s_add_u32 s1, s1, s7 // (op 1696) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1696) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1696) 

s_mov_b32 s48, s20 // (op 1696) reg expression
s_mov_b32 s49, s21 // (op 1696) reg expression
s_mov_b32 s50, s0 // (op 1696) reg expression
s_mov_b32 s51, 131072 // (op 1696) call()

v_and_b32 v13, -4, v1 // (op 1764) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1764) 
v_sub_u32 v14, v1, v13 // (op 1764) 

s_lshr_b32 s6, s0, 1 // (op 1764) 

v_lshlrev_b32 v13, 5, v14 // (op 1764) 

v_lshl_add_u32 v14, s6, 8, v13 // (op 1764) 


v_mov_b32 v19, 0 // (op 1764) convert
v_mov_b32 v18, v14 // (op 1764) convert

v_lshrrev_b32 v20, 2, v1 // (op 1764) 
v_lshlrev_b32 v13, 1, v20 // (op 1764) 

v_lshl_add_u32 v20, s17, 7, v13 // (op 1764) 

v_mul_lo_u32 v15, v20, s25 // (op 1764) most significant: low of low * high
v_mul_hi_u32 v13, v20, s24 // (op 1764) most significant: high of low * low
v_mul_lo_u32 v14, v20, s24 // (op 1764) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 1764) most significant: sum


v_add_co_u32 v22, s[0:1], v18, v14 // (op 1764) least significant half
v_addc_co_u32 v23, s[0:1], v19, v15, s[0:1] // (op 1764) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 1764) 


v_lshrrev_b64 v[14:15], 3, v[22:23] // (op 1764) 

v_mov_b32 v13, v14 // (op 1764) convert

s_mul_i32 s1, s24, s9 // (op 1766) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 1766) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 1766) least significant
s_mul_hi_u32 s7, s24, s8 // (op 1766) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1766) most significant: sum
s_add_u32 s1, s1, s7 // (op 1766) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 1766) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 1766) 

s_mov_b32 s52, s26 // (op 1766) reg expression
s_mov_b32 s53, s27 // (op 1766) reg expression
s_mov_b32 s54, s0 // (op 1766) reg expression
s_mov_b32 s55, 131072 // (op 1766) call()

v_lshrrev_b32 v14, 7, v1 // (op 1828) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1828) 
v_and_b32 v15, 15, v1 // (op 1828) 
s_lshr_b32 s6, s0, 1 // (op 1828) 

v_lshl_add_u32 v18, v14, 4, v15 // (op 1828) 


v_lshl_add_u32 v14, s16, 5, v18 // (op 1828) 

v_and_b32 v15, 63, v1 // (op 1828) 
v_lshrrev_b32 v18, 4, v15 // (op 1828) 

v_lshl_add_u32 v15, s6, 3, v18 // (op 1828) 


v_mul_lo_u32 v18, v14, s28 // (op 1828) 
v_add_u32 v18, v18, v15 // (op 1828) 


s_mul_i32 s1, s28, s23 // (op 1830) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 1830) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 1830) least significant
s_mul_hi_u32 s7, s28, s22 // (op 1830) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1830) most significant: sum
s_add_u32 s1, s1, s7 // (op 1830) most significant: sum


s_mov_b32 s56, s30 // (op 1830) reg expression
s_mov_b32 s57, s31 // (op 1830) reg expression
s_mov_b32 s58, s0 // (op 1830) reg expression
s_mov_b32 s59, 131072 // (op 1830) call()

v_and_b32 v14, -4, v1 // (op 1892) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 1892) 
v_sub_u32 v15, v1, v14 // (op 1892) 

s_lshr_b32 s6, s0, 1 // (op 1892) 

v_lshrrev_b32 v14, 2, v1 // (op 1892) 
v_lshl_add_u32 v19, s6, 3, v15 // (op 1892) 


v_lshlrev_b32 v15, 1, v14 // (op 1892) 

v_lshl_add_u32 v14, s17, 7, v15 // (op 1892) 

v_mul_lo_u32 v15, v14, s32 // (op 1892) 

v_add_u32 v14, v19, v15 // (op 1892) 


s_mul_i32 s1, s32, s9 // (op 1894) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 1894) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 1894) least significant
s_mul_hi_u32 s7, s32, s8 // (op 1894) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 1894) most significant: sum
s_add_u32 s1, s1, s7 // (op 1894) most significant: sum


s_mov_b32 s60, s34 // (op 1894) reg expression
s_mov_b32 s61, s35 // (op 1894) reg expression
s_mov_b32 s62, s0 // (op 1894) reg expression
s_mov_b32 s63, 131072 // (op 1894) call()

v_and_b32 v15, -8, v1 // (op 1985) 
v_sub_u32 v19, v1, v15 // (op 1985) 

v_lshlrev_b32 v15, 4, v19 // (op 1985) 

v_lshrrev_b32 v19, 3, v1 // (op 1985) 
v_lshl_add_u32 v20, v19, 7, v15 // (op 1985) 


v_lshlrev_b32 v15, 2, v20 // (op 1985) 

v_lshrrev_b32 v19, 3, v15 // (op 1985) 

v_and_b32 v15, -4, v1 // (op 2038) 
v_sub_u32 v20, v1, v15 // (op 2038) 

v_lshlrev_b32 v15, 5, v20 // (op 2038) 

v_lshrrev_b32 v20, 2, v1 // (op 2038) 
v_lshl_add_u32 v22, v20, 8, v15 // (op 2038) 


v_lshlrev_b32 v15, 2, v22 // (op 2038) 

v_lshrrev_b32 v20, 3, v15 // (op 2038) 

v_and_b32 v15, -4, v1 // (op 2096) 
v_sub_u32 v22, v1, v15 // (op 2096) 

v_lshrrev_b32 v15, 2, v1 // (op 2096) 
v_lshl_add_u32 v23, v15, 3, v22 // (op 2096) 


s_ashr_i64 s[0:1], s[36:37], 7 // (op 625) 
s_and_b32 s6, s0, 4294967294 // (op 625) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 624) 
s_mov_b32 s7, s0 // (op 624) convert

s_cmp_lt_u32 s6, s7 // (op 624) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 // (op 624) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624:


s_add_u32 s6, s6, 1 // (op 626) 
v_mov_b32 v15, v12 // (op 166) 

buffer_load_dwordx2 v[16:17], v15, s[48:51], 0 offen // (op 166) Load value

v_add_u32 v12, 64, v12 // (op 1684) 

s_barrier  // (op 2282) 
s_lshl_b64 s[0:1], s[24:25], 2 // (op 179) 



v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 179) 


v_mov_b32 v15, v13 // (op 179) 
v_mov_b32 v21, v15 // (op 179) 
buffer_load_dwordx4 v[26:29], v21, s[52:55], 0 offen // (op 179) Load value
v_add_u32 v21, v21, v24 // (op 179) 
buffer_load_dwordx4 v[30:33], v21, s[52:55], 0 offen // (op 179) Load value



v_add_u32 v13, 64, v13 // (op 1756) 
s_waitcnt vmcnt(2)
ds_write_b64 v19, v[16:17] offset:8704 // (op 167) Store local data 


v_mov_b64 v[16:17], s[32:33] // (op 180) convert

s_barrier  // (op 2295) 
v_mov_b32 v15, v14 // (op 180) 



v_mov_b32 v21, v15 // (op 180) 
buffer_load_ubyte v24, v21, s[60:63], 0 offen offset: 0 // (op 180) Load value
v_add_u32 v21, v21, v16 // (op 180) 
buffer_load_ubyte v25, v21, s[60:63], 0 offen offset: 0 // (op 180) Load value



v_add_u32 v14, 4, v14 // (op 1884) 
s_waitcnt vmcnt(3)
ds_write_b128 v20, v[26:29] // (op 177) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 2285) 
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[30:33] offset:64 // (op 177) Store local data 




ds_read_b128 v[26:29], v3 offset:8704 // (op 169) Load local data 
s_barrier  // (op 2309) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 2299) 






ds_read_b128 v[30:33], v5 // (op 171) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v24 offset:8192 // (op 178) Store local data 


v_mov_b32 v15, v18 // (op 173) 

buffer_load_ubyte v16, v15, s[56:59], 0 offen offset: 0 // (op 173) Load value

v_add_u32 v18, 4, v18 // (op 1820) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v25 offset:8196 // (op 178) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 2313) 



ds_read_u8 v15, v0 offset:8192 // (op 175) Load local data 



ds_read_b128 v[34:37], v5 offset:2048 // (op 320) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:33], a[0:3], v16, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 176) 





ds_read_u8 v15, v0 offset:8320 // (op 323) Load local data 



ds_read_b128 v[30:33], v5 offset:4096 // (op 476) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:29], v[34:37], a[4:7], v16, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 324) 





ds_read_u8 v15, v0 offset:8448 // (op 479) Load local data 



ds_read_b128 v[34:37], v5 offset:6144 // (op 632) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:29], v[30:33], a[8:11], v16, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 480) 





ds_read_u8 v15, v0 offset:8576 // (op 635) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:29], v[34:37], a[12:15], v16, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 636) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 624) 
s_mov_b32 s7, s0 // (op 624) convert

s_cmp_lt_u32 s6, s7 // (op 624) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 // (op 624) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopTop_KLoopTail_624 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ForLoopBottom_KLoopTail_624:






















s_barrier  // (op 2403) 







v_cmp_eq_f32 vcc, 0.00000, v10 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123
v_accvgpr_read v12, a0 // (op 46) 
v_accvgpr_read v13, a1 // (op 46) 
v_accvgpr_read v14, a2 // (op 46) 
v_accvgpr_read v15, a3 // (op 46) 
v_mul_f32 v16, v11, v12 // (op 46) 
v_mul_f32 v17, v11, v13 // (op 46) 
v_mul_f32 v18, v11, v14 // (op 46) 
v_mul_f32 v19, v11, v15 // (op 46) 



v_mov_b32 v0, v9 // (op 52) reg expression

buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 52) Store value


v_accvgpr_read v12, a4 // (op 860) 
v_accvgpr_read v13, a5 // (op 860) 
v_accvgpr_read v14, a6 // (op 860) 
v_accvgpr_read v15, a7 // (op 860) 
v_mul_f32 v16, v11, v12 // (op 860) 
v_mul_f32 v17, v11, v13 // (op 860) 
v_mul_f32 v18, v11, v14 // (op 860) 
v_mul_f32 v19, v11, v15 // (op 860) 



s_lshl_b32 s0, s4, 7 // (op 862) 
v_add_u32 v0, v9, s0 // (op 862) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 862) Store value


v_accvgpr_read v12, a8 // (op 872) 
v_accvgpr_read v13, a9 // (op 872) 
v_accvgpr_read v14, a10 // (op 872) 
v_accvgpr_read v15, a11 // (op 872) 
v_mul_f32 v16, v11, v12 // (op 872) 
v_mul_f32 v17, v11, v13 // (op 872) 
v_mul_f32 v18, v11, v14 // (op 872) 
v_mul_f32 v19, v11, v15 // (op 872) 



s_lshl_b32 s0, s4, 8 // (op 874) 
v_add_u32 v0, v9, s0 // (op 874) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 874) Store value


v_accvgpr_read v12, a12 // (op 884) 
v_accvgpr_read v13, a13 // (op 884) 
v_accvgpr_read v14, a14 // (op 884) 
v_accvgpr_read v15, a15 // (op 884) 
v_mul_f32 v16, v11, v12 // (op 884) 
v_mul_f32 v17, v11, v13 // (op 884) 
v_mul_f32 v18, v11, v14 // (op 884) 
v_mul_f32 v19, v11, v15 // (op 884) 



s_lshl_b32 s0, s4, 7 // (op 886) 
s_mul_i32 s1, 3, s0 // (op 886) 

v_add_u32 v0, v9, s1 // (op 886) 


buffer_store_dwordx4 v[16:19], v0, s[44:47], 0 offen // (op 886) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalFalse_0_beta_123:


v_mov_b32 v0, v6 // (op 135) reg expression

buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 135) Load value

v_accvgpr_read v16, a0 // (op 132) 
v_accvgpr_read v17, a1 // (op 132) 
v_accvgpr_read v18, a2 // (op 132) 
v_accvgpr_read v19, a3 // (op 132) 
v_mul_f32 v20, v11, v16 // (op 132) 
v_mul_f32 v21, v11, v17 // (op 132) 
v_mul_f32 v22, v11, v18 // (op 132) 
v_mul_f32 v23, v11, v19 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 136) 
v_mul_f32 v17, v10, v13 // (op 136) 
v_mul_f32 v18, v10, v14 // (op 136) 
v_mul_f32 v19, v10, v15 // (op 136) 

v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 
v_add_f32 v14, v18, v22 // (op 133) 
v_add_f32 v15, v19, v23 // (op 133) 




v_mov_b32 v0, v9 // (op 134) reg expression

buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 134) Store value




s_lshl_b32 s0, s2, 7 // (op 918) 
v_add_u32 v0, v6, s0 // (op 918) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 918) Load value

v_accvgpr_read v16, a4 // (op 915) 
v_accvgpr_read v17, a5 // (op 915) 
v_accvgpr_read v18, a6 // (op 915) 
v_accvgpr_read v19, a7 // (op 915) 
v_mul_f32 v20, v11, v16 // (op 915) 
v_mul_f32 v21, v11, v17 // (op 915) 
v_mul_f32 v22, v11, v18 // (op 915) 
v_mul_f32 v23, v11, v19 // (op 915) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 919) 
v_mul_f32 v17, v10, v13 // (op 919) 
v_mul_f32 v18, v10, v14 // (op 919) 
v_mul_f32 v19, v10, v15 // (op 919) 

v_add_f32 v12, v16, v20 // (op 916) 
v_add_f32 v13, v17, v21 // (op 916) 
v_add_f32 v14, v18, v22 // (op 916) 
v_add_f32 v15, v19, v23 // (op 916) 




s_lshl_b32 s0, s4, 7 // (op 917) 
v_add_u32 v0, v9, s0 // (op 917) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 917) Store value




s_lshl_b32 s0, s2, 8 // (op 930) 
v_add_u32 v0, v6, s0 // (op 930) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 930) Load value

v_accvgpr_read v16, a8 // (op 927) 
v_accvgpr_read v17, a9 // (op 927) 
v_accvgpr_read v18, a10 // (op 927) 
v_accvgpr_read v19, a11 // (op 927) 
v_mul_f32 v20, v11, v16 // (op 927) 
v_mul_f32 v21, v11, v17 // (op 927) 
v_mul_f32 v22, v11, v18 // (op 927) 
v_mul_f32 v23, v11, v19 // (op 927) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 931) 
v_mul_f32 v17, v10, v13 // (op 931) 
v_mul_f32 v18, v10, v14 // (op 931) 
v_mul_f32 v19, v10, v15 // (op 931) 

v_add_f32 v12, v16, v20 // (op 928) 
v_add_f32 v13, v17, v21 // (op 928) 
v_add_f32 v14, v18, v22 // (op 928) 
v_add_f32 v15, v19, v23 // (op 928) 




s_lshl_b32 s0, s4, 8 // (op 929) 
v_add_u32 v0, v9, s0 // (op 929) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 929) Store value




s_lshl_b32 s0, s2, 7 // (op 942) 
s_mul_i32 s1, 3, s0 // (op 942) 

v_add_u32 v0, v6, s1 // (op 942) 


buffer_load_dwordx4 v[12:15], v0, s[40:43], 0 offen // (op 942) Load value

v_accvgpr_read v16, a12 // (op 939) 
v_accvgpr_read v17, a13 // (op 939) 
v_accvgpr_read v18, a14 // (op 939) 
v_accvgpr_read v19, a15 // (op 939) 
v_mul_f32 v20, v11, v16 // (op 939) 
v_mul_f32 v21, v11, v17 // (op 939) 
v_mul_f32 v22, v11, v18 // (op 939) 
v_mul_f32 v23, v11, v19 // (op 939) 

s_waitcnt vmcnt(0)
v_mul_f32 v16, v10, v12 // (op 943) 
v_mul_f32 v17, v10, v13 // (op 943) 
v_mul_f32 v18, v10, v14 // (op 943) 
v_mul_f32 v19, v10, v15 // (op 943) 

v_add_f32 v12, v16, v20 // (op 940) 
v_add_f32 v13, v17, v21 // (op 940) 
v_add_f32 v14, v18, v22 // (op 940) 
v_add_f32 v15, v19, v23 // (op 940) 




s_lshl_b32 s0, s4, 7 // (op 941) 
s_mul_i32 s1, 3, s0 // (op 941) 

v_add_u32 v0, v9, s1 // (op 941) 


buffer_store_dwordx4 v[12:15], v0, s[44:47], 0 offen // (op 941) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_ConditionalBottom_0_beta_123:












s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 21504 // lds bytes
  .amdhsa_accum_offset 68
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
    DataTypeB: F4
    DestDataType: s
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
    macrotile: [32, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 21504
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 66
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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

