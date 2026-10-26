

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2,@function
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 747) 
v_and_b32 v3, 15, v1 // (op 747) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 747) 


v_and_b32 v0, 63, v1 // (op 747) 
v_lshrrev_b32 v3, 4, v0 // (op 747) 

v_lshlrev_b32 v0, 5, v3 // (op 747) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 747) 


v_mul_lo_u32 v0, v3, 6 // (op 747) 

v_lshrrev_b32 v3, 3, v0 // (op 747) 

v_lshrrev_b32 v0, 6, v1 // (op 800) 
v_and_b32 v4, 1, v0 // (op 800) 

v_and_b32 v0, 15, v1 // (op 800) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 800) 


v_and_b32 v0, 63, v1 // (op 800) 
v_lshrrev_b32 v4, 4, v0 // (op 800) 

v_lshlrev_b32 v0, 5, v4 // (op 800) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 800) 


v_mul_lo_u32 v0, v4, 6 // (op 800) 

v_lshrrev_b32 v4, 3, v0 // (op 800) 

v_lshrrev_b32 v0, 6, v1 // (op 853) 
v_and_b32 v5, 1, v0 // (op 853) 

v_and_b32 v0, 15, v1 // (op 853) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 853) 


v_and_b32 v0, 63, v1 // (op 853) 
v_lshrrev_b32 v5, 4, v0 // (op 853) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 853) 


v_lshrrev_b32 v5, 7, v1 // (op 911) 
v_and_b32 v6, 15, v1 // (op 911) 
v_lshl_add_u32 v7, v5, 4, v6 // (op 911) 


v_and_b32 v5, 63, v1 // (op 911) 
v_lshrrev_b32 v6, 4, v5 // (op 911) 

v_lshlrev_b32 v5, 5, v6 // (op 911) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 911) 


v_mul_lo_u32 v5, v6, 6 // (op 911) 

v_lshrrev_b32 v6, 3, v5 // (op 911) 

v_lshrrev_b32 v5, 6, v1 // (op 935) 
v_and_b32 v7, 1, v5 // (op 935) 

v_and_b32 v5, 15, v1 // (op 935) 
v_lshl_add_u32 v8, v7, 4, v5 // (op 935) 


v_and_b32 v5, 63, v1 // (op 935) 
v_lshrrev_b32 v7, 4, v5 // (op 935) 

v_lshlrev_b32 v5, 5, v7 // (op 935) 

v_lshl_add_u32 v7, v8, 7, v5 // (op 935) 


v_mul_lo_u32 v5, v7, 6 // (op 935) 

v_lshrrev_b32 v7, 3, v5 // (op 935) 

v_lshrrev_b32 v5, 6, v1 // (op 959) 
v_and_b32 v8, 1, v5 // (op 959) 

v_and_b32 v5, 15, v1 // (op 959) 
v_lshl_add_u32 v9, v8, 4, v5 // (op 959) 


v_and_b32 v5, 63, v1 // (op 959) 
v_lshrrev_b32 v8, 4, v5 // (op 959) 

v_lshl_add_u32 v5, v9, 2, v8 // (op 959) 


v_mov_b32 v8, s14 // (op 2) Move value
v_mov_b32 v9, s15 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_lshrrev_b32 v10, 3, v1 // (op 457) 
v_lshl_add_u32 v11, s16, 4, v10 // (op 457) 

v_and_b32 v10, -8, v1 // (op 457) 
v_sub_u32 v12, v1, v10 // (op 457) 

v_lshlrev_b32 v10, 4, v12 // (op 457) 

v_mov_b32 v13, 0 // (op 457) convert
v_mov_b32 v12, v10 // (op 457) convert

v_mul_lo_u32 v15, v11, s3 // (op 457) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 457) most significant: high of low * low
v_mul_lo_u32 v14, v11, s2 // (op 457) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 457) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 457) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 457) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 457) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 457) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 457) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 457) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 457) 

v_mov_b32 v10, v12 // (op 457) convert

s_mul_i32 s1, s7, 6 // (op 459) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 459) least significant
s_mul_hi_u32 s14, s6, 6 // (op 459) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 459) most significant: sum

s_mul_i32 s15, s2, s1 // (op 459) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 459) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 459) least significant
s_mul_hi_u32 s19, s2, s0 // (op 459) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 459) most significant: sum
s_add_u32 s15, s15, s19 // (op 459) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 459) 

s_mov_b32 s40, s4 // (op 459) reg expression
s_mov_b32 s41, s5 // (op 459) reg expression
s_mov_b32 s42, s0 // (op 459) reg expression
s_mov_b32 s43, 131072 // (op 459) call()

v_lshrrev_b32 v11, 3, v1 // (op 476) 
v_lshl_add_u32 v12, s16, 4, v11 // (op 476) 

v_and_b32 v11, -8, v1 // (op 476) 
v_sub_u32 v13, v1, v11 // (op 476) 

v_lshlrev_b32 v11, 4, v13 // (op 476) 

v_mov_b32 v15, 0 // (op 476) convert
v_mov_b32 v14, v11 // (op 476) convert

v_mul_lo_u32 v17, v12, s3 // (op 476) most significant: low of low * high
v_mul_hi_u32 v11, v12, s2 // (op 476) most significant: high of low * low
v_mul_lo_u32 v16, v12, s2 // (op 476) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 476) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 476) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 476) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 476) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 476) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 476) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 476) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 476) 

v_mov_b32 v11, v12 // (op 476) convert

s_mul_i32 s1, s7, 6 // (op 478) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 478) least significant
s_mul_hi_u32 s14, s6, 6 // (op 478) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 478) most significant: sum

s_mul_i32 s15, s2, s1 // (op 478) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 478) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 478) least significant
s_mul_hi_u32 s19, s2, s0 // (op 478) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 478) most significant: sum
s_add_u32 s15, s15, s19 // (op 478) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 478) 

s_mov_b32 s40, s4 // (op 478) reg expression
s_mov_b32 s41, s5 // (op 478) reg expression
s_mov_b32 s42, s0 // (op 478) reg expression
s_mov_b32 s43, 131072 // (op 478) call()

v_and_b32 v12, -4, v1 // (op 521) 
v_sub_u32 v13, v1, v12 // (op 521) 

v_lshlrev_b32 v12, 5, v13 // (op 521) 

v_mov_b32 v15, 0 // (op 521) convert
v_mov_b32 v14, v12 // (op 521) convert

v_lshrrev_b32 v12, 2, v1 // (op 521) 
v_lshl_add_u32 v13, s17, 5, v12 // (op 521) 

v_mul_lo_u32 v17, v13, s9 // (op 521) most significant: low of low * high
v_mul_hi_u32 v12, v13, s8 // (op 521) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 521) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 521) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 521) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 521) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 521) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 521) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 521) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 521) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 521) 

v_mov_b32 v14, v12 // (op 521) convert

s_mul_i32 s1, s13, 6 // (op 523) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 523) least significant
s_mul_hi_u32 s14, s12, 6 // (op 523) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 523) most significant: sum

s_mul_i32 s15, s8, s1 // (op 523) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 523) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 523) least significant
s_mul_hi_u32 s19, s8, s0 // (op 523) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 523) most significant: sum
s_add_u32 s15, s15, s19 // (op 523) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 523) 

s_mov_b32 s44, s10 // (op 523) reg expression
s_mov_b32 s45, s11 // (op 523) reg expression
s_mov_b32 s46, s0 // (op 523) reg expression
s_mov_b32 s47, 131072 // (op 523) call()

v_and_b32 v12, -4, v1 // (op 540) 
v_sub_u32 v13, v1, v12 // (op 540) 

v_lshlrev_b32 v12, 5, v13 // (op 540) 

v_mov_b32 v17, 0 // (op 540) convert
v_mov_b32 v16, v12 // (op 540) convert

v_lshrrev_b32 v15, 2, v1 // (op 540) 
v_lshl_add_u32 v12, s17, 5, v15 // (op 540) 

v_mul_lo_u32 v19, v12, s9 // (op 540) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 540) most significant: high of low * low
v_mul_lo_u32 v18, v12, s8 // (op 540) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 540) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 540) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 540) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 540) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 540) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 540) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 540) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 540) 

v_mov_b32 v15, v12 // (op 540) convert

s_mul_i32 s1, s13, 6 // (op 542) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 542) least significant
s_mul_hi_u32 s14, s12, 6 // (op 542) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 542) most significant: sum

s_mul_i32 s15, s8, s1 // (op 542) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 542) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 542) least significant
s_mul_hi_u32 s19, s8, s0 // (op 542) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 542) most significant: sum
s_add_u32 s15, s15, s19 // (op 542) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 542) 

s_mov_b32 s44, s10 // (op 542) reg expression
s_mov_b32 s45, s11 // (op 542) reg expression
s_mov_b32 s46, s0 // (op 542) reg expression
s_mov_b32 s47, 131072 // (op 542) call()

v_lshrrev_b32 v12, 7, v1 // (op 585) 
v_and_b32 v13, 15, v1 // (op 585) 
v_lshl_add_u32 v16, v12, 4, v13 // (op 585) 


v_lshl_add_u32 v12, s16, 4, v16 // (op 585) 

v_and_b32 v13, 63, v1 // (op 585) 
v_lshrrev_b32 v16, 4, v13 // (op 585) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v13, v12, s20 // (op 585) 
v_add_u32 v13, v13, v16 // (op 585) 


s_mul_i32 s1, s20, s7 // (op 587) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 587) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 587) least significant
s_mul_hi_u32 s15, s20, s6 // (op 587) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 587) most significant: sum
s_add_u32 s1, s1, s15 // (op 587) most significant: sum


s_mov_b32 s48, s22 // (op 587) reg expression
s_mov_b32 s49, s23 // (op 587) reg expression
s_mov_b32 s50, s0 // (op 587) reg expression
s_mov_b32 s51, 131072 // (op 587) call()

v_lshrrev_b32 v12, 7, v1 // (op 604) 
v_and_b32 v16, 15, v1 // (op 604) 
v_lshl_add_u32 v17, v12, 4, v16 // (op 604) 


v_lshl_add_u32 v12, s16, 4, v17 // (op 604) 

v_and_b32 v16, 63, v1 // (op 604) 
v_lshrrev_b32 v17, 4, v16 // (op 604) 

v_mul_lo_u32 v16, v12, s20 // (op 604) 
v_add_u32 v16, v16, v17 // (op 604) 


s_mul_i32 s1, s20, s7 // (op 606) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 606) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 606) least significant
s_mul_hi_u32 s15, s20, s6 // (op 606) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 606) most significant: sum
s_add_u32 s1, s1, s15 // (op 606) most significant: sum


s_mov_b32 s48, s22 // (op 606) reg expression
s_mov_b32 s49, s23 // (op 606) reg expression
s_mov_b32 s50, s0 // (op 606) reg expression
s_mov_b32 s51, 131072 // (op 606) call()

v_and_b32 v12, -4, v1 // (op 649) 
v_sub_u32 v17, v1, v12 // (op 649) 

v_lshrrev_b32 v12, 2, v1 // (op 649) 
v_lshl_add_u32 v18, s17, 5, v12 // (op 649) 

v_mul_lo_u32 v12, v18, s24 // (op 649) 

v_add_u32 v18, v17, v12 // (op 649) 


s_mul_i32 s1, s24, s13 // (op 651) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 651) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 651) least significant
s_mul_hi_u32 s15, s24, s12 // (op 651) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 651) most significant: sum
s_add_u32 s1, s1, s15 // (op 651) most significant: sum


s_mov_b32 s52, s26 // (op 651) reg expression
s_mov_b32 s53, s27 // (op 651) reg expression
s_mov_b32 s54, s0 // (op 651) reg expression
s_mov_b32 s55, 131072 // (op 651) call()

v_and_b32 v12, -4, v1 // (op 668) 
v_sub_u32 v17, v1, v12 // (op 668) 

v_lshrrev_b32 v12, 2, v1 // (op 668) 
v_lshl_add_u32 v19, s17, 5, v12 // (op 668) 

v_mul_lo_u32 v12, v19, s24 // (op 668) 

v_add_u32 v19, v17, v12 // (op 668) 


s_mul_i32 s1, s24, s13 // (op 670) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 670) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 670) least significant
s_mul_hi_u32 s15, s24, s12 // (op 670) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 670) most significant: sum
s_add_u32 s1, s1, s15 // (op 670) most significant: sum


s_mov_b32 s52, s26 // (op 670) reg expression
s_mov_b32 s53, s27 // (op 670) reg expression
s_mov_b32 s54, s0 // (op 670) reg expression
s_mov_b32 s55, 131072 // (op 670) call()




buffer_load_ubyte v12, v13, s[48:51], 0 offen offset: 0 // (op 341) Load value



buffer_load_dwordx4 v[20:23], v14, s[44:47], 0 offen // (op 345) Load value
buffer_load_dwordx2 v[24:25], v14, s[44:47], 0 offen offset:16 // (op 345) Load value



buffer_load_ubyte v17, v18, s[52:55], 0 offen offset: 0 // (op 348) Load value



buffer_load_dwordx3 v[26:28], v10, s[40:43], 0 offen // (op 351) Load value



buffer_load_ubyte v29, v13, s[48:51], 0 offen offset:4 // (op 355) Load value




buffer_load_dwordx4 v[30:33], v14, s[44:47], 0 offen offset:96 // (op 359) Load value
buffer_load_dwordx2 v[34:35], v14, s[44:47], 0 offen offset:112 // (op 359) Load value




buffer_load_ubyte v13, v18, s[52:55], 0 offen offset:4 // (op 362) Load value




buffer_load_dwordx3 v[36:38], v10, s[40:43], 0 offen offset:96 // (op 365) Load value

v_and_b32 v10, -4, v1 // (op 788) 
v_sub_u32 v14, v1, v10 // (op 788) 

v_lshlrev_b32 v10, 5, v14 // (op 788) 

v_lshrrev_b32 v14, 2, v1 // (op 788) 
v_lshl_add_u32 v18, v14, 7, v10 // (op 788) 


v_mul_lo_u32 v10, v18, 6 // (op 788) 

v_lshrrev_b32 v14, 3, v10 // (op 788) 




s_waitcnt vmcnt(8)
ds_write_b128 v14, v[20:23] // (op 367) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v14, v[24:25] offset:16 // (op 367) Store local data 

v_and_b32 v10, -4, v1 // (op 841) 
v_sub_u32 v14, v1, v10 // (op 841) 

v_and_b32 v10, 4294967292, v1 // (op 841) 
v_add_u32 v18, v10, v14 // (op 841) 





s_waitcnt vmcnt(6)
ds_write_b8 v18, v17 offset:3072 // (op 368) Store local data 

v_and_b32 v10, -8, v1 // (op 735) 
v_sub_u32 v14, v1, v10 // (op 735) 

v_lshlrev_b32 v10, 4, v14 // (op 735) 

v_lshrrev_b32 v14, 3, v1 // (op 735) 
v_lshl_add_u32 v18, v14, 7, v10 // (op 735) 


v_mul_lo_u32 v10, v18, 6 // (op 735) 

v_lshrrev_b32 v14, 3, v10 // (op 735) 




s_waitcnt vmcnt(5)
ds_write_b96 v14, v[26:28] offset:3200 // (op 369) Store local data 

v_and_b32 v10, -8, v1 // (op 706) 
v_sub_u32 v14, v1, v10 // (op 706) 

v_lshlrev_b32 v10, 4, v14 // (op 706) 

v_lshrrev_b32 v14, 3, v1 // (op 706) 
v_lshl_add_u32 v18, v14, 7, v10 // (op 706) 


v_mul_lo_u32 v10, v18, 6 // (op 706) 

v_lshrrev_b32 v14, 3, v10 // (op 706) 

v_and_b32 v10, -4, v1 // (op 759) 
v_sub_u32 v18, v1, v10 // (op 759) 

v_lshlrev_b32 v10, 5, v18 // (op 759) 

v_lshrrev_b32 v18, 2, v1 // (op 759) 
v_lshl_add_u32 v39, v18, 7, v10 // (op 759) 


v_mul_lo_u32 v10, v39, 6 // (op 759) 

v_lshrrev_b32 v18, 3, v10 // (op 759) 

v_and_b32 v10, -4, v1 // (op 812) 
v_sub_u32 v39, v1, v10 // (op 812) 

v_and_b32 v10, 4294967292, v1 // (op 812) 
v_add_u32 v40, v10, v39 // (op 812) 


v_and_b32 v10, -8, v1 // (op 899) 
v_sub_u32 v39, v1, v10 // (op 899) 

v_lshlrev_b32 v10, 4, v39 // (op 899) 

v_lshrrev_b32 v39, 3, v1 // (op 899) 
v_lshl_add_u32 v41, v39, 7, v10 // (op 899) 


v_mul_lo_u32 v10, v41, 6 // (op 899) 

v_lshrrev_b32 v39, 3, v10 // (op 899) 

v_and_b32 v10, -4, v1 // (op 923) 
v_sub_u32 v41, v1, v10 // (op 923) 

v_lshlrev_b32 v10, 5, v41 // (op 923) 

v_lshrrev_b32 v41, 2, v1 // (op 923) 
v_lshl_add_u32 v42, v41, 7, v10 // (op 923) 


v_mul_lo_u32 v10, v42, 6 // (op 923) 

v_lshrrev_b32 v41, 3, v10 // (op 923) 

v_and_b32 v10, -4, v1 // (op 947) 
v_sub_u32 v42, v1, v10 // (op 947) 

v_and_b32 v10, 4294967292, v1 // (op 947) 
v_add_u32 v43, v10, v42 // (op 947) 


s_mov_b32 s0, 0 // (op 57) call()
s_ashr_i64 s[14:15], s[28:29], 7 // (op 56) 
s_mov_b32 s1, s14 // (op 56) convert

s_and_b32 s14, s1, 4294967294 // (op 56) 

s_cmp_lt_u32 s0, s14 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 // (op 56) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56 if false)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56:
s_add_u32 s0, s0, 2 // (op 58) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 992) 



ds_read_b128 v[44:47], v3 offset:3200 // (op 67) Load local data 
ds_read_b64 v[48:49], v3 offset:3216 // (op 67) Load local data 
s_barrier  // (op 1005) 



ds_read_b128 v[50:53], v4 // (op 72) Load local data 
ds_read_b64 v[54:55], v4 offset:16 // (op 72) Load local data 
s_barrier  // (op 1019) 



ds_read_u8 v10, v0 offset:3072 // (op 81) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:49], v[50:55], a[0:3], v12, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 84) 





v_mov_b32 v10, v16 // (op 77) 

buffer_load_ubyte v12, v10, s[48:51], 0 offen offset:8 // (op 77) Load value


v_mov_b32 v42, v15 // (op 10) 

buffer_load_dwordx4 v[20:23], v42, s[44:47], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[24:25], v42, s[44:47], 0 offen offset:208 // (op 10) Load value


v_mov_b32 v44, v19 // (op 21) 

buffer_load_ubyte v17, v44, s[52:55], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v45, v11 // (op 4) 

buffer_load_dwordx3 v[26:28], v45, s[40:43], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 1042) 



s_waitcnt vmcnt(8)
ds_write_b128 v41, v[30:33] offset:4736 // (op 239) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v41, v[34:35] offset:4752 // (op 239) Store local data 
s_barrier  // (op 1048) 



s_waitcnt vmcnt(6)
ds_write_b8 v43, v13 offset:7808 // (op 240) Store local data 
s_barrier  // (op 1036) 



s_waitcnt vmcnt(5)
ds_write_b96 v39, v[36:38] offset:7936 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1033) 



ds_read_b128 v[46:49], v6 offset:7936 // (op 231) Load local data 
ds_read_b64 v[50:51], v6 offset:7952 // (op 231) Load local data 
s_barrier  // (op 1039) 
s_barrier  // (op 1103) 



ds_read_b128 v[52:55], v7 offset:4736 // (op 233) Load local data 
ds_read_b64 v[56:57], v7 offset:4752 // (op 233) Load local data 
s_barrier  // (op 1045) 
s_barrier  // (op 1109) 



ds_read_u8 v58, v5 offset:7808 // (op 237) Load local data 
s_barrier  // (op 1117) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:51], v[52:57], a[0:3], v29, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 238) 






buffer_load_ubyte v29, v10, s[48:51], 0 offen offset:12 // (op 235) Load value




v_add_u32 v16, 8, v16 // (op 596) 
buffer_load_dwordx4 v[30:33], v42, s[44:47], 0 offen offset:288 // (op 241) Load value
buffer_load_dwordx2 v[34:35], v42, s[44:47], 0 offen offset:304 // (op 241) Load value




v_add_u32 v15, 192, v15 // (op 532) 
buffer_load_ubyte v13, v44, s[52:55], 0 offen offset:12 // (op 242) Load value




v_add_u32 v19, 8, v19 // (op 660) 
buffer_load_dwordx3 v[36:38], v45, s[40:43], 0 offen offset:288 // (op 228) Load value

s_barrier  // (op 1008) 
v_add_u32 v11, 192, v11 // (op 468) 



s_waitcnt vmcnt(8)
ds_write_b128 v18, v[20:23] // (op 13) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v18, v[24:25] offset:16 // (op 13) Store local data 
s_barrier  // (op 1022) 



s_waitcnt vmcnt(6)
ds_write_b8 v40, v17 offset:3072 // (op 26) Store local data 
s_barrier  // (op 995) 



s_waitcnt vmcnt(5)
ds_write_b96 v14, v[26:28] offset:3200 // (op 6) Store local data 
s_ashr_i64 s[14:15], s[28:29], 7 // (op 56) 
s_mov_b32 s1, s14 // (op 56) convert

s_and_b32 s14, s1, 4294967294 // (op 56) 

s_cmp_lt_u32 s0, s14 // (op 56) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 // (op 56) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoop_56 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoop_56:















v_lshrrev_b32 v10, 3, v1 // (op 431) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 431) 
v_lshl_add_u32 v11, s16, 4, v10 // (op 431) 

s_lshr_b32 s14, s0, 1 // (op 431) 

v_and_b32 v10, -8, v1 // (op 431) 
v_sub_u32 v12, v1, v10 // (op 431) 

v_lshlrev_b32 v10, 4, v12 // (op 431) 

v_lshl_add_u32 v12, s14, 8, v10 // (op 431) 


v_mov_b32 v15, 0 // (op 431) convert
v_mov_b32 v14, v12 // (op 431) convert

v_mul_lo_u32 v17, v11, s3 // (op 431) most significant: low of low * high
v_mul_hi_u32 v10, v11, s2 // (op 431) most significant: high of low * low
v_mul_lo_u32 v16, v11, s2 // (op 431) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 431) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 431) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 431) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 431) most significant: low of high * low
v_mul_hi_u32 v10, v16, 6 // (op 431) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 431) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 431) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 431) 

v_mov_b32 v12, v10 // (op 431) convert

s_mul_i32 s1, s7, 6 // (op 433) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 433) least significant
s_mul_hi_u32 s14, s6, 6 // (op 433) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 433) most significant: sum

s_mul_i32 s15, s2, s1 // (op 433) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 433) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 433) least significant
s_mul_hi_u32 s19, s2, s0 // (op 433) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 433) most significant: sum
s_add_u32 s15, s15, s19 // (op 433) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 433) 

s_mov_b32 s40, s4 // (op 433) reg expression
s_mov_b32 s41, s5 // (op 433) reg expression
s_mov_b32 s42, s0 // (op 433) reg expression
s_mov_b32 s43, 131072 // (op 433) call()

v_and_b32 v10, -4, v1 // (op 498) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 498) 
v_sub_u32 v11, v1, v10 // (op 498) 

s_lshr_b32 s14, s0, 1 // (op 498) 

v_lshlrev_b32 v10, 5, v11 // (op 498) 

v_lshl_add_u32 v11, s14, 8, v10 // (op 498) 


v_mov_b32 v15, 0 // (op 498) convert
v_mov_b32 v14, v11 // (op 498) convert

v_lshrrev_b32 v10, 2, v1 // (op 498) 
v_lshl_add_u32 v11, s17, 5, v10 // (op 498) 

v_mul_lo_u32 v17, v11, s9 // (op 498) most significant: low of low * high
v_mul_hi_u32 v10, v11, s8 // (op 498) most significant: high of low * low
v_mul_lo_u32 v16, v11, s8 // (op 498) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 498) most significant: sum


v_add_co_u32 v10, s[0:1], v14, v16 // (op 498) least significant half
v_addc_co_u32 v11, s[0:1], v15, v17, s[0:1] // (op 498) most significant half



v_mul_lo_u32 v15, v11, 6 // (op 498) most significant: low of high * low
v_mul_hi_u32 v16, v10, 6 // (op 498) most significant: high of low * low
v_mul_lo_u32 v14, v10, 6 // (op 498) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 498) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 498) 

v_mov_b32 v14, v10 // (op 498) convert

s_mul_i32 s1, s13, 6 // (op 500) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 500) least significant
s_mul_hi_u32 s14, s12, 6 // (op 500) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 500) most significant: sum

s_mul_i32 s15, s8, s1 // (op 500) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 500) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 500) least significant
s_mul_hi_u32 s19, s8, s0 // (op 500) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 500) most significant: sum
s_add_u32 s15, s15, s19 // (op 500) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 500) 

s_mov_b32 s44, s10 // (op 500) reg expression
s_mov_b32 s45, s11 // (op 500) reg expression
s_mov_b32 s46, s0 // (op 500) reg expression
s_mov_b32 s47, 131072 // (op 500) call()

v_lshrrev_b32 v10, 7, v1 // (op 562) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 562) 
v_and_b32 v11, 15, v1 // (op 562) 
s_lshr_b32 s14, s0, 1 // (op 562) 

v_lshl_add_u32 v15, v10, 4, v11 // (op 562) 


v_lshl_add_u32 v10, s16, 4, v15 // (op 562) 

v_and_b32 v11, 63, v1 // (op 562) 
v_lshrrev_b32 v15, 4, v11 // (op 562) 

v_lshl_add_u32 v11, s14, 3, v15 // (op 562) 


v_mul_lo_u32 v15, v10, s20 // (op 562) 
v_add_u32 v15, v15, v11 // (op 562) 


s_mul_i32 s1, s20, s7 // (op 564) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 564) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 564) least significant
s_mul_hi_u32 s15, s20, s6 // (op 564) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 564) most significant: sum
s_add_u32 s1, s1, s15 // (op 564) most significant: sum


s_mov_b32 s48, s22 // (op 564) reg expression
s_mov_b32 s49, s23 // (op 564) reg expression
s_mov_b32 s50, s0 // (op 564) reg expression
s_mov_b32 s51, 131072 // (op 564) call()

v_and_b32 v10, -4, v1 // (op 626) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 626) 
v_sub_u32 v11, v1, v10 // (op 626) 

s_lshr_b32 s14, s0, 1 // (op 626) 

v_lshrrev_b32 v10, 2, v1 // (op 626) 
v_lshl_add_u32 v16, s14, 3, v11 // (op 626) 


v_lshl_add_u32 v11, s17, 5, v10 // (op 626) 

v_mul_lo_u32 v10, v11, s24 // (op 626) 

v_add_u32 v11, v16, v10 // (op 626) 


s_mul_i32 s1, s24, s13 // (op 628) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 628) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 628) least significant
s_mul_hi_u32 s15, s24, s12 // (op 628) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 628) most significant: sum
s_add_u32 s1, s1, s15 // (op 628) most significant: sum


s_mov_b32 s52, s26 // (op 628) reg expression
s_mov_b32 s53, s27 // (op 628) reg expression
s_mov_b32 s54, s0 // (op 628) reg expression
s_mov_b32 s55, 131072 // (op 628) call()

v_and_b32 v10, -8, v1 // (op 719) 
v_sub_u32 v16, v1, v10 // (op 719) 

v_lshlrev_b32 v10, 4, v16 // (op 719) 

v_lshrrev_b32 v16, 3, v1 // (op 719) 
v_lshl_add_u32 v17, v16, 7, v10 // (op 719) 


v_mul_lo_u32 v10, v17, 6 // (op 719) 

v_lshrrev_b32 v16, 3, v10 // (op 719) 

v_and_b32 v10, -4, v1 // (op 772) 
v_sub_u32 v17, v1, v10 // (op 772) 

v_lshlrev_b32 v10, 5, v17 // (op 772) 

v_lshrrev_b32 v17, 2, v1 // (op 772) 
v_lshl_add_u32 v18, v17, 7, v10 // (op 772) 


v_mul_lo_u32 v10, v18, 6 // (op 772) 

v_lshrrev_b32 v17, 3, v10 // (op 772) 

v_and_b32 v10, -4, v1 // (op 825) 
v_sub_u32 v18, v1, v10 // (op 825) 

v_and_b32 v10, 4294967292, v1 // (op 825) 
v_add_u32 v19, v10, v18 // (op 825) 


s_ashr_i64 s[0:1], s[28:29], 7 // (op 166) 
s_and_b32 s14, s0, 4294967294 // (op 166) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 165) 
s_mov_b32 s15, s0 // (op 165) convert

s_cmp_lt_u32 s14, s15 // (op 165) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 // (op 165) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165:


s_add_u32 s14, s14, 1 // (op 167) 
v_mov_b32 v10, v12 // (op 170) 

buffer_load_dwordx3 v[20:22], v10, s[40:43], 0 offen // (op 170) Load value

v_add_u32 v12, 96, v12 // (op 421) 


s_barrier  // (op 998) 
v_mov_b32 v10, v14 // (op 183) 




buffer_load_dwordx4 v[24:27], v10, s[44:47], 0 offen // (op 183) Load value
buffer_load_dwordx2 v[28:29], v10, s[44:47], 0 offen offset:16 // (op 183) Load value

v_add_u32 v14, 96, v14 // (op 490) 
s_waitcnt vmcnt(2)
ds_write_b96 v16, v[20:22] offset:3200 // (op 171) Store local data 



v_mov_b32 v10, v11 // (op 184) 

s_barrier  // (op 1011) 
buffer_load_ubyte v13, v10, s[52:55], 0 offen offset: 0 // (op 184) Load value




v_add_u32 v11, 4, v11 // (op 618) 
s_waitcnt vmcnt(2)
ds_write_b128 v17, v[24:27] // (op 181) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1001) 
s_waitcnt vmcnt(1)
ds_write_b64 v17, v[28:29] offset:16 // (op 181) Store local data 




ds_read_b128 v[20:23], v3 offset:3200 // (op 173) Load local data 
s_barrier  // (op 1025) 
ds_read_b64 v[24:25], v3 offset:3216 // (op 173) Load local data 



s_waitcnt lgkmcnt(2)
s_barrier  // (op 1015) 
s_waitcnt vmcnt(0)
ds_write_b8 v19, v13 offset:3072 // (op 182) Store local data 




ds_read_b128 v[26:29], v4 // (op 175) Load local data 
ds_read_b64 v[30:31], v4 offset:16 // (op 175) Load local data 


v_mov_b32 v10, v15 // (op 177) 

buffer_load_ubyte v13, v10, s[48:51], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v15, 4, v15 // (op 554) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1029) 



ds_read_u8 v10, v0 offset:3072 // (op 179) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:25], v[26:31], a[0:3], v13, v10 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 180) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 165) 
s_mov_b32 s15, s0 // (op 165) convert

s_cmp_lt_u32 s14, s15 // (op 165) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 // (op 165) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopTop_KLoopTail_165 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ForLoopBottom_KLoopTail_165:






















s_barrier  // (op 1097) 






v_cmp_eq_f32 vcc, 0.00000, v8 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125
s_nop 0
v_accvgpr_read v4, a0 // (op 46) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 46) 
v_accvgpr_read v6, a2 // (op 46) 
v_accvgpr_read v7, a3 // (op 46) 
v_mul_f32 v10, v9, v4 // (op 46) 
v_mul_f32 v11, v9, v5 // (op 46) 
v_mul_f32 v12, v9, v6 // (op 46) 
v_mul_f32 v13, v9, v7 // (op 46) 

v_cvt_f16_f32 v4, v10 // (op 52) 
v_cvt_f16_f32 v5, v11 // (op 52) 
v_cvt_f16_f32 v6, v12 // (op 52) 
v_cvt_f16_f32 v7, v13 // (op 52) 

v_lshrrev_b32 v0, 7, v1 // (op 867) 
v_and_b32 v3, 63, v1 // (op 867) 
v_lshrrev_b32 v10, 4, v3 // (op 867) 
v_lshlrev_b32 v11, 2, v10 // (op 867) 

v_lshl_add_u32 v10, v0, 4, v11 // (op 867) 


v_lshl_add_u32 v0, s16, 4, v10 // (op 867) 

v_lshrrev_b32 v10, 2, v3 // (op 867) 

v_and_b32 v3, 3, v10 // (op 867) 

v_and_b32 v10, 3, v1 // (op 867) 
v_lshl_add_u32 v11, v3, 2, v10 // (op 867) 


v_lshrrev_b32 v3, 6, v1 // (op 867) 
v_and_b32 v10, 1, v3 // (op 867) 

v_lshl_add_u32 v3, v10, 4, v11 // (op 867) 


v_lshl_add_u32 v10, s17, 5, v3 // (op 867) 

v_mul_lo_u32 v3, v10, s30 // (op 867) 

v_add_lshl_u32 v10, v0, v3, 1 // (op 867) 


s_mul_i32 s15, s30, s13 // (op 869) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 869) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 869) least significant
s_mul_hi_u32 s1, s30, s12 // (op 869) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 869) most significant: sum
s_add_u32 s15, s15, s1 // (op 869) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 869) 

s_mov_b32 s8, s32 // (op 869) reg expression
s_mov_b32 s9, s33 // (op 869) reg expression
s_mov_b32 s10, s0 // (op 869) reg expression
s_mov_b32 s11, 131072 // (op 869) call()


v_pack_b32_f16 v12, v4, v5 // (op 54) 
v_pack_b32_f16 v13, v6, v7 // (op 54) 


buffer_store_dwordx2 v[12:13], v10, s[8:11], 0 offen // (op 54) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalFalse_0_beta_125:
v_lshrrev_b32 v0, 7, v1 // (op 689) 
v_and_b32 v3, 63, v1 // (op 689) 
v_lshrrev_b32 v4, 4, v3 // (op 689) 
v_lshlrev_b32 v5, 2, v4 // (op 689) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 689) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 689) 

v_lshrrev_b32 v4, 2, v3 // (op 689) 

v_and_b32 v3, 3, v4 // (op 689) 

v_and_b32 v4, 3, v1 // (op 689) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 689) 


v_lshrrev_b32 v3, 6, v1 // (op 689) 
v_and_b32 v4, 1, v3 // (op 689) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 689) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 689) 

v_mul_lo_u32 v3, v4, s34 // (op 689) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 689) 


s_mul_i32 s15, s34, s13 // (op 691) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 691) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 691) least significant
s_mul_hi_u32 s1, s34, s12 // (op 691) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 691) most significant: sum
s_add_u32 s15, s15, s1 // (op 691) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 691) 

s_mov_b32 s4, s36 // (op 691) reg expression
s_mov_b32 s5, s37 // (op 691) reg expression
s_mov_b32 s6, s0 // (op 691) reg expression
s_mov_b32 s7, 131072 // (op 691) call()





buffer_load_dwordx2 v[6:7], v4, s[4:7], 0 offen // (op 138) Load value



v_accvgpr_read v10, a0 // (op 134) 
v_accvgpr_read v11, a1 // (op 134) 
v_accvgpr_read v12, a2 // (op 134) 
v_accvgpr_read v13, a3 // (op 134) 
v_mul_f32 v14, v9, v10 // (op 134) 
v_mul_f32 v15, v9, v11 // (op 134) 
v_mul_f32 v16, v9, v12 // (op 134) 
v_mul_f32 v17, v9, v13 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v4, v6 // (op 139) 
v_lshrrev_b32 v5, 16, v6 // (op 139) 
v_cvt_f32_f16 v5, v5 // (op 139) 
v_mul_f32 v10, v8, v4 // (op 139) 
v_mul_f32 v11, v8, v5 // (op 139) 
v_cvt_f32_f16 v4, v7 // (op 139) 
v_lshrrev_b32 v5, 16, v7 // (op 139) 
v_cvt_f32_f16 v5, v5 // (op 139) 
v_mul_f32 v12, v8, v4 // (op 139) 
v_mul_f32 v13, v8, v5 // (op 139) 


v_add_f32 v4, v10, v14 // (op 135) 
v_add_f32 v5, v11, v15 // (op 135) 
v_add_f32 v6, v12, v16 // (op 135) 
v_add_f32 v7, v13, v17 // (op 135) 


v_cvt_f16_f32 v10, v4 // (op 136) 
v_cvt_f16_f32 v11, v5 // (op 136) 
v_cvt_f16_f32 v12, v6 // (op 136) 
v_cvt_f16_f32 v13, v7 // (op 136) 

v_lshrrev_b32 v0, 7, v1 // (op 884) 
v_and_b32 v3, 63, v1 // (op 884) 
v_lshrrev_b32 v4, 4, v3 // (op 884) 
v_lshlrev_b32 v5, 2, v4 // (op 884) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 884) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 884) 

v_lshrrev_b32 v4, 2, v3 // (op 884) 

v_and_b32 v3, 3, v4 // (op 884) 

v_and_b32 v4, 3, v1 // (op 884) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 884) 


v_lshrrev_b32 v3, 6, v1 // (op 884) 
v_and_b32 v4, 1, v3 // (op 884) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 884) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 884) 

v_mul_lo_u32 v3, v4, s30 // (op 884) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 884) 


s_mul_i32 s15, s30, s13 // (op 886) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 886) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 886) least significant
s_mul_hi_u32 s1, s30, s12 // (op 886) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 886) most significant: sum
s_add_u32 s15, s15, s1 // (op 886) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 886) 

s_mov_b32 s8, s32 // (op 886) reg expression
s_mov_b32 s9, s33 // (op 886) reg expression
s_mov_b32 s10, s0 // (op 886) reg expression
s_mov_b32 s11, 131072 // (op 886) call()


v_pack_b32_f16 v6, v10, v11 // (op 137) 
v_pack_b32_f16 v7, v12, v13 // (op 137) 


buffer_store_dwordx2 v[6:7], v4, s[8:11], 0 offen // (op 137) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_ConditionalBottom_0_beta_125:




s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end:
.size RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2, .LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2_end-RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 9472 // lds bytes
  .amdhsa_accum_offset 60
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F6
    DestDataType: h
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
    args: [ { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeFree1 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideC0 },
            { type: address, semantic: AddressC } ]
    macrotile: [16, 32, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2
    .symbol: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 9472
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 59
    .agpr_count: 4
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
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
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

