// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 708) 
v_and_b32 v3, 15, v1 // (op 708) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 708) 


v_and_b32 v0, 63, v1 // (op 708) 
v_lshrrev_b32 v3, 4, v0 // (op 708) 

v_lshlrev_b32 v0, 5, v3 // (op 708) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 708) 


v_mul_lo_u32 v0, v3, 6 // (op 708) 

v_lshrrev_b32 v3, 3, v0 // (op 708) 

v_lshrrev_b32 v0, 6, v1 // (op 761) 
v_and_b32 v4, 1, v0 // (op 761) 

v_and_b32 v0, 15, v1 // (op 761) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 761) 


v_and_b32 v0, 63, v1 // (op 761) 
v_lshrrev_b32 v4, 4, v0 // (op 761) 

v_lshlrev_b32 v0, 5, v4 // (op 761) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 761) 


v_mul_lo_u32 v0, v4, 6 // (op 761) 

v_lshrrev_b32 v4, 3, v0 // (op 761) 

v_lshrrev_b32 v0, 7, v1 // (op 819) 
v_and_b32 v5, 15, v1 // (op 819) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 819) 


v_and_b32 v0, 63, v1 // (op 819) 
v_lshrrev_b32 v5, 4, v0 // (op 819) 

v_lshlrev_b32 v0, 5, v5 // (op 819) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 819) 


v_mul_lo_u32 v0, v5, 6 // (op 819) 

v_lshrrev_b32 v5, 3, v0 // (op 819) 

v_lshrrev_b32 v0, 6, v1 // (op 843) 
v_and_b32 v6, 1, v0 // (op 843) 

v_and_b32 v0, 15, v1 // (op 843) 
v_lshl_add_u32 v7, v6, 4, v0 // (op 843) 


v_and_b32 v0, 63, v1 // (op 843) 
v_lshrrev_b32 v6, 4, v0 // (op 843) 

v_lshlrev_b32 v0, 5, v6 // (op 843) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 843) 


v_mul_lo_u32 v0, v6, 6 // (op 843) 

v_lshrrev_b32 v6, 3, v0 // (op 843) 

v_mov_b32 v0, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_lshrrev_b32 v8, 3, v1 // (op 418) 
v_lshl_add_u32 v9, s16, 5, v8 // (op 418) 

v_and_b32 v8, -8, v1 // (op 418) 
v_sub_u32 v10, v1, v8 // (op 418) 

v_lshlrev_b32 v8, 4, v10 // (op 418) 

v_mov_b32 v11, 0 // (op 418) convert
v_mov_b32 v10, v8 // (op 418) convert

v_mul_lo_u32 v13, v9, s3 // (op 418) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 418) most significant: high of low * low
v_mul_lo_u32 v12, v9, s2 // (op 418) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 418) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v10 // (op 418) least significant half
v_addc_co_u32 v13, s[0:1], v13, v11, s[0:1] // (op 418) most significant half



v_mul_lo_u32 v9, v13, 6 // (op 418) most significant: low of high * low
v_mul_hi_u32 v10, v12, 6 // (op 418) most significant: high of low * low
v_mul_lo_u32 v8, v12, 6 // (op 418) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 418) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[8:9] // (op 418) 

v_mov_b32 v8, v10 // (op 418) convert

s_mul_i32 s1, s7, 6 // (op 420) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 420) least significant
s_mul_hi_u32 s14, s6, 6 // (op 420) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 420) most significant: sum

s_mul_i32 s15, s2, s1 // (op 420) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 420) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 420) least significant
s_mul_hi_u32 s19, s2, s0 // (op 420) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 420) most significant: sum
s_add_u32 s15, s15, s19 // (op 420) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 420) 

s_mov_b32 s40, s4 // (op 420) reg expression
s_mov_b32 s41, s5 // (op 420) reg expression
s_mov_b32 s42, s0 // (op 420) reg expression
s_mov_b32 s43, 131072 // (op 420) call()

v_lshrrev_b32 v9, 3, v1 // (op 437) 
v_lshl_add_u32 v10, s16, 5, v9 // (op 437) 

v_and_b32 v9, -8, v1 // (op 437) 
v_sub_u32 v11, v1, v9 // (op 437) 

v_lshlrev_b32 v9, 4, v11 // (op 437) 

v_mov_b32 v13, 0 // (op 437) convert
v_mov_b32 v12, v9 // (op 437) convert

v_mul_lo_u32 v15, v10, s3 // (op 437) most significant: low of low * high
v_mul_hi_u32 v9, v10, s2 // (op 437) most significant: high of low * low
v_mul_lo_u32 v14, v10, s2 // (op 437) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 437) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 437) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 437) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 437) most significant: low of high * low
v_mul_hi_u32 v9, v14, 6 // (op 437) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 437) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 437) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 437) 

v_mov_b32 v9, v10 // (op 437) convert

s_mul_i32 s1, s7, 6 // (op 439) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 439) least significant
s_mul_hi_u32 s14, s6, 6 // (op 439) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 439) most significant: sum

s_mul_i32 s15, s2, s1 // (op 439) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 439) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 439) least significant
s_mul_hi_u32 s19, s2, s0 // (op 439) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 439) most significant: sum
s_add_u32 s15, s15, s19 // (op 439) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 439) 

s_mov_b32 s40, s4 // (op 439) reg expression
s_mov_b32 s41, s5 // (op 439) reg expression
s_mov_b32 s42, s0 // (op 439) reg expression
s_mov_b32 s43, 131072 // (op 439) call()

v_and_b32 v10, -8, v1 // (op 482) 
v_sub_u32 v11, v1, v10 // (op 482) 

v_lshlrev_b32 v10, 4, v11 // (op 482) 

v_mov_b32 v13, 0 // (op 482) convert
v_mov_b32 v12, v10 // (op 482) convert

v_lshrrev_b32 v10, 3, v1 // (op 482) 
v_lshl_add_u32 v11, s17, 5, v10 // (op 482) 

v_mul_lo_u32 v15, v11, s9 // (op 482) most significant: low of low * high
v_mul_hi_u32 v10, v11, s8 // (op 482) most significant: high of low * low
v_mul_lo_u32 v14, v11, s8 // (op 482) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 482) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 482) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 482) most significant half



v_mul_lo_u32 v13, v11, 6 // (op 482) most significant: low of high * low
v_mul_hi_u32 v14, v10, 6 // (op 482) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 482) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 482) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 482) 

v_mov_b32 v12, v10 // (op 482) convert

s_mul_i32 s1, s13, 6 // (op 484) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 484) least significant
s_mul_hi_u32 s14, s12, 6 // (op 484) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 484) most significant: sum

s_mul_i32 s15, s8, s1 // (op 484) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 484) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 484) least significant
s_mul_hi_u32 s19, s8, s0 // (op 484) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 484) most significant: sum
s_add_u32 s15, s15, s19 // (op 484) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 484) 

s_mov_b32 s44, s10 // (op 484) reg expression
s_mov_b32 s45, s11 // (op 484) reg expression
s_mov_b32 s46, s0 // (op 484) reg expression
s_mov_b32 s47, 131072 // (op 484) call()

v_and_b32 v10, -8, v1 // (op 501) 
v_sub_u32 v11, v1, v10 // (op 501) 

v_lshlrev_b32 v10, 4, v11 // (op 501) 

v_mov_b32 v15, 0 // (op 501) convert
v_mov_b32 v14, v10 // (op 501) convert

v_lshrrev_b32 v13, 3, v1 // (op 501) 
v_lshl_add_u32 v10, s17, 5, v13 // (op 501) 

v_mul_lo_u32 v17, v10, s9 // (op 501) most significant: low of low * high
v_mul_hi_u32 v11, v10, s8 // (op 501) most significant: high of low * low
v_mul_lo_u32 v16, v10, s8 // (op 501) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 501) most significant: sum


v_add_co_u32 v10, s[0:1], v14, v16 // (op 501) least significant half
v_addc_co_u32 v11, s[0:1], v15, v17, s[0:1] // (op 501) most significant half



v_mul_lo_u32 v15, v11, 6 // (op 501) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 501) most significant: high of low * low
v_mul_lo_u32 v14, v10, 6 // (op 501) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 501) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[14:15] // (op 501) 

v_mov_b32 v13, v10 // (op 501) convert

s_mul_i32 s1, s13, 6 // (op 503) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 503) least significant
s_mul_hi_u32 s14, s12, 6 // (op 503) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 503) most significant: sum

s_mul_i32 s15, s8, s1 // (op 503) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 503) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 503) least significant
s_mul_hi_u32 s19, s8, s0 // (op 503) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 503) most significant: sum
s_add_u32 s15, s15, s19 // (op 503) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 503) 

s_mov_b32 s44, s10 // (op 503) reg expression
s_mov_b32 s45, s11 // (op 503) reg expression
s_mov_b32 s46, s0 // (op 503) reg expression
s_mov_b32 s47, 131072 // (op 503) call()

v_lshrrev_b32 v10, 7, v1 // (op 546) 
v_and_b32 v11, 15, v1 // (op 546) 
v_lshl_add_u32 v14, v10, 4, v11 // (op 546) 


v_lshl_add_u32 v10, s16, 5, v14 // (op 546) 

v_and_b32 v11, 63, v1 // (op 546) 
v_lshrrev_b32 v14, 4, v11 // (op 546) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v11, v10, s20 // (op 546) 
v_add_u32 v11, v11, v14 // (op 546) 


s_mul_i32 s1, s20, s7 // (op 548) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 548) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 548) least significant
s_mul_hi_u32 s15, s20, s6 // (op 548) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 548) most significant: sum
s_add_u32 s1, s1, s15 // (op 548) most significant: sum


s_mov_b32 s48, s22 // (op 548) reg expression
s_mov_b32 s49, s23 // (op 548) reg expression
s_mov_b32 s50, s0 // (op 548) reg expression
s_mov_b32 s51, 131072 // (op 548) call()

v_lshrrev_b32 v10, 7, v1 // (op 565) 
v_and_b32 v14, 15, v1 // (op 565) 
v_lshl_add_u32 v15, v10, 4, v14 // (op 565) 


v_lshl_add_u32 v10, s16, 5, v15 // (op 565) 

v_and_b32 v14, 63, v1 // (op 565) 
v_lshrrev_b32 v15, 4, v14 // (op 565) 

v_mul_lo_u32 v14, v10, s20 // (op 565) 
v_add_u32 v14, v14, v15 // (op 565) 


s_mul_i32 s1, s20, s7 // (op 567) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 567) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 567) least significant
s_mul_hi_u32 s15, s20, s6 // (op 567) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 567) most significant: sum
s_add_u32 s1, s1, s15 // (op 567) most significant: sum


s_mov_b32 s48, s22 // (op 567) reg expression
s_mov_b32 s49, s23 // (op 567) reg expression
s_mov_b32 s50, s0 // (op 567) reg expression
s_mov_b32 s51, 131072 // (op 567) call()

v_lshrrev_b32 v10, 6, v1 // (op 610) 
v_and_b32 v15, 1, v10 // (op 610) 

v_and_b32 v10, 15, v1 // (op 610) 
v_lshl_add_u32 v16, v15, 4, v10 // (op 610) 


v_lshl_add_u32 v10, s17, 5, v16 // (op 610) 

v_mul_lo_u32 v15, v10, s24 // (op 610) 

v_and_b32 v10, 63, v1 // (op 610) 
v_lshrrev_b32 v16, 4, v10 // (op 610) 

v_add_u32 v10, v16, v15 // (op 610) 


s_mul_i32 s1, s24, s13 // (op 612) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 612) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 612) least significant
s_mul_hi_u32 s15, s24, s12 // (op 612) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 612) most significant: sum
s_add_u32 s1, s1, s15 // (op 612) most significant: sum


s_mov_b32 s52, s26 // (op 612) reg expression
s_mov_b32 s53, s27 // (op 612) reg expression
s_mov_b32 s54, s0 // (op 612) reg expression
s_mov_b32 s55, 131072 // (op 612) call()

v_lshrrev_b32 v15, 6, v1 // (op 629) 
v_and_b32 v16, 1, v15 // (op 629) 

v_and_b32 v15, 15, v1 // (op 629) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 629) 


v_lshl_add_u32 v15, s17, 5, v17 // (op 629) 

v_mul_lo_u32 v16, v15, s24 // (op 629) 

v_and_b32 v15, 63, v1 // (op 629) 
v_lshrrev_b32 v17, 4, v15 // (op 629) 

v_add_u32 v15, v17, v16 // (op 629) 


s_mul_i32 s1, s24, s13 // (op 631) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 631) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 631) least significant
s_mul_hi_u32 s15, s24, s12 // (op 631) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 631) most significant: sum
s_add_u32 s1, s1, s15 // (op 631) most significant: sum


s_mov_b32 s52, s26 // (op 631) reg expression
s_mov_b32 s53, s27 // (op 631) reg expression
s_mov_b32 s54, s0 // (op 631) reg expression
s_mov_b32 s55, 131072 // (op 631) call()




buffer_load_ubyte v16, v11, s[48:51], 0 offen offset: 0 // (op 308) Load value



buffer_load_ubyte v17, v10, s[52:55], 0 offen offset: 0 // (op 313) Load value



buffer_load_dwordx3 v[18:20], v12, s[44:47], 0 offen // (op 317) Load value



buffer_load_dwordx3 v[22:24], v8, s[40:43], 0 offen // (op 320) Load value



buffer_load_ubyte v21, v11, s[48:51], 0 offen offset:4 // (op 324) Load value




buffer_load_ubyte v11, v10, s[52:55], 0 offen offset:4 // (op 329) Load value




buffer_load_dwordx3 v[26:28], v12, s[44:47], 0 offen offset:96 // (op 333) Load value




buffer_load_dwordx3 v[30:32], v8, s[40:43], 0 offen offset:96 // (op 336) Load value

v_and_b32 v8, -8, v1 // (op 749) 
v_sub_u32 v10, v1, v8 // (op 749) 

v_lshlrev_b32 v8, 4, v10 // (op 749) 

v_lshrrev_b32 v10, 3, v1 // (op 749) 
v_lshl_add_u32 v12, v10, 7, v8 // (op 749) 


v_mul_lo_u32 v8, v12, 6 // (op 749) 

v_lshrrev_b32 v10, 3, v8 // (op 749) 




s_waitcnt vmcnt(5)
ds_write_b96 v10, v[18:20] // (op 338) Store local data 

v_and_b32 v8, -8, v1 // (op 696) 
v_sub_u32 v10, v1, v8 // (op 696) 

v_lshlrev_b32 v8, 4, v10 // (op 696) 

v_lshrrev_b32 v10, 3, v1 // (op 696) 
v_lshl_add_u32 v12, v10, 7, v8 // (op 696) 


v_mul_lo_u32 v8, v12, 6 // (op 696) 

v_lshrrev_b32 v10, 3, v8 // (op 696) 




s_waitcnt vmcnt(4)
ds_write_b96 v10, v[22:24] offset:3072 // (op 339) Store local data 

v_and_b32 v8, -8, v1 // (op 667) 
v_sub_u32 v10, v1, v8 // (op 667) 

v_lshlrev_b32 v8, 4, v10 // (op 667) 

v_lshrrev_b32 v10, 3, v1 // (op 667) 
v_lshl_add_u32 v12, v10, 7, v8 // (op 667) 


v_mul_lo_u32 v8, v12, 6 // (op 667) 

v_lshrrev_b32 v10, 3, v8 // (op 667) 

v_and_b32 v8, -8, v1 // (op 720) 
v_sub_u32 v12, v1, v8 // (op 720) 

v_lshlrev_b32 v8, 4, v12 // (op 720) 

v_lshrrev_b32 v12, 3, v1 // (op 720) 
v_lshl_add_u32 v25, v12, 7, v8 // (op 720) 


v_mul_lo_u32 v8, v25, 6 // (op 720) 

v_lshrrev_b32 v12, 3, v8 // (op 720) 

v_and_b32 v8, -8, v1 // (op 807) 
v_sub_u32 v25, v1, v8 // (op 807) 

v_lshlrev_b32 v8, 4, v25 // (op 807) 

v_lshrrev_b32 v25, 3, v1 // (op 807) 
v_lshl_add_u32 v29, v25, 7, v8 // (op 807) 


v_mul_lo_u32 v8, v29, 6 // (op 807) 

v_lshrrev_b32 v25, 3, v8 // (op 807) 

v_and_b32 v8, -8, v1 // (op 831) 
v_sub_u32 v29, v1, v8 // (op 831) 

v_lshlrev_b32 v8, 4, v29 // (op 831) 

v_lshrrev_b32 v29, 3, v1 // (op 831) 
v_lshl_add_u32 v33, v29, 7, v8 // (op 831) 


v_mul_lo_u32 v8, v33, 6 // (op 831) 

v_lshrrev_b32 v29, 3, v8 // (op 831) 

s_mov_b32 s0, 0 // (op 51) call()
s_ashr_i64 s[14:15], s[28:29], 7 // (op 50) 
s_mov_b32 s1, s14 // (op 50) convert

s_and_b32 s14, s1, 4294967294 // (op 50) 

s_cmp_lt_u32 s0, s14 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 // (op 50) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 if false)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50:
s_add_u32 s0, s0, 2 // (op 52) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 872) 



ds_read_b128 v[34:37], v3 offset:3072 // (op 60) Load local data 
ds_read_b64 v[38:39], v3 offset:3088 // (op 60) Load local data 
s_barrier  // (op 885) 



ds_read_b128 v[40:43], v4 // (op 65) Load local data 
ds_read_b64 v[44:45], v4 offset:16 // (op 65) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:39], v[40:45], a[0:3], v16, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 76) 




v_mov_b32 v8, v14 // (op 70) 

buffer_load_ubyte v16, v8, s[48:51], 0 offen offset:8 // (op 70) Load value


v_mov_b32 v33, v15 // (op 74) 

buffer_load_ubyte v17, v33, s[52:55], 0 offen offset:8 // (op 74) Load value


v_mov_b32 v34, v13 // (op 10) 

buffer_load_dwordx3 v[18:20], v34, s[44:47], 0 offen offset:192 // (op 10) Load value


v_mov_b32 v35, v9 // (op 4) 

buffer_load_dwordx3 v[22:24], v35, s[40:43], 0 offen offset:192 // (op 4) Load value
s_barrier  // (op 908) 



s_waitcnt vmcnt(5)
ds_write_b96 v29, v[26:28] offset:6144 // (op 218) Store local data 
s_barrier  // (op 902) 



s_waitcnt vmcnt(4)
ds_write_b96 v25, v[30:32] offset:9216 // (op 208) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 899) 



ds_read_b128 v[36:39], v5 offset:9216 // (op 210) Load local data 
ds_read_b64 v[40:41], v5 offset:9232 // (op 210) Load local data 
s_barrier  // (op 905) 
s_barrier  // (op 955) 



ds_read_b128 v[42:45], v6 offset:6144 // (op 212) Load local data 
ds_read_b64 v[46:47], v6 offset:6160 // (op 212) Load local data 
s_barrier  // (op 961) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:47], a[0:3], v21, v11 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 217) 





buffer_load_ubyte v21, v8, s[48:51], 0 offen offset:12 // (op 214) Load value




v_add_u32 v14, 8, v14 // (op 557) 
buffer_load_ubyte v11, v33, s[52:55], 0 offen offset:12 // (op 216) Load value




v_add_u32 v15, 8, v15 // (op 621) 
buffer_load_dwordx3 v[26:28], v34, s[44:47], 0 offen offset:288 // (op 219) Load value




v_add_u32 v13, 192, v13 // (op 493) 
buffer_load_dwordx3 v[30:32], v35, s[40:43], 0 offen offset:288 // (op 207) Load value

s_barrier  // (op 888) 
v_add_u32 v9, 192, v9 // (op 429) 



s_waitcnt vmcnt(5)
ds_write_b96 v12, v[18:20] // (op 13) Store local data 
s_barrier  // (op 875) 



s_waitcnt vmcnt(4)
ds_write_b96 v10, v[22:24] offset:3072 // (op 6) Store local data 
s_ashr_i64 s[14:15], s[28:29], 7 // (op 50) 
s_mov_b32 s1, s14 // (op 50) convert

s_and_b32 s14, s1, 4294967294 // (op 50) 

s_cmp_lt_u32 s0, s14 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 // (op 50) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50:













v_lshrrev_b32 v29, 3, v1 // (op 392) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 392) 
v_lshl_add_u32 v8, s16, 5, v29 // (op 392) 

s_lshr_b32 s14, s0, 1 // (op 392) 

v_and_b32 v29, -8, v1 // (op 392) 
v_sub_u32 v9, v1, v29 // (op 392) 

v_lshlrev_b32 v10, 4, v9 // (op 392) 

v_lshl_add_u32 v9, s14, 8, v10 // (op 392) 


v_mov_b32 v13, 0 // (op 392) convert
v_mov_b32 v12, v9 // (op 392) convert

v_mul_lo_u32 v15, v8, s3 // (op 392) most significant: low of low * high
v_mul_hi_u32 v29, v8, s2 // (op 392) most significant: high of low * low
v_mul_lo_u32 v14, v8, s2 // (op 392) least significant: low of low * low
v_add_u32 v15, v15, v29 // (op 392) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 392) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 392) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 392) most significant: low of high * low
v_mul_hi_u32 v29, v14, 6 // (op 392) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 392) least significant: low of low * low
v_add_u32 v13, v13, v29 // (op 392) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[12:13] // (op 392) 

v_mov_b32 v10, v8 // (op 392) convert

s_mul_i32 s1, s7, 6 // (op 394) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 394) least significant
s_mul_hi_u32 s14, s6, 6 // (op 394) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 394) most significant: sum

s_mul_i32 s15, s2, s1 // (op 394) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 394) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 394) least significant
s_mul_hi_u32 s19, s2, s0 // (op 394) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 394) most significant: sum
s_add_u32 s15, s15, s19 // (op 394) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 394) 

s_mov_b32 s40, s4 // (op 394) reg expression
s_mov_b32 s41, s5 // (op 394) reg expression
s_mov_b32 s42, s0 // (op 394) reg expression
s_mov_b32 s43, 131072 // (op 394) call()

v_and_b32 v29, -8, v1 // (op 459) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 459) 
v_sub_u32 v8, v1, v29 // (op 459) 

s_lshr_b32 s14, s0, 1 // (op 459) 

v_lshlrev_b32 v9, 4, v8 // (op 459) 

v_lshl_add_u32 v8, s14, 8, v9 // (op 459) 


v_mov_b32 v13, 0 // (op 459) convert
v_mov_b32 v12, v8 // (op 459) convert

v_lshrrev_b32 v29, 3, v1 // (op 459) 
v_lshl_add_u32 v8, s17, 5, v29 // (op 459) 

v_mul_lo_u32 v15, v8, s9 // (op 459) most significant: low of low * high
v_mul_hi_u32 v9, v8, s8 // (op 459) most significant: high of low * low
v_mul_lo_u32 v14, v8, s8 // (op 459) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 459) most significant: sum


v_add_co_u32 v8, s[0:1], v12, v14 // (op 459) least significant half
v_addc_co_u32 v9, s[0:1], v13, v15, s[0:1] // (op 459) most significant half



v_mul_lo_u32 v13, v9, 6 // (op 459) most significant: low of high * low
v_mul_hi_u32 v29, v8, 6 // (op 459) most significant: high of low * low
v_mul_lo_u32 v12, v8, 6 // (op 459) least significant: low of low * low
v_add_u32 v13, v13, v29 // (op 459) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[12:13] // (op 459) 

v_mov_b32 v29, v8 // (op 459) convert

s_mul_i32 s1, s13, 6 // (op 461) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 461) least significant
s_mul_hi_u32 s14, s12, 6 // (op 461) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 461) most significant: sum

s_mul_i32 s15, s8, s1 // (op 461) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 461) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 461) least significant
s_mul_hi_u32 s19, s8, s0 // (op 461) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 461) most significant: sum
s_add_u32 s15, s15, s19 // (op 461) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 461) 

s_mov_b32 s44, s10 // (op 461) reg expression
s_mov_b32 s45, s11 // (op 461) reg expression
s_mov_b32 s46, s0 // (op 461) reg expression
s_mov_b32 s47, 131072 // (op 461) call()

v_lshrrev_b32 v8, 7, v1 // (op 523) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 523) 
v_and_b32 v9, 15, v1 // (op 523) 
s_lshr_b32 s14, s0, 1 // (op 523) 

v_lshl_add_u32 v12, v8, 4, v9 // (op 523) 


v_lshl_add_u32 v8, s16, 5, v12 // (op 523) 

v_and_b32 v9, 63, v1 // (op 523) 
v_lshrrev_b32 v12, 4, v9 // (op 523) 

v_lshl_add_u32 v9, s14, 3, v12 // (op 523) 


v_mul_lo_u32 v12, v8, s20 // (op 523) 
v_add_u32 v12, v12, v9 // (op 523) 


s_mul_i32 s1, s20, s7 // (op 525) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 525) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 525) least significant
s_mul_hi_u32 s15, s20, s6 // (op 525) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 525) most significant: sum
s_add_u32 s1, s1, s15 // (op 525) most significant: sum


s_mov_b32 s48, s22 // (op 525) reg expression
s_mov_b32 s49, s23 // (op 525) reg expression
s_mov_b32 s50, s0 // (op 525) reg expression
s_mov_b32 s51, 131072 // (op 525) call()

v_and_b32 v8, 63, v1 // (op 587) 
s_ashr_i64 s[0:1], s[28:29], 7 // (op 587) 
v_lshrrev_b32 v9, 4, v8 // (op 587) 

s_lshr_b32 s14, s0, 1 // (op 587) 

v_lshrrev_b32 v8, 6, v1 // (op 587) 
v_lshl_add_u32 v13, s14, 3, v9 // (op 587) 


v_and_b32 v9, 1, v8 // (op 587) 

v_and_b32 v8, 15, v1 // (op 587) 
v_lshl_add_u32 v14, v9, 4, v8 // (op 587) 


v_lshl_add_u32 v8, s17, 5, v14 // (op 587) 

v_mul_lo_u32 v9, v8, s24 // (op 587) 

v_add_u32 v8, v13, v9 // (op 587) 


s_mul_i32 s1, s24, s13 // (op 589) most significant: low of low * high
s_mul_i32 s14, s25, s12 // (op 589) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 589) least significant
s_mul_hi_u32 s15, s24, s12 // (op 589) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 589) most significant: sum
s_add_u32 s1, s1, s15 // (op 589) most significant: sum


s_mov_b32 s52, s26 // (op 589) reg expression
s_mov_b32 s53, s27 // (op 589) reg expression
s_mov_b32 s54, s0 // (op 589) reg expression
s_mov_b32 s55, 131072 // (op 589) call()

v_and_b32 v9, -8, v1 // (op 680) 
v_sub_u32 v13, v1, v9 // (op 680) 

v_lshlrev_b32 v9, 4, v13 // (op 680) 

v_lshrrev_b32 v13, 3, v1 // (op 680) 
v_lshl_add_u32 v14, v13, 7, v9 // (op 680) 


v_mul_lo_u32 v9, v14, 6 // (op 680) 

v_lshrrev_b32 v13, 3, v9 // (op 680) 

v_and_b32 v9, -8, v1 // (op 733) 
v_sub_u32 v14, v1, v9 // (op 733) 

v_lshlrev_b32 v9, 4, v14 // (op 733) 

v_lshrrev_b32 v14, 3, v1 // (op 733) 
v_lshl_add_u32 v15, v14, 7, v9 // (op 733) 


v_mul_lo_u32 v9, v15, 6 // (op 733) 

v_lshrrev_b32 v14, 3, v9 // (op 733) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 153) 
s_and_b32 s14, s0, 4294967294 // (op 153) 

s_ashr_i64 s[0:1], s[28:29], 7 // (op 152) 
s_mov_b32 s15, s0 // (op 152) convert

s_cmp_lt_u32 s14, s15 // (op 152) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 // (op 152) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152:


s_add_u32 s14, s14, 1 // (op 154) 
v_mov_b32 v9, v10 // (op 157) 

buffer_load_dwordx3 v[16:18], v9, s[40:43], 0 offen // (op 157) Load value

v_add_u32 v10, 96, v10 // (op 382) 


s_barrier  // (op 878) 
v_mov_b32 v9, v29 // (op 169) 




buffer_load_dwordx3 v[20:22], v9, s[44:47], 0 offen // (op 169) Load value

v_add_u32 v29, 96, v29 // (op 451) 
s_waitcnt vmcnt(1)
ds_write_b96 v13, v[16:18] offset:3072 // (op 158) Store local data 

s_barrier  // (op 891) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 881) 






ds_read_b128 v[30:33], v3 offset:3072 // (op 160) Load local data 
ds_read_b64 v[34:35], v3 offset:3088 // (op 160) Load local data 
s_waitcnt vmcnt(0)
ds_write_b96 v14, v[20:22] // (op 168) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 895) 



ds_read_b128 v[16:19], v4 // (op 162) Load local data 
ds_read_b64 v[20:21], v4 offset:16 // (op 162) Load local data 


v_mov_b32 v9, v12 // (op 164) 

buffer_load_ubyte v11, v9, s[48:51], 0 offen offset: 0 // (op 164) Load value

v_add_u32 v12, 4, v12 // (op 515) 


v_mov_b32 v9, v8 // (op 166) 

buffer_load_ubyte v15, v9, s[52:55], 0 offen offset: 0 // (op 166) Load value

v_add_u32 v8, 4, v8 // (op 579) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:35], v[16:21], a[0:3], v11, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 167) 




s_ashr_i64 s[0:1], s[28:29], 7 // (op 152) 
s_mov_b32 s15, s0 // (op 152) convert

s_cmp_lt_u32 s14, s15 // (op 152) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 // (op 152) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152:





















s_barrier  // (op 949) 




v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v3, 7, v1 // (op 775) 
v_and_b32 v4, 63, v1 // (op 775) 
v_lshrrev_b32 v5, 4, v4 // (op 775) 
v_lshlrev_b32 v6, 2, v5 // (op 775) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 775) 


v_lshl_add_u32 v3, s16, 5, v5 // (op 775) 

v_lshrrev_b32 v5, 2, v4 // (op 775) 

v_and_b32 v4, 3, v5 // (op 775) 

v_and_b32 v5, 3, v1 // (op 775) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 775) 


v_lshrrev_b32 v4, 6, v1 // (op 775) 
v_and_b32 v5, 1, v4 // (op 775) 

v_lshl_add_u32 v4, v5, 4, v6 // (op 775) 


v_lshl_add_u32 v5, s17, 5, v4 // (op 775) 

v_mul_lo_u32 v4, v5, s30 // (op 775) 

v_add_lshl_u32 v5, v3, v4, 2 // (op 775) 


s_mul_i32 s15, s30, s13 // (op 777) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 777) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 777) least significant
s_mul_hi_u32 s1, s30, s12 // (op 777) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 777) most significant: sum
s_add_u32 s15, s15, s1 // (op 777) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 777) 

s_mov_b32 s8, s32 // (op 777) reg expression
s_mov_b32 s9, s33 // (op 777) reg expression
s_mov_b32 s10, s0 // (op 777) reg expression
s_mov_b32 s11, 131072 // (op 777) call()




buffer_store_dwordx4 v[12:15], v5, s[8:11], 0 offen // (op 48) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114:
v_lshrrev_b32 v3, 7, v1 // (op 650) 
v_and_b32 v4, 63, v1 // (op 650) 
v_lshrrev_b32 v5, 4, v4 // (op 650) 
v_lshlrev_b32 v6, 2, v5 // (op 650) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 650) 


v_lshl_add_u32 v3, s16, 5, v5 // (op 650) 

v_lshrrev_b32 v5, 2, v4 // (op 650) 

v_and_b32 v4, 3, v5 // (op 650) 

v_and_b32 v5, 3, v1 // (op 650) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 650) 


v_lshrrev_b32 v4, 6, v1 // (op 650) 
v_and_b32 v5, 1, v4 // (op 650) 

v_lshl_add_u32 v4, v5, 4, v6 // (op 650) 


v_lshl_add_u32 v5, s17, 5, v4 // (op 650) 

v_mul_lo_u32 v4, v5, s34 // (op 650) 

v_add_lshl_u32 v5, v3, v4, 2 // (op 650) 


s_mul_i32 s15, s34, s13 // (op 652) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 652) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 652) least significant
s_mul_hi_u32 s1, s34, s12 // (op 652) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 652) most significant: sum
s_add_u32 s15, s15, s1 // (op 652) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 652) 

s_mov_b32 s4, s36 // (op 652) reg expression
s_mov_b32 s5, s37 // (op 652) reg expression
s_mov_b32 s6, s0 // (op 652) reg expression
s_mov_b32 s7, 131072 // (op 652) call()





buffer_load_dwordx4 v[8:11], v5, s[4:7], 0 offen // (op 126) Load value



v_accvgpr_read v12, a0 // (op 123) 
v_accvgpr_read v13, a1 // (op 123) 
v_accvgpr_read v14, a2 // (op 123) 
v_accvgpr_read v15, a3 // (op 123) 
v_mul_f32 v16, v7, v12 // (op 123) 
v_mul_f32 v17, v7, v13 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v0, v8 // (op 127) 
v_mul_f32 v13, v0, v9 // (op 127) 
v_mul_f32 v14, v0, v10 // (op 127) 
v_mul_f32 v15, v0, v11 // (op 127) 

v_add_f32 v8, v12, v16 // (op 124) 
v_add_f32 v9, v13, v17 // (op 124) 
v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 


v_lshrrev_b32 v3, 7, v1 // (op 792) 
v_and_b32 v4, 63, v1 // (op 792) 
v_lshrrev_b32 v5, 4, v4 // (op 792) 
v_lshlrev_b32 v6, 2, v5 // (op 792) 

v_lshl_add_u32 v5, v3, 4, v6 // (op 792) 


v_lshl_add_u32 v3, s16, 5, v5 // (op 792) 

v_lshrrev_b32 v5, 2, v4 // (op 792) 

v_and_b32 v4, 3, v5 // (op 792) 

v_and_b32 v5, 3, v1 // (op 792) 
v_lshl_add_u32 v6, v4, 2, v5 // (op 792) 


v_lshrrev_b32 v4, 6, v1 // (op 792) 
v_and_b32 v5, 1, v4 // (op 792) 

v_lshl_add_u32 v4, v5, 4, v6 // (op 792) 


v_lshl_add_u32 v5, s17, 5, v4 // (op 792) 

v_mul_lo_u32 v4, v5, s30 // (op 792) 

v_add_lshl_u32 v5, v3, v4, 2 // (op 792) 


s_mul_i32 s15, s30, s13 // (op 794) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 794) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 794) least significant
s_mul_hi_u32 s1, s30, s12 // (op 794) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 794) most significant: sum
s_add_u32 s15, s15, s1 // (op 794) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 2 // (op 794) 

s_mov_b32 s8, s32 // (op 794) reg expression
s_mov_b32 s9, s33 // (op 794) reg expression
s_mov_b32 s10, s0 // (op 794) reg expression
s_mov_b32 s11, 131072 // (op 794) call()




buffer_store_dwordx4 v[8:11], v5, s[8:11], 0 offen // (op 125) Store value





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114:




s_endpgm  // End of RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12288 // lds bytes
  .amdhsa_accum_offset 48
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: B6
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
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12288
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 48
    .agpr_count: 4
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

