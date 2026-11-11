// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 708) 
v_and_b32 v3, 48, v1 // (op 708) 
v_sub_u32 v4, v0, v3 // (op 708) 

v_lshrrev_b32 v3, 4, v0 // (op 708) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 708) 


v_lshrrev_b32 v3, 7, v1 // (op 708) 
v_lshlrev_b32 v4, 4, v3 // (op 708) 

v_lshl_add_u32 v3, v0, 5, v4 // (op 708) 


v_mul_lo_u32 v0, v3, 6 // (op 708) 
v_lshrrev_b32 v4, 3, v0 // (op 708) 

v_lshrrev_b32 v0, 4, v3 // (op 708) 

v_lshlrev_b32 v3, 2, v0 // (op 708) 

v_add_u32 v0, v4, v3 // (op 708) 


v_and_b32 v3, 63, v1 // (op 761) 
v_and_b32 v4, 48, v1 // (op 761) 
v_sub_u32 v5, v3, v4 // (op 761) 

v_lshrrev_b32 v4, 4, v3 // (op 761) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 761) 


v_lshrrev_b32 v4, 6, v1 // (op 761) 
v_and_b32 v5, 1, v4 // (op 761) 

v_lshlrev_b32 v4, 4, v5 // (op 761) 

v_lshl_add_u32 v5, v3, 5, v4 // (op 761) 


v_mul_lo_u32 v3, v5, 6 // (op 761) 
v_lshrrev_b32 v4, 3, v3 // (op 761) 

v_lshrrev_b32 v3, 4, v5 // (op 761) 

v_lshlrev_b32 v5, 2, v3 // (op 761) 

v_add_u32 v3, v4, v5 // (op 761) 


v_and_b32 v4, 63, v1 // (op 819) 
v_and_b32 v5, 48, v1 // (op 819) 
v_sub_u32 v6, v4, v5 // (op 819) 

v_lshrrev_b32 v5, 4, v4 // (op 819) 

v_lshl_add_u32 v4, v5, 5, v6 // (op 819) 


v_lshrrev_b32 v5, 7, v1 // (op 819) 
v_lshlrev_b32 v6, 4, v5 // (op 819) 

v_lshl_add_u32 v5, v4, 5, v6 // (op 819) 


v_mul_lo_u32 v4, v5, 6 // (op 819) 
v_lshrrev_b32 v6, 3, v4 // (op 819) 

v_lshrrev_b32 v4, 4, v5 // (op 819) 

v_lshlrev_b32 v5, 2, v4 // (op 819) 

v_add_u32 v4, v6, v5 // (op 819) 


v_and_b32 v5, 63, v1 // (op 843) 
v_and_b32 v6, 48, v1 // (op 843) 
v_sub_u32 v7, v5, v6 // (op 843) 

v_lshrrev_b32 v6, 4, v5 // (op 843) 

v_lshl_add_u32 v5, v6, 5, v7 // (op 843) 


v_lshrrev_b32 v6, 6, v1 // (op 843) 
v_and_b32 v7, 1, v6 // (op 843) 

v_lshlrev_b32 v6, 4, v7 // (op 843) 

v_lshl_add_u32 v7, v5, 5, v6 // (op 843) 


v_mul_lo_u32 v5, v7, 6 // (op 843) 
v_lshrrev_b32 v6, 3, v5 // (op 843) 

v_lshrrev_b32 v5, 4, v7 // (op 843) 

v_lshlrev_b32 v7, 2, v5 // (op 843) 

v_add_u32 v5, v6, v7 // (op 843) 


v_mov_b32 v6, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_and_b32 v8, -2, v1 // (op 418) 
v_sub_u32 v9, v1, v8 // (op 418) 

v_lshlrev_b32 v8, 4, v9 // (op 418) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 418) 

v_mov_b32 v11, 0 // (op 418) convert
v_mov_b32 v10, v9 // (op 418) convert

v_lshrrev_b32 v8, 1, v1 // (op 418) 
v_mul_lo_u32 v13, v8, s3 // (op 418) most significant: low of low * high
v_mul_hi_u32 v9, v8, s2 // (op 418) most significant: high of low * low
v_mul_lo_u32 v12, v8, s2 // (op 418) least significant: low of low * low
v_add_u32 v13, v13, v9 // (op 418) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v12 // (op 418) least significant half
v_addc_co_u32 v9, s[0:1], v11, v13, s[0:1] // (op 418) most significant half



v_mul_lo_u32 v11, v9, 6 // (op 418) most significant: low of high * low
v_mul_hi_u32 v12, v8, 6 // (op 418) most significant: high of low * low
v_mul_lo_u32 v10, v8, 6 // (op 418) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 418) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[10:11] // (op 418) 

v_mov_b32 v10, v8 // (op 418) convert

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

v_and_b32 v8, -2, v1 // (op 437) 
v_sub_u32 v9, v1, v8 // (op 437) 

v_lshlrev_b32 v8, 4, v9 // (op 437) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 437) 

v_mov_b32 v13, 0 // (op 437) convert
v_mov_b32 v12, v9 // (op 437) convert

v_lshrrev_b32 v11, 1, v1 // (op 437) 
v_mul_lo_u32 v9, v11, s3 // (op 437) most significant: low of low * high
v_mul_hi_u32 v14, v11, s2 // (op 437) most significant: high of low * low
v_mul_lo_u32 v8, v11, s2 // (op 437) least significant: low of low * low
v_add_u32 v9, v9, v14 // (op 437) most significant: sum


v_add_co_u32 v14, s[0:1], v12, v8 // (op 437) least significant half
v_addc_co_u32 v15, s[0:1], v13, v9, s[0:1] // (op 437) most significant half



v_mul_lo_u32 v9, v15, 6 // (op 437) most significant: low of high * low
v_mul_hi_u32 v11, v14, 6 // (op 437) most significant: high of low * low
v_mul_lo_u32 v8, v14, 6 // (op 437) least significant: low of low * low
v_add_u32 v9, v9, v11 // (op 437) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[8:9] // (op 437) 

v_mov_b32 v11, v12 // (op 437) convert

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

v_and_b32 v8, -2, v1 // (op 482) 
v_sub_u32 v9, v1, v8 // (op 482) 

v_lshlrev_b32 v8, 4, v9 // (op 482) 

v_lshl_add_u32 v9, s17, 5, v8 // (op 482) 

v_mov_b32 v13, 0 // (op 482) convert
v_mov_b32 v12, v9 // (op 482) convert

v_lshrrev_b32 v8, 1, v1 // (op 482) 
v_mul_lo_u32 v15, v8, s9 // (op 482) most significant: low of low * high
v_mul_hi_u32 v9, v8, s8 // (op 482) most significant: high of low * low
v_mul_lo_u32 v14, v8, s8 // (op 482) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 482) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 482) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 482) most significant half



v_mul_lo_u32 v9, v15, 6 // (op 482) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 482) most significant: high of low * low
v_mul_lo_u32 v8, v14, 6 // (op 482) least significant: low of low * low
v_add_u32 v9, v9, v12 // (op 482) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[8:9] // (op 482) 

v_mov_b32 v8, v12 // (op 482) convert

s_mul_i32 s1, s7, 6 // (op 484) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 484) least significant
s_mul_hi_u32 s14, s6, 6 // (op 484) most significant: high of low * low
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

v_and_b32 v9, -2, v1 // (op 501) 
v_sub_u32 v12, v1, v9 // (op 501) 

v_lshlrev_b32 v9, 4, v12 // (op 501) 

v_lshl_add_u32 v12, s17, 5, v9 // (op 501) 

v_mov_b32 v15, 0 // (op 501) convert
v_mov_b32 v14, v12 // (op 501) convert

v_lshrrev_b32 v9, 1, v1 // (op 501) 
v_mul_lo_u32 v13, v9, s9 // (op 501) most significant: low of low * high
v_mul_hi_u32 v16, v9, s8 // (op 501) most significant: high of low * low
v_mul_lo_u32 v12, v9, s8 // (op 501) least significant: low of low * low
v_add_u32 v13, v13, v16 // (op 501) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v14 // (op 501) least significant half
v_addc_co_u32 v13, s[0:1], v13, v15, s[0:1] // (op 501) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 501) most significant: low of high * low
v_mul_hi_u32 v9, v12, 6 // (op 501) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 501) least significant: low of low * low
v_add_u32 v15, v15, v9 // (op 501) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 501) 

v_mov_b32 v9, v12 // (op 501) convert

s_mul_i32 s1, s7, 6 // (op 503) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 503) least significant
s_mul_hi_u32 s14, s6, 6 // (op 503) most significant: high of low * low
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

v_lshrrev_b32 v12, 7, v1 // (op 546) 
v_and_b32 v13, 15, v1 // (op 546) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 546) 


v_lshl_add_u32 v12, s16, 5, v14 // (op 546) 

v_and_b32 v13, 63, v1 // (op 546) 
v_lshrrev_b32 v14, 4, v13 // (op 546) 

v_mul_lo_u32 v13, v14, s12 // (op 546) 

v_add_u32 v14, v12, v13 // (op 546) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 548) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 548) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 548) least significant
s_mul_hi_u32 s15, s12, s22 // (op 548) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 548) most significant: sum
s_add_u32 s1, s1, s15 // (op 548) most significant: sum


s_mov_b32 s48, s20 // (op 548) reg expression
s_mov_b32 s49, s21 // (op 548) reg expression
s_mov_b32 s50, s0 // (op 548) reg expression
s_mov_b32 s51, 131072 // (op 548) call()

v_lshrrev_b32 v12, 7, v1 // (op 565) 
v_and_b32 v13, 15, v1 // (op 565) 
v_lshl_add_u32 v15, v12, 4, v13 // (op 565) 


v_lshl_add_u32 v12, s16, 5, v15 // (op 565) 

v_and_b32 v13, 63, v1 // (op 565) 
v_lshrrev_b32 v15, 4, v13 // (op 565) 

v_mul_lo_u32 v13, v15, s12 // (op 565) 

v_add_u32 v15, v12, v13 // (op 565) 


s_mul_i32 s1, s12, s23 // (op 567) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 567) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 567) least significant
s_mul_hi_u32 s15, s12, s22 // (op 567) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 567) most significant: sum
s_add_u32 s1, s1, s15 // (op 567) most significant: sum


s_mov_b32 s48, s20 // (op 567) reg expression
s_mov_b32 s49, s21 // (op 567) reg expression
s_mov_b32 s50, s0 // (op 567) reg expression
s_mov_b32 s51, 131072 // (op 567) call()

v_lshrrev_b32 v12, 6, v1 // (op 610) 
v_and_b32 v13, 1, v12 // (op 610) 

v_and_b32 v12, 15, v1 // (op 610) 
v_lshl_add_u32 v16, v13, 4, v12 // (op 610) 


v_lshl_add_u32 v12, s17, 5, v16 // (op 610) 

v_and_b32 v13, 63, v1 // (op 610) 
v_lshrrev_b32 v16, 4, v13 // (op 610) 

v_mul_lo_u32 v13, v16, s24 // (op 610) 
v_add_u32 v13, v13, v12 // (op 610) 


s_mul_i32 s1, s24, s23 // (op 612) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 612) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 612) least significant
s_mul_hi_u32 s15, s24, s22 // (op 612) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 612) most significant: sum
s_add_u32 s1, s1, s15 // (op 612) most significant: sum


s_mov_b32 s52, s26 // (op 612) reg expression
s_mov_b32 s53, s27 // (op 612) reg expression
s_mov_b32 s54, s0 // (op 612) reg expression
s_mov_b32 s55, 131072 // (op 612) call()

v_lshrrev_b32 v12, 6, v1 // (op 629) 
v_and_b32 v16, 1, v12 // (op 629) 

v_and_b32 v12, 15, v1 // (op 629) 
v_lshl_add_u32 v17, v16, 4, v12 // (op 629) 


v_lshl_add_u32 v12, s17, 5, v17 // (op 629) 

v_and_b32 v16, 63, v1 // (op 629) 
v_lshrrev_b32 v17, 4, v16 // (op 629) 

v_mul_lo_u32 v16, v17, s24 // (op 629) 
v_add_u32 v16, v16, v12 // (op 629) 


s_mul_i32 s1, s24, s23 // (op 631) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 631) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 631) least significant
s_mul_hi_u32 s15, s24, s22 // (op 631) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 631) most significant: sum
s_add_u32 s1, s1, s15 // (op 631) most significant: sum


s_mov_b32 s52, s26 // (op 631) reg expression
s_mov_b32 s53, s27 // (op 631) reg expression
s_mov_b32 s54, s0 // (op 631) reg expression
s_mov_b32 s55, 131072 // (op 631) call()


v_mov_b64 v[18:19], s[12:13] // (op 308) convert

v_mov_b32 v12, v14 // (op 308) reg expression
v_mov_b32 v17, v12 // (op 308) 
buffer_load_ubyte v20, v17, s[48:51], 0 offen offset: 0 // (op 308) Load value




v_mov_b64 v[18:19], s[24:25] // (op 313) convert

v_mov_b32 v12, v13 // (op 313) reg expression
v_mov_b32 v17, v12 // (op 313) 
buffer_load_ubyte v21, v17, s[52:55], 0 offen offset: 0 // (op 313) Load value





v_mov_b32 v12, v8 // (op 317) reg expression

buffer_load_dwordx3 v[22:24], v12, s[44:47], 0 offen // (op 317) Load value



v_mov_b32 v12, v10 // (op 320) reg expression

buffer_load_dwordx3 v[26:28], v12, s[40:43], 0 offen // (op 320) Load value


v_mov_b64 v[18:19], s[12:13] // (op 324) convert

s_lshl_b32 s0, s12, 2 // (op 324) 
v_add_u32 v12, v14, s0 // (op 324) 

v_mov_b32 v17, v12 // (op 324) 
buffer_load_ubyte v25, v17, s[48:51], 0 offen offset: 0 // (op 324) Load value





v_mov_b64 v[18:19], s[24:25] // (op 329) convert

s_lshl_b32 s0, s24, 2 // (op 329) 
v_add_u32 v12, v13, s0 // (op 329) 

v_mov_b32 v14, v12 // (op 329) 
buffer_load_ubyte v17, v14, s[52:55], 0 offen offset: 0 // (op 329) Load value






s_lshl_b64 s[0:1], s[8:9], 7 // (op 333) 
s_mul_i32 s15, s1, 6 // (op 333) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 333) least significant
s_mul_hi_u32 s18, s0, 6 // (op 333) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 333) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 333) 

v_add_u32 v12, v8, s0 // (op 333) 


buffer_load_dwordx3 v[30:32], v12, s[44:47], 0 offen // (op 333) Load value




s_lshl_b64 s[0:1], s[2:3], 7 // (op 336) 
s_mul_i32 s15, s1, 6 // (op 336) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 336) least significant
s_mul_hi_u32 s18, s0, 6 // (op 336) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 336) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 336) 

v_add_u32 v8, v10, s0 // (op 336) 


buffer_load_dwordx3 v[12:14], v8, s[40:43], 0 offen // (op 336) Load value


v_and_b32 v8, -2, v1 // (op 749) 
v_sub_u32 v10, v1, v8 // (op 749) 

v_lshlrev_b32 v8, 4, v10 // (op 749) 

v_lshrrev_b32 v10, 1, v1 // (op 749) 
v_lshl_add_u32 v29, v10, 5, v8 // (op 749) 


v_mul_lo_u32 v8, v29, 6 // (op 749) 
v_lshrrev_b32 v10, 3, v8 // (op 749) 

v_lshrrev_b32 v8, 4, v29 // (op 749) 

v_lshlrev_b32 v29, 2, v8 // (op 749) 

v_add_u32 v8, v10, v29 // (op 749) 





s_waitcnt vmcnt(5)
ds_write_b96 v8, v[22:24] // (op 338) Store local data 

v_and_b32 v8, -2, v1 // (op 696) 
v_sub_u32 v10, v1, v8 // (op 696) 

v_lshlrev_b32 v8, 4, v10 // (op 696) 

v_lshrrev_b32 v10, 1, v1 // (op 696) 
v_lshl_add_u32 v29, v10, 5, v8 // (op 696) 


v_mul_lo_u32 v8, v29, 6 // (op 696) 
v_lshrrev_b32 v10, 3, v8 // (op 696) 

v_lshrrev_b32 v8, 4, v29 // (op 696) 

v_lshlrev_b32 v29, 2, v8 // (op 696) 

v_add_u32 v8, v10, v29 // (op 696) 





s_waitcnt vmcnt(4)
ds_write_b96 v8, v[26:28] offset:4096 // (op 339) Store local data 

v_and_b32 v8, -2, v1 // (op 667) 
v_sub_u32 v10, v1, v8 // (op 667) 

v_lshlrev_b32 v8, 4, v10 // (op 667) 

v_lshrrev_b32 v10, 1, v1 // (op 667) 
v_lshl_add_u32 v29, v10, 5, v8 // (op 667) 


v_mul_lo_u32 v8, v29, 6 // (op 667) 
v_lshrrev_b32 v10, 3, v8 // (op 667) 

v_lshrrev_b32 v8, 4, v29 // (op 667) 

v_lshlrev_b32 v29, 2, v8 // (op 667) 

v_add_u32 v8, v10, v29 // (op 667) 


v_and_b32 v10, -2, v1 // (op 720) 
v_sub_u32 v29, v1, v10 // (op 720) 

v_lshlrev_b32 v10, 4, v29 // (op 720) 

v_lshrrev_b32 v29, 1, v1 // (op 720) 
v_lshl_add_u32 v18, v29, 5, v10 // (op 720) 


v_mul_lo_u32 v10, v18, 6 // (op 720) 
v_lshrrev_b32 v19, 3, v10 // (op 720) 

v_lshrrev_b32 v10, 4, v18 // (op 720) 

v_lshlrev_b32 v18, 2, v10 // (op 720) 

v_add_u32 v10, v19, v18 // (op 720) 


v_and_b32 v29, -2, v1 // (op 807) 
v_sub_u32 v18, v1, v29 // (op 807) 

v_lshlrev_b32 v19, 4, v18 // (op 807) 

v_lshrrev_b32 v18, 1, v1 // (op 807) 
v_lshl_add_u32 v29, v18, 5, v19 // (op 807) 


v_mul_lo_u32 v18, v29, 6 // (op 807) 
v_lshrrev_b32 v19, 3, v18 // (op 807) 

v_lshrrev_b32 v18, 4, v29 // (op 807) 

v_lshlrev_b32 v29, 2, v18 // (op 807) 

v_add_u32 v18, v19, v29 // (op 807) 


v_and_b32 v19, -2, v1 // (op 831) 
v_sub_u32 v29, v1, v19 // (op 831) 

v_lshlrev_b32 v19, 4, v29 // (op 831) 

v_lshrrev_b32 v29, 1, v1 // (op 831) 
v_lshl_add_u32 v33, v29, 5, v19 // (op 831) 


v_mul_lo_u32 v19, v33, 6 // (op 831) 
v_lshrrev_b32 v29, 3, v19 // (op 831) 

v_lshrrev_b32 v19, 4, v33 // (op 831) 

v_lshlrev_b32 v33, 2, v19 // (op 831) 

v_add_u32 v19, v29, v33 // (op 831) 


s_mov_b32 s0, 0 // (op 51) call()
s_ashr_i64 s[14:15], s[6:7], 7 // (op 50) 
s_mov_b32 s1, s14 // (op 50) convert

s_and_b32 s14, s1, 4294967294 // (op 50) 

s_cmp_lt_u32 s0, s14 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 // (op 50) Condition: Top (jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50 if false)
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50:
s_add_u32 s0, s0, 2 // (op 52) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 872) 



ds_read_b96_tr_b6 v[34:36], v0 offset:4096 // (op 60) Transpose load local data 
ds_read_b96_tr_b6 v[37:39], v0 offset:4608 // (op 60) Transpose load local data 
s_barrier  // (op 885) 



ds_read_b96_tr_b6 v[40:42], v3 // (op 65) Transpose load local data 
ds_read_b96_tr_b6 v[43:45], v3 offset:512 // (op 65) Transpose load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:39], v[40:45], a[0:3], v20, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 76) 



v_mov_b64 v[34:35], s[12:13] // (op 70) convert

v_mov_b32 v29, v15 // (op 70) 
s_lshl_b32 s1, s12, 3 // (op 70) 
v_add_u32 v33, v29, s1 // (op 70) 

v_mov_b32 v36, v33 // (op 70) 
buffer_load_ubyte v20, v36, s[48:51], 0 offen offset: 0 // (op 70) Load value




v_mov_b64 v[34:35], s[24:25] // (op 74) convert

v_mov_b32 v33, v16 // (op 74) 
s_lshl_b32 s1, s24, 3 // (op 74) 
v_add_u32 v36, v33, s1 // (op 74) 

v_mov_b32 v37, v36 // (op 74) 
buffer_load_ubyte v21, v37, s[52:55], 0 offen offset: 0 // (op 74) Load value





v_mov_b32 v34, v9 // (op 10) 
s_lshl_b64 s[14:15], s[8:9], 7 // (op 10) 
s_mul_i32 s19, s15, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 10) least significant
s_mul_hi_u32 s1, s14, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 10) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 10) 

s_lshl_b32 s1, s14, 1 // (op 10) 

v_add_u32 v35, v34, s1 // (op 10) 


buffer_load_dwordx3 v[22:24], v35, s[44:47], 0 offen // (op 10) Load value



v_mov_b32 v35, v11 // (op 4) 
s_lshl_b64 s[14:15], s[2:3], 7 // (op 4) 
s_mul_i32 s19, s15, 6 // (op 4) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 4) least significant
s_mul_hi_u32 s1, s14, 6 // (op 4) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 4) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 4) 

s_lshl_b32 s1, s14, 1 // (op 4) 

v_add_u32 v36, v35, s1 // (op 4) 


buffer_load_dwordx3 v[26:28], v36, s[40:43], 0 offen // (op 4) Load value

s_barrier  // (op 908) 



s_waitcnt vmcnt(5)
ds_write_b96 v19, v[30:32] offset:8192 // (op 218) Store local data 
s_barrier  // (op 902) 



s_waitcnt vmcnt(4)
ds_write_b96 v18, v[12:14] offset:12288 // (op 208) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 899) 



ds_read_b96_tr_b6 v[36:38], v4 offset:12288 // (op 210) Transpose load local data 
ds_read_b96_tr_b6 v[39:41], v4 offset:12800 // (op 210) Transpose load local data 
s_barrier  // (op 905) 
s_barrier  // (op 955) 



ds_read_b96_tr_b6 v[42:44], v5 offset:8192 // (op 212) Transpose load local data 
ds_read_b96_tr_b6 v[45:47], v5 offset:8704 // (op 212) Transpose load local data 
s_barrier  // (op 961) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:41], v[42:47], a[0:3], v25, v17 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 217) 



v_mov_b64 v[36:37], s[12:13] // (op 214) convert

s_lshl_b32 s1, s12, 2 // (op 214) 
s_mul_i32 s14, 3, s1 // (op 214) 

v_add_u32 v38, v29, s14 // (op 214) 

v_mov_b32 v39, v38 // (op 214) 
buffer_load_ubyte v25, v39, s[48:51], 0 offen offset: 0 // (op 214) Load value





s_lshl_b32 s1, s12, 3 // (op 557) 
v_add_u32 v15, v15, s1 // (op 557) 

v_mov_b64 v[36:37], s[24:25] // (op 216) convert

s_lshl_b32 s1, s24, 2 // (op 216) 
s_mul_i32 s14, 3, s1 // (op 216) 

v_add_u32 v29, v33, s14 // (op 216) 

v_mov_b32 v38, v29 // (op 216) 
buffer_load_ubyte v17, v38, s[52:55], 0 offen offset: 0 // (op 216) Load value






s_lshl_b32 s1, s24, 3 // (op 621) 
v_add_u32 v16, v16, s1 // (op 621) 

s_lshl_b64 s[14:15], s[8:9], 7 // (op 219) 
s_mul_i32 s19, s15, 6 // (op 219) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 219) least significant
s_mul_hi_u32 s1, s14, 6 // (op 219) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 219) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 219) 

s_mul_i32 s1, 3, s14 // (op 219) 

v_add_u32 v29, v34, s1 // (op 219) 


buffer_load_dwordx3 v[30:32], v29, s[44:47], 0 offen // (op 219) Load value




s_lshl_b64 s[14:15], s[2:3], 7 // (op 207) 
s_mul_i32 s19, s15, 6 // (op 207) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 207) least significant
s_mul_hi_u32 s1, s14, 6 // (op 207) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 207) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 207) 

s_mul_i32 s1, 3, s14 // (op 207) 

v_add_u32 v29, v35, s1 // (op 207) 


s_lshl_b64 s[14:15], s[8:9], 7 // (op 493) 
s_mul_i32 s19, s15, 6 // (op 493) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 493) least significant
s_mul_hi_u32 s1, s14, 6 // (op 493) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 493) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 493) 

s_lshl_b32 s1, s14, 1 // (op 493) 

v_add_u32 v9, v9, s1 // (op 493) 

buffer_load_dwordx3 v[12:14], v29, s[40:43], 0 offen // (op 207) Load value


s_barrier  // (op 888) 
s_lshl_b64 s[14:15], s[2:3], 7 // (op 429) 



s_mul_i32 s19, s15, 6 // (op 429) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 429) least significant
s_mul_hi_u32 s1, s14, 6 // (op 429) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 429) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 429) 

s_lshl_b32 s1, s14, 1 // (op 429) 

v_add_u32 v11, v11, s1 // (op 429) 

s_waitcnt vmcnt(5)
ds_write_b96 v10, v[22:24] // (op 13) Store local data 
s_barrier  // (op 875) 



s_waitcnt vmcnt(4)
ds_write_b96 v8, v[26:28] offset:4096 // (op 6) Store local data 
s_ashr_i64 s[14:15], s[6:7], 7 // (op 50) 
s_mov_b32 s1, s14 // (op 50) convert

s_and_b32 s14, s1, 4294967294 // (op 50) 

s_cmp_lt_u32 s0, s14 // (op 50) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 // (op 50) Condition: Bottom (jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_50 if true)
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_50:













v_and_b32 v8, -2, v1 // (op 392) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 392) 
v_sub_u32 v9, v1, v8 // (op 392) 

s_lshr_b32 s14, s0, 1 // (op 392) 

v_lshlrev_b32 v8, 4, v9 // (op 392) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 392) 

v_mov_b32 v11, 0 // (op 392) convert
v_mov_b32 v10, v9 // (op 392) convert

v_lshrrev_b32 v8, 1, v1 // (op 392) 
v_lshl_add_u32 v9, s14, 8, v8 // (op 392) 


v_mul_lo_u32 v19, v9, s3 // (op 392) most significant: low of low * high
v_mul_hi_u32 v8, v9, s2 // (op 392) most significant: high of low * low
v_mul_lo_u32 v18, v9, s2 // (op 392) least significant: low of low * low
v_add_u32 v19, v19, v8 // (op 392) most significant: sum


v_add_co_u32 v8, s[0:1], v10, v18 // (op 392) least significant half
v_addc_co_u32 v9, s[0:1], v11, v19, s[0:1] // (op 392) most significant half



v_mul_lo_u32 v11, v9, 6 // (op 392) most significant: low of high * low
v_mul_hi_u32 v15, v8, 6 // (op 392) most significant: high of low * low
v_mul_lo_u32 v10, v8, 6 // (op 392) least significant: low of low * low
v_add_u32 v11, v11, v15 // (op 392) most significant: sum


v_lshrrev_b64 v[8:9], 3, v[10:11] // (op 392) 

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

v_lshrrev_b32 v11, 1, v1 // (op 459) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 459) 
v_and_b32 v8, -2, v1 // (op 459) 
s_lshr_b32 s14, s0, 1 // (op 459) 

v_sub_u32 v9, v1, v8 // (op 459) 

v_lshl_add_u32 v8, s14, 8, v11 // (op 459) 


v_lshlrev_b32 v11, 4, v9 // (op 459) 

v_lshl_add_u32 v9, s17, 5, v11 // (op 459) 

v_mov_b32 v19, 0 // (op 459) convert
v_mov_b32 v18, v9 // (op 459) convert

v_mul_lo_u32 v21, v8, s9 // (op 459) most significant: low of low * high
v_mul_hi_u32 v9, v8, s8 // (op 459) most significant: high of low * low
v_mul_lo_u32 v20, v8, s8 // (op 459) least significant: low of low * low
v_add_u32 v21, v21, v9 // (op 459) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 459) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 459) most significant half



v_mul_lo_u32 v9, v21, 6 // (op 459) most significant: low of high * low
v_mul_hi_u32 v11, v20, 6 // (op 459) most significant: high of low * low
v_mul_lo_u32 v8, v20, 6 // (op 459) least significant: low of low * low
v_add_u32 v9, v9, v11 // (op 459) most significant: sum


v_lshrrev_b64 v[18:19], 3, v[8:9] // (op 459) 

v_mov_b32 v11, v18 // (op 459) convert

s_mul_i32 s1, s7, 6 // (op 461) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 461) least significant
s_mul_hi_u32 s14, s6, 6 // (op 461) most significant: high of low * low
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
s_ashr_i64 s[0:1], s[6:7], 7 // (op 523) 
v_and_b32 v9, 15, v1 // (op 523) 
s_lshr_b32 s14, s0, 1 // (op 523) 

v_lshl_add_u32 v15, v8, 4, v9 // (op 523) 


v_lshl_add_u32 v16, s16, 5, v15 // (op 523) 

v_and_b32 v15, 63, v1 // (op 523) 
v_lshrrev_b32 v8, 4, v15 // (op 523) 

v_lshl_add_u32 v9, s14, 3, v8 // (op 523) 


v_mul_lo_u32 v8, v9, s12 // (op 523) 

v_add_u32 v9, v16, v8 // (op 523) 


s_mul_i32 s1, s12, s23 // (op 525) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 525) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 525) least significant
s_mul_hi_u32 s15, s12, s22 // (op 525) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 525) most significant: sum
s_add_u32 s1, s1, s15 // (op 525) most significant: sum


s_mov_b32 s48, s20 // (op 525) reg expression
s_mov_b32 s49, s21 // (op 525) reg expression
s_mov_b32 s50, s0 // (op 525) reg expression
s_mov_b32 s51, 131072 // (op 525) call()

v_and_b32 v8, 63, v1 // (op 587) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 587) 
v_lshrrev_b32 v15, 4, v8 // (op 587) 

s_lshr_b32 s14, s0, 1 // (op 587) 

v_lshrrev_b32 v8, 6, v1 // (op 587) 
v_lshl_add_u32 v16, s14, 3, v15 // (op 587) 


v_and_b32 v15, 1, v8 // (op 587) 

v_and_b32 v8, 15, v1 // (op 587) 
v_lshl_add_u32 v18, v15, 4, v8 // (op 587) 


v_lshl_add_u32 v8, s17, 5, v18 // (op 587) 

v_mul_lo_u32 v15, v16, s24 // (op 587) 
v_add_u32 v15, v15, v8 // (op 587) 


s_mul_i32 s1, s24, s23 // (op 589) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 589) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 589) least significant
s_mul_hi_u32 s15, s24, s22 // (op 589) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 589) most significant: sum
s_add_u32 s1, s1, s15 // (op 589) most significant: sum


s_mov_b32 s52, s26 // (op 589) reg expression
s_mov_b32 s53, s27 // (op 589) reg expression
s_mov_b32 s54, s0 // (op 589) reg expression
s_mov_b32 s55, 131072 // (op 589) call()

v_and_b32 v8, -2, v1 // (op 680) 
v_sub_u32 v16, v1, v8 // (op 680) 

v_lshlrev_b32 v8, 4, v16 // (op 680) 

v_lshrrev_b32 v16, 1, v1 // (op 680) 
v_lshl_add_u32 v18, v16, 5, v8 // (op 680) 


v_mul_lo_u32 v8, v18, 6 // (op 680) 
v_lshrrev_b32 v16, 3, v8 // (op 680) 

v_lshrrev_b32 v8, 4, v18 // (op 680) 

v_lshlrev_b32 v18, 2, v8 // (op 680) 

v_add_u32 v8, v16, v18 // (op 680) 


v_and_b32 v16, -2, v1 // (op 733) 
v_sub_u32 v18, v1, v16 // (op 733) 

v_lshlrev_b32 v16, 4, v18 // (op 733) 

v_lshrrev_b32 v18, 1, v1 // (op 733) 
v_lshl_add_u32 v19, v18, 5, v16 // (op 733) 


v_mul_lo_u32 v16, v19, 6 // (op 733) 
v_lshrrev_b32 v18, 3, v16 // (op 733) 

v_lshrrev_b32 v16, 4, v19 // (op 733) 

v_lshlrev_b32 v19, 2, v16 // (op 733) 

v_add_u32 v16, v18, v19 // (op 733) 


s_ashr_i64 s[0:1], s[6:7], 7 // (op 153) 
s_and_b32 s14, s0, 4294967294 // (op 153) 

s_ashr_i64 s[0:1], s[6:7], 7 // (op 152) 
s_mov_b32 s15, s0 // (op 152) convert

s_cmp_lt_u32 s14, s15 // (op 152) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 // (op 152) Condition: Top (jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152:


s_add_u32 s14, s14, 1 // (op 154) 
v_mov_b32 v12, v10 // (op 157) 
v_mov_b32 v13, v12 // (op 157) reg expression

buffer_load_dwordx3 v[18:20], v13, s[40:43], 0 offen // (op 157) Load value


s_lshl_b64 s[0:1], s[2:3], 7 // (op 382) 
s_mul_i32 s19, s1, 6 // (op 382) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 382) least significant
s_mul_hi_u32 s15, s0, 6 // (op 382) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 382) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 382) 

v_add_u32 v10, v10, s0 // (op 382) 



s_barrier  // (op 878) 
v_mov_b32 v17, v11 // (op 169) 



v_mov_b32 v12, v17 // (op 169) reg expression

buffer_load_dwordx3 v[22:24], v12, s[44:47], 0 offen // (op 169) Load value


s_lshl_b64 s[0:1], s[8:9], 7 // (op 451) 
s_mul_i32 s19, s1, 6 // (op 451) most significant: low of high * low
s_mul_i32 s18, s0, 6 // (op 451) least significant
s_mul_hi_u32 s15, s0, 6 // (op 451) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 451) most significant: sum


s_lshr_b64 s[0:1], s[18:19], 3 // (op 451) 

v_add_u32 v11, v11, s0 // (op 451) 

s_waitcnt vmcnt(1)
ds_write_b96 v8, v[18:20] offset:4096 // (op 158) Store local data 

s_barrier  // (op 891) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 881) 






ds_read_b96_tr_b6 v[26:28], v0 offset:4096 // (op 160) Transpose load local data 
ds_read_b96_tr_b6 v[29:31], v0 offset:4608 // (op 160) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b96 v16, v[22:24] // (op 168) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 895) 



ds_read_b96_tr_b6 v[20:22], v3 // (op 162) Transpose load local data 
ds_read_b96_tr_b6 v[23:25], v3 offset:512 // (op 162) Transpose load local data 

v_mov_b64 v[18:19], s[12:13] // (op 164) convert

v_mov_b32 v17, v9 // (op 164) 
v_mov_b32 v12, v17 // (op 164) reg expression
v_mov_b32 v13, v12 // (op 164) 
buffer_load_ubyte v14, v13, s[48:51], 0 offen offset: 0 // (op 164) Load value




s_lshl_b32 s15, s12, 2 // (op 515) 
v_add_u32 v9, v9, s15 // (op 515) 


v_mov_b64 v[12:13], s[24:25] // (op 166) convert

v_mov_b32 v17, v15 // (op 166) 
v_mov_b32 v18, v17 // (op 166) reg expression
v_mov_b32 v19, v18 // (op 166) 
buffer_load_ubyte v32, v19, s[52:55], 0 offen offset: 0 // (op 166) Load value




s_lshl_b32 s15, s24, 2 // (op 579) 
v_add_u32 v15, v15, s15 // (op 579) 

s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:31], v[20:25], a[0:3], v14, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 167) 




s_ashr_i64 s[0:1], s[6:7], 7 // (op 152) 
s_mov_b32 s15, s0 // (op 152) convert

s_cmp_lt_u32 s14, s15 // (op 152) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 // (op 152) Condition: Bottom (jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_152 if true)
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_152:





















s_barrier  // (op 949) 




v_cmp_eq_f32 vcc, 0.00000, v6 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v0, 7, v1 // (op 775) 
v_and_b32 v3, 63, v1 // (op 775) 
v_lshrrev_b32 v4, 4, v3 // (op 775) 
v_lshlrev_b32 v5, 2, v4 // (op 775) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 775) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 775) 

v_lshrrev_b32 v4, 2, v3 // (op 775) 

v_and_b32 v3, 3, v4 // (op 775) 

v_and_b32 v4, 3, v1 // (op 775) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 775) 


v_lshrrev_b32 v3, 6, v1 // (op 775) 
v_and_b32 v4, 1, v3 // (op 775) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 775) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 775) 

v_mul_lo_u32 v3, v4, s28 // (op 775) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 775) 


s_mul_i32 s1, s28, s33 // (op 777) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 777) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 777) least significant
s_mul_hi_u32 s3, s28, s32 // (op 777) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 777) most significant: sum
s_add_u32 s1, s1, s3 // (op 777) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 777) 

s_mov_b32 s4, s30 // (op 777) reg expression
s_mov_b32 s5, s31 // (op 777) reg expression
s_mov_b32 s6, s2 // (op 777) reg expression
s_mov_b32 s7, 131072 // (op 777) call()




buffer_store_dwordx4 v[12:15], v4, s[4:7], 0 offen // (op 48) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_114:
v_lshrrev_b32 v0, 7, v1 // (op 650) 
v_and_b32 v3, 63, v1 // (op 650) 
v_lshrrev_b32 v4, 4, v3 // (op 650) 
v_lshlrev_b32 v5, 2, v4 // (op 650) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 650) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 650) 

v_lshrrev_b32 v4, 2, v3 // (op 650) 

v_and_b32 v3, 3, v4 // (op 650) 

v_and_b32 v4, 3, v1 // (op 650) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 650) 


v_lshrrev_b32 v3, 6, v1 // (op 650) 
v_and_b32 v4, 1, v3 // (op 650) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 650) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 650) 

v_mul_lo_u32 v3, v4, s34 // (op 650) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 650) 


s_mul_i32 s1, s34, s33 // (op 652) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 652) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 652) least significant
s_mul_hi_u32 s3, s34, s32 // (op 652) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 652) most significant: sum
s_add_u32 s1, s1, s3 // (op 652) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 652) 

s_mov_b32 s8, s36 // (op 652) reg expression
s_mov_b32 s9, s37 // (op 652) reg expression
s_mov_b32 s10, s2 // (op 652) reg expression
s_mov_b32 s11, 131072 // (op 652) call()





buffer_load_dwordx4 v[8:11], v4, s[8:11], 0 offen // (op 126) Load value



v_accvgpr_read v12, a0 // (op 123) 
v_accvgpr_read v13, a1 // (op 123) 
v_accvgpr_read v14, a2 // (op 123) 
v_accvgpr_read v15, a3 // (op 123) 
v_mul_f32 v16, v7, v12 // (op 123) 
v_mul_f32 v17, v7, v13 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v12, v6, v8 // (op 127) 
v_mul_f32 v13, v6, v9 // (op 127) 
v_mul_f32 v14, v6, v10 // (op 127) 
v_mul_f32 v15, v6, v11 // (op 127) 

v_add_f32 v8, v12, v16 // (op 124) 
v_add_f32 v9, v13, v17 // (op 124) 
v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 


v_lshrrev_b32 v0, 7, v1 // (op 792) 
v_and_b32 v3, 63, v1 // (op 792) 
v_lshrrev_b32 v4, 4, v3 // (op 792) 
v_lshlrev_b32 v5, 2, v4 // (op 792) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 792) 


v_lshl_add_u32 v0, s16, 5, v4 // (op 792) 

v_lshrrev_b32 v4, 2, v3 // (op 792) 

v_and_b32 v3, 3, v4 // (op 792) 

v_and_b32 v4, 3, v1 // (op 792) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 792) 


v_lshrrev_b32 v3, 6, v1 // (op 792) 
v_and_b32 v4, 1, v3 // (op 792) 

v_lshl_add_u32 v3, v4, 4, v5 // (op 792) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 792) 

v_mul_lo_u32 v3, v4, s28 // (op 792) 

v_add_lshl_u32 v4, v0, v3, 2 // (op 792) 


s_mul_i32 s1, s28, s33 // (op 794) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 794) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 794) least significant
s_mul_hi_u32 s3, s28, s32 // (op 794) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 794) most significant: sum
s_add_u32 s1, s1, s3 // (op 794) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 794) 

s_mov_b32 s4, s30 // (op 794) reg expression
s_mov_b32 s5, s31 // (op 794) reg expression
s_mov_b32 s6, s2 // (op 794) reg expression
s_mov_b32 s7, 131072 // (op 794) call()




buffer_store_dwordx4 v[8:11], v4, s[4:7], 0 offen // (op 125) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_114:



s_endpgm  // End of RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16384 // lds bytes
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
    macrotile: [32, 32, 128]
    threads: [256, 1, 1]
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
  - .name: RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_NT_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 48
    .agpr_count: 4
    .max_flat_workgroup_size: 256
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

