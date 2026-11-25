// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 6, v1 // (op 877) 
v_and_b32 v3, 15, v1 // (op 877) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 877) 


v_and_b32 v0, 63, v1 // (op 877) 
v_lshrrev_b32 v3, 4, v0 // (op 877) 

v_lshlrev_b32 v0, 5, v3 // (op 877) 

v_lshl_add_u32 v3, v4, 8, v0 // (op 877) 


v_mul_lo_u32 v0, v3, 6 // (op 877) 

v_lshrrev_b32 v3, 3, v0 // (op 877) 

v_and_b32 v0, 48, v1 // (op 929) 
v_and_b32 v4, 63, v1 // (op 929) 
v_sub_u32 v5, v4, v0 // (op 929) 

v_lshrrev_b32 v4, 1, v5 // (op 929) 
v_add_lshl_u32 v6, v0, v4, 4 // (op 929) 


v_and_b32 v0, 1, v5 // (op 929) 

v_lshlrev_b32 v4, 3, v0 // (op 929) 

v_add_u32 v0, v6, v4 // (op 929) 


v_lshrrev_b32 v4, 6, v1 // (op 982) 
v_and_b32 v5, 15, v1 // (op 982) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 982) 


v_and_b32 v4, 63, v1 // (op 982) 
v_lshrrev_b32 v5, 4, v4 // (op 982) 

v_lshl_add_u32 v4, v6, 3, v5 // (op 982) 


v_and_b32 v5, 48, v1 // (op 1035) 
v_and_b32 v6, 15, v1 // (op 1035) 
v_add_u32 v7, v5, v6 // (op 1035) 


v_lshrrev_b32 v5, 6, v1 // (op 1093) 
v_and_b32 v6, 15, v1 // (op 1093) 
v_lshl_add_u32 v8, v5, 4, v6 // (op 1093) 


v_and_b32 v5, 63, v1 // (op 1093) 
v_lshrrev_b32 v6, 4, v5 // (op 1093) 

v_lshlrev_b32 v5, 5, v6 // (op 1093) 

v_lshl_add_u32 v6, v8, 8, v5 // (op 1093) 


v_mul_lo_u32 v5, v6, 6 // (op 1093) 

v_lshrrev_b32 v6, 3, v5 // (op 1093) 

v_and_b32 v5, 48, v1 // (op 1132) 
v_and_b32 v8, 63, v1 // (op 1132) 
v_sub_u32 v9, v8, v5 // (op 1132) 

v_lshrrev_b32 v8, 1, v9 // (op 1132) 
v_add_lshl_u32 v10, v5, v8, 4 // (op 1132) 


v_and_b32 v5, 1, v9 // (op 1132) 

v_lshlrev_b32 v8, 3, v5 // (op 1132) 

v_add_u32 v5, v10, v8 // (op 1132) 


v_lshrrev_b32 v8, 6, v1 // (op 1156) 
v_and_b32 v9, 15, v1 // (op 1156) 
v_lshl_add_u32 v10, v8, 4, v9 // (op 1156) 


v_and_b32 v8, 63, v1 // (op 1156) 
v_lshrrev_b32 v9, 4, v8 // (op 1156) 

v_lshl_add_u32 v8, v10, 3, v9 // (op 1156) 


v_and_b32 v9, 48, v1 // (op 1180) 
v_and_b32 v10, 15, v1 // (op 1180) 
v_add_u32 v11, v9, v10 // (op 1180) 


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_lshrrev_b32 v12, 2, v1 // (op 587) 
v_lshl_add_u32 v13, s16, 4, v12 // (op 587) 

v_and_b32 v12, -4, v1 // (op 587) 
v_sub_u32 v14, v1, v12 // (op 587) 

v_lshlrev_b32 v12, 6, v14 // (op 587) 

v_mov_b32 v15, 0 // (op 587) convert
v_mov_b32 v14, v12 // (op 587) convert

v_mul_lo_u32 v17, v13, s3 // (op 587) most significant: low of low * high
v_mul_hi_u32 v12, v13, s2 // (op 587) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 587) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 587) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 587) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 587) most significant half



v_mul_lo_u32 v13, v17, 6 // (op 587) most significant: low of high * low
v_mul_hi_u32 v14, v16, 6 // (op 587) most significant: high of low * low
v_mul_lo_u32 v12, v16, 6 // (op 587) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 587) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 587) 

v_mov_b32 v12, v14 // (op 587) convert

s_mul_i32 s1, s7, 6 // (op 589) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 589) least significant
s_mul_hi_u32 s14, s6, 6 // (op 589) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 589) most significant: sum

s_mul_i32 s15, s2, s1 // (op 589) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 589) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 589) least significant
s_mul_hi_u32 s19, s2, s0 // (op 589) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 589) most significant: sum
s_add_u32 s15, s15, s19 // (op 589) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 589) 

s_mov_b32 s40, s4 // (op 589) reg expression
s_mov_b32 s41, s5 // (op 589) reg expression
s_mov_b32 s42, s0 // (op 589) reg expression
s_mov_b32 s43, 131072 // (op 589) call()

v_lshrrev_b32 v13, 2, v1 // (op 606) 
v_lshl_add_u32 v14, s16, 4, v13 // (op 606) 

v_and_b32 v13, -4, v1 // (op 606) 
v_sub_u32 v15, v1, v13 // (op 606) 

v_lshlrev_b32 v13, 6, v15 // (op 606) 

v_mov_b32 v17, 0 // (op 606) convert
v_mov_b32 v16, v13 // (op 606) convert

v_mul_lo_u32 v19, v14, s3 // (op 606) most significant: low of low * high
v_mul_hi_u32 v13, v14, s2 // (op 606) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 606) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 606) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 606) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 606) most significant half



v_mul_lo_u32 v17, v19, 6 // (op 606) most significant: low of high * low
v_mul_hi_u32 v13, v18, 6 // (op 606) most significant: high of low * low
v_mul_lo_u32 v16, v18, 6 // (op 606) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 606) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 606) 

v_mov_b32 v13, v14 // (op 606) convert

s_mul_i32 s1, s7, 6 // (op 608) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 608) least significant
s_mul_hi_u32 s14, s6, 6 // (op 608) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 608) most significant: sum

s_mul_i32 s15, s2, s1 // (op 608) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 608) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 608) least significant
s_mul_hi_u32 s19, s2, s0 // (op 608) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 608) most significant: sum
s_add_u32 s15, s15, s19 // (op 608) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 608) 

s_mov_b32 s40, s4 // (op 608) reg expression
s_mov_b32 s41, s5 // (op 608) reg expression
s_mov_b32 s42, s0 // (op 608) reg expression
s_mov_b32 s43, 131072 // (op 608) call()

s_lshl_b32 s0, s17, 4 // (op 651) 
s_mov_b32 s15, 0 // (op 651) convert
s_mov_b32 s14, s0 // (op 651) convert

v_mul_lo_u32 v15, v1, s9 // (op 651) most significant: low of low * high
v_mul_hi_u32 v16, v1, s8 // (op 651) most significant: high of low * low
v_mul_lo_u32 v14, v1, s8 // (op 651) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 651) most significant: sum

v_mov_b32 v16, s14 // (op 651) 
v_mov_b32 v17, s15 // (op 651) 
v_add_co_u32 v14, s[0:1], v14, v16 // (op 651) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 651) most significant half




s_mul_i32 s1, s8, s13 // (op 653) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 653) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 653) least significant
s_mul_hi_u32 s15, s8, s12 // (op 653) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 653) most significant: sum
s_add_u32 s1, s1, s15 // (op 653) most significant: sum


s_mov_b32 s44, s10 // (op 653) reg expression
s_mov_b32 s45, s11 // (op 653) reg expression
s_mov_b32 s46, s0 // (op 653) reg expression
s_mov_b32 s47, 131072 // (op 653) call()

s_lshl_b32 s0, s17, 4 // (op 670) 
s_mov_b32 s15, 0 // (op 670) convert
s_mov_b32 s14, s0 // (op 670) convert

v_mul_lo_u32 v17, v1, s9 // (op 670) most significant: low of low * high
v_mul_hi_u32 v18, v1, s8 // (op 670) most significant: high of low * low
v_mul_lo_u32 v16, v1, s8 // (op 670) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 670) most significant: sum

v_mov_b32 v18, s14 // (op 670) 
v_mov_b32 v19, s15 // (op 670) 
v_add_co_u32 v16, s[0:1], v16, v18 // (op 670) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 670) most significant half




s_mul_i32 s1, s8, s13 // (op 672) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 672) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 672) least significant
s_mul_hi_u32 s15, s8, s12 // (op 672) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 672) most significant: sum
s_add_u32 s1, s1, s15 // (op 672) most significant: sum


s_mov_b32 s44, s10 // (op 672) reg expression
s_mov_b32 s45, s11 // (op 672) reg expression
s_mov_b32 s46, s0 // (op 672) reg expression
s_mov_b32 s47, 131072 // (op 672) call()

v_lshrrev_b32 v18, 3, v1 // (op 715) 
v_lshlrev_b32 v19, 1, v18 // (op 715) 

v_lshl_add_u32 v18, s16, 4, v19 // (op 715) 

v_and_b32 v19, -8, v1 // (op 715) 
v_sub_u32 v20, v1, v19 // (op 715) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v19, v18, s20 // (op 715) 
v_add_u32 v19, v19, v20 // (op 715) 


s_mul_i32 s1, s20, s7 // (op 717) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 717) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 717) least significant
s_mul_hi_u32 s15, s20, s6 // (op 717) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 717) most significant: sum
s_add_u32 s1, s1, s15 // (op 717) most significant: sum


s_mov_b32 s48, s22 // (op 717) reg expression
s_mov_b32 s49, s23 // (op 717) reg expression
s_mov_b32 s50, s0 // (op 717) reg expression
s_mov_b32 s51, 131072 // (op 717) call()

v_lshrrev_b32 v18, 3, v1 // (op 734) 
v_lshlrev_b32 v20, 1, v18 // (op 734) 

v_lshl_add_u32 v18, s16, 4, v20 // (op 734) 

v_and_b32 v20, -8, v1 // (op 734) 
v_sub_u32 v21, v1, v20 // (op 734) 

v_mul_lo_u32 v20, v18, s20 // (op 734) 
v_add_u32 v20, v20, v21 // (op 734) 


s_mul_i32 s1, s20, s7 // (op 736) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 736) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 736) least significant
s_mul_hi_u32 s15, s20, s6 // (op 736) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 736) most significant: sum
s_add_u32 s1, s1, s15 // (op 736) most significant: sum


s_mov_b32 s48, s22 // (op 736) reg expression
s_mov_b32 s49, s23 // (op 736) reg expression
s_mov_b32 s50, s0 // (op 736) reg expression
s_mov_b32 s51, 131072 // (op 736) call()

v_and_b32 v18, -16, v1 // (op 779) 
v_sub_u32 v21, v1, v18 // (op 779) 

v_lshl_add_u32 v18, s17, 4, v21 // (op 779) 

v_lshrrev_b32 v21, 4, v1 // (op 779) 
v_lshlrev_b32 v22, 1, v21 // (op 779) 

v_mul_lo_u32 v21, v22, s24 // (op 779) 
v_add_u32 v21, v21, v18 // (op 779) 


s_mul_i32 s1, s24, s29 // (op 781) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 781) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 781) least significant
s_mul_hi_u32 s15, s24, s28 // (op 781) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 781) most significant: sum
s_add_u32 s1, s1, s15 // (op 781) most significant: sum


s_mov_b32 s52, s26 // (op 781) reg expression
s_mov_b32 s53, s27 // (op 781) reg expression
s_mov_b32 s54, s0 // (op 781) reg expression
s_mov_b32 s55, 131072 // (op 781) call()

v_and_b32 v18, -16, v1 // (op 798) 
v_sub_u32 v22, v1, v18 // (op 798) 

v_lshl_add_u32 v18, s17, 4, v22 // (op 798) 

v_lshrrev_b32 v22, 4, v1 // (op 798) 
v_lshlrev_b32 v23, 1, v22 // (op 798) 

v_mul_lo_u32 v22, v23, s24 // (op 798) 
v_add_u32 v22, v22, v18 // (op 798) 


s_mul_i32 s1, s24, s29 // (op 800) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 800) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 800) least significant
s_mul_hi_u32 s15, s24, s28 // (op 800) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 800) most significant: sum
s_add_u32 s1, s1, s15 // (op 800) most significant: sum


s_mov_b32 s52, s26 // (op 800) reg expression
s_mov_b32 s53, s27 // (op 800) reg expression
s_mov_b32 s54, s0 // (op 800) reg expression
s_mov_b32 s55, 131072 // (op 800) call()

v_lshlrev_b32 v18, 4, v1 // (op 917) 
s_nop 0
v_readfirstlane_b32 s0, v18 // (op 917) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 6 // (op 543) 
v_mov_b64 v[24:25], s[14:15] // (op 543) convert


v_mov_b64 v[26:27], v[14:15] // (op 543) reg expression
s_mov_b32 m0, s0 // (op 543) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 543) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 543) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 543) most significant half

s_add_u32 m0, m0, 1024 // (op 543) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 543) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 543) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 543) most significant half

s_add_u32 m0, m0, 1024 // (op 543) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 543) Load value direct to lds
v_add_co_u32 v26, s[14:15], v26, v24 // (op 543) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 543) most significant half

s_add_u32 m0, m0, 1024 // (op 543) 
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 543) Load value direct to lds




v_mov_b64 v[24:25], s[24:25] // (op 431) convert

v_mov_b32 v18, v21 // (op 431) reg expression
buffer_load_ubyte v26, v18, s[52:55], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v18, v18, v24 // (op 431) 
buffer_load_ubyte v27, v18, s[52:55], 0 offen offset: 0 // (op 431) Load value





buffer_load_dwordx4 v[28:31], v12, s[40:43], 0 offen // (op 434) Load value
buffer_load_dwordx2 v[32:33], v12, s[40:43], 0 offen offset:16 // (op 434) Load value
buffer_load_dwordx4 v[34:37], v12, s[40:43], 0 offen offset:24 // (op 434) Load value
buffer_load_dwordx2 v[38:39], v12, s[40:43], 0 offen offset:40 // (op 434) Load value

v_mov_b64 v[24:25], s[20:21] // (op 437) convert

v_mov_b32 v18, v19 // (op 437) 
buffer_load_ubyte v40, v18, s[48:51], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v18, v18, v24 // (op 437) 
buffer_load_ubyte v41, v18, s[48:51], 0 offen offset: 0 // (op 437) Load value


v_lshlrev_b32 v18, 4, v1 // (op 1120) 
s_nop 0
v_readfirstlane_b32 s0, v18 // (op 1120) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[8:9], 6 // (op 548) 
v_mov_b64 v[24:25], s[14:15] // (op 548) convert


s_lshl_b64 s[14:15], s[8:9], 8 // (op 548) 
v_mov_b32 v18, s14 // (op 548) 
v_mov_b32 v23, s15 // (op 548) 
v_add_co_u32 v42, s[18:19], v14, v18 // (op 548) least significant half
v_addc_co_u32 v43, s[18:19], v15, v23, s[18:19] // (op 548) most significant half




s_add_u32 s1, 4096, s0 // (op 548) 
s_mov_b32 m0, s1 // (op 548) reg expression
buffer_load_dwordx4 v42, s[44:47], 0 offen offset: 0 lds // (op 548) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v24 // (op 548) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 548) most significant half

s_add_u32 m0, m0, 1024 // (op 548) 
buffer_load_dwordx4 v42, s[44:47], 0 offen offset: 0 lds // (op 548) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v24 // (op 548) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 548) most significant half

s_add_u32 m0, m0, 1024 // (op 548) 
buffer_load_dwordx4 v42, s[44:47], 0 offen offset: 0 lds // (op 548) Load value direct to lds
v_add_co_u32 v42, s[14:15], v42, v24 // (op 548) least significant half
v_addc_co_u32 v43, s[14:15], 0, v43, s[14:15] // (op 548) most significant half

s_add_u32 m0, m0, 1024 // (op 548) 
buffer_load_dwordx4 v42, s[44:47], 0 offen offset: 0 lds // (op 548) Load value direct to lds






v_mov_b64 v[14:15], s[24:25] // (op 443) convert

s_lshl_b32 s0, s24, 3 // (op 443) 
v_add_u32 v18, v21, s0 // (op 443) 

buffer_load_ubyte v24, v18, s[52:55], 0 offen offset: 0 // (op 443) Load value
v_add_u32 v18, v18, v14 // (op 443) 
buffer_load_ubyte v25, v18, s[52:55], 0 offen offset: 0 // (op 443) Load value






buffer_load_dwordx4 v[42:45], v12, s[40:43], 0 offen offset:192 // (op 446) Load value
buffer_load_dwordx2 v[46:47], v12, s[40:43], 0 offen offset:208 // (op 446) Load value
buffer_load_dwordx4 v[48:51], v12, s[40:43], 0 offen offset:216 // (op 446) Load value
buffer_load_dwordx2 v[52:53], v12, s[40:43], 0 offen offset:232 // (op 446) Load value


v_mov_b64 v[14:15], s[20:21] // (op 449) convert

v_mov_b32 v12, v19 // (op 449) 
buffer_load_ubyte v54, v12, s[48:51], 0 offen offset:8 // (op 449) Load value
v_add_u32 v12, v12, v14 // (op 449) 
buffer_load_ubyte v55, v12, s[48:51], 0 offen offset:8 // (op 449) Load value



v_and_b32 v12, -16, v1 // (op 1023) 
v_sub_u32 v21, v1, v12 // (op 1023) 

v_lshrrev_b32 v12, 4, v1 // (op 1023) 
v_lshl_add_u32 v23, v12, 5, v21 // (op 1023) 





s_waitcnt vmcnt(19)
ds_write_b8 v23, v26 offset:8192 // (op 452) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v23, v27 offset:8208 // (op 452) Store local data 

v_and_b32 v12, -4, v1 // (op 865) 
v_sub_u32 v21, v1, v12 // (op 865) 

v_lshlrev_b32 v12, 6, v21 // (op 865) 

v_lshrrev_b32 v21, 2, v1 // (op 865) 
v_lshl_add_u32 v23, v21, 8, v12 // (op 865) 


v_mul_lo_u32 v12, v23, 6 // (op 865) 

v_lshrrev_b32 v21, 3, v12 // (op 865) 




s_waitcnt vmcnt(17)
ds_write_b128 v21, v[28:31] offset:8320 // (op 453) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v21, v[32:33] offset:8336 // (op 453) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v21, v[34:37] offset:8344 // (op 453) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v21, v[38:39] offset:8360 // (op 453) Store local data 

v_and_b32 v12, -8, v1 // (op 970) 
v_sub_u32 v21, v1, v12 // (op 970) 

v_lshrrev_b32 v12, 3, v1 // (op 970) 
v_lshl_add_u32 v23, v12, 4, v21 // (op 970) 





s_waitcnt vmcnt(13)
ds_write_b8 v23, v40 offset:11392 // (op 454) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v23, v41 offset:11400 // (op 454) Store local data 

s_barrier  // (op 1236) 



ds_read_b64_tr_b8 v[56:57], v0 // (op 471) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:128 // (op 471) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:1024 // (op 471) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:1152 // (op 471) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1272) 


ds_read_u8 v12, v7 offset:8192 // (op 477) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1217) 



ds_read_b128 v[64:67], v3 offset:8320 // (op 468) Load local data 
ds_read_b64 v[68:69], v3 offset:8336 // (op 468) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1252) 



ds_read_u8 v21, v4 offset:11392 // (op 474) Load local data 
v_and_b32 v23, -4, v1 // (op 836) 
v_sub_u32 v14, v1, v23 // (op 836) 

v_lshlrev_b32 v15, 6, v14 // (op 836) 

v_lshrrev_b32 v14, 2, v1 // (op 836) 
v_lshl_add_u32 v23, v14, 8, v15 // (op 836) 


v_mul_lo_u32 v14, v23, 6 // (op 836) 

v_lshrrev_b32 v15, 3, v14 // (op 836) 

v_lshlrev_b32 v14, 4, v1 // (op 889) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 889) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 941) 
v_sub_u32 v23, v1, v14 // (op 941) 

v_lshrrev_b32 v14, 3, v1 // (op 941) 
v_lshl_add_u32 v18, v14, 4, v23 // (op 941) 


v_and_b32 v14, -16, v1 // (op 994) 
v_sub_u32 v19, v1, v14 // (op 994) 

v_lshrrev_b32 v14, 4, v1 // (op 994) 
v_lshl_add_u32 v23, v14, 5, v19 // (op 994) 


v_and_b32 v14, -4, v1 // (op 1081) 
v_sub_u32 v19, v1, v14 // (op 1081) 

v_lshlrev_b32 v14, 6, v19 // (op 1081) 

v_lshrrev_b32 v19, 2, v1 // (op 1081) 
v_lshl_add_u32 v70, v19, 8, v14 // (op 1081) 


v_mul_lo_u32 v14, v70, 6 // (op 1081) 

v_lshrrev_b32 v19, 3, v14 // (op 1081) 

v_lshlrev_b32 v14, 4, v1 // (op 1105) 
s_nop 0
v_readfirstlane_b32 s1, v14 // (op 1105) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 1144) 
v_sub_u32 v70, v1, v14 // (op 1144) 

v_lshrrev_b32 v14, 3, v1 // (op 1144) 
v_lshl_add_u32 v71, v14, 4, v70 // (op 1144) 


v_and_b32 v14, -16, v1 // (op 1168) 
v_sub_u32 v70, v1, v14 // (op 1168) 

v_lshrrev_b32 v14, 4, v1 // (op 1168) 
v_lshl_add_u32 v72, v14, 5, v70 // (op 1168) 


s_mov_b32 s14, 0 // (op 61) call()
s_ashr_i64 s[18:19], s[12:13], 8 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60:
s_add_u32 s14, s14, 2 // (op 62) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[56:63], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 90) 
s_barrier  // (op 1239) 

s_barrier  // (op 1226) 
s_lshl_b64 s[18:19], s[8:9], 6 // (op 526) 



ds_read_b128 v[64:67], v3 offset:8416 // (op 93) Load local data 
ds_read_b64 v[68:69], v3 offset:8432 // (op 93) Load local data 
v_mov_b64 v[74:75], s[18:19] // (op 526) convert


s_barrier  // (op 1245) 
v_mov_b64 v[76:77], v[16:17] // (op 526) 



ds_read_b64_tr_b8 v[56:57], v0 offset:2048 // (op 97) Transpose load local data 
s_lshl_b64 s[18:19], s[8:9], 8 // (op 526) 
s_lshl_b64 s[56:57], s[18:19], 1 // (op 526) 

v_mov_b32 v14, s56 // (op 526) 
v_mov_b32 v70, s57 // (op 526) 
v_add_co_u32 v78, s[18:19], v76, v14 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], v77, v70, s[18:19] // (op 526) most significant half




s_mov_b32 m0, s0 // (op 526) reg expression
buffer_load_dwordx4 v78, s[44:47], 0 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v74 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v78, s[44:47], 0 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v74 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
buffer_load_dwordx4 v78, s[44:47], 0 offen offset: 0 lds // (op 526) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v74 // (op 526) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 526) most significant half

s_add_u32 m0, m0, 1024 // (op 526) 
ds_read_b64_tr_b8 v[58:59], v0 offset:2176 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:3072 // (op 97) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:3200 // (op 97) Transpose load local data 
buffer_load_dwordx4 v78, s[44:47], 0 offen offset: 0 lds // (op 526) Load value direct to lds



s_barrier  // (op 1262) 
v_mov_b64 v[74:75], s[24:25] // (op 25) convert




ds_read_u8 v21, v4 offset:11396 // (op 101) Load local data 
v_mov_b32 v14, v22 // (op 25) 
s_barrier  // (op 1282) 
s_lshl_b32 s15, s24, 4 // (op 25) 


ds_read_u8 v12, v7 offset:8256 // (op 105) Load local data 
v_add_u32 v70, v14, s15 // (op 25) 

buffer_load_ubyte v26, v70, s[52:55], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v70, v70, v74 // (op 25) 
buffer_load_ubyte v27, v70, s[52:55], 0 offen offset: 0 // (op 25) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:69], v[56:63], a[0:3], v21, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 107) 


v_mov_b32 v70, v13 // (op 4) 

buffer_load_dwordx4 v[28:31], v70, s[40:43], 0 offen offset:384 // (op 4) Load value
buffer_load_dwordx2 v[32:33], v70, s[40:43], 0 offen offset:400 // (op 4) Load value
buffer_load_dwordx4 v[34:37], v70, s[40:43], 0 offen offset:408 // (op 4) Load value
buffer_load_dwordx2 v[38:39], v70, s[40:43], 0 offen offset:424 // (op 4) Load value

v_mov_b64 v[74:75], s[20:21] // (op 17) convert

v_mov_b32 v73, v20 // (op 17) 
v_mov_b32 v78, v73 // (op 17) 
buffer_load_ubyte v40, v78, s[48:51], 0 offen offset:16 // (op 17) Load value
v_add_u32 v78, v78, v74 // (op 17) 
buffer_load_ubyte v41, v78, s[48:51], 0 offen offset:16 // (op 17) Load value


s_barrier  // (op 1317) 



s_waitcnt vmcnt(19)
ds_write_b8 v72, v24 offset:11520 // (op 308) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v72, v25 offset:11536 // (op 308) Store local data 
s_barrier  // (op 1295) 



s_waitcnt vmcnt(17)
ds_write_b128 v19, v[42:45] offset:11648 // (op 287) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v19, v[46:47] offset:11664 // (op 287) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v19, v[48:51] offset:11672 // (op 287) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v19, v[52:53] offset:11688 // (op 287) Store local data 
s_barrier  // (op 1311) 



s_waitcnt vmcnt(13)
ds_write_b8 v71, v54 offset:14720 // (op 307) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v71, v55 offset:14728 // (op 307) Store local data 
s_barrier  // (op 1298) 



ds_read_b64_tr_b8 v[78:79], v5 offset:4096 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[80:81], v5 offset:4224 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v5 offset:5120 // (op 291) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v5 offset:5248 // (op 291) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1314) 


ds_read_u8 v74, v11 offset:11520 // (op 295) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1292) 



ds_read_b128 v[86:89], v6 offset:11648 // (op 289) Load local data 
ds_read_b64 v[90:91], v6 offset:11664 // (op 289) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1308) 



ds_read_u8 v75, v8 offset:14720 // (op 293) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:91], v[78:85], a[0:3], v75, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 296) 
s_barrier  // (op 1302) 








s_lshl_b64 s[18:19], s[8:9], 6 // (op 538) 
v_mov_b64 v[74:75], s[18:19] // (op 538) convert


s_lshl_b64 s[18:19], s[8:9], 8 // (op 538) 
s_mul_i32 s57, 3, s19 // (op 538) most significant: low of low * high
s_mul_i32 s56, 3, s18 // (op 538) least significant
s_mul_hi_u32 s15, 3, s18 // (op 538) most significant: high of low * low
s_add_u32 s57, s57, s15 // (op 538) most significant: sum


v_mov_b32 v78, s56 // (op 538) 
v_mov_b32 v79, s57 // (op 538) 
ds_read_b128 v[80:83], v6 offset:11744 // (op 298) Load local data 
ds_read_b64 v[84:85], v6 offset:11760 // (op 298) Load local data 
v_add_co_u32 v86, s[18:19], v76, v78 // (op 538) least significant half
v_addc_co_u32 v87, s[18:19], v77, v79, s[18:19] // (op 538) most significant half




s_barrier  // (op 1305) 
s_barrier  // (op 1376) 



s_add_u32 s15, 4096, s1 // (op 538) 
s_mov_b32 m0, s15 // (op 538) reg expression
buffer_load_dwordx4 v86, s[44:47], 0 offen offset: 0 lds // (op 538) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v74 // (op 538) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 538) most significant half

s_add_u32 m0, m0, 1024 // (op 538) 
buffer_load_dwordx4 v86, s[44:47], 0 offen offset: 0 lds // (op 538) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v74 // (op 538) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 538) most significant half

s_add_u32 m0, m0, 1024 // (op 538) 
buffer_load_dwordx4 v86, s[44:47], 0 offen offset: 0 lds // (op 538) Load value direct to lds
v_add_co_u32 v86, s[18:19], v86, v74 // (op 538) least significant half
v_addc_co_u32 v87, s[18:19], 0, v87, s[18:19] // (op 538) most significant half

s_add_u32 m0, m0, 1024 // (op 538) 
ds_read_b64_tr_b8 v[88:89], v5 offset:6144 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v5 offset:6272 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[92:93], v5 offset:7168 // (op 300) Transpose load local data 
ds_read_b64_tr_b8 v[94:95], v5 offset:7296 // (op 300) Transpose load local data 
buffer_load_dwordx4 v86, s[44:47], 0 offen offset: 0 lds // (op 538) Load value direct to lds








ds_read_u8 v74, v8 offset:14724 // (op 302) Load local data 
v_mov_b64 v[86:87], s[24:25] // (op 311) convert

s_lshl_b32 s15, s24, 3 // (op 311) 


s_barrier  // (op 1382) 
ds_read_u8 v75, v11 offset:11584 // (op 304) Load local data 
s_mul_i32 s18, 3, s15 // (op 311) 

v_add_u32 v76, v14, s18 // (op 311) 

s_barrier  // (op 1386) 
buffer_load_ubyte v24, v76, s[52:55], 0 offen offset: 0 // (op 311) Load value
v_add_u32 v76, v76, v86 // (op 311) 
buffer_load_ubyte v25, v76, s[52:55], 0 offen offset: 0 // (op 311) Load value


s_lshl_b64 s[18:19], s[8:9], 8 // (op 662) 

s_lshl_b64 s[56:57], s[18:19], 1 // (op 662) 




s_lshl_b32 s15, s24, 4 // (op 790) 
v_add_u32 v22, v22, s15 // (op 790) 

v_mov_b32 v14, s56 // (op 662) 
v_mov_b32 v76, s57 // (op 662) 
v_add_co_u32 v16, s[18:19], v16, v14 // (op 662) least significant half
v_addc_co_u32 v17, s[18:19], v17, v76, s[18:19] // (op 662) most significant half




s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:85], v[88:95], a[0:3], v74, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 305) 
buffer_load_dwordx4 v[42:45], v70, s[40:43], 0 offen offset:576 // (op 286) Load value




buffer_load_dwordx2 v[46:47], v70, s[40:43], 0 offen offset:592 // (op 286) Load value
buffer_load_dwordx4 v[48:51], v70, s[40:43], 0 offen offset:600 // (op 286) Load value
buffer_load_dwordx2 v[52:53], v70, s[40:43], 0 offen offset:616 // (op 286) Load value


v_add_u32 v13, 384, v13 // (op 598) 
v_mov_b64 v[74:75], s[20:21] // (op 310) convert

v_mov_b32 v14, v73 // (op 310) 
buffer_load_ubyte v54, v14, s[48:51], 0 offen offset:24 // (op 310) Load value
v_add_u32 v14, v14, v74 // (op 310) 
buffer_load_ubyte v55, v14, s[48:51], 0 offen offset:24 // (op 310) Load value



s_barrier  // (op 1275) 
v_add_u32 v20, 16, v20 // (op 726) 



s_waitcnt vmcnt(19)
ds_write_b8 v23, v26 offset:8192 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v23, v27 offset:8208 // (op 30) Store local data 
s_barrier  // (op 1220) 



s_waitcnt vmcnt(17)
ds_write_b128 v15, v[28:31] offset:8320 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v15, v[32:33] offset:8336 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v15, v[34:37] offset:8344 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v15, v[38:39] offset:8360 // (op 6) Store local data 
s_barrier  // (op 1255) 



s_waitcnt vmcnt(13)
ds_write_b8 v18, v40 offset:11392 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v18, v41 offset:11400 // (op 21) Store local data 



ds_read_b64_tr_b8 v[56:57], v0 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[58:59], v0 offset:128 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[60:61], v0 offset:1024 // (op 77) Transpose load local data 
ds_read_b64_tr_b8 v[62:63], v0 offset:1152 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 1285) 


ds_read_u8 v12, v7 offset:8192 // (op 87) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1229) 



ds_read_b128 v[64:67], v3 offset:8320 // (op 72) Load local data 
ds_read_b64 v[68:69], v3 offset:8336 // (op 72) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1265) 



ds_read_u8 v21, v4 offset:11392 // (op 82) Load local data 
s_ashr_i64 s[18:19], s[12:13], 8 // (op 60) 
s_mov_b32 s15, s18 // (op 60) convert

s_and_b32 s18, s15, 4294967294 // (op 60) 

s_cmp_lt_u32 s14, s18 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60:
















s_barrier  // (op 1332) 
v_lshrrev_b32 v13, 2, v1 // (op 561) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 561) 
v_lshl_add_u32 v14, s16, 4, v13 // (op 561) 

s_lshr_b32 s14, s0, 1 // (op 561) 

v_and_b32 v13, -4, v1 // (op 561) 
v_sub_u32 v15, v1, v13 // (op 561) 

v_lshlrev_b32 v13, 6, v15 // (op 561) 

v_lshl_add_u32 v15, s14, 9, v13 // (op 561) 


v_mov_b32 v23, 0 // (op 561) convert
v_mov_b32 v22, v15 // (op 561) convert

v_mul_lo_u32 v27, v14, s3 // (op 561) most significant: low of low * high
v_mul_hi_u32 v13, v14, s2 // (op 561) most significant: high of low * low
v_mul_lo_u32 v26, v14, s2 // (op 561) least significant: low of low * low
v_add_u32 v27, v27, v13 // (op 561) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v22 // (op 561) least significant half
v_addc_co_u32 v27, s[0:1], v27, v23, s[0:1] // (op 561) most significant half



v_mul_lo_u32 v23, v27, 6 // (op 561) most significant: low of high * low
v_mul_hi_u32 v13, v26, 6 // (op 561) most significant: high of low * low
v_mul_lo_u32 v22, v26, 6 // (op 561) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 561) most significant: sum


v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 561) 

v_mov_b32 v13, v26 // (op 561) convert

s_mul_i32 s1, s7, 6 // (op 563) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 563) least significant
s_mul_hi_u32 s14, s6, 6 // (op 563) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 563) most significant: sum

s_mul_i32 s15, s2, s1 // (op 563) most significant: low of low * high
s_mul_i32 s18, s3, s0 // (op 563) most significant: low of high * low
s_mul_i32 s14, s2, s0 // (op 563) least significant
s_mul_hi_u32 s19, s2, s0 // (op 563) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 563) most significant: sum
s_add_u32 s15, s15, s19 // (op 563) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 563) 

s_mov_b32 s40, s4 // (op 563) reg expression
s_mov_b32 s41, s5 // (op 563) reg expression
s_mov_b32 s42, s0 // (op 563) reg expression
s_mov_b32 s43, 131072 // (op 563) call()

s_ashr_i64 s[0:1], s[12:13], 8 // (op 628) 
s_lshr_b32 s14, s0, 1 // (op 628) 

v_lshl_add_u32 v14, s14, 9, v1 // (op 628) 

s_lshl_b32 s0, s17, 4 // (op 628) 
s_mov_b32 s15, 0 // (op 628) convert
s_mov_b32 s14, s0 // (op 628) convert

v_mul_lo_u32 v23, v14, s9 // (op 628) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 628) most significant: high of low * low
v_mul_lo_u32 v22, v14, s8 // (op 628) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 628) most significant: sum

v_mov_b32 v15, s14 // (op 628) 
v_mov_b32 v16, s15 // (op 628) 
v_add_co_u32 v22, s[0:1], v22, v15 // (op 628) least significant half
v_addc_co_u32 v23, s[0:1], v23, v16, s[0:1] // (op 628) most significant half





s_mul_i32 s1, s8, s13 // (op 630) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 630) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 630) least significant
s_mul_hi_u32 s15, s8, s12 // (op 630) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 630) most significant: sum
s_add_u32 s1, s1, s15 // (op 630) most significant: sum


s_mov_b32 s44, s10 // (op 630) reg expression
s_mov_b32 s45, s11 // (op 630) reg expression
s_mov_b32 s46, s0 // (op 630) reg expression
s_mov_b32 s47, 131072 // (op 630) call()

v_lshrrev_b32 v14, 3, v1 // (op 692) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 692) 
v_lshlrev_b32 v15, 1, v14 // (op 692) 

s_lshr_b32 s14, s0, 1 // (op 692) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 692) 

v_and_b32 v15, -8, v1 // (op 692) 
v_sub_u32 v16, v1, v15 // (op 692) 

v_lshl_add_u32 v15, s14, 4, v16 // (op 692) 


v_mul_lo_u32 v16, v14, s20 // (op 692) 
v_add_u32 v16, v16, v15 // (op 692) 


s_mul_i32 s1, s20, s7 // (op 694) most significant: low of low * high
s_mul_i32 s14, s21, s6 // (op 694) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 694) least significant
s_mul_hi_u32 s15, s20, s6 // (op 694) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 694) most significant: sum
s_add_u32 s1, s1, s15 // (op 694) most significant: sum


s_mov_b32 s48, s22 // (op 694) reg expression
s_mov_b32 s49, s23 // (op 694) reg expression
s_mov_b32 s50, s0 // (op 694) reg expression
s_mov_b32 s51, 131072 // (op 694) call()

v_lshrrev_b32 v14, 4, v1 // (op 756) 
s_ashr_i64 s[0:1], s[12:13], 8 // (op 756) 
v_lshlrev_b32 v15, 1, v14 // (op 756) 

s_lshr_b32 s14, s0, 1 // (op 756) 

v_and_b32 v14, -16, v1 // (op 756) 
v_lshl_add_u32 v17, s14, 4, v15 // (op 756) 


v_sub_u32 v15, v1, v14 // (op 756) 

v_lshl_add_u32 v14, s17, 4, v15 // (op 756) 

v_mul_lo_u32 v15, v17, s24 // (op 756) 
v_add_u32 v15, v15, v14 // (op 756) 


s_mul_i32 s1, s24, s29 // (op 758) most significant: low of low * high
s_mul_i32 s14, s25, s28 // (op 758) most significant: low of high * low
s_mul_i32 s0, s24, s28 // (op 758) least significant
s_mul_hi_u32 s15, s24, s28 // (op 758) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 758) most significant: sum
s_add_u32 s1, s1, s15 // (op 758) most significant: sum


s_mov_b32 s52, s26 // (op 758) reg expression
s_mov_b32 s53, s27 // (op 758) reg expression
s_mov_b32 s54, s0 // (op 758) reg expression
s_mov_b32 s55, 131072 // (op 758) call()

v_and_b32 v14, -4, v1 // (op 849) 
v_sub_u32 v17, v1, v14 // (op 849) 

v_lshlrev_b32 v14, 6, v17 // (op 849) 

v_lshrrev_b32 v17, 2, v1 // (op 849) 
v_lshl_add_u32 v18, v17, 8, v14 // (op 849) 


v_mul_lo_u32 v14, v18, 6 // (op 849) 

v_lshrrev_b32 v17, 3, v14 // (op 849) 

v_lshlrev_b32 v14, 4, v1 // (op 902) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 902) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -8, v1 // (op 954) 
v_sub_u32 v18, v1, v14 // (op 954) 

v_lshrrev_b32 v14, 3, v1 // (op 954) 
v_lshl_add_u32 v19, v14, 4, v18 // (op 954) 


v_and_b32 v14, -16, v1 // (op 1007) 
v_sub_u32 v18, v1, v14 // (op 1007) 

v_lshrrev_b32 v14, 4, v1 // (op 1007) 
v_lshl_add_u32 v20, v14, 5, v18 // (op 1007) 


s_ashr_i64 s[14:15], s[12:13], 8 // (op 198) 
s_and_b32 s1, s14, 4294967294 // (op 198) 

s_ashr_i64 s[14:15], s[12:13], 8 // (op 197) 
s_mov_b32 s18, s14 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197:


s_add_u32 s1, s1, 1 // (op 199) 
v_mov_b32 v12, v13 // (op 202) 

buffer_load_dwordx4 v[24:27], v12, s[40:43], 0 offen // (op 202) Load value
buffer_load_dwordx2 v[28:29], v12, s[40:43], 0 offen offset:16 // (op 202) Load value
buffer_load_dwordx4 v[30:33], v12, s[40:43], 0 offen offset:24 // (op 202) Load value
buffer_load_dwordx2 v[34:35], v12, s[40:43], 0 offen offset:40 // (op 202) Load value

v_add_u32 v13, 192, v13 // (op 551) 
s_barrier  // (op 1223) 
s_barrier  // (op 1242) 




s_lshl_b64 s[14:15], s[8:9], 6 // (op 531) 
v_mov_b64 v[36:37], s[14:15] // (op 531) convert


v_mov_b64 v[38:39], v[22:23] // (op 531) 
v_mov_b64 v[40:41], v[38:39] // (op 531) reg expression
s_mov_b32 m0, s0 // (op 531) reg expression
buffer_load_dwordx4 v40, s[44:47], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 531) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v40, s[44:47], 0 offen offset: 0 lds // (op 531) Load value direct to lds
v_add_co_u32 v40, s[14:15], v40, v36 // (op 531) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
s_waitcnt vmcnt(5)
ds_write_b128 v17, v[24:27] offset:8320 // (op 203) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v17, v[28:29] offset:8336 // (op 203) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v17, v[30:33] offset:8344 // (op 203) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v17, v[34:35] offset:8360 // (op 203) Store local data 
buffer_load_dwordx4 v40, s[44:47], 0 offen offset: 0 lds // (op 531) Load value direct to lds

v_add_co_u32 v40, s[14:15], v40, v36 // (op 531) least significant half
v_addc_co_u32 v41, s[14:15], 0, v41, s[14:15] // (op 531) most significant half

s_add_u32 m0, m0, 1024 // (op 531) 
buffer_load_dwordx4 v40, s[44:47], 0 offen offset: 0 lds // (op 531) Load value direct to lds



s_lshl_b64 s[14:15], s[8:9], 8 // (op 620) 
v_mov_b32 v12, s14 // (op 620) 
v_mov_b32 v14, s15 // (op 620) 
v_add_co_u32 v22, s[18:19], v22, v12 // (op 620) least significant half
v_addc_co_u32 v23, s[18:19], v23, v14, s[18:19] // (op 620) most significant half





v_mov_b64 v[24:25], s[20:21] // (op 226) convert

v_mov_b32 v12, v16 // (op 226) 
v_mov_b32 v14, v12 // (op 226) 
buffer_load_ubyte v26, v14, s[48:51], 0 offen offset: 0 // (op 226) Load value
v_add_u32 v14, v14, v24 // (op 226) 
buffer_load_ubyte v27, v14, s[48:51], 0 offen offset: 0 // (op 226) Load value



v_add_u32 v16, 8, v16 // (op 684) 

s_barrier  // (op 1258) 
v_mov_b64 v[24:25], s[24:25] // (op 227) convert




v_mov_b32 v12, v15 // (op 227) 
v_mov_b32 v14, v12 // (op 227) reg expression
buffer_load_ubyte v28, v14, s[52:55], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v14, v14, v24 // (op 227) 
buffer_load_ubyte v29, v14, s[52:55], 0 offen offset: 0 // (op 227) Load value



s_lshl_b32 s14, s24, 3 // (op 748) 
v_add_u32 v15, v15, s14 // (op 748) 

s_waitcnt vmcnt(3)
ds_write_b8 v19, v26 offset:11392 // (op 223) Store local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1232) 
s_waitcnt vmcnt(2)
ds_write_b8 v19, v27 offset:11400 // (op 223) Store local data 




ds_read_b128 v[30:33], v3 offset:8320 // (op 205) Load local data 
s_barrier  // (op 1278) 
ds_read_b64 v[34:35], v3 offset:8336 // (op 205) Load local data 



s_barrier  // (op 1248) 
s_waitcnt vmcnt(1)
ds_write_b8 v20, v28 offset:8192 // (op 224) Store local data 



ds_read_b64_tr_b8 v[36:37], v0 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:128 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:1024 // (op 207) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v0 offset:1152 // (op 207) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v29 offset:8208 // (op 224) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1268) 



ds_read_u8 v12, v4 offset:11392 // (op 209) Load local data 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1288) 


ds_read_u8 v14, v7 offset:8192 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:35], v[36:43], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 212) 







ds_read_b128 v[24:27], v3 offset:8416 // (op 214) Load local data 
ds_read_b64 v[28:29], v3 offset:8432 // (op 214) Load local data 



ds_read_b64_tr_b8 v[30:31], v0 offset:2048 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:2176 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:3072 // (op 216) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:3200 // (op 216) Transpose load local data 



ds_read_u8 v12, v4 offset:11396 // (op 218) Load local data 


ds_read_u8 v14, v7 offset:8256 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:29], v[30:37], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b001 // (op 221) 




s_ashr_i64 s[14:15], s[12:13], 8 // (op 197) 
s_mov_b32 s18, s14 // (op 197) convert

s_cmp_lt_u32 s1, s18 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 if true)
RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197:
























s_barrier  // (op 1368) 








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v4, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v5, a1 // (op 50) 
v_accvgpr_read v6, a2 // (op 50) 
v_accvgpr_read v7, a3 // (op 50) 
v_mul_f32 v12, v10, v4 // (op 50) 
v_mul_f32 v13, v10, v5 // (op 50) 
v_mul_f32 v14, v10, v6 // (op 50) 
v_mul_f32 v15, v10, v7 // (op 50) 

v_cvt_f16_f32 v4, v12 // (op 56) 
v_cvt_f16_f32 v5, v13 // (op 56) 
v_cvt_f16_f32 v6, v14 // (op 56) 
v_cvt_f16_f32 v7, v15 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 1049) 
v_and_b32 v3, 63, v1 // (op 1049) 
v_lshrrev_b32 v8, 4, v3 // (op 1049) 
v_lshlrev_b32 v11, 2, v8 // (op 1049) 

v_lshl_add_u32 v8, v0, 4, v11 // (op 1049) 


v_lshl_add_u32 v0, s16, 4, v8 // (op 1049) 

v_lshrrev_b32 v8, 2, v3 // (op 1049) 

v_and_b32 v3, 3, v8 // (op 1049) 

v_and_b32 v8, 3, v1 // (op 1049) 
v_lshl_add_u32 v11, v3, 2, v8 // (op 1049) 


v_lshl_add_u32 v3, s17, 4, v11 // (op 1049) 

v_mul_lo_u32 v8, v3, s30 // (op 1049) 

v_add_lshl_u32 v3, v0, v8, 1 // (op 1049) 


s_mul_i32 s1, s30, s35 // (op 1051) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 1051) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 1051) least significant
s_mul_hi_u32 s3, s30, s34 // (op 1051) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1051) most significant: sum
s_add_u32 s1, s1, s3 // (op 1051) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1051) 

s_mov_b32 s4, s32 // (op 1051) reg expression
s_mov_b32 s5, s33 // (op 1051) reg expression
s_mov_b32 s6, s2 // (op 1051) reg expression
s_mov_b32 s7, 131072 // (op 1051) call()


v_pack_b32_f16 v12, v4, v5 // (op 58) 
v_pack_b32_f16 v13, v6, v7 // (op 58) 


buffer_store_dwordx2 v[12:13], v3, s[4:7], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157:
v_lshrrev_b32 v0, 6, v1 // (op 819) 
v_and_b32 v3, 63, v1 // (op 819) 
v_lshrrev_b32 v4, 4, v3 // (op 819) 
v_lshlrev_b32 v5, 2, v4 // (op 819) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 819) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 819) 

v_lshrrev_b32 v4, 2, v3 // (op 819) 

v_and_b32 v3, 3, v4 // (op 819) 

v_and_b32 v4, 3, v1 // (op 819) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 819) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 819) 

v_mul_lo_u32 v4, v3, s36 // (op 819) 

v_add_lshl_u32 v3, v0, v4, 1 // (op 819) 


s_mul_i32 s1, s36, s35 // (op 821) most significant: low of low * high
s_mul_i32 s2, s37, s34 // (op 821) most significant: low of high * low
s_mul_i32 s0, s36, s34 // (op 821) least significant
s_mul_hi_u32 s3, s36, s34 // (op 821) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 821) most significant: sum
s_add_u32 s1, s1, s3 // (op 821) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 821) 

s_mov_b32 s8, s38 // (op 821) reg expression
s_mov_b32 s9, s39 // (op 821) reg expression
s_mov_b32 s10, s2 // (op 821) reg expression
s_mov_b32 s11, 131072 // (op 821) call()





buffer_load_dwordx2 v[4:5], v3, s[8:11], 0 offen // (op 170) Load value



v_accvgpr_read v12, a0 // (op 166) 
v_accvgpr_read v13, a1 // (op 166) 
v_accvgpr_read v14, a2 // (op 166) 
v_accvgpr_read v15, a3 // (op 166) 
v_mul_f32 v16, v10, v12 // (op 166) 
v_mul_f32 v17, v10, v13 // (op 166) 
v_mul_f32 v18, v10, v14 // (op 166) 
v_mul_f32 v19, v10, v15 // (op 166) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 171) 
v_lshrrev_b32 v7, 16, v4 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v12, v9, v6 // (op 171) 
v_mul_f32 v13, v9, v7 // (op 171) 
v_cvt_f32_f16 v6, v5 // (op 171) 
v_lshrrev_b32 v7, 16, v5 // (op 171) 
v_cvt_f32_f16 v7, v7 // (op 171) 
v_mul_f32 v14, v9, v6 // (op 171) 
v_mul_f32 v15, v9, v7 // (op 171) 


v_add_f32 v4, v12, v16 // (op 167) 
v_add_f32 v5, v13, v17 // (op 167) 
v_add_f32 v6, v14, v18 // (op 167) 
v_add_f32 v7, v15, v19 // (op 167) 


v_cvt_f16_f32 v12, v4 // (op 168) 
v_cvt_f16_f32 v13, v5 // (op 168) 
v_cvt_f16_f32 v14, v6 // (op 168) 
v_cvt_f16_f32 v15, v7 // (op 168) 

v_lshrrev_b32 v0, 6, v1 // (op 1066) 
v_and_b32 v11, 63, v1 // (op 1066) 
v_lshrrev_b32 v3, 4, v11 // (op 1066) 
v_lshlrev_b32 v4, 2, v3 // (op 1066) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1066) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 1066) 

v_lshrrev_b32 v3, 2, v11 // (op 1066) 

v_and_b32 v11, 3, v3 // (op 1066) 

v_and_b32 v3, 3, v1 // (op 1066) 
v_lshl_add_u32 v4, v11, 2, v3 // (op 1066) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 1066) 

v_mul_lo_u32 v11, v3, s30 // (op 1066) 

v_add_lshl_u32 v3, v0, v11, 1 // (op 1066) 


s_mul_i32 s1, s30, s35 // (op 1068) most significant: low of low * high
s_mul_i32 s2, s31, s34 // (op 1068) most significant: low of high * low
s_mul_i32 s0, s30, s34 // (op 1068) least significant
s_mul_hi_u32 s3, s30, s34 // (op 1068) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1068) most significant: sum
s_add_u32 s1, s1, s3 // (op 1068) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 1 // (op 1068) 

s_mov_b32 s4, s32 // (op 1068) reg expression
s_mov_b32 s5, s33 // (op 1068) reg expression
s_mov_b32 s6, s2 // (op 1068) reg expression
s_mov_b32 s7, 131072 // (op 1068) call()


v_pack_b32_f16 v4, v12, v13 // (op 169) 
v_pack_b32_f16 v5, v14, v15 // (op 169) 


buffer_store_dwordx2 v[4:5], v3, s[4:7], 0 offen // (op 169) Store value







s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157:



s_endpgm  // End of RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 14848 // lds bytes
  .amdhsa_accum_offset 96
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
    DataTypeB: B8
    DestDataType: h
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
  - .name: RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_TT_BF6_BF8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 136
    .group_segment_fixed_size: 14848
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 58
    .vgpr_count: 96
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

