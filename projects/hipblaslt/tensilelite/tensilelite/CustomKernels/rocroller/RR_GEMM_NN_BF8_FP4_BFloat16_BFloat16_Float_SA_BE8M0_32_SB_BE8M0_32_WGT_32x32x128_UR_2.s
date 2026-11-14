// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.p2align 8
.type RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2,@function
RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 712) 
v_and_b32 v3, 63, v1 // (op 712) 
v_sub_u32 v4, v3, v0 // (op 712) 

v_lshrrev_b32 v3, 1, v4 // (op 712) 
v_add_lshl_u32 v5, v0, v3, 5 // (op 712) 


v_and_b32 v0, 1, v4 // (op 712) 

v_lshlrev_b32 v3, 3, v0 // (op 712) 

v_lshrrev_b32 v0, 7, v1 // (op 712) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 712) 


v_add_u32 v0, v5, v4 // (op 712) 


v_lshrrev_b32 v3, 6, v1 // (op 765) 
v_and_b32 v4, 1, v3 // (op 765) 

v_and_b32 v3, 15, v1 // (op 765) 
v_lshl_add_u32 v5, v4, 4, v3 // (op 765) 


v_and_b32 v3, 63, v1 // (op 765) 
v_lshrrev_b32 v4, 4, v3 // (op 765) 

v_lshlrev_b32 v3, 5, v4 // (op 765) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 765) 


v_lshlrev_b32 v3, 2, v4 // (op 765) 

v_lshrrev_b32 v4, 3, v3 // (op 765) 

v_and_b32 v3, 48, v1 // (op 823) 
v_and_b32 v5, 63, v1 // (op 823) 
v_sub_u32 v6, v5, v3 // (op 823) 

v_lshrrev_b32 v5, 1, v6 // (op 823) 
v_add_lshl_u32 v7, v3, v5, 5 // (op 823) 


v_and_b32 v3, 1, v6 // (op 823) 

v_lshlrev_b32 v5, 3, v3 // (op 823) 

v_lshrrev_b32 v3, 7, v1 // (op 823) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 823) 


v_add_u32 v3, v7, v6 // (op 823) 


v_lshrrev_b32 v5, 6, v1 // (op 847) 
v_and_b32 v6, 1, v5 // (op 847) 

v_and_b32 v5, 15, v1 // (op 847) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 847) 


v_and_b32 v5, 63, v1 // (op 847) 
v_lshrrev_b32 v6, 4, v5 // (op 847) 

v_lshlrev_b32 v5, 5, v6 // (op 847) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 847) 


v_lshlrev_b32 v5, 2, v6 // (op 847) 

v_lshrrev_b32 v6, 3, v5 // (op 847) 

v_mov_b32 v5, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_and_b32 v8, -2, v1 // (op 422) 
v_sub_u32 v9, v1, v8 // (op 422) 

v_lshlrev_b32 v8, 4, v9 // (op 422) 

v_lshl_add_u32 v9, s16, 5, v8 // (op 422) 

v_lshrrev_b32 v8, 1, v1 // (op 422) 
v_mul_lo_u32 v10, v8, s2 // (op 422) 

v_add_u32 v8, v9, v10 // (op 422) 


s_mul_i32 s1, s2, s7 // (op 424) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 424) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 424) least significant
s_mul_hi_u32 s15, s2, s6 // (op 424) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 424) most significant: sum
s_add_u32 s1, s1, s15 // (op 424) most significant: sum


s_mov_b32 s40, s4 // (op 424) reg expression
s_mov_b32 s41, s5 // (op 424) reg expression
s_mov_b32 s42, s0 // (op 424) reg expression
s_mov_b32 s43, 131072 // (op 424) call()

v_and_b32 v9, -2, v1 // (op 441) 
v_sub_u32 v10, v1, v9 // (op 441) 

v_lshlrev_b32 v9, 4, v10 // (op 441) 

v_lshl_add_u32 v10, s16, 5, v9 // (op 441) 

v_lshrrev_b32 v9, 1, v1 // (op 441) 
v_mul_lo_u32 v11, v9, s2 // (op 441) 

v_add_u32 v9, v10, v11 // (op 441) 


s_mul_i32 s1, s2, s7 // (op 443) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 443) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 443) least significant
s_mul_hi_u32 s15, s2, s6 // (op 443) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 443) most significant: sum
s_add_u32 s1, s1, s15 // (op 443) most significant: sum


s_mov_b32 s40, s4 // (op 443) reg expression
s_mov_b32 s41, s5 // (op 443) reg expression
s_mov_b32 s42, s0 // (op 443) reg expression
s_mov_b32 s43, 131072 // (op 443) call()

v_and_b32 v10, -8, v1 // (op 486) 
v_sub_u32 v11, v1, v10 // (op 486) 

v_lshlrev_b32 v10, 4, v11 // (op 486) 

v_mov_b32 v13, 0 // (op 486) convert
v_mov_b32 v12, v10 // (op 486) convert

v_lshrrev_b32 v10, 3, v1 // (op 486) 
v_lshl_add_u32 v11, s17, 5, v10 // (op 486) 

v_mul_lo_u32 v15, v11, s9 // (op 486) most significant: low of low * high
v_mul_hi_u32 v10, v11, s8 // (op 486) most significant: high of low * low
v_mul_lo_u32 v14, v11, s8 // (op 486) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 486) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v14 // (op 486) least significant half
v_addc_co_u32 v11, s[0:1], v13, v15, s[0:1] // (op 486) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 486) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 486) 

v_mov_b32 v10, v12 // (op 486) convert

s_mul_i32 s1, s8, s13 // (op 488) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 488) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 488) least significant
s_mul_hi_u32 s15, s8, s12 // (op 488) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 488) most significant: sum
s_add_u32 s1, s1, s15 // (op 488) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 488) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 488) 

s_mov_b32 s44, s10 // (op 488) reg expression
s_mov_b32 s45, s11 // (op 488) reg expression
s_mov_b32 s46, s0 // (op 488) reg expression
s_mov_b32 s47, 131072 // (op 488) call()

v_and_b32 v11, -8, v1 // (op 505) 
v_sub_u32 v12, v1, v11 // (op 505) 

v_lshlrev_b32 v11, 4, v12 // (op 505) 

v_mov_b32 v13, 0 // (op 505) convert
v_mov_b32 v12, v11 // (op 505) convert

v_lshrrev_b32 v11, 3, v1 // (op 505) 
v_lshl_add_u32 v14, s17, 5, v11 // (op 505) 

v_mul_lo_u32 v17, v14, s9 // (op 505) most significant: low of low * high
v_mul_hi_u32 v11, v14, s8 // (op 505) most significant: high of low * low
v_mul_lo_u32 v16, v14, s8 // (op 505) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 505) most significant: sum


v_add_co_u32 v14, s[0:1], v12, v16 // (op 505) least significant half
v_addc_co_u32 v15, s[0:1], v13, v17, s[0:1] // (op 505) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 505) 


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 505) 

v_mov_b32 v11, v12 // (op 505) convert

s_mul_i32 s1, s8, s13 // (op 507) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 507) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 507) least significant
s_mul_hi_u32 s15, s8, s12 // (op 507) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 507) most significant: sum
s_add_u32 s1, s1, s15 // (op 507) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 507) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 507) 

s_mov_b32 s44, s10 // (op 507) reg expression
s_mov_b32 s45, s11 // (op 507) reg expression
s_mov_b32 s46, s0 // (op 507) reg expression
s_mov_b32 s47, 131072 // (op 507) call()

v_lshrrev_b32 v12, 7, v1 // (op 550) 
v_and_b32 v13, 15, v1 // (op 550) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 550) 


v_lshl_add_u32 v12, s16, 5, v14 // (op 550) 

v_and_b32 v13, 63, v1 // (op 550) 
v_lshrrev_b32 v14, 4, v13 // (op 550) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v13, v14, s20 // (op 550) 

v_add_u32 v14, v12, v13 // (op 550) 


s_mul_i32 s1, s20, s25 // (op 552) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 552) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 552) least significant
s_mul_hi_u32 s15, s20, s24 // (op 552) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 552) most significant: sum
s_add_u32 s1, s1, s15 // (op 552) most significant: sum


s_mov_b32 s48, s22 // (op 552) reg expression
s_mov_b32 s49, s23 // (op 552) reg expression
s_mov_b32 s50, s0 // (op 552) reg expression
s_mov_b32 s51, 131072 // (op 552) call()

v_lshrrev_b32 v12, 7, v1 // (op 569) 
v_and_b32 v13, 15, v1 // (op 569) 
v_lshl_add_u32 v15, v12, 4, v13 // (op 569) 


v_lshl_add_u32 v12, s16, 5, v15 // (op 569) 

v_and_b32 v13, 63, v1 // (op 569) 
v_lshrrev_b32 v15, 4, v13 // (op 569) 

v_mul_lo_u32 v13, v15, s20 // (op 569) 

v_add_u32 v15, v12, v13 // (op 569) 


s_mul_i32 s1, s20, s25 // (op 571) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 571) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 571) least significant
s_mul_hi_u32 s15, s20, s24 // (op 571) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 571) most significant: sum
s_add_u32 s1, s1, s15 // (op 571) most significant: sum


s_mov_b32 s48, s22 // (op 571) reg expression
s_mov_b32 s49, s23 // (op 571) reg expression
s_mov_b32 s50, s0 // (op 571) reg expression
s_mov_b32 s51, 131072 // (op 571) call()

v_lshrrev_b32 v12, 6, v1 // (op 614) 
v_and_b32 v13, 1, v12 // (op 614) 

v_and_b32 v12, 15, v1 // (op 614) 
v_lshl_add_u32 v16, v13, 4, v12 // (op 614) 


v_lshl_add_u32 v12, s17, 5, v16 // (op 614) 

v_mul_lo_u32 v13, v12, s26 // (op 614) 

v_and_b32 v12, 63, v1 // (op 614) 
v_lshrrev_b32 v16, 4, v12 // (op 614) 

v_add_u32 v12, v16, v13 // (op 614) 


s_mul_i32 s1, s26, s13 // (op 616) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 616) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 616) least significant
s_mul_hi_u32 s15, s26, s12 // (op 616) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 616) most significant: sum
s_add_u32 s1, s1, s15 // (op 616) most significant: sum


s_mov_b32 s52, s28 // (op 616) reg expression
s_mov_b32 s53, s29 // (op 616) reg expression
s_mov_b32 s54, s0 // (op 616) reg expression
s_mov_b32 s55, 131072 // (op 616) call()

v_lshrrev_b32 v13, 6, v1 // (op 633) 
v_and_b32 v16, 1, v13 // (op 633) 

v_and_b32 v13, 15, v1 // (op 633) 
v_lshl_add_u32 v17, v16, 4, v13 // (op 633) 


v_lshl_add_u32 v13, s17, 5, v17 // (op 633) 

v_mul_lo_u32 v16, v13, s26 // (op 633) 

v_and_b32 v13, 63, v1 // (op 633) 
v_lshrrev_b32 v17, 4, v13 // (op 633) 

v_add_u32 v13, v17, v16 // (op 633) 


s_mul_i32 s1, s26, s13 // (op 635) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 635) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 635) least significant
s_mul_hi_u32 s15, s26, s12 // (op 635) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 635) most significant: sum
s_add_u32 s1, s1, s15 // (op 635) most significant: sum


s_mov_b32 s52, s28 // (op 635) reg expression
s_mov_b32 s53, s29 // (op 635) reg expression
s_mov_b32 s54, s0 // (op 635) reg expression
s_mov_b32 s55, 131072 // (op 635) call()


v_mov_b64 v[16:17], s[20:21] // (op 312) convert

v_mov_b32 v18, v14 // (op 312) reg expression
v_mov_b32 v19, v18 // (op 312) 
buffer_load_ubyte v20, v19, s[48:51], 0 offen offset: 0 // (op 312) Load value






buffer_load_ubyte v16, v12, s[52:55], 0 offen offset: 0 // (op 317) Load value



buffer_load_dwordx2 v[18:19], v10, s[44:47], 0 offen // (op 321) Load value


v_mov_b32 v17, v8 // (op 324) reg expression

buffer_load_dwordx4 v[22:25], v17, s[40:43], 0 offen // (op 324) Load value


v_mov_b64 v[26:27], s[20:21] // (op 328) convert

s_lshl_b32 s0, s20, 2 // (op 328) 
v_add_u32 v17, v14, s0 // (op 328) 

v_mov_b32 v21, v17 // (op 328) 
buffer_load_ubyte v28, v21, s[48:51], 0 offen offset: 0 // (op 328) Load value







buffer_load_ubyte v14, v12, s[52:55], 0 offen offset:4 // (op 333) Load value




buffer_load_dwordx2 v[26:27], v10, s[44:47], 0 offen offset:64 // (op 337) Load value



s_lshl_b32 s0, s2, 7 // (op 340) 
v_add_u32 v10, v8, s0 // (op 340) 


buffer_load_dwordx4 v[30:33], v10, s[40:43], 0 offen // (op 340) Load value


v_and_b32 v8, -8, v1 // (op 753) 
v_sub_u32 v10, v1, v8 // (op 753) 

v_lshlrev_b32 v8, 4, v10 // (op 753) 

v_lshrrev_b32 v10, 3, v1 // (op 753) 
v_lshl_add_u32 v12, v10, 7, v8 // (op 753) 


v_lshlrev_b32 v8, 2, v12 // (op 753) 

v_lshrrev_b32 v10, 3, v8 // (op 753) 




s_waitcnt vmcnt(5)
ds_write_b64 v10, v[18:19] // (op 342) Store local data 

v_and_b32 v8, -2, v1 // (op 700) 
v_sub_u32 v10, v1, v8 // (op 700) 

v_lshlrev_b32 v8, 4, v10 // (op 700) 

v_lshrrev_b32 v10, 1, v1 // (op 700) 
v_lshl_add_u32 v12, v10, 5, v8 // (op 700) 





s_waitcnt vmcnt(4)
ds_write_b128 v12, v[22:25] offset:2048 // (op 343) Store local data 

v_and_b32 v8, -2, v1 // (op 671) 
v_sub_u32 v10, v1, v8 // (op 671) 

v_lshlrev_b32 v8, 4, v10 // (op 671) 

v_lshrrev_b32 v10, 1, v1 // (op 671) 
v_lshl_add_u32 v12, v10, 5, v8 // (op 671) 


v_and_b32 v8, -8, v1 // (op 724) 
v_sub_u32 v10, v1, v8 // (op 724) 

v_lshlrev_b32 v8, 4, v10 // (op 724) 

v_lshrrev_b32 v10, 3, v1 // (op 724) 
v_lshl_add_u32 v17, v10, 7, v8 // (op 724) 


v_lshlrev_b32 v8, 2, v17 // (op 724) 

v_lshrrev_b32 v10, 3, v8 // (op 724) 

v_and_b32 v8, -2, v1 // (op 811) 
v_sub_u32 v17, v1, v8 // (op 811) 

v_lshlrev_b32 v8, 4, v17 // (op 811) 

v_lshrrev_b32 v17, 1, v1 // (op 811) 
v_lshl_add_u32 v21, v17, 5, v8 // (op 811) 


v_and_b32 v8, -8, v1 // (op 835) 
v_sub_u32 v17, v1, v8 // (op 835) 

v_lshlrev_b32 v8, 4, v17 // (op 835) 

v_lshrrev_b32 v17, 3, v1 // (op 835) 
v_lshl_add_u32 v29, v17, 7, v8 // (op 835) 


v_lshlrev_b32 v8, 2, v29 // (op 835) 

v_lshrrev_b32 v17, 3, v8 // (op 835) 

s_mov_b32 s0, 0 // (op 53) call()
s_ashr_i64 s[14:15], s[6:7], 7 // (op 52) 
s_mov_b32 s1, s14 // (op 52) convert

s_and_b32 s14, s1, 4294967294 // (op 52) 

s_cmp_lt_u32 s0, s14 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 // (op 52) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52 if false)
RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52:
s_add_u32 s0, s0, 2 // (op 54) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 876) 



ds_read_b64_tr_b8 v[34:35], v0 offset:2048 // (op 62) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v0 offset:2304 // (op 62) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v0 offset:4096 // (op 62) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v0 offset:4352 // (op 62) Transpose load local data 
s_barrier  // (op 889) 



ds_read_b128 v[42:45], v4 // (op 67) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:45], a[0:3], v20, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 78) 



v_mov_b64 v[34:35], s[20:21] // (op 72) convert

v_mov_b32 v8, v15 // (op 72) 
s_lshl_b32 s1, s20, 3 // (op 72) 
v_add_u32 v29, v8, s1 // (op 72) 

v_mov_b32 v36, v29 // (op 72) 
buffer_load_ubyte v20, v36, s[48:51], 0 offen offset: 0 // (op 72) Load value





v_mov_b32 v29, v13 // (op 76) 

buffer_load_ubyte v16, v29, s[52:55], 0 offen offset:8 // (op 76) Load value


v_mov_b32 v34, v11 // (op 10) 

buffer_load_dwordx2 v[18:19], v34, s[44:47], 0 offen offset:128 // (op 10) Load value


v_mov_b32 v35, v9 // (op 4) 
s_lshl_b32 s1, s2, 8 // (op 4) 
v_add_u32 v36, v35, s1 // (op 4) 


buffer_load_dwordx4 v[22:25], v36, s[40:43], 0 offen // (op 4) Load value

s_barrier  // (op 912) 



s_waitcnt vmcnt(5)
ds_write_b64 v17, v[26:27] offset:6144 // (op 222) Store local data 
s_barrier  // (op 906) 



s_waitcnt vmcnt(4)
ds_write_b128 v21, v[30:33] offset:8192 // (op 212) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 903) 



ds_read_b64_tr_b8 v[36:37], v3 offset:8192 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v3 offset:8448 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v3 offset:10240 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v3 offset:10496 // (op 214) Transpose load local data 
s_barrier  // (op 909) 
s_barrier  // (op 963) 



ds_read_b128 v[44:47], v6 offset:6144 // (op 216) Load local data 
s_barrier  // (op 969) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[36:43], v[44:47], a[0:3], v28, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 221) 



v_mov_b64 v[36:37], s[20:21] // (op 218) convert

s_lshl_b32 s1, s20, 2 // (op 218) 
s_mul_i32 s14, 3, s1 // (op 218) 

v_add_u32 v38, v8, s14 // (op 218) 

v_mov_b32 v39, v38 // (op 218) 
buffer_load_ubyte v28, v39, s[48:51], 0 offen offset: 0 // (op 218) Load value







s_lshl_b32 s1, s20, 3 // (op 561) 
v_add_u32 v15, v15, s1 // (op 561) 

buffer_load_ubyte v14, v29, s[52:55], 0 offen offset:12 // (op 220) Load value




v_add_u32 v13, 8, v13 // (op 625) 
buffer_load_dwordx2 v[26:27], v34, s[44:47], 0 offen offset:192 // (op 223) Load value



v_add_u32 v11, 128, v11 // (op 497) 
s_lshl_b32 s1, s2, 7 // (op 211) 
s_mul_i32 s14, 3, s1 // (op 211) 

v_add_u32 v8, v35, s14 // (op 211) 


buffer_load_dwordx4 v[30:33], v8, s[40:43], 0 offen // (op 211) Load value


s_barrier  // (op 892) 
s_lshl_b32 s1, s2, 8 // (op 433) 



v_add_u32 v9, v9, s1 // (op 433) 

s_waitcnt vmcnt(5)
ds_write_b64 v10, v[18:19] // (op 13) Store local data 
s_barrier  // (op 879) 



s_waitcnt vmcnt(4)
ds_write_b128 v12, v[22:25] offset:2048 // (op 6) Store local data 
s_ashr_i64 s[14:15], s[6:7], 7 // (op 52) 
s_mov_b32 s1, s14 // (op 52) convert

s_and_b32 s14, s1, 4294967294 // (op 52) 

s_cmp_lt_u32 s0, s14 // (op 52) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 // (op 52) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoop_52 if true)
RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoop_52:













v_and_b32 v29, -2, v1 // (op 396) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 396) 
v_sub_u32 v8, v1, v29 // (op 396) 

s_lshr_b32 s14, s0, 1 // (op 396) 

v_lshlrev_b32 v29, 4, v8 // (op 396) 

v_lshl_add_u32 v8, s16, 5, v29 // (op 396) 

v_lshrrev_b32 v29, 1, v1 // (op 396) 
v_lshl_add_u32 v9, s14, 8, v29 // (op 396) 


v_mul_lo_u32 v29, v9, s2 // (op 396) 

v_add_u32 v9, v8, v29 // (op 396) 


s_mul_i32 s1, s2, s7 // (op 398) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 398) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 398) least significant
s_mul_hi_u32 s15, s2, s6 // (op 398) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 398) most significant: sum
s_add_u32 s1, s1, s15 // (op 398) most significant: sum


s_mov_b32 s40, s4 // (op 398) reg expression
s_mov_b32 s41, s5 // (op 398) reg expression
s_mov_b32 s42, s0 // (op 398) reg expression
s_mov_b32 s43, 131072 // (op 398) call()

v_and_b32 v8, -8, v1 // (op 463) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 463) 
v_sub_u32 v29, v1, v8 // (op 463) 

s_lshr_b32 s14, s0, 1 // (op 463) 

v_lshlrev_b32 v8, 4, v29 // (op 463) 

v_lshl_add_u32 v29, s14, 8, v8 // (op 463) 


v_mov_b32 v11, 0 // (op 463) convert
v_mov_b32 v10, v29 // (op 463) convert

v_lshrrev_b32 v8, 3, v1 // (op 463) 
v_lshl_add_u32 v29, s17, 5, v8 // (op 463) 

v_mul_lo_u32 v13, v29, s9 // (op 463) most significant: low of low * high
v_mul_hi_u32 v8, v29, s8 // (op 463) most significant: high of low * low
v_mul_lo_u32 v12, v29, s8 // (op 463) least significant: low of low * low
v_add_u32 v13, v13, v8 // (op 463) most significant: sum


v_add_co_u32 v24, s[0:1], v10, v12 // (op 463) least significant half
v_addc_co_u32 v25, s[0:1], v11, v13, s[0:1] // (op 463) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 463) 


v_lshrrev_b64 v[10:11], 3, v[24:25] // (op 463) 

v_mov_b32 v8, v10 // (op 463) convert

s_mul_i32 s1, s8, s13 // (op 465) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 465) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 465) least significant
s_mul_hi_u32 s15, s8, s12 // (op 465) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 465) most significant: sum
s_add_u32 s1, s1, s15 // (op 465) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 465) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 465) 

s_mov_b32 s44, s10 // (op 465) reg expression
s_mov_b32 s45, s11 // (op 465) reg expression
s_mov_b32 s46, s0 // (op 465) reg expression
s_mov_b32 s47, 131072 // (op 465) call()

v_lshrrev_b32 v29, 7, v1 // (op 527) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 527) 
v_and_b32 v10, 15, v1 // (op 527) 
s_lshr_b32 s14, s0, 1 // (op 527) 

v_lshl_add_u32 v11, v29, 4, v10 // (op 527) 


v_lshl_add_u32 v10, s16, 5, v11 // (op 527) 

v_and_b32 v29, 63, v1 // (op 527) 
v_lshrrev_b32 v11, 4, v29 // (op 527) 

v_lshl_add_u32 v29, s14, 3, v11 // (op 527) 


v_mul_lo_u32 v11, v29, s20 // (op 527) 

v_add_u32 v29, v10, v11 // (op 527) 


s_mul_i32 s1, s20, s25 // (op 529) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 529) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 529) least significant
s_mul_hi_u32 s15, s20, s24 // (op 529) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 529) most significant: sum
s_add_u32 s1, s1, s15 // (op 529) most significant: sum


s_mov_b32 s48, s22 // (op 529) reg expression
s_mov_b32 s49, s23 // (op 529) reg expression
s_mov_b32 s50, s0 // (op 529) reg expression
s_mov_b32 s51, 131072 // (op 529) call()

v_and_b32 v10, 63, v1 // (op 591) 
s_ashr_i64 s[0:1], s[6:7], 7 // (op 591) 
v_lshrrev_b32 v11, 4, v10 // (op 591) 

s_lshr_b32 s14, s0, 1 // (op 591) 

v_lshrrev_b32 v10, 6, v1 // (op 591) 
v_lshl_add_u32 v12, s14, 3, v11 // (op 591) 


v_and_b32 v11, 1, v10 // (op 591) 

v_and_b32 v10, 15, v1 // (op 591) 
v_lshl_add_u32 v13, v11, 4, v10 // (op 591) 


v_lshl_add_u32 v10, s17, 5, v13 // (op 591) 

v_mul_lo_u32 v11, v10, s26 // (op 591) 

v_add_u32 v10, v12, v11 // (op 591) 


s_mul_i32 s1, s26, s13 // (op 593) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 593) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 593) least significant
s_mul_hi_u32 s15, s26, s12 // (op 593) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 593) most significant: sum
s_add_u32 s1, s1, s15 // (op 593) most significant: sum


s_mov_b32 s52, s28 // (op 593) reg expression
s_mov_b32 s53, s29 // (op 593) reg expression
s_mov_b32 s54, s0 // (op 593) reg expression
s_mov_b32 s55, 131072 // (op 593) call()

v_and_b32 v11, -2, v1 // (op 684) 
v_sub_u32 v12, v1, v11 // (op 684) 

v_lshlrev_b32 v11, 4, v12 // (op 684) 

v_lshrrev_b32 v12, 1, v1 // (op 684) 
v_lshl_add_u32 v13, v12, 5, v11 // (op 684) 


v_and_b32 v11, -8, v1 // (op 737) 
v_sub_u32 v12, v1, v11 // (op 737) 

v_lshlrev_b32 v11, 4, v12 // (op 737) 

v_lshrrev_b32 v12, 3, v1 // (op 737) 
v_lshl_add_u32 v15, v12, 7, v11 // (op 737) 


v_lshlrev_b32 v11, 2, v15 // (op 737) 

v_lshrrev_b32 v12, 3, v11 // (op 737) 

s_ashr_i64 s[0:1], s[6:7], 7 // (op 157) 
s_and_b32 s14, s0, 4294967294 // (op 157) 

s_ashr_i64 s[0:1], s[6:7], 7 // (op 156) 
s_mov_b32 s15, s0 // (op 156) convert

s_cmp_lt_u32 s14, s15 // (op 156) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156:


s_add_u32 s14, s14, 1 // (op 158) 
v_mov_b32 v11, v9 // (op 161) 
v_mov_b32 v14, v11 // (op 161) reg expression

buffer_load_dwordx4 v[16:19], v14, s[40:43], 0 offen // (op 161) Load value


s_lshl_b32 s15, s2, 7 // (op 386) 
v_add_u32 v9, v9, s15 // (op 386) 



s_barrier  // (op 882) 
v_mov_b32 v11, v8 // (op 173) 




buffer_load_dwordx2 v[14:15], v11, s[44:47], 0 offen // (op 173) Load value

v_add_u32 v8, 64, v8 // (op 455) 
s_waitcnt vmcnt(1)
ds_write_b128 v13, v[16:19] offset:2048 // (op 162) Store local data 

s_barrier  // (op 895) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 885) 






ds_read_b64_tr_b8 v[16:17], v0 offset:2048 // (op 164) Transpose load local data 
ds_read_b64_tr_b8 v[18:19], v0 offset:2304 // (op 164) Transpose load local data 
ds_read_b64_tr_b8 v[20:21], v0 offset:4096 // (op 164) Transpose load local data 
ds_read_b64_tr_b8 v[22:23], v0 offset:4352 // (op 164) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b64 v12, v[14:15] // (op 172) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 899) 



ds_read_b128 v[24:27], v4 // (op 166) Load local data 

v_mov_b64 v[14:15], s[20:21] // (op 168) convert

v_mov_b32 v11, v29 // (op 168) 
v_mov_b32 v28, v11 // (op 168) reg expression
v_mov_b32 v30, v28 // (op 168) 
buffer_load_ubyte v31, v30, s[48:51], 0 offen offset: 0 // (op 168) Load value




s_lshl_b32 s15, s20, 2 // (op 519) 
v_add_u32 v29, v29, s15 // (op 519) 



v_mov_b32 v11, v10 // (op 170) 

buffer_load_ubyte v28, v11, s[52:55], 0 offen offset: 0 // (op 170) Load value

v_add_u32 v10, 4, v10 // (op 583) 
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:23], v[24:27], a[0:3], v31, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 171) 




s_ashr_i64 s[0:1], s[6:7], 7 // (op 156) 
s_mov_b32 s15, s0 // (op 156) convert

s_cmp_lt_u32 s14, s15 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopTop_KLoopTail_156 if true)
RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ForLoopBottom_KLoopTail_156:





















s_barrier  // (op 957) 




v_cmp_eq_f32 vcc, 0.00000, v5 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116
s_nop 0
v_accvgpr_read v8, a0 // (op 42) Wait state hazard: XDL Write Hazard
v_accvgpr_read v9, a1 // (op 42) 
v_accvgpr_read v10, a2 // (op 42) 
v_accvgpr_read v11, a3 // (op 42) 
v_mul_f32 v12, v7, v8 // (op 42) 
v_mul_f32 v13, v7, v9 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 

v_lshrrev_b32 v8, 16, v12 // (op 48) 
v_lshrrev_b32 v9, 16, v13 // (op 48) 
v_lshrrev_b32 v10, 16, v14 // (op 48) 
v_lshrrev_b32 v11, 16, v15 // (op 48) 

v_lshrrev_b32 v0, 7, v1 // (op 779) 
v_and_b32 v6, 63, v1 // (op 779) 
v_lshrrev_b32 v3, 4, v6 // (op 779) 
v_lshlrev_b32 v4, 2, v3 // (op 779) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 779) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 779) 

v_lshrrev_b32 v3, 2, v6 // (op 779) 

v_and_b32 v4, 3, v3 // (op 779) 

v_and_b32 v3, 3, v1 // (op 779) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 779) 


v_lshrrev_b32 v3, 6, v1 // (op 779) 
v_and_b32 v4, 1, v3 // (op 779) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 779) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 779) 

v_mul_lo_u32 v3, v4, s30 // (op 779) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 779) 


s_mul_i32 s15, s30, s13 // (op 781) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 781) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 781) least significant
s_mul_hi_u32 s1, s30, s12 // (op 781) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 781) most significant: sum
s_add_u32 s15, s15, s1 // (op 781) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 781) 

s_mov_b32 s8, s32 // (op 781) reg expression
s_mov_b32 s9, s33 // (op 781) reg expression
s_mov_b32 s10, s0 // (op 781) reg expression
s_mov_b32 s11, 131072 // (op 781) call()


v_pack_b32_f16 v12, v8, v9 // (op 50) 
v_pack_b32_f16 v13, v10, v11 // (op 50) 


buffer_store_dwordx2 v[12:13], v4, s[8:11], 0 offen // (op 50) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalFalse_0_beta_116:
v_lshrrev_b32 v0, 7, v1 // (op 654) 
v_and_b32 v6, 63, v1 // (op 654) 
v_lshrrev_b32 v3, 4, v6 // (op 654) 
v_lshlrev_b32 v4, 2, v3 // (op 654) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 654) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 654) 

v_lshrrev_b32 v3, 2, v6 // (op 654) 

v_and_b32 v4, 3, v3 // (op 654) 

v_and_b32 v3, 3, v1 // (op 654) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 654) 


v_lshrrev_b32 v3, 6, v1 // (op 654) 
v_and_b32 v4, 1, v3 // (op 654) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 654) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 654) 

v_mul_lo_u32 v3, v4, s34 // (op 654) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 654) 


s_mul_i32 s15, s34, s13 // (op 656) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 656) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 656) least significant
s_mul_hi_u32 s1, s34, s12 // (op 656) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 656) most significant: sum
s_add_u32 s15, s15, s1 // (op 656) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 656) 

s_mov_b32 s4, s36 // (op 656) reg expression
s_mov_b32 s5, s37 // (op 656) reg expression
s_mov_b32 s6, s0 // (op 656) reg expression
s_mov_b32 s7, 131072 // (op 656) call()





buffer_load_dwordx2 v[8:9], v4, s[4:7], 0 offen // (op 129) Load value



v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v10, 65535, v8 // (op 130) 
v_lshrrev_b32 v11, 16, v8 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v18, v5, v10 // (op 130) 
v_mul_f32 v19, v5, v11 // (op 130) 
v_and_b32 v10, 65535, v9 // (op 130) 
v_lshrrev_b32 v11, 16, v9 // (op 130) 
v_lshlrev_b32 v10, 16, v10 // (op 130) 
v_lshlrev_b32 v11, 16, v11 // (op 130) 
v_mul_f32 v20, v5, v10 // (op 130) 
v_mul_f32 v21, v5, v11 // (op 130) 


v_add_f32 v8, v18, v14 // (op 126) 
v_add_f32 v9, v19, v15 // (op 126) 
v_add_f32 v10, v20, v16 // (op 126) 
v_add_f32 v11, v21, v17 // (op 126) 


v_lshrrev_b32 v12, 16, v8 // (op 127) 
v_lshrrev_b32 v13, 16, v9 // (op 127) 
v_lshrrev_b32 v14, 16, v10 // (op 127) 
v_lshrrev_b32 v15, 16, v11 // (op 127) 

v_lshrrev_b32 v0, 7, v1 // (op 796) 
v_and_b32 v6, 63, v1 // (op 796) 
v_lshrrev_b32 v3, 4, v6 // (op 796) 
v_lshlrev_b32 v4, 2, v3 // (op 796) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 796) 


v_lshl_add_u32 v0, s16, 5, v3 // (op 796) 

v_lshrrev_b32 v3, 2, v6 // (op 796) 

v_and_b32 v4, 3, v3 // (op 796) 

v_and_b32 v3, 3, v1 // (op 796) 
v_lshl_add_u32 v6, v4, 2, v3 // (op 796) 


v_lshrrev_b32 v3, 6, v1 // (op 796) 
v_and_b32 v4, 1, v3 // (op 796) 

v_lshl_add_u32 v3, v4, 4, v6 // (op 796) 


v_lshl_add_u32 v4, s17, 5, v3 // (op 796) 

v_mul_lo_u32 v3, v4, s30 // (op 796) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 796) 


s_mul_i32 s15, s30, s13 // (op 798) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 798) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 798) least significant
s_mul_hi_u32 s1, s30, s12 // (op 798) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 798) most significant: sum
s_add_u32 s15, s15, s1 // (op 798) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 798) 

s_mov_b32 s8, s32 // (op 798) reg expression
s_mov_b32 s9, s33 // (op 798) reg expression
s_mov_b32 s10, s0 // (op 798) reg expression
s_mov_b32 s11, 131072 // (op 798) call()


v_pack_b32_f16 v8, v12, v13 // (op 128) 
v_pack_b32_f16 v9, v14, v15 // (op 128) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 128) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_ConditionalBottom_0_beta_116:




s_endpgm  // End of RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.LRR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end:
.size RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2, .LRR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2_end-RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: B8
    DataTypeA: B8
    DataTypeB: F4
    DestDataType: b
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
  - .name: RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2
    .symbol: RR_GEMM_NN_BF8_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12288
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

