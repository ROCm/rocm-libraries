// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 871) 
v_and_b32 v3, 63, v1 // (op 871) 
v_sub_u32 v4, v3, v0 // (op 871) 

v_lshrrev_b32 v3, 1, v4 // (op 871) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 871) 


v_and_b32 v0, 1, v4 // (op 871) 

v_lshlrev_b32 v3, 3, v0 // (op 871) 

v_lshrrev_b32 v0, 6, v1 // (op 871) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 871) 


v_add_u32 v0, v5, v4 // (op 871) 


v_and_b32 v3, 63, v1 // (op 924) 
v_and_b32 v4, 48, v1 // (op 924) 
v_sub_u32 v5, v3, v4 // (op 924) 

v_lshrrev_b32 v4, 4, v3 // (op 924) 

v_lshl_add_u32 v3, v4, 5, v5 // (op 924) 


v_lshlrev_b32 v4, 4, v3 // (op 924) 
v_mul_lo_u32 v5, v4, 6 // (op 924) 

v_lshrrev_b32 v4, 3, v5 // (op 924) 

v_and_b32 v5, 268435455, v3 // (op 924) 

v_lshlrev_b32 v3, 2, v5 // (op 924) 

v_add_u32 v5, v4, v3 // (op 924) 


v_lshrrev_b32 v3, 6, v1 // (op 977) 
v_and_b32 v4, 15, v1 // (op 977) 
v_lshl_add_u32 v6, v3, 4, v4 // (op 977) 


v_and_b32 v3, 48, v1 // (op 977) 
v_add_u32 v4, v3, v6 // (op 977) 


v_and_b32 v3, 48, v1 // (op 1030) 
v_and_b32 v6, 15, v1 // (op 1030) 
v_add_u32 v7, v3, v6 // (op 1030) 


v_and_b32 v3, 48, v1 // (op 1103) 
v_and_b32 v6, 63, v1 // (op 1103) 
v_sub_u32 v8, v6, v3 // (op 1103) 

v_lshrrev_b32 v6, 1, v8 // (op 1103) 
v_add_lshl_u32 v9, v3, v6, 4 // (op 1103) 


v_and_b32 v3, 1, v8 // (op 1103) 

v_lshlrev_b32 v6, 3, v3 // (op 1103) 

v_lshrrev_b32 v3, 6, v1 // (op 1103) 
v_lshl_add_u32 v8, v3, 4, v6 // (op 1103) 


v_add_u32 v3, v9, v8 // (op 1103) 


v_and_b32 v6, 63, v1 // (op 1127) 
v_and_b32 v8, 48, v1 // (op 1127) 
v_sub_u32 v9, v6, v8 // (op 1127) 

v_lshrrev_b32 v8, 4, v6 // (op 1127) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 1127) 


v_lshlrev_b32 v8, 4, v6 // (op 1127) 
v_mul_lo_u32 v9, v8, 6 // (op 1127) 

v_lshrrev_b32 v8, 3, v9 // (op 1127) 

v_and_b32 v9, 268435455, v6 // (op 1127) 

v_lshlrev_b32 v6, 2, v9 // (op 1127) 

v_add_u32 v9, v8, v6 // (op 1127) 


v_lshrrev_b32 v6, 6, v1 // (op 1151) 
v_and_b32 v8, 15, v1 // (op 1151) 
v_lshl_add_u32 v10, v6, 4, v8 // (op 1151) 


v_and_b32 v6, 48, v1 // (op 1151) 
v_add_u32 v8, v6, v10 // (op 1151) 


v_and_b32 v6, 48, v1 // (op 1175) 
v_and_b32 v10, 15, v1 // (op 1175) 
v_add_u32 v11, v6, v10 // (op 1175) 


v_mov_b32 v6, s14 // (op 2) Move value
v_mov_b32 v10, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
s_lshl_b32 s0, s16, 4 // (op 582) 
v_mul_lo_u32 v13, v1, s3 // (op 582) most significant: low of low * high
s_mov_b32 s15, 0 // (op 582) convert
s_mov_b32 s14, s0 // (op 582) convert

v_mul_hi_u32 v14, v1, s2 // (op 582) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 582) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 582) most significant: sum

v_mov_b32 v14, s14 // (op 582) 
v_mov_b32 v15, s15 // (op 582) 
v_add_co_u32 v16, s[0:1], v14, v12 // (op 582) least significant half
v_addc_co_u32 v17, s[0:1], v15, v13, s[0:1] // (op 582) most significant half





s_mul_i32 s1, s2, s7 // (op 584) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 584) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 584) least significant
s_mul_hi_u32 s15, s2, s6 // (op 584) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 584) most significant: sum
s_add_u32 s1, s1, s15 // (op 584) most significant: sum


s_mov_b32 s40, s4 // (op 584) reg expression
s_mov_b32 s41, s5 // (op 584) reg expression
s_mov_b32 s42, s0 // (op 584) reg expression
s_mov_b32 s43, 131072 // (op 584) call()

s_lshl_b32 s0, s16, 4 // (op 601) 
v_mul_lo_u32 v13, v1, s3 // (op 601) most significant: low of low * high
v_mul_hi_u32 v14, v1, s2 // (op 601) most significant: high of low * low
v_mul_lo_u32 v12, v1, s2 // (op 601) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 601) most significant: sum

s_mov_b32 s15, 0 // (op 601) convert
s_mov_b32 s14, s0 // (op 601) convert

v_mov_b32 v14, s14 // (op 601) 
v_mov_b32 v15, s15 // (op 601) 
v_add_co_u32 v18, s[0:1], v14, v12 // (op 601) least significant half
v_addc_co_u32 v19, s[0:1], v15, v13, s[0:1] // (op 601) most significant half





s_mul_i32 s1, s2, s7 // (op 603) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 603) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 603) least significant
s_mul_hi_u32 s15, s2, s6 // (op 603) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 603) most significant: sum
s_add_u32 s1, s1, s15 // (op 603) most significant: sum


s_mov_b32 s40, s4 // (op 603) reg expression
s_mov_b32 s41, s5 // (op 603) reg expression
s_mov_b32 s42, s0 // (op 603) reg expression
s_mov_b32 s43, 131072 // (op 603) call()

s_lshl_b32 s0, s17, 4 // (op 646) 
v_lshlrev_b32 v12, 2, v1 // (op 646) 
s_mov_b32 s15, 0 // (op 646) convert
s_mov_b32 s14, s0 // (op 646) convert

v_mul_lo_u32 v15, v12, s9 // (op 646) most significant: low of low * high
v_mul_hi_u32 v13, v12, s8 // (op 646) most significant: high of low * low
v_mul_lo_u32 v14, v12, s8 // (op 646) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 646) most significant: sum

v_mov_b32 v13, s14 // (op 646) 
v_mov_b32 v20, s15 // (op 646) 
v_add_co_u32 v14, s[0:1], v14, v13 // (op 646) least significant half
v_addc_co_u32 v15, s[0:1], v15, v20, s[0:1] // (op 646) most significant half





v_mul_lo_u32 v13, v15, 6 // (op 646) most significant: low of high * low
v_mul_hi_u32 v20, v14, 6 // (op 646) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 646) least significant: low of low * low
v_add_u32 v13, v13, v20 // (op 646) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[12:13] // (op 646) 

v_mov_b32 v12, v14 // (op 646) convert

s_mul_i32 s1, s7, 6 // (op 648) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 648) least significant
s_mul_hi_u32 s14, s6, 6 // (op 648) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 648) most significant: sum

s_mul_i32 s15, s8, s1 // (op 648) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 648) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 648) least significant
s_mul_hi_u32 s19, s8, s0 // (op 648) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 648) most significant: sum
s_add_u32 s15, s15, s19 // (op 648) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 648) 

s_mov_b32 s44, s10 // (op 648) reg expression
s_mov_b32 s45, s11 // (op 648) reg expression
s_mov_b32 s46, s0 // (op 648) reg expression
s_mov_b32 s47, 131072 // (op 648) call()

s_lshl_b32 s0, s17, 4 // (op 665) 
v_lshlrev_b32 v13, 2, v1 // (op 665) 
s_mov_b32 s15, 0 // (op 665) convert
s_mov_b32 s14, s0 // (op 665) convert

v_mul_lo_u32 v15, v13, s9 // (op 665) most significant: low of low * high
v_mul_hi_u32 v20, v13, s8 // (op 665) most significant: high of low * low
v_mul_lo_u32 v14, v13, s8 // (op 665) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 665) most significant: sum

v_mov_b32 v20, s14 // (op 665) 
v_mov_b32 v21, s15 // (op 665) 
v_add_co_u32 v14, s[0:1], v14, v20 // (op 665) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 665) most significant half





v_mul_lo_u32 v21, v15, 6 // (op 665) most significant: low of high * low
v_mul_hi_u32 v13, v14, 6 // (op 665) most significant: high of low * low
v_mul_lo_u32 v20, v14, 6 // (op 665) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 665) most significant: sum


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 665) 

v_mov_b32 v13, v14 // (op 665) convert

s_mul_i32 s1, s7, 6 // (op 667) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 667) least significant
s_mul_hi_u32 s14, s6, 6 // (op 667) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 667) most significant: sum

s_mul_i32 s15, s8, s1 // (op 667) most significant: low of low * high
s_mul_i32 s18, s9, s0 // (op 667) most significant: low of high * low
s_mul_i32 s14, s8, s0 // (op 667) least significant
s_mul_hi_u32 s19, s8, s0 // (op 667) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 667) most significant: sum
s_add_u32 s15, s15, s19 // (op 667) most significant: sum



s_ashr_i64 s[0:1], s[14:15], 3 // (op 667) 

s_mov_b32 s44, s10 // (op 667) reg expression
s_mov_b32 s45, s11 // (op 667) reg expression
s_mov_b32 s46, s0 // (op 667) reg expression
s_mov_b32 s47, 131072 // (op 667) call()

v_and_b32 v14, -16, v1 // (op 710) 
v_sub_u32 v15, v1, v14 // (op 710) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 710) 

v_lshrrev_b32 v15, 4, v1 // (op 710) 
v_lshlrev_b32 v20, 1, v15 // (op 710) 

v_mul_lo_u32 v15, v20, s12 // (op 710) 

v_add_u32 v20, v14, v15 // (op 710) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 712) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 712) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 712) least significant
s_mul_hi_u32 s15, s12, s22 // (op 712) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 712) most significant: sum
s_add_u32 s1, s1, s15 // (op 712) most significant: sum


s_mov_b32 s48, s20 // (op 712) reg expression
s_mov_b32 s49, s21 // (op 712) reg expression
s_mov_b32 s50, s0 // (op 712) reg expression
s_mov_b32 s51, 131072 // (op 712) call()

v_and_b32 v14, -16, v1 // (op 729) 
v_sub_u32 v15, v1, v14 // (op 729) 

v_lshl_add_u32 v14, s16, 4, v15 // (op 729) 

v_lshrrev_b32 v15, 4, v1 // (op 729) 
v_lshlrev_b32 v21, 1, v15 // (op 729) 

v_mul_lo_u32 v15, v21, s12 // (op 729) 

v_add_u32 v21, v14, v15 // (op 729) 


s_mul_i32 s1, s12, s23 // (op 731) most significant: low of low * high
s_mul_i32 s14, s13, s22 // (op 731) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 731) least significant
s_mul_hi_u32 s15, s12, s22 // (op 731) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 731) most significant: sum
s_add_u32 s1, s1, s15 // (op 731) most significant: sum


s_mov_b32 s48, s20 // (op 731) reg expression
s_mov_b32 s49, s21 // (op 731) reg expression
s_mov_b32 s50, s0 // (op 731) reg expression
s_mov_b32 s51, 131072 // (op 731) call()

v_and_b32 v14, -16, v1 // (op 774) 
v_sub_u32 v15, v1, v14 // (op 774) 

v_lshl_add_u32 v14, s17, 4, v15 // (op 774) 

v_lshrrev_b32 v15, 4, v1 // (op 774) 
v_lshlrev_b32 v22, 1, v15 // (op 774) 

v_mul_lo_u32 v15, v22, s24 // (op 774) 
v_add_u32 v15, v15, v14 // (op 774) 


s_mul_i32 s1, s24, s23 // (op 776) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 776) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 776) least significant
s_mul_hi_u32 s15, s24, s22 // (op 776) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 776) most significant: sum
s_add_u32 s1, s1, s15 // (op 776) most significant: sum


s_mov_b32 s52, s26 // (op 776) reg expression
s_mov_b32 s53, s27 // (op 776) reg expression
s_mov_b32 s54, s0 // (op 776) reg expression
s_mov_b32 s55, 131072 // (op 776) call()

v_and_b32 v14, -16, v1 // (op 793) 
v_sub_u32 v22, v1, v14 // (op 793) 

v_lshl_add_u32 v14, s17, 4, v22 // (op 793) 

v_lshrrev_b32 v22, 4, v1 // (op 793) 
v_lshlrev_b32 v23, 1, v22 // (op 793) 

v_mul_lo_u32 v22, v23, s24 // (op 793) 
v_add_u32 v22, v22, v14 // (op 793) 


s_mul_i32 s1, s24, s23 // (op 795) most significant: low of low * high
s_mul_i32 s14, s25, s22 // (op 795) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 795) least significant
s_mul_hi_u32 s15, s24, s22 // (op 795) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 795) most significant: sum
s_add_u32 s1, s1, s15 // (op 795) most significant: sum


s_mov_b32 s52, s26 // (op 795) reg expression
s_mov_b32 s53, s27 // (op 795) reg expression
s_mov_b32 s54, s0 // (op 795) reg expression
s_mov_b32 s55, 131072 // (op 795) call()


s_mul_i32 s1, s9, 6 // (op 424) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 424) least significant
s_mul_hi_u32 s14, s8, 6 // (op 424) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 424) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 424) 


v_mov_b32 v14, v12 // (op 424) reg expression
buffer_load_dwordx3 v[26:28], v14, s[44:47], 0 offen // (op 424) Load value
v_add_u32 v14, v14, v24 // (op 424) 
buffer_load_dwordx3 v[30:32], v14, s[44:47], 0 offen // (op 424) Load value
v_add_u32 v14, v14, v24 // (op 424) 
buffer_load_dwordx3 v[34:36], v14, s[44:47], 0 offen // (op 424) Load value
v_add_u32 v14, v14, v24 // (op 424) 
buffer_load_dwordx3 v[38:40], v14, s[44:47], 0 offen // (op 424) Load value



v_mov_b64 v[24:25], s[24:25] // (op 427) convert

v_mov_b32 v14, v15 // (op 427) reg expression
buffer_load_ubyte v42, v14, s[52:55], 0 offen offset: 0 // (op 427) Load value
v_add_u32 v14, v14, v24 // (op 427) 
buffer_load_ubyte v43, v14, s[52:55], 0 offen offset: 0 // (op 427) Load value


v_lshlrev_b32 v14, 4, v1 // (op 859) 
s_nop 0
v_readfirstlane_b32 s0, v14 // (op 859) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 538) 
v_mov_b64 v[24:25], s[14:15] // (op 538) convert


v_mov_b64 v[44:45], v[16:17] // (op 538) reg expression
s_mov_b32 m0, s0 // (op 538) reg expression
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 538) Load value direct to lds
v_add_co_u32 v44, s[14:15], v44, v24 // (op 538) least significant half
v_addc_co_u32 v45, s[14:15], 0, v45, s[14:15] // (op 538) most significant half

s_add_u32 m0, m0, 1024 // (op 538) 
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 538) Load value direct to lds
v_add_co_u32 v44, s[14:15], v44, v24 // (op 538) least significant half
v_addc_co_u32 v45, s[14:15], 0, v45, s[14:15] // (op 538) most significant half

s_add_u32 m0, m0, 1024 // (op 538) 
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 538) Load value direct to lds
v_add_co_u32 v44, s[14:15], v44, v24 // (op 538) least significant half
v_addc_co_u32 v45, s[14:15], 0, v45, s[14:15] // (op 538) most significant half

s_add_u32 m0, m0, 1024 // (op 538) 
buffer_load_dwordx4 v44, s[40:43], 0 offen offset: 0 lds // (op 538) Load value direct to lds




v_mov_b64 v[24:25], s[12:13] // (op 433) convert

v_mov_b32 v14, v20 // (op 433) reg expression
buffer_load_ubyte v44, v14, s[48:51], 0 offen offset: 0 // (op 433) Load value
v_add_u32 v14, v14, v24 // (op 433) 
buffer_load_ubyte v45, v14, s[48:51], 0 offen offset: 0 // (op 433) Load value



s_mul_i32 s1, s9, 6 // (op 436) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 436) least significant
s_mul_hi_u32 s14, s8, 6 // (op 436) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 436) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 436) 


s_lshl_b64 s[0:1], s[8:9], 8 // (op 436) 
s_mul_i32 s15, s1, 6 // (op 436) most significant: low of high * low
s_mul_i32 s14, s0, 6 // (op 436) least significant
s_mul_hi_u32 s18, s0, 6 // (op 436) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 436) most significant: sum


s_lshr_b64 s[0:1], s[14:15], 3 // (op 436) 

v_add_u32 v14, v12, s0 // (op 436) 

buffer_load_dwordx3 v[46:48], v14, s[44:47], 0 offen // (op 436) Load value
v_add_u32 v14, v14, v24 // (op 436) 
buffer_load_dwordx3 v[50:52], v14, s[44:47], 0 offen // (op 436) Load value
v_add_u32 v14, v14, v24 // (op 436) 
buffer_load_dwordx3 v[54:56], v14, s[44:47], 0 offen // (op 436) Load value
v_add_u32 v14, v14, v24 // (op 436) 
buffer_load_dwordx3 v[58:60], v14, s[44:47], 0 offen // (op 436) Load value




v_mov_b64 v[24:25], s[24:25] // (op 439) convert

s_lshl_b32 s0, s24, 3 // (op 439) 
v_add_u32 v12, v15, s0 // (op 439) 

buffer_load_ubyte v62, v12, s[52:55], 0 offen offset: 0 // (op 439) Load value
v_add_u32 v12, v12, v24 // (op 439) 
buffer_load_ubyte v63, v12, s[52:55], 0 offen offset: 0 // (op 439) Load value



v_lshlrev_b32 v12, 4, v1 // (op 1091) 
s_nop 0
v_readfirstlane_b32 s0, v12 // (op 1091) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[14:15], s[2:3], 6 // (op 543) 
v_mov_b64 v[14:15], s[14:15] // (op 543) convert


s_lshl_b64 s[14:15], s[2:3], 8 // (op 543) 
v_mov_b32 v12, s14 // (op 543) 
v_mov_b32 v29, s15 // (op 543) 
v_add_co_u32 v24, s[18:19], v16, v12 // (op 543) least significant half
v_addc_co_u32 v25, s[18:19], v17, v29, s[18:19] // (op 543) most significant half




s_add_u32 s1, 4096, s0 // (op 543) 
s_mov_b32 m0, s1 // (op 543) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 543) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 543) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 543) most significant half

s_add_u32 m0, m0, 1024 // (op 543) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 543) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 543) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 543) most significant half

s_add_u32 m0, m0, 1024 // (op 543) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 543) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 543) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 543) most significant half

s_add_u32 m0, m0, 1024 // (op 543) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 543) Load value direct to lds






v_mov_b64 v[24:25], s[12:13] // (op 445) convert

s_lshl_b32 s0, s12, 3 // (op 445) 
v_add_u32 v12, v20, s0 // (op 445) 

buffer_load_ubyte v14, v12, s[48:51], 0 offen offset: 0 // (op 445) Load value
v_add_u32 v12, v12, v24 // (op 445) 
buffer_load_ubyte v15, v12, s[48:51], 0 offen offset: 0 // (op 445) Load value



v_lshlrev_b32 v12, 6, v1 // (op 912) 
v_mul_lo_u32 v20, v12, 6 // (op 912) 
v_lshrrev_b32 v29, 3, v20 // (op 912) 

v_lshrrev_b32 v20, 4, v12 // (op 912) 

v_lshlrev_b32 v12, 2, v20 // (op 912) 

v_add_u32 v20, v29, v12 // (op 912) 





s_waitcnt vmcnt(23)
ds_write_b96 v20, v[26:28] offset:8192 // (op 447) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v20, v[30:32] offset:8208 // (op 447) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v20, v[34:36] offset:8224 // (op 447) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v20, v[38:40] offset:8240 // (op 447) Store local data 

v_and_b32 v12, -16, v1 // (op 1018) 
v_sub_u32 v20, v1, v12 // (op 1018) 

v_lshrrev_b32 v12, 4, v1 // (op 1018) 
v_lshl_add_u32 v29, v12, 5, v20 // (op 1018) 





s_waitcnt vmcnt(19)
ds_write_b8 v29, v42 offset:12288 // (op 448) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v29, v43 offset:12304 // (op 448) Store local data 

v_and_b32 v12, -16, v1 // (op 965) 
v_sub_u32 v20, v1, v12 // (op 965) 

v_lshrrev_b32 v12, 4, v1 // (op 965) 
v_lshl_add_u32 v29, v12, 5, v20 // (op 965) 





s_waitcnt vmcnt(13)
ds_write_b8 v29, v44 offset:12416 // (op 450) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v29, v45 offset:12432 // (op 450) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 1231) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 467) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 467) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1272) 


ds_read_u8 v12, v7 offset:12288 // (op 473) Load local data 
s_barrier  // (op 1212) 



ds_read_b64_tr_b8 v[70:71], v0 // (op 464) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:128 // (op 464) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:1024 // (op 464) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:1152 // (op 464) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1252) 


ds_read_u8 v20, v4 offset:12416 // (op 470) Load local data 
v_lshlrev_b32 v29, 4, v1 // (op 831) 
s_nop 0
v_readfirstlane_b32 s0, v29 // (op 831) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 6, v1 // (op 883) 
v_mul_lo_u32 v33, v29, 6 // (op 883) 
v_lshrrev_b32 v37, 3, v33 // (op 883) 

v_lshrrev_b32 v33, 4, v29 // (op 883) 

v_lshlrev_b32 v29, 2, v33 // (op 883) 

v_add_u32 v33, v37, v29 // (op 883) 


v_and_b32 v29, -16, v1 // (op 936) 
v_sub_u32 v37, v1, v29 // (op 936) 

v_lshrrev_b32 v29, 4, v1 // (op 936) 
v_lshl_add_u32 v41, v29, 5, v37 // (op 936) 


v_and_b32 v29, -16, v1 // (op 989) 
v_sub_u32 v37, v1, v29 // (op 989) 

v_lshrrev_b32 v29, 4, v1 // (op 989) 
v_lshl_add_u32 v49, v29, 5, v37 // (op 989) 


v_lshlrev_b32 v29, 4, v1 // (op 1076) 
s_nop 0
v_readfirstlane_b32 s1, v29 // (op 1076) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v29, 6, v1 // (op 1115) 
v_mul_lo_u32 v37, v29, 6 // (op 1115) 
v_lshrrev_b32 v53, 3, v37 // (op 1115) 

v_lshrrev_b32 v37, 4, v29 // (op 1115) 

v_lshlrev_b32 v29, 2, v37 // (op 1115) 

v_add_u32 v37, v53, v29 // (op 1115) 


v_and_b32 v29, -16, v1 // (op 1139) 
v_sub_u32 v53, v1, v29 // (op 1139) 

v_lshrrev_b32 v29, 4, v1 // (op 1139) 
v_lshl_add_u32 v57, v29, 5, v53 // (op 1139) 


v_and_b32 v29, -16, v1 // (op 1163) 
v_sub_u32 v53, v1, v29 // (op 1163) 

v_lshrrev_b32 v29, 4, v1 // (op 1163) 
v_lshl_add_u32 v61, v29, 5, v53 // (op 1163) 


s_mov_b32 s14, 0 // (op 59) call()
s_ashr_i64 s[18:19], s[6:7], 8 // (op 58) 
s_mov_b32 s15, s18 // (op 58) convert

s_and_b32 s18, s15, 4294967294 // (op 58) 

s_cmp_lt_u32 s14, s18 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58:

s_add_u32 s14, s14, 2 // (op 60) 
s_mul_i32 s19, s9, 6 // (op 10) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 10) least significant
s_mul_hi_u32 s15, s8, 6 // (op 10) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 10) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[18:19] // (op 10) 


v_mov_b32 v29, v13 // (op 10) 
s_lshl_b64 s[18:19], s[8:9], 8 // (op 10) 
s_mul_i32 s39, s19, 6 // (op 10) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 10) least significant
s_mul_hi_u32 s15, s18, 6 // (op 10) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 10) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 10) 

s_lshl_b32 s15, s18, 1 // (op 10) 

v_add_u32 v53, v29, s15 // (op 10) 

buffer_load_dwordx3 v[26:28], v53, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v53, v53, v16 // (op 10) 
buffer_load_dwordx3 v[30:32], v53, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v53, v53, v16 // (op 10) 
buffer_load_dwordx3 v[34:36], v53, s[44:47], 0 offen // (op 10) Load value
v_add_u32 v53, v53, v16 // (op 10) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 88) 
buffer_load_dwordx3 v[38:40], v53, s[44:47], 0 offen // (op 10) Load value



s_barrier  // (op 1221) 
v_mov_b64 v[16:17], s[24:25] // (op 25) convert




ds_read_b64_tr_b8 v[70:71], v0 offset:2048 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:2176 // (op 91) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:3072 // (op 91) Transpose load local data 
v_mov_b32 v53, v22 // (op 25) 
s_lshl_b32 s15, s24, 4 // (op 25) 
v_add_u32 v23, v53, s15 // (op 25) 

buffer_load_ubyte v42, v23, s[52:55], 0 offen offset: 0 // (op 25) Load value
v_add_u32 v23, v23, v16 // (op 25) 
buffer_load_ubyte v43, v23, s[52:55], 0 offen offset: 0 // (op 25) Load value


ds_read_b64_tr_b8 v[76:77], v0 offset:3200 // (op 91) Transpose load local data 
s_barrier  // (op 1241) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1215) 




ds_read_b96_tr_b6 v[64:66], v5 offset:10240 // (op 95) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:10496 // (op 95) Transpose load local data 
s_lshl_b64 s[18:19], s[2:3], 6 // (op 522) 
s_barrier  // (op 1262) 
v_mov_b64 v[16:17], s[18:19] // (op 522) convert




ds_read_u8 v20, v4 offset:12480 // (op 99) Load local data 
v_mov_b64 v[24:25], v[18:19] // (op 522) 
s_barrier  // (op 1282) 
s_lshl_b64 s[18:19], s[2:3], 8 // (op 522) 


s_lshl_b64 s[38:39], s[18:19], 1 // (op 522) 

v_mov_b32 v23, s38 // (op 522) 
v_mov_b32 v78, s39 // (op 522) 
v_add_co_u32 v80, s[18:19], v24, v23 // (op 522) least significant half
v_addc_co_u32 v81, s[18:19], v25, v78, s[18:19] // (op 522) most significant half




s_mov_b32 m0, s0 // (op 522) reg expression
ds_read_u8 v12, v7 offset:12352 // (op 103) Load local data 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v16 // (op 522) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[64:69], a[0:3], v20, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 105) 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v16 // (op 522) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 522) Load value direct to lds
v_add_co_u32 v80, s[18:19], v80, v16 // (op 522) least significant half
v_addc_co_u32 v81, s[18:19], 0, v81, s[18:19] // (op 522) most significant half

s_add_u32 m0, m0, 1024 // (op 522) 
buffer_load_dwordx4 v80, s[40:43], 0 offen offset: 0 lds // (op 522) Load value direct to lds



v_mov_b64 v[16:17], s[12:13] // (op 17) convert

v_mov_b32 v23, v21 // (op 17) 
s_lshl_b32 s15, s12, 4 // (op 17) 
v_add_u32 v78, v23, s15 // (op 17) 

buffer_load_ubyte v44, v78, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v78, v78, v16 // (op 17) 
buffer_load_ubyte v45, v78, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1302) 



s_waitcnt vmcnt(23)
ds_write_b96 v37, v[46:48] offset:12544 // (op 302) Store local data 
s_waitcnt vmcnt(22)
ds_write_b96 v37, v[50:52] offset:12560 // (op 302) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v37, v[54:56] offset:12576 // (op 302) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v37, v[58:60] offset:12592 // (op 302) Store local data 
s_barrier  // (op 1314) 



s_waitcnt vmcnt(19)
ds_write_b8 v61, v62 offset:16640 // (op 304) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v61, v63 offset:16656 // (op 304) Store local data 
s_barrier  // (op 1308) 



s_waitcnt vmcnt(13)
ds_write_b8 v57, v14 offset:16768 // (op 303) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v57, v15 offset:16784 // (op 303) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1298) 



ds_read_b96_tr_b6 v[78:80], v9 offset:12544 // (op 287) Transpose load local data 
ds_read_b96_tr_b6 v[81:83], v9 offset:12800 // (op 287) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1311) 


ds_read_u8 v16, v11 offset:16640 // (op 291) Load local data 
s_barrier  // (op 1292) 



ds_read_b64_tr_b8 v[84:85], v3 offset:4096 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v3 offset:4224 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v3 offset:5120 // (op 285) Transpose load local data 
ds_read_b64_tr_b8 v[90:91], v3 offset:5248 // (op 285) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1305) 


ds_read_u8 v17, v8 offset:16768 // (op 289) Load local data 

s_mul_i32 s19, s9, 6 // (op 305) most significant: low of high * low
s_mul_i32 s18, s8, 6 // (op 305) least significant
s_mul_hi_u32 s15, s8, 6 // (op 305) most significant: high of low * low
s_add_u32 s19, s19, s15 // (op 305) most significant: sum

v_lshrrev_b64 v[92:93], 3, s[18:19] // (op 305) 


s_lshl_b64 s[18:19], s[8:9], 8 // (op 305) 
s_mul_i32 s39, s19, 6 // (op 305) most significant: low of high * low
s_mul_i32 s38, s18, 6 // (op 305) least significant
s_mul_hi_u32 s15, s18, 6 // (op 305) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 305) most significant: sum


s_lshr_b64 s[18:19], s[38:39], 3 // (op 305) 

s_mul_i32 s15, 3, s18 // (op 305) 

v_add_u32 v94, v29, s15 // (op 305) 

buffer_load_dwordx3 v[46:48], v94, s[44:47], 0 offen // (op 305) Load value
v_add_u32 v94, v94, v92 // (op 305) 
buffer_load_dwordx3 v[50:52], v94, s[44:47], 0 offen // (op 305) Load value
v_add_u32 v94, v94, v92 // (op 305) 
buffer_load_dwordx3 v[54:56], v94, s[44:47], 0 offen // (op 305) Load value
v_add_u32 v94, v94, v92 // (op 305) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:91], v[78:83], a[0:3], v17, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 292) 
buffer_load_dwordx3 v[58:60], v94, s[44:47], 0 offen // (op 305) Load value











v_mov_b64 v[16:17], s[24:25] // (op 307) convert

s_lshl_b32 s15, s24, 3 // (op 307) 
s_mul_i32 s18, 3, s15 // (op 307) 

v_add_u32 v29, v53, s18 // (op 307) 

buffer_load_ubyte v62, v29, s[52:55], 0 offen offset: 0 // (op 307) Load value
v_add_u32 v29, v29, v16 // (op 307) 
buffer_load_ubyte v63, v29, s[52:55], 0 offen offset: 0 // (op 307) Load value


s_lshl_b64 s[18:19], s[8:9], 8 // (op 657) 

s_mul_i32 s39, s19, 6 // (op 657) most significant: low of high * low
s_barrier  // (op 1295) 
s_mul_i32 s38, s18, 6 // (op 657) least significant

s_mul_hi_u32 s15, s18, 6 // (op 657) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 657) most significant: sum


s_lshl_b32 s15, s24, 4 // (op 785) 
v_add_u32 v22, v22, s15 // (op 785) 

s_lshr_b64 s[18:19], s[38:39], 3 // (op 657) 

s_lshl_b32 s15, s18, 1 // (op 657) 

v_add_u32 v13, v13, s15 // (op 657) 

s_lshl_b64 s[18:19], s[2:3], 6 // (op 533) 
v_mov_b64 v[16:17], s[18:19] // (op 533) convert


s_lshl_b64 s[18:19], s[2:3], 8 // (op 533) 
s_mul_i32 s39, 3, s19 // (op 533) most significant: low of low * high
s_mul_i32 s38, 3, s18 // (op 533) least significant
s_mul_hi_u32 s15, 3, s18 // (op 533) most significant: high of low * low
s_add_u32 s39, s39, s15 // (op 533) most significant: sum


v_mov_b32 v29, s38 // (op 533) 
v_mov_b32 v53, s39 // (op 533) 
v_add_co_u32 v78, s[18:19], v24, v29 // (op 533) least significant half
v_addc_co_u32 v79, s[18:19], v25, v53, s[18:19] // (op 533) most significant half




s_add_u32 s15, 4096, s1 // (op 533) 
s_mov_b32 m0, s15 // (op 533) reg expression
ds_read_b64_tr_b8 v[80:81], v3 offset:6144 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[82:83], v3 offset:6272 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v3 offset:7168 // (op 294) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v3 offset:7296 // (op 294) Transpose load local data 
buffer_load_dwordx4 v78, s[40:43], 0 offen offset: 0 lds // (op 533) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v16 // (op 533) least significant half



v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 533) most significant half

s_add_u32 m0, m0, 1024 // (op 533) 
ds_read_b96_tr_b6 v[88:90], v9 offset:14592 // (op 296) Transpose load local data 
ds_read_b96_tr_b6 v[91:93], v9 offset:14848 // (op 296) Transpose load local data 
buffer_load_dwordx4 v78, s[40:43], 0 offen offset: 0 lds // (op 533) Load value direct to lds
v_add_co_u32 v78, s[18:19], v78, v16 // (op 533) least significant half


s_barrier  // (op 1369) 
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 533) most significant half

s_add_u32 m0, m0, 1024 // (op 533) 
buffer_load_dwordx4 v78, s[40:43], 0 offen offset: 0 lds // (op 533) Load value direct to lds
ds_read_u8 v29, v8 offset:16832 // (op 298) Load local data 
v_add_co_u32 v78, s[18:19], v78, v16 // (op 533) least significant half
v_addc_co_u32 v79, s[18:19], 0, v79, s[18:19] // (op 533) most significant half



s_barrier  // (op 1373) 
s_add_u32 m0, m0, 1024 // (op 533) 
ds_read_u8 v53, v11 offset:16704 // (op 300) Load local data 
buffer_load_dwordx4 v78, s[40:43], 0 offen offset: 0 lds // (op 533) Load value direct to lds





s_barrier  // (op 1377) 
v_mov_b64 v[16:17], s[12:13] // (op 306) convert

s_lshl_b32 s15, s12, 3 // (op 306) 
s_mul_i32 s18, 3, s15 // (op 306) 

v_add_u32 v24, v23, s18 // (op 306) 

buffer_load_ubyte v14, v24, s[48:51], 0 offen offset: 0 // (op 306) Load value
v_add_u32 v24, v24, v16 // (op 306) 
s_lshl_b64 s[18:19], s[2:3], 8 // (op 593) 
s_lshl_b64 s[38:39], s[18:19], 1 // (op 593) 

v_mov_b32 v25, s38 // (op 593) 
buffer_load_ubyte v15, v24, s[48:51], 0 offen offset: 0 // (op 306) Load value


v_mov_b32 v24, s39 // (op 593) 

v_add_co_u32 v18, s[18:19], v18, v25 // (op 593) least significant half
s_barrier  // (op 1234) 
v_addc_co_u32 v19, s[18:19], v19, v24, s[18:19] // (op 593) most significant half







s_lshl_b32 s15, s12, 4 // (op 721) 
v_add_u32 v21, v21, s15 // (op 721) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:87], v[88:93], a[0:3], v29, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 301) 
s_waitcnt vmcnt(23)
ds_write_b96 v33, v[26:28] offset:8192 // (op 13) Store local data 




s_waitcnt vmcnt(22)
ds_write_b96 v33, v[30:32] offset:8208 // (op 13) Store local data 
s_waitcnt vmcnt(21)
ds_write_b96 v33, v[34:36] offset:8224 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b96 v33, v[38:40] offset:8240 // (op 13) Store local data 
s_barrier  // (op 1275) 



s_waitcnt vmcnt(19)
ds_write_b8 v49, v42 offset:12288 // (op 30) Store local data 
s_waitcnt vmcnt(18)
ds_write_b8 v49, v43 offset:12304 // (op 30) Store local data 
s_barrier  // (op 1255) 



s_waitcnt vmcnt(13)
ds_write_b8 v41, v44 offset:12416 // (op 21) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v41, v45 offset:12432 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1244) 



ds_read_b96_tr_b6 v[64:66], v5 offset:8192 // (op 75) Transpose load local data 
ds_read_b96_tr_b6 v[67:69], v5 offset:8448 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1285) 


ds_read_u8 v12, v7 offset:12288 // (op 85) Load local data 
s_barrier  // (op 1224) 



ds_read_b64_tr_b8 v[70:71], v0 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[72:73], v0 offset:128 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[74:75], v0 offset:1024 // (op 70) Transpose load local data 
ds_read_b64_tr_b8 v[76:77], v0 offset:1152 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1265) 


ds_read_u8 v20, v4 offset:12416 // (op 80) Load local data 
s_ashr_i64 s[18:19], s[6:7], 8 // (op 58) 
s_mov_b32 s15, s18 // (op 58) convert

s_and_b32 s18, s15, 4294967294 // (op 58) 

s_cmp_lt_u32 s14, s18 // (op 58) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_58:
















s_barrier  // (op 1327) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 556) 
s_lshr_b32 s14, s0, 1 // (op 556) 

v_lshl_add_u32 v13, s14, 9, v1 // (op 556) 

s_lshl_b32 s0, s16, 4 // (op 556) 
v_mul_lo_u32 v17, v13, s3 // (op 556) most significant: low of low * high
v_mul_hi_u32 v49, v13, s2 // (op 556) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 556) least significant: low of low * low
v_add_u32 v17, v17, v49 // (op 556) most significant: sum


s_mov_b32 s15, 0 // (op 556) convert
s_mov_b32 s14, s0 // (op 556) convert

v_mov_b32 v13, s14 // (op 556) 
v_mov_b32 v49, s15 // (op 556) 
v_add_co_u32 v18, s[0:1], v13, v16 // (op 556) least significant half
v_addc_co_u32 v19, s[0:1], v49, v17, s[0:1] // (op 556) most significant half





s_mul_i32 s1, s2, s7 // (op 558) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 558) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 558) least significant
s_mul_hi_u32 s15, s2, s6 // (op 558) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 558) most significant: sum
s_add_u32 s1, s1, s15 // (op 558) most significant: sum


s_mov_b32 s40, s4 // (op 558) reg expression
s_mov_b32 s41, s5 // (op 558) reg expression
s_mov_b32 s42, s0 // (op 558) reg expression
s_mov_b32 s43, 131072 // (op 558) call()

v_lshlrev_b32 v13, 2, v1 // (op 623) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 623) 
s_lshr_b32 s14, s0, 1 // (op 623) 

v_lshl_add_u32 v49, s14, 9, v13 // (op 623) 


s_lshl_b32 s0, s17, 4 // (op 623) 
s_mov_b32 s15, 0 // (op 623) convert
s_mov_b32 s14, s0 // (op 623) convert

v_mul_lo_u32 v17, v49, s9 // (op 623) most significant: low of low * high
v_mul_hi_u32 v13, v49, s8 // (op 623) most significant: high of low * low
v_mul_lo_u32 v16, v49, s8 // (op 623) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 623) most significant: sum

v_mov_b32 v13, s14 // (op 623) 
v_mov_b32 v53, s15 // (op 623) 
v_add_co_u32 v16, s[0:1], v16, v13 // (op 623) least significant half
v_addc_co_u32 v17, s[0:1], v17, v53, s[0:1] // (op 623) most significant half





v_mul_lo_u32 v45, v17, 6 // (op 623) most significant: low of high * low
v_mul_hi_u32 v13, v16, 6 // (op 623) most significant: high of low * low
v_mul_lo_u32 v44, v16, 6 // (op 623) least significant: low of low * low
v_add_u32 v45, v45, v13 // (op 623) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[44:45] // (op 623) 

v_mov_b32 v13, v16 // (op 623) convert

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

v_and_b32 v49, -16, v1 // (op 687) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 687) 
v_sub_u32 v53, v1, v49 // (op 687) 

s_lshr_b32 s14, s0, 1 // (op 687) 

v_lshl_add_u32 v49, s16, 4, v53 // (op 687) 

v_lshrrev_b32 v53, 4, v1 // (op 687) 
v_lshlrev_b32 v57, 1, v53 // (op 687) 

v_lshl_add_u32 v53, s14, 4, v57 // (op 687) 


v_mul_lo_u32 v57, v53, s12 // (op 687) 

v_add_u32 v53, v49, v57 // (op 687) 


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

v_lshrrev_b32 v49, 4, v1 // (op 751) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 751) 
v_lshlrev_b32 v57, 1, v49 // (op 751) 

s_lshr_b32 s14, s0, 1 // (op 751) 

v_and_b32 v49, -16, v1 // (op 751) 
v_lshl_add_u32 v61, s14, 4, v57 // (op 751) 


v_sub_u32 v57, v1, v49 // (op 751) 

v_lshl_add_u32 v49, s17, 4, v57 // (op 751) 

v_mul_lo_u32 v57, v61, s24 // (op 751) 
v_add_u32 v57, v57, v49 // (op 751) 


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

v_lshlrev_b32 v49, 4, v1 // (op 844) 
s_nop 0
v_readfirstlane_b32 s0, v49 // (op 844) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v49, 6, v1 // (op 896) 
v_mul_lo_u32 v61, v49, 6 // (op 896) 
v_lshrrev_b32 v16, 3, v61 // (op 896) 

v_lshrrev_b32 v17, 4, v49 // (op 896) 

v_lshlrev_b32 v49, 2, v17 // (op 896) 

v_add_u32 v17, v16, v49 // (op 896) 


v_and_b32 v16, -16, v1 // (op 949) 
v_sub_u32 v49, v1, v16 // (op 949) 

v_lshrrev_b32 v16, 4, v1 // (op 949) 
v_lshl_add_u32 v61, v16, 5, v49 // (op 949) 


v_and_b32 v16, -16, v1 // (op 1002) 
v_sub_u32 v49, v1, v16 // (op 1002) 

v_lshrrev_b32 v16, 4, v1 // (op 1002) 
v_lshl_add_u32 v21, v16, 5, v49 // (op 1002) 


s_ashr_i64 s[14:15], s[6:7], 8 // (op 194) 
s_and_b32 s1, s14, 4294967294 // (op 194) 

s_ashr_i64 s[14:15], s[6:7], 8 // (op 193) 
s_mov_b32 s18, s14 // (op 193) convert

s_cmp_lt_u32 s1, s18 // (op 193) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 // (op 193) Condition: Top (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193:
s_barrier  // (op 1218) 
s_add_u32 s1, s1, 1 // (op 195) 

s_lshl_b64 s[14:15], s[2:3], 6 // (op 527) 
v_mov_b64 v[14:15], s[14:15] // (op 527) convert


v_mov_b64 v[22:23], v[18:19] // (op 527) 
v_mov_b64 v[24:25], v[22:23] // (op 527) reg expression
s_mov_b32 m0, s0 // (op 527) reg expression
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 527) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 527) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 527) Load value direct to lds
v_add_co_u32 v24, s[14:15], v24, v14 // (op 527) least significant half
v_addc_co_u32 v25, s[14:15], 0, v25, s[14:15] // (op 527) most significant half

s_add_u32 m0, m0, 1024 // (op 527) 
buffer_load_dwordx4 v24, s[40:43], 0 offen offset: 0 lds // (op 527) Load value direct to lds



s_lshl_b64 s[14:15], s[2:3], 8 // (op 546) 
v_mov_b32 v12, s14 // (op 546) 
v_mov_b32 v20, s15 // (op 546) 
v_add_co_u32 v18, s[18:19], v18, v12 // (op 546) least significant half
v_addc_co_u32 v19, s[18:19], v19, v20, s[18:19] // (op 546) most significant half





s_mul_i32 s15, s9, 6 // (op 221) most significant: low of high * low
s_mul_i32 s14, s8, 6 // (op 221) least significant
s_mul_hi_u32 s18, s8, 6 // (op 221) most significant: high of low * low
s_add_u32 s15, s15, s18 // (op 221) most significant: sum

v_lshrrev_b64 v[14:15], 3, s[14:15] // (op 221) 


v_mov_b32 v12, v13 // (op 221) 
v_mov_b32 v16, v12 // (op 221) reg expression
buffer_load_dwordx3 v[22:24], v16, s[44:47], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v14 // (op 221) 
buffer_load_dwordx3 v[26:28], v16, s[44:47], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v14 // (op 221) 
buffer_load_dwordx3 v[30:32], v16, s[44:47], 0 offen // (op 221) Load value
v_add_u32 v16, v16, v14 // (op 221) 
buffer_load_dwordx3 v[34:36], v16, s[44:47], 0 offen // (op 221) Load value



s_lshl_b64 s[14:15], s[8:9], 8 // (op 615) 
s_mul_i32 s19, s15, 6 // (op 615) most significant: low of high * low
s_mul_i32 s18, s14, 6 // (op 615) least significant
s_mul_hi_u32 s38, s14, 6 // (op 615) most significant: high of low * low
s_add_u32 s19, s19, s38 // (op 615) most significant: sum


s_lshr_b64 s[14:15], s[18:19], 3 // (op 615) 

v_add_u32 v13, v13, s14 // (op 615) 


s_barrier  // (op 1237) 
v_mov_b64 v[14:15], s[12:13] // (op 222) convert




v_mov_b32 v12, v53 // (op 222) 
v_mov_b32 v16, v12 // (op 222) reg expression
buffer_load_ubyte v54, v16, s[48:51], 0 offen offset: 0 // (op 222) Load value
v_add_u32 v16, v16, v14 // (op 222) 
s_waitcnt vmcnt(4)
ds_write_b96 v17, v[22:24] offset:8192 // (op 218) Store local data 
s_waitcnt vmcnt(3)
ds_write_b96 v17, v[26:28] offset:8208 // (op 218) Store local data 
s_waitcnt vmcnt(2)
ds_write_b96 v17, v[30:32] offset:8224 // (op 218) Store local data 
buffer_load_ubyte v55, v16, s[48:51], 0 offen offset: 0 // (op 222) Load value



s_lshl_b32 s14, s12, 3 // (op 679) 
v_add_u32 v53, v53, s14 // (op 679) 

s_waitcnt vmcnt(2)
ds_write_b96 v17, v[34:36] offset:8240 // (op 218) Store local data 


v_mov_b64 v[14:15], s[24:25] // (op 223) convert

s_barrier  // (op 1258) 
v_mov_b32 v12, v57 // (op 223) 



v_mov_b32 v16, v12 // (op 223) reg expression
buffer_load_ubyte v22, v16, s[52:55], 0 offen offset: 0 // (op 223) Load value
v_add_u32 v16, v16, v14 // (op 223) 
buffer_load_ubyte v23, v16, s[52:55], 0 offen offset: 0 // (op 223) Load value



s_lshl_b32 s14, s24, 3 // (op 743) 
v_add_u32 v57, v57, s14 // (op 743) 

s_waitcnt vmcnt(3)
ds_write_b8 v61, v54 offset:12416 // (op 219) Store local data 
s_barrier  // (op 1227) 
s_waitcnt vmcnt(2)
ds_write_b8 v61, v55 offset:12432 // (op 219) Store local data 




ds_read_b64_tr_b8 v[24:25], v0 // (op 201) Transpose load local data 
s_barrier  // (op 1278) 
ds_read_b64_tr_b8 v[26:27], v0 offset:128 // (op 201) Transpose load local data 



ds_read_b64_tr_b8 v[28:29], v0 offset:1024 // (op 201) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:1152 // (op 201) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v21, v22 offset:12288 // (op 220) Store local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1248) 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v23 offset:12304 // (op 220) Store local data 




ds_read_b96_tr_b6 v[32:34], v5 offset:8192 // (op 203) Transpose load local data 
ds_read_b96_tr_b6 v[35:37], v5 offset:8448 // (op 203) Transpose load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1268) 


ds_read_u8 v12, v4 offset:12416 // (op 205) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1288) 


ds_read_u8 v20, v7 offset:12288 // (op 207) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:37], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 208) 







ds_read_b64_tr_b8 v[22:23], v0 offset:2048 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[24:25], v0 offset:2176 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[26:27], v0 offset:3072 // (op 210) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:3200 // (op 210) Transpose load local data 



ds_read_b96_tr_b6 v[30:32], v5 offset:10240 // (op 212) Transpose load local data 
ds_read_b96_tr_b6 v[33:35], v5 offset:10496 // (op 212) Transpose load local data 


ds_read_u8 v12, v4 offset:12480 // (op 214) Load local data 


ds_read_u8 v20, v7 offset:12352 // (op 216) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v12, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 217) 




s_ashr_i64 s[14:15], s[6:7], 8 // (op 193) 
s_mov_b32 s18, s14 // (op 193) convert

s_cmp_lt_u32 s1, s18 // (op 193) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 // (op 193) Condition: Bottom (jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_193 if true)
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_193:























s_barrier  // (op 1361) 








v_cmp_eq_f32 vcc, 0.00000, v6 // (op 155) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155 // (op 155) Condition: False, jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v10, v12 // (op 50) 
v_mul_f32 v17, v10, v13 // (op 50) 
v_mul_f32 v18, v10, v14 // (op 50) 
v_mul_f32 v19, v10, v15 // (op 50) 

v_lshrrev_b32 v0, 6, v1 // (op 1044) 
v_and_b32 v3, 63, v1 // (op 1044) 
v_lshrrev_b32 v4, 4, v3 // (op 1044) 
v_lshlrev_b32 v5, 2, v4 // (op 1044) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1044) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1044) 

v_lshrrev_b32 v4, 2, v3 // (op 1044) 

v_and_b32 v3, 3, v4 // (op 1044) 

v_and_b32 v4, 3, v1 // (op 1044) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1044) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 1044) 

v_mul_lo_u32 v4, v3, s28 // (op 1044) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 1044) 


s_mul_i32 s1, s28, s33 // (op 1046) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1046) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1046) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1046) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1046) most significant: sum
s_add_u32 s1, s1, s3 // (op 1046) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 1046) 

s_mov_b32 s4, s30 // (op 1046) reg expression
s_mov_b32 s5, s31 // (op 1046) reg expression
s_mov_b32 s6, s2 // (op 1046) reg expression
s_mov_b32 s7, 131072 // (op 1046) call()




buffer_store_dwordx4 v[16:19], v3, s[4:7], 0 offen // (op 56) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155 // (op 155) Condition: Done, jump to RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_155:
v_lshrrev_b32 v0, 6, v1 // (op 814) 
v_and_b32 v3, 63, v1 // (op 814) 
v_lshrrev_b32 v4, 4, v3 // (op 814) 
v_lshlrev_b32 v5, 2, v4 // (op 814) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 814) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 814) 

v_lshrrev_b32 v4, 2, v3 // (op 814) 

v_and_b32 v3, 3, v4 // (op 814) 

v_and_b32 v4, 3, v1 // (op 814) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 814) 


v_lshl_add_u32 v3, s17, 4, v5 // (op 814) 

v_mul_lo_u32 v4, v3, s34 // (op 814) 

v_add_lshl_u32 v3, v0, v4, 2 // (op 814) 


s_mul_i32 s1, s34, s33 // (op 816) most significant: low of low * high
s_mul_i32 s2, s35, s32 // (op 816) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 816) least significant
s_mul_hi_u32 s3, s34, s32 // (op 816) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 816) most significant: sum
s_add_u32 s1, s1, s3 // (op 816) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 816) 

s_mov_b32 s8, s36 // (op 816) reg expression
s_mov_b32 s9, s37 // (op 816) reg expression
s_mov_b32 s10, s2 // (op 816) reg expression
s_mov_b32 s11, 131072 // (op 816) call()





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


v_lshrrev_b32 v0, 6, v1 // (op 1061) 
v_and_b32 v11, 63, v1 // (op 1061) 
v_lshrrev_b32 v3, 4, v11 // (op 1061) 
v_lshlrev_b32 v4, 2, v3 // (op 1061) 

v_lshl_add_u32 v3, v0, 4, v4 // (op 1061) 


v_lshl_add_u32 v0, s16, 4, v3 // (op 1061) 

v_lshrrev_b32 v3, 2, v11 // (op 1061) 

v_and_b32 v11, 3, v3 // (op 1061) 

v_and_b32 v3, 3, v1 // (op 1061) 
v_lshl_add_u32 v4, v11, 2, v3 // (op 1061) 


v_lshl_add_u32 v3, s17, 4, v4 // (op 1061) 

v_mul_lo_u32 v11, v3, s28 // (op 1061) 

v_add_lshl_u32 v3, v0, v11, 2 // (op 1061) 


s_mul_i32 s1, s28, s33 // (op 1063) most significant: low of low * high
s_mul_i32 s2, s29, s32 // (op 1063) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1063) least significant
s_mul_hi_u32 s3, s28, s32 // (op 1063) most significant: high of low * low
s_add_u32 s1, s1, s2 // (op 1063) most significant: sum
s_add_u32 s1, s1, s3 // (op 1063) most significant: sum


s_lshl_b64 s[2:3], s[0:1], 2 // (op 1063) 

s_mov_b32 s4, s30 // (op 1063) reg expression
s_mov_b32 s5, s31 // (op 1063) reg expression
s_mov_b32 s6, s2 // (op 1063) reg expression
s_mov_b32 s7, 131072 // (op 1063) call()




buffer_store_dwordx4 v[12:15], v3, s[4:7], 0 offen // (op 166) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_155:



s_endpgm  // End of RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 100
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
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
    DataType: F8
    DataTypeA: F8
    DataTypeB: F6
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
  - .name: RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NT_FP8_FP6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 95
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

