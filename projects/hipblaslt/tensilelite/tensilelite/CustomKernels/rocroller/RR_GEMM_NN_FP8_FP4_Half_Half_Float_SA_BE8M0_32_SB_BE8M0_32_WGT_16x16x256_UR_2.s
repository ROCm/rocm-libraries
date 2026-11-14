// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.p2align 8
.type RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2,@function
RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_exec_begin:


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


v_and_b32 v0, 48, v1 // (op 853) 
v_and_b32 v3, 63, v1 // (op 853) 
v_sub_u32 v4, v3, v0 // (op 853) 

v_lshrrev_b32 v3, 1, v4 // (op 853) 
v_add_lshl_u32 v5, v0, v3, 4 // (op 853) 


v_and_b32 v0, 1, v4 // (op 853) 

v_lshlrev_b32 v3, 3, v0 // (op 853) 

v_lshrrev_b32 v0, 6, v1 // (op 853) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 853) 


v_add_u32 v0, v5, v4 // (op 853) 


v_and_b32 v3, 15, v1 // (op 906) 
v_and_b32 v4, 63, v1 // (op 906) 
v_lshrrev_b32 v5, 4, v4 // (op 906) 

v_lshlrev_b32 v4, 5, v5 // (op 906) 

v_lshl_add_u32 v5, v3, 8, v4 // (op 906) 


v_lshlrev_b32 v3, 2, v5 // (op 906) 

v_lshrrev_b32 v4, 3, v3 // (op 906) 

v_lshrrev_b32 v3, 6, v1 // (op 959) 
v_and_b32 v5, 15, v1 // (op 959) 
v_lshl_add_u32 v6, v3, 4, v5 // (op 959) 


v_and_b32 v3, 48, v1 // (op 959) 
v_add_u32 v5, v3, v6 // (op 959) 


v_and_b32 v3, 15, v1 // (op 1012) 
v_and_b32 v6, 63, v1 // (op 1012) 
v_lshrrev_b32 v7, 4, v6 // (op 1012) 

v_lshl_add_u32 v6, v3, 3, v7 // (op 1012) 


v_and_b32 v3, 48, v1 // (op 1070) 
v_and_b32 v7, 63, v1 // (op 1070) 
v_sub_u32 v8, v7, v3 // (op 1070) 

v_lshrrev_b32 v7, 1, v8 // (op 1070) 
v_add_lshl_u32 v9, v3, v7, 4 // (op 1070) 


v_and_b32 v3, 1, v8 // (op 1070) 

v_lshlrev_b32 v7, 3, v3 // (op 1070) 

v_lshrrev_b32 v3, 6, v1 // (op 1070) 
v_lshl_add_u32 v8, v3, 4, v7 // (op 1070) 


v_add_u32 v3, v9, v8 // (op 1070) 


v_and_b32 v7, 15, v1 // (op 1094) 
v_and_b32 v8, 63, v1 // (op 1094) 
v_lshrrev_b32 v9, 4, v8 // (op 1094) 

v_lshlrev_b32 v8, 5, v9 // (op 1094) 

v_lshl_add_u32 v9, v7, 8, v8 // (op 1094) 


v_lshlrev_b32 v7, 2, v9 // (op 1094) 

v_lshrrev_b32 v8, 3, v7 // (op 1094) 

v_lshrrev_b32 v7, 6, v1 // (op 1118) 
v_and_b32 v9, 15, v1 // (op 1118) 
v_lshl_add_u32 v10, v7, 4, v9 // (op 1118) 


v_and_b32 v7, 48, v1 // (op 1118) 
v_add_u32 v9, v7, v10 // (op 1118) 


v_and_b32 v7, 15, v1 // (op 1142) 
v_and_b32 v10, 63, v1 // (op 1142) 
v_lshrrev_b32 v11, 4, v10 // (op 1142) 

v_lshl_add_u32 v10, v7, 3, v11 // (op 1142) 


v_mov_b32 v7, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_lshlrev_b32 v12, 2, v1 // (op 563) 
v_mul_lo_u32 v13, v12, s2 // (op 563) 

v_lshl_add_u32 v12, s16, 4, v13 // (op 563) 

s_mul_i32 s1, s2, s7 // (op 565) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 565) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 565) least significant
s_mul_hi_u32 s15, s2, s6 // (op 565) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 565) most significant: sum
s_add_u32 s1, s1, s15 // (op 565) most significant: sum


s_mov_b32 s40, s4 // (op 565) reg expression
s_mov_b32 s41, s5 // (op 565) reg expression
s_mov_b32 s42, s0 // (op 565) reg expression
s_mov_b32 s43, 131072 // (op 565) call()

v_lshlrev_b32 v13, 2, v1 // (op 582) 
v_mul_lo_u32 v14, v13, s2 // (op 582) 

v_lshl_add_u32 v13, s16, 4, v14 // (op 582) 

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

v_and_b32 v14, -8, v1 // (op 627) 
v_sub_u32 v15, v1, v14 // (op 627) 

v_lshlrev_b32 v14, 5, v15 // (op 627) 

v_mov_b32 v17, 0 // (op 627) convert
v_mov_b32 v16, v14 // (op 627) convert

v_lshrrev_b32 v14, 3, v1 // (op 627) 
v_lshlrev_b32 v15, 1, v14 // (op 627) 

v_lshl_add_u32 v14, s17, 4, v15 // (op 627) 

v_mul_lo_u32 v19, v14, s9 // (op 627) most significant: low of low * high
v_mul_hi_u32 v15, v14, s8 // (op 627) most significant: high of low * low
v_mul_lo_u32 v18, v14, s8 // (op 627) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 627) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 627) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 627) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 627) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 627) 

v_mov_b32 v14, v16 // (op 627) convert

s_mul_i32 s1, s8, s13 // (op 629) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 629) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 629) least significant
s_mul_hi_u32 s15, s8, s12 // (op 629) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 629) most significant: sum
s_add_u32 s1, s1, s15 // (op 629) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 629) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 629) 

s_mov_b32 s44, s10 // (op 629) reg expression
s_mov_b32 s45, s11 // (op 629) reg expression
s_mov_b32 s46, s0 // (op 629) reg expression
s_mov_b32 s47, 131072 // (op 629) call()

v_and_b32 v15, -8, v1 // (op 646) 
v_sub_u32 v16, v1, v15 // (op 646) 

v_lshlrev_b32 v15, 5, v16 // (op 646) 

v_mov_b32 v17, 0 // (op 646) convert
v_mov_b32 v16, v15 // (op 646) convert

v_lshrrev_b32 v15, 3, v1 // (op 646) 
v_lshlrev_b32 v18, 1, v15 // (op 646) 

v_lshl_add_u32 v15, s17, 4, v18 // (op 646) 

v_mul_lo_u32 v19, v15, s9 // (op 646) most significant: low of low * high
v_mul_hi_u32 v20, v15, s8 // (op 646) most significant: high of low * low
v_mul_lo_u32 v18, v15, s8 // (op 646) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 646) most significant: sum


v_add_co_u32 v20, s[0:1], v16, v18 // (op 646) least significant half
v_addc_co_u32 v21, s[0:1], v17, v19, s[0:1] // (op 646) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 646) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 646) 

v_mov_b32 v15, v18 // (op 646) convert

s_mul_i32 s1, s8, s13 // (op 648) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 648) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 648) least significant
s_mul_hi_u32 s15, s8, s12 // (op 648) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 648) most significant: sum
s_add_u32 s1, s1, s15 // (op 648) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 648) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 648) 

s_mov_b32 s44, s10 // (op 648) reg expression
s_mov_b32 s45, s11 // (op 648) reg expression
s_mov_b32 s46, s0 // (op 648) reg expression
s_mov_b32 s47, 131072 // (op 648) call()

v_and_b32 v16, -16, v1 // (op 691) 
v_sub_u32 v17, v1, v16 // (op 691) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 691) 

v_lshrrev_b32 v17, 4, v1 // (op 691) 
v_lshlrev_b32 v18, 1, v17 // (op 691) 


s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v17, v18, s20 // (op 691) 

v_add_u32 v18, v16, v17 // (op 691) 


s_mul_i32 s1, s20, s25 // (op 693) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 693) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 693) least significant
s_mul_hi_u32 s15, s20, s24 // (op 693) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 693) most significant: sum
s_add_u32 s1, s1, s15 // (op 693) most significant: sum


s_mov_b32 s48, s22 // (op 693) reg expression
s_mov_b32 s49, s23 // (op 693) reg expression
s_mov_b32 s50, s0 // (op 693) reg expression
s_mov_b32 s51, 131072 // (op 693) call()

v_and_b32 v16, -16, v1 // (op 710) 
v_sub_u32 v17, v1, v16 // (op 710) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 710) 

v_lshrrev_b32 v17, 4, v1 // (op 710) 
v_lshlrev_b32 v19, 1, v17 // (op 710) 

v_mul_lo_u32 v17, v19, s20 // (op 710) 

v_add_u32 v19, v16, v17 // (op 710) 


s_mul_i32 s1, s20, s25 // (op 712) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 712) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 712) least significant
s_mul_hi_u32 s15, s20, s24 // (op 712) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 712) most significant: sum
s_add_u32 s1, s1, s15 // (op 712) most significant: sum


s_mov_b32 s48, s22 // (op 712) reg expression
s_mov_b32 s49, s23 // (op 712) reg expression
s_mov_b32 s50, s0 // (op 712) reg expression
s_mov_b32 s51, 131072 // (op 712) call()

v_and_b32 v16, -8, v1 // (op 755) 
v_sub_u32 v17, v1, v16 // (op 755) 

v_lshrrev_b32 v16, 3, v1 // (op 755) 
v_lshlrev_b32 v20, 1, v16 // (op 755) 

v_lshl_add_u32 v16, s17, 4, v20 // (op 755) 

v_mul_lo_u32 v20, v16, s26 // (op 755) 

v_add_u32 v16, v17, v20 // (op 755) 


s_mul_i32 s1, s26, s13 // (op 757) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 757) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 757) least significant
s_mul_hi_u32 s15, s26, s12 // (op 757) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 757) most significant: sum
s_add_u32 s1, s1, s15 // (op 757) most significant: sum


s_mov_b32 s52, s28 // (op 757) reg expression
s_mov_b32 s53, s29 // (op 757) reg expression
s_mov_b32 s54, s0 // (op 757) reg expression
s_mov_b32 s55, 131072 // (op 757) call()

v_and_b32 v17, -8, v1 // (op 774) 
v_sub_u32 v20, v1, v17 // (op 774) 

v_lshrrev_b32 v17, 3, v1 // (op 774) 
v_lshlrev_b32 v21, 1, v17 // (op 774) 

v_lshl_add_u32 v17, s17, 4, v21 // (op 774) 

v_mul_lo_u32 v21, v17, s26 // (op 774) 

v_add_u32 v17, v20, v21 // (op 774) 


s_mul_i32 s1, s26, s13 // (op 776) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 776) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 776) least significant
s_mul_hi_u32 s15, s26, s12 // (op 776) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 776) most significant: sum
s_add_u32 s1, s1, s15 // (op 776) most significant: sum


s_mov_b32 s52, s28 // (op 776) reg expression
s_mov_b32 s53, s29 // (op 776) reg expression
s_mov_b32 s54, s0 // (op 776) reg expression
s_mov_b32 s55, 131072 // (op 776) call()


s_lshl_b64 s[0:1], s[8:9], 2 // (op 428) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 428) 


v_mov_b32 v22, v14 // (op 428) 
buffer_load_dwordx4 v[24:27], v22, s[44:47], 0 offen // (op 428) Load value
v_add_u32 v22, v22, v20 // (op 428) 
buffer_load_dwordx4 v[28:31], v22, s[44:47], 0 offen // (op 428) Load value



v_mov_b64 v[20:21], s[26:27] // (op 431) convert

v_mov_b32 v22, v16 // (op 431) 
buffer_load_ubyte v32, v22, s[52:55], 0 offen offset: 0 // (op 431) Load value
v_add_u32 v22, v22, v20 // (op 431) 
buffer_load_ubyte v33, v22, s[52:55], 0 offen offset: 0 // (op 431) Load value



v_mov_b64 v[20:21], s[2:3] // (op 434) convert

v_mov_b32 v22, v12 // (op 434) reg expression
buffer_load_dwordx4 v[34:37], v22, s[40:43], 0 offen // (op 434) Load value
v_add_u32 v22, v22, v20 // (op 434) 
buffer_load_dwordx4 v[38:41], v22, s[40:43], 0 offen // (op 434) Load value
v_add_u32 v22, v22, v20 // (op 434) 
buffer_load_dwordx4 v[42:45], v22, s[40:43], 0 offen // (op 434) Load value
v_add_u32 v22, v22, v20 // (op 434) 
buffer_load_dwordx4 v[46:49], v22, s[40:43], 0 offen // (op 434) Load value



v_mov_b64 v[20:21], s[20:21] // (op 437) convert

v_mov_b32 v22, v18 // (op 437) reg expression
buffer_load_ubyte v50, v22, s[48:51], 0 offen offset: 0 // (op 437) Load value
v_add_u32 v22, v22, v20 // (op 437) 
buffer_load_ubyte v51, v22, s[48:51], 0 offen offset: 0 // (op 437) Load value



s_lshl_b64 s[0:1], s[8:9], 2 // (op 440) 
v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 440) 


v_mov_b32 v22, v14 // (op 440) 
buffer_load_dwordx4 v[52:55], v22, s[44:47], 0 offen offset:128 // (op 440) Load value
v_add_u32 v22, v22, v20 // (op 440) 
buffer_load_dwordx4 v[56:59], v22, s[44:47], 0 offen offset:128 // (op 440) Load value




v_mov_b64 v[20:21], s[26:27] // (op 443) convert

v_mov_b32 v14, v16 // (op 443) 
buffer_load_ubyte v22, v14, s[52:55], 0 offen offset:8 // (op 443) Load value
v_add_u32 v14, v14, v20 // (op 443) 
buffer_load_ubyte v23, v14, s[52:55], 0 offen offset:8 // (op 443) Load value




v_mov_b64 v[20:21], s[2:3] // (op 446) convert

s_lshl_b32 s0, s2, 8 // (op 446) 
v_add_u32 v14, v12, s0 // (op 446) 

buffer_load_dwordx4 v[60:63], v14, s[40:43], 0 offen // (op 446) Load value
v_add_u32 v14, v14, v20 // (op 446) 
buffer_load_dwordx4 v[64:67], v14, s[40:43], 0 offen // (op 446) Load value
v_add_u32 v14, v14, v20 // (op 446) 
buffer_load_dwordx4 v[68:71], v14, s[40:43], 0 offen // (op 446) Load value
v_add_u32 v14, v14, v20 // (op 446) 
buffer_load_dwordx4 v[72:75], v14, s[40:43], 0 offen // (op 446) Load value




v_mov_b64 v[20:21], s[20:21] // (op 449) convert

s_lshl_b32 s0, s20, 3 // (op 449) 
v_add_u32 v12, v18, s0 // (op 449) 

buffer_load_ubyte v76, v12, s[48:51], 0 offen offset: 0 // (op 449) Load value
v_add_u32 v12, v12, v20 // (op 449) 
buffer_load_ubyte v77, v12, s[48:51], 0 offen offset: 0 // (op 449) Load value



v_and_b32 v12, -8, v1 // (op 894) 
v_sub_u32 v14, v1, v12 // (op 894) 

v_lshlrev_b32 v12, 5, v14 // (op 894) 

v_lshrrev_b32 v14, 3, v1 // (op 894) 
v_lshl_add_u32 v16, v14, 9, v12 // (op 894) 


v_lshlrev_b32 v12, 2, v16 // (op 894) 

v_lshrrev_b32 v14, 3, v12 // (op 894) 




s_waitcnt vmcnt(19)
ds_write_b128 v14, v[24:27] // (op 451) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v14, v[28:31] offset:128 // (op 451) Store local data 

v_and_b32 v12, -8, v1 // (op 1000) 
v_sub_u32 v14, v1, v12 // (op 1000) 

v_lshrrev_b32 v12, 3, v1 // (op 1000) 
v_lshl_add_u32 v16, v12, 4, v14 // (op 1000) 





s_waitcnt vmcnt(17)
ds_write_b8 v16, v32 offset:2048 // (op 452) Store local data 
s_waitcnt vmcnt(16)
ds_write_b8 v16, v33 offset:2056 // (op 452) Store local data 

v_lshlrev_b32 v12, 6, v1 // (op 841) 



s_waitcnt vmcnt(15)
ds_write_b128 v12, v[34:37] offset:2176 // (op 453) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v12, v[38:41] offset:2192 // (op 453) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v12, v[42:45] offset:2208 // (op 453) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v12, v[46:49] offset:2224 // (op 453) Store local data 

v_and_b32 v12, -16, v1 // (op 947) 
v_sub_u32 v14, v1, v12 // (op 947) 

v_lshrrev_b32 v12, 4, v1 // (op 947) 
v_lshl_add_u32 v16, v12, 5, v14 // (op 947) 





s_waitcnt vmcnt(11)
ds_write_b8 v16, v50 offset:6272 // (op 454) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v16, v51 offset:6288 // (op 454) Store local data 

s_waitcnt lgkmcnt(8)
s_barrier  // (op 1198) 



ds_read_b128 v[78:81], v4 // (op 471) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1239) 



ds_read_u8 v12, v6 offset:2048 // (op 477) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1179) 



ds_read_b64_tr_b8 v[82:83], v0 offset:2176 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v0 offset:2304 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:3200 // (op 468) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:3328 // (op 468) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1219) 


ds_read_u8 v14, v5 offset:6272 // (op 474) Load local data 
v_lshlrev_b32 v16, 6, v1 // (op 812) 
v_and_b32 v18, -8, v1 // (op 865) 
v_sub_u32 v20, v1, v18 // (op 865) 

v_lshlrev_b32 v18, 5, v20 // (op 865) 

v_lshrrev_b32 v20, 3, v1 // (op 865) 
v_lshl_add_u32 v21, v20, 9, v18 // (op 865) 


v_lshlrev_b32 v18, 2, v21 // (op 865) 

v_lshrrev_b32 v20, 3, v18 // (op 865) 

v_and_b32 v18, -16, v1 // (op 918) 
v_sub_u32 v21, v1, v18 // (op 918) 

v_lshrrev_b32 v18, 4, v1 // (op 918) 
v_lshl_add_u32 v90, v18, 5, v21 // (op 918) 


v_and_b32 v18, -8, v1 // (op 971) 
v_sub_u32 v21, v1, v18 // (op 971) 

v_lshrrev_b32 v18, 3, v1 // (op 971) 
v_lshl_add_u32 v91, v18, 4, v21 // (op 971) 


v_lshlrev_b32 v18, 6, v1 // (op 1058) 
v_and_b32 v21, -8, v1 // (op 1082) 
v_sub_u32 v92, v1, v21 // (op 1082) 

v_lshlrev_b32 v21, 5, v92 // (op 1082) 

v_lshrrev_b32 v92, 3, v1 // (op 1082) 
v_lshl_add_u32 v93, v92, 9, v21 // (op 1082) 


v_lshlrev_b32 v21, 2, v93 // (op 1082) 

v_lshrrev_b32 v92, 3, v21 // (op 1082) 

v_and_b32 v21, -16, v1 // (op 1106) 
v_sub_u32 v93, v1, v21 // (op 1106) 

v_lshrrev_b32 v21, 4, v1 // (op 1106) 
v_lshl_add_u32 v94, v21, 5, v93 // (op 1106) 


v_and_b32 v21, -8, v1 // (op 1130) 
v_sub_u32 v93, v1, v21 // (op 1130) 

v_lshrrev_b32 v21, 3, v1 // (op 1130) 
v_lshl_add_u32 v95, v21, 4, v93 // (op 1130) 


s_mov_b32 s0, 0 // (op 61) call()
s_ashr_i64 s[14:15], s[6:7], 8 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60:

s_add_u32 s0, s0, 2 // (op 62) 
s_lshl_b64 s[14:15], s[8:9], 2 // (op 10) 
v_lshrrev_b64 v[96:97], 3, s[14:15] // (op 10) 


v_mov_b32 v21, v15 // (op 10) 
v_mov_b32 v93, v21 // (op 10) 
buffer_load_dwordx4 v[24:27], v93, s[44:47], 0 offen offset:256 // (op 10) Load value
v_add_u32 v93, v93, v96 // (op 10) 
buffer_load_dwordx4 v[28:31], v93, s[44:47], 0 offen offset:256 // (op 10) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[78:81], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 90) 

s_barrier  // (op 1188) 
v_mov_b64 v[96:97], s[26:27] // (op 25) convert




ds_read_b64_tr_b8 v[82:83], v0 offset:4224 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v0 offset:4352 // (op 93) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:5248 // (op 93) Transpose load local data 
v_mov_b32 v93, v17 // (op 25) 
v_mov_b32 v98, v93 // (op 25) 
buffer_load_ubyte v32, v98, s[52:55], 0 offen offset:16 // (op 25) Load value
v_add_u32 v98, v98, v96 // (op 25) 
ds_read_b64_tr_b8 v[88:89], v0 offset:5376 // (op 93) Transpose load local data 
buffer_load_ubyte v33, v98, s[52:55], 0 offen offset:16 // (op 25) Load value



s_barrier  // (op 1208) 
v_mov_b64 v[96:97], s[2:3] // (op 4) convert




ds_read_b128 v[78:81], v4 offset:64 // (op 97) Load local data 
v_mov_b32 v98, v13 // (op 4) 
s_barrier  // (op 1229) 
s_lshl_b32 s1, s2, 9 // (op 4) 


ds_read_u8 v14, v5 offset:6336 // (op 101) Load local data 
v_add_u32 v99, v98, s1 // (op 4) 

s_barrier  // (op 1249) 
buffer_load_dwordx4 v[34:37], v99, s[40:43], 0 offen // (op 4) Load value



ds_read_u8 v12, v6 offset:2052 // (op 105) Load local data 
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx4 v[38:41], v99, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[78:81], a[0:3], v14, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 107) 
buffer_load_dwordx4 v[42:45], v99, s[40:43], 0 offen // (op 4) Load value
v_add_u32 v99, v99, v96 // (op 4) 
buffer_load_dwordx4 v[46:49], v99, s[40:43], 0 offen // (op 4) Load value



v_mov_b64 v[96:97], s[20:21] // (op 17) convert

v_mov_b32 v99, v19 // (op 17) 
s_lshl_b32 s1, s20, 4 // (op 17) 
v_add_u32 v100, v99, s1 // (op 17) 

buffer_load_ubyte v50, v100, s[48:51], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v100, v100, v96 // (op 17) 
buffer_load_ubyte v51, v100, s[48:51], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 1269) 



s_waitcnt vmcnt(19)
ds_write_b128 v92, v[52:55] offset:6400 // (op 306) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v92, v[56:59] offset:6528 // (op 306) Store local data 
s_barrier  // (op 1281) 



s_waitcnt vmcnt(17)
ds_write_b8 v95, v22 offset:8448 // (op 308) Store local data 
s_waitcnt vmcnt(16)
ds_write_b8 v95, v23 offset:8456 // (op 308) Store local data 
s_barrier  // (op 1262) 



s_waitcnt vmcnt(15)
ds_write_b128 v18, v[60:63] offset:8576 // (op 287) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v18, v[64:67] offset:8592 // (op 287) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[68:71] offset:8608 // (op 287) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v18, v[72:75] offset:8624 // (op 287) Store local data 
s_barrier  // (op 1275) 



s_waitcnt vmcnt(11)
ds_write_b8 v94, v76 offset:12672 // (op 307) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v94, v77 offset:12688 // (op 307) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1265) 



ds_read_b128 v[100:103], v8 offset:6400 // (op 291) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1278) 



ds_read_u8 v96, v10 offset:8448 // (op 295) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1259) 



ds_read_b64_tr_b8 v[104:105], v3 offset:8576 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v3 offset:8704 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[108:109], v3 offset:9600 // (op 289) Transpose load local data 
ds_read_b64_tr_b8 v[110:111], v3 offset:9728 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1272) 


ds_read_u8 v97, v9 offset:12672 // (op 293) Load local data 

s_lshl_b64 s[14:15], s[8:9], 2 // (op 309) 
v_lshrrev_b64 v[112:113], 3, s[14:15] // (op 309) 


v_mov_b32 v114, v21 // (op 309) 
buffer_load_dwordx4 v[52:55], v114, s[44:47], 0 offen offset:384 // (op 309) Load value
v_add_u32 v114, v114, v112 // (op 309) 
buffer_load_dwordx4 v[56:59], v114, s[44:47], 0 offen offset:384 // (op 309) Load value


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:111], v[100:103], a[0:3], v97, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 296) 









v_add_u32 v15, 256, v15 // (op 638) 
v_mov_b64 v[96:97], s[26:27] // (op 311) convert

v_mov_b32 v21, v93 // (op 311) 
buffer_load_ubyte v22, v21, s[52:55], 0 offen offset:24 // (op 311) Load value
v_add_u32 v21, v21, v96 // (op 311) 
ds_read_b64_tr_b8 v[100:101], v3 offset:10624 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[102:103], v3 offset:10752 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[104:105], v3 offset:11648 // (op 298) Transpose load local data 
ds_read_b64_tr_b8 v[106:107], v3 offset:11776 // (op 298) Transpose load local data 
buffer_load_ubyte v23, v21, s[52:55], 0 offen offset:24 // (op 311) Load value







s_barrier  // (op 1344) 
v_add_u32 v17, 16, v17 // (op 766) 
v_mov_b64 v[96:97], s[2:3] // (op 286) convert

s_lshl_b32 s1, s2, 8 // (op 286) 
s_mul_i32 s14, 3, s1 // (op 286) 

v_add_u32 v21, v98, s14 // (op 286) 

buffer_load_dwordx4 v[60:63], v21, s[40:43], 0 offen // (op 286) Load value
v_add_u32 v21, v21, v96 // (op 286) 
buffer_load_dwordx4 v[64:67], v21, s[40:43], 0 offen // (op 286) Load value
v_add_u32 v21, v21, v96 // (op 286) 
ds_read_b128 v[108:111], v8 offset:6464 // (op 300) Load local data 
buffer_load_dwordx4 v[68:71], v21, s[40:43], 0 offen // (op 286) Load value
v_add_u32 v21, v21, v96 // (op 286) 


s_barrier  // (op 1348) 
ds_read_u8 v93, v9 offset:12736 // (op 302) Load local data 
buffer_load_dwordx4 v[72:75], v21, s[40:43], 0 offen // (op 286) Load value







s_barrier  // (op 1352) 
ds_read_u8 v21, v10 offset:8452 // (op 304) Load local data 
s_lshl_b32 s1, s2, 9 // (op 574) 
v_add_u32 v13, v13, s1 // (op 574) 

s_barrier  // (op 1356) 
v_mov_b64 v[96:97], s[20:21] // (op 310) convert

s_lshl_b32 s1, s20, 3 // (op 310) 
s_mul_i32 s14, 3, s1 // (op 310) 

v_add_u32 v98, v99, s14 // (op 310) 

buffer_load_ubyte v76, v98, s[48:51], 0 offen offset: 0 // (op 310) Load value
v_add_u32 v98, v98, v96 // (op 310) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:107], v[108:111], a[0:3], v93, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 305) 
buffer_load_ubyte v77, v98, s[48:51], 0 offen offset: 0 // (op 310) Load value







s_barrier  // (op 1201) 
s_lshl_b32 s1, s20, 4 // (op 702) 



v_add_u32 v19, v19, s1 // (op 702) 

s_waitcnt vmcnt(19)
ds_write_b128 v20, v[24:27] // (op 13) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v20, v[28:31] offset:128 // (op 13) Store local data 
s_barrier  // (op 1242) 



s_waitcnt vmcnt(17)
ds_write_b8 v91, v32 offset:2048 // (op 30) Store local data 
s_waitcnt vmcnt(16)
ds_write_b8 v91, v33 offset:2056 // (op 30) Store local data 
s_barrier  // (op 1182) 



s_waitcnt vmcnt(15)
ds_write_b128 v16, v[34:37] offset:2176 // (op 6) Store local data 
s_waitcnt vmcnt(14)
ds_write_b128 v16, v[38:41] offset:2192 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b128 v16, v[42:45] offset:2208 // (op 6) Store local data 
s_waitcnt vmcnt(12)
ds_write_b128 v16, v[46:49] offset:2224 // (op 6) Store local data 
s_barrier  // (op 1222) 



s_waitcnt vmcnt(11)
ds_write_b8 v90, v50 offset:6272 // (op 21) Store local data 
s_waitcnt vmcnt(10)
ds_write_b8 v90, v51 offset:6288 // (op 21) Store local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 1211) 



ds_read_b128 v[78:81], v4 // (op 77) Load local data 
s_waitcnt lgkmcnt(7)
s_barrier  // (op 1252) 



ds_read_u8 v12, v6 offset:2048 // (op 87) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1191) 



ds_read_b64_tr_b8 v[82:83], v0 offset:2176 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[84:85], v0 offset:2304 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[86:87], v0 offset:3200 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[88:89], v0 offset:3328 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 1232) 


ds_read_u8 v14, v5 offset:6272 // (op 82) Load local data 
s_ashr_i64 s[14:15], s[6:7], 8 // (op 60) 
s_mov_b32 s1, s14 // (op 60) convert

s_and_b32 s14, s1, 4294967294 // (op 60) 

s_cmp_lt_u32 s0, s14 // (op 60) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoop_60:

















v_lshlrev_b32 v13, 2, v1 // (op 537) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 537) 
s_lshr_b32 s14, s0, 1 // (op 537) 

v_lshl_add_u32 v15, s14, 9, v13 // (op 537) 


v_mul_lo_u32 v13, v15, s2 // (op 537) 

v_lshl_add_u32 v15, s16, 4, v13 // (op 537) 

s_mul_i32 s1, s2, s7 // (op 539) most significant: low of low * high
s_mul_i32 s14, s3, s6 // (op 539) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 539) least significant
s_mul_hi_u32 s15, s2, s6 // (op 539) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 539) most significant: sum
s_add_u32 s1, s1, s15 // (op 539) most significant: sum


s_mov_b32 s40, s4 // (op 539) reg expression
s_mov_b32 s41, s5 // (op 539) reg expression
s_mov_b32 s42, s0 // (op 539) reg expression
s_mov_b32 s43, 131072 // (op 539) call()

v_and_b32 v13, -8, v1 // (op 604) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 604) 
v_sub_u32 v16, v1, v13 // (op 604) 

s_lshr_b32 s14, s0, 1 // (op 604) 

v_lshlrev_b32 v13, 5, v16 // (op 604) 

v_lshl_add_u32 v16, s14, 9, v13 // (op 604) 


v_mov_b32 v21, 0 // (op 604) convert
v_mov_b32 v20, v16 // (op 604) convert

v_lshrrev_b32 v13, 3, v1 // (op 604) 
v_lshlrev_b32 v16, 1, v13 // (op 604) 

v_lshl_add_u32 v13, s17, 4, v16 // (op 604) 

v_mul_lo_u32 v17, v13, s9 // (op 604) most significant: low of low * high
v_mul_hi_u32 v18, v13, s8 // (op 604) most significant: high of low * low
v_mul_lo_u32 v16, v13, s8 // (op 604) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 604) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v16 // (op 604) least significant half
v_addc_co_u32 v19, s[0:1], v21, v17, s[0:1] // (op 604) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 604) 


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 604) 

v_mov_b32 v13, v16 // (op 604) convert

s_mul_i32 s1, s8, s13 // (op 606) most significant: low of low * high
s_mul_i32 s14, s9, s12 // (op 606) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 606) least significant
s_mul_hi_u32 s15, s8, s12 // (op 606) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 606) most significant: sum
s_add_u32 s1, s1, s15 // (op 606) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 606) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 606) 

s_mov_b32 s44, s10 // (op 606) reg expression
s_mov_b32 s45, s11 // (op 606) reg expression
s_mov_b32 s46, s0 // (op 606) reg expression
s_mov_b32 s47, 131072 // (op 606) call()

v_and_b32 v16, -16, v1 // (op 668) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 668) 
v_sub_u32 v17, v1, v16 // (op 668) 

s_lshr_b32 s14, s0, 1 // (op 668) 

v_lshl_add_u32 v16, s16, 4, v17 // (op 668) 

v_lshrrev_b32 v17, 4, v1 // (op 668) 
v_lshlrev_b32 v18, 1, v17 // (op 668) 

v_lshl_add_u32 v17, s14, 4, v18 // (op 668) 


v_mul_lo_u32 v18, v17, s20 // (op 668) 

v_add_u32 v17, v16, v18 // (op 668) 


s_mul_i32 s1, s20, s25 // (op 670) most significant: low of low * high
s_mul_i32 s14, s21, s24 // (op 670) most significant: low of high * low
s_mul_i32 s0, s20, s24 // (op 670) least significant
s_mul_hi_u32 s15, s20, s24 // (op 670) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 670) most significant: sum
s_add_u32 s1, s1, s15 // (op 670) most significant: sum


s_mov_b32 s48, s22 // (op 670) reg expression
s_mov_b32 s49, s23 // (op 670) reg expression
s_mov_b32 s50, s0 // (op 670) reg expression
s_mov_b32 s51, 131072 // (op 670) call()

v_and_b32 v16, -8, v1 // (op 732) 
s_ashr_i64 s[0:1], s[6:7], 8 // (op 732) 
v_sub_u32 v18, v1, v16 // (op 732) 

s_lshr_b32 s14, s0, 1 // (op 732) 

v_lshrrev_b32 v16, 3, v1 // (op 732) 
v_lshl_add_u32 v19, s14, 4, v18 // (op 732) 


v_lshlrev_b32 v18, 1, v16 // (op 732) 

v_lshl_add_u32 v16, s17, 4, v18 // (op 732) 

v_mul_lo_u32 v18, v16, s26 // (op 732) 

v_add_u32 v16, v19, v18 // (op 732) 


s_mul_i32 s1, s26, s13 // (op 734) most significant: low of low * high
s_mul_i32 s14, s27, s12 // (op 734) most significant: low of high * low
s_mul_i32 s0, s26, s12 // (op 734) least significant
s_mul_hi_u32 s15, s26, s12 // (op 734) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 734) most significant: sum
s_add_u32 s1, s1, s15 // (op 734) most significant: sum


s_mov_b32 s52, s28 // (op 734) reg expression
s_mov_b32 s53, s29 // (op 734) reg expression
s_mov_b32 s54, s0 // (op 734) reg expression
s_mov_b32 s55, 131072 // (op 734) call()

v_lshlrev_b32 v18, 6, v1 // (op 825) 
v_and_b32 v19, -8, v1 // (op 878) 
v_sub_u32 v20, v1, v19 // (op 878) 

v_lshlrev_b32 v19, 5, v20 // (op 878) 

v_lshrrev_b32 v20, 3, v1 // (op 878) 
v_lshl_add_u32 v21, v20, 9, v19 // (op 878) 


v_lshlrev_b32 v19, 2, v21 // (op 878) 

v_lshrrev_b32 v20, 3, v19 // (op 878) 

v_and_b32 v19, -16, v1 // (op 931) 
v_sub_u32 v21, v1, v19 // (op 931) 

v_lshrrev_b32 v19, 4, v1 // (op 931) 
v_lshl_add_u32 v24, v19, 5, v21 // (op 931) 


v_and_b32 v19, -8, v1 // (op 984) 
v_sub_u32 v21, v1, v19 // (op 984) 

v_lshrrev_b32 v19, 3, v1 // (op 984) 
v_lshl_add_u32 v25, v19, 4, v21 // (op 984) 


s_ashr_i64 s[0:1], s[6:7], 8 // (op 198) 
s_and_b32 s14, s0, 4294967294 // (op 198) 

s_ashr_i64 s[0:1], s[6:7], 8 // (op 197) 
s_mov_b32 s15, s0 // (op 197) convert

s_cmp_lt_u32 s14, s15 // (op 197) 









s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 // (op 197) Condition: Top (jump to RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197:

s_add_u32 s14, s14, 1 // (op 199) 
v_mov_b64 v[22:23], s[2:3] // (op 202) convert

v_mov_b32 v12, v15 // (op 202) 
v_mov_b32 v14, v12 // (op 202) reg expression
buffer_load_dwordx4 v[26:29], v14, s[40:43], 0 offen // (op 202) Load value
v_add_u32 v14, v14, v22 // (op 202) 
buffer_load_dwordx4 v[30:33], v14, s[40:43], 0 offen // (op 202) Load value
v_add_u32 v14, v14, v22 // (op 202) 
buffer_load_dwordx4 v[34:37], v14, s[40:43], 0 offen // (op 202) Load value
v_add_u32 v14, v14, v22 // (op 202) 
buffer_load_dwordx4 v[38:41], v14, s[40:43], 0 offen // (op 202) Load value



s_lshl_b32 s15, s2, 8 // (op 527) 
v_add_u32 v15, v15, s15 // (op 527) 


s_barrier  // (op 1185) 
s_lshl_b64 s[0:1], s[8:9], 2 // (op 225) 



v_lshrrev_b64 v[22:23], 3, s[0:1] // (op 225) 


v_mov_b32 v12, v13 // (op 225) 
v_mov_b32 v14, v12 // (op 225) 
buffer_load_dwordx4 v[42:45], v14, s[44:47], 0 offen // (op 225) Load value
v_add_u32 v14, v14, v22 // (op 225) 
buffer_load_dwordx4 v[46:49], v14, s[44:47], 0 offen // (op 225) Load value



v_add_u32 v13, 128, v13 // (op 596) 
s_waitcnt vmcnt(5)
ds_write_b128 v18, v[26:29] offset:2176 // (op 203) Store local data 

v_mov_b64 v[22:23], s[20:21] // (op 226) convert

v_mov_b32 v12, v17 // (op 226) 
v_mov_b32 v14, v12 // (op 226) reg expression
s_waitcnt vmcnt(4)
ds_write_b128 v18, v[30:33] offset:2192 // (op 203) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v18, v[34:37] offset:2208 // (op 203) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v18, v[38:41] offset:2224 // (op 203) Store local data 
buffer_load_ubyte v50, v14, s[48:51], 0 offen offset: 0 // (op 226) Load value

v_add_u32 v14, v14, v22 // (op 226) 
s_barrier  // (op 1204) 
buffer_load_ubyte v51, v14, s[48:51], 0 offen offset: 0 // (op 226) Load value






s_lshl_b32 s15, s20, 3 // (op 660) 
v_add_u32 v17, v17, s15 // (op 660) 

s_waitcnt vmcnt(3)
ds_write_b128 v20, v[42:45] // (op 222) Store local data 

v_mov_b64 v[22:23], s[26:27] // (op 227) convert

v_mov_b32 v12, v16 // (op 227) 
v_mov_b32 v14, v12 // (op 227) 
buffer_load_ubyte v26, v14, s[52:55], 0 offen offset: 0 // (op 227) Load value
v_add_u32 v14, v14, v22 // (op 227) 
s_waitcnt vmcnt(3)
ds_write_b128 v20, v[46:49] offset:128 // (op 222) Store local data 
buffer_load_ubyte v27, v14, s[52:55], 0 offen offset: 0 // (op 227) Load value




v_add_u32 v16, 8, v16 // (op 724) 
s_barrier  // (op 1225) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1194) 






ds_read_b64_tr_b8 v[28:29], v0 offset:2176 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:2304 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:3200 // (op 205) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v0 offset:3328 // (op 205) Transpose load local data 
s_waitcnt vmcnt(3)
ds_write_b8 v24, v50 offset:6272 // (op 223) Store local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1215) 
s_waitcnt vmcnt(2)
ds_write_b8 v24, v51 offset:6288 // (op 223) Store local data 




ds_read_b128 v[36:39], v4 // (op 207) Load local data 
s_barrier  // (op 1245) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1235) 





ds_read_u8 v12, v5 offset:6272 // (op 209) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v25, v26 offset:2048 // (op 224) Store local data 
s_waitcnt vmcnt(0)
ds_write_b8 v25, v27 offset:2056 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 1255) 



ds_read_u8 v14, v6 offset:2048 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[28:35], v[36:39], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 212) 







ds_read_b64_tr_b8 v[26:27], v0 offset:4224 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[28:29], v0 offset:4352 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v0 offset:5248 // (op 214) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v0 offset:5376 // (op 214) Transpose load local data 



ds_read_b128 v[34:37], v4 offset:64 // (op 216) Load local data 


ds_read_u8 v12, v5 offset:6336 // (op 218) Load local data 



ds_read_u8 v14, v6 offset:2052 // (op 220) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:37], a[0:3], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 221) 




s_ashr_i64 s[0:1], s[6:7], 8 // (op 197) 
s_mov_b32 s15, s0 // (op 197) convert

s_cmp_lt_u32 s14, s15 // (op 197) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 // (op 197) Condition: Bottom (jump to RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopTop_KLoopTail_197 if true)
RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ForLoopBottom_KLoopTail_197:























s_barrier  // (op 1336) 








v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157
s_nop 0
v_accvgpr_read v12, a0 // (op 50) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_mul_f32 v16, v11, v12 // (op 50) 
v_mul_f32 v17, v11, v13 // (op 50) 
v_mul_f32 v18, v11, v14 // (op 50) 
v_mul_f32 v19, v11, v15 // (op 50) 

v_cvt_f16_f32 v12, v16 // (op 56) 
v_cvt_f16_f32 v13, v17 // (op 56) 
v_cvt_f16_f32 v14, v18 // (op 56) 
v_cvt_f16_f32 v15, v19 // (op 56) 

v_lshrrev_b32 v0, 6, v1 // (op 1026) 
v_and_b32 v3, 63, v1 // (op 1026) 
v_lshrrev_b32 v4, 4, v3 // (op 1026) 
v_lshlrev_b32 v5, 2, v4 // (op 1026) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1026) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1026) 

v_lshrrev_b32 v4, 2, v3 // (op 1026) 

v_and_b32 v3, 3, v4 // (op 1026) 

v_and_b32 v4, 3, v1 // (op 1026) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1026) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1026) 

v_mul_lo_u32 v3, v6, s30 // (op 1026) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 1026) 


s_mul_i32 s15, s30, s13 // (op 1028) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1028) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1028) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1028) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1028) most significant: sum
s_add_u32 s15, s15, s1 // (op 1028) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 1028) 

s_mov_b32 s8, s32 // (op 1028) reg expression
s_mov_b32 s9, s33 // (op 1028) reg expression
s_mov_b32 s10, s0 // (op 1028) reg expression
s_mov_b32 s11, 131072 // (op 1028) call()


v_pack_b32_f16 v8, v12, v13 // (op 58) 
v_pack_b32_f16 v9, v14, v15 // (op 58) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 58) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157
RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalFalse_0_beta_157:
v_lshrrev_b32 v0, 6, v1 // (op 795) 
v_and_b32 v3, 63, v1 // (op 795) 
v_lshrrev_b32 v4, 4, v3 // (op 795) 
v_lshlrev_b32 v5, 2, v4 // (op 795) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 795) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 795) 

v_lshrrev_b32 v4, 2, v3 // (op 795) 

v_and_b32 v3, 3, v4 // (op 795) 

v_and_b32 v4, 3, v1 // (op 795) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 795) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 795) 

v_mul_lo_u32 v3, v6, s34 // (op 795) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 795) 


s_mul_i32 s15, s34, s13 // (op 797) most significant: low of low * high
s_mul_i32 s0, s35, s12 // (op 797) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 797) least significant
s_mul_hi_u32 s1, s34, s12 // (op 797) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 797) most significant: sum
s_add_u32 s15, s15, s1 // (op 797) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 797) 

s_mov_b32 s4, s36 // (op 797) reg expression
s_mov_b32 s5, s37 // (op 797) reg expression
s_mov_b32 s6, s0 // (op 797) reg expression
s_mov_b32 s7, 131072 // (op 797) call()





buffer_load_dwordx2 v[8:9], v4, s[4:7], 0 offen // (op 170) Load value



v_accvgpr_read v12, a0 // (op 166) 
v_accvgpr_read v13, a1 // (op 166) 
v_accvgpr_read v14, a2 // (op 166) 
v_accvgpr_read v15, a3 // (op 166) 
v_mul_f32 v16, v11, v12 // (op 166) 
v_mul_f32 v17, v11, v13 // (op 166) 
v_mul_f32 v18, v11, v14 // (op 166) 
v_mul_f32 v19, v11, v15 // (op 166) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v8 // (op 171) 
v_lshrrev_b32 v13, 16, v8 // (op 171) 
v_cvt_f32_f16 v13, v13 // (op 171) 
v_mul_f32 v20, v7, v12 // (op 171) 
v_mul_f32 v21, v7, v13 // (op 171) 
v_cvt_f32_f16 v12, v9 // (op 171) 
v_lshrrev_b32 v13, 16, v9 // (op 171) 
v_cvt_f32_f16 v13, v13 // (op 171) 
v_mul_f32 v22, v7, v12 // (op 171) 
v_mul_f32 v23, v7, v13 // (op 171) 


v_add_f32 v12, v20, v16 // (op 167) 
v_add_f32 v13, v21, v17 // (op 167) 
v_add_f32 v14, v22, v18 // (op 167) 
v_add_f32 v15, v23, v19 // (op 167) 


v_cvt_f16_f32 v16, v12 // (op 168) 
v_cvt_f16_f32 v17, v13 // (op 168) 
v_cvt_f16_f32 v18, v14 // (op 168) 
v_cvt_f16_f32 v19, v15 // (op 168) 

v_lshrrev_b32 v0, 6, v1 // (op 1043) 
v_and_b32 v3, 63, v1 // (op 1043) 
v_lshrrev_b32 v4, 4, v3 // (op 1043) 
v_lshlrev_b32 v5, 2, v4 // (op 1043) 

v_lshl_add_u32 v4, v0, 4, v5 // (op 1043) 


v_lshl_add_u32 v0, s16, 4, v4 // (op 1043) 

v_lshrrev_b32 v4, 2, v3 // (op 1043) 

v_and_b32 v3, 3, v4 // (op 1043) 

v_and_b32 v4, 3, v1 // (op 1043) 
v_lshl_add_u32 v5, v3, 2, v4 // (op 1043) 


v_lshl_add_u32 v6, s17, 4, v5 // (op 1043) 

v_mul_lo_u32 v3, v6, s30 // (op 1043) 

v_add_lshl_u32 v4, v0, v3, 1 // (op 1043) 


s_mul_i32 s15, s30, s13 // (op 1045) most significant: low of low * high
s_mul_i32 s0, s31, s12 // (op 1045) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1045) least significant
s_mul_hi_u32 s1, s30, s12 // (op 1045) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1045) most significant: sum
s_add_u32 s15, s15, s1 // (op 1045) most significant: sum


s_lshl_b64 s[0:1], s[14:15], 1 // (op 1045) 

s_mov_b32 s8, s32 // (op 1045) reg expression
s_mov_b32 s9, s33 // (op 1045) reg expression
s_mov_b32 s10, s0 // (op 1045) reg expression
s_mov_b32 s11, 131072 // (op 1045) call()


v_pack_b32_f16 v8, v16, v17 // (op 169) 
v_pack_b32_f16 v9, v18, v19 // (op 169) 


buffer_store_dwordx2 v[8:9], v4, s[8:11], 0 offen // (op 169) Store value






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_ConditionalBottom_0_beta_157:




s_endpgm  // End of RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.LRR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end:
.size RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2, .LRR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2_end-RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
  .amdhsa_next_free_vgpr 120
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 12800 // lds bytes
  .amdhsa_accum_offset 116
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
    DataTypeB: F4
    DestDataType: h
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
  - .name: RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2
    .symbol: RR_GEMM_NN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_16x16x256_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 12800
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 115
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

