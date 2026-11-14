// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
.p2align 8
.type RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4,@function
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 55206) 
v_and_b32 v3, 48, v1 // (op 55206) 
v_sub_u32 v4, v0, v3 // (op 55206) 

v_lshrrev_b32 v3, 4, v0 // (op 55206) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 55206) 


v_lshrrev_b32 v3, 7, v1 // (op 55206) 
v_lshlrev_b32 v4, 4, v3 // (op 55206) 

v_lshl_add_u32 v3, v0, 8, v4 // (op 55206) 


v_lshlrev_b32 v0, 2, v3 // (op 55206) 

v_lshrrev_b32 v3, 3, v0 // (op 55206) 

v_and_b32 v0, 63, v1 // (op 55263) 
v_and_b32 v4, 48, v1 // (op 55263) 
v_sub_u32 v5, v0, v4 // (op 55263) 

v_lshrrev_b32 v4, 4, v0 // (op 55263) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 55263) 


v_lshrrev_b32 v4, 6, v1 // (op 55263) 
v_and_b32 v5, 1, v4 // (op 55263) 

v_lshlrev_b32 v4, 4, v5 // (op 55263) 

v_lshl_add_u32 v5, v0, 8, v4 // (op 55263) 


v_lshlrev_b32 v0, 2, v5 // (op 55263) 

v_lshrrev_b32 v4, 3, v0 // (op 55263) 

v_lshrrev_b32 v0, 7, v1 // (op 55321) 
v_and_b32 v5, 15, v1 // (op 55321) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 55321) 


v_and_b32 v0, 63, v1 // (op 55321) 
v_lshrrev_b32 v5, 4, v0 // (op 55321) 

v_lshl_add_u32 v0, v5, 8, v6 // (op 55321) 


v_lshrrev_b32 v5, 6, v1 // (op 55379) 
v_and_b32 v6, 1, v5 // (op 55379) 

v_and_b32 v5, 15, v1 // (op 55379) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 55379) 


v_and_b32 v5, 63, v1 // (op 55379) 
v_lshrrev_b32 v6, 4, v5 // (op 55379) 

v_lshl_add_u32 v5, v6, 8, v7 // (op 55379) 


v_and_b32 v6, 63, v1 // (op 55446) 
v_and_b32 v7, 48, v1 // (op 55446) 
v_sub_u32 v8, v6, v7 // (op 55446) 

v_lshrrev_b32 v7, 4, v6 // (op 55446) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 55446) 


v_lshrrev_b32 v7, 7, v1 // (op 55446) 
v_lshlrev_b32 v8, 4, v7 // (op 55446) 

v_lshl_add_u32 v7, v6, 8, v8 // (op 55446) 


v_lshlrev_b32 v6, 2, v7 // (op 55446) 

v_lshrrev_b32 v7, 3, v6 // (op 55446) 

v_and_b32 v6, 63, v1 // (op 55490) 
v_and_b32 v8, 48, v1 // (op 55490) 
v_sub_u32 v9, v6, v8 // (op 55490) 

v_lshrrev_b32 v8, 4, v6 // (op 55490) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 55490) 


v_lshrrev_b32 v8, 6, v1 // (op 55490) 
v_and_b32 v9, 1, v8 // (op 55490) 

v_lshlrev_b32 v8, 4, v9 // (op 55490) 

v_lshl_add_u32 v9, v6, 8, v8 // (op 55490) 


v_lshlrev_b32 v6, 2, v9 // (op 55490) 

v_lshrrev_b32 v8, 3, v6 // (op 55490) 

v_lshrrev_b32 v6, 7, v1 // (op 55519) 
v_and_b32 v9, 15, v1 // (op 55519) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 55519) 


v_and_b32 v6, 63, v1 // (op 55519) 
v_lshrrev_b32 v9, 4, v6 // (op 55519) 

v_lshl_add_u32 v6, v9, 8, v10 // (op 55519) 


v_lshrrev_b32 v9, 6, v1 // (op 55548) 
v_and_b32 v10, 1, v9 // (op 55548) 

v_and_b32 v9, 15, v1 // (op 55548) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 55548) 


v_and_b32 v9, 63, v1 // (op 55548) 
v_lshrrev_b32 v10, 4, v9 // (op 55548) 

v_lshl_add_u32 v9, v10, 8, v11 // (op 55548) 


v_and_b32 v10, 63, v1 // (op 55592) 
v_and_b32 v11, 48, v1 // (op 55592) 
v_sub_u32 v12, v10, v11 // (op 55592) 

v_lshrrev_b32 v11, 4, v10 // (op 55592) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 55592) 


v_lshrrev_b32 v11, 7, v1 // (op 55592) 
v_lshlrev_b32 v12, 4, v11 // (op 55592) 

v_lshl_add_u32 v11, v10, 8, v12 // (op 55592) 


v_lshlrev_b32 v10, 2, v11 // (op 55592) 

v_lshrrev_b32 v11, 3, v10 // (op 55592) 

v_and_b32 v10, 63, v1 // (op 55636) 
v_and_b32 v12, 48, v1 // (op 55636) 
v_sub_u32 v13, v10, v12 // (op 55636) 

v_lshrrev_b32 v12, 4, v10 // (op 55636) 

v_lshl_add_u32 v10, v12, 5, v13 // (op 55636) 


v_lshrrev_b32 v12, 6, v1 // (op 55636) 
v_and_b32 v13, 1, v12 // (op 55636) 

v_lshlrev_b32 v12, 4, v13 // (op 55636) 

v_lshl_add_u32 v13, v10, 8, v12 // (op 55636) 


v_lshlrev_b32 v10, 2, v13 // (op 55636) 

v_lshrrev_b32 v12, 3, v10 // (op 55636) 

v_lshrrev_b32 v10, 7, v1 // (op 55665) 
v_and_b32 v13, 15, v1 // (op 55665) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 55665) 


v_and_b32 v10, 63, v1 // (op 55665) 
v_lshrrev_b32 v13, 4, v10 // (op 55665) 

v_lshl_add_u32 v10, v13, 8, v14 // (op 55665) 


v_lshrrev_b32 v13, 6, v1 // (op 55694) 
v_and_b32 v14, 1, v13 // (op 55694) 

v_and_b32 v13, 15, v1 // (op 55694) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 55694) 


v_and_b32 v13, 63, v1 // (op 55694) 
v_lshrrev_b32 v14, 4, v13 // (op 55694) 

v_lshl_add_u32 v13, v14, 8, v15 // (op 55694) 


v_and_b32 v14, 63, v1 // (op 55738) 
v_and_b32 v15, 48, v1 // (op 55738) 
v_sub_u32 v16, v14, v15 // (op 55738) 

v_lshrrev_b32 v15, 4, v14 // (op 55738) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 55738) 


v_lshrrev_b32 v15, 7, v1 // (op 55738) 
v_lshlrev_b32 v16, 4, v15 // (op 55738) 

v_lshl_add_u32 v15, v14, 8, v16 // (op 55738) 


v_lshlrev_b32 v14, 2, v15 // (op 55738) 

v_lshrrev_b32 v15, 3, v14 // (op 55738) 

v_and_b32 v14, 63, v1 // (op 55782) 
v_and_b32 v16, 48, v1 // (op 55782) 
v_sub_u32 v17, v14, v16 // (op 55782) 

v_lshrrev_b32 v16, 4, v14 // (op 55782) 

v_lshl_add_u32 v14, v16, 5, v17 // (op 55782) 


v_lshrrev_b32 v16, 6, v1 // (op 55782) 
v_and_b32 v17, 1, v16 // (op 55782) 

v_lshlrev_b32 v16, 4, v17 // (op 55782) 

v_lshl_add_u32 v17, v14, 8, v16 // (op 55782) 


v_lshlrev_b32 v14, 2, v17 // (op 55782) 

v_lshrrev_b32 v16, 3, v14 // (op 55782) 

v_lshrrev_b32 v14, 7, v1 // (op 55811) 
v_and_b32 v17, 15, v1 // (op 55811) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 55811) 


v_and_b32 v14, 63, v1 // (op 55811) 
v_lshrrev_b32 v17, 4, v14 // (op 55811) 

v_lshl_add_u32 v14, v17, 8, v18 // (op 55811) 


v_lshrrev_b32 v17, 6, v1 // (op 55840) 
v_and_b32 v18, 1, v17 // (op 55840) 

v_and_b32 v17, 15, v1 // (op 55840) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 55840) 


v_and_b32 v17, 63, v1 // (op 55840) 
v_lshrrev_b32 v18, 4, v17 // (op 55840) 

v_lshl_add_u32 v17, v18, 8, v19 // (op 55840) 


v_lshrrev_b32 v18, 7, v1 // (op 55145) 
v_and_b32 v19, 63, v1 // (op 55145) 
v_lshrrev_b32 v20, 4, v19 // (op 55145) 
v_lshlrev_b32 v21, 2, v20 // (op 55145) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 55145) 


v_lshl_add_u32 v18, s16, 8, v20 // (op 55145) 

v_lshrrev_b32 v20, 2, v19 // (op 55145) 

v_and_b32 v19, 3, v20 // (op 55145) 

v_and_b32 v20, 3, v1 // (op 55145) 
v_lshl_add_u32 v21, v19, 2, v20 // (op 55145) 


v_lshrrev_b32 v19, 6, v1 // (op 55145) 
v_and_b32 v20, 1, v19 // (op 55145) 

v_lshl_add_u32 v19, v20, 4, v21 // (op 55145) 


v_lshl_add_u32 v20, s17, 8, v19 // (op 55145) 

v_mul_lo_u32 v19, v20, s2 // (op 55145) 

v_add_lshl_u32 v20, v18, v19, 1 // (op 55145) 


v_lshrrev_b32 v18, 7, v1 // (op 55400) 
v_and_b32 v19, 63, v1 // (op 55400) 
v_lshrrev_b32 v21, 4, v19 // (op 55400) 
s_mul_i32 s1, s2, s9 // (op 55147) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 55147) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 55147) least significant
s_mul_hi_u32 s19, s2, s8 // (op 55147) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 55147) most significant: sum
s_add_u32 s1, s1, s19 // (op 55147) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 55147) 

v_lshlrev_b32 v22, 2, v21 // (op 55400) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 55400) 


v_lshl_add_u32 v18, s16, 8, v21 // (op 55400) 

v_lshrrev_b32 v21, 2, v19 // (op 55400) 

v_and_b32 v19, 3, v21 // (op 55400) 

v_and_b32 v21, 3, v1 // (op 55400) 
v_lshl_add_u32 v22, v19, 2, v21 // (op 55400) 


v_lshrrev_b32 v19, 6, v1 // (op 55400) 
v_and_b32 v21, 1, v19 // (op 55400) 

v_lshl_add_u32 v19, v21, 4, v22 // (op 55400) 


v_lshl_add_u32 v21, s17, 8, v19 // (op 55400) 

v_mul_lo_u32 v19, v21, s4 // (op 55400) 

v_add_lshl_u32 v21, v18, v19, 1 // (op 55400) 


s_mov_b32 s40, s6 // (op 55147) reg expression
s_mov_b32 s41, s7 // (op 55147) reg expression
s_mov_b32 s42, s18 // (op 55147) reg expression
s_mov_b32 s43, 131072 // (op 55147) call()

s_mul_i32 s1, s4, s9 // (op 55402) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 55402) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 55402) least significant
s_mul_hi_u32 s7, s4, s8 // (op 55402) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 55402) most significant: sum
s_add_u32 s1, s1, s7 // (op 55402) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 55402) 

s_mov_b32 s44, s10 // (op 55402) reg expression
s_mov_b32 s45, s11 // (op 55402) reg expression
s_mov_b32 s46, s6 // (op 55402) reg expression
s_mov_b32 s47, 131072 // (op 55402) call()



v_mov_b32 v18, s14 // (op 2) Move value
v_mov_b32 v19, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 486) call()
v_accvgpr_write a5, 0.00000 // (op 486) call()
v_accvgpr_write a6, 0.00000 // (op 486) call()
v_accvgpr_write a7, 0.00000 // (op 486) call()
v_accvgpr_write a8, 0.00000 // (op 784) call()
v_accvgpr_write a9, 0.00000 // (op 784) call()
v_accvgpr_write a10, 0.00000 // (op 784) call()
v_accvgpr_write a11, 0.00000 // (op 784) call()
v_accvgpr_write a12, 0.00000 // (op 1082) call()
v_accvgpr_write a13, 0.00000 // (op 1082) call()
v_accvgpr_write a14, 0.00000 // (op 1082) call()
v_accvgpr_write a15, 0.00000 // (op 1082) call()
v_accvgpr_write a16, 0.00000 // (op 1380) call()
v_accvgpr_write a17, 0.00000 // (op 1380) call()
v_accvgpr_write a18, 0.00000 // (op 1380) call()
v_accvgpr_write a19, 0.00000 // (op 1380) call()
v_accvgpr_write a20, 0.00000 // (op 1678) call()
v_accvgpr_write a21, 0.00000 // (op 1678) call()
v_accvgpr_write a22, 0.00000 // (op 1678) call()
v_accvgpr_write a23, 0.00000 // (op 1678) call()
v_accvgpr_write a24, 0.00000 // (op 1976) call()
v_accvgpr_write a25, 0.00000 // (op 1976) call()
v_accvgpr_write a26, 0.00000 // (op 1976) call()
v_accvgpr_write a27, 0.00000 // (op 1976) call()
v_accvgpr_write a28, 0.00000 // (op 2274) call()
v_accvgpr_write a29, 0.00000 // (op 2274) call()
v_accvgpr_write a30, 0.00000 // (op 2274) call()
v_accvgpr_write a31, 0.00000 // (op 2274) call()
v_accvgpr_write a32, 0.00000 // (op 2844) call()
v_accvgpr_write a33, 0.00000 // (op 2844) call()
v_accvgpr_write a34, 0.00000 // (op 2844) call()
v_accvgpr_write a35, 0.00000 // (op 2844) call()
v_accvgpr_write a36, 0.00000 // (op 3829) call()
v_accvgpr_write a37, 0.00000 // (op 3829) call()
v_accvgpr_write a38, 0.00000 // (op 3829) call()
v_accvgpr_write a39, 0.00000 // (op 3829) call()
v_accvgpr_write a40, 0.00000 // (op 3830) call()
v_accvgpr_write a41, 0.00000 // (op 3830) call()
v_accvgpr_write a42, 0.00000 // (op 3830) call()
v_accvgpr_write a43, 0.00000 // (op 3830) call()
v_accvgpr_write a44, 0.00000 // (op 3831) call()
v_accvgpr_write a45, 0.00000 // (op 3831) call()
v_accvgpr_write a46, 0.00000 // (op 3831) call()
v_accvgpr_write a47, 0.00000 // (op 3831) call()
v_accvgpr_write a48, 0.00000 // (op 3832) call()
v_accvgpr_write a49, 0.00000 // (op 3832) call()
v_accvgpr_write a50, 0.00000 // (op 3832) call()
v_accvgpr_write a51, 0.00000 // (op 3832) call()
v_accvgpr_write a52, 0.00000 // (op 3833) call()
v_accvgpr_write a53, 0.00000 // (op 3833) call()
v_accvgpr_write a54, 0.00000 // (op 3833) call()
v_accvgpr_write a55, 0.00000 // (op 3833) call()
v_accvgpr_write a56, 0.00000 // (op 3834) call()
v_accvgpr_write a57, 0.00000 // (op 3834) call()
v_accvgpr_write a58, 0.00000 // (op 3834) call()
v_accvgpr_write a59, 0.00000 // (op 3834) call()
v_accvgpr_write a60, 0.00000 // (op 3835) call()
v_accvgpr_write a61, 0.00000 // (op 3835) call()
v_accvgpr_write a62, 0.00000 // (op 3835) call()
v_accvgpr_write a63, 0.00000 // (op 3835) call()
v_accvgpr_write a64, 0.00000 // (op 5422) call()
v_accvgpr_write a65, 0.00000 // (op 5422) call()
v_accvgpr_write a66, 0.00000 // (op 5422) call()
v_accvgpr_write a67, 0.00000 // (op 5422) call()
v_accvgpr_write a68, 0.00000 // (op 6407) call()
v_accvgpr_write a69, 0.00000 // (op 6407) call()
v_accvgpr_write a70, 0.00000 // (op 6407) call()
v_accvgpr_write a71, 0.00000 // (op 6407) call()
v_accvgpr_write a72, 0.00000 // (op 6408) call()
v_accvgpr_write a73, 0.00000 // (op 6408) call()
v_accvgpr_write a74, 0.00000 // (op 6408) call()
v_accvgpr_write a75, 0.00000 // (op 6408) call()
v_accvgpr_write a76, 0.00000 // (op 6409) call()
v_accvgpr_write a77, 0.00000 // (op 6409) call()
v_accvgpr_write a78, 0.00000 // (op 6409) call()
v_accvgpr_write a79, 0.00000 // (op 6409) call()
v_accvgpr_write a80, 0.00000 // (op 6410) call()
v_accvgpr_write a81, 0.00000 // (op 6410) call()
v_accvgpr_write a82, 0.00000 // (op 6410) call()
v_accvgpr_write a83, 0.00000 // (op 6410) call()
v_accvgpr_write a84, 0.00000 // (op 6411) call()
v_accvgpr_write a85, 0.00000 // (op 6411) call()
v_accvgpr_write a86, 0.00000 // (op 6411) call()
v_accvgpr_write a87, 0.00000 // (op 6411) call()
v_accvgpr_write a88, 0.00000 // (op 6412) call()
v_accvgpr_write a89, 0.00000 // (op 6412) call()
v_accvgpr_write a90, 0.00000 // (op 6412) call()
v_accvgpr_write a91, 0.00000 // (op 6412) call()
v_accvgpr_write a92, 0.00000 // (op 6413) call()
v_accvgpr_write a93, 0.00000 // (op 6413) call()
v_accvgpr_write a94, 0.00000 // (op 6413) call()
v_accvgpr_write a95, 0.00000 // (op 6413) call()
v_accvgpr_write a96, 0.00000 // (op 8000) call()
v_accvgpr_write a97, 0.00000 // (op 8000) call()
v_accvgpr_write a98, 0.00000 // (op 8000) call()
v_accvgpr_write a99, 0.00000 // (op 8000) call()
v_accvgpr_write a100, 0.00000 // (op 8985) call()
v_accvgpr_write a101, 0.00000 // (op 8985) call()
v_accvgpr_write a102, 0.00000 // (op 8985) call()
v_accvgpr_write a103, 0.00000 // (op 8985) call()
v_accvgpr_write a104, 0.00000 // (op 8986) call()
v_accvgpr_write a105, 0.00000 // (op 8986) call()
v_accvgpr_write a106, 0.00000 // (op 8986) call()
v_accvgpr_write a107, 0.00000 // (op 8986) call()
v_accvgpr_write a108, 0.00000 // (op 8987) call()
v_accvgpr_write a109, 0.00000 // (op 8987) call()
v_accvgpr_write a110, 0.00000 // (op 8987) call()
v_accvgpr_write a111, 0.00000 // (op 8987) call()
v_accvgpr_write a112, 0.00000 // (op 8988) call()
v_accvgpr_write a113, 0.00000 // (op 8988) call()
v_accvgpr_write a114, 0.00000 // (op 8988) call()
v_accvgpr_write a115, 0.00000 // (op 8988) call()
v_accvgpr_write a116, 0.00000 // (op 8989) call()
v_accvgpr_write a117, 0.00000 // (op 8989) call()
v_accvgpr_write a118, 0.00000 // (op 8989) call()
v_accvgpr_write a119, 0.00000 // (op 8989) call()
v_accvgpr_write a120, 0.00000 // (op 8990) call()
v_accvgpr_write a121, 0.00000 // (op 8990) call()
v_accvgpr_write a122, 0.00000 // (op 8990) call()
v_accvgpr_write a123, 0.00000 // (op 8990) call()
v_accvgpr_write a124, 0.00000 // (op 8991) call()
v_accvgpr_write a125, 0.00000 // (op 8991) call()
v_accvgpr_write a126, 0.00000 // (op 8991) call()
v_accvgpr_write a127, 0.00000 // (op 8991) call()
v_accvgpr_write a128, 0.00000 // (op 10578) call()
v_accvgpr_write a129, 0.00000 // (op 10578) call()
v_accvgpr_write a130, 0.00000 // (op 10578) call()
v_accvgpr_write a131, 0.00000 // (op 10578) call()
v_accvgpr_write a132, 0.00000 // (op 11563) call()
v_accvgpr_write a133, 0.00000 // (op 11563) call()
v_accvgpr_write a134, 0.00000 // (op 11563) call()
v_accvgpr_write a135, 0.00000 // (op 11563) call()
v_accvgpr_write a136, 0.00000 // (op 11564) call()
v_accvgpr_write a137, 0.00000 // (op 11564) call()
v_accvgpr_write a138, 0.00000 // (op 11564) call()
v_accvgpr_write a139, 0.00000 // (op 11564) call()
v_accvgpr_write a140, 0.00000 // (op 11565) call()
v_accvgpr_write a141, 0.00000 // (op 11565) call()
v_accvgpr_write a142, 0.00000 // (op 11565) call()
v_accvgpr_write a143, 0.00000 // (op 11565) call()
v_accvgpr_write a144, 0.00000 // (op 11566) call()
v_accvgpr_write a145, 0.00000 // (op 11566) call()
v_accvgpr_write a146, 0.00000 // (op 11566) call()
v_accvgpr_write a147, 0.00000 // (op 11566) call()
v_accvgpr_write a148, 0.00000 // (op 11567) call()
v_accvgpr_write a149, 0.00000 // (op 11567) call()
v_accvgpr_write a150, 0.00000 // (op 11567) call()
v_accvgpr_write a151, 0.00000 // (op 11567) call()
v_accvgpr_write a152, 0.00000 // (op 11568) call()
v_accvgpr_write a153, 0.00000 // (op 11568) call()
v_accvgpr_write a154, 0.00000 // (op 11568) call()
v_accvgpr_write a155, 0.00000 // (op 11568) call()
v_accvgpr_write a156, 0.00000 // (op 11569) call()
v_accvgpr_write a157, 0.00000 // (op 11569) call()
v_accvgpr_write a158, 0.00000 // (op 11569) call()
v_accvgpr_write a159, 0.00000 // (op 11569) call()
v_accvgpr_write a160, 0.00000 // (op 13156) call()
v_accvgpr_write a161, 0.00000 // (op 13156) call()
v_accvgpr_write a162, 0.00000 // (op 13156) call()
v_accvgpr_write a163, 0.00000 // (op 13156) call()
v_accvgpr_write a164, 0.00000 // (op 14141) call()
v_accvgpr_write a165, 0.00000 // (op 14141) call()
v_accvgpr_write a166, 0.00000 // (op 14141) call()
v_accvgpr_write a167, 0.00000 // (op 14141) call()
v_accvgpr_write a168, 0.00000 // (op 14142) call()
v_accvgpr_write a169, 0.00000 // (op 14142) call()
v_accvgpr_write a170, 0.00000 // (op 14142) call()
v_accvgpr_write a171, 0.00000 // (op 14142) call()
v_accvgpr_write a172, 0.00000 // (op 14143) call()
v_accvgpr_write a173, 0.00000 // (op 14143) call()
v_accvgpr_write a174, 0.00000 // (op 14143) call()
v_accvgpr_write a175, 0.00000 // (op 14143) call()
v_accvgpr_write a176, 0.00000 // (op 14144) call()
v_accvgpr_write a177, 0.00000 // (op 14144) call()
v_accvgpr_write a178, 0.00000 // (op 14144) call()
v_accvgpr_write a179, 0.00000 // (op 14144) call()
v_accvgpr_write a180, 0.00000 // (op 14145) call()
v_accvgpr_write a181, 0.00000 // (op 14145) call()
v_accvgpr_write a182, 0.00000 // (op 14145) call()
v_accvgpr_write a183, 0.00000 // (op 14145) call()
v_accvgpr_write a184, 0.00000 // (op 14146) call()
v_accvgpr_write a185, 0.00000 // (op 14146) call()
v_accvgpr_write a186, 0.00000 // (op 14146) call()
v_accvgpr_write a187, 0.00000 // (op 14146) call()
v_accvgpr_write a188, 0.00000 // (op 14147) call()
v_accvgpr_write a189, 0.00000 // (op 14147) call()
v_accvgpr_write a190, 0.00000 // (op 14147) call()
v_accvgpr_write a191, 0.00000 // (op 14147) call()
v_accvgpr_write a192, 0.00000 // (op 15734) call()
v_accvgpr_write a193, 0.00000 // (op 15734) call()
v_accvgpr_write a194, 0.00000 // (op 15734) call()
v_accvgpr_write a195, 0.00000 // (op 15734) call()
v_accvgpr_write a196, 0.00000 // (op 16719) call()
v_accvgpr_write a197, 0.00000 // (op 16719) call()
v_accvgpr_write a198, 0.00000 // (op 16719) call()
v_accvgpr_write a199, 0.00000 // (op 16719) call()
v_accvgpr_write a200, 0.00000 // (op 16720) call()
v_accvgpr_write a201, 0.00000 // (op 16720) call()
v_accvgpr_write a202, 0.00000 // (op 16720) call()
v_accvgpr_write a203, 0.00000 // (op 16720) call()
v_accvgpr_write a204, 0.00000 // (op 16721) call()
v_accvgpr_write a205, 0.00000 // (op 16721) call()
v_accvgpr_write a206, 0.00000 // (op 16721) call()
v_accvgpr_write a207, 0.00000 // (op 16721) call()
v_accvgpr_write a208, 0.00000 // (op 16722) call()
v_accvgpr_write a209, 0.00000 // (op 16722) call()
v_accvgpr_write a210, 0.00000 // (op 16722) call()
v_accvgpr_write a211, 0.00000 // (op 16722) call()
v_accvgpr_write a212, 0.00000 // (op 16723) call()
v_accvgpr_write a213, 0.00000 // (op 16723) call()
v_accvgpr_write a214, 0.00000 // (op 16723) call()
v_accvgpr_write a215, 0.00000 // (op 16723) call()
v_accvgpr_write a216, 0.00000 // (op 16724) call()
v_accvgpr_write a217, 0.00000 // (op 16724) call()
v_accvgpr_write a218, 0.00000 // (op 16724) call()
v_accvgpr_write a219, 0.00000 // (op 16724) call()
v_accvgpr_write a220, 0.00000 // (op 16725) call()
v_accvgpr_write a221, 0.00000 // (op 16725) call()
v_accvgpr_write a222, 0.00000 // (op 16725) call()
v_accvgpr_write a223, 0.00000 // (op 16725) call()
v_accvgpr_write a224, 0.00000 // (op 18312) call()
v_accvgpr_write a225, 0.00000 // (op 18312) call()
v_accvgpr_write a226, 0.00000 // (op 18312) call()
v_accvgpr_write a227, 0.00000 // (op 18312) call()
v_accvgpr_write a228, 0.00000 // (op 19297) call()
v_accvgpr_write a229, 0.00000 // (op 19297) call()
v_accvgpr_write a230, 0.00000 // (op 19297) call()
v_accvgpr_write a231, 0.00000 // (op 19297) call()
v_accvgpr_write a232, 0.00000 // (op 19298) call()
v_accvgpr_write a233, 0.00000 // (op 19298) call()
v_accvgpr_write a234, 0.00000 // (op 19298) call()
v_accvgpr_write a235, 0.00000 // (op 19298) call()
v_accvgpr_write a236, 0.00000 // (op 19299) call()
v_accvgpr_write a237, 0.00000 // (op 19299) call()
v_accvgpr_write a238, 0.00000 // (op 19299) call()
v_accvgpr_write a239, 0.00000 // (op 19299) call()
v_accvgpr_write a240, 0.00000 // (op 19300) call()
v_accvgpr_write a241, 0.00000 // (op 19300) call()
v_accvgpr_write a242, 0.00000 // (op 19300) call()
v_accvgpr_write a243, 0.00000 // (op 19300) call()
v_accvgpr_write a244, 0.00000 // (op 19301) call()
v_accvgpr_write a245, 0.00000 // (op 19301) call()
v_accvgpr_write a246, 0.00000 // (op 19301) call()
v_accvgpr_write a247, 0.00000 // (op 19301) call()
v_accvgpr_write a248, 0.00000 // (op 19302) call()
v_accvgpr_write a249, 0.00000 // (op 19302) call()
v_accvgpr_write a250, 0.00000 // (op 19302) call()
v_accvgpr_write a251, 0.00000 // (op 19302) call()
v_accvgpr_write a252, 0.00000 // (op 19303) call()
v_accvgpr_write a253, 0.00000 // (op 19303) call()
v_accvgpr_write a254, 0.00000 // (op 19303) call()
v_accvgpr_write a255, 0.00000 // (op 19303) call()
v_and_b32 v22, -8, v1 // (op 54907) 
v_sub_u32 v23, v1, v22 // (op 54907) 

v_lshlrev_b32 v22, 5, v23 // (op 54907) 

v_lshl_add_u32 v23, s16, 8, v22 // (op 54907) 

v_mov_b32 v25, 0 // (op 54907) convert
v_mov_b32 v24, v23 // (op 54907) convert

v_lshrrev_b32 v22, 3, v1 // (op 54907) 
v_mul_lo_u32 v27, v22, s13 // (op 54907) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 54907) most significant: high of low * low
v_mul_lo_u32 v26, v22, s12 // (op 54907) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 54907) most significant: sum


v_add_co_u32 v22, s[0:1], v24, v26 // (op 54907) least significant half
v_addc_co_u32 v23, s[0:1], v25, v27, s[0:1] // (op 54907) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 54907) 


v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 54907) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 54909) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 54909) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 54909) least significant
s_mul_hi_u32 s7, s12, s22 // (op 54909) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54909) most significant: sum
s_add_u32 s1, s1, s7 // (op 54909) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 54909) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 54909) 

s_mov_b32 s8, s20 // (op 54909) reg expression
s_mov_b32 s9, s21 // (op 54909) reg expression
s_mov_b32 s10, s0 // (op 54909) reg expression
s_mov_b32 s11, 131072 // (op 54909) call()

v_and_b32 v22, -8, v1 // (op 54926) 
v_sub_u32 v23, v1, v22 // (op 54926) 

v_lshlrev_b32 v22, 5, v23 // (op 54926) 

v_lshl_add_u32 v23, s16, 8, v22 // (op 54926) 

v_mov_b32 v27, 0 // (op 54926) convert
v_mov_b32 v26, v23 // (op 54926) convert

v_lshrrev_b32 v22, 3, v1 // (op 54926) 
v_mul_lo_u32 v29, v22, s13 // (op 54926) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 54926) most significant: high of low * low
v_mul_lo_u32 v28, v22, s12 // (op 54926) least significant: low of low * low
v_add_u32 v29, v29, v23 // (op 54926) most significant: sum


v_add_co_u32 v22, s[0:1], v26, v28 // (op 54926) least significant half
v_addc_co_u32 v23, s[0:1], v27, v29, s[0:1] // (op 54926) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 54926) 


v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 54926) 

s_mul_i32 s1, s12, s23 // (op 54928) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 54928) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 54928) least significant
s_mul_hi_u32 s7, s12, s22 // (op 54928) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54928) most significant: sum
s_add_u32 s1, s1, s7 // (op 54928) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54928) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54928) 

s_mov_b32 s8, s20 // (op 54928) reg expression
s_mov_b32 s9, s21 // (op 54928) reg expression
s_mov_b32 s10, s0 // (op 54928) reg expression
s_mov_b32 s11, 131072 // (op 54928) call()

v_and_b32 v22, -8, v1 // (op 54971) 
v_sub_u32 v23, v1, v22 // (op 54971) 

v_lshlrev_b32 v22, 5, v23 // (op 54971) 

v_lshl_add_u32 v23, s17, 8, v22 // (op 54971) 

v_mov_b32 v29, 0 // (op 54971) convert
v_mov_b32 v28, v23 // (op 54971) convert

v_lshrrev_b32 v22, 3, v1 // (op 54971) 
v_mul_lo_u32 v31, v22, s25 // (op 54971) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 54971) most significant: high of low * low
v_mul_lo_u32 v30, v22, s24 // (op 54971) least significant: low of low * low
v_add_u32 v31, v31, v23 // (op 54971) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v28 // (op 54971) least significant half
v_addc_co_u32 v31, s[0:1], v31, v29, s[0:1] // (op 54971) most significant half



v_lshlrev_b64 v[22:23], 2, v[30:31] // (op 54971) 

v_lshrrev_b64 v[28:29], 3, v[22:23] // (op 54971) 

s_mul_i32 s1, s24, s23 // (op 54973) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 54973) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 54973) least significant
s_mul_hi_u32 s7, s24, s22 // (op 54973) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54973) most significant: sum
s_add_u32 s1, s1, s7 // (op 54973) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54973) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54973) 

s_mov_b32 s48, s26 // (op 54973) reg expression
s_mov_b32 s49, s27 // (op 54973) reg expression
s_mov_b32 s50, s0 // (op 54973) reg expression
s_mov_b32 s51, 131072 // (op 54973) call()

v_and_b32 v22, -8, v1 // (op 54990) 
v_sub_u32 v23, v1, v22 // (op 54990) 

v_lshlrev_b32 v22, 5, v23 // (op 54990) 

v_lshl_add_u32 v23, s17, 8, v22 // (op 54990) 

v_mov_b32 v31, 0 // (op 54990) convert
v_mov_b32 v30, v23 // (op 54990) convert

v_lshrrev_b32 v22, 3, v1 // (op 54990) 
v_mul_lo_u32 v33, v22, s25 // (op 54990) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 54990) most significant: high of low * low
v_mul_lo_u32 v32, v22, s24 // (op 54990) least significant: low of low * low
v_add_u32 v33, v33, v23 // (op 54990) most significant: sum

v_add_co_u32 v32, s[0:1], v32, v30 // (op 54990) least significant half
v_addc_co_u32 v33, s[0:1], v33, v31, s[0:1] // (op 54990) most significant half



v_lshlrev_b64 v[22:23], 2, v[32:33] // (op 54990) 

v_lshrrev_b64 v[30:31], 3, v[22:23] // (op 54990) 

s_mul_i32 s1, s24, s23 // (op 54992) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 54992) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 54992) least significant
s_mul_hi_u32 s7, s24, s22 // (op 54992) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54992) most significant: sum
s_add_u32 s1, s1, s7 // (op 54992) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54992) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54992) 

s_mov_b32 s48, s26 // (op 54992) reg expression
s_mov_b32 s49, s27 // (op 54992) reg expression
s_mov_b32 s50, s0 // (op 54992) reg expression
s_mov_b32 s51, 131072 // (op 54992) call()

v_and_b32 v22, -64, v1 // (op 55035) 
v_sub_u32 v23, v1, v22 // (op 55035) 

v_lshlrev_b32 v22, 2, v23 // (op 55035) 

v_lshl_add_u32 v23, s16, 8, v22 // (op 55035) 

v_lshrrev_b32 v22, 6, v1 // (op 55035) 
v_mul_lo_u32 v32, v22, s28 // (op 55035) 

v_add_u32 v22, v23, v32 // (op 55035) 


s_mul_i32 s1, s28, s33 // (op 55037) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 55037) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 55037) least significant
s_mul_hi_u32 s7, s28, s32 // (op 55037) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 55037) most significant: sum
s_add_u32 s1, s1, s7 // (op 55037) most significant: sum


s_mov_b32 s52, s30 // (op 55037) reg expression
s_mov_b32 s53, s31 // (op 55037) reg expression
s_mov_b32 s54, s0 // (op 55037) reg expression
s_mov_b32 s55, 131072 // (op 55037) call()

v_and_b32 v23, -64, v1 // (op 55054) 
v_sub_u32 v32, v1, v23 // (op 55054) 

v_lshlrev_b32 v23, 2, v32 // (op 55054) 

v_lshl_add_u32 v32, s16, 8, v23 // (op 55054) 

v_lshrrev_b32 v23, 6, v1 // (op 55054) 
v_mul_lo_u32 v33, v23, s28 // (op 55054) 

v_add_u32 v23, v32, v33 // (op 55054) 


s_mul_i32 s1, s28, s33 // (op 55056) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 55056) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 55056) least significant
s_mul_hi_u32 s7, s28, s32 // (op 55056) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 55056) most significant: sum
s_add_u32 s1, s1, s7 // (op 55056) most significant: sum


s_mov_b32 s52, s30 // (op 55056) reg expression
s_mov_b32 s53, s31 // (op 55056) reg expression
s_mov_b32 s54, s0 // (op 55056) reg expression
s_mov_b32 s55, 131072 // (op 55056) call()

v_and_b32 v32, -64, v1 // (op 55099) 
v_sub_u32 v33, v1, v32 // (op 55099) 

v_lshlrev_b32 v32, 2, v33 // (op 55099) 

v_lshl_add_u32 v33, s17, 8, v32 // (op 55099) 

v_lshrrev_b32 v32, 6, v1 // (op 55099) 
v_mul_lo_u32 v34, v32, s34 // (op 55099) 
v_add_u32 v34, v34, v33 // (op 55099) 


s_mul_i32 s1, s34, s33 // (op 55101) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 55101) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 55101) least significant
s_mul_hi_u32 s7, s34, s32 // (op 55101) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 55101) most significant: sum
s_add_u32 s1, s1, s7 // (op 55101) most significant: sum


s_mov_b32 s56, s36 // (op 55101) reg expression
s_mov_b32 s57, s37 // (op 55101) reg expression
s_mov_b32 s58, s0 // (op 55101) reg expression
s_mov_b32 s59, 131072 // (op 55101) call()

v_and_b32 v32, -64, v1 // (op 55118) 
v_sub_u32 v33, v1, v32 // (op 55118) 

v_lshlrev_b32 v32, 2, v33 // (op 55118) 

v_lshl_add_u32 v33, s17, 8, v32 // (op 55118) 

v_lshrrev_b32 v32, 6, v1 // (op 55118) 
v_mul_lo_u32 v35, v32, s34 // (op 55118) 
v_add_u32 v35, v35, v33 // (op 55118) 


s_mul_i32 s1, s34, s33 // (op 55120) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 55120) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 55120) least significant
s_mul_hi_u32 s7, s34, s32 // (op 55120) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 55120) most significant: sum
s_add_u32 s1, s1, s7 // (op 55120) most significant: sum


s_mov_b32 s56, s36 // (op 55120) reg expression
s_mov_b32 s57, s37 // (op 55120) reg expression
s_mov_b32 s58, s0 // (op 55120) reg expression
s_mov_b32 s59, 131072 // (op 55120) call()

v_and_b32 v32, -8, v1 // (op 55246) 
v_sub_u32 v33, v1, v32 // (op 55246) 

v_lshlrev_b32 v32, 5, v33 // (op 55246) 

v_lshrrev_b32 v33, 3, v1 // (op 55246) 
v_lshl_add_u32 v36, v33, 8, v32 // (op 55246) 


v_lshlrev_b32 v32, 2, v36 // (op 55246) 

v_lshrrev_b32 v33, 3, v32 // (op 55246) 

s_nop 0
v_readfirstlane_b32 s0, v33 // (op 55246) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 54785) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54785) 

v_lshrrev_b64 v[32:33], 3, s[14:15] // (op 54785) 


v_mov_b64 v[36:37], v[28:29] // (op 54785) reg expression
s_mov_b32 m0, s0 // (op 54785) reg expression
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 54785) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v32 // (op 54785) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 54785) most significant half

s_add_u32 m0, m0, 4096 // (op 54785) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 54785) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v32 // (op 54785) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 54785) most significant half

s_add_u32 m0, m0, 4096 // (op 54785) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 54785) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v32 // (op 54785) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 54785) most significant half

s_add_u32 m0, m0, 4096 // (op 54785) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 54785) Load value direct to lds





v_mov_b32 v32, v34 // (op 53791) reg expression

buffer_load_dword v33, v32, s[56:59], 0 offen // (op 53791) Load value

v_and_b32 v32, -8, v1 // (op 55189) 
v_sub_u32 v36, v1, v32 // (op 55189) 

v_lshlrev_b32 v32, 5, v36 // (op 55189) 

v_lshrrev_b32 v36, 3, v1 // (op 55189) 
v_lshl_add_u32 v37, v36, 8, v32 // (op 55189) 


v_lshlrev_b32 v32, 2, v37 // (op 55189) 

v_lshrrev_b32 v36, 3, v32 // (op 55189) 

s_nop 0
v_readfirstlane_b32 s0, v36 // (op 55189) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 54790) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54790) 

v_lshrrev_b64 v[36:37], 3, s[14:15] // (op 54790) 


v_mov_b64 v[38:39], v[24:25] // (op 54790) reg expression
s_add_u32 s1, 16384, s0 // (op 54790) 
s_mov_b32 m0, s1 // (op 54790) reg expression
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54790) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v36 // (op 54790) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54790) most significant half

s_add_u32 m0, m0, 4096 // (op 54790) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54790) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v36 // (op 54790) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54790) most significant half

s_add_u32 m0, m0, 4096 // (op 54790) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54790) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v36 // (op 54790) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54790) most significant half

s_add_u32 m0, m0, 4096 // (op 54790) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54790) Load value direct to lds






v_mov_b32 v32, v22 // (op 53797) reg expression

buffer_load_dword v36, v32, s[52:55], 0 offen // (op 53797) Load value

v_and_b32 v32, -8, v1 // (op 55473) 
v_sub_u32 v37, v1, v32 // (op 55473) 

v_lshlrev_b32 v32, 5, v37 // (op 55473) 

v_lshrrev_b32 v37, 3, v1 // (op 55473) 
v_lshl_add_u32 v38, v37, 8, v32 // (op 55473) 


v_lshlrev_b32 v32, 2, v38 // (op 55473) 

v_lshrrev_b32 v37, 3, v32 // (op 55473) 

s_nop 0
v_readfirstlane_b32 s0, v37 // (op 55473) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 54795) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54795) 

v_lshrrev_b64 v[38:39], 3, s[14:15] // (op 54795) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 54795) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54795) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54795) 

v_mov_b32 v32, s6 // (op 54795) 
v_mov_b32 v37, s7 // (op 54795) 
v_add_co_u32 v40, s[14:15], v28, v32 // (op 54795) least significant half
v_addc_co_u32 v41, s[14:15], v29, v37, s[14:15] // (op 54795) most significant half




s_add_u32 s1, 32768, s0 // (op 54795) 
s_mov_b32 m0, s1 // (op 54795) reg expression
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 54795) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v38 // (op 54795) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 54795) most significant half

s_add_u32 m0, m0, 4096 // (op 54795) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 54795) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v38 // (op 54795) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 54795) most significant half

s_add_u32 m0, m0, 4096 // (op 54795) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 54795) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v38 // (op 54795) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 54795) most significant half

s_add_u32 m0, m0, 4096 // (op 54795) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 54795) Load value direct to lds






s_lshl_b32 s0, s34, 2 // (op 53803) 
v_add_u32 v32, v34, s0 // (op 53803) 


buffer_load_dword v37, v32, s[56:59], 0 offen // (op 53803) Load value

v_and_b32 v32, -8, v1 // (op 55429) 
v_sub_u32 v38, v1, v32 // (op 55429) 

v_lshlrev_b32 v32, 5, v38 // (op 55429) 

v_lshrrev_b32 v38, 3, v1 // (op 55429) 
v_lshl_add_u32 v39, v38, 8, v32 // (op 55429) 


v_lshlrev_b32 v32, 2, v39 // (op 55429) 

v_lshrrev_b32 v38, 3, v32 // (op 55429) 

s_nop 0
v_readfirstlane_b32 s0, v38 // (op 55429) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 54797) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54797) 

v_lshrrev_b64 v[38:39], 3, s[14:15] // (op 54797) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 54797) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54797) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54797) 

v_mov_b32 v32, s6 // (op 54797) 
v_mov_b32 v40, s7 // (op 54797) 
v_add_co_u32 v42, s[14:15], v24, v32 // (op 54797) least significant half
v_addc_co_u32 v43, s[14:15], v25, v40, s[14:15] // (op 54797) most significant half




s_add_u32 s1, 49152, s0 // (op 54797) 
s_mov_b32 m0, s1 // (op 54797) reg expression
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 54797) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v38 // (op 54797) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54797) most significant half

s_add_u32 m0, m0, 4096 // (op 54797) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 54797) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v38 // (op 54797) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54797) most significant half

s_add_u32 m0, m0, 4096 // (op 54797) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 54797) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v38 // (op 54797) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54797) most significant half

s_add_u32 m0, m0, 4096 // (op 54797) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 54797) Load value direct to lds






s_lshl_b32 s0, s28, 2 // (op 53809) 
v_add_u32 v32, v22, s0 // (op 53809) 


buffer_load_dword v38, v32, s[52:55], 0 offen // (op 53809) Load value

v_and_b32 v32, -8, v1 // (op 55619) 
v_sub_u32 v39, v1, v32 // (op 55619) 

v_lshlrev_b32 v32, 5, v39 // (op 55619) 

v_lshrrev_b32 v39, 3, v1 // (op 55619) 
v_lshl_add_u32 v40, v39, 8, v32 // (op 55619) 


v_lshlrev_b32 v32, 2, v40 // (op 55619) 

v_lshrrev_b32 v39, 3, v32 // (op 55619) 

s_nop 0
v_readfirstlane_b32 s0, v39 // (op 55619) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 54799) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54799) 

v_lshrrev_b64 v[40:41], 3, s[14:15] // (op 54799) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 54799) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54799) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54799) 

s_lshl_b64 s[14:15], s[6:7], 1 // (op 54799) 

v_mov_b32 v32, s14 // (op 54799) 
v_mov_b32 v39, s15 // (op 54799) 
v_add_co_u32 v42, s[6:7], v28, v32 // (op 54799) least significant half
v_addc_co_u32 v43, s[6:7], v29, v39, s[6:7] // (op 54799) most significant half




s_add_u32 s1, 65536, s0 // (op 54799) 
s_mov_b32 m0, s1 // (op 54799) reg expression
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 54799) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v40 // (op 54799) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54799) most significant half

s_add_u32 m0, m0, 4096 // (op 54799) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 54799) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v40 // (op 54799) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54799) most significant half

s_add_u32 m0, m0, 4096 // (op 54799) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 54799) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v40 // (op 54799) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54799) most significant half

s_add_u32 m0, m0, 4096 // (op 54799) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 54799) Load value direct to lds






s_lshl_b32 s0, s34, 3 // (op 53815) 
v_add_u32 v32, v34, s0 // (op 53815) 


buffer_load_dword v39, v32, s[56:59], 0 offen // (op 53815) Load value

v_and_b32 v32, -8, v1 // (op 55575) 
v_sub_u32 v40, v1, v32 // (op 55575) 

v_lshlrev_b32 v32, 5, v40 // (op 55575) 

v_lshrrev_b32 v40, 3, v1 // (op 55575) 
v_lshl_add_u32 v41, v40, 8, v32 // (op 55575) 


v_lshlrev_b32 v32, 2, v41 // (op 55575) 

v_lshrrev_b32 v40, 3, v32 // (op 55575) 

s_nop 0
v_readfirstlane_b32 s0, v40 // (op 55575) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 54801) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54801) 

v_lshrrev_b64 v[40:41], 3, s[14:15] // (op 54801) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 54801) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54801) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54801) 

s_lshl_b64 s[14:15], s[6:7], 1 // (op 54801) 

v_mov_b32 v32, s14 // (op 54801) 
v_mov_b32 v42, s15 // (op 54801) 
v_add_co_u32 v44, s[6:7], v24, v32 // (op 54801) least significant half
v_addc_co_u32 v45, s[6:7], v25, v42, s[6:7] // (op 54801) most significant half




s_add_u32 s1, 81920, s0 // (op 54801) 
s_mov_b32 m0, s1 // (op 54801) reg expression
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54801) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v40 // (op 54801) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54801) most significant half

s_add_u32 m0, m0, 4096 // (op 54801) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54801) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v40 // (op 54801) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54801) most significant half

s_add_u32 m0, m0, 4096 // (op 54801) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54801) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v40 // (op 54801) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54801) most significant half

s_add_u32 m0, m0, 4096 // (op 54801) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54801) Load value direct to lds






s_lshl_b32 s0, s28, 3 // (op 53821) 
v_add_u32 v32, v22, s0 // (op 53821) 


buffer_load_dword v40, v32, s[52:55], 0 offen // (op 53821) Load value

v_and_b32 v32, -8, v1 // (op 55765) 
v_sub_u32 v41, v1, v32 // (op 55765) 

v_lshlrev_b32 v32, 5, v41 // (op 55765) 

v_lshrrev_b32 v41, 3, v1 // (op 55765) 
v_lshl_add_u32 v42, v41, 8, v32 // (op 55765) 


v_lshlrev_b32 v32, 2, v42 // (op 55765) 

v_lshrrev_b32 v41, 3, v32 // (op 55765) 

s_nop 0
v_readfirstlane_b32 s0, v41 // (op 55765) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 54803) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54803) 

v_lshrrev_b64 v[42:43], 3, s[14:15] // (op 54803) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 54803) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54803) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54803) 

s_mul_i32 s15, 3, s7 // (op 54803) most significant: low of low * high
s_mul_i32 s14, 3, s6 // (op 54803) least significant
s_mul_hi_u32 s1, 3, s6 // (op 54803) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 54803) most significant: sum


v_mov_b32 v32, s14 // (op 54803) 
v_mov_b32 v41, s15 // (op 54803) 
v_add_co_u32 v44, s[6:7], v28, v32 // (op 54803) least significant half
v_addc_co_u32 v45, s[6:7], v29, v41, s[6:7] // (op 54803) most significant half




s_add_u32 s1, 98304, s0 // (op 54803) 
s_mov_b32 m0, s1 // (op 54803) reg expression
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 54803) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54803) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54803) most significant half

s_add_u32 m0, m0, 4096 // (op 54803) 
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 54803) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54803) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54803) most significant half

s_add_u32 m0, m0, 4096 // (op 54803) 
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 54803) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54803) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54803) most significant half

s_add_u32 m0, m0, 4096 // (op 54803) 
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 54803) Load value direct to lds







s_lshl_b32 s0, s34, 2 // (op 53827) 
s_mul_i32 s1, 3, s0 // (op 53827) 

v_add_u32 v32, v34, s1 // (op 53827) 


buffer_load_dword v28, v32, s[56:59], 0 offen // (op 53827) Load value


v_and_b32 v29, -8, v1 // (op 55721) 
v_sub_u32 v32, v1, v29 // (op 55721) 

v_lshlrev_b32 v29, 5, v32 // (op 55721) 

v_lshrrev_b32 v32, 3, v1 // (op 55721) 
v_lshl_add_u32 v34, v32, 8, v29 // (op 55721) 


v_lshlrev_b32 v29, 2, v34 // (op 55721) 

v_lshrrev_b32 v32, 3, v29 // (op 55721) 

s_nop 0
v_readfirstlane_b32 s0, v32 // (op 55721) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 54805) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54805) 

v_lshrrev_b64 v[42:43], 3, s[14:15] // (op 54805) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 54805) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54805) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54805) 

s_mul_i32 s15, 3, s7 // (op 54805) most significant: low of low * high
s_mul_i32 s14, 3, s6 // (op 54805) least significant
s_mul_hi_u32 s1, 3, s6 // (op 54805) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 54805) most significant: sum


v_mov_b32 v29, s14 // (op 54805) 
v_mov_b32 v32, s15 // (op 54805) 
v_add_co_u32 v44, s[6:7], v24, v29 // (op 54805) least significant half
v_addc_co_u32 v45, s[6:7], v25, v32, s[6:7] // (op 54805) most significant half




s_add_u32 s1, 114688, s0 // (op 54805) 
s_mov_b32 m0, s1 // (op 54805) reg expression
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54805) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54805) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54805) most significant half

s_add_u32 m0, m0, 4096 // (op 54805) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54805) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54805) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54805) most significant half

s_add_u32 m0, m0, 4096 // (op 54805) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54805) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54805) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54805) most significant half

s_add_u32 m0, m0, 4096 // (op 54805) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54805) Load value direct to lds







s_lshl_b32 s0, s28, 2 // (op 53833) 
s_mul_i32 s1, 3, s0 // (op 53833) 

v_add_u32 v29, v22, s1 // (op 53833) 


buffer_load_dword v32, v29, s[52:55], 0 offen // (op 53833) Load value


v_and_b32 v22, -64, v1 // (op 55362) 
v_sub_u32 v29, v1, v22 // (op 55362) 

v_lshlrev_b32 v22, 2, v29 // (op 55362) 

v_lshrrev_b32 v29, 6, v1 // (op 55362) 
v_lshl_add_u32 v34, v29, 8, v22 // (op 55362) 





v_add_u32 v22, 131072, v34 // (op 53836) 
s_waitcnt vmcnt(35)
ds_write_b32 v22, v33 // (op 53836) Store local data 


v_and_b32 v22, -64, v1 // (op 55304) 
v_sub_u32 v29, v1, v22 // (op 55304) 

v_lshlrev_b32 v22, 2, v29 // (op 55304) 

v_lshrrev_b32 v29, 6, v1 // (op 55304) 
v_lshl_add_u32 v34, v29, 8, v22 // (op 55304) 





v_add_u32 v22, 132096, v34 // (op 53838) 
s_waitcnt vmcnt(30)
ds_write_b32 v22, v36 // (op 53838) Store local data 


s_barrier  // (op 56365) 



ds_read_b64_tr_b4 v[42:43], v4 // (op 53901) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v4 offset:2048 // (op 53901) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 56400) 


v_add_u32 v22, 131072, v5 // (op 53981) 
ds_read_u8 v29, v22 // (op 53981) Load local data 




ds_read_b64_tr_b4 v[46:47], v4 offset:16 // (op 53906) Transpose load local data 
ds_read_b64_tr_b4 v[48:49], v4 offset:2064 // (op 53906) Transpose load local data 


v_add_u32 v22, 131104, v5 // (op 53986) 
ds_read_u8 v34, v22 // (op 53986) Load local data 




ds_read_b64_tr_b4 v[50:51], v4 offset:32 // (op 53911) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:2080 // (op 53911) Transpose load local data 


v_add_u32 v22, 131136, v5 // (op 53991) 
ds_read_u8 v41, v22 // (op 53991) Load local data 




ds_read_b64_tr_b4 v[54:55], v4 offset:48 // (op 53916) Transpose load local data 
ds_read_b64_tr_b4 v[56:57], v4 offset:2096 // (op 53916) Transpose load local data 


v_add_u32 v22, 131168, v5 // (op 53996) 
ds_read_u8 v24, v22 // (op 53996) Load local data 




ds_read_b64_tr_b4 v[58:59], v4 offset:64 // (op 53921) Transpose load local data 
ds_read_b64_tr_b4 v[60:61], v4 offset:2112 // (op 53921) Transpose load local data 


v_add_u32 v22, 131200, v5 // (op 54001) 
ds_read_u8 v25, v22 // (op 54001) Load local data 




ds_read_b64_tr_b4 v[62:63], v4 offset:80 // (op 53926) Transpose load local data 
ds_read_b64_tr_b4 v[64:65], v4 offset:2128 // (op 53926) Transpose load local data 


v_add_u32 v22, 131232, v5 // (op 54006) 
ds_read_u8 v66, v22 // (op 54006) Load local data 




ds_read_b64_tr_b4 v[68:69], v4 offset:96 // (op 53931) Transpose load local data 
ds_read_b64_tr_b4 v[70:71], v4 offset:2144 // (op 53931) Transpose load local data 


v_add_u32 v22, 131264, v5 // (op 54011) 
ds_read_u8 v67, v22 // (op 54011) Load local data 




ds_read_b64_tr_b4 v[72:73], v4 offset:112 // (op 53936) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v4 offset:2160 // (op 53936) Transpose load local data 


v_add_u32 v22, 131296, v5 // (op 54016) 
ds_read_u8 v76, v22 // (op 54016) Load local data 

s_barrier  // (op 56349) 



ds_read_b64_tr_b4 v[78:79], v3 offset:16384 // (op 53861) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v3 offset:18432 // (op 53861) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 56382) 


v_add_u32 v22, 132096, v0 // (op 53941) 
ds_read_u8 v77, v22 // (op 53941) Load local data 




ds_read_b64_tr_b4 v[82:83], v3 offset:16400 // (op 53866) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v3 offset:18448 // (op 53866) Transpose load local data 


v_add_u32 v22, 132128, v0 // (op 53946) 
ds_read_u8 v86, v22 // (op 53946) Load local data 




ds_read_b64_tr_b4 v[88:89], v3 offset:16416 // (op 53871) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v3 offset:18464 // (op 53871) Transpose load local data 


v_add_u32 v22, 132160, v0 // (op 53951) 
ds_read_u8 v87, v22 // (op 53951) Load local data 




ds_read_b64_tr_b4 v[92:93], v3 offset:16432 // (op 53876) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v3 offset:18480 // (op 53876) Transpose load local data 


v_add_u32 v22, 132192, v0 // (op 53956) 
ds_read_u8 v96, v22 // (op 53956) Load local data 




ds_read_b64_tr_b4 v[98:99], v3 offset:16448 // (op 53881) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v3 offset:18496 // (op 53881) Transpose load local data 


v_add_u32 v22, 132224, v0 // (op 53961) 
ds_read_u8 v97, v22 // (op 53961) Load local data 




ds_read_b64_tr_b4 v[102:103], v3 offset:16464 // (op 53886) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v3 offset:18512 // (op 53886) Transpose load local data 


v_add_u32 v22, 132256, v0 // (op 53966) 
ds_read_u8 v106, v22 // (op 53966) Load local data 




ds_read_b64_tr_b4 v[108:109], v3 offset:16480 // (op 53891) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v3 offset:18528 // (op 53891) Transpose load local data 


v_add_u32 v22, 132288, v0 // (op 53971) 
ds_read_u8 v107, v22 // (op 53971) Load local data 




ds_read_b64_tr_b4 v[112:113], v3 offset:16496 // (op 53896) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v3 offset:18544 // (op 53896) Transpose load local data 


v_add_u32 v22, 132320, v0 // (op 53976) 
ds_read_u8 v116, v22 // (op 53976) Load local data 

v_and_b32 v22, -8, v1 // (op 55161) 
v_sub_u32 v117, v1, v22 // (op 55161) 

v_lshlrev_b32 v22, 5, v117 // (op 55161) 

v_lshrrev_b32 v117, 3, v1 // (op 55161) 
v_lshl_add_u32 v118, v117, 8, v22 // (op 55161) 


v_lshlrev_b32 v22, 2, v118 // (op 55161) 

v_lshrrev_b32 v117, 3, v22 // (op 55161) 

s_nop 0
v_readfirstlane_b32 s0, v117 // (op 55161) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 55218) 
v_sub_u32 v117, v1, v22 // (op 55218) 

v_lshlrev_b32 v22, 5, v117 // (op 55218) 

v_lshrrev_b32 v117, 3, v1 // (op 55218) 
v_lshl_add_u32 v118, v117, 8, v22 // (op 55218) 


v_lshlrev_b32 v22, 2, v118 // (op 55218) 

v_lshrrev_b32 v117, 3, v22 // (op 55218) 

s_nop 0
v_readfirstlane_b32 s1, v117 // (op 55218) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 55275) 
v_sub_u32 v117, v1, v22 // (op 55275) 

v_lshlrev_b32 v22, 2, v117 // (op 55275) 

v_lshrrev_b32 v117, 6, v1 // (op 55275) 
v_lshl_add_u32 v118, v117, 8, v22 // (op 55275) 


v_and_b32 v22, -64, v1 // (op 55333) 
v_sub_u32 v117, v1, v22 // (op 55333) 

v_lshlrev_b32 v22, 2, v117 // (op 55333) 

v_lshrrev_b32 v117, 6, v1 // (op 55333) 
v_lshl_add_u32 v119, v117, 8, v22 // (op 55333) 


v_and_b32 v22, -8, v1 // (op 55414) 
v_sub_u32 v117, v1, v22 // (op 55414) 

v_lshlrev_b32 v22, 5, v117 // (op 55414) 

v_lshrrev_b32 v117, 3, v1 // (op 55414) 
v_lshl_add_u32 v120, v117, 8, v22 // (op 55414) 


v_lshlrev_b32 v22, 2, v120 // (op 55414) 

v_lshrrev_b32 v117, 3, v22 // (op 55414) 

s_nop 0
v_readfirstlane_b32 s6, v117 // (op 55414) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 55458) 
v_sub_u32 v117, v1, v22 // (op 55458) 

v_lshlrev_b32 v22, 5, v117 // (op 55458) 

v_lshrrev_b32 v117, 3, v1 // (op 55458) 
v_lshl_add_u32 v120, v117, 8, v22 // (op 55458) 


v_lshlrev_b32 v22, 2, v120 // (op 55458) 

v_lshrrev_b32 v117, 3, v22 // (op 55458) 

s_nop 0
v_readfirstlane_b32 s7, v117 // (op 55458) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 55502) 
v_sub_u32 v117, v1, v22 // (op 55502) 

v_lshlrev_b32 v22, 2, v117 // (op 55502) 

v_lshrrev_b32 v117, 6, v1 // (op 55502) 
v_lshl_add_u32 v120, v117, 8, v22 // (op 55502) 


v_and_b32 v22, -64, v1 // (op 55531) 
v_sub_u32 v117, v1, v22 // (op 55531) 

v_lshlrev_b32 v22, 2, v117 // (op 55531) 

v_lshrrev_b32 v117, 6, v1 // (op 55531) 
v_lshl_add_u32 v121, v117, 8, v22 // (op 55531) 


v_and_b32 v22, -8, v1 // (op 55560) 
v_sub_u32 v117, v1, v22 // (op 55560) 

v_lshlrev_b32 v22, 5, v117 // (op 55560) 

v_lshrrev_b32 v117, 3, v1 // (op 55560) 
v_lshl_add_u32 v122, v117, 8, v22 // (op 55560) 


v_lshlrev_b32 v22, 2, v122 // (op 55560) 

v_lshrrev_b32 v117, 3, v22 // (op 55560) 

s_nop 0
v_readfirstlane_b32 s14, v117 // (op 55560) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 55604) 
v_sub_u32 v117, v1, v22 // (op 55604) 

v_lshlrev_b32 v22, 5, v117 // (op 55604) 

v_lshrrev_b32 v117, 3, v1 // (op 55604) 
v_lshl_add_u32 v122, v117, 8, v22 // (op 55604) 


v_lshlrev_b32 v22, 2, v122 // (op 55604) 

v_lshrrev_b32 v117, 3, v22 // (op 55604) 

s_nop 0
v_readfirstlane_b32 s15, v117 // (op 55604) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 55648) 
v_sub_u32 v117, v1, v22 // (op 55648) 

v_lshlrev_b32 v22, 2, v117 // (op 55648) 

v_lshrrev_b32 v117, 6, v1 // (op 55648) 
v_lshl_add_u32 v122, v117, 8, v22 // (op 55648) 


v_and_b32 v22, -64, v1 // (op 55677) 
v_sub_u32 v117, v1, v22 // (op 55677) 

v_lshlrev_b32 v22, 2, v117 // (op 55677) 

v_lshrrev_b32 v117, 6, v1 // (op 55677) 
v_lshl_add_u32 v123, v117, 8, v22 // (op 55677) 


v_and_b32 v22, -8, v1 // (op 55706) 
v_sub_u32 v117, v1, v22 // (op 55706) 

v_lshlrev_b32 v22, 5, v117 // (op 55706) 

v_lshrrev_b32 v117, 3, v1 // (op 55706) 
v_lshl_add_u32 v124, v117, 8, v22 // (op 55706) 


v_lshlrev_b32 v22, 2, v124 // (op 55706) 

v_lshrrev_b32 v117, 3, v22 // (op 55706) 

s_nop 0
v_readfirstlane_b32 s18, v117 // (op 55706) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 55750) 
v_sub_u32 v117, v1, v22 // (op 55750) 

v_lshlrev_b32 v22, 5, v117 // (op 55750) 

v_lshrrev_b32 v117, 3, v1 // (op 55750) 
v_lshl_add_u32 v124, v117, 8, v22 // (op 55750) 


v_lshlrev_b32 v22, 2, v124 // (op 55750) 

v_lshrrev_b32 v117, 3, v22 // (op 55750) 

s_nop 0
v_readfirstlane_b32 s19, v117 // (op 55750) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 55794) 
v_sub_u32 v117, v1, v22 // (op 55794) 

v_lshlrev_b32 v22, 2, v117 // (op 55794) 

v_lshrrev_b32 v117, 6, v1 // (op 55794) 
v_lshl_add_u32 v124, v117, 8, v22 // (op 55794) 


v_and_b32 v22, -64, v1 // (op 55823) 
v_sub_u32 v117, v1, v22 // (op 55823) 

v_lshlrev_b32 v22, 2, v117 // (op 55823) 

v_lshrrev_b32 v117, 6, v1 // (op 55823) 
v_lshl_add_u32 v125, v117, 8, v22 // (op 55823) 


s_mov_b32 s38, 0 // (op 18531) call()
s_ashr_i64 s[60:61], s[22:23], 7 // (op 18530) 
s_mov_b32 s39, s60 // (op 18530) convert

s_and_b32 s60, s39, 4294967292 // (op 18530) 

s_cmp_lt_u32 s38, s60 // (op 18530) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoop_18530 // (op 18530) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoop_18530 if false)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoop_18530:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[42:45], a[0:3], v77, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 90) 
s_barrier  // (op 56368) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[46:49], a[4:7], v77, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 526) 
s_add_u32 s38, s38, 4 // (op 18532) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[50:53], a[8:11], v77, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 824) 
s_lshl_b64 s[60:61], s[24:25], 5 // (op 54737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:81], v[54:57], a[12:15], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1122) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54737) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[58:61], a[16:19], v77, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1420) 
v_lshrrev_b64 v[126:127], 3, s[62:63] // (op 54737) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[62:65], a[20:23], v77, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1718) 
v_mov_b64 v[128:129], v[30:31] // (op 54737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[68:71], a[24:27], v77, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2016) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[72:75], a[28:31], v77, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2314) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54737) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[42:45], a[32:35], v86, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3745) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54737) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[46:49], a[36:39], v86, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3653) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54737) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v86, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3561) 
v_mov_b32 v22, s62 // (op 54737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[54:57], a[44:47], v86, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3469) 
v_mov_b32 v117, s63 // (op 54737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[82:85], v[58:61], a[48:51], v86, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3377) 
v_add_co_u32 v130, s[60:61], v128, v22 // (op 54737) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[82:85], v[62:65], a[52:55], v86, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3285) 
v_addc_co_u32 v131, s[60:61], v129, v117, s[60:61] // (op 54737) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[82:85], v[68:71], a[56:59], v86, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3193) 
s_mov_b32 m0, s1 // (op 54737) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[82:85], v[72:75], a[60:63], v86, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3101) 
buffer_load_dwordx4 v130, s[48:51], 0 offen offset: 0 lds // (op 54737) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[42:45], a[64:67], v87, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6323) 
v_add_co_u32 v130, s[60:61], v130, v126 // (op 54737) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[46:49], a[68:71], v87, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6231) 
v_addc_co_u32 v131, s[60:61], 0, v131, s[60:61] // (op 54737) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:91], v[50:53], a[72:75], v87, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6139) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[88:91], v[54:57], a[76:79], v87, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6047) 
s_add_u32 m0, m0, 4096 // (op 54737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[88:91], v[58:61], a[80:83], v87, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5955) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[88:91], v[62:65], a[84:87], v87, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5863) 
buffer_load_dwordx4 v130, s[48:51], 0 offen offset: 0 lds // (op 54737) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[88:91], v[68:71], a[88:91], v87, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5771) 
v_add_co_u32 v130, s[60:61], v130, v126 // (op 54737) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[88:91], v[72:75], a[92:95], v87, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5679) 
v_addc_co_u32 v131, s[60:61], 0, v131, s[60:61] // (op 54737) most significant half

s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[92:95], v[42:45], a[96:99], v96, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8901) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[92:95], v[46:49], a[100:103], v96, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8809) 
s_add_u32 m0, m0, 4096 // (op 54737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[92:95], v[50:53], a[104:107], v96, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8717) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:95], v[54:57], a[108:111], v96, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8625) 
buffer_load_dwordx4 v130, s[48:51], 0 offen offset: 0 lds // (op 54737) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:95], v[58:61], a[112:115], v96, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8533) 
v_add_co_u32 v130, s[60:61], v130, v126 // (op 54737) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:95], v[62:65], a[116:119], v96, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8441) 
v_addc_co_u32 v131, s[60:61], 0, v131, s[60:61] // (op 54737) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[92:95], v[68:71], a[120:123], v96, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8349) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[92:95], v[72:75], a[124:127], v96, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8257) 
s_add_u32 m0, m0, 4096 // (op 54737) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[98:101], v[42:45], a[128:131], v97, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11479) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[98:101], v[46:49], a[132:135], v97, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11387) 
buffer_load_dwordx4 v130, s[48:51], 0 offen offset: 0 lds // (op 54737) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[98:101], v[50:53], a[136:139], v97, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11295) 
v_mov_b32 v22, v35 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[98:101], v[54:57], a[140:143], v97, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11203) 
s_lshl_b32 s39, s34, 4 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[98:101], v[58:61], a[144:147], v97, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11111) 
v_add_u32 v117, v22, s39 // (op 25) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[98:101], v[62:65], a[148:151], v97, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11019) 
buffer_load_dword v33, v117, s[56:59], 0 offen // (op 25) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[98:101], v[68:71], a[152:155], v97, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10927) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 56352) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[98:101], v[72:75], a[156:159], v97, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10835) 
s_lshl_b64 s[60:61], s[12:13], 5 // (op 54732) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[102:105], v[42:45], a[160:163], v106, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14057) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54732) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[102:105], v[46:49], a[164:167], v106, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13965) 
v_lshrrev_b64 v[126:127], 3, s[62:63] // (op 54732) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[102:105], v[50:53], a[168:171], v106, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13873) 
v_mov_b64 v[130:131], v[26:27] // (op 54732) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[102:105], v[54:57], a[172:175], v106, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13781) 
s_lshl_b64 s[60:61], s[12:13], 7 // (op 54732) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[102:105], v[58:61], a[176:179], v106, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13689) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54732) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[102:105], v[62:65], a[180:183], v106, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13597) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54732) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[102:105], v[68:71], a[184:187], v106, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13505) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54732) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[102:105], v[72:75], a[188:191], v106, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13413) 
v_mov_b32 v117, s62 // (op 54732) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[108:111], v[42:45], a[192:195], v107, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16635) 
v_mov_b32 v132, s63 // (op 54732) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[108:111], v[46:49], a[196:199], v107, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16543) 
v_add_co_u32 v134, s[60:61], v130, v117 // (op 54732) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:111], v[50:53], a[200:203], v107, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16451) 
v_addc_co_u32 v135, s[60:61], v131, v132, s[60:61] // (op 54732) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[108:111], v[54:57], a[204:207], v107, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16359) 
s_add_u32 s39, 16384, s0 // (op 54732) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[108:111], v[58:61], a[208:211], v107, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16267) 
s_mov_b32 m0, s39 // (op 54732) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[108:111], v[62:65], a[212:215], v107, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16175) 
buffer_load_dwordx4 v134, s[8:11], 0 offen offset: 0 lds // (op 54732) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[108:111], v[68:71], a[216:219], v107, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16083) 
v_add_co_u32 v134, s[60:61], v134, v126 // (op 54732) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[108:111], v[72:75], a[220:223], v107, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15991) 
v_addc_co_u32 v135, s[60:61], 0, v135, s[60:61] // (op 54732) most significant half

s_add_u32 m0, m0, 4096 // (op 54732) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:115], v[42:45], a[224:227], v116, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19213) 
buffer_load_dwordx4 v134, s[8:11], 0 offen offset: 0 lds // (op 54732) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[112:115], v[46:49], a[228:231], v116, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19121) 
v_add_co_u32 v134, s[60:61], v134, v126 // (op 54732) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[112:115], v[50:53], a[232:235], v116, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19029) 
v_addc_co_u32 v135, s[60:61], 0, v135, s[60:61] // (op 54732) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[112:115], v[54:57], a[236:239], v116, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18937) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[112:115], v[58:61], a[240:243], v116, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18845) 
s_add_u32 m0, m0, 4096 // (op 54732) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[112:115], v[62:65], a[244:247], v116, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18753) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[112:115], v[68:71], a[248:251], v116, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18661) 
buffer_load_dwordx4 v134, s[8:11], 0 offen offset: 0 lds // (op 54732) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[112:115], v[72:75], a[252:255], v116, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18569) 
v_add_co_u32 v134, s[60:61], v134, v126 // (op 54732) least significant half
v_addc_co_u32 v135, s[60:61], 0, v135, s[60:61] // (op 54732) most significant half

s_add_u32 m0, m0, 4096 // (op 54732) 
buffer_load_dwordx4 v134, s[8:11], 0 offen offset: 0 lds // (op 54732) Load value direct to lds





v_mov_b32 v117, v23 // (op 17) 
s_lshl_b32 s39, s28, 4 // (op 17) 
v_add_u32 v126, v117, s39 // (op 17) 


buffer_load_dword v36, v126, s[52:55], 0 offen // (op 17) Load value

s_barrier  // (op 56440) 



v_add_u32 v126, 133120, v121 // (op 259) 
s_waitcnt vmcnt(35)
ds_write_b32 v126, v37 // (op 259) Store local data 

s_barrier  // (op 56434) 



v_add_u32 v126, 134144, v120 // (op 258) 
s_waitcnt vmcnt(30)
ds_write_b32 v126, v38 // (op 258) Store local data 

s_barrier  // (op 56423) 



ds_read_b64_tr_b4 v[132:133], v8 offset:32768 // (op 251) Transpose load local data 
ds_read_b64_tr_b4 v[134:135], v8 offset:34816 // (op 251) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 56437) 


v_add_u32 v126, 133120, v9 // (op 255) 
ds_read_u8 v127, v126 // (op 255) Load local data 




ds_read_b64_tr_b4 v[136:137], v8 offset:32784 // (op 533) Transpose load local data 
ds_read_b64_tr_b4 v[138:139], v8 offset:34832 // (op 533) Transpose load local data 


v_add_u32 v126, 133152, v9 // (op 537) 
ds_read_u8 v86, v126 // (op 537) Load local data 




ds_read_b64_tr_b4 v[82:83], v8 offset:32800 // (op 831) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v8 offset:34848 // (op 831) Transpose load local data 


v_add_u32 v126, 133184, v9 // (op 835) 
ds_read_u8 v87, v126 // (op 835) Load local data 




ds_read_b64_tr_b4 v[88:89], v8 offset:32816 // (op 1129) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v8 offset:34864 // (op 1129) Transpose load local data 


v_add_u32 v126, 133216, v9 // (op 1133) 
ds_read_u8 v96, v126 // (op 1133) Load local data 




ds_read_b64_tr_b4 v[92:93], v8 offset:32832 // (op 1427) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v8 offset:34880 // (op 1427) Transpose load local data 


v_add_u32 v126, 133248, v9 // (op 1431) 
ds_read_u8 v97, v126 // (op 1431) Load local data 




ds_read_b64_tr_b4 v[98:99], v8 offset:32848 // (op 1725) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v8 offset:34896 // (op 1725) Transpose load local data 


v_add_u32 v126, 133280, v9 // (op 1729) 
ds_read_u8 v106, v126 // (op 1729) Load local data 




ds_read_b64_tr_b4 v[102:103], v8 offset:32864 // (op 2023) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v8 offset:34912 // (op 2023) Transpose load local data 


v_add_u32 v126, 133312, v9 // (op 2027) 
ds_read_u8 v107, v126 // (op 2027) Load local data 




ds_read_b64_tr_b4 v[108:109], v8 offset:32880 // (op 2321) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v8 offset:34928 // (op 2321) Transpose load local data 


v_add_u32 v126, 133344, v9 // (op 2325) 
ds_read_u8 v140, v126 // (op 2325) Load local data 

s_barrier  // (op 56417) 
s_barrier  // (op 56549) 



ds_read_b64_tr_b4 v[78:79], v7 offset:49152 // (op 249) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v7 offset:51200 // (op 249) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 56430) 


v_add_u32 v126, 134144, v6 // (op 253) 
ds_read_u8 v77, v126 // (op 253) Load local data 




ds_read_b64_tr_b4 v[142:143], v7 offset:49168 // (op 3106) Transpose load local data 
ds_read_b64_tr_b4 v[144:145], v7 offset:51216 // (op 3106) Transpose load local data 


v_add_u32 v126, 134176, v6 // (op 3109) 
ds_read_u8 v34, v126 // (op 3109) Load local data 




ds_read_b64_tr_b4 v[146:147], v7 offset:49184 // (op 5684) Transpose load local data 
ds_read_b64_tr_b4 v[148:149], v7 offset:51232 // (op 5684) Transpose load local data 


v_add_u32 v126, 134208, v6 // (op 5687) 
ds_read_u8 v41, v126 // (op 5687) Load local data 




ds_read_b64_tr_b4 v[150:151], v7 offset:49200 // (op 8262) Transpose load local data 
ds_read_b64_tr_b4 v[152:153], v7 offset:51248 // (op 8262) Transpose load local data 


v_add_u32 v126, 134240, v6 // (op 8265) 
ds_read_u8 v24, v126 // (op 8265) Load local data 




ds_read_b64_tr_b4 v[154:155], v7 offset:49216 // (op 10840) Transpose load local data 
ds_read_b64_tr_b4 v[156:157], v7 offset:51264 // (op 10840) Transpose load local data 


v_add_u32 v126, 134272, v6 // (op 10843) 
ds_read_u8 v25, v126 // (op 10843) Load local data 




ds_read_b64_tr_b4 v[158:159], v7 offset:49232 // (op 13418) Transpose load local data 
ds_read_b64_tr_b4 v[160:161], v7 offset:51280 // (op 13418) Transpose load local data 


v_add_u32 v126, 134304, v6 // (op 13421) 
ds_read_u8 v66, v126 // (op 13421) Load local data 




ds_read_b64_tr_b4 v[162:163], v7 offset:49248 // (op 15996) Transpose load local data 
ds_read_b64_tr_b4 v[164:165], v7 offset:51296 // (op 15996) Transpose load local data 


v_add_u32 v126, 134336, v6 // (op 15999) 
ds_read_u8 v67, v126 // (op 15999) Load local data 




ds_read_b64_tr_b4 v[166:167], v7 offset:49264 // (op 18574) Transpose load local data 
ds_read_b64_tr_b4 v[168:169], v7 offset:51312 // (op 18574) Transpose load local data 


v_add_u32 v126, 134368, v6 // (op 18577) 
ds_read_u8 v76, v126 // (op 18577) Load local data 

s_barrier  // (op 56647) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[132:135], a[0:3], v77, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 256) 
s_barrier  // (op 56427) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[136:139], a[4:7], v77, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 527) 
s_lshl_b64 s[60:61], s[24:25], 5 // (op 54760) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[82:85], a[8:11], v77, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 825) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54760) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:81], v[88:91], a[12:15], v77, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1123) 
v_lshrrev_b64 v[170:171], 3, s[62:63] // (op 54760) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[92:95], a[16:19], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1421) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54760) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[98:101], a[20:23], v77, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1719) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54760) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[102:105], a[24:27], v77, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2017) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54760) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[108:111], a[28:31], v77, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2315) 
s_mul_i32 s63, 5, s61 // (op 54760) most significant: low of low * high
s_mul_i32 s62, 5, s60 // (op 54760) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[142:145], v[132:135], a[32:35], v34, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3746) 
s_mul_hi_u32 s39, 5, s60 // (op 54760) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[142:145], v[136:139], a[36:39], v34, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3654) 
s_add_u32 s63, s63, s39 // (op 54760) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[142:145], v[82:85], a[40:43], v34, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3562) 
v_mov_b32 v126, s62 // (op 54760) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[142:145], v[88:91], a[44:47], v34, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3470) 
v_mov_b32 v141, s63 // (op 54760) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[142:145], v[92:95], a[48:51], v34, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3378) 
v_add_co_u32 v172, s[60:61], v128, v126 // (op 54760) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[142:145], v[98:101], a[52:55], v34, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3286) 
v_addc_co_u32 v173, s[60:61], v129, v141, s[60:61] // (op 54760) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[142:145], v[102:105], a[56:59], v34, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3194) 
s_add_u32 s39, 32768, s7 // (op 54760) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[142:145], v[108:111], a[60:63], v34, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3102) 
s_mov_b32 m0, s39 // (op 54760) reg expression

buffer_load_dwordx4 v172, s[48:51], 0 offen offset: 0 lds // (op 54760) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[146:149], v[132:135], a[64:67], v41, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6324) 
v_add_co_u32 v172, s[60:61], v172, v170 // (op 54760) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[146:149], v[136:139], a[68:71], v41, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6232) 
v_addc_co_u32 v173, s[60:61], 0, v173, s[60:61] // (op 54760) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[146:149], v[82:85], a[72:75], v41, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6140) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[146:149], v[88:91], a[76:79], v41, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6048) 
s_add_u32 m0, m0, 4096 // (op 54760) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[146:149], v[92:95], a[80:83], v41, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5956) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[146:149], v[98:101], a[84:87], v41, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5864) 
buffer_load_dwordx4 v172, s[48:51], 0 offen offset: 0 lds // (op 54760) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[146:149], v[102:105], a[88:91], v41, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5772) 
v_add_co_u32 v172, s[60:61], v172, v170 // (op 54760) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[146:149], v[108:111], a[92:95], v41, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5680) 
v_addc_co_u32 v173, s[60:61], 0, v173, s[60:61] // (op 54760) most significant half


s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[150:153], v[132:135], a[96:99], v24, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8902) 
s_add_u32 m0, m0, 4096 // (op 54760) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[150:153], v[136:139], a[100:103], v24, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8810) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[150:153], v[82:85], a[104:107], v24, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8718) 
buffer_load_dwordx4 v172, s[48:51], 0 offen offset: 0 lds // (op 54760) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[150:153], v[88:91], a[108:111], v24, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8626) 
v_add_co_u32 v172, s[60:61], v172, v170 // (op 54760) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[150:153], v[92:95], a[112:115], v24, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8534) 
v_addc_co_u32 v173, s[60:61], 0, v173, s[60:61] // (op 54760) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[150:153], v[98:101], a[116:119], v24, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8442) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[150:153], v[102:105], a[120:123], v24, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8350) 
s_add_u32 m0, m0, 4096 // (op 54760) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[150:153], v[108:111], a[124:127], v24, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8258) 

buffer_load_dwordx4 v172, s[48:51], 0 offen offset: 0 lds // (op 54760) Load value direct to lds





s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[154:157], v[132:135], a[128:131], v25, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11480) 
s_lshl_b32 s39, s34, 2 // (op 262) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[154:157], v[136:139], a[132:135], v25, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11388) 
s_mul_i32 s60, 5, s39 // (op 262) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[154:157], v[82:85], a[136:139], v25, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11296) 
v_add_u32 v126, v22, s60 // (op 262) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:157], v[88:91], a[140:143], v25, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11204) 
buffer_load_dword v37, v126, s[56:59], 0 offen // (op 262) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[154:157], v[92:95], a[144:147], v25, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11112) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 56420) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[154:157], v[98:101], a[148:151], v25, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11020) 
s_lshl_b64 s[60:61], s[12:13], 5 // (op 54755) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[154:157], v[102:105], a[152:155], v25, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10928) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54755) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[154:157], v[108:111], a[156:159], v25, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10836) 
v_lshrrev_b64 v[152:153], 3, s[62:63] // (op 54755) 



s_lshl_b64 s[60:61], s[12:13], 7 // (op 54755) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[158:161], v[132:135], a[160:163], v66, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14058) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54755) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[158:161], v[136:139], a[164:167], v66, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13966) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54755) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[158:161], v[82:85], a[168:171], v66, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13874) 
s_mul_i32 s63, 5, s61 // (op 54755) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[158:161], v[88:91], a[172:175], v66, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13782) 
s_mul_i32 s62, 5, s60 // (op 54755) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[158:161], v[92:95], a[176:179], v66, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13690) 
s_mul_hi_u32 s39, 5, s60 // (op 54755) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[158:161], v[98:101], a[180:183], v66, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13598) 
s_add_u32 s63, s63, s39 // (op 54755) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[158:161], v[102:105], a[184:187], v66, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13506) 
v_mov_b32 v126, s62 // (op 54755) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[158:161], v[108:111], a[188:191], v66, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13414) 
v_mov_b32 v141, s63 // (op 54755) 

v_add_co_u32 v142, s[60:61], v130, v126 // (op 54755) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[162:165], v[132:135], a[192:195], v67, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16636) 
v_addc_co_u32 v143, s[60:61], v131, v141, s[60:61] // (op 54755) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[162:165], v[136:139], a[196:199], v67, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16544) 
s_add_u32 s39, 49152, s6 // (op 54755) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[162:165], v[82:85], a[200:203], v67, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16452) 
s_mov_b32 m0, s39 // (op 54755) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[162:165], v[88:91], a[204:207], v67, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16360) 
buffer_load_dwordx4 v142, s[8:11], 0 offen offset: 0 lds // (op 54755) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[162:165], v[92:95], a[208:211], v67, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16268) 
v_add_co_u32 v142, s[60:61], v142, v152 // (op 54755) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[162:165], v[98:101], a[212:215], v67, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16176) 
v_addc_co_u32 v143, s[60:61], 0, v143, s[60:61] // (op 54755) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[162:165], v[102:105], a[216:219], v67, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16084) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[162:165], v[108:111], a[220:223], v67, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15992) 
s_add_u32 m0, m0, 4096 // (op 54755) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[166:169], v[132:135], a[224:227], v76, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19214) 
buffer_load_dwordx4 v142, s[8:11], 0 offen offset: 0 lds // (op 54755) Load value direct to lds


v_add_co_u32 v142, s[60:61], v142, v152 // (op 54755) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[166:169], v[136:139], a[228:231], v76, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19122) 
v_addc_co_u32 v143, s[60:61], 0, v143, s[60:61] // (op 54755) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[166:169], v[82:85], a[232:235], v76, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19030) 
s_add_u32 m0, m0, 4096 // (op 54755) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[166:169], v[88:91], a[236:239], v76, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18938) 
buffer_load_dwordx4 v142, s[8:11], 0 offen offset: 0 lds // (op 54755) Load value direct to lds
v_add_co_u32 v142, s[60:61], v142, v152 // (op 54755) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[166:169], v[92:95], a[240:243], v76, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18846) 
v_addc_co_u32 v143, s[60:61], 0, v143, s[60:61] // (op 54755) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:169], v[98:101], a[244:247], v76, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18754) 
s_add_u32 m0, m0, 4096 // (op 54755) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[166:169], v[102:105], a[248:251], v76, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18662) 
buffer_load_dwordx4 v142, s[8:11], 0 offen offset: 0 lds // (op 54755) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[166:169], v[108:111], a[252:255], v76, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18570) 
s_lshl_b32 s39, s28, 2 // (op 261) 


s_mul_i32 s60, 5, s39 // (op 261) 

v_add_u32 v126, v117, s60 // (op 261) 


buffer_load_dword v38, v126, s[52:55], 0 offen // (op 261) Load value

s_barrier  // (op 56466) 



v_add_u32 v126, 133120, v123 // (op 299) 
s_waitcnt vmcnt(35)
ds_write_b32 v126, v39 // (op 299) Store local data 

s_barrier  // (op 56460) 



v_add_u32 v126, 134144, v122 // (op 298) 
s_waitcnt vmcnt(30)
ds_write_b32 v126, v40 // (op 298) Store local data 

s_barrier  // (op 56449) 



v_add_u32 v126, 65536, v12 // (op 291) 
ds_read_b64_tr_b4 v[42:43], v126 // (op 291) Transpose load local data 

v_add_u32 v126, 67584, v12 // (op 291) 
ds_read_b64_tr_b4 v[44:45], v126 // (op 291) Transpose load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 56463) 


v_add_u32 v126, 133120, v13 // (op 295) 
ds_read_u8 v29, v126 // (op 295) Load local data 




v_add_u32 v126, 65552, v12 // (op 541) 
ds_read_b64_tr_b4 v[46:47], v126 // (op 541) Transpose load local data 

v_add_u32 v126, 67600, v12 // (op 541) 
ds_read_b64_tr_b4 v[48:49], v126 // (op 541) Transpose load local data 



v_add_u32 v126, 133152, v13 // (op 545) 
ds_read_u8 v34, v126 // (op 545) Load local data 




v_add_u32 v126, 65568, v12 // (op 839) 
ds_read_b64_tr_b4 v[50:51], v126 // (op 839) Transpose load local data 

v_add_u32 v126, 67616, v12 // (op 839) 
ds_read_b64_tr_b4 v[52:53], v126 // (op 839) Transpose load local data 



v_add_u32 v126, 133184, v13 // (op 843) 
ds_read_u8 v41, v126 // (op 843) Load local data 




v_add_u32 v126, 65584, v12 // (op 1137) 
ds_read_b64_tr_b4 v[54:55], v126 // (op 1137) Transpose load local data 

v_add_u32 v126, 67632, v12 // (op 1137) 
ds_read_b64_tr_b4 v[56:57], v126 // (op 1137) Transpose load local data 



v_add_u32 v126, 133216, v13 // (op 1141) 
ds_read_u8 v24, v126 // (op 1141) Load local data 




v_add_u32 v126, 65600, v12 // (op 1435) 
ds_read_b64_tr_b4 v[58:59], v126 // (op 1435) Transpose load local data 

v_add_u32 v126, 67648, v12 // (op 1435) 
ds_read_b64_tr_b4 v[60:61], v126 // (op 1435) Transpose load local data 



v_add_u32 v126, 133248, v13 // (op 1439) 
ds_read_u8 v25, v126 // (op 1439) Load local data 




v_add_u32 v126, 65616, v12 // (op 1733) 
ds_read_b64_tr_b4 v[62:63], v126 // (op 1733) Transpose load local data 

v_add_u32 v126, 67664, v12 // (op 1733) 
ds_read_b64_tr_b4 v[64:65], v126 // (op 1733) Transpose load local data 



v_add_u32 v126, 133280, v13 // (op 1737) 
ds_read_u8 v66, v126 // (op 1737) Load local data 




v_add_u32 v126, 65632, v12 // (op 2031) 
ds_read_b64_tr_b4 v[68:69], v126 // (op 2031) Transpose load local data 

v_add_u32 v126, 67680, v12 // (op 2031) 
ds_read_b64_tr_b4 v[70:71], v126 // (op 2031) Transpose load local data 



v_add_u32 v126, 133312, v13 // (op 2035) 
ds_read_u8 v67, v126 // (op 2035) Load local data 




v_add_u32 v126, 65648, v12 // (op 2329) 
ds_read_b64_tr_b4 v[72:73], v126 // (op 2329) Transpose load local data 

v_add_u32 v126, 67696, v12 // (op 2329) 
ds_read_b64_tr_b4 v[74:75], v126 // (op 2329) Transpose load local data 



v_add_u32 v126, 133344, v13 // (op 2333) 
ds_read_u8 v116, v126 // (op 2333) Load local data 

s_barrier  // (op 56443) 
s_barrier  // (op 56555) 



v_add_u32 v126, 81920, v11 // (op 289) 
ds_read_b64_tr_b4 v[78:79], v126 // (op 289) Transpose load local data 

v_add_u32 v126, 83968, v11 // (op 289) 
ds_read_b64_tr_b4 v[80:81], v126 // (op 289) Transpose load local data 

s_waitcnt lgkmcnt(15)
s_barrier  // (op 56456) 


v_add_u32 v126, 134144, v10 // (op 293) 
ds_read_u8 v77, v126 // (op 293) Load local data 




v_add_u32 v126, 81936, v11 // (op 3112) 
ds_read_b64_tr_b4 v[112:113], v126 // (op 3112) Transpose load local data 

v_add_u32 v126, 83984, v11 // (op 3112) 
ds_read_b64_tr_b4 v[114:115], v126 // (op 3112) Transpose load local data 



v_add_u32 v126, 134176, v10 // (op 3115) 
ds_read_u8 v86, v126 // (op 3115) Load local data 




v_add_u32 v126, 81952, v11 // (op 5690) 
ds_read_b64_tr_b4 v[132:133], v126 // (op 5690) Transpose load local data 

v_add_u32 v126, 84000, v11 // (op 5690) 
ds_read_b64_tr_b4 v[134:135], v126 // (op 5690) Transpose load local data 



v_add_u32 v126, 134208, v10 // (op 5693) 
ds_read_u8 v87, v126 // (op 5693) Load local data 




v_add_u32 v126, 81968, v11 // (op 8268) 
ds_read_b64_tr_b4 v[136:137], v126 // (op 8268) Transpose load local data 

v_add_u32 v126, 84016, v11 // (op 8268) 
ds_read_b64_tr_b4 v[138:139], v126 // (op 8268) Transpose load local data 



v_add_u32 v126, 134240, v10 // (op 8271) 
ds_read_u8 v96, v126 // (op 8271) Load local data 




v_add_u32 v126, 81984, v11 // (op 10846) 
ds_read_b64_tr_b4 v[140:141], v126 // (op 10846) Transpose load local data 

v_add_u32 v126, 84032, v11 // (op 10846) 
ds_read_b64_tr_b4 v[142:143], v126 // (op 10846) Transpose load local data 



v_add_u32 v126, 134272, v10 // (op 10849) 
ds_read_u8 v97, v126 // (op 10849) Load local data 




v_add_u32 v126, 82000, v11 // (op 13424) 
ds_read_b64_tr_b4 v[144:145], v126 // (op 13424) Transpose load local data 

v_add_u32 v126, 84048, v11 // (op 13424) 
ds_read_b64_tr_b4 v[146:147], v126 // (op 13424) Transpose load local data 



v_add_u32 v126, 134304, v10 // (op 13427) 
ds_read_u8 v106, v126 // (op 13427) Load local data 




v_add_u32 v126, 82016, v11 // (op 16002) 
ds_read_b64_tr_b4 v[148:149], v126 // (op 16002) Transpose load local data 

v_add_u32 v126, 84064, v11 // (op 16002) 
ds_read_b64_tr_b4 v[150:151], v126 // (op 16002) Transpose load local data 



v_add_u32 v126, 134336, v10 // (op 16005) 
ds_read_u8 v107, v126 // (op 16005) Load local data 




v_add_u32 v126, 82032, v11 // (op 18580) 
ds_read_b64_tr_b4 v[152:153], v126 // (op 18580) Transpose load local data 

v_add_u32 v126, 84080, v11 // (op 18580) 
ds_read_b64_tr_b4 v[154:155], v126 // (op 18580) Transpose load local data 



v_add_u32 v126, 134368, v10 // (op 18583) 
ds_read_u8 v127, v126 // (op 18583) Load local data 

s_barrier  // (op 56653) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[42:45], a[0:3], v77, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 296) 
s_barrier  // (op 56453) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[46:49], a[4:7], v77, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 528) 
s_lshl_b64 s[60:61], s[24:25], 5 // (op 54770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[50:53], a[8:11], v77, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 826) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54770) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:81], v[54:57], a[12:15], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1124) 
v_lshrrev_b64 v[156:157], 3, s[62:63] // (op 54770) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[58:61], a[16:19], v77, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1422) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[62:65], a[20:23], v77, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1720) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54770) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[68:71], a[24:27], v77, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2018) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54770) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[72:75], a[28:31], v77, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2316) 
s_mul_i32 s63, 6, s61 // (op 54770) most significant: low of low * high
s_mul_i32 s62, 6, s60 // (op 54770) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[42:45], a[32:35], v86, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3747) 
s_mul_hi_u32 s39, 6, s60 // (op 54770) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[46:49], a[36:39], v86, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3655) 
s_add_u32 s63, s63, s39 // (op 54770) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[50:53], a[40:43], v86, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3563) 
v_mov_b32 v126, s62 // (op 54770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[54:57], a[44:47], v86, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3471) 
v_mov_b32 v158, s63 // (op 54770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[58:61], a[48:51], v86, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3379) 
v_add_co_u32 v160, s[60:61], v128, v126 // (op 54770) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[62:65], a[52:55], v86, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3287) 
v_addc_co_u32 v161, s[60:61], v129, v158, s[60:61] // (op 54770) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[112:115], v[68:71], a[56:59], v86, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3195) 
s_add_u32 s39, 65536, s15 // (op 54770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[112:115], v[72:75], a[60:63], v86, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3103) 
s_mov_b32 m0, s39 // (op 54770) reg expression
buffer_load_dwordx4 v160, s[48:51], 0 offen offset: 0 lds // (op 54770) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[132:135], v[42:45], a[64:67], v87, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6325) 
v_add_co_u32 v160, s[60:61], v160, v156 // (op 54770) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[46:49], a[68:71], v87, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6233) 
v_addc_co_u32 v161, s[60:61], 0, v161, s[60:61] // (op 54770) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[132:135], v[50:53], a[72:75], v87, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6141) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[132:135], v[54:57], a[76:79], v87, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6049) 
s_add_u32 m0, m0, 4096 // (op 54770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[132:135], v[58:61], a[80:83], v87, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5957) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[62:65], a[84:87], v87, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5865) 
buffer_load_dwordx4 v160, s[48:51], 0 offen offset: 0 lds // (op 54770) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[68:71], a[88:91], v87, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5773) 
v_add_co_u32 v160, s[60:61], v160, v156 // (op 54770) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[72:75], a[92:95], v87, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5681) 
v_addc_co_u32 v161, s[60:61], 0, v161, s[60:61] // (op 54770) most significant half


s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[42:45], a[96:99], v96, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8903) 
s_add_u32 m0, m0, 4096 // (op 54770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[46:49], a[100:103], v96, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8811) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[50:53], a[104:107], v96, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8719) 
buffer_load_dwordx4 v160, s[48:51], 0 offen offset: 0 lds // (op 54770) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[136:139], v[54:57], a[108:111], v96, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8627) 
v_add_co_u32 v160, s[60:61], v160, v156 // (op 54770) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[58:61], a[112:115], v96, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8535) 
v_addc_co_u32 v161, s[60:61], 0, v161, s[60:61] // (op 54770) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[62:65], a[116:119], v96, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8443) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[68:71], a[120:123], v96, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8351) 
s_add_u32 m0, m0, 4096 // (op 54770) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[72:75], a[124:127], v96, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8259) 

buffer_load_dwordx4 v160, s[48:51], 0 offen offset: 0 lds // (op 54770) Load value direct to lds





s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[140:143], v[42:45], a[128:131], v97, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11481) 
s_lshl_b32 s39, s34, 2 // (op 302) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[140:143], v[46:49], a[132:135], v97, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11389) 
s_mul_i32 s60, 6, s39 // (op 302) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[140:143], v[50:53], a[136:139], v97, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11297) 
v_add_u32 v126, v22, s60 // (op 302) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[140:143], v[54:57], a[140:143], v97, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11205) 
buffer_load_dword v39, v126, s[56:59], 0 offen // (op 302) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[140:143], v[58:61], a[144:147], v97, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11113) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 56446) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[140:143], v[62:65], a[148:151], v97, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11021) 
s_lshl_b64 s[60:61], s[12:13], 5 // (op 54765) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[140:143], v[68:71], a[152:155], v97, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10929) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54765) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[140:143], v[72:75], a[156:159], v97, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10837) 
v_lshrrev_b64 v[132:133], 3, s[62:63] // (op 54765) 



s_lshl_b64 s[60:61], s[12:13], 7 // (op 54765) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[42:45], a[160:163], v106, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14059) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54765) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[144:147], v[46:49], a[164:167], v106, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13967) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54765) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[50:53], a[168:171], v106, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13875) 
s_mul_i32 s63, 6, s61 // (op 54765) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[54:57], a[172:175], v106, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13783) 
s_mul_i32 s62, 6, s60 // (op 54765) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[144:147], v[58:61], a[176:179], v106, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13691) 
s_mul_hi_u32 s39, 6, s60 // (op 54765) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[144:147], v[62:65], a[180:183], v106, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13599) 
s_add_u32 s63, s63, s39 // (op 54765) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:147], v[68:71], a[184:187], v106, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13507) 
v_mov_b32 v126, s62 // (op 54765) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[144:147], v[72:75], a[188:191], v106, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13415) 
v_mov_b32 v134, s63 // (op 54765) 

v_add_co_u32 v146, s[60:61], v130, v126 // (op 54765) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[148:151], v[42:45], a[192:195], v107, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16637) 
v_addc_co_u32 v147, s[60:61], v131, v134, s[60:61] // (op 54765) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[148:151], v[46:49], a[196:199], v107, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16545) 
s_add_u32 s39, 81920, s14 // (op 54765) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:151], v[50:53], a[200:203], v107, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16453) 
s_mov_b32 m0, s39 // (op 54765) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[148:151], v[54:57], a[204:207], v107, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16361) 
buffer_load_dwordx4 v146, s[8:11], 0 offen offset: 0 lds // (op 54765) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[148:151], v[58:61], a[208:211], v107, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16269) 
v_add_co_u32 v146, s[60:61], v146, v132 // (op 54765) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[148:151], v[62:65], a[212:215], v107, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16177) 
v_addc_co_u32 v147, s[60:61], 0, v147, s[60:61] // (op 54765) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[148:151], v[68:71], a[216:219], v107, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16085) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[148:151], v[72:75], a[220:223], v107, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15993) 
s_add_u32 m0, m0, 4096 // (op 54765) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[152:155], v[42:45], a[224:227], v127, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19215) 
buffer_load_dwordx4 v146, s[8:11], 0 offen offset: 0 lds // (op 54765) Load value direct to lds
v_add_co_u32 v146, s[60:61], v146, v132 // (op 54765) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[152:155], v[46:49], a[228:231], v127, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19123) 
v_addc_co_u32 v147, s[60:61], 0, v147, s[60:61] // (op 54765) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[152:155], v[50:53], a[232:235], v127, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19031) 
s_add_u32 m0, m0, 4096 // (op 54765) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[152:155], v[54:57], a[236:239], v127, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18939) 
buffer_load_dwordx4 v146, s[8:11], 0 offen offset: 0 lds // (op 54765) Load value direct to lds
v_add_co_u32 v146, s[60:61], v146, v132 // (op 54765) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[152:155], v[58:61], a[240:243], v127, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18847) 
v_addc_co_u32 v147, s[60:61], 0, v147, s[60:61] // (op 54765) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[152:155], v[62:65], a[244:247], v127, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18755) 
s_add_u32 m0, m0, 4096 // (op 54765) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[152:155], v[68:71], a[248:251], v127, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18663) 
buffer_load_dwordx4 v146, s[8:11], 0 offen offset: 0 lds // (op 54765) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[152:155], v[72:75], a[252:255], v127, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18571) 
s_lshl_b32 s39, s28, 2 // (op 301) 


s_mul_i32 s60, 6, s39 // (op 301) 

v_add_u32 v126, v117, s60 // (op 301) 


buffer_load_dword v40, v126, s[52:55], 0 offen // (op 301) Load value

s_barrier  // (op 56492) 



v_add_u32 v126, 133120, v125 // (op 339) 
s_waitcnt vmcnt(35)
ds_write_b32 v126, v28 // (op 339) Store local data 

s_barrier  // (op 56486) 



v_add_u32 v126, 134144, v124 // (op 338) 
s_waitcnt vmcnt(30)
ds_write_b32 v126, v32 // (op 338) Store local data 

s_barrier  // (op 56475) 



v_add_u32 v126, 98304, v16 // (op 331) 
ds_read_b64_tr_b4 v[82:83], v126 // (op 331) Transpose load local data 

v_add_u32 v126, 100352, v16 // (op 331) 
ds_read_b64_tr_b4 v[84:85], v126 // (op 331) Transpose load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 56489) 


v_add_u32 v126, 133120, v17 // (op 335) 
ds_read_u8 v76, v126 // (op 335) Load local data 




v_add_u32 v126, 98320, v16 // (op 549) 
ds_read_b64_tr_b4 v[88:89], v126 // (op 549) Transpose load local data 

v_add_u32 v126, 100368, v16 // (op 549) 
ds_read_b64_tr_b4 v[90:91], v126 // (op 549) Transpose load local data 



v_add_u32 v126, 133152, v17 // (op 553) 
ds_read_u8 v86, v126 // (op 553) Load local data 




v_add_u32 v126, 98336, v16 // (op 847) 
ds_read_b64_tr_b4 v[92:93], v126 // (op 847) Transpose load local data 

v_add_u32 v126, 100384, v16 // (op 847) 
ds_read_b64_tr_b4 v[94:95], v126 // (op 847) Transpose load local data 



v_add_u32 v126, 133184, v17 // (op 851) 
ds_read_u8 v87, v126 // (op 851) Load local data 




v_add_u32 v126, 98352, v16 // (op 1145) 
ds_read_b64_tr_b4 v[98:99], v126 // (op 1145) Transpose load local data 

v_add_u32 v126, 100400, v16 // (op 1145) 
ds_read_b64_tr_b4 v[100:101], v126 // (op 1145) Transpose load local data 



v_add_u32 v126, 133216, v17 // (op 1149) 
ds_read_u8 v96, v126 // (op 1149) Load local data 




v_add_u32 v126, 98368, v16 // (op 1443) 
ds_read_b64_tr_b4 v[102:103], v126 // (op 1443) Transpose load local data 

v_add_u32 v126, 100416, v16 // (op 1443) 
ds_read_b64_tr_b4 v[104:105], v126 // (op 1443) Transpose load local data 



v_add_u32 v126, 133248, v17 // (op 1447) 
ds_read_u8 v97, v126 // (op 1447) Load local data 




v_add_u32 v126, 98384, v16 // (op 1741) 
ds_read_b64_tr_b4 v[108:109], v126 // (op 1741) Transpose load local data 

v_add_u32 v126, 100432, v16 // (op 1741) 
ds_read_b64_tr_b4 v[110:111], v126 // (op 1741) Transpose load local data 



v_add_u32 v126, 133280, v17 // (op 1745) 
ds_read_u8 v106, v126 // (op 1745) Load local data 




v_add_u32 v126, 98400, v16 // (op 2039) 
ds_read_b64_tr_b4 v[112:113], v126 // (op 2039) Transpose load local data 

v_add_u32 v126, 100448, v16 // (op 2039) 
ds_read_b64_tr_b4 v[114:115], v126 // (op 2039) Transpose load local data 



v_add_u32 v126, 133312, v17 // (op 2043) 
ds_read_u8 v107, v126 // (op 2043) Load local data 




v_add_u32 v126, 98416, v16 // (op 2337) 
ds_read_b64_tr_b4 v[132:133], v126 // (op 2337) Transpose load local data 

v_add_u32 v126, 100464, v16 // (op 2337) 
ds_read_b64_tr_b4 v[134:135], v126 // (op 2337) Transpose load local data 



v_add_u32 v126, 133344, v17 // (op 2341) 
ds_read_u8 v127, v126 // (op 2341) Load local data 

s_barrier  // (op 56469) 
s_barrier  // (op 56561) 



v_add_u32 v126, 114688, v15 // (op 329) 
ds_read_b64_tr_b4 v[78:79], v126 // (op 329) Transpose load local data 

v_add_u32 v126, 116736, v15 // (op 329) 
ds_read_b64_tr_b4 v[80:81], v126 // (op 329) Transpose load local data 

s_waitcnt lgkmcnt(15)
s_barrier  // (op 56482) 


v_add_u32 v126, 134144, v14 // (op 333) 
ds_read_u8 v77, v126 // (op 333) Load local data 




v_add_u32 v126, 114704, v15 // (op 3118) 
ds_read_b64_tr_b4 v[136:137], v126 // (op 3118) Transpose load local data 

v_add_u32 v126, 116752, v15 // (op 3118) 
ds_read_b64_tr_b4 v[138:139], v126 // (op 3118) Transpose load local data 



v_add_u32 v126, 134176, v14 // (op 3121) 
ds_read_u8 v34, v126 // (op 3121) Load local data 




v_add_u32 v126, 114720, v15 // (op 5696) 
ds_read_b64_tr_b4 v[140:141], v126 // (op 5696) Transpose load local data 

v_add_u32 v126, 116768, v15 // (op 5696) 
ds_read_b64_tr_b4 v[142:143], v126 // (op 5696) Transpose load local data 



v_add_u32 v126, 134208, v14 // (op 5699) 
ds_read_u8 v41, v126 // (op 5699) Load local data 




v_add_u32 v126, 114736, v15 // (op 8274) 
ds_read_b64_tr_b4 v[144:145], v126 // (op 8274) Transpose load local data 

v_add_u32 v126, 116784, v15 // (op 8274) 
ds_read_b64_tr_b4 v[146:147], v126 // (op 8274) Transpose load local data 



v_add_u32 v126, 134240, v14 // (op 8277) 
ds_read_u8 v24, v126 // (op 8277) Load local data 




v_add_u32 v126, 114752, v15 // (op 10852) 
ds_read_b64_tr_b4 v[148:149], v126 // (op 10852) Transpose load local data 

v_add_u32 v126, 116800, v15 // (op 10852) 
ds_read_b64_tr_b4 v[150:151], v126 // (op 10852) Transpose load local data 



v_add_u32 v126, 134272, v14 // (op 10855) 
ds_read_u8 v25, v126 // (op 10855) Load local data 




v_add_u32 v126, 114768, v15 // (op 13430) 
ds_read_b64_tr_b4 v[152:153], v126 // (op 13430) Transpose load local data 

v_add_u32 v126, 116816, v15 // (op 13430) 
ds_read_b64_tr_b4 v[154:155], v126 // (op 13430) Transpose load local data 



v_add_u32 v126, 134304, v14 // (op 13433) 
ds_read_u8 v66, v126 // (op 13433) Load local data 




v_add_u32 v126, 114784, v15 // (op 16008) 
ds_read_b64_tr_b4 v[156:157], v126 // (op 16008) Transpose load local data 

v_add_u32 v126, 116832, v15 // (op 16008) 
ds_read_b64_tr_b4 v[158:159], v126 // (op 16008) Transpose load local data 



v_add_u32 v126, 134336, v14 // (op 16011) 
ds_read_u8 v67, v126 // (op 16011) Load local data 




v_add_u32 v126, 114800, v15 // (op 18586) 
ds_read_b64_tr_b4 v[160:161], v126 // (op 18586) Transpose load local data 

v_add_u32 v126, 116848, v15 // (op 18586) 
ds_read_b64_tr_b4 v[162:163], v126 // (op 18586) Transpose load local data 



v_add_u32 v126, 134368, v14 // (op 18589) 
ds_read_u8 v116, v126 // (op 18589) Load local data 

s_barrier  // (op 56659) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[82:85], a[0:3], v77, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 336) 
s_barrier  // (op 56479) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[88:91], a[4:7], v77, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 529) 
s_lshl_b64 s[60:61], s[24:25], 5 // (op 54780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[92:95], a[8:11], v77, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 827) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54780) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:81], v[98:101], a[12:15], v77, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1125) 
v_lshrrev_b64 v[164:165], 3, s[62:63] // (op 54780) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[102:105], a[16:19], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1423) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[108:111], a[20:23], v77, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1721) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54780) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[112:115], a[24:27], v77, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2019) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54780) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[132:135], a[28:31], v77, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2317) 
s_mul_i32 s63, 7, s61 // (op 54780) most significant: low of low * high
s_mul_i32 s62, 7, s60 // (op 54780) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:139], v[82:85], a[32:35], v34, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3748) 
s_mul_hi_u32 s39, 7, s60 // (op 54780) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[88:91], a[36:39], v34, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3656) 
s_add_u32 s63, s63, s39 // (op 54780) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[136:139], v[92:95], a[40:43], v34, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3564) 
v_mov_b32 v126, s62 // (op 54780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[136:139], v[98:101], a[44:47], v34, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3472) 
v_mov_b32 v166, s63 // (op 54780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[102:105], a[48:51], v34, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3380) 
v_add_co_u32 v168, s[60:61], v128, v126 // (op 54780) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[108:111], a[52:55], v34, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3288) 
v_addc_co_u32 v169, s[60:61], v129, v166, s[60:61] // (op 54780) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[112:115], a[56:59], v34, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3196) 
s_add_u32 s39, 98304, s19 // (op 54780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[132:135], a[60:63], v34, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3104) 
s_mov_b32 m0, s39 // (op 54780) reg expression

buffer_load_dwordx4 v168, s[48:51], 0 offen offset: 0 lds // (op 54780) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:143], v[82:85], a[64:67], v41, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6326) 
v_add_co_u32 v168, s[60:61], v168, v164 // (op 54780) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:143], v[88:91], a[68:71], v41, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6234) 
v_addc_co_u32 v169, s[60:61], 0, v169, s[60:61] // (op 54780) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[92:95], a[72:75], v41, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6142) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[98:101], a[76:79], v41, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6050) 
s_add_u32 m0, m0, 4096 // (op 54780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:143], v[102:105], a[80:83], v41, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5958) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[108:111], a[84:87], v41, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5866) 
buffer_load_dwordx4 v168, s[48:51], 0 offen offset: 0 lds // (op 54780) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[112:115], a[88:91], v41, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5774) 
v_add_co_u32 v168, s[60:61], v168, v164 // (op 54780) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[132:135], a[92:95], v41, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5682) 
v_addc_co_u32 v169, s[60:61], 0, v169, s[60:61] // (op 54780) most significant half


s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:147], v[82:85], a[96:99], v24, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8904) 
s_add_u32 m0, m0, 4096 // (op 54780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[88:91], a[100:103], v24, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8812) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[92:95], a[104:107], v24, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8720) 
buffer_load_dwordx4 v168, s[48:51], 0 offen offset: 0 lds // (op 54780) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[144:147], v[98:101], a[108:111], v24, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8628) 
v_add_co_u32 v168, s[60:61], v168, v164 // (op 54780) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:147], v[102:105], a[112:115], v24, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8536) 
v_addc_co_u32 v169, s[60:61], 0, v169, s[60:61] // (op 54780) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[108:111], a[116:119], v24, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8444) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:147], v[112:115], a[120:123], v24, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8352) 
s_add_u32 m0, m0, 4096 // (op 54780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:147], v[132:135], a[124:127], v24, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8260) 

buffer_load_dwordx4 v168, s[48:51], 0 offen offset: 0 lds // (op 54780) Load value direct to lds




s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:151], v[82:85], a[128:131], v25, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11482) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[148:151], v[88:91], a[132:135], v25, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11390) 
s_lshl_b32 s39, s34, 2 // (op 342) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:151], v[92:95], a[136:139], v25, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11298) 
s_mul_i32 s60, 7, s39 // (op 342) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:151], v[98:101], a[140:143], v25, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11206) 
v_add_u32 v126, v22, s60 // (op 342) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[148:151], v[102:105], a[144:147], v25, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11114) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54982) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:151], v[108:111], a[148:151], v25, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11022) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54982) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[112:115], a[152:155], v25, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10930) 
buffer_load_dword v28, v126, s[56:59], 0 offen // (op 342) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:151], v[132:135], a[156:159], v25, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10838) 

s_lshl_b32 s39, s34, 4 // (op 55110) 
v_add_u32 v35, v35, s39 // (op 55110) 

s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[152:155], v[82:85], a[160:163], v66, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14060) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54982) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[152:155], v[88:91], a[164:167], v66, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13968) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54982) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[152:155], v[92:95], a[168:171], v66, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13876) 
v_mov_b32 v22, s62 // (op 54982) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[152:155], v[98:101], a[172:175], v66, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13784) 
v_mov_b32 v126, s63 // (op 54982) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:155], v[102:105], a[176:179], v66, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13692) 
v_add_co_u32 v30, s[60:61], v30, v22 // (op 54982) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[152:155], v[108:111], a[180:183], v66, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13600) 
v_addc_co_u32 v31, s[60:61], v31, v126, s[60:61] // (op 54982) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[152:155], v[112:115], a[184:187], v66, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13508) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 56472) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:155], v[132:135], a[188:191], v66, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13416) 
s_lshl_b64 s[60:61], s[12:13], 5 // (op 54775) 

s_lshl_b64 s[62:63], s[60:61], 2 // (op 54775) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[156:159], v[82:85], a[192:195], v67, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16638) 
v_lshrrev_b64 v[128:129], 3, s[62:63] // (op 54775) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[156:159], v[88:91], a[196:199], v67, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16546) 
s_lshl_b64 s[60:61], s[12:13], 7 // (op 54775) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[156:159], v[92:95], a[200:203], v67, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16454) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54775) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[156:159], v[98:101], a[204:207], v67, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16362) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54775) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[156:159], v[102:105], a[208:211], v67, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16270) 
s_mul_i32 s63, 7, s61 // (op 54775) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[156:159], v[108:111], a[212:215], v67, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16178) 
s_mul_i32 s62, 7, s60 // (op 54775) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[156:159], v[112:115], a[216:219], v67, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16086) 
s_mul_hi_u32 s39, 7, s60 // (op 54775) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[156:159], v[132:135], a[220:223], v67, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15994) 
s_add_u32 s63, s63, s39 // (op 54775) most significant: sum



v_mov_b32 v22, s62 // (op 54775) 
v_mov_b32 v126, s63 // (op 54775) 
v_add_co_u32 v136, s[60:61], v130, v22 // (op 54775) least significant half
v_addc_co_u32 v137, s[60:61], v131, v126, s[60:61] // (op 54775) most significant half




s_add_u32 s39, 114688, s18 // (op 54775) 
s_mov_b32 m0, s39 // (op 54775) reg expression
buffer_load_dwordx4 v136, s[8:11], 0 offen offset: 0 lds // (op 54775) Load value direct to lds
v_add_co_u32 v136, s[60:61], v136, v128 // (op 54775) least significant half
v_addc_co_u32 v137, s[60:61], 0, v137, s[60:61] // (op 54775) most significant half

s_add_u32 m0, m0, 4096 // (op 54775) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[160:163], v[82:85], a[224:227], v116, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19216) 
buffer_load_dwordx4 v136, s[8:11], 0 offen offset: 0 lds // (op 54775) Load value direct to lds
v_add_co_u32 v136, s[60:61], v136, v128 // (op 54775) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[160:163], v[88:91], a[228:231], v116, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19124) 
v_addc_co_u32 v137, s[60:61], 0, v137, s[60:61] // (op 54775) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[160:163], v[92:95], a[232:235], v116, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19032) 
s_add_u32 m0, m0, 4096 // (op 54775) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[160:163], v[98:101], a[236:239], v116, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18940) 
buffer_load_dwordx4 v136, s[8:11], 0 offen offset: 0 lds // (op 54775) Load value direct to lds
v_add_co_u32 v136, s[60:61], v136, v128 // (op 54775) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:163], v[102:105], a[240:243], v116, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18848) 
v_addc_co_u32 v137, s[60:61], 0, v137, s[60:61] // (op 54775) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[160:163], v[108:111], a[244:247], v116, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18756) 
s_add_u32 m0, m0, 4096 // (op 54775) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[160:163], v[112:115], a[248:251], v116, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18664) 
buffer_load_dwordx4 v136, s[8:11], 0 offen offset: 0 lds // (op 54775) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[160:163], v[132:135], a[252:255], v116, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18572) 
s_lshl_b32 s39, s28, 2 // (op 341) 



s_mul_i32 s60, 7, s39 // (op 341) 

v_add_u32 v22, v117, s60 // (op 341) 


s_lshl_b64 s[60:61], s[12:13], 7 // (op 54918) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54918) 

s_lshr_b64 s[60:61], s[62:63], 3 // (op 54918) 

s_lshl_b64 s[62:63], s[60:61], 2 // (op 54918) 

v_mov_b32 v126, s62 // (op 54918) 
v_mov_b32 v127, s63 // (op 54918) 
v_add_co_u32 v26, s[60:61], v26, v126 // (op 54918) least significant half
v_addc_co_u32 v27, s[60:61], v27, v127, s[60:61] // (op 54918) most significant half




buffer_load_dword v32, v22, s[52:55], 0 offen // (op 341) Load value


s_barrier  // (op 56403) 
s_lshl_b32 s39, s28, 4 // (op 55046) 



v_add_u32 v23, v23, s39 // (op 55046) 

v_add_u32 v22, 131072, v119 // (op 30) 
s_waitcnt vmcnt(35)
ds_write_b32 v22, v33 // (op 30) Store local data 

s_barrier  // (op 56385) 



v_add_u32 v22, 132096, v118 // (op 21) 
s_waitcnt vmcnt(30)
ds_write_b32 v22, v36 // (op 21) Store local data 

s_barrier  // (op 56374) 



ds_read_b64_tr_b4 v[42:43], v4 // (op 77) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v4 offset:2048 // (op 77) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 56410) 


v_add_u32 v22, 131072, v5 // (op 87) 
ds_read_u8 v29, v22 // (op 87) Load local data 




ds_read_b64_tr_b4 v[46:47], v4 offset:16 // (op 521) Transpose load local data 
ds_read_b64_tr_b4 v[48:49], v4 offset:2064 // (op 521) Transpose load local data 


v_add_u32 v22, 131104, v5 // (op 525) 
ds_read_u8 v34, v22 // (op 525) Load local data 




ds_read_b64_tr_b4 v[50:51], v4 offset:32 // (op 819) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:2080 // (op 819) Transpose load local data 


v_add_u32 v22, 131136, v5 // (op 823) 
ds_read_u8 v41, v22 // (op 823) Load local data 




ds_read_b64_tr_b4 v[54:55], v4 offset:48 // (op 1117) Transpose load local data 
ds_read_b64_tr_b4 v[56:57], v4 offset:2096 // (op 1117) Transpose load local data 


v_add_u32 v22, 131168, v5 // (op 1121) 
ds_read_u8 v24, v22 // (op 1121) Load local data 




ds_read_b64_tr_b4 v[58:59], v4 offset:64 // (op 1415) Transpose load local data 
ds_read_b64_tr_b4 v[60:61], v4 offset:2112 // (op 1415) Transpose load local data 


v_add_u32 v22, 131200, v5 // (op 1419) 
ds_read_u8 v25, v22 // (op 1419) Load local data 




ds_read_b64_tr_b4 v[62:63], v4 offset:80 // (op 1713) Transpose load local data 
ds_read_b64_tr_b4 v[64:65], v4 offset:2128 // (op 1713) Transpose load local data 


v_add_u32 v22, 131232, v5 // (op 1717) 
ds_read_u8 v66, v22 // (op 1717) Load local data 




ds_read_b64_tr_b4 v[68:69], v4 offset:96 // (op 2011) Transpose load local data 
ds_read_b64_tr_b4 v[70:71], v4 offset:2144 // (op 2011) Transpose load local data 


v_add_u32 v22, 131264, v5 // (op 2015) 
ds_read_u8 v67, v22 // (op 2015) Load local data 




ds_read_b64_tr_b4 v[72:73], v4 offset:112 // (op 2309) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v4 offset:2160 // (op 2309) Transpose load local data 


v_add_u32 v22, 131296, v5 // (op 2313) 
ds_read_u8 v76, v22 // (op 2313) Load local data 

s_barrier  // (op 56358) 



ds_read_b64_tr_b4 v[78:79], v3 offset:16384 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v3 offset:18432 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 56392) 


v_add_u32 v22, 132096, v0 // (op 82) 
ds_read_u8 v77, v22 // (op 82) Load local data 




ds_read_b64_tr_b4 v[82:83], v3 offset:16400 // (op 3096) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v3 offset:18448 // (op 3096) Transpose load local data 


v_add_u32 v22, 132128, v0 // (op 3099) 
ds_read_u8 v86, v22 // (op 3099) Load local data 




ds_read_b64_tr_b4 v[88:89], v3 offset:16416 // (op 5674) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v3 offset:18464 // (op 5674) Transpose load local data 


v_add_u32 v22, 132160, v0 // (op 5677) 
ds_read_u8 v87, v22 // (op 5677) Load local data 




ds_read_b64_tr_b4 v[92:93], v3 offset:16432 // (op 8252) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v3 offset:18480 // (op 8252) Transpose load local data 


v_add_u32 v22, 132192, v0 // (op 8255) 
ds_read_u8 v96, v22 // (op 8255) Load local data 




ds_read_b64_tr_b4 v[98:99], v3 offset:16448 // (op 10830) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v3 offset:18496 // (op 10830) Transpose load local data 


v_add_u32 v22, 132224, v0 // (op 10833) 
ds_read_u8 v97, v22 // (op 10833) Load local data 




ds_read_b64_tr_b4 v[102:103], v3 offset:16464 // (op 13408) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v3 offset:18512 // (op 13408) Transpose load local data 


v_add_u32 v22, 132256, v0 // (op 13411) 
ds_read_u8 v106, v22 // (op 13411) Load local data 




ds_read_b64_tr_b4 v[108:109], v3 offset:16480 // (op 15986) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v3 offset:18528 // (op 15986) Transpose load local data 


v_add_u32 v22, 132288, v0 // (op 15989) 
ds_read_u8 v107, v22 // (op 15989) Load local data 




ds_read_b64_tr_b4 v[112:113], v3 offset:16496 // (op 18564) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v3 offset:18544 // (op 18564) Transpose load local data 


v_add_u32 v22, 132320, v0 // (op 18567) 
ds_read_u8 v116, v22 // (op 18567) Load local data 

s_ashr_i64 s[60:61], s[22:23], 7 // (op 18530) 
s_mov_b32 s39, s60 // (op 18530) convert

s_and_b32 s60, s39, 4294967292 // (op 18530) 

s_cmp_lt_u32 s38, s60 // (op 18530) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoop_18530 // (op 18530) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoop_18530 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoop_18530:













































s_barrier  // (op 56627) 
v_and_b32 v22, -8, v1 // (op 54818) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 54818) 
v_sub_u32 v23, v1, v22 // (op 54818) 

s_lshr_b32 s6, s0, 2 // (op 54818) 

v_lshlrev_b32 v22, 5, v23 // (op 54818) 

v_lshl_add_u32 v23, s16, 8, v22 // (op 54818) 

v_mov_b32 v31, 0 // (op 54818) convert
v_mov_b32 v30, v23 // (op 54818) convert

v_lshrrev_b32 v29, 3, v1 // (op 54818) 
v_lshl_add_u32 v22, s6, 9, v29 // (op 54818) 


v_mul_lo_u32 v27, v22, s13 // (op 54818) most significant: low of low * high
v_mul_hi_u32 v29, v22, s12 // (op 54818) most significant: high of low * low
v_mul_lo_u32 v26, v22, s12 // (op 54818) least significant: low of low * low
v_add_u32 v27, v27, v29 // (op 54818) most significant: sum


v_add_co_u32 v22, s[0:1], v30, v26 // (op 54818) least significant half
v_addc_co_u32 v23, s[0:1], v31, v27, s[0:1] // (op 54818) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 54818) 


v_lshrrev_b64 v[30:31], 3, v[22:23] // (op 54818) 

s_mul_i32 s1, s12, s23 // (op 54820) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 54820) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 54820) least significant
s_mul_hi_u32 s7, s12, s22 // (op 54820) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54820) most significant: sum
s_add_u32 s1, s1, s7 // (op 54820) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54820) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54820) 

s_mov_b32 s8, s20 // (op 54820) reg expression
s_mov_b32 s9, s21 // (op 54820) reg expression
s_mov_b32 s10, s0 // (op 54820) reg expression
s_mov_b32 s11, 131072 // (op 54820) call()

v_lshrrev_b32 v29, 3, v1 // (op 54948) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 54948) 
v_and_b32 v22, -8, v1 // (op 54948) 
s_lshr_b32 s6, s0, 2 // (op 54948) 

v_sub_u32 v23, v1, v22 // (op 54948) 

v_lshl_add_u32 v22, s6, 9, v29 // (op 54948) 


v_lshlrev_b32 v29, 5, v23 // (op 54948) 

v_lshl_add_u32 v23, s17, 8, v29 // (op 54948) 

v_mov_b32 v25, 0 // (op 54948) convert
v_mov_b32 v24, v23 // (op 54948) convert

v_mul_lo_u32 v27, v22, s25 // (op 54948) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 54948) most significant: high of low * low
v_mul_lo_u32 v26, v22, s24 // (op 54948) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 54948) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v24 // (op 54948) least significant half
v_addc_co_u32 v27, s[0:1], v27, v25, s[0:1] // (op 54948) most significant half



v_lshlrev_b64 v[22:23], 2, v[26:27] // (op 54948) 

v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 54948) 

s_mul_i32 s1, s24, s23 // (op 54950) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 54950) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 54950) least significant
s_mul_hi_u32 s7, s24, s22 // (op 54950) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54950) most significant: sum
s_add_u32 s1, s1, s7 // (op 54950) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54950) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54950) 

s_mov_b32 s48, s26 // (op 54950) reg expression
s_mov_b32 s49, s27 // (op 54950) reg expression
s_mov_b32 s50, s0 // (op 54950) reg expression
s_mov_b32 s51, 131072 // (op 54950) call()

v_and_b32 v29, -64, v1 // (op 55012) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 55012) 
v_sub_u32 v22, v1, v29 // (op 55012) 

s_lshr_b32 s6, s0, 2 // (op 55012) 

v_lshlrev_b32 v23, 2, v22 // (op 55012) 

v_lshl_add_u32 v22, s16, 8, v23 // (op 55012) 

v_lshrrev_b32 v23, 6, v1 // (op 55012) 
v_lshl_add_u32 v29, s6, 4, v23 // (op 55012) 


v_mul_lo_u32 v23, v29, s28 // (op 55012) 

v_add_u32 v29, v22, v23 // (op 55012) 


s_mul_i32 s1, s28, s33 // (op 55014) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 55014) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 55014) least significant
s_mul_hi_u32 s7, s28, s32 // (op 55014) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 55014) most significant: sum
s_add_u32 s1, s1, s7 // (op 55014) most significant: sum


s_mov_b32 s52, s30 // (op 55014) reg expression
s_mov_b32 s53, s31 // (op 55014) reg expression
s_mov_b32 s54, s0 // (op 55014) reg expression
s_mov_b32 s55, 131072 // (op 55014) call()

v_lshrrev_b32 v22, 6, v1 // (op 55076) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 55076) 
v_and_b32 v23, -64, v1 // (op 55076) 
s_lshr_b32 s6, s0, 2 // (op 55076) 

v_sub_u32 v26, v1, v23 // (op 55076) 

v_lshl_add_u32 v23, s6, 4, v22 // (op 55076) 


v_lshlrev_b32 v22, 2, v26 // (op 55076) 

v_lshl_add_u32 v26, s17, 8, v22 // (op 55076) 

v_mul_lo_u32 v22, v23, s34 // (op 55076) 
v_add_u32 v22, v22, v26 // (op 55076) 


s_mul_i32 s1, s34, s33 // (op 55078) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 55078) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 55078) least significant
s_mul_hi_u32 s7, s34, s32 // (op 55078) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 55078) most significant: sum
s_add_u32 s1, s1, s7 // (op 55078) most significant: sum


s_mov_b32 s56, s36 // (op 55078) reg expression
s_mov_b32 s57, s37 // (op 55078) reg expression
s_mov_b32 s58, s0 // (op 55078) reg expression
s_mov_b32 s59, 131072 // (op 55078) call()

v_and_b32 v23, -8, v1 // (op 55174) 
v_sub_u32 v26, v1, v23 // (op 55174) 

v_lshlrev_b32 v23, 5, v26 // (op 55174) 

v_lshrrev_b32 v26, 3, v1 // (op 55174) 
v_lshl_add_u32 v27, v26, 8, v23 // (op 55174) 


v_lshlrev_b32 v23, 2, v27 // (op 55174) 

v_lshrrev_b32 v26, 3, v23 // (op 55174) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 55174) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -8, v1 // (op 55231) 
v_sub_u32 v26, v1, v23 // (op 55231) 

v_lshlrev_b32 v23, 5, v26 // (op 55231) 

v_lshrrev_b32 v26, 3, v1 // (op 55231) 
v_lshl_add_u32 v27, v26, 8, v23 // (op 55231) 


v_lshlrev_b32 v23, 2, v27 // (op 55231) 

v_lshrrev_b32 v26, 3, v23 // (op 55231) 

s_nop 0
v_readfirstlane_b32 s1, v26 // (op 55231) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -64, v1 // (op 55288) 
v_sub_u32 v26, v1, v23 // (op 55288) 

v_lshlrev_b32 v23, 2, v26 // (op 55288) 

v_lshrrev_b32 v26, 6, v1 // (op 55288) 
v_lshl_add_u32 v27, v26, 8, v23 // (op 55288) 


v_and_b32 v23, -64, v1 // (op 55346) 
v_sub_u32 v26, v1, v23 // (op 55346) 

v_lshlrev_b32 v23, 2, v26 // (op 55346) 

v_lshrrev_b32 v26, 6, v1 // (op 55346) 
v_lshl_add_u32 v33, v26, 8, v23 // (op 55346) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 18536) 
s_and_b32 s14, s6, 4294967292 // (op 18536) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 18535) 
s_mov_b32 s15, s6 // (op 18535) convert

s_cmp_lt_u32 s14, s15 // (op 18535) 

















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoopTail_18535 // (op 18535) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoopTail_18535 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoopTail_18535:
s_barrier  // (op 56355) 
s_add_u32 s14, s14, 1 // (op 18537) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 54742) 
s_lshl_b64 s[18:19], s[6:7], 2 // (op 54742) 

v_lshrrev_b64 v[34:35], 3, s[18:19] // (op 54742) 


v_mov_b64 v[36:37], v[30:31] // (op 54742) 
v_mov_b64 v[38:39], v[36:37] // (op 54742) reg expression
s_add_u32 s15, 16384, s0 // (op 54742) 
s_mov_b32 m0, s15 // (op 54742) reg expression
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54742) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54742) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54742) most significant half

s_add_u32 m0, m0, 4096 // (op 54742) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54742) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54742) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54742) most significant half

s_add_u32 m0, m0, 4096 // (op 54742) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54742) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54742) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54742) most significant half

s_add_u32 m0, m0, 4096 // (op 54742) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54742) Load value direct to lds




s_lshl_b64 s[6:7], s[12:13], 7 // (op 54808) 
s_lshl_b64 s[18:19], s[6:7], 2 // (op 54808) 

s_lshr_b64 s[6:7], s[18:19], 3 // (op 54808) 

v_mov_b32 v23, s6 // (op 54808) 
v_mov_b32 v26, s7 // (op 54808) 
v_add_co_u32 v30, s[18:19], v30, v23 // (op 54808) least significant half
v_addc_co_u32 v31, s[18:19], v31, v26, s[18:19] // (op 54808) most significant half




s_barrier  // (op 56371) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 54748) 
s_lshl_b64 s[18:19], s[6:7], 2 // (op 54748) 

v_lshrrev_b64 v[34:35], 3, s[18:19] // (op 54748) 


v_mov_b64 v[36:37], v[24:25] // (op 54748) 
v_mov_b64 v[38:39], v[36:37] // (op 54748) reg expression
s_mov_b32 m0, s1 // (op 54748) reg expression
buffer_load_dwordx4 v38, s[48:51], 0 offen offset: 0 lds // (op 54748) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54748) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54748) most significant half

s_add_u32 m0, m0, 4096 // (op 54748) 
buffer_load_dwordx4 v38, s[48:51], 0 offen offset: 0 lds // (op 54748) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54748) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54748) most significant half

s_add_u32 m0, m0, 4096 // (op 54748) 
buffer_load_dwordx4 v38, s[48:51], 0 offen offset: 0 lds // (op 54748) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54748) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54748) most significant half

s_add_u32 m0, m0, 4096 // (op 54748) 
buffer_load_dwordx4 v38, s[48:51], 0 offen offset: 0 lds // (op 54748) Load value direct to lds



s_lshl_b64 s[6:7], s[24:25], 7 // (op 54940) 
s_lshl_b64 s[18:19], s[6:7], 2 // (op 54940) 

s_lshr_b64 s[6:7], s[18:19], 3 // (op 54940) 

v_mov_b32 v23, s6 // (op 54940) 
v_mov_b32 v26, s7 // (op 54940) 
v_add_co_u32 v24, s[18:19], v24, v23 // (op 54940) least significant half
v_addc_co_u32 v25, s[18:19], v25, v26, s[18:19] // (op 54940) most significant half






v_mov_b32 v23, v29 // (op 195) 
v_mov_b32 v26, v23 // (op 195) reg expression

buffer_load_dword v28, v26, s[52:55], 0 offen // (op 195) Load value


s_lshl_b32 s15, s28, 2 // (op 55004) 
v_add_u32 v29, v29, s15 // (op 55004) 



s_barrier  // (op 56388) 
v_mov_b32 v23, v22 // (op 196) 



v_mov_b32 v26, v23 // (op 196) reg expression

v_add_u32 v32, 132096, v27 // (op 192) 
s_waitcnt vmcnt(0)
ds_write_b32 v32, v28 // (op 192) Store local data 

buffer_load_dword v32, v26, s[56:59], 0 offen // (op 196) Load value



s_lshl_b32 s15, s34, 2 // (op 55068) 
v_add_u32 v22, v22, s15 // (op 55068) 

s_barrier  // (op 56361) 
s_barrier  // (op 56406) 






v_add_u32 v23, 131072, v33 // (op 193) 
ds_read_b64_tr_b4 v[34:35], v3 offset:16384 // (op 183) Transpose load local data 
ds_read_b64_tr_b4 v[36:37], v3 offset:18432 // (op 183) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v23, v32 // (op 193) Store local data 


s_barrier  // (op 56378) 



ds_read_b64_tr_b4 v[38:39], v4 // (op 185) Transpose load local data 
ds_read_b64_tr_b4 v[40:41], v4 offset:2048 // (op 185) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 56396) 


v_add_u32 v23, 132096, v0 // (op 187) 
ds_read_u8 v26, v23 // (op 187) Load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 56413) 


v_add_u32 v23, 131072, v5 // (op 189) 
ds_read_u8 v28, v23 // (op 189) Load local data 




ds_read_b64_tr_b4 v[42:43], v4 offset:16 // (op 500) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:41], a[0:3], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 190) 
ds_read_b64_tr_b4 v[44:45], v4 offset:2064 // (op 500) Transpose load local data 


v_add_u32 v23, 131104, v5 // (op 504) 
ds_read_u8 v32, v23 // (op 504) Load local data 




ds_read_b64_tr_b4 v[46:47], v4 offset:32 // (op 798) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[42:45], a[4:7], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 505) 
ds_read_b64_tr_b4 v[48:49], v4 offset:2080 // (op 798) Transpose load local data 


v_add_u32 v23, 131136, v5 // (op 802) 
ds_read_u8 v50, v23 // (op 802) Load local data 




ds_read_b64_tr_b4 v[52:53], v4 offset:48 // (op 1096) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[46:49], a[8:11], v26, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 803) 
ds_read_b64_tr_b4 v[54:55], v4 offset:2096 // (op 1096) Transpose load local data 


v_add_u32 v23, 131168, v5 // (op 1100) 
ds_read_u8 v51, v23 // (op 1100) Load local data 




ds_read_b64_tr_b4 v[56:57], v4 offset:64 // (op 1394) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[52:55], a[12:15], v26, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1101) 
ds_read_b64_tr_b4 v[58:59], v4 offset:2112 // (op 1394) Transpose load local data 


v_add_u32 v23, 131200, v5 // (op 1398) 
ds_read_u8 v60, v23 // (op 1398) Load local data 




ds_read_b64_tr_b4 v[62:63], v4 offset:80 // (op 1692) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[34:37], v[56:59], a[16:19], v26, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1399) 
ds_read_b64_tr_b4 v[64:65], v4 offset:2128 // (op 1692) Transpose load local data 


v_add_u32 v23, 131232, v5 // (op 1696) 
ds_read_u8 v61, v23 // (op 1696) Load local data 




ds_read_b64_tr_b4 v[66:67], v4 offset:96 // (op 1990) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:37], v[62:65], a[20:23], v26, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1697) 
ds_read_b64_tr_b4 v[68:69], v4 offset:2144 // (op 1990) Transpose load local data 


v_add_u32 v23, 131264, v5 // (op 1994) 
ds_read_u8 v70, v23 // (op 1994) Load local data 




ds_read_b64_tr_b4 v[72:73], v4 offset:112 // (op 2288) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[34:37], v[66:69], a[24:27], v26, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1995) 
ds_read_b64_tr_b4 v[74:75], v4 offset:2160 // (op 2288) Transpose load local data 


v_add_u32 v23, 131296, v5 // (op 2292) 
ds_read_u8 v71, v23 // (op 2292) Load local data 




ds_read_b64_tr_b4 v[76:77], v3 offset:16400 // (op 2856) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[34:37], v[72:75], a[28:31], v26, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2293) 
ds_read_b64_tr_b4 v[78:79], v3 offset:18448 // (op 2856) Transpose load local data 




v_add_u32 v23, 132128, v0 // (op 2859) 
ds_read_u8 v26, v23 // (op 2859) Load local data 




ds_read_b64_tr_b4 v[34:35], v3 offset:16416 // (op 5434) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[38:41], a[32:35], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2861) 
ds_read_b64_tr_b4 v[36:37], v3 offset:18464 // (op 5434) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[42:45], a[36:39], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2892) 
v_add_u32 v23, 132160, v0 // (op 5437) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[46:49], a[40:43], v26, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2923) 
ds_read_u8 v80, v23 // (op 5437) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[52:55], a[44:47], v26, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2954) 
ds_read_b64_tr_b4 v[82:83], v3 offset:16432 // (op 8012) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[56:59], a[48:51], v26, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2985) 
ds_read_b64_tr_b4 v[84:85], v3 offset:18480 // (op 8012) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[62:65], a[52:55], v26, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3016) 
v_add_u32 v23, 132192, v0 // (op 8015) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[66:69], a[56:59], v26, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3047) 
ds_read_u8 v81, v23 // (op 8015) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[72:75], a[60:63], v26, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3078) 
ds_read_b64_tr_b4 v[86:87], v3 offset:16448 // (op 10590) Transpose load local data 


ds_read_b64_tr_b4 v[88:89], v3 offset:18496 // (op 10590) Transpose load local data 


s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:37], v[38:41], a[64:67], v80, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5439) 
v_add_u32 v23, 132224, v0 // (op 10593) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[42:45], a[68:71], v80, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5470) 
ds_read_u8 v26, v23 // (op 10593) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[34:37], v[46:49], a[72:75], v80, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5501) 
ds_read_b64_tr_b4 v[76:77], v3 offset:16464 // (op 13168) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:37], v[52:55], a[76:79], v80, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5532) 
ds_read_b64_tr_b4 v[78:79], v3 offset:18512 // (op 13168) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[56:59], a[80:83], v80, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5563) 
v_add_u32 v23, 132256, v0 // (op 13171) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:37], v[62:65], a[84:87], v80, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5594) 
ds_read_u8 v90, v23 // (op 13171) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[34:37], v[66:69], a[88:91], v80, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5625) 
ds_read_b64_tr_b4 v[92:93], v3 offset:16480 // (op 15746) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[72:75], a[92:95], v80, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5656) 
ds_read_b64_tr_b4 v[94:95], v3 offset:18528 // (op 15746) Transpose load local data 




s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[38:41], a[96:99], v81, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8017) 
v_add_u32 v23, 132288, v0 // (op 15749) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[42:45], a[100:103], v81, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8048) 
ds_read_u8 v80, v23 // (op 15749) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[46:49], a[104:107], v81, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8079) 
ds_read_b64_tr_b4 v[34:35], v3 offset:16496 // (op 18324) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[52:55], a[108:111], v81, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8110) 
ds_read_b64_tr_b4 v[36:37], v3 offset:18544 // (op 18324) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:85], v[56:59], a[112:115], v81, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8141) 
v_add_u32 v23, 132320, v0 // (op 18327) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:85], v[62:65], a[116:119], v81, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8172) 
ds_read_u8 v91, v23 // (op 18327) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:85], v[66:69], a[120:123], v81, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8203) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:85], v[72:75], a[124:127], v81, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8234) 


s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:89], v[38:41], a[128:131], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10595) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:89], v[42:45], a[132:135], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:89], v[46:49], a[136:139], v26, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10657) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:89], v[52:55], a[140:143], v26, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10688) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:89], v[56:59], a[144:147], v26, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10719) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:89], v[62:65], a[148:151], v26, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10750) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:89], v[66:69], a[152:155], v26, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10781) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:89], v[72:75], a[156:159], v26, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10812) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[76:79], v[38:41], a[160:163], v90, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13173) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[76:79], v[42:45], a[164:167], v90, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[76:79], v[46:49], a[168:171], v90, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13235) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[76:79], v[52:55], a[172:175], v90, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13266) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[76:79], v[56:59], a[176:179], v90, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13297) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[76:79], v[62:65], a[180:183], v90, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13328) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[76:79], v[66:69], a[184:187], v90, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13359) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[76:79], v[72:75], a[188:191], v90, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13390) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[92:95], v[38:41], a[192:195], v80, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15751) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[92:95], v[42:45], a[196:199], v80, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15782) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[92:95], v[46:49], a[200:203], v80, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15813) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[92:95], v[52:55], a[204:207], v80, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15844) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:95], v[56:59], a[208:211], v80, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15875) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[92:95], v[62:65], a[212:215], v80, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15906) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[92:95], v[66:69], a[216:219], v80, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15937) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[92:95], v[72:75], a[220:223], v80, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15968) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[34:37], v[38:41], a[224:227], v91, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18329) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[34:37], v[42:45], a[228:231], v91, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18360) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[34:37], v[46:49], a[232:235], v91, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18391) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[34:37], v[52:55], a[236:239], v91, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18422) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[34:37], v[56:59], a[240:243], v91, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18453) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[34:37], v[62:65], a[244:247], v91, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18484) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[34:37], v[66:69], a[248:251], v91, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18515) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[34:37], v[72:75], a[252:255], v91, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18546) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 18535) 
s_mov_b32 s15, s6 // (op 18535) convert

s_cmp_lt_u32 s14, s15 // (op 18535) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoopTail_18535 // (op 18535) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoopTail_18535 if true)
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoopTail_18535:























s_barrier  // (op 56631) 
















v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalFalse_0_beta_135
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_mul_f32 v10, v19, v14 // (op 50) 
v_mul_f32 v11, v19, v15 // (op 50) 
v_mul_f32 v12, v19, v16 // (op 50) 
v_mul_f32 v13, v19, v17 // (op 50) 

v_lshrrev_b32 v14, 16, v10 // (op 56) 
v_lshrrev_b32 v15, 16, v11 // (op 56) 
v_lshrrev_b32 v16, 16, v12 // (op 56) 
v_lshrrev_b32 v17, 16, v13 // (op 56) 


v_pack_b32_f16 v12, v14, v15 // (op 58) 
v_pack_b32_f16 v13, v16, v17 // (op 58) 

v_mov_b32 v0, v21 // (op 58) reg expression

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v14, a4 // (op 22837) 
v_accvgpr_read v15, a5 // (op 22837) 
v_accvgpr_read v16, a6 // (op 22837) 
v_accvgpr_read v17, a7 // (op 22837) 
v_mul_f32 v10, v19, v14 // (op 22837) 
v_mul_f32 v11, v19, v15 // (op 22837) 
v_mul_f32 v12, v19, v16 // (op 22837) 
v_mul_f32 v13, v19, v17 // (op 22837) 

v_lshrrev_b32 v14, 16, v10 // (op 22839) 
v_lshrrev_b32 v15, 16, v11 // (op 22839) 
v_lshrrev_b32 v16, 16, v12 // (op 22839) 
v_lshrrev_b32 v17, 16, v13 // (op 22839) 


v_pack_b32_f16 v12, v14, v15 // (op 22840) 
v_pack_b32_f16 v13, v16, v17 // (op 22840) 

s_lshl_b32 s0, s4, 6 // (op 22840) 
v_add_u32 v0, v21, s0 // (op 22840) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22840) Store value



v_accvgpr_read v14, a8 // (op 22851) 
v_accvgpr_read v15, a9 // (op 22851) 
v_accvgpr_read v16, a10 // (op 22851) 
v_accvgpr_read v17, a11 // (op 22851) 
v_mul_f32 v10, v19, v14 // (op 22851) 
v_mul_f32 v11, v19, v15 // (op 22851) 
v_mul_f32 v12, v19, v16 // (op 22851) 
v_mul_f32 v13, v19, v17 // (op 22851) 

v_lshrrev_b32 v14, 16, v10 // (op 22853) 
v_lshrrev_b32 v15, 16, v11 // (op 22853) 
v_lshrrev_b32 v16, 16, v12 // (op 22853) 
v_lshrrev_b32 v17, 16, v13 // (op 22853) 


v_pack_b32_f16 v12, v14, v15 // (op 22854) 
v_pack_b32_f16 v13, v16, v17 // (op 22854) 

s_lshl_b32 s0, s4, 7 // (op 22854) 
v_add_u32 v0, v21, s0 // (op 22854) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22854) Store value



v_accvgpr_read v14, a12 // (op 22865) 
v_accvgpr_read v15, a13 // (op 22865) 
v_accvgpr_read v16, a14 // (op 22865) 
v_accvgpr_read v17, a15 // (op 22865) 
v_mul_f32 v10, v19, v14 // (op 22865) 
v_mul_f32 v11, v19, v15 // (op 22865) 
v_mul_f32 v12, v19, v16 // (op 22865) 
v_mul_f32 v13, v19, v17 // (op 22865) 

v_lshrrev_b32 v14, 16, v10 // (op 22867) 
v_lshrrev_b32 v15, 16, v11 // (op 22867) 
v_lshrrev_b32 v16, 16, v12 // (op 22867) 
v_lshrrev_b32 v17, 16, v13 // (op 22867) 


v_pack_b32_f16 v12, v14, v15 // (op 22868) 
v_pack_b32_f16 v13, v16, v17 // (op 22868) 

s_lshl_b32 s0, s4, 6 // (op 22868) 
s_mul_i32 s1, 3, s0 // (op 22868) 

v_add_u32 v0, v21, s1 // (op 22868) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22868) Store value



v_accvgpr_read v14, a16 // (op 22879) 
v_accvgpr_read v15, a17 // (op 22879) 
v_accvgpr_read v16, a18 // (op 22879) 
v_accvgpr_read v17, a19 // (op 22879) 
v_mul_f32 v10, v19, v14 // (op 22879) 
v_mul_f32 v11, v19, v15 // (op 22879) 
v_mul_f32 v12, v19, v16 // (op 22879) 
v_mul_f32 v13, v19, v17 // (op 22879) 

v_lshrrev_b32 v14, 16, v10 // (op 22881) 
v_lshrrev_b32 v15, 16, v11 // (op 22881) 
v_lshrrev_b32 v16, 16, v12 // (op 22881) 
v_lshrrev_b32 v17, 16, v13 // (op 22881) 


v_pack_b32_f16 v12, v14, v15 // (op 22882) 
v_pack_b32_f16 v13, v16, v17 // (op 22882) 

s_lshl_b32 s0, s4, 8 // (op 22882) 
v_add_u32 v0, v21, s0 // (op 22882) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22882) Store value



v_accvgpr_read v14, a20 // (op 22893) 
v_accvgpr_read v15, a21 // (op 22893) 
v_accvgpr_read v16, a22 // (op 22893) 
v_accvgpr_read v17, a23 // (op 22893) 
v_mul_f32 v10, v19, v14 // (op 22893) 
v_mul_f32 v11, v19, v15 // (op 22893) 
v_mul_f32 v12, v19, v16 // (op 22893) 
v_mul_f32 v13, v19, v17 // (op 22893) 

v_lshrrev_b32 v14, 16, v10 // (op 22895) 
v_lshrrev_b32 v15, 16, v11 // (op 22895) 
v_lshrrev_b32 v16, 16, v12 // (op 22895) 
v_lshrrev_b32 v17, 16, v13 // (op 22895) 


v_pack_b32_f16 v12, v14, v15 // (op 22896) 
v_pack_b32_f16 v13, v16, v17 // (op 22896) 

s_lshl_b32 s0, s4, 6 // (op 22896) 
s_mul_i32 s1, 5, s0 // (op 22896) 

v_add_u32 v0, v21, s1 // (op 22896) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22896) Store value



v_accvgpr_read v14, a24 // (op 22907) 
v_accvgpr_read v15, a25 // (op 22907) 
v_accvgpr_read v16, a26 // (op 22907) 
v_accvgpr_read v17, a27 // (op 22907) 
v_mul_f32 v10, v19, v14 // (op 22907) 
v_mul_f32 v11, v19, v15 // (op 22907) 
v_mul_f32 v12, v19, v16 // (op 22907) 
v_mul_f32 v13, v19, v17 // (op 22907) 

v_lshrrev_b32 v14, 16, v10 // (op 22909) 
v_lshrrev_b32 v15, 16, v11 // (op 22909) 
v_lshrrev_b32 v16, 16, v12 // (op 22909) 
v_lshrrev_b32 v17, 16, v13 // (op 22909) 


v_pack_b32_f16 v12, v14, v15 // (op 22910) 
v_pack_b32_f16 v13, v16, v17 // (op 22910) 

s_lshl_b32 s0, s4, 6 // (op 22910) 
s_mul_i32 s1, 6, s0 // (op 22910) 

v_add_u32 v0, v21, s1 // (op 22910) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22910) Store value



v_accvgpr_read v14, a28 // (op 22921) 
v_accvgpr_read v15, a29 // (op 22921) 
v_accvgpr_read v16, a30 // (op 22921) 
v_accvgpr_read v17, a31 // (op 22921) 
v_mul_f32 v10, v19, v14 // (op 22921) 
v_mul_f32 v11, v19, v15 // (op 22921) 
v_mul_f32 v12, v19, v16 // (op 22921) 
v_mul_f32 v13, v19, v17 // (op 22921) 

v_lshrrev_b32 v14, 16, v10 // (op 22923) 
v_lshrrev_b32 v15, 16, v11 // (op 22923) 
v_lshrrev_b32 v16, 16, v12 // (op 22923) 
v_lshrrev_b32 v17, 16, v13 // (op 22923) 


v_pack_b32_f16 v12, v14, v15 // (op 22924) 
v_pack_b32_f16 v13, v16, v17 // (op 22924) 

s_lshl_b32 s0, s4, 6 // (op 22924) 
s_mul_i32 s1, 7, s0 // (op 22924) 

v_add_u32 v0, v21, s1 // (op 22924) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22924) Store value



v_accvgpr_read v14, a32 // (op 22977) 
v_accvgpr_read v15, a33 // (op 22977) 
v_accvgpr_read v16, a34 // (op 22977) 
v_accvgpr_read v17, a35 // (op 22977) 
v_mul_f32 v10, v19, v14 // (op 22977) 
v_mul_f32 v11, v19, v15 // (op 22977) 
v_mul_f32 v12, v19, v16 // (op 22977) 
v_mul_f32 v13, v19, v17 // (op 22977) 

v_lshrrev_b32 v14, 16, v10 // (op 22979) 
v_lshrrev_b32 v15, 16, v11 // (op 22979) 
v_lshrrev_b32 v16, 16, v12 // (op 22979) 
v_lshrrev_b32 v17, 16, v13 // (op 22979) 


v_pack_b32_f16 v12, v14, v15 // (op 22981) 
v_pack_b32_f16 v13, v16, v17 // (op 22981) 

v_add_u32 v0, 64, v21 // (op 22981) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22981) Store value



v_accvgpr_read v14, a36 // (op 22999) 
v_accvgpr_read v15, a37 // (op 22999) 
v_accvgpr_read v16, a38 // (op 22999) 
v_accvgpr_read v17, a39 // (op 22999) 
v_mul_f32 v10, v19, v14 // (op 22999) 
v_mul_f32 v11, v19, v15 // (op 22999) 
v_mul_f32 v12, v19, v16 // (op 22999) 
v_mul_f32 v13, v19, v17 // (op 22999) 

v_lshrrev_b32 v14, 16, v10 // (op 23001) 
v_lshrrev_b32 v15, 16, v11 // (op 23001) 
v_lshrrev_b32 v16, 16, v12 // (op 23001) 
v_lshrrev_b32 v17, 16, v13 // (op 23001) 


v_pack_b32_f16 v12, v14, v15 // (op 22983) 
v_pack_b32_f16 v13, v16, v17 // (op 22983) 

s_lshl_b32 s0, s4, 6 // (op 22983) 
s_add_u32 s0, s0, 64 // (op 22983) 
v_add_u32 v0, v21, s0 // (op 22983) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22983) Store value



v_accvgpr_read v14, a40 // (op 23005) 
v_accvgpr_read v15, a41 // (op 23005) 
v_accvgpr_read v16, a42 // (op 23005) 
v_accvgpr_read v17, a43 // (op 23005) 
v_mul_f32 v10, v19, v14 // (op 23005) 
v_mul_f32 v11, v19, v15 // (op 23005) 
v_mul_f32 v12, v19, v16 // (op 23005) 
v_mul_f32 v13, v19, v17 // (op 23005) 

v_lshrrev_b32 v14, 16, v10 // (op 23007) 
v_lshrrev_b32 v15, 16, v11 // (op 23007) 
v_lshrrev_b32 v16, 16, v12 // (op 23007) 
v_lshrrev_b32 v17, 16, v13 // (op 23007) 


v_pack_b32_f16 v12, v14, v15 // (op 22985) 
v_pack_b32_f16 v13, v16, v17 // (op 22985) 

s_lshl_b32 s0, s4, 7 // (op 22985) 
s_add_u32 s0, s0, 64 // (op 22985) 
v_add_u32 v0, v21, s0 // (op 22985) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22985) Store value



v_accvgpr_read v14, a44 // (op 23011) 
v_accvgpr_read v15, a45 // (op 23011) 
v_accvgpr_read v16, a46 // (op 23011) 
v_accvgpr_read v17, a47 // (op 23011) 
v_mul_f32 v10, v19, v14 // (op 23011) 
v_mul_f32 v11, v19, v15 // (op 23011) 
v_mul_f32 v12, v19, v16 // (op 23011) 
v_mul_f32 v13, v19, v17 // (op 23011) 

v_lshrrev_b32 v14, 16, v10 // (op 23013) 
v_lshrrev_b32 v15, 16, v11 // (op 23013) 
v_lshrrev_b32 v16, 16, v12 // (op 23013) 
v_lshrrev_b32 v17, 16, v13 // (op 23013) 


v_pack_b32_f16 v12, v14, v15 // (op 22987) 
v_pack_b32_f16 v13, v16, v17 // (op 22987) 

s_lshl_b32 s0, s4, 6 // (op 22987) 
s_mul_i32 s1, 3, s0 // (op 22987) 
s_add_u32 s1, s1, 64 // (op 22987) 

v_add_u32 v0, v21, s1 // (op 22987) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22987) Store value



v_accvgpr_read v14, a48 // (op 23017) 
v_accvgpr_read v15, a49 // (op 23017) 
v_accvgpr_read v16, a50 // (op 23017) 
v_accvgpr_read v17, a51 // (op 23017) 
v_mul_f32 v10, v19, v14 // (op 23017) 
v_mul_f32 v11, v19, v15 // (op 23017) 
v_mul_f32 v12, v19, v16 // (op 23017) 
v_mul_f32 v13, v19, v17 // (op 23017) 

v_lshrrev_b32 v14, 16, v10 // (op 23019) 
v_lshrrev_b32 v15, 16, v11 // (op 23019) 
v_lshrrev_b32 v16, 16, v12 // (op 23019) 
v_lshrrev_b32 v17, 16, v13 // (op 23019) 


v_pack_b32_f16 v12, v14, v15 // (op 22989) 
v_pack_b32_f16 v13, v16, v17 // (op 22989) 

s_lshl_b32 s0, s4, 8 // (op 22989) 
s_add_u32 s0, s0, 64 // (op 22989) 
v_add_u32 v0, v21, s0 // (op 22989) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22989) Store value



v_accvgpr_read v14, a52 // (op 23023) 
v_accvgpr_read v15, a53 // (op 23023) 
v_accvgpr_read v16, a54 // (op 23023) 
v_accvgpr_read v17, a55 // (op 23023) 
v_mul_f32 v10, v19, v14 // (op 23023) 
v_mul_f32 v11, v19, v15 // (op 23023) 
v_mul_f32 v12, v19, v16 // (op 23023) 
v_mul_f32 v13, v19, v17 // (op 23023) 

v_lshrrev_b32 v14, 16, v10 // (op 23025) 
v_lshrrev_b32 v15, 16, v11 // (op 23025) 
v_lshrrev_b32 v16, 16, v12 // (op 23025) 
v_lshrrev_b32 v17, 16, v13 // (op 23025) 


v_pack_b32_f16 v12, v14, v15 // (op 22991) 
v_pack_b32_f16 v13, v16, v17 // (op 22991) 

s_lshl_b32 s0, s4, 6 // (op 22991) 
s_mul_i32 s1, 5, s0 // (op 22991) 
s_add_u32 s1, s1, 64 // (op 22991) 

v_add_u32 v0, v21, s1 // (op 22991) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22991) Store value



v_accvgpr_read v14, a56 // (op 23029) 
v_accvgpr_read v15, a57 // (op 23029) 
v_accvgpr_read v16, a58 // (op 23029) 
v_accvgpr_read v17, a59 // (op 23029) 
v_mul_f32 v10, v19, v14 // (op 23029) 
v_mul_f32 v11, v19, v15 // (op 23029) 
v_mul_f32 v12, v19, v16 // (op 23029) 
v_mul_f32 v13, v19, v17 // (op 23029) 

v_lshrrev_b32 v14, 16, v10 // (op 23031) 
v_lshrrev_b32 v15, 16, v11 // (op 23031) 
v_lshrrev_b32 v16, 16, v12 // (op 23031) 
v_lshrrev_b32 v17, 16, v13 // (op 23031) 


v_pack_b32_f16 v12, v14, v15 // (op 22993) 
v_pack_b32_f16 v13, v16, v17 // (op 22993) 

s_lshl_b32 s0, s4, 6 // (op 22993) 
s_mul_i32 s1, 6, s0 // (op 22993) 
s_add_u32 s1, s1, 64 // (op 22993) 

v_add_u32 v0, v21, s1 // (op 22993) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22993) Store value



v_accvgpr_read v14, a60 // (op 23035) 
v_accvgpr_read v15, a61 // (op 23035) 
v_accvgpr_read v16, a62 // (op 23035) 
v_accvgpr_read v17, a63 // (op 23035) 
v_mul_f32 v10, v19, v14 // (op 23035) 
v_mul_f32 v11, v19, v15 // (op 23035) 
v_mul_f32 v12, v19, v16 // (op 23035) 
v_mul_f32 v13, v19, v17 // (op 23035) 

v_lshrrev_b32 v14, 16, v10 // (op 23037) 
v_lshrrev_b32 v15, 16, v11 // (op 23037) 
v_lshrrev_b32 v16, 16, v12 // (op 23037) 
v_lshrrev_b32 v17, 16, v13 // (op 23037) 


v_pack_b32_f16 v12, v14, v15 // (op 22995) 
v_pack_b32_f16 v13, v16, v17 // (op 22995) 

s_lshl_b32 s0, s4, 6 // (op 22995) 
s_mul_i32 s1, 7, s0 // (op 22995) 
s_add_u32 s1, s1, 64 // (op 22995) 

v_add_u32 v0, v21, s1 // (op 22995) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 22995) Store value



v_accvgpr_read v14, a64 // (op 23125) 
v_accvgpr_read v15, a65 // (op 23125) 
v_accvgpr_read v16, a66 // (op 23125) 
v_accvgpr_read v17, a67 // (op 23125) 
v_mul_f32 v10, v19, v14 // (op 23125) 
v_mul_f32 v11, v19, v15 // (op 23125) 
v_mul_f32 v12, v19, v16 // (op 23125) 
v_mul_f32 v13, v19, v17 // (op 23125) 

v_lshrrev_b32 v14, 16, v10 // (op 23127) 
v_lshrrev_b32 v15, 16, v11 // (op 23127) 
v_lshrrev_b32 v16, 16, v12 // (op 23127) 
v_lshrrev_b32 v17, 16, v13 // (op 23127) 


v_pack_b32_f16 v12, v14, v15 // (op 23129) 
v_pack_b32_f16 v13, v16, v17 // (op 23129) 

v_add_u32 v0, 128, v21 // (op 23129) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23129) Store value



v_accvgpr_read v14, a68 // (op 23147) 
v_accvgpr_read v15, a69 // (op 23147) 
v_accvgpr_read v16, a70 // (op 23147) 
v_accvgpr_read v17, a71 // (op 23147) 
v_mul_f32 v10, v19, v14 // (op 23147) 
v_mul_f32 v11, v19, v15 // (op 23147) 
v_mul_f32 v12, v19, v16 // (op 23147) 
v_mul_f32 v13, v19, v17 // (op 23147) 

v_lshrrev_b32 v14, 16, v10 // (op 23149) 
v_lshrrev_b32 v15, 16, v11 // (op 23149) 
v_lshrrev_b32 v16, 16, v12 // (op 23149) 
v_lshrrev_b32 v17, 16, v13 // (op 23149) 


v_pack_b32_f16 v12, v14, v15 // (op 23131) 
v_pack_b32_f16 v13, v16, v17 // (op 23131) 

s_lshl_b32 s0, s4, 6 // (op 23131) 
s_add_u32 s0, s0, 128 // (op 23131) 
v_add_u32 v0, v21, s0 // (op 23131) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23131) Store value



v_accvgpr_read v14, a72 // (op 23153) 
v_accvgpr_read v15, a73 // (op 23153) 
v_accvgpr_read v16, a74 // (op 23153) 
v_accvgpr_read v17, a75 // (op 23153) 
v_mul_f32 v10, v19, v14 // (op 23153) 
v_mul_f32 v11, v19, v15 // (op 23153) 
v_mul_f32 v12, v19, v16 // (op 23153) 
v_mul_f32 v13, v19, v17 // (op 23153) 

v_lshrrev_b32 v14, 16, v10 // (op 23155) 
v_lshrrev_b32 v15, 16, v11 // (op 23155) 
v_lshrrev_b32 v16, 16, v12 // (op 23155) 
v_lshrrev_b32 v17, 16, v13 // (op 23155) 


v_pack_b32_f16 v12, v14, v15 // (op 23133) 
v_pack_b32_f16 v13, v16, v17 // (op 23133) 

s_lshl_b32 s0, s4, 7 // (op 23133) 
s_add_u32 s0, s0, 128 // (op 23133) 
v_add_u32 v0, v21, s0 // (op 23133) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23133) Store value



v_accvgpr_read v14, a76 // (op 23159) 
v_accvgpr_read v15, a77 // (op 23159) 
v_accvgpr_read v16, a78 // (op 23159) 
v_accvgpr_read v17, a79 // (op 23159) 
v_mul_f32 v10, v19, v14 // (op 23159) 
v_mul_f32 v11, v19, v15 // (op 23159) 
v_mul_f32 v12, v19, v16 // (op 23159) 
v_mul_f32 v13, v19, v17 // (op 23159) 

v_lshrrev_b32 v14, 16, v10 // (op 23161) 
v_lshrrev_b32 v15, 16, v11 // (op 23161) 
v_lshrrev_b32 v16, 16, v12 // (op 23161) 
v_lshrrev_b32 v17, 16, v13 // (op 23161) 


v_pack_b32_f16 v12, v14, v15 // (op 23135) 
v_pack_b32_f16 v13, v16, v17 // (op 23135) 

s_lshl_b32 s0, s4, 6 // (op 23135) 
s_mul_i32 s1, 3, s0 // (op 23135) 
s_add_u32 s1, s1, 128 // (op 23135) 

v_add_u32 v0, v21, s1 // (op 23135) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23135) Store value



v_accvgpr_read v14, a80 // (op 23165) 
v_accvgpr_read v15, a81 // (op 23165) 
v_accvgpr_read v16, a82 // (op 23165) 
v_accvgpr_read v17, a83 // (op 23165) 
v_mul_f32 v10, v19, v14 // (op 23165) 
v_mul_f32 v11, v19, v15 // (op 23165) 
v_mul_f32 v12, v19, v16 // (op 23165) 
v_mul_f32 v13, v19, v17 // (op 23165) 

v_lshrrev_b32 v14, 16, v10 // (op 23167) 
v_lshrrev_b32 v15, 16, v11 // (op 23167) 
v_lshrrev_b32 v16, 16, v12 // (op 23167) 
v_lshrrev_b32 v17, 16, v13 // (op 23167) 


v_pack_b32_f16 v12, v14, v15 // (op 23137) 
v_pack_b32_f16 v13, v16, v17 // (op 23137) 

s_lshl_b32 s0, s4, 8 // (op 23137) 
s_add_u32 s0, s0, 128 // (op 23137) 
v_add_u32 v0, v21, s0 // (op 23137) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23137) Store value



v_accvgpr_read v14, a84 // (op 23171) 
v_accvgpr_read v15, a85 // (op 23171) 
v_accvgpr_read v16, a86 // (op 23171) 
v_accvgpr_read v17, a87 // (op 23171) 
v_mul_f32 v10, v19, v14 // (op 23171) 
v_mul_f32 v11, v19, v15 // (op 23171) 
v_mul_f32 v12, v19, v16 // (op 23171) 
v_mul_f32 v13, v19, v17 // (op 23171) 

v_lshrrev_b32 v14, 16, v10 // (op 23173) 
v_lshrrev_b32 v15, 16, v11 // (op 23173) 
v_lshrrev_b32 v16, 16, v12 // (op 23173) 
v_lshrrev_b32 v17, 16, v13 // (op 23173) 


v_pack_b32_f16 v12, v14, v15 // (op 23139) 
v_pack_b32_f16 v13, v16, v17 // (op 23139) 

s_lshl_b32 s0, s4, 6 // (op 23139) 
s_mul_i32 s1, 5, s0 // (op 23139) 
s_add_u32 s1, s1, 128 // (op 23139) 

v_add_u32 v0, v21, s1 // (op 23139) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23139) Store value



v_accvgpr_read v14, a88 // (op 23177) 
v_accvgpr_read v15, a89 // (op 23177) 
v_accvgpr_read v16, a90 // (op 23177) 
v_accvgpr_read v17, a91 // (op 23177) 
v_mul_f32 v10, v19, v14 // (op 23177) 
v_mul_f32 v11, v19, v15 // (op 23177) 
v_mul_f32 v12, v19, v16 // (op 23177) 
v_mul_f32 v13, v19, v17 // (op 23177) 

v_lshrrev_b32 v14, 16, v10 // (op 23179) 
v_lshrrev_b32 v15, 16, v11 // (op 23179) 
v_lshrrev_b32 v16, 16, v12 // (op 23179) 
v_lshrrev_b32 v17, 16, v13 // (op 23179) 


v_pack_b32_f16 v12, v14, v15 // (op 23141) 
v_pack_b32_f16 v13, v16, v17 // (op 23141) 

s_lshl_b32 s0, s4, 6 // (op 23141) 
s_mul_i32 s1, 6, s0 // (op 23141) 
s_add_u32 s1, s1, 128 // (op 23141) 

v_add_u32 v0, v21, s1 // (op 23141) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23141) Store value



v_accvgpr_read v14, a92 // (op 23183) 
v_accvgpr_read v15, a93 // (op 23183) 
v_accvgpr_read v16, a94 // (op 23183) 
v_accvgpr_read v17, a95 // (op 23183) 
v_mul_f32 v10, v19, v14 // (op 23183) 
v_mul_f32 v11, v19, v15 // (op 23183) 
v_mul_f32 v12, v19, v16 // (op 23183) 
v_mul_f32 v13, v19, v17 // (op 23183) 

v_lshrrev_b32 v14, 16, v10 // (op 23185) 
v_lshrrev_b32 v15, 16, v11 // (op 23185) 
v_lshrrev_b32 v16, 16, v12 // (op 23185) 
v_lshrrev_b32 v17, 16, v13 // (op 23185) 


v_pack_b32_f16 v12, v14, v15 // (op 23143) 
v_pack_b32_f16 v13, v16, v17 // (op 23143) 

s_lshl_b32 s0, s4, 6 // (op 23143) 
s_mul_i32 s1, 7, s0 // (op 23143) 
s_add_u32 s1, s1, 128 // (op 23143) 

v_add_u32 v0, v21, s1 // (op 23143) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23143) Store value



v_accvgpr_read v14, a96 // (op 23273) 
v_accvgpr_read v15, a97 // (op 23273) 
v_accvgpr_read v16, a98 // (op 23273) 
v_accvgpr_read v17, a99 // (op 23273) 
v_mul_f32 v10, v19, v14 // (op 23273) 
v_mul_f32 v11, v19, v15 // (op 23273) 
v_mul_f32 v12, v19, v16 // (op 23273) 
v_mul_f32 v13, v19, v17 // (op 23273) 

v_lshrrev_b32 v14, 16, v10 // (op 23275) 
v_lshrrev_b32 v15, 16, v11 // (op 23275) 
v_lshrrev_b32 v16, 16, v12 // (op 23275) 
v_lshrrev_b32 v17, 16, v13 // (op 23275) 


v_pack_b32_f16 v12, v14, v15 // (op 23277) 
v_pack_b32_f16 v13, v16, v17 // (op 23277) 

v_add_u32 v0, 192, v21 // (op 23277) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23277) Store value



v_accvgpr_read v14, a100 // (op 23295) 
v_accvgpr_read v15, a101 // (op 23295) 
v_accvgpr_read v16, a102 // (op 23295) 
v_accvgpr_read v17, a103 // (op 23295) 
v_mul_f32 v10, v19, v14 // (op 23295) 
v_mul_f32 v11, v19, v15 // (op 23295) 
v_mul_f32 v12, v19, v16 // (op 23295) 
v_mul_f32 v13, v19, v17 // (op 23295) 

v_lshrrev_b32 v14, 16, v10 // (op 23297) 
v_lshrrev_b32 v15, 16, v11 // (op 23297) 
v_lshrrev_b32 v16, 16, v12 // (op 23297) 
v_lshrrev_b32 v17, 16, v13 // (op 23297) 


v_pack_b32_f16 v12, v14, v15 // (op 23279) 
v_pack_b32_f16 v13, v16, v17 // (op 23279) 

s_lshl_b32 s0, s4, 6 // (op 23279) 
s_add_u32 s0, s0, 192 // (op 23279) 
v_add_u32 v0, v21, s0 // (op 23279) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23279) Store value



v_accvgpr_read v14, a104 // (op 23301) 
v_accvgpr_read v15, a105 // (op 23301) 
v_accvgpr_read v16, a106 // (op 23301) 
v_accvgpr_read v17, a107 // (op 23301) 
v_mul_f32 v10, v19, v14 // (op 23301) 
v_mul_f32 v11, v19, v15 // (op 23301) 
v_mul_f32 v12, v19, v16 // (op 23301) 
v_mul_f32 v13, v19, v17 // (op 23301) 

v_lshrrev_b32 v14, 16, v10 // (op 23303) 
v_lshrrev_b32 v15, 16, v11 // (op 23303) 
v_lshrrev_b32 v16, 16, v12 // (op 23303) 
v_lshrrev_b32 v17, 16, v13 // (op 23303) 


v_pack_b32_f16 v12, v14, v15 // (op 23281) 
v_pack_b32_f16 v13, v16, v17 // (op 23281) 

s_lshl_b32 s0, s4, 7 // (op 23281) 
s_add_u32 s0, s0, 192 // (op 23281) 
v_add_u32 v0, v21, s0 // (op 23281) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23281) Store value



v_accvgpr_read v14, a108 // (op 23307) 
v_accvgpr_read v15, a109 // (op 23307) 
v_accvgpr_read v16, a110 // (op 23307) 
v_accvgpr_read v17, a111 // (op 23307) 
v_mul_f32 v10, v19, v14 // (op 23307) 
v_mul_f32 v11, v19, v15 // (op 23307) 
v_mul_f32 v12, v19, v16 // (op 23307) 
v_mul_f32 v13, v19, v17 // (op 23307) 

v_lshrrev_b32 v14, 16, v10 // (op 23309) 
v_lshrrev_b32 v15, 16, v11 // (op 23309) 
v_lshrrev_b32 v16, 16, v12 // (op 23309) 
v_lshrrev_b32 v17, 16, v13 // (op 23309) 


v_pack_b32_f16 v12, v14, v15 // (op 23283) 
v_pack_b32_f16 v13, v16, v17 // (op 23283) 

s_lshl_b32 s0, s4, 6 // (op 23283) 
s_mul_i32 s1, 3, s0 // (op 23283) 
s_add_u32 s1, s1, 192 // (op 23283) 

v_add_u32 v0, v21, s1 // (op 23283) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23283) Store value



v_accvgpr_read v14, a112 // (op 23313) 
v_accvgpr_read v15, a113 // (op 23313) 
v_accvgpr_read v16, a114 // (op 23313) 
v_accvgpr_read v17, a115 // (op 23313) 
v_mul_f32 v10, v19, v14 // (op 23313) 
v_mul_f32 v11, v19, v15 // (op 23313) 
v_mul_f32 v12, v19, v16 // (op 23313) 
v_mul_f32 v13, v19, v17 // (op 23313) 

v_lshrrev_b32 v14, 16, v10 // (op 23315) 
v_lshrrev_b32 v15, 16, v11 // (op 23315) 
v_lshrrev_b32 v16, 16, v12 // (op 23315) 
v_lshrrev_b32 v17, 16, v13 // (op 23315) 


v_pack_b32_f16 v12, v14, v15 // (op 23285) 
v_pack_b32_f16 v13, v16, v17 // (op 23285) 

s_lshl_b32 s0, s4, 8 // (op 23285) 
s_add_u32 s0, s0, 192 // (op 23285) 
v_add_u32 v0, v21, s0 // (op 23285) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23285) Store value



v_accvgpr_read v14, a116 // (op 23319) 
v_accvgpr_read v15, a117 // (op 23319) 
v_accvgpr_read v16, a118 // (op 23319) 
v_accvgpr_read v17, a119 // (op 23319) 
v_mul_f32 v10, v19, v14 // (op 23319) 
v_mul_f32 v11, v19, v15 // (op 23319) 
v_mul_f32 v12, v19, v16 // (op 23319) 
v_mul_f32 v13, v19, v17 // (op 23319) 

v_lshrrev_b32 v14, 16, v10 // (op 23321) 
v_lshrrev_b32 v15, 16, v11 // (op 23321) 
v_lshrrev_b32 v16, 16, v12 // (op 23321) 
v_lshrrev_b32 v17, 16, v13 // (op 23321) 


v_pack_b32_f16 v12, v14, v15 // (op 23287) 
v_pack_b32_f16 v13, v16, v17 // (op 23287) 

s_lshl_b32 s0, s4, 6 // (op 23287) 
s_mul_i32 s1, 5, s0 // (op 23287) 
s_add_u32 s1, s1, 192 // (op 23287) 

v_add_u32 v0, v21, s1 // (op 23287) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23287) Store value



v_accvgpr_read v14, a120 // (op 23325) 
v_accvgpr_read v15, a121 // (op 23325) 
v_accvgpr_read v16, a122 // (op 23325) 
v_accvgpr_read v17, a123 // (op 23325) 
v_mul_f32 v10, v19, v14 // (op 23325) 
v_mul_f32 v11, v19, v15 // (op 23325) 
v_mul_f32 v12, v19, v16 // (op 23325) 
v_mul_f32 v13, v19, v17 // (op 23325) 

v_lshrrev_b32 v14, 16, v10 // (op 23327) 
v_lshrrev_b32 v15, 16, v11 // (op 23327) 
v_lshrrev_b32 v16, 16, v12 // (op 23327) 
v_lshrrev_b32 v17, 16, v13 // (op 23327) 


v_pack_b32_f16 v12, v14, v15 // (op 23289) 
v_pack_b32_f16 v13, v16, v17 // (op 23289) 

s_lshl_b32 s0, s4, 6 // (op 23289) 
s_mul_i32 s1, 6, s0 // (op 23289) 
s_add_u32 s1, s1, 192 // (op 23289) 

v_add_u32 v0, v21, s1 // (op 23289) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23289) Store value



v_accvgpr_read v14, a124 // (op 23331) 
v_accvgpr_read v15, a125 // (op 23331) 
v_accvgpr_read v16, a126 // (op 23331) 
v_accvgpr_read v17, a127 // (op 23331) 
v_mul_f32 v10, v19, v14 // (op 23331) 
v_mul_f32 v11, v19, v15 // (op 23331) 
v_mul_f32 v12, v19, v16 // (op 23331) 
v_mul_f32 v13, v19, v17 // (op 23331) 

v_lshrrev_b32 v14, 16, v10 // (op 23333) 
v_lshrrev_b32 v15, 16, v11 // (op 23333) 
v_lshrrev_b32 v16, 16, v12 // (op 23333) 
v_lshrrev_b32 v17, 16, v13 // (op 23333) 


v_pack_b32_f16 v12, v14, v15 // (op 23291) 
v_pack_b32_f16 v13, v16, v17 // (op 23291) 

s_lshl_b32 s0, s4, 6 // (op 23291) 
s_mul_i32 s1, 7, s0 // (op 23291) 
s_add_u32 s1, s1, 192 // (op 23291) 

v_add_u32 v0, v21, s1 // (op 23291) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23291) Store value



v_accvgpr_read v14, a128 // (op 23421) 
v_accvgpr_read v15, a129 // (op 23421) 
v_accvgpr_read v16, a130 // (op 23421) 
v_accvgpr_read v17, a131 // (op 23421) 
v_mul_f32 v10, v19, v14 // (op 23421) 
v_mul_f32 v11, v19, v15 // (op 23421) 
v_mul_f32 v12, v19, v16 // (op 23421) 
v_mul_f32 v13, v19, v17 // (op 23421) 

v_lshrrev_b32 v14, 16, v10 // (op 23423) 
v_lshrrev_b32 v15, 16, v11 // (op 23423) 
v_lshrrev_b32 v16, 16, v12 // (op 23423) 
v_lshrrev_b32 v17, 16, v13 // (op 23423) 


v_pack_b32_f16 v12, v14, v15 // (op 23425) 
v_pack_b32_f16 v13, v16, v17 // (op 23425) 

v_add_u32 v0, 256, v21 // (op 23425) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23425) Store value



v_accvgpr_read v14, a132 // (op 23443) 
v_accvgpr_read v15, a133 // (op 23443) 
v_accvgpr_read v16, a134 // (op 23443) 
v_accvgpr_read v17, a135 // (op 23443) 
v_mul_f32 v10, v19, v14 // (op 23443) 
v_mul_f32 v11, v19, v15 // (op 23443) 
v_mul_f32 v12, v19, v16 // (op 23443) 
v_mul_f32 v13, v19, v17 // (op 23443) 

v_lshrrev_b32 v14, 16, v10 // (op 23445) 
v_lshrrev_b32 v15, 16, v11 // (op 23445) 
v_lshrrev_b32 v16, 16, v12 // (op 23445) 
v_lshrrev_b32 v17, 16, v13 // (op 23445) 


v_pack_b32_f16 v12, v14, v15 // (op 23427) 
v_pack_b32_f16 v13, v16, v17 // (op 23427) 

s_lshl_b32 s0, s4, 6 // (op 23427) 
s_add_u32 s0, s0, 256 // (op 23427) 
v_add_u32 v0, v21, s0 // (op 23427) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23427) Store value



v_accvgpr_read v14, a136 // (op 23449) 
v_accvgpr_read v15, a137 // (op 23449) 
v_accvgpr_read v16, a138 // (op 23449) 
v_accvgpr_read v17, a139 // (op 23449) 
v_mul_f32 v10, v19, v14 // (op 23449) 
v_mul_f32 v11, v19, v15 // (op 23449) 
v_mul_f32 v12, v19, v16 // (op 23449) 
v_mul_f32 v13, v19, v17 // (op 23449) 

v_lshrrev_b32 v14, 16, v10 // (op 23451) 
v_lshrrev_b32 v15, 16, v11 // (op 23451) 
v_lshrrev_b32 v16, 16, v12 // (op 23451) 
v_lshrrev_b32 v17, 16, v13 // (op 23451) 


v_pack_b32_f16 v12, v14, v15 // (op 23429) 
v_pack_b32_f16 v13, v16, v17 // (op 23429) 

s_lshl_b32 s0, s4, 7 // (op 23429) 
s_add_u32 s0, s0, 256 // (op 23429) 
v_add_u32 v0, v21, s0 // (op 23429) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23429) Store value



v_accvgpr_read v14, a140 // (op 23455) 
v_accvgpr_read v15, a141 // (op 23455) 
v_accvgpr_read v16, a142 // (op 23455) 
v_accvgpr_read v17, a143 // (op 23455) 
v_mul_f32 v10, v19, v14 // (op 23455) 
v_mul_f32 v11, v19, v15 // (op 23455) 
v_mul_f32 v12, v19, v16 // (op 23455) 
v_mul_f32 v13, v19, v17 // (op 23455) 

v_lshrrev_b32 v14, 16, v10 // (op 23457) 
v_lshrrev_b32 v15, 16, v11 // (op 23457) 
v_lshrrev_b32 v16, 16, v12 // (op 23457) 
v_lshrrev_b32 v17, 16, v13 // (op 23457) 


v_pack_b32_f16 v12, v14, v15 // (op 23431) 
v_pack_b32_f16 v13, v16, v17 // (op 23431) 

s_lshl_b32 s0, s4, 6 // (op 23431) 
s_mul_i32 s1, 3, s0 // (op 23431) 
s_add_u32 s1, s1, 256 // (op 23431) 

v_add_u32 v0, v21, s1 // (op 23431) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23431) Store value



v_accvgpr_read v14, a144 // (op 23461) 
v_accvgpr_read v15, a145 // (op 23461) 
v_accvgpr_read v16, a146 // (op 23461) 
v_accvgpr_read v17, a147 // (op 23461) 
v_mul_f32 v10, v19, v14 // (op 23461) 
v_mul_f32 v11, v19, v15 // (op 23461) 
v_mul_f32 v12, v19, v16 // (op 23461) 
v_mul_f32 v13, v19, v17 // (op 23461) 

v_lshrrev_b32 v14, 16, v10 // (op 23463) 
v_lshrrev_b32 v15, 16, v11 // (op 23463) 
v_lshrrev_b32 v16, 16, v12 // (op 23463) 
v_lshrrev_b32 v17, 16, v13 // (op 23463) 


v_pack_b32_f16 v12, v14, v15 // (op 23433) 
v_pack_b32_f16 v13, v16, v17 // (op 23433) 

s_lshl_b32 s0, s4, 8 // (op 23433) 
s_add_u32 s0, s0, 256 // (op 23433) 
v_add_u32 v0, v21, s0 // (op 23433) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23433) Store value



v_accvgpr_read v14, a148 // (op 23467) 
v_accvgpr_read v15, a149 // (op 23467) 
v_accvgpr_read v16, a150 // (op 23467) 
v_accvgpr_read v17, a151 // (op 23467) 
v_mul_f32 v10, v19, v14 // (op 23467) 
v_mul_f32 v11, v19, v15 // (op 23467) 
v_mul_f32 v12, v19, v16 // (op 23467) 
v_mul_f32 v13, v19, v17 // (op 23467) 

v_lshrrev_b32 v14, 16, v10 // (op 23469) 
v_lshrrev_b32 v15, 16, v11 // (op 23469) 
v_lshrrev_b32 v16, 16, v12 // (op 23469) 
v_lshrrev_b32 v17, 16, v13 // (op 23469) 


v_pack_b32_f16 v12, v14, v15 // (op 23435) 
v_pack_b32_f16 v13, v16, v17 // (op 23435) 

s_lshl_b32 s0, s4, 6 // (op 23435) 
s_mul_i32 s1, 5, s0 // (op 23435) 
s_add_u32 s1, s1, 256 // (op 23435) 

v_add_u32 v0, v21, s1 // (op 23435) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23435) Store value



v_accvgpr_read v14, a152 // (op 23473) 
v_accvgpr_read v15, a153 // (op 23473) 
v_accvgpr_read v16, a154 // (op 23473) 
v_accvgpr_read v17, a155 // (op 23473) 
v_mul_f32 v10, v19, v14 // (op 23473) 
v_mul_f32 v11, v19, v15 // (op 23473) 
v_mul_f32 v12, v19, v16 // (op 23473) 
v_mul_f32 v13, v19, v17 // (op 23473) 

v_lshrrev_b32 v14, 16, v10 // (op 23475) 
v_lshrrev_b32 v15, 16, v11 // (op 23475) 
v_lshrrev_b32 v16, 16, v12 // (op 23475) 
v_lshrrev_b32 v17, 16, v13 // (op 23475) 


v_pack_b32_f16 v12, v14, v15 // (op 23437) 
v_pack_b32_f16 v13, v16, v17 // (op 23437) 

s_lshl_b32 s0, s4, 6 // (op 23437) 
s_mul_i32 s1, 6, s0 // (op 23437) 
s_add_u32 s1, s1, 256 // (op 23437) 

v_add_u32 v0, v21, s1 // (op 23437) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23437) Store value



v_accvgpr_read v14, a156 // (op 23479) 
v_accvgpr_read v15, a157 // (op 23479) 
v_accvgpr_read v16, a158 // (op 23479) 
v_accvgpr_read v17, a159 // (op 23479) 
v_mul_f32 v10, v19, v14 // (op 23479) 
v_mul_f32 v11, v19, v15 // (op 23479) 
v_mul_f32 v12, v19, v16 // (op 23479) 
v_mul_f32 v13, v19, v17 // (op 23479) 

v_lshrrev_b32 v14, 16, v10 // (op 23481) 
v_lshrrev_b32 v15, 16, v11 // (op 23481) 
v_lshrrev_b32 v16, 16, v12 // (op 23481) 
v_lshrrev_b32 v17, 16, v13 // (op 23481) 


v_pack_b32_f16 v12, v14, v15 // (op 23439) 
v_pack_b32_f16 v13, v16, v17 // (op 23439) 

s_lshl_b32 s0, s4, 6 // (op 23439) 
s_mul_i32 s1, 7, s0 // (op 23439) 
s_add_u32 s1, s1, 256 // (op 23439) 

v_add_u32 v0, v21, s1 // (op 23439) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23439) Store value



v_accvgpr_read v14, a160 // (op 23569) 
v_accvgpr_read v15, a161 // (op 23569) 
v_accvgpr_read v16, a162 // (op 23569) 
v_accvgpr_read v17, a163 // (op 23569) 
v_mul_f32 v10, v19, v14 // (op 23569) 
v_mul_f32 v11, v19, v15 // (op 23569) 
v_mul_f32 v12, v19, v16 // (op 23569) 
v_mul_f32 v13, v19, v17 // (op 23569) 

v_lshrrev_b32 v14, 16, v10 // (op 23571) 
v_lshrrev_b32 v15, 16, v11 // (op 23571) 
v_lshrrev_b32 v16, 16, v12 // (op 23571) 
v_lshrrev_b32 v17, 16, v13 // (op 23571) 


v_pack_b32_f16 v12, v14, v15 // (op 23573) 
v_pack_b32_f16 v13, v16, v17 // (op 23573) 

v_add_u32 v0, 320, v21 // (op 23573) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23573) Store value



v_accvgpr_read v14, a164 // (op 23591) 
v_accvgpr_read v15, a165 // (op 23591) 
v_accvgpr_read v16, a166 // (op 23591) 
v_accvgpr_read v17, a167 // (op 23591) 
v_mul_f32 v10, v19, v14 // (op 23591) 
v_mul_f32 v11, v19, v15 // (op 23591) 
v_mul_f32 v12, v19, v16 // (op 23591) 
v_mul_f32 v13, v19, v17 // (op 23591) 

v_lshrrev_b32 v14, 16, v10 // (op 23593) 
v_lshrrev_b32 v15, 16, v11 // (op 23593) 
v_lshrrev_b32 v16, 16, v12 // (op 23593) 
v_lshrrev_b32 v17, 16, v13 // (op 23593) 


v_pack_b32_f16 v12, v14, v15 // (op 23575) 
v_pack_b32_f16 v13, v16, v17 // (op 23575) 

s_lshl_b32 s0, s4, 6 // (op 23575) 
s_add_u32 s0, s0, 320 // (op 23575) 
v_add_u32 v0, v21, s0 // (op 23575) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23575) Store value



v_accvgpr_read v14, a168 // (op 23597) 
v_accvgpr_read v15, a169 // (op 23597) 
v_accvgpr_read v16, a170 // (op 23597) 
v_accvgpr_read v17, a171 // (op 23597) 
v_mul_f32 v10, v19, v14 // (op 23597) 
v_mul_f32 v11, v19, v15 // (op 23597) 
v_mul_f32 v12, v19, v16 // (op 23597) 
v_mul_f32 v13, v19, v17 // (op 23597) 

v_lshrrev_b32 v14, 16, v10 // (op 23599) 
v_lshrrev_b32 v15, 16, v11 // (op 23599) 
v_lshrrev_b32 v16, 16, v12 // (op 23599) 
v_lshrrev_b32 v17, 16, v13 // (op 23599) 


v_pack_b32_f16 v12, v14, v15 // (op 23577) 
v_pack_b32_f16 v13, v16, v17 // (op 23577) 

s_lshl_b32 s0, s4, 7 // (op 23577) 
s_add_u32 s0, s0, 320 // (op 23577) 
v_add_u32 v0, v21, s0 // (op 23577) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23577) Store value



v_accvgpr_read v14, a172 // (op 23603) 
v_accvgpr_read v15, a173 // (op 23603) 
v_accvgpr_read v16, a174 // (op 23603) 
v_accvgpr_read v17, a175 // (op 23603) 
v_mul_f32 v10, v19, v14 // (op 23603) 
v_mul_f32 v11, v19, v15 // (op 23603) 
v_mul_f32 v12, v19, v16 // (op 23603) 
v_mul_f32 v13, v19, v17 // (op 23603) 

v_lshrrev_b32 v14, 16, v10 // (op 23605) 
v_lshrrev_b32 v15, 16, v11 // (op 23605) 
v_lshrrev_b32 v16, 16, v12 // (op 23605) 
v_lshrrev_b32 v17, 16, v13 // (op 23605) 


v_pack_b32_f16 v12, v14, v15 // (op 23579) 
v_pack_b32_f16 v13, v16, v17 // (op 23579) 

s_lshl_b32 s0, s4, 6 // (op 23579) 
s_mul_i32 s1, 3, s0 // (op 23579) 
s_add_u32 s1, s1, 320 // (op 23579) 

v_add_u32 v0, v21, s1 // (op 23579) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23579) Store value



v_accvgpr_read v14, a176 // (op 23609) 
v_accvgpr_read v15, a177 // (op 23609) 
v_accvgpr_read v16, a178 // (op 23609) 
v_accvgpr_read v17, a179 // (op 23609) 
v_mul_f32 v10, v19, v14 // (op 23609) 
v_mul_f32 v11, v19, v15 // (op 23609) 
v_mul_f32 v12, v19, v16 // (op 23609) 
v_mul_f32 v13, v19, v17 // (op 23609) 

v_lshrrev_b32 v14, 16, v10 // (op 23611) 
v_lshrrev_b32 v15, 16, v11 // (op 23611) 
v_lshrrev_b32 v16, 16, v12 // (op 23611) 
v_lshrrev_b32 v17, 16, v13 // (op 23611) 


v_pack_b32_f16 v12, v14, v15 // (op 23581) 
v_pack_b32_f16 v13, v16, v17 // (op 23581) 

s_lshl_b32 s0, s4, 8 // (op 23581) 
s_add_u32 s0, s0, 320 // (op 23581) 
v_add_u32 v0, v21, s0 // (op 23581) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23581) Store value



v_accvgpr_read v14, a180 // (op 23615) 
v_accvgpr_read v15, a181 // (op 23615) 
v_accvgpr_read v16, a182 // (op 23615) 
v_accvgpr_read v17, a183 // (op 23615) 
v_mul_f32 v10, v19, v14 // (op 23615) 
v_mul_f32 v11, v19, v15 // (op 23615) 
v_mul_f32 v12, v19, v16 // (op 23615) 
v_mul_f32 v13, v19, v17 // (op 23615) 

v_lshrrev_b32 v14, 16, v10 // (op 23617) 
v_lshrrev_b32 v15, 16, v11 // (op 23617) 
v_lshrrev_b32 v16, 16, v12 // (op 23617) 
v_lshrrev_b32 v17, 16, v13 // (op 23617) 


v_pack_b32_f16 v12, v14, v15 // (op 23583) 
v_pack_b32_f16 v13, v16, v17 // (op 23583) 

s_lshl_b32 s0, s4, 6 // (op 23583) 
s_mul_i32 s1, 5, s0 // (op 23583) 
s_add_u32 s1, s1, 320 // (op 23583) 

v_add_u32 v0, v21, s1 // (op 23583) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23583) Store value



v_accvgpr_read v14, a184 // (op 23621) 
v_accvgpr_read v15, a185 // (op 23621) 
v_accvgpr_read v16, a186 // (op 23621) 
v_accvgpr_read v17, a187 // (op 23621) 
v_mul_f32 v10, v19, v14 // (op 23621) 
v_mul_f32 v11, v19, v15 // (op 23621) 
v_mul_f32 v12, v19, v16 // (op 23621) 
v_mul_f32 v13, v19, v17 // (op 23621) 

v_lshrrev_b32 v14, 16, v10 // (op 23623) 
v_lshrrev_b32 v15, 16, v11 // (op 23623) 
v_lshrrev_b32 v16, 16, v12 // (op 23623) 
v_lshrrev_b32 v17, 16, v13 // (op 23623) 


v_pack_b32_f16 v12, v14, v15 // (op 23585) 
v_pack_b32_f16 v13, v16, v17 // (op 23585) 

s_lshl_b32 s0, s4, 6 // (op 23585) 
s_mul_i32 s1, 6, s0 // (op 23585) 
s_add_u32 s1, s1, 320 // (op 23585) 

v_add_u32 v0, v21, s1 // (op 23585) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23585) Store value



v_accvgpr_read v14, a188 // (op 23627) 
v_accvgpr_read v15, a189 // (op 23627) 
v_accvgpr_read v16, a190 // (op 23627) 
v_accvgpr_read v17, a191 // (op 23627) 
v_mul_f32 v10, v19, v14 // (op 23627) 
v_mul_f32 v11, v19, v15 // (op 23627) 
v_mul_f32 v12, v19, v16 // (op 23627) 
v_mul_f32 v13, v19, v17 // (op 23627) 

v_lshrrev_b32 v14, 16, v10 // (op 23629) 
v_lshrrev_b32 v15, 16, v11 // (op 23629) 
v_lshrrev_b32 v16, 16, v12 // (op 23629) 
v_lshrrev_b32 v17, 16, v13 // (op 23629) 


v_pack_b32_f16 v12, v14, v15 // (op 23587) 
v_pack_b32_f16 v13, v16, v17 // (op 23587) 

s_lshl_b32 s0, s4, 6 // (op 23587) 
s_mul_i32 s1, 7, s0 // (op 23587) 
s_add_u32 s1, s1, 320 // (op 23587) 

v_add_u32 v0, v21, s1 // (op 23587) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23587) Store value



v_accvgpr_read v14, a192 // (op 23717) 
v_accvgpr_read v15, a193 // (op 23717) 
v_accvgpr_read v16, a194 // (op 23717) 
v_accvgpr_read v17, a195 // (op 23717) 
v_mul_f32 v10, v19, v14 // (op 23717) 
v_mul_f32 v11, v19, v15 // (op 23717) 
v_mul_f32 v12, v19, v16 // (op 23717) 
v_mul_f32 v13, v19, v17 // (op 23717) 

v_lshrrev_b32 v14, 16, v10 // (op 23719) 
v_lshrrev_b32 v15, 16, v11 // (op 23719) 
v_lshrrev_b32 v16, 16, v12 // (op 23719) 
v_lshrrev_b32 v17, 16, v13 // (op 23719) 


v_pack_b32_f16 v12, v14, v15 // (op 23721) 
v_pack_b32_f16 v13, v16, v17 // (op 23721) 

v_add_u32 v0, 384, v21 // (op 23721) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23721) Store value



v_accvgpr_read v14, a196 // (op 23739) 
v_accvgpr_read v15, a197 // (op 23739) 
v_accvgpr_read v16, a198 // (op 23739) 
v_accvgpr_read v17, a199 // (op 23739) 
v_mul_f32 v10, v19, v14 // (op 23739) 
v_mul_f32 v11, v19, v15 // (op 23739) 
v_mul_f32 v12, v19, v16 // (op 23739) 
v_mul_f32 v13, v19, v17 // (op 23739) 

v_lshrrev_b32 v14, 16, v10 // (op 23741) 
v_lshrrev_b32 v15, 16, v11 // (op 23741) 
v_lshrrev_b32 v16, 16, v12 // (op 23741) 
v_lshrrev_b32 v17, 16, v13 // (op 23741) 


v_pack_b32_f16 v12, v14, v15 // (op 23723) 
v_pack_b32_f16 v13, v16, v17 // (op 23723) 

s_lshl_b32 s0, s4, 6 // (op 23723) 
s_add_u32 s0, s0, 384 // (op 23723) 
v_add_u32 v0, v21, s0 // (op 23723) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23723) Store value



v_accvgpr_read v14, a200 // (op 23745) 
v_accvgpr_read v15, a201 // (op 23745) 
v_accvgpr_read v16, a202 // (op 23745) 
v_accvgpr_read v17, a203 // (op 23745) 
v_mul_f32 v10, v19, v14 // (op 23745) 
v_mul_f32 v11, v19, v15 // (op 23745) 
v_mul_f32 v12, v19, v16 // (op 23745) 
v_mul_f32 v13, v19, v17 // (op 23745) 

v_lshrrev_b32 v14, 16, v10 // (op 23747) 
v_lshrrev_b32 v15, 16, v11 // (op 23747) 
v_lshrrev_b32 v16, 16, v12 // (op 23747) 
v_lshrrev_b32 v17, 16, v13 // (op 23747) 


v_pack_b32_f16 v12, v14, v15 // (op 23725) 
v_pack_b32_f16 v13, v16, v17 // (op 23725) 

s_lshl_b32 s0, s4, 7 // (op 23725) 
s_add_u32 s0, s0, 384 // (op 23725) 
v_add_u32 v0, v21, s0 // (op 23725) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23725) Store value



v_accvgpr_read v14, a204 // (op 23751) 
v_accvgpr_read v15, a205 // (op 23751) 
v_accvgpr_read v16, a206 // (op 23751) 
v_accvgpr_read v17, a207 // (op 23751) 
v_mul_f32 v10, v19, v14 // (op 23751) 
v_mul_f32 v11, v19, v15 // (op 23751) 
v_mul_f32 v12, v19, v16 // (op 23751) 
v_mul_f32 v13, v19, v17 // (op 23751) 

v_lshrrev_b32 v14, 16, v10 // (op 23753) 
v_lshrrev_b32 v15, 16, v11 // (op 23753) 
v_lshrrev_b32 v16, 16, v12 // (op 23753) 
v_lshrrev_b32 v17, 16, v13 // (op 23753) 


v_pack_b32_f16 v12, v14, v15 // (op 23727) 
v_pack_b32_f16 v13, v16, v17 // (op 23727) 

s_lshl_b32 s0, s4, 6 // (op 23727) 
s_mul_i32 s1, 3, s0 // (op 23727) 
s_add_u32 s1, s1, 384 // (op 23727) 

v_add_u32 v0, v21, s1 // (op 23727) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23727) Store value



v_accvgpr_read v14, a208 // (op 23757) 
v_accvgpr_read v15, a209 // (op 23757) 
v_accvgpr_read v16, a210 // (op 23757) 
v_accvgpr_read v17, a211 // (op 23757) 
v_mul_f32 v10, v19, v14 // (op 23757) 
v_mul_f32 v11, v19, v15 // (op 23757) 
v_mul_f32 v12, v19, v16 // (op 23757) 
v_mul_f32 v13, v19, v17 // (op 23757) 

v_lshrrev_b32 v14, 16, v10 // (op 23759) 
v_lshrrev_b32 v15, 16, v11 // (op 23759) 
v_lshrrev_b32 v16, 16, v12 // (op 23759) 
v_lshrrev_b32 v17, 16, v13 // (op 23759) 


v_pack_b32_f16 v12, v14, v15 // (op 23729) 
v_pack_b32_f16 v13, v16, v17 // (op 23729) 

s_lshl_b32 s0, s4, 8 // (op 23729) 
s_add_u32 s0, s0, 384 // (op 23729) 
v_add_u32 v0, v21, s0 // (op 23729) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23729) Store value



v_accvgpr_read v14, a212 // (op 23763) 
v_accvgpr_read v15, a213 // (op 23763) 
v_accvgpr_read v16, a214 // (op 23763) 
v_accvgpr_read v17, a215 // (op 23763) 
v_mul_f32 v10, v19, v14 // (op 23763) 
v_mul_f32 v11, v19, v15 // (op 23763) 
v_mul_f32 v12, v19, v16 // (op 23763) 
v_mul_f32 v13, v19, v17 // (op 23763) 

v_lshrrev_b32 v14, 16, v10 // (op 23765) 
v_lshrrev_b32 v15, 16, v11 // (op 23765) 
v_lshrrev_b32 v16, 16, v12 // (op 23765) 
v_lshrrev_b32 v17, 16, v13 // (op 23765) 


v_pack_b32_f16 v12, v14, v15 // (op 23731) 
v_pack_b32_f16 v13, v16, v17 // (op 23731) 

s_lshl_b32 s0, s4, 6 // (op 23731) 
s_mul_i32 s1, 5, s0 // (op 23731) 
s_add_u32 s1, s1, 384 // (op 23731) 

v_add_u32 v0, v21, s1 // (op 23731) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23731) Store value



v_accvgpr_read v14, a216 // (op 23769) 
v_accvgpr_read v15, a217 // (op 23769) 
v_accvgpr_read v16, a218 // (op 23769) 
v_accvgpr_read v17, a219 // (op 23769) 
v_mul_f32 v10, v19, v14 // (op 23769) 
v_mul_f32 v11, v19, v15 // (op 23769) 
v_mul_f32 v12, v19, v16 // (op 23769) 
v_mul_f32 v13, v19, v17 // (op 23769) 

v_lshrrev_b32 v14, 16, v10 // (op 23771) 
v_lshrrev_b32 v15, 16, v11 // (op 23771) 
v_lshrrev_b32 v16, 16, v12 // (op 23771) 
v_lshrrev_b32 v17, 16, v13 // (op 23771) 


v_pack_b32_f16 v12, v14, v15 // (op 23733) 
v_pack_b32_f16 v13, v16, v17 // (op 23733) 

s_lshl_b32 s0, s4, 6 // (op 23733) 
s_mul_i32 s1, 6, s0 // (op 23733) 
s_add_u32 s1, s1, 384 // (op 23733) 

v_add_u32 v0, v21, s1 // (op 23733) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23733) Store value



v_accvgpr_read v14, a220 // (op 23775) 
v_accvgpr_read v15, a221 // (op 23775) 
v_accvgpr_read v16, a222 // (op 23775) 
v_accvgpr_read v17, a223 // (op 23775) 
v_mul_f32 v10, v19, v14 // (op 23775) 
v_mul_f32 v11, v19, v15 // (op 23775) 
v_mul_f32 v12, v19, v16 // (op 23775) 
v_mul_f32 v13, v19, v17 // (op 23775) 

v_lshrrev_b32 v14, 16, v10 // (op 23777) 
v_lshrrev_b32 v15, 16, v11 // (op 23777) 
v_lshrrev_b32 v16, 16, v12 // (op 23777) 
v_lshrrev_b32 v17, 16, v13 // (op 23777) 


v_pack_b32_f16 v12, v14, v15 // (op 23735) 
v_pack_b32_f16 v13, v16, v17 // (op 23735) 

s_lshl_b32 s0, s4, 6 // (op 23735) 
s_mul_i32 s1, 7, s0 // (op 23735) 
s_add_u32 s1, s1, 384 // (op 23735) 

v_add_u32 v0, v21, s1 // (op 23735) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23735) Store value



v_accvgpr_read v14, a224 // (op 23865) 
v_accvgpr_read v15, a225 // (op 23865) 
v_accvgpr_read v16, a226 // (op 23865) 
v_accvgpr_read v17, a227 // (op 23865) 
v_mul_f32 v10, v19, v14 // (op 23865) 
v_mul_f32 v11, v19, v15 // (op 23865) 
v_mul_f32 v12, v19, v16 // (op 23865) 
v_mul_f32 v13, v19, v17 // (op 23865) 

v_lshrrev_b32 v14, 16, v10 // (op 23867) 
v_lshrrev_b32 v15, 16, v11 // (op 23867) 
v_lshrrev_b32 v16, 16, v12 // (op 23867) 
v_lshrrev_b32 v17, 16, v13 // (op 23867) 


v_pack_b32_f16 v12, v14, v15 // (op 23869) 
v_pack_b32_f16 v13, v16, v17 // (op 23869) 

v_add_u32 v0, 448, v21 // (op 23869) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23869) Store value



v_accvgpr_read v14, a228 // (op 23887) 
v_accvgpr_read v15, a229 // (op 23887) 
v_accvgpr_read v16, a230 // (op 23887) 
v_accvgpr_read v17, a231 // (op 23887) 
v_mul_f32 v10, v19, v14 // (op 23887) 
v_mul_f32 v11, v19, v15 // (op 23887) 
v_mul_f32 v12, v19, v16 // (op 23887) 
v_mul_f32 v13, v19, v17 // (op 23887) 

v_lshrrev_b32 v14, 16, v10 // (op 23889) 
v_lshrrev_b32 v15, 16, v11 // (op 23889) 
v_lshrrev_b32 v16, 16, v12 // (op 23889) 
v_lshrrev_b32 v17, 16, v13 // (op 23889) 


v_pack_b32_f16 v12, v14, v15 // (op 23871) 
v_pack_b32_f16 v13, v16, v17 // (op 23871) 

s_lshl_b32 s0, s4, 6 // (op 23871) 
s_add_u32 s0, s0, 448 // (op 23871) 
v_add_u32 v0, v21, s0 // (op 23871) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23871) Store value



v_accvgpr_read v14, a232 // (op 23893) 
v_accvgpr_read v15, a233 // (op 23893) 
v_accvgpr_read v16, a234 // (op 23893) 
v_accvgpr_read v17, a235 // (op 23893) 
v_mul_f32 v10, v19, v14 // (op 23893) 
v_mul_f32 v11, v19, v15 // (op 23893) 
v_mul_f32 v12, v19, v16 // (op 23893) 
v_mul_f32 v13, v19, v17 // (op 23893) 

v_lshrrev_b32 v14, 16, v10 // (op 23895) 
v_lshrrev_b32 v15, 16, v11 // (op 23895) 
v_lshrrev_b32 v16, 16, v12 // (op 23895) 
v_lshrrev_b32 v17, 16, v13 // (op 23895) 


v_pack_b32_f16 v12, v14, v15 // (op 23873) 
v_pack_b32_f16 v13, v16, v17 // (op 23873) 

s_lshl_b32 s0, s4, 7 // (op 23873) 
s_add_u32 s0, s0, 448 // (op 23873) 
v_add_u32 v0, v21, s0 // (op 23873) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23873) Store value



v_accvgpr_read v14, a236 // (op 23899) 
v_accvgpr_read v15, a237 // (op 23899) 
v_accvgpr_read v16, a238 // (op 23899) 
v_accvgpr_read v17, a239 // (op 23899) 
v_mul_f32 v10, v19, v14 // (op 23899) 
v_mul_f32 v11, v19, v15 // (op 23899) 
v_mul_f32 v12, v19, v16 // (op 23899) 
v_mul_f32 v13, v19, v17 // (op 23899) 

v_lshrrev_b32 v14, 16, v10 // (op 23901) 
v_lshrrev_b32 v15, 16, v11 // (op 23901) 
v_lshrrev_b32 v16, 16, v12 // (op 23901) 
v_lshrrev_b32 v17, 16, v13 // (op 23901) 


v_pack_b32_f16 v12, v14, v15 // (op 23875) 
v_pack_b32_f16 v13, v16, v17 // (op 23875) 

s_lshl_b32 s0, s4, 6 // (op 23875) 
s_mul_i32 s1, 3, s0 // (op 23875) 
s_add_u32 s1, s1, 448 // (op 23875) 

v_add_u32 v0, v21, s1 // (op 23875) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23875) Store value



v_accvgpr_read v14, a240 // (op 23905) 
v_accvgpr_read v15, a241 // (op 23905) 
v_accvgpr_read v16, a242 // (op 23905) 
v_accvgpr_read v17, a243 // (op 23905) 
v_mul_f32 v10, v19, v14 // (op 23905) 
v_mul_f32 v11, v19, v15 // (op 23905) 
v_mul_f32 v12, v19, v16 // (op 23905) 
v_mul_f32 v13, v19, v17 // (op 23905) 

v_lshrrev_b32 v14, 16, v10 // (op 23907) 
v_lshrrev_b32 v15, 16, v11 // (op 23907) 
v_lshrrev_b32 v16, 16, v12 // (op 23907) 
v_lshrrev_b32 v17, 16, v13 // (op 23907) 


v_pack_b32_f16 v12, v14, v15 // (op 23877) 
v_pack_b32_f16 v13, v16, v17 // (op 23877) 

s_lshl_b32 s0, s4, 8 // (op 23877) 
s_add_u32 s0, s0, 448 // (op 23877) 
v_add_u32 v0, v21, s0 // (op 23877) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23877) Store value



v_accvgpr_read v14, a244 // (op 23911) 
v_accvgpr_read v15, a245 // (op 23911) 
v_accvgpr_read v16, a246 // (op 23911) 
v_accvgpr_read v17, a247 // (op 23911) 
v_mul_f32 v10, v19, v14 // (op 23911) 
v_mul_f32 v11, v19, v15 // (op 23911) 
v_mul_f32 v12, v19, v16 // (op 23911) 
v_mul_f32 v13, v19, v17 // (op 23911) 

v_lshrrev_b32 v14, 16, v10 // (op 23913) 
v_lshrrev_b32 v15, 16, v11 // (op 23913) 
v_lshrrev_b32 v16, 16, v12 // (op 23913) 
v_lshrrev_b32 v17, 16, v13 // (op 23913) 


v_pack_b32_f16 v12, v14, v15 // (op 23879) 
v_pack_b32_f16 v13, v16, v17 // (op 23879) 

s_lshl_b32 s0, s4, 6 // (op 23879) 
s_mul_i32 s1, 5, s0 // (op 23879) 
s_add_u32 s1, s1, 448 // (op 23879) 

v_add_u32 v0, v21, s1 // (op 23879) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23879) Store value



v_accvgpr_read v14, a248 // (op 23917) 
v_accvgpr_read v15, a249 // (op 23917) 
v_accvgpr_read v16, a250 // (op 23917) 
v_accvgpr_read v17, a251 // (op 23917) 
v_mul_f32 v10, v19, v14 // (op 23917) 
v_mul_f32 v11, v19, v15 // (op 23917) 
v_mul_f32 v12, v19, v16 // (op 23917) 
v_mul_f32 v13, v19, v17 // (op 23917) 

v_lshrrev_b32 v14, 16, v10 // (op 23919) 
v_lshrrev_b32 v15, 16, v11 // (op 23919) 
v_lshrrev_b32 v16, 16, v12 // (op 23919) 
v_lshrrev_b32 v17, 16, v13 // (op 23919) 


v_pack_b32_f16 v12, v14, v15 // (op 23881) 
v_pack_b32_f16 v13, v16, v17 // (op 23881) 

s_lshl_b32 s0, s4, 6 // (op 23881) 
s_mul_i32 s1, 6, s0 // (op 23881) 
s_add_u32 s1, s1, 448 // (op 23881) 

v_add_u32 v0, v21, s1 // (op 23881) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23881) Store value



v_accvgpr_read v14, a252 // (op 23923) 
v_accvgpr_read v15, a253 // (op 23923) 
v_accvgpr_read v16, a254 // (op 23923) 
v_accvgpr_read v17, a255 // (op 23923) 
v_mul_f32 v10, v19, v14 // (op 23923) 
v_mul_f32 v11, v19, v15 // (op 23923) 
v_mul_f32 v12, v19, v16 // (op 23923) 
v_mul_f32 v13, v19, v17 // (op 23923) 

v_lshrrev_b32 v14, 16, v10 // (op 23925) 
v_lshrrev_b32 v15, 16, v11 // (op 23925) 
v_lshrrev_b32 v16, 16, v12 // (op 23925) 
v_lshrrev_b32 v17, 16, v13 // (op 23925) 


v_pack_b32_f16 v12, v14, v15 // (op 23883) 
v_pack_b32_f16 v13, v16, v17 // (op 23883) 

s_lshl_b32 s0, s4, 6 // (op 23883) 
s_mul_i32 s1, 7, s0 // (op 23883) 
s_add_u32 s1, s1, 448 // (op 23883) 

v_add_u32 v0, v21, s1 // (op 23883) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 23883) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalBottom_0_beta_135
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalFalse_0_beta_135:


v_add_u32 v0, 448, v20 // (op 25443) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25443) Load value

v_accvgpr_read v12, a224 // (op 25422) 
v_accvgpr_read v13, a225 // (op 25422) 
v_accvgpr_read v14, a226 // (op 25422) 
v_accvgpr_read v15, a227 // (op 25422) 
v_mul_f32 v8, v19, v12 // (op 25422) 
v_mul_f32 v9, v19, v13 // (op 25422) 
v_mul_f32 v10, v19, v14 // (op 25422) 
v_mul_f32 v11, v19, v15 // (op 25422) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25442) 
v_lshrrev_b32 v7, 16, v16 // (op 25442) 
v_lshlrev_b32 v6, 16, v6 // (op 25442) 
v_lshlrev_b32 v7, 16, v7 // (op 25442) 
v_mul_f32 v12, v18, v6 // (op 25442) 
v_mul_f32 v13, v18, v7 // (op 25442) 
v_and_b32 v6, 65535, v17 // (op 25442) 
v_lshrrev_b32 v7, 16, v17 // (op 25442) 
v_lshlrev_b32 v6, 16, v6 // (op 25442) 
v_lshlrev_b32 v7, 16, v7 // (op 25442) 
v_mul_f32 v14, v18, v6 // (op 25442) 
v_mul_f32 v15, v18, v7 // (op 25442) 


v_add_f32 v4, v12, v8 // (op 25423) 
v_add_f32 v5, v13, v9 // (op 25423) 
v_add_f32 v6, v14, v10 // (op 25423) 
v_add_f32 v7, v15, v11 // (op 25423) 


v_lshrrev_b32 v8, 16, v4 // (op 25424) 
v_lshrrev_b32 v9, 16, v5 // (op 25424) 
v_lshrrev_b32 v10, 16, v6 // (op 25424) 
v_lshrrev_b32 v11, 16, v7 // (op 25424) 


v_pack_b32_f16 v6, v8, v9 // (op 25426) 
v_pack_b32_f16 v7, v10, v11 // (op 25426) 

v_add_u32 v0, 448, v21 // (op 25426) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25426) Store value





s_lshl_b32 s0, s2, 6 // (op 25448) 
s_add_u32 s0, s0, 448 // (op 25448) 
v_add_u32 v0, v20, s0 // (op 25448) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25448) Load value

v_accvgpr_read v12, a228 // (op 25480) 
v_accvgpr_read v13, a229 // (op 25480) 
v_accvgpr_read v14, a230 // (op 25480) 
v_accvgpr_read v15, a231 // (op 25480) 
v_mul_f32 v8, v19, v12 // (op 25480) 
v_mul_f32 v9, v19, v13 // (op 25480) 
v_mul_f32 v10, v19, v14 // (op 25480) 
v_mul_f32 v11, v19, v15 // (op 25480) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25445) 
v_lshrrev_b32 v7, 16, v16 // (op 25445) 
v_lshlrev_b32 v6, 16, v6 // (op 25445) 
v_lshlrev_b32 v7, 16, v7 // (op 25445) 
v_mul_f32 v12, v18, v6 // (op 25445) 
v_mul_f32 v13, v18, v7 // (op 25445) 
v_and_b32 v6, 65535, v17 // (op 25445) 
v_lshrrev_b32 v7, 16, v17 // (op 25445) 
v_lshlrev_b32 v6, 16, v6 // (op 25445) 
v_lshlrev_b32 v7, 16, v7 // (op 25445) 
v_mul_f32 v14, v18, v6 // (op 25445) 
v_mul_f32 v15, v18, v7 // (op 25445) 


v_add_f32 v4, v12, v8 // (op 25446) 
v_add_f32 v5, v13, v9 // (op 25446) 
v_add_f32 v6, v14, v10 // (op 25446) 
v_add_f32 v7, v15, v11 // (op 25446) 


v_lshrrev_b32 v8, 16, v4 // (op 25447) 
v_lshrrev_b32 v9, 16, v5 // (op 25447) 
v_lshrrev_b32 v10, 16, v6 // (op 25447) 
v_lshrrev_b32 v11, 16, v7 // (op 25447) 


v_pack_b32_f16 v6, v8, v9 // (op 25428) 
v_pack_b32_f16 v7, v10, v11 // (op 25428) 

s_lshl_b32 s0, s4, 6 // (op 25428) 
s_add_u32 s0, s0, 448 // (op 25428) 
v_add_u32 v0, v21, s0 // (op 25428) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25428) Store value





s_lshl_b32 s0, s2, 7 // (op 25453) 
s_add_u32 s0, s0, 448 // (op 25453) 
v_add_u32 v0, v20, s0 // (op 25453) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25453) Load value

v_accvgpr_read v12, a232 // (op 25482) 
v_accvgpr_read v13, a233 // (op 25482) 
v_accvgpr_read v14, a234 // (op 25482) 
v_accvgpr_read v15, a235 // (op 25482) 
v_mul_f32 v8, v19, v12 // (op 25482) 
v_mul_f32 v9, v19, v13 // (op 25482) 
v_mul_f32 v10, v19, v14 // (op 25482) 
v_mul_f32 v11, v19, v15 // (op 25482) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25450) 
v_lshrrev_b32 v7, 16, v16 // (op 25450) 
v_lshlrev_b32 v6, 16, v6 // (op 25450) 
v_lshlrev_b32 v7, 16, v7 // (op 25450) 
v_mul_f32 v12, v18, v6 // (op 25450) 
v_mul_f32 v13, v18, v7 // (op 25450) 
v_and_b32 v6, 65535, v17 // (op 25450) 
v_lshrrev_b32 v7, 16, v17 // (op 25450) 
v_lshlrev_b32 v6, 16, v6 // (op 25450) 
v_lshlrev_b32 v7, 16, v7 // (op 25450) 
v_mul_f32 v14, v18, v6 // (op 25450) 
v_mul_f32 v15, v18, v7 // (op 25450) 


v_add_f32 v4, v12, v8 // (op 25451) 
v_add_f32 v5, v13, v9 // (op 25451) 
v_add_f32 v6, v14, v10 // (op 25451) 
v_add_f32 v7, v15, v11 // (op 25451) 


v_lshrrev_b32 v8, 16, v4 // (op 25452) 
v_lshrrev_b32 v9, 16, v5 // (op 25452) 
v_lshrrev_b32 v10, 16, v6 // (op 25452) 
v_lshrrev_b32 v11, 16, v7 // (op 25452) 


v_pack_b32_f16 v6, v8, v9 // (op 25430) 
v_pack_b32_f16 v7, v10, v11 // (op 25430) 

s_lshl_b32 s0, s4, 7 // (op 25430) 
s_add_u32 s0, s0, 448 // (op 25430) 
v_add_u32 v0, v21, s0 // (op 25430) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25430) Store value





s_lshl_b32 s0, s2, 6 // (op 25458) 
s_mul_i32 s1, 3, s0 // (op 25458) 
s_add_u32 s1, s1, 448 // (op 25458) 

v_add_u32 v0, v20, s1 // (op 25458) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25458) Load value

v_accvgpr_read v12, a236 // (op 25484) 
v_accvgpr_read v13, a237 // (op 25484) 
v_accvgpr_read v14, a238 // (op 25484) 
v_accvgpr_read v15, a239 // (op 25484) 
v_mul_f32 v8, v19, v12 // (op 25484) 
v_mul_f32 v9, v19, v13 // (op 25484) 
v_mul_f32 v10, v19, v14 // (op 25484) 
v_mul_f32 v11, v19, v15 // (op 25484) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25455) 
v_lshrrev_b32 v7, 16, v16 // (op 25455) 
v_lshlrev_b32 v6, 16, v6 // (op 25455) 
v_lshlrev_b32 v7, 16, v7 // (op 25455) 
v_mul_f32 v12, v18, v6 // (op 25455) 
v_mul_f32 v13, v18, v7 // (op 25455) 
v_and_b32 v6, 65535, v17 // (op 25455) 
v_lshrrev_b32 v7, 16, v17 // (op 25455) 
v_lshlrev_b32 v6, 16, v6 // (op 25455) 
v_lshlrev_b32 v7, 16, v7 // (op 25455) 
v_mul_f32 v14, v18, v6 // (op 25455) 
v_mul_f32 v15, v18, v7 // (op 25455) 


v_add_f32 v4, v12, v8 // (op 25456) 
v_add_f32 v5, v13, v9 // (op 25456) 
v_add_f32 v6, v14, v10 // (op 25456) 
v_add_f32 v7, v15, v11 // (op 25456) 


v_lshrrev_b32 v8, 16, v4 // (op 25457) 
v_lshrrev_b32 v9, 16, v5 // (op 25457) 
v_lshrrev_b32 v10, 16, v6 // (op 25457) 
v_lshrrev_b32 v11, 16, v7 // (op 25457) 


v_pack_b32_f16 v6, v8, v9 // (op 25432) 
v_pack_b32_f16 v7, v10, v11 // (op 25432) 

s_lshl_b32 s0, s4, 6 // (op 25432) 
s_mul_i32 s1, 3, s0 // (op 25432) 
s_add_u32 s1, s1, 448 // (op 25432) 

v_add_u32 v0, v21, s1 // (op 25432) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25432) Store value





s_lshl_b32 s0, s2, 8 // (op 25463) 
s_add_u32 s0, s0, 448 // (op 25463) 
v_add_u32 v0, v20, s0 // (op 25463) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25463) Load value

v_accvgpr_read v12, a240 // (op 25486) 
v_accvgpr_read v13, a241 // (op 25486) 
v_accvgpr_read v14, a242 // (op 25486) 
v_accvgpr_read v15, a243 // (op 25486) 
v_mul_f32 v8, v19, v12 // (op 25486) 
v_mul_f32 v9, v19, v13 // (op 25486) 
v_mul_f32 v10, v19, v14 // (op 25486) 
v_mul_f32 v11, v19, v15 // (op 25486) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25460) 
v_lshrrev_b32 v7, 16, v16 // (op 25460) 
v_lshlrev_b32 v6, 16, v6 // (op 25460) 
v_lshlrev_b32 v7, 16, v7 // (op 25460) 
v_mul_f32 v12, v18, v6 // (op 25460) 
v_mul_f32 v13, v18, v7 // (op 25460) 
v_and_b32 v6, 65535, v17 // (op 25460) 
v_lshrrev_b32 v7, 16, v17 // (op 25460) 
v_lshlrev_b32 v6, 16, v6 // (op 25460) 
v_lshlrev_b32 v7, 16, v7 // (op 25460) 
v_mul_f32 v14, v18, v6 // (op 25460) 
v_mul_f32 v15, v18, v7 // (op 25460) 


v_add_f32 v4, v12, v8 // (op 25461) 
v_add_f32 v5, v13, v9 // (op 25461) 
v_add_f32 v6, v14, v10 // (op 25461) 
v_add_f32 v7, v15, v11 // (op 25461) 


v_lshrrev_b32 v8, 16, v4 // (op 25462) 
v_lshrrev_b32 v9, 16, v5 // (op 25462) 
v_lshrrev_b32 v10, 16, v6 // (op 25462) 
v_lshrrev_b32 v11, 16, v7 // (op 25462) 


v_pack_b32_f16 v6, v8, v9 // (op 25434) 
v_pack_b32_f16 v7, v10, v11 // (op 25434) 

s_lshl_b32 s0, s4, 8 // (op 25434) 
s_add_u32 s0, s0, 448 // (op 25434) 
v_add_u32 v0, v21, s0 // (op 25434) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25434) Store value





s_lshl_b32 s0, s2, 6 // (op 25468) 
s_mul_i32 s1, 5, s0 // (op 25468) 
s_add_u32 s1, s1, 448 // (op 25468) 

v_add_u32 v0, v20, s1 // (op 25468) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25468) Load value

v_accvgpr_read v12, a244 // (op 25488) 
v_accvgpr_read v13, a245 // (op 25488) 
v_accvgpr_read v14, a246 // (op 25488) 
v_accvgpr_read v15, a247 // (op 25488) 
v_mul_f32 v8, v19, v12 // (op 25488) 
v_mul_f32 v9, v19, v13 // (op 25488) 
v_mul_f32 v10, v19, v14 // (op 25488) 
v_mul_f32 v11, v19, v15 // (op 25488) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25465) 
v_lshrrev_b32 v7, 16, v16 // (op 25465) 
v_lshlrev_b32 v6, 16, v6 // (op 25465) 
v_lshlrev_b32 v7, 16, v7 // (op 25465) 
v_mul_f32 v12, v18, v6 // (op 25465) 
v_mul_f32 v13, v18, v7 // (op 25465) 
v_and_b32 v6, 65535, v17 // (op 25465) 
v_lshrrev_b32 v7, 16, v17 // (op 25465) 
v_lshlrev_b32 v6, 16, v6 // (op 25465) 
v_lshlrev_b32 v7, 16, v7 // (op 25465) 
v_mul_f32 v14, v18, v6 // (op 25465) 
v_mul_f32 v15, v18, v7 // (op 25465) 


v_add_f32 v4, v12, v8 // (op 25466) 
v_add_f32 v5, v13, v9 // (op 25466) 
v_add_f32 v6, v14, v10 // (op 25466) 
v_add_f32 v7, v15, v11 // (op 25466) 


v_lshrrev_b32 v8, 16, v4 // (op 25467) 
v_lshrrev_b32 v9, 16, v5 // (op 25467) 
v_lshrrev_b32 v10, 16, v6 // (op 25467) 
v_lshrrev_b32 v11, 16, v7 // (op 25467) 


v_pack_b32_f16 v6, v8, v9 // (op 25436) 
v_pack_b32_f16 v7, v10, v11 // (op 25436) 

s_lshl_b32 s0, s4, 6 // (op 25436) 
s_mul_i32 s1, 5, s0 // (op 25436) 
s_add_u32 s1, s1, 448 // (op 25436) 

v_add_u32 v0, v21, s1 // (op 25436) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25436) Store value





s_lshl_b32 s0, s2, 6 // (op 25473) 
s_mul_i32 s1, 6, s0 // (op 25473) 
s_add_u32 s1, s1, 448 // (op 25473) 

v_add_u32 v0, v20, s1 // (op 25473) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25473) Load value

v_accvgpr_read v12, a248 // (op 25490) 
v_accvgpr_read v13, a249 // (op 25490) 
v_accvgpr_read v14, a250 // (op 25490) 
v_accvgpr_read v15, a251 // (op 25490) 
v_mul_f32 v8, v19, v12 // (op 25490) 
v_mul_f32 v9, v19, v13 // (op 25490) 
v_mul_f32 v10, v19, v14 // (op 25490) 
v_mul_f32 v11, v19, v15 // (op 25490) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25470) 
v_lshrrev_b32 v7, 16, v16 // (op 25470) 
v_lshlrev_b32 v6, 16, v6 // (op 25470) 
v_lshlrev_b32 v7, 16, v7 // (op 25470) 
v_mul_f32 v12, v18, v6 // (op 25470) 
v_mul_f32 v13, v18, v7 // (op 25470) 
v_and_b32 v6, 65535, v17 // (op 25470) 
v_lshrrev_b32 v7, 16, v17 // (op 25470) 
v_lshlrev_b32 v6, 16, v6 // (op 25470) 
v_lshlrev_b32 v7, 16, v7 // (op 25470) 
v_mul_f32 v14, v18, v6 // (op 25470) 
v_mul_f32 v15, v18, v7 // (op 25470) 


v_add_f32 v4, v12, v8 // (op 25471) 
v_add_f32 v5, v13, v9 // (op 25471) 
v_add_f32 v6, v14, v10 // (op 25471) 
v_add_f32 v7, v15, v11 // (op 25471) 


v_lshrrev_b32 v8, 16, v4 // (op 25472) 
v_lshrrev_b32 v9, 16, v5 // (op 25472) 
v_lshrrev_b32 v10, 16, v6 // (op 25472) 
v_lshrrev_b32 v11, 16, v7 // (op 25472) 


v_pack_b32_f16 v6, v8, v9 // (op 25438) 
v_pack_b32_f16 v7, v10, v11 // (op 25438) 

s_lshl_b32 s0, s4, 6 // (op 25438) 
s_mul_i32 s1, 6, s0 // (op 25438) 
s_add_u32 s1, s1, 448 // (op 25438) 

v_add_u32 v0, v21, s1 // (op 25438) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25438) Store value





s_lshl_b32 s0, s2, 6 // (op 25478) 
s_mul_i32 s1, 7, s0 // (op 25478) 
s_add_u32 s1, s1, 448 // (op 25478) 

v_add_u32 v0, v20, s1 // (op 25478) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25478) Load value

v_accvgpr_read v12, a252 // (op 25492) 
v_accvgpr_read v13, a253 // (op 25492) 
v_accvgpr_read v14, a254 // (op 25492) 
v_accvgpr_read v15, a255 // (op 25492) 
v_mul_f32 v8, v19, v12 // (op 25492) 
v_mul_f32 v9, v19, v13 // (op 25492) 
v_mul_f32 v10, v19, v14 // (op 25492) 
v_mul_f32 v11, v19, v15 // (op 25492) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25475) 
v_lshrrev_b32 v7, 16, v16 // (op 25475) 
v_lshlrev_b32 v6, 16, v6 // (op 25475) 
v_lshlrev_b32 v7, 16, v7 // (op 25475) 
v_mul_f32 v12, v18, v6 // (op 25475) 
v_mul_f32 v13, v18, v7 // (op 25475) 
v_and_b32 v6, 65535, v17 // (op 25475) 
v_lshrrev_b32 v7, 16, v17 // (op 25475) 
v_lshlrev_b32 v6, 16, v6 // (op 25475) 
v_lshlrev_b32 v7, 16, v7 // (op 25475) 
v_mul_f32 v14, v18, v6 // (op 25475) 
v_mul_f32 v15, v18, v7 // (op 25475) 


v_add_f32 v4, v12, v8 // (op 25476) 
v_add_f32 v5, v13, v9 // (op 25476) 
v_add_f32 v6, v14, v10 // (op 25476) 
v_add_f32 v7, v15, v11 // (op 25476) 


v_lshrrev_b32 v8, 16, v4 // (op 25477) 
v_lshrrev_b32 v9, 16, v5 // (op 25477) 
v_lshrrev_b32 v10, 16, v6 // (op 25477) 
v_lshrrev_b32 v11, 16, v7 // (op 25477) 


v_pack_b32_f16 v6, v8, v9 // (op 25440) 
v_pack_b32_f16 v7, v10, v11 // (op 25440) 

s_lshl_b32 s0, s4, 6 // (op 25440) 
s_mul_i32 s1, 7, s0 // (op 25440) 
s_add_u32 s1, s1, 448 // (op 25440) 

v_add_u32 v0, v21, s1 // (op 25440) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25440) Store value





v_add_u32 v0, 384, v20 // (op 25272) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25272) Load value

v_accvgpr_read v12, a192 // (op 25251) 
v_accvgpr_read v13, a193 // (op 25251) 
v_accvgpr_read v14, a194 // (op 25251) 
v_accvgpr_read v15, a195 // (op 25251) 
v_mul_f32 v8, v19, v12 // (op 25251) 
v_mul_f32 v9, v19, v13 // (op 25251) 
v_mul_f32 v10, v19, v14 // (op 25251) 
v_mul_f32 v11, v19, v15 // (op 25251) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25271) 
v_lshrrev_b32 v7, 16, v16 // (op 25271) 
v_lshlrev_b32 v6, 16, v6 // (op 25271) 
v_lshlrev_b32 v7, 16, v7 // (op 25271) 
v_mul_f32 v12, v18, v6 // (op 25271) 
v_mul_f32 v13, v18, v7 // (op 25271) 
v_and_b32 v6, 65535, v17 // (op 25271) 
v_lshrrev_b32 v7, 16, v17 // (op 25271) 
v_lshlrev_b32 v6, 16, v6 // (op 25271) 
v_lshlrev_b32 v7, 16, v7 // (op 25271) 
v_mul_f32 v14, v18, v6 // (op 25271) 
v_mul_f32 v15, v18, v7 // (op 25271) 


v_add_f32 v4, v12, v8 // (op 25252) 
v_add_f32 v5, v13, v9 // (op 25252) 
v_add_f32 v6, v14, v10 // (op 25252) 
v_add_f32 v7, v15, v11 // (op 25252) 


v_lshrrev_b32 v8, 16, v4 // (op 25253) 
v_lshrrev_b32 v9, 16, v5 // (op 25253) 
v_lshrrev_b32 v10, 16, v6 // (op 25253) 
v_lshrrev_b32 v11, 16, v7 // (op 25253) 


v_pack_b32_f16 v6, v8, v9 // (op 25255) 
v_pack_b32_f16 v7, v10, v11 // (op 25255) 

v_add_u32 v0, 384, v21 // (op 25255) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25255) Store value





s_lshl_b32 s0, s2, 6 // (op 25277) 
s_add_u32 s0, s0, 384 // (op 25277) 
v_add_u32 v0, v20, s0 // (op 25277) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25277) Load value

v_accvgpr_read v12, a196 // (op 25309) 
v_accvgpr_read v13, a197 // (op 25309) 
v_accvgpr_read v14, a198 // (op 25309) 
v_accvgpr_read v15, a199 // (op 25309) 
v_mul_f32 v8, v19, v12 // (op 25309) 
v_mul_f32 v9, v19, v13 // (op 25309) 
v_mul_f32 v10, v19, v14 // (op 25309) 
v_mul_f32 v11, v19, v15 // (op 25309) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25274) 
v_lshrrev_b32 v7, 16, v16 // (op 25274) 
v_lshlrev_b32 v6, 16, v6 // (op 25274) 
v_lshlrev_b32 v7, 16, v7 // (op 25274) 
v_mul_f32 v12, v18, v6 // (op 25274) 
v_mul_f32 v13, v18, v7 // (op 25274) 
v_and_b32 v6, 65535, v17 // (op 25274) 
v_lshrrev_b32 v7, 16, v17 // (op 25274) 
v_lshlrev_b32 v6, 16, v6 // (op 25274) 
v_lshlrev_b32 v7, 16, v7 // (op 25274) 
v_mul_f32 v14, v18, v6 // (op 25274) 
v_mul_f32 v15, v18, v7 // (op 25274) 


v_add_f32 v4, v12, v8 // (op 25275) 
v_add_f32 v5, v13, v9 // (op 25275) 
v_add_f32 v6, v14, v10 // (op 25275) 
v_add_f32 v7, v15, v11 // (op 25275) 


v_lshrrev_b32 v8, 16, v4 // (op 25276) 
v_lshrrev_b32 v9, 16, v5 // (op 25276) 
v_lshrrev_b32 v10, 16, v6 // (op 25276) 
v_lshrrev_b32 v11, 16, v7 // (op 25276) 


v_pack_b32_f16 v6, v8, v9 // (op 25257) 
v_pack_b32_f16 v7, v10, v11 // (op 25257) 

s_lshl_b32 s0, s4, 6 // (op 25257) 
s_add_u32 s0, s0, 384 // (op 25257) 
v_add_u32 v0, v21, s0 // (op 25257) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25257) Store value





s_lshl_b32 s0, s2, 7 // (op 25282) 
s_add_u32 s0, s0, 384 // (op 25282) 
v_add_u32 v0, v20, s0 // (op 25282) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25282) Load value

v_accvgpr_read v12, a200 // (op 25311) 
v_accvgpr_read v13, a201 // (op 25311) 
v_accvgpr_read v14, a202 // (op 25311) 
v_accvgpr_read v15, a203 // (op 25311) 
v_mul_f32 v8, v19, v12 // (op 25311) 
v_mul_f32 v9, v19, v13 // (op 25311) 
v_mul_f32 v10, v19, v14 // (op 25311) 
v_mul_f32 v11, v19, v15 // (op 25311) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25279) 
v_lshrrev_b32 v7, 16, v16 // (op 25279) 
v_lshlrev_b32 v6, 16, v6 // (op 25279) 
v_lshlrev_b32 v7, 16, v7 // (op 25279) 
v_mul_f32 v12, v18, v6 // (op 25279) 
v_mul_f32 v13, v18, v7 // (op 25279) 
v_and_b32 v6, 65535, v17 // (op 25279) 
v_lshrrev_b32 v7, 16, v17 // (op 25279) 
v_lshlrev_b32 v6, 16, v6 // (op 25279) 
v_lshlrev_b32 v7, 16, v7 // (op 25279) 
v_mul_f32 v14, v18, v6 // (op 25279) 
v_mul_f32 v15, v18, v7 // (op 25279) 


v_add_f32 v4, v12, v8 // (op 25280) 
v_add_f32 v5, v13, v9 // (op 25280) 
v_add_f32 v6, v14, v10 // (op 25280) 
v_add_f32 v7, v15, v11 // (op 25280) 


v_lshrrev_b32 v8, 16, v4 // (op 25281) 
v_lshrrev_b32 v9, 16, v5 // (op 25281) 
v_lshrrev_b32 v10, 16, v6 // (op 25281) 
v_lshrrev_b32 v11, 16, v7 // (op 25281) 


v_pack_b32_f16 v6, v8, v9 // (op 25259) 
v_pack_b32_f16 v7, v10, v11 // (op 25259) 

s_lshl_b32 s0, s4, 7 // (op 25259) 
s_add_u32 s0, s0, 384 // (op 25259) 
v_add_u32 v0, v21, s0 // (op 25259) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25259) Store value





s_lshl_b32 s0, s2, 6 // (op 25287) 
s_mul_i32 s1, 3, s0 // (op 25287) 
s_add_u32 s1, s1, 384 // (op 25287) 

v_add_u32 v0, v20, s1 // (op 25287) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25287) Load value

v_accvgpr_read v12, a204 // (op 25313) 
v_accvgpr_read v13, a205 // (op 25313) 
v_accvgpr_read v14, a206 // (op 25313) 
v_accvgpr_read v15, a207 // (op 25313) 
v_mul_f32 v8, v19, v12 // (op 25313) 
v_mul_f32 v9, v19, v13 // (op 25313) 
v_mul_f32 v10, v19, v14 // (op 25313) 
v_mul_f32 v11, v19, v15 // (op 25313) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25284) 
v_lshrrev_b32 v7, 16, v16 // (op 25284) 
v_lshlrev_b32 v6, 16, v6 // (op 25284) 
v_lshlrev_b32 v7, 16, v7 // (op 25284) 
v_mul_f32 v12, v18, v6 // (op 25284) 
v_mul_f32 v13, v18, v7 // (op 25284) 
v_and_b32 v6, 65535, v17 // (op 25284) 
v_lshrrev_b32 v7, 16, v17 // (op 25284) 
v_lshlrev_b32 v6, 16, v6 // (op 25284) 
v_lshlrev_b32 v7, 16, v7 // (op 25284) 
v_mul_f32 v14, v18, v6 // (op 25284) 
v_mul_f32 v15, v18, v7 // (op 25284) 


v_add_f32 v4, v12, v8 // (op 25285) 
v_add_f32 v5, v13, v9 // (op 25285) 
v_add_f32 v6, v14, v10 // (op 25285) 
v_add_f32 v7, v15, v11 // (op 25285) 


v_lshrrev_b32 v8, 16, v4 // (op 25286) 
v_lshrrev_b32 v9, 16, v5 // (op 25286) 
v_lshrrev_b32 v10, 16, v6 // (op 25286) 
v_lshrrev_b32 v11, 16, v7 // (op 25286) 


v_pack_b32_f16 v6, v8, v9 // (op 25261) 
v_pack_b32_f16 v7, v10, v11 // (op 25261) 

s_lshl_b32 s0, s4, 6 // (op 25261) 
s_mul_i32 s1, 3, s0 // (op 25261) 
s_add_u32 s1, s1, 384 // (op 25261) 

v_add_u32 v0, v21, s1 // (op 25261) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25261) Store value





s_lshl_b32 s0, s2, 8 // (op 25292) 
s_add_u32 s0, s0, 384 // (op 25292) 
v_add_u32 v0, v20, s0 // (op 25292) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25292) Load value

v_accvgpr_read v12, a208 // (op 25315) 
v_accvgpr_read v13, a209 // (op 25315) 
v_accvgpr_read v14, a210 // (op 25315) 
v_accvgpr_read v15, a211 // (op 25315) 
v_mul_f32 v8, v19, v12 // (op 25315) 
v_mul_f32 v9, v19, v13 // (op 25315) 
v_mul_f32 v10, v19, v14 // (op 25315) 
v_mul_f32 v11, v19, v15 // (op 25315) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25289) 
v_lshrrev_b32 v7, 16, v16 // (op 25289) 
v_lshlrev_b32 v6, 16, v6 // (op 25289) 
v_lshlrev_b32 v7, 16, v7 // (op 25289) 
v_mul_f32 v12, v18, v6 // (op 25289) 
v_mul_f32 v13, v18, v7 // (op 25289) 
v_and_b32 v6, 65535, v17 // (op 25289) 
v_lshrrev_b32 v7, 16, v17 // (op 25289) 
v_lshlrev_b32 v6, 16, v6 // (op 25289) 
v_lshlrev_b32 v7, 16, v7 // (op 25289) 
v_mul_f32 v14, v18, v6 // (op 25289) 
v_mul_f32 v15, v18, v7 // (op 25289) 


v_add_f32 v4, v12, v8 // (op 25290) 
v_add_f32 v5, v13, v9 // (op 25290) 
v_add_f32 v6, v14, v10 // (op 25290) 
v_add_f32 v7, v15, v11 // (op 25290) 


v_lshrrev_b32 v8, 16, v4 // (op 25291) 
v_lshrrev_b32 v9, 16, v5 // (op 25291) 
v_lshrrev_b32 v10, 16, v6 // (op 25291) 
v_lshrrev_b32 v11, 16, v7 // (op 25291) 


v_pack_b32_f16 v6, v8, v9 // (op 25263) 
v_pack_b32_f16 v7, v10, v11 // (op 25263) 

s_lshl_b32 s0, s4, 8 // (op 25263) 
s_add_u32 s0, s0, 384 // (op 25263) 
v_add_u32 v0, v21, s0 // (op 25263) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25263) Store value





s_lshl_b32 s0, s2, 6 // (op 25297) 
s_mul_i32 s1, 5, s0 // (op 25297) 
s_add_u32 s1, s1, 384 // (op 25297) 

v_add_u32 v0, v20, s1 // (op 25297) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25297) Load value

v_accvgpr_read v12, a212 // (op 25317) 
v_accvgpr_read v13, a213 // (op 25317) 
v_accvgpr_read v14, a214 // (op 25317) 
v_accvgpr_read v15, a215 // (op 25317) 
v_mul_f32 v8, v19, v12 // (op 25317) 
v_mul_f32 v9, v19, v13 // (op 25317) 
v_mul_f32 v10, v19, v14 // (op 25317) 
v_mul_f32 v11, v19, v15 // (op 25317) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25294) 
v_lshrrev_b32 v7, 16, v16 // (op 25294) 
v_lshlrev_b32 v6, 16, v6 // (op 25294) 
v_lshlrev_b32 v7, 16, v7 // (op 25294) 
v_mul_f32 v12, v18, v6 // (op 25294) 
v_mul_f32 v13, v18, v7 // (op 25294) 
v_and_b32 v6, 65535, v17 // (op 25294) 
v_lshrrev_b32 v7, 16, v17 // (op 25294) 
v_lshlrev_b32 v6, 16, v6 // (op 25294) 
v_lshlrev_b32 v7, 16, v7 // (op 25294) 
v_mul_f32 v14, v18, v6 // (op 25294) 
v_mul_f32 v15, v18, v7 // (op 25294) 


v_add_f32 v4, v12, v8 // (op 25295) 
v_add_f32 v5, v13, v9 // (op 25295) 
v_add_f32 v6, v14, v10 // (op 25295) 
v_add_f32 v7, v15, v11 // (op 25295) 


v_lshrrev_b32 v8, 16, v4 // (op 25296) 
v_lshrrev_b32 v9, 16, v5 // (op 25296) 
v_lshrrev_b32 v10, 16, v6 // (op 25296) 
v_lshrrev_b32 v11, 16, v7 // (op 25296) 


v_pack_b32_f16 v6, v8, v9 // (op 25265) 
v_pack_b32_f16 v7, v10, v11 // (op 25265) 

s_lshl_b32 s0, s4, 6 // (op 25265) 
s_mul_i32 s1, 5, s0 // (op 25265) 
s_add_u32 s1, s1, 384 // (op 25265) 

v_add_u32 v0, v21, s1 // (op 25265) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25265) Store value





s_lshl_b32 s0, s2, 6 // (op 25302) 
s_mul_i32 s1, 6, s0 // (op 25302) 
s_add_u32 s1, s1, 384 // (op 25302) 

v_add_u32 v0, v20, s1 // (op 25302) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25302) Load value

v_accvgpr_read v12, a216 // (op 25319) 
v_accvgpr_read v13, a217 // (op 25319) 
v_accvgpr_read v14, a218 // (op 25319) 
v_accvgpr_read v15, a219 // (op 25319) 
v_mul_f32 v8, v19, v12 // (op 25319) 
v_mul_f32 v9, v19, v13 // (op 25319) 
v_mul_f32 v10, v19, v14 // (op 25319) 
v_mul_f32 v11, v19, v15 // (op 25319) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25299) 
v_lshrrev_b32 v7, 16, v16 // (op 25299) 
v_lshlrev_b32 v6, 16, v6 // (op 25299) 
v_lshlrev_b32 v7, 16, v7 // (op 25299) 
v_mul_f32 v12, v18, v6 // (op 25299) 
v_mul_f32 v13, v18, v7 // (op 25299) 
v_and_b32 v6, 65535, v17 // (op 25299) 
v_lshrrev_b32 v7, 16, v17 // (op 25299) 
v_lshlrev_b32 v6, 16, v6 // (op 25299) 
v_lshlrev_b32 v7, 16, v7 // (op 25299) 
v_mul_f32 v14, v18, v6 // (op 25299) 
v_mul_f32 v15, v18, v7 // (op 25299) 


v_add_f32 v4, v12, v8 // (op 25300) 
v_add_f32 v5, v13, v9 // (op 25300) 
v_add_f32 v6, v14, v10 // (op 25300) 
v_add_f32 v7, v15, v11 // (op 25300) 


v_lshrrev_b32 v8, 16, v4 // (op 25301) 
v_lshrrev_b32 v9, 16, v5 // (op 25301) 
v_lshrrev_b32 v10, 16, v6 // (op 25301) 
v_lshrrev_b32 v11, 16, v7 // (op 25301) 


v_pack_b32_f16 v6, v8, v9 // (op 25267) 
v_pack_b32_f16 v7, v10, v11 // (op 25267) 

s_lshl_b32 s0, s4, 6 // (op 25267) 
s_mul_i32 s1, 6, s0 // (op 25267) 
s_add_u32 s1, s1, 384 // (op 25267) 

v_add_u32 v0, v21, s1 // (op 25267) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25267) Store value





s_lshl_b32 s0, s2, 6 // (op 25307) 
s_mul_i32 s1, 7, s0 // (op 25307) 
s_add_u32 s1, s1, 384 // (op 25307) 

v_add_u32 v0, v20, s1 // (op 25307) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25307) Load value

v_accvgpr_read v12, a220 // (op 25321) 
v_accvgpr_read v13, a221 // (op 25321) 
v_accvgpr_read v14, a222 // (op 25321) 
v_accvgpr_read v15, a223 // (op 25321) 
v_mul_f32 v8, v19, v12 // (op 25321) 
v_mul_f32 v9, v19, v13 // (op 25321) 
v_mul_f32 v10, v19, v14 // (op 25321) 
v_mul_f32 v11, v19, v15 // (op 25321) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25304) 
v_lshrrev_b32 v7, 16, v16 // (op 25304) 
v_lshlrev_b32 v6, 16, v6 // (op 25304) 
v_lshlrev_b32 v7, 16, v7 // (op 25304) 
v_mul_f32 v12, v18, v6 // (op 25304) 
v_mul_f32 v13, v18, v7 // (op 25304) 
v_and_b32 v6, 65535, v17 // (op 25304) 
v_lshrrev_b32 v7, 16, v17 // (op 25304) 
v_lshlrev_b32 v6, 16, v6 // (op 25304) 
v_lshlrev_b32 v7, 16, v7 // (op 25304) 
v_mul_f32 v14, v18, v6 // (op 25304) 
v_mul_f32 v15, v18, v7 // (op 25304) 


v_add_f32 v4, v12, v8 // (op 25305) 
v_add_f32 v5, v13, v9 // (op 25305) 
v_add_f32 v6, v14, v10 // (op 25305) 
v_add_f32 v7, v15, v11 // (op 25305) 


v_lshrrev_b32 v8, 16, v4 // (op 25306) 
v_lshrrev_b32 v9, 16, v5 // (op 25306) 
v_lshrrev_b32 v10, 16, v6 // (op 25306) 
v_lshrrev_b32 v11, 16, v7 // (op 25306) 


v_pack_b32_f16 v6, v8, v9 // (op 25269) 
v_pack_b32_f16 v7, v10, v11 // (op 25269) 

s_lshl_b32 s0, s4, 6 // (op 25269) 
s_mul_i32 s1, 7, s0 // (op 25269) 
s_add_u32 s1, s1, 384 // (op 25269) 

v_add_u32 v0, v21, s1 // (op 25269) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25269) Store value





v_add_u32 v0, 320, v20 // (op 25101) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25101) Load value

v_accvgpr_read v12, a160 // (op 25080) 
v_accvgpr_read v13, a161 // (op 25080) 
v_accvgpr_read v14, a162 // (op 25080) 
v_accvgpr_read v15, a163 // (op 25080) 
v_mul_f32 v8, v19, v12 // (op 25080) 
v_mul_f32 v9, v19, v13 // (op 25080) 
v_mul_f32 v10, v19, v14 // (op 25080) 
v_mul_f32 v11, v19, v15 // (op 25080) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25100) 
v_lshrrev_b32 v7, 16, v16 // (op 25100) 
v_lshlrev_b32 v6, 16, v6 // (op 25100) 
v_lshlrev_b32 v7, 16, v7 // (op 25100) 
v_mul_f32 v12, v18, v6 // (op 25100) 
v_mul_f32 v13, v18, v7 // (op 25100) 
v_and_b32 v6, 65535, v17 // (op 25100) 
v_lshrrev_b32 v7, 16, v17 // (op 25100) 
v_lshlrev_b32 v6, 16, v6 // (op 25100) 
v_lshlrev_b32 v7, 16, v7 // (op 25100) 
v_mul_f32 v14, v18, v6 // (op 25100) 
v_mul_f32 v15, v18, v7 // (op 25100) 


v_add_f32 v4, v12, v8 // (op 25081) 
v_add_f32 v5, v13, v9 // (op 25081) 
v_add_f32 v6, v14, v10 // (op 25081) 
v_add_f32 v7, v15, v11 // (op 25081) 


v_lshrrev_b32 v8, 16, v4 // (op 25082) 
v_lshrrev_b32 v9, 16, v5 // (op 25082) 
v_lshrrev_b32 v10, 16, v6 // (op 25082) 
v_lshrrev_b32 v11, 16, v7 // (op 25082) 


v_pack_b32_f16 v6, v8, v9 // (op 25084) 
v_pack_b32_f16 v7, v10, v11 // (op 25084) 

v_add_u32 v0, 320, v21 // (op 25084) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25084) Store value





s_lshl_b32 s0, s2, 6 // (op 25106) 
s_add_u32 s0, s0, 320 // (op 25106) 
v_add_u32 v0, v20, s0 // (op 25106) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25106) Load value

v_accvgpr_read v12, a164 // (op 25138) 
v_accvgpr_read v13, a165 // (op 25138) 
v_accvgpr_read v14, a166 // (op 25138) 
v_accvgpr_read v15, a167 // (op 25138) 
v_mul_f32 v8, v19, v12 // (op 25138) 
v_mul_f32 v9, v19, v13 // (op 25138) 
v_mul_f32 v10, v19, v14 // (op 25138) 
v_mul_f32 v11, v19, v15 // (op 25138) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25103) 
v_lshrrev_b32 v7, 16, v16 // (op 25103) 
v_lshlrev_b32 v6, 16, v6 // (op 25103) 
v_lshlrev_b32 v7, 16, v7 // (op 25103) 
v_mul_f32 v12, v18, v6 // (op 25103) 
v_mul_f32 v13, v18, v7 // (op 25103) 
v_and_b32 v6, 65535, v17 // (op 25103) 
v_lshrrev_b32 v7, 16, v17 // (op 25103) 
v_lshlrev_b32 v6, 16, v6 // (op 25103) 
v_lshlrev_b32 v7, 16, v7 // (op 25103) 
v_mul_f32 v14, v18, v6 // (op 25103) 
v_mul_f32 v15, v18, v7 // (op 25103) 


v_add_f32 v4, v12, v8 // (op 25104) 
v_add_f32 v5, v13, v9 // (op 25104) 
v_add_f32 v6, v14, v10 // (op 25104) 
v_add_f32 v7, v15, v11 // (op 25104) 


v_lshrrev_b32 v8, 16, v4 // (op 25105) 
v_lshrrev_b32 v9, 16, v5 // (op 25105) 
v_lshrrev_b32 v10, 16, v6 // (op 25105) 
v_lshrrev_b32 v11, 16, v7 // (op 25105) 


v_pack_b32_f16 v6, v8, v9 // (op 25086) 
v_pack_b32_f16 v7, v10, v11 // (op 25086) 

s_lshl_b32 s0, s4, 6 // (op 25086) 
s_add_u32 s0, s0, 320 // (op 25086) 
v_add_u32 v0, v21, s0 // (op 25086) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25086) Store value





s_lshl_b32 s0, s2, 7 // (op 25111) 
s_add_u32 s0, s0, 320 // (op 25111) 
v_add_u32 v0, v20, s0 // (op 25111) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25111) Load value

v_accvgpr_read v12, a168 // (op 25140) 
v_accvgpr_read v13, a169 // (op 25140) 
v_accvgpr_read v14, a170 // (op 25140) 
v_accvgpr_read v15, a171 // (op 25140) 
v_mul_f32 v8, v19, v12 // (op 25140) 
v_mul_f32 v9, v19, v13 // (op 25140) 
v_mul_f32 v10, v19, v14 // (op 25140) 
v_mul_f32 v11, v19, v15 // (op 25140) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25108) 
v_lshrrev_b32 v7, 16, v16 // (op 25108) 
v_lshlrev_b32 v6, 16, v6 // (op 25108) 
v_lshlrev_b32 v7, 16, v7 // (op 25108) 
v_mul_f32 v12, v18, v6 // (op 25108) 
v_mul_f32 v13, v18, v7 // (op 25108) 
v_and_b32 v6, 65535, v17 // (op 25108) 
v_lshrrev_b32 v7, 16, v17 // (op 25108) 
v_lshlrev_b32 v6, 16, v6 // (op 25108) 
v_lshlrev_b32 v7, 16, v7 // (op 25108) 
v_mul_f32 v14, v18, v6 // (op 25108) 
v_mul_f32 v15, v18, v7 // (op 25108) 


v_add_f32 v4, v12, v8 // (op 25109) 
v_add_f32 v5, v13, v9 // (op 25109) 
v_add_f32 v6, v14, v10 // (op 25109) 
v_add_f32 v7, v15, v11 // (op 25109) 


v_lshrrev_b32 v8, 16, v4 // (op 25110) 
v_lshrrev_b32 v9, 16, v5 // (op 25110) 
v_lshrrev_b32 v10, 16, v6 // (op 25110) 
v_lshrrev_b32 v11, 16, v7 // (op 25110) 


v_pack_b32_f16 v6, v8, v9 // (op 25088) 
v_pack_b32_f16 v7, v10, v11 // (op 25088) 

s_lshl_b32 s0, s4, 7 // (op 25088) 
s_add_u32 s0, s0, 320 // (op 25088) 
v_add_u32 v0, v21, s0 // (op 25088) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25088) Store value





s_lshl_b32 s0, s2, 6 // (op 25116) 
s_mul_i32 s1, 3, s0 // (op 25116) 
s_add_u32 s1, s1, 320 // (op 25116) 

v_add_u32 v0, v20, s1 // (op 25116) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25116) Load value

v_accvgpr_read v12, a172 // (op 25142) 
v_accvgpr_read v13, a173 // (op 25142) 
v_accvgpr_read v14, a174 // (op 25142) 
v_accvgpr_read v15, a175 // (op 25142) 
v_mul_f32 v8, v19, v12 // (op 25142) 
v_mul_f32 v9, v19, v13 // (op 25142) 
v_mul_f32 v10, v19, v14 // (op 25142) 
v_mul_f32 v11, v19, v15 // (op 25142) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25113) 
v_lshrrev_b32 v7, 16, v16 // (op 25113) 
v_lshlrev_b32 v6, 16, v6 // (op 25113) 
v_lshlrev_b32 v7, 16, v7 // (op 25113) 
v_mul_f32 v12, v18, v6 // (op 25113) 
v_mul_f32 v13, v18, v7 // (op 25113) 
v_and_b32 v6, 65535, v17 // (op 25113) 
v_lshrrev_b32 v7, 16, v17 // (op 25113) 
v_lshlrev_b32 v6, 16, v6 // (op 25113) 
v_lshlrev_b32 v7, 16, v7 // (op 25113) 
v_mul_f32 v14, v18, v6 // (op 25113) 
v_mul_f32 v15, v18, v7 // (op 25113) 


v_add_f32 v4, v12, v8 // (op 25114) 
v_add_f32 v5, v13, v9 // (op 25114) 
v_add_f32 v6, v14, v10 // (op 25114) 
v_add_f32 v7, v15, v11 // (op 25114) 


v_lshrrev_b32 v8, 16, v4 // (op 25115) 
v_lshrrev_b32 v9, 16, v5 // (op 25115) 
v_lshrrev_b32 v10, 16, v6 // (op 25115) 
v_lshrrev_b32 v11, 16, v7 // (op 25115) 


v_pack_b32_f16 v6, v8, v9 // (op 25090) 
v_pack_b32_f16 v7, v10, v11 // (op 25090) 

s_lshl_b32 s0, s4, 6 // (op 25090) 
s_mul_i32 s1, 3, s0 // (op 25090) 
s_add_u32 s1, s1, 320 // (op 25090) 

v_add_u32 v0, v21, s1 // (op 25090) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25090) Store value





s_lshl_b32 s0, s2, 8 // (op 25121) 
s_add_u32 s0, s0, 320 // (op 25121) 
v_add_u32 v0, v20, s0 // (op 25121) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25121) Load value

v_accvgpr_read v12, a176 // (op 25144) 
v_accvgpr_read v13, a177 // (op 25144) 
v_accvgpr_read v14, a178 // (op 25144) 
v_accvgpr_read v15, a179 // (op 25144) 
v_mul_f32 v8, v19, v12 // (op 25144) 
v_mul_f32 v9, v19, v13 // (op 25144) 
v_mul_f32 v10, v19, v14 // (op 25144) 
v_mul_f32 v11, v19, v15 // (op 25144) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25118) 
v_lshrrev_b32 v7, 16, v16 // (op 25118) 
v_lshlrev_b32 v6, 16, v6 // (op 25118) 
v_lshlrev_b32 v7, 16, v7 // (op 25118) 
v_mul_f32 v12, v18, v6 // (op 25118) 
v_mul_f32 v13, v18, v7 // (op 25118) 
v_and_b32 v6, 65535, v17 // (op 25118) 
v_lshrrev_b32 v7, 16, v17 // (op 25118) 
v_lshlrev_b32 v6, 16, v6 // (op 25118) 
v_lshlrev_b32 v7, 16, v7 // (op 25118) 
v_mul_f32 v14, v18, v6 // (op 25118) 
v_mul_f32 v15, v18, v7 // (op 25118) 


v_add_f32 v4, v12, v8 // (op 25119) 
v_add_f32 v5, v13, v9 // (op 25119) 
v_add_f32 v6, v14, v10 // (op 25119) 
v_add_f32 v7, v15, v11 // (op 25119) 


v_lshrrev_b32 v8, 16, v4 // (op 25120) 
v_lshrrev_b32 v9, 16, v5 // (op 25120) 
v_lshrrev_b32 v10, 16, v6 // (op 25120) 
v_lshrrev_b32 v11, 16, v7 // (op 25120) 


v_pack_b32_f16 v6, v8, v9 // (op 25092) 
v_pack_b32_f16 v7, v10, v11 // (op 25092) 

s_lshl_b32 s0, s4, 8 // (op 25092) 
s_add_u32 s0, s0, 320 // (op 25092) 
v_add_u32 v0, v21, s0 // (op 25092) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25092) Store value





s_lshl_b32 s0, s2, 6 // (op 25126) 
s_mul_i32 s1, 5, s0 // (op 25126) 
s_add_u32 s1, s1, 320 // (op 25126) 

v_add_u32 v0, v20, s1 // (op 25126) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25126) Load value

v_accvgpr_read v12, a180 // (op 25146) 
v_accvgpr_read v13, a181 // (op 25146) 
v_accvgpr_read v14, a182 // (op 25146) 
v_accvgpr_read v15, a183 // (op 25146) 
v_mul_f32 v8, v19, v12 // (op 25146) 
v_mul_f32 v9, v19, v13 // (op 25146) 
v_mul_f32 v10, v19, v14 // (op 25146) 
v_mul_f32 v11, v19, v15 // (op 25146) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25123) 
v_lshrrev_b32 v7, 16, v16 // (op 25123) 
v_lshlrev_b32 v6, 16, v6 // (op 25123) 
v_lshlrev_b32 v7, 16, v7 // (op 25123) 
v_mul_f32 v12, v18, v6 // (op 25123) 
v_mul_f32 v13, v18, v7 // (op 25123) 
v_and_b32 v6, 65535, v17 // (op 25123) 
v_lshrrev_b32 v7, 16, v17 // (op 25123) 
v_lshlrev_b32 v6, 16, v6 // (op 25123) 
v_lshlrev_b32 v7, 16, v7 // (op 25123) 
v_mul_f32 v14, v18, v6 // (op 25123) 
v_mul_f32 v15, v18, v7 // (op 25123) 


v_add_f32 v4, v12, v8 // (op 25124) 
v_add_f32 v5, v13, v9 // (op 25124) 
v_add_f32 v6, v14, v10 // (op 25124) 
v_add_f32 v7, v15, v11 // (op 25124) 


v_lshrrev_b32 v8, 16, v4 // (op 25125) 
v_lshrrev_b32 v9, 16, v5 // (op 25125) 
v_lshrrev_b32 v10, 16, v6 // (op 25125) 
v_lshrrev_b32 v11, 16, v7 // (op 25125) 


v_pack_b32_f16 v6, v8, v9 // (op 25094) 
v_pack_b32_f16 v7, v10, v11 // (op 25094) 

s_lshl_b32 s0, s4, 6 // (op 25094) 
s_mul_i32 s1, 5, s0 // (op 25094) 
s_add_u32 s1, s1, 320 // (op 25094) 

v_add_u32 v0, v21, s1 // (op 25094) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25094) Store value





s_lshl_b32 s0, s2, 6 // (op 25131) 
s_mul_i32 s1, 6, s0 // (op 25131) 
s_add_u32 s1, s1, 320 // (op 25131) 

v_add_u32 v0, v20, s1 // (op 25131) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25131) Load value

v_accvgpr_read v12, a184 // (op 25148) 
v_accvgpr_read v13, a185 // (op 25148) 
v_accvgpr_read v14, a186 // (op 25148) 
v_accvgpr_read v15, a187 // (op 25148) 
v_mul_f32 v8, v19, v12 // (op 25148) 
v_mul_f32 v9, v19, v13 // (op 25148) 
v_mul_f32 v10, v19, v14 // (op 25148) 
v_mul_f32 v11, v19, v15 // (op 25148) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25128) 
v_lshrrev_b32 v7, 16, v16 // (op 25128) 
v_lshlrev_b32 v6, 16, v6 // (op 25128) 
v_lshlrev_b32 v7, 16, v7 // (op 25128) 
v_mul_f32 v12, v18, v6 // (op 25128) 
v_mul_f32 v13, v18, v7 // (op 25128) 
v_and_b32 v6, 65535, v17 // (op 25128) 
v_lshrrev_b32 v7, 16, v17 // (op 25128) 
v_lshlrev_b32 v6, 16, v6 // (op 25128) 
v_lshlrev_b32 v7, 16, v7 // (op 25128) 
v_mul_f32 v14, v18, v6 // (op 25128) 
v_mul_f32 v15, v18, v7 // (op 25128) 


v_add_f32 v4, v12, v8 // (op 25129) 
v_add_f32 v5, v13, v9 // (op 25129) 
v_add_f32 v6, v14, v10 // (op 25129) 
v_add_f32 v7, v15, v11 // (op 25129) 


v_lshrrev_b32 v8, 16, v4 // (op 25130) 
v_lshrrev_b32 v9, 16, v5 // (op 25130) 
v_lshrrev_b32 v10, 16, v6 // (op 25130) 
v_lshrrev_b32 v11, 16, v7 // (op 25130) 


v_pack_b32_f16 v6, v8, v9 // (op 25096) 
v_pack_b32_f16 v7, v10, v11 // (op 25096) 

s_lshl_b32 s0, s4, 6 // (op 25096) 
s_mul_i32 s1, 6, s0 // (op 25096) 
s_add_u32 s1, s1, 320 // (op 25096) 

v_add_u32 v0, v21, s1 // (op 25096) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25096) Store value





s_lshl_b32 s0, s2, 6 // (op 25136) 
s_mul_i32 s1, 7, s0 // (op 25136) 
s_add_u32 s1, s1, 320 // (op 25136) 

v_add_u32 v0, v20, s1 // (op 25136) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 25136) Load value

v_accvgpr_read v12, a188 // (op 25150) 
v_accvgpr_read v13, a189 // (op 25150) 
v_accvgpr_read v14, a190 // (op 25150) 
v_accvgpr_read v15, a191 // (op 25150) 
v_mul_f32 v8, v19, v12 // (op 25150) 
v_mul_f32 v9, v19, v13 // (op 25150) 
v_mul_f32 v10, v19, v14 // (op 25150) 
v_mul_f32 v11, v19, v15 // (op 25150) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 25133) 
v_lshrrev_b32 v7, 16, v16 // (op 25133) 
v_lshlrev_b32 v6, 16, v6 // (op 25133) 
v_lshlrev_b32 v7, 16, v7 // (op 25133) 
v_mul_f32 v12, v18, v6 // (op 25133) 
v_mul_f32 v13, v18, v7 // (op 25133) 
v_and_b32 v6, 65535, v17 // (op 25133) 
v_lshrrev_b32 v7, 16, v17 // (op 25133) 
v_lshlrev_b32 v6, 16, v6 // (op 25133) 
v_lshlrev_b32 v7, 16, v7 // (op 25133) 
v_mul_f32 v14, v18, v6 // (op 25133) 
v_mul_f32 v15, v18, v7 // (op 25133) 


v_add_f32 v4, v12, v8 // (op 25134) 
v_add_f32 v5, v13, v9 // (op 25134) 
v_add_f32 v6, v14, v10 // (op 25134) 
v_add_f32 v7, v15, v11 // (op 25134) 


v_lshrrev_b32 v8, 16, v4 // (op 25135) 
v_lshrrev_b32 v9, 16, v5 // (op 25135) 
v_lshrrev_b32 v10, 16, v6 // (op 25135) 
v_lshrrev_b32 v11, 16, v7 // (op 25135) 


v_pack_b32_f16 v6, v8, v9 // (op 25098) 
v_pack_b32_f16 v7, v10, v11 // (op 25098) 

s_lshl_b32 s0, s4, 6 // (op 25098) 
s_mul_i32 s1, 7, s0 // (op 25098) 
s_add_u32 s1, s1, 320 // (op 25098) 

v_add_u32 v0, v21, s1 // (op 25098) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 25098) Store value





v_add_u32 v0, 192, v20 // (op 24759) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24759) Load value

v_accvgpr_read v12, a96 // (op 24738) 
v_accvgpr_read v13, a97 // (op 24738) 
v_accvgpr_read v14, a98 // (op 24738) 
v_accvgpr_read v15, a99 // (op 24738) 
v_mul_f32 v8, v19, v12 // (op 24738) 
v_mul_f32 v9, v19, v13 // (op 24738) 
v_mul_f32 v10, v19, v14 // (op 24738) 
v_mul_f32 v11, v19, v15 // (op 24738) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24758) 
v_lshrrev_b32 v7, 16, v16 // (op 24758) 
v_lshlrev_b32 v6, 16, v6 // (op 24758) 
v_lshlrev_b32 v7, 16, v7 // (op 24758) 
v_mul_f32 v12, v18, v6 // (op 24758) 
v_mul_f32 v13, v18, v7 // (op 24758) 
v_and_b32 v6, 65535, v17 // (op 24758) 
v_lshrrev_b32 v7, 16, v17 // (op 24758) 
v_lshlrev_b32 v6, 16, v6 // (op 24758) 
v_lshlrev_b32 v7, 16, v7 // (op 24758) 
v_mul_f32 v14, v18, v6 // (op 24758) 
v_mul_f32 v15, v18, v7 // (op 24758) 


v_add_f32 v4, v12, v8 // (op 24739) 
v_add_f32 v5, v13, v9 // (op 24739) 
v_add_f32 v6, v14, v10 // (op 24739) 
v_add_f32 v7, v15, v11 // (op 24739) 


v_lshrrev_b32 v8, 16, v4 // (op 24740) 
v_lshrrev_b32 v9, 16, v5 // (op 24740) 
v_lshrrev_b32 v10, 16, v6 // (op 24740) 
v_lshrrev_b32 v11, 16, v7 // (op 24740) 


v_pack_b32_f16 v6, v8, v9 // (op 24742) 
v_pack_b32_f16 v7, v10, v11 // (op 24742) 

v_add_u32 v0, 192, v21 // (op 24742) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24742) Store value





s_lshl_b32 s0, s2, 6 // (op 24764) 
s_add_u32 s0, s0, 192 // (op 24764) 
v_add_u32 v0, v20, s0 // (op 24764) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24764) Load value

v_accvgpr_read v12, a100 // (op 24796) 
v_accvgpr_read v13, a101 // (op 24796) 
v_accvgpr_read v14, a102 // (op 24796) 
v_accvgpr_read v15, a103 // (op 24796) 
v_mul_f32 v8, v19, v12 // (op 24796) 
v_mul_f32 v9, v19, v13 // (op 24796) 
v_mul_f32 v10, v19, v14 // (op 24796) 
v_mul_f32 v11, v19, v15 // (op 24796) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24761) 
v_lshrrev_b32 v7, 16, v16 // (op 24761) 
v_lshlrev_b32 v6, 16, v6 // (op 24761) 
v_lshlrev_b32 v7, 16, v7 // (op 24761) 
v_mul_f32 v12, v18, v6 // (op 24761) 
v_mul_f32 v13, v18, v7 // (op 24761) 
v_and_b32 v6, 65535, v17 // (op 24761) 
v_lshrrev_b32 v7, 16, v17 // (op 24761) 
v_lshlrev_b32 v6, 16, v6 // (op 24761) 
v_lshlrev_b32 v7, 16, v7 // (op 24761) 
v_mul_f32 v14, v18, v6 // (op 24761) 
v_mul_f32 v15, v18, v7 // (op 24761) 


v_add_f32 v4, v12, v8 // (op 24762) 
v_add_f32 v5, v13, v9 // (op 24762) 
v_add_f32 v6, v14, v10 // (op 24762) 
v_add_f32 v7, v15, v11 // (op 24762) 


v_lshrrev_b32 v8, 16, v4 // (op 24763) 
v_lshrrev_b32 v9, 16, v5 // (op 24763) 
v_lshrrev_b32 v10, 16, v6 // (op 24763) 
v_lshrrev_b32 v11, 16, v7 // (op 24763) 


v_pack_b32_f16 v6, v8, v9 // (op 24744) 
v_pack_b32_f16 v7, v10, v11 // (op 24744) 

s_lshl_b32 s0, s4, 6 // (op 24744) 
s_add_u32 s0, s0, 192 // (op 24744) 
v_add_u32 v0, v21, s0 // (op 24744) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24744) Store value





s_lshl_b32 s0, s2, 7 // (op 24769) 
s_add_u32 s0, s0, 192 // (op 24769) 
v_add_u32 v0, v20, s0 // (op 24769) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24769) Load value

v_accvgpr_read v12, a104 // (op 24798) 
v_accvgpr_read v13, a105 // (op 24798) 
v_accvgpr_read v14, a106 // (op 24798) 
v_accvgpr_read v15, a107 // (op 24798) 
v_mul_f32 v8, v19, v12 // (op 24798) 
v_mul_f32 v9, v19, v13 // (op 24798) 
v_mul_f32 v10, v19, v14 // (op 24798) 
v_mul_f32 v11, v19, v15 // (op 24798) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24766) 
v_lshrrev_b32 v7, 16, v16 // (op 24766) 
v_lshlrev_b32 v6, 16, v6 // (op 24766) 
v_lshlrev_b32 v7, 16, v7 // (op 24766) 
v_mul_f32 v12, v18, v6 // (op 24766) 
v_mul_f32 v13, v18, v7 // (op 24766) 
v_and_b32 v6, 65535, v17 // (op 24766) 
v_lshrrev_b32 v7, 16, v17 // (op 24766) 
v_lshlrev_b32 v6, 16, v6 // (op 24766) 
v_lshlrev_b32 v7, 16, v7 // (op 24766) 
v_mul_f32 v14, v18, v6 // (op 24766) 
v_mul_f32 v15, v18, v7 // (op 24766) 


v_add_f32 v4, v12, v8 // (op 24767) 
v_add_f32 v5, v13, v9 // (op 24767) 
v_add_f32 v6, v14, v10 // (op 24767) 
v_add_f32 v7, v15, v11 // (op 24767) 


v_lshrrev_b32 v8, 16, v4 // (op 24768) 
v_lshrrev_b32 v9, 16, v5 // (op 24768) 
v_lshrrev_b32 v10, 16, v6 // (op 24768) 
v_lshrrev_b32 v11, 16, v7 // (op 24768) 


v_pack_b32_f16 v6, v8, v9 // (op 24746) 
v_pack_b32_f16 v7, v10, v11 // (op 24746) 

s_lshl_b32 s0, s4, 7 // (op 24746) 
s_add_u32 s0, s0, 192 // (op 24746) 
v_add_u32 v0, v21, s0 // (op 24746) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24746) Store value





s_lshl_b32 s0, s2, 6 // (op 24774) 
s_mul_i32 s1, 3, s0 // (op 24774) 
s_add_u32 s1, s1, 192 // (op 24774) 

v_add_u32 v0, v20, s1 // (op 24774) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24774) Load value

v_accvgpr_read v12, a108 // (op 24800) 
v_accvgpr_read v13, a109 // (op 24800) 
v_accvgpr_read v14, a110 // (op 24800) 
v_accvgpr_read v15, a111 // (op 24800) 
v_mul_f32 v8, v19, v12 // (op 24800) 
v_mul_f32 v9, v19, v13 // (op 24800) 
v_mul_f32 v10, v19, v14 // (op 24800) 
v_mul_f32 v11, v19, v15 // (op 24800) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24771) 
v_lshrrev_b32 v7, 16, v16 // (op 24771) 
v_lshlrev_b32 v6, 16, v6 // (op 24771) 
v_lshlrev_b32 v7, 16, v7 // (op 24771) 
v_mul_f32 v12, v18, v6 // (op 24771) 
v_mul_f32 v13, v18, v7 // (op 24771) 
v_and_b32 v6, 65535, v17 // (op 24771) 
v_lshrrev_b32 v7, 16, v17 // (op 24771) 
v_lshlrev_b32 v6, 16, v6 // (op 24771) 
v_lshlrev_b32 v7, 16, v7 // (op 24771) 
v_mul_f32 v14, v18, v6 // (op 24771) 
v_mul_f32 v15, v18, v7 // (op 24771) 


v_add_f32 v4, v12, v8 // (op 24772) 
v_add_f32 v5, v13, v9 // (op 24772) 
v_add_f32 v6, v14, v10 // (op 24772) 
v_add_f32 v7, v15, v11 // (op 24772) 


v_lshrrev_b32 v8, 16, v4 // (op 24773) 
v_lshrrev_b32 v9, 16, v5 // (op 24773) 
v_lshrrev_b32 v10, 16, v6 // (op 24773) 
v_lshrrev_b32 v11, 16, v7 // (op 24773) 


v_pack_b32_f16 v6, v8, v9 // (op 24748) 
v_pack_b32_f16 v7, v10, v11 // (op 24748) 

s_lshl_b32 s0, s4, 6 // (op 24748) 
s_mul_i32 s1, 3, s0 // (op 24748) 
s_add_u32 s1, s1, 192 // (op 24748) 

v_add_u32 v0, v21, s1 // (op 24748) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24748) Store value





s_lshl_b32 s0, s2, 8 // (op 24779) 
s_add_u32 s0, s0, 192 // (op 24779) 
v_add_u32 v0, v20, s0 // (op 24779) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24779) Load value

v_accvgpr_read v12, a112 // (op 24802) 
v_accvgpr_read v13, a113 // (op 24802) 
v_accvgpr_read v14, a114 // (op 24802) 
v_accvgpr_read v15, a115 // (op 24802) 
v_mul_f32 v8, v19, v12 // (op 24802) 
v_mul_f32 v9, v19, v13 // (op 24802) 
v_mul_f32 v10, v19, v14 // (op 24802) 
v_mul_f32 v11, v19, v15 // (op 24802) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24776) 
v_lshrrev_b32 v7, 16, v16 // (op 24776) 
v_lshlrev_b32 v6, 16, v6 // (op 24776) 
v_lshlrev_b32 v7, 16, v7 // (op 24776) 
v_mul_f32 v12, v18, v6 // (op 24776) 
v_mul_f32 v13, v18, v7 // (op 24776) 
v_and_b32 v6, 65535, v17 // (op 24776) 
v_lshrrev_b32 v7, 16, v17 // (op 24776) 
v_lshlrev_b32 v6, 16, v6 // (op 24776) 
v_lshlrev_b32 v7, 16, v7 // (op 24776) 
v_mul_f32 v14, v18, v6 // (op 24776) 
v_mul_f32 v15, v18, v7 // (op 24776) 


v_add_f32 v4, v12, v8 // (op 24777) 
v_add_f32 v5, v13, v9 // (op 24777) 
v_add_f32 v6, v14, v10 // (op 24777) 
v_add_f32 v7, v15, v11 // (op 24777) 


v_lshrrev_b32 v8, 16, v4 // (op 24778) 
v_lshrrev_b32 v9, 16, v5 // (op 24778) 
v_lshrrev_b32 v10, 16, v6 // (op 24778) 
v_lshrrev_b32 v11, 16, v7 // (op 24778) 


v_pack_b32_f16 v6, v8, v9 // (op 24750) 
v_pack_b32_f16 v7, v10, v11 // (op 24750) 

s_lshl_b32 s0, s4, 8 // (op 24750) 
s_add_u32 s0, s0, 192 // (op 24750) 
v_add_u32 v0, v21, s0 // (op 24750) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24750) Store value





s_lshl_b32 s0, s2, 6 // (op 24784) 
s_mul_i32 s1, 5, s0 // (op 24784) 
s_add_u32 s1, s1, 192 // (op 24784) 

v_add_u32 v0, v20, s1 // (op 24784) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24784) Load value

v_accvgpr_read v12, a116 // (op 24804) 
v_accvgpr_read v13, a117 // (op 24804) 
v_accvgpr_read v14, a118 // (op 24804) 
v_accvgpr_read v15, a119 // (op 24804) 
v_mul_f32 v8, v19, v12 // (op 24804) 
v_mul_f32 v9, v19, v13 // (op 24804) 
v_mul_f32 v10, v19, v14 // (op 24804) 
v_mul_f32 v11, v19, v15 // (op 24804) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24781) 
v_lshrrev_b32 v7, 16, v16 // (op 24781) 
v_lshlrev_b32 v6, 16, v6 // (op 24781) 
v_lshlrev_b32 v7, 16, v7 // (op 24781) 
v_mul_f32 v12, v18, v6 // (op 24781) 
v_mul_f32 v13, v18, v7 // (op 24781) 
v_and_b32 v6, 65535, v17 // (op 24781) 
v_lshrrev_b32 v7, 16, v17 // (op 24781) 
v_lshlrev_b32 v6, 16, v6 // (op 24781) 
v_lshlrev_b32 v7, 16, v7 // (op 24781) 
v_mul_f32 v14, v18, v6 // (op 24781) 
v_mul_f32 v15, v18, v7 // (op 24781) 


v_add_f32 v4, v12, v8 // (op 24782) 
v_add_f32 v5, v13, v9 // (op 24782) 
v_add_f32 v6, v14, v10 // (op 24782) 
v_add_f32 v7, v15, v11 // (op 24782) 


v_lshrrev_b32 v8, 16, v4 // (op 24783) 
v_lshrrev_b32 v9, 16, v5 // (op 24783) 
v_lshrrev_b32 v10, 16, v6 // (op 24783) 
v_lshrrev_b32 v11, 16, v7 // (op 24783) 


v_pack_b32_f16 v6, v8, v9 // (op 24752) 
v_pack_b32_f16 v7, v10, v11 // (op 24752) 

s_lshl_b32 s0, s4, 6 // (op 24752) 
s_mul_i32 s1, 5, s0 // (op 24752) 
s_add_u32 s1, s1, 192 // (op 24752) 

v_add_u32 v0, v21, s1 // (op 24752) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24752) Store value





s_lshl_b32 s0, s2, 6 // (op 24789) 
s_mul_i32 s1, 6, s0 // (op 24789) 
s_add_u32 s1, s1, 192 // (op 24789) 

v_add_u32 v0, v20, s1 // (op 24789) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24789) Load value

v_accvgpr_read v12, a120 // (op 24806) 
v_accvgpr_read v13, a121 // (op 24806) 
v_accvgpr_read v14, a122 // (op 24806) 
v_accvgpr_read v15, a123 // (op 24806) 
v_mul_f32 v8, v19, v12 // (op 24806) 
v_mul_f32 v9, v19, v13 // (op 24806) 
v_mul_f32 v10, v19, v14 // (op 24806) 
v_mul_f32 v11, v19, v15 // (op 24806) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24786) 
v_lshrrev_b32 v7, 16, v16 // (op 24786) 
v_lshlrev_b32 v6, 16, v6 // (op 24786) 
v_lshlrev_b32 v7, 16, v7 // (op 24786) 
v_mul_f32 v12, v18, v6 // (op 24786) 
v_mul_f32 v13, v18, v7 // (op 24786) 
v_and_b32 v6, 65535, v17 // (op 24786) 
v_lshrrev_b32 v7, 16, v17 // (op 24786) 
v_lshlrev_b32 v6, 16, v6 // (op 24786) 
v_lshlrev_b32 v7, 16, v7 // (op 24786) 
v_mul_f32 v14, v18, v6 // (op 24786) 
v_mul_f32 v15, v18, v7 // (op 24786) 


v_add_f32 v4, v12, v8 // (op 24787) 
v_add_f32 v5, v13, v9 // (op 24787) 
v_add_f32 v6, v14, v10 // (op 24787) 
v_add_f32 v7, v15, v11 // (op 24787) 


v_lshrrev_b32 v8, 16, v4 // (op 24788) 
v_lshrrev_b32 v9, 16, v5 // (op 24788) 
v_lshrrev_b32 v10, 16, v6 // (op 24788) 
v_lshrrev_b32 v11, 16, v7 // (op 24788) 


v_pack_b32_f16 v6, v8, v9 // (op 24754) 
v_pack_b32_f16 v7, v10, v11 // (op 24754) 

s_lshl_b32 s0, s4, 6 // (op 24754) 
s_mul_i32 s1, 6, s0 // (op 24754) 
s_add_u32 s1, s1, 192 // (op 24754) 

v_add_u32 v0, v21, s1 // (op 24754) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24754) Store value





s_lshl_b32 s0, s2, 6 // (op 24794) 
s_mul_i32 s1, 7, s0 // (op 24794) 
s_add_u32 s1, s1, 192 // (op 24794) 

v_add_u32 v0, v20, s1 // (op 24794) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24794) Load value

v_accvgpr_read v12, a124 // (op 24808) 
v_accvgpr_read v13, a125 // (op 24808) 
v_accvgpr_read v14, a126 // (op 24808) 
v_accvgpr_read v15, a127 // (op 24808) 
v_mul_f32 v8, v19, v12 // (op 24808) 
v_mul_f32 v9, v19, v13 // (op 24808) 
v_mul_f32 v10, v19, v14 // (op 24808) 
v_mul_f32 v11, v19, v15 // (op 24808) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24791) 
v_lshrrev_b32 v7, 16, v16 // (op 24791) 
v_lshlrev_b32 v6, 16, v6 // (op 24791) 
v_lshlrev_b32 v7, 16, v7 // (op 24791) 
v_mul_f32 v12, v18, v6 // (op 24791) 
v_mul_f32 v13, v18, v7 // (op 24791) 
v_and_b32 v6, 65535, v17 // (op 24791) 
v_lshrrev_b32 v7, 16, v17 // (op 24791) 
v_lshlrev_b32 v6, 16, v6 // (op 24791) 
v_lshlrev_b32 v7, 16, v7 // (op 24791) 
v_mul_f32 v14, v18, v6 // (op 24791) 
v_mul_f32 v15, v18, v7 // (op 24791) 


v_add_f32 v4, v12, v8 // (op 24792) 
v_add_f32 v5, v13, v9 // (op 24792) 
v_add_f32 v6, v14, v10 // (op 24792) 
v_add_f32 v7, v15, v11 // (op 24792) 


v_lshrrev_b32 v8, 16, v4 // (op 24793) 
v_lshrrev_b32 v9, 16, v5 // (op 24793) 
v_lshrrev_b32 v10, 16, v6 // (op 24793) 
v_lshrrev_b32 v11, 16, v7 // (op 24793) 


v_pack_b32_f16 v6, v8, v9 // (op 24756) 
v_pack_b32_f16 v7, v10, v11 // (op 24756) 

s_lshl_b32 s0, s4, 6 // (op 24756) 
s_mul_i32 s1, 7, s0 // (op 24756) 
s_add_u32 s1, s1, 192 // (op 24756) 

v_add_u32 v0, v21, s1 // (op 24756) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24756) Store value





v_add_u32 v0, 128, v20 // (op 24588) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24588) Load value

v_accvgpr_read v12, a64 // (op 24567) 
v_accvgpr_read v13, a65 // (op 24567) 
v_accvgpr_read v14, a66 // (op 24567) 
v_accvgpr_read v15, a67 // (op 24567) 
v_mul_f32 v8, v19, v12 // (op 24567) 
v_mul_f32 v9, v19, v13 // (op 24567) 
v_mul_f32 v10, v19, v14 // (op 24567) 
v_mul_f32 v11, v19, v15 // (op 24567) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24587) 
v_lshrrev_b32 v7, 16, v16 // (op 24587) 
v_lshlrev_b32 v6, 16, v6 // (op 24587) 
v_lshlrev_b32 v7, 16, v7 // (op 24587) 
v_mul_f32 v12, v18, v6 // (op 24587) 
v_mul_f32 v13, v18, v7 // (op 24587) 
v_and_b32 v6, 65535, v17 // (op 24587) 
v_lshrrev_b32 v7, 16, v17 // (op 24587) 
v_lshlrev_b32 v6, 16, v6 // (op 24587) 
v_lshlrev_b32 v7, 16, v7 // (op 24587) 
v_mul_f32 v14, v18, v6 // (op 24587) 
v_mul_f32 v15, v18, v7 // (op 24587) 


v_add_f32 v4, v12, v8 // (op 24568) 
v_add_f32 v5, v13, v9 // (op 24568) 
v_add_f32 v6, v14, v10 // (op 24568) 
v_add_f32 v7, v15, v11 // (op 24568) 


v_lshrrev_b32 v8, 16, v4 // (op 24569) 
v_lshrrev_b32 v9, 16, v5 // (op 24569) 
v_lshrrev_b32 v10, 16, v6 // (op 24569) 
v_lshrrev_b32 v11, 16, v7 // (op 24569) 


v_pack_b32_f16 v6, v8, v9 // (op 24571) 
v_pack_b32_f16 v7, v10, v11 // (op 24571) 

v_add_u32 v0, 128, v21 // (op 24571) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24571) Store value





s_lshl_b32 s0, s2, 6 // (op 24593) 
s_add_u32 s0, s0, 128 // (op 24593) 
v_add_u32 v0, v20, s0 // (op 24593) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24593) Load value

v_accvgpr_read v12, a68 // (op 24625) 
v_accvgpr_read v13, a69 // (op 24625) 
v_accvgpr_read v14, a70 // (op 24625) 
v_accvgpr_read v15, a71 // (op 24625) 
v_mul_f32 v8, v19, v12 // (op 24625) 
v_mul_f32 v9, v19, v13 // (op 24625) 
v_mul_f32 v10, v19, v14 // (op 24625) 
v_mul_f32 v11, v19, v15 // (op 24625) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24590) 
v_lshrrev_b32 v7, 16, v16 // (op 24590) 
v_lshlrev_b32 v6, 16, v6 // (op 24590) 
v_lshlrev_b32 v7, 16, v7 // (op 24590) 
v_mul_f32 v12, v18, v6 // (op 24590) 
v_mul_f32 v13, v18, v7 // (op 24590) 
v_and_b32 v6, 65535, v17 // (op 24590) 
v_lshrrev_b32 v7, 16, v17 // (op 24590) 
v_lshlrev_b32 v6, 16, v6 // (op 24590) 
v_lshlrev_b32 v7, 16, v7 // (op 24590) 
v_mul_f32 v14, v18, v6 // (op 24590) 
v_mul_f32 v15, v18, v7 // (op 24590) 


v_add_f32 v4, v12, v8 // (op 24591) 
v_add_f32 v5, v13, v9 // (op 24591) 
v_add_f32 v6, v14, v10 // (op 24591) 
v_add_f32 v7, v15, v11 // (op 24591) 


v_lshrrev_b32 v8, 16, v4 // (op 24592) 
v_lshrrev_b32 v9, 16, v5 // (op 24592) 
v_lshrrev_b32 v10, 16, v6 // (op 24592) 
v_lshrrev_b32 v11, 16, v7 // (op 24592) 


v_pack_b32_f16 v6, v8, v9 // (op 24573) 
v_pack_b32_f16 v7, v10, v11 // (op 24573) 

s_lshl_b32 s0, s4, 6 // (op 24573) 
s_add_u32 s0, s0, 128 // (op 24573) 
v_add_u32 v0, v21, s0 // (op 24573) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24573) Store value





s_lshl_b32 s0, s2, 7 // (op 24598) 
s_add_u32 s0, s0, 128 // (op 24598) 
v_add_u32 v0, v20, s0 // (op 24598) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 24598) Load value

v_accvgpr_read v12, a72 // (op 24627) 
v_accvgpr_read v13, a73 // (op 24627) 
v_accvgpr_read v14, a74 // (op 24627) 
v_accvgpr_read v15, a75 // (op 24627) 
v_mul_f32 v8, v19, v12 // (op 24627) 
v_mul_f32 v9, v19, v13 // (op 24627) 
v_mul_f32 v10, v19, v14 // (op 24627) 
v_mul_f32 v11, v19, v15 // (op 24627) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v16 // (op 24595) 
v_lshrrev_b32 v7, 16, v16 // (op 24595) 
v_lshlrev_b32 v6, 16, v6 // (op 24595) 
v_lshlrev_b32 v7, 16, v7 // (op 24595) 
v_mul_f32 v12, v18, v6 // (op 24595) 
v_mul_f32 v13, v18, v7 // (op 24595) 
v_and_b32 v6, 65535, v17 // (op 24595) 
v_lshrrev_b32 v7, 16, v17 // (op 24595) 
v_lshlrev_b32 v6, 16, v6 // (op 24595) 
v_lshlrev_b32 v7, 16, v7 // (op 24595) 
v_mul_f32 v14, v18, v6 // (op 24595) 
v_mul_f32 v15, v18, v7 // (op 24595) 


v_add_f32 v4, v12, v8 // (op 24596) 
v_add_f32 v5, v13, v9 // (op 24596) 
v_add_f32 v6, v14, v10 // (op 24596) 
v_add_f32 v7, v15, v11 // (op 24596) 


v_lshrrev_b32 v8, 16, v4 // (op 24597) 
v_lshrrev_b32 v9, 16, v5 // (op 24597) 
v_lshrrev_b32 v10, 16, v6 // (op 24597) 
v_lshrrev_b32 v11, 16, v7 // (op 24597) 


v_pack_b32_f16 v6, v8, v9 // (op 24575) 
v_pack_b32_f16 v7, v10, v11 // (op 24575) 

s_lshl_b32 s0, s4, 7 // (op 24575) 
s_add_u32 s0, s0, 128 // (op 24575) 
v_add_u32 v0, v21, s0 // (op 24575) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 24575) Store value





s_lshl_b32 s0, s2, 6 // (op 24603) 
s_mul_i32 s1, 3, s0 // (op 24603) 
s_add_u32 s1, s1, 128 // (op 24603) 

v_add_u32 v0, v20, s1 // (op 24603) 


v_accvgpr_read v14, a76 // (op 24629) 
v_accvgpr_read v15, a77 // (op 24629) 
v_accvgpr_read v16, a78 // (op 24629) 
v_accvgpr_read v17, a79 // (op 24629) 
v_mul_f32 v10, v19, v14 // (op 24629) 
v_mul_f32 v11, v19, v15 // (op 24629) 
v_mul_f32 v12, v19, v16 // (op 24629) 
v_mul_f32 v13, v19, v17 // (op 24629) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24603) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24600) 
v_lshrrev_b32 v7, 16, v8 // (op 24600) 
v_lshlrev_b32 v6, 16, v6 // (op 24600) 
v_lshlrev_b32 v7, 16, v7 // (op 24600) 
v_mul_f32 v14, v18, v6 // (op 24600) 
v_mul_f32 v15, v18, v7 // (op 24600) 
v_and_b32 v6, 65535, v9 // (op 24600) 
v_lshrrev_b32 v7, 16, v9 // (op 24600) 
v_lshlrev_b32 v6, 16, v6 // (op 24600) 
v_lshlrev_b32 v7, 16, v7 // (op 24600) 
v_mul_f32 v16, v18, v6 // (op 24600) 
v_mul_f32 v17, v18, v7 // (op 24600) 


v_add_f32 v6, v14, v10 // (op 24601) 
v_add_f32 v7, v15, v11 // (op 24601) 
v_add_f32 v8, v16, v12 // (op 24601) 
v_add_f32 v9, v17, v13 // (op 24601) 


v_lshrrev_b32 v10, 16, v6 // (op 24602) 
v_lshrrev_b32 v11, 16, v7 // (op 24602) 
v_lshrrev_b32 v12, 16, v8 // (op 24602) 
v_lshrrev_b32 v13, 16, v9 // (op 24602) 


v_pack_b32_f16 v8, v10, v11 // (op 24577) 
v_pack_b32_f16 v9, v12, v13 // (op 24577) 

s_lshl_b32 s0, s4, 6 // (op 24577) 
s_mul_i32 s1, 3, s0 // (op 24577) 
s_add_u32 s1, s1, 128 // (op 24577) 

v_add_u32 v0, v21, s1 // (op 24577) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24577) Store value





s_lshl_b32 s0, s2, 8 // (op 24608) 
s_add_u32 s0, s0, 128 // (op 24608) 
v_add_u32 v0, v20, s0 // (op 24608) 


v_accvgpr_read v14, a80 // (op 24631) 
v_accvgpr_read v15, a81 // (op 24631) 
v_accvgpr_read v16, a82 // (op 24631) 
v_accvgpr_read v17, a83 // (op 24631) 
v_mul_f32 v10, v19, v14 // (op 24631) 
v_mul_f32 v11, v19, v15 // (op 24631) 
v_mul_f32 v12, v19, v16 // (op 24631) 
v_mul_f32 v13, v19, v17 // (op 24631) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24608) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24605) 
v_lshrrev_b32 v7, 16, v8 // (op 24605) 
v_lshlrev_b32 v6, 16, v6 // (op 24605) 
v_lshlrev_b32 v7, 16, v7 // (op 24605) 
v_mul_f32 v14, v18, v6 // (op 24605) 
v_mul_f32 v15, v18, v7 // (op 24605) 
v_and_b32 v6, 65535, v9 // (op 24605) 
v_lshrrev_b32 v7, 16, v9 // (op 24605) 
v_lshlrev_b32 v6, 16, v6 // (op 24605) 
v_lshlrev_b32 v7, 16, v7 // (op 24605) 
v_mul_f32 v16, v18, v6 // (op 24605) 
v_mul_f32 v17, v18, v7 // (op 24605) 


v_add_f32 v6, v14, v10 // (op 24606) 
v_add_f32 v7, v15, v11 // (op 24606) 
v_add_f32 v8, v16, v12 // (op 24606) 
v_add_f32 v9, v17, v13 // (op 24606) 


v_lshrrev_b32 v10, 16, v6 // (op 24607) 
v_lshrrev_b32 v11, 16, v7 // (op 24607) 
v_lshrrev_b32 v12, 16, v8 // (op 24607) 
v_lshrrev_b32 v13, 16, v9 // (op 24607) 


v_pack_b32_f16 v8, v10, v11 // (op 24579) 
v_pack_b32_f16 v9, v12, v13 // (op 24579) 

s_lshl_b32 s0, s4, 8 // (op 24579) 
s_add_u32 s0, s0, 128 // (op 24579) 
v_add_u32 v0, v21, s0 // (op 24579) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24579) Store value





s_lshl_b32 s0, s2, 6 // (op 24613) 
s_mul_i32 s1, 5, s0 // (op 24613) 
s_add_u32 s1, s1, 128 // (op 24613) 

v_add_u32 v0, v20, s1 // (op 24613) 


v_accvgpr_read v14, a84 // (op 24633) 
v_accvgpr_read v15, a85 // (op 24633) 
v_accvgpr_read v16, a86 // (op 24633) 
v_accvgpr_read v17, a87 // (op 24633) 
v_mul_f32 v10, v19, v14 // (op 24633) 
v_mul_f32 v11, v19, v15 // (op 24633) 
v_mul_f32 v12, v19, v16 // (op 24633) 
v_mul_f32 v13, v19, v17 // (op 24633) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24613) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24610) 
v_lshrrev_b32 v7, 16, v8 // (op 24610) 
v_lshlrev_b32 v6, 16, v6 // (op 24610) 
v_lshlrev_b32 v7, 16, v7 // (op 24610) 
v_mul_f32 v14, v18, v6 // (op 24610) 
v_mul_f32 v15, v18, v7 // (op 24610) 
v_and_b32 v6, 65535, v9 // (op 24610) 
v_lshrrev_b32 v7, 16, v9 // (op 24610) 
v_lshlrev_b32 v6, 16, v6 // (op 24610) 
v_lshlrev_b32 v7, 16, v7 // (op 24610) 
v_mul_f32 v16, v18, v6 // (op 24610) 
v_mul_f32 v17, v18, v7 // (op 24610) 


v_add_f32 v6, v14, v10 // (op 24611) 
v_add_f32 v7, v15, v11 // (op 24611) 
v_add_f32 v8, v16, v12 // (op 24611) 
v_add_f32 v9, v17, v13 // (op 24611) 


v_lshrrev_b32 v10, 16, v6 // (op 24612) 
v_lshrrev_b32 v11, 16, v7 // (op 24612) 
v_lshrrev_b32 v12, 16, v8 // (op 24612) 
v_lshrrev_b32 v13, 16, v9 // (op 24612) 


v_pack_b32_f16 v8, v10, v11 // (op 24581) 
v_pack_b32_f16 v9, v12, v13 // (op 24581) 

s_lshl_b32 s0, s4, 6 // (op 24581) 
s_mul_i32 s1, 5, s0 // (op 24581) 
s_add_u32 s1, s1, 128 // (op 24581) 

v_add_u32 v0, v21, s1 // (op 24581) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24581) Store value





s_lshl_b32 s0, s2, 6 // (op 24618) 
s_mul_i32 s1, 6, s0 // (op 24618) 
s_add_u32 s1, s1, 128 // (op 24618) 

v_add_u32 v0, v20, s1 // (op 24618) 


v_accvgpr_read v14, a88 // (op 24635) 
v_accvgpr_read v15, a89 // (op 24635) 
v_accvgpr_read v16, a90 // (op 24635) 
v_accvgpr_read v17, a91 // (op 24635) 
v_mul_f32 v10, v19, v14 // (op 24635) 
v_mul_f32 v11, v19, v15 // (op 24635) 
v_mul_f32 v12, v19, v16 // (op 24635) 
v_mul_f32 v13, v19, v17 // (op 24635) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24618) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24615) 
v_lshrrev_b32 v7, 16, v8 // (op 24615) 
v_lshlrev_b32 v6, 16, v6 // (op 24615) 
v_lshlrev_b32 v7, 16, v7 // (op 24615) 
v_mul_f32 v14, v18, v6 // (op 24615) 
v_mul_f32 v15, v18, v7 // (op 24615) 
v_and_b32 v6, 65535, v9 // (op 24615) 
v_lshrrev_b32 v7, 16, v9 // (op 24615) 
v_lshlrev_b32 v6, 16, v6 // (op 24615) 
v_lshlrev_b32 v7, 16, v7 // (op 24615) 
v_mul_f32 v16, v18, v6 // (op 24615) 
v_mul_f32 v17, v18, v7 // (op 24615) 


v_add_f32 v6, v14, v10 // (op 24616) 
v_add_f32 v7, v15, v11 // (op 24616) 
v_add_f32 v8, v16, v12 // (op 24616) 
v_add_f32 v9, v17, v13 // (op 24616) 


v_lshrrev_b32 v10, 16, v6 // (op 24617) 
v_lshrrev_b32 v11, 16, v7 // (op 24617) 
v_lshrrev_b32 v12, 16, v8 // (op 24617) 
v_lshrrev_b32 v13, 16, v9 // (op 24617) 


v_pack_b32_f16 v8, v10, v11 // (op 24583) 
v_pack_b32_f16 v9, v12, v13 // (op 24583) 

s_lshl_b32 s0, s4, 6 // (op 24583) 
s_mul_i32 s1, 6, s0 // (op 24583) 
s_add_u32 s1, s1, 128 // (op 24583) 

v_add_u32 v0, v21, s1 // (op 24583) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24583) Store value





s_lshl_b32 s0, s2, 6 // (op 24623) 
s_mul_i32 s1, 7, s0 // (op 24623) 
s_add_u32 s1, s1, 128 // (op 24623) 

v_add_u32 v0, v20, s1 // (op 24623) 


v_accvgpr_read v14, a92 // (op 24637) 
v_accvgpr_read v15, a93 // (op 24637) 
v_accvgpr_read v16, a94 // (op 24637) 
v_accvgpr_read v17, a95 // (op 24637) 
v_mul_f32 v10, v19, v14 // (op 24637) 
v_mul_f32 v11, v19, v15 // (op 24637) 
v_mul_f32 v12, v19, v16 // (op 24637) 
v_mul_f32 v13, v19, v17 // (op 24637) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24623) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24620) 
v_lshrrev_b32 v7, 16, v8 // (op 24620) 
v_lshlrev_b32 v6, 16, v6 // (op 24620) 
v_lshlrev_b32 v7, 16, v7 // (op 24620) 
v_mul_f32 v14, v18, v6 // (op 24620) 
v_mul_f32 v15, v18, v7 // (op 24620) 
v_and_b32 v6, 65535, v9 // (op 24620) 
v_lshrrev_b32 v7, 16, v9 // (op 24620) 
v_lshlrev_b32 v6, 16, v6 // (op 24620) 
v_lshlrev_b32 v7, 16, v7 // (op 24620) 
v_mul_f32 v16, v18, v6 // (op 24620) 
v_mul_f32 v17, v18, v7 // (op 24620) 


v_add_f32 v6, v14, v10 // (op 24621) 
v_add_f32 v7, v15, v11 // (op 24621) 
v_add_f32 v8, v16, v12 // (op 24621) 
v_add_f32 v9, v17, v13 // (op 24621) 


v_lshrrev_b32 v10, 16, v6 // (op 24622) 
v_lshrrev_b32 v11, 16, v7 // (op 24622) 
v_lshrrev_b32 v12, 16, v8 // (op 24622) 
v_lshrrev_b32 v13, 16, v9 // (op 24622) 


v_pack_b32_f16 v8, v10, v11 // (op 24585) 
v_pack_b32_f16 v9, v12, v13 // (op 24585) 

s_lshl_b32 s0, s4, 6 // (op 24585) 
s_mul_i32 s1, 7, s0 // (op 24585) 
s_add_u32 s1, s1, 128 // (op 24585) 

v_add_u32 v0, v21, s1 // (op 24585) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24585) Store value





v_add_u32 v0, 64, v20 // (op 24417) 

v_accvgpr_read v14, a32 // (op 24396) 
v_accvgpr_read v15, a33 // (op 24396) 
v_accvgpr_read v16, a34 // (op 24396) 
v_accvgpr_read v17, a35 // (op 24396) 
v_mul_f32 v10, v19, v14 // (op 24396) 
v_mul_f32 v11, v19, v15 // (op 24396) 
v_mul_f32 v12, v19, v16 // (op 24396) 
v_mul_f32 v13, v19, v17 // (op 24396) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24417) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24416) 
v_lshrrev_b32 v7, 16, v8 // (op 24416) 
v_lshlrev_b32 v6, 16, v6 // (op 24416) 
v_lshlrev_b32 v7, 16, v7 // (op 24416) 
v_mul_f32 v14, v18, v6 // (op 24416) 
v_mul_f32 v15, v18, v7 // (op 24416) 
v_and_b32 v6, 65535, v9 // (op 24416) 
v_lshrrev_b32 v7, 16, v9 // (op 24416) 
v_lshlrev_b32 v6, 16, v6 // (op 24416) 
v_lshlrev_b32 v7, 16, v7 // (op 24416) 
v_mul_f32 v16, v18, v6 // (op 24416) 
v_mul_f32 v17, v18, v7 // (op 24416) 


v_add_f32 v6, v14, v10 // (op 24397) 
v_add_f32 v7, v15, v11 // (op 24397) 
v_add_f32 v8, v16, v12 // (op 24397) 
v_add_f32 v9, v17, v13 // (op 24397) 


v_lshrrev_b32 v10, 16, v6 // (op 24398) 
v_lshrrev_b32 v11, 16, v7 // (op 24398) 
v_lshrrev_b32 v12, 16, v8 // (op 24398) 
v_lshrrev_b32 v13, 16, v9 // (op 24398) 


v_pack_b32_f16 v8, v10, v11 // (op 24400) 
v_pack_b32_f16 v9, v12, v13 // (op 24400) 

v_add_u32 v0, 64, v21 // (op 24400) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24400) Store value





s_lshl_b32 s0, s2, 6 // (op 24422) 
s_add_u32 s0, s0, 64 // (op 24422) 
v_add_u32 v0, v20, s0 // (op 24422) 


v_accvgpr_read v14, a36 // (op 24454) 
v_accvgpr_read v15, a37 // (op 24454) 
v_accvgpr_read v16, a38 // (op 24454) 
v_accvgpr_read v17, a39 // (op 24454) 
v_mul_f32 v10, v19, v14 // (op 24454) 
v_mul_f32 v11, v19, v15 // (op 24454) 
v_mul_f32 v12, v19, v16 // (op 24454) 
v_mul_f32 v13, v19, v17 // (op 24454) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24422) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24419) 
v_lshrrev_b32 v7, 16, v8 // (op 24419) 
v_lshlrev_b32 v6, 16, v6 // (op 24419) 
v_lshlrev_b32 v7, 16, v7 // (op 24419) 
v_mul_f32 v14, v18, v6 // (op 24419) 
v_mul_f32 v15, v18, v7 // (op 24419) 
v_and_b32 v6, 65535, v9 // (op 24419) 
v_lshrrev_b32 v7, 16, v9 // (op 24419) 
v_lshlrev_b32 v6, 16, v6 // (op 24419) 
v_lshlrev_b32 v7, 16, v7 // (op 24419) 
v_mul_f32 v16, v18, v6 // (op 24419) 
v_mul_f32 v17, v18, v7 // (op 24419) 


v_add_f32 v6, v14, v10 // (op 24420) 
v_add_f32 v7, v15, v11 // (op 24420) 
v_add_f32 v8, v16, v12 // (op 24420) 
v_add_f32 v9, v17, v13 // (op 24420) 


v_lshrrev_b32 v10, 16, v6 // (op 24421) 
v_lshrrev_b32 v11, 16, v7 // (op 24421) 
v_lshrrev_b32 v12, 16, v8 // (op 24421) 
v_lshrrev_b32 v13, 16, v9 // (op 24421) 


v_pack_b32_f16 v8, v10, v11 // (op 24402) 
v_pack_b32_f16 v9, v12, v13 // (op 24402) 

s_lshl_b32 s0, s4, 6 // (op 24402) 
s_add_u32 s0, s0, 64 // (op 24402) 
v_add_u32 v0, v21, s0 // (op 24402) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24402) Store value





s_lshl_b32 s0, s2, 7 // (op 24427) 
s_add_u32 s0, s0, 64 // (op 24427) 
v_add_u32 v0, v20, s0 // (op 24427) 


v_accvgpr_read v14, a40 // (op 24456) 
v_accvgpr_read v15, a41 // (op 24456) 
v_accvgpr_read v16, a42 // (op 24456) 
v_accvgpr_read v17, a43 // (op 24456) 
v_mul_f32 v10, v19, v14 // (op 24456) 
v_mul_f32 v11, v19, v15 // (op 24456) 
v_mul_f32 v12, v19, v16 // (op 24456) 
v_mul_f32 v13, v19, v17 // (op 24456) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24427) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24424) 
v_lshrrev_b32 v7, 16, v8 // (op 24424) 
v_lshlrev_b32 v6, 16, v6 // (op 24424) 
v_lshlrev_b32 v7, 16, v7 // (op 24424) 
v_mul_f32 v14, v18, v6 // (op 24424) 
v_mul_f32 v15, v18, v7 // (op 24424) 
v_and_b32 v6, 65535, v9 // (op 24424) 
v_lshrrev_b32 v7, 16, v9 // (op 24424) 
v_lshlrev_b32 v6, 16, v6 // (op 24424) 
v_lshlrev_b32 v7, 16, v7 // (op 24424) 
v_mul_f32 v16, v18, v6 // (op 24424) 
v_mul_f32 v17, v18, v7 // (op 24424) 


v_add_f32 v6, v14, v10 // (op 24425) 
v_add_f32 v7, v15, v11 // (op 24425) 
v_add_f32 v8, v16, v12 // (op 24425) 
v_add_f32 v9, v17, v13 // (op 24425) 


v_lshrrev_b32 v10, 16, v6 // (op 24426) 
v_lshrrev_b32 v11, 16, v7 // (op 24426) 
v_lshrrev_b32 v12, 16, v8 // (op 24426) 
v_lshrrev_b32 v13, 16, v9 // (op 24426) 


v_pack_b32_f16 v8, v10, v11 // (op 24404) 
v_pack_b32_f16 v9, v12, v13 // (op 24404) 

s_lshl_b32 s0, s4, 7 // (op 24404) 
s_add_u32 s0, s0, 64 // (op 24404) 
v_add_u32 v0, v21, s0 // (op 24404) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24404) Store value





s_lshl_b32 s0, s2, 6 // (op 24432) 
s_mul_i32 s1, 3, s0 // (op 24432) 
s_add_u32 s1, s1, 64 // (op 24432) 

v_add_u32 v0, v20, s1 // (op 24432) 


v_accvgpr_read v14, a44 // (op 24458) 
v_accvgpr_read v15, a45 // (op 24458) 
v_accvgpr_read v16, a46 // (op 24458) 
v_accvgpr_read v17, a47 // (op 24458) 
v_mul_f32 v10, v19, v14 // (op 24458) 
v_mul_f32 v11, v19, v15 // (op 24458) 
v_mul_f32 v12, v19, v16 // (op 24458) 
v_mul_f32 v13, v19, v17 // (op 24458) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24432) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24429) 
v_lshrrev_b32 v7, 16, v8 // (op 24429) 
v_lshlrev_b32 v6, 16, v6 // (op 24429) 
v_lshlrev_b32 v7, 16, v7 // (op 24429) 
v_mul_f32 v14, v18, v6 // (op 24429) 
v_mul_f32 v15, v18, v7 // (op 24429) 
v_and_b32 v6, 65535, v9 // (op 24429) 
v_lshrrev_b32 v7, 16, v9 // (op 24429) 
v_lshlrev_b32 v6, 16, v6 // (op 24429) 
v_lshlrev_b32 v7, 16, v7 // (op 24429) 
v_mul_f32 v16, v18, v6 // (op 24429) 
v_mul_f32 v17, v18, v7 // (op 24429) 


v_add_f32 v6, v14, v10 // (op 24430) 
v_add_f32 v7, v15, v11 // (op 24430) 
v_add_f32 v8, v16, v12 // (op 24430) 
v_add_f32 v9, v17, v13 // (op 24430) 


v_lshrrev_b32 v10, 16, v6 // (op 24431) 
v_lshrrev_b32 v11, 16, v7 // (op 24431) 
v_lshrrev_b32 v12, 16, v8 // (op 24431) 
v_lshrrev_b32 v13, 16, v9 // (op 24431) 


v_pack_b32_f16 v8, v10, v11 // (op 24406) 
v_pack_b32_f16 v9, v12, v13 // (op 24406) 

s_lshl_b32 s0, s4, 6 // (op 24406) 
s_mul_i32 s1, 3, s0 // (op 24406) 
s_add_u32 s1, s1, 64 // (op 24406) 

v_add_u32 v0, v21, s1 // (op 24406) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24406) Store value





s_lshl_b32 s0, s2, 8 // (op 24437) 
s_add_u32 s0, s0, 64 // (op 24437) 
v_add_u32 v0, v20, s0 // (op 24437) 


v_accvgpr_read v14, a48 // (op 24460) 
v_accvgpr_read v15, a49 // (op 24460) 
v_accvgpr_read v16, a50 // (op 24460) 
v_accvgpr_read v17, a51 // (op 24460) 
v_mul_f32 v10, v19, v14 // (op 24460) 
v_mul_f32 v11, v19, v15 // (op 24460) 
v_mul_f32 v12, v19, v16 // (op 24460) 
v_mul_f32 v13, v19, v17 // (op 24460) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24437) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24434) 
v_lshrrev_b32 v7, 16, v8 // (op 24434) 
v_lshlrev_b32 v6, 16, v6 // (op 24434) 
v_lshlrev_b32 v7, 16, v7 // (op 24434) 
v_mul_f32 v14, v18, v6 // (op 24434) 
v_mul_f32 v15, v18, v7 // (op 24434) 
v_and_b32 v6, 65535, v9 // (op 24434) 
v_lshrrev_b32 v7, 16, v9 // (op 24434) 
v_lshlrev_b32 v6, 16, v6 // (op 24434) 
v_lshlrev_b32 v7, 16, v7 // (op 24434) 
v_mul_f32 v16, v18, v6 // (op 24434) 
v_mul_f32 v17, v18, v7 // (op 24434) 


v_add_f32 v6, v14, v10 // (op 24435) 
v_add_f32 v7, v15, v11 // (op 24435) 
v_add_f32 v8, v16, v12 // (op 24435) 
v_add_f32 v9, v17, v13 // (op 24435) 


v_lshrrev_b32 v10, 16, v6 // (op 24436) 
v_lshrrev_b32 v11, 16, v7 // (op 24436) 
v_lshrrev_b32 v12, 16, v8 // (op 24436) 
v_lshrrev_b32 v13, 16, v9 // (op 24436) 


v_pack_b32_f16 v8, v10, v11 // (op 24408) 
v_pack_b32_f16 v9, v12, v13 // (op 24408) 

s_lshl_b32 s0, s4, 8 // (op 24408) 
s_add_u32 s0, s0, 64 // (op 24408) 
v_add_u32 v0, v21, s0 // (op 24408) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24408) Store value





s_lshl_b32 s0, s2, 6 // (op 24442) 
s_mul_i32 s1, 5, s0 // (op 24442) 
s_add_u32 s1, s1, 64 // (op 24442) 

v_add_u32 v0, v20, s1 // (op 24442) 


v_accvgpr_read v14, a52 // (op 24462) 
v_accvgpr_read v15, a53 // (op 24462) 
v_accvgpr_read v16, a54 // (op 24462) 
v_accvgpr_read v17, a55 // (op 24462) 
v_mul_f32 v10, v19, v14 // (op 24462) 
v_mul_f32 v11, v19, v15 // (op 24462) 
v_mul_f32 v12, v19, v16 // (op 24462) 
v_mul_f32 v13, v19, v17 // (op 24462) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24442) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24439) 
v_lshrrev_b32 v7, 16, v8 // (op 24439) 
v_lshlrev_b32 v6, 16, v6 // (op 24439) 
v_lshlrev_b32 v7, 16, v7 // (op 24439) 
v_mul_f32 v14, v18, v6 // (op 24439) 
v_mul_f32 v15, v18, v7 // (op 24439) 
v_and_b32 v6, 65535, v9 // (op 24439) 
v_lshrrev_b32 v7, 16, v9 // (op 24439) 
v_lshlrev_b32 v6, 16, v6 // (op 24439) 
v_lshlrev_b32 v7, 16, v7 // (op 24439) 
v_mul_f32 v16, v18, v6 // (op 24439) 
v_mul_f32 v17, v18, v7 // (op 24439) 


v_add_f32 v6, v14, v10 // (op 24440) 
v_add_f32 v7, v15, v11 // (op 24440) 
v_add_f32 v8, v16, v12 // (op 24440) 
v_add_f32 v9, v17, v13 // (op 24440) 


v_lshrrev_b32 v10, 16, v6 // (op 24441) 
v_lshrrev_b32 v11, 16, v7 // (op 24441) 
v_lshrrev_b32 v12, 16, v8 // (op 24441) 
v_lshrrev_b32 v13, 16, v9 // (op 24441) 


v_pack_b32_f16 v8, v10, v11 // (op 24410) 
v_pack_b32_f16 v9, v12, v13 // (op 24410) 

s_lshl_b32 s0, s4, 6 // (op 24410) 
s_mul_i32 s1, 5, s0 // (op 24410) 
s_add_u32 s1, s1, 64 // (op 24410) 

v_add_u32 v0, v21, s1 // (op 24410) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24410) Store value





s_lshl_b32 s0, s2, 6 // (op 24447) 
s_mul_i32 s1, 6, s0 // (op 24447) 
s_add_u32 s1, s1, 64 // (op 24447) 

v_add_u32 v0, v20, s1 // (op 24447) 


v_accvgpr_read v14, a56 // (op 24464) 
v_accvgpr_read v15, a57 // (op 24464) 
v_accvgpr_read v16, a58 // (op 24464) 
v_accvgpr_read v17, a59 // (op 24464) 
v_mul_f32 v10, v19, v14 // (op 24464) 
v_mul_f32 v11, v19, v15 // (op 24464) 
v_mul_f32 v12, v19, v16 // (op 24464) 
v_mul_f32 v13, v19, v17 // (op 24464) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24447) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24444) 
v_lshrrev_b32 v7, 16, v8 // (op 24444) 
v_lshlrev_b32 v6, 16, v6 // (op 24444) 
v_lshlrev_b32 v7, 16, v7 // (op 24444) 
v_mul_f32 v14, v18, v6 // (op 24444) 
v_mul_f32 v15, v18, v7 // (op 24444) 
v_and_b32 v6, 65535, v9 // (op 24444) 
v_lshrrev_b32 v7, 16, v9 // (op 24444) 
v_lshlrev_b32 v6, 16, v6 // (op 24444) 
v_lshlrev_b32 v7, 16, v7 // (op 24444) 
v_mul_f32 v16, v18, v6 // (op 24444) 
v_mul_f32 v17, v18, v7 // (op 24444) 


v_add_f32 v6, v14, v10 // (op 24445) 
v_add_f32 v7, v15, v11 // (op 24445) 
v_add_f32 v8, v16, v12 // (op 24445) 
v_add_f32 v9, v17, v13 // (op 24445) 


v_lshrrev_b32 v10, 16, v6 // (op 24446) 
v_lshrrev_b32 v11, 16, v7 // (op 24446) 
v_lshrrev_b32 v12, 16, v8 // (op 24446) 
v_lshrrev_b32 v13, 16, v9 // (op 24446) 


v_pack_b32_f16 v8, v10, v11 // (op 24412) 
v_pack_b32_f16 v9, v12, v13 // (op 24412) 

s_lshl_b32 s0, s4, 6 // (op 24412) 
s_mul_i32 s1, 6, s0 // (op 24412) 
s_add_u32 s1, s1, 64 // (op 24412) 

v_add_u32 v0, v21, s1 // (op 24412) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24412) Store value





s_lshl_b32 s0, s2, 6 // (op 24452) 
s_mul_i32 s1, 7, s0 // (op 24452) 
s_add_u32 s1, s1, 64 // (op 24452) 

v_add_u32 v0, v20, s1 // (op 24452) 


v_accvgpr_read v14, a60 // (op 24466) 
v_accvgpr_read v15, a61 // (op 24466) 
v_accvgpr_read v16, a62 // (op 24466) 
v_accvgpr_read v17, a63 // (op 24466) 
v_mul_f32 v10, v19, v14 // (op 24466) 
v_mul_f32 v11, v19, v15 // (op 24466) 
v_mul_f32 v12, v19, v16 // (op 24466) 
v_mul_f32 v13, v19, v17 // (op 24466) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24452) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24449) 
v_lshrrev_b32 v7, 16, v8 // (op 24449) 
v_lshlrev_b32 v6, 16, v6 // (op 24449) 
v_lshlrev_b32 v7, 16, v7 // (op 24449) 
v_mul_f32 v14, v18, v6 // (op 24449) 
v_mul_f32 v15, v18, v7 // (op 24449) 
v_and_b32 v6, 65535, v9 // (op 24449) 
v_lshrrev_b32 v7, 16, v9 // (op 24449) 
v_lshlrev_b32 v6, 16, v6 // (op 24449) 
v_lshlrev_b32 v7, 16, v7 // (op 24449) 
v_mul_f32 v16, v18, v6 // (op 24449) 
v_mul_f32 v17, v18, v7 // (op 24449) 


v_add_f32 v6, v14, v10 // (op 24450) 
v_add_f32 v7, v15, v11 // (op 24450) 
v_add_f32 v8, v16, v12 // (op 24450) 
v_add_f32 v9, v17, v13 // (op 24450) 


v_lshrrev_b32 v10, 16, v6 // (op 24451) 
v_lshrrev_b32 v11, 16, v7 // (op 24451) 
v_lshrrev_b32 v12, 16, v8 // (op 24451) 
v_lshrrev_b32 v13, 16, v9 // (op 24451) 


v_pack_b32_f16 v8, v10, v11 // (op 24414) 
v_pack_b32_f16 v9, v12, v13 // (op 24414) 

s_lshl_b32 s0, s4, 6 // (op 24414) 
s_mul_i32 s1, 7, s0 // (op 24414) 
s_add_u32 s1, s1, 64 // (op 24414) 

v_add_u32 v0, v21, s1 // (op 24414) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24414) Store value





v_add_u32 v0, 256, v20 // (op 24930) 

v_accvgpr_read v14, a128 // (op 24909) 
v_accvgpr_read v15, a129 // (op 24909) 
v_accvgpr_read v16, a130 // (op 24909) 
v_accvgpr_read v17, a131 // (op 24909) 
v_mul_f32 v10, v19, v14 // (op 24909) 
v_mul_f32 v11, v19, v15 // (op 24909) 
v_mul_f32 v12, v19, v16 // (op 24909) 
v_mul_f32 v13, v19, v17 // (op 24909) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24930) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24929) 
v_lshrrev_b32 v7, 16, v8 // (op 24929) 
v_lshlrev_b32 v6, 16, v6 // (op 24929) 
v_lshlrev_b32 v7, 16, v7 // (op 24929) 
v_mul_f32 v14, v18, v6 // (op 24929) 
v_mul_f32 v15, v18, v7 // (op 24929) 
v_and_b32 v6, 65535, v9 // (op 24929) 
v_lshrrev_b32 v7, 16, v9 // (op 24929) 
v_lshlrev_b32 v6, 16, v6 // (op 24929) 
v_lshlrev_b32 v7, 16, v7 // (op 24929) 
v_mul_f32 v16, v18, v6 // (op 24929) 
v_mul_f32 v17, v18, v7 // (op 24929) 


v_add_f32 v6, v14, v10 // (op 24910) 
v_add_f32 v7, v15, v11 // (op 24910) 
v_add_f32 v8, v16, v12 // (op 24910) 
v_add_f32 v9, v17, v13 // (op 24910) 


v_lshrrev_b32 v10, 16, v6 // (op 24911) 
v_lshrrev_b32 v11, 16, v7 // (op 24911) 
v_lshrrev_b32 v12, 16, v8 // (op 24911) 
v_lshrrev_b32 v13, 16, v9 // (op 24911) 


v_pack_b32_f16 v8, v10, v11 // (op 24913) 
v_pack_b32_f16 v9, v12, v13 // (op 24913) 

v_add_u32 v0, 256, v21 // (op 24913) 

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24913) Store value





s_lshl_b32 s0, s2, 6 // (op 24935) 
s_add_u32 s0, s0, 256 // (op 24935) 
v_add_u32 v0, v20, s0 // (op 24935) 


v_accvgpr_read v14, a132 // (op 24967) 
v_accvgpr_read v15, a133 // (op 24967) 
v_accvgpr_read v16, a134 // (op 24967) 
v_accvgpr_read v17, a135 // (op 24967) 
v_mul_f32 v10, v19, v14 // (op 24967) 
v_mul_f32 v11, v19, v15 // (op 24967) 
v_mul_f32 v12, v19, v16 // (op 24967) 
v_mul_f32 v13, v19, v17 // (op 24967) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24935) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24932) 
v_lshrrev_b32 v7, 16, v8 // (op 24932) 
v_lshlrev_b32 v6, 16, v6 // (op 24932) 
v_lshlrev_b32 v7, 16, v7 // (op 24932) 
v_mul_f32 v14, v18, v6 // (op 24932) 
v_mul_f32 v15, v18, v7 // (op 24932) 
v_and_b32 v6, 65535, v9 // (op 24932) 
v_lshrrev_b32 v7, 16, v9 // (op 24932) 
v_lshlrev_b32 v6, 16, v6 // (op 24932) 
v_lshlrev_b32 v7, 16, v7 // (op 24932) 
v_mul_f32 v16, v18, v6 // (op 24932) 
v_mul_f32 v17, v18, v7 // (op 24932) 


v_add_f32 v6, v14, v10 // (op 24933) 
v_add_f32 v7, v15, v11 // (op 24933) 
v_add_f32 v8, v16, v12 // (op 24933) 
v_add_f32 v9, v17, v13 // (op 24933) 


v_lshrrev_b32 v10, 16, v6 // (op 24934) 
v_lshrrev_b32 v11, 16, v7 // (op 24934) 
v_lshrrev_b32 v12, 16, v8 // (op 24934) 
v_lshrrev_b32 v13, 16, v9 // (op 24934) 


v_pack_b32_f16 v8, v10, v11 // (op 24915) 
v_pack_b32_f16 v9, v12, v13 // (op 24915) 

s_lshl_b32 s0, s4, 6 // (op 24915) 
s_add_u32 s0, s0, 256 // (op 24915) 
v_add_u32 v0, v21, s0 // (op 24915) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24915) Store value





s_lshl_b32 s0, s2, 7 // (op 24940) 
s_add_u32 s0, s0, 256 // (op 24940) 
v_add_u32 v0, v20, s0 // (op 24940) 


v_accvgpr_read v14, a136 // (op 24969) 
v_accvgpr_read v15, a137 // (op 24969) 
v_accvgpr_read v16, a138 // (op 24969) 
v_accvgpr_read v17, a139 // (op 24969) 
v_mul_f32 v10, v19, v14 // (op 24969) 
v_mul_f32 v11, v19, v15 // (op 24969) 
v_mul_f32 v12, v19, v16 // (op 24969) 
v_mul_f32 v13, v19, v17 // (op 24969) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24940) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24937) 
v_lshrrev_b32 v7, 16, v8 // (op 24937) 
v_lshlrev_b32 v6, 16, v6 // (op 24937) 
v_lshlrev_b32 v7, 16, v7 // (op 24937) 
v_mul_f32 v14, v18, v6 // (op 24937) 
v_mul_f32 v15, v18, v7 // (op 24937) 
v_and_b32 v6, 65535, v9 // (op 24937) 
v_lshrrev_b32 v7, 16, v9 // (op 24937) 
v_lshlrev_b32 v6, 16, v6 // (op 24937) 
v_lshlrev_b32 v7, 16, v7 // (op 24937) 
v_mul_f32 v16, v18, v6 // (op 24937) 
v_mul_f32 v17, v18, v7 // (op 24937) 


v_add_f32 v6, v14, v10 // (op 24938) 
v_add_f32 v7, v15, v11 // (op 24938) 
v_add_f32 v8, v16, v12 // (op 24938) 
v_add_f32 v9, v17, v13 // (op 24938) 


v_lshrrev_b32 v10, 16, v6 // (op 24939) 
v_lshrrev_b32 v11, 16, v7 // (op 24939) 
v_lshrrev_b32 v12, 16, v8 // (op 24939) 
v_lshrrev_b32 v13, 16, v9 // (op 24939) 


v_pack_b32_f16 v8, v10, v11 // (op 24917) 
v_pack_b32_f16 v9, v12, v13 // (op 24917) 

s_lshl_b32 s0, s4, 7 // (op 24917) 
s_add_u32 s0, s0, 256 // (op 24917) 
v_add_u32 v0, v21, s0 // (op 24917) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24917) Store value





s_lshl_b32 s0, s2, 6 // (op 24945) 
s_mul_i32 s1, 3, s0 // (op 24945) 
s_add_u32 s1, s1, 256 // (op 24945) 

v_add_u32 v0, v20, s1 // (op 24945) 


v_accvgpr_read v14, a140 // (op 24971) 
v_accvgpr_read v15, a141 // (op 24971) 
v_accvgpr_read v16, a142 // (op 24971) 
v_accvgpr_read v17, a143 // (op 24971) 
v_mul_f32 v10, v19, v14 // (op 24971) 
v_mul_f32 v11, v19, v15 // (op 24971) 
v_mul_f32 v12, v19, v16 // (op 24971) 
v_mul_f32 v13, v19, v17 // (op 24971) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24945) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24942) 
v_lshrrev_b32 v7, 16, v8 // (op 24942) 
v_lshlrev_b32 v6, 16, v6 // (op 24942) 
v_lshlrev_b32 v7, 16, v7 // (op 24942) 
v_mul_f32 v14, v18, v6 // (op 24942) 
v_mul_f32 v15, v18, v7 // (op 24942) 
v_and_b32 v6, 65535, v9 // (op 24942) 
v_lshrrev_b32 v7, 16, v9 // (op 24942) 
v_lshlrev_b32 v6, 16, v6 // (op 24942) 
v_lshlrev_b32 v7, 16, v7 // (op 24942) 
v_mul_f32 v16, v18, v6 // (op 24942) 
v_mul_f32 v17, v18, v7 // (op 24942) 


v_add_f32 v6, v14, v10 // (op 24943) 
v_add_f32 v7, v15, v11 // (op 24943) 
v_add_f32 v8, v16, v12 // (op 24943) 
v_add_f32 v9, v17, v13 // (op 24943) 


v_lshrrev_b32 v10, 16, v6 // (op 24944) 
v_lshrrev_b32 v11, 16, v7 // (op 24944) 
v_lshrrev_b32 v12, 16, v8 // (op 24944) 
v_lshrrev_b32 v13, 16, v9 // (op 24944) 


v_pack_b32_f16 v8, v10, v11 // (op 24919) 
v_pack_b32_f16 v9, v12, v13 // (op 24919) 

s_lshl_b32 s0, s4, 6 // (op 24919) 
s_mul_i32 s1, 3, s0 // (op 24919) 
s_add_u32 s1, s1, 256 // (op 24919) 

v_add_u32 v0, v21, s1 // (op 24919) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24919) Store value





s_lshl_b32 s0, s2, 8 // (op 24950) 
s_add_u32 s0, s0, 256 // (op 24950) 
v_add_u32 v0, v20, s0 // (op 24950) 


v_accvgpr_read v14, a144 // (op 24973) 
v_accvgpr_read v15, a145 // (op 24973) 
v_accvgpr_read v16, a146 // (op 24973) 
v_accvgpr_read v17, a147 // (op 24973) 
v_mul_f32 v10, v19, v14 // (op 24973) 
v_mul_f32 v11, v19, v15 // (op 24973) 
v_mul_f32 v12, v19, v16 // (op 24973) 
v_mul_f32 v13, v19, v17 // (op 24973) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24950) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24947) 
v_lshrrev_b32 v7, 16, v8 // (op 24947) 
v_lshlrev_b32 v6, 16, v6 // (op 24947) 
v_lshlrev_b32 v7, 16, v7 // (op 24947) 
v_mul_f32 v14, v18, v6 // (op 24947) 
v_mul_f32 v15, v18, v7 // (op 24947) 
v_and_b32 v6, 65535, v9 // (op 24947) 
v_lshrrev_b32 v7, 16, v9 // (op 24947) 
v_lshlrev_b32 v6, 16, v6 // (op 24947) 
v_lshlrev_b32 v7, 16, v7 // (op 24947) 
v_mul_f32 v16, v18, v6 // (op 24947) 
v_mul_f32 v17, v18, v7 // (op 24947) 


v_add_f32 v6, v14, v10 // (op 24948) 
v_add_f32 v7, v15, v11 // (op 24948) 
v_add_f32 v8, v16, v12 // (op 24948) 
v_add_f32 v9, v17, v13 // (op 24948) 


v_lshrrev_b32 v10, 16, v6 // (op 24949) 
v_lshrrev_b32 v11, 16, v7 // (op 24949) 
v_lshrrev_b32 v12, 16, v8 // (op 24949) 
v_lshrrev_b32 v13, 16, v9 // (op 24949) 


v_pack_b32_f16 v8, v10, v11 // (op 24921) 
v_pack_b32_f16 v9, v12, v13 // (op 24921) 

s_lshl_b32 s0, s4, 8 // (op 24921) 
s_add_u32 s0, s0, 256 // (op 24921) 
v_add_u32 v0, v21, s0 // (op 24921) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24921) Store value





s_lshl_b32 s0, s2, 6 // (op 24955) 
s_mul_i32 s1, 5, s0 // (op 24955) 
s_add_u32 s1, s1, 256 // (op 24955) 

v_add_u32 v0, v20, s1 // (op 24955) 


v_accvgpr_read v14, a148 // (op 24975) 
v_accvgpr_read v15, a149 // (op 24975) 
v_accvgpr_read v16, a150 // (op 24975) 
v_accvgpr_read v17, a151 // (op 24975) 
v_mul_f32 v10, v19, v14 // (op 24975) 
v_mul_f32 v11, v19, v15 // (op 24975) 
v_mul_f32 v12, v19, v16 // (op 24975) 
v_mul_f32 v13, v19, v17 // (op 24975) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24955) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24952) 
v_lshrrev_b32 v7, 16, v8 // (op 24952) 
v_lshlrev_b32 v6, 16, v6 // (op 24952) 
v_lshlrev_b32 v7, 16, v7 // (op 24952) 
v_mul_f32 v14, v18, v6 // (op 24952) 
v_mul_f32 v15, v18, v7 // (op 24952) 
v_and_b32 v6, 65535, v9 // (op 24952) 
v_lshrrev_b32 v7, 16, v9 // (op 24952) 
v_lshlrev_b32 v6, 16, v6 // (op 24952) 
v_lshlrev_b32 v7, 16, v7 // (op 24952) 
v_mul_f32 v16, v18, v6 // (op 24952) 
v_mul_f32 v17, v18, v7 // (op 24952) 


v_add_f32 v6, v14, v10 // (op 24953) 
v_add_f32 v7, v15, v11 // (op 24953) 
v_add_f32 v8, v16, v12 // (op 24953) 
v_add_f32 v9, v17, v13 // (op 24953) 


v_lshrrev_b32 v10, 16, v6 // (op 24954) 
v_lshrrev_b32 v11, 16, v7 // (op 24954) 
v_lshrrev_b32 v12, 16, v8 // (op 24954) 
v_lshrrev_b32 v13, 16, v9 // (op 24954) 


v_pack_b32_f16 v8, v10, v11 // (op 24923) 
v_pack_b32_f16 v9, v12, v13 // (op 24923) 

s_lshl_b32 s0, s4, 6 // (op 24923) 
s_mul_i32 s1, 5, s0 // (op 24923) 
s_add_u32 s1, s1, 256 // (op 24923) 

v_add_u32 v0, v21, s1 // (op 24923) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24923) Store value





s_lshl_b32 s0, s2, 6 // (op 24960) 
s_mul_i32 s1, 6, s0 // (op 24960) 
s_add_u32 s1, s1, 256 // (op 24960) 

v_add_u32 v0, v20, s1 // (op 24960) 


v_accvgpr_read v14, a152 // (op 24977) 
v_accvgpr_read v15, a153 // (op 24977) 
v_accvgpr_read v16, a154 // (op 24977) 
v_accvgpr_read v17, a155 // (op 24977) 
v_mul_f32 v10, v19, v14 // (op 24977) 
v_mul_f32 v11, v19, v15 // (op 24977) 
v_mul_f32 v12, v19, v16 // (op 24977) 
v_mul_f32 v13, v19, v17 // (op 24977) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24960) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24957) 
v_lshrrev_b32 v7, 16, v8 // (op 24957) 
v_lshlrev_b32 v6, 16, v6 // (op 24957) 
v_lshlrev_b32 v7, 16, v7 // (op 24957) 
v_mul_f32 v14, v18, v6 // (op 24957) 
v_mul_f32 v15, v18, v7 // (op 24957) 
v_and_b32 v6, 65535, v9 // (op 24957) 
v_lshrrev_b32 v7, 16, v9 // (op 24957) 
v_lshlrev_b32 v6, 16, v6 // (op 24957) 
v_lshlrev_b32 v7, 16, v7 // (op 24957) 
v_mul_f32 v16, v18, v6 // (op 24957) 
v_mul_f32 v17, v18, v7 // (op 24957) 


v_add_f32 v6, v14, v10 // (op 24958) 
v_add_f32 v7, v15, v11 // (op 24958) 
v_add_f32 v8, v16, v12 // (op 24958) 
v_add_f32 v9, v17, v13 // (op 24958) 


v_lshrrev_b32 v10, 16, v6 // (op 24959) 
v_lshrrev_b32 v11, 16, v7 // (op 24959) 
v_lshrrev_b32 v12, 16, v8 // (op 24959) 
v_lshrrev_b32 v13, 16, v9 // (op 24959) 


v_pack_b32_f16 v8, v10, v11 // (op 24925) 
v_pack_b32_f16 v9, v12, v13 // (op 24925) 

s_lshl_b32 s0, s4, 6 // (op 24925) 
s_mul_i32 s1, 6, s0 // (op 24925) 
s_add_u32 s1, s1, 256 // (op 24925) 

v_add_u32 v0, v21, s1 // (op 24925) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24925) Store value





s_lshl_b32 s0, s2, 6 // (op 24965) 
s_mul_i32 s1, 7, s0 // (op 24965) 
s_add_u32 s1, s1, 256 // (op 24965) 

v_add_u32 v0, v20, s1 // (op 24965) 


v_accvgpr_read v14, a156 // (op 24979) 
v_accvgpr_read v15, a157 // (op 24979) 
v_accvgpr_read v16, a158 // (op 24979) 
v_accvgpr_read v17, a159 // (op 24979) 
v_mul_f32 v10, v19, v14 // (op 24979) 
v_mul_f32 v11, v19, v15 // (op 24979) 
v_mul_f32 v12, v19, v16 // (op 24979) 
v_mul_f32 v13, v19, v17 // (op 24979) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24965) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24962) 
v_lshrrev_b32 v7, 16, v8 // (op 24962) 
v_lshlrev_b32 v6, 16, v6 // (op 24962) 
v_lshlrev_b32 v7, 16, v7 // (op 24962) 
v_mul_f32 v14, v18, v6 // (op 24962) 
v_mul_f32 v15, v18, v7 // (op 24962) 
v_and_b32 v6, 65535, v9 // (op 24962) 
v_lshrrev_b32 v7, 16, v9 // (op 24962) 
v_lshlrev_b32 v6, 16, v6 // (op 24962) 
v_lshlrev_b32 v7, 16, v7 // (op 24962) 
v_mul_f32 v16, v18, v6 // (op 24962) 
v_mul_f32 v17, v18, v7 // (op 24962) 


v_add_f32 v6, v14, v10 // (op 24963) 
v_add_f32 v7, v15, v11 // (op 24963) 
v_add_f32 v8, v16, v12 // (op 24963) 
v_add_f32 v9, v17, v13 // (op 24963) 


v_lshrrev_b32 v10, 16, v6 // (op 24964) 
v_lshrrev_b32 v11, 16, v7 // (op 24964) 
v_lshrrev_b32 v12, 16, v8 // (op 24964) 
v_lshrrev_b32 v13, 16, v9 // (op 24964) 


v_pack_b32_f16 v8, v10, v11 // (op 24927) 
v_pack_b32_f16 v9, v12, v13 // (op 24927) 

s_lshl_b32 s0, s4, 6 // (op 24927) 
s_mul_i32 s1, 7, s0 // (op 24927) 
s_add_u32 s1, s1, 256 // (op 24927) 

v_add_u32 v0, v21, s1 // (op 24927) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24927) Store value





v_mov_b32 v0, v20 // (op 148) reg expression

v_accvgpr_read v14, a0 // (op 144) 
v_accvgpr_read v15, a1 // (op 144) 
v_accvgpr_read v16, a2 // (op 144) 
v_accvgpr_read v17, a3 // (op 144) 
v_mul_f32 v10, v19, v14 // (op 144) 
v_mul_f32 v11, v19, v15 // (op 144) 
v_mul_f32 v12, v19, v16 // (op 144) 
v_mul_f32 v13, v19, v17 // (op 144) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 148) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 149) 
v_lshrrev_b32 v7, 16, v8 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v14, v18, v6 // (op 149) 
v_mul_f32 v15, v18, v7 // (op 149) 
v_and_b32 v6, 65535, v9 // (op 149) 
v_lshrrev_b32 v7, 16, v9 // (op 149) 
v_lshlrev_b32 v6, 16, v6 // (op 149) 
v_lshlrev_b32 v7, 16, v7 // (op 149) 
v_mul_f32 v16, v18, v6 // (op 149) 
v_mul_f32 v17, v18, v7 // (op 149) 


v_add_f32 v6, v14, v10 // (op 145) 
v_add_f32 v7, v15, v11 // (op 145) 
v_add_f32 v8, v16, v12 // (op 145) 
v_add_f32 v9, v17, v13 // (op 145) 


v_lshrrev_b32 v10, 16, v6 // (op 146) 
v_lshrrev_b32 v11, 16, v7 // (op 146) 
v_lshrrev_b32 v12, 16, v8 // (op 146) 
v_lshrrev_b32 v13, 16, v9 // (op 146) 


v_pack_b32_f16 v8, v10, v11 // (op 147) 
v_pack_b32_f16 v9, v12, v13 // (op 147) 

v_mov_b32 v0, v21 // (op 147) reg expression

buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 24221) 
v_add_u32 v0, v20, s0 // (op 24221) 


v_accvgpr_read v14, a4 // (op 24217) 
v_accvgpr_read v15, a5 // (op 24217) 
v_accvgpr_read v16, a6 // (op 24217) 
v_accvgpr_read v17, a7 // (op 24217) 
v_mul_f32 v10, v19, v14 // (op 24217) 
v_mul_f32 v11, v19, v15 // (op 24217) 
v_mul_f32 v12, v19, v16 // (op 24217) 
v_mul_f32 v13, v19, v17 // (op 24217) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24221) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24222) 
v_lshrrev_b32 v7, 16, v8 // (op 24222) 
v_lshlrev_b32 v6, 16, v6 // (op 24222) 
v_lshlrev_b32 v7, 16, v7 // (op 24222) 
v_mul_f32 v14, v18, v6 // (op 24222) 
v_mul_f32 v15, v18, v7 // (op 24222) 
v_and_b32 v6, 65535, v9 // (op 24222) 
v_lshrrev_b32 v7, 16, v9 // (op 24222) 
v_lshlrev_b32 v6, 16, v6 // (op 24222) 
v_lshlrev_b32 v7, 16, v7 // (op 24222) 
v_mul_f32 v16, v18, v6 // (op 24222) 
v_mul_f32 v17, v18, v7 // (op 24222) 


v_add_f32 v6, v14, v10 // (op 24218) 
v_add_f32 v7, v15, v11 // (op 24218) 
v_add_f32 v8, v16, v12 // (op 24218) 
v_add_f32 v9, v17, v13 // (op 24218) 


v_lshrrev_b32 v10, 16, v6 // (op 24219) 
v_lshrrev_b32 v11, 16, v7 // (op 24219) 
v_lshrrev_b32 v12, 16, v8 // (op 24219) 
v_lshrrev_b32 v13, 16, v9 // (op 24219) 


v_pack_b32_f16 v8, v10, v11 // (op 24220) 
v_pack_b32_f16 v9, v12, v13 // (op 24220) 

s_lshl_b32 s0, s4, 6 // (op 24220) 
v_add_u32 v0, v21, s0 // (op 24220) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24220) Store value





s_lshl_b32 s0, s2, 7 // (op 24235) 
v_add_u32 v0, v20, s0 // (op 24235) 


v_accvgpr_read v14, a8 // (op 24231) 
v_accvgpr_read v15, a9 // (op 24231) 
v_accvgpr_read v16, a10 // (op 24231) 
v_accvgpr_read v17, a11 // (op 24231) 
v_mul_f32 v10, v19, v14 // (op 24231) 
v_mul_f32 v11, v19, v15 // (op 24231) 
v_mul_f32 v12, v19, v16 // (op 24231) 
v_mul_f32 v13, v19, v17 // (op 24231) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24235) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24236) 
v_lshrrev_b32 v7, 16, v8 // (op 24236) 
v_lshlrev_b32 v6, 16, v6 // (op 24236) 
v_lshlrev_b32 v7, 16, v7 // (op 24236) 
v_mul_f32 v14, v18, v6 // (op 24236) 
v_mul_f32 v15, v18, v7 // (op 24236) 
v_and_b32 v6, 65535, v9 // (op 24236) 
v_lshrrev_b32 v7, 16, v9 // (op 24236) 
v_lshlrev_b32 v6, 16, v6 // (op 24236) 
v_lshlrev_b32 v7, 16, v7 // (op 24236) 
v_mul_f32 v16, v18, v6 // (op 24236) 
v_mul_f32 v17, v18, v7 // (op 24236) 


v_add_f32 v6, v14, v10 // (op 24232) 
v_add_f32 v7, v15, v11 // (op 24232) 
v_add_f32 v8, v16, v12 // (op 24232) 
v_add_f32 v9, v17, v13 // (op 24232) 


v_lshrrev_b32 v10, 16, v6 // (op 24233) 
v_lshrrev_b32 v11, 16, v7 // (op 24233) 
v_lshrrev_b32 v12, 16, v8 // (op 24233) 
v_lshrrev_b32 v13, 16, v9 // (op 24233) 


v_pack_b32_f16 v8, v10, v11 // (op 24234) 
v_pack_b32_f16 v9, v12, v13 // (op 24234) 

s_lshl_b32 s0, s4, 7 // (op 24234) 
v_add_u32 v0, v21, s0 // (op 24234) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24234) Store value





s_lshl_b32 s0, s2, 6 // (op 24249) 
s_mul_i32 s1, 3, s0 // (op 24249) 

v_add_u32 v0, v20, s1 // (op 24249) 


v_accvgpr_read v14, a12 // (op 24245) 
v_accvgpr_read v15, a13 // (op 24245) 
v_accvgpr_read v16, a14 // (op 24245) 
v_accvgpr_read v17, a15 // (op 24245) 
v_mul_f32 v10, v19, v14 // (op 24245) 
v_mul_f32 v11, v19, v15 // (op 24245) 
v_mul_f32 v12, v19, v16 // (op 24245) 
v_mul_f32 v13, v19, v17 // (op 24245) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24249) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24250) 
v_lshrrev_b32 v7, 16, v8 // (op 24250) 
v_lshlrev_b32 v6, 16, v6 // (op 24250) 
v_lshlrev_b32 v7, 16, v7 // (op 24250) 
v_mul_f32 v14, v18, v6 // (op 24250) 
v_mul_f32 v15, v18, v7 // (op 24250) 
v_and_b32 v6, 65535, v9 // (op 24250) 
v_lshrrev_b32 v7, 16, v9 // (op 24250) 
v_lshlrev_b32 v6, 16, v6 // (op 24250) 
v_lshlrev_b32 v7, 16, v7 // (op 24250) 
v_mul_f32 v16, v18, v6 // (op 24250) 
v_mul_f32 v17, v18, v7 // (op 24250) 


v_add_f32 v6, v14, v10 // (op 24246) 
v_add_f32 v7, v15, v11 // (op 24246) 
v_add_f32 v8, v16, v12 // (op 24246) 
v_add_f32 v9, v17, v13 // (op 24246) 


v_lshrrev_b32 v10, 16, v6 // (op 24247) 
v_lshrrev_b32 v11, 16, v7 // (op 24247) 
v_lshrrev_b32 v12, 16, v8 // (op 24247) 
v_lshrrev_b32 v13, 16, v9 // (op 24247) 


v_pack_b32_f16 v8, v10, v11 // (op 24248) 
v_pack_b32_f16 v9, v12, v13 // (op 24248) 

s_lshl_b32 s0, s4, 6 // (op 24248) 
s_mul_i32 s1, 3, s0 // (op 24248) 

v_add_u32 v0, v21, s1 // (op 24248) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24248) Store value





s_lshl_b32 s0, s2, 8 // (op 24263) 
v_add_u32 v0, v20, s0 // (op 24263) 


v_accvgpr_read v14, a16 // (op 24259) 
v_accvgpr_read v15, a17 // (op 24259) 
v_accvgpr_read v16, a18 // (op 24259) 
v_accvgpr_read v17, a19 // (op 24259) 
v_mul_f32 v10, v19, v14 // (op 24259) 
v_mul_f32 v11, v19, v15 // (op 24259) 
v_mul_f32 v12, v19, v16 // (op 24259) 
v_mul_f32 v13, v19, v17 // (op 24259) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24263) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24264) 
v_lshrrev_b32 v7, 16, v8 // (op 24264) 
v_lshlrev_b32 v6, 16, v6 // (op 24264) 
v_lshlrev_b32 v7, 16, v7 // (op 24264) 
v_mul_f32 v14, v18, v6 // (op 24264) 
v_mul_f32 v15, v18, v7 // (op 24264) 
v_and_b32 v6, 65535, v9 // (op 24264) 
v_lshrrev_b32 v7, 16, v9 // (op 24264) 
v_lshlrev_b32 v6, 16, v6 // (op 24264) 
v_lshlrev_b32 v7, 16, v7 // (op 24264) 
v_mul_f32 v16, v18, v6 // (op 24264) 
v_mul_f32 v17, v18, v7 // (op 24264) 


v_add_f32 v6, v14, v10 // (op 24260) 
v_add_f32 v7, v15, v11 // (op 24260) 
v_add_f32 v8, v16, v12 // (op 24260) 
v_add_f32 v9, v17, v13 // (op 24260) 


v_lshrrev_b32 v10, 16, v6 // (op 24261) 
v_lshrrev_b32 v11, 16, v7 // (op 24261) 
v_lshrrev_b32 v12, 16, v8 // (op 24261) 
v_lshrrev_b32 v13, 16, v9 // (op 24261) 


v_pack_b32_f16 v8, v10, v11 // (op 24262) 
v_pack_b32_f16 v9, v12, v13 // (op 24262) 

s_lshl_b32 s0, s4, 8 // (op 24262) 
v_add_u32 v0, v21, s0 // (op 24262) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24262) Store value





s_lshl_b32 s0, s2, 6 // (op 24277) 
s_mul_i32 s1, 5, s0 // (op 24277) 

v_add_u32 v0, v20, s1 // (op 24277) 


v_accvgpr_read v14, a20 // (op 24273) 
v_accvgpr_read v15, a21 // (op 24273) 
v_accvgpr_read v16, a22 // (op 24273) 
v_accvgpr_read v17, a23 // (op 24273) 
v_mul_f32 v10, v19, v14 // (op 24273) 
v_mul_f32 v11, v19, v15 // (op 24273) 
v_mul_f32 v12, v19, v16 // (op 24273) 
v_mul_f32 v13, v19, v17 // (op 24273) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24277) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24278) 
v_lshrrev_b32 v7, 16, v8 // (op 24278) 
v_lshlrev_b32 v6, 16, v6 // (op 24278) 
v_lshlrev_b32 v7, 16, v7 // (op 24278) 
v_mul_f32 v14, v18, v6 // (op 24278) 
v_mul_f32 v15, v18, v7 // (op 24278) 
v_and_b32 v6, 65535, v9 // (op 24278) 
v_lshrrev_b32 v7, 16, v9 // (op 24278) 
v_lshlrev_b32 v6, 16, v6 // (op 24278) 
v_lshlrev_b32 v7, 16, v7 // (op 24278) 
v_mul_f32 v16, v18, v6 // (op 24278) 
v_mul_f32 v17, v18, v7 // (op 24278) 


v_add_f32 v6, v14, v10 // (op 24274) 
v_add_f32 v7, v15, v11 // (op 24274) 
v_add_f32 v8, v16, v12 // (op 24274) 
v_add_f32 v9, v17, v13 // (op 24274) 


v_lshrrev_b32 v10, 16, v6 // (op 24275) 
v_lshrrev_b32 v11, 16, v7 // (op 24275) 
v_lshrrev_b32 v12, 16, v8 // (op 24275) 
v_lshrrev_b32 v13, 16, v9 // (op 24275) 


v_pack_b32_f16 v8, v10, v11 // (op 24276) 
v_pack_b32_f16 v9, v12, v13 // (op 24276) 

s_lshl_b32 s0, s4, 6 // (op 24276) 
s_mul_i32 s1, 5, s0 // (op 24276) 

v_add_u32 v0, v21, s1 // (op 24276) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24276) Store value





s_lshl_b32 s0, s2, 6 // (op 24291) 
s_mul_i32 s1, 6, s0 // (op 24291) 

v_add_u32 v0, v20, s1 // (op 24291) 


v_accvgpr_read v14, a24 // (op 24287) 
v_accvgpr_read v15, a25 // (op 24287) 
v_accvgpr_read v16, a26 // (op 24287) 
v_accvgpr_read v17, a27 // (op 24287) 
v_mul_f32 v10, v19, v14 // (op 24287) 
v_mul_f32 v11, v19, v15 // (op 24287) 
v_mul_f32 v12, v19, v16 // (op 24287) 
v_mul_f32 v13, v19, v17 // (op 24287) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24291) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24292) 
v_lshrrev_b32 v7, 16, v8 // (op 24292) 
v_lshlrev_b32 v6, 16, v6 // (op 24292) 
v_lshlrev_b32 v7, 16, v7 // (op 24292) 
v_mul_f32 v14, v18, v6 // (op 24292) 
v_mul_f32 v15, v18, v7 // (op 24292) 
v_and_b32 v6, 65535, v9 // (op 24292) 
v_lshrrev_b32 v7, 16, v9 // (op 24292) 
v_lshlrev_b32 v6, 16, v6 // (op 24292) 
v_lshlrev_b32 v7, 16, v7 // (op 24292) 
v_mul_f32 v16, v18, v6 // (op 24292) 
v_mul_f32 v17, v18, v7 // (op 24292) 


v_add_f32 v6, v14, v10 // (op 24288) 
v_add_f32 v7, v15, v11 // (op 24288) 
v_add_f32 v8, v16, v12 // (op 24288) 
v_add_f32 v9, v17, v13 // (op 24288) 


v_lshrrev_b32 v10, 16, v6 // (op 24289) 
v_lshrrev_b32 v11, 16, v7 // (op 24289) 
v_lshrrev_b32 v12, 16, v8 // (op 24289) 
v_lshrrev_b32 v13, 16, v9 // (op 24289) 


v_pack_b32_f16 v8, v10, v11 // (op 24290) 
v_pack_b32_f16 v9, v12, v13 // (op 24290) 

s_lshl_b32 s0, s4, 6 // (op 24290) 
s_mul_i32 s1, 6, s0 // (op 24290) 

v_add_u32 v0, v21, s1 // (op 24290) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24290) Store value





s_lshl_b32 s0, s2, 6 // (op 24305) 
s_mul_i32 s1, 7, s0 // (op 24305) 

v_add_u32 v0, v20, s1 // (op 24305) 


v_accvgpr_read v14, a28 // (op 24301) 
v_accvgpr_read v15, a29 // (op 24301) 
v_accvgpr_read v16, a30 // (op 24301) 
v_accvgpr_read v17, a31 // (op 24301) 
v_mul_f32 v10, v19, v14 // (op 24301) 
v_mul_f32 v11, v19, v15 // (op 24301) 
v_mul_f32 v12, v19, v16 // (op 24301) 
v_mul_f32 v13, v19, v17 // (op 24301) 

buffer_load_dwordx2 v[8:9], v0, s[40:43], 0 offen // (op 24305) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v8 // (op 24306) 
v_lshrrev_b32 v7, 16, v8 // (op 24306) 
v_lshlrev_b32 v6, 16, v6 // (op 24306) 
v_lshlrev_b32 v7, 16, v7 // (op 24306) 
v_mul_f32 v14, v18, v6 // (op 24306) 
v_mul_f32 v15, v18, v7 // (op 24306) 
v_and_b32 v6, 65535, v9 // (op 24306) 
v_lshrrev_b32 v7, 16, v9 // (op 24306) 
v_lshlrev_b32 v6, 16, v6 // (op 24306) 
v_lshlrev_b32 v7, 16, v7 // (op 24306) 
v_mul_f32 v16, v18, v6 // (op 24306) 
v_mul_f32 v17, v18, v7 // (op 24306) 


v_add_f32 v6, v14, v10 // (op 24302) 
v_add_f32 v7, v15, v11 // (op 24302) 
v_add_f32 v8, v16, v12 // (op 24302) 
v_add_f32 v9, v17, v13 // (op 24302) 


v_lshrrev_b32 v10, 16, v6 // (op 24303) 
v_lshrrev_b32 v11, 16, v7 // (op 24303) 
v_lshrrev_b32 v12, 16, v8 // (op 24303) 
v_lshrrev_b32 v13, 16, v9 // (op 24303) 


v_pack_b32_f16 v8, v10, v11 // (op 24304) 
v_pack_b32_f16 v9, v12, v13 // (op 24304) 

s_lshl_b32 s0, s4, 6 // (op 24304) 
s_mul_i32 s1, 7, s0 // (op 24304) 

v_add_u32 v0, v21, s1 // (op 24304) 


buffer_store_dwordx2 v[8:9], v0, s[44:47], 0 offen // (op 24304) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalBottom_0_beta_135:








































































s_endpgm  // End of RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
.LRR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_end:
.size RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4, .LRR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_end-RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
  .amdhsa_next_free_vgpr 432
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 135168 // lds bytes
  .amdhsa_accum_offset 176
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: F4
    DestDataType: b
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
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: address, semantic: AddressC },
            { type: int64, semantic: SizeFree1 },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha },
            { type: address, semantic: AddressA },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: SizeSumScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB } ]
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 4
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
    .symbol: RR_GEMM_NT_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 135168
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 174
    .agpr_count: 256
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
      - .name: Tensor_2_size_1_11
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_1_0
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
      - .name: Tensor_0_size_1_1
        .size: 8
        .offset: 64
        .value_kind: by_value
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_5
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_6
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

