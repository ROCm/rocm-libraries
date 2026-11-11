// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
.p2align 8
.type RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4,@function
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 55062) 
v_and_b32 v3, 48, v1 // (op 55062) 
v_sub_u32 v4, v0, v3 // (op 55062) 

v_lshrrev_b32 v3, 4, v0 // (op 55062) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 55062) 


v_lshrrev_b32 v3, 7, v1 // (op 55062) 
v_lshlrev_b32 v4, 4, v3 // (op 55062) 

v_lshl_add_u32 v3, v0, 8, v4 // (op 55062) 


v_lshlrev_b32 v0, 2, v3 // (op 55062) 

v_lshrrev_b32 v3, 3, v0 // (op 55062) 

v_and_b32 v0, 63, v1 // (op 55119) 
v_and_b32 v4, 48, v1 // (op 55119) 
v_sub_u32 v5, v0, v4 // (op 55119) 

v_lshrrev_b32 v4, 4, v0 // (op 55119) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 55119) 


v_lshrrev_b32 v4, 6, v1 // (op 55119) 
v_and_b32 v5, 1, v4 // (op 55119) 

v_lshlrev_b32 v4, 4, v5 // (op 55119) 

v_lshl_add_u32 v5, v0, 8, v4 // (op 55119) 


v_lshlrev_b32 v0, 2, v5 // (op 55119) 

v_lshrrev_b32 v4, 3, v0 // (op 55119) 

v_lshrrev_b32 v0, 7, v1 // (op 55177) 
v_and_b32 v5, 15, v1 // (op 55177) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 55177) 


v_and_b32 v0, 63, v1 // (op 55177) 
v_lshrrev_b32 v5, 4, v0 // (op 55177) 

v_lshl_add_u32 v0, v5, 8, v6 // (op 55177) 


v_lshrrev_b32 v5, 6, v1 // (op 55235) 
v_and_b32 v6, 1, v5 // (op 55235) 

v_and_b32 v5, 15, v1 // (op 55235) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 55235) 


v_and_b32 v5, 63, v1 // (op 55235) 
v_lshrrev_b32 v6, 4, v5 // (op 55235) 

v_lshl_add_u32 v5, v6, 8, v7 // (op 55235) 


v_and_b32 v6, 63, v1 // (op 55302) 
v_and_b32 v7, 48, v1 // (op 55302) 
v_sub_u32 v8, v6, v7 // (op 55302) 

v_lshrrev_b32 v7, 4, v6 // (op 55302) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 55302) 


v_lshrrev_b32 v7, 7, v1 // (op 55302) 
v_lshlrev_b32 v8, 4, v7 // (op 55302) 

v_lshl_add_u32 v7, v6, 8, v8 // (op 55302) 


v_lshlrev_b32 v6, 2, v7 // (op 55302) 

v_lshrrev_b32 v7, 3, v6 // (op 55302) 

v_and_b32 v6, 63, v1 // (op 55346) 
v_and_b32 v8, 48, v1 // (op 55346) 
v_sub_u32 v9, v6, v8 // (op 55346) 

v_lshrrev_b32 v8, 4, v6 // (op 55346) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 55346) 


v_lshrrev_b32 v8, 6, v1 // (op 55346) 
v_and_b32 v9, 1, v8 // (op 55346) 

v_lshlrev_b32 v8, 4, v9 // (op 55346) 

v_lshl_add_u32 v9, v6, 8, v8 // (op 55346) 


v_lshlrev_b32 v6, 2, v9 // (op 55346) 

v_lshrrev_b32 v8, 3, v6 // (op 55346) 

v_lshrrev_b32 v6, 7, v1 // (op 55375) 
v_and_b32 v9, 15, v1 // (op 55375) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 55375) 


v_and_b32 v6, 63, v1 // (op 55375) 
v_lshrrev_b32 v9, 4, v6 // (op 55375) 

v_lshl_add_u32 v6, v9, 8, v10 // (op 55375) 


v_lshrrev_b32 v9, 6, v1 // (op 55404) 
v_and_b32 v10, 1, v9 // (op 55404) 

v_and_b32 v9, 15, v1 // (op 55404) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 55404) 


v_and_b32 v9, 63, v1 // (op 55404) 
v_lshrrev_b32 v10, 4, v9 // (op 55404) 

v_lshl_add_u32 v9, v10, 8, v11 // (op 55404) 


v_and_b32 v10, 63, v1 // (op 55448) 
v_and_b32 v11, 48, v1 // (op 55448) 
v_sub_u32 v12, v10, v11 // (op 55448) 

v_lshrrev_b32 v11, 4, v10 // (op 55448) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 55448) 


v_lshrrev_b32 v11, 7, v1 // (op 55448) 
v_lshlrev_b32 v12, 4, v11 // (op 55448) 

v_lshl_add_u32 v11, v10, 8, v12 // (op 55448) 


v_lshlrev_b32 v10, 2, v11 // (op 55448) 

v_lshrrev_b32 v11, 3, v10 // (op 55448) 

v_and_b32 v10, 63, v1 // (op 55492) 
v_and_b32 v12, 48, v1 // (op 55492) 
v_sub_u32 v13, v10, v12 // (op 55492) 

v_lshrrev_b32 v12, 4, v10 // (op 55492) 

v_lshl_add_u32 v10, v12, 5, v13 // (op 55492) 


v_lshrrev_b32 v12, 6, v1 // (op 55492) 
v_and_b32 v13, 1, v12 // (op 55492) 

v_lshlrev_b32 v12, 4, v13 // (op 55492) 

v_lshl_add_u32 v13, v10, 8, v12 // (op 55492) 


v_lshlrev_b32 v10, 2, v13 // (op 55492) 

v_lshrrev_b32 v12, 3, v10 // (op 55492) 

v_lshrrev_b32 v10, 7, v1 // (op 55521) 
v_and_b32 v13, 15, v1 // (op 55521) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 55521) 


v_and_b32 v10, 63, v1 // (op 55521) 
v_lshrrev_b32 v13, 4, v10 // (op 55521) 

v_lshl_add_u32 v10, v13, 8, v14 // (op 55521) 


v_lshrrev_b32 v13, 6, v1 // (op 55550) 
v_and_b32 v14, 1, v13 // (op 55550) 

v_and_b32 v13, 15, v1 // (op 55550) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 55550) 


v_and_b32 v13, 63, v1 // (op 55550) 
v_lshrrev_b32 v14, 4, v13 // (op 55550) 

v_lshl_add_u32 v13, v14, 8, v15 // (op 55550) 


v_and_b32 v14, 63, v1 // (op 55594) 
v_and_b32 v15, 48, v1 // (op 55594) 
v_sub_u32 v16, v14, v15 // (op 55594) 

v_lshrrev_b32 v15, 4, v14 // (op 55594) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 55594) 


v_lshrrev_b32 v15, 7, v1 // (op 55594) 
v_lshlrev_b32 v16, 4, v15 // (op 55594) 

v_lshl_add_u32 v15, v14, 8, v16 // (op 55594) 


v_lshlrev_b32 v14, 2, v15 // (op 55594) 

v_lshrrev_b32 v15, 3, v14 // (op 55594) 

v_and_b32 v14, 63, v1 // (op 55638) 
v_and_b32 v16, 48, v1 // (op 55638) 
v_sub_u32 v17, v14, v16 // (op 55638) 

v_lshrrev_b32 v16, 4, v14 // (op 55638) 

v_lshl_add_u32 v14, v16, 5, v17 // (op 55638) 


v_lshrrev_b32 v16, 6, v1 // (op 55638) 
v_and_b32 v17, 1, v16 // (op 55638) 

v_lshlrev_b32 v16, 4, v17 // (op 55638) 

v_lshl_add_u32 v17, v14, 8, v16 // (op 55638) 


v_lshlrev_b32 v14, 2, v17 // (op 55638) 

v_lshrrev_b32 v16, 3, v14 // (op 55638) 

v_lshrrev_b32 v14, 7, v1 // (op 55667) 
v_and_b32 v17, 15, v1 // (op 55667) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 55667) 


v_and_b32 v14, 63, v1 // (op 55667) 
v_lshrrev_b32 v17, 4, v14 // (op 55667) 

v_lshl_add_u32 v14, v17, 8, v18 // (op 55667) 


v_lshrrev_b32 v17, 6, v1 // (op 55696) 
v_and_b32 v18, 1, v17 // (op 55696) 

v_and_b32 v17, 15, v1 // (op 55696) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 55696) 


v_and_b32 v17, 63, v1 // (op 55696) 
v_lshrrev_b32 v18, 4, v17 // (op 55696) 

v_lshl_add_u32 v17, v18, 8, v19 // (op 55696) 


v_lshrrev_b32 v18, 7, v1 // (op 55001) 
v_and_b32 v19, 63, v1 // (op 55001) 
v_lshrrev_b32 v20, 4, v19 // (op 55001) 
v_lshlrev_b32 v21, 2, v20 // (op 55001) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 55001) 


v_lshl_add_u32 v18, s16, 8, v20 // (op 55001) 

v_lshrrev_b32 v20, 2, v19 // (op 55001) 

v_and_b32 v19, 3, v20 // (op 55001) 

v_and_b32 v20, 3, v1 // (op 55001) 
v_lshl_add_u32 v21, v19, 2, v20 // (op 55001) 


v_lshrrev_b32 v19, 6, v1 // (op 55001) 
v_and_b32 v20, 1, v19 // (op 55001) 

v_lshl_add_u32 v19, v20, 4, v21 // (op 55001) 


v_lshl_add_u32 v20, s17, 8, v19 // (op 55001) 

v_mul_lo_u32 v19, v20, s2 // (op 55001) 

v_add_lshl_u32 v20, v18, v19, 2 // (op 55001) 


v_lshrrev_b32 v18, 7, v1 // (op 55256) 
v_and_b32 v19, 63, v1 // (op 55256) 
v_lshrrev_b32 v21, 4, v19 // (op 55256) 
s_mul_i32 s1, s2, s9 // (op 55003) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 55003) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 55003) least significant
s_mul_hi_u32 s19, s2, s8 // (op 55003) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 55003) most significant: sum
s_add_u32 s1, s1, s19 // (op 55003) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 55003) 

v_lshlrev_b32 v22, 2, v21 // (op 55256) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 55256) 


v_lshl_add_u32 v18, s16, 8, v21 // (op 55256) 

v_lshrrev_b32 v21, 2, v19 // (op 55256) 

v_and_b32 v19, 3, v21 // (op 55256) 

v_and_b32 v21, 3, v1 // (op 55256) 
v_lshl_add_u32 v22, v19, 2, v21 // (op 55256) 


v_lshrrev_b32 v19, 6, v1 // (op 55256) 
v_and_b32 v21, 1, v19 // (op 55256) 

v_lshl_add_u32 v19, v21, 4, v22 // (op 55256) 


v_lshl_add_u32 v21, s17, 8, v19 // (op 55256) 

v_mul_lo_u32 v19, v21, s4 // (op 55256) 

v_add_lshl_u32 v21, v18, v19, 2 // (op 55256) 


s_mov_b32 s40, s6 // (op 55003) reg expression
s_mov_b32 s41, s7 // (op 55003) reg expression
s_mov_b32 s42, s18 // (op 55003) reg expression
s_mov_b32 s43, 131072 // (op 55003) call()

s_mul_i32 s1, s4, s9 // (op 55258) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 55258) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 55258) least significant
s_mul_hi_u32 s7, s4, s8 // (op 55258) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 55258) most significant: sum
s_add_u32 s1, s1, s7 // (op 55258) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 55258) 

s_mov_b32 s44, s10 // (op 55258) reg expression
s_mov_b32 s45, s11 // (op 55258) reg expression
s_mov_b32 s46, s6 // (op 55258) reg expression
s_mov_b32 s47, 131072 // (op 55258) call()



v_mov_b32 v18, s14 // (op 2) Move value
v_mov_b32 v19, s15 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 482) call()
v_accvgpr_write a5, 0.00000 // (op 482) call()
v_accvgpr_write a6, 0.00000 // (op 482) call()
v_accvgpr_write a7, 0.00000 // (op 482) call()
v_accvgpr_write a8, 0.00000 // (op 780) call()
v_accvgpr_write a9, 0.00000 // (op 780) call()
v_accvgpr_write a10, 0.00000 // (op 780) call()
v_accvgpr_write a11, 0.00000 // (op 780) call()
v_accvgpr_write a12, 0.00000 // (op 1078) call()
v_accvgpr_write a13, 0.00000 // (op 1078) call()
v_accvgpr_write a14, 0.00000 // (op 1078) call()
v_accvgpr_write a15, 0.00000 // (op 1078) call()
v_accvgpr_write a16, 0.00000 // (op 1376) call()
v_accvgpr_write a17, 0.00000 // (op 1376) call()
v_accvgpr_write a18, 0.00000 // (op 1376) call()
v_accvgpr_write a19, 0.00000 // (op 1376) call()
v_accvgpr_write a20, 0.00000 // (op 1674) call()
v_accvgpr_write a21, 0.00000 // (op 1674) call()
v_accvgpr_write a22, 0.00000 // (op 1674) call()
v_accvgpr_write a23, 0.00000 // (op 1674) call()
v_accvgpr_write a24, 0.00000 // (op 1972) call()
v_accvgpr_write a25, 0.00000 // (op 1972) call()
v_accvgpr_write a26, 0.00000 // (op 1972) call()
v_accvgpr_write a27, 0.00000 // (op 1972) call()
v_accvgpr_write a28, 0.00000 // (op 2270) call()
v_accvgpr_write a29, 0.00000 // (op 2270) call()
v_accvgpr_write a30, 0.00000 // (op 2270) call()
v_accvgpr_write a31, 0.00000 // (op 2270) call()
v_accvgpr_write a32, 0.00000 // (op 2840) call()
v_accvgpr_write a33, 0.00000 // (op 2840) call()
v_accvgpr_write a34, 0.00000 // (op 2840) call()
v_accvgpr_write a35, 0.00000 // (op 2840) call()
v_accvgpr_write a36, 0.00000 // (op 3825) call()
v_accvgpr_write a37, 0.00000 // (op 3825) call()
v_accvgpr_write a38, 0.00000 // (op 3825) call()
v_accvgpr_write a39, 0.00000 // (op 3825) call()
v_accvgpr_write a40, 0.00000 // (op 3826) call()
v_accvgpr_write a41, 0.00000 // (op 3826) call()
v_accvgpr_write a42, 0.00000 // (op 3826) call()
v_accvgpr_write a43, 0.00000 // (op 3826) call()
v_accvgpr_write a44, 0.00000 // (op 3827) call()
v_accvgpr_write a45, 0.00000 // (op 3827) call()
v_accvgpr_write a46, 0.00000 // (op 3827) call()
v_accvgpr_write a47, 0.00000 // (op 3827) call()
v_accvgpr_write a48, 0.00000 // (op 3828) call()
v_accvgpr_write a49, 0.00000 // (op 3828) call()
v_accvgpr_write a50, 0.00000 // (op 3828) call()
v_accvgpr_write a51, 0.00000 // (op 3828) call()
v_accvgpr_write a52, 0.00000 // (op 3829) call()
v_accvgpr_write a53, 0.00000 // (op 3829) call()
v_accvgpr_write a54, 0.00000 // (op 3829) call()
v_accvgpr_write a55, 0.00000 // (op 3829) call()
v_accvgpr_write a56, 0.00000 // (op 3830) call()
v_accvgpr_write a57, 0.00000 // (op 3830) call()
v_accvgpr_write a58, 0.00000 // (op 3830) call()
v_accvgpr_write a59, 0.00000 // (op 3830) call()
v_accvgpr_write a60, 0.00000 // (op 3831) call()
v_accvgpr_write a61, 0.00000 // (op 3831) call()
v_accvgpr_write a62, 0.00000 // (op 3831) call()
v_accvgpr_write a63, 0.00000 // (op 3831) call()
v_accvgpr_write a64, 0.00000 // (op 5418) call()
v_accvgpr_write a65, 0.00000 // (op 5418) call()
v_accvgpr_write a66, 0.00000 // (op 5418) call()
v_accvgpr_write a67, 0.00000 // (op 5418) call()
v_accvgpr_write a68, 0.00000 // (op 6403) call()
v_accvgpr_write a69, 0.00000 // (op 6403) call()
v_accvgpr_write a70, 0.00000 // (op 6403) call()
v_accvgpr_write a71, 0.00000 // (op 6403) call()
v_accvgpr_write a72, 0.00000 // (op 6404) call()
v_accvgpr_write a73, 0.00000 // (op 6404) call()
v_accvgpr_write a74, 0.00000 // (op 6404) call()
v_accvgpr_write a75, 0.00000 // (op 6404) call()
v_accvgpr_write a76, 0.00000 // (op 6405) call()
v_accvgpr_write a77, 0.00000 // (op 6405) call()
v_accvgpr_write a78, 0.00000 // (op 6405) call()
v_accvgpr_write a79, 0.00000 // (op 6405) call()
v_accvgpr_write a80, 0.00000 // (op 6406) call()
v_accvgpr_write a81, 0.00000 // (op 6406) call()
v_accvgpr_write a82, 0.00000 // (op 6406) call()
v_accvgpr_write a83, 0.00000 // (op 6406) call()
v_accvgpr_write a84, 0.00000 // (op 6407) call()
v_accvgpr_write a85, 0.00000 // (op 6407) call()
v_accvgpr_write a86, 0.00000 // (op 6407) call()
v_accvgpr_write a87, 0.00000 // (op 6407) call()
v_accvgpr_write a88, 0.00000 // (op 6408) call()
v_accvgpr_write a89, 0.00000 // (op 6408) call()
v_accvgpr_write a90, 0.00000 // (op 6408) call()
v_accvgpr_write a91, 0.00000 // (op 6408) call()
v_accvgpr_write a92, 0.00000 // (op 6409) call()
v_accvgpr_write a93, 0.00000 // (op 6409) call()
v_accvgpr_write a94, 0.00000 // (op 6409) call()
v_accvgpr_write a95, 0.00000 // (op 6409) call()
v_accvgpr_write a96, 0.00000 // (op 7996) call()
v_accvgpr_write a97, 0.00000 // (op 7996) call()
v_accvgpr_write a98, 0.00000 // (op 7996) call()
v_accvgpr_write a99, 0.00000 // (op 7996) call()
v_accvgpr_write a100, 0.00000 // (op 8981) call()
v_accvgpr_write a101, 0.00000 // (op 8981) call()
v_accvgpr_write a102, 0.00000 // (op 8981) call()
v_accvgpr_write a103, 0.00000 // (op 8981) call()
v_accvgpr_write a104, 0.00000 // (op 8982) call()
v_accvgpr_write a105, 0.00000 // (op 8982) call()
v_accvgpr_write a106, 0.00000 // (op 8982) call()
v_accvgpr_write a107, 0.00000 // (op 8982) call()
v_accvgpr_write a108, 0.00000 // (op 8983) call()
v_accvgpr_write a109, 0.00000 // (op 8983) call()
v_accvgpr_write a110, 0.00000 // (op 8983) call()
v_accvgpr_write a111, 0.00000 // (op 8983) call()
v_accvgpr_write a112, 0.00000 // (op 8984) call()
v_accvgpr_write a113, 0.00000 // (op 8984) call()
v_accvgpr_write a114, 0.00000 // (op 8984) call()
v_accvgpr_write a115, 0.00000 // (op 8984) call()
v_accvgpr_write a116, 0.00000 // (op 8985) call()
v_accvgpr_write a117, 0.00000 // (op 8985) call()
v_accvgpr_write a118, 0.00000 // (op 8985) call()
v_accvgpr_write a119, 0.00000 // (op 8985) call()
v_accvgpr_write a120, 0.00000 // (op 8986) call()
v_accvgpr_write a121, 0.00000 // (op 8986) call()
v_accvgpr_write a122, 0.00000 // (op 8986) call()
v_accvgpr_write a123, 0.00000 // (op 8986) call()
v_accvgpr_write a124, 0.00000 // (op 8987) call()
v_accvgpr_write a125, 0.00000 // (op 8987) call()
v_accvgpr_write a126, 0.00000 // (op 8987) call()
v_accvgpr_write a127, 0.00000 // (op 8987) call()
v_accvgpr_write a128, 0.00000 // (op 10574) call()
v_accvgpr_write a129, 0.00000 // (op 10574) call()
v_accvgpr_write a130, 0.00000 // (op 10574) call()
v_accvgpr_write a131, 0.00000 // (op 10574) call()
v_accvgpr_write a132, 0.00000 // (op 11559) call()
v_accvgpr_write a133, 0.00000 // (op 11559) call()
v_accvgpr_write a134, 0.00000 // (op 11559) call()
v_accvgpr_write a135, 0.00000 // (op 11559) call()
v_accvgpr_write a136, 0.00000 // (op 11560) call()
v_accvgpr_write a137, 0.00000 // (op 11560) call()
v_accvgpr_write a138, 0.00000 // (op 11560) call()
v_accvgpr_write a139, 0.00000 // (op 11560) call()
v_accvgpr_write a140, 0.00000 // (op 11561) call()
v_accvgpr_write a141, 0.00000 // (op 11561) call()
v_accvgpr_write a142, 0.00000 // (op 11561) call()
v_accvgpr_write a143, 0.00000 // (op 11561) call()
v_accvgpr_write a144, 0.00000 // (op 11562) call()
v_accvgpr_write a145, 0.00000 // (op 11562) call()
v_accvgpr_write a146, 0.00000 // (op 11562) call()
v_accvgpr_write a147, 0.00000 // (op 11562) call()
v_accvgpr_write a148, 0.00000 // (op 11563) call()
v_accvgpr_write a149, 0.00000 // (op 11563) call()
v_accvgpr_write a150, 0.00000 // (op 11563) call()
v_accvgpr_write a151, 0.00000 // (op 11563) call()
v_accvgpr_write a152, 0.00000 // (op 11564) call()
v_accvgpr_write a153, 0.00000 // (op 11564) call()
v_accvgpr_write a154, 0.00000 // (op 11564) call()
v_accvgpr_write a155, 0.00000 // (op 11564) call()
v_accvgpr_write a156, 0.00000 // (op 11565) call()
v_accvgpr_write a157, 0.00000 // (op 11565) call()
v_accvgpr_write a158, 0.00000 // (op 11565) call()
v_accvgpr_write a159, 0.00000 // (op 11565) call()
v_accvgpr_write a160, 0.00000 // (op 13152) call()
v_accvgpr_write a161, 0.00000 // (op 13152) call()
v_accvgpr_write a162, 0.00000 // (op 13152) call()
v_accvgpr_write a163, 0.00000 // (op 13152) call()
v_accvgpr_write a164, 0.00000 // (op 14137) call()
v_accvgpr_write a165, 0.00000 // (op 14137) call()
v_accvgpr_write a166, 0.00000 // (op 14137) call()
v_accvgpr_write a167, 0.00000 // (op 14137) call()
v_accvgpr_write a168, 0.00000 // (op 14138) call()
v_accvgpr_write a169, 0.00000 // (op 14138) call()
v_accvgpr_write a170, 0.00000 // (op 14138) call()
v_accvgpr_write a171, 0.00000 // (op 14138) call()
v_accvgpr_write a172, 0.00000 // (op 14139) call()
v_accvgpr_write a173, 0.00000 // (op 14139) call()
v_accvgpr_write a174, 0.00000 // (op 14139) call()
v_accvgpr_write a175, 0.00000 // (op 14139) call()
v_accvgpr_write a176, 0.00000 // (op 14140) call()
v_accvgpr_write a177, 0.00000 // (op 14140) call()
v_accvgpr_write a178, 0.00000 // (op 14140) call()
v_accvgpr_write a179, 0.00000 // (op 14140) call()
v_accvgpr_write a180, 0.00000 // (op 14141) call()
v_accvgpr_write a181, 0.00000 // (op 14141) call()
v_accvgpr_write a182, 0.00000 // (op 14141) call()
v_accvgpr_write a183, 0.00000 // (op 14141) call()
v_accvgpr_write a184, 0.00000 // (op 14142) call()
v_accvgpr_write a185, 0.00000 // (op 14142) call()
v_accvgpr_write a186, 0.00000 // (op 14142) call()
v_accvgpr_write a187, 0.00000 // (op 14142) call()
v_accvgpr_write a188, 0.00000 // (op 14143) call()
v_accvgpr_write a189, 0.00000 // (op 14143) call()
v_accvgpr_write a190, 0.00000 // (op 14143) call()
v_accvgpr_write a191, 0.00000 // (op 14143) call()
v_accvgpr_write a192, 0.00000 // (op 15730) call()
v_accvgpr_write a193, 0.00000 // (op 15730) call()
v_accvgpr_write a194, 0.00000 // (op 15730) call()
v_accvgpr_write a195, 0.00000 // (op 15730) call()
v_accvgpr_write a196, 0.00000 // (op 16715) call()
v_accvgpr_write a197, 0.00000 // (op 16715) call()
v_accvgpr_write a198, 0.00000 // (op 16715) call()
v_accvgpr_write a199, 0.00000 // (op 16715) call()
v_accvgpr_write a200, 0.00000 // (op 16716) call()
v_accvgpr_write a201, 0.00000 // (op 16716) call()
v_accvgpr_write a202, 0.00000 // (op 16716) call()
v_accvgpr_write a203, 0.00000 // (op 16716) call()
v_accvgpr_write a204, 0.00000 // (op 16717) call()
v_accvgpr_write a205, 0.00000 // (op 16717) call()
v_accvgpr_write a206, 0.00000 // (op 16717) call()
v_accvgpr_write a207, 0.00000 // (op 16717) call()
v_accvgpr_write a208, 0.00000 // (op 16718) call()
v_accvgpr_write a209, 0.00000 // (op 16718) call()
v_accvgpr_write a210, 0.00000 // (op 16718) call()
v_accvgpr_write a211, 0.00000 // (op 16718) call()
v_accvgpr_write a212, 0.00000 // (op 16719) call()
v_accvgpr_write a213, 0.00000 // (op 16719) call()
v_accvgpr_write a214, 0.00000 // (op 16719) call()
v_accvgpr_write a215, 0.00000 // (op 16719) call()
v_accvgpr_write a216, 0.00000 // (op 16720) call()
v_accvgpr_write a217, 0.00000 // (op 16720) call()
v_accvgpr_write a218, 0.00000 // (op 16720) call()
v_accvgpr_write a219, 0.00000 // (op 16720) call()
v_accvgpr_write a220, 0.00000 // (op 16721) call()
v_accvgpr_write a221, 0.00000 // (op 16721) call()
v_accvgpr_write a222, 0.00000 // (op 16721) call()
v_accvgpr_write a223, 0.00000 // (op 16721) call()
v_accvgpr_write a224, 0.00000 // (op 18308) call()
v_accvgpr_write a225, 0.00000 // (op 18308) call()
v_accvgpr_write a226, 0.00000 // (op 18308) call()
v_accvgpr_write a227, 0.00000 // (op 18308) call()
v_accvgpr_write a228, 0.00000 // (op 19293) call()
v_accvgpr_write a229, 0.00000 // (op 19293) call()
v_accvgpr_write a230, 0.00000 // (op 19293) call()
v_accvgpr_write a231, 0.00000 // (op 19293) call()
v_accvgpr_write a232, 0.00000 // (op 19294) call()
v_accvgpr_write a233, 0.00000 // (op 19294) call()
v_accvgpr_write a234, 0.00000 // (op 19294) call()
v_accvgpr_write a235, 0.00000 // (op 19294) call()
v_accvgpr_write a236, 0.00000 // (op 19295) call()
v_accvgpr_write a237, 0.00000 // (op 19295) call()
v_accvgpr_write a238, 0.00000 // (op 19295) call()
v_accvgpr_write a239, 0.00000 // (op 19295) call()
v_accvgpr_write a240, 0.00000 // (op 19296) call()
v_accvgpr_write a241, 0.00000 // (op 19296) call()
v_accvgpr_write a242, 0.00000 // (op 19296) call()
v_accvgpr_write a243, 0.00000 // (op 19296) call()
v_accvgpr_write a244, 0.00000 // (op 19297) call()
v_accvgpr_write a245, 0.00000 // (op 19297) call()
v_accvgpr_write a246, 0.00000 // (op 19297) call()
v_accvgpr_write a247, 0.00000 // (op 19297) call()
v_accvgpr_write a248, 0.00000 // (op 19298) call()
v_accvgpr_write a249, 0.00000 // (op 19298) call()
v_accvgpr_write a250, 0.00000 // (op 19298) call()
v_accvgpr_write a251, 0.00000 // (op 19298) call()
v_accvgpr_write a252, 0.00000 // (op 19299) call()
v_accvgpr_write a253, 0.00000 // (op 19299) call()
v_accvgpr_write a254, 0.00000 // (op 19299) call()
v_accvgpr_write a255, 0.00000 // (op 19299) call()
v_and_b32 v22, -8, v1 // (op 54763) 
v_sub_u32 v23, v1, v22 // (op 54763) 

v_lshlrev_b32 v22, 5, v23 // (op 54763) 

v_lshl_add_u32 v23, s16, 8, v22 // (op 54763) 

v_mov_b32 v25, 0 // (op 54763) convert
v_mov_b32 v24, v23 // (op 54763) convert

v_lshrrev_b32 v22, 3, v1 // (op 54763) 
v_mul_lo_u32 v27, v22, s13 // (op 54763) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 54763) most significant: high of low * low
v_mul_lo_u32 v26, v22, s12 // (op 54763) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 54763) most significant: sum


v_add_co_u32 v22, s[0:1], v24, v26 // (op 54763) least significant half
v_addc_co_u32 v23, s[0:1], v25, v27, s[0:1] // (op 54763) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 54763) 


v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 54763) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 54765) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 54765) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 54765) least significant
s_mul_hi_u32 s7, s12, s22 // (op 54765) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54765) most significant: sum
s_add_u32 s1, s1, s7 // (op 54765) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 54765) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 54765) 

s_mov_b32 s8, s20 // (op 54765) reg expression
s_mov_b32 s9, s21 // (op 54765) reg expression
s_mov_b32 s10, s0 // (op 54765) reg expression
s_mov_b32 s11, 131072 // (op 54765) call()

v_and_b32 v22, -8, v1 // (op 54782) 
v_sub_u32 v23, v1, v22 // (op 54782) 

v_lshlrev_b32 v22, 5, v23 // (op 54782) 

v_lshl_add_u32 v23, s16, 8, v22 // (op 54782) 

v_mov_b32 v27, 0 // (op 54782) convert
v_mov_b32 v26, v23 // (op 54782) convert

v_lshrrev_b32 v22, 3, v1 // (op 54782) 
v_mul_lo_u32 v29, v22, s13 // (op 54782) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 54782) most significant: high of low * low
v_mul_lo_u32 v28, v22, s12 // (op 54782) least significant: low of low * low
v_add_u32 v29, v29, v23 // (op 54782) most significant: sum


v_add_co_u32 v22, s[0:1], v26, v28 // (op 54782) least significant half
v_addc_co_u32 v23, s[0:1], v27, v29, s[0:1] // (op 54782) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 54782) 


v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 54782) 

s_mul_i32 s1, s12, s23 // (op 54784) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 54784) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 54784) least significant
s_mul_hi_u32 s7, s12, s22 // (op 54784) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54784) most significant: sum
s_add_u32 s1, s1, s7 // (op 54784) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54784) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54784) 

s_mov_b32 s8, s20 // (op 54784) reg expression
s_mov_b32 s9, s21 // (op 54784) reg expression
s_mov_b32 s10, s0 // (op 54784) reg expression
s_mov_b32 s11, 131072 // (op 54784) call()

v_and_b32 v22, -8, v1 // (op 54827) 
v_sub_u32 v23, v1, v22 // (op 54827) 

v_lshlrev_b32 v22, 5, v23 // (op 54827) 

v_lshl_add_u32 v23, s17, 8, v22 // (op 54827) 

v_mov_b32 v29, 0 // (op 54827) convert
v_mov_b32 v28, v23 // (op 54827) convert

v_lshrrev_b32 v22, 3, v1 // (op 54827) 
v_mul_lo_u32 v31, v22, s25 // (op 54827) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 54827) most significant: high of low * low
v_mul_lo_u32 v30, v22, s24 // (op 54827) least significant: low of low * low
v_add_u32 v31, v31, v23 // (op 54827) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v28 // (op 54827) least significant half
v_addc_co_u32 v31, s[0:1], v31, v29, s[0:1] // (op 54827) most significant half



v_lshlrev_b64 v[22:23], 2, v[30:31] // (op 54827) 

v_lshrrev_b64 v[28:29], 3, v[22:23] // (op 54827) 

s_mul_i32 s1, s24, s23 // (op 54829) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 54829) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 54829) least significant
s_mul_hi_u32 s7, s24, s22 // (op 54829) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54829) most significant: sum
s_add_u32 s1, s1, s7 // (op 54829) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54829) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54829) 

s_mov_b32 s48, s26 // (op 54829) reg expression
s_mov_b32 s49, s27 // (op 54829) reg expression
s_mov_b32 s50, s0 // (op 54829) reg expression
s_mov_b32 s51, 131072 // (op 54829) call()

v_and_b32 v22, -8, v1 // (op 54846) 
v_sub_u32 v23, v1, v22 // (op 54846) 

v_lshlrev_b32 v22, 5, v23 // (op 54846) 

v_lshl_add_u32 v23, s17, 8, v22 // (op 54846) 

v_mov_b32 v31, 0 // (op 54846) convert
v_mov_b32 v30, v23 // (op 54846) convert

v_lshrrev_b32 v22, 3, v1 // (op 54846) 
v_mul_lo_u32 v33, v22, s25 // (op 54846) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 54846) most significant: high of low * low
v_mul_lo_u32 v32, v22, s24 // (op 54846) least significant: low of low * low
v_add_u32 v33, v33, v23 // (op 54846) most significant: sum

v_add_co_u32 v32, s[0:1], v32, v30 // (op 54846) least significant half
v_addc_co_u32 v33, s[0:1], v33, v31, s[0:1] // (op 54846) most significant half



v_lshlrev_b64 v[22:23], 2, v[32:33] // (op 54846) 

v_lshrrev_b64 v[30:31], 3, v[22:23] // (op 54846) 

s_mul_i32 s1, s24, s23 // (op 54848) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 54848) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 54848) least significant
s_mul_hi_u32 s7, s24, s22 // (op 54848) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54848) most significant: sum
s_add_u32 s1, s1, s7 // (op 54848) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54848) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54848) 

s_mov_b32 s48, s26 // (op 54848) reg expression
s_mov_b32 s49, s27 // (op 54848) reg expression
s_mov_b32 s50, s0 // (op 54848) reg expression
s_mov_b32 s51, 131072 // (op 54848) call()

v_and_b32 v22, -64, v1 // (op 54891) 
v_sub_u32 v23, v1, v22 // (op 54891) 

v_lshlrev_b32 v22, 2, v23 // (op 54891) 

v_lshl_add_u32 v23, s16, 8, v22 // (op 54891) 

v_lshrrev_b32 v22, 6, v1 // (op 54891) 
v_mul_lo_u32 v32, v22, s28 // (op 54891) 

v_add_u32 v22, v23, v32 // (op 54891) 


s_mul_i32 s1, s28, s33 // (op 54893) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 54893) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 54893) least significant
s_mul_hi_u32 s7, s28, s32 // (op 54893) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54893) most significant: sum
s_add_u32 s1, s1, s7 // (op 54893) most significant: sum


s_mov_b32 s52, s30 // (op 54893) reg expression
s_mov_b32 s53, s31 // (op 54893) reg expression
s_mov_b32 s54, s0 // (op 54893) reg expression
s_mov_b32 s55, 131072 // (op 54893) call()

v_and_b32 v23, -64, v1 // (op 54910) 
v_sub_u32 v32, v1, v23 // (op 54910) 

v_lshlrev_b32 v23, 2, v32 // (op 54910) 

v_lshl_add_u32 v32, s16, 8, v23 // (op 54910) 

v_lshrrev_b32 v23, 6, v1 // (op 54910) 
v_mul_lo_u32 v33, v23, s28 // (op 54910) 

v_add_u32 v23, v32, v33 // (op 54910) 


s_mul_i32 s1, s28, s33 // (op 54912) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 54912) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 54912) least significant
s_mul_hi_u32 s7, s28, s32 // (op 54912) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54912) most significant: sum
s_add_u32 s1, s1, s7 // (op 54912) most significant: sum


s_mov_b32 s52, s30 // (op 54912) reg expression
s_mov_b32 s53, s31 // (op 54912) reg expression
s_mov_b32 s54, s0 // (op 54912) reg expression
s_mov_b32 s55, 131072 // (op 54912) call()

v_and_b32 v32, -64, v1 // (op 54955) 
v_sub_u32 v33, v1, v32 // (op 54955) 

v_lshlrev_b32 v32, 2, v33 // (op 54955) 

v_lshl_add_u32 v33, s17, 8, v32 // (op 54955) 

v_lshrrev_b32 v32, 6, v1 // (op 54955) 
v_mul_lo_u32 v34, v32, s34 // (op 54955) 
v_add_u32 v34, v34, v33 // (op 54955) 


s_mul_i32 s1, s34, s33 // (op 54957) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 54957) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 54957) least significant
s_mul_hi_u32 s7, s34, s32 // (op 54957) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54957) most significant: sum
s_add_u32 s1, s1, s7 // (op 54957) most significant: sum


s_mov_b32 s56, s36 // (op 54957) reg expression
s_mov_b32 s57, s37 // (op 54957) reg expression
s_mov_b32 s58, s0 // (op 54957) reg expression
s_mov_b32 s59, 131072 // (op 54957) call()

v_and_b32 v32, -64, v1 // (op 54974) 
v_sub_u32 v33, v1, v32 // (op 54974) 

v_lshlrev_b32 v32, 2, v33 // (op 54974) 

v_lshl_add_u32 v33, s17, 8, v32 // (op 54974) 

v_lshrrev_b32 v32, 6, v1 // (op 54974) 
v_mul_lo_u32 v35, v32, s34 // (op 54974) 
v_add_u32 v35, v35, v33 // (op 54974) 


s_mul_i32 s1, s34, s33 // (op 54976) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 54976) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 54976) least significant
s_mul_hi_u32 s7, s34, s32 // (op 54976) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54976) most significant: sum
s_add_u32 s1, s1, s7 // (op 54976) most significant: sum


s_mov_b32 s56, s36 // (op 54976) reg expression
s_mov_b32 s57, s37 // (op 54976) reg expression
s_mov_b32 s58, s0 // (op 54976) reg expression
s_mov_b32 s59, 131072 // (op 54976) call()

v_and_b32 v32, -8, v1 // (op 55102) 
v_sub_u32 v33, v1, v32 // (op 55102) 

v_lshlrev_b32 v32, 5, v33 // (op 55102) 

v_lshrrev_b32 v33, 3, v1 // (op 55102) 
v_lshl_add_u32 v36, v33, 8, v32 // (op 55102) 


v_lshlrev_b32 v32, 2, v36 // (op 55102) 

v_lshrrev_b32 v33, 3, v32 // (op 55102) 

s_nop 0
v_readfirstlane_b32 s0, v33 // (op 55102) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 54641) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54641) 

v_lshrrev_b64 v[32:33], 3, s[14:15] // (op 54641) 


v_mov_b64 v[36:37], v[28:29] // (op 54641) reg expression
s_mov_b32 m0, s0 // (op 54641) reg expression
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 54641) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v32 // (op 54641) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 54641) most significant half

s_add_u32 m0, m0, 4096 // (op 54641) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 54641) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v32 // (op 54641) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 54641) most significant half

s_add_u32 m0, m0, 4096 // (op 54641) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 54641) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v32 // (op 54641) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 54641) most significant half

s_add_u32 m0, m0, 4096 // (op 54641) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 54641) Load value direct to lds





v_mov_b32 v32, v34 // (op 53647) reg expression

buffer_load_dword v33, v32, s[56:59], 0 offen // (op 53647) Load value

v_and_b32 v32, -8, v1 // (op 55045) 
v_sub_u32 v36, v1, v32 // (op 55045) 

v_lshlrev_b32 v32, 5, v36 // (op 55045) 

v_lshrrev_b32 v36, 3, v1 // (op 55045) 
v_lshl_add_u32 v37, v36, 8, v32 // (op 55045) 


v_lshlrev_b32 v32, 2, v37 // (op 55045) 

v_lshrrev_b32 v36, 3, v32 // (op 55045) 

s_nop 0
v_readfirstlane_b32 s0, v36 // (op 55045) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 54646) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54646) 

v_lshrrev_b64 v[36:37], 3, s[14:15] // (op 54646) 


v_mov_b64 v[38:39], v[24:25] // (op 54646) reg expression
s_add_u32 s1, 16384, s0 // (op 54646) 
s_mov_b32 m0, s1 // (op 54646) reg expression
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54646) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v36 // (op 54646) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54646) most significant half

s_add_u32 m0, m0, 4096 // (op 54646) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54646) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v36 // (op 54646) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54646) most significant half

s_add_u32 m0, m0, 4096 // (op 54646) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54646) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v36 // (op 54646) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54646) most significant half

s_add_u32 m0, m0, 4096 // (op 54646) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54646) Load value direct to lds






v_mov_b32 v32, v22 // (op 53653) reg expression

buffer_load_dword v36, v32, s[52:55], 0 offen // (op 53653) Load value

v_and_b32 v32, -8, v1 // (op 55329) 
v_sub_u32 v37, v1, v32 // (op 55329) 

v_lshlrev_b32 v32, 5, v37 // (op 55329) 

v_lshrrev_b32 v37, 3, v1 // (op 55329) 
v_lshl_add_u32 v38, v37, 8, v32 // (op 55329) 


v_lshlrev_b32 v32, 2, v38 // (op 55329) 

v_lshrrev_b32 v37, 3, v32 // (op 55329) 

s_nop 0
v_readfirstlane_b32 s0, v37 // (op 55329) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 54651) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54651) 

v_lshrrev_b64 v[38:39], 3, s[14:15] // (op 54651) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 54651) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54651) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54651) 

v_mov_b32 v32, s6 // (op 54651) 
v_mov_b32 v37, s7 // (op 54651) 
v_add_co_u32 v40, s[14:15], v28, v32 // (op 54651) least significant half
v_addc_co_u32 v41, s[14:15], v29, v37, s[14:15] // (op 54651) most significant half




s_add_u32 s1, 32768, s0 // (op 54651) 
s_mov_b32 m0, s1 // (op 54651) reg expression
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 54651) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v38 // (op 54651) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 54651) most significant half

s_add_u32 m0, m0, 4096 // (op 54651) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 54651) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v38 // (op 54651) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 54651) most significant half

s_add_u32 m0, m0, 4096 // (op 54651) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 54651) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v38 // (op 54651) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 54651) most significant half

s_add_u32 m0, m0, 4096 // (op 54651) 
buffer_load_dwordx4 v40, s[48:51], 0 offen offset: 0 lds // (op 54651) Load value direct to lds






s_lshl_b32 s0, s34, 2 // (op 53659) 
v_add_u32 v32, v34, s0 // (op 53659) 


buffer_load_dword v37, v32, s[56:59], 0 offen // (op 53659) Load value

v_and_b32 v32, -8, v1 // (op 55285) 
v_sub_u32 v38, v1, v32 // (op 55285) 

v_lshlrev_b32 v32, 5, v38 // (op 55285) 

v_lshrrev_b32 v38, 3, v1 // (op 55285) 
v_lshl_add_u32 v39, v38, 8, v32 // (op 55285) 


v_lshlrev_b32 v32, 2, v39 // (op 55285) 

v_lshrrev_b32 v38, 3, v32 // (op 55285) 

s_nop 0
v_readfirstlane_b32 s0, v38 // (op 55285) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 54653) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54653) 

v_lshrrev_b64 v[38:39], 3, s[14:15] // (op 54653) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 54653) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54653) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54653) 

v_mov_b32 v32, s6 // (op 54653) 
v_mov_b32 v40, s7 // (op 54653) 
v_add_co_u32 v42, s[14:15], v24, v32 // (op 54653) least significant half
v_addc_co_u32 v43, s[14:15], v25, v40, s[14:15] // (op 54653) most significant half




s_add_u32 s1, 49152, s0 // (op 54653) 
s_mov_b32 m0, s1 // (op 54653) reg expression
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 54653) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v38 // (op 54653) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54653) most significant half

s_add_u32 m0, m0, 4096 // (op 54653) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 54653) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v38 // (op 54653) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54653) most significant half

s_add_u32 m0, m0, 4096 // (op 54653) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 54653) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v38 // (op 54653) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54653) most significant half

s_add_u32 m0, m0, 4096 // (op 54653) 
buffer_load_dwordx4 v42, s[8:11], 0 offen offset: 0 lds // (op 54653) Load value direct to lds






s_lshl_b32 s0, s28, 2 // (op 53665) 
v_add_u32 v32, v22, s0 // (op 53665) 


buffer_load_dword v38, v32, s[52:55], 0 offen // (op 53665) Load value

v_and_b32 v32, -8, v1 // (op 55475) 
v_sub_u32 v39, v1, v32 // (op 55475) 

v_lshlrev_b32 v32, 5, v39 // (op 55475) 

v_lshrrev_b32 v39, 3, v1 // (op 55475) 
v_lshl_add_u32 v40, v39, 8, v32 // (op 55475) 


v_lshlrev_b32 v32, 2, v40 // (op 55475) 

v_lshrrev_b32 v39, 3, v32 // (op 55475) 

s_nop 0
v_readfirstlane_b32 s0, v39 // (op 55475) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 54655) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54655) 

v_lshrrev_b64 v[40:41], 3, s[14:15] // (op 54655) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 54655) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54655) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54655) 

s_lshl_b64 s[14:15], s[6:7], 1 // (op 54655) 

v_mov_b32 v32, s14 // (op 54655) 
v_mov_b32 v39, s15 // (op 54655) 
v_add_co_u32 v42, s[6:7], v28, v32 // (op 54655) least significant half
v_addc_co_u32 v43, s[6:7], v29, v39, s[6:7] // (op 54655) most significant half




s_add_u32 s1, 65536, s0 // (op 54655) 
s_mov_b32 m0, s1 // (op 54655) reg expression
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 54655) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v40 // (op 54655) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54655) most significant half

s_add_u32 m0, m0, 4096 // (op 54655) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 54655) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v40 // (op 54655) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54655) most significant half

s_add_u32 m0, m0, 4096 // (op 54655) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 54655) Load value direct to lds
v_add_co_u32 v42, s[6:7], v42, v40 // (op 54655) least significant half
v_addc_co_u32 v43, s[6:7], 0, v43, s[6:7] // (op 54655) most significant half

s_add_u32 m0, m0, 4096 // (op 54655) 
buffer_load_dwordx4 v42, s[48:51], 0 offen offset: 0 lds // (op 54655) Load value direct to lds






s_lshl_b32 s0, s34, 3 // (op 53671) 
v_add_u32 v32, v34, s0 // (op 53671) 


buffer_load_dword v39, v32, s[56:59], 0 offen // (op 53671) Load value

v_and_b32 v32, -8, v1 // (op 55431) 
v_sub_u32 v40, v1, v32 // (op 55431) 

v_lshlrev_b32 v32, 5, v40 // (op 55431) 

v_lshrrev_b32 v40, 3, v1 // (op 55431) 
v_lshl_add_u32 v41, v40, 8, v32 // (op 55431) 


v_lshlrev_b32 v32, 2, v41 // (op 55431) 

v_lshrrev_b32 v40, 3, v32 // (op 55431) 

s_nop 0
v_readfirstlane_b32 s0, v40 // (op 55431) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 54657) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54657) 

v_lshrrev_b64 v[40:41], 3, s[14:15] // (op 54657) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 54657) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54657) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54657) 

s_lshl_b64 s[14:15], s[6:7], 1 // (op 54657) 

v_mov_b32 v32, s14 // (op 54657) 
v_mov_b32 v42, s15 // (op 54657) 
v_add_co_u32 v44, s[6:7], v24, v32 // (op 54657) least significant half
v_addc_co_u32 v45, s[6:7], v25, v42, s[6:7] // (op 54657) most significant half




s_add_u32 s1, 81920, s0 // (op 54657) 
s_mov_b32 m0, s1 // (op 54657) reg expression
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54657) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v40 // (op 54657) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54657) most significant half

s_add_u32 m0, m0, 4096 // (op 54657) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54657) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v40 // (op 54657) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54657) most significant half

s_add_u32 m0, m0, 4096 // (op 54657) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54657) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v40 // (op 54657) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54657) most significant half

s_add_u32 m0, m0, 4096 // (op 54657) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54657) Load value direct to lds






s_lshl_b32 s0, s28, 3 // (op 53677) 
v_add_u32 v32, v22, s0 // (op 53677) 


buffer_load_dword v40, v32, s[52:55], 0 offen // (op 53677) Load value

v_and_b32 v32, -8, v1 // (op 55621) 
v_sub_u32 v41, v1, v32 // (op 55621) 

v_lshlrev_b32 v32, 5, v41 // (op 55621) 

v_lshrrev_b32 v41, 3, v1 // (op 55621) 
v_lshl_add_u32 v42, v41, 8, v32 // (op 55621) 


v_lshlrev_b32 v32, 2, v42 // (op 55621) 

v_lshrrev_b32 v41, 3, v32 // (op 55621) 

s_nop 0
v_readfirstlane_b32 s0, v41 // (op 55621) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 54659) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54659) 

v_lshrrev_b64 v[42:43], 3, s[14:15] // (op 54659) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 54659) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54659) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54659) 

s_mul_i32 s15, 3, s7 // (op 54659) most significant: low of low * high
s_mul_i32 s14, 3, s6 // (op 54659) least significant
s_mul_hi_u32 s1, 3, s6 // (op 54659) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 54659) most significant: sum


v_mov_b32 v32, s14 // (op 54659) 
v_mov_b32 v41, s15 // (op 54659) 
v_add_co_u32 v44, s[6:7], v28, v32 // (op 54659) least significant half
v_addc_co_u32 v45, s[6:7], v29, v41, s[6:7] // (op 54659) most significant half




s_add_u32 s1, 98304, s0 // (op 54659) 
s_mov_b32 m0, s1 // (op 54659) reg expression
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 54659) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54659) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54659) most significant half

s_add_u32 m0, m0, 4096 // (op 54659) 
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 54659) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54659) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54659) most significant half

s_add_u32 m0, m0, 4096 // (op 54659) 
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 54659) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54659) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54659) most significant half

s_add_u32 m0, m0, 4096 // (op 54659) 
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 54659) Load value direct to lds







s_lshl_b32 s0, s34, 2 // (op 53683) 
s_mul_i32 s1, 3, s0 // (op 53683) 

v_add_u32 v32, v34, s1 // (op 53683) 


buffer_load_dword v28, v32, s[56:59], 0 offen // (op 53683) Load value


v_and_b32 v29, -8, v1 // (op 55577) 
v_sub_u32 v32, v1, v29 // (op 55577) 

v_lshlrev_b32 v29, 5, v32 // (op 55577) 

v_lshrrev_b32 v32, 3, v1 // (op 55577) 
v_lshl_add_u32 v34, v32, 8, v29 // (op 55577) 


v_lshlrev_b32 v29, 2, v34 // (op 55577) 

v_lshrrev_b32 v32, 3, v29 // (op 55577) 

s_nop 0
v_readfirstlane_b32 s0, v32 // (op 55577) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 54661) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54661) 

v_lshrrev_b64 v[42:43], 3, s[14:15] // (op 54661) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 54661) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 54661) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 54661) 

s_mul_i32 s15, 3, s7 // (op 54661) most significant: low of low * high
s_mul_i32 s14, 3, s6 // (op 54661) least significant
s_mul_hi_u32 s1, 3, s6 // (op 54661) most significant: high of low * low
s_add_u32 s15, s15, s1 // (op 54661) most significant: sum


v_mov_b32 v29, s14 // (op 54661) 
v_mov_b32 v32, s15 // (op 54661) 
v_add_co_u32 v44, s[6:7], v24, v29 // (op 54661) least significant half
v_addc_co_u32 v45, s[6:7], v25, v32, s[6:7] // (op 54661) most significant half




s_add_u32 s1, 114688, s0 // (op 54661) 
s_mov_b32 m0, s1 // (op 54661) reg expression
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54661) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54661) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54661) most significant half

s_add_u32 m0, m0, 4096 // (op 54661) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54661) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54661) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54661) most significant half

s_add_u32 m0, m0, 4096 // (op 54661) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54661) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v42 // (op 54661) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 54661) most significant half

s_add_u32 m0, m0, 4096 // (op 54661) 
buffer_load_dwordx4 v44, s[8:11], 0 offen offset: 0 lds // (op 54661) Load value direct to lds







s_lshl_b32 s0, s28, 2 // (op 53689) 
s_mul_i32 s1, 3, s0 // (op 53689) 

v_add_u32 v29, v22, s1 // (op 53689) 


buffer_load_dword v32, v29, s[52:55], 0 offen // (op 53689) Load value


v_and_b32 v22, -64, v1 // (op 55218) 
v_sub_u32 v29, v1, v22 // (op 55218) 

v_lshlrev_b32 v22, 2, v29 // (op 55218) 

v_lshrrev_b32 v29, 6, v1 // (op 55218) 
v_lshl_add_u32 v34, v29, 8, v22 // (op 55218) 





v_add_u32 v22, 131072, v34 // (op 53692) 
s_waitcnt vmcnt(35)
ds_write_b32 v22, v33 // (op 53692) Store local data 


v_and_b32 v22, -64, v1 // (op 55160) 
v_sub_u32 v29, v1, v22 // (op 55160) 

v_lshlrev_b32 v22, 2, v29 // (op 55160) 

v_lshrrev_b32 v29, 6, v1 // (op 55160) 
v_lshl_add_u32 v34, v29, 8, v22 // (op 55160) 





v_add_u32 v22, 132096, v34 // (op 53694) 
s_waitcnt vmcnt(30)
ds_write_b32 v22, v36 // (op 53694) Store local data 


s_barrier  // (op 56221) 



ds_read_b64_tr_b4 v[42:43], v4 // (op 53757) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v4 offset:2048 // (op 53757) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 56256) 


v_add_u32 v22, 131072, v5 // (op 53837) 
ds_read_u8 v29, v22 // (op 53837) Load local data 




ds_read_b64_tr_b4 v[46:47], v4 offset:16 // (op 53762) Transpose load local data 
ds_read_b64_tr_b4 v[48:49], v4 offset:2064 // (op 53762) Transpose load local data 


v_add_u32 v22, 131104, v5 // (op 53842) 
ds_read_u8 v34, v22 // (op 53842) Load local data 




ds_read_b64_tr_b4 v[50:51], v4 offset:32 // (op 53767) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:2080 // (op 53767) Transpose load local data 


v_add_u32 v22, 131136, v5 // (op 53847) 
ds_read_u8 v41, v22 // (op 53847) Load local data 




ds_read_b64_tr_b4 v[54:55], v4 offset:48 // (op 53772) Transpose load local data 
ds_read_b64_tr_b4 v[56:57], v4 offset:2096 // (op 53772) Transpose load local data 


v_add_u32 v22, 131168, v5 // (op 53852) 
ds_read_u8 v24, v22 // (op 53852) Load local data 




ds_read_b64_tr_b4 v[58:59], v4 offset:64 // (op 53777) Transpose load local data 
ds_read_b64_tr_b4 v[60:61], v4 offset:2112 // (op 53777) Transpose load local data 


v_add_u32 v22, 131200, v5 // (op 53857) 
ds_read_u8 v25, v22 // (op 53857) Load local data 




ds_read_b64_tr_b4 v[62:63], v4 offset:80 // (op 53782) Transpose load local data 
ds_read_b64_tr_b4 v[64:65], v4 offset:2128 // (op 53782) Transpose load local data 


v_add_u32 v22, 131232, v5 // (op 53862) 
ds_read_u8 v66, v22 // (op 53862) Load local data 




ds_read_b64_tr_b4 v[68:69], v4 offset:96 // (op 53787) Transpose load local data 
ds_read_b64_tr_b4 v[70:71], v4 offset:2144 // (op 53787) Transpose load local data 


v_add_u32 v22, 131264, v5 // (op 53867) 
ds_read_u8 v67, v22 // (op 53867) Load local data 




ds_read_b64_tr_b4 v[72:73], v4 offset:112 // (op 53792) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v4 offset:2160 // (op 53792) Transpose load local data 


v_add_u32 v22, 131296, v5 // (op 53872) 
ds_read_u8 v76, v22 // (op 53872) Load local data 

s_barrier  // (op 56205) 



ds_read_b64_tr_b4 v[78:79], v3 offset:16384 // (op 53717) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v3 offset:18432 // (op 53717) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 56238) 


v_add_u32 v22, 132096, v0 // (op 53797) 
ds_read_u8 v77, v22 // (op 53797) Load local data 




ds_read_b64_tr_b4 v[82:83], v3 offset:16400 // (op 53722) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v3 offset:18448 // (op 53722) Transpose load local data 


v_add_u32 v22, 132128, v0 // (op 53802) 
ds_read_u8 v86, v22 // (op 53802) Load local data 




ds_read_b64_tr_b4 v[88:89], v3 offset:16416 // (op 53727) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v3 offset:18464 // (op 53727) Transpose load local data 


v_add_u32 v22, 132160, v0 // (op 53807) 
ds_read_u8 v87, v22 // (op 53807) Load local data 




ds_read_b64_tr_b4 v[92:93], v3 offset:16432 // (op 53732) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v3 offset:18480 // (op 53732) Transpose load local data 


v_add_u32 v22, 132192, v0 // (op 53812) 
ds_read_u8 v96, v22 // (op 53812) Load local data 




ds_read_b64_tr_b4 v[98:99], v3 offset:16448 // (op 53737) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v3 offset:18496 // (op 53737) Transpose load local data 


v_add_u32 v22, 132224, v0 // (op 53817) 
ds_read_u8 v97, v22 // (op 53817) Load local data 




ds_read_b64_tr_b4 v[102:103], v3 offset:16464 // (op 53742) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v3 offset:18512 // (op 53742) Transpose load local data 


v_add_u32 v22, 132256, v0 // (op 53822) 
ds_read_u8 v106, v22 // (op 53822) Load local data 




ds_read_b64_tr_b4 v[108:109], v3 offset:16480 // (op 53747) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v3 offset:18528 // (op 53747) Transpose load local data 


v_add_u32 v22, 132288, v0 // (op 53827) 
ds_read_u8 v107, v22 // (op 53827) Load local data 




ds_read_b64_tr_b4 v[112:113], v3 offset:16496 // (op 53752) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v3 offset:18544 // (op 53752) Transpose load local data 


v_add_u32 v22, 132320, v0 // (op 53832) 
ds_read_u8 v116, v22 // (op 53832) Load local data 

v_and_b32 v22, -8, v1 // (op 55017) 
v_sub_u32 v117, v1, v22 // (op 55017) 

v_lshlrev_b32 v22, 5, v117 // (op 55017) 

v_lshrrev_b32 v117, 3, v1 // (op 55017) 
v_lshl_add_u32 v118, v117, 8, v22 // (op 55017) 


v_lshlrev_b32 v22, 2, v118 // (op 55017) 

v_lshrrev_b32 v117, 3, v22 // (op 55017) 

s_nop 0
v_readfirstlane_b32 s0, v117 // (op 55017) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 55074) 
v_sub_u32 v117, v1, v22 // (op 55074) 

v_lshlrev_b32 v22, 5, v117 // (op 55074) 

v_lshrrev_b32 v117, 3, v1 // (op 55074) 
v_lshl_add_u32 v118, v117, 8, v22 // (op 55074) 


v_lshlrev_b32 v22, 2, v118 // (op 55074) 

v_lshrrev_b32 v117, 3, v22 // (op 55074) 

s_nop 0
v_readfirstlane_b32 s1, v117 // (op 55074) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 55131) 
v_sub_u32 v117, v1, v22 // (op 55131) 

v_lshlrev_b32 v22, 2, v117 // (op 55131) 

v_lshrrev_b32 v117, 6, v1 // (op 55131) 
v_lshl_add_u32 v118, v117, 8, v22 // (op 55131) 


v_and_b32 v22, -64, v1 // (op 55189) 
v_sub_u32 v117, v1, v22 // (op 55189) 

v_lshlrev_b32 v22, 2, v117 // (op 55189) 

v_lshrrev_b32 v117, 6, v1 // (op 55189) 
v_lshl_add_u32 v119, v117, 8, v22 // (op 55189) 


v_and_b32 v22, -8, v1 // (op 55270) 
v_sub_u32 v117, v1, v22 // (op 55270) 

v_lshlrev_b32 v22, 5, v117 // (op 55270) 

v_lshrrev_b32 v117, 3, v1 // (op 55270) 
v_lshl_add_u32 v120, v117, 8, v22 // (op 55270) 


v_lshlrev_b32 v22, 2, v120 // (op 55270) 

v_lshrrev_b32 v117, 3, v22 // (op 55270) 

s_nop 0
v_readfirstlane_b32 s6, v117 // (op 55270) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 55314) 
v_sub_u32 v117, v1, v22 // (op 55314) 

v_lshlrev_b32 v22, 5, v117 // (op 55314) 

v_lshrrev_b32 v117, 3, v1 // (op 55314) 
v_lshl_add_u32 v120, v117, 8, v22 // (op 55314) 


v_lshlrev_b32 v22, 2, v120 // (op 55314) 

v_lshrrev_b32 v117, 3, v22 // (op 55314) 

s_nop 0
v_readfirstlane_b32 s7, v117 // (op 55314) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 55358) 
v_sub_u32 v117, v1, v22 // (op 55358) 

v_lshlrev_b32 v22, 2, v117 // (op 55358) 

v_lshrrev_b32 v117, 6, v1 // (op 55358) 
v_lshl_add_u32 v120, v117, 8, v22 // (op 55358) 


v_and_b32 v22, -64, v1 // (op 55387) 
v_sub_u32 v117, v1, v22 // (op 55387) 

v_lshlrev_b32 v22, 2, v117 // (op 55387) 

v_lshrrev_b32 v117, 6, v1 // (op 55387) 
v_lshl_add_u32 v121, v117, 8, v22 // (op 55387) 


v_and_b32 v22, -8, v1 // (op 55416) 
v_sub_u32 v117, v1, v22 // (op 55416) 

v_lshlrev_b32 v22, 5, v117 // (op 55416) 

v_lshrrev_b32 v117, 3, v1 // (op 55416) 
v_lshl_add_u32 v122, v117, 8, v22 // (op 55416) 


v_lshlrev_b32 v22, 2, v122 // (op 55416) 

v_lshrrev_b32 v117, 3, v22 // (op 55416) 

s_nop 0
v_readfirstlane_b32 s14, v117 // (op 55416) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 55460) 
v_sub_u32 v117, v1, v22 // (op 55460) 

v_lshlrev_b32 v22, 5, v117 // (op 55460) 

v_lshrrev_b32 v117, 3, v1 // (op 55460) 
v_lshl_add_u32 v122, v117, 8, v22 // (op 55460) 


v_lshlrev_b32 v22, 2, v122 // (op 55460) 

v_lshrrev_b32 v117, 3, v22 // (op 55460) 

s_nop 0
v_readfirstlane_b32 s15, v117 // (op 55460) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 55504) 
v_sub_u32 v117, v1, v22 // (op 55504) 

v_lshlrev_b32 v22, 2, v117 // (op 55504) 

v_lshrrev_b32 v117, 6, v1 // (op 55504) 
v_lshl_add_u32 v122, v117, 8, v22 // (op 55504) 


v_and_b32 v22, -64, v1 // (op 55533) 
v_sub_u32 v117, v1, v22 // (op 55533) 

v_lshlrev_b32 v22, 2, v117 // (op 55533) 

v_lshrrev_b32 v117, 6, v1 // (op 55533) 
v_lshl_add_u32 v123, v117, 8, v22 // (op 55533) 


v_and_b32 v22, -8, v1 // (op 55562) 
v_sub_u32 v117, v1, v22 // (op 55562) 

v_lshlrev_b32 v22, 5, v117 // (op 55562) 

v_lshrrev_b32 v117, 3, v1 // (op 55562) 
v_lshl_add_u32 v124, v117, 8, v22 // (op 55562) 


v_lshlrev_b32 v22, 2, v124 // (op 55562) 

v_lshrrev_b32 v117, 3, v22 // (op 55562) 

s_nop 0
v_readfirstlane_b32 s18, v117 // (op 55562) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 55606) 
v_sub_u32 v117, v1, v22 // (op 55606) 

v_lshlrev_b32 v22, 5, v117 // (op 55606) 

v_lshrrev_b32 v117, 3, v1 // (op 55606) 
v_lshl_add_u32 v124, v117, 8, v22 // (op 55606) 


v_lshlrev_b32 v22, 2, v124 // (op 55606) 

v_lshrrev_b32 v117, 3, v22 // (op 55606) 

s_nop 0
v_readfirstlane_b32 s19, v117 // (op 55606) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -64, v1 // (op 55650) 
v_sub_u32 v117, v1, v22 // (op 55650) 

v_lshlrev_b32 v22, 2, v117 // (op 55650) 

v_lshrrev_b32 v117, 6, v1 // (op 55650) 
v_lshl_add_u32 v124, v117, 8, v22 // (op 55650) 


v_and_b32 v22, -64, v1 // (op 55679) 
v_sub_u32 v117, v1, v22 // (op 55679) 

v_lshlrev_b32 v22, 2, v117 // (op 55679) 

v_lshrrev_b32 v117, 6, v1 // (op 55679) 
v_lshl_add_u32 v125, v117, 8, v22 // (op 55679) 


s_mov_b32 s38, 0 // (op 18527) call()
s_ashr_i64 s[60:61], s[22:23], 7 // (op 18526) 
s_mov_b32 s39, s60 // (op 18526) convert

s_and_b32 s60, s39, 4294967292 // (op 18526) 

s_cmp_lt_u32 s38, s60 // (op 18526) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoop_18526 // (op 18526) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoop_18526 if false)
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoop_18526:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[42:45], a[0:3], v77, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 88) 
s_barrier  // (op 56224) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[46:49], a[4:7], v77, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 522) 
s_add_u32 s38, s38, 4 // (op 18528) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[50:53], a[8:11], v77, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 820) 
s_lshl_b64 s[60:61], s[24:25], 5 // (op 54593) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:81], v[54:57], a[12:15], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1118) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54593) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[58:61], a[16:19], v77, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1416) 
v_lshrrev_b64 v[126:127], 3, s[62:63] // (op 54593) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[62:65], a[20:23], v77, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1714) 
v_mov_b64 v[128:129], v[30:31] // (op 54593) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[68:71], a[24:27], v77, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2012) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54593) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[72:75], a[28:31], v77, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2310) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54593) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[42:45], a[32:35], v86, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3741) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54593) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[46:49], a[36:39], v86, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3649) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54593) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v86, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3557) 
v_mov_b32 v22, s62 // (op 54593) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[54:57], a[44:47], v86, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3465) 
v_mov_b32 v117, s63 // (op 54593) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[82:85], v[58:61], a[48:51], v86, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3373) 
v_add_co_u32 v130, s[60:61], v128, v22 // (op 54593) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[82:85], v[62:65], a[52:55], v86, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3281) 
v_addc_co_u32 v131, s[60:61], v129, v117, s[60:61] // (op 54593) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[82:85], v[68:71], a[56:59], v86, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3189) 
s_mov_b32 m0, s1 // (op 54593) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[82:85], v[72:75], a[60:63], v86, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3097) 
buffer_load_dwordx4 v130, s[48:51], 0 offen offset: 0 lds // (op 54593) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[42:45], a[64:67], v87, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6319) 
v_add_co_u32 v130, s[60:61], v130, v126 // (op 54593) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[46:49], a[68:71], v87, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6227) 
v_addc_co_u32 v131, s[60:61], 0, v131, s[60:61] // (op 54593) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:91], v[50:53], a[72:75], v87, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6135) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[88:91], v[54:57], a[76:79], v87, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6043) 
s_add_u32 m0, m0, 4096 // (op 54593) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[88:91], v[58:61], a[80:83], v87, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5951) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[88:91], v[62:65], a[84:87], v87, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5859) 
buffer_load_dwordx4 v130, s[48:51], 0 offen offset: 0 lds // (op 54593) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[88:91], v[68:71], a[88:91], v87, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5767) 
v_add_co_u32 v130, s[60:61], v130, v126 // (op 54593) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[88:91], v[72:75], a[92:95], v87, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5675) 
v_addc_co_u32 v131, s[60:61], 0, v131, s[60:61] // (op 54593) most significant half

s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[92:95], v[42:45], a[96:99], v96, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8897) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[92:95], v[46:49], a[100:103], v96, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8805) 
s_add_u32 m0, m0, 4096 // (op 54593) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[92:95], v[50:53], a[104:107], v96, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8713) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:95], v[54:57], a[108:111], v96, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8621) 
buffer_load_dwordx4 v130, s[48:51], 0 offen offset: 0 lds // (op 54593) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:95], v[58:61], a[112:115], v96, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8529) 
v_add_co_u32 v130, s[60:61], v130, v126 // (op 54593) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:95], v[62:65], a[116:119], v96, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8437) 
v_addc_co_u32 v131, s[60:61], 0, v131, s[60:61] // (op 54593) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[92:95], v[68:71], a[120:123], v96, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8345) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[92:95], v[72:75], a[124:127], v96, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8253) 
s_add_u32 m0, m0, 4096 // (op 54593) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[98:101], v[42:45], a[128:131], v97, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11475) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[98:101], v[46:49], a[132:135], v97, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11383) 
buffer_load_dwordx4 v130, s[48:51], 0 offen offset: 0 lds // (op 54593) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[98:101], v[50:53], a[136:139], v97, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11291) 
v_mov_b32 v22, v35 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[98:101], v[54:57], a[140:143], v97, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11199) 
s_lshl_b32 s39, s34, 4 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[98:101], v[58:61], a[144:147], v97, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11107) 
v_add_u32 v117, v22, s39 // (op 25) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[98:101], v[62:65], a[148:151], v97, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11015) 
buffer_load_dword v33, v117, s[56:59], 0 offen // (op 25) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[98:101], v[68:71], a[152:155], v97, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10923) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 56208) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[98:101], v[72:75], a[156:159], v97, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10831) 
s_lshl_b64 s[60:61], s[12:13], 5 // (op 54588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[102:105], v[42:45], a[160:163], v106, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14053) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54588) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[102:105], v[46:49], a[164:167], v106, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13961) 
v_lshrrev_b64 v[126:127], 3, s[62:63] // (op 54588) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[102:105], v[50:53], a[168:171], v106, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13869) 
v_mov_b64 v[130:131], v[26:27] // (op 54588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[102:105], v[54:57], a[172:175], v106, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13777) 
s_lshl_b64 s[60:61], s[12:13], 7 // (op 54588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[102:105], v[58:61], a[176:179], v106, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13685) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54588) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[102:105], v[62:65], a[180:183], v106, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13593) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54588) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[102:105], v[68:71], a[184:187], v106, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13501) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54588) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[102:105], v[72:75], a[188:191], v106, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13409) 
v_mov_b32 v117, s62 // (op 54588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[108:111], v[42:45], a[192:195], v107, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16631) 
v_mov_b32 v132, s63 // (op 54588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[108:111], v[46:49], a[196:199], v107, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16539) 
v_add_co_u32 v134, s[60:61], v130, v117 // (op 54588) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:111], v[50:53], a[200:203], v107, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16447) 
v_addc_co_u32 v135, s[60:61], v131, v132, s[60:61] // (op 54588) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[108:111], v[54:57], a[204:207], v107, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16355) 
s_add_u32 s39, 16384, s0 // (op 54588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[108:111], v[58:61], a[208:211], v107, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16263) 
s_mov_b32 m0, s39 // (op 54588) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[108:111], v[62:65], a[212:215], v107, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16171) 
buffer_load_dwordx4 v134, s[8:11], 0 offen offset: 0 lds // (op 54588) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[108:111], v[68:71], a[216:219], v107, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16079) 
v_add_co_u32 v134, s[60:61], v134, v126 // (op 54588) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[108:111], v[72:75], a[220:223], v107, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15987) 
v_addc_co_u32 v135, s[60:61], 0, v135, s[60:61] // (op 54588) most significant half

s_add_u32 m0, m0, 4096 // (op 54588) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:115], v[42:45], a[224:227], v116, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19209) 
buffer_load_dwordx4 v134, s[8:11], 0 offen offset: 0 lds // (op 54588) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[112:115], v[46:49], a[228:231], v116, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19117) 
v_add_co_u32 v134, s[60:61], v134, v126 // (op 54588) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[112:115], v[50:53], a[232:235], v116, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19025) 
v_addc_co_u32 v135, s[60:61], 0, v135, s[60:61] // (op 54588) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[112:115], v[54:57], a[236:239], v116, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18933) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[112:115], v[58:61], a[240:243], v116, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18841) 
s_add_u32 m0, m0, 4096 // (op 54588) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[112:115], v[62:65], a[244:247], v116, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18749) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[112:115], v[68:71], a[248:251], v116, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18657) 
buffer_load_dwordx4 v134, s[8:11], 0 offen offset: 0 lds // (op 54588) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[112:115], v[72:75], a[252:255], v116, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18565) 
v_add_co_u32 v134, s[60:61], v134, v126 // (op 54588) least significant half
v_addc_co_u32 v135, s[60:61], 0, v135, s[60:61] // (op 54588) most significant half

s_add_u32 m0, m0, 4096 // (op 54588) 
buffer_load_dwordx4 v134, s[8:11], 0 offen offset: 0 lds // (op 54588) Load value direct to lds





v_mov_b32 v117, v23 // (op 17) 
s_lshl_b32 s39, s28, 4 // (op 17) 
v_add_u32 v126, v117, s39 // (op 17) 


buffer_load_dword v36, v126, s[52:55], 0 offen // (op 17) Load value

s_barrier  // (op 56296) 



v_add_u32 v126, 133120, v121 // (op 255) 
s_waitcnt vmcnt(35)
ds_write_b32 v126, v37 // (op 255) Store local data 

s_barrier  // (op 56290) 



v_add_u32 v126, 134144, v120 // (op 254) 
s_waitcnt vmcnt(30)
ds_write_b32 v126, v38 // (op 254) Store local data 

s_barrier  // (op 56279) 



ds_read_b64_tr_b4 v[132:133], v8 offset:32768 // (op 247) Transpose load local data 
ds_read_b64_tr_b4 v[134:135], v8 offset:34816 // (op 247) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 56293) 


v_add_u32 v126, 133120, v9 // (op 251) 
ds_read_u8 v127, v126 // (op 251) Load local data 




ds_read_b64_tr_b4 v[136:137], v8 offset:32784 // (op 529) Transpose load local data 
ds_read_b64_tr_b4 v[138:139], v8 offset:34832 // (op 529) Transpose load local data 


v_add_u32 v126, 133152, v9 // (op 533) 
ds_read_u8 v86, v126 // (op 533) Load local data 




ds_read_b64_tr_b4 v[82:83], v8 offset:32800 // (op 827) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v8 offset:34848 // (op 827) Transpose load local data 


v_add_u32 v126, 133184, v9 // (op 831) 
ds_read_u8 v87, v126 // (op 831) Load local data 




ds_read_b64_tr_b4 v[88:89], v8 offset:32816 // (op 1125) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v8 offset:34864 // (op 1125) Transpose load local data 


v_add_u32 v126, 133216, v9 // (op 1129) 
ds_read_u8 v96, v126 // (op 1129) Load local data 




ds_read_b64_tr_b4 v[92:93], v8 offset:32832 // (op 1423) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v8 offset:34880 // (op 1423) Transpose load local data 


v_add_u32 v126, 133248, v9 // (op 1427) 
ds_read_u8 v97, v126 // (op 1427) Load local data 




ds_read_b64_tr_b4 v[98:99], v8 offset:32848 // (op 1721) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v8 offset:34896 // (op 1721) Transpose load local data 


v_add_u32 v126, 133280, v9 // (op 1725) 
ds_read_u8 v106, v126 // (op 1725) Load local data 




ds_read_b64_tr_b4 v[102:103], v8 offset:32864 // (op 2019) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v8 offset:34912 // (op 2019) Transpose load local data 


v_add_u32 v126, 133312, v9 // (op 2023) 
ds_read_u8 v107, v126 // (op 2023) Load local data 




ds_read_b64_tr_b4 v[108:109], v8 offset:32880 // (op 2317) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v8 offset:34928 // (op 2317) Transpose load local data 


v_add_u32 v126, 133344, v9 // (op 2321) 
ds_read_u8 v140, v126 // (op 2321) Load local data 

s_barrier  // (op 56273) 
s_barrier  // (op 56401) 



ds_read_b64_tr_b4 v[78:79], v7 offset:49152 // (op 245) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v7 offset:51200 // (op 245) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 56286) 


v_add_u32 v126, 134144, v6 // (op 249) 
ds_read_u8 v77, v126 // (op 249) Load local data 




ds_read_b64_tr_b4 v[142:143], v7 offset:49168 // (op 3102) Transpose load local data 
ds_read_b64_tr_b4 v[144:145], v7 offset:51216 // (op 3102) Transpose load local data 


v_add_u32 v126, 134176, v6 // (op 3105) 
ds_read_u8 v34, v126 // (op 3105) Load local data 




ds_read_b64_tr_b4 v[146:147], v7 offset:49184 // (op 5680) Transpose load local data 
ds_read_b64_tr_b4 v[148:149], v7 offset:51232 // (op 5680) Transpose load local data 


v_add_u32 v126, 134208, v6 // (op 5683) 
ds_read_u8 v41, v126 // (op 5683) Load local data 




ds_read_b64_tr_b4 v[150:151], v7 offset:49200 // (op 8258) Transpose load local data 
ds_read_b64_tr_b4 v[152:153], v7 offset:51248 // (op 8258) Transpose load local data 


v_add_u32 v126, 134240, v6 // (op 8261) 
ds_read_u8 v24, v126 // (op 8261) Load local data 




ds_read_b64_tr_b4 v[154:155], v7 offset:49216 // (op 10836) Transpose load local data 
ds_read_b64_tr_b4 v[156:157], v7 offset:51264 // (op 10836) Transpose load local data 


v_add_u32 v126, 134272, v6 // (op 10839) 
ds_read_u8 v25, v126 // (op 10839) Load local data 




ds_read_b64_tr_b4 v[158:159], v7 offset:49232 // (op 13414) Transpose load local data 
ds_read_b64_tr_b4 v[160:161], v7 offset:51280 // (op 13414) Transpose load local data 


v_add_u32 v126, 134304, v6 // (op 13417) 
ds_read_u8 v66, v126 // (op 13417) Load local data 




ds_read_b64_tr_b4 v[162:163], v7 offset:49248 // (op 15992) Transpose load local data 
ds_read_b64_tr_b4 v[164:165], v7 offset:51296 // (op 15992) Transpose load local data 


v_add_u32 v126, 134336, v6 // (op 15995) 
ds_read_u8 v67, v126 // (op 15995) Load local data 




ds_read_b64_tr_b4 v[166:167], v7 offset:49264 // (op 18570) Transpose load local data 
ds_read_b64_tr_b4 v[168:169], v7 offset:51312 // (op 18570) Transpose load local data 


v_add_u32 v126, 134368, v6 // (op 18573) 
ds_read_u8 v76, v126 // (op 18573) Load local data 

s_barrier  // (op 56499) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[132:135], a[0:3], v77, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 252) 
s_barrier  // (op 56283) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[136:139], a[4:7], v77, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 523) 
s_lshl_b64 s[60:61], s[24:25], 5 // (op 54616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[82:85], a[8:11], v77, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 821) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54616) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:81], v[88:91], a[12:15], v77, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1119) 
v_lshrrev_b64 v[170:171], 3, s[62:63] // (op 54616) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[92:95], a[16:19], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1417) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[98:101], a[20:23], v77, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1715) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54616) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[102:105], a[24:27], v77, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2013) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54616) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[108:111], a[28:31], v77, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2311) 
s_mul_i32 s63, 5, s61 // (op 54616) most significant: low of low * high
s_mul_i32 s62, 5, s60 // (op 54616) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[142:145], v[132:135], a[32:35], v34, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3742) 
s_mul_hi_u32 s39, 5, s60 // (op 54616) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[142:145], v[136:139], a[36:39], v34, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3650) 
s_add_u32 s63, s63, s39 // (op 54616) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[142:145], v[82:85], a[40:43], v34, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3558) 
v_mov_b32 v126, s62 // (op 54616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[142:145], v[88:91], a[44:47], v34, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3466) 
v_mov_b32 v141, s63 // (op 54616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[142:145], v[92:95], a[48:51], v34, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3374) 
v_add_co_u32 v172, s[60:61], v128, v126 // (op 54616) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[142:145], v[98:101], a[52:55], v34, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3282) 
v_addc_co_u32 v173, s[60:61], v129, v141, s[60:61] // (op 54616) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[142:145], v[102:105], a[56:59], v34, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3190) 
s_add_u32 s39, 32768, s7 // (op 54616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[142:145], v[108:111], a[60:63], v34, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3098) 
s_mov_b32 m0, s39 // (op 54616) reg expression

buffer_load_dwordx4 v172, s[48:51], 0 offen offset: 0 lds // (op 54616) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[146:149], v[132:135], a[64:67], v41, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6320) 
v_add_co_u32 v172, s[60:61], v172, v170 // (op 54616) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[146:149], v[136:139], a[68:71], v41, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6228) 
v_addc_co_u32 v173, s[60:61], 0, v173, s[60:61] // (op 54616) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[146:149], v[82:85], a[72:75], v41, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6136) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[146:149], v[88:91], a[76:79], v41, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6044) 
s_add_u32 m0, m0, 4096 // (op 54616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[146:149], v[92:95], a[80:83], v41, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5952) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[146:149], v[98:101], a[84:87], v41, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5860) 
buffer_load_dwordx4 v172, s[48:51], 0 offen offset: 0 lds // (op 54616) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[146:149], v[102:105], a[88:91], v41, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5768) 
v_add_co_u32 v172, s[60:61], v172, v170 // (op 54616) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[146:149], v[108:111], a[92:95], v41, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5676) 
v_addc_co_u32 v173, s[60:61], 0, v173, s[60:61] // (op 54616) most significant half


s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[150:153], v[132:135], a[96:99], v24, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8898) 
s_add_u32 m0, m0, 4096 // (op 54616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[150:153], v[136:139], a[100:103], v24, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8806) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[150:153], v[82:85], a[104:107], v24, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8714) 
buffer_load_dwordx4 v172, s[48:51], 0 offen offset: 0 lds // (op 54616) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[150:153], v[88:91], a[108:111], v24, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8622) 
v_add_co_u32 v172, s[60:61], v172, v170 // (op 54616) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[150:153], v[92:95], a[112:115], v24, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8530) 
v_addc_co_u32 v173, s[60:61], 0, v173, s[60:61] // (op 54616) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[150:153], v[98:101], a[116:119], v24, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8438) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[150:153], v[102:105], a[120:123], v24, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8346) 
s_add_u32 m0, m0, 4096 // (op 54616) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[150:153], v[108:111], a[124:127], v24, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8254) 

buffer_load_dwordx4 v172, s[48:51], 0 offen offset: 0 lds // (op 54616) Load value direct to lds





s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[154:157], v[132:135], a[128:131], v25, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11476) 
s_lshl_b32 s39, s34, 2 // (op 258) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[154:157], v[136:139], a[132:135], v25, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11384) 
s_mul_i32 s60, 5, s39 // (op 258) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[154:157], v[82:85], a[136:139], v25, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11292) 
v_add_u32 v126, v22, s60 // (op 258) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[154:157], v[88:91], a[140:143], v25, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11200) 
buffer_load_dword v37, v126, s[56:59], 0 offen // (op 258) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[154:157], v[92:95], a[144:147], v25, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11108) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 56276) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[154:157], v[98:101], a[148:151], v25, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11016) 
s_lshl_b64 s[60:61], s[12:13], 5 // (op 54611) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[154:157], v[102:105], a[152:155], v25, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10924) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54611) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[154:157], v[108:111], a[156:159], v25, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10832) 
v_lshrrev_b64 v[152:153], 3, s[62:63] // (op 54611) 



s_lshl_b64 s[60:61], s[12:13], 7 // (op 54611) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[158:161], v[132:135], a[160:163], v66, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14054) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54611) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[158:161], v[136:139], a[164:167], v66, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13962) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54611) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[158:161], v[82:85], a[168:171], v66, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13870) 
s_mul_i32 s63, 5, s61 // (op 54611) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[158:161], v[88:91], a[172:175], v66, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13778) 
s_mul_i32 s62, 5, s60 // (op 54611) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[158:161], v[92:95], a[176:179], v66, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13686) 
s_mul_hi_u32 s39, 5, s60 // (op 54611) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[158:161], v[98:101], a[180:183], v66, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13594) 
s_add_u32 s63, s63, s39 // (op 54611) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[158:161], v[102:105], a[184:187], v66, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13502) 
v_mov_b32 v126, s62 // (op 54611) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[158:161], v[108:111], a[188:191], v66, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13410) 
v_mov_b32 v141, s63 // (op 54611) 

v_add_co_u32 v142, s[60:61], v130, v126 // (op 54611) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[162:165], v[132:135], a[192:195], v67, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16632) 
v_addc_co_u32 v143, s[60:61], v131, v141, s[60:61] // (op 54611) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[162:165], v[136:139], a[196:199], v67, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16540) 
s_add_u32 s39, 49152, s6 // (op 54611) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[162:165], v[82:85], a[200:203], v67, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16448) 
s_mov_b32 m0, s39 // (op 54611) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[162:165], v[88:91], a[204:207], v67, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16356) 
buffer_load_dwordx4 v142, s[8:11], 0 offen offset: 0 lds // (op 54611) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[162:165], v[92:95], a[208:211], v67, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16264) 
v_add_co_u32 v142, s[60:61], v142, v152 // (op 54611) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[162:165], v[98:101], a[212:215], v67, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16172) 
v_addc_co_u32 v143, s[60:61], 0, v143, s[60:61] // (op 54611) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[162:165], v[102:105], a[216:219], v67, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16080) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[162:165], v[108:111], a[220:223], v67, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15988) 
s_add_u32 m0, m0, 4096 // (op 54611) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[166:169], v[132:135], a[224:227], v76, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19210) 
buffer_load_dwordx4 v142, s[8:11], 0 offen offset: 0 lds // (op 54611) Load value direct to lds


v_add_co_u32 v142, s[60:61], v142, v152 // (op 54611) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[166:169], v[136:139], a[228:231], v76, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19118) 
v_addc_co_u32 v143, s[60:61], 0, v143, s[60:61] // (op 54611) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[166:169], v[82:85], a[232:235], v76, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19026) 
s_add_u32 m0, m0, 4096 // (op 54611) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[166:169], v[88:91], a[236:239], v76, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18934) 
buffer_load_dwordx4 v142, s[8:11], 0 offen offset: 0 lds // (op 54611) Load value direct to lds
v_add_co_u32 v142, s[60:61], v142, v152 // (op 54611) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[166:169], v[92:95], a[240:243], v76, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18842) 
v_addc_co_u32 v143, s[60:61], 0, v143, s[60:61] // (op 54611) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:169], v[98:101], a[244:247], v76, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18750) 
s_add_u32 m0, m0, 4096 // (op 54611) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[166:169], v[102:105], a[248:251], v76, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18658) 
buffer_load_dwordx4 v142, s[8:11], 0 offen offset: 0 lds // (op 54611) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[166:169], v[108:111], a[252:255], v76, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18566) 
s_lshl_b32 s39, s28, 2 // (op 257) 


s_mul_i32 s60, 5, s39 // (op 257) 

v_add_u32 v126, v117, s60 // (op 257) 


buffer_load_dword v38, v126, s[52:55], 0 offen // (op 257) Load value

s_barrier  // (op 56322) 



v_add_u32 v126, 133120, v123 // (op 295) 
s_waitcnt vmcnt(35)
ds_write_b32 v126, v39 // (op 295) Store local data 

s_barrier  // (op 56316) 



v_add_u32 v126, 134144, v122 // (op 294) 
s_waitcnt vmcnt(30)
ds_write_b32 v126, v40 // (op 294) Store local data 

s_barrier  // (op 56305) 



v_add_u32 v126, 65536, v12 // (op 287) 
ds_read_b64_tr_b4 v[42:43], v126 // (op 287) Transpose load local data 

v_add_u32 v126, 67584, v12 // (op 287) 
ds_read_b64_tr_b4 v[44:45], v126 // (op 287) Transpose load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 56319) 


v_add_u32 v126, 133120, v13 // (op 291) 
ds_read_u8 v29, v126 // (op 291) Load local data 




v_add_u32 v126, 65552, v12 // (op 537) 
ds_read_b64_tr_b4 v[46:47], v126 // (op 537) Transpose load local data 

v_add_u32 v126, 67600, v12 // (op 537) 
ds_read_b64_tr_b4 v[48:49], v126 // (op 537) Transpose load local data 



v_add_u32 v126, 133152, v13 // (op 541) 
ds_read_u8 v34, v126 // (op 541) Load local data 




v_add_u32 v126, 65568, v12 // (op 835) 
ds_read_b64_tr_b4 v[50:51], v126 // (op 835) Transpose load local data 

v_add_u32 v126, 67616, v12 // (op 835) 
ds_read_b64_tr_b4 v[52:53], v126 // (op 835) Transpose load local data 



v_add_u32 v126, 133184, v13 // (op 839) 
ds_read_u8 v41, v126 // (op 839) Load local data 




v_add_u32 v126, 65584, v12 // (op 1133) 
ds_read_b64_tr_b4 v[54:55], v126 // (op 1133) Transpose load local data 

v_add_u32 v126, 67632, v12 // (op 1133) 
ds_read_b64_tr_b4 v[56:57], v126 // (op 1133) Transpose load local data 



v_add_u32 v126, 133216, v13 // (op 1137) 
ds_read_u8 v24, v126 // (op 1137) Load local data 




v_add_u32 v126, 65600, v12 // (op 1431) 
ds_read_b64_tr_b4 v[58:59], v126 // (op 1431) Transpose load local data 

v_add_u32 v126, 67648, v12 // (op 1431) 
ds_read_b64_tr_b4 v[60:61], v126 // (op 1431) Transpose load local data 



v_add_u32 v126, 133248, v13 // (op 1435) 
ds_read_u8 v25, v126 // (op 1435) Load local data 




v_add_u32 v126, 65616, v12 // (op 1729) 
ds_read_b64_tr_b4 v[62:63], v126 // (op 1729) Transpose load local data 

v_add_u32 v126, 67664, v12 // (op 1729) 
ds_read_b64_tr_b4 v[64:65], v126 // (op 1729) Transpose load local data 



v_add_u32 v126, 133280, v13 // (op 1733) 
ds_read_u8 v66, v126 // (op 1733) Load local data 




v_add_u32 v126, 65632, v12 // (op 2027) 
ds_read_b64_tr_b4 v[68:69], v126 // (op 2027) Transpose load local data 

v_add_u32 v126, 67680, v12 // (op 2027) 
ds_read_b64_tr_b4 v[70:71], v126 // (op 2027) Transpose load local data 



v_add_u32 v126, 133312, v13 // (op 2031) 
ds_read_u8 v67, v126 // (op 2031) Load local data 




v_add_u32 v126, 65648, v12 // (op 2325) 
ds_read_b64_tr_b4 v[72:73], v126 // (op 2325) Transpose load local data 

v_add_u32 v126, 67696, v12 // (op 2325) 
ds_read_b64_tr_b4 v[74:75], v126 // (op 2325) Transpose load local data 



v_add_u32 v126, 133344, v13 // (op 2329) 
ds_read_u8 v116, v126 // (op 2329) Load local data 

s_barrier  // (op 56299) 
s_barrier  // (op 56407) 



v_add_u32 v126, 81920, v11 // (op 285) 
ds_read_b64_tr_b4 v[78:79], v126 // (op 285) Transpose load local data 

v_add_u32 v126, 83968, v11 // (op 285) 
ds_read_b64_tr_b4 v[80:81], v126 // (op 285) Transpose load local data 

s_waitcnt lgkmcnt(15)
s_barrier  // (op 56312) 


v_add_u32 v126, 134144, v10 // (op 289) 
ds_read_u8 v77, v126 // (op 289) Load local data 




v_add_u32 v126, 81936, v11 // (op 3108) 
ds_read_b64_tr_b4 v[112:113], v126 // (op 3108) Transpose load local data 

v_add_u32 v126, 83984, v11 // (op 3108) 
ds_read_b64_tr_b4 v[114:115], v126 // (op 3108) Transpose load local data 



v_add_u32 v126, 134176, v10 // (op 3111) 
ds_read_u8 v86, v126 // (op 3111) Load local data 




v_add_u32 v126, 81952, v11 // (op 5686) 
ds_read_b64_tr_b4 v[132:133], v126 // (op 5686) Transpose load local data 

v_add_u32 v126, 84000, v11 // (op 5686) 
ds_read_b64_tr_b4 v[134:135], v126 // (op 5686) Transpose load local data 



v_add_u32 v126, 134208, v10 // (op 5689) 
ds_read_u8 v87, v126 // (op 5689) Load local data 




v_add_u32 v126, 81968, v11 // (op 8264) 
ds_read_b64_tr_b4 v[136:137], v126 // (op 8264) Transpose load local data 

v_add_u32 v126, 84016, v11 // (op 8264) 
ds_read_b64_tr_b4 v[138:139], v126 // (op 8264) Transpose load local data 



v_add_u32 v126, 134240, v10 // (op 8267) 
ds_read_u8 v96, v126 // (op 8267) Load local data 




v_add_u32 v126, 81984, v11 // (op 10842) 
ds_read_b64_tr_b4 v[140:141], v126 // (op 10842) Transpose load local data 

v_add_u32 v126, 84032, v11 // (op 10842) 
ds_read_b64_tr_b4 v[142:143], v126 // (op 10842) Transpose load local data 



v_add_u32 v126, 134272, v10 // (op 10845) 
ds_read_u8 v97, v126 // (op 10845) Load local data 




v_add_u32 v126, 82000, v11 // (op 13420) 
ds_read_b64_tr_b4 v[144:145], v126 // (op 13420) Transpose load local data 

v_add_u32 v126, 84048, v11 // (op 13420) 
ds_read_b64_tr_b4 v[146:147], v126 // (op 13420) Transpose load local data 



v_add_u32 v126, 134304, v10 // (op 13423) 
ds_read_u8 v106, v126 // (op 13423) Load local data 




v_add_u32 v126, 82016, v11 // (op 15998) 
ds_read_b64_tr_b4 v[148:149], v126 // (op 15998) Transpose load local data 

v_add_u32 v126, 84064, v11 // (op 15998) 
ds_read_b64_tr_b4 v[150:151], v126 // (op 15998) Transpose load local data 



v_add_u32 v126, 134336, v10 // (op 16001) 
ds_read_u8 v107, v126 // (op 16001) Load local data 




v_add_u32 v126, 82032, v11 // (op 18576) 
ds_read_b64_tr_b4 v[152:153], v126 // (op 18576) Transpose load local data 

v_add_u32 v126, 84080, v11 // (op 18576) 
ds_read_b64_tr_b4 v[154:155], v126 // (op 18576) Transpose load local data 



v_add_u32 v126, 134368, v10 // (op 18579) 
ds_read_u8 v127, v126 // (op 18579) Load local data 

s_barrier  // (op 56505) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[42:45], a[0:3], v77, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 292) 
s_barrier  // (op 56309) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[46:49], a[4:7], v77, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 524) 
s_lshl_b64 s[60:61], s[24:25], 5 // (op 54626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[50:53], a[8:11], v77, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 822) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54626) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:81], v[54:57], a[12:15], v77, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1120) 
v_lshrrev_b64 v[156:157], 3, s[62:63] // (op 54626) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[58:61], a[16:19], v77, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1418) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[62:65], a[20:23], v77, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1716) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54626) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[68:71], a[24:27], v77, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2014) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54626) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[72:75], a[28:31], v77, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2312) 
s_mul_i32 s63, 6, s61 // (op 54626) most significant: low of low * high
s_mul_i32 s62, 6, s60 // (op 54626) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[42:45], a[32:35], v86, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3743) 
s_mul_hi_u32 s39, 6, s60 // (op 54626) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[46:49], a[36:39], v86, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3651) 
s_add_u32 s63, s63, s39 // (op 54626) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[50:53], a[40:43], v86, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3559) 
v_mov_b32 v126, s62 // (op 54626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[54:57], a[44:47], v86, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3467) 
v_mov_b32 v158, s63 // (op 54626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[58:61], a[48:51], v86, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3375) 
v_add_co_u32 v160, s[60:61], v128, v126 // (op 54626) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[62:65], a[52:55], v86, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3283) 
v_addc_co_u32 v161, s[60:61], v129, v158, s[60:61] // (op 54626) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[112:115], v[68:71], a[56:59], v86, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3191) 
s_add_u32 s39, 65536, s15 // (op 54626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[112:115], v[72:75], a[60:63], v86, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3099) 
s_mov_b32 m0, s39 // (op 54626) reg expression
buffer_load_dwordx4 v160, s[48:51], 0 offen offset: 0 lds // (op 54626) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[132:135], v[42:45], a[64:67], v87, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6321) 
v_add_co_u32 v160, s[60:61], v160, v156 // (op 54626) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[46:49], a[68:71], v87, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6229) 
v_addc_co_u32 v161, s[60:61], 0, v161, s[60:61] // (op 54626) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[132:135], v[50:53], a[72:75], v87, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6137) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[132:135], v[54:57], a[76:79], v87, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6045) 
s_add_u32 m0, m0, 4096 // (op 54626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[132:135], v[58:61], a[80:83], v87, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5953) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[62:65], a[84:87], v87, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5861) 
buffer_load_dwordx4 v160, s[48:51], 0 offen offset: 0 lds // (op 54626) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[68:71], a[88:91], v87, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5769) 
v_add_co_u32 v160, s[60:61], v160, v156 // (op 54626) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[72:75], a[92:95], v87, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5677) 
v_addc_co_u32 v161, s[60:61], 0, v161, s[60:61] // (op 54626) most significant half


s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[42:45], a[96:99], v96, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8899) 
s_add_u32 m0, m0, 4096 // (op 54626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[46:49], a[100:103], v96, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8807) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[50:53], a[104:107], v96, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8715) 
buffer_load_dwordx4 v160, s[48:51], 0 offen offset: 0 lds // (op 54626) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[136:139], v[54:57], a[108:111], v96, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8623) 
v_add_co_u32 v160, s[60:61], v160, v156 // (op 54626) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[58:61], a[112:115], v96, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8531) 
v_addc_co_u32 v161, s[60:61], 0, v161, s[60:61] // (op 54626) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[62:65], a[116:119], v96, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8439) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[68:71], a[120:123], v96, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8347) 
s_add_u32 m0, m0, 4096 // (op 54626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[72:75], a[124:127], v96, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8255) 

buffer_load_dwordx4 v160, s[48:51], 0 offen offset: 0 lds // (op 54626) Load value direct to lds





s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[140:143], v[42:45], a[128:131], v97, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11477) 
s_lshl_b32 s39, s34, 2 // (op 298) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[140:143], v[46:49], a[132:135], v97, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11385) 
s_mul_i32 s60, 6, s39 // (op 298) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[140:143], v[50:53], a[136:139], v97, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11293) 
v_add_u32 v126, v22, s60 // (op 298) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[140:143], v[54:57], a[140:143], v97, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11201) 
buffer_load_dword v39, v126, s[56:59], 0 offen // (op 298) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[140:143], v[58:61], a[144:147], v97, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11109) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 56302) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[140:143], v[62:65], a[148:151], v97, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11017) 
s_lshl_b64 s[60:61], s[12:13], 5 // (op 54621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[140:143], v[68:71], a[152:155], v97, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10925) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54621) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[140:143], v[72:75], a[156:159], v97, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10833) 
v_lshrrev_b64 v[132:133], 3, s[62:63] // (op 54621) 



s_lshl_b64 s[60:61], s[12:13], 7 // (op 54621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[42:45], a[160:163], v106, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14055) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54621) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[144:147], v[46:49], a[164:167], v106, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13963) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54621) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[50:53], a[168:171], v106, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13871) 
s_mul_i32 s63, 6, s61 // (op 54621) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[54:57], a[172:175], v106, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13779) 
s_mul_i32 s62, 6, s60 // (op 54621) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[144:147], v[58:61], a[176:179], v106, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13687) 
s_mul_hi_u32 s39, 6, s60 // (op 54621) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[144:147], v[62:65], a[180:183], v106, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13595) 
s_add_u32 s63, s63, s39 // (op 54621) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:147], v[68:71], a[184:187], v106, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13503) 
v_mov_b32 v126, s62 // (op 54621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[144:147], v[72:75], a[188:191], v106, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13411) 
v_mov_b32 v134, s63 // (op 54621) 

v_add_co_u32 v146, s[60:61], v130, v126 // (op 54621) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[148:151], v[42:45], a[192:195], v107, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16633) 
v_addc_co_u32 v147, s[60:61], v131, v134, s[60:61] // (op 54621) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[148:151], v[46:49], a[196:199], v107, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16541) 
s_add_u32 s39, 81920, s14 // (op 54621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[148:151], v[50:53], a[200:203], v107, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16449) 
s_mov_b32 m0, s39 // (op 54621) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[148:151], v[54:57], a[204:207], v107, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16357) 
buffer_load_dwordx4 v146, s[8:11], 0 offen offset: 0 lds // (op 54621) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[148:151], v[58:61], a[208:211], v107, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16265) 
v_add_co_u32 v146, s[60:61], v146, v132 // (op 54621) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[148:151], v[62:65], a[212:215], v107, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16173) 
v_addc_co_u32 v147, s[60:61], 0, v147, s[60:61] // (op 54621) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[148:151], v[68:71], a[216:219], v107, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16081) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[148:151], v[72:75], a[220:223], v107, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15989) 
s_add_u32 m0, m0, 4096 // (op 54621) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[152:155], v[42:45], a[224:227], v127, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19211) 
buffer_load_dwordx4 v146, s[8:11], 0 offen offset: 0 lds // (op 54621) Load value direct to lds
v_add_co_u32 v146, s[60:61], v146, v132 // (op 54621) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[152:155], v[46:49], a[228:231], v127, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19119) 
v_addc_co_u32 v147, s[60:61], 0, v147, s[60:61] // (op 54621) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[152:155], v[50:53], a[232:235], v127, v41 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19027) 
s_add_u32 m0, m0, 4096 // (op 54621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[152:155], v[54:57], a[236:239], v127, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18935) 
buffer_load_dwordx4 v146, s[8:11], 0 offen offset: 0 lds // (op 54621) Load value direct to lds
v_add_co_u32 v146, s[60:61], v146, v132 // (op 54621) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[152:155], v[58:61], a[240:243], v127, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18843) 
v_addc_co_u32 v147, s[60:61], 0, v147, s[60:61] // (op 54621) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[152:155], v[62:65], a[244:247], v127, v66 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18751) 
s_add_u32 m0, m0, 4096 // (op 54621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[152:155], v[68:71], a[248:251], v127, v67 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18659) 
buffer_load_dwordx4 v146, s[8:11], 0 offen offset: 0 lds // (op 54621) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[152:155], v[72:75], a[252:255], v127, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18567) 
s_lshl_b32 s39, s28, 2 // (op 297) 


s_mul_i32 s60, 6, s39 // (op 297) 

v_add_u32 v126, v117, s60 // (op 297) 


buffer_load_dword v40, v126, s[52:55], 0 offen // (op 297) Load value

s_barrier  // (op 56348) 



v_add_u32 v126, 133120, v125 // (op 335) 
s_waitcnt vmcnt(35)
ds_write_b32 v126, v28 // (op 335) Store local data 

s_barrier  // (op 56342) 



v_add_u32 v126, 134144, v124 // (op 334) 
s_waitcnt vmcnt(30)
ds_write_b32 v126, v32 // (op 334) Store local data 

s_barrier  // (op 56331) 



v_add_u32 v126, 98304, v16 // (op 327) 
ds_read_b64_tr_b4 v[82:83], v126 // (op 327) Transpose load local data 

v_add_u32 v126, 100352, v16 // (op 327) 
ds_read_b64_tr_b4 v[84:85], v126 // (op 327) Transpose load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 56345) 


v_add_u32 v126, 133120, v17 // (op 331) 
ds_read_u8 v76, v126 // (op 331) Load local data 




v_add_u32 v126, 98320, v16 // (op 545) 
ds_read_b64_tr_b4 v[88:89], v126 // (op 545) Transpose load local data 

v_add_u32 v126, 100368, v16 // (op 545) 
ds_read_b64_tr_b4 v[90:91], v126 // (op 545) Transpose load local data 



v_add_u32 v126, 133152, v17 // (op 549) 
ds_read_u8 v86, v126 // (op 549) Load local data 




v_add_u32 v126, 98336, v16 // (op 843) 
ds_read_b64_tr_b4 v[92:93], v126 // (op 843) Transpose load local data 

v_add_u32 v126, 100384, v16 // (op 843) 
ds_read_b64_tr_b4 v[94:95], v126 // (op 843) Transpose load local data 



v_add_u32 v126, 133184, v17 // (op 847) 
ds_read_u8 v87, v126 // (op 847) Load local data 




v_add_u32 v126, 98352, v16 // (op 1141) 
ds_read_b64_tr_b4 v[98:99], v126 // (op 1141) Transpose load local data 

v_add_u32 v126, 100400, v16 // (op 1141) 
ds_read_b64_tr_b4 v[100:101], v126 // (op 1141) Transpose load local data 



v_add_u32 v126, 133216, v17 // (op 1145) 
ds_read_u8 v96, v126 // (op 1145) Load local data 




v_add_u32 v126, 98368, v16 // (op 1439) 
ds_read_b64_tr_b4 v[102:103], v126 // (op 1439) Transpose load local data 

v_add_u32 v126, 100416, v16 // (op 1439) 
ds_read_b64_tr_b4 v[104:105], v126 // (op 1439) Transpose load local data 



v_add_u32 v126, 133248, v17 // (op 1443) 
ds_read_u8 v97, v126 // (op 1443) Load local data 




v_add_u32 v126, 98384, v16 // (op 1737) 
ds_read_b64_tr_b4 v[108:109], v126 // (op 1737) Transpose load local data 

v_add_u32 v126, 100432, v16 // (op 1737) 
ds_read_b64_tr_b4 v[110:111], v126 // (op 1737) Transpose load local data 



v_add_u32 v126, 133280, v17 // (op 1741) 
ds_read_u8 v106, v126 // (op 1741) Load local data 




v_add_u32 v126, 98400, v16 // (op 2035) 
ds_read_b64_tr_b4 v[112:113], v126 // (op 2035) Transpose load local data 

v_add_u32 v126, 100448, v16 // (op 2035) 
ds_read_b64_tr_b4 v[114:115], v126 // (op 2035) Transpose load local data 



v_add_u32 v126, 133312, v17 // (op 2039) 
ds_read_u8 v107, v126 // (op 2039) Load local data 




v_add_u32 v126, 98416, v16 // (op 2333) 
ds_read_b64_tr_b4 v[132:133], v126 // (op 2333) Transpose load local data 

v_add_u32 v126, 100464, v16 // (op 2333) 
ds_read_b64_tr_b4 v[134:135], v126 // (op 2333) Transpose load local data 



v_add_u32 v126, 133344, v17 // (op 2337) 
ds_read_u8 v127, v126 // (op 2337) Load local data 

s_barrier  // (op 56325) 
s_barrier  // (op 56413) 



v_add_u32 v126, 114688, v15 // (op 325) 
ds_read_b64_tr_b4 v[78:79], v126 // (op 325) Transpose load local data 

v_add_u32 v126, 116736, v15 // (op 325) 
ds_read_b64_tr_b4 v[80:81], v126 // (op 325) Transpose load local data 

s_waitcnt lgkmcnt(15)
s_barrier  // (op 56338) 


v_add_u32 v126, 134144, v14 // (op 329) 
ds_read_u8 v77, v126 // (op 329) Load local data 




v_add_u32 v126, 114704, v15 // (op 3114) 
ds_read_b64_tr_b4 v[136:137], v126 // (op 3114) Transpose load local data 

v_add_u32 v126, 116752, v15 // (op 3114) 
ds_read_b64_tr_b4 v[138:139], v126 // (op 3114) Transpose load local data 



v_add_u32 v126, 134176, v14 // (op 3117) 
ds_read_u8 v34, v126 // (op 3117) Load local data 




v_add_u32 v126, 114720, v15 // (op 5692) 
ds_read_b64_tr_b4 v[140:141], v126 // (op 5692) Transpose load local data 

v_add_u32 v126, 116768, v15 // (op 5692) 
ds_read_b64_tr_b4 v[142:143], v126 // (op 5692) Transpose load local data 



v_add_u32 v126, 134208, v14 // (op 5695) 
ds_read_u8 v41, v126 // (op 5695) Load local data 




v_add_u32 v126, 114736, v15 // (op 8270) 
ds_read_b64_tr_b4 v[144:145], v126 // (op 8270) Transpose load local data 

v_add_u32 v126, 116784, v15 // (op 8270) 
ds_read_b64_tr_b4 v[146:147], v126 // (op 8270) Transpose load local data 



v_add_u32 v126, 134240, v14 // (op 8273) 
ds_read_u8 v24, v126 // (op 8273) Load local data 




v_add_u32 v126, 114752, v15 // (op 10848) 
ds_read_b64_tr_b4 v[148:149], v126 // (op 10848) Transpose load local data 

v_add_u32 v126, 116800, v15 // (op 10848) 
ds_read_b64_tr_b4 v[150:151], v126 // (op 10848) Transpose load local data 



v_add_u32 v126, 134272, v14 // (op 10851) 
ds_read_u8 v25, v126 // (op 10851) Load local data 




v_add_u32 v126, 114768, v15 // (op 13426) 
ds_read_b64_tr_b4 v[152:153], v126 // (op 13426) Transpose load local data 

v_add_u32 v126, 116816, v15 // (op 13426) 
ds_read_b64_tr_b4 v[154:155], v126 // (op 13426) Transpose load local data 



v_add_u32 v126, 134304, v14 // (op 13429) 
ds_read_u8 v66, v126 // (op 13429) Load local data 




v_add_u32 v126, 114784, v15 // (op 16004) 
ds_read_b64_tr_b4 v[156:157], v126 // (op 16004) Transpose load local data 

v_add_u32 v126, 116832, v15 // (op 16004) 
ds_read_b64_tr_b4 v[158:159], v126 // (op 16004) Transpose load local data 



v_add_u32 v126, 134336, v14 // (op 16007) 
ds_read_u8 v67, v126 // (op 16007) Load local data 




v_add_u32 v126, 114800, v15 // (op 18582) 
ds_read_b64_tr_b4 v[160:161], v126 // (op 18582) Transpose load local data 

v_add_u32 v126, 116848, v15 // (op 18582) 
ds_read_b64_tr_b4 v[162:163], v126 // (op 18582) Transpose load local data 



v_add_u32 v126, 134368, v14 // (op 18585) 
ds_read_u8 v116, v126 // (op 18585) Load local data 

s_barrier  // (op 56511) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[82:85], a[0:3], v77, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 332) 
s_barrier  // (op 56335) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[88:91], a[4:7], v77, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 525) 
s_lshl_b64 s[60:61], s[24:25], 5 // (op 54636) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[92:95], a[8:11], v77, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 823) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54636) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:81], v[98:101], a[12:15], v77, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1121) 
v_lshrrev_b64 v[164:165], 3, s[62:63] // (op 54636) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[102:105], a[16:19], v77, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1419) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54636) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[108:111], a[20:23], v77, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1717) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54636) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[112:115], a[24:27], v77, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2015) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54636) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[132:135], a[28:31], v77, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2313) 
s_mul_i32 s63, 7, s61 // (op 54636) most significant: low of low * high
s_mul_i32 s62, 7, s60 // (op 54636) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:139], v[82:85], a[32:35], v34, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3744) 
s_mul_hi_u32 s39, 7, s60 // (op 54636) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[88:91], a[36:39], v34, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3652) 
s_add_u32 s63, s63, s39 // (op 54636) most significant: sum


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[136:139], v[92:95], a[40:43], v34, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3560) 
v_mov_b32 v126, s62 // (op 54636) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[136:139], v[98:101], a[44:47], v34, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3468) 
v_mov_b32 v166, s63 // (op 54636) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[102:105], a[48:51], v34, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3376) 
v_add_co_u32 v168, s[60:61], v128, v126 // (op 54636) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[108:111], a[52:55], v34, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3284) 
v_addc_co_u32 v169, s[60:61], v129, v166, s[60:61] // (op 54636) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[112:115], a[56:59], v34, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3192) 
s_add_u32 s39, 98304, s19 // (op 54636) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[132:135], a[60:63], v34, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3100) 
s_mov_b32 m0, s39 // (op 54636) reg expression

buffer_load_dwordx4 v168, s[48:51], 0 offen offset: 0 lds // (op 54636) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:143], v[82:85], a[64:67], v41, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6322) 
v_add_co_u32 v168, s[60:61], v168, v164 // (op 54636) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:143], v[88:91], a[68:71], v41, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6230) 
v_addc_co_u32 v169, s[60:61], 0, v169, s[60:61] // (op 54636) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:143], v[92:95], a[72:75], v41, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6138) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[98:101], a[76:79], v41, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6046) 
s_add_u32 m0, m0, 4096 // (op 54636) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:143], v[102:105], a[80:83], v41, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5954) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[108:111], a[84:87], v41, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5862) 
buffer_load_dwordx4 v168, s[48:51], 0 offen offset: 0 lds // (op 54636) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[112:115], a[88:91], v41, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5770) 
v_add_co_u32 v168, s[60:61], v168, v164 // (op 54636) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[132:135], a[92:95], v41, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5678) 
v_addc_co_u32 v169, s[60:61], 0, v169, s[60:61] // (op 54636) most significant half


s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[144:147], v[82:85], a[96:99], v24, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8900) 
s_add_u32 m0, m0, 4096 // (op 54636) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[88:91], a[100:103], v24, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8808) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[92:95], a[104:107], v24, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8716) 
buffer_load_dwordx4 v168, s[48:51], 0 offen offset: 0 lds // (op 54636) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[144:147], v[98:101], a[108:111], v24, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8624) 
v_add_co_u32 v168, s[60:61], v168, v164 // (op 54636) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:147], v[102:105], a[112:115], v24, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8532) 
v_addc_co_u32 v169, s[60:61], 0, v169, s[60:61] // (op 54636) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[108:111], a[116:119], v24, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8440) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:147], v[112:115], a[120:123], v24, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8348) 
s_add_u32 m0, m0, 4096 // (op 54636) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:147], v[132:135], a[124:127], v24, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8256) 

buffer_load_dwordx4 v168, s[48:51], 0 offen offset: 0 lds // (op 54636) Load value direct to lds




s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:151], v[82:85], a[128:131], v25, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11478) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[148:151], v[88:91], a[132:135], v25, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11386) 
s_lshl_b32 s39, s34, 2 // (op 338) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:151], v[92:95], a[136:139], v25, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11294) 
s_mul_i32 s60, 7, s39 // (op 338) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[148:151], v[98:101], a[140:143], v25, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11202) 
v_add_u32 v126, v22, s60 // (op 338) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[148:151], v[102:105], a[144:147], v25, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11110) 
s_lshl_b64 s[60:61], s[24:25], 7 // (op 54838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[148:151], v[108:111], a[148:151], v25, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11018) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54838) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[112:115], a[152:155], v25, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10926) 
buffer_load_dword v28, v126, s[56:59], 0 offen // (op 338) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:151], v[132:135], a[156:159], v25, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10834) 

s_lshl_b32 s39, s34, 4 // (op 54966) 
v_add_u32 v35, v35, s39 // (op 54966) 

s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[152:155], v[82:85], a[160:163], v66, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14056) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54838) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[152:155], v[88:91], a[164:167], v66, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13964) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54838) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[152:155], v[92:95], a[168:171], v66, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13872) 
v_mov_b32 v22, s62 // (op 54838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[152:155], v[98:101], a[172:175], v66, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13780) 
v_mov_b32 v126, s63 // (op 54838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:155], v[102:105], a[176:179], v66, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13688) 
v_add_co_u32 v30, s[60:61], v30, v22 // (op 54838) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[152:155], v[108:111], a[180:183], v66, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13596) 
v_addc_co_u32 v31, s[60:61], v31, v126, s[60:61] // (op 54838) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[152:155], v[112:115], a[184:187], v66, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13504) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 56328) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:155], v[132:135], a[188:191], v66, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13412) 
s_lshl_b64 s[60:61], s[12:13], 5 // (op 54631) 

s_lshl_b64 s[62:63], s[60:61], 2 // (op 54631) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[156:159], v[82:85], a[192:195], v67, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16634) 
v_lshrrev_b64 v[128:129], 3, s[62:63] // (op 54631) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[156:159], v[88:91], a[196:199], v67, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16542) 
s_lshl_b64 s[60:61], s[12:13], 7 // (op 54631) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[156:159], v[92:95], a[200:203], v67, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16450) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54631) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[156:159], v[98:101], a[204:207], v67, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16358) 
s_lshr_b64 s[60:61], s[62:63], 3 // (op 54631) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[156:159], v[102:105], a[208:211], v67, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16266) 
s_mul_i32 s63, 7, s61 // (op 54631) most significant: low of low * high
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[156:159], v[108:111], a[212:215], v67, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16174) 
s_mul_i32 s62, 7, s60 // (op 54631) least significant
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[156:159], v[112:115], a[216:219], v67, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 16082) 
s_mul_hi_u32 s39, 7, s60 // (op 54631) most significant: high of low * low
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[156:159], v[132:135], a[220:223], v67, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15990) 
s_add_u32 s63, s63, s39 // (op 54631) most significant: sum



v_mov_b32 v22, s62 // (op 54631) 
v_mov_b32 v126, s63 // (op 54631) 
v_add_co_u32 v136, s[60:61], v130, v22 // (op 54631) least significant half
v_addc_co_u32 v137, s[60:61], v131, v126, s[60:61] // (op 54631) most significant half




s_add_u32 s39, 114688, s18 // (op 54631) 
s_mov_b32 m0, s39 // (op 54631) reg expression
buffer_load_dwordx4 v136, s[8:11], 0 offen offset: 0 lds // (op 54631) Load value direct to lds
v_add_co_u32 v136, s[60:61], v136, v128 // (op 54631) least significant half
v_addc_co_u32 v137, s[60:61], 0, v137, s[60:61] // (op 54631) most significant half

s_add_u32 m0, m0, 4096 // (op 54631) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[160:163], v[82:85], a[224:227], v116, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19212) 
buffer_load_dwordx4 v136, s[8:11], 0 offen offset: 0 lds // (op 54631) Load value direct to lds
v_add_co_u32 v136, s[60:61], v136, v128 // (op 54631) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[160:163], v[88:91], a[228:231], v116, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19120) 
v_addc_co_u32 v137, s[60:61], 0, v137, s[60:61] // (op 54631) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[160:163], v[92:95], a[232:235], v116, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 19028) 
s_add_u32 m0, m0, 4096 // (op 54631) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[160:163], v[98:101], a[236:239], v116, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18936) 
buffer_load_dwordx4 v136, s[8:11], 0 offen offset: 0 lds // (op 54631) Load value direct to lds
v_add_co_u32 v136, s[60:61], v136, v128 // (op 54631) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[160:163], v[102:105], a[240:243], v116, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18844) 
v_addc_co_u32 v137, s[60:61], 0, v137, s[60:61] // (op 54631) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[160:163], v[108:111], a[244:247], v116, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18752) 
s_add_u32 m0, m0, 4096 // (op 54631) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[160:163], v[112:115], a[248:251], v116, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18660) 
buffer_load_dwordx4 v136, s[8:11], 0 offen offset: 0 lds // (op 54631) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[160:163], v[132:135], a[252:255], v116, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18568) 
s_lshl_b32 s39, s28, 2 // (op 337) 



s_mul_i32 s60, 7, s39 // (op 337) 

v_add_u32 v22, v117, s60 // (op 337) 


s_lshl_b64 s[60:61], s[12:13], 7 // (op 54774) 
s_lshl_b64 s[62:63], s[60:61], 2 // (op 54774) 

s_lshr_b64 s[60:61], s[62:63], 3 // (op 54774) 

s_lshl_b64 s[62:63], s[60:61], 2 // (op 54774) 

v_mov_b32 v126, s62 // (op 54774) 
v_mov_b32 v127, s63 // (op 54774) 
v_add_co_u32 v26, s[60:61], v26, v126 // (op 54774) least significant half
v_addc_co_u32 v27, s[60:61], v27, v127, s[60:61] // (op 54774) most significant half




buffer_load_dword v32, v22, s[52:55], 0 offen // (op 337) Load value


s_barrier  // (op 56259) 
s_lshl_b32 s39, s28, 4 // (op 54902) 



v_add_u32 v23, v23, s39 // (op 54902) 

v_add_u32 v22, 131072, v119 // (op 30) 
s_waitcnt vmcnt(35)
ds_write_b32 v22, v33 // (op 30) Store local data 

s_barrier  // (op 56241) 



v_add_u32 v22, 132096, v118 // (op 21) 
s_waitcnt vmcnt(30)
ds_write_b32 v22, v36 // (op 21) Store local data 

s_barrier  // (op 56230) 



ds_read_b64_tr_b4 v[42:43], v4 // (op 75) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v4 offset:2048 // (op 75) Transpose load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 56266) 


v_add_u32 v22, 131072, v5 // (op 85) 
ds_read_u8 v29, v22 // (op 85) Load local data 




ds_read_b64_tr_b4 v[46:47], v4 offset:16 // (op 517) Transpose load local data 
ds_read_b64_tr_b4 v[48:49], v4 offset:2064 // (op 517) Transpose load local data 


v_add_u32 v22, 131104, v5 // (op 521) 
ds_read_u8 v34, v22 // (op 521) Load local data 




ds_read_b64_tr_b4 v[50:51], v4 offset:32 // (op 815) Transpose load local data 
ds_read_b64_tr_b4 v[52:53], v4 offset:2080 // (op 815) Transpose load local data 


v_add_u32 v22, 131136, v5 // (op 819) 
ds_read_u8 v41, v22 // (op 819) Load local data 




ds_read_b64_tr_b4 v[54:55], v4 offset:48 // (op 1113) Transpose load local data 
ds_read_b64_tr_b4 v[56:57], v4 offset:2096 // (op 1113) Transpose load local data 


v_add_u32 v22, 131168, v5 // (op 1117) 
ds_read_u8 v24, v22 // (op 1117) Load local data 




ds_read_b64_tr_b4 v[58:59], v4 offset:64 // (op 1411) Transpose load local data 
ds_read_b64_tr_b4 v[60:61], v4 offset:2112 // (op 1411) Transpose load local data 


v_add_u32 v22, 131200, v5 // (op 1415) 
ds_read_u8 v25, v22 // (op 1415) Load local data 




ds_read_b64_tr_b4 v[62:63], v4 offset:80 // (op 1709) Transpose load local data 
ds_read_b64_tr_b4 v[64:65], v4 offset:2128 // (op 1709) Transpose load local data 


v_add_u32 v22, 131232, v5 // (op 1713) 
ds_read_u8 v66, v22 // (op 1713) Load local data 




ds_read_b64_tr_b4 v[68:69], v4 offset:96 // (op 2007) Transpose load local data 
ds_read_b64_tr_b4 v[70:71], v4 offset:2144 // (op 2007) Transpose load local data 


v_add_u32 v22, 131264, v5 // (op 2011) 
ds_read_u8 v67, v22 // (op 2011) Load local data 




ds_read_b64_tr_b4 v[72:73], v4 offset:112 // (op 2305) Transpose load local data 
ds_read_b64_tr_b4 v[74:75], v4 offset:2160 // (op 2305) Transpose load local data 


v_add_u32 v22, 131296, v5 // (op 2309) 
ds_read_u8 v76, v22 // (op 2309) Load local data 

s_barrier  // (op 56214) 



ds_read_b64_tr_b4 v[78:79], v3 offset:16384 // (op 70) Transpose load local data 
ds_read_b64_tr_b4 v[80:81], v3 offset:18432 // (op 70) Transpose load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 56248) 


v_add_u32 v22, 132096, v0 // (op 80) 
ds_read_u8 v77, v22 // (op 80) Load local data 




ds_read_b64_tr_b4 v[82:83], v3 offset:16400 // (op 3092) Transpose load local data 
ds_read_b64_tr_b4 v[84:85], v3 offset:18448 // (op 3092) Transpose load local data 


v_add_u32 v22, 132128, v0 // (op 3095) 
ds_read_u8 v86, v22 // (op 3095) Load local data 




ds_read_b64_tr_b4 v[88:89], v3 offset:16416 // (op 5670) Transpose load local data 
ds_read_b64_tr_b4 v[90:91], v3 offset:18464 // (op 5670) Transpose load local data 


v_add_u32 v22, 132160, v0 // (op 5673) 
ds_read_u8 v87, v22 // (op 5673) Load local data 




ds_read_b64_tr_b4 v[92:93], v3 offset:16432 // (op 8248) Transpose load local data 
ds_read_b64_tr_b4 v[94:95], v3 offset:18480 // (op 8248) Transpose load local data 


v_add_u32 v22, 132192, v0 // (op 8251) 
ds_read_u8 v96, v22 // (op 8251) Load local data 




ds_read_b64_tr_b4 v[98:99], v3 offset:16448 // (op 10826) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v3 offset:18496 // (op 10826) Transpose load local data 


v_add_u32 v22, 132224, v0 // (op 10829) 
ds_read_u8 v97, v22 // (op 10829) Load local data 




ds_read_b64_tr_b4 v[102:103], v3 offset:16464 // (op 13404) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v3 offset:18512 // (op 13404) Transpose load local data 


v_add_u32 v22, 132256, v0 // (op 13407) 
ds_read_u8 v106, v22 // (op 13407) Load local data 




ds_read_b64_tr_b4 v[108:109], v3 offset:16480 // (op 15982) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v3 offset:18528 // (op 15982) Transpose load local data 


v_add_u32 v22, 132288, v0 // (op 15985) 
ds_read_u8 v107, v22 // (op 15985) Load local data 




ds_read_b64_tr_b4 v[112:113], v3 offset:16496 // (op 18560) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v3 offset:18544 // (op 18560) Transpose load local data 


v_add_u32 v22, 132320, v0 // (op 18563) 
ds_read_u8 v116, v22 // (op 18563) Load local data 

s_ashr_i64 s[60:61], s[22:23], 7 // (op 18526) 
s_mov_b32 s39, s60 // (op 18526) convert

s_and_b32 s60, s39, 4294967292 // (op 18526) 

s_cmp_lt_u32 s38, s60 // (op 18526) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoop_18526 // (op 18526) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoop_18526 if true)
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoop_18526:













































s_barrier  // (op 56479) 
v_and_b32 v22, -8, v1 // (op 54674) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 54674) 
v_sub_u32 v23, v1, v22 // (op 54674) 

s_lshr_b32 s6, s0, 2 // (op 54674) 

v_lshlrev_b32 v22, 5, v23 // (op 54674) 

v_lshl_add_u32 v23, s16, 8, v22 // (op 54674) 

v_mov_b32 v31, 0 // (op 54674) convert
v_mov_b32 v30, v23 // (op 54674) convert

v_lshrrev_b32 v29, 3, v1 // (op 54674) 
v_lshl_add_u32 v22, s6, 9, v29 // (op 54674) 


v_mul_lo_u32 v27, v22, s13 // (op 54674) most significant: low of low * high
v_mul_hi_u32 v29, v22, s12 // (op 54674) most significant: high of low * low
v_mul_lo_u32 v26, v22, s12 // (op 54674) least significant: low of low * low
v_add_u32 v27, v27, v29 // (op 54674) most significant: sum


v_add_co_u32 v22, s[0:1], v30, v26 // (op 54674) least significant half
v_addc_co_u32 v23, s[0:1], v31, v27, s[0:1] // (op 54674) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 54674) 


v_lshrrev_b64 v[30:31], 3, v[22:23] // (op 54674) 

s_mul_i32 s1, s12, s23 // (op 54676) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 54676) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 54676) least significant
s_mul_hi_u32 s7, s12, s22 // (op 54676) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54676) most significant: sum
s_add_u32 s1, s1, s7 // (op 54676) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54676) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54676) 

s_mov_b32 s8, s20 // (op 54676) reg expression
s_mov_b32 s9, s21 // (op 54676) reg expression
s_mov_b32 s10, s0 // (op 54676) reg expression
s_mov_b32 s11, 131072 // (op 54676) call()

v_lshrrev_b32 v29, 3, v1 // (op 54804) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 54804) 
v_and_b32 v22, -8, v1 // (op 54804) 
s_lshr_b32 s6, s0, 2 // (op 54804) 

v_sub_u32 v23, v1, v22 // (op 54804) 

v_lshl_add_u32 v22, s6, 9, v29 // (op 54804) 


v_lshlrev_b32 v29, 5, v23 // (op 54804) 

v_lshl_add_u32 v23, s17, 8, v29 // (op 54804) 

v_mov_b32 v25, 0 // (op 54804) convert
v_mov_b32 v24, v23 // (op 54804) convert

v_mul_lo_u32 v27, v22, s25 // (op 54804) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 54804) most significant: high of low * low
v_mul_lo_u32 v26, v22, s24 // (op 54804) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 54804) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v24 // (op 54804) least significant half
v_addc_co_u32 v27, s[0:1], v27, v25, s[0:1] // (op 54804) most significant half



v_lshlrev_b64 v[22:23], 2, v[26:27] // (op 54804) 

v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 54804) 

s_mul_i32 s1, s24, s23 // (op 54806) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 54806) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 54806) least significant
s_mul_hi_u32 s7, s24, s22 // (op 54806) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54806) most significant: sum
s_add_u32 s1, s1, s7 // (op 54806) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 54806) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 54806) 

s_mov_b32 s48, s26 // (op 54806) reg expression
s_mov_b32 s49, s27 // (op 54806) reg expression
s_mov_b32 s50, s0 // (op 54806) reg expression
s_mov_b32 s51, 131072 // (op 54806) call()

v_and_b32 v29, -64, v1 // (op 54868) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 54868) 
v_sub_u32 v22, v1, v29 // (op 54868) 

s_lshr_b32 s6, s0, 2 // (op 54868) 

v_lshlrev_b32 v23, 2, v22 // (op 54868) 

v_lshl_add_u32 v22, s16, 8, v23 // (op 54868) 

v_lshrrev_b32 v23, 6, v1 // (op 54868) 
v_lshl_add_u32 v29, s6, 4, v23 // (op 54868) 


v_mul_lo_u32 v23, v29, s28 // (op 54868) 

v_add_u32 v29, v22, v23 // (op 54868) 


s_mul_i32 s1, s28, s33 // (op 54870) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 54870) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 54870) least significant
s_mul_hi_u32 s7, s28, s32 // (op 54870) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54870) most significant: sum
s_add_u32 s1, s1, s7 // (op 54870) most significant: sum


s_mov_b32 s52, s30 // (op 54870) reg expression
s_mov_b32 s53, s31 // (op 54870) reg expression
s_mov_b32 s54, s0 // (op 54870) reg expression
s_mov_b32 s55, 131072 // (op 54870) call()

v_lshrrev_b32 v22, 6, v1 // (op 54932) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 54932) 
v_and_b32 v23, -64, v1 // (op 54932) 
s_lshr_b32 s6, s0, 2 // (op 54932) 

v_sub_u32 v26, v1, v23 // (op 54932) 

v_lshl_add_u32 v23, s6, 4, v22 // (op 54932) 


v_lshlrev_b32 v22, 2, v26 // (op 54932) 

v_lshl_add_u32 v26, s17, 8, v22 // (op 54932) 

v_mul_lo_u32 v22, v23, s34 // (op 54932) 
v_add_u32 v22, v22, v26 // (op 54932) 


s_mul_i32 s1, s34, s33 // (op 54934) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 54934) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 54934) least significant
s_mul_hi_u32 s7, s34, s32 // (op 54934) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 54934) most significant: sum
s_add_u32 s1, s1, s7 // (op 54934) most significant: sum


s_mov_b32 s56, s36 // (op 54934) reg expression
s_mov_b32 s57, s37 // (op 54934) reg expression
s_mov_b32 s58, s0 // (op 54934) reg expression
s_mov_b32 s59, 131072 // (op 54934) call()

v_and_b32 v23, -8, v1 // (op 55030) 
v_sub_u32 v26, v1, v23 // (op 55030) 

v_lshlrev_b32 v23, 5, v26 // (op 55030) 

v_lshrrev_b32 v26, 3, v1 // (op 55030) 
v_lshl_add_u32 v27, v26, 8, v23 // (op 55030) 


v_lshlrev_b32 v23, 2, v27 // (op 55030) 

v_lshrrev_b32 v26, 3, v23 // (op 55030) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 55030) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -8, v1 // (op 55087) 
v_sub_u32 v26, v1, v23 // (op 55087) 

v_lshlrev_b32 v23, 5, v26 // (op 55087) 

v_lshrrev_b32 v26, 3, v1 // (op 55087) 
v_lshl_add_u32 v27, v26, 8, v23 // (op 55087) 


v_lshlrev_b32 v23, 2, v27 // (op 55087) 

v_lshrrev_b32 v26, 3, v23 // (op 55087) 

s_nop 0
v_readfirstlane_b32 s1, v26 // (op 55087) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -64, v1 // (op 55144) 
v_sub_u32 v26, v1, v23 // (op 55144) 

v_lshlrev_b32 v23, 2, v26 // (op 55144) 

v_lshrrev_b32 v26, 6, v1 // (op 55144) 
v_lshl_add_u32 v27, v26, 8, v23 // (op 55144) 


v_and_b32 v23, -64, v1 // (op 55202) 
v_sub_u32 v26, v1, v23 // (op 55202) 

v_lshlrev_b32 v23, 2, v26 // (op 55202) 

v_lshrrev_b32 v26, 6, v1 // (op 55202) 
v_lshl_add_u32 v33, v26, 8, v23 // (op 55202) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 18532) 
s_and_b32 s14, s6, 4294967292 // (op 18532) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 18531) 
s_mov_b32 s15, s6 // (op 18531) convert

s_cmp_lt_u32 s14, s15 // (op 18531) 

















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoopTail_18531 // (op 18531) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoopTail_18531 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoopTail_18531:
s_barrier  // (op 56211) 
s_add_u32 s14, s14, 1 // (op 18533) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 54598) 
s_lshl_b64 s[18:19], s[6:7], 2 // (op 54598) 

v_lshrrev_b64 v[34:35], 3, s[18:19] // (op 54598) 


v_mov_b64 v[36:37], v[30:31] // (op 54598) 
v_mov_b64 v[38:39], v[36:37] // (op 54598) reg expression
s_add_u32 s15, 16384, s0 // (op 54598) 
s_mov_b32 m0, s15 // (op 54598) reg expression
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54598) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54598) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54598) most significant half

s_add_u32 m0, m0, 4096 // (op 54598) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54598) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54598) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54598) most significant half

s_add_u32 m0, m0, 4096 // (op 54598) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54598) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54598) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54598) most significant half

s_add_u32 m0, m0, 4096 // (op 54598) 
buffer_load_dwordx4 v38, s[8:11], 0 offen offset: 0 lds // (op 54598) Load value direct to lds




s_lshl_b64 s[6:7], s[12:13], 7 // (op 54664) 
s_lshl_b64 s[18:19], s[6:7], 2 // (op 54664) 

s_lshr_b64 s[6:7], s[18:19], 3 // (op 54664) 

v_mov_b32 v23, s6 // (op 54664) 
v_mov_b32 v26, s7 // (op 54664) 
v_add_co_u32 v30, s[18:19], v30, v23 // (op 54664) least significant half
v_addc_co_u32 v31, s[18:19], v31, v26, s[18:19] // (op 54664) most significant half




s_barrier  // (op 56227) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 54604) 
s_lshl_b64 s[18:19], s[6:7], 2 // (op 54604) 

v_lshrrev_b64 v[34:35], 3, s[18:19] // (op 54604) 


v_mov_b64 v[36:37], v[24:25] // (op 54604) 
v_mov_b64 v[38:39], v[36:37] // (op 54604) reg expression
s_mov_b32 m0, s1 // (op 54604) reg expression
buffer_load_dwordx4 v38, s[48:51], 0 offen offset: 0 lds // (op 54604) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54604) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54604) most significant half

s_add_u32 m0, m0, 4096 // (op 54604) 
buffer_load_dwordx4 v38, s[48:51], 0 offen offset: 0 lds // (op 54604) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54604) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54604) most significant half

s_add_u32 m0, m0, 4096 // (op 54604) 
buffer_load_dwordx4 v38, s[48:51], 0 offen offset: 0 lds // (op 54604) Load value direct to lds
v_add_co_u32 v38, s[6:7], v38, v34 // (op 54604) least significant half
v_addc_co_u32 v39, s[6:7], 0, v39, s[6:7] // (op 54604) most significant half

s_add_u32 m0, m0, 4096 // (op 54604) 
buffer_load_dwordx4 v38, s[48:51], 0 offen offset: 0 lds // (op 54604) Load value direct to lds



s_lshl_b64 s[6:7], s[24:25], 7 // (op 54796) 
s_lshl_b64 s[18:19], s[6:7], 2 // (op 54796) 

s_lshr_b64 s[6:7], s[18:19], 3 // (op 54796) 

v_mov_b32 v23, s6 // (op 54796) 
v_mov_b32 v26, s7 // (op 54796) 
v_add_co_u32 v24, s[18:19], v24, v23 // (op 54796) least significant half
v_addc_co_u32 v25, s[18:19], v25, v26, s[18:19] // (op 54796) most significant half






v_mov_b32 v23, v29 // (op 191) 
v_mov_b32 v26, v23 // (op 191) reg expression

buffer_load_dword v28, v26, s[52:55], 0 offen // (op 191) Load value


s_lshl_b32 s15, s28, 2 // (op 54860) 
v_add_u32 v29, v29, s15 // (op 54860) 



s_barrier  // (op 56244) 
v_mov_b32 v23, v22 // (op 192) 



v_mov_b32 v26, v23 // (op 192) reg expression

v_add_u32 v32, 132096, v27 // (op 188) 
s_waitcnt vmcnt(0)
ds_write_b32 v32, v28 // (op 188) Store local data 

buffer_load_dword v32, v26, s[56:59], 0 offen // (op 192) Load value



s_lshl_b32 s15, s34, 2 // (op 54924) 
v_add_u32 v22, v22, s15 // (op 54924) 

s_barrier  // (op 56217) 
s_barrier  // (op 56262) 






v_add_u32 v23, 131072, v33 // (op 189) 
ds_read_b64_tr_b4 v[34:35], v3 offset:16384 // (op 179) Transpose load local data 
ds_read_b64_tr_b4 v[36:37], v3 offset:18432 // (op 179) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v23, v32 // (op 189) Store local data 


s_barrier  // (op 56234) 



ds_read_b64_tr_b4 v[38:39], v4 // (op 181) Transpose load local data 
ds_read_b64_tr_b4 v[40:41], v4 offset:2048 // (op 181) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 56252) 


v_add_u32 v23, 132096, v0 // (op 183) 
ds_read_u8 v26, v23 // (op 183) Load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 56269) 


v_add_u32 v23, 131072, v5 // (op 185) 
ds_read_u8 v28, v23 // (op 185) Load local data 




ds_read_b64_tr_b4 v[42:43], v4 offset:16 // (op 496) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:41], a[0:3], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 186) 
ds_read_b64_tr_b4 v[44:45], v4 offset:2064 // (op 496) Transpose load local data 


v_add_u32 v23, 131104, v5 // (op 500) 
ds_read_u8 v32, v23 // (op 500) Load local data 




ds_read_b64_tr_b4 v[46:47], v4 offset:32 // (op 794) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[42:45], a[4:7], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 501) 
ds_read_b64_tr_b4 v[48:49], v4 offset:2080 // (op 794) Transpose load local data 


v_add_u32 v23, 131136, v5 // (op 798) 
ds_read_u8 v50, v23 // (op 798) Load local data 




ds_read_b64_tr_b4 v[52:53], v4 offset:48 // (op 1092) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[46:49], a[8:11], v26, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 799) 
ds_read_b64_tr_b4 v[54:55], v4 offset:2096 // (op 1092) Transpose load local data 


v_add_u32 v23, 131168, v5 // (op 1096) 
ds_read_u8 v51, v23 // (op 1096) Load local data 




ds_read_b64_tr_b4 v[56:57], v4 offset:64 // (op 1390) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[52:55], a[12:15], v26, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1097) 
ds_read_b64_tr_b4 v[58:59], v4 offset:2112 // (op 1390) Transpose load local data 


v_add_u32 v23, 131200, v5 // (op 1394) 
ds_read_u8 v60, v23 // (op 1394) Load local data 




ds_read_b64_tr_b4 v[62:63], v4 offset:80 // (op 1688) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[34:37], v[56:59], a[16:19], v26, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1395) 
ds_read_b64_tr_b4 v[64:65], v4 offset:2128 // (op 1688) Transpose load local data 


v_add_u32 v23, 131232, v5 // (op 1692) 
ds_read_u8 v61, v23 // (op 1692) Load local data 




ds_read_b64_tr_b4 v[66:67], v4 offset:96 // (op 1986) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:37], v[62:65], a[20:23], v26, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1693) 
ds_read_b64_tr_b4 v[68:69], v4 offset:2144 // (op 1986) Transpose load local data 


v_add_u32 v23, 131264, v5 // (op 1990) 
ds_read_u8 v70, v23 // (op 1990) Load local data 




ds_read_b64_tr_b4 v[72:73], v4 offset:112 // (op 2284) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[34:37], v[66:69], a[24:27], v26, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1991) 
ds_read_b64_tr_b4 v[74:75], v4 offset:2160 // (op 2284) Transpose load local data 


v_add_u32 v23, 131296, v5 // (op 2288) 
ds_read_u8 v71, v23 // (op 2288) Load local data 




ds_read_b64_tr_b4 v[76:77], v3 offset:16400 // (op 2852) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[34:37], v[72:75], a[28:31], v26, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2289) 
ds_read_b64_tr_b4 v[78:79], v3 offset:18448 // (op 2852) Transpose load local data 




v_add_u32 v23, 132128, v0 // (op 2855) 
ds_read_u8 v26, v23 // (op 2855) Load local data 




ds_read_b64_tr_b4 v[34:35], v3 offset:16416 // (op 5430) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[38:41], a[32:35], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2857) 
ds_read_b64_tr_b4 v[36:37], v3 offset:18464 // (op 5430) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[42:45], a[36:39], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2888) 
v_add_u32 v23, 132160, v0 // (op 5433) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[46:49], a[40:43], v26, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2919) 
ds_read_u8 v80, v23 // (op 5433) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[52:55], a[44:47], v26, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2950) 
ds_read_b64_tr_b4 v[82:83], v3 offset:16432 // (op 8008) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[56:59], a[48:51], v26, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2981) 
ds_read_b64_tr_b4 v[84:85], v3 offset:18480 // (op 8008) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[62:65], a[52:55], v26, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3012) 
v_add_u32 v23, 132192, v0 // (op 8011) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[66:69], a[56:59], v26, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3043) 
ds_read_u8 v81, v23 // (op 8011) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[72:75], a[60:63], v26, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3074) 
ds_read_b64_tr_b4 v[86:87], v3 offset:16448 // (op 10586) Transpose load local data 


ds_read_b64_tr_b4 v[88:89], v3 offset:18496 // (op 10586) Transpose load local data 


s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:37], v[38:41], a[64:67], v80, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5435) 
v_add_u32 v23, 132224, v0 // (op 10589) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[42:45], a[68:71], v80, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5466) 
ds_read_u8 v26, v23 // (op 10589) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[34:37], v[46:49], a[72:75], v80, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5497) 
ds_read_b64_tr_b4 v[76:77], v3 offset:16464 // (op 13164) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:37], v[52:55], a[76:79], v80, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5528) 
ds_read_b64_tr_b4 v[78:79], v3 offset:18512 // (op 13164) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[56:59], a[80:83], v80, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5559) 
v_add_u32 v23, 132256, v0 // (op 13167) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:37], v[62:65], a[84:87], v80, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5590) 
ds_read_u8 v90, v23 // (op 13167) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[34:37], v[66:69], a[88:91], v80, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5621) 
ds_read_b64_tr_b4 v[92:93], v3 offset:16480 // (op 15742) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[72:75], a[92:95], v80, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5652) 
ds_read_b64_tr_b4 v[94:95], v3 offset:18528 // (op 15742) Transpose load local data 




s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[38:41], a[96:99], v81, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8013) 
v_add_u32 v23, 132288, v0 // (op 15745) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[42:45], a[100:103], v81, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8044) 
ds_read_u8 v80, v23 // (op 15745) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[46:49], a[104:107], v81, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8075) 
ds_read_b64_tr_b4 v[34:35], v3 offset:16496 // (op 18320) Transpose load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[52:55], a[108:111], v81, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8106) 
ds_read_b64_tr_b4 v[36:37], v3 offset:18544 // (op 18320) Transpose load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[82:85], v[56:59], a[112:115], v81, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8137) 
v_add_u32 v23, 132320, v0 // (op 18323) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:85], v[62:65], a[116:119], v81, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8168) 
ds_read_u8 v91, v23 // (op 18323) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:85], v[66:69], a[120:123], v81, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8199) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:85], v[72:75], a[124:127], v81, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8230) 


s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[86:89], v[38:41], a[128:131], v26, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10591) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[86:89], v[42:45], a[132:135], v26, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10622) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[86:89], v[46:49], a[136:139], v26, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10653) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[86:89], v[52:55], a[140:143], v26, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10684) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:89], v[56:59], a[144:147], v26, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10715) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[86:89], v[62:65], a[148:151], v26, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10746) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[86:89], v[66:69], a[152:155], v26, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10777) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:89], v[72:75], a[156:159], v26, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10808) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[76:79], v[38:41], a[160:163], v90, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13169) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[76:79], v[42:45], a[164:167], v90, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13200) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[76:79], v[46:49], a[168:171], v90, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13231) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[76:79], v[52:55], a[172:175], v90, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13262) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[76:79], v[56:59], a[176:179], v90, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13293) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[76:79], v[62:65], a[180:183], v90, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13324) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[76:79], v[66:69], a[184:187], v90, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13355) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[76:79], v[72:75], a[188:191], v90, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13386) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[92:95], v[38:41], a[192:195], v80, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15747) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[92:95], v[42:45], a[196:199], v80, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15778) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[92:95], v[46:49], a[200:203], v80, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15809) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[92:95], v[52:55], a[204:207], v80, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15840) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:95], v[56:59], a[208:211], v80, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15871) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[92:95], v[62:65], a[212:215], v80, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15902) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[92:95], v[66:69], a[216:219], v80, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15933) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[92:95], v[72:75], a[220:223], v80, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15964) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[34:37], v[38:41], a[224:227], v91, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18325) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[34:37], v[42:45], a[228:231], v91, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18356) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[34:37], v[46:49], a[232:235], v91, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18387) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[34:37], v[52:55], a[236:239], v91, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18418) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[34:37], v[56:59], a[240:243], v91, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18449) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[34:37], v[62:65], a[244:247], v91, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18480) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[34:37], v[66:69], a[248:251], v91, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18511) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[34:37], v[72:75], a[252:255], v91, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 18542) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 18531) 
s_mov_b32 s15, s6 // (op 18531) convert

s_cmp_lt_u32 s14, s15 // (op 18531) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoopTail_18531 // (op 18531) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoopTail_18531 if true)
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoopTail_18531:























s_barrier  // (op 56483) 
















v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalFalse_0_beta_133
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_mul_f32 v10, v19, v14 // (op 50) 
v_mul_f32 v11, v19, v15 // (op 50) 
v_mul_f32 v12, v19, v16 // (op 50) 
v_mul_f32 v13, v19, v17 // (op 50) 



v_mov_b32 v0, v21 // (op 56) reg expression

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 56) Store value


v_accvgpr_read v14, a4 // (op 22833) 
v_accvgpr_read v15, a5 // (op 22833) 
v_accvgpr_read v16, a6 // (op 22833) 
v_accvgpr_read v17, a7 // (op 22833) 
v_mul_f32 v10, v19, v14 // (op 22833) 
v_mul_f32 v11, v19, v15 // (op 22833) 
v_mul_f32 v12, v19, v16 // (op 22833) 
v_mul_f32 v13, v19, v17 // (op 22833) 



s_lshl_b32 s0, s4, 7 // (op 22835) 
v_add_u32 v0, v21, s0 // (op 22835) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22835) Store value


v_accvgpr_read v14, a8 // (op 22845) 
v_accvgpr_read v15, a9 // (op 22845) 
v_accvgpr_read v16, a10 // (op 22845) 
v_accvgpr_read v17, a11 // (op 22845) 
v_mul_f32 v10, v19, v14 // (op 22845) 
v_mul_f32 v11, v19, v15 // (op 22845) 
v_mul_f32 v12, v19, v16 // (op 22845) 
v_mul_f32 v13, v19, v17 // (op 22845) 



s_lshl_b32 s0, s4, 8 // (op 22847) 
v_add_u32 v0, v21, s0 // (op 22847) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22847) Store value


v_accvgpr_read v14, a12 // (op 22857) 
v_accvgpr_read v15, a13 // (op 22857) 
v_accvgpr_read v16, a14 // (op 22857) 
v_accvgpr_read v17, a15 // (op 22857) 
v_mul_f32 v10, v19, v14 // (op 22857) 
v_mul_f32 v11, v19, v15 // (op 22857) 
v_mul_f32 v12, v19, v16 // (op 22857) 
v_mul_f32 v13, v19, v17 // (op 22857) 



s_lshl_b32 s0, s4, 7 // (op 22859) 
s_mul_i32 s1, 3, s0 // (op 22859) 

v_add_u32 v0, v21, s1 // (op 22859) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22859) Store value


v_accvgpr_read v14, a16 // (op 22869) 
v_accvgpr_read v15, a17 // (op 22869) 
v_accvgpr_read v16, a18 // (op 22869) 
v_accvgpr_read v17, a19 // (op 22869) 
v_mul_f32 v10, v19, v14 // (op 22869) 
v_mul_f32 v11, v19, v15 // (op 22869) 
v_mul_f32 v12, v19, v16 // (op 22869) 
v_mul_f32 v13, v19, v17 // (op 22869) 



s_lshl_b32 s0, s4, 9 // (op 22871) 
v_add_u32 v0, v21, s0 // (op 22871) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22871) Store value


v_accvgpr_read v14, a20 // (op 22881) 
v_accvgpr_read v15, a21 // (op 22881) 
v_accvgpr_read v16, a22 // (op 22881) 
v_accvgpr_read v17, a23 // (op 22881) 
v_mul_f32 v10, v19, v14 // (op 22881) 
v_mul_f32 v11, v19, v15 // (op 22881) 
v_mul_f32 v12, v19, v16 // (op 22881) 
v_mul_f32 v13, v19, v17 // (op 22881) 



s_lshl_b32 s0, s4, 7 // (op 22883) 
s_mul_i32 s1, 5, s0 // (op 22883) 

v_add_u32 v0, v21, s1 // (op 22883) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22883) Store value


v_accvgpr_read v14, a24 // (op 22893) 
v_accvgpr_read v15, a25 // (op 22893) 
v_accvgpr_read v16, a26 // (op 22893) 
v_accvgpr_read v17, a27 // (op 22893) 
v_mul_f32 v10, v19, v14 // (op 22893) 
v_mul_f32 v11, v19, v15 // (op 22893) 
v_mul_f32 v12, v19, v16 // (op 22893) 
v_mul_f32 v13, v19, v17 // (op 22893) 



s_lshl_b32 s0, s4, 7 // (op 22895) 
s_mul_i32 s1, 6, s0 // (op 22895) 

v_add_u32 v0, v21, s1 // (op 22895) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22895) Store value


v_accvgpr_read v14, a28 // (op 22905) 
v_accvgpr_read v15, a29 // (op 22905) 
v_accvgpr_read v16, a30 // (op 22905) 
v_accvgpr_read v17, a31 // (op 22905) 
v_mul_f32 v10, v19, v14 // (op 22905) 
v_mul_f32 v11, v19, v15 // (op 22905) 
v_mul_f32 v12, v19, v16 // (op 22905) 
v_mul_f32 v13, v19, v17 // (op 22905) 



s_lshl_b32 s0, s4, 7 // (op 22907) 
s_mul_i32 s1, 7, s0 // (op 22907) 

v_add_u32 v0, v21, s1 // (op 22907) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22907) Store value


v_accvgpr_read v14, a32 // (op 22959) 
v_accvgpr_read v15, a33 // (op 22959) 
v_accvgpr_read v16, a34 // (op 22959) 
v_accvgpr_read v17, a35 // (op 22959) 
v_mul_f32 v10, v19, v14 // (op 22959) 
v_mul_f32 v11, v19, v15 // (op 22959) 
v_mul_f32 v12, v19, v16 // (op 22959) 
v_mul_f32 v13, v19, v17 // (op 22959) 



v_add_u32 v0, 128, v21 // (op 22962) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22962) Store value


v_accvgpr_read v14, a36 // (op 22980) 
v_accvgpr_read v15, a37 // (op 22980) 
v_accvgpr_read v16, a38 // (op 22980) 
v_accvgpr_read v17, a39 // (op 22980) 
v_mul_f32 v10, v19, v14 // (op 22980) 
v_mul_f32 v11, v19, v15 // (op 22980) 
v_mul_f32 v12, v19, v16 // (op 22980) 
v_mul_f32 v13, v19, v17 // (op 22980) 



s_lshl_b32 s0, s4, 7 // (op 22964) 
s_add_u32 s0, s0, 128 // (op 22964) 
v_add_u32 v0, v21, s0 // (op 22964) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22964) Store value


v_accvgpr_read v14, a40 // (op 22985) 
v_accvgpr_read v15, a41 // (op 22985) 
v_accvgpr_read v16, a42 // (op 22985) 
v_accvgpr_read v17, a43 // (op 22985) 
v_mul_f32 v10, v19, v14 // (op 22985) 
v_mul_f32 v11, v19, v15 // (op 22985) 
v_mul_f32 v12, v19, v16 // (op 22985) 
v_mul_f32 v13, v19, v17 // (op 22985) 



s_lshl_b32 s0, s4, 8 // (op 22966) 
s_add_u32 s0, s0, 128 // (op 22966) 
v_add_u32 v0, v21, s0 // (op 22966) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22966) Store value


v_accvgpr_read v14, a44 // (op 22990) 
v_accvgpr_read v15, a45 // (op 22990) 
v_accvgpr_read v16, a46 // (op 22990) 
v_accvgpr_read v17, a47 // (op 22990) 
v_mul_f32 v10, v19, v14 // (op 22990) 
v_mul_f32 v11, v19, v15 // (op 22990) 
v_mul_f32 v12, v19, v16 // (op 22990) 
v_mul_f32 v13, v19, v17 // (op 22990) 



s_lshl_b32 s0, s4, 7 // (op 22968) 
s_mul_i32 s1, 3, s0 // (op 22968) 
s_add_u32 s1, s1, 128 // (op 22968) 

v_add_u32 v0, v21, s1 // (op 22968) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22968) Store value


v_accvgpr_read v14, a48 // (op 22995) 
v_accvgpr_read v15, a49 // (op 22995) 
v_accvgpr_read v16, a50 // (op 22995) 
v_accvgpr_read v17, a51 // (op 22995) 
v_mul_f32 v10, v19, v14 // (op 22995) 
v_mul_f32 v11, v19, v15 // (op 22995) 
v_mul_f32 v12, v19, v16 // (op 22995) 
v_mul_f32 v13, v19, v17 // (op 22995) 



s_lshl_b32 s0, s4, 9 // (op 22970) 
s_add_u32 s0, s0, 128 // (op 22970) 
v_add_u32 v0, v21, s0 // (op 22970) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22970) Store value


v_accvgpr_read v14, a52 // (op 23000) 
v_accvgpr_read v15, a53 // (op 23000) 
v_accvgpr_read v16, a54 // (op 23000) 
v_accvgpr_read v17, a55 // (op 23000) 
v_mul_f32 v10, v19, v14 // (op 23000) 
v_mul_f32 v11, v19, v15 // (op 23000) 
v_mul_f32 v12, v19, v16 // (op 23000) 
v_mul_f32 v13, v19, v17 // (op 23000) 



s_lshl_b32 s0, s4, 7 // (op 22972) 
s_mul_i32 s1, 5, s0 // (op 22972) 
s_add_u32 s1, s1, 128 // (op 22972) 

v_add_u32 v0, v21, s1 // (op 22972) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22972) Store value


v_accvgpr_read v14, a56 // (op 23005) 
v_accvgpr_read v15, a57 // (op 23005) 
v_accvgpr_read v16, a58 // (op 23005) 
v_accvgpr_read v17, a59 // (op 23005) 
v_mul_f32 v10, v19, v14 // (op 23005) 
v_mul_f32 v11, v19, v15 // (op 23005) 
v_mul_f32 v12, v19, v16 // (op 23005) 
v_mul_f32 v13, v19, v17 // (op 23005) 



s_lshl_b32 s0, s4, 7 // (op 22974) 
s_mul_i32 s1, 6, s0 // (op 22974) 
s_add_u32 s1, s1, 128 // (op 22974) 

v_add_u32 v0, v21, s1 // (op 22974) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22974) Store value


v_accvgpr_read v14, a60 // (op 23010) 
v_accvgpr_read v15, a61 // (op 23010) 
v_accvgpr_read v16, a62 // (op 23010) 
v_accvgpr_read v17, a63 // (op 23010) 
v_mul_f32 v10, v19, v14 // (op 23010) 
v_mul_f32 v11, v19, v15 // (op 23010) 
v_mul_f32 v12, v19, v16 // (op 23010) 
v_mul_f32 v13, v19, v17 // (op 23010) 



s_lshl_b32 s0, s4, 7 // (op 22976) 
s_mul_i32 s1, 7, s0 // (op 22976) 
s_add_u32 s1, s1, 128 // (op 22976) 

v_add_u32 v0, v21, s1 // (op 22976) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 22976) Store value


v_accvgpr_read v14, a64 // (op 23091) 
v_accvgpr_read v15, a65 // (op 23091) 
v_accvgpr_read v16, a66 // (op 23091) 
v_accvgpr_read v17, a67 // (op 23091) 
v_mul_f32 v10, v19, v14 // (op 23091) 
v_mul_f32 v11, v19, v15 // (op 23091) 
v_mul_f32 v12, v19, v16 // (op 23091) 
v_mul_f32 v13, v19, v17 // (op 23091) 



v_add_u32 v0, 256, v21 // (op 23094) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23094) Store value


v_accvgpr_read v14, a68 // (op 23112) 
v_accvgpr_read v15, a69 // (op 23112) 
v_accvgpr_read v16, a70 // (op 23112) 
v_accvgpr_read v17, a71 // (op 23112) 
v_mul_f32 v10, v19, v14 // (op 23112) 
v_mul_f32 v11, v19, v15 // (op 23112) 
v_mul_f32 v12, v19, v16 // (op 23112) 
v_mul_f32 v13, v19, v17 // (op 23112) 



s_lshl_b32 s0, s4, 7 // (op 23096) 
s_add_u32 s0, s0, 256 // (op 23096) 
v_add_u32 v0, v21, s0 // (op 23096) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23096) Store value


v_accvgpr_read v14, a72 // (op 23117) 
v_accvgpr_read v15, a73 // (op 23117) 
v_accvgpr_read v16, a74 // (op 23117) 
v_accvgpr_read v17, a75 // (op 23117) 
v_mul_f32 v10, v19, v14 // (op 23117) 
v_mul_f32 v11, v19, v15 // (op 23117) 
v_mul_f32 v12, v19, v16 // (op 23117) 
v_mul_f32 v13, v19, v17 // (op 23117) 



s_lshl_b32 s0, s4, 8 // (op 23098) 
s_add_u32 s0, s0, 256 // (op 23098) 
v_add_u32 v0, v21, s0 // (op 23098) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23098) Store value


v_accvgpr_read v14, a76 // (op 23122) 
v_accvgpr_read v15, a77 // (op 23122) 
v_accvgpr_read v16, a78 // (op 23122) 
v_accvgpr_read v17, a79 // (op 23122) 
v_mul_f32 v10, v19, v14 // (op 23122) 
v_mul_f32 v11, v19, v15 // (op 23122) 
v_mul_f32 v12, v19, v16 // (op 23122) 
v_mul_f32 v13, v19, v17 // (op 23122) 



s_lshl_b32 s0, s4, 7 // (op 23100) 
s_mul_i32 s1, 3, s0 // (op 23100) 
s_add_u32 s1, s1, 256 // (op 23100) 

v_add_u32 v0, v21, s1 // (op 23100) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23100) Store value


v_accvgpr_read v14, a80 // (op 23127) 
v_accvgpr_read v15, a81 // (op 23127) 
v_accvgpr_read v16, a82 // (op 23127) 
v_accvgpr_read v17, a83 // (op 23127) 
v_mul_f32 v10, v19, v14 // (op 23127) 
v_mul_f32 v11, v19, v15 // (op 23127) 
v_mul_f32 v12, v19, v16 // (op 23127) 
v_mul_f32 v13, v19, v17 // (op 23127) 



s_lshl_b32 s0, s4, 9 // (op 23102) 
s_add_u32 s0, s0, 256 // (op 23102) 
v_add_u32 v0, v21, s0 // (op 23102) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23102) Store value


v_accvgpr_read v14, a84 // (op 23132) 
v_accvgpr_read v15, a85 // (op 23132) 
v_accvgpr_read v16, a86 // (op 23132) 
v_accvgpr_read v17, a87 // (op 23132) 
v_mul_f32 v10, v19, v14 // (op 23132) 
v_mul_f32 v11, v19, v15 // (op 23132) 
v_mul_f32 v12, v19, v16 // (op 23132) 
v_mul_f32 v13, v19, v17 // (op 23132) 



s_lshl_b32 s0, s4, 7 // (op 23104) 
s_mul_i32 s1, 5, s0 // (op 23104) 
s_add_u32 s1, s1, 256 // (op 23104) 

v_add_u32 v0, v21, s1 // (op 23104) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23104) Store value


v_accvgpr_read v14, a88 // (op 23137) 
v_accvgpr_read v15, a89 // (op 23137) 
v_accvgpr_read v16, a90 // (op 23137) 
v_accvgpr_read v17, a91 // (op 23137) 
v_mul_f32 v10, v19, v14 // (op 23137) 
v_mul_f32 v11, v19, v15 // (op 23137) 
v_mul_f32 v12, v19, v16 // (op 23137) 
v_mul_f32 v13, v19, v17 // (op 23137) 



s_lshl_b32 s0, s4, 7 // (op 23106) 
s_mul_i32 s1, 6, s0 // (op 23106) 
s_add_u32 s1, s1, 256 // (op 23106) 

v_add_u32 v0, v21, s1 // (op 23106) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23106) Store value


v_accvgpr_read v14, a92 // (op 23142) 
v_accvgpr_read v15, a93 // (op 23142) 
v_accvgpr_read v16, a94 // (op 23142) 
v_accvgpr_read v17, a95 // (op 23142) 
v_mul_f32 v10, v19, v14 // (op 23142) 
v_mul_f32 v11, v19, v15 // (op 23142) 
v_mul_f32 v12, v19, v16 // (op 23142) 
v_mul_f32 v13, v19, v17 // (op 23142) 



s_lshl_b32 s0, s4, 7 // (op 23108) 
s_mul_i32 s1, 7, s0 // (op 23108) 
s_add_u32 s1, s1, 256 // (op 23108) 

v_add_u32 v0, v21, s1 // (op 23108) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23108) Store value


v_accvgpr_read v14, a96 // (op 23223) 
v_accvgpr_read v15, a97 // (op 23223) 
v_accvgpr_read v16, a98 // (op 23223) 
v_accvgpr_read v17, a99 // (op 23223) 
v_mul_f32 v10, v19, v14 // (op 23223) 
v_mul_f32 v11, v19, v15 // (op 23223) 
v_mul_f32 v12, v19, v16 // (op 23223) 
v_mul_f32 v13, v19, v17 // (op 23223) 



v_add_u32 v0, 384, v21 // (op 23226) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23226) Store value


v_accvgpr_read v14, a100 // (op 23244) 
v_accvgpr_read v15, a101 // (op 23244) 
v_accvgpr_read v16, a102 // (op 23244) 
v_accvgpr_read v17, a103 // (op 23244) 
v_mul_f32 v10, v19, v14 // (op 23244) 
v_mul_f32 v11, v19, v15 // (op 23244) 
v_mul_f32 v12, v19, v16 // (op 23244) 
v_mul_f32 v13, v19, v17 // (op 23244) 



s_lshl_b32 s0, s4, 7 // (op 23228) 
s_add_u32 s0, s0, 384 // (op 23228) 
v_add_u32 v0, v21, s0 // (op 23228) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23228) Store value


v_accvgpr_read v14, a104 // (op 23249) 
v_accvgpr_read v15, a105 // (op 23249) 
v_accvgpr_read v16, a106 // (op 23249) 
v_accvgpr_read v17, a107 // (op 23249) 
v_mul_f32 v10, v19, v14 // (op 23249) 
v_mul_f32 v11, v19, v15 // (op 23249) 
v_mul_f32 v12, v19, v16 // (op 23249) 
v_mul_f32 v13, v19, v17 // (op 23249) 



s_lshl_b32 s0, s4, 8 // (op 23230) 
s_add_u32 s0, s0, 384 // (op 23230) 
v_add_u32 v0, v21, s0 // (op 23230) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23230) Store value


v_accvgpr_read v14, a108 // (op 23254) 
v_accvgpr_read v15, a109 // (op 23254) 
v_accvgpr_read v16, a110 // (op 23254) 
v_accvgpr_read v17, a111 // (op 23254) 
v_mul_f32 v10, v19, v14 // (op 23254) 
v_mul_f32 v11, v19, v15 // (op 23254) 
v_mul_f32 v12, v19, v16 // (op 23254) 
v_mul_f32 v13, v19, v17 // (op 23254) 



s_lshl_b32 s0, s4, 7 // (op 23232) 
s_mul_i32 s1, 3, s0 // (op 23232) 
s_add_u32 s1, s1, 384 // (op 23232) 

v_add_u32 v0, v21, s1 // (op 23232) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23232) Store value


v_accvgpr_read v14, a112 // (op 23259) 
v_accvgpr_read v15, a113 // (op 23259) 
v_accvgpr_read v16, a114 // (op 23259) 
v_accvgpr_read v17, a115 // (op 23259) 
v_mul_f32 v10, v19, v14 // (op 23259) 
v_mul_f32 v11, v19, v15 // (op 23259) 
v_mul_f32 v12, v19, v16 // (op 23259) 
v_mul_f32 v13, v19, v17 // (op 23259) 



s_lshl_b32 s0, s4, 9 // (op 23234) 
s_add_u32 s0, s0, 384 // (op 23234) 
v_add_u32 v0, v21, s0 // (op 23234) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23234) Store value


v_accvgpr_read v14, a116 // (op 23264) 
v_accvgpr_read v15, a117 // (op 23264) 
v_accvgpr_read v16, a118 // (op 23264) 
v_accvgpr_read v17, a119 // (op 23264) 
v_mul_f32 v10, v19, v14 // (op 23264) 
v_mul_f32 v11, v19, v15 // (op 23264) 
v_mul_f32 v12, v19, v16 // (op 23264) 
v_mul_f32 v13, v19, v17 // (op 23264) 



s_lshl_b32 s0, s4, 7 // (op 23236) 
s_mul_i32 s1, 5, s0 // (op 23236) 
s_add_u32 s1, s1, 384 // (op 23236) 

v_add_u32 v0, v21, s1 // (op 23236) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23236) Store value


v_accvgpr_read v14, a120 // (op 23269) 
v_accvgpr_read v15, a121 // (op 23269) 
v_accvgpr_read v16, a122 // (op 23269) 
v_accvgpr_read v17, a123 // (op 23269) 
v_mul_f32 v10, v19, v14 // (op 23269) 
v_mul_f32 v11, v19, v15 // (op 23269) 
v_mul_f32 v12, v19, v16 // (op 23269) 
v_mul_f32 v13, v19, v17 // (op 23269) 



s_lshl_b32 s0, s4, 7 // (op 23238) 
s_mul_i32 s1, 6, s0 // (op 23238) 
s_add_u32 s1, s1, 384 // (op 23238) 

v_add_u32 v0, v21, s1 // (op 23238) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23238) Store value


v_accvgpr_read v14, a124 // (op 23274) 
v_accvgpr_read v15, a125 // (op 23274) 
v_accvgpr_read v16, a126 // (op 23274) 
v_accvgpr_read v17, a127 // (op 23274) 
v_mul_f32 v10, v19, v14 // (op 23274) 
v_mul_f32 v11, v19, v15 // (op 23274) 
v_mul_f32 v12, v19, v16 // (op 23274) 
v_mul_f32 v13, v19, v17 // (op 23274) 



s_lshl_b32 s0, s4, 7 // (op 23240) 
s_mul_i32 s1, 7, s0 // (op 23240) 
s_add_u32 s1, s1, 384 // (op 23240) 

v_add_u32 v0, v21, s1 // (op 23240) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23240) Store value


v_accvgpr_read v14, a128 // (op 23355) 
v_accvgpr_read v15, a129 // (op 23355) 
v_accvgpr_read v16, a130 // (op 23355) 
v_accvgpr_read v17, a131 // (op 23355) 
v_mul_f32 v10, v19, v14 // (op 23355) 
v_mul_f32 v11, v19, v15 // (op 23355) 
v_mul_f32 v12, v19, v16 // (op 23355) 
v_mul_f32 v13, v19, v17 // (op 23355) 



v_add_u32 v0, 512, v21 // (op 23358) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23358) Store value


v_accvgpr_read v14, a132 // (op 23376) 
v_accvgpr_read v15, a133 // (op 23376) 
v_accvgpr_read v16, a134 // (op 23376) 
v_accvgpr_read v17, a135 // (op 23376) 
v_mul_f32 v10, v19, v14 // (op 23376) 
v_mul_f32 v11, v19, v15 // (op 23376) 
v_mul_f32 v12, v19, v16 // (op 23376) 
v_mul_f32 v13, v19, v17 // (op 23376) 



s_lshl_b32 s0, s4, 7 // (op 23360) 
s_add_u32 s0, s0, 512 // (op 23360) 
v_add_u32 v0, v21, s0 // (op 23360) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23360) Store value


v_accvgpr_read v14, a136 // (op 23381) 
v_accvgpr_read v15, a137 // (op 23381) 
v_accvgpr_read v16, a138 // (op 23381) 
v_accvgpr_read v17, a139 // (op 23381) 
v_mul_f32 v10, v19, v14 // (op 23381) 
v_mul_f32 v11, v19, v15 // (op 23381) 
v_mul_f32 v12, v19, v16 // (op 23381) 
v_mul_f32 v13, v19, v17 // (op 23381) 



s_lshl_b32 s0, s4, 8 // (op 23362) 
s_add_u32 s0, s0, 512 // (op 23362) 
v_add_u32 v0, v21, s0 // (op 23362) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23362) Store value


v_accvgpr_read v14, a140 // (op 23386) 
v_accvgpr_read v15, a141 // (op 23386) 
v_accvgpr_read v16, a142 // (op 23386) 
v_accvgpr_read v17, a143 // (op 23386) 
v_mul_f32 v10, v19, v14 // (op 23386) 
v_mul_f32 v11, v19, v15 // (op 23386) 
v_mul_f32 v12, v19, v16 // (op 23386) 
v_mul_f32 v13, v19, v17 // (op 23386) 



s_lshl_b32 s0, s4, 7 // (op 23364) 
s_mul_i32 s1, 3, s0 // (op 23364) 
s_add_u32 s1, s1, 512 // (op 23364) 

v_add_u32 v0, v21, s1 // (op 23364) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23364) Store value


v_accvgpr_read v14, a144 // (op 23391) 
v_accvgpr_read v15, a145 // (op 23391) 
v_accvgpr_read v16, a146 // (op 23391) 
v_accvgpr_read v17, a147 // (op 23391) 
v_mul_f32 v10, v19, v14 // (op 23391) 
v_mul_f32 v11, v19, v15 // (op 23391) 
v_mul_f32 v12, v19, v16 // (op 23391) 
v_mul_f32 v13, v19, v17 // (op 23391) 



s_lshl_b32 s0, s4, 9 // (op 23366) 
s_add_u32 s0, s0, 512 // (op 23366) 
v_add_u32 v0, v21, s0 // (op 23366) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23366) Store value


v_accvgpr_read v14, a148 // (op 23396) 
v_accvgpr_read v15, a149 // (op 23396) 
v_accvgpr_read v16, a150 // (op 23396) 
v_accvgpr_read v17, a151 // (op 23396) 
v_mul_f32 v10, v19, v14 // (op 23396) 
v_mul_f32 v11, v19, v15 // (op 23396) 
v_mul_f32 v12, v19, v16 // (op 23396) 
v_mul_f32 v13, v19, v17 // (op 23396) 



s_lshl_b32 s0, s4, 7 // (op 23368) 
s_mul_i32 s1, 5, s0 // (op 23368) 
s_add_u32 s1, s1, 512 // (op 23368) 

v_add_u32 v0, v21, s1 // (op 23368) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23368) Store value


v_accvgpr_read v14, a152 // (op 23401) 
v_accvgpr_read v15, a153 // (op 23401) 
v_accvgpr_read v16, a154 // (op 23401) 
v_accvgpr_read v17, a155 // (op 23401) 
v_mul_f32 v10, v19, v14 // (op 23401) 
v_mul_f32 v11, v19, v15 // (op 23401) 
v_mul_f32 v12, v19, v16 // (op 23401) 
v_mul_f32 v13, v19, v17 // (op 23401) 



s_lshl_b32 s0, s4, 7 // (op 23370) 
s_mul_i32 s1, 6, s0 // (op 23370) 
s_add_u32 s1, s1, 512 // (op 23370) 

v_add_u32 v0, v21, s1 // (op 23370) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23370) Store value


v_accvgpr_read v14, a156 // (op 23406) 
v_accvgpr_read v15, a157 // (op 23406) 
v_accvgpr_read v16, a158 // (op 23406) 
v_accvgpr_read v17, a159 // (op 23406) 
v_mul_f32 v10, v19, v14 // (op 23406) 
v_mul_f32 v11, v19, v15 // (op 23406) 
v_mul_f32 v12, v19, v16 // (op 23406) 
v_mul_f32 v13, v19, v17 // (op 23406) 



s_lshl_b32 s0, s4, 7 // (op 23372) 
s_mul_i32 s1, 7, s0 // (op 23372) 
s_add_u32 s1, s1, 512 // (op 23372) 

v_add_u32 v0, v21, s1 // (op 23372) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23372) Store value


v_accvgpr_read v14, a160 // (op 23487) 
v_accvgpr_read v15, a161 // (op 23487) 
v_accvgpr_read v16, a162 // (op 23487) 
v_accvgpr_read v17, a163 // (op 23487) 
v_mul_f32 v10, v19, v14 // (op 23487) 
v_mul_f32 v11, v19, v15 // (op 23487) 
v_mul_f32 v12, v19, v16 // (op 23487) 
v_mul_f32 v13, v19, v17 // (op 23487) 



v_add_u32 v0, 640, v21 // (op 23490) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23490) Store value


v_accvgpr_read v14, a164 // (op 23508) 
v_accvgpr_read v15, a165 // (op 23508) 
v_accvgpr_read v16, a166 // (op 23508) 
v_accvgpr_read v17, a167 // (op 23508) 
v_mul_f32 v10, v19, v14 // (op 23508) 
v_mul_f32 v11, v19, v15 // (op 23508) 
v_mul_f32 v12, v19, v16 // (op 23508) 
v_mul_f32 v13, v19, v17 // (op 23508) 



s_lshl_b32 s0, s4, 7 // (op 23492) 
s_add_u32 s0, s0, 640 // (op 23492) 
v_add_u32 v0, v21, s0 // (op 23492) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23492) Store value


v_accvgpr_read v14, a168 // (op 23513) 
v_accvgpr_read v15, a169 // (op 23513) 
v_accvgpr_read v16, a170 // (op 23513) 
v_accvgpr_read v17, a171 // (op 23513) 
v_mul_f32 v10, v19, v14 // (op 23513) 
v_mul_f32 v11, v19, v15 // (op 23513) 
v_mul_f32 v12, v19, v16 // (op 23513) 
v_mul_f32 v13, v19, v17 // (op 23513) 



s_lshl_b32 s0, s4, 8 // (op 23494) 
s_add_u32 s0, s0, 640 // (op 23494) 
v_add_u32 v0, v21, s0 // (op 23494) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23494) Store value


v_accvgpr_read v14, a172 // (op 23518) 
v_accvgpr_read v15, a173 // (op 23518) 
v_accvgpr_read v16, a174 // (op 23518) 
v_accvgpr_read v17, a175 // (op 23518) 
v_mul_f32 v10, v19, v14 // (op 23518) 
v_mul_f32 v11, v19, v15 // (op 23518) 
v_mul_f32 v12, v19, v16 // (op 23518) 
v_mul_f32 v13, v19, v17 // (op 23518) 



s_lshl_b32 s0, s4, 7 // (op 23496) 
s_mul_i32 s1, 3, s0 // (op 23496) 
s_add_u32 s1, s1, 640 // (op 23496) 

v_add_u32 v0, v21, s1 // (op 23496) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23496) Store value


v_accvgpr_read v14, a176 // (op 23523) 
v_accvgpr_read v15, a177 // (op 23523) 
v_accvgpr_read v16, a178 // (op 23523) 
v_accvgpr_read v17, a179 // (op 23523) 
v_mul_f32 v10, v19, v14 // (op 23523) 
v_mul_f32 v11, v19, v15 // (op 23523) 
v_mul_f32 v12, v19, v16 // (op 23523) 
v_mul_f32 v13, v19, v17 // (op 23523) 



s_lshl_b32 s0, s4, 9 // (op 23498) 
s_add_u32 s0, s0, 640 // (op 23498) 
v_add_u32 v0, v21, s0 // (op 23498) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23498) Store value


v_accvgpr_read v14, a180 // (op 23528) 
v_accvgpr_read v15, a181 // (op 23528) 
v_accvgpr_read v16, a182 // (op 23528) 
v_accvgpr_read v17, a183 // (op 23528) 
v_mul_f32 v10, v19, v14 // (op 23528) 
v_mul_f32 v11, v19, v15 // (op 23528) 
v_mul_f32 v12, v19, v16 // (op 23528) 
v_mul_f32 v13, v19, v17 // (op 23528) 



s_lshl_b32 s0, s4, 7 // (op 23500) 
s_mul_i32 s1, 5, s0 // (op 23500) 
s_add_u32 s1, s1, 640 // (op 23500) 

v_add_u32 v0, v21, s1 // (op 23500) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23500) Store value


v_accvgpr_read v14, a184 // (op 23533) 
v_accvgpr_read v15, a185 // (op 23533) 
v_accvgpr_read v16, a186 // (op 23533) 
v_accvgpr_read v17, a187 // (op 23533) 
v_mul_f32 v10, v19, v14 // (op 23533) 
v_mul_f32 v11, v19, v15 // (op 23533) 
v_mul_f32 v12, v19, v16 // (op 23533) 
v_mul_f32 v13, v19, v17 // (op 23533) 



s_lshl_b32 s0, s4, 7 // (op 23502) 
s_mul_i32 s1, 6, s0 // (op 23502) 
s_add_u32 s1, s1, 640 // (op 23502) 

v_add_u32 v0, v21, s1 // (op 23502) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23502) Store value


v_accvgpr_read v14, a188 // (op 23538) 
v_accvgpr_read v15, a189 // (op 23538) 
v_accvgpr_read v16, a190 // (op 23538) 
v_accvgpr_read v17, a191 // (op 23538) 
v_mul_f32 v10, v19, v14 // (op 23538) 
v_mul_f32 v11, v19, v15 // (op 23538) 
v_mul_f32 v12, v19, v16 // (op 23538) 
v_mul_f32 v13, v19, v17 // (op 23538) 



s_lshl_b32 s0, s4, 7 // (op 23504) 
s_mul_i32 s1, 7, s0 // (op 23504) 
s_add_u32 s1, s1, 640 // (op 23504) 

v_add_u32 v0, v21, s1 // (op 23504) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23504) Store value


v_accvgpr_read v14, a192 // (op 23619) 
v_accvgpr_read v15, a193 // (op 23619) 
v_accvgpr_read v16, a194 // (op 23619) 
v_accvgpr_read v17, a195 // (op 23619) 
v_mul_f32 v10, v19, v14 // (op 23619) 
v_mul_f32 v11, v19, v15 // (op 23619) 
v_mul_f32 v12, v19, v16 // (op 23619) 
v_mul_f32 v13, v19, v17 // (op 23619) 



v_add_u32 v0, 768, v21 // (op 23622) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23622) Store value


v_accvgpr_read v14, a196 // (op 23640) 
v_accvgpr_read v15, a197 // (op 23640) 
v_accvgpr_read v16, a198 // (op 23640) 
v_accvgpr_read v17, a199 // (op 23640) 
v_mul_f32 v10, v19, v14 // (op 23640) 
v_mul_f32 v11, v19, v15 // (op 23640) 
v_mul_f32 v12, v19, v16 // (op 23640) 
v_mul_f32 v13, v19, v17 // (op 23640) 



s_lshl_b32 s0, s4, 7 // (op 23624) 
s_add_u32 s0, s0, 768 // (op 23624) 
v_add_u32 v0, v21, s0 // (op 23624) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23624) Store value


v_accvgpr_read v14, a200 // (op 23645) 
v_accvgpr_read v15, a201 // (op 23645) 
v_accvgpr_read v16, a202 // (op 23645) 
v_accvgpr_read v17, a203 // (op 23645) 
v_mul_f32 v10, v19, v14 // (op 23645) 
v_mul_f32 v11, v19, v15 // (op 23645) 
v_mul_f32 v12, v19, v16 // (op 23645) 
v_mul_f32 v13, v19, v17 // (op 23645) 



s_lshl_b32 s0, s4, 8 // (op 23626) 
s_add_u32 s0, s0, 768 // (op 23626) 
v_add_u32 v0, v21, s0 // (op 23626) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23626) Store value


v_accvgpr_read v14, a204 // (op 23650) 
v_accvgpr_read v15, a205 // (op 23650) 
v_accvgpr_read v16, a206 // (op 23650) 
v_accvgpr_read v17, a207 // (op 23650) 
v_mul_f32 v10, v19, v14 // (op 23650) 
v_mul_f32 v11, v19, v15 // (op 23650) 
v_mul_f32 v12, v19, v16 // (op 23650) 
v_mul_f32 v13, v19, v17 // (op 23650) 



s_lshl_b32 s0, s4, 7 // (op 23628) 
s_mul_i32 s1, 3, s0 // (op 23628) 
s_add_u32 s1, s1, 768 // (op 23628) 

v_add_u32 v0, v21, s1 // (op 23628) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23628) Store value


v_accvgpr_read v14, a208 // (op 23655) 
v_accvgpr_read v15, a209 // (op 23655) 
v_accvgpr_read v16, a210 // (op 23655) 
v_accvgpr_read v17, a211 // (op 23655) 
v_mul_f32 v10, v19, v14 // (op 23655) 
v_mul_f32 v11, v19, v15 // (op 23655) 
v_mul_f32 v12, v19, v16 // (op 23655) 
v_mul_f32 v13, v19, v17 // (op 23655) 



s_lshl_b32 s0, s4, 9 // (op 23630) 
s_add_u32 s0, s0, 768 // (op 23630) 
v_add_u32 v0, v21, s0 // (op 23630) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23630) Store value


v_accvgpr_read v14, a212 // (op 23660) 
v_accvgpr_read v15, a213 // (op 23660) 
v_accvgpr_read v16, a214 // (op 23660) 
v_accvgpr_read v17, a215 // (op 23660) 
v_mul_f32 v10, v19, v14 // (op 23660) 
v_mul_f32 v11, v19, v15 // (op 23660) 
v_mul_f32 v12, v19, v16 // (op 23660) 
v_mul_f32 v13, v19, v17 // (op 23660) 



s_lshl_b32 s0, s4, 7 // (op 23632) 
s_mul_i32 s1, 5, s0 // (op 23632) 
s_add_u32 s1, s1, 768 // (op 23632) 

v_add_u32 v0, v21, s1 // (op 23632) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23632) Store value


v_accvgpr_read v14, a216 // (op 23665) 
v_accvgpr_read v15, a217 // (op 23665) 
v_accvgpr_read v16, a218 // (op 23665) 
v_accvgpr_read v17, a219 // (op 23665) 
v_mul_f32 v10, v19, v14 // (op 23665) 
v_mul_f32 v11, v19, v15 // (op 23665) 
v_mul_f32 v12, v19, v16 // (op 23665) 
v_mul_f32 v13, v19, v17 // (op 23665) 



s_lshl_b32 s0, s4, 7 // (op 23634) 
s_mul_i32 s1, 6, s0 // (op 23634) 
s_add_u32 s1, s1, 768 // (op 23634) 

v_add_u32 v0, v21, s1 // (op 23634) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23634) Store value


v_accvgpr_read v14, a220 // (op 23670) 
v_accvgpr_read v15, a221 // (op 23670) 
v_accvgpr_read v16, a222 // (op 23670) 
v_accvgpr_read v17, a223 // (op 23670) 
v_mul_f32 v10, v19, v14 // (op 23670) 
v_mul_f32 v11, v19, v15 // (op 23670) 
v_mul_f32 v12, v19, v16 // (op 23670) 
v_mul_f32 v13, v19, v17 // (op 23670) 



s_lshl_b32 s0, s4, 7 // (op 23636) 
s_mul_i32 s1, 7, s0 // (op 23636) 
s_add_u32 s1, s1, 768 // (op 23636) 

v_add_u32 v0, v21, s1 // (op 23636) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23636) Store value


v_accvgpr_read v14, a224 // (op 23751) 
v_accvgpr_read v15, a225 // (op 23751) 
v_accvgpr_read v16, a226 // (op 23751) 
v_accvgpr_read v17, a227 // (op 23751) 
v_mul_f32 v10, v19, v14 // (op 23751) 
v_mul_f32 v11, v19, v15 // (op 23751) 
v_mul_f32 v12, v19, v16 // (op 23751) 
v_mul_f32 v13, v19, v17 // (op 23751) 



v_add_u32 v0, 896, v21 // (op 23754) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23754) Store value


v_accvgpr_read v14, a228 // (op 23772) 
v_accvgpr_read v15, a229 // (op 23772) 
v_accvgpr_read v16, a230 // (op 23772) 
v_accvgpr_read v17, a231 // (op 23772) 
v_mul_f32 v10, v19, v14 // (op 23772) 
v_mul_f32 v11, v19, v15 // (op 23772) 
v_mul_f32 v12, v19, v16 // (op 23772) 
v_mul_f32 v13, v19, v17 // (op 23772) 



s_lshl_b32 s0, s4, 7 // (op 23756) 
s_add_u32 s0, s0, 896 // (op 23756) 
v_add_u32 v0, v21, s0 // (op 23756) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23756) Store value


v_accvgpr_read v14, a232 // (op 23777) 
v_accvgpr_read v15, a233 // (op 23777) 
v_accvgpr_read v16, a234 // (op 23777) 
v_accvgpr_read v17, a235 // (op 23777) 
v_mul_f32 v10, v19, v14 // (op 23777) 
v_mul_f32 v11, v19, v15 // (op 23777) 
v_mul_f32 v12, v19, v16 // (op 23777) 
v_mul_f32 v13, v19, v17 // (op 23777) 



s_lshl_b32 s0, s4, 8 // (op 23758) 
s_add_u32 s0, s0, 896 // (op 23758) 
v_add_u32 v0, v21, s0 // (op 23758) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23758) Store value


v_accvgpr_read v14, a236 // (op 23782) 
v_accvgpr_read v15, a237 // (op 23782) 
v_accvgpr_read v16, a238 // (op 23782) 
v_accvgpr_read v17, a239 // (op 23782) 
v_mul_f32 v10, v19, v14 // (op 23782) 
v_mul_f32 v11, v19, v15 // (op 23782) 
v_mul_f32 v12, v19, v16 // (op 23782) 
v_mul_f32 v13, v19, v17 // (op 23782) 



s_lshl_b32 s0, s4, 7 // (op 23760) 
s_mul_i32 s1, 3, s0 // (op 23760) 
s_add_u32 s1, s1, 896 // (op 23760) 

v_add_u32 v0, v21, s1 // (op 23760) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23760) Store value


v_accvgpr_read v14, a240 // (op 23787) 
v_accvgpr_read v15, a241 // (op 23787) 
v_accvgpr_read v16, a242 // (op 23787) 
v_accvgpr_read v17, a243 // (op 23787) 
v_mul_f32 v10, v19, v14 // (op 23787) 
v_mul_f32 v11, v19, v15 // (op 23787) 
v_mul_f32 v12, v19, v16 // (op 23787) 
v_mul_f32 v13, v19, v17 // (op 23787) 



s_lshl_b32 s0, s4, 9 // (op 23762) 
s_add_u32 s0, s0, 896 // (op 23762) 
v_add_u32 v0, v21, s0 // (op 23762) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23762) Store value


v_accvgpr_read v14, a244 // (op 23792) 
v_accvgpr_read v15, a245 // (op 23792) 
v_accvgpr_read v16, a246 // (op 23792) 
v_accvgpr_read v17, a247 // (op 23792) 
v_mul_f32 v10, v19, v14 // (op 23792) 
v_mul_f32 v11, v19, v15 // (op 23792) 
v_mul_f32 v12, v19, v16 // (op 23792) 
v_mul_f32 v13, v19, v17 // (op 23792) 



s_lshl_b32 s0, s4, 7 // (op 23764) 
s_mul_i32 s1, 5, s0 // (op 23764) 
s_add_u32 s1, s1, 896 // (op 23764) 

v_add_u32 v0, v21, s1 // (op 23764) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23764) Store value


v_accvgpr_read v14, a248 // (op 23797) 
v_accvgpr_read v15, a249 // (op 23797) 
v_accvgpr_read v16, a250 // (op 23797) 
v_accvgpr_read v17, a251 // (op 23797) 
v_mul_f32 v10, v19, v14 // (op 23797) 
v_mul_f32 v11, v19, v15 // (op 23797) 
v_mul_f32 v12, v19, v16 // (op 23797) 
v_mul_f32 v13, v19, v17 // (op 23797) 



s_lshl_b32 s0, s4, 7 // (op 23766) 
s_mul_i32 s1, 6, s0 // (op 23766) 
s_add_u32 s1, s1, 896 // (op 23766) 

v_add_u32 v0, v21, s1 // (op 23766) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23766) Store value


v_accvgpr_read v14, a252 // (op 23802) 
v_accvgpr_read v15, a253 // (op 23802) 
v_accvgpr_read v16, a254 // (op 23802) 
v_accvgpr_read v17, a255 // (op 23802) 
v_mul_f32 v10, v19, v14 // (op 23802) 
v_mul_f32 v11, v19, v15 // (op 23802) 
v_mul_f32 v12, v19, v16 // (op 23802) 
v_mul_f32 v13, v19, v17 // (op 23802) 



s_lshl_b32 s0, s4, 7 // (op 23768) 
s_mul_i32 s1, 7, s0 // (op 23768) 
s_add_u32 s1, s1, 896 // (op 23768) 

v_add_u32 v0, v21, s1 // (op 23768) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 23768) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalBottom_0_beta_133
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalFalse_0_beta_133:


v_add_u32 v0, 896, v20 // (op 25202) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25202) Load value

v_accvgpr_read v10, a224 // (op 25182) 
v_accvgpr_read v11, a225 // (op 25182) 
v_accvgpr_read v12, a226 // (op 25182) 
v_accvgpr_read v13, a227 // (op 25182) 
v_mul_f32 v6, v19, v10 // (op 25182) 
v_mul_f32 v7, v19, v11 // (op 25182) 
v_mul_f32 v8, v19, v12 // (op 25182) 
v_mul_f32 v9, v19, v13 // (op 25182) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25201) 
v_mul_f32 v11, v18, v15 // (op 25201) 
v_mul_f32 v12, v18, v16 // (op 25201) 
v_mul_f32 v13, v18, v17 // (op 25201) 

v_add_f32 v14, v10, v6 // (op 25183) 
v_add_f32 v15, v11, v7 // (op 25183) 
v_add_f32 v16, v12, v8 // (op 25183) 
v_add_f32 v17, v13, v9 // (op 25183) 




v_add_u32 v0, 896, v21 // (op 25185) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25185) Store value




s_lshl_b32 s0, s2, 7 // (op 25206) 
s_add_u32 s0, s0, 896 // (op 25206) 
v_add_u32 v0, v20, s0 // (op 25206) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25206) Load value

v_accvgpr_read v10, a228 // (op 25232) 
v_accvgpr_read v11, a229 // (op 25232) 
v_accvgpr_read v12, a230 // (op 25232) 
v_accvgpr_read v13, a231 // (op 25232) 
v_mul_f32 v6, v19, v10 // (op 25232) 
v_mul_f32 v7, v19, v11 // (op 25232) 
v_mul_f32 v8, v19, v12 // (op 25232) 
v_mul_f32 v9, v19, v13 // (op 25232) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25204) 
v_mul_f32 v11, v18, v15 // (op 25204) 
v_mul_f32 v12, v18, v16 // (op 25204) 
v_mul_f32 v13, v18, v17 // (op 25204) 

v_add_f32 v14, v10, v6 // (op 25205) 
v_add_f32 v15, v11, v7 // (op 25205) 
v_add_f32 v16, v12, v8 // (op 25205) 
v_add_f32 v17, v13, v9 // (op 25205) 




s_lshl_b32 s0, s4, 7 // (op 25187) 
s_add_u32 s0, s0, 896 // (op 25187) 
v_add_u32 v0, v21, s0 // (op 25187) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25187) Store value




s_lshl_b32 s0, s2, 8 // (op 25210) 
s_add_u32 s0, s0, 896 // (op 25210) 
v_add_u32 v0, v20, s0 // (op 25210) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25210) Load value

v_accvgpr_read v10, a232 // (op 25234) 
v_accvgpr_read v11, a233 // (op 25234) 
v_accvgpr_read v12, a234 // (op 25234) 
v_accvgpr_read v13, a235 // (op 25234) 
v_mul_f32 v6, v19, v10 // (op 25234) 
v_mul_f32 v7, v19, v11 // (op 25234) 
v_mul_f32 v8, v19, v12 // (op 25234) 
v_mul_f32 v9, v19, v13 // (op 25234) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25208) 
v_mul_f32 v11, v18, v15 // (op 25208) 
v_mul_f32 v12, v18, v16 // (op 25208) 
v_mul_f32 v13, v18, v17 // (op 25208) 

v_add_f32 v14, v10, v6 // (op 25209) 
v_add_f32 v15, v11, v7 // (op 25209) 
v_add_f32 v16, v12, v8 // (op 25209) 
v_add_f32 v17, v13, v9 // (op 25209) 




s_lshl_b32 s0, s4, 8 // (op 25189) 
s_add_u32 s0, s0, 896 // (op 25189) 
v_add_u32 v0, v21, s0 // (op 25189) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25189) Store value




s_lshl_b32 s0, s2, 7 // (op 25214) 
s_mul_i32 s1, 3, s0 // (op 25214) 
s_add_u32 s1, s1, 896 // (op 25214) 

v_add_u32 v0, v20, s1 // (op 25214) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25214) Load value

v_accvgpr_read v10, a236 // (op 25236) 
v_accvgpr_read v11, a237 // (op 25236) 
v_accvgpr_read v12, a238 // (op 25236) 
v_accvgpr_read v13, a239 // (op 25236) 
v_mul_f32 v6, v19, v10 // (op 25236) 
v_mul_f32 v7, v19, v11 // (op 25236) 
v_mul_f32 v8, v19, v12 // (op 25236) 
v_mul_f32 v9, v19, v13 // (op 25236) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25212) 
v_mul_f32 v11, v18, v15 // (op 25212) 
v_mul_f32 v12, v18, v16 // (op 25212) 
v_mul_f32 v13, v18, v17 // (op 25212) 

v_add_f32 v14, v10, v6 // (op 25213) 
v_add_f32 v15, v11, v7 // (op 25213) 
v_add_f32 v16, v12, v8 // (op 25213) 
v_add_f32 v17, v13, v9 // (op 25213) 




s_lshl_b32 s0, s4, 7 // (op 25191) 
s_mul_i32 s1, 3, s0 // (op 25191) 
s_add_u32 s1, s1, 896 // (op 25191) 

v_add_u32 v0, v21, s1 // (op 25191) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25191) Store value




s_lshl_b32 s0, s2, 9 // (op 25218) 
s_add_u32 s0, s0, 896 // (op 25218) 
v_add_u32 v0, v20, s0 // (op 25218) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25218) Load value

v_accvgpr_read v10, a240 // (op 25238) 
v_accvgpr_read v11, a241 // (op 25238) 
v_accvgpr_read v12, a242 // (op 25238) 
v_accvgpr_read v13, a243 // (op 25238) 
v_mul_f32 v6, v19, v10 // (op 25238) 
v_mul_f32 v7, v19, v11 // (op 25238) 
v_mul_f32 v8, v19, v12 // (op 25238) 
v_mul_f32 v9, v19, v13 // (op 25238) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25216) 
v_mul_f32 v11, v18, v15 // (op 25216) 
v_mul_f32 v12, v18, v16 // (op 25216) 
v_mul_f32 v13, v18, v17 // (op 25216) 

v_add_f32 v14, v10, v6 // (op 25217) 
v_add_f32 v15, v11, v7 // (op 25217) 
v_add_f32 v16, v12, v8 // (op 25217) 
v_add_f32 v17, v13, v9 // (op 25217) 




s_lshl_b32 s0, s4, 9 // (op 25193) 
s_add_u32 s0, s0, 896 // (op 25193) 
v_add_u32 v0, v21, s0 // (op 25193) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25193) Store value




s_lshl_b32 s0, s2, 7 // (op 25222) 
s_mul_i32 s1, 5, s0 // (op 25222) 
s_add_u32 s1, s1, 896 // (op 25222) 

v_add_u32 v0, v20, s1 // (op 25222) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25222) Load value

v_accvgpr_read v10, a244 // (op 25240) 
v_accvgpr_read v11, a245 // (op 25240) 
v_accvgpr_read v12, a246 // (op 25240) 
v_accvgpr_read v13, a247 // (op 25240) 
v_mul_f32 v6, v19, v10 // (op 25240) 
v_mul_f32 v7, v19, v11 // (op 25240) 
v_mul_f32 v8, v19, v12 // (op 25240) 
v_mul_f32 v9, v19, v13 // (op 25240) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25220) 
v_mul_f32 v11, v18, v15 // (op 25220) 
v_mul_f32 v12, v18, v16 // (op 25220) 
v_mul_f32 v13, v18, v17 // (op 25220) 

v_add_f32 v14, v10, v6 // (op 25221) 
v_add_f32 v15, v11, v7 // (op 25221) 
v_add_f32 v16, v12, v8 // (op 25221) 
v_add_f32 v17, v13, v9 // (op 25221) 




s_lshl_b32 s0, s4, 7 // (op 25195) 
s_mul_i32 s1, 5, s0 // (op 25195) 
s_add_u32 s1, s1, 896 // (op 25195) 

v_add_u32 v0, v21, s1 // (op 25195) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25195) Store value




s_lshl_b32 s0, s2, 7 // (op 25226) 
s_mul_i32 s1, 6, s0 // (op 25226) 
s_add_u32 s1, s1, 896 // (op 25226) 

v_add_u32 v0, v20, s1 // (op 25226) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25226) Load value

v_accvgpr_read v10, a248 // (op 25242) 
v_accvgpr_read v11, a249 // (op 25242) 
v_accvgpr_read v12, a250 // (op 25242) 
v_accvgpr_read v13, a251 // (op 25242) 
v_mul_f32 v6, v19, v10 // (op 25242) 
v_mul_f32 v7, v19, v11 // (op 25242) 
v_mul_f32 v8, v19, v12 // (op 25242) 
v_mul_f32 v9, v19, v13 // (op 25242) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25224) 
v_mul_f32 v11, v18, v15 // (op 25224) 
v_mul_f32 v12, v18, v16 // (op 25224) 
v_mul_f32 v13, v18, v17 // (op 25224) 

v_add_f32 v14, v10, v6 // (op 25225) 
v_add_f32 v15, v11, v7 // (op 25225) 
v_add_f32 v16, v12, v8 // (op 25225) 
v_add_f32 v17, v13, v9 // (op 25225) 




s_lshl_b32 s0, s4, 7 // (op 25197) 
s_mul_i32 s1, 6, s0 // (op 25197) 
s_add_u32 s1, s1, 896 // (op 25197) 

v_add_u32 v0, v21, s1 // (op 25197) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25197) Store value




s_lshl_b32 s0, s2, 7 // (op 25230) 
s_mul_i32 s1, 7, s0 // (op 25230) 
s_add_u32 s1, s1, 896 // (op 25230) 

v_add_u32 v0, v20, s1 // (op 25230) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25230) Load value

v_accvgpr_read v10, a252 // (op 25244) 
v_accvgpr_read v11, a253 // (op 25244) 
v_accvgpr_read v12, a254 // (op 25244) 
v_accvgpr_read v13, a255 // (op 25244) 
v_mul_f32 v6, v19, v10 // (op 25244) 
v_mul_f32 v7, v19, v11 // (op 25244) 
v_mul_f32 v8, v19, v12 // (op 25244) 
v_mul_f32 v9, v19, v13 // (op 25244) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25228) 
v_mul_f32 v11, v18, v15 // (op 25228) 
v_mul_f32 v12, v18, v16 // (op 25228) 
v_mul_f32 v13, v18, v17 // (op 25228) 

v_add_f32 v14, v10, v6 // (op 25229) 
v_add_f32 v15, v11, v7 // (op 25229) 
v_add_f32 v16, v12, v8 // (op 25229) 
v_add_f32 v17, v13, v9 // (op 25229) 




s_lshl_b32 s0, s4, 7 // (op 25199) 
s_mul_i32 s1, 7, s0 // (op 25199) 
s_add_u32 s1, s1, 896 // (op 25199) 

v_add_u32 v0, v21, s1 // (op 25199) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25199) Store value




v_add_u32 v0, 768, v20 // (op 25047) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25047) Load value

v_accvgpr_read v10, a192 // (op 25027) 
v_accvgpr_read v11, a193 // (op 25027) 
v_accvgpr_read v12, a194 // (op 25027) 
v_accvgpr_read v13, a195 // (op 25027) 
v_mul_f32 v6, v19, v10 // (op 25027) 
v_mul_f32 v7, v19, v11 // (op 25027) 
v_mul_f32 v8, v19, v12 // (op 25027) 
v_mul_f32 v9, v19, v13 // (op 25027) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25046) 
v_mul_f32 v11, v18, v15 // (op 25046) 
v_mul_f32 v12, v18, v16 // (op 25046) 
v_mul_f32 v13, v18, v17 // (op 25046) 

v_add_f32 v14, v10, v6 // (op 25028) 
v_add_f32 v15, v11, v7 // (op 25028) 
v_add_f32 v16, v12, v8 // (op 25028) 
v_add_f32 v17, v13, v9 // (op 25028) 




v_add_u32 v0, 768, v21 // (op 25030) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25030) Store value




s_lshl_b32 s0, s2, 7 // (op 25051) 
s_add_u32 s0, s0, 768 // (op 25051) 
v_add_u32 v0, v20, s0 // (op 25051) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25051) Load value

v_accvgpr_read v10, a196 // (op 25077) 
v_accvgpr_read v11, a197 // (op 25077) 
v_accvgpr_read v12, a198 // (op 25077) 
v_accvgpr_read v13, a199 // (op 25077) 
v_mul_f32 v6, v19, v10 // (op 25077) 
v_mul_f32 v7, v19, v11 // (op 25077) 
v_mul_f32 v8, v19, v12 // (op 25077) 
v_mul_f32 v9, v19, v13 // (op 25077) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25049) 
v_mul_f32 v11, v18, v15 // (op 25049) 
v_mul_f32 v12, v18, v16 // (op 25049) 
v_mul_f32 v13, v18, v17 // (op 25049) 

v_add_f32 v14, v10, v6 // (op 25050) 
v_add_f32 v15, v11, v7 // (op 25050) 
v_add_f32 v16, v12, v8 // (op 25050) 
v_add_f32 v17, v13, v9 // (op 25050) 




s_lshl_b32 s0, s4, 7 // (op 25032) 
s_add_u32 s0, s0, 768 // (op 25032) 
v_add_u32 v0, v21, s0 // (op 25032) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25032) Store value




s_lshl_b32 s0, s2, 8 // (op 25055) 
s_add_u32 s0, s0, 768 // (op 25055) 
v_add_u32 v0, v20, s0 // (op 25055) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25055) Load value

v_accvgpr_read v10, a200 // (op 25079) 
v_accvgpr_read v11, a201 // (op 25079) 
v_accvgpr_read v12, a202 // (op 25079) 
v_accvgpr_read v13, a203 // (op 25079) 
v_mul_f32 v6, v19, v10 // (op 25079) 
v_mul_f32 v7, v19, v11 // (op 25079) 
v_mul_f32 v8, v19, v12 // (op 25079) 
v_mul_f32 v9, v19, v13 // (op 25079) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25053) 
v_mul_f32 v11, v18, v15 // (op 25053) 
v_mul_f32 v12, v18, v16 // (op 25053) 
v_mul_f32 v13, v18, v17 // (op 25053) 

v_add_f32 v14, v10, v6 // (op 25054) 
v_add_f32 v15, v11, v7 // (op 25054) 
v_add_f32 v16, v12, v8 // (op 25054) 
v_add_f32 v17, v13, v9 // (op 25054) 




s_lshl_b32 s0, s4, 8 // (op 25034) 
s_add_u32 s0, s0, 768 // (op 25034) 
v_add_u32 v0, v21, s0 // (op 25034) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25034) Store value




s_lshl_b32 s0, s2, 7 // (op 25059) 
s_mul_i32 s1, 3, s0 // (op 25059) 
s_add_u32 s1, s1, 768 // (op 25059) 

v_add_u32 v0, v20, s1 // (op 25059) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25059) Load value

v_accvgpr_read v10, a204 // (op 25081) 
v_accvgpr_read v11, a205 // (op 25081) 
v_accvgpr_read v12, a206 // (op 25081) 
v_accvgpr_read v13, a207 // (op 25081) 
v_mul_f32 v6, v19, v10 // (op 25081) 
v_mul_f32 v7, v19, v11 // (op 25081) 
v_mul_f32 v8, v19, v12 // (op 25081) 
v_mul_f32 v9, v19, v13 // (op 25081) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25057) 
v_mul_f32 v11, v18, v15 // (op 25057) 
v_mul_f32 v12, v18, v16 // (op 25057) 
v_mul_f32 v13, v18, v17 // (op 25057) 

v_add_f32 v14, v10, v6 // (op 25058) 
v_add_f32 v15, v11, v7 // (op 25058) 
v_add_f32 v16, v12, v8 // (op 25058) 
v_add_f32 v17, v13, v9 // (op 25058) 




s_lshl_b32 s0, s4, 7 // (op 25036) 
s_mul_i32 s1, 3, s0 // (op 25036) 
s_add_u32 s1, s1, 768 // (op 25036) 

v_add_u32 v0, v21, s1 // (op 25036) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25036) Store value




s_lshl_b32 s0, s2, 9 // (op 25063) 
s_add_u32 s0, s0, 768 // (op 25063) 
v_add_u32 v0, v20, s0 // (op 25063) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25063) Load value

v_accvgpr_read v10, a208 // (op 25083) 
v_accvgpr_read v11, a209 // (op 25083) 
v_accvgpr_read v12, a210 // (op 25083) 
v_accvgpr_read v13, a211 // (op 25083) 
v_mul_f32 v6, v19, v10 // (op 25083) 
v_mul_f32 v7, v19, v11 // (op 25083) 
v_mul_f32 v8, v19, v12 // (op 25083) 
v_mul_f32 v9, v19, v13 // (op 25083) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25061) 
v_mul_f32 v11, v18, v15 // (op 25061) 
v_mul_f32 v12, v18, v16 // (op 25061) 
v_mul_f32 v13, v18, v17 // (op 25061) 

v_add_f32 v14, v10, v6 // (op 25062) 
v_add_f32 v15, v11, v7 // (op 25062) 
v_add_f32 v16, v12, v8 // (op 25062) 
v_add_f32 v17, v13, v9 // (op 25062) 




s_lshl_b32 s0, s4, 9 // (op 25038) 
s_add_u32 s0, s0, 768 // (op 25038) 
v_add_u32 v0, v21, s0 // (op 25038) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25038) Store value




s_lshl_b32 s0, s2, 7 // (op 25067) 
s_mul_i32 s1, 5, s0 // (op 25067) 
s_add_u32 s1, s1, 768 // (op 25067) 

v_add_u32 v0, v20, s1 // (op 25067) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25067) Load value

v_accvgpr_read v10, a212 // (op 25085) 
v_accvgpr_read v11, a213 // (op 25085) 
v_accvgpr_read v12, a214 // (op 25085) 
v_accvgpr_read v13, a215 // (op 25085) 
v_mul_f32 v6, v19, v10 // (op 25085) 
v_mul_f32 v7, v19, v11 // (op 25085) 
v_mul_f32 v8, v19, v12 // (op 25085) 
v_mul_f32 v9, v19, v13 // (op 25085) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25065) 
v_mul_f32 v11, v18, v15 // (op 25065) 
v_mul_f32 v12, v18, v16 // (op 25065) 
v_mul_f32 v13, v18, v17 // (op 25065) 

v_add_f32 v14, v10, v6 // (op 25066) 
v_add_f32 v15, v11, v7 // (op 25066) 
v_add_f32 v16, v12, v8 // (op 25066) 
v_add_f32 v17, v13, v9 // (op 25066) 




s_lshl_b32 s0, s4, 7 // (op 25040) 
s_mul_i32 s1, 5, s0 // (op 25040) 
s_add_u32 s1, s1, 768 // (op 25040) 

v_add_u32 v0, v21, s1 // (op 25040) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25040) Store value




s_lshl_b32 s0, s2, 7 // (op 25071) 
s_mul_i32 s1, 6, s0 // (op 25071) 
s_add_u32 s1, s1, 768 // (op 25071) 

v_add_u32 v0, v20, s1 // (op 25071) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25071) Load value

v_accvgpr_read v10, a216 // (op 25087) 
v_accvgpr_read v11, a217 // (op 25087) 
v_accvgpr_read v12, a218 // (op 25087) 
v_accvgpr_read v13, a219 // (op 25087) 
v_mul_f32 v6, v19, v10 // (op 25087) 
v_mul_f32 v7, v19, v11 // (op 25087) 
v_mul_f32 v8, v19, v12 // (op 25087) 
v_mul_f32 v9, v19, v13 // (op 25087) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25069) 
v_mul_f32 v11, v18, v15 // (op 25069) 
v_mul_f32 v12, v18, v16 // (op 25069) 
v_mul_f32 v13, v18, v17 // (op 25069) 

v_add_f32 v14, v10, v6 // (op 25070) 
v_add_f32 v15, v11, v7 // (op 25070) 
v_add_f32 v16, v12, v8 // (op 25070) 
v_add_f32 v17, v13, v9 // (op 25070) 




s_lshl_b32 s0, s4, 7 // (op 25042) 
s_mul_i32 s1, 6, s0 // (op 25042) 
s_add_u32 s1, s1, 768 // (op 25042) 

v_add_u32 v0, v21, s1 // (op 25042) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25042) Store value




s_lshl_b32 s0, s2, 7 // (op 25075) 
s_mul_i32 s1, 7, s0 // (op 25075) 
s_add_u32 s1, s1, 768 // (op 25075) 

v_add_u32 v0, v20, s1 // (op 25075) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 25075) Load value

v_accvgpr_read v10, a220 // (op 25089) 
v_accvgpr_read v11, a221 // (op 25089) 
v_accvgpr_read v12, a222 // (op 25089) 
v_accvgpr_read v13, a223 // (op 25089) 
v_mul_f32 v6, v19, v10 // (op 25089) 
v_mul_f32 v7, v19, v11 // (op 25089) 
v_mul_f32 v8, v19, v12 // (op 25089) 
v_mul_f32 v9, v19, v13 // (op 25089) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 25073) 
v_mul_f32 v11, v18, v15 // (op 25073) 
v_mul_f32 v12, v18, v16 // (op 25073) 
v_mul_f32 v13, v18, v17 // (op 25073) 

v_add_f32 v14, v10, v6 // (op 25074) 
v_add_f32 v15, v11, v7 // (op 25074) 
v_add_f32 v16, v12, v8 // (op 25074) 
v_add_f32 v17, v13, v9 // (op 25074) 




s_lshl_b32 s0, s4, 7 // (op 25044) 
s_mul_i32 s1, 7, s0 // (op 25044) 
s_add_u32 s1, s1, 768 // (op 25044) 

v_add_u32 v0, v21, s1 // (op 25044) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 25044) Store value




v_add_u32 v0, 640, v20 // (op 24892) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24892) Load value

v_accvgpr_read v10, a160 // (op 24872) 
v_accvgpr_read v11, a161 // (op 24872) 
v_accvgpr_read v12, a162 // (op 24872) 
v_accvgpr_read v13, a163 // (op 24872) 
v_mul_f32 v6, v19, v10 // (op 24872) 
v_mul_f32 v7, v19, v11 // (op 24872) 
v_mul_f32 v8, v19, v12 // (op 24872) 
v_mul_f32 v9, v19, v13 // (op 24872) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24891) 
v_mul_f32 v11, v18, v15 // (op 24891) 
v_mul_f32 v12, v18, v16 // (op 24891) 
v_mul_f32 v13, v18, v17 // (op 24891) 

v_add_f32 v14, v10, v6 // (op 24873) 
v_add_f32 v15, v11, v7 // (op 24873) 
v_add_f32 v16, v12, v8 // (op 24873) 
v_add_f32 v17, v13, v9 // (op 24873) 




v_add_u32 v0, 640, v21 // (op 24875) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24875) Store value




s_lshl_b32 s0, s2, 7 // (op 24896) 
s_add_u32 s0, s0, 640 // (op 24896) 
v_add_u32 v0, v20, s0 // (op 24896) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24896) Load value

v_accvgpr_read v10, a164 // (op 24922) 
v_accvgpr_read v11, a165 // (op 24922) 
v_accvgpr_read v12, a166 // (op 24922) 
v_accvgpr_read v13, a167 // (op 24922) 
v_mul_f32 v6, v19, v10 // (op 24922) 
v_mul_f32 v7, v19, v11 // (op 24922) 
v_mul_f32 v8, v19, v12 // (op 24922) 
v_mul_f32 v9, v19, v13 // (op 24922) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24894) 
v_mul_f32 v11, v18, v15 // (op 24894) 
v_mul_f32 v12, v18, v16 // (op 24894) 
v_mul_f32 v13, v18, v17 // (op 24894) 

v_add_f32 v14, v10, v6 // (op 24895) 
v_add_f32 v15, v11, v7 // (op 24895) 
v_add_f32 v16, v12, v8 // (op 24895) 
v_add_f32 v17, v13, v9 // (op 24895) 




s_lshl_b32 s0, s4, 7 // (op 24877) 
s_add_u32 s0, s0, 640 // (op 24877) 
v_add_u32 v0, v21, s0 // (op 24877) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24877) Store value




s_lshl_b32 s0, s2, 8 // (op 24900) 
s_add_u32 s0, s0, 640 // (op 24900) 
v_add_u32 v0, v20, s0 // (op 24900) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24900) Load value

v_accvgpr_read v10, a168 // (op 24924) 
v_accvgpr_read v11, a169 // (op 24924) 
v_accvgpr_read v12, a170 // (op 24924) 
v_accvgpr_read v13, a171 // (op 24924) 
v_mul_f32 v6, v19, v10 // (op 24924) 
v_mul_f32 v7, v19, v11 // (op 24924) 
v_mul_f32 v8, v19, v12 // (op 24924) 
v_mul_f32 v9, v19, v13 // (op 24924) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24898) 
v_mul_f32 v11, v18, v15 // (op 24898) 
v_mul_f32 v12, v18, v16 // (op 24898) 
v_mul_f32 v13, v18, v17 // (op 24898) 

v_add_f32 v14, v10, v6 // (op 24899) 
v_add_f32 v15, v11, v7 // (op 24899) 
v_add_f32 v16, v12, v8 // (op 24899) 
v_add_f32 v17, v13, v9 // (op 24899) 




s_lshl_b32 s0, s4, 8 // (op 24879) 
s_add_u32 s0, s0, 640 // (op 24879) 
v_add_u32 v0, v21, s0 // (op 24879) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24879) Store value




s_lshl_b32 s0, s2, 7 // (op 24904) 
s_mul_i32 s1, 3, s0 // (op 24904) 
s_add_u32 s1, s1, 640 // (op 24904) 

v_add_u32 v0, v20, s1 // (op 24904) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24904) Load value

v_accvgpr_read v10, a172 // (op 24926) 
v_accvgpr_read v11, a173 // (op 24926) 
v_accvgpr_read v12, a174 // (op 24926) 
v_accvgpr_read v13, a175 // (op 24926) 
v_mul_f32 v6, v19, v10 // (op 24926) 
v_mul_f32 v7, v19, v11 // (op 24926) 
v_mul_f32 v8, v19, v12 // (op 24926) 
v_mul_f32 v9, v19, v13 // (op 24926) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24902) 
v_mul_f32 v11, v18, v15 // (op 24902) 
v_mul_f32 v12, v18, v16 // (op 24902) 
v_mul_f32 v13, v18, v17 // (op 24902) 

v_add_f32 v14, v10, v6 // (op 24903) 
v_add_f32 v15, v11, v7 // (op 24903) 
v_add_f32 v16, v12, v8 // (op 24903) 
v_add_f32 v17, v13, v9 // (op 24903) 




s_lshl_b32 s0, s4, 7 // (op 24881) 
s_mul_i32 s1, 3, s0 // (op 24881) 
s_add_u32 s1, s1, 640 // (op 24881) 

v_add_u32 v0, v21, s1 // (op 24881) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24881) Store value




s_lshl_b32 s0, s2, 9 // (op 24908) 
s_add_u32 s0, s0, 640 // (op 24908) 
v_add_u32 v0, v20, s0 // (op 24908) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24908) Load value

v_accvgpr_read v10, a176 // (op 24928) 
v_accvgpr_read v11, a177 // (op 24928) 
v_accvgpr_read v12, a178 // (op 24928) 
v_accvgpr_read v13, a179 // (op 24928) 
v_mul_f32 v6, v19, v10 // (op 24928) 
v_mul_f32 v7, v19, v11 // (op 24928) 
v_mul_f32 v8, v19, v12 // (op 24928) 
v_mul_f32 v9, v19, v13 // (op 24928) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24906) 
v_mul_f32 v11, v18, v15 // (op 24906) 
v_mul_f32 v12, v18, v16 // (op 24906) 
v_mul_f32 v13, v18, v17 // (op 24906) 

v_add_f32 v14, v10, v6 // (op 24907) 
v_add_f32 v15, v11, v7 // (op 24907) 
v_add_f32 v16, v12, v8 // (op 24907) 
v_add_f32 v17, v13, v9 // (op 24907) 




s_lshl_b32 s0, s4, 9 // (op 24883) 
s_add_u32 s0, s0, 640 // (op 24883) 
v_add_u32 v0, v21, s0 // (op 24883) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24883) Store value




s_lshl_b32 s0, s2, 7 // (op 24912) 
s_mul_i32 s1, 5, s0 // (op 24912) 
s_add_u32 s1, s1, 640 // (op 24912) 

v_add_u32 v0, v20, s1 // (op 24912) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24912) Load value

v_accvgpr_read v10, a180 // (op 24930) 
v_accvgpr_read v11, a181 // (op 24930) 
v_accvgpr_read v12, a182 // (op 24930) 
v_accvgpr_read v13, a183 // (op 24930) 
v_mul_f32 v6, v19, v10 // (op 24930) 
v_mul_f32 v7, v19, v11 // (op 24930) 
v_mul_f32 v8, v19, v12 // (op 24930) 
v_mul_f32 v9, v19, v13 // (op 24930) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24910) 
v_mul_f32 v11, v18, v15 // (op 24910) 
v_mul_f32 v12, v18, v16 // (op 24910) 
v_mul_f32 v13, v18, v17 // (op 24910) 

v_add_f32 v14, v10, v6 // (op 24911) 
v_add_f32 v15, v11, v7 // (op 24911) 
v_add_f32 v16, v12, v8 // (op 24911) 
v_add_f32 v17, v13, v9 // (op 24911) 




s_lshl_b32 s0, s4, 7 // (op 24885) 
s_mul_i32 s1, 5, s0 // (op 24885) 
s_add_u32 s1, s1, 640 // (op 24885) 

v_add_u32 v0, v21, s1 // (op 24885) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24885) Store value




s_lshl_b32 s0, s2, 7 // (op 24916) 
s_mul_i32 s1, 6, s0 // (op 24916) 
s_add_u32 s1, s1, 640 // (op 24916) 

v_add_u32 v0, v20, s1 // (op 24916) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24916) Load value

v_accvgpr_read v10, a184 // (op 24932) 
v_accvgpr_read v11, a185 // (op 24932) 
v_accvgpr_read v12, a186 // (op 24932) 
v_accvgpr_read v13, a187 // (op 24932) 
v_mul_f32 v6, v19, v10 // (op 24932) 
v_mul_f32 v7, v19, v11 // (op 24932) 
v_mul_f32 v8, v19, v12 // (op 24932) 
v_mul_f32 v9, v19, v13 // (op 24932) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24914) 
v_mul_f32 v11, v18, v15 // (op 24914) 
v_mul_f32 v12, v18, v16 // (op 24914) 
v_mul_f32 v13, v18, v17 // (op 24914) 

v_add_f32 v14, v10, v6 // (op 24915) 
v_add_f32 v15, v11, v7 // (op 24915) 
v_add_f32 v16, v12, v8 // (op 24915) 
v_add_f32 v17, v13, v9 // (op 24915) 




s_lshl_b32 s0, s4, 7 // (op 24887) 
s_mul_i32 s1, 6, s0 // (op 24887) 
s_add_u32 s1, s1, 640 // (op 24887) 

v_add_u32 v0, v21, s1 // (op 24887) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24887) Store value




s_lshl_b32 s0, s2, 7 // (op 24920) 
s_mul_i32 s1, 7, s0 // (op 24920) 
s_add_u32 s1, s1, 640 // (op 24920) 

v_add_u32 v0, v20, s1 // (op 24920) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24920) Load value

v_accvgpr_read v10, a188 // (op 24934) 
v_accvgpr_read v11, a189 // (op 24934) 
v_accvgpr_read v12, a190 // (op 24934) 
v_accvgpr_read v13, a191 // (op 24934) 
v_mul_f32 v6, v19, v10 // (op 24934) 
v_mul_f32 v7, v19, v11 // (op 24934) 
v_mul_f32 v8, v19, v12 // (op 24934) 
v_mul_f32 v9, v19, v13 // (op 24934) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24918) 
v_mul_f32 v11, v18, v15 // (op 24918) 
v_mul_f32 v12, v18, v16 // (op 24918) 
v_mul_f32 v13, v18, v17 // (op 24918) 

v_add_f32 v14, v10, v6 // (op 24919) 
v_add_f32 v15, v11, v7 // (op 24919) 
v_add_f32 v16, v12, v8 // (op 24919) 
v_add_f32 v17, v13, v9 // (op 24919) 




s_lshl_b32 s0, s4, 7 // (op 24889) 
s_mul_i32 s1, 7, s0 // (op 24889) 
s_add_u32 s1, s1, 640 // (op 24889) 

v_add_u32 v0, v21, s1 // (op 24889) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24889) Store value




v_add_u32 v0, 512, v20 // (op 24737) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24737) Load value

v_accvgpr_read v10, a128 // (op 24717) 
v_accvgpr_read v11, a129 // (op 24717) 
v_accvgpr_read v12, a130 // (op 24717) 
v_accvgpr_read v13, a131 // (op 24717) 
v_mul_f32 v6, v19, v10 // (op 24717) 
v_mul_f32 v7, v19, v11 // (op 24717) 
v_mul_f32 v8, v19, v12 // (op 24717) 
v_mul_f32 v9, v19, v13 // (op 24717) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24736) 
v_mul_f32 v11, v18, v15 // (op 24736) 
v_mul_f32 v12, v18, v16 // (op 24736) 
v_mul_f32 v13, v18, v17 // (op 24736) 

v_add_f32 v14, v10, v6 // (op 24718) 
v_add_f32 v15, v11, v7 // (op 24718) 
v_add_f32 v16, v12, v8 // (op 24718) 
v_add_f32 v17, v13, v9 // (op 24718) 




v_add_u32 v0, 512, v21 // (op 24720) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24720) Store value




s_lshl_b32 s0, s2, 7 // (op 24741) 
s_add_u32 s0, s0, 512 // (op 24741) 
v_add_u32 v0, v20, s0 // (op 24741) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24741) Load value

v_accvgpr_read v10, a132 // (op 24767) 
v_accvgpr_read v11, a133 // (op 24767) 
v_accvgpr_read v12, a134 // (op 24767) 
v_accvgpr_read v13, a135 // (op 24767) 
v_mul_f32 v6, v19, v10 // (op 24767) 
v_mul_f32 v7, v19, v11 // (op 24767) 
v_mul_f32 v8, v19, v12 // (op 24767) 
v_mul_f32 v9, v19, v13 // (op 24767) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24739) 
v_mul_f32 v11, v18, v15 // (op 24739) 
v_mul_f32 v12, v18, v16 // (op 24739) 
v_mul_f32 v13, v18, v17 // (op 24739) 

v_add_f32 v14, v10, v6 // (op 24740) 
v_add_f32 v15, v11, v7 // (op 24740) 
v_add_f32 v16, v12, v8 // (op 24740) 
v_add_f32 v17, v13, v9 // (op 24740) 




s_lshl_b32 s0, s4, 7 // (op 24722) 
s_add_u32 s0, s0, 512 // (op 24722) 
v_add_u32 v0, v21, s0 // (op 24722) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24722) Store value




s_lshl_b32 s0, s2, 8 // (op 24745) 
s_add_u32 s0, s0, 512 // (op 24745) 
v_add_u32 v0, v20, s0 // (op 24745) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24745) Load value

v_accvgpr_read v10, a136 // (op 24769) 
v_accvgpr_read v11, a137 // (op 24769) 
v_accvgpr_read v12, a138 // (op 24769) 
v_accvgpr_read v13, a139 // (op 24769) 
v_mul_f32 v6, v19, v10 // (op 24769) 
v_mul_f32 v7, v19, v11 // (op 24769) 
v_mul_f32 v8, v19, v12 // (op 24769) 
v_mul_f32 v9, v19, v13 // (op 24769) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24743) 
v_mul_f32 v11, v18, v15 // (op 24743) 
v_mul_f32 v12, v18, v16 // (op 24743) 
v_mul_f32 v13, v18, v17 // (op 24743) 

v_add_f32 v14, v10, v6 // (op 24744) 
v_add_f32 v15, v11, v7 // (op 24744) 
v_add_f32 v16, v12, v8 // (op 24744) 
v_add_f32 v17, v13, v9 // (op 24744) 




s_lshl_b32 s0, s4, 8 // (op 24724) 
s_add_u32 s0, s0, 512 // (op 24724) 
v_add_u32 v0, v21, s0 // (op 24724) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24724) Store value




s_lshl_b32 s0, s2, 7 // (op 24749) 
s_mul_i32 s1, 3, s0 // (op 24749) 
s_add_u32 s1, s1, 512 // (op 24749) 

v_add_u32 v0, v20, s1 // (op 24749) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24749) Load value

v_accvgpr_read v10, a140 // (op 24771) 
v_accvgpr_read v11, a141 // (op 24771) 
v_accvgpr_read v12, a142 // (op 24771) 
v_accvgpr_read v13, a143 // (op 24771) 
v_mul_f32 v6, v19, v10 // (op 24771) 
v_mul_f32 v7, v19, v11 // (op 24771) 
v_mul_f32 v8, v19, v12 // (op 24771) 
v_mul_f32 v9, v19, v13 // (op 24771) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24747) 
v_mul_f32 v11, v18, v15 // (op 24747) 
v_mul_f32 v12, v18, v16 // (op 24747) 
v_mul_f32 v13, v18, v17 // (op 24747) 

v_add_f32 v14, v10, v6 // (op 24748) 
v_add_f32 v15, v11, v7 // (op 24748) 
v_add_f32 v16, v12, v8 // (op 24748) 
v_add_f32 v17, v13, v9 // (op 24748) 




s_lshl_b32 s0, s4, 7 // (op 24726) 
s_mul_i32 s1, 3, s0 // (op 24726) 
s_add_u32 s1, s1, 512 // (op 24726) 

v_add_u32 v0, v21, s1 // (op 24726) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24726) Store value




s_lshl_b32 s0, s2, 9 // (op 24753) 
s_add_u32 s0, s0, 512 // (op 24753) 
v_add_u32 v0, v20, s0 // (op 24753) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24753) Load value

v_accvgpr_read v10, a144 // (op 24773) 
v_accvgpr_read v11, a145 // (op 24773) 
v_accvgpr_read v12, a146 // (op 24773) 
v_accvgpr_read v13, a147 // (op 24773) 
v_mul_f32 v6, v19, v10 // (op 24773) 
v_mul_f32 v7, v19, v11 // (op 24773) 
v_mul_f32 v8, v19, v12 // (op 24773) 
v_mul_f32 v9, v19, v13 // (op 24773) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24751) 
v_mul_f32 v11, v18, v15 // (op 24751) 
v_mul_f32 v12, v18, v16 // (op 24751) 
v_mul_f32 v13, v18, v17 // (op 24751) 

v_add_f32 v14, v10, v6 // (op 24752) 
v_add_f32 v15, v11, v7 // (op 24752) 
v_add_f32 v16, v12, v8 // (op 24752) 
v_add_f32 v17, v13, v9 // (op 24752) 




s_lshl_b32 s0, s4, 9 // (op 24728) 
s_add_u32 s0, s0, 512 // (op 24728) 
v_add_u32 v0, v21, s0 // (op 24728) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24728) Store value




s_lshl_b32 s0, s2, 7 // (op 24757) 
s_mul_i32 s1, 5, s0 // (op 24757) 
s_add_u32 s1, s1, 512 // (op 24757) 

v_add_u32 v0, v20, s1 // (op 24757) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24757) Load value

v_accvgpr_read v10, a148 // (op 24775) 
v_accvgpr_read v11, a149 // (op 24775) 
v_accvgpr_read v12, a150 // (op 24775) 
v_accvgpr_read v13, a151 // (op 24775) 
v_mul_f32 v6, v19, v10 // (op 24775) 
v_mul_f32 v7, v19, v11 // (op 24775) 
v_mul_f32 v8, v19, v12 // (op 24775) 
v_mul_f32 v9, v19, v13 // (op 24775) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24755) 
v_mul_f32 v11, v18, v15 // (op 24755) 
v_mul_f32 v12, v18, v16 // (op 24755) 
v_mul_f32 v13, v18, v17 // (op 24755) 

v_add_f32 v14, v10, v6 // (op 24756) 
v_add_f32 v15, v11, v7 // (op 24756) 
v_add_f32 v16, v12, v8 // (op 24756) 
v_add_f32 v17, v13, v9 // (op 24756) 




s_lshl_b32 s0, s4, 7 // (op 24730) 
s_mul_i32 s1, 5, s0 // (op 24730) 
s_add_u32 s1, s1, 512 // (op 24730) 

v_add_u32 v0, v21, s1 // (op 24730) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24730) Store value




s_lshl_b32 s0, s2, 7 // (op 24761) 
s_mul_i32 s1, 6, s0 // (op 24761) 
s_add_u32 s1, s1, 512 // (op 24761) 

v_add_u32 v0, v20, s1 // (op 24761) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24761) Load value

v_accvgpr_read v10, a152 // (op 24777) 
v_accvgpr_read v11, a153 // (op 24777) 
v_accvgpr_read v12, a154 // (op 24777) 
v_accvgpr_read v13, a155 // (op 24777) 
v_mul_f32 v6, v19, v10 // (op 24777) 
v_mul_f32 v7, v19, v11 // (op 24777) 
v_mul_f32 v8, v19, v12 // (op 24777) 
v_mul_f32 v9, v19, v13 // (op 24777) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24759) 
v_mul_f32 v11, v18, v15 // (op 24759) 
v_mul_f32 v12, v18, v16 // (op 24759) 
v_mul_f32 v13, v18, v17 // (op 24759) 

v_add_f32 v14, v10, v6 // (op 24760) 
v_add_f32 v15, v11, v7 // (op 24760) 
v_add_f32 v16, v12, v8 // (op 24760) 
v_add_f32 v17, v13, v9 // (op 24760) 




s_lshl_b32 s0, s4, 7 // (op 24732) 
s_mul_i32 s1, 6, s0 // (op 24732) 
s_add_u32 s1, s1, 512 // (op 24732) 

v_add_u32 v0, v21, s1 // (op 24732) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24732) Store value




s_lshl_b32 s0, s2, 7 // (op 24765) 
s_mul_i32 s1, 7, s0 // (op 24765) 
s_add_u32 s1, s1, 512 // (op 24765) 

v_add_u32 v0, v20, s1 // (op 24765) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24765) Load value

v_accvgpr_read v10, a156 // (op 24779) 
v_accvgpr_read v11, a157 // (op 24779) 
v_accvgpr_read v12, a158 // (op 24779) 
v_accvgpr_read v13, a159 // (op 24779) 
v_mul_f32 v6, v19, v10 // (op 24779) 
v_mul_f32 v7, v19, v11 // (op 24779) 
v_mul_f32 v8, v19, v12 // (op 24779) 
v_mul_f32 v9, v19, v13 // (op 24779) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24763) 
v_mul_f32 v11, v18, v15 // (op 24763) 
v_mul_f32 v12, v18, v16 // (op 24763) 
v_mul_f32 v13, v18, v17 // (op 24763) 

v_add_f32 v14, v10, v6 // (op 24764) 
v_add_f32 v15, v11, v7 // (op 24764) 
v_add_f32 v16, v12, v8 // (op 24764) 
v_add_f32 v17, v13, v9 // (op 24764) 




s_lshl_b32 s0, s4, 7 // (op 24734) 
s_mul_i32 s1, 7, s0 // (op 24734) 
s_add_u32 s1, s1, 512 // (op 24734) 

v_add_u32 v0, v21, s1 // (op 24734) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24734) Store value




v_add_u32 v0, 384, v20 // (op 24582) 

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24582) Load value

v_accvgpr_read v10, a96 // (op 24562) 
v_accvgpr_read v11, a97 // (op 24562) 
v_accvgpr_read v12, a98 // (op 24562) 
v_accvgpr_read v13, a99 // (op 24562) 
v_mul_f32 v6, v19, v10 // (op 24562) 
v_mul_f32 v7, v19, v11 // (op 24562) 
v_mul_f32 v8, v19, v12 // (op 24562) 
v_mul_f32 v9, v19, v13 // (op 24562) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24581) 
v_mul_f32 v11, v18, v15 // (op 24581) 
v_mul_f32 v12, v18, v16 // (op 24581) 
v_mul_f32 v13, v18, v17 // (op 24581) 

v_add_f32 v14, v10, v6 // (op 24563) 
v_add_f32 v15, v11, v7 // (op 24563) 
v_add_f32 v16, v12, v8 // (op 24563) 
v_add_f32 v17, v13, v9 // (op 24563) 




v_add_u32 v0, 384, v21 // (op 24565) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24565) Store value




s_lshl_b32 s0, s2, 7 // (op 24586) 
s_add_u32 s0, s0, 384 // (op 24586) 
v_add_u32 v0, v20, s0 // (op 24586) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24586) Load value

v_accvgpr_read v10, a100 // (op 24612) 
v_accvgpr_read v11, a101 // (op 24612) 
v_accvgpr_read v12, a102 // (op 24612) 
v_accvgpr_read v13, a103 // (op 24612) 
v_mul_f32 v6, v19, v10 // (op 24612) 
v_mul_f32 v7, v19, v11 // (op 24612) 
v_mul_f32 v8, v19, v12 // (op 24612) 
v_mul_f32 v9, v19, v13 // (op 24612) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24584) 
v_mul_f32 v11, v18, v15 // (op 24584) 
v_mul_f32 v12, v18, v16 // (op 24584) 
v_mul_f32 v13, v18, v17 // (op 24584) 

v_add_f32 v14, v10, v6 // (op 24585) 
v_add_f32 v15, v11, v7 // (op 24585) 
v_add_f32 v16, v12, v8 // (op 24585) 
v_add_f32 v17, v13, v9 // (op 24585) 




s_lshl_b32 s0, s4, 7 // (op 24567) 
s_add_u32 s0, s0, 384 // (op 24567) 
v_add_u32 v0, v21, s0 // (op 24567) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24567) Store value




s_lshl_b32 s0, s2, 8 // (op 24590) 
s_add_u32 s0, s0, 384 // (op 24590) 
v_add_u32 v0, v20, s0 // (op 24590) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24590) Load value

v_accvgpr_read v10, a104 // (op 24614) 
v_accvgpr_read v11, a105 // (op 24614) 
v_accvgpr_read v12, a106 // (op 24614) 
v_accvgpr_read v13, a107 // (op 24614) 
v_mul_f32 v6, v19, v10 // (op 24614) 
v_mul_f32 v7, v19, v11 // (op 24614) 
v_mul_f32 v8, v19, v12 // (op 24614) 
v_mul_f32 v9, v19, v13 // (op 24614) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 24588) 
v_mul_f32 v11, v18, v15 // (op 24588) 
v_mul_f32 v12, v18, v16 // (op 24588) 
v_mul_f32 v13, v18, v17 // (op 24588) 

v_add_f32 v14, v10, v6 // (op 24589) 
v_add_f32 v15, v11, v7 // (op 24589) 
v_add_f32 v16, v12, v8 // (op 24589) 
v_add_f32 v17, v13, v9 // (op 24589) 




s_lshl_b32 s0, s4, 8 // (op 24569) 
s_add_u32 s0, s0, 384 // (op 24569) 
v_add_u32 v0, v21, s0 // (op 24569) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24569) Store value




s_lshl_b32 s0, s2, 7 // (op 24594) 
s_mul_i32 s1, 3, s0 // (op 24594) 
s_add_u32 s1, s1, 384 // (op 24594) 

v_add_u32 v0, v20, s1 // (op 24594) 


v_accvgpr_read v14, a108 // (op 24616) 
v_accvgpr_read v15, a109 // (op 24616) 
v_accvgpr_read v16, a110 // (op 24616) 
v_accvgpr_read v17, a111 // (op 24616) 
v_mul_f32 v10, v19, v14 // (op 24616) 
v_mul_f32 v11, v19, v15 // (op 24616) 
v_mul_f32 v12, v19, v16 // (op 24616) 
v_mul_f32 v13, v19, v17 // (op 24616) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24594) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24592) 
v_mul_f32 v7, v18, v15 // (op 24592) 
v_mul_f32 v8, v18, v16 // (op 24592) 
v_mul_f32 v9, v18, v17 // (op 24592) 

v_add_f32 v14, v6, v10 // (op 24593) 
v_add_f32 v15, v7, v11 // (op 24593) 
v_add_f32 v16, v8, v12 // (op 24593) 
v_add_f32 v17, v9, v13 // (op 24593) 




s_lshl_b32 s0, s4, 7 // (op 24571) 
s_mul_i32 s1, 3, s0 // (op 24571) 
s_add_u32 s1, s1, 384 // (op 24571) 

v_add_u32 v0, v21, s1 // (op 24571) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24571) Store value




s_lshl_b32 s0, s2, 9 // (op 24598) 
s_add_u32 s0, s0, 384 // (op 24598) 
v_add_u32 v0, v20, s0 // (op 24598) 


v_accvgpr_read v14, a112 // (op 24618) 
v_accvgpr_read v15, a113 // (op 24618) 
v_accvgpr_read v16, a114 // (op 24618) 
v_accvgpr_read v17, a115 // (op 24618) 
v_mul_f32 v10, v19, v14 // (op 24618) 
v_mul_f32 v11, v19, v15 // (op 24618) 
v_mul_f32 v12, v19, v16 // (op 24618) 
v_mul_f32 v13, v19, v17 // (op 24618) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24598) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24596) 
v_mul_f32 v7, v18, v15 // (op 24596) 
v_mul_f32 v8, v18, v16 // (op 24596) 
v_mul_f32 v9, v18, v17 // (op 24596) 

v_add_f32 v14, v6, v10 // (op 24597) 
v_add_f32 v15, v7, v11 // (op 24597) 
v_add_f32 v16, v8, v12 // (op 24597) 
v_add_f32 v17, v9, v13 // (op 24597) 




s_lshl_b32 s0, s4, 9 // (op 24573) 
s_add_u32 s0, s0, 384 // (op 24573) 
v_add_u32 v0, v21, s0 // (op 24573) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24573) Store value




s_lshl_b32 s0, s2, 7 // (op 24602) 
s_mul_i32 s1, 5, s0 // (op 24602) 
s_add_u32 s1, s1, 384 // (op 24602) 

v_add_u32 v0, v20, s1 // (op 24602) 


v_accvgpr_read v14, a116 // (op 24620) 
v_accvgpr_read v15, a117 // (op 24620) 
v_accvgpr_read v16, a118 // (op 24620) 
v_accvgpr_read v17, a119 // (op 24620) 
v_mul_f32 v10, v19, v14 // (op 24620) 
v_mul_f32 v11, v19, v15 // (op 24620) 
v_mul_f32 v12, v19, v16 // (op 24620) 
v_mul_f32 v13, v19, v17 // (op 24620) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24602) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24600) 
v_mul_f32 v7, v18, v15 // (op 24600) 
v_mul_f32 v8, v18, v16 // (op 24600) 
v_mul_f32 v9, v18, v17 // (op 24600) 

v_add_f32 v14, v6, v10 // (op 24601) 
v_add_f32 v15, v7, v11 // (op 24601) 
v_add_f32 v16, v8, v12 // (op 24601) 
v_add_f32 v17, v9, v13 // (op 24601) 




s_lshl_b32 s0, s4, 7 // (op 24575) 
s_mul_i32 s1, 5, s0 // (op 24575) 
s_add_u32 s1, s1, 384 // (op 24575) 

v_add_u32 v0, v21, s1 // (op 24575) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24575) Store value




s_lshl_b32 s0, s2, 7 // (op 24606) 
s_mul_i32 s1, 6, s0 // (op 24606) 
s_add_u32 s1, s1, 384 // (op 24606) 

v_add_u32 v0, v20, s1 // (op 24606) 


v_accvgpr_read v14, a120 // (op 24622) 
v_accvgpr_read v15, a121 // (op 24622) 
v_accvgpr_read v16, a122 // (op 24622) 
v_accvgpr_read v17, a123 // (op 24622) 
v_mul_f32 v10, v19, v14 // (op 24622) 
v_mul_f32 v11, v19, v15 // (op 24622) 
v_mul_f32 v12, v19, v16 // (op 24622) 
v_mul_f32 v13, v19, v17 // (op 24622) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24606) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24604) 
v_mul_f32 v7, v18, v15 // (op 24604) 
v_mul_f32 v8, v18, v16 // (op 24604) 
v_mul_f32 v9, v18, v17 // (op 24604) 

v_add_f32 v14, v6, v10 // (op 24605) 
v_add_f32 v15, v7, v11 // (op 24605) 
v_add_f32 v16, v8, v12 // (op 24605) 
v_add_f32 v17, v9, v13 // (op 24605) 




s_lshl_b32 s0, s4, 7 // (op 24577) 
s_mul_i32 s1, 6, s0 // (op 24577) 
s_add_u32 s1, s1, 384 // (op 24577) 

v_add_u32 v0, v21, s1 // (op 24577) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24577) Store value




s_lshl_b32 s0, s2, 7 // (op 24610) 
s_mul_i32 s1, 7, s0 // (op 24610) 
s_add_u32 s1, s1, 384 // (op 24610) 

v_add_u32 v0, v20, s1 // (op 24610) 


v_accvgpr_read v14, a124 // (op 24624) 
v_accvgpr_read v15, a125 // (op 24624) 
v_accvgpr_read v16, a126 // (op 24624) 
v_accvgpr_read v17, a127 // (op 24624) 
v_mul_f32 v10, v19, v14 // (op 24624) 
v_mul_f32 v11, v19, v15 // (op 24624) 
v_mul_f32 v12, v19, v16 // (op 24624) 
v_mul_f32 v13, v19, v17 // (op 24624) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24610) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24608) 
v_mul_f32 v7, v18, v15 // (op 24608) 
v_mul_f32 v8, v18, v16 // (op 24608) 
v_mul_f32 v9, v18, v17 // (op 24608) 

v_add_f32 v14, v6, v10 // (op 24609) 
v_add_f32 v15, v7, v11 // (op 24609) 
v_add_f32 v16, v8, v12 // (op 24609) 
v_add_f32 v17, v9, v13 // (op 24609) 




s_lshl_b32 s0, s4, 7 // (op 24579) 
s_mul_i32 s1, 7, s0 // (op 24579) 
s_add_u32 s1, s1, 384 // (op 24579) 

v_add_u32 v0, v21, s1 // (op 24579) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24579) Store value




v_add_u32 v0, 256, v20 // (op 24427) 

s_nop 0
v_accvgpr_read v14, a64 // (op 24407) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v15, a65 // (op 24407) 
v_accvgpr_read v16, a66 // (op 24407) 
v_accvgpr_read v17, a67 // (op 24407) 
v_mul_f32 v10, v19, v14 // (op 24407) 
v_mul_f32 v11, v19, v15 // (op 24407) 
v_mul_f32 v12, v19, v16 // (op 24407) 
v_mul_f32 v13, v19, v17 // (op 24407) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24427) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24426) 
v_mul_f32 v7, v18, v15 // (op 24426) 
v_mul_f32 v8, v18, v16 // (op 24426) 
v_mul_f32 v9, v18, v17 // (op 24426) 

v_add_f32 v14, v6, v10 // (op 24408) 
v_add_f32 v15, v7, v11 // (op 24408) 
v_add_f32 v16, v8, v12 // (op 24408) 
v_add_f32 v17, v9, v13 // (op 24408) 




v_add_u32 v0, 256, v21 // (op 24410) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24410) Store value




s_lshl_b32 s0, s2, 7 // (op 24431) 
s_add_u32 s0, s0, 256 // (op 24431) 
v_add_u32 v0, v20, s0 // (op 24431) 


v_accvgpr_read v14, a68 // (op 24457) 
v_accvgpr_read v15, a69 // (op 24457) 
v_accvgpr_read v16, a70 // (op 24457) 
v_accvgpr_read v17, a71 // (op 24457) 
v_mul_f32 v10, v19, v14 // (op 24457) 
v_mul_f32 v11, v19, v15 // (op 24457) 
v_mul_f32 v12, v19, v16 // (op 24457) 
v_mul_f32 v13, v19, v17 // (op 24457) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24431) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24429) 
v_mul_f32 v7, v18, v15 // (op 24429) 
v_mul_f32 v8, v18, v16 // (op 24429) 
v_mul_f32 v9, v18, v17 // (op 24429) 

v_add_f32 v14, v6, v10 // (op 24430) 
v_add_f32 v15, v7, v11 // (op 24430) 
v_add_f32 v16, v8, v12 // (op 24430) 
v_add_f32 v17, v9, v13 // (op 24430) 




s_lshl_b32 s0, s4, 7 // (op 24412) 
s_add_u32 s0, s0, 256 // (op 24412) 
v_add_u32 v0, v21, s0 // (op 24412) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24412) Store value




s_lshl_b32 s0, s2, 8 // (op 24435) 
s_add_u32 s0, s0, 256 // (op 24435) 
v_add_u32 v0, v20, s0 // (op 24435) 


v_accvgpr_read v14, a72 // (op 24459) 
v_accvgpr_read v15, a73 // (op 24459) 
v_accvgpr_read v16, a74 // (op 24459) 
v_accvgpr_read v17, a75 // (op 24459) 
v_mul_f32 v10, v19, v14 // (op 24459) 
v_mul_f32 v11, v19, v15 // (op 24459) 
v_mul_f32 v12, v19, v16 // (op 24459) 
v_mul_f32 v13, v19, v17 // (op 24459) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24435) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24433) 
v_mul_f32 v7, v18, v15 // (op 24433) 
v_mul_f32 v8, v18, v16 // (op 24433) 
v_mul_f32 v9, v18, v17 // (op 24433) 

v_add_f32 v14, v6, v10 // (op 24434) 
v_add_f32 v15, v7, v11 // (op 24434) 
v_add_f32 v16, v8, v12 // (op 24434) 
v_add_f32 v17, v9, v13 // (op 24434) 




s_lshl_b32 s0, s4, 8 // (op 24414) 
s_add_u32 s0, s0, 256 // (op 24414) 
v_add_u32 v0, v21, s0 // (op 24414) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24414) Store value




s_lshl_b32 s0, s2, 7 // (op 24439) 
s_mul_i32 s1, 3, s0 // (op 24439) 
s_add_u32 s1, s1, 256 // (op 24439) 

v_add_u32 v0, v20, s1 // (op 24439) 


v_accvgpr_read v14, a76 // (op 24461) 
v_accvgpr_read v15, a77 // (op 24461) 
v_accvgpr_read v16, a78 // (op 24461) 
v_accvgpr_read v17, a79 // (op 24461) 
v_mul_f32 v10, v19, v14 // (op 24461) 
v_mul_f32 v11, v19, v15 // (op 24461) 
v_mul_f32 v12, v19, v16 // (op 24461) 
v_mul_f32 v13, v19, v17 // (op 24461) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24439) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24437) 
v_mul_f32 v7, v18, v15 // (op 24437) 
v_mul_f32 v8, v18, v16 // (op 24437) 
v_mul_f32 v9, v18, v17 // (op 24437) 

v_add_f32 v14, v6, v10 // (op 24438) 
v_add_f32 v15, v7, v11 // (op 24438) 
v_add_f32 v16, v8, v12 // (op 24438) 
v_add_f32 v17, v9, v13 // (op 24438) 




s_lshl_b32 s0, s4, 7 // (op 24416) 
s_mul_i32 s1, 3, s0 // (op 24416) 
s_add_u32 s1, s1, 256 // (op 24416) 

v_add_u32 v0, v21, s1 // (op 24416) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24416) Store value




s_lshl_b32 s0, s2, 9 // (op 24443) 
s_add_u32 s0, s0, 256 // (op 24443) 
v_add_u32 v0, v20, s0 // (op 24443) 


v_accvgpr_read v14, a80 // (op 24463) 
v_accvgpr_read v15, a81 // (op 24463) 
v_accvgpr_read v16, a82 // (op 24463) 
v_accvgpr_read v17, a83 // (op 24463) 
v_mul_f32 v10, v19, v14 // (op 24463) 
v_mul_f32 v11, v19, v15 // (op 24463) 
v_mul_f32 v12, v19, v16 // (op 24463) 
v_mul_f32 v13, v19, v17 // (op 24463) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24443) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24441) 
v_mul_f32 v7, v18, v15 // (op 24441) 
v_mul_f32 v8, v18, v16 // (op 24441) 
v_mul_f32 v9, v18, v17 // (op 24441) 

v_add_f32 v14, v6, v10 // (op 24442) 
v_add_f32 v15, v7, v11 // (op 24442) 
v_add_f32 v16, v8, v12 // (op 24442) 
v_add_f32 v17, v9, v13 // (op 24442) 




s_lshl_b32 s0, s4, 9 // (op 24418) 
s_add_u32 s0, s0, 256 // (op 24418) 
v_add_u32 v0, v21, s0 // (op 24418) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24418) Store value




s_lshl_b32 s0, s2, 7 // (op 24447) 
s_mul_i32 s1, 5, s0 // (op 24447) 
s_add_u32 s1, s1, 256 // (op 24447) 

v_add_u32 v0, v20, s1 // (op 24447) 


v_accvgpr_read v14, a84 // (op 24465) 
v_accvgpr_read v15, a85 // (op 24465) 
v_accvgpr_read v16, a86 // (op 24465) 
v_accvgpr_read v17, a87 // (op 24465) 
v_mul_f32 v10, v19, v14 // (op 24465) 
v_mul_f32 v11, v19, v15 // (op 24465) 
v_mul_f32 v12, v19, v16 // (op 24465) 
v_mul_f32 v13, v19, v17 // (op 24465) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24447) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24445) 
v_mul_f32 v7, v18, v15 // (op 24445) 
v_mul_f32 v8, v18, v16 // (op 24445) 
v_mul_f32 v9, v18, v17 // (op 24445) 

v_add_f32 v14, v6, v10 // (op 24446) 
v_add_f32 v15, v7, v11 // (op 24446) 
v_add_f32 v16, v8, v12 // (op 24446) 
v_add_f32 v17, v9, v13 // (op 24446) 




s_lshl_b32 s0, s4, 7 // (op 24420) 
s_mul_i32 s1, 5, s0 // (op 24420) 
s_add_u32 s1, s1, 256 // (op 24420) 

v_add_u32 v0, v21, s1 // (op 24420) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24420) Store value




s_lshl_b32 s0, s2, 7 // (op 24451) 
s_mul_i32 s1, 6, s0 // (op 24451) 
s_add_u32 s1, s1, 256 // (op 24451) 

v_add_u32 v0, v20, s1 // (op 24451) 


v_accvgpr_read v14, a88 // (op 24467) 
v_accvgpr_read v15, a89 // (op 24467) 
v_accvgpr_read v16, a90 // (op 24467) 
v_accvgpr_read v17, a91 // (op 24467) 
v_mul_f32 v10, v19, v14 // (op 24467) 
v_mul_f32 v11, v19, v15 // (op 24467) 
v_mul_f32 v12, v19, v16 // (op 24467) 
v_mul_f32 v13, v19, v17 // (op 24467) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24451) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24449) 
v_mul_f32 v7, v18, v15 // (op 24449) 
v_mul_f32 v8, v18, v16 // (op 24449) 
v_mul_f32 v9, v18, v17 // (op 24449) 

v_add_f32 v14, v6, v10 // (op 24450) 
v_add_f32 v15, v7, v11 // (op 24450) 
v_add_f32 v16, v8, v12 // (op 24450) 
v_add_f32 v17, v9, v13 // (op 24450) 




s_lshl_b32 s0, s4, 7 // (op 24422) 
s_mul_i32 s1, 6, s0 // (op 24422) 
s_add_u32 s1, s1, 256 // (op 24422) 

v_add_u32 v0, v21, s1 // (op 24422) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24422) Store value




s_lshl_b32 s0, s2, 7 // (op 24455) 
s_mul_i32 s1, 7, s0 // (op 24455) 
s_add_u32 s1, s1, 256 // (op 24455) 

v_add_u32 v0, v20, s1 // (op 24455) 


v_accvgpr_read v14, a92 // (op 24469) 
v_accvgpr_read v15, a93 // (op 24469) 
v_accvgpr_read v16, a94 // (op 24469) 
v_accvgpr_read v17, a95 // (op 24469) 
v_mul_f32 v10, v19, v14 // (op 24469) 
v_mul_f32 v11, v19, v15 // (op 24469) 
v_mul_f32 v12, v19, v16 // (op 24469) 
v_mul_f32 v13, v19, v17 // (op 24469) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24455) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24453) 
v_mul_f32 v7, v18, v15 // (op 24453) 
v_mul_f32 v8, v18, v16 // (op 24453) 
v_mul_f32 v9, v18, v17 // (op 24453) 

v_add_f32 v14, v6, v10 // (op 24454) 
v_add_f32 v15, v7, v11 // (op 24454) 
v_add_f32 v16, v8, v12 // (op 24454) 
v_add_f32 v17, v9, v13 // (op 24454) 




s_lshl_b32 s0, s4, 7 // (op 24424) 
s_mul_i32 s1, 7, s0 // (op 24424) 
s_add_u32 s1, s1, 256 // (op 24424) 

v_add_u32 v0, v21, s1 // (op 24424) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24424) Store value




v_add_u32 v0, 128, v20 // (op 24272) 

s_nop 0
v_accvgpr_read v14, a32 // (op 24252) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v15, a33 // (op 24252) 
v_accvgpr_read v16, a34 // (op 24252) 
v_accvgpr_read v17, a35 // (op 24252) 
v_mul_f32 v10, v19, v14 // (op 24252) 
v_mul_f32 v11, v19, v15 // (op 24252) 
v_mul_f32 v12, v19, v16 // (op 24252) 
v_mul_f32 v13, v19, v17 // (op 24252) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24272) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24271) 
v_mul_f32 v7, v18, v15 // (op 24271) 
v_mul_f32 v8, v18, v16 // (op 24271) 
v_mul_f32 v9, v18, v17 // (op 24271) 

v_add_f32 v14, v6, v10 // (op 24253) 
v_add_f32 v15, v7, v11 // (op 24253) 
v_add_f32 v16, v8, v12 // (op 24253) 
v_add_f32 v17, v9, v13 // (op 24253) 




v_add_u32 v0, 128, v21 // (op 24255) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24255) Store value




s_lshl_b32 s0, s2, 7 // (op 24276) 
s_add_u32 s0, s0, 128 // (op 24276) 
v_add_u32 v0, v20, s0 // (op 24276) 


v_accvgpr_read v14, a36 // (op 24302) 
v_accvgpr_read v15, a37 // (op 24302) 
v_accvgpr_read v16, a38 // (op 24302) 
v_accvgpr_read v17, a39 // (op 24302) 
v_mul_f32 v10, v19, v14 // (op 24302) 
v_mul_f32 v11, v19, v15 // (op 24302) 
v_mul_f32 v12, v19, v16 // (op 24302) 
v_mul_f32 v13, v19, v17 // (op 24302) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24276) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24274) 
v_mul_f32 v7, v18, v15 // (op 24274) 
v_mul_f32 v8, v18, v16 // (op 24274) 
v_mul_f32 v9, v18, v17 // (op 24274) 

v_add_f32 v14, v6, v10 // (op 24275) 
v_add_f32 v15, v7, v11 // (op 24275) 
v_add_f32 v16, v8, v12 // (op 24275) 
v_add_f32 v17, v9, v13 // (op 24275) 




s_lshl_b32 s0, s4, 7 // (op 24257) 
s_add_u32 s0, s0, 128 // (op 24257) 
v_add_u32 v0, v21, s0 // (op 24257) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24257) Store value




s_lshl_b32 s0, s2, 8 // (op 24280) 
s_add_u32 s0, s0, 128 // (op 24280) 
v_add_u32 v0, v20, s0 // (op 24280) 


v_accvgpr_read v14, a40 // (op 24304) 
v_accvgpr_read v15, a41 // (op 24304) 
v_accvgpr_read v16, a42 // (op 24304) 
v_accvgpr_read v17, a43 // (op 24304) 
v_mul_f32 v10, v19, v14 // (op 24304) 
v_mul_f32 v11, v19, v15 // (op 24304) 
v_mul_f32 v12, v19, v16 // (op 24304) 
v_mul_f32 v13, v19, v17 // (op 24304) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24280) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24278) 
v_mul_f32 v7, v18, v15 // (op 24278) 
v_mul_f32 v8, v18, v16 // (op 24278) 
v_mul_f32 v9, v18, v17 // (op 24278) 

v_add_f32 v14, v6, v10 // (op 24279) 
v_add_f32 v15, v7, v11 // (op 24279) 
v_add_f32 v16, v8, v12 // (op 24279) 
v_add_f32 v17, v9, v13 // (op 24279) 




s_lshl_b32 s0, s4, 8 // (op 24259) 
s_add_u32 s0, s0, 128 // (op 24259) 
v_add_u32 v0, v21, s0 // (op 24259) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24259) Store value




s_lshl_b32 s0, s2, 7 // (op 24284) 
s_mul_i32 s1, 3, s0 // (op 24284) 
s_add_u32 s1, s1, 128 // (op 24284) 

v_add_u32 v0, v20, s1 // (op 24284) 


v_accvgpr_read v14, a44 // (op 24306) 
v_accvgpr_read v15, a45 // (op 24306) 
v_accvgpr_read v16, a46 // (op 24306) 
v_accvgpr_read v17, a47 // (op 24306) 
v_mul_f32 v10, v19, v14 // (op 24306) 
v_mul_f32 v11, v19, v15 // (op 24306) 
v_mul_f32 v12, v19, v16 // (op 24306) 
v_mul_f32 v13, v19, v17 // (op 24306) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24284) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24282) 
v_mul_f32 v7, v18, v15 // (op 24282) 
v_mul_f32 v8, v18, v16 // (op 24282) 
v_mul_f32 v9, v18, v17 // (op 24282) 

v_add_f32 v14, v6, v10 // (op 24283) 
v_add_f32 v15, v7, v11 // (op 24283) 
v_add_f32 v16, v8, v12 // (op 24283) 
v_add_f32 v17, v9, v13 // (op 24283) 




s_lshl_b32 s0, s4, 7 // (op 24261) 
s_mul_i32 s1, 3, s0 // (op 24261) 
s_add_u32 s1, s1, 128 // (op 24261) 

v_add_u32 v0, v21, s1 // (op 24261) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24261) Store value




s_lshl_b32 s0, s2, 9 // (op 24288) 
s_add_u32 s0, s0, 128 // (op 24288) 
v_add_u32 v0, v20, s0 // (op 24288) 


v_accvgpr_read v14, a48 // (op 24308) 
v_accvgpr_read v15, a49 // (op 24308) 
v_accvgpr_read v16, a50 // (op 24308) 
v_accvgpr_read v17, a51 // (op 24308) 
v_mul_f32 v10, v19, v14 // (op 24308) 
v_mul_f32 v11, v19, v15 // (op 24308) 
v_mul_f32 v12, v19, v16 // (op 24308) 
v_mul_f32 v13, v19, v17 // (op 24308) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24288) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24286) 
v_mul_f32 v7, v18, v15 // (op 24286) 
v_mul_f32 v8, v18, v16 // (op 24286) 
v_mul_f32 v9, v18, v17 // (op 24286) 

v_add_f32 v14, v6, v10 // (op 24287) 
v_add_f32 v15, v7, v11 // (op 24287) 
v_add_f32 v16, v8, v12 // (op 24287) 
v_add_f32 v17, v9, v13 // (op 24287) 




s_lshl_b32 s0, s4, 9 // (op 24263) 
s_add_u32 s0, s0, 128 // (op 24263) 
v_add_u32 v0, v21, s0 // (op 24263) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24263) Store value




s_lshl_b32 s0, s2, 7 // (op 24292) 
s_mul_i32 s1, 5, s0 // (op 24292) 
s_add_u32 s1, s1, 128 // (op 24292) 

v_add_u32 v0, v20, s1 // (op 24292) 


v_accvgpr_read v14, a52 // (op 24310) 
v_accvgpr_read v15, a53 // (op 24310) 
v_accvgpr_read v16, a54 // (op 24310) 
v_accvgpr_read v17, a55 // (op 24310) 
v_mul_f32 v10, v19, v14 // (op 24310) 
v_mul_f32 v11, v19, v15 // (op 24310) 
v_mul_f32 v12, v19, v16 // (op 24310) 
v_mul_f32 v13, v19, v17 // (op 24310) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24292) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24290) 
v_mul_f32 v7, v18, v15 // (op 24290) 
v_mul_f32 v8, v18, v16 // (op 24290) 
v_mul_f32 v9, v18, v17 // (op 24290) 

v_add_f32 v14, v6, v10 // (op 24291) 
v_add_f32 v15, v7, v11 // (op 24291) 
v_add_f32 v16, v8, v12 // (op 24291) 
v_add_f32 v17, v9, v13 // (op 24291) 




s_lshl_b32 s0, s4, 7 // (op 24265) 
s_mul_i32 s1, 5, s0 // (op 24265) 
s_add_u32 s1, s1, 128 // (op 24265) 

v_add_u32 v0, v21, s1 // (op 24265) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24265) Store value




s_lshl_b32 s0, s2, 7 // (op 24296) 
s_mul_i32 s1, 6, s0 // (op 24296) 
s_add_u32 s1, s1, 128 // (op 24296) 

v_add_u32 v0, v20, s1 // (op 24296) 


v_accvgpr_read v14, a56 // (op 24312) 
v_accvgpr_read v15, a57 // (op 24312) 
v_accvgpr_read v16, a58 // (op 24312) 
v_accvgpr_read v17, a59 // (op 24312) 
v_mul_f32 v10, v19, v14 // (op 24312) 
v_mul_f32 v11, v19, v15 // (op 24312) 
v_mul_f32 v12, v19, v16 // (op 24312) 
v_mul_f32 v13, v19, v17 // (op 24312) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24296) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24294) 
v_mul_f32 v7, v18, v15 // (op 24294) 
v_mul_f32 v8, v18, v16 // (op 24294) 
v_mul_f32 v9, v18, v17 // (op 24294) 

v_add_f32 v14, v6, v10 // (op 24295) 
v_add_f32 v15, v7, v11 // (op 24295) 
v_add_f32 v16, v8, v12 // (op 24295) 
v_add_f32 v17, v9, v13 // (op 24295) 




s_lshl_b32 s0, s4, 7 // (op 24267) 
s_mul_i32 s1, 6, s0 // (op 24267) 
s_add_u32 s1, s1, 128 // (op 24267) 

v_add_u32 v0, v21, s1 // (op 24267) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24267) Store value




s_lshl_b32 s0, s2, 7 // (op 24300) 
s_mul_i32 s1, 7, s0 // (op 24300) 
s_add_u32 s1, s1, 128 // (op 24300) 

v_add_u32 v0, v20, s1 // (op 24300) 


v_accvgpr_read v14, a60 // (op 24314) 
v_accvgpr_read v15, a61 // (op 24314) 
v_accvgpr_read v16, a62 // (op 24314) 
v_accvgpr_read v17, a63 // (op 24314) 
v_mul_f32 v10, v19, v14 // (op 24314) 
v_mul_f32 v11, v19, v15 // (op 24314) 
v_mul_f32 v12, v19, v16 // (op 24314) 
v_mul_f32 v13, v19, v17 // (op 24314) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24300) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24298) 
v_mul_f32 v7, v18, v15 // (op 24298) 
v_mul_f32 v8, v18, v16 // (op 24298) 
v_mul_f32 v9, v18, v17 // (op 24298) 

v_add_f32 v14, v6, v10 // (op 24299) 
v_add_f32 v15, v7, v11 // (op 24299) 
v_add_f32 v16, v8, v12 // (op 24299) 
v_add_f32 v17, v9, v13 // (op 24299) 




s_lshl_b32 s0, s4, 7 // (op 24269) 
s_mul_i32 s1, 7, s0 // (op 24269) 
s_add_u32 s1, s1, 128 // (op 24269) 

v_add_u32 v0, v21, s1 // (op 24269) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24269) Store value




v_mov_b32 v0, v20 // (op 145) reg expression

s_nop 0
v_accvgpr_read v14, a0 // (op 142) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v10, v19, v14 // (op 142) 
v_mul_f32 v11, v19, v15 // (op 142) 
v_mul_f32 v12, v19, v16 // (op 142) 
v_mul_f32 v13, v19, v17 // (op 142) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 145) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 146) 
v_mul_f32 v7, v18, v15 // (op 146) 
v_mul_f32 v8, v18, v16 // (op 146) 
v_mul_f32 v9, v18, v17 // (op 146) 

v_add_f32 v14, v6, v10 // (op 143) 
v_add_f32 v15, v7, v11 // (op 143) 
v_add_f32 v16, v8, v12 // (op 143) 
v_add_f32 v17, v9, v13 // (op 143) 




v_mov_b32 v0, v21 // (op 144) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 24090) 
v_add_u32 v0, v20, s0 // (op 24090) 


v_accvgpr_read v14, a4 // (op 24087) 
v_accvgpr_read v15, a5 // (op 24087) 
v_accvgpr_read v16, a6 // (op 24087) 
v_accvgpr_read v17, a7 // (op 24087) 
v_mul_f32 v10, v19, v14 // (op 24087) 
v_mul_f32 v11, v19, v15 // (op 24087) 
v_mul_f32 v12, v19, v16 // (op 24087) 
v_mul_f32 v13, v19, v17 // (op 24087) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24090) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24091) 
v_mul_f32 v7, v18, v15 // (op 24091) 
v_mul_f32 v8, v18, v16 // (op 24091) 
v_mul_f32 v9, v18, v17 // (op 24091) 

v_add_f32 v14, v6, v10 // (op 24088) 
v_add_f32 v15, v7, v11 // (op 24088) 
v_add_f32 v16, v8, v12 // (op 24088) 
v_add_f32 v17, v9, v13 // (op 24088) 




s_lshl_b32 s0, s4, 7 // (op 24089) 
v_add_u32 v0, v21, s0 // (op 24089) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24089) Store value




s_lshl_b32 s0, s2, 8 // (op 24102) 
v_add_u32 v0, v20, s0 // (op 24102) 


v_accvgpr_read v14, a8 // (op 24099) 
v_accvgpr_read v15, a9 // (op 24099) 
v_accvgpr_read v16, a10 // (op 24099) 
v_accvgpr_read v17, a11 // (op 24099) 
v_mul_f32 v10, v19, v14 // (op 24099) 
v_mul_f32 v11, v19, v15 // (op 24099) 
v_mul_f32 v12, v19, v16 // (op 24099) 
v_mul_f32 v13, v19, v17 // (op 24099) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24102) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24103) 
v_mul_f32 v7, v18, v15 // (op 24103) 
v_mul_f32 v8, v18, v16 // (op 24103) 
v_mul_f32 v9, v18, v17 // (op 24103) 

v_add_f32 v14, v6, v10 // (op 24100) 
v_add_f32 v15, v7, v11 // (op 24100) 
v_add_f32 v16, v8, v12 // (op 24100) 
v_add_f32 v17, v9, v13 // (op 24100) 




s_lshl_b32 s0, s4, 8 // (op 24101) 
v_add_u32 v0, v21, s0 // (op 24101) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24101) Store value




s_lshl_b32 s0, s2, 7 // (op 24114) 
s_mul_i32 s1, 3, s0 // (op 24114) 

v_add_u32 v0, v20, s1 // (op 24114) 


v_accvgpr_read v14, a12 // (op 24111) 
v_accvgpr_read v15, a13 // (op 24111) 
v_accvgpr_read v16, a14 // (op 24111) 
v_accvgpr_read v17, a15 // (op 24111) 
v_mul_f32 v10, v19, v14 // (op 24111) 
v_mul_f32 v11, v19, v15 // (op 24111) 
v_mul_f32 v12, v19, v16 // (op 24111) 
v_mul_f32 v13, v19, v17 // (op 24111) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24114) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24115) 
v_mul_f32 v7, v18, v15 // (op 24115) 
v_mul_f32 v8, v18, v16 // (op 24115) 
v_mul_f32 v9, v18, v17 // (op 24115) 

v_add_f32 v14, v6, v10 // (op 24112) 
v_add_f32 v15, v7, v11 // (op 24112) 
v_add_f32 v16, v8, v12 // (op 24112) 
v_add_f32 v17, v9, v13 // (op 24112) 




s_lshl_b32 s0, s4, 7 // (op 24113) 
s_mul_i32 s1, 3, s0 // (op 24113) 

v_add_u32 v0, v21, s1 // (op 24113) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24113) Store value




s_lshl_b32 s0, s2, 9 // (op 24126) 
v_add_u32 v0, v20, s0 // (op 24126) 


v_accvgpr_read v14, a16 // (op 24123) 
v_accvgpr_read v15, a17 // (op 24123) 
v_accvgpr_read v16, a18 // (op 24123) 
v_accvgpr_read v17, a19 // (op 24123) 
v_mul_f32 v10, v19, v14 // (op 24123) 
v_mul_f32 v11, v19, v15 // (op 24123) 
v_mul_f32 v12, v19, v16 // (op 24123) 
v_mul_f32 v13, v19, v17 // (op 24123) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24126) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24127) 
v_mul_f32 v7, v18, v15 // (op 24127) 
v_mul_f32 v8, v18, v16 // (op 24127) 
v_mul_f32 v9, v18, v17 // (op 24127) 

v_add_f32 v14, v6, v10 // (op 24124) 
v_add_f32 v15, v7, v11 // (op 24124) 
v_add_f32 v16, v8, v12 // (op 24124) 
v_add_f32 v17, v9, v13 // (op 24124) 




s_lshl_b32 s0, s4, 9 // (op 24125) 
v_add_u32 v0, v21, s0 // (op 24125) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24125) Store value




s_lshl_b32 s0, s2, 7 // (op 24138) 
s_mul_i32 s1, 5, s0 // (op 24138) 

v_add_u32 v0, v20, s1 // (op 24138) 


v_accvgpr_read v14, a20 // (op 24135) 
v_accvgpr_read v15, a21 // (op 24135) 
v_accvgpr_read v16, a22 // (op 24135) 
v_accvgpr_read v17, a23 // (op 24135) 
v_mul_f32 v10, v19, v14 // (op 24135) 
v_mul_f32 v11, v19, v15 // (op 24135) 
v_mul_f32 v12, v19, v16 // (op 24135) 
v_mul_f32 v13, v19, v17 // (op 24135) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24138) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24139) 
v_mul_f32 v7, v18, v15 // (op 24139) 
v_mul_f32 v8, v18, v16 // (op 24139) 
v_mul_f32 v9, v18, v17 // (op 24139) 

v_add_f32 v14, v6, v10 // (op 24136) 
v_add_f32 v15, v7, v11 // (op 24136) 
v_add_f32 v16, v8, v12 // (op 24136) 
v_add_f32 v17, v9, v13 // (op 24136) 




s_lshl_b32 s0, s4, 7 // (op 24137) 
s_mul_i32 s1, 5, s0 // (op 24137) 

v_add_u32 v0, v21, s1 // (op 24137) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24137) Store value




s_lshl_b32 s0, s2, 7 // (op 24150) 
s_mul_i32 s1, 6, s0 // (op 24150) 

v_add_u32 v0, v20, s1 // (op 24150) 


v_accvgpr_read v14, a24 // (op 24147) 
v_accvgpr_read v15, a25 // (op 24147) 
v_accvgpr_read v16, a26 // (op 24147) 
v_accvgpr_read v17, a27 // (op 24147) 
v_mul_f32 v10, v19, v14 // (op 24147) 
v_mul_f32 v11, v19, v15 // (op 24147) 
v_mul_f32 v12, v19, v16 // (op 24147) 
v_mul_f32 v13, v19, v17 // (op 24147) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24150) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24151) 
v_mul_f32 v7, v18, v15 // (op 24151) 
v_mul_f32 v8, v18, v16 // (op 24151) 
v_mul_f32 v9, v18, v17 // (op 24151) 

v_add_f32 v14, v6, v10 // (op 24148) 
v_add_f32 v15, v7, v11 // (op 24148) 
v_add_f32 v16, v8, v12 // (op 24148) 
v_add_f32 v17, v9, v13 // (op 24148) 




s_lshl_b32 s0, s4, 7 // (op 24149) 
s_mul_i32 s1, 6, s0 // (op 24149) 

v_add_u32 v0, v21, s1 // (op 24149) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24149) Store value




s_lshl_b32 s0, s2, 7 // (op 24162) 
s_mul_i32 s1, 7, s0 // (op 24162) 

v_add_u32 v0, v20, s1 // (op 24162) 


v_accvgpr_read v14, a28 // (op 24159) 
v_accvgpr_read v15, a29 // (op 24159) 
v_accvgpr_read v16, a30 // (op 24159) 
v_accvgpr_read v17, a31 // (op 24159) 
v_mul_f32 v10, v19, v14 // (op 24159) 
v_mul_f32 v11, v19, v15 // (op 24159) 
v_mul_f32 v12, v19, v16 // (op 24159) 
v_mul_f32 v13, v19, v17 // (op 24159) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 24162) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v6, v18, v14 // (op 24163) 
v_mul_f32 v7, v18, v15 // (op 24163) 
v_mul_f32 v8, v18, v16 // (op 24163) 
v_mul_f32 v9, v18, v17 // (op 24163) 

v_add_f32 v14, v6, v10 // (op 24160) 
v_add_f32 v15, v7, v11 // (op 24160) 
v_add_f32 v16, v8, v12 // (op 24160) 
v_add_f32 v17, v9, v13 // (op 24160) 




s_lshl_b32 s0, s4, 7 // (op 24161) 
s_mul_i32 s1, 7, s0 // (op 24161) 

v_add_u32 v0, v21, s1 // (op 24161) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 24161) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalBottom_0_beta_133:








































































s_endpgm  // End of RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
.LRR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_end:
.size RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4, .LRR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_end-RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
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
  - .name: RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
    .symbol: RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4.kd
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

