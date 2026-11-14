// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4
.p2align 8
.type RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4,@function
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 13790) 
v_and_b32 v3, 48, v1 // (op 13790) 
v_sub_u32 v4, v0, v3 // (op 13790) 

v_lshrrev_b32 v3, 4, v0 // (op 13790) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 13790) 


v_lshrrev_b32 v3, 7, v1 // (op 13790) 
v_lshlrev_b32 v4, 4, v3 // (op 13790) 

v_lshl_add_u32 v3, v0, 7, v4 // (op 13790) 


v_lshlrev_b32 v0, 2, v3 // (op 13790) 

v_lshrrev_b32 v3, 3, v0 // (op 13790) 

v_lshrrev_b32 v0, 6, v1 // (op 13847) 
v_and_b32 v4, 1, v0 // (op 13847) 

v_and_b32 v0, 15, v1 // (op 13847) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 13847) 


v_and_b32 v0, 63, v1 // (op 13847) 
v_lshrrev_b32 v4, 4, v0 // (op 13847) 

v_lshlrev_b32 v0, 5, v4 // (op 13847) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 13847) 


v_lshlrev_b32 v0, 2, v4 // (op 13847) 

v_lshrrev_b32 v4, 3, v0 // (op 13847) 

v_lshrrev_b32 v0, 7, v1 // (op 13905) 
v_and_b32 v5, 15, v1 // (op 13905) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 13905) 


v_and_b32 v0, 63, v1 // (op 13905) 
v_lshrrev_b32 v5, 4, v0 // (op 13905) 

v_lshl_add_u32 v0, v5, 7, v6 // (op 13905) 


v_lshrrev_b32 v5, 6, v1 // (op 13963) 
v_and_b32 v6, 1, v5 // (op 13963) 

v_and_b32 v5, 15, v1 // (op 13963) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 13963) 


v_and_b32 v5, 63, v1 // (op 13963) 
v_lshrrev_b32 v6, 4, v5 // (op 13963) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 13963) 


v_and_b32 v6, 63, v1 // (op 14030) 
v_and_b32 v7, 48, v1 // (op 14030) 
v_sub_u32 v8, v6, v7 // (op 14030) 

v_lshrrev_b32 v7, 4, v6 // (op 14030) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 14030) 


v_lshrrev_b32 v7, 7, v1 // (op 14030) 
v_lshlrev_b32 v8, 4, v7 // (op 14030) 

v_lshl_add_u32 v7, v6, 7, v8 // (op 14030) 


v_lshlrev_b32 v6, 2, v7 // (op 14030) 

v_lshrrev_b32 v7, 3, v6 // (op 14030) 

v_lshrrev_b32 v6, 6, v1 // (op 14074) 
v_and_b32 v8, 1, v6 // (op 14074) 

v_and_b32 v6, 15, v1 // (op 14074) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 14074) 


v_and_b32 v6, 63, v1 // (op 14074) 
v_lshrrev_b32 v8, 4, v6 // (op 14074) 

v_lshlrev_b32 v6, 5, v8 // (op 14074) 

v_lshl_add_u32 v8, v9, 7, v6 // (op 14074) 


v_lshlrev_b32 v6, 2, v8 // (op 14074) 

v_lshrrev_b32 v8, 3, v6 // (op 14074) 

v_lshrrev_b32 v6, 7, v1 // (op 14103) 
v_and_b32 v9, 15, v1 // (op 14103) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 14103) 


v_and_b32 v6, 63, v1 // (op 14103) 
v_lshrrev_b32 v9, 4, v6 // (op 14103) 

v_lshl_add_u32 v6, v9, 7, v10 // (op 14103) 


v_lshrrev_b32 v9, 6, v1 // (op 14132) 
v_and_b32 v10, 1, v9 // (op 14132) 

v_and_b32 v9, 15, v1 // (op 14132) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 14132) 


v_and_b32 v9, 63, v1 // (op 14132) 
v_lshrrev_b32 v10, 4, v9 // (op 14132) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 14132) 


v_and_b32 v10, 63, v1 // (op 14176) 
v_and_b32 v11, 48, v1 // (op 14176) 
v_sub_u32 v12, v10, v11 // (op 14176) 

v_lshrrev_b32 v11, 4, v10 // (op 14176) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 14176) 


v_lshrrev_b32 v11, 7, v1 // (op 14176) 
v_lshlrev_b32 v12, 4, v11 // (op 14176) 

v_lshl_add_u32 v11, v10, 7, v12 // (op 14176) 


v_lshlrev_b32 v10, 2, v11 // (op 14176) 

v_lshrrev_b32 v11, 3, v10 // (op 14176) 

v_lshrrev_b32 v10, 6, v1 // (op 14220) 
v_and_b32 v12, 1, v10 // (op 14220) 

v_and_b32 v10, 15, v1 // (op 14220) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 14220) 


v_and_b32 v10, 63, v1 // (op 14220) 
v_lshrrev_b32 v12, 4, v10 // (op 14220) 

v_lshlrev_b32 v10, 5, v12 // (op 14220) 

v_lshl_add_u32 v12, v13, 7, v10 // (op 14220) 


v_lshlrev_b32 v10, 2, v12 // (op 14220) 

v_lshrrev_b32 v12, 3, v10 // (op 14220) 

v_lshrrev_b32 v10, 7, v1 // (op 14249) 
v_and_b32 v13, 15, v1 // (op 14249) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 14249) 


v_and_b32 v10, 63, v1 // (op 14249) 
v_lshrrev_b32 v13, 4, v10 // (op 14249) 

v_lshl_add_u32 v10, v13, 7, v14 // (op 14249) 


v_lshrrev_b32 v13, 6, v1 // (op 14278) 
v_and_b32 v14, 1, v13 // (op 14278) 

v_and_b32 v13, 15, v1 // (op 14278) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 14278) 


v_and_b32 v13, 63, v1 // (op 14278) 
v_lshrrev_b32 v14, 4, v13 // (op 14278) 

v_lshl_add_u32 v13, v15, 2, v14 // (op 14278) 


v_and_b32 v14, 63, v1 // (op 14322) 
v_and_b32 v15, 48, v1 // (op 14322) 
v_sub_u32 v16, v14, v15 // (op 14322) 

v_lshrrev_b32 v15, 4, v14 // (op 14322) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 14322) 


v_lshrrev_b32 v15, 7, v1 // (op 14322) 
v_lshlrev_b32 v16, 4, v15 // (op 14322) 

v_lshl_add_u32 v15, v14, 7, v16 // (op 14322) 


v_lshlrev_b32 v14, 2, v15 // (op 14322) 

v_lshrrev_b32 v15, 3, v14 // (op 14322) 

v_lshrrev_b32 v14, 6, v1 // (op 14366) 
v_and_b32 v16, 1, v14 // (op 14366) 

v_and_b32 v14, 15, v1 // (op 14366) 
v_lshl_add_u32 v17, v16, 4, v14 // (op 14366) 


v_and_b32 v14, 63, v1 // (op 14366) 
v_lshrrev_b32 v16, 4, v14 // (op 14366) 

v_lshlrev_b32 v14, 5, v16 // (op 14366) 

v_lshl_add_u32 v16, v17, 7, v14 // (op 14366) 


v_lshlrev_b32 v14, 2, v16 // (op 14366) 

v_lshrrev_b32 v16, 3, v14 // (op 14366) 

v_lshrrev_b32 v14, 7, v1 // (op 14395) 
v_and_b32 v17, 15, v1 // (op 14395) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 14395) 


v_and_b32 v14, 63, v1 // (op 14395) 
v_lshrrev_b32 v17, 4, v14 // (op 14395) 

v_lshl_add_u32 v14, v17, 7, v18 // (op 14395) 


v_lshrrev_b32 v17, 6, v1 // (op 14424) 
v_and_b32 v18, 1, v17 // (op 14424) 

v_and_b32 v17, 15, v1 // (op 14424) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 14424) 


v_and_b32 v17, 63, v1 // (op 14424) 
v_lshrrev_b32 v18, 4, v17 // (op 14424) 

v_lshl_add_u32 v17, v19, 2, v18 // (op 14424) 


v_lshrrev_b32 v18, 7, v1 // (op 13729) 
v_and_b32 v19, 63, v1 // (op 13729) 
v_lshrrev_b32 v20, 4, v19 // (op 13729) 
v_lshlrev_b32 v21, 2, v20 // (op 13729) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 13729) 


v_lshl_add_u32 v18, s16, 7, v20 // (op 13729) 

v_lshrrev_b32 v20, 2, v19 // (op 13729) 

v_and_b32 v19, 3, v20 // (op 13729) 

v_and_b32 v20, 3, v1 // (op 13729) 
v_lshl_add_u32 v21, v19, 2, v20 // (op 13729) 


v_lshrrev_b32 v19, 6, v1 // (op 13729) 
v_and_b32 v20, 1, v19 // (op 13729) 

v_lshl_add_u32 v19, v20, 4, v21 // (op 13729) 


v_lshl_add_u32 v20, s17, 7, v19 // (op 13729) 

v_mul_lo_u32 v19, v20, s2 // (op 13729) 

v_add_lshl_u32 v20, v18, v19, 1 // (op 13729) 


v_lshrrev_b32 v18, 7, v1 // (op 13984) 
v_and_b32 v19, 63, v1 // (op 13984) 
v_lshrrev_b32 v21, 4, v19 // (op 13984) 
s_mul_i32 s1, s2, s9 // (op 13731) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 13731) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 13731) least significant
s_mul_hi_u32 s19, s2, s8 // (op 13731) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13731) most significant: sum
s_add_u32 s1, s1, s19 // (op 13731) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 13731) 

v_lshlrev_b32 v22, 2, v21 // (op 13984) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 13984) 


v_lshl_add_u32 v18, s16, 7, v21 // (op 13984) 

v_lshrrev_b32 v21, 2, v19 // (op 13984) 

v_and_b32 v19, 3, v21 // (op 13984) 

v_and_b32 v21, 3, v1 // (op 13984) 
v_lshl_add_u32 v22, v19, 2, v21 // (op 13984) 


v_lshrrev_b32 v19, 6, v1 // (op 13984) 
v_and_b32 v21, 1, v19 // (op 13984) 

v_lshl_add_u32 v19, v21, 4, v22 // (op 13984) 


v_lshl_add_u32 v21, s17, 7, v19 // (op 13984) 

v_mul_lo_u32 v19, v21, s4 // (op 13984) 

v_add_lshl_u32 v21, v18, v19, 1 // (op 13984) 


s_mov_b32 s40, s6 // (op 13731) reg expression
s_mov_b32 s41, s7 // (op 13731) reg expression
s_mov_b32 s42, s18 // (op 13731) reg expression
s_mov_b32 s43, 131072 // (op 13731) call()

s_mul_i32 s1, s4, s9 // (op 13986) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 13986) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 13986) least significant
s_mul_hi_u32 s7, s4, s8 // (op 13986) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13986) most significant: sum
s_add_u32 s1, s1, s7 // (op 13986) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 13986) 

s_mov_b32 s44, s10 // (op 13986) reg expression
s_mov_b32 s45, s11 // (op 13986) reg expression
s_mov_b32 s46, s6 // (op 13986) reg expression
s_mov_b32 s47, 131072 // (op 13986) call()


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
v_accvgpr_write a16, 0.00000 // (op 1516) call()
v_accvgpr_write a17, 0.00000 // (op 1516) call()
v_accvgpr_write a18, 0.00000 // (op 1516) call()
v_accvgpr_write a19, 0.00000 // (op 1516) call()
v_accvgpr_write a20, 0.00000 // (op 2009) call()
v_accvgpr_write a21, 0.00000 // (op 2009) call()
v_accvgpr_write a22, 0.00000 // (op 2009) call()
v_accvgpr_write a23, 0.00000 // (op 2009) call()
v_accvgpr_write a24, 0.00000 // (op 2010) call()
v_accvgpr_write a25, 0.00000 // (op 2010) call()
v_accvgpr_write a26, 0.00000 // (op 2010) call()
v_accvgpr_write a27, 0.00000 // (op 2010) call()
v_accvgpr_write a28, 0.00000 // (op 2011) call()
v_accvgpr_write a29, 0.00000 // (op 2011) call()
v_accvgpr_write a30, 0.00000 // (op 2011) call()
v_accvgpr_write a31, 0.00000 // (op 2011) call()
v_accvgpr_write a32, 0.00000 // (op 2806) call()
v_accvgpr_write a33, 0.00000 // (op 2806) call()
v_accvgpr_write a34, 0.00000 // (op 2806) call()
v_accvgpr_write a35, 0.00000 // (op 2806) call()
v_accvgpr_write a36, 0.00000 // (op 3299) call()
v_accvgpr_write a37, 0.00000 // (op 3299) call()
v_accvgpr_write a38, 0.00000 // (op 3299) call()
v_accvgpr_write a39, 0.00000 // (op 3299) call()
v_accvgpr_write a40, 0.00000 // (op 3300) call()
v_accvgpr_write a41, 0.00000 // (op 3300) call()
v_accvgpr_write a42, 0.00000 // (op 3300) call()
v_accvgpr_write a43, 0.00000 // (op 3300) call()
v_accvgpr_write a44, 0.00000 // (op 3301) call()
v_accvgpr_write a45, 0.00000 // (op 3301) call()
v_accvgpr_write a46, 0.00000 // (op 3301) call()
v_accvgpr_write a47, 0.00000 // (op 3301) call()
v_accvgpr_write a48, 0.00000 // (op 4096) call()
v_accvgpr_write a49, 0.00000 // (op 4096) call()
v_accvgpr_write a50, 0.00000 // (op 4096) call()
v_accvgpr_write a51, 0.00000 // (op 4096) call()
v_accvgpr_write a52, 0.00000 // (op 4589) call()
v_accvgpr_write a53, 0.00000 // (op 4589) call()
v_accvgpr_write a54, 0.00000 // (op 4589) call()
v_accvgpr_write a55, 0.00000 // (op 4589) call()
v_accvgpr_write a56, 0.00000 // (op 4590) call()
v_accvgpr_write a57, 0.00000 // (op 4590) call()
v_accvgpr_write a58, 0.00000 // (op 4590) call()
v_accvgpr_write a59, 0.00000 // (op 4590) call()
v_accvgpr_write a60, 0.00000 // (op 4591) call()
v_accvgpr_write a61, 0.00000 // (op 4591) call()
v_accvgpr_write a62, 0.00000 // (op 4591) call()
v_accvgpr_write a63, 0.00000 // (op 4591) call()
v_and_b32 v22, -4, v1 // (op 13491) 
v_sub_u32 v23, v1, v22 // (op 13491) 

v_lshlrev_b32 v22, 5, v23 // (op 13491) 

v_lshl_add_u32 v23, s16, 7, v22 // (op 13491) 

v_mov_b32 v25, 0 // (op 13491) convert
v_mov_b32 v24, v23 // (op 13491) convert

v_lshrrev_b32 v22, 2, v1 // (op 13491) 
v_mul_lo_u32 v27, v22, s13 // (op 13491) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 13491) most significant: high of low * low
v_mul_lo_u32 v26, v22, s12 // (op 13491) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 13491) most significant: sum


v_add_co_u32 v22, s[0:1], v24, v26 // (op 13491) least significant half
v_addc_co_u32 v23, s[0:1], v25, v27, s[0:1] // (op 13491) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 13491) 


v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 13491) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 13493) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 13493) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 13493) least significant
s_mul_hi_u32 s7, s12, s22 // (op 13493) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13493) most significant: sum
s_add_u32 s1, s1, s7 // (op 13493) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 13493) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 13493) 

s_mov_b32 s48, s20 // (op 13493) reg expression
s_mov_b32 s49, s21 // (op 13493) reg expression
s_mov_b32 s50, s0 // (op 13493) reg expression
s_mov_b32 s51, 131072 // (op 13493) call()

v_and_b32 v22, -4, v1 // (op 13510) 
v_sub_u32 v23, v1, v22 // (op 13510) 

v_lshlrev_b32 v22, 5, v23 // (op 13510) 

v_lshl_add_u32 v23, s16, 7, v22 // (op 13510) 

v_mov_b32 v27, 0 // (op 13510) convert
v_mov_b32 v26, v23 // (op 13510) convert

v_lshrrev_b32 v22, 2, v1 // (op 13510) 
v_mul_lo_u32 v29, v22, s13 // (op 13510) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 13510) most significant: high of low * low
v_mul_lo_u32 v28, v22, s12 // (op 13510) least significant: low of low * low
v_add_u32 v29, v29, v23 // (op 13510) most significant: sum


v_add_co_u32 v22, s[0:1], v26, v28 // (op 13510) least significant half
v_addc_co_u32 v23, s[0:1], v27, v29, s[0:1] // (op 13510) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 13510) 


v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 13510) 

s_mul_i32 s1, s12, s23 // (op 13512) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 13512) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 13512) least significant
s_mul_hi_u32 s7, s12, s22 // (op 13512) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13512) most significant: sum
s_add_u32 s1, s1, s7 // (op 13512) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 13512) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 13512) 

s_mov_b32 s48, s20 // (op 13512) reg expression
s_mov_b32 s49, s21 // (op 13512) reg expression
s_mov_b32 s50, s0 // (op 13512) reg expression
s_mov_b32 s51, 131072 // (op 13512) call()

v_and_b32 v22, -4, v1 // (op 13555) 
v_sub_u32 v23, v1, v22 // (op 13555) 

v_lshlrev_b32 v22, 5, v23 // (op 13555) 

v_mov_b32 v29, 0 // (op 13555) convert
v_mov_b32 v28, v22 // (op 13555) convert

v_lshrrev_b32 v22, 2, v1 // (op 13555) 
v_lshl_add_u32 v23, s17, 7, v22 // (op 13555) 

v_mul_lo_u32 v31, v23, s25 // (op 13555) most significant: low of low * high
v_mul_hi_u32 v22, v23, s24 // (op 13555) most significant: high of low * low
v_mul_lo_u32 v30, v23, s24 // (op 13555) least significant: low of low * low
v_add_u32 v31, v31, v22 // (op 13555) most significant: sum


v_add_co_u32 v22, s[0:1], v28, v30 // (op 13555) least significant half
v_addc_co_u32 v23, s[0:1], v29, v31, s[0:1] // (op 13555) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 13555) 


v_lshrrev_b64 v[28:29], 3, v[22:23] // (op 13555) 

s_mul_i32 s1, s24, s9 // (op 13557) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 13557) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13557) least significant
s_mul_hi_u32 s7, s24, s8 // (op 13557) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13557) most significant: sum
s_add_u32 s1, s1, s7 // (op 13557) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 13557) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 13557) 

s_mov_b32 s52, s26 // (op 13557) reg expression
s_mov_b32 s53, s27 // (op 13557) reg expression
s_mov_b32 s54, s0 // (op 13557) reg expression
s_mov_b32 s55, 131072 // (op 13557) call()

v_and_b32 v22, -4, v1 // (op 13574) 
v_sub_u32 v23, v1, v22 // (op 13574) 

v_lshlrev_b32 v22, 5, v23 // (op 13574) 

v_mov_b32 v31, 0 // (op 13574) convert
v_mov_b32 v30, v22 // (op 13574) convert

v_lshrrev_b32 v22, 2, v1 // (op 13574) 
v_lshl_add_u32 v23, s17, 7, v22 // (op 13574) 

v_mul_lo_u32 v33, v23, s25 // (op 13574) most significant: low of low * high
v_mul_hi_u32 v22, v23, s24 // (op 13574) most significant: high of low * low
v_mul_lo_u32 v32, v23, s24 // (op 13574) least significant: low of low * low
v_add_u32 v33, v33, v22 // (op 13574) most significant: sum


v_add_co_u32 v22, s[0:1], v30, v32 // (op 13574) least significant half
v_addc_co_u32 v23, s[0:1], v31, v33, s[0:1] // (op 13574) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 13574) 


v_lshrrev_b64 v[30:31], 3, v[22:23] // (op 13574) 

s_mul_i32 s1, s24, s9 // (op 13576) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 13576) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13576) least significant
s_mul_hi_u32 s7, s24, s8 // (op 13576) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13576) most significant: sum
s_add_u32 s1, s1, s7 // (op 13576) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 13576) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 13576) 

s_mov_b32 s52, s26 // (op 13576) reg expression
s_mov_b32 s53, s27 // (op 13576) reg expression
s_mov_b32 s54, s0 // (op 13576) reg expression
s_mov_b32 s55, 131072 // (op 13576) call()

v_and_b32 v22, -128, v1 // (op 13619) 
v_sub_u32 v23, v1, v22 // (op 13619) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 13619) 

v_lshrrev_b32 v23, 7, v1 // (op 13619) 
v_lshlrev_b32 v32, 1, v23 // (op 13619) 

v_mul_lo_u32 v23, v32, s28 // (op 13619) 

v_add_u32 v32, v22, v23 // (op 13619) 


s_mul_i32 s1, s28, s33 // (op 13621) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 13621) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 13621) least significant
s_mul_hi_u32 s7, s28, s32 // (op 13621) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13621) most significant: sum
s_add_u32 s1, s1, s7 // (op 13621) most significant: sum


s_mov_b32 s56, s30 // (op 13621) reg expression
s_mov_b32 s57, s31 // (op 13621) reg expression
s_mov_b32 s58, s0 // (op 13621) reg expression
s_mov_b32 s59, 131072 // (op 13621) call()

v_and_b32 v22, -128, v1 // (op 13638) 
v_sub_u32 v23, v1, v22 // (op 13638) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 13638) 

v_lshrrev_b32 v23, 7, v1 // (op 13638) 
v_lshlrev_b32 v33, 1, v23 // (op 13638) 

v_mul_lo_u32 v23, v33, s28 // (op 13638) 

v_add_u32 v33, v22, v23 // (op 13638) 


s_mul_i32 s1, s28, s33 // (op 13640) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 13640) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 13640) least significant
s_mul_hi_u32 s7, s28, s32 // (op 13640) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13640) most significant: sum
s_add_u32 s1, s1, s7 // (op 13640) most significant: sum


s_mov_b32 s56, s30 // (op 13640) reg expression
s_mov_b32 s57, s31 // (op 13640) reg expression
s_mov_b32 s58, s0 // (op 13640) reg expression
s_mov_b32 s59, 131072 // (op 13640) call()

v_and_b32 v22, -4, v1 // (op 13683) 
v_sub_u32 v23, v1, v22 // (op 13683) 

v_lshrrev_b32 v22, 2, v1 // (op 13683) 
v_lshlrev_b32 v34, 1, v22 // (op 13683) 

v_lshl_add_u32 v22, s17, 7, v34 // (op 13683) 

v_mul_lo_u32 v34, v22, s34 // (op 13683) 

v_add_u32 v22, v23, v34 // (op 13683) 


s_mul_i32 s1, s34, s9 // (op 13685) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 13685) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 13685) least significant
s_mul_hi_u32 s7, s34, s8 // (op 13685) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13685) most significant: sum
s_add_u32 s1, s1, s7 // (op 13685) most significant: sum


s_mov_b32 s60, s36 // (op 13685) reg expression
s_mov_b32 s61, s37 // (op 13685) reg expression
s_mov_b32 s62, s0 // (op 13685) reg expression
s_mov_b32 s63, 131072 // (op 13685) call()

v_and_b32 v23, -4, v1 // (op 13702) 
v_sub_u32 v34, v1, v23 // (op 13702) 

v_lshrrev_b32 v23, 2, v1 // (op 13702) 
v_lshlrev_b32 v35, 1, v23 // (op 13702) 

v_lshl_add_u32 v23, s17, 7, v35 // (op 13702) 

v_mul_lo_u32 v35, v23, s34 // (op 13702) 

v_add_u32 v23, v34, v35 // (op 13702) 


s_mul_i32 s1, s34, s9 // (op 13704) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 13704) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 13704) least significant
s_mul_hi_u32 s7, s34, s8 // (op 13704) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13704) most significant: sum
s_add_u32 s1, s1, s7 // (op 13704) most significant: sum


s_mov_b32 s60, s36 // (op 13704) reg expression
s_mov_b32 s61, s37 // (op 13704) reg expression
s_mov_b32 s62, s0 // (op 13704) reg expression
s_mov_b32 s63, 131072 // (op 13704) call()

v_and_b32 v34, -4, v1 // (op 13830) 
v_sub_u32 v35, v1, v34 // (op 13830) 

v_lshlrev_b32 v34, 5, v35 // (op 13830) 

v_lshrrev_b32 v35, 2, v1 // (op 13830) 
v_lshl_add_u32 v36, v35, 7, v34 // (op 13830) 


v_lshlrev_b32 v34, 2, v36 // (op 13830) 

v_lshrrev_b32 v35, 3, v34 // (op 13830) 

s_nop 0
v_readfirstlane_b32 s0, v35 // (op 13830) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 13417) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13417) 

v_lshrrev_b64 v[34:35], 3, s[10:11] // (op 13417) 


v_mov_b64 v[36:37], v[28:29] // (op 13417) 
s_mov_b32 m0, s0 // (op 13417) reg expression
buffer_load_dwordx4 v36, s[52:55], 0 offen offset: 0 lds // (op 13417) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v34 // (op 13417) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 13417) most significant half

s_add_u32 m0, m0, 4096 // (op 13417) 
buffer_load_dwordx4 v36, s[52:55], 0 offen offset: 0 lds // (op 13417) Load value direct to lds




v_mov_b64 v[34:35], s[34:35] // (op 12967) convert

v_mov_b32 v36, v22 // (op 12967) 
buffer_load_ubyte v38, v36, s[60:63], 0 offen offset: 0 // (op 12967) Load value
v_add_u32 v36, v36, v34 // (op 12967) 
buffer_load_ubyte v39, v36, s[60:63], 0 offen offset: 0 // (op 12967) Load value


v_and_b32 v34, -4, v1 // (op 13773) 
v_sub_u32 v35, v1, v34 // (op 13773) 

v_lshlrev_b32 v34, 5, v35 // (op 13773) 

v_lshrrev_b32 v35, 2, v1 // (op 13773) 
v_lshl_add_u32 v36, v35, 7, v34 // (op 13773) 


v_lshlrev_b32 v37, 2, v36 // (op 13773) 

v_lshrrev_b32 v34, 3, v37 // (op 13773) 

s_nop 0
v_readfirstlane_b32 s0, v34 // (op 13773) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 13422) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13422) 

v_lshrrev_b64 v[34:35], 3, s[10:11] // (op 13422) 


v_mov_b64 v[36:37], v[24:25] // (op 13422) reg expression
s_add_u32 s1, 8192, s0 // (op 13422) 
s_mov_b32 m0, s1 // (op 13422) reg expression
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 13422) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v34 // (op 13422) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 13422) most significant half

s_add_u32 m0, m0, 4096 // (op 13422) 
buffer_load_dwordx4 v36, s[48:51], 0 offen offset: 0 lds // (op 13422) Load value direct to lds





v_mov_b64 v[34:35], s[28:29] // (op 12973) convert

v_mov_b32 v36, v32 // (op 12973) reg expression
buffer_load_ubyte v40, v36, s[56:59], 0 offen offset: 0 // (op 12973) Load value
v_add_u32 v36, v36, v34 // (op 12973) 
buffer_load_ubyte v41, v36, s[56:59], 0 offen offset: 0 // (op 12973) Load value


v_and_b32 v34, -4, v1 // (op 14057) 
v_sub_u32 v35, v1, v34 // (op 14057) 

v_lshlrev_b32 v34, 5, v35 // (op 14057) 

v_lshrrev_b32 v35, 2, v1 // (op 14057) 
v_lshl_add_u32 v36, v35, 7, v34 // (op 14057) 


v_lshlrev_b32 v37, 2, v36 // (op 14057) 

v_lshrrev_b32 v34, 3, v37 // (op 14057) 

s_nop 0
v_readfirstlane_b32 s0, v34 // (op 14057) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 13427) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13427) 

v_lshrrev_b64 v[34:35], 3, s[10:11] // (op 13427) 


v_mov_b64 v[36:37], v[28:29] // (op 13427) 
s_add_u32 s1, 16384, s0 // (op 13427) 
s_mov_b32 m0, s1 // (op 13427) reg expression
buffer_load_dwordx4 v36, s[52:55], 64 offen offset: 0 lds // (op 13427) Load value direct to lds
v_add_co_u32 v36, s[6:7], v36, v34 // (op 13427) least significant half
v_addc_co_u32 v37, s[6:7], 0, v37, s[6:7] // (op 13427) most significant half

s_add_u32 m0, m0, 4096 // (op 13427) 
buffer_load_dwordx4 v36, s[52:55], 64 offen offset: 0 lds // (op 13427) Load value direct to lds





v_mov_b64 v[34:35], s[34:35] // (op 12979) convert

v_mov_b32 v36, v22 // (op 12979) 
buffer_load_ubyte v42, v36, s[60:63], 0 offen offset:4 // (op 12979) Load value
v_add_u32 v36, v36, v34 // (op 12979) 
buffer_load_ubyte v43, v36, s[60:63], 0 offen offset:4 // (op 12979) Load value


v_and_b32 v34, -4, v1 // (op 14013) 
v_sub_u32 v35, v1, v34 // (op 14013) 

v_lshlrev_b32 v34, 5, v35 // (op 14013) 

v_lshrrev_b32 v35, 2, v1 // (op 14013) 
v_lshl_add_u32 v36, v35, 7, v34 // (op 14013) 


v_lshlrev_b32 v37, 2, v36 // (op 14013) 

v_lshrrev_b32 v34, 3, v37 // (op 14013) 

s_nop 0
v_readfirstlane_b32 s0, v34 // (op 14013) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 13429) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13429) 

v_lshrrev_b64 v[34:35], 3, s[10:11] // (op 13429) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 13429) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13429) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 13429) 

v_mov_b32 v36, s6 // (op 13429) 
v_mov_b32 v37, s7 // (op 13429) 
v_add_co_u32 v44, s[10:11], v24, v36 // (op 13429) least significant half
v_addc_co_u32 v45, s[10:11], v25, v37, s[10:11] // (op 13429) most significant half




s_add_u32 s1, 24576, s0 // (op 13429) 
s_mov_b32 m0, s1 // (op 13429) reg expression
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 13429) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v34 // (op 13429) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 13429) most significant half

s_add_u32 m0, m0, 4096 // (op 13429) 
buffer_load_dwordx4 v44, s[48:51], 0 offen offset: 0 lds // (op 13429) Load value direct to lds





v_mov_b64 v[34:35], s[28:29] // (op 12985) convert

s_lshl_b32 s0, s28, 2 // (op 12985) 
v_add_u32 v36, v32, s0 // (op 12985) 

buffer_load_ubyte v44, v36, s[56:59], 0 offen offset: 0 // (op 12985) Load value
v_add_u32 v36, v36, v34 // (op 12985) 
buffer_load_ubyte v45, v36, s[56:59], 0 offen offset: 0 // (op 12985) Load value


v_and_b32 v34, -4, v1 // (op 14203) 
v_sub_u32 v35, v1, v34 // (op 14203) 

v_lshlrev_b32 v34, 5, v35 // (op 14203) 

v_lshrrev_b32 v35, 2, v1 // (op 14203) 
v_lshl_add_u32 v36, v35, 7, v34 // (op 14203) 


v_lshlrev_b32 v37, 2, v36 // (op 14203) 

v_lshrrev_b32 v34, 3, v37 // (op 14203) 

s_nop 0
v_readfirstlane_b32 s0, v34 // (op 14203) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 13431) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13431) 

v_lshrrev_b64 v[34:35], 3, s[10:11] // (op 13431) 


v_mov_b64 v[36:37], v[28:29] // (op 13431) 
s_add_u32 s1, 32768, s0 // (op 13431) 
s_mov_b32 m0, s1 // (op 13431) reg expression
s_mov_b32 s6, 128 // (op 13431) call()
buffer_load_dwordx4 v36, s[52:55], s6 offen offset: 0 lds // (op 13431) Load value direct to lds
v_add_co_u32 v36, s[10:11], v36, v34 // (op 13431) least significant half
v_addc_co_u32 v37, s[10:11], 0, v37, s[10:11] // (op 13431) most significant half

s_add_u32 m0, m0, 4096 // (op 13431) 
buffer_load_dwordx4 v36, s[52:55], s6 offen offset: 0 lds // (op 13431) Load value direct to lds






v_mov_b64 v[34:35], s[34:35] // (op 12991) convert

v_mov_b32 v36, v22 // (op 12991) 
buffer_load_ubyte v46, v36, s[60:63], 0 offen offset:8 // (op 12991) Load value
v_add_u32 v36, v36, v34 // (op 12991) 
buffer_load_ubyte v47, v36, s[60:63], 0 offen offset:8 // (op 12991) Load value


v_and_b32 v34, -4, v1 // (op 14159) 
v_sub_u32 v35, v1, v34 // (op 14159) 

v_lshlrev_b32 v34, 5, v35 // (op 14159) 

v_lshrrev_b32 v35, 2, v1 // (op 14159) 
v_lshl_add_u32 v36, v35, 7, v34 // (op 14159) 


v_lshlrev_b32 v37, 2, v36 // (op 14159) 

v_lshrrev_b32 v34, 3, v37 // (op 14159) 

s_nop 0
v_readfirstlane_b32 s0, v34 // (op 14159) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 13433) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13433) 

v_lshrrev_b64 v[34:35], 3, s[10:11] // (op 13433) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 13433) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13433) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 13433) 

s_lshl_b64 s[10:11], s[6:7], 1 // (op 13433) 

v_mov_b32 v36, s10 // (op 13433) 
v_mov_b32 v37, s11 // (op 13433) 
v_add_co_u32 v48, s[6:7], v24, v36 // (op 13433) least significant half
v_addc_co_u32 v49, s[6:7], v25, v37, s[6:7] // (op 13433) most significant half




s_add_u32 s1, 40960, s0 // (op 13433) 
s_mov_b32 m0, s1 // (op 13433) reg expression
buffer_load_dwordx4 v48, s[48:51], 0 offen offset: 0 lds // (op 13433) Load value direct to lds
v_add_co_u32 v48, s[6:7], v48, v34 // (op 13433) least significant half
v_addc_co_u32 v49, s[6:7], 0, v49, s[6:7] // (op 13433) most significant half

s_add_u32 m0, m0, 4096 // (op 13433) 
buffer_load_dwordx4 v48, s[48:51], 0 offen offset: 0 lds // (op 13433) Load value direct to lds





v_mov_b64 v[34:35], s[28:29] // (op 12997) convert

s_lshl_b32 s0, s28, 3 // (op 12997) 
v_add_u32 v36, v32, s0 // (op 12997) 

buffer_load_ubyte v48, v36, s[56:59], 0 offen offset: 0 // (op 12997) Load value
v_add_u32 v36, v36, v34 // (op 12997) 
buffer_load_ubyte v49, v36, s[56:59], 0 offen offset: 0 // (op 12997) Load value


v_and_b32 v34, -4, v1 // (op 14349) 
v_sub_u32 v35, v1, v34 // (op 14349) 

v_lshlrev_b32 v34, 5, v35 // (op 14349) 

v_lshrrev_b32 v35, 2, v1 // (op 14349) 
v_lshl_add_u32 v36, v35, 7, v34 // (op 14349) 


v_lshlrev_b32 v37, 2, v36 // (op 14349) 

v_lshrrev_b32 v34, 3, v37 // (op 14349) 

s_nop 0
v_readfirstlane_b32 s0, v34 // (op 14349) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 13435) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13435) 

v_lshrrev_b64 v[34:35], 3, s[10:11] // (op 13435) 


v_mov_b64 v[36:37], v[28:29] // (op 13435) 
s_add_u32 s1, 49152, s0 // (op 13435) 
s_mov_b32 m0, s1 // (op 13435) reg expression
s_mov_b32 s6, 192 // (op 13435) call()
buffer_load_dwordx4 v36, s[52:55], s6 offen offset: 0 lds // (op 13435) Load value direct to lds
v_add_co_u32 v36, s[10:11], v36, v34 // (op 13435) least significant half
v_addc_co_u32 v37, s[10:11], 0, v37, s[10:11] // (op 13435) most significant half

s_add_u32 m0, m0, 4096 // (op 13435) 
buffer_load_dwordx4 v36, s[52:55], s6 offen offset: 0 lds // (op 13435) Load value direct to lds







v_mov_b64 v[28:29], s[34:35] // (op 13003) convert

v_mov_b32 v34, v22 // (op 13003) 
buffer_load_ubyte v36, v34, s[60:63], 0 offen offset:12 // (op 13003) Load value
v_add_u32 v34, v34, v28 // (op 13003) 
buffer_load_ubyte v37, v34, s[60:63], 0 offen offset:12 // (op 13003) Load value



v_and_b32 v22, -4, v1 // (op 14305) 
v_sub_u32 v28, v1, v22 // (op 14305) 

v_lshlrev_b32 v22, 5, v28 // (op 14305) 

v_lshrrev_b32 v28, 2, v1 // (op 14305) 
v_lshl_add_u32 v29, v28, 7, v22 // (op 14305) 


v_lshlrev_b32 v22, 2, v29 // (op 14305) 

v_lshrrev_b32 v28, 3, v22 // (op 14305) 

s_nop 0
v_readfirstlane_b32 s0, v28 // (op 14305) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 13437) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13437) 

v_lshrrev_b64 v[28:29], 3, s[10:11] // (op 13437) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 13437) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 13437) 

s_lshr_b64 s[6:7], s[10:11], 3 // (op 13437) 

s_mul_i32 s11, 3, s7 // (op 13437) most significant: low of low * high
s_mul_i32 s10, 3, s6 // (op 13437) least significant
s_mul_hi_u32 s1, 3, s6 // (op 13437) most significant: high of low * low
s_add_u32 s11, s11, s1 // (op 13437) most significant: sum


v_mov_b32 v22, s10 // (op 13437) 
v_mov_b32 v34, s11 // (op 13437) 
v_add_co_u32 v50, s[6:7], v24, v22 // (op 13437) least significant half
v_addc_co_u32 v51, s[6:7], v25, v34, s[6:7] // (op 13437) most significant half




s_add_u32 s1, 57344, s0 // (op 13437) 
s_mov_b32 m0, s1 // (op 13437) reg expression
buffer_load_dwordx4 v50, s[48:51], 0 offen offset: 0 lds // (op 13437) Load value direct to lds
v_add_co_u32 v50, s[6:7], v50, v28 // (op 13437) least significant half
v_addc_co_u32 v51, s[6:7], 0, v51, s[6:7] // (op 13437) most significant half

s_add_u32 m0, m0, 4096 // (op 13437) 
buffer_load_dwordx4 v50, s[48:51], 0 offen offset: 0 lds // (op 13437) Load value direct to lds






v_mov_b64 v[24:25], s[28:29] // (op 13009) convert

s_lshl_b32 s0, s28, 2 // (op 13009) 
s_mul_i32 s1, 3, s0 // (op 13009) 

v_add_u32 v22, v32, s1 // (op 13009) 

buffer_load_ubyte v28, v22, s[56:59], 0 offen offset: 0 // (op 13009) Load value
v_add_u32 v22, v22, v24 // (op 13009) 
buffer_load_ubyte v29, v22, s[56:59], 0 offen offset: 0 // (op 13009) Load value



v_and_b32 v22, -4, v1 // (op 13946) 
v_sub_u32 v32, v1, v22 // (op 13946) 

v_lshrrev_b32 v22, 2, v1 // (op 13946) 
v_lshl_add_u32 v24, v22, 3, v32 // (op 13946) 





v_add_u32 v22, 65536, v24 // (op 13012) 
s_waitcnt vmcnt(29)
ds_write_b8 v22, v38 // (op 13012) Store local data 

v_add_u32 v22, 65540, v24 // (op 13012) 
s_waitcnt vmcnt(28)
ds_write_b8 v22, v39 // (op 13012) Store local data 


v_and_b32 v22, -128, v1 // (op 13888) 
v_sub_u32 v32, v1, v22 // (op 13888) 

v_lshrrev_b32 v22, 7, v1 // (op 13888) 
v_lshl_add_u32 v24, v22, 8, v32 // (op 13888) 





v_add_u32 v22, 66048, v24 // (op 13014) 
s_waitcnt vmcnt(25)
ds_write_b8 v22, v40 // (op 13014) Store local data 

v_add_u32 v22, 66176, v24 // (op 13014) 
s_waitcnt vmcnt(24)
ds_write_b8 v22, v41 // (op 13014) Store local data 


s_barrier  // (op 14645) 



ds_read_b128 v[50:53], v4 // (op 13057) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 14680) 



v_add_u32 v22, 65536, v5 // (op 13097) 
ds_read_u8 v32, v22 // (op 13097) Load local data 




ds_read_b128 v[54:57], v4 offset:2048 // (op 13062) Load local data 



v_add_u32 v22, 65664, v5 // (op 13102) 
ds_read_u8 v24, v22 // (op 13102) Load local data 




ds_read_b128 v[58:61], v4 offset:4096 // (op 13067) Load local data 



v_add_u32 v22, 65792, v5 // (op 13107) 
ds_read_u8 v25, v22 // (op 13107) Load local data 




ds_read_b128 v[62:65], v4 offset:6144 // (op 13072) Load local data 



v_add_u32 v22, 65920, v5 // (op 13112) 
ds_read_u8 v34, v22 // (op 13112) Load local data 

s_barrier  // (op 14629) 



ds_read_b64_tr_b4 v[66:67], v3 offset:8192 // (op 13037) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v3 offset:9216 // (op 13037) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 14662) 


v_add_u32 v22, 66048, v0 // (op 13077) 
ds_read_u8 v35, v22 // (op 13077) Load local data 




ds_read_b64_tr_b4 v[70:71], v3 offset:8208 // (op 13042) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v3 offset:9232 // (op 13042) Transpose load local data 


v_add_u32 v22, 66080, v0 // (op 13082) 
ds_read_u8 v74, v22 // (op 13082) Load local data 




ds_read_b64_tr_b4 v[76:77], v3 offset:8224 // (op 13047) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v3 offset:9248 // (op 13047) Transpose load local data 


v_add_u32 v22, 66112, v0 // (op 13087) 
ds_read_u8 v75, v22 // (op 13087) Load local data 




ds_read_b64_tr_b4 v[80:81], v3 offset:8240 // (op 13052) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v3 offset:9264 // (op 13052) Transpose load local data 


v_add_u32 v22, 66144, v0 // (op 13092) 
ds_read_u8 v84, v22 // (op 13092) Load local data 

v_and_b32 v22, -4, v1 // (op 13745) 
v_sub_u32 v85, v1, v22 // (op 13745) 

v_lshlrev_b32 v22, 5, v85 // (op 13745) 

v_lshrrev_b32 v85, 2, v1 // (op 13745) 
v_lshl_add_u32 v86, v85, 7, v22 // (op 13745) 


v_lshlrev_b32 v22, 2, v86 // (op 13745) 

v_lshrrev_b32 v85, 3, v22 // (op 13745) 

s_nop 0
v_readfirstlane_b32 s0, v85 // (op 13745) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 13802) 
v_sub_u32 v85, v1, v22 // (op 13802) 

v_lshlrev_b32 v22, 5, v85 // (op 13802) 

v_lshrrev_b32 v85, 2, v1 // (op 13802) 
v_lshl_add_u32 v86, v85, 7, v22 // (op 13802) 


v_lshlrev_b32 v22, 2, v86 // (op 13802) 

v_lshrrev_b32 v85, 3, v22 // (op 13802) 

s_nop 0
v_readfirstlane_b32 s1, v85 // (op 13802) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -128, v1 // (op 13859) 
v_sub_u32 v85, v1, v22 // (op 13859) 

v_lshrrev_b32 v22, 7, v1 // (op 13859) 
v_lshl_add_u32 v86, v22, 8, v85 // (op 13859) 


v_and_b32 v22, -4, v1 // (op 13917) 
v_sub_u32 v85, v1, v22 // (op 13917) 

v_lshrrev_b32 v22, 2, v1 // (op 13917) 
v_lshl_add_u32 v87, v22, 3, v85 // (op 13917) 


v_and_b32 v22, -4, v1 // (op 13998) 
v_sub_u32 v85, v1, v22 // (op 13998) 

v_lshlrev_b32 v22, 5, v85 // (op 13998) 

v_lshrrev_b32 v85, 2, v1 // (op 13998) 
v_lshl_add_u32 v88, v85, 7, v22 // (op 13998) 


v_lshlrev_b32 v22, 2, v88 // (op 13998) 

v_lshrrev_b32 v85, 3, v22 // (op 13998) 

s_nop 0
v_readfirstlane_b32 s6, v85 // (op 13998) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 14042) 
v_sub_u32 v85, v1, v22 // (op 14042) 

v_lshlrev_b32 v22, 5, v85 // (op 14042) 

v_lshrrev_b32 v85, 2, v1 // (op 14042) 
v_lshl_add_u32 v88, v85, 7, v22 // (op 14042) 


v_lshlrev_b32 v22, 2, v88 // (op 14042) 

v_lshrrev_b32 v85, 3, v22 // (op 14042) 

s_nop 0
v_readfirstlane_b32 s7, v85 // (op 14042) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -128, v1 // (op 14086) 
v_sub_u32 v85, v1, v22 // (op 14086) 

v_lshrrev_b32 v22, 7, v1 // (op 14086) 
v_lshl_add_u32 v88, v22, 8, v85 // (op 14086) 


v_and_b32 v22, -4, v1 // (op 14115) 
v_sub_u32 v85, v1, v22 // (op 14115) 

v_lshrrev_b32 v22, 2, v1 // (op 14115) 
v_lshl_add_u32 v89, v22, 3, v85 // (op 14115) 


v_and_b32 v22, -4, v1 // (op 14144) 
v_sub_u32 v85, v1, v22 // (op 14144) 

v_lshlrev_b32 v22, 5, v85 // (op 14144) 

v_lshrrev_b32 v85, 2, v1 // (op 14144) 
v_lshl_add_u32 v90, v85, 7, v22 // (op 14144) 


v_lshlrev_b32 v22, 2, v90 // (op 14144) 

v_lshrrev_b32 v85, 3, v22 // (op 14144) 

s_nop 0
v_readfirstlane_b32 s10, v85 // (op 14144) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 14188) 
v_sub_u32 v85, v1, v22 // (op 14188) 

v_lshlrev_b32 v22, 5, v85 // (op 14188) 

v_lshrrev_b32 v85, 2, v1 // (op 14188) 
v_lshl_add_u32 v90, v85, 7, v22 // (op 14188) 


v_lshlrev_b32 v22, 2, v90 // (op 14188) 

v_lshrrev_b32 v85, 3, v22 // (op 14188) 

s_nop 0
v_readfirstlane_b32 s11, v85 // (op 14188) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -128, v1 // (op 14232) 
v_sub_u32 v85, v1, v22 // (op 14232) 

v_lshrrev_b32 v22, 7, v1 // (op 14232) 
v_lshl_add_u32 v90, v22, 8, v85 // (op 14232) 


v_and_b32 v22, -4, v1 // (op 14261) 
v_sub_u32 v85, v1, v22 // (op 14261) 

v_lshrrev_b32 v22, 2, v1 // (op 14261) 
v_lshl_add_u32 v91, v22, 3, v85 // (op 14261) 


v_and_b32 v22, -4, v1 // (op 14290) 
v_sub_u32 v85, v1, v22 // (op 14290) 

v_lshlrev_b32 v22, 5, v85 // (op 14290) 

v_lshrrev_b32 v85, 2, v1 // (op 14290) 
v_lshl_add_u32 v92, v85, 7, v22 // (op 14290) 


v_lshlrev_b32 v22, 2, v92 // (op 14290) 

v_lshrrev_b32 v85, 3, v22 // (op 14290) 

s_nop 0
v_readfirstlane_b32 s14, v85 // (op 14290) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 14334) 
v_sub_u32 v85, v1, v22 // (op 14334) 

v_lshlrev_b32 v22, 5, v85 // (op 14334) 

v_lshrrev_b32 v85, 2, v1 // (op 14334) 
v_lshl_add_u32 v92, v85, 7, v22 // (op 14334) 


v_lshlrev_b32 v22, 2, v92 // (op 14334) 

v_lshrrev_b32 v85, 3, v22 // (op 14334) 

s_nop 0
v_readfirstlane_b32 s15, v85 // (op 14334) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -128, v1 // (op 14378) 
v_sub_u32 v85, v1, v22 // (op 14378) 

v_lshrrev_b32 v22, 7, v1 // (op 14378) 
v_lshl_add_u32 v92, v22, 8, v85 // (op 14378) 


v_and_b32 v22, -4, v1 // (op 14407) 
v_sub_u32 v85, v1, v22 // (op 14407) 

v_lshrrev_b32 v22, 2, v1 // (op 14407) 
v_lshl_add_u32 v93, v22, 3, v85 // (op 14407) 


s_mov_b32 s18, 0 // (op 2901) call()
s_ashr_i64 s[38:39], s[22:23], 7 // (op 2900) 
s_mov_b32 s19, s38 // (op 2900) convert

s_and_b32 s38, s19, 4294967292 // (op 2900) 

s_cmp_lt_u32 s18, s38 // (op 2900) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_2900 // (op 2900) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_2900 if false)
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopTop_KLoop_2900:
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[50:53], a[0:3], v35, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 90) 
s_barrier  // (op 14648) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[54:57], a[4:7], v35, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 526) 
s_add_u32 s18, s18, 4 // (op 2902) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[58:61], a[8:11], v35, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 824) 
s_lshl_b64 s[38:39], s[24:25], 6 // (op 13369) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[62:65], a[12:15], v35, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1122) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13369) 

v_lshrrev_b64 v[94:95], 3, s[64:65] // (op 13369) 


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[50:53], a[16:19], v74, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1925) 
v_mov_b64 v[96:97], v[30:31] // (op 13369) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[54:57], a[20:23], v74, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1833) 
v_mov_b64 v[98:99], v[96:97] // (op 13369) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[58:61], a[24:27], v74, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1741) 
s_mov_b32 m0, s1 // (op 13369) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[62:65], a[28:31], v74, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1649) 
s_mov_b32 s19, 256 // (op 13369) call()
buffer_load_dwordx4 v98, s[52:55], s19 offen offset: 0 lds // (op 13369) Load value direct to lds
v_add_co_u32 v98, s[38:39], v98, v94 // (op 13369) least significant half
v_addc_co_u32 v99, s[38:39], 0, v99, s[38:39] // (op 13369) most significant half

s_add_u32 m0, m0, 4096 // (op 13369) 
buffer_load_dwordx4 v98, s[52:55], s19 offen offset: 0 lds // (op 13369) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[50:53], a[32:35], v75, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3215) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[54:57], a[36:39], v75, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3123) 
v_mov_b64 v[94:95], s[34:35] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[58:61], a[40:43], v75, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3031) 
v_mov_b32 v22, v23 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[62:65], a[44:47], v75, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2939) 
v_mov_b32 v85, v22 // (op 25) 
buffer_load_ubyte v38, v85, s[60:63], 0 offen offset:16 // (op 25) Load value
v_add_u32 v85, v85, v94 // (op 25) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[50:53], a[48:51], v84, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4505) 
buffer_load_ubyte v39, v85, s[60:63], 0 offen offset:16 // (op 25) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[54:57], a[52:55], v84, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4413) 
s_barrier  // (op 14632) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[58:61], a[56:59], v84, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4321) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 13364) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[62:65], a[60:63], v84, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4229) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13364) 

v_lshrrev_b64 v[94:95], 3, s[64:65] // (op 13364) 


v_mov_b64 v[98:99], v[26:27] // (op 13364) 
s_lshl_b64 s[38:39], s[12:13], 7 // (op 13364) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13364) 

s_lshr_b64 s[38:39], s[64:65], 3 // (op 13364) 

s_lshl_b64 s[64:65], s[38:39], 2 // (op 13364) 

v_mov_b32 v85, s64 // (op 13364) 
v_mov_b32 v100, s65 // (op 13364) 
v_add_co_u32 v102, s[38:39], v98, v85 // (op 13364) least significant half
v_addc_co_u32 v103, s[38:39], v99, v100, s[38:39] // (op 13364) most significant half




s_add_u32 s19, 8192, s0 // (op 13364) 
s_mov_b32 m0, s19 // (op 13364) reg expression
buffer_load_dwordx4 v102, s[48:51], 0 offen offset: 0 lds // (op 13364) Load value direct to lds
v_add_co_u32 v102, s[38:39], v102, v94 // (op 13364) least significant half
v_addc_co_u32 v103, s[38:39], 0, v103, s[38:39] // (op 13364) most significant half

s_add_u32 m0, m0, 4096 // (op 13364) 
buffer_load_dwordx4 v102, s[48:51], 0 offen offset: 0 lds // (op 13364) Load value direct to lds




v_mov_b64 v[94:95], s[28:29] // (op 17) convert

v_mov_b32 v85, v33 // (op 17) 
s_lshl_b32 s19, s28, 4 // (op 17) 
v_add_u32 v100, v85, s19 // (op 17) 

buffer_load_ubyte v40, v100, s[56:59], 0 offen offset: 0 // (op 17) Load value
v_add_u32 v100, v100, v94 // (op 17) 
buffer_load_ubyte v41, v100, s[56:59], 0 offen offset: 0 // (op 17) Load value


s_barrier  // (op 14720) 



v_add_u32 v94, 66560, v89 // (op 259) 
s_waitcnt vmcnt(29)
ds_write_b8 v94, v42 // (op 259) Store local data 

v_add_u32 v94, 66564, v89 // (op 259) 
s_waitcnt vmcnt(28)
ds_write_b8 v94, v43 // (op 259) Store local data 

s_barrier  // (op 14714) 



v_add_u32 v94, 67072, v88 // (op 258) 
s_waitcnt vmcnt(25)
ds_write_b8 v94, v44 // (op 258) Store local data 

v_add_u32 v94, 67200, v88 // (op 258) 
s_waitcnt vmcnt(24)
ds_write_b8 v94, v45 // (op 258) Store local data 

s_barrier  // (op 14703) 



ds_read_b128 v[100:103], v8 offset:16384 // (op 251) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 14717) 



v_add_u32 v94, 66560, v9 // (op 255) 
ds_read_u8 v95, v94 // (op 255) Load local data 




ds_read_b128 v[104:107], v8 offset:18432 // (op 533) Load local data 



v_add_u32 v94, 66688, v9 // (op 537) 
ds_read_u8 v74, v94 // (op 537) Load local data 




ds_read_b128 v[70:73], v8 offset:20480 // (op 831) Load local data 



v_add_u32 v94, 66816, v9 // (op 835) 
ds_read_u8 v75, v94 // (op 835) Load local data 




ds_read_b128 v[76:79], v8 offset:22528 // (op 1129) Load local data 



v_add_u32 v94, 66944, v9 // (op 1133) 
ds_read_u8 v108, v94 // (op 1133) Load local data 

s_barrier  // (op 14697) 
s_barrier  // (op 14829) 



ds_read_b64_tr_b4 v[66:67], v7 offset:24576 // (op 249) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v7 offset:25600 // (op 249) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 14710) 


v_add_u32 v94, 67072, v6 // (op 253) 
ds_read_u8 v35, v94 // (op 253) Load local data 




ds_read_b64_tr_b4 v[110:111], v7 offset:24592 // (op 1654) Transpose load local data 
ds_read_b64_tr_b4 v[112:113], v7 offset:25616 // (op 1654) Transpose load local data 


v_add_u32 v94, 67104, v6 // (op 1657) 
ds_read_u8 v24, v94 // (op 1657) Load local data 




ds_read_b64_tr_b4 v[114:115], v7 offset:24608 // (op 2944) Transpose load local data 
ds_read_b64_tr_b4 v[116:117], v7 offset:25632 // (op 2944) Transpose load local data 


v_add_u32 v94, 67136, v6 // (op 2947) 
ds_read_u8 v25, v94 // (op 2947) Load local data 




ds_read_b64_tr_b4 v[118:119], v7 offset:24624 // (op 4234) Transpose load local data 
ds_read_b64_tr_b4 v[120:121], v7 offset:25648 // (op 4234) Transpose load local data 


v_add_u32 v94, 67168, v6 // (op 4237) 
ds_read_u8 v34, v94 // (op 4237) Load local data 

s_barrier  // (op 14887) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[100:103], a[0:3], v35, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 256) 
s_barrier  // (op 14707) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[104:107], a[4:7], v35, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 527) 
s_lshl_b64 s[38:39], s[24:25], 6 // (op 13392) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[70:73], a[8:11], v35, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 825) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13392) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[76:79], a[12:15], v35, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1123) 
v_lshrrev_b64 v[122:123], 3, s[64:65] // (op 13392) 


v_mov_b64 v[124:125], v[96:97] // (op 13392) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[110:113], v[100:103], a[16:19], v24, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1926) 
s_add_u32 s19, 16384, s7 // (op 13392) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[110:113], v[104:107], a[20:23], v24, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1834) 
s_mov_b32 m0, s19 // (op 13392) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:113], v[70:73], a[24:27], v24, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1742) 
s_mov_b32 s38, 320 // (op 13392) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[110:113], v[76:79], a[28:31], v24, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1650) 
buffer_load_dwordx4 v124, s[52:55], s38 offen offset: 0 lds // (op 13392) Load value direct to lds

v_add_co_u32 v124, s[64:65], v124, v122 // (op 13392) least significant half
v_addc_co_u32 v125, s[64:65], 0, v125, s[64:65] // (op 13392) most significant half

s_add_u32 m0, m0, 4096 // (op 13392) 
buffer_load_dwordx4 v124, s[52:55], s38 offen offset: 0 lds // (op 13392) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[114:117], v[100:103], a[32:35], v25, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3216) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[114:117], v[104:107], a[36:39], v25, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3124) 
v_mov_b64 v[112:113], s[34:35] // (op 262) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[114:117], v[70:73], a[40:43], v25, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3032) 
v_mov_b32 v94, v22 // (op 262) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[114:117], v[76:79], a[44:47], v25, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2940) 
buffer_load_ubyte v42, v94, s[60:63], 0 offen offset:20 // (op 262) Load value

v_add_u32 v94, v94, v112 // (op 262) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[118:121], v[100:103], a[48:51], v34, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4506) 
buffer_load_ubyte v43, v94, s[60:63], 0 offen offset:20 // (op 262) Load value




s_barrier  // (op 14700) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[118:121], v[104:107], a[52:55], v34, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4414) 


s_lshl_b64 s[38:39], s[12:13], 6 // (op 13387) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[118:121], v[70:73], a[56:59], v34, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4322) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13387) 

v_lshrrev_b64 v[94:95], 3, s[64:65] // (op 13387) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[118:121], v[76:79], a[60:63], v34, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4230) 
s_lshl_b64 s[38:39], s[12:13], 7 // (op 13387) 


s_lshl_b64 s[64:65], s[38:39], 2 // (op 13387) 

s_lshr_b64 s[38:39], s[64:65], 3 // (op 13387) 

s_mul_i32 s65, 5, s39 // (op 13387) most significant: low of low * high
s_mul_i32 s64, 5, s38 // (op 13387) least significant
s_mul_hi_u32 s19, 5, s38 // (op 13387) most significant: high of low * low
s_add_u32 s65, s65, s19 // (op 13387) most significant: sum


v_mov_b32 v100, s64 // (op 13387) 
v_mov_b32 v101, s65 // (op 13387) 
v_add_co_u32 v102, s[38:39], v98, v100 // (op 13387) least significant half
v_addc_co_u32 v103, s[38:39], v99, v101, s[38:39] // (op 13387) most significant half




s_add_u32 s19, 24576, s6 // (op 13387) 
s_mov_b32 m0, s19 // (op 13387) reg expression
buffer_load_dwordx4 v102, s[48:51], 0 offen offset: 0 lds // (op 13387) Load value direct to lds
v_add_co_u32 v102, s[38:39], v102, v94 // (op 13387) least significant half
v_addc_co_u32 v103, s[38:39], 0, v103, s[38:39] // (op 13387) most significant half

s_add_u32 m0, m0, 4096 // (op 13387) 
buffer_load_dwordx4 v102, s[48:51], 0 offen offset: 0 lds // (op 13387) Load value direct to lds




v_mov_b64 v[94:95], s[28:29] // (op 261) convert

s_lshl_b32 s19, s28, 2 // (op 261) 
s_mul_i32 s38, 5, s19 // (op 261) 

v_add_u32 v100, v85, s38 // (op 261) 

buffer_load_ubyte v44, v100, s[56:59], 0 offen offset: 0 // (op 261) Load value
v_add_u32 v100, v100, v94 // (op 261) 
buffer_load_ubyte v45, v100, s[56:59], 0 offen offset: 0 // (op 261) Load value


s_barrier  // (op 14746) 



v_add_u32 v94, 66560, v91 // (op 299) 
s_waitcnt vmcnt(29)
ds_write_b8 v94, v46 // (op 299) Store local data 

v_add_u32 v94, 66564, v91 // (op 299) 
s_waitcnt vmcnt(28)
ds_write_b8 v94, v47 // (op 299) Store local data 

s_barrier  // (op 14740) 



v_add_u32 v94, 67072, v90 // (op 298) 
s_waitcnt vmcnt(25)
ds_write_b8 v94, v48 // (op 298) Store local data 

v_add_u32 v94, 67200, v90 // (op 298) 
s_waitcnt vmcnt(24)
ds_write_b8 v94, v49 // (op 298) Store local data 

s_barrier  // (op 14729) 



ds_read_b128 v[50:53], v12 offset:32768 // (op 291) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 14743) 



v_add_u32 v94, 66560, v13 // (op 295) 
ds_read_u8 v32, v94 // (op 295) Load local data 




ds_read_b128 v[54:57], v12 offset:34816 // (op 541) Load local data 



v_add_u32 v94, 66688, v13 // (op 545) 
ds_read_u8 v24, v94 // (op 545) Load local data 




ds_read_b128 v[58:61], v12 offset:36864 // (op 839) Load local data 



v_add_u32 v94, 66816, v13 // (op 843) 
ds_read_u8 v25, v94 // (op 843) Load local data 




ds_read_b128 v[62:65], v12 offset:38912 // (op 1137) Load local data 



v_add_u32 v94, 66944, v13 // (op 1141) 
ds_read_u8 v84, v94 // (op 1141) Load local data 

s_barrier  // (op 14723) 
s_barrier  // (op 14835) 



ds_read_b64_tr_b4 v[66:67], v11 offset:40960 // (op 289) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v11 offset:41984 // (op 289) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 14736) 


v_add_u32 v94, 67072, v10 // (op 293) 
ds_read_u8 v35, v94 // (op 293) Load local data 




ds_read_b64_tr_b4 v[80:81], v11 offset:40976 // (op 1660) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v11 offset:42000 // (op 1660) Transpose load local data 


v_add_u32 v94, 67104, v10 // (op 1663) 
ds_read_u8 v74, v94 // (op 1663) Load local data 




ds_read_b64_tr_b4 v[100:101], v11 offset:40992 // (op 2950) Transpose load local data 
ds_read_b64_tr_b4 v[102:103], v11 offset:42016 // (op 2950) Transpose load local data 


v_add_u32 v94, 67136, v10 // (op 2953) 
ds_read_u8 v75, v94 // (op 2953) Load local data 




ds_read_b64_tr_b4 v[104:105], v11 offset:41008 // (op 4240) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v11 offset:42032 // (op 4240) Transpose load local data 


v_add_u32 v94, 67168, v10 // (op 4243) 
ds_read_u8 v95, v94 // (op 4243) Load local data 

s_barrier  // (op 14893) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[50:53], a[0:3], v35, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 296) 
s_barrier  // (op 14733) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[54:57], a[4:7], v35, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 528) 
s_lshl_b64 s[38:39], s[24:25], 6 // (op 13402) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[58:61], a[8:11], v35, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 826) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13402) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[62:65], a[12:15], v35, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1124) 
v_lshrrev_b64 v[108:109], 3, s[64:65] // (op 13402) 


v_mov_b64 v[110:111], v[96:97] // (op 13402) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[80:83], v[50:53], a[16:19], v74, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1927) 
s_add_u32 s19, 32768, s11 // (op 13402) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[80:83], v[54:57], a[20:23], v74, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1835) 
s_mov_b32 m0, s19 // (op 13402) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[80:83], v[58:61], a[24:27], v74, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1743) 
s_mov_b32 s38, 384 // (op 13402) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[62:65], a[28:31], v74, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1651) 
buffer_load_dwordx4 v110, s[52:55], s38 offen offset: 0 lds // (op 13402) Load value direct to lds
v_add_co_u32 v110, s[64:65], v110, v108 // (op 13402) least significant half
v_addc_co_u32 v111, s[64:65], 0, v111, s[64:65] // (op 13402) most significant half

s_add_u32 m0, m0, 4096 // (op 13402) 
buffer_load_dwordx4 v110, s[52:55], s38 offen offset: 0 lds // (op 13402) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[100:103], v[50:53], a[32:35], v75, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3217) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[100:103], v[54:57], a[36:39], v75, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3125) 
v_mov_b64 v[108:109], s[34:35] // (op 302) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:103], v[58:61], a[40:43], v75, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3033) 
v_mov_b32 v94, v22 // (op 302) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:103], v[62:65], a[44:47], v75, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2941) 
buffer_load_ubyte v46, v94, s[60:63], 0 offen offset:24 // (op 302) Load value

v_add_u32 v94, v94, v108 // (op 302) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:107], v[50:53], a[48:51], v95, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4507) 
buffer_load_ubyte v47, v94, s[60:63], 0 offen offset:24 // (op 302) Load value


s_barrier  // (op 14726) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:107], v[54:57], a[52:55], v95, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4415) 

s_lshl_b64 s[38:39], s[12:13], 6 // (op 13397) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:107], v[58:61], a[56:59], v95, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4323) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13397) 

v_lshrrev_b64 v[100:101], 3, s[64:65] // (op 13397) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:107], v[62:65], a[60:63], v95, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4231) 
s_lshl_b64 s[38:39], s[12:13], 7 // (op 13397) 


s_lshl_b64 s[64:65], s[38:39], 2 // (op 13397) 

s_lshr_b64 s[38:39], s[64:65], 3 // (op 13397) 

s_mul_i32 s65, 6, s39 // (op 13397) most significant: low of low * high
s_mul_i32 s64, 6, s38 // (op 13397) least significant
s_mul_hi_u32 s19, 6, s38 // (op 13397) most significant: high of low * low
s_add_u32 s65, s65, s19 // (op 13397) most significant: sum


v_mov_b32 v94, s64 // (op 13397) 
v_mov_b32 v95, s65 // (op 13397) 
v_add_co_u32 v102, s[38:39], v98, v94 // (op 13397) least significant half
v_addc_co_u32 v103, s[38:39], v99, v95, s[38:39] // (op 13397) most significant half




s_add_u32 s19, 40960, s10 // (op 13397) 
s_mov_b32 m0, s19 // (op 13397) reg expression
buffer_load_dwordx4 v102, s[48:51], 0 offen offset: 0 lds // (op 13397) Load value direct to lds
v_add_co_u32 v102, s[38:39], v102, v100 // (op 13397) least significant half
v_addc_co_u32 v103, s[38:39], 0, v103, s[38:39] // (op 13397) most significant half

s_add_u32 m0, m0, 4096 // (op 13397) 
buffer_load_dwordx4 v102, s[48:51], 0 offen offset: 0 lds // (op 13397) Load value direct to lds




v_mov_b64 v[94:95], s[28:29] // (op 301) convert

s_lshl_b32 s19, s28, 2 // (op 301) 
s_mul_i32 s38, 6, s19 // (op 301) 

v_add_u32 v100, v85, s38 // (op 301) 

buffer_load_ubyte v48, v100, s[56:59], 0 offen offset: 0 // (op 301) Load value
v_add_u32 v100, v100, v94 // (op 301) 
buffer_load_ubyte v49, v100, s[56:59], 0 offen offset: 0 // (op 301) Load value


s_barrier  // (op 14772) 



v_add_u32 v94, 66560, v93 // (op 339) 
s_waitcnt vmcnt(29)
ds_write_b8 v94, v36 // (op 339) Store local data 

v_add_u32 v94, 66564, v93 // (op 339) 
s_waitcnt vmcnt(28)
ds_write_b8 v94, v37 // (op 339) Store local data 

s_barrier  // (op 14766) 



v_add_u32 v94, 67072, v92 // (op 338) 
s_waitcnt vmcnt(25)
ds_write_b8 v94, v28 // (op 338) Store local data 

v_add_u32 v94, 67200, v92 // (op 338) 
s_waitcnt vmcnt(24)
ds_write_b8 v94, v29 // (op 338) Store local data 

s_barrier  // (op 14755) 



ds_read_b128 v[70:73], v16 offset:49152 // (op 331) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 14769) 



v_add_u32 v94, 66560, v17 // (op 335) 
ds_read_u8 v34, v94 // (op 335) Load local data 




ds_read_b128 v[76:79], v16 offset:51200 // (op 549) Load local data 



v_add_u32 v94, 66688, v17 // (op 553) 
ds_read_u8 v74, v94 // (op 553) Load local data 




ds_read_b128 v[80:83], v16 offset:53248 // (op 847) Load local data 



v_add_u32 v94, 66816, v17 // (op 851) 
ds_read_u8 v75, v94 // (op 851) Load local data 




ds_read_b128 v[100:103], v16 offset:55296 // (op 1145) Load local data 



v_add_u32 v94, 66944, v17 // (op 1149) 
ds_read_u8 v95, v94 // (op 1149) Load local data 

s_barrier  // (op 14749) 
s_barrier  // (op 14841) 



ds_read_b64_tr_b4 v[66:67], v15 offset:57344 // (op 329) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v15 offset:58368 // (op 329) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 14762) 


v_add_u32 v94, 67072, v14 // (op 333) 
ds_read_u8 v35, v94 // (op 333) Load local data 




ds_read_b64_tr_b4 v[104:105], v15 offset:57360 // (op 1666) Transpose load local data 
ds_read_b64_tr_b4 v[106:107], v15 offset:58384 // (op 1666) Transpose load local data 


v_add_u32 v94, 67104, v14 // (op 1669) 
ds_read_u8 v24, v94 // (op 1669) Load local data 




ds_read_b64_tr_b4 v[108:109], v15 offset:57376 // (op 2956) Transpose load local data 
ds_read_b64_tr_b4 v[110:111], v15 offset:58400 // (op 2956) Transpose load local data 


v_add_u32 v94, 67136, v14 // (op 2959) 
ds_read_u8 v25, v94 // (op 2959) Load local data 




ds_read_b64_tr_b4 v[112:113], v15 offset:57392 // (op 4246) Transpose load local data 
ds_read_b64_tr_b4 v[114:115], v15 offset:58416 // (op 4246) Transpose load local data 


v_add_u32 v94, 67168, v14 // (op 4249) 
ds_read_u8 v84, v94 // (op 4249) Load local data 

s_barrier  // (op 14899) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[70:73], a[0:3], v35, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 336) 
s_barrier  // (op 14759) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[76:79], a[4:7], v35, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 529) 
s_lshl_b64 s[38:39], s[24:25], 6 // (op 13412) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[80:83], a[8:11], v35, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 827) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13412) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[100:103], a[12:15], v35, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1125) 
v_lshrrev_b64 v[116:117], 3, s[64:65] // (op 13412) 


v_mov_b64 v[118:119], v[96:97] // (op 13412) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[70:73], a[16:19], v24, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1928) 
s_add_u32 s19, 49152, s15 // (op 13412) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[76:79], a[20:23], v24, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1836) 
s_mov_b32 m0, s19 // (op 13412) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:107], v[80:83], a[24:27], v24, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1744) 
s_mov_b32 s38, 448 // (op 13412) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:107], v[100:103], a[28:31], v24, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1652) 
buffer_load_dwordx4 v118, s[52:55], s38 offen offset: 0 lds // (op 13412) Load value direct to lds

v_add_co_u32 v118, s[64:65], v118, v116 // (op 13412) least significant half
v_addc_co_u32 v119, s[64:65], 0, v119, s[64:65] // (op 13412) most significant half

s_add_u32 m0, m0, 4096 // (op 13412) 
buffer_load_dwordx4 v118, s[52:55], s38 offen offset: 0 lds // (op 13412) Load value direct to lds




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[108:111], v[70:73], a[32:35], v25, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3218) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:111], v[76:79], a[36:39], v25, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3126) 
v_add_co_u32 v30, vcc, 256, v30 // (op 13566) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[80:83], a[40:43], v25, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3034) 
v_addc_co_u32 v31, vcc, 0, v31, vcc // (op 13566) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[100:103], a[44:47], v25, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2942) 
v_mov_b64 v[96:97], s[34:35] // (op 342) convert


v_mov_b32 v94, v22 // (op 342) 
buffer_load_ubyte v36, v94, s[60:63], 0 offen offset:28 // (op 342) Load value
v_add_u32 v94, v94, v96 // (op 342) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[70:73], a[48:51], v84, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4508) 
buffer_load_ubyte v37, v94, s[60:63], 0 offen offset:28 // (op 342) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[76:79], a[52:55], v84, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4416) 
s_barrier  // (op 14752) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[112:115], v[80:83], a[56:59], v84, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4324) 
v_add_u32 v23, 16, v23 // (op 13694) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 13407) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[112:115], v[100:103], a[60:63], v84, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4232) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13407) 




v_lshrrev_b64 v[94:95], 3, s[64:65] // (op 13407) 


s_lshl_b64 s[38:39], s[12:13], 7 // (op 13407) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13407) 

s_lshr_b64 s[38:39], s[64:65], 3 // (op 13407) 

s_mul_i32 s65, 7, s39 // (op 13407) most significant: low of low * high
s_mul_i32 s64, 7, s38 // (op 13407) least significant
s_mul_hi_u32 s19, 7, s38 // (op 13407) most significant: high of low * low
s_add_u32 s65, s65, s19 // (op 13407) most significant: sum


v_mov_b32 v22, s64 // (op 13407) 
v_mov_b32 v96, s65 // (op 13407) 
v_add_co_u32 v100, s[38:39], v98, v22 // (op 13407) least significant half
v_addc_co_u32 v101, s[38:39], v99, v96, s[38:39] // (op 13407) most significant half




s_add_u32 s19, 57344, s14 // (op 13407) 
s_mov_b32 m0, s19 // (op 13407) reg expression
buffer_load_dwordx4 v100, s[48:51], 0 offen offset: 0 lds // (op 13407) Load value direct to lds
v_add_co_u32 v100, s[38:39], v100, v94 // (op 13407) least significant half
v_addc_co_u32 v101, s[38:39], 0, v101, s[38:39] // (op 13407) most significant half

s_add_u32 m0, m0, 4096 // (op 13407) 
buffer_load_dwordx4 v100, s[48:51], 0 offen offset: 0 lds // (op 13407) Load value direct to lds





v_mov_b64 v[94:95], s[28:29] // (op 341) convert

s_lshl_b32 s19, s28, 2 // (op 341) 
s_mul_i32 s38, 7, s19 // (op 341) 

v_add_u32 v22, v85, s38 // (op 341) 

s_lshl_b64 s[38:39], s[12:13], 7 // (op 13502) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 13502) 

s_lshr_b64 s[38:39], s[64:65], 3 // (op 13502) 

s_lshl_b64 s[64:65], s[38:39], 2 // (op 13502) 

v_mov_b32 v96, s64 // (op 13502) 
v_mov_b32 v97, s65 // (op 13502) 
v_add_co_u32 v26, s[38:39], v26, v96 // (op 13502) least significant half
v_addc_co_u32 v27, s[38:39], v27, v97, s[38:39] // (op 13502) most significant half




buffer_load_ubyte v28, v22, s[56:59], 0 offen offset: 0 // (op 341) Load value
v_add_u32 v22, v22, v94 // (op 341) 
buffer_load_ubyte v29, v22, s[56:59], 0 offen offset: 0 // (op 341) Load value



s_barrier  // (op 14683) 
s_lshl_b32 s19, s28, 4 // (op 13630) 



v_add_u32 v33, v33, s19 // (op 13630) 

v_add_u32 v22, 65536, v87 // (op 30) 
s_waitcnt vmcnt(29)
ds_write_b8 v22, v38 // (op 30) Store local data 

v_add_u32 v22, 65540, v87 // (op 30) 
s_waitcnt vmcnt(28)
ds_write_b8 v22, v39 // (op 30) Store local data 

s_barrier  // (op 14665) 



v_add_u32 v22, 66048, v86 // (op 21) 
s_waitcnt vmcnt(25)
ds_write_b8 v22, v40 // (op 21) Store local data 

v_add_u32 v22, 66176, v86 // (op 21) 
s_waitcnt vmcnt(24)
ds_write_b8 v22, v41 // (op 21) Store local data 

s_barrier  // (op 14654) 



ds_read_b128 v[50:53], v4 // (op 77) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 14690) 



v_add_u32 v22, 65536, v5 // (op 87) 
ds_read_u8 v32, v22 // (op 87) Load local data 




ds_read_b128 v[54:57], v4 offset:2048 // (op 521) Load local data 



v_add_u32 v22, 65664, v5 // (op 525) 
ds_read_u8 v24, v22 // (op 525) Load local data 




ds_read_b128 v[58:61], v4 offset:4096 // (op 819) Load local data 



v_add_u32 v22, 65792, v5 // (op 823) 
ds_read_u8 v25, v22 // (op 823) Load local data 




ds_read_b128 v[62:65], v4 offset:6144 // (op 1117) Load local data 



v_add_u32 v22, 65920, v5 // (op 1121) 
ds_read_u8 v34, v22 // (op 1121) Load local data 

s_barrier  // (op 14638) 



ds_read_b64_tr_b4 v[66:67], v3 offset:8192 // (op 72) Transpose load local data 
ds_read_b64_tr_b4 v[68:69], v3 offset:9216 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 14672) 


v_add_u32 v22, 66048, v0 // (op 82) 
ds_read_u8 v35, v22 // (op 82) Load local data 




ds_read_b64_tr_b4 v[70:71], v3 offset:8208 // (op 1644) Transpose load local data 
ds_read_b64_tr_b4 v[72:73], v3 offset:9232 // (op 1644) Transpose load local data 


v_add_u32 v22, 66080, v0 // (op 1647) 
ds_read_u8 v74, v22 // (op 1647) Load local data 




ds_read_b64_tr_b4 v[76:77], v3 offset:8224 // (op 2934) Transpose load local data 
ds_read_b64_tr_b4 v[78:79], v3 offset:9248 // (op 2934) Transpose load local data 


v_add_u32 v22, 66112, v0 // (op 2937) 
ds_read_u8 v75, v22 // (op 2937) Load local data 




ds_read_b64_tr_b4 v[80:81], v3 offset:8240 // (op 4224) Transpose load local data 
ds_read_b64_tr_b4 v[82:83], v3 offset:9264 // (op 4224) Transpose load local data 


v_add_u32 v22, 66144, v0 // (op 4227) 
ds_read_u8 v84, v22 // (op 4227) Load local data 

s_ashr_i64 s[38:39], s[22:23], 7 // (op 2900) 
s_mov_b32 s19, s38 // (op 2900) convert

s_and_b32 s38, s19, 4294967292 // (op 2900) 

s_cmp_lt_u32 s18, s38 // (op 2900) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopTop_KLoop_2900 // (op 2900) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopTop_KLoop_2900 if true)
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopBottom_KLoop_2900:




























s_barrier  // (op 14853) 
v_and_b32 v22, -4, v1 // (op 13450) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 13450) 
v_sub_u32 v23, v1, v22 // (op 13450) 

s_lshr_b32 s6, s0, 2 // (op 13450) 

v_lshlrev_b32 v22, 5, v23 // (op 13450) 

v_lshl_add_u32 v23, s16, 7, v22 // (op 13450) 

v_mov_b32 v27, 0 // (op 13450) convert
v_mov_b32 v26, v23 // (op 13450) convert

v_lshrrev_b32 v22, 2, v1 // (op 13450) 
v_lshl_add_u32 v23, s6, 9, v22 // (op 13450) 


v_mul_lo_u32 v31, v23, s13 // (op 13450) most significant: low of low * high
v_mul_hi_u32 v22, v23, s12 // (op 13450) most significant: high of low * low
v_mul_lo_u32 v30, v23, s12 // (op 13450) least significant: low of low * low
v_add_u32 v31, v31, v22 // (op 13450) most significant: sum


v_add_co_u32 v32, s[0:1], v26, v30 // (op 13450) least significant half
v_addc_co_u32 v33, s[0:1], v27, v31, s[0:1] // (op 13450) most significant half

v_lshlrev_b64 v[32:33], 2, v[32:33] // (op 13450) 


v_lshrrev_b64 v[26:27], 3, v[32:33] // (op 13450) 

s_mul_i32 s1, s12, s23 // (op 13452) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 13452) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 13452) least significant
s_mul_hi_u32 s7, s12, s22 // (op 13452) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13452) most significant: sum
s_add_u32 s1, s1, s7 // (op 13452) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 13452) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 13452) 

s_mov_b32 s48, s20 // (op 13452) reg expression
s_mov_b32 s49, s21 // (op 13452) reg expression
s_mov_b32 s50, s0 // (op 13452) reg expression
s_mov_b32 s51, 131072 // (op 13452) call()

v_and_b32 v22, -4, v1 // (op 13532) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 13532) 
v_sub_u32 v23, v1, v22 // (op 13532) 

s_lshr_b32 s6, s0, 2 // (op 13532) 

v_lshlrev_b32 v22, 5, v23 // (op 13532) 

v_lshl_add_u32 v23, s6, 9, v22 // (op 13532) 


v_mov_b32 v31, 0 // (op 13532) convert
v_mov_b32 v30, v23 // (op 13532) convert

v_lshrrev_b32 v22, 2, v1 // (op 13532) 
v_lshl_add_u32 v23, s17, 7, v22 // (op 13532) 

v_mul_lo_u32 v33, v23, s25 // (op 13532) most significant: low of low * high
v_mul_hi_u32 v22, v23, s24 // (op 13532) most significant: high of low * low
v_mul_lo_u32 v32, v23, s24 // (op 13532) least significant: low of low * low
v_add_u32 v33, v33, v22 // (op 13532) most significant: sum


v_add_co_u32 v22, s[0:1], v30, v32 // (op 13532) least significant half
v_addc_co_u32 v23, s[0:1], v31, v33, s[0:1] // (op 13532) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 13532) 


v_lshrrev_b64 v[30:31], 3, v[22:23] // (op 13532) 

s_mul_i32 s1, s24, s9 // (op 13534) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 13534) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13534) least significant
s_mul_hi_u32 s7, s24, s8 // (op 13534) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13534) most significant: sum
s_add_u32 s1, s1, s7 // (op 13534) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 13534) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 13534) 

s_mov_b32 s52, s26 // (op 13534) reg expression
s_mov_b32 s53, s27 // (op 13534) reg expression
s_mov_b32 s54, s0 // (op 13534) reg expression
s_mov_b32 s55, 131072 // (op 13534) call()

v_and_b32 v22, -128, v1 // (op 13596) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 13596) 
v_sub_u32 v23, v1, v22 // (op 13596) 

s_lshr_b32 s6, s0, 2 // (op 13596) 

v_lshl_add_u32 v22, s16, 7, v23 // (op 13596) 

v_lshrrev_b32 v23, 7, v1 // (op 13596) 
v_lshlrev_b32 v24, 1, v23 // (op 13596) 

v_lshl_add_u32 v23, s6, 4, v24 // (op 13596) 


v_mul_lo_u32 v24, v23, s28 // (op 13596) 

v_add_u32 v23, v22, v24 // (op 13596) 


s_mul_i32 s1, s28, s33 // (op 13598) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 13598) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 13598) least significant
s_mul_hi_u32 s7, s28, s32 // (op 13598) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13598) most significant: sum
s_add_u32 s1, s1, s7 // (op 13598) most significant: sum


s_mov_b32 s56, s30 // (op 13598) reg expression
s_mov_b32 s57, s31 // (op 13598) reg expression
s_mov_b32 s58, s0 // (op 13598) reg expression
s_mov_b32 s59, 131072 // (op 13598) call()

v_and_b32 v22, -4, v1 // (op 13660) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 13660) 
v_sub_u32 v24, v1, v22 // (op 13660) 

s_lshr_b32 s6, s0, 2 // (op 13660) 

v_lshrrev_b32 v22, 2, v1 // (op 13660) 
v_lshl_add_u32 v32, s6, 4, v24 // (op 13660) 


v_lshlrev_b32 v24, 1, v22 // (op 13660) 

v_lshl_add_u32 v22, s17, 7, v24 // (op 13660) 

v_mul_lo_u32 v24, v22, s34 // (op 13660) 

v_add_u32 v22, v32, v24 // (op 13660) 


s_mul_i32 s1, s34, s9 // (op 13662) most significant: low of low * high
s_mul_i32 s6, s35, s8 // (op 13662) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 13662) least significant
s_mul_hi_u32 s7, s34, s8 // (op 13662) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13662) most significant: sum
s_add_u32 s1, s1, s7 // (op 13662) most significant: sum


s_mov_b32 s60, s36 // (op 13662) reg expression
s_mov_b32 s61, s37 // (op 13662) reg expression
s_mov_b32 s62, s0 // (op 13662) reg expression
s_mov_b32 s63, 131072 // (op 13662) call()

v_and_b32 v24, -4, v1 // (op 13758) 
v_sub_u32 v32, v1, v24 // (op 13758) 

v_lshlrev_b32 v24, 5, v32 // (op 13758) 

v_lshrrev_b32 v32, 2, v1 // (op 13758) 
v_lshl_add_u32 v33, v32, 7, v24 // (op 13758) 


v_lshlrev_b32 v24, 2, v33 // (op 13758) 

v_lshrrev_b32 v32, 3, v24 // (op 13758) 

s_nop 0
v_readfirstlane_b32 s0, v32 // (op 13758) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -4, v1 // (op 13815) 
v_sub_u32 v32, v1, v24 // (op 13815) 

v_lshlrev_b32 v24, 5, v32 // (op 13815) 

v_lshrrev_b32 v32, 2, v1 // (op 13815) 
v_lshl_add_u32 v33, v32, 7, v24 // (op 13815) 


v_lshlrev_b32 v24, 2, v33 // (op 13815) 

v_lshrrev_b32 v32, 3, v24 // (op 13815) 

s_nop 0
v_readfirstlane_b32 s1, v32 // (op 13815) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v24, -128, v1 // (op 13872) 
v_sub_u32 v32, v1, v24 // (op 13872) 

v_lshrrev_b32 v24, 7, v1 // (op 13872) 
v_lshl_add_u32 v33, v24, 8, v32 // (op 13872) 


v_and_b32 v24, -4, v1 // (op 13930) 
v_sub_u32 v32, v1, v24 // (op 13930) 

v_lshrrev_b32 v24, 2, v1 // (op 13930) 
v_lshl_add_u32 v38, v24, 3, v32 // (op 13930) 


s_ashr_i64 s[6:7], s[22:23], 7 // (op 4196) 
s_and_b32 s10, s6, 4294967292 // (op 4196) 

s_ashr_i64 s[6:7], s[22:23], 7 // (op 4195) 
s_mov_b32 s11, s6 // (op 4195) convert

s_cmp_lt_u32 s10, s11 // (op 4195) 


















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_4195 // (op 4195) Condition: Top (jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_4195 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_4195:
s_barrier  // (op 14635) 
s_add_u32 s10, s10, 1 // (op 4197) 

s_lshl_b64 s[6:7], s[12:13], 6 // (op 13374) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 13374) 

v_lshrrev_b64 v[24:25], 3, s[14:15] // (op 13374) 


v_mov_b64 v[28:29], v[26:27] // (op 13374) 
v_mov_b64 v[34:35], v[28:29] // (op 13374) reg expression
s_add_u32 s11, 8192, s0 // (op 13374) 
s_mov_b32 m0, s11 // (op 13374) reg expression
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 13374) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v24 // (op 13374) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 13374) most significant half

s_add_u32 m0, m0, 4096 // (op 13374) 
buffer_load_dwordx4 v34, s[48:51], 0 offen offset: 0 lds // (op 13374) Load value direct to lds




s_lshl_b64 s[6:7], s[12:13], 7 // (op 13440) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 13440) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 13440) 

v_mov_b32 v32, s6 // (op 13440) 
v_mov_b32 v24, s7 // (op 13440) 
v_add_co_u32 v26, s[14:15], v26, v32 // (op 13440) least significant half
v_addc_co_u32 v27, s[14:15], v27, v24, s[14:15] // (op 13440) most significant half




s_barrier  // (op 14651) 

s_lshl_b64 s[6:7], s[24:25], 6 // (op 13380) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 13380) 

v_lshrrev_b64 v[24:25], 3, s[14:15] // (op 13380) 


v_mov_b64 v[28:29], v[30:31] // (op 13380) 
v_mov_b64 v[34:35], v[28:29] // (op 13380) 
s_mov_b32 m0, s1 // (op 13380) reg expression
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 13380) Load value direct to lds
v_add_co_u32 v34, s[6:7], v34, v24 // (op 13380) least significant half
v_addc_co_u32 v35, s[6:7], 0, v35, s[6:7] // (op 13380) most significant half

s_add_u32 m0, m0, 4096 // (op 13380) 
buffer_load_dwordx4 v34, s[52:55], 0 offen offset: 0 lds // (op 13380) Load value direct to lds



v_add_co_u32 v30, s[6:7], 64, v30 // (op 13524) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 13524) most significant half


v_mov_b64 v[24:25], s[28:29] // (op 195) convert

v_mov_b32 v32, v23 // (op 195) 
v_mov_b32 v28, v32 // (op 195) reg expression
buffer_load_ubyte v34, v28, s[56:59], 0 offen offset: 0 // (op 195) Load value
v_add_u32 v28, v28, v24 // (op 195) 
buffer_load_ubyte v35, v28, s[56:59], 0 offen offset: 0 // (op 195) Load value



s_lshl_b32 s11, s28, 2 // (op 13588) 
v_add_u32 v23, v23, s11 // (op 13588) 


s_barrier  // (op 14668) 
v_mov_b64 v[24:25], s[34:35] // (op 196) convert




v_mov_b32 v32, v22 // (op 196) 
v_mov_b32 v28, v32 // (op 196) 
v_add_u32 v29, 66048, v33 // (op 192) 
s_waitcnt vmcnt(1)
ds_write_b8 v29, v34 // (op 192) Store local data 

v_add_u32 v29, 66176, v33 // (op 192) 
buffer_load_ubyte v36, v28, s[60:63], 0 offen offset: 0 // (op 196) Load value
v_add_u32 v28, v28, v24 // (op 196) 
buffer_load_ubyte v37, v28, s[60:63], 0 offen offset: 0 // (op 196) Load value



v_add_u32 v22, 4, v22 // (op 13652) 
s_waitcnt vmcnt(2)
ds_write_b8 v29, v35 // (op 192) Store local data 


s_barrier  // (op 14641) 



s_barrier  // (op 14686) 
ds_read_b64_tr_b4 v[40:41], v3 offset:8192 // (op 183) Transpose load local data 



v_add_u32 v32, 65536, v38 // (op 193) 
ds_read_b64_tr_b4 v[42:43], v3 offset:9216 // (op 183) Transpose load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v32, v36 // (op 193) Store local data 

s_barrier  // (op 14658) 
v_add_u32 v32, 65540, v38 // (op 193) 



ds_read_b128 v[44:47], v4 // (op 185) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v32, v37 // (op 193) Store local data 


s_waitcnt lgkmcnt(5)
s_barrier  // (op 14676) 


v_add_u32 v32, 66048, v0 // (op 187) 
ds_read_u8 v39, v32 // (op 187) Load local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 14693) 



v_add_u32 v32, 65536, v5 // (op 189) 
ds_read_u8 v24, v32 // (op 189) Load local data 




ds_read_b128 v[34:37], v4 offset:2048 // (op 500) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[40:43], v[44:47], a[0:3], v39, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 190) 



v_add_u32 v25, 65664, v5 // (op 504) 
ds_read_u8 v32, v25 // (op 504) Load local data 




ds_read_b128 v[48:51], v4 offset:4096 // (op 798) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[40:43], v[34:37], a[4:7], v39, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 505) 



v_add_u32 v25, 65792, v5 // (op 802) 
ds_read_u8 v28, v25 // (op 802) Load local data 




ds_read_b128 v[52:55], v4 offset:6144 // (op 1096) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[40:43], v[48:51], a[8:11], v39, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 803) 



v_add_u32 v25, 65920, v5 // (op 1100) 
ds_read_u8 v29, v25 // (op 1100) Load local data 




ds_read_b64_tr_b4 v[56:57], v3 offset:8208 // (op 1528) Transpose load local data 
ds_read_b64_tr_b4 v[58:59], v3 offset:9232 // (op 1528) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[40:43], v[52:55], a[12:15], v39, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1101) 




v_add_u32 v25, 66080, v0 // (op 1531) 
ds_read_u8 v39, v25 // (op 1531) Load local data 




ds_read_b64_tr_b4 v[40:41], v3 offset:8224 // (op 2818) Transpose load local data 
ds_read_b64_tr_b4 v[42:43], v3 offset:9248 // (op 2818) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[44:47], a[16:19], v39, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1533) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[34:37], a[20:23], v39, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1564) 
v_add_u32 v25, 66112, v0 // (op 2821) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v39, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1595) 
ds_read_u8 v60, v25 // (op 2821) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[52:55], a[28:31], v39, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1626) 
ds_read_b64_tr_b4 v[62:63], v3 offset:8240 // (op 4108) Transpose load local data 


ds_read_b64_tr_b4 v[64:65], v3 offset:9264 // (op 4108) Transpose load local data 


v_add_u32 v25, 66144, v0 // (op 4111) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[40:43], v[44:47], a[32:35], v60, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2823) 
ds_read_u8 v39, v25 // (op 4111) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[40:43], v[34:37], a[36:39], v60, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2854) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[40:43], v[48:51], a[40:43], v60, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2885) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[40:43], v[52:55], a[44:47], v60, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2916) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[62:65], v[44:47], a[48:51], v39, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4113) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[62:65], v[34:37], a[52:55], v39, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4144) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[62:65], v[48:51], a[56:59], v39, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4175) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[62:65], v[52:55], a[60:63], v39, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4206) 




s_ashr_i64 s[6:7], s[22:23], 7 // (op 4195) 
s_mov_b32 s11, s6 // (op 4195) convert

s_cmp_lt_u32 s10, s11 // (op 4195) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_4195 // (op 4195) Condition: Bottom (jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopTop_KLoopTail_4195 if true)
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ForLoopBottom_KLoopTail_4195:























s_barrier  // (op 14871) 

















v_cmp_eq_f32 vcc, 0.00000, v18 // (op 135) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_135
v_accvgpr_read v14, a0 // (op 50) 
v_accvgpr_read v15, a1 // (op 50) 
v_accvgpr_read v16, a2 // (op 50) 
v_accvgpr_read v17, a3 // (op 50) 
v_mul_f32 v10, v19, v14 // (op 50) 
v_mul_f32 v11, v19, v15 // (op 50) 
v_mul_f32 v12, v19, v16 // (op 50) 
v_mul_f32 v13, v19, v17 // (op 50) 

v_cvt_f16_f32 v14, v10 // (op 56) 
v_cvt_f16_f32 v15, v11 // (op 56) 
v_cvt_f16_f32 v16, v12 // (op 56) 
v_cvt_f16_f32 v17, v13 // (op 56) 


v_pack_b32_f16 v12, v14, v15 // (op 58) 
v_pack_b32_f16 v13, v16, v17 // (op 58) 

v_mov_b32 v0, v21 // (op 58) reg expression

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 58) Store value



v_accvgpr_read v14, a4 // (op 5877) 
v_accvgpr_read v15, a5 // (op 5877) 
v_accvgpr_read v16, a6 // (op 5877) 
v_accvgpr_read v17, a7 // (op 5877) 
v_mul_f32 v10, v19, v14 // (op 5877) 
v_mul_f32 v11, v19, v15 // (op 5877) 
v_mul_f32 v12, v19, v16 // (op 5877) 
v_mul_f32 v13, v19, v17 // (op 5877) 

v_cvt_f16_f32 v14, v10 // (op 5879) 
v_cvt_f16_f32 v15, v11 // (op 5879) 
v_cvt_f16_f32 v16, v12 // (op 5879) 
v_cvt_f16_f32 v17, v13 // (op 5879) 


v_pack_b32_f16 v12, v14, v15 // (op 5880) 
v_pack_b32_f16 v13, v16, v17 // (op 5880) 

s_lshl_b32 s0, s4, 6 // (op 5880) 
v_add_u32 v0, v21, s0 // (op 5880) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 5880) Store value



v_accvgpr_read v14, a8 // (op 5891) 
v_accvgpr_read v15, a9 // (op 5891) 
v_accvgpr_read v16, a10 // (op 5891) 
v_accvgpr_read v17, a11 // (op 5891) 
v_mul_f32 v10, v19, v14 // (op 5891) 
v_mul_f32 v11, v19, v15 // (op 5891) 
v_mul_f32 v12, v19, v16 // (op 5891) 
v_mul_f32 v13, v19, v17 // (op 5891) 

v_cvt_f16_f32 v14, v10 // (op 5893) 
v_cvt_f16_f32 v15, v11 // (op 5893) 
v_cvt_f16_f32 v16, v12 // (op 5893) 
v_cvt_f16_f32 v17, v13 // (op 5893) 


v_pack_b32_f16 v12, v14, v15 // (op 5894) 
v_pack_b32_f16 v13, v16, v17 // (op 5894) 

s_lshl_b32 s0, s4, 7 // (op 5894) 
v_add_u32 v0, v21, s0 // (op 5894) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 5894) Store value



v_accvgpr_read v14, a12 // (op 5905) 
v_accvgpr_read v15, a13 // (op 5905) 
v_accvgpr_read v16, a14 // (op 5905) 
v_accvgpr_read v17, a15 // (op 5905) 
v_mul_f32 v10, v19, v14 // (op 5905) 
v_mul_f32 v11, v19, v15 // (op 5905) 
v_mul_f32 v12, v19, v16 // (op 5905) 
v_mul_f32 v13, v19, v17 // (op 5905) 

v_cvt_f16_f32 v14, v10 // (op 5907) 
v_cvt_f16_f32 v15, v11 // (op 5907) 
v_cvt_f16_f32 v16, v12 // (op 5907) 
v_cvt_f16_f32 v17, v13 // (op 5907) 


v_pack_b32_f16 v12, v14, v15 // (op 5908) 
v_pack_b32_f16 v13, v16, v17 // (op 5908) 

s_lshl_b32 s0, s4, 6 // (op 5908) 
s_mul_i32 s1, 3, s0 // (op 5908) 

v_add_u32 v0, v21, s1 // (op 5908) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 5908) Store value



v_accvgpr_read v14, a16 // (op 5941) 
v_accvgpr_read v15, a17 // (op 5941) 
v_accvgpr_read v16, a18 // (op 5941) 
v_accvgpr_read v17, a19 // (op 5941) 
v_mul_f32 v10, v19, v14 // (op 5941) 
v_mul_f32 v11, v19, v15 // (op 5941) 
v_mul_f32 v12, v19, v16 // (op 5941) 
v_mul_f32 v13, v19, v17 // (op 5941) 

v_cvt_f16_f32 v14, v10 // (op 5943) 
v_cvt_f16_f32 v15, v11 // (op 5943) 
v_cvt_f16_f32 v16, v12 // (op 5943) 
v_cvt_f16_f32 v17, v13 // (op 5943) 


v_pack_b32_f16 v12, v14, v15 // (op 5945) 
v_pack_b32_f16 v13, v16, v17 // (op 5945) 

v_add_u32 v0, 64, v21 // (op 5945) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 5945) Store value



v_accvgpr_read v14, a20 // (op 5955) 
v_accvgpr_read v15, a21 // (op 5955) 
v_accvgpr_read v16, a22 // (op 5955) 
v_accvgpr_read v17, a23 // (op 5955) 
v_mul_f32 v10, v19, v14 // (op 5955) 
v_mul_f32 v11, v19, v15 // (op 5955) 
v_mul_f32 v12, v19, v16 // (op 5955) 
v_mul_f32 v13, v19, v17 // (op 5955) 

v_cvt_f16_f32 v14, v10 // (op 5957) 
v_cvt_f16_f32 v15, v11 // (op 5957) 
v_cvt_f16_f32 v16, v12 // (op 5957) 
v_cvt_f16_f32 v17, v13 // (op 5957) 


v_pack_b32_f16 v12, v14, v15 // (op 5947) 
v_pack_b32_f16 v13, v16, v17 // (op 5947) 

s_lshl_b32 s0, s4, 6 // (op 5947) 
s_add_u32 s0, s0, 64 // (op 5947) 
v_add_u32 v0, v21, s0 // (op 5947) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 5947) Store value



v_accvgpr_read v14, a24 // (op 5961) 
v_accvgpr_read v15, a25 // (op 5961) 
v_accvgpr_read v16, a26 // (op 5961) 
v_accvgpr_read v17, a27 // (op 5961) 
v_mul_f32 v10, v19, v14 // (op 5961) 
v_mul_f32 v11, v19, v15 // (op 5961) 
v_mul_f32 v12, v19, v16 // (op 5961) 
v_mul_f32 v13, v19, v17 // (op 5961) 

v_cvt_f16_f32 v14, v10 // (op 5963) 
v_cvt_f16_f32 v15, v11 // (op 5963) 
v_cvt_f16_f32 v16, v12 // (op 5963) 
v_cvt_f16_f32 v17, v13 // (op 5963) 


v_pack_b32_f16 v12, v14, v15 // (op 5949) 
v_pack_b32_f16 v13, v16, v17 // (op 5949) 

s_lshl_b32 s0, s4, 7 // (op 5949) 
s_add_u32 s0, s0, 64 // (op 5949) 
v_add_u32 v0, v21, s0 // (op 5949) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 5949) Store value



v_accvgpr_read v14, a28 // (op 5967) 
v_accvgpr_read v15, a29 // (op 5967) 
v_accvgpr_read v16, a30 // (op 5967) 
v_accvgpr_read v17, a31 // (op 5967) 
v_mul_f32 v10, v19, v14 // (op 5967) 
v_mul_f32 v11, v19, v15 // (op 5967) 
v_mul_f32 v12, v19, v16 // (op 5967) 
v_mul_f32 v13, v19, v17 // (op 5967) 

v_cvt_f16_f32 v14, v10 // (op 5969) 
v_cvt_f16_f32 v15, v11 // (op 5969) 
v_cvt_f16_f32 v16, v12 // (op 5969) 
v_cvt_f16_f32 v17, v13 // (op 5969) 


v_pack_b32_f16 v12, v14, v15 // (op 5951) 
v_pack_b32_f16 v13, v16, v17 // (op 5951) 

s_lshl_b32 s0, s4, 6 // (op 5951) 
s_mul_i32 s1, 3, s0 // (op 5951) 
s_add_u32 s1, s1, 64 // (op 5951) 

v_add_u32 v0, v21, s1 // (op 5951) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 5951) Store value



v_accvgpr_read v14, a32 // (op 6017) 
v_accvgpr_read v15, a33 // (op 6017) 
v_accvgpr_read v16, a34 // (op 6017) 
v_accvgpr_read v17, a35 // (op 6017) 
v_mul_f32 v10, v19, v14 // (op 6017) 
v_mul_f32 v11, v19, v15 // (op 6017) 
v_mul_f32 v12, v19, v16 // (op 6017) 
v_mul_f32 v13, v19, v17 // (op 6017) 

v_cvt_f16_f32 v14, v10 // (op 6019) 
v_cvt_f16_f32 v15, v11 // (op 6019) 
v_cvt_f16_f32 v16, v12 // (op 6019) 
v_cvt_f16_f32 v17, v13 // (op 6019) 


v_pack_b32_f16 v12, v14, v15 // (op 6021) 
v_pack_b32_f16 v13, v16, v17 // (op 6021) 

v_add_u32 v0, 128, v21 // (op 6021) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 6021) Store value



v_accvgpr_read v14, a36 // (op 6031) 
v_accvgpr_read v15, a37 // (op 6031) 
v_accvgpr_read v16, a38 // (op 6031) 
v_accvgpr_read v17, a39 // (op 6031) 
v_mul_f32 v10, v19, v14 // (op 6031) 
v_mul_f32 v11, v19, v15 // (op 6031) 
v_mul_f32 v12, v19, v16 // (op 6031) 
v_mul_f32 v13, v19, v17 // (op 6031) 

v_cvt_f16_f32 v14, v10 // (op 6033) 
v_cvt_f16_f32 v15, v11 // (op 6033) 
v_cvt_f16_f32 v16, v12 // (op 6033) 
v_cvt_f16_f32 v17, v13 // (op 6033) 


v_pack_b32_f16 v12, v14, v15 // (op 6023) 
v_pack_b32_f16 v13, v16, v17 // (op 6023) 

s_lshl_b32 s0, s4, 6 // (op 6023) 
s_add_u32 s0, s0, 128 // (op 6023) 
v_add_u32 v0, v21, s0 // (op 6023) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 6023) Store value



v_accvgpr_read v14, a40 // (op 6037) 
v_accvgpr_read v15, a41 // (op 6037) 
v_accvgpr_read v16, a42 // (op 6037) 
v_accvgpr_read v17, a43 // (op 6037) 
v_mul_f32 v10, v19, v14 // (op 6037) 
v_mul_f32 v11, v19, v15 // (op 6037) 
v_mul_f32 v12, v19, v16 // (op 6037) 
v_mul_f32 v13, v19, v17 // (op 6037) 

v_cvt_f16_f32 v14, v10 // (op 6039) 
v_cvt_f16_f32 v15, v11 // (op 6039) 
v_cvt_f16_f32 v16, v12 // (op 6039) 
v_cvt_f16_f32 v17, v13 // (op 6039) 


v_pack_b32_f16 v12, v14, v15 // (op 6025) 
v_pack_b32_f16 v13, v16, v17 // (op 6025) 

s_lshl_b32 s0, s4, 7 // (op 6025) 
s_add_u32 s0, s0, 128 // (op 6025) 
v_add_u32 v0, v21, s0 // (op 6025) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 6025) Store value



v_accvgpr_read v14, a44 // (op 6043) 
v_accvgpr_read v15, a45 // (op 6043) 
v_accvgpr_read v16, a46 // (op 6043) 
v_accvgpr_read v17, a47 // (op 6043) 
v_mul_f32 v10, v19, v14 // (op 6043) 
v_mul_f32 v11, v19, v15 // (op 6043) 
v_mul_f32 v12, v19, v16 // (op 6043) 
v_mul_f32 v13, v19, v17 // (op 6043) 

v_cvt_f16_f32 v14, v10 // (op 6045) 
v_cvt_f16_f32 v15, v11 // (op 6045) 
v_cvt_f16_f32 v16, v12 // (op 6045) 
v_cvt_f16_f32 v17, v13 // (op 6045) 


v_pack_b32_f16 v12, v14, v15 // (op 6027) 
v_pack_b32_f16 v13, v16, v17 // (op 6027) 

s_lshl_b32 s0, s4, 6 // (op 6027) 
s_mul_i32 s1, 3, s0 // (op 6027) 
s_add_u32 s1, s1, 128 // (op 6027) 

v_add_u32 v0, v21, s1 // (op 6027) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 6027) Store value



v_accvgpr_read v14, a48 // (op 6093) 
v_accvgpr_read v15, a49 // (op 6093) 
v_accvgpr_read v16, a50 // (op 6093) 
v_accvgpr_read v17, a51 // (op 6093) 
v_mul_f32 v10, v19, v14 // (op 6093) 
v_mul_f32 v11, v19, v15 // (op 6093) 
v_mul_f32 v12, v19, v16 // (op 6093) 
v_mul_f32 v13, v19, v17 // (op 6093) 

v_cvt_f16_f32 v14, v10 // (op 6095) 
v_cvt_f16_f32 v15, v11 // (op 6095) 
v_cvt_f16_f32 v16, v12 // (op 6095) 
v_cvt_f16_f32 v17, v13 // (op 6095) 


v_pack_b32_f16 v12, v14, v15 // (op 6097) 
v_pack_b32_f16 v13, v16, v17 // (op 6097) 

v_add_u32 v0, 192, v21 // (op 6097) 

buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 6097) Store value



v_accvgpr_read v14, a52 // (op 6107) 
v_accvgpr_read v15, a53 // (op 6107) 
v_accvgpr_read v16, a54 // (op 6107) 
v_accvgpr_read v17, a55 // (op 6107) 
v_mul_f32 v10, v19, v14 // (op 6107) 
v_mul_f32 v11, v19, v15 // (op 6107) 
v_mul_f32 v12, v19, v16 // (op 6107) 
v_mul_f32 v13, v19, v17 // (op 6107) 

v_cvt_f16_f32 v14, v10 // (op 6109) 
v_cvt_f16_f32 v15, v11 // (op 6109) 
v_cvt_f16_f32 v16, v12 // (op 6109) 
v_cvt_f16_f32 v17, v13 // (op 6109) 


v_pack_b32_f16 v12, v14, v15 // (op 6099) 
v_pack_b32_f16 v13, v16, v17 // (op 6099) 

s_lshl_b32 s0, s4, 6 // (op 6099) 
s_add_u32 s0, s0, 192 // (op 6099) 
v_add_u32 v0, v21, s0 // (op 6099) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 6099) Store value



v_accvgpr_read v14, a56 // (op 6113) 
v_accvgpr_read v15, a57 // (op 6113) 
v_accvgpr_read v16, a58 // (op 6113) 
v_accvgpr_read v17, a59 // (op 6113) 
v_mul_f32 v10, v19, v14 // (op 6113) 
v_mul_f32 v11, v19, v15 // (op 6113) 
v_mul_f32 v12, v19, v16 // (op 6113) 
v_mul_f32 v13, v19, v17 // (op 6113) 

v_cvt_f16_f32 v14, v10 // (op 6115) 
v_cvt_f16_f32 v15, v11 // (op 6115) 
v_cvt_f16_f32 v16, v12 // (op 6115) 
v_cvt_f16_f32 v17, v13 // (op 6115) 


v_pack_b32_f16 v12, v14, v15 // (op 6101) 
v_pack_b32_f16 v13, v16, v17 // (op 6101) 

s_lshl_b32 s0, s4, 7 // (op 6101) 
s_add_u32 s0, s0, 192 // (op 6101) 
v_add_u32 v0, v21, s0 // (op 6101) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 6101) Store value



v_accvgpr_read v14, a60 // (op 6119) 
v_accvgpr_read v15, a61 // (op 6119) 
v_accvgpr_read v16, a62 // (op 6119) 
v_accvgpr_read v17, a63 // (op 6119) 
v_mul_f32 v10, v19, v14 // (op 6119) 
v_mul_f32 v11, v19, v15 // (op 6119) 
v_mul_f32 v12, v19, v16 // (op 6119) 
v_mul_f32 v13, v19, v17 // (op 6119) 

v_cvt_f16_f32 v14, v10 // (op 6121) 
v_cvt_f16_f32 v15, v11 // (op 6121) 
v_cvt_f16_f32 v16, v12 // (op 6121) 
v_cvt_f16_f32 v17, v13 // (op 6121) 


v_pack_b32_f16 v12, v14, v15 // (op 6103) 
v_pack_b32_f16 v13, v16, v17 // (op 6103) 

s_lshl_b32 s0, s4, 6 // (op 6103) 
s_mul_i32 s1, 3, s0 // (op 6103) 
s_add_u32 s1, s1, 192 // (op 6103) 

v_add_u32 v0, v21, s1 // (op 6103) 


buffer_store_dwordx2 v[12:13], v0, s[44:47], 0 offen // (op 6103) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_135
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ConditionalFalse_0_beta_135:


v_add_u32 v0, 192, v20 // (op 6491) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6491) Load value

v_accvgpr_read v12, a48 // (op 6478) 
v_accvgpr_read v13, a49 // (op 6478) 
v_accvgpr_read v14, a50 // (op 6478) 
v_accvgpr_read v15, a51 // (op 6478) 
v_mul_f32 v8, v19, v12 // (op 6478) 
v_mul_f32 v9, v19, v13 // (op 6478) 
v_mul_f32 v10, v19, v14 // (op 6478) 
v_mul_f32 v11, v19, v15 // (op 6478) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6490) 
v_lshrrev_b32 v7, 16, v16 // (op 6490) 
v_cvt_f32_f16 v7, v7 // (op 6490) 
v_mul_f32 v12, v18, v6 // (op 6490) 
v_mul_f32 v13, v18, v7 // (op 6490) 
v_cvt_f32_f16 v6, v17 // (op 6490) 
v_lshrrev_b32 v7, 16, v17 // (op 6490) 
v_cvt_f32_f16 v7, v7 // (op 6490) 
v_mul_f32 v14, v18, v6 // (op 6490) 
v_mul_f32 v15, v18, v7 // (op 6490) 


v_add_f32 v4, v12, v8 // (op 6479) 
v_add_f32 v5, v13, v9 // (op 6479) 
v_add_f32 v6, v14, v10 // (op 6479) 
v_add_f32 v7, v15, v11 // (op 6479) 


v_cvt_f16_f32 v8, v4 // (op 6480) 
v_cvt_f16_f32 v9, v5 // (op 6480) 
v_cvt_f16_f32 v10, v6 // (op 6480) 
v_cvt_f16_f32 v11, v7 // (op 6480) 


v_pack_b32_f16 v6, v8, v9 // (op 6482) 
v_pack_b32_f16 v7, v10, v11 // (op 6482) 

v_add_u32 v0, 192, v21 // (op 6482) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6482) Store value





s_lshl_b32 s0, s2, 6 // (op 6496) 
s_add_u32 s0, s0, 192 // (op 6496) 
v_add_u32 v0, v20, s0 // (op 6496) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6496) Load value

v_accvgpr_read v12, a52 // (op 6508) 
v_accvgpr_read v13, a53 // (op 6508) 
v_accvgpr_read v14, a54 // (op 6508) 
v_accvgpr_read v15, a55 // (op 6508) 
v_mul_f32 v8, v19, v12 // (op 6508) 
v_mul_f32 v9, v19, v13 // (op 6508) 
v_mul_f32 v10, v19, v14 // (op 6508) 
v_mul_f32 v11, v19, v15 // (op 6508) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6493) 
v_lshrrev_b32 v7, 16, v16 // (op 6493) 
v_cvt_f32_f16 v7, v7 // (op 6493) 
v_mul_f32 v12, v18, v6 // (op 6493) 
v_mul_f32 v13, v18, v7 // (op 6493) 
v_cvt_f32_f16 v6, v17 // (op 6493) 
v_lshrrev_b32 v7, 16, v17 // (op 6493) 
v_cvt_f32_f16 v7, v7 // (op 6493) 
v_mul_f32 v14, v18, v6 // (op 6493) 
v_mul_f32 v15, v18, v7 // (op 6493) 


v_add_f32 v4, v12, v8 // (op 6494) 
v_add_f32 v5, v13, v9 // (op 6494) 
v_add_f32 v6, v14, v10 // (op 6494) 
v_add_f32 v7, v15, v11 // (op 6494) 


v_cvt_f16_f32 v8, v4 // (op 6495) 
v_cvt_f16_f32 v9, v5 // (op 6495) 
v_cvt_f16_f32 v10, v6 // (op 6495) 
v_cvt_f16_f32 v11, v7 // (op 6495) 


v_pack_b32_f16 v6, v8, v9 // (op 6484) 
v_pack_b32_f16 v7, v10, v11 // (op 6484) 

s_lshl_b32 s0, s4, 6 // (op 6484) 
s_add_u32 s0, s0, 192 // (op 6484) 
v_add_u32 v0, v21, s0 // (op 6484) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6484) Store value





s_lshl_b32 s0, s2, 7 // (op 6501) 
s_add_u32 s0, s0, 192 // (op 6501) 
v_add_u32 v0, v20, s0 // (op 6501) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6501) Load value

v_accvgpr_read v12, a56 // (op 6510) 
v_accvgpr_read v13, a57 // (op 6510) 
v_accvgpr_read v14, a58 // (op 6510) 
v_accvgpr_read v15, a59 // (op 6510) 
v_mul_f32 v8, v19, v12 // (op 6510) 
v_mul_f32 v9, v19, v13 // (op 6510) 
v_mul_f32 v10, v19, v14 // (op 6510) 
v_mul_f32 v11, v19, v15 // (op 6510) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6498) 
v_lshrrev_b32 v7, 16, v16 // (op 6498) 
v_cvt_f32_f16 v7, v7 // (op 6498) 
v_mul_f32 v12, v18, v6 // (op 6498) 
v_mul_f32 v13, v18, v7 // (op 6498) 
v_cvt_f32_f16 v6, v17 // (op 6498) 
v_lshrrev_b32 v7, 16, v17 // (op 6498) 
v_cvt_f32_f16 v7, v7 // (op 6498) 
v_mul_f32 v14, v18, v6 // (op 6498) 
v_mul_f32 v15, v18, v7 // (op 6498) 


v_add_f32 v4, v12, v8 // (op 6499) 
v_add_f32 v5, v13, v9 // (op 6499) 
v_add_f32 v6, v14, v10 // (op 6499) 
v_add_f32 v7, v15, v11 // (op 6499) 


v_cvt_f16_f32 v8, v4 // (op 6500) 
v_cvt_f16_f32 v9, v5 // (op 6500) 
v_cvt_f16_f32 v10, v6 // (op 6500) 
v_cvt_f16_f32 v11, v7 // (op 6500) 


v_pack_b32_f16 v6, v8, v9 // (op 6486) 
v_pack_b32_f16 v7, v10, v11 // (op 6486) 

s_lshl_b32 s0, s4, 7 // (op 6486) 
s_add_u32 s0, s0, 192 // (op 6486) 
v_add_u32 v0, v21, s0 // (op 6486) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6486) Store value





s_lshl_b32 s0, s2, 6 // (op 6506) 
s_mul_i32 s1, 3, s0 // (op 6506) 
s_add_u32 s1, s1, 192 // (op 6506) 

v_add_u32 v0, v20, s1 // (op 6506) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6506) Load value

v_accvgpr_read v12, a60 // (op 6512) 
v_accvgpr_read v13, a61 // (op 6512) 
v_accvgpr_read v14, a62 // (op 6512) 
v_accvgpr_read v15, a63 // (op 6512) 
v_mul_f32 v8, v19, v12 // (op 6512) 
v_mul_f32 v9, v19, v13 // (op 6512) 
v_mul_f32 v10, v19, v14 // (op 6512) 
v_mul_f32 v11, v19, v15 // (op 6512) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6503) 
v_lshrrev_b32 v7, 16, v16 // (op 6503) 
v_cvt_f32_f16 v7, v7 // (op 6503) 
v_mul_f32 v12, v18, v6 // (op 6503) 
v_mul_f32 v13, v18, v7 // (op 6503) 
v_cvt_f32_f16 v6, v17 // (op 6503) 
v_lshrrev_b32 v7, 16, v17 // (op 6503) 
v_cvt_f32_f16 v7, v7 // (op 6503) 
v_mul_f32 v14, v18, v6 // (op 6503) 
v_mul_f32 v15, v18, v7 // (op 6503) 


v_add_f32 v4, v12, v8 // (op 6504) 
v_add_f32 v5, v13, v9 // (op 6504) 
v_add_f32 v6, v14, v10 // (op 6504) 
v_add_f32 v7, v15, v11 // (op 6504) 


v_cvt_f16_f32 v8, v4 // (op 6505) 
v_cvt_f16_f32 v9, v5 // (op 6505) 
v_cvt_f16_f32 v10, v6 // (op 6505) 
v_cvt_f16_f32 v11, v7 // (op 6505) 


v_pack_b32_f16 v6, v8, v9 // (op 6488) 
v_pack_b32_f16 v7, v10, v11 // (op 6488) 

s_lshl_b32 s0, s4, 6 // (op 6488) 
s_mul_i32 s1, 3, s0 // (op 6488) 
s_add_u32 s1, s1, 192 // (op 6488) 

v_add_u32 v0, v21, s1 // (op 6488) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6488) Store value





v_add_u32 v0, 128, v20 // (op 6404) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6404) Load value

v_accvgpr_read v12, a32 // (op 6391) 
v_accvgpr_read v13, a33 // (op 6391) 
v_accvgpr_read v14, a34 // (op 6391) 
v_accvgpr_read v15, a35 // (op 6391) 
v_mul_f32 v8, v19, v12 // (op 6391) 
v_mul_f32 v9, v19, v13 // (op 6391) 
v_mul_f32 v10, v19, v14 // (op 6391) 
v_mul_f32 v11, v19, v15 // (op 6391) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6403) 
v_lshrrev_b32 v7, 16, v16 // (op 6403) 
v_cvt_f32_f16 v7, v7 // (op 6403) 
v_mul_f32 v12, v18, v6 // (op 6403) 
v_mul_f32 v13, v18, v7 // (op 6403) 
v_cvt_f32_f16 v6, v17 // (op 6403) 
v_lshrrev_b32 v7, 16, v17 // (op 6403) 
v_cvt_f32_f16 v7, v7 // (op 6403) 
v_mul_f32 v14, v18, v6 // (op 6403) 
v_mul_f32 v15, v18, v7 // (op 6403) 


v_add_f32 v4, v12, v8 // (op 6392) 
v_add_f32 v5, v13, v9 // (op 6392) 
v_add_f32 v6, v14, v10 // (op 6392) 
v_add_f32 v7, v15, v11 // (op 6392) 


v_cvt_f16_f32 v8, v4 // (op 6393) 
v_cvt_f16_f32 v9, v5 // (op 6393) 
v_cvt_f16_f32 v10, v6 // (op 6393) 
v_cvt_f16_f32 v11, v7 // (op 6393) 


v_pack_b32_f16 v6, v8, v9 // (op 6395) 
v_pack_b32_f16 v7, v10, v11 // (op 6395) 

v_add_u32 v0, 128, v21 // (op 6395) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6395) Store value





s_lshl_b32 s0, s2, 6 // (op 6409) 
s_add_u32 s0, s0, 128 // (op 6409) 
v_add_u32 v0, v20, s0 // (op 6409) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6409) Load value

v_accvgpr_read v12, a36 // (op 6421) 
v_accvgpr_read v13, a37 // (op 6421) 
v_accvgpr_read v14, a38 // (op 6421) 
v_accvgpr_read v15, a39 // (op 6421) 
v_mul_f32 v8, v19, v12 // (op 6421) 
v_mul_f32 v9, v19, v13 // (op 6421) 
v_mul_f32 v10, v19, v14 // (op 6421) 
v_mul_f32 v11, v19, v15 // (op 6421) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6406) 
v_lshrrev_b32 v7, 16, v16 // (op 6406) 
v_cvt_f32_f16 v7, v7 // (op 6406) 
v_mul_f32 v12, v18, v6 // (op 6406) 
v_mul_f32 v13, v18, v7 // (op 6406) 
v_cvt_f32_f16 v6, v17 // (op 6406) 
v_lshrrev_b32 v7, 16, v17 // (op 6406) 
v_cvt_f32_f16 v7, v7 // (op 6406) 
v_mul_f32 v14, v18, v6 // (op 6406) 
v_mul_f32 v15, v18, v7 // (op 6406) 


v_add_f32 v4, v12, v8 // (op 6407) 
v_add_f32 v5, v13, v9 // (op 6407) 
v_add_f32 v6, v14, v10 // (op 6407) 
v_add_f32 v7, v15, v11 // (op 6407) 


v_cvt_f16_f32 v8, v4 // (op 6408) 
v_cvt_f16_f32 v9, v5 // (op 6408) 
v_cvt_f16_f32 v10, v6 // (op 6408) 
v_cvt_f16_f32 v11, v7 // (op 6408) 


v_pack_b32_f16 v6, v8, v9 // (op 6397) 
v_pack_b32_f16 v7, v10, v11 // (op 6397) 

s_lshl_b32 s0, s4, 6 // (op 6397) 
s_add_u32 s0, s0, 128 // (op 6397) 
v_add_u32 v0, v21, s0 // (op 6397) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6397) Store value





s_lshl_b32 s0, s2, 7 // (op 6414) 
s_add_u32 s0, s0, 128 // (op 6414) 
v_add_u32 v0, v20, s0 // (op 6414) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6414) Load value

v_accvgpr_read v12, a40 // (op 6423) 
v_accvgpr_read v13, a41 // (op 6423) 
v_accvgpr_read v14, a42 // (op 6423) 
v_accvgpr_read v15, a43 // (op 6423) 
v_mul_f32 v8, v19, v12 // (op 6423) 
v_mul_f32 v9, v19, v13 // (op 6423) 
v_mul_f32 v10, v19, v14 // (op 6423) 
v_mul_f32 v11, v19, v15 // (op 6423) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6411) 
v_lshrrev_b32 v7, 16, v16 // (op 6411) 
v_cvt_f32_f16 v7, v7 // (op 6411) 
v_mul_f32 v12, v18, v6 // (op 6411) 
v_mul_f32 v13, v18, v7 // (op 6411) 
v_cvt_f32_f16 v6, v17 // (op 6411) 
v_lshrrev_b32 v7, 16, v17 // (op 6411) 
v_cvt_f32_f16 v7, v7 // (op 6411) 
v_mul_f32 v14, v18, v6 // (op 6411) 
v_mul_f32 v15, v18, v7 // (op 6411) 


v_add_f32 v4, v12, v8 // (op 6412) 
v_add_f32 v5, v13, v9 // (op 6412) 
v_add_f32 v6, v14, v10 // (op 6412) 
v_add_f32 v7, v15, v11 // (op 6412) 


v_cvt_f16_f32 v8, v4 // (op 6413) 
v_cvt_f16_f32 v9, v5 // (op 6413) 
v_cvt_f16_f32 v10, v6 // (op 6413) 
v_cvt_f16_f32 v11, v7 // (op 6413) 


v_pack_b32_f16 v6, v8, v9 // (op 6399) 
v_pack_b32_f16 v7, v10, v11 // (op 6399) 

s_lshl_b32 s0, s4, 7 // (op 6399) 
s_add_u32 s0, s0, 128 // (op 6399) 
v_add_u32 v0, v21, s0 // (op 6399) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6399) Store value





s_lshl_b32 s0, s2, 6 // (op 6419) 
s_mul_i32 s1, 3, s0 // (op 6419) 
s_add_u32 s1, s1, 128 // (op 6419) 

v_add_u32 v0, v20, s1 // (op 6419) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6419) Load value

v_accvgpr_read v12, a44 // (op 6425) 
v_accvgpr_read v13, a45 // (op 6425) 
v_accvgpr_read v14, a46 // (op 6425) 
v_accvgpr_read v15, a47 // (op 6425) 
v_mul_f32 v8, v19, v12 // (op 6425) 
v_mul_f32 v9, v19, v13 // (op 6425) 
v_mul_f32 v10, v19, v14 // (op 6425) 
v_mul_f32 v11, v19, v15 // (op 6425) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6416) 
v_lshrrev_b32 v7, 16, v16 // (op 6416) 
v_cvt_f32_f16 v7, v7 // (op 6416) 
v_mul_f32 v12, v18, v6 // (op 6416) 
v_mul_f32 v13, v18, v7 // (op 6416) 
v_cvt_f32_f16 v6, v17 // (op 6416) 
v_lshrrev_b32 v7, 16, v17 // (op 6416) 
v_cvt_f32_f16 v7, v7 // (op 6416) 
v_mul_f32 v14, v18, v6 // (op 6416) 
v_mul_f32 v15, v18, v7 // (op 6416) 


v_add_f32 v4, v12, v8 // (op 6417) 
v_add_f32 v5, v13, v9 // (op 6417) 
v_add_f32 v6, v14, v10 // (op 6417) 
v_add_f32 v7, v15, v11 // (op 6417) 


v_cvt_f16_f32 v8, v4 // (op 6418) 
v_cvt_f16_f32 v9, v5 // (op 6418) 
v_cvt_f16_f32 v10, v6 // (op 6418) 
v_cvt_f16_f32 v11, v7 // (op 6418) 


v_pack_b32_f16 v6, v8, v9 // (op 6401) 
v_pack_b32_f16 v7, v10, v11 // (op 6401) 

s_lshl_b32 s0, s4, 6 // (op 6401) 
s_mul_i32 s1, 3, s0 // (op 6401) 
s_add_u32 s1, s1, 128 // (op 6401) 

v_add_u32 v0, v21, s1 // (op 6401) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6401) Store value





v_add_u32 v0, 64, v20 // (op 6317) 

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6317) Load value

v_accvgpr_read v12, a16 // (op 6304) 
v_accvgpr_read v13, a17 // (op 6304) 
v_accvgpr_read v14, a18 // (op 6304) 
v_accvgpr_read v15, a19 // (op 6304) 
v_mul_f32 v8, v19, v12 // (op 6304) 
v_mul_f32 v9, v19, v13 // (op 6304) 
v_mul_f32 v10, v19, v14 // (op 6304) 
v_mul_f32 v11, v19, v15 // (op 6304) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6316) 
v_lshrrev_b32 v7, 16, v16 // (op 6316) 
v_cvt_f32_f16 v7, v7 // (op 6316) 
v_mul_f32 v12, v18, v6 // (op 6316) 
v_mul_f32 v13, v18, v7 // (op 6316) 
v_cvt_f32_f16 v6, v17 // (op 6316) 
v_lshrrev_b32 v7, 16, v17 // (op 6316) 
v_cvt_f32_f16 v7, v7 // (op 6316) 
v_mul_f32 v14, v18, v6 // (op 6316) 
v_mul_f32 v15, v18, v7 // (op 6316) 


v_add_f32 v4, v12, v8 // (op 6305) 
v_add_f32 v5, v13, v9 // (op 6305) 
v_add_f32 v6, v14, v10 // (op 6305) 
v_add_f32 v7, v15, v11 // (op 6305) 


v_cvt_f16_f32 v8, v4 // (op 6306) 
v_cvt_f16_f32 v9, v5 // (op 6306) 
v_cvt_f16_f32 v10, v6 // (op 6306) 
v_cvt_f16_f32 v11, v7 // (op 6306) 


v_pack_b32_f16 v6, v8, v9 // (op 6308) 
v_pack_b32_f16 v7, v10, v11 // (op 6308) 

v_add_u32 v0, 64, v21 // (op 6308) 

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6308) Store value





s_lshl_b32 s0, s2, 6 // (op 6322) 
s_add_u32 s0, s0, 64 // (op 6322) 
v_add_u32 v0, v20, s0 // (op 6322) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6322) Load value

v_accvgpr_read v12, a20 // (op 6334) 
v_accvgpr_read v13, a21 // (op 6334) 
v_accvgpr_read v14, a22 // (op 6334) 
v_accvgpr_read v15, a23 // (op 6334) 
v_mul_f32 v8, v19, v12 // (op 6334) 
v_mul_f32 v9, v19, v13 // (op 6334) 
v_mul_f32 v10, v19, v14 // (op 6334) 
v_mul_f32 v11, v19, v15 // (op 6334) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6319) 
v_lshrrev_b32 v7, 16, v16 // (op 6319) 
v_cvt_f32_f16 v7, v7 // (op 6319) 
v_mul_f32 v12, v18, v6 // (op 6319) 
v_mul_f32 v13, v18, v7 // (op 6319) 
v_cvt_f32_f16 v6, v17 // (op 6319) 
v_lshrrev_b32 v7, 16, v17 // (op 6319) 
v_cvt_f32_f16 v7, v7 // (op 6319) 
v_mul_f32 v14, v18, v6 // (op 6319) 
v_mul_f32 v15, v18, v7 // (op 6319) 


v_add_f32 v4, v12, v8 // (op 6320) 
v_add_f32 v5, v13, v9 // (op 6320) 
v_add_f32 v6, v14, v10 // (op 6320) 
v_add_f32 v7, v15, v11 // (op 6320) 


v_cvt_f16_f32 v8, v4 // (op 6321) 
v_cvt_f16_f32 v9, v5 // (op 6321) 
v_cvt_f16_f32 v10, v6 // (op 6321) 
v_cvt_f16_f32 v11, v7 // (op 6321) 


v_pack_b32_f16 v6, v8, v9 // (op 6310) 
v_pack_b32_f16 v7, v10, v11 // (op 6310) 

s_lshl_b32 s0, s4, 6 // (op 6310) 
s_add_u32 s0, s0, 64 // (op 6310) 
v_add_u32 v0, v21, s0 // (op 6310) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6310) Store value





s_lshl_b32 s0, s2, 7 // (op 6327) 
s_add_u32 s0, s0, 64 // (op 6327) 
v_add_u32 v0, v20, s0 // (op 6327) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6327) Load value

v_accvgpr_read v12, a24 // (op 6336) 
v_accvgpr_read v13, a25 // (op 6336) 
v_accvgpr_read v14, a26 // (op 6336) 
v_accvgpr_read v15, a27 // (op 6336) 
v_mul_f32 v8, v19, v12 // (op 6336) 
v_mul_f32 v9, v19, v13 // (op 6336) 
v_mul_f32 v10, v19, v14 // (op 6336) 
v_mul_f32 v11, v19, v15 // (op 6336) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6324) 
v_lshrrev_b32 v7, 16, v16 // (op 6324) 
v_cvt_f32_f16 v7, v7 // (op 6324) 
v_mul_f32 v12, v18, v6 // (op 6324) 
v_mul_f32 v13, v18, v7 // (op 6324) 
v_cvt_f32_f16 v6, v17 // (op 6324) 
v_lshrrev_b32 v7, 16, v17 // (op 6324) 
v_cvt_f32_f16 v7, v7 // (op 6324) 
v_mul_f32 v14, v18, v6 // (op 6324) 
v_mul_f32 v15, v18, v7 // (op 6324) 


v_add_f32 v4, v12, v8 // (op 6325) 
v_add_f32 v5, v13, v9 // (op 6325) 
v_add_f32 v6, v14, v10 // (op 6325) 
v_add_f32 v7, v15, v11 // (op 6325) 


v_cvt_f16_f32 v8, v4 // (op 6326) 
v_cvt_f16_f32 v9, v5 // (op 6326) 
v_cvt_f16_f32 v10, v6 // (op 6326) 
v_cvt_f16_f32 v11, v7 // (op 6326) 


v_pack_b32_f16 v6, v8, v9 // (op 6312) 
v_pack_b32_f16 v7, v10, v11 // (op 6312) 

s_lshl_b32 s0, s4, 7 // (op 6312) 
s_add_u32 s0, s0, 64 // (op 6312) 
v_add_u32 v0, v21, s0 // (op 6312) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6312) Store value





s_lshl_b32 s0, s2, 6 // (op 6332) 
s_mul_i32 s1, 3, s0 // (op 6332) 
s_add_u32 s1, s1, 64 // (op 6332) 

v_add_u32 v0, v20, s1 // (op 6332) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6332) Load value

v_accvgpr_read v12, a28 // (op 6338) 
v_accvgpr_read v13, a29 // (op 6338) 
v_accvgpr_read v14, a30 // (op 6338) 
v_accvgpr_read v15, a31 // (op 6338) 
v_mul_f32 v8, v19, v12 // (op 6338) 
v_mul_f32 v9, v19, v13 // (op 6338) 
v_mul_f32 v10, v19, v14 // (op 6338) 
v_mul_f32 v11, v19, v15 // (op 6338) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6329) 
v_lshrrev_b32 v7, 16, v16 // (op 6329) 
v_cvt_f32_f16 v7, v7 // (op 6329) 
v_mul_f32 v12, v18, v6 // (op 6329) 
v_mul_f32 v13, v18, v7 // (op 6329) 
v_cvt_f32_f16 v6, v17 // (op 6329) 
v_lshrrev_b32 v7, 16, v17 // (op 6329) 
v_cvt_f32_f16 v7, v7 // (op 6329) 
v_mul_f32 v14, v18, v6 // (op 6329) 
v_mul_f32 v15, v18, v7 // (op 6329) 


v_add_f32 v4, v12, v8 // (op 6330) 
v_add_f32 v5, v13, v9 // (op 6330) 
v_add_f32 v6, v14, v10 // (op 6330) 
v_add_f32 v7, v15, v11 // (op 6330) 


v_cvt_f16_f32 v8, v4 // (op 6331) 
v_cvt_f16_f32 v9, v5 // (op 6331) 
v_cvt_f16_f32 v10, v6 // (op 6331) 
v_cvt_f16_f32 v11, v7 // (op 6331) 


v_pack_b32_f16 v6, v8, v9 // (op 6314) 
v_pack_b32_f16 v7, v10, v11 // (op 6314) 

s_lshl_b32 s0, s4, 6 // (op 6314) 
s_mul_i32 s1, 3, s0 // (op 6314) 
s_add_u32 s1, s1, 64 // (op 6314) 

v_add_u32 v0, v21, s1 // (op 6314) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6314) Store value





v_mov_b32 v0, v20 // (op 148) reg expression

buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 148) Load value

v_accvgpr_read v12, a0 // (op 144) 
v_accvgpr_read v13, a1 // (op 144) 
v_accvgpr_read v14, a2 // (op 144) 
v_accvgpr_read v15, a3 // (op 144) 
v_mul_f32 v8, v19, v12 // (op 144) 
v_mul_f32 v9, v19, v13 // (op 144) 
v_mul_f32 v10, v19, v14 // (op 144) 
v_mul_f32 v11, v19, v15 // (op 144) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 149) 
v_lshrrev_b32 v7, 16, v16 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v12, v18, v6 // (op 149) 
v_mul_f32 v13, v18, v7 // (op 149) 
v_cvt_f32_f16 v6, v17 // (op 149) 
v_lshrrev_b32 v7, 16, v17 // (op 149) 
v_cvt_f32_f16 v7, v7 // (op 149) 
v_mul_f32 v14, v18, v6 // (op 149) 
v_mul_f32 v15, v18, v7 // (op 149) 


v_add_f32 v4, v12, v8 // (op 145) 
v_add_f32 v5, v13, v9 // (op 145) 
v_add_f32 v6, v14, v10 // (op 145) 
v_add_f32 v7, v15, v11 // (op 145) 


v_cvt_f16_f32 v8, v4 // (op 146) 
v_cvt_f16_f32 v9, v5 // (op 146) 
v_cvt_f16_f32 v10, v6 // (op 146) 
v_cvt_f16_f32 v11, v7 // (op 146) 


v_pack_b32_f16 v6, v8, v9 // (op 147) 
v_pack_b32_f16 v7, v10, v11 // (op 147) 

v_mov_b32 v0, v21 // (op 147) reg expression

buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 147) Store value





s_lshl_b32 s0, s2, 6 // (op 6225) 
v_add_u32 v0, v20, s0 // (op 6225) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6225) Load value

v_accvgpr_read v12, a4 // (op 6221) 
v_accvgpr_read v13, a5 // (op 6221) 
v_accvgpr_read v14, a6 // (op 6221) 
v_accvgpr_read v15, a7 // (op 6221) 
v_mul_f32 v8, v19, v12 // (op 6221) 
v_mul_f32 v9, v19, v13 // (op 6221) 
v_mul_f32 v10, v19, v14 // (op 6221) 
v_mul_f32 v11, v19, v15 // (op 6221) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6226) 
v_lshrrev_b32 v7, 16, v16 // (op 6226) 
v_cvt_f32_f16 v7, v7 // (op 6226) 
v_mul_f32 v12, v18, v6 // (op 6226) 
v_mul_f32 v13, v18, v7 // (op 6226) 
v_cvt_f32_f16 v6, v17 // (op 6226) 
v_lshrrev_b32 v7, 16, v17 // (op 6226) 
v_cvt_f32_f16 v7, v7 // (op 6226) 
v_mul_f32 v14, v18, v6 // (op 6226) 
v_mul_f32 v15, v18, v7 // (op 6226) 


v_add_f32 v4, v12, v8 // (op 6222) 
v_add_f32 v5, v13, v9 // (op 6222) 
v_add_f32 v6, v14, v10 // (op 6222) 
v_add_f32 v7, v15, v11 // (op 6222) 


v_cvt_f16_f32 v8, v4 // (op 6223) 
v_cvt_f16_f32 v9, v5 // (op 6223) 
v_cvt_f16_f32 v10, v6 // (op 6223) 
v_cvt_f16_f32 v11, v7 // (op 6223) 


v_pack_b32_f16 v6, v8, v9 // (op 6224) 
v_pack_b32_f16 v7, v10, v11 // (op 6224) 

s_lshl_b32 s0, s4, 6 // (op 6224) 
v_add_u32 v0, v21, s0 // (op 6224) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6224) Store value





s_lshl_b32 s0, s2, 7 // (op 6239) 
v_add_u32 v0, v20, s0 // (op 6239) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6239) Load value

v_accvgpr_read v12, a8 // (op 6235) 
v_accvgpr_read v13, a9 // (op 6235) 
v_accvgpr_read v14, a10 // (op 6235) 
v_accvgpr_read v15, a11 // (op 6235) 
v_mul_f32 v8, v19, v12 // (op 6235) 
v_mul_f32 v9, v19, v13 // (op 6235) 
v_mul_f32 v10, v19, v14 // (op 6235) 
v_mul_f32 v11, v19, v15 // (op 6235) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6240) 
v_lshrrev_b32 v7, 16, v16 // (op 6240) 
v_cvt_f32_f16 v7, v7 // (op 6240) 
v_mul_f32 v12, v18, v6 // (op 6240) 
v_mul_f32 v13, v18, v7 // (op 6240) 
v_cvt_f32_f16 v6, v17 // (op 6240) 
v_lshrrev_b32 v7, 16, v17 // (op 6240) 
v_cvt_f32_f16 v7, v7 // (op 6240) 
v_mul_f32 v14, v18, v6 // (op 6240) 
v_mul_f32 v15, v18, v7 // (op 6240) 


v_add_f32 v4, v12, v8 // (op 6236) 
v_add_f32 v5, v13, v9 // (op 6236) 
v_add_f32 v6, v14, v10 // (op 6236) 
v_add_f32 v7, v15, v11 // (op 6236) 


v_cvt_f16_f32 v8, v4 // (op 6237) 
v_cvt_f16_f32 v9, v5 // (op 6237) 
v_cvt_f16_f32 v10, v6 // (op 6237) 
v_cvt_f16_f32 v11, v7 // (op 6237) 


v_pack_b32_f16 v6, v8, v9 // (op 6238) 
v_pack_b32_f16 v7, v10, v11 // (op 6238) 

s_lshl_b32 s0, s4, 7 // (op 6238) 
v_add_u32 v0, v21, s0 // (op 6238) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6238) Store value





s_lshl_b32 s0, s2, 6 // (op 6253) 
s_mul_i32 s1, 3, s0 // (op 6253) 

v_add_u32 v0, v20, s1 // (op 6253) 


buffer_load_dwordx2 v[16:17], v0, s[40:43], 0 offen // (op 6253) Load value

v_accvgpr_read v12, a12 // (op 6249) 
v_accvgpr_read v13, a13 // (op 6249) 
v_accvgpr_read v14, a14 // (op 6249) 
v_accvgpr_read v15, a15 // (op 6249) 
v_mul_f32 v8, v19, v12 // (op 6249) 
v_mul_f32 v9, v19, v13 // (op 6249) 
v_mul_f32 v10, v19, v14 // (op 6249) 
v_mul_f32 v11, v19, v15 // (op 6249) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v16 // (op 6254) 
v_lshrrev_b32 v7, 16, v16 // (op 6254) 
v_cvt_f32_f16 v7, v7 // (op 6254) 
v_mul_f32 v12, v18, v6 // (op 6254) 
v_mul_f32 v13, v18, v7 // (op 6254) 
v_cvt_f32_f16 v6, v17 // (op 6254) 
v_lshrrev_b32 v7, 16, v17 // (op 6254) 
v_cvt_f32_f16 v7, v7 // (op 6254) 
v_mul_f32 v14, v18, v6 // (op 6254) 
v_mul_f32 v15, v18, v7 // (op 6254) 


v_add_f32 v4, v12, v8 // (op 6250) 
v_add_f32 v5, v13, v9 // (op 6250) 
v_add_f32 v6, v14, v10 // (op 6250) 
v_add_f32 v7, v15, v11 // (op 6250) 


v_cvt_f16_f32 v8, v4 // (op 6251) 
v_cvt_f16_f32 v9, v5 // (op 6251) 
v_cvt_f16_f32 v10, v6 // (op 6251) 
v_cvt_f16_f32 v11, v7 // (op 6251) 


v_pack_b32_f16 v6, v8, v9 // (op 6252) 
v_pack_b32_f16 v7, v10, v11 // (op 6252) 

s_lshl_b32 s0, s4, 6 // (op 6252) 
s_mul_i32 s1, 3, s0 // (op 6252) 

v_add_u32 v0, v21, s1 // (op 6252) 


buffer_store_dwordx2 v[6:7], v0, s[44:47], 0 offen // (op 6252) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_ConditionalBottom_0_beta_135:
























s_endpgm  // End of RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4
.LRR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_end:
.size RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4, .LRR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_end-RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4
  .amdhsa_next_free_vgpr 192
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 67584 // lds bytes
  .amdhsa_accum_offset 128
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
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 4
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4
    .symbol: RR_GEMM_NN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 67584
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 66
    .vgpr_count: 126
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
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
      - .name: Tensor_2_size_1_4
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
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_1_6
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_7
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_7_stride_1_9
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

