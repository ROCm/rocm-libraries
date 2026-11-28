// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
.p2align 8
.type RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4,@function
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_exec_begin:


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


v_and_b32 v0, 63, v1 // (op 3556) 
v_and_b32 v3, 48, v1 // (op 3556) 
v_sub_u32 v4, v0, v3 // (op 3556) 

v_lshrrev_b32 v3, 4, v0 // (op 3556) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 3556) 


v_lshrrev_b32 v3, 7, v1 // (op 3556) 
v_lshlrev_b32 v4, 4, v3 // (op 3556) 

v_lshl_add_u32 v3, v0, 6, v4 // (op 3556) 


v_lshlrev_b32 v0, 2, v3 // (op 3556) 

v_lshrrev_b32 v3, 3, v0 // (op 3556) 

v_and_b32 v0, 63, v1 // (op 3614) 
v_and_b32 v4, 48, v1 // (op 3614) 
v_sub_u32 v5, v0, v4 // (op 3614) 

v_lshrrev_b32 v4, 4, v0 // (op 3614) 

v_lshl_add_u32 v0, v4, 5, v5 // (op 3614) 


v_lshrrev_b32 v4, 6, v1 // (op 3614) 
v_and_b32 v5, 1, v4 // (op 3614) 

v_lshlrev_b32 v4, 4, v5 // (op 3614) 

v_lshl_add_u32 v5, v0, 6, v4 // (op 3614) 


v_lshlrev_b32 v0, 2, v5 // (op 3614) 

v_lshrrev_b32 v4, 3, v0 // (op 3614) 

v_lshrrev_b32 v0, 7, v1 // (op 3672) 
v_and_b32 v5, 15, v1 // (op 3672) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 3672) 


v_and_b32 v0, 63, v1 // (op 3672) 
v_lshrrev_b32 v5, 4, v0 // (op 3672) 

v_lshl_add_u32 v0, v5, 6, v6 // (op 3672) 


v_lshrrev_b32 v5, 6, v1 // (op 3730) 
v_and_b32 v6, 1, v5 // (op 3730) 

v_and_b32 v5, 15, v1 // (op 3730) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 3730) 


v_and_b32 v5, 63, v1 // (op 3730) 
v_lshrrev_b32 v6, 4, v5 // (op 3730) 

v_lshl_add_u32 v5, v6, 6, v7 // (op 3730) 


v_and_b32 v6, 63, v1 // (op 3782) 
v_and_b32 v7, 48, v1 // (op 3782) 
v_sub_u32 v8, v6, v7 // (op 3782) 

v_lshrrev_b32 v7, 4, v6 // (op 3782) 

v_lshl_add_u32 v6, v7, 5, v8 // (op 3782) 


v_lshrrev_b32 v7, 7, v1 // (op 3782) 
v_lshlrev_b32 v8, 4, v7 // (op 3782) 

v_lshl_add_u32 v7, v6, 6, v8 // (op 3782) 


v_lshlrev_b32 v6, 2, v7 // (op 3782) 

v_lshrrev_b32 v7, 3, v6 // (op 3782) 

v_and_b32 v6, 63, v1 // (op 3811) 
v_and_b32 v8, 48, v1 // (op 3811) 
v_sub_u32 v9, v6, v8 // (op 3811) 

v_lshrrev_b32 v8, 4, v6 // (op 3811) 

v_lshl_add_u32 v6, v8, 5, v9 // (op 3811) 


v_lshrrev_b32 v8, 6, v1 // (op 3811) 
v_and_b32 v9, 1, v8 // (op 3811) 

v_lshlrev_b32 v8, 4, v9 // (op 3811) 

v_lshl_add_u32 v9, v6, 6, v8 // (op 3811) 


v_lshlrev_b32 v6, 2, v9 // (op 3811) 

v_lshrrev_b32 v8, 3, v6 // (op 3811) 

v_lshrrev_b32 v6, 7, v1 // (op 3840) 
v_and_b32 v9, 15, v1 // (op 3840) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 3840) 


v_and_b32 v6, 63, v1 // (op 3840) 
v_lshrrev_b32 v9, 4, v6 // (op 3840) 

v_lshl_add_u32 v6, v9, 6, v10 // (op 3840) 


v_lshrrev_b32 v9, 6, v1 // (op 3869) 
v_and_b32 v10, 1, v9 // (op 3869) 

v_and_b32 v9, 15, v1 // (op 3869) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 3869) 


v_and_b32 v9, 63, v1 // (op 3869) 
v_lshrrev_b32 v10, 4, v9 // (op 3869) 

v_lshl_add_u32 v9, v10, 6, v11 // (op 3869) 


v_and_b32 v10, 63, v1 // (op 3898) 
v_and_b32 v11, 48, v1 // (op 3898) 
v_sub_u32 v12, v10, v11 // (op 3898) 

v_lshrrev_b32 v11, 4, v10 // (op 3898) 

v_lshl_add_u32 v10, v11, 5, v12 // (op 3898) 


v_lshrrev_b32 v11, 7, v1 // (op 3898) 
v_lshlrev_b32 v12, 4, v11 // (op 3898) 

v_lshl_add_u32 v11, v10, 6, v12 // (op 3898) 


v_lshlrev_b32 v10, 2, v11 // (op 3898) 

v_lshrrev_b32 v11, 3, v10 // (op 3898) 

v_and_b32 v10, 63, v1 // (op 3927) 
v_and_b32 v12, 48, v1 // (op 3927) 
v_sub_u32 v13, v10, v12 // (op 3927) 

v_lshrrev_b32 v12, 4, v10 // (op 3927) 

v_lshl_add_u32 v10, v12, 5, v13 // (op 3927) 


v_lshrrev_b32 v12, 6, v1 // (op 3927) 
v_and_b32 v13, 1, v12 // (op 3927) 

v_lshlrev_b32 v12, 4, v13 // (op 3927) 

v_lshl_add_u32 v13, v10, 6, v12 // (op 3927) 


v_lshlrev_b32 v10, 2, v13 // (op 3927) 

v_lshrrev_b32 v12, 3, v10 // (op 3927) 

v_lshrrev_b32 v10, 7, v1 // (op 3956) 
v_and_b32 v13, 15, v1 // (op 3956) 
v_lshl_add_u32 v14, v10, 4, v13 // (op 3956) 


v_and_b32 v10, 63, v1 // (op 3956) 
v_lshrrev_b32 v13, 4, v10 // (op 3956) 

v_lshl_add_u32 v10, v13, 6, v14 // (op 3956) 


v_lshrrev_b32 v13, 6, v1 // (op 3985) 
v_and_b32 v14, 1, v13 // (op 3985) 

v_and_b32 v13, 15, v1 // (op 3985) 
v_lshl_add_u32 v15, v14, 4, v13 // (op 3985) 


v_and_b32 v13, 63, v1 // (op 3985) 
v_lshrrev_b32 v14, 4, v13 // (op 3985) 

v_lshl_add_u32 v13, v14, 6, v15 // (op 3985) 


v_and_b32 v14, 63, v1 // (op 4014) 
v_and_b32 v15, 48, v1 // (op 4014) 
v_sub_u32 v16, v14, v15 // (op 4014) 

v_lshrrev_b32 v15, 4, v14 // (op 4014) 

v_lshl_add_u32 v14, v15, 5, v16 // (op 4014) 


v_lshrrev_b32 v15, 7, v1 // (op 4014) 
v_lshlrev_b32 v16, 4, v15 // (op 4014) 

v_lshl_add_u32 v15, v14, 6, v16 // (op 4014) 


v_lshlrev_b32 v14, 2, v15 // (op 4014) 

v_lshrrev_b32 v15, 3, v14 // (op 4014) 

v_and_b32 v14, 63, v1 // (op 4043) 
v_and_b32 v16, 48, v1 // (op 4043) 
v_sub_u32 v17, v14, v16 // (op 4043) 

v_lshrrev_b32 v16, 4, v14 // (op 4043) 

v_lshl_add_u32 v14, v16, 5, v17 // (op 4043) 


v_lshrrev_b32 v16, 6, v1 // (op 4043) 
v_and_b32 v17, 1, v16 // (op 4043) 

v_lshlrev_b32 v16, 4, v17 // (op 4043) 

v_lshl_add_u32 v17, v14, 6, v16 // (op 4043) 


v_lshlrev_b32 v14, 2, v17 // (op 4043) 

v_lshrrev_b32 v16, 3, v14 // (op 4043) 

v_lshrrev_b32 v14, 7, v1 // (op 4072) 
v_and_b32 v17, 15, v1 // (op 4072) 
v_lshl_add_u32 v18, v14, 4, v17 // (op 4072) 


v_and_b32 v14, 63, v1 // (op 4072) 
v_lshrrev_b32 v17, 4, v14 // (op 4072) 

v_lshl_add_u32 v14, v17, 6, v18 // (op 4072) 


v_lshrrev_b32 v17, 6, v1 // (op 4101) 
v_and_b32 v18, 1, v17 // (op 4101) 

v_and_b32 v17, 15, v1 // (op 4101) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 4101) 


v_and_b32 v17, 63, v1 // (op 4101) 
v_lshrrev_b32 v18, 4, v17 // (op 4101) 

v_lshl_add_u32 v17, v18, 6, v19 // (op 4101) 


v_lshrrev_b32 v18, 7, v1 // (op 3494) 
v_and_b32 v19, 63, v1 // (op 3494) 
v_lshrrev_b32 v20, 4, v19 // (op 3494) 
v_lshlrev_b32 v21, 2, v20 // (op 3494) 

v_lshl_add_u32 v20, v18, 4, v21 // (op 3494) 


v_lshl_add_u32 v18, s16, 6, v20 // (op 3494) 

v_lshrrev_b32 v20, 2, v19 // (op 3494) 

v_and_b32 v19, 3, v20 // (op 3494) 

v_and_b32 v20, 3, v1 // (op 3494) 
v_lshl_add_u32 v21, v19, 2, v20 // (op 3494) 


v_lshrrev_b32 v19, 6, v1 // (op 3494) 
v_and_b32 v20, 1, v19 // (op 3494) 

v_lshl_add_u32 v19, v20, 4, v21 // (op 3494) 


v_lshl_add_u32 v20, s17, 6, v19 // (op 3494) 

v_mul_lo_u32 v19, v20, s2 // (op 3494) 

v_add_lshl_u32 v20, v18, v19, 2 // (op 3494) 


v_lshrrev_b32 v18, 7, v1 // (op 3751) 
v_and_b32 v19, 63, v1 // (op 3751) 
v_lshrrev_b32 v21, 4, v19 // (op 3751) 
s_mul_i32 s1, s2, s9 // (op 3496) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 3496) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 3496) least significant
s_mul_hi_u32 s19, s2, s8 // (op 3496) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3496) most significant: sum
s_add_u32 s1, s1, s19 // (op 3496) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3496) 

v_lshlrev_b32 v22, 2, v21 // (op 3751) 

v_lshl_add_u32 v21, v18, 4, v22 // (op 3751) 


v_lshl_add_u32 v18, s16, 6, v21 // (op 3751) 

v_lshrrev_b32 v21, 2, v19 // (op 3751) 

v_and_b32 v19, 3, v21 // (op 3751) 

v_and_b32 v21, 3, v1 // (op 3751) 
v_lshl_add_u32 v22, v19, 2, v21 // (op 3751) 


v_lshrrev_b32 v19, 6, v1 // (op 3751) 
v_and_b32 v21, 1, v19 // (op 3751) 

v_lshl_add_u32 v19, v21, 4, v22 // (op 3751) 


v_lshl_add_u32 v21, s17, 6, v19 // (op 3751) 

v_mul_lo_u32 v19, v21, s4 // (op 3751) 

v_add_lshl_u32 v21, v18, v19, 2 // (op 3751) 


s_mov_b32 s40, s6 // (op 3496) reg expression
s_mov_b32 s41, s7 // (op 3496) reg expression
s_mov_b32 s42, s18 // (op 3496) reg expression
s_mov_b32 s43, 131072 // (op 3496) call()

s_mul_i32 s1, s4, s9 // (op 3753) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 3753) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 3753) least significant
s_mul_hi_u32 s7, s4, s8 // (op 3753) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3753) most significant: sum
s_add_u32 s1, s1, s7 // (op 3753) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3753) 

s_mov_b32 s44, s10 // (op 3753) reg expression
s_mov_b32 s45, s11 // (op 3753) reg expression
s_mov_b32 s46, s6 // (op 3753) reg expression
s_mov_b32 s47, 131072 // (op 3753) call()



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
v_accvgpr_write a8, 0.00000 // (op 848) call()
v_accvgpr_write a9, 0.00000 // (op 848) call()
v_accvgpr_write a10, 0.00000 // (op 848) call()
v_accvgpr_write a11, 0.00000 // (op 848) call()
v_accvgpr_write a12, 0.00000 // (op 1095) call()
v_accvgpr_write a13, 0.00000 // (op 1095) call()
v_accvgpr_write a14, 0.00000 // (op 1095) call()
v_accvgpr_write a15, 0.00000 // (op 1095) call()
v_and_b32 v22, -2, v1 // (op 3256) 
v_sub_u32 v23, v1, v22 // (op 3256) 

v_lshlrev_b32 v22, 5, v23 // (op 3256) 

v_lshl_add_u32 v23, s16, 6, v22 // (op 3256) 

v_mov_b32 v25, 0 // (op 3256) convert
v_mov_b32 v24, v23 // (op 3256) convert

v_lshrrev_b32 v22, 1, v1 // (op 3256) 
v_mul_lo_u32 v27, v22, s13 // (op 3256) most significant: low of low * high
v_mul_hi_u32 v23, v22, s12 // (op 3256) most significant: high of low * low
v_mul_lo_u32 v26, v22, s12 // (op 3256) least significant: low of low * low
v_add_u32 v27, v27, v23 // (op 3256) most significant: sum


v_add_co_u32 v22, s[0:1], v24, v26 // (op 3256) least significant half
v_addc_co_u32 v23, s[0:1], v25, v27, s[0:1] // (op 3256) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 3256) 


v_lshrrev_b64 v[24:25], 3, v[22:23] // (op 3256) 

v_mov_b32 v22, v24 // (op 3256) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 3258) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3258) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3258) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3258) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3258) most significant: sum
s_add_u32 s1, s1, s7 // (op 3258) most significant: sum


s_lshl_b64 s[14:15], s[0:1], 2 // (op 3258) 

s_ashr_i64 s[0:1], s[14:15], 3 // (op 3258) 

s_mov_b32 s8, s20 // (op 3258) reg expression
s_mov_b32 s9, s21 // (op 3258) reg expression
s_mov_b32 s10, s0 // (op 3258) reg expression
s_mov_b32 s11, 131072 // (op 3258) call()

v_and_b32 v23, -2, v1 // (op 3275) 
v_sub_u32 v24, v1, v23 // (op 3275) 

v_lshlrev_b32 v23, 5, v24 // (op 3275) 

v_lshl_add_u32 v24, s16, 6, v23 // (op 3275) 

v_mov_b32 v27, 0 // (op 3275) convert
v_mov_b32 v26, v24 // (op 3275) convert

v_lshrrev_b32 v23, 1, v1 // (op 3275) 
v_mul_lo_u32 v25, v23, s13 // (op 3275) most significant: low of low * high
v_mul_hi_u32 v28, v23, s12 // (op 3275) most significant: high of low * low
v_mul_lo_u32 v24, v23, s12 // (op 3275) least significant: low of low * low
v_add_u32 v25, v25, v28 // (op 3275) most significant: sum


v_add_co_u32 v28, s[0:1], v26, v24 // (op 3275) least significant half
v_addc_co_u32 v29, s[0:1], v27, v25, s[0:1] // (op 3275) most significant half

v_lshlrev_b64 v[28:29], 2, v[28:29] // (op 3275) 


v_lshrrev_b64 v[26:27], 3, v[28:29] // (op 3275) 

v_mov_b32 v23, v26 // (op 3275) convert

s_mul_i32 s1, s12, s23 // (op 3277) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3277) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3277) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3277) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3277) most significant: sum
s_add_u32 s1, s1, s7 // (op 3277) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3277) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3277) 

s_mov_b32 s8, s20 // (op 3277) reg expression
s_mov_b32 s9, s21 // (op 3277) reg expression
s_mov_b32 s10, s0 // (op 3277) reg expression
s_mov_b32 s11, 131072 // (op 3277) call()

v_and_b32 v24, -2, v1 // (op 3320) 
v_sub_u32 v25, v1, v24 // (op 3320) 

v_lshlrev_b32 v24, 5, v25 // (op 3320) 

v_lshl_add_u32 v25, s17, 6, v24 // (op 3320) 

v_mov_b32 v27, 0 // (op 3320) convert
v_mov_b32 v26, v25 // (op 3320) convert

v_lshrrev_b32 v24, 1, v1 // (op 3320) 
v_mul_lo_u32 v29, v24, s25 // (op 3320) most significant: low of low * high
v_mul_hi_u32 v25, v24, s24 // (op 3320) most significant: high of low * low
v_mul_lo_u32 v28, v24, s24 // (op 3320) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 3320) most significant: sum

v_add_co_u32 v28, s[0:1], v28, v26 // (op 3320) least significant half
v_addc_co_u32 v29, s[0:1], v29, v27, s[0:1] // (op 3320) most significant half



v_lshlrev_b64 v[24:25], 2, v[28:29] // (op 3320) 

v_lshrrev_b64 v[26:27], 3, v[24:25] // (op 3320) 

v_mov_b32 v24, v26 // (op 3320) convert

s_mul_i32 s1, s24, s23 // (op 3322) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 3322) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 3322) least significant
s_mul_hi_u32 s7, s24, s22 // (op 3322) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3322) most significant: sum
s_add_u32 s1, s1, s7 // (op 3322) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3322) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3322) 

s_mov_b32 s48, s26 // (op 3322) reg expression
s_mov_b32 s49, s27 // (op 3322) reg expression
s_mov_b32 s50, s0 // (op 3322) reg expression
s_mov_b32 s51, 131072 // (op 3322) call()

v_and_b32 v25, -2, v1 // (op 3339) 
v_sub_u32 v26, v1, v25 // (op 3339) 

v_lshlrev_b32 v25, 5, v26 // (op 3339) 

v_lshl_add_u32 v26, s17, 6, v25 // (op 3339) 

v_mov_b32 v29, 0 // (op 3339) convert
v_mov_b32 v28, v26 // (op 3339) convert

v_lshrrev_b32 v25, 1, v1 // (op 3339) 
v_mul_lo_u32 v27, v25, s25 // (op 3339) most significant: low of low * high
v_mul_hi_u32 v30, v25, s24 // (op 3339) most significant: high of low * low
v_mul_lo_u32 v26, v25, s24 // (op 3339) least significant: low of low * low
v_add_u32 v27, v27, v30 // (op 3339) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v28 // (op 3339) least significant half
v_addc_co_u32 v27, s[0:1], v27, v29, s[0:1] // (op 3339) most significant half



v_lshlrev_b64 v[28:29], 2, v[26:27] // (op 3339) 

v_lshrrev_b64 v[26:27], 3, v[28:29] // (op 3339) 

v_mov_b32 v25, v26 // (op 3339) convert

s_mul_i32 s1, s24, s23 // (op 3341) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 3341) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 3341) least significant
s_mul_hi_u32 s7, s24, s22 // (op 3341) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3341) most significant: sum
s_add_u32 s1, s1, s7 // (op 3341) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3341) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3341) 

s_mov_b32 s48, s26 // (op 3341) reg expression
s_mov_b32 s49, s27 // (op 3341) reg expression
s_mov_b32 s50, s0 // (op 3341) reg expression
s_mov_b32 s51, 131072 // (op 3341) call()

v_and_b32 v26, -64, v1 // (op 3384) 
v_sub_u32 v27, v1, v26 // (op 3384) 

v_lshl_add_u32 v26, s16, 6, v27 // (op 3384) 

v_lshrrev_b32 v27, 6, v1 // (op 3384) 
v_mul_lo_u32 v28, v27, s28 // (op 3384) 

v_add_u32 v27, v26, v28 // (op 3384) 


s_mul_i32 s1, s28, s33 // (op 3386) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3386) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3386) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3386) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3386) most significant: sum
s_add_u32 s1, s1, s7 // (op 3386) most significant: sum


s_mov_b32 s52, s30 // (op 3386) reg expression
s_mov_b32 s53, s31 // (op 3386) reg expression
s_mov_b32 s54, s0 // (op 3386) reg expression
s_mov_b32 s55, 131072 // (op 3386) call()

v_and_b32 v26, -64, v1 // (op 3403) 
v_sub_u32 v28, v1, v26 // (op 3403) 

v_lshl_add_u32 v26, s16, 6, v28 // (op 3403) 

v_lshrrev_b32 v28, 6, v1 // (op 3403) 
v_mul_lo_u32 v29, v28, s28 // (op 3403) 

v_add_u32 v28, v26, v29 // (op 3403) 


s_mul_i32 s1, s28, s33 // (op 3405) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3405) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3405) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3405) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3405) most significant: sum
s_add_u32 s1, s1, s7 // (op 3405) most significant: sum


s_mov_b32 s52, s30 // (op 3405) reg expression
s_mov_b32 s53, s31 // (op 3405) reg expression
s_mov_b32 s54, s0 // (op 3405) reg expression
s_mov_b32 s55, 131072 // (op 3405) call()

v_and_b32 v26, -64, v1 // (op 3448) 
v_sub_u32 v29, v1, v26 // (op 3448) 

v_lshl_add_u32 v26, s17, 6, v29 // (op 3448) 

v_lshrrev_b32 v29, 6, v1 // (op 3448) 
v_mul_lo_u32 v30, v29, s34 // (op 3448) 
v_add_u32 v30, v30, v26 // (op 3448) 


s_mul_i32 s1, s34, s33 // (op 3450) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 3450) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 3450) least significant
s_mul_hi_u32 s7, s34, s32 // (op 3450) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3450) most significant: sum
s_add_u32 s1, s1, s7 // (op 3450) most significant: sum


s_mov_b32 s56, s36 // (op 3450) reg expression
s_mov_b32 s57, s37 // (op 3450) reg expression
s_mov_b32 s58, s0 // (op 3450) reg expression
s_mov_b32 s59, 131072 // (op 3450) call()

v_and_b32 v26, -64, v1 // (op 3467) 
v_sub_u32 v29, v1, v26 // (op 3467) 

v_lshl_add_u32 v26, s17, 6, v29 // (op 3467) 

v_lshrrev_b32 v29, 6, v1 // (op 3467) 
v_mul_lo_u32 v31, v29, s34 // (op 3467) 
v_add_u32 v31, v31, v26 // (op 3467) 


s_mul_i32 s1, s34, s33 // (op 3469) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 3469) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 3469) least significant
s_mul_hi_u32 s7, s34, s32 // (op 3469) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3469) most significant: sum
s_add_u32 s1, s1, s7 // (op 3469) most significant: sum


s_mov_b32 s56, s36 // (op 3469) reg expression
s_mov_b32 s57, s37 // (op 3469) reg expression
s_mov_b32 s58, s0 // (op 3469) reg expression
s_mov_b32 s59, 131072 // (op 3469) call()



v_mov_b32 v26, v24 // (op 2968) reg expression

buffer_load_dwordx4 v[32:35], v26, s[48:51], 0 offen // (op 2968) Load value



v_mov_b32 v26, v30 // (op 2971) reg expression

buffer_load_ubyte v29, v26, s[56:59], 0 offen offset: 0 // (op 2971) Load value



v_mov_b32 v26, v22 // (op 2974) reg expression

buffer_load_dwordx4 v[36:39], v26, s[8:11], 0 offen // (op 2974) Load value



v_mov_b32 v26, v27 // (op 2977) reg expression

buffer_load_ubyte v40, v26, s[52:55], 0 offen offset: 0 // (op 2977) Load value



s_lshl_b64 s[0:1], s[24:25], 7 // (op 2980) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 2980) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 2980) 

v_add_u32 v26, v24, s0 // (op 2980) 


buffer_load_dwordx4 v[42:45], v26, s[48:51], 0 offen // (op 2980) Load value



s_lshl_b32 s0, s34, 2 // (op 2983) 
v_add_u32 v26, v30, s0 // (op 2983) 


buffer_load_ubyte v41, v26, s[56:59], 0 offen offset: 0 // (op 2983) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 2986) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 2986) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 2986) 

v_add_u32 v26, v22, s0 // (op 2986) 


buffer_load_dwordx4 v[46:49], v26, s[8:11], 0 offen // (op 2986) Load value



s_lshl_b32 s0, s28, 2 // (op 2989) 
v_add_u32 v26, v27, s0 // (op 2989) 


buffer_load_ubyte v50, v26, s[52:55], 0 offen offset: 0 // (op 2989) Load value



s_lshl_b64 s[0:1], s[24:25], 7 // (op 2992) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 2992) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 2992) 

s_lshl_b32 s6, s0, 1 // (op 2992) 

v_add_u32 v26, v24, s6 // (op 2992) 


buffer_load_dwordx4 v[52:55], v26, s[48:51], 0 offen // (op 2992) Load value



s_lshl_b32 s0, s34, 3 // (op 2995) 
v_add_u32 v26, v30, s0 // (op 2995) 


buffer_load_ubyte v51, v26, s[56:59], 0 offen offset: 0 // (op 2995) Load value



s_lshl_b64 s[0:1], s[12:13], 7 // (op 2998) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 2998) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 2998) 

s_lshl_b32 s6, s0, 1 // (op 2998) 

v_add_u32 v26, v22, s6 // (op 2998) 


buffer_load_dwordx4 v[56:59], v26, s[8:11], 0 offen // (op 2998) Load value



s_lshl_b32 s0, s28, 3 // (op 3001) 
v_add_u32 v26, v27, s0 // (op 3001) 


buffer_load_ubyte v60, v26, s[52:55], 0 offen offset: 0 // (op 3001) Load value



s_lshl_b64 s[0:1], s[24:25], 7 // (op 3004) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 3004) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 3004) 

s_mul_i32 s6, 3, s0 // (op 3004) 

v_add_u32 v26, v24, s6 // (op 3004) 


buffer_load_dwordx4 v[62:65], v26, s[48:51], 0 offen // (op 3004) Load value




s_lshl_b32 s0, s34, 2 // (op 3007) 
s_mul_i32 s1, 3, s0 // (op 3007) 

v_add_u32 v24, v30, s1 // (op 3007) 


buffer_load_ubyte v26, v24, s[56:59], 0 offen offset: 0 // (op 3007) Load value




s_lshl_b64 s[0:1], s[12:13], 7 // (op 3010) 
s_lshl_b64 s[6:7], s[0:1], 2 // (op 3010) 

s_lshr_b64 s[0:1], s[6:7], 3 // (op 3010) 

s_mul_i32 s6, 3, s0 // (op 3010) 

v_add_u32 v24, v22, s6 // (op 3010) 


buffer_load_dwordx4 v[66:69], v24, s[8:11], 0 offen // (op 3010) Load value




s_lshl_b32 s0, s28, 2 // (op 3013) 
s_mul_i32 s1, 3, s0 // (op 3013) 

v_add_u32 v22, v27, s1 // (op 3013) 


buffer_load_ubyte v24, v22, s[52:55], 0 offen offset: 0 // (op 3013) Load value


v_and_b32 v22, -2, v1 // (op 3597) 
v_sub_u32 v27, v1, v22 // (op 3597) 

v_lshlrev_b32 v22, 5, v27 // (op 3597) 

v_lshrrev_b32 v27, 1, v1 // (op 3597) 
v_lshl_add_u32 v30, v27, 6, v22 // (op 3597) 


v_lshlrev_b32 v22, 2, v30 // (op 3597) 

v_lshrrev_b32 v27, 3, v22 // (op 3597) 




s_waitcnt vmcnt(15)
ds_write_b128 v27, v[32:35] // (op 3015) Store local data 

v_and_b32 v22, -64, v1 // (op 3713) 
v_sub_u32 v27, v1, v22 // (op 3713) 

v_and_b32 v22, 4294967232, v1 // (op 3713) 
v_add_u32 v30, v22, v27 // (op 3713) 





s_waitcnt vmcnt(14)
ds_write_b8 v30, v29 offset:4096 // (op 3016) Store local data 

v_and_b32 v22, -2, v1 // (op 3539) 
v_sub_u32 v27, v1, v22 // (op 3539) 

v_lshlrev_b32 v22, 5, v27 // (op 3539) 

v_lshrrev_b32 v27, 1, v1 // (op 3539) 
v_lshl_add_u32 v30, v27, 6, v22 // (op 3539) 


v_lshlrev_b32 v22, 2, v30 // (op 3539) 

v_lshrrev_b32 v27, 3, v22 // (op 3539) 




s_waitcnt vmcnt(13)
ds_write_b128 v27, v[36:39] offset:4352 // (op 3017) Store local data 

v_and_b32 v22, -64, v1 // (op 3655) 
v_sub_u32 v27, v1, v22 // (op 3655) 

v_and_b32 v22, 4294967232, v1 // (op 3655) 
v_add_u32 v30, v22, v27 // (op 3655) 





s_waitcnt vmcnt(12)
ds_write_b8 v30, v40 offset:8448 // (op 3018) Store local data 

v_and_b32 v22, -2, v1 // (op 3510) 
v_sub_u32 v27, v1, v22 // (op 3510) 

v_lshlrev_b32 v22, 5, v27 // (op 3510) 

v_lshrrev_b32 v27, 1, v1 // (op 3510) 
v_lshl_add_u32 v30, v27, 6, v22 // (op 3510) 


v_lshlrev_b32 v22, 2, v30 // (op 3510) 

v_lshrrev_b32 v27, 3, v22 // (op 3510) 

v_and_b32 v22, -2, v1 // (op 3568) 
v_sub_u32 v30, v1, v22 // (op 3568) 

v_lshlrev_b32 v22, 5, v30 // (op 3568) 

v_lshrrev_b32 v30, 1, v1 // (op 3568) 
v_lshl_add_u32 v61, v30, 6, v22 // (op 3568) 


v_lshlrev_b32 v22, 2, v61 // (op 3568) 

v_lshrrev_b32 v30, 3, v22 // (op 3568) 

v_and_b32 v22, -64, v1 // (op 3626) 
v_sub_u32 v61, v1, v22 // (op 3626) 

v_and_b32 v22, 4294967232, v1 // (op 3626) 
v_add_u32 v70, v22, v61 // (op 3626) 


v_and_b32 v22, -64, v1 // (op 3684) 
v_sub_u32 v61, v1, v22 // (op 3684) 

v_and_b32 v22, 4294967232, v1 // (op 3684) 
v_add_u32 v71, v22, v61 // (op 3684) 


v_and_b32 v22, -2, v1 // (op 3765) 
v_sub_u32 v61, v1, v22 // (op 3765) 

v_lshlrev_b32 v22, 5, v61 // (op 3765) 

v_lshrrev_b32 v61, 1, v1 // (op 3765) 
v_lshl_add_u32 v72, v61, 6, v22 // (op 3765) 


v_lshlrev_b32 v22, 2, v72 // (op 3765) 

v_lshrrev_b32 v61, 3, v22 // (op 3765) 

v_and_b32 v22, -2, v1 // (op 3794) 
v_sub_u32 v72, v1, v22 // (op 3794) 

v_lshlrev_b32 v22, 5, v72 // (op 3794) 

v_lshrrev_b32 v72, 1, v1 // (op 3794) 
v_lshl_add_u32 v73, v72, 6, v22 // (op 3794) 


v_lshlrev_b32 v22, 2, v73 // (op 3794) 

v_lshrrev_b32 v72, 3, v22 // (op 3794) 

v_and_b32 v22, -64, v1 // (op 3823) 
v_sub_u32 v73, v1, v22 // (op 3823) 

v_and_b32 v22, 4294967232, v1 // (op 3823) 
v_add_u32 v74, v22, v73 // (op 3823) 


v_and_b32 v22, -64, v1 // (op 3852) 
v_sub_u32 v73, v1, v22 // (op 3852) 

v_and_b32 v22, 4294967232, v1 // (op 3852) 
v_add_u32 v75, v22, v73 // (op 3852) 


v_and_b32 v22, -2, v1 // (op 3881) 
v_sub_u32 v73, v1, v22 // (op 3881) 

v_lshlrev_b32 v22, 5, v73 // (op 3881) 

v_lshrrev_b32 v73, 1, v1 // (op 3881) 
v_lshl_add_u32 v76, v73, 6, v22 // (op 3881) 


v_lshlrev_b32 v22, 2, v76 // (op 3881) 

v_lshrrev_b32 v73, 3, v22 // (op 3881) 

v_and_b32 v22, -2, v1 // (op 3910) 
v_sub_u32 v76, v1, v22 // (op 3910) 

v_lshlrev_b32 v22, 5, v76 // (op 3910) 

v_lshrrev_b32 v76, 1, v1 // (op 3910) 
v_lshl_add_u32 v77, v76, 6, v22 // (op 3910) 


v_lshlrev_b32 v22, 2, v77 // (op 3910) 

v_lshrrev_b32 v76, 3, v22 // (op 3910) 

v_and_b32 v22, -64, v1 // (op 3939) 
v_sub_u32 v77, v1, v22 // (op 3939) 

v_and_b32 v22, 4294967232, v1 // (op 3939) 
v_add_u32 v78, v22, v77 // (op 3939) 


v_and_b32 v22, -64, v1 // (op 3968) 
v_sub_u32 v77, v1, v22 // (op 3968) 

v_and_b32 v22, 4294967232, v1 // (op 3968) 
v_add_u32 v79, v22, v77 // (op 3968) 


v_and_b32 v22, -2, v1 // (op 3997) 
v_sub_u32 v77, v1, v22 // (op 3997) 

v_lshlrev_b32 v22, 5, v77 // (op 3997) 

v_lshrrev_b32 v77, 1, v1 // (op 3997) 
v_lshl_add_u32 v80, v77, 6, v22 // (op 3997) 


v_lshlrev_b32 v22, 2, v80 // (op 3997) 

v_lshrrev_b32 v77, 3, v22 // (op 3997) 

v_and_b32 v22, -2, v1 // (op 4026) 
v_sub_u32 v80, v1, v22 // (op 4026) 

v_lshlrev_b32 v22, 5, v80 // (op 4026) 

v_lshrrev_b32 v80, 1, v1 // (op 4026) 
v_lshl_add_u32 v81, v80, 6, v22 // (op 4026) 


v_lshlrev_b32 v22, 2, v81 // (op 4026) 

v_lshrrev_b32 v80, 3, v22 // (op 4026) 

v_and_b32 v22, -64, v1 // (op 4055) 
v_sub_u32 v81, v1, v22 // (op 4055) 

v_and_b32 v22, 4294967232, v1 // (op 4055) 
v_add_u32 v82, v22, v81 // (op 4055) 


v_and_b32 v22, -64, v1 // (op 4084) 
v_sub_u32 v81, v1, v22 // (op 4084) 

v_and_b32 v22, 4294967232, v1 // (op 4084) 
v_add_u32 v83, v22, v81 // (op 4084) 


s_mov_b32 s0, 0 // (op 881) call()
s_ashr_i64 s[6:7], s[22:23], 7 // (op 880) 
s_mov_b32 s1, s6 // (op 880) convert

s_and_b32 s6, s1, 4294967292 // (op 880) 

s_cmp_lt_u32 s0, s6 // (op 880) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoop_880 // (op 880) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoop_880 if false)
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoop_880:


s_add_u32 s0, s0, 4 // (op 882) 
v_mov_b32 v22, v25 // (op 10) 
s_lshl_b64 s[6:7], s[24:25], 7 // (op 10) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 10) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 10) 

s_lshl_b32 s1, s6, 2 // (op 10) 

v_add_u32 v81, v22, s1 // (op 10) 


buffer_load_dwordx4 v[32:35], v81, s[48:51], 0 offen // (op 10) Load value

s_waitcnt lgkmcnt(1)
s_barrier  // (op 4184) 





v_mov_b32 v81, v31 // (op 25) 
s_lshl_b32 s1, s34, 4 // (op 25) 
v_add_u32 v84, v81, s1 // (op 25) 


buffer_load_ubyte v29, v84, s[56:59], 0 offen offset: 0 // (op 25) Load value

ds_read_b64_tr_b4 v[84:85], v3 offset:4352 // (op 70) Transpose load local data 


ds_read_b64_tr_b4 v[86:87], v3 offset:4864 // (op 70) Transpose load local data 
v_mov_b32 v88, v23 // (op 4) 
s_barrier  // (op 4197) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 4) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 4) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 4) 

s_lshl_b32 s1, s6, 2 // (op 4) 

v_add_u32 v89, v88, s1 // (op 4) 


ds_read_b64_tr_b4 v[90:91], v4 // (op 75) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v4 offset:512 // (op 75) Transpose load local data 
buffer_load_dwordx4 v[36:39], v89, s[8:11], 0 offen // (op 4) Load value



v_mov_b32 v89, v28 // (op 17) 
s_lshl_b32 s1, s28, 4 // (op 17) 
v_add_u32 v94, v89, s1 // (op 17) 


s_waitcnt lgkmcnt(4)
s_barrier  // (op 4211) 
buffer_load_ubyte v40, v94, s[52:55], 0 offen offset: 0 // (op 17) Load value



s_barrier  // (op 4248) 
ds_read_u8 v94, v0 offset:8448 // (op 80) Load local data 



s_barrier  // (op 4225) 
s_waitcnt vmcnt(15)
ds_write_b128 v72, v[42:45] offset:8704 // (op 253) Store local data 


ds_read_u8 v95, v5 offset:4096 // (op 85) Load local data 



s_barrier  // (op 4260) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[90:93], a[0:3], v94, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 88) 



ds_read_b64_tr_b4 v[96:97], v4 offset:16 // (op 517) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v4 offset:528 // (op 517) Transpose load local data 
s_waitcnt vmcnt(14)
ds_write_b8 v75, v41 offset:12800 // (op 255) Store local data 


ds_read_u8 v100, v5 offset:4128 // (op 521) Load local data 



s_barrier  // (op 4242) 
ds_read_b64_tr_b4 v[102:103], v3 offset:4368 // (op 914) Transpose load local data 



ds_read_b64_tr_b4 v[104:105], v3 offset:4880 // (op 914) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:87], v[96:99], a[4:7], v94, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 522) 




ds_read_u8 v94, v0 offset:8480 // (op 917) Load local data 
s_waitcnt vmcnt(13)
ds_write_b128 v61, v[46:49] offset:13056 // (op 243) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:105], v[90:93], a[8:11], v94, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1011) 


s_barrier  // (op 4254) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:105], v[96:99], a[12:15], v94, v100 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 919) 
s_waitcnt vmcnt(12)
ds_write_b8 v74, v50 offset:17152 // (op 254) Store local data 






s_lshl_b64 s[6:7], s[24:25], 7 // (op 256) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 256) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 256) 

s_mul_i32 s1, 5, s6 // (op 256) 

v_add_u32 v84, v22, s1 // (op 256) 


buffer_load_dwordx4 v[42:45], v84, s[48:51], 0 offen // (op 256) Load value

s_waitcnt lgkmcnt(1)
s_barrier  // (op 4239) 





s_lshl_b32 s1, s34, 2 // (op 258) 
s_mul_i32 s6, 5, s1 // (op 258) 

v_add_u32 v84, v81, s6 // (op 258) 


ds_read_b64_tr_b4 v[90:91], v7 offset:13056 // (op 245) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v7 offset:13568 // (op 245) Transpose load local data 
buffer_load_ubyte v41, v84, s[56:59], 0 offen offset: 0 // (op 258) Load value



s_barrier  // (op 4245) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 242) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 242) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 242) 

s_mul_i32 s1, 5, s6 // (op 242) 

v_add_u32 v84, v88, s1 // (op 242) 


ds_read_b64_tr_b4 v[94:95], v8 offset:8704 // (op 247) Transpose load local data 
ds_read_b64_tr_b4 v[96:97], v8 offset:9216 // (op 247) Transpose load local data 
buffer_load_dwordx4 v[46:49], v84, s[8:11], 0 offen // (op 242) Load value



s_lshl_b32 s1, s28, 2 // (op 257) 
s_mul_i32 s6, 5, s1 // (op 257) 

v_add_u32 v84, v89, s6 // (op 257) 


s_waitcnt lgkmcnt(4)
s_barrier  // (op 4251) 
buffer_load_ubyte v50, v84, s[52:55], 0 offen offset: 0 // (op 257) Load value



s_barrier  // (op 4272) 
ds_read_u8 v84, v6 offset:17152 // (op 249) Load local data 



s_barrier  // (op 4257) 
s_waitcnt vmcnt(15)
ds_write_b128 v76, v[52:55] offset:17408 // (op 293) Store local data 


ds_read_u8 v85, v9 offset:12800 // (op 251) Load local data 



s_barrier  // (op 4284) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[94:97], a[0:3], v84, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 252) 



ds_read_b64_tr_b4 v[98:99], v8 offset:8720 // (op 529) Transpose load local data 
ds_read_b64_tr_b4 v[100:101], v8 offset:9232 // (op 529) Transpose load local data 
s_waitcnt vmcnt(14)
ds_write_b8 v79, v51 offset:21504 // (op 295) Store local data 


s_barrier  // (op 4385) 
s_barrier  // (op 4266) 
ds_read_u8 v86, v9 offset:12832 // (op 533) Load local data 



s_waitcnt vmcnt(13)
ds_write_b128 v73, v[56:59] offset:21760 // (op 283) Store local data 



s_barrier  // (op 4389) 
s_barrier  // (op 4278) 
ds_read_b64_tr_b4 v[102:103], v7 offset:13072 // (op 924) Transpose load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:93], v[98:101], a[4:7], v84, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 523) 
ds_read_b64_tr_b4 v[104:105], v7 offset:13584 // (op 924) Transpose load local data 


s_waitcnt vmcnt(12)
ds_write_b8 v78, v60 offset:25856 // (op 294) Store local data 


s_barrier  // (op 4429) 
ds_read_u8 v84, v6 offset:17184 // (op 927) Load local data 
s_barrier  // (op 4433) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:105], v[94:97], a[8:11], v84, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1012) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:105], v[98:101], a[12:15], v84, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 920) 






s_barrier  // (op 4263) 
s_lshl_b64 s[6:7], s[24:25], 7 // (op 296) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 296) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 296) 

s_mul_i32 s1, 6, s6 // (op 296) 

v_add_u32 v84, v22, s1 // (op 296) 


ds_read_b64_tr_b4 v[90:91], v11 offset:21760 // (op 285) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v11 offset:22272 // (op 285) Transpose load local data 
buffer_load_dwordx4 v[52:55], v84, s[48:51], 0 offen // (op 296) Load value



s_barrier  // (op 4269) 
s_lshl_b32 s1, s34, 2 // (op 298) 



s_mul_i32 s6, 6, s1 // (op 298) 

v_add_u32 v84, v81, s6 // (op 298) 


ds_read_b64_tr_b4 v[94:95], v12 offset:17408 // (op 287) Transpose load local data 
ds_read_b64_tr_b4 v[96:97], v12 offset:17920 // (op 287) Transpose load local data 
buffer_load_ubyte v51, v84, s[56:59], 0 offen offset: 0 // (op 298) Load value



s_barrier  // (op 4275) 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 282) 


ds_read_u8 v84, v10 offset:25856 // (op 289) Load local data 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 282) 

s_barrier  // (op 4281) 
s_lshr_b64 s[6:7], s[14:15], 3 // (op 282) 



ds_read_u8 v85, v13 offset:21504 // (op 291) Load local data 
s_mul_i32 s1, 6, s6 // (op 282) 




v_add_u32 v86, v88, s1 // (op 282) 


ds_read_b64_tr_b4 v[98:99], v12 offset:17424 // (op 537) Transpose load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[94:97], a[0:3], v84, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 292) 
ds_read_b64_tr_b4 v[100:101], v12 offset:17936 // (op 537) Transpose load local data 
buffer_load_dwordx4 v[56:59], v86, s[8:11], 0 offen // (op 282) Load value





s_barrier  // (op 4393) 
ds_read_u8 v86, v13 offset:21536 // (op 541) Load local data 
s_lshl_b32 s1, s28, 2 // (op 297) 
s_mul_i32 s6, 6, s1 // (op 297) 




s_barrier  // (op 4397) 
v_add_u32 v87, v89, s6 // (op 297) 


ds_read_b64_tr_b4 v[102:103], v11 offset:21776 // (op 930) Transpose load local data 
ds_read_b64_tr_b4 v[104:105], v11 offset:22288 // (op 930) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:93], v[98:101], a[4:7], v84, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 524) 




s_barrier  // (op 4437) 
ds_read_u8 v84, v10 offset:25888 // (op 933) Load local data 
buffer_load_ubyte v60, v87, s[52:55], 0 offen offset: 0 // (op 297) Load value

s_barrier  // (op 4296) 
s_barrier  // (op 4441) 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[102:105], v[94:97], a[8:11], v84, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1013) 
s_waitcnt vmcnt(15)
ds_write_b128 v80, v[62:65] offset:8704 // (op 333) Store local data 


s_barrier  // (op 4308) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[102:105], v[98:101], a[12:15], v84, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 921) 







s_waitcnt vmcnt(14)
ds_write_b8 v83, v26 offset:12800 // (op 335) Store local data 
s_barrier  // (op 4290) 



s_waitcnt vmcnt(13)
ds_write_b128 v77, v[66:69] offset:13056 // (op 323) Store local data 
s_barrier  // (op 4302) 



s_waitcnt vmcnt(12)
ds_write_b8 v82, v24 offset:17152 // (op 334) Store local data 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 336) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 336) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 336) 

s_mul_i32 s1, 7, s6 // (op 336) 

v_add_u32 v84, v22, s1 // (op 336) 


s_waitcnt lgkmcnt(1)
s_barrier  // (op 4287) 
buffer_load_dwordx4 v[62:65], v84, s[48:51], 0 offen // (op 336) Load value





ds_read_b64_tr_b4 v[84:85], v15 offset:13056 // (op 325) Transpose load local data 


ds_read_b64_tr_b4 v[86:87], v15 offset:13568 // (op 325) Transpose load local data 
s_lshl_b32 s1, s34, 2 // (op 338) 
s_barrier  // (op 4293) 
s_mul_i32 s6, 7, s1 // (op 338) 




v_add_u32 v22, v81, s6 // (op 338) 


s_lshl_b64 s[6:7], s[24:25], 7 // (op 3331) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 3331) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 3331) 

s_lshl_b32 s1, s6, 2 // (op 3331) 

v_add_u32 v25, v25, s1 // (op 3331) 

ds_read_b64_tr_b4 v[90:91], v16 offset:8704 // (op 327) Transpose load local data 
ds_read_b64_tr_b4 v[92:93], v16 offset:9216 // (op 327) Transpose load local data 
buffer_load_ubyte v26, v22, s[56:59], 0 offen offset: 0 // (op 338) Load value


s_waitcnt lgkmcnt(4)
s_barrier  // (op 4299) 




ds_read_u8 v22, v14 offset:17152 // (op 329) Load local data 
s_lshl_b32 s1, s34, 4 // (op 3459) 
s_barrier  // (op 4305) 
v_add_u32 v31, v31, s1 // (op 3459) 



ds_read_u8 v81, v17 offset:12800 // (op 331) Load local data 
s_lshl_b64 s[6:7], s[12:13], 7 // (op 322) 



s_lshl_b64 s[14:15], s[6:7], 2 // (op 322) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 322) 

s_mul_i32 s1, 7, s6 // (op 322) 

v_add_u32 v94, v88, s1 // (op 322) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[84:87], v[90:93], a[0:3], v22, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 332) 
ds_read_b64_tr_b4 v[96:97], v16 offset:8720 // (op 545) Transpose load local data 
ds_read_b64_tr_b4 v[98:99], v16 offset:9232 // (op 545) Transpose load local data 
buffer_load_dwordx4 v[66:69], v94, s[8:11], 0 offen // (op 322) Load value






s_barrier  // (op 4401) 
ds_read_u8 v88, v17 offset:12832 // (op 549) Load local data 
s_lshl_b32 s1, s28, 2 // (op 337) 
s_mul_i32 s6, 7, s1 // (op 337) 




s_barrier  // (op 4405) 
v_add_u32 v94, v89, s6 // (op 337) 


s_lshl_b64 s[6:7], s[12:13], 7 // (op 3267) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 3267) 

s_lshr_b64 s[6:7], s[14:15], 3 // (op 3267) 

s_lshl_b32 s1, s6, 2 // (op 3267) 

v_add_u32 v23, v23, s1 // (op 3267) 

ds_read_b64_tr_b4 v[100:101], v15 offset:13072 // (op 936) Transpose load local data 
ds_read_b64_tr_b4 v[102:103], v15 offset:13584 // (op 936) Transpose load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[84:87], v[96:99], a[4:7], v22, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 525) 




s_barrier  // (op 4445) 
ds_read_u8 v22, v14 offset:17184 // (op 939) Load local data 
buffer_load_ubyte v24, v94, s[52:55], 0 offen offset: 0 // (op 337) Load value


s_barrier  // (op 4449) 
s_barrier  // (op 4200) 
s_lshl_b32 s1, s28, 4 // (op 3395) 



v_add_u32 v28, v28, s1 // (op 3395) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:103], v[90:93], a[8:11], v22, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1014) 
s_waitcnt vmcnt(15)
ds_write_b128 v30, v[32:35] // (op 13) Store local data 


s_barrier  // (op 4228) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:103], v[96:99], a[12:15], v22, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 922) 







s_waitcnt vmcnt(14)
ds_write_b8 v71, v29 offset:4096 // (op 30) Store local data 
s_barrier  // (op 4187) 



s_waitcnt vmcnt(13)
ds_write_b128 v27, v[36:39] offset:4352 // (op 6) Store local data 
s_barrier  // (op 4214) 



s_waitcnt vmcnt(12)
ds_write_b8 v70, v40 offset:8448 // (op 21) Store local data 
s_ashr_i64 s[6:7], s[22:23], 7 // (op 880) 
s_mov_b32 s1, s6 // (op 880) convert

s_and_b32 s6, s1, 4294967292 // (op 880) 

s_cmp_lt_u32 s0, s6 // (op 880) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoop_880 // (op 880) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoop_880 if true)
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoop_880:

























v_and_b32 v25, -2, v1 // (op 3227) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3227) 
v_sub_u32 v61, v1, v25 // (op 3227) 

s_lshr_b32 s6, s0, 2 // (op 3227) 

v_lshlrev_b32 v25, 5, v61 // (op 3227) 

v_lshl_add_u32 v61, s16, 6, v25 // (op 3227) 

v_mov_b32 v23, 0 // (op 3227) convert
v_mov_b32 v22, v61 // (op 3227) convert

v_lshrrev_b32 v25, 1, v1 // (op 3227) 
v_lshl_add_u32 v61, s6, 9, v25 // (op 3227) 


v_mul_lo_u32 v29, v61, s13 // (op 3227) most significant: low of low * high
v_mul_hi_u32 v25, v61, s12 // (op 3227) most significant: high of low * low
v_mul_lo_u32 v28, v61, s12 // (op 3227) least significant: low of low * low
v_add_u32 v29, v29, v25 // (op 3227) most significant: sum


v_add_co_u32 v30, s[0:1], v22, v28 // (op 3227) least significant half
v_addc_co_u32 v31, s[0:1], v23, v29, s[0:1] // (op 3227) most significant half

v_lshlrev_b64 v[30:31], 2, v[30:31] // (op 3227) 


v_lshrrev_b64 v[22:23], 3, v[30:31] // (op 3227) 

v_mov_b32 v25, v22 // (op 3227) convert

s_mul_i32 s1, s12, s23 // (op 3229) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 3229) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 3229) least significant
s_mul_hi_u32 s7, s12, s22 // (op 3229) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3229) most significant: sum
s_add_u32 s1, s1, s7 // (op 3229) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3229) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3229) 

s_mov_b32 s8, s20 // (op 3229) reg expression
s_mov_b32 s9, s21 // (op 3229) reg expression
s_mov_b32 s10, s0 // (op 3229) reg expression
s_mov_b32 s11, 131072 // (op 3229) call()

v_lshrrev_b32 v61, 1, v1 // (op 3297) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3297) 
v_and_b32 v22, -2, v1 // (op 3297) 
s_lshr_b32 s6, s0, 2 // (op 3297) 

v_sub_u32 v23, v1, v22 // (op 3297) 

v_lshl_add_u32 v22, s6, 9, v61 // (op 3297) 


v_lshlrev_b32 v61, 5, v23 // (op 3297) 

v_lshl_add_u32 v23, s17, 6, v61 // (op 3297) 

v_mov_b32 v29, 0 // (op 3297) convert
v_mov_b32 v28, v23 // (op 3297) convert

v_mul_lo_u32 v31, v22, s25 // (op 3297) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 3297) most significant: high of low * low
v_mul_lo_u32 v30, v22, s24 // (op 3297) least significant: low of low * low
v_add_u32 v31, v31, v23 // (op 3297) most significant: sum

v_add_co_u32 v30, s[0:1], v30, v28 // (op 3297) least significant half
v_addc_co_u32 v31, s[0:1], v31, v29, s[0:1] // (op 3297) most significant half



v_lshlrev_b64 v[22:23], 2, v[30:31] // (op 3297) 

v_lshrrev_b64 v[28:29], 3, v[22:23] // (op 3297) 

v_mov_b32 v27, v28 // (op 3297) convert

s_mul_i32 s1, s24, s23 // (op 3299) most significant: low of low * high
s_mul_i32 s6, s25, s22 // (op 3299) most significant: low of high * low
s_mul_i32 s0, s24, s22 // (op 3299) least significant
s_mul_hi_u32 s7, s24, s22 // (op 3299) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3299) most significant: sum
s_add_u32 s1, s1, s7 // (op 3299) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 3299) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 3299) 

s_mov_b32 s48, s26 // (op 3299) reg expression
s_mov_b32 s49, s27 // (op 3299) reg expression
s_mov_b32 s50, s0 // (op 3299) reg expression
s_mov_b32 s51, 131072 // (op 3299) call()

v_and_b32 v61, -64, v1 // (op 3361) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3361) 
v_sub_u32 v22, v1, v61 // (op 3361) 

s_lshr_b32 s6, s0, 2 // (op 3361) 

v_lshl_add_u32 v23, s16, 6, v22 // (op 3361) 

v_lshrrev_b32 v22, 6, v1 // (op 3361) 
v_lshl_add_u32 v61, s6, 4, v22 // (op 3361) 


v_mul_lo_u32 v22, v61, s28 // (op 3361) 

v_add_u32 v61, v23, v22 // (op 3361) 


s_mul_i32 s1, s28, s33 // (op 3363) most significant: low of low * high
s_mul_i32 s6, s29, s32 // (op 3363) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 3363) least significant
s_mul_hi_u32 s7, s28, s32 // (op 3363) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3363) most significant: sum
s_add_u32 s1, s1, s7 // (op 3363) most significant: sum


s_mov_b32 s52, s30 // (op 3363) reg expression
s_mov_b32 s53, s31 // (op 3363) reg expression
s_mov_b32 s54, s0 // (op 3363) reg expression
s_mov_b32 s55, 131072 // (op 3363) call()

v_lshrrev_b32 v22, 6, v1 // (op 3425) 
s_ashr_i64 s[0:1], s[22:23], 7 // (op 3425) 
v_and_b32 v23, -64, v1 // (op 3425) 
s_lshr_b32 s6, s0, 2 // (op 3425) 

v_sub_u32 v28, v1, v23 // (op 3425) 

v_lshl_add_u32 v23, s6, 4, v22 // (op 3425) 


v_lshl_add_u32 v22, s17, 6, v28 // (op 3425) 

v_mul_lo_u32 v28, v23, s34 // (op 3425) 
v_add_u32 v28, v28, v22 // (op 3425) 


s_mul_i32 s1, s34, s33 // (op 3427) most significant: low of low * high
s_mul_i32 s6, s35, s32 // (op 3427) most significant: low of high * low
s_mul_i32 s0, s34, s32 // (op 3427) least significant
s_mul_hi_u32 s7, s34, s32 // (op 3427) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 3427) most significant: sum
s_add_u32 s1, s1, s7 // (op 3427) most significant: sum


s_mov_b32 s56, s36 // (op 3427) reg expression
s_mov_b32 s57, s37 // (op 3427) reg expression
s_mov_b32 s58, s0 // (op 3427) reg expression
s_mov_b32 s59, 131072 // (op 3427) call()

v_and_b32 v22, -2, v1 // (op 3523) 
v_sub_u32 v23, v1, v22 // (op 3523) 

v_lshlrev_b32 v22, 5, v23 // (op 3523) 

v_lshrrev_b32 v23, 1, v1 // (op 3523) 
v_lshl_add_u32 v29, v23, 6, v22 // (op 3523) 


v_lshlrev_b32 v22, 2, v29 // (op 3523) 

v_lshrrev_b32 v23, 3, v22 // (op 3523) 

v_and_b32 v22, -2, v1 // (op 3581) 
v_sub_u32 v29, v1, v22 // (op 3581) 

v_lshlrev_b32 v22, 5, v29 // (op 3581) 

v_lshrrev_b32 v29, 1, v1 // (op 3581) 
v_lshl_add_u32 v30, v29, 6, v22 // (op 3581) 


v_lshlrev_b32 v22, 2, v30 // (op 3581) 

v_lshrrev_b32 v29, 3, v22 // (op 3581) 

v_and_b32 v22, -64, v1 // (op 3639) 
v_sub_u32 v30, v1, v22 // (op 3639) 

v_and_b32 v22, 4294967232, v1 // (op 3639) 
v_add_u32 v31, v22, v30 // (op 3639) 


v_and_b32 v22, -64, v1 // (op 3697) 
v_sub_u32 v30, v1, v22 // (op 3697) 

v_and_b32 v22, 4294967232, v1 // (op 3697) 
v_add_u32 v32, v22, v30 // (op 3697) 


s_ashr_i64 s[0:1], s[22:23], 7 // (op 886) 
s_and_b32 s6, s0, 4294967292 // (op 886) 

s_ashr_i64 s[0:1], s[22:23], 7 // (op 885) 
s_mov_b32 s7, s0 // (op 885) convert

s_cmp_lt_u32 s6, s7 // (op 885) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoopTail_885 // (op 885) Condition: Top (jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoopTail_885 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoopTail_885:


s_add_u32 s6, s6, 1 // (op 887) 
v_mov_b32 v22, v25 // (op 176) 
v_mov_b32 v24, v22 // (op 176) reg expression

buffer_load_dwordx4 v[34:37], v24, s[8:11], 0 offen // (op 176) Load value


s_lshl_b64 s[0:1], s[12:13], 7 // (op 3217) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 3217) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 3217) 

v_add_u32 v25, v25, s0 // (op 3217) 



s_barrier  // (op 4190) 
v_mov_b32 v22, v27 // (op 190) 



v_mov_b32 v24, v22 // (op 190) reg expression

buffer_load_dwordx4 v[38:41], v24, s[48:51], 0 offen // (op 190) Load value


s_lshl_b64 s[0:1], s[24:25], 7 // (op 3289) 
s_lshl_b64 s[14:15], s[0:1], 2 // (op 3289) 

s_lshr_b64 s[0:1], s[14:15], 3 // (op 3289) 

v_add_u32 v27, v27, s0 // (op 3289) 

s_waitcnt vmcnt(1)
ds_write_b128 v23, v[34:37] offset:4352 // (op 177) Store local data 



v_mov_b32 v22, v61 // (op 191) 
s_barrier  // (op 4203) 
v_mov_b32 v24, v22 // (op 191) reg expression




buffer_load_ubyte v26, v24, s[52:55], 0 offen offset: 0 // (op 191) Load value


s_lshl_b32 s7, s28, 2 // (op 3353) 
v_add_u32 v61, v61, s7 // (op 3353) 

s_waitcnt vmcnt(1)
ds_write_b128 v29, v[38:41] // (op 187) Store local data 



v_mov_b32 v22, v28 // (op 192) 
s_barrier  // (op 4217) 
v_mov_b32 v24, v22 // (op 192) reg expression




buffer_load_ubyte v30, v24, s[56:59], 0 offen offset: 0 // (op 192) Load value


s_lshl_b32 s7, s34, 2 // (op 3417) 
v_add_u32 v28, v28, s7 // (op 3417) 

s_waitcnt vmcnt(1)
ds_write_b8 v31, v26 offset:8448 // (op 188) Store local data 

s_waitcnt lgkmcnt(2)
s_barrier  // (op 4193) 



s_barrier  // (op 4231) 
ds_read_b64_tr_b4 v[34:35], v3 offset:4352 // (op 179) Transpose load local data 



ds_read_b64_tr_b4 v[36:37], v3 offset:4864 // (op 179) Transpose load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v32, v30 offset:4096 // (op 189) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 4207) 



ds_read_b64_tr_b4 v[38:39], v4 // (op 181) Transpose load local data 
ds_read_b64_tr_b4 v[40:41], v4 offset:512 // (op 181) Transpose load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 4221) 


ds_read_u8 v22, v0 offset:8448 // (op 183) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 4235) 


ds_read_u8 v24, v5 offset:4096 // (op 185) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[38:41], a[0:3], v22, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 186) 
ds_read_b64_tr_b4 v[42:43], v4 offset:16 // (op 496) Transpose load local data 
ds_read_b64_tr_b4 v[44:45], v4 offset:528 // (op 496) Transpose load local data 


ds_read_u8 v26, v5 offset:4128 // (op 500) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[42:45], a[4:7], v22, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 501) 
ds_read_b64_tr_b4 v[46:47], v3 offset:4368 // (op 860) Transpose load local data 


ds_read_b64_tr_b4 v[48:49], v3 offset:4880 // (op 860) Transpose load local data 


ds_read_u8 v22, v0 offset:8480 // (op 863) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[38:41], a[8:11], v22, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 865) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[42:45], a[12:15], v22, v26 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 896) 




s_ashr_i64 s[0:1], s[22:23], 7 // (op 885) 
s_mov_b32 s7, s0 // (op 885) convert

s_cmp_lt_u32 s6, s7 // (op 885) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoopTail_885 // (op 885) Condition: Bottom (jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopTop_KLoopTail_885 if true)
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ForLoopBottom_KLoopTail_885:























s_barrier  // (op 4413) 
















v_cmp_eq_f32 vcc, 0.00000, v18 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalFalse_0_beta_133
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


v_accvgpr_read v14, a4 // (op 1617) 
v_accvgpr_read v15, a5 // (op 1617) 
v_accvgpr_read v16, a6 // (op 1617) 
v_accvgpr_read v17, a7 // (op 1617) 
v_mul_f32 v10, v19, v14 // (op 1617) 
v_mul_f32 v11, v19, v15 // (op 1617) 
v_mul_f32 v12, v19, v16 // (op 1617) 
v_mul_f32 v13, v19, v17 // (op 1617) 



s_lshl_b32 s0, s4, 7 // (op 1619) 
v_add_u32 v0, v21, s0 // (op 1619) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 1619) Store value


v_accvgpr_read v14, a8 // (op 1641) 
v_accvgpr_read v15, a9 // (op 1641) 
v_accvgpr_read v16, a10 // (op 1641) 
v_accvgpr_read v17, a11 // (op 1641) 
v_mul_f32 v10, v19, v14 // (op 1641) 
v_mul_f32 v11, v19, v15 // (op 1641) 
v_mul_f32 v12, v19, v16 // (op 1641) 
v_mul_f32 v13, v19, v17 // (op 1641) 



v_add_u32 v0, 128, v21 // (op 1644) 

buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 1644) Store value


v_accvgpr_read v14, a12 // (op 1650) 
v_accvgpr_read v15, a13 // (op 1650) 
v_accvgpr_read v16, a14 // (op 1650) 
v_accvgpr_read v17, a15 // (op 1650) 
v_mul_f32 v10, v19, v14 // (op 1650) 
v_mul_f32 v11, v19, v15 // (op 1650) 
v_mul_f32 v12, v19, v16 // (op 1650) 
v_mul_f32 v13, v19, v17 // (op 1650) 



s_lshl_b32 s0, s4, 7 // (op 1646) 
s_add_u32 s0, s0, 128 // (op 1646) 
v_add_u32 v0, v21, s0 // (op 1646) 


buffer_store_dwordx4 v[10:13], v0, s[44:47], 0 offen // (op 1646) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalBottom_0_beta_133
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v20 // (op 1730) 

buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 1730) Load value

v_accvgpr_read v10, a8 // (op 1722) 
v_accvgpr_read v11, a9 // (op 1722) 
v_accvgpr_read v12, a10 // (op 1722) 
v_accvgpr_read v13, a11 // (op 1722) 
v_mul_f32 v6, v19, v10 // (op 1722) 
v_mul_f32 v7, v19, v11 // (op 1722) 
v_mul_f32 v8, v19, v12 // (op 1722) 
v_mul_f32 v9, v19, v13 // (op 1722) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 1729) 
v_mul_f32 v11, v18, v15 // (op 1729) 
v_mul_f32 v12, v18, v16 // (op 1729) 
v_mul_f32 v13, v18, v17 // (op 1729) 

v_add_f32 v14, v10, v6 // (op 1723) 
v_add_f32 v15, v11, v7 // (op 1723) 
v_add_f32 v16, v12, v8 // (op 1723) 
v_add_f32 v17, v13, v9 // (op 1723) 




v_add_u32 v0, 128, v21 // (op 1725) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1725) Store value




s_lshl_b32 s0, s2, 7 // (op 1734) 
s_add_u32 s0, s0, 128 // (op 1734) 
v_add_u32 v0, v20, s0 // (op 1734) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 1734) Load value

v_accvgpr_read v10, a12 // (op 1736) 
v_accvgpr_read v11, a13 // (op 1736) 
v_accvgpr_read v12, a14 // (op 1736) 
v_accvgpr_read v13, a15 // (op 1736) 
v_mul_f32 v6, v19, v10 // (op 1736) 
v_mul_f32 v7, v19, v11 // (op 1736) 
v_mul_f32 v8, v19, v12 // (op 1736) 
v_mul_f32 v9, v19, v13 // (op 1736) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 1732) 
v_mul_f32 v11, v18, v15 // (op 1732) 
v_mul_f32 v12, v18, v16 // (op 1732) 
v_mul_f32 v13, v18, v17 // (op 1732) 

v_add_f32 v14, v10, v6 // (op 1733) 
v_add_f32 v15, v11, v7 // (op 1733) 
v_add_f32 v16, v12, v8 // (op 1733) 
v_add_f32 v17, v13, v9 // (op 1733) 




s_lshl_b32 s0, s4, 7 // (op 1727) 
s_add_u32 s0, s0, 128 // (op 1727) 
v_add_u32 v0, v21, s0 // (op 1727) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1727) Store value




v_mov_b32 v0, v20 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 145) Load value

v_accvgpr_read v10, a0 // (op 142) 
v_accvgpr_read v11, a1 // (op 142) 
v_accvgpr_read v12, a2 // (op 142) 
v_accvgpr_read v13, a3 // (op 142) 
v_mul_f32 v6, v19, v10 // (op 142) 
v_mul_f32 v7, v19, v11 // (op 142) 
v_mul_f32 v8, v19, v12 // (op 142) 
v_mul_f32 v9, v19, v13 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 146) 
v_mul_f32 v11, v18, v15 // (op 146) 
v_mul_f32 v12, v18, v16 // (op 146) 
v_mul_f32 v13, v18, v17 // (op 146) 

v_add_f32 v14, v10, v6 // (op 143) 
v_add_f32 v15, v11, v7 // (op 143) 
v_add_f32 v16, v12, v8 // (op 143) 
v_add_f32 v17, v13, v9 // (op 143) 




v_mov_b32 v0, v21 // (op 144) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 1692) 
v_add_u32 v0, v20, s0 // (op 1692) 


buffer_load_dwordx4 v[14:17], v0, s[40:43], 0 offen // (op 1692) Load value

v_accvgpr_read v10, a4 // (op 1689) 
v_accvgpr_read v11, a5 // (op 1689) 
v_accvgpr_read v12, a6 // (op 1689) 
v_accvgpr_read v13, a7 // (op 1689) 
v_mul_f32 v6, v19, v10 // (op 1689) 
v_mul_f32 v7, v19, v11 // (op 1689) 
v_mul_f32 v8, v19, v12 // (op 1689) 
v_mul_f32 v9, v19, v13 // (op 1689) 

s_waitcnt vmcnt(0)
v_mul_f32 v10, v18, v14 // (op 1693) 
v_mul_f32 v11, v18, v15 // (op 1693) 
v_mul_f32 v12, v18, v16 // (op 1693) 
v_mul_f32 v13, v18, v17 // (op 1693) 

v_add_f32 v14, v10, v6 // (op 1690) 
v_add_f32 v15, v11, v7 // (op 1690) 
v_add_f32 v16, v12, v8 // (op 1690) 
v_add_f32 v17, v13, v9 // (op 1690) 




s_lshl_b32 s0, s4, 7 // (op 1691) 
v_add_u32 v0, v21, s0 // (op 1691) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 1691) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_ConditionalBottom_0_beta_133:












s_endpgm  // End of RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
.LRR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_end:
.size RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4, .LRR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4_end-RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
  .amdhsa_next_free_vgpr 124
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 26112 // lds bytes
  .amdhsa_accum_offset 108
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
    macrotile: [64, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
    unroll: 4
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4
    .symbol: RR_GEMM_NT_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_4.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 26112
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 60
    .vgpr_count: 106
    .agpr_count: 16
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

