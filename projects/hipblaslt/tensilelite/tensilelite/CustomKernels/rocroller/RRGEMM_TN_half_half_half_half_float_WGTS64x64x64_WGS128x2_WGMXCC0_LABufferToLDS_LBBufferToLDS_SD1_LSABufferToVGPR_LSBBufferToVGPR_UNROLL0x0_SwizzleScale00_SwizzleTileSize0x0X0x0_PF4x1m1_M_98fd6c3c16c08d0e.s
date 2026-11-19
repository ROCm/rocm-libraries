

.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e
.p2align 8
.type RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e,@function
RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e:



s_load_dwordx2 s[4:5], s[0:1], 0 // Load scalar value
s_load_dwordx2 s[6:7], s[0:1], 8 // Load scalar value
s_load_dwordx2 s[8:9], s[0:1], 16 // Load scalar value
s_load_dwordx2 s[10:11], s[0:1], 24 // Load scalar value
s_load_dwordx2 s[12:13], s[0:1], 32 // Load scalar value
s_load_dwordx2 s[14:15], s[0:1], 40 // Load scalar value
s_load_dword s16, s[0:1], 48 // Load scalar value
s_load_dword s17, s[0:1], 52 // Load scalar value
s_load_dwordx2 s[18:19], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[20:21], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 104 // Load scalar value
v_bfe_u32 v1, v0, 0, 10 // 
v_bfe_u32 v2, v0, 10, 10 // 


v_and_b32 v0, 63, v1 // (op 1243) 
v_lshrrev_b32 v3, 7, v1 // (op 1243) 
v_and_b32 v4, 31, v1 // (op 1243) 
v_lshrrev_b32 v5, 5, v0 // (op 1243) 

v_lshl_add_u32 v0, v3, 5, v4 // (op 1243) 


v_lshlrev_b32 v4, 2, v5 // (op 1243) 

v_lshl_add_u32 v3, v0, 6, v4 // (op 1243) 


v_lshlrev_b32 v0, 1, v3 // (op 1243) 

v_lshrrev_b32 v3, 6, v1 // (op 1295) 
v_and_b32 v4, 63, v1 // (op 1295) 
v_and_b32 v5, 1, v3 // (op 1295) 

v_and_b32 v3, 31, v1 // (op 1295) 
v_lshrrev_b32 v6, 5, v4 // (op 1295) 

v_lshl_add_u32 v4, v5, 5, v3 // (op 1295) 


v_lshlrev_b32 v3, 2, v6 // (op 1295) 

v_lshl_add_u32 v5, v4, 6, v3 // (op 1295) 


v_lshlrev_b32 v4, 1, v5 // (op 1295) 

v_and_b32 v3, 63, v1 // (op 1368) 
v_lshrrev_b32 v5, 7, v1 // (op 1368) 
v_and_b32 v6, 31, v1 // (op 1368) 
v_lshrrev_b32 v7, 5, v3 // (op 1368) 

v_lshl_add_u32 v3, v5, 5, v6 // (op 1368) 


v_lshlrev_b32 v6, 2, v7 // (op 1368) 

v_lshl_add_u32 v5, v3, 6, v6 // (op 1368) 


v_lshlrev_b32 v3, 1, v5 // (op 1368) 

v_lshrrev_b32 v5, 6, v1 // (op 1407) 
v_and_b32 v6, 63, v1 // (op 1407) 
v_and_b32 v7, 1, v5 // (op 1407) 

v_and_b32 v5, 31, v1 // (op 1407) 
v_lshrrev_b32 v8, 5, v6 // (op 1407) 

v_lshl_add_u32 v6, v7, 5, v5 // (op 1407) 


v_lshlrev_b32 v5, 2, v8 // (op 1407) 

v_lshl_add_u32 v7, v6, 6, v5 // (op 1407) 


v_lshlrev_b32 v6, 1, v7 // (op 1407) 

v_and_b32 v5, 63, v1 // (op 1446) 
v_lshrrev_b32 v7, 7, v1 // (op 1446) 
v_and_b32 v8, 31, v1 // (op 1446) 
v_lshrrev_b32 v9, 5, v5 // (op 1446) 

v_lshl_add_u32 v5, v7, 5, v8 // (op 1446) 


v_lshlrev_b32 v8, 2, v9 // (op 1446) 

v_lshl_add_u32 v7, v5, 6, v8 // (op 1446) 


v_lshlrev_b32 v5, 1, v7 // (op 1446) 

v_lshrrev_b32 v7, 6, v1 // (op 1485) 
v_and_b32 v8, 63, v1 // (op 1485) 
v_and_b32 v9, 1, v7 // (op 1485) 

v_and_b32 v7, 31, v1 // (op 1485) 
v_lshrrev_b32 v10, 5, v8 // (op 1485) 

v_lshl_add_u32 v8, v9, 5, v7 // (op 1485) 


v_lshlrev_b32 v7, 2, v10 // (op 1485) 

v_lshl_add_u32 v9, v8, 6, v7 // (op 1485) 


v_lshlrev_b32 v8, 1, v9 // (op 1485) 

v_and_b32 v7, 63, v1 // (op 1524) 
v_lshrrev_b32 v9, 7, v1 // (op 1524) 
v_and_b32 v10, 31, v1 // (op 1524) 
v_lshrrev_b32 v11, 5, v7 // (op 1524) 

v_lshl_add_u32 v7, v9, 5, v10 // (op 1524) 


v_lshlrev_b32 v10, 2, v11 // (op 1524) 

v_lshl_add_u32 v9, v7, 6, v10 // (op 1524) 


v_lshlrev_b32 v7, 1, v9 // (op 1524) 

v_lshrrev_b32 v9, 6, v1 // (op 1563) 
v_and_b32 v10, 63, v1 // (op 1563) 
v_and_b32 v11, 1, v9 // (op 1563) 

v_and_b32 v9, 31, v1 // (op 1563) 
v_lshrrev_b32 v12, 5, v10 // (op 1563) 

v_lshl_add_u32 v10, v11, 5, v9 // (op 1563) 


v_lshlrev_b32 v9, 2, v12 // (op 1563) 

v_lshl_add_u32 v11, v10, 6, v9 // (op 1563) 


v_lshlrev_b32 v10, 1, v11 // (op 1563) 

s_waitcnt lgkmcnt(0)
v_mov_b32 v9, s17 // (op 2) Move value

v_mov_b32 v11, s16 // (op 24) Move value


v_accvgpr_write a0, 0.00000 // (op 44) call()
v_accvgpr_write a1, 0.00000 // (op 44) call()
v_accvgpr_write a2, 0.00000 // (op 44) call()
v_accvgpr_write a3, 0.00000 // (op 44) call()
v_accvgpr_write a4, 0.00000 // (op 44) call()
v_accvgpr_write a5, 0.00000 // (op 44) call()
v_accvgpr_write a6, 0.00000 // (op 44) call()
v_accvgpr_write a7, 0.00000 // (op 44) call()
v_accvgpr_write a8, 0.00000 // (op 44) call()
v_accvgpr_write a9, 0.00000 // (op 44) call()
v_accvgpr_write a10, 0.00000 // (op 44) call()
v_accvgpr_write a11, 0.00000 // (op 44) call()
v_accvgpr_write a12, 0.00000 // (op 44) call()
v_accvgpr_write a13, 0.00000 // (op 44) call()
v_accvgpr_write a14, 0.00000 // (op 44) call()
v_accvgpr_write a15, 0.00000 // (op 44) call()
v_and_b32 v12, -32, v1 // (op 1082) 
v_lshrrev_b32 v13, 5, v1 // (op 1082) 
v_sub_u32 v14, v1, v12 // (op 1082) 

v_lshl_add_u32 v12, s2, 6, v13 // (op 1082) 

v_lshlrev_b32 v13, 1, v14 // (op 1082) 

v_mov_b32 v15, 0 // (op 1082) convert
v_mov_b32 v14, v13 // (op 1082) convert

v_mul_lo_u32 v17, v12, s19 // (op 1082) most significant: low of low * high
v_mul_hi_u32 v13, v12, s18 // (op 1082) most significant: high of low * low
v_mul_lo_u32 v16, v12, s18 // (op 1082) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 1082) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 1082) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 1082) most significant half



v_lshlrev_b64 v[14:15], 1, v[16:17] // (op 1082) 

s_lshl_b64 s[0:1], s[4:5], 1 // (op 1084) 
s_mov_b32 s32, s6 // (op 1084) reg expression
s_mov_b32 s33, s7 // (op 1084) reg expression
s_mov_b32 s34, s0 // (op 1084) reg expression
s_mov_b32 s35, 131072 // (op 1084) call()

v_and_b32 v13, -32, v1 // (op 1101) 
v_lshrrev_b32 v12, 5, v1 // (op 1101) 
v_sub_u32 v16, v1, v13 // (op 1101) 

v_lshl_add_u32 v13, s2, 6, v12 // (op 1101) 

v_lshlrev_b32 v12, 1, v16 // (op 1101) 

v_mov_b32 v17, 0 // (op 1101) convert
v_mov_b32 v16, v12 // (op 1101) convert

v_mul_lo_u32 v19, v13, s19 // (op 1101) most significant: low of low * high
v_mul_hi_u32 v12, v13, s18 // (op 1101) most significant: high of low * low
v_mul_lo_u32 v18, v13, s18 // (op 1101) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 1101) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 1101) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 1101) most significant half



v_lshlrev_b64 v[12:13], 1, v[18:19] // (op 1101) 

s_lshl_b64 s[0:1], s[4:5], 1 // (op 1103) 
s_mov_b32 s32, s6 // (op 1103) reg expression
s_mov_b32 s33, s7 // (op 1103) reg expression
s_mov_b32 s34, s0 // (op 1103) reg expression
s_mov_b32 s35, 131072 // (op 1103) call()

v_and_b32 v16, -32, v1 // (op 1146) 
v_lshrrev_b32 v17, 5, v1 // (op 1146) 
v_sub_u32 v18, v1, v16 // (op 1146) 

v_lshl_add_u32 v16, s3, 6, v17 // (op 1146) 

v_lshlrev_b32 v17, 1, v18 // (op 1146) 

v_mul_lo_u32 v19, v16, s23 // (op 1146) most significant: low of low * high
v_mul_hi_u32 v20, v16, s22 // (op 1146) most significant: high of low * low
v_mul_lo_u32 v18, v16, s22 // (op 1146) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 1146) most significant: sum


v_mov_b32 v21, 0 // (op 1146) convert
v_mov_b32 v20, v17 // (op 1146) convert

v_add_co_u32 v16, s[0:1], v20, v18 // (op 1146) least significant half
v_addc_co_u32 v17, s[0:1], v21, v19, s[0:1] // (op 1146) most significant half

v_lshlrev_b64 v[16:17], 1, v[16:17] // (op 1146) 


s_lshl_b64 s[0:1], s[8:9], 1 // (op 1148) 
s_mov_b32 s36, s10 // (op 1148) reg expression
s_mov_b32 s37, s11 // (op 1148) reg expression
s_mov_b32 s38, s0 // (op 1148) reg expression
s_mov_b32 s39, 131072 // (op 1148) call()

v_and_b32 v18, -32, v1 // (op 1165) 
v_lshrrev_b32 v19, 5, v1 // (op 1165) 
v_sub_u32 v20, v1, v18 // (op 1165) 

v_lshl_add_u32 v18, s3, 6, v19 // (op 1165) 

v_lshlrev_b32 v19, 1, v20 // (op 1165) 

v_mul_lo_u32 v21, v18, s23 // (op 1165) most significant: low of low * high
v_mul_hi_u32 v22, v18, s22 // (op 1165) most significant: high of low * low
v_mul_lo_u32 v20, v18, s22 // (op 1165) least significant: low of low * low
v_add_u32 v21, v21, v22 // (op 1165) most significant: sum


v_mov_b32 v23, 0 // (op 1165) convert
v_mov_b32 v22, v19 // (op 1165) convert

v_add_co_u32 v18, s[0:1], v22, v20 // (op 1165) least significant half
v_addc_co_u32 v19, s[0:1], v23, v21, s[0:1] // (op 1165) most significant half

v_lshlrev_b64 v[18:19], 1, v[18:19] // (op 1165) 


s_lshl_b64 s[0:1], s[8:9], 1 // (op 1167) 
s_mov_b32 s36, s10 // (op 1167) reg expression
s_mov_b32 s37, s11 // (op 1167) reg expression
s_mov_b32 s38, s0 // (op 1167) reg expression
s_mov_b32 s39, 131072 // (op 1167) call()

v_and_b32 v20, -32, v1 // (op 1283) 
v_lshrrev_b32 v21, 5, v1 // (op 1283) 
v_sub_u32 v22, v1, v20 // (op 1283) 

v_lshlrev_b32 v20, 1, v22 // (op 1283) 

v_lshl_add_u32 v22, v21, 6, v20 // (op 1283) 


v_lshlrev_b32 v21, 1, v22 // (op 1283) 

s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1283) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[16:17], s[22:23], 3 // (op 1023) 
v_lshlrev_b64 v[20:21], 1, s[16:17] // (op 1023) 


v_mov_b64 v[22:23], v[16:17] // (op 1023) 
s_mov_b32 m0, s1 // (op 1023) reg expression
buffer_load_dword v22, s[36:39], 0 offen offset: 0 lds // (op 1023) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1023) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1023) most significant half

s_add_u32 m0, m0, 1024 // (op 1023) 
buffer_load_dword v22, s[36:39], 0 offen offset: 0 lds // (op 1023) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1023) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1023) most significant half

s_add_u32 m0, m0, 1024 // (op 1023) 
buffer_load_dword v22, s[36:39], 0 offen offset: 0 lds // (op 1023) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1023) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1023) most significant half

s_add_u32 m0, m0, 1024 // (op 1023) 
buffer_load_dword v22, s[36:39], 0 offen offset: 0 lds // (op 1023) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1023) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1023) most significant half

s_add_u32 m0, m0, 1024 // (op 1023) 
buffer_load_dword v22, s[36:39], 0 offen offset: 0 lds // (op 1023) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1023) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1023) most significant half

s_add_u32 m0, m0, 1024 // (op 1023) 
buffer_load_dword v22, s[36:39], 0 offen offset: 0 lds // (op 1023) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1023) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1023) most significant half

s_add_u32 m0, m0, 1024 // (op 1023) 
buffer_load_dword v22, s[36:39], 0 offen offset: 0 lds // (op 1023) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1023) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1023) most significant half

s_add_u32 m0, m0, 1024 // (op 1023) 
buffer_load_dword v22, s[36:39], 0 offen offset: 0 lds // (op 1023) Load value direct to lds



v_and_b32 v20, -32, v1 // (op 1231) 
v_lshrrev_b32 v21, 5, v1 // (op 1231) 
v_sub_u32 v22, v1, v20 // (op 1231) 

v_lshlrev_b32 v20, 1, v22 // (op 1231) 

v_lshl_add_u32 v22, v21, 6, v20 // (op 1231) 


v_lshlrev_b32 v21, 1, v22 // (op 1231) 

s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1231) 


s_lshl_b64 s[16:17], s[18:19], 3 // (op 1028) 
v_lshlrev_b64 v[20:21], 1, s[16:17] // (op 1028) 


v_mov_b64 v[22:23], v[14:15] // (op 1028) 
s_add_u32 s0, 8192, s1 // (op 1028) 
s_mov_b32 m0, s0 // (op 1028) reg expression
buffer_load_dword v22, s[32:35], 0 offen offset: 0 lds // (op 1028) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1028) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1028) most significant half

s_add_u32 m0, m0, 1024 // (op 1028) 
buffer_load_dword v22, s[32:35], 0 offen offset: 0 lds // (op 1028) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1028) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1028) most significant half

s_add_u32 m0, m0, 1024 // (op 1028) 
buffer_load_dword v22, s[32:35], 0 offen offset: 0 lds // (op 1028) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1028) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1028) most significant half

s_add_u32 m0, m0, 1024 // (op 1028) 
buffer_load_dword v22, s[32:35], 0 offen offset: 0 lds // (op 1028) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1028) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1028) most significant half

s_add_u32 m0, m0, 1024 // (op 1028) 
buffer_load_dword v22, s[32:35], 0 offen offset: 0 lds // (op 1028) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1028) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1028) most significant half

s_add_u32 m0, m0, 1024 // (op 1028) 
buffer_load_dword v22, s[32:35], 0 offen offset: 0 lds // (op 1028) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1028) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1028) most significant half

s_add_u32 m0, m0, 1024 // (op 1028) 
buffer_load_dword v22, s[32:35], 0 offen offset: 0 lds // (op 1028) Load value direct to lds
v_add_co_u32 v22, s[16:17], v22, v20 // (op 1028) least significant half
v_addc_co_u32 v23, s[16:17], 0, v23, s[16:17] // (op 1028) most significant half

s_add_u32 m0, m0, 1024 // (op 1028) 
buffer_load_dword v22, s[32:35], 0 offen offset: 0 lds // (op 1028) Load value direct to lds




v_and_b32 v20, -32, v1 // (op 1395) 
v_lshrrev_b32 v21, 5, v1 // (op 1395) 
v_sub_u32 v22, v1, v20 // (op 1395) 

v_lshlrev_b32 v20, 1, v22 // (op 1395) 

v_lshl_add_u32 v22, v21, 6, v20 // (op 1395) 


v_lshlrev_b32 v21, 1, v22 // (op 1395) 

s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1395) 


s_lshl_b64 s[16:17], s[22:23], 3 // (op 1033) 
v_lshlrev_b64 v[20:21], 1, s[16:17] // (op 1033) 


v_mov_b64 v[22:23], v[16:17] // (op 1033) 
s_add_u32 s0, 16384, s1 // (op 1033) 
s_mov_b32 m0, s0 // (op 1033) reg expression
s_mov_b32 s17, 128 // (op 1033) call()
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1033) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1033) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1033) most significant half

s_add_u32 m0, m0, 1024 // (op 1033) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1033) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1033) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1033) most significant half

s_add_u32 m0, m0, 1024 // (op 1033) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1033) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1033) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1033) most significant half

s_add_u32 m0, m0, 1024 // (op 1033) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1033) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1033) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1033) most significant half

s_add_u32 m0, m0, 1024 // (op 1033) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1033) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1033) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1033) most significant half

s_add_u32 m0, m0, 1024 // (op 1033) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1033) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1033) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1033) most significant half

s_add_u32 m0, m0, 1024 // (op 1033) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1033) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1033) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1033) most significant half

s_add_u32 m0, m0, 1024 // (op 1033) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1033) Load value direct to lds





v_and_b32 v20, -32, v1 // (op 1356) 
v_lshrrev_b32 v21, 5, v1 // (op 1356) 
v_sub_u32 v22, v1, v20 // (op 1356) 

v_lshlrev_b32 v20, 1, v22 // (op 1356) 

v_lshl_add_u32 v22, v21, 6, v20 // (op 1356) 


v_lshlrev_b32 v21, 1, v22 // (op 1356) 

s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1356) 


s_lshl_b64 s[16:17], s[18:19], 3 // (op 1035) 
v_lshlrev_b64 v[20:21], 1, s[16:17] // (op 1035) 


v_mov_b64 v[22:23], v[14:15] // (op 1035) 
s_add_u32 s0, 24576, s1 // (op 1035) 
s_mov_b32 m0, s0 // (op 1035) reg expression
s_mov_b32 s17, 128 // (op 1035) call()
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1035) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1035) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1035) most significant half

s_add_u32 m0, m0, 1024 // (op 1035) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1035) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1035) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1035) most significant half

s_add_u32 m0, m0, 1024 // (op 1035) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1035) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1035) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1035) most significant half

s_add_u32 m0, m0, 1024 // (op 1035) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1035) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1035) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1035) most significant half

s_add_u32 m0, m0, 1024 // (op 1035) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1035) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1035) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1035) most significant half

s_add_u32 m0, m0, 1024 // (op 1035) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1035) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1035) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1035) most significant half

s_add_u32 m0, m0, 1024 // (op 1035) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1035) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1035) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1035) most significant half

s_add_u32 m0, m0, 1024 // (op 1035) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1035) Load value direct to lds





v_and_b32 v20, -32, v1 // (op 1473) 
v_lshrrev_b32 v21, 5, v1 // (op 1473) 
v_sub_u32 v22, v1, v20 // (op 1473) 

v_lshlrev_b32 v20, 1, v22 // (op 1473) 

v_lshl_add_u32 v22, v21, 6, v20 // (op 1473) 


v_lshlrev_b32 v21, 1, v22 // (op 1473) 

s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1473) 


s_lshl_b64 s[16:17], s[22:23], 3 // (op 1037) 
v_lshlrev_b64 v[20:21], 1, s[16:17] // (op 1037) 


v_mov_b64 v[22:23], v[16:17] // (op 1037) 
s_add_u32 s0, 32768, s1 // (op 1037) 
s_mov_b32 m0, s0 // (op 1037) reg expression
s_mov_b32 s17, 256 // (op 1037) call()
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1037) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1037) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1037) most significant half

s_add_u32 m0, m0, 1024 // (op 1037) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1037) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1037) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1037) most significant half

s_add_u32 m0, m0, 1024 // (op 1037) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1037) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1037) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1037) most significant half

s_add_u32 m0, m0, 1024 // (op 1037) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1037) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1037) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1037) most significant half

s_add_u32 m0, m0, 1024 // (op 1037) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1037) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1037) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1037) most significant half

s_add_u32 m0, m0, 1024 // (op 1037) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1037) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1037) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1037) most significant half

s_add_u32 m0, m0, 1024 // (op 1037) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1037) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1037) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1037) most significant half

s_add_u32 m0, m0, 1024 // (op 1037) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1037) Load value direct to lds





v_and_b32 v20, -32, v1 // (op 1434) 
v_lshrrev_b32 v21, 5, v1 // (op 1434) 
v_sub_u32 v22, v1, v20 // (op 1434) 

v_lshlrev_b32 v20, 1, v22 // (op 1434) 

v_lshl_add_u32 v22, v21, 6, v20 // (op 1434) 


v_lshlrev_b32 v21, 1, v22 // (op 1434) 

s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1434) 


s_lshl_b64 s[16:17], s[18:19], 3 // (op 1039) 
v_lshlrev_b64 v[20:21], 1, s[16:17] // (op 1039) 


v_mov_b64 v[22:23], v[14:15] // (op 1039) 
s_add_u32 s0, 40960, s1 // (op 1039) 
s_mov_b32 m0, s0 // (op 1039) reg expression
s_mov_b32 s17, 256 // (op 1039) call()
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1039) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1039) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1039) most significant half

s_add_u32 m0, m0, 1024 // (op 1039) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1039) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1039) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1039) most significant half

s_add_u32 m0, m0, 1024 // (op 1039) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1039) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1039) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1039) most significant half

s_add_u32 m0, m0, 1024 // (op 1039) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1039) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1039) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1039) most significant half

s_add_u32 m0, m0, 1024 // (op 1039) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1039) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1039) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1039) most significant half

s_add_u32 m0, m0, 1024 // (op 1039) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1039) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1039) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1039) most significant half

s_add_u32 m0, m0, 1024 // (op 1039) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1039) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1039) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1039) most significant half

s_add_u32 m0, m0, 1024 // (op 1039) 
buffer_load_dword v22, s[32:35], s17 offen offset: 0 lds // (op 1039) Load value direct to lds





v_and_b32 v20, -32, v1 // (op 1551) 
v_lshrrev_b32 v21, 5, v1 // (op 1551) 
v_sub_u32 v22, v1, v20 // (op 1551) 

v_lshlrev_b32 v20, 1, v22 // (op 1551) 

v_lshl_add_u32 v22, v21, 6, v20 // (op 1551) 


v_lshlrev_b32 v21, 1, v22 // (op 1551) 

s_nop 0
v_readfirstlane_b32 s1, v21 // (op 1551) 


s_lshl_b64 s[16:17], s[22:23], 3 // (op 1041) 
v_lshlrev_b64 v[20:21], 1, s[16:17] // (op 1041) 


v_mov_b64 v[22:23], v[16:17] // (op 1041) 
s_add_u32 s0, 49152, s1 // (op 1041) 
s_mov_b32 m0, s0 // (op 1041) reg expression
s_mov_b32 s17, 384 // (op 1041) call()
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1041) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1041) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1041) most significant half

s_add_u32 m0, m0, 1024 // (op 1041) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1041) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1041) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1041) most significant half

s_add_u32 m0, m0, 1024 // (op 1041) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1041) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1041) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1041) most significant half

s_add_u32 m0, m0, 1024 // (op 1041) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1041) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1041) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1041) most significant half

s_add_u32 m0, m0, 1024 // (op 1041) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1041) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1041) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1041) most significant half

s_add_u32 m0, m0, 1024 // (op 1041) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1041) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1041) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1041) most significant half

s_add_u32 m0, m0, 1024 // (op 1041) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1041) Load value direct to lds
v_add_co_u32 v22, s[40:41], v22, v20 // (op 1041) least significant half
v_addc_co_u32 v23, s[40:41], 0, v23, s[40:41] // (op 1041) most significant half

s_add_u32 m0, m0, 1024 // (op 1041) 
buffer_load_dword v22, s[36:39], s17 offen offset: 0 lds // (op 1041) Load value direct to lds






v_and_b32 v17, -32, v1 // (op 1512) 
v_lshrrev_b32 v16, 5, v1 // (op 1512) 
v_sub_u32 v20, v1, v17 // (op 1512) 

v_lshlrev_b32 v17, 1, v20 // (op 1512) 

v_lshl_add_u32 v20, v16, 6, v17 // (op 1512) 


v_lshlrev_b32 v17, 1, v20 // (op 1512) 

s_nop 0
v_readfirstlane_b32 s1, v17 // (op 1512) 


s_lshl_b64 s[16:17], s[18:19], 3 // (op 1043) 
v_lshlrev_b64 v[16:17], 1, s[16:17] // (op 1043) 


v_mov_b64 v[20:21], v[14:15] // (op 1043) 
s_add_u32 s0, 57344, s1 // (op 1043) 
s_mov_b32 m0, s0 // (op 1043) reg expression
s_mov_b32 s17, 384 // (op 1043) call()
buffer_load_dword v20, s[32:35], s17 offen offset: 0 lds // (op 1043) Load value direct to lds
v_add_co_u32 v20, s[40:41], v20, v16 // (op 1043) least significant half
v_addc_co_u32 v21, s[40:41], 0, v21, s[40:41] // (op 1043) most significant half

s_add_u32 m0, m0, 1024 // (op 1043) 
buffer_load_dword v20, s[32:35], s17 offen offset: 0 lds // (op 1043) Load value direct to lds
v_add_co_u32 v20, s[40:41], v20, v16 // (op 1043) least significant half
v_addc_co_u32 v21, s[40:41], 0, v21, s[40:41] // (op 1043) most significant half

s_add_u32 m0, m0, 1024 // (op 1043) 
buffer_load_dword v20, s[32:35], s17 offen offset: 0 lds // (op 1043) Load value direct to lds
v_add_co_u32 v20, s[40:41], v20, v16 // (op 1043) least significant half
v_addc_co_u32 v21, s[40:41], 0, v21, s[40:41] // (op 1043) most significant half

s_add_u32 m0, m0, 1024 // (op 1043) 
buffer_load_dword v20, s[32:35], s17 offen offset: 0 lds // (op 1043) Load value direct to lds
v_add_co_u32 v20, s[40:41], v20, v16 // (op 1043) least significant half
v_addc_co_u32 v21, s[40:41], 0, v21, s[40:41] // (op 1043) most significant half

s_add_u32 m0, m0, 1024 // (op 1043) 
buffer_load_dword v20, s[32:35], s17 offen offset: 0 lds // (op 1043) Load value direct to lds
v_add_co_u32 v20, s[40:41], v20, v16 // (op 1043) least significant half
v_addc_co_u32 v21, s[40:41], 0, v21, s[40:41] // (op 1043) most significant half

s_add_u32 m0, m0, 1024 // (op 1043) 
buffer_load_dword v20, s[32:35], s17 offen offset: 0 lds // (op 1043) Load value direct to lds
v_add_co_u32 v20, s[40:41], v20, v16 // (op 1043) least significant half
v_addc_co_u32 v21, s[40:41], 0, v21, s[40:41] // (op 1043) most significant half

s_add_u32 m0, m0, 1024 // (op 1043) 
buffer_load_dword v20, s[32:35], s17 offen offset: 0 lds // (op 1043) Load value direct to lds
v_add_co_u32 v20, s[40:41], v20, v16 // (op 1043) least significant half
v_addc_co_u32 v21, s[40:41], 0, v21, s[40:41] // (op 1043) most significant half

s_add_u32 m0, m0, 1024 // (op 1043) 
buffer_load_dword v20, s[32:35], s17 offen offset: 0 lds // (op 1043) Load value direct to lds






s_waitcnt vmcnt(48)
s_barrier  // (op 693) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 15 and the queue size is 64, so a waitcnt of 48 is required.




ds_read_b64 v[16:17], v4 // (op 758) Load local data 



ds_read_b64 v[14:15], v4 offset:16 // (op 761) Load local data 



ds_read_b64 v[20:21], v4 offset:32 // (op 764) Load local data 



ds_read_b64 v[22:23], v4 offset:48 // (op 767) Load local data 



ds_read_b64 v[24:25], v4 offset:64 // (op 770) Load local data 



ds_read_b64 v[26:27], v4 offset:80 // (op 773) Load local data 



ds_read_b64 v[28:29], v4 offset:96 // (op 776) Load local data 



ds_read_b64 v[30:31], v4 offset:112 // (op 779) Load local data 



ds_read_b64 v[32:33], v0 offset:8192 // (op 734) Load local data 



ds_read_b64 v[34:35], v0 offset:8208 // (op 737) Load local data 



ds_read_b64 v[36:37], v0 offset:8224 // (op 740) Load local data 



ds_read_b64 v[38:39], v0 offset:8240 // (op 743) Load local data 



ds_read_b64 v[40:41], v0 offset:8256 // (op 746) Load local data 



ds_read_b64 v[42:43], v0 offset:8272 // (op 749) Load local data 



ds_read_b64 v[44:45], v0 offset:8288 // (op 752) Load local data 



ds_read_b64 v[46:47], v0 offset:8304 // (op 755) Load local data 
v_and_b32 v48, -32, v1 // (op 1203) 
v_lshrrev_b32 v49, 5, v1 // (op 1203) 
v_sub_u32 v50, v1, v48 // (op 1203) 

v_lshlrev_b32 v48, 1, v50 // (op 1203) 

v_lshl_add_u32 v50, v49, 6, v48 // (op 1203) 


v_lshlrev_b32 v49, 1, v50 // (op 1203) 

s_nop 0
v_readfirstlane_b32 s1, v49 // (op 1203) 

v_and_b32 v48, -32, v1 // (op 1255) 
v_lshrrev_b32 v49, 5, v1 // (op 1255) 
v_sub_u32 v50, v1, v48 // (op 1255) 

v_lshlrev_b32 v48, 1, v50 // (op 1255) 

v_lshl_add_u32 v50, v49, 6, v48 // (op 1255) 


v_lshlrev_b32 v49, 1, v50 // (op 1255) 

s_nop 0
v_readfirstlane_b32 s0, v49 // (op 1255) 

v_and_b32 v48, -32, v1 // (op 1341) 
v_lshrrev_b32 v49, 5, v1 // (op 1341) 
v_sub_u32 v50, v1, v48 // (op 1341) 

v_lshlrev_b32 v48, 1, v50 // (op 1341) 

v_lshl_add_u32 v50, v49, 6, v48 // (op 1341) 


v_lshlrev_b32 v49, 1, v50 // (op 1341) 

s_nop 0
v_readfirstlane_b32 s17, v49 // (op 1341) 

v_and_b32 v48, -32, v1 // (op 1380) 
v_lshrrev_b32 v49, 5, v1 // (op 1380) 
v_sub_u32 v50, v1, v48 // (op 1380) 

v_lshlrev_b32 v48, 1, v50 // (op 1380) 

v_lshl_add_u32 v50, v49, 6, v48 // (op 1380) 


v_lshlrev_b32 v49, 1, v50 // (op 1380) 

s_nop 0
v_readfirstlane_b32 s16, v49 // (op 1380) 

v_and_b32 v48, -32, v1 // (op 1419) 
v_lshrrev_b32 v49, 5, v1 // (op 1419) 
v_sub_u32 v50, v1, v48 // (op 1419) 

v_lshlrev_b32 v48, 1, v50 // (op 1419) 

v_lshl_add_u32 v50, v49, 6, v48 // (op 1419) 


v_lshlrev_b32 v49, 1, v50 // (op 1419) 

s_nop 0
v_readfirstlane_b32 s40, v49 // (op 1419) 

v_and_b32 v48, -32, v1 // (op 1458) 
v_lshrrev_b32 v49, 5, v1 // (op 1458) 
v_sub_u32 v50, v1, v48 // (op 1458) 

v_lshlrev_b32 v48, 1, v50 // (op 1458) 

v_lshl_add_u32 v50, v49, 6, v48 // (op 1458) 


v_lshlrev_b32 v49, 1, v50 // (op 1458) 

s_nop 0
v_readfirstlane_b32 s41, v49 // (op 1458) 

v_and_b32 v48, -32, v1 // (op 1497) 
v_lshrrev_b32 v49, 5, v1 // (op 1497) 
v_sub_u32 v50, v1, v48 // (op 1497) 

v_lshlrev_b32 v48, 1, v50 // (op 1497) 

v_lshl_add_u32 v50, v49, 6, v48 // (op 1497) 


v_lshlrev_b32 v49, 1, v50 // (op 1497) 

s_nop 0
v_readfirstlane_b32 s42, v49 // (op 1497) 

v_and_b32 v48, -32, v1 // (op 1536) 
v_lshrrev_b32 v49, 5, v1 // (op 1536) 
v_sub_u32 v50, v1, v48 // (op 1536) 

v_lshlrev_b32 v48, 1, v50 // (op 1536) 

v_lshl_add_u32 v50, v49, 6, v48 // (op 1536) 


v_lshlrev_b32 v49, 1, v50 // (op 1536) 

s_nop 0
v_readfirstlane_b32 s43, v49 // (op 1536) 

s_mov_b32 s44, 0 // (op 40) call()
s_ashr_i64 s[46:47], s[20:21], 6 // (op 39) 
s_mov_b32 s45, s46 // (op 39) convert

s_and_b32 s46, s45, 4294967292 // (op 39) 

s_cmp_lt_u32 s44, s46 // (op 39) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopBottom_KLoop_39 // (op 39) Condition: Top (jump to RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopBottom_KLoop_39 if false)
RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopTop_KLoop_39:
s_waitcnt vmcnt(32)
s_barrier  // (op 811) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 15 and the queue size is 48, so a waitcnt of 32 is required.

s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[32:33], v[16:17], a[0:15] // (op 57) WaitCnt Needed: Intersects with registers in 'DSQueue', at 7 and the queue size is 15, so a waitcnt of 7 is required.

s_barrier  // (op 1733) 




ds_read_b64 v[16:17], v6 offset:16384 // (op 333) Load local data 
s_add_u32 s44, s44, 4 // (op 41) 



ds_read_b64 v[48:49], v6 offset:16400 // (op 338) Load local data 
s_lshl_b64 s[46:47], s[22:23], 3 // (op 976) 



v_lshlrev_b64 v[50:51], 1, s[46:47] // (op 976) 


v_mov_b64 v[52:53], v[18:19] // (op 976) 
v_mov_b64 v[54:55], v[52:53] // (op 976) 
s_mov_b32 m0, s0 // (op 976) reg expression
s_mov_b32 s45, 512 // (op 976) call()
s_waitcnt lgkmcnt(8)
v_mfma_f32_32x32x8f16 a[0:15], v[34:35], v[14:15], a[0:15] // (op 66) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 9, so a waitcnt of 8 is required.

buffer_load_dword v54, s[36:39], s45 offen offset: 0 lds // (op 976) Load value direct to lds
v_add_co_u32 v54, s[46:47], v54, v50 // (op 976) least significant half
v_addc_co_u32 v55, s[46:47], 0, v55, s[46:47] // (op 976) most significant half

s_add_u32 m0, m0, 1024 // (op 976) 
buffer_load_dword v54, s[36:39], s45 offen offset: 0 lds // (op 976) Load value direct to lds
v_add_co_u32 v54, s[46:47], v54, v50 // (op 976) least significant half
v_addc_co_u32 v55, s[46:47], 0, v55, s[46:47] // (op 976) most significant half

s_add_u32 m0, m0, 1024 // (op 976) 
buffer_load_dword v54, s[36:39], s45 offen offset: 0 lds // (op 976) Load value direct to lds
s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[36:37], v[20:21], a[0:15] // (op 75) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 8, so a waitcnt of 7 is required.

v_add_co_u32 v54, s[46:47], v54, v50 // (op 976) least significant half
v_addc_co_u32 v55, s[46:47], 0, v55, s[46:47] // (op 976) most significant half

s_add_u32 m0, m0, 1024 // (op 976) 
buffer_load_dword v54, s[36:39], s45 offen offset: 0 lds // (op 976) Load value direct to lds
v_add_co_u32 v54, s[46:47], v54, v50 // (op 976) least significant half
v_addc_co_u32 v55, s[46:47], 0, v55, s[46:47] // (op 976) most significant half

s_add_u32 m0, m0, 1024 // (op 976) 
ds_read_b64 v[56:57], v6 offset:16416 // (op 343) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[38:39], v[22:23], a[0:15] // (op 84) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 8, so a waitcnt of 7 is required.




ds_read_b64 v[58:59], v6 offset:16432 // (op 348) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[40:41], v[24:25], a[0:15] // (op 93) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 8, so a waitcnt of 7 is required.




ds_read_b64 v[60:61], v6 offset:16448 // (op 353) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[42:43], v[26:27], a[0:15] // (op 102) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 8, so a waitcnt of 7 is required.




ds_read_b64 v[62:63], v6 offset:16464 // (op 358) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[44:45], v[28:29], a[0:15] // (op 111) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 8, so a waitcnt of 7 is required.




ds_read_b64 v[64:65], v6 offset:16480 // (op 363) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[46:47], v[30:31], a[0:15] // (op 120) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 8, so a waitcnt of 7 is required.




ds_read_b64 v[66:67], v6 offset:16496 // (op 368) Load local data 
buffer_load_dword v54, s[36:39], s45 offen offset: 0 lds // (op 976) Load value direct to lds
v_add_co_u32 v54, s[46:47], v54, v50 // (op 976) least significant half



v_addc_co_u32 v55, s[46:47], 0, v55, s[46:47] // (op 976) most significant half

ds_read_b64 v[32:33], v3 offset:24576 // (op 331) Load local data 
s_add_u32 m0, m0, 1024 // (op 976) 



ds_read_b64 v[34:35], v3 offset:24592 // (op 336) Load local data 
buffer_load_dword v54, s[36:39], s45 offen offset: 0 lds // (op 976) Load value direct to lds



ds_read_b64 v[36:37], v3 offset:24608 // (op 341) Load local data 
v_add_co_u32 v54, s[46:47], v54, v50 // (op 976) least significant half



v_addc_co_u32 v55, s[46:47], 0, v55, s[46:47] // (op 976) most significant half

s_add_u32 m0, m0, 1024 // (op 976) 
buffer_load_dword v54, s[36:39], s45 offen offset: 0 lds // (op 976) Load value direct to lds
v_add_co_u32 v54, s[46:47], v54, v50 // (op 976) least significant half
v_addc_co_u32 v55, s[46:47], 0, v55, s[46:47] // (op 976) most significant half

s_add_u32 m0, m0, 1024 // (op 976) 
buffer_load_dword v54, s[36:39], s45 offen offset: 0 lds // (op 976) Load value direct to lds



ds_read_b64 v[50:51], v3 offset:24624 // (op 346) Load local data 



s_barrier  // (op 1723) 
ds_read_b64 v[38:39], v3 offset:24640 // (op 351) Load local data 




s_lshl_b64 s[46:47], s[18:19], 3 // (op 971) 
v_lshlrev_b64 v[54:55], 1, s[46:47] // (op 971) 


v_mov_b64 v[68:69], v[12:13] // (op 971) 
v_mov_b64 v[70:71], v[68:69] // (op 971) 
s_add_u32 s45, 8192, s1 // (op 971) 
s_mov_b32 m0, s45 // (op 971) reg expression
s_mov_b32 s46, 512 // (op 971) call()
ds_read_b64 v[72:73], v3 offset:24656 // (op 356) Load local data 
buffer_load_dword v70, s[32:35], s46 offen offset: 0 lds // (op 971) Load value direct to lds



ds_read_b64 v[40:41], v3 offset:24672 // (op 361) Load local data 
v_add_co_u32 v70, s[48:49], v70, v54 // (op 971) least significant half



v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 971) most significant half

s_add_u32 m0, m0, 1024 // (op 971) 
ds_read_b64 v[74:75], v3 offset:24688 // (op 366) Load local data 
buffer_load_dword v70, s[32:35], s46 offen offset: 0 lds // (op 971) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 971) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 971) most significant half

s_add_u32 m0, m0, 1024 // (op 971) 
buffer_load_dword v70, s[32:35], s46 offen offset: 0 lds // (op 971) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 971) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 971) most significant half

s_add_u32 m0, m0, 1024 // (op 971) 
buffer_load_dword v70, s[32:35], s46 offen offset: 0 lds // (op 971) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 971) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 971) most significant half

s_add_u32 m0, m0, 1024 // (op 971) 
buffer_load_dword v70, s[32:35], s46 offen offset: 0 lds // (op 971) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 971) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 971) most significant half

s_add_u32 m0, m0, 1024 // (op 971) 
buffer_load_dword v70, s[32:35], s46 offen offset: 0 lds // (op 971) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 971) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 971) most significant half

s_add_u32 m0, m0, 1024 // (op 971) 
buffer_load_dword v70, s[32:35], s46 offen offset: 0 lds // (op 971) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 971) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 971) most significant half

s_add_u32 m0, m0, 1024 // (op 971) 
buffer_load_dword v70, s[32:35], s46 offen offset: 0 lds // (op 971) Load value direct to lds




s_waitcnt vmcnt(32)
s_barrier  // (op 853) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 15 and the queue size is 48, so a waitcnt of 32 is required.

s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[32:33], v[16:17], a[0:15] // (op 334) WaitCnt Needed: Intersects with registers in 'DSQueue', at 8 and the queue size is 16, so a waitcnt of 7 is required.

s_barrier  // (op 1746) 




ds_read_b64 v[16:17], v8 offset:32768 // (op 423) Load local data 
s_lshl_b64 s[46:47], s[22:23], 3 // (op 998) 



ds_read_b64 v[14:15], v8 offset:32784 // (op 428) Load local data 
v_lshlrev_b64 v[54:55], 1, s[46:47] // (op 998) 





ds_read_b64 v[20:21], v8 offset:32800 // (op 433) Load local data 
v_mov_b64 v[70:71], v[52:53] // (op 998) 



ds_read_b64 v[22:23], v8 offset:32816 // (op 438) Load local data 
s_waitcnt lgkmcnt(10)
v_mfma_f32_32x32x8f16 a[0:15], v[34:35], v[48:49], a[0:15] // (op 339) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 11, so a waitcnt of 10 is required.





s_add_u32 s45, 16384, s16 // (op 998) 
s_mov_b32 m0, s45 // (op 998) reg expression
s_mov_b32 s46, 640 // (op 998) call()
s_waitcnt lgkmcnt(9)
v_mfma_f32_32x32x8f16 a[0:15], v[36:37], v[56:57], a[0:15] // (op 344) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 10, so a waitcnt of 9 is required.

ds_read_b64 v[24:25], v8 offset:32832 // (op 443) Load local data 




ds_read_b64 v[26:27], v8 offset:32848 // (op 448) Load local data 
s_waitcnt lgkmcnt(10)
v_mfma_f32_32x32x8f16 a[0:15], v[50:51], v[58:59], a[0:15] // (op 349) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 11, so a waitcnt of 10 is required.






ds_read_b64 v[28:29], v8 offset:32864 // (op 453) Load local data 



ds_read_b64 v[30:31], v8 offset:32880 // (op 458) Load local data 
s_waitcnt lgkmcnt(11)
v_mfma_f32_32x32x8f16 a[0:15], v[38:39], v[60:61], a[0:15] // (op 354) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 12, so a waitcnt of 11 is required.





ds_read_b64 v[32:33], v5 offset:40960 // (op 421) Load local data 
s_waitcnt lgkmcnt(11)
v_mfma_f32_32x32x8f16 a[0:15], v[72:73], v[62:63], a[0:15] // (op 359) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 12, so a waitcnt of 11 is required.






ds_read_b64 v[34:35], v5 offset:40976 // (op 426) Load local data 



ds_read_b64 v[36:37], v5 offset:40992 // (op 431) Load local data 
buffer_load_dword v70, s[36:39], s46 offen offset: 0 lds // (op 998) Load value direct to lds



ds_read_b64 v[42:43], v5 offset:41008 // (op 436) Load local data 
s_waitcnt lgkmcnt(13)
v_mfma_f32_32x32x8f16 a[0:15], v[40:41], v[64:65], a[0:15] // (op 364) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 14, so a waitcnt of 13 is required.





v_add_co_u32 v70, s[48:49], v70, v54 // (op 998) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 998) most significant half

s_add_u32 m0, m0, 1024 // (op 998) 
buffer_load_dword v70, s[36:39], s46 offen offset: 0 lds // (op 998) Load value direct to lds
ds_read_b64 v[44:45], v5 offset:41024 // (op 441) Load local data 
v_add_co_u32 v70, s[48:49], v70, v54 // (op 998) least significant half



v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 998) most significant half

s_add_u32 m0, m0, 1024 // (op 998) 
s_waitcnt lgkmcnt(13)
v_mfma_f32_32x32x8f16 a[0:15], v[74:75], v[66:67], a[0:15] // (op 369) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 14, so a waitcnt of 13 is required.

buffer_load_dword v70, s[36:39], s46 offen offset: 0 lds // (op 998) Load value direct to lds


ds_read_b64 v[46:47], v5 offset:41040 // (op 446) Load local data 
v_add_co_u32 v70, s[48:49], v70, v54 // (op 998) least significant half



v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 998) most significant half

ds_read_b64 v[38:39], v5 offset:41056 // (op 451) Load local data 
s_add_u32 m0, m0, 1024 // (op 998) 



buffer_load_dword v70, s[36:39], s46 offen offset: 0 lds // (op 998) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 998) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 998) most significant half

s_add_u32 m0, m0, 1024 // (op 998) 
ds_read_b64 v[50:51], v5 offset:41072 // (op 456) Load local data 
buffer_load_dword v70, s[36:39], s46 offen offset: 0 lds // (op 998) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 998) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 998) most significant half

s_add_u32 m0, m0, 1024 // (op 998) 
buffer_load_dword v70, s[36:39], s46 offen offset: 0 lds // (op 998) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 998) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 998) most significant half

s_add_u32 m0, m0, 1024 // (op 998) 
buffer_load_dword v70, s[36:39], s46 offen offset: 0 lds // (op 998) Load value direct to lds
v_add_co_u32 v70, s[48:49], v70, v54 // (op 998) least significant half
v_addc_co_u32 v71, s[48:49], 0, v71, s[48:49] // (op 998) most significant half

s_add_u32 m0, m0, 1024 // (op 998) 
buffer_load_dword v70, s[36:39], s46 offen offset: 0 lds // (op 998) Load value direct to lds




s_barrier  // (op 1743) 

s_lshl_b64 s[46:47], s[18:19], 3 // (op 993) 
v_lshlrev_b64 v[48:49], 1, s[46:47] // (op 993) 


v_mov_b64 v[66:67], v[68:69] // (op 993) 
s_add_u32 s45, 24576, s17 // (op 993) 
s_mov_b32 m0, s45 // (op 993) reg expression
s_mov_b32 s46, 640 // (op 993) call()
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 993) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 993) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 993) most significant half

s_add_u32 m0, m0, 1024 // (op 993) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 993) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 993) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 993) most significant half

s_add_u32 m0, m0, 1024 // (op 993) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 993) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 993) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 993) most significant half

s_add_u32 m0, m0, 1024 // (op 993) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 993) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 993) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 993) most significant half

s_add_u32 m0, m0, 1024 // (op 993) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 993) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 993) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 993) most significant half

s_add_u32 m0, m0, 1024 // (op 993) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 993) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 993) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 993) most significant half

s_add_u32 m0, m0, 1024 // (op 993) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 993) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 993) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 993) most significant half

s_add_u32 m0, m0, 1024 // (op 993) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 993) Load value direct to lds




s_waitcnt vmcnt(32)
s_barrier  // (op 895) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 15 and the queue size is 48, so a waitcnt of 32 is required.

s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[32:33], v[16:17], a[0:15] // (op 424) WaitCnt Needed: Intersects with registers in 'DSQueue', at 8 and the queue size is 16, so a waitcnt of 7 is required.

s_barrier  // (op 1752) 




ds_read_b64 v[16:17], v10 offset:49152 // (op 513) Load local data 
s_lshl_b64 s[46:47], s[22:23], 3 // (op 1008) 



v_lshlrev_b64 v[48:49], 1, s[46:47] // (op 1008) 


v_mov_b64 v[66:67], v[52:53] // (op 1008) 
ds_read_b64 v[64:65], v10 offset:49168 // (op 518) Load local data 
s_add_u32 s45, 32768, s41 // (op 1008) 



s_mov_b32 m0, s45 // (op 1008) reg expression
ds_read_b64 v[62:63], v10 offset:49184 // (op 523) Load local data 
s_waitcnt lgkmcnt(9)
v_mfma_f32_32x32x8f16 a[0:15], v[34:35], v[14:15], a[0:15] // (op 429) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 10, so a waitcnt of 9 is required.




ds_read_b64 v[60:61], v10 offset:49200 // (op 528) Load local data 



s_mov_b32 s46, 768 // (op 1008) call()
s_waitcnt lgkmcnt(9)
v_mfma_f32_32x32x8f16 a[0:15], v[36:37], v[20:21], a[0:15] // (op 434) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 10, so a waitcnt of 9 is required.

ds_read_b64 v[58:59], v10 offset:49216 // (op 533) Load local data 



ds_read_b64 v[56:57], v10 offset:49232 // (op 538) Load local data 
s_waitcnt lgkmcnt(10)
v_mfma_f32_32x32x8f16 a[0:15], v[42:43], v[22:23], a[0:15] // (op 439) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 11, so a waitcnt of 10 is required.




ds_read_b64 v[54:55], v10 offset:49248 // (op 543) Load local data 



ds_read_b64 v[70:71], v10 offset:49264 // (op 548) Load local data 
s_waitcnt lgkmcnt(11)
v_mfma_f32_32x32x8f16 a[0:15], v[44:45], v[24:25], a[0:15] // (op 444) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 12, so a waitcnt of 11 is required.




ds_read_b64 v[32:33], v7 offset:57344 // (op 511) Load local data 
s_waitcnt lgkmcnt(11)
v_mfma_f32_32x32x8f16 a[0:15], v[46:47], v[26:27], a[0:15] // (op 449) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 12, so a waitcnt of 11 is required.




ds_read_b64 v[34:35], v7 offset:57360 // (op 516) Load local data 



ds_read_b64 v[36:37], v7 offset:57376 // (op 521) Load local data 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1008) Load value direct to lds



ds_read_b64 v[40:41], v7 offset:57392 // (op 526) Load local data 
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1008) least significant half



v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1008) most significant half

s_add_u32 m0, m0, 1024 // (op 1008) 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1008) Load value direct to lds
s_waitcnt lgkmcnt(13)
v_mfma_f32_32x32x8f16 a[0:15], v[38:39], v[28:29], a[0:15] // (op 454) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 14, so a waitcnt of 13 is required.

v_add_co_u32 v66, s[48:49], v66, v48 // (op 1008) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1008) most significant half

s_add_u32 m0, m0, 1024 // (op 1008) 
ds_read_b64 v[72:73], v7 offset:57408 // (op 531) Load local data 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1008) Load value direct to lds



v_add_co_u32 v66, s[48:49], v66, v48 // (op 1008) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1008) most significant half

s_add_u32 m0, m0, 1024 // (op 1008) 
s_waitcnt lgkmcnt(13)
v_mfma_f32_32x32x8f16 a[0:15], v[50:51], v[30:31], a[0:15] // (op 459) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 14, so a waitcnt of 13 is required.

buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1008) Load value direct to lds

ds_read_b64 v[50:51], v7 offset:57424 // (op 536) Load local data 
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1008) least significant half



v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1008) most significant half

s_add_u32 m0, m0, 1024 // (op 1008) 
ds_read_b64 v[74:75], v7 offset:57440 // (op 541) Load local data 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1008) Load value direct to lds



v_add_co_u32 v66, s[48:49], v66, v48 // (op 1008) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1008) most significant half

s_add_u32 m0, m0, 1024 // (op 1008) 
ds_read_b64 v[76:77], v7 offset:57456 // (op 546) Load local data 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1008) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1008) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1008) most significant half

s_add_u32 m0, m0, 1024 // (op 1008) 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1008) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1008) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1008) most significant half

s_add_u32 m0, m0, 1024 // (op 1008) 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1008) Load value direct to lds




s_barrier  // (op 1749) 

s_lshl_b64 s[46:47], s[18:19], 3 // (op 1003) 
v_lshlrev_b64 v[48:49], 1, s[46:47] // (op 1003) 


v_mov_b64 v[66:67], v[68:69] // (op 1003) 
s_add_u32 s45, 40960, s40 // (op 1003) 
s_mov_b32 m0, s45 // (op 1003) reg expression
s_mov_b32 s46, 768 // (op 1003) call()
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 1003) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1003) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1003) most significant half

s_add_u32 m0, m0, 1024 // (op 1003) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 1003) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1003) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1003) most significant half

s_add_u32 m0, m0, 1024 // (op 1003) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 1003) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1003) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1003) most significant half

s_add_u32 m0, m0, 1024 // (op 1003) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 1003) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1003) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1003) most significant half

s_add_u32 m0, m0, 1024 // (op 1003) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 1003) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1003) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1003) most significant half

s_add_u32 m0, m0, 1024 // (op 1003) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 1003) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1003) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1003) most significant half

s_add_u32 m0, m0, 1024 // (op 1003) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 1003) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1003) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1003) most significant half

s_add_u32 m0, m0, 1024 // (op 1003) 
buffer_load_dword v66, s[32:35], s46 offen offset: 0 lds // (op 1003) Load value direct to lds




s_waitcnt vmcnt(32)
s_barrier  // (op 937) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 15 and the queue size is 48, so a waitcnt of 32 is required.

s_waitcnt lgkmcnt(7)
v_mfma_f32_32x32x8f16 a[0:15], v[32:33], v[16:17], a[0:15] // (op 514) WaitCnt Needed: Intersects with registers in 'DSQueue', at 8 and the queue size is 16, so a waitcnt of 7 is required.

s_barrier  // (op 1758) 




ds_read_b64 v[16:17], v4 // (op 54) Load local data 
s_lshl_b64 s[46:47], s[22:23], 3 // (op 1018) 



ds_read_b64 v[14:15], v4 offset:16 // (op 64) Load local data 
v_lshlrev_b64 v[48:49], 1, s[46:47] // (op 1018) 





ds_read_b64 v[20:21], v4 offset:32 // (op 73) Load local data 
v_mov_b64 v[66:67], v[52:53] // (op 1018) 



ds_read_b64 v[22:23], v4 offset:48 // (op 82) Load local data 
s_waitcnt lgkmcnt(10)
v_mfma_f32_32x32x8f16 a[0:15], v[34:35], v[64:65], a[0:15] // (op 519) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 11, so a waitcnt of 10 is required.





s_add_u32 s45, 49152, s43 // (op 1018) 
s_mov_b32 m0, s45 // (op 1018) reg expression
s_mov_b32 s46, 896 // (op 1018) call()
s_waitcnt lgkmcnt(9)
v_mfma_f32_32x32x8f16 a[0:15], v[36:37], v[62:63], a[0:15] // (op 524) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 10, so a waitcnt of 9 is required.

ds_read_b64 v[24:25], v4 offset:64 // (op 91) Load local data 




ds_read_b64 v[26:27], v4 offset:80 // (op 100) Load local data 
s_waitcnt lgkmcnt(10)
v_mfma_f32_32x32x8f16 a[0:15], v[40:41], v[60:61], a[0:15] // (op 529) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 11, so a waitcnt of 10 is required.





ds_read_b64 v[28:29], v4 offset:96 // (op 109) Load local data 



ds_read_b64 v[30:31], v4 offset:112 // (op 118) Load local data 
s_waitcnt lgkmcnt(11)
v_mfma_f32_32x32x8f16 a[0:15], v[72:73], v[58:59], a[0:15] // (op 534) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 12, so a waitcnt of 11 is required.






ds_read_b64 v[32:33], v0 offset:8192 // (op 49) Load local data 



ds_read_b64 v[34:35], v0 offset:8208 // (op 60) Load local data 
s_waitcnt lgkmcnt(12)
v_mfma_f32_32x32x8f16 a[0:15], v[50:51], v[56:57], a[0:15] // (op 539) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 13, so a waitcnt of 12 is required.






ds_read_b64 v[36:37], v0 offset:8224 // (op 69) Load local data 



ds_read_b64 v[38:39], v0 offset:8240 // (op 78) Load local data 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1018) Load value direct to lds



v_add_co_u32 v66, s[48:49], v66, v48 // (op 1018) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1018) most significant half

s_add_u32 m0, m0, 1024 // (op 1018) 
s_waitcnt lgkmcnt(13)
v_mfma_f32_32x32x8f16 a[0:15], v[74:75], v[54:55], a[0:15] // (op 544) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 14, so a waitcnt of 13 is required.

buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1018) Load value direct to lds


v_add_co_u32 v66, s[48:49], v66, v48 // (op 1018) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1018) most significant half

s_add_u32 m0, m0, 1024 // (op 1018) 
ds_read_b64 v[40:41], v0 offset:8256 // (op 87) Load local data 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1018) Load value direct to lds



ds_read_b64 v[42:43], v0 offset:8272 // (op 96) Load local data 
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1018) least significant half



s_waitcnt lgkmcnt(14)
v_mfma_f32_32x32x8f16 a[0:15], v[76:77], v[70:71], a[0:15] // (op 549) WaitCnt Needed: Intersects with registers in 'DSQueue', at 0 and the queue size is 15, so a waitcnt of 14 is required.

v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1018) most significant half



s_add_u32 m0, m0, 1024 // (op 1018) 
ds_read_b64 v[44:45], v0 offset:8288 // (op 105) Load local data 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1018) Load value direct to lds



ds_read_b64 v[46:47], v0 offset:8304 // (op 114) Load local data 
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1018) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1018) most significant half

s_add_u32 m0, m0, 1024 // (op 1018) 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1018) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1018) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1018) most significant half

s_add_u32 m0, m0, 1024 // (op 1018) 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1018) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1018) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1018) most significant half

s_add_u32 m0, m0, 1024 // (op 1018) 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1018) Load value direct to lds
v_add_co_u32 v66, s[48:49], v66, v48 // (op 1018) least significant half
v_addc_co_u32 v67, s[48:49], 0, v67, s[48:49] // (op 1018) most significant half

s_add_u32 m0, m0, 1024 // (op 1018) 
buffer_load_dword v66, s[36:39], s46 offen offset: 0 lds // (op 1018) Load value direct to lds





s_barrier  // (op 1755) 
v_add_co_u32 v18, vcc, 512, v18 // (op 1157) least significant half

v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 1157) most significant half
s_lshl_b64 s[46:47], s[18:19], 3 // (op 1013) 
v_lshlrev_b64 v[66:67], 1, s[46:47] // (op 1013) 


v_mov_b64 v[64:65], v[68:69] // (op 1013) 
s_add_u32 s45, 57344, s42 // (op 1013) 
s_mov_b32 m0, s45 // (op 1013) reg expression
s_mov_b32 s46, 896 // (op 1013) call()
buffer_load_dword v64, s[32:35], s46 offen offset: 0 lds // (op 1013) Load value direct to lds
v_add_co_u32 v64, s[48:49], v64, v66 // (op 1013) least significant half
v_addc_co_u32 v65, s[48:49], 0, v65, s[48:49] // (op 1013) most significant half

s_add_u32 m0, m0, 1024 // (op 1013) 
buffer_load_dword v64, s[32:35], s46 offen offset: 0 lds // (op 1013) Load value direct to lds
v_add_co_u32 v64, s[48:49], v64, v66 // (op 1013) least significant half
v_addc_co_u32 v65, s[48:49], 0, v65, s[48:49] // (op 1013) most significant half

s_add_u32 m0, m0, 1024 // (op 1013) 
buffer_load_dword v64, s[32:35], s46 offen offset: 0 lds // (op 1013) Load value direct to lds
v_add_co_u32 v64, s[48:49], v64, v66 // (op 1013) least significant half
v_addc_co_u32 v65, s[48:49], 0, v65, s[48:49] // (op 1013) most significant half

s_add_u32 m0, m0, 1024 // (op 1013) 
buffer_load_dword v64, s[32:35], s46 offen offset: 0 lds // (op 1013) Load value direct to lds
v_add_co_u32 v64, s[48:49], v64, v66 // (op 1013) least significant half
v_addc_co_u32 v65, s[48:49], 0, v65, s[48:49] // (op 1013) most significant half

s_add_u32 m0, m0, 1024 // (op 1013) 
buffer_load_dword v64, s[32:35], s46 offen offset: 0 lds // (op 1013) Load value direct to lds
v_add_co_u32 v64, s[48:49], v64, v66 // (op 1013) least significant half
v_addc_co_u32 v65, s[48:49], 0, v65, s[48:49] // (op 1013) most significant half

s_add_u32 m0, m0, 1024 // (op 1013) 
buffer_load_dword v64, s[32:35], s46 offen offset: 0 lds // (op 1013) Load value direct to lds
v_add_co_u32 v64, s[48:49], v64, v66 // (op 1013) least significant half
v_addc_co_u32 v65, s[48:49], 0, v65, s[48:49] // (op 1013) most significant half

s_add_u32 m0, m0, 1024 // (op 1013) 
buffer_load_dword v64, s[32:35], s46 offen offset: 0 lds // (op 1013) Load value direct to lds
v_add_co_u32 v64, s[48:49], v64, v66 // (op 1013) least significant half
v_addc_co_u32 v65, s[48:49], 0, v65, s[48:49] // (op 1013) most significant half

s_add_u32 m0, m0, 1024 // (op 1013) 
buffer_load_dword v64, s[32:35], s46 offen offset: 0 lds // (op 1013) Load value direct to lds





v_add_co_u32 v12, vcc, 512, v12 // (op 1093) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 1093) most significant half
s_ashr_i64 s[46:47], s[20:21], 6 // (op 39) 
s_mov_b32 s45, s46 // (op 39) convert

s_and_b32 s46, s45, 4294967292 // (op 39) 

s_cmp_lt_u32 s44, s46 // (op 39) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopTop_KLoop_39 // (op 39) Condition: Bottom (jump to RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopTop_KLoop_39 if true)
RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopBottom_KLoop_39:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// DEBUG: Wait after branch



























s_barrier  // (op 1767) 
v_and_b32 v12, -32, v1 // (op 1056) 
v_lshrrev_b32 v13, 5, v1 // (op 1056) 
v_sub_u32 v14, v1, v12 // (op 1056) 

v_lshl_add_u32 v12, s2, 6, v13 // (op 1056) 

v_lshlrev_b32 v13, 1, v14 // (op 1056) 

s_ashr_i64 s[0:1], s[20:21], 6 // (op 1056) 
s_lshr_b32 s17, s0, 2 // (op 1056) 

v_lshl_add_u32 v14, s17, 8, v13 // (op 1056) 


v_mov_b32 v17, 0 // (op 1056) convert
v_mov_b32 v16, v14 // (op 1056) convert

v_mul_lo_u32 v15, v12, s19 // (op 1056) most significant: low of low * high
v_mul_hi_u32 v13, v12, s18 // (op 1056) most significant: high of low * low
v_mul_lo_u32 v14, v12, s18 // (op 1056) least significant: low of low * low
v_add_u32 v15, v15, v13 // (op 1056) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v16 // (op 1056) least significant half
v_addc_co_u32 v15, s[0:1], v15, v17, s[0:1] // (op 1056) most significant half



v_lshlrev_b64 v[12:13], 1, v[14:15] // (op 1056) 

s_lshl_b64 s[0:1], s[4:5], 1 // (op 1058) 
s_mov_b32 s32, s6 // (op 1058) reg expression
s_mov_b32 s33, s7 // (op 1058) reg expression
s_mov_b32 s34, s0 // (op 1058) reg expression
s_mov_b32 s35, 131072 // (op 1058) call()

v_and_b32 v14, -32, v1 // (op 1123) 
v_lshrrev_b32 v15, 5, v1 // (op 1123) 
v_sub_u32 v16, v1, v14 // (op 1123) 

v_lshl_add_u32 v14, s3, 6, v15 // (op 1123) 

v_lshlrev_b32 v15, 1, v16 // (op 1123) 

v_mul_lo_u32 v17, v14, s23 // (op 1123) most significant: low of low * high
v_mul_hi_u32 v18, v14, s22 // (op 1123) most significant: high of low * low
v_mul_lo_u32 v16, v14, s22 // (op 1123) least significant: low of low * low
v_add_u32 v17, v17, v18 // (op 1123) most significant: sum


s_ashr_i64 s[0:1], s[20:21], 6 // (op 1123) 
s_lshr_b32 s17, s0, 2 // (op 1123) 

v_lshl_add_u32 v14, s17, 8, v15 // (op 1123) 


v_mov_b32 v19, 0 // (op 1123) convert
v_mov_b32 v18, v14 // (op 1123) convert

v_add_co_u32 v14, s[0:1], v18, v16 // (op 1123) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 1123) most significant half

v_lshlrev_b64 v[14:15], 1, v[14:15] // (op 1123) 


s_lshl_b64 s[0:1], s[8:9], 1 // (op 1125) 
s_mov_b32 s36, s10 // (op 1125) reg expression
s_mov_b32 s37, s11 // (op 1125) reg expression
s_mov_b32 s38, s0 // (op 1125) reg expression
s_mov_b32 s39, 131072 // (op 1125) call()

v_and_b32 v16, -32, v1 // (op 1216) 
v_lshrrev_b32 v17, 5, v1 // (op 1216) 
v_sub_u32 v18, v1, v16 // (op 1216) 

v_lshlrev_b32 v16, 1, v18 // (op 1216) 

v_lshl_add_u32 v18, v17, 6, v16 // (op 1216) 


v_lshlrev_b32 v17, 1, v18 // (op 1216) 

s_nop 0
v_readfirstlane_b32 s1, v17 // (op 1216) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -32, v1 // (op 1268) 
v_lshrrev_b32 v17, 5, v1 // (op 1268) 
v_sub_u32 v18, v1, v16 // (op 1268) 

v_lshlrev_b32 v16, 1, v18 // (op 1268) 

v_lshl_add_u32 v18, v17, 6, v16 // (op 1268) 


v_lshlrev_b32 v17, 1, v18 // (op 1268) 

s_nop 0
v_readfirstlane_b32 s0, v17 // (op 1268) Wait state hazard: VALU Write Readlane Hazard

s_ashr_i64 s[16:17], s[20:21], 6 // (op 218) 
s_and_b32 s40, s16, 4294967292 // (op 218) 

s_ashr_i64 s[16:17], s[20:21], 6 // (op 217) 
s_mov_b32 s41, s16 // (op 217) convert

s_cmp_lt_u32 s40, s41 // (op 217) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopBottom_KLoopTail_217 // (op 217) Condition: Top (jump to RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopBottom_KLoopTail_217 if false)
RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopTop_KLoopTail_217:
s_barrier  // (op 1726) 
s_add_u32 s40, s40, 1 // (op 219) 

s_lshl_b64 s[16:17], s[18:19], 3 // (op 981) 
v_lshlrev_b64 v[16:17], 1, s[16:17] // (op 981) 


v_mov_b64 v[18:19], v[12:13] // (op 981) 
v_mov_b64 v[20:21], v[18:19] // (op 981) 
s_add_u32 s17, 8192, s1 // (op 981) 
s_mov_b32 m0, s17 // (op 981) reg expression
buffer_load_dword v20, s[32:35], 0 offen offset: 0 lds // (op 981) Load value direct to lds
v_add_co_u32 v20, s[42:43], v20, v16 // (op 981) least significant half
v_addc_co_u32 v21, s[42:43], 0, v21, s[42:43] // (op 981) most significant half

s_add_u32 m0, m0, 1024 // (op 981) 
buffer_load_dword v20, s[32:35], 0 offen offset: 0 lds // (op 981) Load value direct to lds
v_add_co_u32 v20, s[42:43], v20, v16 // (op 981) least significant half
v_addc_co_u32 v21, s[42:43], 0, v21, s[42:43] // (op 981) most significant half

s_add_u32 m0, m0, 1024 // (op 981) 
buffer_load_dword v20, s[32:35], 0 offen offset: 0 lds // (op 981) Load value direct to lds
v_add_co_u32 v20, s[42:43], v20, v16 // (op 981) least significant half
v_addc_co_u32 v21, s[42:43], 0, v21, s[42:43] // (op 981) most significant half

s_add_u32 m0, m0, 1024 // (op 981) 
buffer_load_dword v20, s[32:35], 0 offen offset: 0 lds // (op 981) Load value direct to lds
v_add_co_u32 v20, s[42:43], v20, v16 // (op 981) least significant half
v_addc_co_u32 v21, s[42:43], 0, v21, s[42:43] // (op 981) most significant half

s_add_u32 m0, m0, 1024 // (op 981) 
buffer_load_dword v20, s[32:35], 0 offen offset: 0 lds // (op 981) Load value direct to lds
v_add_co_u32 v20, s[42:43], v20, v16 // (op 981) least significant half
v_addc_co_u32 v21, s[42:43], 0, v21, s[42:43] // (op 981) most significant half

s_add_u32 m0, m0, 1024 // (op 981) 
buffer_load_dword v20, s[32:35], 0 offen offset: 0 lds // (op 981) Load value direct to lds
v_add_co_u32 v20, s[42:43], v20, v16 // (op 981) least significant half
v_addc_co_u32 v21, s[42:43], 0, v21, s[42:43] // (op 981) most significant half

s_add_u32 m0, m0, 1024 // (op 981) 
buffer_load_dword v20, s[32:35], 0 offen offset: 0 lds // (op 981) Load value direct to lds
v_add_co_u32 v20, s[42:43], v20, v16 // (op 981) least significant half
v_addc_co_u32 v21, s[42:43], 0, v21, s[42:43] // (op 981) most significant half

s_add_u32 m0, m0, 1024 // (op 981) 
buffer_load_dword v20, s[32:35], 0 offen offset: 0 lds // (op 981) Load value direct to lds




v_add_co_u32 v12, vcc, 128, v12 // (op 1046) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 1046) most significant half
s_barrier  // (op 1736) 

s_lshl_b64 s[16:17], s[22:23], 3 // (op 987) 
v_lshlrev_b64 v[16:17], 1, s[16:17] // (op 987) 


v_mov_b64 v[18:19], v[14:15] // (op 987) 
v_mov_b64 v[20:21], v[18:19] // (op 987) 
s_mov_b32 m0, s0 // (op 987) reg expression
buffer_load_dword v20, s[36:39], 0 offen offset: 0 lds // (op 987) Load value direct to lds
v_add_co_u32 v20, s[16:17], v20, v16 // (op 987) least significant half
v_addc_co_u32 v21, s[16:17], 0, v21, s[16:17] // (op 987) most significant half

s_add_u32 m0, m0, 1024 // (op 987) 
buffer_load_dword v20, s[36:39], 0 offen offset: 0 lds // (op 987) Load value direct to lds
v_add_co_u32 v20, s[16:17], v20, v16 // (op 987) least significant half
v_addc_co_u32 v21, s[16:17], 0, v21, s[16:17] // (op 987) most significant half

s_add_u32 m0, m0, 1024 // (op 987) 
buffer_load_dword v20, s[36:39], 0 offen offset: 0 lds // (op 987) Load value direct to lds
v_add_co_u32 v20, s[16:17], v20, v16 // (op 987) least significant half
v_addc_co_u32 v21, s[16:17], 0, v21, s[16:17] // (op 987) most significant half

s_add_u32 m0, m0, 1024 // (op 987) 
buffer_load_dword v20, s[36:39], 0 offen offset: 0 lds // (op 987) Load value direct to lds
v_add_co_u32 v20, s[16:17], v20, v16 // (op 987) least significant half
v_addc_co_u32 v21, s[16:17], 0, v21, s[16:17] // (op 987) most significant half

s_add_u32 m0, m0, 1024 // (op 987) 
buffer_load_dword v20, s[36:39], 0 offen offset: 0 lds // (op 987) Load value direct to lds
v_add_co_u32 v20, s[16:17], v20, v16 // (op 987) least significant half
v_addc_co_u32 v21, s[16:17], 0, v21, s[16:17] // (op 987) most significant half

s_add_u32 m0, m0, 1024 // (op 987) 
buffer_load_dword v20, s[36:39], 0 offen offset: 0 lds // (op 987) Load value direct to lds
v_add_co_u32 v20, s[16:17], v20, v16 // (op 987) least significant half
v_addc_co_u32 v21, s[16:17], 0, v21, s[16:17] // (op 987) most significant half

s_add_u32 m0, m0, 1024 // (op 987) 
buffer_load_dword v20, s[36:39], 0 offen offset: 0 lds // (op 987) Load value direct to lds
v_add_co_u32 v20, s[16:17], v20, v16 // (op 987) least significant half
v_addc_co_u32 v21, s[16:17], 0, v21, s[16:17] // (op 987) most significant half

s_add_u32 m0, m0, 1024 // (op 987) 
buffer_load_dword v20, s[36:39], 0 offen offset: 0 lds // (op 987) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 1115) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1115) most significant half
s_waitcnt vmcnt(8)
s_barrier  // (op 1729) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 7 and the queue size is 16, so a waitcnt of 8 is required.




ds_read_b64 v[16:17], v0 offset:8192 // (op 225) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 1739) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 7 and the queue size is 8, so a waitcnt of 0 is required.




ds_read_b64 v[18:19], v4 // (op 227) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x8f16 a[0:15], v[16:17], v[18:19], a[0:15] // (op 228) WaitCnt Needed: Intersects with registers in 'DSQueue', at 1 and the queue size is 2, so a waitcnt of 0 is required.






ds_read_b64 v[16:17], v0 offset:8208 // (op 230) Load local data 



ds_read_b64 v[18:19], v4 offset:16 // (op 232) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x8f16 a[0:15], v[16:17], v[18:19], a[0:15] // (op 233) WaitCnt Needed: Intersects with registers in 'DSQueue', at 1 and the queue size is 2, so a waitcnt of 0 is required.






ds_read_b64 v[16:17], v0 offset:8224 // (op 235) Load local data 



ds_read_b64 v[18:19], v4 offset:32 // (op 237) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x8f16 a[0:15], v[16:17], v[18:19], a[0:15] // (op 238) WaitCnt Needed: Intersects with registers in 'DSQueue', at 1 and the queue size is 2, so a waitcnt of 0 is required.






ds_read_b64 v[16:17], v0 offset:8240 // (op 240) Load local data 



ds_read_b64 v[18:19], v4 offset:48 // (op 242) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x8f16 a[0:15], v[16:17], v[18:19], a[0:15] // (op 243) WaitCnt Needed: Intersects with registers in 'DSQueue', at 1 and the queue size is 2, so a waitcnt of 0 is required.






ds_read_b64 v[16:17], v0 offset:8256 // (op 245) Load local data 



ds_read_b64 v[18:19], v4 offset:64 // (op 247) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x8f16 a[0:15], v[16:17], v[18:19], a[0:15] // (op 248) WaitCnt Needed: Intersects with registers in 'DSQueue', at 1 and the queue size is 2, so a waitcnt of 0 is required.






ds_read_b64 v[16:17], v0 offset:8272 // (op 250) Load local data 



ds_read_b64 v[18:19], v4 offset:80 // (op 252) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x8f16 a[0:15], v[16:17], v[18:19], a[0:15] // (op 253) WaitCnt Needed: Intersects with registers in 'DSQueue', at 1 and the queue size is 2, so a waitcnt of 0 is required.






ds_read_b64 v[16:17], v0 offset:8288 // (op 255) Load local data 



ds_read_b64 v[18:19], v4 offset:96 // (op 257) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x8f16 a[0:15], v[16:17], v[18:19], a[0:15] // (op 258) WaitCnt Needed: Intersects with registers in 'DSQueue', at 1 and the queue size is 2, so a waitcnt of 0 is required.






ds_read_b64 v[16:17], v0 offset:8304 // (op 260) Load local data 



ds_read_b64 v[18:19], v4 offset:112 // (op 262) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x8f16 a[0:15], v[16:17], v[18:19], a[0:15] // (op 263) WaitCnt Needed: Intersects with registers in 'DSQueue', at 1 and the queue size is 2, so a waitcnt of 0 is required.



s_ashr_i64 s[16:17], s[20:21], 6 // (op 217) MFMA remaining: 8
s_mov_b32 s41, s16 // (op 217) convert

s_cmp_lt_u32 s40, s41 // (op 217) MFMA remaining: 6

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopTop_KLoopTail_217 // (op 217) Condition: Bottom (jump to RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopTop_KLoopTail_217 if true)
RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ForLoopBottom_KLoopTail_217:
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// DEBUG: Wait after branch














s_barrier  // (op 1801) MFMA remaining: 2








v_cmp_eq_f32 vcc, 0.00000, v9 // (op 177) MFMA remaining: 1
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ConditionalFalse_0_beta_177 // (op 177) Condition: False, jump to RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ConditionalFalse_0_beta_177
s_nop 3
v_accvgpr_read v12, a0 // (op 29) Wait state hazard: XDL Write Hazard
v_accvgpr_read v13, a1 // (op 29) 
v_accvgpr_read v14, a2 // (op 29) 
v_accvgpr_read v15, a3 // (op 29) 
v_accvgpr_read v16, a4 // (op 29) 
v_accvgpr_read v17, a5 // (op 29) 
v_accvgpr_read v18, a6 // (op 29) 
v_accvgpr_read v19, a7 // (op 29) 
v_accvgpr_read v20, a8 // (op 29) 
v_accvgpr_read v21, a9 // (op 29) 
v_accvgpr_read v22, a10 // (op 29) 
v_accvgpr_read v23, a11 // (op 29) 
v_accvgpr_read v24, a12 // (op 29) 
v_accvgpr_read v25, a13 // (op 29) 
v_accvgpr_read v26, a14 // (op 29) 
v_accvgpr_read v27, a15 // (op 29) 
v_mul_f32 v28, v11, v12 // (op 29) 
v_mul_f32 v29, v11, v13 // (op 29) 
v_mul_f32 v30, v11, v14 // (op 29) 
v_mul_f32 v31, v11, v15 // (op 29) 
v_mul_f32 v32, v11, v16 // (op 29) 
v_mul_f32 v33, v11, v17 // (op 29) 
v_mul_f32 v34, v11, v18 // (op 29) 
v_mul_f32 v35, v11, v19 // (op 29) 
v_mul_f32 v36, v11, v20 // (op 29) 
v_mul_f32 v37, v11, v21 // (op 29) 
v_mul_f32 v38, v11, v22 // (op 29) 
v_mul_f32 v39, v11, v23 // (op 29) 
v_mul_f32 v40, v11, v24 // (op 29) 
v_mul_f32 v41, v11, v25 // (op 29) 
v_mul_f32 v42, v11, v26 // (op 29) 
v_mul_f32 v43, v11, v27 // (op 29) 

v_cvt_f16_f32 v12, v28 // (op 35) 
v_cvt_f16_f32 v13, v29 // (op 35) 
v_cvt_f16_f32 v14, v30 // (op 35) 
v_cvt_f16_f32 v15, v31 // (op 35) 
v_cvt_f16_f32 v16, v32 // (op 35) 
v_cvt_f16_f32 v17, v33 // (op 35) 
v_cvt_f16_f32 v18, v34 // (op 35) 
v_cvt_f16_f32 v19, v35 // (op 35) 
v_cvt_f16_f32 v20, v36 // (op 35) 
v_cvt_f16_f32 v21, v37 // (op 35) 
v_cvt_f16_f32 v22, v38 // (op 35) 
v_cvt_f16_f32 v23, v39 // (op 35) 
v_cvt_f16_f32 v24, v40 // (op 35) 
v_cvt_f16_f32 v25, v41 // (op 35) 
v_cvt_f16_f32 v26, v42 // (op 35) 
v_cvt_f16_f32 v27, v43 // (op 35) 

v_and_b32 v0, 63, v1 // (op 1309) 
v_lshrrev_b32 v10, 6, v1 // (op 1309) 
v_lshrrev_b32 v8, 2, v0 // (op 1309) 
v_and_b32 v7, 3, v1 // (op 1309) 
v_and_b32 v6, 7, v8 // (op 1309) 

v_lshrrev_b32 v8, 5, v0 // (op 1309) 

v_lshl_add_u32 v0, v6, 2, v7 // (op 1309) 


v_and_b32 v7, 1, v10 // (op 1309) 

v_lshrrev_b32 v10, 7, v1 // (op 1309) 
v_lshl_add_u32 v6, v7, 5, v0 // (op 1309) 


v_lshlrev_b32 v0, 2, v8 // (op 1309) 

v_lshl_add_u32 v5, s3, 6, v6 // (op 1309) 

v_lshl_add_u32 v4, v10, 5, v0 // (op 1309) 


v_mul_lo_u32 v0, v5, s26 // (op 1309) 

v_lshl_add_u32 v3, s2, 6, v4 // (op 1309) 

v_add_lshl_u32 v10, v3, v0, 1 // (op 1309) 


s_lshl_b64 s[0:1], s[28:29], 1 // (op 1311) 
s_mov_b32 s8, s30 // (op 1311) reg expression
s_mov_b32 s9, s31 // (op 1311) reg expression
s_mov_b32 s10, s0 // (op 1311) reg expression
s_mov_b32 s11, 131072 // (op 1311) call()


v_pack_b32_f16 v28, v12, v13 // (op 37) 
v_pack_b32_f16 v29, v14, v15 // (op 37) 
v_pack_b32_f16 v30, v16, v17 // (op 37) 
v_pack_b32_f16 v31, v18, v19 // (op 37) 
v_pack_b32_f16 v32, v20, v21 // (op 37) 
v_pack_b32_f16 v33, v22, v23 // (op 37) 
v_pack_b32_f16 v34, v24, v25 // (op 37) 
v_pack_b32_f16 v35, v26, v27 // (op 37) 


buffer_store_dwordx2 v[28:29], v10, s[8:11], 0 offen // (op 37) Store value
buffer_store_dwordx2 v[30:31], v10, s[8:11], 0 offen offset:16 // (op 37) Store value
buffer_store_dwordx2 v[32:33], v10, s[8:11], 0 offen offset:32 // (op 37) Store value
buffer_store_dwordx2 v[34:35], v10, s[8:11], 0 offen offset:48 // (op 37) Store value



s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ConditionalBottom_0_beta_177 // (op 177) Condition: Done, jump to RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ConditionalBottom_0_beta_177
RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ConditionalFalse_0_beta_177:
v_and_b32 v0, 63, v1 // (op 1186) 
v_lshrrev_b32 v10, 6, v1 // (op 1186) 
v_lshrrev_b32 v8, 2, v0 // (op 1186) 
v_and_b32 v7, 3, v1 // (op 1186) 
v_and_b32 v6, 7, v8 // (op 1186) 

v_lshrrev_b32 v8, 5, v0 // (op 1186) 

v_lshl_add_u32 v0, v6, 2, v7 // (op 1186) 


v_and_b32 v7, 1, v10 // (op 1186) 

v_lshrrev_b32 v10, 7, v1 // (op 1186) 
v_lshl_add_u32 v6, v7, 5, v0 // (op 1186) 


v_lshlrev_b32 v0, 2, v8 // (op 1186) 

v_lshl_add_u32 v5, s3, 6, v6 // (op 1186) 

v_lshl_add_u32 v4, v10, 5, v0 // (op 1186) 


v_mul_lo_u32 v0, v5, s24 // (op 1186) 

v_lshl_add_u32 v3, s2, 6, v4 // (op 1186) 

v_add_lshl_u32 v10, v3, v0, 1 // (op 1186) 


s_lshl_b64 s[0:1], s[12:13], 1 // (op 1188) 
s_mov_b32 s4, s14 // (op 1188) reg expression
s_mov_b32 s5, s15 // (op 1188) reg expression
s_mov_b32 s6, s0 // (op 1188) reg expression
s_mov_b32 s7, 131072 // (op 1188) call()






buffer_load_dwordx2 v[12:13], v10, s[4:7], 0 offen // (op 190) Load value
buffer_load_dwordx2 v[14:15], v10, s[4:7], 0 offen offset:16 // (op 190) Load value
buffer_load_dwordx2 v[16:17], v10, s[4:7], 0 offen offset:32 // (op 190) Load value
buffer_load_dwordx2 v[18:19], v10, s[4:7], 0 offen offset:48 // (op 190) Load value



v_accvgpr_read v20, a0 // (op 186) 
v_accvgpr_read v21, a1 // (op 186) 
v_accvgpr_read v22, a2 // (op 186) 
v_accvgpr_read v23, a3 // (op 186) 
v_accvgpr_read v24, a4 // (op 186) 
v_accvgpr_read v25, a5 // (op 186) 
v_accvgpr_read v26, a6 // (op 186) 
v_accvgpr_read v27, a7 // (op 186) 
v_accvgpr_read v28, a8 // (op 186) 
v_accvgpr_read v29, a9 // (op 186) 
v_accvgpr_read v30, a10 // (op 186) 
v_accvgpr_read v31, a11 // (op 186) 
v_accvgpr_read v32, a12 // (op 186) 
v_accvgpr_read v33, a13 // (op 186) 
v_accvgpr_read v34, a14 // (op 186) 
v_accvgpr_read v35, a15 // (op 186) 
v_mul_f32 v36, v11, v20 // (op 186) 
v_mul_f32 v37, v11, v21 // (op 186) 
v_mul_f32 v38, v11, v22 // (op 186) 
v_mul_f32 v39, v11, v23 // (op 186) 
v_mul_f32 v40, v11, v24 // (op 186) 
v_mul_f32 v41, v11, v25 // (op 186) 
v_mul_f32 v42, v11, v26 // (op 186) 
v_mul_f32 v43, v11, v27 // (op 186) 
v_mul_f32 v44, v11, v28 // (op 186) 
v_mul_f32 v45, v11, v29 // (op 186) 
v_mul_f32 v46, v11, v30 // (op 186) 
v_mul_f32 v47, v11, v31 // (op 186) 
v_mul_f32 v48, v11, v32 // (op 186) 
v_mul_f32 v49, v11, v33 // (op 186) 
v_mul_f32 v50, v11, v34 // (op 186) 
v_mul_f32 v51, v11, v35 // (op 186) 

s_waitcnt vmcnt(3)
v_cvt_f32_f16 v4, v12 // (op 191) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 0 and the queue size is 4, so a waitcnt of 3 is required.

v_lshrrev_b32 v5, 16, v12 // (op 191) 
v_cvt_f32_f16 v5, v5 // (op 191) 
v_mul_f32 v20, v9, v4 // (op 191) 
v_mul_f32 v21, v9, v5 // (op 191) 
v_cvt_f32_f16 v4, v13 // (op 191) 
v_lshrrev_b32 v5, 16, v13 // (op 191) 
v_cvt_f32_f16 v5, v5 // (op 191) 
v_mul_f32 v22, v9, v4 // (op 191) 
v_mul_f32 v23, v9, v5 // (op 191) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v4, v14 // (op 191) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 0 and the queue size is 3, so a waitcnt of 2 is required.

v_lshrrev_b32 v5, 16, v14 // (op 191) 
v_cvt_f32_f16 v5, v5 // (op 191) 
v_mul_f32 v24, v9, v4 // (op 191) 
v_mul_f32 v25, v9, v5 // (op 191) 
v_cvt_f32_f16 v4, v15 // (op 191) 
v_lshrrev_b32 v5, 16, v15 // (op 191) 
v_cvt_f32_f16 v5, v5 // (op 191) 
v_mul_f32 v26, v9, v4 // (op 191) 
v_mul_f32 v27, v9, v5 // (op 191) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v4, v16 // (op 191) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 0 and the queue size is 2, so a waitcnt of 1 is required.

v_lshrrev_b32 v5, 16, v16 // (op 191) 
v_cvt_f32_f16 v5, v5 // (op 191) 
v_mul_f32 v28, v9, v4 // (op 191) 
v_mul_f32 v29, v9, v5 // (op 191) 
v_cvt_f32_f16 v4, v17 // (op 191) 
v_lshrrev_b32 v5, 16, v17 // (op 191) 
v_cvt_f32_f16 v5, v5 // (op 191) 
v_mul_f32 v30, v9, v4 // (op 191) 
v_mul_f32 v31, v9, v5 // (op 191) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v4, v18 // (op 191) WaitCnt Needed: Intersects with registers in 'LoadQueue', at 0 and the queue size is 1, so a waitcnt of 0 is required.

v_lshrrev_b32 v5, 16, v18 // (op 191) 
v_cvt_f32_f16 v5, v5 // (op 191) 
v_mul_f32 v32, v9, v4 // (op 191) 
v_mul_f32 v33, v9, v5 // (op 191) 
v_cvt_f32_f16 v4, v19 // (op 191) 
v_lshrrev_b32 v5, 16, v19 // (op 191) 
v_cvt_f32_f16 v5, v5 // (op 191) 
v_mul_f32 v34, v9, v4 // (op 191) 
v_mul_f32 v35, v9, v5 // (op 191) 


v_add_f32 v52, v20, v36 // (op 187) 
v_add_f32 v53, v21, v37 // (op 187) 
v_add_f32 v54, v22, v38 // (op 187) 
v_add_f32 v55, v23, v39 // (op 187) 
v_add_f32 v56, v24, v40 // (op 187) 
v_add_f32 v57, v25, v41 // (op 187) 
v_add_f32 v58, v26, v42 // (op 187) 
v_add_f32 v59, v27, v43 // (op 187) 
v_add_f32 v60, v28, v44 // (op 187) 
v_add_f32 v61, v29, v45 // (op 187) 
v_add_f32 v62, v30, v46 // (op 187) 
v_add_f32 v63, v31, v47 // (op 187) 
v_add_f32 v64, v32, v48 // (op 187) 
v_add_f32 v65, v33, v49 // (op 187) 
v_add_f32 v66, v34, v50 // (op 187) 
v_add_f32 v67, v35, v51 // (op 187) 


v_cvt_f16_f32 v36, v52 // (op 188) 
v_cvt_f16_f32 v37, v53 // (op 188) 
v_cvt_f16_f32 v38, v54 // (op 188) 
v_cvt_f16_f32 v39, v55 // (op 188) 
v_cvt_f16_f32 v40, v56 // (op 188) 
v_cvt_f16_f32 v41, v57 // (op 188) 
v_cvt_f16_f32 v42, v58 // (op 188) 
v_cvt_f16_f32 v43, v59 // (op 188) 
v_cvt_f16_f32 v44, v60 // (op 188) 
v_cvt_f16_f32 v45, v61 // (op 188) 
v_cvt_f16_f32 v46, v62 // (op 188) 
v_cvt_f16_f32 v47, v63 // (op 188) 
v_cvt_f16_f32 v48, v64 // (op 188) 
v_cvt_f16_f32 v49, v65 // (op 188) 
v_cvt_f16_f32 v50, v66 // (op 188) 
v_cvt_f16_f32 v51, v67 // (op 188) 

v_and_b32 v0, 63, v1 // (op 1326) 
v_lshrrev_b32 v10, 6, v1 // (op 1326) 
v_lshrrev_b32 v8, 2, v0 // (op 1326) 
v_and_b32 v7, 3, v1 // (op 1326) 
v_and_b32 v6, 7, v8 // (op 1326) 

v_lshrrev_b32 v8, 5, v0 // (op 1326) 

v_lshl_add_u32 v0, v6, 2, v7 // (op 1326) 


v_and_b32 v7, 1, v10 // (op 1326) 

v_lshrrev_b32 v10, 7, v1 // (op 1326) 
v_lshl_add_u32 v6, v7, 5, v0 // (op 1326) 


v_lshlrev_b32 v0, 2, v8 // (op 1326) 

v_lshl_add_u32 v5, s3, 6, v6 // (op 1326) 

v_lshl_add_u32 v4, v10, 5, v0 // (op 1326) 


v_mul_lo_u32 v0, v5, s26 // (op 1326) 

v_lshl_add_u32 v3, s2, 6, v4 // (op 1326) 

v_add_lshl_u32 v10, v3, v0, 1 // (op 1326) 


s_lshl_b64 s[0:1], s[28:29], 1 // (op 1328) 
s_mov_b32 s8, s30 // (op 1328) reg expression
s_mov_b32 s9, s31 // (op 1328) reg expression
s_mov_b32 s10, s0 // (op 1328) reg expression
s_mov_b32 s11, 131072 // (op 1328) call()


v_pack_b32_f16 v28, v36, v37 // (op 189) 
v_pack_b32_f16 v29, v38, v39 // (op 189) 
v_pack_b32_f16 v30, v40, v41 // (op 189) 
v_pack_b32_f16 v31, v42, v43 // (op 189) 
v_pack_b32_f16 v32, v44, v45 // (op 189) 
v_pack_b32_f16 v33, v46, v47 // (op 189) 
v_pack_b32_f16 v34, v48, v49 // (op 189) 
v_pack_b32_f16 v35, v50, v51 // (op 189) 


buffer_store_dwordx2 v[28:29], v10, s[8:11], 0 offen // (op 189) Store value
buffer_store_dwordx2 v[30:31], v10, s[8:11], 0 offen offset:16 // (op 189) Store value
buffer_store_dwordx2 v[32:33], v10, s[8:11], 0 offen offset:32 // (op 189) Store value
buffer_store_dwordx2 v[34:35], v10, s[8:11], 0 offen offset:48 // (op 189) Store value







RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_ConditionalBottom_0_beta_177:



s_endpgm  // End of RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e
.LRRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_end:

.size RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e, .LRRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e_end-RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e
.rodata
.p2align 6
.amdhsa_kernel RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e
  .amdhsa_next_free_vgpr 96
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 65536 // lds bytes
  .amdhsa_accum_offset 80
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
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
    DataType: h
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: 1
    TransposeB: 0
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
  CustomKernel:
    args: [ { type: int64, semantic: StrideA1 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB1 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideC1 },
            { type: address, semantic: AddressC },
            { type: float32, semantic: Alpha },
            { type: float32, semantic: Beta },
            { type: int64, semantic: StrideA0 },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideB0 },
            { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: int64, semantic: StrideD1 },
            { type: address, semantic: AddressD } ]
    macrotile: [64, 64, 64]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 16, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e
    .symbol: RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e.kd
    .kernarg_segment_size: 112
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 50
    .vgpr_count: 78
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_8, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 56, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_11, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 80, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_extent_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_extent, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_extent_2
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_extent, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 48, name: Tensor_2_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_extent_4
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_extent, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 96, name: Tensor_4_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: user_Float_Value_6
        .size: 4
        .offset: 48
        .expression: {type: CommandArgument, size: 4, offset: 144, name: user_Float_Value_6, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_8
        .size: 4
        .offset: 52
        .expression: {type: CommandArgument, size: 4, offset: 148, name: user_Float_Value_8, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_stride_0_9
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 32, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_1_10
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_12
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 88, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_stride_1_13
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 136, name: Tensor_4_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_15_stride_1_14
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_15_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_15_extent_15
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_15_extent, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_15_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_15_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata

