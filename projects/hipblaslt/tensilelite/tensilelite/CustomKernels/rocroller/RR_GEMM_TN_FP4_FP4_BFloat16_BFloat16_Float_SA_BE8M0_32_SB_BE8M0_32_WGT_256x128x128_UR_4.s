// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4
.p2align 8
.type RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4,@function
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_exec_begin:


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


v_and_b32 v0, 4294967168, v1 // (op 20799) 
v_and_b32 v3, 15, v1 // (op 20799) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 20799) 


v_and_b32 v0, 63, v1 // (op 20799) 
v_lshrrev_b32 v3, 4, v0 // (op 20799) 

v_lshlrev_b32 v0, 5, v3 // (op 20799) 

v_add_lshl_u32 v3, v4, v0, 2 // (op 20799) 


v_lshrrev_b32 v0, 3, v3 // (op 20799) 

v_lshrrev_b32 v3, 6, v1 // (op 20856) 
v_and_b32 v4, 1, v3 // (op 20856) 

v_and_b32 v3, 15, v1 // (op 20856) 
v_lshl_add_u32 v5, v4, 6, v3 // (op 20856) 


v_and_b32 v3, 63, v1 // (op 20856) 
v_lshrrev_b32 v4, 4, v3 // (op 20856) 

v_lshlrev_b32 v3, 5, v4 // (op 20856) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 20856) 


v_lshlrev_b32 v3, 2, v4 // (op 20856) 

v_lshrrev_b32 v4, 3, v3 // (op 20856) 

v_and_b32 v3, 4294967168, v1 // (op 20923) 
v_and_b32 v5, 15, v1 // (op 20923) 
v_add_lshl_u32 v6, v3, v5, 7 // (op 20923) 


v_and_b32 v3, 63, v1 // (op 20923) 
v_lshrrev_b32 v5, 4, v3 // (op 20923) 

v_lshlrev_b32 v3, 5, v5 // (op 20923) 

v_add_lshl_u32 v5, v6, v3, 2 // (op 20923) 


v_lshrrev_b32 v3, 3, v5 // (op 20923) 

v_lshrrev_b32 v5, 6, v1 // (op 20967) 
v_and_b32 v6, 1, v5 // (op 20967) 

v_and_b32 v5, 15, v1 // (op 20967) 
v_lshl_add_u32 v7, v6, 6, v5 // (op 20967) 


v_and_b32 v5, 63, v1 // (op 20967) 
v_lshrrev_b32 v6, 4, v5 // (op 20967) 

v_lshlrev_b32 v5, 5, v6 // (op 20967) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 20967) 


v_lshlrev_b32 v5, 2, v6 // (op 20967) 

v_lshrrev_b32 v6, 3, v5 // (op 20967) 

v_and_b32 v5, 4294967168, v1 // (op 21011) 
v_and_b32 v7, 15, v1 // (op 21011) 
v_add_lshl_u32 v8, v5, v7, 7 // (op 21011) 


v_and_b32 v5, 63, v1 // (op 21011) 
v_lshrrev_b32 v7, 4, v5 // (op 21011) 

v_lshlrev_b32 v5, 5, v7 // (op 21011) 

v_add_lshl_u32 v7, v8, v5, 2 // (op 21011) 


v_lshrrev_b32 v5, 3, v7 // (op 21011) 

v_lshrrev_b32 v7, 6, v1 // (op 21055) 
v_and_b32 v8, 1, v7 // (op 21055) 

v_and_b32 v7, 15, v1 // (op 21055) 
v_lshl_add_u32 v9, v8, 6, v7 // (op 21055) 


v_and_b32 v7, 63, v1 // (op 21055) 
v_lshrrev_b32 v8, 4, v7 // (op 21055) 

v_lshlrev_b32 v7, 5, v8 // (op 21055) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 21055) 


v_lshlrev_b32 v7, 2, v8 // (op 21055) 

v_lshrrev_b32 v8, 3, v7 // (op 21055) 

v_and_b32 v7, 4294967168, v1 // (op 21099) 
v_and_b32 v9, 15, v1 // (op 21099) 
v_add_lshl_u32 v10, v7, v9, 7 // (op 21099) 


v_and_b32 v7, 63, v1 // (op 21099) 
v_lshrrev_b32 v9, 4, v7 // (op 21099) 

v_lshlrev_b32 v7, 5, v9 // (op 21099) 

v_add_lshl_u32 v9, v10, v7, 2 // (op 21099) 


v_lshrrev_b32 v7, 3, v9 // (op 21099) 

v_lshrrev_b32 v9, 6, v1 // (op 21143) 
v_and_b32 v10, 1, v9 // (op 21143) 

v_and_b32 v9, 15, v1 // (op 21143) 
v_lshl_add_u32 v11, v10, 6, v9 // (op 21143) 


v_and_b32 v9, 63, v1 // (op 21143) 
v_lshrrev_b32 v10, 4, v9 // (op 21143) 

v_lshlrev_b32 v9, 5, v10 // (op 21143) 

v_lshl_add_u32 v10, v11, 7, v9 // (op 21143) 


v_lshlrev_b32 v9, 2, v10 // (op 21143) 

v_lshrrev_b32 v10, 3, v9 // (op 21143) 

v_and_b32 v9, 4294967168, v1 // (op 20738) 
v_and_b32 v11, 63, v1 // (op 20738) 
v_lshrrev_b32 v12, 4, v11 // (op 20738) 
v_lshlrev_b32 v13, 2, v12 // (op 20738) 

v_add_u32 v12, v9, v13 // (op 20738) 


v_lshl_add_u32 v9, s16, 8, v12 // (op 20738) 

v_lshrrev_b32 v12, 2, v11 // (op 20738) 

v_and_b32 v11, 3, v12 // (op 20738) 

v_and_b32 v12, 3, v1 // (op 20738) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 20738) 


v_lshrrev_b32 v11, 6, v1 // (op 20738) 
v_and_b32 v12, 1, v11 // (op 20738) 

v_lshl_add_u32 v11, v12, 6, v13 // (op 20738) 


v_lshl_add_u32 v12, s17, 7, v11 // (op 20738) 

v_mul_lo_u32 v11, v12, s2 // (op 20738) 

v_add_lshl_u32 v12, v9, v11, 1 // (op 20738) 


v_and_b32 v9, 4294967168, v1 // (op 20877) 
v_and_b32 v11, 63, v1 // (op 20877) 
v_lshrrev_b32 v13, 4, v11 // (op 20877) 
s_mul_i32 s1, s2, s9 // (op 20740) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 20740) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 20740) least significant
s_mul_hi_u32 s19, s2, s8 // (op 20740) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 20740) most significant: sum
s_add_u32 s1, s1, s19 // (op 20740) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 20740) 

v_lshlrev_b32 v14, 2, v13 // (op 20877) 

v_add_u32 v13, v9, v14 // (op 20877) 


v_lshl_add_u32 v9, s16, 8, v13 // (op 20877) 

v_lshrrev_b32 v13, 2, v11 // (op 20877) 

v_and_b32 v11, 3, v13 // (op 20877) 

v_and_b32 v13, 3, v1 // (op 20877) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 20877) 


v_lshrrev_b32 v11, 6, v1 // (op 20877) 
v_and_b32 v13, 1, v11 // (op 20877) 

v_lshl_add_u32 v11, v13, 6, v14 // (op 20877) 


v_lshl_add_u32 v13, s17, 7, v11 // (op 20877) 

v_mul_lo_u32 v11, v13, s4 // (op 20877) 

v_add_lshl_u32 v13, v9, v11, 1 // (op 20877) 


s_mov_b32 s40, s6 // (op 20740) reg expression
s_mov_b32 s41, s7 // (op 20740) reg expression
s_mov_b32 s42, s18 // (op 20740) reg expression
s_mov_b32 s43, 131072 // (op 20740) call()

s_mul_i32 s1, s4, s9 // (op 20879) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 20879) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 20879) least significant
s_mul_hi_u32 s7, s4, s8 // (op 20879) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20879) most significant: sum
s_add_u32 s1, s1, s7 // (op 20879) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 20879) 

s_mov_b32 s44, s10 // (op 20879) reg expression
s_mov_b32 s45, s11 // (op 20879) reg expression
s_mov_b32 s46, s6 // (op 20879) reg expression
s_mov_b32 s47, 131072 // (op 20879) call()


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_accvgpr_write a4, 0.00000 // (op 387) call()
v_accvgpr_write a5, 0.00000 // (op 387) call()
v_accvgpr_write a6, 0.00000 // (op 387) call()
v_accvgpr_write a7, 0.00000 // (op 387) call()
v_accvgpr_write a8, 0.00000 // (op 613) call()
v_accvgpr_write a9, 0.00000 // (op 613) call()
v_accvgpr_write a10, 0.00000 // (op 613) call()
v_accvgpr_write a11, 0.00000 // (op 613) call()
v_accvgpr_write a12, 0.00000 // (op 839) call()
v_accvgpr_write a13, 0.00000 // (op 839) call()
v_accvgpr_write a14, 0.00000 // (op 839) call()
v_accvgpr_write a15, 0.00000 // (op 839) call()
v_accvgpr_write a16, 0.00000 // (op 1201) call()
v_accvgpr_write a17, 0.00000 // (op 1201) call()
v_accvgpr_write a18, 0.00000 // (op 1201) call()
v_accvgpr_write a19, 0.00000 // (op 1201) call()
v_accvgpr_write a20, 0.00000 // (op 1614) call()
v_accvgpr_write a21, 0.00000 // (op 1614) call()
v_accvgpr_write a22, 0.00000 // (op 1614) call()
v_accvgpr_write a23, 0.00000 // (op 1614) call()
v_accvgpr_write a24, 0.00000 // (op 1615) call()
v_accvgpr_write a25, 0.00000 // (op 1615) call()
v_accvgpr_write a26, 0.00000 // (op 1615) call()
v_accvgpr_write a27, 0.00000 // (op 1615) call()
v_accvgpr_write a28, 0.00000 // (op 1616) call()
v_accvgpr_write a29, 0.00000 // (op 1616) call()
v_accvgpr_write a30, 0.00000 // (op 1616) call()
v_accvgpr_write a31, 0.00000 // (op 1616) call()
v_accvgpr_write a32, 0.00000 // (op 2203) call()
v_accvgpr_write a33, 0.00000 // (op 2203) call()
v_accvgpr_write a34, 0.00000 // (op 2203) call()
v_accvgpr_write a35, 0.00000 // (op 2203) call()
v_accvgpr_write a36, 0.00000 // (op 2616) call()
v_accvgpr_write a37, 0.00000 // (op 2616) call()
v_accvgpr_write a38, 0.00000 // (op 2616) call()
v_accvgpr_write a39, 0.00000 // (op 2616) call()
v_accvgpr_write a40, 0.00000 // (op 2617) call()
v_accvgpr_write a41, 0.00000 // (op 2617) call()
v_accvgpr_write a42, 0.00000 // (op 2617) call()
v_accvgpr_write a43, 0.00000 // (op 2617) call()
v_accvgpr_write a44, 0.00000 // (op 2618) call()
v_accvgpr_write a45, 0.00000 // (op 2618) call()
v_accvgpr_write a46, 0.00000 // (op 2618) call()
v_accvgpr_write a47, 0.00000 // (op 2618) call()
v_accvgpr_write a48, 0.00000 // (op 3205) call()
v_accvgpr_write a49, 0.00000 // (op 3205) call()
v_accvgpr_write a50, 0.00000 // (op 3205) call()
v_accvgpr_write a51, 0.00000 // (op 3205) call()
v_accvgpr_write a52, 0.00000 // (op 3618) call()
v_accvgpr_write a53, 0.00000 // (op 3618) call()
v_accvgpr_write a54, 0.00000 // (op 3618) call()
v_accvgpr_write a55, 0.00000 // (op 3618) call()
v_accvgpr_write a56, 0.00000 // (op 3619) call()
v_accvgpr_write a57, 0.00000 // (op 3619) call()
v_accvgpr_write a58, 0.00000 // (op 3619) call()
v_accvgpr_write a59, 0.00000 // (op 3619) call()
v_accvgpr_write a60, 0.00000 // (op 3620) call()
v_accvgpr_write a61, 0.00000 // (op 3620) call()
v_accvgpr_write a62, 0.00000 // (op 3620) call()
v_accvgpr_write a63, 0.00000 // (op 3620) call()
v_accvgpr_write a64, 0.00000 // (op 4207) call()
v_accvgpr_write a65, 0.00000 // (op 4207) call()
v_accvgpr_write a66, 0.00000 // (op 4207) call()
v_accvgpr_write a67, 0.00000 // (op 4207) call()
v_accvgpr_write a68, 0.00000 // (op 4620) call()
v_accvgpr_write a69, 0.00000 // (op 4620) call()
v_accvgpr_write a70, 0.00000 // (op 4620) call()
v_accvgpr_write a71, 0.00000 // (op 4620) call()
v_accvgpr_write a72, 0.00000 // (op 4621) call()
v_accvgpr_write a73, 0.00000 // (op 4621) call()
v_accvgpr_write a74, 0.00000 // (op 4621) call()
v_accvgpr_write a75, 0.00000 // (op 4621) call()
v_accvgpr_write a76, 0.00000 // (op 4622) call()
v_accvgpr_write a77, 0.00000 // (op 4622) call()
v_accvgpr_write a78, 0.00000 // (op 4622) call()
v_accvgpr_write a79, 0.00000 // (op 4622) call()
v_accvgpr_write a80, 0.00000 // (op 5209) call()
v_accvgpr_write a81, 0.00000 // (op 5209) call()
v_accvgpr_write a82, 0.00000 // (op 5209) call()
v_accvgpr_write a83, 0.00000 // (op 5209) call()
v_accvgpr_write a84, 0.00000 // (op 5622) call()
v_accvgpr_write a85, 0.00000 // (op 5622) call()
v_accvgpr_write a86, 0.00000 // (op 5622) call()
v_accvgpr_write a87, 0.00000 // (op 5622) call()
v_accvgpr_write a88, 0.00000 // (op 5623) call()
v_accvgpr_write a89, 0.00000 // (op 5623) call()
v_accvgpr_write a90, 0.00000 // (op 5623) call()
v_accvgpr_write a91, 0.00000 // (op 5623) call()
v_accvgpr_write a92, 0.00000 // (op 5624) call()
v_accvgpr_write a93, 0.00000 // (op 5624) call()
v_accvgpr_write a94, 0.00000 // (op 5624) call()
v_accvgpr_write a95, 0.00000 // (op 5624) call()
v_accvgpr_write a96, 0.00000 // (op 6211) call()
v_accvgpr_write a97, 0.00000 // (op 6211) call()
v_accvgpr_write a98, 0.00000 // (op 6211) call()
v_accvgpr_write a99, 0.00000 // (op 6211) call()
v_accvgpr_write a100, 0.00000 // (op 6624) call()
v_accvgpr_write a101, 0.00000 // (op 6624) call()
v_accvgpr_write a102, 0.00000 // (op 6624) call()
v_accvgpr_write a103, 0.00000 // (op 6624) call()
v_accvgpr_write a104, 0.00000 // (op 6625) call()
v_accvgpr_write a105, 0.00000 // (op 6625) call()
v_accvgpr_write a106, 0.00000 // (op 6625) call()
v_accvgpr_write a107, 0.00000 // (op 6625) call()
v_accvgpr_write a108, 0.00000 // (op 6626) call()
v_accvgpr_write a109, 0.00000 // (op 6626) call()
v_accvgpr_write a110, 0.00000 // (op 6626) call()
v_accvgpr_write a111, 0.00000 // (op 6626) call()
v_accvgpr_write a112, 0.00000 // (op 7213) call()
v_accvgpr_write a113, 0.00000 // (op 7213) call()
v_accvgpr_write a114, 0.00000 // (op 7213) call()
v_accvgpr_write a115, 0.00000 // (op 7213) call()
v_accvgpr_write a116, 0.00000 // (op 7626) call()
v_accvgpr_write a117, 0.00000 // (op 7626) call()
v_accvgpr_write a118, 0.00000 // (op 7626) call()
v_accvgpr_write a119, 0.00000 // (op 7626) call()
v_accvgpr_write a120, 0.00000 // (op 7627) call()
v_accvgpr_write a121, 0.00000 // (op 7627) call()
v_accvgpr_write a122, 0.00000 // (op 7627) call()
v_accvgpr_write a123, 0.00000 // (op 7627) call()
v_accvgpr_write a124, 0.00000 // (op 7628) call()
v_accvgpr_write a125, 0.00000 // (op 7628) call()
v_accvgpr_write a126, 0.00000 // (op 7628) call()
v_accvgpr_write a127, 0.00000 // (op 7628) call()
v_lshrrev_b32 v14, 2, v1 // (op 20628) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 20628) 

v_and_b32 v14, -4, v1 // (op 20628) 
v_sub_u32 v16, v1, v14 // (op 20628) 

v_lshlrev_b32 v14, 5, v16 // (op 20628) 

v_mov_b32 v17, 0 // (op 20628) convert
v_mov_b32 v16, v14 // (op 20628) convert

v_mul_lo_u32 v19, v15, s13 // (op 20628) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 20628) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 20628) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 20628) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 20628) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 20628) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 20628) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 20628) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 20630) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 20630) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 20630) least significant
s_mul_hi_u32 s7, s12, s22 // (op 20630) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20630) most significant: sum
s_add_u32 s1, s1, s7 // (op 20630) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20630) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20630) 

s_mov_b32 s48, s20 // (op 20630) reg expression
s_mov_b32 s49, s21 // (op 20630) reg expression
s_mov_b32 s50, s0 // (op 20630) reg expression
s_mov_b32 s51, 131072 // (op 20630) call()

v_lshrrev_b32 v14, 2, v1 // (op 20647) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 20647) 

v_and_b32 v14, -4, v1 // (op 20647) 
v_sub_u32 v18, v1, v14 // (op 20647) 

v_lshlrev_b32 v14, 5, v18 // (op 20647) 

v_mov_b32 v19, 0 // (op 20647) convert
v_mov_b32 v18, v14 // (op 20647) convert

v_mul_lo_u32 v21, v15, s13 // (op 20647) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 20647) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 20647) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 20647) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 20647) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 20647) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 20647) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 20647) 

s_mul_i32 s1, s12, s23 // (op 20649) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 20649) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 20649) least significant
s_mul_hi_u32 s7, s12, s22 // (op 20649) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20649) most significant: sum
s_add_u32 s1, s1, s7 // (op 20649) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20649) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20649) 

s_mov_b32 s48, s20 // (op 20649) reg expression
s_mov_b32 s49, s21 // (op 20649) reg expression
s_mov_b32 s50, s0 // (op 20649) reg expression
s_mov_b32 s51, 131072 // (op 20649) call()

v_and_b32 v14, -4, v1 // (op 20692) 
v_sub_u32 v15, v1, v14 // (op 20692) 

v_lshlrev_b32 v14, 5, v15 // (op 20692) 

v_mov_b32 v21, 0 // (op 20692) convert
v_mov_b32 v20, v14 // (op 20692) convert

v_lshrrev_b32 v14, 2, v1 // (op 20692) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 20692) 

v_mul_lo_u32 v23, v15, s25 // (op 20692) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 20692) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 20692) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 20692) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 20692) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 20692) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20692) 


v_lshrrev_b64 v[20:21], 3, v[14:15] // (op 20692) 

s_mul_i32 s1, s24, s9 // (op 20694) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 20694) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 20694) least significant
s_mul_hi_u32 s7, s24, s8 // (op 20694) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20694) most significant: sum
s_add_u32 s1, s1, s7 // (op 20694) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20694) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20694) 

s_mov_b32 s52, s26 // (op 20694) reg expression
s_mov_b32 s53, s27 // (op 20694) reg expression
s_mov_b32 s54, s0 // (op 20694) reg expression
s_mov_b32 s55, 131072 // (op 20694) call()

v_and_b32 v14, -4, v1 // (op 20711) 
v_sub_u32 v15, v1, v14 // (op 20711) 

v_lshlrev_b32 v14, 5, v15 // (op 20711) 

v_mov_b32 v23, 0 // (op 20711) convert
v_mov_b32 v22, v14 // (op 20711) convert

v_lshrrev_b32 v14, 2, v1 // (op 20711) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 20711) 

v_mul_lo_u32 v25, v15, s25 // (op 20711) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 20711) most significant: high of low * low
v_mul_lo_u32 v24, v15, s24 // (op 20711) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 20711) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 20711) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 20711) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20711) 


v_lshrrev_b64 v[22:23], 3, v[14:15] // (op 20711) 

s_mul_i32 s1, s24, s9 // (op 20713) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 20713) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 20713) least significant
s_mul_hi_u32 s7, s24, s8 // (op 20713) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20713) most significant: sum
s_add_u32 s1, s1, s7 // (op 20713) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20713) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20713) 

s_mov_b32 s52, s26 // (op 20713) reg expression
s_mov_b32 s53, s27 // (op 20713) reg expression
s_mov_b32 s54, s0 // (op 20713) reg expression
s_mov_b32 s55, 131072 // (op 20713) call()

v_and_b32 v14, 63, v1 // (op 21165) 
v_lshrrev_b32 v15, 4, v14 // (op 21165) 
v_and_b32 v24, 3, v15 // (op 21165) 

v_and_b32 v15, 15, v1 // (op 21165) 
v_lshl_add_u32 v25, v24, 4, v15 // (op 21165) 


v_and_b32 v15, 4294967168, v1 // (op 21165) 
v_add_u32 v24, v15, v25 // (op 21165) 


v_lshl_add_u32 v15, s16, 8, v24 // (op 21165) 

v_lshrrev_b32 v24, 6, v14 // (op 21165) 

v_lshlrev_b32 v14, 2, v24 // (op 21165) 

v_mul_lo_u32 v24, v15, s28 // (op 21165) 
v_add_u32 v24, v24, v14 // (op 21165) 


s_mul_i32 s1, s28, s23 // (op 21167) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 21167) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 21167) least significant
s_mul_hi_u32 s7, s28, s22 // (op 21167) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 21167) most significant: sum
s_add_u32 s1, s1, s7 // (op 21167) most significant: sum


s_mov_b32 s56, s30 // (op 21167) reg expression
s_mov_b32 s57, s31 // (op 21167) reg expression
s_mov_b32 s58, s0 // (op 21167) reg expression
s_mov_b32 s59, 131072 // (op 21167) call()

v_and_b32 v14, 63, v1 // (op 21186) 
v_lshrrev_b32 v15, 4, v14 // (op 21186) 
v_and_b32 v25, 3, v15 // (op 21186) 

v_and_b32 v15, 15, v1 // (op 21186) 
v_lshl_add_u32 v26, v25, 4, v15 // (op 21186) 


v_and_b32 v15, 4294967168, v1 // (op 21186) 
v_add_u32 v25, v15, v26 // (op 21186) 


v_lshl_add_u32 v15, s16, 8, v25 // (op 21186) 

v_lshrrev_b32 v25, 6, v14 // (op 21186) 

v_lshlrev_b32 v14, 2, v25 // (op 21186) 

v_mul_lo_u32 v25, v15, s28 // (op 21186) 
v_add_u32 v25, v25, v14 // (op 21186) 


s_mul_i32 s1, s28, s23 // (op 21188) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 21188) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 21188) least significant
s_mul_hi_u32 s7, s28, s22 // (op 21188) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 21188) most significant: sum
s_add_u32 s1, s1, s7 // (op 21188) most significant: sum


s_mov_b32 s56, s30 // (op 21188) reg expression
s_mov_b32 s57, s31 // (op 21188) reg expression
s_mov_b32 s58, s0 // (op 21188) reg expression
s_mov_b32 s59, 131072 // (op 21188) call()

v_and_b32 v14, 63, v1 // (op 21217) 
v_lshrrev_b32 v15, 4, v14 // (op 21217) 
v_and_b32 v26, 3, v15 // (op 21217) 

v_and_b32 v15, 15, v1 // (op 21217) 
v_lshl_add_u32 v27, v26, 4, v15 // (op 21217) 


v_lshrrev_b32 v15, 6, v1 // (op 21217) 
v_and_b32 v26, 1, v15 // (op 21217) 

v_lshl_add_u32 v15, v26, 6, v27 // (op 21217) 


v_lshl_add_u32 v26, s17, 7, v15 // (op 21217) 

v_mul_lo_u32 v15, v26, s32 // (op 21217) 

v_lshrrev_b32 v26, 6, v14 // (op 21217) 

v_lshl_add_u32 v14, v26, 2, v15 // (op 21217) 


s_mul_i32 s1, s32, s9 // (op 21219) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 21219) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 21219) least significant
s_mul_hi_u32 s7, s32, s8 // (op 21219) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 21219) most significant: sum
s_add_u32 s1, s1, s7 // (op 21219) most significant: sum


s_mov_b32 s60, s34 // (op 21219) reg expression
s_mov_b32 s61, s35 // (op 21219) reg expression
s_mov_b32 s62, s0 // (op 21219) reg expression
s_mov_b32 s63, 131072 // (op 21219) call()

v_and_b32 v15, 63, v1 // (op 21238) 
v_lshrrev_b32 v26, 4, v15 // (op 21238) 
v_and_b32 v27, 3, v26 // (op 21238) 

v_and_b32 v26, 15, v1 // (op 21238) 
v_lshl_add_u32 v28, v27, 4, v26 // (op 21238) 


v_lshrrev_b32 v26, 6, v1 // (op 21238) 
v_and_b32 v27, 1, v26 // (op 21238) 

v_lshl_add_u32 v26, v27, 6, v28 // (op 21238) 


v_lshl_add_u32 v27, s17, 7, v26 // (op 21238) 

v_mul_lo_u32 v26, v27, s32 // (op 21238) 

v_lshrrev_b32 v27, 6, v15 // (op 21238) 

v_lshl_add_u32 v15, v27, 2, v26 // (op 21238) 


s_mul_i32 s1, s32, s9 // (op 21240) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 21240) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 21240) least significant
s_mul_hi_u32 s7, s32, s8 // (op 21240) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 21240) most significant: sum
s_add_u32 s1, s1, s7 // (op 21240) most significant: sum


s_mov_b32 s60, s34 // (op 21240) reg expression
s_mov_b32 s61, s35 // (op 21240) reg expression
s_mov_b32 s62, s0 // (op 21240) reg expression
s_mov_b32 s63, 131072 // (op 21240) call()

v_and_b32 v26, -4, v1 // (op 20839) 
v_sub_u32 v27, v1, v26 // (op 20839) 

v_lshlrev_b32 v26, 5, v27 // (op 20839) 

v_lshrrev_b32 v27, 2, v1 // (op 20839) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 20839) 


v_lshlrev_b32 v26, 2, v28 // (op 20839) 

v_lshrrev_b32 v27, 3, v26 // (op 20839) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 20839) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 20538) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20538) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20538) 


v_mov_b64 v[28:29], v[20:21] // (op 20538) 
s_mov_b32 m0, s0 // (op 20538) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 20538) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 20538) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 20538) most significant half

s_add_u32 m0, m0, 4096 // (op 20538) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 20538) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 20782) 
v_sub_u32 v27, v1, v26 // (op 20782) 

v_lshlrev_b32 v26, 5, v27 // (op 20782) 

v_lshrrev_b32 v27, 2, v1 // (op 20782) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 20782) 


v_lshlrev_b32 v26, 2, v28 // (op 20782) 

v_lshrrev_b32 v27, 3, v26 // (op 20782) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 20782) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 20543) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20543) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20543) 


v_mov_b64 v[28:29], v[16:17] // (op 20543) 
s_add_u32 s1, 8192, s0 // (op 20543) 
s_mov_b32 m0, s1 // (op 20543) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20543) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 20543) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 20543) most significant half

s_add_u32 m0, m0, 4096 // (op 20543) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20543) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 20543) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 20543) most significant half

s_add_u32 m0, m0, 4096 // (op 20543) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20543) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 20543) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 20543) most significant half

s_add_u32 m0, m0, 4096 // (op 20543) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20543) Load value direct to lds






v_mov_b32 v26, v14 // (op 76) reg expression

buffer_load_dwordx4 v[28:31], v26, s[60:63], 0 offen // (op 76) Load value




s_lshl_b32 s0, s28, 6 // (op 4385) 
v_add_u32 v14, v24, s0 // (op 4385) 


buffer_load_dwordx4 v[32:35], v14, s[56:59], 0 offen // (op 4385) Load value



v_mov_b32 v14, v24 // (op 72) reg expression

buffer_load_dwordx4 v[36:39], v14, s[56:59], 0 offen // (op 72) Load value


v_and_b32 v14, -4, v1 // (op 20950) 
v_sub_u32 v24, v1, v14 // (op 20950) 

v_lshlrev_b32 v14, 5, v24 // (op 20950) 

v_lshrrev_b32 v24, 2, v1 // (op 20950) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20950) 


v_lshlrev_b32 v14, 2, v26 // (op 20950) 

v_lshrrev_b32 v24, 3, v14 // (op 20950) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20950) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 20548) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20548) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20548) 


v_mov_b64 v[40:41], v[20:21] // (op 20548) 
s_add_u32 s1, 24576, s0 // (op 20548) 
s_mov_b32 m0, s1 // (op 20548) reg expression
buffer_load_dwordx4 v40, s[52:55], 64 offen offset: 0 lds // (op 20548) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v26 // (op 20548) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 20548) most significant half

s_add_u32 m0, m0, 4096 // (op 20548) 
buffer_load_dwordx4 v40, s[52:55], 64 offen offset: 0 lds // (op 20548) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 20906) 
v_sub_u32 v24, v1, v14 // (op 20906) 

v_lshlrev_b32 v14, 5, v24 // (op 20906) 

v_lshrrev_b32 v24, 2, v1 // (op 20906) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20906) 


v_lshlrev_b32 v14, 2, v26 // (op 20906) 

v_lshrrev_b32 v24, 3, v14 // (op 20906) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20906) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 20550) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20550) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20550) 


v_mov_b64 v[40:41], v[16:17] // (op 20550) 
s_add_u32 s1, 32768, s0 // (op 20550) 
s_mov_b32 m0, s1 // (op 20550) reg expression
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20550) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v26 // (op 20550) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 20550) most significant half

s_add_u32 m0, m0, 4096 // (op 20550) 
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20550) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v26 // (op 20550) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 20550) most significant half

s_add_u32 m0, m0, 4096 // (op 20550) 
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20550) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v26 // (op 20550) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 20550) most significant half

s_add_u32 m0, m0, 4096 // (op 20550) 
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20550) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 21038) 
v_sub_u32 v24, v1, v14 // (op 21038) 

v_lshlrev_b32 v14, 5, v24 // (op 21038) 

v_lshrrev_b32 v24, 2, v1 // (op 21038) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 21038) 


v_lshlrev_b32 v14, 2, v26 // (op 21038) 

v_lshrrev_b32 v24, 3, v14 // (op 21038) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 21038) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 20552) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20552) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20552) 


v_mov_b64 v[40:41], v[20:21] // (op 20552) 
s_add_u32 s1, 49152, s0 // (op 20552) 
s_mov_b32 m0, s1 // (op 20552) reg expression
s_mov_b32 s6, 128 // (op 20552) call()
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20552) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20552) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20552) most significant half

s_add_u32 m0, m0, 4096 // (op 20552) 
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20552) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 20994) 
v_sub_u32 v24, v1, v14 // (op 20994) 

v_lshlrev_b32 v14, 5, v24 // (op 20994) 

v_lshrrev_b32 v24, 2, v1 // (op 20994) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20994) 


v_lshlrev_b32 v14, 2, v26 // (op 20994) 

v_lshrrev_b32 v24, 3, v14 // (op 20994) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20994) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 20554) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20554) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20554) 


v_mov_b64 v[40:41], v[16:17] // (op 20554) 
s_add_u32 s1, 57344, s0 // (op 20554) 
s_mov_b32 m0, s1 // (op 20554) reg expression
s_mov_b32 s6, 128 // (op 20554) call()
buffer_load_dwordx4 v40, s[48:51], s6 offen offset: 0 lds // (op 20554) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20554) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20554) most significant half

s_add_u32 m0, m0, 4096 // (op 20554) 
buffer_load_dwordx4 v40, s[48:51], s6 offen offset: 0 lds // (op 20554) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20554) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20554) most significant half

s_add_u32 m0, m0, 4096 // (op 20554) 
buffer_load_dwordx4 v40, s[48:51], s6 offen offset: 0 lds // (op 20554) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20554) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20554) most significant half

s_add_u32 m0, m0, 4096 // (op 20554) 
buffer_load_dwordx4 v40, s[48:51], s6 offen offset: 0 lds // (op 20554) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 21126) 
v_sub_u32 v24, v1, v14 // (op 21126) 

v_lshlrev_b32 v14, 5, v24 // (op 21126) 

v_lshrrev_b32 v24, 2, v1 // (op 21126) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 21126) 


v_lshlrev_b32 v14, 2, v26 // (op 21126) 

v_lshrrev_b32 v24, 3, v14 // (op 21126) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 21126) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 20556) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20556) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20556) 


v_mov_b64 v[40:41], v[20:21] // (op 20556) 
s_add_u32 s1, 73728, s0 // (op 20556) 
s_mov_b32 m0, s1 // (op 20556) reg expression
s_mov_b32 s6, 192 // (op 20556) call()
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20556) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20556) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20556) most significant half

s_add_u32 m0, m0, 4096 // (op 20556) 
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20556) Load value direct to lds






v_and_b32 v14, -4, v1 // (op 21082) 
v_sub_u32 v24, v1, v14 // (op 21082) 

v_lshlrev_b32 v14, 5, v24 // (op 21082) 

v_lshrrev_b32 v24, 2, v1 // (op 21082) 
v_lshl_add_u32 v20, v24, 7, v14 // (op 21082) 


v_lshlrev_b32 v14, 2, v20 // (op 21082) 

v_lshrrev_b32 v24, 3, v14 // (op 21082) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 21082) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 20558) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20558) 

v_lshrrev_b64 v[20:21], 3, s[10:11] // (op 20558) 


v_mov_b64 v[26:27], v[16:17] // (op 20558) 
s_add_u32 s1, 81920, s0 // (op 20558) 
s_mov_b32 m0, s1 // (op 20558) reg expression
s_mov_b32 s6, 192 // (op 20558) call()
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 20558) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v20 // (op 20558) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 20558) most significant half

s_add_u32 m0, m0, 4096 // (op 20558) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 20558) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v20 // (op 20558) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 20558) most significant half

s_add_u32 m0, m0, 4096 // (op 20558) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 20558) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v20 // (op 20558) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 20558) most significant half

s_add_u32 m0, m0, 4096 // (op 20558) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 20558) Load value direct to lds






s_waitcnt vmcnt(25)
s_barrier  // (op 21683) 



ds_read_b128 v[40:43], v4 // (op 19866) Load local data 



ds_read_b128 v[44:47], v4 offset:1024 // (op 19871) Load local data 



ds_read_b128 v[48:51], v4 offset:2048 // (op 19876) Load local data 



ds_read_b128 v[52:55], v4 offset:3072 // (op 19881) Load local data 
s_waitcnt vmcnt(21)
s_barrier  // (op 21667) 



ds_read_b128 v[56:59], v0 offset:8192 // (op 19826) Load local data 



ds_read_b128 v[60:63], v0 offset:9216 // (op 19831) Load local data 



ds_read_b128 v[64:67], v0 offset:10240 // (op 19836) Load local data 



ds_read_b128 v[68:71], v0 offset:11264 // (op 19841) Load local data 



ds_read_b128 v[72:75], v0 offset:12288 // (op 19846) Load local data 



ds_read_b128 v[76:79], v0 offset:13312 // (op 19851) Load local data 



ds_read_b128 v[80:83], v0 offset:14336 // (op 19856) Load local data 



ds_read_b128 v[84:87], v0 offset:15360 // (op 19861) Load local data 
v_and_b32 v14, -4, v1 // (op 20754) 
v_sub_u32 v24, v1, v14 // (op 20754) 

v_lshlrev_b32 v14, 5, v24 // (op 20754) 

v_lshrrev_b32 v24, 2, v1 // (op 20754) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20754) 


v_lshlrev_b32 v14, 2, v16 // (op 20754) 

v_lshrrev_b32 v24, 3, v14 // (op 20754) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20754) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20811) 
v_sub_u32 v24, v1, v14 // (op 20811) 

v_lshlrev_b32 v14, 5, v24 // (op 20811) 

v_lshrrev_b32 v24, 2, v1 // (op 20811) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20811) 


v_lshlrev_b32 v14, 2, v16 // (op 20811) 

v_lshrrev_b32 v24, 3, v14 // (op 20811) 

s_nop 0
v_readfirstlane_b32 s1, v24 // (op 20811) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20891) 
v_sub_u32 v24, v1, v14 // (op 20891) 

v_lshlrev_b32 v14, 5, v24 // (op 20891) 

v_lshrrev_b32 v24, 2, v1 // (op 20891) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20891) 


v_lshlrev_b32 v14, 2, v16 // (op 20891) 

v_lshrrev_b32 v24, 3, v14 // (op 20891) 

s_nop 0
v_readfirstlane_b32 s6, v24 // (op 20891) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20935) 
v_sub_u32 v24, v1, v14 // (op 20935) 

v_lshlrev_b32 v14, 5, v24 // (op 20935) 

v_lshrrev_b32 v24, 2, v1 // (op 20935) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20935) 


v_lshlrev_b32 v14, 2, v16 // (op 20935) 

v_lshrrev_b32 v24, 3, v14 // (op 20935) 

s_nop 0
v_readfirstlane_b32 s7, v24 // (op 20935) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20979) 
v_sub_u32 v24, v1, v14 // (op 20979) 

v_lshlrev_b32 v14, 5, v24 // (op 20979) 

v_lshrrev_b32 v24, 2, v1 // (op 20979) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20979) 


v_lshlrev_b32 v14, 2, v16 // (op 20979) 

v_lshrrev_b32 v24, 3, v14 // (op 20979) 

s_nop 0
v_readfirstlane_b32 s10, v24 // (op 20979) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 21023) 
v_sub_u32 v24, v1, v14 // (op 21023) 

v_lshlrev_b32 v14, 5, v24 // (op 21023) 

v_lshrrev_b32 v24, 2, v1 // (op 21023) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 21023) 


v_lshlrev_b32 v14, 2, v16 // (op 21023) 

v_lshrrev_b32 v24, 3, v14 // (op 21023) 

s_nop 0
v_readfirstlane_b32 s11, v24 // (op 21023) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 21067) 
v_sub_u32 v24, v1, v14 // (op 21067) 

v_lshlrev_b32 v14, 5, v24 // (op 21067) 

v_lshrrev_b32 v24, 2, v1 // (op 21067) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 21067) 


v_lshlrev_b32 v14, 2, v16 // (op 21067) 

v_lshrrev_b32 v24, 3, v14 // (op 21067) 

s_nop 0
v_readfirstlane_b32 s14, v24 // (op 21067) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 21111) 
v_sub_u32 v24, v1, v14 // (op 21111) 

v_lshlrev_b32 v14, 5, v24 // (op 21111) 

v_lshrrev_b32 v24, 2, v1 // (op 21111) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 21111) 


v_lshlrev_b32 v14, 2, v16 // (op 21111) 

v_lshrrev_b32 v24, 3, v14 // (op 21111) 

s_nop 0
v_readfirstlane_b32 s15, v24 // (op 21111) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 7269) call()
s_ashr_i64 s[38:39], s[36:37], 7 // (op 7268) 
s_mov_b32 s19, s38 // (op 7268) convert

s_and_b32 s38, s19, 4294967292 // (op 7268) 

s_cmp_lt_u32 s18, s38 // (op 7268) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopBottom_KLoop_7268 // (op 7268) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopBottom_KLoop_7268 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopTop_KLoop_7268:
s_add_u32 s18, s18, 4 // (op 7270) 
s_waitcnt vmcnt(20)
v_mov_b32 v88, v28 // (op 20417) reg expression
v_mov_b32 v89, v29 // (op 20417) reg expression
v_mov_b32 v90, v30 // (op 20417) reg expression
v_mov_b32 v91, v31 // (op 20417) reg expression
s_waitcnt lgkmcnt(8)
s_barrier  // (op 21686) 
s_waitcnt vmcnt(19)
v_mov_b32 v92, v32 // (op 20430) reg expression

v_mov_b32 v93, v33 // (op 20430) reg expression
v_mov_b32 v94, v34 // (op 20430) reg expression
v_mov_b32 v95, v35 // (op 20430) reg expression
s_lshl_b64 s[38:39], s[24:25], 6 // (op 20491) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20491) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20491) 


v_mov_b64 v[20:21], v[22:23] // (op 20491) 
v_mov_b64 v[26:27], v[20:21] // (op 20491) 
s_mov_b32 m0, s1 // (op 20491) reg expression
s_mov_b32 s19, 256 // (op 20491) call()
s_waitcnt vmcnt(18)
v_mov_b32 v96, v36 // (op 20443) reg expression
v_mov_b32 v97, v37 // (op 20443) reg expression
v_mov_b32 v98, v38 // (op 20443) reg expression
v_mov_b32 v99, v39 // (op 20443) reg expression
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 20491) Load value direct to lds
v_add_co_u32 v26, s[38:39], v26, v16 // (op 20491) least significant half
v_addc_co_u32 v27, s[38:39], 0, v27, s[38:39] // (op 20491) most significant half

s_add_u32 m0, m0, 4096 // (op 20491) 
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 20491) Load value direct to lds



v_bfe_u32 v100, v96, 0, 8 // (op 18664) 
v_bfe_u32 v101, v96, 8, 8 // (op 18664) 
v_bfe_u32 v102, v96, 16, 8 // (op 18664) 
v_bfe_u32 v103, v96, 24, 8 // (op 18664) 
v_permlane16_swap_b32 v100, v101 // (op 18664) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18664) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18664) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18664) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21670) 

s_lshl_b64 s[38:39], s[12:13], 6 // (op 20486) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20486) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20486) 


v_mov_b64 v[26:27], v[18:19] // (op 20486) 
v_mov_b64 v[104:105], v[26:27] // (op 20486) 
s_add_u32 s19, 8192, s0 // (op 20486) 
s_mov_b32 m0, s19 // (op 20486) reg expression
s_mov_b32 s38, 256 // (op 20486) call()
buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20486) Load value direct to lds
v_add_co_u32 v104, s[64:65], v104, v16 // (op 20486) least significant half
v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20486) most significant half

s_add_u32 m0, m0, 4096 // (op 20486) 
v_bfe_u32 v106, v88, 0, 8 // (op 18925) 
v_bfe_u32 v107, v88, 8, 8 // (op 18925) 
v_bfe_u32 v108, v88, 16, 8 // (op 18925) 
v_bfe_u32 v109, v88, 24, 8 // (op 18925) 
v_permlane16_swap_b32 v106, v107 // (op 18925) 
s_nop 0
v_permlane16_swap_b32 v108, v109 // (op 18925) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 18925) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 18925) 
buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20486) Load value direct to lds
v_add_co_u32 v104, s[64:65], v104, v16 // (op 20486) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[40:43], a[0:3], v100, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 78) 
v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20486) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[44:47], a[4:7], v100, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 421) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[48:51], a[8:11], v100, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 647) 
s_add_u32 m0, m0, 4096 // (op 20486) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[52:55], a[12:15], v100, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 873) 
buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20486) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[40:43], a[16:19], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1545) 
v_add_co_u32 v104, s[64:65], v104, v16 // (op 20486) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[44:47], a[20:23], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1469) 
v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20486) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[48:51], a[24:27], v101, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1393) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[52:55], a[28:31], v101, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1317) 
s_add_u32 m0, m0, 4096 // (op 20486) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[40:43], a[32:35], v102, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2547) 
buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20486) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[44:47], a[36:39], v102, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2471) 
v_mov_b32 v14, v15 // (op 20425) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[48:51], a[40:43], v102, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2395) 
v_add_u32 v24, 16, v14 // (op 20425) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[52:55], a[44:47], v102, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2319) 
v_bfe_u32 v110, v92, 0, 8 // (op 18729) 
v_bfe_u32 v111, v92, 8, 8 // (op 18729) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[40:43], a[48:51], v103, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3549) 
v_bfe_u32 v112, v92, 16, 8 // (op 18729) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[44:47], a[52:55], v103, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3473) 
v_bfe_u32 v113, v92, 24, 8 // (op 18729) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[48:51], a[56:59], v103, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3397) 
v_permlane16_swap_b32 v110, v111 // (op 18729) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[52:55], a[60:63], v103, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3321) 
v_permlane16_swap_b32 v112, v113 // (op 18729) 

s_nop 1
v_permlane32_swap_b32 v110, v112 // (op 18729) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v111, v113 // (op 18729) 
s_waitcnt vmcnt(22)
s_barrier  // (op 21705) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[40:43], a[64:67], v110, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4551) 
ds_read_b128 v[100:103], v6 offset:24576 // (op 216) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[44:47], a[68:71], v110, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4475) 
ds_read_b128 v[114:117], v6 offset:25600 // (op 432) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[48:51], a[72:75], v110, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4399) 
ds_read_b128 v[118:121], v6 offset:26624 // (op 658) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[52:55], a[76:79], v110, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4323) 
ds_read_b128 v[122:125], v6 offset:27648 // (op 884) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[40:43], a[80:83], v111, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5553) 
s_waitcnt vmcnt(18)
s_barrier  // (op 21699) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[44:47], a[84:87], v111, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5477) 
ds_read_b128 v[56:59], v3 offset:32768 // (op 214) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[48:51], a[88:91], v111, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5401) 
ds_read_b128 v[60:63], v3 offset:33792 // (op 1327) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[52:55], a[92:95], v111, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5325) 
ds_read_b128 v[64:67], v3 offset:34816 // (op 2329) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[40:43], a[96:99], v112, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6555) 
ds_read_b128 v[68:71], v3 offset:35840 // (op 3331) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[44:47], a[100:103], v112, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6479) 
ds_read_b128 v[72:75], v3 offset:36864 // (op 4333) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[48:51], a[104:107], v112, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6403) 
ds_read_b128 v[76:79], v3 offset:37888 // (op 5335) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[52:55], a[108:111], v112, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6327) 
buffer_load_dwordx4 v[28:31], v24, s[60:63], 0 offen // (op 20425) Load value


v_add_u32 v15, 16, v15 // (op 21228) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[40:43], a[112:115], v113, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7557) 
ds_read_b128 v[80:83], v3 offset:38912 // (op 6337) Load local data 
v_mov_b32 v14, v25 // (op 20438) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[44:47], a[116:119], v113, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7481) 
s_lshl_b32 s19, s28, 6 // (op 20438) 
s_add_u32 s38, 16, s19 // (op 20438) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[48:51], a[120:123], v113, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7405) 
v_add_u32 v24, v14, s38 // (op 20438) 


ds_read_b128 v[126:129], v3 offset:39936 // (op 7339) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[52:55], a[124:127], v113, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7329) 
buffer_load_dwordx4 v[32:35], v24, s[56:59], 0 offen // (op 20438) Load value





v_add_u32 v24, 16, v14 // (op 20451) 

v_bfe_u32 v104, v97, 0, 8 // (op 18679) 
v_bfe_u32 v105, v97, 8, 8 // (op 18679) 
v_bfe_u32 v106, v97, 16, 8 // (op 18679) 
v_bfe_u32 v107, v97, 24, 8 // (op 18679) 
v_permlane16_swap_b32 v104, v105 // (op 18679) 
s_nop 0
v_permlane16_swap_b32 v106, v107 // (op 18679) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v104, v106 // (op 18679) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v105, v107 // (op 18679) 
buffer_load_dwordx4 v[36:39], v24, s[56:59], 0 offen // (op 20451) Load value


v_add_u32 v25, 16, v25 // (op 21176) 
v_bfe_u32 v108, v89, 0, 8 // (op 18940) 
v_bfe_u32 v109, v89, 8, 8 // (op 18940) 
v_bfe_u32 v110, v89, 16, 8 // (op 18940) 
v_bfe_u32 v111, v89, 24, 8 // (op 18940) 
v_permlane16_swap_b32 v108, v109 // (op 18940) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 18940) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 18940) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 18940) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 21709) 

s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[100:103], a[0:3], v104, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 221) 
s_lshl_b64 s[38:39], s[24:25], 6 // (op 20513) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[114:117], a[4:7], v104, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 422) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20513) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[118:121], a[8:11], v104, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 648) 
v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20513) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[122:125], a[12:15], v104, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 874) 
v_mov_b64 v[112:113], v[20:21] // (op 20513) 
s_add_u32 s19, 24576, s7 // (op 20513) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[100:103], a[16:19], v105, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1546) 
s_mov_b32 m0, s19 // (op 20513) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[114:117], a[20:23], v105, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1470) 
s_mov_b32 s38, 320 // (op 20513) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[118:121], a[24:27], v105, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1394) 
v_bfe_u32 v130, v93, 0, 8 // (op 18744) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[122:125], a[28:31], v105, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1318) 
v_bfe_u32 v131, v93, 8, 8 // (op 18744) 
v_bfe_u32 v132, v93, 16, 8 // (op 18744) 
v_bfe_u32 v133, v93, 24, 8 // (op 18744) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[100:103], a[32:35], v106, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2548) 
v_permlane16_swap_b32 v130, v131 // (op 18744) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[114:117], a[36:39], v106, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2472) 
v_permlane16_swap_b32 v132, v133 // (op 18744) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[118:121], a[40:43], v106, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2396) 
s_nop 0
v_permlane32_swap_b32 v130, v132 // (op 18744) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[122:125], a[44:47], v106, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2320) 
v_permlane32_swap_b32 v131, v133 // (op 18744) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[100:103], a[48:51], v107, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3550) 
s_waitcnt vmcnt(19)
s_barrier  // (op 21719) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[114:117], a[52:55], v107, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3474) 
ds_read_b128 v[40:43], v8 offset:49152 // (op 244) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[118:121], a[56:59], v107, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3398) 
ds_read_b128 v[44:47], v8 offset:50176 // (op 442) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[122:125], a[60:63], v107, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3322) 
ds_read_b128 v[48:51], v8 offset:51200 // (op 668) Load local data 




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[100:103], a[64:67], v130, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4552) 
ds_read_b128 v[52:55], v8 offset:52224 // (op 894) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[114:117], a[68:71], v130, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4476) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[118:121], a[72:75], v130, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4400) 
s_waitcnt vmcnt(15)
s_barrier  // (op 21713) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[122:125], a[76:79], v130, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4324) 
ds_read_b128 v[56:59], v5 offset:57344 // (op 242) Load local data 



s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[100:103], a[80:83], v131, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5554) 
ds_read_b128 v[60:63], v5 offset:58368 // (op 1337) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[114:117], a[84:87], v131, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5478) 
ds_read_b128 v[64:67], v5 offset:59392 // (op 2339) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[118:121], a[88:91], v131, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5402) 
ds_read_b128 v[68:71], v5 offset:60416 // (op 3341) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[122:125], a[92:95], v131, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5326) 
ds_read_b128 v[72:75], v5 offset:61440 // (op 4343) Load local data 
buffer_load_dwordx4 v112, s[52:55], s38 offen offset: 0 lds // (op 20513) Load value direct to lds



s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[100:103], a[96:99], v132, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6556) 
ds_read_b128 v[76:79], v5 offset:62464 // (op 5345) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[114:117], a[100:103], v132, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6480) 
v_add_co_u32 v112, s[64:65], v112, v16 // (op 20513) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[118:121], a[104:107], v132, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6404) 
v_addc_co_u32 v113, s[64:65], 0, v113, s[64:65] // (op 20513) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[122:125], a[108:111], v132, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6328) 
s_add_u32 m0, m0, 4096 // (op 20513) 



s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[100:103], a[112:115], v133, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7558) 

ds_read_b128 v[80:83], v5 offset:63488 // (op 6347) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[114:117], a[116:119], v133, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7482) 
ds_read_b128 v[84:87], v5 offset:64512 // (op 7349) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[118:121], a[120:123], v133, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7406) 
buffer_load_dwordx4 v112, s[52:55], s38 offen offset: 0 lds // (op 20513) Load value direct to lds





s_barrier  // (op 21702) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[122:125], a[124:127], v133, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7330) 





s_lshl_b64 s[38:39], s[12:13], 6 // (op 20508) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20508) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20508) 


v_mov_b64 v[100:101], v[26:27] // (op 20508) 
s_add_u32 s19, 32768, s6 // (op 20508) 
s_mov_b32 m0, s19 // (op 20508) reg expression
s_mov_b32 s38, 320 // (op 20508) call()
buffer_load_dwordx4 v100, s[48:51], s38 offen offset: 0 lds // (op 20508) Load value direct to lds
v_add_co_u32 v100, s[64:65], v100, v16 // (op 20508) least significant half
v_addc_co_u32 v101, s[64:65], 0, v101, s[64:65] // (op 20508) most significant half

s_add_u32 m0, m0, 4096 // (op 20508) 
buffer_load_dwordx4 v100, s[48:51], s38 offen offset: 0 lds // (op 20508) Load value direct to lds
v_add_co_u32 v100, s[64:65], v100, v16 // (op 20508) least significant half
v_addc_co_u32 v101, s[64:65], 0, v101, s[64:65] // (op 20508) most significant half

s_add_u32 m0, m0, 4096 // (op 20508) 
buffer_load_dwordx4 v100, s[48:51], s38 offen offset: 0 lds // (op 20508) Load value direct to lds
v_add_co_u32 v100, s[64:65], v100, v16 // (op 20508) least significant half
v_addc_co_u32 v101, s[64:65], 0, v101, s[64:65] // (op 20508) most significant half

s_add_u32 m0, m0, 4096 // (op 20508) 
buffer_load_dwordx4 v100, s[48:51], s38 offen offset: 0 lds // (op 20508) Load value direct to lds




v_bfe_u32 v100, v98, 0, 8 // (op 18696) 
v_bfe_u32 v101, v98, 8, 8 // (op 18696) 
v_bfe_u32 v102, v98, 16, 8 // (op 18696) 
v_bfe_u32 v103, v98, 24, 8 // (op 18696) 
v_permlane16_swap_b32 v100, v101 // (op 18696) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18696) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18696) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18696) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 21722) 

s_lshl_b64 s[38:39], s[24:25], 6 // (op 20523) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20523) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20523) 


v_mov_b64 v[104:105], v[20:21] // (op 20523) 
s_add_u32 s19, 49152, s11 // (op 20523) 
s_mov_b32 m0, s19 // (op 20523) reg expression
s_mov_b32 s38, 384 // (op 20523) call()
v_bfe_u32 v106, v90, 0, 8 // (op 18958) 
v_bfe_u32 v107, v90, 8, 8 // (op 18958) 
v_bfe_u32 v108, v90, 16, 8 // (op 18958) 
v_bfe_u32 v109, v90, 24, 8 // (op 18958) 
v_permlane16_swap_b32 v106, v107 // (op 18958) 
s_nop 0
v_permlane16_swap_b32 v108, v109 // (op 18958) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 18958) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 18958) 
buffer_load_dwordx4 v104, s[52:55], s38 offen offset: 0 lds // (op 20523) Load value direct to lds
v_add_co_u32 v104, s[64:65], v104, v16 // (op 20523) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[40:43], a[0:3], v100, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 249) 
v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20523) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[44:47], a[4:7], v100, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 423) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[48:51], a[8:11], v100, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 649) 
s_add_u32 m0, m0, 4096 // (op 20523) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[52:55], a[12:15], v100, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 875) 
v_bfe_u32 v110, v94, 0, 8 // (op 18762) 
v_bfe_u32 v111, v94, 8, 8 // (op 18762) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[40:43], a[16:19], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1547) 
v_bfe_u32 v112, v94, 16, 8 // (op 18762) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[44:47], a[20:23], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1471) 
v_bfe_u32 v113, v94, 24, 8 // (op 18762) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[48:51], a[24:27], v101, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1395) 
buffer_load_dwordx4 v104, s[52:55], s38 offen offset: 0 lds // (op 20523) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[52:55], a[28:31], v101, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1319) 
v_permlane16_swap_b32 v110, v111 // (op 18762) 
v_permlane16_swap_b32 v112, v113 // (op 18762) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[40:43], a[32:35], v102, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2549) 
s_nop 0
v_permlane32_swap_b32 v110, v112 // (op 18762) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[44:47], a[36:39], v102, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2473) 
v_permlane32_swap_b32 v111, v113 // (op 18762) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[48:51], a[40:43], v102, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2397) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21716) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[52:55], a[44:47], v102, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2321) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 20518) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20518) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[40:43], a[48:51], v103, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3551) 
v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20518) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[44:47], a[52:55], v103, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3475) 
v_mov_b64 v[104:105], v[26:27] // (op 20518) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[48:51], a[56:59], v103, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3399) 
s_add_u32 s19, 57344, s10 // (op 20518) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[52:55], a[60:63], v103, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3323) 
s_mov_b32 m0, s19 // (op 20518) reg expression

s_mov_b32 s38, 384 // (op 20518) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[40:43], a[64:67], v110, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4553) 
buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20518) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[44:47], a[68:71], v110, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4477) 
v_add_co_u32 v104, s[64:65], v104, v16 // (op 20518) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[48:51], a[72:75], v110, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4401) 
v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20518) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[52:55], a[76:79], v110, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4325) 
s_add_u32 m0, m0, 4096 // (op 20518) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[40:43], a[80:83], v111, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5555) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[44:47], a[84:87], v111, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5479) 
s_waitcnt vmcnt(22)
s_barrier  // (op 21731) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[48:51], a[88:91], v111, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5403) 
v_add_u32 v14, 73728, v10 // (op 272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[52:55], a[92:95], v111, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5327) 
ds_read_b128 v[100:103], v14 // (op 272) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[40:43], a[96:99], v112, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6557) 
v_add_u32 v14, 74752, v10 // (op 452) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[44:47], a[100:103], v112, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6481) 
ds_read_b128 v[114:117], v14 // (op 452) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[48:51], a[104:107], v112, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6405) 
v_add_u32 v14, 75776, v10 // (op 678) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[52:55], a[108:111], v112, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6329) 
ds_read_b128 v[118:121], v14 // (op 678) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[40:43], a[112:115], v113, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7559) 
v_add_u32 v14, 76800, v10 // (op 904) 
ds_read_b128 v[122:125], v14 // (op 904) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[44:47], a[116:119], v113, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7483) 
s_waitcnt vmcnt(18)
s_barrier  // (op 21725) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[48:51], a[120:123], v113, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7407) 
v_add_u32 v14, 81920, v7 // (op 270) 
ds_read_b128 v[56:59], v14 // (op 270) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[52:55], a[124:127], v113, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7331) 
v_add_u32 v14, 82944, v7 // (op 1347) 


ds_read_b128 v[60:63], v14 // (op 1347) Load local data 

buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20518) Load value direct to lds



v_add_u32 v14, 83968, v7 // (op 2349) 
ds_read_b128 v[64:67], v14 // (op 2349) Load local data 

v_add_co_u32 v104, s[64:65], v104, v16 // (op 20518) least significant half



v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20518) most significant half

s_add_u32 m0, m0, 4096 // (op 20518) 
v_add_u32 v14, 84992, v7 // (op 3351) 
ds_read_b128 v[68:71], v14 // (op 3351) Load local data 

buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20518) Load value direct to lds



v_add_u32 v14, 86016, v7 // (op 4353) 
ds_read_b128 v[72:75], v14 // (op 4353) Load local data 

v_add_co_u32 v104, s[64:65], v104, v16 // (op 20518) least significant half



v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20518) most significant half

s_add_u32 m0, m0, 4096 // (op 20518) 
v_add_u32 v14, 87040, v7 // (op 5355) 
buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20518) Load value direct to lds




ds_read_b128 v[76:79], v14 // (op 5355) Load local data 




v_add_u32 v14, 88064, v7 // (op 6357) 
ds_read_b128 v[80:83], v14 // (op 6357) Load local data 




v_add_u32 v14, 89088, v7 // (op 7359) 
ds_read_b128 v[104:107], v14 // (op 7359) Load local data 

v_bfe_u32 v108, v99, 0, 8 // (op 18713) 
v_bfe_u32 v109, v99, 8, 8 // (op 18713) 
v_bfe_u32 v110, v99, 16, 8 // (op 18713) 
v_bfe_u32 v111, v99, 24, 8 // (op 18713) 
v_permlane16_swap_b32 v108, v109 // (op 18713) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 18713) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 18713) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 18713) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 21734) 


s_lshl_b64 s[38:39], s[24:25], 6 // (op 20533) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20533) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20533) 


v_mov_b64 v[112:113], v[20:21] // (op 20533) 
s_add_u32 s19, 73728, s15 // (op 20533) 
s_mov_b32 m0, s19 // (op 20533) reg expression
s_mov_b32 s38, 448 // (op 20533) call()
v_bfe_u32 v96, v91, 0, 8 // (op 18976) 
v_bfe_u32 v97, v91, 8, 8 // (op 18976) 
v_bfe_u32 v98, v91, 16, 8 // (op 18976) 
v_bfe_u32 v99, v91, 24, 8 // (op 18976) 
v_permlane16_swap_b32 v96, v97 // (op 18976) 
s_nop 0
v_permlane16_swap_b32 v98, v99 // (op 18976) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v96, v98 // (op 18976) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v97, v99 // (op 18976) 
buffer_load_dwordx4 v112, s[52:55], s38 offen offset: 0 lds // (op 20533) Load value direct to lds

v_add_co_u32 v112, s[64:65], v112, v16 // (op 20533) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[100:103], a[0:3], v108, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 277) 
v_addc_co_u32 v113, s[64:65], 0, v113, s[64:65] // (op 20533) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[114:117], a[4:7], v108, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 424) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[118:121], a[8:11], v108, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 650) 
s_add_u32 m0, m0, 4096 // (op 20533) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[122:125], a[12:15], v108, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 876) 
v_bfe_u32 v88, v95, 0, 8 // (op 18780) 
v_bfe_u32 v89, v95, 8, 8 // (op 18780) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[100:103], a[16:19], v109, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1548) 
v_bfe_u32 v90, v95, 16, 8 // (op 18780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[114:117], a[20:23], v109, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1472) 
v_bfe_u32 v91, v95, 24, 8 // (op 18780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[118:121], a[24:27], v109, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1396) 
buffer_load_dwordx4 v112, s[52:55], s38 offen offset: 0 lds // (op 20533) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[122:125], a[28:31], v109, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1320) 
v_permlane16_swap_b32 v88, v89 // (op 18780) 
v_permlane16_swap_b32 v90, v91 // (op 18780) 
v_add_co_u32 v22, vcc, 256, v22 // (op 20703) least significant half
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[100:103], a[32:35], v110, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2550) 
v_permlane32_swap_b32 v88, v90 // (op 18780) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[114:117], a[36:39], v110, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2474) 
v_permlane32_swap_b32 v89, v91 // (op 18780) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[118:121], a[40:43], v110, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2398) 
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 20703) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[122:125], a[44:47], v110, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2322) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21728) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[100:103], a[48:51], v111, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3552) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 20528) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[114:117], a[52:55], v111, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3476) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20528) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[118:121], a[56:59], v111, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3400) 
v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20528) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[122:125], a[60:63], v111, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3324) 
v_mov_b64 v[20:21], v[26:27] // (op 20528) 

s_add_u32 s19, 81920, s14 // (op 20528) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[100:103], a[64:67], v88, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4554) 
s_mov_b32 m0, s19 // (op 20528) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[114:117], a[68:71], v88, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4478) 
s_mov_b32 s38, 448 // (op 20528) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[118:121], a[72:75], v88, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4402) 
buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 20528) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[122:125], a[76:79], v88, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4326) 
v_add_co_u32 v20, s[64:65], v20, v16 // (op 20528) least significant half
v_addc_co_u32 v21, s[64:65], 0, v21, s[64:65] // (op 20528) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[100:103], a[80:83], v89, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5556) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[114:117], a[84:87], v89, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5480) 
s_add_u32 m0, m0, 4096 // (op 20528) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[118:121], a[88:91], v89, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5404) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[122:125], a[92:95], v89, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5328) 
s_waitcnt vmcnt(22)
s_barrier  // (op 21692) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[100:103], a[96:99], v90, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6558) 
ds_read_b128 v[40:43], v4 // (op 67) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[114:117], a[100:103], v90, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6482) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 418) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[118:121], a[104:107], v90, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6406) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 644) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[122:125], a[108:111], v90, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6330) 
ds_read_b128 v[52:55], v4 offset:3072 // (op 870) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[100:103], a[112:115], v91, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7560) 
s_waitcnt vmcnt(18)
s_barrier  // (op 21676) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[114:117], a[116:119], v91, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7484) 
ds_read_b128 v[56:59], v0 offset:8192 // (op 62) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:107], v[118:121], a[120:123], v91, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7408) 
ds_read_b128 v[60:63], v0 offset:9216 // (op 1313) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:107], v[122:125], a[124:127], v91, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7332) 
ds_read_b128 v[64:67], v0 offset:10240 // (op 2315) Load local data 







ds_read_b128 v[68:71], v0 offset:11264 // (op 3317) Load local data 
buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 20528) Load value direct to lds



ds_read_b128 v[72:75], v0 offset:12288 // (op 4319) Load local data 
v_add_co_u32 v20, s[64:65], v20, v16 // (op 20528) least significant half



ds_read_b128 v[76:79], v0 offset:13312 // (op 5321) Load local data 
v_addc_co_u32 v21, s[64:65], 0, v21, s[64:65] // (op 20528) most significant half




s_add_u32 m0, m0, 4096 // (op 20528) 
ds_read_b128 v[80:83], v0 offset:14336 // (op 6323) Load local data 
buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 20528) Load value direct to lds



v_add_co_u32 v20, s[64:65], v20, v16 // (op 20528) least significant half
v_addc_co_u32 v21, s[64:65], 0, v21, s[64:65] // (op 20528) most significant half

s_add_u32 m0, m0, 4096 // (op 20528) 
buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 20528) Load value direct to lds




ds_read_b128 v[84:87], v0 offset:15360 // (op 7325) Load local data 

v_add_co_u32 v18, vcc, 256, v18 // (op 20639) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 20639) most significant half
s_ashr_i64 s[38:39], s[36:37], 7 // (op 7268) 
s_mov_b32 s19, s38 // (op 7268) convert

s_and_b32 s38, s19, 4294967292 // (op 7268) 

s_cmp_lt_u32 s18, s38 // (op 7268) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopTop_KLoop_7268 // (op 7268) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopTop_KLoop_7268 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopBottom_KLoop_7268:















s_barrier  // (op 21843) 
v_lshrrev_b32 v14, 2, v1 // (op 20571) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 20571) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 20571) 

s_lshr_b32 s6, s0, 2 // (op 20571) 

v_and_b32 v14, -4, v1 // (op 20571) 
v_sub_u32 v16, v1, v14 // (op 20571) 

v_lshlrev_b32 v14, 5, v16 // (op 20571) 

v_lshl_add_u32 v16, s6, 9, v14 // (op 20571) 


v_mov_b32 v27, 0 // (op 20571) convert
v_mov_b32 v26, v16 // (op 20571) convert

v_mul_lo_u32 v17, v15, s13 // (op 20571) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 20571) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 20571) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 20571) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v26 // (op 20571) least significant half
v_addc_co_u32 v17, s[0:1], v17, v27, s[0:1] // (op 20571) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 20571) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 20571) 

s_mul_i32 s1, s12, s23 // (op 20573) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 20573) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 20573) least significant
s_mul_hi_u32 s7, s12, s22 // (op 20573) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20573) most significant: sum
s_add_u32 s1, s1, s7 // (op 20573) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20573) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20573) 

s_mov_b32 s48, s20 // (op 20573) reg expression
s_mov_b32 s49, s21 // (op 20573) reg expression
s_mov_b32 s50, s0 // (op 20573) reg expression
s_mov_b32 s51, 131072 // (op 20573) call()

v_and_b32 v14, -4, v1 // (op 20669) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 20669) 
v_sub_u32 v15, v1, v14 // (op 20669) 

s_lshr_b32 s6, s0, 2 // (op 20669) 

v_lshlrev_b32 v14, 5, v15 // (op 20669) 

v_lshl_add_u32 v15, s6, 9, v14 // (op 20669) 


v_mov_b32 v19, 0 // (op 20669) convert
v_mov_b32 v18, v15 // (op 20669) convert

v_lshrrev_b32 v14, 2, v1 // (op 20669) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 20669) 

v_mul_lo_u32 v21, v15, s25 // (op 20669) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 20669) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 20669) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 20669) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 20669) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 20669) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20669) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 20669) 

s_mul_i32 s1, s24, s9 // (op 20671) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 20671) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 20671) least significant
s_mul_hi_u32 s7, s24, s8 // (op 20671) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20671) most significant: sum
s_add_u32 s1, s1, s7 // (op 20671) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20671) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20671) 

s_mov_b32 s52, s26 // (op 20671) reg expression
s_mov_b32 s53, s27 // (op 20671) reg expression
s_mov_b32 s54, s0 // (op 20671) reg expression
s_mov_b32 s55, 131072 // (op 20671) call()

v_and_b32 v14, -4, v1 // (op 20767) 
v_sub_u32 v15, v1, v14 // (op 20767) 

v_lshlrev_b32 v14, 5, v15 // (op 20767) 

v_lshrrev_b32 v15, 2, v1 // (op 20767) 
v_lshl_add_u32 v20, v15, 7, v14 // (op 20767) 


v_lshlrev_b32 v14, 2, v20 // (op 20767) 

v_lshrrev_b32 v15, 3, v14 // (op 20767) 

s_nop 0
v_readfirstlane_b32 s0, v15 // (op 20767) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20824) 
v_sub_u32 v15, v1, v14 // (op 20824) 

v_lshlrev_b32 v14, 5, v15 // (op 20824) 

v_lshrrev_b32 v15, 2, v1 // (op 20824) 
v_lshl_add_u32 v20, v15, 7, v14 // (op 20824) 


v_lshlrev_b32 v14, 2, v20 // (op 20824) 

v_lshrrev_b32 v15, 3, v14 // (op 20824) 

s_nop 0
v_readfirstlane_b32 s1, v15 // (op 20824) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, 63, v1 // (op 21265) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 21265) 
v_lshrrev_b32 v15, 4, v14 // (op 21265) 
s_lshr_b32 s10, s6, 2 // (op 21265) 

v_and_b32 v20, 3, v15 // (op 21265) 

v_and_b32 v15, 15, v1 // (op 21265) 
v_lshl_add_u32 v21, v20, 4, v15 // (op 21265) 


v_and_b32 v15, 4294967168, v1 // (op 21265) 
v_add_u32 v20, v15, v21 // (op 21265) 


v_lshl_add_u32 v15, s16, 8, v20 // (op 21265) 

v_lshrrev_b32 v20, 6, v14 // (op 21265) 

v_lshlrev_b32 v14, 2, v20 // (op 21265) 

v_lshl_add_u32 v20, s10, 4, v14 // (op 21265) 


v_mul_lo_u32 v14, v15, s28 // (op 21265) 
v_add_u32 v14, v14, v20 // (op 21265) 


s_mul_i32 s7, s28, s23 // (op 21267) most significant: low of low * high
s_mul_i32 s10, s29, s22 // (op 21267) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 21267) least significant
s_mul_hi_u32 s11, s28, s22 // (op 21267) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 21267) most significant: sum
s_add_u32 s7, s7, s11 // (op 21267) most significant: sum


s_mov_b32 s56, s30 // (op 21267) reg expression
s_mov_b32 s57, s31 // (op 21267) reg expression
s_mov_b32 s58, s6 // (op 21267) reg expression
s_mov_b32 s59, 131072 // (op 21267) call()

v_and_b32 v15, 63, v1 // (op 21292) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 21292) 
v_lshrrev_b32 v20, 6, v15 // (op 21292) 
s_lshr_b32 s10, s6, 2 // (op 21292) 

v_lshlrev_b32 v21, 2, v20 // (op 21292) 

v_lshl_add_u32 v20, s10, 4, v21 // (op 21292) 


v_lshrrev_b32 v21, 4, v15 // (op 21292) 

v_and_b32 v15, 3, v21 // (op 21292) 

v_and_b32 v21, 15, v1 // (op 21292) 
v_lshl_add_u32 v22, v15, 4, v21 // (op 21292) 


v_lshrrev_b32 v15, 6, v1 // (op 21292) 
v_and_b32 v21, 1, v15 // (op 21292) 

v_lshl_add_u32 v15, v21, 6, v22 // (op 21292) 


v_lshl_add_u32 v21, s17, 7, v15 // (op 21292) 

v_mul_lo_u32 v15, v21, s32 // (op 21292) 

v_add_u32 v21, v20, v15 // (op 21292) 


s_mul_i32 s7, s32, s9 // (op 21294) most significant: low of low * high
s_mul_i32 s10, s33, s8 // (op 21294) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 21294) least significant
s_mul_hi_u32 s11, s32, s8 // (op 21294) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 21294) most significant: sum
s_add_u32 s7, s7, s11 // (op 21294) most significant: sum


s_mov_b32 s60, s34 // (op 21294) reg expression
s_mov_b32 s61, s35 // (op 21294) reg expression
s_mov_b32 s62, s6 // (op 21294) reg expression
s_mov_b32 s63, 131072 // (op 21294) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 7301) 
s_and_b32 s10, s6, 4294967292 // (op 7301) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 7300) 
s_mov_b32 s11, s6 // (op 7300) convert

s_cmp_lt_u32 s10, s11 // (op 7300) 
















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopBottom_KLoopTail_7300 // (op 7300) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopBottom_KLoopTail_7300 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopTop_KLoopTail_7300:
s_barrier  // (op 21673) 
s_add_u32 s10, s10, 1 // (op 7302) 

s_lshl_b64 s[6:7], s[12:13], 6 // (op 20496) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 20496) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 20496) 


v_mov_b64 v[24:25], v[16:17] // (op 20496) 
v_mov_b64 v[26:27], v[24:25] // (op 20496) 
s_add_u32 s11, 8192, s0 // (op 20496) 
s_mov_b32 m0, s11 // (op 20496) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 20496) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 20496) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 20496) most significant half

s_add_u32 m0, m0, 4096 // (op 20496) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 20496) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 20496) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 20496) most significant half

s_add_u32 m0, m0, 4096 // (op 20496) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 20496) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 20496) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 20496) most significant half

s_add_u32 m0, m0, 4096 // (op 20496) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 20496) Load value direct to lds




v_add_co_u32 v16, s[6:7], 64, v16 // (op 20561) least significant half
v_addc_co_u32 v17, s[6:7], 0, v17, s[6:7] // (op 20561) most significant half

s_barrier  // (op 21689) 

s_lshl_b64 s[6:7], s[24:25], 6 // (op 20502) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 20502) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 20502) 


v_mov_b64 v[24:25], v[18:19] // (op 20502) 
v_mov_b64 v[26:27], v[24:25] // (op 20502) 
s_mov_b32 m0, s1 // (op 20502) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 20502) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 20502) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 20502) most significant half

s_add_u32 m0, m0, 4096 // (op 20502) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 20502) Load value direct to lds



v_add_co_u32 v18, s[6:7], 64, v18 // (op 20661) least significant half
v_addc_co_u32 v19, s[6:7], 0, v19, s[6:7] // (op 20661) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 21679) 



ds_read_b128 v[22:25], v0 offset:8192 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 21695) 



ds_read_b128 v[26:29], v4 // (op 166) Load local data 


v_mov_b32 v15, v14 // (op 168) 
v_mov_b32 v20, v15 // (op 168) reg expression

buffer_load_dwordx4 v[30:33], v20, s[56:59], 0 offen // (op 168) Load value



v_mov_b32 v20, v21 // (op 170) 
v_mov_b32 v34, v20 // (op 170) reg expression

buffer_load_dwordx4 v[36:39], v34, s[60:63], 0 offen // (op 170) Load value


v_add_u32 v21, 4, v21 // (op 21282) 
s_waitcnt vmcnt(1)
v_bfe_u32 v40, v30, 0, 8 // (op 19121) 



v_bfe_u32 v41, v30, 8, 8 // (op 19121) 
v_bfe_u32 v42, v30, 16, 8 // (op 19121) 
v_bfe_u32 v43, v30, 24, 8 // (op 19121) 
v_permlane16_swap_b32 v40, v41 // (op 19121) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 401) Load local data 
v_permlane16_swap_b32 v42, v43 // (op 19121) 



s_nop 1
v_permlane32_swap_b32 v40, v42 // (op 19121) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v41, v43 // (op 19121) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 627) Load local data 




ds_read_b128 v[30:33], v4 offset:3072 // (op 853) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v52, v36, 0, 8 // (op 19181) 



v_bfe_u32 v53, v36, 8, 8 // (op 19181) 
v_bfe_u32 v54, v36, 16, 8 // (op 19181) 
v_bfe_u32 v55, v36, 24, 8 // (op 19181) 
v_permlane16_swap_b32 v52, v53 // (op 19181) 
s_nop 0
v_permlane16_swap_b32 v54, v55 // (op 19181) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 19181) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 19181) 
ds_read_b128 v[56:59], v0 offset:9216 // (op 1213) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 171) 
ds_read_b128 v[34:37], v0 offset:10240 // (op 2215) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[44:47], a[4:7], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 404) 
ds_read_b128 v[60:63], v0 offset:11264 // (op 3217) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[48:51], a[8:11], v40, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 630) 
ds_read_b128 v[64:67], v0 offset:12288 // (op 4219) Load local data 


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:25], v[30:33], a[12:15], v40, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 856) 
s_lshl_b32 s11, s28, 6 // (op 4228) 

v_add_u32 v20, v15, s11 // (op 4228) 


buffer_load_dwordx4 v[22:25], v20, s[56:59], 0 offen // (op 4228) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[26:29], a[16:19], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1217) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v41, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1271) 
v_add_u32 v14, 4, v14 // (op 21255) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[30:33], a[28:31], v41, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1298) 
s_waitcnt vmcnt(0)
v_bfe_u32 v68, v22, 0, 8 // (op 19135) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[44:47], a[20:23], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1244) 
v_bfe_u32 v69, v22, 8, 8 // (op 19135) 

v_bfe_u32 v70, v22, 16, 8 // (op 19135) 
v_bfe_u32 v71, v22, 24, 8 // (op 19135) 
v_permlane16_swap_b32 v68, v69 // (op 19135) 
s_nop 0
v_permlane16_swap_b32 v70, v71 // (op 19135) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[34:37], v[26:29], a[32:35], v42, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2219) 
ds_read_b128 v[56:59], v0 offset:13312 // (op 5221) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[34:37], v[48:51], a[40:43], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2273) 
v_permlane32_swap_b32 v68, v70 // (op 19135) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[34:37], v[30:33], a[44:47], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2300) 
v_permlane32_swap_b32 v69, v71 // (op 19135) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[34:37], v[44:47], a[36:39], v42, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2246) 
ds_read_b128 v[22:25], v0 offset:14336 // (op 6223) Load local data 




ds_read_b128 v[34:37], v0 offset:15360 // (op 7225) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:63], v[26:29], a[48:51], v43, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3221) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[48:51], a[56:59], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3275) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[60:63], v[30:33], a[60:63], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3302) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:63], v[44:47], a[52:55], v43, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3248) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[64:67], v[44:47], a[68:71], v68, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4250) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[64:67], v[26:29], a[64:67], v68, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4223) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[64:67], v[48:51], a[72:75], v68, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4277) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[64:67], v[30:33], a[76:79], v68, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4304) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[56:59], v[44:47], a[84:87], v69, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5252) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[56:59], v[26:29], a[80:83], v69, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5225) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[56:59], v[48:51], a[88:91], v69, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5279) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[56:59], v[30:33], a[92:95], v69, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5306) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[22:25], v[44:47], a[100:103], v70, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6254) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[22:25], v[26:29], a[96:99], v70, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6227) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[22:25], v[48:51], a[104:107], v70, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6281) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[22:25], v[30:33], a[108:111], v70, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6308) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[34:37], v[44:47], a[116:119], v71, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7256) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[34:37], v[26:29], a[112:115], v71, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7229) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[34:37], v[48:51], a[120:123], v71, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7283) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:37], v[30:33], a[124:127], v71, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7310) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 7300) 
s_mov_b32 s11, s6 // (op 7300) convert

s_cmp_lt_u32 s10, s11 // (op 7300) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopTop_KLoopTail_7300 // (op 7300) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopTop_KLoopTail_7300 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ForLoopBottom_KLoopTail_7300:





















s_barrier  // (op 21849) 









v_cmp_eq_f32 vcc, 0.00000, v9 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ConditionalFalse_0_beta_116
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v11, v4 // (op 42) 
v_mul_f32 v15, v11, v5 // (op 42) 
v_mul_f32 v16, v11, v6 // (op 42) 
v_mul_f32 v17, v11, v7 // (op 42) 

v_lshrrev_b32 v4, 16, v14 // (op 48) 
v_lshrrev_b32 v5, 16, v15 // (op 48) 
v_lshrrev_b32 v6, 16, v16 // (op 48) 
v_lshrrev_b32 v7, 16, v17 // (op 48) 


v_pack_b32_f16 v14, v4, v5 // (op 50) 
v_pack_b32_f16 v15, v6, v7 // (op 50) 

v_mov_b32 v0, v13 // (op 50) reg expression

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 50) Store value



v_accvgpr_read v4, a4 // (op 9202) 
v_accvgpr_read v5, a5 // (op 9202) 
v_accvgpr_read v6, a6 // (op 9202) 
v_accvgpr_read v7, a7 // (op 9202) 
v_mul_f32 v14, v11, v4 // (op 9202) 
v_mul_f32 v15, v11, v5 // (op 9202) 
v_mul_f32 v16, v11, v6 // (op 9202) 
v_mul_f32 v17, v11, v7 // (op 9202) 

v_lshrrev_b32 v4, 16, v14 // (op 9204) 
v_lshrrev_b32 v5, 16, v15 // (op 9204) 
v_lshrrev_b32 v6, 16, v16 // (op 9204) 
v_lshrrev_b32 v7, 16, v17 // (op 9204) 


v_pack_b32_f16 v14, v4, v5 // (op 9205) 
v_pack_b32_f16 v15, v6, v7 // (op 9205) 

s_lshl_b32 s0, s4, 5 // (op 9205) 
v_add_u32 v0, v13, s0 // (op 9205) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9205) Store value



v_accvgpr_read v4, a8 // (op 9216) 
v_accvgpr_read v5, a9 // (op 9216) 
v_accvgpr_read v6, a10 // (op 9216) 
v_accvgpr_read v7, a11 // (op 9216) 
v_mul_f32 v14, v11, v4 // (op 9216) 
v_mul_f32 v15, v11, v5 // (op 9216) 
v_mul_f32 v16, v11, v6 // (op 9216) 
v_mul_f32 v17, v11, v7 // (op 9216) 

v_lshrrev_b32 v4, 16, v14 // (op 9218) 
v_lshrrev_b32 v5, 16, v15 // (op 9218) 
v_lshrrev_b32 v6, 16, v16 // (op 9218) 
v_lshrrev_b32 v7, 16, v17 // (op 9218) 


v_pack_b32_f16 v14, v4, v5 // (op 9219) 
v_pack_b32_f16 v15, v6, v7 // (op 9219) 

s_lshl_b32 s0, s4, 6 // (op 9219) 
v_add_u32 v0, v13, s0 // (op 9219) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9219) Store value



v_accvgpr_read v4, a12 // (op 9230) 
v_accvgpr_read v5, a13 // (op 9230) 
v_accvgpr_read v6, a14 // (op 9230) 
v_accvgpr_read v7, a15 // (op 9230) 
v_mul_f32 v14, v11, v4 // (op 9230) 
v_mul_f32 v15, v11, v5 // (op 9230) 
v_mul_f32 v16, v11, v6 // (op 9230) 
v_mul_f32 v17, v11, v7 // (op 9230) 

v_lshrrev_b32 v4, 16, v14 // (op 9232) 
v_lshrrev_b32 v5, 16, v15 // (op 9232) 
v_lshrrev_b32 v6, 16, v16 // (op 9232) 
v_lshrrev_b32 v7, 16, v17 // (op 9232) 


v_pack_b32_f16 v14, v4, v5 // (op 9233) 
v_pack_b32_f16 v15, v6, v7 // (op 9233) 

s_lshl_b32 s0, s4, 5 // (op 9233) 
s_mul_i32 s1, 3, s0 // (op 9233) 

v_add_u32 v0, v13, s1 // (op 9233) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9233) Store value



v_accvgpr_read v4, a16 // (op 9266) 
v_accvgpr_read v5, a17 // (op 9266) 
v_accvgpr_read v6, a18 // (op 9266) 
v_accvgpr_read v7, a19 // (op 9266) 
v_mul_f32 v14, v11, v4 // (op 9266) 
v_mul_f32 v15, v11, v5 // (op 9266) 
v_mul_f32 v16, v11, v6 // (op 9266) 
v_mul_f32 v17, v11, v7 // (op 9266) 

v_lshrrev_b32 v4, 16, v14 // (op 9268) 
v_lshrrev_b32 v5, 16, v15 // (op 9268) 
v_lshrrev_b32 v6, 16, v16 // (op 9268) 
v_lshrrev_b32 v7, 16, v17 // (op 9268) 


v_pack_b32_f16 v14, v4, v5 // (op 9270) 
v_pack_b32_f16 v15, v6, v7 // (op 9270) 

v_add_u32 v0, 32, v13 // (op 9270) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9270) Store value



v_accvgpr_read v4, a20 // (op 9280) 
v_accvgpr_read v5, a21 // (op 9280) 
v_accvgpr_read v6, a22 // (op 9280) 
v_accvgpr_read v7, a23 // (op 9280) 
v_mul_f32 v14, v11, v4 // (op 9280) 
v_mul_f32 v15, v11, v5 // (op 9280) 
v_mul_f32 v16, v11, v6 // (op 9280) 
v_mul_f32 v17, v11, v7 // (op 9280) 

v_lshrrev_b32 v4, 16, v14 // (op 9282) 
v_lshrrev_b32 v5, 16, v15 // (op 9282) 
v_lshrrev_b32 v6, 16, v16 // (op 9282) 
v_lshrrev_b32 v7, 16, v17 // (op 9282) 


v_pack_b32_f16 v14, v4, v5 // (op 9272) 
v_pack_b32_f16 v15, v6, v7 // (op 9272) 

s_lshl_b32 s0, s4, 5 // (op 9272) 
s_add_u32 s0, s0, 32 // (op 9272) 
v_add_u32 v0, v13, s0 // (op 9272) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9272) Store value



v_accvgpr_read v4, a24 // (op 9286) 
v_accvgpr_read v5, a25 // (op 9286) 
v_accvgpr_read v6, a26 // (op 9286) 
v_accvgpr_read v7, a27 // (op 9286) 
v_mul_f32 v14, v11, v4 // (op 9286) 
v_mul_f32 v15, v11, v5 // (op 9286) 
v_mul_f32 v16, v11, v6 // (op 9286) 
v_mul_f32 v17, v11, v7 // (op 9286) 

v_lshrrev_b32 v4, 16, v14 // (op 9288) 
v_lshrrev_b32 v5, 16, v15 // (op 9288) 
v_lshrrev_b32 v6, 16, v16 // (op 9288) 
v_lshrrev_b32 v7, 16, v17 // (op 9288) 


v_pack_b32_f16 v14, v4, v5 // (op 9274) 
v_pack_b32_f16 v15, v6, v7 // (op 9274) 

s_lshl_b32 s0, s4, 6 // (op 9274) 
s_add_u32 s0, s0, 32 // (op 9274) 
v_add_u32 v0, v13, s0 // (op 9274) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9274) Store value



v_accvgpr_read v4, a28 // (op 9292) 
v_accvgpr_read v5, a29 // (op 9292) 
v_accvgpr_read v6, a30 // (op 9292) 
v_accvgpr_read v7, a31 // (op 9292) 
v_mul_f32 v14, v11, v4 // (op 9292) 
v_mul_f32 v15, v11, v5 // (op 9292) 
v_mul_f32 v16, v11, v6 // (op 9292) 
v_mul_f32 v17, v11, v7 // (op 9292) 

v_lshrrev_b32 v4, 16, v14 // (op 9294) 
v_lshrrev_b32 v5, 16, v15 // (op 9294) 
v_lshrrev_b32 v6, 16, v16 // (op 9294) 
v_lshrrev_b32 v7, 16, v17 // (op 9294) 


v_pack_b32_f16 v14, v4, v5 // (op 9276) 
v_pack_b32_f16 v15, v6, v7 // (op 9276) 

s_lshl_b32 s0, s4, 5 // (op 9276) 
s_mul_i32 s1, 3, s0 // (op 9276) 
s_add_u32 s1, s1, 32 // (op 9276) 

v_add_u32 v0, v13, s1 // (op 9276) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9276) Store value



v_accvgpr_read v4, a32 // (op 9342) 
v_accvgpr_read v5, a33 // (op 9342) 
v_accvgpr_read v6, a34 // (op 9342) 
v_accvgpr_read v7, a35 // (op 9342) 
v_mul_f32 v14, v11, v4 // (op 9342) 
v_mul_f32 v15, v11, v5 // (op 9342) 
v_mul_f32 v16, v11, v6 // (op 9342) 
v_mul_f32 v17, v11, v7 // (op 9342) 

v_lshrrev_b32 v4, 16, v14 // (op 9344) 
v_lshrrev_b32 v5, 16, v15 // (op 9344) 
v_lshrrev_b32 v6, 16, v16 // (op 9344) 
v_lshrrev_b32 v7, 16, v17 // (op 9344) 


v_pack_b32_f16 v14, v4, v5 // (op 9346) 
v_pack_b32_f16 v15, v6, v7 // (op 9346) 

v_add_u32 v0, 64, v13 // (op 9346) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9346) Store value



v_accvgpr_read v4, a36 // (op 9356) 
v_accvgpr_read v5, a37 // (op 9356) 
v_accvgpr_read v6, a38 // (op 9356) 
v_accvgpr_read v7, a39 // (op 9356) 
v_mul_f32 v14, v11, v4 // (op 9356) 
v_mul_f32 v15, v11, v5 // (op 9356) 
v_mul_f32 v16, v11, v6 // (op 9356) 
v_mul_f32 v17, v11, v7 // (op 9356) 

v_lshrrev_b32 v4, 16, v14 // (op 9358) 
v_lshrrev_b32 v5, 16, v15 // (op 9358) 
v_lshrrev_b32 v6, 16, v16 // (op 9358) 
v_lshrrev_b32 v7, 16, v17 // (op 9358) 


v_pack_b32_f16 v14, v4, v5 // (op 9348) 
v_pack_b32_f16 v15, v6, v7 // (op 9348) 

s_lshl_b32 s0, s4, 5 // (op 9348) 
s_add_u32 s0, s0, 64 // (op 9348) 
v_add_u32 v0, v13, s0 // (op 9348) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9348) Store value



v_accvgpr_read v4, a40 // (op 9362) 
v_accvgpr_read v5, a41 // (op 9362) 
v_accvgpr_read v6, a42 // (op 9362) 
v_accvgpr_read v7, a43 // (op 9362) 
v_mul_f32 v14, v11, v4 // (op 9362) 
v_mul_f32 v15, v11, v5 // (op 9362) 
v_mul_f32 v16, v11, v6 // (op 9362) 
v_mul_f32 v17, v11, v7 // (op 9362) 

v_lshrrev_b32 v4, 16, v14 // (op 9364) 
v_lshrrev_b32 v5, 16, v15 // (op 9364) 
v_lshrrev_b32 v6, 16, v16 // (op 9364) 
v_lshrrev_b32 v7, 16, v17 // (op 9364) 


v_pack_b32_f16 v14, v4, v5 // (op 9350) 
v_pack_b32_f16 v15, v6, v7 // (op 9350) 

s_lshl_b32 s0, s4, 6 // (op 9350) 
s_add_u32 s0, s0, 64 // (op 9350) 
v_add_u32 v0, v13, s0 // (op 9350) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9350) Store value



v_accvgpr_read v4, a44 // (op 9368) 
v_accvgpr_read v5, a45 // (op 9368) 
v_accvgpr_read v6, a46 // (op 9368) 
v_accvgpr_read v7, a47 // (op 9368) 
v_mul_f32 v14, v11, v4 // (op 9368) 
v_mul_f32 v15, v11, v5 // (op 9368) 
v_mul_f32 v16, v11, v6 // (op 9368) 
v_mul_f32 v17, v11, v7 // (op 9368) 

v_lshrrev_b32 v4, 16, v14 // (op 9370) 
v_lshrrev_b32 v5, 16, v15 // (op 9370) 
v_lshrrev_b32 v6, 16, v16 // (op 9370) 
v_lshrrev_b32 v7, 16, v17 // (op 9370) 


v_pack_b32_f16 v14, v4, v5 // (op 9352) 
v_pack_b32_f16 v15, v6, v7 // (op 9352) 

s_lshl_b32 s0, s4, 5 // (op 9352) 
s_mul_i32 s1, 3, s0 // (op 9352) 
s_add_u32 s1, s1, 64 // (op 9352) 

v_add_u32 v0, v13, s1 // (op 9352) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9352) Store value



v_accvgpr_read v4, a48 // (op 9418) 
v_accvgpr_read v5, a49 // (op 9418) 
v_accvgpr_read v6, a50 // (op 9418) 
v_accvgpr_read v7, a51 // (op 9418) 
v_mul_f32 v14, v11, v4 // (op 9418) 
v_mul_f32 v15, v11, v5 // (op 9418) 
v_mul_f32 v16, v11, v6 // (op 9418) 
v_mul_f32 v17, v11, v7 // (op 9418) 

v_lshrrev_b32 v4, 16, v14 // (op 9420) 
v_lshrrev_b32 v5, 16, v15 // (op 9420) 
v_lshrrev_b32 v6, 16, v16 // (op 9420) 
v_lshrrev_b32 v7, 16, v17 // (op 9420) 


v_pack_b32_f16 v14, v4, v5 // (op 9422) 
v_pack_b32_f16 v15, v6, v7 // (op 9422) 

v_add_u32 v0, 96, v13 // (op 9422) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9422) Store value



v_accvgpr_read v4, a52 // (op 9432) 
v_accvgpr_read v5, a53 // (op 9432) 
v_accvgpr_read v6, a54 // (op 9432) 
v_accvgpr_read v7, a55 // (op 9432) 
v_mul_f32 v14, v11, v4 // (op 9432) 
v_mul_f32 v15, v11, v5 // (op 9432) 
v_mul_f32 v16, v11, v6 // (op 9432) 
v_mul_f32 v17, v11, v7 // (op 9432) 

v_lshrrev_b32 v4, 16, v14 // (op 9434) 
v_lshrrev_b32 v5, 16, v15 // (op 9434) 
v_lshrrev_b32 v6, 16, v16 // (op 9434) 
v_lshrrev_b32 v7, 16, v17 // (op 9434) 


v_pack_b32_f16 v14, v4, v5 // (op 9424) 
v_pack_b32_f16 v15, v6, v7 // (op 9424) 

s_lshl_b32 s0, s4, 5 // (op 9424) 
s_add_u32 s0, s0, 96 // (op 9424) 
v_add_u32 v0, v13, s0 // (op 9424) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9424) Store value



v_accvgpr_read v4, a56 // (op 9438) 
v_accvgpr_read v5, a57 // (op 9438) 
v_accvgpr_read v6, a58 // (op 9438) 
v_accvgpr_read v7, a59 // (op 9438) 
v_mul_f32 v14, v11, v4 // (op 9438) 
v_mul_f32 v15, v11, v5 // (op 9438) 
v_mul_f32 v16, v11, v6 // (op 9438) 
v_mul_f32 v17, v11, v7 // (op 9438) 

v_lshrrev_b32 v4, 16, v14 // (op 9440) 
v_lshrrev_b32 v5, 16, v15 // (op 9440) 
v_lshrrev_b32 v6, 16, v16 // (op 9440) 
v_lshrrev_b32 v7, 16, v17 // (op 9440) 


v_pack_b32_f16 v14, v4, v5 // (op 9426) 
v_pack_b32_f16 v15, v6, v7 // (op 9426) 

s_lshl_b32 s0, s4, 6 // (op 9426) 
s_add_u32 s0, s0, 96 // (op 9426) 
v_add_u32 v0, v13, s0 // (op 9426) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9426) Store value



v_accvgpr_read v4, a60 // (op 9444) 
v_accvgpr_read v5, a61 // (op 9444) 
v_accvgpr_read v6, a62 // (op 9444) 
v_accvgpr_read v7, a63 // (op 9444) 
v_mul_f32 v14, v11, v4 // (op 9444) 
v_mul_f32 v15, v11, v5 // (op 9444) 
v_mul_f32 v16, v11, v6 // (op 9444) 
v_mul_f32 v17, v11, v7 // (op 9444) 

v_lshrrev_b32 v4, 16, v14 // (op 9446) 
v_lshrrev_b32 v5, 16, v15 // (op 9446) 
v_lshrrev_b32 v6, 16, v16 // (op 9446) 
v_lshrrev_b32 v7, 16, v17 // (op 9446) 


v_pack_b32_f16 v14, v4, v5 // (op 9428) 
v_pack_b32_f16 v15, v6, v7 // (op 9428) 

s_lshl_b32 s0, s4, 5 // (op 9428) 
s_mul_i32 s1, 3, s0 // (op 9428) 
s_add_u32 s1, s1, 96 // (op 9428) 

v_add_u32 v0, v13, s1 // (op 9428) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9428) Store value



v_accvgpr_read v4, a64 // (op 9494) 
v_accvgpr_read v5, a65 // (op 9494) 
v_accvgpr_read v6, a66 // (op 9494) 
v_accvgpr_read v7, a67 // (op 9494) 
v_mul_f32 v14, v11, v4 // (op 9494) 
v_mul_f32 v15, v11, v5 // (op 9494) 
v_mul_f32 v16, v11, v6 // (op 9494) 
v_mul_f32 v17, v11, v7 // (op 9494) 

v_lshrrev_b32 v4, 16, v14 // (op 9496) 
v_lshrrev_b32 v5, 16, v15 // (op 9496) 
v_lshrrev_b32 v6, 16, v16 // (op 9496) 
v_lshrrev_b32 v7, 16, v17 // (op 9496) 


v_pack_b32_f16 v14, v4, v5 // (op 9498) 
v_pack_b32_f16 v15, v6, v7 // (op 9498) 

v_add_u32 v0, 128, v13 // (op 9498) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9498) Store value



v_accvgpr_read v4, a68 // (op 9508) 
v_accvgpr_read v5, a69 // (op 9508) 
v_accvgpr_read v6, a70 // (op 9508) 
v_accvgpr_read v7, a71 // (op 9508) 
v_mul_f32 v14, v11, v4 // (op 9508) 
v_mul_f32 v15, v11, v5 // (op 9508) 
v_mul_f32 v16, v11, v6 // (op 9508) 
v_mul_f32 v17, v11, v7 // (op 9508) 

v_lshrrev_b32 v4, 16, v14 // (op 9510) 
v_lshrrev_b32 v5, 16, v15 // (op 9510) 
v_lshrrev_b32 v6, 16, v16 // (op 9510) 
v_lshrrev_b32 v7, 16, v17 // (op 9510) 


v_pack_b32_f16 v14, v4, v5 // (op 9500) 
v_pack_b32_f16 v15, v6, v7 // (op 9500) 

s_lshl_b32 s0, s4, 5 // (op 9500) 
s_add_u32 s0, s0, 128 // (op 9500) 
v_add_u32 v0, v13, s0 // (op 9500) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9500) Store value



v_accvgpr_read v4, a72 // (op 9514) 
v_accvgpr_read v5, a73 // (op 9514) 
v_accvgpr_read v6, a74 // (op 9514) 
v_accvgpr_read v7, a75 // (op 9514) 
v_mul_f32 v14, v11, v4 // (op 9514) 
v_mul_f32 v15, v11, v5 // (op 9514) 
v_mul_f32 v16, v11, v6 // (op 9514) 
v_mul_f32 v17, v11, v7 // (op 9514) 

v_lshrrev_b32 v4, 16, v14 // (op 9516) 
v_lshrrev_b32 v5, 16, v15 // (op 9516) 
v_lshrrev_b32 v6, 16, v16 // (op 9516) 
v_lshrrev_b32 v7, 16, v17 // (op 9516) 


v_pack_b32_f16 v14, v4, v5 // (op 9502) 
v_pack_b32_f16 v15, v6, v7 // (op 9502) 

s_lshl_b32 s0, s4, 6 // (op 9502) 
s_add_u32 s0, s0, 128 // (op 9502) 
v_add_u32 v0, v13, s0 // (op 9502) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9502) Store value



v_accvgpr_read v4, a76 // (op 9520) 
v_accvgpr_read v5, a77 // (op 9520) 
v_accvgpr_read v6, a78 // (op 9520) 
v_accvgpr_read v7, a79 // (op 9520) 
v_mul_f32 v14, v11, v4 // (op 9520) 
v_mul_f32 v15, v11, v5 // (op 9520) 
v_mul_f32 v16, v11, v6 // (op 9520) 
v_mul_f32 v17, v11, v7 // (op 9520) 

v_lshrrev_b32 v4, 16, v14 // (op 9522) 
v_lshrrev_b32 v5, 16, v15 // (op 9522) 
v_lshrrev_b32 v6, 16, v16 // (op 9522) 
v_lshrrev_b32 v7, 16, v17 // (op 9522) 


v_pack_b32_f16 v14, v4, v5 // (op 9504) 
v_pack_b32_f16 v15, v6, v7 // (op 9504) 

s_lshl_b32 s0, s4, 5 // (op 9504) 
s_mul_i32 s1, 3, s0 // (op 9504) 
s_add_u32 s1, s1, 128 // (op 9504) 

v_add_u32 v0, v13, s1 // (op 9504) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9504) Store value



v_accvgpr_read v4, a80 // (op 9570) 
v_accvgpr_read v5, a81 // (op 9570) 
v_accvgpr_read v6, a82 // (op 9570) 
v_accvgpr_read v7, a83 // (op 9570) 
v_mul_f32 v14, v11, v4 // (op 9570) 
v_mul_f32 v15, v11, v5 // (op 9570) 
v_mul_f32 v16, v11, v6 // (op 9570) 
v_mul_f32 v17, v11, v7 // (op 9570) 

v_lshrrev_b32 v4, 16, v14 // (op 9572) 
v_lshrrev_b32 v5, 16, v15 // (op 9572) 
v_lshrrev_b32 v6, 16, v16 // (op 9572) 
v_lshrrev_b32 v7, 16, v17 // (op 9572) 


v_pack_b32_f16 v14, v4, v5 // (op 9574) 
v_pack_b32_f16 v15, v6, v7 // (op 9574) 

v_add_u32 v0, 160, v13 // (op 9574) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9574) Store value



v_accvgpr_read v4, a84 // (op 9584) 
v_accvgpr_read v5, a85 // (op 9584) 
v_accvgpr_read v6, a86 // (op 9584) 
v_accvgpr_read v7, a87 // (op 9584) 
v_mul_f32 v14, v11, v4 // (op 9584) 
v_mul_f32 v15, v11, v5 // (op 9584) 
v_mul_f32 v16, v11, v6 // (op 9584) 
v_mul_f32 v17, v11, v7 // (op 9584) 

v_lshrrev_b32 v4, 16, v14 // (op 9586) 
v_lshrrev_b32 v5, 16, v15 // (op 9586) 
v_lshrrev_b32 v6, 16, v16 // (op 9586) 
v_lshrrev_b32 v7, 16, v17 // (op 9586) 


v_pack_b32_f16 v14, v4, v5 // (op 9576) 
v_pack_b32_f16 v15, v6, v7 // (op 9576) 

s_lshl_b32 s0, s4, 5 // (op 9576) 
s_add_u32 s0, s0, 160 // (op 9576) 
v_add_u32 v0, v13, s0 // (op 9576) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9576) Store value



v_accvgpr_read v4, a88 // (op 9590) 
v_accvgpr_read v5, a89 // (op 9590) 
v_accvgpr_read v6, a90 // (op 9590) 
v_accvgpr_read v7, a91 // (op 9590) 
v_mul_f32 v14, v11, v4 // (op 9590) 
v_mul_f32 v15, v11, v5 // (op 9590) 
v_mul_f32 v16, v11, v6 // (op 9590) 
v_mul_f32 v17, v11, v7 // (op 9590) 

v_lshrrev_b32 v4, 16, v14 // (op 9592) 
v_lshrrev_b32 v5, 16, v15 // (op 9592) 
v_lshrrev_b32 v6, 16, v16 // (op 9592) 
v_lshrrev_b32 v7, 16, v17 // (op 9592) 


v_pack_b32_f16 v14, v4, v5 // (op 9578) 
v_pack_b32_f16 v15, v6, v7 // (op 9578) 

s_lshl_b32 s0, s4, 6 // (op 9578) 
s_add_u32 s0, s0, 160 // (op 9578) 
v_add_u32 v0, v13, s0 // (op 9578) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9578) Store value



v_accvgpr_read v4, a92 // (op 9596) 
v_accvgpr_read v5, a93 // (op 9596) 
v_accvgpr_read v6, a94 // (op 9596) 
v_accvgpr_read v7, a95 // (op 9596) 
v_mul_f32 v14, v11, v4 // (op 9596) 
v_mul_f32 v15, v11, v5 // (op 9596) 
v_mul_f32 v16, v11, v6 // (op 9596) 
v_mul_f32 v17, v11, v7 // (op 9596) 

v_lshrrev_b32 v4, 16, v14 // (op 9598) 
v_lshrrev_b32 v5, 16, v15 // (op 9598) 
v_lshrrev_b32 v6, 16, v16 // (op 9598) 
v_lshrrev_b32 v7, 16, v17 // (op 9598) 


v_pack_b32_f16 v14, v4, v5 // (op 9580) 
v_pack_b32_f16 v15, v6, v7 // (op 9580) 

s_lshl_b32 s0, s4, 5 // (op 9580) 
s_mul_i32 s1, 3, s0 // (op 9580) 
s_add_u32 s1, s1, 160 // (op 9580) 

v_add_u32 v0, v13, s1 // (op 9580) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9580) Store value



v_accvgpr_read v4, a96 // (op 9646) 
v_accvgpr_read v5, a97 // (op 9646) 
v_accvgpr_read v6, a98 // (op 9646) 
v_accvgpr_read v7, a99 // (op 9646) 
v_mul_f32 v14, v11, v4 // (op 9646) 
v_mul_f32 v15, v11, v5 // (op 9646) 
v_mul_f32 v16, v11, v6 // (op 9646) 
v_mul_f32 v17, v11, v7 // (op 9646) 

v_lshrrev_b32 v4, 16, v14 // (op 9648) 
v_lshrrev_b32 v5, 16, v15 // (op 9648) 
v_lshrrev_b32 v6, 16, v16 // (op 9648) 
v_lshrrev_b32 v7, 16, v17 // (op 9648) 


v_pack_b32_f16 v14, v4, v5 // (op 9650) 
v_pack_b32_f16 v15, v6, v7 // (op 9650) 

v_add_u32 v0, 192, v13 // (op 9650) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9650) Store value



v_accvgpr_read v4, a100 // (op 9660) 
v_accvgpr_read v5, a101 // (op 9660) 
v_accvgpr_read v6, a102 // (op 9660) 
v_accvgpr_read v7, a103 // (op 9660) 
v_mul_f32 v14, v11, v4 // (op 9660) 
v_mul_f32 v15, v11, v5 // (op 9660) 
v_mul_f32 v16, v11, v6 // (op 9660) 
v_mul_f32 v17, v11, v7 // (op 9660) 

v_lshrrev_b32 v4, 16, v14 // (op 9662) 
v_lshrrev_b32 v5, 16, v15 // (op 9662) 
v_lshrrev_b32 v6, 16, v16 // (op 9662) 
v_lshrrev_b32 v7, 16, v17 // (op 9662) 


v_pack_b32_f16 v14, v4, v5 // (op 9652) 
v_pack_b32_f16 v15, v6, v7 // (op 9652) 

s_lshl_b32 s0, s4, 5 // (op 9652) 
s_add_u32 s0, s0, 192 // (op 9652) 
v_add_u32 v0, v13, s0 // (op 9652) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9652) Store value



v_accvgpr_read v4, a104 // (op 9666) 
v_accvgpr_read v5, a105 // (op 9666) 
v_accvgpr_read v6, a106 // (op 9666) 
v_accvgpr_read v7, a107 // (op 9666) 
v_mul_f32 v14, v11, v4 // (op 9666) 
v_mul_f32 v15, v11, v5 // (op 9666) 
v_mul_f32 v16, v11, v6 // (op 9666) 
v_mul_f32 v17, v11, v7 // (op 9666) 

v_lshrrev_b32 v4, 16, v14 // (op 9668) 
v_lshrrev_b32 v5, 16, v15 // (op 9668) 
v_lshrrev_b32 v6, 16, v16 // (op 9668) 
v_lshrrev_b32 v7, 16, v17 // (op 9668) 


v_pack_b32_f16 v14, v4, v5 // (op 9654) 
v_pack_b32_f16 v15, v6, v7 // (op 9654) 

s_lshl_b32 s0, s4, 6 // (op 9654) 
s_add_u32 s0, s0, 192 // (op 9654) 
v_add_u32 v0, v13, s0 // (op 9654) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9654) Store value



v_accvgpr_read v4, a108 // (op 9672) 
v_accvgpr_read v5, a109 // (op 9672) 
v_accvgpr_read v6, a110 // (op 9672) 
v_accvgpr_read v7, a111 // (op 9672) 
v_mul_f32 v14, v11, v4 // (op 9672) 
v_mul_f32 v15, v11, v5 // (op 9672) 
v_mul_f32 v16, v11, v6 // (op 9672) 
v_mul_f32 v17, v11, v7 // (op 9672) 

v_lshrrev_b32 v4, 16, v14 // (op 9674) 
v_lshrrev_b32 v5, 16, v15 // (op 9674) 
v_lshrrev_b32 v6, 16, v16 // (op 9674) 
v_lshrrev_b32 v7, 16, v17 // (op 9674) 


v_pack_b32_f16 v14, v4, v5 // (op 9656) 
v_pack_b32_f16 v15, v6, v7 // (op 9656) 

s_lshl_b32 s0, s4, 5 // (op 9656) 
s_mul_i32 s1, 3, s0 // (op 9656) 
s_add_u32 s1, s1, 192 // (op 9656) 

v_add_u32 v0, v13, s1 // (op 9656) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9656) Store value



v_accvgpr_read v4, a112 // (op 9722) 
v_accvgpr_read v5, a113 // (op 9722) 
v_accvgpr_read v6, a114 // (op 9722) 
v_accvgpr_read v7, a115 // (op 9722) 
v_mul_f32 v14, v11, v4 // (op 9722) 
v_mul_f32 v15, v11, v5 // (op 9722) 
v_mul_f32 v16, v11, v6 // (op 9722) 
v_mul_f32 v17, v11, v7 // (op 9722) 

v_lshrrev_b32 v4, 16, v14 // (op 9724) 
v_lshrrev_b32 v5, 16, v15 // (op 9724) 
v_lshrrev_b32 v6, 16, v16 // (op 9724) 
v_lshrrev_b32 v7, 16, v17 // (op 9724) 


v_pack_b32_f16 v14, v4, v5 // (op 9726) 
v_pack_b32_f16 v15, v6, v7 // (op 9726) 

v_add_u32 v0, 224, v13 // (op 9726) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9726) Store value



v_accvgpr_read v4, a116 // (op 9736) 
v_accvgpr_read v5, a117 // (op 9736) 
v_accvgpr_read v6, a118 // (op 9736) 
v_accvgpr_read v7, a119 // (op 9736) 
v_mul_f32 v14, v11, v4 // (op 9736) 
v_mul_f32 v15, v11, v5 // (op 9736) 
v_mul_f32 v16, v11, v6 // (op 9736) 
v_mul_f32 v17, v11, v7 // (op 9736) 

v_lshrrev_b32 v4, 16, v14 // (op 9738) 
v_lshrrev_b32 v5, 16, v15 // (op 9738) 
v_lshrrev_b32 v6, 16, v16 // (op 9738) 
v_lshrrev_b32 v7, 16, v17 // (op 9738) 


v_pack_b32_f16 v14, v4, v5 // (op 9728) 
v_pack_b32_f16 v15, v6, v7 // (op 9728) 

s_lshl_b32 s0, s4, 5 // (op 9728) 
s_add_u32 s0, s0, 224 // (op 9728) 
v_add_u32 v0, v13, s0 // (op 9728) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9728) Store value



v_accvgpr_read v4, a120 // (op 9742) 
v_accvgpr_read v5, a121 // (op 9742) 
v_accvgpr_read v6, a122 // (op 9742) 
v_accvgpr_read v7, a123 // (op 9742) 
v_mul_f32 v14, v11, v4 // (op 9742) 
v_mul_f32 v15, v11, v5 // (op 9742) 
v_mul_f32 v16, v11, v6 // (op 9742) 
v_mul_f32 v17, v11, v7 // (op 9742) 

v_lshrrev_b32 v4, 16, v14 // (op 9744) 
v_lshrrev_b32 v5, 16, v15 // (op 9744) 
v_lshrrev_b32 v6, 16, v16 // (op 9744) 
v_lshrrev_b32 v7, 16, v17 // (op 9744) 


v_pack_b32_f16 v14, v4, v5 // (op 9730) 
v_pack_b32_f16 v15, v6, v7 // (op 9730) 

s_lshl_b32 s0, s4, 6 // (op 9730) 
s_add_u32 s0, s0, 224 // (op 9730) 
v_add_u32 v0, v13, s0 // (op 9730) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9730) Store value



v_accvgpr_read v4, a124 // (op 9748) 
v_accvgpr_read v5, a125 // (op 9748) 
v_accvgpr_read v6, a126 // (op 9748) 
v_accvgpr_read v7, a127 // (op 9748) 
v_mul_f32 v14, v11, v4 // (op 9748) 
v_mul_f32 v15, v11, v5 // (op 9748) 
v_mul_f32 v16, v11, v6 // (op 9748) 
v_mul_f32 v17, v11, v7 // (op 9748) 

v_lshrrev_b32 v4, 16, v14 // (op 9750) 
v_lshrrev_b32 v5, 16, v15 // (op 9750) 
v_lshrrev_b32 v6, 16, v16 // (op 9750) 
v_lshrrev_b32 v7, 16, v17 // (op 9750) 


v_pack_b32_f16 v14, v4, v5 // (op 9732) 
v_pack_b32_f16 v15, v6, v7 // (op 9732) 

s_lshl_b32 s0, s4, 5 // (op 9732) 
s_mul_i32 s1, 3, s0 // (op 9732) 
s_add_u32 s1, s1, 224 // (op 9732) 

v_add_u32 v0, v13, s1 // (op 9732) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 9732) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ConditionalBottom_0_beta_116
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ConditionalFalse_0_beta_116:


v_add_u32 v0, 224, v12 // (op 10524) 

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10524) Load value

v_accvgpr_read v14, a112 // (op 10511) 
v_accvgpr_read v15, a113 // (op 10511) 
v_accvgpr_read v16, a114 // (op 10511) 
v_accvgpr_read v17, a115 // (op 10511) 
v_mul_f32 v18, v11, v14 // (op 10511) 
v_mul_f32 v19, v11, v15 // (op 10511) 
v_mul_f32 v20, v11, v16 // (op 10511) 
v_mul_f32 v21, v11, v17 // (op 10511) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10523) 
v_lshrrev_b32 v7, 16, v4 // (op 10523) 
v_lshlrev_b32 v6, 16, v6 // (op 10523) 
v_lshlrev_b32 v7, 16, v7 // (op 10523) 
v_mul_f32 v14, v9, v6 // (op 10523) 
v_mul_f32 v15, v9, v7 // (op 10523) 
v_and_b32 v6, 65535, v5 // (op 10523) 
v_lshrrev_b32 v7, 16, v5 // (op 10523) 
v_lshlrev_b32 v6, 16, v6 // (op 10523) 
v_lshlrev_b32 v7, 16, v7 // (op 10523) 
v_mul_f32 v16, v9, v6 // (op 10523) 
v_mul_f32 v17, v9, v7 // (op 10523) 


v_add_f32 v4, v14, v18 // (op 10512) 
v_add_f32 v5, v15, v19 // (op 10512) 
v_add_f32 v6, v16, v20 // (op 10512) 
v_add_f32 v7, v17, v21 // (op 10512) 


v_lshrrev_b32 v14, 16, v4 // (op 10513) 
v_lshrrev_b32 v15, 16, v5 // (op 10513) 
v_lshrrev_b32 v16, 16, v6 // (op 10513) 
v_lshrrev_b32 v17, 16, v7 // (op 10513) 


v_pack_b32_f16 v4, v14, v15 // (op 10515) 
v_pack_b32_f16 v5, v16, v17 // (op 10515) 

v_add_u32 v0, 224, v13 // (op 10515) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10515) Store value





s_lshl_b32 s0, s2, 5 // (op 10529) 
s_add_u32 s0, s0, 224 // (op 10529) 
v_add_u32 v0, v12, s0 // (op 10529) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10529) Load value

v_accvgpr_read v14, a116 // (op 10541) 
v_accvgpr_read v15, a117 // (op 10541) 
v_accvgpr_read v16, a118 // (op 10541) 
v_accvgpr_read v17, a119 // (op 10541) 
v_mul_f32 v18, v11, v14 // (op 10541) 
v_mul_f32 v19, v11, v15 // (op 10541) 
v_mul_f32 v20, v11, v16 // (op 10541) 
v_mul_f32 v21, v11, v17 // (op 10541) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10526) 
v_lshrrev_b32 v7, 16, v4 // (op 10526) 
v_lshlrev_b32 v6, 16, v6 // (op 10526) 
v_lshlrev_b32 v7, 16, v7 // (op 10526) 
v_mul_f32 v14, v9, v6 // (op 10526) 
v_mul_f32 v15, v9, v7 // (op 10526) 
v_and_b32 v6, 65535, v5 // (op 10526) 
v_lshrrev_b32 v7, 16, v5 // (op 10526) 
v_lshlrev_b32 v6, 16, v6 // (op 10526) 
v_lshlrev_b32 v7, 16, v7 // (op 10526) 
v_mul_f32 v16, v9, v6 // (op 10526) 
v_mul_f32 v17, v9, v7 // (op 10526) 


v_add_f32 v4, v14, v18 // (op 10527) 
v_add_f32 v5, v15, v19 // (op 10527) 
v_add_f32 v6, v16, v20 // (op 10527) 
v_add_f32 v7, v17, v21 // (op 10527) 


v_lshrrev_b32 v14, 16, v4 // (op 10528) 
v_lshrrev_b32 v15, 16, v5 // (op 10528) 
v_lshrrev_b32 v16, 16, v6 // (op 10528) 
v_lshrrev_b32 v17, 16, v7 // (op 10528) 


v_pack_b32_f16 v4, v14, v15 // (op 10517) 
v_pack_b32_f16 v5, v16, v17 // (op 10517) 

s_lshl_b32 s0, s4, 5 // (op 10517) 
s_add_u32 s0, s0, 224 // (op 10517) 
v_add_u32 v0, v13, s0 // (op 10517) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10517) Store value





s_lshl_b32 s0, s2, 6 // (op 10534) 
s_add_u32 s0, s0, 224 // (op 10534) 
v_add_u32 v0, v12, s0 // (op 10534) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10534) Load value

v_accvgpr_read v14, a120 // (op 10543) 
v_accvgpr_read v15, a121 // (op 10543) 
v_accvgpr_read v16, a122 // (op 10543) 
v_accvgpr_read v17, a123 // (op 10543) 
v_mul_f32 v18, v11, v14 // (op 10543) 
v_mul_f32 v19, v11, v15 // (op 10543) 
v_mul_f32 v20, v11, v16 // (op 10543) 
v_mul_f32 v21, v11, v17 // (op 10543) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10531) 
v_lshrrev_b32 v7, 16, v4 // (op 10531) 
v_lshlrev_b32 v6, 16, v6 // (op 10531) 
v_lshlrev_b32 v7, 16, v7 // (op 10531) 
v_mul_f32 v14, v9, v6 // (op 10531) 
v_mul_f32 v15, v9, v7 // (op 10531) 
v_and_b32 v6, 65535, v5 // (op 10531) 
v_lshrrev_b32 v7, 16, v5 // (op 10531) 
v_lshlrev_b32 v6, 16, v6 // (op 10531) 
v_lshlrev_b32 v7, 16, v7 // (op 10531) 
v_mul_f32 v16, v9, v6 // (op 10531) 
v_mul_f32 v17, v9, v7 // (op 10531) 


v_add_f32 v4, v14, v18 // (op 10532) 
v_add_f32 v5, v15, v19 // (op 10532) 
v_add_f32 v6, v16, v20 // (op 10532) 
v_add_f32 v7, v17, v21 // (op 10532) 


v_lshrrev_b32 v14, 16, v4 // (op 10533) 
v_lshrrev_b32 v15, 16, v5 // (op 10533) 
v_lshrrev_b32 v16, 16, v6 // (op 10533) 
v_lshrrev_b32 v17, 16, v7 // (op 10533) 


v_pack_b32_f16 v4, v14, v15 // (op 10519) 
v_pack_b32_f16 v5, v16, v17 // (op 10519) 

s_lshl_b32 s0, s4, 6 // (op 10519) 
s_add_u32 s0, s0, 224 // (op 10519) 
v_add_u32 v0, v13, s0 // (op 10519) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10519) Store value





s_lshl_b32 s0, s2, 5 // (op 10539) 
s_mul_i32 s1, 3, s0 // (op 10539) 
s_add_u32 s1, s1, 224 // (op 10539) 

v_add_u32 v0, v12, s1 // (op 10539) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10539) Load value

v_accvgpr_read v14, a124 // (op 10545) 
v_accvgpr_read v15, a125 // (op 10545) 
v_accvgpr_read v16, a126 // (op 10545) 
v_accvgpr_read v17, a127 // (op 10545) 
v_mul_f32 v18, v11, v14 // (op 10545) 
v_mul_f32 v19, v11, v15 // (op 10545) 
v_mul_f32 v20, v11, v16 // (op 10545) 
v_mul_f32 v21, v11, v17 // (op 10545) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10536) 
v_lshrrev_b32 v7, 16, v4 // (op 10536) 
v_lshlrev_b32 v6, 16, v6 // (op 10536) 
v_lshlrev_b32 v7, 16, v7 // (op 10536) 
v_mul_f32 v14, v9, v6 // (op 10536) 
v_mul_f32 v15, v9, v7 // (op 10536) 
v_and_b32 v6, 65535, v5 // (op 10536) 
v_lshrrev_b32 v7, 16, v5 // (op 10536) 
v_lshlrev_b32 v6, 16, v6 // (op 10536) 
v_lshlrev_b32 v7, 16, v7 // (op 10536) 
v_mul_f32 v16, v9, v6 // (op 10536) 
v_mul_f32 v17, v9, v7 // (op 10536) 


v_add_f32 v4, v14, v18 // (op 10537) 
v_add_f32 v5, v15, v19 // (op 10537) 
v_add_f32 v6, v16, v20 // (op 10537) 
v_add_f32 v7, v17, v21 // (op 10537) 


v_lshrrev_b32 v14, 16, v4 // (op 10538) 
v_lshrrev_b32 v15, 16, v5 // (op 10538) 
v_lshrrev_b32 v16, 16, v6 // (op 10538) 
v_lshrrev_b32 v17, 16, v7 // (op 10538) 


v_pack_b32_f16 v4, v14, v15 // (op 10521) 
v_pack_b32_f16 v5, v16, v17 // (op 10521) 

s_lshl_b32 s0, s4, 5 // (op 10521) 
s_mul_i32 s1, 3, s0 // (op 10521) 
s_add_u32 s1, s1, 224 // (op 10521) 

v_add_u32 v0, v13, s1 // (op 10521) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10521) Store value





v_add_u32 v0, 192, v12 // (op 10437) 

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10437) Load value

v_accvgpr_read v14, a96 // (op 10424) 
v_accvgpr_read v15, a97 // (op 10424) 
v_accvgpr_read v16, a98 // (op 10424) 
v_accvgpr_read v17, a99 // (op 10424) 
v_mul_f32 v18, v11, v14 // (op 10424) 
v_mul_f32 v19, v11, v15 // (op 10424) 
v_mul_f32 v20, v11, v16 // (op 10424) 
v_mul_f32 v21, v11, v17 // (op 10424) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10436) 
v_lshrrev_b32 v7, 16, v4 // (op 10436) 
v_lshlrev_b32 v6, 16, v6 // (op 10436) 
v_lshlrev_b32 v7, 16, v7 // (op 10436) 
v_mul_f32 v14, v9, v6 // (op 10436) 
v_mul_f32 v15, v9, v7 // (op 10436) 
v_and_b32 v6, 65535, v5 // (op 10436) 
v_lshrrev_b32 v7, 16, v5 // (op 10436) 
v_lshlrev_b32 v6, 16, v6 // (op 10436) 
v_lshlrev_b32 v7, 16, v7 // (op 10436) 
v_mul_f32 v16, v9, v6 // (op 10436) 
v_mul_f32 v17, v9, v7 // (op 10436) 


v_add_f32 v4, v14, v18 // (op 10425) 
v_add_f32 v5, v15, v19 // (op 10425) 
v_add_f32 v6, v16, v20 // (op 10425) 
v_add_f32 v7, v17, v21 // (op 10425) 


v_lshrrev_b32 v14, 16, v4 // (op 10426) 
v_lshrrev_b32 v15, 16, v5 // (op 10426) 
v_lshrrev_b32 v16, 16, v6 // (op 10426) 
v_lshrrev_b32 v17, 16, v7 // (op 10426) 


v_pack_b32_f16 v4, v14, v15 // (op 10428) 
v_pack_b32_f16 v5, v16, v17 // (op 10428) 

v_add_u32 v0, 192, v13 // (op 10428) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10428) Store value





s_lshl_b32 s0, s2, 5 // (op 10442) 
s_add_u32 s0, s0, 192 // (op 10442) 
v_add_u32 v0, v12, s0 // (op 10442) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10442) Load value

v_accvgpr_read v14, a100 // (op 10454) 
v_accvgpr_read v15, a101 // (op 10454) 
v_accvgpr_read v16, a102 // (op 10454) 
v_accvgpr_read v17, a103 // (op 10454) 
v_mul_f32 v18, v11, v14 // (op 10454) 
v_mul_f32 v19, v11, v15 // (op 10454) 
v_mul_f32 v20, v11, v16 // (op 10454) 
v_mul_f32 v21, v11, v17 // (op 10454) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10439) 
v_lshrrev_b32 v7, 16, v4 // (op 10439) 
v_lshlrev_b32 v6, 16, v6 // (op 10439) 
v_lshlrev_b32 v7, 16, v7 // (op 10439) 
v_mul_f32 v14, v9, v6 // (op 10439) 
v_mul_f32 v15, v9, v7 // (op 10439) 
v_and_b32 v6, 65535, v5 // (op 10439) 
v_lshrrev_b32 v7, 16, v5 // (op 10439) 
v_lshlrev_b32 v6, 16, v6 // (op 10439) 
v_lshlrev_b32 v7, 16, v7 // (op 10439) 
v_mul_f32 v16, v9, v6 // (op 10439) 
v_mul_f32 v17, v9, v7 // (op 10439) 


v_add_f32 v4, v14, v18 // (op 10440) 
v_add_f32 v5, v15, v19 // (op 10440) 
v_add_f32 v6, v16, v20 // (op 10440) 
v_add_f32 v7, v17, v21 // (op 10440) 


v_lshrrev_b32 v14, 16, v4 // (op 10441) 
v_lshrrev_b32 v15, 16, v5 // (op 10441) 
v_lshrrev_b32 v16, 16, v6 // (op 10441) 
v_lshrrev_b32 v17, 16, v7 // (op 10441) 


v_pack_b32_f16 v4, v14, v15 // (op 10430) 
v_pack_b32_f16 v5, v16, v17 // (op 10430) 

s_lshl_b32 s0, s4, 5 // (op 10430) 
s_add_u32 s0, s0, 192 // (op 10430) 
v_add_u32 v0, v13, s0 // (op 10430) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10430) Store value





s_lshl_b32 s0, s2, 6 // (op 10447) 
s_add_u32 s0, s0, 192 // (op 10447) 
v_add_u32 v0, v12, s0 // (op 10447) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10447) Load value

v_accvgpr_read v14, a104 // (op 10456) 
v_accvgpr_read v15, a105 // (op 10456) 
v_accvgpr_read v16, a106 // (op 10456) 
v_accvgpr_read v17, a107 // (op 10456) 
v_mul_f32 v18, v11, v14 // (op 10456) 
v_mul_f32 v19, v11, v15 // (op 10456) 
v_mul_f32 v20, v11, v16 // (op 10456) 
v_mul_f32 v21, v11, v17 // (op 10456) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10444) 
v_lshrrev_b32 v7, 16, v4 // (op 10444) 
v_lshlrev_b32 v6, 16, v6 // (op 10444) 
v_lshlrev_b32 v7, 16, v7 // (op 10444) 
v_mul_f32 v14, v9, v6 // (op 10444) 
v_mul_f32 v15, v9, v7 // (op 10444) 
v_and_b32 v6, 65535, v5 // (op 10444) 
v_lshrrev_b32 v7, 16, v5 // (op 10444) 
v_lshlrev_b32 v6, 16, v6 // (op 10444) 
v_lshlrev_b32 v7, 16, v7 // (op 10444) 
v_mul_f32 v16, v9, v6 // (op 10444) 
v_mul_f32 v17, v9, v7 // (op 10444) 


v_add_f32 v4, v14, v18 // (op 10445) 
v_add_f32 v5, v15, v19 // (op 10445) 
v_add_f32 v6, v16, v20 // (op 10445) 
v_add_f32 v7, v17, v21 // (op 10445) 


v_lshrrev_b32 v14, 16, v4 // (op 10446) 
v_lshrrev_b32 v15, 16, v5 // (op 10446) 
v_lshrrev_b32 v16, 16, v6 // (op 10446) 
v_lshrrev_b32 v17, 16, v7 // (op 10446) 


v_pack_b32_f16 v4, v14, v15 // (op 10432) 
v_pack_b32_f16 v5, v16, v17 // (op 10432) 

s_lshl_b32 s0, s4, 6 // (op 10432) 
s_add_u32 s0, s0, 192 // (op 10432) 
v_add_u32 v0, v13, s0 // (op 10432) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10432) Store value





s_lshl_b32 s0, s2, 5 // (op 10452) 
s_mul_i32 s1, 3, s0 // (op 10452) 
s_add_u32 s1, s1, 192 // (op 10452) 

v_add_u32 v0, v12, s1 // (op 10452) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10452) Load value

v_accvgpr_read v14, a108 // (op 10458) 
v_accvgpr_read v15, a109 // (op 10458) 
v_accvgpr_read v16, a110 // (op 10458) 
v_accvgpr_read v17, a111 // (op 10458) 
v_mul_f32 v18, v11, v14 // (op 10458) 
v_mul_f32 v19, v11, v15 // (op 10458) 
v_mul_f32 v20, v11, v16 // (op 10458) 
v_mul_f32 v21, v11, v17 // (op 10458) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10449) 
v_lshrrev_b32 v7, 16, v4 // (op 10449) 
v_lshlrev_b32 v6, 16, v6 // (op 10449) 
v_lshlrev_b32 v7, 16, v7 // (op 10449) 
v_mul_f32 v14, v9, v6 // (op 10449) 
v_mul_f32 v15, v9, v7 // (op 10449) 
v_and_b32 v6, 65535, v5 // (op 10449) 
v_lshrrev_b32 v7, 16, v5 // (op 10449) 
v_lshlrev_b32 v6, 16, v6 // (op 10449) 
v_lshlrev_b32 v7, 16, v7 // (op 10449) 
v_mul_f32 v16, v9, v6 // (op 10449) 
v_mul_f32 v17, v9, v7 // (op 10449) 


v_add_f32 v4, v14, v18 // (op 10450) 
v_add_f32 v5, v15, v19 // (op 10450) 
v_add_f32 v6, v16, v20 // (op 10450) 
v_add_f32 v7, v17, v21 // (op 10450) 


v_lshrrev_b32 v14, 16, v4 // (op 10451) 
v_lshrrev_b32 v15, 16, v5 // (op 10451) 
v_lshrrev_b32 v16, 16, v6 // (op 10451) 
v_lshrrev_b32 v17, 16, v7 // (op 10451) 


v_pack_b32_f16 v4, v14, v15 // (op 10434) 
v_pack_b32_f16 v5, v16, v17 // (op 10434) 

s_lshl_b32 s0, s4, 5 // (op 10434) 
s_mul_i32 s1, 3, s0 // (op 10434) 
s_add_u32 s1, s1, 192 // (op 10434) 

v_add_u32 v0, v13, s1 // (op 10434) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10434) Store value





v_add_u32 v0, 160, v12 // (op 10350) 

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10350) Load value

v_accvgpr_read v14, a80 // (op 10337) 
v_accvgpr_read v15, a81 // (op 10337) 
v_accvgpr_read v16, a82 // (op 10337) 
v_accvgpr_read v17, a83 // (op 10337) 
v_mul_f32 v18, v11, v14 // (op 10337) 
v_mul_f32 v19, v11, v15 // (op 10337) 
v_mul_f32 v20, v11, v16 // (op 10337) 
v_mul_f32 v21, v11, v17 // (op 10337) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10349) 
v_lshrrev_b32 v7, 16, v4 // (op 10349) 
v_lshlrev_b32 v6, 16, v6 // (op 10349) 
v_lshlrev_b32 v7, 16, v7 // (op 10349) 
v_mul_f32 v14, v9, v6 // (op 10349) 
v_mul_f32 v15, v9, v7 // (op 10349) 
v_and_b32 v6, 65535, v5 // (op 10349) 
v_lshrrev_b32 v7, 16, v5 // (op 10349) 
v_lshlrev_b32 v6, 16, v6 // (op 10349) 
v_lshlrev_b32 v7, 16, v7 // (op 10349) 
v_mul_f32 v16, v9, v6 // (op 10349) 
v_mul_f32 v17, v9, v7 // (op 10349) 


v_add_f32 v4, v14, v18 // (op 10338) 
v_add_f32 v5, v15, v19 // (op 10338) 
v_add_f32 v6, v16, v20 // (op 10338) 
v_add_f32 v7, v17, v21 // (op 10338) 


v_lshrrev_b32 v14, 16, v4 // (op 10339) 
v_lshrrev_b32 v15, 16, v5 // (op 10339) 
v_lshrrev_b32 v16, 16, v6 // (op 10339) 
v_lshrrev_b32 v17, 16, v7 // (op 10339) 


v_pack_b32_f16 v4, v14, v15 // (op 10341) 
v_pack_b32_f16 v5, v16, v17 // (op 10341) 

v_add_u32 v0, 160, v13 // (op 10341) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10341) Store value





s_lshl_b32 s0, s2, 5 // (op 10355) 
s_add_u32 s0, s0, 160 // (op 10355) 
v_add_u32 v0, v12, s0 // (op 10355) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10355) Load value

v_accvgpr_read v14, a84 // (op 10367) 
v_accvgpr_read v15, a85 // (op 10367) 
v_accvgpr_read v16, a86 // (op 10367) 
v_accvgpr_read v17, a87 // (op 10367) 
v_mul_f32 v18, v11, v14 // (op 10367) 
v_mul_f32 v19, v11, v15 // (op 10367) 
v_mul_f32 v20, v11, v16 // (op 10367) 
v_mul_f32 v21, v11, v17 // (op 10367) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10352) 
v_lshrrev_b32 v7, 16, v4 // (op 10352) 
v_lshlrev_b32 v6, 16, v6 // (op 10352) 
v_lshlrev_b32 v7, 16, v7 // (op 10352) 
v_mul_f32 v14, v9, v6 // (op 10352) 
v_mul_f32 v15, v9, v7 // (op 10352) 
v_and_b32 v6, 65535, v5 // (op 10352) 
v_lshrrev_b32 v7, 16, v5 // (op 10352) 
v_lshlrev_b32 v6, 16, v6 // (op 10352) 
v_lshlrev_b32 v7, 16, v7 // (op 10352) 
v_mul_f32 v16, v9, v6 // (op 10352) 
v_mul_f32 v17, v9, v7 // (op 10352) 


v_add_f32 v4, v14, v18 // (op 10353) 
v_add_f32 v5, v15, v19 // (op 10353) 
v_add_f32 v6, v16, v20 // (op 10353) 
v_add_f32 v7, v17, v21 // (op 10353) 


v_lshrrev_b32 v14, 16, v4 // (op 10354) 
v_lshrrev_b32 v15, 16, v5 // (op 10354) 
v_lshrrev_b32 v16, 16, v6 // (op 10354) 
v_lshrrev_b32 v17, 16, v7 // (op 10354) 


v_pack_b32_f16 v4, v14, v15 // (op 10343) 
v_pack_b32_f16 v5, v16, v17 // (op 10343) 

s_lshl_b32 s0, s4, 5 // (op 10343) 
s_add_u32 s0, s0, 160 // (op 10343) 
v_add_u32 v0, v13, s0 // (op 10343) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10343) Store value





s_lshl_b32 s0, s2, 6 // (op 10360) 
s_add_u32 s0, s0, 160 // (op 10360) 
v_add_u32 v0, v12, s0 // (op 10360) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10360) Load value

v_accvgpr_read v14, a88 // (op 10369) 
v_accvgpr_read v15, a89 // (op 10369) 
v_accvgpr_read v16, a90 // (op 10369) 
v_accvgpr_read v17, a91 // (op 10369) 
v_mul_f32 v18, v11, v14 // (op 10369) 
v_mul_f32 v19, v11, v15 // (op 10369) 
v_mul_f32 v20, v11, v16 // (op 10369) 
v_mul_f32 v21, v11, v17 // (op 10369) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10357) 
v_lshrrev_b32 v7, 16, v4 // (op 10357) 
v_lshlrev_b32 v6, 16, v6 // (op 10357) 
v_lshlrev_b32 v7, 16, v7 // (op 10357) 
v_mul_f32 v14, v9, v6 // (op 10357) 
v_mul_f32 v15, v9, v7 // (op 10357) 
v_and_b32 v6, 65535, v5 // (op 10357) 
v_lshrrev_b32 v7, 16, v5 // (op 10357) 
v_lshlrev_b32 v6, 16, v6 // (op 10357) 
v_lshlrev_b32 v7, 16, v7 // (op 10357) 
v_mul_f32 v16, v9, v6 // (op 10357) 
v_mul_f32 v17, v9, v7 // (op 10357) 


v_add_f32 v4, v14, v18 // (op 10358) 
v_add_f32 v5, v15, v19 // (op 10358) 
v_add_f32 v6, v16, v20 // (op 10358) 
v_add_f32 v7, v17, v21 // (op 10358) 


v_lshrrev_b32 v14, 16, v4 // (op 10359) 
v_lshrrev_b32 v15, 16, v5 // (op 10359) 
v_lshrrev_b32 v16, 16, v6 // (op 10359) 
v_lshrrev_b32 v17, 16, v7 // (op 10359) 


v_pack_b32_f16 v4, v14, v15 // (op 10345) 
v_pack_b32_f16 v5, v16, v17 // (op 10345) 

s_lshl_b32 s0, s4, 6 // (op 10345) 
s_add_u32 s0, s0, 160 // (op 10345) 
v_add_u32 v0, v13, s0 // (op 10345) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10345) Store value





s_lshl_b32 s0, s2, 5 // (op 10365) 
s_mul_i32 s1, 3, s0 // (op 10365) 
s_add_u32 s1, s1, 160 // (op 10365) 

v_add_u32 v0, v12, s1 // (op 10365) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10365) Load value

v_accvgpr_read v14, a92 // (op 10371) 
v_accvgpr_read v15, a93 // (op 10371) 
v_accvgpr_read v16, a94 // (op 10371) 
v_accvgpr_read v17, a95 // (op 10371) 
v_mul_f32 v18, v11, v14 // (op 10371) 
v_mul_f32 v19, v11, v15 // (op 10371) 
v_mul_f32 v20, v11, v16 // (op 10371) 
v_mul_f32 v21, v11, v17 // (op 10371) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10362) 
v_lshrrev_b32 v7, 16, v4 // (op 10362) 
v_lshlrev_b32 v6, 16, v6 // (op 10362) 
v_lshlrev_b32 v7, 16, v7 // (op 10362) 
v_mul_f32 v14, v9, v6 // (op 10362) 
v_mul_f32 v15, v9, v7 // (op 10362) 
v_and_b32 v6, 65535, v5 // (op 10362) 
v_lshrrev_b32 v7, 16, v5 // (op 10362) 
v_lshlrev_b32 v6, 16, v6 // (op 10362) 
v_lshlrev_b32 v7, 16, v7 // (op 10362) 
v_mul_f32 v16, v9, v6 // (op 10362) 
v_mul_f32 v17, v9, v7 // (op 10362) 


v_add_f32 v4, v14, v18 // (op 10363) 
v_add_f32 v5, v15, v19 // (op 10363) 
v_add_f32 v6, v16, v20 // (op 10363) 
v_add_f32 v7, v17, v21 // (op 10363) 


v_lshrrev_b32 v14, 16, v4 // (op 10364) 
v_lshrrev_b32 v15, 16, v5 // (op 10364) 
v_lshrrev_b32 v16, 16, v6 // (op 10364) 
v_lshrrev_b32 v17, 16, v7 // (op 10364) 


v_pack_b32_f16 v4, v14, v15 // (op 10347) 
v_pack_b32_f16 v5, v16, v17 // (op 10347) 

s_lshl_b32 s0, s4, 5 // (op 10347) 
s_mul_i32 s1, 3, s0 // (op 10347) 
s_add_u32 s1, s1, 160 // (op 10347) 

v_add_u32 v0, v13, s1 // (op 10347) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10347) Store value





v_add_u32 v0, 128, v12 // (op 10263) 

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10263) Load value

v_accvgpr_read v14, a64 // (op 10250) 
v_accvgpr_read v15, a65 // (op 10250) 
v_accvgpr_read v16, a66 // (op 10250) 
v_accvgpr_read v17, a67 // (op 10250) 
v_mul_f32 v18, v11, v14 // (op 10250) 
v_mul_f32 v19, v11, v15 // (op 10250) 
v_mul_f32 v20, v11, v16 // (op 10250) 
v_mul_f32 v21, v11, v17 // (op 10250) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10262) 
v_lshrrev_b32 v7, 16, v4 // (op 10262) 
v_lshlrev_b32 v6, 16, v6 // (op 10262) 
v_lshlrev_b32 v7, 16, v7 // (op 10262) 
v_mul_f32 v14, v9, v6 // (op 10262) 
v_mul_f32 v15, v9, v7 // (op 10262) 
v_and_b32 v6, 65535, v5 // (op 10262) 
v_lshrrev_b32 v7, 16, v5 // (op 10262) 
v_lshlrev_b32 v6, 16, v6 // (op 10262) 
v_lshlrev_b32 v7, 16, v7 // (op 10262) 
v_mul_f32 v16, v9, v6 // (op 10262) 
v_mul_f32 v17, v9, v7 // (op 10262) 


v_add_f32 v4, v14, v18 // (op 10251) 
v_add_f32 v5, v15, v19 // (op 10251) 
v_add_f32 v6, v16, v20 // (op 10251) 
v_add_f32 v7, v17, v21 // (op 10251) 


v_lshrrev_b32 v14, 16, v4 // (op 10252) 
v_lshrrev_b32 v15, 16, v5 // (op 10252) 
v_lshrrev_b32 v16, 16, v6 // (op 10252) 
v_lshrrev_b32 v17, 16, v7 // (op 10252) 


v_pack_b32_f16 v4, v14, v15 // (op 10254) 
v_pack_b32_f16 v5, v16, v17 // (op 10254) 

v_add_u32 v0, 128, v13 // (op 10254) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10254) Store value





s_lshl_b32 s0, s2, 5 // (op 10268) 
s_add_u32 s0, s0, 128 // (op 10268) 
v_add_u32 v0, v12, s0 // (op 10268) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10268) Load value

v_accvgpr_read v14, a68 // (op 10280) 
v_accvgpr_read v15, a69 // (op 10280) 
v_accvgpr_read v16, a70 // (op 10280) 
v_accvgpr_read v17, a71 // (op 10280) 
v_mul_f32 v18, v11, v14 // (op 10280) 
v_mul_f32 v19, v11, v15 // (op 10280) 
v_mul_f32 v20, v11, v16 // (op 10280) 
v_mul_f32 v21, v11, v17 // (op 10280) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10265) 
v_lshrrev_b32 v7, 16, v4 // (op 10265) 
v_lshlrev_b32 v6, 16, v6 // (op 10265) 
v_lshlrev_b32 v7, 16, v7 // (op 10265) 
v_mul_f32 v14, v9, v6 // (op 10265) 
v_mul_f32 v15, v9, v7 // (op 10265) 
v_and_b32 v6, 65535, v5 // (op 10265) 
v_lshrrev_b32 v7, 16, v5 // (op 10265) 
v_lshlrev_b32 v6, 16, v6 // (op 10265) 
v_lshlrev_b32 v7, 16, v7 // (op 10265) 
v_mul_f32 v16, v9, v6 // (op 10265) 
v_mul_f32 v17, v9, v7 // (op 10265) 


v_add_f32 v4, v14, v18 // (op 10266) 
v_add_f32 v5, v15, v19 // (op 10266) 
v_add_f32 v6, v16, v20 // (op 10266) 
v_add_f32 v7, v17, v21 // (op 10266) 


v_lshrrev_b32 v14, 16, v4 // (op 10267) 
v_lshrrev_b32 v15, 16, v5 // (op 10267) 
v_lshrrev_b32 v16, 16, v6 // (op 10267) 
v_lshrrev_b32 v17, 16, v7 // (op 10267) 


v_pack_b32_f16 v4, v14, v15 // (op 10256) 
v_pack_b32_f16 v5, v16, v17 // (op 10256) 

s_lshl_b32 s0, s4, 5 // (op 10256) 
s_add_u32 s0, s0, 128 // (op 10256) 
v_add_u32 v0, v13, s0 // (op 10256) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10256) Store value





s_lshl_b32 s0, s2, 6 // (op 10273) 
s_add_u32 s0, s0, 128 // (op 10273) 
v_add_u32 v0, v12, s0 // (op 10273) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10273) Load value

v_accvgpr_read v14, a72 // (op 10282) 
v_accvgpr_read v15, a73 // (op 10282) 
v_accvgpr_read v16, a74 // (op 10282) 
v_accvgpr_read v17, a75 // (op 10282) 
v_mul_f32 v18, v11, v14 // (op 10282) 
v_mul_f32 v19, v11, v15 // (op 10282) 
v_mul_f32 v20, v11, v16 // (op 10282) 
v_mul_f32 v21, v11, v17 // (op 10282) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10270) 
v_lshrrev_b32 v7, 16, v4 // (op 10270) 
v_lshlrev_b32 v6, 16, v6 // (op 10270) 
v_lshlrev_b32 v7, 16, v7 // (op 10270) 
v_mul_f32 v14, v9, v6 // (op 10270) 
v_mul_f32 v15, v9, v7 // (op 10270) 
v_and_b32 v6, 65535, v5 // (op 10270) 
v_lshrrev_b32 v7, 16, v5 // (op 10270) 
v_lshlrev_b32 v6, 16, v6 // (op 10270) 
v_lshlrev_b32 v7, 16, v7 // (op 10270) 
v_mul_f32 v16, v9, v6 // (op 10270) 
v_mul_f32 v17, v9, v7 // (op 10270) 


v_add_f32 v4, v14, v18 // (op 10271) 
v_add_f32 v5, v15, v19 // (op 10271) 
v_add_f32 v6, v16, v20 // (op 10271) 
v_add_f32 v7, v17, v21 // (op 10271) 


v_lshrrev_b32 v14, 16, v4 // (op 10272) 
v_lshrrev_b32 v15, 16, v5 // (op 10272) 
v_lshrrev_b32 v16, 16, v6 // (op 10272) 
v_lshrrev_b32 v17, 16, v7 // (op 10272) 


v_pack_b32_f16 v4, v14, v15 // (op 10258) 
v_pack_b32_f16 v5, v16, v17 // (op 10258) 

s_lshl_b32 s0, s4, 6 // (op 10258) 
s_add_u32 s0, s0, 128 // (op 10258) 
v_add_u32 v0, v13, s0 // (op 10258) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10258) Store value





s_lshl_b32 s0, s2, 5 // (op 10278) 
s_mul_i32 s1, 3, s0 // (op 10278) 
s_add_u32 s1, s1, 128 // (op 10278) 

v_add_u32 v0, v12, s1 // (op 10278) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10278) Load value

v_accvgpr_read v14, a76 // (op 10284) 
v_accvgpr_read v15, a77 // (op 10284) 
v_accvgpr_read v16, a78 // (op 10284) 
v_accvgpr_read v17, a79 // (op 10284) 
v_mul_f32 v18, v11, v14 // (op 10284) 
v_mul_f32 v19, v11, v15 // (op 10284) 
v_mul_f32 v20, v11, v16 // (op 10284) 
v_mul_f32 v21, v11, v17 // (op 10284) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10275) 
v_lshrrev_b32 v7, 16, v4 // (op 10275) 
v_lshlrev_b32 v6, 16, v6 // (op 10275) 
v_lshlrev_b32 v7, 16, v7 // (op 10275) 
v_mul_f32 v14, v9, v6 // (op 10275) 
v_mul_f32 v15, v9, v7 // (op 10275) 
v_and_b32 v6, 65535, v5 // (op 10275) 
v_lshrrev_b32 v7, 16, v5 // (op 10275) 
v_lshlrev_b32 v6, 16, v6 // (op 10275) 
v_lshlrev_b32 v7, 16, v7 // (op 10275) 
v_mul_f32 v16, v9, v6 // (op 10275) 
v_mul_f32 v17, v9, v7 // (op 10275) 


v_add_f32 v4, v14, v18 // (op 10276) 
v_add_f32 v5, v15, v19 // (op 10276) 
v_add_f32 v6, v16, v20 // (op 10276) 
v_add_f32 v7, v17, v21 // (op 10276) 


v_lshrrev_b32 v14, 16, v4 // (op 10277) 
v_lshrrev_b32 v15, 16, v5 // (op 10277) 
v_lshrrev_b32 v16, 16, v6 // (op 10277) 
v_lshrrev_b32 v17, 16, v7 // (op 10277) 


v_pack_b32_f16 v4, v14, v15 // (op 10260) 
v_pack_b32_f16 v5, v16, v17 // (op 10260) 

s_lshl_b32 s0, s4, 5 // (op 10260) 
s_mul_i32 s1, 3, s0 // (op 10260) 
s_add_u32 s1, s1, 128 // (op 10260) 

v_add_u32 v0, v13, s1 // (op 10260) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10260) Store value





v_add_u32 v0, 96, v12 // (op 10176) 

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10176) Load value

v_accvgpr_read v14, a48 // (op 10163) 
v_accvgpr_read v15, a49 // (op 10163) 
v_accvgpr_read v16, a50 // (op 10163) 
v_accvgpr_read v17, a51 // (op 10163) 
v_mul_f32 v18, v11, v14 // (op 10163) 
v_mul_f32 v19, v11, v15 // (op 10163) 
v_mul_f32 v20, v11, v16 // (op 10163) 
v_mul_f32 v21, v11, v17 // (op 10163) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10175) 
v_lshrrev_b32 v7, 16, v4 // (op 10175) 
v_lshlrev_b32 v6, 16, v6 // (op 10175) 
v_lshlrev_b32 v7, 16, v7 // (op 10175) 
v_mul_f32 v14, v9, v6 // (op 10175) 
v_mul_f32 v15, v9, v7 // (op 10175) 
v_and_b32 v6, 65535, v5 // (op 10175) 
v_lshrrev_b32 v7, 16, v5 // (op 10175) 
v_lshlrev_b32 v6, 16, v6 // (op 10175) 
v_lshlrev_b32 v7, 16, v7 // (op 10175) 
v_mul_f32 v16, v9, v6 // (op 10175) 
v_mul_f32 v17, v9, v7 // (op 10175) 


v_add_f32 v4, v14, v18 // (op 10164) 
v_add_f32 v5, v15, v19 // (op 10164) 
v_add_f32 v6, v16, v20 // (op 10164) 
v_add_f32 v7, v17, v21 // (op 10164) 


v_lshrrev_b32 v14, 16, v4 // (op 10165) 
v_lshrrev_b32 v15, 16, v5 // (op 10165) 
v_lshrrev_b32 v16, 16, v6 // (op 10165) 
v_lshrrev_b32 v17, 16, v7 // (op 10165) 


v_pack_b32_f16 v4, v14, v15 // (op 10167) 
v_pack_b32_f16 v5, v16, v17 // (op 10167) 

v_add_u32 v0, 96, v13 // (op 10167) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10167) Store value





s_lshl_b32 s0, s2, 5 // (op 10181) 
s_add_u32 s0, s0, 96 // (op 10181) 
v_add_u32 v0, v12, s0 // (op 10181) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10181) Load value

v_accvgpr_read v14, a52 // (op 10193) 
v_accvgpr_read v15, a53 // (op 10193) 
v_accvgpr_read v16, a54 // (op 10193) 
v_accvgpr_read v17, a55 // (op 10193) 
v_mul_f32 v18, v11, v14 // (op 10193) 
v_mul_f32 v19, v11, v15 // (op 10193) 
v_mul_f32 v20, v11, v16 // (op 10193) 
v_mul_f32 v21, v11, v17 // (op 10193) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10178) 
v_lshrrev_b32 v7, 16, v4 // (op 10178) 
v_lshlrev_b32 v6, 16, v6 // (op 10178) 
v_lshlrev_b32 v7, 16, v7 // (op 10178) 
v_mul_f32 v14, v9, v6 // (op 10178) 
v_mul_f32 v15, v9, v7 // (op 10178) 
v_and_b32 v6, 65535, v5 // (op 10178) 
v_lshrrev_b32 v7, 16, v5 // (op 10178) 
v_lshlrev_b32 v6, 16, v6 // (op 10178) 
v_lshlrev_b32 v7, 16, v7 // (op 10178) 
v_mul_f32 v16, v9, v6 // (op 10178) 
v_mul_f32 v17, v9, v7 // (op 10178) 


v_add_f32 v4, v14, v18 // (op 10179) 
v_add_f32 v5, v15, v19 // (op 10179) 
v_add_f32 v6, v16, v20 // (op 10179) 
v_add_f32 v7, v17, v21 // (op 10179) 


v_lshrrev_b32 v14, 16, v4 // (op 10180) 
v_lshrrev_b32 v15, 16, v5 // (op 10180) 
v_lshrrev_b32 v16, 16, v6 // (op 10180) 
v_lshrrev_b32 v17, 16, v7 // (op 10180) 


v_pack_b32_f16 v4, v14, v15 // (op 10169) 
v_pack_b32_f16 v5, v16, v17 // (op 10169) 

s_lshl_b32 s0, s4, 5 // (op 10169) 
s_add_u32 s0, s0, 96 // (op 10169) 
v_add_u32 v0, v13, s0 // (op 10169) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10169) Store value





s_lshl_b32 s0, s2, 6 // (op 10186) 
s_add_u32 s0, s0, 96 // (op 10186) 
v_add_u32 v0, v12, s0 // (op 10186) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10186) Load value

v_accvgpr_read v14, a56 // (op 10195) 
v_accvgpr_read v15, a57 // (op 10195) 
v_accvgpr_read v16, a58 // (op 10195) 
v_accvgpr_read v17, a59 // (op 10195) 
v_mul_f32 v18, v11, v14 // (op 10195) 
v_mul_f32 v19, v11, v15 // (op 10195) 
v_mul_f32 v20, v11, v16 // (op 10195) 
v_mul_f32 v21, v11, v17 // (op 10195) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10183) 
v_lshrrev_b32 v7, 16, v4 // (op 10183) 
v_lshlrev_b32 v6, 16, v6 // (op 10183) 
v_lshlrev_b32 v7, 16, v7 // (op 10183) 
v_mul_f32 v14, v9, v6 // (op 10183) 
v_mul_f32 v15, v9, v7 // (op 10183) 
v_and_b32 v6, 65535, v5 // (op 10183) 
v_lshrrev_b32 v7, 16, v5 // (op 10183) 
v_lshlrev_b32 v6, 16, v6 // (op 10183) 
v_lshlrev_b32 v7, 16, v7 // (op 10183) 
v_mul_f32 v16, v9, v6 // (op 10183) 
v_mul_f32 v17, v9, v7 // (op 10183) 


v_add_f32 v4, v14, v18 // (op 10184) 
v_add_f32 v5, v15, v19 // (op 10184) 
v_add_f32 v6, v16, v20 // (op 10184) 
v_add_f32 v7, v17, v21 // (op 10184) 


v_lshrrev_b32 v14, 16, v4 // (op 10185) 
v_lshrrev_b32 v15, 16, v5 // (op 10185) 
v_lshrrev_b32 v16, 16, v6 // (op 10185) 
v_lshrrev_b32 v17, 16, v7 // (op 10185) 


v_pack_b32_f16 v4, v14, v15 // (op 10171) 
v_pack_b32_f16 v5, v16, v17 // (op 10171) 

s_lshl_b32 s0, s4, 6 // (op 10171) 
s_add_u32 s0, s0, 96 // (op 10171) 
v_add_u32 v0, v13, s0 // (op 10171) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10171) Store value





s_lshl_b32 s0, s2, 5 // (op 10191) 
s_mul_i32 s1, 3, s0 // (op 10191) 
s_add_u32 s1, s1, 96 // (op 10191) 

v_add_u32 v0, v12, s1 // (op 10191) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10191) Load value

v_accvgpr_read v14, a60 // (op 10197) 
v_accvgpr_read v15, a61 // (op 10197) 
v_accvgpr_read v16, a62 // (op 10197) 
v_accvgpr_read v17, a63 // (op 10197) 
v_mul_f32 v18, v11, v14 // (op 10197) 
v_mul_f32 v19, v11, v15 // (op 10197) 
v_mul_f32 v20, v11, v16 // (op 10197) 
v_mul_f32 v21, v11, v17 // (op 10197) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10188) 
v_lshrrev_b32 v7, 16, v4 // (op 10188) 
v_lshlrev_b32 v6, 16, v6 // (op 10188) 
v_lshlrev_b32 v7, 16, v7 // (op 10188) 
v_mul_f32 v14, v9, v6 // (op 10188) 
v_mul_f32 v15, v9, v7 // (op 10188) 
v_and_b32 v6, 65535, v5 // (op 10188) 
v_lshrrev_b32 v7, 16, v5 // (op 10188) 
v_lshlrev_b32 v6, 16, v6 // (op 10188) 
v_lshlrev_b32 v7, 16, v7 // (op 10188) 
v_mul_f32 v16, v9, v6 // (op 10188) 
v_mul_f32 v17, v9, v7 // (op 10188) 


v_add_f32 v4, v14, v18 // (op 10189) 
v_add_f32 v5, v15, v19 // (op 10189) 
v_add_f32 v6, v16, v20 // (op 10189) 
v_add_f32 v7, v17, v21 // (op 10189) 


v_lshrrev_b32 v14, 16, v4 // (op 10190) 
v_lshrrev_b32 v15, 16, v5 // (op 10190) 
v_lshrrev_b32 v16, 16, v6 // (op 10190) 
v_lshrrev_b32 v17, 16, v7 // (op 10190) 


v_pack_b32_f16 v4, v14, v15 // (op 10173) 
v_pack_b32_f16 v5, v16, v17 // (op 10173) 

s_lshl_b32 s0, s4, 5 // (op 10173) 
s_mul_i32 s1, 3, s0 // (op 10173) 
s_add_u32 s1, s1, 96 // (op 10173) 

v_add_u32 v0, v13, s1 // (op 10173) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10173) Store value





v_add_u32 v0, 64, v12 // (op 10089) 

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10089) Load value

v_accvgpr_read v14, a32 // (op 10076) 
v_accvgpr_read v15, a33 // (op 10076) 
v_accvgpr_read v16, a34 // (op 10076) 
v_accvgpr_read v17, a35 // (op 10076) 
v_mul_f32 v18, v11, v14 // (op 10076) 
v_mul_f32 v19, v11, v15 // (op 10076) 
v_mul_f32 v20, v11, v16 // (op 10076) 
v_mul_f32 v21, v11, v17 // (op 10076) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10088) 
v_lshrrev_b32 v7, 16, v4 // (op 10088) 
v_lshlrev_b32 v6, 16, v6 // (op 10088) 
v_lshlrev_b32 v7, 16, v7 // (op 10088) 
v_mul_f32 v14, v9, v6 // (op 10088) 
v_mul_f32 v15, v9, v7 // (op 10088) 
v_and_b32 v6, 65535, v5 // (op 10088) 
v_lshrrev_b32 v7, 16, v5 // (op 10088) 
v_lshlrev_b32 v6, 16, v6 // (op 10088) 
v_lshlrev_b32 v7, 16, v7 // (op 10088) 
v_mul_f32 v16, v9, v6 // (op 10088) 
v_mul_f32 v17, v9, v7 // (op 10088) 


v_add_f32 v4, v14, v18 // (op 10077) 
v_add_f32 v5, v15, v19 // (op 10077) 
v_add_f32 v6, v16, v20 // (op 10077) 
v_add_f32 v7, v17, v21 // (op 10077) 


v_lshrrev_b32 v14, 16, v4 // (op 10078) 
v_lshrrev_b32 v15, 16, v5 // (op 10078) 
v_lshrrev_b32 v16, 16, v6 // (op 10078) 
v_lshrrev_b32 v17, 16, v7 // (op 10078) 


v_pack_b32_f16 v4, v14, v15 // (op 10080) 
v_pack_b32_f16 v5, v16, v17 // (op 10080) 

v_add_u32 v0, 64, v13 // (op 10080) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10080) Store value





s_lshl_b32 s0, s2, 5 // (op 10094) 
s_add_u32 s0, s0, 64 // (op 10094) 
v_add_u32 v0, v12, s0 // (op 10094) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10094) Load value

v_accvgpr_read v14, a36 // (op 10106) 
v_accvgpr_read v15, a37 // (op 10106) 
v_accvgpr_read v16, a38 // (op 10106) 
v_accvgpr_read v17, a39 // (op 10106) 
v_mul_f32 v18, v11, v14 // (op 10106) 
v_mul_f32 v19, v11, v15 // (op 10106) 
v_mul_f32 v20, v11, v16 // (op 10106) 
v_mul_f32 v21, v11, v17 // (op 10106) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10091) 
v_lshrrev_b32 v7, 16, v4 // (op 10091) 
v_lshlrev_b32 v6, 16, v6 // (op 10091) 
v_lshlrev_b32 v7, 16, v7 // (op 10091) 
v_mul_f32 v14, v9, v6 // (op 10091) 
v_mul_f32 v15, v9, v7 // (op 10091) 
v_and_b32 v6, 65535, v5 // (op 10091) 
v_lshrrev_b32 v7, 16, v5 // (op 10091) 
v_lshlrev_b32 v6, 16, v6 // (op 10091) 
v_lshlrev_b32 v7, 16, v7 // (op 10091) 
v_mul_f32 v16, v9, v6 // (op 10091) 
v_mul_f32 v17, v9, v7 // (op 10091) 


v_add_f32 v4, v14, v18 // (op 10092) 
v_add_f32 v5, v15, v19 // (op 10092) 
v_add_f32 v6, v16, v20 // (op 10092) 
v_add_f32 v7, v17, v21 // (op 10092) 


v_lshrrev_b32 v14, 16, v4 // (op 10093) 
v_lshrrev_b32 v15, 16, v5 // (op 10093) 
v_lshrrev_b32 v16, 16, v6 // (op 10093) 
v_lshrrev_b32 v17, 16, v7 // (op 10093) 


v_pack_b32_f16 v4, v14, v15 // (op 10082) 
v_pack_b32_f16 v5, v16, v17 // (op 10082) 

s_lshl_b32 s0, s4, 5 // (op 10082) 
s_add_u32 s0, s0, 64 // (op 10082) 
v_add_u32 v0, v13, s0 // (op 10082) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10082) Store value





s_lshl_b32 s0, s2, 6 // (op 10099) 
s_add_u32 s0, s0, 64 // (op 10099) 
v_add_u32 v0, v12, s0 // (op 10099) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10099) Load value

v_accvgpr_read v14, a40 // (op 10108) 
v_accvgpr_read v15, a41 // (op 10108) 
v_accvgpr_read v16, a42 // (op 10108) 
v_accvgpr_read v17, a43 // (op 10108) 
v_mul_f32 v18, v11, v14 // (op 10108) 
v_mul_f32 v19, v11, v15 // (op 10108) 
v_mul_f32 v20, v11, v16 // (op 10108) 
v_mul_f32 v21, v11, v17 // (op 10108) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10096) 
v_lshrrev_b32 v7, 16, v4 // (op 10096) 
v_lshlrev_b32 v6, 16, v6 // (op 10096) 
v_lshlrev_b32 v7, 16, v7 // (op 10096) 
v_mul_f32 v14, v9, v6 // (op 10096) 
v_mul_f32 v15, v9, v7 // (op 10096) 
v_and_b32 v6, 65535, v5 // (op 10096) 
v_lshrrev_b32 v7, 16, v5 // (op 10096) 
v_lshlrev_b32 v6, 16, v6 // (op 10096) 
v_lshlrev_b32 v7, 16, v7 // (op 10096) 
v_mul_f32 v16, v9, v6 // (op 10096) 
v_mul_f32 v17, v9, v7 // (op 10096) 


v_add_f32 v4, v14, v18 // (op 10097) 
v_add_f32 v5, v15, v19 // (op 10097) 
v_add_f32 v6, v16, v20 // (op 10097) 
v_add_f32 v7, v17, v21 // (op 10097) 


v_lshrrev_b32 v14, 16, v4 // (op 10098) 
v_lshrrev_b32 v15, 16, v5 // (op 10098) 
v_lshrrev_b32 v16, 16, v6 // (op 10098) 
v_lshrrev_b32 v17, 16, v7 // (op 10098) 


v_pack_b32_f16 v4, v14, v15 // (op 10084) 
v_pack_b32_f16 v5, v16, v17 // (op 10084) 

s_lshl_b32 s0, s4, 6 // (op 10084) 
s_add_u32 s0, s0, 64 // (op 10084) 
v_add_u32 v0, v13, s0 // (op 10084) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10084) Store value





s_lshl_b32 s0, s2, 5 // (op 10104) 
s_mul_i32 s1, 3, s0 // (op 10104) 
s_add_u32 s1, s1, 64 // (op 10104) 

v_add_u32 v0, v12, s1 // (op 10104) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10104) Load value

v_accvgpr_read v14, a44 // (op 10110) 
v_accvgpr_read v15, a45 // (op 10110) 
v_accvgpr_read v16, a46 // (op 10110) 
v_accvgpr_read v17, a47 // (op 10110) 
v_mul_f32 v18, v11, v14 // (op 10110) 
v_mul_f32 v19, v11, v15 // (op 10110) 
v_mul_f32 v20, v11, v16 // (op 10110) 
v_mul_f32 v21, v11, v17 // (op 10110) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10101) 
v_lshrrev_b32 v7, 16, v4 // (op 10101) 
v_lshlrev_b32 v6, 16, v6 // (op 10101) 
v_lshlrev_b32 v7, 16, v7 // (op 10101) 
v_mul_f32 v14, v9, v6 // (op 10101) 
v_mul_f32 v15, v9, v7 // (op 10101) 
v_and_b32 v6, 65535, v5 // (op 10101) 
v_lshrrev_b32 v7, 16, v5 // (op 10101) 
v_lshlrev_b32 v6, 16, v6 // (op 10101) 
v_lshlrev_b32 v7, 16, v7 // (op 10101) 
v_mul_f32 v16, v9, v6 // (op 10101) 
v_mul_f32 v17, v9, v7 // (op 10101) 


v_add_f32 v4, v14, v18 // (op 10102) 
v_add_f32 v5, v15, v19 // (op 10102) 
v_add_f32 v6, v16, v20 // (op 10102) 
v_add_f32 v7, v17, v21 // (op 10102) 


v_lshrrev_b32 v14, 16, v4 // (op 10103) 
v_lshrrev_b32 v15, 16, v5 // (op 10103) 
v_lshrrev_b32 v16, 16, v6 // (op 10103) 
v_lshrrev_b32 v17, 16, v7 // (op 10103) 


v_pack_b32_f16 v4, v14, v15 // (op 10086) 
v_pack_b32_f16 v5, v16, v17 // (op 10086) 

s_lshl_b32 s0, s4, 5 // (op 10086) 
s_mul_i32 s1, 3, s0 // (op 10086) 
s_add_u32 s1, s1, 64 // (op 10086) 

v_add_u32 v0, v13, s1 // (op 10086) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 10086) Store value





v_add_u32 v0, 32, v12 // (op 10002) 

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10002) Load value

v_accvgpr_read v14, a16 // (op 9989) 
v_accvgpr_read v15, a17 // (op 9989) 
v_accvgpr_read v16, a18 // (op 9989) 
v_accvgpr_read v17, a19 // (op 9989) 
v_mul_f32 v18, v11, v14 // (op 9989) 
v_mul_f32 v19, v11, v15 // (op 9989) 
v_mul_f32 v20, v11, v16 // (op 9989) 
v_mul_f32 v21, v11, v17 // (op 9989) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10001) 
v_lshrrev_b32 v7, 16, v4 // (op 10001) 
v_lshlrev_b32 v6, 16, v6 // (op 10001) 
v_lshlrev_b32 v7, 16, v7 // (op 10001) 
v_mul_f32 v14, v9, v6 // (op 10001) 
v_mul_f32 v15, v9, v7 // (op 10001) 
v_and_b32 v6, 65535, v5 // (op 10001) 
v_lshrrev_b32 v7, 16, v5 // (op 10001) 
v_lshlrev_b32 v6, 16, v6 // (op 10001) 
v_lshlrev_b32 v7, 16, v7 // (op 10001) 
v_mul_f32 v16, v9, v6 // (op 10001) 
v_mul_f32 v17, v9, v7 // (op 10001) 


v_add_f32 v4, v14, v18 // (op 9990) 
v_add_f32 v5, v15, v19 // (op 9990) 
v_add_f32 v6, v16, v20 // (op 9990) 
v_add_f32 v7, v17, v21 // (op 9990) 


v_lshrrev_b32 v14, 16, v4 // (op 9991) 
v_lshrrev_b32 v15, 16, v5 // (op 9991) 
v_lshrrev_b32 v16, 16, v6 // (op 9991) 
v_lshrrev_b32 v17, 16, v7 // (op 9991) 


v_pack_b32_f16 v4, v14, v15 // (op 9993) 
v_pack_b32_f16 v5, v16, v17 // (op 9993) 

v_add_u32 v0, 32, v13 // (op 9993) 

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 9993) Store value





s_lshl_b32 s0, s2, 5 // (op 10007) 
s_add_u32 s0, s0, 32 // (op 10007) 
v_add_u32 v0, v12, s0 // (op 10007) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10007) Load value

v_accvgpr_read v14, a20 // (op 10019) 
v_accvgpr_read v15, a21 // (op 10019) 
v_accvgpr_read v16, a22 // (op 10019) 
v_accvgpr_read v17, a23 // (op 10019) 
v_mul_f32 v18, v11, v14 // (op 10019) 
v_mul_f32 v19, v11, v15 // (op 10019) 
v_mul_f32 v20, v11, v16 // (op 10019) 
v_mul_f32 v21, v11, v17 // (op 10019) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10004) 
v_lshrrev_b32 v7, 16, v4 // (op 10004) 
v_lshlrev_b32 v6, 16, v6 // (op 10004) 
v_lshlrev_b32 v7, 16, v7 // (op 10004) 
v_mul_f32 v14, v9, v6 // (op 10004) 
v_mul_f32 v15, v9, v7 // (op 10004) 
v_and_b32 v6, 65535, v5 // (op 10004) 
v_lshrrev_b32 v7, 16, v5 // (op 10004) 
v_lshlrev_b32 v6, 16, v6 // (op 10004) 
v_lshlrev_b32 v7, 16, v7 // (op 10004) 
v_mul_f32 v16, v9, v6 // (op 10004) 
v_mul_f32 v17, v9, v7 // (op 10004) 


v_add_f32 v4, v14, v18 // (op 10005) 
v_add_f32 v5, v15, v19 // (op 10005) 
v_add_f32 v6, v16, v20 // (op 10005) 
v_add_f32 v7, v17, v21 // (op 10005) 


v_lshrrev_b32 v14, 16, v4 // (op 10006) 
v_lshrrev_b32 v15, 16, v5 // (op 10006) 
v_lshrrev_b32 v16, 16, v6 // (op 10006) 
v_lshrrev_b32 v17, 16, v7 // (op 10006) 


v_pack_b32_f16 v4, v14, v15 // (op 9995) 
v_pack_b32_f16 v5, v16, v17 // (op 9995) 

s_lshl_b32 s0, s4, 5 // (op 9995) 
s_add_u32 s0, s0, 32 // (op 9995) 
v_add_u32 v0, v13, s0 // (op 9995) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 9995) Store value





s_lshl_b32 s0, s2, 6 // (op 10012) 
s_add_u32 s0, s0, 32 // (op 10012) 
v_add_u32 v0, v12, s0 // (op 10012) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10012) Load value

v_accvgpr_read v14, a24 // (op 10021) 
v_accvgpr_read v15, a25 // (op 10021) 
v_accvgpr_read v16, a26 // (op 10021) 
v_accvgpr_read v17, a27 // (op 10021) 
v_mul_f32 v18, v11, v14 // (op 10021) 
v_mul_f32 v19, v11, v15 // (op 10021) 
v_mul_f32 v20, v11, v16 // (op 10021) 
v_mul_f32 v21, v11, v17 // (op 10021) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10009) 
v_lshrrev_b32 v7, 16, v4 // (op 10009) 
v_lshlrev_b32 v6, 16, v6 // (op 10009) 
v_lshlrev_b32 v7, 16, v7 // (op 10009) 
v_mul_f32 v14, v9, v6 // (op 10009) 
v_mul_f32 v15, v9, v7 // (op 10009) 
v_and_b32 v6, 65535, v5 // (op 10009) 
v_lshrrev_b32 v7, 16, v5 // (op 10009) 
v_lshlrev_b32 v6, 16, v6 // (op 10009) 
v_lshlrev_b32 v7, 16, v7 // (op 10009) 
v_mul_f32 v16, v9, v6 // (op 10009) 
v_mul_f32 v17, v9, v7 // (op 10009) 


v_add_f32 v4, v14, v18 // (op 10010) 
v_add_f32 v5, v15, v19 // (op 10010) 
v_add_f32 v6, v16, v20 // (op 10010) 
v_add_f32 v7, v17, v21 // (op 10010) 


v_lshrrev_b32 v14, 16, v4 // (op 10011) 
v_lshrrev_b32 v15, 16, v5 // (op 10011) 
v_lshrrev_b32 v16, 16, v6 // (op 10011) 
v_lshrrev_b32 v17, 16, v7 // (op 10011) 


v_pack_b32_f16 v4, v14, v15 // (op 9997) 
v_pack_b32_f16 v5, v16, v17 // (op 9997) 

s_lshl_b32 s0, s4, 6 // (op 9997) 
s_add_u32 s0, s0, 32 // (op 9997) 
v_add_u32 v0, v13, s0 // (op 9997) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 9997) Store value





s_lshl_b32 s0, s2, 5 // (op 10017) 
s_mul_i32 s1, 3, s0 // (op 10017) 
s_add_u32 s1, s1, 32 // (op 10017) 

v_add_u32 v0, v12, s1 // (op 10017) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 10017) Load value

v_accvgpr_read v14, a28 // (op 10023) 
v_accvgpr_read v15, a29 // (op 10023) 
v_accvgpr_read v16, a30 // (op 10023) 
v_accvgpr_read v17, a31 // (op 10023) 
v_mul_f32 v18, v11, v14 // (op 10023) 
v_mul_f32 v19, v11, v15 // (op 10023) 
v_mul_f32 v20, v11, v16 // (op 10023) 
v_mul_f32 v21, v11, v17 // (op 10023) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 10014) 
v_lshrrev_b32 v7, 16, v4 // (op 10014) 
v_lshlrev_b32 v6, 16, v6 // (op 10014) 
v_lshlrev_b32 v7, 16, v7 // (op 10014) 
v_mul_f32 v14, v9, v6 // (op 10014) 
v_mul_f32 v15, v9, v7 // (op 10014) 
v_and_b32 v6, 65535, v5 // (op 10014) 
v_lshrrev_b32 v7, 16, v5 // (op 10014) 
v_lshlrev_b32 v6, 16, v6 // (op 10014) 
v_lshlrev_b32 v7, 16, v7 // (op 10014) 
v_mul_f32 v16, v9, v6 // (op 10014) 
v_mul_f32 v17, v9, v7 // (op 10014) 


v_add_f32 v4, v14, v18 // (op 10015) 
v_add_f32 v5, v15, v19 // (op 10015) 
v_add_f32 v6, v16, v20 // (op 10015) 
v_add_f32 v7, v17, v21 // (op 10015) 


v_lshrrev_b32 v14, 16, v4 // (op 10016) 
v_lshrrev_b32 v15, 16, v5 // (op 10016) 
v_lshrrev_b32 v16, 16, v6 // (op 10016) 
v_lshrrev_b32 v17, 16, v7 // (op 10016) 


v_pack_b32_f16 v4, v14, v15 // (op 9999) 
v_pack_b32_f16 v5, v16, v17 // (op 9999) 

s_lshl_b32 s0, s4, 5 // (op 9999) 
s_mul_i32 s1, 3, s0 // (op 9999) 
s_add_u32 s1, s1, 32 // (op 9999) 

v_add_u32 v0, v13, s1 // (op 9999) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 9999) Store value





v_mov_b32 v0, v12 // (op 129) reg expression

buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 129) Load value

v_accvgpr_read v14, a0 // (op 125) 
v_accvgpr_read v15, a1 // (op 125) 
v_accvgpr_read v16, a2 // (op 125) 
v_accvgpr_read v17, a3 // (op 125) 
v_mul_f32 v18, v11, v14 // (op 125) 
v_mul_f32 v19, v11, v15 // (op 125) 
v_mul_f32 v20, v11, v16 // (op 125) 
v_mul_f32 v21, v11, v17 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 130) 
v_lshrrev_b32 v7, 16, v4 // (op 130) 
v_lshlrev_b32 v6, 16, v6 // (op 130) 
v_lshlrev_b32 v7, 16, v7 // (op 130) 
v_mul_f32 v14, v9, v6 // (op 130) 
v_mul_f32 v15, v9, v7 // (op 130) 
v_and_b32 v6, 65535, v5 // (op 130) 
v_lshrrev_b32 v7, 16, v5 // (op 130) 
v_lshlrev_b32 v6, 16, v6 // (op 130) 
v_lshlrev_b32 v7, 16, v7 // (op 130) 
v_mul_f32 v16, v9, v6 // (op 130) 
v_mul_f32 v17, v9, v7 // (op 130) 


v_add_f32 v4, v14, v18 // (op 126) 
v_add_f32 v5, v15, v19 // (op 126) 
v_add_f32 v6, v16, v20 // (op 126) 
v_add_f32 v7, v17, v21 // (op 126) 


v_lshrrev_b32 v14, 16, v4 // (op 127) 
v_lshrrev_b32 v15, 16, v5 // (op 127) 
v_lshrrev_b32 v16, 16, v6 // (op 127) 
v_lshrrev_b32 v17, 16, v7 // (op 127) 


v_pack_b32_f16 v4, v14, v15 // (op 128) 
v_pack_b32_f16 v5, v16, v17 // (op 128) 

v_mov_b32 v0, v13 // (op 128) reg expression

buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 9910) 
v_add_u32 v0, v12, s0 // (op 9910) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 9910) Load value

v_accvgpr_read v14, a4 // (op 9906) 
v_accvgpr_read v15, a5 // (op 9906) 
v_accvgpr_read v16, a6 // (op 9906) 
v_accvgpr_read v17, a7 // (op 9906) 
v_mul_f32 v18, v11, v14 // (op 9906) 
v_mul_f32 v19, v11, v15 // (op 9906) 
v_mul_f32 v20, v11, v16 // (op 9906) 
v_mul_f32 v21, v11, v17 // (op 9906) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 9911) 
v_lshrrev_b32 v7, 16, v4 // (op 9911) 
v_lshlrev_b32 v6, 16, v6 // (op 9911) 
v_lshlrev_b32 v7, 16, v7 // (op 9911) 
v_mul_f32 v14, v9, v6 // (op 9911) 
v_mul_f32 v15, v9, v7 // (op 9911) 
v_and_b32 v6, 65535, v5 // (op 9911) 
v_lshrrev_b32 v7, 16, v5 // (op 9911) 
v_lshlrev_b32 v6, 16, v6 // (op 9911) 
v_lshlrev_b32 v7, 16, v7 // (op 9911) 
v_mul_f32 v16, v9, v6 // (op 9911) 
v_mul_f32 v17, v9, v7 // (op 9911) 


v_add_f32 v4, v14, v18 // (op 9907) 
v_add_f32 v5, v15, v19 // (op 9907) 
v_add_f32 v6, v16, v20 // (op 9907) 
v_add_f32 v7, v17, v21 // (op 9907) 


v_lshrrev_b32 v14, 16, v4 // (op 9908) 
v_lshrrev_b32 v15, 16, v5 // (op 9908) 
v_lshrrev_b32 v16, 16, v6 // (op 9908) 
v_lshrrev_b32 v17, 16, v7 // (op 9908) 


v_pack_b32_f16 v4, v14, v15 // (op 9909) 
v_pack_b32_f16 v5, v16, v17 // (op 9909) 

s_lshl_b32 s0, s4, 5 // (op 9909) 
v_add_u32 v0, v13, s0 // (op 9909) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 9909) Store value





s_lshl_b32 s0, s2, 6 // (op 9924) 
v_add_u32 v0, v12, s0 // (op 9924) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 9924) Load value

v_accvgpr_read v14, a8 // (op 9920) 
v_accvgpr_read v15, a9 // (op 9920) 
v_accvgpr_read v16, a10 // (op 9920) 
v_accvgpr_read v17, a11 // (op 9920) 
v_mul_f32 v18, v11, v14 // (op 9920) 
v_mul_f32 v19, v11, v15 // (op 9920) 
v_mul_f32 v20, v11, v16 // (op 9920) 
v_mul_f32 v21, v11, v17 // (op 9920) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 9925) 
v_lshrrev_b32 v7, 16, v4 // (op 9925) 
v_lshlrev_b32 v6, 16, v6 // (op 9925) 
v_lshlrev_b32 v7, 16, v7 // (op 9925) 
v_mul_f32 v14, v9, v6 // (op 9925) 
v_mul_f32 v15, v9, v7 // (op 9925) 
v_and_b32 v6, 65535, v5 // (op 9925) 
v_lshrrev_b32 v7, 16, v5 // (op 9925) 
v_lshlrev_b32 v6, 16, v6 // (op 9925) 
v_lshlrev_b32 v7, 16, v7 // (op 9925) 
v_mul_f32 v16, v9, v6 // (op 9925) 
v_mul_f32 v17, v9, v7 // (op 9925) 


v_add_f32 v4, v14, v18 // (op 9921) 
v_add_f32 v5, v15, v19 // (op 9921) 
v_add_f32 v6, v16, v20 // (op 9921) 
v_add_f32 v7, v17, v21 // (op 9921) 


v_lshrrev_b32 v14, 16, v4 // (op 9922) 
v_lshrrev_b32 v15, 16, v5 // (op 9922) 
v_lshrrev_b32 v16, 16, v6 // (op 9922) 
v_lshrrev_b32 v17, 16, v7 // (op 9922) 


v_pack_b32_f16 v4, v14, v15 // (op 9923) 
v_pack_b32_f16 v5, v16, v17 // (op 9923) 

s_lshl_b32 s0, s4, 6 // (op 9923) 
v_add_u32 v0, v13, s0 // (op 9923) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 9923) Store value





s_lshl_b32 s0, s2, 5 // (op 9938) 
s_mul_i32 s1, 3, s0 // (op 9938) 

v_add_u32 v0, v12, s1 // (op 9938) 


buffer_load_dwordx2 v[4:5], v0, s[40:43], 0 offen // (op 9938) Load value

v_accvgpr_read v14, a12 // (op 9934) 
v_accvgpr_read v15, a13 // (op 9934) 
v_accvgpr_read v16, a14 // (op 9934) 
v_accvgpr_read v17, a15 // (op 9934) 
v_mul_f32 v18, v11, v14 // (op 9934) 
v_mul_f32 v19, v11, v15 // (op 9934) 
v_mul_f32 v20, v11, v16 // (op 9934) 
v_mul_f32 v21, v11, v17 // (op 9934) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 9939) 
v_lshrrev_b32 v7, 16, v4 // (op 9939) 
v_lshlrev_b32 v6, 16, v6 // (op 9939) 
v_lshlrev_b32 v7, 16, v7 // (op 9939) 
v_mul_f32 v14, v9, v6 // (op 9939) 
v_mul_f32 v15, v9, v7 // (op 9939) 
v_and_b32 v6, 65535, v5 // (op 9939) 
v_lshrrev_b32 v7, 16, v5 // (op 9939) 
v_lshlrev_b32 v6, 16, v6 // (op 9939) 
v_lshlrev_b32 v7, 16, v7 // (op 9939) 
v_mul_f32 v16, v9, v6 // (op 9939) 
v_mul_f32 v17, v9, v7 // (op 9939) 


v_add_f32 v4, v14, v18 // (op 9935) 
v_add_f32 v5, v15, v19 // (op 9935) 
v_add_f32 v6, v16, v20 // (op 9935) 
v_add_f32 v7, v17, v21 // (op 9935) 


v_lshrrev_b32 v14, 16, v4 // (op 9936) 
v_lshrrev_b32 v15, 16, v5 // (op 9936) 
v_lshrrev_b32 v16, 16, v6 // (op 9936) 
v_lshrrev_b32 v17, 16, v7 // (op 9936) 


v_pack_b32_f16 v4, v14, v15 // (op 9937) 
v_pack_b32_f16 v5, v16, v17 // (op 9937) 

s_lshl_b32 s0, s4, 5 // (op 9937) 
s_mul_i32 s1, 3, s0 // (op 9937) 

v_add_u32 v0, v13, s1 // (op 9937) 


buffer_store_dwordx2 v[4:5], v0, s[44:47], 0 offen // (op 9937) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_ConditionalBottom_0_beta_116:








































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4
.LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_end:
.size RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4, .LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_end-RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4
  .amdhsa_next_free_vgpr 264
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 98304 // lds bytes
  .amdhsa_accum_offset 136
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
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DataTypeA: F4
    DataTypeB: F4
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
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
            { type: int64, semantic: SizeFree0 },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum } ]
    macrotile: [256, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 4]
  StaggerU: 0
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4
    .symbol: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 98304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 66
    .vgpr_count: 134
    .agpr_count: 128
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 48
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 52
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

