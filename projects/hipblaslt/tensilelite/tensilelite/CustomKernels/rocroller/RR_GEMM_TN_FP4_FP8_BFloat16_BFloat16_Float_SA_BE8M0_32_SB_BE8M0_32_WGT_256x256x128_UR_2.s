// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2,@function
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_exec_begin:


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


v_and_b32 v0, 4294967168, v1 // (op 27128) 
v_and_b32 v3, 15, v1 // (op 27128) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 27128) 


v_and_b32 v0, 63, v1 // (op 27128) 
v_lshrrev_b32 v3, 4, v0 // (op 27128) 

v_lshlrev_b32 v0, 5, v3 // (op 27128) 

v_add_lshl_u32 v3, v4, v0, 2 // (op 27128) 


v_lshrrev_b32 v0, 3, v3 // (op 27128) 

v_lshrrev_b32 v3, 6, v1 // (op 27185) 
v_and_b32 v4, 1, v3 // (op 27185) 

v_and_b32 v3, 15, v1 // (op 27185) 
v_lshl_add_u32 v5, v4, 7, v3 // (op 27185) 


v_and_b32 v3, 48, v1 // (op 27185) 
v_lshl_add_u32 v4, v5, 7, v3 // (op 27185) 


v_and_b32 v3, 4294967168, v1 // (op 27252) 
v_and_b32 v5, 15, v1 // (op 27252) 
v_add_lshl_u32 v6, v3, v5, 7 // (op 27252) 


v_and_b32 v3, 63, v1 // (op 27252) 
v_lshrrev_b32 v5, 4, v3 // (op 27252) 

v_lshlrev_b32 v3, 5, v5 // (op 27252) 

v_add_lshl_u32 v5, v6, v3, 2 // (op 27252) 


v_lshrrev_b32 v3, 3, v5 // (op 27252) 

v_lshrrev_b32 v5, 6, v1 // (op 27296) 
v_and_b32 v6, 1, v5 // (op 27296) 

v_and_b32 v5, 15, v1 // (op 27296) 
v_lshl_add_u32 v7, v6, 7, v5 // (op 27296) 


v_and_b32 v5, 48, v1 // (op 27296) 
v_lshl_add_u32 v6, v7, 7, v5 // (op 27296) 


v_and_b32 v5, 4294967168, v1 // (op 27067) 
v_and_b32 v7, 63, v1 // (op 27067) 
v_lshrrev_b32 v8, 4, v7 // (op 27067) 
v_lshlrev_b32 v9, 2, v8 // (op 27067) 

v_add_u32 v8, v5, v9 // (op 27067) 


v_lshl_add_u32 v5, s16, 8, v8 // (op 27067) 

v_lshrrev_b32 v8, 2, v7 // (op 27067) 

v_and_b32 v7, 3, v8 // (op 27067) 

v_and_b32 v8, 3, v1 // (op 27067) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 27067) 


v_lshrrev_b32 v7, 6, v1 // (op 27067) 
v_and_b32 v8, 1, v7 // (op 27067) 

v_lshl_add_u32 v7, v8, 7, v9 // (op 27067) 


v_lshl_add_u32 v8, s17, 8, v7 // (op 27067) 

v_mul_lo_u32 v7, v8, s2 // (op 27067) 

v_add_lshl_u32 v8, v5, v7, 1 // (op 27067) 


v_and_b32 v5, 4294967168, v1 // (op 27206) 
v_and_b32 v7, 63, v1 // (op 27206) 
v_lshrrev_b32 v9, 4, v7 // (op 27206) 
s_mul_i32 s1, s2, s9 // (op 27069) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 27069) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 27069) least significant
s_mul_hi_u32 s19, s2, s8 // (op 27069) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 27069) most significant: sum
s_add_u32 s1, s1, s19 // (op 27069) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 27069) 

v_lshlrev_b32 v10, 2, v9 // (op 27206) 

v_add_u32 v9, v5, v10 // (op 27206) 


v_lshl_add_u32 v5, s16, 8, v9 // (op 27206) 

v_lshrrev_b32 v9, 2, v7 // (op 27206) 

v_and_b32 v7, 3, v9 // (op 27206) 

v_and_b32 v9, 3, v1 // (op 27206) 
v_lshl_add_u32 v10, v7, 2, v9 // (op 27206) 


v_lshrrev_b32 v7, 6, v1 // (op 27206) 
v_and_b32 v9, 1, v7 // (op 27206) 

v_lshl_add_u32 v7, v9, 7, v10 // (op 27206) 


v_lshl_add_u32 v9, s17, 8, v7 // (op 27206) 

v_mul_lo_u32 v7, v9, s4 // (op 27206) 

v_add_lshl_u32 v9, v5, v7, 1 // (op 27206) 


s_mov_b32 s40, s6 // (op 27069) reg expression
s_mov_b32 s41, s7 // (op 27069) reg expression
s_mov_b32 s42, s18 // (op 27069) reg expression
s_mov_b32 s43, 131072 // (op 27069) call()

s_mul_i32 s1, s4, s9 // (op 27208) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 27208) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 27208) least significant
s_mul_hi_u32 s7, s4, s8 // (op 27208) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 27208) most significant: sum
s_add_u32 s1, s1, s7 // (op 27208) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 27208) 

s_mov_b32 s44, s10 // (op 27208) reg expression
s_mov_b32 s45, s11 // (op 27208) reg expression
s_mov_b32 s46, s6 // (op 27208) reg expression
s_mov_b32 s47, 131072 // (op 27208) call()


v_mov_b32 v5, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_accvgpr_write a4, 0.00000 // (op 281) call()
v_accvgpr_write a5, 0.00000 // (op 281) call()
v_accvgpr_write a6, 0.00000 // (op 281) call()
v_accvgpr_write a7, 0.00000 // (op 281) call()
v_accvgpr_write a8, 0.00000 // (op 417) call()
v_accvgpr_write a9, 0.00000 // (op 417) call()
v_accvgpr_write a10, 0.00000 // (op 417) call()
v_accvgpr_write a11, 0.00000 // (op 417) call()
v_accvgpr_write a12, 0.00000 // (op 553) call()
v_accvgpr_write a13, 0.00000 // (op 553) call()
v_accvgpr_write a14, 0.00000 // (op 553) call()
v_accvgpr_write a15, 0.00000 // (op 553) call()
v_accvgpr_write a16, 0.00000 // (op 689) call()
v_accvgpr_write a17, 0.00000 // (op 689) call()
v_accvgpr_write a18, 0.00000 // (op 689) call()
v_accvgpr_write a19, 0.00000 // (op 689) call()
v_accvgpr_write a20, 0.00000 // (op 825) call()
v_accvgpr_write a21, 0.00000 // (op 825) call()
v_accvgpr_write a22, 0.00000 // (op 825) call()
v_accvgpr_write a23, 0.00000 // (op 825) call()
v_accvgpr_write a24, 0.00000 // (op 961) call()
v_accvgpr_write a25, 0.00000 // (op 961) call()
v_accvgpr_write a26, 0.00000 // (op 961) call()
v_accvgpr_write a27, 0.00000 // (op 961) call()
v_accvgpr_write a28, 0.00000 // (op 1097) call()
v_accvgpr_write a29, 0.00000 // (op 1097) call()
v_accvgpr_write a30, 0.00000 // (op 1097) call()
v_accvgpr_write a31, 0.00000 // (op 1097) call()
v_accvgpr_write a32, 0.00000 // (op 1409) call()
v_accvgpr_write a33, 0.00000 // (op 1409) call()
v_accvgpr_write a34, 0.00000 // (op 1409) call()
v_accvgpr_write a35, 0.00000 // (op 1409) call()
v_accvgpr_write a36, 0.00000 // (op 1930) call()
v_accvgpr_write a37, 0.00000 // (op 1930) call()
v_accvgpr_write a38, 0.00000 // (op 1930) call()
v_accvgpr_write a39, 0.00000 // (op 1930) call()
v_accvgpr_write a40, 0.00000 // (op 1931) call()
v_accvgpr_write a41, 0.00000 // (op 1931) call()
v_accvgpr_write a42, 0.00000 // (op 1931) call()
v_accvgpr_write a43, 0.00000 // (op 1931) call()
v_accvgpr_write a44, 0.00000 // (op 1932) call()
v_accvgpr_write a45, 0.00000 // (op 1932) call()
v_accvgpr_write a46, 0.00000 // (op 1932) call()
v_accvgpr_write a47, 0.00000 // (op 1932) call()
v_accvgpr_write a48, 0.00000 // (op 1933) call()
v_accvgpr_write a49, 0.00000 // (op 1933) call()
v_accvgpr_write a50, 0.00000 // (op 1933) call()
v_accvgpr_write a51, 0.00000 // (op 1933) call()
v_accvgpr_write a52, 0.00000 // (op 1934) call()
v_accvgpr_write a53, 0.00000 // (op 1934) call()
v_accvgpr_write a54, 0.00000 // (op 1934) call()
v_accvgpr_write a55, 0.00000 // (op 1934) call()
v_accvgpr_write a56, 0.00000 // (op 1935) call()
v_accvgpr_write a57, 0.00000 // (op 1935) call()
v_accvgpr_write a58, 0.00000 // (op 1935) call()
v_accvgpr_write a59, 0.00000 // (op 1935) call()
v_accvgpr_write a60, 0.00000 // (op 1936) call()
v_accvgpr_write a61, 0.00000 // (op 1936) call()
v_accvgpr_write a62, 0.00000 // (op 1936) call()
v_accvgpr_write a63, 0.00000 // (op 1936) call()
v_accvgpr_write a64, 0.00000 // (op 2627) call()
v_accvgpr_write a65, 0.00000 // (op 2627) call()
v_accvgpr_write a66, 0.00000 // (op 2627) call()
v_accvgpr_write a67, 0.00000 // (op 2627) call()
v_accvgpr_write a68, 0.00000 // (op 3148) call()
v_accvgpr_write a69, 0.00000 // (op 3148) call()
v_accvgpr_write a70, 0.00000 // (op 3148) call()
v_accvgpr_write a71, 0.00000 // (op 3148) call()
v_accvgpr_write a72, 0.00000 // (op 3149) call()
v_accvgpr_write a73, 0.00000 // (op 3149) call()
v_accvgpr_write a74, 0.00000 // (op 3149) call()
v_accvgpr_write a75, 0.00000 // (op 3149) call()
v_accvgpr_write a76, 0.00000 // (op 3150) call()
v_accvgpr_write a77, 0.00000 // (op 3150) call()
v_accvgpr_write a78, 0.00000 // (op 3150) call()
v_accvgpr_write a79, 0.00000 // (op 3150) call()
v_accvgpr_write a80, 0.00000 // (op 3151) call()
v_accvgpr_write a81, 0.00000 // (op 3151) call()
v_accvgpr_write a82, 0.00000 // (op 3151) call()
v_accvgpr_write a83, 0.00000 // (op 3151) call()
v_accvgpr_write a84, 0.00000 // (op 3152) call()
v_accvgpr_write a85, 0.00000 // (op 3152) call()
v_accvgpr_write a86, 0.00000 // (op 3152) call()
v_accvgpr_write a87, 0.00000 // (op 3152) call()
v_accvgpr_write a88, 0.00000 // (op 3153) call()
v_accvgpr_write a89, 0.00000 // (op 3153) call()
v_accvgpr_write a90, 0.00000 // (op 3153) call()
v_accvgpr_write a91, 0.00000 // (op 3153) call()
v_accvgpr_write a92, 0.00000 // (op 3154) call()
v_accvgpr_write a93, 0.00000 // (op 3154) call()
v_accvgpr_write a94, 0.00000 // (op 3154) call()
v_accvgpr_write a95, 0.00000 // (op 3154) call()
v_accvgpr_write a96, 0.00000 // (op 3845) call()
v_accvgpr_write a97, 0.00000 // (op 3845) call()
v_accvgpr_write a98, 0.00000 // (op 3845) call()
v_accvgpr_write a99, 0.00000 // (op 3845) call()
v_accvgpr_write a100, 0.00000 // (op 4366) call()
v_accvgpr_write a101, 0.00000 // (op 4366) call()
v_accvgpr_write a102, 0.00000 // (op 4366) call()
v_accvgpr_write a103, 0.00000 // (op 4366) call()
v_accvgpr_write a104, 0.00000 // (op 4367) call()
v_accvgpr_write a105, 0.00000 // (op 4367) call()
v_accvgpr_write a106, 0.00000 // (op 4367) call()
v_accvgpr_write a107, 0.00000 // (op 4367) call()
v_accvgpr_write a108, 0.00000 // (op 4368) call()
v_accvgpr_write a109, 0.00000 // (op 4368) call()
v_accvgpr_write a110, 0.00000 // (op 4368) call()
v_accvgpr_write a111, 0.00000 // (op 4368) call()
v_accvgpr_write a112, 0.00000 // (op 4369) call()
v_accvgpr_write a113, 0.00000 // (op 4369) call()
v_accvgpr_write a114, 0.00000 // (op 4369) call()
v_accvgpr_write a115, 0.00000 // (op 4369) call()
v_accvgpr_write a116, 0.00000 // (op 4370) call()
v_accvgpr_write a117, 0.00000 // (op 4370) call()
v_accvgpr_write a118, 0.00000 // (op 4370) call()
v_accvgpr_write a119, 0.00000 // (op 4370) call()
v_accvgpr_write a120, 0.00000 // (op 4371) call()
v_accvgpr_write a121, 0.00000 // (op 4371) call()
v_accvgpr_write a122, 0.00000 // (op 4371) call()
v_accvgpr_write a123, 0.00000 // (op 4371) call()
v_accvgpr_write a124, 0.00000 // (op 4372) call()
v_accvgpr_write a125, 0.00000 // (op 4372) call()
v_accvgpr_write a126, 0.00000 // (op 4372) call()
v_accvgpr_write a127, 0.00000 // (op 4372) call()
v_accvgpr_write a128, 0.00000 // (op 5063) call()
v_accvgpr_write a129, 0.00000 // (op 5063) call()
v_accvgpr_write a130, 0.00000 // (op 5063) call()
v_accvgpr_write a131, 0.00000 // (op 5063) call()
v_accvgpr_write a132, 0.00000 // (op 5584) call()
v_accvgpr_write a133, 0.00000 // (op 5584) call()
v_accvgpr_write a134, 0.00000 // (op 5584) call()
v_accvgpr_write a135, 0.00000 // (op 5584) call()
v_accvgpr_write a136, 0.00000 // (op 5585) call()
v_accvgpr_write a137, 0.00000 // (op 5585) call()
v_accvgpr_write a138, 0.00000 // (op 5585) call()
v_accvgpr_write a139, 0.00000 // (op 5585) call()
v_accvgpr_write a140, 0.00000 // (op 5586) call()
v_accvgpr_write a141, 0.00000 // (op 5586) call()
v_accvgpr_write a142, 0.00000 // (op 5586) call()
v_accvgpr_write a143, 0.00000 // (op 5586) call()
v_accvgpr_write a144, 0.00000 // (op 5587) call()
v_accvgpr_write a145, 0.00000 // (op 5587) call()
v_accvgpr_write a146, 0.00000 // (op 5587) call()
v_accvgpr_write a147, 0.00000 // (op 5587) call()
v_accvgpr_write a148, 0.00000 // (op 5588) call()
v_accvgpr_write a149, 0.00000 // (op 5588) call()
v_accvgpr_write a150, 0.00000 // (op 5588) call()
v_accvgpr_write a151, 0.00000 // (op 5588) call()
v_accvgpr_write a152, 0.00000 // (op 5589) call()
v_accvgpr_write a153, 0.00000 // (op 5589) call()
v_accvgpr_write a154, 0.00000 // (op 5589) call()
v_accvgpr_write a155, 0.00000 // (op 5589) call()
v_accvgpr_write a156, 0.00000 // (op 5590) call()
v_accvgpr_write a157, 0.00000 // (op 5590) call()
v_accvgpr_write a158, 0.00000 // (op 5590) call()
v_accvgpr_write a159, 0.00000 // (op 5590) call()
v_accvgpr_write a160, 0.00000 // (op 6281) call()
v_accvgpr_write a161, 0.00000 // (op 6281) call()
v_accvgpr_write a162, 0.00000 // (op 6281) call()
v_accvgpr_write a163, 0.00000 // (op 6281) call()
v_accvgpr_write a164, 0.00000 // (op 6802) call()
v_accvgpr_write a165, 0.00000 // (op 6802) call()
v_accvgpr_write a166, 0.00000 // (op 6802) call()
v_accvgpr_write a167, 0.00000 // (op 6802) call()
v_accvgpr_write a168, 0.00000 // (op 6803) call()
v_accvgpr_write a169, 0.00000 // (op 6803) call()
v_accvgpr_write a170, 0.00000 // (op 6803) call()
v_accvgpr_write a171, 0.00000 // (op 6803) call()
v_accvgpr_write a172, 0.00000 // (op 6804) call()
v_accvgpr_write a173, 0.00000 // (op 6804) call()
v_accvgpr_write a174, 0.00000 // (op 6804) call()
v_accvgpr_write a175, 0.00000 // (op 6804) call()
v_accvgpr_write a176, 0.00000 // (op 6805) call()
v_accvgpr_write a177, 0.00000 // (op 6805) call()
v_accvgpr_write a178, 0.00000 // (op 6805) call()
v_accvgpr_write a179, 0.00000 // (op 6805) call()
v_accvgpr_write a180, 0.00000 // (op 6806) call()
v_accvgpr_write a181, 0.00000 // (op 6806) call()
v_accvgpr_write a182, 0.00000 // (op 6806) call()
v_accvgpr_write a183, 0.00000 // (op 6806) call()
v_accvgpr_write a184, 0.00000 // (op 6807) call()
v_accvgpr_write a185, 0.00000 // (op 6807) call()
v_accvgpr_write a186, 0.00000 // (op 6807) call()
v_accvgpr_write a187, 0.00000 // (op 6807) call()
v_accvgpr_write a188, 0.00000 // (op 6808) call()
v_accvgpr_write a189, 0.00000 // (op 6808) call()
v_accvgpr_write a190, 0.00000 // (op 6808) call()
v_accvgpr_write a191, 0.00000 // (op 6808) call()
v_accvgpr_write a192, 0.00000 // (op 7499) call()
v_accvgpr_write a193, 0.00000 // (op 7499) call()
v_accvgpr_write a194, 0.00000 // (op 7499) call()
v_accvgpr_write a195, 0.00000 // (op 7499) call()
v_accvgpr_write a196, 0.00000 // (op 8020) call()
v_accvgpr_write a197, 0.00000 // (op 8020) call()
v_accvgpr_write a198, 0.00000 // (op 8020) call()
v_accvgpr_write a199, 0.00000 // (op 8020) call()
v_accvgpr_write a200, 0.00000 // (op 8021) call()
v_accvgpr_write a201, 0.00000 // (op 8021) call()
v_accvgpr_write a202, 0.00000 // (op 8021) call()
v_accvgpr_write a203, 0.00000 // (op 8021) call()
v_accvgpr_write a204, 0.00000 // (op 8022) call()
v_accvgpr_write a205, 0.00000 // (op 8022) call()
v_accvgpr_write a206, 0.00000 // (op 8022) call()
v_accvgpr_write a207, 0.00000 // (op 8022) call()
v_accvgpr_write a208, 0.00000 // (op 8023) call()
v_accvgpr_write a209, 0.00000 // (op 8023) call()
v_accvgpr_write a210, 0.00000 // (op 8023) call()
v_accvgpr_write a211, 0.00000 // (op 8023) call()
v_accvgpr_write a212, 0.00000 // (op 8024) call()
v_accvgpr_write a213, 0.00000 // (op 8024) call()
v_accvgpr_write a214, 0.00000 // (op 8024) call()
v_accvgpr_write a215, 0.00000 // (op 8024) call()
v_accvgpr_write a216, 0.00000 // (op 8025) call()
v_accvgpr_write a217, 0.00000 // (op 8025) call()
v_accvgpr_write a218, 0.00000 // (op 8025) call()
v_accvgpr_write a219, 0.00000 // (op 8025) call()
v_accvgpr_write a220, 0.00000 // (op 8026) call()
v_accvgpr_write a221, 0.00000 // (op 8026) call()
v_accvgpr_write a222, 0.00000 // (op 8026) call()
v_accvgpr_write a223, 0.00000 // (op 8026) call()
v_accvgpr_write a224, 0.00000 // (op 8717) call()
v_accvgpr_write a225, 0.00000 // (op 8717) call()
v_accvgpr_write a226, 0.00000 // (op 8717) call()
v_accvgpr_write a227, 0.00000 // (op 8717) call()
v_accvgpr_write a228, 0.00000 // (op 9238) call()
v_accvgpr_write a229, 0.00000 // (op 9238) call()
v_accvgpr_write a230, 0.00000 // (op 9238) call()
v_accvgpr_write a231, 0.00000 // (op 9238) call()
v_accvgpr_write a232, 0.00000 // (op 9239) call()
v_accvgpr_write a233, 0.00000 // (op 9239) call()
v_accvgpr_write a234, 0.00000 // (op 9239) call()
v_accvgpr_write a235, 0.00000 // (op 9239) call()
v_accvgpr_write a236, 0.00000 // (op 9240) call()
v_accvgpr_write a237, 0.00000 // (op 9240) call()
v_accvgpr_write a238, 0.00000 // (op 9240) call()
v_accvgpr_write a239, 0.00000 // (op 9240) call()
v_accvgpr_write a240, 0.00000 // (op 9241) call()
v_accvgpr_write a241, 0.00000 // (op 9241) call()
v_accvgpr_write a242, 0.00000 // (op 9241) call()
v_accvgpr_write a243, 0.00000 // (op 9241) call()
v_accvgpr_write a244, 0.00000 // (op 9242) call()
v_accvgpr_write a245, 0.00000 // (op 9242) call()
v_accvgpr_write a246, 0.00000 // (op 9242) call()
v_accvgpr_write a247, 0.00000 // (op 9242) call()
v_accvgpr_write a248, 0.00000 // (op 9243) call()
v_accvgpr_write a249, 0.00000 // (op 9243) call()
v_accvgpr_write a250, 0.00000 // (op 9243) call()
v_accvgpr_write a251, 0.00000 // (op 9243) call()
v_accvgpr_write a252, 0.00000 // (op 9244) call()
v_accvgpr_write a253, 0.00000 // (op 9244) call()
v_accvgpr_write a254, 0.00000 // (op 9244) call()
v_accvgpr_write a255, 0.00000 // (op 9244) call()
v_lshrrev_b32 v10, 2, v1 // (op 26957) 
v_lshl_add_u32 v11, s16, 8, v10 // (op 26957) 

v_and_b32 v10, -4, v1 // (op 26957) 
v_sub_u32 v12, v1, v10 // (op 26957) 

v_lshlrev_b32 v10, 5, v12 // (op 26957) 

v_mov_b32 v13, 0 // (op 26957) convert
v_mov_b32 v12, v10 // (op 26957) convert

v_mul_lo_u32 v15, v11, s13 // (op 26957) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 26957) most significant: high of low * low
v_mul_lo_u32 v14, v11, s12 // (op 26957) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 26957) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 26957) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 26957) most significant half



v_lshlrev_b64 v[10:11], 2, v[14:15] // (op 26957) 

v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 26957) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 26959) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 26959) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 26959) least significant
s_mul_hi_u32 s7, s12, s22 // (op 26959) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 26959) most significant: sum
s_add_u32 s1, s1, s7 // (op 26959) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 26959) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 26959) 

s_mov_b32 s48, s20 // (op 26959) reg expression
s_mov_b32 s49, s21 // (op 26959) reg expression
s_mov_b32 s50, s0 // (op 26959) reg expression
s_mov_b32 s51, 131072 // (op 26959) call()

v_lshrrev_b32 v10, 2, v1 // (op 26976) 
v_lshl_add_u32 v11, s16, 8, v10 // (op 26976) 

v_and_b32 v10, -4, v1 // (op 26976) 
v_sub_u32 v14, v1, v10 // (op 26976) 

v_lshlrev_b32 v10, 5, v14 // (op 26976) 

v_mov_b32 v15, 0 // (op 26976) convert
v_mov_b32 v14, v10 // (op 26976) convert

v_mul_lo_u32 v17, v11, s13 // (op 26976) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 26976) most significant: high of low * low
v_mul_lo_u32 v16, v11, s12 // (op 26976) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 26976) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 26976) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 26976) most significant half



v_lshlrev_b64 v[10:11], 2, v[16:17] // (op 26976) 

v_lshrrev_b64 v[14:15], 3, v[10:11] // (op 26976) 

s_mul_i32 s1, s12, s23 // (op 26978) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 26978) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 26978) least significant
s_mul_hi_u32 s7, s12, s22 // (op 26978) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 26978) most significant: sum
s_add_u32 s1, s1, s7 // (op 26978) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 26978) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 26978) 

s_mov_b32 s48, s20 // (op 26978) reg expression
s_mov_b32 s49, s21 // (op 26978) reg expression
s_mov_b32 s50, s0 // (op 26978) reg expression
s_mov_b32 s51, 131072 // (op 26978) call()

v_and_b32 v10, -8, v1 // (op 27021) 
v_sub_u32 v11, v1, v10 // (op 27021) 

v_lshlrev_b32 v10, 4, v11 // (op 27021) 

v_mov_b32 v17, 0 // (op 27021) convert
v_mov_b32 v16, v10 // (op 27021) convert

v_lshrrev_b32 v10, 3, v1 // (op 27021) 
v_lshl_add_u32 v11, s17, 8, v10 // (op 27021) 

v_mul_lo_u32 v19, v11, s25 // (op 27021) most significant: low of low * high
v_mul_hi_u32 v10, v11, s24 // (op 27021) most significant: high of low * low
v_mul_lo_u32 v18, v11, s24 // (op 27021) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 27021) most significant: sum


v_add_co_u32 v10, s[0:1], v16, v18 // (op 27021) least significant half
v_addc_co_u32 v11, s[0:1], v17, v19, s[0:1] // (op 27021) most significant half



s_mul_i32 s1, s24, s9 // (op 27023) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 27023) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 27023) least significant
s_mul_hi_u32 s7, s24, s8 // (op 27023) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 27023) most significant: sum
s_add_u32 s1, s1, s7 // (op 27023) most significant: sum


s_mov_b32 s52, s26 // (op 27023) reg expression
s_mov_b32 s53, s27 // (op 27023) reg expression
s_mov_b32 s54, s0 // (op 27023) reg expression
s_mov_b32 s55, 131072 // (op 27023) call()

v_and_b32 v16, -8, v1 // (op 27040) 
v_sub_u32 v17, v1, v16 // (op 27040) 

v_lshlrev_b32 v16, 4, v17 // (op 27040) 

v_mov_b32 v19, 0 // (op 27040) convert
v_mov_b32 v18, v16 // (op 27040) convert

v_lshrrev_b32 v16, 3, v1 // (op 27040) 
v_lshl_add_u32 v17, s17, 8, v16 // (op 27040) 

v_mul_lo_u32 v21, v17, s25 // (op 27040) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 27040) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 27040) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 27040) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 27040) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 27040) most significant half



s_mul_i32 s1, s24, s9 // (op 27042) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 27042) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 27042) least significant
s_mul_hi_u32 s7, s24, s8 // (op 27042) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 27042) most significant: sum
s_add_u32 s1, s1, s7 // (op 27042) most significant: sum


s_mov_b32 s52, s26 // (op 27042) reg expression
s_mov_b32 s53, s27 // (op 27042) reg expression
s_mov_b32 s54, s0 // (op 27042) reg expression
s_mov_b32 s55, 131072 // (op 27042) call()

v_and_b32 v18, 63, v1 // (op 27318) 
v_lshrrev_b32 v19, 4, v18 // (op 27318) 
v_and_b32 v20, 3, v19 // (op 27318) 

v_and_b32 v19, 15, v1 // (op 27318) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 27318) 


v_and_b32 v19, 4294967168, v1 // (op 27318) 
v_add_u32 v20, v19, v21 // (op 27318) 


v_lshl_add_u32 v19, s16, 8, v20 // (op 27318) 

v_lshrrev_b32 v20, 6, v18 // (op 27318) 

v_lshlrev_b32 v18, 2, v20 // (op 27318) 

v_mul_lo_u32 v20, v19, s28 // (op 27318) 
v_add_u32 v20, v20, v18 // (op 27318) 


s_mul_i32 s1, s28, s23 // (op 27320) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 27320) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 27320) least significant
s_mul_hi_u32 s7, s28, s22 // (op 27320) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 27320) most significant: sum
s_add_u32 s1, s1, s7 // (op 27320) most significant: sum


s_mov_b32 s56, s30 // (op 27320) reg expression
s_mov_b32 s57, s31 // (op 27320) reg expression
s_mov_b32 s58, s0 // (op 27320) reg expression
s_mov_b32 s59, 131072 // (op 27320) call()

v_and_b32 v18, 63, v1 // (op 27339) 
v_lshrrev_b32 v19, 4, v18 // (op 27339) 
v_and_b32 v21, 3, v19 // (op 27339) 

v_and_b32 v19, 15, v1 // (op 27339) 
v_lshl_add_u32 v22, v21, 4, v19 // (op 27339) 


v_and_b32 v19, 4294967168, v1 // (op 27339) 
v_add_u32 v21, v19, v22 // (op 27339) 


v_lshl_add_u32 v19, s16, 8, v21 // (op 27339) 

v_lshrrev_b32 v21, 6, v18 // (op 27339) 

v_lshlrev_b32 v18, 2, v21 // (op 27339) 

v_mul_lo_u32 v21, v19, s28 // (op 27339) 
v_add_u32 v21, v21, v18 // (op 27339) 


s_mul_i32 s1, s28, s23 // (op 27341) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 27341) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 27341) least significant
s_mul_hi_u32 s7, s28, s22 // (op 27341) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 27341) most significant: sum
s_add_u32 s1, s1, s7 // (op 27341) most significant: sum


s_mov_b32 s56, s30 // (op 27341) reg expression
s_mov_b32 s57, s31 // (op 27341) reg expression
s_mov_b32 s58, s0 // (op 27341) reg expression
s_mov_b32 s59, 131072 // (op 27341) call()

v_and_b32 v18, 63, v1 // (op 27370) 
v_lshrrev_b32 v19, 4, v18 // (op 27370) 
v_and_b32 v22, 3, v19 // (op 27370) 

v_and_b32 v19, 15, v1 // (op 27370) 
v_lshl_add_u32 v23, v22, 4, v19 // (op 27370) 


v_lshrrev_b32 v19, 6, v1 // (op 27370) 
v_and_b32 v22, 1, v19 // (op 27370) 

v_lshl_add_u32 v19, v22, 7, v23 // (op 27370) 


v_lshl_add_u32 v22, s17, 8, v19 // (op 27370) 

v_mul_lo_u32 v19, v22, s32 // (op 27370) 

v_lshrrev_b32 v22, 6, v18 // (op 27370) 

v_lshl_add_u32 v18, v22, 2, v19 // (op 27370) 


s_mul_i32 s1, s32, s9 // (op 27372) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 27372) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 27372) least significant
s_mul_hi_u32 s7, s32, s8 // (op 27372) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 27372) most significant: sum
s_add_u32 s1, s1, s7 // (op 27372) most significant: sum


s_mov_b32 s60, s34 // (op 27372) reg expression
s_mov_b32 s61, s35 // (op 27372) reg expression
s_mov_b32 s62, s0 // (op 27372) reg expression
s_mov_b32 s63, 131072 // (op 27372) call()

v_and_b32 v19, 63, v1 // (op 27391) 
v_lshrrev_b32 v22, 4, v19 // (op 27391) 
v_and_b32 v23, 3, v22 // (op 27391) 

v_and_b32 v22, 15, v1 // (op 27391) 
v_lshl_add_u32 v24, v23, 4, v22 // (op 27391) 


v_lshrrev_b32 v22, 6, v1 // (op 27391) 
v_and_b32 v23, 1, v22 // (op 27391) 

v_lshl_add_u32 v22, v23, 7, v24 // (op 27391) 


v_lshl_add_u32 v23, s17, 8, v22 // (op 27391) 

v_mul_lo_u32 v22, v23, s32 // (op 27391) 

v_lshrrev_b32 v23, 6, v19 // (op 27391) 

v_lshl_add_u32 v19, v23, 2, v22 // (op 27391) 


s_mul_i32 s1, s32, s9 // (op 27393) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 27393) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 27393) least significant
s_mul_hi_u32 s7, s32, s8 // (op 27393) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 27393) most significant: sum
s_add_u32 s1, s1, s7 // (op 27393) most significant: sum


s_mov_b32 s60, s34 // (op 27393) reg expression
s_mov_b32 s61, s35 // (op 27393) reg expression
s_mov_b32 s62, s0 // (op 27393) reg expression
s_mov_b32 s63, 131072 // (op 27393) call()

v_and_b32 v22, -8, v1 // (op 27168) 
v_sub_u32 v23, v1, v22 // (op 27168) 

v_lshlrev_b32 v22, 4, v23 // (op 27168) 

v_lshrrev_b32 v23, 3, v1 // (op 27168) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 27168) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 27168) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 26843) 
v_mov_b64 v[22:23], s[6:7] // (op 26843) convert


v_mov_b64 v[24:25], v[10:11] // (op 26843) 
s_mov_b32 m0, s0 // (op 26843) reg expression
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 26843) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26843) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26843) most significant half

s_add_u32 m0, m0, 4096 // (op 26843) 
buffer_load_dwordx4 v24, s[52:55], 0 offen offset: 0 lds // (op 26843) Load value direct to lds



v_and_b32 v22, -4, v1 // (op 27111) 
v_sub_u32 v23, v1, v22 // (op 27111) 

v_lshlrev_b32 v22, 5, v23 // (op 27111) 

v_lshrrev_b32 v23, 2, v1 // (op 27111) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 27111) 


v_lshlrev_b32 v22, 2, v24 // (op 27111) 

v_lshrrev_b32 v23, 3, v22 // (op 27111) 

s_nop 0
v_readfirstlane_b32 s0, v23 // (op 27111) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 26848) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 26848) 

v_lshrrev_b64 v[22:23], 3, s[10:11] // (op 26848) 


v_mov_b64 v[24:25], v[12:13] // (op 26848) 
s_add_u32 s1, 32768, s0 // (op 26848) 
s_mov_b32 m0, s1 // (op 26848) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds
v_add_co_u32 v24, s[6:7], v24, v22 // (op 26848) least significant half
v_addc_co_u32 v25, s[6:7], 0, v25, s[6:7] // (op 26848) most significant half

s_add_u32 m0, m0, 4096 // (op 26848) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 26848) Load value direct to lds






v_mov_b32 v22, v18 // (op 76) reg expression

buffer_load_dwordx2 v[24:25], v22, s[60:63], 0 offen // (op 76) Load value



s_lshl_b32 s0, s32, 6 // (op 726) 
v_add_u32 v22, v18, s0 // (op 726) 


buffer_load_dwordx2 v[26:27], v22, s[60:63], 0 offen // (op 726) Load value




s_lshl_b32 s0, s28, 6 // (op 5311) 
v_add_u32 v18, v20, s0 // (op 5311) 


buffer_load_dwordx2 v[22:23], v18, s[56:59], 0 offen // (op 5311) Load value



v_mov_b32 v18, v20 // (op 72) reg expression

buffer_load_dwordx2 v[28:29], v18, s[56:59], 0 offen // (op 72) Load value


v_and_b32 v18, -8, v1 // (op 27279) 
v_sub_u32 v20, v1, v18 // (op 27279) 

v_lshlrev_b32 v18, 4, v20 // (op 27279) 

v_lshrrev_b32 v20, 3, v1 // (op 27279) 
v_lshl_add_u32 v30, v20, 7, v18 // (op 27279) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 27279) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 26853) 
v_mov_b64 v[30:31], s[6:7] // (op 26853) convert


v_mov_b64 v[32:33], v[10:11] // (op 26853) 
s_add_u32 s1, 49152, s0 // (op 26853) 
s_mov_b32 m0, s1 // (op 26853) reg expression
s_mov_b32 s6, 128 // (op 26853) call()
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 26853) Load value direct to lds
v_add_co_u32 v32, s[10:11], v32, v30 // (op 26853) least significant half
v_addc_co_u32 v33, s[10:11], 0, v33, s[10:11] // (op 26853) most significant half

s_add_u32 m0, m0, 4096 // (op 26853) 
buffer_load_dwordx4 v32, s[52:55], s6 offen offset: 0 lds // (op 26853) Load value direct to lds






v_and_b32 v18, -4, v1 // (op 27235) 
v_sub_u32 v20, v1, v18 // (op 27235) 

v_lshlrev_b32 v18, 5, v20 // (op 27235) 

v_lshrrev_b32 v20, 2, v1 // (op 27235) 
v_lshl_add_u32 v10, v20, 7, v18 // (op 27235) 


v_lshlrev_b32 v11, 2, v10 // (op 27235) 

v_lshrrev_b32 v10, 3, v11 // (op 27235) 

s_nop 0
v_readfirstlane_b32 s0, v10 // (op 27235) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 26855) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 26855) 

v_lshrrev_b64 v[10:11], 3, s[10:11] // (op 26855) 


v_mov_b64 v[30:31], v[12:13] // (op 26855) 
s_add_u32 s1, 81920, s0 // (op 26855) 
s_mov_b32 m0, s1 // (op 26855) reg expression
buffer_load_dwordx4 v30, s[48:51], 64 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v10 // (op 26855) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], 64 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v10 // (op 26855) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], 64 offen offset: 0 lds // (op 26855) Load value direct to lds
v_add_co_u32 v30, s[6:7], v30, v10 // (op 26855) least significant half
v_addc_co_u32 v31, s[6:7], 0, v31, s[6:7] // (op 26855) most significant half

s_add_u32 m0, m0, 4096 // (op 26855) 
buffer_load_dwordx4 v30, s[48:51], 64 offen offset: 0 lds // (op 26855) Load value direct to lds





s_waitcnt vmcnt(20)
s_barrier  // (op 27808) 



ds_read_b128 v[30:33], v4 // (op 26194) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 26194) Load local data 



ds_read_b128 v[38:41], v4 offset:2048 // (op 26199) Load local data 
ds_read_b128 v[42:45], v4 offset:2112 // (op 26199) Load local data 



ds_read_b128 v[46:49], v4 offset:4096 // (op 26204) Load local data 
ds_read_b128 v[50:53], v4 offset:4160 // (op 26204) Load local data 



ds_read_b128 v[54:57], v4 offset:6144 // (op 26209) Load local data 
ds_read_b128 v[58:61], v4 offset:6208 // (op 26209) Load local data 



ds_read_b128 v[62:65], v4 offset:8192 // (op 26214) Load local data 
ds_read_b128 v[66:69], v4 offset:8256 // (op 26214) Load local data 



ds_read_b128 v[70:73], v4 offset:10240 // (op 26219) Load local data 
ds_read_b128 v[74:77], v4 offset:10304 // (op 26219) Load local data 



ds_read_b128 v[78:81], v4 offset:12288 // (op 26224) Load local data 
ds_read_b128 v[82:85], v4 offset:12352 // (op 26224) Load local data 



ds_read_b128 v[86:89], v4 offset:14336 // (op 26229) Load local data 
ds_read_b128 v[90:93], v4 offset:14400 // (op 26229) Load local data 
s_waitcnt vmcnt(16)
s_barrier  // (op 27792) 



ds_read_b128 v[10:13], v0 offset:32768 // (op 26154) Load local data 



ds_read_b128 v[94:97], v0 offset:33792 // (op 26159) Load local data 



ds_read_b128 v[98:101], v0 offset:34816 // (op 26164) Load local data 



ds_read_b128 v[102:105], v0 offset:35840 // (op 26169) Load local data 



ds_read_b128 v[106:109], v0 offset:36864 // (op 26174) Load local data 



ds_read_b128 v[110:113], v0 offset:37888 // (op 26179) Load local data 



ds_read_b128 v[114:117], v0 offset:38912 // (op 26184) Load local data 



ds_read_b128 v[118:121], v0 offset:39936 // (op 26189) Load local data 
v_and_b32 v18, -4, v1 // (op 27096) 
v_sub_u32 v20, v1, v18 // (op 27096) 

v_lshlrev_b32 v18, 5, v20 // (op 27096) 

v_lshrrev_b32 v20, 2, v1 // (op 27096) 
v_lshl_add_u32 v122, v20, 7, v18 // (op 27096) 


v_lshlrev_b32 v18, 2, v122 // (op 27096) 

v_lshrrev_b32 v20, 3, v18 // (op 27096) 

s_nop 0
v_readfirstlane_b32 s0, v20 // (op 27096) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 27153) 
v_sub_u32 v20, v1, v18 // (op 27153) 

v_lshlrev_b32 v18, 4, v20 // (op 27153) 

v_lshrrev_b32 v20, 3, v1 // (op 27153) 
v_lshl_add_u32 v122, v20, 7, v18 // (op 27153) 


s_nop 0
v_readfirstlane_b32 s1, v122 // (op 27153) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 27220) 
v_sub_u32 v20, v1, v18 // (op 27220) 

v_lshlrev_b32 v18, 5, v20 // (op 27220) 

v_lshrrev_b32 v20, 2, v1 // (op 27220) 
v_lshl_add_u32 v122, v20, 7, v18 // (op 27220) 


v_lshlrev_b32 v18, 2, v122 // (op 27220) 

v_lshrrev_b32 v20, 3, v18 // (op 27220) 

s_nop 0
v_readfirstlane_b32 s6, v20 // (op 27220) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 27264) 
v_sub_u32 v20, v1, v18 // (op 27264) 

v_lshlrev_b32 v18, 4, v20 // (op 27264) 

v_lshrrev_b32 v20, 3, v1 // (op 27264) 
v_lshl_add_u32 v122, v20, 7, v18 // (op 27264) 


s_nop 0
v_readfirstlane_b32 s7, v122 // (op 27264) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s10, 0 // (op 8908) call()
s_ashr_i64 s[14:15], s[36:37], 7 // (op 8907) 
s_mov_b32 s11, s14 // (op 8907) convert

s_and_b32 s14, s11, 4294967294 // (op 8907) 

s_cmp_lt_u32 s10, s14 // (op 8907) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopBottom_KLoop_8907 // (op 8907) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopBottom_KLoop_8907 if false)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopTop_KLoop_8907:
s_add_u32 s10, s10, 2 // (op 8909) 
s_waitcnt vmcnt(15)
v_mov_b32 v122, v24 // (op 26729) reg expression
v_mov_b32 v123, v25 // (op 26729) reg expression
s_waitcnt lgkmcnt(8)
s_barrier  // (op 27811) 
s_waitcnt vmcnt(14)
v_mov_b32 v124, v26 // (op 26742) reg expression

v_mov_b32 v125, v27 // (op 26742) reg expression
s_lshl_b64 s[14:15], s[24:25], 5 // (op 26816) 
v_mov_b64 v[126:127], s[14:15] // (op 26816) convert


v_mov_b64 v[128:129], v[16:17] // (op 26816) 
v_mov_b64 v[130:131], v[128:129] // (op 26816) 
s_mov_b32 m0, s1 // (op 26816) reg expression
s_mov_b32 s11, 256 // (op 26816) call()
buffer_load_dwordx4 v130, s[52:55], s11 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[14:15], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[14:15], 0, v131, s[14:15] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
buffer_load_dwordx4 v130, s[52:55], s11 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[14:15], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[14:15], 0, v131, s[14:15] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
buffer_load_dwordx4 v130, s[52:55], s11 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[14:15], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[14:15], 0, v131, s[14:15] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
s_waitcnt vmcnt(16)
v_mov_b32 v132, v22 // (op 26755) reg expression
v_mov_b32 v133, v23 // (op 26755) reg expression
s_waitcnt vmcnt(15)
v_mov_b32 v134, v28 // (op 26768) reg expression
v_mov_b32 v135, v29 // (op 26768) reg expression
buffer_load_dwordx4 v130, s[52:55], s11 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[14:15], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[14:15], 0, v131, s[14:15] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
buffer_load_dwordx4 v130, s[52:55], s11 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[14:15], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[14:15], 0, v131, s[14:15] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
buffer_load_dwordx4 v130, s[52:55], s11 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[14:15], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[14:15], 0, v131, s[14:15] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
v_bfe_u32 v136, v134, 0, 8 // (op 24807) 
v_bfe_u32 v137, v134, 8, 8 // (op 24807) 
v_bfe_u32 v138, v134, 16, 8 // (op 24807) 
v_bfe_u32 v139, v134, 24, 8 // (op 24807) 
v_permlane16_swap_b32 v136, v137 // (op 24807) 
s_nop 0
v_permlane16_swap_b32 v138, v139 // (op 24807) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 24807) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 24807) 
buffer_load_dwordx4 v130, s[52:55], s11 offen offset: 0 lds // (op 26816) Load value direct to lds
v_add_co_u32 v130, s[14:15], v130, v126 // (op 26816) least significant half
v_addc_co_u32 v131, s[14:15], 0, v131, s[14:15] // (op 26816) most significant half

s_add_u32 m0, m0, 4096 // (op 26816) 
buffer_load_dwordx4 v130, s[52:55], s11 offen offset: 0 lds // (op 26816) Load value direct to lds



v_bfe_u32 v140, v122, 0, 8 // (op 24998) 
v_bfe_u32 v141, v122, 8, 8 // (op 24998) 
v_bfe_u32 v142, v122, 16, 8 // (op 24998) 
v_bfe_u32 v143, v122, 24, 8 // (op 24998) 
v_permlane16_swap_b32 v140, v141 // (op 24998) 
s_nop 0
v_permlane16_swap_b32 v142, v143 // (op 24998) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v140, v142 // (op 24998) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v141, v143 // (op 24998) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 27795) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:13], v[30:37], a[0:3], v136, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 78) 
s_lshl_b64 s[14:15], s[12:13], 6 // (op 26811) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[10:13], v[38:45], a[4:7], v136, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 315) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 26811) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[10:13], v[46:53], a[8:11], v136, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 451) 
v_lshrrev_b64 v[126:127], 3, s[18:19] // (op 26811) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[10:13], v[54:61], a[12:15], v136, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 587) 
v_mov_b64 v[130:131], v[14:15] // (op 26811) 
v_mov_b64 v[144:145], v[130:131] // (op 26811) 
s_add_u32 s11, 32768, s0 // (op 26811) 
s_mov_b32 m0, s11 // (op 26811) reg expression
s_mov_b32 s14, 128 // (op 26811) call()
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 26811) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v126 // (op 26811) least significant half
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 26811) most significant half

s_add_u32 m0, m0, 4096 // (op 26811) 
v_bfe_u32 v146, v124, 0, 8 // (op 25030) 
v_bfe_u32 v147, v124, 8, 8 // (op 25030) 
v_bfe_u32 v148, v124, 16, 8 // (op 25030) 
v_bfe_u32 v149, v124, 24, 8 // (op 25030) 
v_permlane16_swap_b32 v146, v147 // (op 25030) 
s_nop 0
v_permlane16_swap_b32 v148, v149 // (op 25030) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v146, v148 // (op 25030) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v147, v149 // (op 25030) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 26811) Load value direct to lds
v_add_co_u32 v144, s[18:19], v144, v126 // (op 26811) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[10:13], v[62:69], a[16:19], v136, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 723) 
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 26811) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[10:13], v[70:77], a[20:23], v136, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 859) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[78:85], a[24:27], v136, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 995) 
s_add_u32 m0, m0, 4096 // (op 26811) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[10:13], v[86:93], a[28:31], v136, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1131) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 26811) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[94:97], v[30:37], a[32:35], v137, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1899) 
v_add_co_u32 v144, s[18:19], v144, v126 // (op 26811) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[94:97], v[38:45], a[36:39], v137, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1861) 
v_addc_co_u32 v145, s[18:19], 0, v145, s[18:19] // (op 26811) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[94:97], v[46:53], a[40:43], v137, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1823) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:97], v[54:61], a[44:47], v137, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1785) 
s_add_u32 m0, m0, 4096 // (op 26811) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[94:97], v[62:69], a[48:51], v137, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1747) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[94:97], v[70:77], a[52:55], v137, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1709) 
buffer_load_dwordx4 v144, s[48:51], s14 offen offset: 0 lds // (op 26811) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[94:97], v[78:85], a[56:59], v137, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1671) 
v_mov_b32 v18, v19 // (op 26737) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[94:97], v[86:93], a[60:63], v137, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1633) 
v_add_u32 v20, 8, v18 // (op 26737) 

v_bfe_u32 v150, v132, 0, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[98:101], v[30:37], a[64:67], v138, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3117) 
v_bfe_u32 v151, v132, 8, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[98:101], v[38:45], a[68:71], v138, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3079) 
v_bfe_u32 v152, v132, 16, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[98:101], v[46:53], a[72:75], v138, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3041) 
v_bfe_u32 v153, v132, 24, 8 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:101], v[54:61], a[76:79], v138, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3003) 
v_permlane16_swap_b32 v150, v151 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[98:101], v[62:69], a[80:83], v138, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2965) 
v_permlane16_swap_b32 v152, v153 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:101], v[70:77], a[84:87], v138, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2927) 
s_nop 0
v_permlane32_swap_b32 v150, v152 // (op 24838) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[98:101], v[78:85], a[88:91], v138, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2889) 
v_permlane32_swap_b32 v151, v153 // (op 24838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[98:101], v[86:93], a[92:95], v138, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2851) 
s_waitcnt vmcnt(16)
s_barrier  // (op 27830) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[102:105], v[30:37], a[96:99], v139, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4335) 
ds_read_b128 v[154:157], v6 offset:49152 // (op 216) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[102:105], v[38:45], a[100:103], v139, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4297) 
ds_read_b128 v[158:161], v6 offset:49216 // (op 216) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[102:105], v[46:53], a[104:107], v139, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4259) 
ds_read_b128 v[162:165], v6 offset:51200 // (op 324) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[102:105], v[54:61], a[108:111], v139, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4221) 
ds_read_b128 v[166:169], v6 offset:51264 // (op 324) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[102:105], v[62:69], a[112:115], v139, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4183) 
ds_read_b128 v[170:173], v6 offset:53248 // (op 460) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[102:105], v[70:77], a[116:119], v139, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4145) 
ds_read_b128 v[174:177], v6 offset:53312 // (op 460) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[102:105], v[78:85], a[120:123], v139, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4107) 
buffer_load_dwordx2 v[24:25], v20, s[60:63], 0 offen // (op 26737) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[102:105], v[86:93], a[124:127], v139, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4069) 
s_lshl_b32 s11, s32, 6 // (op 26750) 

s_add_u32 s11, s11, 8 // (op 26750) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[106:109], v[30:37], a[128:131], v150, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5553) 
v_add_u32 v20, v18, s11 // (op 26750) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[106:109], v[38:45], a[132:135], v150, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5515) 
buffer_load_dwordx2 v[26:27], v20, s[60:63], 0 offen // (op 26750) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[106:109], v[46:53], a[136:139], v150, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5477) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[106:109], v[54:61], a[140:143], v150, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5439) 
v_add_u32 v19, 8, v19 // (op 27381) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[106:109], v[62:69], a[144:147], v150, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5401) 
v_mov_b32 v18, v21 // (op 26763) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[106:109], v[70:77], a[148:151], v150, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5363) 
s_lshl_b32 s11, s28, 6 // (op 26763) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[106:109], v[78:85], a[152:155], v150, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5325) 
s_add_u32 s14, 8, s11 // (op 26763) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[106:109], v[86:93], a[156:159], v150, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5287) 
v_add_u32 v20, v18, s14 // (op 26763) 


buffer_load_dwordx2 v[22:23], v20, s[56:59], 0 offen // (op 26763) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[110:113], v[30:37], a[160:163], v151, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6771) 
v_add_u32 v20, 8, v18 // (op 26776) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[110:113], v[38:45], a[164:167], v151, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6733) 
buffer_load_dwordx2 v[28:29], v20, s[56:59], 0 offen // (op 26776) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[110:113], v[46:53], a[168:171], v151, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6695) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[110:113], v[54:61], a[172:175], v151, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6657) 
v_add_u32 v21, 8, v21 // (op 27329) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[110:113], v[62:69], a[176:179], v151, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6619) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[110:113], v[70:77], a[180:183], v151, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6581) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[110:113], v[78:85], a[184:187], v151, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6543) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[110:113], v[86:93], a[188:191], v151, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6505) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[114:117], v[30:37], a[192:195], v152, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7989) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[114:117], v[38:45], a[196:199], v152, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7951) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[114:117], v[46:53], a[200:203], v152, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7913) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[114:117], v[54:61], a[204:207], v152, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7875) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[114:117], v[62:69], a[208:211], v152, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7837) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[114:117], v[70:77], a[212:215], v152, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7799) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[114:117], v[78:85], a[216:219], v152, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7761) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[114:117], v[86:93], a[220:223], v152, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7723) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[118:121], v[30:37], a[224:227], v153, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9207) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[118:121], v[38:45], a[228:231], v153, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9169) 
ds_read_b128 v[178:181], v6 offset:55296 // (op 596) Load local data 
ds_read_b128 v[182:185], v6 offset:55360 // (op 596) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[118:121], v[46:53], a[232:235], v153, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9131) 
ds_read_b128 v[186:189], v6 offset:57344 // (op 732) Load local data 
ds_read_b128 v[190:193], v6 offset:57408 // (op 732) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[118:121], v[54:61], a[236:239], v153, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9093) 
ds_read_b128 v[194:197], v6 offset:59392 // (op 868) Load local data 

ds_read_b128 v[198:201], v6 offset:59456 // (op 868) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[62:69], a[240:243], v153, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9055) 
ds_read_b128 v[136:139], v6 offset:61440 // (op 1004) Load local data 
ds_read_b128 v[140:143], v6 offset:61504 // (op 1004) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[70:77], a[244:247], v153, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9017) 
ds_read_b128 v[202:205], v6 offset:63488 // (op 1140) Load local data 
ds_read_b128 v[206:209], v6 offset:63552 // (op 1140) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[78:85], a[248:251], v153, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8979) 
s_waitcnt vmcnt(16)
s_barrier  // (op 27824) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[86:93], a[252:255], v153, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8941) 
v_add_u32 v18, 81920, v3 // (op 214) 


ds_read_b128 v[10:13], v18 // (op 214) Load local data 




v_add_u32 v18, 82944, v3 // (op 1641) 
ds_read_b128 v[94:97], v18 // (op 1641) Load local data 




v_add_u32 v18, 83968, v3 // (op 2859) 
ds_read_b128 v[98:101], v18 // (op 2859) Load local data 




v_add_u32 v18, 84992, v3 // (op 4077) 
ds_read_b128 v[102:105], v18 // (op 4077) Load local data 




v_add_u32 v18, 86016, v3 // (op 5295) 
ds_read_b128 v[106:109], v18 // (op 5295) Load local data 




v_add_u32 v18, 87040, v3 // (op 6513) 
ds_read_b128 v[110:113], v18 // (op 6513) Load local data 




v_add_u32 v18, 88064, v3 // (op 7731) 
ds_read_b128 v[114:117], v18 // (op 7731) Load local data 




v_add_u32 v18, 89088, v3 // (op 8949) 
ds_read_b128 v[144:147], v18 // (op 8949) Load local data 

v_bfe_u32 v148, v135, 0, 8 // (op 24822) 
v_bfe_u32 v149, v135, 8, 8 // (op 24822) 
v_bfe_u32 v150, v135, 16, 8 // (op 24822) 
v_bfe_u32 v151, v135, 24, 8 // (op 24822) 
v_permlane16_swap_b32 v148, v149 // (op 24822) 
s_nop 0
v_permlane16_swap_b32 v150, v151 // (op 24822) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v148, v150 // (op 24822) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v149, v151 // (op 24822) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 27834) 


s_lshl_b64 s[14:15], s[24:25], 5 // (op 26838) 
v_mov_b64 v[126:127], s[14:15] // (op 26838) convert


v_mov_b64 v[134:135], v[128:129] // (op 26838) 
s_add_u32 s11, 49152, s7 // (op 26838) 
s_mov_b32 m0, s11 // (op 26838) reg expression
s_mov_b32 s14, 384 // (op 26838) call()
buffer_load_dwordx4 v134, s[52:55], s14 offen offset: 0 lds // (op 26838) Load value direct to lds
v_add_co_u32 v134, s[18:19], v134, v126 // (op 26838) least significant half
v_addc_co_u32 v135, s[18:19], 0, v135, s[18:19] // (op 26838) most significant half

s_add_u32 m0, m0, 4096 // (op 26838) 
buffer_load_dwordx4 v134, s[52:55], s14 offen offset: 0 lds // (op 26838) Load value direct to lds
v_add_co_u32 v134, s[18:19], v134, v126 // (op 26838) least significant half
v_addc_co_u32 v135, s[18:19], 0, v135, s[18:19] // (op 26838) most significant half

s_add_u32 m0, m0, 4096 // (op 26838) 
buffer_load_dwordx4 v134, s[52:55], s14 offen offset: 0 lds // (op 26838) Load value direct to lds
v_add_co_u32 v134, s[18:19], v134, v126 // (op 26838) least significant half
v_addc_co_u32 v135, s[18:19], 0, v135, s[18:19] // (op 26838) most significant half

s_add_u32 m0, m0, 4096 // (op 26838) 
v_bfe_u32 v210, v123, 0, 8 // (op 25013) 
v_bfe_u32 v211, v123, 8, 8 // (op 25013) 
v_bfe_u32 v212, v123, 16, 8 // (op 25013) 
v_bfe_u32 v213, v123, 24, 8 // (op 25013) 
v_permlane16_swap_b32 v210, v211 // (op 25013) 
s_nop 0
v_permlane16_swap_b32 v212, v213 // (op 25013) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v210, v212 // (op 25013) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v211, v213 // (op 25013) 
buffer_load_dwordx4 v134, s[52:55], s14 offen offset: 0 lds // (op 26838) Load value direct to lds

v_add_co_u32 v134, s[18:19], v134, v126 // (op 26838) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:13], v[154:161], a[0:3], v148, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 221) 
v_addc_co_u32 v135, s[18:19], 0, v135, s[18:19] // (op 26838) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[10:13], v[162:169], a[4:7], v148, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 316) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[10:13], v[170:177], a[8:11], v148, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 452) 
s_add_u32 m0, m0, 4096 // (op 26838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[10:13], v[178:185], a[12:15], v148, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 588) 
buffer_load_dwordx4 v134, s[52:55], s14 offen offset: 0 lds // (op 26838) Load value direct to lds
v_add_co_u32 v134, s[18:19], v134, v126 // (op 26838) least significant half
v_addc_co_u32 v135, s[18:19], 0, v135, s[18:19] // (op 26838) most significant half

s_add_u32 m0, m0, 4096 // (op 26838) 
buffer_load_dwordx4 v134, s[52:55], s14 offen offset: 0 lds // (op 26838) Load value direct to lds
v_add_co_u32 v134, s[18:19], v134, v126 // (op 26838) least significant half
v_addc_co_u32 v135, s[18:19], 0, v135, s[18:19] // (op 26838) most significant half

s_add_u32 m0, m0, 4096 // (op 26838) 
v_bfe_u32 v214, v125, 0, 8 // (op 25045) 
v_bfe_u32 v215, v125, 8, 8 // (op 25045) 
v_bfe_u32 v216, v125, 16, 8 // (op 25045) 
v_bfe_u32 v217, v125, 24, 8 // (op 25045) 
v_permlane16_swap_b32 v214, v215 // (op 25045) 
s_nop 0
v_permlane16_swap_b32 v216, v217 // (op 25045) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v214, v216 // (op 25045) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v215, v217 // (op 25045) 
buffer_load_dwordx4 v134, s[52:55], s14 offen offset: 0 lds // (op 26838) Load value direct to lds

v_add_co_u32 v134, s[18:19], v134, v126 // (op 26838) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[10:13], v[186:193], a[16:19], v148, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 724) 
v_addc_co_u32 v135, s[18:19], 0, v135, s[18:19] // (op 26838) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[10:13], v[194:201], a[20:23], v148, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 860) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[136:143], a[24:27], v148, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 996) 
s_add_u32 m0, m0, 4096 // (op 26838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[10:13], v[202:209], a[28:31], v148, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1132) 
v_bfe_u32 v122, v133, 0, 8 // (op 24853) 
buffer_load_dwordx4 v134, s[52:55], s14 offen offset: 0 lds // (op 26838) Load value direct to lds




v_bfe_u32 v123, v133, 8, 8 // (op 24853) 

v_bfe_u32 v124, v133, 16, 8 // (op 24853) 
v_bfe_u32 v125, v133, 24, 8 // (op 24853) 
v_permlane16_swap_b32 v122, v123 // (op 24853) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[94:97], v[154:161], a[32:35], v149, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1900) 
v_permlane16_swap_b32 v124, v125 // (op 24853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[94:97], v[162:169], a[36:39], v149, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1862) 
v_add_co_u32 v16, vcc, 256, v16 // (op 27032) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[94:97], v[170:177], a[40:43], v149, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1824) 
v_permlane32_swap_b32 v122, v124 // (op 24853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:97], v[178:185], a[44:47], v149, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1786) 
v_permlane32_swap_b32 v123, v125 // (op 24853) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[94:97], v[186:193], a[48:51], v149, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1748) 
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 27032) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[94:97], v[194:201], a[52:55], v149, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1710) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 27827) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[94:97], v[136:143], a[56:59], v149, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1672) 
s_lshl_b64 s[14:15], s[12:13], 6 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[94:97], v[202:209], a[60:63], v149, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1634) 
s_lshl_b64 s[18:19], s[14:15], 2 // (op 26833) 

v_lshrrev_b64 v[152:153], 3, s[18:19] // (op 26833) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[98:101], v[154:161], a[64:67], v150, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3118) 
v_mov_b64 v[126:127], v[130:131] // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[98:101], v[162:169], a[68:71], v150, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3080) 
s_add_u32 s11, 81920, s6 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[98:101], v[170:177], a[72:75], v150, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3042) 
s_mov_b32 m0, s11 // (op 26833) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:101], v[178:185], a[76:79], v150, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3004) 
s_mov_b32 s14, 192 // (op 26833) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[98:101], v[186:193], a[80:83], v150, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2966) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 26833) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:101], v[194:201], a[84:87], v150, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2928) 
v_add_co_u32 v126, s[18:19], v126, v152 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[98:101], v[136:143], a[88:91], v150, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2890) 
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[98:101], v[202:209], a[92:95], v150, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2852) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[102:105], v[154:161], a[96:99], v151, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4336) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[102:105], v[162:169], a[100:103], v151, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4298) 
s_waitcnt vmcnt(17)
s_barrier  // (op 27817) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[102:105], v[170:177], a[104:107], v151, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4260) 
ds_read_b128 v[30:33], v4 // (op 67) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[102:105], v[178:185], a[108:111], v151, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4222) 
ds_read_b128 v[34:37], v4 offset:64 // (op 67) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[102:105], v[186:193], a[112:115], v151, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4184) 
ds_read_b128 v[38:41], v4 offset:2048 // (op 312) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[102:105], v[194:201], a[116:119], v151, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4146) 
ds_read_b128 v[42:45], v4 offset:2112 // (op 312) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[102:105], v[136:143], a[120:123], v151, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4108) 
ds_read_b128 v[46:49], v4 offset:4096 // (op 448) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[102:105], v[202:209], a[124:127], v151, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4070) 
ds_read_b128 v[50:53], v4 offset:4160 // (op 448) Load local data 

buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 26833) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[106:109], v[154:161], a[128:131], v122, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5554) 
v_add_co_u32 v126, s[18:19], v126, v152 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[106:109], v[162:169], a[132:135], v122, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5516) 
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[106:109], v[170:177], a[136:139], v122, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5478) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[106:109], v[178:185], a[140:143], v122, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5440) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[106:109], v[186:193], a[144:147], v122, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5402) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[106:109], v[194:201], a[148:151], v122, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5364) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 26833) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[106:109], v[136:143], a[152:155], v122, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5326) 
v_add_co_u32 v126, s[18:19], v126, v152 // (op 26833) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[106:109], v[202:209], a[156:159], v122, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5288) 
v_addc_co_u32 v127, s[18:19], 0, v127, s[18:19] // (op 26833) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[110:113], v[154:161], a[160:163], v123, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6772) 
s_add_u32 m0, m0, 4096 // (op 26833) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[110:113], v[162:169], a[164:167], v123, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6734) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[110:113], v[170:177], a[168:171], v123, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6696) 
buffer_load_dwordx4 v126, s[48:51], s14 offen offset: 0 lds // (op 26833) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[110:113], v[178:185], a[172:175], v123, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6658) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[110:113], v[186:193], a[176:179], v123, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6620) 
v_add_co_u32 v14, vcc, 128, v14 // (op 26968) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[110:113], v[194:201], a[180:183], v123, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6582) 
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 26968) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[110:113], v[136:143], a[184:187], v123, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6544) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[110:113], v[202:209], a[188:191], v123, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6506) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[114:117], v[154:161], a[192:195], v124, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7990) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[114:117], v[162:169], a[196:199], v124, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7952) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[114:117], v[170:177], a[200:203], v124, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7914) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[114:117], v[178:185], a[204:207], v124, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7876) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[114:117], v[186:193], a[208:211], v124, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7838) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[114:117], v[194:201], a[212:215], v124, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7800) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[114:117], v[136:143], a[216:219], v124, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7762) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[114:117], v[202:209], a[220:223], v124, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7724) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[144:147], v[154:161], a[224:227], v125, v210 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9208) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[144:147], v[162:169], a[228:231], v125, v211 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9170) 
ds_read_b128 v[54:57], v4 offset:6144 // (op 584) Load local data 

ds_read_b128 v[58:61], v4 offset:6208 // (op 584) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[144:147], v[170:177], a[232:235], v125, v212 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9132) 
ds_read_b128 v[62:65], v4 offset:8192 // (op 720) Load local data 

ds_read_b128 v[66:69], v4 offset:8256 // (op 720) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[144:147], v[178:185], a[236:239], v125, v213 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9094) 
ds_read_b128 v[70:73], v4 offset:10240 // (op 856) Load local data 


ds_read_b128 v[74:77], v4 offset:10304 // (op 856) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[144:147], v[186:193], a[240:243], v125, v214 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9056) 
ds_read_b128 v[78:81], v4 offset:12288 // (op 992) Load local data 

ds_read_b128 v[82:85], v4 offset:12352 // (op 992) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[144:147], v[194:201], a[244:247], v125, v215 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 9018) 
ds_read_b128 v[86:89], v4 offset:14336 // (op 1128) Load local data 

ds_read_b128 v[90:93], v4 offset:14400 // (op 1128) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[144:147], v[136:143], a[248:251], v125, v216 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8980) 
s_waitcnt vmcnt(16)
s_barrier  // (op 27801) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[144:147], v[202:209], a[252:255], v125, v217 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8942) 
ds_read_b128 v[10:13], v0 offset:32768 // (op 62) Load local data 







ds_read_b128 v[94:97], v0 offset:33792 // (op 1629) Load local data 



ds_read_b128 v[98:101], v0 offset:34816 // (op 2847) Load local data 



ds_read_b128 v[102:105], v0 offset:35840 // (op 4065) Load local data 



ds_read_b128 v[106:109], v0 offset:36864 // (op 5283) Load local data 



ds_read_b128 v[110:113], v0 offset:37888 // (op 6501) Load local data 



ds_read_b128 v[114:117], v0 offset:38912 // (op 7719) Load local data 



ds_read_b128 v[118:121], v0 offset:39936 // (op 8937) Load local data 
s_ashr_i64 s[14:15], s[36:37], 7 // (op 8907) 
s_mov_b32 s11, s14 // (op 8907) convert

s_and_b32 s14, s11, 4294967294 // (op 8907) 

s_cmp_lt_u32 s10, s14 // (op 8907) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopTop_KLoop_8907 // (op 8907) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopTop_KLoop_8907 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopBottom_KLoop_8907:















s_barrier  // (op 27926) 
v_lshrrev_b32 v14, 2, v1 // (op 26868) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 26868) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 26868) 

s_lshr_b32 s6, s0, 1 // (op 26868) 

v_and_b32 v14, -4, v1 // (op 26868) 
v_sub_u32 v16, v1, v14 // (op 26868) 

v_lshlrev_b32 v14, 5, v16 // (op 26868) 

v_lshl_add_u32 v16, s6, 8, v14 // (op 26868) 


v_mov_b32 v21, 0 // (op 26868) convert
v_mov_b32 v20, v16 // (op 26868) convert

v_mul_lo_u32 v17, v15, s13 // (op 26868) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 26868) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 26868) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 26868) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v20 // (op 26868) least significant half
v_addc_co_u32 v17, s[0:1], v17, v21, s[0:1] // (op 26868) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 26868) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 26868) 

s_mul_i32 s1, s12, s23 // (op 26870) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 26870) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 26870) least significant
s_mul_hi_u32 s7, s12, s22 // (op 26870) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 26870) most significant: sum
s_add_u32 s1, s1, s7 // (op 26870) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 26870) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 26870) 

s_mov_b32 s48, s20 // (op 26870) reg expression
s_mov_b32 s49, s21 // (op 26870) reg expression
s_mov_b32 s50, s0 // (op 26870) reg expression
s_mov_b32 s51, 131072 // (op 26870) call()

v_and_b32 v14, -8, v1 // (op 26998) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 26998) 
v_sub_u32 v15, v1, v14 // (op 26998) 

s_lshr_b32 s6, s0, 1 // (op 26998) 

v_lshlrev_b32 v14, 4, v15 // (op 26998) 

v_lshl_add_u32 v15, s6, 8, v14 // (op 26998) 


v_mov_b32 v19, 0 // (op 26998) convert
v_mov_b32 v18, v15 // (op 26998) convert

v_lshrrev_b32 v14, 3, v1 // (op 26998) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 26998) 

v_mul_lo_u32 v21, v15, s25 // (op 26998) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 26998) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 26998) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 26998) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 26998) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 26998) most significant half



s_mul_i32 s1, s24, s9 // (op 27000) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 27000) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 27000) least significant
s_mul_hi_u32 s7, s24, s8 // (op 27000) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 27000) most significant: sum
s_add_u32 s1, s1, s7 // (op 27000) most significant: sum


s_mov_b32 s52, s26 // (op 27000) reg expression
s_mov_b32 s53, s27 // (op 27000) reg expression
s_mov_b32 s54, s0 // (op 27000) reg expression
s_mov_b32 s55, 131072 // (op 27000) call()

v_and_b32 v18, -4, v1 // (op 27081) 
v_sub_u32 v19, v1, v18 // (op 27081) 

v_lshlrev_b32 v18, 5, v19 // (op 27081) 

v_lshrrev_b32 v19, 2, v1 // (op 27081) 
v_lshl_add_u32 v20, v19, 7, v18 // (op 27081) 


v_lshlrev_b32 v21, 2, v20 // (op 27081) 

v_lshrrev_b32 v18, 3, v21 // (op 27081) 

s_nop 0
v_readfirstlane_b32 s0, v18 // (op 27081) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -8, v1 // (op 27140) 
v_sub_u32 v19, v1, v18 // (op 27140) 

v_lshlrev_b32 v18, 4, v19 // (op 27140) 

v_lshrrev_b32 v19, 3, v1 // (op 27140) 
v_lshl_add_u32 v20, v19, 7, v18 // (op 27140) 


s_nop 0
v_readfirstlane_b32 s1, v20 // (op 27140) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, 63, v1 // (op 27418) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 27418) 
v_lshrrev_b32 v19, 4, v18 // (op 27418) 
s_lshr_b32 s10, s6, 1 // (op 27418) 

v_and_b32 v20, 3, v19 // (op 27418) 

v_and_b32 v19, 15, v1 // (op 27418) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 27418) 


v_and_b32 v19, 4294967168, v1 // (op 27418) 
v_add_u32 v20, v19, v21 // (op 27418) 


v_lshl_add_u32 v19, s16, 8, v20 // (op 27418) 

v_lshrrev_b32 v20, 6, v18 // (op 27418) 

v_lshlrev_b32 v18, 2, v20 // (op 27418) 

v_lshl_add_u32 v20, s10, 3, v18 // (op 27418) 


v_mul_lo_u32 v18, v19, s28 // (op 27418) 
v_add_u32 v18, v18, v20 // (op 27418) 


s_mul_i32 s7, s28, s23 // (op 27420) most significant: low of low * high
s_mul_i32 s10, s29, s22 // (op 27420) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 27420) least significant
s_mul_hi_u32 s11, s28, s22 // (op 27420) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 27420) most significant: sum
s_add_u32 s7, s7, s11 // (op 27420) most significant: sum


s_mov_b32 s56, s30 // (op 27420) reg expression
s_mov_b32 s57, s31 // (op 27420) reg expression
s_mov_b32 s58, s6 // (op 27420) reg expression
s_mov_b32 s59, 131072 // (op 27420) call()

v_and_b32 v19, 63, v1 // (op 27445) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 27445) 
v_lshrrev_b32 v20, 6, v19 // (op 27445) 
s_lshr_b32 s10, s6, 1 // (op 27445) 

v_lshlrev_b32 v21, 2, v20 // (op 27445) 

v_lshl_add_u32 v20, s10, 3, v21 // (op 27445) 


v_lshrrev_b32 v21, 4, v19 // (op 27445) 

v_and_b32 v19, 3, v21 // (op 27445) 

v_and_b32 v21, 15, v1 // (op 27445) 
v_lshl_add_u32 v30, v19, 4, v21 // (op 27445) 


v_lshrrev_b32 v19, 6, v1 // (op 27445) 
v_and_b32 v21, 1, v19 // (op 27445) 

v_lshl_add_u32 v19, v21, 7, v30 // (op 27445) 


v_lshl_add_u32 v21, s17, 8, v19 // (op 27445) 

v_mul_lo_u32 v19, v21, s32 // (op 27445) 

v_add_u32 v21, v20, v19 // (op 27445) 


s_mul_i32 s7, s32, s9 // (op 27447) most significant: low of low * high
s_mul_i32 s10, s33, s8 // (op 27447) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 27447) least significant
s_mul_hi_u32 s11, s32, s8 // (op 27447) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 27447) most significant: sum
s_add_u32 s7, s7, s11 // (op 27447) most significant: sum


s_mov_b32 s60, s34 // (op 27447) reg expression
s_mov_b32 s61, s35 // (op 27447) reg expression
s_mov_b32 s62, s6 // (op 27447) reg expression
s_mov_b32 s63, 131072 // (op 27447) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 2823) 
s_and_b32 s10, s6, 4294967294 // (op 2823) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 2822) 
s_mov_b32 s11, s6 // (op 2822) convert

s_cmp_lt_u32 s10, s11 // (op 2822) 

















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopBottom_KLoopTail_2822 // (op 2822) Condition: Top (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopBottom_KLoopTail_2822 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopTop_KLoopTail_2822:
s_barrier  // (op 27798) 
s_add_u32 s10, s10, 1 // (op 2824) 

s_lshl_b64 s[6:7], s[12:13], 6 // (op 26821) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 26821) 

v_lshrrev_b64 v[10:11], 3, s[14:15] // (op 26821) 


v_mov_b64 v[12:13], v[16:17] // (op 26821) 
v_mov_b64 v[22:23], v[12:13] // (op 26821) 
s_add_u32 s11, 32768, s0 // (op 26821) 
s_mov_b32 m0, s11 // (op 26821) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26821) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26821) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26821) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26821) most significant half

s_add_u32 m0, m0, 4096 // (op 26821) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 26821) Load value direct to lds




v_add_co_u32 v16, s[6:7], 64, v16 // (op 26858) least significant half
v_addc_co_u32 v17, s[6:7], 0, v17, s[6:7] // (op 26858) most significant half

s_barrier  // (op 27814) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 26827) 
v_mov_b64 v[10:11], s[6:7] // (op 26827) convert


v_mov_b64 v[12:13], v[14:15] // (op 26827) 
v_mov_b64 v[22:23], v[12:13] // (op 26827) 
s_mov_b32 m0, s1 // (op 26827) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26827) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26827) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26827) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26827) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26827) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26827) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 26827) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v10 // (op 26827) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 26827) most significant half

s_add_u32 m0, m0, 4096 // (op 26827) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 26827) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 26990) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 26990) most significant half
s_waitcnt vmcnt(8)
s_barrier  // (op 27804) 



ds_read_b128 v[10:13], v0 offset:32768 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 27820) 



ds_read_b128 v[22:25], v4 // (op 166) Load local data 
ds_read_b128 v[26:29], v4 offset:64 // (op 166) Load local data 


v_mov_b32 v19, v18 // (op 168) 
v_mov_b32 v20, v19 // (op 168) reg expression

buffer_load_dwordx2 v[30:31], v20, s[56:59], 0 offen // (op 168) Load value



v_mov_b32 v20, v21 // (op 170) 
v_mov_b32 v32, v20 // (op 170) reg expression

buffer_load_dwordx2 v[34:35], v32, s[60:63], 0 offen // (op 170) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v36, v30, 0, 8 // (op 25190) 



v_bfe_u32 v37, v30, 8, 8 // (op 25190) 
v_bfe_u32 v38, v30, 16, 8 // (op 25190) 
v_bfe_u32 v39, v30, 24, 8 // (op 25190) 
v_permlane16_swap_b32 v36, v37 // (op 25190) 
s_nop 0
v_permlane16_swap_b32 v38, v39 // (op 25190) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v36, v38 // (op 25190) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v37, v39 // (op 25190) 
ds_read_b128 v[40:43], v4 offset:2048 // (op 295) Load local data 

ds_read_b128 v[44:47], v4 offset:2112 // (op 295) Load local data 



ds_read_b128 v[48:51], v4 offset:4096 // (op 431) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v30, v34, 0, 8 // (op 25282) 
v_bfe_u32 v31, v34, 8, 8 // (op 25282) 
v_bfe_u32 v32, v34, 16, 8 // (op 25282) 
v_bfe_u32 v33, v34, 24, 8 // (op 25282) 
v_permlane16_swap_b32 v30, v31 // (op 25282) 
s_nop 0
v_permlane16_swap_b32 v32, v33 // (op 25282) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v30, v32 // (op 25282) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v31, v33 // (op 25282) 
ds_read_b128 v[52:55], v4 offset:4160 // (op 431) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[10:13], v[22:29], a[0:3], v36, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 171) 
ds_read_b128 v[56:59], v4 offset:6144 // (op 567) Load local data 
ds_read_b128 v[60:63], v4 offset:6208 // (op 567) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[10:13], v[40:47], a[4:7], v36, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 298) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[10:13], v[48:55], a[8:11], v36, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 434) 
ds_read_b128 v[64:67], v4 offset:8192 // (op 703) Load local data 
ds_read_b128 v[68:71], v4 offset:8256 // (op 703) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[10:13], v[56:63], a[12:15], v36, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 570) 


s_lshl_b32 s11, s32, 6 // (op 708) 
v_add_u32 v34, v20, s11 // (op 708) 


buffer_load_dwordx2 v[72:73], v34, s[60:63], 0 offen // (op 708) Load value





v_add_u32 v21, 4, v21 // (op 27435) 
ds_read_b128 v[74:77], v4 offset:10240 // (op 839) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v82, v72, 0, 8 // (op 25296) 
v_bfe_u32 v83, v72, 8, 8 // (op 25296) 
v_bfe_u32 v84, v72, 16, 8 // (op 25296) 
v_bfe_u32 v85, v72, 24, 8 // (op 25296) 
v_permlane16_swap_b32 v82, v83 // (op 25296) 
s_nop 0
v_permlane16_swap_b32 v84, v85 // (op 25296) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v82, v84 // (op 25296) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v83, v85 // (op 25296) 
ds_read_b128 v[78:81], v4 offset:10304 // (op 839) Load local data 




ds_read_b128 v[86:89], v4 offset:12288 // (op 975) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[10:13], v[64:71], a[16:19], v36, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 706) 
ds_read_b128 v[90:93], v4 offset:12352 // (op 975) Load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[10:13], v[74:81], a[20:23], v36, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 842) 
ds_read_b128 v[94:97], v4 offset:14336 // (op 1111) Load local data 
ds_read_b128 v[98:101], v4 offset:14400 // (op 1111) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[86:93], a[24:27], v36, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 978) 



ds_read_b128 v[102:105], v0 offset:33792 // (op 1421) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[10:13], v[94:101], a[28:31], v36, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1114) 




ds_read_b128 v[10:13], v0 offset:34816 // (op 2639) Load local data 



ds_read_b128 v[106:109], v0 offset:35840 // (op 3857) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[102:105], v[22:29], a[32:35], v37, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1425) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[102:105], v[48:55], a[40:43], v37, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1479) 
ds_read_b128 v[110:113], v0 offset:36864 // (op 5075) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[102:105], v[56:63], a[44:47], v37, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1506) 
s_lshl_b32 s11, s28, 6 // (op 5084) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:105], v[40:47], a[36:39], v37, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1452) 
v_add_u32 v20, v19, s11 // (op 5084) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:105], v[64:71], a[48:51], v37, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1533) 
buffer_load_dwordx2 v[34:35], v20, s[56:59], 0 offen // (op 5084) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:105], v[86:93], a[56:59], v37, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1587) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:105], v[94:101], a[60:63], v37, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1614) 
v_add_u32 v18, 4, v18 // (op 27408) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:105], v[74:81], a[52:55], v37, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 1560) 
ds_read_b128 v[114:117], v0 offset:37888 // (op 6293) Load local data 




ds_read_b128 v[102:105], v0 offset:38912 // (op 7511) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[10:13], v[22:29], a[64:67], v38, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2643) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[48:55], a[72:75], v38, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2697) 
ds_read_b128 v[118:121], v0 offset:39936 // (op 8729) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:13], v[56:63], a[76:79], v38, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2724) 
s_waitcnt vmcnt(0)
v_bfe_u32 v122, v34, 0, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[40:47], a[68:71], v38, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2670) 
v_bfe_u32 v123, v34, 8, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[10:13], v[64:71], a[80:83], v38, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2751) 
v_bfe_u32 v124, v34, 16, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:13], v[86:93], a[88:91], v38, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2805) 
v_bfe_u32 v125, v34, 24, 8 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:13], v[94:101], a[92:95], v38, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2832) 
v_permlane16_swap_b32 v122, v123 // (op 25204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[10:13], v[74:81], a[84:87], v38, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 2778) 
v_permlane16_swap_b32 v124, v125 // (op 25204) 

s_nop 1
v_permlane32_swap_b32 v122, v124 // (op 25204) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v123, v125 // (op 25204) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[106:109], v[22:29], a[96:99], v39, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3861) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[106:109], v[48:55], a[104:107], v39, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3915) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[106:109], v[56:63], a[108:111], v39, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3942) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[106:109], v[40:47], a[100:103], v39, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3888) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[106:109], v[64:71], a[112:115], v39, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3969) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[106:109], v[86:93], a[120:123], v39, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4023) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[106:109], v[94:101], a[124:127], v39, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 4050) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[106:109], v[74:81], a[116:119], v39, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 3996) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[110:113], v[40:47], a[132:135], v122, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5106) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[110:113], v[74:81], a[148:151], v122, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5214) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[110:113], v[22:29], a[128:131], v122, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5079) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[110:113], v[48:55], a[136:139], v122, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5133) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[110:113], v[56:63], a[140:143], v122, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5160) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[110:113], v[64:71], a[144:147], v122, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5187) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[110:113], v[86:93], a[152:155], v122, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5241) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[110:113], v[94:101], a[156:159], v122, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 5268) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[114:117], v[40:47], a[164:167], v123, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6324) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[114:117], v[74:81], a[180:183], v123, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6432) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[114:117], v[22:29], a[160:163], v123, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6297) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[48:55], a[168:171], v123, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6351) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[56:63], a[172:175], v123, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6378) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[114:117], v[64:71], a[176:179], v123, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6405) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[114:117], v[86:93], a[184:187], v123, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6459) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[114:117], v[94:101], a[188:191], v123, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 6486) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[102:105], v[40:47], a[196:199], v124, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7542) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[102:105], v[74:81], a[212:215], v124, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7650) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[102:105], v[22:29], a[192:195], v124, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7515) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[102:105], v[48:55], a[200:203], v124, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7569) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[102:105], v[56:63], a[204:207], v124, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7596) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[102:105], v[64:71], a[208:211], v124, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7623) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[102:105], v[86:93], a[216:219], v124, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7677) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[102:105], v[94:101], a[220:223], v124, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 7704) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[118:121], v[40:47], a[228:231], v125, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8760) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[74:81], a[244:247], v125, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8868) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[118:121], v[22:29], a[224:227], v125, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8733) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[118:121], v[48:55], a[232:235], v125, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8787) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[118:121], v[56:63], a[236:239], v125, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8814) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[64:71], a[240:243], v125, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8841) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[86:93], a[248:251], v125, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8895) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[94:101], a[252:255], v125, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b000 // (op 8922) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 2822) 
s_mov_b32 s11, s6 // (op 2822) convert

s_cmp_lt_u32 s10, s11 // (op 2822) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopTop_KLoopTail_2822 // (op 2822) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopTop_KLoopTail_2822 if true)
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ForLoopBottom_KLoopTail_2822:





















s_barrier  // (op 27876) 





v_cmp_eq_f32 vcc, 0.00000, v5 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ConditionalFalse_0_beta_116
v_accvgpr_read v10, a0 // (op 42) 
v_accvgpr_read v11, a1 // (op 42) 
v_accvgpr_read v12, a2 // (op 42) 
v_accvgpr_read v13, a3 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 
v_mul_f32 v16, v7, v12 // (op 42) 
v_mul_f32 v17, v7, v13 // (op 42) 

v_lshrrev_b32 v10, 16, v14 // (op 48) 
v_lshrrev_b32 v11, 16, v15 // (op 48) 
v_lshrrev_b32 v12, 16, v16 // (op 48) 
v_lshrrev_b32 v13, 16, v17 // (op 48) 


v_pack_b32_f16 v14, v10, v11 // (op 50) 
v_pack_b32_f16 v15, v12, v13 // (op 50) 

v_mov_b32 v0, v9 // (op 50) reg expression

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 50) Store value



v_accvgpr_read v10, a4 // (op 11114) 
v_accvgpr_read v11, a5 // (op 11114) 
v_accvgpr_read v12, a6 // (op 11114) 
v_accvgpr_read v13, a7 // (op 11114) 
v_mul_f32 v14, v7, v10 // (op 11114) 
v_mul_f32 v15, v7, v11 // (op 11114) 
v_mul_f32 v16, v7, v12 // (op 11114) 
v_mul_f32 v17, v7, v13 // (op 11114) 

v_lshrrev_b32 v10, 16, v14 // (op 11116) 
v_lshrrev_b32 v11, 16, v15 // (op 11116) 
v_lshrrev_b32 v12, 16, v16 // (op 11116) 
v_lshrrev_b32 v13, 16, v17 // (op 11116) 


v_pack_b32_f16 v14, v10, v11 // (op 11117) 
v_pack_b32_f16 v15, v12, v13 // (op 11117) 

s_lshl_b32 s0, s4, 5 // (op 11117) 
v_add_u32 v0, v9, s0 // (op 11117) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11117) Store value



v_accvgpr_read v10, a8 // (op 11128) 
v_accvgpr_read v11, a9 // (op 11128) 
v_accvgpr_read v12, a10 // (op 11128) 
v_accvgpr_read v13, a11 // (op 11128) 
v_mul_f32 v14, v7, v10 // (op 11128) 
v_mul_f32 v15, v7, v11 // (op 11128) 
v_mul_f32 v16, v7, v12 // (op 11128) 
v_mul_f32 v17, v7, v13 // (op 11128) 

v_lshrrev_b32 v10, 16, v14 // (op 11130) 
v_lshrrev_b32 v11, 16, v15 // (op 11130) 
v_lshrrev_b32 v12, 16, v16 // (op 11130) 
v_lshrrev_b32 v13, 16, v17 // (op 11130) 


v_pack_b32_f16 v14, v10, v11 // (op 11131) 
v_pack_b32_f16 v15, v12, v13 // (op 11131) 

s_lshl_b32 s0, s4, 6 // (op 11131) 
v_add_u32 v0, v9, s0 // (op 11131) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11131) Store value



v_accvgpr_read v10, a12 // (op 11142) 
v_accvgpr_read v11, a13 // (op 11142) 
v_accvgpr_read v12, a14 // (op 11142) 
v_accvgpr_read v13, a15 // (op 11142) 
v_mul_f32 v14, v7, v10 // (op 11142) 
v_mul_f32 v15, v7, v11 // (op 11142) 
v_mul_f32 v16, v7, v12 // (op 11142) 
v_mul_f32 v17, v7, v13 // (op 11142) 

v_lshrrev_b32 v10, 16, v14 // (op 11144) 
v_lshrrev_b32 v11, 16, v15 // (op 11144) 
v_lshrrev_b32 v12, 16, v16 // (op 11144) 
v_lshrrev_b32 v13, 16, v17 // (op 11144) 


v_pack_b32_f16 v14, v10, v11 // (op 11145) 
v_pack_b32_f16 v15, v12, v13 // (op 11145) 

s_lshl_b32 s0, s4, 5 // (op 11145) 
s_mul_i32 s1, 3, s0 // (op 11145) 

v_add_u32 v0, v9, s1 // (op 11145) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11145) Store value



v_accvgpr_read v10, a16 // (op 11156) 
v_accvgpr_read v11, a17 // (op 11156) 
v_accvgpr_read v12, a18 // (op 11156) 
v_accvgpr_read v13, a19 // (op 11156) 
v_mul_f32 v14, v7, v10 // (op 11156) 
v_mul_f32 v15, v7, v11 // (op 11156) 
v_mul_f32 v16, v7, v12 // (op 11156) 
v_mul_f32 v17, v7, v13 // (op 11156) 

v_lshrrev_b32 v10, 16, v14 // (op 11158) 
v_lshrrev_b32 v11, 16, v15 // (op 11158) 
v_lshrrev_b32 v12, 16, v16 // (op 11158) 
v_lshrrev_b32 v13, 16, v17 // (op 11158) 


v_pack_b32_f16 v14, v10, v11 // (op 11159) 
v_pack_b32_f16 v15, v12, v13 // (op 11159) 

s_lshl_b32 s0, s4, 7 // (op 11159) 
v_add_u32 v0, v9, s0 // (op 11159) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11159) Store value



v_accvgpr_read v10, a20 // (op 11170) 
v_accvgpr_read v11, a21 // (op 11170) 
v_accvgpr_read v12, a22 // (op 11170) 
v_accvgpr_read v13, a23 // (op 11170) 
v_mul_f32 v14, v7, v10 // (op 11170) 
v_mul_f32 v15, v7, v11 // (op 11170) 
v_mul_f32 v16, v7, v12 // (op 11170) 
v_mul_f32 v17, v7, v13 // (op 11170) 

v_lshrrev_b32 v10, 16, v14 // (op 11172) 
v_lshrrev_b32 v11, 16, v15 // (op 11172) 
v_lshrrev_b32 v12, 16, v16 // (op 11172) 
v_lshrrev_b32 v13, 16, v17 // (op 11172) 


v_pack_b32_f16 v14, v10, v11 // (op 11173) 
v_pack_b32_f16 v15, v12, v13 // (op 11173) 

s_lshl_b32 s0, s4, 5 // (op 11173) 
s_mul_i32 s1, 5, s0 // (op 11173) 

v_add_u32 v0, v9, s1 // (op 11173) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11173) Store value



v_accvgpr_read v10, a24 // (op 11184) 
v_accvgpr_read v11, a25 // (op 11184) 
v_accvgpr_read v12, a26 // (op 11184) 
v_accvgpr_read v13, a27 // (op 11184) 
v_mul_f32 v14, v7, v10 // (op 11184) 
v_mul_f32 v15, v7, v11 // (op 11184) 
v_mul_f32 v16, v7, v12 // (op 11184) 
v_mul_f32 v17, v7, v13 // (op 11184) 

v_lshrrev_b32 v10, 16, v14 // (op 11186) 
v_lshrrev_b32 v11, 16, v15 // (op 11186) 
v_lshrrev_b32 v12, 16, v16 // (op 11186) 
v_lshrrev_b32 v13, 16, v17 // (op 11186) 


v_pack_b32_f16 v14, v10, v11 // (op 11187) 
v_pack_b32_f16 v15, v12, v13 // (op 11187) 

s_lshl_b32 s0, s4, 5 // (op 11187) 
s_mul_i32 s1, 6, s0 // (op 11187) 

v_add_u32 v0, v9, s1 // (op 11187) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11187) Store value



v_accvgpr_read v10, a28 // (op 11198) 
v_accvgpr_read v11, a29 // (op 11198) 
v_accvgpr_read v12, a30 // (op 11198) 
v_accvgpr_read v13, a31 // (op 11198) 
v_mul_f32 v14, v7, v10 // (op 11198) 
v_mul_f32 v15, v7, v11 // (op 11198) 
v_mul_f32 v16, v7, v12 // (op 11198) 
v_mul_f32 v17, v7, v13 // (op 11198) 

v_lshrrev_b32 v10, 16, v14 // (op 11200) 
v_lshrrev_b32 v11, 16, v15 // (op 11200) 
v_lshrrev_b32 v12, 16, v16 // (op 11200) 
v_lshrrev_b32 v13, 16, v17 // (op 11200) 


v_pack_b32_f16 v14, v10, v11 // (op 11201) 
v_pack_b32_f16 v15, v12, v13 // (op 11201) 

s_lshl_b32 s0, s4, 5 // (op 11201) 
s_mul_i32 s1, 7, s0 // (op 11201) 

v_add_u32 v0, v9, s1 // (op 11201) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11201) Store value



v_accvgpr_read v10, a32 // (op 11254) 
v_accvgpr_read v11, a33 // (op 11254) 
v_accvgpr_read v12, a34 // (op 11254) 
v_accvgpr_read v13, a35 // (op 11254) 
v_mul_f32 v14, v7, v10 // (op 11254) 
v_mul_f32 v15, v7, v11 // (op 11254) 
v_mul_f32 v16, v7, v12 // (op 11254) 
v_mul_f32 v17, v7, v13 // (op 11254) 

v_lshrrev_b32 v10, 16, v14 // (op 11256) 
v_lshrrev_b32 v11, 16, v15 // (op 11256) 
v_lshrrev_b32 v12, 16, v16 // (op 11256) 
v_lshrrev_b32 v13, 16, v17 // (op 11256) 


v_pack_b32_f16 v14, v10, v11 // (op 11258) 
v_pack_b32_f16 v15, v12, v13 // (op 11258) 

v_add_u32 v0, 32, v9 // (op 11258) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11258) Store value



v_accvgpr_read v10, a36 // (op 11276) 
v_accvgpr_read v11, a37 // (op 11276) 
v_accvgpr_read v12, a38 // (op 11276) 
v_accvgpr_read v13, a39 // (op 11276) 
v_mul_f32 v14, v7, v10 // (op 11276) 
v_mul_f32 v15, v7, v11 // (op 11276) 
v_mul_f32 v16, v7, v12 // (op 11276) 
v_mul_f32 v17, v7, v13 // (op 11276) 

v_lshrrev_b32 v10, 16, v14 // (op 11278) 
v_lshrrev_b32 v11, 16, v15 // (op 11278) 
v_lshrrev_b32 v12, 16, v16 // (op 11278) 
v_lshrrev_b32 v13, 16, v17 // (op 11278) 


v_pack_b32_f16 v14, v10, v11 // (op 11260) 
v_pack_b32_f16 v15, v12, v13 // (op 11260) 

s_lshl_b32 s0, s4, 5 // (op 11260) 
s_add_u32 s0, s0, 32 // (op 11260) 
v_add_u32 v0, v9, s0 // (op 11260) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11260) Store value



v_accvgpr_read v10, a40 // (op 11282) 
v_accvgpr_read v11, a41 // (op 11282) 
v_accvgpr_read v12, a42 // (op 11282) 
v_accvgpr_read v13, a43 // (op 11282) 
v_mul_f32 v14, v7, v10 // (op 11282) 
v_mul_f32 v15, v7, v11 // (op 11282) 
v_mul_f32 v16, v7, v12 // (op 11282) 
v_mul_f32 v17, v7, v13 // (op 11282) 

v_lshrrev_b32 v10, 16, v14 // (op 11284) 
v_lshrrev_b32 v11, 16, v15 // (op 11284) 
v_lshrrev_b32 v12, 16, v16 // (op 11284) 
v_lshrrev_b32 v13, 16, v17 // (op 11284) 


v_pack_b32_f16 v14, v10, v11 // (op 11262) 
v_pack_b32_f16 v15, v12, v13 // (op 11262) 

s_lshl_b32 s0, s4, 6 // (op 11262) 
s_add_u32 s0, s0, 32 // (op 11262) 
v_add_u32 v0, v9, s0 // (op 11262) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11262) Store value



v_accvgpr_read v10, a44 // (op 11288) 
v_accvgpr_read v11, a45 // (op 11288) 
v_accvgpr_read v12, a46 // (op 11288) 
v_accvgpr_read v13, a47 // (op 11288) 
v_mul_f32 v14, v7, v10 // (op 11288) 
v_mul_f32 v15, v7, v11 // (op 11288) 
v_mul_f32 v16, v7, v12 // (op 11288) 
v_mul_f32 v17, v7, v13 // (op 11288) 

v_lshrrev_b32 v10, 16, v14 // (op 11290) 
v_lshrrev_b32 v11, 16, v15 // (op 11290) 
v_lshrrev_b32 v12, 16, v16 // (op 11290) 
v_lshrrev_b32 v13, 16, v17 // (op 11290) 


v_pack_b32_f16 v14, v10, v11 // (op 11264) 
v_pack_b32_f16 v15, v12, v13 // (op 11264) 

s_lshl_b32 s0, s4, 5 // (op 11264) 
s_mul_i32 s1, 3, s0 // (op 11264) 
s_add_u32 s1, s1, 32 // (op 11264) 

v_add_u32 v0, v9, s1 // (op 11264) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11264) Store value



v_accvgpr_read v10, a48 // (op 11294) 
v_accvgpr_read v11, a49 // (op 11294) 
v_accvgpr_read v12, a50 // (op 11294) 
v_accvgpr_read v13, a51 // (op 11294) 
v_mul_f32 v14, v7, v10 // (op 11294) 
v_mul_f32 v15, v7, v11 // (op 11294) 
v_mul_f32 v16, v7, v12 // (op 11294) 
v_mul_f32 v17, v7, v13 // (op 11294) 

v_lshrrev_b32 v10, 16, v14 // (op 11296) 
v_lshrrev_b32 v11, 16, v15 // (op 11296) 
v_lshrrev_b32 v12, 16, v16 // (op 11296) 
v_lshrrev_b32 v13, 16, v17 // (op 11296) 


v_pack_b32_f16 v14, v10, v11 // (op 11266) 
v_pack_b32_f16 v15, v12, v13 // (op 11266) 

s_lshl_b32 s0, s4, 7 // (op 11266) 
s_add_u32 s0, s0, 32 // (op 11266) 
v_add_u32 v0, v9, s0 // (op 11266) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11266) Store value



v_accvgpr_read v10, a52 // (op 11300) 
v_accvgpr_read v11, a53 // (op 11300) 
v_accvgpr_read v12, a54 // (op 11300) 
v_accvgpr_read v13, a55 // (op 11300) 
v_mul_f32 v14, v7, v10 // (op 11300) 
v_mul_f32 v15, v7, v11 // (op 11300) 
v_mul_f32 v16, v7, v12 // (op 11300) 
v_mul_f32 v17, v7, v13 // (op 11300) 

v_lshrrev_b32 v10, 16, v14 // (op 11302) 
v_lshrrev_b32 v11, 16, v15 // (op 11302) 
v_lshrrev_b32 v12, 16, v16 // (op 11302) 
v_lshrrev_b32 v13, 16, v17 // (op 11302) 


v_pack_b32_f16 v14, v10, v11 // (op 11268) 
v_pack_b32_f16 v15, v12, v13 // (op 11268) 

s_lshl_b32 s0, s4, 5 // (op 11268) 
s_mul_i32 s1, 5, s0 // (op 11268) 
s_add_u32 s1, s1, 32 // (op 11268) 

v_add_u32 v0, v9, s1 // (op 11268) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11268) Store value



v_accvgpr_read v10, a56 // (op 11306) 
v_accvgpr_read v11, a57 // (op 11306) 
v_accvgpr_read v12, a58 // (op 11306) 
v_accvgpr_read v13, a59 // (op 11306) 
v_mul_f32 v14, v7, v10 // (op 11306) 
v_mul_f32 v15, v7, v11 // (op 11306) 
v_mul_f32 v16, v7, v12 // (op 11306) 
v_mul_f32 v17, v7, v13 // (op 11306) 

v_lshrrev_b32 v10, 16, v14 // (op 11308) 
v_lshrrev_b32 v11, 16, v15 // (op 11308) 
v_lshrrev_b32 v12, 16, v16 // (op 11308) 
v_lshrrev_b32 v13, 16, v17 // (op 11308) 


v_pack_b32_f16 v14, v10, v11 // (op 11270) 
v_pack_b32_f16 v15, v12, v13 // (op 11270) 

s_lshl_b32 s0, s4, 5 // (op 11270) 
s_mul_i32 s1, 6, s0 // (op 11270) 
s_add_u32 s1, s1, 32 // (op 11270) 

v_add_u32 v0, v9, s1 // (op 11270) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11270) Store value



v_accvgpr_read v10, a60 // (op 11312) 
v_accvgpr_read v11, a61 // (op 11312) 
v_accvgpr_read v12, a62 // (op 11312) 
v_accvgpr_read v13, a63 // (op 11312) 
v_mul_f32 v14, v7, v10 // (op 11312) 
v_mul_f32 v15, v7, v11 // (op 11312) 
v_mul_f32 v16, v7, v12 // (op 11312) 
v_mul_f32 v17, v7, v13 // (op 11312) 

v_lshrrev_b32 v10, 16, v14 // (op 11314) 
v_lshrrev_b32 v11, 16, v15 // (op 11314) 
v_lshrrev_b32 v12, 16, v16 // (op 11314) 
v_lshrrev_b32 v13, 16, v17 // (op 11314) 


v_pack_b32_f16 v14, v10, v11 // (op 11272) 
v_pack_b32_f16 v15, v12, v13 // (op 11272) 

s_lshl_b32 s0, s4, 5 // (op 11272) 
s_mul_i32 s1, 7, s0 // (op 11272) 
s_add_u32 s1, s1, 32 // (op 11272) 

v_add_u32 v0, v9, s1 // (op 11272) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11272) Store value



v_accvgpr_read v10, a64 // (op 11402) 
v_accvgpr_read v11, a65 // (op 11402) 
v_accvgpr_read v12, a66 // (op 11402) 
v_accvgpr_read v13, a67 // (op 11402) 
v_mul_f32 v14, v7, v10 // (op 11402) 
v_mul_f32 v15, v7, v11 // (op 11402) 
v_mul_f32 v16, v7, v12 // (op 11402) 
v_mul_f32 v17, v7, v13 // (op 11402) 

v_lshrrev_b32 v10, 16, v14 // (op 11404) 
v_lshrrev_b32 v11, 16, v15 // (op 11404) 
v_lshrrev_b32 v12, 16, v16 // (op 11404) 
v_lshrrev_b32 v13, 16, v17 // (op 11404) 


v_pack_b32_f16 v14, v10, v11 // (op 11406) 
v_pack_b32_f16 v15, v12, v13 // (op 11406) 

v_add_u32 v0, 64, v9 // (op 11406) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11406) Store value



v_accvgpr_read v10, a68 // (op 11424) 
v_accvgpr_read v11, a69 // (op 11424) 
v_accvgpr_read v12, a70 // (op 11424) 
v_accvgpr_read v13, a71 // (op 11424) 
v_mul_f32 v14, v7, v10 // (op 11424) 
v_mul_f32 v15, v7, v11 // (op 11424) 
v_mul_f32 v16, v7, v12 // (op 11424) 
v_mul_f32 v17, v7, v13 // (op 11424) 

v_lshrrev_b32 v10, 16, v14 // (op 11426) 
v_lshrrev_b32 v11, 16, v15 // (op 11426) 
v_lshrrev_b32 v12, 16, v16 // (op 11426) 
v_lshrrev_b32 v13, 16, v17 // (op 11426) 


v_pack_b32_f16 v14, v10, v11 // (op 11408) 
v_pack_b32_f16 v15, v12, v13 // (op 11408) 

s_lshl_b32 s0, s4, 5 // (op 11408) 
s_add_u32 s0, s0, 64 // (op 11408) 
v_add_u32 v0, v9, s0 // (op 11408) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11408) Store value



v_accvgpr_read v10, a72 // (op 11430) 
v_accvgpr_read v11, a73 // (op 11430) 
v_accvgpr_read v12, a74 // (op 11430) 
v_accvgpr_read v13, a75 // (op 11430) 
v_mul_f32 v14, v7, v10 // (op 11430) 
v_mul_f32 v15, v7, v11 // (op 11430) 
v_mul_f32 v16, v7, v12 // (op 11430) 
v_mul_f32 v17, v7, v13 // (op 11430) 

v_lshrrev_b32 v10, 16, v14 // (op 11432) 
v_lshrrev_b32 v11, 16, v15 // (op 11432) 
v_lshrrev_b32 v12, 16, v16 // (op 11432) 
v_lshrrev_b32 v13, 16, v17 // (op 11432) 


v_pack_b32_f16 v14, v10, v11 // (op 11410) 
v_pack_b32_f16 v15, v12, v13 // (op 11410) 

s_lshl_b32 s0, s4, 6 // (op 11410) 
s_add_u32 s0, s0, 64 // (op 11410) 
v_add_u32 v0, v9, s0 // (op 11410) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11410) Store value



v_accvgpr_read v10, a76 // (op 11436) 
v_accvgpr_read v11, a77 // (op 11436) 
v_accvgpr_read v12, a78 // (op 11436) 
v_accvgpr_read v13, a79 // (op 11436) 
v_mul_f32 v14, v7, v10 // (op 11436) 
v_mul_f32 v15, v7, v11 // (op 11436) 
v_mul_f32 v16, v7, v12 // (op 11436) 
v_mul_f32 v17, v7, v13 // (op 11436) 

v_lshrrev_b32 v10, 16, v14 // (op 11438) 
v_lshrrev_b32 v11, 16, v15 // (op 11438) 
v_lshrrev_b32 v12, 16, v16 // (op 11438) 
v_lshrrev_b32 v13, 16, v17 // (op 11438) 


v_pack_b32_f16 v14, v10, v11 // (op 11412) 
v_pack_b32_f16 v15, v12, v13 // (op 11412) 

s_lshl_b32 s0, s4, 5 // (op 11412) 
s_mul_i32 s1, 3, s0 // (op 11412) 
s_add_u32 s1, s1, 64 // (op 11412) 

v_add_u32 v0, v9, s1 // (op 11412) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11412) Store value



v_accvgpr_read v10, a80 // (op 11442) 
v_accvgpr_read v11, a81 // (op 11442) 
v_accvgpr_read v12, a82 // (op 11442) 
v_accvgpr_read v13, a83 // (op 11442) 
v_mul_f32 v14, v7, v10 // (op 11442) 
v_mul_f32 v15, v7, v11 // (op 11442) 
v_mul_f32 v16, v7, v12 // (op 11442) 
v_mul_f32 v17, v7, v13 // (op 11442) 

v_lshrrev_b32 v10, 16, v14 // (op 11444) 
v_lshrrev_b32 v11, 16, v15 // (op 11444) 
v_lshrrev_b32 v12, 16, v16 // (op 11444) 
v_lshrrev_b32 v13, 16, v17 // (op 11444) 


v_pack_b32_f16 v14, v10, v11 // (op 11414) 
v_pack_b32_f16 v15, v12, v13 // (op 11414) 

s_lshl_b32 s0, s4, 7 // (op 11414) 
s_add_u32 s0, s0, 64 // (op 11414) 
v_add_u32 v0, v9, s0 // (op 11414) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11414) Store value



v_accvgpr_read v10, a84 // (op 11448) 
v_accvgpr_read v11, a85 // (op 11448) 
v_accvgpr_read v12, a86 // (op 11448) 
v_accvgpr_read v13, a87 // (op 11448) 
v_mul_f32 v14, v7, v10 // (op 11448) 
v_mul_f32 v15, v7, v11 // (op 11448) 
v_mul_f32 v16, v7, v12 // (op 11448) 
v_mul_f32 v17, v7, v13 // (op 11448) 

v_lshrrev_b32 v10, 16, v14 // (op 11450) 
v_lshrrev_b32 v11, 16, v15 // (op 11450) 
v_lshrrev_b32 v12, 16, v16 // (op 11450) 
v_lshrrev_b32 v13, 16, v17 // (op 11450) 


v_pack_b32_f16 v14, v10, v11 // (op 11416) 
v_pack_b32_f16 v15, v12, v13 // (op 11416) 

s_lshl_b32 s0, s4, 5 // (op 11416) 
s_mul_i32 s1, 5, s0 // (op 11416) 
s_add_u32 s1, s1, 64 // (op 11416) 

v_add_u32 v0, v9, s1 // (op 11416) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11416) Store value



v_accvgpr_read v10, a88 // (op 11454) 
v_accvgpr_read v11, a89 // (op 11454) 
v_accvgpr_read v12, a90 // (op 11454) 
v_accvgpr_read v13, a91 // (op 11454) 
v_mul_f32 v14, v7, v10 // (op 11454) 
v_mul_f32 v15, v7, v11 // (op 11454) 
v_mul_f32 v16, v7, v12 // (op 11454) 
v_mul_f32 v17, v7, v13 // (op 11454) 

v_lshrrev_b32 v10, 16, v14 // (op 11456) 
v_lshrrev_b32 v11, 16, v15 // (op 11456) 
v_lshrrev_b32 v12, 16, v16 // (op 11456) 
v_lshrrev_b32 v13, 16, v17 // (op 11456) 


v_pack_b32_f16 v14, v10, v11 // (op 11418) 
v_pack_b32_f16 v15, v12, v13 // (op 11418) 

s_lshl_b32 s0, s4, 5 // (op 11418) 
s_mul_i32 s1, 6, s0 // (op 11418) 
s_add_u32 s1, s1, 64 // (op 11418) 

v_add_u32 v0, v9, s1 // (op 11418) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11418) Store value



v_accvgpr_read v10, a92 // (op 11460) 
v_accvgpr_read v11, a93 // (op 11460) 
v_accvgpr_read v12, a94 // (op 11460) 
v_accvgpr_read v13, a95 // (op 11460) 
v_mul_f32 v14, v7, v10 // (op 11460) 
v_mul_f32 v15, v7, v11 // (op 11460) 
v_mul_f32 v16, v7, v12 // (op 11460) 
v_mul_f32 v17, v7, v13 // (op 11460) 

v_lshrrev_b32 v10, 16, v14 // (op 11462) 
v_lshrrev_b32 v11, 16, v15 // (op 11462) 
v_lshrrev_b32 v12, 16, v16 // (op 11462) 
v_lshrrev_b32 v13, 16, v17 // (op 11462) 


v_pack_b32_f16 v14, v10, v11 // (op 11420) 
v_pack_b32_f16 v15, v12, v13 // (op 11420) 

s_lshl_b32 s0, s4, 5 // (op 11420) 
s_mul_i32 s1, 7, s0 // (op 11420) 
s_add_u32 s1, s1, 64 // (op 11420) 

v_add_u32 v0, v9, s1 // (op 11420) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11420) Store value



v_accvgpr_read v10, a96 // (op 11550) 
v_accvgpr_read v11, a97 // (op 11550) 
v_accvgpr_read v12, a98 // (op 11550) 
v_accvgpr_read v13, a99 // (op 11550) 
v_mul_f32 v14, v7, v10 // (op 11550) 
v_mul_f32 v15, v7, v11 // (op 11550) 
v_mul_f32 v16, v7, v12 // (op 11550) 
v_mul_f32 v17, v7, v13 // (op 11550) 

v_lshrrev_b32 v10, 16, v14 // (op 11552) 
v_lshrrev_b32 v11, 16, v15 // (op 11552) 
v_lshrrev_b32 v12, 16, v16 // (op 11552) 
v_lshrrev_b32 v13, 16, v17 // (op 11552) 


v_pack_b32_f16 v14, v10, v11 // (op 11554) 
v_pack_b32_f16 v15, v12, v13 // (op 11554) 

v_add_u32 v0, 96, v9 // (op 11554) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11554) Store value



v_accvgpr_read v10, a100 // (op 11572) 
v_accvgpr_read v11, a101 // (op 11572) 
v_accvgpr_read v12, a102 // (op 11572) 
v_accvgpr_read v13, a103 // (op 11572) 
v_mul_f32 v14, v7, v10 // (op 11572) 
v_mul_f32 v15, v7, v11 // (op 11572) 
v_mul_f32 v16, v7, v12 // (op 11572) 
v_mul_f32 v17, v7, v13 // (op 11572) 

v_lshrrev_b32 v10, 16, v14 // (op 11574) 
v_lshrrev_b32 v11, 16, v15 // (op 11574) 
v_lshrrev_b32 v12, 16, v16 // (op 11574) 
v_lshrrev_b32 v13, 16, v17 // (op 11574) 


v_pack_b32_f16 v14, v10, v11 // (op 11556) 
v_pack_b32_f16 v15, v12, v13 // (op 11556) 

s_lshl_b32 s0, s4, 5 // (op 11556) 
s_add_u32 s0, s0, 96 // (op 11556) 
v_add_u32 v0, v9, s0 // (op 11556) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11556) Store value



v_accvgpr_read v10, a104 // (op 11578) 
v_accvgpr_read v11, a105 // (op 11578) 
v_accvgpr_read v12, a106 // (op 11578) 
v_accvgpr_read v13, a107 // (op 11578) 
v_mul_f32 v14, v7, v10 // (op 11578) 
v_mul_f32 v15, v7, v11 // (op 11578) 
v_mul_f32 v16, v7, v12 // (op 11578) 
v_mul_f32 v17, v7, v13 // (op 11578) 

v_lshrrev_b32 v10, 16, v14 // (op 11580) 
v_lshrrev_b32 v11, 16, v15 // (op 11580) 
v_lshrrev_b32 v12, 16, v16 // (op 11580) 
v_lshrrev_b32 v13, 16, v17 // (op 11580) 


v_pack_b32_f16 v14, v10, v11 // (op 11558) 
v_pack_b32_f16 v15, v12, v13 // (op 11558) 

s_lshl_b32 s0, s4, 6 // (op 11558) 
s_add_u32 s0, s0, 96 // (op 11558) 
v_add_u32 v0, v9, s0 // (op 11558) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11558) Store value



v_accvgpr_read v10, a108 // (op 11584) 
v_accvgpr_read v11, a109 // (op 11584) 
v_accvgpr_read v12, a110 // (op 11584) 
v_accvgpr_read v13, a111 // (op 11584) 
v_mul_f32 v14, v7, v10 // (op 11584) 
v_mul_f32 v15, v7, v11 // (op 11584) 
v_mul_f32 v16, v7, v12 // (op 11584) 
v_mul_f32 v17, v7, v13 // (op 11584) 

v_lshrrev_b32 v10, 16, v14 // (op 11586) 
v_lshrrev_b32 v11, 16, v15 // (op 11586) 
v_lshrrev_b32 v12, 16, v16 // (op 11586) 
v_lshrrev_b32 v13, 16, v17 // (op 11586) 


v_pack_b32_f16 v14, v10, v11 // (op 11560) 
v_pack_b32_f16 v15, v12, v13 // (op 11560) 

s_lshl_b32 s0, s4, 5 // (op 11560) 
s_mul_i32 s1, 3, s0 // (op 11560) 
s_add_u32 s1, s1, 96 // (op 11560) 

v_add_u32 v0, v9, s1 // (op 11560) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11560) Store value



v_accvgpr_read v10, a112 // (op 11590) 
v_accvgpr_read v11, a113 // (op 11590) 
v_accvgpr_read v12, a114 // (op 11590) 
v_accvgpr_read v13, a115 // (op 11590) 
v_mul_f32 v14, v7, v10 // (op 11590) 
v_mul_f32 v15, v7, v11 // (op 11590) 
v_mul_f32 v16, v7, v12 // (op 11590) 
v_mul_f32 v17, v7, v13 // (op 11590) 

v_lshrrev_b32 v10, 16, v14 // (op 11592) 
v_lshrrev_b32 v11, 16, v15 // (op 11592) 
v_lshrrev_b32 v12, 16, v16 // (op 11592) 
v_lshrrev_b32 v13, 16, v17 // (op 11592) 


v_pack_b32_f16 v14, v10, v11 // (op 11562) 
v_pack_b32_f16 v15, v12, v13 // (op 11562) 

s_lshl_b32 s0, s4, 7 // (op 11562) 
s_add_u32 s0, s0, 96 // (op 11562) 
v_add_u32 v0, v9, s0 // (op 11562) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11562) Store value



v_accvgpr_read v10, a116 // (op 11596) 
v_accvgpr_read v11, a117 // (op 11596) 
v_accvgpr_read v12, a118 // (op 11596) 
v_accvgpr_read v13, a119 // (op 11596) 
v_mul_f32 v14, v7, v10 // (op 11596) 
v_mul_f32 v15, v7, v11 // (op 11596) 
v_mul_f32 v16, v7, v12 // (op 11596) 
v_mul_f32 v17, v7, v13 // (op 11596) 

v_lshrrev_b32 v10, 16, v14 // (op 11598) 
v_lshrrev_b32 v11, 16, v15 // (op 11598) 
v_lshrrev_b32 v12, 16, v16 // (op 11598) 
v_lshrrev_b32 v13, 16, v17 // (op 11598) 


v_pack_b32_f16 v14, v10, v11 // (op 11564) 
v_pack_b32_f16 v15, v12, v13 // (op 11564) 

s_lshl_b32 s0, s4, 5 // (op 11564) 
s_mul_i32 s1, 5, s0 // (op 11564) 
s_add_u32 s1, s1, 96 // (op 11564) 

v_add_u32 v0, v9, s1 // (op 11564) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11564) Store value



v_accvgpr_read v10, a120 // (op 11602) 
v_accvgpr_read v11, a121 // (op 11602) 
v_accvgpr_read v12, a122 // (op 11602) 
v_accvgpr_read v13, a123 // (op 11602) 
v_mul_f32 v14, v7, v10 // (op 11602) 
v_mul_f32 v15, v7, v11 // (op 11602) 
v_mul_f32 v16, v7, v12 // (op 11602) 
v_mul_f32 v17, v7, v13 // (op 11602) 

v_lshrrev_b32 v10, 16, v14 // (op 11604) 
v_lshrrev_b32 v11, 16, v15 // (op 11604) 
v_lshrrev_b32 v12, 16, v16 // (op 11604) 
v_lshrrev_b32 v13, 16, v17 // (op 11604) 


v_pack_b32_f16 v14, v10, v11 // (op 11566) 
v_pack_b32_f16 v15, v12, v13 // (op 11566) 

s_lshl_b32 s0, s4, 5 // (op 11566) 
s_mul_i32 s1, 6, s0 // (op 11566) 
s_add_u32 s1, s1, 96 // (op 11566) 

v_add_u32 v0, v9, s1 // (op 11566) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11566) Store value



v_accvgpr_read v10, a124 // (op 11608) 
v_accvgpr_read v11, a125 // (op 11608) 
v_accvgpr_read v12, a126 // (op 11608) 
v_accvgpr_read v13, a127 // (op 11608) 
v_mul_f32 v14, v7, v10 // (op 11608) 
v_mul_f32 v15, v7, v11 // (op 11608) 
v_mul_f32 v16, v7, v12 // (op 11608) 
v_mul_f32 v17, v7, v13 // (op 11608) 

v_lshrrev_b32 v10, 16, v14 // (op 11610) 
v_lshrrev_b32 v11, 16, v15 // (op 11610) 
v_lshrrev_b32 v12, 16, v16 // (op 11610) 
v_lshrrev_b32 v13, 16, v17 // (op 11610) 


v_pack_b32_f16 v14, v10, v11 // (op 11568) 
v_pack_b32_f16 v15, v12, v13 // (op 11568) 

s_lshl_b32 s0, s4, 5 // (op 11568) 
s_mul_i32 s1, 7, s0 // (op 11568) 
s_add_u32 s1, s1, 96 // (op 11568) 

v_add_u32 v0, v9, s1 // (op 11568) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11568) Store value



v_accvgpr_read v10, a128 // (op 11698) 
v_accvgpr_read v11, a129 // (op 11698) 
v_accvgpr_read v12, a130 // (op 11698) 
v_accvgpr_read v13, a131 // (op 11698) 
v_mul_f32 v14, v7, v10 // (op 11698) 
v_mul_f32 v15, v7, v11 // (op 11698) 
v_mul_f32 v16, v7, v12 // (op 11698) 
v_mul_f32 v17, v7, v13 // (op 11698) 

v_lshrrev_b32 v10, 16, v14 // (op 11700) 
v_lshrrev_b32 v11, 16, v15 // (op 11700) 
v_lshrrev_b32 v12, 16, v16 // (op 11700) 
v_lshrrev_b32 v13, 16, v17 // (op 11700) 


v_pack_b32_f16 v14, v10, v11 // (op 11702) 
v_pack_b32_f16 v15, v12, v13 // (op 11702) 

v_add_u32 v0, 128, v9 // (op 11702) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11702) Store value



v_accvgpr_read v10, a132 // (op 11720) 
v_accvgpr_read v11, a133 // (op 11720) 
v_accvgpr_read v12, a134 // (op 11720) 
v_accvgpr_read v13, a135 // (op 11720) 
v_mul_f32 v14, v7, v10 // (op 11720) 
v_mul_f32 v15, v7, v11 // (op 11720) 
v_mul_f32 v16, v7, v12 // (op 11720) 
v_mul_f32 v17, v7, v13 // (op 11720) 

v_lshrrev_b32 v10, 16, v14 // (op 11722) 
v_lshrrev_b32 v11, 16, v15 // (op 11722) 
v_lshrrev_b32 v12, 16, v16 // (op 11722) 
v_lshrrev_b32 v13, 16, v17 // (op 11722) 


v_pack_b32_f16 v14, v10, v11 // (op 11704) 
v_pack_b32_f16 v15, v12, v13 // (op 11704) 

s_lshl_b32 s0, s4, 5 // (op 11704) 
s_add_u32 s0, s0, 128 // (op 11704) 
v_add_u32 v0, v9, s0 // (op 11704) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11704) Store value



v_accvgpr_read v10, a136 // (op 11726) 
v_accvgpr_read v11, a137 // (op 11726) 
v_accvgpr_read v12, a138 // (op 11726) 
v_accvgpr_read v13, a139 // (op 11726) 
v_mul_f32 v14, v7, v10 // (op 11726) 
v_mul_f32 v15, v7, v11 // (op 11726) 
v_mul_f32 v16, v7, v12 // (op 11726) 
v_mul_f32 v17, v7, v13 // (op 11726) 

v_lshrrev_b32 v10, 16, v14 // (op 11728) 
v_lshrrev_b32 v11, 16, v15 // (op 11728) 
v_lshrrev_b32 v12, 16, v16 // (op 11728) 
v_lshrrev_b32 v13, 16, v17 // (op 11728) 


v_pack_b32_f16 v14, v10, v11 // (op 11706) 
v_pack_b32_f16 v15, v12, v13 // (op 11706) 

s_lshl_b32 s0, s4, 6 // (op 11706) 
s_add_u32 s0, s0, 128 // (op 11706) 
v_add_u32 v0, v9, s0 // (op 11706) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11706) Store value



v_accvgpr_read v10, a140 // (op 11732) 
v_accvgpr_read v11, a141 // (op 11732) 
v_accvgpr_read v12, a142 // (op 11732) 
v_accvgpr_read v13, a143 // (op 11732) 
v_mul_f32 v14, v7, v10 // (op 11732) 
v_mul_f32 v15, v7, v11 // (op 11732) 
v_mul_f32 v16, v7, v12 // (op 11732) 
v_mul_f32 v17, v7, v13 // (op 11732) 

v_lshrrev_b32 v10, 16, v14 // (op 11734) 
v_lshrrev_b32 v11, 16, v15 // (op 11734) 
v_lshrrev_b32 v12, 16, v16 // (op 11734) 
v_lshrrev_b32 v13, 16, v17 // (op 11734) 


v_pack_b32_f16 v14, v10, v11 // (op 11708) 
v_pack_b32_f16 v15, v12, v13 // (op 11708) 

s_lshl_b32 s0, s4, 5 // (op 11708) 
s_mul_i32 s1, 3, s0 // (op 11708) 
s_add_u32 s1, s1, 128 // (op 11708) 

v_add_u32 v0, v9, s1 // (op 11708) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11708) Store value



v_accvgpr_read v10, a144 // (op 11738) 
v_accvgpr_read v11, a145 // (op 11738) 
v_accvgpr_read v12, a146 // (op 11738) 
v_accvgpr_read v13, a147 // (op 11738) 
v_mul_f32 v14, v7, v10 // (op 11738) 
v_mul_f32 v15, v7, v11 // (op 11738) 
v_mul_f32 v16, v7, v12 // (op 11738) 
v_mul_f32 v17, v7, v13 // (op 11738) 

v_lshrrev_b32 v10, 16, v14 // (op 11740) 
v_lshrrev_b32 v11, 16, v15 // (op 11740) 
v_lshrrev_b32 v12, 16, v16 // (op 11740) 
v_lshrrev_b32 v13, 16, v17 // (op 11740) 


v_pack_b32_f16 v14, v10, v11 // (op 11710) 
v_pack_b32_f16 v15, v12, v13 // (op 11710) 

s_lshl_b32 s0, s4, 7 // (op 11710) 
s_add_u32 s0, s0, 128 // (op 11710) 
v_add_u32 v0, v9, s0 // (op 11710) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11710) Store value



v_accvgpr_read v10, a148 // (op 11744) 
v_accvgpr_read v11, a149 // (op 11744) 
v_accvgpr_read v12, a150 // (op 11744) 
v_accvgpr_read v13, a151 // (op 11744) 
v_mul_f32 v14, v7, v10 // (op 11744) 
v_mul_f32 v15, v7, v11 // (op 11744) 
v_mul_f32 v16, v7, v12 // (op 11744) 
v_mul_f32 v17, v7, v13 // (op 11744) 

v_lshrrev_b32 v10, 16, v14 // (op 11746) 
v_lshrrev_b32 v11, 16, v15 // (op 11746) 
v_lshrrev_b32 v12, 16, v16 // (op 11746) 
v_lshrrev_b32 v13, 16, v17 // (op 11746) 


v_pack_b32_f16 v14, v10, v11 // (op 11712) 
v_pack_b32_f16 v15, v12, v13 // (op 11712) 

s_lshl_b32 s0, s4, 5 // (op 11712) 
s_mul_i32 s1, 5, s0 // (op 11712) 
s_add_u32 s1, s1, 128 // (op 11712) 

v_add_u32 v0, v9, s1 // (op 11712) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11712) Store value



v_accvgpr_read v10, a152 // (op 11750) 
v_accvgpr_read v11, a153 // (op 11750) 
v_accvgpr_read v12, a154 // (op 11750) 
v_accvgpr_read v13, a155 // (op 11750) 
v_mul_f32 v14, v7, v10 // (op 11750) 
v_mul_f32 v15, v7, v11 // (op 11750) 
v_mul_f32 v16, v7, v12 // (op 11750) 
v_mul_f32 v17, v7, v13 // (op 11750) 

v_lshrrev_b32 v10, 16, v14 // (op 11752) 
v_lshrrev_b32 v11, 16, v15 // (op 11752) 
v_lshrrev_b32 v12, 16, v16 // (op 11752) 
v_lshrrev_b32 v13, 16, v17 // (op 11752) 


v_pack_b32_f16 v14, v10, v11 // (op 11714) 
v_pack_b32_f16 v15, v12, v13 // (op 11714) 

s_lshl_b32 s0, s4, 5 // (op 11714) 
s_mul_i32 s1, 6, s0 // (op 11714) 
s_add_u32 s1, s1, 128 // (op 11714) 

v_add_u32 v0, v9, s1 // (op 11714) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11714) Store value



v_accvgpr_read v10, a156 // (op 11756) 
v_accvgpr_read v11, a157 // (op 11756) 
v_accvgpr_read v12, a158 // (op 11756) 
v_accvgpr_read v13, a159 // (op 11756) 
v_mul_f32 v14, v7, v10 // (op 11756) 
v_mul_f32 v15, v7, v11 // (op 11756) 
v_mul_f32 v16, v7, v12 // (op 11756) 
v_mul_f32 v17, v7, v13 // (op 11756) 

v_lshrrev_b32 v10, 16, v14 // (op 11758) 
v_lshrrev_b32 v11, 16, v15 // (op 11758) 
v_lshrrev_b32 v12, 16, v16 // (op 11758) 
v_lshrrev_b32 v13, 16, v17 // (op 11758) 


v_pack_b32_f16 v14, v10, v11 // (op 11716) 
v_pack_b32_f16 v15, v12, v13 // (op 11716) 

s_lshl_b32 s0, s4, 5 // (op 11716) 
s_mul_i32 s1, 7, s0 // (op 11716) 
s_add_u32 s1, s1, 128 // (op 11716) 

v_add_u32 v0, v9, s1 // (op 11716) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11716) Store value



v_accvgpr_read v10, a160 // (op 11846) 
v_accvgpr_read v11, a161 // (op 11846) 
v_accvgpr_read v12, a162 // (op 11846) 
v_accvgpr_read v13, a163 // (op 11846) 
v_mul_f32 v14, v7, v10 // (op 11846) 
v_mul_f32 v15, v7, v11 // (op 11846) 
v_mul_f32 v16, v7, v12 // (op 11846) 
v_mul_f32 v17, v7, v13 // (op 11846) 

v_lshrrev_b32 v10, 16, v14 // (op 11848) 
v_lshrrev_b32 v11, 16, v15 // (op 11848) 
v_lshrrev_b32 v12, 16, v16 // (op 11848) 
v_lshrrev_b32 v13, 16, v17 // (op 11848) 


v_pack_b32_f16 v14, v10, v11 // (op 11850) 
v_pack_b32_f16 v15, v12, v13 // (op 11850) 

v_add_u32 v0, 160, v9 // (op 11850) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11850) Store value



v_accvgpr_read v10, a164 // (op 11868) 
v_accvgpr_read v11, a165 // (op 11868) 
v_accvgpr_read v12, a166 // (op 11868) 
v_accvgpr_read v13, a167 // (op 11868) 
v_mul_f32 v14, v7, v10 // (op 11868) 
v_mul_f32 v15, v7, v11 // (op 11868) 
v_mul_f32 v16, v7, v12 // (op 11868) 
v_mul_f32 v17, v7, v13 // (op 11868) 

v_lshrrev_b32 v10, 16, v14 // (op 11870) 
v_lshrrev_b32 v11, 16, v15 // (op 11870) 
v_lshrrev_b32 v12, 16, v16 // (op 11870) 
v_lshrrev_b32 v13, 16, v17 // (op 11870) 


v_pack_b32_f16 v14, v10, v11 // (op 11852) 
v_pack_b32_f16 v15, v12, v13 // (op 11852) 

s_lshl_b32 s0, s4, 5 // (op 11852) 
s_add_u32 s0, s0, 160 // (op 11852) 
v_add_u32 v0, v9, s0 // (op 11852) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11852) Store value



v_accvgpr_read v10, a168 // (op 11874) 
v_accvgpr_read v11, a169 // (op 11874) 
v_accvgpr_read v12, a170 // (op 11874) 
v_accvgpr_read v13, a171 // (op 11874) 
v_mul_f32 v14, v7, v10 // (op 11874) 
v_mul_f32 v15, v7, v11 // (op 11874) 
v_mul_f32 v16, v7, v12 // (op 11874) 
v_mul_f32 v17, v7, v13 // (op 11874) 

v_lshrrev_b32 v10, 16, v14 // (op 11876) 
v_lshrrev_b32 v11, 16, v15 // (op 11876) 
v_lshrrev_b32 v12, 16, v16 // (op 11876) 
v_lshrrev_b32 v13, 16, v17 // (op 11876) 


v_pack_b32_f16 v14, v10, v11 // (op 11854) 
v_pack_b32_f16 v15, v12, v13 // (op 11854) 

s_lshl_b32 s0, s4, 6 // (op 11854) 
s_add_u32 s0, s0, 160 // (op 11854) 
v_add_u32 v0, v9, s0 // (op 11854) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11854) Store value



v_accvgpr_read v10, a172 // (op 11880) 
v_accvgpr_read v11, a173 // (op 11880) 
v_accvgpr_read v12, a174 // (op 11880) 
v_accvgpr_read v13, a175 // (op 11880) 
v_mul_f32 v14, v7, v10 // (op 11880) 
v_mul_f32 v15, v7, v11 // (op 11880) 
v_mul_f32 v16, v7, v12 // (op 11880) 
v_mul_f32 v17, v7, v13 // (op 11880) 

v_lshrrev_b32 v10, 16, v14 // (op 11882) 
v_lshrrev_b32 v11, 16, v15 // (op 11882) 
v_lshrrev_b32 v12, 16, v16 // (op 11882) 
v_lshrrev_b32 v13, 16, v17 // (op 11882) 


v_pack_b32_f16 v14, v10, v11 // (op 11856) 
v_pack_b32_f16 v15, v12, v13 // (op 11856) 

s_lshl_b32 s0, s4, 5 // (op 11856) 
s_mul_i32 s1, 3, s0 // (op 11856) 
s_add_u32 s1, s1, 160 // (op 11856) 

v_add_u32 v0, v9, s1 // (op 11856) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11856) Store value



v_accvgpr_read v10, a176 // (op 11886) 
v_accvgpr_read v11, a177 // (op 11886) 
v_accvgpr_read v12, a178 // (op 11886) 
v_accvgpr_read v13, a179 // (op 11886) 
v_mul_f32 v14, v7, v10 // (op 11886) 
v_mul_f32 v15, v7, v11 // (op 11886) 
v_mul_f32 v16, v7, v12 // (op 11886) 
v_mul_f32 v17, v7, v13 // (op 11886) 

v_lshrrev_b32 v10, 16, v14 // (op 11888) 
v_lshrrev_b32 v11, 16, v15 // (op 11888) 
v_lshrrev_b32 v12, 16, v16 // (op 11888) 
v_lshrrev_b32 v13, 16, v17 // (op 11888) 


v_pack_b32_f16 v14, v10, v11 // (op 11858) 
v_pack_b32_f16 v15, v12, v13 // (op 11858) 

s_lshl_b32 s0, s4, 7 // (op 11858) 
s_add_u32 s0, s0, 160 // (op 11858) 
v_add_u32 v0, v9, s0 // (op 11858) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11858) Store value



v_accvgpr_read v10, a180 // (op 11892) 
v_accvgpr_read v11, a181 // (op 11892) 
v_accvgpr_read v12, a182 // (op 11892) 
v_accvgpr_read v13, a183 // (op 11892) 
v_mul_f32 v14, v7, v10 // (op 11892) 
v_mul_f32 v15, v7, v11 // (op 11892) 
v_mul_f32 v16, v7, v12 // (op 11892) 
v_mul_f32 v17, v7, v13 // (op 11892) 

v_lshrrev_b32 v10, 16, v14 // (op 11894) 
v_lshrrev_b32 v11, 16, v15 // (op 11894) 
v_lshrrev_b32 v12, 16, v16 // (op 11894) 
v_lshrrev_b32 v13, 16, v17 // (op 11894) 


v_pack_b32_f16 v14, v10, v11 // (op 11860) 
v_pack_b32_f16 v15, v12, v13 // (op 11860) 

s_lshl_b32 s0, s4, 5 // (op 11860) 
s_mul_i32 s1, 5, s0 // (op 11860) 
s_add_u32 s1, s1, 160 // (op 11860) 

v_add_u32 v0, v9, s1 // (op 11860) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11860) Store value



v_accvgpr_read v10, a184 // (op 11898) 
v_accvgpr_read v11, a185 // (op 11898) 
v_accvgpr_read v12, a186 // (op 11898) 
v_accvgpr_read v13, a187 // (op 11898) 
v_mul_f32 v14, v7, v10 // (op 11898) 
v_mul_f32 v15, v7, v11 // (op 11898) 
v_mul_f32 v16, v7, v12 // (op 11898) 
v_mul_f32 v17, v7, v13 // (op 11898) 

v_lshrrev_b32 v10, 16, v14 // (op 11900) 
v_lshrrev_b32 v11, 16, v15 // (op 11900) 
v_lshrrev_b32 v12, 16, v16 // (op 11900) 
v_lshrrev_b32 v13, 16, v17 // (op 11900) 


v_pack_b32_f16 v14, v10, v11 // (op 11862) 
v_pack_b32_f16 v15, v12, v13 // (op 11862) 

s_lshl_b32 s0, s4, 5 // (op 11862) 
s_mul_i32 s1, 6, s0 // (op 11862) 
s_add_u32 s1, s1, 160 // (op 11862) 

v_add_u32 v0, v9, s1 // (op 11862) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11862) Store value



v_accvgpr_read v10, a188 // (op 11904) 
v_accvgpr_read v11, a189 // (op 11904) 
v_accvgpr_read v12, a190 // (op 11904) 
v_accvgpr_read v13, a191 // (op 11904) 
v_mul_f32 v14, v7, v10 // (op 11904) 
v_mul_f32 v15, v7, v11 // (op 11904) 
v_mul_f32 v16, v7, v12 // (op 11904) 
v_mul_f32 v17, v7, v13 // (op 11904) 

v_lshrrev_b32 v10, 16, v14 // (op 11906) 
v_lshrrev_b32 v11, 16, v15 // (op 11906) 
v_lshrrev_b32 v12, 16, v16 // (op 11906) 
v_lshrrev_b32 v13, 16, v17 // (op 11906) 


v_pack_b32_f16 v14, v10, v11 // (op 11864) 
v_pack_b32_f16 v15, v12, v13 // (op 11864) 

s_lshl_b32 s0, s4, 5 // (op 11864) 
s_mul_i32 s1, 7, s0 // (op 11864) 
s_add_u32 s1, s1, 160 // (op 11864) 

v_add_u32 v0, v9, s1 // (op 11864) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11864) Store value



v_accvgpr_read v10, a192 // (op 11994) 
v_accvgpr_read v11, a193 // (op 11994) 
v_accvgpr_read v12, a194 // (op 11994) 
v_accvgpr_read v13, a195 // (op 11994) 
v_mul_f32 v14, v7, v10 // (op 11994) 
v_mul_f32 v15, v7, v11 // (op 11994) 
v_mul_f32 v16, v7, v12 // (op 11994) 
v_mul_f32 v17, v7, v13 // (op 11994) 

v_lshrrev_b32 v10, 16, v14 // (op 11996) 
v_lshrrev_b32 v11, 16, v15 // (op 11996) 
v_lshrrev_b32 v12, 16, v16 // (op 11996) 
v_lshrrev_b32 v13, 16, v17 // (op 11996) 


v_pack_b32_f16 v14, v10, v11 // (op 11998) 
v_pack_b32_f16 v15, v12, v13 // (op 11998) 

v_add_u32 v0, 192, v9 // (op 11998) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 11998) Store value



v_accvgpr_read v10, a196 // (op 12016) 
v_accvgpr_read v11, a197 // (op 12016) 
v_accvgpr_read v12, a198 // (op 12016) 
v_accvgpr_read v13, a199 // (op 12016) 
v_mul_f32 v14, v7, v10 // (op 12016) 
v_mul_f32 v15, v7, v11 // (op 12016) 
v_mul_f32 v16, v7, v12 // (op 12016) 
v_mul_f32 v17, v7, v13 // (op 12016) 

v_lshrrev_b32 v10, 16, v14 // (op 12018) 
v_lshrrev_b32 v11, 16, v15 // (op 12018) 
v_lshrrev_b32 v12, 16, v16 // (op 12018) 
v_lshrrev_b32 v13, 16, v17 // (op 12018) 


v_pack_b32_f16 v14, v10, v11 // (op 12000) 
v_pack_b32_f16 v15, v12, v13 // (op 12000) 

s_lshl_b32 s0, s4, 5 // (op 12000) 
s_add_u32 s0, s0, 192 // (op 12000) 
v_add_u32 v0, v9, s0 // (op 12000) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12000) Store value



v_accvgpr_read v10, a200 // (op 12022) 
v_accvgpr_read v11, a201 // (op 12022) 
v_accvgpr_read v12, a202 // (op 12022) 
v_accvgpr_read v13, a203 // (op 12022) 
v_mul_f32 v14, v7, v10 // (op 12022) 
v_mul_f32 v15, v7, v11 // (op 12022) 
v_mul_f32 v16, v7, v12 // (op 12022) 
v_mul_f32 v17, v7, v13 // (op 12022) 

v_lshrrev_b32 v10, 16, v14 // (op 12024) 
v_lshrrev_b32 v11, 16, v15 // (op 12024) 
v_lshrrev_b32 v12, 16, v16 // (op 12024) 
v_lshrrev_b32 v13, 16, v17 // (op 12024) 


v_pack_b32_f16 v14, v10, v11 // (op 12002) 
v_pack_b32_f16 v15, v12, v13 // (op 12002) 

s_lshl_b32 s0, s4, 6 // (op 12002) 
s_add_u32 s0, s0, 192 // (op 12002) 
v_add_u32 v0, v9, s0 // (op 12002) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12002) Store value



v_accvgpr_read v10, a204 // (op 12028) 
v_accvgpr_read v11, a205 // (op 12028) 
v_accvgpr_read v12, a206 // (op 12028) 
v_accvgpr_read v13, a207 // (op 12028) 
v_mul_f32 v14, v7, v10 // (op 12028) 
v_mul_f32 v15, v7, v11 // (op 12028) 
v_mul_f32 v16, v7, v12 // (op 12028) 
v_mul_f32 v17, v7, v13 // (op 12028) 

v_lshrrev_b32 v10, 16, v14 // (op 12030) 
v_lshrrev_b32 v11, 16, v15 // (op 12030) 
v_lshrrev_b32 v12, 16, v16 // (op 12030) 
v_lshrrev_b32 v13, 16, v17 // (op 12030) 


v_pack_b32_f16 v14, v10, v11 // (op 12004) 
v_pack_b32_f16 v15, v12, v13 // (op 12004) 

s_lshl_b32 s0, s4, 5 // (op 12004) 
s_mul_i32 s1, 3, s0 // (op 12004) 
s_add_u32 s1, s1, 192 // (op 12004) 

v_add_u32 v0, v9, s1 // (op 12004) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12004) Store value



v_accvgpr_read v10, a208 // (op 12034) 
v_accvgpr_read v11, a209 // (op 12034) 
v_accvgpr_read v12, a210 // (op 12034) 
v_accvgpr_read v13, a211 // (op 12034) 
v_mul_f32 v14, v7, v10 // (op 12034) 
v_mul_f32 v15, v7, v11 // (op 12034) 
v_mul_f32 v16, v7, v12 // (op 12034) 
v_mul_f32 v17, v7, v13 // (op 12034) 

v_lshrrev_b32 v10, 16, v14 // (op 12036) 
v_lshrrev_b32 v11, 16, v15 // (op 12036) 
v_lshrrev_b32 v12, 16, v16 // (op 12036) 
v_lshrrev_b32 v13, 16, v17 // (op 12036) 


v_pack_b32_f16 v14, v10, v11 // (op 12006) 
v_pack_b32_f16 v15, v12, v13 // (op 12006) 

s_lshl_b32 s0, s4, 7 // (op 12006) 
s_add_u32 s0, s0, 192 // (op 12006) 
v_add_u32 v0, v9, s0 // (op 12006) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12006) Store value



v_accvgpr_read v10, a212 // (op 12040) 
v_accvgpr_read v11, a213 // (op 12040) 
v_accvgpr_read v12, a214 // (op 12040) 
v_accvgpr_read v13, a215 // (op 12040) 
v_mul_f32 v14, v7, v10 // (op 12040) 
v_mul_f32 v15, v7, v11 // (op 12040) 
v_mul_f32 v16, v7, v12 // (op 12040) 
v_mul_f32 v17, v7, v13 // (op 12040) 

v_lshrrev_b32 v10, 16, v14 // (op 12042) 
v_lshrrev_b32 v11, 16, v15 // (op 12042) 
v_lshrrev_b32 v12, 16, v16 // (op 12042) 
v_lshrrev_b32 v13, 16, v17 // (op 12042) 


v_pack_b32_f16 v14, v10, v11 // (op 12008) 
v_pack_b32_f16 v15, v12, v13 // (op 12008) 

s_lshl_b32 s0, s4, 5 // (op 12008) 
s_mul_i32 s1, 5, s0 // (op 12008) 
s_add_u32 s1, s1, 192 // (op 12008) 

v_add_u32 v0, v9, s1 // (op 12008) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12008) Store value



v_accvgpr_read v10, a216 // (op 12046) 
v_accvgpr_read v11, a217 // (op 12046) 
v_accvgpr_read v12, a218 // (op 12046) 
v_accvgpr_read v13, a219 // (op 12046) 
v_mul_f32 v14, v7, v10 // (op 12046) 
v_mul_f32 v15, v7, v11 // (op 12046) 
v_mul_f32 v16, v7, v12 // (op 12046) 
v_mul_f32 v17, v7, v13 // (op 12046) 

v_lshrrev_b32 v10, 16, v14 // (op 12048) 
v_lshrrev_b32 v11, 16, v15 // (op 12048) 
v_lshrrev_b32 v12, 16, v16 // (op 12048) 
v_lshrrev_b32 v13, 16, v17 // (op 12048) 


v_pack_b32_f16 v14, v10, v11 // (op 12010) 
v_pack_b32_f16 v15, v12, v13 // (op 12010) 

s_lshl_b32 s0, s4, 5 // (op 12010) 
s_mul_i32 s1, 6, s0 // (op 12010) 
s_add_u32 s1, s1, 192 // (op 12010) 

v_add_u32 v0, v9, s1 // (op 12010) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12010) Store value



v_accvgpr_read v10, a220 // (op 12052) 
v_accvgpr_read v11, a221 // (op 12052) 
v_accvgpr_read v12, a222 // (op 12052) 
v_accvgpr_read v13, a223 // (op 12052) 
v_mul_f32 v14, v7, v10 // (op 12052) 
v_mul_f32 v15, v7, v11 // (op 12052) 
v_mul_f32 v16, v7, v12 // (op 12052) 
v_mul_f32 v17, v7, v13 // (op 12052) 

v_lshrrev_b32 v10, 16, v14 // (op 12054) 
v_lshrrev_b32 v11, 16, v15 // (op 12054) 
v_lshrrev_b32 v12, 16, v16 // (op 12054) 
v_lshrrev_b32 v13, 16, v17 // (op 12054) 


v_pack_b32_f16 v14, v10, v11 // (op 12012) 
v_pack_b32_f16 v15, v12, v13 // (op 12012) 

s_lshl_b32 s0, s4, 5 // (op 12012) 
s_mul_i32 s1, 7, s0 // (op 12012) 
s_add_u32 s1, s1, 192 // (op 12012) 

v_add_u32 v0, v9, s1 // (op 12012) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12012) Store value



v_accvgpr_read v10, a224 // (op 12142) 
v_accvgpr_read v11, a225 // (op 12142) 
v_accvgpr_read v12, a226 // (op 12142) 
v_accvgpr_read v13, a227 // (op 12142) 
v_mul_f32 v14, v7, v10 // (op 12142) 
v_mul_f32 v15, v7, v11 // (op 12142) 
v_mul_f32 v16, v7, v12 // (op 12142) 
v_mul_f32 v17, v7, v13 // (op 12142) 

v_lshrrev_b32 v10, 16, v14 // (op 12144) 
v_lshrrev_b32 v11, 16, v15 // (op 12144) 
v_lshrrev_b32 v12, 16, v16 // (op 12144) 
v_lshrrev_b32 v13, 16, v17 // (op 12144) 


v_pack_b32_f16 v14, v10, v11 // (op 12146) 
v_pack_b32_f16 v15, v12, v13 // (op 12146) 

v_add_u32 v0, 224, v9 // (op 12146) 

buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12146) Store value



v_accvgpr_read v10, a228 // (op 12164) 
v_accvgpr_read v11, a229 // (op 12164) 
v_accvgpr_read v12, a230 // (op 12164) 
v_accvgpr_read v13, a231 // (op 12164) 
v_mul_f32 v14, v7, v10 // (op 12164) 
v_mul_f32 v15, v7, v11 // (op 12164) 
v_mul_f32 v16, v7, v12 // (op 12164) 
v_mul_f32 v17, v7, v13 // (op 12164) 

v_lshrrev_b32 v10, 16, v14 // (op 12166) 
v_lshrrev_b32 v11, 16, v15 // (op 12166) 
v_lshrrev_b32 v12, 16, v16 // (op 12166) 
v_lshrrev_b32 v13, 16, v17 // (op 12166) 


v_pack_b32_f16 v14, v10, v11 // (op 12148) 
v_pack_b32_f16 v15, v12, v13 // (op 12148) 

s_lshl_b32 s0, s4, 5 // (op 12148) 
s_add_u32 s0, s0, 224 // (op 12148) 
v_add_u32 v0, v9, s0 // (op 12148) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12148) Store value



v_accvgpr_read v10, a232 // (op 12170) 
v_accvgpr_read v11, a233 // (op 12170) 
v_accvgpr_read v12, a234 // (op 12170) 
v_accvgpr_read v13, a235 // (op 12170) 
v_mul_f32 v14, v7, v10 // (op 12170) 
v_mul_f32 v15, v7, v11 // (op 12170) 
v_mul_f32 v16, v7, v12 // (op 12170) 
v_mul_f32 v17, v7, v13 // (op 12170) 

v_lshrrev_b32 v10, 16, v14 // (op 12172) 
v_lshrrev_b32 v11, 16, v15 // (op 12172) 
v_lshrrev_b32 v12, 16, v16 // (op 12172) 
v_lshrrev_b32 v13, 16, v17 // (op 12172) 


v_pack_b32_f16 v14, v10, v11 // (op 12150) 
v_pack_b32_f16 v15, v12, v13 // (op 12150) 

s_lshl_b32 s0, s4, 6 // (op 12150) 
s_add_u32 s0, s0, 224 // (op 12150) 
v_add_u32 v0, v9, s0 // (op 12150) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12150) Store value



v_accvgpr_read v10, a236 // (op 12176) 
v_accvgpr_read v11, a237 // (op 12176) 
v_accvgpr_read v12, a238 // (op 12176) 
v_accvgpr_read v13, a239 // (op 12176) 
v_mul_f32 v14, v7, v10 // (op 12176) 
v_mul_f32 v15, v7, v11 // (op 12176) 
v_mul_f32 v16, v7, v12 // (op 12176) 
v_mul_f32 v17, v7, v13 // (op 12176) 

v_lshrrev_b32 v10, 16, v14 // (op 12178) 
v_lshrrev_b32 v11, 16, v15 // (op 12178) 
v_lshrrev_b32 v12, 16, v16 // (op 12178) 
v_lshrrev_b32 v13, 16, v17 // (op 12178) 


v_pack_b32_f16 v14, v10, v11 // (op 12152) 
v_pack_b32_f16 v15, v12, v13 // (op 12152) 

s_lshl_b32 s0, s4, 5 // (op 12152) 
s_mul_i32 s1, 3, s0 // (op 12152) 
s_add_u32 s1, s1, 224 // (op 12152) 

v_add_u32 v0, v9, s1 // (op 12152) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12152) Store value



v_accvgpr_read v10, a240 // (op 12182) 
v_accvgpr_read v11, a241 // (op 12182) 
v_accvgpr_read v12, a242 // (op 12182) 
v_accvgpr_read v13, a243 // (op 12182) 
v_mul_f32 v14, v7, v10 // (op 12182) 
v_mul_f32 v15, v7, v11 // (op 12182) 
v_mul_f32 v16, v7, v12 // (op 12182) 
v_mul_f32 v17, v7, v13 // (op 12182) 

v_lshrrev_b32 v10, 16, v14 // (op 12184) 
v_lshrrev_b32 v11, 16, v15 // (op 12184) 
v_lshrrev_b32 v12, 16, v16 // (op 12184) 
v_lshrrev_b32 v13, 16, v17 // (op 12184) 


v_pack_b32_f16 v14, v10, v11 // (op 12154) 
v_pack_b32_f16 v15, v12, v13 // (op 12154) 

s_lshl_b32 s0, s4, 7 // (op 12154) 
s_add_u32 s0, s0, 224 // (op 12154) 
v_add_u32 v0, v9, s0 // (op 12154) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12154) Store value



v_accvgpr_read v10, a244 // (op 12188) 
v_accvgpr_read v11, a245 // (op 12188) 
v_accvgpr_read v12, a246 // (op 12188) 
v_accvgpr_read v13, a247 // (op 12188) 
v_mul_f32 v14, v7, v10 // (op 12188) 
v_mul_f32 v15, v7, v11 // (op 12188) 
v_mul_f32 v16, v7, v12 // (op 12188) 
v_mul_f32 v17, v7, v13 // (op 12188) 

v_lshrrev_b32 v10, 16, v14 // (op 12190) 
v_lshrrev_b32 v11, 16, v15 // (op 12190) 
v_lshrrev_b32 v12, 16, v16 // (op 12190) 
v_lshrrev_b32 v13, 16, v17 // (op 12190) 


v_pack_b32_f16 v14, v10, v11 // (op 12156) 
v_pack_b32_f16 v15, v12, v13 // (op 12156) 

s_lshl_b32 s0, s4, 5 // (op 12156) 
s_mul_i32 s1, 5, s0 // (op 12156) 
s_add_u32 s1, s1, 224 // (op 12156) 

v_add_u32 v0, v9, s1 // (op 12156) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12156) Store value



v_accvgpr_read v10, a248 // (op 12194) 
v_accvgpr_read v11, a249 // (op 12194) 
v_accvgpr_read v12, a250 // (op 12194) 
v_accvgpr_read v13, a251 // (op 12194) 
v_mul_f32 v14, v7, v10 // (op 12194) 
v_mul_f32 v15, v7, v11 // (op 12194) 
v_mul_f32 v16, v7, v12 // (op 12194) 
v_mul_f32 v17, v7, v13 // (op 12194) 

v_lshrrev_b32 v10, 16, v14 // (op 12196) 
v_lshrrev_b32 v11, 16, v15 // (op 12196) 
v_lshrrev_b32 v12, 16, v16 // (op 12196) 
v_lshrrev_b32 v13, 16, v17 // (op 12196) 


v_pack_b32_f16 v14, v10, v11 // (op 12158) 
v_pack_b32_f16 v15, v12, v13 // (op 12158) 

s_lshl_b32 s0, s4, 5 // (op 12158) 
s_mul_i32 s1, 6, s0 // (op 12158) 
s_add_u32 s1, s1, 224 // (op 12158) 

v_add_u32 v0, v9, s1 // (op 12158) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12158) Store value



v_accvgpr_read v10, a252 // (op 12200) 
v_accvgpr_read v11, a253 // (op 12200) 
v_accvgpr_read v12, a254 // (op 12200) 
v_accvgpr_read v13, a255 // (op 12200) 
v_mul_f32 v14, v7, v10 // (op 12200) 
v_mul_f32 v15, v7, v11 // (op 12200) 
v_mul_f32 v16, v7, v12 // (op 12200) 
v_mul_f32 v17, v7, v13 // (op 12200) 

v_lshrrev_b32 v10, 16, v14 // (op 12202) 
v_lshrrev_b32 v11, 16, v15 // (op 12202) 
v_lshrrev_b32 v12, 16, v16 // (op 12202) 
v_lshrrev_b32 v13, 16, v17 // (op 12202) 


v_pack_b32_f16 v14, v10, v11 // (op 12160) 
v_pack_b32_f16 v15, v12, v13 // (op 12160) 

s_lshl_b32 s0, s4, 5 // (op 12160) 
s_mul_i32 s1, 7, s0 // (op 12160) 
s_add_u32 s1, s1, 224 // (op 12160) 

v_add_u32 v0, v9, s1 // (op 12160) 


buffer_store_dwordx2 v[14:15], v0, s[44:47], 0 offen // (op 12160) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ConditionalBottom_0_beta_116
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ConditionalFalse_0_beta_116:


v_add_u32 v0, 224, v8 // (op 13720) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13720) Load value

v_accvgpr_read v12, a224 // (op 13699) 
v_accvgpr_read v13, a225 // (op 13699) 
v_accvgpr_read v14, a226 // (op 13699) 
v_accvgpr_read v15, a227 // (op 13699) 
v_mul_f32 v16, v7, v12 // (op 13699) 
v_mul_f32 v17, v7, v13 // (op 13699) 
v_mul_f32 v18, v7, v14 // (op 13699) 
v_mul_f32 v19, v7, v15 // (op 13699) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13719) 
v_lshrrev_b32 v13, 16, v10 // (op 13719) 
v_lshlrev_b32 v12, 16, v12 // (op 13719) 
v_lshlrev_b32 v13, 16, v13 // (op 13719) 
v_mul_f32 v20, v5, v12 // (op 13719) 
v_mul_f32 v21, v5, v13 // (op 13719) 
v_and_b32 v12, 65535, v11 // (op 13719) 
v_lshrrev_b32 v13, 16, v11 // (op 13719) 
v_lshlrev_b32 v12, 16, v12 // (op 13719) 
v_lshlrev_b32 v13, 16, v13 // (op 13719) 
v_mul_f32 v22, v5, v12 // (op 13719) 
v_mul_f32 v23, v5, v13 // (op 13719) 


v_add_f32 v10, v20, v16 // (op 13700) 
v_add_f32 v11, v21, v17 // (op 13700) 
v_add_f32 v12, v22, v18 // (op 13700) 
v_add_f32 v13, v23, v19 // (op 13700) 


v_lshrrev_b32 v14, 16, v10 // (op 13701) 
v_lshrrev_b32 v15, 16, v11 // (op 13701) 
v_lshrrev_b32 v16, 16, v12 // (op 13701) 
v_lshrrev_b32 v17, 16, v13 // (op 13701) 


v_pack_b32_f16 v10, v14, v15 // (op 13703) 
v_pack_b32_f16 v11, v16, v17 // (op 13703) 

v_add_u32 v0, 224, v9 // (op 13703) 

buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13703) Store value





s_lshl_b32 s0, s2, 5 // (op 13725) 
s_add_u32 s0, s0, 224 // (op 13725) 
v_add_u32 v0, v8, s0 // (op 13725) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13725) Load value

v_accvgpr_read v12, a228 // (op 13757) 
v_accvgpr_read v13, a229 // (op 13757) 
v_accvgpr_read v14, a230 // (op 13757) 
v_accvgpr_read v15, a231 // (op 13757) 
v_mul_f32 v16, v7, v12 // (op 13757) 
v_mul_f32 v17, v7, v13 // (op 13757) 
v_mul_f32 v18, v7, v14 // (op 13757) 
v_mul_f32 v19, v7, v15 // (op 13757) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13722) 
v_lshrrev_b32 v13, 16, v10 // (op 13722) 
v_lshlrev_b32 v12, 16, v12 // (op 13722) 
v_lshlrev_b32 v13, 16, v13 // (op 13722) 
v_mul_f32 v20, v5, v12 // (op 13722) 
v_mul_f32 v21, v5, v13 // (op 13722) 
v_and_b32 v12, 65535, v11 // (op 13722) 
v_lshrrev_b32 v13, 16, v11 // (op 13722) 
v_lshlrev_b32 v12, 16, v12 // (op 13722) 
v_lshlrev_b32 v13, 16, v13 // (op 13722) 
v_mul_f32 v22, v5, v12 // (op 13722) 
v_mul_f32 v23, v5, v13 // (op 13722) 


v_add_f32 v10, v20, v16 // (op 13723) 
v_add_f32 v11, v21, v17 // (op 13723) 
v_add_f32 v12, v22, v18 // (op 13723) 
v_add_f32 v13, v23, v19 // (op 13723) 


v_lshrrev_b32 v14, 16, v10 // (op 13724) 
v_lshrrev_b32 v15, 16, v11 // (op 13724) 
v_lshrrev_b32 v16, 16, v12 // (op 13724) 
v_lshrrev_b32 v17, 16, v13 // (op 13724) 


v_pack_b32_f16 v10, v14, v15 // (op 13705) 
v_pack_b32_f16 v11, v16, v17 // (op 13705) 

s_lshl_b32 s0, s4, 5 // (op 13705) 
s_add_u32 s0, s0, 224 // (op 13705) 
v_add_u32 v0, v9, s0 // (op 13705) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13705) Store value





s_lshl_b32 s0, s2, 6 // (op 13730) 
s_add_u32 s0, s0, 224 // (op 13730) 
v_add_u32 v0, v8, s0 // (op 13730) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13730) Load value

v_accvgpr_read v12, a232 // (op 13759) 
v_accvgpr_read v13, a233 // (op 13759) 
v_accvgpr_read v14, a234 // (op 13759) 
v_accvgpr_read v15, a235 // (op 13759) 
v_mul_f32 v16, v7, v12 // (op 13759) 
v_mul_f32 v17, v7, v13 // (op 13759) 
v_mul_f32 v18, v7, v14 // (op 13759) 
v_mul_f32 v19, v7, v15 // (op 13759) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13727) 
v_lshrrev_b32 v13, 16, v10 // (op 13727) 
v_lshlrev_b32 v12, 16, v12 // (op 13727) 
v_lshlrev_b32 v13, 16, v13 // (op 13727) 
v_mul_f32 v20, v5, v12 // (op 13727) 
v_mul_f32 v21, v5, v13 // (op 13727) 
v_and_b32 v12, 65535, v11 // (op 13727) 
v_lshrrev_b32 v13, 16, v11 // (op 13727) 
v_lshlrev_b32 v12, 16, v12 // (op 13727) 
v_lshlrev_b32 v13, 16, v13 // (op 13727) 
v_mul_f32 v22, v5, v12 // (op 13727) 
v_mul_f32 v23, v5, v13 // (op 13727) 


v_add_f32 v10, v20, v16 // (op 13728) 
v_add_f32 v11, v21, v17 // (op 13728) 
v_add_f32 v12, v22, v18 // (op 13728) 
v_add_f32 v13, v23, v19 // (op 13728) 


v_lshrrev_b32 v14, 16, v10 // (op 13729) 
v_lshrrev_b32 v15, 16, v11 // (op 13729) 
v_lshrrev_b32 v16, 16, v12 // (op 13729) 
v_lshrrev_b32 v17, 16, v13 // (op 13729) 


v_pack_b32_f16 v10, v14, v15 // (op 13707) 
v_pack_b32_f16 v11, v16, v17 // (op 13707) 

s_lshl_b32 s0, s4, 6 // (op 13707) 
s_add_u32 s0, s0, 224 // (op 13707) 
v_add_u32 v0, v9, s0 // (op 13707) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13707) Store value





s_lshl_b32 s0, s2, 5 // (op 13735) 
s_mul_i32 s1, 3, s0 // (op 13735) 
s_add_u32 s1, s1, 224 // (op 13735) 

v_add_u32 v0, v8, s1 // (op 13735) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13735) Load value

v_accvgpr_read v12, a236 // (op 13761) 
v_accvgpr_read v13, a237 // (op 13761) 
v_accvgpr_read v14, a238 // (op 13761) 
v_accvgpr_read v15, a239 // (op 13761) 
v_mul_f32 v16, v7, v12 // (op 13761) 
v_mul_f32 v17, v7, v13 // (op 13761) 
v_mul_f32 v18, v7, v14 // (op 13761) 
v_mul_f32 v19, v7, v15 // (op 13761) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13732) 
v_lshrrev_b32 v13, 16, v10 // (op 13732) 
v_lshlrev_b32 v12, 16, v12 // (op 13732) 
v_lshlrev_b32 v13, 16, v13 // (op 13732) 
v_mul_f32 v20, v5, v12 // (op 13732) 
v_mul_f32 v21, v5, v13 // (op 13732) 
v_and_b32 v12, 65535, v11 // (op 13732) 
v_lshrrev_b32 v13, 16, v11 // (op 13732) 
v_lshlrev_b32 v12, 16, v12 // (op 13732) 
v_lshlrev_b32 v13, 16, v13 // (op 13732) 
v_mul_f32 v22, v5, v12 // (op 13732) 
v_mul_f32 v23, v5, v13 // (op 13732) 


v_add_f32 v10, v20, v16 // (op 13733) 
v_add_f32 v11, v21, v17 // (op 13733) 
v_add_f32 v12, v22, v18 // (op 13733) 
v_add_f32 v13, v23, v19 // (op 13733) 


v_lshrrev_b32 v14, 16, v10 // (op 13734) 
v_lshrrev_b32 v15, 16, v11 // (op 13734) 
v_lshrrev_b32 v16, 16, v12 // (op 13734) 
v_lshrrev_b32 v17, 16, v13 // (op 13734) 


v_pack_b32_f16 v10, v14, v15 // (op 13709) 
v_pack_b32_f16 v11, v16, v17 // (op 13709) 

s_lshl_b32 s0, s4, 5 // (op 13709) 
s_mul_i32 s1, 3, s0 // (op 13709) 
s_add_u32 s1, s1, 224 // (op 13709) 

v_add_u32 v0, v9, s1 // (op 13709) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13709) Store value





s_lshl_b32 s0, s2, 7 // (op 13740) 
s_add_u32 s0, s0, 224 // (op 13740) 
v_add_u32 v0, v8, s0 // (op 13740) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13740) Load value

v_accvgpr_read v12, a240 // (op 13763) 
v_accvgpr_read v13, a241 // (op 13763) 
v_accvgpr_read v14, a242 // (op 13763) 
v_accvgpr_read v15, a243 // (op 13763) 
v_mul_f32 v16, v7, v12 // (op 13763) 
v_mul_f32 v17, v7, v13 // (op 13763) 
v_mul_f32 v18, v7, v14 // (op 13763) 
v_mul_f32 v19, v7, v15 // (op 13763) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13737) 
v_lshrrev_b32 v13, 16, v10 // (op 13737) 
v_lshlrev_b32 v12, 16, v12 // (op 13737) 
v_lshlrev_b32 v13, 16, v13 // (op 13737) 
v_mul_f32 v20, v5, v12 // (op 13737) 
v_mul_f32 v21, v5, v13 // (op 13737) 
v_and_b32 v12, 65535, v11 // (op 13737) 
v_lshrrev_b32 v13, 16, v11 // (op 13737) 
v_lshlrev_b32 v12, 16, v12 // (op 13737) 
v_lshlrev_b32 v13, 16, v13 // (op 13737) 
v_mul_f32 v22, v5, v12 // (op 13737) 
v_mul_f32 v23, v5, v13 // (op 13737) 


v_add_f32 v10, v20, v16 // (op 13738) 
v_add_f32 v11, v21, v17 // (op 13738) 
v_add_f32 v12, v22, v18 // (op 13738) 
v_add_f32 v13, v23, v19 // (op 13738) 


v_lshrrev_b32 v14, 16, v10 // (op 13739) 
v_lshrrev_b32 v15, 16, v11 // (op 13739) 
v_lshrrev_b32 v16, 16, v12 // (op 13739) 
v_lshrrev_b32 v17, 16, v13 // (op 13739) 


v_pack_b32_f16 v10, v14, v15 // (op 13711) 
v_pack_b32_f16 v11, v16, v17 // (op 13711) 

s_lshl_b32 s0, s4, 7 // (op 13711) 
s_add_u32 s0, s0, 224 // (op 13711) 
v_add_u32 v0, v9, s0 // (op 13711) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13711) Store value





s_lshl_b32 s0, s2, 5 // (op 13745) 
s_mul_i32 s1, 5, s0 // (op 13745) 
s_add_u32 s1, s1, 224 // (op 13745) 

v_add_u32 v0, v8, s1 // (op 13745) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13745) Load value

v_accvgpr_read v12, a244 // (op 13765) 
v_accvgpr_read v13, a245 // (op 13765) 
v_accvgpr_read v14, a246 // (op 13765) 
v_accvgpr_read v15, a247 // (op 13765) 
v_mul_f32 v16, v7, v12 // (op 13765) 
v_mul_f32 v17, v7, v13 // (op 13765) 
v_mul_f32 v18, v7, v14 // (op 13765) 
v_mul_f32 v19, v7, v15 // (op 13765) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13742) 
v_lshrrev_b32 v13, 16, v10 // (op 13742) 
v_lshlrev_b32 v12, 16, v12 // (op 13742) 
v_lshlrev_b32 v13, 16, v13 // (op 13742) 
v_mul_f32 v20, v5, v12 // (op 13742) 
v_mul_f32 v21, v5, v13 // (op 13742) 
v_and_b32 v12, 65535, v11 // (op 13742) 
v_lshrrev_b32 v13, 16, v11 // (op 13742) 
v_lshlrev_b32 v12, 16, v12 // (op 13742) 
v_lshlrev_b32 v13, 16, v13 // (op 13742) 
v_mul_f32 v22, v5, v12 // (op 13742) 
v_mul_f32 v23, v5, v13 // (op 13742) 


v_add_f32 v10, v20, v16 // (op 13743) 
v_add_f32 v11, v21, v17 // (op 13743) 
v_add_f32 v12, v22, v18 // (op 13743) 
v_add_f32 v13, v23, v19 // (op 13743) 


v_lshrrev_b32 v14, 16, v10 // (op 13744) 
v_lshrrev_b32 v15, 16, v11 // (op 13744) 
v_lshrrev_b32 v16, 16, v12 // (op 13744) 
v_lshrrev_b32 v17, 16, v13 // (op 13744) 


v_pack_b32_f16 v10, v14, v15 // (op 13713) 
v_pack_b32_f16 v11, v16, v17 // (op 13713) 

s_lshl_b32 s0, s4, 5 // (op 13713) 
s_mul_i32 s1, 5, s0 // (op 13713) 
s_add_u32 s1, s1, 224 // (op 13713) 

v_add_u32 v0, v9, s1 // (op 13713) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13713) Store value





s_lshl_b32 s0, s2, 5 // (op 13750) 
s_mul_i32 s1, 6, s0 // (op 13750) 
s_add_u32 s1, s1, 224 // (op 13750) 

v_add_u32 v0, v8, s1 // (op 13750) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13750) Load value

v_accvgpr_read v12, a248 // (op 13767) 
v_accvgpr_read v13, a249 // (op 13767) 
v_accvgpr_read v14, a250 // (op 13767) 
v_accvgpr_read v15, a251 // (op 13767) 
v_mul_f32 v16, v7, v12 // (op 13767) 
v_mul_f32 v17, v7, v13 // (op 13767) 
v_mul_f32 v18, v7, v14 // (op 13767) 
v_mul_f32 v19, v7, v15 // (op 13767) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13747) 
v_lshrrev_b32 v13, 16, v10 // (op 13747) 
v_lshlrev_b32 v12, 16, v12 // (op 13747) 
v_lshlrev_b32 v13, 16, v13 // (op 13747) 
v_mul_f32 v20, v5, v12 // (op 13747) 
v_mul_f32 v21, v5, v13 // (op 13747) 
v_and_b32 v12, 65535, v11 // (op 13747) 
v_lshrrev_b32 v13, 16, v11 // (op 13747) 
v_lshlrev_b32 v12, 16, v12 // (op 13747) 
v_lshlrev_b32 v13, 16, v13 // (op 13747) 
v_mul_f32 v22, v5, v12 // (op 13747) 
v_mul_f32 v23, v5, v13 // (op 13747) 


v_add_f32 v10, v20, v16 // (op 13748) 
v_add_f32 v11, v21, v17 // (op 13748) 
v_add_f32 v12, v22, v18 // (op 13748) 
v_add_f32 v13, v23, v19 // (op 13748) 


v_lshrrev_b32 v14, 16, v10 // (op 13749) 
v_lshrrev_b32 v15, 16, v11 // (op 13749) 
v_lshrrev_b32 v16, 16, v12 // (op 13749) 
v_lshrrev_b32 v17, 16, v13 // (op 13749) 


v_pack_b32_f16 v10, v14, v15 // (op 13715) 
v_pack_b32_f16 v11, v16, v17 // (op 13715) 

s_lshl_b32 s0, s4, 5 // (op 13715) 
s_mul_i32 s1, 6, s0 // (op 13715) 
s_add_u32 s1, s1, 224 // (op 13715) 

v_add_u32 v0, v9, s1 // (op 13715) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13715) Store value





s_lshl_b32 s0, s2, 5 // (op 13755) 
s_mul_i32 s1, 7, s0 // (op 13755) 
s_add_u32 s1, s1, 224 // (op 13755) 

v_add_u32 v0, v8, s1 // (op 13755) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13755) Load value

v_accvgpr_read v12, a252 // (op 13769) 
v_accvgpr_read v13, a253 // (op 13769) 
v_accvgpr_read v14, a254 // (op 13769) 
v_accvgpr_read v15, a255 // (op 13769) 
v_mul_f32 v16, v7, v12 // (op 13769) 
v_mul_f32 v17, v7, v13 // (op 13769) 
v_mul_f32 v18, v7, v14 // (op 13769) 
v_mul_f32 v19, v7, v15 // (op 13769) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13752) 
v_lshrrev_b32 v13, 16, v10 // (op 13752) 
v_lshlrev_b32 v12, 16, v12 // (op 13752) 
v_lshlrev_b32 v13, 16, v13 // (op 13752) 
v_mul_f32 v20, v5, v12 // (op 13752) 
v_mul_f32 v21, v5, v13 // (op 13752) 
v_and_b32 v12, 65535, v11 // (op 13752) 
v_lshrrev_b32 v13, 16, v11 // (op 13752) 
v_lshlrev_b32 v12, 16, v12 // (op 13752) 
v_lshlrev_b32 v13, 16, v13 // (op 13752) 
v_mul_f32 v22, v5, v12 // (op 13752) 
v_mul_f32 v23, v5, v13 // (op 13752) 


v_add_f32 v10, v20, v16 // (op 13753) 
v_add_f32 v11, v21, v17 // (op 13753) 
v_add_f32 v12, v22, v18 // (op 13753) 
v_add_f32 v13, v23, v19 // (op 13753) 


v_lshrrev_b32 v14, 16, v10 // (op 13754) 
v_lshrrev_b32 v15, 16, v11 // (op 13754) 
v_lshrrev_b32 v16, 16, v12 // (op 13754) 
v_lshrrev_b32 v17, 16, v13 // (op 13754) 


v_pack_b32_f16 v10, v14, v15 // (op 13717) 
v_pack_b32_f16 v11, v16, v17 // (op 13717) 

s_lshl_b32 s0, s4, 5 // (op 13717) 
s_mul_i32 s1, 7, s0 // (op 13717) 
s_add_u32 s1, s1, 224 // (op 13717) 

v_add_u32 v0, v9, s1 // (op 13717) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13717) Store value





v_add_u32 v0, 160, v8 // (op 13378) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13378) Load value

v_accvgpr_read v12, a160 // (op 13357) 
v_accvgpr_read v13, a161 // (op 13357) 
v_accvgpr_read v14, a162 // (op 13357) 
v_accvgpr_read v15, a163 // (op 13357) 
v_mul_f32 v16, v7, v12 // (op 13357) 
v_mul_f32 v17, v7, v13 // (op 13357) 
v_mul_f32 v18, v7, v14 // (op 13357) 
v_mul_f32 v19, v7, v15 // (op 13357) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13377) 
v_lshrrev_b32 v13, 16, v10 // (op 13377) 
v_lshlrev_b32 v12, 16, v12 // (op 13377) 
v_lshlrev_b32 v13, 16, v13 // (op 13377) 
v_mul_f32 v20, v5, v12 // (op 13377) 
v_mul_f32 v21, v5, v13 // (op 13377) 
v_and_b32 v12, 65535, v11 // (op 13377) 
v_lshrrev_b32 v13, 16, v11 // (op 13377) 
v_lshlrev_b32 v12, 16, v12 // (op 13377) 
v_lshlrev_b32 v13, 16, v13 // (op 13377) 
v_mul_f32 v22, v5, v12 // (op 13377) 
v_mul_f32 v23, v5, v13 // (op 13377) 


v_add_f32 v10, v20, v16 // (op 13358) 
v_add_f32 v11, v21, v17 // (op 13358) 
v_add_f32 v12, v22, v18 // (op 13358) 
v_add_f32 v13, v23, v19 // (op 13358) 


v_lshrrev_b32 v14, 16, v10 // (op 13359) 
v_lshrrev_b32 v15, 16, v11 // (op 13359) 
v_lshrrev_b32 v16, 16, v12 // (op 13359) 
v_lshrrev_b32 v17, 16, v13 // (op 13359) 


v_pack_b32_f16 v10, v14, v15 // (op 13361) 
v_pack_b32_f16 v11, v16, v17 // (op 13361) 

v_add_u32 v0, 160, v9 // (op 13361) 

buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13361) Store value





s_lshl_b32 s0, s2, 5 // (op 13383) 
s_add_u32 s0, s0, 160 // (op 13383) 
v_add_u32 v0, v8, s0 // (op 13383) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13383) Load value

v_accvgpr_read v12, a164 // (op 13415) 
v_accvgpr_read v13, a165 // (op 13415) 
v_accvgpr_read v14, a166 // (op 13415) 
v_accvgpr_read v15, a167 // (op 13415) 
v_mul_f32 v16, v7, v12 // (op 13415) 
v_mul_f32 v17, v7, v13 // (op 13415) 
v_mul_f32 v18, v7, v14 // (op 13415) 
v_mul_f32 v19, v7, v15 // (op 13415) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13380) 
v_lshrrev_b32 v13, 16, v10 // (op 13380) 
v_lshlrev_b32 v12, 16, v12 // (op 13380) 
v_lshlrev_b32 v13, 16, v13 // (op 13380) 
v_mul_f32 v20, v5, v12 // (op 13380) 
v_mul_f32 v21, v5, v13 // (op 13380) 
v_and_b32 v12, 65535, v11 // (op 13380) 
v_lshrrev_b32 v13, 16, v11 // (op 13380) 
v_lshlrev_b32 v12, 16, v12 // (op 13380) 
v_lshlrev_b32 v13, 16, v13 // (op 13380) 
v_mul_f32 v22, v5, v12 // (op 13380) 
v_mul_f32 v23, v5, v13 // (op 13380) 


v_add_f32 v10, v20, v16 // (op 13381) 
v_add_f32 v11, v21, v17 // (op 13381) 
v_add_f32 v12, v22, v18 // (op 13381) 
v_add_f32 v13, v23, v19 // (op 13381) 


v_lshrrev_b32 v14, 16, v10 // (op 13382) 
v_lshrrev_b32 v15, 16, v11 // (op 13382) 
v_lshrrev_b32 v16, 16, v12 // (op 13382) 
v_lshrrev_b32 v17, 16, v13 // (op 13382) 


v_pack_b32_f16 v10, v14, v15 // (op 13363) 
v_pack_b32_f16 v11, v16, v17 // (op 13363) 

s_lshl_b32 s0, s4, 5 // (op 13363) 
s_add_u32 s0, s0, 160 // (op 13363) 
v_add_u32 v0, v9, s0 // (op 13363) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13363) Store value





s_lshl_b32 s0, s2, 6 // (op 13388) 
s_add_u32 s0, s0, 160 // (op 13388) 
v_add_u32 v0, v8, s0 // (op 13388) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13388) Load value

v_accvgpr_read v12, a168 // (op 13417) 
v_accvgpr_read v13, a169 // (op 13417) 
v_accvgpr_read v14, a170 // (op 13417) 
v_accvgpr_read v15, a171 // (op 13417) 
v_mul_f32 v16, v7, v12 // (op 13417) 
v_mul_f32 v17, v7, v13 // (op 13417) 
v_mul_f32 v18, v7, v14 // (op 13417) 
v_mul_f32 v19, v7, v15 // (op 13417) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13385) 
v_lshrrev_b32 v13, 16, v10 // (op 13385) 
v_lshlrev_b32 v12, 16, v12 // (op 13385) 
v_lshlrev_b32 v13, 16, v13 // (op 13385) 
v_mul_f32 v20, v5, v12 // (op 13385) 
v_mul_f32 v21, v5, v13 // (op 13385) 
v_and_b32 v12, 65535, v11 // (op 13385) 
v_lshrrev_b32 v13, 16, v11 // (op 13385) 
v_lshlrev_b32 v12, 16, v12 // (op 13385) 
v_lshlrev_b32 v13, 16, v13 // (op 13385) 
v_mul_f32 v22, v5, v12 // (op 13385) 
v_mul_f32 v23, v5, v13 // (op 13385) 


v_add_f32 v10, v20, v16 // (op 13386) 
v_add_f32 v11, v21, v17 // (op 13386) 
v_add_f32 v12, v22, v18 // (op 13386) 
v_add_f32 v13, v23, v19 // (op 13386) 


v_lshrrev_b32 v14, 16, v10 // (op 13387) 
v_lshrrev_b32 v15, 16, v11 // (op 13387) 
v_lshrrev_b32 v16, 16, v12 // (op 13387) 
v_lshrrev_b32 v17, 16, v13 // (op 13387) 


v_pack_b32_f16 v10, v14, v15 // (op 13365) 
v_pack_b32_f16 v11, v16, v17 // (op 13365) 

s_lshl_b32 s0, s4, 6 // (op 13365) 
s_add_u32 s0, s0, 160 // (op 13365) 
v_add_u32 v0, v9, s0 // (op 13365) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13365) Store value





s_lshl_b32 s0, s2, 5 // (op 13393) 
s_mul_i32 s1, 3, s0 // (op 13393) 
s_add_u32 s1, s1, 160 // (op 13393) 

v_add_u32 v0, v8, s1 // (op 13393) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13393) Load value

v_accvgpr_read v12, a172 // (op 13419) 
v_accvgpr_read v13, a173 // (op 13419) 
v_accvgpr_read v14, a174 // (op 13419) 
v_accvgpr_read v15, a175 // (op 13419) 
v_mul_f32 v16, v7, v12 // (op 13419) 
v_mul_f32 v17, v7, v13 // (op 13419) 
v_mul_f32 v18, v7, v14 // (op 13419) 
v_mul_f32 v19, v7, v15 // (op 13419) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13390) 
v_lshrrev_b32 v13, 16, v10 // (op 13390) 
v_lshlrev_b32 v12, 16, v12 // (op 13390) 
v_lshlrev_b32 v13, 16, v13 // (op 13390) 
v_mul_f32 v20, v5, v12 // (op 13390) 
v_mul_f32 v21, v5, v13 // (op 13390) 
v_and_b32 v12, 65535, v11 // (op 13390) 
v_lshrrev_b32 v13, 16, v11 // (op 13390) 
v_lshlrev_b32 v12, 16, v12 // (op 13390) 
v_lshlrev_b32 v13, 16, v13 // (op 13390) 
v_mul_f32 v22, v5, v12 // (op 13390) 
v_mul_f32 v23, v5, v13 // (op 13390) 


v_add_f32 v10, v20, v16 // (op 13391) 
v_add_f32 v11, v21, v17 // (op 13391) 
v_add_f32 v12, v22, v18 // (op 13391) 
v_add_f32 v13, v23, v19 // (op 13391) 


v_lshrrev_b32 v14, 16, v10 // (op 13392) 
v_lshrrev_b32 v15, 16, v11 // (op 13392) 
v_lshrrev_b32 v16, 16, v12 // (op 13392) 
v_lshrrev_b32 v17, 16, v13 // (op 13392) 


v_pack_b32_f16 v10, v14, v15 // (op 13367) 
v_pack_b32_f16 v11, v16, v17 // (op 13367) 

s_lshl_b32 s0, s4, 5 // (op 13367) 
s_mul_i32 s1, 3, s0 // (op 13367) 
s_add_u32 s1, s1, 160 // (op 13367) 

v_add_u32 v0, v9, s1 // (op 13367) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13367) Store value





s_lshl_b32 s0, s2, 7 // (op 13398) 
s_add_u32 s0, s0, 160 // (op 13398) 
v_add_u32 v0, v8, s0 // (op 13398) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13398) Load value

v_accvgpr_read v12, a176 // (op 13421) 
v_accvgpr_read v13, a177 // (op 13421) 
v_accvgpr_read v14, a178 // (op 13421) 
v_accvgpr_read v15, a179 // (op 13421) 
v_mul_f32 v16, v7, v12 // (op 13421) 
v_mul_f32 v17, v7, v13 // (op 13421) 
v_mul_f32 v18, v7, v14 // (op 13421) 
v_mul_f32 v19, v7, v15 // (op 13421) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13395) 
v_lshrrev_b32 v13, 16, v10 // (op 13395) 
v_lshlrev_b32 v12, 16, v12 // (op 13395) 
v_lshlrev_b32 v13, 16, v13 // (op 13395) 
v_mul_f32 v20, v5, v12 // (op 13395) 
v_mul_f32 v21, v5, v13 // (op 13395) 
v_and_b32 v12, 65535, v11 // (op 13395) 
v_lshrrev_b32 v13, 16, v11 // (op 13395) 
v_lshlrev_b32 v12, 16, v12 // (op 13395) 
v_lshlrev_b32 v13, 16, v13 // (op 13395) 
v_mul_f32 v22, v5, v12 // (op 13395) 
v_mul_f32 v23, v5, v13 // (op 13395) 


v_add_f32 v10, v20, v16 // (op 13396) 
v_add_f32 v11, v21, v17 // (op 13396) 
v_add_f32 v12, v22, v18 // (op 13396) 
v_add_f32 v13, v23, v19 // (op 13396) 


v_lshrrev_b32 v14, 16, v10 // (op 13397) 
v_lshrrev_b32 v15, 16, v11 // (op 13397) 
v_lshrrev_b32 v16, 16, v12 // (op 13397) 
v_lshrrev_b32 v17, 16, v13 // (op 13397) 


v_pack_b32_f16 v10, v14, v15 // (op 13369) 
v_pack_b32_f16 v11, v16, v17 // (op 13369) 

s_lshl_b32 s0, s4, 7 // (op 13369) 
s_add_u32 s0, s0, 160 // (op 13369) 
v_add_u32 v0, v9, s0 // (op 13369) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13369) Store value





s_lshl_b32 s0, s2, 5 // (op 13403) 
s_mul_i32 s1, 5, s0 // (op 13403) 
s_add_u32 s1, s1, 160 // (op 13403) 

v_add_u32 v0, v8, s1 // (op 13403) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13403) Load value

v_accvgpr_read v12, a180 // (op 13423) 
v_accvgpr_read v13, a181 // (op 13423) 
v_accvgpr_read v14, a182 // (op 13423) 
v_accvgpr_read v15, a183 // (op 13423) 
v_mul_f32 v16, v7, v12 // (op 13423) 
v_mul_f32 v17, v7, v13 // (op 13423) 
v_mul_f32 v18, v7, v14 // (op 13423) 
v_mul_f32 v19, v7, v15 // (op 13423) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13400) 
v_lshrrev_b32 v13, 16, v10 // (op 13400) 
v_lshlrev_b32 v12, 16, v12 // (op 13400) 
v_lshlrev_b32 v13, 16, v13 // (op 13400) 
v_mul_f32 v20, v5, v12 // (op 13400) 
v_mul_f32 v21, v5, v13 // (op 13400) 
v_and_b32 v12, 65535, v11 // (op 13400) 
v_lshrrev_b32 v13, 16, v11 // (op 13400) 
v_lshlrev_b32 v12, 16, v12 // (op 13400) 
v_lshlrev_b32 v13, 16, v13 // (op 13400) 
v_mul_f32 v22, v5, v12 // (op 13400) 
v_mul_f32 v23, v5, v13 // (op 13400) 


v_add_f32 v10, v20, v16 // (op 13401) 
v_add_f32 v11, v21, v17 // (op 13401) 
v_add_f32 v12, v22, v18 // (op 13401) 
v_add_f32 v13, v23, v19 // (op 13401) 


v_lshrrev_b32 v14, 16, v10 // (op 13402) 
v_lshrrev_b32 v15, 16, v11 // (op 13402) 
v_lshrrev_b32 v16, 16, v12 // (op 13402) 
v_lshrrev_b32 v17, 16, v13 // (op 13402) 


v_pack_b32_f16 v10, v14, v15 // (op 13371) 
v_pack_b32_f16 v11, v16, v17 // (op 13371) 

s_lshl_b32 s0, s4, 5 // (op 13371) 
s_mul_i32 s1, 5, s0 // (op 13371) 
s_add_u32 s1, s1, 160 // (op 13371) 

v_add_u32 v0, v9, s1 // (op 13371) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13371) Store value





s_lshl_b32 s0, s2, 5 // (op 13408) 
s_mul_i32 s1, 6, s0 // (op 13408) 
s_add_u32 s1, s1, 160 // (op 13408) 

v_add_u32 v0, v8, s1 // (op 13408) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13408) Load value

v_accvgpr_read v12, a184 // (op 13425) 
v_accvgpr_read v13, a185 // (op 13425) 
v_accvgpr_read v14, a186 // (op 13425) 
v_accvgpr_read v15, a187 // (op 13425) 
v_mul_f32 v16, v7, v12 // (op 13425) 
v_mul_f32 v17, v7, v13 // (op 13425) 
v_mul_f32 v18, v7, v14 // (op 13425) 
v_mul_f32 v19, v7, v15 // (op 13425) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13405) 
v_lshrrev_b32 v13, 16, v10 // (op 13405) 
v_lshlrev_b32 v12, 16, v12 // (op 13405) 
v_lshlrev_b32 v13, 16, v13 // (op 13405) 
v_mul_f32 v20, v5, v12 // (op 13405) 
v_mul_f32 v21, v5, v13 // (op 13405) 
v_and_b32 v12, 65535, v11 // (op 13405) 
v_lshrrev_b32 v13, 16, v11 // (op 13405) 
v_lshlrev_b32 v12, 16, v12 // (op 13405) 
v_lshlrev_b32 v13, 16, v13 // (op 13405) 
v_mul_f32 v22, v5, v12 // (op 13405) 
v_mul_f32 v23, v5, v13 // (op 13405) 


v_add_f32 v10, v20, v16 // (op 13406) 
v_add_f32 v11, v21, v17 // (op 13406) 
v_add_f32 v12, v22, v18 // (op 13406) 
v_add_f32 v13, v23, v19 // (op 13406) 


v_lshrrev_b32 v14, 16, v10 // (op 13407) 
v_lshrrev_b32 v15, 16, v11 // (op 13407) 
v_lshrrev_b32 v16, 16, v12 // (op 13407) 
v_lshrrev_b32 v17, 16, v13 // (op 13407) 


v_pack_b32_f16 v10, v14, v15 // (op 13373) 
v_pack_b32_f16 v11, v16, v17 // (op 13373) 

s_lshl_b32 s0, s4, 5 // (op 13373) 
s_mul_i32 s1, 6, s0 // (op 13373) 
s_add_u32 s1, s1, 160 // (op 13373) 

v_add_u32 v0, v9, s1 // (op 13373) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13373) Store value





s_lshl_b32 s0, s2, 5 // (op 13413) 
s_mul_i32 s1, 7, s0 // (op 13413) 
s_add_u32 s1, s1, 160 // (op 13413) 

v_add_u32 v0, v8, s1 // (op 13413) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13413) Load value

v_accvgpr_read v12, a188 // (op 13427) 
v_accvgpr_read v13, a189 // (op 13427) 
v_accvgpr_read v14, a190 // (op 13427) 
v_accvgpr_read v15, a191 // (op 13427) 
v_mul_f32 v16, v7, v12 // (op 13427) 
v_mul_f32 v17, v7, v13 // (op 13427) 
v_mul_f32 v18, v7, v14 // (op 13427) 
v_mul_f32 v19, v7, v15 // (op 13427) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13410) 
v_lshrrev_b32 v13, 16, v10 // (op 13410) 
v_lshlrev_b32 v12, 16, v12 // (op 13410) 
v_lshlrev_b32 v13, 16, v13 // (op 13410) 
v_mul_f32 v20, v5, v12 // (op 13410) 
v_mul_f32 v21, v5, v13 // (op 13410) 
v_and_b32 v12, 65535, v11 // (op 13410) 
v_lshrrev_b32 v13, 16, v11 // (op 13410) 
v_lshlrev_b32 v12, 16, v12 // (op 13410) 
v_lshlrev_b32 v13, 16, v13 // (op 13410) 
v_mul_f32 v22, v5, v12 // (op 13410) 
v_mul_f32 v23, v5, v13 // (op 13410) 


v_add_f32 v10, v20, v16 // (op 13411) 
v_add_f32 v11, v21, v17 // (op 13411) 
v_add_f32 v12, v22, v18 // (op 13411) 
v_add_f32 v13, v23, v19 // (op 13411) 


v_lshrrev_b32 v14, 16, v10 // (op 13412) 
v_lshrrev_b32 v15, 16, v11 // (op 13412) 
v_lshrrev_b32 v16, 16, v12 // (op 13412) 
v_lshrrev_b32 v17, 16, v13 // (op 13412) 


v_pack_b32_f16 v10, v14, v15 // (op 13375) 
v_pack_b32_f16 v11, v16, v17 // (op 13375) 

s_lshl_b32 s0, s4, 5 // (op 13375) 
s_mul_i32 s1, 7, s0 // (op 13375) 
s_add_u32 s1, s1, 160 // (op 13375) 

v_add_u32 v0, v9, s1 // (op 13375) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13375) Store value





v_add_u32 v0, 128, v8 // (op 13207) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13207) Load value

v_accvgpr_read v12, a128 // (op 13186) 
v_accvgpr_read v13, a129 // (op 13186) 
v_accvgpr_read v14, a130 // (op 13186) 
v_accvgpr_read v15, a131 // (op 13186) 
v_mul_f32 v16, v7, v12 // (op 13186) 
v_mul_f32 v17, v7, v13 // (op 13186) 
v_mul_f32 v18, v7, v14 // (op 13186) 
v_mul_f32 v19, v7, v15 // (op 13186) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13206) 
v_lshrrev_b32 v13, 16, v10 // (op 13206) 
v_lshlrev_b32 v12, 16, v12 // (op 13206) 
v_lshlrev_b32 v13, 16, v13 // (op 13206) 
v_mul_f32 v20, v5, v12 // (op 13206) 
v_mul_f32 v21, v5, v13 // (op 13206) 
v_and_b32 v12, 65535, v11 // (op 13206) 
v_lshrrev_b32 v13, 16, v11 // (op 13206) 
v_lshlrev_b32 v12, 16, v12 // (op 13206) 
v_lshlrev_b32 v13, 16, v13 // (op 13206) 
v_mul_f32 v22, v5, v12 // (op 13206) 
v_mul_f32 v23, v5, v13 // (op 13206) 


v_add_f32 v10, v20, v16 // (op 13187) 
v_add_f32 v11, v21, v17 // (op 13187) 
v_add_f32 v12, v22, v18 // (op 13187) 
v_add_f32 v13, v23, v19 // (op 13187) 


v_lshrrev_b32 v14, 16, v10 // (op 13188) 
v_lshrrev_b32 v15, 16, v11 // (op 13188) 
v_lshrrev_b32 v16, 16, v12 // (op 13188) 
v_lshrrev_b32 v17, 16, v13 // (op 13188) 


v_pack_b32_f16 v10, v14, v15 // (op 13190) 
v_pack_b32_f16 v11, v16, v17 // (op 13190) 

v_add_u32 v0, 128, v9 // (op 13190) 

buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13190) Store value





s_lshl_b32 s0, s2, 5 // (op 13212) 
s_add_u32 s0, s0, 128 // (op 13212) 
v_add_u32 v0, v8, s0 // (op 13212) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13212) Load value

v_accvgpr_read v12, a132 // (op 13244) 
v_accvgpr_read v13, a133 // (op 13244) 
v_accvgpr_read v14, a134 // (op 13244) 
v_accvgpr_read v15, a135 // (op 13244) 
v_mul_f32 v16, v7, v12 // (op 13244) 
v_mul_f32 v17, v7, v13 // (op 13244) 
v_mul_f32 v18, v7, v14 // (op 13244) 
v_mul_f32 v19, v7, v15 // (op 13244) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13209) 
v_lshrrev_b32 v13, 16, v10 // (op 13209) 
v_lshlrev_b32 v12, 16, v12 // (op 13209) 
v_lshlrev_b32 v13, 16, v13 // (op 13209) 
v_mul_f32 v20, v5, v12 // (op 13209) 
v_mul_f32 v21, v5, v13 // (op 13209) 
v_and_b32 v12, 65535, v11 // (op 13209) 
v_lshrrev_b32 v13, 16, v11 // (op 13209) 
v_lshlrev_b32 v12, 16, v12 // (op 13209) 
v_lshlrev_b32 v13, 16, v13 // (op 13209) 
v_mul_f32 v22, v5, v12 // (op 13209) 
v_mul_f32 v23, v5, v13 // (op 13209) 


v_add_f32 v10, v20, v16 // (op 13210) 
v_add_f32 v11, v21, v17 // (op 13210) 
v_add_f32 v12, v22, v18 // (op 13210) 
v_add_f32 v13, v23, v19 // (op 13210) 


v_lshrrev_b32 v14, 16, v10 // (op 13211) 
v_lshrrev_b32 v15, 16, v11 // (op 13211) 
v_lshrrev_b32 v16, 16, v12 // (op 13211) 
v_lshrrev_b32 v17, 16, v13 // (op 13211) 


v_pack_b32_f16 v10, v14, v15 // (op 13192) 
v_pack_b32_f16 v11, v16, v17 // (op 13192) 

s_lshl_b32 s0, s4, 5 // (op 13192) 
s_add_u32 s0, s0, 128 // (op 13192) 
v_add_u32 v0, v9, s0 // (op 13192) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13192) Store value





s_lshl_b32 s0, s2, 6 // (op 13217) 
s_add_u32 s0, s0, 128 // (op 13217) 
v_add_u32 v0, v8, s0 // (op 13217) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13217) Load value

v_accvgpr_read v12, a136 // (op 13246) 
v_accvgpr_read v13, a137 // (op 13246) 
v_accvgpr_read v14, a138 // (op 13246) 
v_accvgpr_read v15, a139 // (op 13246) 
v_mul_f32 v16, v7, v12 // (op 13246) 
v_mul_f32 v17, v7, v13 // (op 13246) 
v_mul_f32 v18, v7, v14 // (op 13246) 
v_mul_f32 v19, v7, v15 // (op 13246) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13214) 
v_lshrrev_b32 v13, 16, v10 // (op 13214) 
v_lshlrev_b32 v12, 16, v12 // (op 13214) 
v_lshlrev_b32 v13, 16, v13 // (op 13214) 
v_mul_f32 v20, v5, v12 // (op 13214) 
v_mul_f32 v21, v5, v13 // (op 13214) 
v_and_b32 v12, 65535, v11 // (op 13214) 
v_lshrrev_b32 v13, 16, v11 // (op 13214) 
v_lshlrev_b32 v12, 16, v12 // (op 13214) 
v_lshlrev_b32 v13, 16, v13 // (op 13214) 
v_mul_f32 v22, v5, v12 // (op 13214) 
v_mul_f32 v23, v5, v13 // (op 13214) 


v_add_f32 v10, v20, v16 // (op 13215) 
v_add_f32 v11, v21, v17 // (op 13215) 
v_add_f32 v12, v22, v18 // (op 13215) 
v_add_f32 v13, v23, v19 // (op 13215) 


v_lshrrev_b32 v14, 16, v10 // (op 13216) 
v_lshrrev_b32 v15, 16, v11 // (op 13216) 
v_lshrrev_b32 v16, 16, v12 // (op 13216) 
v_lshrrev_b32 v17, 16, v13 // (op 13216) 


v_pack_b32_f16 v10, v14, v15 // (op 13194) 
v_pack_b32_f16 v11, v16, v17 // (op 13194) 

s_lshl_b32 s0, s4, 6 // (op 13194) 
s_add_u32 s0, s0, 128 // (op 13194) 
v_add_u32 v0, v9, s0 // (op 13194) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13194) Store value





s_lshl_b32 s0, s2, 5 // (op 13222) 
s_mul_i32 s1, 3, s0 // (op 13222) 
s_add_u32 s1, s1, 128 // (op 13222) 

v_add_u32 v0, v8, s1 // (op 13222) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13222) Load value

v_accvgpr_read v12, a140 // (op 13248) 
v_accvgpr_read v13, a141 // (op 13248) 
v_accvgpr_read v14, a142 // (op 13248) 
v_accvgpr_read v15, a143 // (op 13248) 
v_mul_f32 v16, v7, v12 // (op 13248) 
v_mul_f32 v17, v7, v13 // (op 13248) 
v_mul_f32 v18, v7, v14 // (op 13248) 
v_mul_f32 v19, v7, v15 // (op 13248) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13219) 
v_lshrrev_b32 v13, 16, v10 // (op 13219) 
v_lshlrev_b32 v12, 16, v12 // (op 13219) 
v_lshlrev_b32 v13, 16, v13 // (op 13219) 
v_mul_f32 v20, v5, v12 // (op 13219) 
v_mul_f32 v21, v5, v13 // (op 13219) 
v_and_b32 v12, 65535, v11 // (op 13219) 
v_lshrrev_b32 v13, 16, v11 // (op 13219) 
v_lshlrev_b32 v12, 16, v12 // (op 13219) 
v_lshlrev_b32 v13, 16, v13 // (op 13219) 
v_mul_f32 v22, v5, v12 // (op 13219) 
v_mul_f32 v23, v5, v13 // (op 13219) 


v_add_f32 v10, v20, v16 // (op 13220) 
v_add_f32 v11, v21, v17 // (op 13220) 
v_add_f32 v12, v22, v18 // (op 13220) 
v_add_f32 v13, v23, v19 // (op 13220) 


v_lshrrev_b32 v14, 16, v10 // (op 13221) 
v_lshrrev_b32 v15, 16, v11 // (op 13221) 
v_lshrrev_b32 v16, 16, v12 // (op 13221) 
v_lshrrev_b32 v17, 16, v13 // (op 13221) 


v_pack_b32_f16 v10, v14, v15 // (op 13196) 
v_pack_b32_f16 v11, v16, v17 // (op 13196) 

s_lshl_b32 s0, s4, 5 // (op 13196) 
s_mul_i32 s1, 3, s0 // (op 13196) 
s_add_u32 s1, s1, 128 // (op 13196) 

v_add_u32 v0, v9, s1 // (op 13196) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13196) Store value





s_lshl_b32 s0, s2, 7 // (op 13227) 
s_add_u32 s0, s0, 128 // (op 13227) 
v_add_u32 v0, v8, s0 // (op 13227) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13227) Load value

v_accvgpr_read v12, a144 // (op 13250) 
v_accvgpr_read v13, a145 // (op 13250) 
v_accvgpr_read v14, a146 // (op 13250) 
v_accvgpr_read v15, a147 // (op 13250) 
v_mul_f32 v16, v7, v12 // (op 13250) 
v_mul_f32 v17, v7, v13 // (op 13250) 
v_mul_f32 v18, v7, v14 // (op 13250) 
v_mul_f32 v19, v7, v15 // (op 13250) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13224) 
v_lshrrev_b32 v13, 16, v10 // (op 13224) 
v_lshlrev_b32 v12, 16, v12 // (op 13224) 
v_lshlrev_b32 v13, 16, v13 // (op 13224) 
v_mul_f32 v20, v5, v12 // (op 13224) 
v_mul_f32 v21, v5, v13 // (op 13224) 
v_and_b32 v12, 65535, v11 // (op 13224) 
v_lshrrev_b32 v13, 16, v11 // (op 13224) 
v_lshlrev_b32 v12, 16, v12 // (op 13224) 
v_lshlrev_b32 v13, 16, v13 // (op 13224) 
v_mul_f32 v22, v5, v12 // (op 13224) 
v_mul_f32 v23, v5, v13 // (op 13224) 


v_add_f32 v10, v20, v16 // (op 13225) 
v_add_f32 v11, v21, v17 // (op 13225) 
v_add_f32 v12, v22, v18 // (op 13225) 
v_add_f32 v13, v23, v19 // (op 13225) 


v_lshrrev_b32 v14, 16, v10 // (op 13226) 
v_lshrrev_b32 v15, 16, v11 // (op 13226) 
v_lshrrev_b32 v16, 16, v12 // (op 13226) 
v_lshrrev_b32 v17, 16, v13 // (op 13226) 


v_pack_b32_f16 v10, v14, v15 // (op 13198) 
v_pack_b32_f16 v11, v16, v17 // (op 13198) 

s_lshl_b32 s0, s4, 7 // (op 13198) 
s_add_u32 s0, s0, 128 // (op 13198) 
v_add_u32 v0, v9, s0 // (op 13198) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13198) Store value





s_lshl_b32 s0, s2, 5 // (op 13232) 
s_mul_i32 s1, 5, s0 // (op 13232) 
s_add_u32 s1, s1, 128 // (op 13232) 

v_add_u32 v0, v8, s1 // (op 13232) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13232) Load value

v_accvgpr_read v12, a148 // (op 13252) 
v_accvgpr_read v13, a149 // (op 13252) 
v_accvgpr_read v14, a150 // (op 13252) 
v_accvgpr_read v15, a151 // (op 13252) 
v_mul_f32 v16, v7, v12 // (op 13252) 
v_mul_f32 v17, v7, v13 // (op 13252) 
v_mul_f32 v18, v7, v14 // (op 13252) 
v_mul_f32 v19, v7, v15 // (op 13252) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13229) 
v_lshrrev_b32 v13, 16, v10 // (op 13229) 
v_lshlrev_b32 v12, 16, v12 // (op 13229) 
v_lshlrev_b32 v13, 16, v13 // (op 13229) 
v_mul_f32 v20, v5, v12 // (op 13229) 
v_mul_f32 v21, v5, v13 // (op 13229) 
v_and_b32 v12, 65535, v11 // (op 13229) 
v_lshrrev_b32 v13, 16, v11 // (op 13229) 
v_lshlrev_b32 v12, 16, v12 // (op 13229) 
v_lshlrev_b32 v13, 16, v13 // (op 13229) 
v_mul_f32 v22, v5, v12 // (op 13229) 
v_mul_f32 v23, v5, v13 // (op 13229) 


v_add_f32 v10, v20, v16 // (op 13230) 
v_add_f32 v11, v21, v17 // (op 13230) 
v_add_f32 v12, v22, v18 // (op 13230) 
v_add_f32 v13, v23, v19 // (op 13230) 


v_lshrrev_b32 v14, 16, v10 // (op 13231) 
v_lshrrev_b32 v15, 16, v11 // (op 13231) 
v_lshrrev_b32 v16, 16, v12 // (op 13231) 
v_lshrrev_b32 v17, 16, v13 // (op 13231) 


v_pack_b32_f16 v10, v14, v15 // (op 13200) 
v_pack_b32_f16 v11, v16, v17 // (op 13200) 

s_lshl_b32 s0, s4, 5 // (op 13200) 
s_mul_i32 s1, 5, s0 // (op 13200) 
s_add_u32 s1, s1, 128 // (op 13200) 

v_add_u32 v0, v9, s1 // (op 13200) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13200) Store value





s_lshl_b32 s0, s2, 5 // (op 13237) 
s_mul_i32 s1, 6, s0 // (op 13237) 
s_add_u32 s1, s1, 128 // (op 13237) 

v_add_u32 v0, v8, s1 // (op 13237) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13237) Load value

v_accvgpr_read v12, a152 // (op 13254) 
v_accvgpr_read v13, a153 // (op 13254) 
v_accvgpr_read v14, a154 // (op 13254) 
v_accvgpr_read v15, a155 // (op 13254) 
v_mul_f32 v16, v7, v12 // (op 13254) 
v_mul_f32 v17, v7, v13 // (op 13254) 
v_mul_f32 v18, v7, v14 // (op 13254) 
v_mul_f32 v19, v7, v15 // (op 13254) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13234) 
v_lshrrev_b32 v13, 16, v10 // (op 13234) 
v_lshlrev_b32 v12, 16, v12 // (op 13234) 
v_lshlrev_b32 v13, 16, v13 // (op 13234) 
v_mul_f32 v20, v5, v12 // (op 13234) 
v_mul_f32 v21, v5, v13 // (op 13234) 
v_and_b32 v12, 65535, v11 // (op 13234) 
v_lshrrev_b32 v13, 16, v11 // (op 13234) 
v_lshlrev_b32 v12, 16, v12 // (op 13234) 
v_lshlrev_b32 v13, 16, v13 // (op 13234) 
v_mul_f32 v22, v5, v12 // (op 13234) 
v_mul_f32 v23, v5, v13 // (op 13234) 


v_add_f32 v10, v20, v16 // (op 13235) 
v_add_f32 v11, v21, v17 // (op 13235) 
v_add_f32 v12, v22, v18 // (op 13235) 
v_add_f32 v13, v23, v19 // (op 13235) 


v_lshrrev_b32 v14, 16, v10 // (op 13236) 
v_lshrrev_b32 v15, 16, v11 // (op 13236) 
v_lshrrev_b32 v16, 16, v12 // (op 13236) 
v_lshrrev_b32 v17, 16, v13 // (op 13236) 


v_pack_b32_f16 v10, v14, v15 // (op 13202) 
v_pack_b32_f16 v11, v16, v17 // (op 13202) 

s_lshl_b32 s0, s4, 5 // (op 13202) 
s_mul_i32 s1, 6, s0 // (op 13202) 
s_add_u32 s1, s1, 128 // (op 13202) 

v_add_u32 v0, v9, s1 // (op 13202) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13202) Store value





s_lshl_b32 s0, s2, 5 // (op 13242) 
s_mul_i32 s1, 7, s0 // (op 13242) 
s_add_u32 s1, s1, 128 // (op 13242) 

v_add_u32 v0, v8, s1 // (op 13242) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13242) Load value

v_accvgpr_read v12, a156 // (op 13256) 
v_accvgpr_read v13, a157 // (op 13256) 
v_accvgpr_read v14, a158 // (op 13256) 
v_accvgpr_read v15, a159 // (op 13256) 
v_mul_f32 v16, v7, v12 // (op 13256) 
v_mul_f32 v17, v7, v13 // (op 13256) 
v_mul_f32 v18, v7, v14 // (op 13256) 
v_mul_f32 v19, v7, v15 // (op 13256) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13239) 
v_lshrrev_b32 v13, 16, v10 // (op 13239) 
v_lshlrev_b32 v12, 16, v12 // (op 13239) 
v_lshlrev_b32 v13, 16, v13 // (op 13239) 
v_mul_f32 v20, v5, v12 // (op 13239) 
v_mul_f32 v21, v5, v13 // (op 13239) 
v_and_b32 v12, 65535, v11 // (op 13239) 
v_lshrrev_b32 v13, 16, v11 // (op 13239) 
v_lshlrev_b32 v12, 16, v12 // (op 13239) 
v_lshlrev_b32 v13, 16, v13 // (op 13239) 
v_mul_f32 v22, v5, v12 // (op 13239) 
v_mul_f32 v23, v5, v13 // (op 13239) 


v_add_f32 v10, v20, v16 // (op 13240) 
v_add_f32 v11, v21, v17 // (op 13240) 
v_add_f32 v12, v22, v18 // (op 13240) 
v_add_f32 v13, v23, v19 // (op 13240) 


v_lshrrev_b32 v14, 16, v10 // (op 13241) 
v_lshrrev_b32 v15, 16, v11 // (op 13241) 
v_lshrrev_b32 v16, 16, v12 // (op 13241) 
v_lshrrev_b32 v17, 16, v13 // (op 13241) 


v_pack_b32_f16 v10, v14, v15 // (op 13204) 
v_pack_b32_f16 v11, v16, v17 // (op 13204) 

s_lshl_b32 s0, s4, 5 // (op 13204) 
s_mul_i32 s1, 7, s0 // (op 13204) 
s_add_u32 s1, s1, 128 // (op 13204) 

v_add_u32 v0, v9, s1 // (op 13204) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13204) Store value





v_add_u32 v0, 96, v8 // (op 13036) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13036) Load value

v_accvgpr_read v12, a96 // (op 13015) 
v_accvgpr_read v13, a97 // (op 13015) 
v_accvgpr_read v14, a98 // (op 13015) 
v_accvgpr_read v15, a99 // (op 13015) 
v_mul_f32 v16, v7, v12 // (op 13015) 
v_mul_f32 v17, v7, v13 // (op 13015) 
v_mul_f32 v18, v7, v14 // (op 13015) 
v_mul_f32 v19, v7, v15 // (op 13015) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13035) 
v_lshrrev_b32 v13, 16, v10 // (op 13035) 
v_lshlrev_b32 v12, 16, v12 // (op 13035) 
v_lshlrev_b32 v13, 16, v13 // (op 13035) 
v_mul_f32 v20, v5, v12 // (op 13035) 
v_mul_f32 v21, v5, v13 // (op 13035) 
v_and_b32 v12, 65535, v11 // (op 13035) 
v_lshrrev_b32 v13, 16, v11 // (op 13035) 
v_lshlrev_b32 v12, 16, v12 // (op 13035) 
v_lshlrev_b32 v13, 16, v13 // (op 13035) 
v_mul_f32 v22, v5, v12 // (op 13035) 
v_mul_f32 v23, v5, v13 // (op 13035) 


v_add_f32 v10, v20, v16 // (op 13016) 
v_add_f32 v11, v21, v17 // (op 13016) 
v_add_f32 v12, v22, v18 // (op 13016) 
v_add_f32 v13, v23, v19 // (op 13016) 


v_lshrrev_b32 v14, 16, v10 // (op 13017) 
v_lshrrev_b32 v15, 16, v11 // (op 13017) 
v_lshrrev_b32 v16, 16, v12 // (op 13017) 
v_lshrrev_b32 v17, 16, v13 // (op 13017) 


v_pack_b32_f16 v10, v14, v15 // (op 13019) 
v_pack_b32_f16 v11, v16, v17 // (op 13019) 

v_add_u32 v0, 96, v9 // (op 13019) 

buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13019) Store value





s_lshl_b32 s0, s2, 5 // (op 13041) 
s_add_u32 s0, s0, 96 // (op 13041) 
v_add_u32 v0, v8, s0 // (op 13041) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13041) Load value

v_accvgpr_read v12, a100 // (op 13073) 
v_accvgpr_read v13, a101 // (op 13073) 
v_accvgpr_read v14, a102 // (op 13073) 
v_accvgpr_read v15, a103 // (op 13073) 
v_mul_f32 v16, v7, v12 // (op 13073) 
v_mul_f32 v17, v7, v13 // (op 13073) 
v_mul_f32 v18, v7, v14 // (op 13073) 
v_mul_f32 v19, v7, v15 // (op 13073) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13038) 
v_lshrrev_b32 v13, 16, v10 // (op 13038) 
v_lshlrev_b32 v12, 16, v12 // (op 13038) 
v_lshlrev_b32 v13, 16, v13 // (op 13038) 
v_mul_f32 v20, v5, v12 // (op 13038) 
v_mul_f32 v21, v5, v13 // (op 13038) 
v_and_b32 v12, 65535, v11 // (op 13038) 
v_lshrrev_b32 v13, 16, v11 // (op 13038) 
v_lshlrev_b32 v12, 16, v12 // (op 13038) 
v_lshlrev_b32 v13, 16, v13 // (op 13038) 
v_mul_f32 v22, v5, v12 // (op 13038) 
v_mul_f32 v23, v5, v13 // (op 13038) 


v_add_f32 v10, v20, v16 // (op 13039) 
v_add_f32 v11, v21, v17 // (op 13039) 
v_add_f32 v12, v22, v18 // (op 13039) 
v_add_f32 v13, v23, v19 // (op 13039) 


v_lshrrev_b32 v14, 16, v10 // (op 13040) 
v_lshrrev_b32 v15, 16, v11 // (op 13040) 
v_lshrrev_b32 v16, 16, v12 // (op 13040) 
v_lshrrev_b32 v17, 16, v13 // (op 13040) 


v_pack_b32_f16 v10, v14, v15 // (op 13021) 
v_pack_b32_f16 v11, v16, v17 // (op 13021) 

s_lshl_b32 s0, s4, 5 // (op 13021) 
s_add_u32 s0, s0, 96 // (op 13021) 
v_add_u32 v0, v9, s0 // (op 13021) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13021) Store value





s_lshl_b32 s0, s2, 6 // (op 13046) 
s_add_u32 s0, s0, 96 // (op 13046) 
v_add_u32 v0, v8, s0 // (op 13046) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13046) Load value

v_accvgpr_read v12, a104 // (op 13075) 
v_accvgpr_read v13, a105 // (op 13075) 
v_accvgpr_read v14, a106 // (op 13075) 
v_accvgpr_read v15, a107 // (op 13075) 
v_mul_f32 v16, v7, v12 // (op 13075) 
v_mul_f32 v17, v7, v13 // (op 13075) 
v_mul_f32 v18, v7, v14 // (op 13075) 
v_mul_f32 v19, v7, v15 // (op 13075) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13043) 
v_lshrrev_b32 v13, 16, v10 // (op 13043) 
v_lshlrev_b32 v12, 16, v12 // (op 13043) 
v_lshlrev_b32 v13, 16, v13 // (op 13043) 
v_mul_f32 v20, v5, v12 // (op 13043) 
v_mul_f32 v21, v5, v13 // (op 13043) 
v_and_b32 v12, 65535, v11 // (op 13043) 
v_lshrrev_b32 v13, 16, v11 // (op 13043) 
v_lshlrev_b32 v12, 16, v12 // (op 13043) 
v_lshlrev_b32 v13, 16, v13 // (op 13043) 
v_mul_f32 v22, v5, v12 // (op 13043) 
v_mul_f32 v23, v5, v13 // (op 13043) 


v_add_f32 v10, v20, v16 // (op 13044) 
v_add_f32 v11, v21, v17 // (op 13044) 
v_add_f32 v12, v22, v18 // (op 13044) 
v_add_f32 v13, v23, v19 // (op 13044) 


v_lshrrev_b32 v14, 16, v10 // (op 13045) 
v_lshrrev_b32 v15, 16, v11 // (op 13045) 
v_lshrrev_b32 v16, 16, v12 // (op 13045) 
v_lshrrev_b32 v17, 16, v13 // (op 13045) 


v_pack_b32_f16 v10, v14, v15 // (op 13023) 
v_pack_b32_f16 v11, v16, v17 // (op 13023) 

s_lshl_b32 s0, s4, 6 // (op 13023) 
s_add_u32 s0, s0, 96 // (op 13023) 
v_add_u32 v0, v9, s0 // (op 13023) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13023) Store value





s_lshl_b32 s0, s2, 5 // (op 13051) 
s_mul_i32 s1, 3, s0 // (op 13051) 
s_add_u32 s1, s1, 96 // (op 13051) 

v_add_u32 v0, v8, s1 // (op 13051) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13051) Load value

v_accvgpr_read v12, a108 // (op 13077) 
v_accvgpr_read v13, a109 // (op 13077) 
v_accvgpr_read v14, a110 // (op 13077) 
v_accvgpr_read v15, a111 // (op 13077) 
v_mul_f32 v16, v7, v12 // (op 13077) 
v_mul_f32 v17, v7, v13 // (op 13077) 
v_mul_f32 v18, v7, v14 // (op 13077) 
v_mul_f32 v19, v7, v15 // (op 13077) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13048) 
v_lshrrev_b32 v13, 16, v10 // (op 13048) 
v_lshlrev_b32 v12, 16, v12 // (op 13048) 
v_lshlrev_b32 v13, 16, v13 // (op 13048) 
v_mul_f32 v20, v5, v12 // (op 13048) 
v_mul_f32 v21, v5, v13 // (op 13048) 
v_and_b32 v12, 65535, v11 // (op 13048) 
v_lshrrev_b32 v13, 16, v11 // (op 13048) 
v_lshlrev_b32 v12, 16, v12 // (op 13048) 
v_lshlrev_b32 v13, 16, v13 // (op 13048) 
v_mul_f32 v22, v5, v12 // (op 13048) 
v_mul_f32 v23, v5, v13 // (op 13048) 


v_add_f32 v10, v20, v16 // (op 13049) 
v_add_f32 v11, v21, v17 // (op 13049) 
v_add_f32 v12, v22, v18 // (op 13049) 
v_add_f32 v13, v23, v19 // (op 13049) 


v_lshrrev_b32 v14, 16, v10 // (op 13050) 
v_lshrrev_b32 v15, 16, v11 // (op 13050) 
v_lshrrev_b32 v16, 16, v12 // (op 13050) 
v_lshrrev_b32 v17, 16, v13 // (op 13050) 


v_pack_b32_f16 v10, v14, v15 // (op 13025) 
v_pack_b32_f16 v11, v16, v17 // (op 13025) 

s_lshl_b32 s0, s4, 5 // (op 13025) 
s_mul_i32 s1, 3, s0 // (op 13025) 
s_add_u32 s1, s1, 96 // (op 13025) 

v_add_u32 v0, v9, s1 // (op 13025) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13025) Store value





s_lshl_b32 s0, s2, 7 // (op 13056) 
s_add_u32 s0, s0, 96 // (op 13056) 
v_add_u32 v0, v8, s0 // (op 13056) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13056) Load value

v_accvgpr_read v12, a112 // (op 13079) 
v_accvgpr_read v13, a113 // (op 13079) 
v_accvgpr_read v14, a114 // (op 13079) 
v_accvgpr_read v15, a115 // (op 13079) 
v_mul_f32 v16, v7, v12 // (op 13079) 
v_mul_f32 v17, v7, v13 // (op 13079) 
v_mul_f32 v18, v7, v14 // (op 13079) 
v_mul_f32 v19, v7, v15 // (op 13079) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13053) 
v_lshrrev_b32 v13, 16, v10 // (op 13053) 
v_lshlrev_b32 v12, 16, v12 // (op 13053) 
v_lshlrev_b32 v13, 16, v13 // (op 13053) 
v_mul_f32 v20, v5, v12 // (op 13053) 
v_mul_f32 v21, v5, v13 // (op 13053) 
v_and_b32 v12, 65535, v11 // (op 13053) 
v_lshrrev_b32 v13, 16, v11 // (op 13053) 
v_lshlrev_b32 v12, 16, v12 // (op 13053) 
v_lshlrev_b32 v13, 16, v13 // (op 13053) 
v_mul_f32 v22, v5, v12 // (op 13053) 
v_mul_f32 v23, v5, v13 // (op 13053) 


v_add_f32 v10, v20, v16 // (op 13054) 
v_add_f32 v11, v21, v17 // (op 13054) 
v_add_f32 v12, v22, v18 // (op 13054) 
v_add_f32 v13, v23, v19 // (op 13054) 


v_lshrrev_b32 v14, 16, v10 // (op 13055) 
v_lshrrev_b32 v15, 16, v11 // (op 13055) 
v_lshrrev_b32 v16, 16, v12 // (op 13055) 
v_lshrrev_b32 v17, 16, v13 // (op 13055) 


v_pack_b32_f16 v10, v14, v15 // (op 13027) 
v_pack_b32_f16 v11, v16, v17 // (op 13027) 

s_lshl_b32 s0, s4, 7 // (op 13027) 
s_add_u32 s0, s0, 96 // (op 13027) 
v_add_u32 v0, v9, s0 // (op 13027) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13027) Store value





s_lshl_b32 s0, s2, 5 // (op 13061) 
s_mul_i32 s1, 5, s0 // (op 13061) 
s_add_u32 s1, s1, 96 // (op 13061) 

v_add_u32 v0, v8, s1 // (op 13061) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13061) Load value

v_accvgpr_read v12, a116 // (op 13081) 
v_accvgpr_read v13, a117 // (op 13081) 
v_accvgpr_read v14, a118 // (op 13081) 
v_accvgpr_read v15, a119 // (op 13081) 
v_mul_f32 v16, v7, v12 // (op 13081) 
v_mul_f32 v17, v7, v13 // (op 13081) 
v_mul_f32 v18, v7, v14 // (op 13081) 
v_mul_f32 v19, v7, v15 // (op 13081) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13058) 
v_lshrrev_b32 v13, 16, v10 // (op 13058) 
v_lshlrev_b32 v12, 16, v12 // (op 13058) 
v_lshlrev_b32 v13, 16, v13 // (op 13058) 
v_mul_f32 v20, v5, v12 // (op 13058) 
v_mul_f32 v21, v5, v13 // (op 13058) 
v_and_b32 v12, 65535, v11 // (op 13058) 
v_lshrrev_b32 v13, 16, v11 // (op 13058) 
v_lshlrev_b32 v12, 16, v12 // (op 13058) 
v_lshlrev_b32 v13, 16, v13 // (op 13058) 
v_mul_f32 v22, v5, v12 // (op 13058) 
v_mul_f32 v23, v5, v13 // (op 13058) 


v_add_f32 v10, v20, v16 // (op 13059) 
v_add_f32 v11, v21, v17 // (op 13059) 
v_add_f32 v12, v22, v18 // (op 13059) 
v_add_f32 v13, v23, v19 // (op 13059) 


v_lshrrev_b32 v14, 16, v10 // (op 13060) 
v_lshrrev_b32 v15, 16, v11 // (op 13060) 
v_lshrrev_b32 v16, 16, v12 // (op 13060) 
v_lshrrev_b32 v17, 16, v13 // (op 13060) 


v_pack_b32_f16 v10, v14, v15 // (op 13029) 
v_pack_b32_f16 v11, v16, v17 // (op 13029) 

s_lshl_b32 s0, s4, 5 // (op 13029) 
s_mul_i32 s1, 5, s0 // (op 13029) 
s_add_u32 s1, s1, 96 // (op 13029) 

v_add_u32 v0, v9, s1 // (op 13029) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13029) Store value





s_lshl_b32 s0, s2, 5 // (op 13066) 
s_mul_i32 s1, 6, s0 // (op 13066) 
s_add_u32 s1, s1, 96 // (op 13066) 

v_add_u32 v0, v8, s1 // (op 13066) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13066) Load value

v_accvgpr_read v12, a120 // (op 13083) 
v_accvgpr_read v13, a121 // (op 13083) 
v_accvgpr_read v14, a122 // (op 13083) 
v_accvgpr_read v15, a123 // (op 13083) 
v_mul_f32 v16, v7, v12 // (op 13083) 
v_mul_f32 v17, v7, v13 // (op 13083) 
v_mul_f32 v18, v7, v14 // (op 13083) 
v_mul_f32 v19, v7, v15 // (op 13083) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13063) 
v_lshrrev_b32 v13, 16, v10 // (op 13063) 
v_lshlrev_b32 v12, 16, v12 // (op 13063) 
v_lshlrev_b32 v13, 16, v13 // (op 13063) 
v_mul_f32 v20, v5, v12 // (op 13063) 
v_mul_f32 v21, v5, v13 // (op 13063) 
v_and_b32 v12, 65535, v11 // (op 13063) 
v_lshrrev_b32 v13, 16, v11 // (op 13063) 
v_lshlrev_b32 v12, 16, v12 // (op 13063) 
v_lshlrev_b32 v13, 16, v13 // (op 13063) 
v_mul_f32 v22, v5, v12 // (op 13063) 
v_mul_f32 v23, v5, v13 // (op 13063) 


v_add_f32 v10, v20, v16 // (op 13064) 
v_add_f32 v11, v21, v17 // (op 13064) 
v_add_f32 v12, v22, v18 // (op 13064) 
v_add_f32 v13, v23, v19 // (op 13064) 


v_lshrrev_b32 v14, 16, v10 // (op 13065) 
v_lshrrev_b32 v15, 16, v11 // (op 13065) 
v_lshrrev_b32 v16, 16, v12 // (op 13065) 
v_lshrrev_b32 v17, 16, v13 // (op 13065) 


v_pack_b32_f16 v10, v14, v15 // (op 13031) 
v_pack_b32_f16 v11, v16, v17 // (op 13031) 

s_lshl_b32 s0, s4, 5 // (op 13031) 
s_mul_i32 s1, 6, s0 // (op 13031) 
s_add_u32 s1, s1, 96 // (op 13031) 

v_add_u32 v0, v9, s1 // (op 13031) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13031) Store value





s_lshl_b32 s0, s2, 5 // (op 13071) 
s_mul_i32 s1, 7, s0 // (op 13071) 
s_add_u32 s1, s1, 96 // (op 13071) 

v_add_u32 v0, v8, s1 // (op 13071) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13071) Load value

v_accvgpr_read v12, a124 // (op 13085) 
v_accvgpr_read v13, a125 // (op 13085) 
v_accvgpr_read v14, a126 // (op 13085) 
v_accvgpr_read v15, a127 // (op 13085) 
v_mul_f32 v16, v7, v12 // (op 13085) 
v_mul_f32 v17, v7, v13 // (op 13085) 
v_mul_f32 v18, v7, v14 // (op 13085) 
v_mul_f32 v19, v7, v15 // (op 13085) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13068) 
v_lshrrev_b32 v13, 16, v10 // (op 13068) 
v_lshlrev_b32 v12, 16, v12 // (op 13068) 
v_lshlrev_b32 v13, 16, v13 // (op 13068) 
v_mul_f32 v20, v5, v12 // (op 13068) 
v_mul_f32 v21, v5, v13 // (op 13068) 
v_and_b32 v12, 65535, v11 // (op 13068) 
v_lshrrev_b32 v13, 16, v11 // (op 13068) 
v_lshlrev_b32 v12, 16, v12 // (op 13068) 
v_lshlrev_b32 v13, 16, v13 // (op 13068) 
v_mul_f32 v22, v5, v12 // (op 13068) 
v_mul_f32 v23, v5, v13 // (op 13068) 


v_add_f32 v10, v20, v16 // (op 13069) 
v_add_f32 v11, v21, v17 // (op 13069) 
v_add_f32 v12, v22, v18 // (op 13069) 
v_add_f32 v13, v23, v19 // (op 13069) 


v_lshrrev_b32 v14, 16, v10 // (op 13070) 
v_lshrrev_b32 v15, 16, v11 // (op 13070) 
v_lshrrev_b32 v16, 16, v12 // (op 13070) 
v_lshrrev_b32 v17, 16, v13 // (op 13070) 


v_pack_b32_f16 v10, v14, v15 // (op 13033) 
v_pack_b32_f16 v11, v16, v17 // (op 13033) 

s_lshl_b32 s0, s4, 5 // (op 13033) 
s_mul_i32 s1, 7, s0 // (op 13033) 
s_add_u32 s1, s1, 96 // (op 13033) 

v_add_u32 v0, v9, s1 // (op 13033) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13033) Store value





v_add_u32 v0, 64, v8 // (op 12865) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12865) Load value

v_accvgpr_read v12, a64 // (op 12844) 
v_accvgpr_read v13, a65 // (op 12844) 
v_accvgpr_read v14, a66 // (op 12844) 
v_accvgpr_read v15, a67 // (op 12844) 
v_mul_f32 v16, v7, v12 // (op 12844) 
v_mul_f32 v17, v7, v13 // (op 12844) 
v_mul_f32 v18, v7, v14 // (op 12844) 
v_mul_f32 v19, v7, v15 // (op 12844) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12864) 
v_lshrrev_b32 v13, 16, v10 // (op 12864) 
v_lshlrev_b32 v12, 16, v12 // (op 12864) 
v_lshlrev_b32 v13, 16, v13 // (op 12864) 
v_mul_f32 v20, v5, v12 // (op 12864) 
v_mul_f32 v21, v5, v13 // (op 12864) 
v_and_b32 v12, 65535, v11 // (op 12864) 
v_lshrrev_b32 v13, 16, v11 // (op 12864) 
v_lshlrev_b32 v12, 16, v12 // (op 12864) 
v_lshlrev_b32 v13, 16, v13 // (op 12864) 
v_mul_f32 v22, v5, v12 // (op 12864) 
v_mul_f32 v23, v5, v13 // (op 12864) 


v_add_f32 v10, v20, v16 // (op 12845) 
v_add_f32 v11, v21, v17 // (op 12845) 
v_add_f32 v12, v22, v18 // (op 12845) 
v_add_f32 v13, v23, v19 // (op 12845) 


v_lshrrev_b32 v14, 16, v10 // (op 12846) 
v_lshrrev_b32 v15, 16, v11 // (op 12846) 
v_lshrrev_b32 v16, 16, v12 // (op 12846) 
v_lshrrev_b32 v17, 16, v13 // (op 12846) 


v_pack_b32_f16 v10, v14, v15 // (op 12848) 
v_pack_b32_f16 v11, v16, v17 // (op 12848) 

v_add_u32 v0, 64, v9 // (op 12848) 

buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12848) Store value





s_lshl_b32 s0, s2, 5 // (op 12870) 
s_add_u32 s0, s0, 64 // (op 12870) 
v_add_u32 v0, v8, s0 // (op 12870) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12870) Load value

v_accvgpr_read v12, a68 // (op 12902) 
v_accvgpr_read v13, a69 // (op 12902) 
v_accvgpr_read v14, a70 // (op 12902) 
v_accvgpr_read v15, a71 // (op 12902) 
v_mul_f32 v16, v7, v12 // (op 12902) 
v_mul_f32 v17, v7, v13 // (op 12902) 
v_mul_f32 v18, v7, v14 // (op 12902) 
v_mul_f32 v19, v7, v15 // (op 12902) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12867) 
v_lshrrev_b32 v13, 16, v10 // (op 12867) 
v_lshlrev_b32 v12, 16, v12 // (op 12867) 
v_lshlrev_b32 v13, 16, v13 // (op 12867) 
v_mul_f32 v20, v5, v12 // (op 12867) 
v_mul_f32 v21, v5, v13 // (op 12867) 
v_and_b32 v12, 65535, v11 // (op 12867) 
v_lshrrev_b32 v13, 16, v11 // (op 12867) 
v_lshlrev_b32 v12, 16, v12 // (op 12867) 
v_lshlrev_b32 v13, 16, v13 // (op 12867) 
v_mul_f32 v22, v5, v12 // (op 12867) 
v_mul_f32 v23, v5, v13 // (op 12867) 


v_add_f32 v10, v20, v16 // (op 12868) 
v_add_f32 v11, v21, v17 // (op 12868) 
v_add_f32 v12, v22, v18 // (op 12868) 
v_add_f32 v13, v23, v19 // (op 12868) 


v_lshrrev_b32 v14, 16, v10 // (op 12869) 
v_lshrrev_b32 v15, 16, v11 // (op 12869) 
v_lshrrev_b32 v16, 16, v12 // (op 12869) 
v_lshrrev_b32 v17, 16, v13 // (op 12869) 


v_pack_b32_f16 v10, v14, v15 // (op 12850) 
v_pack_b32_f16 v11, v16, v17 // (op 12850) 

s_lshl_b32 s0, s4, 5 // (op 12850) 
s_add_u32 s0, s0, 64 // (op 12850) 
v_add_u32 v0, v9, s0 // (op 12850) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12850) Store value





s_lshl_b32 s0, s2, 6 // (op 12875) 
s_add_u32 s0, s0, 64 // (op 12875) 
v_add_u32 v0, v8, s0 // (op 12875) 


buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12875) Load value

v_accvgpr_read v12, a72 // (op 12904) 
v_accvgpr_read v13, a73 // (op 12904) 
v_accvgpr_read v14, a74 // (op 12904) 
v_accvgpr_read v15, a75 // (op 12904) 
v_mul_f32 v16, v7, v12 // (op 12904) 
v_mul_f32 v17, v7, v13 // (op 12904) 
v_mul_f32 v18, v7, v14 // (op 12904) 
v_mul_f32 v19, v7, v15 // (op 12904) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12872) 
v_lshrrev_b32 v13, 16, v10 // (op 12872) 
v_lshlrev_b32 v12, 16, v12 // (op 12872) 
v_lshlrev_b32 v13, 16, v13 // (op 12872) 
v_mul_f32 v20, v5, v12 // (op 12872) 
v_mul_f32 v21, v5, v13 // (op 12872) 
v_and_b32 v12, 65535, v11 // (op 12872) 
v_lshrrev_b32 v13, 16, v11 // (op 12872) 
v_lshlrev_b32 v12, 16, v12 // (op 12872) 
v_lshlrev_b32 v13, 16, v13 // (op 12872) 
v_mul_f32 v22, v5, v12 // (op 12872) 
v_mul_f32 v23, v5, v13 // (op 12872) 


v_add_f32 v10, v20, v16 // (op 12873) 
v_add_f32 v11, v21, v17 // (op 12873) 
v_add_f32 v12, v22, v18 // (op 12873) 
v_add_f32 v13, v23, v19 // (op 12873) 


v_lshrrev_b32 v14, 16, v10 // (op 12874) 
v_lshrrev_b32 v15, 16, v11 // (op 12874) 
v_lshrrev_b32 v16, 16, v12 // (op 12874) 
v_lshrrev_b32 v17, 16, v13 // (op 12874) 


v_pack_b32_f16 v10, v14, v15 // (op 12852) 
v_pack_b32_f16 v11, v16, v17 // (op 12852) 

s_lshl_b32 s0, s4, 6 // (op 12852) 
s_add_u32 s0, s0, 64 // (op 12852) 
v_add_u32 v0, v9, s0 // (op 12852) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12852) Store value





s_lshl_b32 s0, s2, 5 // (op 12880) 
s_mul_i32 s1, 3, s0 // (op 12880) 
s_add_u32 s1, s1, 64 // (op 12880) 

v_add_u32 v0, v8, s1 // (op 12880) 


v_accvgpr_read v10, a76 // (op 12906) 
v_accvgpr_read v11, a77 // (op 12906) 
v_accvgpr_read v12, a78 // (op 12906) 
v_accvgpr_read v13, a79 // (op 12906) 
v_mul_f32 v14, v7, v10 // (op 12906) 
v_mul_f32 v15, v7, v11 // (op 12906) 
v_mul_f32 v16, v7, v12 // (op 12906) 
v_mul_f32 v17, v7, v13 // (op 12906) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12880) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12877) 
v_lshrrev_b32 v13, 16, v10 // (op 12877) 
v_lshlrev_b32 v12, 16, v12 // (op 12877) 
v_lshlrev_b32 v13, 16, v13 // (op 12877) 
v_mul_f32 v18, v5, v12 // (op 12877) 
v_mul_f32 v19, v5, v13 // (op 12877) 
v_and_b32 v12, 65535, v11 // (op 12877) 
v_lshrrev_b32 v13, 16, v11 // (op 12877) 
v_lshlrev_b32 v12, 16, v12 // (op 12877) 
v_lshlrev_b32 v13, 16, v13 // (op 12877) 
v_mul_f32 v20, v5, v12 // (op 12877) 
v_mul_f32 v21, v5, v13 // (op 12877) 


v_add_f32 v10, v18, v14 // (op 12878) 
v_add_f32 v11, v19, v15 // (op 12878) 
v_add_f32 v12, v20, v16 // (op 12878) 
v_add_f32 v13, v21, v17 // (op 12878) 


v_lshrrev_b32 v14, 16, v10 // (op 12879) 
v_lshrrev_b32 v15, 16, v11 // (op 12879) 
v_lshrrev_b32 v16, 16, v12 // (op 12879) 
v_lshrrev_b32 v17, 16, v13 // (op 12879) 


v_pack_b32_f16 v10, v14, v15 // (op 12854) 
v_pack_b32_f16 v11, v16, v17 // (op 12854) 

s_lshl_b32 s0, s4, 5 // (op 12854) 
s_mul_i32 s1, 3, s0 // (op 12854) 
s_add_u32 s1, s1, 64 // (op 12854) 

v_add_u32 v0, v9, s1 // (op 12854) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12854) Store value





s_lshl_b32 s0, s2, 7 // (op 12885) 
s_add_u32 s0, s0, 64 // (op 12885) 
v_add_u32 v0, v8, s0 // (op 12885) 


v_accvgpr_read v10, a80 // (op 12908) 
v_accvgpr_read v11, a81 // (op 12908) 
v_accvgpr_read v12, a82 // (op 12908) 
v_accvgpr_read v13, a83 // (op 12908) 
v_mul_f32 v14, v7, v10 // (op 12908) 
v_mul_f32 v15, v7, v11 // (op 12908) 
v_mul_f32 v16, v7, v12 // (op 12908) 
v_mul_f32 v17, v7, v13 // (op 12908) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12885) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12882) 
v_lshrrev_b32 v13, 16, v10 // (op 12882) 
v_lshlrev_b32 v12, 16, v12 // (op 12882) 
v_lshlrev_b32 v13, 16, v13 // (op 12882) 
v_mul_f32 v18, v5, v12 // (op 12882) 
v_mul_f32 v19, v5, v13 // (op 12882) 
v_and_b32 v12, 65535, v11 // (op 12882) 
v_lshrrev_b32 v13, 16, v11 // (op 12882) 
v_lshlrev_b32 v12, 16, v12 // (op 12882) 
v_lshlrev_b32 v13, 16, v13 // (op 12882) 
v_mul_f32 v20, v5, v12 // (op 12882) 
v_mul_f32 v21, v5, v13 // (op 12882) 


v_add_f32 v10, v18, v14 // (op 12883) 
v_add_f32 v11, v19, v15 // (op 12883) 
v_add_f32 v12, v20, v16 // (op 12883) 
v_add_f32 v13, v21, v17 // (op 12883) 


v_lshrrev_b32 v14, 16, v10 // (op 12884) 
v_lshrrev_b32 v15, 16, v11 // (op 12884) 
v_lshrrev_b32 v16, 16, v12 // (op 12884) 
v_lshrrev_b32 v17, 16, v13 // (op 12884) 


v_pack_b32_f16 v10, v14, v15 // (op 12856) 
v_pack_b32_f16 v11, v16, v17 // (op 12856) 

s_lshl_b32 s0, s4, 7 // (op 12856) 
s_add_u32 s0, s0, 64 // (op 12856) 
v_add_u32 v0, v9, s0 // (op 12856) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12856) Store value





s_lshl_b32 s0, s2, 5 // (op 12890) 
s_mul_i32 s1, 5, s0 // (op 12890) 
s_add_u32 s1, s1, 64 // (op 12890) 

v_add_u32 v0, v8, s1 // (op 12890) 


v_accvgpr_read v10, a84 // (op 12910) 
v_accvgpr_read v11, a85 // (op 12910) 
v_accvgpr_read v12, a86 // (op 12910) 
v_accvgpr_read v13, a87 // (op 12910) 
v_mul_f32 v14, v7, v10 // (op 12910) 
v_mul_f32 v15, v7, v11 // (op 12910) 
v_mul_f32 v16, v7, v12 // (op 12910) 
v_mul_f32 v17, v7, v13 // (op 12910) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12890) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12887) 
v_lshrrev_b32 v13, 16, v10 // (op 12887) 
v_lshlrev_b32 v12, 16, v12 // (op 12887) 
v_lshlrev_b32 v13, 16, v13 // (op 12887) 
v_mul_f32 v18, v5, v12 // (op 12887) 
v_mul_f32 v19, v5, v13 // (op 12887) 
v_and_b32 v12, 65535, v11 // (op 12887) 
v_lshrrev_b32 v13, 16, v11 // (op 12887) 
v_lshlrev_b32 v12, 16, v12 // (op 12887) 
v_lshlrev_b32 v13, 16, v13 // (op 12887) 
v_mul_f32 v20, v5, v12 // (op 12887) 
v_mul_f32 v21, v5, v13 // (op 12887) 


v_add_f32 v10, v18, v14 // (op 12888) 
v_add_f32 v11, v19, v15 // (op 12888) 
v_add_f32 v12, v20, v16 // (op 12888) 
v_add_f32 v13, v21, v17 // (op 12888) 


v_lshrrev_b32 v14, 16, v10 // (op 12889) 
v_lshrrev_b32 v15, 16, v11 // (op 12889) 
v_lshrrev_b32 v16, 16, v12 // (op 12889) 
v_lshrrev_b32 v17, 16, v13 // (op 12889) 


v_pack_b32_f16 v10, v14, v15 // (op 12858) 
v_pack_b32_f16 v11, v16, v17 // (op 12858) 

s_lshl_b32 s0, s4, 5 // (op 12858) 
s_mul_i32 s1, 5, s0 // (op 12858) 
s_add_u32 s1, s1, 64 // (op 12858) 

v_add_u32 v0, v9, s1 // (op 12858) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12858) Store value





s_lshl_b32 s0, s2, 5 // (op 12895) 
s_mul_i32 s1, 6, s0 // (op 12895) 
s_add_u32 s1, s1, 64 // (op 12895) 

v_add_u32 v0, v8, s1 // (op 12895) 


v_accvgpr_read v10, a88 // (op 12912) 
v_accvgpr_read v11, a89 // (op 12912) 
v_accvgpr_read v12, a90 // (op 12912) 
v_accvgpr_read v13, a91 // (op 12912) 
v_mul_f32 v14, v7, v10 // (op 12912) 
v_mul_f32 v15, v7, v11 // (op 12912) 
v_mul_f32 v16, v7, v12 // (op 12912) 
v_mul_f32 v17, v7, v13 // (op 12912) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12895) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12892) 
v_lshrrev_b32 v13, 16, v10 // (op 12892) 
v_lshlrev_b32 v12, 16, v12 // (op 12892) 
v_lshlrev_b32 v13, 16, v13 // (op 12892) 
v_mul_f32 v18, v5, v12 // (op 12892) 
v_mul_f32 v19, v5, v13 // (op 12892) 
v_and_b32 v12, 65535, v11 // (op 12892) 
v_lshrrev_b32 v13, 16, v11 // (op 12892) 
v_lshlrev_b32 v12, 16, v12 // (op 12892) 
v_lshlrev_b32 v13, 16, v13 // (op 12892) 
v_mul_f32 v20, v5, v12 // (op 12892) 
v_mul_f32 v21, v5, v13 // (op 12892) 


v_add_f32 v10, v18, v14 // (op 12893) 
v_add_f32 v11, v19, v15 // (op 12893) 
v_add_f32 v12, v20, v16 // (op 12893) 
v_add_f32 v13, v21, v17 // (op 12893) 


v_lshrrev_b32 v14, 16, v10 // (op 12894) 
v_lshrrev_b32 v15, 16, v11 // (op 12894) 
v_lshrrev_b32 v16, 16, v12 // (op 12894) 
v_lshrrev_b32 v17, 16, v13 // (op 12894) 


v_pack_b32_f16 v10, v14, v15 // (op 12860) 
v_pack_b32_f16 v11, v16, v17 // (op 12860) 

s_lshl_b32 s0, s4, 5 // (op 12860) 
s_mul_i32 s1, 6, s0 // (op 12860) 
s_add_u32 s1, s1, 64 // (op 12860) 

v_add_u32 v0, v9, s1 // (op 12860) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12860) Store value





s_lshl_b32 s0, s2, 5 // (op 12900) 
s_mul_i32 s1, 7, s0 // (op 12900) 
s_add_u32 s1, s1, 64 // (op 12900) 

v_add_u32 v0, v8, s1 // (op 12900) 


v_accvgpr_read v10, a92 // (op 12914) 
v_accvgpr_read v11, a93 // (op 12914) 
v_accvgpr_read v12, a94 // (op 12914) 
v_accvgpr_read v13, a95 // (op 12914) 
v_mul_f32 v14, v7, v10 // (op 12914) 
v_mul_f32 v15, v7, v11 // (op 12914) 
v_mul_f32 v16, v7, v12 // (op 12914) 
v_mul_f32 v17, v7, v13 // (op 12914) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12900) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12897) 
v_lshrrev_b32 v13, 16, v10 // (op 12897) 
v_lshlrev_b32 v12, 16, v12 // (op 12897) 
v_lshlrev_b32 v13, 16, v13 // (op 12897) 
v_mul_f32 v18, v5, v12 // (op 12897) 
v_mul_f32 v19, v5, v13 // (op 12897) 
v_and_b32 v12, 65535, v11 // (op 12897) 
v_lshrrev_b32 v13, 16, v11 // (op 12897) 
v_lshlrev_b32 v12, 16, v12 // (op 12897) 
v_lshlrev_b32 v13, 16, v13 // (op 12897) 
v_mul_f32 v20, v5, v12 // (op 12897) 
v_mul_f32 v21, v5, v13 // (op 12897) 


v_add_f32 v10, v18, v14 // (op 12898) 
v_add_f32 v11, v19, v15 // (op 12898) 
v_add_f32 v12, v20, v16 // (op 12898) 
v_add_f32 v13, v21, v17 // (op 12898) 


v_lshrrev_b32 v14, 16, v10 // (op 12899) 
v_lshrrev_b32 v15, 16, v11 // (op 12899) 
v_lshrrev_b32 v16, 16, v12 // (op 12899) 
v_lshrrev_b32 v17, 16, v13 // (op 12899) 


v_pack_b32_f16 v10, v14, v15 // (op 12862) 
v_pack_b32_f16 v11, v16, v17 // (op 12862) 

s_lshl_b32 s0, s4, 5 // (op 12862) 
s_mul_i32 s1, 7, s0 // (op 12862) 
s_add_u32 s1, s1, 64 // (op 12862) 

v_add_u32 v0, v9, s1 // (op 12862) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12862) Store value





v_add_u32 v0, 32, v8 // (op 12694) 

v_accvgpr_read v10, a32 // (op 12673) 
v_accvgpr_read v11, a33 // (op 12673) 
v_accvgpr_read v12, a34 // (op 12673) 
v_accvgpr_read v13, a35 // (op 12673) 
v_mul_f32 v14, v7, v10 // (op 12673) 
v_mul_f32 v15, v7, v11 // (op 12673) 
v_mul_f32 v16, v7, v12 // (op 12673) 
v_mul_f32 v17, v7, v13 // (op 12673) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12694) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12693) 
v_lshrrev_b32 v13, 16, v10 // (op 12693) 
v_lshlrev_b32 v12, 16, v12 // (op 12693) 
v_lshlrev_b32 v13, 16, v13 // (op 12693) 
v_mul_f32 v18, v5, v12 // (op 12693) 
v_mul_f32 v19, v5, v13 // (op 12693) 
v_and_b32 v12, 65535, v11 // (op 12693) 
v_lshrrev_b32 v13, 16, v11 // (op 12693) 
v_lshlrev_b32 v12, 16, v12 // (op 12693) 
v_lshlrev_b32 v13, 16, v13 // (op 12693) 
v_mul_f32 v20, v5, v12 // (op 12693) 
v_mul_f32 v21, v5, v13 // (op 12693) 


v_add_f32 v10, v18, v14 // (op 12674) 
v_add_f32 v11, v19, v15 // (op 12674) 
v_add_f32 v12, v20, v16 // (op 12674) 
v_add_f32 v13, v21, v17 // (op 12674) 


v_lshrrev_b32 v14, 16, v10 // (op 12675) 
v_lshrrev_b32 v15, 16, v11 // (op 12675) 
v_lshrrev_b32 v16, 16, v12 // (op 12675) 
v_lshrrev_b32 v17, 16, v13 // (op 12675) 


v_pack_b32_f16 v10, v14, v15 // (op 12677) 
v_pack_b32_f16 v11, v16, v17 // (op 12677) 

v_add_u32 v0, 32, v9 // (op 12677) 

buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12677) Store value





s_lshl_b32 s0, s2, 5 // (op 12699) 
s_add_u32 s0, s0, 32 // (op 12699) 
v_add_u32 v0, v8, s0 // (op 12699) 


v_accvgpr_read v10, a36 // (op 12731) 
v_accvgpr_read v11, a37 // (op 12731) 
v_accvgpr_read v12, a38 // (op 12731) 
v_accvgpr_read v13, a39 // (op 12731) 
v_mul_f32 v14, v7, v10 // (op 12731) 
v_mul_f32 v15, v7, v11 // (op 12731) 
v_mul_f32 v16, v7, v12 // (op 12731) 
v_mul_f32 v17, v7, v13 // (op 12731) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12699) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12696) 
v_lshrrev_b32 v13, 16, v10 // (op 12696) 
v_lshlrev_b32 v12, 16, v12 // (op 12696) 
v_lshlrev_b32 v13, 16, v13 // (op 12696) 
v_mul_f32 v18, v5, v12 // (op 12696) 
v_mul_f32 v19, v5, v13 // (op 12696) 
v_and_b32 v12, 65535, v11 // (op 12696) 
v_lshrrev_b32 v13, 16, v11 // (op 12696) 
v_lshlrev_b32 v12, 16, v12 // (op 12696) 
v_lshlrev_b32 v13, 16, v13 // (op 12696) 
v_mul_f32 v20, v5, v12 // (op 12696) 
v_mul_f32 v21, v5, v13 // (op 12696) 


v_add_f32 v10, v18, v14 // (op 12697) 
v_add_f32 v11, v19, v15 // (op 12697) 
v_add_f32 v12, v20, v16 // (op 12697) 
v_add_f32 v13, v21, v17 // (op 12697) 


v_lshrrev_b32 v14, 16, v10 // (op 12698) 
v_lshrrev_b32 v15, 16, v11 // (op 12698) 
v_lshrrev_b32 v16, 16, v12 // (op 12698) 
v_lshrrev_b32 v17, 16, v13 // (op 12698) 


v_pack_b32_f16 v10, v14, v15 // (op 12679) 
v_pack_b32_f16 v11, v16, v17 // (op 12679) 

s_lshl_b32 s0, s4, 5 // (op 12679) 
s_add_u32 s0, s0, 32 // (op 12679) 
v_add_u32 v0, v9, s0 // (op 12679) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12679) Store value





s_lshl_b32 s0, s2, 6 // (op 12704) 
s_add_u32 s0, s0, 32 // (op 12704) 
v_add_u32 v0, v8, s0 // (op 12704) 


v_accvgpr_read v10, a40 // (op 12733) 
v_accvgpr_read v11, a41 // (op 12733) 
v_accvgpr_read v12, a42 // (op 12733) 
v_accvgpr_read v13, a43 // (op 12733) 
v_mul_f32 v14, v7, v10 // (op 12733) 
v_mul_f32 v15, v7, v11 // (op 12733) 
v_mul_f32 v16, v7, v12 // (op 12733) 
v_mul_f32 v17, v7, v13 // (op 12733) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12704) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12701) 
v_lshrrev_b32 v13, 16, v10 // (op 12701) 
v_lshlrev_b32 v12, 16, v12 // (op 12701) 
v_lshlrev_b32 v13, 16, v13 // (op 12701) 
v_mul_f32 v18, v5, v12 // (op 12701) 
v_mul_f32 v19, v5, v13 // (op 12701) 
v_and_b32 v12, 65535, v11 // (op 12701) 
v_lshrrev_b32 v13, 16, v11 // (op 12701) 
v_lshlrev_b32 v12, 16, v12 // (op 12701) 
v_lshlrev_b32 v13, 16, v13 // (op 12701) 
v_mul_f32 v20, v5, v12 // (op 12701) 
v_mul_f32 v21, v5, v13 // (op 12701) 


v_add_f32 v10, v18, v14 // (op 12702) 
v_add_f32 v11, v19, v15 // (op 12702) 
v_add_f32 v12, v20, v16 // (op 12702) 
v_add_f32 v13, v21, v17 // (op 12702) 


v_lshrrev_b32 v14, 16, v10 // (op 12703) 
v_lshrrev_b32 v15, 16, v11 // (op 12703) 
v_lshrrev_b32 v16, 16, v12 // (op 12703) 
v_lshrrev_b32 v17, 16, v13 // (op 12703) 


v_pack_b32_f16 v10, v14, v15 // (op 12681) 
v_pack_b32_f16 v11, v16, v17 // (op 12681) 

s_lshl_b32 s0, s4, 6 // (op 12681) 
s_add_u32 s0, s0, 32 // (op 12681) 
v_add_u32 v0, v9, s0 // (op 12681) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12681) Store value





s_lshl_b32 s0, s2, 5 // (op 12709) 
s_mul_i32 s1, 3, s0 // (op 12709) 
s_add_u32 s1, s1, 32 // (op 12709) 

v_add_u32 v0, v8, s1 // (op 12709) 


v_accvgpr_read v10, a44 // (op 12735) 
v_accvgpr_read v11, a45 // (op 12735) 
v_accvgpr_read v12, a46 // (op 12735) 
v_accvgpr_read v13, a47 // (op 12735) 
v_mul_f32 v14, v7, v10 // (op 12735) 
v_mul_f32 v15, v7, v11 // (op 12735) 
v_mul_f32 v16, v7, v12 // (op 12735) 
v_mul_f32 v17, v7, v13 // (op 12735) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12709) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12706) 
v_lshrrev_b32 v13, 16, v10 // (op 12706) 
v_lshlrev_b32 v12, 16, v12 // (op 12706) 
v_lshlrev_b32 v13, 16, v13 // (op 12706) 
v_mul_f32 v18, v5, v12 // (op 12706) 
v_mul_f32 v19, v5, v13 // (op 12706) 
v_and_b32 v12, 65535, v11 // (op 12706) 
v_lshrrev_b32 v13, 16, v11 // (op 12706) 
v_lshlrev_b32 v12, 16, v12 // (op 12706) 
v_lshlrev_b32 v13, 16, v13 // (op 12706) 
v_mul_f32 v20, v5, v12 // (op 12706) 
v_mul_f32 v21, v5, v13 // (op 12706) 


v_add_f32 v10, v18, v14 // (op 12707) 
v_add_f32 v11, v19, v15 // (op 12707) 
v_add_f32 v12, v20, v16 // (op 12707) 
v_add_f32 v13, v21, v17 // (op 12707) 


v_lshrrev_b32 v14, 16, v10 // (op 12708) 
v_lshrrev_b32 v15, 16, v11 // (op 12708) 
v_lshrrev_b32 v16, 16, v12 // (op 12708) 
v_lshrrev_b32 v17, 16, v13 // (op 12708) 


v_pack_b32_f16 v10, v14, v15 // (op 12683) 
v_pack_b32_f16 v11, v16, v17 // (op 12683) 

s_lshl_b32 s0, s4, 5 // (op 12683) 
s_mul_i32 s1, 3, s0 // (op 12683) 
s_add_u32 s1, s1, 32 // (op 12683) 

v_add_u32 v0, v9, s1 // (op 12683) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12683) Store value





s_lshl_b32 s0, s2, 7 // (op 12714) 
s_add_u32 s0, s0, 32 // (op 12714) 
v_add_u32 v0, v8, s0 // (op 12714) 


v_accvgpr_read v10, a48 // (op 12737) 
v_accvgpr_read v11, a49 // (op 12737) 
v_accvgpr_read v12, a50 // (op 12737) 
v_accvgpr_read v13, a51 // (op 12737) 
v_mul_f32 v14, v7, v10 // (op 12737) 
v_mul_f32 v15, v7, v11 // (op 12737) 
v_mul_f32 v16, v7, v12 // (op 12737) 
v_mul_f32 v17, v7, v13 // (op 12737) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12714) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12711) 
v_lshrrev_b32 v13, 16, v10 // (op 12711) 
v_lshlrev_b32 v12, 16, v12 // (op 12711) 
v_lshlrev_b32 v13, 16, v13 // (op 12711) 
v_mul_f32 v18, v5, v12 // (op 12711) 
v_mul_f32 v19, v5, v13 // (op 12711) 
v_and_b32 v12, 65535, v11 // (op 12711) 
v_lshrrev_b32 v13, 16, v11 // (op 12711) 
v_lshlrev_b32 v12, 16, v12 // (op 12711) 
v_lshlrev_b32 v13, 16, v13 // (op 12711) 
v_mul_f32 v20, v5, v12 // (op 12711) 
v_mul_f32 v21, v5, v13 // (op 12711) 


v_add_f32 v10, v18, v14 // (op 12712) 
v_add_f32 v11, v19, v15 // (op 12712) 
v_add_f32 v12, v20, v16 // (op 12712) 
v_add_f32 v13, v21, v17 // (op 12712) 


v_lshrrev_b32 v14, 16, v10 // (op 12713) 
v_lshrrev_b32 v15, 16, v11 // (op 12713) 
v_lshrrev_b32 v16, 16, v12 // (op 12713) 
v_lshrrev_b32 v17, 16, v13 // (op 12713) 


v_pack_b32_f16 v10, v14, v15 // (op 12685) 
v_pack_b32_f16 v11, v16, v17 // (op 12685) 

s_lshl_b32 s0, s4, 7 // (op 12685) 
s_add_u32 s0, s0, 32 // (op 12685) 
v_add_u32 v0, v9, s0 // (op 12685) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12685) Store value





s_lshl_b32 s0, s2, 5 // (op 12719) 
s_mul_i32 s1, 5, s0 // (op 12719) 
s_add_u32 s1, s1, 32 // (op 12719) 

v_add_u32 v0, v8, s1 // (op 12719) 


v_accvgpr_read v10, a52 // (op 12739) 
v_accvgpr_read v11, a53 // (op 12739) 
v_accvgpr_read v12, a54 // (op 12739) 
v_accvgpr_read v13, a55 // (op 12739) 
v_mul_f32 v14, v7, v10 // (op 12739) 
v_mul_f32 v15, v7, v11 // (op 12739) 
v_mul_f32 v16, v7, v12 // (op 12739) 
v_mul_f32 v17, v7, v13 // (op 12739) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12719) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12716) 
v_lshrrev_b32 v13, 16, v10 // (op 12716) 
v_lshlrev_b32 v12, 16, v12 // (op 12716) 
v_lshlrev_b32 v13, 16, v13 // (op 12716) 
v_mul_f32 v18, v5, v12 // (op 12716) 
v_mul_f32 v19, v5, v13 // (op 12716) 
v_and_b32 v12, 65535, v11 // (op 12716) 
v_lshrrev_b32 v13, 16, v11 // (op 12716) 
v_lshlrev_b32 v12, 16, v12 // (op 12716) 
v_lshlrev_b32 v13, 16, v13 // (op 12716) 
v_mul_f32 v20, v5, v12 // (op 12716) 
v_mul_f32 v21, v5, v13 // (op 12716) 


v_add_f32 v10, v18, v14 // (op 12717) 
v_add_f32 v11, v19, v15 // (op 12717) 
v_add_f32 v12, v20, v16 // (op 12717) 
v_add_f32 v13, v21, v17 // (op 12717) 


v_lshrrev_b32 v14, 16, v10 // (op 12718) 
v_lshrrev_b32 v15, 16, v11 // (op 12718) 
v_lshrrev_b32 v16, 16, v12 // (op 12718) 
v_lshrrev_b32 v17, 16, v13 // (op 12718) 


v_pack_b32_f16 v10, v14, v15 // (op 12687) 
v_pack_b32_f16 v11, v16, v17 // (op 12687) 

s_lshl_b32 s0, s4, 5 // (op 12687) 
s_mul_i32 s1, 5, s0 // (op 12687) 
s_add_u32 s1, s1, 32 // (op 12687) 

v_add_u32 v0, v9, s1 // (op 12687) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12687) Store value





s_lshl_b32 s0, s2, 5 // (op 12724) 
s_mul_i32 s1, 6, s0 // (op 12724) 
s_add_u32 s1, s1, 32 // (op 12724) 

v_add_u32 v0, v8, s1 // (op 12724) 


v_accvgpr_read v10, a56 // (op 12741) 
v_accvgpr_read v11, a57 // (op 12741) 
v_accvgpr_read v12, a58 // (op 12741) 
v_accvgpr_read v13, a59 // (op 12741) 
v_mul_f32 v14, v7, v10 // (op 12741) 
v_mul_f32 v15, v7, v11 // (op 12741) 
v_mul_f32 v16, v7, v12 // (op 12741) 
v_mul_f32 v17, v7, v13 // (op 12741) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12724) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12721) 
v_lshrrev_b32 v13, 16, v10 // (op 12721) 
v_lshlrev_b32 v12, 16, v12 // (op 12721) 
v_lshlrev_b32 v13, 16, v13 // (op 12721) 
v_mul_f32 v18, v5, v12 // (op 12721) 
v_mul_f32 v19, v5, v13 // (op 12721) 
v_and_b32 v12, 65535, v11 // (op 12721) 
v_lshrrev_b32 v13, 16, v11 // (op 12721) 
v_lshlrev_b32 v12, 16, v12 // (op 12721) 
v_lshlrev_b32 v13, 16, v13 // (op 12721) 
v_mul_f32 v20, v5, v12 // (op 12721) 
v_mul_f32 v21, v5, v13 // (op 12721) 


v_add_f32 v10, v18, v14 // (op 12722) 
v_add_f32 v11, v19, v15 // (op 12722) 
v_add_f32 v12, v20, v16 // (op 12722) 
v_add_f32 v13, v21, v17 // (op 12722) 


v_lshrrev_b32 v14, 16, v10 // (op 12723) 
v_lshrrev_b32 v15, 16, v11 // (op 12723) 
v_lshrrev_b32 v16, 16, v12 // (op 12723) 
v_lshrrev_b32 v17, 16, v13 // (op 12723) 


v_pack_b32_f16 v10, v14, v15 // (op 12689) 
v_pack_b32_f16 v11, v16, v17 // (op 12689) 

s_lshl_b32 s0, s4, 5 // (op 12689) 
s_mul_i32 s1, 6, s0 // (op 12689) 
s_add_u32 s1, s1, 32 // (op 12689) 

v_add_u32 v0, v9, s1 // (op 12689) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12689) Store value





s_lshl_b32 s0, s2, 5 // (op 12729) 
s_mul_i32 s1, 7, s0 // (op 12729) 
s_add_u32 s1, s1, 32 // (op 12729) 

v_add_u32 v0, v8, s1 // (op 12729) 


v_accvgpr_read v10, a60 // (op 12743) 
v_accvgpr_read v11, a61 // (op 12743) 
v_accvgpr_read v12, a62 // (op 12743) 
v_accvgpr_read v13, a63 // (op 12743) 
v_mul_f32 v14, v7, v10 // (op 12743) 
v_mul_f32 v15, v7, v11 // (op 12743) 
v_mul_f32 v16, v7, v12 // (op 12743) 
v_mul_f32 v17, v7, v13 // (op 12743) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12729) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12726) 
v_lshrrev_b32 v13, 16, v10 // (op 12726) 
v_lshlrev_b32 v12, 16, v12 // (op 12726) 
v_lshlrev_b32 v13, 16, v13 // (op 12726) 
v_mul_f32 v18, v5, v12 // (op 12726) 
v_mul_f32 v19, v5, v13 // (op 12726) 
v_and_b32 v12, 65535, v11 // (op 12726) 
v_lshrrev_b32 v13, 16, v11 // (op 12726) 
v_lshlrev_b32 v12, 16, v12 // (op 12726) 
v_lshlrev_b32 v13, 16, v13 // (op 12726) 
v_mul_f32 v20, v5, v12 // (op 12726) 
v_mul_f32 v21, v5, v13 // (op 12726) 


v_add_f32 v10, v18, v14 // (op 12727) 
v_add_f32 v11, v19, v15 // (op 12727) 
v_add_f32 v12, v20, v16 // (op 12727) 
v_add_f32 v13, v21, v17 // (op 12727) 


v_lshrrev_b32 v14, 16, v10 // (op 12728) 
v_lshrrev_b32 v15, 16, v11 // (op 12728) 
v_lshrrev_b32 v16, 16, v12 // (op 12728) 
v_lshrrev_b32 v17, 16, v13 // (op 12728) 


v_pack_b32_f16 v10, v14, v15 // (op 12691) 
v_pack_b32_f16 v11, v16, v17 // (op 12691) 

s_lshl_b32 s0, s4, 5 // (op 12691) 
s_mul_i32 s1, 7, s0 // (op 12691) 
s_add_u32 s1, s1, 32 // (op 12691) 

v_add_u32 v0, v9, s1 // (op 12691) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12691) Store value





v_add_u32 v0, 192, v8 // (op 13549) 

v_accvgpr_read v10, a192 // (op 13528) 
v_accvgpr_read v11, a193 // (op 13528) 
v_accvgpr_read v12, a194 // (op 13528) 
v_accvgpr_read v13, a195 // (op 13528) 
v_mul_f32 v14, v7, v10 // (op 13528) 
v_mul_f32 v15, v7, v11 // (op 13528) 
v_mul_f32 v16, v7, v12 // (op 13528) 
v_mul_f32 v17, v7, v13 // (op 13528) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13549) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13548) 
v_lshrrev_b32 v13, 16, v10 // (op 13548) 
v_lshlrev_b32 v12, 16, v12 // (op 13548) 
v_lshlrev_b32 v13, 16, v13 // (op 13548) 
v_mul_f32 v18, v5, v12 // (op 13548) 
v_mul_f32 v19, v5, v13 // (op 13548) 
v_and_b32 v12, 65535, v11 // (op 13548) 
v_lshrrev_b32 v13, 16, v11 // (op 13548) 
v_lshlrev_b32 v12, 16, v12 // (op 13548) 
v_lshlrev_b32 v13, 16, v13 // (op 13548) 
v_mul_f32 v20, v5, v12 // (op 13548) 
v_mul_f32 v21, v5, v13 // (op 13548) 


v_add_f32 v10, v18, v14 // (op 13529) 
v_add_f32 v11, v19, v15 // (op 13529) 
v_add_f32 v12, v20, v16 // (op 13529) 
v_add_f32 v13, v21, v17 // (op 13529) 


v_lshrrev_b32 v14, 16, v10 // (op 13530) 
v_lshrrev_b32 v15, 16, v11 // (op 13530) 
v_lshrrev_b32 v16, 16, v12 // (op 13530) 
v_lshrrev_b32 v17, 16, v13 // (op 13530) 


v_pack_b32_f16 v10, v14, v15 // (op 13532) 
v_pack_b32_f16 v11, v16, v17 // (op 13532) 

v_add_u32 v0, 192, v9 // (op 13532) 

buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13532) Store value





s_lshl_b32 s0, s2, 5 // (op 13554) 
s_add_u32 s0, s0, 192 // (op 13554) 
v_add_u32 v0, v8, s0 // (op 13554) 


v_accvgpr_read v10, a196 // (op 13586) 
v_accvgpr_read v11, a197 // (op 13586) 
v_accvgpr_read v12, a198 // (op 13586) 
v_accvgpr_read v13, a199 // (op 13586) 
v_mul_f32 v14, v7, v10 // (op 13586) 
v_mul_f32 v15, v7, v11 // (op 13586) 
v_mul_f32 v16, v7, v12 // (op 13586) 
v_mul_f32 v17, v7, v13 // (op 13586) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13554) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13551) 
v_lshrrev_b32 v13, 16, v10 // (op 13551) 
v_lshlrev_b32 v12, 16, v12 // (op 13551) 
v_lshlrev_b32 v13, 16, v13 // (op 13551) 
v_mul_f32 v18, v5, v12 // (op 13551) 
v_mul_f32 v19, v5, v13 // (op 13551) 
v_and_b32 v12, 65535, v11 // (op 13551) 
v_lshrrev_b32 v13, 16, v11 // (op 13551) 
v_lshlrev_b32 v12, 16, v12 // (op 13551) 
v_lshlrev_b32 v13, 16, v13 // (op 13551) 
v_mul_f32 v20, v5, v12 // (op 13551) 
v_mul_f32 v21, v5, v13 // (op 13551) 


v_add_f32 v10, v18, v14 // (op 13552) 
v_add_f32 v11, v19, v15 // (op 13552) 
v_add_f32 v12, v20, v16 // (op 13552) 
v_add_f32 v13, v21, v17 // (op 13552) 


v_lshrrev_b32 v14, 16, v10 // (op 13553) 
v_lshrrev_b32 v15, 16, v11 // (op 13553) 
v_lshrrev_b32 v16, 16, v12 // (op 13553) 
v_lshrrev_b32 v17, 16, v13 // (op 13553) 


v_pack_b32_f16 v10, v14, v15 // (op 13534) 
v_pack_b32_f16 v11, v16, v17 // (op 13534) 

s_lshl_b32 s0, s4, 5 // (op 13534) 
s_add_u32 s0, s0, 192 // (op 13534) 
v_add_u32 v0, v9, s0 // (op 13534) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13534) Store value





s_lshl_b32 s0, s2, 6 // (op 13559) 
s_add_u32 s0, s0, 192 // (op 13559) 
v_add_u32 v0, v8, s0 // (op 13559) 


v_accvgpr_read v10, a200 // (op 13588) 
v_accvgpr_read v11, a201 // (op 13588) 
v_accvgpr_read v12, a202 // (op 13588) 
v_accvgpr_read v13, a203 // (op 13588) 
v_mul_f32 v14, v7, v10 // (op 13588) 
v_mul_f32 v15, v7, v11 // (op 13588) 
v_mul_f32 v16, v7, v12 // (op 13588) 
v_mul_f32 v17, v7, v13 // (op 13588) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13559) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13556) 
v_lshrrev_b32 v13, 16, v10 // (op 13556) 
v_lshlrev_b32 v12, 16, v12 // (op 13556) 
v_lshlrev_b32 v13, 16, v13 // (op 13556) 
v_mul_f32 v18, v5, v12 // (op 13556) 
v_mul_f32 v19, v5, v13 // (op 13556) 
v_and_b32 v12, 65535, v11 // (op 13556) 
v_lshrrev_b32 v13, 16, v11 // (op 13556) 
v_lshlrev_b32 v12, 16, v12 // (op 13556) 
v_lshlrev_b32 v13, 16, v13 // (op 13556) 
v_mul_f32 v20, v5, v12 // (op 13556) 
v_mul_f32 v21, v5, v13 // (op 13556) 


v_add_f32 v10, v18, v14 // (op 13557) 
v_add_f32 v11, v19, v15 // (op 13557) 
v_add_f32 v12, v20, v16 // (op 13557) 
v_add_f32 v13, v21, v17 // (op 13557) 


v_lshrrev_b32 v14, 16, v10 // (op 13558) 
v_lshrrev_b32 v15, 16, v11 // (op 13558) 
v_lshrrev_b32 v16, 16, v12 // (op 13558) 
v_lshrrev_b32 v17, 16, v13 // (op 13558) 


v_pack_b32_f16 v10, v14, v15 // (op 13536) 
v_pack_b32_f16 v11, v16, v17 // (op 13536) 

s_lshl_b32 s0, s4, 6 // (op 13536) 
s_add_u32 s0, s0, 192 // (op 13536) 
v_add_u32 v0, v9, s0 // (op 13536) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13536) Store value





s_lshl_b32 s0, s2, 5 // (op 13564) 
s_mul_i32 s1, 3, s0 // (op 13564) 
s_add_u32 s1, s1, 192 // (op 13564) 

v_add_u32 v0, v8, s1 // (op 13564) 


v_accvgpr_read v10, a204 // (op 13590) 
v_accvgpr_read v11, a205 // (op 13590) 
v_accvgpr_read v12, a206 // (op 13590) 
v_accvgpr_read v13, a207 // (op 13590) 
v_mul_f32 v14, v7, v10 // (op 13590) 
v_mul_f32 v15, v7, v11 // (op 13590) 
v_mul_f32 v16, v7, v12 // (op 13590) 
v_mul_f32 v17, v7, v13 // (op 13590) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13564) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13561) 
v_lshrrev_b32 v13, 16, v10 // (op 13561) 
v_lshlrev_b32 v12, 16, v12 // (op 13561) 
v_lshlrev_b32 v13, 16, v13 // (op 13561) 
v_mul_f32 v18, v5, v12 // (op 13561) 
v_mul_f32 v19, v5, v13 // (op 13561) 
v_and_b32 v12, 65535, v11 // (op 13561) 
v_lshrrev_b32 v13, 16, v11 // (op 13561) 
v_lshlrev_b32 v12, 16, v12 // (op 13561) 
v_lshlrev_b32 v13, 16, v13 // (op 13561) 
v_mul_f32 v20, v5, v12 // (op 13561) 
v_mul_f32 v21, v5, v13 // (op 13561) 


v_add_f32 v10, v18, v14 // (op 13562) 
v_add_f32 v11, v19, v15 // (op 13562) 
v_add_f32 v12, v20, v16 // (op 13562) 
v_add_f32 v13, v21, v17 // (op 13562) 


v_lshrrev_b32 v14, 16, v10 // (op 13563) 
v_lshrrev_b32 v15, 16, v11 // (op 13563) 
v_lshrrev_b32 v16, 16, v12 // (op 13563) 
v_lshrrev_b32 v17, 16, v13 // (op 13563) 


v_pack_b32_f16 v10, v14, v15 // (op 13538) 
v_pack_b32_f16 v11, v16, v17 // (op 13538) 

s_lshl_b32 s0, s4, 5 // (op 13538) 
s_mul_i32 s1, 3, s0 // (op 13538) 
s_add_u32 s1, s1, 192 // (op 13538) 

v_add_u32 v0, v9, s1 // (op 13538) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13538) Store value





s_lshl_b32 s0, s2, 7 // (op 13569) 
s_add_u32 s0, s0, 192 // (op 13569) 
v_add_u32 v0, v8, s0 // (op 13569) 


v_accvgpr_read v10, a208 // (op 13592) 
v_accvgpr_read v11, a209 // (op 13592) 
v_accvgpr_read v12, a210 // (op 13592) 
v_accvgpr_read v13, a211 // (op 13592) 
v_mul_f32 v14, v7, v10 // (op 13592) 
v_mul_f32 v15, v7, v11 // (op 13592) 
v_mul_f32 v16, v7, v12 // (op 13592) 
v_mul_f32 v17, v7, v13 // (op 13592) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13569) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13566) 
v_lshrrev_b32 v13, 16, v10 // (op 13566) 
v_lshlrev_b32 v12, 16, v12 // (op 13566) 
v_lshlrev_b32 v13, 16, v13 // (op 13566) 
v_mul_f32 v18, v5, v12 // (op 13566) 
v_mul_f32 v19, v5, v13 // (op 13566) 
v_and_b32 v12, 65535, v11 // (op 13566) 
v_lshrrev_b32 v13, 16, v11 // (op 13566) 
v_lshlrev_b32 v12, 16, v12 // (op 13566) 
v_lshlrev_b32 v13, 16, v13 // (op 13566) 
v_mul_f32 v20, v5, v12 // (op 13566) 
v_mul_f32 v21, v5, v13 // (op 13566) 


v_add_f32 v10, v18, v14 // (op 13567) 
v_add_f32 v11, v19, v15 // (op 13567) 
v_add_f32 v12, v20, v16 // (op 13567) 
v_add_f32 v13, v21, v17 // (op 13567) 


v_lshrrev_b32 v14, 16, v10 // (op 13568) 
v_lshrrev_b32 v15, 16, v11 // (op 13568) 
v_lshrrev_b32 v16, 16, v12 // (op 13568) 
v_lshrrev_b32 v17, 16, v13 // (op 13568) 


v_pack_b32_f16 v10, v14, v15 // (op 13540) 
v_pack_b32_f16 v11, v16, v17 // (op 13540) 

s_lshl_b32 s0, s4, 7 // (op 13540) 
s_add_u32 s0, s0, 192 // (op 13540) 
v_add_u32 v0, v9, s0 // (op 13540) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13540) Store value





s_lshl_b32 s0, s2, 5 // (op 13574) 
s_mul_i32 s1, 5, s0 // (op 13574) 
s_add_u32 s1, s1, 192 // (op 13574) 

v_add_u32 v0, v8, s1 // (op 13574) 


v_accvgpr_read v10, a212 // (op 13594) 
v_accvgpr_read v11, a213 // (op 13594) 
v_accvgpr_read v12, a214 // (op 13594) 
v_accvgpr_read v13, a215 // (op 13594) 
v_mul_f32 v14, v7, v10 // (op 13594) 
v_mul_f32 v15, v7, v11 // (op 13594) 
v_mul_f32 v16, v7, v12 // (op 13594) 
v_mul_f32 v17, v7, v13 // (op 13594) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13574) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13571) 
v_lshrrev_b32 v13, 16, v10 // (op 13571) 
v_lshlrev_b32 v12, 16, v12 // (op 13571) 
v_lshlrev_b32 v13, 16, v13 // (op 13571) 
v_mul_f32 v18, v5, v12 // (op 13571) 
v_mul_f32 v19, v5, v13 // (op 13571) 
v_and_b32 v12, 65535, v11 // (op 13571) 
v_lshrrev_b32 v13, 16, v11 // (op 13571) 
v_lshlrev_b32 v12, 16, v12 // (op 13571) 
v_lshlrev_b32 v13, 16, v13 // (op 13571) 
v_mul_f32 v20, v5, v12 // (op 13571) 
v_mul_f32 v21, v5, v13 // (op 13571) 


v_add_f32 v10, v18, v14 // (op 13572) 
v_add_f32 v11, v19, v15 // (op 13572) 
v_add_f32 v12, v20, v16 // (op 13572) 
v_add_f32 v13, v21, v17 // (op 13572) 


v_lshrrev_b32 v14, 16, v10 // (op 13573) 
v_lshrrev_b32 v15, 16, v11 // (op 13573) 
v_lshrrev_b32 v16, 16, v12 // (op 13573) 
v_lshrrev_b32 v17, 16, v13 // (op 13573) 


v_pack_b32_f16 v10, v14, v15 // (op 13542) 
v_pack_b32_f16 v11, v16, v17 // (op 13542) 

s_lshl_b32 s0, s4, 5 // (op 13542) 
s_mul_i32 s1, 5, s0 // (op 13542) 
s_add_u32 s1, s1, 192 // (op 13542) 

v_add_u32 v0, v9, s1 // (op 13542) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13542) Store value





s_lshl_b32 s0, s2, 5 // (op 13579) 
s_mul_i32 s1, 6, s0 // (op 13579) 
s_add_u32 s1, s1, 192 // (op 13579) 

v_add_u32 v0, v8, s1 // (op 13579) 


v_accvgpr_read v10, a216 // (op 13596) 
v_accvgpr_read v11, a217 // (op 13596) 
v_accvgpr_read v12, a218 // (op 13596) 
v_accvgpr_read v13, a219 // (op 13596) 
v_mul_f32 v14, v7, v10 // (op 13596) 
v_mul_f32 v15, v7, v11 // (op 13596) 
v_mul_f32 v16, v7, v12 // (op 13596) 
v_mul_f32 v17, v7, v13 // (op 13596) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13579) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13576) 
v_lshrrev_b32 v13, 16, v10 // (op 13576) 
v_lshlrev_b32 v12, 16, v12 // (op 13576) 
v_lshlrev_b32 v13, 16, v13 // (op 13576) 
v_mul_f32 v18, v5, v12 // (op 13576) 
v_mul_f32 v19, v5, v13 // (op 13576) 
v_and_b32 v12, 65535, v11 // (op 13576) 
v_lshrrev_b32 v13, 16, v11 // (op 13576) 
v_lshlrev_b32 v12, 16, v12 // (op 13576) 
v_lshlrev_b32 v13, 16, v13 // (op 13576) 
v_mul_f32 v20, v5, v12 // (op 13576) 
v_mul_f32 v21, v5, v13 // (op 13576) 


v_add_f32 v10, v18, v14 // (op 13577) 
v_add_f32 v11, v19, v15 // (op 13577) 
v_add_f32 v12, v20, v16 // (op 13577) 
v_add_f32 v13, v21, v17 // (op 13577) 


v_lshrrev_b32 v14, 16, v10 // (op 13578) 
v_lshrrev_b32 v15, 16, v11 // (op 13578) 
v_lshrrev_b32 v16, 16, v12 // (op 13578) 
v_lshrrev_b32 v17, 16, v13 // (op 13578) 


v_pack_b32_f16 v10, v14, v15 // (op 13544) 
v_pack_b32_f16 v11, v16, v17 // (op 13544) 

s_lshl_b32 s0, s4, 5 // (op 13544) 
s_mul_i32 s1, 6, s0 // (op 13544) 
s_add_u32 s1, s1, 192 // (op 13544) 

v_add_u32 v0, v9, s1 // (op 13544) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13544) Store value





s_lshl_b32 s0, s2, 5 // (op 13584) 
s_mul_i32 s1, 7, s0 // (op 13584) 
s_add_u32 s1, s1, 192 // (op 13584) 

v_add_u32 v0, v8, s1 // (op 13584) 


v_accvgpr_read v10, a220 // (op 13598) 
v_accvgpr_read v11, a221 // (op 13598) 
v_accvgpr_read v12, a222 // (op 13598) 
v_accvgpr_read v13, a223 // (op 13598) 
v_mul_f32 v14, v7, v10 // (op 13598) 
v_mul_f32 v15, v7, v11 // (op 13598) 
v_mul_f32 v16, v7, v12 // (op 13598) 
v_mul_f32 v17, v7, v13 // (op 13598) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 13584) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 13581) 
v_lshrrev_b32 v13, 16, v10 // (op 13581) 
v_lshlrev_b32 v12, 16, v12 // (op 13581) 
v_lshlrev_b32 v13, 16, v13 // (op 13581) 
v_mul_f32 v18, v5, v12 // (op 13581) 
v_mul_f32 v19, v5, v13 // (op 13581) 
v_and_b32 v12, 65535, v11 // (op 13581) 
v_lshrrev_b32 v13, 16, v11 // (op 13581) 
v_lshlrev_b32 v12, 16, v12 // (op 13581) 
v_lshlrev_b32 v13, 16, v13 // (op 13581) 
v_mul_f32 v20, v5, v12 // (op 13581) 
v_mul_f32 v21, v5, v13 // (op 13581) 


v_add_f32 v10, v18, v14 // (op 13582) 
v_add_f32 v11, v19, v15 // (op 13582) 
v_add_f32 v12, v20, v16 // (op 13582) 
v_add_f32 v13, v21, v17 // (op 13582) 


v_lshrrev_b32 v14, 16, v10 // (op 13583) 
v_lshrrev_b32 v15, 16, v11 // (op 13583) 
v_lshrrev_b32 v16, 16, v12 // (op 13583) 
v_lshrrev_b32 v17, 16, v13 // (op 13583) 


v_pack_b32_f16 v10, v14, v15 // (op 13546) 
v_pack_b32_f16 v11, v16, v17 // (op 13546) 

s_lshl_b32 s0, s4, 5 // (op 13546) 
s_mul_i32 s1, 7, s0 // (op 13546) 
s_add_u32 s1, s1, 192 // (op 13546) 

v_add_u32 v0, v9, s1 // (op 13546) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 13546) Store value





v_mov_b32 v0, v8 // (op 129) reg expression

v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 129) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 130) 
v_lshrrev_b32 v13, 16, v10 // (op 130) 
v_lshlrev_b32 v12, 16, v12 // (op 130) 
v_lshlrev_b32 v13, 16, v13 // (op 130) 
v_mul_f32 v18, v5, v12 // (op 130) 
v_mul_f32 v19, v5, v13 // (op 130) 
v_and_b32 v12, 65535, v11 // (op 130) 
v_lshrrev_b32 v13, 16, v11 // (op 130) 
v_lshlrev_b32 v12, 16, v12 // (op 130) 
v_lshlrev_b32 v13, 16, v13 // (op 130) 
v_mul_f32 v20, v5, v12 // (op 130) 
v_mul_f32 v21, v5, v13 // (op 130) 


v_add_f32 v10, v18, v14 // (op 126) 
v_add_f32 v11, v19, v15 // (op 126) 
v_add_f32 v12, v20, v16 // (op 126) 
v_add_f32 v13, v21, v17 // (op 126) 


v_lshrrev_b32 v14, 16, v10 // (op 127) 
v_lshrrev_b32 v15, 16, v11 // (op 127) 
v_lshrrev_b32 v16, 16, v12 // (op 127) 
v_lshrrev_b32 v17, 16, v13 // (op 127) 


v_pack_b32_f16 v10, v14, v15 // (op 128) 
v_pack_b32_f16 v11, v16, v17 // (op 128) 

v_mov_b32 v0, v9 // (op 128) reg expression

buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 12498) 
v_add_u32 v0, v8, s0 // (op 12498) 


v_accvgpr_read v10, a4 // (op 12494) 
v_accvgpr_read v11, a5 // (op 12494) 
v_accvgpr_read v12, a6 // (op 12494) 
v_accvgpr_read v13, a7 // (op 12494) 
v_mul_f32 v14, v7, v10 // (op 12494) 
v_mul_f32 v15, v7, v11 // (op 12494) 
v_mul_f32 v16, v7, v12 // (op 12494) 
v_mul_f32 v17, v7, v13 // (op 12494) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12498) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12499) 
v_lshrrev_b32 v13, 16, v10 // (op 12499) 
v_lshlrev_b32 v12, 16, v12 // (op 12499) 
v_lshlrev_b32 v13, 16, v13 // (op 12499) 
v_mul_f32 v18, v5, v12 // (op 12499) 
v_mul_f32 v19, v5, v13 // (op 12499) 
v_and_b32 v12, 65535, v11 // (op 12499) 
v_lshrrev_b32 v13, 16, v11 // (op 12499) 
v_lshlrev_b32 v12, 16, v12 // (op 12499) 
v_lshlrev_b32 v13, 16, v13 // (op 12499) 
v_mul_f32 v20, v5, v12 // (op 12499) 
v_mul_f32 v21, v5, v13 // (op 12499) 


v_add_f32 v10, v18, v14 // (op 12495) 
v_add_f32 v11, v19, v15 // (op 12495) 
v_add_f32 v12, v20, v16 // (op 12495) 
v_add_f32 v13, v21, v17 // (op 12495) 


v_lshrrev_b32 v14, 16, v10 // (op 12496) 
v_lshrrev_b32 v15, 16, v11 // (op 12496) 
v_lshrrev_b32 v16, 16, v12 // (op 12496) 
v_lshrrev_b32 v17, 16, v13 // (op 12496) 


v_pack_b32_f16 v10, v14, v15 // (op 12497) 
v_pack_b32_f16 v11, v16, v17 // (op 12497) 

s_lshl_b32 s0, s4, 5 // (op 12497) 
v_add_u32 v0, v9, s0 // (op 12497) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12497) Store value





s_lshl_b32 s0, s2, 6 // (op 12512) 
v_add_u32 v0, v8, s0 // (op 12512) 


v_accvgpr_read v10, a8 // (op 12508) 
v_accvgpr_read v11, a9 // (op 12508) 
v_accvgpr_read v12, a10 // (op 12508) 
v_accvgpr_read v13, a11 // (op 12508) 
v_mul_f32 v14, v7, v10 // (op 12508) 
v_mul_f32 v15, v7, v11 // (op 12508) 
v_mul_f32 v16, v7, v12 // (op 12508) 
v_mul_f32 v17, v7, v13 // (op 12508) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12512) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12513) 
v_lshrrev_b32 v13, 16, v10 // (op 12513) 
v_lshlrev_b32 v12, 16, v12 // (op 12513) 
v_lshlrev_b32 v13, 16, v13 // (op 12513) 
v_mul_f32 v18, v5, v12 // (op 12513) 
v_mul_f32 v19, v5, v13 // (op 12513) 
v_and_b32 v12, 65535, v11 // (op 12513) 
v_lshrrev_b32 v13, 16, v11 // (op 12513) 
v_lshlrev_b32 v12, 16, v12 // (op 12513) 
v_lshlrev_b32 v13, 16, v13 // (op 12513) 
v_mul_f32 v20, v5, v12 // (op 12513) 
v_mul_f32 v21, v5, v13 // (op 12513) 


v_add_f32 v10, v18, v14 // (op 12509) 
v_add_f32 v11, v19, v15 // (op 12509) 
v_add_f32 v12, v20, v16 // (op 12509) 
v_add_f32 v13, v21, v17 // (op 12509) 


v_lshrrev_b32 v14, 16, v10 // (op 12510) 
v_lshrrev_b32 v15, 16, v11 // (op 12510) 
v_lshrrev_b32 v16, 16, v12 // (op 12510) 
v_lshrrev_b32 v17, 16, v13 // (op 12510) 


v_pack_b32_f16 v10, v14, v15 // (op 12511) 
v_pack_b32_f16 v11, v16, v17 // (op 12511) 

s_lshl_b32 s0, s4, 6 // (op 12511) 
v_add_u32 v0, v9, s0 // (op 12511) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12511) Store value





s_lshl_b32 s0, s2, 5 // (op 12526) 
s_mul_i32 s1, 3, s0 // (op 12526) 

v_add_u32 v0, v8, s1 // (op 12526) 


v_accvgpr_read v10, a12 // (op 12522) 
v_accvgpr_read v11, a13 // (op 12522) 
v_accvgpr_read v12, a14 // (op 12522) 
v_accvgpr_read v13, a15 // (op 12522) 
v_mul_f32 v14, v7, v10 // (op 12522) 
v_mul_f32 v15, v7, v11 // (op 12522) 
v_mul_f32 v16, v7, v12 // (op 12522) 
v_mul_f32 v17, v7, v13 // (op 12522) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12526) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12527) 
v_lshrrev_b32 v13, 16, v10 // (op 12527) 
v_lshlrev_b32 v12, 16, v12 // (op 12527) 
v_lshlrev_b32 v13, 16, v13 // (op 12527) 
v_mul_f32 v18, v5, v12 // (op 12527) 
v_mul_f32 v19, v5, v13 // (op 12527) 
v_and_b32 v12, 65535, v11 // (op 12527) 
v_lshrrev_b32 v13, 16, v11 // (op 12527) 
v_lshlrev_b32 v12, 16, v12 // (op 12527) 
v_lshlrev_b32 v13, 16, v13 // (op 12527) 
v_mul_f32 v20, v5, v12 // (op 12527) 
v_mul_f32 v21, v5, v13 // (op 12527) 


v_add_f32 v10, v18, v14 // (op 12523) 
v_add_f32 v11, v19, v15 // (op 12523) 
v_add_f32 v12, v20, v16 // (op 12523) 
v_add_f32 v13, v21, v17 // (op 12523) 


v_lshrrev_b32 v14, 16, v10 // (op 12524) 
v_lshrrev_b32 v15, 16, v11 // (op 12524) 
v_lshrrev_b32 v16, 16, v12 // (op 12524) 
v_lshrrev_b32 v17, 16, v13 // (op 12524) 


v_pack_b32_f16 v10, v14, v15 // (op 12525) 
v_pack_b32_f16 v11, v16, v17 // (op 12525) 

s_lshl_b32 s0, s4, 5 // (op 12525) 
s_mul_i32 s1, 3, s0 // (op 12525) 

v_add_u32 v0, v9, s1 // (op 12525) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12525) Store value





s_lshl_b32 s0, s2, 7 // (op 12540) 
v_add_u32 v0, v8, s0 // (op 12540) 


v_accvgpr_read v10, a16 // (op 12536) 
v_accvgpr_read v11, a17 // (op 12536) 
v_accvgpr_read v12, a18 // (op 12536) 
v_accvgpr_read v13, a19 // (op 12536) 
v_mul_f32 v14, v7, v10 // (op 12536) 
v_mul_f32 v15, v7, v11 // (op 12536) 
v_mul_f32 v16, v7, v12 // (op 12536) 
v_mul_f32 v17, v7, v13 // (op 12536) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12540) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12541) 
v_lshrrev_b32 v13, 16, v10 // (op 12541) 
v_lshlrev_b32 v12, 16, v12 // (op 12541) 
v_lshlrev_b32 v13, 16, v13 // (op 12541) 
v_mul_f32 v18, v5, v12 // (op 12541) 
v_mul_f32 v19, v5, v13 // (op 12541) 
v_and_b32 v12, 65535, v11 // (op 12541) 
v_lshrrev_b32 v13, 16, v11 // (op 12541) 
v_lshlrev_b32 v12, 16, v12 // (op 12541) 
v_lshlrev_b32 v13, 16, v13 // (op 12541) 
v_mul_f32 v20, v5, v12 // (op 12541) 
v_mul_f32 v21, v5, v13 // (op 12541) 


v_add_f32 v10, v18, v14 // (op 12537) 
v_add_f32 v11, v19, v15 // (op 12537) 
v_add_f32 v12, v20, v16 // (op 12537) 
v_add_f32 v13, v21, v17 // (op 12537) 


v_lshrrev_b32 v14, 16, v10 // (op 12538) 
v_lshrrev_b32 v15, 16, v11 // (op 12538) 
v_lshrrev_b32 v16, 16, v12 // (op 12538) 
v_lshrrev_b32 v17, 16, v13 // (op 12538) 


v_pack_b32_f16 v10, v14, v15 // (op 12539) 
v_pack_b32_f16 v11, v16, v17 // (op 12539) 

s_lshl_b32 s0, s4, 7 // (op 12539) 
v_add_u32 v0, v9, s0 // (op 12539) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12539) Store value





s_lshl_b32 s0, s2, 5 // (op 12554) 
s_mul_i32 s1, 5, s0 // (op 12554) 

v_add_u32 v0, v8, s1 // (op 12554) 


v_accvgpr_read v10, a20 // (op 12550) 
v_accvgpr_read v11, a21 // (op 12550) 
v_accvgpr_read v12, a22 // (op 12550) 
v_accvgpr_read v13, a23 // (op 12550) 
v_mul_f32 v14, v7, v10 // (op 12550) 
v_mul_f32 v15, v7, v11 // (op 12550) 
v_mul_f32 v16, v7, v12 // (op 12550) 
v_mul_f32 v17, v7, v13 // (op 12550) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12554) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12555) 
v_lshrrev_b32 v13, 16, v10 // (op 12555) 
v_lshlrev_b32 v12, 16, v12 // (op 12555) 
v_lshlrev_b32 v13, 16, v13 // (op 12555) 
v_mul_f32 v18, v5, v12 // (op 12555) 
v_mul_f32 v19, v5, v13 // (op 12555) 
v_and_b32 v12, 65535, v11 // (op 12555) 
v_lshrrev_b32 v13, 16, v11 // (op 12555) 
v_lshlrev_b32 v12, 16, v12 // (op 12555) 
v_lshlrev_b32 v13, 16, v13 // (op 12555) 
v_mul_f32 v20, v5, v12 // (op 12555) 
v_mul_f32 v21, v5, v13 // (op 12555) 


v_add_f32 v10, v18, v14 // (op 12551) 
v_add_f32 v11, v19, v15 // (op 12551) 
v_add_f32 v12, v20, v16 // (op 12551) 
v_add_f32 v13, v21, v17 // (op 12551) 


v_lshrrev_b32 v14, 16, v10 // (op 12552) 
v_lshrrev_b32 v15, 16, v11 // (op 12552) 
v_lshrrev_b32 v16, 16, v12 // (op 12552) 
v_lshrrev_b32 v17, 16, v13 // (op 12552) 


v_pack_b32_f16 v10, v14, v15 // (op 12553) 
v_pack_b32_f16 v11, v16, v17 // (op 12553) 

s_lshl_b32 s0, s4, 5 // (op 12553) 
s_mul_i32 s1, 5, s0 // (op 12553) 

v_add_u32 v0, v9, s1 // (op 12553) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12553) Store value





s_lshl_b32 s0, s2, 5 // (op 12568) 
s_mul_i32 s1, 6, s0 // (op 12568) 

v_add_u32 v0, v8, s1 // (op 12568) 


v_accvgpr_read v10, a24 // (op 12564) 
v_accvgpr_read v11, a25 // (op 12564) 
v_accvgpr_read v12, a26 // (op 12564) 
v_accvgpr_read v13, a27 // (op 12564) 
v_mul_f32 v14, v7, v10 // (op 12564) 
v_mul_f32 v15, v7, v11 // (op 12564) 
v_mul_f32 v16, v7, v12 // (op 12564) 
v_mul_f32 v17, v7, v13 // (op 12564) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12568) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12569) 
v_lshrrev_b32 v13, 16, v10 // (op 12569) 
v_lshlrev_b32 v12, 16, v12 // (op 12569) 
v_lshlrev_b32 v13, 16, v13 // (op 12569) 
v_mul_f32 v18, v5, v12 // (op 12569) 
v_mul_f32 v19, v5, v13 // (op 12569) 
v_and_b32 v12, 65535, v11 // (op 12569) 
v_lshrrev_b32 v13, 16, v11 // (op 12569) 
v_lshlrev_b32 v12, 16, v12 // (op 12569) 
v_lshlrev_b32 v13, 16, v13 // (op 12569) 
v_mul_f32 v20, v5, v12 // (op 12569) 
v_mul_f32 v21, v5, v13 // (op 12569) 


v_add_f32 v10, v18, v14 // (op 12565) 
v_add_f32 v11, v19, v15 // (op 12565) 
v_add_f32 v12, v20, v16 // (op 12565) 
v_add_f32 v13, v21, v17 // (op 12565) 


v_lshrrev_b32 v14, 16, v10 // (op 12566) 
v_lshrrev_b32 v15, 16, v11 // (op 12566) 
v_lshrrev_b32 v16, 16, v12 // (op 12566) 
v_lshrrev_b32 v17, 16, v13 // (op 12566) 


v_pack_b32_f16 v10, v14, v15 // (op 12567) 
v_pack_b32_f16 v11, v16, v17 // (op 12567) 

s_lshl_b32 s0, s4, 5 // (op 12567) 
s_mul_i32 s1, 6, s0 // (op 12567) 

v_add_u32 v0, v9, s1 // (op 12567) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12567) Store value





s_lshl_b32 s0, s2, 5 // (op 12582) 
s_mul_i32 s1, 7, s0 // (op 12582) 

v_add_u32 v0, v8, s1 // (op 12582) 


v_accvgpr_read v10, a28 // (op 12578) 
v_accvgpr_read v11, a29 // (op 12578) 
v_accvgpr_read v12, a30 // (op 12578) 
v_accvgpr_read v13, a31 // (op 12578) 
v_mul_f32 v14, v7, v10 // (op 12578) 
v_mul_f32 v15, v7, v11 // (op 12578) 
v_mul_f32 v16, v7, v12 // (op 12578) 
v_mul_f32 v17, v7, v13 // (op 12578) 

buffer_load_dwordx2 v[10:11], v0, s[40:43], 0 offen // (op 12582) Load value

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 12583) 
v_lshrrev_b32 v13, 16, v10 // (op 12583) 
v_lshlrev_b32 v12, 16, v12 // (op 12583) 
v_lshlrev_b32 v13, 16, v13 // (op 12583) 
v_mul_f32 v18, v5, v12 // (op 12583) 
v_mul_f32 v19, v5, v13 // (op 12583) 
v_and_b32 v12, 65535, v11 // (op 12583) 
v_lshrrev_b32 v13, 16, v11 // (op 12583) 
v_lshlrev_b32 v12, 16, v12 // (op 12583) 
v_lshlrev_b32 v13, 16, v13 // (op 12583) 
v_mul_f32 v20, v5, v12 // (op 12583) 
v_mul_f32 v21, v5, v13 // (op 12583) 


v_add_f32 v10, v18, v14 // (op 12579) 
v_add_f32 v11, v19, v15 // (op 12579) 
v_add_f32 v12, v20, v16 // (op 12579) 
v_add_f32 v13, v21, v17 // (op 12579) 


v_lshrrev_b32 v14, 16, v10 // (op 12580) 
v_lshrrev_b32 v15, 16, v11 // (op 12580) 
v_lshrrev_b32 v16, 16, v12 // (op 12580) 
v_lshrrev_b32 v17, 16, v13 // (op 12580) 


v_pack_b32_f16 v10, v14, v15 // (op 12581) 
v_pack_b32_f16 v11, v16, v17 // (op 12581) 

s_lshl_b32 s0, s4, 5 // (op 12581) 
s_mul_i32 s1, 7, s0 // (op 12581) 

v_add_u32 v0, v9, s1 // (op 12581) 


buffer_store_dwordx2 v[10:11], v0, s[44:47], 0 offen // (op 12581) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_ConditionalBottom_0_beta_116:








































































s_endpgm  // End of RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2
.LRR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_end:
.size RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2, .LRR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2_end-RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2
  .amdhsa_next_free_vgpr 476
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 98304 // lds bytes
  .amdhsa_accum_offset 220
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
    DataTypeB: F8
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
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2
    .symbol: RR_GEMM_TN_FP4_FP8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 98304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 218
    .agpr_count: 256
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

