

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4,@function
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 31106) 
v_mul_lo_u32 v3, v0, 12 // (op 31106) 

v_and_b32 v0, 15, v1 // (op 31106) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 31106) 


v_and_b32 v0, 63, v1 // (op 31106) 
v_lshrrev_b32 v3, 4, v0 // (op 31106) 

v_lshlrev_b32 v0, 5, v3 // (op 31106) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 31106) 


v_lshlrev_b32 v0, 2, v3 // (op 31106) 

v_lshrrev_b32 v3, 3, v0 // (op 31106) 

v_lshrrev_b32 v0, 6, v1 // (op 31163) 
v_and_b32 v4, 1, v0 // (op 31163) 

v_and_b32 v0, 15, v1 // (op 31163) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 31163) 


v_and_b32 v0, 63, v1 // (op 31163) 
v_lshrrev_b32 v4, 4, v0 // (op 31163) 

v_lshlrev_b32 v0, 5, v4 // (op 31163) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 31163) 


v_lshlrev_b32 v0, 2, v4 // (op 31163) 

v_lshrrev_b32 v4, 3, v0 // (op 31163) 

v_lshrrev_b32 v0, 7, v1 // (op 31230) 
v_mul_lo_u32 v5, v0, 12 // (op 31230) 

v_and_b32 v0, 15, v1 // (op 31230) 
v_lshl_add_u32 v6, v5, 4, v0 // (op 31230) 


v_and_b32 v0, 63, v1 // (op 31230) 
v_lshrrev_b32 v5, 4, v0 // (op 31230) 

v_lshlrev_b32 v0, 5, v5 // (op 31230) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 31230) 


v_lshlrev_b32 v0, 2, v5 // (op 31230) 

v_lshrrev_b32 v5, 3, v0 // (op 31230) 

v_lshrrev_b32 v0, 6, v1 // (op 31274) 
v_and_b32 v6, 1, v0 // (op 31274) 

v_and_b32 v0, 15, v1 // (op 31274) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 31274) 


v_and_b32 v0, 63, v1 // (op 31274) 
v_lshrrev_b32 v6, 4, v0 // (op 31274) 

v_lshlrev_b32 v0, 5, v6 // (op 31274) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 31274) 


v_lshlrev_b32 v0, 2, v6 // (op 31274) 

v_lshrrev_b32 v6, 3, v0 // (op 31274) 

v_lshrrev_b32 v0, 7, v1 // (op 31318) 
v_mul_lo_u32 v7, v0, 12 // (op 31318) 

v_and_b32 v0, 15, v1 // (op 31318) 
v_lshl_add_u32 v8, v7, 4, v0 // (op 31318) 


v_and_b32 v0, 63, v1 // (op 31318) 
v_lshrrev_b32 v7, 4, v0 // (op 31318) 

v_lshlrev_b32 v0, 5, v7 // (op 31318) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 31318) 


v_lshlrev_b32 v0, 2, v7 // (op 31318) 

v_lshrrev_b32 v7, 3, v0 // (op 31318) 

v_lshrrev_b32 v0, 6, v1 // (op 31362) 
v_and_b32 v8, 1, v0 // (op 31362) 

v_and_b32 v0, 15, v1 // (op 31362) 
v_lshl_add_u32 v9, v8, 6, v0 // (op 31362) 


v_and_b32 v0, 63, v1 // (op 31362) 
v_lshrrev_b32 v8, 4, v0 // (op 31362) 

v_lshlrev_b32 v0, 5, v8 // (op 31362) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 31362) 


v_lshlrev_b32 v0, 2, v8 // (op 31362) 

v_lshrrev_b32 v8, 3, v0 // (op 31362) 

v_lshrrev_b32 v0, 7, v1 // (op 31406) 
v_mul_lo_u32 v9, v0, 12 // (op 31406) 

v_and_b32 v0, 15, v1 // (op 31406) 
v_lshl_add_u32 v10, v9, 4, v0 // (op 31406) 


v_and_b32 v0, 63, v1 // (op 31406) 
v_lshrrev_b32 v9, 4, v0 // (op 31406) 

v_lshlrev_b32 v0, 5, v9 // (op 31406) 

v_lshl_add_u32 v9, v10, 7, v0 // (op 31406) 


v_lshlrev_b32 v0, 2, v9 // (op 31406) 

v_lshrrev_b32 v9, 3, v0 // (op 31406) 

v_lshrrev_b32 v0, 6, v1 // (op 31450) 
v_and_b32 v10, 1, v0 // (op 31450) 

v_and_b32 v0, 15, v1 // (op 31450) 
v_lshl_add_u32 v11, v10, 6, v0 // (op 31450) 


v_and_b32 v0, 63, v1 // (op 31450) 
v_lshrrev_b32 v10, 4, v0 // (op 31450) 

v_lshlrev_b32 v0, 5, v10 // (op 31450) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 31450) 


v_lshlrev_b32 v0, 2, v10 // (op 31450) 

v_lshrrev_b32 v10, 3, v0 // (op 31450) 

v_lshrrev_b32 v0, 7, v1 // (op 31045) 
v_mul_lo_u32 v11, v0, 12 // (op 31045) 

v_and_b32 v0, 63, v1 // (op 31045) 
v_lshrrev_b32 v12, 4, v0 // (op 31045) 
v_lshlrev_b32 v13, 2, v12 // (op 31045) 

v_lshl_add_u32 v12, v11, 4, v13 // (op 31045) 


v_mov_b32 v11, 192 // (op 31045) 
v_mul_lo_u32 v13, s16, v11 // (op 31045) 

v_add_u32 v13, v13, v12 // (op 31045) 

v_lshrrev_b32 v11, 2, v0 // (op 31045) 

v_and_b32 v0, 3, v11 // (op 31045) 

v_and_b32 v11, 3, v1 // (op 31045) 
v_lshl_add_u32 v12, v0, 2, v11 // (op 31045) 


v_lshrrev_b32 v0, 6, v1 // (op 31045) 
v_and_b32 v11, 1, v0 // (op 31045) 

v_lshl_add_u32 v0, v11, 6, v12 // (op 31045) 


v_lshl_add_u32 v11, s17, 7, v0 // (op 31045) 

v_mul_lo_u32 v0, v11, s2 // (op 31045) 

v_add_lshl_u32 v11, v13, v0, 1 // (op 31045) 


v_lshrrev_b32 v0, 7, v1 // (op 31184) 
v_mul_lo_u32 v12, v0, 12 // (op 31184) 

v_and_b32 v0, 63, v1 // (op 31184) 
v_lshrrev_b32 v13, 4, v0 // (op 31184) 
s_mul_i32 s1, s2, s9 // (op 31047) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 31047) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 31047) least significant
s_mul_hi_u32 s19, s2, s8 // (op 31047) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 31047) most significant: sum
s_add_u32 s1, s1, s19 // (op 31047) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 31047) 

v_lshlrev_b32 v14, 2, v13 // (op 31184) 

v_lshl_add_u32 v13, v12, 4, v14 // (op 31184) 


v_mov_b32 v12, 192 // (op 31184) 
v_mul_lo_u32 v14, s16, v12 // (op 31184) 

v_add_u32 v14, v14, v13 // (op 31184) 

v_lshrrev_b32 v12, 2, v0 // (op 31184) 

v_and_b32 v0, 3, v12 // (op 31184) 

v_and_b32 v12, 3, v1 // (op 31184) 
v_lshl_add_u32 v13, v0, 2, v12 // (op 31184) 


v_lshrrev_b32 v0, 6, v1 // (op 31184) 
v_and_b32 v12, 1, v0 // (op 31184) 

v_lshl_add_u32 v0, v12, 6, v13 // (op 31184) 


v_lshl_add_u32 v12, s17, 7, v0 // (op 31184) 

v_mul_lo_u32 v0, v12, s4 // (op 31184) 

v_add_lshl_u32 v12, v14, v0, 1 // (op 31184) 


s_mov_b32 s40, s6 // (op 31047) reg expression
s_mov_b32 s41, s7 // (op 31047) reg expression
s_mov_b32 s42, s18 // (op 31047) reg expression
s_mov_b32 s43, 131072 // (op 31047) call()

s_mul_i32 s1, s4, s9 // (op 31186) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 31186) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 31186) least significant
s_mul_hi_u32 s7, s4, s8 // (op 31186) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31186) most significant: sum
s_add_u32 s1, s1, s7 // (op 31186) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 1 // (op 31186) 

s_mov_b32 s44, s10 // (op 31186) reg expression
s_mov_b32 s45, s11 // (op 31186) reg expression
s_mov_b32 s46, s6 // (op 31186) reg expression
s_mov_b32 s47, 131072 // (op 31186) call()


v_mov_b32 v0, s14 // (op 2) Move value
v_mov_b32 v13, s15 // (op 35) Move value


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
v_accvgpr_write a128, 0.00000 // (op 8215) call()
v_accvgpr_write a129, 0.00000 // (op 8215) call()
v_accvgpr_write a130, 0.00000 // (op 8215) call()
v_accvgpr_write a131, 0.00000 // (op 8215) call()
v_accvgpr_write a132, 0.00000 // (op 8628) call()
v_accvgpr_write a133, 0.00000 // (op 8628) call()
v_accvgpr_write a134, 0.00000 // (op 8628) call()
v_accvgpr_write a135, 0.00000 // (op 8628) call()
v_accvgpr_write a136, 0.00000 // (op 8629) call()
v_accvgpr_write a137, 0.00000 // (op 8629) call()
v_accvgpr_write a138, 0.00000 // (op 8629) call()
v_accvgpr_write a139, 0.00000 // (op 8629) call()
v_accvgpr_write a140, 0.00000 // (op 8630) call()
v_accvgpr_write a141, 0.00000 // (op 8630) call()
v_accvgpr_write a142, 0.00000 // (op 8630) call()
v_accvgpr_write a143, 0.00000 // (op 8630) call()
v_accvgpr_write a144, 0.00000 // (op 9217) call()
v_accvgpr_write a145, 0.00000 // (op 9217) call()
v_accvgpr_write a146, 0.00000 // (op 9217) call()
v_accvgpr_write a147, 0.00000 // (op 9217) call()
v_accvgpr_write a148, 0.00000 // (op 9630) call()
v_accvgpr_write a149, 0.00000 // (op 9630) call()
v_accvgpr_write a150, 0.00000 // (op 9630) call()
v_accvgpr_write a151, 0.00000 // (op 9630) call()
v_accvgpr_write a152, 0.00000 // (op 9631) call()
v_accvgpr_write a153, 0.00000 // (op 9631) call()
v_accvgpr_write a154, 0.00000 // (op 9631) call()
v_accvgpr_write a155, 0.00000 // (op 9631) call()
v_accvgpr_write a156, 0.00000 // (op 9632) call()
v_accvgpr_write a157, 0.00000 // (op 9632) call()
v_accvgpr_write a158, 0.00000 // (op 9632) call()
v_accvgpr_write a159, 0.00000 // (op 9632) call()
v_accvgpr_write a160, 0.00000 // (op 10219) call()
v_accvgpr_write a161, 0.00000 // (op 10219) call()
v_accvgpr_write a162, 0.00000 // (op 10219) call()
v_accvgpr_write a163, 0.00000 // (op 10219) call()
v_accvgpr_write a164, 0.00000 // (op 10632) call()
v_accvgpr_write a165, 0.00000 // (op 10632) call()
v_accvgpr_write a166, 0.00000 // (op 10632) call()
v_accvgpr_write a167, 0.00000 // (op 10632) call()
v_accvgpr_write a168, 0.00000 // (op 10633) call()
v_accvgpr_write a169, 0.00000 // (op 10633) call()
v_accvgpr_write a170, 0.00000 // (op 10633) call()
v_accvgpr_write a171, 0.00000 // (op 10633) call()
v_accvgpr_write a172, 0.00000 // (op 10634) call()
v_accvgpr_write a173, 0.00000 // (op 10634) call()
v_accvgpr_write a174, 0.00000 // (op 10634) call()
v_accvgpr_write a175, 0.00000 // (op 10634) call()
v_accvgpr_write a176, 0.00000 // (op 11221) call()
v_accvgpr_write a177, 0.00000 // (op 11221) call()
v_accvgpr_write a178, 0.00000 // (op 11221) call()
v_accvgpr_write a179, 0.00000 // (op 11221) call()
v_accvgpr_write a180, 0.00000 // (op 11634) call()
v_accvgpr_write a181, 0.00000 // (op 11634) call()
v_accvgpr_write a182, 0.00000 // (op 11634) call()
v_accvgpr_write a183, 0.00000 // (op 11634) call()
v_accvgpr_write a184, 0.00000 // (op 11635) call()
v_accvgpr_write a185, 0.00000 // (op 11635) call()
v_accvgpr_write a186, 0.00000 // (op 11635) call()
v_accvgpr_write a187, 0.00000 // (op 11635) call()
v_accvgpr_write a188, 0.00000 // (op 11636) call()
v_accvgpr_write a189, 0.00000 // (op 11636) call()
v_accvgpr_write a190, 0.00000 // (op 11636) call()
v_accvgpr_write a191, 0.00000 // (op 11636) call()
v_lshrrev_b32 v14, 2, v1 // (op 30935) 
v_mov_b32 v15, 192 // (op 30935) 
v_mul_lo_u32 v16, s16, v15 // (op 30935) 

v_add_u32 v16, v16, v14 // (op 30935) 

v_and_b32 v14, -4, v1 // (op 30935) 
v_sub_u32 v15, v1, v14 // (op 30935) 

v_lshlrev_b32 v14, 5, v15 // (op 30935) 

v_mov_b32 v19, 0 // (op 30935) convert
v_mov_b32 v18, v14 // (op 30935) convert

v_mul_lo_u32 v15, v16, s13 // (op 30935) most significant: low of low * high
v_mul_hi_u32 v17, v16, s12 // (op 30935) most significant: high of low * low
v_mul_lo_u32 v14, v16, s12 // (op 30935) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 30935) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v18 // (op 30935) least significant half
v_addc_co_u32 v15, s[0:1], v15, v19, s[0:1] // (op 30935) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 30935) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 30935) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 30937) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 30937) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 30937) least significant
s_mul_hi_u32 s7, s12, s22 // (op 30937) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30937) most significant: sum
s_add_u32 s1, s1, s7 // (op 30937) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30937) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30937) 

s_mov_b32 s48, s20 // (op 30937) reg expression
s_mov_b32 s49, s21 // (op 30937) reg expression
s_mov_b32 s50, s0 // (op 30937) reg expression
s_mov_b32 s51, 131072 // (op 30937) call()

v_lshrrev_b32 v16, 2, v1 // (op 30954) 
v_mov_b32 v17, 192 // (op 30954) 
v_mul_lo_u32 v18, s16, v17 // (op 30954) 

v_add_u32 v18, v18, v16 // (op 30954) 

v_and_b32 v16, -4, v1 // (op 30954) 
v_sub_u32 v17, v1, v16 // (op 30954) 

v_lshlrev_b32 v16, 5, v17 // (op 30954) 

v_mov_b32 v21, 0 // (op 30954) convert
v_mov_b32 v20, v16 // (op 30954) convert

v_mul_lo_u32 v17, v18, s13 // (op 30954) most significant: low of low * high
v_mul_hi_u32 v19, v18, s12 // (op 30954) most significant: high of low * low
v_mul_lo_u32 v16, v18, s12 // (op 30954) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 30954) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v20 // (op 30954) least significant half
v_addc_co_u32 v17, s[0:1], v17, v21, s[0:1] // (op 30954) most significant half



v_lshlrev_b64 v[18:19], 2, v[16:17] // (op 30954) 

v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 30954) 

s_mul_i32 s1, s12, s23 // (op 30956) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 30956) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 30956) least significant
s_mul_hi_u32 s7, s12, s22 // (op 30956) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30956) most significant: sum
s_add_u32 s1, s1, s7 // (op 30956) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30956) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30956) 

s_mov_b32 s48, s20 // (op 30956) reg expression
s_mov_b32 s49, s21 // (op 30956) reg expression
s_mov_b32 s50, s0 // (op 30956) reg expression
s_mov_b32 s51, 131072 // (op 30956) call()

v_and_b32 v18, -4, v1 // (op 30999) 
v_sub_u32 v19, v1, v18 // (op 30999) 

v_lshlrev_b32 v18, 5, v19 // (op 30999) 

v_mov_b32 v21, 0 // (op 30999) convert
v_mov_b32 v20, v18 // (op 30999) convert

v_lshrrev_b32 v18, 2, v1 // (op 30999) 
v_lshl_add_u32 v19, s17, 7, v18 // (op 30999) 

v_mul_lo_u32 v23, v19, s25 // (op 30999) most significant: low of low * high
v_mul_hi_u32 v18, v19, s24 // (op 30999) most significant: high of low * low
v_mul_lo_u32 v22, v19, s24 // (op 30999) least significant: low of low * low
v_add_u32 v23, v23, v18 // (op 30999) most significant: sum


v_add_co_u32 v18, s[0:1], v20, v22 // (op 30999) least significant half
v_addc_co_u32 v19, s[0:1], v21, v23, s[0:1] // (op 30999) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 30999) 


v_lshrrev_b64 v[20:21], 3, v[18:19] // (op 30999) 

s_mul_i32 s1, s24, s9 // (op 31001) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 31001) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 31001) least significant
s_mul_hi_u32 s7, s24, s8 // (op 31001) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31001) most significant: sum
s_add_u32 s1, s1, s7 // (op 31001) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 31001) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 31001) 

s_mov_b32 s52, s26 // (op 31001) reg expression
s_mov_b32 s53, s27 // (op 31001) reg expression
s_mov_b32 s54, s0 // (op 31001) reg expression
s_mov_b32 s55, 131072 // (op 31001) call()

v_and_b32 v18, -4, v1 // (op 31018) 
v_sub_u32 v19, v1, v18 // (op 31018) 

v_lshlrev_b32 v18, 5, v19 // (op 31018) 

v_mov_b32 v23, 0 // (op 31018) convert
v_mov_b32 v22, v18 // (op 31018) convert

v_lshrrev_b32 v18, 2, v1 // (op 31018) 
v_lshl_add_u32 v19, s17, 7, v18 // (op 31018) 

v_mul_lo_u32 v25, v19, s25 // (op 31018) most significant: low of low * high
v_mul_hi_u32 v18, v19, s24 // (op 31018) most significant: high of low * low
v_mul_lo_u32 v24, v19, s24 // (op 31018) least significant: low of low * low
v_add_u32 v25, v25, v18 // (op 31018) most significant: sum


v_add_co_u32 v18, s[0:1], v22, v24 // (op 31018) least significant half
v_addc_co_u32 v19, s[0:1], v23, v25, s[0:1] // (op 31018) most significant half

v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 31018) 


v_lshrrev_b64 v[22:23], 3, v[18:19] // (op 31018) 

s_mul_i32 s1, s24, s9 // (op 31020) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 31020) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 31020) least significant
s_mul_hi_u32 s7, s24, s8 // (op 31020) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31020) most significant: sum
s_add_u32 s1, s1, s7 // (op 31020) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 31020) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 31020) 

s_mov_b32 s52, s26 // (op 31020) reg expression
s_mov_b32 s53, s27 // (op 31020) reg expression
s_mov_b32 s54, s0 // (op 31020) reg expression
s_mov_b32 s55, 131072 // (op 31020) call()

v_and_b32 v18, 63, v1 // (op 31472) 
v_lshrrev_b32 v19, 4, v18 // (op 31472) 
v_and_b32 v24, 3, v19 // (op 31472) 

v_and_b32 v19, 15, v1 // (op 31472) 
v_lshl_add_u32 v25, v24, 4, v19 // (op 31472) 


v_lshrrev_b32 v19, 7, v1 // (op 31472) 
v_mul_lo_u32 v24, v19, 3 // (op 31472) 

v_lshl_add_u32 v19, v24, 6, v25 // (op 31472) 


v_mov_b32 v24, 192 // (op 31472) 
v_mul_lo_u32 v25, s16, v24 // (op 31472) 

v_add_u32 v25, v25, v19 // (op 31472) 

v_lshrrev_b32 v19, 6, v18 // (op 31472) 

v_lshlrev_b32 v18, 2, v19 // (op 31472) 

v_mul_lo_u32 v19, v25, s28 // (op 31472) 
v_add_u32 v19, v19, v18 // (op 31472) 


s_mul_i32 s1, s28, s23 // (op 31474) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 31474) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 31474) least significant
s_mul_hi_u32 s7, s28, s22 // (op 31474) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31474) most significant: sum
s_add_u32 s1, s1, s7 // (op 31474) most significant: sum


s_mov_b32 s56, s30 // (op 31474) reg expression
s_mov_b32 s57, s31 // (op 31474) reg expression
s_mov_b32 s58, s0 // (op 31474) reg expression
s_mov_b32 s59, 131072 // (op 31474) call()

v_and_b32 v18, 63, v1 // (op 31493) 
v_lshrrev_b32 v24, 4, v18 // (op 31493) 
v_and_b32 v25, 3, v24 // (op 31493) 

v_and_b32 v24, 15, v1 // (op 31493) 
v_lshl_add_u32 v26, v25, 4, v24 // (op 31493) 


v_lshrrev_b32 v24, 7, v1 // (op 31493) 
v_mul_lo_u32 v25, v24, 3 // (op 31493) 

v_lshl_add_u32 v24, v25, 6, v26 // (op 31493) 


v_mov_b32 v25, 192 // (op 31493) 
v_mul_lo_u32 v26, s16, v25 // (op 31493) 

v_add_u32 v26, v26, v24 // (op 31493) 

v_lshrrev_b32 v24, 6, v18 // (op 31493) 

v_lshlrev_b32 v18, 2, v24 // (op 31493) 

v_mul_lo_u32 v24, v26, s28 // (op 31493) 
v_add_u32 v24, v24, v18 // (op 31493) 


s_mul_i32 s1, s28, s23 // (op 31495) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 31495) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 31495) least significant
s_mul_hi_u32 s7, s28, s22 // (op 31495) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31495) most significant: sum
s_add_u32 s1, s1, s7 // (op 31495) most significant: sum


s_mov_b32 s56, s30 // (op 31495) reg expression
s_mov_b32 s57, s31 // (op 31495) reg expression
s_mov_b32 s58, s0 // (op 31495) reg expression
s_mov_b32 s59, 131072 // (op 31495) call()

v_and_b32 v18, 63, v1 // (op 31524) 
v_lshrrev_b32 v25, 4, v18 // (op 31524) 
v_and_b32 v26, 3, v25 // (op 31524) 

v_and_b32 v25, 15, v1 // (op 31524) 
v_lshl_add_u32 v27, v26, 4, v25 // (op 31524) 


v_lshrrev_b32 v25, 6, v1 // (op 31524) 
v_and_b32 v26, 1, v25 // (op 31524) 

v_lshl_add_u32 v25, v26, 6, v27 // (op 31524) 


v_lshl_add_u32 v26, s17, 7, v25 // (op 31524) 

v_mul_lo_u32 v25, v26, s32 // (op 31524) 

v_lshrrev_b32 v26, 6, v18 // (op 31524) 

v_lshl_add_u32 v18, v26, 2, v25 // (op 31524) 


s_mul_i32 s1, s32, s9 // (op 31526) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 31526) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 31526) least significant
s_mul_hi_u32 s7, s32, s8 // (op 31526) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31526) most significant: sum
s_add_u32 s1, s1, s7 // (op 31526) most significant: sum


s_mov_b32 s60, s34 // (op 31526) reg expression
s_mov_b32 s61, s35 // (op 31526) reg expression
s_mov_b32 s62, s0 // (op 31526) reg expression
s_mov_b32 s63, 131072 // (op 31526) call()

v_and_b32 v25, 63, v1 // (op 31545) 
v_lshrrev_b32 v26, 4, v25 // (op 31545) 
v_and_b32 v27, 3, v26 // (op 31545) 

v_and_b32 v26, 15, v1 // (op 31545) 
v_lshl_add_u32 v28, v27, 4, v26 // (op 31545) 


v_lshrrev_b32 v26, 6, v1 // (op 31545) 
v_and_b32 v27, 1, v26 // (op 31545) 

v_lshl_add_u32 v26, v27, 6, v28 // (op 31545) 


v_lshl_add_u32 v27, s17, 7, v26 // (op 31545) 

v_mul_lo_u32 v26, v27, s32 // (op 31545) 

v_lshrrev_b32 v27, 6, v25 // (op 31545) 

v_lshl_add_u32 v25, v27, 2, v26 // (op 31545) 


s_mul_i32 s1, s32, s9 // (op 31547) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 31547) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 31547) least significant
s_mul_hi_u32 s7, s32, s8 // (op 31547) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31547) most significant: sum
s_add_u32 s1, s1, s7 // (op 31547) most significant: sum


s_mov_b32 s60, s34 // (op 31547) reg expression
s_mov_b32 s61, s35 // (op 31547) reg expression
s_mov_b32 s62, s0 // (op 31547) reg expression
s_mov_b32 s63, 131072 // (op 31547) call()

v_and_b32 v26, -4, v1 // (op 31146) 
v_sub_u32 v27, v1, v26 // (op 31146) 

v_lshlrev_b32 v26, 5, v27 // (op 31146) 

v_lshrrev_b32 v27, 2, v1 // (op 31146) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 31146) 


v_lshlrev_b32 v26, 2, v28 // (op 31146) 

v_lshrrev_b32 v27, 3, v26 // (op 31146) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 31146) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 30829) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30829) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30829) 


v_mov_b64 v[28:29], v[20:21] // (op 30829) 
s_mov_b32 m0, s0 // (op 30829) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30829) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30829) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30829) most significant half

s_add_u32 m0, m0, 2048 // (op 30829) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30829) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30829) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30829) most significant half

s_add_u32 m0, m0, 2048 // (op 30829) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30829) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30829) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30829) most significant half

s_add_u32 m0, m0, 2048 // (op 30829) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30829) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 31089) 
v_sub_u32 v27, v1, v26 // (op 31089) 

v_lshlrev_b32 v26, 5, v27 // (op 31089) 

v_lshrrev_b32 v27, 2, v1 // (op 31089) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 31089) 


v_lshlrev_b32 v26, 2, v28 // (op 31089) 

v_lshrrev_b32 v27, 3, v26 // (op 31089) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 31089) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 30834) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30834) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30834) 


v_mov_b64 v[28:29], v[14:15] // (op 30834) 
s_add_u32 s1, 8192, s0 // (op 30834) 
s_mov_b32 m0, s1 // (op 30834) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30834) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30834) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30834) most significant half

s_add_u32 m0, m0, 2048 // (op 30834) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30834) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30834) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30834) most significant half

s_add_u32 m0, m0, 2048 // (op 30834) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30834) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30834) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30834) most significant half

s_add_u32 m0, m0, 2048 // (op 30834) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30834) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30834) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30834) most significant half

s_add_u32 m0, m0, 2048 // (op 30834) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30834) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30834) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30834) most significant half

s_add_u32 m0, m0, 2048 // (op 30834) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30834) Load value direct to lds






v_mov_b32 v26, v18 // (op 76) reg expression

buffer_load_dwordx4 v[28:31], v26, s[60:63], 0 offen // (op 76) Load value




s_lshl_b32 s0, s28, 6 // (op 4385) 
v_add_u32 v18, v19, s0 // (op 4385) 


buffer_load_dwordx4 v[32:35], v18, s[56:59], 0 offen // (op 4385) Load value



s_lshl_b32 s0, s28, 7 // (op 8393) 
v_add_u32 v18, v19, s0 // (op 8393) 


buffer_load_dwordx4 v[36:39], v18, s[56:59], 0 offen // (op 8393) Load value



v_mov_b32 v18, v19 // (op 72) reg expression

buffer_load_dwordx4 v[40:43], v18, s[56:59], 0 offen // (op 72) Load value


v_and_b32 v18, -4, v1 // (op 31257) 
v_sub_u32 v19, v1, v18 // (op 31257) 

v_lshlrev_b32 v18, 5, v19 // (op 31257) 

v_lshrrev_b32 v19, 2, v1 // (op 31257) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31257) 


v_lshlrev_b32 v27, 2, v26 // (op 31257) 

v_lshrrev_b32 v26, 3, v27 // (op 31257) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31257) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 30839) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30839) 

v_lshrrev_b64 v[18:19], 3, s[10:11] // (op 30839) 


v_mov_b64 v[26:27], v[20:21] // (op 30839) 
s_add_u32 s1, 20480, s0 // (op 30839) 
s_mov_b32 m0, s1 // (op 30839) reg expression
buffer_load_dwordx4 v26, s[52:55], 64 offen offset: 0 lds // (op 30839) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v18 // (op 30839) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30839) most significant half

s_add_u32 m0, m0, 2048 // (op 30839) 
buffer_load_dwordx4 v26, s[52:55], 64 offen offset: 0 lds // (op 30839) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v18 // (op 30839) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30839) most significant half

s_add_u32 m0, m0, 2048 // (op 30839) 
buffer_load_dwordx4 v26, s[52:55], 64 offen offset: 0 lds // (op 30839) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v18 // (op 30839) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30839) most significant half

s_add_u32 m0, m0, 2048 // (op 30839) 
buffer_load_dwordx4 v26, s[52:55], 64 offen offset: 0 lds // (op 30839) Load value direct to lds




v_and_b32 v18, -4, v1 // (op 31213) 
v_sub_u32 v19, v1, v18 // (op 31213) 

v_lshlrev_b32 v18, 5, v19 // (op 31213) 

v_lshrrev_b32 v19, 2, v1 // (op 31213) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31213) 


v_lshlrev_b32 v27, 2, v26 // (op 31213) 

v_lshrrev_b32 v26, 3, v27 // (op 31213) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31213) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 30841) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30841) 

v_lshrrev_b64 v[18:19], 3, s[10:11] // (op 30841) 


v_mov_b64 v[26:27], v[14:15] // (op 30841) 
s_add_u32 s1, 28672, s0 // (op 30841) 
s_mov_b32 m0, s1 // (op 30841) reg expression
buffer_load_dwordx4 v26, s[48:51], 64 offen offset: 0 lds // (op 30841) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v18 // (op 30841) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30841) most significant half

s_add_u32 m0, m0, 2048 // (op 30841) 
buffer_load_dwordx4 v26, s[48:51], 64 offen offset: 0 lds // (op 30841) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v18 // (op 30841) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30841) most significant half

s_add_u32 m0, m0, 2048 // (op 30841) 
buffer_load_dwordx4 v26, s[48:51], 64 offen offset: 0 lds // (op 30841) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v18 // (op 30841) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30841) most significant half

s_add_u32 m0, m0, 2048 // (op 30841) 
buffer_load_dwordx4 v26, s[48:51], 64 offen offset: 0 lds // (op 30841) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v18 // (op 30841) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30841) most significant half

s_add_u32 m0, m0, 2048 // (op 30841) 
buffer_load_dwordx4 v26, s[48:51], 64 offen offset: 0 lds // (op 30841) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v18 // (op 30841) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30841) most significant half

s_add_u32 m0, m0, 2048 // (op 30841) 
buffer_load_dwordx4 v26, s[48:51], 64 offen offset: 0 lds // (op 30841) Load value direct to lds




v_and_b32 v18, -4, v1 // (op 31345) 
v_sub_u32 v19, v1, v18 // (op 31345) 

v_lshlrev_b32 v18, 5, v19 // (op 31345) 

v_lshrrev_b32 v19, 2, v1 // (op 31345) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31345) 


v_lshlrev_b32 v27, 2, v26 // (op 31345) 

v_lshrrev_b32 v26, 3, v27 // (op 31345) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31345) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 30843) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30843) 

v_lshrrev_b64 v[18:19], 3, s[10:11] // (op 30843) 


v_mov_b64 v[26:27], v[20:21] // (op 30843) 
s_add_u32 s1, 40960, s0 // (op 30843) 
s_mov_b32 m0, s1 // (op 30843) reg expression
s_mov_b32 s6, 128 // (op 30843) call()
buffer_load_dwordx4 v26, s[52:55], s6 offen offset: 0 lds // (op 30843) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30843) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30843) most significant half

s_add_u32 m0, m0, 2048 // (op 30843) 
buffer_load_dwordx4 v26, s[52:55], s6 offen offset: 0 lds // (op 30843) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30843) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30843) most significant half

s_add_u32 m0, m0, 2048 // (op 30843) 
buffer_load_dwordx4 v26, s[52:55], s6 offen offset: 0 lds // (op 30843) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30843) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30843) most significant half

s_add_u32 m0, m0, 2048 // (op 30843) 
buffer_load_dwordx4 v26, s[52:55], s6 offen offset: 0 lds // (op 30843) Load value direct to lds





v_and_b32 v18, -4, v1 // (op 31301) 
v_sub_u32 v19, v1, v18 // (op 31301) 

v_lshlrev_b32 v18, 5, v19 // (op 31301) 

v_lshrrev_b32 v19, 2, v1 // (op 31301) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31301) 


v_lshlrev_b32 v27, 2, v26 // (op 31301) 

v_lshrrev_b32 v26, 3, v27 // (op 31301) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31301) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 30845) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30845) 

v_lshrrev_b64 v[18:19], 3, s[10:11] // (op 30845) 


v_mov_b64 v[26:27], v[14:15] // (op 30845) 
s_add_u32 s1, 49152, s0 // (op 30845) 
s_mov_b32 m0, s1 // (op 30845) reg expression
s_mov_b32 s6, 128 // (op 30845) call()
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30845) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30845) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30845) most significant half

s_add_u32 m0, m0, 2048 // (op 30845) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30845) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30845) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30845) most significant half

s_add_u32 m0, m0, 2048 // (op 30845) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30845) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30845) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30845) most significant half

s_add_u32 m0, m0, 2048 // (op 30845) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30845) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30845) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30845) most significant half

s_add_u32 m0, m0, 2048 // (op 30845) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30845) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30845) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30845) most significant half

s_add_u32 m0, m0, 2048 // (op 30845) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30845) Load value direct to lds





v_and_b32 v18, -4, v1 // (op 31433) 
v_sub_u32 v19, v1, v18 // (op 31433) 

v_lshlrev_b32 v18, 5, v19 // (op 31433) 

v_lshrrev_b32 v19, 2, v1 // (op 31433) 
v_lshl_add_u32 v26, v19, 7, v18 // (op 31433) 


v_lshlrev_b32 v27, 2, v26 // (op 31433) 

v_lshrrev_b32 v26, 3, v27 // (op 31433) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31433) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 30847) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30847) 

v_lshrrev_b64 v[18:19], 3, s[10:11] // (op 30847) 


v_mov_b64 v[26:27], v[20:21] // (op 30847) 
s_add_u32 s1, 61440, s0 // (op 30847) 
s_mov_b32 m0, s1 // (op 30847) reg expression
s_mov_b32 s6, 192 // (op 30847) call()
buffer_load_dwordx4 v26, s[52:55], s6 offen offset: 0 lds // (op 30847) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30847) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30847) most significant half

s_add_u32 m0, m0, 2048 // (op 30847) 
buffer_load_dwordx4 v26, s[52:55], s6 offen offset: 0 lds // (op 30847) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30847) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30847) most significant half

s_add_u32 m0, m0, 2048 // (op 30847) 
buffer_load_dwordx4 v26, s[52:55], s6 offen offset: 0 lds // (op 30847) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v18 // (op 30847) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30847) most significant half

s_add_u32 m0, m0, 2048 // (op 30847) 
buffer_load_dwordx4 v26, s[52:55], s6 offen offset: 0 lds // (op 30847) Load value direct to lds






v_and_b32 v18, -4, v1 // (op 31389) 
v_sub_u32 v19, v1, v18 // (op 31389) 

v_lshlrev_b32 v18, 5, v19 // (op 31389) 

v_lshrrev_b32 v19, 2, v1 // (op 31389) 
v_lshl_add_u32 v20, v19, 7, v18 // (op 31389) 


v_lshlrev_b32 v21, 2, v20 // (op 31389) 

v_lshrrev_b32 v18, 3, v21 // (op 31389) 

s_nop 0
v_readfirstlane_b32 s0, v18 // (op 31389) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 30849) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30849) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30849) 


v_mov_b64 v[18:19], v[14:15] // (op 30849) 
s_add_u32 s1, 69632, s0 // (op 30849) 
s_mov_b32 m0, s1 // (op 30849) reg expression
s_mov_b32 s6, 192 // (op 30849) call()
buffer_load_dwordx4 v18, s[48:51], s6 offen offset: 0 lds // (op 30849) Load value direct to lds
v_add_co_u32 v18, s[10:11], v18, v26 // (op 30849) least significant half
v_addc_co_u32 v19, s[10:11], 0, v19, s[10:11] // (op 30849) most significant half

s_add_u32 m0, m0, 2048 // (op 30849) 
buffer_load_dwordx4 v18, s[48:51], s6 offen offset: 0 lds // (op 30849) Load value direct to lds
v_add_co_u32 v18, s[10:11], v18, v26 // (op 30849) least significant half
v_addc_co_u32 v19, s[10:11], 0, v19, s[10:11] // (op 30849) most significant half

s_add_u32 m0, m0, 2048 // (op 30849) 
buffer_load_dwordx4 v18, s[48:51], s6 offen offset: 0 lds // (op 30849) Load value direct to lds
v_add_co_u32 v18, s[10:11], v18, v26 // (op 30849) least significant half
v_addc_co_u32 v19, s[10:11], 0, v19, s[10:11] // (op 30849) most significant half

s_add_u32 m0, m0, 2048 // (op 30849) 
buffer_load_dwordx4 v18, s[48:51], s6 offen offset: 0 lds // (op 30849) Load value direct to lds
v_add_co_u32 v18, s[10:11], v18, v26 // (op 30849) least significant half
v_addc_co_u32 v19, s[10:11], 0, v19, s[10:11] // (op 30849) most significant half

s_add_u32 m0, m0, 2048 // (op 30849) 
buffer_load_dwordx4 v18, s[48:51], s6 offen offset: 0 lds // (op 30849) Load value direct to lds
v_add_co_u32 v18, s[10:11], v18, v26 // (op 30849) least significant half
v_addc_co_u32 v19, s[10:11], 0, v19, s[10:11] // (op 30849) most significant half

s_add_u32 m0, m0, 2048 // (op 30849) 
buffer_load_dwordx4 v18, s[48:51], s6 offen offset: 0 lds // (op 30849) Load value direct to lds






s_waitcnt vmcnt(40)
s_barrier  // (op 32146) 



ds_read_b128 v[18:21], v4 // (op 29902) Load local data 



ds_read_b128 v[44:47], v4 offset:1024 // (op 29907) Load local data 



ds_read_b128 v[48:51], v4 offset:2048 // (op 29912) Load local data 



ds_read_b128 v[52:55], v4 offset:3072 // (op 29917) Load local data 
s_waitcnt vmcnt(34)
s_barrier  // (op 32130) 



ds_read_b128 v[56:59], v3 offset:8192 // (op 29842) Load local data 



ds_read_b128 v[60:63], v3 offset:9216 // (op 29847) Load local data 



ds_read_b128 v[64:67], v3 offset:10240 // (op 29852) Load local data 



ds_read_b128 v[68:71], v3 offset:11264 // (op 29857) Load local data 



ds_read_b128 v[72:75], v3 offset:12288 // (op 29862) Load local data 



ds_read_b128 v[76:79], v3 offset:13312 // (op 29867) Load local data 



ds_read_b128 v[80:83], v3 offset:14336 // (op 29872) Load local data 



ds_read_b128 v[84:87], v3 offset:15360 // (op 29877) Load local data 



ds_read_b128 v[88:91], v3 offset:16384 // (op 29882) Load local data 



ds_read_b128 v[92:95], v3 offset:17408 // (op 29887) Load local data 



ds_read_b128 v[96:99], v3 offset:18432 // (op 29892) Load local data 



ds_read_b128 v[100:103], v3 offset:19456 // (op 29897) Load local data 
v_and_b32 v14, -4, v1 // (op 31061) 
v_sub_u32 v15, v1, v14 // (op 31061) 

v_lshlrev_b32 v14, 5, v15 // (op 31061) 

v_lshrrev_b32 v15, 2, v1 // (op 31061) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31061) 


v_lshlrev_b32 v27, 2, v26 // (op 31061) 

v_lshrrev_b32 v26, 3, v27 // (op 31061) 

s_nop 0
v_readfirstlane_b32 s0, v26 // (op 31061) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31118) 
v_sub_u32 v15, v1, v14 // (op 31118) 

v_lshlrev_b32 v14, 5, v15 // (op 31118) 

v_lshrrev_b32 v15, 2, v1 // (op 31118) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31118) 


v_lshlrev_b32 v27, 2, v26 // (op 31118) 

v_lshrrev_b32 v26, 3, v27 // (op 31118) 

s_nop 0
v_readfirstlane_b32 s1, v26 // (op 31118) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31198) 
v_sub_u32 v15, v1, v14 // (op 31198) 

v_lshlrev_b32 v14, 5, v15 // (op 31198) 

v_lshrrev_b32 v15, 2, v1 // (op 31198) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31198) 


v_lshlrev_b32 v27, 2, v26 // (op 31198) 

v_lshrrev_b32 v26, 3, v27 // (op 31198) 

s_nop 0
v_readfirstlane_b32 s6, v26 // (op 31198) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31242) 
v_sub_u32 v15, v1, v14 // (op 31242) 

v_lshlrev_b32 v14, 5, v15 // (op 31242) 

v_lshrrev_b32 v15, 2, v1 // (op 31242) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31242) 


v_lshlrev_b32 v27, 2, v26 // (op 31242) 

v_lshrrev_b32 v26, 3, v27 // (op 31242) 

s_nop 0
v_readfirstlane_b32 s7, v26 // (op 31242) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31286) 
v_sub_u32 v15, v1, v14 // (op 31286) 

v_lshlrev_b32 v14, 5, v15 // (op 31286) 

v_lshrrev_b32 v15, 2, v1 // (op 31286) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31286) 


v_lshlrev_b32 v27, 2, v26 // (op 31286) 

v_lshrrev_b32 v26, 3, v27 // (op 31286) 

s_nop 0
v_readfirstlane_b32 s10, v26 // (op 31286) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31330) 
v_sub_u32 v15, v1, v14 // (op 31330) 

v_lshlrev_b32 v14, 5, v15 // (op 31330) 

v_lshrrev_b32 v15, 2, v1 // (op 31330) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31330) 


v_lshlrev_b32 v27, 2, v26 // (op 31330) 

v_lshrrev_b32 v26, 3, v27 // (op 31330) 

s_nop 0
v_readfirstlane_b32 s11, v26 // (op 31330) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31374) 
v_sub_u32 v15, v1, v14 // (op 31374) 

v_lshlrev_b32 v14, 5, v15 // (op 31374) 

v_lshrrev_b32 v15, 2, v1 // (op 31374) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31374) 


v_lshlrev_b32 v27, 2, v26 // (op 31374) 

v_lshrrev_b32 v26, 3, v27 // (op 31374) 

s_nop 0
v_readfirstlane_b32 s14, v26 // (op 31374) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 31418) 
v_sub_u32 v15, v1, v14 // (op 31418) 

v_lshlrev_b32 v14, 5, v15 // (op 31418) 

v_lshrrev_b32 v15, 2, v1 // (op 31418) 
v_lshl_add_u32 v26, v15, 7, v14 // (op 31418) 


v_lshlrev_b32 v27, 2, v26 // (op 31418) 

v_lshrrev_b32 v26, 3, v27 // (op 31418) 

s_nop 0
v_readfirstlane_b32 s15, v26 // (op 31418) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 11277) call()
s_ashr_i64 s[38:39], s[36:37], 7 // (op 11276) 
s_mov_b32 s19, s38 // (op 11276) convert

s_and_b32 s38, s19, 4294967292 // (op 11276) 

s_cmp_lt_u32 s18, s38 // (op 11276) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopBottom_KLoop_11276 // (op 11276) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopBottom_KLoop_11276 if false)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopTop_KLoop_11276:
s_add_u32 s18, s18, 4 // (op 11278) 
s_waitcnt vmcnt(33)
v_mov_b32 v104, v28 // (op 30687) reg expression
v_mov_b32 v105, v29 // (op 30687) reg expression
v_mov_b32 v106, v30 // (op 30687) reg expression
v_mov_b32 v107, v31 // (op 30687) reg expression
s_waitcnt lgkmcnt(12)
s_barrier  // (op 32149) 
s_waitcnt vmcnt(32)
v_mov_b32 v108, v32 // (op 30700) reg expression

v_mov_b32 v109, v33 // (op 30700) reg expression
v_mov_b32 v110, v34 // (op 30700) reg expression
v_mov_b32 v111, v35 // (op 30700) reg expression
s_lshl_b64 s[38:39], s[24:25], 5 // (op 30782) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30782) 

v_lshrrev_b64 v[14:15], 3, s[64:65] // (op 30782) 


v_mov_b64 v[26:27], v[22:23] // (op 30782) 
v_mov_b64 v[112:113], v[26:27] // (op 30782) 
s_mov_b32 m0, s1 // (op 30782) reg expression
s_mov_b32 s19, 256 // (op 30782) call()
buffer_load_dwordx4 v112, s[52:55], s19 offen offset: 0 lds // (op 30782) Load value direct to lds
v_add_co_u32 v112, s[38:39], v112, v14 // (op 30782) least significant half
v_addc_co_u32 v113, s[38:39], 0, v113, s[38:39] // (op 30782) most significant half

s_add_u32 m0, m0, 2048 // (op 30782) 
buffer_load_dwordx4 v112, s[52:55], s19 offen offset: 0 lds // (op 30782) Load value direct to lds
v_add_co_u32 v112, s[38:39], v112, v14 // (op 30782) least significant half
v_addc_co_u32 v113, s[38:39], 0, v113, s[38:39] // (op 30782) most significant half

s_add_u32 m0, m0, 2048 // (op 30782) 
buffer_load_dwordx4 v112, s[52:55], s19 offen offset: 0 lds // (op 30782) Load value direct to lds
v_add_co_u32 v112, s[38:39], v112, v14 // (op 30782) least significant half
v_addc_co_u32 v113, s[38:39], 0, v113, s[38:39] // (op 30782) most significant half

s_add_u32 m0, m0, 2048 // (op 30782) 
s_waitcnt vmcnt(34)
v_mov_b32 v114, v36 // (op 30713) reg expression
v_mov_b32 v115, v37 // (op 30713) reg expression
v_mov_b32 v116, v38 // (op 30713) reg expression
v_mov_b32 v117, v39 // (op 30713) reg expression
s_waitcnt vmcnt(33)
v_mov_b32 v118, v40 // (op 30726) reg expression
v_mov_b32 v119, v41 // (op 30726) reg expression
v_mov_b32 v120, v42 // (op 30726) reg expression
v_mov_b32 v121, v43 // (op 30726) reg expression
buffer_load_dwordx4 v112, s[52:55], s19 offen offset: 0 lds // (op 30782) Load value direct to lds



v_bfe_u32 v122, v118, 0, 8 // (op 28124) 
v_bfe_u32 v123, v118, 8, 8 // (op 28124) 
v_bfe_u32 v124, v118, 16, 8 // (op 28124) 
v_bfe_u32 v125, v118, 24, 8 // (op 28124) 
v_permlane16_swap_b32 v122, v123 // (op 28124) 
s_nop 0
v_permlane16_swap_b32 v124, v125 // (op 28124) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v122, v124 // (op 28124) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v123, v125 // (op 28124) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 32133) 

s_lshl_b64 s[38:39], s[12:13], 5 // (op 30777) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30777) 

v_lshrrev_b64 v[14:15], 3, s[64:65] // (op 30777) 


v_mov_b64 v[112:113], v[16:17] // (op 30777) 
v_mov_b64 v[126:127], v[112:113] // (op 30777) 
s_add_u32 s19, 8192, s0 // (op 30777) 
s_mov_b32 m0, s19 // (op 30777) reg expression
s_mov_b32 s38, 256 // (op 30777) call()
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30777) Load value direct to lds
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30777) least significant half
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30777) most significant half

s_add_u32 m0, m0, 2048 // (op 30777) 
v_bfe_u32 v128, v104, 0, 8 // (op 28517) 
v_bfe_u32 v129, v104, 8, 8 // (op 28517) 
v_bfe_u32 v130, v104, 16, 8 // (op 28517) 
v_bfe_u32 v131, v104, 24, 8 // (op 28517) 
v_permlane16_swap_b32 v128, v129 // (op 28517) 
s_nop 0
v_permlane16_swap_b32 v130, v131 // (op 28517) Wait state hazard: VALU Write Readlane Hazard
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30777) Load value direct to lds
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30777) least significant half
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30777) most significant half

s_add_u32 m0, m0, 2048 // (op 30777) 
v_permlane32_swap_b32 v128, v130 // (op 28517) 
v_permlane32_swap_b32 v129, v131 // (op 28517) 
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30777) Load value direct to lds
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30777) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[18:21], a[0:3], v122, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 78) 
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30777) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[44:47], a[4:7], v122, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 421) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[48:51], a[8:11], v122, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 647) 
s_add_u32 m0, m0, 2048 // (op 30777) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[52:55], a[12:15], v122, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 873) 
v_bfe_u32 v132, v108, 0, 8 // (op 28189) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[18:21], a[16:19], v123, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1545) 
v_bfe_u32 v133, v108, 8, 8 // (op 28189) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[44:47], a[20:23], v123, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1469) 
v_bfe_u32 v134, v108, 16, 8 // (op 28189) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[48:51], a[24:27], v123, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1393) 
v_bfe_u32 v135, v108, 24, 8 // (op 28189) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[52:55], a[28:31], v123, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1317) 
v_permlane16_swap_b32 v132, v133 // (op 28189) 
v_permlane16_swap_b32 v134, v135 // (op 28189) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[18:21], a[32:35], v124, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2547) 
s_nop 0
v_permlane32_swap_b32 v132, v134 // (op 28189) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[44:47], a[36:39], v124, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2471) 
v_permlane32_swap_b32 v133, v135 // (op 28189) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[48:51], a[40:43], v124, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2395) 
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30777) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[52:55], a[44:47], v124, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2319) 
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30777) least significant half
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30777) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[18:21], a[48:51], v125, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3549) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[44:47], a[52:55], v125, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3473) 
s_add_u32 m0, m0, 2048 // (op 30777) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[48:51], a[56:59], v125, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3397) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[52:55], a[60:63], v125, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3321) 
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30777) Load value direct to lds

v_add_co_u32 v126, s[64:65], v126, v14 // (op 30777) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[18:21], a[64:67], v132, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4551) 
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30777) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[44:47], a[68:71], v132, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4475) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[48:51], a[72:75], v132, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4399) 
s_add_u32 m0, m0, 2048 // (op 30777) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[52:55], a[76:79], v132, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4323) 
v_bfe_u32 v122, v114, 0, 8 // (op 28257) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[18:21], a[80:83], v133, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5553) 
v_bfe_u32 v123, v114, 8, 8 // (op 28257) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[44:47], a[84:87], v133, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5477) 
v_bfe_u32 v124, v114, 16, 8 // (op 28257) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[48:51], a[88:91], v133, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5401) 
v_bfe_u32 v125, v114, 24, 8 // (op 28257) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[52:55], a[92:95], v133, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5325) 
v_permlane16_swap_b32 v122, v123 // (op 28257) 
v_permlane16_swap_b32 v124, v125 // (op 28257) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[18:21], a[96:99], v134, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6555) 
s_nop 0
v_permlane32_swap_b32 v122, v124 // (op 28257) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[44:47], a[100:103], v134, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6479) 
v_permlane32_swap_b32 v123, v125 // (op 28257) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[48:51], a[104:107], v134, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6403) 
s_waitcnt vmcnt(35)
s_barrier  // (op 32168) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[52:55], a[108:111], v134, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6327) 
ds_read_b128 v[136:139], v6 offset:20480 // (op 216) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[18:21], a[112:115], v135, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7557) 
ds_read_b128 v[140:143], v6 offset:21504 // (op 432) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[44:47], a[116:119], v135, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7481) 
ds_read_b128 v[144:147], v6 offset:22528 // (op 658) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[48:51], a[120:123], v135, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7405) 
ds_read_b128 v[148:151], v6 offset:23552 // (op 884) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[52:55], a[124:127], v135, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7329) 

s_waitcnt vmcnt(29)
s_barrier  // (op 32162) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[88:91], v[18:21], a[128:131], v122, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8559) 
ds_read_b128 v[56:59], v5 offset:28672 // (op 214) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[88:91], v[44:47], a[132:135], v122, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8483) 
ds_read_b128 v[132:135], v5 offset:29696 // (op 1327) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[88:91], v[48:51], a[136:139], v122, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8407) 
ds_read_b128 v[152:155], v5 offset:30720 // (op 2329) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[88:91], v[52:55], a[140:143], v122, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8331) 
ds_read_b128 v[156:159], v5 offset:31744 // (op 3331) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:95], v[18:21], a[144:147], v123, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9561) 
ds_read_b128 v[160:163], v5 offset:32768 // (op 4333) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[92:95], v[44:47], a[148:151], v123, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9485) 
ds_read_b128 v[68:71], v5 offset:33792 // (op 5335) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[48:51], a[152:155], v123, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9409) 
ds_read_b128 v[72:75], v5 offset:34816 // (op 6337) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[52:55], a[156:159], v123, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9333) 
ds_read_b128 v[164:167], v5 offset:35840 // (op 7339) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:99], v[18:21], a[160:163], v124, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10563) 
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30777) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[96:99], v[44:47], a[164:167], v124, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10487) 
v_mov_b32 v14, v25 // (op 30695) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[96:99], v[48:51], a[168:171], v124, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10411) 
v_add_u32 v15, 16, v14 // (op 30695) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:99], v[52:55], a[172:175], v124, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10335) 
ds_read_b128 v[80:83], v5 offset:36864 // (op 8341) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[18:21], a[176:179], v125, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11565) 
ds_read_b128 v[168:171], v5 offset:37888 // (op 9343) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[44:47], a[180:183], v125, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11489) 
ds_read_b128 v[88:91], v5 offset:38912 // (op 10345) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[48:51], a[184:187], v125, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11413) 
buffer_load_dwordx4 v[28:31], v15, s[60:63], 0 offen // (op 30695) Load value


v_add_u32 v25, 16, v25 // (op 31535) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[52:55], a[188:191], v125, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11337) 
v_mov_b32 v14, v24 // (op 30708) 


s_lshl_b32 s19, s28, 6 // (op 30708) 
s_add_u32 s38, 16, s19 // (op 30708) 

v_add_u32 v15, v14, s38 // (op 30708) 


buffer_load_dwordx4 v[32:35], v15, s[56:59], 0 offen // (op 30708) Load value

ds_read_b128 v[122:125], v5 offset:39936 // (op 11347) Load local data 


s_lshl_b32 s19, s28, 7 // (op 30721) 
s_add_u32 s38, 16, s19 // (op 30721) 

v_add_u32 v15, v14, s38 // (op 30721) 


v_bfe_u32 v126, v119, 0, 8 // (op 28139) 
v_bfe_u32 v127, v119, 8, 8 // (op 28139) 
v_bfe_u32 v128, v119, 16, 8 // (op 28139) 
v_bfe_u32 v129, v119, 24, 8 // (op 28139) 
v_permlane16_swap_b32 v126, v127 // (op 28139) 
s_nop 0
v_permlane16_swap_b32 v128, v129 // (op 28139) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 28139) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 28139) 
buffer_load_dwordx4 v[36:39], v15, s[56:59], 0 offen // (op 30721) Load value



v_add_u32 v15, 16, v14 // (op 30734) 

v_bfe_u32 v172, v105, 0, 8 // (op 28532) 
v_bfe_u32 v173, v105, 8, 8 // (op 28532) 
v_bfe_u32 v174, v105, 16, 8 // (op 28532) 
v_bfe_u32 v175, v105, 24, 8 // (op 28532) 
v_permlane16_swap_b32 v172, v173 // (op 28532) 
s_nop 0
v_permlane16_swap_b32 v174, v175 // (op 28532) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v172, v174 // (op 28532) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v173, v175 // (op 28532) 
buffer_load_dwordx4 v[40:43], v15, s[56:59], 0 offen // (op 30734) Load value


s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[136:139], a[0:3], v126, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 221) 
s_barrier  // (op 32172) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[140:143], a[4:7], v126, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 422) 
v_add_u32 v24, 16, v24 // (op 31483) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[144:147], a[8:11], v126, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 648) 
s_lshl_b64 s[38:39], s[24:25], 5 // (op 30804) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[148:151], a[12:15], v126, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 874) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30804) 

v_lshrrev_b64 v[14:15], 3, s[64:65] // (op 30804) 


s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[136:139], a[16:19], v127, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1546) 
v_mov_b64 v[130:131], v[26:27] // (op 30804) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[140:143], a[20:23], v127, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1470) 
s_add_u32 s19, 20480, s7 // (op 30804) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[144:147], a[24:27], v127, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1394) 
s_mov_b32 m0, s19 // (op 30804) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[148:151], a[28:31], v127, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1318) 
s_mov_b32 s38, 320 // (op 30804) call()

buffer_load_dwordx4 v130, s[52:55], s38 offen offset: 0 lds // (op 30804) Load value direct to lds
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[152:155], v[136:139], a[32:35], v128, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2548) 
v_add_co_u32 v130, s[64:65], v130, v14 // (op 30804) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[152:155], v[140:143], a[36:39], v128, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2472) 
v_addc_co_u32 v131, s[64:65], 0, v131, s[64:65] // (op 30804) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[152:155], v[144:147], a[40:43], v128, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2396) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[152:155], v[148:151], a[44:47], v128, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2320) 
s_add_u32 m0, m0, 2048 // (op 30804) 

s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[136:139], a[48:51], v129, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3550) 
v_bfe_u32 v132, v109, 0, 8 // (op 28204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[140:143], a[52:55], v129, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3474) 
v_bfe_u32 v133, v109, 8, 8 // (op 28204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[144:147], a[56:59], v129, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3398) 
v_bfe_u32 v134, v109, 16, 8 // (op 28204) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[148:151], a[60:63], v129, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3322) 
v_bfe_u32 v135, v109, 24, 8 // (op 28204) 


v_permlane16_swap_b32 v132, v133 // (op 28204) 
s_nop 0
v_permlane16_swap_b32 v134, v135 // (op 28204) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v132, v134 // (op 28204) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v133, v135 // (op 28204) 
s_waitcnt vmcnt(31)
s_barrier  // (op 32182) 



s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[136:139], a[64:67], v132, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4552) 
ds_read_b128 v[18:21], v8 offset:40960 // (op 244) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[160:163], v[140:143], a[68:71], v132, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4476) 
ds_read_b128 v[44:47], v8 offset:41984 // (op 442) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[160:163], v[144:147], a[72:75], v132, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4400) 
ds_read_b128 v[48:51], v8 offset:43008 // (op 668) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[148:151], a[76:79], v132, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4324) 
v_bfe_u32 v126, v115, 0, 8 // (op 28272) 

v_bfe_u32 v127, v115, 8, 8 // (op 28272) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[136:139], a[80:83], v133, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5554) 
v_bfe_u32 v128, v115, 16, 8 // (op 28272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[68:71], v[140:143], a[84:87], v133, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5478) 
v_bfe_u32 v129, v115, 24, 8 // (op 28272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[68:71], v[144:147], a[88:91], v133, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5402) 
v_permlane16_swap_b32 v126, v127 // (op 28272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[68:71], v[148:151], a[92:95], v133, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5326) 
v_permlane16_swap_b32 v128, v129 // (op 28272) 
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 28272) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[72:75], v[136:139], a[96:99], v134, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6556) 
v_permlane32_swap_b32 v127, v129 // (op 28272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[140:143], a[100:103], v134, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6480) 
ds_read_b128 v[52:55], v8 offset:44032 // (op 894) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[72:75], v[144:147], a[104:107], v134, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6404) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:75], v[148:151], a[108:111], v134, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6328) 
s_waitcnt vmcnt(25)
s_barrier  // (op 32176) 



s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[164:167], v[136:139], a[112:115], v135, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7558) 
ds_read_b128 v[56:59], v7 offset:49152 // (op 242) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[164:167], v[140:143], a[116:119], v135, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7482) 
ds_read_b128 v[60:63], v7 offset:50176 // (op 1337) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[144:147], a[120:123], v135, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7406) 
ds_read_b128 v[64:67], v7 offset:51200 // (op 2339) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[148:151], a[124:127], v135, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7330) 
ds_read_b128 v[76:79], v7 offset:52224 // (op 3341) Load local data 





s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[80:83], v[136:139], a[128:131], v126, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8560) 
ds_read_b128 v[84:87], v7 offset:53248 // (op 4343) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[80:83], v[140:143], a[132:135], v126, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8484) 
ds_read_b128 v[92:95], v7 offset:54272 // (op 5345) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[80:83], v[144:147], a[136:139], v126, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8408) 
ds_read_b128 v[100:103], v7 offset:55296 // (op 6347) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[80:83], v[148:151], a[140:143], v126, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8332) 
ds_read_b128 v[132:135], v7 offset:56320 // (op 7349) Load local data 



s_waitcnt lgkmcnt(14)
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[168:171], v[136:139], a[144:147], v127, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9562) 
buffer_load_dwordx4 v130, s[52:55], s38 offen offset: 0 lds // (op 30804) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[168:171], v[140:143], a[148:151], v127, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9486) 
v_add_co_u32 v130, s[64:65], v130, v14 // (op 30804) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[168:171], v[144:147], a[152:155], v127, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9410) 
v_addc_co_u32 v131, s[64:65], 0, v131, s[64:65] // (op 30804) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[168:171], v[148:151], a[156:159], v127, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9334) 

s_add_u32 m0, m0, 2048 // (op 30804) 
s_waitcnt lgkmcnt(13)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[88:91], v[136:139], a[160:163], v128, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10564) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[88:91], v[140:143], a[164:167], v128, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10488) 
ds_read_b128 v[152:155], v7 offset:57344 // (op 8351) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:91], v[144:147], a[168:171], v128, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10412) 
buffer_load_dwordx4 v130, s[52:55], s38 offen offset: 0 lds // (op 30804) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[88:91], v[148:151], a[172:175], v128, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10336) 
v_add_co_u32 v130, s[64:65], v130, v14 // (op 30804) least significant half
v_addc_co_u32 v131, s[64:65], 0, v131, s[64:65] // (op 30804) most significant half

s_waitcnt lgkmcnt(13)
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[122:125], v[136:139], a[176:179], v129, v172 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11566) 

s_add_u32 m0, m0, 2048 // (op 30804) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[122:125], v[140:143], a[180:183], v129, v173 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11490) 

ds_read_b128 v[136:139], v7 offset:58368 // (op 9353) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[122:125], v[144:147], a[184:187], v129, v174 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11414) 
ds_read_b128 v[140:143], v7 offset:59392 // (op 10355) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[122:125], v[148:151], a[188:191], v129, v175 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11338) 
buffer_load_dwordx4 v130, s[52:55], s38 offen offset: 0 lds // (op 30804) Load value direct to lds








s_barrier  // (op 32165) 
ds_read_b128 v[122:125], v7 offset:60416 // (op 11357) Load local data 

s_lshl_b64 s[38:39], s[12:13], 5 // (op 30799) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30799) 

v_lshrrev_b64 v[14:15], 3, s[64:65] // (op 30799) 


v_mov_b64 v[126:127], v[112:113] // (op 30799) 
s_add_u32 s19, 28672, s6 // (op 30799) 
s_mov_b32 m0, s19 // (op 30799) reg expression
s_mov_b32 s38, 320 // (op 30799) call()
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30799) Load value direct to lds
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30799) least significant half
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30799) most significant half

s_add_u32 m0, m0, 2048 // (op 30799) 
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30799) Load value direct to lds
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30799) least significant half
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30799) most significant half

s_add_u32 m0, m0, 2048 // (op 30799) 
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30799) Load value direct to lds
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30799) least significant half
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30799) most significant half

s_add_u32 m0, m0, 2048 // (op 30799) 
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30799) Load value direct to lds
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30799) least significant half
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30799) most significant half

s_add_u32 m0, m0, 2048 // (op 30799) 
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30799) Load value direct to lds
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30799) least significant half
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30799) most significant half

s_add_u32 m0, m0, 2048 // (op 30799) 
buffer_load_dwordx4 v126, s[48:51], s38 offen offset: 0 lds // (op 30799) Load value direct to lds




v_bfe_u32 v126, v120, 0, 8 // (op 28156) 
v_bfe_u32 v127, v120, 8, 8 // (op 28156) 
v_bfe_u32 v128, v120, 16, 8 // (op 28156) 
v_bfe_u32 v129, v120, 24, 8 // (op 28156) 
v_permlane16_swap_b32 v126, v127 // (op 28156) 
s_nop 0
v_permlane16_swap_b32 v128, v129 // (op 28156) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 28156) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 28156) 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 32185) 

s_lshl_b64 s[38:39], s[24:25], 5 // (op 30814) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30814) 

v_lshrrev_b64 v[14:15], 3, s[64:65] // (op 30814) 


v_mov_b64 v[130:131], v[26:27] // (op 30814) 
s_add_u32 s19, 40960, s11 // (op 30814) 
s_mov_b32 m0, s19 // (op 30814) reg expression
s_mov_b32 s38, 384 // (op 30814) call()
v_bfe_u32 v144, v106, 0, 8 // (op 28550) 
v_bfe_u32 v145, v106, 8, 8 // (op 28550) 
v_bfe_u32 v146, v106, 16, 8 // (op 28550) 
v_bfe_u32 v147, v106, 24, 8 // (op 28550) 
v_permlane16_swap_b32 v144, v145 // (op 28550) 
s_nop 0
v_permlane16_swap_b32 v146, v147 // (op 28550) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v144, v146 // (op 28550) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v145, v147 // (op 28550) 
buffer_load_dwordx4 v130, s[52:55], s38 offen offset: 0 lds // (op 30814) Load value direct to lds
v_add_co_u32 v130, s[64:65], v130, v14 // (op 30814) least significant half
s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[18:21], a[0:3], v126, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 249) 
v_addc_co_u32 v131, s[64:65], 0, v131, s[64:65] // (op 30814) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[44:47], a[4:7], v126, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 423) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[48:51], a[8:11], v126, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 649) 
s_add_u32 m0, m0, 2048 // (op 30814) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[52:55], a[12:15], v126, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 875) 
v_bfe_u32 v148, v110, 0, 8 // (op 28222) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[18:21], a[16:19], v127, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1547) 
v_bfe_u32 v149, v110, 8, 8 // (op 28222) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[44:47], a[20:23], v127, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1471) 
v_bfe_u32 v150, v110, 16, 8 // (op 28222) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[48:51], a[24:27], v127, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1395) 
v_bfe_u32 v151, v110, 24, 8 // (op 28222) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[52:55], a[28:31], v127, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1319) 
v_permlane16_swap_b32 v148, v149 // (op 28222) 
v_permlane16_swap_b32 v150, v151 // (op 28222) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[18:21], a[32:35], v128, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2549) 
s_nop 0
v_permlane32_swap_b32 v148, v150 // (op 28222) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[44:47], a[36:39], v128, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2473) 
v_permlane32_swap_b32 v149, v151 // (op 28222) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[48:51], a[40:43], v128, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2397) 
buffer_load_dwordx4 v130, s[52:55], s38 offen offset: 0 lds // (op 30814) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[52:55], a[44:47], v128, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2321) 
v_add_co_u32 v130, s[64:65], v130, v14 // (op 30814) least significant half
v_addc_co_u32 v131, s[64:65], 0, v131, s[64:65] // (op 30814) most significant half

s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[18:21], a[48:51], v129, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3551) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[44:47], a[52:55], v129, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3475) 
s_add_u32 m0, m0, 2048 // (op 30814) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[48:51], a[56:59], v129, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3399) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[52:55], a[60:63], v129, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3323) 
buffer_load_dwordx4 v130, s[52:55], s38 offen offset: 0 lds // (op 30814) Load value direct to lds

v_add_co_u32 v130, s[64:65], v130, v14 // (op 30814) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[18:21], a[64:67], v148, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4553) 
v_addc_co_u32 v131, s[64:65], 0, v131, s[64:65] // (op 30814) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[44:47], a[68:71], v148, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4477) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[48:51], a[72:75], v148, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4401) 
s_add_u32 m0, m0, 2048 // (op 30814) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[52:55], a[76:79], v148, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4325) 
v_bfe_u32 v126, v116, 0, 8 // (op 28290) 
v_bfe_u32 v127, v116, 8, 8 // (op 28290) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[18:21], a[80:83], v149, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5555) 
v_bfe_u32 v128, v116, 16, 8 // (op 28290) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[44:47], a[84:87], v149, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5479) 
v_bfe_u32 v129, v116, 24, 8 // (op 28290) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[48:51], a[88:91], v149, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5403) 
v_permlane16_swap_b32 v126, v127 // (op 28290) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[52:55], a[92:95], v149, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5327) 
v_permlane16_swap_b32 v128, v129 // (op 28290) 
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 28290) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[100:103], v[18:21], a[96:99], v150, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6557) 
v_permlane32_swap_b32 v127, v129 // (op 28290) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[100:103], v[44:47], a[100:103], v150, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6481) 
s_waitcnt vmcnt(33)
s_barrier  // (op 32194) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[100:103], v[48:51], a[104:107], v150, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6405) 
ds_read_b128 v[156:159], v10 offset:61440 // (op 272) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[100:103], v[52:55], a[108:111], v150, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6329) 
ds_read_b128 v[160:163], v10 offset:62464 // (op 452) Load local data 



ds_read_b128 v[164:167], v10 offset:63488 // (op 678) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[18:21], a[112:115], v151, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7559) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[44:47], a[116:119], v151, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7483) 
ds_read_b128 v[168:171], v10 offset:64512 // (op 904) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[48:51], a[120:123], v151, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7407) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:135], v[52:55], a[124:127], v151, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7331) 
s_waitcnt vmcnt(27)
s_barrier  // (op 32188) 





s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:155], v[18:21], a[128:131], v126, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8561) 
v_add_u32 v132, 69632, v9 // (op 270) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:155], v[44:47], a[132:135], v126, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8485) 
ds_read_b128 v[56:59], v132 // (op 270) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:155], v[48:51], a[136:139], v126, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8409) 
v_add_u32 v132, 70656, v9 // (op 1347) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[52:55], a[140:143], v126, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8333) 
ds_read_b128 v[68:71], v132 // (op 1347) Load local data 





s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[136:139], v[18:21], a[144:147], v127, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9563) 
v_add_u32 v132, 71680, v9 // (op 2349) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[136:139], v[44:47], a[148:151], v127, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9487) 
ds_read_b128 v[72:75], v132 // (op 2349) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[136:139], v[48:51], a[152:155], v127, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9411) 
v_add_u32 v132, 72704, v9 // (op 3351) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[136:139], v[52:55], a[156:159], v127, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9335) 
ds_read_b128 v[80:83], v132 // (op 3351) Load local data 





s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[140:143], v[18:21], a[160:163], v128, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10565) 
v_add_u32 v132, 73728, v9 // (op 4353) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[140:143], v[44:47], a[164:167], v128, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10489) 
ds_read_b128 v[88:91], v132 // (op 4353) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[48:51], a[168:171], v128, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10413) 
v_add_u32 v132, 74752, v9 // (op 5355) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[52:55], a[172:175], v128, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10337) 
ds_read_b128 v[96:99], v132 // (op 5355) Load local data 





s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[122:125], v[18:21], a[176:179], v129, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11567) 
v_add_u32 v132, 75776, v9 // (op 6357) 
ds_read_b128 v[140:143], v132 // (op 6357) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[122:125], v[44:47], a[180:183], v129, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11491) 
v_add_u32 v132, 76800, v9 // (op 7359) 
ds_read_b128 v[92:95], v132 // (op 7359) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[122:125], v[48:51], a[184:187], v129, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11415) 
v_add_u32 v132, 77824, v9 // (op 8361) 
buffer_load_dwordx4 v130, s[52:55], s38 offen offset: 0 lds // (op 30814) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[122:125], v[52:55], a[188:191], v129, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11339) 
s_barrier  // (op 32179) 




s_lshl_b64 s[38:39], s[12:13], 5 // (op 30809) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30809) 

v_lshrrev_b64 v[14:15], 3, s[64:65] // (op 30809) 


v_mov_b64 v[122:123], v[112:113] // (op 30809) 
s_add_u32 s19, 49152, s10 // (op 30809) 
s_mov_b32 m0, s19 // (op 30809) reg expression
s_mov_b32 s38, 384 // (op 30809) call()
ds_read_b128 v[100:103], v132 // (op 8361) Load local data 

buffer_load_dwordx4 v122, s[48:51], s38 offen offset: 0 lds // (op 30809) Load value direct to lds



v_add_u32 v124, 78848, v9 // (op 9363) 
v_add_co_u32 v122, s[64:65], v122, v14 // (op 30809) least significant half
v_addc_co_u32 v123, s[64:65], 0, v123, s[64:65] // (op 30809) most significant half

s_add_u32 m0, m0, 2048 // (op 30809) 
ds_read_b128 v[136:139], v124 // (op 9363) Load local data 

buffer_load_dwordx4 v122, s[48:51], s38 offen offset: 0 lds // (op 30809) Load value direct to lds



v_add_u32 v124, 79872, v9 // (op 10365) 
ds_read_b128 v[132:135], v124 // (op 10365) Load local data 

v_add_co_u32 v122, s[64:65], v122, v14 // (op 30809) least significant half



v_addc_co_u32 v123, s[64:65], 0, v123, s[64:65] // (op 30809) most significant half

s_add_u32 m0, m0, 2048 // (op 30809) 
v_add_u32 v124, 80896, v9 // (op 11367) 
ds_read_b128 v[128:131], v124 // (op 11367) Load local data 

buffer_load_dwordx4 v122, s[48:51], s38 offen offset: 0 lds // (op 30809) Load value direct to lds
v_add_co_u32 v122, s[64:65], v122, v14 // (op 30809) least significant half
v_addc_co_u32 v123, s[64:65], 0, v123, s[64:65] // (op 30809) most significant half

s_add_u32 m0, m0, 2048 // (op 30809) 
buffer_load_dwordx4 v122, s[48:51], s38 offen offset: 0 lds // (op 30809) Load value direct to lds
v_add_co_u32 v122, s[64:65], v122, v14 // (op 30809) least significant half
v_addc_co_u32 v123, s[64:65], 0, v123, s[64:65] // (op 30809) most significant half

s_add_u32 m0, m0, 2048 // (op 30809) 
buffer_load_dwordx4 v122, s[48:51], s38 offen offset: 0 lds // (op 30809) Load value direct to lds
v_add_co_u32 v122, s[64:65], v122, v14 // (op 30809) least significant half
v_addc_co_u32 v123, s[64:65], 0, v123, s[64:65] // (op 30809) most significant half

s_add_u32 m0, m0, 2048 // (op 30809) 
buffer_load_dwordx4 v122, s[48:51], s38 offen offset: 0 lds // (op 30809) Load value direct to lds




v_bfe_u32 v122, v121, 0, 8 // (op 28173) 
v_bfe_u32 v123, v121, 8, 8 // (op 28173) 
v_bfe_u32 v124, v121, 16, 8 // (op 28173) 
v_bfe_u32 v125, v121, 24, 8 // (op 28173) 
v_permlane16_swap_b32 v122, v123 // (op 28173) 
s_nop 0
v_permlane16_swap_b32 v124, v125 // (op 28173) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v122, v124 // (op 28173) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v123, v125 // (op 28173) 
s_waitcnt lgkmcnt(12)
s_barrier  // (op 32197) 


s_lshl_b64 s[38:39], s[24:25], 5 // (op 30824) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30824) 

v_lshrrev_b64 v[14:15], 3, s[64:65] // (op 30824) 


v_mov_b64 v[126:127], v[26:27] // (op 30824) 
s_add_u32 s19, 61440, s15 // (op 30824) 
s_mov_b32 m0, s19 // (op 30824) reg expression
s_mov_b32 s38, 448 // (op 30824) call()
v_bfe_u32 v118, v107, 0, 8 // (op 28568) 
v_bfe_u32 v119, v107, 8, 8 // (op 28568) 
v_bfe_u32 v120, v107, 16, 8 // (op 28568) 
v_bfe_u32 v121, v107, 24, 8 // (op 28568) 
v_permlane16_swap_b32 v118, v119 // (op 28568) 
s_nop 0
v_permlane16_swap_b32 v120, v121 // (op 28568) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v118, v120 // (op 28568) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v119, v121 // (op 28568) 
buffer_load_dwordx4 v126, s[52:55], s38 offen offset: 0 lds // (op 30824) Load value direct to lds

v_add_co_u32 v126, s[64:65], v126, v14 // (op 30824) least significant half
s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[156:159], a[0:3], v122, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 277) 
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30824) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[160:163], a[4:7], v122, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 424) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[164:167], a[8:11], v122, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 650) 
s_add_u32 m0, m0, 2048 // (op 30824) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[168:171], a[12:15], v122, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 876) 
v_bfe_u32 v104, v111, 0, 8 // (op 28240) 
s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[68:71], v[156:159], a[16:19], v123, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1548) 
v_bfe_u32 v105, v111, 8, 8 // (op 28240) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[68:71], v[160:163], a[20:23], v123, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1472) 
v_bfe_u32 v106, v111, 16, 8 // (op 28240) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:71], v[164:167], a[24:27], v123, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1396) 
v_bfe_u32 v107, v111, 24, 8 // (op 28240) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[68:71], v[168:171], a[28:31], v123, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1320) 
v_permlane16_swap_b32 v104, v105 // (op 28240) 
v_permlane16_swap_b32 v106, v107 // (op 28240) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:75], v[156:159], a[32:35], v124, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2550) 
s_nop 0
v_permlane32_swap_b32 v104, v106 // (op 28240) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[72:75], v[160:163], a[36:39], v124, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2474) 
v_permlane32_swap_b32 v105, v107 // (op 28240) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[72:75], v[164:167], a[40:43], v124, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2398) 
buffer_load_dwordx4 v126, s[52:55], s38 offen offset: 0 lds // (op 30824) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:75], v[168:171], a[44:47], v124, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2322) 
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30824) least significant half
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30824) most significant half

s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[156:159], a[48:51], v125, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3552) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[160:163], a[52:55], v125, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3476) 
s_add_u32 m0, m0, 2048 // (op 30824) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[164:167], a[56:59], v125, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3400) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[168:171], a[60:63], v125, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3324) 
s_waitcnt vmcnt(32)
s_barrier  // (op 32155) 




s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[156:159], a[64:67], v104, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4554) 
ds_read_b128 v[18:21], v4 // (op 67) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[160:163], a[68:71], v104, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4478) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 418) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:91], v[164:167], a[72:75], v104, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4402) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 644) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[88:91], v[168:171], a[76:79], v104, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4326) 
v_bfe_u32 v108, v117, 0, 8 // (op 28308) 
v_bfe_u32 v109, v117, 8, 8 // (op 28308) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[96:99], v[156:159], a[80:83], v105, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5556) 
v_bfe_u32 v110, v117, 16, 8 // (op 28308) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[96:99], v[160:163], a[84:87], v105, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5480) 
v_bfe_u32 v111, v117, 24, 8 // (op 28308) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[96:99], v[164:167], a[88:91], v105, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5404) 
v_permlane16_swap_b32 v108, v109 // (op 28308) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[96:99], v[168:171], a[92:95], v105, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5328) 
v_permlane16_swap_b32 v110, v111 // (op 28308) 
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 28308) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[140:143], v[156:159], a[96:99], v106, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6558) 
v_permlane32_swap_b32 v109, v111 // (op 28308) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[140:143], v[160:163], a[100:103], v106, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6482) 
ds_read_b128 v[52:55], v4 offset:3072 // (op 870) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[164:167], a[104:107], v106, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6406) 
s_waitcnt vmcnt(26)
s_barrier  // (op 32139) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[140:143], v[168:171], a[108:111], v106, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6330) 
ds_read_b128 v[56:59], v3 offset:8192 // (op 62) Load local data 




s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:95], v[156:159], a[112:115], v107, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7560) 
ds_read_b128 v[60:63], v3 offset:9216 // (op 1313) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:95], v[160:163], a[116:119], v107, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7484) 
ds_read_b128 v[64:67], v3 offset:10240 // (op 2315) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[92:95], v[164:167], a[120:123], v107, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7408) 
ds_read_b128 v[68:71], v3 offset:11264 // (op 3317) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[92:95], v[168:171], a[124:127], v107, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7332) 
ds_read_b128 v[72:75], v3 offset:12288 // (op 4319) Load local data 




s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[100:103], v[156:159], a[128:131], v108, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8562) 
ds_read_b128 v[76:79], v3 offset:13312 // (op 5321) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[100:103], v[160:163], a[132:135], v108, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8486) 
ds_read_b128 v[80:83], v3 offset:14336 // (op 6323) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[100:103], v[164:167], a[136:139], v108, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8410) 
ds_read_b128 v[84:87], v3 offset:15360 // (op 7325) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[100:103], v[168:171], a[140:143], v108, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8334) 
buffer_load_dwordx4 v126, s[52:55], s38 offen offset: 0 lds // (op 30824) Load value direct to lds
v_add_co_u32 v126, s[64:65], v126, v14 // (op 30824) least significant half
s_waitcnt lgkmcnt(14)
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[136:139], v[156:159], a[144:147], v109, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9564) 
v_addc_co_u32 v127, s[64:65], 0, v127, s[64:65] // (op 30824) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[136:139], v[160:163], a[148:151], v109, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9488) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[136:139], v[164:167], a[152:155], v109, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9412) 
s_add_u32 m0, m0, 2048 // (op 30824) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[136:139], v[168:171], a[156:159], v109, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9336) 

ds_read_b128 v[88:91], v3 offset:16384 // (op 8327) Load local data 



s_waitcnt lgkmcnt(14)
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:135], v[156:159], a[160:163], v110, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10566) 
ds_read_b128 v[92:95], v3 offset:17408 // (op 9329) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[132:135], v[160:163], a[164:167], v110, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10490) 
buffer_load_dwordx4 v126, s[52:55], s38 offen offset: 0 lds // (op 30824) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[132:135], v[164:167], a[168:171], v110, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10414) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[132:135], v[168:171], a[172:175], v110, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10338) 
ds_read_b128 v[96:99], v3 offset:18432 // (op 10331) Load local data 




s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[156:159], a[176:179], v111, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11568) 
s_barrier  // (op 32191) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[160:163], a[180:183], v111, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11492) 
v_add_co_u32 v22, vcc, 256, v22 // (op 31010) least significant half

v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 31010) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[128:131], v[164:167], a[184:187], v111, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11416) 
s_lshl_b64 s[38:39], s[12:13], 5 // (op 30819) 

s_lshl_b64 s[64:65], s[38:39], 2 // (op 30819) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[128:131], v[168:171], a[188:191], v111, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11340) 
v_lshrrev_b64 v[14:15], 3, s[64:65] // (op 30819) 






v_mov_b64 v[26:27], v[112:113] // (op 30819) 
s_add_u32 s19, 69632, s14 // (op 30819) 
s_mov_b32 m0, s19 // (op 30819) reg expression
s_mov_b32 s38, 448 // (op 30819) call()
buffer_load_dwordx4 v26, s[48:51], s38 offen offset: 0 lds // (op 30819) Load value direct to lds
v_add_co_u32 v26, s[64:65], v26, v14 // (op 30819) least significant half
v_addc_co_u32 v27, s[64:65], 0, v27, s[64:65] // (op 30819) most significant half

s_add_u32 m0, m0, 2048 // (op 30819) 
ds_read_b128 v[100:103], v3 offset:19456 // (op 11333) Load local data 
buffer_load_dwordx4 v26, s[48:51], s38 offen offset: 0 lds // (op 30819) Load value direct to lds
v_add_co_u32 v26, s[64:65], v26, v14 // (op 30819) least significant half
v_addc_co_u32 v27, s[64:65], 0, v27, s[64:65] // (op 30819) most significant half

s_add_u32 m0, m0, 2048 // (op 30819) 
buffer_load_dwordx4 v26, s[48:51], s38 offen offset: 0 lds // (op 30819) Load value direct to lds
v_add_co_u32 v26, s[64:65], v26, v14 // (op 30819) least significant half
v_addc_co_u32 v27, s[64:65], 0, v27, s[64:65] // (op 30819) most significant half

s_add_u32 m0, m0, 2048 // (op 30819) 
buffer_load_dwordx4 v26, s[48:51], s38 offen offset: 0 lds // (op 30819) Load value direct to lds
v_add_co_u32 v26, s[64:65], v26, v14 // (op 30819) least significant half
v_addc_co_u32 v27, s[64:65], 0, v27, s[64:65] // (op 30819) most significant half

s_add_u32 m0, m0, 2048 // (op 30819) 
buffer_load_dwordx4 v26, s[48:51], s38 offen offset: 0 lds // (op 30819) Load value direct to lds
v_add_co_u32 v26, s[64:65], v26, v14 // (op 30819) least significant half
v_addc_co_u32 v27, s[64:65], 0, v27, s[64:65] // (op 30819) most significant half

s_add_u32 m0, m0, 2048 // (op 30819) 
buffer_load_dwordx4 v26, s[48:51], s38 offen offset: 0 lds // (op 30819) Load value direct to lds





v_add_co_u32 v16, vcc, 256, v16 // (op 30946) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 30946) most significant half
s_ashr_i64 s[38:39], s[36:37], 7 // (op 11276) 
s_mov_b32 s19, s38 // (op 11276) convert

s_and_b32 s38, s19, 4294967292 // (op 11276) 

s_cmp_lt_u32 s18, s38 // (op 11276) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopTop_KLoop_11276 // (op 11276) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopTop_KLoop_11276 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopBottom_KLoop_11276:
















s_barrier  // (op 32346) 
v_lshrrev_b32 v14, 2, v1 // (op 30862) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 30862) 
v_mov_b32 v15, 192 // (op 30862) 
s_lshr_b32 s6, s0, 2 // (op 30862) 

v_mul_lo_u32 v16, s16, v15 // (op 30862) 

v_add_u32 v16, v16, v14 // (op 30862) 

v_and_b32 v14, -4, v1 // (op 30862) 
v_sub_u32 v15, v1, v14 // (op 30862) 

v_lshlrev_b32 v14, 5, v15 // (op 30862) 

v_lshl_add_u32 v15, s6, 9, v14 // (op 30862) 


v_mov_b32 v27, 0 // (op 30862) convert
v_mov_b32 v26, v15 // (op 30862) convert

v_mul_lo_u32 v15, v16, s13 // (op 30862) most significant: low of low * high
v_mul_hi_u32 v17, v16, s12 // (op 30862) most significant: high of low * low
v_mul_lo_u32 v14, v16, s12 // (op 30862) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 30862) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 30862) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 30862) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 30862) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 30862) 

s_mul_i32 s1, s12, s23 // (op 30864) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 30864) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 30864) least significant
s_mul_hi_u32 s7, s12, s22 // (op 30864) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30864) most significant: sum
s_add_u32 s1, s1, s7 // (op 30864) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30864) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30864) 

s_mov_b32 s48, s20 // (op 30864) reg expression
s_mov_b32 s49, s21 // (op 30864) reg expression
s_mov_b32 s50, s0 // (op 30864) reg expression
s_mov_b32 s51, 131072 // (op 30864) call()

v_and_b32 v16, -4, v1 // (op 30976) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 30976) 
v_sub_u32 v17, v1, v16 // (op 30976) 

s_lshr_b32 s6, s0, 2 // (op 30976) 

v_lshlrev_b32 v16, 5, v17 // (op 30976) 

v_lshl_add_u32 v17, s6, 9, v16 // (op 30976) 


v_mov_b32 v19, 0 // (op 30976) convert
v_mov_b32 v18, v17 // (op 30976) convert

v_lshrrev_b32 v16, 2, v1 // (op 30976) 
v_lshl_add_u32 v17, s17, 7, v16 // (op 30976) 

v_mul_lo_u32 v21, v17, s25 // (op 30976) most significant: low of low * high
v_mul_hi_u32 v16, v17, s24 // (op 30976) most significant: high of low * low
v_mul_lo_u32 v20, v17, s24 // (op 30976) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 30976) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 30976) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 30976) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 30976) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 30976) 

s_mul_i32 s1, s24, s9 // (op 30978) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 30978) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 30978) least significant
s_mul_hi_u32 s7, s24, s8 // (op 30978) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30978) most significant: sum
s_add_u32 s1, s1, s7 // (op 30978) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30978) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30978) 

s_mov_b32 s52, s26 // (op 30978) reg expression
s_mov_b32 s53, s27 // (op 30978) reg expression
s_mov_b32 s54, s0 // (op 30978) reg expression
s_mov_b32 s55, 131072 // (op 30978) call()

v_and_b32 v16, -4, v1 // (op 31074) 
v_sub_u32 v17, v1, v16 // (op 31074) 

v_lshlrev_b32 v16, 5, v17 // (op 31074) 

v_lshrrev_b32 v17, 2, v1 // (op 31074) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 31074) 


v_lshlrev_b32 v16, 2, v20 // (op 31074) 

v_lshrrev_b32 v17, 3, v16 // (op 31074) 

s_nop 0
v_readfirstlane_b32 s0, v17 // (op 31074) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 31131) 
v_sub_u32 v17, v1, v16 // (op 31131) 

v_lshlrev_b32 v16, 5, v17 // (op 31131) 

v_lshrrev_b32 v17, 2, v1 // (op 31131) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 31131) 


v_lshlrev_b32 v16, 2, v20 // (op 31131) 

v_lshrrev_b32 v17, 3, v16 // (op 31131) 

s_nop 0
v_readfirstlane_b32 s1, v17 // (op 31131) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, 63, v1 // (op 31572) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 31572) 
v_lshrrev_b32 v17, 4, v16 // (op 31572) 
s_lshr_b32 s10, s6, 2 // (op 31572) 

v_and_b32 v20, 3, v17 // (op 31572) 

v_and_b32 v17, 15, v1 // (op 31572) 
v_lshl_add_u32 v21, v20, 4, v17 // (op 31572) 


v_lshrrev_b32 v17, 7, v1 // (op 31572) 
v_mul_lo_u32 v20, v17, 3 // (op 31572) 

v_lshl_add_u32 v17, v20, 6, v21 // (op 31572) 


v_mov_b32 v20, 192 // (op 31572) 
v_mul_lo_u32 v21, s16, v20 // (op 31572) 

v_add_u32 v21, v21, v17 // (op 31572) 

v_lshrrev_b32 v17, 6, v16 // (op 31572) 

v_lshlrev_b32 v16, 2, v17 // (op 31572) 

v_lshl_add_u32 v17, s10, 4, v16 // (op 31572) 


v_mul_lo_u32 v16, v21, s28 // (op 31572) 
v_add_u32 v16, v16, v17 // (op 31572) 


s_mul_i32 s7, s28, s23 // (op 31574) most significant: low of low * high
s_mul_i32 s10, s29, s22 // (op 31574) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 31574) least significant
s_mul_hi_u32 s11, s28, s22 // (op 31574) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 31574) most significant: sum
s_add_u32 s7, s7, s11 // (op 31574) most significant: sum


s_mov_b32 s56, s30 // (op 31574) reg expression
s_mov_b32 s57, s31 // (op 31574) reg expression
s_mov_b32 s58, s6 // (op 31574) reg expression
s_mov_b32 s59, 131072 // (op 31574) call()

v_and_b32 v17, 63, v1 // (op 31599) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 31599) 
v_lshrrev_b32 v20, 6, v17 // (op 31599) 
s_lshr_b32 s10, s6, 2 // (op 31599) 

v_lshlrev_b32 v21, 2, v20 // (op 31599) 

v_lshl_add_u32 v20, s10, 4, v21 // (op 31599) 


v_lshrrev_b32 v21, 4, v17 // (op 31599) 

v_and_b32 v17, 3, v21 // (op 31599) 

v_and_b32 v21, 15, v1 // (op 31599) 
v_lshl_add_u32 v22, v17, 4, v21 // (op 31599) 


v_lshrrev_b32 v17, 6, v1 // (op 31599) 
v_and_b32 v21, 1, v17 // (op 31599) 

v_lshl_add_u32 v17, v21, 6, v22 // (op 31599) 


v_lshl_add_u32 v21, s17, 7, v17 // (op 31599) 

v_mul_lo_u32 v17, v21, s32 // (op 31599) 

v_add_u32 v21, v20, v17 // (op 31599) 


s_mul_i32 s7, s32, s9 // (op 31601) most significant: low of low * high
s_mul_i32 s10, s33, s8 // (op 31601) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 31601) least significant
s_mul_hi_u32 s11, s32, s8 // (op 31601) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 31601) most significant: sum
s_add_u32 s7, s7, s11 // (op 31601) most significant: sum


s_mov_b32 s60, s34 // (op 31601) reg expression
s_mov_b32 s61, s35 // (op 31601) reg expression
s_mov_b32 s62, s6 // (op 31601) reg expression
s_mov_b32 s63, 131072 // (op 31601) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 11282) 
s_and_b32 s10, s6, 4294967292 // (op 11282) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 11281) 
s_mov_b32 s11, s6 // (op 11281) convert

s_cmp_lt_u32 s10, s11 // (op 11281) 




















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopBottom_KLoopTail_11281 // (op 11281) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopBottom_KLoopTail_11281 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopTop_KLoopTail_11281:
s_barrier  // (op 32136) 
s_add_u32 s10, s10, 1 // (op 11283) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 30787) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 30787) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 30787) 


v_mov_b64 v[24:25], v[14:15] // (op 30787) 
v_mov_b64 v[26:27], v[24:25] // (op 30787) 
s_add_u32 s11, 8192, s0 // (op 30787) 
s_mov_b32 m0, s11 // (op 30787) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30787) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30787) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30787) most significant half

s_add_u32 m0, m0, 2048 // (op 30787) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30787) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30787) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30787) most significant half

s_add_u32 m0, m0, 2048 // (op 30787) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30787) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30787) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30787) most significant half

s_add_u32 m0, m0, 2048 // (op 30787) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30787) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30787) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30787) most significant half

s_add_u32 m0, m0, 2048 // (op 30787) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30787) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30787) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30787) most significant half

s_add_u32 m0, m0, 2048 // (op 30787) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30787) Load value direct to lds




v_add_co_u32 v14, s[6:7], 64, v14 // (op 30852) least significant half
v_addc_co_u32 v15, s[6:7], 0, v15, s[6:7] // (op 30852) most significant half

s_barrier  // (op 32152) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 30793) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 30793) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 30793) 


v_mov_b64 v[24:25], v[18:19] // (op 30793) 
v_mov_b64 v[26:27], v[24:25] // (op 30793) 
s_mov_b32 m0, s1 // (op 30793) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30793) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30793) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30793) most significant half

s_add_u32 m0, m0, 2048 // (op 30793) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30793) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30793) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30793) most significant half

s_add_u32 m0, m0, 2048 // (op 30793) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30793) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30793) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30793) most significant half

s_add_u32 m0, m0, 2048 // (op 30793) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30793) Load value direct to lds



v_add_co_u32 v18, s[6:7], 64, v18 // (op 30968) least significant half
v_addc_co_u32 v19, s[6:7], 0, v19, s[6:7] // (op 30968) most significant half

s_waitcnt vmcnt(4)
s_barrier  // (op 32142) 



ds_read_b128 v[22:25], v3 offset:8192 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 32158) 



ds_read_b128 v[26:29], v4 // (op 166) Load local data 


v_mov_b32 v17, v16 // (op 168) 
v_mov_b32 v20, v17 // (op 168) reg expression

buffer_load_dwordx4 v[30:33], v20, s[56:59], 0 offen // (op 168) Load value



v_mov_b32 v20, v21 // (op 170) 
v_mov_b32 v34, v20 // (op 170) reg expression

buffer_load_dwordx4 v[36:39], v34, s[60:63], 0 offen // (op 170) Load value


v_add_u32 v21, 4, v21 // (op 31589) 
s_waitcnt vmcnt(1)
v_bfe_u32 v40, v30, 0, 8 // (op 28777) 



v_bfe_u32 v41, v30, 8, 8 // (op 28777) 
v_bfe_u32 v42, v30, 16, 8 // (op 28777) 
v_bfe_u32 v43, v30, 24, 8 // (op 28777) 
v_permlane16_swap_b32 v40, v41 // (op 28777) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 401) Load local data 
v_permlane16_swap_b32 v42, v43 // (op 28777) 



s_nop 1
v_permlane32_swap_b32 v40, v42 // (op 28777) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v41, v43 // (op 28777) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 627) Load local data 




ds_read_b128 v[30:33], v4 offset:3072 // (op 853) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v52, v36, 0, 8 // (op 28867) 



v_bfe_u32 v53, v36, 8, 8 // (op 28867) 
v_bfe_u32 v54, v36, 16, 8 // (op 28867) 
v_bfe_u32 v55, v36, 24, 8 // (op 28867) 
v_permlane16_swap_b32 v52, v53 // (op 28867) 
s_nop 0
v_permlane16_swap_b32 v54, v55 // (op 28867) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 28867) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 28867) 
ds_read_b128 v[56:59], v3 offset:9216 // (op 1213) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 171) 
ds_read_b128 v[34:37], v3 offset:10240 // (op 2215) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[44:47], a[4:7], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 404) 
ds_read_b128 v[60:63], v3 offset:11264 // (op 3217) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[48:51], a[8:11], v40, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 630) 
ds_read_b128 v[64:67], v3 offset:12288 // (op 4219) Load local data 


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:25], v[30:33], a[12:15], v40, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 856) 
s_lshl_b32 s11, s28, 6 // (op 4228) 

v_add_u32 v20, v17, s11 // (op 4228) 


buffer_load_dwordx4 v[22:25], v20, s[56:59], 0 offen // (op 4228) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[26:29], a[16:19], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1217) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v41, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1271) 
s_waitcnt vmcnt(0)
v_bfe_u32 v68, v22, 0, 8 // (op 28791) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[30:33], a[28:31], v41, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1298) 
v_bfe_u32 v69, v22, 8, 8 // (op 28791) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[44:47], a[20:23], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1244) 
v_bfe_u32 v70, v22, 16, 8 // (op 28791) 

v_bfe_u32 v71, v22, 24, 8 // (op 28791) 
v_permlane16_swap_b32 v68, v69 // (op 28791) 
s_nop 0
v_permlane16_swap_b32 v70, v71 // (op 28791) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[34:37], v[26:29], a[32:35], v42, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2219) 
ds_read_b128 v[56:59], v3 offset:13312 // (op 5221) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[34:37], v[48:51], a[40:43], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2273) 
v_permlane32_swap_b32 v68, v70 // (op 28791) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[34:37], v[30:33], a[44:47], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2300) 
v_permlane32_swap_b32 v69, v71 // (op 28791) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[34:37], v[44:47], a[36:39], v42, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2246) 
ds_read_b128 v[22:25], v3 offset:14336 // (op 6223) Load local data 




ds_read_b128 v[34:37], v3 offset:15360 // (op 7225) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:63], v[26:29], a[48:51], v43, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3221) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[48:51], a[56:59], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3275) 
ds_read_b128 v[72:75], v3 offset:16384 // (op 8227) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[60:63], v[30:33], a[60:63], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3302) 
s_lshl_b32 s11, s28, 7 // (op 8236) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:63], v[44:47], a[52:55], v43, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3248) 
v_add_u32 v20, v17, s11 // (op 8236) 




buffer_load_dwordx4 v[60:63], v20, s[56:59], 0 offen // (op 8236) Load value


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[64:67], v[44:47], a[68:71], v68, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4250) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[64:67], v[26:29], a[64:67], v68, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4223) 
v_add_u32 v16, 4, v16 // (op 31562) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[64:67], v[48:51], a[72:75], v68, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4277) 
ds_read_b128 v[38:41], v3 offset:17408 // (op 9229) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[64:67], v[30:33], a[76:79], v68, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4304) 
ds_read_b128 v[76:79], v3 offset:18432 // (op 10231) Load local data 




s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[56:59], v[44:47], a[84:87], v69, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5252) 
s_waitcnt vmcnt(0)
v_bfe_u32 v64, v60, 0, 8 // (op 28805) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[56:59], v[26:29], a[80:83], v69, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5225) 
v_bfe_u32 v65, v60, 8, 8 // (op 28805) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[56:59], v[48:51], a[88:91], v69, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5279) 
v_bfe_u32 v66, v60, 16, 8 // (op 28805) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[56:59], v[30:33], a[92:95], v69, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5306) 
v_bfe_u32 v67, v60, 24, 8 // (op 28805) 

v_permlane16_swap_b32 v64, v65 // (op 28805) 
s_nop 0
v_permlane16_swap_b32 v66, v67 // (op 28805) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[22:25], v[44:47], a[100:103], v70, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6254) 
s_nop 0
v_permlane32_swap_b32 v64, v66 // (op 28805) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[22:25], v[26:29], a[96:99], v70, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6227) 
v_permlane32_swap_b32 v65, v67 // (op 28805) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[22:25], v[48:51], a[104:107], v70, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6281) 
ds_read_b128 v[56:59], v3 offset:19456 // (op 11233) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[22:25], v[30:33], a[108:111], v70, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6308) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[34:37], v[44:47], a[116:119], v71, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7256) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[34:37], v[26:29], a[112:115], v71, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7229) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[34:37], v[48:51], a[120:123], v71, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7283) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:37], v[30:33], a[124:127], v71, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7310) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[72:75], v[44:47], a[132:135], v64, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8258) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[72:75], v[26:29], a[128:131], v64, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8231) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[72:75], v[48:51], a[136:139], v64, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8285) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[72:75], v[30:33], a[140:143], v64, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8312) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[38:41], v[44:47], a[148:151], v65, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9260) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[38:41], v[26:29], a[144:147], v65, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9233) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[38:41], v[48:51], a[152:155], v65, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9287) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[38:41], v[30:33], a[156:159], v65, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9314) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[76:79], v[44:47], a[164:167], v66, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10262) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[76:79], v[26:29], a[160:163], v66, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10235) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[76:79], v[48:51], a[168:171], v66, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10289) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[76:79], v[30:33], a[172:175], v66, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10316) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[56:59], v[44:47], a[180:183], v67, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11264) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[56:59], v[26:29], a[176:179], v67, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11237) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[56:59], v[48:51], a[184:187], v67, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11291) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[56:59], v[30:33], a[188:191], v67, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11318) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 11281) 
s_mov_b32 s11, s6 // (op 11281) convert

s_cmp_lt_u32 s10, s11 // (op 11281) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopTop_KLoopTail_11281 // (op 11281) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopTop_KLoopTail_11281 if true)
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ForLoopBottom_KLoopTail_11281:





















s_barrier  // (op 32350) 









v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ConditionalFalse_0_beta_116
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v13, v4 // (op 42) 
v_mul_f32 v15, v13, v5 // (op 42) 
v_mul_f32 v16, v13, v6 // (op 42) 
v_mul_f32 v17, v13, v7 // (op 42) 

v_cvt_f16_f32 v4, v14 // (op 48) 
v_cvt_f16_f32 v5, v15 // (op 48) 
v_cvt_f16_f32 v6, v16 // (op 48) 
v_cvt_f16_f32 v7, v17 // (op 48) 


v_pack_b32_f16 v8, v4, v5 // (op 50) 
v_pack_b32_f16 v9, v6, v7 // (op 50) 

v_mov_b32 v3, v12 // (op 50) reg expression

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 50) Store value



v_accvgpr_read v4, a4 // (op 13706) 
v_accvgpr_read v5, a5 // (op 13706) 
v_accvgpr_read v6, a6 // (op 13706) 
v_accvgpr_read v7, a7 // (op 13706) 
v_mul_f32 v14, v13, v4 // (op 13706) 
v_mul_f32 v15, v13, v5 // (op 13706) 
v_mul_f32 v16, v13, v6 // (op 13706) 
v_mul_f32 v17, v13, v7 // (op 13706) 

v_cvt_f16_f32 v4, v14 // (op 13708) 
v_cvt_f16_f32 v5, v15 // (op 13708) 
v_cvt_f16_f32 v6, v16 // (op 13708) 
v_cvt_f16_f32 v7, v17 // (op 13708) 


v_pack_b32_f16 v8, v4, v5 // (op 13709) 
v_pack_b32_f16 v9, v6, v7 // (op 13709) 

s_lshl_b32 s0, s4, 5 // (op 13709) 
v_add_u32 v3, v12, s0 // (op 13709) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13709) Store value



v_accvgpr_read v4, a8 // (op 13720) 
v_accvgpr_read v5, a9 // (op 13720) 
v_accvgpr_read v6, a10 // (op 13720) 
v_accvgpr_read v7, a11 // (op 13720) 
v_mul_f32 v14, v13, v4 // (op 13720) 
v_mul_f32 v15, v13, v5 // (op 13720) 
v_mul_f32 v16, v13, v6 // (op 13720) 
v_mul_f32 v17, v13, v7 // (op 13720) 

v_cvt_f16_f32 v4, v14 // (op 13722) 
v_cvt_f16_f32 v5, v15 // (op 13722) 
v_cvt_f16_f32 v6, v16 // (op 13722) 
v_cvt_f16_f32 v7, v17 // (op 13722) 


v_pack_b32_f16 v8, v4, v5 // (op 13723) 
v_pack_b32_f16 v9, v6, v7 // (op 13723) 

s_lshl_b32 s0, s4, 6 // (op 13723) 
v_add_u32 v3, v12, s0 // (op 13723) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13723) Store value



v_accvgpr_read v4, a12 // (op 13734) 
v_accvgpr_read v5, a13 // (op 13734) 
v_accvgpr_read v6, a14 // (op 13734) 
v_accvgpr_read v7, a15 // (op 13734) 
v_mul_f32 v14, v13, v4 // (op 13734) 
v_mul_f32 v15, v13, v5 // (op 13734) 
v_mul_f32 v16, v13, v6 // (op 13734) 
v_mul_f32 v17, v13, v7 // (op 13734) 

v_cvt_f16_f32 v4, v14 // (op 13736) 
v_cvt_f16_f32 v5, v15 // (op 13736) 
v_cvt_f16_f32 v6, v16 // (op 13736) 
v_cvt_f16_f32 v7, v17 // (op 13736) 


v_pack_b32_f16 v8, v4, v5 // (op 13737) 
v_pack_b32_f16 v9, v6, v7 // (op 13737) 

s_lshl_b32 s0, s4, 5 // (op 13737) 
s_mul_i32 s1, 3, s0 // (op 13737) 

v_add_u32 v3, v12, s1 // (op 13737) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13737) Store value



v_accvgpr_read v4, a16 // (op 13770) 
v_accvgpr_read v5, a17 // (op 13770) 
v_accvgpr_read v6, a18 // (op 13770) 
v_accvgpr_read v7, a19 // (op 13770) 
v_mul_f32 v14, v13, v4 // (op 13770) 
v_mul_f32 v15, v13, v5 // (op 13770) 
v_mul_f32 v16, v13, v6 // (op 13770) 
v_mul_f32 v17, v13, v7 // (op 13770) 

v_cvt_f16_f32 v4, v14 // (op 13772) 
v_cvt_f16_f32 v5, v15 // (op 13772) 
v_cvt_f16_f32 v6, v16 // (op 13772) 
v_cvt_f16_f32 v7, v17 // (op 13772) 


v_pack_b32_f16 v8, v4, v5 // (op 13774) 
v_pack_b32_f16 v9, v6, v7 // (op 13774) 

v_add_u32 v3, 32, v12 // (op 13774) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13774) Store value



v_accvgpr_read v4, a20 // (op 13784) 
v_accvgpr_read v5, a21 // (op 13784) 
v_accvgpr_read v6, a22 // (op 13784) 
v_accvgpr_read v7, a23 // (op 13784) 
v_mul_f32 v14, v13, v4 // (op 13784) 
v_mul_f32 v15, v13, v5 // (op 13784) 
v_mul_f32 v16, v13, v6 // (op 13784) 
v_mul_f32 v17, v13, v7 // (op 13784) 

v_cvt_f16_f32 v4, v14 // (op 13786) 
v_cvt_f16_f32 v5, v15 // (op 13786) 
v_cvt_f16_f32 v6, v16 // (op 13786) 
v_cvt_f16_f32 v7, v17 // (op 13786) 


v_pack_b32_f16 v8, v4, v5 // (op 13776) 
v_pack_b32_f16 v9, v6, v7 // (op 13776) 

s_lshl_b32 s0, s4, 5 // (op 13776) 
s_add_u32 s0, s0, 32 // (op 13776) 
v_add_u32 v3, v12, s0 // (op 13776) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13776) Store value



v_accvgpr_read v4, a24 // (op 13790) 
v_accvgpr_read v5, a25 // (op 13790) 
v_accvgpr_read v6, a26 // (op 13790) 
v_accvgpr_read v7, a27 // (op 13790) 
v_mul_f32 v14, v13, v4 // (op 13790) 
v_mul_f32 v15, v13, v5 // (op 13790) 
v_mul_f32 v16, v13, v6 // (op 13790) 
v_mul_f32 v17, v13, v7 // (op 13790) 

v_cvt_f16_f32 v4, v14 // (op 13792) 
v_cvt_f16_f32 v5, v15 // (op 13792) 
v_cvt_f16_f32 v6, v16 // (op 13792) 
v_cvt_f16_f32 v7, v17 // (op 13792) 


v_pack_b32_f16 v8, v4, v5 // (op 13778) 
v_pack_b32_f16 v9, v6, v7 // (op 13778) 

s_lshl_b32 s0, s4, 6 // (op 13778) 
s_add_u32 s0, s0, 32 // (op 13778) 
v_add_u32 v3, v12, s0 // (op 13778) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13778) Store value



v_accvgpr_read v4, a28 // (op 13796) 
v_accvgpr_read v5, a29 // (op 13796) 
v_accvgpr_read v6, a30 // (op 13796) 
v_accvgpr_read v7, a31 // (op 13796) 
v_mul_f32 v14, v13, v4 // (op 13796) 
v_mul_f32 v15, v13, v5 // (op 13796) 
v_mul_f32 v16, v13, v6 // (op 13796) 
v_mul_f32 v17, v13, v7 // (op 13796) 

v_cvt_f16_f32 v4, v14 // (op 13798) 
v_cvt_f16_f32 v5, v15 // (op 13798) 
v_cvt_f16_f32 v6, v16 // (op 13798) 
v_cvt_f16_f32 v7, v17 // (op 13798) 


v_pack_b32_f16 v8, v4, v5 // (op 13780) 
v_pack_b32_f16 v9, v6, v7 // (op 13780) 

s_lshl_b32 s0, s4, 5 // (op 13780) 
s_mul_i32 s1, 3, s0 // (op 13780) 
s_add_u32 s1, s1, 32 // (op 13780) 

v_add_u32 v3, v12, s1 // (op 13780) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13780) Store value



v_accvgpr_read v4, a32 // (op 13846) 
v_accvgpr_read v5, a33 // (op 13846) 
v_accvgpr_read v6, a34 // (op 13846) 
v_accvgpr_read v7, a35 // (op 13846) 
v_mul_f32 v14, v13, v4 // (op 13846) 
v_mul_f32 v15, v13, v5 // (op 13846) 
v_mul_f32 v16, v13, v6 // (op 13846) 
v_mul_f32 v17, v13, v7 // (op 13846) 

v_cvt_f16_f32 v4, v14 // (op 13848) 
v_cvt_f16_f32 v5, v15 // (op 13848) 
v_cvt_f16_f32 v6, v16 // (op 13848) 
v_cvt_f16_f32 v7, v17 // (op 13848) 


v_pack_b32_f16 v8, v4, v5 // (op 13850) 
v_pack_b32_f16 v9, v6, v7 // (op 13850) 

v_add_u32 v3, 64, v12 // (op 13850) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13850) Store value



v_accvgpr_read v4, a36 // (op 13860) 
v_accvgpr_read v5, a37 // (op 13860) 
v_accvgpr_read v6, a38 // (op 13860) 
v_accvgpr_read v7, a39 // (op 13860) 
v_mul_f32 v14, v13, v4 // (op 13860) 
v_mul_f32 v15, v13, v5 // (op 13860) 
v_mul_f32 v16, v13, v6 // (op 13860) 
v_mul_f32 v17, v13, v7 // (op 13860) 

v_cvt_f16_f32 v4, v14 // (op 13862) 
v_cvt_f16_f32 v5, v15 // (op 13862) 
v_cvt_f16_f32 v6, v16 // (op 13862) 
v_cvt_f16_f32 v7, v17 // (op 13862) 


v_pack_b32_f16 v8, v4, v5 // (op 13852) 
v_pack_b32_f16 v9, v6, v7 // (op 13852) 

s_lshl_b32 s0, s4, 5 // (op 13852) 
s_add_u32 s0, s0, 64 // (op 13852) 
v_add_u32 v3, v12, s0 // (op 13852) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13852) Store value



v_accvgpr_read v4, a40 // (op 13866) 
v_accvgpr_read v5, a41 // (op 13866) 
v_accvgpr_read v6, a42 // (op 13866) 
v_accvgpr_read v7, a43 // (op 13866) 
v_mul_f32 v14, v13, v4 // (op 13866) 
v_mul_f32 v15, v13, v5 // (op 13866) 
v_mul_f32 v16, v13, v6 // (op 13866) 
v_mul_f32 v17, v13, v7 // (op 13866) 

v_cvt_f16_f32 v4, v14 // (op 13868) 
v_cvt_f16_f32 v5, v15 // (op 13868) 
v_cvt_f16_f32 v6, v16 // (op 13868) 
v_cvt_f16_f32 v7, v17 // (op 13868) 


v_pack_b32_f16 v8, v4, v5 // (op 13854) 
v_pack_b32_f16 v9, v6, v7 // (op 13854) 

s_lshl_b32 s0, s4, 6 // (op 13854) 
s_add_u32 s0, s0, 64 // (op 13854) 
v_add_u32 v3, v12, s0 // (op 13854) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13854) Store value



v_accvgpr_read v4, a44 // (op 13872) 
v_accvgpr_read v5, a45 // (op 13872) 
v_accvgpr_read v6, a46 // (op 13872) 
v_accvgpr_read v7, a47 // (op 13872) 
v_mul_f32 v14, v13, v4 // (op 13872) 
v_mul_f32 v15, v13, v5 // (op 13872) 
v_mul_f32 v16, v13, v6 // (op 13872) 
v_mul_f32 v17, v13, v7 // (op 13872) 

v_cvt_f16_f32 v4, v14 // (op 13874) 
v_cvt_f16_f32 v5, v15 // (op 13874) 
v_cvt_f16_f32 v6, v16 // (op 13874) 
v_cvt_f16_f32 v7, v17 // (op 13874) 


v_pack_b32_f16 v8, v4, v5 // (op 13856) 
v_pack_b32_f16 v9, v6, v7 // (op 13856) 

s_lshl_b32 s0, s4, 5 // (op 13856) 
s_mul_i32 s1, 3, s0 // (op 13856) 
s_add_u32 s1, s1, 64 // (op 13856) 

v_add_u32 v3, v12, s1 // (op 13856) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13856) Store value



v_accvgpr_read v4, a48 // (op 13922) 
v_accvgpr_read v5, a49 // (op 13922) 
v_accvgpr_read v6, a50 // (op 13922) 
v_accvgpr_read v7, a51 // (op 13922) 
v_mul_f32 v14, v13, v4 // (op 13922) 
v_mul_f32 v15, v13, v5 // (op 13922) 
v_mul_f32 v16, v13, v6 // (op 13922) 
v_mul_f32 v17, v13, v7 // (op 13922) 

v_cvt_f16_f32 v4, v14 // (op 13924) 
v_cvt_f16_f32 v5, v15 // (op 13924) 
v_cvt_f16_f32 v6, v16 // (op 13924) 
v_cvt_f16_f32 v7, v17 // (op 13924) 


v_pack_b32_f16 v8, v4, v5 // (op 13926) 
v_pack_b32_f16 v9, v6, v7 // (op 13926) 

v_add_u32 v3, 96, v12 // (op 13926) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13926) Store value



v_accvgpr_read v4, a52 // (op 13936) 
v_accvgpr_read v5, a53 // (op 13936) 
v_accvgpr_read v6, a54 // (op 13936) 
v_accvgpr_read v7, a55 // (op 13936) 
v_mul_f32 v14, v13, v4 // (op 13936) 
v_mul_f32 v15, v13, v5 // (op 13936) 
v_mul_f32 v16, v13, v6 // (op 13936) 
v_mul_f32 v17, v13, v7 // (op 13936) 

v_cvt_f16_f32 v4, v14 // (op 13938) 
v_cvt_f16_f32 v5, v15 // (op 13938) 
v_cvt_f16_f32 v6, v16 // (op 13938) 
v_cvt_f16_f32 v7, v17 // (op 13938) 


v_pack_b32_f16 v8, v4, v5 // (op 13928) 
v_pack_b32_f16 v9, v6, v7 // (op 13928) 

s_lshl_b32 s0, s4, 5 // (op 13928) 
s_add_u32 s0, s0, 96 // (op 13928) 
v_add_u32 v3, v12, s0 // (op 13928) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13928) Store value



v_accvgpr_read v4, a56 // (op 13942) 
v_accvgpr_read v5, a57 // (op 13942) 
v_accvgpr_read v6, a58 // (op 13942) 
v_accvgpr_read v7, a59 // (op 13942) 
v_mul_f32 v14, v13, v4 // (op 13942) 
v_mul_f32 v15, v13, v5 // (op 13942) 
v_mul_f32 v16, v13, v6 // (op 13942) 
v_mul_f32 v17, v13, v7 // (op 13942) 

v_cvt_f16_f32 v4, v14 // (op 13944) 
v_cvt_f16_f32 v5, v15 // (op 13944) 
v_cvt_f16_f32 v6, v16 // (op 13944) 
v_cvt_f16_f32 v7, v17 // (op 13944) 


v_pack_b32_f16 v8, v4, v5 // (op 13930) 
v_pack_b32_f16 v9, v6, v7 // (op 13930) 

s_lshl_b32 s0, s4, 6 // (op 13930) 
s_add_u32 s0, s0, 96 // (op 13930) 
v_add_u32 v3, v12, s0 // (op 13930) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13930) Store value



v_accvgpr_read v4, a60 // (op 13948) 
v_accvgpr_read v5, a61 // (op 13948) 
v_accvgpr_read v6, a62 // (op 13948) 
v_accvgpr_read v7, a63 // (op 13948) 
v_mul_f32 v14, v13, v4 // (op 13948) 
v_mul_f32 v15, v13, v5 // (op 13948) 
v_mul_f32 v16, v13, v6 // (op 13948) 
v_mul_f32 v17, v13, v7 // (op 13948) 

v_cvt_f16_f32 v4, v14 // (op 13950) 
v_cvt_f16_f32 v5, v15 // (op 13950) 
v_cvt_f16_f32 v6, v16 // (op 13950) 
v_cvt_f16_f32 v7, v17 // (op 13950) 


v_pack_b32_f16 v8, v4, v5 // (op 13932) 
v_pack_b32_f16 v9, v6, v7 // (op 13932) 

s_lshl_b32 s0, s4, 5 // (op 13932) 
s_mul_i32 s1, 3, s0 // (op 13932) 
s_add_u32 s1, s1, 96 // (op 13932) 

v_add_u32 v3, v12, s1 // (op 13932) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 13932) Store value



v_accvgpr_read v4, a64 // (op 13998) 
v_accvgpr_read v5, a65 // (op 13998) 
v_accvgpr_read v6, a66 // (op 13998) 
v_accvgpr_read v7, a67 // (op 13998) 
v_mul_f32 v14, v13, v4 // (op 13998) 
v_mul_f32 v15, v13, v5 // (op 13998) 
v_mul_f32 v16, v13, v6 // (op 13998) 
v_mul_f32 v17, v13, v7 // (op 13998) 

v_cvt_f16_f32 v4, v14 // (op 14000) 
v_cvt_f16_f32 v5, v15 // (op 14000) 
v_cvt_f16_f32 v6, v16 // (op 14000) 
v_cvt_f16_f32 v7, v17 // (op 14000) 


v_pack_b32_f16 v8, v4, v5 // (op 14002) 
v_pack_b32_f16 v9, v6, v7 // (op 14002) 

v_add_u32 v3, 128, v12 // (op 14002) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14002) Store value



v_accvgpr_read v4, a68 // (op 14012) 
v_accvgpr_read v5, a69 // (op 14012) 
v_accvgpr_read v6, a70 // (op 14012) 
v_accvgpr_read v7, a71 // (op 14012) 
v_mul_f32 v14, v13, v4 // (op 14012) 
v_mul_f32 v15, v13, v5 // (op 14012) 
v_mul_f32 v16, v13, v6 // (op 14012) 
v_mul_f32 v17, v13, v7 // (op 14012) 

v_cvt_f16_f32 v4, v14 // (op 14014) 
v_cvt_f16_f32 v5, v15 // (op 14014) 
v_cvt_f16_f32 v6, v16 // (op 14014) 
v_cvt_f16_f32 v7, v17 // (op 14014) 


v_pack_b32_f16 v8, v4, v5 // (op 14004) 
v_pack_b32_f16 v9, v6, v7 // (op 14004) 

s_lshl_b32 s0, s4, 5 // (op 14004) 
s_add_u32 s0, s0, 128 // (op 14004) 
v_add_u32 v3, v12, s0 // (op 14004) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14004) Store value



v_accvgpr_read v4, a72 // (op 14018) 
v_accvgpr_read v5, a73 // (op 14018) 
v_accvgpr_read v6, a74 // (op 14018) 
v_accvgpr_read v7, a75 // (op 14018) 
v_mul_f32 v14, v13, v4 // (op 14018) 
v_mul_f32 v15, v13, v5 // (op 14018) 
v_mul_f32 v16, v13, v6 // (op 14018) 
v_mul_f32 v17, v13, v7 // (op 14018) 

v_cvt_f16_f32 v4, v14 // (op 14020) 
v_cvt_f16_f32 v5, v15 // (op 14020) 
v_cvt_f16_f32 v6, v16 // (op 14020) 
v_cvt_f16_f32 v7, v17 // (op 14020) 


v_pack_b32_f16 v8, v4, v5 // (op 14006) 
v_pack_b32_f16 v9, v6, v7 // (op 14006) 

s_lshl_b32 s0, s4, 6 // (op 14006) 
s_add_u32 s0, s0, 128 // (op 14006) 
v_add_u32 v3, v12, s0 // (op 14006) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14006) Store value



v_accvgpr_read v4, a76 // (op 14024) 
v_accvgpr_read v5, a77 // (op 14024) 
v_accvgpr_read v6, a78 // (op 14024) 
v_accvgpr_read v7, a79 // (op 14024) 
v_mul_f32 v14, v13, v4 // (op 14024) 
v_mul_f32 v15, v13, v5 // (op 14024) 
v_mul_f32 v16, v13, v6 // (op 14024) 
v_mul_f32 v17, v13, v7 // (op 14024) 

v_cvt_f16_f32 v4, v14 // (op 14026) 
v_cvt_f16_f32 v5, v15 // (op 14026) 
v_cvt_f16_f32 v6, v16 // (op 14026) 
v_cvt_f16_f32 v7, v17 // (op 14026) 


v_pack_b32_f16 v8, v4, v5 // (op 14008) 
v_pack_b32_f16 v9, v6, v7 // (op 14008) 

s_lshl_b32 s0, s4, 5 // (op 14008) 
s_mul_i32 s1, 3, s0 // (op 14008) 
s_add_u32 s1, s1, 128 // (op 14008) 

v_add_u32 v3, v12, s1 // (op 14008) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14008) Store value



v_accvgpr_read v4, a80 // (op 14074) 
v_accvgpr_read v5, a81 // (op 14074) 
v_accvgpr_read v6, a82 // (op 14074) 
v_accvgpr_read v7, a83 // (op 14074) 
v_mul_f32 v14, v13, v4 // (op 14074) 
v_mul_f32 v15, v13, v5 // (op 14074) 
v_mul_f32 v16, v13, v6 // (op 14074) 
v_mul_f32 v17, v13, v7 // (op 14074) 

v_cvt_f16_f32 v4, v14 // (op 14076) 
v_cvt_f16_f32 v5, v15 // (op 14076) 
v_cvt_f16_f32 v6, v16 // (op 14076) 
v_cvt_f16_f32 v7, v17 // (op 14076) 


v_pack_b32_f16 v8, v4, v5 // (op 14078) 
v_pack_b32_f16 v9, v6, v7 // (op 14078) 

v_add_u32 v3, 160, v12 // (op 14078) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14078) Store value



v_accvgpr_read v4, a84 // (op 14088) 
v_accvgpr_read v5, a85 // (op 14088) 
v_accvgpr_read v6, a86 // (op 14088) 
v_accvgpr_read v7, a87 // (op 14088) 
v_mul_f32 v14, v13, v4 // (op 14088) 
v_mul_f32 v15, v13, v5 // (op 14088) 
v_mul_f32 v16, v13, v6 // (op 14088) 
v_mul_f32 v17, v13, v7 // (op 14088) 

v_cvt_f16_f32 v4, v14 // (op 14090) 
v_cvt_f16_f32 v5, v15 // (op 14090) 
v_cvt_f16_f32 v6, v16 // (op 14090) 
v_cvt_f16_f32 v7, v17 // (op 14090) 


v_pack_b32_f16 v8, v4, v5 // (op 14080) 
v_pack_b32_f16 v9, v6, v7 // (op 14080) 

s_lshl_b32 s0, s4, 5 // (op 14080) 
s_add_u32 s0, s0, 160 // (op 14080) 
v_add_u32 v3, v12, s0 // (op 14080) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14080) Store value



v_accvgpr_read v4, a88 // (op 14094) 
v_accvgpr_read v5, a89 // (op 14094) 
v_accvgpr_read v6, a90 // (op 14094) 
v_accvgpr_read v7, a91 // (op 14094) 
v_mul_f32 v14, v13, v4 // (op 14094) 
v_mul_f32 v15, v13, v5 // (op 14094) 
v_mul_f32 v16, v13, v6 // (op 14094) 
v_mul_f32 v17, v13, v7 // (op 14094) 

v_cvt_f16_f32 v4, v14 // (op 14096) 
v_cvt_f16_f32 v5, v15 // (op 14096) 
v_cvt_f16_f32 v6, v16 // (op 14096) 
v_cvt_f16_f32 v7, v17 // (op 14096) 


v_pack_b32_f16 v8, v4, v5 // (op 14082) 
v_pack_b32_f16 v9, v6, v7 // (op 14082) 

s_lshl_b32 s0, s4, 6 // (op 14082) 
s_add_u32 s0, s0, 160 // (op 14082) 
v_add_u32 v3, v12, s0 // (op 14082) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14082) Store value



v_accvgpr_read v4, a92 // (op 14100) 
v_accvgpr_read v5, a93 // (op 14100) 
v_accvgpr_read v6, a94 // (op 14100) 
v_accvgpr_read v7, a95 // (op 14100) 
v_mul_f32 v14, v13, v4 // (op 14100) 
v_mul_f32 v15, v13, v5 // (op 14100) 
v_mul_f32 v16, v13, v6 // (op 14100) 
v_mul_f32 v17, v13, v7 // (op 14100) 

v_cvt_f16_f32 v4, v14 // (op 14102) 
v_cvt_f16_f32 v5, v15 // (op 14102) 
v_cvt_f16_f32 v6, v16 // (op 14102) 
v_cvt_f16_f32 v7, v17 // (op 14102) 


v_pack_b32_f16 v8, v4, v5 // (op 14084) 
v_pack_b32_f16 v9, v6, v7 // (op 14084) 

s_lshl_b32 s0, s4, 5 // (op 14084) 
s_mul_i32 s1, 3, s0 // (op 14084) 
s_add_u32 s1, s1, 160 // (op 14084) 

v_add_u32 v3, v12, s1 // (op 14084) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14084) Store value



v_accvgpr_read v4, a96 // (op 14150) 
v_accvgpr_read v5, a97 // (op 14150) 
v_accvgpr_read v6, a98 // (op 14150) 
v_accvgpr_read v7, a99 // (op 14150) 
v_mul_f32 v14, v13, v4 // (op 14150) 
v_mul_f32 v15, v13, v5 // (op 14150) 
v_mul_f32 v16, v13, v6 // (op 14150) 
v_mul_f32 v17, v13, v7 // (op 14150) 

v_cvt_f16_f32 v4, v14 // (op 14152) 
v_cvt_f16_f32 v5, v15 // (op 14152) 
v_cvt_f16_f32 v6, v16 // (op 14152) 
v_cvt_f16_f32 v7, v17 // (op 14152) 


v_pack_b32_f16 v8, v4, v5 // (op 14154) 
v_pack_b32_f16 v9, v6, v7 // (op 14154) 

v_add_u32 v3, 192, v12 // (op 14154) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14154) Store value



v_accvgpr_read v4, a100 // (op 14164) 
v_accvgpr_read v5, a101 // (op 14164) 
v_accvgpr_read v6, a102 // (op 14164) 
v_accvgpr_read v7, a103 // (op 14164) 
v_mul_f32 v14, v13, v4 // (op 14164) 
v_mul_f32 v15, v13, v5 // (op 14164) 
v_mul_f32 v16, v13, v6 // (op 14164) 
v_mul_f32 v17, v13, v7 // (op 14164) 

v_cvt_f16_f32 v4, v14 // (op 14166) 
v_cvt_f16_f32 v5, v15 // (op 14166) 
v_cvt_f16_f32 v6, v16 // (op 14166) 
v_cvt_f16_f32 v7, v17 // (op 14166) 


v_pack_b32_f16 v8, v4, v5 // (op 14156) 
v_pack_b32_f16 v9, v6, v7 // (op 14156) 

s_lshl_b32 s0, s4, 5 // (op 14156) 
s_add_u32 s0, s0, 192 // (op 14156) 
v_add_u32 v3, v12, s0 // (op 14156) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14156) Store value



v_accvgpr_read v4, a104 // (op 14170) 
v_accvgpr_read v5, a105 // (op 14170) 
v_accvgpr_read v6, a106 // (op 14170) 
v_accvgpr_read v7, a107 // (op 14170) 
v_mul_f32 v14, v13, v4 // (op 14170) 
v_mul_f32 v15, v13, v5 // (op 14170) 
v_mul_f32 v16, v13, v6 // (op 14170) 
v_mul_f32 v17, v13, v7 // (op 14170) 

v_cvt_f16_f32 v4, v14 // (op 14172) 
v_cvt_f16_f32 v5, v15 // (op 14172) 
v_cvt_f16_f32 v6, v16 // (op 14172) 
v_cvt_f16_f32 v7, v17 // (op 14172) 


v_pack_b32_f16 v8, v4, v5 // (op 14158) 
v_pack_b32_f16 v9, v6, v7 // (op 14158) 

s_lshl_b32 s0, s4, 6 // (op 14158) 
s_add_u32 s0, s0, 192 // (op 14158) 
v_add_u32 v3, v12, s0 // (op 14158) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14158) Store value



v_accvgpr_read v4, a108 // (op 14176) 
v_accvgpr_read v5, a109 // (op 14176) 
v_accvgpr_read v6, a110 // (op 14176) 
v_accvgpr_read v7, a111 // (op 14176) 
v_mul_f32 v14, v13, v4 // (op 14176) 
v_mul_f32 v15, v13, v5 // (op 14176) 
v_mul_f32 v16, v13, v6 // (op 14176) 
v_mul_f32 v17, v13, v7 // (op 14176) 

v_cvt_f16_f32 v4, v14 // (op 14178) 
v_cvt_f16_f32 v5, v15 // (op 14178) 
v_cvt_f16_f32 v6, v16 // (op 14178) 
v_cvt_f16_f32 v7, v17 // (op 14178) 


v_pack_b32_f16 v8, v4, v5 // (op 14160) 
v_pack_b32_f16 v9, v6, v7 // (op 14160) 

s_lshl_b32 s0, s4, 5 // (op 14160) 
s_mul_i32 s1, 3, s0 // (op 14160) 
s_add_u32 s1, s1, 192 // (op 14160) 

v_add_u32 v3, v12, s1 // (op 14160) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14160) Store value



v_accvgpr_read v4, a112 // (op 14226) 
v_accvgpr_read v5, a113 // (op 14226) 
v_accvgpr_read v6, a114 // (op 14226) 
v_accvgpr_read v7, a115 // (op 14226) 
v_mul_f32 v14, v13, v4 // (op 14226) 
v_mul_f32 v15, v13, v5 // (op 14226) 
v_mul_f32 v16, v13, v6 // (op 14226) 
v_mul_f32 v17, v13, v7 // (op 14226) 

v_cvt_f16_f32 v4, v14 // (op 14228) 
v_cvt_f16_f32 v5, v15 // (op 14228) 
v_cvt_f16_f32 v6, v16 // (op 14228) 
v_cvt_f16_f32 v7, v17 // (op 14228) 


v_pack_b32_f16 v8, v4, v5 // (op 14230) 
v_pack_b32_f16 v9, v6, v7 // (op 14230) 

v_add_u32 v3, 224, v12 // (op 14230) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14230) Store value



v_accvgpr_read v4, a116 // (op 14240) 
v_accvgpr_read v5, a117 // (op 14240) 
v_accvgpr_read v6, a118 // (op 14240) 
v_accvgpr_read v7, a119 // (op 14240) 
v_mul_f32 v14, v13, v4 // (op 14240) 
v_mul_f32 v15, v13, v5 // (op 14240) 
v_mul_f32 v16, v13, v6 // (op 14240) 
v_mul_f32 v17, v13, v7 // (op 14240) 

v_cvt_f16_f32 v4, v14 // (op 14242) 
v_cvt_f16_f32 v5, v15 // (op 14242) 
v_cvt_f16_f32 v6, v16 // (op 14242) 
v_cvt_f16_f32 v7, v17 // (op 14242) 


v_pack_b32_f16 v8, v4, v5 // (op 14232) 
v_pack_b32_f16 v9, v6, v7 // (op 14232) 

s_lshl_b32 s0, s4, 5 // (op 14232) 
s_add_u32 s0, s0, 224 // (op 14232) 
v_add_u32 v3, v12, s0 // (op 14232) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14232) Store value



v_accvgpr_read v4, a120 // (op 14246) 
v_accvgpr_read v5, a121 // (op 14246) 
v_accvgpr_read v6, a122 // (op 14246) 
v_accvgpr_read v7, a123 // (op 14246) 
v_mul_f32 v14, v13, v4 // (op 14246) 
v_mul_f32 v15, v13, v5 // (op 14246) 
v_mul_f32 v16, v13, v6 // (op 14246) 
v_mul_f32 v17, v13, v7 // (op 14246) 

v_cvt_f16_f32 v4, v14 // (op 14248) 
v_cvt_f16_f32 v5, v15 // (op 14248) 
v_cvt_f16_f32 v6, v16 // (op 14248) 
v_cvt_f16_f32 v7, v17 // (op 14248) 


v_pack_b32_f16 v8, v4, v5 // (op 14234) 
v_pack_b32_f16 v9, v6, v7 // (op 14234) 

s_lshl_b32 s0, s4, 6 // (op 14234) 
s_add_u32 s0, s0, 224 // (op 14234) 
v_add_u32 v3, v12, s0 // (op 14234) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14234) Store value



v_accvgpr_read v4, a124 // (op 14252) 
v_accvgpr_read v5, a125 // (op 14252) 
v_accvgpr_read v6, a126 // (op 14252) 
v_accvgpr_read v7, a127 // (op 14252) 
v_mul_f32 v14, v13, v4 // (op 14252) 
v_mul_f32 v15, v13, v5 // (op 14252) 
v_mul_f32 v16, v13, v6 // (op 14252) 
v_mul_f32 v17, v13, v7 // (op 14252) 

v_cvt_f16_f32 v4, v14 // (op 14254) 
v_cvt_f16_f32 v5, v15 // (op 14254) 
v_cvt_f16_f32 v6, v16 // (op 14254) 
v_cvt_f16_f32 v7, v17 // (op 14254) 


v_pack_b32_f16 v8, v4, v5 // (op 14236) 
v_pack_b32_f16 v9, v6, v7 // (op 14236) 

s_lshl_b32 s0, s4, 5 // (op 14236) 
s_mul_i32 s1, 3, s0 // (op 14236) 
s_add_u32 s1, s1, 224 // (op 14236) 

v_add_u32 v3, v12, s1 // (op 14236) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14236) Store value



v_accvgpr_read v4, a128 // (op 14302) 
v_accvgpr_read v5, a129 // (op 14302) 
v_accvgpr_read v6, a130 // (op 14302) 
v_accvgpr_read v7, a131 // (op 14302) 
v_mul_f32 v14, v13, v4 // (op 14302) 
v_mul_f32 v15, v13, v5 // (op 14302) 
v_mul_f32 v16, v13, v6 // (op 14302) 
v_mul_f32 v17, v13, v7 // (op 14302) 

v_cvt_f16_f32 v4, v14 // (op 14304) 
v_cvt_f16_f32 v5, v15 // (op 14304) 
v_cvt_f16_f32 v6, v16 // (op 14304) 
v_cvt_f16_f32 v7, v17 // (op 14304) 


v_pack_b32_f16 v8, v4, v5 // (op 14306) 
v_pack_b32_f16 v9, v6, v7 // (op 14306) 

v_add_u32 v3, 256, v12 // (op 14306) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14306) Store value



v_accvgpr_read v4, a132 // (op 14316) 
v_accvgpr_read v5, a133 // (op 14316) 
v_accvgpr_read v6, a134 // (op 14316) 
v_accvgpr_read v7, a135 // (op 14316) 
v_mul_f32 v14, v13, v4 // (op 14316) 
v_mul_f32 v15, v13, v5 // (op 14316) 
v_mul_f32 v16, v13, v6 // (op 14316) 
v_mul_f32 v17, v13, v7 // (op 14316) 

v_cvt_f16_f32 v4, v14 // (op 14318) 
v_cvt_f16_f32 v5, v15 // (op 14318) 
v_cvt_f16_f32 v6, v16 // (op 14318) 
v_cvt_f16_f32 v7, v17 // (op 14318) 


v_pack_b32_f16 v8, v4, v5 // (op 14308) 
v_pack_b32_f16 v9, v6, v7 // (op 14308) 

s_lshl_b32 s0, s4, 5 // (op 14308) 
s_add_u32 s0, s0, 256 // (op 14308) 
v_add_u32 v3, v12, s0 // (op 14308) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14308) Store value



v_accvgpr_read v4, a136 // (op 14322) 
v_accvgpr_read v5, a137 // (op 14322) 
v_accvgpr_read v6, a138 // (op 14322) 
v_accvgpr_read v7, a139 // (op 14322) 
v_mul_f32 v14, v13, v4 // (op 14322) 
v_mul_f32 v15, v13, v5 // (op 14322) 
v_mul_f32 v16, v13, v6 // (op 14322) 
v_mul_f32 v17, v13, v7 // (op 14322) 

v_cvt_f16_f32 v4, v14 // (op 14324) 
v_cvt_f16_f32 v5, v15 // (op 14324) 
v_cvt_f16_f32 v6, v16 // (op 14324) 
v_cvt_f16_f32 v7, v17 // (op 14324) 


v_pack_b32_f16 v8, v4, v5 // (op 14310) 
v_pack_b32_f16 v9, v6, v7 // (op 14310) 

s_lshl_b32 s0, s4, 6 // (op 14310) 
s_add_u32 s0, s0, 256 // (op 14310) 
v_add_u32 v3, v12, s0 // (op 14310) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14310) Store value



v_accvgpr_read v4, a140 // (op 14328) 
v_accvgpr_read v5, a141 // (op 14328) 
v_accvgpr_read v6, a142 // (op 14328) 
v_accvgpr_read v7, a143 // (op 14328) 
v_mul_f32 v14, v13, v4 // (op 14328) 
v_mul_f32 v15, v13, v5 // (op 14328) 
v_mul_f32 v16, v13, v6 // (op 14328) 
v_mul_f32 v17, v13, v7 // (op 14328) 

v_cvt_f16_f32 v4, v14 // (op 14330) 
v_cvt_f16_f32 v5, v15 // (op 14330) 
v_cvt_f16_f32 v6, v16 // (op 14330) 
v_cvt_f16_f32 v7, v17 // (op 14330) 


v_pack_b32_f16 v8, v4, v5 // (op 14312) 
v_pack_b32_f16 v9, v6, v7 // (op 14312) 

s_lshl_b32 s0, s4, 5 // (op 14312) 
s_mul_i32 s1, 3, s0 // (op 14312) 
s_add_u32 s1, s1, 256 // (op 14312) 

v_add_u32 v3, v12, s1 // (op 14312) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14312) Store value



v_accvgpr_read v4, a144 // (op 14378) 
v_accvgpr_read v5, a145 // (op 14378) 
v_accvgpr_read v6, a146 // (op 14378) 
v_accvgpr_read v7, a147 // (op 14378) 
v_mul_f32 v14, v13, v4 // (op 14378) 
v_mul_f32 v15, v13, v5 // (op 14378) 
v_mul_f32 v16, v13, v6 // (op 14378) 
v_mul_f32 v17, v13, v7 // (op 14378) 

v_cvt_f16_f32 v4, v14 // (op 14380) 
v_cvt_f16_f32 v5, v15 // (op 14380) 
v_cvt_f16_f32 v6, v16 // (op 14380) 
v_cvt_f16_f32 v7, v17 // (op 14380) 


v_pack_b32_f16 v8, v4, v5 // (op 14382) 
v_pack_b32_f16 v9, v6, v7 // (op 14382) 

v_add_u32 v3, 288, v12 // (op 14382) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14382) Store value



v_accvgpr_read v4, a148 // (op 14392) 
v_accvgpr_read v5, a149 // (op 14392) 
v_accvgpr_read v6, a150 // (op 14392) 
v_accvgpr_read v7, a151 // (op 14392) 
v_mul_f32 v14, v13, v4 // (op 14392) 
v_mul_f32 v15, v13, v5 // (op 14392) 
v_mul_f32 v16, v13, v6 // (op 14392) 
v_mul_f32 v17, v13, v7 // (op 14392) 

v_cvt_f16_f32 v4, v14 // (op 14394) 
v_cvt_f16_f32 v5, v15 // (op 14394) 
v_cvt_f16_f32 v6, v16 // (op 14394) 
v_cvt_f16_f32 v7, v17 // (op 14394) 


v_pack_b32_f16 v8, v4, v5 // (op 14384) 
v_pack_b32_f16 v9, v6, v7 // (op 14384) 

s_lshl_b32 s0, s4, 5 // (op 14384) 
s_add_u32 s0, s0, 288 // (op 14384) 
v_add_u32 v3, v12, s0 // (op 14384) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14384) Store value



v_accvgpr_read v4, a152 // (op 14398) 
v_accvgpr_read v5, a153 // (op 14398) 
v_accvgpr_read v6, a154 // (op 14398) 
v_accvgpr_read v7, a155 // (op 14398) 
v_mul_f32 v14, v13, v4 // (op 14398) 
v_mul_f32 v15, v13, v5 // (op 14398) 
v_mul_f32 v16, v13, v6 // (op 14398) 
v_mul_f32 v17, v13, v7 // (op 14398) 

v_cvt_f16_f32 v4, v14 // (op 14400) 
v_cvt_f16_f32 v5, v15 // (op 14400) 
v_cvt_f16_f32 v6, v16 // (op 14400) 
v_cvt_f16_f32 v7, v17 // (op 14400) 


v_pack_b32_f16 v8, v4, v5 // (op 14386) 
v_pack_b32_f16 v9, v6, v7 // (op 14386) 

s_lshl_b32 s0, s4, 6 // (op 14386) 
s_add_u32 s0, s0, 288 // (op 14386) 
v_add_u32 v3, v12, s0 // (op 14386) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14386) Store value



v_accvgpr_read v4, a156 // (op 14404) 
v_accvgpr_read v5, a157 // (op 14404) 
v_accvgpr_read v6, a158 // (op 14404) 
v_accvgpr_read v7, a159 // (op 14404) 
v_mul_f32 v14, v13, v4 // (op 14404) 
v_mul_f32 v15, v13, v5 // (op 14404) 
v_mul_f32 v16, v13, v6 // (op 14404) 
v_mul_f32 v17, v13, v7 // (op 14404) 

v_cvt_f16_f32 v4, v14 // (op 14406) 
v_cvt_f16_f32 v5, v15 // (op 14406) 
v_cvt_f16_f32 v6, v16 // (op 14406) 
v_cvt_f16_f32 v7, v17 // (op 14406) 


v_pack_b32_f16 v8, v4, v5 // (op 14388) 
v_pack_b32_f16 v9, v6, v7 // (op 14388) 

s_lshl_b32 s0, s4, 5 // (op 14388) 
s_mul_i32 s1, 3, s0 // (op 14388) 
s_add_u32 s1, s1, 288 // (op 14388) 

v_add_u32 v3, v12, s1 // (op 14388) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14388) Store value



v_accvgpr_read v4, a160 // (op 14454) 
v_accvgpr_read v5, a161 // (op 14454) 
v_accvgpr_read v6, a162 // (op 14454) 
v_accvgpr_read v7, a163 // (op 14454) 
v_mul_f32 v14, v13, v4 // (op 14454) 
v_mul_f32 v15, v13, v5 // (op 14454) 
v_mul_f32 v16, v13, v6 // (op 14454) 
v_mul_f32 v17, v13, v7 // (op 14454) 

v_cvt_f16_f32 v4, v14 // (op 14456) 
v_cvt_f16_f32 v5, v15 // (op 14456) 
v_cvt_f16_f32 v6, v16 // (op 14456) 
v_cvt_f16_f32 v7, v17 // (op 14456) 


v_pack_b32_f16 v8, v4, v5 // (op 14458) 
v_pack_b32_f16 v9, v6, v7 // (op 14458) 

v_add_u32 v3, 320, v12 // (op 14458) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14458) Store value



v_accvgpr_read v4, a164 // (op 14468) 
v_accvgpr_read v5, a165 // (op 14468) 
v_accvgpr_read v6, a166 // (op 14468) 
v_accvgpr_read v7, a167 // (op 14468) 
v_mul_f32 v14, v13, v4 // (op 14468) 
v_mul_f32 v15, v13, v5 // (op 14468) 
v_mul_f32 v16, v13, v6 // (op 14468) 
v_mul_f32 v17, v13, v7 // (op 14468) 

v_cvt_f16_f32 v4, v14 // (op 14470) 
v_cvt_f16_f32 v5, v15 // (op 14470) 
v_cvt_f16_f32 v6, v16 // (op 14470) 
v_cvt_f16_f32 v7, v17 // (op 14470) 


v_pack_b32_f16 v8, v4, v5 // (op 14460) 
v_pack_b32_f16 v9, v6, v7 // (op 14460) 

s_lshl_b32 s0, s4, 5 // (op 14460) 
s_add_u32 s0, s0, 320 // (op 14460) 
v_add_u32 v3, v12, s0 // (op 14460) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14460) Store value



v_accvgpr_read v4, a168 // (op 14474) 
v_accvgpr_read v5, a169 // (op 14474) 
v_accvgpr_read v6, a170 // (op 14474) 
v_accvgpr_read v7, a171 // (op 14474) 
v_mul_f32 v14, v13, v4 // (op 14474) 
v_mul_f32 v15, v13, v5 // (op 14474) 
v_mul_f32 v16, v13, v6 // (op 14474) 
v_mul_f32 v17, v13, v7 // (op 14474) 

v_cvt_f16_f32 v4, v14 // (op 14476) 
v_cvt_f16_f32 v5, v15 // (op 14476) 
v_cvt_f16_f32 v6, v16 // (op 14476) 
v_cvt_f16_f32 v7, v17 // (op 14476) 


v_pack_b32_f16 v8, v4, v5 // (op 14462) 
v_pack_b32_f16 v9, v6, v7 // (op 14462) 

s_lshl_b32 s0, s4, 6 // (op 14462) 
s_add_u32 s0, s0, 320 // (op 14462) 
v_add_u32 v3, v12, s0 // (op 14462) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14462) Store value



v_accvgpr_read v4, a172 // (op 14480) 
v_accvgpr_read v5, a173 // (op 14480) 
v_accvgpr_read v6, a174 // (op 14480) 
v_accvgpr_read v7, a175 // (op 14480) 
v_mul_f32 v14, v13, v4 // (op 14480) 
v_mul_f32 v15, v13, v5 // (op 14480) 
v_mul_f32 v16, v13, v6 // (op 14480) 
v_mul_f32 v17, v13, v7 // (op 14480) 

v_cvt_f16_f32 v4, v14 // (op 14482) 
v_cvt_f16_f32 v5, v15 // (op 14482) 
v_cvt_f16_f32 v6, v16 // (op 14482) 
v_cvt_f16_f32 v7, v17 // (op 14482) 


v_pack_b32_f16 v8, v4, v5 // (op 14464) 
v_pack_b32_f16 v9, v6, v7 // (op 14464) 

s_lshl_b32 s0, s4, 5 // (op 14464) 
s_mul_i32 s1, 3, s0 // (op 14464) 
s_add_u32 s1, s1, 320 // (op 14464) 

v_add_u32 v3, v12, s1 // (op 14464) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14464) Store value



v_accvgpr_read v4, a176 // (op 14530) 
v_accvgpr_read v5, a177 // (op 14530) 
v_accvgpr_read v6, a178 // (op 14530) 
v_accvgpr_read v7, a179 // (op 14530) 
v_mul_f32 v14, v13, v4 // (op 14530) 
v_mul_f32 v15, v13, v5 // (op 14530) 
v_mul_f32 v16, v13, v6 // (op 14530) 
v_mul_f32 v17, v13, v7 // (op 14530) 

v_cvt_f16_f32 v4, v14 // (op 14532) 
v_cvt_f16_f32 v5, v15 // (op 14532) 
v_cvt_f16_f32 v6, v16 // (op 14532) 
v_cvt_f16_f32 v7, v17 // (op 14532) 


v_pack_b32_f16 v8, v4, v5 // (op 14534) 
v_pack_b32_f16 v9, v6, v7 // (op 14534) 

v_add_u32 v3, 352, v12 // (op 14534) 

buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14534) Store value



v_accvgpr_read v4, a180 // (op 14544) 
v_accvgpr_read v5, a181 // (op 14544) 
v_accvgpr_read v6, a182 // (op 14544) 
v_accvgpr_read v7, a183 // (op 14544) 
v_mul_f32 v14, v13, v4 // (op 14544) 
v_mul_f32 v15, v13, v5 // (op 14544) 
v_mul_f32 v16, v13, v6 // (op 14544) 
v_mul_f32 v17, v13, v7 // (op 14544) 

v_cvt_f16_f32 v4, v14 // (op 14546) 
v_cvt_f16_f32 v5, v15 // (op 14546) 
v_cvt_f16_f32 v6, v16 // (op 14546) 
v_cvt_f16_f32 v7, v17 // (op 14546) 


v_pack_b32_f16 v8, v4, v5 // (op 14536) 
v_pack_b32_f16 v9, v6, v7 // (op 14536) 

s_lshl_b32 s0, s4, 5 // (op 14536) 
s_add_u32 s0, s0, 352 // (op 14536) 
v_add_u32 v3, v12, s0 // (op 14536) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14536) Store value



v_accvgpr_read v4, a184 // (op 14550) 
v_accvgpr_read v5, a185 // (op 14550) 
v_accvgpr_read v6, a186 // (op 14550) 
v_accvgpr_read v7, a187 // (op 14550) 
v_mul_f32 v14, v13, v4 // (op 14550) 
v_mul_f32 v15, v13, v5 // (op 14550) 
v_mul_f32 v16, v13, v6 // (op 14550) 
v_mul_f32 v17, v13, v7 // (op 14550) 

v_cvt_f16_f32 v4, v14 // (op 14552) 
v_cvt_f16_f32 v5, v15 // (op 14552) 
v_cvt_f16_f32 v6, v16 // (op 14552) 
v_cvt_f16_f32 v7, v17 // (op 14552) 


v_pack_b32_f16 v8, v4, v5 // (op 14538) 
v_pack_b32_f16 v9, v6, v7 // (op 14538) 

s_lshl_b32 s0, s4, 6 // (op 14538) 
s_add_u32 s0, s0, 352 // (op 14538) 
v_add_u32 v3, v12, s0 // (op 14538) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14538) Store value



v_accvgpr_read v4, a188 // (op 14556) 
v_accvgpr_read v5, a189 // (op 14556) 
v_accvgpr_read v6, a190 // (op 14556) 
v_accvgpr_read v7, a191 // (op 14556) 
v_mul_f32 v14, v13, v4 // (op 14556) 
v_mul_f32 v15, v13, v5 // (op 14556) 
v_mul_f32 v16, v13, v6 // (op 14556) 
v_mul_f32 v17, v13, v7 // (op 14556) 

v_cvt_f16_f32 v4, v14 // (op 14558) 
v_cvt_f16_f32 v5, v15 // (op 14558) 
v_cvt_f16_f32 v6, v16 // (op 14558) 
v_cvt_f16_f32 v7, v17 // (op 14558) 


v_pack_b32_f16 v8, v4, v5 // (op 14540) 
v_pack_b32_f16 v9, v6, v7 // (op 14540) 

s_lshl_b32 s0, s4, 5 // (op 14540) 
s_mul_i32 s1, 3, s0 // (op 14540) 
s_add_u32 s1, s1, 352 // (op 14540) 

v_add_u32 v3, v12, s1 // (op 14540) 


buffer_store_dwordx2 v[8:9], v3, s[44:47], 0 offen // (op 14540) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ConditionalBottom_0_beta_116
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ConditionalFalse_0_beta_116:


v_add_u32 v3, 352, v11 // (op 15736) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15736) Load value

v_accvgpr_read v6, a176 // (op 15723) 
v_accvgpr_read v7, a177 // (op 15723) 
v_accvgpr_read v8, a178 // (op 15723) 
v_accvgpr_read v9, a179 // (op 15723) 
v_mul_f32 v14, v13, v6 // (op 15723) 
v_mul_f32 v15, v13, v7 // (op 15723) 
v_mul_f32 v16, v13, v8 // (op 15723) 
v_mul_f32 v17, v13, v9 // (op 15723) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15735) 
v_lshrrev_b32 v7, 16, v4 // (op 15735) 
v_cvt_f32_f16 v7, v7 // (op 15735) 
v_mul_f32 v18, v0, v6 // (op 15735) 
v_mul_f32 v19, v0, v7 // (op 15735) 
v_cvt_f32_f16 v6, v5 // (op 15735) 
v_lshrrev_b32 v7, 16, v5 // (op 15735) 
v_cvt_f32_f16 v7, v7 // (op 15735) 
v_mul_f32 v20, v0, v6 // (op 15735) 
v_mul_f32 v21, v0, v7 // (op 15735) 


v_add_f32 v4, v18, v14 // (op 15724) 
v_add_f32 v5, v19, v15 // (op 15724) 
v_add_f32 v6, v20, v16 // (op 15724) 
v_add_f32 v7, v21, v17 // (op 15724) 


v_cvt_f16_f32 v14, v4 // (op 15725) 
v_cvt_f16_f32 v15, v5 // (op 15725) 
v_cvt_f16_f32 v16, v6 // (op 15725) 
v_cvt_f16_f32 v17, v7 // (op 15725) 


v_pack_b32_f16 v4, v14, v15 // (op 15727) 
v_pack_b32_f16 v5, v16, v17 // (op 15727) 

v_add_u32 v3, 352, v12 // (op 15727) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15727) Store value





s_lshl_b32 s0, s2, 5 // (op 15741) 
s_add_u32 s0, s0, 352 // (op 15741) 
v_add_u32 v3, v11, s0 // (op 15741) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15741) Load value

v_accvgpr_read v6, a180 // (op 15753) 
v_accvgpr_read v7, a181 // (op 15753) 
v_accvgpr_read v8, a182 // (op 15753) 
v_accvgpr_read v9, a183 // (op 15753) 
v_mul_f32 v14, v13, v6 // (op 15753) 
v_mul_f32 v15, v13, v7 // (op 15753) 
v_mul_f32 v16, v13, v8 // (op 15753) 
v_mul_f32 v17, v13, v9 // (op 15753) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15738) 
v_lshrrev_b32 v7, 16, v4 // (op 15738) 
v_cvt_f32_f16 v7, v7 // (op 15738) 
v_mul_f32 v18, v0, v6 // (op 15738) 
v_mul_f32 v19, v0, v7 // (op 15738) 
v_cvt_f32_f16 v6, v5 // (op 15738) 
v_lshrrev_b32 v7, 16, v5 // (op 15738) 
v_cvt_f32_f16 v7, v7 // (op 15738) 
v_mul_f32 v20, v0, v6 // (op 15738) 
v_mul_f32 v21, v0, v7 // (op 15738) 


v_add_f32 v4, v18, v14 // (op 15739) 
v_add_f32 v5, v19, v15 // (op 15739) 
v_add_f32 v6, v20, v16 // (op 15739) 
v_add_f32 v7, v21, v17 // (op 15739) 


v_cvt_f16_f32 v14, v4 // (op 15740) 
v_cvt_f16_f32 v15, v5 // (op 15740) 
v_cvt_f16_f32 v16, v6 // (op 15740) 
v_cvt_f16_f32 v17, v7 // (op 15740) 


v_pack_b32_f16 v4, v14, v15 // (op 15729) 
v_pack_b32_f16 v5, v16, v17 // (op 15729) 

s_lshl_b32 s0, s4, 5 // (op 15729) 
s_add_u32 s0, s0, 352 // (op 15729) 
v_add_u32 v3, v12, s0 // (op 15729) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15729) Store value





s_lshl_b32 s0, s2, 6 // (op 15746) 
s_add_u32 s0, s0, 352 // (op 15746) 
v_add_u32 v3, v11, s0 // (op 15746) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15746) Load value

v_accvgpr_read v6, a184 // (op 15755) 
v_accvgpr_read v7, a185 // (op 15755) 
v_accvgpr_read v8, a186 // (op 15755) 
v_accvgpr_read v9, a187 // (op 15755) 
v_mul_f32 v14, v13, v6 // (op 15755) 
v_mul_f32 v15, v13, v7 // (op 15755) 
v_mul_f32 v16, v13, v8 // (op 15755) 
v_mul_f32 v17, v13, v9 // (op 15755) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15743) 
v_lshrrev_b32 v7, 16, v4 // (op 15743) 
v_cvt_f32_f16 v7, v7 // (op 15743) 
v_mul_f32 v18, v0, v6 // (op 15743) 
v_mul_f32 v19, v0, v7 // (op 15743) 
v_cvt_f32_f16 v6, v5 // (op 15743) 
v_lshrrev_b32 v7, 16, v5 // (op 15743) 
v_cvt_f32_f16 v7, v7 // (op 15743) 
v_mul_f32 v20, v0, v6 // (op 15743) 
v_mul_f32 v21, v0, v7 // (op 15743) 


v_add_f32 v4, v18, v14 // (op 15744) 
v_add_f32 v5, v19, v15 // (op 15744) 
v_add_f32 v6, v20, v16 // (op 15744) 
v_add_f32 v7, v21, v17 // (op 15744) 


v_cvt_f16_f32 v14, v4 // (op 15745) 
v_cvt_f16_f32 v15, v5 // (op 15745) 
v_cvt_f16_f32 v16, v6 // (op 15745) 
v_cvt_f16_f32 v17, v7 // (op 15745) 


v_pack_b32_f16 v4, v14, v15 // (op 15731) 
v_pack_b32_f16 v5, v16, v17 // (op 15731) 

s_lshl_b32 s0, s4, 6 // (op 15731) 
s_add_u32 s0, s0, 352 // (op 15731) 
v_add_u32 v3, v12, s0 // (op 15731) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15731) Store value





s_lshl_b32 s0, s2, 5 // (op 15751) 
s_mul_i32 s1, 3, s0 // (op 15751) 
s_add_u32 s1, s1, 352 // (op 15751) 

v_add_u32 v3, v11, s1 // (op 15751) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15751) Load value

v_accvgpr_read v6, a188 // (op 15757) 
v_accvgpr_read v7, a189 // (op 15757) 
v_accvgpr_read v8, a190 // (op 15757) 
v_accvgpr_read v9, a191 // (op 15757) 
v_mul_f32 v14, v13, v6 // (op 15757) 
v_mul_f32 v15, v13, v7 // (op 15757) 
v_mul_f32 v16, v13, v8 // (op 15757) 
v_mul_f32 v17, v13, v9 // (op 15757) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15748) 
v_lshrrev_b32 v7, 16, v4 // (op 15748) 
v_cvt_f32_f16 v7, v7 // (op 15748) 
v_mul_f32 v18, v0, v6 // (op 15748) 
v_mul_f32 v19, v0, v7 // (op 15748) 
v_cvt_f32_f16 v6, v5 // (op 15748) 
v_lshrrev_b32 v7, 16, v5 // (op 15748) 
v_cvt_f32_f16 v7, v7 // (op 15748) 
v_mul_f32 v20, v0, v6 // (op 15748) 
v_mul_f32 v21, v0, v7 // (op 15748) 


v_add_f32 v4, v18, v14 // (op 15749) 
v_add_f32 v5, v19, v15 // (op 15749) 
v_add_f32 v6, v20, v16 // (op 15749) 
v_add_f32 v7, v21, v17 // (op 15749) 


v_cvt_f16_f32 v14, v4 // (op 15750) 
v_cvt_f16_f32 v15, v5 // (op 15750) 
v_cvt_f16_f32 v16, v6 // (op 15750) 
v_cvt_f16_f32 v17, v7 // (op 15750) 


v_pack_b32_f16 v4, v14, v15 // (op 15733) 
v_pack_b32_f16 v5, v16, v17 // (op 15733) 

s_lshl_b32 s0, s4, 5 // (op 15733) 
s_mul_i32 s1, 3, s0 // (op 15733) 
s_add_u32 s1, s1, 352 // (op 15733) 

v_add_u32 v3, v12, s1 // (op 15733) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15733) Store value





v_add_u32 v3, 320, v11 // (op 15649) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15649) Load value

v_accvgpr_read v6, a160 // (op 15636) 
v_accvgpr_read v7, a161 // (op 15636) 
v_accvgpr_read v8, a162 // (op 15636) 
v_accvgpr_read v9, a163 // (op 15636) 
v_mul_f32 v14, v13, v6 // (op 15636) 
v_mul_f32 v15, v13, v7 // (op 15636) 
v_mul_f32 v16, v13, v8 // (op 15636) 
v_mul_f32 v17, v13, v9 // (op 15636) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15648) 
v_lshrrev_b32 v7, 16, v4 // (op 15648) 
v_cvt_f32_f16 v7, v7 // (op 15648) 
v_mul_f32 v18, v0, v6 // (op 15648) 
v_mul_f32 v19, v0, v7 // (op 15648) 
v_cvt_f32_f16 v6, v5 // (op 15648) 
v_lshrrev_b32 v7, 16, v5 // (op 15648) 
v_cvt_f32_f16 v7, v7 // (op 15648) 
v_mul_f32 v20, v0, v6 // (op 15648) 
v_mul_f32 v21, v0, v7 // (op 15648) 


v_add_f32 v4, v18, v14 // (op 15637) 
v_add_f32 v5, v19, v15 // (op 15637) 
v_add_f32 v6, v20, v16 // (op 15637) 
v_add_f32 v7, v21, v17 // (op 15637) 


v_cvt_f16_f32 v14, v4 // (op 15638) 
v_cvt_f16_f32 v15, v5 // (op 15638) 
v_cvt_f16_f32 v16, v6 // (op 15638) 
v_cvt_f16_f32 v17, v7 // (op 15638) 


v_pack_b32_f16 v4, v14, v15 // (op 15640) 
v_pack_b32_f16 v5, v16, v17 // (op 15640) 

v_add_u32 v3, 320, v12 // (op 15640) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15640) Store value





s_lshl_b32 s0, s2, 5 // (op 15654) 
s_add_u32 s0, s0, 320 // (op 15654) 
v_add_u32 v3, v11, s0 // (op 15654) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15654) Load value

v_accvgpr_read v6, a164 // (op 15666) 
v_accvgpr_read v7, a165 // (op 15666) 
v_accvgpr_read v8, a166 // (op 15666) 
v_accvgpr_read v9, a167 // (op 15666) 
v_mul_f32 v14, v13, v6 // (op 15666) 
v_mul_f32 v15, v13, v7 // (op 15666) 
v_mul_f32 v16, v13, v8 // (op 15666) 
v_mul_f32 v17, v13, v9 // (op 15666) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15651) 
v_lshrrev_b32 v7, 16, v4 // (op 15651) 
v_cvt_f32_f16 v7, v7 // (op 15651) 
v_mul_f32 v18, v0, v6 // (op 15651) 
v_mul_f32 v19, v0, v7 // (op 15651) 
v_cvt_f32_f16 v6, v5 // (op 15651) 
v_lshrrev_b32 v7, 16, v5 // (op 15651) 
v_cvt_f32_f16 v7, v7 // (op 15651) 
v_mul_f32 v20, v0, v6 // (op 15651) 
v_mul_f32 v21, v0, v7 // (op 15651) 


v_add_f32 v4, v18, v14 // (op 15652) 
v_add_f32 v5, v19, v15 // (op 15652) 
v_add_f32 v6, v20, v16 // (op 15652) 
v_add_f32 v7, v21, v17 // (op 15652) 


v_cvt_f16_f32 v14, v4 // (op 15653) 
v_cvt_f16_f32 v15, v5 // (op 15653) 
v_cvt_f16_f32 v16, v6 // (op 15653) 
v_cvt_f16_f32 v17, v7 // (op 15653) 


v_pack_b32_f16 v4, v14, v15 // (op 15642) 
v_pack_b32_f16 v5, v16, v17 // (op 15642) 

s_lshl_b32 s0, s4, 5 // (op 15642) 
s_add_u32 s0, s0, 320 // (op 15642) 
v_add_u32 v3, v12, s0 // (op 15642) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15642) Store value





s_lshl_b32 s0, s2, 6 // (op 15659) 
s_add_u32 s0, s0, 320 // (op 15659) 
v_add_u32 v3, v11, s0 // (op 15659) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15659) Load value

v_accvgpr_read v6, a168 // (op 15668) 
v_accvgpr_read v7, a169 // (op 15668) 
v_accvgpr_read v8, a170 // (op 15668) 
v_accvgpr_read v9, a171 // (op 15668) 
v_mul_f32 v14, v13, v6 // (op 15668) 
v_mul_f32 v15, v13, v7 // (op 15668) 
v_mul_f32 v16, v13, v8 // (op 15668) 
v_mul_f32 v17, v13, v9 // (op 15668) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15656) 
v_lshrrev_b32 v7, 16, v4 // (op 15656) 
v_cvt_f32_f16 v7, v7 // (op 15656) 
v_mul_f32 v18, v0, v6 // (op 15656) 
v_mul_f32 v19, v0, v7 // (op 15656) 
v_cvt_f32_f16 v6, v5 // (op 15656) 
v_lshrrev_b32 v7, 16, v5 // (op 15656) 
v_cvt_f32_f16 v7, v7 // (op 15656) 
v_mul_f32 v20, v0, v6 // (op 15656) 
v_mul_f32 v21, v0, v7 // (op 15656) 


v_add_f32 v4, v18, v14 // (op 15657) 
v_add_f32 v5, v19, v15 // (op 15657) 
v_add_f32 v6, v20, v16 // (op 15657) 
v_add_f32 v7, v21, v17 // (op 15657) 


v_cvt_f16_f32 v14, v4 // (op 15658) 
v_cvt_f16_f32 v15, v5 // (op 15658) 
v_cvt_f16_f32 v16, v6 // (op 15658) 
v_cvt_f16_f32 v17, v7 // (op 15658) 


v_pack_b32_f16 v4, v14, v15 // (op 15644) 
v_pack_b32_f16 v5, v16, v17 // (op 15644) 

s_lshl_b32 s0, s4, 6 // (op 15644) 
s_add_u32 s0, s0, 320 // (op 15644) 
v_add_u32 v3, v12, s0 // (op 15644) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15644) Store value





s_lshl_b32 s0, s2, 5 // (op 15664) 
s_mul_i32 s1, 3, s0 // (op 15664) 
s_add_u32 s1, s1, 320 // (op 15664) 

v_add_u32 v3, v11, s1 // (op 15664) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15664) Load value

v_accvgpr_read v6, a172 // (op 15670) 
v_accvgpr_read v7, a173 // (op 15670) 
v_accvgpr_read v8, a174 // (op 15670) 
v_accvgpr_read v9, a175 // (op 15670) 
v_mul_f32 v14, v13, v6 // (op 15670) 
v_mul_f32 v15, v13, v7 // (op 15670) 
v_mul_f32 v16, v13, v8 // (op 15670) 
v_mul_f32 v17, v13, v9 // (op 15670) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15661) 
v_lshrrev_b32 v7, 16, v4 // (op 15661) 
v_cvt_f32_f16 v7, v7 // (op 15661) 
v_mul_f32 v18, v0, v6 // (op 15661) 
v_mul_f32 v19, v0, v7 // (op 15661) 
v_cvt_f32_f16 v6, v5 // (op 15661) 
v_lshrrev_b32 v7, 16, v5 // (op 15661) 
v_cvt_f32_f16 v7, v7 // (op 15661) 
v_mul_f32 v20, v0, v6 // (op 15661) 
v_mul_f32 v21, v0, v7 // (op 15661) 


v_add_f32 v4, v18, v14 // (op 15662) 
v_add_f32 v5, v19, v15 // (op 15662) 
v_add_f32 v6, v20, v16 // (op 15662) 
v_add_f32 v7, v21, v17 // (op 15662) 


v_cvt_f16_f32 v14, v4 // (op 15663) 
v_cvt_f16_f32 v15, v5 // (op 15663) 
v_cvt_f16_f32 v16, v6 // (op 15663) 
v_cvt_f16_f32 v17, v7 // (op 15663) 


v_pack_b32_f16 v4, v14, v15 // (op 15646) 
v_pack_b32_f16 v5, v16, v17 // (op 15646) 

s_lshl_b32 s0, s4, 5 // (op 15646) 
s_mul_i32 s1, 3, s0 // (op 15646) 
s_add_u32 s1, s1, 320 // (op 15646) 

v_add_u32 v3, v12, s1 // (op 15646) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15646) Store value





v_add_u32 v3, 288, v11 // (op 15562) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15562) Load value

v_accvgpr_read v6, a144 // (op 15549) 
v_accvgpr_read v7, a145 // (op 15549) 
v_accvgpr_read v8, a146 // (op 15549) 
v_accvgpr_read v9, a147 // (op 15549) 
v_mul_f32 v14, v13, v6 // (op 15549) 
v_mul_f32 v15, v13, v7 // (op 15549) 
v_mul_f32 v16, v13, v8 // (op 15549) 
v_mul_f32 v17, v13, v9 // (op 15549) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15561) 
v_lshrrev_b32 v7, 16, v4 // (op 15561) 
v_cvt_f32_f16 v7, v7 // (op 15561) 
v_mul_f32 v18, v0, v6 // (op 15561) 
v_mul_f32 v19, v0, v7 // (op 15561) 
v_cvt_f32_f16 v6, v5 // (op 15561) 
v_lshrrev_b32 v7, 16, v5 // (op 15561) 
v_cvt_f32_f16 v7, v7 // (op 15561) 
v_mul_f32 v20, v0, v6 // (op 15561) 
v_mul_f32 v21, v0, v7 // (op 15561) 


v_add_f32 v4, v18, v14 // (op 15550) 
v_add_f32 v5, v19, v15 // (op 15550) 
v_add_f32 v6, v20, v16 // (op 15550) 
v_add_f32 v7, v21, v17 // (op 15550) 


v_cvt_f16_f32 v14, v4 // (op 15551) 
v_cvt_f16_f32 v15, v5 // (op 15551) 
v_cvt_f16_f32 v16, v6 // (op 15551) 
v_cvt_f16_f32 v17, v7 // (op 15551) 


v_pack_b32_f16 v4, v14, v15 // (op 15553) 
v_pack_b32_f16 v5, v16, v17 // (op 15553) 

v_add_u32 v3, 288, v12 // (op 15553) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15553) Store value





s_lshl_b32 s0, s2, 5 // (op 15567) 
s_add_u32 s0, s0, 288 // (op 15567) 
v_add_u32 v3, v11, s0 // (op 15567) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15567) Load value

v_accvgpr_read v6, a148 // (op 15579) 
v_accvgpr_read v7, a149 // (op 15579) 
v_accvgpr_read v8, a150 // (op 15579) 
v_accvgpr_read v9, a151 // (op 15579) 
v_mul_f32 v14, v13, v6 // (op 15579) 
v_mul_f32 v15, v13, v7 // (op 15579) 
v_mul_f32 v16, v13, v8 // (op 15579) 
v_mul_f32 v17, v13, v9 // (op 15579) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15564) 
v_lshrrev_b32 v7, 16, v4 // (op 15564) 
v_cvt_f32_f16 v7, v7 // (op 15564) 
v_mul_f32 v18, v0, v6 // (op 15564) 
v_mul_f32 v19, v0, v7 // (op 15564) 
v_cvt_f32_f16 v6, v5 // (op 15564) 
v_lshrrev_b32 v7, 16, v5 // (op 15564) 
v_cvt_f32_f16 v7, v7 // (op 15564) 
v_mul_f32 v20, v0, v6 // (op 15564) 
v_mul_f32 v21, v0, v7 // (op 15564) 


v_add_f32 v4, v18, v14 // (op 15565) 
v_add_f32 v5, v19, v15 // (op 15565) 
v_add_f32 v6, v20, v16 // (op 15565) 
v_add_f32 v7, v21, v17 // (op 15565) 


v_cvt_f16_f32 v14, v4 // (op 15566) 
v_cvt_f16_f32 v15, v5 // (op 15566) 
v_cvt_f16_f32 v16, v6 // (op 15566) 
v_cvt_f16_f32 v17, v7 // (op 15566) 


v_pack_b32_f16 v4, v14, v15 // (op 15555) 
v_pack_b32_f16 v5, v16, v17 // (op 15555) 

s_lshl_b32 s0, s4, 5 // (op 15555) 
s_add_u32 s0, s0, 288 // (op 15555) 
v_add_u32 v3, v12, s0 // (op 15555) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15555) Store value





s_lshl_b32 s0, s2, 6 // (op 15572) 
s_add_u32 s0, s0, 288 // (op 15572) 
v_add_u32 v3, v11, s0 // (op 15572) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15572) Load value

v_accvgpr_read v6, a152 // (op 15581) 
v_accvgpr_read v7, a153 // (op 15581) 
v_accvgpr_read v8, a154 // (op 15581) 
v_accvgpr_read v9, a155 // (op 15581) 
v_mul_f32 v14, v13, v6 // (op 15581) 
v_mul_f32 v15, v13, v7 // (op 15581) 
v_mul_f32 v16, v13, v8 // (op 15581) 
v_mul_f32 v17, v13, v9 // (op 15581) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15569) 
v_lshrrev_b32 v7, 16, v4 // (op 15569) 
v_cvt_f32_f16 v7, v7 // (op 15569) 
v_mul_f32 v18, v0, v6 // (op 15569) 
v_mul_f32 v19, v0, v7 // (op 15569) 
v_cvt_f32_f16 v6, v5 // (op 15569) 
v_lshrrev_b32 v7, 16, v5 // (op 15569) 
v_cvt_f32_f16 v7, v7 // (op 15569) 
v_mul_f32 v20, v0, v6 // (op 15569) 
v_mul_f32 v21, v0, v7 // (op 15569) 


v_add_f32 v4, v18, v14 // (op 15570) 
v_add_f32 v5, v19, v15 // (op 15570) 
v_add_f32 v6, v20, v16 // (op 15570) 
v_add_f32 v7, v21, v17 // (op 15570) 


v_cvt_f16_f32 v14, v4 // (op 15571) 
v_cvt_f16_f32 v15, v5 // (op 15571) 
v_cvt_f16_f32 v16, v6 // (op 15571) 
v_cvt_f16_f32 v17, v7 // (op 15571) 


v_pack_b32_f16 v4, v14, v15 // (op 15557) 
v_pack_b32_f16 v5, v16, v17 // (op 15557) 

s_lshl_b32 s0, s4, 6 // (op 15557) 
s_add_u32 s0, s0, 288 // (op 15557) 
v_add_u32 v3, v12, s0 // (op 15557) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15557) Store value





s_lshl_b32 s0, s2, 5 // (op 15577) 
s_mul_i32 s1, 3, s0 // (op 15577) 
s_add_u32 s1, s1, 288 // (op 15577) 

v_add_u32 v3, v11, s1 // (op 15577) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15577) Load value

v_accvgpr_read v6, a156 // (op 15583) 
v_accvgpr_read v7, a157 // (op 15583) 
v_accvgpr_read v8, a158 // (op 15583) 
v_accvgpr_read v9, a159 // (op 15583) 
v_mul_f32 v14, v13, v6 // (op 15583) 
v_mul_f32 v15, v13, v7 // (op 15583) 
v_mul_f32 v16, v13, v8 // (op 15583) 
v_mul_f32 v17, v13, v9 // (op 15583) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15574) 
v_lshrrev_b32 v7, 16, v4 // (op 15574) 
v_cvt_f32_f16 v7, v7 // (op 15574) 
v_mul_f32 v18, v0, v6 // (op 15574) 
v_mul_f32 v19, v0, v7 // (op 15574) 
v_cvt_f32_f16 v6, v5 // (op 15574) 
v_lshrrev_b32 v7, 16, v5 // (op 15574) 
v_cvt_f32_f16 v7, v7 // (op 15574) 
v_mul_f32 v20, v0, v6 // (op 15574) 
v_mul_f32 v21, v0, v7 // (op 15574) 


v_add_f32 v4, v18, v14 // (op 15575) 
v_add_f32 v5, v19, v15 // (op 15575) 
v_add_f32 v6, v20, v16 // (op 15575) 
v_add_f32 v7, v21, v17 // (op 15575) 


v_cvt_f16_f32 v14, v4 // (op 15576) 
v_cvt_f16_f32 v15, v5 // (op 15576) 
v_cvt_f16_f32 v16, v6 // (op 15576) 
v_cvt_f16_f32 v17, v7 // (op 15576) 


v_pack_b32_f16 v4, v14, v15 // (op 15559) 
v_pack_b32_f16 v5, v16, v17 // (op 15559) 

s_lshl_b32 s0, s4, 5 // (op 15559) 
s_mul_i32 s1, 3, s0 // (op 15559) 
s_add_u32 s1, s1, 288 // (op 15559) 

v_add_u32 v3, v12, s1 // (op 15559) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15559) Store value





v_add_u32 v3, 256, v11 // (op 15475) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15475) Load value

v_accvgpr_read v6, a128 // (op 15462) 
v_accvgpr_read v7, a129 // (op 15462) 
v_accvgpr_read v8, a130 // (op 15462) 
v_accvgpr_read v9, a131 // (op 15462) 
v_mul_f32 v14, v13, v6 // (op 15462) 
v_mul_f32 v15, v13, v7 // (op 15462) 
v_mul_f32 v16, v13, v8 // (op 15462) 
v_mul_f32 v17, v13, v9 // (op 15462) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15474) 
v_lshrrev_b32 v7, 16, v4 // (op 15474) 
v_cvt_f32_f16 v7, v7 // (op 15474) 
v_mul_f32 v18, v0, v6 // (op 15474) 
v_mul_f32 v19, v0, v7 // (op 15474) 
v_cvt_f32_f16 v6, v5 // (op 15474) 
v_lshrrev_b32 v7, 16, v5 // (op 15474) 
v_cvt_f32_f16 v7, v7 // (op 15474) 
v_mul_f32 v20, v0, v6 // (op 15474) 
v_mul_f32 v21, v0, v7 // (op 15474) 


v_add_f32 v4, v18, v14 // (op 15463) 
v_add_f32 v5, v19, v15 // (op 15463) 
v_add_f32 v6, v20, v16 // (op 15463) 
v_add_f32 v7, v21, v17 // (op 15463) 


v_cvt_f16_f32 v14, v4 // (op 15464) 
v_cvt_f16_f32 v15, v5 // (op 15464) 
v_cvt_f16_f32 v16, v6 // (op 15464) 
v_cvt_f16_f32 v17, v7 // (op 15464) 


v_pack_b32_f16 v4, v14, v15 // (op 15466) 
v_pack_b32_f16 v5, v16, v17 // (op 15466) 

v_add_u32 v3, 256, v12 // (op 15466) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15466) Store value





s_lshl_b32 s0, s2, 5 // (op 15480) 
s_add_u32 s0, s0, 256 // (op 15480) 
v_add_u32 v3, v11, s0 // (op 15480) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15480) Load value

v_accvgpr_read v6, a132 // (op 15492) 
v_accvgpr_read v7, a133 // (op 15492) 
v_accvgpr_read v8, a134 // (op 15492) 
v_accvgpr_read v9, a135 // (op 15492) 
v_mul_f32 v14, v13, v6 // (op 15492) 
v_mul_f32 v15, v13, v7 // (op 15492) 
v_mul_f32 v16, v13, v8 // (op 15492) 
v_mul_f32 v17, v13, v9 // (op 15492) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15477) 
v_lshrrev_b32 v7, 16, v4 // (op 15477) 
v_cvt_f32_f16 v7, v7 // (op 15477) 
v_mul_f32 v18, v0, v6 // (op 15477) 
v_mul_f32 v19, v0, v7 // (op 15477) 
v_cvt_f32_f16 v6, v5 // (op 15477) 
v_lshrrev_b32 v7, 16, v5 // (op 15477) 
v_cvt_f32_f16 v7, v7 // (op 15477) 
v_mul_f32 v20, v0, v6 // (op 15477) 
v_mul_f32 v21, v0, v7 // (op 15477) 


v_add_f32 v4, v18, v14 // (op 15478) 
v_add_f32 v5, v19, v15 // (op 15478) 
v_add_f32 v6, v20, v16 // (op 15478) 
v_add_f32 v7, v21, v17 // (op 15478) 


v_cvt_f16_f32 v14, v4 // (op 15479) 
v_cvt_f16_f32 v15, v5 // (op 15479) 
v_cvt_f16_f32 v16, v6 // (op 15479) 
v_cvt_f16_f32 v17, v7 // (op 15479) 


v_pack_b32_f16 v4, v14, v15 // (op 15468) 
v_pack_b32_f16 v5, v16, v17 // (op 15468) 

s_lshl_b32 s0, s4, 5 // (op 15468) 
s_add_u32 s0, s0, 256 // (op 15468) 
v_add_u32 v3, v12, s0 // (op 15468) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15468) Store value





s_lshl_b32 s0, s2, 6 // (op 15485) 
s_add_u32 s0, s0, 256 // (op 15485) 
v_add_u32 v3, v11, s0 // (op 15485) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15485) Load value

v_accvgpr_read v6, a136 // (op 15494) 
v_accvgpr_read v7, a137 // (op 15494) 
v_accvgpr_read v8, a138 // (op 15494) 
v_accvgpr_read v9, a139 // (op 15494) 
v_mul_f32 v14, v13, v6 // (op 15494) 
v_mul_f32 v15, v13, v7 // (op 15494) 
v_mul_f32 v16, v13, v8 // (op 15494) 
v_mul_f32 v17, v13, v9 // (op 15494) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15482) 
v_lshrrev_b32 v7, 16, v4 // (op 15482) 
v_cvt_f32_f16 v7, v7 // (op 15482) 
v_mul_f32 v18, v0, v6 // (op 15482) 
v_mul_f32 v19, v0, v7 // (op 15482) 
v_cvt_f32_f16 v6, v5 // (op 15482) 
v_lshrrev_b32 v7, 16, v5 // (op 15482) 
v_cvt_f32_f16 v7, v7 // (op 15482) 
v_mul_f32 v20, v0, v6 // (op 15482) 
v_mul_f32 v21, v0, v7 // (op 15482) 


v_add_f32 v4, v18, v14 // (op 15483) 
v_add_f32 v5, v19, v15 // (op 15483) 
v_add_f32 v6, v20, v16 // (op 15483) 
v_add_f32 v7, v21, v17 // (op 15483) 


v_cvt_f16_f32 v14, v4 // (op 15484) 
v_cvt_f16_f32 v15, v5 // (op 15484) 
v_cvt_f16_f32 v16, v6 // (op 15484) 
v_cvt_f16_f32 v17, v7 // (op 15484) 


v_pack_b32_f16 v4, v14, v15 // (op 15470) 
v_pack_b32_f16 v5, v16, v17 // (op 15470) 

s_lshl_b32 s0, s4, 6 // (op 15470) 
s_add_u32 s0, s0, 256 // (op 15470) 
v_add_u32 v3, v12, s0 // (op 15470) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15470) Store value





s_lshl_b32 s0, s2, 5 // (op 15490) 
s_mul_i32 s1, 3, s0 // (op 15490) 
s_add_u32 s1, s1, 256 // (op 15490) 

v_add_u32 v3, v11, s1 // (op 15490) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15490) Load value

v_accvgpr_read v6, a140 // (op 15496) 
v_accvgpr_read v7, a141 // (op 15496) 
v_accvgpr_read v8, a142 // (op 15496) 
v_accvgpr_read v9, a143 // (op 15496) 
v_mul_f32 v14, v13, v6 // (op 15496) 
v_mul_f32 v15, v13, v7 // (op 15496) 
v_mul_f32 v16, v13, v8 // (op 15496) 
v_mul_f32 v17, v13, v9 // (op 15496) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15487) 
v_lshrrev_b32 v7, 16, v4 // (op 15487) 
v_cvt_f32_f16 v7, v7 // (op 15487) 
v_mul_f32 v18, v0, v6 // (op 15487) 
v_mul_f32 v19, v0, v7 // (op 15487) 
v_cvt_f32_f16 v6, v5 // (op 15487) 
v_lshrrev_b32 v7, 16, v5 // (op 15487) 
v_cvt_f32_f16 v7, v7 // (op 15487) 
v_mul_f32 v20, v0, v6 // (op 15487) 
v_mul_f32 v21, v0, v7 // (op 15487) 


v_add_f32 v4, v18, v14 // (op 15488) 
v_add_f32 v5, v19, v15 // (op 15488) 
v_add_f32 v6, v20, v16 // (op 15488) 
v_add_f32 v7, v21, v17 // (op 15488) 


v_cvt_f16_f32 v14, v4 // (op 15489) 
v_cvt_f16_f32 v15, v5 // (op 15489) 
v_cvt_f16_f32 v16, v6 // (op 15489) 
v_cvt_f16_f32 v17, v7 // (op 15489) 


v_pack_b32_f16 v4, v14, v15 // (op 15472) 
v_pack_b32_f16 v5, v16, v17 // (op 15472) 

s_lshl_b32 s0, s4, 5 // (op 15472) 
s_mul_i32 s1, 3, s0 // (op 15472) 
s_add_u32 s1, s1, 256 // (op 15472) 

v_add_u32 v3, v12, s1 // (op 15472) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15472) Store value





v_add_u32 v3, 224, v11 // (op 15388) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15388) Load value

v_accvgpr_read v6, a112 // (op 15375) 
v_accvgpr_read v7, a113 // (op 15375) 
v_accvgpr_read v8, a114 // (op 15375) 
v_accvgpr_read v9, a115 // (op 15375) 
v_mul_f32 v14, v13, v6 // (op 15375) 
v_mul_f32 v15, v13, v7 // (op 15375) 
v_mul_f32 v16, v13, v8 // (op 15375) 
v_mul_f32 v17, v13, v9 // (op 15375) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15387) 
v_lshrrev_b32 v7, 16, v4 // (op 15387) 
v_cvt_f32_f16 v7, v7 // (op 15387) 
v_mul_f32 v18, v0, v6 // (op 15387) 
v_mul_f32 v19, v0, v7 // (op 15387) 
v_cvt_f32_f16 v6, v5 // (op 15387) 
v_lshrrev_b32 v7, 16, v5 // (op 15387) 
v_cvt_f32_f16 v7, v7 // (op 15387) 
v_mul_f32 v20, v0, v6 // (op 15387) 
v_mul_f32 v21, v0, v7 // (op 15387) 


v_add_f32 v4, v18, v14 // (op 15376) 
v_add_f32 v5, v19, v15 // (op 15376) 
v_add_f32 v6, v20, v16 // (op 15376) 
v_add_f32 v7, v21, v17 // (op 15376) 


v_cvt_f16_f32 v14, v4 // (op 15377) 
v_cvt_f16_f32 v15, v5 // (op 15377) 
v_cvt_f16_f32 v16, v6 // (op 15377) 
v_cvt_f16_f32 v17, v7 // (op 15377) 


v_pack_b32_f16 v4, v14, v15 // (op 15379) 
v_pack_b32_f16 v5, v16, v17 // (op 15379) 

v_add_u32 v3, 224, v12 // (op 15379) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15379) Store value





s_lshl_b32 s0, s2, 5 // (op 15393) 
s_add_u32 s0, s0, 224 // (op 15393) 
v_add_u32 v3, v11, s0 // (op 15393) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15393) Load value

v_accvgpr_read v6, a116 // (op 15405) 
v_accvgpr_read v7, a117 // (op 15405) 
v_accvgpr_read v8, a118 // (op 15405) 
v_accvgpr_read v9, a119 // (op 15405) 
v_mul_f32 v14, v13, v6 // (op 15405) 
v_mul_f32 v15, v13, v7 // (op 15405) 
v_mul_f32 v16, v13, v8 // (op 15405) 
v_mul_f32 v17, v13, v9 // (op 15405) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15390) 
v_lshrrev_b32 v7, 16, v4 // (op 15390) 
v_cvt_f32_f16 v7, v7 // (op 15390) 
v_mul_f32 v18, v0, v6 // (op 15390) 
v_mul_f32 v19, v0, v7 // (op 15390) 
v_cvt_f32_f16 v6, v5 // (op 15390) 
v_lshrrev_b32 v7, 16, v5 // (op 15390) 
v_cvt_f32_f16 v7, v7 // (op 15390) 
v_mul_f32 v20, v0, v6 // (op 15390) 
v_mul_f32 v21, v0, v7 // (op 15390) 


v_add_f32 v4, v18, v14 // (op 15391) 
v_add_f32 v5, v19, v15 // (op 15391) 
v_add_f32 v6, v20, v16 // (op 15391) 
v_add_f32 v7, v21, v17 // (op 15391) 


v_cvt_f16_f32 v14, v4 // (op 15392) 
v_cvt_f16_f32 v15, v5 // (op 15392) 
v_cvt_f16_f32 v16, v6 // (op 15392) 
v_cvt_f16_f32 v17, v7 // (op 15392) 


v_pack_b32_f16 v4, v14, v15 // (op 15381) 
v_pack_b32_f16 v5, v16, v17 // (op 15381) 

s_lshl_b32 s0, s4, 5 // (op 15381) 
s_add_u32 s0, s0, 224 // (op 15381) 
v_add_u32 v3, v12, s0 // (op 15381) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15381) Store value





s_lshl_b32 s0, s2, 6 // (op 15398) 
s_add_u32 s0, s0, 224 // (op 15398) 
v_add_u32 v3, v11, s0 // (op 15398) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15398) Load value

v_accvgpr_read v6, a120 // (op 15407) 
v_accvgpr_read v7, a121 // (op 15407) 
v_accvgpr_read v8, a122 // (op 15407) 
v_accvgpr_read v9, a123 // (op 15407) 
v_mul_f32 v14, v13, v6 // (op 15407) 
v_mul_f32 v15, v13, v7 // (op 15407) 
v_mul_f32 v16, v13, v8 // (op 15407) 
v_mul_f32 v17, v13, v9 // (op 15407) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15395) 
v_lshrrev_b32 v7, 16, v4 // (op 15395) 
v_cvt_f32_f16 v7, v7 // (op 15395) 
v_mul_f32 v18, v0, v6 // (op 15395) 
v_mul_f32 v19, v0, v7 // (op 15395) 
v_cvt_f32_f16 v6, v5 // (op 15395) 
v_lshrrev_b32 v7, 16, v5 // (op 15395) 
v_cvt_f32_f16 v7, v7 // (op 15395) 
v_mul_f32 v20, v0, v6 // (op 15395) 
v_mul_f32 v21, v0, v7 // (op 15395) 


v_add_f32 v4, v18, v14 // (op 15396) 
v_add_f32 v5, v19, v15 // (op 15396) 
v_add_f32 v6, v20, v16 // (op 15396) 
v_add_f32 v7, v21, v17 // (op 15396) 


v_cvt_f16_f32 v14, v4 // (op 15397) 
v_cvt_f16_f32 v15, v5 // (op 15397) 
v_cvt_f16_f32 v16, v6 // (op 15397) 
v_cvt_f16_f32 v17, v7 // (op 15397) 


v_pack_b32_f16 v4, v14, v15 // (op 15383) 
v_pack_b32_f16 v5, v16, v17 // (op 15383) 

s_lshl_b32 s0, s4, 6 // (op 15383) 
s_add_u32 s0, s0, 224 // (op 15383) 
v_add_u32 v3, v12, s0 // (op 15383) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15383) Store value





s_lshl_b32 s0, s2, 5 // (op 15403) 
s_mul_i32 s1, 3, s0 // (op 15403) 
s_add_u32 s1, s1, 224 // (op 15403) 

v_add_u32 v3, v11, s1 // (op 15403) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15403) Load value

v_accvgpr_read v6, a124 // (op 15409) 
v_accvgpr_read v7, a125 // (op 15409) 
v_accvgpr_read v8, a126 // (op 15409) 
v_accvgpr_read v9, a127 // (op 15409) 
v_mul_f32 v14, v13, v6 // (op 15409) 
v_mul_f32 v15, v13, v7 // (op 15409) 
v_mul_f32 v16, v13, v8 // (op 15409) 
v_mul_f32 v17, v13, v9 // (op 15409) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15400) 
v_lshrrev_b32 v7, 16, v4 // (op 15400) 
v_cvt_f32_f16 v7, v7 // (op 15400) 
v_mul_f32 v18, v0, v6 // (op 15400) 
v_mul_f32 v19, v0, v7 // (op 15400) 
v_cvt_f32_f16 v6, v5 // (op 15400) 
v_lshrrev_b32 v7, 16, v5 // (op 15400) 
v_cvt_f32_f16 v7, v7 // (op 15400) 
v_mul_f32 v20, v0, v6 // (op 15400) 
v_mul_f32 v21, v0, v7 // (op 15400) 


v_add_f32 v4, v18, v14 // (op 15401) 
v_add_f32 v5, v19, v15 // (op 15401) 
v_add_f32 v6, v20, v16 // (op 15401) 
v_add_f32 v7, v21, v17 // (op 15401) 


v_cvt_f16_f32 v14, v4 // (op 15402) 
v_cvt_f16_f32 v15, v5 // (op 15402) 
v_cvt_f16_f32 v16, v6 // (op 15402) 
v_cvt_f16_f32 v17, v7 // (op 15402) 


v_pack_b32_f16 v4, v14, v15 // (op 15385) 
v_pack_b32_f16 v5, v16, v17 // (op 15385) 

s_lshl_b32 s0, s4, 5 // (op 15385) 
s_mul_i32 s1, 3, s0 // (op 15385) 
s_add_u32 s1, s1, 224 // (op 15385) 

v_add_u32 v3, v12, s1 // (op 15385) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15385) Store value





v_add_u32 v3, 192, v11 // (op 15301) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15301) Load value

v_accvgpr_read v6, a96 // (op 15288) 
v_accvgpr_read v7, a97 // (op 15288) 
v_accvgpr_read v8, a98 // (op 15288) 
v_accvgpr_read v9, a99 // (op 15288) 
v_mul_f32 v14, v13, v6 // (op 15288) 
v_mul_f32 v15, v13, v7 // (op 15288) 
v_mul_f32 v16, v13, v8 // (op 15288) 
v_mul_f32 v17, v13, v9 // (op 15288) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15300) 
v_lshrrev_b32 v7, 16, v4 // (op 15300) 
v_cvt_f32_f16 v7, v7 // (op 15300) 
v_mul_f32 v18, v0, v6 // (op 15300) 
v_mul_f32 v19, v0, v7 // (op 15300) 
v_cvt_f32_f16 v6, v5 // (op 15300) 
v_lshrrev_b32 v7, 16, v5 // (op 15300) 
v_cvt_f32_f16 v7, v7 // (op 15300) 
v_mul_f32 v20, v0, v6 // (op 15300) 
v_mul_f32 v21, v0, v7 // (op 15300) 


v_add_f32 v4, v18, v14 // (op 15289) 
v_add_f32 v5, v19, v15 // (op 15289) 
v_add_f32 v6, v20, v16 // (op 15289) 
v_add_f32 v7, v21, v17 // (op 15289) 


v_cvt_f16_f32 v14, v4 // (op 15290) 
v_cvt_f16_f32 v15, v5 // (op 15290) 
v_cvt_f16_f32 v16, v6 // (op 15290) 
v_cvt_f16_f32 v17, v7 // (op 15290) 


v_pack_b32_f16 v4, v14, v15 // (op 15292) 
v_pack_b32_f16 v5, v16, v17 // (op 15292) 

v_add_u32 v3, 192, v12 // (op 15292) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15292) Store value





s_lshl_b32 s0, s2, 5 // (op 15306) 
s_add_u32 s0, s0, 192 // (op 15306) 
v_add_u32 v3, v11, s0 // (op 15306) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15306) Load value

v_accvgpr_read v6, a100 // (op 15318) 
v_accvgpr_read v7, a101 // (op 15318) 
v_accvgpr_read v8, a102 // (op 15318) 
v_accvgpr_read v9, a103 // (op 15318) 
v_mul_f32 v14, v13, v6 // (op 15318) 
v_mul_f32 v15, v13, v7 // (op 15318) 
v_mul_f32 v16, v13, v8 // (op 15318) 
v_mul_f32 v17, v13, v9 // (op 15318) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15303) 
v_lshrrev_b32 v7, 16, v4 // (op 15303) 
v_cvt_f32_f16 v7, v7 // (op 15303) 
v_mul_f32 v18, v0, v6 // (op 15303) 
v_mul_f32 v19, v0, v7 // (op 15303) 
v_cvt_f32_f16 v6, v5 // (op 15303) 
v_lshrrev_b32 v7, 16, v5 // (op 15303) 
v_cvt_f32_f16 v7, v7 // (op 15303) 
v_mul_f32 v20, v0, v6 // (op 15303) 
v_mul_f32 v21, v0, v7 // (op 15303) 


v_add_f32 v4, v18, v14 // (op 15304) 
v_add_f32 v5, v19, v15 // (op 15304) 
v_add_f32 v6, v20, v16 // (op 15304) 
v_add_f32 v7, v21, v17 // (op 15304) 


v_cvt_f16_f32 v14, v4 // (op 15305) 
v_cvt_f16_f32 v15, v5 // (op 15305) 
v_cvt_f16_f32 v16, v6 // (op 15305) 
v_cvt_f16_f32 v17, v7 // (op 15305) 


v_pack_b32_f16 v4, v14, v15 // (op 15294) 
v_pack_b32_f16 v5, v16, v17 // (op 15294) 

s_lshl_b32 s0, s4, 5 // (op 15294) 
s_add_u32 s0, s0, 192 // (op 15294) 
v_add_u32 v3, v12, s0 // (op 15294) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15294) Store value





s_lshl_b32 s0, s2, 6 // (op 15311) 
s_add_u32 s0, s0, 192 // (op 15311) 
v_add_u32 v3, v11, s0 // (op 15311) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15311) Load value

v_accvgpr_read v6, a104 // (op 15320) 
v_accvgpr_read v7, a105 // (op 15320) 
v_accvgpr_read v8, a106 // (op 15320) 
v_accvgpr_read v9, a107 // (op 15320) 
v_mul_f32 v14, v13, v6 // (op 15320) 
v_mul_f32 v15, v13, v7 // (op 15320) 
v_mul_f32 v16, v13, v8 // (op 15320) 
v_mul_f32 v17, v13, v9 // (op 15320) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15308) 
v_lshrrev_b32 v7, 16, v4 // (op 15308) 
v_cvt_f32_f16 v7, v7 // (op 15308) 
v_mul_f32 v18, v0, v6 // (op 15308) 
v_mul_f32 v19, v0, v7 // (op 15308) 
v_cvt_f32_f16 v6, v5 // (op 15308) 
v_lshrrev_b32 v7, 16, v5 // (op 15308) 
v_cvt_f32_f16 v7, v7 // (op 15308) 
v_mul_f32 v20, v0, v6 // (op 15308) 
v_mul_f32 v21, v0, v7 // (op 15308) 


v_add_f32 v4, v18, v14 // (op 15309) 
v_add_f32 v5, v19, v15 // (op 15309) 
v_add_f32 v6, v20, v16 // (op 15309) 
v_add_f32 v7, v21, v17 // (op 15309) 


v_cvt_f16_f32 v14, v4 // (op 15310) 
v_cvt_f16_f32 v15, v5 // (op 15310) 
v_cvt_f16_f32 v16, v6 // (op 15310) 
v_cvt_f16_f32 v17, v7 // (op 15310) 


v_pack_b32_f16 v4, v14, v15 // (op 15296) 
v_pack_b32_f16 v5, v16, v17 // (op 15296) 

s_lshl_b32 s0, s4, 6 // (op 15296) 
s_add_u32 s0, s0, 192 // (op 15296) 
v_add_u32 v3, v12, s0 // (op 15296) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15296) Store value





s_lshl_b32 s0, s2, 5 // (op 15316) 
s_mul_i32 s1, 3, s0 // (op 15316) 
s_add_u32 s1, s1, 192 // (op 15316) 

v_add_u32 v3, v11, s1 // (op 15316) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15316) Load value

v_accvgpr_read v6, a108 // (op 15322) 
v_accvgpr_read v7, a109 // (op 15322) 
v_accvgpr_read v8, a110 // (op 15322) 
v_accvgpr_read v9, a111 // (op 15322) 
v_mul_f32 v14, v13, v6 // (op 15322) 
v_mul_f32 v15, v13, v7 // (op 15322) 
v_mul_f32 v16, v13, v8 // (op 15322) 
v_mul_f32 v17, v13, v9 // (op 15322) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15313) 
v_lshrrev_b32 v7, 16, v4 // (op 15313) 
v_cvt_f32_f16 v7, v7 // (op 15313) 
v_mul_f32 v18, v0, v6 // (op 15313) 
v_mul_f32 v19, v0, v7 // (op 15313) 
v_cvt_f32_f16 v6, v5 // (op 15313) 
v_lshrrev_b32 v7, 16, v5 // (op 15313) 
v_cvt_f32_f16 v7, v7 // (op 15313) 
v_mul_f32 v20, v0, v6 // (op 15313) 
v_mul_f32 v21, v0, v7 // (op 15313) 


v_add_f32 v4, v18, v14 // (op 15314) 
v_add_f32 v5, v19, v15 // (op 15314) 
v_add_f32 v6, v20, v16 // (op 15314) 
v_add_f32 v7, v21, v17 // (op 15314) 


v_cvt_f16_f32 v14, v4 // (op 15315) 
v_cvt_f16_f32 v15, v5 // (op 15315) 
v_cvt_f16_f32 v16, v6 // (op 15315) 
v_cvt_f16_f32 v17, v7 // (op 15315) 


v_pack_b32_f16 v4, v14, v15 // (op 15298) 
v_pack_b32_f16 v5, v16, v17 // (op 15298) 

s_lshl_b32 s0, s4, 5 // (op 15298) 
s_mul_i32 s1, 3, s0 // (op 15298) 
s_add_u32 s1, s1, 192 // (op 15298) 

v_add_u32 v3, v12, s1 // (op 15298) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15298) Store value





v_add_u32 v3, 160, v11 // (op 15214) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15214) Load value

v_accvgpr_read v6, a80 // (op 15201) 
v_accvgpr_read v7, a81 // (op 15201) 
v_accvgpr_read v8, a82 // (op 15201) 
v_accvgpr_read v9, a83 // (op 15201) 
v_mul_f32 v14, v13, v6 // (op 15201) 
v_mul_f32 v15, v13, v7 // (op 15201) 
v_mul_f32 v16, v13, v8 // (op 15201) 
v_mul_f32 v17, v13, v9 // (op 15201) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15213) 
v_lshrrev_b32 v7, 16, v4 // (op 15213) 
v_cvt_f32_f16 v7, v7 // (op 15213) 
v_mul_f32 v18, v0, v6 // (op 15213) 
v_mul_f32 v19, v0, v7 // (op 15213) 
v_cvt_f32_f16 v6, v5 // (op 15213) 
v_lshrrev_b32 v7, 16, v5 // (op 15213) 
v_cvt_f32_f16 v7, v7 // (op 15213) 
v_mul_f32 v20, v0, v6 // (op 15213) 
v_mul_f32 v21, v0, v7 // (op 15213) 


v_add_f32 v4, v18, v14 // (op 15202) 
v_add_f32 v5, v19, v15 // (op 15202) 
v_add_f32 v6, v20, v16 // (op 15202) 
v_add_f32 v7, v21, v17 // (op 15202) 


v_cvt_f16_f32 v14, v4 // (op 15203) 
v_cvt_f16_f32 v15, v5 // (op 15203) 
v_cvt_f16_f32 v16, v6 // (op 15203) 
v_cvt_f16_f32 v17, v7 // (op 15203) 


v_pack_b32_f16 v4, v14, v15 // (op 15205) 
v_pack_b32_f16 v5, v16, v17 // (op 15205) 

v_add_u32 v3, 160, v12 // (op 15205) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15205) Store value





s_lshl_b32 s0, s2, 5 // (op 15219) 
s_add_u32 s0, s0, 160 // (op 15219) 
v_add_u32 v3, v11, s0 // (op 15219) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15219) Load value

v_accvgpr_read v6, a84 // (op 15231) 
v_accvgpr_read v7, a85 // (op 15231) 
v_accvgpr_read v8, a86 // (op 15231) 
v_accvgpr_read v9, a87 // (op 15231) 
v_mul_f32 v14, v13, v6 // (op 15231) 
v_mul_f32 v15, v13, v7 // (op 15231) 
v_mul_f32 v16, v13, v8 // (op 15231) 
v_mul_f32 v17, v13, v9 // (op 15231) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15216) 
v_lshrrev_b32 v7, 16, v4 // (op 15216) 
v_cvt_f32_f16 v7, v7 // (op 15216) 
v_mul_f32 v18, v0, v6 // (op 15216) 
v_mul_f32 v19, v0, v7 // (op 15216) 
v_cvt_f32_f16 v6, v5 // (op 15216) 
v_lshrrev_b32 v7, 16, v5 // (op 15216) 
v_cvt_f32_f16 v7, v7 // (op 15216) 
v_mul_f32 v20, v0, v6 // (op 15216) 
v_mul_f32 v21, v0, v7 // (op 15216) 


v_add_f32 v4, v18, v14 // (op 15217) 
v_add_f32 v5, v19, v15 // (op 15217) 
v_add_f32 v6, v20, v16 // (op 15217) 
v_add_f32 v7, v21, v17 // (op 15217) 


v_cvt_f16_f32 v14, v4 // (op 15218) 
v_cvt_f16_f32 v15, v5 // (op 15218) 
v_cvt_f16_f32 v16, v6 // (op 15218) 
v_cvt_f16_f32 v17, v7 // (op 15218) 


v_pack_b32_f16 v4, v14, v15 // (op 15207) 
v_pack_b32_f16 v5, v16, v17 // (op 15207) 

s_lshl_b32 s0, s4, 5 // (op 15207) 
s_add_u32 s0, s0, 160 // (op 15207) 
v_add_u32 v3, v12, s0 // (op 15207) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15207) Store value





s_lshl_b32 s0, s2, 6 // (op 15224) 
s_add_u32 s0, s0, 160 // (op 15224) 
v_add_u32 v3, v11, s0 // (op 15224) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15224) Load value

v_accvgpr_read v6, a88 // (op 15233) 
v_accvgpr_read v7, a89 // (op 15233) 
v_accvgpr_read v8, a90 // (op 15233) 
v_accvgpr_read v9, a91 // (op 15233) 
v_mul_f32 v14, v13, v6 // (op 15233) 
v_mul_f32 v15, v13, v7 // (op 15233) 
v_mul_f32 v16, v13, v8 // (op 15233) 
v_mul_f32 v17, v13, v9 // (op 15233) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15221) 
v_lshrrev_b32 v7, 16, v4 // (op 15221) 
v_cvt_f32_f16 v7, v7 // (op 15221) 
v_mul_f32 v18, v0, v6 // (op 15221) 
v_mul_f32 v19, v0, v7 // (op 15221) 
v_cvt_f32_f16 v6, v5 // (op 15221) 
v_lshrrev_b32 v7, 16, v5 // (op 15221) 
v_cvt_f32_f16 v7, v7 // (op 15221) 
v_mul_f32 v20, v0, v6 // (op 15221) 
v_mul_f32 v21, v0, v7 // (op 15221) 


v_add_f32 v4, v18, v14 // (op 15222) 
v_add_f32 v5, v19, v15 // (op 15222) 
v_add_f32 v6, v20, v16 // (op 15222) 
v_add_f32 v7, v21, v17 // (op 15222) 


v_cvt_f16_f32 v14, v4 // (op 15223) 
v_cvt_f16_f32 v15, v5 // (op 15223) 
v_cvt_f16_f32 v16, v6 // (op 15223) 
v_cvt_f16_f32 v17, v7 // (op 15223) 


v_pack_b32_f16 v4, v14, v15 // (op 15209) 
v_pack_b32_f16 v5, v16, v17 // (op 15209) 

s_lshl_b32 s0, s4, 6 // (op 15209) 
s_add_u32 s0, s0, 160 // (op 15209) 
v_add_u32 v3, v12, s0 // (op 15209) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15209) Store value





s_lshl_b32 s0, s2, 5 // (op 15229) 
s_mul_i32 s1, 3, s0 // (op 15229) 
s_add_u32 s1, s1, 160 // (op 15229) 

v_add_u32 v3, v11, s1 // (op 15229) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15229) Load value

v_accvgpr_read v6, a92 // (op 15235) 
v_accvgpr_read v7, a93 // (op 15235) 
v_accvgpr_read v8, a94 // (op 15235) 
v_accvgpr_read v9, a95 // (op 15235) 
v_mul_f32 v14, v13, v6 // (op 15235) 
v_mul_f32 v15, v13, v7 // (op 15235) 
v_mul_f32 v16, v13, v8 // (op 15235) 
v_mul_f32 v17, v13, v9 // (op 15235) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15226) 
v_lshrrev_b32 v7, 16, v4 // (op 15226) 
v_cvt_f32_f16 v7, v7 // (op 15226) 
v_mul_f32 v18, v0, v6 // (op 15226) 
v_mul_f32 v19, v0, v7 // (op 15226) 
v_cvt_f32_f16 v6, v5 // (op 15226) 
v_lshrrev_b32 v7, 16, v5 // (op 15226) 
v_cvt_f32_f16 v7, v7 // (op 15226) 
v_mul_f32 v20, v0, v6 // (op 15226) 
v_mul_f32 v21, v0, v7 // (op 15226) 


v_add_f32 v4, v18, v14 // (op 15227) 
v_add_f32 v5, v19, v15 // (op 15227) 
v_add_f32 v6, v20, v16 // (op 15227) 
v_add_f32 v7, v21, v17 // (op 15227) 


v_cvt_f16_f32 v14, v4 // (op 15228) 
v_cvt_f16_f32 v15, v5 // (op 15228) 
v_cvt_f16_f32 v16, v6 // (op 15228) 
v_cvt_f16_f32 v17, v7 // (op 15228) 


v_pack_b32_f16 v4, v14, v15 // (op 15211) 
v_pack_b32_f16 v5, v16, v17 // (op 15211) 

s_lshl_b32 s0, s4, 5 // (op 15211) 
s_mul_i32 s1, 3, s0 // (op 15211) 
s_add_u32 s1, s1, 160 // (op 15211) 

v_add_u32 v3, v12, s1 // (op 15211) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15211) Store value





v_add_u32 v3, 96, v11 // (op 15040) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15040) Load value

v_accvgpr_read v6, a48 // (op 15027) 
v_accvgpr_read v7, a49 // (op 15027) 
v_accvgpr_read v8, a50 // (op 15027) 
v_accvgpr_read v9, a51 // (op 15027) 
v_mul_f32 v14, v13, v6 // (op 15027) 
v_mul_f32 v15, v13, v7 // (op 15027) 
v_mul_f32 v16, v13, v8 // (op 15027) 
v_mul_f32 v17, v13, v9 // (op 15027) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15039) 
v_lshrrev_b32 v7, 16, v4 // (op 15039) 
v_cvt_f32_f16 v7, v7 // (op 15039) 
v_mul_f32 v18, v0, v6 // (op 15039) 
v_mul_f32 v19, v0, v7 // (op 15039) 
v_cvt_f32_f16 v6, v5 // (op 15039) 
v_lshrrev_b32 v7, 16, v5 // (op 15039) 
v_cvt_f32_f16 v7, v7 // (op 15039) 
v_mul_f32 v20, v0, v6 // (op 15039) 
v_mul_f32 v21, v0, v7 // (op 15039) 


v_add_f32 v4, v18, v14 // (op 15028) 
v_add_f32 v5, v19, v15 // (op 15028) 
v_add_f32 v6, v20, v16 // (op 15028) 
v_add_f32 v7, v21, v17 // (op 15028) 


v_cvt_f16_f32 v14, v4 // (op 15029) 
v_cvt_f16_f32 v15, v5 // (op 15029) 
v_cvt_f16_f32 v16, v6 // (op 15029) 
v_cvt_f16_f32 v17, v7 // (op 15029) 


v_pack_b32_f16 v4, v14, v15 // (op 15031) 
v_pack_b32_f16 v5, v16, v17 // (op 15031) 

v_add_u32 v3, 96, v12 // (op 15031) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15031) Store value





s_lshl_b32 s0, s2, 5 // (op 15045) 
s_add_u32 s0, s0, 96 // (op 15045) 
v_add_u32 v3, v11, s0 // (op 15045) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15045) Load value

v_accvgpr_read v6, a52 // (op 15057) 
v_accvgpr_read v7, a53 // (op 15057) 
v_accvgpr_read v8, a54 // (op 15057) 
v_accvgpr_read v9, a55 // (op 15057) 
v_mul_f32 v14, v13, v6 // (op 15057) 
v_mul_f32 v15, v13, v7 // (op 15057) 
v_mul_f32 v16, v13, v8 // (op 15057) 
v_mul_f32 v17, v13, v9 // (op 15057) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15042) 
v_lshrrev_b32 v7, 16, v4 // (op 15042) 
v_cvt_f32_f16 v7, v7 // (op 15042) 
v_mul_f32 v18, v0, v6 // (op 15042) 
v_mul_f32 v19, v0, v7 // (op 15042) 
v_cvt_f32_f16 v6, v5 // (op 15042) 
v_lshrrev_b32 v7, 16, v5 // (op 15042) 
v_cvt_f32_f16 v7, v7 // (op 15042) 
v_mul_f32 v20, v0, v6 // (op 15042) 
v_mul_f32 v21, v0, v7 // (op 15042) 


v_add_f32 v4, v18, v14 // (op 15043) 
v_add_f32 v5, v19, v15 // (op 15043) 
v_add_f32 v6, v20, v16 // (op 15043) 
v_add_f32 v7, v21, v17 // (op 15043) 


v_cvt_f16_f32 v14, v4 // (op 15044) 
v_cvt_f16_f32 v15, v5 // (op 15044) 
v_cvt_f16_f32 v16, v6 // (op 15044) 
v_cvt_f16_f32 v17, v7 // (op 15044) 


v_pack_b32_f16 v4, v14, v15 // (op 15033) 
v_pack_b32_f16 v5, v16, v17 // (op 15033) 

s_lshl_b32 s0, s4, 5 // (op 15033) 
s_add_u32 s0, s0, 96 // (op 15033) 
v_add_u32 v3, v12, s0 // (op 15033) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15033) Store value





s_lshl_b32 s0, s2, 6 // (op 15050) 
s_add_u32 s0, s0, 96 // (op 15050) 
v_add_u32 v3, v11, s0 // (op 15050) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15050) Load value

v_accvgpr_read v6, a56 // (op 15059) 
v_accvgpr_read v7, a57 // (op 15059) 
v_accvgpr_read v8, a58 // (op 15059) 
v_accvgpr_read v9, a59 // (op 15059) 
v_mul_f32 v14, v13, v6 // (op 15059) 
v_mul_f32 v15, v13, v7 // (op 15059) 
v_mul_f32 v16, v13, v8 // (op 15059) 
v_mul_f32 v17, v13, v9 // (op 15059) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15047) 
v_lshrrev_b32 v7, 16, v4 // (op 15047) 
v_cvt_f32_f16 v7, v7 // (op 15047) 
v_mul_f32 v18, v0, v6 // (op 15047) 
v_mul_f32 v19, v0, v7 // (op 15047) 
v_cvt_f32_f16 v6, v5 // (op 15047) 
v_lshrrev_b32 v7, 16, v5 // (op 15047) 
v_cvt_f32_f16 v7, v7 // (op 15047) 
v_mul_f32 v20, v0, v6 // (op 15047) 
v_mul_f32 v21, v0, v7 // (op 15047) 


v_add_f32 v4, v18, v14 // (op 15048) 
v_add_f32 v5, v19, v15 // (op 15048) 
v_add_f32 v6, v20, v16 // (op 15048) 
v_add_f32 v7, v21, v17 // (op 15048) 


v_cvt_f16_f32 v14, v4 // (op 15049) 
v_cvt_f16_f32 v15, v5 // (op 15049) 
v_cvt_f16_f32 v16, v6 // (op 15049) 
v_cvt_f16_f32 v17, v7 // (op 15049) 


v_pack_b32_f16 v4, v14, v15 // (op 15035) 
v_pack_b32_f16 v5, v16, v17 // (op 15035) 

s_lshl_b32 s0, s4, 6 // (op 15035) 
s_add_u32 s0, s0, 96 // (op 15035) 
v_add_u32 v3, v12, s0 // (op 15035) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15035) Store value





s_lshl_b32 s0, s2, 5 // (op 15055) 
s_mul_i32 s1, 3, s0 // (op 15055) 
s_add_u32 s1, s1, 96 // (op 15055) 

v_add_u32 v3, v11, s1 // (op 15055) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15055) Load value

v_accvgpr_read v6, a60 // (op 15061) 
v_accvgpr_read v7, a61 // (op 15061) 
v_accvgpr_read v8, a62 // (op 15061) 
v_accvgpr_read v9, a63 // (op 15061) 
v_mul_f32 v14, v13, v6 // (op 15061) 
v_mul_f32 v15, v13, v7 // (op 15061) 
v_mul_f32 v16, v13, v8 // (op 15061) 
v_mul_f32 v17, v13, v9 // (op 15061) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15052) 
v_lshrrev_b32 v7, 16, v4 // (op 15052) 
v_cvt_f32_f16 v7, v7 // (op 15052) 
v_mul_f32 v18, v0, v6 // (op 15052) 
v_mul_f32 v19, v0, v7 // (op 15052) 
v_cvt_f32_f16 v6, v5 // (op 15052) 
v_lshrrev_b32 v7, 16, v5 // (op 15052) 
v_cvt_f32_f16 v7, v7 // (op 15052) 
v_mul_f32 v20, v0, v6 // (op 15052) 
v_mul_f32 v21, v0, v7 // (op 15052) 


v_add_f32 v4, v18, v14 // (op 15053) 
v_add_f32 v5, v19, v15 // (op 15053) 
v_add_f32 v6, v20, v16 // (op 15053) 
v_add_f32 v7, v21, v17 // (op 15053) 


v_cvt_f16_f32 v14, v4 // (op 15054) 
v_cvt_f16_f32 v15, v5 // (op 15054) 
v_cvt_f16_f32 v16, v6 // (op 15054) 
v_cvt_f16_f32 v17, v7 // (op 15054) 


v_pack_b32_f16 v4, v14, v15 // (op 15037) 
v_pack_b32_f16 v5, v16, v17 // (op 15037) 

s_lshl_b32 s0, s4, 5 // (op 15037) 
s_mul_i32 s1, 3, s0 // (op 15037) 
s_add_u32 s1, s1, 96 // (op 15037) 

v_add_u32 v3, v12, s1 // (op 15037) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15037) Store value





v_add_u32 v3, 64, v11 // (op 14953) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14953) Load value

v_accvgpr_read v6, a32 // (op 14940) 
v_accvgpr_read v7, a33 // (op 14940) 
v_accvgpr_read v8, a34 // (op 14940) 
v_accvgpr_read v9, a35 // (op 14940) 
v_mul_f32 v14, v13, v6 // (op 14940) 
v_mul_f32 v15, v13, v7 // (op 14940) 
v_mul_f32 v16, v13, v8 // (op 14940) 
v_mul_f32 v17, v13, v9 // (op 14940) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14952) 
v_lshrrev_b32 v7, 16, v4 // (op 14952) 
v_cvt_f32_f16 v7, v7 // (op 14952) 
v_mul_f32 v18, v0, v6 // (op 14952) 
v_mul_f32 v19, v0, v7 // (op 14952) 
v_cvt_f32_f16 v6, v5 // (op 14952) 
v_lshrrev_b32 v7, 16, v5 // (op 14952) 
v_cvt_f32_f16 v7, v7 // (op 14952) 
v_mul_f32 v20, v0, v6 // (op 14952) 
v_mul_f32 v21, v0, v7 // (op 14952) 


v_add_f32 v4, v18, v14 // (op 14941) 
v_add_f32 v5, v19, v15 // (op 14941) 
v_add_f32 v6, v20, v16 // (op 14941) 
v_add_f32 v7, v21, v17 // (op 14941) 


v_cvt_f16_f32 v14, v4 // (op 14942) 
v_cvt_f16_f32 v15, v5 // (op 14942) 
v_cvt_f16_f32 v16, v6 // (op 14942) 
v_cvt_f16_f32 v17, v7 // (op 14942) 


v_pack_b32_f16 v4, v14, v15 // (op 14944) 
v_pack_b32_f16 v5, v16, v17 // (op 14944) 

v_add_u32 v3, 64, v12 // (op 14944) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14944) Store value





s_lshl_b32 s0, s2, 5 // (op 14958) 
s_add_u32 s0, s0, 64 // (op 14958) 
v_add_u32 v3, v11, s0 // (op 14958) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14958) Load value

v_accvgpr_read v6, a36 // (op 14970) 
v_accvgpr_read v7, a37 // (op 14970) 
v_accvgpr_read v8, a38 // (op 14970) 
v_accvgpr_read v9, a39 // (op 14970) 
v_mul_f32 v14, v13, v6 // (op 14970) 
v_mul_f32 v15, v13, v7 // (op 14970) 
v_mul_f32 v16, v13, v8 // (op 14970) 
v_mul_f32 v17, v13, v9 // (op 14970) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14955) 
v_lshrrev_b32 v7, 16, v4 // (op 14955) 
v_cvt_f32_f16 v7, v7 // (op 14955) 
v_mul_f32 v18, v0, v6 // (op 14955) 
v_mul_f32 v19, v0, v7 // (op 14955) 
v_cvt_f32_f16 v6, v5 // (op 14955) 
v_lshrrev_b32 v7, 16, v5 // (op 14955) 
v_cvt_f32_f16 v7, v7 // (op 14955) 
v_mul_f32 v20, v0, v6 // (op 14955) 
v_mul_f32 v21, v0, v7 // (op 14955) 


v_add_f32 v4, v18, v14 // (op 14956) 
v_add_f32 v5, v19, v15 // (op 14956) 
v_add_f32 v6, v20, v16 // (op 14956) 
v_add_f32 v7, v21, v17 // (op 14956) 


v_cvt_f16_f32 v14, v4 // (op 14957) 
v_cvt_f16_f32 v15, v5 // (op 14957) 
v_cvt_f16_f32 v16, v6 // (op 14957) 
v_cvt_f16_f32 v17, v7 // (op 14957) 


v_pack_b32_f16 v4, v14, v15 // (op 14946) 
v_pack_b32_f16 v5, v16, v17 // (op 14946) 

s_lshl_b32 s0, s4, 5 // (op 14946) 
s_add_u32 s0, s0, 64 // (op 14946) 
v_add_u32 v3, v12, s0 // (op 14946) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14946) Store value





s_lshl_b32 s0, s2, 6 // (op 14963) 
s_add_u32 s0, s0, 64 // (op 14963) 
v_add_u32 v3, v11, s0 // (op 14963) 


buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14963) Load value

v_accvgpr_read v6, a40 // (op 14972) 
v_accvgpr_read v7, a41 // (op 14972) 
v_accvgpr_read v8, a42 // (op 14972) 
v_accvgpr_read v9, a43 // (op 14972) 
v_mul_f32 v14, v13, v6 // (op 14972) 
v_mul_f32 v15, v13, v7 // (op 14972) 
v_mul_f32 v16, v13, v8 // (op 14972) 
v_mul_f32 v17, v13, v9 // (op 14972) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14960) 
v_lshrrev_b32 v7, 16, v4 // (op 14960) 
v_cvt_f32_f16 v7, v7 // (op 14960) 
v_mul_f32 v18, v0, v6 // (op 14960) 
v_mul_f32 v19, v0, v7 // (op 14960) 
v_cvt_f32_f16 v6, v5 // (op 14960) 
v_lshrrev_b32 v7, 16, v5 // (op 14960) 
v_cvt_f32_f16 v7, v7 // (op 14960) 
v_mul_f32 v20, v0, v6 // (op 14960) 
v_mul_f32 v21, v0, v7 // (op 14960) 


v_add_f32 v4, v18, v14 // (op 14961) 
v_add_f32 v5, v19, v15 // (op 14961) 
v_add_f32 v6, v20, v16 // (op 14961) 
v_add_f32 v7, v21, v17 // (op 14961) 


v_cvt_f16_f32 v14, v4 // (op 14962) 
v_cvt_f16_f32 v15, v5 // (op 14962) 
v_cvt_f16_f32 v16, v6 // (op 14962) 
v_cvt_f16_f32 v17, v7 // (op 14962) 


v_pack_b32_f16 v4, v14, v15 // (op 14948) 
v_pack_b32_f16 v5, v16, v17 // (op 14948) 

s_lshl_b32 s0, s4, 6 // (op 14948) 
s_add_u32 s0, s0, 64 // (op 14948) 
v_add_u32 v3, v12, s0 // (op 14948) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14948) Store value





s_lshl_b32 s0, s2, 5 // (op 14968) 
s_mul_i32 s1, 3, s0 // (op 14968) 
s_add_u32 s1, s1, 64 // (op 14968) 

v_add_u32 v3, v11, s1 // (op 14968) 


v_accvgpr_read v4, a44 // (op 14974) 
v_accvgpr_read v5, a45 // (op 14974) 
v_accvgpr_read v6, a46 // (op 14974) 
v_accvgpr_read v7, a47 // (op 14974) 
v_mul_f32 v14, v13, v4 // (op 14974) 
v_mul_f32 v15, v13, v5 // (op 14974) 
v_mul_f32 v16, v13, v6 // (op 14974) 
v_mul_f32 v17, v13, v7 // (op 14974) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14968) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14965) 
v_lshrrev_b32 v7, 16, v4 // (op 14965) 
v_cvt_f32_f16 v7, v7 // (op 14965) 
v_mul_f32 v18, v0, v6 // (op 14965) 
v_mul_f32 v19, v0, v7 // (op 14965) 
v_cvt_f32_f16 v6, v5 // (op 14965) 
v_lshrrev_b32 v7, 16, v5 // (op 14965) 
v_cvt_f32_f16 v7, v7 // (op 14965) 
v_mul_f32 v20, v0, v6 // (op 14965) 
v_mul_f32 v21, v0, v7 // (op 14965) 


v_add_f32 v4, v18, v14 // (op 14966) 
v_add_f32 v5, v19, v15 // (op 14966) 
v_add_f32 v6, v20, v16 // (op 14966) 
v_add_f32 v7, v21, v17 // (op 14966) 


v_cvt_f16_f32 v14, v4 // (op 14967) 
v_cvt_f16_f32 v15, v5 // (op 14967) 
v_cvt_f16_f32 v16, v6 // (op 14967) 
v_cvt_f16_f32 v17, v7 // (op 14967) 


v_pack_b32_f16 v4, v14, v15 // (op 14950) 
v_pack_b32_f16 v5, v16, v17 // (op 14950) 

s_lshl_b32 s0, s4, 5 // (op 14950) 
s_mul_i32 s1, 3, s0 // (op 14950) 
s_add_u32 s1, s1, 64 // (op 14950) 

v_add_u32 v3, v12, s1 // (op 14950) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14950) Store value





v_add_u32 v3, 32, v11 // (op 14866) 

v_accvgpr_read v4, a16 // (op 14853) 
v_accvgpr_read v5, a17 // (op 14853) 
v_accvgpr_read v6, a18 // (op 14853) 
v_accvgpr_read v7, a19 // (op 14853) 
v_mul_f32 v14, v13, v4 // (op 14853) 
v_mul_f32 v15, v13, v5 // (op 14853) 
v_mul_f32 v16, v13, v6 // (op 14853) 
v_mul_f32 v17, v13, v7 // (op 14853) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14866) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14865) 
v_lshrrev_b32 v7, 16, v4 // (op 14865) 
v_cvt_f32_f16 v7, v7 // (op 14865) 
v_mul_f32 v18, v0, v6 // (op 14865) 
v_mul_f32 v19, v0, v7 // (op 14865) 
v_cvt_f32_f16 v6, v5 // (op 14865) 
v_lshrrev_b32 v7, 16, v5 // (op 14865) 
v_cvt_f32_f16 v7, v7 // (op 14865) 
v_mul_f32 v20, v0, v6 // (op 14865) 
v_mul_f32 v21, v0, v7 // (op 14865) 


v_add_f32 v4, v18, v14 // (op 14854) 
v_add_f32 v5, v19, v15 // (op 14854) 
v_add_f32 v6, v20, v16 // (op 14854) 
v_add_f32 v7, v21, v17 // (op 14854) 


v_cvt_f16_f32 v14, v4 // (op 14855) 
v_cvt_f16_f32 v15, v5 // (op 14855) 
v_cvt_f16_f32 v16, v6 // (op 14855) 
v_cvt_f16_f32 v17, v7 // (op 14855) 


v_pack_b32_f16 v4, v14, v15 // (op 14857) 
v_pack_b32_f16 v5, v16, v17 // (op 14857) 

v_add_u32 v3, 32, v12 // (op 14857) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14857) Store value





s_lshl_b32 s0, s2, 5 // (op 14871) 
s_add_u32 s0, s0, 32 // (op 14871) 
v_add_u32 v3, v11, s0 // (op 14871) 


v_accvgpr_read v4, a20 // (op 14883) 
v_accvgpr_read v5, a21 // (op 14883) 
v_accvgpr_read v6, a22 // (op 14883) 
v_accvgpr_read v7, a23 // (op 14883) 
v_mul_f32 v14, v13, v4 // (op 14883) 
v_mul_f32 v15, v13, v5 // (op 14883) 
v_mul_f32 v16, v13, v6 // (op 14883) 
v_mul_f32 v17, v13, v7 // (op 14883) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14871) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14868) 
v_lshrrev_b32 v7, 16, v4 // (op 14868) 
v_cvt_f32_f16 v7, v7 // (op 14868) 
v_mul_f32 v18, v0, v6 // (op 14868) 
v_mul_f32 v19, v0, v7 // (op 14868) 
v_cvt_f32_f16 v6, v5 // (op 14868) 
v_lshrrev_b32 v7, 16, v5 // (op 14868) 
v_cvt_f32_f16 v7, v7 // (op 14868) 
v_mul_f32 v20, v0, v6 // (op 14868) 
v_mul_f32 v21, v0, v7 // (op 14868) 


v_add_f32 v4, v18, v14 // (op 14869) 
v_add_f32 v5, v19, v15 // (op 14869) 
v_add_f32 v6, v20, v16 // (op 14869) 
v_add_f32 v7, v21, v17 // (op 14869) 


v_cvt_f16_f32 v14, v4 // (op 14870) 
v_cvt_f16_f32 v15, v5 // (op 14870) 
v_cvt_f16_f32 v16, v6 // (op 14870) 
v_cvt_f16_f32 v17, v7 // (op 14870) 


v_pack_b32_f16 v4, v14, v15 // (op 14859) 
v_pack_b32_f16 v5, v16, v17 // (op 14859) 

s_lshl_b32 s0, s4, 5 // (op 14859) 
s_add_u32 s0, s0, 32 // (op 14859) 
v_add_u32 v3, v12, s0 // (op 14859) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14859) Store value





s_lshl_b32 s0, s2, 6 // (op 14876) 
s_add_u32 s0, s0, 32 // (op 14876) 
v_add_u32 v3, v11, s0 // (op 14876) 


v_accvgpr_read v4, a24 // (op 14885) 
v_accvgpr_read v5, a25 // (op 14885) 
v_accvgpr_read v6, a26 // (op 14885) 
v_accvgpr_read v7, a27 // (op 14885) 
v_mul_f32 v14, v13, v4 // (op 14885) 
v_mul_f32 v15, v13, v5 // (op 14885) 
v_mul_f32 v16, v13, v6 // (op 14885) 
v_mul_f32 v17, v13, v7 // (op 14885) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14876) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14873) 
v_lshrrev_b32 v7, 16, v4 // (op 14873) 
v_cvt_f32_f16 v7, v7 // (op 14873) 
v_mul_f32 v18, v0, v6 // (op 14873) 
v_mul_f32 v19, v0, v7 // (op 14873) 
v_cvt_f32_f16 v6, v5 // (op 14873) 
v_lshrrev_b32 v7, 16, v5 // (op 14873) 
v_cvt_f32_f16 v7, v7 // (op 14873) 
v_mul_f32 v20, v0, v6 // (op 14873) 
v_mul_f32 v21, v0, v7 // (op 14873) 


v_add_f32 v4, v18, v14 // (op 14874) 
v_add_f32 v5, v19, v15 // (op 14874) 
v_add_f32 v6, v20, v16 // (op 14874) 
v_add_f32 v7, v21, v17 // (op 14874) 


v_cvt_f16_f32 v14, v4 // (op 14875) 
v_cvt_f16_f32 v15, v5 // (op 14875) 
v_cvt_f16_f32 v16, v6 // (op 14875) 
v_cvt_f16_f32 v17, v7 // (op 14875) 


v_pack_b32_f16 v4, v14, v15 // (op 14861) 
v_pack_b32_f16 v5, v16, v17 // (op 14861) 

s_lshl_b32 s0, s4, 6 // (op 14861) 
s_add_u32 s0, s0, 32 // (op 14861) 
v_add_u32 v3, v12, s0 // (op 14861) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14861) Store value





s_lshl_b32 s0, s2, 5 // (op 14881) 
s_mul_i32 s1, 3, s0 // (op 14881) 
s_add_u32 s1, s1, 32 // (op 14881) 

v_add_u32 v3, v11, s1 // (op 14881) 


v_accvgpr_read v4, a28 // (op 14887) 
v_accvgpr_read v5, a29 // (op 14887) 
v_accvgpr_read v6, a30 // (op 14887) 
v_accvgpr_read v7, a31 // (op 14887) 
v_mul_f32 v14, v13, v4 // (op 14887) 
v_mul_f32 v15, v13, v5 // (op 14887) 
v_mul_f32 v16, v13, v6 // (op 14887) 
v_mul_f32 v17, v13, v7 // (op 14887) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14881) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14878) 
v_lshrrev_b32 v7, 16, v4 // (op 14878) 
v_cvt_f32_f16 v7, v7 // (op 14878) 
v_mul_f32 v18, v0, v6 // (op 14878) 
v_mul_f32 v19, v0, v7 // (op 14878) 
v_cvt_f32_f16 v6, v5 // (op 14878) 
v_lshrrev_b32 v7, 16, v5 // (op 14878) 
v_cvt_f32_f16 v7, v7 // (op 14878) 
v_mul_f32 v20, v0, v6 // (op 14878) 
v_mul_f32 v21, v0, v7 // (op 14878) 


v_add_f32 v4, v18, v14 // (op 14879) 
v_add_f32 v5, v19, v15 // (op 14879) 
v_add_f32 v6, v20, v16 // (op 14879) 
v_add_f32 v7, v21, v17 // (op 14879) 


v_cvt_f16_f32 v14, v4 // (op 14880) 
v_cvt_f16_f32 v15, v5 // (op 14880) 
v_cvt_f16_f32 v16, v6 // (op 14880) 
v_cvt_f16_f32 v17, v7 // (op 14880) 


v_pack_b32_f16 v4, v14, v15 // (op 14863) 
v_pack_b32_f16 v5, v16, v17 // (op 14863) 

s_lshl_b32 s0, s4, 5 // (op 14863) 
s_mul_i32 s1, 3, s0 // (op 14863) 
s_add_u32 s1, s1, 32 // (op 14863) 

v_add_u32 v3, v12, s1 // (op 14863) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14863) Store value





v_add_u32 v3, 128, v11 // (op 15127) 

v_accvgpr_read v4, a64 // (op 15114) 
v_accvgpr_read v5, a65 // (op 15114) 
v_accvgpr_read v6, a66 // (op 15114) 
v_accvgpr_read v7, a67 // (op 15114) 
v_mul_f32 v14, v13, v4 // (op 15114) 
v_mul_f32 v15, v13, v5 // (op 15114) 
v_mul_f32 v16, v13, v6 // (op 15114) 
v_mul_f32 v17, v13, v7 // (op 15114) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15127) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15126) 
v_lshrrev_b32 v7, 16, v4 // (op 15126) 
v_cvt_f32_f16 v7, v7 // (op 15126) 
v_mul_f32 v18, v0, v6 // (op 15126) 
v_mul_f32 v19, v0, v7 // (op 15126) 
v_cvt_f32_f16 v6, v5 // (op 15126) 
v_lshrrev_b32 v7, 16, v5 // (op 15126) 
v_cvt_f32_f16 v7, v7 // (op 15126) 
v_mul_f32 v20, v0, v6 // (op 15126) 
v_mul_f32 v21, v0, v7 // (op 15126) 


v_add_f32 v4, v18, v14 // (op 15115) 
v_add_f32 v5, v19, v15 // (op 15115) 
v_add_f32 v6, v20, v16 // (op 15115) 
v_add_f32 v7, v21, v17 // (op 15115) 


v_cvt_f16_f32 v14, v4 // (op 15116) 
v_cvt_f16_f32 v15, v5 // (op 15116) 
v_cvt_f16_f32 v16, v6 // (op 15116) 
v_cvt_f16_f32 v17, v7 // (op 15116) 


v_pack_b32_f16 v4, v14, v15 // (op 15118) 
v_pack_b32_f16 v5, v16, v17 // (op 15118) 

v_add_u32 v3, 128, v12 // (op 15118) 

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15118) Store value





s_lshl_b32 s0, s2, 5 // (op 15132) 
s_add_u32 s0, s0, 128 // (op 15132) 
v_add_u32 v3, v11, s0 // (op 15132) 


v_accvgpr_read v4, a68 // (op 15144) 
v_accvgpr_read v5, a69 // (op 15144) 
v_accvgpr_read v6, a70 // (op 15144) 
v_accvgpr_read v7, a71 // (op 15144) 
v_mul_f32 v14, v13, v4 // (op 15144) 
v_mul_f32 v15, v13, v5 // (op 15144) 
v_mul_f32 v16, v13, v6 // (op 15144) 
v_mul_f32 v17, v13, v7 // (op 15144) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15132) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15129) 
v_lshrrev_b32 v7, 16, v4 // (op 15129) 
v_cvt_f32_f16 v7, v7 // (op 15129) 
v_mul_f32 v18, v0, v6 // (op 15129) 
v_mul_f32 v19, v0, v7 // (op 15129) 
v_cvt_f32_f16 v6, v5 // (op 15129) 
v_lshrrev_b32 v7, 16, v5 // (op 15129) 
v_cvt_f32_f16 v7, v7 // (op 15129) 
v_mul_f32 v20, v0, v6 // (op 15129) 
v_mul_f32 v21, v0, v7 // (op 15129) 


v_add_f32 v4, v18, v14 // (op 15130) 
v_add_f32 v5, v19, v15 // (op 15130) 
v_add_f32 v6, v20, v16 // (op 15130) 
v_add_f32 v7, v21, v17 // (op 15130) 


v_cvt_f16_f32 v14, v4 // (op 15131) 
v_cvt_f16_f32 v15, v5 // (op 15131) 
v_cvt_f16_f32 v16, v6 // (op 15131) 
v_cvt_f16_f32 v17, v7 // (op 15131) 


v_pack_b32_f16 v4, v14, v15 // (op 15120) 
v_pack_b32_f16 v5, v16, v17 // (op 15120) 

s_lshl_b32 s0, s4, 5 // (op 15120) 
s_add_u32 s0, s0, 128 // (op 15120) 
v_add_u32 v3, v12, s0 // (op 15120) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15120) Store value





s_lshl_b32 s0, s2, 6 // (op 15137) 
s_add_u32 s0, s0, 128 // (op 15137) 
v_add_u32 v3, v11, s0 // (op 15137) 


v_accvgpr_read v4, a72 // (op 15146) 
v_accvgpr_read v5, a73 // (op 15146) 
v_accvgpr_read v6, a74 // (op 15146) 
v_accvgpr_read v7, a75 // (op 15146) 
v_mul_f32 v14, v13, v4 // (op 15146) 
v_mul_f32 v15, v13, v5 // (op 15146) 
v_mul_f32 v16, v13, v6 // (op 15146) 
v_mul_f32 v17, v13, v7 // (op 15146) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15137) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15134) 
v_lshrrev_b32 v7, 16, v4 // (op 15134) 
v_cvt_f32_f16 v7, v7 // (op 15134) 
v_mul_f32 v18, v0, v6 // (op 15134) 
v_mul_f32 v19, v0, v7 // (op 15134) 
v_cvt_f32_f16 v6, v5 // (op 15134) 
v_lshrrev_b32 v7, 16, v5 // (op 15134) 
v_cvt_f32_f16 v7, v7 // (op 15134) 
v_mul_f32 v20, v0, v6 // (op 15134) 
v_mul_f32 v21, v0, v7 // (op 15134) 


v_add_f32 v4, v18, v14 // (op 15135) 
v_add_f32 v5, v19, v15 // (op 15135) 
v_add_f32 v6, v20, v16 // (op 15135) 
v_add_f32 v7, v21, v17 // (op 15135) 


v_cvt_f16_f32 v14, v4 // (op 15136) 
v_cvt_f16_f32 v15, v5 // (op 15136) 
v_cvt_f16_f32 v16, v6 // (op 15136) 
v_cvt_f16_f32 v17, v7 // (op 15136) 


v_pack_b32_f16 v4, v14, v15 // (op 15122) 
v_pack_b32_f16 v5, v16, v17 // (op 15122) 

s_lshl_b32 s0, s4, 6 // (op 15122) 
s_add_u32 s0, s0, 128 // (op 15122) 
v_add_u32 v3, v12, s0 // (op 15122) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15122) Store value





s_lshl_b32 s0, s2, 5 // (op 15142) 
s_mul_i32 s1, 3, s0 // (op 15142) 
s_add_u32 s1, s1, 128 // (op 15142) 

v_add_u32 v3, v11, s1 // (op 15142) 


v_accvgpr_read v4, a76 // (op 15148) 
v_accvgpr_read v5, a77 // (op 15148) 
v_accvgpr_read v6, a78 // (op 15148) 
v_accvgpr_read v7, a79 // (op 15148) 
v_mul_f32 v14, v13, v4 // (op 15148) 
v_mul_f32 v15, v13, v5 // (op 15148) 
v_mul_f32 v16, v13, v6 // (op 15148) 
v_mul_f32 v17, v13, v7 // (op 15148) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 15142) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 15139) 
v_lshrrev_b32 v7, 16, v4 // (op 15139) 
v_cvt_f32_f16 v7, v7 // (op 15139) 
v_mul_f32 v18, v0, v6 // (op 15139) 
v_mul_f32 v19, v0, v7 // (op 15139) 
v_cvt_f32_f16 v6, v5 // (op 15139) 
v_lshrrev_b32 v7, 16, v5 // (op 15139) 
v_cvt_f32_f16 v7, v7 // (op 15139) 
v_mul_f32 v20, v0, v6 // (op 15139) 
v_mul_f32 v21, v0, v7 // (op 15139) 


v_add_f32 v4, v18, v14 // (op 15140) 
v_add_f32 v5, v19, v15 // (op 15140) 
v_add_f32 v6, v20, v16 // (op 15140) 
v_add_f32 v7, v21, v17 // (op 15140) 


v_cvt_f16_f32 v14, v4 // (op 15141) 
v_cvt_f16_f32 v15, v5 // (op 15141) 
v_cvt_f16_f32 v16, v6 // (op 15141) 
v_cvt_f16_f32 v17, v7 // (op 15141) 


v_pack_b32_f16 v4, v14, v15 // (op 15124) 
v_pack_b32_f16 v5, v16, v17 // (op 15124) 

s_lshl_b32 s0, s4, 5 // (op 15124) 
s_mul_i32 s1, 3, s0 // (op 15124) 
s_add_u32 s1, s1, 128 // (op 15124) 

v_add_u32 v3, v12, s1 // (op 15124) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 15124) Store value





v_mov_b32 v3, v11 // (op 129) reg expression

v_accvgpr_read v4, a0 // (op 125) 
v_accvgpr_read v5, a1 // (op 125) 
v_accvgpr_read v6, a2 // (op 125) 
v_accvgpr_read v7, a3 // (op 125) 
v_mul_f32 v14, v13, v4 // (op 125) 
v_mul_f32 v15, v13, v5 // (op 125) 
v_mul_f32 v16, v13, v6 // (op 125) 
v_mul_f32 v17, v13, v7 // (op 125) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 129) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 130) 
v_lshrrev_b32 v7, 16, v4 // (op 130) 
v_cvt_f32_f16 v7, v7 // (op 130) 
v_mul_f32 v18, v0, v6 // (op 130) 
v_mul_f32 v19, v0, v7 // (op 130) 
v_cvt_f32_f16 v6, v5 // (op 130) 
v_lshrrev_b32 v7, 16, v5 // (op 130) 
v_cvt_f32_f16 v7, v7 // (op 130) 
v_mul_f32 v20, v0, v6 // (op 130) 
v_mul_f32 v21, v0, v7 // (op 130) 


v_add_f32 v4, v18, v14 // (op 126) 
v_add_f32 v5, v19, v15 // (op 126) 
v_add_f32 v6, v20, v16 // (op 126) 
v_add_f32 v7, v21, v17 // (op 126) 


v_cvt_f16_f32 v14, v4 // (op 127) 
v_cvt_f16_f32 v15, v5 // (op 127) 
v_cvt_f16_f32 v16, v6 // (op 127) 
v_cvt_f16_f32 v17, v7 // (op 127) 


v_pack_b32_f16 v4, v14, v15 // (op 128) 
v_pack_b32_f16 v5, v16, v17 // (op 128) 

v_mov_b32 v3, v12 // (op 128) reg expression

buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 14774) 
v_add_u32 v3, v11, s0 // (op 14774) 


v_accvgpr_read v4, a4 // (op 14770) 
v_accvgpr_read v5, a5 // (op 14770) 
v_accvgpr_read v6, a6 // (op 14770) 
v_accvgpr_read v7, a7 // (op 14770) 
v_mul_f32 v14, v13, v4 // (op 14770) 
v_mul_f32 v15, v13, v5 // (op 14770) 
v_mul_f32 v16, v13, v6 // (op 14770) 
v_mul_f32 v17, v13, v7 // (op 14770) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14774) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14775) 
v_lshrrev_b32 v7, 16, v4 // (op 14775) 
v_cvt_f32_f16 v7, v7 // (op 14775) 
v_mul_f32 v18, v0, v6 // (op 14775) 
v_mul_f32 v19, v0, v7 // (op 14775) 
v_cvt_f32_f16 v6, v5 // (op 14775) 
v_lshrrev_b32 v7, 16, v5 // (op 14775) 
v_cvt_f32_f16 v7, v7 // (op 14775) 
v_mul_f32 v20, v0, v6 // (op 14775) 
v_mul_f32 v21, v0, v7 // (op 14775) 


v_add_f32 v4, v18, v14 // (op 14771) 
v_add_f32 v5, v19, v15 // (op 14771) 
v_add_f32 v6, v20, v16 // (op 14771) 
v_add_f32 v7, v21, v17 // (op 14771) 


v_cvt_f16_f32 v14, v4 // (op 14772) 
v_cvt_f16_f32 v15, v5 // (op 14772) 
v_cvt_f16_f32 v16, v6 // (op 14772) 
v_cvt_f16_f32 v17, v7 // (op 14772) 


v_pack_b32_f16 v4, v14, v15 // (op 14773) 
v_pack_b32_f16 v5, v16, v17 // (op 14773) 

s_lshl_b32 s0, s4, 5 // (op 14773) 
v_add_u32 v3, v12, s0 // (op 14773) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14773) Store value





s_lshl_b32 s0, s2, 6 // (op 14788) 
v_add_u32 v3, v11, s0 // (op 14788) 


v_accvgpr_read v4, a8 // (op 14784) 
v_accvgpr_read v5, a9 // (op 14784) 
v_accvgpr_read v6, a10 // (op 14784) 
v_accvgpr_read v7, a11 // (op 14784) 
v_mul_f32 v14, v13, v4 // (op 14784) 
v_mul_f32 v15, v13, v5 // (op 14784) 
v_mul_f32 v16, v13, v6 // (op 14784) 
v_mul_f32 v17, v13, v7 // (op 14784) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14788) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14789) 
v_lshrrev_b32 v7, 16, v4 // (op 14789) 
v_cvt_f32_f16 v7, v7 // (op 14789) 
v_mul_f32 v18, v0, v6 // (op 14789) 
v_mul_f32 v19, v0, v7 // (op 14789) 
v_cvt_f32_f16 v6, v5 // (op 14789) 
v_lshrrev_b32 v7, 16, v5 // (op 14789) 
v_cvt_f32_f16 v7, v7 // (op 14789) 
v_mul_f32 v20, v0, v6 // (op 14789) 
v_mul_f32 v21, v0, v7 // (op 14789) 


v_add_f32 v4, v18, v14 // (op 14785) 
v_add_f32 v5, v19, v15 // (op 14785) 
v_add_f32 v6, v20, v16 // (op 14785) 
v_add_f32 v7, v21, v17 // (op 14785) 


v_cvt_f16_f32 v14, v4 // (op 14786) 
v_cvt_f16_f32 v15, v5 // (op 14786) 
v_cvt_f16_f32 v16, v6 // (op 14786) 
v_cvt_f16_f32 v17, v7 // (op 14786) 


v_pack_b32_f16 v4, v14, v15 // (op 14787) 
v_pack_b32_f16 v5, v16, v17 // (op 14787) 

s_lshl_b32 s0, s4, 6 // (op 14787) 
v_add_u32 v3, v12, s0 // (op 14787) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14787) Store value





s_lshl_b32 s0, s2, 5 // (op 14802) 
s_mul_i32 s1, 3, s0 // (op 14802) 

v_add_u32 v3, v11, s1 // (op 14802) 


v_accvgpr_read v4, a12 // (op 14798) 
v_accvgpr_read v5, a13 // (op 14798) 
v_accvgpr_read v6, a14 // (op 14798) 
v_accvgpr_read v7, a15 // (op 14798) 
v_mul_f32 v14, v13, v4 // (op 14798) 
v_mul_f32 v15, v13, v5 // (op 14798) 
v_mul_f32 v16, v13, v6 // (op 14798) 
v_mul_f32 v17, v13, v7 // (op 14798) 

buffer_load_dwordx2 v[4:5], v3, s[40:43], 0 offen // (op 14802) Load value

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v6, v4 // (op 14803) 
v_lshrrev_b32 v7, 16, v4 // (op 14803) 
v_cvt_f32_f16 v7, v7 // (op 14803) 
v_mul_f32 v18, v0, v6 // (op 14803) 
v_mul_f32 v19, v0, v7 // (op 14803) 
v_cvt_f32_f16 v6, v5 // (op 14803) 
v_lshrrev_b32 v7, 16, v5 // (op 14803) 
v_cvt_f32_f16 v7, v7 // (op 14803) 
v_mul_f32 v20, v0, v6 // (op 14803) 
v_mul_f32 v21, v0, v7 // (op 14803) 


v_add_f32 v4, v18, v14 // (op 14799) 
v_add_f32 v5, v19, v15 // (op 14799) 
v_add_f32 v6, v20, v16 // (op 14799) 
v_add_f32 v7, v21, v17 // (op 14799) 


v_cvt_f16_f32 v14, v4 // (op 14800) 
v_cvt_f16_f32 v15, v5 // (op 14800) 
v_cvt_f16_f32 v16, v6 // (op 14800) 
v_cvt_f16_f32 v17, v7 // (op 14800) 


v_pack_b32_f16 v4, v14, v15 // (op 14801) 
v_pack_b32_f16 v5, v16, v17 // (op 14801) 

s_lshl_b32 s0, s4, 5 // (op 14801) 
s_mul_i32 s1, 3, s0 // (op 14801) 

v_add_u32 v3, v12, s1 // (op 14801) 


buffer_store_dwordx2 v[4:5], v3, s[44:47], 0 offen // (op 14801) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_ConditionalBottom_0_beta_116:
























































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4
.LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_end:
.size RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4, .LRR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4_end-RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4
  .amdhsa_next_free_vgpr 368
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 81920 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DataTypeA: F4
    DataTypeB: F4
    DestDataType: h
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
    macrotile: [192, 128, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [12, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4
    .symbol: RR_GEMM_TN_FP4_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_192x128x128_UR_4.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 81920
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 66
    .vgpr_count: 176
    .agpr_count: 192
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
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

