

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_exec_begin:


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


v_and_b32 v0, 4294967232, v1 // (op 30750) 
v_and_b32 v3, 15, v1 // (op 30750) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 30750) 


v_and_b32 v0, 63, v1 // (op 30750) 
v_lshrrev_b32 v3, 4, v0 // (op 30750) 

v_lshlrev_b32 v0, 5, v3 // (op 30750) 

v_add_lshl_u32 v3, v4, v0, 2 // (op 30750) 


v_lshrrev_b32 v0, 3, v3 // (op 30750) 

v_and_b32 v3, 15, v1 // (op 30807) 
v_and_b32 v4, 63, v1 // (op 30807) 
v_lshrrev_b32 v5, 4, v4 // (op 30807) 

v_lshlrev_b32 v4, 5, v5 // (op 30807) 

v_lshl_add_u32 v5, v3, 7, v4 // (op 30807) 


v_lshlrev_b32 v3, 2, v5 // (op 30807) 

v_lshrrev_b32 v4, 3, v3 // (op 30807) 

v_and_b32 v3, 4294967232, v1 // (op 30874) 
v_and_b32 v5, 15, v1 // (op 30874) 
v_add_lshl_u32 v6, v3, v5, 7 // (op 30874) 


v_and_b32 v3, 63, v1 // (op 30874) 
v_lshrrev_b32 v5, 4, v3 // (op 30874) 

v_lshlrev_b32 v3, 5, v5 // (op 30874) 

v_add_lshl_u32 v5, v6, v3, 2 // (op 30874) 


v_lshrrev_b32 v3, 3, v5 // (op 30874) 

v_and_b32 v5, 15, v1 // (op 30918) 
v_and_b32 v6, 63, v1 // (op 30918) 
v_lshrrev_b32 v7, 4, v6 // (op 30918) 

v_lshlrev_b32 v6, 5, v7 // (op 30918) 

v_lshl_add_u32 v7, v5, 7, v6 // (op 30918) 


v_lshlrev_b32 v5, 2, v7 // (op 30918) 

v_lshrrev_b32 v6, 3, v5 // (op 30918) 

v_and_b32 v5, 4294967232, v1 // (op 30962) 
v_and_b32 v7, 15, v1 // (op 30962) 
v_add_lshl_u32 v8, v5, v7, 7 // (op 30962) 


v_and_b32 v5, 63, v1 // (op 30962) 
v_lshrrev_b32 v7, 4, v5 // (op 30962) 

v_lshlrev_b32 v5, 5, v7 // (op 30962) 

v_add_lshl_u32 v7, v8, v5, 2 // (op 30962) 


v_lshrrev_b32 v5, 3, v7 // (op 30962) 

v_and_b32 v7, 15, v1 // (op 31006) 
v_and_b32 v8, 63, v1 // (op 31006) 
v_lshrrev_b32 v9, 4, v8 // (op 31006) 

v_lshlrev_b32 v8, 5, v9 // (op 31006) 

v_lshl_add_u32 v9, v7, 7, v8 // (op 31006) 


v_lshlrev_b32 v7, 2, v9 // (op 31006) 

v_lshrrev_b32 v8, 3, v7 // (op 31006) 

v_and_b32 v7, 4294967232, v1 // (op 31050) 
v_and_b32 v9, 15, v1 // (op 31050) 
v_add_lshl_u32 v10, v7, v9, 7 // (op 31050) 


v_and_b32 v7, 63, v1 // (op 31050) 
v_lshrrev_b32 v9, 4, v7 // (op 31050) 

v_lshlrev_b32 v7, 5, v9 // (op 31050) 

v_add_lshl_u32 v9, v10, v7, 2 // (op 31050) 


v_lshrrev_b32 v7, 3, v9 // (op 31050) 

v_and_b32 v9, 15, v1 // (op 31094) 
v_and_b32 v10, 63, v1 // (op 31094) 
v_lshrrev_b32 v11, 4, v10 // (op 31094) 

v_lshlrev_b32 v10, 5, v11 // (op 31094) 

v_lshl_add_u32 v11, v9, 7, v10 // (op 31094) 


v_lshlrev_b32 v9, 2, v11 // (op 31094) 

v_lshrrev_b32 v10, 3, v9 // (op 31094) 

v_and_b32 v9, 4294967232, v1 // (op 30689) 
v_and_b32 v11, 63, v1 // (op 30689) 
v_lshrrev_b32 v12, 4, v11 // (op 30689) 
v_lshlrev_b32 v13, 2, v12 // (op 30689) 

v_add_u32 v12, v9, v13 // (op 30689) 


v_lshl_add_u32 v9, s16, 7, v12 // (op 30689) 

v_lshrrev_b32 v12, 2, v11 // (op 30689) 

v_and_b32 v11, 3, v12 // (op 30689) 

v_and_b32 v12, 3, v1 // (op 30689) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 30689) 


v_mov_b32 v11, 192 // (op 30689) 
v_mul_lo_u32 v12, s17, v11 // (op 30689) 

v_add_u32 v12, v12, v13 // (op 30689) 

v_mul_lo_u32 v11, v12, s2 // (op 30689) 

v_add_lshl_u32 v12, v9, v11, 2 // (op 30689) 


v_and_b32 v9, 4294967232, v1 // (op 30828) 
v_and_b32 v11, 63, v1 // (op 30828) 
v_lshrrev_b32 v13, 4, v11 // (op 30828) 
s_mul_i32 s1, s2, s9 // (op 30691) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 30691) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 30691) least significant
s_mul_hi_u32 s19, s2, s8 // (op 30691) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 30691) most significant: sum
s_add_u32 s1, s1, s19 // (op 30691) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 30691) 

v_lshlrev_b32 v14, 2, v13 // (op 30828) 

v_add_u32 v13, v9, v14 // (op 30828) 


v_lshl_add_u32 v9, s16, 7, v13 // (op 30828) 

v_lshrrev_b32 v13, 2, v11 // (op 30828) 

v_and_b32 v11, 3, v13 // (op 30828) 

v_and_b32 v13, 3, v1 // (op 30828) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 30828) 


v_mov_b32 v11, 192 // (op 30828) 
v_mul_lo_u32 v13, s17, v11 // (op 30828) 

v_add_u32 v13, v13, v14 // (op 30828) 

v_mul_lo_u32 v11, v13, s4 // (op 30828) 

v_add_lshl_u32 v13, v9, v11, 2 // (op 30828) 


s_mov_b32 s40, s6 // (op 30691) reg expression
s_mov_b32 s41, s7 // (op 30691) reg expression
s_mov_b32 s42, s18 // (op 30691) reg expression
s_mov_b32 s43, 131072 // (op 30691) call()

s_mul_i32 s1, s4, s9 // (op 30830) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 30830) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 30830) least significant
s_mul_hi_u32 s7, s4, s8 // (op 30830) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30830) most significant: sum
s_add_u32 s1, s1, s7 // (op 30830) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30830) 

s_mov_b32 s44, s10 // (op 30830) reg expression
s_mov_b32 s45, s11 // (op 30830) reg expression
s_mov_b32 s46, s6 // (op 30830) reg expression
s_mov_b32 s47, 131072 // (op 30830) call()


v_mov_b32 v9, s14 // (op 2) Move value
v_mov_b32 v11, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_accvgpr_write a4, 0.00000 // (op 383) call()
v_accvgpr_write a5, 0.00000 // (op 383) call()
v_accvgpr_write a6, 0.00000 // (op 383) call()
v_accvgpr_write a7, 0.00000 // (op 383) call()
v_accvgpr_write a8, 0.00000 // (op 609) call()
v_accvgpr_write a9, 0.00000 // (op 609) call()
v_accvgpr_write a10, 0.00000 // (op 609) call()
v_accvgpr_write a11, 0.00000 // (op 609) call()
v_accvgpr_write a12, 0.00000 // (op 835) call()
v_accvgpr_write a13, 0.00000 // (op 835) call()
v_accvgpr_write a14, 0.00000 // (op 835) call()
v_accvgpr_write a15, 0.00000 // (op 835) call()
v_accvgpr_write a16, 0.00000 // (op 1061) call()
v_accvgpr_write a17, 0.00000 // (op 1061) call()
v_accvgpr_write a18, 0.00000 // (op 1061) call()
v_accvgpr_write a19, 0.00000 // (op 1061) call()
v_accvgpr_write a20, 0.00000 // (op 1287) call()
v_accvgpr_write a21, 0.00000 // (op 1287) call()
v_accvgpr_write a22, 0.00000 // (op 1287) call()
v_accvgpr_write a23, 0.00000 // (op 1287) call()
v_accvgpr_write a24, 0.00000 // (op 1513) call()
v_accvgpr_write a25, 0.00000 // (op 1513) call()
v_accvgpr_write a26, 0.00000 // (op 1513) call()
v_accvgpr_write a27, 0.00000 // (op 1513) call()
v_accvgpr_write a28, 0.00000 // (op 1739) call()
v_accvgpr_write a29, 0.00000 // (op 1739) call()
v_accvgpr_write a30, 0.00000 // (op 1739) call()
v_accvgpr_write a31, 0.00000 // (op 1739) call()
v_accvgpr_write a32, 0.00000 // (op 1965) call()
v_accvgpr_write a33, 0.00000 // (op 1965) call()
v_accvgpr_write a34, 0.00000 // (op 1965) call()
v_accvgpr_write a35, 0.00000 // (op 1965) call()
v_accvgpr_write a36, 0.00000 // (op 2191) call()
v_accvgpr_write a37, 0.00000 // (op 2191) call()
v_accvgpr_write a38, 0.00000 // (op 2191) call()
v_accvgpr_write a39, 0.00000 // (op 2191) call()
v_accvgpr_write a40, 0.00000 // (op 2417) call()
v_accvgpr_write a41, 0.00000 // (op 2417) call()
v_accvgpr_write a42, 0.00000 // (op 2417) call()
v_accvgpr_write a43, 0.00000 // (op 2417) call()
v_accvgpr_write a44, 0.00000 // (op 2643) call()
v_accvgpr_write a45, 0.00000 // (op 2643) call()
v_accvgpr_write a46, 0.00000 // (op 2643) call()
v_accvgpr_write a47, 0.00000 // (op 2643) call()
v_accvgpr_write a48, 0.00000 // (op 3277) call()
v_accvgpr_write a49, 0.00000 // (op 3277) call()
v_accvgpr_write a50, 0.00000 // (op 3277) call()
v_accvgpr_write a51, 0.00000 // (op 3277) call()
v_accvgpr_write a52, 0.00000 // (op 4514) call()
v_accvgpr_write a53, 0.00000 // (op 4514) call()
v_accvgpr_write a54, 0.00000 // (op 4514) call()
v_accvgpr_write a55, 0.00000 // (op 4514) call()
v_accvgpr_write a56, 0.00000 // (op 4515) call()
v_accvgpr_write a57, 0.00000 // (op 4515) call()
v_accvgpr_write a58, 0.00000 // (op 4515) call()
v_accvgpr_write a59, 0.00000 // (op 4515) call()
v_accvgpr_write a60, 0.00000 // (op 4516) call()
v_accvgpr_write a61, 0.00000 // (op 4516) call()
v_accvgpr_write a62, 0.00000 // (op 4516) call()
v_accvgpr_write a63, 0.00000 // (op 4516) call()
v_accvgpr_write a64, 0.00000 // (op 4517) call()
v_accvgpr_write a65, 0.00000 // (op 4517) call()
v_accvgpr_write a66, 0.00000 // (op 4517) call()
v_accvgpr_write a67, 0.00000 // (op 4517) call()
v_accvgpr_write a68, 0.00000 // (op 4518) call()
v_accvgpr_write a69, 0.00000 // (op 4518) call()
v_accvgpr_write a70, 0.00000 // (op 4518) call()
v_accvgpr_write a71, 0.00000 // (op 4518) call()
v_accvgpr_write a72, 0.00000 // (op 4519) call()
v_accvgpr_write a73, 0.00000 // (op 4519) call()
v_accvgpr_write a74, 0.00000 // (op 4519) call()
v_accvgpr_write a75, 0.00000 // (op 4519) call()
v_accvgpr_write a76, 0.00000 // (op 4520) call()
v_accvgpr_write a77, 0.00000 // (op 4520) call()
v_accvgpr_write a78, 0.00000 // (op 4520) call()
v_accvgpr_write a79, 0.00000 // (op 4520) call()
v_accvgpr_write a80, 0.00000 // (op 4521) call()
v_accvgpr_write a81, 0.00000 // (op 4521) call()
v_accvgpr_write a82, 0.00000 // (op 4521) call()
v_accvgpr_write a83, 0.00000 // (op 4521) call()
v_accvgpr_write a84, 0.00000 // (op 4522) call()
v_accvgpr_write a85, 0.00000 // (op 4522) call()
v_accvgpr_write a86, 0.00000 // (op 4522) call()
v_accvgpr_write a87, 0.00000 // (op 4522) call()
v_accvgpr_write a88, 0.00000 // (op 4523) call()
v_accvgpr_write a89, 0.00000 // (op 4523) call()
v_accvgpr_write a90, 0.00000 // (op 4523) call()
v_accvgpr_write a91, 0.00000 // (op 4523) call()
v_accvgpr_write a92, 0.00000 // (op 4524) call()
v_accvgpr_write a93, 0.00000 // (op 4524) call()
v_accvgpr_write a94, 0.00000 // (op 4524) call()
v_accvgpr_write a95, 0.00000 // (op 4524) call()
v_accvgpr_write a96, 0.00000 // (op 6279) call()
v_accvgpr_write a97, 0.00000 // (op 6279) call()
v_accvgpr_write a98, 0.00000 // (op 6279) call()
v_accvgpr_write a99, 0.00000 // (op 6279) call()
v_accvgpr_write a100, 0.00000 // (op 7516) call()
v_accvgpr_write a101, 0.00000 // (op 7516) call()
v_accvgpr_write a102, 0.00000 // (op 7516) call()
v_accvgpr_write a103, 0.00000 // (op 7516) call()
v_accvgpr_write a104, 0.00000 // (op 7517) call()
v_accvgpr_write a105, 0.00000 // (op 7517) call()
v_accvgpr_write a106, 0.00000 // (op 7517) call()
v_accvgpr_write a107, 0.00000 // (op 7517) call()
v_accvgpr_write a108, 0.00000 // (op 7518) call()
v_accvgpr_write a109, 0.00000 // (op 7518) call()
v_accvgpr_write a110, 0.00000 // (op 7518) call()
v_accvgpr_write a111, 0.00000 // (op 7518) call()
v_accvgpr_write a112, 0.00000 // (op 7519) call()
v_accvgpr_write a113, 0.00000 // (op 7519) call()
v_accvgpr_write a114, 0.00000 // (op 7519) call()
v_accvgpr_write a115, 0.00000 // (op 7519) call()
v_accvgpr_write a116, 0.00000 // (op 7520) call()
v_accvgpr_write a117, 0.00000 // (op 7520) call()
v_accvgpr_write a118, 0.00000 // (op 7520) call()
v_accvgpr_write a119, 0.00000 // (op 7520) call()
v_accvgpr_write a120, 0.00000 // (op 7521) call()
v_accvgpr_write a121, 0.00000 // (op 7521) call()
v_accvgpr_write a122, 0.00000 // (op 7521) call()
v_accvgpr_write a123, 0.00000 // (op 7521) call()
v_accvgpr_write a124, 0.00000 // (op 7522) call()
v_accvgpr_write a125, 0.00000 // (op 7522) call()
v_accvgpr_write a126, 0.00000 // (op 7522) call()
v_accvgpr_write a127, 0.00000 // (op 7522) call()
v_accvgpr_write a128, 0.00000 // (op 7523) call()
v_accvgpr_write a129, 0.00000 // (op 7523) call()
v_accvgpr_write a130, 0.00000 // (op 7523) call()
v_accvgpr_write a131, 0.00000 // (op 7523) call()
v_accvgpr_write a132, 0.00000 // (op 7524) call()
v_accvgpr_write a133, 0.00000 // (op 7524) call()
v_accvgpr_write a134, 0.00000 // (op 7524) call()
v_accvgpr_write a135, 0.00000 // (op 7524) call()
v_accvgpr_write a136, 0.00000 // (op 7525) call()
v_accvgpr_write a137, 0.00000 // (op 7525) call()
v_accvgpr_write a138, 0.00000 // (op 7525) call()
v_accvgpr_write a139, 0.00000 // (op 7525) call()
v_accvgpr_write a140, 0.00000 // (op 7526) call()
v_accvgpr_write a141, 0.00000 // (op 7526) call()
v_accvgpr_write a142, 0.00000 // (op 7526) call()
v_accvgpr_write a143, 0.00000 // (op 7526) call()
v_accvgpr_write a144, 0.00000 // (op 9281) call()
v_accvgpr_write a145, 0.00000 // (op 9281) call()
v_accvgpr_write a146, 0.00000 // (op 9281) call()
v_accvgpr_write a147, 0.00000 // (op 9281) call()
v_accvgpr_write a148, 0.00000 // (op 10518) call()
v_accvgpr_write a149, 0.00000 // (op 10518) call()
v_accvgpr_write a150, 0.00000 // (op 10518) call()
v_accvgpr_write a151, 0.00000 // (op 10518) call()
v_accvgpr_write a152, 0.00000 // (op 10519) call()
v_accvgpr_write a153, 0.00000 // (op 10519) call()
v_accvgpr_write a154, 0.00000 // (op 10519) call()
v_accvgpr_write a155, 0.00000 // (op 10519) call()
v_accvgpr_write a156, 0.00000 // (op 10520) call()
v_accvgpr_write a157, 0.00000 // (op 10520) call()
v_accvgpr_write a158, 0.00000 // (op 10520) call()
v_accvgpr_write a159, 0.00000 // (op 10520) call()
v_accvgpr_write a160, 0.00000 // (op 10521) call()
v_accvgpr_write a161, 0.00000 // (op 10521) call()
v_accvgpr_write a162, 0.00000 // (op 10521) call()
v_accvgpr_write a163, 0.00000 // (op 10521) call()
v_accvgpr_write a164, 0.00000 // (op 10522) call()
v_accvgpr_write a165, 0.00000 // (op 10522) call()
v_accvgpr_write a166, 0.00000 // (op 10522) call()
v_accvgpr_write a167, 0.00000 // (op 10522) call()
v_accvgpr_write a168, 0.00000 // (op 10523) call()
v_accvgpr_write a169, 0.00000 // (op 10523) call()
v_accvgpr_write a170, 0.00000 // (op 10523) call()
v_accvgpr_write a171, 0.00000 // (op 10523) call()
v_accvgpr_write a172, 0.00000 // (op 10524) call()
v_accvgpr_write a173, 0.00000 // (op 10524) call()
v_accvgpr_write a174, 0.00000 // (op 10524) call()
v_accvgpr_write a175, 0.00000 // (op 10524) call()
v_accvgpr_write a176, 0.00000 // (op 10525) call()
v_accvgpr_write a177, 0.00000 // (op 10525) call()
v_accvgpr_write a178, 0.00000 // (op 10525) call()
v_accvgpr_write a179, 0.00000 // (op 10525) call()
v_accvgpr_write a180, 0.00000 // (op 10526) call()
v_accvgpr_write a181, 0.00000 // (op 10526) call()
v_accvgpr_write a182, 0.00000 // (op 10526) call()
v_accvgpr_write a183, 0.00000 // (op 10526) call()
v_accvgpr_write a184, 0.00000 // (op 10527) call()
v_accvgpr_write a185, 0.00000 // (op 10527) call()
v_accvgpr_write a186, 0.00000 // (op 10527) call()
v_accvgpr_write a187, 0.00000 // (op 10527) call()
v_accvgpr_write a188, 0.00000 // (op 10528) call()
v_accvgpr_write a189, 0.00000 // (op 10528) call()
v_accvgpr_write a190, 0.00000 // (op 10528) call()
v_accvgpr_write a191, 0.00000 // (op 10528) call()
v_lshrrev_b32 v14, 2, v1 // (op 30579) 
v_lshl_add_u32 v15, s16, 7, v14 // (op 30579) 

v_and_b32 v14, -4, v1 // (op 30579) 
v_sub_u32 v16, v1, v14 // (op 30579) 

v_lshlrev_b32 v14, 5, v16 // (op 30579) 

v_mov_b32 v17, 0 // (op 30579) convert
v_mov_b32 v16, v14 // (op 30579) convert

v_mul_lo_u32 v19, v15, s13 // (op 30579) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 30579) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 30579) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 30579) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 30579) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 30579) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 30579) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 30579) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 30581) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 30581) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 30581) least significant
s_mul_hi_u32 s7, s12, s22 // (op 30581) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30581) most significant: sum
s_add_u32 s1, s1, s7 // (op 30581) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30581) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30581) 

s_mov_b32 s48, s20 // (op 30581) reg expression
s_mov_b32 s49, s21 // (op 30581) reg expression
s_mov_b32 s50, s0 // (op 30581) reg expression
s_mov_b32 s51, 131072 // (op 30581) call()

v_lshrrev_b32 v14, 2, v1 // (op 30598) 
v_lshl_add_u32 v15, s16, 7, v14 // (op 30598) 

v_and_b32 v14, -4, v1 // (op 30598) 
v_sub_u32 v18, v1, v14 // (op 30598) 

v_lshlrev_b32 v14, 5, v18 // (op 30598) 

v_mov_b32 v19, 0 // (op 30598) convert
v_mov_b32 v18, v14 // (op 30598) convert

v_mul_lo_u32 v21, v15, s13 // (op 30598) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 30598) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 30598) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 30598) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 30598) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 30598) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 30598) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 30598) 

s_mul_i32 s1, s12, s23 // (op 30600) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 30600) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 30600) least significant
s_mul_hi_u32 s7, s12, s22 // (op 30600) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30600) most significant: sum
s_add_u32 s1, s1, s7 // (op 30600) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30600) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30600) 

s_mov_b32 s48, s20 // (op 30600) reg expression
s_mov_b32 s49, s21 // (op 30600) reg expression
s_mov_b32 s50, s0 // (op 30600) reg expression
s_mov_b32 s51, 131072 // (op 30600) call()

v_and_b32 v14, -4, v1 // (op 30643) 
v_sub_u32 v15, v1, v14 // (op 30643) 

v_lshlrev_b32 v14, 5, v15 // (op 30643) 

v_mov_b32 v21, 0 // (op 30643) convert
v_mov_b32 v20, v14 // (op 30643) convert

v_lshrrev_b32 v14, 2, v1 // (op 30643) 
v_mov_b32 v15, 192 // (op 30643) 
v_mul_lo_u32 v22, s17, v15 // (op 30643) 

v_add_u32 v22, v22, v14 // (op 30643) 

v_mul_lo_u32 v15, v22, s25 // (op 30643) most significant: low of low * high
v_mul_hi_u32 v23, v22, s24 // (op 30643) most significant: high of low * low
v_mul_lo_u32 v14, v22, s24 // (op 30643) least significant: low of low * low
v_add_u32 v15, v15, v23 // (op 30643) most significant: sum


v_add_co_u32 v22, s[0:1], v20, v14 // (op 30643) least significant half
v_addc_co_u32 v23, s[0:1], v21, v15, s[0:1] // (op 30643) most significant half

v_lshlrev_b64 v[22:23], 2, v[22:23] // (op 30643) 


v_lshrrev_b64 v[14:15], 3, v[22:23] // (op 30643) 

s_mul_i32 s1, s24, s9 // (op 30645) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 30645) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 30645) least significant
s_mul_hi_u32 s7, s24, s8 // (op 30645) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30645) most significant: sum
s_add_u32 s1, s1, s7 // (op 30645) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30645) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30645) 

s_mov_b32 s52, s26 // (op 30645) reg expression
s_mov_b32 s53, s27 // (op 30645) reg expression
s_mov_b32 s54, s0 // (op 30645) reg expression
s_mov_b32 s55, 131072 // (op 30645) call()

v_and_b32 v20, -4, v1 // (op 30662) 
v_sub_u32 v21, v1, v20 // (op 30662) 

v_lshlrev_b32 v20, 5, v21 // (op 30662) 

v_mov_b32 v23, 0 // (op 30662) convert
v_mov_b32 v22, v20 // (op 30662) convert

v_lshrrev_b32 v20, 2, v1 // (op 30662) 
v_mov_b32 v21, 192 // (op 30662) 
v_mul_lo_u32 v24, s17, v21 // (op 30662) 

v_add_u32 v24, v24, v20 // (op 30662) 

v_mul_lo_u32 v21, v24, s25 // (op 30662) most significant: low of low * high
v_mul_hi_u32 v25, v24, s24 // (op 30662) most significant: high of low * low
v_mul_lo_u32 v20, v24, s24 // (op 30662) least significant: low of low * low
v_add_u32 v21, v21, v25 // (op 30662) most significant: sum


v_add_co_u32 v24, s[0:1], v22, v20 // (op 30662) least significant half
v_addc_co_u32 v25, s[0:1], v23, v21, s[0:1] // (op 30662) most significant half

v_lshlrev_b64 v[24:25], 2, v[24:25] // (op 30662) 


v_lshrrev_b64 v[20:21], 3, v[24:25] // (op 30662) 

s_mul_i32 s1, s24, s9 // (op 30664) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 30664) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 30664) least significant
s_mul_hi_u32 s7, s24, s8 // (op 30664) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30664) most significant: sum
s_add_u32 s1, s1, s7 // (op 30664) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30664) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30664) 

s_mov_b32 s52, s26 // (op 30664) reg expression
s_mov_b32 s53, s27 // (op 30664) reg expression
s_mov_b32 s54, s0 // (op 30664) reg expression
s_mov_b32 s55, 131072 // (op 30664) call()

v_and_b32 v22, 63, v1 // (op 31116) 
v_lshrrev_b32 v23, 4, v22 // (op 31116) 
v_and_b32 v24, 3, v23 // (op 31116) 

v_and_b32 v23, 15, v1 // (op 31116) 
v_lshl_add_u32 v25, v24, 4, v23 // (op 31116) 


v_and_b32 v23, 4294967232, v1 // (op 31116) 
v_add_u32 v24, v23, v25 // (op 31116) 


v_lshl_add_u32 v23, s16, 7, v24 // (op 31116) 

v_lshrrev_b32 v24, 6, v22 // (op 31116) 

v_lshlrev_b32 v22, 2, v24 // (op 31116) 

v_mul_lo_u32 v24, v23, s28 // (op 31116) 
v_add_u32 v24, v24, v22 // (op 31116) 


s_mul_i32 s1, s28, s23 // (op 31118) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 31118) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 31118) least significant
s_mul_hi_u32 s7, s28, s22 // (op 31118) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31118) most significant: sum
s_add_u32 s1, s1, s7 // (op 31118) most significant: sum


s_mov_b32 s56, s30 // (op 31118) reg expression
s_mov_b32 s57, s31 // (op 31118) reg expression
s_mov_b32 s58, s0 // (op 31118) reg expression
s_mov_b32 s59, 131072 // (op 31118) call()

v_and_b32 v22, 63, v1 // (op 31137) 
v_lshrrev_b32 v23, 4, v22 // (op 31137) 
v_and_b32 v25, 3, v23 // (op 31137) 

v_and_b32 v23, 15, v1 // (op 31137) 
v_lshl_add_u32 v26, v25, 4, v23 // (op 31137) 


v_and_b32 v23, 4294967232, v1 // (op 31137) 
v_add_u32 v25, v23, v26 // (op 31137) 


v_lshl_add_u32 v23, s16, 7, v25 // (op 31137) 

v_lshrrev_b32 v25, 6, v22 // (op 31137) 

v_lshlrev_b32 v22, 2, v25 // (op 31137) 

v_mul_lo_u32 v25, v23, s28 // (op 31137) 
v_add_u32 v25, v25, v22 // (op 31137) 


s_mul_i32 s1, s28, s23 // (op 31139) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 31139) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 31139) least significant
s_mul_hi_u32 s7, s28, s22 // (op 31139) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31139) most significant: sum
s_add_u32 s1, s1, s7 // (op 31139) most significant: sum


s_mov_b32 s56, s30 // (op 31139) reg expression
s_mov_b32 s57, s31 // (op 31139) reg expression
s_mov_b32 s58, s0 // (op 31139) reg expression
s_mov_b32 s59, 131072 // (op 31139) call()

v_and_b32 v22, 63, v1 // (op 31168) 
v_lshrrev_b32 v23, 4, v22 // (op 31168) 
v_and_b32 v26, 3, v23 // (op 31168) 

v_and_b32 v23, 15, v1 // (op 31168) 
v_lshl_add_u32 v27, v26, 4, v23 // (op 31168) 


v_mov_b32 v23, 192 // (op 31168) 
v_mul_lo_u32 v26, s17, v23 // (op 31168) 

v_add_u32 v26, v26, v27 // (op 31168) 

v_mul_lo_u32 v23, v26, s32 // (op 31168) 

v_lshrrev_b32 v26, 6, v22 // (op 31168) 

v_lshl_add_u32 v22, v26, 2, v23 // (op 31168) 


s_mul_i32 s1, s32, s9 // (op 31170) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 31170) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 31170) least significant
s_mul_hi_u32 s7, s32, s8 // (op 31170) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31170) most significant: sum
s_add_u32 s1, s1, s7 // (op 31170) most significant: sum


s_mov_b32 s60, s34 // (op 31170) reg expression
s_mov_b32 s61, s35 // (op 31170) reg expression
s_mov_b32 s62, s0 // (op 31170) reg expression
s_mov_b32 s63, 131072 // (op 31170) call()

v_and_b32 v23, 63, v1 // (op 31189) 
v_lshrrev_b32 v26, 4, v23 // (op 31189) 
v_and_b32 v27, 3, v26 // (op 31189) 

v_and_b32 v26, 15, v1 // (op 31189) 
v_lshl_add_u32 v28, v27, 4, v26 // (op 31189) 


v_mov_b32 v26, 192 // (op 31189) 
v_mul_lo_u32 v27, s17, v26 // (op 31189) 

v_add_u32 v27, v27, v28 // (op 31189) 

v_mul_lo_u32 v26, v27, s32 // (op 31189) 

v_lshrrev_b32 v27, 6, v23 // (op 31189) 

v_lshl_add_u32 v23, v27, 2, v26 // (op 31189) 


s_mul_i32 s1, s32, s9 // (op 31191) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 31191) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 31191) least significant
s_mul_hi_u32 s7, s32, s8 // (op 31191) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 31191) most significant: sum
s_add_u32 s1, s1, s7 // (op 31191) most significant: sum


s_mov_b32 s60, s34 // (op 31191) reg expression
s_mov_b32 s61, s35 // (op 31191) reg expression
s_mov_b32 s62, s0 // (op 31191) reg expression
s_mov_b32 s63, 131072 // (op 31191) call()

v_and_b32 v26, -4, v1 // (op 30790) 
v_sub_u32 v27, v1, v26 // (op 30790) 

v_lshlrev_b32 v26, 5, v27 // (op 30790) 

v_lshrrev_b32 v27, 2, v1 // (op 30790) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 30790) 


v_lshlrev_b32 v26, 2, v28 // (op 30790) 

v_lshrrev_b32 v27, 3, v26 // (op 30790) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 30790) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 30473) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30473) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30473) 


v_mov_b64 v[28:29], v[14:15] // (op 30473) 
s_mov_b32 m0, s0 // (op 30473) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30473) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30473) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30473) most significant half

s_add_u32 m0, m0, 2048 // (op 30473) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30473) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30473) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30473) most significant half

s_add_u32 m0, m0, 2048 // (op 30473) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30473) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30473) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30473) most significant half

s_add_u32 m0, m0, 2048 // (op 30473) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30473) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30473) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30473) most significant half

s_add_u32 m0, m0, 2048 // (op 30473) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30473) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30473) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30473) most significant half

s_add_u32 m0, m0, 2048 // (op 30473) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 30473) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 30733) 
v_sub_u32 v27, v1, v26 // (op 30733) 

v_lshlrev_b32 v26, 5, v27 // (op 30733) 

v_lshrrev_b32 v27, 2, v1 // (op 30733) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 30733) 


v_lshlrev_b32 v26, 2, v28 // (op 30733) 

v_lshrrev_b32 v27, 3, v26 // (op 30733) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 30733) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 30478) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30478) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30478) 


v_mov_b64 v[28:29], v[16:17] // (op 30478) 
s_add_u32 s1, 12288, s0 // (op 30478) 
s_mov_b32 m0, s1 // (op 30478) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30478) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30478) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30478) most significant half

s_add_u32 m0, m0, 2048 // (op 30478) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30478) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30478) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30478) most significant half

s_add_u32 m0, m0, 2048 // (op 30478) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30478) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 30478) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 30478) most significant half

s_add_u32 m0, m0, 2048 // (op 30478) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 30478) Load value direct to lds






v_mov_b32 v26, v22 // (op 74) reg expression

buffer_load_dwordx4 v[28:31], v26, s[60:63], 0 offen // (op 74) Load value



s_lshl_b32 s0, s32, 6 // (op 1100) 
v_add_u32 v26, v22, s0 // (op 1100) 


buffer_load_dwordx4 v[32:35], v26, s[60:63], 0 offen // (op 1100) Load value



s_lshl_b32 s0, s32, 7 // (op 2004) 
v_add_u32 v26, v22, s0 // (op 2004) 


buffer_load_dwordx4 v[36:39], v26, s[60:63], 0 offen // (op 2004) Load value




v_mov_b32 v22, v24 // (op 70) reg expression

buffer_load_dwordx4 v[40:43], v22, s[56:59], 0 offen // (op 70) Load value


v_and_b32 v22, -4, v1 // (op 30901) 
v_sub_u32 v24, v1, v22 // (op 30901) 

v_lshlrev_b32 v22, 5, v24 // (op 30901) 

v_lshrrev_b32 v24, 2, v1 // (op 30901) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30901) 


v_lshlrev_b32 v22, 2, v26 // (op 30901) 

v_lshrrev_b32 v24, 3, v22 // (op 30901) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 30901) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 30483) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30483) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30483) 


v_mov_b64 v[44:45], v[14:15] // (op 30483) 
s_add_u32 s1, 20480, s0 // (op 30483) 
s_mov_b32 m0, s1 // (op 30483) reg expression
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30483) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 30483) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 30483) most significant half

s_add_u32 m0, m0, 2048 // (op 30483) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30483) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 30483) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 30483) most significant half

s_add_u32 m0, m0, 2048 // (op 30483) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30483) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 30483) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 30483) most significant half

s_add_u32 m0, m0, 2048 // (op 30483) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30483) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 30483) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 30483) most significant half

s_add_u32 m0, m0, 2048 // (op 30483) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30483) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 30483) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 30483) most significant half

s_add_u32 m0, m0, 2048 // (op 30483) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 30483) Load value direct to lds




v_and_b32 v22, -4, v1 // (op 30857) 
v_sub_u32 v24, v1, v22 // (op 30857) 

v_lshlrev_b32 v22, 5, v24 // (op 30857) 

v_lshrrev_b32 v24, 2, v1 // (op 30857) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30857) 


v_lshlrev_b32 v22, 2, v26 // (op 30857) 

v_lshrrev_b32 v24, 3, v22 // (op 30857) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 30857) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 30485) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30485) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30485) 


v_mov_b64 v[44:45], v[16:17] // (op 30485) 
s_add_u32 s1, 32768, s0 // (op 30485) 
s_mov_b32 m0, s1 // (op 30485) reg expression
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 30485) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 30485) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 30485) most significant half

s_add_u32 m0, m0, 2048 // (op 30485) 
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 30485) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 30485) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 30485) most significant half

s_add_u32 m0, m0, 2048 // (op 30485) 
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 30485) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 30485) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 30485) most significant half

s_add_u32 m0, m0, 2048 // (op 30485) 
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 30485) Load value direct to lds




v_and_b32 v22, -4, v1 // (op 30989) 
v_sub_u32 v24, v1, v22 // (op 30989) 

v_lshlrev_b32 v22, 5, v24 // (op 30989) 

v_lshrrev_b32 v24, 2, v1 // (op 30989) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30989) 


v_lshlrev_b32 v22, 2, v26 // (op 30989) 

v_lshrrev_b32 v24, 3, v22 // (op 30989) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 30989) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 30487) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30487) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30487) 


v_mov_b64 v[44:45], v[14:15] // (op 30487) 
s_add_u32 s1, 40960, s0 // (op 30487) 
s_mov_b32 m0, s1 // (op 30487) reg expression
s_mov_b32 s6, 128 // (op 30487) call()
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30487) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30487) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30487) most significant half

s_add_u32 m0, m0, 2048 // (op 30487) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30487) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30487) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30487) most significant half

s_add_u32 m0, m0, 2048 // (op 30487) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30487) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30487) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30487) most significant half

s_add_u32 m0, m0, 2048 // (op 30487) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30487) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30487) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30487) most significant half

s_add_u32 m0, m0, 2048 // (op 30487) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30487) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30487) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30487) most significant half

s_add_u32 m0, m0, 2048 // (op 30487) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30487) Load value direct to lds





v_and_b32 v22, -4, v1 // (op 30945) 
v_sub_u32 v24, v1, v22 // (op 30945) 

v_lshlrev_b32 v22, 5, v24 // (op 30945) 

v_lshrrev_b32 v24, 2, v1 // (op 30945) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30945) 


v_lshlrev_b32 v22, 2, v26 // (op 30945) 

v_lshrrev_b32 v24, 3, v22 // (op 30945) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 30945) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 30489) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30489) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30489) 


v_mov_b64 v[44:45], v[16:17] // (op 30489) 
s_add_u32 s1, 53248, s0 // (op 30489) 
s_mov_b32 m0, s1 // (op 30489) reg expression
s_mov_b32 s6, 128 // (op 30489) call()
buffer_load_dwordx4 v44, s[48:51], s6 offen offset: 0 lds // (op 30489) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30489) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30489) most significant half

s_add_u32 m0, m0, 2048 // (op 30489) 
buffer_load_dwordx4 v44, s[48:51], s6 offen offset: 0 lds // (op 30489) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30489) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30489) most significant half

s_add_u32 m0, m0, 2048 // (op 30489) 
buffer_load_dwordx4 v44, s[48:51], s6 offen offset: 0 lds // (op 30489) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30489) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30489) most significant half

s_add_u32 m0, m0, 2048 // (op 30489) 
buffer_load_dwordx4 v44, s[48:51], s6 offen offset: 0 lds // (op 30489) Load value direct to lds





v_and_b32 v22, -4, v1 // (op 31077) 
v_sub_u32 v24, v1, v22 // (op 31077) 

v_lshlrev_b32 v22, 5, v24 // (op 31077) 

v_lshrrev_b32 v24, 2, v1 // (op 31077) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 31077) 


v_lshlrev_b32 v22, 2, v26 // (op 31077) 

v_lshrrev_b32 v24, 3, v22 // (op 31077) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 31077) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 30491) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30491) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 30491) 


v_mov_b64 v[44:45], v[14:15] // (op 30491) 
s_add_u32 s1, 61440, s0 // (op 30491) 
s_mov_b32 m0, s1 // (op 30491) reg expression
s_mov_b32 s6, 192 // (op 30491) call()
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30491) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30491) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30491) most significant half

s_add_u32 m0, m0, 2048 // (op 30491) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30491) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30491) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30491) most significant half

s_add_u32 m0, m0, 2048 // (op 30491) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30491) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30491) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30491) most significant half

s_add_u32 m0, m0, 2048 // (op 30491) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30491) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30491) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30491) most significant half

s_add_u32 m0, m0, 2048 // (op 30491) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30491) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 30491) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 30491) most significant half

s_add_u32 m0, m0, 2048 // (op 30491) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 30491) Load value direct to lds






v_and_b32 v22, -4, v1 // (op 31033) 
v_sub_u32 v24, v1, v22 // (op 31033) 

v_lshlrev_b32 v22, 5, v24 // (op 31033) 

v_lshrrev_b32 v24, 2, v1 // (op 31033) 
v_lshl_add_u32 v14, v24, 7, v22 // (op 31033) 


v_lshlrev_b32 v15, 2, v14 // (op 31033) 

v_lshrrev_b32 v14, 3, v15 // (op 31033) 

s_nop 0
v_readfirstlane_b32 s0, v14 // (op 31033) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 5 // (op 30493) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 30493) 

v_lshrrev_b64 v[14:15], 3, s[10:11] // (op 30493) 


v_mov_b64 v[26:27], v[16:17] // (op 30493) 
s_add_u32 s1, 73728, s0 // (op 30493) 
s_mov_b32 m0, s1 // (op 30493) reg expression
s_mov_b32 s6, 192 // (op 30493) call()
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30493) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v14 // (op 30493) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30493) most significant half

s_add_u32 m0, m0, 2048 // (op 30493) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30493) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v14 // (op 30493) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30493) most significant half

s_add_u32 m0, m0, 2048 // (op 30493) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30493) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v14 // (op 30493) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 30493) most significant half

s_add_u32 m0, m0, 2048 // (op 30493) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 30493) Load value direct to lds






s_waitcnt vmcnt(38)
s_barrier  // (op 31790) 



ds_read_b128 v[14:17], v4 // (op 29506) Load local data 



ds_read_b128 v[44:47], v4 offset:1024 // (op 29511) Load local data 



ds_read_b128 v[48:51], v4 offset:2048 // (op 29516) Load local data 



ds_read_b128 v[52:55], v4 offset:3072 // (op 29521) Load local data 



ds_read_b128 v[56:59], v4 offset:4096 // (op 29526) Load local data 



ds_read_b128 v[60:63], v4 offset:5120 // (op 29531) Load local data 



ds_read_b128 v[64:67], v4 offset:6144 // (op 29536) Load local data 



ds_read_b128 v[68:71], v4 offset:7168 // (op 29541) Load local data 



ds_read_b128 v[72:75], v4 offset:8192 // (op 29546) Load local data 



ds_read_b128 v[76:79], v4 offset:9216 // (op 29551) Load local data 



ds_read_b128 v[80:83], v4 offset:10240 // (op 29556) Load local data 



ds_read_b128 v[84:87], v4 offset:11264 // (op 29561) Load local data 
s_waitcnt vmcnt(34)
s_barrier  // (op 31773) 



ds_read_b128 v[88:91], v0 offset:12288 // (op 29486) Load local data 



ds_read_b128 v[92:95], v0 offset:13312 // (op 29491) Load local data 



ds_read_b128 v[96:99], v0 offset:14336 // (op 29496) Load local data 



ds_read_b128 v[100:103], v0 offset:15360 // (op 29501) Load local data 
v_and_b32 v22, -4, v1 // (op 30718) 
v_sub_u32 v24, v1, v22 // (op 30718) 

v_lshlrev_b32 v22, 5, v24 // (op 30718) 

v_lshrrev_b32 v24, 2, v1 // (op 30718) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30718) 


v_lshlrev_b32 v22, 2, v26 // (op 30718) 

v_lshrrev_b32 v24, 3, v22 // (op 30718) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 30718) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 30775) 
v_sub_u32 v24, v1, v22 // (op 30775) 

v_lshlrev_b32 v22, 5, v24 // (op 30775) 

v_lshrrev_b32 v24, 2, v1 // (op 30775) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30775) 


v_lshlrev_b32 v22, 2, v26 // (op 30775) 

v_lshrrev_b32 v24, 3, v22 // (op 30775) 

s_nop 0
v_readfirstlane_b32 s1, v24 // (op 30775) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 30842) 
v_sub_u32 v24, v1, v22 // (op 30842) 

v_lshlrev_b32 v22, 5, v24 // (op 30842) 

v_lshrrev_b32 v24, 2, v1 // (op 30842) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30842) 


v_lshlrev_b32 v22, 2, v26 // (op 30842) 

v_lshrrev_b32 v24, 3, v22 // (op 30842) 

s_nop 0
v_readfirstlane_b32 s6, v24 // (op 30842) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 30886) 
v_sub_u32 v24, v1, v22 // (op 30886) 

v_lshlrev_b32 v22, 5, v24 // (op 30886) 

v_lshrrev_b32 v24, 2, v1 // (op 30886) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30886) 


v_lshlrev_b32 v22, 2, v26 // (op 30886) 

v_lshrrev_b32 v24, 3, v22 // (op 30886) 

s_nop 0
v_readfirstlane_b32 s7, v24 // (op 30886) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 30930) 
v_sub_u32 v24, v1, v22 // (op 30930) 

v_lshlrev_b32 v22, 5, v24 // (op 30930) 

v_lshrrev_b32 v24, 2, v1 // (op 30930) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30930) 


v_lshlrev_b32 v22, 2, v26 // (op 30930) 

v_lshrrev_b32 v24, 3, v22 // (op 30930) 

s_nop 0
v_readfirstlane_b32 s10, v24 // (op 30930) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 30974) 
v_sub_u32 v24, v1, v22 // (op 30974) 

v_lshlrev_b32 v22, 5, v24 // (op 30974) 

v_lshrrev_b32 v24, 2, v1 // (op 30974) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 30974) 


v_lshlrev_b32 v22, 2, v26 // (op 30974) 

v_lshrrev_b32 v24, 3, v22 // (op 30974) 

s_nop 0
v_readfirstlane_b32 s11, v24 // (op 30974) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 31018) 
v_sub_u32 v24, v1, v22 // (op 31018) 

v_lshlrev_b32 v22, 5, v24 // (op 31018) 

v_lshrrev_b32 v24, 2, v1 // (op 31018) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 31018) 


v_lshlrev_b32 v22, 2, v26 // (op 31018) 

v_lshrrev_b32 v24, 3, v22 // (op 31018) 

s_nop 0
v_readfirstlane_b32 s14, v24 // (op 31018) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 31062) 
v_sub_u32 v24, v1, v22 // (op 31062) 

v_lshlrev_b32 v22, 5, v24 // (op 31062) 

v_lshrrev_b32 v24, 2, v1 // (op 31062) 
v_lshl_add_u32 v26, v24, 7, v22 // (op 31062) 


v_lshlrev_b32 v22, 2, v26 // (op 31062) 

v_lshrrev_b32 v24, 3, v22 // (op 31062) 

s_nop 0
v_readfirstlane_b32 s15, v24 // (op 31062) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 9580) call()
s_ashr_i64 s[38:39], s[36:37], 7 // (op 9579) 
s_mov_b32 s19, s38 // (op 9579) convert

s_and_b32 s38, s19, 4294967292 // (op 9579) 

s_cmp_lt_u32 s18, s38 // (op 9579) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopBottom_KLoop_9579 // (op 9579) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopBottom_KLoop_9579 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopTop_KLoop_9579:
s_add_u32 s18, s18, 4 // (op 9581) 
s_waitcnt vmcnt(33)
v_mov_b32 v104, v28 // (op 30331) reg expression
v_mov_b32 v105, v29 // (op 30331) reg expression
v_mov_b32 v106, v30 // (op 30331) reg expression
v_mov_b32 v107, v31 // (op 30331) reg expression
s_waitcnt lgkmcnt(4)
s_barrier  // (op 31793) 
s_waitcnt vmcnt(32)
v_mov_b32 v108, v32 // (op 30344) reg expression

v_mov_b32 v109, v33 // (op 30344) reg expression
v_mov_b32 v110, v34 // (op 30344) reg expression
v_mov_b32 v111, v35 // (op 30344) reg expression
s_lshl_b64 s[38:39], s[24:25], 5 // (op 30426) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30426) 

v_lshrrev_b64 v[26:27], 3, s[64:65] // (op 30426) 


v_mov_b64 v[112:113], v[20:21] // (op 30426) 
v_mov_b64 v[114:115], v[112:113] // (op 30426) 
s_mov_b32 m0, s1 // (op 30426) reg expression
s_mov_b32 s19, 256 // (op 30426) call()
buffer_load_dwordx4 v114, s[52:55], s19 offen offset: 0 lds // (op 30426) Load value direct to lds
v_add_co_u32 v114, s[38:39], v114, v26 // (op 30426) least significant half
v_addc_co_u32 v115, s[38:39], 0, v115, s[38:39] // (op 30426) most significant half

s_add_u32 m0, m0, 2048 // (op 30426) 
buffer_load_dwordx4 v114, s[52:55], s19 offen offset: 0 lds // (op 30426) Load value direct to lds
v_add_co_u32 v114, s[38:39], v114, v26 // (op 30426) least significant half
v_addc_co_u32 v115, s[38:39], 0, v115, s[38:39] // (op 30426) most significant half

s_add_u32 m0, m0, 2048 // (op 30426) 
buffer_load_dwordx4 v114, s[52:55], s19 offen offset: 0 lds // (op 30426) Load value direct to lds
v_add_co_u32 v114, s[38:39], v114, v26 // (op 30426) least significant half
v_addc_co_u32 v115, s[38:39], 0, v115, s[38:39] // (op 30426) most significant half

s_add_u32 m0, m0, 2048 // (op 30426) 
s_waitcnt vmcnt(34)
v_mov_b32 v116, v36 // (op 30357) reg expression
v_mov_b32 v117, v37 // (op 30357) reg expression
v_mov_b32 v118, v38 // (op 30357) reg expression
v_mov_b32 v119, v39 // (op 30357) reg expression
s_waitcnt vmcnt(33)
v_mov_b32 v120, v40 // (op 30370) reg expression
v_mov_b32 v121, v41 // (op 30370) reg expression
v_mov_b32 v122, v42 // (op 30370) reg expression
v_mov_b32 v123, v43 // (op 30370) reg expression
buffer_load_dwordx4 v114, s[52:55], s19 offen offset: 0 lds // (op 30426) Load value direct to lds
v_add_co_u32 v114, s[38:39], v114, v26 // (op 30426) least significant half
v_addc_co_u32 v115, s[38:39], 0, v115, s[38:39] // (op 30426) most significant half

s_add_u32 m0, m0, 2048 // (op 30426) 
buffer_load_dwordx4 v114, s[52:55], s19 offen offset: 0 lds // (op 30426) Load value direct to lds
v_add_co_u32 v114, s[38:39], v114, v26 // (op 30426) least significant half
v_addc_co_u32 v115, s[38:39], 0, v115, s[38:39] // (op 30426) most significant half

s_add_u32 m0, m0, 2048 // (op 30426) 
v_bfe_u32 v124, v120, 0, 8 // (op 27645) 
v_bfe_u32 v125, v120, 8, 8 // (op 27645) 
v_bfe_u32 v126, v120, 16, 8 // (op 27645) 
v_bfe_u32 v127, v120, 24, 8 // (op 27645) 
v_permlane16_swap_b32 v124, v125 // (op 27645) 
s_nop 0
v_permlane16_swap_b32 v126, v127 // (op 27645) Wait state hazard: VALU Write Readlane Hazard
buffer_load_dwordx4 v114, s[52:55], s19 offen offset: 0 lds // (op 30426) Load value direct to lds



s_nop 0
v_permlane32_swap_b32 v124, v126 // (op 27645) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v125, v127 // (op 27645) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 31776) 

s_lshl_b64 s[38:39], s[12:13], 5 // (op 30421) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30421) 

v_lshrrev_b64 v[26:27], 3, s[64:65] // (op 30421) 


v_mov_b64 v[114:115], v[18:19] // (op 30421) 
v_mov_b64 v[128:129], v[114:115] // (op 30421) 
s_add_u32 s19, 12288, s0 // (op 30421) 
s_mov_b32 m0, s19 // (op 30421) reg expression
s_mov_b32 s38, 256 // (op 30421) call()
v_bfe_u32 v130, v104, 0, 8 // (op 27902) 
v_bfe_u32 v131, v104, 8, 8 // (op 27902) 
v_bfe_u32 v132, v104, 16, 8 // (op 27902) 
v_bfe_u32 v133, v104, 24, 8 // (op 27902) 
v_permlane16_swap_b32 v130, v131 // (op 27902) 
s_nop 0
v_permlane16_swap_b32 v132, v133 // (op 27902) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v130, v132 // (op 27902) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v131, v133 // (op 27902) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 30421) Load value direct to lds
v_add_co_u32 v128, s[64:65], v128, v26 // (op 30421) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[14:17], a[0:3], v124, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 76) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 30421) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[44:47], a[4:7], v124, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 417) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[48:51], a[8:11], v124, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 643) 
s_add_u32 m0, m0, 2048 // (op 30421) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[52:55], a[12:15], v124, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 869) 
v_bfe_u32 v134, v108, 0, 8 // (op 27970) 
v_bfe_u32 v135, v108, 8, 8 // (op 27970) 
v_bfe_u32 v136, v108, 16, 8 // (op 27970) 
v_bfe_u32 v137, v108, 24, 8 // (op 27970) 
v_permlane16_swap_b32 v134, v135 // (op 27970) 
s_nop 0
v_permlane16_swap_b32 v136, v137 // (op 27970) Wait state hazard: VALU Write Readlane Hazard
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 30421) Load value direct to lds
v_add_co_u32 v128, s[64:65], v128, v26 // (op 30421) least significant half
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 30421) most significant half

s_add_u32 m0, m0, 2048 // (op 30421) 
v_permlane32_swap_b32 v134, v136 // (op 27970) 
v_permlane32_swap_b32 v135, v137 // (op 27970) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 30421) Load value direct to lds
v_add_co_u32 v128, s[64:65], v128, v26 // (op 30421) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[56:59], a[16:19], v124, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1095) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 30421) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[60:63], a[20:23], v124, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1321) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[64:67], a[24:27], v124, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1547) 
s_add_u32 m0, m0, 2048 // (op 30421) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[68:71], a[28:31], v124, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1773) 
v_bfe_u32 v138, v116, 0, 8 // (op 28038) 
v_bfe_u32 v139, v116, 8, 8 // (op 28038) 
v_bfe_u32 v140, v116, 16, 8 // (op 28038) 
v_bfe_u32 v141, v116, 24, 8 // (op 28038) 
v_permlane16_swap_b32 v138, v139 // (op 28038) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 28038) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 28038) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 28038) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 30421) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[72:75], a[32:35], v124, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1999) 
v_mov_b32 v22, v23 // (op 30339) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[76:79], a[36:39], v124, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2225) 
v_add_u32 v24, 16, v22 // (op 30339) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[80:83], a[40:43], v124, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2451) 
buffer_load_dwordx4 v[28:31], v24, s[60:63], 0 offen // (op 30339) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[84:87], a[44:47], v124, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2677) 
s_lshl_b32 s19, s32, 6 // (op 30352) 
s_add_u32 s19, s19, 16 // (op 30352) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[14:17], a[48:51], v125, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4445) 
v_add_u32 v24, v22, s19 // (op 30352) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[44:47], a[52:55], v125, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4369) 
buffer_load_dwordx4 v[32:35], v24, s[60:63], 0 offen // (op 30352) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[48:51], a[56:59], v125, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4293) 
s_lshl_b32 s19, s32, 7 // (op 30365) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[52:55], a[60:63], v125, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4217) 
s_add_u32 s19, s19, 16 // (op 30365) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:95], v[56:59], a[64:67], v125, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4141) 
v_add_u32 v24, v22, s19 // (op 30365) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:95], v[60:63], a[68:71], v125, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4065) 
s_waitcnt vmcnt(36)
s_barrier  // (op 31813) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[64:67], a[72:75], v125, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3989) 
ds_read_b128 v[142:145], v6 offset:20480 // (op 212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[68:71], a[76:79], v125, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3913) 
ds_read_b128 v[146:149], v6 offset:21504 // (op 428) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[72:75], a[80:83], v125, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3837) 
ds_read_b128 v[150:153], v6 offset:22528 // (op 654) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[76:79], a[84:87], v125, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3761) 
ds_read_b128 v[154:157], v6 offset:23552 // (op 880) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[80:83], a[88:91], v125, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3685) 
ds_read_b128 v[158:161], v6 offset:24576 // (op 1106) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[84:87], a[92:95], v125, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3609) 
ds_read_b128 v[162:165], v6 offset:25600 // (op 1332) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[14:17], a[96:99], v126, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7447) 
ds_read_b128 v[166:169], v6 offset:26624 // (op 1558) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[44:47], a[100:103], v126, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7371) 
ds_read_b128 v[170:173], v6 offset:27648 // (op 1784) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:99], v[48:51], a[104:107], v126, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7295) 
buffer_load_dwordx4 v[36:39], v24, s[60:63], 0 offen // (op 30365) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:99], v[52:55], a[108:111], v126, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7219) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:99], v[56:59], a[112:115], v126, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7143) 
v_add_u32 v23, 16, v23 // (op 31179) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:99], v[60:63], a[116:119], v126, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7067) 
v_mov_b32 v22, v25 // (op 30378) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:99], v[64:67], a[120:123], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6991) 
v_add_u32 v24, 16, v22 // (op 30378) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:99], v[68:71], a[124:127], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6915) 
buffer_load_dwordx4 v[40:43], v24, s[56:59], 0 offen // (op 30378) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[72:75], a[128:131], v126, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6839) 
v_add_u32 v25, 16, v25 // (op 31127) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[76:79], a[132:135], v126, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6763) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[80:83], a[136:139], v126, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6687) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[84:87], a[140:143], v126, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6611) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[100:103], v[14:17], a[144:147], v127, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10449) 
ds_read_b128 v[174:177], v6 offset:28672 // (op 2010) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[100:103], v[44:47], a[148:151], v127, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10373) 
ds_read_b128 v[178:181], v6 offset:29696 // (op 2236) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[48:51], a[152:155], v127, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10297) 
ds_read_b128 v[182:185], v6 offset:30720 // (op 2462) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[52:55], a[156:159], v127, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10221) 
ds_read_b128 v[186:189], v6 offset:31744 // (op 2688) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[100:103], v[56:59], a[160:163], v127, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10145) 
s_waitcnt vmcnt(34)
s_barrier  // (op 31806) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[100:103], v[60:63], a[164:167], v127, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10069) 
ds_read_b128 v[88:91], v3 offset:32768 // (op 210) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[100:103], v[64:67], a[168:171], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9993) 
ds_read_b128 v[92:95], v3 offset:33792 // (op 3619) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[100:103], v[68:71], a[172:175], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9917) 
ds_read_b128 v[96:99], v3 offset:34816 // (op 6621) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[72:75], a[176:179], v127, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9841) 
ds_read_b128 v[128:131], v3 offset:35840 // (op 9623) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[76:79], a[180:183], v127, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9765) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[80:83], a[184:187], v127, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9689) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[84:87], a[188:191], v127, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9613) 


v_bfe_u32 v124, v121, 0, 8 // (op 27660) 
v_bfe_u32 v125, v121, 8, 8 // (op 27660) 
v_bfe_u32 v126, v121, 16, 8 // (op 27660) 
v_bfe_u32 v127, v121, 24, 8 // (op 27660) 
v_permlane16_swap_b32 v124, v125 // (op 27660) 
s_nop 0
v_permlane16_swap_b32 v126, v127 // (op 27660) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v124, v126 // (op 27660) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v125, v127 // (op 27660) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 31817) 

s_lshl_b64 s[38:39], s[24:25], 5 // (op 30448) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30448) 

v_lshrrev_b64 v[26:27], 3, s[64:65] // (op 30448) 


v_mov_b64 v[132:133], v[112:113] // (op 30448) 
s_add_u32 s19, 20480, s7 // (op 30448) 
s_mov_b32 m0, s19 // (op 30448) reg expression
s_mov_b32 s38, 320 // (op 30448) call()
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 30448) Load value direct to lds
v_add_co_u32 v132, s[64:65], v132, v26 // (op 30448) least significant half
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 30448) most significant half

s_add_u32 m0, m0, 2048 // (op 30448) 
v_bfe_u32 v134, v105, 0, 8 // (op 27917) 
v_bfe_u32 v135, v105, 8, 8 // (op 27917) 
v_bfe_u32 v136, v105, 16, 8 // (op 27917) 
v_bfe_u32 v137, v105, 24, 8 // (op 27917) 
v_permlane16_swap_b32 v134, v135 // (op 27917) 
s_nop 0
v_permlane16_swap_b32 v136, v137 // (op 27917) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v134, v136 // (op 27917) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v135, v137 // (op 27917) 
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 30448) Load value direct to lds
v_add_co_u32 v132, s[64:65], v132, v26 // (op 30448) least significant half
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 30448) most significant half

s_add_u32 m0, m0, 2048 // (op 30448) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[142:145], a[0:3], v124, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 217) 
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 30448) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[146:149], a[4:7], v124, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 418) 
v_add_co_u32 v132, s[64:65], v132, v26 // (op 30448) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[150:153], a[8:11], v124, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 644) 
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 30448) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[154:157], a[12:15], v124, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 870) 
s_add_u32 m0, m0, 2048 // (op 30448) 
v_bfe_u32 v138, v109, 0, 8 // (op 27985) 
v_bfe_u32 v139, v109, 8, 8 // (op 27985) 
v_bfe_u32 v140, v109, 16, 8 // (op 27985) 
v_bfe_u32 v141, v109, 24, 8 // (op 27985) 
v_permlane16_swap_b32 v138, v139 // (op 27985) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 27985) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 27985) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 27985) 
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 30448) Load value direct to lds
v_add_co_u32 v132, s[64:65], v132, v26 // (op 30448) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[158:161], a[16:19], v124, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1096) 
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 30448) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[162:165], a[20:23], v124, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1322) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[166:169], a[24:27], v124, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1548) 
s_add_u32 m0, m0, 2048 // (op 30448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[170:173], a[28:31], v124, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1774) 
v_bfe_u32 v190, v117, 0, 8 // (op 28053) 
v_bfe_u32 v191, v117, 8, 8 // (op 28053) 
v_bfe_u32 v192, v117, 16, 8 // (op 28053) 
v_bfe_u32 v193, v117, 24, 8 // (op 28053) 
v_permlane16_swap_b32 v190, v191 // (op 28053) 
s_nop 0
v_permlane16_swap_b32 v192, v193 // (op 28053) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v190, v192 // (op 28053) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v191, v193 // (op 28053) 
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 30448) Load value direct to lds
v_add_co_u32 v132, s[64:65], v132, v26 // (op 30448) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[174:177], a[32:35], v124, v190 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2000) 
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 30448) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[178:181], a[36:39], v124, v191 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2226) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[182:185], a[40:43], v124, v192 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2452) 
s_add_u32 m0, m0, 2048 // (op 30448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[186:189], a[44:47], v124, v193 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2678) 
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 30448) Load value direct to lds




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[142:145], a[48:51], v125, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4446) 
s_waitcnt vmcnt(34)
s_barrier  // (op 31828) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[146:149], a[52:55], v125, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4370) 
ds_read_b128 v[14:17], v8 offset:40960 // (op 240) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[150:153], a[56:59], v125, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4294) 
ds_read_b128 v[44:47], v8 offset:41984 // (op 438) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[154:157], a[60:63], v125, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4218) 
ds_read_b128 v[48:51], v8 offset:43008 // (op 664) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:95], v[158:161], a[64:67], v125, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4142) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 31810) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:95], v[162:165], a[68:71], v125, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4066) 
s_lshl_b64 s[38:39], s[12:13], 5 // (op 30443) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[166:169], a[72:75], v125, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3990) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30443) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[170:173], a[76:79], v125, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3914) 
v_lshrrev_b64 v[26:27], 3, s[64:65] // (op 30443) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[174:177], a[80:83], v125, v190 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3838) 
v_mov_b64 v[132:133], v[114:115] // (op 30443) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[178:181], a[84:87], v125, v191 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3762) 
s_add_u32 s19, 32768, s6 // (op 30443) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[182:185], a[88:91], v125, v192 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3686) 
s_mov_b32 m0, s19 // (op 30443) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[186:189], a[92:95], v125, v193 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3610) 
s_mov_b32 s38, 320 // (op 30443) call()
ds_read_b128 v[52:55], v8 offset:44032 // (op 890) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[142:145], a[96:99], v126, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7448) 
ds_read_b128 v[56:59], v8 offset:45056 // (op 1116) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[146:149], a[100:103], v126, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7372) 
ds_read_b128 v[60:63], v8 offset:46080 // (op 1342) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:99], v[150:153], a[104:107], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7296) 
ds_read_b128 v[64:67], v8 offset:47104 // (op 1568) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:99], v[154:157], a[108:111], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7220) 
ds_read_b128 v[68:71], v8 offset:48128 // (op 1794) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:99], v[158:161], a[112:115], v126, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7144) 
buffer_load_dwordx4 v132, s[48:51], s38 offen offset: 0 lds // (op 30443) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:99], v[162:165], a[116:119], v126, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7068) 
v_add_co_u32 v132, s[64:65], v132, v26 // (op 30443) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:99], v[166:169], a[120:123], v126, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6992) 
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 30443) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:99], v[170:173], a[124:127], v126, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6916) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[174:177], a[128:131], v126, v190 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6840) 
s_add_u32 m0, m0, 2048 // (op 30443) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[178:181], a[132:135], v126, v191 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6764) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[182:185], a[136:139], v126, v192 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6688) 
buffer_load_dwordx4 v132, s[48:51], s38 offen offset: 0 lds // (op 30443) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[186:189], a[140:143], v126, v193 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6612) 
v_add_co_u32 v132, s[64:65], v132, v26 // (op 30443) least significant half
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 30443) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[128:131], v[142:145], a[144:147], v127, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10450) 

s_add_u32 m0, m0, 2048 // (op 30443) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[128:131], v[146:149], a[148:151], v127, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10374) 

ds_read_b128 v[72:75], v8 offset:49152 // (op 2020) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[128:131], v[150:153], a[152:155], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10298) 
ds_read_b128 v[142:145], v8 offset:50176 // (op 2246) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[128:131], v[154:157], a[156:159], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10222) 
ds_read_b128 v[146:149], v8 offset:51200 // (op 2472) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[128:131], v[158:161], a[160:163], v127, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10146) 
buffer_load_dwordx4 v132, s[48:51], s38 offen offset: 0 lds // (op 30443) Load value direct to lds

v_add_co_u32 v132, s[64:65], v132, v26 // (op 30443) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[162:165], a[164:167], v127, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10070) 
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 30443) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[128:131], v[166:169], a[168:171], v127, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9994) 
s_add_u32 m0, m0, 2048 // (op 30443) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[128:131], v[170:173], a[172:175], v127, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9918) 
ds_read_b128 v[134:137], v8 offset:52224 // (op 2698) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[128:131], v[174:177], a[176:179], v127, v190 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9842) 
s_waitcnt vmcnt(33)
s_barrier  // (op 31821) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[178:181], a[180:183], v127, v191 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9766) 
ds_read_b128 v[88:91], v5 offset:53248 // (op 238) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[128:131], v[182:185], a[184:187], v127, v192 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9690) 
ds_read_b128 v[92:95], v5 offset:54272 // (op 3629) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[128:131], v[186:189], a[188:191], v127, v193 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9614) 
ds_read_b128 v[96:99], v5 offset:55296 // (op 6631) Load local data 







ds_read_b128 v[100:103], v5 offset:56320 // (op 9633) Load local data 
buffer_load_dwordx4 v132, s[48:51], s38 offen offset: 0 lds // (op 30443) Load value direct to lds




v_bfe_u32 v138, v122, 0, 8 // (op 27677) 
v_bfe_u32 v139, v122, 8, 8 // (op 27677) 
v_bfe_u32 v140, v122, 16, 8 // (op 27677) 
v_bfe_u32 v141, v122, 24, 8 // (op 27677) 
v_permlane16_swap_b32 v138, v139 // (op 27677) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 27677) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 27677) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 27677) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 31831) 

s_lshl_b64 s[38:39], s[24:25], 5 // (op 30458) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30458) 

v_lshrrev_b64 v[26:27], 3, s[64:65] // (op 30458) 


v_mov_b64 v[124:125], v[112:113] // (op 30458) 
s_add_u32 s19, 40960, s11 // (op 30458) 
s_mov_b32 m0, s19 // (op 30458) reg expression
s_mov_b32 s38, 384 // (op 30458) call()
v_bfe_u32 v126, v106, 0, 8 // (op 27935) 
v_bfe_u32 v127, v106, 8, 8 // (op 27935) 
v_bfe_u32 v128, v106, 16, 8 // (op 27935) 
v_bfe_u32 v129, v106, 24, 8 // (op 27935) 
v_permlane16_swap_b32 v126, v127 // (op 27935) 
s_nop 0
v_permlane16_swap_b32 v128, v129 // (op 27935) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 27935) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 27935) 
buffer_load_dwordx4 v124, s[52:55], s38 offen offset: 0 lds // (op 30458) Load value direct to lds
v_add_co_u32 v124, s[64:65], v124, v26 // (op 30458) least significant half
v_addc_co_u32 v125, s[64:65], 0, v125, s[64:65] // (op 30458) most significant half

s_add_u32 m0, m0, 2048 // (op 30458) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[14:17], a[0:3], v138, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 245) 
buffer_load_dwordx4 v124, s[52:55], s38 offen offset: 0 lds // (op 30458) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[44:47], a[4:7], v138, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 419) 
v_add_co_u32 v124, s[64:65], v124, v26 // (op 30458) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[48:51], a[8:11], v138, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 645) 
v_addc_co_u32 v125, s[64:65], 0, v125, s[64:65] // (op 30458) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[52:55], a[12:15], v138, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 871) 
s_add_u32 m0, m0, 2048 // (op 30458) 
v_bfe_u32 v130, v110, 0, 8 // (op 28003) 
v_bfe_u32 v131, v110, 8, 8 // (op 28003) 
v_bfe_u32 v132, v110, 16, 8 // (op 28003) 
v_bfe_u32 v133, v110, 24, 8 // (op 28003) 
v_permlane16_swap_b32 v130, v131 // (op 28003) 
s_nop 0
v_permlane16_swap_b32 v132, v133 // (op 28003) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v130, v132 // (op 28003) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v131, v133 // (op 28003) 
buffer_load_dwordx4 v124, s[52:55], s38 offen offset: 0 lds // (op 30458) Load value direct to lds
v_add_co_u32 v124, s[64:65], v124, v26 // (op 30458) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[56:59], a[16:19], v138, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1097) 
v_addc_co_u32 v125, s[64:65], 0, v125, s[64:65] // (op 30458) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[60:63], a[20:23], v138, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1323) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[64:67], a[24:27], v138, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1549) 
s_add_u32 m0, m0, 2048 // (op 30458) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[68:71], a[28:31], v138, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1775) 
v_bfe_u32 v150, v118, 0, 8 // (op 28071) 
v_bfe_u32 v151, v118, 8, 8 // (op 28071) 
v_bfe_u32 v152, v118, 16, 8 // (op 28071) 
v_bfe_u32 v153, v118, 24, 8 // (op 28071) 
v_permlane16_swap_b32 v150, v151 // (op 28071) 
s_nop 0
v_permlane16_swap_b32 v152, v153 // (op 28071) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v150, v152 // (op 28071) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v151, v153 // (op 28071) 
buffer_load_dwordx4 v124, s[52:55], s38 offen offset: 0 lds // (op 30458) Load value direct to lds
v_add_co_u32 v124, s[64:65], v124, v26 // (op 30458) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[72:75], a[32:35], v138, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2001) 
v_addc_co_u32 v125, s[64:65], 0, v125, s[64:65] // (op 30458) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[142:145], a[36:39], v138, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2227) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[146:149], a[40:43], v138, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2453) 
s_add_u32 m0, m0, 2048 // (op 30458) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[134:137], a[44:47], v138, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2679) 
buffer_load_dwordx4 v124, s[52:55], s38 offen offset: 0 lds // (op 30458) Load value direct to lds
v_add_co_u32 v124, s[64:65], v124, v26 // (op 30458) least significant half
v_addc_co_u32 v125, s[64:65], 0, v125, s[64:65] // (op 30458) most significant half

s_add_u32 m0, m0, 2048 // (op 30458) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[14:17], a[48:51], v139, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4447) 
s_waitcnt vmcnt(33)
s_barrier  // (op 31841) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[44:47], a[52:55], v139, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4371) 
ds_read_b128 v[76:79], v10 offset:61440 // (op 268) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[48:51], a[56:59], v139, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4295) 
ds_read_b128 v[80:83], v10 offset:62464 // (op 448) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[52:55], a[60:63], v139, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4219) 
ds_read_b128 v[84:87], v10 offset:63488 // (op 674) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:95], v[56:59], a[64:67], v139, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4143) 
ds_read_b128 v[154:157], v10 offset:64512 // (op 900) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:95], v[60:63], a[68:71], v139, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4067) 
v_add_u32 v22, 65536, v10 // (op 1126) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[64:67], a[72:75], v139, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3991) 
ds_read_b128 v[158:161], v22 // (op 1126) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[68:71], a[76:79], v139, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3915) 
v_add_u32 v22, 66560, v10 // (op 1352) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[72:75], a[80:83], v139, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3839) 
ds_read_b128 v[162:165], v22 // (op 1352) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[142:145], a[84:87], v139, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3763) 
v_add_u32 v22, 67584, v10 // (op 1578) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[146:149], a[88:91], v139, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3687) 
ds_read_b128 v[166:169], v22 // (op 1578) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[134:137], a[92:95], v139, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3611) 
buffer_load_dwordx4 v124, s[52:55], s38 offen offset: 0 lds // (op 30458) Load value direct to lds







s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[14:17], a[96:99], v140, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7449) 
v_add_u32 v22, 68608, v10 // (op 1804) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[44:47], a[100:103], v140, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7373) 
ds_read_b128 v[170:173], v22 // (op 1804) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:99], v[48:51], a[104:107], v140, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7297) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 31825) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:99], v[52:55], a[108:111], v140, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7221) 
s_lshl_b64 s[38:39], s[12:13], 5 // (op 30453) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:99], v[56:59], a[112:115], v140, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7145) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30453) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:99], v[60:63], a[116:119], v140, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7069) 
v_lshrrev_b64 v[26:27], 3, s[64:65] // (op 30453) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:99], v[64:67], a[120:123], v140, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6993) 
v_mov_b64 v[124:125], v[114:115] // (op 30453) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:99], v[68:71], a[124:127], v140, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6917) 
s_add_u32 s19, 53248, s10 // (op 30453) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[72:75], a[128:131], v140, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6841) 
s_mov_b32 m0, s19 // (op 30453) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[142:145], a[132:135], v140, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6765) 
s_mov_b32 s38, 384 // (op 30453) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[146:149], a[136:139], v140, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6689) 
buffer_load_dwordx4 v124, s[48:51], s38 offen offset: 0 lds // (op 30453) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[134:137], a[140:143], v140, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6613) 
v_add_co_u32 v124, s[64:65], v124, v26 // (op 30453) least significant half
v_addc_co_u32 v125, s[64:65], 0, v125, s[64:65] // (op 30453) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[100:103], v[14:17], a[144:147], v141, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10451) 
s_add_u32 m0, m0, 2048 // (op 30453) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[100:103], v[44:47], a[148:151], v141, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10375) 
v_add_u32 v22, 69632, v10 // (op 2030) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[48:51], a[152:155], v141, v128 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10299) 
ds_read_b128 v[174:177], v22 // (op 2030) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[52:55], a[156:159], v141, v129 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10223) 
v_add_u32 v22, 70656, v10 // (op 2256) 

ds_read_b128 v[126:129], v22 // (op 2256) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[100:103], v[56:59], a[160:163], v141, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10147) 
v_add_u32 v22, 71680, v10 // (op 2482) 
ds_read_b128 v[178:181], v22 // (op 2482) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[100:103], v[60:63], a[164:167], v141, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10071) 
v_add_u32 v22, 72704, v10 // (op 2708) 
buffer_load_dwordx4 v124, s[48:51], s38 offen offset: 0 lds // (op 30453) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[100:103], v[64:67], a[168:171], v141, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9995) 
v_add_co_u32 v124, s[64:65], v124, v26 // (op 30453) least significant half
v_addc_co_u32 v125, s[64:65], 0, v125, s[64:65] // (op 30453) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[100:103], v[68:71], a[172:175], v141, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9919) 

s_add_u32 m0, m0, 2048 // (op 30453) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[72:75], a[176:179], v141, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9843) 
ds_read_b128 v[130:133], v22 // (op 2708) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[142:145], a[180:183], v141, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9767) 

s_waitcnt vmcnt(32)
s_barrier  // (op 31834) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[146:149], a[184:187], v141, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9691) 
v_add_u32 v22, 73728, v7 // (op 266) 

ds_read_b128 v[88:91], v22 // (op 266) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[134:137], a[188:191], v141, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9615) 
v_add_u32 v22, 74752, v7 // (op 3639) 



ds_read_b128 v[92:95], v22 // (op 3639) Load local data 

buffer_load_dwordx4 v124, s[48:51], s38 offen offset: 0 lds // (op 30453) Load value direct to lds



v_add_u32 v22, 75776, v7 // (op 6641) 
ds_read_b128 v[96:99], v22 // (op 6641) Load local data 

v_add_co_u32 v124, s[64:65], v124, v26 // (op 30453) least significant half



v_addc_co_u32 v125, s[64:65], 0, v125, s[64:65] // (op 30453) most significant half

s_add_u32 m0, m0, 2048 // (op 30453) 
v_add_u32 v22, 76800, v7 // (op 9643) 
ds_read_b128 v[134:137], v22 // (op 9643) Load local data 

buffer_load_dwordx4 v124, s[48:51], s38 offen offset: 0 lds // (op 30453) Load value direct to lds




v_bfe_u32 v138, v123, 0, 8 // (op 27694) 
v_bfe_u32 v139, v123, 8, 8 // (op 27694) 
v_bfe_u32 v140, v123, 16, 8 // (op 27694) 
v_bfe_u32 v141, v123, 24, 8 // (op 27694) 
v_permlane16_swap_b32 v138, v139 // (op 27694) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 27694) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 27694) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 27694) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 31844) 


s_lshl_b64 s[38:39], s[24:25], 5 // (op 30468) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30468) 

v_lshrrev_b64 v[26:27], 3, s[64:65] // (op 30468) 


v_mov_b64 v[120:121], v[112:113] // (op 30468) 
s_add_u32 s19, 61440, s15 // (op 30468) 
s_mov_b32 m0, s19 // (op 30468) reg expression
s_mov_b32 s38, 448 // (op 30468) call()
v_bfe_u32 v122, v107, 0, 8 // (op 27953) 
v_bfe_u32 v123, v107, 8, 8 // (op 27953) 
v_bfe_u32 v124, v107, 16, 8 // (op 27953) 
v_bfe_u32 v125, v107, 24, 8 // (op 27953) 
v_permlane16_swap_b32 v122, v123 // (op 27953) 
s_nop 0
v_permlane16_swap_b32 v124, v125 // (op 27953) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v122, v124 // (op 27953) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v123, v125 // (op 27953) 
buffer_load_dwordx4 v120, s[52:55], s38 offen offset: 0 lds // (op 30468) Load value direct to lds

v_add_co_u32 v120, s[64:65], v120, v26 // (op 30468) least significant half
v_addc_co_u32 v121, s[64:65], 0, v121, s[64:65] // (op 30468) most significant half

s_add_u32 m0, m0, 2048 // (op 30468) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[88:91], v[76:79], a[0:3], v138, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 273) 
buffer_load_dwordx4 v120, s[52:55], s38 offen offset: 0 lds // (op 30468) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[88:91], v[80:83], a[4:7], v138, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 420) 
v_add_co_u32 v120, s[64:65], v120, v26 // (op 30468) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[88:91], v[84:87], a[8:11], v138, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 646) 
v_addc_co_u32 v121, s[64:65], 0, v121, s[64:65] // (op 30468) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[88:91], v[154:157], a[12:15], v138, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 872) 
s_add_u32 m0, m0, 2048 // (op 30468) 
v_bfe_u32 v104, v111, 0, 8 // (op 28021) 
v_bfe_u32 v105, v111, 8, 8 // (op 28021) 
v_bfe_u32 v106, v111, 16, 8 // (op 28021) 
v_bfe_u32 v107, v111, 24, 8 // (op 28021) 
v_permlane16_swap_b32 v104, v105 // (op 28021) 
s_nop 0
v_permlane16_swap_b32 v106, v107 // (op 28021) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v104, v106 // (op 28021) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v105, v107 // (op 28021) 
buffer_load_dwordx4 v120, s[52:55], s38 offen offset: 0 lds // (op 30468) Load value direct to lds

v_add_co_u32 v120, s[64:65], v120, v26 // (op 30468) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:91], v[158:161], a[16:19], v138, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1098) 
v_addc_co_u32 v121, s[64:65], 0, v121, s[64:65] // (op 30468) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:91], v[162:165], a[20:23], v138, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1324) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:91], v[166:169], a[24:27], v138, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1550) 
s_add_u32 m0, m0, 2048 // (op 30468) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:91], v[170:173], a[28:31], v138, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1776) 
v_bfe_u32 v108, v119, 0, 8 // (op 28089) 
v_bfe_u32 v109, v119, 8, 8 // (op 28089) 
v_bfe_u32 v110, v119, 16, 8 // (op 28089) 
v_bfe_u32 v111, v119, 24, 8 // (op 28089) 
v_permlane16_swap_b32 v108, v109 // (op 28089) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 28089) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 28089) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 28089) 
buffer_load_dwordx4 v120, s[52:55], s38 offen offset: 0 lds // (op 30468) Load value direct to lds

v_add_co_u32 v120, s[64:65], v120, v26 // (op 30468) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:91], v[174:177], a[32:35], v138, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2002) 
v_addc_co_u32 v121, s[64:65], 0, v121, s[64:65] // (op 30468) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:91], v[126:129], a[36:39], v138, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2228) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:91], v[178:181], a[40:43], v138, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2454) 
s_add_u32 m0, m0, 2048 // (op 30468) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:91], v[130:133], a[44:47], v138, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2680) 
buffer_load_dwordx4 v120, s[52:55], s38 offen offset: 0 lds // (op 30468) Load value direct to lds
v_add_co_u32 v120, s[64:65], v120, v26 // (op 30468) least significant half
v_addc_co_u32 v121, s[64:65], 0, v121, s[64:65] // (op 30468) most significant half

s_add_u32 m0, m0, 2048 // (op 30468) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[92:95], v[76:79], a[48:51], v139, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4448) 
s_waitcnt vmcnt(33)
s_barrier  // (op 31799) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[92:95], v[80:83], a[52:55], v139, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4372) 
ds_read_b128 v[14:17], v4 // (op 65) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[92:95], v[84:87], a[56:59], v139, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4296) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 414) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[92:95], v[154:157], a[60:63], v139, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4220) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 640) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[92:95], v[158:161], a[64:67], v139, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4144) 
ds_read_b128 v[52:55], v4 offset:3072 // (op 866) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[92:95], v[162:165], a[68:71], v139, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4068) 
ds_read_b128 v[56:59], v4 offset:4096 // (op 1092) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[92:95], v[166:169], a[72:75], v139, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3992) 
ds_read_b128 v[60:63], v4 offset:5120 // (op 1318) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[92:95], v[170:173], a[76:79], v139, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3916) 
ds_read_b128 v[64:67], v4 offset:6144 // (op 1544) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[92:95], v[174:177], a[80:83], v139, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3840) 
buffer_load_dwordx4 v120, s[52:55], s38 offen offset: 0 lds // (op 30468) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[92:95], v[126:129], a[84:87], v139, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3764) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[92:95], v[178:181], a[88:91], v139, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3688) 
v_add_co_u32 v20, vcc, 256, v20 // (op 30654) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[92:95], v[130:133], a[92:95], v139, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3612) 
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 30654) most significant half
s_waitcnt lgkmcnt(7)
s_barrier  // (op 31838) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[96:99], v[76:79], a[96:99], v140, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7450) 
ds_read_b128 v[68:71], v4 offset:7168 // (op 1770) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[96:99], v[80:83], a[100:103], v140, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7374) 
s_lshl_b64 s[38:39], s[12:13], 5 // (op 30463) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[96:99], v[84:87], a[104:107], v140, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7298) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 30463) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[96:99], v[154:157], a[108:111], v140, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7222) 
v_lshrrev_b64 v[26:27], 3, s[64:65] // (op 30463) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:99], v[158:161], a[112:115], v140, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7146) 
v_mov_b64 v[112:113], v[114:115] // (op 30463) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:99], v[162:165], a[116:119], v140, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7070) 
s_add_u32 s19, 73728, s14 // (op 30463) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:99], v[166:169], a[120:123], v140, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6994) 
s_mov_b32 m0, s19 // (op 30463) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:99], v[170:173], a[124:127], v140, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6918) 
s_mov_b32 s38, 448 // (op 30463) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[174:177], a[128:131], v140, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6842) 
buffer_load_dwordx4 v112, s[48:51], s38 offen offset: 0 lds // (op 30463) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[126:129], a[132:135], v140, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6766) 
v_add_co_u32 v112, s[64:65], v112, v26 // (op 30463) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[178:181], a[136:139], v140, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6690) 
v_addc_co_u32 v113, s[64:65], 0, v113, s[64:65] // (op 30463) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[130:133], a[140:143], v140, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6614) 
s_add_u32 m0, m0, 2048 // (op 30463) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[76:79], a[144:147], v141, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10452) 
ds_read_b128 v[72:75], v4 offset:8192 // (op 1996) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[80:83], a[148:151], v141, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10376) 
ds_read_b128 v[76:79], v4 offset:9216 // (op 2222) Load local data 
buffer_load_dwordx4 v112, s[48:51], s38 offen offset: 0 lds // (op 30463) Load value direct to lds



v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[84:87], a[152:155], v141, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10300) 
ds_read_b128 v[80:83], v4 offset:10240 // (op 2448) Load local data 
v_add_co_u32 v112, s[64:65], v112, v26 // (op 30463) least significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[154:157], a[156:159], v141, v125 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10224) 
ds_read_b128 v[84:87], v4 offset:11264 // (op 2674) Load local data 


v_addc_co_u32 v113, s[64:65], 0, v113, s[64:65] // (op 30463) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[134:137], v[158:161], a[160:163], v141, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10148) 

s_add_u32 m0, m0, 2048 // (op 30463) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[134:137], v[162:165], a[164:167], v141, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10072) 

s_waitcnt vmcnt(32)
s_barrier  // (op 31782) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[134:137], v[166:169], a[168:171], v141, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9996) 
ds_read_b128 v[88:91], v0 offset:12288 // (op 60) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[134:137], v[170:173], a[172:175], v141, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9920) 
ds_read_b128 v[92:95], v0 offset:13312 // (op 3605) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[134:137], v[174:177], a[176:179], v141, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9844) 
ds_read_b128 v[96:99], v0 offset:14336 // (op 6607) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[134:137], v[126:129], a[180:183], v141, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9768) 
ds_read_b128 v[100:103], v0 offset:15360 // (op 9609) Load local data 

buffer_load_dwordx4 v112, s[48:51], s38 offen offset: 0 lds // (op 30463) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[134:137], v[178:181], a[184:187], v141, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9692) 
v_add_co_u32 v112, s[64:65], v112, v26 // (op 30463) least significant half

v_addc_co_u32 v113, s[64:65], 0, v113, s[64:65] // (op 30463) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[134:137], v[130:133], a[188:191], v141, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9616) 




s_add_u32 m0, m0, 2048 // (op 30463) 
buffer_load_dwordx4 v112, s[48:51], s38 offen offset: 0 lds // (op 30463) Load value direct to lds





v_add_co_u32 v18, vcc, 256, v18 // (op 30590) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 30590) most significant half
s_ashr_i64 s[38:39], s[36:37], 7 // (op 9579) 
s_mov_b32 s19, s38 // (op 9579) convert

s_and_b32 s38, s19, 4294967292 // (op 9579) 

s_cmp_lt_u32 s18, s38 // (op 9579) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopTop_KLoop_9579 // (op 9579) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopTop_KLoop_9579 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopBottom_KLoop_9579:
















s_barrier  // (op 31933) 
v_lshrrev_b32 v14, 2, v1 // (op 30506) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 30506) 
v_lshl_add_u32 v15, s16, 7, v14 // (op 30506) 

s_lshr_b32 s6, s0, 2 // (op 30506) 

v_and_b32 v14, -4, v1 // (op 30506) 
v_sub_u32 v16, v1, v14 // (op 30506) 

v_lshlrev_b32 v14, 5, v16 // (op 30506) 

v_lshl_add_u32 v16, s6, 9, v14 // (op 30506) 


v_mov_b32 v27, 0 // (op 30506) convert
v_mov_b32 v26, v16 // (op 30506) convert

v_mul_lo_u32 v17, v15, s13 // (op 30506) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 30506) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 30506) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 30506) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v26 // (op 30506) least significant half
v_addc_co_u32 v17, s[0:1], v17, v27, s[0:1] // (op 30506) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 30506) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 30506) 

s_mul_i32 s1, s12, s23 // (op 30508) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 30508) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 30508) least significant
s_mul_hi_u32 s7, s12, s22 // (op 30508) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30508) most significant: sum
s_add_u32 s1, s1, s7 // (op 30508) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30508) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30508) 

s_mov_b32 s48, s20 // (op 30508) reg expression
s_mov_b32 s49, s21 // (op 30508) reg expression
s_mov_b32 s50, s0 // (op 30508) reg expression
s_mov_b32 s51, 131072 // (op 30508) call()

v_and_b32 v14, -4, v1 // (op 30620) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 30620) 
v_sub_u32 v15, v1, v14 // (op 30620) 

s_lshr_b32 s6, s0, 2 // (op 30620) 

v_lshlrev_b32 v14, 5, v15 // (op 30620) 

v_lshl_add_u32 v15, s6, 9, v14 // (op 30620) 


v_mov_b32 v19, 0 // (op 30620) convert
v_mov_b32 v18, v15 // (op 30620) convert

v_lshrrev_b32 v14, 2, v1 // (op 30620) 
v_mov_b32 v15, 192 // (op 30620) 
v_mul_lo_u32 v20, s17, v15 // (op 30620) 

v_add_u32 v20, v20, v14 // (op 30620) 

v_mul_lo_u32 v15, v20, s25 // (op 30620) most significant: low of low * high
v_mul_hi_u32 v21, v20, s24 // (op 30620) most significant: high of low * low
v_mul_lo_u32 v14, v20, s24 // (op 30620) least significant: low of low * low
v_add_u32 v15, v15, v21 // (op 30620) most significant: sum


v_add_co_u32 v20, s[0:1], v18, v14 // (op 30620) least significant half
v_addc_co_u32 v21, s[0:1], v19, v15, s[0:1] // (op 30620) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 30620) 


v_lshrrev_b64 v[14:15], 3, v[20:21] // (op 30620) 

s_mul_i32 s1, s24, s9 // (op 30622) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 30622) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 30622) least significant
s_mul_hi_u32 s7, s24, s8 // (op 30622) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 30622) most significant: sum
s_add_u32 s1, s1, s7 // (op 30622) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 30622) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 30622) 

s_mov_b32 s52, s26 // (op 30622) reg expression
s_mov_b32 s53, s27 // (op 30622) reg expression
s_mov_b32 s54, s0 // (op 30622) reg expression
s_mov_b32 s55, 131072 // (op 30622) call()

v_and_b32 v18, -4, v1 // (op 30703) 
v_sub_u32 v19, v1, v18 // (op 30703) 

v_lshlrev_b32 v18, 5, v19 // (op 30703) 

v_lshrrev_b32 v19, 2, v1 // (op 30703) 
v_lshl_add_u32 v20, v19, 7, v18 // (op 30703) 


v_lshlrev_b32 v18, 2, v20 // (op 30703) 

v_lshrrev_b32 v19, 3, v18 // (op 30703) 

s_nop 0
v_readfirstlane_b32 s0, v19 // (op 30703) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 30762) 
v_sub_u32 v19, v1, v18 // (op 30762) 

v_lshlrev_b32 v18, 5, v19 // (op 30762) 

v_lshrrev_b32 v19, 2, v1 // (op 30762) 
v_lshl_add_u32 v20, v19, 7, v18 // (op 30762) 


v_lshlrev_b32 v18, 2, v20 // (op 30762) 

v_lshrrev_b32 v19, 3, v18 // (op 30762) 

s_nop 0
v_readfirstlane_b32 s1, v19 // (op 30762) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, 63, v1 // (op 31216) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 31216) 
v_lshrrev_b32 v19, 4, v18 // (op 31216) 
s_lshr_b32 s10, s6, 2 // (op 31216) 

v_and_b32 v20, 3, v19 // (op 31216) 

v_and_b32 v19, 15, v1 // (op 31216) 
v_lshl_add_u32 v21, v20, 4, v19 // (op 31216) 


v_and_b32 v19, 4294967232, v1 // (op 31216) 
v_add_u32 v20, v19, v21 // (op 31216) 


v_lshl_add_u32 v19, s16, 7, v20 // (op 31216) 

v_lshrrev_b32 v20, 6, v18 // (op 31216) 

v_lshlrev_b32 v18, 2, v20 // (op 31216) 

v_lshl_add_u32 v20, s10, 4, v18 // (op 31216) 


v_mul_lo_u32 v18, v19, s28 // (op 31216) 
v_add_u32 v18, v18, v20 // (op 31216) 


s_mul_i32 s7, s28, s23 // (op 31218) most significant: low of low * high
s_mul_i32 s10, s29, s22 // (op 31218) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 31218) least significant
s_mul_hi_u32 s11, s28, s22 // (op 31218) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 31218) most significant: sum
s_add_u32 s7, s7, s11 // (op 31218) most significant: sum


s_mov_b32 s56, s30 // (op 31218) reg expression
s_mov_b32 s57, s31 // (op 31218) reg expression
s_mov_b32 s58, s6 // (op 31218) reg expression
s_mov_b32 s59, 131072 // (op 31218) call()

v_and_b32 v19, 63, v1 // (op 31243) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 31243) 
v_lshrrev_b32 v20, 6, v19 // (op 31243) 
s_lshr_b32 s10, s6, 2 // (op 31243) 

v_lshlrev_b32 v21, 2, v20 // (op 31243) 

v_lshl_add_u32 v20, s10, 4, v21 // (op 31243) 


v_lshrrev_b32 v21, 4, v19 // (op 31243) 

v_and_b32 v19, 3, v21 // (op 31243) 

v_and_b32 v21, 15, v1 // (op 31243) 
v_lshl_add_u32 v22, v19, 4, v21 // (op 31243) 


v_mov_b32 v19, 192 // (op 31243) 
v_mul_lo_u32 v21, s17, v19 // (op 31243) 

v_add_u32 v21, v21, v22 // (op 31243) 

v_mul_lo_u32 v19, v21, s32 // (op 31243) 

v_add_u32 v21, v20, v19 // (op 31243) 


s_mul_i32 s7, s32, s9 // (op 31245) most significant: low of low * high
s_mul_i32 s10, s33, s8 // (op 31245) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 31245) least significant
s_mul_hi_u32 s11, s32, s8 // (op 31245) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 31245) most significant: sum
s_add_u32 s7, s7, s11 // (op 31245) most significant: sum


s_mov_b32 s60, s34 // (op 31245) reg expression
s_mov_b32 s61, s35 // (op 31245) reg expression
s_mov_b32 s62, s6 // (op 31245) reg expression
s_mov_b32 s63, 131072 // (op 31245) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 9477) 
s_and_b32 s10, s6, 4294967292 // (op 9477) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 9476) 
s_mov_b32 s11, s6 // (op 9476) convert

s_cmp_lt_u32 s10, s11 // (op 9476) 




















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopBottom_KLoopTail_9476 // (op 9476) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopBottom_KLoopTail_9476 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopTop_KLoopTail_9476:
s_barrier  // (op 31779) 
s_add_u32 s10, s10, 1 // (op 9478) 

s_lshl_b64 s[6:7], s[12:13], 5 // (op 30431) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 30431) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 30431) 


v_mov_b64 v[24:25], v[16:17] // (op 30431) 
v_mov_b64 v[26:27], v[24:25] // (op 30431) 
s_add_u32 s11, 12288, s0 // (op 30431) 
s_mov_b32 m0, s11 // (op 30431) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30431) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30431) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30431) most significant half

s_add_u32 m0, m0, 2048 // (op 30431) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30431) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30431) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30431) most significant half

s_add_u32 m0, m0, 2048 // (op 30431) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30431) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30431) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30431) most significant half

s_add_u32 m0, m0, 2048 // (op 30431) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 30431) Load value direct to lds




v_add_co_u32 v16, s[6:7], 64, v16 // (op 30496) least significant half
v_addc_co_u32 v17, s[6:7], 0, v17, s[6:7] // (op 30496) most significant half

s_barrier  // (op 31796) 

s_lshl_b64 s[6:7], s[24:25], 5 // (op 30437) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 30437) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 30437) 


v_mov_b64 v[24:25], v[14:15] // (op 30437) 
v_mov_b64 v[26:27], v[24:25] // (op 30437) 
s_mov_b32 m0, s1 // (op 30437) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30437) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30437) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30437) most significant half

s_add_u32 m0, m0, 2048 // (op 30437) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30437) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30437) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30437) most significant half

s_add_u32 m0, m0, 2048 // (op 30437) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30437) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30437) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30437) most significant half

s_add_u32 m0, m0, 2048 // (op 30437) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30437) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30437) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30437) most significant half

s_add_u32 m0, m0, 2048 // (op 30437) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30437) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 30437) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 30437) most significant half

s_add_u32 m0, m0, 2048 // (op 30437) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 30437) Load value direct to lds



v_add_co_u32 v14, s[6:7], 64, v14 // (op 30612) least significant half
v_addc_co_u32 v15, s[6:7], 0, v15, s[6:7] // (op 30612) most significant half

s_waitcnt vmcnt(6)
s_barrier  // (op 31786) 



ds_read_b128 v[22:25], v0 offset:12288 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 31802) 



ds_read_b128 v[26:29], v4 // (op 162) Load local data 


v_mov_b32 v19, v18 // (op 164) 
v_mov_b32 v20, v19 // (op 164) reg expression

buffer_load_dwordx4 v[30:33], v20, s[56:59], 0 offen // (op 164) Load value


v_add_u32 v18, 4, v18 // (op 31206) 


v_mov_b32 v19, v21 // (op 166) 
v_mov_b32 v20, v19 // (op 166) reg expression

buffer_load_dwordx4 v[34:37], v20, s[60:63], 0 offen // (op 166) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v38, v30, 0, 8 // (op 28298) 



v_bfe_u32 v39, v30, 8, 8 // (op 28298) 
v_bfe_u32 v40, v30, 16, 8 // (op 28298) 
v_bfe_u32 v41, v30, 24, 8 // (op 28298) 
v_permlane16_swap_b32 v38, v39 // (op 28298) 
ds_read_b128 v[42:45], v4 offset:1024 // (op 397) Load local data 
v_permlane16_swap_b32 v40, v41 // (op 28298) 



s_nop 1
v_permlane32_swap_b32 v38, v40 // (op 28298) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v39, v41 // (op 28298) 
ds_read_b128 v[46:49], v4 offset:2048 // (op 623) Load local data 




ds_read_b128 v[30:33], v4 offset:3072 // (op 849) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v34, 0, 8 // (op 28360) 



v_bfe_u32 v51, v34, 8, 8 // (op 28360) 
v_bfe_u32 v52, v34, 16, 8 // (op 28360) 
v_bfe_u32 v53, v34, 24, 8 // (op 28360) 
v_permlane16_swap_b32 v50, v51 // (op 28360) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 28360) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 28360) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 28360) 
ds_read_b128 v[54:57], v4 offset:4096 // (op 1075) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v38, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 167) 
s_lshl_b32 s11, s32, 6 // (op 1080) 
v_add_u32 v20, v19, s11 // (op 1080) 


buffer_load_dwordx4 v[34:37], v20, s[60:63], 0 offen // (op 1080) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[42:45], a[4:7], v38, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 400) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[46:49], a[8:11], v38, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 626) 
ds_read_b128 v[58:61], v4 offset:5120 // (op 1301) Load local data 



ds_read_b128 v[62:65], v4 offset:6144 // (op 1527) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:25], v[30:33], a[12:15], v38, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 852) 



ds_read_b128 v[66:69], v4 offset:7168 // (op 1753) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v70, v34, 0, 8 // (op 28374) 



v_bfe_u32 v71, v34, 8, 8 // (op 28374) 
v_bfe_u32 v72, v34, 16, 8 // (op 28374) 
v_bfe_u32 v73, v34, 24, 8 // (op 28374) 
v_permlane16_swap_b32 v70, v71 // (op 28374) 
s_nop 0
v_permlane16_swap_b32 v72, v73 // (op 28374) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v70, v72 // (op 28374) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v71, v73 // (op 28374) 
ds_read_b128 v[74:77], v4 offset:8192 // (op 1979) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[54:57], a[16:19], v38, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1078) 
s_lshl_b32 s11, s32, 7 // (op 1984) 
v_add_u32 v20, v19, s11 // (op 1984) 


buffer_load_dwordx4 v[34:37], v20, s[60:63], 0 offen // (op 1984) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[58:61], a[20:23], v38, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1304) 




v_add_u32 v21, 4, v21 // (op 31233) 
ds_read_b128 v[78:81], v4 offset:9216 // (op 2205) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[62:65], a[24:27], v38, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1530) 



ds_read_b128 v[82:85], v4 offset:10240 // (op 2431) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:25], v[66:69], a[28:31], v38, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1756) 



s_waitcnt vmcnt(0)
v_bfe_u32 v86, v34, 0, 8 // (op 28388) 
v_bfe_u32 v87, v34, 8, 8 // (op 28388) 
v_bfe_u32 v88, v34, 16, 8 // (op 28388) 
v_bfe_u32 v89, v34, 24, 8 // (op 28388) 
v_permlane16_swap_b32 v86, v87 // (op 28388) 
s_nop 0
v_permlane16_swap_b32 v88, v89 // (op 28388) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v86, v88 // (op 28388) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v87, v89 // (op 28388) 
ds_read_b128 v[90:93], v4 offset:11264 // (op 2657) Load local data 




s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[22:25], v[74:77], a[32:35], v38, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1982) 
ds_read_b128 v[34:37], v0 offset:13312 // (op 3289) Load local data 



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[22:25], v[78:81], a[36:39], v38, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2208) 
ds_read_b128 v[94:97], v0 offset:14336 // (op 6291) Load local data 



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[22:25], v[82:85], a[40:43], v38, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2434) 
ds_read_b128 v[98:101], v0 offset:15360 // (op 9293) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[22:25], v[90:93], a[44:47], v38, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2660) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[34:37], v[26:29], a[48:51], v39, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3293) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[34:37], v[46:49], a[56:59], v39, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3347) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[34:37], v[30:33], a[60:63], v39, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3374) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[34:37], v[42:45], a[52:55], v39, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3320) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:37], v[54:57], a[64:67], v39, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3401) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[34:37], v[62:65], a[72:75], v39, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3455) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:37], v[66:69], a[76:79], v39, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3482) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[58:61], a[68:71], v39, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3428) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[74:77], a[80:83], v39, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3509) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[34:37], v[82:85], a[88:91], v39, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3563) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[90:93], a[92:95], v39, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3590) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:37], v[78:81], a[84:87], v39, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3536) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[26:29], a[96:99], v40, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6295) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[94:97], v[46:49], a[104:107], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6349) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[30:33], a[108:111], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6376) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[94:97], v[42:45], a[100:103], v40, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6322) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[94:97], v[54:57], a[112:115], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6403) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[62:65], a[120:123], v40, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6457) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[94:97], v[66:69], a[124:127], v40, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6484) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[94:97], v[58:61], a[116:119], v40, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[94:97], v[74:77], a[128:131], v40, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6511) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[94:97], v[82:85], a[136:139], v40, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6565) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[94:97], v[90:93], a[140:143], v40, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6592) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[94:97], v[78:81], a[132:135], v40, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6538) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[98:101], v[26:29], a[144:147], v41, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9297) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[98:101], v[46:49], a[152:155], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9351) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[98:101], v[30:33], a[156:159], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9378) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[98:101], v[42:45], a[148:151], v41, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9324) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[98:101], v[54:57], a[160:163], v41, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9405) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:101], v[62:65], a[168:171], v41, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9459) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[98:101], v[66:69], a[172:175], v41, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9486) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[98:101], v[58:61], a[164:167], v41, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9432) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[98:101], v[74:77], a[176:179], v41, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9513) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[98:101], v[82:85], a[184:187], v41, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9567) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[98:101], v[90:93], a[188:191], v41, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9594) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[98:101], v[78:81], a[180:183], v41, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9540) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 9476) 
s_mov_b32 s11, s6 // (op 9476) convert

s_cmp_lt_u32 s10, s11 // (op 9476) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopTop_KLoopTail_9476 // (op 9476) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopTop_KLoopTail_9476 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ForLoopBottom_KLoopTail_9476:





















s_barrier  // (op 31919) 









v_cmp_eq_f32 vcc, 0.00000, v9 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ConditionalFalse_0_beta_114
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v11, v4 // (op 42) 
v_mul_f32 v15, v11, v5 // (op 42) 
v_mul_f32 v16, v11, v6 // (op 42) 
v_mul_f32 v17, v11, v7 // (op 42) 



v_mov_b32 v0, v13 // (op 48) reg expression

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 48) Store value


v_accvgpr_read v4, a4 // (op 13734) 
v_accvgpr_read v5, a5 // (op 13734) 
v_accvgpr_read v6, a6 // (op 13734) 
v_accvgpr_read v7, a7 // (op 13734) 
v_mul_f32 v14, v11, v4 // (op 13734) 
v_mul_f32 v15, v11, v5 // (op 13734) 
v_mul_f32 v16, v11, v6 // (op 13734) 
v_mul_f32 v17, v11, v7 // (op 13734) 



s_lshl_b32 s0, s4, 6 // (op 13736) 
v_add_u32 v0, v13, s0 // (op 13736) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13736) Store value


v_accvgpr_read v4, a8 // (op 13746) 
v_accvgpr_read v5, a9 // (op 13746) 
v_accvgpr_read v6, a10 // (op 13746) 
v_accvgpr_read v7, a11 // (op 13746) 
v_mul_f32 v14, v11, v4 // (op 13746) 
v_mul_f32 v15, v11, v5 // (op 13746) 
v_mul_f32 v16, v11, v6 // (op 13746) 
v_mul_f32 v17, v11, v7 // (op 13746) 



s_lshl_b32 s0, s4, 7 // (op 13748) 
v_add_u32 v0, v13, s0 // (op 13748) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13748) Store value


v_accvgpr_read v4, a12 // (op 13758) 
v_accvgpr_read v5, a13 // (op 13758) 
v_accvgpr_read v6, a14 // (op 13758) 
v_accvgpr_read v7, a15 // (op 13758) 
v_mul_f32 v14, v11, v4 // (op 13758) 
v_mul_f32 v15, v11, v5 // (op 13758) 
v_mul_f32 v16, v11, v6 // (op 13758) 
v_mul_f32 v17, v11, v7 // (op 13758) 



s_lshl_b32 s0, s4, 6 // (op 13760) 
s_mul_i32 s1, 3, s0 // (op 13760) 

v_add_u32 v0, v13, s1 // (op 13760) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13760) Store value


v_accvgpr_read v4, a16 // (op 13770) 
v_accvgpr_read v5, a17 // (op 13770) 
v_accvgpr_read v6, a18 // (op 13770) 
v_accvgpr_read v7, a19 // (op 13770) 
v_mul_f32 v14, v11, v4 // (op 13770) 
v_mul_f32 v15, v11, v5 // (op 13770) 
v_mul_f32 v16, v11, v6 // (op 13770) 
v_mul_f32 v17, v11, v7 // (op 13770) 



s_lshl_b32 s0, s4, 8 // (op 13772) 
v_add_u32 v0, v13, s0 // (op 13772) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13772) Store value


v_accvgpr_read v4, a20 // (op 13782) 
v_accvgpr_read v5, a21 // (op 13782) 
v_accvgpr_read v6, a22 // (op 13782) 
v_accvgpr_read v7, a23 // (op 13782) 
v_mul_f32 v14, v11, v4 // (op 13782) 
v_mul_f32 v15, v11, v5 // (op 13782) 
v_mul_f32 v16, v11, v6 // (op 13782) 
v_mul_f32 v17, v11, v7 // (op 13782) 



s_lshl_b32 s0, s4, 6 // (op 13784) 
s_mul_i32 s1, 5, s0 // (op 13784) 

v_add_u32 v0, v13, s1 // (op 13784) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13784) Store value


v_accvgpr_read v4, a24 // (op 13794) 
v_accvgpr_read v5, a25 // (op 13794) 
v_accvgpr_read v6, a26 // (op 13794) 
v_accvgpr_read v7, a27 // (op 13794) 
v_mul_f32 v14, v11, v4 // (op 13794) 
v_mul_f32 v15, v11, v5 // (op 13794) 
v_mul_f32 v16, v11, v6 // (op 13794) 
v_mul_f32 v17, v11, v7 // (op 13794) 



s_lshl_b32 s0, s4, 6 // (op 13796) 
s_mul_i32 s1, 6, s0 // (op 13796) 

v_add_u32 v0, v13, s1 // (op 13796) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13796) Store value


v_accvgpr_read v4, a28 // (op 13806) 
v_accvgpr_read v5, a29 // (op 13806) 
v_accvgpr_read v6, a30 // (op 13806) 
v_accvgpr_read v7, a31 // (op 13806) 
v_mul_f32 v14, v11, v4 // (op 13806) 
v_mul_f32 v15, v11, v5 // (op 13806) 
v_mul_f32 v16, v11, v6 // (op 13806) 
v_mul_f32 v17, v11, v7 // (op 13806) 



s_lshl_b32 s0, s4, 6 // (op 13808) 
s_mul_i32 s1, 7, s0 // (op 13808) 

v_add_u32 v0, v13, s1 // (op 13808) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13808) Store value


v_accvgpr_read v4, a32 // (op 13818) 
v_accvgpr_read v5, a33 // (op 13818) 
v_accvgpr_read v6, a34 // (op 13818) 
v_accvgpr_read v7, a35 // (op 13818) 
v_mul_f32 v14, v11, v4 // (op 13818) 
v_mul_f32 v15, v11, v5 // (op 13818) 
v_mul_f32 v16, v11, v6 // (op 13818) 
v_mul_f32 v17, v11, v7 // (op 13818) 



s_lshl_b32 s0, s4, 9 // (op 13820) 
v_add_u32 v0, v13, s0 // (op 13820) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13820) Store value


v_accvgpr_read v4, a36 // (op 13830) 
v_accvgpr_read v5, a37 // (op 13830) 
v_accvgpr_read v6, a38 // (op 13830) 
v_accvgpr_read v7, a39 // (op 13830) 
v_mul_f32 v14, v11, v4 // (op 13830) 
v_mul_f32 v15, v11, v5 // (op 13830) 
v_mul_f32 v16, v11, v6 // (op 13830) 
v_mul_f32 v17, v11, v7 // (op 13830) 



s_lshl_b32 s0, s4, 6 // (op 13832) 
s_mul_i32 s1, 9, s0 // (op 13832) 

v_add_u32 v0, v13, s1 // (op 13832) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13832) Store value


v_accvgpr_read v4, a40 // (op 13842) 
v_accvgpr_read v5, a41 // (op 13842) 
v_accvgpr_read v6, a42 // (op 13842) 
v_accvgpr_read v7, a43 // (op 13842) 
v_mul_f32 v14, v11, v4 // (op 13842) 
v_mul_f32 v15, v11, v5 // (op 13842) 
v_mul_f32 v16, v11, v6 // (op 13842) 
v_mul_f32 v17, v11, v7 // (op 13842) 



s_lshl_b32 s0, s4, 6 // (op 13844) 
s_mul_i32 s1, 10, s0 // (op 13844) 

v_add_u32 v0, v13, s1 // (op 13844) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13844) Store value


v_accvgpr_read v4, a44 // (op 13854) 
v_accvgpr_read v5, a45 // (op 13854) 
v_accvgpr_read v6, a46 // (op 13854) 
v_accvgpr_read v7, a47 // (op 13854) 
v_mul_f32 v14, v11, v4 // (op 13854) 
v_mul_f32 v15, v11, v5 // (op 13854) 
v_mul_f32 v16, v11, v6 // (op 13854) 
v_mul_f32 v17, v11, v7 // (op 13854) 



s_lshl_b32 s0, s4, 6 // (op 13856) 
s_mul_i32 s1, 11, s0 // (op 13856) 

v_add_u32 v0, v13, s1 // (op 13856) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13856) Store value


v_accvgpr_read v4, a48 // (op 13928) 
v_accvgpr_read v5, a49 // (op 13928) 
v_accvgpr_read v6, a50 // (op 13928) 
v_accvgpr_read v7, a51 // (op 13928) 
v_mul_f32 v14, v11, v4 // (op 13928) 
v_mul_f32 v15, v11, v5 // (op 13928) 
v_mul_f32 v16, v11, v6 // (op 13928) 
v_mul_f32 v17, v11, v7 // (op 13928) 



v_add_u32 v0, 64, v13 // (op 13931) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13931) Store value


v_accvgpr_read v4, a52 // (op 13957) 
v_accvgpr_read v5, a53 // (op 13957) 
v_accvgpr_read v6, a54 // (op 13957) 
v_accvgpr_read v7, a55 // (op 13957) 
v_mul_f32 v14, v11, v4 // (op 13957) 
v_mul_f32 v15, v11, v5 // (op 13957) 
v_mul_f32 v16, v11, v6 // (op 13957) 
v_mul_f32 v17, v11, v7 // (op 13957) 



s_lshl_b32 s0, s4, 6 // (op 13933) 
s_add_u32 s0, s0, 64 // (op 13933) 
v_add_u32 v0, v13, s0 // (op 13933) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13933) Store value


v_accvgpr_read v4, a56 // (op 13962) 
v_accvgpr_read v5, a57 // (op 13962) 
v_accvgpr_read v6, a58 // (op 13962) 
v_accvgpr_read v7, a59 // (op 13962) 
v_mul_f32 v14, v11, v4 // (op 13962) 
v_mul_f32 v15, v11, v5 // (op 13962) 
v_mul_f32 v16, v11, v6 // (op 13962) 
v_mul_f32 v17, v11, v7 // (op 13962) 



s_lshl_b32 s0, s4, 7 // (op 13935) 
s_add_u32 s0, s0, 64 // (op 13935) 
v_add_u32 v0, v13, s0 // (op 13935) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13935) Store value


v_accvgpr_read v4, a60 // (op 13967) 
v_accvgpr_read v5, a61 // (op 13967) 
v_accvgpr_read v6, a62 // (op 13967) 
v_accvgpr_read v7, a63 // (op 13967) 
v_mul_f32 v14, v11, v4 // (op 13967) 
v_mul_f32 v15, v11, v5 // (op 13967) 
v_mul_f32 v16, v11, v6 // (op 13967) 
v_mul_f32 v17, v11, v7 // (op 13967) 



s_lshl_b32 s0, s4, 6 // (op 13937) 
s_mul_i32 s1, 3, s0 // (op 13937) 
s_add_u32 s1, s1, 64 // (op 13937) 

v_add_u32 v0, v13, s1 // (op 13937) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13937) Store value


v_accvgpr_read v4, a64 // (op 13972) 
v_accvgpr_read v5, a65 // (op 13972) 
v_accvgpr_read v6, a66 // (op 13972) 
v_accvgpr_read v7, a67 // (op 13972) 
v_mul_f32 v14, v11, v4 // (op 13972) 
v_mul_f32 v15, v11, v5 // (op 13972) 
v_mul_f32 v16, v11, v6 // (op 13972) 
v_mul_f32 v17, v11, v7 // (op 13972) 



s_lshl_b32 s0, s4, 8 // (op 13939) 
s_add_u32 s0, s0, 64 // (op 13939) 
v_add_u32 v0, v13, s0 // (op 13939) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13939) Store value


v_accvgpr_read v4, a68 // (op 13977) 
v_accvgpr_read v5, a69 // (op 13977) 
v_accvgpr_read v6, a70 // (op 13977) 
v_accvgpr_read v7, a71 // (op 13977) 
v_mul_f32 v14, v11, v4 // (op 13977) 
v_mul_f32 v15, v11, v5 // (op 13977) 
v_mul_f32 v16, v11, v6 // (op 13977) 
v_mul_f32 v17, v11, v7 // (op 13977) 



s_lshl_b32 s0, s4, 6 // (op 13941) 
s_mul_i32 s1, 5, s0 // (op 13941) 
s_add_u32 s1, s1, 64 // (op 13941) 

v_add_u32 v0, v13, s1 // (op 13941) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13941) Store value


v_accvgpr_read v4, a72 // (op 13982) 
v_accvgpr_read v5, a73 // (op 13982) 
v_accvgpr_read v6, a74 // (op 13982) 
v_accvgpr_read v7, a75 // (op 13982) 
v_mul_f32 v14, v11, v4 // (op 13982) 
v_mul_f32 v15, v11, v5 // (op 13982) 
v_mul_f32 v16, v11, v6 // (op 13982) 
v_mul_f32 v17, v11, v7 // (op 13982) 



s_lshl_b32 s0, s4, 6 // (op 13943) 
s_mul_i32 s1, 6, s0 // (op 13943) 
s_add_u32 s1, s1, 64 // (op 13943) 

v_add_u32 v0, v13, s1 // (op 13943) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13943) Store value


v_accvgpr_read v4, a76 // (op 13987) 
v_accvgpr_read v5, a77 // (op 13987) 
v_accvgpr_read v6, a78 // (op 13987) 
v_accvgpr_read v7, a79 // (op 13987) 
v_mul_f32 v14, v11, v4 // (op 13987) 
v_mul_f32 v15, v11, v5 // (op 13987) 
v_mul_f32 v16, v11, v6 // (op 13987) 
v_mul_f32 v17, v11, v7 // (op 13987) 



s_lshl_b32 s0, s4, 6 // (op 13945) 
s_mul_i32 s1, 7, s0 // (op 13945) 
s_add_u32 s1, s1, 64 // (op 13945) 

v_add_u32 v0, v13, s1 // (op 13945) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13945) Store value


v_accvgpr_read v4, a80 // (op 13992) 
v_accvgpr_read v5, a81 // (op 13992) 
v_accvgpr_read v6, a82 // (op 13992) 
v_accvgpr_read v7, a83 // (op 13992) 
v_mul_f32 v14, v11, v4 // (op 13992) 
v_mul_f32 v15, v11, v5 // (op 13992) 
v_mul_f32 v16, v11, v6 // (op 13992) 
v_mul_f32 v17, v11, v7 // (op 13992) 



s_lshl_b32 s0, s4, 9 // (op 13947) 
s_add_u32 s0, s0, 64 // (op 13947) 
v_add_u32 v0, v13, s0 // (op 13947) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13947) Store value


v_accvgpr_read v4, a84 // (op 13997) 
v_accvgpr_read v5, a85 // (op 13997) 
v_accvgpr_read v6, a86 // (op 13997) 
v_accvgpr_read v7, a87 // (op 13997) 
v_mul_f32 v14, v11, v4 // (op 13997) 
v_mul_f32 v15, v11, v5 // (op 13997) 
v_mul_f32 v16, v11, v6 // (op 13997) 
v_mul_f32 v17, v11, v7 // (op 13997) 



s_lshl_b32 s0, s4, 6 // (op 13949) 
s_mul_i32 s1, 9, s0 // (op 13949) 
s_add_u32 s1, s1, 64 // (op 13949) 

v_add_u32 v0, v13, s1 // (op 13949) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13949) Store value


v_accvgpr_read v4, a88 // (op 14002) 
v_accvgpr_read v5, a89 // (op 14002) 
v_accvgpr_read v6, a90 // (op 14002) 
v_accvgpr_read v7, a91 // (op 14002) 
v_mul_f32 v14, v11, v4 // (op 14002) 
v_mul_f32 v15, v11, v5 // (op 14002) 
v_mul_f32 v16, v11, v6 // (op 14002) 
v_mul_f32 v17, v11, v7 // (op 14002) 



s_lshl_b32 s0, s4, 6 // (op 13951) 
s_mul_i32 s1, 10, s0 // (op 13951) 
s_add_u32 s1, s1, 64 // (op 13951) 

v_add_u32 v0, v13, s1 // (op 13951) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13951) Store value


v_accvgpr_read v4, a92 // (op 14007) 
v_accvgpr_read v5, a93 // (op 14007) 
v_accvgpr_read v6, a94 // (op 14007) 
v_accvgpr_read v7, a95 // (op 14007) 
v_mul_f32 v14, v11, v4 // (op 14007) 
v_mul_f32 v15, v11, v5 // (op 14007) 
v_mul_f32 v16, v11, v6 // (op 14007) 
v_mul_f32 v17, v11, v7 // (op 14007) 



s_lshl_b32 s0, s4, 6 // (op 13953) 
s_mul_i32 s1, 11, s0 // (op 13953) 
s_add_u32 s1, s1, 64 // (op 13953) 

v_add_u32 v0, v13, s1 // (op 13953) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 13953) Store value


v_accvgpr_read v4, a96 // (op 14124) 
v_accvgpr_read v5, a97 // (op 14124) 
v_accvgpr_read v6, a98 // (op 14124) 
v_accvgpr_read v7, a99 // (op 14124) 
v_mul_f32 v14, v11, v4 // (op 14124) 
v_mul_f32 v15, v11, v5 // (op 14124) 
v_mul_f32 v16, v11, v6 // (op 14124) 
v_mul_f32 v17, v11, v7 // (op 14124) 



v_add_u32 v0, 128, v13 // (op 14127) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14127) Store value


v_accvgpr_read v4, a100 // (op 14153) 
v_accvgpr_read v5, a101 // (op 14153) 
v_accvgpr_read v6, a102 // (op 14153) 
v_accvgpr_read v7, a103 // (op 14153) 
v_mul_f32 v14, v11, v4 // (op 14153) 
v_mul_f32 v15, v11, v5 // (op 14153) 
v_mul_f32 v16, v11, v6 // (op 14153) 
v_mul_f32 v17, v11, v7 // (op 14153) 



s_lshl_b32 s0, s4, 6 // (op 14129) 
s_add_u32 s0, s0, 128 // (op 14129) 
v_add_u32 v0, v13, s0 // (op 14129) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14129) Store value


v_accvgpr_read v4, a104 // (op 14158) 
v_accvgpr_read v5, a105 // (op 14158) 
v_accvgpr_read v6, a106 // (op 14158) 
v_accvgpr_read v7, a107 // (op 14158) 
v_mul_f32 v14, v11, v4 // (op 14158) 
v_mul_f32 v15, v11, v5 // (op 14158) 
v_mul_f32 v16, v11, v6 // (op 14158) 
v_mul_f32 v17, v11, v7 // (op 14158) 



s_lshl_b32 s0, s4, 7 // (op 14131) 
s_add_u32 s0, s0, 128 // (op 14131) 
v_add_u32 v0, v13, s0 // (op 14131) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14131) Store value


v_accvgpr_read v4, a108 // (op 14163) 
v_accvgpr_read v5, a109 // (op 14163) 
v_accvgpr_read v6, a110 // (op 14163) 
v_accvgpr_read v7, a111 // (op 14163) 
v_mul_f32 v14, v11, v4 // (op 14163) 
v_mul_f32 v15, v11, v5 // (op 14163) 
v_mul_f32 v16, v11, v6 // (op 14163) 
v_mul_f32 v17, v11, v7 // (op 14163) 



s_lshl_b32 s0, s4, 6 // (op 14133) 
s_mul_i32 s1, 3, s0 // (op 14133) 
s_add_u32 s1, s1, 128 // (op 14133) 

v_add_u32 v0, v13, s1 // (op 14133) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14133) Store value


v_accvgpr_read v4, a112 // (op 14168) 
v_accvgpr_read v5, a113 // (op 14168) 
v_accvgpr_read v6, a114 // (op 14168) 
v_accvgpr_read v7, a115 // (op 14168) 
v_mul_f32 v14, v11, v4 // (op 14168) 
v_mul_f32 v15, v11, v5 // (op 14168) 
v_mul_f32 v16, v11, v6 // (op 14168) 
v_mul_f32 v17, v11, v7 // (op 14168) 



s_lshl_b32 s0, s4, 8 // (op 14135) 
s_add_u32 s0, s0, 128 // (op 14135) 
v_add_u32 v0, v13, s0 // (op 14135) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14135) Store value


v_accvgpr_read v4, a116 // (op 14173) 
v_accvgpr_read v5, a117 // (op 14173) 
v_accvgpr_read v6, a118 // (op 14173) 
v_accvgpr_read v7, a119 // (op 14173) 
v_mul_f32 v14, v11, v4 // (op 14173) 
v_mul_f32 v15, v11, v5 // (op 14173) 
v_mul_f32 v16, v11, v6 // (op 14173) 
v_mul_f32 v17, v11, v7 // (op 14173) 



s_lshl_b32 s0, s4, 6 // (op 14137) 
s_mul_i32 s1, 5, s0 // (op 14137) 
s_add_u32 s1, s1, 128 // (op 14137) 

v_add_u32 v0, v13, s1 // (op 14137) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14137) Store value


v_accvgpr_read v4, a120 // (op 14178) 
v_accvgpr_read v5, a121 // (op 14178) 
v_accvgpr_read v6, a122 // (op 14178) 
v_accvgpr_read v7, a123 // (op 14178) 
v_mul_f32 v14, v11, v4 // (op 14178) 
v_mul_f32 v15, v11, v5 // (op 14178) 
v_mul_f32 v16, v11, v6 // (op 14178) 
v_mul_f32 v17, v11, v7 // (op 14178) 



s_lshl_b32 s0, s4, 6 // (op 14139) 
s_mul_i32 s1, 6, s0 // (op 14139) 
s_add_u32 s1, s1, 128 // (op 14139) 

v_add_u32 v0, v13, s1 // (op 14139) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14139) Store value


v_accvgpr_read v4, a124 // (op 14183) 
v_accvgpr_read v5, a125 // (op 14183) 
v_accvgpr_read v6, a126 // (op 14183) 
v_accvgpr_read v7, a127 // (op 14183) 
v_mul_f32 v14, v11, v4 // (op 14183) 
v_mul_f32 v15, v11, v5 // (op 14183) 
v_mul_f32 v16, v11, v6 // (op 14183) 
v_mul_f32 v17, v11, v7 // (op 14183) 



s_lshl_b32 s0, s4, 6 // (op 14141) 
s_mul_i32 s1, 7, s0 // (op 14141) 
s_add_u32 s1, s1, 128 // (op 14141) 

v_add_u32 v0, v13, s1 // (op 14141) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14141) Store value


v_accvgpr_read v4, a128 // (op 14188) 
v_accvgpr_read v5, a129 // (op 14188) 
v_accvgpr_read v6, a130 // (op 14188) 
v_accvgpr_read v7, a131 // (op 14188) 
v_mul_f32 v14, v11, v4 // (op 14188) 
v_mul_f32 v15, v11, v5 // (op 14188) 
v_mul_f32 v16, v11, v6 // (op 14188) 
v_mul_f32 v17, v11, v7 // (op 14188) 



s_lshl_b32 s0, s4, 9 // (op 14143) 
s_add_u32 s0, s0, 128 // (op 14143) 
v_add_u32 v0, v13, s0 // (op 14143) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14143) Store value


v_accvgpr_read v4, a132 // (op 14193) 
v_accvgpr_read v5, a133 // (op 14193) 
v_accvgpr_read v6, a134 // (op 14193) 
v_accvgpr_read v7, a135 // (op 14193) 
v_mul_f32 v14, v11, v4 // (op 14193) 
v_mul_f32 v15, v11, v5 // (op 14193) 
v_mul_f32 v16, v11, v6 // (op 14193) 
v_mul_f32 v17, v11, v7 // (op 14193) 



s_lshl_b32 s0, s4, 6 // (op 14145) 
s_mul_i32 s1, 9, s0 // (op 14145) 
s_add_u32 s1, s1, 128 // (op 14145) 

v_add_u32 v0, v13, s1 // (op 14145) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14145) Store value


v_accvgpr_read v4, a136 // (op 14198) 
v_accvgpr_read v5, a137 // (op 14198) 
v_accvgpr_read v6, a138 // (op 14198) 
v_accvgpr_read v7, a139 // (op 14198) 
v_mul_f32 v14, v11, v4 // (op 14198) 
v_mul_f32 v15, v11, v5 // (op 14198) 
v_mul_f32 v16, v11, v6 // (op 14198) 
v_mul_f32 v17, v11, v7 // (op 14198) 



s_lshl_b32 s0, s4, 6 // (op 14147) 
s_mul_i32 s1, 10, s0 // (op 14147) 
s_add_u32 s1, s1, 128 // (op 14147) 

v_add_u32 v0, v13, s1 // (op 14147) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14147) Store value


v_accvgpr_read v4, a140 // (op 14203) 
v_accvgpr_read v5, a141 // (op 14203) 
v_accvgpr_read v6, a142 // (op 14203) 
v_accvgpr_read v7, a143 // (op 14203) 
v_mul_f32 v14, v11, v4 // (op 14203) 
v_mul_f32 v15, v11, v5 // (op 14203) 
v_mul_f32 v16, v11, v6 // (op 14203) 
v_mul_f32 v17, v11, v7 // (op 14203) 



s_lshl_b32 s0, s4, 6 // (op 14149) 
s_mul_i32 s1, 11, s0 // (op 14149) 
s_add_u32 s1, s1, 128 // (op 14149) 

v_add_u32 v0, v13, s1 // (op 14149) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14149) Store value


v_accvgpr_read v4, a144 // (op 14320) 
v_accvgpr_read v5, a145 // (op 14320) 
v_accvgpr_read v6, a146 // (op 14320) 
v_accvgpr_read v7, a147 // (op 14320) 
v_mul_f32 v14, v11, v4 // (op 14320) 
v_mul_f32 v15, v11, v5 // (op 14320) 
v_mul_f32 v16, v11, v6 // (op 14320) 
v_mul_f32 v17, v11, v7 // (op 14320) 



v_add_u32 v0, 192, v13 // (op 14323) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14323) Store value


v_accvgpr_read v4, a148 // (op 14349) 
v_accvgpr_read v5, a149 // (op 14349) 
v_accvgpr_read v6, a150 // (op 14349) 
v_accvgpr_read v7, a151 // (op 14349) 
v_mul_f32 v14, v11, v4 // (op 14349) 
v_mul_f32 v15, v11, v5 // (op 14349) 
v_mul_f32 v16, v11, v6 // (op 14349) 
v_mul_f32 v17, v11, v7 // (op 14349) 



s_lshl_b32 s0, s4, 6 // (op 14325) 
s_add_u32 s0, s0, 192 // (op 14325) 
v_add_u32 v0, v13, s0 // (op 14325) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14325) Store value


v_accvgpr_read v4, a152 // (op 14354) 
v_accvgpr_read v5, a153 // (op 14354) 
v_accvgpr_read v6, a154 // (op 14354) 
v_accvgpr_read v7, a155 // (op 14354) 
v_mul_f32 v14, v11, v4 // (op 14354) 
v_mul_f32 v15, v11, v5 // (op 14354) 
v_mul_f32 v16, v11, v6 // (op 14354) 
v_mul_f32 v17, v11, v7 // (op 14354) 



s_lshl_b32 s0, s4, 7 // (op 14327) 
s_add_u32 s0, s0, 192 // (op 14327) 
v_add_u32 v0, v13, s0 // (op 14327) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14327) Store value


v_accvgpr_read v4, a156 // (op 14359) 
v_accvgpr_read v5, a157 // (op 14359) 
v_accvgpr_read v6, a158 // (op 14359) 
v_accvgpr_read v7, a159 // (op 14359) 
v_mul_f32 v14, v11, v4 // (op 14359) 
v_mul_f32 v15, v11, v5 // (op 14359) 
v_mul_f32 v16, v11, v6 // (op 14359) 
v_mul_f32 v17, v11, v7 // (op 14359) 



s_lshl_b32 s0, s4, 6 // (op 14329) 
s_mul_i32 s1, 3, s0 // (op 14329) 
s_add_u32 s1, s1, 192 // (op 14329) 

v_add_u32 v0, v13, s1 // (op 14329) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14329) Store value


v_accvgpr_read v4, a160 // (op 14364) 
v_accvgpr_read v5, a161 // (op 14364) 
v_accvgpr_read v6, a162 // (op 14364) 
v_accvgpr_read v7, a163 // (op 14364) 
v_mul_f32 v14, v11, v4 // (op 14364) 
v_mul_f32 v15, v11, v5 // (op 14364) 
v_mul_f32 v16, v11, v6 // (op 14364) 
v_mul_f32 v17, v11, v7 // (op 14364) 



s_lshl_b32 s0, s4, 8 // (op 14331) 
s_add_u32 s0, s0, 192 // (op 14331) 
v_add_u32 v0, v13, s0 // (op 14331) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14331) Store value


v_accvgpr_read v4, a164 // (op 14369) 
v_accvgpr_read v5, a165 // (op 14369) 
v_accvgpr_read v6, a166 // (op 14369) 
v_accvgpr_read v7, a167 // (op 14369) 
v_mul_f32 v14, v11, v4 // (op 14369) 
v_mul_f32 v15, v11, v5 // (op 14369) 
v_mul_f32 v16, v11, v6 // (op 14369) 
v_mul_f32 v17, v11, v7 // (op 14369) 



s_lshl_b32 s0, s4, 6 // (op 14333) 
s_mul_i32 s1, 5, s0 // (op 14333) 
s_add_u32 s1, s1, 192 // (op 14333) 

v_add_u32 v0, v13, s1 // (op 14333) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14333) Store value


v_accvgpr_read v4, a168 // (op 14374) 
v_accvgpr_read v5, a169 // (op 14374) 
v_accvgpr_read v6, a170 // (op 14374) 
v_accvgpr_read v7, a171 // (op 14374) 
v_mul_f32 v14, v11, v4 // (op 14374) 
v_mul_f32 v15, v11, v5 // (op 14374) 
v_mul_f32 v16, v11, v6 // (op 14374) 
v_mul_f32 v17, v11, v7 // (op 14374) 



s_lshl_b32 s0, s4, 6 // (op 14335) 
s_mul_i32 s1, 6, s0 // (op 14335) 
s_add_u32 s1, s1, 192 // (op 14335) 

v_add_u32 v0, v13, s1 // (op 14335) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14335) Store value


v_accvgpr_read v4, a172 // (op 14379) 
v_accvgpr_read v5, a173 // (op 14379) 
v_accvgpr_read v6, a174 // (op 14379) 
v_accvgpr_read v7, a175 // (op 14379) 
v_mul_f32 v14, v11, v4 // (op 14379) 
v_mul_f32 v15, v11, v5 // (op 14379) 
v_mul_f32 v16, v11, v6 // (op 14379) 
v_mul_f32 v17, v11, v7 // (op 14379) 



s_lshl_b32 s0, s4, 6 // (op 14337) 
s_mul_i32 s1, 7, s0 // (op 14337) 
s_add_u32 s1, s1, 192 // (op 14337) 

v_add_u32 v0, v13, s1 // (op 14337) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14337) Store value


v_accvgpr_read v4, a176 // (op 14384) 
v_accvgpr_read v5, a177 // (op 14384) 
v_accvgpr_read v6, a178 // (op 14384) 
v_accvgpr_read v7, a179 // (op 14384) 
v_mul_f32 v14, v11, v4 // (op 14384) 
v_mul_f32 v15, v11, v5 // (op 14384) 
v_mul_f32 v16, v11, v6 // (op 14384) 
v_mul_f32 v17, v11, v7 // (op 14384) 



s_lshl_b32 s0, s4, 9 // (op 14339) 
s_add_u32 s0, s0, 192 // (op 14339) 
v_add_u32 v0, v13, s0 // (op 14339) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14339) Store value


v_accvgpr_read v4, a180 // (op 14389) 
v_accvgpr_read v5, a181 // (op 14389) 
v_accvgpr_read v6, a182 // (op 14389) 
v_accvgpr_read v7, a183 // (op 14389) 
v_mul_f32 v14, v11, v4 // (op 14389) 
v_mul_f32 v15, v11, v5 // (op 14389) 
v_mul_f32 v16, v11, v6 // (op 14389) 
v_mul_f32 v17, v11, v7 // (op 14389) 



s_lshl_b32 s0, s4, 6 // (op 14341) 
s_mul_i32 s1, 9, s0 // (op 14341) 
s_add_u32 s1, s1, 192 // (op 14341) 

v_add_u32 v0, v13, s1 // (op 14341) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14341) Store value


v_accvgpr_read v4, a184 // (op 14394) 
v_accvgpr_read v5, a185 // (op 14394) 
v_accvgpr_read v6, a186 // (op 14394) 
v_accvgpr_read v7, a187 // (op 14394) 
v_mul_f32 v14, v11, v4 // (op 14394) 
v_mul_f32 v15, v11, v5 // (op 14394) 
v_mul_f32 v16, v11, v6 // (op 14394) 
v_mul_f32 v17, v11, v7 // (op 14394) 



s_lshl_b32 s0, s4, 6 // (op 14343) 
s_mul_i32 s1, 10, s0 // (op 14343) 
s_add_u32 s1, s1, 192 // (op 14343) 

v_add_u32 v0, v13, s1 // (op 14343) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14343) Store value


v_accvgpr_read v4, a188 // (op 14399) 
v_accvgpr_read v5, a189 // (op 14399) 
v_accvgpr_read v6, a190 // (op 14399) 
v_accvgpr_read v7, a191 // (op 14399) 
v_mul_f32 v14, v11, v4 // (op 14399) 
v_mul_f32 v15, v11, v5 // (op 14399) 
v_mul_f32 v16, v11, v6 // (op 14399) 
v_mul_f32 v17, v11, v7 // (op 14399) 



s_lshl_b32 s0, s4, 6 // (op 14345) 
s_mul_i32 s1, 11, s0 // (op 14345) 
s_add_u32 s1, s1, 192 // (op 14345) 

v_add_u32 v0, v13, s1 // (op 14345) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 14345) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ConditionalBottom_0_beta_114
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ConditionalFalse_0_beta_114:


v_add_u32 v0, 192, v12 // (op 15407) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15407) Load value

v_accvgpr_read v14, a144 // (op 15379) 
v_accvgpr_read v15, a145 // (op 15379) 
v_accvgpr_read v16, a146 // (op 15379) 
v_accvgpr_read v17, a147 // (op 15379) 
v_mul_f32 v18, v11, v14 // (op 15379) 
v_mul_f32 v19, v11, v15 // (op 15379) 
v_mul_f32 v20, v11, v16 // (op 15379) 
v_mul_f32 v21, v11, v17 // (op 15379) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15406) 
v_mul_f32 v15, v9, v5 // (op 15406) 
v_mul_f32 v16, v9, v6 // (op 15406) 
v_mul_f32 v17, v9, v7 // (op 15406) 

v_add_f32 v4, v14, v18 // (op 15380) 
v_add_f32 v5, v15, v19 // (op 15380) 
v_add_f32 v6, v16, v20 // (op 15380) 
v_add_f32 v7, v17, v21 // (op 15380) 




v_add_u32 v0, 192, v13 // (op 15382) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15382) Store value




s_lshl_b32 s0, s2, 6 // (op 15411) 
s_add_u32 s0, s0, 192 // (op 15411) 
v_add_u32 v0, v12, s0 // (op 15411) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15411) Load value

v_accvgpr_read v14, a148 // (op 15453) 
v_accvgpr_read v15, a149 // (op 15453) 
v_accvgpr_read v16, a150 // (op 15453) 
v_accvgpr_read v17, a151 // (op 15453) 
v_mul_f32 v18, v11, v14 // (op 15453) 
v_mul_f32 v19, v11, v15 // (op 15453) 
v_mul_f32 v20, v11, v16 // (op 15453) 
v_mul_f32 v21, v11, v17 // (op 15453) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15409) 
v_mul_f32 v15, v9, v5 // (op 15409) 
v_mul_f32 v16, v9, v6 // (op 15409) 
v_mul_f32 v17, v9, v7 // (op 15409) 

v_add_f32 v4, v14, v18 // (op 15410) 
v_add_f32 v5, v15, v19 // (op 15410) 
v_add_f32 v6, v16, v20 // (op 15410) 
v_add_f32 v7, v17, v21 // (op 15410) 




s_lshl_b32 s0, s4, 6 // (op 15384) 
s_add_u32 s0, s0, 192 // (op 15384) 
v_add_u32 v0, v13, s0 // (op 15384) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15384) Store value




s_lshl_b32 s0, s2, 7 // (op 15415) 
s_add_u32 s0, s0, 192 // (op 15415) 
v_add_u32 v0, v12, s0 // (op 15415) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15415) Load value

v_accvgpr_read v14, a152 // (op 15455) 
v_accvgpr_read v15, a153 // (op 15455) 
v_accvgpr_read v16, a154 // (op 15455) 
v_accvgpr_read v17, a155 // (op 15455) 
v_mul_f32 v18, v11, v14 // (op 15455) 
v_mul_f32 v19, v11, v15 // (op 15455) 
v_mul_f32 v20, v11, v16 // (op 15455) 
v_mul_f32 v21, v11, v17 // (op 15455) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15413) 
v_mul_f32 v15, v9, v5 // (op 15413) 
v_mul_f32 v16, v9, v6 // (op 15413) 
v_mul_f32 v17, v9, v7 // (op 15413) 

v_add_f32 v4, v14, v18 // (op 15414) 
v_add_f32 v5, v15, v19 // (op 15414) 
v_add_f32 v6, v16, v20 // (op 15414) 
v_add_f32 v7, v17, v21 // (op 15414) 




s_lshl_b32 s0, s4, 7 // (op 15386) 
s_add_u32 s0, s0, 192 // (op 15386) 
v_add_u32 v0, v13, s0 // (op 15386) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15386) Store value




s_lshl_b32 s0, s2, 6 // (op 15419) 
s_mul_i32 s1, 3, s0 // (op 15419) 
s_add_u32 s1, s1, 192 // (op 15419) 

v_add_u32 v0, v12, s1 // (op 15419) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15419) Load value

v_accvgpr_read v14, a156 // (op 15457) 
v_accvgpr_read v15, a157 // (op 15457) 
v_accvgpr_read v16, a158 // (op 15457) 
v_accvgpr_read v17, a159 // (op 15457) 
v_mul_f32 v18, v11, v14 // (op 15457) 
v_mul_f32 v19, v11, v15 // (op 15457) 
v_mul_f32 v20, v11, v16 // (op 15457) 
v_mul_f32 v21, v11, v17 // (op 15457) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15417) 
v_mul_f32 v15, v9, v5 // (op 15417) 
v_mul_f32 v16, v9, v6 // (op 15417) 
v_mul_f32 v17, v9, v7 // (op 15417) 

v_add_f32 v4, v14, v18 // (op 15418) 
v_add_f32 v5, v15, v19 // (op 15418) 
v_add_f32 v6, v16, v20 // (op 15418) 
v_add_f32 v7, v17, v21 // (op 15418) 




s_lshl_b32 s0, s4, 6 // (op 15388) 
s_mul_i32 s1, 3, s0 // (op 15388) 
s_add_u32 s1, s1, 192 // (op 15388) 

v_add_u32 v0, v13, s1 // (op 15388) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15388) Store value




s_lshl_b32 s0, s2, 8 // (op 15423) 
s_add_u32 s0, s0, 192 // (op 15423) 
v_add_u32 v0, v12, s0 // (op 15423) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15423) Load value

v_accvgpr_read v14, a160 // (op 15459) 
v_accvgpr_read v15, a161 // (op 15459) 
v_accvgpr_read v16, a162 // (op 15459) 
v_accvgpr_read v17, a163 // (op 15459) 
v_mul_f32 v18, v11, v14 // (op 15459) 
v_mul_f32 v19, v11, v15 // (op 15459) 
v_mul_f32 v20, v11, v16 // (op 15459) 
v_mul_f32 v21, v11, v17 // (op 15459) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15421) 
v_mul_f32 v15, v9, v5 // (op 15421) 
v_mul_f32 v16, v9, v6 // (op 15421) 
v_mul_f32 v17, v9, v7 // (op 15421) 

v_add_f32 v4, v14, v18 // (op 15422) 
v_add_f32 v5, v15, v19 // (op 15422) 
v_add_f32 v6, v16, v20 // (op 15422) 
v_add_f32 v7, v17, v21 // (op 15422) 




s_lshl_b32 s0, s4, 8 // (op 15390) 
s_add_u32 s0, s0, 192 // (op 15390) 
v_add_u32 v0, v13, s0 // (op 15390) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15390) Store value




s_lshl_b32 s0, s2, 6 // (op 15427) 
s_mul_i32 s1, 5, s0 // (op 15427) 
s_add_u32 s1, s1, 192 // (op 15427) 

v_add_u32 v0, v12, s1 // (op 15427) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15427) Load value

v_accvgpr_read v14, a164 // (op 15461) 
v_accvgpr_read v15, a165 // (op 15461) 
v_accvgpr_read v16, a166 // (op 15461) 
v_accvgpr_read v17, a167 // (op 15461) 
v_mul_f32 v18, v11, v14 // (op 15461) 
v_mul_f32 v19, v11, v15 // (op 15461) 
v_mul_f32 v20, v11, v16 // (op 15461) 
v_mul_f32 v21, v11, v17 // (op 15461) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15425) 
v_mul_f32 v15, v9, v5 // (op 15425) 
v_mul_f32 v16, v9, v6 // (op 15425) 
v_mul_f32 v17, v9, v7 // (op 15425) 

v_add_f32 v4, v14, v18 // (op 15426) 
v_add_f32 v5, v15, v19 // (op 15426) 
v_add_f32 v6, v16, v20 // (op 15426) 
v_add_f32 v7, v17, v21 // (op 15426) 




s_lshl_b32 s0, s4, 6 // (op 15392) 
s_mul_i32 s1, 5, s0 // (op 15392) 
s_add_u32 s1, s1, 192 // (op 15392) 

v_add_u32 v0, v13, s1 // (op 15392) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15392) Store value




s_lshl_b32 s0, s2, 6 // (op 15431) 
s_mul_i32 s1, 6, s0 // (op 15431) 
s_add_u32 s1, s1, 192 // (op 15431) 

v_add_u32 v0, v12, s1 // (op 15431) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15431) Load value

v_accvgpr_read v14, a168 // (op 15463) 
v_accvgpr_read v15, a169 // (op 15463) 
v_accvgpr_read v16, a170 // (op 15463) 
v_accvgpr_read v17, a171 // (op 15463) 
v_mul_f32 v18, v11, v14 // (op 15463) 
v_mul_f32 v19, v11, v15 // (op 15463) 
v_mul_f32 v20, v11, v16 // (op 15463) 
v_mul_f32 v21, v11, v17 // (op 15463) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15429) 
v_mul_f32 v15, v9, v5 // (op 15429) 
v_mul_f32 v16, v9, v6 // (op 15429) 
v_mul_f32 v17, v9, v7 // (op 15429) 

v_add_f32 v4, v14, v18 // (op 15430) 
v_add_f32 v5, v15, v19 // (op 15430) 
v_add_f32 v6, v16, v20 // (op 15430) 
v_add_f32 v7, v17, v21 // (op 15430) 




s_lshl_b32 s0, s4, 6 // (op 15394) 
s_mul_i32 s1, 6, s0 // (op 15394) 
s_add_u32 s1, s1, 192 // (op 15394) 

v_add_u32 v0, v13, s1 // (op 15394) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15394) Store value




s_lshl_b32 s0, s2, 6 // (op 15435) 
s_mul_i32 s1, 7, s0 // (op 15435) 
s_add_u32 s1, s1, 192 // (op 15435) 

v_add_u32 v0, v12, s1 // (op 15435) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15435) Load value

v_accvgpr_read v14, a172 // (op 15465) 
v_accvgpr_read v15, a173 // (op 15465) 
v_accvgpr_read v16, a174 // (op 15465) 
v_accvgpr_read v17, a175 // (op 15465) 
v_mul_f32 v18, v11, v14 // (op 15465) 
v_mul_f32 v19, v11, v15 // (op 15465) 
v_mul_f32 v20, v11, v16 // (op 15465) 
v_mul_f32 v21, v11, v17 // (op 15465) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15433) 
v_mul_f32 v15, v9, v5 // (op 15433) 
v_mul_f32 v16, v9, v6 // (op 15433) 
v_mul_f32 v17, v9, v7 // (op 15433) 

v_add_f32 v4, v14, v18 // (op 15434) 
v_add_f32 v5, v15, v19 // (op 15434) 
v_add_f32 v6, v16, v20 // (op 15434) 
v_add_f32 v7, v17, v21 // (op 15434) 




s_lshl_b32 s0, s4, 6 // (op 15396) 
s_mul_i32 s1, 7, s0 // (op 15396) 
s_add_u32 s1, s1, 192 // (op 15396) 

v_add_u32 v0, v13, s1 // (op 15396) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15396) Store value




s_lshl_b32 s0, s2, 9 // (op 15439) 
s_add_u32 s0, s0, 192 // (op 15439) 
v_add_u32 v0, v12, s0 // (op 15439) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15439) Load value

v_accvgpr_read v14, a176 // (op 15467) 
v_accvgpr_read v15, a177 // (op 15467) 
v_accvgpr_read v16, a178 // (op 15467) 
v_accvgpr_read v17, a179 // (op 15467) 
v_mul_f32 v18, v11, v14 // (op 15467) 
v_mul_f32 v19, v11, v15 // (op 15467) 
v_mul_f32 v20, v11, v16 // (op 15467) 
v_mul_f32 v21, v11, v17 // (op 15467) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15437) 
v_mul_f32 v15, v9, v5 // (op 15437) 
v_mul_f32 v16, v9, v6 // (op 15437) 
v_mul_f32 v17, v9, v7 // (op 15437) 

v_add_f32 v4, v14, v18 // (op 15438) 
v_add_f32 v5, v15, v19 // (op 15438) 
v_add_f32 v6, v16, v20 // (op 15438) 
v_add_f32 v7, v17, v21 // (op 15438) 




s_lshl_b32 s0, s4, 9 // (op 15398) 
s_add_u32 s0, s0, 192 // (op 15398) 
v_add_u32 v0, v13, s0 // (op 15398) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15398) Store value




s_lshl_b32 s0, s2, 6 // (op 15443) 
s_mul_i32 s1, 9, s0 // (op 15443) 
s_add_u32 s1, s1, 192 // (op 15443) 

v_add_u32 v0, v12, s1 // (op 15443) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15443) Load value

v_accvgpr_read v14, a180 // (op 15469) 
v_accvgpr_read v15, a181 // (op 15469) 
v_accvgpr_read v16, a182 // (op 15469) 
v_accvgpr_read v17, a183 // (op 15469) 
v_mul_f32 v18, v11, v14 // (op 15469) 
v_mul_f32 v19, v11, v15 // (op 15469) 
v_mul_f32 v20, v11, v16 // (op 15469) 
v_mul_f32 v21, v11, v17 // (op 15469) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15441) 
v_mul_f32 v15, v9, v5 // (op 15441) 
v_mul_f32 v16, v9, v6 // (op 15441) 
v_mul_f32 v17, v9, v7 // (op 15441) 

v_add_f32 v4, v14, v18 // (op 15442) 
v_add_f32 v5, v15, v19 // (op 15442) 
v_add_f32 v6, v16, v20 // (op 15442) 
v_add_f32 v7, v17, v21 // (op 15442) 




s_lshl_b32 s0, s4, 6 // (op 15400) 
s_mul_i32 s1, 9, s0 // (op 15400) 
s_add_u32 s1, s1, 192 // (op 15400) 

v_add_u32 v0, v13, s1 // (op 15400) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15400) Store value




s_lshl_b32 s0, s2, 6 // (op 15447) 
s_mul_i32 s1, 10, s0 // (op 15447) 
s_add_u32 s1, s1, 192 // (op 15447) 

v_add_u32 v0, v12, s1 // (op 15447) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15447) Load value

v_accvgpr_read v14, a184 // (op 15471) 
v_accvgpr_read v15, a185 // (op 15471) 
v_accvgpr_read v16, a186 // (op 15471) 
v_accvgpr_read v17, a187 // (op 15471) 
v_mul_f32 v18, v11, v14 // (op 15471) 
v_mul_f32 v19, v11, v15 // (op 15471) 
v_mul_f32 v20, v11, v16 // (op 15471) 
v_mul_f32 v21, v11, v17 // (op 15471) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15445) 
v_mul_f32 v15, v9, v5 // (op 15445) 
v_mul_f32 v16, v9, v6 // (op 15445) 
v_mul_f32 v17, v9, v7 // (op 15445) 

v_add_f32 v4, v14, v18 // (op 15446) 
v_add_f32 v5, v15, v19 // (op 15446) 
v_add_f32 v6, v16, v20 // (op 15446) 
v_add_f32 v7, v17, v21 // (op 15446) 




s_lshl_b32 s0, s4, 6 // (op 15402) 
s_mul_i32 s1, 10, s0 // (op 15402) 
s_add_u32 s1, s1, 192 // (op 15402) 

v_add_u32 v0, v13, s1 // (op 15402) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15402) Store value




s_lshl_b32 s0, s2, 6 // (op 15451) 
s_mul_i32 s1, 11, s0 // (op 15451) 
s_add_u32 s1, s1, 192 // (op 15451) 

v_add_u32 v0, v12, s1 // (op 15451) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15451) Load value

v_accvgpr_read v14, a188 // (op 15473) 
v_accvgpr_read v15, a189 // (op 15473) 
v_accvgpr_read v16, a190 // (op 15473) 
v_accvgpr_read v17, a191 // (op 15473) 
v_mul_f32 v18, v11, v14 // (op 15473) 
v_mul_f32 v19, v11, v15 // (op 15473) 
v_mul_f32 v20, v11, v16 // (op 15473) 
v_mul_f32 v21, v11, v17 // (op 15473) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15449) 
v_mul_f32 v15, v9, v5 // (op 15449) 
v_mul_f32 v16, v9, v6 // (op 15449) 
v_mul_f32 v17, v9, v7 // (op 15449) 

v_add_f32 v4, v14, v18 // (op 15450) 
v_add_f32 v5, v15, v19 // (op 15450) 
v_add_f32 v6, v16, v20 // (op 15450) 
v_add_f32 v7, v17, v21 // (op 15450) 




s_lshl_b32 s0, s4, 6 // (op 15404) 
s_mul_i32 s1, 11, s0 // (op 15404) 
s_add_u32 s1, s1, 192 // (op 15404) 

v_add_u32 v0, v13, s1 // (op 15404) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15404) Store value




v_add_u32 v0, 128, v12 // (op 15176) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15176) Load value

v_accvgpr_read v14, a96 // (op 15148) 
v_accvgpr_read v15, a97 // (op 15148) 
v_accvgpr_read v16, a98 // (op 15148) 
v_accvgpr_read v17, a99 // (op 15148) 
v_mul_f32 v18, v11, v14 // (op 15148) 
v_mul_f32 v19, v11, v15 // (op 15148) 
v_mul_f32 v20, v11, v16 // (op 15148) 
v_mul_f32 v21, v11, v17 // (op 15148) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15175) 
v_mul_f32 v15, v9, v5 // (op 15175) 
v_mul_f32 v16, v9, v6 // (op 15175) 
v_mul_f32 v17, v9, v7 // (op 15175) 

v_add_f32 v4, v14, v18 // (op 15149) 
v_add_f32 v5, v15, v19 // (op 15149) 
v_add_f32 v6, v16, v20 // (op 15149) 
v_add_f32 v7, v17, v21 // (op 15149) 




v_add_u32 v0, 128, v13 // (op 15151) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15151) Store value




s_lshl_b32 s0, s2, 6 // (op 15180) 
s_add_u32 s0, s0, 128 // (op 15180) 
v_add_u32 v0, v12, s0 // (op 15180) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15180) Load value

v_accvgpr_read v14, a100 // (op 15222) 
v_accvgpr_read v15, a101 // (op 15222) 
v_accvgpr_read v16, a102 // (op 15222) 
v_accvgpr_read v17, a103 // (op 15222) 
v_mul_f32 v18, v11, v14 // (op 15222) 
v_mul_f32 v19, v11, v15 // (op 15222) 
v_mul_f32 v20, v11, v16 // (op 15222) 
v_mul_f32 v21, v11, v17 // (op 15222) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15178) 
v_mul_f32 v15, v9, v5 // (op 15178) 
v_mul_f32 v16, v9, v6 // (op 15178) 
v_mul_f32 v17, v9, v7 // (op 15178) 

v_add_f32 v4, v14, v18 // (op 15179) 
v_add_f32 v5, v15, v19 // (op 15179) 
v_add_f32 v6, v16, v20 // (op 15179) 
v_add_f32 v7, v17, v21 // (op 15179) 




s_lshl_b32 s0, s4, 6 // (op 15153) 
s_add_u32 s0, s0, 128 // (op 15153) 
v_add_u32 v0, v13, s0 // (op 15153) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15153) Store value




s_lshl_b32 s0, s2, 7 // (op 15184) 
s_add_u32 s0, s0, 128 // (op 15184) 
v_add_u32 v0, v12, s0 // (op 15184) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15184) Load value

v_accvgpr_read v14, a104 // (op 15224) 
v_accvgpr_read v15, a105 // (op 15224) 
v_accvgpr_read v16, a106 // (op 15224) 
v_accvgpr_read v17, a107 // (op 15224) 
v_mul_f32 v18, v11, v14 // (op 15224) 
v_mul_f32 v19, v11, v15 // (op 15224) 
v_mul_f32 v20, v11, v16 // (op 15224) 
v_mul_f32 v21, v11, v17 // (op 15224) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15182) 
v_mul_f32 v15, v9, v5 // (op 15182) 
v_mul_f32 v16, v9, v6 // (op 15182) 
v_mul_f32 v17, v9, v7 // (op 15182) 

v_add_f32 v4, v14, v18 // (op 15183) 
v_add_f32 v5, v15, v19 // (op 15183) 
v_add_f32 v6, v16, v20 // (op 15183) 
v_add_f32 v7, v17, v21 // (op 15183) 




s_lshl_b32 s0, s4, 7 // (op 15155) 
s_add_u32 s0, s0, 128 // (op 15155) 
v_add_u32 v0, v13, s0 // (op 15155) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15155) Store value




s_lshl_b32 s0, s2, 6 // (op 15188) 
s_mul_i32 s1, 3, s0 // (op 15188) 
s_add_u32 s1, s1, 128 // (op 15188) 

v_add_u32 v0, v12, s1 // (op 15188) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15188) Load value

v_accvgpr_read v14, a108 // (op 15226) 
v_accvgpr_read v15, a109 // (op 15226) 
v_accvgpr_read v16, a110 // (op 15226) 
v_accvgpr_read v17, a111 // (op 15226) 
v_mul_f32 v18, v11, v14 // (op 15226) 
v_mul_f32 v19, v11, v15 // (op 15226) 
v_mul_f32 v20, v11, v16 // (op 15226) 
v_mul_f32 v21, v11, v17 // (op 15226) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15186) 
v_mul_f32 v15, v9, v5 // (op 15186) 
v_mul_f32 v16, v9, v6 // (op 15186) 
v_mul_f32 v17, v9, v7 // (op 15186) 

v_add_f32 v4, v14, v18 // (op 15187) 
v_add_f32 v5, v15, v19 // (op 15187) 
v_add_f32 v6, v16, v20 // (op 15187) 
v_add_f32 v7, v17, v21 // (op 15187) 




s_lshl_b32 s0, s4, 6 // (op 15157) 
s_mul_i32 s1, 3, s0 // (op 15157) 
s_add_u32 s1, s1, 128 // (op 15157) 

v_add_u32 v0, v13, s1 // (op 15157) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15157) Store value




s_lshl_b32 s0, s2, 8 // (op 15192) 
s_add_u32 s0, s0, 128 // (op 15192) 
v_add_u32 v0, v12, s0 // (op 15192) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15192) Load value

v_accvgpr_read v14, a112 // (op 15228) 
v_accvgpr_read v15, a113 // (op 15228) 
v_accvgpr_read v16, a114 // (op 15228) 
v_accvgpr_read v17, a115 // (op 15228) 
v_mul_f32 v18, v11, v14 // (op 15228) 
v_mul_f32 v19, v11, v15 // (op 15228) 
v_mul_f32 v20, v11, v16 // (op 15228) 
v_mul_f32 v21, v11, v17 // (op 15228) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15190) 
v_mul_f32 v15, v9, v5 // (op 15190) 
v_mul_f32 v16, v9, v6 // (op 15190) 
v_mul_f32 v17, v9, v7 // (op 15190) 

v_add_f32 v4, v14, v18 // (op 15191) 
v_add_f32 v5, v15, v19 // (op 15191) 
v_add_f32 v6, v16, v20 // (op 15191) 
v_add_f32 v7, v17, v21 // (op 15191) 




s_lshl_b32 s0, s4, 8 // (op 15159) 
s_add_u32 s0, s0, 128 // (op 15159) 
v_add_u32 v0, v13, s0 // (op 15159) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15159) Store value




s_lshl_b32 s0, s2, 6 // (op 15196) 
s_mul_i32 s1, 5, s0 // (op 15196) 
s_add_u32 s1, s1, 128 // (op 15196) 

v_add_u32 v0, v12, s1 // (op 15196) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15196) Load value

v_accvgpr_read v14, a116 // (op 15230) 
v_accvgpr_read v15, a117 // (op 15230) 
v_accvgpr_read v16, a118 // (op 15230) 
v_accvgpr_read v17, a119 // (op 15230) 
v_mul_f32 v18, v11, v14 // (op 15230) 
v_mul_f32 v19, v11, v15 // (op 15230) 
v_mul_f32 v20, v11, v16 // (op 15230) 
v_mul_f32 v21, v11, v17 // (op 15230) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15194) 
v_mul_f32 v15, v9, v5 // (op 15194) 
v_mul_f32 v16, v9, v6 // (op 15194) 
v_mul_f32 v17, v9, v7 // (op 15194) 

v_add_f32 v4, v14, v18 // (op 15195) 
v_add_f32 v5, v15, v19 // (op 15195) 
v_add_f32 v6, v16, v20 // (op 15195) 
v_add_f32 v7, v17, v21 // (op 15195) 




s_lshl_b32 s0, s4, 6 // (op 15161) 
s_mul_i32 s1, 5, s0 // (op 15161) 
s_add_u32 s1, s1, 128 // (op 15161) 

v_add_u32 v0, v13, s1 // (op 15161) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15161) Store value




s_lshl_b32 s0, s2, 6 // (op 15200) 
s_mul_i32 s1, 6, s0 // (op 15200) 
s_add_u32 s1, s1, 128 // (op 15200) 

v_add_u32 v0, v12, s1 // (op 15200) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15200) Load value

v_accvgpr_read v14, a120 // (op 15232) 
v_accvgpr_read v15, a121 // (op 15232) 
v_accvgpr_read v16, a122 // (op 15232) 
v_accvgpr_read v17, a123 // (op 15232) 
v_mul_f32 v18, v11, v14 // (op 15232) 
v_mul_f32 v19, v11, v15 // (op 15232) 
v_mul_f32 v20, v11, v16 // (op 15232) 
v_mul_f32 v21, v11, v17 // (op 15232) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15198) 
v_mul_f32 v15, v9, v5 // (op 15198) 
v_mul_f32 v16, v9, v6 // (op 15198) 
v_mul_f32 v17, v9, v7 // (op 15198) 

v_add_f32 v4, v14, v18 // (op 15199) 
v_add_f32 v5, v15, v19 // (op 15199) 
v_add_f32 v6, v16, v20 // (op 15199) 
v_add_f32 v7, v17, v21 // (op 15199) 




s_lshl_b32 s0, s4, 6 // (op 15163) 
s_mul_i32 s1, 6, s0 // (op 15163) 
s_add_u32 s1, s1, 128 // (op 15163) 

v_add_u32 v0, v13, s1 // (op 15163) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15163) Store value




s_lshl_b32 s0, s2, 6 // (op 15204) 
s_mul_i32 s1, 7, s0 // (op 15204) 
s_add_u32 s1, s1, 128 // (op 15204) 

v_add_u32 v0, v12, s1 // (op 15204) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15204) Load value

v_accvgpr_read v14, a124 // (op 15234) 
v_accvgpr_read v15, a125 // (op 15234) 
v_accvgpr_read v16, a126 // (op 15234) 
v_accvgpr_read v17, a127 // (op 15234) 
v_mul_f32 v18, v11, v14 // (op 15234) 
v_mul_f32 v19, v11, v15 // (op 15234) 
v_mul_f32 v20, v11, v16 // (op 15234) 
v_mul_f32 v21, v11, v17 // (op 15234) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15202) 
v_mul_f32 v15, v9, v5 // (op 15202) 
v_mul_f32 v16, v9, v6 // (op 15202) 
v_mul_f32 v17, v9, v7 // (op 15202) 

v_add_f32 v4, v14, v18 // (op 15203) 
v_add_f32 v5, v15, v19 // (op 15203) 
v_add_f32 v6, v16, v20 // (op 15203) 
v_add_f32 v7, v17, v21 // (op 15203) 




s_lshl_b32 s0, s4, 6 // (op 15165) 
s_mul_i32 s1, 7, s0 // (op 15165) 
s_add_u32 s1, s1, 128 // (op 15165) 

v_add_u32 v0, v13, s1 // (op 15165) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15165) Store value




s_lshl_b32 s0, s2, 9 // (op 15208) 
s_add_u32 s0, s0, 128 // (op 15208) 
v_add_u32 v0, v12, s0 // (op 15208) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15208) Load value

v_accvgpr_read v14, a128 // (op 15236) 
v_accvgpr_read v15, a129 // (op 15236) 
v_accvgpr_read v16, a130 // (op 15236) 
v_accvgpr_read v17, a131 // (op 15236) 
v_mul_f32 v18, v11, v14 // (op 15236) 
v_mul_f32 v19, v11, v15 // (op 15236) 
v_mul_f32 v20, v11, v16 // (op 15236) 
v_mul_f32 v21, v11, v17 // (op 15236) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15206) 
v_mul_f32 v15, v9, v5 // (op 15206) 
v_mul_f32 v16, v9, v6 // (op 15206) 
v_mul_f32 v17, v9, v7 // (op 15206) 

v_add_f32 v4, v14, v18 // (op 15207) 
v_add_f32 v5, v15, v19 // (op 15207) 
v_add_f32 v6, v16, v20 // (op 15207) 
v_add_f32 v7, v17, v21 // (op 15207) 




s_lshl_b32 s0, s4, 9 // (op 15167) 
s_add_u32 s0, s0, 128 // (op 15167) 
v_add_u32 v0, v13, s0 // (op 15167) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15167) Store value




s_lshl_b32 s0, s2, 6 // (op 15212) 
s_mul_i32 s1, 9, s0 // (op 15212) 
s_add_u32 s1, s1, 128 // (op 15212) 

v_add_u32 v0, v12, s1 // (op 15212) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15212) Load value

v_accvgpr_read v14, a132 // (op 15238) 
v_accvgpr_read v15, a133 // (op 15238) 
v_accvgpr_read v16, a134 // (op 15238) 
v_accvgpr_read v17, a135 // (op 15238) 
v_mul_f32 v18, v11, v14 // (op 15238) 
v_mul_f32 v19, v11, v15 // (op 15238) 
v_mul_f32 v20, v11, v16 // (op 15238) 
v_mul_f32 v21, v11, v17 // (op 15238) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15210) 
v_mul_f32 v15, v9, v5 // (op 15210) 
v_mul_f32 v16, v9, v6 // (op 15210) 
v_mul_f32 v17, v9, v7 // (op 15210) 

v_add_f32 v4, v14, v18 // (op 15211) 
v_add_f32 v5, v15, v19 // (op 15211) 
v_add_f32 v6, v16, v20 // (op 15211) 
v_add_f32 v7, v17, v21 // (op 15211) 




s_lshl_b32 s0, s4, 6 // (op 15169) 
s_mul_i32 s1, 9, s0 // (op 15169) 
s_add_u32 s1, s1, 128 // (op 15169) 

v_add_u32 v0, v13, s1 // (op 15169) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15169) Store value




s_lshl_b32 s0, s2, 6 // (op 15216) 
s_mul_i32 s1, 10, s0 // (op 15216) 
s_add_u32 s1, s1, 128 // (op 15216) 

v_add_u32 v0, v12, s1 // (op 15216) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15216) Load value

v_accvgpr_read v14, a136 // (op 15240) 
v_accvgpr_read v15, a137 // (op 15240) 
v_accvgpr_read v16, a138 // (op 15240) 
v_accvgpr_read v17, a139 // (op 15240) 
v_mul_f32 v18, v11, v14 // (op 15240) 
v_mul_f32 v19, v11, v15 // (op 15240) 
v_mul_f32 v20, v11, v16 // (op 15240) 
v_mul_f32 v21, v11, v17 // (op 15240) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15214) 
v_mul_f32 v15, v9, v5 // (op 15214) 
v_mul_f32 v16, v9, v6 // (op 15214) 
v_mul_f32 v17, v9, v7 // (op 15214) 

v_add_f32 v4, v14, v18 // (op 15215) 
v_add_f32 v5, v15, v19 // (op 15215) 
v_add_f32 v6, v16, v20 // (op 15215) 
v_add_f32 v7, v17, v21 // (op 15215) 




s_lshl_b32 s0, s4, 6 // (op 15171) 
s_mul_i32 s1, 10, s0 // (op 15171) 
s_add_u32 s1, s1, 128 // (op 15171) 

v_add_u32 v0, v13, s1 // (op 15171) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15171) Store value




s_lshl_b32 s0, s2, 6 // (op 15220) 
s_mul_i32 s1, 11, s0 // (op 15220) 
s_add_u32 s1, s1, 128 // (op 15220) 

v_add_u32 v0, v12, s1 // (op 15220) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 15220) Load value

v_accvgpr_read v14, a140 // (op 15242) 
v_accvgpr_read v15, a141 // (op 15242) 
v_accvgpr_read v16, a142 // (op 15242) 
v_accvgpr_read v17, a143 // (op 15242) 
v_mul_f32 v18, v11, v14 // (op 15242) 
v_mul_f32 v19, v11, v15 // (op 15242) 
v_mul_f32 v20, v11, v16 // (op 15242) 
v_mul_f32 v21, v11, v17 // (op 15242) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 15218) 
v_mul_f32 v15, v9, v5 // (op 15218) 
v_mul_f32 v16, v9, v6 // (op 15218) 
v_mul_f32 v17, v9, v7 // (op 15218) 

v_add_f32 v4, v14, v18 // (op 15219) 
v_add_f32 v5, v15, v19 // (op 15219) 
v_add_f32 v6, v16, v20 // (op 15219) 
v_add_f32 v7, v17, v21 // (op 15219) 




s_lshl_b32 s0, s4, 6 // (op 15173) 
s_mul_i32 s1, 11, s0 // (op 15173) 
s_add_u32 s1, s1, 128 // (op 15173) 

v_add_u32 v0, v13, s1 // (op 15173) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 15173) Store value




v_add_u32 v0, 64, v12 // (op 14945) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14945) Load value

v_accvgpr_read v14, a48 // (op 14917) 
v_accvgpr_read v15, a49 // (op 14917) 
v_accvgpr_read v16, a50 // (op 14917) 
v_accvgpr_read v17, a51 // (op 14917) 
v_mul_f32 v18, v11, v14 // (op 14917) 
v_mul_f32 v19, v11, v15 // (op 14917) 
v_mul_f32 v20, v11, v16 // (op 14917) 
v_mul_f32 v21, v11, v17 // (op 14917) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14944) 
v_mul_f32 v15, v9, v5 // (op 14944) 
v_mul_f32 v16, v9, v6 // (op 14944) 
v_mul_f32 v17, v9, v7 // (op 14944) 

v_add_f32 v4, v14, v18 // (op 14918) 
v_add_f32 v5, v15, v19 // (op 14918) 
v_add_f32 v6, v16, v20 // (op 14918) 
v_add_f32 v7, v17, v21 // (op 14918) 




v_add_u32 v0, 64, v13 // (op 14920) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14920) Store value




s_lshl_b32 s0, s2, 6 // (op 14949) 
s_add_u32 s0, s0, 64 // (op 14949) 
v_add_u32 v0, v12, s0 // (op 14949) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14949) Load value

v_accvgpr_read v14, a52 // (op 14991) 
v_accvgpr_read v15, a53 // (op 14991) 
v_accvgpr_read v16, a54 // (op 14991) 
v_accvgpr_read v17, a55 // (op 14991) 
v_mul_f32 v18, v11, v14 // (op 14991) 
v_mul_f32 v19, v11, v15 // (op 14991) 
v_mul_f32 v20, v11, v16 // (op 14991) 
v_mul_f32 v21, v11, v17 // (op 14991) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14947) 
v_mul_f32 v15, v9, v5 // (op 14947) 
v_mul_f32 v16, v9, v6 // (op 14947) 
v_mul_f32 v17, v9, v7 // (op 14947) 

v_add_f32 v4, v14, v18 // (op 14948) 
v_add_f32 v5, v15, v19 // (op 14948) 
v_add_f32 v6, v16, v20 // (op 14948) 
v_add_f32 v7, v17, v21 // (op 14948) 




s_lshl_b32 s0, s4, 6 // (op 14922) 
s_add_u32 s0, s0, 64 // (op 14922) 
v_add_u32 v0, v13, s0 // (op 14922) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14922) Store value




s_lshl_b32 s0, s2, 7 // (op 14953) 
s_add_u32 s0, s0, 64 // (op 14953) 
v_add_u32 v0, v12, s0 // (op 14953) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14953) Load value

v_accvgpr_read v14, a56 // (op 14993) 
v_accvgpr_read v15, a57 // (op 14993) 
v_accvgpr_read v16, a58 // (op 14993) 
v_accvgpr_read v17, a59 // (op 14993) 
v_mul_f32 v18, v11, v14 // (op 14993) 
v_mul_f32 v19, v11, v15 // (op 14993) 
v_mul_f32 v20, v11, v16 // (op 14993) 
v_mul_f32 v21, v11, v17 // (op 14993) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14951) 
v_mul_f32 v15, v9, v5 // (op 14951) 
v_mul_f32 v16, v9, v6 // (op 14951) 
v_mul_f32 v17, v9, v7 // (op 14951) 

v_add_f32 v4, v14, v18 // (op 14952) 
v_add_f32 v5, v15, v19 // (op 14952) 
v_add_f32 v6, v16, v20 // (op 14952) 
v_add_f32 v7, v17, v21 // (op 14952) 




s_lshl_b32 s0, s4, 7 // (op 14924) 
s_add_u32 s0, s0, 64 // (op 14924) 
v_add_u32 v0, v13, s0 // (op 14924) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14924) Store value




s_lshl_b32 s0, s2, 6 // (op 14957) 
s_mul_i32 s1, 3, s0 // (op 14957) 
s_add_u32 s1, s1, 64 // (op 14957) 

v_add_u32 v0, v12, s1 // (op 14957) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14957) Load value

v_accvgpr_read v14, a60 // (op 14995) 
v_accvgpr_read v15, a61 // (op 14995) 
v_accvgpr_read v16, a62 // (op 14995) 
v_accvgpr_read v17, a63 // (op 14995) 
v_mul_f32 v18, v11, v14 // (op 14995) 
v_mul_f32 v19, v11, v15 // (op 14995) 
v_mul_f32 v20, v11, v16 // (op 14995) 
v_mul_f32 v21, v11, v17 // (op 14995) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14955) 
v_mul_f32 v15, v9, v5 // (op 14955) 
v_mul_f32 v16, v9, v6 // (op 14955) 
v_mul_f32 v17, v9, v7 // (op 14955) 

v_add_f32 v4, v14, v18 // (op 14956) 
v_add_f32 v5, v15, v19 // (op 14956) 
v_add_f32 v6, v16, v20 // (op 14956) 
v_add_f32 v7, v17, v21 // (op 14956) 




s_lshl_b32 s0, s4, 6 // (op 14926) 
s_mul_i32 s1, 3, s0 // (op 14926) 
s_add_u32 s1, s1, 64 // (op 14926) 

v_add_u32 v0, v13, s1 // (op 14926) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14926) Store value




s_lshl_b32 s0, s2, 8 // (op 14961) 
s_add_u32 s0, s0, 64 // (op 14961) 
v_add_u32 v0, v12, s0 // (op 14961) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14961) Load value

v_accvgpr_read v14, a64 // (op 14997) 
v_accvgpr_read v15, a65 // (op 14997) 
v_accvgpr_read v16, a66 // (op 14997) 
v_accvgpr_read v17, a67 // (op 14997) 
v_mul_f32 v18, v11, v14 // (op 14997) 
v_mul_f32 v19, v11, v15 // (op 14997) 
v_mul_f32 v20, v11, v16 // (op 14997) 
v_mul_f32 v21, v11, v17 // (op 14997) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14959) 
v_mul_f32 v15, v9, v5 // (op 14959) 
v_mul_f32 v16, v9, v6 // (op 14959) 
v_mul_f32 v17, v9, v7 // (op 14959) 

v_add_f32 v4, v14, v18 // (op 14960) 
v_add_f32 v5, v15, v19 // (op 14960) 
v_add_f32 v6, v16, v20 // (op 14960) 
v_add_f32 v7, v17, v21 // (op 14960) 




s_lshl_b32 s0, s4, 8 // (op 14928) 
s_add_u32 s0, s0, 64 // (op 14928) 
v_add_u32 v0, v13, s0 // (op 14928) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14928) Store value




s_lshl_b32 s0, s2, 6 // (op 14965) 
s_mul_i32 s1, 5, s0 // (op 14965) 
s_add_u32 s1, s1, 64 // (op 14965) 

v_add_u32 v0, v12, s1 // (op 14965) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14965) Load value

v_accvgpr_read v14, a68 // (op 14999) 
v_accvgpr_read v15, a69 // (op 14999) 
v_accvgpr_read v16, a70 // (op 14999) 
v_accvgpr_read v17, a71 // (op 14999) 
v_mul_f32 v18, v11, v14 // (op 14999) 
v_mul_f32 v19, v11, v15 // (op 14999) 
v_mul_f32 v20, v11, v16 // (op 14999) 
v_mul_f32 v21, v11, v17 // (op 14999) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14963) 
v_mul_f32 v15, v9, v5 // (op 14963) 
v_mul_f32 v16, v9, v6 // (op 14963) 
v_mul_f32 v17, v9, v7 // (op 14963) 

v_add_f32 v4, v14, v18 // (op 14964) 
v_add_f32 v5, v15, v19 // (op 14964) 
v_add_f32 v6, v16, v20 // (op 14964) 
v_add_f32 v7, v17, v21 // (op 14964) 




s_lshl_b32 s0, s4, 6 // (op 14930) 
s_mul_i32 s1, 5, s0 // (op 14930) 
s_add_u32 s1, s1, 64 // (op 14930) 

v_add_u32 v0, v13, s1 // (op 14930) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14930) Store value




s_lshl_b32 s0, s2, 6 // (op 14969) 
s_mul_i32 s1, 6, s0 // (op 14969) 
s_add_u32 s1, s1, 64 // (op 14969) 

v_add_u32 v0, v12, s1 // (op 14969) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14969) Load value

v_accvgpr_read v14, a72 // (op 15001) 
v_accvgpr_read v15, a73 // (op 15001) 
v_accvgpr_read v16, a74 // (op 15001) 
v_accvgpr_read v17, a75 // (op 15001) 
v_mul_f32 v18, v11, v14 // (op 15001) 
v_mul_f32 v19, v11, v15 // (op 15001) 
v_mul_f32 v20, v11, v16 // (op 15001) 
v_mul_f32 v21, v11, v17 // (op 15001) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14967) 
v_mul_f32 v15, v9, v5 // (op 14967) 
v_mul_f32 v16, v9, v6 // (op 14967) 
v_mul_f32 v17, v9, v7 // (op 14967) 

v_add_f32 v4, v14, v18 // (op 14968) 
v_add_f32 v5, v15, v19 // (op 14968) 
v_add_f32 v6, v16, v20 // (op 14968) 
v_add_f32 v7, v17, v21 // (op 14968) 




s_lshl_b32 s0, s4, 6 // (op 14932) 
s_mul_i32 s1, 6, s0 // (op 14932) 
s_add_u32 s1, s1, 64 // (op 14932) 

v_add_u32 v0, v13, s1 // (op 14932) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14932) Store value




s_lshl_b32 s0, s2, 6 // (op 14973) 
s_mul_i32 s1, 7, s0 // (op 14973) 
s_add_u32 s1, s1, 64 // (op 14973) 

v_add_u32 v0, v12, s1 // (op 14973) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14973) Load value

v_accvgpr_read v14, a76 // (op 15003) 
v_accvgpr_read v15, a77 // (op 15003) 
v_accvgpr_read v16, a78 // (op 15003) 
v_accvgpr_read v17, a79 // (op 15003) 
v_mul_f32 v18, v11, v14 // (op 15003) 
v_mul_f32 v19, v11, v15 // (op 15003) 
v_mul_f32 v20, v11, v16 // (op 15003) 
v_mul_f32 v21, v11, v17 // (op 15003) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14971) 
v_mul_f32 v15, v9, v5 // (op 14971) 
v_mul_f32 v16, v9, v6 // (op 14971) 
v_mul_f32 v17, v9, v7 // (op 14971) 

v_add_f32 v4, v14, v18 // (op 14972) 
v_add_f32 v5, v15, v19 // (op 14972) 
v_add_f32 v6, v16, v20 // (op 14972) 
v_add_f32 v7, v17, v21 // (op 14972) 




s_lshl_b32 s0, s4, 6 // (op 14934) 
s_mul_i32 s1, 7, s0 // (op 14934) 
s_add_u32 s1, s1, 64 // (op 14934) 

v_add_u32 v0, v13, s1 // (op 14934) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14934) Store value




s_lshl_b32 s0, s2, 9 // (op 14977) 
s_add_u32 s0, s0, 64 // (op 14977) 
v_add_u32 v0, v12, s0 // (op 14977) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14977) Load value

v_accvgpr_read v14, a80 // (op 15005) 
v_accvgpr_read v15, a81 // (op 15005) 
v_accvgpr_read v16, a82 // (op 15005) 
v_accvgpr_read v17, a83 // (op 15005) 
v_mul_f32 v18, v11, v14 // (op 15005) 
v_mul_f32 v19, v11, v15 // (op 15005) 
v_mul_f32 v20, v11, v16 // (op 15005) 
v_mul_f32 v21, v11, v17 // (op 15005) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14975) 
v_mul_f32 v15, v9, v5 // (op 14975) 
v_mul_f32 v16, v9, v6 // (op 14975) 
v_mul_f32 v17, v9, v7 // (op 14975) 

v_add_f32 v4, v14, v18 // (op 14976) 
v_add_f32 v5, v15, v19 // (op 14976) 
v_add_f32 v6, v16, v20 // (op 14976) 
v_add_f32 v7, v17, v21 // (op 14976) 




s_lshl_b32 s0, s4, 9 // (op 14936) 
s_add_u32 s0, s0, 64 // (op 14936) 
v_add_u32 v0, v13, s0 // (op 14936) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14936) Store value




s_lshl_b32 s0, s2, 6 // (op 14981) 
s_mul_i32 s1, 9, s0 // (op 14981) 
s_add_u32 s1, s1, 64 // (op 14981) 

v_add_u32 v0, v12, s1 // (op 14981) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14981) Load value

v_accvgpr_read v14, a84 // (op 15007) 
v_accvgpr_read v15, a85 // (op 15007) 
v_accvgpr_read v16, a86 // (op 15007) 
v_accvgpr_read v17, a87 // (op 15007) 
v_mul_f32 v18, v11, v14 // (op 15007) 
v_mul_f32 v19, v11, v15 // (op 15007) 
v_mul_f32 v20, v11, v16 // (op 15007) 
v_mul_f32 v21, v11, v17 // (op 15007) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14979) 
v_mul_f32 v15, v9, v5 // (op 14979) 
v_mul_f32 v16, v9, v6 // (op 14979) 
v_mul_f32 v17, v9, v7 // (op 14979) 

v_add_f32 v4, v14, v18 // (op 14980) 
v_add_f32 v5, v15, v19 // (op 14980) 
v_add_f32 v6, v16, v20 // (op 14980) 
v_add_f32 v7, v17, v21 // (op 14980) 




s_lshl_b32 s0, s4, 6 // (op 14938) 
s_mul_i32 s1, 9, s0 // (op 14938) 
s_add_u32 s1, s1, 64 // (op 14938) 

v_add_u32 v0, v13, s1 // (op 14938) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14938) Store value




s_lshl_b32 s0, s2, 6 // (op 14985) 
s_mul_i32 s1, 10, s0 // (op 14985) 
s_add_u32 s1, s1, 64 // (op 14985) 

v_add_u32 v0, v12, s1 // (op 14985) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14985) Load value

v_accvgpr_read v14, a88 // (op 15009) 
v_accvgpr_read v15, a89 // (op 15009) 
v_accvgpr_read v16, a90 // (op 15009) 
v_accvgpr_read v17, a91 // (op 15009) 
v_mul_f32 v18, v11, v14 // (op 15009) 
v_mul_f32 v19, v11, v15 // (op 15009) 
v_mul_f32 v20, v11, v16 // (op 15009) 
v_mul_f32 v21, v11, v17 // (op 15009) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 14983) 
v_mul_f32 v15, v9, v5 // (op 14983) 
v_mul_f32 v16, v9, v6 // (op 14983) 
v_mul_f32 v17, v9, v7 // (op 14983) 

v_add_f32 v4, v14, v18 // (op 14984) 
v_add_f32 v5, v15, v19 // (op 14984) 
v_add_f32 v6, v16, v20 // (op 14984) 
v_add_f32 v7, v17, v21 // (op 14984) 




s_lshl_b32 s0, s4, 6 // (op 14940) 
s_mul_i32 s1, 10, s0 // (op 14940) 
s_add_u32 s1, s1, 64 // (op 14940) 

v_add_u32 v0, v13, s1 // (op 14940) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14940) Store value




s_lshl_b32 s0, s2, 6 // (op 14989) 
s_mul_i32 s1, 11, s0 // (op 14989) 
s_add_u32 s1, s1, 64 // (op 14989) 

v_add_u32 v0, v12, s1 // (op 14989) 


v_accvgpr_read v4, a92 // (op 15011) 
v_accvgpr_read v5, a93 // (op 15011) 
v_accvgpr_read v6, a94 // (op 15011) 
v_accvgpr_read v7, a95 // (op 15011) 
v_mul_f32 v14, v11, v4 // (op 15011) 
v_mul_f32 v15, v11, v5 // (op 15011) 
v_mul_f32 v16, v11, v6 // (op 15011) 
v_mul_f32 v17, v11, v7 // (op 15011) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14989) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14987) 
v_mul_f32 v19, v9, v5 // (op 14987) 
v_mul_f32 v20, v9, v6 // (op 14987) 
v_mul_f32 v21, v9, v7 // (op 14987) 

v_add_f32 v4, v18, v14 // (op 14988) 
v_add_f32 v5, v19, v15 // (op 14988) 
v_add_f32 v6, v20, v16 // (op 14988) 
v_add_f32 v7, v21, v17 // (op 14988) 




s_lshl_b32 s0, s4, 6 // (op 14942) 
s_mul_i32 s1, 11, s0 // (op 14942) 
s_add_u32 s1, s1, 64 // (op 14942) 

v_add_u32 v0, v13, s1 // (op 14942) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14942) Store value




v_mov_b32 v0, v12 // (op 126) reg expression

s_nop 0
v_accvgpr_read v4, a0 // (op 123) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v5, a1 // (op 123) 
v_accvgpr_read v6, a2 // (op 123) 
v_accvgpr_read v7, a3 // (op 123) 
v_mul_f32 v14, v11, v4 // (op 123) 
v_mul_f32 v15, v11, v5 // (op 123) 
v_mul_f32 v16, v11, v6 // (op 123) 
v_mul_f32 v17, v11, v7 // (op 123) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 126) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 127) 
v_mul_f32 v19, v9, v5 // (op 127) 
v_mul_f32 v20, v9, v6 // (op 127) 
v_mul_f32 v21, v9, v7 // (op 127) 

v_add_f32 v4, v18, v14 // (op 124) 
v_add_f32 v5, v19, v15 // (op 124) 
v_add_f32 v6, v20, v16 // (op 124) 
v_add_f32 v7, v21, v17 // (op 124) 




v_mov_b32 v0, v13 // (op 125) reg expression

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 14667) 
v_add_u32 v0, v12, s0 // (op 14667) 


v_accvgpr_read v4, a4 // (op 14664) 
v_accvgpr_read v5, a5 // (op 14664) 
v_accvgpr_read v6, a6 // (op 14664) 
v_accvgpr_read v7, a7 // (op 14664) 
v_mul_f32 v14, v11, v4 // (op 14664) 
v_mul_f32 v15, v11, v5 // (op 14664) 
v_mul_f32 v16, v11, v6 // (op 14664) 
v_mul_f32 v17, v11, v7 // (op 14664) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14667) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14668) 
v_mul_f32 v19, v9, v5 // (op 14668) 
v_mul_f32 v20, v9, v6 // (op 14668) 
v_mul_f32 v21, v9, v7 // (op 14668) 

v_add_f32 v4, v18, v14 // (op 14665) 
v_add_f32 v5, v19, v15 // (op 14665) 
v_add_f32 v6, v20, v16 // (op 14665) 
v_add_f32 v7, v21, v17 // (op 14665) 




s_lshl_b32 s0, s4, 6 // (op 14666) 
v_add_u32 v0, v13, s0 // (op 14666) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14666) Store value




s_lshl_b32 s0, s2, 7 // (op 14679) 
v_add_u32 v0, v12, s0 // (op 14679) 


v_accvgpr_read v4, a8 // (op 14676) 
v_accvgpr_read v5, a9 // (op 14676) 
v_accvgpr_read v6, a10 // (op 14676) 
v_accvgpr_read v7, a11 // (op 14676) 
v_mul_f32 v14, v11, v4 // (op 14676) 
v_mul_f32 v15, v11, v5 // (op 14676) 
v_mul_f32 v16, v11, v6 // (op 14676) 
v_mul_f32 v17, v11, v7 // (op 14676) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14679) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14680) 
v_mul_f32 v19, v9, v5 // (op 14680) 
v_mul_f32 v20, v9, v6 // (op 14680) 
v_mul_f32 v21, v9, v7 // (op 14680) 

v_add_f32 v4, v18, v14 // (op 14677) 
v_add_f32 v5, v19, v15 // (op 14677) 
v_add_f32 v6, v20, v16 // (op 14677) 
v_add_f32 v7, v21, v17 // (op 14677) 




s_lshl_b32 s0, s4, 7 // (op 14678) 
v_add_u32 v0, v13, s0 // (op 14678) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14678) Store value




s_lshl_b32 s0, s2, 6 // (op 14691) 
s_mul_i32 s1, 3, s0 // (op 14691) 

v_add_u32 v0, v12, s1 // (op 14691) 


v_accvgpr_read v4, a12 // (op 14688) 
v_accvgpr_read v5, a13 // (op 14688) 
v_accvgpr_read v6, a14 // (op 14688) 
v_accvgpr_read v7, a15 // (op 14688) 
v_mul_f32 v14, v11, v4 // (op 14688) 
v_mul_f32 v15, v11, v5 // (op 14688) 
v_mul_f32 v16, v11, v6 // (op 14688) 
v_mul_f32 v17, v11, v7 // (op 14688) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14691) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14692) 
v_mul_f32 v19, v9, v5 // (op 14692) 
v_mul_f32 v20, v9, v6 // (op 14692) 
v_mul_f32 v21, v9, v7 // (op 14692) 

v_add_f32 v4, v18, v14 // (op 14689) 
v_add_f32 v5, v19, v15 // (op 14689) 
v_add_f32 v6, v20, v16 // (op 14689) 
v_add_f32 v7, v21, v17 // (op 14689) 




s_lshl_b32 s0, s4, 6 // (op 14690) 
s_mul_i32 s1, 3, s0 // (op 14690) 

v_add_u32 v0, v13, s1 // (op 14690) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14690) Store value




s_lshl_b32 s0, s2, 8 // (op 14703) 
v_add_u32 v0, v12, s0 // (op 14703) 


v_accvgpr_read v4, a16 // (op 14700) 
v_accvgpr_read v5, a17 // (op 14700) 
v_accvgpr_read v6, a18 // (op 14700) 
v_accvgpr_read v7, a19 // (op 14700) 
v_mul_f32 v14, v11, v4 // (op 14700) 
v_mul_f32 v15, v11, v5 // (op 14700) 
v_mul_f32 v16, v11, v6 // (op 14700) 
v_mul_f32 v17, v11, v7 // (op 14700) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14703) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14704) 
v_mul_f32 v19, v9, v5 // (op 14704) 
v_mul_f32 v20, v9, v6 // (op 14704) 
v_mul_f32 v21, v9, v7 // (op 14704) 

v_add_f32 v4, v18, v14 // (op 14701) 
v_add_f32 v5, v19, v15 // (op 14701) 
v_add_f32 v6, v20, v16 // (op 14701) 
v_add_f32 v7, v21, v17 // (op 14701) 




s_lshl_b32 s0, s4, 8 // (op 14702) 
v_add_u32 v0, v13, s0 // (op 14702) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14702) Store value




s_lshl_b32 s0, s2, 6 // (op 14715) 
s_mul_i32 s1, 5, s0 // (op 14715) 

v_add_u32 v0, v12, s1 // (op 14715) 


v_accvgpr_read v4, a20 // (op 14712) 
v_accvgpr_read v5, a21 // (op 14712) 
v_accvgpr_read v6, a22 // (op 14712) 
v_accvgpr_read v7, a23 // (op 14712) 
v_mul_f32 v14, v11, v4 // (op 14712) 
v_mul_f32 v15, v11, v5 // (op 14712) 
v_mul_f32 v16, v11, v6 // (op 14712) 
v_mul_f32 v17, v11, v7 // (op 14712) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14715) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14716) 
v_mul_f32 v19, v9, v5 // (op 14716) 
v_mul_f32 v20, v9, v6 // (op 14716) 
v_mul_f32 v21, v9, v7 // (op 14716) 

v_add_f32 v4, v18, v14 // (op 14713) 
v_add_f32 v5, v19, v15 // (op 14713) 
v_add_f32 v6, v20, v16 // (op 14713) 
v_add_f32 v7, v21, v17 // (op 14713) 




s_lshl_b32 s0, s4, 6 // (op 14714) 
s_mul_i32 s1, 5, s0 // (op 14714) 

v_add_u32 v0, v13, s1 // (op 14714) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14714) Store value




s_lshl_b32 s0, s2, 6 // (op 14727) 
s_mul_i32 s1, 6, s0 // (op 14727) 

v_add_u32 v0, v12, s1 // (op 14727) 


v_accvgpr_read v4, a24 // (op 14724) 
v_accvgpr_read v5, a25 // (op 14724) 
v_accvgpr_read v6, a26 // (op 14724) 
v_accvgpr_read v7, a27 // (op 14724) 
v_mul_f32 v14, v11, v4 // (op 14724) 
v_mul_f32 v15, v11, v5 // (op 14724) 
v_mul_f32 v16, v11, v6 // (op 14724) 
v_mul_f32 v17, v11, v7 // (op 14724) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14727) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14728) 
v_mul_f32 v19, v9, v5 // (op 14728) 
v_mul_f32 v20, v9, v6 // (op 14728) 
v_mul_f32 v21, v9, v7 // (op 14728) 

v_add_f32 v4, v18, v14 // (op 14725) 
v_add_f32 v5, v19, v15 // (op 14725) 
v_add_f32 v6, v20, v16 // (op 14725) 
v_add_f32 v7, v21, v17 // (op 14725) 




s_lshl_b32 s0, s4, 6 // (op 14726) 
s_mul_i32 s1, 6, s0 // (op 14726) 

v_add_u32 v0, v13, s1 // (op 14726) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14726) Store value




s_lshl_b32 s0, s2, 6 // (op 14739) 
s_mul_i32 s1, 7, s0 // (op 14739) 

v_add_u32 v0, v12, s1 // (op 14739) 


v_accvgpr_read v4, a28 // (op 14736) 
v_accvgpr_read v5, a29 // (op 14736) 
v_accvgpr_read v6, a30 // (op 14736) 
v_accvgpr_read v7, a31 // (op 14736) 
v_mul_f32 v14, v11, v4 // (op 14736) 
v_mul_f32 v15, v11, v5 // (op 14736) 
v_mul_f32 v16, v11, v6 // (op 14736) 
v_mul_f32 v17, v11, v7 // (op 14736) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14739) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14740) 
v_mul_f32 v19, v9, v5 // (op 14740) 
v_mul_f32 v20, v9, v6 // (op 14740) 
v_mul_f32 v21, v9, v7 // (op 14740) 

v_add_f32 v4, v18, v14 // (op 14737) 
v_add_f32 v5, v19, v15 // (op 14737) 
v_add_f32 v6, v20, v16 // (op 14737) 
v_add_f32 v7, v21, v17 // (op 14737) 




s_lshl_b32 s0, s4, 6 // (op 14738) 
s_mul_i32 s1, 7, s0 // (op 14738) 

v_add_u32 v0, v13, s1 // (op 14738) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14738) Store value




s_lshl_b32 s0, s2, 9 // (op 14751) 
v_add_u32 v0, v12, s0 // (op 14751) 


v_accvgpr_read v4, a32 // (op 14748) 
v_accvgpr_read v5, a33 // (op 14748) 
v_accvgpr_read v6, a34 // (op 14748) 
v_accvgpr_read v7, a35 // (op 14748) 
v_mul_f32 v14, v11, v4 // (op 14748) 
v_mul_f32 v15, v11, v5 // (op 14748) 
v_mul_f32 v16, v11, v6 // (op 14748) 
v_mul_f32 v17, v11, v7 // (op 14748) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14751) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14752) 
v_mul_f32 v19, v9, v5 // (op 14752) 
v_mul_f32 v20, v9, v6 // (op 14752) 
v_mul_f32 v21, v9, v7 // (op 14752) 

v_add_f32 v4, v18, v14 // (op 14749) 
v_add_f32 v5, v19, v15 // (op 14749) 
v_add_f32 v6, v20, v16 // (op 14749) 
v_add_f32 v7, v21, v17 // (op 14749) 




s_lshl_b32 s0, s4, 9 // (op 14750) 
v_add_u32 v0, v13, s0 // (op 14750) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14750) Store value




s_lshl_b32 s0, s2, 6 // (op 14763) 
s_mul_i32 s1, 9, s0 // (op 14763) 

v_add_u32 v0, v12, s1 // (op 14763) 


v_accvgpr_read v4, a36 // (op 14760) 
v_accvgpr_read v5, a37 // (op 14760) 
v_accvgpr_read v6, a38 // (op 14760) 
v_accvgpr_read v7, a39 // (op 14760) 
v_mul_f32 v14, v11, v4 // (op 14760) 
v_mul_f32 v15, v11, v5 // (op 14760) 
v_mul_f32 v16, v11, v6 // (op 14760) 
v_mul_f32 v17, v11, v7 // (op 14760) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14763) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14764) 
v_mul_f32 v19, v9, v5 // (op 14764) 
v_mul_f32 v20, v9, v6 // (op 14764) 
v_mul_f32 v21, v9, v7 // (op 14764) 

v_add_f32 v4, v18, v14 // (op 14761) 
v_add_f32 v5, v19, v15 // (op 14761) 
v_add_f32 v6, v20, v16 // (op 14761) 
v_add_f32 v7, v21, v17 // (op 14761) 




s_lshl_b32 s0, s4, 6 // (op 14762) 
s_mul_i32 s1, 9, s0 // (op 14762) 

v_add_u32 v0, v13, s1 // (op 14762) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14762) Store value




s_lshl_b32 s0, s2, 6 // (op 14775) 
s_mul_i32 s1, 10, s0 // (op 14775) 

v_add_u32 v0, v12, s1 // (op 14775) 


v_accvgpr_read v4, a40 // (op 14772) 
v_accvgpr_read v5, a41 // (op 14772) 
v_accvgpr_read v6, a42 // (op 14772) 
v_accvgpr_read v7, a43 // (op 14772) 
v_mul_f32 v14, v11, v4 // (op 14772) 
v_mul_f32 v15, v11, v5 // (op 14772) 
v_mul_f32 v16, v11, v6 // (op 14772) 
v_mul_f32 v17, v11, v7 // (op 14772) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14775) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14776) 
v_mul_f32 v19, v9, v5 // (op 14776) 
v_mul_f32 v20, v9, v6 // (op 14776) 
v_mul_f32 v21, v9, v7 // (op 14776) 

v_add_f32 v4, v18, v14 // (op 14773) 
v_add_f32 v5, v19, v15 // (op 14773) 
v_add_f32 v6, v20, v16 // (op 14773) 
v_add_f32 v7, v21, v17 // (op 14773) 




s_lshl_b32 s0, s4, 6 // (op 14774) 
s_mul_i32 s1, 10, s0 // (op 14774) 

v_add_u32 v0, v13, s1 // (op 14774) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14774) Store value




s_lshl_b32 s0, s2, 6 // (op 14787) 
s_mul_i32 s1, 11, s0 // (op 14787) 

v_add_u32 v0, v12, s1 // (op 14787) 


v_accvgpr_read v4, a44 // (op 14784) 
v_accvgpr_read v5, a45 // (op 14784) 
v_accvgpr_read v6, a46 // (op 14784) 
v_accvgpr_read v7, a47 // (op 14784) 
v_mul_f32 v14, v11, v4 // (op 14784) 
v_mul_f32 v15, v11, v5 // (op 14784) 
v_mul_f32 v16, v11, v6 // (op 14784) 
v_mul_f32 v17, v11, v7 // (op 14784) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 14787) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 14788) 
v_mul_f32 v19, v9, v5 // (op 14788) 
v_mul_f32 v20, v9, v6 // (op 14788) 
v_mul_f32 v21, v9, v7 // (op 14788) 

v_add_f32 v4, v18, v14 // (op 14785) 
v_add_f32 v5, v19, v15 // (op 14785) 
v_add_f32 v6, v20, v16 // (op 14785) 
v_add_f32 v7, v21, v17 // (op 14785) 




s_lshl_b32 s0, s4, 6 // (op 14786) 
s_mul_i32 s1, 11, s0 // (op 14786) 

v_add_u32 v0, v13, s1 // (op 14786) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 14786) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_ConditionalBottom_0_beta_114:
























































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4
  .amdhsa_next_free_vgpr 388
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 81920 // lds bytes
  .amdhsa_accum_offset 196
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
    DestDataType: s
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
    macrotile: [128, 192, 128]
    threads: [128, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 12]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x192x128_UR_4.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 81920
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 66
    .vgpr_count: 194
    .agpr_count: 192
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {type: Multiply, lhs: {type: Convert, arg: {type: Kernel Argument, name: BitwiseXor_18, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: -6148914691236517205}}, rhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 192}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 63}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 128}}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, rhs: {type: LiteralValue.Int64, dataType: Int64, value: 0}}, offset: 136, size: 8}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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

