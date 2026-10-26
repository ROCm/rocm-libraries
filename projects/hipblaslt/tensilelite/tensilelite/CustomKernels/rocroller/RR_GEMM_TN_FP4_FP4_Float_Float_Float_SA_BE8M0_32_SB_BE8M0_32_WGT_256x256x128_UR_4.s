

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_exec_begin:


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


v_and_b32 v0, 4294967168, v1 // (op 41142) 
v_and_b32 v3, 15, v1 // (op 41142) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 41142) 


v_and_b32 v0, 63, v1 // (op 41142) 
v_lshrrev_b32 v3, 4, v0 // (op 41142) 

v_lshlrev_b32 v0, 5, v3 // (op 41142) 

v_add_lshl_u32 v3, v4, v0, 2 // (op 41142) 


v_lshrrev_b32 v0, 3, v3 // (op 41142) 

v_lshrrev_b32 v3, 6, v1 // (op 41199) 
v_and_b32 v4, 1, v3 // (op 41199) 

v_and_b32 v3, 15, v1 // (op 41199) 
v_lshl_add_u32 v5, v4, 7, v3 // (op 41199) 


v_and_b32 v3, 63, v1 // (op 41199) 
v_lshrrev_b32 v4, 4, v3 // (op 41199) 

v_lshlrev_b32 v3, 5, v4 // (op 41199) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 41199) 


v_lshlrev_b32 v3, 2, v4 // (op 41199) 

v_lshrrev_b32 v4, 3, v3 // (op 41199) 

v_and_b32 v3, 4294967168, v1 // (op 41266) 
v_and_b32 v5, 15, v1 // (op 41266) 
v_add_lshl_u32 v6, v3, v5, 7 // (op 41266) 


v_and_b32 v3, 63, v1 // (op 41266) 
v_lshrrev_b32 v5, 4, v3 // (op 41266) 

v_lshlrev_b32 v3, 5, v5 // (op 41266) 

v_add_lshl_u32 v5, v6, v3, 2 // (op 41266) 


v_lshrrev_b32 v3, 3, v5 // (op 41266) 

v_lshrrev_b32 v5, 6, v1 // (op 41310) 
v_and_b32 v6, 1, v5 // (op 41310) 

v_and_b32 v5, 15, v1 // (op 41310) 
v_lshl_add_u32 v7, v6, 7, v5 // (op 41310) 


v_and_b32 v5, 63, v1 // (op 41310) 
v_lshrrev_b32 v6, 4, v5 // (op 41310) 

v_lshlrev_b32 v5, 5, v6 // (op 41310) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 41310) 


v_lshlrev_b32 v5, 2, v6 // (op 41310) 

v_lshrrev_b32 v6, 3, v5 // (op 41310) 

v_and_b32 v5, 4294967168, v1 // (op 41354) 
v_and_b32 v7, 15, v1 // (op 41354) 
v_add_lshl_u32 v8, v5, v7, 7 // (op 41354) 


v_and_b32 v5, 63, v1 // (op 41354) 
v_lshrrev_b32 v7, 4, v5 // (op 41354) 

v_lshlrev_b32 v5, 5, v7 // (op 41354) 

v_add_lshl_u32 v7, v8, v5, 2 // (op 41354) 


v_lshrrev_b32 v5, 3, v7 // (op 41354) 

v_lshrrev_b32 v7, 6, v1 // (op 41398) 
v_and_b32 v8, 1, v7 // (op 41398) 

v_and_b32 v7, 15, v1 // (op 41398) 
v_lshl_add_u32 v9, v8, 7, v7 // (op 41398) 


v_and_b32 v7, 63, v1 // (op 41398) 
v_lshrrev_b32 v8, 4, v7 // (op 41398) 

v_lshlrev_b32 v7, 5, v8 // (op 41398) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 41398) 


v_lshlrev_b32 v7, 2, v8 // (op 41398) 

v_lshrrev_b32 v8, 3, v7 // (op 41398) 

v_and_b32 v7, 4294967168, v1 // (op 41442) 
v_and_b32 v9, 15, v1 // (op 41442) 
v_add_lshl_u32 v10, v7, v9, 7 // (op 41442) 


v_and_b32 v7, 63, v1 // (op 41442) 
v_lshrrev_b32 v9, 4, v7 // (op 41442) 

v_lshlrev_b32 v7, 5, v9 // (op 41442) 

v_add_lshl_u32 v9, v10, v7, 2 // (op 41442) 


v_lshrrev_b32 v7, 3, v9 // (op 41442) 

v_lshrrev_b32 v9, 6, v1 // (op 41486) 
v_and_b32 v10, 1, v9 // (op 41486) 

v_and_b32 v9, 15, v1 // (op 41486) 
v_lshl_add_u32 v11, v10, 7, v9 // (op 41486) 


v_and_b32 v9, 63, v1 // (op 41486) 
v_lshrrev_b32 v10, 4, v9 // (op 41486) 

v_lshlrev_b32 v9, 5, v10 // (op 41486) 

v_lshl_add_u32 v10, v11, 7, v9 // (op 41486) 


v_lshlrev_b32 v9, 2, v10 // (op 41486) 

v_lshrrev_b32 v10, 3, v9 // (op 41486) 

v_and_b32 v9, 4294967168, v1 // (op 41081) 
v_and_b32 v11, 63, v1 // (op 41081) 
v_lshrrev_b32 v12, 4, v11 // (op 41081) 
v_lshlrev_b32 v13, 2, v12 // (op 41081) 

v_add_u32 v12, v9, v13 // (op 41081) 


v_lshl_add_u32 v9, s16, 8, v12 // (op 41081) 

v_lshrrev_b32 v12, 2, v11 // (op 41081) 

v_and_b32 v11, 3, v12 // (op 41081) 

v_and_b32 v12, 3, v1 // (op 41081) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 41081) 


v_lshrrev_b32 v11, 6, v1 // (op 41081) 
v_and_b32 v12, 1, v11 // (op 41081) 

v_lshl_add_u32 v11, v12, 7, v13 // (op 41081) 


v_lshl_add_u32 v12, s17, 8, v11 // (op 41081) 

v_mul_lo_u32 v11, v12, s2 // (op 41081) 

v_add_lshl_u32 v12, v9, v11, 2 // (op 41081) 


v_and_b32 v9, 4294967168, v1 // (op 41220) 
v_and_b32 v11, 63, v1 // (op 41220) 
v_lshrrev_b32 v13, 4, v11 // (op 41220) 
s_mul_i32 s1, s2, s9 // (op 41083) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 41083) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 41083) least significant
s_mul_hi_u32 s19, s2, s8 // (op 41083) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 41083) most significant: sum
s_add_u32 s1, s1, s19 // (op 41083) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 41083) 

v_lshlrev_b32 v14, 2, v13 // (op 41220) 

v_add_u32 v13, v9, v14 // (op 41220) 


v_lshl_add_u32 v9, s16, 8, v13 // (op 41220) 

v_lshrrev_b32 v13, 2, v11 // (op 41220) 

v_and_b32 v11, 3, v13 // (op 41220) 

v_and_b32 v13, 3, v1 // (op 41220) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 41220) 


v_lshrrev_b32 v11, 6, v1 // (op 41220) 
v_and_b32 v13, 1, v11 // (op 41220) 

v_lshl_add_u32 v11, v13, 7, v14 // (op 41220) 


v_lshl_add_u32 v13, s17, 8, v11 // (op 41220) 

v_mul_lo_u32 v11, v13, s4 // (op 41220) 

v_add_lshl_u32 v13, v9, v11, 2 // (op 41220) 


s_mov_b32 s40, s6 // (op 41083) reg expression
s_mov_b32 s41, s7 // (op 41083) reg expression
s_mov_b32 s42, s18 // (op 41083) reg expression
s_mov_b32 s43, 131072 // (op 41083) call()

s_mul_i32 s1, s4, s9 // (op 41222) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 41222) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 41222) least significant
s_mul_hi_u32 s7, s4, s8 // (op 41222) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 41222) most significant: sum
s_add_u32 s1, s1, s7 // (op 41222) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 41222) 

s_mov_b32 s44, s10 // (op 41222) reg expression
s_mov_b32 s45, s11 // (op 41222) reg expression
s_mov_b32 s46, s6 // (op 41222) reg expression
s_mov_b32 s47, 131072 // (op 41222) call()


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
v_accvgpr_write a32, 0.00000 // (op 2237) call()
v_accvgpr_write a33, 0.00000 // (op 2237) call()
v_accvgpr_write a34, 0.00000 // (op 2237) call()
v_accvgpr_write a35, 0.00000 // (op 2237) call()
v_accvgpr_write a36, 0.00000 // (op 3062) call()
v_accvgpr_write a37, 0.00000 // (op 3062) call()
v_accvgpr_write a38, 0.00000 // (op 3062) call()
v_accvgpr_write a39, 0.00000 // (op 3062) call()
v_accvgpr_write a40, 0.00000 // (op 3063) call()
v_accvgpr_write a41, 0.00000 // (op 3063) call()
v_accvgpr_write a42, 0.00000 // (op 3063) call()
v_accvgpr_write a43, 0.00000 // (op 3063) call()
v_accvgpr_write a44, 0.00000 // (op 3064) call()
v_accvgpr_write a45, 0.00000 // (op 3064) call()
v_accvgpr_write a46, 0.00000 // (op 3064) call()
v_accvgpr_write a47, 0.00000 // (op 3064) call()
v_accvgpr_write a48, 0.00000 // (op 3065) call()
v_accvgpr_write a49, 0.00000 // (op 3065) call()
v_accvgpr_write a50, 0.00000 // (op 3065) call()
v_accvgpr_write a51, 0.00000 // (op 3065) call()
v_accvgpr_write a52, 0.00000 // (op 3066) call()
v_accvgpr_write a53, 0.00000 // (op 3066) call()
v_accvgpr_write a54, 0.00000 // (op 3066) call()
v_accvgpr_write a55, 0.00000 // (op 3066) call()
v_accvgpr_write a56, 0.00000 // (op 3067) call()
v_accvgpr_write a57, 0.00000 // (op 3067) call()
v_accvgpr_write a58, 0.00000 // (op 3067) call()
v_accvgpr_write a59, 0.00000 // (op 3067) call()
v_accvgpr_write a60, 0.00000 // (op 3068) call()
v_accvgpr_write a61, 0.00000 // (op 3068) call()
v_accvgpr_write a62, 0.00000 // (op 3068) call()
v_accvgpr_write a63, 0.00000 // (op 3068) call()
v_accvgpr_write a64, 0.00000 // (op 4239) call()
v_accvgpr_write a65, 0.00000 // (op 4239) call()
v_accvgpr_write a66, 0.00000 // (op 4239) call()
v_accvgpr_write a67, 0.00000 // (op 4239) call()
v_accvgpr_write a68, 0.00000 // (op 5064) call()
v_accvgpr_write a69, 0.00000 // (op 5064) call()
v_accvgpr_write a70, 0.00000 // (op 5064) call()
v_accvgpr_write a71, 0.00000 // (op 5064) call()
v_accvgpr_write a72, 0.00000 // (op 5065) call()
v_accvgpr_write a73, 0.00000 // (op 5065) call()
v_accvgpr_write a74, 0.00000 // (op 5065) call()
v_accvgpr_write a75, 0.00000 // (op 5065) call()
v_accvgpr_write a76, 0.00000 // (op 5066) call()
v_accvgpr_write a77, 0.00000 // (op 5066) call()
v_accvgpr_write a78, 0.00000 // (op 5066) call()
v_accvgpr_write a79, 0.00000 // (op 5066) call()
v_accvgpr_write a80, 0.00000 // (op 5067) call()
v_accvgpr_write a81, 0.00000 // (op 5067) call()
v_accvgpr_write a82, 0.00000 // (op 5067) call()
v_accvgpr_write a83, 0.00000 // (op 5067) call()
v_accvgpr_write a84, 0.00000 // (op 5068) call()
v_accvgpr_write a85, 0.00000 // (op 5068) call()
v_accvgpr_write a86, 0.00000 // (op 5068) call()
v_accvgpr_write a87, 0.00000 // (op 5068) call()
v_accvgpr_write a88, 0.00000 // (op 5069) call()
v_accvgpr_write a89, 0.00000 // (op 5069) call()
v_accvgpr_write a90, 0.00000 // (op 5069) call()
v_accvgpr_write a91, 0.00000 // (op 5069) call()
v_accvgpr_write a92, 0.00000 // (op 5070) call()
v_accvgpr_write a93, 0.00000 // (op 5070) call()
v_accvgpr_write a94, 0.00000 // (op 5070) call()
v_accvgpr_write a95, 0.00000 // (op 5070) call()
v_accvgpr_write a96, 0.00000 // (op 6241) call()
v_accvgpr_write a97, 0.00000 // (op 6241) call()
v_accvgpr_write a98, 0.00000 // (op 6241) call()
v_accvgpr_write a99, 0.00000 // (op 6241) call()
v_accvgpr_write a100, 0.00000 // (op 7066) call()
v_accvgpr_write a101, 0.00000 // (op 7066) call()
v_accvgpr_write a102, 0.00000 // (op 7066) call()
v_accvgpr_write a103, 0.00000 // (op 7066) call()
v_accvgpr_write a104, 0.00000 // (op 7067) call()
v_accvgpr_write a105, 0.00000 // (op 7067) call()
v_accvgpr_write a106, 0.00000 // (op 7067) call()
v_accvgpr_write a107, 0.00000 // (op 7067) call()
v_accvgpr_write a108, 0.00000 // (op 7068) call()
v_accvgpr_write a109, 0.00000 // (op 7068) call()
v_accvgpr_write a110, 0.00000 // (op 7068) call()
v_accvgpr_write a111, 0.00000 // (op 7068) call()
v_accvgpr_write a112, 0.00000 // (op 7069) call()
v_accvgpr_write a113, 0.00000 // (op 7069) call()
v_accvgpr_write a114, 0.00000 // (op 7069) call()
v_accvgpr_write a115, 0.00000 // (op 7069) call()
v_accvgpr_write a116, 0.00000 // (op 7070) call()
v_accvgpr_write a117, 0.00000 // (op 7070) call()
v_accvgpr_write a118, 0.00000 // (op 7070) call()
v_accvgpr_write a119, 0.00000 // (op 7070) call()
v_accvgpr_write a120, 0.00000 // (op 7071) call()
v_accvgpr_write a121, 0.00000 // (op 7071) call()
v_accvgpr_write a122, 0.00000 // (op 7071) call()
v_accvgpr_write a123, 0.00000 // (op 7071) call()
v_accvgpr_write a124, 0.00000 // (op 7072) call()
v_accvgpr_write a125, 0.00000 // (op 7072) call()
v_accvgpr_write a126, 0.00000 // (op 7072) call()
v_accvgpr_write a127, 0.00000 // (op 7072) call()
v_accvgpr_write a128, 0.00000 // (op 8243) call()
v_accvgpr_write a129, 0.00000 // (op 8243) call()
v_accvgpr_write a130, 0.00000 // (op 8243) call()
v_accvgpr_write a131, 0.00000 // (op 8243) call()
v_accvgpr_write a132, 0.00000 // (op 9068) call()
v_accvgpr_write a133, 0.00000 // (op 9068) call()
v_accvgpr_write a134, 0.00000 // (op 9068) call()
v_accvgpr_write a135, 0.00000 // (op 9068) call()
v_accvgpr_write a136, 0.00000 // (op 9069) call()
v_accvgpr_write a137, 0.00000 // (op 9069) call()
v_accvgpr_write a138, 0.00000 // (op 9069) call()
v_accvgpr_write a139, 0.00000 // (op 9069) call()
v_accvgpr_write a140, 0.00000 // (op 9070) call()
v_accvgpr_write a141, 0.00000 // (op 9070) call()
v_accvgpr_write a142, 0.00000 // (op 9070) call()
v_accvgpr_write a143, 0.00000 // (op 9070) call()
v_accvgpr_write a144, 0.00000 // (op 9071) call()
v_accvgpr_write a145, 0.00000 // (op 9071) call()
v_accvgpr_write a146, 0.00000 // (op 9071) call()
v_accvgpr_write a147, 0.00000 // (op 9071) call()
v_accvgpr_write a148, 0.00000 // (op 9072) call()
v_accvgpr_write a149, 0.00000 // (op 9072) call()
v_accvgpr_write a150, 0.00000 // (op 9072) call()
v_accvgpr_write a151, 0.00000 // (op 9072) call()
v_accvgpr_write a152, 0.00000 // (op 9073) call()
v_accvgpr_write a153, 0.00000 // (op 9073) call()
v_accvgpr_write a154, 0.00000 // (op 9073) call()
v_accvgpr_write a155, 0.00000 // (op 9073) call()
v_accvgpr_write a156, 0.00000 // (op 9074) call()
v_accvgpr_write a157, 0.00000 // (op 9074) call()
v_accvgpr_write a158, 0.00000 // (op 9074) call()
v_accvgpr_write a159, 0.00000 // (op 9074) call()
v_accvgpr_write a160, 0.00000 // (op 10245) call()
v_accvgpr_write a161, 0.00000 // (op 10245) call()
v_accvgpr_write a162, 0.00000 // (op 10245) call()
v_accvgpr_write a163, 0.00000 // (op 10245) call()
v_accvgpr_write a164, 0.00000 // (op 11070) call()
v_accvgpr_write a165, 0.00000 // (op 11070) call()
v_accvgpr_write a166, 0.00000 // (op 11070) call()
v_accvgpr_write a167, 0.00000 // (op 11070) call()
v_accvgpr_write a168, 0.00000 // (op 11071) call()
v_accvgpr_write a169, 0.00000 // (op 11071) call()
v_accvgpr_write a170, 0.00000 // (op 11071) call()
v_accvgpr_write a171, 0.00000 // (op 11071) call()
v_accvgpr_write a172, 0.00000 // (op 11072) call()
v_accvgpr_write a173, 0.00000 // (op 11072) call()
v_accvgpr_write a174, 0.00000 // (op 11072) call()
v_accvgpr_write a175, 0.00000 // (op 11072) call()
v_accvgpr_write a176, 0.00000 // (op 11073) call()
v_accvgpr_write a177, 0.00000 // (op 11073) call()
v_accvgpr_write a178, 0.00000 // (op 11073) call()
v_accvgpr_write a179, 0.00000 // (op 11073) call()
v_accvgpr_write a180, 0.00000 // (op 11074) call()
v_accvgpr_write a181, 0.00000 // (op 11074) call()
v_accvgpr_write a182, 0.00000 // (op 11074) call()
v_accvgpr_write a183, 0.00000 // (op 11074) call()
v_accvgpr_write a184, 0.00000 // (op 11075) call()
v_accvgpr_write a185, 0.00000 // (op 11075) call()
v_accvgpr_write a186, 0.00000 // (op 11075) call()
v_accvgpr_write a187, 0.00000 // (op 11075) call()
v_accvgpr_write a188, 0.00000 // (op 11076) call()
v_accvgpr_write a189, 0.00000 // (op 11076) call()
v_accvgpr_write a190, 0.00000 // (op 11076) call()
v_accvgpr_write a191, 0.00000 // (op 11076) call()
v_accvgpr_write a192, 0.00000 // (op 12247) call()
v_accvgpr_write a193, 0.00000 // (op 12247) call()
v_accvgpr_write a194, 0.00000 // (op 12247) call()
v_accvgpr_write a195, 0.00000 // (op 12247) call()
v_accvgpr_write a196, 0.00000 // (op 13072) call()
v_accvgpr_write a197, 0.00000 // (op 13072) call()
v_accvgpr_write a198, 0.00000 // (op 13072) call()
v_accvgpr_write a199, 0.00000 // (op 13072) call()
v_accvgpr_write a200, 0.00000 // (op 13073) call()
v_accvgpr_write a201, 0.00000 // (op 13073) call()
v_accvgpr_write a202, 0.00000 // (op 13073) call()
v_accvgpr_write a203, 0.00000 // (op 13073) call()
v_accvgpr_write a204, 0.00000 // (op 13074) call()
v_accvgpr_write a205, 0.00000 // (op 13074) call()
v_accvgpr_write a206, 0.00000 // (op 13074) call()
v_accvgpr_write a207, 0.00000 // (op 13074) call()
v_accvgpr_write a208, 0.00000 // (op 13075) call()
v_accvgpr_write a209, 0.00000 // (op 13075) call()
v_accvgpr_write a210, 0.00000 // (op 13075) call()
v_accvgpr_write a211, 0.00000 // (op 13075) call()
v_accvgpr_write a212, 0.00000 // (op 13076) call()
v_accvgpr_write a213, 0.00000 // (op 13076) call()
v_accvgpr_write a214, 0.00000 // (op 13076) call()
v_accvgpr_write a215, 0.00000 // (op 13076) call()
v_accvgpr_write a216, 0.00000 // (op 13077) call()
v_accvgpr_write a217, 0.00000 // (op 13077) call()
v_accvgpr_write a218, 0.00000 // (op 13077) call()
v_accvgpr_write a219, 0.00000 // (op 13077) call()
v_accvgpr_write a220, 0.00000 // (op 13078) call()
v_accvgpr_write a221, 0.00000 // (op 13078) call()
v_accvgpr_write a222, 0.00000 // (op 13078) call()
v_accvgpr_write a223, 0.00000 // (op 13078) call()
v_accvgpr_write a224, 0.00000 // (op 14249) call()
v_accvgpr_write a225, 0.00000 // (op 14249) call()
v_accvgpr_write a226, 0.00000 // (op 14249) call()
v_accvgpr_write a227, 0.00000 // (op 14249) call()
v_accvgpr_write a228, 0.00000 // (op 15074) call()
v_accvgpr_write a229, 0.00000 // (op 15074) call()
v_accvgpr_write a230, 0.00000 // (op 15074) call()
v_accvgpr_write a231, 0.00000 // (op 15074) call()
v_accvgpr_write a232, 0.00000 // (op 15075) call()
v_accvgpr_write a233, 0.00000 // (op 15075) call()
v_accvgpr_write a234, 0.00000 // (op 15075) call()
v_accvgpr_write a235, 0.00000 // (op 15075) call()
v_accvgpr_write a236, 0.00000 // (op 15076) call()
v_accvgpr_write a237, 0.00000 // (op 15076) call()
v_accvgpr_write a238, 0.00000 // (op 15076) call()
v_accvgpr_write a239, 0.00000 // (op 15076) call()
v_accvgpr_write a240, 0.00000 // (op 15077) call()
v_accvgpr_write a241, 0.00000 // (op 15077) call()
v_accvgpr_write a242, 0.00000 // (op 15077) call()
v_accvgpr_write a243, 0.00000 // (op 15077) call()
v_accvgpr_write a244, 0.00000 // (op 15078) call()
v_accvgpr_write a245, 0.00000 // (op 15078) call()
v_accvgpr_write a246, 0.00000 // (op 15078) call()
v_accvgpr_write a247, 0.00000 // (op 15078) call()
v_accvgpr_write a248, 0.00000 // (op 15079) call()
v_accvgpr_write a249, 0.00000 // (op 15079) call()
v_accvgpr_write a250, 0.00000 // (op 15079) call()
v_accvgpr_write a251, 0.00000 // (op 15079) call()
v_accvgpr_write a252, 0.00000 // (op 15080) call()
v_accvgpr_write a253, 0.00000 // (op 15080) call()
v_accvgpr_write a254, 0.00000 // (op 15080) call()
v_accvgpr_write a255, 0.00000 // (op 15080) call()
v_lshrrev_b32 v14, 2, v1 // (op 40971) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 40971) 

v_and_b32 v14, -4, v1 // (op 40971) 
v_sub_u32 v16, v1, v14 // (op 40971) 

v_lshlrev_b32 v14, 5, v16 // (op 40971) 

v_mov_b32 v17, 0 // (op 40971) convert
v_mov_b32 v16, v14 // (op 40971) convert

v_mul_lo_u32 v19, v15, s13 // (op 40971) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 40971) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 40971) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 40971) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 40971) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 40971) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 40971) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 40971) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 40973) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 40973) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 40973) least significant
s_mul_hi_u32 s7, s12, s22 // (op 40973) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 40973) most significant: sum
s_add_u32 s1, s1, s7 // (op 40973) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 40973) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 40973) 

s_mov_b32 s48, s20 // (op 40973) reg expression
s_mov_b32 s49, s21 // (op 40973) reg expression
s_mov_b32 s50, s0 // (op 40973) reg expression
s_mov_b32 s51, 131072 // (op 40973) call()

v_lshrrev_b32 v14, 2, v1 // (op 40990) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 40990) 

v_and_b32 v14, -4, v1 // (op 40990) 
v_sub_u32 v18, v1, v14 // (op 40990) 

v_lshlrev_b32 v14, 5, v18 // (op 40990) 

v_mov_b32 v19, 0 // (op 40990) convert
v_mov_b32 v18, v14 // (op 40990) convert

v_mul_lo_u32 v21, v15, s13 // (op 40990) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 40990) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 40990) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 40990) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 40990) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 40990) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 40990) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 40990) 

s_mul_i32 s1, s12, s23 // (op 40992) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 40992) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 40992) least significant
s_mul_hi_u32 s7, s12, s22 // (op 40992) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 40992) most significant: sum
s_add_u32 s1, s1, s7 // (op 40992) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 40992) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 40992) 

s_mov_b32 s48, s20 // (op 40992) reg expression
s_mov_b32 s49, s21 // (op 40992) reg expression
s_mov_b32 s50, s0 // (op 40992) reg expression
s_mov_b32 s51, 131072 // (op 40992) call()

v_and_b32 v14, -4, v1 // (op 41035) 
v_sub_u32 v15, v1, v14 // (op 41035) 

v_lshlrev_b32 v14, 5, v15 // (op 41035) 

v_mov_b32 v21, 0 // (op 41035) convert
v_mov_b32 v20, v14 // (op 41035) convert

v_lshrrev_b32 v14, 2, v1 // (op 41035) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 41035) 

v_mul_lo_u32 v23, v15, s25 // (op 41035) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 41035) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 41035) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 41035) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 41035) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 41035) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 41035) 


v_lshrrev_b64 v[20:21], 3, v[14:15] // (op 41035) 

s_mul_i32 s1, s24, s9 // (op 41037) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 41037) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 41037) least significant
s_mul_hi_u32 s7, s24, s8 // (op 41037) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 41037) most significant: sum
s_add_u32 s1, s1, s7 // (op 41037) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 41037) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 41037) 

s_mov_b32 s52, s26 // (op 41037) reg expression
s_mov_b32 s53, s27 // (op 41037) reg expression
s_mov_b32 s54, s0 // (op 41037) reg expression
s_mov_b32 s55, 131072 // (op 41037) call()

v_and_b32 v14, -4, v1 // (op 41054) 
v_sub_u32 v15, v1, v14 // (op 41054) 

v_lshlrev_b32 v14, 5, v15 // (op 41054) 

v_mov_b32 v23, 0 // (op 41054) convert
v_mov_b32 v22, v14 // (op 41054) convert

v_lshrrev_b32 v14, 2, v1 // (op 41054) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 41054) 

v_mul_lo_u32 v25, v15, s25 // (op 41054) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 41054) most significant: high of low * low
v_mul_lo_u32 v24, v15, s24 // (op 41054) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 41054) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 41054) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 41054) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 41054) 


v_lshrrev_b64 v[22:23], 3, v[14:15] // (op 41054) 

s_mul_i32 s1, s24, s9 // (op 41056) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 41056) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 41056) least significant
s_mul_hi_u32 s7, s24, s8 // (op 41056) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 41056) most significant: sum
s_add_u32 s1, s1, s7 // (op 41056) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 41056) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 41056) 

s_mov_b32 s52, s26 // (op 41056) reg expression
s_mov_b32 s53, s27 // (op 41056) reg expression
s_mov_b32 s54, s0 // (op 41056) reg expression
s_mov_b32 s55, 131072 // (op 41056) call()

v_and_b32 v14, 63, v1 // (op 41508) 
v_lshrrev_b32 v15, 4, v14 // (op 41508) 
v_and_b32 v24, 3, v15 // (op 41508) 

v_and_b32 v15, 15, v1 // (op 41508) 
v_lshl_add_u32 v25, v24, 4, v15 // (op 41508) 


v_and_b32 v15, 4294967168, v1 // (op 41508) 
v_add_u32 v24, v15, v25 // (op 41508) 


v_lshl_add_u32 v15, s16, 8, v24 // (op 41508) 

v_lshrrev_b32 v24, 6, v14 // (op 41508) 

v_lshlrev_b32 v14, 2, v24 // (op 41508) 

v_mul_lo_u32 v24, v15, s28 // (op 41508) 
v_add_u32 v24, v24, v14 // (op 41508) 


s_mul_i32 s1, s28, s23 // (op 41510) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 41510) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 41510) least significant
s_mul_hi_u32 s7, s28, s22 // (op 41510) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 41510) most significant: sum
s_add_u32 s1, s1, s7 // (op 41510) most significant: sum


s_mov_b32 s56, s30 // (op 41510) reg expression
s_mov_b32 s57, s31 // (op 41510) reg expression
s_mov_b32 s58, s0 // (op 41510) reg expression
s_mov_b32 s59, 131072 // (op 41510) call()

v_and_b32 v14, 63, v1 // (op 41529) 
v_lshrrev_b32 v15, 4, v14 // (op 41529) 
v_and_b32 v25, 3, v15 // (op 41529) 

v_and_b32 v15, 15, v1 // (op 41529) 
v_lshl_add_u32 v26, v25, 4, v15 // (op 41529) 


v_and_b32 v15, 4294967168, v1 // (op 41529) 
v_add_u32 v25, v15, v26 // (op 41529) 


v_lshl_add_u32 v15, s16, 8, v25 // (op 41529) 

v_lshrrev_b32 v25, 6, v14 // (op 41529) 

v_lshlrev_b32 v14, 2, v25 // (op 41529) 

v_mul_lo_u32 v25, v15, s28 // (op 41529) 
v_add_u32 v25, v25, v14 // (op 41529) 


s_mul_i32 s1, s28, s23 // (op 41531) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 41531) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 41531) least significant
s_mul_hi_u32 s7, s28, s22 // (op 41531) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 41531) most significant: sum
s_add_u32 s1, s1, s7 // (op 41531) most significant: sum


s_mov_b32 s56, s30 // (op 41531) reg expression
s_mov_b32 s57, s31 // (op 41531) reg expression
s_mov_b32 s58, s0 // (op 41531) reg expression
s_mov_b32 s59, 131072 // (op 41531) call()

v_and_b32 v14, 63, v1 // (op 41560) 
v_lshrrev_b32 v15, 4, v14 // (op 41560) 
v_and_b32 v26, 3, v15 // (op 41560) 

v_and_b32 v15, 15, v1 // (op 41560) 
v_lshl_add_u32 v27, v26, 4, v15 // (op 41560) 


v_lshrrev_b32 v15, 6, v1 // (op 41560) 
v_and_b32 v26, 1, v15 // (op 41560) 

v_lshl_add_u32 v15, v26, 7, v27 // (op 41560) 


v_lshl_add_u32 v26, s17, 8, v15 // (op 41560) 

v_mul_lo_u32 v15, v26, s32 // (op 41560) 

v_lshrrev_b32 v26, 6, v14 // (op 41560) 

v_lshl_add_u32 v14, v26, 2, v15 // (op 41560) 


s_mul_i32 s1, s32, s9 // (op 41562) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 41562) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 41562) least significant
s_mul_hi_u32 s7, s32, s8 // (op 41562) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 41562) most significant: sum
s_add_u32 s1, s1, s7 // (op 41562) most significant: sum


s_mov_b32 s60, s34 // (op 41562) reg expression
s_mov_b32 s61, s35 // (op 41562) reg expression
s_mov_b32 s62, s0 // (op 41562) reg expression
s_mov_b32 s63, 131072 // (op 41562) call()

v_and_b32 v15, 63, v1 // (op 41581) 
v_lshrrev_b32 v26, 4, v15 // (op 41581) 
v_and_b32 v27, 3, v26 // (op 41581) 

v_and_b32 v26, 15, v1 // (op 41581) 
v_lshl_add_u32 v28, v27, 4, v26 // (op 41581) 


v_lshrrev_b32 v26, 6, v1 // (op 41581) 
v_and_b32 v27, 1, v26 // (op 41581) 

v_lshl_add_u32 v26, v27, 7, v28 // (op 41581) 


v_lshl_add_u32 v27, s17, 8, v26 // (op 41581) 

v_mul_lo_u32 v26, v27, s32 // (op 41581) 

v_lshrrev_b32 v27, 6, v15 // (op 41581) 

v_lshl_add_u32 v15, v27, 2, v26 // (op 41581) 


s_mul_i32 s1, s32, s9 // (op 41583) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 41583) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 41583) least significant
s_mul_hi_u32 s7, s32, s8 // (op 41583) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 41583) most significant: sum
s_add_u32 s1, s1, s7 // (op 41583) most significant: sum


s_mov_b32 s60, s34 // (op 41583) reg expression
s_mov_b32 s61, s35 // (op 41583) reg expression
s_mov_b32 s62, s0 // (op 41583) reg expression
s_mov_b32 s63, 131072 // (op 41583) call()

v_and_b32 v26, -4, v1 // (op 41182) 
v_sub_u32 v27, v1, v26 // (op 41182) 

v_lshlrev_b32 v26, 5, v27 // (op 41182) 

v_lshrrev_b32 v27, 2, v1 // (op 41182) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 41182) 


v_lshlrev_b32 v26, 2, v28 // (op 41182) 

v_lshrrev_b32 v27, 3, v26 // (op 41182) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 41182) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 40849) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 40849) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 40849) 


v_mov_b64 v[28:29], v[20:21] // (op 40849) 
s_mov_b32 m0, s0 // (op 40849) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 40849) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 40849) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 40849) most significant half

s_add_u32 m0, m0, 4096 // (op 40849) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 40849) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 40849) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 40849) most significant half

s_add_u32 m0, m0, 4096 // (op 40849) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 40849) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 40849) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 40849) most significant half

s_add_u32 m0, m0, 4096 // (op 40849) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 40849) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 41125) 
v_sub_u32 v27, v1, v26 // (op 41125) 

v_lshlrev_b32 v26, 5, v27 // (op 41125) 

v_lshrrev_b32 v27, 2, v1 // (op 41125) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 41125) 


v_lshlrev_b32 v26, 2, v28 // (op 41125) 

v_lshrrev_b32 v27, 3, v26 // (op 41125) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 41125) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 40854) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 40854) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 40854) 


v_mov_b64 v[28:29], v[16:17] // (op 40854) 
s_add_u32 s1, 16384, s0 // (op 40854) 
s_mov_b32 m0, s1 // (op 40854) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 40854) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 40854) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 40854) most significant half

s_add_u32 m0, m0, 4096 // (op 40854) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 40854) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 40854) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 40854) most significant half

s_add_u32 m0, m0, 4096 // (op 40854) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 40854) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 40854) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 40854) most significant half

s_add_u32 m0, m0, 4096 // (op 40854) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 40854) Load value direct to lds






v_mov_b32 v26, v14 // (op 74) reg expression

buffer_load_dwordx4 v[28:31], v26, s[60:63], 0 offen // (op 74) Load value



s_lshl_b32 s0, s32, 6 // (op 1100) 
v_add_u32 v26, v14, s0 // (op 1100) 


buffer_load_dwordx4 v[32:35], v26, s[60:63], 0 offen // (op 1100) Load value




s_lshl_b32 s0, s28, 6 // (op 8529) 
v_add_u32 v14, v24, s0 // (op 8529) 


buffer_load_dwordx4 v[36:39], v14, s[56:59], 0 offen // (op 8529) Load value



v_mov_b32 v14, v24 // (op 70) reg expression

buffer_load_dwordx4 v[40:43], v14, s[56:59], 0 offen // (op 70) Load value


v_and_b32 v14, -4, v1 // (op 41293) 
v_sub_u32 v24, v1, v14 // (op 41293) 

v_lshlrev_b32 v14, 5, v24 // (op 41293) 

v_lshrrev_b32 v24, 2, v1 // (op 41293) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41293) 


v_lshlrev_b32 v14, 2, v26 // (op 41293) 

v_lshrrev_b32 v24, 3, v14 // (op 41293) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 41293) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 40859) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 40859) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 40859) 


v_mov_b64 v[44:45], v[20:21] // (op 40859) 
s_add_u32 s1, 32768, s0 // (op 40859) 
s_mov_b32 m0, s1 // (op 40859) reg expression
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 40859) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 40859) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 40859) most significant half

s_add_u32 m0, m0, 4096 // (op 40859) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 40859) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 40859) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 40859) most significant half

s_add_u32 m0, m0, 4096 // (op 40859) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 40859) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 40859) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 40859) most significant half

s_add_u32 m0, m0, 4096 // (op 40859) 
buffer_load_dwordx4 v44, s[52:55], 64 offen offset: 0 lds // (op 40859) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 41249) 
v_sub_u32 v24, v1, v14 // (op 41249) 

v_lshlrev_b32 v14, 5, v24 // (op 41249) 

v_lshrrev_b32 v24, 2, v1 // (op 41249) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41249) 


v_lshlrev_b32 v14, 2, v26 // (op 41249) 

v_lshrrev_b32 v24, 3, v14 // (op 41249) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 41249) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 40861) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 40861) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 40861) 


v_mov_b64 v[44:45], v[16:17] // (op 40861) 
s_add_u32 s1, 49152, s0 // (op 40861) 
s_mov_b32 m0, s1 // (op 40861) reg expression
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 40861) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 40861) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 40861) most significant half

s_add_u32 m0, m0, 4096 // (op 40861) 
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 40861) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 40861) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 40861) most significant half

s_add_u32 m0, m0, 4096 // (op 40861) 
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 40861) Load value direct to lds
v_add_co_u32 v44, s[6:7], v44, v26 // (op 40861) least significant half
v_addc_co_u32 v45, s[6:7], 0, v45, s[6:7] // (op 40861) most significant half

s_add_u32 m0, m0, 4096 // (op 40861) 
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 40861) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 41381) 
v_sub_u32 v24, v1, v14 // (op 41381) 

v_lshlrev_b32 v14, 5, v24 // (op 41381) 

v_lshrrev_b32 v24, 2, v1 // (op 41381) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41381) 


v_lshlrev_b32 v14, 2, v26 // (op 41381) 

v_lshrrev_b32 v24, 3, v14 // (op 41381) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 41381) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 40863) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 40863) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 40863) 


v_mov_b64 v[44:45], v[20:21] // (op 40863) 
s_add_u32 s1, 65536, s0 // (op 40863) 
s_mov_b32 m0, s1 // (op 40863) reg expression
s_mov_b32 s6, 128 // (op 40863) call()
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 40863) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 40863) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 40863) most significant half

s_add_u32 m0, m0, 4096 // (op 40863) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 40863) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 40863) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 40863) most significant half

s_add_u32 m0, m0, 4096 // (op 40863) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 40863) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 40863) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 40863) most significant half

s_add_u32 m0, m0, 4096 // (op 40863) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 40863) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 41337) 
v_sub_u32 v24, v1, v14 // (op 41337) 

v_lshlrev_b32 v14, 5, v24 // (op 41337) 

v_lshrrev_b32 v24, 2, v1 // (op 41337) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41337) 


v_lshlrev_b32 v14, 2, v26 // (op 41337) 

v_lshrrev_b32 v24, 3, v14 // (op 41337) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 41337) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 40865) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 40865) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 40865) 


v_mov_b64 v[44:45], v[16:17] // (op 40865) 
s_add_u32 s1, 81920, s0 // (op 40865) 
s_mov_b32 m0, s1 // (op 40865) reg expression
s_mov_b32 s6, 128 // (op 40865) call()
buffer_load_dwordx4 v44, s[48:51], s6 offen offset: 0 lds // (op 40865) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 40865) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 40865) most significant half

s_add_u32 m0, m0, 4096 // (op 40865) 
buffer_load_dwordx4 v44, s[48:51], s6 offen offset: 0 lds // (op 40865) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 40865) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 40865) most significant half

s_add_u32 m0, m0, 4096 // (op 40865) 
buffer_load_dwordx4 v44, s[48:51], s6 offen offset: 0 lds // (op 40865) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 40865) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 40865) most significant half

s_add_u32 m0, m0, 4096 // (op 40865) 
buffer_load_dwordx4 v44, s[48:51], s6 offen offset: 0 lds // (op 40865) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 41469) 
v_sub_u32 v24, v1, v14 // (op 41469) 

v_lshlrev_b32 v14, 5, v24 // (op 41469) 

v_lshrrev_b32 v24, 2, v1 // (op 41469) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41469) 


v_lshlrev_b32 v14, 2, v26 // (op 41469) 

v_lshrrev_b32 v24, 3, v14 // (op 41469) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 41469) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 40867) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 40867) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 40867) 


v_mov_b64 v[44:45], v[20:21] // (op 40867) 
s_add_u32 s1, 98304, s0 // (op 40867) 
s_mov_b32 m0, s1 // (op 40867) reg expression
s_mov_b32 s6, 192 // (op 40867) call()
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 40867) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 40867) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 40867) most significant half

s_add_u32 m0, m0, 4096 // (op 40867) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 40867) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 40867) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 40867) most significant half

s_add_u32 m0, m0, 4096 // (op 40867) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 40867) Load value direct to lds
v_add_co_u32 v44, s[10:11], v44, v26 // (op 40867) least significant half
v_addc_co_u32 v45, s[10:11], 0, v45, s[10:11] // (op 40867) most significant half

s_add_u32 m0, m0, 4096 // (op 40867) 
buffer_load_dwordx4 v44, s[52:55], s6 offen offset: 0 lds // (op 40867) Load value direct to lds






v_and_b32 v14, -4, v1 // (op 41425) 
v_sub_u32 v24, v1, v14 // (op 41425) 

v_lshlrev_b32 v14, 5, v24 // (op 41425) 

v_lshrrev_b32 v24, 2, v1 // (op 41425) 
v_lshl_add_u32 v20, v24, 7, v14 // (op 41425) 


v_lshlrev_b32 v14, 2, v20 // (op 41425) 

v_lshrrev_b32 v24, 3, v14 // (op 41425) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 41425) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 40869) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 40869) 

v_lshrrev_b64 v[20:21], 3, s[10:11] // (op 40869) 


v_mov_b64 v[26:27], v[16:17] // (op 40869) 
s_add_u32 s1, 114688, s0 // (op 40869) 
s_mov_b32 m0, s1 // (op 40869) reg expression
s_mov_b32 s6, 192 // (op 40869) call()
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 40869) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v20 // (op 40869) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 40869) most significant half

s_add_u32 m0, m0, 4096 // (op 40869) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 40869) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v20 // (op 40869) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 40869) most significant half

s_add_u32 m0, m0, 4096 // (op 40869) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 40869) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v20 // (op 40869) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 40869) most significant half

s_add_u32 m0, m0, 4096 // (op 40869) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 40869) Load value direct to lds






s_waitcnt vmcnt(32)
s_barrier  // (op 42311) 



ds_read_b128 v[44:47], v4 // (op 39710) Load local data 



ds_read_b128 v[48:51], v4 offset:1024 // (op 39715) Load local data 



ds_read_b128 v[52:55], v4 offset:2048 // (op 39720) Load local data 



ds_read_b128 v[56:59], v4 offset:3072 // (op 39725) Load local data 



ds_read_b128 v[60:63], v4 offset:4096 // (op 39730) Load local data 



ds_read_b128 v[64:67], v4 offset:5120 // (op 39735) Load local data 



ds_read_b128 v[68:71], v4 offset:6144 // (op 39740) Load local data 



ds_read_b128 v[72:75], v4 offset:7168 // (op 39745) Load local data 
s_waitcnt vmcnt(28)
s_barrier  // (op 42294) 



ds_read_b128 v[76:79], v0 offset:16384 // (op 39670) Load local data 



ds_read_b128 v[80:83], v0 offset:17408 // (op 39675) Load local data 



ds_read_b128 v[84:87], v0 offset:18432 // (op 39680) Load local data 



ds_read_b128 v[88:91], v0 offset:19456 // (op 39685) Load local data 



ds_read_b128 v[92:95], v0 offset:20480 // (op 39690) Load local data 



ds_read_b128 v[96:99], v0 offset:21504 // (op 39695) Load local data 



ds_read_b128 v[100:103], v0 offset:22528 // (op 39700) Load local data 



ds_read_b128 v[104:107], v0 offset:23552 // (op 39705) Load local data 
v_and_b32 v14, -4, v1 // (op 41097) 
v_sub_u32 v24, v1, v14 // (op 41097) 

v_lshlrev_b32 v14, 5, v24 // (op 41097) 

v_lshrrev_b32 v24, 2, v1 // (op 41097) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41097) 


v_lshlrev_b32 v14, 2, v16 // (op 41097) 

v_lshrrev_b32 v24, 3, v14 // (op 41097) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 41097) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41154) 
v_sub_u32 v24, v1, v14 // (op 41154) 

v_lshlrev_b32 v14, 5, v24 // (op 41154) 

v_lshrrev_b32 v24, 2, v1 // (op 41154) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41154) 


v_lshlrev_b32 v14, 2, v16 // (op 41154) 

v_lshrrev_b32 v24, 3, v14 // (op 41154) 

s_nop 0
v_readfirstlane_b32 s1, v24 // (op 41154) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41234) 
v_sub_u32 v24, v1, v14 // (op 41234) 

v_lshlrev_b32 v14, 5, v24 // (op 41234) 

v_lshrrev_b32 v24, 2, v1 // (op 41234) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41234) 


v_lshlrev_b32 v14, 2, v16 // (op 41234) 

v_lshrrev_b32 v24, 3, v14 // (op 41234) 

s_nop 0
v_readfirstlane_b32 s6, v24 // (op 41234) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41278) 
v_sub_u32 v24, v1, v14 // (op 41278) 

v_lshlrev_b32 v14, 5, v24 // (op 41278) 

v_lshrrev_b32 v24, 2, v1 // (op 41278) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41278) 


v_lshlrev_b32 v14, 2, v16 // (op 41278) 

v_lshrrev_b32 v24, 3, v14 // (op 41278) 

s_nop 0
v_readfirstlane_b32 s7, v24 // (op 41278) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41322) 
v_sub_u32 v24, v1, v14 // (op 41322) 

v_lshlrev_b32 v14, 5, v24 // (op 41322) 

v_lshrrev_b32 v24, 2, v1 // (op 41322) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41322) 


v_lshlrev_b32 v14, 2, v16 // (op 41322) 

v_lshrrev_b32 v24, 3, v14 // (op 41322) 

s_nop 0
v_readfirstlane_b32 s10, v24 // (op 41322) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41366) 
v_sub_u32 v24, v1, v14 // (op 41366) 

v_lshlrev_b32 v14, 5, v24 // (op 41366) 

v_lshrrev_b32 v24, 2, v1 // (op 41366) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41366) 


v_lshlrev_b32 v14, 2, v16 // (op 41366) 

v_lshrrev_b32 v24, 3, v14 // (op 41366) 

s_nop 0
v_readfirstlane_b32 s11, v24 // (op 41366) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41410) 
v_sub_u32 v24, v1, v14 // (op 41410) 

v_lshlrev_b32 v14, 5, v24 // (op 41410) 

v_lshrrev_b32 v24, 2, v1 // (op 41410) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41410) 


v_lshlrev_b32 v14, 2, v16 // (op 41410) 

v_lshrrev_b32 v24, 3, v14 // (op 41410) 

s_nop 0
v_readfirstlane_b32 s14, v24 // (op 41410) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41454) 
v_sub_u32 v24, v1, v14 // (op 41454) 

v_lshlrev_b32 v14, 5, v24 // (op 41454) 

v_lshrrev_b32 v24, 2, v1 // (op 41454) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41454) 


v_lshlrev_b32 v14, 2, v16 // (op 41454) 

v_lshrrev_b32 v24, 3, v14 // (op 41454) 

s_nop 0
v_readfirstlane_b32 s15, v24 // (op 41454) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 12384) call()
s_ashr_i64 s[38:39], s[36:37], 7 // (op 12383) 
s_mov_b32 s19, s38 // (op 12383) convert

s_and_b32 s38, s19, 4294967292 // (op 12383) 

s_cmp_lt_u32 s18, s38 // (op 12383) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoop_12383 // (op 12383) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoop_12383 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoop_12383:
s_add_u32 s18, s18, 4 // (op 12385) 
s_waitcnt vmcnt(27)
v_mov_b32 v108, v28 // (op 40707) reg expression
v_mov_b32 v109, v29 // (op 40707) reg expression
v_mov_b32 v110, v30 // (op 40707) reg expression
v_mov_b32 v111, v31 // (op 40707) reg expression
s_waitcnt lgkmcnt(8)
s_barrier  // (op 42314) 
s_waitcnt vmcnt(26)
v_mov_b32 v112, v32 // (op 40720) reg expression

v_mov_b32 v113, v33 // (op 40720) reg expression
v_mov_b32 v114, v34 // (op 40720) reg expression
v_mov_b32 v115, v35 // (op 40720) reg expression
s_lshl_b64 s[38:39], s[24:25], 6 // (op 40802) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 40802) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 40802) 


v_mov_b64 v[20:21], v[22:23] // (op 40802) 
v_mov_b64 v[26:27], v[20:21] // (op 40802) 
s_mov_b32 m0, s1 // (op 40802) reg expression
s_mov_b32 s19, 256 // (op 40802) call()
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 40802) Load value direct to lds
v_add_co_u32 v26, s[38:39], v26, v16 // (op 40802) least significant half
v_addc_co_u32 v27, s[38:39], 0, v27, s[38:39] // (op 40802) most significant half

s_add_u32 m0, m0, 4096 // (op 40802) 
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 40802) Load value direct to lds
v_add_co_u32 v26, s[38:39], v26, v16 // (op 40802) least significant half
v_addc_co_u32 v27, s[38:39], 0, v27, s[38:39] // (op 40802) most significant half

s_add_u32 m0, m0, 4096 // (op 40802) 
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 40802) Load value direct to lds
v_add_co_u32 v26, s[38:39], v26, v16 // (op 40802) least significant half
v_addc_co_u32 v27, s[38:39], 0, v27, s[38:39] // (op 40802) most significant half

s_add_u32 m0, m0, 4096 // (op 40802) 
s_waitcnt vmcnt(28)
v_mov_b32 v116, v36 // (op 40733) reg expression
v_mov_b32 v117, v37 // (op 40733) reg expression
v_mov_b32 v118, v38 // (op 40733) reg expression
v_mov_b32 v119, v39 // (op 40733) reg expression
s_waitcnt vmcnt(27)
v_mov_b32 v120, v40 // (op 40746) reg expression
v_mov_b32 v121, v41 // (op 40746) reg expression
v_mov_b32 v122, v42 // (op 40746) reg expression
v_mov_b32 v123, v43 // (op 40746) reg expression
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 40802) Load value direct to lds



v_bfe_u32 v124, v120, 0, 8 // (op 37294) 
v_bfe_u32 v125, v120, 8, 8 // (op 37294) 
v_bfe_u32 v126, v120, 16, 8 // (op 37294) 
v_bfe_u32 v127, v120, 24, 8 // (op 37294) 
v_permlane16_swap_b32 v124, v125 // (op 37294) 
s_nop 0
v_permlane16_swap_b32 v126, v127 // (op 37294) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v124, v126 // (op 37294) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v125, v127 // (op 37294) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 42297) 

s_lshl_b64 s[38:39], s[12:13], 6 // (op 40797) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 40797) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 40797) 


v_mov_b64 v[26:27], v[18:19] // (op 40797) 
v_mov_b64 v[128:129], v[26:27] // (op 40797) 
s_add_u32 s19, 16384, s0 // (op 40797) 
s_mov_b32 m0, s19 // (op 40797) reg expression
s_mov_b32 s38, 256 // (op 40797) call()
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40797) Load value direct to lds
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40797) least significant half
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40797) most significant half

s_add_u32 m0, m0, 4096 // (op 40797) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40797) Load value direct to lds
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40797) least significant half
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40797) most significant half

s_add_u32 m0, m0, 4096 // (op 40797) 
v_bfe_u32 v130, v108, 0, 8 // (op 37683) 
v_bfe_u32 v131, v108, 8, 8 // (op 37683) 
v_bfe_u32 v132, v108, 16, 8 // (op 37683) 
v_bfe_u32 v133, v108, 24, 8 // (op 37683) 
v_permlane16_swap_b32 v130, v131 // (op 37683) 
s_nop 0
v_permlane16_swap_b32 v132, v133 // (op 37683) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v130, v132 // (op 37683) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v131, v133 // (op 37683) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40797) Load value direct to lds
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40797) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[44:47], a[0:3], v124, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 76) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40797) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[48:51], a[4:7], v124, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 417) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[52:55], a[8:11], v124, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 643) 
s_add_u32 m0, m0, 4096 // (op 40797) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[56:59], a[12:15], v124, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 869) 
v_bfe_u32 v134, v112, 0, 8 // (op 37751) 
v_bfe_u32 v135, v112, 8, 8 // (op 37751) 
v_bfe_u32 v136, v112, 16, 8 // (op 37751) 
v_bfe_u32 v137, v112, 24, 8 // (op 37751) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40797) Load value direct to lds




v_permlane16_swap_b32 v134, v135 // (op 37751) 


v_permlane16_swap_b32 v136, v137 // (op 37751) 
v_mov_b32 v14, v15 // (op 40715) 
v_add_u32 v24, 16, v14 // (op 40715) 

buffer_load_dwordx4 v[28:31], v24, s[60:63], 0 offen // (op 40715) Load value

v_permlane32_swap_b32 v134, v136 // (op 37751) 


v_permlane32_swap_b32 v135, v137 // (op 37751) 
s_lshl_b32 s19, s32, 6 // (op 40728) 
s_add_u32 s19, s19, 16 // (op 40728) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[60:63], a[16:19], v124, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1095) 
v_add_u32 v24, v14, s19 // (op 40728) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[64:67], a[20:23], v124, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1321) 
v_bfe_u32 v138, v116, 0, 8 // (op 37359) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[68:71], a[24:27], v124, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1547) 
v_bfe_u32 v139, v116, 8, 8 // (op 37359) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[72:75], a[28:31], v124, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1773) 
v_bfe_u32 v140, v116, 16, 8 // (op 37359) 
v_bfe_u32 v141, v116, 24, 8 // (op 37359) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[44:47], a[32:35], v125, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2993) 
v_permlane16_swap_b32 v138, v139 // (op 37359) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[48:51], a[36:39], v125, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2917) 
v_permlane16_swap_b32 v140, v141 // (op 37359) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[52:55], a[40:43], v125, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2841) 
s_nop 0
v_permlane32_swap_b32 v138, v140 // (op 37359) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[56:59], a[44:47], v125, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2765) 
v_permlane32_swap_b32 v139, v141 // (op 37359) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[60:63], a[48:51], v125, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2689) 
s_waitcnt vmcnt(29)
s_barrier  // (op 42334) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[64:67], a[52:55], v125, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2613) 
ds_read_b128 v[142:145], v6 offset:32768 // (op 212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[68:71], a[56:59], v125, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2537) 
ds_read_b128 v[146:149], v6 offset:33792 // (op 428) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[72:75], a[60:63], v125, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2461) 
ds_read_b128 v[150:153], v6 offset:34816 // (op 654) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[44:47], a[64:67], v126, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4995) 
ds_read_b128 v[154:157], v6 offset:35840 // (op 880) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[48:51], a[68:71], v126, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4919) 
ds_read_b128 v[158:161], v6 offset:36864 // (op 1106) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[52:55], a[72:75], v126, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4843) 
ds_read_b128 v[162:165], v6 offset:37888 // (op 1332) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[56:59], a[76:79], v126, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4767) 
ds_read_b128 v[166:169], v6 offset:38912 // (op 1558) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[84:87], v[60:63], a[80:83], v126, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4691) 
ds_read_b128 v[80:83], v6 offset:39936 // (op 1784) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[84:87], v[64:67], a[84:87], v126, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4615) 
buffer_load_dwordx4 v[32:35], v24, s[60:63], 0 offen // (op 40728) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:87], v[68:71], a[88:91], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4539) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[84:87], v[72:75], a[92:95], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4463) 
v_add_u32 v15, 16, v15 // (op 41571) 
v_mov_b32 v14, v25 // (op 40741) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[44:47], a[96:99], v127, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6997) 
s_lshl_b32 s19, s28, 6 // (op 40741) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[48:51], a[100:103], v127, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6921) 
s_add_u32 s38, 16, s19 // (op 40741) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:91], v[52:55], a[104:107], v127, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6845) 
v_add_u32 v24, v14, s38 // (op 40741) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:91], v[56:59], a[108:111], v127, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6769) 
buffer_load_dwordx4 v[36:39], v24, s[56:59], 0 offen // (op 40741) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[88:91], v[60:63], a[112:115], v127, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6693) 
v_add_u32 v24, 16, v14 // (op 40754) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[88:91], v[64:67], a[116:119], v127, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6617) 
buffer_load_dwordx4 v[40:43], v24, s[56:59], 0 offen // (op 40754) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[88:91], v[68:71], a[120:123], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6541) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[88:91], v[72:75], a[124:127], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6465) 
v_add_u32 v25, 16, v25 // (op 41519) 

s_waitcnt vmcnt(28)
s_barrier  // (op 42327) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[92:95], v[44:47], a[128:131], v138, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8999) 
ds_read_b128 v[76:79], v3 offset:49152 // (op 210) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[92:95], v[48:51], a[132:135], v138, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8923) 
ds_read_b128 v[84:87], v3 offset:50176 // (op 2471) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[92:95], v[52:55], a[136:139], v138, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8847) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[92:95], v[56:59], a[140:143], v138, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8771) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:95], v[60:63], a[144:147], v138, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8695) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[92:95], v[64:67], a[148:151], v138, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8619) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[68:71], a[152:155], v138, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8543) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[72:75], a[156:159], v138, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8467) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:99], v[44:47], a[160:163], v139, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11001) 
ds_read_b128 v[88:91], v3 offset:51200 // (op 4473) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[96:99], v[48:51], a[164:167], v139, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10925) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[96:99], v[52:55], a[168:171], v139, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10849) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:99], v[56:59], a[172:175], v139, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10773) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[96:99], v[60:63], a[176:179], v139, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10697) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[96:99], v[64:67], a[180:183], v139, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[96:99], v[68:71], a[184:187], v139, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10545) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[96:99], v[72:75], a[188:191], v139, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10469) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[100:103], v[44:47], a[192:195], v140, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13003) 
ds_read_b128 v[92:95], v3 offset:52224 // (op 6475) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[100:103], v[48:51], a[196:199], v140, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12927) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[100:103], v[52:55], a[200:203], v140, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12851) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[100:103], v[56:59], a[204:207], v140, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12775) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[100:103], v[60:63], a[208:211], v140, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12699) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[100:103], v[64:67], a[212:215], v140, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12623) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[100:103], v[68:71], a[216:219], v140, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12547) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[100:103], v[72:75], a[220:223], v140, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12471) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[104:107], v[44:47], a[224:227], v141, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15005) 
ds_read_b128 v[96:99], v3 offset:53248 // (op 8477) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[104:107], v[48:51], a[228:231], v141, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14929) 
ds_read_b128 v[100:103], v3 offset:54272 // (op 10479) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[104:107], v[52:55], a[232:235], v141, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14853) 
ds_read_b128 v[124:127], v3 offset:55296 // (op 12481) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[104:107], v[56:59], a[236:239], v141, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14777) 
ds_read_b128 v[170:173], v3 offset:56320 // (op 14483) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[104:107], v[60:63], a[240:243], v141, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14701) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[104:107], v[64:67], a[244:247], v141, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14625) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[104:107], v[68:71], a[248:251], v141, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14549) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[104:107], v[72:75], a[252:255], v141, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14473) 


v_bfe_u32 v128, v121, 0, 8 // (op 37309) 
v_bfe_u32 v129, v121, 8, 8 // (op 37309) 
v_bfe_u32 v130, v121, 16, 8 // (op 37309) 
v_bfe_u32 v131, v121, 24, 8 // (op 37309) 
v_permlane16_swap_b32 v128, v129 // (op 37309) 
s_nop 0
v_permlane16_swap_b32 v130, v131 // (op 37309) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v128, v130 // (op 37309) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v129, v131 // (op 37309) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 42338) 

s_lshl_b64 s[38:39], s[24:25], 6 // (op 40824) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 40824) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 40824) 


v_mov_b64 v[132:133], v[20:21] // (op 40824) 
s_add_u32 s19, 32768, s7 // (op 40824) 
s_mov_b32 m0, s19 // (op 40824) reg expression
s_mov_b32 s38, 320 // (op 40824) call()
v_bfe_u32 v134, v109, 0, 8 // (op 37698) 
v_bfe_u32 v135, v109, 8, 8 // (op 37698) 
v_bfe_u32 v136, v109, 16, 8 // (op 37698) 
v_bfe_u32 v137, v109, 24, 8 // (op 37698) 
v_permlane16_swap_b32 v134, v135 // (op 37698) 
s_nop 0
v_permlane16_swap_b32 v136, v137 // (op 37698) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v134, v136 // (op 37698) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v135, v137 // (op 37698) 
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 40824) Load value direct to lds
v_add_co_u32 v132, s[64:65], v132, v16 // (op 40824) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[142:145], a[0:3], v128, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 217) 
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 40824) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[146:149], a[4:7], v128, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 418) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[150:153], a[8:11], v128, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 644) 
s_add_u32 m0, m0, 4096 // (op 40824) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[154:157], a[12:15], v128, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 870) 
v_bfe_u32 v138, v113, 0, 8 // (op 37766) 
v_bfe_u32 v139, v113, 8, 8 // (op 37766) 
v_bfe_u32 v140, v113, 16, 8 // (op 37766) 
v_bfe_u32 v141, v113, 24, 8 // (op 37766) 
v_permlane16_swap_b32 v138, v139 // (op 37766) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 37766) Wait state hazard: VALU Write Readlane Hazard
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 40824) Load value direct to lds
v_add_co_u32 v132, s[64:65], v132, v16 // (op 40824) least significant half
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 40824) most significant half

s_add_u32 m0, m0, 4096 // (op 40824) 
v_permlane32_swap_b32 v138, v140 // (op 37766) 
v_permlane32_swap_b32 v139, v141 // (op 37766) 
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 40824) Load value direct to lds
v_add_co_u32 v132, s[64:65], v132, v16 // (op 40824) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[158:161], a[16:19], v128, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1096) 
v_addc_co_u32 v133, s[64:65], 0, v133, s[64:65] // (op 40824) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[162:165], a[20:23], v128, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1322) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[166:169], a[24:27], v128, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1548) 
s_add_u32 m0, m0, 4096 // (op 40824) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[80:83], a[28:31], v128, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1774) 
v_bfe_u32 v174, v117, 0, 8 // (op 37374) 
v_bfe_u32 v175, v117, 8, 8 // (op 37374) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:87], v[142:145], a[32:35], v129, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2994) 
v_bfe_u32 v176, v117, 16, 8 // (op 37374) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:87], v[146:149], a[36:39], v129, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2918) 
v_bfe_u32 v177, v117, 24, 8 // (op 37374) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:87], v[150:153], a[40:43], v129, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2842) 
v_permlane16_swap_b32 v174, v175 // (op 37374) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:87], v[154:157], a[44:47], v129, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2766) 
v_permlane16_swap_b32 v176, v177 // (op 37374) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[158:161], a[48:51], v129, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2690) 
s_nop 0
v_permlane32_swap_b32 v174, v176 // (op 37374) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[162:165], a[52:55], v129, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2614) 
v_permlane32_swap_b32 v175, v177 // (op 37374) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[166:169], a[56:59], v129, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2538) 
s_waitcnt vmcnt(27)
s_barrier  // (op 42349) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[80:83], a[60:63], v129, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2462) 
v_add_u32 v14, 65536, v8 // (op 240) 
ds_read_b128 v[44:47], v14 // (op 240) Load local data 




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[142:145], a[64:67], v130, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4996) 
v_add_u32 v14, 66560, v8 // (op 438) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[146:149], a[68:71], v130, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4920) 
ds_read_b128 v[48:51], v14 // (op 438) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:91], v[150:153], a[72:75], v130, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4844) 
v_add_u32 v14, 67584, v8 // (op 664) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[88:91], v[154:157], a[76:79], v130, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4768) 
ds_read_b128 v[52:55], v14 // (op 664) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[88:91], v[158:161], a[80:83], v130, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4692) 
v_add_u32 v14, 68608, v8 // (op 890) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[88:91], v[162:165], a[84:87], v130, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4616) 
ds_read_b128 v[56:59], v14 // (op 890) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[88:91], v[166:169], a[88:91], v130, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4540) 
v_add_u32 v14, 69632, v8 // (op 1116) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[88:91], v[80:83], a[92:95], v130, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4464) 
ds_read_b128 v[60:63], v14 // (op 1116) Load local data 




s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[92:95], v[142:145], a[96:99], v131, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6998) 
v_add_u32 v14, 70656, v8 // (op 1342) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[92:95], v[146:149], a[100:103], v131, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6922) 
ds_read_b128 v[64:67], v14 // (op 1342) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[92:95], v[150:153], a[104:107], v131, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6846) 
v_add_u32 v14, 71680, v8 // (op 1568) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:95], v[154:157], a[108:111], v131, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6770) 
ds_read_b128 v[68:71], v14 // (op 1568) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:95], v[158:161], a[112:115], v131, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6694) 
v_add_u32 v14, 72704, v8 // (op 1794) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:95], v[162:165], a[116:119], v131, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6618) 
ds_read_b128 v[72:75], v14 // (op 1794) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[92:95], v[166:169], a[120:123], v131, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6542) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[92:95], v[80:83], a[124:127], v131, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6466) 
s_waitcnt vmcnt(23)
s_barrier  // (op 42342) 




s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[142:145], a[128:131], v174, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9000) 
v_add_u32 v14, 81920, v5 // (op 238) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[146:149], a[132:135], v174, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8924) 
ds_read_b128 v[76:79], v14 // (op 238) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[150:153], a[136:139], v174, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8848) 
v_add_u32 v14, 82944, v5 // (op 2481) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[154:157], a[140:143], v174, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8772) 
ds_read_b128 v[84:87], v14 // (op 2481) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:99], v[158:161], a[144:147], v174, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8696) 
buffer_load_dwordx4 v132, s[52:55], s38 offen offset: 0 lds // (op 40824) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[96:99], v[162:165], a[148:151], v174, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8620) 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 42331) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[96:99], v[166:169], a[152:155], v174, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8544) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 40819) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[96:99], v[80:83], a[156:159], v174, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8468) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 40819) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 40819) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[100:103], v[142:145], a[160:163], v175, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11002) 
v_add_u32 v14, 83968, v5 // (op 4483) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[100:103], v[146:149], a[164:167], v175, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10926) 
ds_read_b128 v[88:91], v14 // (op 4483) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[100:103], v[150:153], a[168:171], v175, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10850) 
v_mov_b64 v[128:129], v[26:27] // (op 40819) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[100:103], v[154:157], a[172:175], v175, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10774) 
s_add_u32 s19, 49152, s6 // (op 40819) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[158:161], a[176:179], v175, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10698) 
s_mov_b32 m0, s19 // (op 40819) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[162:165], a[180:183], v175, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10622) 
s_mov_b32 s38, 320 // (op 40819) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[166:169], a[184:187], v175, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10546) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40819) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[80:83], a[188:191], v175, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10470) 
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40819) least significant half
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40819) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:127], v[142:145], a[192:195], v176, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13004) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[124:127], v[146:149], a[196:199], v176, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12928) 
s_add_u32 m0, m0, 4096 // (op 40819) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[124:127], v[150:153], a[200:203], v176, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12852) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[124:127], v[154:157], a[204:207], v176, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12776) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40819) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:127], v[158:161], a[208:211], v176, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12700) 
v_add_u32 v14, 84992, v5 // (op 6485) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[124:127], v[162:165], a[212:215], v176, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12624) 
ds_read_b128 v[92:95], v14 // (op 6485) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[124:127], v[166:169], a[216:219], v176, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12548) 
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40819) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[124:127], v[80:83], a[220:223], v176, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12472) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40819) most significant half





v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[170:173], v[142:145], a[224:227], v177, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15006) 
s_add_u32 m0, m0, 4096 // (op 40819) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[170:173], v[146:149], a[228:231], v177, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14930) 
v_add_u32 v14, 86016, v5 // (op 8487) 

ds_read_b128 v[124:127], v14 // (op 8487) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[170:173], v[150:153], a[232:235], v177, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14854) 
v_add_u32 v14, 87040, v5 // (op 10489) 

ds_read_b128 v[130:133], v14 // (op 10489) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[170:173], v[154:157], a[236:239], v177, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14778) 
v_add_u32 v14, 88064, v5 // (op 12491) 


ds_read_b128 v[134:137], v14 // (op 12491) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[170:173], v[158:161], a[240:243], v177, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14702) 
v_add_u32 v14, 89088, v5 // (op 14493) 

ds_read_b128 v[142:145], v14 // (op 14493) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[170:173], v[162:165], a[244:247], v177, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14626) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40819) Load value direct to lds

v_add_co_u32 v128, s[64:65], v128, v16 // (op 40819) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[170:173], v[166:169], a[248:251], v177, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14550) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40819) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[170:173], v[80:83], a[252:255], v177, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14474) 
s_add_u32 m0, m0, 4096 // (op 40819) 



buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40819) Load value direct to lds




v_bfe_u32 v138, v122, 0, 8 // (op 37326) 
v_bfe_u32 v139, v122, 8, 8 // (op 37326) 
v_bfe_u32 v140, v122, 16, 8 // (op 37326) 
v_bfe_u32 v141, v122, 24, 8 // (op 37326) 
v_permlane16_swap_b32 v138, v139 // (op 37326) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 37326) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 37326) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 37326) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 42352) 

s_lshl_b64 s[38:39], s[24:25], 6 // (op 40834) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 40834) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 40834) 


v_mov_b64 v[128:129], v[20:21] // (op 40834) 
s_add_u32 s19, 65536, s11 // (op 40834) 
s_mov_b32 m0, s19 // (op 40834) reg expression
s_mov_b32 s38, 384 // (op 40834) call()
v_bfe_u32 v146, v110, 0, 8 // (op 37716) 
v_bfe_u32 v147, v110, 8, 8 // (op 37716) 
v_bfe_u32 v148, v110, 16, 8 // (op 37716) 
v_bfe_u32 v149, v110, 24, 8 // (op 37716) 
buffer_load_dwordx4 v128, s[52:55], s38 offen offset: 0 lds // (op 40834) Load value direct to lds
v_permlane16_swap_b32 v146, v147 // (op 37716) 
v_permlane16_swap_b32 v148, v149 // (op 37716) 
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40834) least significant half
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40834) most significant half

s_add_u32 m0, m0, 4096 // (op 40834) 
v_permlane32_swap_b32 v146, v148 // (op 37716) 
v_permlane32_swap_b32 v147, v149 // (op 37716) 
buffer_load_dwordx4 v128, s[52:55], s38 offen offset: 0 lds // (op 40834) Load value direct to lds
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40834) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[44:47], a[0:3], v138, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 245) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40834) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[48:51], a[4:7], v138, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 419) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[52:55], a[8:11], v138, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 645) 
s_add_u32 m0, m0, 4096 // (op 40834) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[56:59], a[12:15], v138, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 871) 
v_bfe_u32 v150, v114, 0, 8 // (op 37784) 
v_bfe_u32 v151, v114, 8, 8 // (op 37784) 
v_bfe_u32 v152, v114, 16, 8 // (op 37784) 
v_bfe_u32 v153, v114, 24, 8 // (op 37784) 
v_permlane16_swap_b32 v150, v151 // (op 37784) 
s_nop 0
v_permlane16_swap_b32 v152, v153 // (op 37784) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v150, v152 // (op 37784) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v151, v153 // (op 37784) 
buffer_load_dwordx4 v128, s[52:55], s38 offen offset: 0 lds // (op 40834) Load value direct to lds
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40834) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[60:63], a[16:19], v138, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1097) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40834) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[64:67], a[20:23], v138, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1323) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[68:71], a[24:27], v138, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1549) 
s_add_u32 m0, m0, 4096 // (op 40834) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[72:75], a[28:31], v138, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1775) 
v_bfe_u32 v154, v118, 0, 8 // (op 37392) 
v_bfe_u32 v155, v118, 8, 8 // (op 37392) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:87], v[44:47], a[32:35], v139, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2995) 
v_bfe_u32 v156, v118, 16, 8 // (op 37392) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:87], v[48:51], a[36:39], v139, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2919) 
buffer_load_dwordx4 v128, s[52:55], s38 offen offset: 0 lds // (op 40834) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:87], v[52:55], a[40:43], v139, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2843) 
v_bfe_u32 v157, v118, 24, 8 // (op 37392) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:87], v[56:59], a[44:47], v139, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2767) 
v_permlane16_swap_b32 v154, v155 // (op 37392) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[60:63], a[48:51], v139, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2691) 
v_permlane16_swap_b32 v156, v157 // (op 37392) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[64:67], a[52:55], v139, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2615) 
s_nop 0
v_permlane32_swap_b32 v154, v156 // (op 37392) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[68:71], a[56:59], v139, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2539) 
v_permlane32_swap_b32 v155, v157 // (op 37392) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[72:75], a[60:63], v139, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2463) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 42346) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[44:47], a[64:67], v140, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4997) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 40829) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[48:51], a[68:71], v140, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4921) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 40829) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:91], v[52:55], a[72:75], v140, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4845) 
v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 40829) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[88:91], v[56:59], a[76:79], v140, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4769) 
v_mov_b64 v[128:129], v[26:27] // (op 40829) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[88:91], v[60:63], a[80:83], v140, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4693) 
s_add_u32 s19, 81920, s10 // (op 40829) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[88:91], v[64:67], a[84:87], v140, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4617) 
s_mov_b32 m0, s19 // (op 40829) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[88:91], v[68:71], a[88:91], v140, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4541) 
s_mov_b32 s38, 384 // (op 40829) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[88:91], v[72:75], a[92:95], v140, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4465) 
s_waitcnt vmcnt(28)
s_barrier  // (op 42362) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[92:95], v[44:47], a[96:99], v141, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6999) 
v_add_u32 v14, 98304, v10 // (op 268) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[92:95], v[48:51], a[100:103], v141, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6923) 
ds_read_b128 v[96:99], v14 // (op 268) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[92:95], v[52:55], a[104:107], v141, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6847) 
v_add_u32 v14, 99328, v10 // (op 448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:95], v[56:59], a[108:111], v141, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6771) 
ds_read_b128 v[100:103], v14 // (op 448) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:95], v[60:63], a[112:115], v141, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6695) 
v_add_u32 v14, 100352, v10 // (op 674) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:95], v[64:67], a[116:119], v141, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6619) 
ds_read_b128 v[104:107], v14 // (op 674) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[92:95], v[68:71], a[120:123], v141, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6543) 
v_add_u32 v14, 101376, v10 // (op 900) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[92:95], v[72:75], a[124:127], v141, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6467) 
ds_read_b128 v[158:161], v14 // (op 900) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[44:47], a[128:131], v154, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9001) 
v_add_u32 v14, 102400, v10 // (op 1126) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[48:51], a[132:135], v154, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8925) 
ds_read_b128 v[138:141], v14 // (op 1126) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[52:55], a[136:139], v154, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8849) 
v_add_u32 v14, 103424, v10 // (op 1352) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[124:127], v[56:59], a[140:143], v154, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8773) 
ds_read_b128 v[162:165], v14 // (op 1352) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:127], v[60:63], a[144:147], v154, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8697) 
v_add_u32 v14, 104448, v10 // (op 1578) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[124:127], v[64:67], a[148:151], v154, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8621) 
ds_read_b128 v[166:169], v14 // (op 1578) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[68:71], a[152:155], v154, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8545) 
v_add_u32 v14, 105472, v10 // (op 1804) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[124:127], v[72:75], a[156:159], v154, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8469) 
ds_read_b128 v[170:173], v14 // (op 1804) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[130:133], v[44:47], a[160:163], v155, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11003) 
s_waitcnt vmcnt(24)
s_barrier  // (op 42355) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[130:133], v[48:51], a[164:167], v155, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10927) 
v_add_u32 v14, 114688, v7 // (op 266) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[130:133], v[52:55], a[168:171], v155, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10851) 
ds_read_b128 v[76:79], v14 // (op 266) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[130:133], v[56:59], a[172:175], v155, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10775) 
v_add_u32 v14, 115712, v7 // (op 2491) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[130:133], v[60:63], a[176:179], v155, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10699) 
ds_read_b128 v[80:83], v14 // (op 2491) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:133], v[64:67], a[180:183], v155, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10623) 
v_add_u32 v14, 116736, v7 // (op 4493) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[130:133], v[68:71], a[184:187], v155, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10547) 
ds_read_b128 v[84:87], v14 // (op 4493) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[130:133], v[72:75], a[188:191], v155, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10471) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40829) Load value direct to lds

v_add_co_u32 v128, s[64:65], v128, v16 // (op 40829) least significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[134:137], v[44:47], a[192:195], v156, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13005) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40829) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[134:137], v[48:51], a[196:199], v156, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12929) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[134:137], v[52:55], a[200:203], v156, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12853) 
s_add_u32 m0, m0, 4096 // (op 40829) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[134:137], v[56:59], a[204:207], v156, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12777) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[60:63], a[208:211], v156, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12701) 
v_add_u32 v14, 117760, v7 // (op 6495) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[64:67], a[212:215], v156, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12625) 
ds_read_b128 v[88:91], v14 // (op 6495) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[68:71], a[216:219], v156, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12549) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40829) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[72:75], a[220:223], v156, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12473) 
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40829) least significant half

v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40829) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[44:47], a[224:227], v157, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15007) 
s_add_u32 m0, m0, 4096 // (op 40829) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[48:51], a[228:231], v157, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14931) 
v_add_u32 v14, 118784, v7 // (op 8497) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[52:55], a[232:235], v157, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14855) 
ds_read_b128 v[92:95], v14 // (op 8497) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[56:59], a[236:239], v157, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14779) 
v_add_u32 v14, 119808, v7 // (op 10499) 

ds_read_b128 v[124:127], v14 // (op 10499) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[142:145], v[60:63], a[240:243], v157, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14703) 
v_add_u32 v14, 120832, v7 // (op 12501) 
buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40829) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[142:145], v[64:67], a[244:247], v157, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14627) 
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40829) least significant half
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40829) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[142:145], v[68:71], a[248:251], v157, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14551) 
s_add_u32 m0, m0, 4096 // (op 40829) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[142:145], v[72:75], a[252:255], v157, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14475) 



ds_read_b128 v[130:133], v14 // (op 12501) Load local data 

buffer_load_dwordx4 v128, s[48:51], s38 offen offset: 0 lds // (op 40829) Load value direct to lds







v_add_u32 v14, 121856, v7 // (op 14503) 
ds_read_b128 v[134:137], v14 // (op 14503) Load local data 

v_bfe_u32 v142, v123, 0, 8 // (op 37343) 
v_bfe_u32 v143, v123, 8, 8 // (op 37343) 
v_bfe_u32 v144, v123, 16, 8 // (op 37343) 
v_bfe_u32 v145, v123, 24, 8 // (op 37343) 
v_permlane16_swap_b32 v142, v143 // (op 37343) 
s_nop 0
v_permlane16_swap_b32 v144, v145 // (op 37343) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v142, v144 // (op 37343) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v143, v145 // (op 37343) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 42365) 


s_lshl_b64 s[38:39], s[24:25], 6 // (op 40844) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 40844) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 40844) 


v_mov_b64 v[128:129], v[20:21] // (op 40844) 
s_add_u32 s19, 98304, s15 // (op 40844) 
s_mov_b32 m0, s19 // (op 40844) reg expression
s_mov_b32 s38, 448 // (op 40844) call()
v_bfe_u32 v120, v111, 0, 8 // (op 37734) 
v_bfe_u32 v121, v111, 8, 8 // (op 37734) 
v_bfe_u32 v122, v111, 16, 8 // (op 37734) 
v_bfe_u32 v123, v111, 24, 8 // (op 37734) 
v_permlane16_swap_b32 v120, v121 // (op 37734) 
s_nop 0
v_permlane16_swap_b32 v122, v123 // (op 37734) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v120, v122 // (op 37734) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v121, v123 // (op 37734) 
buffer_load_dwordx4 v128, s[52:55], s38 offen offset: 0 lds // (op 40844) Load value direct to lds

v_add_co_u32 v128, s[64:65], v128, v16 // (op 40844) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[96:99], a[0:3], v142, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 273) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40844) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[100:103], a[4:7], v142, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 420) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[104:107], a[8:11], v142, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 646) 
s_add_u32 m0, m0, 4096 // (op 40844) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[158:161], a[12:15], v142, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 872) 
v_bfe_u32 v108, v115, 0, 8 // (op 37802) 
v_bfe_u32 v109, v115, 8, 8 // (op 37802) 
v_bfe_u32 v110, v115, 16, 8 // (op 37802) 
v_bfe_u32 v111, v115, 24, 8 // (op 37802) 
v_permlane16_swap_b32 v108, v109 // (op 37802) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 37802) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 37802) Wait state hazard: VALU Write Readlane Hazard
buffer_load_dwordx4 v128, s[52:55], s38 offen offset: 0 lds // (op 40844) Load value direct to lds
v_permlane32_swap_b32 v109, v111 // (op 37802) 
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40844) least significant half

v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40844) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[138:141], a[16:19], v142, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1098) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[162:165], a[20:23], v142, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1324) 
s_add_u32 m0, m0, 4096 // (op 40844) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[166:169], a[24:27], v142, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1550) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[170:173], a[28:31], v142, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1776) 
v_bfe_u32 v112, v119, 0, 8 // (op 37410) 
v_bfe_u32 v113, v119, 8, 8 // (op 37410) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[96:99], a[32:35], v143, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2996) 
v_bfe_u32 v114, v119, 16, 8 // (op 37410) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[100:103], a[36:39], v143, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2920) 
v_bfe_u32 v115, v119, 24, 8 // (op 37410) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[104:107], a[40:43], v143, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2844) 
v_permlane16_swap_b32 v112, v113 // (op 37410) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[158:161], a[44:47], v143, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2768) 
v_permlane16_swap_b32 v114, v115 // (op 37410) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[138:141], a[48:51], v143, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2692) 
s_nop 0
v_permlane32_swap_b32 v112, v114 // (op 37410) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[162:165], a[52:55], v143, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2616) 
v_permlane32_swap_b32 v113, v115 // (op 37410) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[166:169], a[56:59], v143, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2540) 
s_waitcnt vmcnt(26)
s_barrier  // (op 42320) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[170:173], a[60:63], v143, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2464) 
ds_read_b128 v[44:47], v4 // (op 65) Load local data 



ds_read_b128 v[48:51], v4 offset:1024 // (op 414) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[96:99], a[64:67], v144, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4998) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[100:103], a[68:71], v144, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4922) 
ds_read_b128 v[52:55], v4 offset:2048 // (op 640) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[104:107], a[72:75], v144, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4846) 
ds_read_b128 v[56:59], v4 offset:3072 // (op 866) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[158:161], a[76:79], v144, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4770) 
ds_read_b128 v[60:63], v4 offset:4096 // (op 1092) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[84:87], v[138:141], a[80:83], v144, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4694) 
ds_read_b128 v[64:67], v4 offset:5120 // (op 1318) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[84:87], v[162:165], a[84:87], v144, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4618) 
ds_read_b128 v[68:71], v4 offset:6144 // (op 1544) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:87], v[166:169], a[88:91], v144, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4542) 
ds_read_b128 v[72:75], v4 offset:7168 // (op 1770) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[84:87], v[170:173], a[92:95], v144, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4466) 
buffer_load_dwordx4 v128, s[52:55], s38 offen offset: 0 lds // (op 40844) Load value direct to lds
v_add_co_u32 v128, s[64:65], v128, v16 // (op 40844) least significant half
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[96:99], a[96:99], v145, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7000) 
v_addc_co_u32 v129, s[64:65], 0, v129, s[64:65] // (op 40844) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[100:103], a[100:103], v145, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6924) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:91], v[104:107], a[104:107], v145, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6848) 
s_add_u32 m0, m0, 4096 // (op 40844) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:91], v[158:161], a[108:111], v145, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6772) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[88:91], v[138:141], a[112:115], v145, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6696) 
s_waitcnt vmcnt(23)
s_barrier  // (op 42303) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[88:91], v[162:165], a[116:119], v145, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6620) 
ds_read_b128 v[76:79], v0 offset:16384 // (op 60) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[88:91], v[166:169], a[120:123], v145, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6544) 
buffer_load_dwordx4 v128, s[52:55], s38 offen offset: 0 lds // (op 40844) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[88:91], v[170:173], a[124:127], v145, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6468) 

v_add_co_u32 v22, vcc, 256, v22 // (op 41046) least significant half



s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[92:95], v[96:99], a[128:131], v112, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9002) 
ds_read_b128 v[80:83], v0 offset:17408 // (op 2457) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[92:95], v[100:103], a[132:135], v112, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8926) 
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 41046) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[92:95], v[104:107], a[136:139], v112, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8850) 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 42359) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[92:95], v[158:161], a[140:143], v112, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8774) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 40839) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:95], v[138:141], a[144:147], v112, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8698) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 40839) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[92:95], v[162:165], a[148:151], v112, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8622) 
v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 40839) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[166:169], a[152:155], v112, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8546) 
v_mov_b64 v[20:21], v[26:27] // (op 40839) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[170:173], a[156:159], v112, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8470) 
s_add_u32 s19, 114688, s14 // (op 40839) 
s_mov_b32 m0, s19 // (op 40839) reg expression



v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[124:127], v[96:99], a[160:163], v113, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11004) 
ds_read_b128 v[84:87], v0 offset:18432 // (op 4459) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[124:127], v[100:103], a[164:167], v113, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10928) 
s_mov_b32 s38, 448 // (op 40839) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[124:127], v[104:107], a[168:171], v113, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10852) 
buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 40839) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[124:127], v[158:161], a[172:175], v113, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10776) 
v_add_co_u32 v20, s[64:65], v20, v16 // (op 40839) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[138:141], a[176:179], v113, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10700) 
v_addc_co_u32 v21, s[64:65], 0, v21, s[64:65] // (op 40839) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[162:165], a[180:183], v113, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10624) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[166:169], a[184:187], v113, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10548) 
s_add_u32 m0, m0, 4096 // (op 40839) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[170:173], a[188:191], v113, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10472) 

buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 40839) Load value direct to lds



v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[96:99], a[192:195], v114, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13006) 
ds_read_b128 v[88:91], v0 offset:19456 // (op 6461) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[100:103], a[196:199], v114, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12930) 
v_add_co_u32 v20, s[64:65], v20, v16 // (op 40839) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[104:107], a[200:203], v114, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12854) 
v_addc_co_u32 v21, s[64:65], 0, v21, s[64:65] // (op 40839) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[158:161], a[204:207], v114, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12778) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[130:133], v[138:141], a[208:211], v114, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12702) 
s_add_u32 m0, m0, 4096 // (op 40839) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:133], v[162:165], a[212:215], v114, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12626) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[130:133], v[166:169], a[216:219], v114, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12550) 
buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 40839) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[130:133], v[170:173], a[220:223], v114, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12474) 
v_add_co_u32 v20, s[64:65], v20, v16 // (op 40839) least significant half

v_addc_co_u32 v21, s[64:65], 0, v21, s[64:65] // (op 40839) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[134:137], v[96:99], a[224:227], v115, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15008) 
s_add_u32 m0, m0, 4096 // (op 40839) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[134:137], v[100:103], a[228:231], v115, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14932) 
ds_read_b128 v[92:95], v0 offset:20480 // (op 8463) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[134:137], v[104:107], a[232:235], v115, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14856) 
ds_read_b128 v[96:99], v0 offset:21504 // (op 10465) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[134:137], v[158:161], a[236:239], v115, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14780) 
ds_read_b128 v[100:103], v0 offset:22528 // (op 12467) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[134:137], v[138:141], a[240:243], v115, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14704) 
buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 40839) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[134:137], v[162:165], a[244:247], v115, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14628) 
v_add_co_u32 v18, vcc, 256, v18 // (op 40982) least significant half

v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 40982) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[134:137], v[166:169], a[248:251], v115, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14552) 
ds_read_b128 v[104:107], v0 offset:23552 // (op 14469) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[134:137], v[170:173], a[252:255], v115, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14476) 




s_ashr_i64 s[38:39], s[36:37], 7 // (op 12383) 
s_mov_b32 s19, s38 // (op 12383) convert

s_and_b32 s38, s19, 4294967292 // (op 12383) 

s_cmp_lt_u32 s18, s38 // (op 12383) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoop_12383 // (op 12383) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoop_12383 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoop_12383:
















s_barrier  // (op 42454) 
v_lshrrev_b32 v14, 2, v1 // (op 40882) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 40882) 
v_lshl_add_u32 v15, s16, 8, v14 // (op 40882) 

s_lshr_b32 s6, s0, 2 // (op 40882) 

v_and_b32 v14, -4, v1 // (op 40882) 
v_sub_u32 v16, v1, v14 // (op 40882) 

v_lshlrev_b32 v14, 5, v16 // (op 40882) 

v_lshl_add_u32 v16, s6, 9, v14 // (op 40882) 


v_mov_b32 v27, 0 // (op 40882) convert
v_mov_b32 v26, v16 // (op 40882) convert

v_mul_lo_u32 v17, v15, s13 // (op 40882) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 40882) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 40882) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 40882) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v26 // (op 40882) least significant half
v_addc_co_u32 v17, s[0:1], v17, v27, s[0:1] // (op 40882) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 40882) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 40882) 

s_mul_i32 s1, s12, s23 // (op 40884) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 40884) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 40884) least significant
s_mul_hi_u32 s7, s12, s22 // (op 40884) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 40884) most significant: sum
s_add_u32 s1, s1, s7 // (op 40884) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 40884) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 40884) 

s_mov_b32 s48, s20 // (op 40884) reg expression
s_mov_b32 s49, s21 // (op 40884) reg expression
s_mov_b32 s50, s0 // (op 40884) reg expression
s_mov_b32 s51, 131072 // (op 40884) call()

v_and_b32 v14, -4, v1 // (op 41012) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 41012) 
v_sub_u32 v15, v1, v14 // (op 41012) 

s_lshr_b32 s6, s0, 2 // (op 41012) 

v_lshlrev_b32 v14, 5, v15 // (op 41012) 

v_lshl_add_u32 v15, s6, 9, v14 // (op 41012) 


v_mov_b32 v19, 0 // (op 41012) convert
v_mov_b32 v18, v15 // (op 41012) convert

v_lshrrev_b32 v14, 2, v1 // (op 41012) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 41012) 

v_mul_lo_u32 v21, v15, s25 // (op 41012) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 41012) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 41012) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 41012) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 41012) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 41012) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 41012) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 41012) 

s_mul_i32 s1, s24, s9 // (op 41014) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 41014) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 41014) least significant
s_mul_hi_u32 s7, s24, s8 // (op 41014) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 41014) most significant: sum
s_add_u32 s1, s1, s7 // (op 41014) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 41014) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 41014) 

s_mov_b32 s52, s26 // (op 41014) reg expression
s_mov_b32 s53, s27 // (op 41014) reg expression
s_mov_b32 s54, s0 // (op 41014) reg expression
s_mov_b32 s55, 131072 // (op 41014) call()

v_and_b32 v14, -4, v1 // (op 41110) 
v_sub_u32 v15, v1, v14 // (op 41110) 

v_lshlrev_b32 v14, 5, v15 // (op 41110) 

v_lshrrev_b32 v15, 2, v1 // (op 41110) 
v_lshl_add_u32 v20, v15, 7, v14 // (op 41110) 


v_lshlrev_b32 v14, 2, v20 // (op 41110) 

v_lshrrev_b32 v15, 3, v14 // (op 41110) 

s_nop 0
v_readfirstlane_b32 s0, v15 // (op 41110) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41167) 
v_sub_u32 v15, v1, v14 // (op 41167) 

v_lshlrev_b32 v14, 5, v15 // (op 41167) 

v_lshrrev_b32 v15, 2, v1 // (op 41167) 
v_lshl_add_u32 v20, v15, 7, v14 // (op 41167) 


v_lshlrev_b32 v14, 2, v20 // (op 41167) 

v_lshrrev_b32 v15, 3, v14 // (op 41167) 

s_nop 0
v_readfirstlane_b32 s1, v15 // (op 41167) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, 63, v1 // (op 41608) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 41608) 
v_lshrrev_b32 v15, 4, v14 // (op 41608) 
s_lshr_b32 s10, s6, 2 // (op 41608) 

v_and_b32 v20, 3, v15 // (op 41608) 

v_and_b32 v15, 15, v1 // (op 41608) 
v_lshl_add_u32 v21, v20, 4, v15 // (op 41608) 


v_and_b32 v15, 4294967168, v1 // (op 41608) 
v_add_u32 v20, v15, v21 // (op 41608) 


v_lshl_add_u32 v15, s16, 8, v20 // (op 41608) 

v_lshrrev_b32 v20, 6, v14 // (op 41608) 

v_lshlrev_b32 v14, 2, v20 // (op 41608) 

v_lshl_add_u32 v20, s10, 4, v14 // (op 41608) 


v_mul_lo_u32 v14, v15, s28 // (op 41608) 
v_add_u32 v14, v14, v20 // (op 41608) 


s_mul_i32 s7, s28, s23 // (op 41610) most significant: low of low * high
s_mul_i32 s10, s29, s22 // (op 41610) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 41610) least significant
s_mul_hi_u32 s11, s28, s22 // (op 41610) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 41610) most significant: sum
s_add_u32 s7, s7, s11 // (op 41610) most significant: sum


s_mov_b32 s56, s30 // (op 41610) reg expression
s_mov_b32 s57, s31 // (op 41610) reg expression
s_mov_b32 s58, s6 // (op 41610) reg expression
s_mov_b32 s59, 131072 // (op 41610) call()

v_and_b32 v15, 63, v1 // (op 41635) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 41635) 
v_lshrrev_b32 v20, 6, v15 // (op 41635) 
s_lshr_b32 s10, s6, 2 // (op 41635) 

v_lshlrev_b32 v21, 2, v20 // (op 41635) 

v_lshl_add_u32 v20, s10, 4, v21 // (op 41635) 


v_lshrrev_b32 v21, 4, v15 // (op 41635) 

v_and_b32 v15, 3, v21 // (op 41635) 

v_and_b32 v21, 15, v1 // (op 41635) 
v_lshl_add_u32 v22, v15, 4, v21 // (op 41635) 


v_lshrrev_b32 v15, 6, v1 // (op 41635) 
v_and_b32 v21, 1, v15 // (op 41635) 

v_lshl_add_u32 v15, v21, 7, v22 // (op 41635) 


v_lshl_add_u32 v21, s17, 8, v15 // (op 41635) 

v_mul_lo_u32 v15, v21, s32 // (op 41635) 

v_add_u32 v21, v20, v15 // (op 41635) 


s_mul_i32 s7, s32, s9 // (op 41637) most significant: low of low * high
s_mul_i32 s10, s33, s8 // (op 41637) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 41637) least significant
s_mul_hi_u32 s11, s32, s8 // (op 41637) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 41637) most significant: sum
s_add_u32 s7, s7, s11 // (op 41637) most significant: sum


s_mov_b32 s60, s34 // (op 41637) reg expression
s_mov_b32 s61, s35 // (op 41637) reg expression
s_mov_b32 s62, s6 // (op 41637) reg expression
s_mov_b32 s63, 131072 // (op 41637) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 14445) 
s_and_b32 s10, s6, 4294967292 // (op 14445) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 14444) 
s_mov_b32 s11, s6 // (op 14444) convert

s_cmp_lt_u32 s10, s11 // (op 14444) 




















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoopTail_14444 // (op 14444) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoopTail_14444 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoopTail_14444:
s_barrier  // (op 42300) 
s_add_u32 s10, s10, 1 // (op 14446) 

s_lshl_b64 s[6:7], s[12:13], 6 // (op 40807) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 40807) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 40807) 


v_mov_b64 v[24:25], v[16:17] // (op 40807) 
v_mov_b64 v[26:27], v[24:25] // (op 40807) 
s_add_u32 s11, 16384, s0 // (op 40807) 
s_mov_b32 m0, s11 // (op 40807) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 40807) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 40807) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 40807) most significant half

s_add_u32 m0, m0, 4096 // (op 40807) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 40807) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 40807) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 40807) most significant half

s_add_u32 m0, m0, 4096 // (op 40807) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 40807) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 40807) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 40807) most significant half

s_add_u32 m0, m0, 4096 // (op 40807) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 40807) Load value direct to lds




v_add_co_u32 v16, s[6:7], 64, v16 // (op 40872) least significant half
v_addc_co_u32 v17, s[6:7], 0, v17, s[6:7] // (op 40872) most significant half

s_barrier  // (op 42317) 

s_lshl_b64 s[6:7], s[24:25], 6 // (op 40813) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 40813) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 40813) 


v_mov_b64 v[24:25], v[18:19] // (op 40813) 
v_mov_b64 v[26:27], v[24:25] // (op 40813) 
s_mov_b32 m0, s1 // (op 40813) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 40813) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 40813) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 40813) most significant half

s_add_u32 m0, m0, 4096 // (op 40813) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 40813) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 40813) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 40813) most significant half

s_add_u32 m0, m0, 4096 // (op 40813) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 40813) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 40813) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 40813) most significant half

s_add_u32 m0, m0, 4096 // (op 40813) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 40813) Load value direct to lds



v_add_co_u32 v18, s[6:7], 64, v18 // (op 41004) least significant half
v_addc_co_u32 v19, s[6:7], 0, v19, s[6:7] // (op 41004) most significant half

s_waitcnt vmcnt(4)
s_barrier  // (op 42307) 



ds_read_b128 v[22:25], v0 offset:16384 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 42323) 



ds_read_b128 v[26:29], v4 // (op 162) Load local data 


v_mov_b32 v15, v14 // (op 164) 
v_mov_b32 v20, v15 // (op 164) reg expression

buffer_load_dwordx4 v[30:33], v20, s[56:59], 0 offen // (op 164) Load value



v_mov_b32 v20, v21 // (op 166) 
v_mov_b32 v34, v20 // (op 166) reg expression

buffer_load_dwordx4 v[36:39], v34, s[60:63], 0 offen // (op 166) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v40, v30, 0, 8 // (op 38075) 



v_bfe_u32 v41, v30, 8, 8 // (op 38075) 
v_bfe_u32 v42, v30, 16, 8 // (op 38075) 
v_bfe_u32 v43, v30, 24, 8 // (op 38075) 
v_permlane16_swap_b32 v40, v41 // (op 38075) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 397) Load local data 
v_permlane16_swap_b32 v42, v43 // (op 38075) 



s_nop 1
v_permlane32_swap_b32 v40, v42 // (op 38075) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v41, v43 // (op 38075) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 623) Load local data 




ds_read_b128 v[30:33], v4 offset:3072 // (op 849) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v52, v36, 0, 8 // (op 38167) 



v_bfe_u32 v53, v36, 8, 8 // (op 38167) 
v_bfe_u32 v54, v36, 16, 8 // (op 38167) 
v_bfe_u32 v55, v36, 24, 8 // (op 38167) 
v_permlane16_swap_b32 v52, v53 // (op 38167) 
s_nop 0
v_permlane16_swap_b32 v54, v55 // (op 38167) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 38167) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 38167) 
ds_read_b128 v[56:59], v4 offset:4096 // (op 1075) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 167) 
s_lshl_b32 s11, s32, 6 // (op 1080) 
v_add_u32 v34, v20, s11 // (op 1080) 


buffer_load_dwordx4 v[36:39], v34, s[60:63], 0 offen // (op 1080) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[44:47], a[4:7], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 400) 




v_add_u32 v21, 4, v21 // (op 41625) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[48:51], a[8:11], v40, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 626) 
ds_read_b128 v[60:63], v4 offset:5120 // (op 1301) Load local data 



ds_read_b128 v[64:67], v4 offset:6144 // (op 1527) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:25], v[30:33], a[12:15], v40, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 852) 



ds_read_b128 v[68:71], v4 offset:7168 // (op 1753) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v72, v36, 0, 8 // (op 38181) 



v_bfe_u32 v73, v36, 8, 8 // (op 38181) 
v_bfe_u32 v74, v36, 16, 8 // (op 38181) 
v_bfe_u32 v75, v36, 24, 8 // (op 38181) 
v_permlane16_swap_b32 v72, v73 // (op 38181) 
s_nop 0
v_permlane16_swap_b32 v74, v75 // (op 38181) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v72, v74 // (op 38181) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v73, v75 // (op 38181) 
ds_read_b128 v[76:79], v0 offset:17408 // (op 2249) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[56:59], a[16:19], v40, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1078) 
ds_read_b128 v[34:37], v0 offset:18432 // (op 4251) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[60:63], a[20:23], v40, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1304) 
ds_read_b128 v[80:83], v0 offset:19456 // (op 6253) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[64:67], a[24:27], v40, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1530) 
ds_read_b128 v[84:87], v0 offset:20480 // (op 8255) Load local data 


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:25], v[68:71], a[28:31], v40, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1756) 
s_lshl_b32 s11, s28, 6 // (op 8264) 

v_add_u32 v20, v15, s11 // (op 8264) 


buffer_load_dwordx4 v[22:25], v20, s[56:59], 0 offen // (op 8264) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[26:29], a[32:35], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2253) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[48:51], a[40:43], v41, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2307) 
v_add_u32 v14, 4, v14 // (op 41598) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[30:33], a[44:47], v41, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2334) 
ds_read_b128 v[88:91], v0 offset:21504 // (op 10257) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[44:47], a[36:39], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2280) 
ds_read_b128 v[92:95], v0 offset:22528 // (op 12259) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[56:59], a[48:51], v41, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2361) 
s_waitcnt vmcnt(0)
v_bfe_u32 v96, v22, 0, 8 // (op 38089) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[64:67], a[56:59], v41, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2415) 
v_bfe_u32 v97, v22, 8, 8 // (op 38089) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[68:71], a[60:63], v41, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2442) 
v_bfe_u32 v98, v22, 16, 8 // (op 38089) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[60:63], a[52:55], v41, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2388) 
v_bfe_u32 v99, v22, 24, 8 // (op 38089) 

v_permlane16_swap_b32 v96, v97 // (op 38089) 
s_nop 0
v_permlane16_swap_b32 v98, v99 // (op 38089) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:37], v[26:29], a[64:67], v42, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4255) 
s_nop 0
v_permlane32_swap_b32 v96, v98 // (op 38089) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[34:37], v[48:51], a[72:75], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4309) 
v_permlane32_swap_b32 v97, v99 // (op 38089) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:37], v[30:33], a[76:79], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4336) 
ds_read_b128 v[22:25], v0 offset:23552 // (op 14261) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[44:47], a[68:71], v42, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4282) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[56:59], a[80:83], v42, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4363) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[34:37], v[64:67], a[88:91], v42, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4417) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[68:71], a[92:95], v42, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4444) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:37], v[60:63], a[84:87], v42, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4390) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[26:29], a[96:99], v43, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6257) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[48:51], a[104:107], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6311) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[30:33], a[108:111], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6338) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[44:47], a[100:103], v43, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6284) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[80:83], v[56:59], a[112:115], v43, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6365) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[64:67], a[120:123], v43, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6419) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[80:83], v[68:71], a[124:127], v43, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6446) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[80:83], v[60:63], a[116:119], v43, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6392) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:87], v[44:47], a[132:135], v96, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8286) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[84:87], v[60:63], a[148:151], v96, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8394) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:87], v[26:29], a[128:131], v96, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8259) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:87], v[48:51], a[136:139], v96, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8313) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:87], v[30:33], a[140:143], v96, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8340) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[84:87], v[56:59], a[144:147], v96, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8367) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:87], v[64:67], a[152:155], v96, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8421) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:87], v[68:71], a[156:159], v96, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8448) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[88:91], v[44:47], a[164:167], v97, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10288) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[88:91], v[60:63], a[180:183], v97, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10396) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[88:91], v[26:29], a[160:163], v97, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10261) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:91], v[48:51], a[168:171], v97, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10315) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[88:91], v[30:33], a[172:175], v97, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10342) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[88:91], v[56:59], a[176:179], v97, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10369) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[88:91], v[64:67], a[184:187], v97, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10423) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[88:91], v[68:71], a[188:191], v97, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10450) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[92:95], v[44:47], a[196:199], v98, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12290) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[92:95], v[60:63], a[212:215], v98, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12398) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[92:95], v[26:29], a[192:195], v98, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12263) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[92:95], v[48:51], a[200:203], v98, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12317) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[92:95], v[30:33], a[204:207], v98, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12344) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:95], v[56:59], a[208:211], v98, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12371) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[92:95], v[64:67], a[216:219], v98, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12425) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[92:95], v[68:71], a[220:223], v98, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12452) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[22:25], v[44:47], a[228:231], v99, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14292) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[22:25], v[60:63], a[244:247], v99, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14400) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[22:25], v[26:29], a[224:227], v99, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14265) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[22:25], v[48:51], a[232:235], v99, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14319) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[22:25], v[30:33], a[236:239], v99, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14346) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[22:25], v[56:59], a[240:243], v99, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14373) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[22:25], v[64:67], a[248:251], v99, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14427) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[22:25], v[68:71], a[252:255], v99, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14454) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 14444) 
s_mov_b32 s11, s6 // (op 14444) convert

s_cmp_lt_u32 s10, s11 // (op 14444) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoopTail_14444 // (op 14444) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopTop_KLoopTail_14444 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ForLoopBottom_KLoopTail_14444:





















s_barrier  // (op 42482) 









v_cmp_eq_f32 vcc, 0.00000, v9 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalFalse_0_beta_114
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


v_accvgpr_read v4, a4 // (op 18198) 
v_accvgpr_read v5, a5 // (op 18198) 
v_accvgpr_read v6, a6 // (op 18198) 
v_accvgpr_read v7, a7 // (op 18198) 
v_mul_f32 v14, v11, v4 // (op 18198) 
v_mul_f32 v15, v11, v5 // (op 18198) 
v_mul_f32 v16, v11, v6 // (op 18198) 
v_mul_f32 v17, v11, v7 // (op 18198) 



s_lshl_b32 s0, s4, 6 // (op 18200) 
v_add_u32 v0, v13, s0 // (op 18200) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18200) Store value


v_accvgpr_read v4, a8 // (op 18210) 
v_accvgpr_read v5, a9 // (op 18210) 
v_accvgpr_read v6, a10 // (op 18210) 
v_accvgpr_read v7, a11 // (op 18210) 
v_mul_f32 v14, v11, v4 // (op 18210) 
v_mul_f32 v15, v11, v5 // (op 18210) 
v_mul_f32 v16, v11, v6 // (op 18210) 
v_mul_f32 v17, v11, v7 // (op 18210) 



s_lshl_b32 s0, s4, 7 // (op 18212) 
v_add_u32 v0, v13, s0 // (op 18212) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18212) Store value


v_accvgpr_read v4, a12 // (op 18222) 
v_accvgpr_read v5, a13 // (op 18222) 
v_accvgpr_read v6, a14 // (op 18222) 
v_accvgpr_read v7, a15 // (op 18222) 
v_mul_f32 v14, v11, v4 // (op 18222) 
v_mul_f32 v15, v11, v5 // (op 18222) 
v_mul_f32 v16, v11, v6 // (op 18222) 
v_mul_f32 v17, v11, v7 // (op 18222) 



s_lshl_b32 s0, s4, 6 // (op 18224) 
s_mul_i32 s1, 3, s0 // (op 18224) 

v_add_u32 v0, v13, s1 // (op 18224) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18224) Store value


v_accvgpr_read v4, a16 // (op 18234) 
v_accvgpr_read v5, a17 // (op 18234) 
v_accvgpr_read v6, a18 // (op 18234) 
v_accvgpr_read v7, a19 // (op 18234) 
v_mul_f32 v14, v11, v4 // (op 18234) 
v_mul_f32 v15, v11, v5 // (op 18234) 
v_mul_f32 v16, v11, v6 // (op 18234) 
v_mul_f32 v17, v11, v7 // (op 18234) 



s_lshl_b32 s0, s4, 8 // (op 18236) 
v_add_u32 v0, v13, s0 // (op 18236) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18236) Store value


v_accvgpr_read v4, a20 // (op 18246) 
v_accvgpr_read v5, a21 // (op 18246) 
v_accvgpr_read v6, a22 // (op 18246) 
v_accvgpr_read v7, a23 // (op 18246) 
v_mul_f32 v14, v11, v4 // (op 18246) 
v_mul_f32 v15, v11, v5 // (op 18246) 
v_mul_f32 v16, v11, v6 // (op 18246) 
v_mul_f32 v17, v11, v7 // (op 18246) 



s_lshl_b32 s0, s4, 6 // (op 18248) 
s_mul_i32 s1, 5, s0 // (op 18248) 

v_add_u32 v0, v13, s1 // (op 18248) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18248) Store value


v_accvgpr_read v4, a24 // (op 18258) 
v_accvgpr_read v5, a25 // (op 18258) 
v_accvgpr_read v6, a26 // (op 18258) 
v_accvgpr_read v7, a27 // (op 18258) 
v_mul_f32 v14, v11, v4 // (op 18258) 
v_mul_f32 v15, v11, v5 // (op 18258) 
v_mul_f32 v16, v11, v6 // (op 18258) 
v_mul_f32 v17, v11, v7 // (op 18258) 



s_lshl_b32 s0, s4, 6 // (op 18260) 
s_mul_i32 s1, 6, s0 // (op 18260) 

v_add_u32 v0, v13, s1 // (op 18260) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18260) Store value


v_accvgpr_read v4, a28 // (op 18270) 
v_accvgpr_read v5, a29 // (op 18270) 
v_accvgpr_read v6, a30 // (op 18270) 
v_accvgpr_read v7, a31 // (op 18270) 
v_mul_f32 v14, v11, v4 // (op 18270) 
v_mul_f32 v15, v11, v5 // (op 18270) 
v_mul_f32 v16, v11, v6 // (op 18270) 
v_mul_f32 v17, v11, v7 // (op 18270) 



s_lshl_b32 s0, s4, 6 // (op 18272) 
s_mul_i32 s1, 7, s0 // (op 18272) 

v_add_u32 v0, v13, s1 // (op 18272) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18272) Store value


v_accvgpr_read v4, a32 // (op 18324) 
v_accvgpr_read v5, a33 // (op 18324) 
v_accvgpr_read v6, a34 // (op 18324) 
v_accvgpr_read v7, a35 // (op 18324) 
v_mul_f32 v14, v11, v4 // (op 18324) 
v_mul_f32 v15, v11, v5 // (op 18324) 
v_mul_f32 v16, v11, v6 // (op 18324) 
v_mul_f32 v17, v11, v7 // (op 18324) 



v_add_u32 v0, 64, v13 // (op 18327) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18327) Store value


v_accvgpr_read v4, a36 // (op 18345) 
v_accvgpr_read v5, a37 // (op 18345) 
v_accvgpr_read v6, a38 // (op 18345) 
v_accvgpr_read v7, a39 // (op 18345) 
v_mul_f32 v14, v11, v4 // (op 18345) 
v_mul_f32 v15, v11, v5 // (op 18345) 
v_mul_f32 v16, v11, v6 // (op 18345) 
v_mul_f32 v17, v11, v7 // (op 18345) 



s_lshl_b32 s0, s4, 6 // (op 18329) 
s_add_u32 s0, s0, 64 // (op 18329) 
v_add_u32 v0, v13, s0 // (op 18329) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18329) Store value


v_accvgpr_read v4, a40 // (op 18350) 
v_accvgpr_read v5, a41 // (op 18350) 
v_accvgpr_read v6, a42 // (op 18350) 
v_accvgpr_read v7, a43 // (op 18350) 
v_mul_f32 v14, v11, v4 // (op 18350) 
v_mul_f32 v15, v11, v5 // (op 18350) 
v_mul_f32 v16, v11, v6 // (op 18350) 
v_mul_f32 v17, v11, v7 // (op 18350) 



s_lshl_b32 s0, s4, 7 // (op 18331) 
s_add_u32 s0, s0, 64 // (op 18331) 
v_add_u32 v0, v13, s0 // (op 18331) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18331) Store value


v_accvgpr_read v4, a44 // (op 18355) 
v_accvgpr_read v5, a45 // (op 18355) 
v_accvgpr_read v6, a46 // (op 18355) 
v_accvgpr_read v7, a47 // (op 18355) 
v_mul_f32 v14, v11, v4 // (op 18355) 
v_mul_f32 v15, v11, v5 // (op 18355) 
v_mul_f32 v16, v11, v6 // (op 18355) 
v_mul_f32 v17, v11, v7 // (op 18355) 



s_lshl_b32 s0, s4, 6 // (op 18333) 
s_mul_i32 s1, 3, s0 // (op 18333) 
s_add_u32 s1, s1, 64 // (op 18333) 

v_add_u32 v0, v13, s1 // (op 18333) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18333) Store value


v_accvgpr_read v4, a48 // (op 18360) 
v_accvgpr_read v5, a49 // (op 18360) 
v_accvgpr_read v6, a50 // (op 18360) 
v_accvgpr_read v7, a51 // (op 18360) 
v_mul_f32 v14, v11, v4 // (op 18360) 
v_mul_f32 v15, v11, v5 // (op 18360) 
v_mul_f32 v16, v11, v6 // (op 18360) 
v_mul_f32 v17, v11, v7 // (op 18360) 



s_lshl_b32 s0, s4, 8 // (op 18335) 
s_add_u32 s0, s0, 64 // (op 18335) 
v_add_u32 v0, v13, s0 // (op 18335) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18335) Store value


v_accvgpr_read v4, a52 // (op 18365) 
v_accvgpr_read v5, a53 // (op 18365) 
v_accvgpr_read v6, a54 // (op 18365) 
v_accvgpr_read v7, a55 // (op 18365) 
v_mul_f32 v14, v11, v4 // (op 18365) 
v_mul_f32 v15, v11, v5 // (op 18365) 
v_mul_f32 v16, v11, v6 // (op 18365) 
v_mul_f32 v17, v11, v7 // (op 18365) 



s_lshl_b32 s0, s4, 6 // (op 18337) 
s_mul_i32 s1, 5, s0 // (op 18337) 
s_add_u32 s1, s1, 64 // (op 18337) 

v_add_u32 v0, v13, s1 // (op 18337) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18337) Store value


v_accvgpr_read v4, a56 // (op 18370) 
v_accvgpr_read v5, a57 // (op 18370) 
v_accvgpr_read v6, a58 // (op 18370) 
v_accvgpr_read v7, a59 // (op 18370) 
v_mul_f32 v14, v11, v4 // (op 18370) 
v_mul_f32 v15, v11, v5 // (op 18370) 
v_mul_f32 v16, v11, v6 // (op 18370) 
v_mul_f32 v17, v11, v7 // (op 18370) 



s_lshl_b32 s0, s4, 6 // (op 18339) 
s_mul_i32 s1, 6, s0 // (op 18339) 
s_add_u32 s1, s1, 64 // (op 18339) 

v_add_u32 v0, v13, s1 // (op 18339) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18339) Store value


v_accvgpr_read v4, a60 // (op 18375) 
v_accvgpr_read v5, a61 // (op 18375) 
v_accvgpr_read v6, a62 // (op 18375) 
v_accvgpr_read v7, a63 // (op 18375) 
v_mul_f32 v14, v11, v4 // (op 18375) 
v_mul_f32 v15, v11, v5 // (op 18375) 
v_mul_f32 v16, v11, v6 // (op 18375) 
v_mul_f32 v17, v11, v7 // (op 18375) 



s_lshl_b32 s0, s4, 6 // (op 18341) 
s_mul_i32 s1, 7, s0 // (op 18341) 
s_add_u32 s1, s1, 64 // (op 18341) 

v_add_u32 v0, v13, s1 // (op 18341) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18341) Store value


v_accvgpr_read v4, a64 // (op 18456) 
v_accvgpr_read v5, a65 // (op 18456) 
v_accvgpr_read v6, a66 // (op 18456) 
v_accvgpr_read v7, a67 // (op 18456) 
v_mul_f32 v14, v11, v4 // (op 18456) 
v_mul_f32 v15, v11, v5 // (op 18456) 
v_mul_f32 v16, v11, v6 // (op 18456) 
v_mul_f32 v17, v11, v7 // (op 18456) 



v_add_u32 v0, 128, v13 // (op 18459) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18459) Store value


v_accvgpr_read v4, a68 // (op 18477) 
v_accvgpr_read v5, a69 // (op 18477) 
v_accvgpr_read v6, a70 // (op 18477) 
v_accvgpr_read v7, a71 // (op 18477) 
v_mul_f32 v14, v11, v4 // (op 18477) 
v_mul_f32 v15, v11, v5 // (op 18477) 
v_mul_f32 v16, v11, v6 // (op 18477) 
v_mul_f32 v17, v11, v7 // (op 18477) 



s_lshl_b32 s0, s4, 6 // (op 18461) 
s_add_u32 s0, s0, 128 // (op 18461) 
v_add_u32 v0, v13, s0 // (op 18461) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18461) Store value


v_accvgpr_read v4, a72 // (op 18482) 
v_accvgpr_read v5, a73 // (op 18482) 
v_accvgpr_read v6, a74 // (op 18482) 
v_accvgpr_read v7, a75 // (op 18482) 
v_mul_f32 v14, v11, v4 // (op 18482) 
v_mul_f32 v15, v11, v5 // (op 18482) 
v_mul_f32 v16, v11, v6 // (op 18482) 
v_mul_f32 v17, v11, v7 // (op 18482) 



s_lshl_b32 s0, s4, 7 // (op 18463) 
s_add_u32 s0, s0, 128 // (op 18463) 
v_add_u32 v0, v13, s0 // (op 18463) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18463) Store value


v_accvgpr_read v4, a76 // (op 18487) 
v_accvgpr_read v5, a77 // (op 18487) 
v_accvgpr_read v6, a78 // (op 18487) 
v_accvgpr_read v7, a79 // (op 18487) 
v_mul_f32 v14, v11, v4 // (op 18487) 
v_mul_f32 v15, v11, v5 // (op 18487) 
v_mul_f32 v16, v11, v6 // (op 18487) 
v_mul_f32 v17, v11, v7 // (op 18487) 



s_lshl_b32 s0, s4, 6 // (op 18465) 
s_mul_i32 s1, 3, s0 // (op 18465) 
s_add_u32 s1, s1, 128 // (op 18465) 

v_add_u32 v0, v13, s1 // (op 18465) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18465) Store value


v_accvgpr_read v4, a80 // (op 18492) 
v_accvgpr_read v5, a81 // (op 18492) 
v_accvgpr_read v6, a82 // (op 18492) 
v_accvgpr_read v7, a83 // (op 18492) 
v_mul_f32 v14, v11, v4 // (op 18492) 
v_mul_f32 v15, v11, v5 // (op 18492) 
v_mul_f32 v16, v11, v6 // (op 18492) 
v_mul_f32 v17, v11, v7 // (op 18492) 



s_lshl_b32 s0, s4, 8 // (op 18467) 
s_add_u32 s0, s0, 128 // (op 18467) 
v_add_u32 v0, v13, s0 // (op 18467) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18467) Store value


v_accvgpr_read v4, a84 // (op 18497) 
v_accvgpr_read v5, a85 // (op 18497) 
v_accvgpr_read v6, a86 // (op 18497) 
v_accvgpr_read v7, a87 // (op 18497) 
v_mul_f32 v14, v11, v4 // (op 18497) 
v_mul_f32 v15, v11, v5 // (op 18497) 
v_mul_f32 v16, v11, v6 // (op 18497) 
v_mul_f32 v17, v11, v7 // (op 18497) 



s_lshl_b32 s0, s4, 6 // (op 18469) 
s_mul_i32 s1, 5, s0 // (op 18469) 
s_add_u32 s1, s1, 128 // (op 18469) 

v_add_u32 v0, v13, s1 // (op 18469) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18469) Store value


v_accvgpr_read v4, a88 // (op 18502) 
v_accvgpr_read v5, a89 // (op 18502) 
v_accvgpr_read v6, a90 // (op 18502) 
v_accvgpr_read v7, a91 // (op 18502) 
v_mul_f32 v14, v11, v4 // (op 18502) 
v_mul_f32 v15, v11, v5 // (op 18502) 
v_mul_f32 v16, v11, v6 // (op 18502) 
v_mul_f32 v17, v11, v7 // (op 18502) 



s_lshl_b32 s0, s4, 6 // (op 18471) 
s_mul_i32 s1, 6, s0 // (op 18471) 
s_add_u32 s1, s1, 128 // (op 18471) 

v_add_u32 v0, v13, s1 // (op 18471) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18471) Store value


v_accvgpr_read v4, a92 // (op 18507) 
v_accvgpr_read v5, a93 // (op 18507) 
v_accvgpr_read v6, a94 // (op 18507) 
v_accvgpr_read v7, a95 // (op 18507) 
v_mul_f32 v14, v11, v4 // (op 18507) 
v_mul_f32 v15, v11, v5 // (op 18507) 
v_mul_f32 v16, v11, v6 // (op 18507) 
v_mul_f32 v17, v11, v7 // (op 18507) 



s_lshl_b32 s0, s4, 6 // (op 18473) 
s_mul_i32 s1, 7, s0 // (op 18473) 
s_add_u32 s1, s1, 128 // (op 18473) 

v_add_u32 v0, v13, s1 // (op 18473) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18473) Store value


v_accvgpr_read v4, a96 // (op 18588) 
v_accvgpr_read v5, a97 // (op 18588) 
v_accvgpr_read v6, a98 // (op 18588) 
v_accvgpr_read v7, a99 // (op 18588) 
v_mul_f32 v14, v11, v4 // (op 18588) 
v_mul_f32 v15, v11, v5 // (op 18588) 
v_mul_f32 v16, v11, v6 // (op 18588) 
v_mul_f32 v17, v11, v7 // (op 18588) 



v_add_u32 v0, 192, v13 // (op 18591) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18591) Store value


v_accvgpr_read v4, a100 // (op 18609) 
v_accvgpr_read v5, a101 // (op 18609) 
v_accvgpr_read v6, a102 // (op 18609) 
v_accvgpr_read v7, a103 // (op 18609) 
v_mul_f32 v14, v11, v4 // (op 18609) 
v_mul_f32 v15, v11, v5 // (op 18609) 
v_mul_f32 v16, v11, v6 // (op 18609) 
v_mul_f32 v17, v11, v7 // (op 18609) 



s_lshl_b32 s0, s4, 6 // (op 18593) 
s_add_u32 s0, s0, 192 // (op 18593) 
v_add_u32 v0, v13, s0 // (op 18593) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18593) Store value


v_accvgpr_read v4, a104 // (op 18614) 
v_accvgpr_read v5, a105 // (op 18614) 
v_accvgpr_read v6, a106 // (op 18614) 
v_accvgpr_read v7, a107 // (op 18614) 
v_mul_f32 v14, v11, v4 // (op 18614) 
v_mul_f32 v15, v11, v5 // (op 18614) 
v_mul_f32 v16, v11, v6 // (op 18614) 
v_mul_f32 v17, v11, v7 // (op 18614) 



s_lshl_b32 s0, s4, 7 // (op 18595) 
s_add_u32 s0, s0, 192 // (op 18595) 
v_add_u32 v0, v13, s0 // (op 18595) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18595) Store value


v_accvgpr_read v4, a108 // (op 18619) 
v_accvgpr_read v5, a109 // (op 18619) 
v_accvgpr_read v6, a110 // (op 18619) 
v_accvgpr_read v7, a111 // (op 18619) 
v_mul_f32 v14, v11, v4 // (op 18619) 
v_mul_f32 v15, v11, v5 // (op 18619) 
v_mul_f32 v16, v11, v6 // (op 18619) 
v_mul_f32 v17, v11, v7 // (op 18619) 



s_lshl_b32 s0, s4, 6 // (op 18597) 
s_mul_i32 s1, 3, s0 // (op 18597) 
s_add_u32 s1, s1, 192 // (op 18597) 

v_add_u32 v0, v13, s1 // (op 18597) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18597) Store value


v_accvgpr_read v4, a112 // (op 18624) 
v_accvgpr_read v5, a113 // (op 18624) 
v_accvgpr_read v6, a114 // (op 18624) 
v_accvgpr_read v7, a115 // (op 18624) 
v_mul_f32 v14, v11, v4 // (op 18624) 
v_mul_f32 v15, v11, v5 // (op 18624) 
v_mul_f32 v16, v11, v6 // (op 18624) 
v_mul_f32 v17, v11, v7 // (op 18624) 



s_lshl_b32 s0, s4, 8 // (op 18599) 
s_add_u32 s0, s0, 192 // (op 18599) 
v_add_u32 v0, v13, s0 // (op 18599) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18599) Store value


v_accvgpr_read v4, a116 // (op 18629) 
v_accvgpr_read v5, a117 // (op 18629) 
v_accvgpr_read v6, a118 // (op 18629) 
v_accvgpr_read v7, a119 // (op 18629) 
v_mul_f32 v14, v11, v4 // (op 18629) 
v_mul_f32 v15, v11, v5 // (op 18629) 
v_mul_f32 v16, v11, v6 // (op 18629) 
v_mul_f32 v17, v11, v7 // (op 18629) 



s_lshl_b32 s0, s4, 6 // (op 18601) 
s_mul_i32 s1, 5, s0 // (op 18601) 
s_add_u32 s1, s1, 192 // (op 18601) 

v_add_u32 v0, v13, s1 // (op 18601) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18601) Store value


v_accvgpr_read v4, a120 // (op 18634) 
v_accvgpr_read v5, a121 // (op 18634) 
v_accvgpr_read v6, a122 // (op 18634) 
v_accvgpr_read v7, a123 // (op 18634) 
v_mul_f32 v14, v11, v4 // (op 18634) 
v_mul_f32 v15, v11, v5 // (op 18634) 
v_mul_f32 v16, v11, v6 // (op 18634) 
v_mul_f32 v17, v11, v7 // (op 18634) 



s_lshl_b32 s0, s4, 6 // (op 18603) 
s_mul_i32 s1, 6, s0 // (op 18603) 
s_add_u32 s1, s1, 192 // (op 18603) 

v_add_u32 v0, v13, s1 // (op 18603) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18603) Store value


v_accvgpr_read v4, a124 // (op 18639) 
v_accvgpr_read v5, a125 // (op 18639) 
v_accvgpr_read v6, a126 // (op 18639) 
v_accvgpr_read v7, a127 // (op 18639) 
v_mul_f32 v14, v11, v4 // (op 18639) 
v_mul_f32 v15, v11, v5 // (op 18639) 
v_mul_f32 v16, v11, v6 // (op 18639) 
v_mul_f32 v17, v11, v7 // (op 18639) 



s_lshl_b32 s0, s4, 6 // (op 18605) 
s_mul_i32 s1, 7, s0 // (op 18605) 
s_add_u32 s1, s1, 192 // (op 18605) 

v_add_u32 v0, v13, s1 // (op 18605) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18605) Store value


v_accvgpr_read v4, a128 // (op 18720) 
v_accvgpr_read v5, a129 // (op 18720) 
v_accvgpr_read v6, a130 // (op 18720) 
v_accvgpr_read v7, a131 // (op 18720) 
v_mul_f32 v14, v11, v4 // (op 18720) 
v_mul_f32 v15, v11, v5 // (op 18720) 
v_mul_f32 v16, v11, v6 // (op 18720) 
v_mul_f32 v17, v11, v7 // (op 18720) 



v_add_u32 v0, 256, v13 // (op 18723) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18723) Store value


v_accvgpr_read v4, a132 // (op 18741) 
v_accvgpr_read v5, a133 // (op 18741) 
v_accvgpr_read v6, a134 // (op 18741) 
v_accvgpr_read v7, a135 // (op 18741) 
v_mul_f32 v14, v11, v4 // (op 18741) 
v_mul_f32 v15, v11, v5 // (op 18741) 
v_mul_f32 v16, v11, v6 // (op 18741) 
v_mul_f32 v17, v11, v7 // (op 18741) 



s_lshl_b32 s0, s4, 6 // (op 18725) 
s_add_u32 s0, s0, 256 // (op 18725) 
v_add_u32 v0, v13, s0 // (op 18725) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18725) Store value


v_accvgpr_read v4, a136 // (op 18746) 
v_accvgpr_read v5, a137 // (op 18746) 
v_accvgpr_read v6, a138 // (op 18746) 
v_accvgpr_read v7, a139 // (op 18746) 
v_mul_f32 v14, v11, v4 // (op 18746) 
v_mul_f32 v15, v11, v5 // (op 18746) 
v_mul_f32 v16, v11, v6 // (op 18746) 
v_mul_f32 v17, v11, v7 // (op 18746) 



s_lshl_b32 s0, s4, 7 // (op 18727) 
s_add_u32 s0, s0, 256 // (op 18727) 
v_add_u32 v0, v13, s0 // (op 18727) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18727) Store value


v_accvgpr_read v4, a140 // (op 18751) 
v_accvgpr_read v5, a141 // (op 18751) 
v_accvgpr_read v6, a142 // (op 18751) 
v_accvgpr_read v7, a143 // (op 18751) 
v_mul_f32 v14, v11, v4 // (op 18751) 
v_mul_f32 v15, v11, v5 // (op 18751) 
v_mul_f32 v16, v11, v6 // (op 18751) 
v_mul_f32 v17, v11, v7 // (op 18751) 



s_lshl_b32 s0, s4, 6 // (op 18729) 
s_mul_i32 s1, 3, s0 // (op 18729) 
s_add_u32 s1, s1, 256 // (op 18729) 

v_add_u32 v0, v13, s1 // (op 18729) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18729) Store value


v_accvgpr_read v4, a144 // (op 18756) 
v_accvgpr_read v5, a145 // (op 18756) 
v_accvgpr_read v6, a146 // (op 18756) 
v_accvgpr_read v7, a147 // (op 18756) 
v_mul_f32 v14, v11, v4 // (op 18756) 
v_mul_f32 v15, v11, v5 // (op 18756) 
v_mul_f32 v16, v11, v6 // (op 18756) 
v_mul_f32 v17, v11, v7 // (op 18756) 



s_lshl_b32 s0, s4, 8 // (op 18731) 
s_add_u32 s0, s0, 256 // (op 18731) 
v_add_u32 v0, v13, s0 // (op 18731) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18731) Store value


v_accvgpr_read v4, a148 // (op 18761) 
v_accvgpr_read v5, a149 // (op 18761) 
v_accvgpr_read v6, a150 // (op 18761) 
v_accvgpr_read v7, a151 // (op 18761) 
v_mul_f32 v14, v11, v4 // (op 18761) 
v_mul_f32 v15, v11, v5 // (op 18761) 
v_mul_f32 v16, v11, v6 // (op 18761) 
v_mul_f32 v17, v11, v7 // (op 18761) 



s_lshl_b32 s0, s4, 6 // (op 18733) 
s_mul_i32 s1, 5, s0 // (op 18733) 
s_add_u32 s1, s1, 256 // (op 18733) 

v_add_u32 v0, v13, s1 // (op 18733) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18733) Store value


v_accvgpr_read v4, a152 // (op 18766) 
v_accvgpr_read v5, a153 // (op 18766) 
v_accvgpr_read v6, a154 // (op 18766) 
v_accvgpr_read v7, a155 // (op 18766) 
v_mul_f32 v14, v11, v4 // (op 18766) 
v_mul_f32 v15, v11, v5 // (op 18766) 
v_mul_f32 v16, v11, v6 // (op 18766) 
v_mul_f32 v17, v11, v7 // (op 18766) 



s_lshl_b32 s0, s4, 6 // (op 18735) 
s_mul_i32 s1, 6, s0 // (op 18735) 
s_add_u32 s1, s1, 256 // (op 18735) 

v_add_u32 v0, v13, s1 // (op 18735) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18735) Store value


v_accvgpr_read v4, a156 // (op 18771) 
v_accvgpr_read v5, a157 // (op 18771) 
v_accvgpr_read v6, a158 // (op 18771) 
v_accvgpr_read v7, a159 // (op 18771) 
v_mul_f32 v14, v11, v4 // (op 18771) 
v_mul_f32 v15, v11, v5 // (op 18771) 
v_mul_f32 v16, v11, v6 // (op 18771) 
v_mul_f32 v17, v11, v7 // (op 18771) 



s_lshl_b32 s0, s4, 6 // (op 18737) 
s_mul_i32 s1, 7, s0 // (op 18737) 
s_add_u32 s1, s1, 256 // (op 18737) 

v_add_u32 v0, v13, s1 // (op 18737) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18737) Store value


v_accvgpr_read v4, a160 // (op 18852) 
v_accvgpr_read v5, a161 // (op 18852) 
v_accvgpr_read v6, a162 // (op 18852) 
v_accvgpr_read v7, a163 // (op 18852) 
v_mul_f32 v14, v11, v4 // (op 18852) 
v_mul_f32 v15, v11, v5 // (op 18852) 
v_mul_f32 v16, v11, v6 // (op 18852) 
v_mul_f32 v17, v11, v7 // (op 18852) 



v_add_u32 v0, 320, v13 // (op 18855) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18855) Store value


v_accvgpr_read v4, a164 // (op 18873) 
v_accvgpr_read v5, a165 // (op 18873) 
v_accvgpr_read v6, a166 // (op 18873) 
v_accvgpr_read v7, a167 // (op 18873) 
v_mul_f32 v14, v11, v4 // (op 18873) 
v_mul_f32 v15, v11, v5 // (op 18873) 
v_mul_f32 v16, v11, v6 // (op 18873) 
v_mul_f32 v17, v11, v7 // (op 18873) 



s_lshl_b32 s0, s4, 6 // (op 18857) 
s_add_u32 s0, s0, 320 // (op 18857) 
v_add_u32 v0, v13, s0 // (op 18857) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18857) Store value


v_accvgpr_read v4, a168 // (op 18878) 
v_accvgpr_read v5, a169 // (op 18878) 
v_accvgpr_read v6, a170 // (op 18878) 
v_accvgpr_read v7, a171 // (op 18878) 
v_mul_f32 v14, v11, v4 // (op 18878) 
v_mul_f32 v15, v11, v5 // (op 18878) 
v_mul_f32 v16, v11, v6 // (op 18878) 
v_mul_f32 v17, v11, v7 // (op 18878) 



s_lshl_b32 s0, s4, 7 // (op 18859) 
s_add_u32 s0, s0, 320 // (op 18859) 
v_add_u32 v0, v13, s0 // (op 18859) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18859) Store value


v_accvgpr_read v4, a172 // (op 18883) 
v_accvgpr_read v5, a173 // (op 18883) 
v_accvgpr_read v6, a174 // (op 18883) 
v_accvgpr_read v7, a175 // (op 18883) 
v_mul_f32 v14, v11, v4 // (op 18883) 
v_mul_f32 v15, v11, v5 // (op 18883) 
v_mul_f32 v16, v11, v6 // (op 18883) 
v_mul_f32 v17, v11, v7 // (op 18883) 



s_lshl_b32 s0, s4, 6 // (op 18861) 
s_mul_i32 s1, 3, s0 // (op 18861) 
s_add_u32 s1, s1, 320 // (op 18861) 

v_add_u32 v0, v13, s1 // (op 18861) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18861) Store value


v_accvgpr_read v4, a176 // (op 18888) 
v_accvgpr_read v5, a177 // (op 18888) 
v_accvgpr_read v6, a178 // (op 18888) 
v_accvgpr_read v7, a179 // (op 18888) 
v_mul_f32 v14, v11, v4 // (op 18888) 
v_mul_f32 v15, v11, v5 // (op 18888) 
v_mul_f32 v16, v11, v6 // (op 18888) 
v_mul_f32 v17, v11, v7 // (op 18888) 



s_lshl_b32 s0, s4, 8 // (op 18863) 
s_add_u32 s0, s0, 320 // (op 18863) 
v_add_u32 v0, v13, s0 // (op 18863) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18863) Store value


v_accvgpr_read v4, a180 // (op 18893) 
v_accvgpr_read v5, a181 // (op 18893) 
v_accvgpr_read v6, a182 // (op 18893) 
v_accvgpr_read v7, a183 // (op 18893) 
v_mul_f32 v14, v11, v4 // (op 18893) 
v_mul_f32 v15, v11, v5 // (op 18893) 
v_mul_f32 v16, v11, v6 // (op 18893) 
v_mul_f32 v17, v11, v7 // (op 18893) 



s_lshl_b32 s0, s4, 6 // (op 18865) 
s_mul_i32 s1, 5, s0 // (op 18865) 
s_add_u32 s1, s1, 320 // (op 18865) 

v_add_u32 v0, v13, s1 // (op 18865) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18865) Store value


v_accvgpr_read v4, a184 // (op 18898) 
v_accvgpr_read v5, a185 // (op 18898) 
v_accvgpr_read v6, a186 // (op 18898) 
v_accvgpr_read v7, a187 // (op 18898) 
v_mul_f32 v14, v11, v4 // (op 18898) 
v_mul_f32 v15, v11, v5 // (op 18898) 
v_mul_f32 v16, v11, v6 // (op 18898) 
v_mul_f32 v17, v11, v7 // (op 18898) 



s_lshl_b32 s0, s4, 6 // (op 18867) 
s_mul_i32 s1, 6, s0 // (op 18867) 
s_add_u32 s1, s1, 320 // (op 18867) 

v_add_u32 v0, v13, s1 // (op 18867) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18867) Store value


v_accvgpr_read v4, a188 // (op 18903) 
v_accvgpr_read v5, a189 // (op 18903) 
v_accvgpr_read v6, a190 // (op 18903) 
v_accvgpr_read v7, a191 // (op 18903) 
v_mul_f32 v14, v11, v4 // (op 18903) 
v_mul_f32 v15, v11, v5 // (op 18903) 
v_mul_f32 v16, v11, v6 // (op 18903) 
v_mul_f32 v17, v11, v7 // (op 18903) 



s_lshl_b32 s0, s4, 6 // (op 18869) 
s_mul_i32 s1, 7, s0 // (op 18869) 
s_add_u32 s1, s1, 320 // (op 18869) 

v_add_u32 v0, v13, s1 // (op 18869) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18869) Store value


v_accvgpr_read v4, a192 // (op 18984) 
v_accvgpr_read v5, a193 // (op 18984) 
v_accvgpr_read v6, a194 // (op 18984) 
v_accvgpr_read v7, a195 // (op 18984) 
v_mul_f32 v14, v11, v4 // (op 18984) 
v_mul_f32 v15, v11, v5 // (op 18984) 
v_mul_f32 v16, v11, v6 // (op 18984) 
v_mul_f32 v17, v11, v7 // (op 18984) 



v_add_u32 v0, 384, v13 // (op 18987) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18987) Store value


v_accvgpr_read v4, a196 // (op 19005) 
v_accvgpr_read v5, a197 // (op 19005) 
v_accvgpr_read v6, a198 // (op 19005) 
v_accvgpr_read v7, a199 // (op 19005) 
v_mul_f32 v14, v11, v4 // (op 19005) 
v_mul_f32 v15, v11, v5 // (op 19005) 
v_mul_f32 v16, v11, v6 // (op 19005) 
v_mul_f32 v17, v11, v7 // (op 19005) 



s_lshl_b32 s0, s4, 6 // (op 18989) 
s_add_u32 s0, s0, 384 // (op 18989) 
v_add_u32 v0, v13, s0 // (op 18989) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18989) Store value


v_accvgpr_read v4, a200 // (op 19010) 
v_accvgpr_read v5, a201 // (op 19010) 
v_accvgpr_read v6, a202 // (op 19010) 
v_accvgpr_read v7, a203 // (op 19010) 
v_mul_f32 v14, v11, v4 // (op 19010) 
v_mul_f32 v15, v11, v5 // (op 19010) 
v_mul_f32 v16, v11, v6 // (op 19010) 
v_mul_f32 v17, v11, v7 // (op 19010) 



s_lshl_b32 s0, s4, 7 // (op 18991) 
s_add_u32 s0, s0, 384 // (op 18991) 
v_add_u32 v0, v13, s0 // (op 18991) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18991) Store value


v_accvgpr_read v4, a204 // (op 19015) 
v_accvgpr_read v5, a205 // (op 19015) 
v_accvgpr_read v6, a206 // (op 19015) 
v_accvgpr_read v7, a207 // (op 19015) 
v_mul_f32 v14, v11, v4 // (op 19015) 
v_mul_f32 v15, v11, v5 // (op 19015) 
v_mul_f32 v16, v11, v6 // (op 19015) 
v_mul_f32 v17, v11, v7 // (op 19015) 



s_lshl_b32 s0, s4, 6 // (op 18993) 
s_mul_i32 s1, 3, s0 // (op 18993) 
s_add_u32 s1, s1, 384 // (op 18993) 

v_add_u32 v0, v13, s1 // (op 18993) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18993) Store value


v_accvgpr_read v4, a208 // (op 19020) 
v_accvgpr_read v5, a209 // (op 19020) 
v_accvgpr_read v6, a210 // (op 19020) 
v_accvgpr_read v7, a211 // (op 19020) 
v_mul_f32 v14, v11, v4 // (op 19020) 
v_mul_f32 v15, v11, v5 // (op 19020) 
v_mul_f32 v16, v11, v6 // (op 19020) 
v_mul_f32 v17, v11, v7 // (op 19020) 



s_lshl_b32 s0, s4, 8 // (op 18995) 
s_add_u32 s0, s0, 384 // (op 18995) 
v_add_u32 v0, v13, s0 // (op 18995) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18995) Store value


v_accvgpr_read v4, a212 // (op 19025) 
v_accvgpr_read v5, a213 // (op 19025) 
v_accvgpr_read v6, a214 // (op 19025) 
v_accvgpr_read v7, a215 // (op 19025) 
v_mul_f32 v14, v11, v4 // (op 19025) 
v_mul_f32 v15, v11, v5 // (op 19025) 
v_mul_f32 v16, v11, v6 // (op 19025) 
v_mul_f32 v17, v11, v7 // (op 19025) 



s_lshl_b32 s0, s4, 6 // (op 18997) 
s_mul_i32 s1, 5, s0 // (op 18997) 
s_add_u32 s1, s1, 384 // (op 18997) 

v_add_u32 v0, v13, s1 // (op 18997) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18997) Store value


v_accvgpr_read v4, a216 // (op 19030) 
v_accvgpr_read v5, a217 // (op 19030) 
v_accvgpr_read v6, a218 // (op 19030) 
v_accvgpr_read v7, a219 // (op 19030) 
v_mul_f32 v14, v11, v4 // (op 19030) 
v_mul_f32 v15, v11, v5 // (op 19030) 
v_mul_f32 v16, v11, v6 // (op 19030) 
v_mul_f32 v17, v11, v7 // (op 19030) 



s_lshl_b32 s0, s4, 6 // (op 18999) 
s_mul_i32 s1, 6, s0 // (op 18999) 
s_add_u32 s1, s1, 384 // (op 18999) 

v_add_u32 v0, v13, s1 // (op 18999) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 18999) Store value


v_accvgpr_read v4, a220 // (op 19035) 
v_accvgpr_read v5, a221 // (op 19035) 
v_accvgpr_read v6, a222 // (op 19035) 
v_accvgpr_read v7, a223 // (op 19035) 
v_mul_f32 v14, v11, v4 // (op 19035) 
v_mul_f32 v15, v11, v5 // (op 19035) 
v_mul_f32 v16, v11, v6 // (op 19035) 
v_mul_f32 v17, v11, v7 // (op 19035) 



s_lshl_b32 s0, s4, 6 // (op 19001) 
s_mul_i32 s1, 7, s0 // (op 19001) 
s_add_u32 s1, s1, 384 // (op 19001) 

v_add_u32 v0, v13, s1 // (op 19001) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 19001) Store value


v_accvgpr_read v4, a224 // (op 19116) 
v_accvgpr_read v5, a225 // (op 19116) 
v_accvgpr_read v6, a226 // (op 19116) 
v_accvgpr_read v7, a227 // (op 19116) 
v_mul_f32 v14, v11, v4 // (op 19116) 
v_mul_f32 v15, v11, v5 // (op 19116) 
v_mul_f32 v16, v11, v6 // (op 19116) 
v_mul_f32 v17, v11, v7 // (op 19116) 



v_add_u32 v0, 448, v13 // (op 19119) 

buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 19119) Store value


v_accvgpr_read v4, a228 // (op 19137) 
v_accvgpr_read v5, a229 // (op 19137) 
v_accvgpr_read v6, a230 // (op 19137) 
v_accvgpr_read v7, a231 // (op 19137) 
v_mul_f32 v14, v11, v4 // (op 19137) 
v_mul_f32 v15, v11, v5 // (op 19137) 
v_mul_f32 v16, v11, v6 // (op 19137) 
v_mul_f32 v17, v11, v7 // (op 19137) 



s_lshl_b32 s0, s4, 6 // (op 19121) 
s_add_u32 s0, s0, 448 // (op 19121) 
v_add_u32 v0, v13, s0 // (op 19121) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 19121) Store value


v_accvgpr_read v4, a232 // (op 19142) 
v_accvgpr_read v5, a233 // (op 19142) 
v_accvgpr_read v6, a234 // (op 19142) 
v_accvgpr_read v7, a235 // (op 19142) 
v_mul_f32 v14, v11, v4 // (op 19142) 
v_mul_f32 v15, v11, v5 // (op 19142) 
v_mul_f32 v16, v11, v6 // (op 19142) 
v_mul_f32 v17, v11, v7 // (op 19142) 



s_lshl_b32 s0, s4, 7 // (op 19123) 
s_add_u32 s0, s0, 448 // (op 19123) 
v_add_u32 v0, v13, s0 // (op 19123) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 19123) Store value


v_accvgpr_read v4, a236 // (op 19147) 
v_accvgpr_read v5, a237 // (op 19147) 
v_accvgpr_read v6, a238 // (op 19147) 
v_accvgpr_read v7, a239 // (op 19147) 
v_mul_f32 v14, v11, v4 // (op 19147) 
v_mul_f32 v15, v11, v5 // (op 19147) 
v_mul_f32 v16, v11, v6 // (op 19147) 
v_mul_f32 v17, v11, v7 // (op 19147) 



s_lshl_b32 s0, s4, 6 // (op 19125) 
s_mul_i32 s1, 3, s0 // (op 19125) 
s_add_u32 s1, s1, 448 // (op 19125) 

v_add_u32 v0, v13, s1 // (op 19125) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 19125) Store value


v_accvgpr_read v4, a240 // (op 19152) 
v_accvgpr_read v5, a241 // (op 19152) 
v_accvgpr_read v6, a242 // (op 19152) 
v_accvgpr_read v7, a243 // (op 19152) 
v_mul_f32 v14, v11, v4 // (op 19152) 
v_mul_f32 v15, v11, v5 // (op 19152) 
v_mul_f32 v16, v11, v6 // (op 19152) 
v_mul_f32 v17, v11, v7 // (op 19152) 



s_lshl_b32 s0, s4, 8 // (op 19127) 
s_add_u32 s0, s0, 448 // (op 19127) 
v_add_u32 v0, v13, s0 // (op 19127) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 19127) Store value


v_accvgpr_read v4, a244 // (op 19157) 
v_accvgpr_read v5, a245 // (op 19157) 
v_accvgpr_read v6, a246 // (op 19157) 
v_accvgpr_read v7, a247 // (op 19157) 
v_mul_f32 v14, v11, v4 // (op 19157) 
v_mul_f32 v15, v11, v5 // (op 19157) 
v_mul_f32 v16, v11, v6 // (op 19157) 
v_mul_f32 v17, v11, v7 // (op 19157) 



s_lshl_b32 s0, s4, 6 // (op 19129) 
s_mul_i32 s1, 5, s0 // (op 19129) 
s_add_u32 s1, s1, 448 // (op 19129) 

v_add_u32 v0, v13, s1 // (op 19129) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 19129) Store value


v_accvgpr_read v4, a248 // (op 19162) 
v_accvgpr_read v5, a249 // (op 19162) 
v_accvgpr_read v6, a250 // (op 19162) 
v_accvgpr_read v7, a251 // (op 19162) 
v_mul_f32 v14, v11, v4 // (op 19162) 
v_mul_f32 v15, v11, v5 // (op 19162) 
v_mul_f32 v16, v11, v6 // (op 19162) 
v_mul_f32 v17, v11, v7 // (op 19162) 



s_lshl_b32 s0, s4, 6 // (op 19131) 
s_mul_i32 s1, 6, s0 // (op 19131) 
s_add_u32 s1, s1, 448 // (op 19131) 

v_add_u32 v0, v13, s1 // (op 19131) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 19131) Store value


v_accvgpr_read v4, a252 // (op 19167) 
v_accvgpr_read v5, a253 // (op 19167) 
v_accvgpr_read v6, a254 // (op 19167) 
v_accvgpr_read v7, a255 // (op 19167) 
v_mul_f32 v14, v11, v4 // (op 19167) 
v_mul_f32 v15, v11, v5 // (op 19167) 
v_mul_f32 v16, v11, v6 // (op 19167) 
v_mul_f32 v17, v11, v7 // (op 19167) 



s_lshl_b32 s0, s4, 6 // (op 19133) 
s_mul_i32 s1, 7, s0 // (op 19133) 
s_add_u32 s1, s1, 448 // (op 19133) 

v_add_u32 v0, v13, s1 // (op 19133) 


buffer_store_dwordx4 v[14:17], v0, s[44:47], 0 offen // (op 19133) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalBottom_0_beta_114
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalFalse_0_beta_114:


v_add_u32 v0, 448, v12 // (op 20567) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20567) Load value

v_accvgpr_read v14, a224 // (op 20547) 
v_accvgpr_read v15, a225 // (op 20547) 
v_accvgpr_read v16, a226 // (op 20547) 
v_accvgpr_read v17, a227 // (op 20547) 
v_mul_f32 v18, v11, v14 // (op 20547) 
v_mul_f32 v19, v11, v15 // (op 20547) 
v_mul_f32 v20, v11, v16 // (op 20547) 
v_mul_f32 v21, v11, v17 // (op 20547) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20566) 
v_mul_f32 v15, v9, v5 // (op 20566) 
v_mul_f32 v16, v9, v6 // (op 20566) 
v_mul_f32 v17, v9, v7 // (op 20566) 

v_add_f32 v4, v14, v18 // (op 20548) 
v_add_f32 v5, v15, v19 // (op 20548) 
v_add_f32 v6, v16, v20 // (op 20548) 
v_add_f32 v7, v17, v21 // (op 20548) 




v_add_u32 v0, 448, v13 // (op 20550) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20550) Store value




s_lshl_b32 s0, s2, 6 // (op 20571) 
s_add_u32 s0, s0, 448 // (op 20571) 
v_add_u32 v0, v12, s0 // (op 20571) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20571) Load value

v_accvgpr_read v14, a228 // (op 20597) 
v_accvgpr_read v15, a229 // (op 20597) 
v_accvgpr_read v16, a230 // (op 20597) 
v_accvgpr_read v17, a231 // (op 20597) 
v_mul_f32 v18, v11, v14 // (op 20597) 
v_mul_f32 v19, v11, v15 // (op 20597) 
v_mul_f32 v20, v11, v16 // (op 20597) 
v_mul_f32 v21, v11, v17 // (op 20597) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20569) 
v_mul_f32 v15, v9, v5 // (op 20569) 
v_mul_f32 v16, v9, v6 // (op 20569) 
v_mul_f32 v17, v9, v7 // (op 20569) 

v_add_f32 v4, v14, v18 // (op 20570) 
v_add_f32 v5, v15, v19 // (op 20570) 
v_add_f32 v6, v16, v20 // (op 20570) 
v_add_f32 v7, v17, v21 // (op 20570) 




s_lshl_b32 s0, s4, 6 // (op 20552) 
s_add_u32 s0, s0, 448 // (op 20552) 
v_add_u32 v0, v13, s0 // (op 20552) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20552) Store value




s_lshl_b32 s0, s2, 7 // (op 20575) 
s_add_u32 s0, s0, 448 // (op 20575) 
v_add_u32 v0, v12, s0 // (op 20575) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20575) Load value

v_accvgpr_read v14, a232 // (op 20599) 
v_accvgpr_read v15, a233 // (op 20599) 
v_accvgpr_read v16, a234 // (op 20599) 
v_accvgpr_read v17, a235 // (op 20599) 
v_mul_f32 v18, v11, v14 // (op 20599) 
v_mul_f32 v19, v11, v15 // (op 20599) 
v_mul_f32 v20, v11, v16 // (op 20599) 
v_mul_f32 v21, v11, v17 // (op 20599) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20573) 
v_mul_f32 v15, v9, v5 // (op 20573) 
v_mul_f32 v16, v9, v6 // (op 20573) 
v_mul_f32 v17, v9, v7 // (op 20573) 

v_add_f32 v4, v14, v18 // (op 20574) 
v_add_f32 v5, v15, v19 // (op 20574) 
v_add_f32 v6, v16, v20 // (op 20574) 
v_add_f32 v7, v17, v21 // (op 20574) 




s_lshl_b32 s0, s4, 7 // (op 20554) 
s_add_u32 s0, s0, 448 // (op 20554) 
v_add_u32 v0, v13, s0 // (op 20554) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20554) Store value




s_lshl_b32 s0, s2, 6 // (op 20579) 
s_mul_i32 s1, 3, s0 // (op 20579) 
s_add_u32 s1, s1, 448 // (op 20579) 

v_add_u32 v0, v12, s1 // (op 20579) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20579) Load value

v_accvgpr_read v14, a236 // (op 20601) 
v_accvgpr_read v15, a237 // (op 20601) 
v_accvgpr_read v16, a238 // (op 20601) 
v_accvgpr_read v17, a239 // (op 20601) 
v_mul_f32 v18, v11, v14 // (op 20601) 
v_mul_f32 v19, v11, v15 // (op 20601) 
v_mul_f32 v20, v11, v16 // (op 20601) 
v_mul_f32 v21, v11, v17 // (op 20601) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20577) 
v_mul_f32 v15, v9, v5 // (op 20577) 
v_mul_f32 v16, v9, v6 // (op 20577) 
v_mul_f32 v17, v9, v7 // (op 20577) 

v_add_f32 v4, v14, v18 // (op 20578) 
v_add_f32 v5, v15, v19 // (op 20578) 
v_add_f32 v6, v16, v20 // (op 20578) 
v_add_f32 v7, v17, v21 // (op 20578) 




s_lshl_b32 s0, s4, 6 // (op 20556) 
s_mul_i32 s1, 3, s0 // (op 20556) 
s_add_u32 s1, s1, 448 // (op 20556) 

v_add_u32 v0, v13, s1 // (op 20556) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20556) Store value




s_lshl_b32 s0, s2, 8 // (op 20583) 
s_add_u32 s0, s0, 448 // (op 20583) 
v_add_u32 v0, v12, s0 // (op 20583) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20583) Load value

v_accvgpr_read v14, a240 // (op 20603) 
v_accvgpr_read v15, a241 // (op 20603) 
v_accvgpr_read v16, a242 // (op 20603) 
v_accvgpr_read v17, a243 // (op 20603) 
v_mul_f32 v18, v11, v14 // (op 20603) 
v_mul_f32 v19, v11, v15 // (op 20603) 
v_mul_f32 v20, v11, v16 // (op 20603) 
v_mul_f32 v21, v11, v17 // (op 20603) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20581) 
v_mul_f32 v15, v9, v5 // (op 20581) 
v_mul_f32 v16, v9, v6 // (op 20581) 
v_mul_f32 v17, v9, v7 // (op 20581) 

v_add_f32 v4, v14, v18 // (op 20582) 
v_add_f32 v5, v15, v19 // (op 20582) 
v_add_f32 v6, v16, v20 // (op 20582) 
v_add_f32 v7, v17, v21 // (op 20582) 




s_lshl_b32 s0, s4, 8 // (op 20558) 
s_add_u32 s0, s0, 448 // (op 20558) 
v_add_u32 v0, v13, s0 // (op 20558) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20558) Store value




s_lshl_b32 s0, s2, 6 // (op 20587) 
s_mul_i32 s1, 5, s0 // (op 20587) 
s_add_u32 s1, s1, 448 // (op 20587) 

v_add_u32 v0, v12, s1 // (op 20587) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20587) Load value

v_accvgpr_read v14, a244 // (op 20605) 
v_accvgpr_read v15, a245 // (op 20605) 
v_accvgpr_read v16, a246 // (op 20605) 
v_accvgpr_read v17, a247 // (op 20605) 
v_mul_f32 v18, v11, v14 // (op 20605) 
v_mul_f32 v19, v11, v15 // (op 20605) 
v_mul_f32 v20, v11, v16 // (op 20605) 
v_mul_f32 v21, v11, v17 // (op 20605) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20585) 
v_mul_f32 v15, v9, v5 // (op 20585) 
v_mul_f32 v16, v9, v6 // (op 20585) 
v_mul_f32 v17, v9, v7 // (op 20585) 

v_add_f32 v4, v14, v18 // (op 20586) 
v_add_f32 v5, v15, v19 // (op 20586) 
v_add_f32 v6, v16, v20 // (op 20586) 
v_add_f32 v7, v17, v21 // (op 20586) 




s_lshl_b32 s0, s4, 6 // (op 20560) 
s_mul_i32 s1, 5, s0 // (op 20560) 
s_add_u32 s1, s1, 448 // (op 20560) 

v_add_u32 v0, v13, s1 // (op 20560) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20560) Store value




s_lshl_b32 s0, s2, 6 // (op 20591) 
s_mul_i32 s1, 6, s0 // (op 20591) 
s_add_u32 s1, s1, 448 // (op 20591) 

v_add_u32 v0, v12, s1 // (op 20591) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20591) Load value

v_accvgpr_read v14, a248 // (op 20607) 
v_accvgpr_read v15, a249 // (op 20607) 
v_accvgpr_read v16, a250 // (op 20607) 
v_accvgpr_read v17, a251 // (op 20607) 
v_mul_f32 v18, v11, v14 // (op 20607) 
v_mul_f32 v19, v11, v15 // (op 20607) 
v_mul_f32 v20, v11, v16 // (op 20607) 
v_mul_f32 v21, v11, v17 // (op 20607) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20589) 
v_mul_f32 v15, v9, v5 // (op 20589) 
v_mul_f32 v16, v9, v6 // (op 20589) 
v_mul_f32 v17, v9, v7 // (op 20589) 

v_add_f32 v4, v14, v18 // (op 20590) 
v_add_f32 v5, v15, v19 // (op 20590) 
v_add_f32 v6, v16, v20 // (op 20590) 
v_add_f32 v7, v17, v21 // (op 20590) 




s_lshl_b32 s0, s4, 6 // (op 20562) 
s_mul_i32 s1, 6, s0 // (op 20562) 
s_add_u32 s1, s1, 448 // (op 20562) 

v_add_u32 v0, v13, s1 // (op 20562) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20562) Store value




s_lshl_b32 s0, s2, 6 // (op 20595) 
s_mul_i32 s1, 7, s0 // (op 20595) 
s_add_u32 s1, s1, 448 // (op 20595) 

v_add_u32 v0, v12, s1 // (op 20595) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20595) Load value

v_accvgpr_read v14, a252 // (op 20609) 
v_accvgpr_read v15, a253 // (op 20609) 
v_accvgpr_read v16, a254 // (op 20609) 
v_accvgpr_read v17, a255 // (op 20609) 
v_mul_f32 v18, v11, v14 // (op 20609) 
v_mul_f32 v19, v11, v15 // (op 20609) 
v_mul_f32 v20, v11, v16 // (op 20609) 
v_mul_f32 v21, v11, v17 // (op 20609) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20593) 
v_mul_f32 v15, v9, v5 // (op 20593) 
v_mul_f32 v16, v9, v6 // (op 20593) 
v_mul_f32 v17, v9, v7 // (op 20593) 

v_add_f32 v4, v14, v18 // (op 20594) 
v_add_f32 v5, v15, v19 // (op 20594) 
v_add_f32 v6, v16, v20 // (op 20594) 
v_add_f32 v7, v17, v21 // (op 20594) 




s_lshl_b32 s0, s4, 6 // (op 20564) 
s_mul_i32 s1, 7, s0 // (op 20564) 
s_add_u32 s1, s1, 448 // (op 20564) 

v_add_u32 v0, v13, s1 // (op 20564) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20564) Store value




v_add_u32 v0, 384, v12 // (op 20412) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20412) Load value

v_accvgpr_read v14, a192 // (op 20392) 
v_accvgpr_read v15, a193 // (op 20392) 
v_accvgpr_read v16, a194 // (op 20392) 
v_accvgpr_read v17, a195 // (op 20392) 
v_mul_f32 v18, v11, v14 // (op 20392) 
v_mul_f32 v19, v11, v15 // (op 20392) 
v_mul_f32 v20, v11, v16 // (op 20392) 
v_mul_f32 v21, v11, v17 // (op 20392) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20411) 
v_mul_f32 v15, v9, v5 // (op 20411) 
v_mul_f32 v16, v9, v6 // (op 20411) 
v_mul_f32 v17, v9, v7 // (op 20411) 

v_add_f32 v4, v14, v18 // (op 20393) 
v_add_f32 v5, v15, v19 // (op 20393) 
v_add_f32 v6, v16, v20 // (op 20393) 
v_add_f32 v7, v17, v21 // (op 20393) 




v_add_u32 v0, 384, v13 // (op 20395) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20395) Store value




s_lshl_b32 s0, s2, 6 // (op 20416) 
s_add_u32 s0, s0, 384 // (op 20416) 
v_add_u32 v0, v12, s0 // (op 20416) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20416) Load value

v_accvgpr_read v14, a196 // (op 20442) 
v_accvgpr_read v15, a197 // (op 20442) 
v_accvgpr_read v16, a198 // (op 20442) 
v_accvgpr_read v17, a199 // (op 20442) 
v_mul_f32 v18, v11, v14 // (op 20442) 
v_mul_f32 v19, v11, v15 // (op 20442) 
v_mul_f32 v20, v11, v16 // (op 20442) 
v_mul_f32 v21, v11, v17 // (op 20442) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20414) 
v_mul_f32 v15, v9, v5 // (op 20414) 
v_mul_f32 v16, v9, v6 // (op 20414) 
v_mul_f32 v17, v9, v7 // (op 20414) 

v_add_f32 v4, v14, v18 // (op 20415) 
v_add_f32 v5, v15, v19 // (op 20415) 
v_add_f32 v6, v16, v20 // (op 20415) 
v_add_f32 v7, v17, v21 // (op 20415) 




s_lshl_b32 s0, s4, 6 // (op 20397) 
s_add_u32 s0, s0, 384 // (op 20397) 
v_add_u32 v0, v13, s0 // (op 20397) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20397) Store value




s_lshl_b32 s0, s2, 7 // (op 20420) 
s_add_u32 s0, s0, 384 // (op 20420) 
v_add_u32 v0, v12, s0 // (op 20420) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20420) Load value

v_accvgpr_read v14, a200 // (op 20444) 
v_accvgpr_read v15, a201 // (op 20444) 
v_accvgpr_read v16, a202 // (op 20444) 
v_accvgpr_read v17, a203 // (op 20444) 
v_mul_f32 v18, v11, v14 // (op 20444) 
v_mul_f32 v19, v11, v15 // (op 20444) 
v_mul_f32 v20, v11, v16 // (op 20444) 
v_mul_f32 v21, v11, v17 // (op 20444) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20418) 
v_mul_f32 v15, v9, v5 // (op 20418) 
v_mul_f32 v16, v9, v6 // (op 20418) 
v_mul_f32 v17, v9, v7 // (op 20418) 

v_add_f32 v4, v14, v18 // (op 20419) 
v_add_f32 v5, v15, v19 // (op 20419) 
v_add_f32 v6, v16, v20 // (op 20419) 
v_add_f32 v7, v17, v21 // (op 20419) 




s_lshl_b32 s0, s4, 7 // (op 20399) 
s_add_u32 s0, s0, 384 // (op 20399) 
v_add_u32 v0, v13, s0 // (op 20399) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20399) Store value




s_lshl_b32 s0, s2, 6 // (op 20424) 
s_mul_i32 s1, 3, s0 // (op 20424) 
s_add_u32 s1, s1, 384 // (op 20424) 

v_add_u32 v0, v12, s1 // (op 20424) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20424) Load value

v_accvgpr_read v14, a204 // (op 20446) 
v_accvgpr_read v15, a205 // (op 20446) 
v_accvgpr_read v16, a206 // (op 20446) 
v_accvgpr_read v17, a207 // (op 20446) 
v_mul_f32 v18, v11, v14 // (op 20446) 
v_mul_f32 v19, v11, v15 // (op 20446) 
v_mul_f32 v20, v11, v16 // (op 20446) 
v_mul_f32 v21, v11, v17 // (op 20446) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20422) 
v_mul_f32 v15, v9, v5 // (op 20422) 
v_mul_f32 v16, v9, v6 // (op 20422) 
v_mul_f32 v17, v9, v7 // (op 20422) 

v_add_f32 v4, v14, v18 // (op 20423) 
v_add_f32 v5, v15, v19 // (op 20423) 
v_add_f32 v6, v16, v20 // (op 20423) 
v_add_f32 v7, v17, v21 // (op 20423) 




s_lshl_b32 s0, s4, 6 // (op 20401) 
s_mul_i32 s1, 3, s0 // (op 20401) 
s_add_u32 s1, s1, 384 // (op 20401) 

v_add_u32 v0, v13, s1 // (op 20401) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20401) Store value




s_lshl_b32 s0, s2, 8 // (op 20428) 
s_add_u32 s0, s0, 384 // (op 20428) 
v_add_u32 v0, v12, s0 // (op 20428) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20428) Load value

v_accvgpr_read v14, a208 // (op 20448) 
v_accvgpr_read v15, a209 // (op 20448) 
v_accvgpr_read v16, a210 // (op 20448) 
v_accvgpr_read v17, a211 // (op 20448) 
v_mul_f32 v18, v11, v14 // (op 20448) 
v_mul_f32 v19, v11, v15 // (op 20448) 
v_mul_f32 v20, v11, v16 // (op 20448) 
v_mul_f32 v21, v11, v17 // (op 20448) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20426) 
v_mul_f32 v15, v9, v5 // (op 20426) 
v_mul_f32 v16, v9, v6 // (op 20426) 
v_mul_f32 v17, v9, v7 // (op 20426) 

v_add_f32 v4, v14, v18 // (op 20427) 
v_add_f32 v5, v15, v19 // (op 20427) 
v_add_f32 v6, v16, v20 // (op 20427) 
v_add_f32 v7, v17, v21 // (op 20427) 




s_lshl_b32 s0, s4, 8 // (op 20403) 
s_add_u32 s0, s0, 384 // (op 20403) 
v_add_u32 v0, v13, s0 // (op 20403) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20403) Store value




s_lshl_b32 s0, s2, 6 // (op 20432) 
s_mul_i32 s1, 5, s0 // (op 20432) 
s_add_u32 s1, s1, 384 // (op 20432) 

v_add_u32 v0, v12, s1 // (op 20432) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20432) Load value

v_accvgpr_read v14, a212 // (op 20450) 
v_accvgpr_read v15, a213 // (op 20450) 
v_accvgpr_read v16, a214 // (op 20450) 
v_accvgpr_read v17, a215 // (op 20450) 
v_mul_f32 v18, v11, v14 // (op 20450) 
v_mul_f32 v19, v11, v15 // (op 20450) 
v_mul_f32 v20, v11, v16 // (op 20450) 
v_mul_f32 v21, v11, v17 // (op 20450) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20430) 
v_mul_f32 v15, v9, v5 // (op 20430) 
v_mul_f32 v16, v9, v6 // (op 20430) 
v_mul_f32 v17, v9, v7 // (op 20430) 

v_add_f32 v4, v14, v18 // (op 20431) 
v_add_f32 v5, v15, v19 // (op 20431) 
v_add_f32 v6, v16, v20 // (op 20431) 
v_add_f32 v7, v17, v21 // (op 20431) 




s_lshl_b32 s0, s4, 6 // (op 20405) 
s_mul_i32 s1, 5, s0 // (op 20405) 
s_add_u32 s1, s1, 384 // (op 20405) 

v_add_u32 v0, v13, s1 // (op 20405) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20405) Store value




s_lshl_b32 s0, s2, 6 // (op 20436) 
s_mul_i32 s1, 6, s0 // (op 20436) 
s_add_u32 s1, s1, 384 // (op 20436) 

v_add_u32 v0, v12, s1 // (op 20436) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20436) Load value

v_accvgpr_read v14, a216 // (op 20452) 
v_accvgpr_read v15, a217 // (op 20452) 
v_accvgpr_read v16, a218 // (op 20452) 
v_accvgpr_read v17, a219 // (op 20452) 
v_mul_f32 v18, v11, v14 // (op 20452) 
v_mul_f32 v19, v11, v15 // (op 20452) 
v_mul_f32 v20, v11, v16 // (op 20452) 
v_mul_f32 v21, v11, v17 // (op 20452) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20434) 
v_mul_f32 v15, v9, v5 // (op 20434) 
v_mul_f32 v16, v9, v6 // (op 20434) 
v_mul_f32 v17, v9, v7 // (op 20434) 

v_add_f32 v4, v14, v18 // (op 20435) 
v_add_f32 v5, v15, v19 // (op 20435) 
v_add_f32 v6, v16, v20 // (op 20435) 
v_add_f32 v7, v17, v21 // (op 20435) 




s_lshl_b32 s0, s4, 6 // (op 20407) 
s_mul_i32 s1, 6, s0 // (op 20407) 
s_add_u32 s1, s1, 384 // (op 20407) 

v_add_u32 v0, v13, s1 // (op 20407) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20407) Store value




s_lshl_b32 s0, s2, 6 // (op 20440) 
s_mul_i32 s1, 7, s0 // (op 20440) 
s_add_u32 s1, s1, 384 // (op 20440) 

v_add_u32 v0, v12, s1 // (op 20440) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20440) Load value

v_accvgpr_read v14, a220 // (op 20454) 
v_accvgpr_read v15, a221 // (op 20454) 
v_accvgpr_read v16, a222 // (op 20454) 
v_accvgpr_read v17, a223 // (op 20454) 
v_mul_f32 v18, v11, v14 // (op 20454) 
v_mul_f32 v19, v11, v15 // (op 20454) 
v_mul_f32 v20, v11, v16 // (op 20454) 
v_mul_f32 v21, v11, v17 // (op 20454) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20438) 
v_mul_f32 v15, v9, v5 // (op 20438) 
v_mul_f32 v16, v9, v6 // (op 20438) 
v_mul_f32 v17, v9, v7 // (op 20438) 

v_add_f32 v4, v14, v18 // (op 20439) 
v_add_f32 v5, v15, v19 // (op 20439) 
v_add_f32 v6, v16, v20 // (op 20439) 
v_add_f32 v7, v17, v21 // (op 20439) 




s_lshl_b32 s0, s4, 6 // (op 20409) 
s_mul_i32 s1, 7, s0 // (op 20409) 
s_add_u32 s1, s1, 384 // (op 20409) 

v_add_u32 v0, v13, s1 // (op 20409) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20409) Store value




v_add_u32 v0, 320, v12 // (op 20257) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20257) Load value

v_accvgpr_read v14, a160 // (op 20237) 
v_accvgpr_read v15, a161 // (op 20237) 
v_accvgpr_read v16, a162 // (op 20237) 
v_accvgpr_read v17, a163 // (op 20237) 
v_mul_f32 v18, v11, v14 // (op 20237) 
v_mul_f32 v19, v11, v15 // (op 20237) 
v_mul_f32 v20, v11, v16 // (op 20237) 
v_mul_f32 v21, v11, v17 // (op 20237) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20256) 
v_mul_f32 v15, v9, v5 // (op 20256) 
v_mul_f32 v16, v9, v6 // (op 20256) 
v_mul_f32 v17, v9, v7 // (op 20256) 

v_add_f32 v4, v14, v18 // (op 20238) 
v_add_f32 v5, v15, v19 // (op 20238) 
v_add_f32 v6, v16, v20 // (op 20238) 
v_add_f32 v7, v17, v21 // (op 20238) 




v_add_u32 v0, 320, v13 // (op 20240) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20240) Store value




s_lshl_b32 s0, s2, 6 // (op 20261) 
s_add_u32 s0, s0, 320 // (op 20261) 
v_add_u32 v0, v12, s0 // (op 20261) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20261) Load value

v_accvgpr_read v14, a164 // (op 20287) 
v_accvgpr_read v15, a165 // (op 20287) 
v_accvgpr_read v16, a166 // (op 20287) 
v_accvgpr_read v17, a167 // (op 20287) 
v_mul_f32 v18, v11, v14 // (op 20287) 
v_mul_f32 v19, v11, v15 // (op 20287) 
v_mul_f32 v20, v11, v16 // (op 20287) 
v_mul_f32 v21, v11, v17 // (op 20287) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20259) 
v_mul_f32 v15, v9, v5 // (op 20259) 
v_mul_f32 v16, v9, v6 // (op 20259) 
v_mul_f32 v17, v9, v7 // (op 20259) 

v_add_f32 v4, v14, v18 // (op 20260) 
v_add_f32 v5, v15, v19 // (op 20260) 
v_add_f32 v6, v16, v20 // (op 20260) 
v_add_f32 v7, v17, v21 // (op 20260) 




s_lshl_b32 s0, s4, 6 // (op 20242) 
s_add_u32 s0, s0, 320 // (op 20242) 
v_add_u32 v0, v13, s0 // (op 20242) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20242) Store value




s_lshl_b32 s0, s2, 7 // (op 20265) 
s_add_u32 s0, s0, 320 // (op 20265) 
v_add_u32 v0, v12, s0 // (op 20265) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20265) Load value

v_accvgpr_read v14, a168 // (op 20289) 
v_accvgpr_read v15, a169 // (op 20289) 
v_accvgpr_read v16, a170 // (op 20289) 
v_accvgpr_read v17, a171 // (op 20289) 
v_mul_f32 v18, v11, v14 // (op 20289) 
v_mul_f32 v19, v11, v15 // (op 20289) 
v_mul_f32 v20, v11, v16 // (op 20289) 
v_mul_f32 v21, v11, v17 // (op 20289) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20263) 
v_mul_f32 v15, v9, v5 // (op 20263) 
v_mul_f32 v16, v9, v6 // (op 20263) 
v_mul_f32 v17, v9, v7 // (op 20263) 

v_add_f32 v4, v14, v18 // (op 20264) 
v_add_f32 v5, v15, v19 // (op 20264) 
v_add_f32 v6, v16, v20 // (op 20264) 
v_add_f32 v7, v17, v21 // (op 20264) 




s_lshl_b32 s0, s4, 7 // (op 20244) 
s_add_u32 s0, s0, 320 // (op 20244) 
v_add_u32 v0, v13, s0 // (op 20244) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20244) Store value




s_lshl_b32 s0, s2, 6 // (op 20269) 
s_mul_i32 s1, 3, s0 // (op 20269) 
s_add_u32 s1, s1, 320 // (op 20269) 

v_add_u32 v0, v12, s1 // (op 20269) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20269) Load value

v_accvgpr_read v14, a172 // (op 20291) 
v_accvgpr_read v15, a173 // (op 20291) 
v_accvgpr_read v16, a174 // (op 20291) 
v_accvgpr_read v17, a175 // (op 20291) 
v_mul_f32 v18, v11, v14 // (op 20291) 
v_mul_f32 v19, v11, v15 // (op 20291) 
v_mul_f32 v20, v11, v16 // (op 20291) 
v_mul_f32 v21, v11, v17 // (op 20291) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20267) 
v_mul_f32 v15, v9, v5 // (op 20267) 
v_mul_f32 v16, v9, v6 // (op 20267) 
v_mul_f32 v17, v9, v7 // (op 20267) 

v_add_f32 v4, v14, v18 // (op 20268) 
v_add_f32 v5, v15, v19 // (op 20268) 
v_add_f32 v6, v16, v20 // (op 20268) 
v_add_f32 v7, v17, v21 // (op 20268) 




s_lshl_b32 s0, s4, 6 // (op 20246) 
s_mul_i32 s1, 3, s0 // (op 20246) 
s_add_u32 s1, s1, 320 // (op 20246) 

v_add_u32 v0, v13, s1 // (op 20246) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20246) Store value




s_lshl_b32 s0, s2, 8 // (op 20273) 
s_add_u32 s0, s0, 320 // (op 20273) 
v_add_u32 v0, v12, s0 // (op 20273) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20273) Load value

v_accvgpr_read v14, a176 // (op 20293) 
v_accvgpr_read v15, a177 // (op 20293) 
v_accvgpr_read v16, a178 // (op 20293) 
v_accvgpr_read v17, a179 // (op 20293) 
v_mul_f32 v18, v11, v14 // (op 20293) 
v_mul_f32 v19, v11, v15 // (op 20293) 
v_mul_f32 v20, v11, v16 // (op 20293) 
v_mul_f32 v21, v11, v17 // (op 20293) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20271) 
v_mul_f32 v15, v9, v5 // (op 20271) 
v_mul_f32 v16, v9, v6 // (op 20271) 
v_mul_f32 v17, v9, v7 // (op 20271) 

v_add_f32 v4, v14, v18 // (op 20272) 
v_add_f32 v5, v15, v19 // (op 20272) 
v_add_f32 v6, v16, v20 // (op 20272) 
v_add_f32 v7, v17, v21 // (op 20272) 




s_lshl_b32 s0, s4, 8 // (op 20248) 
s_add_u32 s0, s0, 320 // (op 20248) 
v_add_u32 v0, v13, s0 // (op 20248) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20248) Store value




s_lshl_b32 s0, s2, 6 // (op 20277) 
s_mul_i32 s1, 5, s0 // (op 20277) 
s_add_u32 s1, s1, 320 // (op 20277) 

v_add_u32 v0, v12, s1 // (op 20277) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20277) Load value

v_accvgpr_read v14, a180 // (op 20295) 
v_accvgpr_read v15, a181 // (op 20295) 
v_accvgpr_read v16, a182 // (op 20295) 
v_accvgpr_read v17, a183 // (op 20295) 
v_mul_f32 v18, v11, v14 // (op 20295) 
v_mul_f32 v19, v11, v15 // (op 20295) 
v_mul_f32 v20, v11, v16 // (op 20295) 
v_mul_f32 v21, v11, v17 // (op 20295) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20275) 
v_mul_f32 v15, v9, v5 // (op 20275) 
v_mul_f32 v16, v9, v6 // (op 20275) 
v_mul_f32 v17, v9, v7 // (op 20275) 

v_add_f32 v4, v14, v18 // (op 20276) 
v_add_f32 v5, v15, v19 // (op 20276) 
v_add_f32 v6, v16, v20 // (op 20276) 
v_add_f32 v7, v17, v21 // (op 20276) 




s_lshl_b32 s0, s4, 6 // (op 20250) 
s_mul_i32 s1, 5, s0 // (op 20250) 
s_add_u32 s1, s1, 320 // (op 20250) 

v_add_u32 v0, v13, s1 // (op 20250) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20250) Store value




s_lshl_b32 s0, s2, 6 // (op 20281) 
s_mul_i32 s1, 6, s0 // (op 20281) 
s_add_u32 s1, s1, 320 // (op 20281) 

v_add_u32 v0, v12, s1 // (op 20281) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20281) Load value

v_accvgpr_read v14, a184 // (op 20297) 
v_accvgpr_read v15, a185 // (op 20297) 
v_accvgpr_read v16, a186 // (op 20297) 
v_accvgpr_read v17, a187 // (op 20297) 
v_mul_f32 v18, v11, v14 // (op 20297) 
v_mul_f32 v19, v11, v15 // (op 20297) 
v_mul_f32 v20, v11, v16 // (op 20297) 
v_mul_f32 v21, v11, v17 // (op 20297) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20279) 
v_mul_f32 v15, v9, v5 // (op 20279) 
v_mul_f32 v16, v9, v6 // (op 20279) 
v_mul_f32 v17, v9, v7 // (op 20279) 

v_add_f32 v4, v14, v18 // (op 20280) 
v_add_f32 v5, v15, v19 // (op 20280) 
v_add_f32 v6, v16, v20 // (op 20280) 
v_add_f32 v7, v17, v21 // (op 20280) 




s_lshl_b32 s0, s4, 6 // (op 20252) 
s_mul_i32 s1, 6, s0 // (op 20252) 
s_add_u32 s1, s1, 320 // (op 20252) 

v_add_u32 v0, v13, s1 // (op 20252) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20252) Store value




s_lshl_b32 s0, s2, 6 // (op 20285) 
s_mul_i32 s1, 7, s0 // (op 20285) 
s_add_u32 s1, s1, 320 // (op 20285) 

v_add_u32 v0, v12, s1 // (op 20285) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20285) Load value

v_accvgpr_read v14, a188 // (op 20299) 
v_accvgpr_read v15, a189 // (op 20299) 
v_accvgpr_read v16, a190 // (op 20299) 
v_accvgpr_read v17, a191 // (op 20299) 
v_mul_f32 v18, v11, v14 // (op 20299) 
v_mul_f32 v19, v11, v15 // (op 20299) 
v_mul_f32 v20, v11, v16 // (op 20299) 
v_mul_f32 v21, v11, v17 // (op 20299) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20283) 
v_mul_f32 v15, v9, v5 // (op 20283) 
v_mul_f32 v16, v9, v6 // (op 20283) 
v_mul_f32 v17, v9, v7 // (op 20283) 

v_add_f32 v4, v14, v18 // (op 20284) 
v_add_f32 v5, v15, v19 // (op 20284) 
v_add_f32 v6, v16, v20 // (op 20284) 
v_add_f32 v7, v17, v21 // (op 20284) 




s_lshl_b32 s0, s4, 6 // (op 20254) 
s_mul_i32 s1, 7, s0 // (op 20254) 
s_add_u32 s1, s1, 320 // (op 20254) 

v_add_u32 v0, v13, s1 // (op 20254) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20254) Store value




v_add_u32 v0, 256, v12 // (op 20102) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20102) Load value

v_accvgpr_read v14, a128 // (op 20082) 
v_accvgpr_read v15, a129 // (op 20082) 
v_accvgpr_read v16, a130 // (op 20082) 
v_accvgpr_read v17, a131 // (op 20082) 
v_mul_f32 v18, v11, v14 // (op 20082) 
v_mul_f32 v19, v11, v15 // (op 20082) 
v_mul_f32 v20, v11, v16 // (op 20082) 
v_mul_f32 v21, v11, v17 // (op 20082) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20101) 
v_mul_f32 v15, v9, v5 // (op 20101) 
v_mul_f32 v16, v9, v6 // (op 20101) 
v_mul_f32 v17, v9, v7 // (op 20101) 

v_add_f32 v4, v14, v18 // (op 20083) 
v_add_f32 v5, v15, v19 // (op 20083) 
v_add_f32 v6, v16, v20 // (op 20083) 
v_add_f32 v7, v17, v21 // (op 20083) 




v_add_u32 v0, 256, v13 // (op 20085) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20085) Store value




s_lshl_b32 s0, s2, 6 // (op 20106) 
s_add_u32 s0, s0, 256 // (op 20106) 
v_add_u32 v0, v12, s0 // (op 20106) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20106) Load value

v_accvgpr_read v14, a132 // (op 20132) 
v_accvgpr_read v15, a133 // (op 20132) 
v_accvgpr_read v16, a134 // (op 20132) 
v_accvgpr_read v17, a135 // (op 20132) 
v_mul_f32 v18, v11, v14 // (op 20132) 
v_mul_f32 v19, v11, v15 // (op 20132) 
v_mul_f32 v20, v11, v16 // (op 20132) 
v_mul_f32 v21, v11, v17 // (op 20132) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20104) 
v_mul_f32 v15, v9, v5 // (op 20104) 
v_mul_f32 v16, v9, v6 // (op 20104) 
v_mul_f32 v17, v9, v7 // (op 20104) 

v_add_f32 v4, v14, v18 // (op 20105) 
v_add_f32 v5, v15, v19 // (op 20105) 
v_add_f32 v6, v16, v20 // (op 20105) 
v_add_f32 v7, v17, v21 // (op 20105) 




s_lshl_b32 s0, s4, 6 // (op 20087) 
s_add_u32 s0, s0, 256 // (op 20087) 
v_add_u32 v0, v13, s0 // (op 20087) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20087) Store value




s_lshl_b32 s0, s2, 7 // (op 20110) 
s_add_u32 s0, s0, 256 // (op 20110) 
v_add_u32 v0, v12, s0 // (op 20110) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20110) Load value

v_accvgpr_read v14, a136 // (op 20134) 
v_accvgpr_read v15, a137 // (op 20134) 
v_accvgpr_read v16, a138 // (op 20134) 
v_accvgpr_read v17, a139 // (op 20134) 
v_mul_f32 v18, v11, v14 // (op 20134) 
v_mul_f32 v19, v11, v15 // (op 20134) 
v_mul_f32 v20, v11, v16 // (op 20134) 
v_mul_f32 v21, v11, v17 // (op 20134) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20108) 
v_mul_f32 v15, v9, v5 // (op 20108) 
v_mul_f32 v16, v9, v6 // (op 20108) 
v_mul_f32 v17, v9, v7 // (op 20108) 

v_add_f32 v4, v14, v18 // (op 20109) 
v_add_f32 v5, v15, v19 // (op 20109) 
v_add_f32 v6, v16, v20 // (op 20109) 
v_add_f32 v7, v17, v21 // (op 20109) 




s_lshl_b32 s0, s4, 7 // (op 20089) 
s_add_u32 s0, s0, 256 // (op 20089) 
v_add_u32 v0, v13, s0 // (op 20089) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20089) Store value




s_lshl_b32 s0, s2, 6 // (op 20114) 
s_mul_i32 s1, 3, s0 // (op 20114) 
s_add_u32 s1, s1, 256 // (op 20114) 

v_add_u32 v0, v12, s1 // (op 20114) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20114) Load value

v_accvgpr_read v14, a140 // (op 20136) 
v_accvgpr_read v15, a141 // (op 20136) 
v_accvgpr_read v16, a142 // (op 20136) 
v_accvgpr_read v17, a143 // (op 20136) 
v_mul_f32 v18, v11, v14 // (op 20136) 
v_mul_f32 v19, v11, v15 // (op 20136) 
v_mul_f32 v20, v11, v16 // (op 20136) 
v_mul_f32 v21, v11, v17 // (op 20136) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20112) 
v_mul_f32 v15, v9, v5 // (op 20112) 
v_mul_f32 v16, v9, v6 // (op 20112) 
v_mul_f32 v17, v9, v7 // (op 20112) 

v_add_f32 v4, v14, v18 // (op 20113) 
v_add_f32 v5, v15, v19 // (op 20113) 
v_add_f32 v6, v16, v20 // (op 20113) 
v_add_f32 v7, v17, v21 // (op 20113) 




s_lshl_b32 s0, s4, 6 // (op 20091) 
s_mul_i32 s1, 3, s0 // (op 20091) 
s_add_u32 s1, s1, 256 // (op 20091) 

v_add_u32 v0, v13, s1 // (op 20091) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20091) Store value




s_lshl_b32 s0, s2, 8 // (op 20118) 
s_add_u32 s0, s0, 256 // (op 20118) 
v_add_u32 v0, v12, s0 // (op 20118) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20118) Load value

v_accvgpr_read v14, a144 // (op 20138) 
v_accvgpr_read v15, a145 // (op 20138) 
v_accvgpr_read v16, a146 // (op 20138) 
v_accvgpr_read v17, a147 // (op 20138) 
v_mul_f32 v18, v11, v14 // (op 20138) 
v_mul_f32 v19, v11, v15 // (op 20138) 
v_mul_f32 v20, v11, v16 // (op 20138) 
v_mul_f32 v21, v11, v17 // (op 20138) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20116) 
v_mul_f32 v15, v9, v5 // (op 20116) 
v_mul_f32 v16, v9, v6 // (op 20116) 
v_mul_f32 v17, v9, v7 // (op 20116) 

v_add_f32 v4, v14, v18 // (op 20117) 
v_add_f32 v5, v15, v19 // (op 20117) 
v_add_f32 v6, v16, v20 // (op 20117) 
v_add_f32 v7, v17, v21 // (op 20117) 




s_lshl_b32 s0, s4, 8 // (op 20093) 
s_add_u32 s0, s0, 256 // (op 20093) 
v_add_u32 v0, v13, s0 // (op 20093) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20093) Store value




s_lshl_b32 s0, s2, 6 // (op 20122) 
s_mul_i32 s1, 5, s0 // (op 20122) 
s_add_u32 s1, s1, 256 // (op 20122) 

v_add_u32 v0, v12, s1 // (op 20122) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20122) Load value

v_accvgpr_read v14, a148 // (op 20140) 
v_accvgpr_read v15, a149 // (op 20140) 
v_accvgpr_read v16, a150 // (op 20140) 
v_accvgpr_read v17, a151 // (op 20140) 
v_mul_f32 v18, v11, v14 // (op 20140) 
v_mul_f32 v19, v11, v15 // (op 20140) 
v_mul_f32 v20, v11, v16 // (op 20140) 
v_mul_f32 v21, v11, v17 // (op 20140) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20120) 
v_mul_f32 v15, v9, v5 // (op 20120) 
v_mul_f32 v16, v9, v6 // (op 20120) 
v_mul_f32 v17, v9, v7 // (op 20120) 

v_add_f32 v4, v14, v18 // (op 20121) 
v_add_f32 v5, v15, v19 // (op 20121) 
v_add_f32 v6, v16, v20 // (op 20121) 
v_add_f32 v7, v17, v21 // (op 20121) 




s_lshl_b32 s0, s4, 6 // (op 20095) 
s_mul_i32 s1, 5, s0 // (op 20095) 
s_add_u32 s1, s1, 256 // (op 20095) 

v_add_u32 v0, v13, s1 // (op 20095) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20095) Store value




s_lshl_b32 s0, s2, 6 // (op 20126) 
s_mul_i32 s1, 6, s0 // (op 20126) 
s_add_u32 s1, s1, 256 // (op 20126) 

v_add_u32 v0, v12, s1 // (op 20126) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20126) Load value

v_accvgpr_read v14, a152 // (op 20142) 
v_accvgpr_read v15, a153 // (op 20142) 
v_accvgpr_read v16, a154 // (op 20142) 
v_accvgpr_read v17, a155 // (op 20142) 
v_mul_f32 v18, v11, v14 // (op 20142) 
v_mul_f32 v19, v11, v15 // (op 20142) 
v_mul_f32 v20, v11, v16 // (op 20142) 
v_mul_f32 v21, v11, v17 // (op 20142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20124) 
v_mul_f32 v15, v9, v5 // (op 20124) 
v_mul_f32 v16, v9, v6 // (op 20124) 
v_mul_f32 v17, v9, v7 // (op 20124) 

v_add_f32 v4, v14, v18 // (op 20125) 
v_add_f32 v5, v15, v19 // (op 20125) 
v_add_f32 v6, v16, v20 // (op 20125) 
v_add_f32 v7, v17, v21 // (op 20125) 




s_lshl_b32 s0, s4, 6 // (op 20097) 
s_mul_i32 s1, 6, s0 // (op 20097) 
s_add_u32 s1, s1, 256 // (op 20097) 

v_add_u32 v0, v13, s1 // (op 20097) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20097) Store value




s_lshl_b32 s0, s2, 6 // (op 20130) 
s_mul_i32 s1, 7, s0 // (op 20130) 
s_add_u32 s1, s1, 256 // (op 20130) 

v_add_u32 v0, v12, s1 // (op 20130) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 20130) Load value

v_accvgpr_read v14, a156 // (op 20144) 
v_accvgpr_read v15, a157 // (op 20144) 
v_accvgpr_read v16, a158 // (op 20144) 
v_accvgpr_read v17, a159 // (op 20144) 
v_mul_f32 v18, v11, v14 // (op 20144) 
v_mul_f32 v19, v11, v15 // (op 20144) 
v_mul_f32 v20, v11, v16 // (op 20144) 
v_mul_f32 v21, v11, v17 // (op 20144) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 20128) 
v_mul_f32 v15, v9, v5 // (op 20128) 
v_mul_f32 v16, v9, v6 // (op 20128) 
v_mul_f32 v17, v9, v7 // (op 20128) 

v_add_f32 v4, v14, v18 // (op 20129) 
v_add_f32 v5, v15, v19 // (op 20129) 
v_add_f32 v6, v16, v20 // (op 20129) 
v_add_f32 v7, v17, v21 // (op 20129) 




s_lshl_b32 s0, s4, 6 // (op 20099) 
s_mul_i32 s1, 7, s0 // (op 20099) 
s_add_u32 s1, s1, 256 // (op 20099) 

v_add_u32 v0, v13, s1 // (op 20099) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 20099) Store value




v_add_u32 v0, 192, v12 // (op 19947) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19947) Load value

v_accvgpr_read v14, a96 // (op 19927) 
v_accvgpr_read v15, a97 // (op 19927) 
v_accvgpr_read v16, a98 // (op 19927) 
v_accvgpr_read v17, a99 // (op 19927) 
v_mul_f32 v18, v11, v14 // (op 19927) 
v_mul_f32 v19, v11, v15 // (op 19927) 
v_mul_f32 v20, v11, v16 // (op 19927) 
v_mul_f32 v21, v11, v17 // (op 19927) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 19946) 
v_mul_f32 v15, v9, v5 // (op 19946) 
v_mul_f32 v16, v9, v6 // (op 19946) 
v_mul_f32 v17, v9, v7 // (op 19946) 

v_add_f32 v4, v14, v18 // (op 19928) 
v_add_f32 v5, v15, v19 // (op 19928) 
v_add_f32 v6, v16, v20 // (op 19928) 
v_add_f32 v7, v17, v21 // (op 19928) 




v_add_u32 v0, 192, v13 // (op 19930) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19930) Store value




s_lshl_b32 s0, s2, 6 // (op 19951) 
s_add_u32 s0, s0, 192 // (op 19951) 
v_add_u32 v0, v12, s0 // (op 19951) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19951) Load value

v_accvgpr_read v14, a100 // (op 19977) 
v_accvgpr_read v15, a101 // (op 19977) 
v_accvgpr_read v16, a102 // (op 19977) 
v_accvgpr_read v17, a103 // (op 19977) 
v_mul_f32 v18, v11, v14 // (op 19977) 
v_mul_f32 v19, v11, v15 // (op 19977) 
v_mul_f32 v20, v11, v16 // (op 19977) 
v_mul_f32 v21, v11, v17 // (op 19977) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 19949) 
v_mul_f32 v15, v9, v5 // (op 19949) 
v_mul_f32 v16, v9, v6 // (op 19949) 
v_mul_f32 v17, v9, v7 // (op 19949) 

v_add_f32 v4, v14, v18 // (op 19950) 
v_add_f32 v5, v15, v19 // (op 19950) 
v_add_f32 v6, v16, v20 // (op 19950) 
v_add_f32 v7, v17, v21 // (op 19950) 




s_lshl_b32 s0, s4, 6 // (op 19932) 
s_add_u32 s0, s0, 192 // (op 19932) 
v_add_u32 v0, v13, s0 // (op 19932) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19932) Store value




s_lshl_b32 s0, s2, 7 // (op 19955) 
s_add_u32 s0, s0, 192 // (op 19955) 
v_add_u32 v0, v12, s0 // (op 19955) 


buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19955) Load value

v_accvgpr_read v14, a104 // (op 19979) 
v_accvgpr_read v15, a105 // (op 19979) 
v_accvgpr_read v16, a106 // (op 19979) 
v_accvgpr_read v17, a107 // (op 19979) 
v_mul_f32 v18, v11, v14 // (op 19979) 
v_mul_f32 v19, v11, v15 // (op 19979) 
v_mul_f32 v20, v11, v16 // (op 19979) 
v_mul_f32 v21, v11, v17 // (op 19979) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 19953) 
v_mul_f32 v15, v9, v5 // (op 19953) 
v_mul_f32 v16, v9, v6 // (op 19953) 
v_mul_f32 v17, v9, v7 // (op 19953) 

v_add_f32 v4, v14, v18 // (op 19954) 
v_add_f32 v5, v15, v19 // (op 19954) 
v_add_f32 v6, v16, v20 // (op 19954) 
v_add_f32 v7, v17, v21 // (op 19954) 




s_lshl_b32 s0, s4, 7 // (op 19934) 
s_add_u32 s0, s0, 192 // (op 19934) 
v_add_u32 v0, v13, s0 // (op 19934) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19934) Store value




s_lshl_b32 s0, s2, 6 // (op 19959) 
s_mul_i32 s1, 3, s0 // (op 19959) 
s_add_u32 s1, s1, 192 // (op 19959) 

v_add_u32 v0, v12, s1 // (op 19959) 


v_accvgpr_read v4, a108 // (op 19981) 
v_accvgpr_read v5, a109 // (op 19981) 
v_accvgpr_read v6, a110 // (op 19981) 
v_accvgpr_read v7, a111 // (op 19981) 
v_mul_f32 v14, v11, v4 // (op 19981) 
v_mul_f32 v15, v11, v5 // (op 19981) 
v_mul_f32 v16, v11, v6 // (op 19981) 
v_mul_f32 v17, v11, v7 // (op 19981) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19959) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19957) 
v_mul_f32 v19, v9, v5 // (op 19957) 
v_mul_f32 v20, v9, v6 // (op 19957) 
v_mul_f32 v21, v9, v7 // (op 19957) 

v_add_f32 v4, v18, v14 // (op 19958) 
v_add_f32 v5, v19, v15 // (op 19958) 
v_add_f32 v6, v20, v16 // (op 19958) 
v_add_f32 v7, v21, v17 // (op 19958) 




s_lshl_b32 s0, s4, 6 // (op 19936) 
s_mul_i32 s1, 3, s0 // (op 19936) 
s_add_u32 s1, s1, 192 // (op 19936) 

v_add_u32 v0, v13, s1 // (op 19936) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19936) Store value




s_lshl_b32 s0, s2, 8 // (op 19963) 
s_add_u32 s0, s0, 192 // (op 19963) 
v_add_u32 v0, v12, s0 // (op 19963) 


v_accvgpr_read v4, a112 // (op 19983) 
v_accvgpr_read v5, a113 // (op 19983) 
v_accvgpr_read v6, a114 // (op 19983) 
v_accvgpr_read v7, a115 // (op 19983) 
v_mul_f32 v14, v11, v4 // (op 19983) 
v_mul_f32 v15, v11, v5 // (op 19983) 
v_mul_f32 v16, v11, v6 // (op 19983) 
v_mul_f32 v17, v11, v7 // (op 19983) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19963) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19961) 
v_mul_f32 v19, v9, v5 // (op 19961) 
v_mul_f32 v20, v9, v6 // (op 19961) 
v_mul_f32 v21, v9, v7 // (op 19961) 

v_add_f32 v4, v18, v14 // (op 19962) 
v_add_f32 v5, v19, v15 // (op 19962) 
v_add_f32 v6, v20, v16 // (op 19962) 
v_add_f32 v7, v21, v17 // (op 19962) 




s_lshl_b32 s0, s4, 8 // (op 19938) 
s_add_u32 s0, s0, 192 // (op 19938) 
v_add_u32 v0, v13, s0 // (op 19938) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19938) Store value




s_lshl_b32 s0, s2, 6 // (op 19967) 
s_mul_i32 s1, 5, s0 // (op 19967) 
s_add_u32 s1, s1, 192 // (op 19967) 

v_add_u32 v0, v12, s1 // (op 19967) 


v_accvgpr_read v4, a116 // (op 19985) 
v_accvgpr_read v5, a117 // (op 19985) 
v_accvgpr_read v6, a118 // (op 19985) 
v_accvgpr_read v7, a119 // (op 19985) 
v_mul_f32 v14, v11, v4 // (op 19985) 
v_mul_f32 v15, v11, v5 // (op 19985) 
v_mul_f32 v16, v11, v6 // (op 19985) 
v_mul_f32 v17, v11, v7 // (op 19985) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19967) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19965) 
v_mul_f32 v19, v9, v5 // (op 19965) 
v_mul_f32 v20, v9, v6 // (op 19965) 
v_mul_f32 v21, v9, v7 // (op 19965) 

v_add_f32 v4, v18, v14 // (op 19966) 
v_add_f32 v5, v19, v15 // (op 19966) 
v_add_f32 v6, v20, v16 // (op 19966) 
v_add_f32 v7, v21, v17 // (op 19966) 




s_lshl_b32 s0, s4, 6 // (op 19940) 
s_mul_i32 s1, 5, s0 // (op 19940) 
s_add_u32 s1, s1, 192 // (op 19940) 

v_add_u32 v0, v13, s1 // (op 19940) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19940) Store value




s_lshl_b32 s0, s2, 6 // (op 19971) 
s_mul_i32 s1, 6, s0 // (op 19971) 
s_add_u32 s1, s1, 192 // (op 19971) 

v_add_u32 v0, v12, s1 // (op 19971) 


v_accvgpr_read v4, a120 // (op 19987) 
v_accvgpr_read v5, a121 // (op 19987) 
v_accvgpr_read v6, a122 // (op 19987) 
v_accvgpr_read v7, a123 // (op 19987) 
v_mul_f32 v14, v11, v4 // (op 19987) 
v_mul_f32 v15, v11, v5 // (op 19987) 
v_mul_f32 v16, v11, v6 // (op 19987) 
v_mul_f32 v17, v11, v7 // (op 19987) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19971) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19969) 
v_mul_f32 v19, v9, v5 // (op 19969) 
v_mul_f32 v20, v9, v6 // (op 19969) 
v_mul_f32 v21, v9, v7 // (op 19969) 

v_add_f32 v4, v18, v14 // (op 19970) 
v_add_f32 v5, v19, v15 // (op 19970) 
v_add_f32 v6, v20, v16 // (op 19970) 
v_add_f32 v7, v21, v17 // (op 19970) 




s_lshl_b32 s0, s4, 6 // (op 19942) 
s_mul_i32 s1, 6, s0 // (op 19942) 
s_add_u32 s1, s1, 192 // (op 19942) 

v_add_u32 v0, v13, s1 // (op 19942) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19942) Store value




s_lshl_b32 s0, s2, 6 // (op 19975) 
s_mul_i32 s1, 7, s0 // (op 19975) 
s_add_u32 s1, s1, 192 // (op 19975) 

v_add_u32 v0, v12, s1 // (op 19975) 


v_accvgpr_read v4, a124 // (op 19989) 
v_accvgpr_read v5, a125 // (op 19989) 
v_accvgpr_read v6, a126 // (op 19989) 
v_accvgpr_read v7, a127 // (op 19989) 
v_mul_f32 v14, v11, v4 // (op 19989) 
v_mul_f32 v15, v11, v5 // (op 19989) 
v_mul_f32 v16, v11, v6 // (op 19989) 
v_mul_f32 v17, v11, v7 // (op 19989) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19975) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19973) 
v_mul_f32 v19, v9, v5 // (op 19973) 
v_mul_f32 v20, v9, v6 // (op 19973) 
v_mul_f32 v21, v9, v7 // (op 19973) 

v_add_f32 v4, v18, v14 // (op 19974) 
v_add_f32 v5, v19, v15 // (op 19974) 
v_add_f32 v6, v20, v16 // (op 19974) 
v_add_f32 v7, v21, v17 // (op 19974) 




s_lshl_b32 s0, s4, 6 // (op 19944) 
s_mul_i32 s1, 7, s0 // (op 19944) 
s_add_u32 s1, s1, 192 // (op 19944) 

v_add_u32 v0, v13, s1 // (op 19944) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19944) Store value




v_add_u32 v0, 128, v12 // (op 19792) 

s_nop 0
v_accvgpr_read v4, a64 // (op 19772) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v5, a65 // (op 19772) 
v_accvgpr_read v6, a66 // (op 19772) 
v_accvgpr_read v7, a67 // (op 19772) 
v_mul_f32 v14, v11, v4 // (op 19772) 
v_mul_f32 v15, v11, v5 // (op 19772) 
v_mul_f32 v16, v11, v6 // (op 19772) 
v_mul_f32 v17, v11, v7 // (op 19772) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19792) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19791) 
v_mul_f32 v19, v9, v5 // (op 19791) 
v_mul_f32 v20, v9, v6 // (op 19791) 
v_mul_f32 v21, v9, v7 // (op 19791) 

v_add_f32 v4, v18, v14 // (op 19773) 
v_add_f32 v5, v19, v15 // (op 19773) 
v_add_f32 v6, v20, v16 // (op 19773) 
v_add_f32 v7, v21, v17 // (op 19773) 




v_add_u32 v0, 128, v13 // (op 19775) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19775) Store value




s_lshl_b32 s0, s2, 6 // (op 19796) 
s_add_u32 s0, s0, 128 // (op 19796) 
v_add_u32 v0, v12, s0 // (op 19796) 


v_accvgpr_read v4, a68 // (op 19822) 
v_accvgpr_read v5, a69 // (op 19822) 
v_accvgpr_read v6, a70 // (op 19822) 
v_accvgpr_read v7, a71 // (op 19822) 
v_mul_f32 v14, v11, v4 // (op 19822) 
v_mul_f32 v15, v11, v5 // (op 19822) 
v_mul_f32 v16, v11, v6 // (op 19822) 
v_mul_f32 v17, v11, v7 // (op 19822) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19796) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19794) 
v_mul_f32 v19, v9, v5 // (op 19794) 
v_mul_f32 v20, v9, v6 // (op 19794) 
v_mul_f32 v21, v9, v7 // (op 19794) 

v_add_f32 v4, v18, v14 // (op 19795) 
v_add_f32 v5, v19, v15 // (op 19795) 
v_add_f32 v6, v20, v16 // (op 19795) 
v_add_f32 v7, v21, v17 // (op 19795) 




s_lshl_b32 s0, s4, 6 // (op 19777) 
s_add_u32 s0, s0, 128 // (op 19777) 
v_add_u32 v0, v13, s0 // (op 19777) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19777) Store value




s_lshl_b32 s0, s2, 7 // (op 19800) 
s_add_u32 s0, s0, 128 // (op 19800) 
v_add_u32 v0, v12, s0 // (op 19800) 


v_accvgpr_read v4, a72 // (op 19824) 
v_accvgpr_read v5, a73 // (op 19824) 
v_accvgpr_read v6, a74 // (op 19824) 
v_accvgpr_read v7, a75 // (op 19824) 
v_mul_f32 v14, v11, v4 // (op 19824) 
v_mul_f32 v15, v11, v5 // (op 19824) 
v_mul_f32 v16, v11, v6 // (op 19824) 
v_mul_f32 v17, v11, v7 // (op 19824) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19800) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19798) 
v_mul_f32 v19, v9, v5 // (op 19798) 
v_mul_f32 v20, v9, v6 // (op 19798) 
v_mul_f32 v21, v9, v7 // (op 19798) 

v_add_f32 v4, v18, v14 // (op 19799) 
v_add_f32 v5, v19, v15 // (op 19799) 
v_add_f32 v6, v20, v16 // (op 19799) 
v_add_f32 v7, v21, v17 // (op 19799) 




s_lshl_b32 s0, s4, 7 // (op 19779) 
s_add_u32 s0, s0, 128 // (op 19779) 
v_add_u32 v0, v13, s0 // (op 19779) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19779) Store value




s_lshl_b32 s0, s2, 6 // (op 19804) 
s_mul_i32 s1, 3, s0 // (op 19804) 
s_add_u32 s1, s1, 128 // (op 19804) 

v_add_u32 v0, v12, s1 // (op 19804) 


v_accvgpr_read v4, a76 // (op 19826) 
v_accvgpr_read v5, a77 // (op 19826) 
v_accvgpr_read v6, a78 // (op 19826) 
v_accvgpr_read v7, a79 // (op 19826) 
v_mul_f32 v14, v11, v4 // (op 19826) 
v_mul_f32 v15, v11, v5 // (op 19826) 
v_mul_f32 v16, v11, v6 // (op 19826) 
v_mul_f32 v17, v11, v7 // (op 19826) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19804) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19802) 
v_mul_f32 v19, v9, v5 // (op 19802) 
v_mul_f32 v20, v9, v6 // (op 19802) 
v_mul_f32 v21, v9, v7 // (op 19802) 

v_add_f32 v4, v18, v14 // (op 19803) 
v_add_f32 v5, v19, v15 // (op 19803) 
v_add_f32 v6, v20, v16 // (op 19803) 
v_add_f32 v7, v21, v17 // (op 19803) 




s_lshl_b32 s0, s4, 6 // (op 19781) 
s_mul_i32 s1, 3, s0 // (op 19781) 
s_add_u32 s1, s1, 128 // (op 19781) 

v_add_u32 v0, v13, s1 // (op 19781) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19781) Store value




s_lshl_b32 s0, s2, 8 // (op 19808) 
s_add_u32 s0, s0, 128 // (op 19808) 
v_add_u32 v0, v12, s0 // (op 19808) 


v_accvgpr_read v4, a80 // (op 19828) 
v_accvgpr_read v5, a81 // (op 19828) 
v_accvgpr_read v6, a82 // (op 19828) 
v_accvgpr_read v7, a83 // (op 19828) 
v_mul_f32 v14, v11, v4 // (op 19828) 
v_mul_f32 v15, v11, v5 // (op 19828) 
v_mul_f32 v16, v11, v6 // (op 19828) 
v_mul_f32 v17, v11, v7 // (op 19828) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19808) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19806) 
v_mul_f32 v19, v9, v5 // (op 19806) 
v_mul_f32 v20, v9, v6 // (op 19806) 
v_mul_f32 v21, v9, v7 // (op 19806) 

v_add_f32 v4, v18, v14 // (op 19807) 
v_add_f32 v5, v19, v15 // (op 19807) 
v_add_f32 v6, v20, v16 // (op 19807) 
v_add_f32 v7, v21, v17 // (op 19807) 




s_lshl_b32 s0, s4, 8 // (op 19783) 
s_add_u32 s0, s0, 128 // (op 19783) 
v_add_u32 v0, v13, s0 // (op 19783) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19783) Store value




s_lshl_b32 s0, s2, 6 // (op 19812) 
s_mul_i32 s1, 5, s0 // (op 19812) 
s_add_u32 s1, s1, 128 // (op 19812) 

v_add_u32 v0, v12, s1 // (op 19812) 


v_accvgpr_read v4, a84 // (op 19830) 
v_accvgpr_read v5, a85 // (op 19830) 
v_accvgpr_read v6, a86 // (op 19830) 
v_accvgpr_read v7, a87 // (op 19830) 
v_mul_f32 v14, v11, v4 // (op 19830) 
v_mul_f32 v15, v11, v5 // (op 19830) 
v_mul_f32 v16, v11, v6 // (op 19830) 
v_mul_f32 v17, v11, v7 // (op 19830) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19812) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19810) 
v_mul_f32 v19, v9, v5 // (op 19810) 
v_mul_f32 v20, v9, v6 // (op 19810) 
v_mul_f32 v21, v9, v7 // (op 19810) 

v_add_f32 v4, v18, v14 // (op 19811) 
v_add_f32 v5, v19, v15 // (op 19811) 
v_add_f32 v6, v20, v16 // (op 19811) 
v_add_f32 v7, v21, v17 // (op 19811) 




s_lshl_b32 s0, s4, 6 // (op 19785) 
s_mul_i32 s1, 5, s0 // (op 19785) 
s_add_u32 s1, s1, 128 // (op 19785) 

v_add_u32 v0, v13, s1 // (op 19785) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19785) Store value




s_lshl_b32 s0, s2, 6 // (op 19816) 
s_mul_i32 s1, 6, s0 // (op 19816) 
s_add_u32 s1, s1, 128 // (op 19816) 

v_add_u32 v0, v12, s1 // (op 19816) 


v_accvgpr_read v4, a88 // (op 19832) 
v_accvgpr_read v5, a89 // (op 19832) 
v_accvgpr_read v6, a90 // (op 19832) 
v_accvgpr_read v7, a91 // (op 19832) 
v_mul_f32 v14, v11, v4 // (op 19832) 
v_mul_f32 v15, v11, v5 // (op 19832) 
v_mul_f32 v16, v11, v6 // (op 19832) 
v_mul_f32 v17, v11, v7 // (op 19832) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19816) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19814) 
v_mul_f32 v19, v9, v5 // (op 19814) 
v_mul_f32 v20, v9, v6 // (op 19814) 
v_mul_f32 v21, v9, v7 // (op 19814) 

v_add_f32 v4, v18, v14 // (op 19815) 
v_add_f32 v5, v19, v15 // (op 19815) 
v_add_f32 v6, v20, v16 // (op 19815) 
v_add_f32 v7, v21, v17 // (op 19815) 




s_lshl_b32 s0, s4, 6 // (op 19787) 
s_mul_i32 s1, 6, s0 // (op 19787) 
s_add_u32 s1, s1, 128 // (op 19787) 

v_add_u32 v0, v13, s1 // (op 19787) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19787) Store value




s_lshl_b32 s0, s2, 6 // (op 19820) 
s_mul_i32 s1, 7, s0 // (op 19820) 
s_add_u32 s1, s1, 128 // (op 19820) 

v_add_u32 v0, v12, s1 // (op 19820) 


v_accvgpr_read v4, a92 // (op 19834) 
v_accvgpr_read v5, a93 // (op 19834) 
v_accvgpr_read v6, a94 // (op 19834) 
v_accvgpr_read v7, a95 // (op 19834) 
v_mul_f32 v14, v11, v4 // (op 19834) 
v_mul_f32 v15, v11, v5 // (op 19834) 
v_mul_f32 v16, v11, v6 // (op 19834) 
v_mul_f32 v17, v11, v7 // (op 19834) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19820) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19818) 
v_mul_f32 v19, v9, v5 // (op 19818) 
v_mul_f32 v20, v9, v6 // (op 19818) 
v_mul_f32 v21, v9, v7 // (op 19818) 

v_add_f32 v4, v18, v14 // (op 19819) 
v_add_f32 v5, v19, v15 // (op 19819) 
v_add_f32 v6, v20, v16 // (op 19819) 
v_add_f32 v7, v21, v17 // (op 19819) 




s_lshl_b32 s0, s4, 6 // (op 19789) 
s_mul_i32 s1, 7, s0 // (op 19789) 
s_add_u32 s1, s1, 128 // (op 19789) 

v_add_u32 v0, v13, s1 // (op 19789) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19789) Store value




v_add_u32 v0, 64, v12 // (op 19637) 

s_nop 0
v_accvgpr_read v4, a32 // (op 19617) Wait state hazard: Buffer Store Read Hazard
v_accvgpr_read v5, a33 // (op 19617) 
v_accvgpr_read v6, a34 // (op 19617) 
v_accvgpr_read v7, a35 // (op 19617) 
v_mul_f32 v14, v11, v4 // (op 19617) 
v_mul_f32 v15, v11, v5 // (op 19617) 
v_mul_f32 v16, v11, v6 // (op 19617) 
v_mul_f32 v17, v11, v7 // (op 19617) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19637) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19636) 
v_mul_f32 v19, v9, v5 // (op 19636) 
v_mul_f32 v20, v9, v6 // (op 19636) 
v_mul_f32 v21, v9, v7 // (op 19636) 

v_add_f32 v4, v18, v14 // (op 19618) 
v_add_f32 v5, v19, v15 // (op 19618) 
v_add_f32 v6, v20, v16 // (op 19618) 
v_add_f32 v7, v21, v17 // (op 19618) 




v_add_u32 v0, 64, v13 // (op 19620) 

buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19620) Store value




s_lshl_b32 s0, s2, 6 // (op 19641) 
s_add_u32 s0, s0, 64 // (op 19641) 
v_add_u32 v0, v12, s0 // (op 19641) 


v_accvgpr_read v4, a36 // (op 19667) 
v_accvgpr_read v5, a37 // (op 19667) 
v_accvgpr_read v6, a38 // (op 19667) 
v_accvgpr_read v7, a39 // (op 19667) 
v_mul_f32 v14, v11, v4 // (op 19667) 
v_mul_f32 v15, v11, v5 // (op 19667) 
v_mul_f32 v16, v11, v6 // (op 19667) 
v_mul_f32 v17, v11, v7 // (op 19667) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19641) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19639) 
v_mul_f32 v19, v9, v5 // (op 19639) 
v_mul_f32 v20, v9, v6 // (op 19639) 
v_mul_f32 v21, v9, v7 // (op 19639) 

v_add_f32 v4, v18, v14 // (op 19640) 
v_add_f32 v5, v19, v15 // (op 19640) 
v_add_f32 v6, v20, v16 // (op 19640) 
v_add_f32 v7, v21, v17 // (op 19640) 




s_lshl_b32 s0, s4, 6 // (op 19622) 
s_add_u32 s0, s0, 64 // (op 19622) 
v_add_u32 v0, v13, s0 // (op 19622) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19622) Store value




s_lshl_b32 s0, s2, 7 // (op 19645) 
s_add_u32 s0, s0, 64 // (op 19645) 
v_add_u32 v0, v12, s0 // (op 19645) 


v_accvgpr_read v4, a40 // (op 19669) 
v_accvgpr_read v5, a41 // (op 19669) 
v_accvgpr_read v6, a42 // (op 19669) 
v_accvgpr_read v7, a43 // (op 19669) 
v_mul_f32 v14, v11, v4 // (op 19669) 
v_mul_f32 v15, v11, v5 // (op 19669) 
v_mul_f32 v16, v11, v6 // (op 19669) 
v_mul_f32 v17, v11, v7 // (op 19669) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19645) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19643) 
v_mul_f32 v19, v9, v5 // (op 19643) 
v_mul_f32 v20, v9, v6 // (op 19643) 
v_mul_f32 v21, v9, v7 // (op 19643) 

v_add_f32 v4, v18, v14 // (op 19644) 
v_add_f32 v5, v19, v15 // (op 19644) 
v_add_f32 v6, v20, v16 // (op 19644) 
v_add_f32 v7, v21, v17 // (op 19644) 




s_lshl_b32 s0, s4, 7 // (op 19624) 
s_add_u32 s0, s0, 64 // (op 19624) 
v_add_u32 v0, v13, s0 // (op 19624) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19624) Store value




s_lshl_b32 s0, s2, 6 // (op 19649) 
s_mul_i32 s1, 3, s0 // (op 19649) 
s_add_u32 s1, s1, 64 // (op 19649) 

v_add_u32 v0, v12, s1 // (op 19649) 


v_accvgpr_read v4, a44 // (op 19671) 
v_accvgpr_read v5, a45 // (op 19671) 
v_accvgpr_read v6, a46 // (op 19671) 
v_accvgpr_read v7, a47 // (op 19671) 
v_mul_f32 v14, v11, v4 // (op 19671) 
v_mul_f32 v15, v11, v5 // (op 19671) 
v_mul_f32 v16, v11, v6 // (op 19671) 
v_mul_f32 v17, v11, v7 // (op 19671) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19649) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19647) 
v_mul_f32 v19, v9, v5 // (op 19647) 
v_mul_f32 v20, v9, v6 // (op 19647) 
v_mul_f32 v21, v9, v7 // (op 19647) 

v_add_f32 v4, v18, v14 // (op 19648) 
v_add_f32 v5, v19, v15 // (op 19648) 
v_add_f32 v6, v20, v16 // (op 19648) 
v_add_f32 v7, v21, v17 // (op 19648) 




s_lshl_b32 s0, s4, 6 // (op 19626) 
s_mul_i32 s1, 3, s0 // (op 19626) 
s_add_u32 s1, s1, 64 // (op 19626) 

v_add_u32 v0, v13, s1 // (op 19626) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19626) Store value




s_lshl_b32 s0, s2, 8 // (op 19653) 
s_add_u32 s0, s0, 64 // (op 19653) 
v_add_u32 v0, v12, s0 // (op 19653) 


v_accvgpr_read v4, a48 // (op 19673) 
v_accvgpr_read v5, a49 // (op 19673) 
v_accvgpr_read v6, a50 // (op 19673) 
v_accvgpr_read v7, a51 // (op 19673) 
v_mul_f32 v14, v11, v4 // (op 19673) 
v_mul_f32 v15, v11, v5 // (op 19673) 
v_mul_f32 v16, v11, v6 // (op 19673) 
v_mul_f32 v17, v11, v7 // (op 19673) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19653) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19651) 
v_mul_f32 v19, v9, v5 // (op 19651) 
v_mul_f32 v20, v9, v6 // (op 19651) 
v_mul_f32 v21, v9, v7 // (op 19651) 

v_add_f32 v4, v18, v14 // (op 19652) 
v_add_f32 v5, v19, v15 // (op 19652) 
v_add_f32 v6, v20, v16 // (op 19652) 
v_add_f32 v7, v21, v17 // (op 19652) 




s_lshl_b32 s0, s4, 8 // (op 19628) 
s_add_u32 s0, s0, 64 // (op 19628) 
v_add_u32 v0, v13, s0 // (op 19628) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19628) Store value




s_lshl_b32 s0, s2, 6 // (op 19657) 
s_mul_i32 s1, 5, s0 // (op 19657) 
s_add_u32 s1, s1, 64 // (op 19657) 

v_add_u32 v0, v12, s1 // (op 19657) 


v_accvgpr_read v4, a52 // (op 19675) 
v_accvgpr_read v5, a53 // (op 19675) 
v_accvgpr_read v6, a54 // (op 19675) 
v_accvgpr_read v7, a55 // (op 19675) 
v_mul_f32 v14, v11, v4 // (op 19675) 
v_mul_f32 v15, v11, v5 // (op 19675) 
v_mul_f32 v16, v11, v6 // (op 19675) 
v_mul_f32 v17, v11, v7 // (op 19675) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19657) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19655) 
v_mul_f32 v19, v9, v5 // (op 19655) 
v_mul_f32 v20, v9, v6 // (op 19655) 
v_mul_f32 v21, v9, v7 // (op 19655) 

v_add_f32 v4, v18, v14 // (op 19656) 
v_add_f32 v5, v19, v15 // (op 19656) 
v_add_f32 v6, v20, v16 // (op 19656) 
v_add_f32 v7, v21, v17 // (op 19656) 




s_lshl_b32 s0, s4, 6 // (op 19630) 
s_mul_i32 s1, 5, s0 // (op 19630) 
s_add_u32 s1, s1, 64 // (op 19630) 

v_add_u32 v0, v13, s1 // (op 19630) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19630) Store value




s_lshl_b32 s0, s2, 6 // (op 19661) 
s_mul_i32 s1, 6, s0 // (op 19661) 
s_add_u32 s1, s1, 64 // (op 19661) 

v_add_u32 v0, v12, s1 // (op 19661) 


v_accvgpr_read v4, a56 // (op 19677) 
v_accvgpr_read v5, a57 // (op 19677) 
v_accvgpr_read v6, a58 // (op 19677) 
v_accvgpr_read v7, a59 // (op 19677) 
v_mul_f32 v14, v11, v4 // (op 19677) 
v_mul_f32 v15, v11, v5 // (op 19677) 
v_mul_f32 v16, v11, v6 // (op 19677) 
v_mul_f32 v17, v11, v7 // (op 19677) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19661) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19659) 
v_mul_f32 v19, v9, v5 // (op 19659) 
v_mul_f32 v20, v9, v6 // (op 19659) 
v_mul_f32 v21, v9, v7 // (op 19659) 

v_add_f32 v4, v18, v14 // (op 19660) 
v_add_f32 v5, v19, v15 // (op 19660) 
v_add_f32 v6, v20, v16 // (op 19660) 
v_add_f32 v7, v21, v17 // (op 19660) 




s_lshl_b32 s0, s4, 6 // (op 19632) 
s_mul_i32 s1, 6, s0 // (op 19632) 
s_add_u32 s1, s1, 64 // (op 19632) 

v_add_u32 v0, v13, s1 // (op 19632) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19632) Store value




s_lshl_b32 s0, s2, 6 // (op 19665) 
s_mul_i32 s1, 7, s0 // (op 19665) 
s_add_u32 s1, s1, 64 // (op 19665) 

v_add_u32 v0, v12, s1 // (op 19665) 


v_accvgpr_read v4, a60 // (op 19679) 
v_accvgpr_read v5, a61 // (op 19679) 
v_accvgpr_read v6, a62 // (op 19679) 
v_accvgpr_read v7, a63 // (op 19679) 
v_mul_f32 v14, v11, v4 // (op 19679) 
v_mul_f32 v15, v11, v5 // (op 19679) 
v_mul_f32 v16, v11, v6 // (op 19679) 
v_mul_f32 v17, v11, v7 // (op 19679) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19665) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19663) 
v_mul_f32 v19, v9, v5 // (op 19663) 
v_mul_f32 v20, v9, v6 // (op 19663) 
v_mul_f32 v21, v9, v7 // (op 19663) 

v_add_f32 v4, v18, v14 // (op 19664) 
v_add_f32 v5, v19, v15 // (op 19664) 
v_add_f32 v6, v20, v16 // (op 19664) 
v_add_f32 v7, v21, v17 // (op 19664) 




s_lshl_b32 s0, s4, 6 // (op 19634) 
s_mul_i32 s1, 7, s0 // (op 19634) 
s_add_u32 s1, s1, 64 // (op 19634) 

v_add_u32 v0, v13, s1 // (op 19634) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19634) Store value




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




s_lshl_b32 s0, s2, 6 // (op 19455) 
v_add_u32 v0, v12, s0 // (op 19455) 


v_accvgpr_read v4, a4 // (op 19452) 
v_accvgpr_read v5, a5 // (op 19452) 
v_accvgpr_read v6, a6 // (op 19452) 
v_accvgpr_read v7, a7 // (op 19452) 
v_mul_f32 v14, v11, v4 // (op 19452) 
v_mul_f32 v15, v11, v5 // (op 19452) 
v_mul_f32 v16, v11, v6 // (op 19452) 
v_mul_f32 v17, v11, v7 // (op 19452) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19455) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19456) 
v_mul_f32 v19, v9, v5 // (op 19456) 
v_mul_f32 v20, v9, v6 // (op 19456) 
v_mul_f32 v21, v9, v7 // (op 19456) 

v_add_f32 v4, v18, v14 // (op 19453) 
v_add_f32 v5, v19, v15 // (op 19453) 
v_add_f32 v6, v20, v16 // (op 19453) 
v_add_f32 v7, v21, v17 // (op 19453) 




s_lshl_b32 s0, s4, 6 // (op 19454) 
v_add_u32 v0, v13, s0 // (op 19454) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19454) Store value




s_lshl_b32 s0, s2, 7 // (op 19467) 
v_add_u32 v0, v12, s0 // (op 19467) 


v_accvgpr_read v4, a8 // (op 19464) 
v_accvgpr_read v5, a9 // (op 19464) 
v_accvgpr_read v6, a10 // (op 19464) 
v_accvgpr_read v7, a11 // (op 19464) 
v_mul_f32 v14, v11, v4 // (op 19464) 
v_mul_f32 v15, v11, v5 // (op 19464) 
v_mul_f32 v16, v11, v6 // (op 19464) 
v_mul_f32 v17, v11, v7 // (op 19464) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19467) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19468) 
v_mul_f32 v19, v9, v5 // (op 19468) 
v_mul_f32 v20, v9, v6 // (op 19468) 
v_mul_f32 v21, v9, v7 // (op 19468) 

v_add_f32 v4, v18, v14 // (op 19465) 
v_add_f32 v5, v19, v15 // (op 19465) 
v_add_f32 v6, v20, v16 // (op 19465) 
v_add_f32 v7, v21, v17 // (op 19465) 




s_lshl_b32 s0, s4, 7 // (op 19466) 
v_add_u32 v0, v13, s0 // (op 19466) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19466) Store value




s_lshl_b32 s0, s2, 6 // (op 19479) 
s_mul_i32 s1, 3, s0 // (op 19479) 

v_add_u32 v0, v12, s1 // (op 19479) 


v_accvgpr_read v4, a12 // (op 19476) 
v_accvgpr_read v5, a13 // (op 19476) 
v_accvgpr_read v6, a14 // (op 19476) 
v_accvgpr_read v7, a15 // (op 19476) 
v_mul_f32 v14, v11, v4 // (op 19476) 
v_mul_f32 v15, v11, v5 // (op 19476) 
v_mul_f32 v16, v11, v6 // (op 19476) 
v_mul_f32 v17, v11, v7 // (op 19476) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19479) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19480) 
v_mul_f32 v19, v9, v5 // (op 19480) 
v_mul_f32 v20, v9, v6 // (op 19480) 
v_mul_f32 v21, v9, v7 // (op 19480) 

v_add_f32 v4, v18, v14 // (op 19477) 
v_add_f32 v5, v19, v15 // (op 19477) 
v_add_f32 v6, v20, v16 // (op 19477) 
v_add_f32 v7, v21, v17 // (op 19477) 




s_lshl_b32 s0, s4, 6 // (op 19478) 
s_mul_i32 s1, 3, s0 // (op 19478) 

v_add_u32 v0, v13, s1 // (op 19478) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19478) Store value




s_lshl_b32 s0, s2, 8 // (op 19491) 
v_add_u32 v0, v12, s0 // (op 19491) 


v_accvgpr_read v4, a16 // (op 19488) 
v_accvgpr_read v5, a17 // (op 19488) 
v_accvgpr_read v6, a18 // (op 19488) 
v_accvgpr_read v7, a19 // (op 19488) 
v_mul_f32 v14, v11, v4 // (op 19488) 
v_mul_f32 v15, v11, v5 // (op 19488) 
v_mul_f32 v16, v11, v6 // (op 19488) 
v_mul_f32 v17, v11, v7 // (op 19488) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19491) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19492) 
v_mul_f32 v19, v9, v5 // (op 19492) 
v_mul_f32 v20, v9, v6 // (op 19492) 
v_mul_f32 v21, v9, v7 // (op 19492) 

v_add_f32 v4, v18, v14 // (op 19489) 
v_add_f32 v5, v19, v15 // (op 19489) 
v_add_f32 v6, v20, v16 // (op 19489) 
v_add_f32 v7, v21, v17 // (op 19489) 




s_lshl_b32 s0, s4, 8 // (op 19490) 
v_add_u32 v0, v13, s0 // (op 19490) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19490) Store value




s_lshl_b32 s0, s2, 6 // (op 19503) 
s_mul_i32 s1, 5, s0 // (op 19503) 

v_add_u32 v0, v12, s1 // (op 19503) 


v_accvgpr_read v4, a20 // (op 19500) 
v_accvgpr_read v5, a21 // (op 19500) 
v_accvgpr_read v6, a22 // (op 19500) 
v_accvgpr_read v7, a23 // (op 19500) 
v_mul_f32 v14, v11, v4 // (op 19500) 
v_mul_f32 v15, v11, v5 // (op 19500) 
v_mul_f32 v16, v11, v6 // (op 19500) 
v_mul_f32 v17, v11, v7 // (op 19500) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19503) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19504) 
v_mul_f32 v19, v9, v5 // (op 19504) 
v_mul_f32 v20, v9, v6 // (op 19504) 
v_mul_f32 v21, v9, v7 // (op 19504) 

v_add_f32 v4, v18, v14 // (op 19501) 
v_add_f32 v5, v19, v15 // (op 19501) 
v_add_f32 v6, v20, v16 // (op 19501) 
v_add_f32 v7, v21, v17 // (op 19501) 




s_lshl_b32 s0, s4, 6 // (op 19502) 
s_mul_i32 s1, 5, s0 // (op 19502) 

v_add_u32 v0, v13, s1 // (op 19502) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19502) Store value




s_lshl_b32 s0, s2, 6 // (op 19515) 
s_mul_i32 s1, 6, s0 // (op 19515) 

v_add_u32 v0, v12, s1 // (op 19515) 


v_accvgpr_read v4, a24 // (op 19512) 
v_accvgpr_read v5, a25 // (op 19512) 
v_accvgpr_read v6, a26 // (op 19512) 
v_accvgpr_read v7, a27 // (op 19512) 
v_mul_f32 v14, v11, v4 // (op 19512) 
v_mul_f32 v15, v11, v5 // (op 19512) 
v_mul_f32 v16, v11, v6 // (op 19512) 
v_mul_f32 v17, v11, v7 // (op 19512) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19515) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19516) 
v_mul_f32 v19, v9, v5 // (op 19516) 
v_mul_f32 v20, v9, v6 // (op 19516) 
v_mul_f32 v21, v9, v7 // (op 19516) 

v_add_f32 v4, v18, v14 // (op 19513) 
v_add_f32 v5, v19, v15 // (op 19513) 
v_add_f32 v6, v20, v16 // (op 19513) 
v_add_f32 v7, v21, v17 // (op 19513) 




s_lshl_b32 s0, s4, 6 // (op 19514) 
s_mul_i32 s1, 6, s0 // (op 19514) 

v_add_u32 v0, v13, s1 // (op 19514) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19514) Store value




s_lshl_b32 s0, s2, 6 // (op 19527) 
s_mul_i32 s1, 7, s0 // (op 19527) 

v_add_u32 v0, v12, s1 // (op 19527) 


v_accvgpr_read v4, a28 // (op 19524) 
v_accvgpr_read v5, a29 // (op 19524) 
v_accvgpr_read v6, a30 // (op 19524) 
v_accvgpr_read v7, a31 // (op 19524) 
v_mul_f32 v14, v11, v4 // (op 19524) 
v_mul_f32 v15, v11, v5 // (op 19524) 
v_mul_f32 v16, v11, v6 // (op 19524) 
v_mul_f32 v17, v11, v7 // (op 19524) 

buffer_load_dwordx4 v[4:7], v0, s[40:43], 0 offen // (op 19527) Load value

s_waitcnt vmcnt(0)
v_mul_f32 v18, v9, v4 // (op 19528) 
v_mul_f32 v19, v9, v5 // (op 19528) 
v_mul_f32 v20, v9, v6 // (op 19528) 
v_mul_f32 v21, v9, v7 // (op 19528) 

v_add_f32 v4, v18, v14 // (op 19525) 
v_add_f32 v5, v19, v15 // (op 19525) 
v_add_f32 v6, v20, v16 // (op 19525) 
v_add_f32 v7, v21, v17 // (op 19525) 




s_lshl_b32 s0, s4, 6 // (op 19526) 
s_mul_i32 s1, 7, s0 // (op 19526) 

v_add_u32 v0, v13, s1 // (op 19526) 


buffer_store_dwordx4 v[4:7], v0, s[44:47], 0 offen // (op 19526) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_ConditionalBottom_0_beta_114:








































































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
  .amdhsa_next_free_vgpr 436
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 131072 // lds bytes
  .amdhsa_accum_offset 180
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
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 131072
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 66
    .vgpr_count: 178
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

