// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 20587) 
v_and_b32 v3, 15, v1 // (op 20587) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 20587) 


v_and_b32 v0, 63, v1 // (op 20587) 
v_lshrrev_b32 v3, 4, v0 // (op 20587) 

v_lshlrev_b32 v0, 5, v3 // (op 20587) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 20587) 


v_lshlrev_b32 v0, 2, v3 // (op 20587) 

v_lshrrev_b32 v3, 3, v0 // (op 20587) 

v_lshrrev_b32 v0, 6, v1 // (op 20644) 
v_and_b32 v4, 1, v0 // (op 20644) 

v_and_b32 v0, 15, v1 // (op 20644) 
v_lshl_add_u32 v5, v4, 7, v0 // (op 20644) 


v_and_b32 v0, 63, v1 // (op 20644) 
v_lshrrev_b32 v4, 4, v0 // (op 20644) 

v_lshlrev_b32 v0, 5, v4 // (op 20644) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 20644) 


v_lshlrev_b32 v0, 2, v4 // (op 20644) 

v_lshrrev_b32 v4, 3, v0 // (op 20644) 

v_lshrrev_b32 v0, 7, v1 // (op 20711) 
v_and_b32 v5, 15, v1 // (op 20711) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 20711) 


v_and_b32 v0, 63, v1 // (op 20711) 
v_lshrrev_b32 v5, 4, v0 // (op 20711) 

v_lshlrev_b32 v0, 5, v5 // (op 20711) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 20711) 


v_lshlrev_b32 v0, 2, v5 // (op 20711) 

v_lshrrev_b32 v5, 3, v0 // (op 20711) 

v_lshrrev_b32 v0, 6, v1 // (op 20755) 
v_and_b32 v6, 1, v0 // (op 20755) 

v_and_b32 v0, 15, v1 // (op 20755) 
v_lshl_add_u32 v7, v6, 7, v0 // (op 20755) 


v_and_b32 v0, 63, v1 // (op 20755) 
v_lshrrev_b32 v6, 4, v0 // (op 20755) 

v_lshlrev_b32 v0, 5, v6 // (op 20755) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 20755) 


v_lshlrev_b32 v0, 2, v6 // (op 20755) 

v_lshrrev_b32 v6, 3, v0 // (op 20755) 

v_lshrrev_b32 v0, 7, v1 // (op 20799) 
v_and_b32 v7, 15, v1 // (op 20799) 
v_lshl_add_u32 v8, v0, 6, v7 // (op 20799) 


v_and_b32 v0, 63, v1 // (op 20799) 
v_lshrrev_b32 v7, 4, v0 // (op 20799) 

v_lshlrev_b32 v0, 5, v7 // (op 20799) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 20799) 


v_lshlrev_b32 v0, 2, v7 // (op 20799) 

v_lshrrev_b32 v7, 3, v0 // (op 20799) 

v_lshrrev_b32 v0, 6, v1 // (op 20843) 
v_and_b32 v8, 1, v0 // (op 20843) 

v_and_b32 v0, 15, v1 // (op 20843) 
v_lshl_add_u32 v9, v8, 7, v0 // (op 20843) 


v_and_b32 v0, 63, v1 // (op 20843) 
v_lshrrev_b32 v8, 4, v0 // (op 20843) 

v_lshlrev_b32 v0, 5, v8 // (op 20843) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 20843) 


v_lshlrev_b32 v0, 2, v8 // (op 20843) 

v_lshrrev_b32 v8, 3, v0 // (op 20843) 

v_lshrrev_b32 v0, 7, v1 // (op 20887) 
v_and_b32 v9, 15, v1 // (op 20887) 
v_lshl_add_u32 v10, v0, 6, v9 // (op 20887) 


v_and_b32 v0, 63, v1 // (op 20887) 
v_lshrrev_b32 v9, 4, v0 // (op 20887) 

v_lshlrev_b32 v0, 5, v9 // (op 20887) 

v_lshl_add_u32 v9, v10, 7, v0 // (op 20887) 


v_lshlrev_b32 v0, 2, v9 // (op 20887) 

v_lshrrev_b32 v9, 3, v0 // (op 20887) 

v_lshrrev_b32 v0, 6, v1 // (op 20931) 
v_and_b32 v10, 1, v0 // (op 20931) 

v_and_b32 v0, 15, v1 // (op 20931) 
v_lshl_add_u32 v11, v10, 7, v0 // (op 20931) 


v_and_b32 v0, 63, v1 // (op 20931) 
v_lshrrev_b32 v10, 4, v0 // (op 20931) 

v_lshlrev_b32 v0, 5, v10 // (op 20931) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 20931) 


v_lshlrev_b32 v0, 2, v10 // (op 20931) 

v_lshrrev_b32 v10, 3, v0 // (op 20931) 

v_lshrrev_b32 v0, 7, v1 // (op 20526) 
v_and_b32 v11, 63, v1 // (op 20526) 
v_lshrrev_b32 v12, 4, v11 // (op 20526) 
v_lshlrev_b32 v13, 2, v12 // (op 20526) 

v_lshl_add_u32 v12, v0, 6, v13 // (op 20526) 


v_lshl_add_u32 v0, s16, 7, v12 // (op 20526) 

v_lshrrev_b32 v12, 2, v11 // (op 20526) 

v_and_b32 v11, 3, v12 // (op 20526) 

v_and_b32 v12, 3, v1 // (op 20526) 
v_lshl_add_u32 v13, v11, 2, v12 // (op 20526) 


v_lshrrev_b32 v11, 6, v1 // (op 20526) 
v_and_b32 v12, 1, v11 // (op 20526) 

v_lshl_add_u32 v11, v12, 7, v13 // (op 20526) 


v_lshl_add_u32 v12, s17, 8, v11 // (op 20526) 

v_mul_lo_u32 v11, v12, s2 // (op 20526) 

v_add_lshl_u32 v12, v0, v11, 2 // (op 20526) 


v_lshrrev_b32 v0, 7, v1 // (op 20665) 
v_and_b32 v11, 63, v1 // (op 20665) 
v_lshrrev_b32 v13, 4, v11 // (op 20665) 
s_mul_i32 s1, s2, s9 // (op 20528) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 20528) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 20528) least significant
s_mul_hi_u32 s19, s2, s8 // (op 20528) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 20528) most significant: sum
s_add_u32 s1, s1, s19 // (op 20528) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20528) 

v_lshlrev_b32 v14, 2, v13 // (op 20665) 

v_lshl_add_u32 v13, v0, 6, v14 // (op 20665) 


v_lshl_add_u32 v0, s16, 7, v13 // (op 20665) 

v_lshrrev_b32 v13, 2, v11 // (op 20665) 

v_and_b32 v11, 3, v13 // (op 20665) 

v_and_b32 v13, 3, v1 // (op 20665) 
v_lshl_add_u32 v14, v11, 2, v13 // (op 20665) 


v_lshrrev_b32 v11, 6, v1 // (op 20665) 
v_and_b32 v13, 1, v11 // (op 20665) 

v_lshl_add_u32 v11, v13, 7, v14 // (op 20665) 


v_lshl_add_u32 v13, s17, 8, v11 // (op 20665) 

v_mul_lo_u32 v11, v13, s4 // (op 20665) 

v_add_lshl_u32 v13, v0, v11, 2 // (op 20665) 


s_mov_b32 s40, s6 // (op 20528) reg expression
s_mov_b32 s41, s7 // (op 20528) reg expression
s_mov_b32 s42, s18 // (op 20528) reg expression
s_mov_b32 s43, 131072 // (op 20528) call()

s_mul_i32 s1, s4, s9 // (op 20667) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 20667) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 20667) least significant
s_mul_hi_u32 s7, s4, s8 // (op 20667) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20667) most significant: sum
s_add_u32 s1, s1, s7 // (op 20667) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20667) 

s_mov_b32 s44, s10 // (op 20667) reg expression
s_mov_b32 s45, s11 // (op 20667) reg expression
s_mov_b32 s46, s6 // (op 20667) reg expression
s_mov_b32 s47, 131072 // (op 20667) call()


v_mov_b32 v0, s14 // (op 2) Move value
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
v_lshrrev_b32 v14, 2, v1 // (op 20416) 
v_lshl_add_u32 v15, s16, 7, v14 // (op 20416) 

v_and_b32 v14, -4, v1 // (op 20416) 
v_sub_u32 v16, v1, v14 // (op 20416) 

v_lshlrev_b32 v14, 5, v16 // (op 20416) 

v_mov_b32 v17, 0 // (op 20416) convert
v_mov_b32 v16, v14 // (op 20416) convert

v_mul_lo_u32 v19, v15, s13 // (op 20416) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 20416) most significant: high of low * low
v_mul_lo_u32 v18, v15, s12 // (op 20416) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 20416) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 20416) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 20416) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 20416) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 20416) 


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s12, s23 // (op 20418) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 20418) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 20418) least significant
s_mul_hi_u32 s7, s12, s22 // (op 20418) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20418) most significant: sum
s_add_u32 s1, s1, s7 // (op 20418) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20418) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20418) 

s_mov_b32 s48, s20 // (op 20418) reg expression
s_mov_b32 s49, s21 // (op 20418) reg expression
s_mov_b32 s50, s0 // (op 20418) reg expression
s_mov_b32 s51, 131072 // (op 20418) call()

v_lshrrev_b32 v14, 2, v1 // (op 20435) 
v_lshl_add_u32 v15, s16, 7, v14 // (op 20435) 

v_and_b32 v14, -4, v1 // (op 20435) 
v_sub_u32 v18, v1, v14 // (op 20435) 

v_lshlrev_b32 v14, 5, v18 // (op 20435) 

v_mov_b32 v19, 0 // (op 20435) convert
v_mov_b32 v18, v14 // (op 20435) convert

v_mul_lo_u32 v21, v15, s13 // (op 20435) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 20435) most significant: high of low * low
v_mul_lo_u32 v20, v15, s12 // (op 20435) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 20435) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 20435) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 20435) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 20435) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 20435) 

s_mul_i32 s1, s12, s23 // (op 20437) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 20437) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 20437) least significant
s_mul_hi_u32 s7, s12, s22 // (op 20437) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20437) most significant: sum
s_add_u32 s1, s1, s7 // (op 20437) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20437) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20437) 

s_mov_b32 s48, s20 // (op 20437) reg expression
s_mov_b32 s49, s21 // (op 20437) reg expression
s_mov_b32 s50, s0 // (op 20437) reg expression
s_mov_b32 s51, 131072 // (op 20437) call()

v_and_b32 v14, -4, v1 // (op 20480) 
v_sub_u32 v15, v1, v14 // (op 20480) 

v_lshlrev_b32 v14, 5, v15 // (op 20480) 

v_mov_b32 v21, 0 // (op 20480) convert
v_mov_b32 v20, v14 // (op 20480) convert

v_lshrrev_b32 v14, 2, v1 // (op 20480) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 20480) 

v_mul_lo_u32 v23, v15, s25 // (op 20480) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 20480) most significant: high of low * low
v_mul_lo_u32 v22, v15, s24 // (op 20480) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 20480) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 20480) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 20480) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20480) 


v_lshrrev_b64 v[20:21], 3, v[14:15] // (op 20480) 

s_mul_i32 s1, s24, s9 // (op 20482) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 20482) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 20482) least significant
s_mul_hi_u32 s7, s24, s8 // (op 20482) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20482) most significant: sum
s_add_u32 s1, s1, s7 // (op 20482) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20482) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20482) 

s_mov_b32 s52, s26 // (op 20482) reg expression
s_mov_b32 s53, s27 // (op 20482) reg expression
s_mov_b32 s54, s0 // (op 20482) reg expression
s_mov_b32 s55, 131072 // (op 20482) call()

v_and_b32 v14, -4, v1 // (op 20499) 
v_sub_u32 v15, v1, v14 // (op 20499) 

v_lshlrev_b32 v14, 5, v15 // (op 20499) 

v_mov_b32 v23, 0 // (op 20499) convert
v_mov_b32 v22, v14 // (op 20499) convert

v_lshrrev_b32 v14, 2, v1 // (op 20499) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 20499) 

v_mul_lo_u32 v25, v15, s25 // (op 20499) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 20499) most significant: high of low * low
v_mul_lo_u32 v24, v15, s24 // (op 20499) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 20499) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 20499) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 20499) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20499) 


v_lshrrev_b64 v[22:23], 3, v[14:15] // (op 20499) 

s_mul_i32 s1, s24, s9 // (op 20501) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 20501) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 20501) least significant
s_mul_hi_u32 s7, s24, s8 // (op 20501) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20501) most significant: sum
s_add_u32 s1, s1, s7 // (op 20501) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20501) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20501) 

s_mov_b32 s52, s26 // (op 20501) reg expression
s_mov_b32 s53, s27 // (op 20501) reg expression
s_mov_b32 s54, s0 // (op 20501) reg expression
s_mov_b32 s55, 131072 // (op 20501) call()

v_and_b32 v14, 63, v1 // (op 20953) 
v_lshrrev_b32 v15, 4, v14 // (op 20953) 
v_and_b32 v24, 3, v15 // (op 20953) 

v_and_b32 v15, 15, v1 // (op 20953) 
v_lshl_add_u32 v25, v24, 4, v15 // (op 20953) 


v_lshrrev_b32 v15, 7, v1 // (op 20953) 
v_lshl_add_u32 v24, v15, 6, v25 // (op 20953) 


v_lshl_add_u32 v15, s16, 7, v24 // (op 20953) 

v_lshrrev_b32 v24, 6, v14 // (op 20953) 

v_lshlrev_b32 v14, 2, v24 // (op 20953) 

v_mul_lo_u32 v24, v15, s28 // (op 20953) 
v_add_u32 v24, v24, v14 // (op 20953) 


s_mul_i32 s1, s28, s23 // (op 20955) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 20955) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 20955) least significant
s_mul_hi_u32 s7, s28, s22 // (op 20955) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20955) most significant: sum
s_add_u32 s1, s1, s7 // (op 20955) most significant: sum


s_mov_b32 s56, s30 // (op 20955) reg expression
s_mov_b32 s57, s31 // (op 20955) reg expression
s_mov_b32 s58, s0 // (op 20955) reg expression
s_mov_b32 s59, 131072 // (op 20955) call()

v_and_b32 v14, 63, v1 // (op 20974) 
v_lshrrev_b32 v15, 4, v14 // (op 20974) 
v_and_b32 v25, 3, v15 // (op 20974) 

v_and_b32 v15, 15, v1 // (op 20974) 
v_lshl_add_u32 v26, v25, 4, v15 // (op 20974) 


v_lshrrev_b32 v15, 7, v1 // (op 20974) 
v_lshl_add_u32 v25, v15, 6, v26 // (op 20974) 


v_lshl_add_u32 v15, s16, 7, v25 // (op 20974) 

v_lshrrev_b32 v25, 6, v14 // (op 20974) 

v_lshlrev_b32 v14, 2, v25 // (op 20974) 

v_mul_lo_u32 v25, v15, s28 // (op 20974) 
v_add_u32 v25, v25, v14 // (op 20974) 


s_mul_i32 s1, s28, s23 // (op 20976) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 20976) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 20976) least significant
s_mul_hi_u32 s7, s28, s22 // (op 20976) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20976) most significant: sum
s_add_u32 s1, s1, s7 // (op 20976) most significant: sum


s_mov_b32 s56, s30 // (op 20976) reg expression
s_mov_b32 s57, s31 // (op 20976) reg expression
s_mov_b32 s58, s0 // (op 20976) reg expression
s_mov_b32 s59, 131072 // (op 20976) call()

v_and_b32 v14, 63, v1 // (op 21005) 
v_lshrrev_b32 v15, 4, v14 // (op 21005) 
v_and_b32 v26, 3, v15 // (op 21005) 

v_and_b32 v15, 15, v1 // (op 21005) 
v_lshl_add_u32 v27, v26, 4, v15 // (op 21005) 


v_lshrrev_b32 v15, 6, v1 // (op 21005) 
v_and_b32 v26, 1, v15 // (op 21005) 

v_lshl_add_u32 v15, v26, 7, v27 // (op 21005) 


v_lshl_add_u32 v26, s17, 8, v15 // (op 21005) 

v_mul_lo_u32 v15, v26, s32 // (op 21005) 

v_lshrrev_b32 v26, 6, v14 // (op 21005) 

v_lshl_add_u32 v14, v26, 2, v15 // (op 21005) 


s_mul_i32 s1, s32, s9 // (op 21007) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 21007) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 21007) least significant
s_mul_hi_u32 s7, s32, s8 // (op 21007) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 21007) most significant: sum
s_add_u32 s1, s1, s7 // (op 21007) most significant: sum


s_mov_b32 s60, s34 // (op 21007) reg expression
s_mov_b32 s61, s35 // (op 21007) reg expression
s_mov_b32 s62, s0 // (op 21007) reg expression
s_mov_b32 s63, 131072 // (op 21007) call()

v_and_b32 v15, 63, v1 // (op 21026) 
v_lshrrev_b32 v26, 4, v15 // (op 21026) 
v_and_b32 v27, 3, v26 // (op 21026) 

v_and_b32 v26, 15, v1 // (op 21026) 
v_lshl_add_u32 v28, v27, 4, v26 // (op 21026) 


v_lshrrev_b32 v26, 6, v1 // (op 21026) 
v_and_b32 v27, 1, v26 // (op 21026) 

v_lshl_add_u32 v26, v27, 7, v28 // (op 21026) 


v_lshl_add_u32 v27, s17, 8, v26 // (op 21026) 

v_mul_lo_u32 v26, v27, s32 // (op 21026) 

v_lshrrev_b32 v27, 6, v15 // (op 21026) 

v_lshl_add_u32 v15, v27, 2, v26 // (op 21026) 


s_mul_i32 s1, s32, s9 // (op 21028) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 21028) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 21028) least significant
s_mul_hi_u32 s7, s32, s8 // (op 21028) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 21028) most significant: sum
s_add_u32 s1, s1, s7 // (op 21028) most significant: sum


s_mov_b32 s60, s34 // (op 21028) reg expression
s_mov_b32 s61, s35 // (op 21028) reg expression
s_mov_b32 s62, s0 // (op 21028) reg expression
s_mov_b32 s63, 131072 // (op 21028) call()

v_and_b32 v26, -4, v1 // (op 20627) 
v_sub_u32 v27, v1, v26 // (op 20627) 

v_lshlrev_b32 v26, 5, v27 // (op 20627) 

v_lshrrev_b32 v27, 2, v1 // (op 20627) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 20627) 


v_lshlrev_b32 v26, 2, v28 // (op 20627) 

v_lshrrev_b32 v27, 3, v26 // (op 20627) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 20627) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 20326) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20326) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20326) 


v_mov_b64 v[28:29], v[20:21] // (op 20326) 
s_mov_b32 m0, s0 // (op 20326) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 20326) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 20326) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 20326) most significant half

s_add_u32 m0, m0, 4096 // (op 20326) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 20326) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 20326) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 20326) most significant half

s_add_u32 m0, m0, 4096 // (op 20326) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 20326) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 20326) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 20326) most significant half

s_add_u32 m0, m0, 4096 // (op 20326) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 20326) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 20570) 
v_sub_u32 v27, v1, v26 // (op 20570) 

v_lshlrev_b32 v26, 5, v27 // (op 20570) 

v_lshrrev_b32 v27, 2, v1 // (op 20570) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 20570) 


v_lshlrev_b32 v26, 2, v28 // (op 20570) 

v_lshrrev_b32 v27, 3, v26 // (op 20570) 

s_nop 0
v_readfirstlane_b32 s0, v27 // (op 20570) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 20331) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20331) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20331) 


v_mov_b64 v[28:29], v[16:17] // (op 20331) 
s_add_u32 s1, 16384, s0 // (op 20331) 
s_mov_b32 m0, s1 // (op 20331) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20331) Load value direct to lds
v_add_co_u32 v28, s[6:7], v28, v26 // (op 20331) least significant half
v_addc_co_u32 v29, s[6:7], 0, v29, s[6:7] // (op 20331) most significant half

s_add_u32 m0, m0, 4096 // (op 20331) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20331) Load value direct to lds






v_mov_b32 v26, v14 // (op 74) reg expression

buffer_load_dwordx4 v[28:31], v26, s[60:63], 0 offen // (op 74) Load value



s_lshl_b32 s0, s32, 6 // (op 1100) 
v_add_u32 v26, v14, s0 // (op 1100) 


buffer_load_dwordx4 v[32:35], v26, s[60:63], 0 offen // (op 1100) Load value




v_mov_b32 v14, v24 // (op 70) reg expression

buffer_load_dwordx4 v[36:39], v14, s[56:59], 0 offen // (op 70) Load value


v_and_b32 v14, -4, v1 // (op 20738) 
v_sub_u32 v24, v1, v14 // (op 20738) 

v_lshlrev_b32 v14, 5, v24 // (op 20738) 

v_lshrrev_b32 v24, 2, v1 // (op 20738) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20738) 


v_lshlrev_b32 v14, 2, v26 // (op 20738) 

v_lshrrev_b32 v24, 3, v14 // (op 20738) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20738) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 20336) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20336) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20336) 


v_mov_b64 v[40:41], v[20:21] // (op 20336) 
s_add_u32 s1, 24576, s0 // (op 20336) 
s_mov_b32 m0, s1 // (op 20336) reg expression
buffer_load_dwordx4 v40, s[52:55], 64 offen offset: 0 lds // (op 20336) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v26 // (op 20336) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 20336) most significant half

s_add_u32 m0, m0, 4096 // (op 20336) 
buffer_load_dwordx4 v40, s[52:55], 64 offen offset: 0 lds // (op 20336) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v26 // (op 20336) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 20336) most significant half

s_add_u32 m0, m0, 4096 // (op 20336) 
buffer_load_dwordx4 v40, s[52:55], 64 offen offset: 0 lds // (op 20336) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v26 // (op 20336) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 20336) most significant half

s_add_u32 m0, m0, 4096 // (op 20336) 
buffer_load_dwordx4 v40, s[52:55], 64 offen offset: 0 lds // (op 20336) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 20694) 
v_sub_u32 v24, v1, v14 // (op 20694) 

v_lshlrev_b32 v14, 5, v24 // (op 20694) 

v_lshrrev_b32 v24, 2, v1 // (op 20694) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20694) 


v_lshlrev_b32 v14, 2, v26 // (op 20694) 

v_lshrrev_b32 v24, 3, v14 // (op 20694) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20694) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 20338) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20338) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20338) 


v_mov_b64 v[40:41], v[16:17] // (op 20338) 
s_add_u32 s1, 40960, s0 // (op 20338) 
s_mov_b32 m0, s1 // (op 20338) reg expression
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20338) Load value direct to lds
v_add_co_u32 v40, s[6:7], v40, v26 // (op 20338) least significant half
v_addc_co_u32 v41, s[6:7], 0, v41, s[6:7] // (op 20338) most significant half

s_add_u32 m0, m0, 4096 // (op 20338) 
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20338) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 20826) 
v_sub_u32 v24, v1, v14 // (op 20826) 

v_lshlrev_b32 v14, 5, v24 // (op 20826) 

v_lshrrev_b32 v24, 2, v1 // (op 20826) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20826) 


v_lshlrev_b32 v14, 2, v26 // (op 20826) 

v_lshrrev_b32 v24, 3, v14 // (op 20826) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20826) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 20340) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20340) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20340) 


v_mov_b64 v[40:41], v[20:21] // (op 20340) 
s_add_u32 s1, 49152, s0 // (op 20340) 
s_mov_b32 m0, s1 // (op 20340) reg expression
s_mov_b32 s6, 128 // (op 20340) call()
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20340) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20340) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20340) most significant half

s_add_u32 m0, m0, 4096 // (op 20340) 
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20340) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20340) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20340) most significant half

s_add_u32 m0, m0, 4096 // (op 20340) 
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20340) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20340) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20340) most significant half

s_add_u32 m0, m0, 4096 // (op 20340) 
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20340) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 20782) 
v_sub_u32 v24, v1, v14 // (op 20782) 

v_lshlrev_b32 v14, 5, v24 // (op 20782) 

v_lshrrev_b32 v24, 2, v1 // (op 20782) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20782) 


v_lshlrev_b32 v14, 2, v26 // (op 20782) 

v_lshrrev_b32 v24, 3, v14 // (op 20782) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20782) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 20342) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20342) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20342) 


v_mov_b64 v[40:41], v[16:17] // (op 20342) 
s_add_u32 s1, 65536, s0 // (op 20342) 
s_mov_b32 m0, s1 // (op 20342) reg expression
s_mov_b32 s6, 128 // (op 20342) call()
buffer_load_dwordx4 v40, s[48:51], s6 offen offset: 0 lds // (op 20342) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20342) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20342) most significant half

s_add_u32 m0, m0, 4096 // (op 20342) 
buffer_load_dwordx4 v40, s[48:51], s6 offen offset: 0 lds // (op 20342) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 20914) 
v_sub_u32 v24, v1, v14 // (op 20914) 

v_lshlrev_b32 v14, 5, v24 // (op 20914) 

v_lshrrev_b32 v24, 2, v1 // (op 20914) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20914) 


v_lshlrev_b32 v14, 2, v26 // (op 20914) 

v_lshrrev_b32 v24, 3, v14 // (op 20914) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20914) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 6 // (op 20344) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20344) 

v_lshrrev_b64 v[26:27], 3, s[10:11] // (op 20344) 


v_mov_b64 v[40:41], v[20:21] // (op 20344) 
s_add_u32 s1, 73728, s0 // (op 20344) 
s_mov_b32 m0, s1 // (op 20344) reg expression
s_mov_b32 s6, 192 // (op 20344) call()
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20344) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20344) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20344) most significant half

s_add_u32 m0, m0, 4096 // (op 20344) 
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20344) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20344) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20344) most significant half

s_add_u32 m0, m0, 4096 // (op 20344) 
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20344) Load value direct to lds
v_add_co_u32 v40, s[10:11], v40, v26 // (op 20344) least significant half
v_addc_co_u32 v41, s[10:11], 0, v41, s[10:11] // (op 20344) most significant half

s_add_u32 m0, m0, 4096 // (op 20344) 
buffer_load_dwordx4 v40, s[52:55], s6 offen offset: 0 lds // (op 20344) Load value direct to lds






v_and_b32 v14, -4, v1 // (op 20870) 
v_sub_u32 v24, v1, v14 // (op 20870) 

v_lshlrev_b32 v14, 5, v24 // (op 20870) 

v_lshrrev_b32 v24, 2, v1 // (op 20870) 
v_lshl_add_u32 v20, v24, 7, v14 // (op 20870) 


v_lshlrev_b32 v14, 2, v20 // (op 20870) 

v_lshrrev_b32 v24, 3, v14 // (op 20870) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20870) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[12:13], 6 // (op 20346) 
s_lshl_b64 s[10:11], s[6:7], 2 // (op 20346) 

v_lshrrev_b64 v[20:21], 3, s[10:11] // (op 20346) 


v_mov_b64 v[26:27], v[16:17] // (op 20346) 
s_add_u32 s1, 90112, s0 // (op 20346) 
s_mov_b32 m0, s1 // (op 20346) reg expression
s_mov_b32 s6, 192 // (op 20346) call()
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 20346) Load value direct to lds
v_add_co_u32 v26, s[10:11], v26, v20 // (op 20346) least significant half
v_addc_co_u32 v27, s[10:11], 0, v27, s[10:11] // (op 20346) most significant half

s_add_u32 m0, m0, 4096 // (op 20346) 
buffer_load_dwordx4 v26, s[48:51], s6 offen offset: 0 lds // (op 20346) Load value direct to lds






s_waitcnt vmcnt(23)
s_barrier  // (op 21471) 



ds_read_b128 v[40:43], v4 // (op 19634) Load local data 



ds_read_b128 v[44:47], v4 offset:1024 // (op 19639) Load local data 



ds_read_b128 v[48:51], v4 offset:2048 // (op 19644) Load local data 



ds_read_b128 v[52:55], v4 offset:3072 // (op 19649) Load local data 



ds_read_b128 v[56:59], v4 offset:4096 // (op 19654) Load local data 



ds_read_b128 v[60:63], v4 offset:5120 // (op 19659) Load local data 



ds_read_b128 v[64:67], v4 offset:6144 // (op 19664) Load local data 



ds_read_b128 v[68:71], v4 offset:7168 // (op 19669) Load local data 
s_waitcnt vmcnt(21)
s_barrier  // (op 21454) 



ds_read_b128 v[72:75], v3 offset:16384 // (op 19614) Load local data 



ds_read_b128 v[76:79], v3 offset:17408 // (op 19619) Load local data 



ds_read_b128 v[80:83], v3 offset:18432 // (op 19624) Load local data 



ds_read_b128 v[84:87], v3 offset:19456 // (op 19629) Load local data 
v_and_b32 v14, -4, v1 // (op 20542) 
v_sub_u32 v24, v1, v14 // (op 20542) 

v_lshlrev_b32 v14, 5, v24 // (op 20542) 

v_lshrrev_b32 v24, 2, v1 // (op 20542) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20542) 


v_lshlrev_b32 v14, 2, v16 // (op 20542) 

v_lshrrev_b32 v24, 3, v14 // (op 20542) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20542) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20599) 
v_sub_u32 v24, v1, v14 // (op 20599) 

v_lshlrev_b32 v14, 5, v24 // (op 20599) 

v_lshrrev_b32 v24, 2, v1 // (op 20599) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20599) 


v_lshlrev_b32 v14, 2, v16 // (op 20599) 

v_lshrrev_b32 v24, 3, v14 // (op 20599) 

s_nop 0
v_readfirstlane_b32 s1, v24 // (op 20599) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20679) 
v_sub_u32 v24, v1, v14 // (op 20679) 

v_lshlrev_b32 v14, 5, v24 // (op 20679) 

v_lshrrev_b32 v24, 2, v1 // (op 20679) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20679) 


v_lshlrev_b32 v14, 2, v16 // (op 20679) 

v_lshrrev_b32 v24, 3, v14 // (op 20679) 

s_nop 0
v_readfirstlane_b32 s6, v24 // (op 20679) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20723) 
v_sub_u32 v24, v1, v14 // (op 20723) 

v_lshlrev_b32 v14, 5, v24 // (op 20723) 

v_lshrrev_b32 v24, 2, v1 // (op 20723) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20723) 


v_lshlrev_b32 v14, 2, v16 // (op 20723) 

v_lshrrev_b32 v24, 3, v14 // (op 20723) 

s_nop 0
v_readfirstlane_b32 s7, v24 // (op 20723) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20767) 
v_sub_u32 v24, v1, v14 // (op 20767) 

v_lshlrev_b32 v14, 5, v24 // (op 20767) 

v_lshrrev_b32 v24, 2, v1 // (op 20767) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20767) 


v_lshlrev_b32 v14, 2, v16 // (op 20767) 

v_lshrrev_b32 v24, 3, v14 // (op 20767) 

s_nop 0
v_readfirstlane_b32 s10, v24 // (op 20767) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20811) 
v_sub_u32 v24, v1, v14 // (op 20811) 

v_lshlrev_b32 v14, 5, v24 // (op 20811) 

v_lshrrev_b32 v24, 2, v1 // (op 20811) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20811) 


v_lshlrev_b32 v14, 2, v16 // (op 20811) 

v_lshrrev_b32 v24, 3, v14 // (op 20811) 

s_nop 0
v_readfirstlane_b32 s11, v24 // (op 20811) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20855) 
v_sub_u32 v24, v1, v14 // (op 20855) 

v_lshlrev_b32 v14, 5, v24 // (op 20855) 

v_lshrrev_b32 v24, 2, v1 // (op 20855) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20855) 


v_lshlrev_b32 v14, 2, v16 // (op 20855) 

v_lshrrev_b32 v24, 3, v14 // (op 20855) 

s_nop 0
v_readfirstlane_b32 s14, v24 // (op 20855) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20899) 
v_sub_u32 v24, v1, v14 // (op 20899) 

v_lshlrev_b32 v14, 5, v24 // (op 20899) 

v_lshrrev_b32 v24, 2, v1 // (op 20899) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20899) 


v_lshlrev_b32 v14, 2, v16 // (op 20899) 

v_lshrrev_b32 v24, 3, v14 // (op 20899) 

s_nop 0
v_readfirstlane_b32 s15, v24 // (op 20899) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s18, 0 // (op 4403) call()
s_ashr_i64 s[38:39], s[36:37], 7 // (op 4402) 
s_mov_b32 s19, s38 // (op 4402) convert

s_and_b32 s38, s19, 4294967292 // (op 4402) 

s_cmp_lt_u32 s18, s38 // (op 4402) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopBottom_KLoop_4402 // (op 4402) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopBottom_KLoop_4402 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopTop_KLoop_4402:
s_add_u32 s18, s18, 4 // (op 4404) 
s_waitcnt vmcnt(20)
v_mov_b32 v88, v28 // (op 20205) reg expression
v_mov_b32 v89, v29 // (op 20205) reg expression
v_mov_b32 v90, v30 // (op 20205) reg expression
v_mov_b32 v91, v31 // (op 20205) reg expression
s_waitcnt lgkmcnt(4)
s_barrier  // (op 21474) 
s_waitcnt vmcnt(19)
v_mov_b32 v92, v32 // (op 20218) reg expression

v_mov_b32 v93, v33 // (op 20218) reg expression
v_mov_b32 v94, v34 // (op 20218) reg expression
v_mov_b32 v95, v35 // (op 20218) reg expression
s_lshl_b64 s[38:39], s[24:25], 6 // (op 20279) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20279) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20279) 


v_mov_b64 v[20:21], v[22:23] // (op 20279) 
v_mov_b64 v[26:27], v[20:21] // (op 20279) 
s_mov_b32 m0, s1 // (op 20279) reg expression
s_mov_b32 s19, 256 // (op 20279) call()
s_waitcnt vmcnt(18)
v_mov_b32 v96, v36 // (op 20231) reg expression
v_mov_b32 v97, v37 // (op 20231) reg expression
v_mov_b32 v98, v38 // (op 20231) reg expression
v_mov_b32 v99, v39 // (op 20231) reg expression
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 20279) Load value direct to lds
v_add_co_u32 v26, s[38:39], v26, v16 // (op 20279) least significant half
v_addc_co_u32 v27, s[38:39], 0, v27, s[38:39] // (op 20279) most significant half

s_add_u32 m0, m0, 4096 // (op 20279) 
v_bfe_u32 v100, v96, 0, 8 // (op 18393) 
v_bfe_u32 v101, v96, 8, 8 // (op 18393) 
v_bfe_u32 v102, v96, 16, 8 // (op 18393) 
v_bfe_u32 v103, v96, 24, 8 // (op 18393) 
v_permlane16_swap_b32 v100, v101 // (op 18393) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18393) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18393) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18393) 
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 20279) Load value direct to lds
v_add_co_u32 v26, s[38:39], v26, v16 // (op 20279) least significant half
v_addc_co_u32 v27, s[38:39], 0, v27, s[38:39] // (op 20279) most significant half

s_add_u32 m0, m0, 4096 // (op 20279) 
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 20279) Load value direct to lds
v_add_co_u32 v26, s[38:39], v26, v16 // (op 20279) least significant half
v_addc_co_u32 v27, s[38:39], 0, v27, s[38:39] // (op 20279) most significant half

s_add_u32 m0, m0, 4096 // (op 20279) 
v_bfe_u32 v104, v88, 0, 8 // (op 18586) 
v_bfe_u32 v105, v88, 8, 8 // (op 18586) 
v_bfe_u32 v106, v88, 16, 8 // (op 18586) 
v_bfe_u32 v107, v88, 24, 8 // (op 18586) 
v_permlane16_swap_b32 v104, v105 // (op 18586) 
s_nop 0
v_permlane16_swap_b32 v106, v107 // (op 18586) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v104, v106 // (op 18586) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v105, v107 // (op 18586) 
buffer_load_dwordx4 v26, s[52:55], s19 offen offset: 0 lds // (op 20279) Load value direct to lds



s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[40:43], a[0:3], v100, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 76) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21457) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[44:47], a[4:7], v100, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 417) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 20274) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[48:51], a[8:11], v100, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 643) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20274) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[52:55], a[12:15], v100, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 869) 
v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20274) 


v_mov_b64 v[26:27], v[18:19] // (op 20274) 
v_mov_b64 v[108:109], v[26:27] // (op 20274) 
s_add_u32 s19, 16384, s0 // (op 20274) 
s_mov_b32 m0, s19 // (op 20274) reg expression
s_mov_b32 s38, 256 // (op 20274) call()
buffer_load_dwordx4 v108, s[48:51], s38 offen offset: 0 lds // (op 20274) Load value direct to lds
v_add_co_u32 v108, s[64:65], v108, v16 // (op 20274) least significant half
v_addc_co_u32 v109, s[64:65], 0, v109, s[64:65] // (op 20274) most significant half

s_add_u32 m0, m0, 4096 // (op 20274) 
buffer_load_dwordx4 v108, s[48:51], s38 offen offset: 0 lds // (op 20274) Load value direct to lds




v_bfe_u32 v108, v92, 0, 8 // (op 18654) 


v_bfe_u32 v109, v92, 8, 8 // (op 18654) 
v_bfe_u32 v110, v92, 16, 8 // (op 18654) 
v_bfe_u32 v111, v92, 24, 8 // (op 18654) 
v_permlane16_swap_b32 v108, v109 // (op 18654) 
v_mov_b32 v14, v15 // (op 20213) 
v_permlane16_swap_b32 v110, v111 // (op 18654) 
v_add_u32 v24, 16, v14 // (op 20213) 

s_nop 0
v_permlane32_swap_b32 v108, v110 // (op 18654) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 18654) 
buffer_load_dwordx4 v[28:31], v24, s[60:63], 0 offen // (op 20213) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:75], v[56:59], a[16:19], v100, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1095) 
s_lshl_b32 s19, s32, 6 // (op 20226) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[60:63], a[20:23], v100, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1321) 
s_add_u32 s19, s19, 16 // (op 20226) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[64:67], a[24:27], v100, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1547) 
v_add_u32 v24, v14, s19 // (op 20226) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[68:71], a[28:31], v100, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1773) 
buffer_load_dwordx4 v[32:35], v24, s[60:63], 0 offen // (op 20226) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[40:43], a[32:35], v101, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2993) 
v_add_u32 v15, 16, v15 // (op 21016) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[44:47], a[36:39], v101, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2917) 
v_mov_b32 v14, v25 // (op 20239) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[48:51], a[40:43], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2841) 
v_add_u32 v24, 16, v14 // (op 20239) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[52:55], a[44:47], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2765) 
buffer_load_dwordx4 v[36:39], v24, s[56:59], 0 offen // (op 20239) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[56:59], a[48:51], v101, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2689) 
v_add_u32 v25, 16, v25 // (op 20964) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[60:63], a[52:55], v101, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2613) 
s_waitcnt vmcnt(23)
s_barrier  // (op 21494) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[64:67], a[56:59], v101, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2537) 
ds_read_b128 v[112:115], v6 offset:24576 // (op 212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[68:71], a[60:63], v101, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2461) 
ds_read_b128 v[116:119], v6 offset:25600 // (op 428) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[80:83], v[40:43], a[64:67], v102, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4995) 
ds_read_b128 v[120:123], v6 offset:26624 // (op 654) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[80:83], v[44:47], a[68:71], v102, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4919) 
ds_read_b128 v[124:127], v6 offset:27648 // (op 880) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[48:51], a[72:75], v102, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4843) 
ds_read_b128 v[128:131], v6 offset:28672 // (op 1106) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[80:83], v[52:55], a[76:79], v102, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4767) 
ds_read_b128 v[132:135], v6 offset:29696 // (op 1332) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:83], v[56:59], a[80:83], v102, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4691) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:83], v[60:63], a[84:87], v102, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4615) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:83], v[64:67], a[88:91], v102, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4539) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:83], v[68:71], a[92:95], v102, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4463) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[84:87], v[40:43], a[96:99], v103, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6997) 
ds_read_b128 v[136:139], v6 offset:30720 // (op 1558) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:87], v[44:47], a[100:103], v103, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6921) 
ds_read_b128 v[140:143], v6 offset:31744 // (op 1784) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[84:87], v[48:51], a[104:107], v103, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6845) 
s_waitcnt vmcnt(21)
s_barrier  // (op 21487) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[84:87], v[52:55], a[108:111], v103, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6769) 
ds_read_b128 v[72:75], v5 offset:40960 // (op 210) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[56:59], a[112:115], v103, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6693) 
ds_read_b128 v[76:79], v5 offset:41984 // (op 2471) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[60:63], a[116:119], v103, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6617) 
ds_read_b128 v[80:83], v5 offset:43008 // (op 4473) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[64:67], a[120:123], v103, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6541) 
ds_read_b128 v[104:107], v5 offset:44032 // (op 6475) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[68:71], a[124:127], v103, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6465) 


v_bfe_u32 v100, v97, 0, 8 // (op 18408) 
v_bfe_u32 v101, v97, 8, 8 // (op 18408) 
v_bfe_u32 v102, v97, 16, 8 // (op 18408) 
v_bfe_u32 v103, v97, 24, 8 // (op 18408) 
v_permlane16_swap_b32 v100, v101 // (op 18408) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18408) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18408) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18408) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 21498) 

s_lshl_b64 s[38:39], s[24:25], 6 // (op 20301) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20301) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20301) 


v_mov_b64 v[108:109], v[20:21] // (op 20301) 
s_add_u32 s19, 24576, s7 // (op 20301) 
s_mov_b32 m0, s19 // (op 20301) reg expression
s_mov_b32 s38, 320 // (op 20301) call()
v_bfe_u32 v144, v89, 0, 8 // (op 18601) 
v_bfe_u32 v145, v89, 8, 8 // (op 18601) 
v_bfe_u32 v146, v89, 16, 8 // (op 18601) 
v_bfe_u32 v147, v89, 24, 8 // (op 18601) 
v_permlane16_swap_b32 v144, v145 // (op 18601) 
s_nop 0
v_permlane16_swap_b32 v146, v147 // (op 18601) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v144, v146 // (op 18601) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v145, v147 // (op 18601) 
buffer_load_dwordx4 v108, s[52:55], s38 offen offset: 0 lds // (op 20301) Load value direct to lds
v_add_co_u32 v108, s[64:65], v108, v16 // (op 20301) least significant half
v_addc_co_u32 v109, s[64:65], 0, v109, s[64:65] // (op 20301) most significant half

s_add_u32 m0, m0, 4096 // (op 20301) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[112:115], a[0:3], v100, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 217) 
buffer_load_dwordx4 v108, s[52:55], s38 offen offset: 0 lds // (op 20301) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[116:119], a[4:7], v100, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 418) 
v_add_co_u32 v108, s[64:65], v108, v16 // (op 20301) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[120:123], a[8:11], v100, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 644) 
v_addc_co_u32 v109, s[64:65], 0, v109, s[64:65] // (op 20301) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[124:127], a[12:15], v100, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 870) 
s_add_u32 m0, m0, 4096 // (op 20301) 
v_bfe_u32 v148, v93, 0, 8 // (op 18669) 
v_bfe_u32 v149, v93, 8, 8 // (op 18669) 
v_bfe_u32 v150, v93, 16, 8 // (op 18669) 
v_bfe_u32 v151, v93, 24, 8 // (op 18669) 
buffer_load_dwordx4 v108, s[52:55], s38 offen offset: 0 lds // (op 20301) Load value direct to lds
v_permlane16_swap_b32 v148, v149 // (op 18669) 
v_permlane16_swap_b32 v150, v151 // (op 18669) 
v_add_co_u32 v108, s[64:65], v108, v16 // (op 20301) least significant half
v_addc_co_u32 v109, s[64:65], 0, v109, s[64:65] // (op 20301) most significant half

s_add_u32 m0, m0, 4096 // (op 20301) 
v_permlane32_swap_b32 v148, v150 // (op 18669) 
v_permlane32_swap_b32 v149, v151 // (op 18669) 
buffer_load_dwordx4 v108, s[52:55], s38 offen offset: 0 lds // (op 20301) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:75], v[128:131], a[16:19], v100, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1096) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21491) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[132:135], a[20:23], v100, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1322) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 20296) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[136:139], a[24:27], v100, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1548) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20296) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[140:143], a[28:31], v100, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1774) 
v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20296) 


v_mov_b64 v[108:109], v[26:27] // (op 20296) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[112:115], a[32:35], v101, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2994) 
s_add_u32 s19, 40960, s6 // (op 20296) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[116:119], a[36:39], v101, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2918) 
s_mov_b32 m0, s19 // (op 20296) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[120:123], a[40:43], v101, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2842) 
s_mov_b32 s38, 320 // (op 20296) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[124:127], a[44:47], v101, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2766) 
buffer_load_dwordx4 v108, s[48:51], s38 offen offset: 0 lds // (op 20296) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[128:131], a[48:51], v101, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2690) 
v_add_co_u32 v108, s[64:65], v108, v16 // (op 20296) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[132:135], a[52:55], v101, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2614) 
v_addc_co_u32 v109, s[64:65], 0, v109, s[64:65] // (op 20296) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[136:139], a[56:59], v101, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2538) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[140:143], a[60:63], v101, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2462) 
s_add_u32 m0, m0, 4096 // (op 20296) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[80:83], v[112:115], a[64:67], v102, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4996) 
buffer_load_dwordx4 v108, s[48:51], s38 offen offset: 0 lds // (op 20296) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[80:83], v[116:119], a[68:71], v102, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4920) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[120:123], a[72:75], v102, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4844) 
s_waitcnt vmcnt(23)
s_barrier  // (op 21509) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[80:83], v[124:127], a[76:79], v102, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4768) 
ds_read_b128 v[40:43], v8 offset:49152 // (op 240) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:83], v[128:131], a[80:83], v102, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4692) 
ds_read_b128 v[44:47], v8 offset:50176 // (op 438) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:83], v[132:135], a[84:87], v102, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4616) 
ds_read_b128 v[48:51], v8 offset:51200 // (op 664) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:83], v[136:139], a[88:91], v102, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4540) 
ds_read_b128 v[52:55], v8 offset:52224 // (op 890) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:83], v[140:143], a[92:95], v102, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4464) 
ds_read_b128 v[56:59], v8 offset:53248 // (op 1116) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[112:115], a[96:99], v103, v144 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6998) 
ds_read_b128 v[60:63], v8 offset:54272 // (op 1342) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[116:119], a[100:103], v103, v145 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6922) 
ds_read_b128 v[64:67], v8 offset:55296 // (op 1568) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[120:123], a[104:107], v103, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6846) 
ds_read_b128 v[108:111], v8 offset:56320 // (op 1794) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[124:127], a[108:111], v103, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6770) 
s_waitcnt vmcnt(21)
s_barrier  // (op 21502) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[128:131], a[112:115], v103, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6694) 
v_add_u32 v14, 65536, v7 // (op 238) 

ds_read_b128 v[72:75], v14 // (op 238) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[132:135], a[116:119], v103, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6618) 
v_add_u32 v14, 66560, v7 // (op 2481) 

ds_read_b128 v[76:79], v14 // (op 2481) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:107], v[136:139], a[120:123], v103, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6542) 
v_add_u32 v14, 67584, v7 // (op 4483) 

ds_read_b128 v[80:83], v14 // (op 4483) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:107], v[140:143], a[124:127], v103, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6466) 
v_add_u32 v14, 68608, v7 // (op 6485) 




ds_read_b128 v[84:87], v14 // (op 6485) Load local data 

v_bfe_u32 v100, v98, 0, 8 // (op 18425) 
v_bfe_u32 v101, v98, 8, 8 // (op 18425) 
v_bfe_u32 v102, v98, 16, 8 // (op 18425) 
v_bfe_u32 v103, v98, 24, 8 // (op 18425) 
v_permlane16_swap_b32 v100, v101 // (op 18425) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18425) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18425) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18425) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 21512) 

s_lshl_b64 s[38:39], s[24:25], 6 // (op 20311) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20311) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20311) 


v_mov_b64 v[104:105], v[20:21] // (op 20311) 
s_add_u32 s19, 49152, s11 // (op 20311) 
s_mov_b32 m0, s19 // (op 20311) reg expression
s_mov_b32 s38, 384 // (op 20311) call()
v_bfe_u32 v112, v90, 0, 8 // (op 18619) 
v_bfe_u32 v113, v90, 8, 8 // (op 18619) 
v_bfe_u32 v114, v90, 16, 8 // (op 18619) 
v_bfe_u32 v115, v90, 24, 8 // (op 18619) 
v_permlane16_swap_b32 v112, v113 // (op 18619) 
s_nop 0
v_permlane16_swap_b32 v114, v115 // (op 18619) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v112, v114 // (op 18619) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v113, v115 // (op 18619) 
buffer_load_dwordx4 v104, s[52:55], s38 offen offset: 0 lds // (op 20311) Load value direct to lds
v_add_co_u32 v104, s[64:65], v104, v16 // (op 20311) least significant half
v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20311) most significant half

s_add_u32 m0, m0, 4096 // (op 20311) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[40:43], a[0:3], v100, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 245) 
buffer_load_dwordx4 v104, s[52:55], s38 offen offset: 0 lds // (op 20311) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[44:47], a[4:7], v100, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 419) 
v_add_co_u32 v104, s[64:65], v104, v16 // (op 20311) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[48:51], a[8:11], v100, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 645) 
v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20311) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[52:55], a[12:15], v100, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 871) 
s_add_u32 m0, m0, 4096 // (op 20311) 
v_bfe_u32 v116, v94, 0, 8 // (op 18687) 
v_bfe_u32 v117, v94, 8, 8 // (op 18687) 
v_bfe_u32 v118, v94, 16, 8 // (op 18687) 
v_bfe_u32 v119, v94, 24, 8 // (op 18687) 
v_permlane16_swap_b32 v116, v117 // (op 18687) 
s_nop 0
v_permlane16_swap_b32 v118, v119 // (op 18687) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v116, v118 // (op 18687) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v117, v119 // (op 18687) 
buffer_load_dwordx4 v104, s[52:55], s38 offen offset: 0 lds // (op 20311) Load value direct to lds
v_add_co_u32 v104, s[64:65], v104, v16 // (op 20311) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:75], v[56:59], a[16:19], v100, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1097) 
v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20311) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[60:63], a[20:23], v100, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1323) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[64:67], a[24:27], v100, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1549) 
s_add_u32 m0, m0, 4096 // (op 20311) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[108:111], a[28:31], v100, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1775) 
buffer_load_dwordx4 v104, s[52:55], s38 offen offset: 0 lds // (op 20311) Load value direct to lds




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[40:43], a[32:35], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2995) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21506) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[44:47], a[36:39], v101, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2919) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 20306) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[48:51], a[40:43], v101, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2843) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20306) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[52:55], a[44:47], v101, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2767) 
v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20306) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[56:59], a[48:51], v101, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2691) 
v_mov_b64 v[104:105], v[26:27] // (op 20306) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[60:63], a[52:55], v101, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2615) 
s_add_u32 s19, 65536, s10 // (op 20306) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[64:67], a[56:59], v101, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2539) 
s_mov_b32 m0, s19 // (op 20306) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[108:111], a[60:63], v101, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2463) 
s_mov_b32 s38, 384 // (op 20306) call()
buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20306) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[80:83], v[40:43], a[64:67], v102, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4997) 
v_add_co_u32 v104, s[64:65], v104, v16 // (op 20306) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[80:83], v[44:47], a[68:71], v102, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4921) 
v_addc_co_u32 v105, s[64:65], 0, v105, s[64:65] // (op 20306) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[48:51], a[72:75], v102, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4845) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[80:83], v[52:55], a[76:79], v102, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4769) 
s_add_u32 m0, m0, 4096 // (op 20306) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:83], v[56:59], a[80:83], v102, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4693) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:83], v[60:63], a[84:87], v102, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4617) 
buffer_load_dwordx4 v104, s[48:51], s38 offen offset: 0 lds // (op 20306) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:83], v[64:67], a[88:91], v102, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4541) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:83], v[108:111], a[92:95], v102, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4465) 
s_waitcnt vmcnt(23)
s_barrier  // (op 21522) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[84:87], v[40:43], a[96:99], v103, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6999) 
v_add_u32 v14, 73728, v10 // (op 268) 
ds_read_b128 v[68:71], v14 // (op 268) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:87], v[44:47], a[100:103], v103, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6923) 
v_add_u32 v14, 74752, v10 // (op 448) 
ds_read_b128 v[104:107], v14 // (op 448) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[84:87], v[48:51], a[104:107], v103, v114 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6847) 
v_add_u32 v14, 75776, v10 // (op 674) 
ds_read_b128 v[120:123], v14 // (op 674) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[84:87], v[52:55], a[108:111], v103, v115 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6771) 
v_add_u32 v14, 76800, v10 // (op 900) 

ds_read_b128 v[112:115], v14 // (op 900) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[56:59], a[112:115], v103, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6695) 
v_add_u32 v14, 77824, v10 // (op 1126) 
ds_read_b128 v[124:127], v14 // (op 1126) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[60:63], a[116:119], v103, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6619) 
v_add_u32 v14, 78848, v10 // (op 1352) 
ds_read_b128 v[128:131], v14 // (op 1352) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[64:67], a[120:123], v103, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6543) 
v_add_u32 v14, 79872, v10 // (op 1578) 
ds_read_b128 v[132:135], v14 // (op 1578) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[108:111], a[124:127], v103, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6467) 
v_add_u32 v14, 80896, v10 // (op 1804) 



ds_read_b128 v[100:103], v14 // (op 1804) Load local data 

s_waitcnt vmcnt(21)
s_barrier  // (op 21515) 



v_add_u32 v14, 90112, v9 // (op 266) 
ds_read_b128 v[72:75], v14 // (op 266) Load local data 




v_add_u32 v14, 91136, v9 // (op 2491) 
ds_read_b128 v[76:79], v14 // (op 2491) Load local data 




v_add_u32 v14, 92160, v9 // (op 4493) 
ds_read_b128 v[80:83], v14 // (op 4493) Load local data 




v_add_u32 v14, 93184, v9 // (op 6495) 
ds_read_b128 v[108:111], v14 // (op 6495) Load local data 

v_bfe_u32 v116, v99, 0, 8 // (op 18442) 
v_bfe_u32 v117, v99, 8, 8 // (op 18442) 
v_bfe_u32 v118, v99, 16, 8 // (op 18442) 
v_bfe_u32 v119, v99, 24, 8 // (op 18442) 
v_permlane16_swap_b32 v116, v117 // (op 18442) 
s_nop 0
v_permlane16_swap_b32 v118, v119 // (op 18442) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v116, v118 // (op 18442) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v117, v119 // (op 18442) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 21525) 


s_lshl_b64 s[38:39], s[24:25], 6 // (op 20321) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20321) 

v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20321) 


v_mov_b64 v[96:97], v[20:21] // (op 20321) 
s_add_u32 s19, 73728, s15 // (op 20321) 
s_mov_b32 m0, s19 // (op 20321) reg expression
s_mov_b32 s38, 448 // (op 20321) call()
v_bfe_u32 v136, v91, 0, 8 // (op 18637) 
v_bfe_u32 v137, v91, 8, 8 // (op 18637) 
v_bfe_u32 v138, v91, 16, 8 // (op 18637) 
v_bfe_u32 v139, v91, 24, 8 // (op 18637) 
v_permlane16_swap_b32 v136, v137 // (op 18637) 
s_nop 0
v_permlane16_swap_b32 v138, v139 // (op 18637) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 18637) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 18637) 
buffer_load_dwordx4 v96, s[52:55], s38 offen offset: 0 lds // (op 20321) Load value direct to lds

v_add_co_u32 v96, s[64:65], v96, v16 // (op 20321) least significant half
v_addc_co_u32 v97, s[64:65], 0, v97, s[64:65] // (op 20321) most significant half

s_add_u32 m0, m0, 4096 // (op 20321) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:75], v[68:71], a[0:3], v116, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 273) 
buffer_load_dwordx4 v96, s[52:55], s38 offen offset: 0 lds // (op 20321) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:75], v[104:107], a[4:7], v116, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 420) 
v_add_co_u32 v96, s[64:65], v96, v16 // (op 20321) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[120:123], a[8:11], v116, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 646) 
v_addc_co_u32 v97, s[64:65], 0, v97, s[64:65] // (op 20321) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[112:115], a[12:15], v116, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 872) 
s_add_u32 m0, m0, 4096 // (op 20321) 
v_bfe_u32 v88, v95, 0, 8 // (op 18705) 
v_bfe_u32 v89, v95, 8, 8 // (op 18705) 
v_bfe_u32 v90, v95, 16, 8 // (op 18705) 
v_bfe_u32 v91, v95, 24, 8 // (op 18705) 
v_permlane16_swap_b32 v88, v89 // (op 18705) 
s_nop 0
v_permlane16_swap_b32 v90, v91 // (op 18705) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v88, v90 // (op 18705) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v89, v91 // (op 18705) 
buffer_load_dwordx4 v96, s[52:55], s38 offen offset: 0 lds // (op 20321) Load value direct to lds

v_add_co_u32 v96, s[64:65], v96, v16 // (op 20321) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:75], v[124:127], a[16:19], v116, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1098) 
v_addc_co_u32 v97, s[64:65], 0, v97, s[64:65] // (op 20321) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:75], v[128:131], a[20:23], v116, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1324) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[132:135], a[24:27], v116, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1550) 
s_add_u32 m0, m0, 4096 // (op 20321) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[100:103], a[28:31], v116, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1776) 
buffer_load_dwordx4 v96, s[52:55], s38 offen offset: 0 lds // (op 20321) Load value direct to lds





s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[68:71], a[32:35], v117, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2996) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[104:107], a[36:39], v117, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2920) 
v_add_co_u32 v22, vcc, 256, v22 // (op 20491) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[120:123], a[40:43], v117, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2844) 
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 20491) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[112:115], a[44:47], v117, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2768) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21519) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[124:127], a[48:51], v117, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2692) 
s_lshl_b64 s[38:39], s[12:13], 6 // (op 20316) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[128:131], a[52:55], v117, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2616) 
s_lshl_b64 s[64:65], s[38:39], 2 // (op 20316) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[132:135], a[56:59], v117, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2540) 
v_lshrrev_b64 v[16:17], 3, s[64:65] // (op 20316) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[100:103], a[60:63], v117, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2464) 
v_mov_b64 v[20:21], v[26:27] // (op 20316) 
s_add_u32 s19, 90112, s14 // (op 20316) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[80:83], v[68:71], a[64:67], v118, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4998) 
s_mov_b32 m0, s19 // (op 20316) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[80:83], v[104:107], a[68:71], v118, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4922) 
s_mov_b32 s38, 448 // (op 20316) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[120:123], a[72:75], v118, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4846) 
buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 20316) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[80:83], v[112:115], a[76:79], v118, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4770) 
v_add_co_u32 v20, s[64:65], v20, v16 // (op 20316) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:83], v[124:127], a[80:83], v118, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4694) 
v_addc_co_u32 v21, s[64:65], 0, v21, s[64:65] // (op 20316) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:83], v[128:131], a[84:87], v118, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4618) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:83], v[132:135], a[88:91], v118, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4542) 
s_add_u32 m0, m0, 4096 // (op 20316) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:83], v[100:103], a[92:95], v118, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4466) 
buffer_load_dwordx4 v20, s[48:51], s38 offen offset: 0 lds // (op 20316) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[108:111], v[68:71], a[96:99], v119, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7000) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[108:111], v[104:107], a[100:103], v119, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6924) 
v_add_co_u32 v18, vcc, 256, v18 // (op 20427) least significant half

v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 20427) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:111], v[120:123], a[104:107], v119, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6848) 
s_waitcnt vmcnt(23)
s_barrier  // (op 21480) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[112:115], a[108:111], v119, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6772) 
ds_read_b128 v[40:43], v4 // (op 65) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[124:127], a[112:115], v119, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6696) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 414) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:111], v[128:131], a[116:119], v119, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6620) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 640) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[108:111], v[132:135], a[120:123], v119, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6544) 
ds_read_b128 v[52:55], v4 offset:3072 // (op 866) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[108:111], v[100:103], a[124:127], v119, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6468) 
ds_read_b128 v[56:59], v4 offset:4096 // (op 1092) Load local data 







ds_read_b128 v[60:63], v4 offset:5120 // (op 1318) Load local data 



ds_read_b128 v[64:67], v4 offset:6144 // (op 1544) Load local data 



ds_read_b128 v[68:71], v4 offset:7168 // (op 1770) Load local data 
s_waitcnt vmcnt(21)
s_barrier  // (op 21463) 



ds_read_b128 v[72:75], v3 offset:16384 // (op 60) Load local data 



ds_read_b128 v[76:79], v3 offset:17408 // (op 2457) Load local data 



ds_read_b128 v[80:83], v3 offset:18432 // (op 4459) Load local data 



ds_read_b128 v[84:87], v3 offset:19456 // (op 6461) Load local data 
s_ashr_i64 s[38:39], s[36:37], 7 // (op 4402) 
s_mov_b32 s19, s38 // (op 4402) convert

s_and_b32 s38, s19, 4294967292 // (op 4402) 

s_cmp_lt_u32 s18, s38 // (op 4402) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopTop_KLoop_4402 // (op 4402) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopTop_KLoop_4402 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopBottom_KLoop_4402:















s_barrier  // (op 21578) 
v_lshrrev_b32 v14, 2, v1 // (op 20359) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 20359) 
v_lshl_add_u32 v15, s16, 7, v14 // (op 20359) 

s_lshr_b32 s6, s0, 2 // (op 20359) 

v_and_b32 v14, -4, v1 // (op 20359) 
v_sub_u32 v16, v1, v14 // (op 20359) 

v_lshlrev_b32 v14, 5, v16 // (op 20359) 

v_lshl_add_u32 v16, s6, 9, v14 // (op 20359) 


v_mov_b32 v27, 0 // (op 20359) convert
v_mov_b32 v26, v16 // (op 20359) convert

v_mul_lo_u32 v17, v15, s13 // (op 20359) most significant: low of low * high
v_mul_hi_u32 v14, v15, s12 // (op 20359) most significant: high of low * low
v_mul_lo_u32 v16, v15, s12 // (op 20359) least significant: low of low * low
v_add_u32 v17, v17, v14 // (op 20359) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v26 // (op 20359) least significant half
v_addc_co_u32 v17, s[0:1], v17, v27, s[0:1] // (op 20359) most significant half



v_lshlrev_b64 v[14:15], 2, v[16:17] // (op 20359) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 20359) 

s_mul_i32 s1, s12, s23 // (op 20361) most significant: low of low * high
s_mul_i32 s6, s13, s22 // (op 20361) most significant: low of high * low
s_mul_i32 s0, s12, s22 // (op 20361) least significant
s_mul_hi_u32 s7, s12, s22 // (op 20361) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20361) most significant: sum
s_add_u32 s1, s1, s7 // (op 20361) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20361) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20361) 

s_mov_b32 s48, s20 // (op 20361) reg expression
s_mov_b32 s49, s21 // (op 20361) reg expression
s_mov_b32 s50, s0 // (op 20361) reg expression
s_mov_b32 s51, 131072 // (op 20361) call()

v_and_b32 v14, -4, v1 // (op 20457) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 20457) 
v_sub_u32 v15, v1, v14 // (op 20457) 

s_lshr_b32 s6, s0, 2 // (op 20457) 

v_lshlrev_b32 v14, 5, v15 // (op 20457) 

v_lshl_add_u32 v15, s6, 9, v14 // (op 20457) 


v_mov_b32 v19, 0 // (op 20457) convert
v_mov_b32 v18, v15 // (op 20457) convert

v_lshrrev_b32 v14, 2, v1 // (op 20457) 
v_lshl_add_u32 v15, s17, 8, v14 // (op 20457) 

v_mul_lo_u32 v21, v15, s25 // (op 20457) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 20457) most significant: high of low * low
v_mul_lo_u32 v20, v15, s24 // (op 20457) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 20457) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v20 // (op 20457) least significant half
v_addc_co_u32 v15, s[0:1], v19, v21, s[0:1] // (op 20457) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20457) 


v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 20457) 

s_mul_i32 s1, s24, s9 // (op 20459) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 20459) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 20459) least significant
s_mul_hi_u32 s7, s24, s8 // (op 20459) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 20459) most significant: sum
s_add_u32 s1, s1, s7 // (op 20459) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 20459) 

s_ashr_i64 s[0:1], s[6:7], 3 // (op 20459) 

s_mov_b32 s52, s26 // (op 20459) reg expression
s_mov_b32 s53, s27 // (op 20459) reg expression
s_mov_b32 s54, s0 // (op 20459) reg expression
s_mov_b32 s55, 131072 // (op 20459) call()

v_and_b32 v14, -4, v1 // (op 20555) 
v_sub_u32 v15, v1, v14 // (op 20555) 

v_lshlrev_b32 v14, 5, v15 // (op 20555) 

v_lshrrev_b32 v15, 2, v1 // (op 20555) 
v_lshl_add_u32 v20, v15, 7, v14 // (op 20555) 


v_lshlrev_b32 v14, 2, v20 // (op 20555) 

v_lshrrev_b32 v15, 3, v14 // (op 20555) 

s_nop 0
v_readfirstlane_b32 s0, v15 // (op 20555) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20612) 
v_sub_u32 v15, v1, v14 // (op 20612) 

v_lshlrev_b32 v14, 5, v15 // (op 20612) 

v_lshrrev_b32 v15, 2, v1 // (op 20612) 
v_lshl_add_u32 v20, v15, 7, v14 // (op 20612) 


v_lshlrev_b32 v14, 2, v20 // (op 20612) 

v_lshrrev_b32 v15, 3, v14 // (op 20612) 

s_nop 0
v_readfirstlane_b32 s1, v15 // (op 20612) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, 63, v1 // (op 21053) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 21053) 
v_lshrrev_b32 v15, 4, v14 // (op 21053) 
s_lshr_b32 s10, s6, 2 // (op 21053) 

v_and_b32 v20, 3, v15 // (op 21053) 

v_and_b32 v15, 15, v1 // (op 21053) 
v_lshl_add_u32 v21, v20, 4, v15 // (op 21053) 


v_lshrrev_b32 v15, 7, v1 // (op 21053) 
v_lshl_add_u32 v20, v15, 6, v21 // (op 21053) 


v_lshl_add_u32 v15, s16, 7, v20 // (op 21053) 

v_lshrrev_b32 v20, 6, v14 // (op 21053) 

v_lshlrev_b32 v14, 2, v20 // (op 21053) 

v_lshl_add_u32 v20, s10, 4, v14 // (op 21053) 


v_mul_lo_u32 v14, v15, s28 // (op 21053) 
v_add_u32 v14, v14, v20 // (op 21053) 


s_mul_i32 s7, s28, s23 // (op 21055) most significant: low of low * high
s_mul_i32 s10, s29, s22 // (op 21055) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 21055) least significant
s_mul_hi_u32 s11, s28, s22 // (op 21055) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 21055) most significant: sum
s_add_u32 s7, s7, s11 // (op 21055) most significant: sum


s_mov_b32 s56, s30 // (op 21055) reg expression
s_mov_b32 s57, s31 // (op 21055) reg expression
s_mov_b32 s58, s6 // (op 21055) reg expression
s_mov_b32 s59, 131072 // (op 21055) call()

v_and_b32 v15, 63, v1 // (op 21080) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 21080) 
v_lshrrev_b32 v20, 6, v15 // (op 21080) 
s_lshr_b32 s10, s6, 2 // (op 21080) 

v_lshlrev_b32 v21, 2, v20 // (op 21080) 

v_lshl_add_u32 v20, s10, 4, v21 // (op 21080) 


v_lshrrev_b32 v21, 4, v15 // (op 21080) 

v_and_b32 v15, 3, v21 // (op 21080) 

v_and_b32 v21, 15, v1 // (op 21080) 
v_lshl_add_u32 v22, v15, 4, v21 // (op 21080) 


v_lshrrev_b32 v15, 6, v1 // (op 21080) 
v_and_b32 v21, 1, v15 // (op 21080) 

v_lshl_add_u32 v15, v21, 7, v22 // (op 21080) 


v_lshl_add_u32 v21, s17, 8, v15 // (op 21080) 

v_mul_lo_u32 v15, v21, s32 // (op 21080) 

v_add_u32 v21, v20, v15 // (op 21080) 


s_mul_i32 s7, s32, s9 // (op 21082) most significant: low of low * high
s_mul_i32 s10, s33, s8 // (op 21082) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 21082) least significant
s_mul_hi_u32 s11, s32, s8 // (op 21082) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 21082) most significant: sum
s_add_u32 s7, s7, s11 // (op 21082) most significant: sum


s_mov_b32 s60, s34 // (op 21082) reg expression
s_mov_b32 s61, s35 // (op 21082) reg expression
s_mov_b32 s62, s6 // (op 21082) reg expression
s_mov_b32 s63, 131072 // (op 21082) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 6410) 
s_and_b32 s10, s6, 4294967292 // (op 6410) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 6409) 
s_mov_b32 s11, s6 // (op 6409) convert

s_cmp_lt_u32 s10, s11 // (op 6409) 
















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopBottom_KLoopTail_6409 // (op 6409) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopBottom_KLoopTail_6409 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopTop_KLoopTail_6409:
s_barrier  // (op 21460) 
s_add_u32 s10, s10, 1 // (op 6411) 

s_lshl_b64 s[6:7], s[12:13], 6 // (op 20284) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 20284) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 20284) 


v_mov_b64 v[24:25], v[16:17] // (op 20284) 
v_mov_b64 v[26:27], v[24:25] // (op 20284) 
s_add_u32 s11, 16384, s0 // (op 20284) 
s_mov_b32 m0, s11 // (op 20284) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 20284) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 20284) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 20284) most significant half

s_add_u32 m0, m0, 4096 // (op 20284) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 20284) Load value direct to lds




v_add_co_u32 v16, s[6:7], 64, v16 // (op 20349) least significant half
v_addc_co_u32 v17, s[6:7], 0, v17, s[6:7] // (op 20349) most significant half

s_barrier  // (op 21477) 

s_lshl_b64 s[6:7], s[24:25], 6 // (op 20290) 
s_lshl_b64 s[14:15], s[6:7], 2 // (op 20290) 

v_lshrrev_b64 v[22:23], 3, s[14:15] // (op 20290) 


v_mov_b64 v[24:25], v[18:19] // (op 20290) 
v_mov_b64 v[26:27], v[24:25] // (op 20290) 
s_mov_b32 m0, s1 // (op 20290) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 20290) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 20290) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 20290) most significant half

s_add_u32 m0, m0, 4096 // (op 20290) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 20290) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 20290) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 20290) most significant half

s_add_u32 m0, m0, 4096 // (op 20290) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 20290) Load value direct to lds
v_add_co_u32 v26, s[6:7], v26, v22 // (op 20290) least significant half
v_addc_co_u32 v27, s[6:7], 0, v27, s[6:7] // (op 20290) most significant half

s_add_u32 m0, m0, 4096 // (op 20290) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 20290) Load value direct to lds



v_add_co_u32 v18, s[6:7], 64, v18 // (op 20449) least significant half
v_addc_co_u32 v19, s[6:7], 0, v19, s[6:7] // (op 20449) most significant half

s_waitcnt vmcnt(4)
s_barrier  // (op 21467) 



ds_read_b128 v[22:25], v3 offset:16384 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 21483) 



ds_read_b128 v[26:29], v4 // (op 162) Load local data 


v_mov_b32 v15, v14 // (op 164) 
v_mov_b32 v20, v15 // (op 164) reg expression

buffer_load_dwordx4 v[30:33], v20, s[56:59], 0 offen // (op 164) Load value


v_add_u32 v14, 4, v14 // (op 21043) 


v_mov_b32 v15, v21 // (op 166) 
v_mov_b32 v20, v15 // (op 166) reg expression

buffer_load_dwordx4 v[34:37], v20, s[60:63], 0 offen // (op 166) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v38, v30, 0, 8 // (op 18850) 



v_bfe_u32 v39, v30, 8, 8 // (op 18850) 
v_bfe_u32 v40, v30, 16, 8 // (op 18850) 
v_bfe_u32 v41, v30, 24, 8 // (op 18850) 
v_permlane16_swap_b32 v38, v39 // (op 18850) 
ds_read_b128 v[42:45], v4 offset:1024 // (op 397) Load local data 
v_permlane16_swap_b32 v40, v41 // (op 18850) 



s_nop 1
v_permlane32_swap_b32 v38, v40 // (op 18850) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v39, v41 // (op 18850) 
ds_read_b128 v[46:49], v4 offset:2048 // (op 623) Load local data 




ds_read_b128 v[30:33], v4 offset:3072 // (op 849) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v34, 0, 8 // (op 18896) 



v_bfe_u32 v51, v34, 8, 8 // (op 18896) 
v_bfe_u32 v52, v34, 16, 8 // (op 18896) 
v_bfe_u32 v53, v34, 24, 8 // (op 18896) 
v_permlane16_swap_b32 v50, v51 // (op 18896) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 18896) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 18896) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 18896) 
ds_read_b128 v[54:57], v4 offset:4096 // (op 1075) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[26:29], a[0:3], v38, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 167) 
s_lshl_b32 s11, s32, 6 // (op 1080) 
v_add_u32 v20, v15, s11 // (op 1080) 


buffer_load_dwordx4 v[34:37], v20, s[60:63], 0 offen // (op 1080) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[42:45], a[4:7], v38, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 400) 




v_add_u32 v21, 4, v21 // (op 21070) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:25], v[46:49], a[8:11], v38, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 626) 
ds_read_b128 v[58:61], v4 offset:5120 // (op 1301) Load local data 



ds_read_b128 v[62:65], v4 offset:6144 // (op 1527) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[22:25], v[30:33], a[12:15], v38, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 852) 



ds_read_b128 v[66:69], v4 offset:7168 // (op 1753) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v70, v34, 0, 8 // (op 18910) 



v_bfe_u32 v71, v34, 8, 8 // (op 18910) 
v_bfe_u32 v72, v34, 16, 8 // (op 18910) 
v_bfe_u32 v73, v34, 24, 8 // (op 18910) 
v_permlane16_swap_b32 v70, v71 // (op 18910) 
s_nop 0
v_permlane16_swap_b32 v72, v73 // (op 18910) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v70, v72 // (op 18910) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v71, v73 // (op 18910) 
ds_read_b128 v[74:77], v3 offset:17408 // (op 2249) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[54:57], a[16:19], v38, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1078) 
ds_read_b128 v[34:37], v3 offset:18432 // (op 4251) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[58:61], a[20:23], v38, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1304) 
ds_read_b128 v[78:81], v3 offset:19456 // (op 6253) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[62:65], a[24:27], v38, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1530) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:25], v[66:69], a[28:31], v38, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1756) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[26:29], a[32:35], v39, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2253) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:77], v[46:49], a[40:43], v39, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2307) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[74:77], v[30:33], a[44:47], v39, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2334) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[42:45], a[36:39], v39, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2280) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[74:77], v[54:57], a[48:51], v39, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2361) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:77], v[62:65], a[56:59], v39, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[66:69], a[60:63], v39, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2442) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:77], v[58:61], a[52:55], v39, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2388) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:37], v[26:29], a[64:67], v40, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4255) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[34:37], v[46:49], a[72:75], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4309) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:37], v[30:33], a[76:79], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4336) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[42:45], a[68:71], v40, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4282) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[54:57], a[80:83], v40, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4363) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[34:37], v[62:65], a[88:91], v40, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4417) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[66:69], a[92:95], v40, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4444) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:37], v[58:61], a[84:87], v40, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4390) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[26:29], a[96:99], v41, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6257) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[46:49], a[104:107], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6311) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[30:33], a[108:111], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6338) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[42:45], a[100:103], v41, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6284) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[78:81], v[54:57], a[112:115], v41, v70 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6365) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[62:65], a[120:123], v41, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6419) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[78:81], v[66:69], a[124:127], v41, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6446) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[78:81], v[58:61], a[116:119], v41, v71 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6392) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 6409) 
s_mov_b32 s11, s6 // (op 6409) convert

s_cmp_lt_u32 s10, s11 // (op 6409) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopTop_KLoopTail_6409 // (op 6409) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopTop_KLoopTail_6409 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ForLoopBottom_KLoopTail_6409:





















s_barrier  // (op 21602) 









v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ConditionalFalse_0_beta_114
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v11, v4 // (op 42) 
v_mul_f32 v15, v11, v5 // (op 42) 
v_mul_f32 v16, v11, v6 // (op 42) 
v_mul_f32 v17, v11, v7 // (op 42) 



v_mov_b32 v3, v13 // (op 48) reg expression

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 48) Store value


v_accvgpr_read v4, a4 // (op 9214) 
v_accvgpr_read v5, a5 // (op 9214) 
v_accvgpr_read v6, a6 // (op 9214) 
v_accvgpr_read v7, a7 // (op 9214) 
v_mul_f32 v14, v11, v4 // (op 9214) 
v_mul_f32 v15, v11, v5 // (op 9214) 
v_mul_f32 v16, v11, v6 // (op 9214) 
v_mul_f32 v17, v11, v7 // (op 9214) 



s_lshl_b32 s0, s4, 6 // (op 9216) 
v_add_u32 v3, v13, s0 // (op 9216) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9216) Store value


v_accvgpr_read v4, a8 // (op 9226) 
v_accvgpr_read v5, a9 // (op 9226) 
v_accvgpr_read v6, a10 // (op 9226) 
v_accvgpr_read v7, a11 // (op 9226) 
v_mul_f32 v14, v11, v4 // (op 9226) 
v_mul_f32 v15, v11, v5 // (op 9226) 
v_mul_f32 v16, v11, v6 // (op 9226) 
v_mul_f32 v17, v11, v7 // (op 9226) 



s_lshl_b32 s0, s4, 7 // (op 9228) 
v_add_u32 v3, v13, s0 // (op 9228) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9228) Store value


v_accvgpr_read v4, a12 // (op 9238) 
v_accvgpr_read v5, a13 // (op 9238) 
v_accvgpr_read v6, a14 // (op 9238) 
v_accvgpr_read v7, a15 // (op 9238) 
v_mul_f32 v14, v11, v4 // (op 9238) 
v_mul_f32 v15, v11, v5 // (op 9238) 
v_mul_f32 v16, v11, v6 // (op 9238) 
v_mul_f32 v17, v11, v7 // (op 9238) 



s_lshl_b32 s0, s4, 6 // (op 9240) 
s_mul_i32 s1, 3, s0 // (op 9240) 

v_add_u32 v3, v13, s1 // (op 9240) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9240) Store value


v_accvgpr_read v4, a16 // (op 9250) 
v_accvgpr_read v5, a17 // (op 9250) 
v_accvgpr_read v6, a18 // (op 9250) 
v_accvgpr_read v7, a19 // (op 9250) 
v_mul_f32 v14, v11, v4 // (op 9250) 
v_mul_f32 v15, v11, v5 // (op 9250) 
v_mul_f32 v16, v11, v6 // (op 9250) 
v_mul_f32 v17, v11, v7 // (op 9250) 



s_lshl_b32 s0, s4, 8 // (op 9252) 
v_add_u32 v3, v13, s0 // (op 9252) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9252) Store value


v_accvgpr_read v4, a20 // (op 9262) 
v_accvgpr_read v5, a21 // (op 9262) 
v_accvgpr_read v6, a22 // (op 9262) 
v_accvgpr_read v7, a23 // (op 9262) 
v_mul_f32 v14, v11, v4 // (op 9262) 
v_mul_f32 v15, v11, v5 // (op 9262) 
v_mul_f32 v16, v11, v6 // (op 9262) 
v_mul_f32 v17, v11, v7 // (op 9262) 



s_lshl_b32 s0, s4, 6 // (op 9264) 
s_mul_i32 s1, 5, s0 // (op 9264) 

v_add_u32 v3, v13, s1 // (op 9264) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9264) Store value


v_accvgpr_read v4, a24 // (op 9274) 
v_accvgpr_read v5, a25 // (op 9274) 
v_accvgpr_read v6, a26 // (op 9274) 
v_accvgpr_read v7, a27 // (op 9274) 
v_mul_f32 v14, v11, v4 // (op 9274) 
v_mul_f32 v15, v11, v5 // (op 9274) 
v_mul_f32 v16, v11, v6 // (op 9274) 
v_mul_f32 v17, v11, v7 // (op 9274) 



s_lshl_b32 s0, s4, 6 // (op 9276) 
s_mul_i32 s1, 6, s0 // (op 9276) 

v_add_u32 v3, v13, s1 // (op 9276) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9276) Store value


v_accvgpr_read v4, a28 // (op 9286) 
v_accvgpr_read v5, a29 // (op 9286) 
v_accvgpr_read v6, a30 // (op 9286) 
v_accvgpr_read v7, a31 // (op 9286) 
v_mul_f32 v14, v11, v4 // (op 9286) 
v_mul_f32 v15, v11, v5 // (op 9286) 
v_mul_f32 v16, v11, v6 // (op 9286) 
v_mul_f32 v17, v11, v7 // (op 9286) 



s_lshl_b32 s0, s4, 6 // (op 9288) 
s_mul_i32 s1, 7, s0 // (op 9288) 

v_add_u32 v3, v13, s1 // (op 9288) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9288) Store value


v_accvgpr_read v4, a32 // (op 9340) 
v_accvgpr_read v5, a33 // (op 9340) 
v_accvgpr_read v6, a34 // (op 9340) 
v_accvgpr_read v7, a35 // (op 9340) 
v_mul_f32 v14, v11, v4 // (op 9340) 
v_mul_f32 v15, v11, v5 // (op 9340) 
v_mul_f32 v16, v11, v6 // (op 9340) 
v_mul_f32 v17, v11, v7 // (op 9340) 



v_add_u32 v3, 64, v13 // (op 9343) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9343) Store value


v_accvgpr_read v4, a36 // (op 9361) 
v_accvgpr_read v5, a37 // (op 9361) 
v_accvgpr_read v6, a38 // (op 9361) 
v_accvgpr_read v7, a39 // (op 9361) 
v_mul_f32 v14, v11, v4 // (op 9361) 
v_mul_f32 v15, v11, v5 // (op 9361) 
v_mul_f32 v16, v11, v6 // (op 9361) 
v_mul_f32 v17, v11, v7 // (op 9361) 



s_lshl_b32 s0, s4, 6 // (op 9345) 
s_add_u32 s0, s0, 64 // (op 9345) 
v_add_u32 v3, v13, s0 // (op 9345) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9345) Store value


v_accvgpr_read v4, a40 // (op 9366) 
v_accvgpr_read v5, a41 // (op 9366) 
v_accvgpr_read v6, a42 // (op 9366) 
v_accvgpr_read v7, a43 // (op 9366) 
v_mul_f32 v14, v11, v4 // (op 9366) 
v_mul_f32 v15, v11, v5 // (op 9366) 
v_mul_f32 v16, v11, v6 // (op 9366) 
v_mul_f32 v17, v11, v7 // (op 9366) 



s_lshl_b32 s0, s4, 7 // (op 9347) 
s_add_u32 s0, s0, 64 // (op 9347) 
v_add_u32 v3, v13, s0 // (op 9347) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9347) Store value


v_accvgpr_read v4, a44 // (op 9371) 
v_accvgpr_read v5, a45 // (op 9371) 
v_accvgpr_read v6, a46 // (op 9371) 
v_accvgpr_read v7, a47 // (op 9371) 
v_mul_f32 v14, v11, v4 // (op 9371) 
v_mul_f32 v15, v11, v5 // (op 9371) 
v_mul_f32 v16, v11, v6 // (op 9371) 
v_mul_f32 v17, v11, v7 // (op 9371) 



s_lshl_b32 s0, s4, 6 // (op 9349) 
s_mul_i32 s1, 3, s0 // (op 9349) 
s_add_u32 s1, s1, 64 // (op 9349) 

v_add_u32 v3, v13, s1 // (op 9349) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9349) Store value


v_accvgpr_read v4, a48 // (op 9376) 
v_accvgpr_read v5, a49 // (op 9376) 
v_accvgpr_read v6, a50 // (op 9376) 
v_accvgpr_read v7, a51 // (op 9376) 
v_mul_f32 v14, v11, v4 // (op 9376) 
v_mul_f32 v15, v11, v5 // (op 9376) 
v_mul_f32 v16, v11, v6 // (op 9376) 
v_mul_f32 v17, v11, v7 // (op 9376) 



s_lshl_b32 s0, s4, 8 // (op 9351) 
s_add_u32 s0, s0, 64 // (op 9351) 
v_add_u32 v3, v13, s0 // (op 9351) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9351) Store value


v_accvgpr_read v4, a52 // (op 9381) 
v_accvgpr_read v5, a53 // (op 9381) 
v_accvgpr_read v6, a54 // (op 9381) 
v_accvgpr_read v7, a55 // (op 9381) 
v_mul_f32 v14, v11, v4 // (op 9381) 
v_mul_f32 v15, v11, v5 // (op 9381) 
v_mul_f32 v16, v11, v6 // (op 9381) 
v_mul_f32 v17, v11, v7 // (op 9381) 



s_lshl_b32 s0, s4, 6 // (op 9353) 
s_mul_i32 s1, 5, s0 // (op 9353) 
s_add_u32 s1, s1, 64 // (op 9353) 

v_add_u32 v3, v13, s1 // (op 9353) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9353) Store value


v_accvgpr_read v4, a56 // (op 9386) 
v_accvgpr_read v5, a57 // (op 9386) 
v_accvgpr_read v6, a58 // (op 9386) 
v_accvgpr_read v7, a59 // (op 9386) 
v_mul_f32 v14, v11, v4 // (op 9386) 
v_mul_f32 v15, v11, v5 // (op 9386) 
v_mul_f32 v16, v11, v6 // (op 9386) 
v_mul_f32 v17, v11, v7 // (op 9386) 



s_lshl_b32 s0, s4, 6 // (op 9355) 
s_mul_i32 s1, 6, s0 // (op 9355) 
s_add_u32 s1, s1, 64 // (op 9355) 

v_add_u32 v3, v13, s1 // (op 9355) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9355) Store value


v_accvgpr_read v4, a60 // (op 9391) 
v_accvgpr_read v5, a61 // (op 9391) 
v_accvgpr_read v6, a62 // (op 9391) 
v_accvgpr_read v7, a63 // (op 9391) 
v_mul_f32 v14, v11, v4 // (op 9391) 
v_mul_f32 v15, v11, v5 // (op 9391) 
v_mul_f32 v16, v11, v6 // (op 9391) 
v_mul_f32 v17, v11, v7 // (op 9391) 



s_lshl_b32 s0, s4, 6 // (op 9357) 
s_mul_i32 s1, 7, s0 // (op 9357) 
s_add_u32 s1, s1, 64 // (op 9357) 

v_add_u32 v3, v13, s1 // (op 9357) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9357) Store value


v_accvgpr_read v4, a64 // (op 9472) 
v_accvgpr_read v5, a65 // (op 9472) 
v_accvgpr_read v6, a66 // (op 9472) 
v_accvgpr_read v7, a67 // (op 9472) 
v_mul_f32 v14, v11, v4 // (op 9472) 
v_mul_f32 v15, v11, v5 // (op 9472) 
v_mul_f32 v16, v11, v6 // (op 9472) 
v_mul_f32 v17, v11, v7 // (op 9472) 



v_add_u32 v3, 128, v13 // (op 9475) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9475) Store value


v_accvgpr_read v4, a68 // (op 9493) 
v_accvgpr_read v5, a69 // (op 9493) 
v_accvgpr_read v6, a70 // (op 9493) 
v_accvgpr_read v7, a71 // (op 9493) 
v_mul_f32 v14, v11, v4 // (op 9493) 
v_mul_f32 v15, v11, v5 // (op 9493) 
v_mul_f32 v16, v11, v6 // (op 9493) 
v_mul_f32 v17, v11, v7 // (op 9493) 



s_lshl_b32 s0, s4, 6 // (op 9477) 
s_add_u32 s0, s0, 128 // (op 9477) 
v_add_u32 v3, v13, s0 // (op 9477) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9477) Store value


v_accvgpr_read v4, a72 // (op 9498) 
v_accvgpr_read v5, a73 // (op 9498) 
v_accvgpr_read v6, a74 // (op 9498) 
v_accvgpr_read v7, a75 // (op 9498) 
v_mul_f32 v14, v11, v4 // (op 9498) 
v_mul_f32 v15, v11, v5 // (op 9498) 
v_mul_f32 v16, v11, v6 // (op 9498) 
v_mul_f32 v17, v11, v7 // (op 9498) 



s_lshl_b32 s0, s4, 7 // (op 9479) 
s_add_u32 s0, s0, 128 // (op 9479) 
v_add_u32 v3, v13, s0 // (op 9479) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9479) Store value


v_accvgpr_read v4, a76 // (op 9503) 
v_accvgpr_read v5, a77 // (op 9503) 
v_accvgpr_read v6, a78 // (op 9503) 
v_accvgpr_read v7, a79 // (op 9503) 
v_mul_f32 v14, v11, v4 // (op 9503) 
v_mul_f32 v15, v11, v5 // (op 9503) 
v_mul_f32 v16, v11, v6 // (op 9503) 
v_mul_f32 v17, v11, v7 // (op 9503) 



s_lshl_b32 s0, s4, 6 // (op 9481) 
s_mul_i32 s1, 3, s0 // (op 9481) 
s_add_u32 s1, s1, 128 // (op 9481) 

v_add_u32 v3, v13, s1 // (op 9481) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9481) Store value


v_accvgpr_read v4, a80 // (op 9508) 
v_accvgpr_read v5, a81 // (op 9508) 
v_accvgpr_read v6, a82 // (op 9508) 
v_accvgpr_read v7, a83 // (op 9508) 
v_mul_f32 v14, v11, v4 // (op 9508) 
v_mul_f32 v15, v11, v5 // (op 9508) 
v_mul_f32 v16, v11, v6 // (op 9508) 
v_mul_f32 v17, v11, v7 // (op 9508) 



s_lshl_b32 s0, s4, 8 // (op 9483) 
s_add_u32 s0, s0, 128 // (op 9483) 
v_add_u32 v3, v13, s0 // (op 9483) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9483) Store value


v_accvgpr_read v4, a84 // (op 9513) 
v_accvgpr_read v5, a85 // (op 9513) 
v_accvgpr_read v6, a86 // (op 9513) 
v_accvgpr_read v7, a87 // (op 9513) 
v_mul_f32 v14, v11, v4 // (op 9513) 
v_mul_f32 v15, v11, v5 // (op 9513) 
v_mul_f32 v16, v11, v6 // (op 9513) 
v_mul_f32 v17, v11, v7 // (op 9513) 



s_lshl_b32 s0, s4, 6 // (op 9485) 
s_mul_i32 s1, 5, s0 // (op 9485) 
s_add_u32 s1, s1, 128 // (op 9485) 

v_add_u32 v3, v13, s1 // (op 9485) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9485) Store value


v_accvgpr_read v4, a88 // (op 9518) 
v_accvgpr_read v5, a89 // (op 9518) 
v_accvgpr_read v6, a90 // (op 9518) 
v_accvgpr_read v7, a91 // (op 9518) 
v_mul_f32 v14, v11, v4 // (op 9518) 
v_mul_f32 v15, v11, v5 // (op 9518) 
v_mul_f32 v16, v11, v6 // (op 9518) 
v_mul_f32 v17, v11, v7 // (op 9518) 



s_lshl_b32 s0, s4, 6 // (op 9487) 
s_mul_i32 s1, 6, s0 // (op 9487) 
s_add_u32 s1, s1, 128 // (op 9487) 

v_add_u32 v3, v13, s1 // (op 9487) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9487) Store value


v_accvgpr_read v4, a92 // (op 9523) 
v_accvgpr_read v5, a93 // (op 9523) 
v_accvgpr_read v6, a94 // (op 9523) 
v_accvgpr_read v7, a95 // (op 9523) 
v_mul_f32 v14, v11, v4 // (op 9523) 
v_mul_f32 v15, v11, v5 // (op 9523) 
v_mul_f32 v16, v11, v6 // (op 9523) 
v_mul_f32 v17, v11, v7 // (op 9523) 



s_lshl_b32 s0, s4, 6 // (op 9489) 
s_mul_i32 s1, 7, s0 // (op 9489) 
s_add_u32 s1, s1, 128 // (op 9489) 

v_add_u32 v3, v13, s1 // (op 9489) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9489) Store value


v_accvgpr_read v4, a96 // (op 9604) 
v_accvgpr_read v5, a97 // (op 9604) 
v_accvgpr_read v6, a98 // (op 9604) 
v_accvgpr_read v7, a99 // (op 9604) 
v_mul_f32 v14, v11, v4 // (op 9604) 
v_mul_f32 v15, v11, v5 // (op 9604) 
v_mul_f32 v16, v11, v6 // (op 9604) 
v_mul_f32 v17, v11, v7 // (op 9604) 



v_add_u32 v3, 192, v13 // (op 9607) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9607) Store value


v_accvgpr_read v4, a100 // (op 9625) 
v_accvgpr_read v5, a101 // (op 9625) 
v_accvgpr_read v6, a102 // (op 9625) 
v_accvgpr_read v7, a103 // (op 9625) 
v_mul_f32 v14, v11, v4 // (op 9625) 
v_mul_f32 v15, v11, v5 // (op 9625) 
v_mul_f32 v16, v11, v6 // (op 9625) 
v_mul_f32 v17, v11, v7 // (op 9625) 



s_lshl_b32 s0, s4, 6 // (op 9609) 
s_add_u32 s0, s0, 192 // (op 9609) 
v_add_u32 v3, v13, s0 // (op 9609) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9609) Store value


v_accvgpr_read v4, a104 // (op 9630) 
v_accvgpr_read v5, a105 // (op 9630) 
v_accvgpr_read v6, a106 // (op 9630) 
v_accvgpr_read v7, a107 // (op 9630) 
v_mul_f32 v14, v11, v4 // (op 9630) 
v_mul_f32 v15, v11, v5 // (op 9630) 
v_mul_f32 v16, v11, v6 // (op 9630) 
v_mul_f32 v17, v11, v7 // (op 9630) 



s_lshl_b32 s0, s4, 7 // (op 9611) 
s_add_u32 s0, s0, 192 // (op 9611) 
v_add_u32 v3, v13, s0 // (op 9611) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9611) Store value


v_accvgpr_read v4, a108 // (op 9635) 
v_accvgpr_read v5, a109 // (op 9635) 
v_accvgpr_read v6, a110 // (op 9635) 
v_accvgpr_read v7, a111 // (op 9635) 
v_mul_f32 v14, v11, v4 // (op 9635) 
v_mul_f32 v15, v11, v5 // (op 9635) 
v_mul_f32 v16, v11, v6 // (op 9635) 
v_mul_f32 v17, v11, v7 // (op 9635) 



s_lshl_b32 s0, s4, 6 // (op 9613) 
s_mul_i32 s1, 3, s0 // (op 9613) 
s_add_u32 s1, s1, 192 // (op 9613) 

v_add_u32 v3, v13, s1 // (op 9613) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9613) Store value


v_accvgpr_read v4, a112 // (op 9640) 
v_accvgpr_read v5, a113 // (op 9640) 
v_accvgpr_read v6, a114 // (op 9640) 
v_accvgpr_read v7, a115 // (op 9640) 
v_mul_f32 v14, v11, v4 // (op 9640) 
v_mul_f32 v15, v11, v5 // (op 9640) 
v_mul_f32 v16, v11, v6 // (op 9640) 
v_mul_f32 v17, v11, v7 // (op 9640) 



s_lshl_b32 s0, s4, 8 // (op 9615) 
s_add_u32 s0, s0, 192 // (op 9615) 
v_add_u32 v3, v13, s0 // (op 9615) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9615) Store value


v_accvgpr_read v4, a116 // (op 9645) 
v_accvgpr_read v5, a117 // (op 9645) 
v_accvgpr_read v6, a118 // (op 9645) 
v_accvgpr_read v7, a119 // (op 9645) 
v_mul_f32 v14, v11, v4 // (op 9645) 
v_mul_f32 v15, v11, v5 // (op 9645) 
v_mul_f32 v16, v11, v6 // (op 9645) 
v_mul_f32 v17, v11, v7 // (op 9645) 



s_lshl_b32 s0, s4, 6 // (op 9617) 
s_mul_i32 s1, 5, s0 // (op 9617) 
s_add_u32 s1, s1, 192 // (op 9617) 

v_add_u32 v3, v13, s1 // (op 9617) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9617) Store value


v_accvgpr_read v4, a120 // (op 9650) 
v_accvgpr_read v5, a121 // (op 9650) 
v_accvgpr_read v6, a122 // (op 9650) 
v_accvgpr_read v7, a123 // (op 9650) 
v_mul_f32 v14, v11, v4 // (op 9650) 
v_mul_f32 v15, v11, v5 // (op 9650) 
v_mul_f32 v16, v11, v6 // (op 9650) 
v_mul_f32 v17, v11, v7 // (op 9650) 



s_lshl_b32 s0, s4, 6 // (op 9619) 
s_mul_i32 s1, 6, s0 // (op 9619) 
s_add_u32 s1, s1, 192 // (op 9619) 

v_add_u32 v3, v13, s1 // (op 9619) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9619) Store value


v_accvgpr_read v4, a124 // (op 9655) 
v_accvgpr_read v5, a125 // (op 9655) 
v_accvgpr_read v6, a126 // (op 9655) 
v_accvgpr_read v7, a127 // (op 9655) 
v_mul_f32 v14, v11, v4 // (op 9655) 
v_mul_f32 v15, v11, v5 // (op 9655) 
v_mul_f32 v16, v11, v6 // (op 9655) 
v_mul_f32 v17, v11, v7 // (op 9655) 



s_lshl_b32 s0, s4, 6 // (op 9621) 
s_mul_i32 s1, 7, s0 // (op 9621) 
s_add_u32 s1, s1, 192 // (op 9621) 

v_add_u32 v3, v13, s1 // (op 9621) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 9621) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ConditionalBottom_0_beta_114
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ConditionalFalse_0_beta_114:


v_add_u32 v3, 192, v12 // (op 10331) 

buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10331) Load value

v_accvgpr_read v14, a96 // (op 10311) 
v_accvgpr_read v15, a97 // (op 10311) 
v_accvgpr_read v16, a98 // (op 10311) 
v_accvgpr_read v17, a99 // (op 10311) 
v_mul_f32 v18, v11, v14 // (op 10311) 
v_mul_f32 v19, v11, v15 // (op 10311) 
v_mul_f32 v20, v11, v16 // (op 10311) 
v_mul_f32 v21, v11, v17 // (op 10311) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10330) 
v_mul_f32 v15, v0, v5 // (op 10330) 
v_mul_f32 v16, v0, v6 // (op 10330) 
v_mul_f32 v17, v0, v7 // (op 10330) 

v_add_f32 v4, v14, v18 // (op 10312) 
v_add_f32 v5, v15, v19 // (op 10312) 
v_add_f32 v6, v16, v20 // (op 10312) 
v_add_f32 v7, v17, v21 // (op 10312) 




v_add_u32 v3, 192, v13 // (op 10314) 

buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10314) Store value




s_lshl_b32 s0, s2, 6 // (op 10335) 
s_add_u32 s0, s0, 192 // (op 10335) 
v_add_u32 v3, v12, s0 // (op 10335) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10335) Load value

v_accvgpr_read v14, a100 // (op 10361) 
v_accvgpr_read v15, a101 // (op 10361) 
v_accvgpr_read v16, a102 // (op 10361) 
v_accvgpr_read v17, a103 // (op 10361) 
v_mul_f32 v18, v11, v14 // (op 10361) 
v_mul_f32 v19, v11, v15 // (op 10361) 
v_mul_f32 v20, v11, v16 // (op 10361) 
v_mul_f32 v21, v11, v17 // (op 10361) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10333) 
v_mul_f32 v15, v0, v5 // (op 10333) 
v_mul_f32 v16, v0, v6 // (op 10333) 
v_mul_f32 v17, v0, v7 // (op 10333) 

v_add_f32 v4, v14, v18 // (op 10334) 
v_add_f32 v5, v15, v19 // (op 10334) 
v_add_f32 v6, v16, v20 // (op 10334) 
v_add_f32 v7, v17, v21 // (op 10334) 




s_lshl_b32 s0, s4, 6 // (op 10316) 
s_add_u32 s0, s0, 192 // (op 10316) 
v_add_u32 v3, v13, s0 // (op 10316) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10316) Store value




s_lshl_b32 s0, s2, 7 // (op 10339) 
s_add_u32 s0, s0, 192 // (op 10339) 
v_add_u32 v3, v12, s0 // (op 10339) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10339) Load value

v_accvgpr_read v14, a104 // (op 10363) 
v_accvgpr_read v15, a105 // (op 10363) 
v_accvgpr_read v16, a106 // (op 10363) 
v_accvgpr_read v17, a107 // (op 10363) 
v_mul_f32 v18, v11, v14 // (op 10363) 
v_mul_f32 v19, v11, v15 // (op 10363) 
v_mul_f32 v20, v11, v16 // (op 10363) 
v_mul_f32 v21, v11, v17 // (op 10363) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10337) 
v_mul_f32 v15, v0, v5 // (op 10337) 
v_mul_f32 v16, v0, v6 // (op 10337) 
v_mul_f32 v17, v0, v7 // (op 10337) 

v_add_f32 v4, v14, v18 // (op 10338) 
v_add_f32 v5, v15, v19 // (op 10338) 
v_add_f32 v6, v16, v20 // (op 10338) 
v_add_f32 v7, v17, v21 // (op 10338) 




s_lshl_b32 s0, s4, 7 // (op 10318) 
s_add_u32 s0, s0, 192 // (op 10318) 
v_add_u32 v3, v13, s0 // (op 10318) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10318) Store value




s_lshl_b32 s0, s2, 6 // (op 10343) 
s_mul_i32 s1, 3, s0 // (op 10343) 
s_add_u32 s1, s1, 192 // (op 10343) 

v_add_u32 v3, v12, s1 // (op 10343) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10343) Load value

v_accvgpr_read v14, a108 // (op 10365) 
v_accvgpr_read v15, a109 // (op 10365) 
v_accvgpr_read v16, a110 // (op 10365) 
v_accvgpr_read v17, a111 // (op 10365) 
v_mul_f32 v18, v11, v14 // (op 10365) 
v_mul_f32 v19, v11, v15 // (op 10365) 
v_mul_f32 v20, v11, v16 // (op 10365) 
v_mul_f32 v21, v11, v17 // (op 10365) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10341) 
v_mul_f32 v15, v0, v5 // (op 10341) 
v_mul_f32 v16, v0, v6 // (op 10341) 
v_mul_f32 v17, v0, v7 // (op 10341) 

v_add_f32 v4, v14, v18 // (op 10342) 
v_add_f32 v5, v15, v19 // (op 10342) 
v_add_f32 v6, v16, v20 // (op 10342) 
v_add_f32 v7, v17, v21 // (op 10342) 




s_lshl_b32 s0, s4, 6 // (op 10320) 
s_mul_i32 s1, 3, s0 // (op 10320) 
s_add_u32 s1, s1, 192 // (op 10320) 

v_add_u32 v3, v13, s1 // (op 10320) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10320) Store value




s_lshl_b32 s0, s2, 8 // (op 10347) 
s_add_u32 s0, s0, 192 // (op 10347) 
v_add_u32 v3, v12, s0 // (op 10347) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10347) Load value

v_accvgpr_read v14, a112 // (op 10367) 
v_accvgpr_read v15, a113 // (op 10367) 
v_accvgpr_read v16, a114 // (op 10367) 
v_accvgpr_read v17, a115 // (op 10367) 
v_mul_f32 v18, v11, v14 // (op 10367) 
v_mul_f32 v19, v11, v15 // (op 10367) 
v_mul_f32 v20, v11, v16 // (op 10367) 
v_mul_f32 v21, v11, v17 // (op 10367) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10345) 
v_mul_f32 v15, v0, v5 // (op 10345) 
v_mul_f32 v16, v0, v6 // (op 10345) 
v_mul_f32 v17, v0, v7 // (op 10345) 

v_add_f32 v4, v14, v18 // (op 10346) 
v_add_f32 v5, v15, v19 // (op 10346) 
v_add_f32 v6, v16, v20 // (op 10346) 
v_add_f32 v7, v17, v21 // (op 10346) 




s_lshl_b32 s0, s4, 8 // (op 10322) 
s_add_u32 s0, s0, 192 // (op 10322) 
v_add_u32 v3, v13, s0 // (op 10322) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10322) Store value




s_lshl_b32 s0, s2, 6 // (op 10351) 
s_mul_i32 s1, 5, s0 // (op 10351) 
s_add_u32 s1, s1, 192 // (op 10351) 

v_add_u32 v3, v12, s1 // (op 10351) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10351) Load value

v_accvgpr_read v14, a116 // (op 10369) 
v_accvgpr_read v15, a117 // (op 10369) 
v_accvgpr_read v16, a118 // (op 10369) 
v_accvgpr_read v17, a119 // (op 10369) 
v_mul_f32 v18, v11, v14 // (op 10369) 
v_mul_f32 v19, v11, v15 // (op 10369) 
v_mul_f32 v20, v11, v16 // (op 10369) 
v_mul_f32 v21, v11, v17 // (op 10369) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10349) 
v_mul_f32 v15, v0, v5 // (op 10349) 
v_mul_f32 v16, v0, v6 // (op 10349) 
v_mul_f32 v17, v0, v7 // (op 10349) 

v_add_f32 v4, v14, v18 // (op 10350) 
v_add_f32 v5, v15, v19 // (op 10350) 
v_add_f32 v6, v16, v20 // (op 10350) 
v_add_f32 v7, v17, v21 // (op 10350) 




s_lshl_b32 s0, s4, 6 // (op 10324) 
s_mul_i32 s1, 5, s0 // (op 10324) 
s_add_u32 s1, s1, 192 // (op 10324) 

v_add_u32 v3, v13, s1 // (op 10324) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10324) Store value




s_lshl_b32 s0, s2, 6 // (op 10355) 
s_mul_i32 s1, 6, s0 // (op 10355) 
s_add_u32 s1, s1, 192 // (op 10355) 

v_add_u32 v3, v12, s1 // (op 10355) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10355) Load value

v_accvgpr_read v14, a120 // (op 10371) 
v_accvgpr_read v15, a121 // (op 10371) 
v_accvgpr_read v16, a122 // (op 10371) 
v_accvgpr_read v17, a123 // (op 10371) 
v_mul_f32 v18, v11, v14 // (op 10371) 
v_mul_f32 v19, v11, v15 // (op 10371) 
v_mul_f32 v20, v11, v16 // (op 10371) 
v_mul_f32 v21, v11, v17 // (op 10371) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10353) 
v_mul_f32 v15, v0, v5 // (op 10353) 
v_mul_f32 v16, v0, v6 // (op 10353) 
v_mul_f32 v17, v0, v7 // (op 10353) 

v_add_f32 v4, v14, v18 // (op 10354) 
v_add_f32 v5, v15, v19 // (op 10354) 
v_add_f32 v6, v16, v20 // (op 10354) 
v_add_f32 v7, v17, v21 // (op 10354) 




s_lshl_b32 s0, s4, 6 // (op 10326) 
s_mul_i32 s1, 6, s0 // (op 10326) 
s_add_u32 s1, s1, 192 // (op 10326) 

v_add_u32 v3, v13, s1 // (op 10326) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10326) Store value




s_lshl_b32 s0, s2, 6 // (op 10359) 
s_mul_i32 s1, 7, s0 // (op 10359) 
s_add_u32 s1, s1, 192 // (op 10359) 

v_add_u32 v3, v12, s1 // (op 10359) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10359) Load value

v_accvgpr_read v14, a124 // (op 10373) 
v_accvgpr_read v15, a125 // (op 10373) 
v_accvgpr_read v16, a126 // (op 10373) 
v_accvgpr_read v17, a127 // (op 10373) 
v_mul_f32 v18, v11, v14 // (op 10373) 
v_mul_f32 v19, v11, v15 // (op 10373) 
v_mul_f32 v20, v11, v16 // (op 10373) 
v_mul_f32 v21, v11, v17 // (op 10373) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10357) 
v_mul_f32 v15, v0, v5 // (op 10357) 
v_mul_f32 v16, v0, v6 // (op 10357) 
v_mul_f32 v17, v0, v7 // (op 10357) 

v_add_f32 v4, v14, v18 // (op 10358) 
v_add_f32 v5, v15, v19 // (op 10358) 
v_add_f32 v6, v16, v20 // (op 10358) 
v_add_f32 v7, v17, v21 // (op 10358) 




s_lshl_b32 s0, s4, 6 // (op 10328) 
s_mul_i32 s1, 7, s0 // (op 10328) 
s_add_u32 s1, s1, 192 // (op 10328) 

v_add_u32 v3, v13, s1 // (op 10328) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10328) Store value




v_add_u32 v3, 128, v12 // (op 10176) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10176) Load value

v_accvgpr_read v14, a64 // (op 10156) 
v_accvgpr_read v15, a65 // (op 10156) 
v_accvgpr_read v16, a66 // (op 10156) 
v_accvgpr_read v17, a67 // (op 10156) 
v_mul_f32 v18, v11, v14 // (op 10156) 
v_mul_f32 v19, v11, v15 // (op 10156) 
v_mul_f32 v20, v11, v16 // (op 10156) 
v_mul_f32 v21, v11, v17 // (op 10156) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10175) 
v_mul_f32 v15, v0, v5 // (op 10175) 
v_mul_f32 v16, v0, v6 // (op 10175) 
v_mul_f32 v17, v0, v7 // (op 10175) 

v_add_f32 v4, v14, v18 // (op 10157) 
v_add_f32 v5, v15, v19 // (op 10157) 
v_add_f32 v6, v16, v20 // (op 10157) 
v_add_f32 v7, v17, v21 // (op 10157) 




v_add_u32 v3, 128, v13 // (op 10159) 

buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10159) Store value




s_lshl_b32 s0, s2, 6 // (op 10180) 
s_add_u32 s0, s0, 128 // (op 10180) 
v_add_u32 v3, v12, s0 // (op 10180) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10180) Load value

v_accvgpr_read v14, a68 // (op 10206) 
v_accvgpr_read v15, a69 // (op 10206) 
v_accvgpr_read v16, a70 // (op 10206) 
v_accvgpr_read v17, a71 // (op 10206) 
v_mul_f32 v18, v11, v14 // (op 10206) 
v_mul_f32 v19, v11, v15 // (op 10206) 
v_mul_f32 v20, v11, v16 // (op 10206) 
v_mul_f32 v21, v11, v17 // (op 10206) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10178) 
v_mul_f32 v15, v0, v5 // (op 10178) 
v_mul_f32 v16, v0, v6 // (op 10178) 
v_mul_f32 v17, v0, v7 // (op 10178) 

v_add_f32 v4, v14, v18 // (op 10179) 
v_add_f32 v5, v15, v19 // (op 10179) 
v_add_f32 v6, v16, v20 // (op 10179) 
v_add_f32 v7, v17, v21 // (op 10179) 




s_lshl_b32 s0, s4, 6 // (op 10161) 
s_add_u32 s0, s0, 128 // (op 10161) 
v_add_u32 v3, v13, s0 // (op 10161) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10161) Store value




s_lshl_b32 s0, s2, 7 // (op 10184) 
s_add_u32 s0, s0, 128 // (op 10184) 
v_add_u32 v3, v12, s0 // (op 10184) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10184) Load value

v_accvgpr_read v14, a72 // (op 10208) 
v_accvgpr_read v15, a73 // (op 10208) 
v_accvgpr_read v16, a74 // (op 10208) 
v_accvgpr_read v17, a75 // (op 10208) 
v_mul_f32 v18, v11, v14 // (op 10208) 
v_mul_f32 v19, v11, v15 // (op 10208) 
v_mul_f32 v20, v11, v16 // (op 10208) 
v_mul_f32 v21, v11, v17 // (op 10208) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10182) 
v_mul_f32 v15, v0, v5 // (op 10182) 
v_mul_f32 v16, v0, v6 // (op 10182) 
v_mul_f32 v17, v0, v7 // (op 10182) 

v_add_f32 v4, v14, v18 // (op 10183) 
v_add_f32 v5, v15, v19 // (op 10183) 
v_add_f32 v6, v16, v20 // (op 10183) 
v_add_f32 v7, v17, v21 // (op 10183) 




s_lshl_b32 s0, s4, 7 // (op 10163) 
s_add_u32 s0, s0, 128 // (op 10163) 
v_add_u32 v3, v13, s0 // (op 10163) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10163) Store value




s_lshl_b32 s0, s2, 6 // (op 10188) 
s_mul_i32 s1, 3, s0 // (op 10188) 
s_add_u32 s1, s1, 128 // (op 10188) 

v_add_u32 v3, v12, s1 // (op 10188) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10188) Load value

v_accvgpr_read v14, a76 // (op 10210) 
v_accvgpr_read v15, a77 // (op 10210) 
v_accvgpr_read v16, a78 // (op 10210) 
v_accvgpr_read v17, a79 // (op 10210) 
v_mul_f32 v18, v11, v14 // (op 10210) 
v_mul_f32 v19, v11, v15 // (op 10210) 
v_mul_f32 v20, v11, v16 // (op 10210) 
v_mul_f32 v21, v11, v17 // (op 10210) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10186) 
v_mul_f32 v15, v0, v5 // (op 10186) 
v_mul_f32 v16, v0, v6 // (op 10186) 
v_mul_f32 v17, v0, v7 // (op 10186) 

v_add_f32 v4, v14, v18 // (op 10187) 
v_add_f32 v5, v15, v19 // (op 10187) 
v_add_f32 v6, v16, v20 // (op 10187) 
v_add_f32 v7, v17, v21 // (op 10187) 




s_lshl_b32 s0, s4, 6 // (op 10165) 
s_mul_i32 s1, 3, s0 // (op 10165) 
s_add_u32 s1, s1, 128 // (op 10165) 

v_add_u32 v3, v13, s1 // (op 10165) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10165) Store value




s_lshl_b32 s0, s2, 8 // (op 10192) 
s_add_u32 s0, s0, 128 // (op 10192) 
v_add_u32 v3, v12, s0 // (op 10192) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10192) Load value

v_accvgpr_read v14, a80 // (op 10212) 
v_accvgpr_read v15, a81 // (op 10212) 
v_accvgpr_read v16, a82 // (op 10212) 
v_accvgpr_read v17, a83 // (op 10212) 
v_mul_f32 v18, v11, v14 // (op 10212) 
v_mul_f32 v19, v11, v15 // (op 10212) 
v_mul_f32 v20, v11, v16 // (op 10212) 
v_mul_f32 v21, v11, v17 // (op 10212) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10190) 
v_mul_f32 v15, v0, v5 // (op 10190) 
v_mul_f32 v16, v0, v6 // (op 10190) 
v_mul_f32 v17, v0, v7 // (op 10190) 

v_add_f32 v4, v14, v18 // (op 10191) 
v_add_f32 v5, v15, v19 // (op 10191) 
v_add_f32 v6, v16, v20 // (op 10191) 
v_add_f32 v7, v17, v21 // (op 10191) 




s_lshl_b32 s0, s4, 8 // (op 10167) 
s_add_u32 s0, s0, 128 // (op 10167) 
v_add_u32 v3, v13, s0 // (op 10167) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10167) Store value




s_lshl_b32 s0, s2, 6 // (op 10196) 
s_mul_i32 s1, 5, s0 // (op 10196) 
s_add_u32 s1, s1, 128 // (op 10196) 

v_add_u32 v3, v12, s1 // (op 10196) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10196) Load value

v_accvgpr_read v14, a84 // (op 10214) 
v_accvgpr_read v15, a85 // (op 10214) 
v_accvgpr_read v16, a86 // (op 10214) 
v_accvgpr_read v17, a87 // (op 10214) 
v_mul_f32 v18, v11, v14 // (op 10214) 
v_mul_f32 v19, v11, v15 // (op 10214) 
v_mul_f32 v20, v11, v16 // (op 10214) 
v_mul_f32 v21, v11, v17 // (op 10214) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10194) 
v_mul_f32 v15, v0, v5 // (op 10194) 
v_mul_f32 v16, v0, v6 // (op 10194) 
v_mul_f32 v17, v0, v7 // (op 10194) 

v_add_f32 v4, v14, v18 // (op 10195) 
v_add_f32 v5, v15, v19 // (op 10195) 
v_add_f32 v6, v16, v20 // (op 10195) 
v_add_f32 v7, v17, v21 // (op 10195) 




s_lshl_b32 s0, s4, 6 // (op 10169) 
s_mul_i32 s1, 5, s0 // (op 10169) 
s_add_u32 s1, s1, 128 // (op 10169) 

v_add_u32 v3, v13, s1 // (op 10169) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10169) Store value




s_lshl_b32 s0, s2, 6 // (op 10200) 
s_mul_i32 s1, 6, s0 // (op 10200) 
s_add_u32 s1, s1, 128 // (op 10200) 

v_add_u32 v3, v12, s1 // (op 10200) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10200) Load value

v_accvgpr_read v14, a88 // (op 10216) 
v_accvgpr_read v15, a89 // (op 10216) 
v_accvgpr_read v16, a90 // (op 10216) 
v_accvgpr_read v17, a91 // (op 10216) 
v_mul_f32 v18, v11, v14 // (op 10216) 
v_mul_f32 v19, v11, v15 // (op 10216) 
v_mul_f32 v20, v11, v16 // (op 10216) 
v_mul_f32 v21, v11, v17 // (op 10216) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10198) 
v_mul_f32 v15, v0, v5 // (op 10198) 
v_mul_f32 v16, v0, v6 // (op 10198) 
v_mul_f32 v17, v0, v7 // (op 10198) 

v_add_f32 v4, v14, v18 // (op 10199) 
v_add_f32 v5, v15, v19 // (op 10199) 
v_add_f32 v6, v16, v20 // (op 10199) 
v_add_f32 v7, v17, v21 // (op 10199) 




s_lshl_b32 s0, s4, 6 // (op 10171) 
s_mul_i32 s1, 6, s0 // (op 10171) 
s_add_u32 s1, s1, 128 // (op 10171) 

v_add_u32 v3, v13, s1 // (op 10171) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10171) Store value




s_lshl_b32 s0, s2, 6 // (op 10204) 
s_mul_i32 s1, 7, s0 // (op 10204) 
s_add_u32 s1, s1, 128 // (op 10204) 

v_add_u32 v3, v12, s1 // (op 10204) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10204) Load value

v_accvgpr_read v14, a92 // (op 10218) 
v_accvgpr_read v15, a93 // (op 10218) 
v_accvgpr_read v16, a94 // (op 10218) 
v_accvgpr_read v17, a95 // (op 10218) 
v_mul_f32 v18, v11, v14 // (op 10218) 
v_mul_f32 v19, v11, v15 // (op 10218) 
v_mul_f32 v20, v11, v16 // (op 10218) 
v_mul_f32 v21, v11, v17 // (op 10218) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10202) 
v_mul_f32 v15, v0, v5 // (op 10202) 
v_mul_f32 v16, v0, v6 // (op 10202) 
v_mul_f32 v17, v0, v7 // (op 10202) 

v_add_f32 v4, v14, v18 // (op 10203) 
v_add_f32 v5, v15, v19 // (op 10203) 
v_add_f32 v6, v16, v20 // (op 10203) 
v_add_f32 v7, v17, v21 // (op 10203) 




s_lshl_b32 s0, s4, 6 // (op 10173) 
s_mul_i32 s1, 7, s0 // (op 10173) 
s_add_u32 s1, s1, 128 // (op 10173) 

v_add_u32 v3, v13, s1 // (op 10173) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10173) Store value




v_add_u32 v3, 64, v12 // (op 10021) 

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10021) Load value

v_accvgpr_read v14, a32 // (op 10001) 
v_accvgpr_read v15, a33 // (op 10001) 
v_accvgpr_read v16, a34 // (op 10001) 
v_accvgpr_read v17, a35 // (op 10001) 
v_mul_f32 v18, v11, v14 // (op 10001) 
v_mul_f32 v19, v11, v15 // (op 10001) 
v_mul_f32 v20, v11, v16 // (op 10001) 
v_mul_f32 v21, v11, v17 // (op 10001) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10020) 
v_mul_f32 v15, v0, v5 // (op 10020) 
v_mul_f32 v16, v0, v6 // (op 10020) 
v_mul_f32 v17, v0, v7 // (op 10020) 

v_add_f32 v4, v14, v18 // (op 10002) 
v_add_f32 v5, v15, v19 // (op 10002) 
v_add_f32 v6, v16, v20 // (op 10002) 
v_add_f32 v7, v17, v21 // (op 10002) 




v_add_u32 v3, 64, v13 // (op 10004) 

buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10004) Store value




s_lshl_b32 s0, s2, 6 // (op 10025) 
s_add_u32 s0, s0, 64 // (op 10025) 
v_add_u32 v3, v12, s0 // (op 10025) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10025) Load value

v_accvgpr_read v14, a36 // (op 10051) 
v_accvgpr_read v15, a37 // (op 10051) 
v_accvgpr_read v16, a38 // (op 10051) 
v_accvgpr_read v17, a39 // (op 10051) 
v_mul_f32 v18, v11, v14 // (op 10051) 
v_mul_f32 v19, v11, v15 // (op 10051) 
v_mul_f32 v20, v11, v16 // (op 10051) 
v_mul_f32 v21, v11, v17 // (op 10051) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10023) 
v_mul_f32 v15, v0, v5 // (op 10023) 
v_mul_f32 v16, v0, v6 // (op 10023) 
v_mul_f32 v17, v0, v7 // (op 10023) 

v_add_f32 v4, v14, v18 // (op 10024) 
v_add_f32 v5, v15, v19 // (op 10024) 
v_add_f32 v6, v16, v20 // (op 10024) 
v_add_f32 v7, v17, v21 // (op 10024) 




s_lshl_b32 s0, s4, 6 // (op 10006) 
s_add_u32 s0, s0, 64 // (op 10006) 
v_add_u32 v3, v13, s0 // (op 10006) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10006) Store value




s_lshl_b32 s0, s2, 7 // (op 10029) 
s_add_u32 s0, s0, 64 // (op 10029) 
v_add_u32 v3, v12, s0 // (op 10029) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10029) Load value

v_accvgpr_read v14, a40 // (op 10053) 
v_accvgpr_read v15, a41 // (op 10053) 
v_accvgpr_read v16, a42 // (op 10053) 
v_accvgpr_read v17, a43 // (op 10053) 
v_mul_f32 v18, v11, v14 // (op 10053) 
v_mul_f32 v19, v11, v15 // (op 10053) 
v_mul_f32 v20, v11, v16 // (op 10053) 
v_mul_f32 v21, v11, v17 // (op 10053) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10027) 
v_mul_f32 v15, v0, v5 // (op 10027) 
v_mul_f32 v16, v0, v6 // (op 10027) 
v_mul_f32 v17, v0, v7 // (op 10027) 

v_add_f32 v4, v14, v18 // (op 10028) 
v_add_f32 v5, v15, v19 // (op 10028) 
v_add_f32 v6, v16, v20 // (op 10028) 
v_add_f32 v7, v17, v21 // (op 10028) 




s_lshl_b32 s0, s4, 7 // (op 10008) 
s_add_u32 s0, s0, 64 // (op 10008) 
v_add_u32 v3, v13, s0 // (op 10008) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10008) Store value




s_lshl_b32 s0, s2, 6 // (op 10033) 
s_mul_i32 s1, 3, s0 // (op 10033) 
s_add_u32 s1, s1, 64 // (op 10033) 

v_add_u32 v3, v12, s1 // (op 10033) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10033) Load value

v_accvgpr_read v14, a44 // (op 10055) 
v_accvgpr_read v15, a45 // (op 10055) 
v_accvgpr_read v16, a46 // (op 10055) 
v_accvgpr_read v17, a47 // (op 10055) 
v_mul_f32 v18, v11, v14 // (op 10055) 
v_mul_f32 v19, v11, v15 // (op 10055) 
v_mul_f32 v20, v11, v16 // (op 10055) 
v_mul_f32 v21, v11, v17 // (op 10055) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10031) 
v_mul_f32 v15, v0, v5 // (op 10031) 
v_mul_f32 v16, v0, v6 // (op 10031) 
v_mul_f32 v17, v0, v7 // (op 10031) 

v_add_f32 v4, v14, v18 // (op 10032) 
v_add_f32 v5, v15, v19 // (op 10032) 
v_add_f32 v6, v16, v20 // (op 10032) 
v_add_f32 v7, v17, v21 // (op 10032) 




s_lshl_b32 s0, s4, 6 // (op 10010) 
s_mul_i32 s1, 3, s0 // (op 10010) 
s_add_u32 s1, s1, 64 // (op 10010) 

v_add_u32 v3, v13, s1 // (op 10010) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10010) Store value




s_lshl_b32 s0, s2, 8 // (op 10037) 
s_add_u32 s0, s0, 64 // (op 10037) 
v_add_u32 v3, v12, s0 // (op 10037) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10037) Load value

v_accvgpr_read v14, a48 // (op 10057) 
v_accvgpr_read v15, a49 // (op 10057) 
v_accvgpr_read v16, a50 // (op 10057) 
v_accvgpr_read v17, a51 // (op 10057) 
v_mul_f32 v18, v11, v14 // (op 10057) 
v_mul_f32 v19, v11, v15 // (op 10057) 
v_mul_f32 v20, v11, v16 // (op 10057) 
v_mul_f32 v21, v11, v17 // (op 10057) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10035) 
v_mul_f32 v15, v0, v5 // (op 10035) 
v_mul_f32 v16, v0, v6 // (op 10035) 
v_mul_f32 v17, v0, v7 // (op 10035) 

v_add_f32 v4, v14, v18 // (op 10036) 
v_add_f32 v5, v15, v19 // (op 10036) 
v_add_f32 v6, v16, v20 // (op 10036) 
v_add_f32 v7, v17, v21 // (op 10036) 




s_lshl_b32 s0, s4, 8 // (op 10012) 
s_add_u32 s0, s0, 64 // (op 10012) 
v_add_u32 v3, v13, s0 // (op 10012) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10012) Store value




s_lshl_b32 s0, s2, 6 // (op 10041) 
s_mul_i32 s1, 5, s0 // (op 10041) 
s_add_u32 s1, s1, 64 // (op 10041) 

v_add_u32 v3, v12, s1 // (op 10041) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10041) Load value

v_accvgpr_read v14, a52 // (op 10059) 
v_accvgpr_read v15, a53 // (op 10059) 
v_accvgpr_read v16, a54 // (op 10059) 
v_accvgpr_read v17, a55 // (op 10059) 
v_mul_f32 v18, v11, v14 // (op 10059) 
v_mul_f32 v19, v11, v15 // (op 10059) 
v_mul_f32 v20, v11, v16 // (op 10059) 
v_mul_f32 v21, v11, v17 // (op 10059) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10039) 
v_mul_f32 v15, v0, v5 // (op 10039) 
v_mul_f32 v16, v0, v6 // (op 10039) 
v_mul_f32 v17, v0, v7 // (op 10039) 

v_add_f32 v4, v14, v18 // (op 10040) 
v_add_f32 v5, v15, v19 // (op 10040) 
v_add_f32 v6, v16, v20 // (op 10040) 
v_add_f32 v7, v17, v21 // (op 10040) 




s_lshl_b32 s0, s4, 6 // (op 10014) 
s_mul_i32 s1, 5, s0 // (op 10014) 
s_add_u32 s1, s1, 64 // (op 10014) 

v_add_u32 v3, v13, s1 // (op 10014) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10014) Store value




s_lshl_b32 s0, s2, 6 // (op 10045) 
s_mul_i32 s1, 6, s0 // (op 10045) 
s_add_u32 s1, s1, 64 // (op 10045) 

v_add_u32 v3, v12, s1 // (op 10045) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10045) Load value

v_accvgpr_read v14, a56 // (op 10061) 
v_accvgpr_read v15, a57 // (op 10061) 
v_accvgpr_read v16, a58 // (op 10061) 
v_accvgpr_read v17, a59 // (op 10061) 
v_mul_f32 v18, v11, v14 // (op 10061) 
v_mul_f32 v19, v11, v15 // (op 10061) 
v_mul_f32 v20, v11, v16 // (op 10061) 
v_mul_f32 v21, v11, v17 // (op 10061) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10043) 
v_mul_f32 v15, v0, v5 // (op 10043) 
v_mul_f32 v16, v0, v6 // (op 10043) 
v_mul_f32 v17, v0, v7 // (op 10043) 

v_add_f32 v4, v14, v18 // (op 10044) 
v_add_f32 v5, v15, v19 // (op 10044) 
v_add_f32 v6, v16, v20 // (op 10044) 
v_add_f32 v7, v17, v21 // (op 10044) 




s_lshl_b32 s0, s4, 6 // (op 10016) 
s_mul_i32 s1, 6, s0 // (op 10016) 
s_add_u32 s1, s1, 64 // (op 10016) 

v_add_u32 v3, v13, s1 // (op 10016) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10016) Store value




s_lshl_b32 s0, s2, 6 // (op 10049) 
s_mul_i32 s1, 7, s0 // (op 10049) 
s_add_u32 s1, s1, 64 // (op 10049) 

v_add_u32 v3, v12, s1 // (op 10049) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 10049) Load value

v_accvgpr_read v14, a60 // (op 10063) 
v_accvgpr_read v15, a61 // (op 10063) 
v_accvgpr_read v16, a62 // (op 10063) 
v_accvgpr_read v17, a63 // (op 10063) 
v_mul_f32 v18, v11, v14 // (op 10063) 
v_mul_f32 v19, v11, v15 // (op 10063) 
v_mul_f32 v20, v11, v16 // (op 10063) 
v_mul_f32 v21, v11, v17 // (op 10063) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 10047) 
v_mul_f32 v15, v0, v5 // (op 10047) 
v_mul_f32 v16, v0, v6 // (op 10047) 
v_mul_f32 v17, v0, v7 // (op 10047) 

v_add_f32 v4, v14, v18 // (op 10048) 
v_add_f32 v5, v15, v19 // (op 10048) 
v_add_f32 v6, v16, v20 // (op 10048) 
v_add_f32 v7, v17, v21 // (op 10048) 




s_lshl_b32 s0, s4, 6 // (op 10018) 
s_mul_i32 s1, 7, s0 // (op 10018) 
s_add_u32 s1, s1, 64 // (op 10018) 

v_add_u32 v3, v13, s1 // (op 10018) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 10018) Store value




v_mov_b32 v3, v12 // (op 126) reg expression

s_nop 0
buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 126) Load value

v_accvgpr_read v14, a0 // (op 123) 
v_accvgpr_read v15, a1 // (op 123) 
v_accvgpr_read v16, a2 // (op 123) 
v_accvgpr_read v17, a3 // (op 123) 
v_mul_f32 v18, v11, v14 // (op 123) 
v_mul_f32 v19, v11, v15 // (op 123) 
v_mul_f32 v20, v11, v16 // (op 123) 
v_mul_f32 v21, v11, v17 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 127) 
v_mul_f32 v15, v0, v5 // (op 127) 
v_mul_f32 v16, v0, v6 // (op 127) 
v_mul_f32 v17, v0, v7 // (op 127) 

v_add_f32 v4, v14, v18 // (op 124) 
v_add_f32 v5, v15, v19 // (op 124) 
v_add_f32 v6, v16, v20 // (op 124) 
v_add_f32 v7, v17, v21 // (op 124) 




v_mov_b32 v3, v13 // (op 125) reg expression

buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 9839) 
v_add_u32 v3, v12, s0 // (op 9839) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 9839) Load value

v_accvgpr_read v14, a4 // (op 9836) 
v_accvgpr_read v15, a5 // (op 9836) 
v_accvgpr_read v16, a6 // (op 9836) 
v_accvgpr_read v17, a7 // (op 9836) 
v_mul_f32 v18, v11, v14 // (op 9836) 
v_mul_f32 v19, v11, v15 // (op 9836) 
v_mul_f32 v20, v11, v16 // (op 9836) 
v_mul_f32 v21, v11, v17 // (op 9836) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 9840) 
v_mul_f32 v15, v0, v5 // (op 9840) 
v_mul_f32 v16, v0, v6 // (op 9840) 
v_mul_f32 v17, v0, v7 // (op 9840) 

v_add_f32 v4, v14, v18 // (op 9837) 
v_add_f32 v5, v15, v19 // (op 9837) 
v_add_f32 v6, v16, v20 // (op 9837) 
v_add_f32 v7, v17, v21 // (op 9837) 




s_lshl_b32 s0, s4, 6 // (op 9838) 
v_add_u32 v3, v13, s0 // (op 9838) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 9838) Store value




s_lshl_b32 s0, s2, 7 // (op 9851) 
v_add_u32 v3, v12, s0 // (op 9851) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 9851) Load value

v_accvgpr_read v14, a8 // (op 9848) 
v_accvgpr_read v15, a9 // (op 9848) 
v_accvgpr_read v16, a10 // (op 9848) 
v_accvgpr_read v17, a11 // (op 9848) 
v_mul_f32 v18, v11, v14 // (op 9848) 
v_mul_f32 v19, v11, v15 // (op 9848) 
v_mul_f32 v20, v11, v16 // (op 9848) 
v_mul_f32 v21, v11, v17 // (op 9848) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 9852) 
v_mul_f32 v15, v0, v5 // (op 9852) 
v_mul_f32 v16, v0, v6 // (op 9852) 
v_mul_f32 v17, v0, v7 // (op 9852) 

v_add_f32 v4, v14, v18 // (op 9849) 
v_add_f32 v5, v15, v19 // (op 9849) 
v_add_f32 v6, v16, v20 // (op 9849) 
v_add_f32 v7, v17, v21 // (op 9849) 




s_lshl_b32 s0, s4, 7 // (op 9850) 
v_add_u32 v3, v13, s0 // (op 9850) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 9850) Store value




s_lshl_b32 s0, s2, 6 // (op 9863) 
s_mul_i32 s1, 3, s0 // (op 9863) 

v_add_u32 v3, v12, s1 // (op 9863) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 9863) Load value

v_accvgpr_read v14, a12 // (op 9860) 
v_accvgpr_read v15, a13 // (op 9860) 
v_accvgpr_read v16, a14 // (op 9860) 
v_accvgpr_read v17, a15 // (op 9860) 
v_mul_f32 v18, v11, v14 // (op 9860) 
v_mul_f32 v19, v11, v15 // (op 9860) 
v_mul_f32 v20, v11, v16 // (op 9860) 
v_mul_f32 v21, v11, v17 // (op 9860) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 9864) 
v_mul_f32 v15, v0, v5 // (op 9864) 
v_mul_f32 v16, v0, v6 // (op 9864) 
v_mul_f32 v17, v0, v7 // (op 9864) 

v_add_f32 v4, v14, v18 // (op 9861) 
v_add_f32 v5, v15, v19 // (op 9861) 
v_add_f32 v6, v16, v20 // (op 9861) 
v_add_f32 v7, v17, v21 // (op 9861) 




s_lshl_b32 s0, s4, 6 // (op 9862) 
s_mul_i32 s1, 3, s0 // (op 9862) 

v_add_u32 v3, v13, s1 // (op 9862) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 9862) Store value




s_lshl_b32 s0, s2, 8 // (op 9875) 
v_add_u32 v3, v12, s0 // (op 9875) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 9875) Load value

v_accvgpr_read v14, a16 // (op 9872) 
v_accvgpr_read v15, a17 // (op 9872) 
v_accvgpr_read v16, a18 // (op 9872) 
v_accvgpr_read v17, a19 // (op 9872) 
v_mul_f32 v18, v11, v14 // (op 9872) 
v_mul_f32 v19, v11, v15 // (op 9872) 
v_mul_f32 v20, v11, v16 // (op 9872) 
v_mul_f32 v21, v11, v17 // (op 9872) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 9876) 
v_mul_f32 v15, v0, v5 // (op 9876) 
v_mul_f32 v16, v0, v6 // (op 9876) 
v_mul_f32 v17, v0, v7 // (op 9876) 

v_add_f32 v4, v14, v18 // (op 9873) 
v_add_f32 v5, v15, v19 // (op 9873) 
v_add_f32 v6, v16, v20 // (op 9873) 
v_add_f32 v7, v17, v21 // (op 9873) 




s_lshl_b32 s0, s4, 8 // (op 9874) 
v_add_u32 v3, v13, s0 // (op 9874) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 9874) Store value




s_lshl_b32 s0, s2, 6 // (op 9887) 
s_mul_i32 s1, 5, s0 // (op 9887) 

v_add_u32 v3, v12, s1 // (op 9887) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 9887) Load value

v_accvgpr_read v14, a20 // (op 9884) 
v_accvgpr_read v15, a21 // (op 9884) 
v_accvgpr_read v16, a22 // (op 9884) 
v_accvgpr_read v17, a23 // (op 9884) 
v_mul_f32 v18, v11, v14 // (op 9884) 
v_mul_f32 v19, v11, v15 // (op 9884) 
v_mul_f32 v20, v11, v16 // (op 9884) 
v_mul_f32 v21, v11, v17 // (op 9884) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 9888) 
v_mul_f32 v15, v0, v5 // (op 9888) 
v_mul_f32 v16, v0, v6 // (op 9888) 
v_mul_f32 v17, v0, v7 // (op 9888) 

v_add_f32 v4, v14, v18 // (op 9885) 
v_add_f32 v5, v15, v19 // (op 9885) 
v_add_f32 v6, v16, v20 // (op 9885) 
v_add_f32 v7, v17, v21 // (op 9885) 




s_lshl_b32 s0, s4, 6 // (op 9886) 
s_mul_i32 s1, 5, s0 // (op 9886) 

v_add_u32 v3, v13, s1 // (op 9886) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 9886) Store value




s_lshl_b32 s0, s2, 6 // (op 9899) 
s_mul_i32 s1, 6, s0 // (op 9899) 

v_add_u32 v3, v12, s1 // (op 9899) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 9899) Load value

v_accvgpr_read v14, a24 // (op 9896) 
v_accvgpr_read v15, a25 // (op 9896) 
v_accvgpr_read v16, a26 // (op 9896) 
v_accvgpr_read v17, a27 // (op 9896) 
v_mul_f32 v18, v11, v14 // (op 9896) 
v_mul_f32 v19, v11, v15 // (op 9896) 
v_mul_f32 v20, v11, v16 // (op 9896) 
v_mul_f32 v21, v11, v17 // (op 9896) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 9900) 
v_mul_f32 v15, v0, v5 // (op 9900) 
v_mul_f32 v16, v0, v6 // (op 9900) 
v_mul_f32 v17, v0, v7 // (op 9900) 

v_add_f32 v4, v14, v18 // (op 9897) 
v_add_f32 v5, v15, v19 // (op 9897) 
v_add_f32 v6, v16, v20 // (op 9897) 
v_add_f32 v7, v17, v21 // (op 9897) 




s_lshl_b32 s0, s4, 6 // (op 9898) 
s_mul_i32 s1, 6, s0 // (op 9898) 

v_add_u32 v3, v13, s1 // (op 9898) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 9898) Store value




s_lshl_b32 s0, s2, 6 // (op 9911) 
s_mul_i32 s1, 7, s0 // (op 9911) 

v_add_u32 v3, v12, s1 // (op 9911) 


buffer_load_dwordx4 v[4:7], v3, s[40:43], 0 offen // (op 9911) Load value

v_accvgpr_read v14, a28 // (op 9908) 
v_accvgpr_read v15, a29 // (op 9908) 
v_accvgpr_read v16, a30 // (op 9908) 
v_accvgpr_read v17, a31 // (op 9908) 
v_mul_f32 v18, v11, v14 // (op 9908) 
v_mul_f32 v19, v11, v15 // (op 9908) 
v_mul_f32 v20, v11, v16 // (op 9908) 
v_mul_f32 v21, v11, v17 // (op 9908) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v4 // (op 9912) 
v_mul_f32 v15, v0, v5 // (op 9912) 
v_mul_f32 v16, v0, v6 // (op 9912) 
v_mul_f32 v17, v0, v7 // (op 9912) 

v_add_f32 v4, v14, v18 // (op 9909) 
v_add_f32 v5, v15, v19 // (op 9909) 
v_add_f32 v6, v16, v20 // (op 9909) 
v_add_f32 v7, v17, v21 // (op 9909) 




s_lshl_b32 s0, s4, 6 // (op 9910) 
s_mul_i32 s1, 7, s0 // (op 9910) 

v_add_u32 v3, v13, s1 // (op 9910) 


buffer_store_dwordx4 v[4:7], v3, s[44:47], 0 offen // (op 9910) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_ConditionalBottom_0_beta_114:








































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4_end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4
  .amdhsa_next_free_vgpr 280
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 98304 // lds bytes
  .amdhsa_accum_offset 152
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
    macrotile: [128, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_4.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 98304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 66
    .vgpr_count: 152
    .agpr_count: 128
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
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

