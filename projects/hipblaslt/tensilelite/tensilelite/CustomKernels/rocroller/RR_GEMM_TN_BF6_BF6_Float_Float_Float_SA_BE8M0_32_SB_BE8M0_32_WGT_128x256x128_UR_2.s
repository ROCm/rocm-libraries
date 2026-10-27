// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2,@function
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_exec_begin:


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


v_lshrrev_b32 v0, 7, v1 // (op 13644) 
v_and_b32 v3, 15, v1 // (op 13644) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 13644) 


v_and_b32 v0, 63, v1 // (op 13644) 
v_lshrrev_b32 v3, 4, v0 // (op 13644) 

v_lshlrev_b32 v0, 5, v3 // (op 13644) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 13644) 


v_mul_lo_u32 v0, v3, 6 // (op 13644) 

v_lshrrev_b32 v3, 3, v0 // (op 13644) 

v_lshrrev_b32 v0, 6, v1 // (op 13702) 
v_and_b32 v4, 1, v0 // (op 13702) 

v_and_b32 v0, 15, v1 // (op 13702) 
v_lshl_add_u32 v5, v4, 7, v0 // (op 13702) 


v_and_b32 v0, 63, v1 // (op 13702) 
v_lshrrev_b32 v4, 4, v0 // (op 13702) 

v_lshlrev_b32 v0, 5, v4 // (op 13702) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 13702) 


v_mul_lo_u32 v0, v4, 6 // (op 13702) 

v_lshrrev_b32 v4, 3, v0 // (op 13702) 

v_lshrrev_b32 v0, 7, v1 // (op 13754) 
v_and_b32 v5, 15, v1 // (op 13754) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 13754) 


v_and_b32 v0, 63, v1 // (op 13754) 
v_lshrrev_b32 v5, 4, v0 // (op 13754) 

v_lshlrev_b32 v0, 5, v5 // (op 13754) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 13754) 


v_mul_lo_u32 v0, v5, 6 // (op 13754) 

v_lshrrev_b32 v5, 3, v0 // (op 13754) 

v_lshrrev_b32 v0, 6, v1 // (op 13783) 
v_and_b32 v6, 1, v0 // (op 13783) 

v_and_b32 v0, 15, v1 // (op 13783) 
v_lshl_add_u32 v7, v6, 7, v0 // (op 13783) 


v_and_b32 v0, 63, v1 // (op 13783) 
v_lshrrev_b32 v6, 4, v0 // (op 13783) 

v_lshlrev_b32 v0, 5, v6 // (op 13783) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 13783) 


v_mul_lo_u32 v0, v6, 6 // (op 13783) 

v_lshrrev_b32 v6, 3, v0 // (op 13783) 

v_lshrrev_b32 v0, 7, v1 // (op 13582) 
v_and_b32 v7, 63, v1 // (op 13582) 
v_lshrrev_b32 v8, 4, v7 // (op 13582) 
v_lshlrev_b32 v9, 2, v8 // (op 13582) 

v_lshl_add_u32 v8, v0, 6, v9 // (op 13582) 


v_lshl_add_u32 v0, s16, 7, v8 // (op 13582) 

v_lshrrev_b32 v8, 2, v7 // (op 13582) 

v_and_b32 v7, 3, v8 // (op 13582) 

v_and_b32 v8, 3, v1 // (op 13582) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 13582) 


v_lshrrev_b32 v7, 6, v1 // (op 13582) 
v_and_b32 v8, 1, v7 // (op 13582) 

v_lshl_add_u32 v7, v8, 7, v9 // (op 13582) 


v_lshl_add_u32 v8, s17, 8, v7 // (op 13582) 

v_mul_lo_u32 v7, v8, s2 // (op 13582) 

v_add_lshl_u32 v8, v0, v7, 2 // (op 13582) 


v_lshrrev_b32 v0, 7, v1 // (op 13723) 
v_and_b32 v7, 63, v1 // (op 13723) 
v_lshrrev_b32 v9, 4, v7 // (op 13723) 
s_mul_i32 s1, s2, s9 // (op 13584) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 13584) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 13584) least significant
s_mul_hi_u32 s19, s2, s8 // (op 13584) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13584) most significant: sum
s_add_u32 s1, s1, s19 // (op 13584) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13584) 

v_lshlrev_b32 v10, 2, v9 // (op 13723) 

v_lshl_add_u32 v9, v0, 6, v10 // (op 13723) 


v_lshl_add_u32 v0, s16, 7, v9 // (op 13723) 

v_lshrrev_b32 v9, 2, v7 // (op 13723) 

v_and_b32 v7, 3, v9 // (op 13723) 

v_and_b32 v9, 3, v1 // (op 13723) 
v_lshl_add_u32 v10, v7, 2, v9 // (op 13723) 


v_lshrrev_b32 v7, 6, v1 // (op 13723) 
v_and_b32 v9, 1, v7 // (op 13723) 

v_lshl_add_u32 v7, v9, 7, v10 // (op 13723) 


v_lshl_add_u32 v9, s17, 8, v7 // (op 13723) 

v_mul_lo_u32 v7, v9, s4 // (op 13723) 

v_add_lshl_u32 v9, v0, v7, 2 // (op 13723) 


s_mov_b32 s40, s6 // (op 13584) reg expression
s_mov_b32 s41, s7 // (op 13584) reg expression
s_mov_b32 s42, s18 // (op 13584) reg expression
s_mov_b32 s43, 131072 // (op 13584) call()

s_mul_i32 s1, s4, s9 // (op 13725) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 13725) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 13725) least significant
s_mul_hi_u32 s7, s4, s8 // (op 13725) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13725) most significant: sum
s_add_u32 s1, s1, s7 // (op 13725) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 13725) 

s_mov_b32 s44, s10 // (op 13725) reg expression
s_mov_b32 s45, s11 // (op 13725) reg expression
s_mov_b32 s46, s6 // (op 13725) reg expression
s_mov_b32 s47, 131072 // (op 13725) call()


v_mov_b32 v0, s14 // (op 2) Move value
v_mov_b32 v7, s15 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_accvgpr_write a4, 0.00000 // (op 277) call()
v_accvgpr_write a5, 0.00000 // (op 277) call()
v_accvgpr_write a6, 0.00000 // (op 277) call()
v_accvgpr_write a7, 0.00000 // (op 277) call()
v_accvgpr_write a8, 0.00000 // (op 413) call()
v_accvgpr_write a9, 0.00000 // (op 413) call()
v_accvgpr_write a10, 0.00000 // (op 413) call()
v_accvgpr_write a11, 0.00000 // (op 413) call()
v_accvgpr_write a12, 0.00000 // (op 549) call()
v_accvgpr_write a13, 0.00000 // (op 549) call()
v_accvgpr_write a14, 0.00000 // (op 549) call()
v_accvgpr_write a15, 0.00000 // (op 549) call()
v_accvgpr_write a16, 0.00000 // (op 685) call()
v_accvgpr_write a17, 0.00000 // (op 685) call()
v_accvgpr_write a18, 0.00000 // (op 685) call()
v_accvgpr_write a19, 0.00000 // (op 685) call()
v_accvgpr_write a20, 0.00000 // (op 821) call()
v_accvgpr_write a21, 0.00000 // (op 821) call()
v_accvgpr_write a22, 0.00000 // (op 821) call()
v_accvgpr_write a23, 0.00000 // (op 821) call()
v_accvgpr_write a24, 0.00000 // (op 957) call()
v_accvgpr_write a25, 0.00000 // (op 957) call()
v_accvgpr_write a26, 0.00000 // (op 957) call()
v_accvgpr_write a27, 0.00000 // (op 957) call()
v_accvgpr_write a28, 0.00000 // (op 1093) call()
v_accvgpr_write a29, 0.00000 // (op 1093) call()
v_accvgpr_write a30, 0.00000 // (op 1093) call()
v_accvgpr_write a31, 0.00000 // (op 1093) call()
v_accvgpr_write a32, 0.00000 // (op 1405) call()
v_accvgpr_write a33, 0.00000 // (op 1405) call()
v_accvgpr_write a34, 0.00000 // (op 1405) call()
v_accvgpr_write a35, 0.00000 // (op 1405) call()
v_accvgpr_write a36, 0.00000 // (op 1926) call()
v_accvgpr_write a37, 0.00000 // (op 1926) call()
v_accvgpr_write a38, 0.00000 // (op 1926) call()
v_accvgpr_write a39, 0.00000 // (op 1926) call()
v_accvgpr_write a40, 0.00000 // (op 1927) call()
v_accvgpr_write a41, 0.00000 // (op 1927) call()
v_accvgpr_write a42, 0.00000 // (op 1927) call()
v_accvgpr_write a43, 0.00000 // (op 1927) call()
v_accvgpr_write a44, 0.00000 // (op 1928) call()
v_accvgpr_write a45, 0.00000 // (op 1928) call()
v_accvgpr_write a46, 0.00000 // (op 1928) call()
v_accvgpr_write a47, 0.00000 // (op 1928) call()
v_accvgpr_write a48, 0.00000 // (op 1929) call()
v_accvgpr_write a49, 0.00000 // (op 1929) call()
v_accvgpr_write a50, 0.00000 // (op 1929) call()
v_accvgpr_write a51, 0.00000 // (op 1929) call()
v_accvgpr_write a52, 0.00000 // (op 1930) call()
v_accvgpr_write a53, 0.00000 // (op 1930) call()
v_accvgpr_write a54, 0.00000 // (op 1930) call()
v_accvgpr_write a55, 0.00000 // (op 1930) call()
v_accvgpr_write a56, 0.00000 // (op 1931) call()
v_accvgpr_write a57, 0.00000 // (op 1931) call()
v_accvgpr_write a58, 0.00000 // (op 1931) call()
v_accvgpr_write a59, 0.00000 // (op 1931) call()
v_accvgpr_write a60, 0.00000 // (op 1932) call()
v_accvgpr_write a61, 0.00000 // (op 1932) call()
v_accvgpr_write a62, 0.00000 // (op 1932) call()
v_accvgpr_write a63, 0.00000 // (op 1932) call()
v_accvgpr_write a64, 0.00000 // (op 2623) call()
v_accvgpr_write a65, 0.00000 // (op 2623) call()
v_accvgpr_write a66, 0.00000 // (op 2623) call()
v_accvgpr_write a67, 0.00000 // (op 2623) call()
v_accvgpr_write a68, 0.00000 // (op 3144) call()
v_accvgpr_write a69, 0.00000 // (op 3144) call()
v_accvgpr_write a70, 0.00000 // (op 3144) call()
v_accvgpr_write a71, 0.00000 // (op 3144) call()
v_accvgpr_write a72, 0.00000 // (op 3145) call()
v_accvgpr_write a73, 0.00000 // (op 3145) call()
v_accvgpr_write a74, 0.00000 // (op 3145) call()
v_accvgpr_write a75, 0.00000 // (op 3145) call()
v_accvgpr_write a76, 0.00000 // (op 3146) call()
v_accvgpr_write a77, 0.00000 // (op 3146) call()
v_accvgpr_write a78, 0.00000 // (op 3146) call()
v_accvgpr_write a79, 0.00000 // (op 3146) call()
v_accvgpr_write a80, 0.00000 // (op 3147) call()
v_accvgpr_write a81, 0.00000 // (op 3147) call()
v_accvgpr_write a82, 0.00000 // (op 3147) call()
v_accvgpr_write a83, 0.00000 // (op 3147) call()
v_accvgpr_write a84, 0.00000 // (op 3148) call()
v_accvgpr_write a85, 0.00000 // (op 3148) call()
v_accvgpr_write a86, 0.00000 // (op 3148) call()
v_accvgpr_write a87, 0.00000 // (op 3148) call()
v_accvgpr_write a88, 0.00000 // (op 3149) call()
v_accvgpr_write a89, 0.00000 // (op 3149) call()
v_accvgpr_write a90, 0.00000 // (op 3149) call()
v_accvgpr_write a91, 0.00000 // (op 3149) call()
v_accvgpr_write a92, 0.00000 // (op 3150) call()
v_accvgpr_write a93, 0.00000 // (op 3150) call()
v_accvgpr_write a94, 0.00000 // (op 3150) call()
v_accvgpr_write a95, 0.00000 // (op 3150) call()
v_accvgpr_write a96, 0.00000 // (op 3841) call()
v_accvgpr_write a97, 0.00000 // (op 3841) call()
v_accvgpr_write a98, 0.00000 // (op 3841) call()
v_accvgpr_write a99, 0.00000 // (op 3841) call()
v_accvgpr_write a100, 0.00000 // (op 4362) call()
v_accvgpr_write a101, 0.00000 // (op 4362) call()
v_accvgpr_write a102, 0.00000 // (op 4362) call()
v_accvgpr_write a103, 0.00000 // (op 4362) call()
v_accvgpr_write a104, 0.00000 // (op 4363) call()
v_accvgpr_write a105, 0.00000 // (op 4363) call()
v_accvgpr_write a106, 0.00000 // (op 4363) call()
v_accvgpr_write a107, 0.00000 // (op 4363) call()
v_accvgpr_write a108, 0.00000 // (op 4364) call()
v_accvgpr_write a109, 0.00000 // (op 4364) call()
v_accvgpr_write a110, 0.00000 // (op 4364) call()
v_accvgpr_write a111, 0.00000 // (op 4364) call()
v_accvgpr_write a112, 0.00000 // (op 4365) call()
v_accvgpr_write a113, 0.00000 // (op 4365) call()
v_accvgpr_write a114, 0.00000 // (op 4365) call()
v_accvgpr_write a115, 0.00000 // (op 4365) call()
v_accvgpr_write a116, 0.00000 // (op 4366) call()
v_accvgpr_write a117, 0.00000 // (op 4366) call()
v_accvgpr_write a118, 0.00000 // (op 4366) call()
v_accvgpr_write a119, 0.00000 // (op 4366) call()
v_accvgpr_write a120, 0.00000 // (op 4367) call()
v_accvgpr_write a121, 0.00000 // (op 4367) call()
v_accvgpr_write a122, 0.00000 // (op 4367) call()
v_accvgpr_write a123, 0.00000 // (op 4367) call()
v_accvgpr_write a124, 0.00000 // (op 4368) call()
v_accvgpr_write a125, 0.00000 // (op 4368) call()
v_accvgpr_write a126, 0.00000 // (op 4368) call()
v_accvgpr_write a127, 0.00000 // (op 4368) call()
v_lshrrev_b32 v10, 1, v1 // (op 13472) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 13472) 

v_and_b32 v10, -2, v1 // (op 13472) 
v_sub_u32 v12, v1, v10 // (op 13472) 

v_lshlrev_b32 v10, 6, v12 // (op 13472) 

v_mov_b32 v13, 0 // (op 13472) convert
v_mov_b32 v12, v10 // (op 13472) convert

v_mul_lo_u32 v15, v11, s13 // (op 13472) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 13472) most significant: high of low * low
v_mul_lo_u32 v14, v11, s12 // (op 13472) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 13472) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 13472) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 13472) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 13472) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 13472) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 13472) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 13472) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 13472) 

v_mov_b32 v10, v12 // (op 13472) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 13474) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 13474) least significant
s_mul_hi_u32 s6, s22, 6 // (op 13474) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13474) most significant: sum

s_mul_i32 s7, s12, s1 // (op 13474) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 13474) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 13474) least significant
s_mul_hi_u32 s11, s12, s0 // (op 13474) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13474) most significant: sum
s_add_u32 s7, s7, s11 // (op 13474) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13474) 

s_mov_b32 s48, s20 // (op 13474) reg expression
s_mov_b32 s49, s21 // (op 13474) reg expression
s_mov_b32 s50, s0 // (op 13474) reg expression
s_mov_b32 s51, 131072 // (op 13474) call()

v_lshrrev_b32 v11, 1, v1 // (op 13491) 
v_lshl_add_u32 v12, s16, 7, v11 // (op 13491) 

v_and_b32 v11, -2, v1 // (op 13491) 
v_sub_u32 v13, v1, v11 // (op 13491) 

v_lshlrev_b32 v11, 6, v13 // (op 13491) 

v_mov_b32 v15, 0 // (op 13491) convert
v_mov_b32 v14, v11 // (op 13491) convert

v_mul_lo_u32 v17, v12, s13 // (op 13491) most significant: low of low * high
v_mul_hi_u32 v11, v12, s12 // (op 13491) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 13491) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 13491) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 13491) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 13491) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 13491) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 13491) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 13491) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 13491) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 13491) 

v_mov_b32 v11, v12 // (op 13491) convert

s_mul_i32 s1, s23, 6 // (op 13493) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 13493) least significant
s_mul_hi_u32 s6, s22, 6 // (op 13493) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13493) most significant: sum

s_mul_i32 s7, s12, s1 // (op 13493) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 13493) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 13493) least significant
s_mul_hi_u32 s11, s12, s0 // (op 13493) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13493) most significant: sum
s_add_u32 s7, s7, s11 // (op 13493) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13493) 

s_mov_b32 s48, s20 // (op 13493) reg expression
s_mov_b32 s49, s21 // (op 13493) reg expression
s_mov_b32 s50, s0 // (op 13493) reg expression
s_mov_b32 s51, 131072 // (op 13493) call()

v_and_b32 v12, -2, v1 // (op 13536) 
v_sub_u32 v13, v1, v12 // (op 13536) 

v_lshlrev_b32 v12, 6, v13 // (op 13536) 

v_mov_b32 v15, 0 // (op 13536) convert
v_mov_b32 v14, v12 // (op 13536) convert

v_and_b32 v12, 4294967294, v1 // (op 13536) 
v_lshl_add_u32 v13, s17, 8, v12 // (op 13536) 

v_mul_lo_u32 v17, v13, s25 // (op 13536) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 13536) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 13536) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 13536) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 13536) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 13536) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 13536) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 13536) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 13536) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 13536) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 13536) 

v_mov_b32 v14, v12 // (op 13536) convert

s_mul_i32 s1, s9, 6 // (op 13538) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 13538) least significant
s_mul_hi_u32 s6, s8, 6 // (op 13538) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13538) most significant: sum

s_mul_i32 s7, s24, s1 // (op 13538) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 13538) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 13538) least significant
s_mul_hi_u32 s11, s24, s0 // (op 13538) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13538) most significant: sum
s_add_u32 s7, s7, s11 // (op 13538) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13538) 

s_mov_b32 s52, s26 // (op 13538) reg expression
s_mov_b32 s53, s27 // (op 13538) reg expression
s_mov_b32 s54, s0 // (op 13538) reg expression
s_mov_b32 s55, 131072 // (op 13538) call()

v_and_b32 v12, -2, v1 // (op 13555) 
v_sub_u32 v13, v1, v12 // (op 13555) 

v_lshlrev_b32 v12, 6, v13 // (op 13555) 

v_mov_b32 v17, 0 // (op 13555) convert
v_mov_b32 v16, v12 // (op 13555) convert

v_and_b32 v15, 4294967294, v1 // (op 13555) 
v_lshl_add_u32 v12, s17, 8, v15 // (op 13555) 

v_mul_lo_u32 v19, v12, s25 // (op 13555) most significant: low of low * high
v_mul_hi_u32 v13, v12, s24 // (op 13555) most significant: high of low * low
v_mul_lo_u32 v18, v12, s24 // (op 13555) least significant: low of low * low
v_add_u32 v19, v19, v13 // (op 13555) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 13555) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 13555) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 13555) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 13555) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 13555) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 13555) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 13555) 

v_mov_b32 v15, v12 // (op 13555) convert

s_mul_i32 s1, s9, 6 // (op 13557) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 13557) least significant
s_mul_hi_u32 s6, s8, 6 // (op 13557) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13557) most significant: sum

s_mul_i32 s7, s24, s1 // (op 13557) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 13557) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 13557) least significant
s_mul_hi_u32 s11, s24, s0 // (op 13557) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13557) most significant: sum
s_add_u32 s7, s7, s11 // (op 13557) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13557) 

s_mov_b32 s52, s26 // (op 13557) reg expression
s_mov_b32 s53, s27 // (op 13557) reg expression
s_mov_b32 s54, s0 // (op 13557) reg expression
s_mov_b32 s55, 131072 // (op 13557) call()

v_and_b32 v12, 63, v1 // (op 13805) 
v_lshrrev_b32 v13, 4, v12 // (op 13805) 
v_and_b32 v16, 3, v13 // (op 13805) 

v_and_b32 v13, 15, v1 // (op 13805) 
v_lshl_add_u32 v17, v16, 4, v13 // (op 13805) 


v_lshrrev_b32 v13, 7, v1 // (op 13805) 
v_lshl_add_u32 v16, v13, 6, v17 // (op 13805) 


v_lshl_add_u32 v13, s16, 7, v16 // (op 13805) 

v_lshrrev_b32 v16, 6, v12 // (op 13805) 

v_lshlrev_b32 v12, 2, v16 // (op 13805) 

v_mul_lo_u32 v16, v13, s28 // (op 13805) 
v_add_u32 v16, v16, v12 // (op 13805) 


s_mul_i32 s1, s28, s23 // (op 13807) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 13807) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 13807) least significant
s_mul_hi_u32 s7, s28, s22 // (op 13807) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13807) most significant: sum
s_add_u32 s1, s1, s7 // (op 13807) most significant: sum


s_mov_b32 s56, s30 // (op 13807) reg expression
s_mov_b32 s57, s31 // (op 13807) reg expression
s_mov_b32 s58, s0 // (op 13807) reg expression
s_mov_b32 s59, 131072 // (op 13807) call()

v_and_b32 v12, 63, v1 // (op 13826) 
v_lshrrev_b32 v13, 4, v12 // (op 13826) 
v_and_b32 v17, 3, v13 // (op 13826) 

v_and_b32 v13, 15, v1 // (op 13826) 
v_lshl_add_u32 v18, v17, 4, v13 // (op 13826) 


v_lshrrev_b32 v13, 7, v1 // (op 13826) 
v_lshl_add_u32 v17, v13, 6, v18 // (op 13826) 


v_lshl_add_u32 v13, s16, 7, v17 // (op 13826) 

v_lshrrev_b32 v17, 6, v12 // (op 13826) 

v_lshlrev_b32 v12, 2, v17 // (op 13826) 

v_mul_lo_u32 v17, v13, s28 // (op 13826) 
v_add_u32 v17, v17, v12 // (op 13826) 


s_mul_i32 s1, s28, s23 // (op 13828) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 13828) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 13828) least significant
s_mul_hi_u32 s7, s28, s22 // (op 13828) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13828) most significant: sum
s_add_u32 s1, s1, s7 // (op 13828) most significant: sum


s_mov_b32 s56, s30 // (op 13828) reg expression
s_mov_b32 s57, s31 // (op 13828) reg expression
s_mov_b32 s58, s0 // (op 13828) reg expression
s_mov_b32 s59, 131072 // (op 13828) call()

v_and_b32 v12, 63, v1 // (op 13857) 
v_lshrrev_b32 v13, 4, v12 // (op 13857) 
v_and_b32 v18, 3, v13 // (op 13857) 

v_and_b32 v13, 15, v1 // (op 13857) 
v_lshl_add_u32 v19, v18, 4, v13 // (op 13857) 


v_lshrrev_b32 v13, 6, v1 // (op 13857) 
v_and_b32 v18, 1, v13 // (op 13857) 

v_lshl_add_u32 v13, v18, 7, v19 // (op 13857) 


v_lshl_add_u32 v18, s17, 8, v13 // (op 13857) 

v_mul_lo_u32 v13, v18, s32 // (op 13857) 

v_lshrrev_b32 v18, 6, v12 // (op 13857) 

v_lshl_add_u32 v12, v18, 2, v13 // (op 13857) 


s_mul_i32 s1, s32, s9 // (op 13859) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 13859) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 13859) least significant
s_mul_hi_u32 s7, s32, s8 // (op 13859) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13859) most significant: sum
s_add_u32 s1, s1, s7 // (op 13859) most significant: sum


s_mov_b32 s60, s34 // (op 13859) reg expression
s_mov_b32 s61, s35 // (op 13859) reg expression
s_mov_b32 s62, s0 // (op 13859) reg expression
s_mov_b32 s63, 131072 // (op 13859) call()

v_and_b32 v13, 63, v1 // (op 13878) 
v_lshrrev_b32 v18, 4, v13 // (op 13878) 
v_and_b32 v19, 3, v18 // (op 13878) 

v_and_b32 v18, 15, v1 // (op 13878) 
v_lshl_add_u32 v20, v19, 4, v18 // (op 13878) 


v_lshrrev_b32 v18, 6, v1 // (op 13878) 
v_and_b32 v19, 1, v18 // (op 13878) 

v_lshl_add_u32 v18, v19, 7, v20 // (op 13878) 


v_lshl_add_u32 v19, s17, 8, v18 // (op 13878) 

v_mul_lo_u32 v18, v19, s32 // (op 13878) 

v_lshrrev_b32 v19, 6, v13 // (op 13878) 

v_lshl_add_u32 v13, v19, 2, v18 // (op 13878) 


s_mul_i32 s1, s32, s9 // (op 13880) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 13880) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 13880) least significant
s_mul_hi_u32 s7, s32, s8 // (op 13880) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13880) most significant: sum
s_add_u32 s1, s1, s7 // (op 13880) most significant: sum


s_mov_b32 s60, s34 // (op 13880) reg expression
s_mov_b32 s61, s35 // (op 13880) reg expression
s_mov_b32 s62, s0 // (op 13880) reg expression
s_mov_b32 s63, 131072 // (op 13880) call()


s_mul_i32 s1, s25, 6 // (op 12826) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 12826) least significant
s_mul_hi_u32 s6, s24, 6 // (op 12826) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 12826) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 12826) 


v_mov_b32 v20, v14 // (op 12826) 
buffer_load_dwordx4 v[22:25], v20, s[52:55], 0 offen // (op 12826) Load value
buffer_load_dwordx2 v[26:27], v20, s[52:55], 0 offen offset:16 // (op 12826) Load value
buffer_load_dwordx4 v[28:31], v20, s[52:55], 0 offen offset:24 // (op 12826) Load value
buffer_load_dwordx2 v[32:33], v20, s[52:55], 0 offen offset:40 // (op 12826) Load value
v_add_u32 v20, v20, v18 // (op 12826) 
buffer_load_dwordx4 v[34:37], v20, s[52:55], 0 offen // (op 12826) Load value
buffer_load_dwordx2 v[38:39], v20, s[52:55], 0 offen offset:16 // (op 12826) Load value
buffer_load_dwordx4 v[40:43], v20, s[52:55], 0 offen offset:24 // (op 12826) Load value
buffer_load_dwordx2 v[44:45], v20, s[52:55], 0 offen offset:40 // (op 12826) Load value





buffer_load_dwordx4 v[46:49], v10, s[48:51], 0 offen // (op 12829) Load value
buffer_load_dwordx2 v[50:51], v10, s[48:51], 0 offen offset:16 // (op 12829) Load value
buffer_load_dwordx4 v[52:55], v10, s[48:51], 0 offen offset:24 // (op 12829) Load value
buffer_load_dwordx2 v[56:57], v10, s[48:51], 0 offen offset:40 // (op 12829) Load value


v_mov_b32 v18, v12 // (op 74) reg expression

buffer_load_dwordx2 v[20:21], v18, s[60:63], 0 offen // (op 74) Load value



s_lshl_b32 s0, s32, 6 // (op 722) 
v_add_u32 v18, v12, s0 // (op 722) 


buffer_load_dwordx2 v[58:59], v18, s[60:63], 0 offen // (op 722) Load value




v_mov_b32 v12, v16 // (op 70) reg expression

buffer_load_dwordx2 v[18:19], v12, s[56:59], 0 offen // (op 70) Load value



s_mul_i32 s1, s25, 6 // (op 12832) most significant: low of high * low
s_mul_i32 s0, s24, 6 // (op 12832) least significant
s_mul_hi_u32 s6, s24, 6 // (op 12832) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 12832) most significant: sum

v_lshrrev_b64 v[60:61], 3, s[0:1] // (op 12832) 


v_mov_b32 v12, v14 // (op 12832) 
buffer_load_dwordx4 v[62:65], v12, s[52:55], 0 offen offset:96 // (op 12832) Load value
buffer_load_dwordx2 v[66:67], v12, s[52:55], 0 offen offset:112 // (op 12832) Load value
buffer_load_dwordx4 v[68:71], v12, s[52:55], 0 offen offset:120 // (op 12832) Load value
buffer_load_dwordx2 v[72:73], v12, s[52:55], 0 offen offset:136 // (op 12832) Load value
v_add_u32 v12, v12, v60 // (op 12832) 
buffer_load_dwordx4 v[74:77], v12, s[52:55], 0 offen offset:96 // (op 12832) Load value
buffer_load_dwordx2 v[78:79], v12, s[52:55], 0 offen offset:112 // (op 12832) Load value
buffer_load_dwordx4 v[80:83], v12, s[52:55], 0 offen offset:120 // (op 12832) Load value
buffer_load_dwordx2 v[84:85], v12, s[52:55], 0 offen offset:136 // (op 12832) Load value






buffer_load_dwordx4 v[86:89], v10, s[48:51], 0 offen offset:96 // (op 12835) Load value
buffer_load_dwordx2 v[90:91], v10, s[48:51], 0 offen offset:112 // (op 12835) Load value
buffer_load_dwordx4 v[92:95], v10, s[48:51], 0 offen offset:120 // (op 12835) Load value
buffer_load_dwordx2 v[96:97], v10, s[48:51], 0 offen offset:136 // (op 12835) Load value

v_and_b32 v10, -2, v1 // (op 13685) 
v_sub_u32 v12, v1, v10 // (op 13685) 

v_lshlrev_b32 v10, 6, v12 // (op 13685) 

v_lshrrev_b32 v12, 1, v1 // (op 13685) 
v_lshl_add_u32 v14, v12, 8, v10 // (op 13685) 


v_mul_lo_u32 v10, v14, 6 // (op 13685) 

v_lshrrev_b32 v12, 3, v10 // (op 13685) 




s_waitcnt vmcnt(26)
ds_write_b128 v12, v[22:25] // (op 12837) Store local data 
s_waitcnt vmcnt(25)
ds_write_b64 v12, v[26:27] offset:16 // (op 12837) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v12, v[28:31] offset:24 // (op 12837) Store local data 
s_waitcnt vmcnt(23)
ds_write_b64 v12, v[32:33] offset:40 // (op 12837) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v12, v[34:37] offset:96 // (op 12837) Store local data 
s_waitcnt vmcnt(21)
ds_write_b64 v12, v[38:39] offset:112 // (op 12837) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v12, v[40:43] offset:120 // (op 12837) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v12, v[44:45] offset:136 // (op 12837) Store local data 

v_and_b32 v10, -2, v1 // (op 13627) 
v_sub_u32 v12, v1, v10 // (op 13627) 

v_lshlrev_b32 v10, 6, v12 // (op 13627) 

v_lshrrev_b32 v12, 1, v1 // (op 13627) 
v_lshl_add_u32 v14, v12, 7, v10 // (op 13627) 


v_mul_lo_u32 v10, v14, 6 // (op 13627) 

v_lshrrev_b32 v12, 3, v10 // (op 13627) 




s_waitcnt vmcnt(18)
ds_write_b128 v12, v[46:49] offset:24576 // (op 12838) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v12, v[50:51] offset:24592 // (op 12838) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v12, v[52:55] offset:24600 // (op 12838) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v12, v[56:57] offset:24616 // (op 12838) Store local data 

v_and_b32 v10, -2, v1 // (op 13598) 
v_sub_u32 v12, v1, v10 // (op 13598) 

v_lshlrev_b32 v10, 6, v12 // (op 13598) 

v_lshrrev_b32 v12, 1, v1 // (op 13598) 
v_lshl_add_u32 v14, v12, 7, v10 // (op 13598) 


v_mul_lo_u32 v10, v14, 6 // (op 13598) 

v_lshrrev_b32 v12, 3, v10 // (op 13598) 

v_and_b32 v10, -2, v1 // (op 13656) 
v_sub_u32 v14, v1, v10 // (op 13656) 

v_lshlrev_b32 v10, 6, v14 // (op 13656) 

v_lshrrev_b32 v14, 1, v1 // (op 13656) 
v_lshl_add_u32 v16, v14, 8, v10 // (op 13656) 


v_mul_lo_u32 v10, v16, 6 // (op 13656) 

v_lshrrev_b32 v14, 3, v10 // (op 13656) 

v_and_b32 v10, -2, v1 // (op 13737) 
v_sub_u32 v16, v1, v10 // (op 13737) 

v_lshlrev_b32 v10, 6, v16 // (op 13737) 

v_lshrrev_b32 v16, 1, v1 // (op 13737) 
v_lshl_add_u32 v60, v16, 7, v10 // (op 13737) 


v_mul_lo_u32 v10, v60, 6 // (op 13737) 

v_lshrrev_b32 v16, 3, v10 // (op 13737) 

v_and_b32 v10, -2, v1 // (op 13766) 
v_sub_u32 v60, v1, v10 // (op 13766) 

v_lshlrev_b32 v10, 6, v60 // (op 13766) 

v_lshrrev_b32 v60, 1, v1 // (op 13766) 
v_lshl_add_u32 v61, v60, 8, v10 // (op 13766) 


v_mul_lo_u32 v10, v61, 6 // (op 13766) 

v_lshrrev_b32 v60, 3, v10 // (op 13766) 

s_mov_b32 s0, 0 // (op 2814) call()
s_ashr_i64 s[6:7], s[36:37], 7 // (op 2813) 
s_mov_b32 s1, s6 // (op 2813) convert

s_and_b32 s6, s1, 4294967294 // (op 2813) 

s_cmp_lt_u32 s0, s6 // (op 2813) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoop_2813 // (op 2813) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoop_2813 if false)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoop_2813:
s_add_u32 s0, s0, 2 // (op 2815) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14078) 



ds_read_b128 v[98:101], v3 offset:24576 // (op 60) Load local data 
ds_read_b64 v[102:103], v3 offset:24592 // (op 60) Load local data 
s_barrier  // (op 14091) 



ds_read_b128 v[104:107], v4 // (op 65) Load local data 
ds_read_b64 v[108:109], v4 offset:16 // (op 65) Load local data 



ds_read_b128 v[110:113], v4 offset:1536 // (op 308) Load local data 
ds_read_b64 v[114:115], v4 offset:1552 // (op 308) Load local data 
s_waitcnt vmcnt(14)
v_mov_b32 v116, v20 // (op 13341) reg expression



v_mov_b32 v117, v21 // (op 13341) reg expression
s_waitcnt vmcnt(12)
v_mov_b32 v118, v18 // (op 13368) reg expression
v_mov_b32 v119, v19 // (op 13368) reg expression
ds_read_b128 v[120:123], v4 offset:3072 // (op 444) Load local data 
ds_read_b64 v[124:125], v4 offset:3088 // (op 444) Load local data 
v_bfe_u32 v126, v118, 0, 8 // (op 12144) 



v_bfe_u32 v127, v118, 8, 8 // (op 12144) 
v_bfe_u32 v128, v118, 16, 8 // (op 12144) 
v_bfe_u32 v129, v118, 24, 8 // (op 12144) 
v_permlane16_swap_b32 v126, v127 // (op 12144) 
ds_read_b128 v[130:133], v4 offset:4608 // (op 580) Load local data 
v_permlane16_swap_b32 v128, v129 // (op 12144) 
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 12144) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 12144) 
ds_read_b64 v[134:135], v4 offset:4624 // (op 580) Load local data 



v_bfe_u32 v136, v116, 0, 8 // (op 12239) 
v_bfe_u32 v137, v116, 8, 8 // (op 12239) 
v_bfe_u32 v138, v116, 16, 8 // (op 12239) 
v_bfe_u32 v139, v116, 24, 8 // (op 12239) 
v_permlane16_swap_b32 v136, v137 // (op 12239) 
ds_read_b128 v[140:143], v4 offset:6144 // (op 716) Load local data 
ds_read_b64 v[144:145], v4 offset:6160 // (op 716) Load local data 
v_permlane16_swap_b32 v138, v139 // (op 12239) 



s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 12239) Wait state hazard: VALU Write Readlane Hazard

v_permlane32_swap_b32 v137, v139 // (op 12239) 
s_mul_i32 s7, s25, 6 // (op 10) most significant: low of high * low
s_mul_i32 s6, s24, 6 // (op 10) least significant
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:103], v[104:109], a[0:3], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 76) 
s_mul_hi_u32 s1, s24, 6 // (op 10) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 10) most significant: sum

s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:103], v[110:115], a[4:7], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 311) 
v_lshrrev_b64 v[146:147], 3, s[6:7] // (op 10) 


v_mov_b32 v10, v15 // (op 10) 
v_mov_b32 v61, v10 // (op 10) 
buffer_load_dwordx4 v[22:25], v61, s[52:55], 0 offen offset:192 // (op 10) Load value
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:103], v[120:125], a[8:11], v126, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 447) 
buffer_load_dwordx2 v[26:27], v61, s[52:55], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[28:31], v61, s[52:55], 0 offen offset:216 // (op 10) Load value
v_mov_b32 v148, v58 // (op 13354) reg expression
v_mov_b32 v149, v59 // (op 13354) reg expression
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:103], v[130:135], a[12:15], v126, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 583) 
v_bfe_u32 v150, v148, 0, 8 // (op 12271) 
v_bfe_u32 v151, v148, 8, 8 // (op 12271) 
v_bfe_u32 v152, v148, 16, 8 // (op 12271) 
v_bfe_u32 v153, v148, 24, 8 // (op 12271) 
v_permlane16_swap_b32 v150, v151 // (op 12271) 
ds_read_b128 v[154:157], v4 offset:7680 // (op 852) Load local data 
ds_read_b64 v[158:159], v4 offset:7696 // (op 852) Load local data 
v_permlane16_swap_b32 v152, v153 // (op 12271) 



ds_read_b128 v[160:163], v4 offset:9216 // (op 988) Load local data 
s_nop 0
v_permlane32_swap_b32 v150, v152 // (op 12271) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v151, v153 // (op 12271) 
ds_read_b64 v[164:165], v4 offset:9232 // (op 988) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:103], v[140:145], a[16:19], v126, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 719) 
ds_read_b128 v[166:169], v4 offset:10752 // (op 1124) Load local data 
ds_read_b64 v[170:171], v4 offset:10768 // (op 1124) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:103], v[154:159], a[20:23], v126, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 855) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:103], v[160:165], a[24:27], v126, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 991) 
ds_read_b128 v[172:175], v3 offset:26112 // (op 1625) Load local data 
ds_read_b64 v[176:177], v3 offset:26128 // (op 1625) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:103], v[166:171], a[28:31], v126, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1127) 




ds_read_b128 v[98:101], v3 offset:27648 // (op 2843) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[172:177], v[104:109], a[32:35], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1895) 
ds_read_b64 v[102:103], v3 offset:27664 // (op 2843) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:177], v[110:115], a[36:39], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1857) 
ds_read_b128 v[178:181], v3 offset:29184 // (op 4061) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:177], v[120:125], a[40:43], v127, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1819) 
ds_read_b64 v[182:183], v3 offset:29200 // (op 4061) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:177], v[130:135], a[44:47], v127, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1781) 
buffer_load_dwordx2 v[32:33], v61, s[52:55], 0 offen offset:232 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[172:177], v[140:145], a[48:51], v127, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1743) 
v_add_u32 v61, v61, v146 // (op 10) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[172:177], v[154:159], a[52:55], v127, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1705) 
buffer_load_dwordx4 v[34:37], v61, s[52:55], 0 offen offset:192 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[172:177], v[160:165], a[56:59], v127, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1667) 
buffer_load_dwordx2 v[38:39], v61, s[52:55], 0 offen offset:208 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[172:177], v[166:171], a[60:63], v127, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1629) 
buffer_load_dwordx4 v[40:43], v61, s[52:55], 0 offen offset:216 // (op 10) Load value

buffer_load_dwordx2 v[44:45], v61, s[52:55], 0 offen offset:232 // (op 10) Load value




v_mov_b32 v61, v11 // (op 4) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[98:103], v[104:109], a[64:67], v128, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3113) 
buffer_load_dwordx4 v[46:49], v61, s[48:51], 0 offen offset:192 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[98:103], v[110:115], a[68:71], v128, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3075) 
buffer_load_dwordx2 v[50:51], v61, s[48:51], 0 offen offset:208 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[98:103], v[120:125], a[72:75], v128, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3037) 
buffer_load_dwordx4 v[52:55], v61, s[48:51], 0 offen offset:216 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:103], v[130:135], a[76:79], v128, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2999) 
buffer_load_dwordx2 v[56:57], v61, s[48:51], 0 offen offset:232 // (op 4) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[98:103], v[140:145], a[80:83], v128, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2961) 
s_barrier  // (op 14114) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:103], v[154:159], a[84:87], v128, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2923) 
v_mov_b32 v146, v13 // (op 13349) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[98:103], v[160:165], a[88:91], v128, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2885) 
v_add_u32 v147, 8, v146 // (op 13349) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[98:103], v[166:171], a[92:95], v128, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2847) 
s_waitcnt vmcnt(23)
ds_write_b128 v60, v[62:65] offset:36864 // (op 218) Store local data 

s_waitcnt vmcnt(22)
ds_write_b64 v60, v[66:67] offset:36880 // (op 218) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[178:183], v[104:109], a[96:99], v129, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4331) 
s_waitcnt vmcnt(21)
ds_write_b128 v60, v[68:71] offset:36888 // (op 218) Store local data 

s_waitcnt vmcnt(20)
ds_write_b64 v60, v[72:73] offset:36904 // (op 218) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[178:183], v[110:115], a[100:103], v129, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4293) 
s_waitcnt vmcnt(19)
ds_write_b128 v60, v[74:77] offset:36960 // (op 218) Store local data 

s_waitcnt vmcnt(18)
ds_write_b64 v60, v[78:79] offset:36976 // (op 218) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[178:183], v[120:125], a[104:107], v129, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4255) 
s_waitcnt vmcnt(17)
ds_write_b128 v60, v[80:83] offset:36984 // (op 218) Store local data 

s_waitcnt vmcnt(16)
ds_write_b64 v60, v[84:85] offset:37000 // (op 218) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[178:183], v[130:135], a[108:111], v129, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4217) 


s_barrier  // (op 14108) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[178:183], v[140:145], a[112:115], v129, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4179) 
s_waitcnt vmcnt(15)
ds_write_b128 v16, v[86:89] offset:61440 // (op 208) Store local data 

s_waitcnt vmcnt(14)
ds_write_b64 v16, v[90:91] offset:61456 // (op 208) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[178:183], v[154:159], a[116:119], v129, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4141) 
s_waitcnt vmcnt(13)
ds_write_b128 v16, v[92:95] offset:61464 // (op 208) Store local data 

s_waitcnt vmcnt(12)
ds_write_b64 v16, v[96:97] offset:61480 // (op 208) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[178:183], v[160:165], a[120:123], v129, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4103) 
buffer_load_dwordx2 v[20:21], v147, s[60:63], 0 offen // (op 13349) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[178:183], v[166:171], a[124:127], v129, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4065) 
s_lshl_b32 s1, s32, 6 // (op 13363) 




s_add_u32 s1, s1, 8 // (op 13363) 
v_add_u32 v147, v146, s1 // (op 13363) 


buffer_load_dwordx2 v[58:59], v147, s[60:63], 0 offen // (op 13363) Load value

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14105) 






v_add_u32 v13, 8, v13 // (op 13868) 
v_mov_b32 v98, v17 // (op 13376) 
v_add_u32 v99, 8, v98 // (op 13376) 

buffer_load_dwordx2 v[18:19], v99, s[56:59], 0 offen // (op 13376) Load value


v_add_u32 v17, 8, v17 // (op 13816) 
ds_read_b128 v[98:101], v5 offset:61440 // (op 210) Load local data 

ds_read_b64 v[102:103], v5 offset:61456 // (op 210) Load local data 
s_mul_i32 s7, s25, 6 // (op 219) most significant: low of high * low
s_barrier  // (op 14111) 
s_mul_i32 s6, s24, 6 // (op 219) least significant



s_mul_hi_u32 s1, s24, 6 // (op 219) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 219) most significant: sum

v_lshrrev_b64 v[104:105], 3, s[6:7] // (op 219) 


v_mov_b32 v106, v10 // (op 219) 
ds_read_b128 v[110:113], v6 offset:36864 // (op 212) Load local data 
ds_read_b64 v[114:115], v6 offset:36880 // (op 212) Load local data 
buffer_load_dwordx4 v[62:65], v106, s[52:55], 0 offen offset:288 // (op 219) Load value



v_bfe_u32 v120, v119, 0, 8 // (op 12159) 
v_bfe_u32 v121, v119, 8, 8 // (op 12159) 
v_bfe_u32 v122, v119, 16, 8 // (op 12159) 
v_bfe_u32 v123, v119, 24, 8 // (op 12159) 
v_permlane16_swap_b32 v120, v121 // (op 12159) 
ds_read_b128 v[124:127], v6 offset:38400 // (op 320) Load local data 
ds_read_b64 v[128:129], v6 offset:38416 // (op 320) Load local data 
buffer_load_dwordx2 v[66:67], v106, s[52:55], 0 offen offset:304 // (op 219) Load value



v_permlane16_swap_b32 v122, v123 // (op 12159) 
ds_read_b128 v[130:133], v6 offset:39936 // (op 456) Load local data 
s_nop 0
v_permlane32_swap_b32 v120, v122 // (op 12159) Wait state hazard: VALU Write Readlane Hazard
buffer_load_dwordx4 v[68:71], v106, s[52:55], 0 offen offset:312 // (op 219) Load value
v_permlane32_swap_b32 v121, v123 // (op 12159) 
ds_read_b64 v[134:135], v6 offset:39952 // (op 456) Load local data 




v_bfe_u32 v136, v117, 0, 8 // (op 12254) 
v_bfe_u32 v137, v117, 8, 8 // (op 12254) 
v_bfe_u32 v138, v117, 16, 8 // (op 12254) 
v_bfe_u32 v139, v117, 24, 8 // (op 12254) 
v_permlane16_swap_b32 v136, v137 // (op 12254) 
ds_read_b128 v[140:143], v6 offset:41472 // (op 592) Load local data 
ds_read_b64 v[144:145], v6 offset:41488 // (op 592) Load local data 
v_permlane16_swap_b32 v138, v139 // (op 12254) 



s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 12254) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 12254) 
ds_read_b128 v[150:153], v6 offset:43008 // (op 728) Load local data 

ds_read_b64 v[154:155], v6 offset:43024 // (op 728) Load local data 



s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:103], v[110:115], a[0:3], v120, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 217) 
ds_read_b128 v[156:159], v6 offset:44544 // (op 864) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:103], v[124:129], a[4:7], v120, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 312) 
v_bfe_u32 v116, v149, 0, 8 // (op 12286) 
v_bfe_u32 v117, v149, 8, 8 // (op 12286) 
v_bfe_u32 v118, v149, 16, 8 // (op 12286) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:103], v[130:135], a[8:11], v120, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 448) 
v_bfe_u32 v119, v149, 24, 8 // (op 12286) 
v_permlane16_swap_b32 v116, v117 // (op 12286) 
s_nop 0
v_permlane16_swap_b32 v118, v119 // (op 12286) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:103], v[140:145], a[12:15], v120, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 584) 
s_nop 0
v_permlane32_swap_b32 v116, v118 // (op 12286) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v117, v119 // (op 12286) 
ds_read_b64 v[160:161], v6 offset:44560 // (op 864) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:103], v[150:155], a[16:19], v120, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 720) 
ds_read_b128 v[162:165], v6 offset:46080 // (op 1000) Load local data 
ds_read_b64 v[166:167], v6 offset:46096 // (op 1000) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:103], v[156:161], a[20:23], v120, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 856) 



ds_read_b128 v[168:171], v6 offset:47616 // (op 1136) Load local data 
ds_read_b64 v[172:173], v6 offset:47632 // (op 1136) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:103], v[162:167], a[24:27], v120, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 992) 
buffer_load_dwordx2 v[72:73], v106, s[52:55], 0 offen offset:328 // (op 219) Load value



s_barrier  // (op 14161) 
v_add_u32 v106, v106, v104 // (op 219) 
ds_read_b128 v[174:177], v5 offset:62976 // (op 1637) Load local data 
ds_read_b64 v[178:179], v5 offset:62992 // (op 1637) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:103], v[168:173], a[28:31], v120, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1128) 




ds_read_b128 v[98:101], v5 offset:64512 // (op 2855) Load local data 
buffer_load_dwordx4 v[74:77], v106, s[52:55], 0 offen offset:288 // (op 219) Load value
buffer_load_dwordx2 v[78:79], v106, s[52:55], 0 offen offset:304 // (op 219) Load value
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[174:179], v[110:115], a[32:35], v121, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1896) 
ds_read_b64 v[102:103], v5 offset:64528 // (op 2855) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[174:179], v[130:135], a[40:43], v121, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1820) 
v_add_u32 v107, 66048, v5 // (op 4073) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[174:179], v[140:145], a[44:47], v121, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1782) 
ds_read_b128 v[180:183], v107 // (op 4073) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[174:179], v[124:129], a[36:39], v121, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1858) 
ds_read_b64 v[184:185], v107 offset:16 // (op 4073) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[174:179], v[150:155], a[48:51], v121, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1744) 
s_barrier  // (op 14205) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[174:179], v[162:167], a[56:59], v121, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1668) 
buffer_load_dwordx4 v[80:83], v106, s[52:55], 0 offen offset:312 // (op 219) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[174:179], v[168:173], a[60:63], v121, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1630) 
buffer_load_dwordx2 v[84:85], v106, s[52:55], 0 offen offset:328 // (op 219) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[174:179], v[156:161], a[52:55], v121, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1706) 




v_add_u32 v15, 192, v15 // (op 13547) 
buffer_load_dwordx4 v[86:89], v61, s[48:51], 0 offen offset:288 // (op 207) Load value
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[98:103], v[110:115], a[64:67], v122, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3114) 
buffer_load_dwordx2 v[90:91], v61, s[48:51], 0 offen offset:304 // (op 207) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[98:103], v[130:135], a[72:75], v122, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3038) 
buffer_load_dwordx4 v[92:95], v61, s[48:51], 0 offen offset:312 // (op 207) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:103], v[140:145], a[76:79], v122, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3000) 
buffer_load_dwordx2 v[96:97], v61, s[48:51], 0 offen offset:328 // (op 207) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[98:103], v[124:129], a[68:71], v122, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3076) 
s_barrier  // (op 14094) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[98:103], v[150:155], a[80:83], v122, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2962) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[98:103], v[162:167], a[88:91], v122, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2886) 
v_add_u32 v11, 192, v11 // (op 13483) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[98:103], v[168:173], a[92:95], v122, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2848) 
s_waitcnt vmcnt(26)
ds_write_b128 v14, v[22:25] // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:103], v[156:161], a[84:87], v122, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2924) 
s_waitcnt vmcnt(25)
ds_write_b64 v14, v[26:27] offset:16 // (op 13) Store local data 

s_waitcnt vmcnt(24)
ds_write_b128 v14, v[28:31] offset:24 // (op 13) Store local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[180:185], v[110:115], a[96:99], v123, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4332) 
s_waitcnt vmcnt(23)
ds_write_b64 v14, v[32:33] offset:40 // (op 13) Store local data 

s_waitcnt vmcnt(22)
ds_write_b128 v14, v[34:37] offset:96 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:185], v[130:135], a[104:107], v123, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4256) 
s_waitcnt vmcnt(21)
ds_write_b64 v14, v[38:39] offset:112 // (op 13) Store local data 

s_waitcnt vmcnt(20)
ds_write_b128 v14, v[40:43] offset:120 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:185], v[140:145], a[108:111], v123, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4218) 
s_waitcnt vmcnt(19)
ds_write_b64 v14, v[44:45] offset:136 // (op 13) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[180:185], v[124:129], a[100:103], v123, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4294) 
s_barrier  // (op 14081) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[180:185], v[150:155], a[112:115], v123, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4180) 
s_waitcnt vmcnt(18)
ds_write_b128 v12, v[46:49] offset:24576 // (op 6) Store local data 

s_waitcnt vmcnt(17)
ds_write_b64 v12, v[50:51] offset:24592 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[180:185], v[162:167], a[120:123], v123, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4104) 
s_waitcnt vmcnt(16)
ds_write_b128 v12, v[52:55] offset:24600 // (op 6) Store local data 

s_waitcnt vmcnt(15)
ds_write_b64 v12, v[56:57] offset:24616 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[180:185], v[168:173], a[124:127], v123, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4066) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[180:185], v[156:161], a[116:119], v123, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4142) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 2813) 
s_mov_b32 s1, s6 // (op 2813) convert

s_and_b32 s6, s1, 4294967294 // (op 2813) 

s_cmp_lt_u32 s0, s6 // (op 2813) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoop_2813 // (op 2813) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoop_2813 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoop_2813:













v_lshrrev_b32 v10, 1, v1 // (op 13415) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 13415) 
v_lshl_add_u32 v11, s16, 7, v10 // (op 13415) 

s_lshr_b32 s6, s0, 1 // (op 13415) 

v_and_b32 v10, -2, v1 // (op 13415) 
v_sub_u32 v12, v1, v10 // (op 13415) 

v_lshlrev_b32 v10, 6, v12 // (op 13415) 

v_lshl_add_u32 v12, s6, 8, v10 // (op 13415) 


v_mov_b32 v61, 0 // (op 13415) convert
v_mov_b32 v60, v12 // (op 13415) convert

v_mul_lo_u32 v13, v11, s13 // (op 13415) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 13415) most significant: high of low * low
v_mul_lo_u32 v12, v11, s12 // (op 13415) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 13415) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v60 // (op 13415) least significant half
v_addc_co_u32 v13, s[0:1], v13, v61, s[0:1] // (op 13415) most significant half



v_mul_lo_u32 v11, v13, 6 // (op 13415) most significant: low of high * low
v_mul_hi_u32 v14, v12, 6 // (op 13415) most significant: high of low * low
v_mul_lo_u32 v10, v12, 6 // (op 13415) least significant: low of low * low
v_add_u32 v11, v11, v14 // (op 13415) most significant: sum


v_lshrrev_b64 v[60:61], 3, v[10:11] // (op 13415) 

v_mov_b32 v10, v60 // (op 13415) convert

s_mul_i32 s1, s23, 6 // (op 13417) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 13417) least significant
s_mul_hi_u32 s6, s22, 6 // (op 13417) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13417) most significant: sum

s_mul_i32 s7, s12, s1 // (op 13417) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 13417) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 13417) least significant
s_mul_hi_u32 s11, s12, s0 // (op 13417) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13417) most significant: sum
s_add_u32 s7, s7, s11 // (op 13417) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13417) 

s_mov_b32 s48, s20 // (op 13417) reg expression
s_mov_b32 s49, s21 // (op 13417) reg expression
s_mov_b32 s50, s0 // (op 13417) reg expression
s_mov_b32 s51, 131072 // (op 13417) call()

v_and_b32 v11, -2, v1 // (op 13513) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 13513) 
v_sub_u32 v12, v1, v11 // (op 13513) 

s_lshr_b32 s6, s0, 1 // (op 13513) 

v_lshlrev_b32 v11, 6, v12 // (op 13513) 

v_lshl_add_u32 v12, s6, 8, v11 // (op 13513) 


v_mov_b32 v61, 0 // (op 13513) convert
v_mov_b32 v60, v12 // (op 13513) convert

v_and_b32 v11, 4294967294, v1 // (op 13513) 
v_lshl_add_u32 v12, s17, 8, v11 // (op 13513) 

v_mul_lo_u32 v17, v12, s25 // (op 13513) most significant: low of low * high
v_mul_hi_u32 v11, v12, s24 // (op 13513) most significant: high of low * low
v_mul_lo_u32 v16, v12, s24 // (op 13513) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 13513) most significant: sum


v_add_co_u32 v14, s[0:1], v60, v16 // (op 13513) least significant half
v_addc_co_u32 v15, s[0:1], v61, v17, s[0:1] // (op 13513) most significant half



v_mul_lo_u32 v13, v15, 6 // (op 13513) most significant: low of high * low
v_mul_hi_u32 v11, v14, 6 // (op 13513) most significant: high of low * low
v_mul_lo_u32 v12, v14, 6 // (op 13513) least significant: low of low * low
v_add_u32 v13, v13, v11 // (op 13513) most significant: sum


v_lshrrev_b64 v[60:61], 3, v[12:13] // (op 13513) 

v_mov_b32 v11, v60 // (op 13513) convert

s_mul_i32 s1, s9, 6 // (op 13515) most significant: low of high * low
s_mul_i32 s0, s8, 6 // (op 13515) least significant
s_mul_hi_u32 s6, s8, 6 // (op 13515) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13515) most significant: sum

s_mul_i32 s7, s24, s1 // (op 13515) most significant: low of low * high
s_mul_i32 s10, s25, s0 // (op 13515) most significant: low of high * low
s_mul_i32 s6, s24, s0 // (op 13515) least significant
s_mul_hi_u32 s11, s24, s0 // (op 13515) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13515) most significant: sum
s_add_u32 s7, s7, s11 // (op 13515) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13515) 

s_mov_b32 s52, s26 // (op 13515) reg expression
s_mov_b32 s53, s27 // (op 13515) reg expression
s_mov_b32 s54, s0 // (op 13515) reg expression
s_mov_b32 s55, 131072 // (op 13515) call()

v_and_b32 v12, -2, v1 // (op 13611) 
v_sub_u32 v13, v1, v12 // (op 13611) 

v_lshlrev_b32 v12, 6, v13 // (op 13611) 

v_lshrrev_b32 v13, 1, v1 // (op 13611) 
v_lshl_add_u32 v14, v13, 7, v12 // (op 13611) 


v_mul_lo_u32 v12, v14, 6 // (op 13611) 

v_lshrrev_b32 v13, 3, v12 // (op 13611) 

v_and_b32 v12, -2, v1 // (op 13669) 
v_sub_u32 v14, v1, v12 // (op 13669) 

v_lshlrev_b32 v12, 6, v14 // (op 13669) 

v_lshrrev_b32 v14, 1, v1 // (op 13669) 
v_lshl_add_u32 v15, v14, 8, v12 // (op 13669) 


v_mul_lo_u32 v12, v15, 6 // (op 13669) 

v_lshrrev_b32 v14, 3, v12 // (op 13669) 

v_and_b32 v12, 63, v1 // (op 13905) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 13905) 
v_lshrrev_b32 v15, 4, v12 // (op 13905) 
s_lshr_b32 s6, s0, 1 // (op 13905) 

v_and_b32 v16, 3, v15 // (op 13905) 

v_and_b32 v15, 15, v1 // (op 13905) 
v_lshl_add_u32 v17, v16, 4, v15 // (op 13905) 


v_lshrrev_b32 v15, 7, v1 // (op 13905) 
v_lshl_add_u32 v16, v15, 6, v17 // (op 13905) 


v_lshl_add_u32 v15, s16, 7, v16 // (op 13905) 

v_lshrrev_b32 v16, 6, v12 // (op 13905) 

v_lshlrev_b32 v12, 2, v16 // (op 13905) 

v_lshl_add_u32 v16, s6, 3, v12 // (op 13905) 


v_mul_lo_u32 v12, v15, s28 // (op 13905) 
v_add_u32 v12, v12, v16 // (op 13905) 


s_mul_i32 s1, s28, s23 // (op 13907) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 13907) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 13907) least significant
s_mul_hi_u32 s7, s28, s22 // (op 13907) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13907) most significant: sum
s_add_u32 s1, s1, s7 // (op 13907) most significant: sum


s_mov_b32 s56, s30 // (op 13907) reg expression
s_mov_b32 s57, s31 // (op 13907) reg expression
s_mov_b32 s58, s0 // (op 13907) reg expression
s_mov_b32 s59, 131072 // (op 13907) call()

v_and_b32 v15, 63, v1 // (op 13932) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 13932) 
v_lshrrev_b32 v16, 6, v15 // (op 13932) 
s_lshr_b32 s6, s0, 1 // (op 13932) 

v_lshlrev_b32 v17, 2, v16 // (op 13932) 

v_lshl_add_u32 v16, s6, 3, v17 // (op 13932) 


v_lshrrev_b32 v17, 4, v15 // (op 13932) 

v_and_b32 v15, 3, v17 // (op 13932) 

v_and_b32 v17, 15, v1 // (op 13932) 
v_lshl_add_u32 v22, v15, 4, v17 // (op 13932) 


v_lshrrev_b32 v15, 6, v1 // (op 13932) 
v_and_b32 v17, 1, v15 // (op 13932) 

v_lshl_add_u32 v15, v17, 7, v22 // (op 13932) 


v_lshl_add_u32 v17, s17, 8, v15 // (op 13932) 

v_mul_lo_u32 v15, v17, s32 // (op 13932) 

v_add_u32 v17, v16, v15 // (op 13932) 


s_mul_i32 s1, s32, s9 // (op 13934) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 13934) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 13934) least significant
s_mul_hi_u32 s7, s32, s8 // (op 13934) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13934) most significant: sum
s_add_u32 s1, s1, s7 // (op 13934) most significant: sum


s_mov_b32 s60, s34 // (op 13934) reg expression
s_mov_b32 s61, s35 // (op 13934) reg expression
s_mov_b32 s62, s0 // (op 13934) reg expression
s_mov_b32 s63, 131072 // (op 13934) call()

s_ashr_i64 s[0:1], s[36:37], 7 // (op 4010) 
s_and_b32 s6, s0, 4294967294 // (op 4010) 

s_ashr_i64 s[0:1], s[36:37], 7 // (op 4009) 
s_mov_b32 s7, s0 // (op 4009) convert

s_cmp_lt_u32 s6, s7 // (op 4009) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoopTail_4009 // (op 4009) Condition: Top (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoopTail_4009 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoopTail_4009:


s_add_u32 s6, s6, 1 // (op 4011) 
v_mov_b32 v15, v10 // (op 157) 

buffer_load_dwordx4 v[18:21], v15, s[48:51], 0 offen // (op 157) Load value
buffer_load_dwordx2 v[22:23], v15, s[48:51], 0 offen offset:16 // (op 157) Load value
buffer_load_dwordx4 v[24:27], v15, s[48:51], 0 offen offset:24 // (op 157) Load value
buffer_load_dwordx2 v[28:29], v15, s[48:51], 0 offen offset:40 // (op 157) Load value

v_add_u32 v10, 96, v10 // (op 13405) 

s_barrier  // (op 14084) 
s_mul_i32 s1, s25, 6 // (op 169) most significant: low of high * low



s_mul_i32 s0, s24, 6 // (op 169) least significant
s_mul_hi_u32 s7, s24, 6 // (op 169) most significant: high of low * low
s_add_u32 s1, s1, s7 // (op 169) most significant: sum

v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 169) 


v_mov_b32 v15, v11 // (op 169) 
v_mov_b32 v16, v15 // (op 169) 
buffer_load_dwordx4 v[32:35], v16, s[52:55], 0 offen // (op 169) Load value
buffer_load_dwordx2 v[36:37], v16, s[52:55], 0 offen offset:16 // (op 169) Load value
s_waitcnt vmcnt(5)
ds_write_b128 v13, v[18:21] offset:24576 // (op 158) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v13, v[22:23] offset:24592 // (op 158) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v13, v[24:27] offset:24600 // (op 158) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v13, v[28:29] offset:24616 // (op 158) Store local data 
buffer_load_dwordx4 v[38:41], v16, s[52:55], 0 offen offset:24 // (op 169) Load value

buffer_load_dwordx2 v[42:43], v16, s[52:55], 0 offen offset:40 // (op 169) Load value
v_add_u32 v16, v16, v30 // (op 169) 
buffer_load_dwordx4 v[44:47], v16, s[52:55], 0 offen // (op 169) Load value
buffer_load_dwordx2 v[48:49], v16, s[52:55], 0 offen offset:16 // (op 169) Load value
buffer_load_dwordx4 v[50:53], v16, s[52:55], 0 offen offset:24 // (op 169) Load value
buffer_load_dwordx2 v[54:55], v16, s[52:55], 0 offen offset:40 // (op 169) Load value



v_add_u32 v11, 96, v11 // (op 13505) 
s_barrier  // (op 14097) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14087) 






ds_read_b128 v[18:21], v3 offset:24576 // (op 160) Load local data 
ds_read_b64 v[22:23], v3 offset:24592 // (op 160) Load local data 
s_waitcnt vmcnt(7)
ds_write_b128 v14, v[32:35] // (op 168) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v14, v[36:37] offset:16 // (op 168) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v14, v[38:41] offset:24 // (op 168) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v14, v[42:43] offset:40 // (op 168) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v14, v[44:47] offset:96 // (op 168) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v14, v[48:49] offset:112 // (op 168) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v14, v[50:53] offset:120 // (op 168) Store local data 
s_waitcnt vmcnt(0)
ds_write_b64 v14, v[54:55] offset:136 // (op 168) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14101) 



ds_read_b128 v[24:27], v4 // (op 162) Load local data 
ds_read_b64 v[28:29], v4 offset:16 // (op 162) Load local data 


v_mov_b32 v15, v12 // (op 164) 
v_mov_b32 v16, v15 // (op 164) reg expression

buffer_load_dwordx2 v[30:31], v16, s[56:59], 0 offen // (op 164) Load value


v_add_u32 v12, 4, v12 // (op 13895) 


v_mov_b32 v15, v17 // (op 166) 
v_mov_b32 v16, v15 // (op 166) reg expression

buffer_load_dwordx2 v[32:33], v16, s[60:63], 0 offen // (op 166) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v34, v30, 0, 8 // (op 12367) 



v_bfe_u32 v35, v30, 8, 8 // (op 12367) 
v_bfe_u32 v36, v30, 16, 8 // (op 12367) 
v_bfe_u32 v37, v30, 24, 8 // (op 12367) 
v_permlane16_swap_b32 v34, v35 // (op 12367) 
ds_read_b128 v[38:41], v4 offset:1536 // (op 291) Load local data 
v_permlane16_swap_b32 v36, v37 // (op 12367) 
s_nop 1
v_permlane32_swap_b32 v34, v36 // (op 12367) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v35, v37 // (op 12367) 
ds_read_b64 v[42:43], v4 offset:1552 // (op 291) Load local data 




ds_read_b128 v[44:47], v4 offset:3072 // (op 427) Load local data 
ds_read_b64 v[48:49], v4 offset:3088 // (op 427) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v32, 0, 8 // (op 12413) 



v_bfe_u32 v51, v32, 8, 8 // (op 12413) 
v_bfe_u32 v52, v32, 16, 8 // (op 12413) 
v_bfe_u32 v53, v32, 24, 8 // (op 12413) 
v_permlane16_swap_b32 v50, v51 // (op 12413) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 12413) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 12413) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 12413) 
ds_read_b128 v[54:57], v4 offset:4608 // (op 563) Load local data 

ds_read_b64 v[58:59], v4 offset:4624 // (op 563) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:29], a[0:3], v34, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 167) 
ds_read_b128 v[60:63], v4 offset:6144 // (op 699) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:23], v[38:43], a[4:7], v34, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 294) 
ds_read_b64 v[64:65], v4 offset:6160 // (op 699) Load local data 


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[44:49], a[8:11], v34, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 430) 
s_lshl_b32 s7, s32, 6 // (op 704) 
v_add_u32 v16, v15, s7 // (op 704) 


buffer_load_dwordx2 v[30:31], v16, s[60:63], 0 offen // (op 704) Load value

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:23], v[54:59], a[12:15], v34, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 566) 




v_add_u32 v17, 4, v17 // (op 13922) 
ds_read_b128 v[66:69], v4 offset:7680 // (op 835) Load local data 
ds_read_b64 v[70:71], v4 offset:7696 // (op 835) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v72, v30, 0, 8 // (op 12427) 



v_bfe_u32 v73, v30, 8, 8 // (op 12427) 
v_bfe_u32 v74, v30, 16, 8 // (op 12427) 
v_bfe_u32 v75, v30, 24, 8 // (op 12427) 
v_permlane16_swap_b32 v72, v73 // (op 12427) 
s_nop 0
v_permlane16_swap_b32 v74, v75 // (op 12427) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v72, v74 // (op 12427) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v73, v75 // (op 12427) 
ds_read_b128 v[76:79], v4 offset:9216 // (op 971) Load local data 

ds_read_b64 v[80:81], v4 offset:9232 // (op 971) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[18:23], v[60:65], a[16:19], v34, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 702) 
ds_read_b128 v[82:85], v4 offset:10752 // (op 1107) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[18:23], v[66:71], a[20:23], v34, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 838) 
ds_read_b64 v[86:87], v4 offset:10768 // (op 1107) Load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[18:23], v[76:81], a[24:27], v34, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 974) 
ds_read_b128 v[88:91], v3 offset:26112 // (op 1417) Load local data 
ds_read_b64 v[92:93], v3 offset:26128 // (op 1417) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[18:23], v[82:87], a[28:31], v34, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1110) 




ds_read_b128 v[18:21], v3 offset:27648 // (op 2635) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[88:93], v[24:29], a[32:35], v35, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1421) 
ds_read_b64 v[22:23], v3 offset:27664 // (op 2635) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[88:93], v[44:49], a[40:43], v35, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1475) 
ds_read_b128 v[94:97], v3 offset:29184 // (op 3853) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[88:93], v[54:59], a[44:47], v35, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1502) 
ds_read_b64 v[98:99], v3 offset:29200 // (op 3853) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[88:93], v[38:43], a[36:39], v35, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[88:93], v[60:65], a[48:51], v35, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1529) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[88:93], v[76:81], a[56:59], v35, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1583) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[88:93], v[82:87], a[60:63], v35, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1610) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[88:93], v[66:71], a[52:55], v35, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 1556) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:23], v[24:29], a[64:67], v36, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2639) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:23], v[44:49], a[72:75], v36, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2693) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[18:23], v[54:59], a[76:79], v36, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2720) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[18:23], v[38:43], a[68:71], v36, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2666) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[18:23], v[60:65], a[80:83], v36, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2747) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[18:23], v[76:81], a[88:91], v36, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2801) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[18:23], v[82:87], a[92:95], v36, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2828) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[18:23], v[66:71], a[84:87], v36, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 2774) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:99], v[24:29], a[96:99], v37, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3857) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[94:99], v[44:49], a[104:107], v37, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3911) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:99], v[54:59], a[108:111], v37, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3938) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[94:99], v[38:43], a[100:103], v37, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3884) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[94:99], v[60:65], a[112:115], v37, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3965) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:99], v[76:81], a[120:123], v37, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4019) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[94:99], v[82:87], a[124:127], v37, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 4046) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[94:99], v[66:71], a[116:119], v37, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b011 // (op 3992) 




s_ashr_i64 s[0:1], s[36:37], 7 // (op 4009) 
s_mov_b32 s7, s0 // (op 4009) convert

s_cmp_lt_u32 s6, s7 // (op 4009) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoopTail_4009 // (op 4009) Condition: Bottom (jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopTop_KLoopTail_4009 if true)
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ForLoopBottom_KLoopTail_4009:





















s_barrier  // (op 14191) 





v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalFalse_0_beta_114
v_accvgpr_read v10, a0 // (op 42) 
v_accvgpr_read v11, a1 // (op 42) 
v_accvgpr_read v12, a2 // (op 42) 
v_accvgpr_read v13, a3 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 
v_mul_f32 v16, v7, v12 // (op 42) 
v_mul_f32 v17, v7, v13 // (op 42) 



v_mov_b32 v3, v9 // (op 48) reg expression

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 48) Store value


v_accvgpr_read v10, a4 // (op 5646) 
v_accvgpr_read v11, a5 // (op 5646) 
v_accvgpr_read v12, a6 // (op 5646) 
v_accvgpr_read v13, a7 // (op 5646) 
v_mul_f32 v14, v7, v10 // (op 5646) 
v_mul_f32 v15, v7, v11 // (op 5646) 
v_mul_f32 v16, v7, v12 // (op 5646) 
v_mul_f32 v17, v7, v13 // (op 5646) 



s_lshl_b32 s0, s4, 6 // (op 5648) 
v_add_u32 v3, v9, s0 // (op 5648) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5648) Store value


v_accvgpr_read v10, a8 // (op 5658) 
v_accvgpr_read v11, a9 // (op 5658) 
v_accvgpr_read v12, a10 // (op 5658) 
v_accvgpr_read v13, a11 // (op 5658) 
v_mul_f32 v14, v7, v10 // (op 5658) 
v_mul_f32 v15, v7, v11 // (op 5658) 
v_mul_f32 v16, v7, v12 // (op 5658) 
v_mul_f32 v17, v7, v13 // (op 5658) 



s_lshl_b32 s0, s4, 7 // (op 5660) 
v_add_u32 v3, v9, s0 // (op 5660) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5660) Store value


v_accvgpr_read v10, a12 // (op 5670) 
v_accvgpr_read v11, a13 // (op 5670) 
v_accvgpr_read v12, a14 // (op 5670) 
v_accvgpr_read v13, a15 // (op 5670) 
v_mul_f32 v14, v7, v10 // (op 5670) 
v_mul_f32 v15, v7, v11 // (op 5670) 
v_mul_f32 v16, v7, v12 // (op 5670) 
v_mul_f32 v17, v7, v13 // (op 5670) 



s_lshl_b32 s0, s4, 6 // (op 5672) 
s_mul_i32 s1, 3, s0 // (op 5672) 

v_add_u32 v3, v9, s1 // (op 5672) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5672) Store value


v_accvgpr_read v10, a16 // (op 5682) 
v_accvgpr_read v11, a17 // (op 5682) 
v_accvgpr_read v12, a18 // (op 5682) 
v_accvgpr_read v13, a19 // (op 5682) 
v_mul_f32 v14, v7, v10 // (op 5682) 
v_mul_f32 v15, v7, v11 // (op 5682) 
v_mul_f32 v16, v7, v12 // (op 5682) 
v_mul_f32 v17, v7, v13 // (op 5682) 



s_lshl_b32 s0, s4, 8 // (op 5684) 
v_add_u32 v3, v9, s0 // (op 5684) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5684) Store value


v_accvgpr_read v10, a20 // (op 5694) 
v_accvgpr_read v11, a21 // (op 5694) 
v_accvgpr_read v12, a22 // (op 5694) 
v_accvgpr_read v13, a23 // (op 5694) 
v_mul_f32 v14, v7, v10 // (op 5694) 
v_mul_f32 v15, v7, v11 // (op 5694) 
v_mul_f32 v16, v7, v12 // (op 5694) 
v_mul_f32 v17, v7, v13 // (op 5694) 



s_lshl_b32 s0, s4, 6 // (op 5696) 
s_mul_i32 s1, 5, s0 // (op 5696) 

v_add_u32 v3, v9, s1 // (op 5696) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5696) Store value


v_accvgpr_read v10, a24 // (op 5706) 
v_accvgpr_read v11, a25 // (op 5706) 
v_accvgpr_read v12, a26 // (op 5706) 
v_accvgpr_read v13, a27 // (op 5706) 
v_mul_f32 v14, v7, v10 // (op 5706) 
v_mul_f32 v15, v7, v11 // (op 5706) 
v_mul_f32 v16, v7, v12 // (op 5706) 
v_mul_f32 v17, v7, v13 // (op 5706) 



s_lshl_b32 s0, s4, 6 // (op 5708) 
s_mul_i32 s1, 6, s0 // (op 5708) 

v_add_u32 v3, v9, s1 // (op 5708) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5708) Store value


v_accvgpr_read v10, a28 // (op 5718) 
v_accvgpr_read v11, a29 // (op 5718) 
v_accvgpr_read v12, a30 // (op 5718) 
v_accvgpr_read v13, a31 // (op 5718) 
v_mul_f32 v14, v7, v10 // (op 5718) 
v_mul_f32 v15, v7, v11 // (op 5718) 
v_mul_f32 v16, v7, v12 // (op 5718) 
v_mul_f32 v17, v7, v13 // (op 5718) 



s_lshl_b32 s0, s4, 6 // (op 5720) 
s_mul_i32 s1, 7, s0 // (op 5720) 

v_add_u32 v3, v9, s1 // (op 5720) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5720) Store value


v_accvgpr_read v10, a32 // (op 5772) 
v_accvgpr_read v11, a33 // (op 5772) 
v_accvgpr_read v12, a34 // (op 5772) 
v_accvgpr_read v13, a35 // (op 5772) 
v_mul_f32 v14, v7, v10 // (op 5772) 
v_mul_f32 v15, v7, v11 // (op 5772) 
v_mul_f32 v16, v7, v12 // (op 5772) 
v_mul_f32 v17, v7, v13 // (op 5772) 



v_add_u32 v3, 64, v9 // (op 5775) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5775) Store value


v_accvgpr_read v10, a36 // (op 5793) 
v_accvgpr_read v11, a37 // (op 5793) 
v_accvgpr_read v12, a38 // (op 5793) 
v_accvgpr_read v13, a39 // (op 5793) 
v_mul_f32 v14, v7, v10 // (op 5793) 
v_mul_f32 v15, v7, v11 // (op 5793) 
v_mul_f32 v16, v7, v12 // (op 5793) 
v_mul_f32 v17, v7, v13 // (op 5793) 



s_lshl_b32 s0, s4, 6 // (op 5777) 
s_add_u32 s0, s0, 64 // (op 5777) 
v_add_u32 v3, v9, s0 // (op 5777) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5777) Store value


v_accvgpr_read v10, a40 // (op 5798) 
v_accvgpr_read v11, a41 // (op 5798) 
v_accvgpr_read v12, a42 // (op 5798) 
v_accvgpr_read v13, a43 // (op 5798) 
v_mul_f32 v14, v7, v10 // (op 5798) 
v_mul_f32 v15, v7, v11 // (op 5798) 
v_mul_f32 v16, v7, v12 // (op 5798) 
v_mul_f32 v17, v7, v13 // (op 5798) 



s_lshl_b32 s0, s4, 7 // (op 5779) 
s_add_u32 s0, s0, 64 // (op 5779) 
v_add_u32 v3, v9, s0 // (op 5779) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5779) Store value


v_accvgpr_read v10, a44 // (op 5803) 
v_accvgpr_read v11, a45 // (op 5803) 
v_accvgpr_read v12, a46 // (op 5803) 
v_accvgpr_read v13, a47 // (op 5803) 
v_mul_f32 v14, v7, v10 // (op 5803) 
v_mul_f32 v15, v7, v11 // (op 5803) 
v_mul_f32 v16, v7, v12 // (op 5803) 
v_mul_f32 v17, v7, v13 // (op 5803) 



s_lshl_b32 s0, s4, 6 // (op 5781) 
s_mul_i32 s1, 3, s0 // (op 5781) 
s_add_u32 s1, s1, 64 // (op 5781) 

v_add_u32 v3, v9, s1 // (op 5781) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5781) Store value


v_accvgpr_read v10, a48 // (op 5808) 
v_accvgpr_read v11, a49 // (op 5808) 
v_accvgpr_read v12, a50 // (op 5808) 
v_accvgpr_read v13, a51 // (op 5808) 
v_mul_f32 v14, v7, v10 // (op 5808) 
v_mul_f32 v15, v7, v11 // (op 5808) 
v_mul_f32 v16, v7, v12 // (op 5808) 
v_mul_f32 v17, v7, v13 // (op 5808) 



s_lshl_b32 s0, s4, 8 // (op 5783) 
s_add_u32 s0, s0, 64 // (op 5783) 
v_add_u32 v3, v9, s0 // (op 5783) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5783) Store value


v_accvgpr_read v10, a52 // (op 5813) 
v_accvgpr_read v11, a53 // (op 5813) 
v_accvgpr_read v12, a54 // (op 5813) 
v_accvgpr_read v13, a55 // (op 5813) 
v_mul_f32 v14, v7, v10 // (op 5813) 
v_mul_f32 v15, v7, v11 // (op 5813) 
v_mul_f32 v16, v7, v12 // (op 5813) 
v_mul_f32 v17, v7, v13 // (op 5813) 



s_lshl_b32 s0, s4, 6 // (op 5785) 
s_mul_i32 s1, 5, s0 // (op 5785) 
s_add_u32 s1, s1, 64 // (op 5785) 

v_add_u32 v3, v9, s1 // (op 5785) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5785) Store value


v_accvgpr_read v10, a56 // (op 5818) 
v_accvgpr_read v11, a57 // (op 5818) 
v_accvgpr_read v12, a58 // (op 5818) 
v_accvgpr_read v13, a59 // (op 5818) 
v_mul_f32 v14, v7, v10 // (op 5818) 
v_mul_f32 v15, v7, v11 // (op 5818) 
v_mul_f32 v16, v7, v12 // (op 5818) 
v_mul_f32 v17, v7, v13 // (op 5818) 



s_lshl_b32 s0, s4, 6 // (op 5787) 
s_mul_i32 s1, 6, s0 // (op 5787) 
s_add_u32 s1, s1, 64 // (op 5787) 

v_add_u32 v3, v9, s1 // (op 5787) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5787) Store value


v_accvgpr_read v10, a60 // (op 5823) 
v_accvgpr_read v11, a61 // (op 5823) 
v_accvgpr_read v12, a62 // (op 5823) 
v_accvgpr_read v13, a63 // (op 5823) 
v_mul_f32 v14, v7, v10 // (op 5823) 
v_mul_f32 v15, v7, v11 // (op 5823) 
v_mul_f32 v16, v7, v12 // (op 5823) 
v_mul_f32 v17, v7, v13 // (op 5823) 



s_lshl_b32 s0, s4, 6 // (op 5789) 
s_mul_i32 s1, 7, s0 // (op 5789) 
s_add_u32 s1, s1, 64 // (op 5789) 

v_add_u32 v3, v9, s1 // (op 5789) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5789) Store value


v_accvgpr_read v10, a64 // (op 5904) 
v_accvgpr_read v11, a65 // (op 5904) 
v_accvgpr_read v12, a66 // (op 5904) 
v_accvgpr_read v13, a67 // (op 5904) 
v_mul_f32 v14, v7, v10 // (op 5904) 
v_mul_f32 v15, v7, v11 // (op 5904) 
v_mul_f32 v16, v7, v12 // (op 5904) 
v_mul_f32 v17, v7, v13 // (op 5904) 



v_add_u32 v3, 128, v9 // (op 5907) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5907) Store value


v_accvgpr_read v10, a68 // (op 5925) 
v_accvgpr_read v11, a69 // (op 5925) 
v_accvgpr_read v12, a70 // (op 5925) 
v_accvgpr_read v13, a71 // (op 5925) 
v_mul_f32 v14, v7, v10 // (op 5925) 
v_mul_f32 v15, v7, v11 // (op 5925) 
v_mul_f32 v16, v7, v12 // (op 5925) 
v_mul_f32 v17, v7, v13 // (op 5925) 



s_lshl_b32 s0, s4, 6 // (op 5909) 
s_add_u32 s0, s0, 128 // (op 5909) 
v_add_u32 v3, v9, s0 // (op 5909) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5909) Store value


v_accvgpr_read v10, a72 // (op 5930) 
v_accvgpr_read v11, a73 // (op 5930) 
v_accvgpr_read v12, a74 // (op 5930) 
v_accvgpr_read v13, a75 // (op 5930) 
v_mul_f32 v14, v7, v10 // (op 5930) 
v_mul_f32 v15, v7, v11 // (op 5930) 
v_mul_f32 v16, v7, v12 // (op 5930) 
v_mul_f32 v17, v7, v13 // (op 5930) 



s_lshl_b32 s0, s4, 7 // (op 5911) 
s_add_u32 s0, s0, 128 // (op 5911) 
v_add_u32 v3, v9, s0 // (op 5911) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5911) Store value


v_accvgpr_read v10, a76 // (op 5935) 
v_accvgpr_read v11, a77 // (op 5935) 
v_accvgpr_read v12, a78 // (op 5935) 
v_accvgpr_read v13, a79 // (op 5935) 
v_mul_f32 v14, v7, v10 // (op 5935) 
v_mul_f32 v15, v7, v11 // (op 5935) 
v_mul_f32 v16, v7, v12 // (op 5935) 
v_mul_f32 v17, v7, v13 // (op 5935) 



s_lshl_b32 s0, s4, 6 // (op 5913) 
s_mul_i32 s1, 3, s0 // (op 5913) 
s_add_u32 s1, s1, 128 // (op 5913) 

v_add_u32 v3, v9, s1 // (op 5913) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5913) Store value


v_accvgpr_read v10, a80 // (op 5940) 
v_accvgpr_read v11, a81 // (op 5940) 
v_accvgpr_read v12, a82 // (op 5940) 
v_accvgpr_read v13, a83 // (op 5940) 
v_mul_f32 v14, v7, v10 // (op 5940) 
v_mul_f32 v15, v7, v11 // (op 5940) 
v_mul_f32 v16, v7, v12 // (op 5940) 
v_mul_f32 v17, v7, v13 // (op 5940) 



s_lshl_b32 s0, s4, 8 // (op 5915) 
s_add_u32 s0, s0, 128 // (op 5915) 
v_add_u32 v3, v9, s0 // (op 5915) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5915) Store value


v_accvgpr_read v10, a84 // (op 5945) 
v_accvgpr_read v11, a85 // (op 5945) 
v_accvgpr_read v12, a86 // (op 5945) 
v_accvgpr_read v13, a87 // (op 5945) 
v_mul_f32 v14, v7, v10 // (op 5945) 
v_mul_f32 v15, v7, v11 // (op 5945) 
v_mul_f32 v16, v7, v12 // (op 5945) 
v_mul_f32 v17, v7, v13 // (op 5945) 



s_lshl_b32 s0, s4, 6 // (op 5917) 
s_mul_i32 s1, 5, s0 // (op 5917) 
s_add_u32 s1, s1, 128 // (op 5917) 

v_add_u32 v3, v9, s1 // (op 5917) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5917) Store value


v_accvgpr_read v10, a88 // (op 5950) 
v_accvgpr_read v11, a89 // (op 5950) 
v_accvgpr_read v12, a90 // (op 5950) 
v_accvgpr_read v13, a91 // (op 5950) 
v_mul_f32 v14, v7, v10 // (op 5950) 
v_mul_f32 v15, v7, v11 // (op 5950) 
v_mul_f32 v16, v7, v12 // (op 5950) 
v_mul_f32 v17, v7, v13 // (op 5950) 



s_lshl_b32 s0, s4, 6 // (op 5919) 
s_mul_i32 s1, 6, s0 // (op 5919) 
s_add_u32 s1, s1, 128 // (op 5919) 

v_add_u32 v3, v9, s1 // (op 5919) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5919) Store value


v_accvgpr_read v10, a92 // (op 5955) 
v_accvgpr_read v11, a93 // (op 5955) 
v_accvgpr_read v12, a94 // (op 5955) 
v_accvgpr_read v13, a95 // (op 5955) 
v_mul_f32 v14, v7, v10 // (op 5955) 
v_mul_f32 v15, v7, v11 // (op 5955) 
v_mul_f32 v16, v7, v12 // (op 5955) 
v_mul_f32 v17, v7, v13 // (op 5955) 



s_lshl_b32 s0, s4, 6 // (op 5921) 
s_mul_i32 s1, 7, s0 // (op 5921) 
s_add_u32 s1, s1, 128 // (op 5921) 

v_add_u32 v3, v9, s1 // (op 5921) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5921) Store value


v_accvgpr_read v10, a96 // (op 6036) 
v_accvgpr_read v11, a97 // (op 6036) 
v_accvgpr_read v12, a98 // (op 6036) 
v_accvgpr_read v13, a99 // (op 6036) 
v_mul_f32 v14, v7, v10 // (op 6036) 
v_mul_f32 v15, v7, v11 // (op 6036) 
v_mul_f32 v16, v7, v12 // (op 6036) 
v_mul_f32 v17, v7, v13 // (op 6036) 



v_add_u32 v3, 192, v9 // (op 6039) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6039) Store value


v_accvgpr_read v10, a100 // (op 6057) 
v_accvgpr_read v11, a101 // (op 6057) 
v_accvgpr_read v12, a102 // (op 6057) 
v_accvgpr_read v13, a103 // (op 6057) 
v_mul_f32 v14, v7, v10 // (op 6057) 
v_mul_f32 v15, v7, v11 // (op 6057) 
v_mul_f32 v16, v7, v12 // (op 6057) 
v_mul_f32 v17, v7, v13 // (op 6057) 



s_lshl_b32 s0, s4, 6 // (op 6041) 
s_add_u32 s0, s0, 192 // (op 6041) 
v_add_u32 v3, v9, s0 // (op 6041) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6041) Store value


v_accvgpr_read v10, a104 // (op 6062) 
v_accvgpr_read v11, a105 // (op 6062) 
v_accvgpr_read v12, a106 // (op 6062) 
v_accvgpr_read v13, a107 // (op 6062) 
v_mul_f32 v14, v7, v10 // (op 6062) 
v_mul_f32 v15, v7, v11 // (op 6062) 
v_mul_f32 v16, v7, v12 // (op 6062) 
v_mul_f32 v17, v7, v13 // (op 6062) 



s_lshl_b32 s0, s4, 7 // (op 6043) 
s_add_u32 s0, s0, 192 // (op 6043) 
v_add_u32 v3, v9, s0 // (op 6043) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6043) Store value


v_accvgpr_read v10, a108 // (op 6067) 
v_accvgpr_read v11, a109 // (op 6067) 
v_accvgpr_read v12, a110 // (op 6067) 
v_accvgpr_read v13, a111 // (op 6067) 
v_mul_f32 v14, v7, v10 // (op 6067) 
v_mul_f32 v15, v7, v11 // (op 6067) 
v_mul_f32 v16, v7, v12 // (op 6067) 
v_mul_f32 v17, v7, v13 // (op 6067) 



s_lshl_b32 s0, s4, 6 // (op 6045) 
s_mul_i32 s1, 3, s0 // (op 6045) 
s_add_u32 s1, s1, 192 // (op 6045) 

v_add_u32 v3, v9, s1 // (op 6045) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6045) Store value


v_accvgpr_read v10, a112 // (op 6072) 
v_accvgpr_read v11, a113 // (op 6072) 
v_accvgpr_read v12, a114 // (op 6072) 
v_accvgpr_read v13, a115 // (op 6072) 
v_mul_f32 v14, v7, v10 // (op 6072) 
v_mul_f32 v15, v7, v11 // (op 6072) 
v_mul_f32 v16, v7, v12 // (op 6072) 
v_mul_f32 v17, v7, v13 // (op 6072) 



s_lshl_b32 s0, s4, 8 // (op 6047) 
s_add_u32 s0, s0, 192 // (op 6047) 
v_add_u32 v3, v9, s0 // (op 6047) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6047) Store value


v_accvgpr_read v10, a116 // (op 6077) 
v_accvgpr_read v11, a117 // (op 6077) 
v_accvgpr_read v12, a118 // (op 6077) 
v_accvgpr_read v13, a119 // (op 6077) 
v_mul_f32 v14, v7, v10 // (op 6077) 
v_mul_f32 v15, v7, v11 // (op 6077) 
v_mul_f32 v16, v7, v12 // (op 6077) 
v_mul_f32 v17, v7, v13 // (op 6077) 



s_lshl_b32 s0, s4, 6 // (op 6049) 
s_mul_i32 s1, 5, s0 // (op 6049) 
s_add_u32 s1, s1, 192 // (op 6049) 

v_add_u32 v3, v9, s1 // (op 6049) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6049) Store value


v_accvgpr_read v10, a120 // (op 6082) 
v_accvgpr_read v11, a121 // (op 6082) 
v_accvgpr_read v12, a122 // (op 6082) 
v_accvgpr_read v13, a123 // (op 6082) 
v_mul_f32 v14, v7, v10 // (op 6082) 
v_mul_f32 v15, v7, v11 // (op 6082) 
v_mul_f32 v16, v7, v12 // (op 6082) 
v_mul_f32 v17, v7, v13 // (op 6082) 



s_lshl_b32 s0, s4, 6 // (op 6051) 
s_mul_i32 s1, 6, s0 // (op 6051) 
s_add_u32 s1, s1, 192 // (op 6051) 

v_add_u32 v3, v9, s1 // (op 6051) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6051) Store value


v_accvgpr_read v10, a124 // (op 6087) 
v_accvgpr_read v11, a125 // (op 6087) 
v_accvgpr_read v12, a126 // (op 6087) 
v_accvgpr_read v13, a127 // (op 6087) 
v_mul_f32 v14, v7, v10 // (op 6087) 
v_mul_f32 v15, v7, v11 // (op 6087) 
v_mul_f32 v16, v7, v12 // (op 6087) 
v_mul_f32 v17, v7, v13 // (op 6087) 



s_lshl_b32 s0, s4, 6 // (op 6053) 
s_mul_i32 s1, 7, s0 // (op 6053) 
s_add_u32 s1, s1, 192 // (op 6053) 

v_add_u32 v3, v9, s1 // (op 6053) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6053) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalFalse_0_beta_114:


v_add_u32 v3, 192, v8 // (op 6763) 

buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6763) Load value

v_accvgpr_read v14, a96 // (op 6743) 
v_accvgpr_read v15, a97 // (op 6743) 
v_accvgpr_read v16, a98 // (op 6743) 
v_accvgpr_read v17, a99 // (op 6743) 
v_mul_f32 v18, v7, v14 // (op 6743) 
v_mul_f32 v19, v7, v15 // (op 6743) 
v_mul_f32 v20, v7, v16 // (op 6743) 
v_mul_f32 v21, v7, v17 // (op 6743) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6762) 
v_mul_f32 v15, v0, v11 // (op 6762) 
v_mul_f32 v16, v0, v12 // (op 6762) 
v_mul_f32 v17, v0, v13 // (op 6762) 

v_add_f32 v10, v14, v18 // (op 6744) 
v_add_f32 v11, v15, v19 // (op 6744) 
v_add_f32 v12, v16, v20 // (op 6744) 
v_add_f32 v13, v17, v21 // (op 6744) 




v_add_u32 v3, 192, v9 // (op 6746) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6746) Store value




s_lshl_b32 s0, s2, 6 // (op 6767) 
s_add_u32 s0, s0, 192 // (op 6767) 
v_add_u32 v3, v8, s0 // (op 6767) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6767) Load value

v_accvgpr_read v14, a100 // (op 6793) 
v_accvgpr_read v15, a101 // (op 6793) 
v_accvgpr_read v16, a102 // (op 6793) 
v_accvgpr_read v17, a103 // (op 6793) 
v_mul_f32 v18, v7, v14 // (op 6793) 
v_mul_f32 v19, v7, v15 // (op 6793) 
v_mul_f32 v20, v7, v16 // (op 6793) 
v_mul_f32 v21, v7, v17 // (op 6793) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6765) 
v_mul_f32 v15, v0, v11 // (op 6765) 
v_mul_f32 v16, v0, v12 // (op 6765) 
v_mul_f32 v17, v0, v13 // (op 6765) 

v_add_f32 v10, v14, v18 // (op 6766) 
v_add_f32 v11, v15, v19 // (op 6766) 
v_add_f32 v12, v16, v20 // (op 6766) 
v_add_f32 v13, v17, v21 // (op 6766) 




s_lshl_b32 s0, s4, 6 // (op 6748) 
s_add_u32 s0, s0, 192 // (op 6748) 
v_add_u32 v3, v9, s0 // (op 6748) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6748) Store value




s_lshl_b32 s0, s2, 7 // (op 6771) 
s_add_u32 s0, s0, 192 // (op 6771) 
v_add_u32 v3, v8, s0 // (op 6771) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6771) Load value

v_accvgpr_read v14, a104 // (op 6795) 
v_accvgpr_read v15, a105 // (op 6795) 
v_accvgpr_read v16, a106 // (op 6795) 
v_accvgpr_read v17, a107 // (op 6795) 
v_mul_f32 v18, v7, v14 // (op 6795) 
v_mul_f32 v19, v7, v15 // (op 6795) 
v_mul_f32 v20, v7, v16 // (op 6795) 
v_mul_f32 v21, v7, v17 // (op 6795) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6769) 
v_mul_f32 v15, v0, v11 // (op 6769) 
v_mul_f32 v16, v0, v12 // (op 6769) 
v_mul_f32 v17, v0, v13 // (op 6769) 

v_add_f32 v10, v14, v18 // (op 6770) 
v_add_f32 v11, v15, v19 // (op 6770) 
v_add_f32 v12, v16, v20 // (op 6770) 
v_add_f32 v13, v17, v21 // (op 6770) 




s_lshl_b32 s0, s4, 7 // (op 6750) 
s_add_u32 s0, s0, 192 // (op 6750) 
v_add_u32 v3, v9, s0 // (op 6750) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6750) Store value




s_lshl_b32 s0, s2, 6 // (op 6775) 
s_mul_i32 s1, 3, s0 // (op 6775) 
s_add_u32 s1, s1, 192 // (op 6775) 

v_add_u32 v3, v8, s1 // (op 6775) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6775) Load value

v_accvgpr_read v14, a108 // (op 6797) 
v_accvgpr_read v15, a109 // (op 6797) 
v_accvgpr_read v16, a110 // (op 6797) 
v_accvgpr_read v17, a111 // (op 6797) 
v_mul_f32 v18, v7, v14 // (op 6797) 
v_mul_f32 v19, v7, v15 // (op 6797) 
v_mul_f32 v20, v7, v16 // (op 6797) 
v_mul_f32 v21, v7, v17 // (op 6797) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6773) 
v_mul_f32 v15, v0, v11 // (op 6773) 
v_mul_f32 v16, v0, v12 // (op 6773) 
v_mul_f32 v17, v0, v13 // (op 6773) 

v_add_f32 v10, v14, v18 // (op 6774) 
v_add_f32 v11, v15, v19 // (op 6774) 
v_add_f32 v12, v16, v20 // (op 6774) 
v_add_f32 v13, v17, v21 // (op 6774) 




s_lshl_b32 s0, s4, 6 // (op 6752) 
s_mul_i32 s1, 3, s0 // (op 6752) 
s_add_u32 s1, s1, 192 // (op 6752) 

v_add_u32 v3, v9, s1 // (op 6752) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6752) Store value




s_lshl_b32 s0, s2, 8 // (op 6779) 
s_add_u32 s0, s0, 192 // (op 6779) 
v_add_u32 v3, v8, s0 // (op 6779) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6779) Load value

v_accvgpr_read v14, a112 // (op 6799) 
v_accvgpr_read v15, a113 // (op 6799) 
v_accvgpr_read v16, a114 // (op 6799) 
v_accvgpr_read v17, a115 // (op 6799) 
v_mul_f32 v18, v7, v14 // (op 6799) 
v_mul_f32 v19, v7, v15 // (op 6799) 
v_mul_f32 v20, v7, v16 // (op 6799) 
v_mul_f32 v21, v7, v17 // (op 6799) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6777) 
v_mul_f32 v15, v0, v11 // (op 6777) 
v_mul_f32 v16, v0, v12 // (op 6777) 
v_mul_f32 v17, v0, v13 // (op 6777) 

v_add_f32 v10, v14, v18 // (op 6778) 
v_add_f32 v11, v15, v19 // (op 6778) 
v_add_f32 v12, v16, v20 // (op 6778) 
v_add_f32 v13, v17, v21 // (op 6778) 




s_lshl_b32 s0, s4, 8 // (op 6754) 
s_add_u32 s0, s0, 192 // (op 6754) 
v_add_u32 v3, v9, s0 // (op 6754) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6754) Store value




s_lshl_b32 s0, s2, 6 // (op 6783) 
s_mul_i32 s1, 5, s0 // (op 6783) 
s_add_u32 s1, s1, 192 // (op 6783) 

v_add_u32 v3, v8, s1 // (op 6783) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6783) Load value

v_accvgpr_read v14, a116 // (op 6801) 
v_accvgpr_read v15, a117 // (op 6801) 
v_accvgpr_read v16, a118 // (op 6801) 
v_accvgpr_read v17, a119 // (op 6801) 
v_mul_f32 v18, v7, v14 // (op 6801) 
v_mul_f32 v19, v7, v15 // (op 6801) 
v_mul_f32 v20, v7, v16 // (op 6801) 
v_mul_f32 v21, v7, v17 // (op 6801) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6781) 
v_mul_f32 v15, v0, v11 // (op 6781) 
v_mul_f32 v16, v0, v12 // (op 6781) 
v_mul_f32 v17, v0, v13 // (op 6781) 

v_add_f32 v10, v14, v18 // (op 6782) 
v_add_f32 v11, v15, v19 // (op 6782) 
v_add_f32 v12, v16, v20 // (op 6782) 
v_add_f32 v13, v17, v21 // (op 6782) 




s_lshl_b32 s0, s4, 6 // (op 6756) 
s_mul_i32 s1, 5, s0 // (op 6756) 
s_add_u32 s1, s1, 192 // (op 6756) 

v_add_u32 v3, v9, s1 // (op 6756) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6756) Store value




s_lshl_b32 s0, s2, 6 // (op 6787) 
s_mul_i32 s1, 6, s0 // (op 6787) 
s_add_u32 s1, s1, 192 // (op 6787) 

v_add_u32 v3, v8, s1 // (op 6787) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6787) Load value

v_accvgpr_read v14, a120 // (op 6803) 
v_accvgpr_read v15, a121 // (op 6803) 
v_accvgpr_read v16, a122 // (op 6803) 
v_accvgpr_read v17, a123 // (op 6803) 
v_mul_f32 v18, v7, v14 // (op 6803) 
v_mul_f32 v19, v7, v15 // (op 6803) 
v_mul_f32 v20, v7, v16 // (op 6803) 
v_mul_f32 v21, v7, v17 // (op 6803) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6785) 
v_mul_f32 v15, v0, v11 // (op 6785) 
v_mul_f32 v16, v0, v12 // (op 6785) 
v_mul_f32 v17, v0, v13 // (op 6785) 

v_add_f32 v10, v14, v18 // (op 6786) 
v_add_f32 v11, v15, v19 // (op 6786) 
v_add_f32 v12, v16, v20 // (op 6786) 
v_add_f32 v13, v17, v21 // (op 6786) 




s_lshl_b32 s0, s4, 6 // (op 6758) 
s_mul_i32 s1, 6, s0 // (op 6758) 
s_add_u32 s1, s1, 192 // (op 6758) 

v_add_u32 v3, v9, s1 // (op 6758) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6758) Store value




s_lshl_b32 s0, s2, 6 // (op 6791) 
s_mul_i32 s1, 7, s0 // (op 6791) 
s_add_u32 s1, s1, 192 // (op 6791) 

v_add_u32 v3, v8, s1 // (op 6791) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6791) Load value

v_accvgpr_read v14, a124 // (op 6805) 
v_accvgpr_read v15, a125 // (op 6805) 
v_accvgpr_read v16, a126 // (op 6805) 
v_accvgpr_read v17, a127 // (op 6805) 
v_mul_f32 v18, v7, v14 // (op 6805) 
v_mul_f32 v19, v7, v15 // (op 6805) 
v_mul_f32 v20, v7, v16 // (op 6805) 
v_mul_f32 v21, v7, v17 // (op 6805) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6789) 
v_mul_f32 v15, v0, v11 // (op 6789) 
v_mul_f32 v16, v0, v12 // (op 6789) 
v_mul_f32 v17, v0, v13 // (op 6789) 

v_add_f32 v10, v14, v18 // (op 6790) 
v_add_f32 v11, v15, v19 // (op 6790) 
v_add_f32 v12, v16, v20 // (op 6790) 
v_add_f32 v13, v17, v21 // (op 6790) 




s_lshl_b32 s0, s4, 6 // (op 6760) 
s_mul_i32 s1, 7, s0 // (op 6760) 
s_add_u32 s1, s1, 192 // (op 6760) 

v_add_u32 v3, v9, s1 // (op 6760) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6760) Store value




v_add_u32 v3, 128, v8 // (op 6608) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6608) Load value

v_accvgpr_read v14, a64 // (op 6588) 
v_accvgpr_read v15, a65 // (op 6588) 
v_accvgpr_read v16, a66 // (op 6588) 
v_accvgpr_read v17, a67 // (op 6588) 
v_mul_f32 v18, v7, v14 // (op 6588) 
v_mul_f32 v19, v7, v15 // (op 6588) 
v_mul_f32 v20, v7, v16 // (op 6588) 
v_mul_f32 v21, v7, v17 // (op 6588) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6607) 
v_mul_f32 v15, v0, v11 // (op 6607) 
v_mul_f32 v16, v0, v12 // (op 6607) 
v_mul_f32 v17, v0, v13 // (op 6607) 

v_add_f32 v10, v14, v18 // (op 6589) 
v_add_f32 v11, v15, v19 // (op 6589) 
v_add_f32 v12, v16, v20 // (op 6589) 
v_add_f32 v13, v17, v21 // (op 6589) 




v_add_u32 v3, 128, v9 // (op 6591) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6591) Store value




s_lshl_b32 s0, s2, 6 // (op 6612) 
s_add_u32 s0, s0, 128 // (op 6612) 
v_add_u32 v3, v8, s0 // (op 6612) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6612) Load value

v_accvgpr_read v14, a68 // (op 6638) 
v_accvgpr_read v15, a69 // (op 6638) 
v_accvgpr_read v16, a70 // (op 6638) 
v_accvgpr_read v17, a71 // (op 6638) 
v_mul_f32 v18, v7, v14 // (op 6638) 
v_mul_f32 v19, v7, v15 // (op 6638) 
v_mul_f32 v20, v7, v16 // (op 6638) 
v_mul_f32 v21, v7, v17 // (op 6638) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6610) 
v_mul_f32 v15, v0, v11 // (op 6610) 
v_mul_f32 v16, v0, v12 // (op 6610) 
v_mul_f32 v17, v0, v13 // (op 6610) 

v_add_f32 v10, v14, v18 // (op 6611) 
v_add_f32 v11, v15, v19 // (op 6611) 
v_add_f32 v12, v16, v20 // (op 6611) 
v_add_f32 v13, v17, v21 // (op 6611) 




s_lshl_b32 s0, s4, 6 // (op 6593) 
s_add_u32 s0, s0, 128 // (op 6593) 
v_add_u32 v3, v9, s0 // (op 6593) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6593) Store value




s_lshl_b32 s0, s2, 7 // (op 6616) 
s_add_u32 s0, s0, 128 // (op 6616) 
v_add_u32 v3, v8, s0 // (op 6616) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6616) Load value

v_accvgpr_read v14, a72 // (op 6640) 
v_accvgpr_read v15, a73 // (op 6640) 
v_accvgpr_read v16, a74 // (op 6640) 
v_accvgpr_read v17, a75 // (op 6640) 
v_mul_f32 v18, v7, v14 // (op 6640) 
v_mul_f32 v19, v7, v15 // (op 6640) 
v_mul_f32 v20, v7, v16 // (op 6640) 
v_mul_f32 v21, v7, v17 // (op 6640) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6614) 
v_mul_f32 v15, v0, v11 // (op 6614) 
v_mul_f32 v16, v0, v12 // (op 6614) 
v_mul_f32 v17, v0, v13 // (op 6614) 

v_add_f32 v10, v14, v18 // (op 6615) 
v_add_f32 v11, v15, v19 // (op 6615) 
v_add_f32 v12, v16, v20 // (op 6615) 
v_add_f32 v13, v17, v21 // (op 6615) 




s_lshl_b32 s0, s4, 7 // (op 6595) 
s_add_u32 s0, s0, 128 // (op 6595) 
v_add_u32 v3, v9, s0 // (op 6595) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6595) Store value




s_lshl_b32 s0, s2, 6 // (op 6620) 
s_mul_i32 s1, 3, s0 // (op 6620) 
s_add_u32 s1, s1, 128 // (op 6620) 

v_add_u32 v3, v8, s1 // (op 6620) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6620) Load value

v_accvgpr_read v14, a76 // (op 6642) 
v_accvgpr_read v15, a77 // (op 6642) 
v_accvgpr_read v16, a78 // (op 6642) 
v_accvgpr_read v17, a79 // (op 6642) 
v_mul_f32 v18, v7, v14 // (op 6642) 
v_mul_f32 v19, v7, v15 // (op 6642) 
v_mul_f32 v20, v7, v16 // (op 6642) 
v_mul_f32 v21, v7, v17 // (op 6642) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6618) 
v_mul_f32 v15, v0, v11 // (op 6618) 
v_mul_f32 v16, v0, v12 // (op 6618) 
v_mul_f32 v17, v0, v13 // (op 6618) 

v_add_f32 v10, v14, v18 // (op 6619) 
v_add_f32 v11, v15, v19 // (op 6619) 
v_add_f32 v12, v16, v20 // (op 6619) 
v_add_f32 v13, v17, v21 // (op 6619) 




s_lshl_b32 s0, s4, 6 // (op 6597) 
s_mul_i32 s1, 3, s0 // (op 6597) 
s_add_u32 s1, s1, 128 // (op 6597) 

v_add_u32 v3, v9, s1 // (op 6597) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6597) Store value




s_lshl_b32 s0, s2, 8 // (op 6624) 
s_add_u32 s0, s0, 128 // (op 6624) 
v_add_u32 v3, v8, s0 // (op 6624) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6624) Load value

v_accvgpr_read v14, a80 // (op 6644) 
v_accvgpr_read v15, a81 // (op 6644) 
v_accvgpr_read v16, a82 // (op 6644) 
v_accvgpr_read v17, a83 // (op 6644) 
v_mul_f32 v18, v7, v14 // (op 6644) 
v_mul_f32 v19, v7, v15 // (op 6644) 
v_mul_f32 v20, v7, v16 // (op 6644) 
v_mul_f32 v21, v7, v17 // (op 6644) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6622) 
v_mul_f32 v15, v0, v11 // (op 6622) 
v_mul_f32 v16, v0, v12 // (op 6622) 
v_mul_f32 v17, v0, v13 // (op 6622) 

v_add_f32 v10, v14, v18 // (op 6623) 
v_add_f32 v11, v15, v19 // (op 6623) 
v_add_f32 v12, v16, v20 // (op 6623) 
v_add_f32 v13, v17, v21 // (op 6623) 




s_lshl_b32 s0, s4, 8 // (op 6599) 
s_add_u32 s0, s0, 128 // (op 6599) 
v_add_u32 v3, v9, s0 // (op 6599) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6599) Store value




s_lshl_b32 s0, s2, 6 // (op 6628) 
s_mul_i32 s1, 5, s0 // (op 6628) 
s_add_u32 s1, s1, 128 // (op 6628) 

v_add_u32 v3, v8, s1 // (op 6628) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6628) Load value

v_accvgpr_read v14, a84 // (op 6646) 
v_accvgpr_read v15, a85 // (op 6646) 
v_accvgpr_read v16, a86 // (op 6646) 
v_accvgpr_read v17, a87 // (op 6646) 
v_mul_f32 v18, v7, v14 // (op 6646) 
v_mul_f32 v19, v7, v15 // (op 6646) 
v_mul_f32 v20, v7, v16 // (op 6646) 
v_mul_f32 v21, v7, v17 // (op 6646) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6626) 
v_mul_f32 v15, v0, v11 // (op 6626) 
v_mul_f32 v16, v0, v12 // (op 6626) 
v_mul_f32 v17, v0, v13 // (op 6626) 

v_add_f32 v10, v14, v18 // (op 6627) 
v_add_f32 v11, v15, v19 // (op 6627) 
v_add_f32 v12, v16, v20 // (op 6627) 
v_add_f32 v13, v17, v21 // (op 6627) 




s_lshl_b32 s0, s4, 6 // (op 6601) 
s_mul_i32 s1, 5, s0 // (op 6601) 
s_add_u32 s1, s1, 128 // (op 6601) 

v_add_u32 v3, v9, s1 // (op 6601) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6601) Store value




s_lshl_b32 s0, s2, 6 // (op 6632) 
s_mul_i32 s1, 6, s0 // (op 6632) 
s_add_u32 s1, s1, 128 // (op 6632) 

v_add_u32 v3, v8, s1 // (op 6632) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6632) Load value

v_accvgpr_read v14, a88 // (op 6648) 
v_accvgpr_read v15, a89 // (op 6648) 
v_accvgpr_read v16, a90 // (op 6648) 
v_accvgpr_read v17, a91 // (op 6648) 
v_mul_f32 v18, v7, v14 // (op 6648) 
v_mul_f32 v19, v7, v15 // (op 6648) 
v_mul_f32 v20, v7, v16 // (op 6648) 
v_mul_f32 v21, v7, v17 // (op 6648) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6630) 
v_mul_f32 v15, v0, v11 // (op 6630) 
v_mul_f32 v16, v0, v12 // (op 6630) 
v_mul_f32 v17, v0, v13 // (op 6630) 

v_add_f32 v10, v14, v18 // (op 6631) 
v_add_f32 v11, v15, v19 // (op 6631) 
v_add_f32 v12, v16, v20 // (op 6631) 
v_add_f32 v13, v17, v21 // (op 6631) 




s_lshl_b32 s0, s4, 6 // (op 6603) 
s_mul_i32 s1, 6, s0 // (op 6603) 
s_add_u32 s1, s1, 128 // (op 6603) 

v_add_u32 v3, v9, s1 // (op 6603) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6603) Store value




s_lshl_b32 s0, s2, 6 // (op 6636) 
s_mul_i32 s1, 7, s0 // (op 6636) 
s_add_u32 s1, s1, 128 // (op 6636) 

v_add_u32 v3, v8, s1 // (op 6636) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6636) Load value

v_accvgpr_read v14, a92 // (op 6650) 
v_accvgpr_read v15, a93 // (op 6650) 
v_accvgpr_read v16, a94 // (op 6650) 
v_accvgpr_read v17, a95 // (op 6650) 
v_mul_f32 v18, v7, v14 // (op 6650) 
v_mul_f32 v19, v7, v15 // (op 6650) 
v_mul_f32 v20, v7, v16 // (op 6650) 
v_mul_f32 v21, v7, v17 // (op 6650) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6634) 
v_mul_f32 v15, v0, v11 // (op 6634) 
v_mul_f32 v16, v0, v12 // (op 6634) 
v_mul_f32 v17, v0, v13 // (op 6634) 

v_add_f32 v10, v14, v18 // (op 6635) 
v_add_f32 v11, v15, v19 // (op 6635) 
v_add_f32 v12, v16, v20 // (op 6635) 
v_add_f32 v13, v17, v21 // (op 6635) 




s_lshl_b32 s0, s4, 6 // (op 6605) 
s_mul_i32 s1, 7, s0 // (op 6605) 
s_add_u32 s1, s1, 128 // (op 6605) 

v_add_u32 v3, v9, s1 // (op 6605) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6605) Store value




v_add_u32 v3, 64, v8 // (op 6453) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6453) Load value

v_accvgpr_read v14, a32 // (op 6433) 
v_accvgpr_read v15, a33 // (op 6433) 
v_accvgpr_read v16, a34 // (op 6433) 
v_accvgpr_read v17, a35 // (op 6433) 
v_mul_f32 v18, v7, v14 // (op 6433) 
v_mul_f32 v19, v7, v15 // (op 6433) 
v_mul_f32 v20, v7, v16 // (op 6433) 
v_mul_f32 v21, v7, v17 // (op 6433) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6452) 
v_mul_f32 v15, v0, v11 // (op 6452) 
v_mul_f32 v16, v0, v12 // (op 6452) 
v_mul_f32 v17, v0, v13 // (op 6452) 

v_add_f32 v10, v14, v18 // (op 6434) 
v_add_f32 v11, v15, v19 // (op 6434) 
v_add_f32 v12, v16, v20 // (op 6434) 
v_add_f32 v13, v17, v21 // (op 6434) 




v_add_u32 v3, 64, v9 // (op 6436) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6436) Store value




s_lshl_b32 s0, s2, 6 // (op 6457) 
s_add_u32 s0, s0, 64 // (op 6457) 
v_add_u32 v3, v8, s0 // (op 6457) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6457) Load value

v_accvgpr_read v14, a36 // (op 6483) 
v_accvgpr_read v15, a37 // (op 6483) 
v_accvgpr_read v16, a38 // (op 6483) 
v_accvgpr_read v17, a39 // (op 6483) 
v_mul_f32 v18, v7, v14 // (op 6483) 
v_mul_f32 v19, v7, v15 // (op 6483) 
v_mul_f32 v20, v7, v16 // (op 6483) 
v_mul_f32 v21, v7, v17 // (op 6483) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6455) 
v_mul_f32 v15, v0, v11 // (op 6455) 
v_mul_f32 v16, v0, v12 // (op 6455) 
v_mul_f32 v17, v0, v13 // (op 6455) 

v_add_f32 v10, v14, v18 // (op 6456) 
v_add_f32 v11, v15, v19 // (op 6456) 
v_add_f32 v12, v16, v20 // (op 6456) 
v_add_f32 v13, v17, v21 // (op 6456) 




s_lshl_b32 s0, s4, 6 // (op 6438) 
s_add_u32 s0, s0, 64 // (op 6438) 
v_add_u32 v3, v9, s0 // (op 6438) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6438) Store value




s_lshl_b32 s0, s2, 7 // (op 6461) 
s_add_u32 s0, s0, 64 // (op 6461) 
v_add_u32 v3, v8, s0 // (op 6461) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6461) Load value

v_accvgpr_read v14, a40 // (op 6485) 
v_accvgpr_read v15, a41 // (op 6485) 
v_accvgpr_read v16, a42 // (op 6485) 
v_accvgpr_read v17, a43 // (op 6485) 
v_mul_f32 v18, v7, v14 // (op 6485) 
v_mul_f32 v19, v7, v15 // (op 6485) 
v_mul_f32 v20, v7, v16 // (op 6485) 
v_mul_f32 v21, v7, v17 // (op 6485) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6459) 
v_mul_f32 v15, v0, v11 // (op 6459) 
v_mul_f32 v16, v0, v12 // (op 6459) 
v_mul_f32 v17, v0, v13 // (op 6459) 

v_add_f32 v10, v14, v18 // (op 6460) 
v_add_f32 v11, v15, v19 // (op 6460) 
v_add_f32 v12, v16, v20 // (op 6460) 
v_add_f32 v13, v17, v21 // (op 6460) 




s_lshl_b32 s0, s4, 7 // (op 6440) 
s_add_u32 s0, s0, 64 // (op 6440) 
v_add_u32 v3, v9, s0 // (op 6440) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6440) Store value




s_lshl_b32 s0, s2, 6 // (op 6465) 
s_mul_i32 s1, 3, s0 // (op 6465) 
s_add_u32 s1, s1, 64 // (op 6465) 

v_add_u32 v3, v8, s1 // (op 6465) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6465) Load value

v_accvgpr_read v14, a44 // (op 6487) 
v_accvgpr_read v15, a45 // (op 6487) 
v_accvgpr_read v16, a46 // (op 6487) 
v_accvgpr_read v17, a47 // (op 6487) 
v_mul_f32 v18, v7, v14 // (op 6487) 
v_mul_f32 v19, v7, v15 // (op 6487) 
v_mul_f32 v20, v7, v16 // (op 6487) 
v_mul_f32 v21, v7, v17 // (op 6487) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6463) 
v_mul_f32 v15, v0, v11 // (op 6463) 
v_mul_f32 v16, v0, v12 // (op 6463) 
v_mul_f32 v17, v0, v13 // (op 6463) 

v_add_f32 v10, v14, v18 // (op 6464) 
v_add_f32 v11, v15, v19 // (op 6464) 
v_add_f32 v12, v16, v20 // (op 6464) 
v_add_f32 v13, v17, v21 // (op 6464) 




s_lshl_b32 s0, s4, 6 // (op 6442) 
s_mul_i32 s1, 3, s0 // (op 6442) 
s_add_u32 s1, s1, 64 // (op 6442) 

v_add_u32 v3, v9, s1 // (op 6442) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6442) Store value




s_lshl_b32 s0, s2, 8 // (op 6469) 
s_add_u32 s0, s0, 64 // (op 6469) 
v_add_u32 v3, v8, s0 // (op 6469) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6469) Load value

v_accvgpr_read v14, a48 // (op 6489) 
v_accvgpr_read v15, a49 // (op 6489) 
v_accvgpr_read v16, a50 // (op 6489) 
v_accvgpr_read v17, a51 // (op 6489) 
v_mul_f32 v18, v7, v14 // (op 6489) 
v_mul_f32 v19, v7, v15 // (op 6489) 
v_mul_f32 v20, v7, v16 // (op 6489) 
v_mul_f32 v21, v7, v17 // (op 6489) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6467) 
v_mul_f32 v15, v0, v11 // (op 6467) 
v_mul_f32 v16, v0, v12 // (op 6467) 
v_mul_f32 v17, v0, v13 // (op 6467) 

v_add_f32 v10, v14, v18 // (op 6468) 
v_add_f32 v11, v15, v19 // (op 6468) 
v_add_f32 v12, v16, v20 // (op 6468) 
v_add_f32 v13, v17, v21 // (op 6468) 




s_lshl_b32 s0, s4, 8 // (op 6444) 
s_add_u32 s0, s0, 64 // (op 6444) 
v_add_u32 v3, v9, s0 // (op 6444) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6444) Store value




s_lshl_b32 s0, s2, 6 // (op 6473) 
s_mul_i32 s1, 5, s0 // (op 6473) 
s_add_u32 s1, s1, 64 // (op 6473) 

v_add_u32 v3, v8, s1 // (op 6473) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6473) Load value

v_accvgpr_read v14, a52 // (op 6491) 
v_accvgpr_read v15, a53 // (op 6491) 
v_accvgpr_read v16, a54 // (op 6491) 
v_accvgpr_read v17, a55 // (op 6491) 
v_mul_f32 v18, v7, v14 // (op 6491) 
v_mul_f32 v19, v7, v15 // (op 6491) 
v_mul_f32 v20, v7, v16 // (op 6491) 
v_mul_f32 v21, v7, v17 // (op 6491) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6471) 
v_mul_f32 v15, v0, v11 // (op 6471) 
v_mul_f32 v16, v0, v12 // (op 6471) 
v_mul_f32 v17, v0, v13 // (op 6471) 

v_add_f32 v10, v14, v18 // (op 6472) 
v_add_f32 v11, v15, v19 // (op 6472) 
v_add_f32 v12, v16, v20 // (op 6472) 
v_add_f32 v13, v17, v21 // (op 6472) 




s_lshl_b32 s0, s4, 6 // (op 6446) 
s_mul_i32 s1, 5, s0 // (op 6446) 
s_add_u32 s1, s1, 64 // (op 6446) 

v_add_u32 v3, v9, s1 // (op 6446) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6446) Store value




s_lshl_b32 s0, s2, 6 // (op 6477) 
s_mul_i32 s1, 6, s0 // (op 6477) 
s_add_u32 s1, s1, 64 // (op 6477) 

v_add_u32 v3, v8, s1 // (op 6477) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6477) Load value

v_accvgpr_read v14, a56 // (op 6493) 
v_accvgpr_read v15, a57 // (op 6493) 
v_accvgpr_read v16, a58 // (op 6493) 
v_accvgpr_read v17, a59 // (op 6493) 
v_mul_f32 v18, v7, v14 // (op 6493) 
v_mul_f32 v19, v7, v15 // (op 6493) 
v_mul_f32 v20, v7, v16 // (op 6493) 
v_mul_f32 v21, v7, v17 // (op 6493) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6475) 
v_mul_f32 v15, v0, v11 // (op 6475) 
v_mul_f32 v16, v0, v12 // (op 6475) 
v_mul_f32 v17, v0, v13 // (op 6475) 

v_add_f32 v10, v14, v18 // (op 6476) 
v_add_f32 v11, v15, v19 // (op 6476) 
v_add_f32 v12, v16, v20 // (op 6476) 
v_add_f32 v13, v17, v21 // (op 6476) 




s_lshl_b32 s0, s4, 6 // (op 6448) 
s_mul_i32 s1, 6, s0 // (op 6448) 
s_add_u32 s1, s1, 64 // (op 6448) 

v_add_u32 v3, v9, s1 // (op 6448) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6448) Store value




s_lshl_b32 s0, s2, 6 // (op 6481) 
s_mul_i32 s1, 7, s0 // (op 6481) 
s_add_u32 s1, s1, 64 // (op 6481) 

v_add_u32 v3, v8, s1 // (op 6481) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6481) Load value

v_accvgpr_read v14, a60 // (op 6495) 
v_accvgpr_read v15, a61 // (op 6495) 
v_accvgpr_read v16, a62 // (op 6495) 
v_accvgpr_read v17, a63 // (op 6495) 
v_mul_f32 v18, v7, v14 // (op 6495) 
v_mul_f32 v19, v7, v15 // (op 6495) 
v_mul_f32 v20, v7, v16 // (op 6495) 
v_mul_f32 v21, v7, v17 // (op 6495) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6479) 
v_mul_f32 v15, v0, v11 // (op 6479) 
v_mul_f32 v16, v0, v12 // (op 6479) 
v_mul_f32 v17, v0, v13 // (op 6479) 

v_add_f32 v10, v14, v18 // (op 6480) 
v_add_f32 v11, v15, v19 // (op 6480) 
v_add_f32 v12, v16, v20 // (op 6480) 
v_add_f32 v13, v17, v21 // (op 6480) 




s_lshl_b32 s0, s4, 6 // (op 6450) 
s_mul_i32 s1, 7, s0 // (op 6450) 
s_add_u32 s1, s1, 64 // (op 6450) 

v_add_u32 v3, v9, s1 // (op 6450) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6450) Store value




v_mov_b32 v3, v8 // (op 126) reg expression

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 126) Load value

v_accvgpr_read v14, a0 // (op 123) 
v_accvgpr_read v15, a1 // (op 123) 
v_accvgpr_read v16, a2 // (op 123) 
v_accvgpr_read v17, a3 // (op 123) 
v_mul_f32 v18, v7, v14 // (op 123) 
v_mul_f32 v19, v7, v15 // (op 123) 
v_mul_f32 v20, v7, v16 // (op 123) 
v_mul_f32 v21, v7, v17 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 127) 
v_mul_f32 v15, v0, v11 // (op 127) 
v_mul_f32 v16, v0, v12 // (op 127) 
v_mul_f32 v17, v0, v13 // (op 127) 

v_add_f32 v10, v14, v18 // (op 124) 
v_add_f32 v11, v15, v19 // (op 124) 
v_add_f32 v12, v16, v20 // (op 124) 
v_add_f32 v13, v17, v21 // (op 124) 




v_mov_b32 v3, v9 // (op 125) reg expression

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 6271) 
v_add_u32 v3, v8, s0 // (op 6271) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6271) Load value

v_accvgpr_read v14, a4 // (op 6268) 
v_accvgpr_read v15, a5 // (op 6268) 
v_accvgpr_read v16, a6 // (op 6268) 
v_accvgpr_read v17, a7 // (op 6268) 
v_mul_f32 v18, v7, v14 // (op 6268) 
v_mul_f32 v19, v7, v15 // (op 6268) 
v_mul_f32 v20, v7, v16 // (op 6268) 
v_mul_f32 v21, v7, v17 // (op 6268) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6272) 
v_mul_f32 v15, v0, v11 // (op 6272) 
v_mul_f32 v16, v0, v12 // (op 6272) 
v_mul_f32 v17, v0, v13 // (op 6272) 

v_add_f32 v10, v14, v18 // (op 6269) 
v_add_f32 v11, v15, v19 // (op 6269) 
v_add_f32 v12, v16, v20 // (op 6269) 
v_add_f32 v13, v17, v21 // (op 6269) 




s_lshl_b32 s0, s4, 6 // (op 6270) 
v_add_u32 v3, v9, s0 // (op 6270) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6270) Store value




s_lshl_b32 s0, s2, 7 // (op 6283) 
v_add_u32 v3, v8, s0 // (op 6283) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6283) Load value

v_accvgpr_read v14, a8 // (op 6280) 
v_accvgpr_read v15, a9 // (op 6280) 
v_accvgpr_read v16, a10 // (op 6280) 
v_accvgpr_read v17, a11 // (op 6280) 
v_mul_f32 v18, v7, v14 // (op 6280) 
v_mul_f32 v19, v7, v15 // (op 6280) 
v_mul_f32 v20, v7, v16 // (op 6280) 
v_mul_f32 v21, v7, v17 // (op 6280) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6284) 
v_mul_f32 v15, v0, v11 // (op 6284) 
v_mul_f32 v16, v0, v12 // (op 6284) 
v_mul_f32 v17, v0, v13 // (op 6284) 

v_add_f32 v10, v14, v18 // (op 6281) 
v_add_f32 v11, v15, v19 // (op 6281) 
v_add_f32 v12, v16, v20 // (op 6281) 
v_add_f32 v13, v17, v21 // (op 6281) 




s_lshl_b32 s0, s4, 7 // (op 6282) 
v_add_u32 v3, v9, s0 // (op 6282) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6282) Store value




s_lshl_b32 s0, s2, 6 // (op 6295) 
s_mul_i32 s1, 3, s0 // (op 6295) 

v_add_u32 v3, v8, s1 // (op 6295) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6295) Load value

v_accvgpr_read v14, a12 // (op 6292) 
v_accvgpr_read v15, a13 // (op 6292) 
v_accvgpr_read v16, a14 // (op 6292) 
v_accvgpr_read v17, a15 // (op 6292) 
v_mul_f32 v18, v7, v14 // (op 6292) 
v_mul_f32 v19, v7, v15 // (op 6292) 
v_mul_f32 v20, v7, v16 // (op 6292) 
v_mul_f32 v21, v7, v17 // (op 6292) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6296) 
v_mul_f32 v15, v0, v11 // (op 6296) 
v_mul_f32 v16, v0, v12 // (op 6296) 
v_mul_f32 v17, v0, v13 // (op 6296) 

v_add_f32 v10, v14, v18 // (op 6293) 
v_add_f32 v11, v15, v19 // (op 6293) 
v_add_f32 v12, v16, v20 // (op 6293) 
v_add_f32 v13, v17, v21 // (op 6293) 




s_lshl_b32 s0, s4, 6 // (op 6294) 
s_mul_i32 s1, 3, s0 // (op 6294) 

v_add_u32 v3, v9, s1 // (op 6294) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6294) Store value




s_lshl_b32 s0, s2, 8 // (op 6307) 
v_add_u32 v3, v8, s0 // (op 6307) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6307) Load value

v_accvgpr_read v14, a16 // (op 6304) 
v_accvgpr_read v15, a17 // (op 6304) 
v_accvgpr_read v16, a18 // (op 6304) 
v_accvgpr_read v17, a19 // (op 6304) 
v_mul_f32 v18, v7, v14 // (op 6304) 
v_mul_f32 v19, v7, v15 // (op 6304) 
v_mul_f32 v20, v7, v16 // (op 6304) 
v_mul_f32 v21, v7, v17 // (op 6304) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6308) 
v_mul_f32 v15, v0, v11 // (op 6308) 
v_mul_f32 v16, v0, v12 // (op 6308) 
v_mul_f32 v17, v0, v13 // (op 6308) 

v_add_f32 v10, v14, v18 // (op 6305) 
v_add_f32 v11, v15, v19 // (op 6305) 
v_add_f32 v12, v16, v20 // (op 6305) 
v_add_f32 v13, v17, v21 // (op 6305) 




s_lshl_b32 s0, s4, 8 // (op 6306) 
v_add_u32 v3, v9, s0 // (op 6306) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6306) Store value




s_lshl_b32 s0, s2, 6 // (op 6319) 
s_mul_i32 s1, 5, s0 // (op 6319) 

v_add_u32 v3, v8, s1 // (op 6319) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6319) Load value

v_accvgpr_read v14, a20 // (op 6316) 
v_accvgpr_read v15, a21 // (op 6316) 
v_accvgpr_read v16, a22 // (op 6316) 
v_accvgpr_read v17, a23 // (op 6316) 
v_mul_f32 v18, v7, v14 // (op 6316) 
v_mul_f32 v19, v7, v15 // (op 6316) 
v_mul_f32 v20, v7, v16 // (op 6316) 
v_mul_f32 v21, v7, v17 // (op 6316) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6320) 
v_mul_f32 v15, v0, v11 // (op 6320) 
v_mul_f32 v16, v0, v12 // (op 6320) 
v_mul_f32 v17, v0, v13 // (op 6320) 

v_add_f32 v10, v14, v18 // (op 6317) 
v_add_f32 v11, v15, v19 // (op 6317) 
v_add_f32 v12, v16, v20 // (op 6317) 
v_add_f32 v13, v17, v21 // (op 6317) 




s_lshl_b32 s0, s4, 6 // (op 6318) 
s_mul_i32 s1, 5, s0 // (op 6318) 

v_add_u32 v3, v9, s1 // (op 6318) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6318) Store value




s_lshl_b32 s0, s2, 6 // (op 6331) 
s_mul_i32 s1, 6, s0 // (op 6331) 

v_add_u32 v3, v8, s1 // (op 6331) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6331) Load value

v_accvgpr_read v14, a24 // (op 6328) 
v_accvgpr_read v15, a25 // (op 6328) 
v_accvgpr_read v16, a26 // (op 6328) 
v_accvgpr_read v17, a27 // (op 6328) 
v_mul_f32 v18, v7, v14 // (op 6328) 
v_mul_f32 v19, v7, v15 // (op 6328) 
v_mul_f32 v20, v7, v16 // (op 6328) 
v_mul_f32 v21, v7, v17 // (op 6328) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6332) 
v_mul_f32 v15, v0, v11 // (op 6332) 
v_mul_f32 v16, v0, v12 // (op 6332) 
v_mul_f32 v17, v0, v13 // (op 6332) 

v_add_f32 v10, v14, v18 // (op 6329) 
v_add_f32 v11, v15, v19 // (op 6329) 
v_add_f32 v12, v16, v20 // (op 6329) 
v_add_f32 v13, v17, v21 // (op 6329) 




s_lshl_b32 s0, s4, 6 // (op 6330) 
s_mul_i32 s1, 6, s0 // (op 6330) 

v_add_u32 v3, v9, s1 // (op 6330) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6330) Store value




s_lshl_b32 s0, s2, 6 // (op 6343) 
s_mul_i32 s1, 7, s0 // (op 6343) 

v_add_u32 v3, v8, s1 // (op 6343) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6343) Load value

v_accvgpr_read v14, a28 // (op 6340) 
v_accvgpr_read v15, a29 // (op 6340) 
v_accvgpr_read v16, a30 // (op 6340) 
v_accvgpr_read v17, a31 // (op 6340) 
v_mul_f32 v18, v7, v14 // (op 6340) 
v_mul_f32 v19, v7, v15 // (op 6340) 
v_mul_f32 v20, v7, v16 // (op 6340) 
v_mul_f32 v21, v7, v17 // (op 6340) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6344) 
v_mul_f32 v15, v0, v11 // (op 6344) 
v_mul_f32 v16, v0, v12 // (op 6344) 
v_mul_f32 v17, v0, v13 // (op 6344) 

v_add_f32 v10, v14, v18 // (op 6341) 
v_add_f32 v11, v15, v19 // (op 6341) 
v_add_f32 v12, v16, v20 // (op 6341) 
v_add_f32 v13, v17, v21 // (op 6341) 




s_lshl_b32 s0, s4, 6 // (op 6342) 
s_mul_i32 s1, 7, s0 // (op 6342) 

v_add_u32 v3, v9, s1 // (op 6342) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6342) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_ConditionalBottom_0_beta_114:








































s_endpgm  // End of RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
.LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_end:
.size RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2, .LRR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_end-RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
  .amdhsa_next_free_vgpr 316
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 73728 // lds bytes
  .amdhsa_accum_offset 188
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
    DataType: B6
    DataTypeA: B6
    DataTypeB: B6
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
  - .name: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2
    .symbol: RR_GEMM_TN_BF6_BF6_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 73728
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 186
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF6, pointerType: PointerGlobal}
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

