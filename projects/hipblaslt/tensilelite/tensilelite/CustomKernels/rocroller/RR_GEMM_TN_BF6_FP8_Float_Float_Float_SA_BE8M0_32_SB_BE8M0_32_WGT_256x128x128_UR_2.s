

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2
.p2align 8
.type RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2,@function
RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_exec_begin:


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


v_and_b32 v0, 4294967168, v1 // (op 13742) 
v_and_b32 v3, 15, v1 // (op 13742) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 13742) 


v_and_b32 v0, 63, v1 // (op 13742) 
v_lshrrev_b32 v3, 4, v0 // (op 13742) 

v_lshlrev_b32 v0, 5, v3 // (op 13742) 

v_add_u32 v3, v4, v0 // (op 13742) 


v_mul_lo_u32 v0, v3, 6 // (op 13742) 

v_lshrrev_b32 v3, 3, v0 // (op 13742) 

v_lshrrev_b32 v0, 6, v1 // (op 13799) 
v_and_b32 v4, 1, v0 // (op 13799) 

v_and_b32 v0, 15, v1 // (op 13799) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 13799) 


v_and_b32 v0, 48, v1 // (op 13799) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 13799) 


v_and_b32 v0, 4294967168, v1 // (op 13851) 
v_and_b32 v5, 15, v1 // (op 13851) 
v_add_lshl_u32 v6, v0, v5, 7 // (op 13851) 


v_and_b32 v0, 63, v1 // (op 13851) 
v_lshrrev_b32 v5, 4, v0 // (op 13851) 

v_lshlrev_b32 v0, 5, v5 // (op 13851) 

v_add_u32 v5, v6, v0 // (op 13851) 


v_mul_lo_u32 v0, v5, 6 // (op 13851) 

v_lshrrev_b32 v5, 3, v0 // (op 13851) 

v_lshrrev_b32 v0, 6, v1 // (op 13895) 
v_and_b32 v6, 1, v0 // (op 13895) 

v_and_b32 v0, 15, v1 // (op 13895) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 13895) 


v_and_b32 v0, 48, v1 // (op 13895) 
v_lshl_add_u32 v6, v7, 7, v0 // (op 13895) 


v_and_b32 v0, 4294967168, v1 // (op 13680) 
v_and_b32 v7, 63, v1 // (op 13680) 
v_lshrrev_b32 v8, 4, v7 // (op 13680) 
v_lshlrev_b32 v9, 2, v8 // (op 13680) 

v_add_u32 v8, v0, v9 // (op 13680) 


v_lshl_add_u32 v0, s16, 8, v8 // (op 13680) 

v_lshrrev_b32 v8, 2, v7 // (op 13680) 

v_and_b32 v7, 3, v8 // (op 13680) 

v_and_b32 v8, 3, v1 // (op 13680) 
v_lshl_add_u32 v9, v7, 2, v8 // (op 13680) 


v_lshrrev_b32 v7, 6, v1 // (op 13680) 
v_and_b32 v8, 1, v7 // (op 13680) 

v_lshl_add_u32 v7, v8, 6, v9 // (op 13680) 


v_lshl_add_u32 v8, s17, 7, v7 // (op 13680) 

v_mul_lo_u32 v7, v8, s2 // (op 13680) 

v_add_lshl_u32 v8, v0, v7, 2 // (op 13680) 


v_and_b32 v0, 4294967168, v1 // (op 13820) 
v_and_b32 v7, 63, v1 // (op 13820) 
v_lshrrev_b32 v9, 4, v7 // (op 13820) 
s_mul_i32 s1, s2, s9 // (op 13682) most significant: low of low * high
s_mul_i32 s18, s3, s8 // (op 13682) most significant: low of high * low
s_mul_i32 s0, s2, s8 // (op 13682) least significant
s_mul_hi_u32 s19, s2, s8 // (op 13682) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13682) most significant: sum
s_add_u32 s1, s1, s19 // (op 13682) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 13682) 

v_lshlrev_b32 v10, 2, v9 // (op 13820) 

v_add_u32 v9, v0, v10 // (op 13820) 


v_lshl_add_u32 v0, s16, 8, v9 // (op 13820) 

v_lshrrev_b32 v9, 2, v7 // (op 13820) 

v_and_b32 v7, 3, v9 // (op 13820) 

v_and_b32 v9, 3, v1 // (op 13820) 
v_lshl_add_u32 v10, v7, 2, v9 // (op 13820) 


v_lshrrev_b32 v7, 6, v1 // (op 13820) 
v_and_b32 v9, 1, v7 // (op 13820) 

v_lshl_add_u32 v7, v9, 6, v10 // (op 13820) 


v_lshl_add_u32 v9, s17, 7, v7 // (op 13820) 

v_mul_lo_u32 v7, v9, s4 // (op 13820) 

v_add_lshl_u32 v9, v0, v7, 2 // (op 13820) 


s_mov_b32 s40, s6 // (op 13682) reg expression
s_mov_b32 s41, s7 // (op 13682) reg expression
s_mov_b32 s42, s18 // (op 13682) reg expression
s_mov_b32 s43, 131072 // (op 13682) call()

s_mul_i32 s1, s4, s9 // (op 13822) most significant: low of low * high

s_mul_i32 s6, s5, s8 // (op 13822) most significant: low of high * low
s_mul_i32 s0, s4, s8 // (op 13822) least significant
s_mul_hi_u32 s7, s4, s8 // (op 13822) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13822) most significant: sum
s_add_u32 s1, s1, s7 // (op 13822) most significant: sum


s_lshl_b64 s[6:7], s[0:1], 2 // (op 13822) 

s_mov_b32 s44, s10 // (op 13822) reg expression
s_mov_b32 s45, s11 // (op 13822) reg expression
s_mov_b32 s46, s6 // (op 13822) reg expression
s_mov_b32 s47, 131072 // (op 13822) call()


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
v_accvgpr_write a16, 0.00000 // (op 773) call()
v_accvgpr_write a17, 0.00000 // (op 773) call()
v_accvgpr_write a18, 0.00000 // (op 773) call()
v_accvgpr_write a19, 0.00000 // (op 773) call()
v_accvgpr_write a20, 0.00000 // (op 1034) call()
v_accvgpr_write a21, 0.00000 // (op 1034) call()
v_accvgpr_write a22, 0.00000 // (op 1034) call()
v_accvgpr_write a23, 0.00000 // (op 1034) call()
v_accvgpr_write a24, 0.00000 // (op 1035) call()
v_accvgpr_write a25, 0.00000 // (op 1035) call()
v_accvgpr_write a26, 0.00000 // (op 1035) call()
v_accvgpr_write a27, 0.00000 // (op 1035) call()
v_accvgpr_write a28, 0.00000 // (op 1036) call()
v_accvgpr_write a29, 0.00000 // (op 1036) call()
v_accvgpr_write a30, 0.00000 // (op 1036) call()
v_accvgpr_write a31, 0.00000 // (op 1036) call()
v_accvgpr_write a32, 0.00000 // (op 1383) call()
v_accvgpr_write a33, 0.00000 // (op 1383) call()
v_accvgpr_write a34, 0.00000 // (op 1383) call()
v_accvgpr_write a35, 0.00000 // (op 1383) call()
v_accvgpr_write a36, 0.00000 // (op 1644) call()
v_accvgpr_write a37, 0.00000 // (op 1644) call()
v_accvgpr_write a38, 0.00000 // (op 1644) call()
v_accvgpr_write a39, 0.00000 // (op 1644) call()
v_accvgpr_write a40, 0.00000 // (op 1645) call()
v_accvgpr_write a41, 0.00000 // (op 1645) call()
v_accvgpr_write a42, 0.00000 // (op 1645) call()
v_accvgpr_write a43, 0.00000 // (op 1645) call()
v_accvgpr_write a44, 0.00000 // (op 1646) call()
v_accvgpr_write a45, 0.00000 // (op 1646) call()
v_accvgpr_write a46, 0.00000 // (op 1646) call()
v_accvgpr_write a47, 0.00000 // (op 1646) call()
v_accvgpr_write a48, 0.00000 // (op 1993) call()
v_accvgpr_write a49, 0.00000 // (op 1993) call()
v_accvgpr_write a50, 0.00000 // (op 1993) call()
v_accvgpr_write a51, 0.00000 // (op 1993) call()
v_accvgpr_write a52, 0.00000 // (op 2254) call()
v_accvgpr_write a53, 0.00000 // (op 2254) call()
v_accvgpr_write a54, 0.00000 // (op 2254) call()
v_accvgpr_write a55, 0.00000 // (op 2254) call()
v_accvgpr_write a56, 0.00000 // (op 2255) call()
v_accvgpr_write a57, 0.00000 // (op 2255) call()
v_accvgpr_write a58, 0.00000 // (op 2255) call()
v_accvgpr_write a59, 0.00000 // (op 2255) call()
v_accvgpr_write a60, 0.00000 // (op 2256) call()
v_accvgpr_write a61, 0.00000 // (op 2256) call()
v_accvgpr_write a62, 0.00000 // (op 2256) call()
v_accvgpr_write a63, 0.00000 // (op 2256) call()
v_accvgpr_write a64, 0.00000 // (op 2603) call()
v_accvgpr_write a65, 0.00000 // (op 2603) call()
v_accvgpr_write a66, 0.00000 // (op 2603) call()
v_accvgpr_write a67, 0.00000 // (op 2603) call()
v_accvgpr_write a68, 0.00000 // (op 2864) call()
v_accvgpr_write a69, 0.00000 // (op 2864) call()
v_accvgpr_write a70, 0.00000 // (op 2864) call()
v_accvgpr_write a71, 0.00000 // (op 2864) call()
v_accvgpr_write a72, 0.00000 // (op 2865) call()
v_accvgpr_write a73, 0.00000 // (op 2865) call()
v_accvgpr_write a74, 0.00000 // (op 2865) call()
v_accvgpr_write a75, 0.00000 // (op 2865) call()
v_accvgpr_write a76, 0.00000 // (op 2866) call()
v_accvgpr_write a77, 0.00000 // (op 2866) call()
v_accvgpr_write a78, 0.00000 // (op 2866) call()
v_accvgpr_write a79, 0.00000 // (op 2866) call()
v_accvgpr_write a80, 0.00000 // (op 3213) call()
v_accvgpr_write a81, 0.00000 // (op 3213) call()
v_accvgpr_write a82, 0.00000 // (op 3213) call()
v_accvgpr_write a83, 0.00000 // (op 3213) call()
v_accvgpr_write a84, 0.00000 // (op 3474) call()
v_accvgpr_write a85, 0.00000 // (op 3474) call()
v_accvgpr_write a86, 0.00000 // (op 3474) call()
v_accvgpr_write a87, 0.00000 // (op 3474) call()
v_accvgpr_write a88, 0.00000 // (op 3475) call()
v_accvgpr_write a89, 0.00000 // (op 3475) call()
v_accvgpr_write a90, 0.00000 // (op 3475) call()
v_accvgpr_write a91, 0.00000 // (op 3475) call()
v_accvgpr_write a92, 0.00000 // (op 3476) call()
v_accvgpr_write a93, 0.00000 // (op 3476) call()
v_accvgpr_write a94, 0.00000 // (op 3476) call()
v_accvgpr_write a95, 0.00000 // (op 3476) call()
v_accvgpr_write a96, 0.00000 // (op 3823) call()
v_accvgpr_write a97, 0.00000 // (op 3823) call()
v_accvgpr_write a98, 0.00000 // (op 3823) call()
v_accvgpr_write a99, 0.00000 // (op 3823) call()
v_accvgpr_write a100, 0.00000 // (op 4084) call()
v_accvgpr_write a101, 0.00000 // (op 4084) call()
v_accvgpr_write a102, 0.00000 // (op 4084) call()
v_accvgpr_write a103, 0.00000 // (op 4084) call()
v_accvgpr_write a104, 0.00000 // (op 4085) call()
v_accvgpr_write a105, 0.00000 // (op 4085) call()
v_accvgpr_write a106, 0.00000 // (op 4085) call()
v_accvgpr_write a107, 0.00000 // (op 4085) call()
v_accvgpr_write a108, 0.00000 // (op 4086) call()
v_accvgpr_write a109, 0.00000 // (op 4086) call()
v_accvgpr_write a110, 0.00000 // (op 4086) call()
v_accvgpr_write a111, 0.00000 // (op 4086) call()
v_accvgpr_write a112, 0.00000 // (op 4433) call()
v_accvgpr_write a113, 0.00000 // (op 4433) call()
v_accvgpr_write a114, 0.00000 // (op 4433) call()
v_accvgpr_write a115, 0.00000 // (op 4433) call()
v_accvgpr_write a116, 0.00000 // (op 4694) call()
v_accvgpr_write a117, 0.00000 // (op 4694) call()
v_accvgpr_write a118, 0.00000 // (op 4694) call()
v_accvgpr_write a119, 0.00000 // (op 4694) call()
v_accvgpr_write a120, 0.00000 // (op 4695) call()
v_accvgpr_write a121, 0.00000 // (op 4695) call()
v_accvgpr_write a122, 0.00000 // (op 4695) call()
v_accvgpr_write a123, 0.00000 // (op 4695) call()
v_accvgpr_write a124, 0.00000 // (op 4696) call()
v_accvgpr_write a125, 0.00000 // (op 4696) call()
v_accvgpr_write a126, 0.00000 // (op 4696) call()
v_accvgpr_write a127, 0.00000 // (op 4696) call()
v_and_b32 v10, 4294967294, v1 // (op 13570) 
v_lshl_add_u32 v11, s16, 8, v10 // (op 13570) 

v_and_b32 v10, -2, v1 // (op 13570) 
v_sub_u32 v12, v1, v10 // (op 13570) 

v_lshlrev_b32 v10, 6, v12 // (op 13570) 

v_mov_b32 v13, 0 // (op 13570) convert
v_mov_b32 v12, v10 // (op 13570) convert

v_mul_lo_u32 v15, v11, s13 // (op 13570) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 13570) most significant: high of low * low
v_mul_lo_u32 v14, v11, s12 // (op 13570) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 13570) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 13570) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 13570) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 13570) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 13570) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 13570) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 13570) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 13570) 

v_mov_b32 v10, v12 // (op 13570) convert


s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s23, 6 // (op 13572) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 13572) least significant
s_mul_hi_u32 s6, s22, 6 // (op 13572) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13572) most significant: sum

s_mul_i32 s7, s12, s1 // (op 13572) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 13572) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 13572) least significant
s_mul_hi_u32 s11, s12, s0 // (op 13572) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13572) most significant: sum
s_add_u32 s7, s7, s11 // (op 13572) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13572) 

s_mov_b32 s48, s20 // (op 13572) reg expression
s_mov_b32 s49, s21 // (op 13572) reg expression
s_mov_b32 s50, s0 // (op 13572) reg expression
s_mov_b32 s51, 131072 // (op 13572) call()

v_and_b32 v11, 4294967294, v1 // (op 13589) 
v_lshl_add_u32 v12, s16, 8, v11 // (op 13589) 

v_and_b32 v11, -2, v1 // (op 13589) 
v_sub_u32 v13, v1, v11 // (op 13589) 

v_lshlrev_b32 v11, 6, v13 // (op 13589) 

v_mov_b32 v15, 0 // (op 13589) convert
v_mov_b32 v14, v11 // (op 13589) convert

v_mul_lo_u32 v17, v12, s13 // (op 13589) most significant: low of low * high
v_mul_hi_u32 v11, v12, s12 // (op 13589) most significant: high of low * low
v_mul_lo_u32 v16, v12, s12 // (op 13589) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 13589) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 13589) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 13589) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 13589) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 13589) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 13589) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 13589) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 13589) 

v_mov_b32 v11, v12 // (op 13589) convert

s_mul_i32 s1, s23, 6 // (op 13591) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 13591) least significant
s_mul_hi_u32 s6, s22, 6 // (op 13591) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13591) most significant: sum

s_mul_i32 s7, s12, s1 // (op 13591) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 13591) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 13591) least significant
s_mul_hi_u32 s11, s12, s0 // (op 13591) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13591) most significant: sum
s_add_u32 s7, s7, s11 // (op 13591) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13591) 

s_mov_b32 s48, s20 // (op 13591) reg expression
s_mov_b32 s49, s21 // (op 13591) reg expression
s_mov_b32 s50, s0 // (op 13591) reg expression
s_mov_b32 s51, 131072 // (op 13591) call()

v_and_b32 v12, -8, v1 // (op 13634) 
v_sub_u32 v13, v1, v12 // (op 13634) 

v_lshlrev_b32 v12, 4, v13 // (op 13634) 

v_mov_b32 v15, 0 // (op 13634) convert
v_mov_b32 v14, v12 // (op 13634) convert

v_lshrrev_b32 v12, 3, v1 // (op 13634) 
v_lshl_add_u32 v13, s17, 7, v12 // (op 13634) 

v_mul_lo_u32 v17, v13, s25 // (op 13634) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 13634) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 13634) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 13634) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 13634) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 13634) most significant half



s_mul_i32 s1, s24, s9 // (op 13636) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 13636) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13636) least significant
s_mul_hi_u32 s7, s24, s8 // (op 13636) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13636) most significant: sum
s_add_u32 s1, s1, s7 // (op 13636) most significant: sum


s_mov_b32 s52, s26 // (op 13636) reg expression
s_mov_b32 s53, s27 // (op 13636) reg expression
s_mov_b32 s54, s0 // (op 13636) reg expression
s_mov_b32 s55, 131072 // (op 13636) call()

v_and_b32 v14, -8, v1 // (op 13653) 
v_sub_u32 v15, v1, v14 // (op 13653) 

v_lshlrev_b32 v14, 4, v15 // (op 13653) 

v_mov_b32 v17, 0 // (op 13653) convert
v_mov_b32 v16, v14 // (op 13653) convert

v_lshrrev_b32 v14, 3, v1 // (op 13653) 
v_lshl_add_u32 v15, s17, 7, v14 // (op 13653) 

v_mul_lo_u32 v19, v15, s25 // (op 13653) most significant: low of low * high
v_mul_hi_u32 v14, v15, s24 // (op 13653) most significant: high of low * low
v_mul_lo_u32 v18, v15, s24 // (op 13653) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 13653) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 13653) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 13653) most significant half



s_mul_i32 s1, s24, s9 // (op 13655) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 13655) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13655) least significant
s_mul_hi_u32 s7, s24, s8 // (op 13655) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13655) most significant: sum
s_add_u32 s1, s1, s7 // (op 13655) most significant: sum


s_mov_b32 s52, s26 // (op 13655) reg expression
s_mov_b32 s53, s27 // (op 13655) reg expression
s_mov_b32 s54, s0 // (op 13655) reg expression
s_mov_b32 s55, 131072 // (op 13655) call()

v_and_b32 v16, 63, v1 // (op 13917) 
v_lshrrev_b32 v17, 4, v16 // (op 13917) 
v_and_b32 v18, 3, v17 // (op 13917) 

v_and_b32 v17, 15, v1 // (op 13917) 
v_lshl_add_u32 v19, v18, 4, v17 // (op 13917) 


v_and_b32 v17, 4294967168, v1 // (op 13917) 
v_add_u32 v18, v17, v19 // (op 13917) 


v_lshl_add_u32 v17, s16, 8, v18 // (op 13917) 

v_lshrrev_b32 v18, 6, v16 // (op 13917) 

v_lshlrev_b32 v16, 2, v18 // (op 13917) 

v_mul_lo_u32 v18, v17, s28 // (op 13917) 
v_add_u32 v18, v18, v16 // (op 13917) 


s_mul_i32 s1, s28, s23 // (op 13919) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 13919) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 13919) least significant
s_mul_hi_u32 s7, s28, s22 // (op 13919) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13919) most significant: sum
s_add_u32 s1, s1, s7 // (op 13919) most significant: sum


s_mov_b32 s56, s30 // (op 13919) reg expression
s_mov_b32 s57, s31 // (op 13919) reg expression
s_mov_b32 s58, s0 // (op 13919) reg expression
s_mov_b32 s59, 131072 // (op 13919) call()

v_and_b32 v16, 63, v1 // (op 13938) 
v_lshrrev_b32 v17, 4, v16 // (op 13938) 
v_and_b32 v19, 3, v17 // (op 13938) 

v_and_b32 v17, 15, v1 // (op 13938) 
v_lshl_add_u32 v20, v19, 4, v17 // (op 13938) 


v_and_b32 v17, 4294967168, v1 // (op 13938) 
v_add_u32 v19, v17, v20 // (op 13938) 


v_lshl_add_u32 v17, s16, 8, v19 // (op 13938) 

v_lshrrev_b32 v19, 6, v16 // (op 13938) 

v_lshlrev_b32 v16, 2, v19 // (op 13938) 

v_mul_lo_u32 v19, v17, s28 // (op 13938) 
v_add_u32 v19, v19, v16 // (op 13938) 


s_mul_i32 s1, s28, s23 // (op 13940) most significant: low of low * high
s_mul_i32 s6, s29, s22 // (op 13940) most significant: low of high * low
s_mul_i32 s0, s28, s22 // (op 13940) least significant
s_mul_hi_u32 s7, s28, s22 // (op 13940) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13940) most significant: sum
s_add_u32 s1, s1, s7 // (op 13940) most significant: sum


s_mov_b32 s56, s30 // (op 13940) reg expression
s_mov_b32 s57, s31 // (op 13940) reg expression
s_mov_b32 s58, s0 // (op 13940) reg expression
s_mov_b32 s59, 131072 // (op 13940) call()

v_and_b32 v16, 63, v1 // (op 13969) 
v_lshrrev_b32 v17, 4, v16 // (op 13969) 
v_and_b32 v20, 3, v17 // (op 13969) 

v_and_b32 v17, 15, v1 // (op 13969) 
v_lshl_add_u32 v21, v20, 4, v17 // (op 13969) 


v_lshrrev_b32 v17, 6, v1 // (op 13969) 
v_and_b32 v20, 1, v17 // (op 13969) 

v_lshl_add_u32 v17, v20, 6, v21 // (op 13969) 


v_lshl_add_u32 v20, s17, 7, v17 // (op 13969) 

v_mul_lo_u32 v17, v20, s32 // (op 13969) 

v_lshrrev_b32 v20, 6, v16 // (op 13969) 

v_lshl_add_u32 v16, v20, 2, v17 // (op 13969) 


s_mul_i32 s1, s32, s9 // (op 13971) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 13971) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 13971) least significant
s_mul_hi_u32 s7, s32, s8 // (op 13971) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13971) most significant: sum
s_add_u32 s1, s1, s7 // (op 13971) most significant: sum


s_mov_b32 s60, s34 // (op 13971) reg expression
s_mov_b32 s61, s35 // (op 13971) reg expression
s_mov_b32 s62, s0 // (op 13971) reg expression
s_mov_b32 s63, 131072 // (op 13971) call()

v_and_b32 v17, 63, v1 // (op 13990) 
v_lshrrev_b32 v20, 4, v17 // (op 13990) 
v_and_b32 v21, 3, v20 // (op 13990) 

v_and_b32 v20, 15, v1 // (op 13990) 
v_lshl_add_u32 v22, v21, 4, v20 // (op 13990) 


v_lshrrev_b32 v20, 6, v1 // (op 13990) 
v_and_b32 v21, 1, v20 // (op 13990) 

v_lshl_add_u32 v20, v21, 6, v22 // (op 13990) 


v_lshl_add_u32 v21, s17, 7, v20 // (op 13990) 

v_mul_lo_u32 v20, v21, s32 // (op 13990) 

v_lshrrev_b32 v21, 6, v17 // (op 13990) 

v_lshl_add_u32 v17, v21, 2, v20 // (op 13990) 


s_mul_i32 s1, s32, s9 // (op 13992) most significant: low of low * high
s_mul_i32 s6, s33, s8 // (op 13992) most significant: low of high * low
s_mul_i32 s0, s32, s8 // (op 13992) least significant
s_mul_hi_u32 s7, s32, s8 // (op 13992) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13992) most significant: sum
s_add_u32 s1, s1, s7 // (op 13992) most significant: sum


s_mov_b32 s60, s34 // (op 13992) reg expression
s_mov_b32 s61, s35 // (op 13992) reg expression
s_mov_b32 s62, s0 // (op 13992) reg expression
s_mov_b32 s63, 131072 // (op 13992) call()

v_and_b32 v20, -8, v1 // (op 13782) 
v_sub_u32 v21, v1, v20 // (op 13782) 

v_lshlrev_b32 v20, 4, v21 // (op 13782) 

v_lshrrev_b32 v21, 3, v1 // (op 13782) 
v_lshl_add_u32 v22, v21, 7, v20 // (op 13782) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 13782) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 13495) 
v_mov_b64 v[20:21], s[6:7] // (op 13495) convert


v_mov_b64 v[22:23], v[12:13] // (op 13495) 
s_mov_b32 m0, s0 // (op 13495) reg expression
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 13495) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v20 // (op 13495) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 13495) most significant half

s_add_u32 m0, m0, 4096 // (op 13495) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 13495) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v20 // (op 13495) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 13495) most significant half

s_add_u32 m0, m0, 4096 // (op 13495) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 13495) Load value direct to lds
v_add_co_u32 v22, s[6:7], v22, v20 // (op 13495) least significant half
v_addc_co_u32 v23, s[6:7], 0, v23, s[6:7] // (op 13495) most significant half

s_add_u32 m0, m0, 4096 // (op 13495) 
buffer_load_dwordx4 v22, s[52:55], 0 offen offset: 0 lds // (op 13495) Load value direct to lds




s_mul_i32 s1, s13, 6 // (op 12903) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 12903) least significant
s_mul_hi_u32 s6, s12, 6 // (op 12903) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 12903) most significant: sum

v_lshrrev_b64 v[20:21], 3, s[0:1] // (op 12903) 


v_mov_b32 v22, v10 // (op 12903) 
buffer_load_dwordx4 v[24:27], v22, s[48:51], 0 offen // (op 12903) Load value
buffer_load_dwordx2 v[28:29], v22, s[48:51], 0 offen offset:16 // (op 12903) Load value
buffer_load_dwordx4 v[30:33], v22, s[48:51], 0 offen offset:24 // (op 12903) Load value
buffer_load_dwordx2 v[34:35], v22, s[48:51], 0 offen offset:40 // (op 12903) Load value
v_add_u32 v22, v22, v20 // (op 12903) 
buffer_load_dwordx4 v[36:39], v22, s[48:51], 0 offen // (op 12903) Load value
buffer_load_dwordx2 v[40:41], v22, s[48:51], 0 offen offset:16 // (op 12903) Load value
buffer_load_dwordx4 v[42:45], v22, s[48:51], 0 offen offset:24 // (op 12903) Load value
buffer_load_dwordx2 v[46:47], v22, s[48:51], 0 offen offset:40 // (op 12903) Load value




v_mov_b32 v20, v16 // (op 74) reg expression

buffer_load_dwordx2 v[22:23], v20, s[60:63], 0 offen // (op 74) Load value




s_lshl_b32 s0, s28, 6 // (op 2743) 
v_add_u32 v16, v18, s0 // (op 2743) 


buffer_load_dwordx2 v[20:21], v16, s[56:59], 0 offen // (op 2743) Load value



v_mov_b32 v16, v18 // (op 70) reg expression

buffer_load_dwordx2 v[48:49], v16, s[56:59], 0 offen // (op 70) Load value


v_and_b32 v16, -8, v1 // (op 13878) 
v_sub_u32 v18, v1, v16 // (op 13878) 

v_lshlrev_b32 v16, 4, v18 // (op 13878) 

v_lshrrev_b32 v18, 3, v1 // (op 13878) 
v_lshl_add_u32 v50, v18, 7, v16 // (op 13878) 


s_nop 0
v_readfirstlane_b32 s0, v50 // (op 13878) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[6:7], s[24:25], 5 // (op 13500) 
v_mov_b64 v[50:51], s[6:7] // (op 13500) convert


v_mov_b64 v[52:53], v[12:13] // (op 13500) 
s_add_u32 s1, 16384, s0 // (op 13500) 
s_mov_b32 m0, s1 // (op 13500) reg expression
s_mov_b32 s6, 128 // (op 13500) call()
buffer_load_dwordx4 v52, s[52:55], s6 offen offset: 0 lds // (op 13500) Load value direct to lds
v_add_co_u32 v52, s[10:11], v52, v50 // (op 13500) least significant half
v_addc_co_u32 v53, s[10:11], 0, v53, s[10:11] // (op 13500) most significant half

s_add_u32 m0, m0, 4096 // (op 13500) 
buffer_load_dwordx4 v52, s[52:55], s6 offen offset: 0 lds // (op 13500) Load value direct to lds
v_add_co_u32 v52, s[10:11], v52, v50 // (op 13500) least significant half
v_addc_co_u32 v53, s[10:11], 0, v53, s[10:11] // (op 13500) most significant half

s_add_u32 m0, m0, 4096 // (op 13500) 
buffer_load_dwordx4 v52, s[52:55], s6 offen offset: 0 lds // (op 13500) Load value direct to lds
v_add_co_u32 v52, s[10:11], v52, v50 // (op 13500) least significant half
v_addc_co_u32 v53, s[10:11], 0, v53, s[10:11] // (op 13500) most significant half

s_add_u32 m0, m0, 4096 // (op 13500) 
buffer_load_dwordx4 v52, s[52:55], s6 offen offset: 0 lds // (op 13500) Load value direct to lds







s_mul_i32 s1, s13, 6 // (op 12909) most significant: low of high * low
s_mul_i32 s0, s12, 6 // (op 12909) least significant
s_mul_hi_u32 s6, s12, 6 // (op 12909) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 12909) most significant: sum

v_lshrrev_b64 v[12:13], 3, s[0:1] // (op 12909) 


v_mov_b32 v16, v10 // (op 12909) 
buffer_load_dwordx4 v[50:53], v16, s[48:51], 0 offen offset:96 // (op 12909) Load value
buffer_load_dwordx2 v[54:55], v16, s[48:51], 0 offen offset:112 // (op 12909) Load value
buffer_load_dwordx4 v[56:59], v16, s[48:51], 0 offen offset:120 // (op 12909) Load value
buffer_load_dwordx2 v[60:61], v16, s[48:51], 0 offen offset:136 // (op 12909) Load value
v_add_u32 v16, v16, v12 // (op 12909) 
buffer_load_dwordx4 v[62:65], v16, s[48:51], 0 offen offset:96 // (op 12909) Load value
buffer_load_dwordx2 v[66:67], v16, s[48:51], 0 offen offset:112 // (op 12909) Load value
buffer_load_dwordx4 v[68:71], v16, s[48:51], 0 offen offset:120 // (op 12909) Load value
buffer_load_dwordx2 v[72:73], v16, s[48:51], 0 offen offset:136 // (op 12909) Load value



v_and_b32 v10, -2, v1 // (op 13725) 
v_sub_u32 v16, v1, v10 // (op 13725) 

v_lshlrev_b32 v10, 6, v16 // (op 13725) 

v_lshrrev_b32 v16, 1, v1 // (op 13725) 
v_lshl_add_u32 v18, v16, 8, v10 // (op 13725) 


v_mul_lo_u32 v10, v18, 6 // (op 13725) 

v_lshrrev_b32 v16, 3, v10 // (op 13725) 




s_waitcnt vmcnt(22)
ds_write_b128 v16, v[24:27] offset:32768 // (op 12912) Store local data 
s_waitcnt vmcnt(21)
ds_write_b64 v16, v[28:29] offset:32784 // (op 12912) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v16, v[30:33] offset:32792 // (op 12912) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v16, v[34:35] offset:32808 // (op 12912) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v16, v[36:39] offset:32864 // (op 12912) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v16, v[40:41] offset:32880 // (op 12912) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v16, v[42:45] offset:32888 // (op 12912) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v16, v[46:47] offset:32904 // (op 12912) Store local data 

v_and_b32 v10, -2, v1 // (op 13696) 
v_sub_u32 v16, v1, v10 // (op 13696) 

v_lshlrev_b32 v10, 6, v16 // (op 13696) 

v_lshrrev_b32 v16, 1, v1 // (op 13696) 
v_lshl_add_u32 v18, v16, 8, v10 // (op 13696) 


v_mul_lo_u32 v10, v18, 6 // (op 13696) 

v_lshrrev_b32 v16, 3, v10 // (op 13696) 

v_and_b32 v10, -8, v1 // (op 13754) 
v_sub_u32 v18, v1, v10 // (op 13754) 

v_lshlrev_b32 v10, 4, v18 // (op 13754) 

v_lshrrev_b32 v18, 3, v1 // (op 13754) 
v_lshl_add_u32 v12, v18, 7, v10 // (op 13754) 


s_nop 0
v_readfirstlane_b32 s0, v12 // (op 13754) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -2, v1 // (op 13834) 
v_sub_u32 v18, v1, v10 // (op 13834) 

v_lshlrev_b32 v10, 6, v18 // (op 13834) 

v_lshrrev_b32 v18, 1, v1 // (op 13834) 
v_lshl_add_u32 v12, v18, 8, v10 // (op 13834) 


v_mul_lo_u32 v10, v12, 6 // (op 13834) 

v_lshrrev_b32 v18, 3, v10 // (op 13834) 

v_and_b32 v10, -8, v1 // (op 13863) 
v_sub_u32 v12, v1, v10 // (op 13863) 

v_lshlrev_b32 v10, 4, v12 // (op 13863) 

v_lshrrev_b32 v12, 3, v1 // (op 13863) 
v_lshl_add_u32 v13, v12, 7, v10 // (op 13863) 


s_nop 0
v_readfirstlane_b32 s1, v13 // (op 13863) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s6, 0 // (op 4435) call()
s_ashr_i64 s[10:11], s[36:37], 7 // (op 4434) 
s_mov_b32 s7, s10 // (op 4434) convert

s_and_b32 s10, s7, 4294967294 // (op 4434) 

s_cmp_lt_u32 s6, s10 // (op 4434) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopBottom_KLoop_4434 // (op 4434) Condition: Top (jump to RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopBottom_KLoop_4434 if false)
RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopTop_KLoop_4434:
s_add_u32 s6, s6, 2 // (op 4436) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14190) 



ds_read_b128 v[74:77], v3 offset:32768 // (op 60) Load local data 
ds_read_b64 v[78:79], v3 offset:32784 // (op 60) Load local data 
s_barrier  // (op 14203) 



ds_read_b128 v[80:83], v4 // (op 65) Load local data 
ds_read_b128 v[84:87], v4 offset:64 // (op 65) Load local data 



ds_read_b128 v[88:91], v4 offset:2048 // (op 308) Load local data 
ds_read_b128 v[92:95], v4 offset:2112 // (op 308) Load local data 
s_waitcnt vmcnt(14)
v_mov_b32 v12, v22 // (op 13415) reg expression



v_mov_b32 v13, v23 // (op 13415) reg expression
s_waitcnt vmcnt(12)
v_mov_b32 v96, v48 // (op 13442) reg expression
v_mov_b32 v97, v49 // (op 13442) reg expression
ds_read_b128 v[98:101], v4 offset:4096 // (op 444) Load local data 
ds_read_b128 v[102:105], v4 offset:4160 // (op 444) Load local data 
v_bfe_u32 v106, v96, 0, 8 // (op 12224) 



v_bfe_u32 v107, v96, 8, 8 // (op 12224) 
v_bfe_u32 v108, v96, 16, 8 // (op 12224) 
v_bfe_u32 v109, v96, 24, 8 // (op 12224) 
v_permlane16_swap_b32 v106, v107 // (op 12224) 
s_nop 0
v_permlane16_swap_b32 v108, v109 // (op 12224) Wait state hazard: VALU Write Readlane Hazard
ds_read_b128 v[110:113], v4 offset:6144 // (op 580) Load local data 
s_nop 0
v_permlane32_swap_b32 v106, v108 // (op 12224) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 12224) 
ds_read_b128 v[114:117], v4 offset:6208 // (op 580) Load local data 



v_bfe_u32 v118, v12, 0, 8 // (op 12351) 
v_bfe_u32 v119, v12, 8, 8 // (op 12351) 
v_bfe_u32 v120, v12, 16, 8 // (op 12351) 
v_bfe_u32 v121, v12, 24, 8 // (op 12351) 
v_permlane16_swap_b32 v118, v119 // (op 12351) 
ds_read_b128 v[122:125], v3 offset:34304 // (op 885) Load local data 
v_permlane16_swap_b32 v120, v121 // (op 12351) 
ds_read_b64 v[126:127], v3 offset:34320 // (op 885) Load local data 
s_nop 0
v_permlane32_swap_b32 v118, v120 // (op 12351) Wait state hazard: VALU Write Readlane Hazard



v_permlane32_swap_b32 v119, v121 // (op 12351) 
ds_read_b128 v[128:131], v3 offset:35840 // (op 1495) Load local data 
ds_read_b64 v[132:133], v3 offset:35856 // (op 1495) Load local data 



s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:79], v[80:87], a[0:3], v106, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 76) 
ds_read_b128 v[134:137], v3 offset:37376 // (op 2105) Load local data 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:79], v[88:95], a[4:7], v106, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 311) 
ds_read_b64 v[138:139], v3 offset:37392 // (op 2105) Load local data 



s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[74:79], v[98:105], a[8:11], v106, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 447) 
ds_read_b128 v[140:143], v3 offset:38912 // (op 2715) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:79], v[110:117], a[12:15], v106, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 583) 
ds_read_b64 v[144:145], v3 offset:38928 // (op 2715) Load local data 




s_barrier  // (op 14206) 
v_mov_b32 v74, v20 // (op 13428) reg expression

v_mov_b32 v75, v21 // (op 13428) reg expression
s_lshl_b64 s[10:11], s[24:25], 5 // (op 13479) 
v_mov_b64 v[76:77], s[10:11] // (op 13479) convert


s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[122:127], v[88:95], a[20:23], v107, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 965) 
v_mov_b64 v[78:79], v[14:15] // (op 13479) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[122:127], v[98:105], a[24:27], v107, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 927) 
v_mov_b64 v[146:147], v[78:79] // (op 13479) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[122:127], v[110:117], a[28:31], v107, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 889) 
s_mov_b32 m0, s0 // (op 13479) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[122:127], v[80:87], a[16:19], v107, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1003) 
s_mov_b32 s7, 256 // (op 13479) call()

buffer_load_dwordx4 v146, s[52:55], s7 offen offset: 0 lds // (op 13479) Load value direct to lds
v_add_co_u32 v146, s[10:11], v146, v76 // (op 13479) least significant half
v_addc_co_u32 v147, s[10:11], 0, v147, s[10:11] // (op 13479) most significant half

s_add_u32 m0, m0, 4096 // (op 13479) 
buffer_load_dwordx4 v146, s[52:55], s7 offen offset: 0 lds // (op 13479) Load value direct to lds
v_add_co_u32 v146, s[10:11], v146, v76 // (op 13479) least significant half
v_addc_co_u32 v147, s[10:11], 0, v147, s[10:11] // (op 13479) most significant half

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[128:133], v[80:87], a[32:35], v108, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1613) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[128:133], v[88:95], a[36:39], v108, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1575) 
s_add_u32 m0, m0, 4096 // (op 13479) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[128:133], v[98:105], a[40:43], v108, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1537) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[128:133], v[110:117], a[44:47], v108, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1499) 
buffer_load_dwordx4 v146, s[52:55], s7 offen offset: 0 lds // (op 13479) Load value direct to lds

v_add_co_u32 v146, s[10:11], v146, v76 // (op 13479) least significant half
v_addc_co_u32 v147, s[10:11], 0, v147, s[10:11] // (op 13479) most significant half

s_add_u32 m0, m0, 4096 // (op 13479) 
v_bfe_u32 v122, v74, 0, 8 // (op 12255) 
v_bfe_u32 v123, v74, 8, 8 // (op 12255) 
v_bfe_u32 v124, v74, 16, 8 // (op 12255) 
v_bfe_u32 v125, v74, 24, 8 // (op 12255) 
v_permlane16_swap_b32 v122, v123 // (op 12255) 
ds_read_b128 v[126:129], v3 offset:40448 // (op 3325) Load local data 
v_permlane16_swap_b32 v124, v125 // (op 12255) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[134:139], v[80:87], a[48:51], v109, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2223) 
s_nop 0
v_permlane32_swap_b32 v122, v124 // (op 12255) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[134:139], v[88:95], a[52:55], v109, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2185) 
ds_read_b64 v[130:131], v3 offset:40464 // (op 3325) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[134:139], v[98:105], a[56:59], v109, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2147) 
v_permlane32_swap_b32 v123, v125 // (op 12255) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[134:139], v[110:117], a[60:63], v109, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2109) 
ds_read_b128 v[148:151], v3 offset:41984 // (op 3935) Load local data 


ds_read_b64 v[152:153], v3 offset:42000 // (op 3935) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:145], v[88:95], a[68:71], v122, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2795) 
ds_read_b128 v[132:135], v3 offset:43520 // (op 4545) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[140:145], v[98:105], a[72:75], v122, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2757) 
ds_read_b64 v[136:137], v3 offset:43536 // (op 4545) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:145], v[110:117], a[76:79], v122, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2719) 
buffer_load_dwordx4 v146, s[52:55], s7 offen offset: 0 lds // (op 13479) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:145], v[80:87], a[64:67], v122, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2833) 
s_mul_i32 s11, s13, 6 // (op 4) most significant: low of high * low

s_mul_i32 s10, s12, 6 // (op 4) least significant
s_mul_hi_u32 s7, s12, 6 // (op 4) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 4) most significant: sum

v_lshrrev_b64 v[76:77], 3, s[10:11] // (op 4) 


v_mov_b32 v10, v11 // (op 4) 
v_mov_b32 v106, v10 // (op 4) 
buffer_load_dwordx4 v[24:27], v106, s[48:51], 0 offen offset:192 // (op 4) Load value
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:131], v[88:95], a[84:87], v123, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3405) 
buffer_load_dwordx2 v[28:29], v106, s[48:51], 0 offen offset:208 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:131], v[98:105], a[88:91], v123, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3367) 
buffer_load_dwordx4 v[30:33], v106, s[48:51], 0 offen offset:216 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[126:131], v[110:117], a[92:95], v123, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3329) 
buffer_load_dwordx2 v[34:35], v106, s[48:51], 0 offen offset:232 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:131], v[80:87], a[80:83], v123, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3443) 
v_add_u32 v106, v106, v76 // (op 4) 

buffer_load_dwordx4 v[36:39], v106, s[48:51], 0 offen offset:192 // (op 4) Load value
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:153], v[88:95], a[100:103], v124, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4015) 
buffer_load_dwordx2 v[40:41], v106, s[48:51], 0 offen offset:208 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[98:105], a[104:107], v124, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3977) 
buffer_load_dwordx4 v[42:45], v106, s[48:51], 0 offen offset:216 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:153], v[110:117], a[108:111], v124, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3939) 
buffer_load_dwordx2 v[46:47], v106, s[48:51], 0 offen offset:232 // (op 4) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:153], v[80:87], a[96:99], v124, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4053) 
s_barrier  // (op 14219) 




v_mov_b32 v76, v17 // (op 13423) 
v_add_u32 v77, 8, v76 // (op 13423) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:137], v[88:95], a[116:119], v125, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4625) 
s_waitcnt vmcnt(19)
ds_write_b128 v18, v[50:53] offset:57344 // (op 208) Store local data 

s_waitcnt vmcnt(18)
ds_write_b64 v18, v[54:55] offset:57360 // (op 208) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:137], v[98:105], a[120:123], v125, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4587) 
s_waitcnt vmcnt(17)
ds_write_b128 v18, v[56:59] offset:57368 // (op 208) Store local data 

s_waitcnt vmcnt(16)
ds_write_b64 v18, v[60:61] offset:57384 // (op 208) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:137], v[110:117], a[124:127], v125, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4549) 
s_waitcnt vmcnt(15)
ds_write_b128 v18, v[62:65] offset:57440 // (op 208) Store local data 

s_waitcnt vmcnt(14)
ds_write_b64 v18, v[66:67] offset:57456 // (op 208) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:137], v[80:87], a[112:115], v125, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4663) 
s_waitcnt vmcnt(13)
ds_write_b128 v18, v[68:71] offset:57464 // (op 208) Store local data 




s_waitcnt vmcnt(12)
ds_write_b64 v18, v[72:73] offset:57480 // (op 208) Store local data 
buffer_load_dwordx2 v[22:23], v77, s[60:63], 0 offen // (op 13423) Load value


v_add_u32 v17, 8, v17 // (op 13980) 


v_mov_b32 v76, v19 // (op 13437) 
s_lshl_b32 s7, s28, 6 // (op 13437) 
s_add_u32 s10, 8, s7 // (op 13437) 

v_add_u32 v77, v76, s10 // (op 13437) 


buffer_load_dwordx2 v[20:21], v77, s[56:59], 0 offen // (op 13437) Load value

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14216) 





v_add_u32 v77, 8, v76 // (op 13450) 

ds_read_b128 v[80:83], v5 offset:57344 // (op 210) Load local data 
ds_read_b64 v[84:85], v5 offset:57360 // (op 210) Load local data 
buffer_load_dwordx2 v[48:49], v77, s[56:59], 0 offen // (op 13450) Load value


s_barrier  // (op 14222) 



s_barrier  // (op 14225) 
v_add_u32 v19, 8, v19 // (op 13928) 

s_lshl_b64 s[10:11], s[24:25], 5 // (op 13490) 
v_mov_b64 v[76:77], s[10:11] // (op 13490) convert


v_mov_b64 v[86:87], v[78:79] // (op 13490) 
s_add_u32 s7, 16384, s1 // (op 13490) 
s_mov_b32 m0, s7 // (op 13490) reg expression
s_mov_b32 s10, 384 // (op 13490) call()
ds_read_b128 v[88:91], v6 offset:16384 // (op 212) Load local data 
ds_read_b128 v[92:95], v6 offset:16448 // (op 212) Load local data 
buffer_load_dwordx4 v86, s[52:55], s10 offen offset: 0 lds // (op 13490) Load value direct to lds



v_add_co_u32 v86, s[14:15], v86, v76 // (op 13490) least significant half
v_addc_co_u32 v87, s[14:15], 0, v87, s[14:15] // (op 13490) most significant half

s_add_u32 m0, m0, 4096 // (op 13490) 
v_bfe_u32 v98, v97, 0, 8 // (op 12239) 
v_bfe_u32 v99, v97, 8, 8 // (op 12239) 
v_bfe_u32 v100, v97, 16, 8 // (op 12239) 
v_bfe_u32 v101, v97, 24, 8 // (op 12239) 
v_permlane16_swap_b32 v98, v99 // (op 12239) 
buffer_load_dwordx4 v86, s[52:55], s10 offen offset: 0 lds // (op 13490) Load value direct to lds
v_permlane16_swap_b32 v100, v101 // (op 12239) 
v_add_co_u32 v86, s[14:15], v86, v76 // (op 13490) least significant half
v_addc_co_u32 v87, s[14:15], 0, v87, s[14:15] // (op 13490) most significant half

s_add_u32 m0, m0, 4096 // (op 13490) 
v_permlane32_swap_b32 v98, v100 // (op 12239) 
v_permlane32_swap_b32 v99, v101 // (op 12239) 
ds_read_b128 v[102:105], v6 offset:18432 // (op 320) Load local data 

ds_read_b128 v[106:109], v6 offset:18496 // (op 320) Load local data 



v_bfe_u32 v110, v13, 0, 8 // (op 12366) 
v_bfe_u32 v111, v13, 8, 8 // (op 12366) 
v_bfe_u32 v112, v13, 16, 8 // (op 12366) 
v_bfe_u32 v113, v13, 24, 8 // (op 12366) 
v_permlane16_swap_b32 v110, v111 // (op 12366) 
ds_read_b128 v[114:117], v6 offset:20480 // (op 456) Load local data 
v_permlane16_swap_b32 v112, v113 // (op 12366) 
s_nop 1
v_permlane32_swap_b32 v110, v112 // (op 12366) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v111, v113 // (op 12366) 
buffer_load_dwordx4 v86, s[52:55], s10 offen offset: 0 lds // (op 13490) Load value direct to lds

v_add_co_u32 v86, s[14:15], v86, v76 // (op 13490) least significant half
v_addc_co_u32 v87, s[14:15], 0, v87, s[14:15] // (op 13490) most significant half

s_add_u32 m0, m0, 4096 // (op 13490) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[80:85], v[88:95], a[0:3], v98, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 217) 
ds_read_b128 v[118:121], v6 offset:20544 // (op 456) Load local data 



ds_read_b128 v[122:125], v6 offset:22528 // (op 592) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[80:85], v[102:109], a[4:7], v98, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 312) 
ds_read_b128 v[126:129], v6 offset:22592 // (op 592) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[80:85], v[114:121], a[8:11], v98, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 448) 



ds_read_b128 v[130:133], v5 offset:58880 // (op 897) Load local data 
ds_read_b64 v[134:135], v5 offset:58896 // (op 897) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[80:85], v[122:129], a[12:15], v98, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 584) 




ds_read_b128 v[80:83], v5 offset:60416 // (op 1507) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[130:135], v[88:95], a[16:19], v99, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1004) 
ds_read_b64 v[84:85], v5 offset:60432 // (op 1507) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[130:135], v[114:121], a[24:27], v99, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 928) 
ds_read_b128 v[136:139], v5 offset:61952 // (op 2117) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[130:135], v[122:129], a[28:31], v99, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 890) 
ds_read_b64 v[140:141], v5 offset:61968 // (op 2117) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:135], v[102:109], a[20:23], v99, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 966) 
ds_read_b128 v[142:145], v5 offset:63488 // (op 2727) Load local data 

ds_read_b64 v[146:147], v5 offset:63504 // (op 2727) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:85], v[88:95], a[32:35], v100, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1614) 
v_bfe_u32 v130, v75, 0, 8 // (op 12270) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:85], v[114:121], a[40:43], v100, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1538) 
v_bfe_u32 v131, v75, 8, 8 // (op 12270) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:85], v[122:129], a[44:47], v100, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1500) 
v_bfe_u32 v132, v75, 16, 8 // (op 12270) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:85], v[102:109], a[36:39], v100, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1576) 
v_bfe_u32 v133, v75, 24, 8 // (op 12270) 

v_permlane16_swap_b32 v130, v131 // (op 12270) 
ds_read_b128 v[80:83], v5 offset:65024 // (op 3337) Load local data 
v_permlane16_swap_b32 v132, v133 // (op 12270) 
s_nop 1
v_permlane32_swap_b32 v130, v132 // (op 12270) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v131, v133 // (op 12270) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:141], v[88:95], a[48:51], v101, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2224) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:141], v[114:121], a[56:59], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2148) 
ds_read_b64 v[84:85], v5 offset:65040 // (op 3337) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:141], v[122:129], a[60:63], v101, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2110) 
v_add_u32 v12, 66560, v5 // (op 3947) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:141], v[102:109], a[52:55], v101, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2186) 
ds_read_b128 v[148:151], v12 // (op 3947) Load local data 


ds_read_b64 v[152:153], v12 offset:16 // (op 3947) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[142:147], v[102:109], a[68:71], v130, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2796) 
v_add_u32 v12, 68096, v5 // (op 4557) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[142:147], v[88:95], a[64:67], v130, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2834) 
ds_read_b128 v[96:99], v12 // (op 4557) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[142:147], v[114:121], a[72:75], v130, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2758) 
ds_read_b64 v[100:101], v12 offset:16 // (op 4557) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[142:147], v[122:129], a[76:79], v130, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2720) 

s_barrier  // (op 14325) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[80:85], v[102:109], a[84:87], v131, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3406) 
buffer_load_dwordx4 v86, s[52:55], s10 offen offset: 0 lds // (op 13490) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[80:85], v[88:95], a[80:83], v131, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3444) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[80:85], v[114:121], a[88:91], v131, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3368) 
v_add_co_u32 v14, vcc, 256, v14 // (op 13645) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[80:85], v[122:129], a[92:95], v131, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3330) 
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 13645) most significant half

s_mul_i32 s11, s13, 6 // (op 207) most significant: low of high * low
s_mul_i32 s10, s12, 6 // (op 207) least significant
s_mul_hi_u32 s7, s12, 6 // (op 207) most significant: high of low * low
s_add_u32 s11, s11, s7 // (op 207) most significant: sum

v_lshrrev_b64 v[12:13], 3, s[10:11] // (op 207) 


v_mov_b32 v74, v10 // (op 207) 
buffer_load_dwordx4 v[50:53], v74, s[48:51], 0 offen offset:288 // (op 207) Load value
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:153], v[102:109], a[100:103], v132, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4016) 
buffer_load_dwordx2 v[54:55], v74, s[48:51], 0 offen offset:304 // (op 207) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:153], v[88:95], a[96:99], v132, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4054) 
buffer_load_dwordx4 v[56:59], v74, s[48:51], 0 offen offset:312 // (op 207) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:153], v[114:121], a[104:107], v132, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3978) 
buffer_load_dwordx2 v[60:61], v74, s[48:51], 0 offen offset:328 // (op 207) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:153], v[122:129], a[108:111], v132, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3940) 
v_add_u32 v74, v74, v12 // (op 207) 

buffer_load_dwordx4 v[62:65], v74, s[48:51], 0 offen offset:288 // (op 207) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[96:101], v[102:109], a[116:119], v133, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4626) 
buffer_load_dwordx2 v[66:67], v74, s[48:51], 0 offen offset:304 // (op 207) Load value

buffer_load_dwordx4 v[68:71], v74, s[48:51], 0 offen offset:312 // (op 207) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[96:101], v[88:95], a[112:115], v133, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4664) 
buffer_load_dwordx2 v[72:73], v74, s[48:51], 0 offen offset:328 // (op 207) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[96:101], v[114:121], a[120:123], v133, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4588) 
s_barrier  // (op 14193) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[96:101], v[122:129], a[124:127], v133, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4550) 
v_add_u32 v11, 192, v11 // (op 13581) 




s_waitcnt vmcnt(22)
ds_write_b128 v16, v[24:27] offset:32768 // (op 6) Store local data 
s_waitcnt vmcnt(21)
ds_write_b64 v16, v[28:29] offset:32784 // (op 6) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v16, v[30:33] offset:32792 // (op 6) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v16, v[34:35] offset:32808 // (op 6) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v16, v[36:39] offset:32864 // (op 6) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v16, v[40:41] offset:32880 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v16, v[42:45] offset:32888 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v16, v[46:47] offset:32904 // (op 6) Store local data 
s_ashr_i64 s[10:11], s[36:37], 7 // (op 4434) 
s_mov_b32 s7, s10 // (op 4434) convert

s_and_b32 s10, s7, 4294967294 // (op 4434) 

s_cmp_lt_u32 s6, s10 // (op 4434) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopTop_KLoop_4434 // (op 4434) Condition: Bottom (jump to RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopTop_KLoop_4434 if true)
RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopBottom_KLoop_4434:












s_barrier  // (op 14303) 
v_and_b32 v10, 4294967294, v1 // (op 13513) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 13513) 
v_lshl_add_u32 v11, s16, 8, v10 // (op 13513) 

s_lshr_b32 s6, s0, 1 // (op 13513) 

v_and_b32 v10, -2, v1 // (op 13513) 
v_sub_u32 v12, v1, v10 // (op 13513) 

v_lshlrev_b32 v10, 6, v12 // (op 13513) 

v_lshl_add_u32 v12, s6, 8, v10 // (op 13513) 


v_mov_b32 v19, 0 // (op 13513) convert
v_mov_b32 v18, v12 // (op 13513) convert

v_mul_lo_u32 v13, v11, s13 // (op 13513) most significant: low of low * high
v_mul_hi_u32 v10, v11, s12 // (op 13513) most significant: high of low * low
v_mul_lo_u32 v12, v11, s12 // (op 13513) least significant: low of low * low
v_add_u32 v13, v13, v10 // (op 13513) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v18 // (op 13513) least significant half
v_addc_co_u32 v13, s[0:1], v13, v19, s[0:1] // (op 13513) most significant half



v_mul_lo_u32 v11, v13, 6 // (op 13513) most significant: low of high * low
v_mul_hi_u32 v14, v12, 6 // (op 13513) most significant: high of low * low
v_mul_lo_u32 v10, v12, 6 // (op 13513) least significant: low of low * low
v_add_u32 v11, v11, v14 // (op 13513) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 13513) 

v_mov_b32 v10, v12 // (op 13513) convert

s_mul_i32 s1, s23, 6 // (op 13515) most significant: low of high * low
s_mul_i32 s0, s22, 6 // (op 13515) least significant
s_mul_hi_u32 s6, s22, 6 // (op 13515) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13515) most significant: sum

s_mul_i32 s7, s12, s1 // (op 13515) most significant: low of low * high
s_mul_i32 s10, s13, s0 // (op 13515) most significant: low of high * low
s_mul_i32 s6, s12, s0 // (op 13515) least significant
s_mul_hi_u32 s11, s12, s0 // (op 13515) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 13515) most significant: sum
s_add_u32 s7, s7, s11 // (op 13515) most significant: sum



s_ashr_i64 s[0:1], s[6:7], 3 // (op 13515) 

s_mov_b32 s48, s20 // (op 13515) reg expression
s_mov_b32 s49, s21 // (op 13515) reg expression
s_mov_b32 s50, s0 // (op 13515) reg expression
s_mov_b32 s51, 131072 // (op 13515) call()

v_and_b32 v11, -8, v1 // (op 13611) 
s_ashr_i64 s[0:1], s[36:37], 7 // (op 13611) 
v_sub_u32 v12, v1, v11 // (op 13611) 

s_lshr_b32 s6, s0, 1 // (op 13611) 

v_lshlrev_b32 v11, 4, v12 // (op 13611) 

v_lshl_add_u32 v12, s6, 8, v11 // (op 13611) 


v_mov_b32 v19, 0 // (op 13611) convert
v_mov_b32 v18, v12 // (op 13611) convert

v_lshrrev_b32 v11, 3, v1 // (op 13611) 
v_lshl_add_u32 v12, s17, 7, v11 // (op 13611) 

v_mul_lo_u32 v17, v12, s25 // (op 13611) most significant: low of low * high
v_mul_hi_u32 v11, v12, s24 // (op 13611) most significant: high of low * low
v_mul_lo_u32 v16, v12, s24 // (op 13611) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 13611) most significant: sum


v_add_co_u32 v14, s[0:1], v18, v16 // (op 13611) least significant half
v_addc_co_u32 v15, s[0:1], v19, v17, s[0:1] // (op 13611) most significant half



s_mul_i32 s1, s24, s9 // (op 13613) most significant: low of low * high
s_mul_i32 s6, s25, s8 // (op 13613) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 13613) least significant
s_mul_hi_u32 s7, s24, s8 // (op 13613) most significant: high of low * low
s_add_u32 s1, s1, s6 // (op 13613) most significant: sum
s_add_u32 s1, s1, s7 // (op 13613) most significant: sum


s_mov_b32 s52, s26 // (op 13613) reg expression
s_mov_b32 s53, s27 // (op 13613) reg expression
s_mov_b32 s54, s0 // (op 13613) reg expression
s_mov_b32 s55, 131072 // (op 13613) call()

v_and_b32 v11, -2, v1 // (op 13709) 
v_sub_u32 v12, v1, v11 // (op 13709) 

v_lshlrev_b32 v11, 6, v12 // (op 13709) 

v_lshrrev_b32 v12, 1, v1 // (op 13709) 
v_lshl_add_u32 v13, v12, 8, v11 // (op 13709) 


v_mul_lo_u32 v11, v13, 6 // (op 13709) 

v_lshrrev_b32 v12, 3, v11 // (op 13709) 

v_and_b32 v11, -8, v1 // (op 13767) 
v_sub_u32 v13, v1, v11 // (op 13767) 

v_lshlrev_b32 v11, 4, v13 // (op 13767) 

v_lshrrev_b32 v13, 3, v1 // (op 13767) 
v_lshl_add_u32 v16, v13, 7, v11 // (op 13767) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 13767) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v11, 63, v1 // (op 14017) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 14017) 
v_lshrrev_b32 v13, 4, v11 // (op 14017) 
s_lshr_b32 s1, s6, 1 // (op 14017) 

v_and_b32 v16, 3, v13 // (op 14017) 

v_and_b32 v13, 15, v1 // (op 14017) 
v_lshl_add_u32 v17, v16, 4, v13 // (op 14017) 


v_and_b32 v13, 4294967168, v1 // (op 14017) 
v_add_u32 v16, v13, v17 // (op 14017) 


v_lshl_add_u32 v13, s16, 8, v16 // (op 14017) 

v_lshrrev_b32 v16, 6, v11 // (op 14017) 

v_lshlrev_b32 v11, 2, v16 // (op 14017) 

v_lshl_add_u32 v16, s1, 3, v11 // (op 14017) 


v_mul_lo_u32 v11, v13, s28 // (op 14017) 
v_add_u32 v11, v11, v16 // (op 14017) 


s_mul_i32 s7, s28, s23 // (op 14019) most significant: low of low * high
s_mul_i32 s1, s29, s22 // (op 14019) most significant: low of high * low
s_mul_i32 s6, s28, s22 // (op 14019) least significant
s_mul_hi_u32 s10, s28, s22 // (op 14019) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 14019) most significant: sum
s_add_u32 s7, s7, s10 // (op 14019) most significant: sum


s_mov_b32 s56, s30 // (op 14019) reg expression
s_mov_b32 s57, s31 // (op 14019) reg expression
s_mov_b32 s58, s6 // (op 14019) reg expression
s_mov_b32 s59, 131072 // (op 14019) call()

v_and_b32 v13, 63, v1 // (op 14044) 
s_ashr_i64 s[6:7], s[36:37], 7 // (op 14044) 
v_lshrrev_b32 v16, 6, v13 // (op 14044) 
s_lshr_b32 s1, s6, 1 // (op 14044) 

v_lshlrev_b32 v17, 2, v16 // (op 14044) 

v_lshl_add_u32 v16, s1, 3, v17 // (op 14044) 


v_lshrrev_b32 v17, 4, v13 // (op 14044) 

v_and_b32 v13, 3, v17 // (op 14044) 

v_and_b32 v17, 15, v1 // (op 14044) 
v_lshl_add_u32 v18, v13, 4, v17 // (op 14044) 


v_lshrrev_b32 v13, 6, v1 // (op 14044) 
v_and_b32 v17, 1, v13 // (op 14044) 

v_lshl_add_u32 v13, v17, 6, v18 // (op 14044) 


v_lshl_add_u32 v17, s17, 7, v13 // (op 14044) 

v_mul_lo_u32 v13, v17, s32 // (op 14044) 

v_add_u32 v17, v16, v13 // (op 14044) 


s_mul_i32 s7, s32, s9 // (op 14046) most significant: low of low * high
s_mul_i32 s1, s33, s8 // (op 14046) most significant: low of high * low
s_mul_i32 s6, s32, s8 // (op 14046) least significant
s_mul_hi_u32 s10, s32, s8 // (op 14046) most significant: high of low * low
s_add_u32 s7, s7, s1 // (op 14046) most significant: sum
s_add_u32 s7, s7, s10 // (op 14046) most significant: sum


s_mov_b32 s60, s34 // (op 14046) reg expression
s_mov_b32 s61, s35 // (op 14046) reg expression
s_mov_b32 s62, s6 // (op 14046) reg expression
s_mov_b32 s63, 131072 // (op 14046) call()

s_ashr_i64 s[6:7], s[36:37], 7 // (op 4521) 
s_and_b32 s1, s6, 4294967294 // (op 4521) 

s_ashr_i64 s[6:7], s[36:37], 7 // (op 4520) 
s_mov_b32 s10, s6 // (op 4520) convert

s_cmp_lt_u32 s1, s10 // (op 4520) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopBottom_KLoopTail_4520 // (op 4520) Condition: Top (jump to RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopBottom_KLoopTail_4520 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopTop_KLoopTail_4520:

s_add_u32 s1, s1, 1 // (op 4522) 
s_mul_i32 s7, s13, 6 // (op 157) most significant: low of high * low
s_mul_i32 s6, s12, 6 // (op 157) least significant
s_mul_hi_u32 s10, s12, 6 // (op 157) most significant: high of low * low
s_add_u32 s7, s7, s10 // (op 157) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[6:7] // (op 157) 


v_mov_b32 v13, v10 // (op 157) 
v_mov_b32 v16, v13 // (op 157) 
buffer_load_dwordx4 v[20:23], v16, s[48:51], 0 offen // (op 157) Load value
buffer_load_dwordx2 v[24:25], v16, s[48:51], 0 offen offset:16 // (op 157) Load value
buffer_load_dwordx4 v[26:29], v16, s[48:51], 0 offen offset:24 // (op 157) Load value
buffer_load_dwordx2 v[30:31], v16, s[48:51], 0 offen offset:40 // (op 157) Load value
v_add_u32 v16, v16, v18 // (op 157) 
buffer_load_dwordx4 v[32:35], v16, s[48:51], 0 offen // (op 157) Load value
buffer_load_dwordx2 v[36:37], v16, s[48:51], 0 offen offset:16 // (op 157) Load value
buffer_load_dwordx4 v[38:41], v16, s[48:51], 0 offen offset:24 // (op 157) Load value
buffer_load_dwordx2 v[42:43], v16, s[48:51], 0 offen offset:40 // (op 157) Load value



v_add_u32 v10, 96, v10 // (op 13503) 
s_barrier  // (op 14196) 
s_barrier  // (op 14209) 




s_lshl_b64 s[6:7], s[24:25], 5 // (op 13484) 
v_mov_b64 v[18:19], s[6:7] // (op 13484) convert


v_mov_b64 v[44:45], v[14:15] // (op 13484) 
v_mov_b64 v[46:47], v[44:45] // (op 13484) 
s_mov_b32 m0, s0 // (op 13484) reg expression
buffer_load_dwordx4 v46, s[52:55], 0 offen offset: 0 lds // (op 13484) Load value direct to lds
v_add_co_u32 v46, s[6:7], v46, v18 // (op 13484) least significant half
v_addc_co_u32 v47, s[6:7], 0, v47, s[6:7] // (op 13484) most significant half

s_add_u32 m0, m0, 4096 // (op 13484) 
s_waitcnt vmcnt(8)
ds_write_b128 v12, v[20:23] offset:32768 // (op 158) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v12, v[24:25] offset:32784 // (op 158) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v12, v[26:29] offset:32792 // (op 158) Store local data 
s_waitcnt vmcnt(5)
ds_write_b64 v12, v[30:31] offset:32808 // (op 158) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v12, v[32:35] offset:32864 // (op 158) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v12, v[36:37] offset:32880 // (op 158) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v12, v[38:41] offset:32888 // (op 158) Store local data 
buffer_load_dwordx4 v46, s[52:55], 0 offen offset: 0 lds // (op 13484) Load value direct to lds
v_add_co_u32 v46, s[6:7], v46, v18 // (op 13484) least significant half
v_addc_co_u32 v47, s[6:7], 0, v47, s[6:7] // (op 13484) most significant half

s_add_u32 m0, m0, 4096 // (op 13484) 
buffer_load_dwordx4 v46, s[52:55], 0 offen offset: 0 lds // (op 13484) Load value direct to lds
v_add_co_u32 v46, s[6:7], v46, v18 // (op 13484) least significant half
v_addc_co_u32 v47, s[6:7], 0, v47, s[6:7] // (op 13484) most significant half

s_add_u32 m0, m0, 4096 // (op 13484) 
buffer_load_dwordx4 v46, s[52:55], 0 offen offset: 0 lds // (op 13484) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 13603) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 13603) most significant half
s_waitcnt vmcnt(4)
ds_write_b64 v12, v[42:43] offset:32904 // (op 158) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14199) 



ds_read_b128 v[18:21], v3 offset:32768 // (op 160) Load local data 
ds_read_b64 v[22:23], v3 offset:32784 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 14212) 



ds_read_b128 v[24:27], v4 // (op 162) Load local data 
ds_read_b128 v[28:31], v4 offset:64 // (op 162) Load local data 


v_mov_b32 v13, v11 // (op 164) 
v_mov_b32 v16, v13 // (op 164) reg expression

buffer_load_dwordx2 v[32:33], v16, s[56:59], 0 offen // (op 164) Load value



v_mov_b32 v16, v17 // (op 166) 
v_mov_b32 v34, v16 // (op 166) reg expression

buffer_load_dwordx2 v[36:37], v34, s[60:63], 0 offen // (op 166) Load value


v_add_u32 v17, 4, v17 // (op 14034) 
s_waitcnt vmcnt(1)
v_bfe_u32 v38, v32, 0, 8 // (op 12447) 



v_bfe_u32 v39, v32, 8, 8 // (op 12447) 
v_bfe_u32 v40, v32, 16, 8 // (op 12447) 
v_bfe_u32 v41, v32, 24, 8 // (op 12447) 
v_permlane16_swap_b32 v38, v39 // (op 12447) 
ds_read_b128 v[42:45], v4 offset:2048 // (op 291) Load local data 
ds_read_b128 v[46:49], v4 offset:2112 // (op 291) Load local data 
v_permlane16_swap_b32 v40, v41 // (op 12447) 



s_nop 1
v_permlane32_swap_b32 v38, v40 // (op 12447) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v39, v41 // (op 12447) 
ds_read_b128 v[50:53], v4 offset:4096 // (op 427) Load local data 

ds_read_b128 v[54:57], v4 offset:4160 // (op 427) Load local data 



ds_read_b128 v[58:61], v4 offset:6144 // (op 563) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v32, v36, 0, 8 // (op 12507) 
v_bfe_u32 v33, v36, 8, 8 // (op 12507) 
v_bfe_u32 v34, v36, 16, 8 // (op 12507) 
v_bfe_u32 v35, v36, 24, 8 // (op 12507) 
v_permlane16_swap_b32 v32, v33 // (op 12507) 
s_nop 0
v_permlane16_swap_b32 v34, v35 // (op 12507) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v32, v34 // (op 12507) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v33, v35 // (op 12507) 
ds_read_b128 v[62:65], v4 offset:6208 // (op 563) Load local data 




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:23], v[24:31], a[0:3], v38, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 167) 
ds_read_b128 v[66:69], v3 offset:34304 // (op 785) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:23], v[42:49], a[4:7], v38, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 294) 
ds_read_b64 v[70:71], v3 offset:34320 // (op 785) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:23], v[50:57], a[8:11], v38, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 430) 
ds_read_b128 v[72:75], v3 offset:35840 // (op 1395) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:23], v[58:65], a[12:15], v38, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 566) 
ds_read_b64 v[76:77], v3 offset:35856 // (op 1395) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[66:71], v[24:31], a[16:19], v39, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 789) 
ds_read_b128 v[18:21], v3 offset:37376 // (op 2005) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[66:71], v[50:57], a[24:27], v39, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 843) 
ds_read_b64 v[22:23], v3 offset:37392 // (op 2005) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[66:71], v[58:65], a[28:31], v39, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 870) 
ds_read_b128 v[78:81], v3 offset:38912 // (op 2615) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:71], v[42:49], a[20:23], v39, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 816) 
ds_read_b64 v[82:83], v3 offset:38928 // (op 2615) Load local data 



s_lshl_b32 s6, s28, 6 // (op 2624) 
v_add_u32 v16, v13, s6 // (op 2624) 


buffer_load_dwordx2 v[36:37], v16, s[56:59], 0 offen // (op 2624) Load value

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:77], v[24:31], a[32:35], v40, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1399) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[72:77], v[50:57], a[40:43], v40, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1453) 
v_add_u32 v11, 4, v11 // (op 14007) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:77], v[58:65], a[44:47], v40, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1480) 
s_waitcnt vmcnt(0)
v_bfe_u32 v66, v36, 0, 8 // (op 12461) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[72:77], v[42:49], a[36:39], v40, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 1426) 
v_bfe_u32 v67, v36, 8, 8 // (op 12461) 

v_bfe_u32 v68, v36, 16, 8 // (op 12461) 
v_bfe_u32 v69, v36, 24, 8 // (op 12461) 
v_permlane16_swap_b32 v66, v67 // (op 12461) 
s_nop 0
v_permlane16_swap_b32 v68, v69 // (op 12461) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v66, v68 // (op 12461) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v67, v69 // (op 12461) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[18:23], v[24:31], a[48:51], v41, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2009) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:23], v[50:57], a[56:59], v41, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2063) 
ds_read_b128 v[70:73], v3 offset:40448 // (op 3225) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:23], v[58:65], a[60:63], v41, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2090) 
ds_read_b64 v[74:75], v3 offset:40464 // (op 3225) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:23], v[42:49], a[52:55], v41, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2036) 
ds_read_b128 v[84:87], v3 offset:41984 // (op 3835) Load local data 


ds_read_b64 v[88:89], v3 offset:42000 // (op 3835) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:83], v[42:49], a[68:71], v66, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2646) 
ds_read_b128 v[18:21], v3 offset:43520 // (op 4445) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:83], v[24:31], a[64:67], v66, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2619) 
ds_read_b64 v[22:23], v3 offset:43536 // (op 4445) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:83], v[50:57], a[72:75], v66, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2673) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[78:83], v[58:65], a[76:79], v66, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 2700) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:75], v[42:49], a[84:87], v67, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3256) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:75], v[24:31], a[80:83], v67, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3229) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:75], v[50:57], a[88:91], v67, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3283) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:75], v[58:65], a[92:95], v67, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3310) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[84:89], v[42:49], a[100:103], v68, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3866) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[84:89], v[24:31], a[96:99], v68, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3839) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[84:89], v[50:57], a[104:107], v68, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3893) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[84:89], v[58:65], a[108:111], v68, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 3920) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[18:23], v[42:49], a[116:119], v69, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4476) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[18:23], v[24:31], a[112:115], v69, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4449) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[18:23], v[50:57], a[120:123], v69, v34 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4503) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[18:23], v[58:65], a[124:127], v69, v35 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b011 blgp:0b000 // (op 4530) 




s_ashr_i64 s[6:7], s[36:37], 7 // (op 4520) 
s_mov_b32 s10, s6 // (op 4520) convert

s_cmp_lt_u32 s1, s10 // (op 4520) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopTop_KLoopTail_4520 // (op 4520) Condition: Bottom (jump to RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopTop_KLoopTail_4520 if true)
RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ForLoopBottom_KLoopTail_4520:





















s_barrier  // (op 14313) 





v_cmp_eq_f32 vcc, 0.00000, v0 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ConditionalFalse_0_beta_114
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


v_accvgpr_read v10, a16 // (op 5704) 
v_accvgpr_read v11, a17 // (op 5704) 
v_accvgpr_read v12, a18 // (op 5704) 
v_accvgpr_read v13, a19 // (op 5704) 
v_mul_f32 v14, v7, v10 // (op 5704) 
v_mul_f32 v15, v7, v11 // (op 5704) 
v_mul_f32 v16, v7, v12 // (op 5704) 
v_mul_f32 v17, v7, v13 // (op 5704) 



v_add_u32 v3, 64, v9 // (op 5707) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5707) Store value


v_accvgpr_read v10, a20 // (op 5717) 
v_accvgpr_read v11, a21 // (op 5717) 
v_accvgpr_read v12, a22 // (op 5717) 
v_accvgpr_read v13, a23 // (op 5717) 
v_mul_f32 v14, v7, v10 // (op 5717) 
v_mul_f32 v15, v7, v11 // (op 5717) 
v_mul_f32 v16, v7, v12 // (op 5717) 
v_mul_f32 v17, v7, v13 // (op 5717) 



s_lshl_b32 s0, s4, 6 // (op 5709) 
s_add_u32 s0, s0, 64 // (op 5709) 
v_add_u32 v3, v9, s0 // (op 5709) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5709) Store value


v_accvgpr_read v10, a24 // (op 5722) 
v_accvgpr_read v11, a25 // (op 5722) 
v_accvgpr_read v12, a26 // (op 5722) 
v_accvgpr_read v13, a27 // (op 5722) 
v_mul_f32 v14, v7, v10 // (op 5722) 
v_mul_f32 v15, v7, v11 // (op 5722) 
v_mul_f32 v16, v7, v12 // (op 5722) 
v_mul_f32 v17, v7, v13 // (op 5722) 



s_lshl_b32 s0, s4, 7 // (op 5711) 
s_add_u32 s0, s0, 64 // (op 5711) 
v_add_u32 v3, v9, s0 // (op 5711) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5711) Store value


v_accvgpr_read v10, a28 // (op 5727) 
v_accvgpr_read v11, a29 // (op 5727) 
v_accvgpr_read v12, a30 // (op 5727) 
v_accvgpr_read v13, a31 // (op 5727) 
v_mul_f32 v14, v7, v10 // (op 5727) 
v_mul_f32 v15, v7, v11 // (op 5727) 
v_mul_f32 v16, v7, v12 // (op 5727) 
v_mul_f32 v17, v7, v13 // (op 5727) 



s_lshl_b32 s0, s4, 6 // (op 5713) 
s_mul_i32 s1, 3, s0 // (op 5713) 
s_add_u32 s1, s1, 64 // (op 5713) 

v_add_u32 v3, v9, s1 // (op 5713) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5713) Store value


v_accvgpr_read v10, a32 // (op 5772) 
v_accvgpr_read v11, a33 // (op 5772) 
v_accvgpr_read v12, a34 // (op 5772) 
v_accvgpr_read v13, a35 // (op 5772) 
v_mul_f32 v14, v7, v10 // (op 5772) 
v_mul_f32 v15, v7, v11 // (op 5772) 
v_mul_f32 v16, v7, v12 // (op 5772) 
v_mul_f32 v17, v7, v13 // (op 5772) 



v_add_u32 v3, 128, v9 // (op 5775) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5775) Store value


v_accvgpr_read v10, a36 // (op 5785) 
v_accvgpr_read v11, a37 // (op 5785) 
v_accvgpr_read v12, a38 // (op 5785) 
v_accvgpr_read v13, a39 // (op 5785) 
v_mul_f32 v14, v7, v10 // (op 5785) 
v_mul_f32 v15, v7, v11 // (op 5785) 
v_mul_f32 v16, v7, v12 // (op 5785) 
v_mul_f32 v17, v7, v13 // (op 5785) 



s_lshl_b32 s0, s4, 6 // (op 5777) 
s_add_u32 s0, s0, 128 // (op 5777) 
v_add_u32 v3, v9, s0 // (op 5777) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5777) Store value


v_accvgpr_read v10, a40 // (op 5790) 
v_accvgpr_read v11, a41 // (op 5790) 
v_accvgpr_read v12, a42 // (op 5790) 
v_accvgpr_read v13, a43 // (op 5790) 
v_mul_f32 v14, v7, v10 // (op 5790) 
v_mul_f32 v15, v7, v11 // (op 5790) 
v_mul_f32 v16, v7, v12 // (op 5790) 
v_mul_f32 v17, v7, v13 // (op 5790) 



s_lshl_b32 s0, s4, 7 // (op 5779) 
s_add_u32 s0, s0, 128 // (op 5779) 
v_add_u32 v3, v9, s0 // (op 5779) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5779) Store value


v_accvgpr_read v10, a44 // (op 5795) 
v_accvgpr_read v11, a45 // (op 5795) 
v_accvgpr_read v12, a46 // (op 5795) 
v_accvgpr_read v13, a47 // (op 5795) 
v_mul_f32 v14, v7, v10 // (op 5795) 
v_mul_f32 v15, v7, v11 // (op 5795) 
v_mul_f32 v16, v7, v12 // (op 5795) 
v_mul_f32 v17, v7, v13 // (op 5795) 



s_lshl_b32 s0, s4, 6 // (op 5781) 
s_mul_i32 s1, 3, s0 // (op 5781) 
s_add_u32 s1, s1, 128 // (op 5781) 

v_add_u32 v3, v9, s1 // (op 5781) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5781) Store value


v_accvgpr_read v10, a48 // (op 5840) 
v_accvgpr_read v11, a49 // (op 5840) 
v_accvgpr_read v12, a50 // (op 5840) 
v_accvgpr_read v13, a51 // (op 5840) 
v_mul_f32 v14, v7, v10 // (op 5840) 
v_mul_f32 v15, v7, v11 // (op 5840) 
v_mul_f32 v16, v7, v12 // (op 5840) 
v_mul_f32 v17, v7, v13 // (op 5840) 



v_add_u32 v3, 192, v9 // (op 5843) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5843) Store value


v_accvgpr_read v10, a52 // (op 5853) 
v_accvgpr_read v11, a53 // (op 5853) 
v_accvgpr_read v12, a54 // (op 5853) 
v_accvgpr_read v13, a55 // (op 5853) 
v_mul_f32 v14, v7, v10 // (op 5853) 
v_mul_f32 v15, v7, v11 // (op 5853) 
v_mul_f32 v16, v7, v12 // (op 5853) 
v_mul_f32 v17, v7, v13 // (op 5853) 



s_lshl_b32 s0, s4, 6 // (op 5845) 
s_add_u32 s0, s0, 192 // (op 5845) 
v_add_u32 v3, v9, s0 // (op 5845) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5845) Store value


v_accvgpr_read v10, a56 // (op 5858) 
v_accvgpr_read v11, a57 // (op 5858) 
v_accvgpr_read v12, a58 // (op 5858) 
v_accvgpr_read v13, a59 // (op 5858) 
v_mul_f32 v14, v7, v10 // (op 5858) 
v_mul_f32 v15, v7, v11 // (op 5858) 
v_mul_f32 v16, v7, v12 // (op 5858) 
v_mul_f32 v17, v7, v13 // (op 5858) 



s_lshl_b32 s0, s4, 7 // (op 5847) 
s_add_u32 s0, s0, 192 // (op 5847) 
v_add_u32 v3, v9, s0 // (op 5847) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5847) Store value


v_accvgpr_read v10, a60 // (op 5863) 
v_accvgpr_read v11, a61 // (op 5863) 
v_accvgpr_read v12, a62 // (op 5863) 
v_accvgpr_read v13, a63 // (op 5863) 
v_mul_f32 v14, v7, v10 // (op 5863) 
v_mul_f32 v15, v7, v11 // (op 5863) 
v_mul_f32 v16, v7, v12 // (op 5863) 
v_mul_f32 v17, v7, v13 // (op 5863) 



s_lshl_b32 s0, s4, 6 // (op 5849) 
s_mul_i32 s1, 3, s0 // (op 5849) 
s_add_u32 s1, s1, 192 // (op 5849) 

v_add_u32 v3, v9, s1 // (op 5849) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5849) Store value


v_accvgpr_read v10, a64 // (op 5908) 
v_accvgpr_read v11, a65 // (op 5908) 
v_accvgpr_read v12, a66 // (op 5908) 
v_accvgpr_read v13, a67 // (op 5908) 
v_mul_f32 v14, v7, v10 // (op 5908) 
v_mul_f32 v15, v7, v11 // (op 5908) 
v_mul_f32 v16, v7, v12 // (op 5908) 
v_mul_f32 v17, v7, v13 // (op 5908) 



v_add_u32 v3, 256, v9 // (op 5911) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5911) Store value


v_accvgpr_read v10, a68 // (op 5921) 
v_accvgpr_read v11, a69 // (op 5921) 
v_accvgpr_read v12, a70 // (op 5921) 
v_accvgpr_read v13, a71 // (op 5921) 
v_mul_f32 v14, v7, v10 // (op 5921) 
v_mul_f32 v15, v7, v11 // (op 5921) 
v_mul_f32 v16, v7, v12 // (op 5921) 
v_mul_f32 v17, v7, v13 // (op 5921) 



s_lshl_b32 s0, s4, 6 // (op 5913) 
s_add_u32 s0, s0, 256 // (op 5913) 
v_add_u32 v3, v9, s0 // (op 5913) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5913) Store value


v_accvgpr_read v10, a72 // (op 5926) 
v_accvgpr_read v11, a73 // (op 5926) 
v_accvgpr_read v12, a74 // (op 5926) 
v_accvgpr_read v13, a75 // (op 5926) 
v_mul_f32 v14, v7, v10 // (op 5926) 
v_mul_f32 v15, v7, v11 // (op 5926) 
v_mul_f32 v16, v7, v12 // (op 5926) 
v_mul_f32 v17, v7, v13 // (op 5926) 



s_lshl_b32 s0, s4, 7 // (op 5915) 
s_add_u32 s0, s0, 256 // (op 5915) 
v_add_u32 v3, v9, s0 // (op 5915) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5915) Store value


v_accvgpr_read v10, a76 // (op 5931) 
v_accvgpr_read v11, a77 // (op 5931) 
v_accvgpr_read v12, a78 // (op 5931) 
v_accvgpr_read v13, a79 // (op 5931) 
v_mul_f32 v14, v7, v10 // (op 5931) 
v_mul_f32 v15, v7, v11 // (op 5931) 
v_mul_f32 v16, v7, v12 // (op 5931) 
v_mul_f32 v17, v7, v13 // (op 5931) 



s_lshl_b32 s0, s4, 6 // (op 5917) 
s_mul_i32 s1, 3, s0 // (op 5917) 
s_add_u32 s1, s1, 256 // (op 5917) 

v_add_u32 v3, v9, s1 // (op 5917) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5917) Store value


v_accvgpr_read v10, a80 // (op 5976) 
v_accvgpr_read v11, a81 // (op 5976) 
v_accvgpr_read v12, a82 // (op 5976) 
v_accvgpr_read v13, a83 // (op 5976) 
v_mul_f32 v14, v7, v10 // (op 5976) 
v_mul_f32 v15, v7, v11 // (op 5976) 
v_mul_f32 v16, v7, v12 // (op 5976) 
v_mul_f32 v17, v7, v13 // (op 5976) 



v_add_u32 v3, 320, v9 // (op 5979) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5979) Store value


v_accvgpr_read v10, a84 // (op 5989) 
v_accvgpr_read v11, a85 // (op 5989) 
v_accvgpr_read v12, a86 // (op 5989) 
v_accvgpr_read v13, a87 // (op 5989) 
v_mul_f32 v14, v7, v10 // (op 5989) 
v_mul_f32 v15, v7, v11 // (op 5989) 
v_mul_f32 v16, v7, v12 // (op 5989) 
v_mul_f32 v17, v7, v13 // (op 5989) 



s_lshl_b32 s0, s4, 6 // (op 5981) 
s_add_u32 s0, s0, 320 // (op 5981) 
v_add_u32 v3, v9, s0 // (op 5981) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5981) Store value


v_accvgpr_read v10, a88 // (op 5994) 
v_accvgpr_read v11, a89 // (op 5994) 
v_accvgpr_read v12, a90 // (op 5994) 
v_accvgpr_read v13, a91 // (op 5994) 
v_mul_f32 v14, v7, v10 // (op 5994) 
v_mul_f32 v15, v7, v11 // (op 5994) 
v_mul_f32 v16, v7, v12 // (op 5994) 
v_mul_f32 v17, v7, v13 // (op 5994) 



s_lshl_b32 s0, s4, 7 // (op 5983) 
s_add_u32 s0, s0, 320 // (op 5983) 
v_add_u32 v3, v9, s0 // (op 5983) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5983) Store value


v_accvgpr_read v10, a92 // (op 5999) 
v_accvgpr_read v11, a93 // (op 5999) 
v_accvgpr_read v12, a94 // (op 5999) 
v_accvgpr_read v13, a95 // (op 5999) 
v_mul_f32 v14, v7, v10 // (op 5999) 
v_mul_f32 v15, v7, v11 // (op 5999) 
v_mul_f32 v16, v7, v12 // (op 5999) 
v_mul_f32 v17, v7, v13 // (op 5999) 



s_lshl_b32 s0, s4, 6 // (op 5985) 
s_mul_i32 s1, 3, s0 // (op 5985) 
s_add_u32 s1, s1, 320 // (op 5985) 

v_add_u32 v3, v9, s1 // (op 5985) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 5985) Store value


v_accvgpr_read v10, a96 // (op 6044) 
v_accvgpr_read v11, a97 // (op 6044) 
v_accvgpr_read v12, a98 // (op 6044) 
v_accvgpr_read v13, a99 // (op 6044) 
v_mul_f32 v14, v7, v10 // (op 6044) 
v_mul_f32 v15, v7, v11 // (op 6044) 
v_mul_f32 v16, v7, v12 // (op 6044) 
v_mul_f32 v17, v7, v13 // (op 6044) 



v_add_u32 v3, 384, v9 // (op 6047) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6047) Store value


v_accvgpr_read v10, a100 // (op 6057) 
v_accvgpr_read v11, a101 // (op 6057) 
v_accvgpr_read v12, a102 // (op 6057) 
v_accvgpr_read v13, a103 // (op 6057) 
v_mul_f32 v14, v7, v10 // (op 6057) 
v_mul_f32 v15, v7, v11 // (op 6057) 
v_mul_f32 v16, v7, v12 // (op 6057) 
v_mul_f32 v17, v7, v13 // (op 6057) 



s_lshl_b32 s0, s4, 6 // (op 6049) 
s_add_u32 s0, s0, 384 // (op 6049) 
v_add_u32 v3, v9, s0 // (op 6049) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6049) Store value


v_accvgpr_read v10, a104 // (op 6062) 
v_accvgpr_read v11, a105 // (op 6062) 
v_accvgpr_read v12, a106 // (op 6062) 
v_accvgpr_read v13, a107 // (op 6062) 
v_mul_f32 v14, v7, v10 // (op 6062) 
v_mul_f32 v15, v7, v11 // (op 6062) 
v_mul_f32 v16, v7, v12 // (op 6062) 
v_mul_f32 v17, v7, v13 // (op 6062) 



s_lshl_b32 s0, s4, 7 // (op 6051) 
s_add_u32 s0, s0, 384 // (op 6051) 
v_add_u32 v3, v9, s0 // (op 6051) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6051) Store value


v_accvgpr_read v10, a108 // (op 6067) 
v_accvgpr_read v11, a109 // (op 6067) 
v_accvgpr_read v12, a110 // (op 6067) 
v_accvgpr_read v13, a111 // (op 6067) 
v_mul_f32 v14, v7, v10 // (op 6067) 
v_mul_f32 v15, v7, v11 // (op 6067) 
v_mul_f32 v16, v7, v12 // (op 6067) 
v_mul_f32 v17, v7, v13 // (op 6067) 



s_lshl_b32 s0, s4, 6 // (op 6053) 
s_mul_i32 s1, 3, s0 // (op 6053) 
s_add_u32 s1, s1, 384 // (op 6053) 

v_add_u32 v3, v9, s1 // (op 6053) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6053) Store value


v_accvgpr_read v10, a112 // (op 6112) 
v_accvgpr_read v11, a113 // (op 6112) 
v_accvgpr_read v12, a114 // (op 6112) 
v_accvgpr_read v13, a115 // (op 6112) 
v_mul_f32 v14, v7, v10 // (op 6112) 
v_mul_f32 v15, v7, v11 // (op 6112) 
v_mul_f32 v16, v7, v12 // (op 6112) 
v_mul_f32 v17, v7, v13 // (op 6112) 



v_add_u32 v3, 448, v9 // (op 6115) 

buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6115) Store value


v_accvgpr_read v10, a116 // (op 6125) 
v_accvgpr_read v11, a117 // (op 6125) 
v_accvgpr_read v12, a118 // (op 6125) 
v_accvgpr_read v13, a119 // (op 6125) 
v_mul_f32 v14, v7, v10 // (op 6125) 
v_mul_f32 v15, v7, v11 // (op 6125) 
v_mul_f32 v16, v7, v12 // (op 6125) 
v_mul_f32 v17, v7, v13 // (op 6125) 



s_lshl_b32 s0, s4, 6 // (op 6117) 
s_add_u32 s0, s0, 448 // (op 6117) 
v_add_u32 v3, v9, s0 // (op 6117) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6117) Store value


v_accvgpr_read v10, a120 // (op 6130) 
v_accvgpr_read v11, a121 // (op 6130) 
v_accvgpr_read v12, a122 // (op 6130) 
v_accvgpr_read v13, a123 // (op 6130) 
v_mul_f32 v14, v7, v10 // (op 6130) 
v_mul_f32 v15, v7, v11 // (op 6130) 
v_mul_f32 v16, v7, v12 // (op 6130) 
v_mul_f32 v17, v7, v13 // (op 6130) 



s_lshl_b32 s0, s4, 7 // (op 6119) 
s_add_u32 s0, s0, 448 // (op 6119) 
v_add_u32 v3, v9, s0 // (op 6119) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6119) Store value


v_accvgpr_read v10, a124 // (op 6135) 
v_accvgpr_read v11, a125 // (op 6135) 
v_accvgpr_read v12, a126 // (op 6135) 
v_accvgpr_read v13, a127 // (op 6135) 
v_mul_f32 v14, v7, v10 // (op 6135) 
v_mul_f32 v15, v7, v11 // (op 6135) 
v_mul_f32 v16, v7, v12 // (op 6135) 
v_mul_f32 v17, v7, v13 // (op 6135) 



s_lshl_b32 s0, s4, 6 // (op 6121) 
s_mul_i32 s1, 3, s0 // (op 6121) 
s_add_u32 s1, s1, 448 // (op 6121) 

v_add_u32 v3, v9, s1 // (op 6121) 


buffer_store_dwordx4 v[14:17], v3, s[44:47], 0 offen // (op 6121) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ConditionalBottom_0_beta_114
RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ConditionalFalse_0_beta_114:


v_add_u32 v3, 448, v8 // (op 6851) 

buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6851) Load value

v_accvgpr_read v14, a112 // (op 6839) 
v_accvgpr_read v15, a113 // (op 6839) 
v_accvgpr_read v16, a114 // (op 6839) 
v_accvgpr_read v17, a115 // (op 6839) 
v_mul_f32 v18, v7, v14 // (op 6839) 
v_mul_f32 v19, v7, v15 // (op 6839) 
v_mul_f32 v20, v7, v16 // (op 6839) 
v_mul_f32 v21, v7, v17 // (op 6839) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6850) 
v_mul_f32 v15, v0, v11 // (op 6850) 
v_mul_f32 v16, v0, v12 // (op 6850) 
v_mul_f32 v17, v0, v13 // (op 6850) 

v_add_f32 v10, v14, v18 // (op 6840) 
v_add_f32 v11, v15, v19 // (op 6840) 
v_add_f32 v12, v16, v20 // (op 6840) 
v_add_f32 v13, v17, v21 // (op 6840) 




v_add_u32 v3, 448, v9 // (op 6842) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6842) Store value




s_lshl_b32 s0, s2, 6 // (op 6855) 
s_add_u32 s0, s0, 448 // (op 6855) 
v_add_u32 v3, v8, s0 // (op 6855) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6855) Load value

v_accvgpr_read v14, a116 // (op 6865) 
v_accvgpr_read v15, a117 // (op 6865) 
v_accvgpr_read v16, a118 // (op 6865) 
v_accvgpr_read v17, a119 // (op 6865) 
v_mul_f32 v18, v7, v14 // (op 6865) 
v_mul_f32 v19, v7, v15 // (op 6865) 
v_mul_f32 v20, v7, v16 // (op 6865) 
v_mul_f32 v21, v7, v17 // (op 6865) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6853) 
v_mul_f32 v15, v0, v11 // (op 6853) 
v_mul_f32 v16, v0, v12 // (op 6853) 
v_mul_f32 v17, v0, v13 // (op 6853) 

v_add_f32 v10, v14, v18 // (op 6854) 
v_add_f32 v11, v15, v19 // (op 6854) 
v_add_f32 v12, v16, v20 // (op 6854) 
v_add_f32 v13, v17, v21 // (op 6854) 




s_lshl_b32 s0, s4, 6 // (op 6844) 
s_add_u32 s0, s0, 448 // (op 6844) 
v_add_u32 v3, v9, s0 // (op 6844) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6844) Store value




s_lshl_b32 s0, s2, 7 // (op 6859) 
s_add_u32 s0, s0, 448 // (op 6859) 
v_add_u32 v3, v8, s0 // (op 6859) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6859) Load value

v_accvgpr_read v14, a120 // (op 6867) 
v_accvgpr_read v15, a121 // (op 6867) 
v_accvgpr_read v16, a122 // (op 6867) 
v_accvgpr_read v17, a123 // (op 6867) 
v_mul_f32 v18, v7, v14 // (op 6867) 
v_mul_f32 v19, v7, v15 // (op 6867) 
v_mul_f32 v20, v7, v16 // (op 6867) 
v_mul_f32 v21, v7, v17 // (op 6867) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6857) 
v_mul_f32 v15, v0, v11 // (op 6857) 
v_mul_f32 v16, v0, v12 // (op 6857) 
v_mul_f32 v17, v0, v13 // (op 6857) 

v_add_f32 v10, v14, v18 // (op 6858) 
v_add_f32 v11, v15, v19 // (op 6858) 
v_add_f32 v12, v16, v20 // (op 6858) 
v_add_f32 v13, v17, v21 // (op 6858) 




s_lshl_b32 s0, s4, 7 // (op 6846) 
s_add_u32 s0, s0, 448 // (op 6846) 
v_add_u32 v3, v9, s0 // (op 6846) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6846) Store value




s_lshl_b32 s0, s2, 6 // (op 6863) 
s_mul_i32 s1, 3, s0 // (op 6863) 
s_add_u32 s1, s1, 448 // (op 6863) 

v_add_u32 v3, v8, s1 // (op 6863) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6863) Load value

v_accvgpr_read v14, a124 // (op 6869) 
v_accvgpr_read v15, a125 // (op 6869) 
v_accvgpr_read v16, a126 // (op 6869) 
v_accvgpr_read v17, a127 // (op 6869) 
v_mul_f32 v18, v7, v14 // (op 6869) 
v_mul_f32 v19, v7, v15 // (op 6869) 
v_mul_f32 v20, v7, v16 // (op 6869) 
v_mul_f32 v21, v7, v17 // (op 6869) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6861) 
v_mul_f32 v15, v0, v11 // (op 6861) 
v_mul_f32 v16, v0, v12 // (op 6861) 
v_mul_f32 v17, v0, v13 // (op 6861) 

v_add_f32 v10, v14, v18 // (op 6862) 
v_add_f32 v11, v15, v19 // (op 6862) 
v_add_f32 v12, v16, v20 // (op 6862) 
v_add_f32 v13, v17, v21 // (op 6862) 




s_lshl_b32 s0, s4, 6 // (op 6848) 
s_mul_i32 s1, 3, s0 // (op 6848) 
s_add_u32 s1, s1, 448 // (op 6848) 

v_add_u32 v3, v9, s1 // (op 6848) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6848) Store value




v_add_u32 v3, 384, v8 // (op 6772) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6772) Load value

v_accvgpr_read v14, a96 // (op 6760) 
v_accvgpr_read v15, a97 // (op 6760) 
v_accvgpr_read v16, a98 // (op 6760) 
v_accvgpr_read v17, a99 // (op 6760) 
v_mul_f32 v18, v7, v14 // (op 6760) 
v_mul_f32 v19, v7, v15 // (op 6760) 
v_mul_f32 v20, v7, v16 // (op 6760) 
v_mul_f32 v21, v7, v17 // (op 6760) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6771) 
v_mul_f32 v15, v0, v11 // (op 6771) 
v_mul_f32 v16, v0, v12 // (op 6771) 
v_mul_f32 v17, v0, v13 // (op 6771) 

v_add_f32 v10, v14, v18 // (op 6761) 
v_add_f32 v11, v15, v19 // (op 6761) 
v_add_f32 v12, v16, v20 // (op 6761) 
v_add_f32 v13, v17, v21 // (op 6761) 




v_add_u32 v3, 384, v9 // (op 6763) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6763) Store value




s_lshl_b32 s0, s2, 6 // (op 6776) 
s_add_u32 s0, s0, 384 // (op 6776) 
v_add_u32 v3, v8, s0 // (op 6776) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6776) Load value

v_accvgpr_read v14, a100 // (op 6786) 
v_accvgpr_read v15, a101 // (op 6786) 
v_accvgpr_read v16, a102 // (op 6786) 
v_accvgpr_read v17, a103 // (op 6786) 
v_mul_f32 v18, v7, v14 // (op 6786) 
v_mul_f32 v19, v7, v15 // (op 6786) 
v_mul_f32 v20, v7, v16 // (op 6786) 
v_mul_f32 v21, v7, v17 // (op 6786) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6774) 
v_mul_f32 v15, v0, v11 // (op 6774) 
v_mul_f32 v16, v0, v12 // (op 6774) 
v_mul_f32 v17, v0, v13 // (op 6774) 

v_add_f32 v10, v14, v18 // (op 6775) 
v_add_f32 v11, v15, v19 // (op 6775) 
v_add_f32 v12, v16, v20 // (op 6775) 
v_add_f32 v13, v17, v21 // (op 6775) 




s_lshl_b32 s0, s4, 6 // (op 6765) 
s_add_u32 s0, s0, 384 // (op 6765) 
v_add_u32 v3, v9, s0 // (op 6765) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6765) Store value




s_lshl_b32 s0, s2, 7 // (op 6780) 
s_add_u32 s0, s0, 384 // (op 6780) 
v_add_u32 v3, v8, s0 // (op 6780) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6780) Load value

v_accvgpr_read v14, a104 // (op 6788) 
v_accvgpr_read v15, a105 // (op 6788) 
v_accvgpr_read v16, a106 // (op 6788) 
v_accvgpr_read v17, a107 // (op 6788) 
v_mul_f32 v18, v7, v14 // (op 6788) 
v_mul_f32 v19, v7, v15 // (op 6788) 
v_mul_f32 v20, v7, v16 // (op 6788) 
v_mul_f32 v21, v7, v17 // (op 6788) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6778) 
v_mul_f32 v15, v0, v11 // (op 6778) 
v_mul_f32 v16, v0, v12 // (op 6778) 
v_mul_f32 v17, v0, v13 // (op 6778) 

v_add_f32 v10, v14, v18 // (op 6779) 
v_add_f32 v11, v15, v19 // (op 6779) 
v_add_f32 v12, v16, v20 // (op 6779) 
v_add_f32 v13, v17, v21 // (op 6779) 




s_lshl_b32 s0, s4, 7 // (op 6767) 
s_add_u32 s0, s0, 384 // (op 6767) 
v_add_u32 v3, v9, s0 // (op 6767) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6767) Store value




s_lshl_b32 s0, s2, 6 // (op 6784) 
s_mul_i32 s1, 3, s0 // (op 6784) 
s_add_u32 s1, s1, 384 // (op 6784) 

v_add_u32 v3, v8, s1 // (op 6784) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6784) Load value

v_accvgpr_read v14, a108 // (op 6790) 
v_accvgpr_read v15, a109 // (op 6790) 
v_accvgpr_read v16, a110 // (op 6790) 
v_accvgpr_read v17, a111 // (op 6790) 
v_mul_f32 v18, v7, v14 // (op 6790) 
v_mul_f32 v19, v7, v15 // (op 6790) 
v_mul_f32 v20, v7, v16 // (op 6790) 
v_mul_f32 v21, v7, v17 // (op 6790) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6782) 
v_mul_f32 v15, v0, v11 // (op 6782) 
v_mul_f32 v16, v0, v12 // (op 6782) 
v_mul_f32 v17, v0, v13 // (op 6782) 

v_add_f32 v10, v14, v18 // (op 6783) 
v_add_f32 v11, v15, v19 // (op 6783) 
v_add_f32 v12, v16, v20 // (op 6783) 
v_add_f32 v13, v17, v21 // (op 6783) 




s_lshl_b32 s0, s4, 6 // (op 6769) 
s_mul_i32 s1, 3, s0 // (op 6769) 
s_add_u32 s1, s1, 384 // (op 6769) 

v_add_u32 v3, v9, s1 // (op 6769) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6769) Store value




v_add_u32 v3, 320, v8 // (op 6693) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6693) Load value

v_accvgpr_read v14, a80 // (op 6681) 
v_accvgpr_read v15, a81 // (op 6681) 
v_accvgpr_read v16, a82 // (op 6681) 
v_accvgpr_read v17, a83 // (op 6681) 
v_mul_f32 v18, v7, v14 // (op 6681) 
v_mul_f32 v19, v7, v15 // (op 6681) 
v_mul_f32 v20, v7, v16 // (op 6681) 
v_mul_f32 v21, v7, v17 // (op 6681) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6692) 
v_mul_f32 v15, v0, v11 // (op 6692) 
v_mul_f32 v16, v0, v12 // (op 6692) 
v_mul_f32 v17, v0, v13 // (op 6692) 

v_add_f32 v10, v14, v18 // (op 6682) 
v_add_f32 v11, v15, v19 // (op 6682) 
v_add_f32 v12, v16, v20 // (op 6682) 
v_add_f32 v13, v17, v21 // (op 6682) 




v_add_u32 v3, 320, v9 // (op 6684) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6684) Store value




s_lshl_b32 s0, s2, 6 // (op 6697) 
s_add_u32 s0, s0, 320 // (op 6697) 
v_add_u32 v3, v8, s0 // (op 6697) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6697) Load value

v_accvgpr_read v14, a84 // (op 6707) 
v_accvgpr_read v15, a85 // (op 6707) 
v_accvgpr_read v16, a86 // (op 6707) 
v_accvgpr_read v17, a87 // (op 6707) 
v_mul_f32 v18, v7, v14 // (op 6707) 
v_mul_f32 v19, v7, v15 // (op 6707) 
v_mul_f32 v20, v7, v16 // (op 6707) 
v_mul_f32 v21, v7, v17 // (op 6707) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6695) 
v_mul_f32 v15, v0, v11 // (op 6695) 
v_mul_f32 v16, v0, v12 // (op 6695) 
v_mul_f32 v17, v0, v13 // (op 6695) 

v_add_f32 v10, v14, v18 // (op 6696) 
v_add_f32 v11, v15, v19 // (op 6696) 
v_add_f32 v12, v16, v20 // (op 6696) 
v_add_f32 v13, v17, v21 // (op 6696) 




s_lshl_b32 s0, s4, 6 // (op 6686) 
s_add_u32 s0, s0, 320 // (op 6686) 
v_add_u32 v3, v9, s0 // (op 6686) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6686) Store value




s_lshl_b32 s0, s2, 7 // (op 6701) 
s_add_u32 s0, s0, 320 // (op 6701) 
v_add_u32 v3, v8, s0 // (op 6701) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6701) Load value

v_accvgpr_read v14, a88 // (op 6709) 
v_accvgpr_read v15, a89 // (op 6709) 
v_accvgpr_read v16, a90 // (op 6709) 
v_accvgpr_read v17, a91 // (op 6709) 
v_mul_f32 v18, v7, v14 // (op 6709) 
v_mul_f32 v19, v7, v15 // (op 6709) 
v_mul_f32 v20, v7, v16 // (op 6709) 
v_mul_f32 v21, v7, v17 // (op 6709) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6699) 
v_mul_f32 v15, v0, v11 // (op 6699) 
v_mul_f32 v16, v0, v12 // (op 6699) 
v_mul_f32 v17, v0, v13 // (op 6699) 

v_add_f32 v10, v14, v18 // (op 6700) 
v_add_f32 v11, v15, v19 // (op 6700) 
v_add_f32 v12, v16, v20 // (op 6700) 
v_add_f32 v13, v17, v21 // (op 6700) 




s_lshl_b32 s0, s4, 7 // (op 6688) 
s_add_u32 s0, s0, 320 // (op 6688) 
v_add_u32 v3, v9, s0 // (op 6688) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6688) Store value




s_lshl_b32 s0, s2, 6 // (op 6705) 
s_mul_i32 s1, 3, s0 // (op 6705) 
s_add_u32 s1, s1, 320 // (op 6705) 

v_add_u32 v3, v8, s1 // (op 6705) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6705) Load value

v_accvgpr_read v14, a92 // (op 6711) 
v_accvgpr_read v15, a93 // (op 6711) 
v_accvgpr_read v16, a94 // (op 6711) 
v_accvgpr_read v17, a95 // (op 6711) 
v_mul_f32 v18, v7, v14 // (op 6711) 
v_mul_f32 v19, v7, v15 // (op 6711) 
v_mul_f32 v20, v7, v16 // (op 6711) 
v_mul_f32 v21, v7, v17 // (op 6711) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6703) 
v_mul_f32 v15, v0, v11 // (op 6703) 
v_mul_f32 v16, v0, v12 // (op 6703) 
v_mul_f32 v17, v0, v13 // (op 6703) 

v_add_f32 v10, v14, v18 // (op 6704) 
v_add_f32 v11, v15, v19 // (op 6704) 
v_add_f32 v12, v16, v20 // (op 6704) 
v_add_f32 v13, v17, v21 // (op 6704) 




s_lshl_b32 s0, s4, 6 // (op 6690) 
s_mul_i32 s1, 3, s0 // (op 6690) 
s_add_u32 s1, s1, 320 // (op 6690) 

v_add_u32 v3, v9, s1 // (op 6690) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6690) Store value




v_add_u32 v3, 256, v8 // (op 6614) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6614) Load value

v_accvgpr_read v14, a64 // (op 6602) 
v_accvgpr_read v15, a65 // (op 6602) 
v_accvgpr_read v16, a66 // (op 6602) 
v_accvgpr_read v17, a67 // (op 6602) 
v_mul_f32 v18, v7, v14 // (op 6602) 
v_mul_f32 v19, v7, v15 // (op 6602) 
v_mul_f32 v20, v7, v16 // (op 6602) 
v_mul_f32 v21, v7, v17 // (op 6602) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6613) 
v_mul_f32 v15, v0, v11 // (op 6613) 
v_mul_f32 v16, v0, v12 // (op 6613) 
v_mul_f32 v17, v0, v13 // (op 6613) 

v_add_f32 v10, v14, v18 // (op 6603) 
v_add_f32 v11, v15, v19 // (op 6603) 
v_add_f32 v12, v16, v20 // (op 6603) 
v_add_f32 v13, v17, v21 // (op 6603) 




v_add_u32 v3, 256, v9 // (op 6605) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6605) Store value




s_lshl_b32 s0, s2, 6 // (op 6618) 
s_add_u32 s0, s0, 256 // (op 6618) 
v_add_u32 v3, v8, s0 // (op 6618) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6618) Load value

v_accvgpr_read v14, a68 // (op 6628) 
v_accvgpr_read v15, a69 // (op 6628) 
v_accvgpr_read v16, a70 // (op 6628) 
v_accvgpr_read v17, a71 // (op 6628) 
v_mul_f32 v18, v7, v14 // (op 6628) 
v_mul_f32 v19, v7, v15 // (op 6628) 
v_mul_f32 v20, v7, v16 // (op 6628) 
v_mul_f32 v21, v7, v17 // (op 6628) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6616) 
v_mul_f32 v15, v0, v11 // (op 6616) 
v_mul_f32 v16, v0, v12 // (op 6616) 
v_mul_f32 v17, v0, v13 // (op 6616) 

v_add_f32 v10, v14, v18 // (op 6617) 
v_add_f32 v11, v15, v19 // (op 6617) 
v_add_f32 v12, v16, v20 // (op 6617) 
v_add_f32 v13, v17, v21 // (op 6617) 




s_lshl_b32 s0, s4, 6 // (op 6607) 
s_add_u32 s0, s0, 256 // (op 6607) 
v_add_u32 v3, v9, s0 // (op 6607) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6607) Store value




s_lshl_b32 s0, s2, 7 // (op 6622) 
s_add_u32 s0, s0, 256 // (op 6622) 
v_add_u32 v3, v8, s0 // (op 6622) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6622) Load value

v_accvgpr_read v14, a72 // (op 6630) 
v_accvgpr_read v15, a73 // (op 6630) 
v_accvgpr_read v16, a74 // (op 6630) 
v_accvgpr_read v17, a75 // (op 6630) 
v_mul_f32 v18, v7, v14 // (op 6630) 
v_mul_f32 v19, v7, v15 // (op 6630) 
v_mul_f32 v20, v7, v16 // (op 6630) 
v_mul_f32 v21, v7, v17 // (op 6630) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6620) 
v_mul_f32 v15, v0, v11 // (op 6620) 
v_mul_f32 v16, v0, v12 // (op 6620) 
v_mul_f32 v17, v0, v13 // (op 6620) 

v_add_f32 v10, v14, v18 // (op 6621) 
v_add_f32 v11, v15, v19 // (op 6621) 
v_add_f32 v12, v16, v20 // (op 6621) 
v_add_f32 v13, v17, v21 // (op 6621) 




s_lshl_b32 s0, s4, 7 // (op 6609) 
s_add_u32 s0, s0, 256 // (op 6609) 
v_add_u32 v3, v9, s0 // (op 6609) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6609) Store value




s_lshl_b32 s0, s2, 6 // (op 6626) 
s_mul_i32 s1, 3, s0 // (op 6626) 
s_add_u32 s1, s1, 256 // (op 6626) 

v_add_u32 v3, v8, s1 // (op 6626) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6626) Load value

v_accvgpr_read v14, a76 // (op 6632) 
v_accvgpr_read v15, a77 // (op 6632) 
v_accvgpr_read v16, a78 // (op 6632) 
v_accvgpr_read v17, a79 // (op 6632) 
v_mul_f32 v18, v7, v14 // (op 6632) 
v_mul_f32 v19, v7, v15 // (op 6632) 
v_mul_f32 v20, v7, v16 // (op 6632) 
v_mul_f32 v21, v7, v17 // (op 6632) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6624) 
v_mul_f32 v15, v0, v11 // (op 6624) 
v_mul_f32 v16, v0, v12 // (op 6624) 
v_mul_f32 v17, v0, v13 // (op 6624) 

v_add_f32 v10, v14, v18 // (op 6625) 
v_add_f32 v11, v15, v19 // (op 6625) 
v_add_f32 v12, v16, v20 // (op 6625) 
v_add_f32 v13, v17, v21 // (op 6625) 




s_lshl_b32 s0, s4, 6 // (op 6611) 
s_mul_i32 s1, 3, s0 // (op 6611) 
s_add_u32 s1, s1, 256 // (op 6611) 

v_add_u32 v3, v9, s1 // (op 6611) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6611) Store value




v_add_u32 v3, 192, v8 // (op 6535) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6535) Load value

v_accvgpr_read v14, a48 // (op 6523) 
v_accvgpr_read v15, a49 // (op 6523) 
v_accvgpr_read v16, a50 // (op 6523) 
v_accvgpr_read v17, a51 // (op 6523) 
v_mul_f32 v18, v7, v14 // (op 6523) 
v_mul_f32 v19, v7, v15 // (op 6523) 
v_mul_f32 v20, v7, v16 // (op 6523) 
v_mul_f32 v21, v7, v17 // (op 6523) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6534) 
v_mul_f32 v15, v0, v11 // (op 6534) 
v_mul_f32 v16, v0, v12 // (op 6534) 
v_mul_f32 v17, v0, v13 // (op 6534) 

v_add_f32 v10, v14, v18 // (op 6524) 
v_add_f32 v11, v15, v19 // (op 6524) 
v_add_f32 v12, v16, v20 // (op 6524) 
v_add_f32 v13, v17, v21 // (op 6524) 




v_add_u32 v3, 192, v9 // (op 6526) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6526) Store value




s_lshl_b32 s0, s2, 6 // (op 6539) 
s_add_u32 s0, s0, 192 // (op 6539) 
v_add_u32 v3, v8, s0 // (op 6539) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6539) Load value

v_accvgpr_read v14, a52 // (op 6549) 
v_accvgpr_read v15, a53 // (op 6549) 
v_accvgpr_read v16, a54 // (op 6549) 
v_accvgpr_read v17, a55 // (op 6549) 
v_mul_f32 v18, v7, v14 // (op 6549) 
v_mul_f32 v19, v7, v15 // (op 6549) 
v_mul_f32 v20, v7, v16 // (op 6549) 
v_mul_f32 v21, v7, v17 // (op 6549) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6537) 
v_mul_f32 v15, v0, v11 // (op 6537) 
v_mul_f32 v16, v0, v12 // (op 6537) 
v_mul_f32 v17, v0, v13 // (op 6537) 

v_add_f32 v10, v14, v18 // (op 6538) 
v_add_f32 v11, v15, v19 // (op 6538) 
v_add_f32 v12, v16, v20 // (op 6538) 
v_add_f32 v13, v17, v21 // (op 6538) 




s_lshl_b32 s0, s4, 6 // (op 6528) 
s_add_u32 s0, s0, 192 // (op 6528) 
v_add_u32 v3, v9, s0 // (op 6528) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6528) Store value




s_lshl_b32 s0, s2, 7 // (op 6543) 
s_add_u32 s0, s0, 192 // (op 6543) 
v_add_u32 v3, v8, s0 // (op 6543) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6543) Load value

v_accvgpr_read v14, a56 // (op 6551) 
v_accvgpr_read v15, a57 // (op 6551) 
v_accvgpr_read v16, a58 // (op 6551) 
v_accvgpr_read v17, a59 // (op 6551) 
v_mul_f32 v18, v7, v14 // (op 6551) 
v_mul_f32 v19, v7, v15 // (op 6551) 
v_mul_f32 v20, v7, v16 // (op 6551) 
v_mul_f32 v21, v7, v17 // (op 6551) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6541) 
v_mul_f32 v15, v0, v11 // (op 6541) 
v_mul_f32 v16, v0, v12 // (op 6541) 
v_mul_f32 v17, v0, v13 // (op 6541) 

v_add_f32 v10, v14, v18 // (op 6542) 
v_add_f32 v11, v15, v19 // (op 6542) 
v_add_f32 v12, v16, v20 // (op 6542) 
v_add_f32 v13, v17, v21 // (op 6542) 




s_lshl_b32 s0, s4, 7 // (op 6530) 
s_add_u32 s0, s0, 192 // (op 6530) 
v_add_u32 v3, v9, s0 // (op 6530) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6530) Store value




s_lshl_b32 s0, s2, 6 // (op 6547) 
s_mul_i32 s1, 3, s0 // (op 6547) 
s_add_u32 s1, s1, 192 // (op 6547) 

v_add_u32 v3, v8, s1 // (op 6547) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6547) Load value

v_accvgpr_read v14, a60 // (op 6553) 
v_accvgpr_read v15, a61 // (op 6553) 
v_accvgpr_read v16, a62 // (op 6553) 
v_accvgpr_read v17, a63 // (op 6553) 
v_mul_f32 v18, v7, v14 // (op 6553) 
v_mul_f32 v19, v7, v15 // (op 6553) 
v_mul_f32 v20, v7, v16 // (op 6553) 
v_mul_f32 v21, v7, v17 // (op 6553) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6545) 
v_mul_f32 v15, v0, v11 // (op 6545) 
v_mul_f32 v16, v0, v12 // (op 6545) 
v_mul_f32 v17, v0, v13 // (op 6545) 

v_add_f32 v10, v14, v18 // (op 6546) 
v_add_f32 v11, v15, v19 // (op 6546) 
v_add_f32 v12, v16, v20 // (op 6546) 
v_add_f32 v13, v17, v21 // (op 6546) 




s_lshl_b32 s0, s4, 6 // (op 6532) 
s_mul_i32 s1, 3, s0 // (op 6532) 
s_add_u32 s1, s1, 192 // (op 6532) 

v_add_u32 v3, v9, s1 // (op 6532) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6532) Store value




v_add_u32 v3, 128, v8 // (op 6456) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6456) Load value

v_accvgpr_read v14, a32 // (op 6444) 
v_accvgpr_read v15, a33 // (op 6444) 
v_accvgpr_read v16, a34 // (op 6444) 
v_accvgpr_read v17, a35 // (op 6444) 
v_mul_f32 v18, v7, v14 // (op 6444) 
v_mul_f32 v19, v7, v15 // (op 6444) 
v_mul_f32 v20, v7, v16 // (op 6444) 
v_mul_f32 v21, v7, v17 // (op 6444) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6455) 
v_mul_f32 v15, v0, v11 // (op 6455) 
v_mul_f32 v16, v0, v12 // (op 6455) 
v_mul_f32 v17, v0, v13 // (op 6455) 

v_add_f32 v10, v14, v18 // (op 6445) 
v_add_f32 v11, v15, v19 // (op 6445) 
v_add_f32 v12, v16, v20 // (op 6445) 
v_add_f32 v13, v17, v21 // (op 6445) 




v_add_u32 v3, 128, v9 // (op 6447) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6447) Store value




s_lshl_b32 s0, s2, 6 // (op 6460) 
s_add_u32 s0, s0, 128 // (op 6460) 
v_add_u32 v3, v8, s0 // (op 6460) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6460) Load value

v_accvgpr_read v14, a36 // (op 6470) 
v_accvgpr_read v15, a37 // (op 6470) 
v_accvgpr_read v16, a38 // (op 6470) 
v_accvgpr_read v17, a39 // (op 6470) 
v_mul_f32 v18, v7, v14 // (op 6470) 
v_mul_f32 v19, v7, v15 // (op 6470) 
v_mul_f32 v20, v7, v16 // (op 6470) 
v_mul_f32 v21, v7, v17 // (op 6470) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6458) 
v_mul_f32 v15, v0, v11 // (op 6458) 
v_mul_f32 v16, v0, v12 // (op 6458) 
v_mul_f32 v17, v0, v13 // (op 6458) 

v_add_f32 v10, v14, v18 // (op 6459) 
v_add_f32 v11, v15, v19 // (op 6459) 
v_add_f32 v12, v16, v20 // (op 6459) 
v_add_f32 v13, v17, v21 // (op 6459) 




s_lshl_b32 s0, s4, 6 // (op 6449) 
s_add_u32 s0, s0, 128 // (op 6449) 
v_add_u32 v3, v9, s0 // (op 6449) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6449) Store value




s_lshl_b32 s0, s2, 7 // (op 6464) 
s_add_u32 s0, s0, 128 // (op 6464) 
v_add_u32 v3, v8, s0 // (op 6464) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6464) Load value

v_accvgpr_read v14, a40 // (op 6472) 
v_accvgpr_read v15, a41 // (op 6472) 
v_accvgpr_read v16, a42 // (op 6472) 
v_accvgpr_read v17, a43 // (op 6472) 
v_mul_f32 v18, v7, v14 // (op 6472) 
v_mul_f32 v19, v7, v15 // (op 6472) 
v_mul_f32 v20, v7, v16 // (op 6472) 
v_mul_f32 v21, v7, v17 // (op 6472) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6462) 
v_mul_f32 v15, v0, v11 // (op 6462) 
v_mul_f32 v16, v0, v12 // (op 6462) 
v_mul_f32 v17, v0, v13 // (op 6462) 

v_add_f32 v10, v14, v18 // (op 6463) 
v_add_f32 v11, v15, v19 // (op 6463) 
v_add_f32 v12, v16, v20 // (op 6463) 
v_add_f32 v13, v17, v21 // (op 6463) 




s_lshl_b32 s0, s4, 7 // (op 6451) 
s_add_u32 s0, s0, 128 // (op 6451) 
v_add_u32 v3, v9, s0 // (op 6451) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6451) Store value




s_lshl_b32 s0, s2, 6 // (op 6468) 
s_mul_i32 s1, 3, s0 // (op 6468) 
s_add_u32 s1, s1, 128 // (op 6468) 

v_add_u32 v3, v8, s1 // (op 6468) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6468) Load value

v_accvgpr_read v14, a44 // (op 6474) 
v_accvgpr_read v15, a45 // (op 6474) 
v_accvgpr_read v16, a46 // (op 6474) 
v_accvgpr_read v17, a47 // (op 6474) 
v_mul_f32 v18, v7, v14 // (op 6474) 
v_mul_f32 v19, v7, v15 // (op 6474) 
v_mul_f32 v20, v7, v16 // (op 6474) 
v_mul_f32 v21, v7, v17 // (op 6474) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6466) 
v_mul_f32 v15, v0, v11 // (op 6466) 
v_mul_f32 v16, v0, v12 // (op 6466) 
v_mul_f32 v17, v0, v13 // (op 6466) 

v_add_f32 v10, v14, v18 // (op 6467) 
v_add_f32 v11, v15, v19 // (op 6467) 
v_add_f32 v12, v16, v20 // (op 6467) 
v_add_f32 v13, v17, v21 // (op 6467) 




s_lshl_b32 s0, s4, 6 // (op 6453) 
s_mul_i32 s1, 3, s0 // (op 6453) 
s_add_u32 s1, s1, 128 // (op 6453) 

v_add_u32 v3, v9, s1 // (op 6453) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6453) Store value




v_add_u32 v3, 64, v8 // (op 6377) 

s_nop 0
buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6377) Load value

v_accvgpr_read v14, a16 // (op 6365) 
v_accvgpr_read v15, a17 // (op 6365) 
v_accvgpr_read v16, a18 // (op 6365) 
v_accvgpr_read v17, a19 // (op 6365) 
v_mul_f32 v18, v7, v14 // (op 6365) 
v_mul_f32 v19, v7, v15 // (op 6365) 
v_mul_f32 v20, v7, v16 // (op 6365) 
v_mul_f32 v21, v7, v17 // (op 6365) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6376) 
v_mul_f32 v15, v0, v11 // (op 6376) 
v_mul_f32 v16, v0, v12 // (op 6376) 
v_mul_f32 v17, v0, v13 // (op 6376) 

v_add_f32 v10, v14, v18 // (op 6366) 
v_add_f32 v11, v15, v19 // (op 6366) 
v_add_f32 v12, v16, v20 // (op 6366) 
v_add_f32 v13, v17, v21 // (op 6366) 




v_add_u32 v3, 64, v9 // (op 6368) 

buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6368) Store value




s_lshl_b32 s0, s2, 6 // (op 6381) 
s_add_u32 s0, s0, 64 // (op 6381) 
v_add_u32 v3, v8, s0 // (op 6381) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6381) Load value

v_accvgpr_read v14, a20 // (op 6391) 
v_accvgpr_read v15, a21 // (op 6391) 
v_accvgpr_read v16, a22 // (op 6391) 
v_accvgpr_read v17, a23 // (op 6391) 
v_mul_f32 v18, v7, v14 // (op 6391) 
v_mul_f32 v19, v7, v15 // (op 6391) 
v_mul_f32 v20, v7, v16 // (op 6391) 
v_mul_f32 v21, v7, v17 // (op 6391) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6379) 
v_mul_f32 v15, v0, v11 // (op 6379) 
v_mul_f32 v16, v0, v12 // (op 6379) 
v_mul_f32 v17, v0, v13 // (op 6379) 

v_add_f32 v10, v14, v18 // (op 6380) 
v_add_f32 v11, v15, v19 // (op 6380) 
v_add_f32 v12, v16, v20 // (op 6380) 
v_add_f32 v13, v17, v21 // (op 6380) 




s_lshl_b32 s0, s4, 6 // (op 6370) 
s_add_u32 s0, s0, 64 // (op 6370) 
v_add_u32 v3, v9, s0 // (op 6370) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6370) Store value




s_lshl_b32 s0, s2, 7 // (op 6385) 
s_add_u32 s0, s0, 64 // (op 6385) 
v_add_u32 v3, v8, s0 // (op 6385) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6385) Load value

v_accvgpr_read v14, a24 // (op 6393) 
v_accvgpr_read v15, a25 // (op 6393) 
v_accvgpr_read v16, a26 // (op 6393) 
v_accvgpr_read v17, a27 // (op 6393) 
v_mul_f32 v18, v7, v14 // (op 6393) 
v_mul_f32 v19, v7, v15 // (op 6393) 
v_mul_f32 v20, v7, v16 // (op 6393) 
v_mul_f32 v21, v7, v17 // (op 6393) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6383) 
v_mul_f32 v15, v0, v11 // (op 6383) 
v_mul_f32 v16, v0, v12 // (op 6383) 
v_mul_f32 v17, v0, v13 // (op 6383) 

v_add_f32 v10, v14, v18 // (op 6384) 
v_add_f32 v11, v15, v19 // (op 6384) 
v_add_f32 v12, v16, v20 // (op 6384) 
v_add_f32 v13, v17, v21 // (op 6384) 




s_lshl_b32 s0, s4, 7 // (op 6372) 
s_add_u32 s0, s0, 64 // (op 6372) 
v_add_u32 v3, v9, s0 // (op 6372) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6372) Store value




s_lshl_b32 s0, s2, 6 // (op 6389) 
s_mul_i32 s1, 3, s0 // (op 6389) 
s_add_u32 s1, s1, 64 // (op 6389) 

v_add_u32 v3, v8, s1 // (op 6389) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6389) Load value

v_accvgpr_read v14, a28 // (op 6395) 
v_accvgpr_read v15, a29 // (op 6395) 
v_accvgpr_read v16, a30 // (op 6395) 
v_accvgpr_read v17, a31 // (op 6395) 
v_mul_f32 v18, v7, v14 // (op 6395) 
v_mul_f32 v19, v7, v15 // (op 6395) 
v_mul_f32 v20, v7, v16 // (op 6395) 
v_mul_f32 v21, v7, v17 // (op 6395) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6387) 
v_mul_f32 v15, v0, v11 // (op 6387) 
v_mul_f32 v16, v0, v12 // (op 6387) 
v_mul_f32 v17, v0, v13 // (op 6387) 

v_add_f32 v10, v14, v18 // (op 6388) 
v_add_f32 v11, v15, v19 // (op 6388) 
v_add_f32 v12, v16, v20 // (op 6388) 
v_add_f32 v13, v17, v21 // (op 6388) 




s_lshl_b32 s0, s4, 6 // (op 6374) 
s_mul_i32 s1, 3, s0 // (op 6374) 
s_add_u32 s1, s1, 64 // (op 6374) 

v_add_u32 v3, v9, s1 // (op 6374) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6374) Store value




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




s_lshl_b32 s0, s2, 6 // (op 6291) 
v_add_u32 v3, v8, s0 // (op 6291) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6291) Load value

v_accvgpr_read v14, a4 // (op 6288) 
v_accvgpr_read v15, a5 // (op 6288) 
v_accvgpr_read v16, a6 // (op 6288) 
v_accvgpr_read v17, a7 // (op 6288) 
v_mul_f32 v18, v7, v14 // (op 6288) 
v_mul_f32 v19, v7, v15 // (op 6288) 
v_mul_f32 v20, v7, v16 // (op 6288) 
v_mul_f32 v21, v7, v17 // (op 6288) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6292) 
v_mul_f32 v15, v0, v11 // (op 6292) 
v_mul_f32 v16, v0, v12 // (op 6292) 
v_mul_f32 v17, v0, v13 // (op 6292) 

v_add_f32 v10, v14, v18 // (op 6289) 
v_add_f32 v11, v15, v19 // (op 6289) 
v_add_f32 v12, v16, v20 // (op 6289) 
v_add_f32 v13, v17, v21 // (op 6289) 




s_lshl_b32 s0, s4, 6 // (op 6290) 
v_add_u32 v3, v9, s0 // (op 6290) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6290) Store value




s_lshl_b32 s0, s2, 7 // (op 6303) 
v_add_u32 v3, v8, s0 // (op 6303) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6303) Load value

v_accvgpr_read v14, a8 // (op 6300) 
v_accvgpr_read v15, a9 // (op 6300) 
v_accvgpr_read v16, a10 // (op 6300) 
v_accvgpr_read v17, a11 // (op 6300) 
v_mul_f32 v18, v7, v14 // (op 6300) 
v_mul_f32 v19, v7, v15 // (op 6300) 
v_mul_f32 v20, v7, v16 // (op 6300) 
v_mul_f32 v21, v7, v17 // (op 6300) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6304) 
v_mul_f32 v15, v0, v11 // (op 6304) 
v_mul_f32 v16, v0, v12 // (op 6304) 
v_mul_f32 v17, v0, v13 // (op 6304) 

v_add_f32 v10, v14, v18 // (op 6301) 
v_add_f32 v11, v15, v19 // (op 6301) 
v_add_f32 v12, v16, v20 // (op 6301) 
v_add_f32 v13, v17, v21 // (op 6301) 




s_lshl_b32 s0, s4, 7 // (op 6302) 
v_add_u32 v3, v9, s0 // (op 6302) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6302) Store value




s_lshl_b32 s0, s2, 6 // (op 6315) 
s_mul_i32 s1, 3, s0 // (op 6315) 

v_add_u32 v3, v8, s1 // (op 6315) 


buffer_load_dwordx4 v[10:13], v3, s[40:43], 0 offen // (op 6315) Load value

v_accvgpr_read v14, a12 // (op 6312) 
v_accvgpr_read v15, a13 // (op 6312) 
v_accvgpr_read v16, a14 // (op 6312) 
v_accvgpr_read v17, a15 // (op 6312) 
v_mul_f32 v18, v7, v14 // (op 6312) 
v_mul_f32 v19, v7, v15 // (op 6312) 
v_mul_f32 v20, v7, v16 // (op 6312) 
v_mul_f32 v21, v7, v17 // (op 6312) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v0, v10 // (op 6316) 
v_mul_f32 v15, v0, v11 // (op 6316) 
v_mul_f32 v16, v0, v12 // (op 6316) 
v_mul_f32 v17, v0, v13 // (op 6316) 

v_add_f32 v10, v14, v18 // (op 6313) 
v_add_f32 v11, v15, v19 // (op 6313) 
v_add_f32 v12, v16, v20 // (op 6313) 
v_add_f32 v13, v17, v21 // (op 6313) 




s_lshl_b32 s0, s4, 6 // (op 6314) 
s_mul_i32 s1, 3, s0 // (op 6314) 

v_add_u32 v3, v9, s1 // (op 6314) 


buffer_store_dwordx4 v[10:13], v3, s[44:47], 0 offen // (op 6314) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_ConditionalBottom_0_beta_114:








































s_endpgm  // End of RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2
.LRR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_end:
.size RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2, .LRR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_end-RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2
  .amdhsa_next_free_vgpr 284
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 81920 // lds bytes
  .amdhsa_accum_offset 156
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
    DataTypeB: F8
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
    macrotile: [256, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesX, TilesY, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2
    .symbol: RR_GEMM_TN_BF6_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 81920
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 64
    .vgpr_count: 154
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

