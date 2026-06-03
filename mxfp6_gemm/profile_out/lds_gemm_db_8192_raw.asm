;; lds_gemm_db<256,256,192, 2,2, MIN_OCC=1, SWZ=16, DB=true, float>  (gfx950)
;; Production MXFP6 GEMM kernel @ 8192^3 — ISA extracted from RCV code.json
;; 1883 code entries

; _ZN5mxfp611lds_gemm_dbILi256ELi256ELi192ELi2ELi2ELi1ELi16ELb1EfEEvPKvS2_PKhS4_PT7_iiiiS4_S4_
s_load_dwordx2 s[18:19], s[0:1], 0x48
s_load_dwordx8 s[4:11], s[0:1], 0x0
s_load_dwordx2 s[16:17], s[0:1], 0x20
s_load_dwordx4 s[12:15], s[0:1], 0x28
v_lshrrev_b32_e32 v3, 6, v0
v_or_b32_e32 v119, 0x100, v0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s20, s18, 4
s_abs_i32 s21, s20
v_cvt_f32_u32_e32 v1, s21
s_sub_i32 s22, 0, s21
s_mul_i32 s3, s18, s3
s_add_i32 s3, s3, s2
v_rcp_iflag_f32_e32 v1, v1
s_abs_i32 s18, s3
s_xor_b32 s2, s3, s20
s_ashr_i32 s2, s2, 31
v_mul_f32_e32 v1, 0x4f7ffffe, v1
v_cvt_u32_f32_e32 v1, v1
v_mov_b32_e32 v63, 0
v_lshlrev_b32_e32 v47, 10, v3
v_mul_u32_u24_e32 v3, 0x1c72, v119
v_readfirstlane_b32 s23, v1
s_mul_i32 s22, s22, s23
s_mul_hi_u32 s22, s23, s22
s_add_i32 s23, s23, s22
s_mul_hi_u32 s22, s18, s23
s_mul_i32 s23, s22, s21
s_sub_i32 s18, s18, s23
s_add_i32 s23, s22, 1
s_sub_i32 s24, s18, s21
s_cmp_ge_u32 s18, s21
s_cselect_b32 s22, s23, s22
s_cselect_b32 s18, s24, s18
s_add_i32 s23, s22, 1
s_cmp_ge_u32 s18, s21
s_cselect_b32 s18, s23, s22
s_xor_b32 s18, s18, s2
s_sub_i32 s2, s18, s2
s_lshl_b32 s18, s2, 4
s_sub_i32 s19, s19, s18
s_min_i32 s19, s19, 16
s_abs_i32 s21, s19
v_cvt_f32_u32_e32 v2, s21
s_sub_i32 s22, 0, s21
s_mul_i32 s2, s2, s20
s_sub_i32 s2, s3, s2
v_rcp_iflag_f32_e32 v2, v2
s_abs_i32 s20, s2
s_xor_b32 s3, s2, s19
s_ashr_i32 s3, s3, 31
v_mul_f32_e32 v2, 0x4f7ffffe, v2
v_cvt_u32_f32_e32 v2, v2
v_lshrrev_b32_e32 v3, 16, v3
v_mov_b32_e32 v65, v63
v_or_b32_e32 v51, 0x1000, v47
v_readfirstlane_b32 s24, v2
s_mul_i32 s22, s22, s24
s_mul_hi_u32 s22, s24, s22
s_add_i32 s24, s24, s22
s_mul_hi_u32 s22, s20, s24
s_mul_i32 s24, s22, s21
s_sub_i32 s20, s20, s24
s_add_i32 s24, s22, 1
s_sub_i32 s25, s20, s21
s_cmp_ge_u32 s20, s21
s_cselect_b32 s22, s24, s22
s_cselect_b32 s20, s25, s20
s_add_i32 s24, s22, 1
s_cmp_ge_u32 s20, s21
s_cselect_b32 s20, s24, s22
s_xor_b32 s20, s20, s3
s_sub_i32 s22, s20, s3
s_mul_i32 s3, s22, s19
s_sub_i32 s20, s2, s3
s_add_i32 s20, s20, s18
s_lshl_b32 s18, s22, 8
s_mul_i32 s3, s18, s14
v_mul_u32_u24_e32 v2, 0x1c72, v0
s_mul_hi_i32 s2, s18, s14
s_add_u32 s4, s4, s3
v_lshrrev_b32_e32 v2, 16, v2
s_addc_u32 s5, s5, s2
s_lshl_b32 s19, s20, 8
v_mul_lo_u16_e32 v4, 9, v2
s_mul_i32 s3, s19, s15
v_sub_u16_e32 v6, v0, v4
v_mov_b64_e32 v[12:13], s[4:5]
s_mul_hi_i32 s2, s19, s15
s_add_u32 s6, s6, s3
s_mul_hi_i32 s21, s13, 0x55555556
v_mad_i64_i32 v[4:5], s[24:25], s14, v2, v[12:13]
v_lshlrev_b16_e32 v62, 4, v6
s_addc_u32 s7, s7, s2
s_lshr_b32 s3, s21, 31
v_lshl_add_u64 v[4:5], v[4:5], 0, v[62:63]
s_add_i32 s21, s21, s3
v_readfirstlane_b32 s3, v47
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[4:5], off
v_mul_lo_u16_e32 v4, 9, v3
v_sub_u16_e32 v6, v119, v4
v_mad_i64_i32 v[4:5], s[24:25], s14, v3, v[12:13]
v_lshlrev_b16_e32 v64, 4, v6
v_lshl_add_u64 v[4:5], v[4:5], 0, v[64:65]
v_or_b32_e32 v120, 0x200, v0
v_readfirstlane_b32 s3, v51
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[4:5], off
v_mul_u32_u24_e32 v4, 0x1c72, v120
v_lshrrev_b32_e32 v4, 16, v4
v_mul_lo_u16_e32 v5, 9, v4
v_sub_u16_e32 v5, v120, v5
v_mad_i64_i32 v[6:7], s[24:25], s14, v4, v[12:13]
v_lshlrev_b16_e32 v66, 4, v5
v_mov_b32_e32 v67, v63
v_lshl_add_u64 v[6:7], v[6:7], 0, v[66:67]
v_or_b32_e32 v55, 0x2000, v47
v_mov_b32_e32 v69, v63
v_readfirstlane_b32 s3, v55
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[6:7], off
v_or_b32_e32 v6, 0x300, v0
v_mul_u32_u24_e32 v5, 0x1c72, v6
v_lshrrev_b32_e32 v5, 16, v5
v_mul_lo_u16_e32 v7, 9, v5
v_sub_u16_e32 v8, v6, v7
v_mad_i64_i32 v[6:7], s[24:25], s14, v5, v[12:13]
v_lshlrev_b16_e32 v68, 4, v8
v_lshl_add_u64 v[6:7], v[6:7], 0, v[68:69]
v_or_b32_e32 v59, 0x3000, v47
v_mov_b32_e32 v71, v63
v_readfirstlane_b32 s3, v59
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[6:7], off
v_or_b32_e32 v7, 0x400, v0
v_mul_u32_u24_e32 v6, 0x1c72, v7
v_lshrrev_b32_e32 v6, 16, v6
v_mul_lo_u16_e32 v8, 9, v6
v_sub_u16_e32 v7, v7, v8
v_mad_i64_i32 v[8:9], s[24:25], s14, v6, v[12:13]
v_lshlrev_b16_e32 v70, 4, v7
v_lshl_add_u64 v[8:9], v[8:9], 0, v[70:71]
v_or_b32_e32 v116, 0x4000, v47
v_mov_b32_e32 v73, v63
v_readfirstlane_b32 s3, v116
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[8:9], off
v_or_b32_e32 v8, 0x500, v0
v_mul_u32_u24_e32 v7, 0x1c72, v8
v_lshrrev_b32_e32 v7, 16, v7
v_mul_lo_u16_e32 v9, 9, v7
v_sub_u16_e32 v10, v8, v9
v_mad_i64_i32 v[8:9], s[24:25], s14, v7, v[12:13]
v_lshlrev_b16_e32 v72, 4, v10
v_lshl_add_u64 v[8:9], v[8:9], 0, v[72:73]
v_or_b32_e32 v117, 0x5000, v47
v_mov_b32_e32 v75, v63
v_readfirstlane_b32 s3, v117
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[8:9], off
v_or_b32_e32 v9, 0x600, v0
v_mul_u32_u24_e32 v8, 0x1c72, v9
v_lshrrev_b32_e32 v8, 16, v8
v_mul_lo_u16_e32 v10, 9, v8
v_sub_u16_e32 v9, v9, v10
v_mad_i64_i32 v[10:11], s[24:25], s14, v8, v[12:13]
v_lshlrev_b16_e32 v74, 4, v9
v_lshl_add_u64 v[10:11], v[10:11], 0, v[74:75]
v_or_b32_e32 v122, 0x6000, v47
v_mov_b32_e32 v77, v63
v_readfirstlane_b32 s3, v122
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[10:11], off
v_or_b32_e32 v10, 0x700, v0
v_mul_u32_u24_e32 v9, 0x1c72, v10
v_lshrrev_b32_e32 v9, 16, v9
v_mul_lo_u16_e32 v11, 9, v9
v_sub_u16_e32 v14, v10, v11
v_mad_i64_i32 v[10:11], s[24:25], s14, v9, v[12:13]
v_lshlrev_b16_e32 v76, 4, v14
v_lshl_add_u64 v[10:11], v[10:11], 0, v[76:77]
v_or_b32_e32 v123, 0x7000, v47
v_mov_b32_e32 v79, v63
v_readfirstlane_b32 s3, v123
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[10:11], off
v_or_b32_e32 v11, 0x800, v0
v_mul_u32_u24_e32 v10, 0x1c72, v11
v_lshrrev_b32_e32 v10, 16, v10
v_mul_lo_u16_e32 v14, 9, v10
v_sub_u16_e32 v11, v11, v14
v_mad_i64_i32 v[12:13], s[24:25], s14, v10, v[12:13]
v_lshlrev_b16_e32 v78, 4, v11
v_lshl_add_u64 v[12:13], v[12:13], 0, v[78:79]
v_or_b32_e32 v124, 0x8000, v47
v_or_b32_e32 v125, 0x9000, v47
v_readfirstlane_b32 s3, v124
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[12:13], off
v_mov_b64_e32 v[12:13], s[6:7]
v_mad_i64_i32 v[14:15], s[24:25], s15, v2, v[12:13]
v_lshl_add_u64 v[14:15], v[14:15], 0, v[62:63]
v_readfirstlane_b32 s3, v125
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[14:15], off
v_or_b32_e32 v126, 0xa000, v47
v_mad_i64_i32 v[14:15], s[24:25], s15, v3, v[12:13]
v_lshl_add_u64 v[14:15], v[14:15], 0, v[64:65]
v_readfirstlane_b32 s3, v126
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[14:15], off
v_or_b32_e32 v127, 0xb000, v47
v_mad_i64_i32 v[14:15], s[24:25], s15, v4, v[12:13]
v_lshl_add_u64 v[14:15], v[14:15], 0, v[66:67]
v_readfirstlane_b32 s3, v127
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[14:15], off
v_or_b32_e32 v128, 0xc000, v47
v_mad_i64_i32 v[14:15], s[24:25], s15, v5, v[12:13]
v_lshl_add_u64 v[14:15], v[14:15], 0, v[68:69]
v_readfirstlane_b32 s3, v128
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[14:15], off
v_or_b32_e32 v129, 0xd000, v47
v_mad_i64_i32 v[14:15], s[24:25], s15, v6, v[12:13]
v_lshl_add_u64 v[14:15], v[14:15], 0, v[70:71]
v_readfirstlane_b32 s3, v129
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[14:15], off
v_lshrrev_b32_e32 v61, 7, v0
v_mad_i64_i32 v[14:15], s[24:25], s15, v7, v[12:13]
v_lshl_add_u64 v[14:15], v[14:15], 0, v[72:73]
v_or_b32_e32 v130, 0xe000, v47
v_lshl_or_b32 v16, s22, 1, v61
v_readfirstlane_b32 s3, v130
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[14:15], off
v_or_b32_e32 v131, 0xf000, v47
v_mad_i64_i32 v[14:15], s[24:25], s15, v8, v[12:13]
v_lshl_add_u64 v[14:15], v[14:15], 0, v[74:75]
v_mul_lo_u32 v11, v16, s21
v_and_b32_e32 v60, 63, v0
v_bfe_u32 v1, v0, 6, 1
s_lshl_b32 s2, s20, 1
v_readfirstlane_b32 s3, v131
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[14:15], off
v_or_b32_e32 v132, 0x10000, v47
v_mad_i64_i32 v[14:15], s[24:25], s15, v9, v[12:13]
v_mad_i64_i32 v[12:13], s[24:25], s15, v10, v[12:13]
v_lshlrev_b32_e32 v134, 6, v11
v_or_b32_e32 v17, s2, v1
v_readfirstlane_b32 s3, v132
v_lshl_add_u64 v[12:13], v[12:13], 0, v[78:79]
v_or_b32_e32 v133, 0x11000, v47
v_or_b32_e32 v11, v134, v60
v_lshl_add_u64 v[14:15], v[14:15], 0, v[76:77]
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[14:15], off
v_readfirstlane_b32 s3, v133
s_mov_b32 m0, s3
global_load_lds_dwordx4 v[12:13], off
s_mov_b32 s23, 0
v_mad_i64_i32 v[12:13], s[24:25], v11, 12, s[8:9]
v_mul_lo_u32 v11, v17, s21
v_lshl_or_b32 v11, v11, 6, v60
global_load_dwordx3 v[48:50], v[12:13], off
v_mad_i64_i32 v[14:15], s[24:25], v11, 12, s[10:11]
global_load_dwordx3 v[44:46], v[14:15], off
s_cmp_lt_i32 s13, 6
v_and_b32_e32 v118, 31, v0
v_lshrrev_b32_e32 v121, 5, v60
s_cbranch_scc1 1698
v_mad_i64_i32 v[80:81], s[24:25], s14, v2, 0
v_mad_i64_i32 v[98:99], s[24:25], s15, v2, 0
v_and_b32_e32 v2, 31, v0
v_lshl_or_b32 v2, v1, 7, v2
s_add_i32 s3, 0, 0x12000
v_mad_i64_i32 v[82:83], s[24:25], s14, v3, 0
v_mad_i64_i32 v[100:101], s[24:25], s15, v3, 0
v_and_b32_e32 v3, 0x9f, v0
v_mul_u32_u24_e32 v155, 0x90, v2
v_mad_u32_u24 v2, v121, 24, s3
s_add_i32 s3, 0, 0x1b000
v_mul_u32_u24_e32 v154, 0x90, v3
v_mad_i64_i32 v[84:85], s[24:25], s14, v4, 0
v_mad_i64_i32 v[102:103], s[24:25], s15, v4, 0
v_mad_u32_u24 v3, v121, 24, s3
v_mad_i64_i32 v[86:87], s[24:25], s14, v5, 0
v_mad_i64_i32 v[104:105], s[24:25], s15, v5, 0
s_add_i32 s3, 0, 0x12030
v_mad_u32_u24 v4, v121, 24, s3
v_mad_i64_i32 v[88:89], s[24:25], s14, v6, 0
v_mad_i64_i32 v[92:93], s[24:25], s14, v8, 0
s_add_i32 s3, 0, 0x1b030
v_mad_u32_u24 v5, v121, 24, s3
v_mad_i64_i32 v[106:107], s[24:25], s15, v6, 0
v_mad_i64_i32 v[110:111], s[24:25], s15, v8, 0
v_add_u32_e32 v8, s2, v1
v_mad_i64_i32 v[90:91], s[24:25], s14, v7, 0
s_add_i32 s3, 0, 0x12060
v_mad_u32_u24 v6, v121, 24, s3
v_mad_i64_i32 v[94:95], s[24:25], s14, v9, 0
v_mad_i64_i32 v[96:97], s[24:25], s14, v10, 0
s_add_i32 s3, 0, 0x1b060
v_mad_i64_i32 v[108:109], s[24:25], s15, v7, 0
v_mad_i64_i32 v[112:113], s[24:25], s15, v9, 0
v_mad_i64_i32 v[114:115], s[24:25], s15, v10, 0
v_mad_u32_u24 v7, v121, 24, s3
v_mul_lo_u32 v8, s21, v8
v_or_b32_e32 v135, 0x12000, v47
v_or_b32_e32 v136, 0x13000, v47
v_or_b32_e32 v137, 0x14000, v47
v_or_b32_e32 v138, 0x15000, v47
v_or_b32_e32 v139, 0x16000, v47
v_or_b32_e32 v140, 0x17000, v47
v_or_b32_e32 v141, 0x18000, v47
v_or_b32_e32 v142, 0x19000, v47
v_or_b32_e32 v143, 0x1a000, v47
v_or_b32_e32 v144, 0x1b000, v47
v_or_b32_e32 v145, 0x1c000, v47
v_or_b32_e32 v146, 0x1d000, v47
v_or_b32_e32 v147, 0x1e000, v47
v_or_b32_e32 v148, 0x1f000, v47
v_or_b32_e32 v149, 0x20000, v47
v_or_b32_e32 v150, 0x21000, v47
v_or_b32_e32 v151, 0x22000, v47
v_or_b32_e32 v152, 0x23000, v47
v_mad_u32_u24 v153, v121, 24, 0
s_movk_i32 s24, 0x90
v_lshlrev_b32_e32 v156, 6, v8
v_accvgpr_write_b32 a255, 0
v_accvgpr_write_b32 a254, 0
v_accvgpr_write_b32 a253, 0
v_accvgpr_write_b32 a252, 0
v_accvgpr_write_b32 a251, 0
v_accvgpr_write_b32 a250, 0
v_accvgpr_write_b32 a249, 0
v_accvgpr_write_b32 a248, 0
v_accvgpr_write_b32 a247, 0
v_accvgpr_write_b32 a246, 0
v_accvgpr_write_b32 a245, 0
v_accvgpr_write_b32 a244, 0
v_accvgpr_write_b32 a243, 0
v_accvgpr_write_b32 a242, 0
v_accvgpr_write_b32 a241, 0
v_accvgpr_write_b32 a240, 0
v_accvgpr_write_b32 a239, 0
v_accvgpr_write_b32 a238, 0
v_accvgpr_write_b32 a237, 0
v_accvgpr_write_b32 a236, 0
v_accvgpr_write_b32 a235, 0
v_accvgpr_write_b32 a234, 0
v_accvgpr_write_b32 a233, 0
v_accvgpr_write_b32 a232, 0
v_accvgpr_write_b32 a231, 0
v_accvgpr_write_b32 a230, 0
v_accvgpr_write_b32 a229, 0
v_accvgpr_write_b32 a228, 0
v_accvgpr_write_b32 a227, 0
v_accvgpr_write_b32 a226, 0
v_accvgpr_write_b32 a225, 0
v_accvgpr_write_b32 a224, 0
v_accvgpr_write_b32 a223, 0
v_accvgpr_write_b32 a222, 0
v_accvgpr_write_b32 a221, 0
v_accvgpr_write_b32 a220, 0
v_accvgpr_write_b32 a219, 0
v_accvgpr_write_b32 a218, 0
v_accvgpr_write_b32 a217, 0
v_accvgpr_write_b32 a216, 0
v_accvgpr_write_b32 a215, 0
v_accvgpr_write_b32 a214, 0
v_accvgpr_write_b32 a213, 0
v_accvgpr_write_b32 a212, 0
v_accvgpr_write_b32 a211, 0
v_accvgpr_write_b32 a210, 0
v_accvgpr_write_b32 a209, 0
v_accvgpr_write_b32 a208, 0
v_accvgpr_write_b32 a207, 0
v_accvgpr_write_b32 a206, 0
v_accvgpr_write_b32 a205, 0
v_accvgpr_write_b32 a204, 0
v_accvgpr_write_b32 a203, 0
v_accvgpr_write_b32 a202, 0
v_accvgpr_write_b32 a201, 0
v_accvgpr_write_b32 a200, 0
v_accvgpr_write_b32 a199, 0
v_accvgpr_write_b32 a198, 0
v_accvgpr_write_b32 a197, 0
v_accvgpr_write_b32 a196, 0
v_accvgpr_write_b32 a195, 0
v_accvgpr_write_b32 a194, 0
v_accvgpr_write_b32 a193, 0
v_accvgpr_write_b32 a192, 0
v_accvgpr_write_b32 a191, 0
v_accvgpr_write_b32 a190, 0
v_accvgpr_write_b32 a189, 0
v_accvgpr_write_b32 a188, 0
v_accvgpr_write_b32 a187, 0
v_accvgpr_write_b32 a186, 0
v_accvgpr_write_b32 a185, 0
v_accvgpr_write_b32 a184, 0
v_accvgpr_write_b32 a183, 0
v_accvgpr_write_b32 a182, 0
v_accvgpr_write_b32 a181, 0
v_accvgpr_write_b32 a180, 0
v_accvgpr_write_b32 a179, 0
v_accvgpr_write_b32 a178, 0
v_accvgpr_write_b32 a177, 0
v_accvgpr_write_b32 a176, 0
v_accvgpr_write_b32 a175, 0
v_accvgpr_write_b32 a174, 0
v_accvgpr_write_b32 a173, 0
v_accvgpr_write_b32 a172, 0
v_accvgpr_write_b32 a171, 0
v_accvgpr_write_b32 a170, 0
v_accvgpr_write_b32 a169, 0
v_accvgpr_write_b32 a168, 0
v_accvgpr_write_b32 a167, 0
v_accvgpr_write_b32 a166, 0
v_accvgpr_write_b32 a165, 0
v_accvgpr_write_b32 a164, 0
v_accvgpr_write_b32 a163, 0
v_accvgpr_write_b32 a162, 0
v_accvgpr_write_b32 a161, 0
v_accvgpr_write_b32 a160, 0
v_accvgpr_write_b32 a159, 0
v_accvgpr_write_b32 a158, 0
v_accvgpr_write_b32 a157, 0
v_accvgpr_write_b32 a156, 0
v_accvgpr_write_b32 a155, 0
v_accvgpr_write_b32 a154, 0
v_accvgpr_write_b32 a153, 0
v_accvgpr_write_b32 a152, 0
v_accvgpr_write_b32 a151, 0
v_accvgpr_write_b32 a150, 0
v_accvgpr_write_b32 a149, 0
v_accvgpr_write_b32 a148, 0
v_accvgpr_write_b32 a147, 0
v_accvgpr_write_b32 a146, 0
v_accvgpr_write_b32 a145, 0
v_accvgpr_write_b32 a144, 0
v_accvgpr_write_b32 a143, 0
v_accvgpr_write_b32 a142, 0
v_accvgpr_write_b32 a141, 0
v_accvgpr_write_b32 a140, 0
v_accvgpr_write_b32 a139, 0
v_accvgpr_write_b32 a138, 0
v_accvgpr_write_b32 a137, 0
v_accvgpr_write_b32 a136, 0
v_accvgpr_write_b32 a135, 0
v_accvgpr_write_b32 a134, 0
v_accvgpr_write_b32 a133, 0
v_accvgpr_write_b32 a132, 0
v_accvgpr_write_b32 a131, 0
v_accvgpr_write_b32 a130, 0
v_accvgpr_write_b32 a129, 0
v_accvgpr_write_b32 a128, 0
v_accvgpr_write_b32 a127, 0
v_accvgpr_write_b32 a126, 0
v_accvgpr_write_b32 a125, 0
v_accvgpr_write_b32 a124, 0
v_accvgpr_write_b32 a123, 0
v_accvgpr_write_b32 a122, 0
v_accvgpr_write_b32 a121, 0
v_accvgpr_write_b32 a120, 0
v_accvgpr_write_b32 a119, 0
v_accvgpr_write_b32 a118, 0
v_accvgpr_write_b32 a117, 0
v_accvgpr_write_b32 a116, 0
v_accvgpr_write_b32 a115, 0
v_accvgpr_write_b32 a114, 0
v_accvgpr_write_b32 a113, 0
v_accvgpr_write_b32 a112, 0
v_accvgpr_write_b32 a111, 0
v_accvgpr_write_b32 a110, 0
v_accvgpr_write_b32 a109, 0
v_accvgpr_write_b32 a108, 0
v_accvgpr_write_b32 a107, 0
v_accvgpr_write_b32 a106, 0
v_accvgpr_write_b32 a105, 0
v_accvgpr_write_b32 a104, 0
v_accvgpr_write_b32 a103, 0
v_accvgpr_write_b32 a102, 0
v_accvgpr_write_b32 a101, 0
v_accvgpr_write_b32 a100, 0
v_accvgpr_write_b32 a99, 0
v_accvgpr_write_b32 a98, 0
v_accvgpr_write_b32 a97, 0
v_accvgpr_write_b32 a96, 0
v_accvgpr_write_b32 a95, 0
v_accvgpr_write_b32 a94, 0
v_accvgpr_write_b32 a93, 0
v_accvgpr_write_b32 a92, 0
v_accvgpr_write_b32 a91, 0
v_accvgpr_write_b32 a90, 0
v_accvgpr_write_b32 a89, 0
v_accvgpr_write_b32 a88, 0
v_accvgpr_write_b32 a87, 0
v_accvgpr_write_b32 a86, 0
v_accvgpr_write_b32 a85, 0
v_accvgpr_write_b32 a84, 0
v_accvgpr_write_b32 a83, 0
v_accvgpr_write_b32 a82, 0
v_accvgpr_write_b32 a81, 0
v_accvgpr_write_b32 a80, 0
v_accvgpr_write_b32 a79, 0
v_accvgpr_write_b32 a78, 0
v_accvgpr_write_b32 a77, 0
v_accvgpr_write_b32 a76, 0
v_accvgpr_write_b32 a75, 0
v_accvgpr_write_b32 a74, 0
v_accvgpr_write_b32 a73, 0
v_accvgpr_write_b32 a72, 0
v_accvgpr_write_b32 a71, 0
v_accvgpr_write_b32 a70, 0
v_accvgpr_write_b32 a69, 0
v_accvgpr_write_b32 a68, 0
v_accvgpr_write_b32 a67, 0
v_accvgpr_write_b32 a66, 0
v_accvgpr_write_b32 a65, 0
v_accvgpr_write_b32 a64, 0
v_accvgpr_write_b32 a63, 0
v_accvgpr_write_b32 a62, 0
v_accvgpr_write_b32 a61, 0
v_accvgpr_write_b32 a60, 0
v_accvgpr_write_b32 a59, 0
v_accvgpr_write_b32 a58, 0
v_accvgpr_write_b32 a57, 0
v_accvgpr_write_b32 a56, 0
v_accvgpr_write_b32 a55, 0
v_accvgpr_write_b32 a54, 0
v_accvgpr_write_b32 a53, 0
v_accvgpr_write_b32 a52, 0
v_accvgpr_write_b32 a51, 0
v_accvgpr_write_b32 a50, 0
v_accvgpr_write_b32 a49, 0
v_accvgpr_write_b32 a48, 0
v_accvgpr_write_b32 a47, 0
v_accvgpr_write_b32 a46, 0
v_accvgpr_write_b32 a45, 0
v_accvgpr_write_b32 a44, 0
v_accvgpr_write_b32 a43, 0
v_accvgpr_write_b32 a42, 0
v_accvgpr_write_b32 a41, 0
v_accvgpr_write_b32 a40, 0
v_accvgpr_write_b32 a39, 0
v_accvgpr_write_b32 a38, 0
v_accvgpr_write_b32 a37, 0
v_accvgpr_write_b32 a36, 0
v_accvgpr_write_b32 a35, 0
v_accvgpr_write_b32 a34, 0
v_accvgpr_write_b32 a33, 0
v_accvgpr_write_b32 a32, 0
v_accvgpr_write_b32 a31, 0
v_accvgpr_write_b32 a30, 0
v_accvgpr_write_b32 a29, 0
v_accvgpr_write_b32 a28, 0
v_accvgpr_write_b32 a27, 0
v_accvgpr_write_b32 a26, 0
v_accvgpr_write_b32 a25, 0
v_accvgpr_write_b32 a24, 0
v_accvgpr_write_b32 a23, 0
v_accvgpr_write_b32 a22, 0
v_accvgpr_write_b32 a21, 0
v_accvgpr_write_b32 a20, 0
v_accvgpr_write_b32 a19, 0
v_accvgpr_write_b32 a18, 0
v_accvgpr_write_b32 a17, 0
v_accvgpr_write_b32 a16, 0
v_accvgpr_write_b32 a15, 0
v_accvgpr_write_b32 a14, 0
v_accvgpr_write_b32 a13, 0
v_accvgpr_write_b32 a12, 0
v_accvgpr_write_b32 a11, 0
v_accvgpr_write_b32 a10, 0
v_accvgpr_write_b32 a9, 0
v_accvgpr_write_b32 a8, 0
v_accvgpr_write_b32 a7, 0
v_accvgpr_write_b32 a6, 0
v_accvgpr_write_b32 a5, 0
v_accvgpr_write_b32 a4, 0
v_accvgpr_write_b32 a3, 0
v_accvgpr_write_b32 a2, 0
v_accvgpr_write_b32 a1, 0
v_accvgpr_write_b32 a0, 0
v_add_u32_e32 v157, v2, v154
v_add_u32_e32 v158, v3, v155
v_add_u32_e32 v159, v4, v154
v_add_u32_e32 v160, v5, v155
v_add_u32_e32 v161, v6, v154
v_add_u32_e32 v162, v7, v155
s_branch 370
v_add_u32_e32 v2, 16, v157
s_barrier
ds_read2st64_b64 v[4:7], v2 offset1:9
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27
v_add_u32_e32 v26, 16, v158
v_and_b32_e32 v163, 0xff, v56
v_bfe_u32 v170, v56, 8, 8
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v12, v4
v_mov_b32_e32 v13, v5
ds_read_b128 v[8:11], v157
ds_read_b128 v[2:5], v157 offset:4608
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v24, v16
v_mov_b32_e32 v25, v17
ds_read2st64_b64 v[28:31], v26 offset1:9
ds_read_b128 v[20:23], v157 offset:9216
ds_read_b128 v[14:17], v157 offset:13824
ds_read2st64_b64 v[40:43], v26 offset0:18 offset1:27
v_and_b32_e32 v172, 0xff, v52
v_bfe_u32 v173, v52, 8, 8
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v36, v28
v_mov_b32_e32 v37, v29
ds_read_b128 v[32:35], v158
ds_read_b128 v[26:29], v158 offset:4608
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v168, v40
v_mov_b32_e32 v169, v41
ds_read_b128 v[164:167], v158 offset:9216
ds_read_b128 v[38:41], v158 offset:13824
v_bfe_u32 v174, v52, 16, 8
v_lshrrev_b32_e32 v52, 24, v52
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[8:13], a[240:255], v172, v163 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v173, v163 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[164:169], v[8:13], a[208:223], v174, v163 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[38:43], v[8:13], a[192:207], v52, v163 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[2:7], a[176:191], v172, v170 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[2:7], a[160:175], v173, v170 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[164:169], v[2:7], a[144:159], v174, v170 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[38:43], v[2:7], a[128:143], v52, v170 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_add_u32_e32 v2, 16, v159
v_bfe_u32 v171, v56, 16, 8
v_lshrrev_b32_e32 v56, 24, v56
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[20:25], a[112:127], v172, v171 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[26:31], v[20:25], a[96:111], v173, v171 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[164:169], v[20:25], a[80:95], v174, v171 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[38:43], v[20:25], a[64:79], v52, v171 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[14:19], a[48:63], v172, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[14:19], a[32:47], v173, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[164:169], v[14:19], a[16:31], v174, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:43], v[14:19], a[0:15], v52, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read2st64_b64 v[4:7], v2 offset1:9
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27
v_add_u32_e32 v26, 16, v160
v_and_b32_e32 v52, 0xff, v57
v_bfe_u32 v56, v57, 8, 8
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v12, v4
v_mov_b32_e32 v13, v5
ds_read_b128 v[8:11], v159
ds_read_b128 v[2:5], v159 offset:4608
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v24, v16
v_mov_b32_e32 v25, v17
ds_read2st64_b64 v[28:31], v26 offset1:9
ds_read_b128 v[20:23], v159 offset:9216
ds_read_b128 v[14:17], v159 offset:13824
ds_read2st64_b64 v[40:43], v26 offset0:18 offset1:27
v_and_b32_e32 v170, 0xff, v53
v_bfe_u32 v171, v53, 8, 8
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v36, v28
v_mov_b32_e32 v37, v29
ds_read_b128 v[32:35], v160
ds_read_b128 v[26:29], v160 offset:4608
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v168, v40
v_mov_b32_e32 v169, v41
ds_read_b128 v[164:167], v160 offset:9216
ds_read_b128 v[38:41], v160 offset:13824
v_bfe_u32 v172, v53, 16, 8
v_lshrrev_b32_e32 v53, 24, v53
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[8:13], a[240:255], v170, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v171, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[164:169], v[8:13], a[208:223], v172, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[38:43], v[8:13], a[192:207], v53, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[2:7], a[176:191], v170, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[2:7], a[160:175], v171, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[164:169], v[2:7], a[144:159], v172, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[38:43], v[2:7], a[128:143], v53, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_add_u32_e32 v2, 16, v161
v_bfe_u32 v163, v57, 16, 8
v_lshrrev_b32_e32 v57, 24, v57
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[20:25], a[112:127], v170, v163 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[26:31], v[20:25], a[96:111], v171, v163 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[164:169], v[20:25], a[80:95], v172, v163 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[38:43], v[20:25], a[64:79], v53, v163 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[14:19], a[48:63], v170, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[14:19], a[32:47], v171, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[164:169], v[14:19], a[16:31], v172, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:43], v[14:19], a[0:15], v53, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read2st64_b64 v[4:7], v2 offset1:9
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27
v_add_u32_e32 v26, 16, v162
s_addk_i32 s24, 0x120
s_add_i32 s2, s23, 1
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v12, v4
v_mov_b32_e32 v13, v5
ds_read_b128 v[8:11], v161
ds_read_b128 v[2:5], v161 offset:4608
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v24, v16
v_mov_b32_e32 v25, v17
ds_read2st64_b64 v[28:31], v26 offset1:9
ds_read_b128 v[20:23], v161 offset:9216
ds_read_b128 v[14:17], v161 offset:13824
ds_read2st64_b64 v[40:43], v26 offset0:18 offset1:27
v_add_u32_e32 v156, 0x80, v156
s_cmp_lt_i32 s2, s21
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v36, v28
v_mov_b32_e32 v37, v29
ds_read_b128 v[32:35], v162
ds_read_b128 v[26:29], v162 offset:4608
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v168, v40
v_mov_b32_e32 v169, v41
ds_read_b128 v[164:167], v162 offset:9216
ds_read_b128 v[38:41], v162 offset:13824
v_add_u32_e32 v134, 0x80, v134
v_and_b32_e32 v52, 0xff, v58
v_bfe_u32 v53, v58, 8, 8
v_bfe_u32 v56, v58, 16, 8
v_lshrrev_b32_e32 v57, 24, v58
v_and_b32_e32 v58, 0xff, v54
v_bfe_u32 v163, v54, 8, 8
v_bfe_u32 v170, v54, 16, 8
v_lshrrev_b32_e32 v54, 24, v54
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[8:13], a[240:255], v58, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v163, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[164:169], v[8:13], a[208:223], v170, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[38:43], v[8:13], a[192:207], v54, v52 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[2:7], a[176:191], v58, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[2:7], a[160:175], v163, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[164:169], v[2:7], a[144:159], v170, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[38:43], v[2:7], a[128:143], v54, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[20:25], a[112:127], v58, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[26:31], v[20:25], a[96:111], v163, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[164:169], v[20:25], a[80:95], v170, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[38:43], v[20:25], a[64:79], v54, v56 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[14:19], a[48:63], v58, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[14:19], a[32:47], v163, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[164:169], v[14:19], a[16:31], v170, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:43], v[14:19], a[0:15], v54, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_barrier
s_cbranch_scc0 954
s_add_u32 s2, s4, s24
s_addc_u32 s3, s5, 0
v_lshl_add_u64 v[2:3], s[2:3], 0, v[80:81]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[62:63]
v_readfirstlane_b32 s25, v135
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[82:83]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[64:65]
v_readfirstlane_b32 s25, v136
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[84:85]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[66:67]
v_readfirstlane_b32 s25, v137
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[86:87]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[68:69]
v_readfirstlane_b32 s25, v138
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[88:89]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[70:71]
v_readfirstlane_b32 s25, v139
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[90:91]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[72:73]
v_readfirstlane_b32 s25, v140
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[92:93]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[74:75]
v_readfirstlane_b32 s25, v141
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[94:95]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[76:77]
v_readfirstlane_b32 s25, v142
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[96:97]
v_readfirstlane_b32 s2, v143
v_lshl_add_u64 v[2:3], v[2:3], 0, v[78:79]
s_mov_b32 m0, s2
global_load_lds_dwordx4 v[2:3], off
s_add_u32 s2, s6, s24
s_addc_u32 s3, s7, 0
v_lshl_add_u64 v[2:3], s[2:3], 0, v[98:99]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[62:63]
v_readfirstlane_b32 s25, v144
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[100:101]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[64:65]
v_readfirstlane_b32 s25, v145
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[102:103]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[66:67]
v_readfirstlane_b32 s25, v146
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[104:105]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[68:69]
v_readfirstlane_b32 s25, v147
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[106:107]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[70:71]
v_readfirstlane_b32 s25, v148
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[108:109]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[72:73]
v_readfirstlane_b32 s25, v149
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[110:111]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[74:75]
v_readfirstlane_b32 s25, v150
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[112:113]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[76:77]
v_readfirstlane_b32 s25, v151
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[2:3], 0, v[114:115]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[78:79]
v_readfirstlane_b32 s2, v152
v_add_u32_e32 v163, v60, v134
s_mov_b32 m0, s2
global_load_lds_dwordx4 v[2:3], off
v_add_u32_e32 v2, 64, v163
v_add_u32_e32 v164, v60, v156
v_mad_i64_i32 v[2:3], s[2:3], v2, 12, s[8:9]
v_add_u32_e32 v4, 64, v164
v_add_u32_e32 v165, v153, v154
v_mad_i64_i32 v[4:5], s[2:3], v4, 12, s[10:11]
global_load_dwordx3 v[56:58], v[2:3], off
v_add_u32_e32 v2, 16, v165
global_load_dwordx3 v[52:54], v[4:5], off
s_barrier
ds_read2st64_b64 v[40:43], v2 offset1:9
ds_read2st64_b64 v[4:7], v2 offset0:18 offset1:27
v_add_u32_e32 v166, v153, v155
v_add_u32_e32 v8, 16, v166
v_and_b32_e32 v167, 0xff, v48
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v172, v40
v_mov_b32_e32 v173, v41
ds_read_b128 v[168:171], v165
ds_read_b128 v[38:41], v165 offset:4608
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v24, v4
v_mov_b32_e32 v25, v5
ds_read2st64_b64 v[16:19], v8 offset0:72 offset1:81
ds_read_b128 v[20:23], v165 offset:9216
ds_read_b128 v[2:5], v165 offset:13824
ds_read2st64_b64 v[10:13], v8 offset0:90 offset1:99
v_bfe_u32 v174, v48, 8, 8
v_bfe_u32 v175, v48, 16, 8
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v36, v16
v_mov_b32_e32 v37, v17
ds_read_b128 v[32:35], v166 offset:36864
ds_read_b128 v[14:17], v166 offset:41472
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v30, v10
v_mov_b32_e32 v31, v11
ds_read_b128 v[26:29], v166 offset:46080
ds_read_b128 v[8:11], v166 offset:50688
v_lshrrev_b32_e32 v176, 24, v48
v_and_b32_e32 v177, 0xff, v44
v_bfe_u32 v178, v44, 8, 8
v_bfe_u32 v179, v44, 16, 8
v_lshrrev_b32_e32 v180, 24, v44
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[168:173], a[240:255], v177, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[14:19], v[168:173], a[224:239], v178, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[26:31], v[168:173], a[208:223], v179, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[8:13], v[168:173], a[192:207], v180, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[38:43], a[176:191], v177, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[14:19], v[38:43], a[160:175], v178, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[26:31], v[38:43], a[144:159], v179, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[8:13], v[38:43], a[128:143], v180, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[20:25], a[112:127], v177, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[14:19], v[20:25], a[96:111], v178, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[26:31], v[20:25], a[80:95], v179, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[8:13], v[20:25], a[64:79], v180, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[2:7], a[48:63], v177, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[14:19], v[2:7], a[32:47], v178, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[26:31], v[2:7], a[16:31], v179, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[8:13], v[2:7], a[0:15], v180, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_add_u32_e32 v2, 64, v165
ds_read2st64_b64 v[4:7], v2 offset1:9
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27
v_add_u32_e32 v26, 64, v166
v_and_b32_e32 v167, 0xff, v49
v_bfe_u32 v174, v49, 8, 8
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v12, v4
v_mov_b32_e32 v13, v5
ds_read_b128 v[8:11], v165 offset:48
ds_read_b128 v[2:5], v165 offset:4656
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v24, v16
v_mov_b32_e32 v25, v17
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81
ds_read_b128 v[20:23], v165 offset:9264
ds_read_b128 v[14:17], v165 offset:13872
ds_read2st64_b64 v[40:43], v26 offset0:90 offset1:99
v_and_b32_e32 v177, 0xff, v45
v_bfe_u32 v178, v45, 8, 8
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v36, v28
v_mov_b32_e32 v37, v29
ds_read_b128 v[32:35], v166 offset:36912
ds_read_b128 v[26:29], v166 offset:41520
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v172, v40
v_mov_b32_e32 v173, v41
ds_read_b128 v[168:171], v166 offset:46128
ds_read_b128 v[38:41], v166 offset:50736
v_bfe_u32 v179, v45, 16, 8
v_lshrrev_b32_e32 v180, 24, v45
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[8:13], a[240:255], v177, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v178, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[168:173], v[8:13], a[208:223], v179, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[38:43], v[8:13], a[192:207], v180, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[2:7], a[176:191], v177, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[2:7], a[160:175], v178, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[168:173], v[2:7], a[144:159], v179, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[38:43], v[2:7], a[128:143], v180, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_add_u32_e32 v2, 0x70, v165
v_bfe_u32 v175, v49, 16, 8
v_lshrrev_b32_e32 v176, 24, v49
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[20:25], a[112:127], v177, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[26:31], v[20:25], a[96:111], v178, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[168:173], v[20:25], a[80:95], v179, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[38:43], v[20:25], a[64:79], v180, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[14:19], a[48:63], v177, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[14:19], a[32:47], v178, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[168:173], v[14:19], a[16:31], v179, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:43], v[14:19], a[0:15], v180, v176 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read2st64_b64 v[4:7], v2 offset1:9
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27
v_add_u32_e32 v26, 0x70, v166
s_add_i32 s23, s23, 2
s_cmp_ge_i32 s23, s21
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v12, v4
v_mov_b32_e32 v13, v5
ds_read_b128 v[8:11], v165 offset:96
ds_read_b128 v[2:5], v165 offset:4704
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v24, v16
v_mov_b32_e32 v25, v17
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81
ds_read_b128 v[20:23], v165 offset:9312
ds_read_b128 v[14:17], v165 offset:13920
ds_read2st64_b64 v[40:43], v26 offset0:90 offset1:99
s_cselect_b64 s[2:3], -1, 0
s_and_b64 vcc, exec, s[2:3]
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v36, v28
v_mov_b32_e32 v37, v29
ds_read_b128 v[32:35], v166 offset:36960
ds_read_b128 v[26:29], v166 offset:41568
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v172, v40
v_mov_b32_e32 v173, v41
ds_read_b128 v[168:171], v166 offset:46176
ds_read_b128 v[38:41], v166 offset:50784
v_and_b32_e32 v167, 0xff, v50
v_bfe_u32 v174, v50, 8, 8
v_bfe_u32 v175, v50, 16, 8
v_lshrrev_b32_e32 v165, 24, v50
v_and_b32_e32 v176, 0xff, v46
v_bfe_u32 v177, v46, 8, 8
v_bfe_u32 v178, v46, 16, 8
v_lshrrev_b32_e32 v166, 24, v46
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[8:13], a[240:255], v176, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v177, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[168:173], v[8:13], a[208:223], v178, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[38:43], v[8:13], a[192:207], v166, v167 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[2:7], a[176:191], v176, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[2:7], a[160:175], v177, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[168:173], v[2:7], a[144:159], v178, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[38:43], v[2:7], a[128:143], v166, v174 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[20:25], a[112:127], v176, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[26:31], v[20:25], a[96:111], v177, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[168:173], v[20:25], a[80:95], v178, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[38:43], v[20:25], a[64:79], v166, v175 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[14:19], a[48:63], v176, v165 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[14:19], a[32:47], v177, v165 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[168:173], v[14:19], a[16:31], v178, v165 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:43], v[14:19], a[0:15], v166, v165 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_barrier
s_cbranch_vccnz 162
s_add_i32 s25, s24, 0x90
s_add_u32 s26, s4, s25
s_addc_u32 s27, s5, 0
v_lshl_add_u64 v[2:3], s[26:27], 0, v[80:81]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[62:63]
v_readfirstlane_b32 s28, v47
s_mov_b32 m0, s28
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[82:83]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[64:65]
v_readfirstlane_b32 s28, v51
s_mov_b32 m0, s28
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[84:85]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[66:67]
v_readfirstlane_b32 s28, v55
s_mov_b32 m0, s28
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[86:87]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[68:69]
v_readfirstlane_b32 s28, v59
s_mov_b32 m0, s28
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[88:89]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[70:71]
v_readfirstlane_b32 s28, v116
s_mov_b32 m0, s28
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[90:91]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[72:73]
v_readfirstlane_b32 s28, v117
s_mov_b32 m0, s28
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[92:93]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[74:75]
v_readfirstlane_b32 s28, v122
s_mov_b32 m0, s28
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[94:95]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[76:77]
v_readfirstlane_b32 s28, v123
s_mov_b32 m0, s28
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[96:97]
v_readfirstlane_b32 s26, v124
v_lshl_add_u64 v[2:3], v[2:3], 0, v[78:79]
s_mov_b32 m0, s26
global_load_lds_dwordx4 v[2:3], off
s_add_u32 s26, s6, s25
s_addc_u32 s27, s7, 0
v_lshl_add_u64 v[2:3], s[26:27], 0, v[98:99]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[62:63]
v_readfirstlane_b32 s25, v125
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[100:101]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[64:65]
v_readfirstlane_b32 s25, v126
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[102:103]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[66:67]
v_readfirstlane_b32 s25, v127
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[104:105]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[68:69]
v_readfirstlane_b32 s25, v128
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[106:107]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[70:71]
v_readfirstlane_b32 s25, v129
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[108:109]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[72:73]
v_readfirstlane_b32 s25, v130
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[110:111]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[74:75]
v_readfirstlane_b32 s25, v131
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[112:113]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[76:77]
v_readfirstlane_b32 s25, v132
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_lshl_add_u64 v[2:3], s[26:27], 0, v[114:115]
v_lshl_add_u64 v[2:3], v[2:3], 0, v[78:79]
v_readfirstlane_b32 s25, v133
s_mov_b32 m0, s25
global_load_lds_dwordx4 v[2:3], off
v_add_u32_e32 v2, 0x80, v163
v_mad_i64_i32 v[2:3], s[26:27], v2, 12, s[8:9]
v_add_u32_e32 v4, 0x80, v164
global_load_dwordx3 v[48:50], v[2:3], off
v_mad_i64_i32 v[4:5], s[26:27], v4, 12, s[10:11]
global_load_dwordx3 v[44:46], v[4:5], off
s_andn2_b64 vcc, exec, s[2:3]
s_cbranch_vccnz 64471
s_waitcnt vmcnt(0)
s_branch 64469
v_accvgpr_write_b32 a0, 0
v_accvgpr_mov_b32 a1, a0
v_accvgpr_mov_b32 a2, a0
v_accvgpr_mov_b32 a3, a0
v_accvgpr_mov_b32 a4, a0
v_accvgpr_mov_b32 a5, a0
v_accvgpr_mov_b32 a6, a0
v_accvgpr_mov_b32 a7, a0
v_accvgpr_mov_b32 a8, a0
v_accvgpr_mov_b32 a9, a0
v_accvgpr_mov_b32 a10, a0
v_accvgpr_mov_b32 a11, a0
v_accvgpr_mov_b32 a12, a0
v_accvgpr_mov_b32 a13, a0
v_accvgpr_mov_b32 a14, a0
v_accvgpr_mov_b32 a15, a0
v_accvgpr_mov_b32 a16, a0
v_accvgpr_mov_b32 a17, a0
v_accvgpr_mov_b32 a18, a0
v_accvgpr_mov_b32 a19, a0
v_accvgpr_mov_b32 a20, a0
v_accvgpr_mov_b32 a21, a0
v_accvgpr_mov_b32 a22, a0
v_accvgpr_mov_b32 a23, a0
v_accvgpr_mov_b32 a24, a0
v_accvgpr_mov_b32 a25, a0
v_accvgpr_mov_b32 a26, a0
v_accvgpr_mov_b32 a27, a0
v_accvgpr_mov_b32 a28, a0
v_accvgpr_mov_b32 a29, a0
v_accvgpr_mov_b32 a30, a0
v_accvgpr_mov_b32 a31, a0
v_accvgpr_mov_b32 a32, a0
v_accvgpr_mov_b32 a33, a0
v_accvgpr_mov_b32 a34, a0
v_accvgpr_mov_b32 a35, a0
v_accvgpr_mov_b32 a36, a0
v_accvgpr_mov_b32 a37, a0
v_accvgpr_mov_b32 a38, a0
v_accvgpr_mov_b32 a39, a0
v_accvgpr_mov_b32 a40, a0
v_accvgpr_mov_b32 a41, a0
v_accvgpr_mov_b32 a42, a0
v_accvgpr_mov_b32 a43, a0
v_accvgpr_mov_b32 a44, a0
v_accvgpr_mov_b32 a45, a0
v_accvgpr_mov_b32 a46, a0
v_accvgpr_mov_b32 a47, a0
v_accvgpr_mov_b32 a48, a0
v_accvgpr_mov_b32 a49, a0
v_accvgpr_mov_b32 a50, a0
v_accvgpr_mov_b32 a51, a0
v_accvgpr_mov_b32 a52, a0
v_accvgpr_mov_b32 a53, a0
v_accvgpr_mov_b32 a54, a0
v_accvgpr_mov_b32 a55, a0
v_accvgpr_mov_b32 a56, a0
v_accvgpr_mov_b32 a57, a0
v_accvgpr_mov_b32 a58, a0
v_accvgpr_mov_b32 a59, a0
v_accvgpr_mov_b32 a60, a0
v_accvgpr_mov_b32 a61, a0
v_accvgpr_mov_b32 a62, a0
v_accvgpr_mov_b32 a63, a0
v_accvgpr_mov_b32 a64, a0
v_accvgpr_mov_b32 a65, a0
v_accvgpr_mov_b32 a66, a0
v_accvgpr_mov_b32 a67, a0
v_accvgpr_mov_b32 a68, a0
v_accvgpr_mov_b32 a69, a0
v_accvgpr_mov_b32 a70, a0
v_accvgpr_mov_b32 a71, a0
v_accvgpr_mov_b32 a72, a0
v_accvgpr_mov_b32 a73, a0
v_accvgpr_mov_b32 a74, a0
v_accvgpr_mov_b32 a75, a0
v_accvgpr_mov_b32 a76, a0
v_accvgpr_mov_b32 a77, a0
v_accvgpr_mov_b32 a78, a0
v_accvgpr_mov_b32 a79, a0
v_accvgpr_mov_b32 a80, a0
v_accvgpr_mov_b32 a81, a0
v_accvgpr_mov_b32 a82, a0
v_accvgpr_mov_b32 a83, a0
v_accvgpr_mov_b32 a84, a0
v_accvgpr_mov_b32 a85, a0
v_accvgpr_mov_b32 a86, a0
v_accvgpr_mov_b32 a87, a0
v_accvgpr_mov_b32 a88, a0
v_accvgpr_mov_b32 a89, a0
v_accvgpr_mov_b32 a90, a0
v_accvgpr_mov_b32 a91, a0
v_accvgpr_mov_b32 a92, a0
v_accvgpr_mov_b32 a93, a0
v_accvgpr_mov_b32 a94, a0
v_accvgpr_mov_b32 a95, a0
v_accvgpr_mov_b32 a96, a0
v_accvgpr_mov_b32 a97, a0
v_accvgpr_mov_b32 a98, a0
v_accvgpr_mov_b32 a99, a0
v_accvgpr_mov_b32 a100, a0
v_accvgpr_mov_b32 a101, a0
v_accvgpr_mov_b32 a102, a0
v_accvgpr_mov_b32 a103, a0
v_accvgpr_mov_b32 a104, a0
v_accvgpr_mov_b32 a105, a0
v_accvgpr_mov_b32 a106, a0
v_accvgpr_mov_b32 a107, a0
v_accvgpr_mov_b32 a108, a0
v_accvgpr_mov_b32 a109, a0
v_accvgpr_mov_b32 a110, a0
v_accvgpr_mov_b32 a111, a0
v_accvgpr_mov_b32 a112, a0
v_accvgpr_mov_b32 a113, a0
v_accvgpr_mov_b32 a114, a0
v_accvgpr_mov_b32 a115, a0
v_accvgpr_mov_b32 a116, a0
v_accvgpr_mov_b32 a117, a0
v_accvgpr_mov_b32 a118, a0
v_accvgpr_mov_b32 a119, a0
v_accvgpr_mov_b32 a120, a0
v_accvgpr_mov_b32 a121, a0
v_accvgpr_mov_b32 a122, a0
v_accvgpr_mov_b32 a123, a0
v_accvgpr_mov_b32 a124, a0
v_accvgpr_mov_b32 a125, a0
v_accvgpr_mov_b32 a126, a0
v_accvgpr_mov_b32 a127, a0
v_accvgpr_mov_b32 a128, a0
v_accvgpr_mov_b32 a129, a0
v_accvgpr_mov_b32 a130, a0
v_accvgpr_mov_b32 a131, a0
v_accvgpr_mov_b32 a132, a0
v_accvgpr_mov_b32 a133, a0
v_accvgpr_mov_b32 a134, a0
v_accvgpr_mov_b32 a135, a0
v_accvgpr_mov_b32 a136, a0
v_accvgpr_mov_b32 a137, a0
v_accvgpr_mov_b32 a138, a0
v_accvgpr_mov_b32 a139, a0
v_accvgpr_mov_b32 a140, a0
v_accvgpr_mov_b32 a141, a0
v_accvgpr_mov_b32 a142, a0
v_accvgpr_mov_b32 a143, a0
v_accvgpr_mov_b32 a144, a0
v_accvgpr_mov_b32 a145, a0
v_accvgpr_mov_b32 a146, a0
v_accvgpr_mov_b32 a147, a0
v_accvgpr_mov_b32 a148, a0
v_accvgpr_mov_b32 a149, a0
v_accvgpr_mov_b32 a150, a0
v_accvgpr_mov_b32 a151, a0
v_accvgpr_mov_b32 a152, a0
v_accvgpr_mov_b32 a153, a0
v_accvgpr_mov_b32 a154, a0
v_accvgpr_mov_b32 a155, a0
v_accvgpr_mov_b32 a156, a0
v_accvgpr_mov_b32 a157, a0
v_accvgpr_mov_b32 a158, a0
v_accvgpr_mov_b32 a159, a0
v_accvgpr_mov_b32 a160, a0
v_accvgpr_mov_b32 a161, a0
v_accvgpr_mov_b32 a162, a0
v_accvgpr_mov_b32 a163, a0
v_accvgpr_mov_b32 a164, a0
v_accvgpr_mov_b32 a165, a0
v_accvgpr_mov_b32 a166, a0
v_accvgpr_mov_b32 a167, a0
v_accvgpr_mov_b32 a168, a0
v_accvgpr_mov_b32 a169, a0
v_accvgpr_mov_b32 a170, a0
v_accvgpr_mov_b32 a171, a0
v_accvgpr_mov_b32 a172, a0
v_accvgpr_mov_b32 a173, a0
v_accvgpr_mov_b32 a174, a0
v_accvgpr_mov_b32 a175, a0
v_accvgpr_mov_b32 a176, a0
v_accvgpr_mov_b32 a177, a0
v_accvgpr_mov_b32 a178, a0
v_accvgpr_mov_b32 a179, a0
v_accvgpr_mov_b32 a180, a0
v_accvgpr_mov_b32 a181, a0
v_accvgpr_mov_b32 a182, a0
v_accvgpr_mov_b32 a183, a0
v_accvgpr_mov_b32 a184, a0
v_accvgpr_mov_b32 a185, a0
v_accvgpr_mov_b32 a186, a0
v_accvgpr_mov_b32 a187, a0
v_accvgpr_mov_b32 a188, a0
v_accvgpr_mov_b32 a189, a0
v_accvgpr_mov_b32 a190, a0
v_accvgpr_mov_b32 a191, a0
v_accvgpr_mov_b32 a192, a0
v_accvgpr_mov_b32 a193, a0
v_accvgpr_mov_b32 a194, a0
v_accvgpr_mov_b32 a195, a0
v_accvgpr_mov_b32 a196, a0
v_accvgpr_mov_b32 a197, a0
v_accvgpr_mov_b32 a198, a0
v_accvgpr_mov_b32 a199, a0
v_accvgpr_mov_b32 a200, a0
v_accvgpr_mov_b32 a201, a0
v_accvgpr_mov_b32 a202, a0
v_accvgpr_mov_b32 a203, a0
v_accvgpr_mov_b32 a204, a0
v_accvgpr_mov_b32 a205, a0
v_accvgpr_mov_b32 a206, a0
v_accvgpr_mov_b32 a207, a0
v_accvgpr_mov_b32 a208, a0
v_accvgpr_mov_b32 a209, a0
v_accvgpr_mov_b32 a210, a0
v_accvgpr_mov_b32 a211, a0
v_accvgpr_mov_b32 a212, a0
v_accvgpr_mov_b32 a213, a0
v_accvgpr_mov_b32 a214, a0
v_accvgpr_mov_b32 a215, a0
v_accvgpr_mov_b32 a216, a0
v_accvgpr_mov_b32 a217, a0
v_accvgpr_mov_b32 a218, a0
v_accvgpr_mov_b32 a219, a0
v_accvgpr_mov_b32 a220, a0
v_accvgpr_mov_b32 a221, a0
v_accvgpr_mov_b32 a222, a0
v_accvgpr_mov_b32 a223, a0
v_accvgpr_mov_b32 a224, a0
v_accvgpr_mov_b32 a225, a0
v_accvgpr_mov_b32 a226, a0
v_accvgpr_mov_b32 a227, a0
v_accvgpr_mov_b32 a228, a0
v_accvgpr_mov_b32 a229, a0
v_accvgpr_mov_b32 a230, a0
v_accvgpr_mov_b32 a231, a0
v_accvgpr_mov_b32 a232, a0
v_accvgpr_mov_b32 a233, a0
v_accvgpr_mov_b32 a234, a0
v_accvgpr_mov_b32 a235, a0
v_accvgpr_mov_b32 a236, a0
v_accvgpr_mov_b32 a237, a0
v_accvgpr_mov_b32 a238, a0
v_accvgpr_mov_b32 a239, a0
v_accvgpr_mov_b32 a240, a0
v_accvgpr_mov_b32 a241, a0
v_accvgpr_mov_b32 a242, a0
v_accvgpr_mov_b32 a243, a0
v_accvgpr_mov_b32 a244, a0
v_accvgpr_mov_b32 a245, a0
v_accvgpr_mov_b32 a246, a0
v_accvgpr_mov_b32 a247, a0
v_accvgpr_mov_b32 a248, a0
v_accvgpr_mov_b32 a249, a0
v_accvgpr_mov_b32 a250, a0
v_accvgpr_mov_b32 a251, a0
v_accvgpr_mov_b32 a252, a0
v_accvgpr_mov_b32 a253, a0
v_accvgpr_mov_b32 a254, a0
v_accvgpr_mov_b32 a255, a0
s_load_dwordx4 s[0:3], s[0:1], 0x38
s_cmp_ge_i32 s23, s21
s_cbranch_scc1 377
v_and_b32_e32 v2, 0x9f, v0
v_mad_u32_u24 v14, v121, 24, 0
s_movk_i32 s8, 0x90
v_mad_u32_u24 v52, v2, s8, v14
v_add_u32_e32 v2, 16, v52
s_waitcnt vmcnt(0)
s_waitcnt lgkmcnt(0)
s_barrier
ds_read2st64_b64 v[4:7], v2 offset1:9
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27
v_and_b32_e32 v15, 31, v0
v_lshl_or_b32 v15, v1, 7, v15
v_mad_u32_u24 v56, v15, s8, v14
v_add_u32_e32 v26, 16, v56
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v12, v4
v_mov_b32_e32 v13, v5
ds_read_b128 v[8:11], v52
ds_read_b128 v[2:5], v52 offset:4608
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v24, v16
v_mov_b32_e32 v25, v17
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81
ds_read_b128 v[20:23], v52 offset:9216
ds_read_b128 v[14:17], v52 offset:13824
ds_read2st64_b64 v[40:43], v26 offset0:90 offset1:99
v_and_b32_e32 v53, 0xff, v48
v_bfe_u32 v54, v48, 8, 8
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v36, v28
v_mov_b32_e32 v37, v29
ds_read_b128 v[32:35], v56 offset:36864
ds_read_b128 v[26:29], v56 offset:41472
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v66, v40
v_mov_b32_e32 v67, v41
ds_read_b128 v[62:65], v56 offset:46080
ds_read_b128 v[38:41], v56 offset:50688
v_and_b32_e32 v58, 0xff, v44
v_bfe_u32 v68, v44, 8, 8
v_bfe_u32 v69, v44, 16, 8
v_lshrrev_b32_e32 v44, 24, v44
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[8:13], a[240:255], v58, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v68, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[62:67], v[8:13], a[208:223], v69, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[38:43], v[8:13], a[192:207], v44, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[2:7], a[176:191], v58, v54 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[2:7], a[160:175], v68, v54 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[62:67], v[2:7], a[144:159], v69, v54 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[38:43], v[2:7], a[128:143], v44, v54 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_add_u32_e32 v2, 64, v52
v_bfe_u32 v57, v48, 16, 8
v_lshrrev_b32_e32 v48, 24, v48
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[20:25], a[112:127], v58, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[26:31], v[20:25], a[96:111], v68, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[62:67], v[20:25], a[80:95], v69, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[38:43], v[20:25], a[64:79], v44, v57 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[14:19], a[48:63], v58, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[14:19], a[32:47], v68, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[62:67], v[14:19], a[16:31], v69, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:43], v[14:19], a[0:15], v44, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read2st64_b64 v[4:7], v2 offset1:9
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27
v_add_u32_e32 v26, 64, v56
v_and_b32_e32 v44, 0xff, v49
v_bfe_u32 v48, v49, 8, 8
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v12, v4
v_mov_b32_e32 v13, v5
ds_read_b128 v[8:11], v52 offset:48
ds_read_b128 v[2:5], v52 offset:4656
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v24, v16
v_mov_b32_e32 v25, v17
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81
ds_read_b128 v[20:23], v52 offset:9264
ds_read_b128 v[14:17], v52 offset:13872
ds_read2st64_b64 v[40:43], v26 offset0:90 offset1:99
v_and_b32_e32 v54, 0xff, v45
v_bfe_u32 v57, v45, 8, 8
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v36, v28
v_mov_b32_e32 v37, v29
ds_read_b128 v[32:35], v56 offset:36912
ds_read_b128 v[26:29], v56 offset:41520
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v66, v40
v_mov_b32_e32 v67, v41
ds_read_b128 v[62:65], v56 offset:46128
ds_read_b128 v[38:41], v56 offset:50736
v_bfe_u32 v58, v45, 16, 8
v_lshrrev_b32_e32 v45, 24, v45
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[8:13], a[240:255], v54, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v57, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[62:67], v[8:13], a[208:223], v58, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[38:43], v[8:13], a[192:207], v45, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[2:7], a[176:191], v54, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[2:7], a[160:175], v57, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[62:67], v[2:7], a[144:159], v58, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[38:43], v[2:7], a[128:143], v45, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_add_u32_e32 v2, 0x70, v52
v_bfe_u32 v53, v49, 16, 8
v_lshrrev_b32_e32 v49, 24, v49
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[20:25], a[112:127], v54, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[26:31], v[20:25], a[96:111], v57, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[62:67], v[20:25], a[80:95], v58, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[38:43], v[20:25], a[64:79], v45, v53 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[14:19], a[48:63], v54, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[14:19], a[32:47], v57, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[62:67], v[14:19], a[16:31], v58, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:43], v[14:19], a[0:15], v45, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read2st64_b64 v[4:7], v2 offset1:9
ds_read2st64_b64 v[16:19], v2 offset0:18 offset1:27
v_add_u32_e32 v26, 0x70, v56
v_and_b32_e32 v44, 0xff, v50
v_bfe_u32 v45, v50, 8, 8
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v12, v4
v_mov_b32_e32 v13, v5
ds_read_b128 v[8:11], v52 offset:96
ds_read_b128 v[2:5], v52 offset:4704
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v24, v16
v_mov_b32_e32 v25, v17
ds_read2st64_b64 v[28:31], v26 offset0:72 offset1:81
ds_read_b128 v[20:23], v52 offset:9312
ds_read_b128 v[14:17], v52 offset:13920
ds_read2st64_b64 v[40:43], v26 offset0:90 offset1:99
v_bfe_u32 v48, v50, 16, 8
v_lshrrev_b32_e32 v49, 24, v50
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v36, v28
v_mov_b32_e32 v37, v29
ds_read_b128 v[32:35], v56 offset:36960
ds_read_b128 v[26:29], v56 offset:41568
s_waitcnt lgkmcnt(2)
v_mov_b32_e32 v66, v40
v_mov_b32_e32 v67, v41
ds_read_b128 v[62:65], v56 offset:46176
ds_read_b128 v[38:41], v56 offset:50784
v_and_b32_e32 v50, 0xff, v46
v_bfe_u32 v52, v46, 8, 8
v_bfe_u32 v53, v46, 16, 8
v_lshrrev_b32_e32 v46, 24, v46
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[32:37], v[8:13], a[240:255], v50, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[26:31], v[8:13], a[224:239], v52, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[62:67], v[8:13], a[208:223], v53, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[38:43], v[8:13], a[192:207], v46, v44 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[32:37], v[2:7], a[176:191], v50, v45 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[26:31], v[2:7], a[160:175], v52, v45 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[62:67], v[2:7], a[144:159], v53, v45 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[38:43], v[2:7], a[128:143], v46, v45 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[32:37], v[20:25], a[112:127], v50, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[26:31], v[20:25], a[96:111], v52, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[62:67], v[20:25], a[80:95], v53, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[38:43], v[20:25], a[64:79], v46, v48 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[32:37], v[14:19], a[48:63], v50, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[26:31], v[14:19], a[32:47], v52, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[62:67], v[14:19], a[16:31], v53, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:43], v[14:19], a[0:15], v46, v49 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_mul_i32 s10, s21, 3
s_cmp_lt_i32 s10, s13
s_cbranch_scc1 4
v_and_b32_e32 v2, 31, v0
s_cbranch_execz 2
v_mov_b32_e32 v118, v2
s_branch 361
s_mov_b32 s11, 0x55555556
v_mul_hi_u32 v14, v0, s11
v_mul_hi_u32 v16, v119, s11
v_mul_hi_u32 v18, v120, s11
v_mul_u32_u24_e32 v2, 3, v14
v_mul_u32_u24_e32 v6, 3, v16
v_mul_u32_u24_e32 v10, 3, v18
v_sub_u32_e32 v4, v0, v2
v_mad_i64_i32 v[2:3], s[8:9], s14, v14, 0
v_sub_u32_e32 v8, v119, v6
v_mad_i64_i32 v[6:7], s[8:9], s14, v16, 0
v_sub_u32_e32 v12, v120, v10
v_mad_i64_i32 v[10:11], s[8:9], s14, v18, 0
v_mad_i64_i32 v[14:15], s[8:9], s15, v14, 0
v_mad_i64_i32 v[16:17], s[8:9], s15, v16, 0
v_mad_i64_i32 v[18:19], s[8:9], s15, v18, 0
s_lshl_b32 s8, s22, 3
v_lshl_or_b32 v26, v1, 7, v118
v_lshl_or_b32 v25, v61, 2, s8
v_mul_u32_u24_e32 v35, 48, v26
v_and_b32_e32 v26, 0x9f, v0
v_lshlrev_b32_e32 v30, 2, v1
v_mul_u32_u24_e32 v34, 48, v26
v_mul_lo_u32 v26, v25, s13
v_mov_b32_e32 v5, 0
v_or_b32_e32 v25, 0x60, v0
v_add_u32_e32 v27, s13, v26
v_lshl_or_b32 v32, s20, 3, v30
v_mad_u32_u24 v24, v121, 24, 0
v_mov_b32_e32 v61, v5
v_mul_u32_u24_e32 v25, 48, v25
v_add_u32_e32 v28, s13, v27
v_or_b32_e32 v30, 3, v32
v_or_b32_e32 v31, 2, v32
v_mul_lo_u32 v32, s13, v32
v_lshlrev_b32_e32 v4, 4, v4
v_lshlrev_b32_e32 v8, 4, v8
v_mov_b32_e32 v9, v5
v_lshlrev_b32_e32 v12, 4, v12
v_mov_b32_e32 v13, v5
s_waitcnt lgkmcnt(0)
v_lshl_add_u64 v[20:21], s[0:1], 0, v[60:61]
v_lshl_add_u64 v[22:23], s[2:3], 0, v[60:61]
v_add_u32_e32 v29, s13, v28
v_mul_lo_u32 v30, s13, v30
v_mul_lo_u32 v31, s13, v31
v_add_u32_e32 v33, s13, v32
s_mul_i32 s0, s21, 0x90
v_add_u32_e32 v34, v24, v34
v_add_u32_e32 v35, v24, v35
v_add_u32_e32 v36, v24, v25
s_ashr_i32 s1, s0, 31
v_readfirstlane_b32 s3, v47
v_add_u32_e32 v24, s10, v26
v_add_u32_e32 v38, s10, v32
s_add_u32 s2, s4, s0
s_mov_b32 m0, s3
v_ashrrev_i32_e32 v25, 31, v24
v_ashrrev_i32_e32 v39, 31, v38
s_addc_u32 s3, s5, s1
v_lshlrev_b64 v[24:25], 6, v[24:25]
v_lshlrev_b64 v[38:39], 6, v[38:39]
v_lshl_add_u64 v[40:41], s[2:3], 0, v[2:3]
v_lshl_add_u64 v[42:43], s[2:3], 0, v[6:7]
v_lshl_add_u64 v[44:45], s[2:3], 0, v[10:11]
s_add_u32 s2, s6, s0
v_lshl_add_u64 v[48:49], v[20:21], 0, v[24:25]
v_lshl_add_u64 v[24:25], v[22:23], 0, v[38:39]
v_lshl_add_u64 v[38:39], v[40:41], 0, v[4:5]
v_lshl_add_u64 v[40:41], v[42:43], 0, v[8:9]
v_lshl_add_u64 v[42:43], v[44:45], 0, v[12:13]
s_addc_u32 s3, s7, s1
v_readfirstlane_b32 s11, v51
v_readfirstlane_b32 s14, v55
global_load_lds_dwordx4 v[38:39], off
s_mov_b32 m0, s11
v_lshl_add_u64 v[38:39], s[2:3], 0, v[14:15]
global_load_lds_dwordx4 v[40:41], off
s_mov_b32 m0, s14
v_lshl_add_u64 v[40:41], s[2:3], 0, v[16:17]
global_load_lds_dwordx4 v[42:43], off
v_lshl_add_u64 v[42:43], s[2:3], 0, v[18:19]
v_readfirstlane_b32 s15, v59
v_readfirstlane_b32 s20, v116
v_readfirstlane_b32 s21, v117
s_mov_b32 m0, s15
v_lshl_add_u64 v[38:39], v[38:39], 0, v[4:5]
v_lshl_add_u64 v[40:41], v[40:41], 0, v[8:9]
v_lshl_add_u64 v[42:43], v[42:43], 0, v[12:13]
global_load_lds_dwordx4 v[38:39], off
s_mov_b32 m0, s20
v_add_u32_e32 v38, s10, v33
global_load_lds_dwordx4 v[40:41], off
s_mov_b32 m0, s21
v_ashrrev_i32_e32 v39, 31, v38
global_load_lds_dwordx4 v[42:43], off
s_waitcnt vmcnt(0)
s_barrier
global_load_ubyte v37, v[48:49], off
global_load_ubyte v46, v[24:25], off
v_lshlrev_b64 v[38:39], 6, v[38:39]
v_lshl_add_u64 v[44:45], v[22:23], 0, v[38:39]
ds_read_b128 v[38:41], v34
ds_read_b64 v[42:43], v34 offset:16
ds_read_b128 v[60:63], v35 offset:12288
ds_read_b64 v[64:65], v35 offset:12304
s_add_i32 s0, s0, 48
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[240:255], v[60:65], v[38:43], a[240:255], v46, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
global_load_ubyte v48, v[44:45], off
ds_read_b128 v[60:63], v35 offset:13824
ds_read_b64 v[64:65], v35 offset:13840
v_add_u32_e32 v44, s10, v31
v_ashrrev_i32_e32 v45, 31, v44
v_lshlrev_b64 v[44:45], 6, v[44:45]
v_lshl_add_u64 v[44:45], v[22:23], 0, v[44:45]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[224:239], v[60:65], v[38:43], a[224:239], v48, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
global_load_ubyte v49, v[44:45], off
ds_read_b128 v[60:63], v35 offset:15360
ds_read_b64 v[64:65], v35 offset:15376
v_add_u32_e32 v44, s10, v30
v_ashrrev_i32_e32 v45, 31, v44
v_lshlrev_b64 v[44:45], 6, v[44:45]
v_lshl_add_u64 v[44:45], v[22:23], 0, v[44:45]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[208:223], v[60:65], v[38:43], a[208:223], v49, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
global_load_ubyte v50, v[44:45], off
ds_read_b128 v[60:63], v35 offset:16896
ds_read_b64 v[64:65], v35 offset:16912
v_add_u32_e32 v44, s10, v27
v_ashrrev_i32_e32 v45, 31, v44
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[192:207], v[60:65], v[38:43], a[192:207], v50, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_add_u32_e32 v38, s10, v28
v_lshlrev_b64 v[44:45], 6, v[44:45]
v_ashrrev_i32_e32 v39, 31, v38
v_lshl_add_u64 v[44:45], v[20:21], 0, v[44:45]
v_lshlrev_b64 v[38:39], 6, v[38:39]
global_load_ubyte v37, v[44:45], off
v_lshl_add_u64 v[44:45], v[20:21], 0, v[38:39]
ds_read_b128 v[38:41], v34 offset:1536
ds_read_b64 v[42:43], v34 offset:1552
ds_read_b128 v[60:63], v35 offset:12288
ds_read_b64 v[64:65], v35 offset:12304
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[176:191], v[60:65], v[38:43], a[176:191], v46, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read_b128 v[60:63], v35 offset:13824
ds_read_b64 v[64:65], v35 offset:13840
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[160:175], v[60:65], v[38:43], a[160:175], v48, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read_b128 v[60:63], v35 offset:15360
ds_read_b64 v[64:65], v35 offset:15376
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[144:159], v[60:65], v[38:43], a[144:159], v49, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read_b128 v[60:63], v35 offset:16896
ds_read_b64 v[64:65], v35 offset:16912
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[128:143], v[60:65], v[38:43], a[128:143], v50, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
v_add_u32_e32 v38, s10, v29
v_ashrrev_i32_e32 v39, 31, v38
v_lshlrev_b64 v[38:39], 6, v[38:39]
global_load_ubyte v37, v[44:45], off
v_lshl_add_u64 v[44:45], v[20:21], 0, v[38:39]
ds_read_b128 v[38:41], v34 offset:3072
ds_read_b64 v[42:43], v34 offset:3088
ds_read_b128 v[60:63], v35 offset:12288
ds_read_b64 v[64:65], v35 offset:12304
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[112:127], v[60:65], v[38:43], a[112:127], v46, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read_b128 v[60:63], v35 offset:13824
ds_read_b64 v[64:65], v35 offset:13840
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[96:111], v[60:65], v[38:43], a[96:111], v48, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read_b128 v[60:63], v35 offset:15360
ds_read_b64 v[64:65], v35 offset:15376
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[80:95], v[60:65], v[38:43], a[80:95], v49, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read_b128 v[60:63], v35 offset:16896
ds_read_b64 v[64:65], v35 offset:16912
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[64:79], v[60:65], v[38:43], a[64:79], v50, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
global_load_ubyte v37, v[44:45], off
s_nop 0
global_load_ubyte v24, v[24:25], off
ds_read_b128 v[38:41], v36
ds_read_b64 v[42:43], v36 offset:16
ds_read_b128 v[60:63], v35 offset:12288
ds_read_b64 v[64:65], v35 offset:12304
s_add_i32 s10, s10, 1
s_cmp_ge_i32 s10, s13
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[60:65], v[38:43], a[48:63], v24, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read_b128 v[60:63], v35 offset:13824
ds_read_b64 v[64:65], v35 offset:13840
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[60:65], v[38:43], a[32:47], v48, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read_b128 v[60:63], v35 offset:15360
ds_read_b64 v[64:65], v35 offset:15376
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[60:65], v[38:43], a[16:31], v49, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
ds_read_b128 v[60:63], v35 offset:16896
ds_read_b64 v[64:65], v35 offset:16912
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[60:65], v[38:43], a[0:15], v50, v37 op_sel_hi:[0,0,0] cbsz:2 blgp:2
s_barrier
s_cbranch_scc0 65249
v_and_b32_e32 v2, 0x80, v0
v_add_u32_e32 v3, s18, v2
v_lshl_or_b32 v2, v1, 7, s19
v_or_b32_e32 v6, v3, v118
s_waitcnt lgkmcnt(0)
v_mad_i64_i32 v[4:5], s[0:1], v6, s12, 0
v_ashrrev_i32_e32 v3, 31, v2
v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
v_lshlrev_b64 v[2:3], 2, v[2:3]
v_lshrrev_b32_e32 v0, 1, v0
v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
v_and_b32_e32 v0, 16, v0
v_mov_b32_e32 v1, 0
v_lshl_add_u64 v[4:5], v[4:5], 0, v[0:1]
global_store_dwordx4 v[4:5], a[240:243], off
global_store_dwordx4 v[4:5], a[244:247], off offset:32
global_store_dwordx4 v[4:5], a[248:251], off offset:64
global_store_dwordx4 v[4:5], a[252:255], off offset:96
global_store_dwordx4 v[4:5], a[224:227], off offset:128
global_store_dwordx4 v[4:5], a[228:231], off offset:160
global_store_dwordx4 v[4:5], a[232:235], off offset:192
global_store_dwordx4 v[4:5], a[236:239], off offset:224
global_store_dwordx4 v[4:5], a[208:211], off offset:256
global_store_dwordx4 v[4:5], a[212:215], off offset:288
global_store_dwordx4 v[4:5], a[216:219], off offset:320
global_store_dwordx4 v[4:5], a[220:223], off offset:352
global_store_dwordx4 v[4:5], a[192:195], off offset:384
global_store_dwordx4 v[4:5], a[196:199], off offset:416
global_store_dwordx4 v[4:5], a[200:203], off offset:448
global_store_dwordx4 v[4:5], a[204:207], off offset:480
v_or_b32_e32 v4, 32, v6
v_mad_i64_i32 v[4:5], s[0:1], v4, s12, 0
v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
v_lshl_add_u64 v[4:5], v[4:5], 0, v[0:1]
global_store_dwordx4 v[4:5], a[176:179], off
global_store_dwordx4 v[4:5], a[180:183], off offset:32
global_store_dwordx4 v[4:5], a[184:187], off offset:64
global_store_dwordx4 v[4:5], a[188:191], off offset:96
global_store_dwordx4 v[4:5], a[160:163], off offset:128
global_store_dwordx4 v[4:5], a[164:167], off offset:160
global_store_dwordx4 v[4:5], a[168:171], off offset:192
global_store_dwordx4 v[4:5], a[172:175], off offset:224
global_store_dwordx4 v[4:5], a[144:147], off offset:256
global_store_dwordx4 v[4:5], a[148:151], off offset:288
global_store_dwordx4 v[4:5], a[152:155], off offset:320
global_store_dwordx4 v[4:5], a[156:159], off offset:352
global_store_dwordx4 v[4:5], a[128:131], off offset:384
global_store_dwordx4 v[4:5], a[132:135], off offset:416
global_store_dwordx4 v[4:5], a[136:139], off offset:448
global_store_dwordx4 v[4:5], a[140:143], off offset:480
v_or_b32_e32 v4, 64, v6
v_mad_i64_i32 v[4:5], s[0:1], v4, s12, 0
v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
v_lshl_add_u64 v[4:5], v[4:5], 0, v[0:1]
global_store_dwordx4 v[4:5], a[112:115], off
global_store_dwordx4 v[4:5], a[116:119], off offset:32
global_store_dwordx4 v[4:5], a[120:123], off offset:64
global_store_dwordx4 v[4:5], a[124:127], off offset:96
global_store_dwordx4 v[4:5], a[96:99], off offset:128
global_store_dwordx4 v[4:5], a[100:103], off offset:160
global_store_dwordx4 v[4:5], a[104:107], off offset:192
global_store_dwordx4 v[4:5], a[108:111], off offset:224
global_store_dwordx4 v[4:5], a[80:83], off offset:256
global_store_dwordx4 v[4:5], a[84:87], off offset:288
global_store_dwordx4 v[4:5], a[88:91], off offset:320
global_store_dwordx4 v[4:5], a[92:95], off offset:352
global_store_dwordx4 v[4:5], a[64:67], off offset:384
global_store_dwordx4 v[4:5], a[68:71], off offset:416
global_store_dwordx4 v[4:5], a[72:75], off offset:448
global_store_dwordx4 v[4:5], a[76:79], off offset:480
v_or_b32_e32 v4, 0x60, v6
v_mad_i64_i32 v[4:5], s[0:1], v4, s12, 0
v_lshl_add_u64 v[4:5], v[4:5], 2, s[16:17]
v_lshl_add_u64 v[2:3], v[4:5], 0, v[2:3]
v_lshl_add_u64 v[0:1], v[2:3], 0, v[0:1]
global_store_dwordx4 v[0:1], a[48:51], off
global_store_dwordx4 v[0:1], a[52:55], off offset:32
global_store_dwordx4 v[0:1], a[56:59], off offset:64
global_store_dwordx4 v[0:1], a[60:63], off offset:96
global_store_dwordx4 v[0:1], a[32:35], off offset:128
global_store_dwordx4 v[0:1], a[36:39], off offset:160
global_store_dwordx4 v[0:1], a[40:43], off offset:192
global_store_dwordx4 v[0:1], a[44:47], off offset:224
global_store_dwordx4 v[0:1], a[16:19], off offset:256
global_store_dwordx4 v[0:1], a[20:23], off offset:288
global_store_dwordx4 v[0:1], a[24:27], off offset:320
global_store_dwordx4 v[0:1], a[28:31], off offset:352
global_store_dwordx4 v[0:1], a[0:3], off offset:384
global_store_dwordx4 v[0:1], a[4:7], off offset:416
global_store_dwordx4 v[0:1], a[8:11], off offset:448
global_store_dwordx4 v[0:1], a[12:15], off offset:480
s_endpgm