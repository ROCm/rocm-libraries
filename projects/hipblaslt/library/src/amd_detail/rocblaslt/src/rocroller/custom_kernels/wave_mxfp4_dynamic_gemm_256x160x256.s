; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_dynamic_gemm_256x160x256
	.p2align	8
	.type	wave_mxfp4_dynamic_gemm_256x160x256,@function
wave_mxfp4_dynamic_gemm_256x160x256:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	s_load_dwordx2 s[84:85], s[0:1], 0x38
	s_load_dwordx2 s[86:87], s[0:1], 0x48
	s_load_dwordx2 s[88:89], s[0:1], 0x58
	s_mov_b64 s[64:65], s[6:7]
	s_mov_b64 s[82:83], s[10:11]
	s_waitcnt lgkmcnt(0)
	s_sub_u32 s6, 0, s84
	s_subb_u32 s7, 0, s85
	s_add_u32 s10, s84, -1
	s_mov_b64 s[22:23], s[0:1]
	v_cmp_lt_i64_e64 s[0:1], s[84:85], 1
	s_addc_u32 s11, s85, -1
	s_mov_b64 s[68:69], s[2:3]
	s_and_b64 s[2:3], s[0:1], exec
	s_cselect_b32 s3, s7, s11
	s_cselect_b32 s2, s6, s10
	s_ashr_i32 s6, s3, 31
	s_lshr_b32 s6, s6, 24
	s_add_u32 s2, s2, s6
	s_addc_u32 s3, s3, 0
	s_ashr_i64 s[2:3], s[2:3], 8
	s_sub_u32 s6, 0, s2
	s_subb_u32 s7, 0, s3
	v_and_b32_e32 v174, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_add_u32 s2, s2, 1
	s_addc_u32 s3, s3, 0
	v_lshrrev_b32_e32 v172, 6, v174
	v_lshlrev_b32_e32 v7, 3, v0
	v_readfirstlane_b32 s94, v0
	s_and_b64 s[0:1], s[0:1], exec
	v_lshlrev_b32_e32 v12, 5, v0
	v_lshl_or_b32 v0, v172, 1, v7
	s_mov_b32 s18, s17
	s_cselect_b32 s96, s6, s2
	v_lshl_or_b32 v1, v172, 3, v12
	v_readfirstlane_b32 s0, v0
	v_bfe_u32 v147, v174, 4, 2
	s_mov_b64 s[50:51], s[4:5]
	s_mov_b32 s17, 0
	s_mov_b64 s[4:5], -1
	s_cselect_b32 s97, s7, s3
	v_readfirstlane_b32 s20, v1
	v_writelane_b32 v254, s0, 0
	s_cmp_gt_i32 s96, 1
	v_lshrrev_b32_e32 v176, 3, v174
	v_lshrrev_b32_e32 v173, 5, v174
	s_mul_hi_u32 s63, s18, 0xa0
	s_mul_i32 s62, s18, 0xa0
	v_lshrrev_b32_e32 v177, 4, v174
	v_and_b32_e32 v13, 15, v174
	v_lshlrev_b32_e32 v151, 6, v147
	v_lshlrev_b32_e32 v175, 2, v174
	v_lshlrev_b32_e32 v3, 1, v174
	v_lshlrev_b32_e32 v112, 5, v147
	s_mul_i32 s95, s94, 0x50
	s_mul_i32 s18, s94, 5
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s0, s97, 31
	s_mov_b32 s1, s0
	s_xor_b64 s[2:3], s[0:1], s[96:97]
	s_lshr_b32 s4, s3, 31
	s_add_u32 s2, s2, s4
	s_addc_u32 s3, s3, 0
	s_lshr_b64 s[2:3], s[2:3], 1
	s_xor_b64 s[0:1], s[2:3], s[0:1]
	s_lshl_b64 s[26:27], s[0:1], 1
	s_ashr_i32 s0, s85, 31
	s_mov_b32 s1, s0
	s_xor_b64 s[2:3], s[84:85], s[0:1]
	s_lshr_b32 s4, s3, 31
	s_add_u32 s2, s2, s4
	s_addc_u32 s3, s3, 0
	s_ashr_i64 s[2:3], s[2:3], 1
	v_or_b32_e32 v0, v176, v12
	s_lshl_b64 s[10:11], s[16:17], 8
	s_xor_b64 s[90:91], s[2:3], s[0:1]
	v_or_b32_e32 v122, s10, v0
	v_mad_u64_u32 v[240:241], s[0:1], s90, v122, 0
	s_and_b32 s0, s90, 0x3fff
	v_or_b32_e32 v124, 64, v122
	v_or_b32_e32 v126, 0x80, v122
	v_or_b32_e32 v128, 0xc0, v122
	s_bitset1_b32 s0, 14
	v_mad_u64_u32 v[242:243], s[2:3], s90, v124, 0
	v_mad_u64_u32 v[244:245], s[2:3], s90, v126, 0
	v_mad_u64_u32 v[120:121], s[2:3], s90, v128, 0
	s_lshl_b32 s4, s0, 16
	s_and_b32 s0, s69, 0xffff
	s_and_b32 s2, s65, 0xffff
	v_or_b32_e32 v0, v173, v7
	s_or_b32 s1, s0, s4
	s_or_b32 s3, s2, s4
	v_or_b32_e32 v130, s62, v0
	v_mov_b32_e32 v131, s63
	s_mov_b64 s[4:5], 0x50
	v_lshl_add_u64 v[84:85], v[130:131], 0, s[4:5]
	s_mov_b64 s[4:5], 0x60
	v_lshl_add_u64 v[252:253], v[130:131], 0, s[4:5]
	s_mov_b64 s[4:5], 0x70
	v_and_b32_e32 v0, 0xcf, v174
	v_lshl_add_u64 v[86:87], v[130:131], 0, s[4:5]
	s_mov_b64 s[4:5], 0x80
	v_or_b32_e32 v154, s10, v0
	v_lshlrev_b32_e32 v0, 5, v177
	v_lshl_add_u64 v[148:149], v[130:131], 0, s[4:5]
	s_mov_b64 s[4:5], 0x90
	v_sub_u32_e32 v0, v3, v0
	v_lshl_add_u64 v[152:153], v[130:131], 0, s[4:5]
	s_lshr_b32 s5, s16, 24
	v_add_u32_e32 v0, v0, v112
	s_mul_i32 s5, s86, s5
	s_mul_hi_u32 s6, s86, s10
	v_ashrrev_i32_e32 v1, 31, v0
	s_add_i32 s5, s6, s5
	s_mul_i32 s6, s87, s10
	v_xor_b32_e32 v0, v1, v0
	s_add_i32 s53, s5, s6
	s_and_b32 s5, s86, 0x3fff
	v_ashrrev_i32_e32 v2, 31, v0
	s_or_b32 s33, s5, 0x4000
	v_lshrrev_b32_e32 v2, 25, v2
	v_add_u32_e32 v0, v0, v2
	s_add_u32 s5, s62, s95
	s_movk_i32 s4, 0xffc0
	v_ashrrev_i32_e32 v0, 7, v0
	s_addc_u32 s6, s63, 0
	v_mad_i32_i24 v183, v177, s4, v175
	v_xor_b32_e32 v185, v0, v1
	v_or_b32_e32 v162, s5, v13
	v_mov_b32_e32 v163, s6
	s_mul_i32 s5, s88, s63
	s_mul_hi_u32 s6, s88, s62
	v_add_u32_e32 v121, v183, v151
	v_lshlrev_b32_e32 v0, 8, v185
	s_add_i32 s5, s6, s5
	s_mul_i32 s6, s89, s62
	v_mov_b32_e32 v123, s11
	s_mov_b32 s71, 0x27000
	s_mov_b32 s70, 0x7ffffffe
	v_mov_b32_e32 v125, s11
	v_mov_b32_e32 v127, s11
	v_mov_b32_e32 v129, s11
	v_mov_b32_e32 v155, s11
	s_mul_i32 s52, s86, s10
	v_mov_b32_e32 v157, s11
	v_sub_u32_e32 v0, v121, v0
	v_mov_b32_e32 v159, s11
	v_writelane_b32 v254, s10, 1
	v_or_b32_e32 v115, s95, v13
	s_add_i32 s61, s5, s6
	s_and_b32 s5, s88, 0x3fff
	v_add_u32_e32 v188, s18, v175
	v_mad_i32_i24 v189, v177, s4, v151
	s_mov_b32 s0, s68
	s_mov_b32 s2, s64
	v_or_b32_e32 v132, 16, v130
	v_mov_b32_e32 v133, s63
	v_lshl_add_u64 v[248:249], v[130:131], 0, 32
	v_lshl_add_u64 v[250:251], v[130:131], 0, 48
	v_lshl_add_u64 v[82:83], v[130:131], 0, 64
	v_and_b32_e32 v117, 0xc0, v174
	v_or_b32_e32 v184, 2, v121
	v_or_b32_e32 v156, 16, v154
	v_or_b32_e32 v146, 1, v0
	v_or_b32_e32 v150, 3, v121
	v_xor_b32_e32 v186, -4, v121
	v_ashrrev_i32_e32 v187, 31, v121
	v_or_b32_e32 v158, 32, v154
	v_or_b32_e32 v160, 48, v154
	v_writelane_b32 v254, s11, 2
	v_mov_b32_e32 v161, s11
	s_mul_i32 s60, s88, s62
	s_or_b32 s54, s5, 0x4000
	v_lshl_add_u64 v[164:165], v[162:163], 0, 16
	v_add_u32_e32 v119, 16, v115
	v_add_u32_e32 v190, v189, v188
	v_lshl_add_u64 v[166:167], v[162:163], 0, 32
	v_lshl_add_u64 v[168:169], v[162:163], 0, 48
	v_lshl_add_u64 v[170:171], v[162:163], 0, 64
	v_mov_b32_e32 v92, s18
	s_mov_b64 s[66:67], s[70:71]
	s_mov_b64 s[4:5], 0
	s_branch .LBB0_3
.LBB0_2:
.LBB0_3:
	s_or_b32 s6, s20, 0x80
	v_writelane_b32 v254, s6, 3
	s_or_b32 s6, s20, 0xc0
	v_writelane_b32 v254, s6, 4
	s_andn2_b64 vcc, exec, s[4:5]
	v_readlane_b32 s6, v254, 0
	s_mul_i32 s4, s94, 0x2800
	s_or_b32 s42, s6, 0x90
	v_bfe_u32 v182, v174, 2, 3
	v_writelane_b32 v254, s4, 5
	s_or_b32 s81, s20, 64
	s_or_b32 s55, s6, 16
	s_or_b32 s74, s6, 32
	s_or_b32 s75, s6, 48
	s_or_b32 s76, s6, 64
	s_or_b32 s77, s6, 0x50
	s_or_b32 s78, s6, 0x60
	s_or_b32 s79, s6, 0x70
	s_or_b32 s80, s6, 0x80
	s_mov_b32 s24, 0
	v_lshrrev_b32_e32 v116, 2, v174
	v_and_b32_e32 v118, 31, v174
	v_lshlrev_b32_e32 v246, 2, v182
	v_lshlrev_b32_e32 v114, 3, v174
	v_writelane_b32 v254, s42, 6
	s_cbranch_vccnz .LBB0_10
	s_ashr_i32 s0, s85, 31
	s_mov_b32 s1, s0
	s_xor_b64 s[2:3], s[84:85], s[0:1]
	s_lshr_b32 s4, s3, 31
	s_add_u32 s2, s2, s4
	s_addc_u32 s3, s3, 0
	s_ashr_i64 s[2:3], s[2:3], 1
	v_or_b32_e32 v0, v176, v12
	s_lshl_b64 s[52:53], s[16:17], 8
	s_xor_b64 s[90:91], s[2:3], s[0:1]
	v_or_b32_e32 v122, s52, v0
	v_mad_u64_u32 v[240:241], s[0:1], s90, v122, 0
	s_and_b32 s0, s90, 0x3fff
	s_bitset1_b32 s0, 14
	v_writelane_b32 v254, s22, 7
	v_mov_b32_e32 v123, s53
	s_lshl_b32 s2, s0, 16
	s_and_b32 s0, s69, 0xffff
	v_writelane_b32 v254, s23, 8
	v_bitop3_b32 v0, v176, 7, v174 bitop3:0x48
	s_or_b32 s69, s0, s2
	v_cmp_gt_i64_e64 s[0:1], s[12:13], v[122:123]
	v_lshlrev_b32_e32 v32, 5, v0
	v_mov_b32_e32 v33, 0
	v_writelane_b32 v254, s0, 9
	v_cmp_gt_i64_e32 vcc, s[84:85], v[32:33]
	v_or_b32_e32 v124, 64, v122
	v_writelane_b32 v254, s1, 10
	s_and_b64 s[6:7], s[0:1], vcc
	v_writelane_b32 v254, s20, 11
	v_mov_b32_e32 v125, s53
	v_mad_u64_u32 v[242:243], s[0:1], s90, v124, 0
	v_lshlrev_b32_e32 v35, 4, v0
	v_cmp_gt_i64_e64 s[0:1], s[12:13], v[124:125]
	v_writelane_b32 v254, s18, 12
	v_add_u32_e32 v0, v35, v240
	v_bfrev_b32_e32 v1, -2
	s_lshl_b32 s3, s20, 7
	v_writelane_b32 v254, s0, 13
	v_or_b32_e32 v126, 0x80, v122
	s_mov_b32 s71, 0x27000
	s_mov_b32 s70, 0x7ffffffe
	v_cndmask_b32_e64 v0, v1, v0, s[6:7]
	s_mov_b32 m0, s3
	v_writelane_b32 v254, s1, 14
	s_and_b64 s[18:19], s[0:1], vcc
	v_mov_b32_e32 v127, s53
	v_mad_u64_u32 v[244:245], s[0:1], s90, v126, 0
	buffer_load_dwordx4 v0, s[68:71], 0 offen lds
	v_add_u32_e32 v0, v35, v242
	v_cmp_gt_i64_e64 s[0:1], s[12:13], v[126:127]
	v_cndmask_b32_e64 v0, v1, v0, s[18:19]
	s_add_i32 m0, s3, 0x2000
	v_writelane_b32 v254, s0, 15
	v_or_b32_e32 v128, 0xc0, v122
	buffer_load_dwordx4 v0, s[68:71], 0 offen lds
	v_writelane_b32 v254, s1, 16
	s_and_b64 s[18:19], s[0:1], vcc
	v_add_u32_e32 v0, v35, v244
	v_mov_b32_e32 v129, s53
	v_mad_u64_u32 v[120:121], s[0:1], s90, v128, 0
	v_cndmask_b32_e64 v0, v1, v0, s[18:19]
	s_add_i32 m0, s3, 0x4000
	v_cmp_gt_i64_e64 s[0:1], s[12:13], v[128:129]
	buffer_load_dwordx4 v0, s[68:71], 0 offen lds
	s_add_i32 m0, s3, 0x6000
	v_writelane_b32 v254, s0, 17
	s_and_b64 vcc, s[0:1], vcc
	s_ashr_i32 s98, s91, 31
	v_writelane_b32 v254, s1, 18
	s_add_u32 s0, s90, s98
	s_mov_b32 s99, s98
	s_addc_u32 s1, s91, s98
	s_xor_b64 s[0:1], s[0:1], s[98:99]
	v_cvt_f32_u32_e32 v16, s0
	v_cvt_f32_u32_e32 v17, s1
	v_writelane_b32 v254, s3, 19
	s_sub_u32 s3, 0, s0
	s_subb_u32 s4, 0, s1
	v_fmamk_f32 v16, v17, 0x4f800000, v16
	v_rcp_f32_e32 v16, v16
	v_bitop3_b32 v9, v116, v173, 7 bitop3:0x6c
	v_sub_u32_e32 v4, v118, v246
	v_lshl_add_u32 v4, v9, 2, v4
	v_mul_f32_e32 v16, 0x5f7ffffc, v16
	v_mul_f32_e32 v17, 0x2f800000, v16
	v_trunc_f32_e32 v17, v17
	v_fmamk_f32 v16, v17, 0xcf800000, v16
	v_cvt_u32_f32_e32 v16, v16
	v_ashrrev_i32_e32 v6, 31, v4
	v_xor_b32_e32 v4, v6, v4
	v_ashrrev_i32_e32 v8, 31, v4
	v_readfirstlane_b32 s5, v16
	v_cvt_u32_f32_e32 v16, v17
	s_mul_hi_u32 s7, s3, s5
	s_mul_i32 s6, s4, s5
	s_mul_i32 s18, s3, s5
	v_readfirstlane_b32 s10, v16
	s_mul_i32 s11, s3, s10
	s_add_i32 s7, s7, s11
	s_add_i32 s7, s7, s6
	s_mul_i32 s11, s5, s7
	s_mul_hi_u32 s17, s5, s18
	s_mul_hi_u32 s6, s5, s7
	s_add_u32 s11, s17, s11
	s_addc_u32 s6, 0, s6
	s_mul_hi_u32 s17, s10, s18
	s_mul_i32 s18, s10, s18
	s_add_u32 s11, s11, s18
	s_mul_hi_u32 s19, s10, s7
	s_addc_u32 s6, s6, s17
	s_addc_u32 s11, s19, 0
	s_mul_i32 s7, s10, s7
	s_add_u32 s6, s6, s7
	s_addc_u32 s7, 0, s11
	s_add_u32 s5, s5, s6
	s_addc_u32 s6, s10, s7
	v_lshrrev_b32_e32 v8, 29, v8
	s_mul_i32 s7, s3, s6
	s_mul_hi_u32 s10, s3, s5
	v_add_u32_e32 v4, v4, v8
	s_add_i32 s7, s10, s7
	s_mul_i32 s4, s4, s5
	v_lshrrev_b32_e32 v4, 3, v4
	s_add_i32 s7, s7, s4
	s_mul_i32 s3, s3, s5
	v_add_u32_e32 v0, v35, v120
	v_xor_b32_e32 v4, v4, v6
	v_lshrrev_b32_e32 v6, 1, v9
	s_mul_hi_u32 s10, s6, s3
	s_mul_i32 s11, s6, s3
	s_mul_i32 s17, s5, s7
	s_mul_hi_u32 s3, s5, s3
	v_cndmask_b32_e32 v0, v1, v0, vcc
	v_lshl_add_u32 v5, s94, 7, v175
	v_lshlrev_b32_e32 v2, 4, v173
	v_mul_i32_i24_e32 v6, 0xfffffe00, v6
	v_mul_i32_i24_e32 v8, -16, v116
	s_mul_hi_u32 s18, s5, s7
	s_add_u32 s3, s3, s17
	buffer_load_dwordx4 v0, s[68:71], 0 offen lds
	v_lshl_add_u32 v0, v9, 8, v5
	v_lshlrev_b32_e32 v4, 9, v4
	v_add3_u32 v10, v8, v2, v6
	s_addc_u32 s17, 0, s18
	v_add3_u32 v10, v10, v0, v4
	s_add_u32 s3, s3, s11
	v_not_b32_e32 v14, v10
	s_mul_hi_u32 s4, s6, s7
	s_addc_u32 s3, s17, s10
	v_ashrrev_i32_e32 v11, 31, v10
	v_cmp_gt_i32_e32 vcc, 0, v10
	v_ashrrev_i32_e32 v15, 31, v14
	s_addc_u32 s4, s4, 0
	s_mul_i32 s7, s6, s7
	v_cndmask_b32_e32 v15, v11, v15, vcc
	s_add_u32 s3, s3, s7
	v_cndmask_b32_e32 v14, v10, v14, vcc
	s_addc_u32 s4, 0, s4
	v_mov_b32_e32 v16, v15
	v_mov_b32_e32 v17, v15
	s_add_u32 s3, s5, s3
	v_lshl_add_u64 v[16:17], v[14:15], 0, v[16:17]
	s_addc_u32 s6, s6, s4
	v_xor_b32_e32 v22, v16, v15
	v_xor_b32_e32 v14, v17, v15
	v_mad_u64_u32 v[16:17], s[4:5], v22, s6, 0
	v_mul_hi_u32 v18, v22, s3
	v_mov_b32_e32 v19, v33
	v_lshl_add_u64 v[16:17], v[18:19], 0, v[16:17]
	v_mad_u64_u32 v[20:21], s[4:5], v14, s3, 0
	v_add_co_u32_e32 v16, vcc, v16, v20
	v_mad_u64_u32 v[18:19], s[4:5], v14, s6, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v21, vcc
	v_mov_b32_e32 v17, s24
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[18:19]
	v_mul_lo_u32 v20, s1, v16
	v_mul_lo_u32 v17, s0, v17
	v_mad_u64_u32 v[18:19], s[4:5], s0, v16, 0
	v_add3_u32 v17, v19, v17, v20
	v_sub_u32_e32 v19, v14, v17
	v_sub_co_u32_e32 v18, vcc, v22, v18
	v_mov_b32_e32 v20, s1
	s_nop 0
	v_subb_co_u32_e64 v19, s[20:21], v19, v20, vcc
	v_subrev_co_u32_e64 v21, s[20:21], s0, v18
	v_cmp_le_u32_e64 s[22:23], s0, v21
	s_nop 0
	v_subbrev_co_u32_e64 v19, s[20:21], 0, v19, s[20:21]
	v_cmp_le_u32_e64 s[20:21], s1, v19
	v_add_u32_e32 v21, 1, v16
	v_subb_co_u32_e32 v14, vcc, v14, v17, vcc
	v_cndmask_b32_e64 v22, 0, -1, s[20:21]
	v_cmp_eq_u32_e64 s[20:21], s1, v19
	v_cndmask_b32_e64 v19, 0, -1, s[22:23]
	v_cmp_le_u32_e32 vcc, s1, v14
	v_cndmask_b32_e64 v19, v22, v19, s[20:21]
	v_cmp_ne_u32_e64 s[20:21], 0, v19
	v_add_u32_e32 v19, 2, v16
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cndmask_b32_e64 v19, v21, v19, s[20:21]
	v_cmp_le_u32_e64 s[20:21], s0, v18
	v_cmp_eq_u32_e32 vcc, s1, v14
	v_xor_b32_e32 v15, s98, v15
	v_cndmask_b32_e64 v14, 0, -1, s[20:21]
	v_cndmask_b32_e32 v14, v17, v14, vcc
	v_cmp_ne_u32_e32 vcc, 0, v14
	v_mov_b32_e32 v17, v33
	v_or_b32_e32 v7, v173, v7
	v_cndmask_b32_e32 v14, v16, v19, vcc
	v_xor_b32_e32 v14, v14, v15
	v_sub_u32_e32 v14, v14, v15
	v_xad_u32 v21, v14, v11, s62
	v_mov_b32_e32 v14, v11
	v_mov_b32_e32 v15, v11
	v_lshl_add_u64 v[14:15], v[10:11], 0, v[14:15]
	v_xor_b32_e32 v22, v14, v11
	v_xor_b32_e32 v10, v15, v11
	v_mad_u64_u32 v[14:15], s[4:5], v22, s6, 0
	v_mul_hi_u32 v16, v22, s3
	v_lshl_add_u64 v[14:15], v[16:17], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[4:5], v10, s3, 0
	v_add_co_u32_e32 v14, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[4:5], v10, s6, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s24
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v16, s1, v14
	v_mul_lo_u32 v17, s0, v15
	v_mad_u64_u32 v[14:15], s[4:5], s0, v14, 0
	v_add3_u32 v15, v15, v17, v16
	v_sub_u32_e32 v16, v10, v15
	v_sub_co_u32_e32 v14, vcc, v22, v14
	v_or_b32_e32 v130, s62, v7
	s_nop 0
	v_subb_co_u32_e64 v16, s[20:21], v16, v20, vcc
	v_subrev_co_u32_e64 v17, s[20:21], s0, v14
	v_cmp_le_u32_e64 s[24:25], s0, v17
	s_nop 0
	v_subbrev_co_u32_e64 v18, s[22:23], 0, v16, s[20:21]
	v_cmp_le_u32_e64 s[22:23], s1, v18
	v_cndmask_b32_e64 v22, 0, -1, s[24:25]
	v_subb_co_u32_e64 v16, s[20:21], v16, v20, s[20:21]
	v_cndmask_b32_e64 v19, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s1, v18
	v_subb_co_u32_e32 v10, vcc, v10, v15, vcc
	s_nop 0
	v_cndmask_b32_e64 v19, v19, v22, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v19
	v_subrev_co_u32_e64 v19, s[20:21], s0, v17
	v_cmp_le_u32_e32 vcc, s1, v10
	s_nop 0
	v_subbrev_co_u32_e64 v16, s[20:21], 0, v16, s[20:21]
	v_cmp_le_u32_e64 s[20:21], s0, v14
	v_cndmask_b32_e64 v16, v18, v16, s[22:23]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s1, v10
	v_cndmask_b32_e64 v18, 0, -1, s[20:21]
	v_sub_u32_e32 v7, v9, v182
	v_cndmask_b32_e32 v15, v15, v18, vcc
	v_cmp_ne_u32_e32 vcc, 0, v15
	v_cndmask_b32_e64 v15, v17, v19, s[22:23]
	v_lshlrev_b32_e32 v9, 8, v173
	v_cndmask_b32_e32 v14, v14, v15, vcc
	v_cndmask_b32_e32 v10, v10, v16, vcc
	v_xor_b32_e32 v14, v14, v11
	v_xor_b32_e32 v10, v10, v11
	v_sub_co_u32_e32 v14, vcc, v14, v11
	v_sub_u32_e32 v9, v114, v9
	s_nop 0
	v_subb_co_u32_e32 v10, vcc, v10, v11, vcc
	v_ashrrev_i32_e32 v10, 31, v10
	v_and_b32_e32 v10, s90, v10
	v_add_u32_e32 v14, v10, v14
	s_and_b32 s0, s65, 0xffff
	v_mov_b32_e32 v131, s63
	v_lshl_add_u32 v10, v7, 5, v9
	s_or_b32 s65, s0, s2
	v_cmp_gt_i64_e64 s[0:1], s[14:15], v[130:131]
	v_ashrrev_i32_e32 v11, 31, v10
	v_readlane_b32 s7, v254, 0
	v_cmp_gt_i64_e32 vcc, s[84:85], v[10:11]
	v_writelane_b32 v254, s0, 20
	s_and_b64 s[22:23], s[0:1], vcc
	s_mov_b64 s[2:3], s[14:15]
	v_writelane_b32 v254, s1, 21
	s_lshl_b32 s0, s7, 7
	v_or_b32_e32 v132, 16, v130
	v_mov_b32_e32 v133, s63
	v_writelane_b32 v254, s0, 22
	s_add_i32 m0, s0, 0x18000
	v_cmp_gt_i64_e64 s[0:1], s[2:3], v[132:133]
	v_mul_lo_u32 v15, v21, s90
	v_add_u32_e32 v16, v14, v15
	v_writelane_b32 v254, s0, 23
	s_mov_b32 s66, s70
	s_mov_b32 s67, s71
	v_writelane_b32 v254, s1, 24
	v_cndmask_b32_e64 v7, v1, v16, s[22:23]
	s_and_b64 s[24:25], s[0:1], vcc
	v_writelane_b32 v254, s55, 25
	s_lshl_b32 s0, s55, 7
	v_lshl_add_u64 v[248:249], v[130:131], 0, 32
	buffer_load_dword v7, s[64:67], 0 offen lds
	v_writelane_b32 v254, s0, 26
	s_add_i32 m0, s0, 0x18000
	v_cmp_gt_i64_e64 s[0:1], s[2:3], v[248:249]
	s_lshl_b64 s[10:11], s[90:91], 4
	v_add3_u32 v7, v15, s10, v14
	v_writelane_b32 v254, s0, 27
	v_cndmask_b32_e64 v9, v1, v7, s[24:25]
	s_and_b64 s[26:27], s[0:1], vcc
	v_writelane_b32 v254, s1, 28
	v_writelane_b32 v254, s74, 29
	s_lshl_b32 s0, s74, 7
	v_lshl_add_u64 v[250:251], v[130:131], 0, 48
	buffer_load_dword v9, s[64:67], 0 offen lds
	v_writelane_b32 v254, s0, 30
	s_add_i32 m0, s0, 0x18000
	v_cmp_gt_i64_e64 s[0:1], s[2:3], v[250:251]
	v_add_u32_e32 v7, s10, v7
	v_cndmask_b32_e64 v9, v1, v7, s[26:27]
	v_writelane_b32 v254, s0, 31
	s_and_b64 s[28:29], s[0:1], vcc
	v_lshl_add_u64 v[14:15], v[130:131], 0, 64
	v_writelane_b32 v254, s1, 32
	v_writelane_b32 v254, s75, 33
	s_lshl_b32 s0, s75, 7
	buffer_load_dword v9, s[64:67], 0 offen lds
	v_writelane_b32 v254, s0, 34
	s_add_i32 m0, s0, 0x18000
	v_cmp_gt_i64_e64 s[0:1], s[2:3], v[14:15]
	v_add_u32_e32 v7, s10, v7
	v_cndmask_b32_e64 v9, v1, v7, s[28:29]
	v_writelane_b32 v254, s0, 35
	s_and_b64 s[30:31], s[0:1], vcc
	buffer_load_dword v9, s[64:67], 0 offen lds
	v_writelane_b32 v254, s1, 36
	v_writelane_b32 v254, s76, 37
	s_lshl_b32 s0, s76, 7
	v_writelane_b32 v254, s0, 38
	s_add_i32 m0, s0, 0x18000
	s_mov_b64 s[0:1], 0x50
	scratch_store_dwordx2 off, v[14:15], off
	v_lshl_add_u64 v[14:15], v[130:131], 0, s[0:1]
	v_cmp_gt_i64_e64 s[0:1], s[2:3], v[14:15]
	v_add_u32_e32 v7, s10, v7
	v_cndmask_b32_e64 v9, v1, v7, s[30:31]
	v_writelane_b32 v254, s0, 39
	s_and_b64 s[34:35], s[0:1], vcc
	buffer_load_dword v9, s[64:67], 0 offen lds
	v_writelane_b32 v254, s1, 40
	v_writelane_b32 v254, s77, 41
	s_lshl_b32 s0, s77, 7
	v_writelane_b32 v254, s0, 42
	s_add_i32 m0, s0, 0x18000
	s_mov_b64 s[0:1], 0x60
	v_lshl_add_u64 v[252:253], v[130:131], 0, s[0:1]
	v_cmp_gt_i64_e64 s[0:1], s[2:3], v[252:253]
	v_add_u32_e32 v7, s10, v7
	v_cndmask_b32_e64 v9, v1, v7, s[34:35]
	v_writelane_b32 v254, s0, 43
	s_and_b64 s[36:37], s[0:1], vcc
	buffer_load_dword v9, s[64:67], 0 offen lds
	v_writelane_b32 v254, s1, 44
	v_writelane_b32 v254, s78, 45
	s_lshl_b32 s0, s78, 7
	v_writelane_b32 v254, s0, 46
	s_add_i32 m0, s0, 0x18000
	s_mov_b64 s[0:1], 0x70
	scratch_store_dwordx2 off, v[14:15], off offset:8
	v_lshl_add_u64 v[14:15], v[130:131], 0, s[0:1]
	v_cmp_gt_i64_e64 s[0:1], s[2:3], v[14:15]
	v_add_u32_e32 v7, s10, v7
	v_cndmask_b32_e64 v9, v1, v7, s[36:37]
	v_writelane_b32 v254, s0, 47
	s_and_b64 s[38:39], s[0:1], vcc
	buffer_load_dword v9, s[64:67], 0 offen lds
	v_writelane_b32 v254, s1, 48
	v_writelane_b32 v254, s79, 49
	s_lshl_b32 s0, s79, 7
	v_writelane_b32 v254, s0, 50
	s_add_i32 m0, s0, 0x18000
	s_mov_b64 s[0:1], 0x80
	v_lshl_add_u64 v[148:149], v[130:131], 0, s[0:1]
	v_cmp_gt_i64_e64 s[0:1], s[2:3], v[148:149]
	v_add_u32_e32 v7, s10, v7
	v_cndmask_b32_e64 v9, v1, v7, s[38:39]
	v_writelane_b32 v254, s0, 51
	s_and_b64 s[40:41], s[0:1], vcc
	buffer_load_dword v9, s[64:67], 0 offen lds
	v_writelane_b32 v254, s1, 52
	v_writelane_b32 v254, s80, 53
	s_lshl_b32 s0, s80, 7
	v_writelane_b32 v254, s0, 54
	s_add_i32 m0, s0, 0x18000
	s_mov_b64 s[0:1], 0x90
	v_lshl_add_u64 v[152:153], v[130:131], 0, s[0:1]
	v_cmp_gt_i64_e64 s[0:1], s[2:3], v[152:153]
	v_add_u32_e32 v7, s10, v7
	v_cndmask_b32_e64 v9, v1, v7, s[40:41]
	v_writelane_b32 v254, s0, 55
	v_add_u32_e32 v7, s10, v7
	s_and_b64 vcc, s[0:1], vcc
	v_writelane_b32 v254, s1, 56
	s_lshl_b32 s0, s42, 7
	buffer_load_dword v9, s[64:67], 0 offen lds
	v_cndmask_b32_e32 v7, v1, v7, vcc
	s_add_i32 m0, s0, 0x18000
	s_lshr_b32 s1, s16, 24
	buffer_load_dword v7, s[64:67], 0 offen lds
	v_and_b32_e32 v7, 0xcf, v174
	v_mov_b32_e32 v113, v33
	v_or_b32_e32 v154, s52, v7
	v_mov_b32_e32 v155, s53
	s_mul_i32 s1, s86, s1
	s_mul_hi_u32 s4, s86, s52
	v_cmp_gt_i64_e32 vcc, s[84:85], v[112:113]
	v_cmp_gt_i64_e64 s[6:7], s[12:13], v[154:155]
	s_add_i32 s1, s4, s1
	s_mul_i32 s4, s87, s52
	v_writelane_b32 v254, s0, 57
	s_and_b64 s[44:45], s[6:7], vcc
	s_add_i32 s5, s1, s4
	s_mul_i32 s4, s86, s52
	s_add_u32 s72, s50, s4
	v_writelane_b32 v254, s50, 58
	s_movk_i32 s0, 0xffc0
	v_and_b32_e32 v117, 0xc0, v174
	v_writelane_b32 v254, s51, 59
	v_writelane_b32 v254, s4, 60
	v_mad_i32_i24 v183, v177, s0, v175
	s_addc_u32 s1, s51, s5
	v_writelane_b32 v254, s5, 61
	s_and_b32 s4, s86, 0x3fff
	v_add_u32_e32 v121, v183, v151
	v_mul_lo_u32 v7, s86, v117
	s_bitset1_b32 s4, 14
	v_add_u32_e32 v9, v7, v121
	s_and_b32 s1, s1, 0xffff
	v_writelane_b32 v254, s4, 62
	s_lshl_b32 s4, s4, 16
	scratch_store_dwordx2 off, v[14:15], off offset:16
	s_or_b32 s73, s1, s4
	s_mov_b32 s74, s70
	s_mov_b32 s75, s71
	v_cndmask_b32_e64 v14, v1, v9, s[44:45]
	buffer_load_ubyte v45, v14, s[72:75], 0 offen
	v_or_b32_e32 v14, 0x80, v112
	v_mov_b32_e32 v15, v33
	s_mov_b64 s[18:19], s[62:63]
	v_cmp_gt_i64_e64 s[62:63], s[84:85], v[14:15]
	v_writelane_b32 v254, s6, 63
	s_and_b64 s[44:45], s[6:7], s[62:63]
	v_or_b32_e32 v9, 2, v9
	v_cndmask_b32_e64 v9, v1, v9, s[44:45]
	buffer_load_ubyte v44, v9, s[72:75], 0 offen
	v_lshlrev_b32_e32 v9, 5, v177
	v_sub_u32_e32 v3, v3, v9
	v_add_u32_e32 v3, v3, v112
	v_ashrrev_i32_e32 v9, 31, v3
	v_xor_b32_e32 v3, v9, v3
	v_ashrrev_i32_e32 v14, 31, v3
	v_lshrrev_b32_e32 v14, 25, v14
	v_add_u32_e32 v3, v3, v14
	v_ashrrev_i32_e32 v3, 7, v3
	v_xor_b32_e32 v185, v3, v9
	v_or_b32_e32 v156, 16, v154
	v_mov_b32_e32 v157, s53
	v_or_b32_e32 v9, 1, v121
	v_lshlrev_b32_e32 v14, 8, v185
	v_cmp_gt_i64_e64 s[44:45], s[12:13], v[156:157]
	v_add_u32_e32 v15, v185, v117
	v_sub_u32_e32 v146, v9, v14
	s_and_b64 s[48:49], s[44:45], vcc
	v_mad_u64_u32 v[16:17], s[4:5], s86, v15, v[146:147]
	v_cndmask_b32_e64 v3, v1, v16, s[48:49]
	v_or_b32_e32 v150, 3, v121
	v_cmp_gt_i32_e64 s[50:51], 0, v121
	v_xor_b32_e32 v186, -4, v121
	buffer_load_ubyte v193, v3, s[72:75], 0 offen
	v_cndmask_b32_e64 v3, v150, v186, s[50:51]
	v_ashrrev_i32_e32 v16, 31, v3
	v_add_u32_sdwa v3, v3, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v3, 8, v3
	v_ashrrev_i32_e32 v187, 31, v121
	v_xor_b32_e32 v3, v3, v187
	v_lshlrev_b32_e32 v16, 8, v3
	v_add_u32_e32 v17, v3, v117
	v_sub_u32_e32 v18, v150, v16
	s_and_b64 s[48:49], s[44:45], s[62:63]
	v_mad_u64_u32 v[20:21], s[4:5], s86, v17, v[18:19]
	v_cndmask_b32_e64 v3, v1, v20, s[48:49]
	v_or_b32_e32 v158, 32, v154
	v_mov_b32_e32 v159, s53
	buffer_load_ubyte v191, v3, s[72:75], 0 offen
	v_cmp_gt_i64_e64 s[48:49], s[12:13], v[158:159]
	v_lshl_add_u32 v3, s86, 5, v7
	v_or_b32_e32 v184, 2, v121
	s_and_b64 s[50:51], s[48:49], vcc
	v_add_u32_e32 v7, v3, v121
	v_cndmask_b32_e64 v7, v1, v7, s[50:51]
	s_and_b64 s[50:51], s[48:49], s[62:63]
	v_add_u32_e32 v3, v3, v184
	v_cndmask_b32_e64 v3, v1, v3, s[50:51]
	v_or_b32_e32 v160, 48, v154
	v_writelane_b32 v254, s52, 1
	buffer_load_ubyte v195, v7, s[72:75], 0 offen
	buffer_load_ubyte v192, v3, s[72:75], 0 offen
	v_mov_b32_e32 v161, s53
	v_cmp_gt_i64_e64 s[50:51], s[12:13], v[160:161]
	v_add_u32_e32 v3, 32, v15
	v_writelane_b32 v254, s53, 2
	s_and_b64 s[52:53], s[50:51], vcc
	v_mad_u64_u32 v[20:21], s[4:5], s86, v3, v[146:147]
	v_cndmask_b32_e64 v3, v1, v20, s[52:53]
	buffer_load_ubyte v196, v3, s[72:75], 0 offen
	s_and_b64 s[52:53], s[50:51], s[62:63]
	v_add_u32_e32 v3, 32, v17
	v_mad_u64_u32 v[18:19], s[4:5], s86, v3, v[18:19]
	s_add_u32 s1, s18, s95
	s_addc_u32 s4, s19, 0
	v_or_b32_e32 v162, s1, v13
	s_bfe_u32 s1, s95, 0x10004
	v_bitop3_b32 v7, v187, v121, s1 bitop3:0x1e
	v_cndmask_b32_e64 v3, v1, v18, s[52:53]
	v_ashrrev_i32_e32 v18, 31, v7
	v_add_u32_sdwa v7, v7, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v7, 8, v7
	s_and_b32 s17, s95, 0x60
	v_mov_b32_e32 v163, s4
	v_xad_u32 v18, v7, v187, s17
	s_movk_i32 s4, 0xfd
	v_mov_b32_e32 v7, s1
	v_writelane_b32 v255, s7, 0
	v_bitop3_b32 v34, v121, s4, v7 bitop3:0xc8
	s_mul_i32 s4, s88, s19
	s_mul_hi_u32 s5, s88, s18
	v_cmp_gt_i64_e64 s[52:53], s[2:3], v[162:163]
	s_add_i32 s4, s5, s4
	s_mul_i32 s5, s89, s18
	v_writelane_b32 v255, s18, 1
	s_and_b64 s[54:55], s[52:53], vcc
	s_add_i32 s23, s4, s5
	v_writelane_b32 v255, s19, 2
	s_mul_i32 s22, s88, s18
	s_add_u32 s76, s8, s22
	v_writelane_b32 v255, s8, 3
	v_mad_u64_u32 v[26:27], s[4:5], v18, s88, v[34:35]
	s_nop 0
	v_writelane_b32 v255, s9, 4
	v_writelane_b32 v255, s22, 5
	s_addc_u32 s4, s9, s23
	s_and_b32 s5, s88, 0x3fff
	v_writelane_b32 v255, s23, 6
	s_bitset1_b32 s5, 14
	s_and_b32 s4, s4, 0xffff
	v_writelane_b32 v255, s5, 7
	s_lshl_b32 s5, s5, 16
	buffer_load_ubyte v194, v3, s[72:75], 0 offen
	v_or_b32_e32 v3, s1, v121
	s_or_b32 s77, s4, s5
	s_mov_b32 s78, s70
	s_mov_b32 s79, s71
	v_cndmask_b32_e64 v7, v1, v26, s[54:55]
	buffer_load_ubyte v47, v7, s[76:79], 0 offen
	v_or_b32_e32 v7, 2, v3
	v_cmp_gt_i32_e64 s[56:57], -2, v3
	v_bitop3_b32 v3, v121, -3, s1 bitop3:0x36
	v_or_b32_e32 v115, s95, v13
	v_cndmask_b32_e64 v3, v7, v3, s[56:57]
	v_ashrrev_i32_e32 v7, 31, v3
	v_add_u32_sdwa v3, v3, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v3, 8, v3
	v_cndmask_b32_e64 v7, 0, -1, s[56:57]
	v_xor_b32_e32 v3, v3, v7
	v_add_u32_e32 v21, s17, v3
	v_or_b32_e32 v7, s1, v184
	v_lshlrev_b32_e32 v19, 8, v3
	v_sub_u32_e32 v7, v7, v19
	v_mul_lo_u32 v20, v21, s88
	s_and_b64 s[54:55], s[52:53], s[62:63]
	v_add_u32_e32 v3, v7, v20
	s_mul_i32 s1, s94, 0xffffff85
	v_cndmask_b32_e64 v3, v1, v3, s[54:55]
	v_add_u32_e32 v188, s1, v5
	v_add_u32_e32 v119, 16, v115
	v_mad_i32_i24 v189, v177, s0, v151
	buffer_load_ubyte v46, v3, s[76:79], 0 offen
	v_lshrrev_b32_e32 v3, 5, v119
	v_add_u32_e32 v190, v189, v188
	v_mad_i32_i24 v5, v3, -2, v190
	v_add_u32_e32 v22, 1, v5
	v_cmp_gt_i32_e64 s[58:59], -1, v5
	v_sub_u32_e32 v23, 0xfffe, v5
	s_mul_i32 s18, s94, 5
	v_cndmask_b32_e64 v22, v22, v23, s[58:59]
	v_ashrrev_i16_e32 v23, 15, v22
	v_add_u16_sdwa v22, v22, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
	v_cndmask_b32_e64 v23, 0, -1, s[58:59]
	v_xor_b32_sdwa v22, sext(v22), v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD
	v_mad_i32_i24 v29, v3, -2, s18
	v_mov_b32_e32 v30, 8
	v_lshl_add_u64 v[164:165], v[162:163], 0, 16
	v_and_b32_e32 v27, 0xe0, v119
	v_add_u32_e32 v9, v9, v29
	v_lshlrev_b32_sdwa v23, v30, sext(v22) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_cmp_gt_i64_e64 s[54:55], s[2:3], v[164:165]
	v_add_u32_sdwa v24, v27, sext(v22) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_sub_u32_e32 v28, v9, v23
	s_and_b64 s[56:57], s[54:55], vcc
	v_mad_u64_u32 v[24:25], s[0:1], s88, v24, v[28:29]
	v_cndmask_b32_e64 v9, v1, v24, s[56:57]
	buffer_load_ubyte v53, v9, s[76:79], 0 offen
	v_add_u32_e32 v9, 3, v5
	v_cmp_gt_i32_e64 s[58:59], -3, v5
	v_sub_u32_e32 v5, 0xfffc, v5
	s_and_b64 s[56:57], s[54:55], s[62:63]
	v_cndmask_b32_e64 v5, v9, v5, s[58:59]
	v_ashrrev_i16_e32 v9, 15, v5
	v_add_u16_sdwa v5, v5, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
	v_cndmask_b32_e64 v9, 0, -1, s[58:59]
	v_xor_b32_sdwa v25, sext(v5), v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD
	v_add_u32_e32 v9, v150, v29
	v_lshlrev_b32_sdwa v24, v30, sext(v25) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v5, v27, sext(v25) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_sub_u32_e32 v30, v9, v24
	v_mad_u64_u32 v[36:37], s[0:1], s88, v5, v[30:31]
	v_lshl_add_u64 v[166:167], v[162:163], 0, 32
	v_cndmask_b32_e64 v5, v1, v36, s[56:57]
	v_cmp_gt_i64_e64 s[56:57], s[2:3], v[166:167]
	s_lshl_b32 s4, s88, 5
	buffer_load_ubyte v52, v5, s[76:79], 0 offen
	s_and_b64 s[58:59], s[56:57], vcc
	v_add_u32_e32 v5, s4, v26
	v_cndmask_b32_e64 v9, v1, v5, s[58:59]
	buffer_load_ubyte v55, v9, s[76:79], 0 offen
	v_add_u32_e32 v9, s4, v20
	s_and_b64 s[58:59], s[56:57], s[62:63]
	v_add_u32_e32 v26, v7, v9
	s_add_i32 s0, s95, 48
	v_cndmask_b32_e64 v26, v1, v26, s[58:59]
	v_lshl_add_u64 v[168:169], v[162:163], 0, 48
	s_and_b32 s5, s0, 0xe0
	buffer_load_ubyte v54, v26, s[76:79], 0 offen
	v_cmp_gt_i64_e64 s[58:59], s[2:3], v[168:169]
	v_add_u32_sdwa v26, s5, sext(v22) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	s_and_b64 s[60:61], s[58:59], vcc
	v_mad_u64_u32 v[28:29], s[0:1], s88, v26, v[28:29]
	v_cndmask_b32_e64 v27, v1, v28, s[60:61]
	buffer_load_ubyte v198, v27, s[76:79], 0 offen
	v_add_u32_sdwa v27, s5, sext(v25) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	s_and_b64 s[60:61], s[58:59], s[62:63]
	v_mad_u64_u32 v[28:29], s[0:1], s88, v27, v[30:31]
	v_lshl_add_u64 v[170:171], v[162:163], 0, 64
	v_cndmask_b32_e64 v28, v1, v28, s[60:61]
	v_writelane_b32 v255, s2, 8
	v_add_u32_e32 v5, s4, v5
	buffer_load_ubyte v197, v28, s[76:79], 0 offen
	v_cmp_gt_i64_e64 s[60:61], s[2:3], v[170:171]
	s_and_b64 vcc, s[60:61], vcc
	v_cndmask_b32_e32 v5, v1, v5, vcc
	buffer_load_ubyte v199, v5, s[76:79], 0 offen
	s_and_b64 vcc, s[60:61], s[62:63]
	v_add3_u32 v5, v9, s4, v7
	v_cndmask_b32_e32 v1, v1, v5, vcc
	buffer_load_ubyte v200, v1, s[76:79], 0 offen
	s_cmp_eq_u32 s94, 0
	v_writelane_b32 v255, s3, 9
	s_cselect_b64 s[0:1], -1, 0
	v_writelane_b32 v255, s0, 10
	s_mov_b32 s93, 0x18000
	s_and_b64 vcc, exec, s[0:1]
	v_writelane_b32 v255, s1, 11
	s_barrier
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	s_ashr_i32 s0, s97, 31
	s_mov_b32 s1, s0
	s_xor_b64 s[4:5], s[0:1], s[96:97]
	s_lshr_b32 s9, s5, 31
	s_add_u32 s4, s4, s9
	s_addc_u32 s5, s5, 0
	v_and_b32_e32 v1, 7, v174
	s_ashr_i64 s[4:5], s[4:5], 1
	v_lshlrev_b32_e32 v7, 7, v174
	v_lshlrev_b32_e32 v9, 11, v177
	s_xor_b64 s[0:1], s[4:5], s[0:1]
	v_bitop3_b32 v5, v147, v174, 7 bitop3:0x78
	v_sub_u32_e32 v7, v7, v9
	s_mul_i32 s4, s94, 0x2800
	v_bitop3_b32 v1, v147, v1, 4 bitop3:0x36
	s_lshl_b64 s[8:9], s[0:1], 1
	v_lshl_add_u32 v9, v172, 13, v7
	v_lshlrev_b32_e32 v5, 4, v5
	v_add_u32_e32 v7, s4, v7
	v_lshlrev_b32_e32 v1, 4, v1
	v_cmp_lt_i64_e64 s[0:1], s[0:1], 1
	s_mov_b32 s80, 0
	v_or_b32_e32 v48, v9, v5
	v_or_b32_e32 v50, v7, v5
	v_or_b32_e32 v49, v1, v9
	v_or_b32_e32 v51, v1, v7
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_11
	v_add_u32_e32 v31, 0x41, v18
	v_mul_lo_u32 v56, s88, v31
	v_add_u32_e32 v31, 33, v18
	v_mul_lo_u32 v18, s88, v18
	v_add_u32_e32 v58, s88, v18
	v_add_u32_e32 v18, 33, v117
	v_add3_u32 v13, s95, v13, 16
	v_mul_i32_i24_e32 v28, 0xffffffc0, v177
	v_add_u32_e32 v30, 1, v117
	v_mul_lo_u32 v18, s86, v18
	v_and_b32_e32 v13, 0x1e0, v13
	v_mul_i32_i24_e32 v29, -2, v3
	v_add3_u32 v59, v28, v18, v151
	v_mul_lo_u32 v18, s86, v30
	v_add_u32_sdwa v25, v13, sext(v25) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_sdwa v13, v13, sext(v22) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add3_u32 v60, v28, v18, v151
	v_add_u32_e32 v18, v28, v29
	v_mul_lo_u32 v13, s88, v13
	v_add3_u32 v13, v13, s88, v18
	v_mul_lo_u32 v27, s88, v27
	v_mul_lo_u32 v26, s88, v26
	v_mul_lo_u32 v25, s88, v25
	v_add3_u32 v13, v13, v151, s18
	v_add3_u32 v27, v27, s88, v18
	v_add3_u32 v26, v26, s88, v18
	v_add3_u32 v25, v25, s88, v18
	v_sub_u32_e32 v13, v13, v23
	v_add_u32_e32 v18, 0x41, v21
	v_add_u32_e32 v201, 1, v13
	v_add_u32_e32 v13, s94, v28
	v_mul_lo_u32 v18, s88, v18
	v_add3_u32 v18, v13, v18, v151
	v_sub_u32_e32 v18, v18, v19
	v_add_u32_e32 v202, 2, v18
	v_add_u32_e32 v18, 33, v21
	v_writelane_b32 v255, s96, 12
	v_mul_lo_u32 v18, s88, v18
	v_add3_u32 v18, v13, v18, v151
	v_writelane_b32 v255, s97, 13
	v_writelane_b32 v255, s82, 14
	v_sub_u32_e32 v18, v18, v19
	v_add_u32_e32 v203, 2, v18
	v_writelane_b32 v255, s83, 15
	v_add_u32_e32 v18, s88, v20
	v_writelane_b32 v255, s12, 16
	v_add3_u32 v13, v13, v18, v151
	v_sub_u32_e32 v13, v13, v19
	v_writelane_b32 v255, s13, 17
	v_writelane_b32 v255, s81, 18
	s_lshl_b32 s0, s81, 7
	v_add_u32_e32 v204, 2, v13
	v_add_u32_e32 v13, 33, v17
	v_writelane_b32 v255, s0, 19
	v_readlane_b32 s0, v254, 3
	v_mul_lo_u32 v13, s86, v13
	s_lshl_b32 s0, s0, 7
	v_mov_b32_e32 v3, v33
	v_ashrrev_i32_e32 v9, 31, v8
	v_add3_u32 v13, v28, v13, v151
	v_writelane_b32 v255, s0, 20
	v_readlane_b32 s0, v254, 4
	v_ashrrev_i32_e32 v7, 31, v6
	v_sub_u32_e32 v13, v13, v16
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[8:9]
	s_lshl_b32 s0, s0, 7
	v_ashrrev_i32_e32 v5, 31, v4
	v_add_u32_e32 v205, 3, v13
	v_mul_lo_u32 v13, s86, v17
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[6:7]
	v_writelane_b32 v255, s0, 21
	v_mov_b32_e32 v1, v33
	v_add_u32_e32 v13, s86, v13
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	s_add_u32 s4, s8, -1
	v_writelane_b32 v255, s8, 22
	v_add3_u32 v13, v28, v13, v151
	v_lshl_add_u64 v[0:1], v[2:3], 0, v[0:1]
	s_movk_i32 s0, 0xf7ff
	v_writelane_b32 v255, s9, 23
	s_addc_u32 s5, s9, -1
	v_sub_u32_e32 v13, v13, v16
	v_sub_co_u32_e32 v36, vcc, s0, v0
	s_mov_b64 s[0:1], 0x800
	s_lshl_b32 s8, s16, 8
	v_add_u32_e32 v206, 3, v13
	v_add_u32_e32 v13, 33, v15
	v_lshl_add_u64 v[38:39], v[0:1], 0, s[0:1]
	v_or_b32_e32 v0, s8, v12
	v_mul_lo_u32 v13, s86, v13
	v_add_u32_e32 v0, v0, v176
	v_add3_u32 v13, v28, v13, v151
	v_subb_co_u32_e32 v37, vcc, -1, v1, vcc
	v_add_u32_e32 v1, 0xc0, v0
	v_sub_u32_e32 v13, v13, v14
	v_mul_lo_u32 v1, v1, s90
	v_add_u32_e32 v207, 1, v13
	v_mul_lo_u32 v13, s86, v15
	v_add_u32_e32 v209, 0x80, v1
	v_or_b32_e32 v1, 0x80, v0
	v_writelane_b32 v255, s94, 24
	v_add_u32_e32 v13, s86, v13
	v_mul_lo_u32 v1, v1, s90
	v_add3_u32 v27, v27, v151, s18
	v_add3_u32 v26, v26, v151, s18
	v_add3_u32 v25, v25, v151, s18
	v_add3_u32 v13, v28, v13, v151
	v_add_u32_e32 v210, 0x80, v1
	v_add_u32_e32 v1, 64, v0
	v_mul_lo_u32 v0, v0, s90
	v_readlane_b32 s22, v255, 1
	v_sub_u32_e32 v27, v27, v24
	v_sub_u32_e32 v26, v26, v23
	v_sub_u32_e32 v24, v25, v24
	v_sub_u32_e32 v13, v13, v14
	s_mov_b64 s[0:1], 0x100
	v_mul_lo_u32 v1, v1, s90
	v_add_u32_e32 v212, 0x80, v0
	v_mov_b32_e32 v0, 0
	v_readlane_b32 s23, v255, 2
	v_readlane_b32 s26, v254, 13
	v_readlane_b32 s28, v254, 17
	v_readlane_b32 s30, v254, 20
	v_readlane_b32 s14, v254, 23
	v_readlane_b32 s34, v254, 27
	v_readlane_b32 s36, v254, 31
	v_readlane_b32 s6, v254, 35
	v_readlane_b32 s38, v254, 39
	v_readlane_b32 s40, v254, 43
	v_readlane_b32 s82, v254, 47
	v_readlane_b32 s42, v254, 51
	v_readlane_b32 s24, v254, 55
	v_readlane_b32 s2, v254, 63
	v_mul_lo_u32 v57, s88, v31
	v_add_u32_e32 v61, 3, v27
	v_add_u32_e32 v62, 1, v26
	v_add_u32_e32 v63, 3, v24
	v_add_u32_e32 v208, 1, v13
	v_lshl_add_u64 v[40:41], v[10:11], 0, s[0:1]
	v_or_b32_e32 v42, 0x100, v32
	v_mov_b32_e32 v43, v33
	v_add_u32_e32 v211, 0x80, v1
	s_mov_b64 s[94:95], 0
	s_mov_b32 s87, 0
	s_mov_b32 s8, 0x8000
	s_mov_b32 s9, 0x1d000
	v_bfrev_b32_e32 v213, -2
	s_mov_b32 s99, s98
	s_mov_b32 s66, s70
	s_mov_b32 s67, s71
	s_mov_b32 s74, s70
	s_mov_b32 s75, s71
	s_mov_b32 s78, s70
	s_mov_b32 s79, s71
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v64, v0
	v_mov_b32_e32 v65, v0
	v_mov_b32_e32 v66, v0
	v_mov_b32_e32 v67, v0
	v_mov_b32_e32 v68, v0
	v_mov_b32_e32 v69, v0
	v_mov_b32_e32 v70, v0
	v_mov_b32_e32 v71, v0
	v_mov_b32_e32 v72, v0
	v_mov_b32_e32 v73, v0
	v_mov_b32_e32 v74, v0
	v_mov_b32_e32 v75, v0
	v_mov_b32_e32 v76, v0
	v_mov_b32_e32 v77, v0
	v_mov_b32_e32 v78, v0
	v_mov_b32_e32 v79, v0
	v_mov_b32_e32 v80, v0
	v_mov_b32_e32 v81, v0
	v_mov_b32_e32 v82, v0
	v_mov_b32_e32 v83, v0
	v_mov_b32_e32 v84, v0
	v_mov_b32_e32 v85, v0
	v_mov_b32_e32 v86, v0
	v_mov_b32_e32 v87, v0
	v_mov_b32_e32 v88, v0
	v_mov_b32_e32 v89, v0
	v_mov_b32_e32 v90, v0
	v_mov_b32_e32 v91, v0
	v_mov_b32_e32 v92, v0
	v_mov_b32_e32 v93, v0
	v_mov_b32_e32 v94, v0
	v_mov_b32_e32 v95, v0
	v_mov_b32_e32 v96, v0
	v_mov_b32_e32 v97, v0
	v_mov_b32_e32 v98, v0
	v_mov_b32_e32 v99, v0
	v_mov_b32_e32 v100, v0
	v_mov_b32_e32 v101, v0
	v_mov_b32_e32 v102, v0
	v_mov_b32_e32 v103, v0
	v_mov_b32_e32 v104, v0
	v_mov_b32_e32 v105, v0
	v_mov_b32_e32 v106, v0
	v_mov_b32_e32 v107, v0
	v_mov_b32_e32 v108, v0
	v_mov_b32_e32 v109, v0
	v_mov_b32_e32 v110, v0
	v_mov_b32_e32 v111, v0
	v_readlane_b32 s23, v254, 19
	v_readlane_b32 s27, v254, 14
	v_readlane_b32 s29, v254, 18
	v_readlane_b32 s31, v254, 21
	v_readlane_b32 s11, v254, 22
	v_readlane_b32 s15, v254, 24
	v_readlane_b32 s21, v254, 26
	v_readlane_b32 s35, v254, 28
	v_readlane_b32 s33, v254, 30
	v_readlane_b32 s92, v254, 34
	v_readlane_b32 s37, v254, 32
	v_readlane_b32 s20, v254, 38
	v_readlane_b32 s7, v254, 36
	v_readlane_b32 s39, v254, 40
	v_readlane_b32 s46, v254, 42
	v_readlane_b32 s41, v254, 44
	v_readlane_b32 s47, v254, 46
	v_readlane_b32 s83, v254, 48
	v_readlane_b32 s12, v254, 50
	v_readlane_b32 s43, v254, 52
	v_readlane_b32 s81, v254, 54
	v_readlane_b32 s25, v254, 56
	v_readlane_b32 s13, v254, 57
	v_readlane_b32 s3, v255, 0
	scratch_store_dwordx2 off, v[250:251], off offset:56
	scratch_store_dwordx2 off, v[248:249], off offset:48
	scratch_store_dwordx2 off, v[244:245], off offset:40
	scratch_store_dwordx2 off, v[242:243], off offset:32
	scratch_store_dwordx2 off, v[240:241], off offset:24
.LBB0_8:
	s_mov_b32 s89, s80
	s_mov_b32 s80, s8
	s_mov_b32 s8, s93
	s_mov_b32 s93, s9
	v_lshl_add_u64 v[214:215], v[42:43], 0, s[94:95]
	v_readlane_b32 s0, v254, 9
	v_cmp_gt_i64_e32 vcc, s[84:85], v[214:215]
	v_readlane_b32 s1, v254, 10
	v_add_u32_e32 v32, v35, v212
	s_and_b64 s[16:17], s[0:1], vcc
	v_cndmask_b32_e64 v32, v213, v32, s[16:17]
	s_add_i32 m0, s80, s23
	v_readlane_b32 s0, v255, 19
	s_waitcnt vmcnt(0)
	s_barrier
	s_barrier
	buffer_load_dwordx4 v32, s[68:71], 0 offen lds
	v_add_u32_e32 v32, v35, v211
	s_and_b64 s[16:17], s[26:27], vcc
	s_add_i32 m0, s80, s0
	v_readlane_b32 s0, v254, 15
	v_cndmask_b32_e64 v32, v213, v32, s[16:17]
	v_readlane_b32 s1, v254, 16
	buffer_load_dwordx4 v32, s[68:71], 0 offen lds
	v_add_u32_e32 v32, v35, v210
	s_and_b64 s[16:17], s[0:1], vcc
	v_readlane_b32 s0, v255, 20
	v_cndmask_b32_e64 v32, v213, v32, s[16:17]
	s_add_i32 m0, s80, s0
	v_readlane_b32 s0, v255, 21
	buffer_load_dwordx4 v32, s[68:71], 0 offen lds
	s_and_b64 vcc, s[28:29], vcc
	s_add_i32 m0, s80, s0
	s_add_u32 s16, s90, s98
	s_addc_u32 s17, s91, s98
	s_xor_b64 s[96:97], s[16:17], s[98:99]
	v_cvt_f32_u32_e32 v214, s96
	v_cvt_f32_u32_e32 v215, s97
	v_add_u32_e32 v32, v35, v209
	v_cndmask_b32_e32 v32, v213, v32, vcc
	buffer_load_dwordx4 v32, s[68:71], 0 offen lds
	v_fmac_f32_e32 v214, 0x4f800000, v215
	v_rcp_f32_e32 v32, v214
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_sub_u32 s9, 0, s96
	s_subb_u32 s16, 0, s97
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v216, 0x2f800000, v32
	v_trunc_f32_e32 v216, v216
	v_fmac_f32_e32 v32, 0xcf800000, v216
	v_cvt_u32_f32_e32 v32, v32
	v_cvt_u32_f32_e32 v216, v216
	v_cndmask_b32_e32 v215, v39, v37, vcc
	v_cndmask_b32_e32 v214, v38, v36, vcc
	v_readfirstlane_b32 s17, v32
	v_readfirstlane_b32 vcc_lo, v216
	s_mul_hi_u32 s63, s9, s17
	s_mul_i32 vcc_hi, s9, vcc_lo
	s_mul_i32 s62, s16, s17
	s_add_i32 s63, s63, vcc_hi
	s_add_i32 s63, s63, s62
	s_mul_i32 s0, s9, s17
	s_mul_i32 vcc_hi, s17, s63
	s_mul_hi_u32 s1, s17, s0
	s_mul_hi_u32 s62, s17, s63
	s_add_u32 s1, s1, vcc_hi
	s_addc_u32 s62, 0, s62
	s_mul_hi_u32 s18, vcc_lo, s0
	s_mul_i32 s0, vcc_lo, s0
	s_add_u32 s0, s1, s0
	s_mul_hi_u32 vcc_hi, vcc_lo, s63
	s_addc_u32 s0, s62, s18
	s_addc_u32 s1, vcc_hi, 0
	s_mul_i32 s18, vcc_lo, s63
	s_add_u32 s0, s0, s18
	s_addc_u32 s1, 0, s1
	s_add_u32 s0, s17, s0
	s_addc_u32 s1, vcc_lo, s1
	s_mul_i32 s17, s9, s1
	s_mul_hi_u32 s18, s9, s0
	s_add_i32 s17, s18, s17
	s_mul_i32 s16, s16, s0
	s_add_i32 s17, s17, s16
	s_mul_i32 s9, s9, s0
	s_mul_hi_u32 s18, s1, s9
	s_mul_i32 s19, s1, s9
	s_mul_i32 s63, s0, s17
	s_mul_hi_u32 s9, s0, s9
	s_mul_hi_u32 s62, s0, s17
	s_add_u32 s9, s9, s63
	s_addc_u32 s62, 0, s62
	s_add_u32 s9, s9, s19
	s_mul_hi_u32 s16, s1, s17
	s_addc_u32 s9, s62, s18
	s_addc_u32 s16, s16, 0
	s_mul_i32 s17, s1, s17
	s_add_u32 s9, s9, s17
	v_ashrrev_i32_e32 v216, 31, v215
	s_addc_u32 s16, 0, s16
	v_mov_b32_e32 v217, v216
	s_add_u32 s0, s0, s9
	v_lshl_add_u64 v[214:215], v[214:215], 0, v[216:217]
	s_addc_u32 s1, s1, s16
	v_xor_b32_e32 v222, v214, v216
	v_xor_b32_e32 v217, v215, v216
	v_mad_u64_u32 v[214:215], s[16:17], v222, s1, 0
	v_mul_hi_u32 v32, v222, s0
	v_lshl_add_u64 v[214:215], v[32:33], 0, v[214:215]
	v_mad_u64_u32 v[220:221], s[16:17], v217, s0, 0
	v_add_co_u32_e32 v32, vcc, v214, v220
	v_mad_u64_u32 v[218:219], s[16:17], v217, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v214, vcc, v215, v221, vcc
	v_mov_b32_e32 v215, s87
	s_nop 0
	v_addc_co_u32_e32 v219, vcc, 0, v219, vcc
	v_lshl_add_u64 v[214:215], v[214:215], 0, v[218:219]
	v_mul_lo_u32 v32, s97, v214
	v_mul_lo_u32 v215, s96, v215
	v_mad_u64_u32 v[218:219], s[16:17], s96, v214, 0
	v_add3_u32 v32, v219, v215, v32
	v_sub_u32_e32 v215, v217, v32
	v_mov_b32_e32 v223, s97
	v_sub_co_u32_e32 v218, vcc, v222, v218
	s_add_i32 m0, s93, s11
	s_nop 0
	v_subb_co_u32_e64 v215, s[16:17], v215, v223, vcc
	v_subrev_co_u32_e64 v219, s[16:17], s96, v218
	v_subb_co_u32_e32 v32, vcc, v217, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v215, s[16:17], 0, v215, s[16:17]
	v_cmp_le_u32_e64 s[16:17], s97, v215
	v_cmp_le_u32_e32 vcc, s97, v32
	s_nop 0
	v_cndmask_b32_e64 v220, 0, -1, s[16:17]
	v_cmp_le_u32_e64 s[16:17], s96, v219
	v_cndmask_b32_e64 v217, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s96, v218
	v_cndmask_b32_e64 v219, 0, -1, s[16:17]
	v_cmp_eq_u32_e64 s[16:17], s97, v215
	v_cndmask_b32_e64 v218, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s97, v32
	v_cndmask_b32_e64 v215, v220, v219, s[16:17]
	v_add_u32_e32 v219, 2, v214
	v_add_u32_e32 v220, 1, v214
	v_cmp_ne_u32_e64 s[16:17], 0, v215
	v_cndmask_b32_e32 v32, v217, v218, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v215, v220, v219, s[16:17]
	s_nop 0
	v_cndmask_b32_e32 v32, v214, v215, vcc
	v_xor_b32_e32 v214, s98, v216
	v_xor_b32_e32 v32, v32, v214
	v_sub_u32_e32 v32, v32, v214
	v_ashrrev_i32_e32 v214, 31, v39
	v_mov_b32_e32 v215, v214
	v_lshl_add_u64 v[216:217], v[38:39], 0, v[214:215]
	v_xor_b32_e32 v224, v216, v214
	v_xad_u32 v222, v32, v214, s22
	v_xor_b32_e32 v215, v217, v214
	v_mad_u64_u32 v[216:217], s[16:17], v224, s1, 0
	v_mul_hi_u32 v32, v224, s0
	v_lshl_add_u64 v[216:217], v[32:33], 0, v[216:217]
	v_mad_u64_u32 v[220:221], s[16:17], v215, s0, 0
	v_add_co_u32_e32 v32, vcc, v216, v220
	v_mad_u64_u32 v[218:219], s[16:17], v215, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v216, vcc, v217, v221, vcc
	v_mov_b32_e32 v217, s87
	s_nop 0
	v_addc_co_u32_e32 v219, vcc, 0, v219, vcc
	v_lshl_add_u64 v[216:217], v[216:217], 0, v[218:219]
	v_mul_lo_u32 v32, s97, v216
	v_mul_lo_u32 v218, s96, v217
	v_mad_u64_u32 v[216:217], s[16:17], s96, v216, 0
	v_add3_u32 v32, v217, v218, v32
	v_sub_u32_e32 v217, v215, v32
	v_sub_co_u32_e32 v216, vcc, v224, v216
	s_nop 1
	v_subb_co_u32_e64 v217, s[16:17], v217, v223, vcc
	v_subrev_co_u32_e64 v218, s[16:17], s96, v216
	v_subb_co_u32_e32 v32, vcc, v215, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v219, s[62:63], 0, v217, s[16:17]
	v_cmp_le_u32_e64 s[62:63], s97, v219
	v_subb_co_u32_e64 v217, s[16:17], v217, v223, s[16:17]
	s_nop 0
	v_cndmask_b32_e64 v220, 0, -1, s[62:63]
	v_cmp_le_u32_e64 s[62:63], s96, v218
	v_cmp_le_u32_e32 vcc, s97, v32
	s_nop 0
	v_cndmask_b32_e64 v221, 0, -1, s[62:63]
	v_cmp_eq_u32_e64 s[62:63], s97, v219
	v_cndmask_b32_e64 v215, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s96, v216
	v_cndmask_b32_e64 v220, v220, v221, s[62:63]
	v_subrev_co_u32_e64 v221, s[16:17], s96, v218
	s_nop 1
	v_subbrev_co_u32_e64 v217, s[16:17], 0, v217, s[16:17]
	v_cmp_ne_u32_e64 s[16:17], 0, v220
	s_nop 1
	v_cndmask_b32_e64 v217, v219, v217, s[16:17]
	v_cndmask_b32_e64 v219, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s97, v32
	s_nop 1
	v_cndmask_b32_e32 v215, v215, v219, vcc
	v_cmp_ne_u32_e32 vcc, 0, v215
	v_cndmask_b32_e64 v215, v218, v221, s[16:17]
	s_nop 0
	v_cndmask_b32_e32 v215, v216, v215, vcc
	v_cndmask_b32_e32 v32, v32, v217, vcc
	v_xor_b32_e32 v215, v215, v214
	v_xor_b32_e32 v32, v32, v214
	v_sub_co_u32_e32 v215, vcc, v215, v214
	v_mul_lo_u32 v216, v222, s90
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v214, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s90, v32
	v_add_u32_e32 v32, v32, v215
	v_lshl_add_u64 v[214:215], v[40:41], 0, s[94:95]
	v_cmp_gt_i64_e32 vcc, s[84:85], v[214:215]
	v_add_u32_e32 v217, v32, v216
	s_and_b64 s[16:17], s[30:31], vcc
	v_cndmask_b32_e64 v214, v213, v217, s[16:17]
	v_add3_u32 v32, v216, s10, v32
	s_and_b64 s[16:17], s[14:15], vcc
	buffer_load_dword v214, s[64:67], 0 offen lds
	v_cndmask_b32_e64 v214, v213, v32, s[16:17]
	s_add_i32 m0, s93, s21
	v_add_u32_e32 v32, s10, v32
	s_and_b64 s[16:17], s[34:35], vcc
	buffer_load_dword v214, s[64:67], 0 offen lds
	v_cndmask_b32_e64 v214, v213, v32, s[16:17]
	s_add_i32 m0, s93, s33
	v_add_u32_e32 v32, s10, v32
	s_and_b64 s[16:17], s[36:37], vcc
	buffer_load_dword v214, s[64:67], 0 offen lds
	v_cndmask_b32_e64 v214, v213, v32, s[16:17]
	s_add_i32 m0, s93, s92
	v_add_u32_e32 v32, s10, v32
	s_and_b64 s[16:17], s[6:7], vcc
	buffer_load_dword v214, s[64:67], 0 offen lds
	v_cndmask_b32_e64 v214, v213, v32, s[16:17]
	s_add_i32 m0, s93, s20
	v_add_u32_e32 v32, s10, v32
	s_and_b64 s[16:17], s[38:39], vcc
	buffer_load_dword v214, s[64:67], 0 offen lds
	v_cndmask_b32_e64 v214, v213, v32, s[16:17]
	s_add_i32 m0, s93, s46
	v_add_u32_e32 v32, s10, v32
	s_and_b64 s[16:17], s[40:41], vcc
	buffer_load_dword v214, s[64:67], 0 offen lds
	v_cndmask_b32_e64 v214, v213, v32, s[16:17]
	s_add_i32 m0, s93, s47
	v_add_u32_e32 v32, s10, v32
	s_and_b64 s[16:17], s[82:83], vcc
	buffer_load_dword v214, s[64:67], 0 offen lds
	v_cndmask_b32_e64 v214, v213, v32, s[16:17]
	s_add_i32 m0, s93, s12
	v_add_u32_e32 v32, s10, v32
	s_and_b64 s[16:17], s[42:43], vcc
	buffer_load_dword v214, s[64:67], 0 offen lds
	v_cndmask_b32_e64 v214, v213, v32, s[16:17]
	s_add_i32 m0, s93, s81
	v_add_u32_e32 v32, s10, v32
	s_and_b64 vcc, s[24:25], vcc
	buffer_load_dword v214, s[64:67], 0 offen lds
	v_cndmask_b32_e32 v32, v213, v32, vcc
	s_add_i32 m0, s93, s13
	s_nop 0
	buffer_load_dword v32, s[64:67], 0 offen lds
	v_lshl_add_u64 v[214:215], v[112:113], 0, s[94:95]
	s_mov_b64 s[0:1], 0x100
	v_lshl_add_u64 v[216:217], v[214:215], 0, s[0:1]
	v_cmp_gt_i64_e32 vcc, s[84:85], v[216:217]
	v_add_u32_e32 v32, v175, v60
	s_and_b64 s[16:17], s[2:3], vcc
	v_cndmask_b32_e64 v216, v213, v32, s[16:17]
	s_mov_b64 s[16:17], 0x180
	v_lshl_add_u64 v[214:215], v[214:215], 0, s[16:17]
	v_cmp_gt_i64_e64 s[16:17], s[84:85], v[214:215]
	v_add_u32_e32 v32, 2, v32
	s_and_b64 s[62:63], s[2:3], s[16:17]
	v_cndmask_b32_e64 v214, v213, v32, s[62:63]
	v_add_u32_e32 v32, v175, v208
	s_and_b64 s[62:63], s[44:45], vcc
	v_cndmask_b32_e64 v215, v213, v32, s[62:63]
	v_add_u32_e32 v32, v175, v206
	s_and_b64 s[62:63], s[44:45], s[16:17]
	v_cndmask_b32_e64 v217, v213, v32, s[62:63]
	v_add_u32_e32 v32, v175, v59
	s_and_b64 s[62:63], s[48:49], vcc
	v_cndmask_b32_e64 v218, v213, v32, s[62:63]
	v_add_u32_e32 v32, 2, v32
	s_and_b64 s[62:63], s[48:49], s[16:17]
	v_cndmask_b32_e64 v219, v213, v32, s[62:63]
	v_add_u32_e32 v32, v175, v207
	s_and_b64 s[62:63], s[50:51], vcc
	v_cndmask_b32_e64 v220, v213, v32, s[62:63]
	v_add_u32_e32 v32, v175, v205
	s_and_b64 s[62:63], s[50:51], s[16:17]
	v_cndmask_b32_e64 v221, v213, v32, s[62:63]
	buffer_load_ubyte v32, v216, s[72:75], 0 offen
	s_nop 0
	buffer_load_ubyte v214, v214, s[72:75], 0 offen
	s_nop 0
	buffer_load_ubyte v215, v215, s[72:75], 0 offen
	s_nop 0
	buffer_load_ubyte v216, v217, s[72:75], 0 offen
	s_nop 0
	buffer_load_ubyte v217, v218, s[72:75], 0 offen
	s_nop 0
	buffer_load_ubyte v218, v219, s[72:75], 0 offen
	s_nop 0
	buffer_load_ubyte v219, v220, s[72:75], 0 offen
	s_nop 0
	buffer_load_ubyte v220, v221, s[72:75], 0 offen
	v_add_u32_e32 v221, v58, v34
	s_and_b64 s[62:63], s[52:53], vcc
	v_cndmask_b32_e64 v221, v213, v221, s[62:63]
	v_add_u32_e32 v222, v175, v204
	s_and_b64 s[62:63], s[52:53], s[16:17]
	v_cndmask_b32_e64 v222, v213, v222, s[62:63]
	v_add_u32_e32 v223, v175, v201
	s_and_b64 s[62:63], s[54:55], vcc
	v_cndmask_b32_e64 v223, v213, v223, s[62:63]
	v_add_u32_e32 v224, v175, v63
	s_and_b64 s[62:63], s[54:55], s[16:17]
	v_cndmask_b32_e64 v224, v213, v224, s[62:63]
	v_add_u32_e32 v225, v57, v34
	s_and_b64 s[62:63], s[56:57], vcc
	v_cndmask_b32_e64 v225, v213, v225, s[62:63]
	v_add_u32_e32 v226, v175, v203
	s_and_b64 s[62:63], s[56:57], s[16:17]
	v_cndmask_b32_e64 v226, v213, v226, s[62:63]
	v_add_u32_e32 v227, v175, v62
	s_and_b64 s[62:63], s[58:59], vcc
	v_cndmask_b32_e64 v227, v213, v227, s[62:63]
	v_add_u32_e32 v228, v175, v61
	s_and_b64 s[62:63], s[58:59], s[16:17]
	v_cndmask_b32_e64 v228, v213, v228, s[62:63]
	buffer_load_ubyte v221, v221, s[76:79], 0 offen
	s_nop 0
	buffer_load_ubyte v222, v222, s[76:79], 0 offen
	s_nop 0
	buffer_load_ubyte v223, v223, s[76:79], 0 offen
	s_nop 0
	buffer_load_ubyte v224, v224, s[76:79], 0 offen
	s_nop 0
	buffer_load_ubyte v225, v225, s[76:79], 0 offen
	s_nop 0
	buffer_load_ubyte v226, v226, s[76:79], 0 offen
	s_nop 0
	buffer_load_ubyte v227, v227, s[76:79], 0 offen
	s_nop 0
	buffer_load_ubyte v228, v228, s[76:79], 0 offen
	v_add_u32_e32 v229, v56, v34
	s_and_b64 vcc, s[60:61], vcc
	v_cndmask_b32_e32 v229, v213, v229, vcc
	v_add_u32_e32 v230, v175, v202
	s_and_b64 vcc, s[60:61], s[16:17]
	v_cndmask_b32_e32 v230, v213, v230, vcc
	buffer_load_ubyte v229, v229, s[76:79], 0 offen
	s_nop 0
	buffer_load_ubyte v230, v230, s[76:79], 0 offen
	v_add_u32_e32 v231, s89, v48
	ds_read_b128 v[232:235], v231
	ds_read_b128 v[236:239], v231 offset:2048
	ds_read_b128 v[240:243], v231 offset:4096
	ds_read_b128 v[244:247], v231 offset:6144
	v_add_u32_e32 v231, s8, v50
	ds_read_b128 v[248:251], v231
	ds_read_b128 v[134:137], v231 offset:2048
	ds_read_b128 v[138:141], v231 offset:4096
	ds_read_b128 v[142:145], v231 offset:6144
	ds_read_b128 v[178:181], v231 offset:8192
	s_barrier
	s_setprio 1
	v_and_b32_e32 v45, 0xff, v45
	v_and_b32_e32 v47, 0xff, v47
	v_and_b32_e32 v53, 0xff, v53
	v_and_b32_e32 v55, 0xff, v55
	v_and_b32_e32 v198, 0xff, v198
	v_and_b32_e32 v199, 0xff, v199
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[232:235], v[248:251], v[0:3], v45, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[232:235], v[134:137], v[8:11], v45, v53 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[232:235], v[138:141], v[4:7], v45, v55 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[232:235], v[142:145], v[12:15], v45, v198 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[232:235], v[178:181], v[16:19], v45, v199 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v45, 0xff, v193
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[236:239], v[248:251], v[20:23], v45, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[236:239], v[134:137], v[24:27], v45, v53 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[236:239], v[138:141], v[28:31], v45, v55 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[236:239], v[142:145], v[64:67], v45, v198 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[236:239], v[178:181], v[68:71], v45, v199 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v45, 0xff, v195
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[240:243], v[248:251], v[72:75], v45, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[240:243], v[134:137], v[76:79], v45, v53 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[240:243], v[138:141], v[80:83], v45, v55 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[240:243], v[142:145], v[84:87], v45, v198 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[240:243], v[178:181], v[88:91], v45, v199 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v45, 0xff, v196
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[244:247], v[248:251], v[92:95], v45, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[244:247], v[134:137], v[96:99], v45, v53 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[244:247], v[138:141], v[100:103], v45, v55 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[244:247], v[142:145], v[104:107], v45, v198 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[244:247], v[178:181], v[108:111], v45, v199 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v45, s89, v49
	ds_read_b128 v[134:137], v45
	ds_read_b128 v[138:141], v45 offset:2048
	ds_read_b128 v[142:145], v45 offset:4096
	ds_read_b128 v[178:181], v45 offset:6144
	v_add_u32_e32 v45, s8, v51
	ds_read_b128 v[232:235], v45
	ds_read_b128 v[236:239], v45 offset:2048
	ds_read_b128 v[240:243], v45 offset:4096
	ds_read_b128 v[244:247], v45 offset:6144
	ds_read_b128 v[248:251], v45 offset:8192
	s_barrier
	s_setprio 1
	v_and_b32_e32 v44, 0xff, v44
	v_and_b32_e32 v45, 0xff, v46
	v_and_b32_e32 v46, 0xff, v52
	v_and_b32_e32 v47, 0xff, v54
	v_and_b32_e32 v52, 0xff, v197
	v_and_b32_e32 v53, 0xff, v200
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[134:137], v[232:235], v[0:3], v44, v45 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[134:137], v[236:239], v[8:11], v44, v46 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[134:137], v[240:243], v[4:7], v44, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[134:137], v[244:247], v[12:15], v44, v52 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[134:137], v[248:251], v[16:19], v44, v53 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v44, 0xff, v191
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[138:141], v[232:235], v[20:23], v44, v45 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[138:141], v[236:239], v[24:27], v44, v46 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[138:141], v[240:243], v[28:31], v44, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[138:141], v[244:247], v[64:67], v44, v52 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[138:141], v[248:251], v[68:71], v44, v53 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v44, 0xff, v192
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[142:145], v[232:235], v[72:75], v44, v45 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[142:145], v[236:239], v[76:79], v44, v46 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[142:145], v[240:243], v[80:83], v44, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[142:145], v[244:247], v[84:87], v44, v52 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[142:145], v[248:251], v[88:91], v44, v53 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v44, 0xff, v194
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[178:181], v[232:235], v[92:95], v44, v45 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[178:181], v[236:239], v[96:99], v44, v46 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[178:181], v[240:243], v[100:103], v44, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[178:181], v[244:247], v[104:107], v44, v52 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[178:181], v[248:251], v[108:111], v44, v53 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_u32 s4, s4, -1
	s_addc_u32 s5, s5, -1
	s_movk_i32 s16, 0xf800
	s_add_u32 s94, s94, 0x100
	s_mov_b32 s17, -1
	s_mov_b64 s[0:1], 0x800
	s_addc_u32 s95, s95, 0
	v_add_u32_e32 v34, s88, v34
	v_add_u32_e32 v59, s86, v59
	v_add_u32_e32 v60, s86, v60
	v_add_u32_e32 v61, s88, v61
	v_add_u32_e32 v62, s88, v62
	v_add_u32_e32 v63, s88, v63
	v_add_u32_e32 v201, s88, v201
	v_add_u32_e32 v202, s88, v202
	v_add_u32_e32 v203, s88, v203
	v_add_u32_e32 v204, s88, v204
	v_add_u32_e32 v205, s86, v205
	v_add_u32_e32 v206, s86, v206
	v_add_u32_e32 v207, s86, v207
	v_add_u32_e32 v208, s86, v208
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[16:17]
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[0:1]
	v_add_u32_e32 v209, 0x80, v209
	v_add_u32_e32 v210, 0x80, v210
	v_add_u32_e32 v211, 0x80, v211
	v_add_u32_e32 v212, 0x80, v212
	s_cmp_lg_u64 s[4:5], 0
	s_mov_b32 s9, s8
	s_mov_b32 s8, s89
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v200, v230
	v_mov_b32_e32 v199, v229
	v_mov_b32_e32 v197, v228
	v_mov_b32_e32 v198, v227
	v_mov_b32_e32 v54, v226
	v_mov_b32_e32 v55, v225
	v_mov_b32_e32 v52, v224
	v_mov_b32_e32 v53, v223
	v_mov_b32_e32 v46, v222
	v_mov_b32_e32 v47, v221
	v_mov_b32_e32 v194, v220
	v_mov_b32_e32 v196, v219
	v_mov_b32_e32 v192, v218
	v_mov_b32_e32 v195, v217
	v_mov_b32_e32 v191, v216
	v_mov_b32_e32 v193, v215
	v_mov_b32_e32 v44, v214
	v_mov_b32_e32 v45, v32
	s_cbranch_scc1 .LBB0_8
	scratch_load_dwordx2 v[240:241], off, off offset:24
	scratch_load_dwordx2 v[242:243], off, off offset:32
	scratch_load_dwordx2 v[244:245], off, off offset:40
	scratch_load_dwordx2 v[248:249], off, off offset:48
	scratch_load_dwordx2 v[250:251], off, off offset:56
	v_readlane_b32 s12, v255, 16
	v_readlane_b32 s82, v255, 14
	v_readlane_b32 s62, v255, 1
	v_readlane_b32 s96, v255, 12
	v_readlane_b32 s26, v255, 22
	v_mov_b32_e32 v200, v230
	v_mov_b32_e32 v199, v229
	v_mov_b32_e32 v197, v228
	v_mov_b32_e32 v198, v227
	v_mov_b32_e32 v54, v226
	v_mov_b32_e32 v55, v225
	v_mov_b32_e32 v52, v224
	v_mov_b32_e32 v53, v223
	v_mov_b32_e32 v46, v222
	v_mov_b32_e32 v47, v221
	v_mov_b32_e32 v194, v220
	v_mov_b32_e32 v196, v219
	v_mov_b32_e32 v192, v218
	v_mov_b32_e32 v195, v217
	v_mov_b32_e32 v191, v216
	v_mov_b32_e32 v193, v215
	v_mov_b32_e32 v44, v214
	v_mov_b32_e32 v45, v32
	v_readlane_b32 s13, v255, 17
	v_readlane_b32 s83, v255, 15
	v_readlane_b32 s63, v255, 2
	v_readlane_b32 s97, v255, 13
	v_readlane_b32 s94, v255, 24
	v_readlane_b32 s81, v255, 18
	v_readlane_b32 s18, v254, 12
	v_readlane_b32 s27, v255, 23
	v_lshlrev_b32_e32 v246, 2, v182
	s_branch .LBB0_12
.LBB0_10:
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, v64
	v_mov_b32_e32 v66, v64
	v_mov_b32_e32 v67, v64
	v_mov_b64_e32 v[70:71], v[66:67]
	v_mov_b64_e32 v[74:75], v[66:67]
	v_mov_b64_e32 v[78:79], v[66:67]
	v_mov_b64_e32 v[0:1], v[64:65]
	v_mov_b64_e32 v[4:5], v[64:65]
	v_mov_b64_e32 v[8:9], v[64:65]
	v_mov_b64_e32 v[12:13], v[64:65]
	v_mov_b64_e32 v[16:17], v[64:65]
	v_mov_b64_e32 v[20:21], v[64:65]
	v_mov_b64_e32 v[24:25], v[64:65]
	v_mov_b64_e32 v[28:29], v[64:65]
	v_mov_b64_e32 v[32:33], v[64:65]
	v_mov_b64_e32 v[36:37], v[64:65]
	v_mov_b64_e32 v[40:41], v[64:65]
	v_mov_b64_e32 v[44:45], v[64:65]
	v_mov_b64_e32 v[48:49], v[64:65]
	v_mov_b64_e32 v[52:53], v[64:65]
	v_mov_b64_e32 v[56:57], v[64:65]
	v_mov_b64_e32 v[60:61], v[64:65]
	s_mov_b64 s[68:69], s[0:1]
	s_mov_b64 s[64:65], s[2:3]
	v_mov_b64_e32 v[68:69], v[64:65]
	v_mov_b64_e32 v[72:73], v[64:65]
	v_mov_b64_e32 v[76:77], v[64:65]
	v_mov_b64_e32 v[2:3], v[66:67]
	v_mov_b64_e32 v[6:7], v[66:67]
	v_mov_b64_e32 v[10:11], v[66:67]
	v_mov_b64_e32 v[14:15], v[66:67]
	v_mov_b64_e32 v[18:19], v[66:67]
	v_mov_b64_e32 v[22:23], v[66:67]
	v_mov_b64_e32 v[26:27], v[66:67]
	v_mov_b64_e32 v[30:31], v[66:67]
	v_mov_b64_e32 v[34:35], v[66:67]
	v_mov_b64_e32 v[38:39], v[66:67]
	v_mov_b64_e32 v[42:43], v[66:67]
	v_mov_b64_e32 v[46:47], v[66:67]
	v_mov_b64_e32 v[50:51], v[66:67]
	v_mov_b64_e32 v[54:55], v[66:67]
	v_mov_b64_e32 v[58:59], v[66:67]
	v_mov_b64_e32 v[62:63], v[66:67]
	s_branch .LBB0_15
.LBB0_11:
	v_mov_b32_e32 v3, 0
	v_readlane_b32 s62, v255, 1
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v0, v3
	v_mov_b32_e32 v11, v3
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v9, v3
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v14, v3
	v_mov_b32_e32 v13, v3
	v_mov_b32_e32 v12, v3
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v17, v3
	v_mov_b32_e32 v16, v3
	v_mov_b32_e32 v23, v3
	v_mov_b32_e32 v22, v3
	v_mov_b32_e32 v21, v3
	v_mov_b32_e32 v20, v3
	v_mov_b32_e32 v27, v3
	v_mov_b32_e32 v26, v3
	v_mov_b32_e32 v25, v3
	v_mov_b32_e32 v24, v3
	v_mov_b32_e32 v31, v3
	v_mov_b32_e32 v30, v3
	v_mov_b32_e32 v29, v3
	v_mov_b32_e32 v28, v3
	v_mov_b32_e32 v67, v3
	v_mov_b32_e32 v66, v3
	v_mov_b32_e32 v65, v3
	v_mov_b32_e32 v64, v3
	v_mov_b32_e32 v71, v3
	v_mov_b32_e32 v70, v3
	v_mov_b32_e32 v69, v3
	v_mov_b32_e32 v68, v3
	v_mov_b32_e32 v75, v3
	v_mov_b32_e32 v74, v3
	v_mov_b32_e32 v73, v3
	v_mov_b32_e32 v72, v3
	v_mov_b32_e32 v79, v3
	v_mov_b32_e32 v78, v3
	v_mov_b32_e32 v77, v3
	v_mov_b32_e32 v76, v3
	v_mov_b32_e32 v83, v3
	v_mov_b32_e32 v82, v3
	v_mov_b32_e32 v81, v3
	v_mov_b32_e32 v80, v3
	v_mov_b32_e32 v87, v3
	v_mov_b32_e32 v86, v3
	v_mov_b32_e32 v85, v3
	v_mov_b32_e32 v84, v3
	v_mov_b32_e32 v91, v3
	v_mov_b32_e32 v90, v3
	v_mov_b32_e32 v89, v3
	v_mov_b32_e32 v88, v3
	v_mov_b32_e32 v95, v3
	v_mov_b32_e32 v94, v3
	v_mov_b32_e32 v93, v3
	v_mov_b32_e32 v92, v3
	v_mov_b32_e32 v99, v3
	v_mov_b32_e32 v98, v3
	v_mov_b32_e32 v97, v3
	v_mov_b32_e32 v96, v3
	v_mov_b32_e32 v103, v3
	v_mov_b32_e32 v102, v3
	v_mov_b32_e32 v101, v3
	v_mov_b32_e32 v100, v3
	v_mov_b32_e32 v107, v3
	v_mov_b32_e32 v106, v3
	v_mov_b32_e32 v105, v3
	v_mov_b32_e32 v104, v3
	v_mov_b32_e32 v111, v3
	v_mov_b32_e32 v110, v3
	v_mov_b32_e32 v109, v3
	v_mov_b32_e32 v108, v3
	v_readlane_b32 s63, v255, 2
	s_mov_b64 s[26:27], s[8:9]
.LBB0_12:
	v_readlane_b32 s0, v255, 10
	v_readlane_b32 s1, v255, 11
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_14
	s_barrier
.LBB0_14:
	v_add_u32_e32 v113, s80, v48
	s_waitcnt vmcnt(19)
	s_barrier
	v_add_u32_e32 v32, s93, v50
	ds_read_b128 v[36:39], v113
	ds_read_b128 v[134:137], v32
	ds_read_b128 v[138:141], v32 offset:2048
	s_waitcnt vmcnt(17)
	v_and_b32_e32 v34, 0xff, v45
	s_waitcnt vmcnt(9)
	v_and_b32_e32 v202, 0xff, v47
	v_add_u32_e32 v201, s80, v49
	v_add_u32_e32 v33, s93, v51
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[36:39], v[134:137], v[0:3], v34, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v201
	ds_read_b128 v[142:145], v33
	ds_read_b128 v[178:181], v32 offset:8192
	ds_read_b128 v[208:211], v113 offset:2048
	v_and_b32_e32 v35, 0xff, v44
	s_waitcnt vmcnt(8)
	v_and_b32_e32 v203, 0xff, v46
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v204, 0xff, v53
	ds_read_b128 v[216:219], v33 offset:2048
	ds_read_b128 v[220:223], v33 offset:4096
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[40:43], v[142:145], v[0:3], v35, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[224:227], v32 offset:4096
	ds_read_b128 v[228:231], v32 offset:6144
	s_waitcnt vmcnt(6)
	v_and_b32_e32 v205, 0xff, v52
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v206, 0xff, v55
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[36:39], v[138:141], v[8:11], v34, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(4)
	v_and_b32_e32 v207, 0xff, v54
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v198, 0xff, v198
	ds_read_b128 v[232:235], v33 offset:6144
	ds_read_b128 v[236:239], v33 offset:8192
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[40:43], v[216:219], v[0:3], v35, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[212:215], v201 offset:2048
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v197, 0xff, v197
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v199, 0xff, v199
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[36:39], v[224:227], v[4:7], v34, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v200, 0xff, v200
	v_readlane_b32 s8, v255, 3
	v_readlane_b32 s50, v254, 58
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[40:43], v[220:223], v[0:3], v35, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v4, 0xff, v193
	v_and_b32_e32 v5, 0xff, v191
	v_readlane_b32 s22, v254, 7
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[36:39], v[228:231], v[12:15], v34, v198 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_readlane_b32 s52, v254, 60
	v_readlane_b32 s60, v255, 5
	v_readlane_b32 s14, v255, 8
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[40:43], v[232:235], v[0:3], v35, v197 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b64 s[66:67], s[70:71]
	v_readlane_b32 s20, v254, 11
	v_readlane_b32 s9, v255, 4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[36:39], v[178:181], v[16:19], v34, v199 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_readlane_b32 s51, v254, 59
	v_readlane_b32 s23, v254, 8
	v_readlane_b32 s55, v254, 25
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[40:43], v[236:239], v[0:3], v35, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_readlane_b32 s74, v254, 29
	v_readlane_b32 s75, v254, 33
	v_readlane_b32 s76, v254, 37
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[208:211], v[134:137], v[20:23], v4, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_readlane_b32 s77, v254, 41
	v_readlane_b32 s78, v254, 45
	v_readlane_b32 s79, v254, 49
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[212:215], v[142:145], v[0:3], v5, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_readlane_b32 s80, v254, 53
	v_readlane_b32 s53, v254, 61
	v_readlane_b32 s33, v254, 62
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[208:211], v[138:141], v[24:27], v4, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_readlane_b32 s61, v255, 6
	v_readlane_b32 s54, v255, 7
	v_readlane_b32 s15, v255, 9
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[212:215], v[216:219], v[0:3], v5, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[208:211], v[224:227], v[28:31], v4, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[212:215], v[220:223], v[0:3], v5, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[208:211], v[228:231], v[64:67], v4, v198 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[212:215], v[232:235], v[0:3], v5, v197 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[208:211], v[178:181], v[68:71], v4, v199 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	ds_read_b128 v[64:67], v113 offset:4096
	ds_read_b128 v[208:211], v113 offset:6144
	v_and_b32_e32 v4, 0xff, v195
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[212:215], v[236:239], v[0:3], v5, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[68:71], v201 offset:4096
	ds_read_b128 v[212:215], v201 offset:6144
	v_and_b32_e32 v5, 0xff, v192
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[64:67], v[134:137], v[72:75], v4, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[68:71], v[142:145], v[0:3], v5, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[64:67], v[138:141], v[76:79], v4, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[68:71], v[216:219], v[0:3], v5, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[64:67], v[224:227], v[80:83], v4, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[68:71], v[220:223], v[0:3], v5, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 1
	v_and_b32_e32 v80, 0xff, v196
	v_and_b32_e32 v81, 0xff, v194
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[64:67], v[228:231], v[84:87], v4, v198 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	scratch_load_dwordx2 v[82:83], off, off
	s_nop 1
	scratch_load_dwordx2 v[84:85], off, off offset:8
	scratch_load_dwordx2 v[86:87], off, off offset:16
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[68:71], v[232:235], v[0:3], v5, v197 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[64:67], v[178:181], v[88:91], v4, v199 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[208:211], v[138:141], v[96:99], v80, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[212:215], v[216:219], v[64:67], v81, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[208:211], v[224:227], v[100:103], v80, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[212:215], v[220:223], v[64:67], v81, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[208:211], v[228:231], v[104:107], v80, v198 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[68:71], v[236:239], v[0:3], v5, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[208:211], v[134:137], v[92:95], v80, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[212:215], v[232:235], v[64:67], v81, v197 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 1
	v_mov_b32_e32 v92, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[208:211], v[178:181], v[108:111], v80, v199 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[212:215], v[142:145], v[0:3], v81, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[212:215], v[236:239], v[64:67], v81, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
.LBB0_15:
	s_load_dwordx2 s[72:73], s[22:23], 0x60
	v_cmp_gt_i64_e64 s[0:1], s[96:97], 1
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s11, s27, 0
	s_cselect_b32 s10, s26, 0
	v_mov_b64_e32 v[80:81], s[96:97]
	v_cmp_lt_i64_e32 vcc, s[10:11], v[80:81]
	v_add_u32_e32 v100, 48, v115
	s_cbranch_vccnz .LBB0_17
	s_mov_b64 s[0:1], 0
	s_branch .LBB0_18
.LBB0_17:
	s_mov_b64 s[0:1], -1
.LBB0_18:
	v_cmp_gt_i64_e64 s[18:19], s[14:15], v[162:163]
	v_cmp_gt_i64_e64 s[16:17], s[14:15], v[164:165]
	v_cmp_gt_i64_e64 s[6:7], s[14:15], v[166:167]
	v_cmp_gt_i64_e64 s[2:3], s[14:15], v[168:169]
	s_andn2_b64 vcc, exec, s[0:1]
	v_cmp_gt_i64_e64 s[0:1], s[14:15], v[170:171]
	s_cbranch_vccnz .LBB0_22
	v_cmp_gt_i32_e32 vcc, 0, v121
	v_bfe_u32 v95, v115, 4, 1
	s_waitcnt vmcnt(0)
	v_cmp_gt_i64_e64 s[44:45], s[14:15], v[86:87]
	v_cndmask_b32_e32 v80, v150, v186, vcc
	v_ashrrev_i32_e32 v81, 31, v80
	v_add_u32_sdwa v80, v80, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v80, 8, v80
	v_xor_b32_e32 v93, v80, v187
	v_or_b32_e32 v80, v95, v183
	v_add_u32_e32 v86, v80, v151
	v_or_b32_e32 v80, 2, v86
	v_sub_u32_e32 v81, -3, v86
	v_cmp_gt_i32_e32 vcc, -2, v86
	v_cmp_gt_i64_e64 s[4:5], s[12:13], v[122:123]
	v_mov_b32_e32 v87, 5
	v_cndmask_b32_e32 v80, v80, v81, vcc
	v_ashrrev_i32_e32 v81, 31, v80
	v_add_u32_sdwa v80, v80, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_lshrrev_b32_e32 v80, 8, v80
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_xor_b32_e32 v80, v80, v81
	v_lshlrev_b32_e32 v104, 8, v80
	v_mov_b32_e32 v80, 10
	v_lshrrev_b16_sdwa v80, v80, sext(v119) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_and_b32_e32 v80, 31, v80
	v_writelane_b32 v254, s4, 9
	v_add_u16_e32 v80, v119, v80
	v_ashrrev_i16_sdwa v80, v87, sext(v80) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
	v_writelane_b32 v254, s5, 10
	v_cmp_gt_i64_e64 s[4:5], s[12:13], v[124:125]
	s_add_u32 s56, s50, s52
	v_bfe_i32 v101, v80, 0, 16
	v_writelane_b32 v254, s4, 15
	v_mad_i32_i24 v80, v101, -2, v190
	v_cmp_gt_i64_e64 s[38:39], s[14:15], v[82:83]
	v_writelane_b32 v254, s5, 16
	s_addc_u32 s4, s51, s53
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s33, 16
	v_add_u32_e32 v81, 1, v80
	v_sub_u32_e32 v82, 0xfffe, v80
	v_cmp_gt_i32_e32 vcc, -1, v80
	s_or_b32 s57, s4, s5
	s_add_u32 s60, s8, s60
	v_cndmask_b32_e32 v81, v81, v82, vcc
	v_ashrrev_i16_e32 v82, 15, v81
	s_addc_u32 s8, s9, s61
	v_add_u16_sdwa v81, v81, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
	v_cndmask_b32_e64 v82, 0, -1, vcc
	s_and_b32 s8, s8, 0xffff
	s_lshl_b32 s9, s54, 16
	v_xor_b32_sdwa v81, sext(v81), v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD
	s_or_b32 s61, s8, s9
	v_bfe_i32 v102, v81, 0, 16
	v_add_u32_e32 v81, 3, v80
	v_sub_u32_e32 v82, 0xfffc, v80
	v_cmp_gt_i32_e32 vcc, -3, v80
	v_readlane_b32 s8, v254, 3
	v_cmp_gt_i64_e64 s[40:41], s[14:15], v[84:85]
	v_cndmask_b32_e32 v80, v81, v82, vcc
	v_bitop3_b32 v84, v116, v173, 7 bitop3:0x6c
	s_lshl_b32 s33, s8, 7
	v_readlane_b32 s8, v254, 4
	v_sub_u32_e32 v90, v118, v246
	v_ashrrev_i16_e32 v81, 15, v80
	s_lshl_b32 s54, s8, 7
	v_lshlrev_b32_e32 v85, 8, v84
	s_lshl_b32 s8, s94, 7
	v_lshl_add_u32 v90, v84, 2, v90
	v_add_u16_sdwa v80, v80, v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_add3_u32 v82, s8, v175, v85
	v_ashrrev_i32_e32 v91, 31, v90
	v_readlane_b32 s8, v254, 0
	v_xor_b32_sdwa v80, sext(v80), v81 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD
	v_xor_b32_e32 v90, v91, v90
	s_lshl_b32 s89, s8, 7
	v_readlane_b32 s8, v254, 6
	v_bfe_i32 v105, v80, 0, 16
	v_lshlrev_b32_e32 v80, 7, v174
	v_lshlrev_b32_e32 v81, 11, v177
	v_ashrrev_i32_e32 v98, 31, v90
	s_lshl_b32 s99, s80, 7
	s_lshl_b32 s80, s8, 7
	v_readlane_b32 s8, v254, 5
	v_sub_u32_e32 v89, v80, v81
	v_lshrrev_b32_e32 v98, 29, v98
	s_add_i32 s8, s8, 0x22000
	v_cmp_gt_i64_e64 s[24:25], s[12:13], v[126:127]
	v_add_u32_e32 v90, v90, v98
	v_add_u32_e32 v126, s8, v89
	v_lshl_add_u32 v89, v172, 13, v89
	v_lshrrev_b32_e32 v90, 3, v90
	v_add_u32_e32 v127, 0x10000, v89
	v_ashrrev_i32_e32 v89, 31, v86
	v_xor_b32_e32 v90, v90, v91
	v_xor_b32_e32 v86, v89, v86
	v_lshlrev_b32_e32 v98, 9, v90
	v_ashrrev_i32_e32 v90, 31, v86
	v_add_u32_sdwa v86, v86, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v86, 8, v86
	v_xor_b32_e32 v86, v86, v89
	v_lshrrev_b32_e32 v89, 26, v115
	v_and_b32_e32 v89, 32, v89
	s_movk_i32 s8, 0xe0
	v_bitop3_b32 v107, v115, v89, s8 bitop3:0x6c
	v_or_b32_e32 v89, v95, v121
	v_or_b32_e32 v90, 2, v89
	v_sub_u32_e32 v91, -3, v89
	v_cmp_gt_i32_e32 vcc, -2, v89
	v_mul_i32_i24_e32 v88, -2, v101
	v_cmp_gt_i64_e64 s[26:27], s[12:13], v[128:129]
	v_cndmask_b32_e32 v89, v90, v91, vcc
	v_ashrrev_i32_e32 v90, 31, v89
	v_add_u32_sdwa v89, v89, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v89, 8, v89
	v_cndmask_b32_e64 v90, 0, -1, vcc
	v_xor_b32_e32 v108, v89, v90
	v_ashrrev_i16_e32 v89, 15, v100
	v_lshrrev_b16_e32 v89, 11, v89
	v_add_u16_e32 v89, v100, v89
	v_ashrrev_i16_e32 v89, 5, v89
	v_lshlrev_b32_sdwa v129, v87, sext(v89) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add3_u32 v87, v188, v88, v189
	v_add_u32_e32 v88, 3, v87
	v_sub_u32_e32 v89, 0xfffc, v87
	v_cmp_gt_i32_e32 vcc, -3, v87
	v_cmp_gt_i64_e64 s[28:29], s[14:15], v[130:131]
	v_add3_u32 v91, v107, v86, s10
	v_cndmask_b32_e32 v87, v88, v89, vcc
	v_ashrrev_i16_e32 v88, 15, v87
	v_add_u16_sdwa v87, v87, v88 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
	v_cndmask_b32_e64 v88, 0, -1, vcc
	v_xor_b32_sdwa v87, sext(v87), v88 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD
	v_bfe_i32 v130, v87, 0, 16
	v_add3_u32 v87, v151, v183, v95
	v_add_u32_e32 v86, 64, v91
	v_and_b32_e32 v90, 0xff, v87
	v_mad_u64_u32 v[86:87], s[8:9], s88, v86, v[90:91]
	v_add_u32_e32 v87, 32, v91
	v_lshlrev_b32_e32 v128, 5, v101
	v_mad_u64_u32 v[88:89], s[8:9], s88, v87, v[90:91]
	v_mad_u64_u32 v[90:91], s[8:9], s88, v91, v[90:91]
	v_lshlrev_b32_e32 v103, 8, v102
	v_add3_u32 v87, v129, v130, s10
	v_add3_u32 v89, v129, v102, s10
	v_add3_u32 v91, v128, v105, s10
	v_add3_u32 v102, v128, v102, s10
	v_mad_i32_i24 v101, v101, -2, v121
	v_mul_lo_u32 v87, s88, v87
	v_mul_lo_u32 v89, s88, v89
	v_mul_lo_u32 v91, s88, v91
	v_mul_lo_u32 v102, s88, v102
	v_add3_u32 v87, v101, v87, v92
	v_add3_u32 v89, v101, v89, v92
	v_add3_u32 v91, v101, v91, v92
	v_add3_u32 v92, v101, v102, v92
	v_sub_u32_e32 v92, v92, v103
	v_add_u32_e32 v101, 1, v92
	v_add3_u32 v92, v107, v108, s10
	v_sub_u32_e32 v89, v89, v103
	v_add_u32_e32 v102, 64, v92
	v_add_u32_e32 v103, 32, v92
	v_mul_lo_u32 v92, s88, v92
	v_mul_lo_u32 v102, s88, v102
	v_mul_lo_u32 v103, s88, v103
	v_add3_u32 v92, v121, v92, v95
	v_add_u32_e32 v107, s10, v117
	v_add3_u32 v102, v121, v102, v95
	v_add3_u32 v103, v121, v103, v95
	v_sub_u32_e32 v92, v92, v104
	v_add_u32_e32 v95, v107, v93
	v_lshlrev_b32_e32 v94, 8, v93
	v_sub_u32_e32 v102, v102, v104
	v_sub_u32_e32 v103, v103, v104
	v_add_u32_e32 v104, 2, v92
	v_mad_u64_u32 v[92:93], s[8:9], s86, v95, v[150:151]
	v_lshlrev_b32_e32 v106, 8, v105
	v_sub_u32_e32 v105, v92, v94
	v_add_u32_e32 v92, 32, v95
	v_mad_u64_u32 v[92:93], s[8:9], s86, v92, v[150:151]
	v_sub_u32_e32 v87, v87, v106
	v_sub_u32_e32 v91, v91, v106
	v_sub_u32_e32 v106, v92, v94
	v_add_u32_e32 v94, v107, v185
	v_mad_u64_u32 v[92:93], s[8:9], s86, v94, v[146:147]
	v_add_u32_e32 v93, 32, v94
	v_mov_b32_e32 v83, 0
	v_mad_u64_u32 v[94:95], s[8:9], s86, v93, v[146:147]
	v_and_b32_e32 v80, 7, v174
	v_lshlrev_b32_e32 v96, 4, v173
	v_mov_b32_e32 v97, v83
	v_and_b32_e32 v85, 0xfffffe00, v85
	v_mul_i32_i24_e32 v122, -16, v116
	s_lshl_b64 s[8:9], s[10:11], 11
	v_bitop3_b32 v80, v147, v80, 4 bitop3:0x36
	v_sub_u32_e32 v110, 0, v85
	v_ashrrev_i32_e32 v123, 31, v122
	v_lshl_add_u64 v[96:97], s[8:9], 0, v[96:97]
	v_writelane_b32 v255, s82, 14
	v_lshlrev_b32_e32 v124, 4, v80
	v_bitop3_b32 v80, v147, v174, 7 bitop3:0x78
	v_ashrrev_i32_e32 v111, 31, v110
	v_lshl_add_u64 v[96:97], v[96:97], 0, v[122:123]
	v_writelane_b32 v255, s83, 15
	v_lshlrev_b32_e32 v125, 4, v80
	v_bitop3_b32 v80, v176, 7, v174 bitop3:0x48
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshlrev_b32_e32 v85, 8, v173
	v_lshl_add_u64 v[96:97], v[96:97], 0, v[110:111]
	v_cmp_gt_i64_e64 s[48:49], s[12:13], v[154:155]
	v_cmp_gt_i64_e64 s[4:5], s[12:13], v[156:157]
	v_cmp_gt_i64_e64 s[50:51], s[12:13], v[158:159]
	v_writelane_b32 v255, s12, 16
	v_lshlrev_b32_e32 v109, 4, v80
	v_sub_u32_e32 v84, v84, v182
	v_sub_u32_e32 v85, v114, v85
	v_add_u32_e32 v95, 32, v107
	v_lshl_add_u64 v[96:97], v[96:97], 0, v[98:99]
	v_writelane_b32 v255, s13, 17
	v_cmp_gt_i64_e64 s[52:53], s[12:13], v[160:161]
	s_lshl_b32 s12, s20, 7
	s_lshl_b32 s13, s81, 7
	v_lshl_add_u32 v84, v84, 5, v85
	s_lshl_b32 s92, s55, 7
	s_lshl_b32 s93, s74, 7
	s_lshl_b32 s94, s75, 7
	s_lshl_b32 s95, s76, 7
	s_mov_b64 s[20:21], s[96:97]
	s_lshl_b32 s96, s77, 7
	s_lshl_b32 s97, s78, 7
	s_lshl_b32 s98, s79, 7
	s_lshl_b64 s[74:75], s[90:91], 4
	v_mul_lo_u32 v108, s86, v107
	v_mul_lo_u32 v107, s86, v95
	v_lshl_add_u64 v[96:97], v[96:97], 0, v[82:83]
	v_lshl_or_b32 v82, s10, 7, v109
	s_ashr_i32 s78, s91, 31
	v_cmp_gt_i64_e64 s[30:31], s[14:15], v[132:133]
	v_cmp_gt_i64_e64 s[34:35], s[14:15], v[248:249]
	v_cmp_gt_i64_e64 s[36:37], s[14:15], v[250:251]
	v_cmp_gt_i64_e64 s[42:43], s[14:15], v[252:253]
	v_cmp_gt_i64_e64 s[46:47], s[14:15], v[148:149]
	v_cmp_gt_i64_e64 s[14:15], s[14:15], v[152:153]
	s_mov_b32 s59, 0x27000
	s_mov_b32 s58, 0x7ffffffe
	v_lshlrev_b32_e32 v80, 5, v80
	v_mov_b32_e32 v81, v83
	v_ashrrev_i32_e32 v85, 31, v84
	v_mov_b32_e32 v113, v83
	v_add_u32_e32 v87, 3, v87
	v_add_u32_e32 v89, 1, v89
	v_add_u32_e32 v91, 3, v91
	v_add_u32_e32 v102, 2, v102
	v_add_u32_e32 v103, 2, v103
	v_add_u32_e32 v93, v184, v108
	v_add_u32_e32 v95, v121, v107
	v_add_u32_e32 v107, v184, v107
	v_add_u32_e32 v108, v121, v108
	s_lshl_b64 s[76:77], s[10:11], 8
	v_add_u32_e32 v109, v82, v120
	v_add_u32_e32 v110, v82, v244
	v_add_u32_e32 v111, v82, v242
	v_add_u32_e32 v114, v82, v240
	v_bfrev_b32_e32 v116, -2
	s_add_i32 s12, s12, 0x10000
	s_add_i32 s13, s13, 0x10000
	s_add_i32 s33, s33, 0x10000
	s_add_i32 s75, s54, 0x10000
	s_mov_b32 s79, s78
	s_add_i32 s89, s89, 0x22000
	s_add_i32 s92, s92, 0x22000
	s_add_i32 s93, s93, 0x22000
	s_add_i32 s94, s94, 0x22000
	s_add_i32 s95, s95, 0x22000
	s_add_i32 s96, s96, 0x22000
	s_add_i32 s97, s97, 0x22000
	s_add_i32 s98, s98, 0x22000
	s_add_i32 s99, s99, 0x22000
	s_add_i32 s80, s80, 0x22000
	v_add_u32_e32 v118, v126, v125
	v_add_u32_e32 v120, v126, v124
	v_add_u32_e32 v121, v127, v125
	v_add_u32_e32 v122, v127, v124
	v_mov_b64_e32 v[98:99], s[20:21]
	s_mov_b32 s81, 0
	s_mov_b32 s82, 0
	s_mov_b64 s[22:23], s[62:63]
.LBB0_20:
	v_lshl_add_u64 v[124:125], v[80:81], 0, s[76:77]
	v_readlane_b32 s8, v254, 9
	v_cmp_gt_i64_e32 vcc, s[84:85], v[124:125]
	v_readlane_b32 s9, v254, 10
	s_and_b64 s[8:9], s[8:9], vcc
	s_mov_b32 m0, s12
	v_cndmask_b32_e64 v123, v116, v114, s[8:9]
	v_readlane_b32 s8, v254, 15
	v_readlane_b32 s9, v254, 16
	s_and_b64 s[8:9], s[8:9], vcc
	buffer_load_dwordx4 v123, s[68:71], 0 offen lds
	v_cndmask_b32_e64 v138, v116, v111, s[8:9]
	s_and_b64 s[8:9], s[24:25], vcc
	s_and_b64 vcc, s[26:27], vcc
	v_cndmask_b32_e64 v139, v116, v110, s[8:9]
	s_add_u32 s8, s90, s78
	s_addc_u32 s9, s91, s78
	s_xor_b64 s[62:63], s[8:9], s[78:79]
	v_cvt_f32_u32_e32 v82, s62
	v_cvt_f32_u32_e32 v124, s63
	s_sub_u32 s8, 0, s62
	v_cndmask_b32_e32 v140, v116, v109, vcc
	s_subb_u32 s9, 0, s63
	v_fmac_f32_e32 v82, 0x4f800000, v124
	v_rcp_f32_e32 v82, v82
	s_mov_b32 m0, s13
	v_add_u32_e32 v109, 0x80, v109
	buffer_load_dwordx4 v138, s[68:71], 0 offen lds
	v_mul_f32_e32 v82, 0x5f7ffffc, v82
	v_mul_f32_e32 v124, 0x2f800000, v82
	v_trunc_f32_e32 v124, v124
	v_fmac_f32_e32 v82, 0xcf800000, v124
	v_cvt_u32_f32_e32 v124, v124
	v_cvt_u32_f32_e32 v82, v82
	s_mov_b32 m0, s33
	v_add_u32_e32 v110, 0x80, v110
	v_readfirstlane_b32 s20, v124
	v_readfirstlane_b32 s87, v82
	s_mul_i32 s83, s8, s20
	s_mul_hi_u32 vcc_lo, s8, s87
	s_add_i32 s83, vcc_lo, s83
	s_mul_i32 vcc_lo, s9, s87
	s_add_i32 s83, s83, vcc_lo
	s_mul_i32 vcc_lo, s8, s87
	s_mul_hi_u32 vcc_hi, s87, vcc_lo
	s_mul_i32 s55, s87, s83
	s_add_u32 s55, vcc_hi, s55
	s_mul_hi_u32 vcc_hi, s87, s83
	s_addc_u32 vcc_hi, 0, vcc_hi
	s_mul_i32 s54, s20, vcc_lo
	s_add_u32 s21, s55, s54
	s_mul_hi_u32 s21, s20, vcc_lo
	s_addc_u32 s21, vcc_hi, s21
	s_mul_hi_u32 s54, s20, s83
	s_mul_i32 s55, s20, s83
	s_addc_u32 s54, s54, 0
	s_add_u32 s21, s21, s55
	s_addc_u32 s54, 0, s54
	s_add_u32 s21, s87, s21
	s_addc_u32 s20, s20, s54
	s_mul_hi_u32 s54, s8, s21
	s_mul_i32 s55, s8, s20
	s_add_i32 s54, s54, s55
	s_mul_i32 s9, s9, s21
	s_add_i32 s54, s54, s9
	s_mul_i32 s8, s8, s21
	s_mul_hi_u32 s9, s21, s8
	s_mul_i32 s55, s21, s54
	s_add_u32 s9, s9, s55
	s_mul_hi_u32 s55, s21, s54
	s_addc_u32 s55, 0, s55
	s_mul_i32 s83, s20, s8
	s_add_u32 s9, s9, s83
	s_mul_hi_u32 s8, s20, s8
	s_addc_u32 s8, s55, s8
	s_mul_hi_u32 s9, s20, s54
	s_mul_i32 s54, s20, s54
	s_addc_u32 s9, s9, 0
	s_add_u32 s8, s8, s54
	s_addc_u32 s9, 0, s9
	v_lshl_add_u64 v[124:125], v[112:113], 0, s[76:77]
	s_add_u32 s21, s21, s8
	v_cmp_gt_i64_e32 vcc, s[84:85], v[124:125]
	s_addc_u32 s20, s20, s9
	v_add_u32_e32 v82, s81, v108
	s_and_b64 s[8:9], s[48:49], vcc
	v_cndmask_b32_e64 v141, v116, v82, s[8:9]
	s_mov_b64 s[8:9], 0x80
	v_lshl_add_u64 v[124:125], v[124:125], 0, s[8:9]
	v_cmp_gt_i64_e64 s[8:9], s[84:85], v[124:125]
	v_add_u32_e32 v82, s81, v93
	s_and_b64 s[54:55], s[48:49], s[8:9]
	v_cndmask_b32_e64 v142, v116, v82, s[54:55]
	v_add_u32_e32 v82, s81, v92
	s_and_b64 s[54:55], s[4:5], vcc
	v_cndmask_b32_e64 v143, v116, v82, s[54:55]
	v_add_u32_e32 v82, s81, v105
	s_and_b64 s[54:55], s[4:5], s[8:9]
	v_cndmask_b32_e64 v144, v116, v82, s[54:55]
	v_add_u32_e32 v82, s81, v95
	s_and_b64 s[54:55], s[50:51], vcc
	v_cndmask_b32_e64 v145, v116, v82, s[54:55]
	v_add_u32_e32 v82, s81, v107
	s_and_b64 s[54:55], s[50:51], s[8:9]
	v_cndmask_b32_e64 v146, v116, v82, s[54:55]
	v_add_u32_e32 v82, s81, v94
	s_and_b64 s[54:55], s[52:53], vcc
	v_cndmask_b32_e64 v148, v116, v82, s[54:55]
	v_add_u32_e32 v82, s81, v106
	s_and_b64 s[54:55], s[52:53], s[8:9]
	v_cndmask_b32_e64 v149, v116, v82, s[54:55]
	v_add_u32_e32 v82, s82, v90
	s_and_b64 s[54:55], s[18:19], vcc
	v_cndmask_b32_e64 v150, v116, v82, s[54:55]
	v_add_u32_e32 v82, s82, v104
	s_and_b64 s[54:55], s[18:19], s[8:9]
	v_cndmask_b32_e64 v151, v116, v82, s[54:55]
	v_add_u32_e32 v82, s82, v101
	s_and_b64 s[54:55], s[16:17], vcc
	v_cndmask_b32_e64 v152, v116, v82, s[54:55]
	v_add_u32_e32 v82, s82, v91
	s_and_b64 s[54:55], s[16:17], s[8:9]
	v_cndmask_b32_e64 v153, v116, v82, s[54:55]
	v_add_u32_e32 v82, s82, v88
	s_and_b64 s[54:55], s[6:7], vcc
	v_ashrrev_i32_e32 v124, 31, v97
	v_cndmask_b32_e64 v154, v116, v82, s[54:55]
	v_add_u32_e32 v82, s82, v103
	s_and_b64 s[54:55], s[6:7], s[8:9]
	v_xor_b32_e32 v127, v124, v97
	v_cndmask_b32_e64 v155, v116, v82, s[54:55]
	v_add_u32_e32 v82, s82, v89
	s_and_b64 s[54:55], s[2:3], vcc
	v_ashrrev_i32_e32 v128, 31, v127
	v_cndmask_b32_e64 v156, v116, v82, s[54:55]
	v_add_u32_e32 v82, s82, v87
	s_and_b64 s[54:55], s[2:3], s[8:9]
	v_xor_b32_e32 v126, v124, v96
	v_mov_b32_e32 v129, v128
	v_cndmask_b32_e64 v157, v116, v82, s[54:55]
	v_add_u32_e32 v82, s82, v86
	s_and_b64 vcc, s[0:1], vcc
	v_lshl_add_u64 v[126:127], v[126:127], 0, v[128:129]
	v_cndmask_b32_e32 v158, v116, v82, vcc
	v_add_u32_e32 v82, s82, v102
	s_and_b64 vcc, s[0:1], s[8:9]
	v_xor_b32_e32 v160, v126, v128
	v_cndmask_b32_e32 v159, v116, v82, vcc
	v_xor_b32_e32 v129, v127, v128
	v_mul_hi_u32 v82, v160, s21
	v_mad_u64_u32 v[130:131], s[8:9], v160, s20, 0
	v_mad_u64_u32 v[126:127], s[8:9], v129, s21, 0
	v_lshl_add_u64 v[130:131], v[82:83], 0, v[130:131]
	v_add_co_u32_e32 v82, vcc, v130, v126
	v_mov_b32_e32 v125, v124
	s_nop 0
	v_addc_co_u32_e32 v126, vcc, v131, v127, vcc
	v_lshl_add_u64 v[130:131], v[96:97], 0, v[124:125]
	v_xor_b32_e32 v125, v131, v124
	v_xor_b32_e32 v161, v130, v124
	v_mad_u64_u32 v[132:133], s[8:9], v129, s20, 0
	v_mad_u64_u32 v[130:131], s[8:9], v161, s20, 0
	v_mad_u64_u32 v[134:135], s[8:9], v125, s20, 0
	v_mul_hi_u32 v82, v161, s21
	s_mov_b32 s20, 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_mad_u64_u32 v[136:137], s[8:9], v125, s21, 0
	v_lshl_add_u64 v[130:131], v[82:83], 0, v[130:131]
	v_mov_b32_e32 v127, s20
	v_add_co_u32_e32 v82, vcc, v130, v136
	v_lshl_add_u64 v[126:127], v[126:127], 0, v[132:133]
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v137, vcc
	v_mul_lo_u32 v131, s63, v126
	v_mul_lo_u32 v127, s62, v127
	v_mad_u64_u32 v[132:133], s[8:9], s62, v126, 0
	v_addc_co_u32_e32 v135, vcc, 0, v135, vcc
	v_add3_u32 v127, v133, v127, v131
	v_mov_b32_e32 v82, s63
	v_sub_u32_e32 v131, v129, v127
	v_sub_co_u32_e32 v132, vcc, v160, v132
	v_xor_b32_e32 v128, s78, v128
	s_nop 0
	v_subb_co_u32_e64 v131, s[8:9], v131, v82, vcc
	v_subb_co_u32_e32 v127, vcc, v129, v127, vcc
	v_cmp_le_u32_e32 vcc, s62, v132
	buffer_load_dwordx4 v139, s[68:71], 0 offen lds
	s_mov_b32 m0, s75
	v_cndmask_b32_e64 v129, 0, -1, vcc
	v_subrev_co_u32_e32 v132, vcc, s62, v132
	buffer_load_dwordx4 v140, s[68:71], 0 offen lds
	s_nop 0
	v_subbrev_co_u32_e32 v131, vcc, 0, v131, vcc
	v_cmp_le_u32_e32 vcc, s62, v132
	s_mov_b32 m0, s89
	v_add_u32_e32 v111, 0x80, v111
	v_cndmask_b32_e64 v132, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s63, v127
	v_add_u32_e32 v114, 0x80, v114
	s_nop 0
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s63, v131
	s_nop 1
	v_cndmask_b32_e64 v136, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v127
	s_nop 1
	v_cndmask_b32_e32 v127, v133, v129, vcc
	v_cmp_eq_u32_e32 vcc, s63, v131
	v_mov_b32_e32 v131, s20
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_cndmask_b32_e32 v129, v136, v132, vcc
	v_mul_lo_u32 v132, s63, v130
	v_mul_lo_u32 v133, s62, v131
	v_mad_u64_u32 v[130:131], s[8:9], s62, v130, 0
	v_add3_u32 v131, v131, v133, v132
	v_sub_u32_e32 v132, v125, v131
	v_sub_co_u32_e32 v130, vcc, v161, v130
	v_add_u32_e32 v133, 2, v126
	s_nop 0
	v_subb_co_u32_e64 v132, s[8:9], v132, v82, vcc
	v_subb_co_u32_e32 v125, vcc, v125, v131, vcc
	v_cmp_le_u32_e32 vcc, s62, v130
	v_add_u32_e32 v134, 1, v126
	s_waitcnt vmcnt(0)
	ds_read_b128 v[160:163], v121
	ds_read_b128 v[164:167], v121 offset:2048
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_ne_u32_e32 vcc, 0, v129
	s_nop 1
	v_cndmask_b32_e32 v129, v134, v133, vcc
	v_subrev_co_u32_e32 v133, vcc, s62, v130
	s_nop 1
	v_subbrev_co_u32_e64 v134, s[8:9], 0, v132, vcc
	v_subb_co_u32_e32 v82, vcc, v132, v82, vcc
	v_cmp_le_u32_e32 vcc, s63, v125
	v_cmp_le_u32_e64 s[8:9], s62, v133
	s_nop 0
	v_cndmask_b32_e64 v132, 0, -1, vcc
	v_cmp_ne_u32_e32 vcc, 0, v127
	v_cndmask_b32_e64 v135, 0, -1, s[8:9]
	s_nop 0
	v_cndmask_b32_e32 v129, v126, v129, vcc
	v_cmp_le_u32_e32 vcc, s63, v134
	v_xor_b32_e32 v129, v129, v128
	v_sub_u32_e32 v128, v129, v128
	v_cndmask_b32_e64 v126, 0, -1, vcc
	v_subrev_co_u32_e32 v127, vcc, s62, v133
	s_mov_b32 s62, s58
	s_nop 0
	v_subbrev_co_u32_e32 v82, vcc, 0, v82, vcc
	v_cmp_eq_u32_e32 vcc, s63, v125
	s_nop 1
	v_cndmask_b32_e32 v131, v132, v131, vcc
	v_cmp_eq_u32_e32 vcc, s63, v134
	s_mov_b32 s63, s59
	s_nop 0
	v_cndmask_b32_e32 v126, v126, v135, vcc
	v_cmp_ne_u32_e32 vcc, 0, v126
	s_nop 1
	v_cndmask_b32_e32 v82, v134, v82, vcc
	v_cndmask_b32_e32 v126, v133, v127, vcc
	v_cmp_ne_u32_e32 vcc, 0, v131
	s_nop 1
	v_cndmask_b32_e32 v82, v125, v82, vcc
	v_cndmask_b32_e32 v125, v130, v126, vcc
	v_xor_b32_e32 v125, v125, v124
	v_xor_b32_e32 v82, v82, v124
	v_sub_co_u32_e32 v125, vcc, v125, v124
	v_lshl_add_u64 v[126:127], v[84:85], 0, s[76:77]
	s_nop 0
	v_subb_co_u32_e32 v82, vcc, v82, v124, vcc
	v_ashrrev_i32_e32 v82, 31, v82
	v_xad_u32 v124, v128, v124, s22
	v_and_b32_e32 v82, s90, v82
	v_mul_lo_u32 v124, v124, s90
	v_add_u32_e32 v82, v82, v125
	v_cmp_gt_i64_e32 vcc, s[84:85], v[126:127]
	v_add_u32_e32 v125, v82, v124
	s_and_b64 s[8:9], s[28:29], vcc
	v_cndmask_b32_e64 v126, v116, v125, s[8:9]
	v_add3_u32 v82, v124, s74, v82
	s_and_b64 s[8:9], s[30:31], vcc
	v_cndmask_b32_e64 v127, v116, v82, s[8:9]
	v_add_u32_e32 v82, s74, v82
	s_and_b64 s[8:9], s[34:35], vcc
	v_cndmask_b32_e64 v128, v116, v82, s[8:9]
	v_add_u32_e32 v82, s74, v82
	s_and_b64 s[8:9], s[36:37], vcc
	v_cndmask_b32_e64 v129, v116, v82, s[8:9]
	v_add_u32_e32 v82, s74, v82
	s_and_b64 s[8:9], s[38:39], vcc
	v_cndmask_b32_e64 v130, v116, v82, s[8:9]
	v_add_u32_e32 v82, s74, v82
	s_and_b64 s[8:9], s[40:41], vcc
	v_cndmask_b32_e64 v131, v116, v82, s[8:9]
	v_add_u32_e32 v82, s74, v82
	s_and_b64 s[8:9], s[42:43], vcc
	v_cndmask_b32_e64 v132, v116, v82, s[8:9]
	v_add_u32_e32 v82, s74, v82
	s_and_b64 s[8:9], s[44:45], vcc
	v_cndmask_b32_e64 v133, v116, v82, s[8:9]
	v_add_u32_e32 v82, s74, v82
	s_and_b64 s[8:9], s[46:47], vcc
	v_cndmask_b32_e64 v134, v116, v82, s[8:9]
	v_add_u32_e32 v82, s74, v82
	s_and_b64 vcc, s[14:15], vcc
	v_cndmask_b32_e32 v135, v116, v82, vcc
	buffer_load_ubyte v136, v141, s[56:59], 0 offen
	buffer_load_ubyte v137, v150, s[60:63], 0 offen
	buffer_load_ubyte v138, v152, s[60:63], 0 offen
	buffer_load_ubyte v139, v154, s[60:63], 0 offen
	s_nop 0
	buffer_load_ubyte v150, v142, s[56:59], 0 offen
	buffer_load_ubyte v140, v143, s[56:59], 0 offen
	s_nop 0
	buffer_load_ubyte v144, v144, s[56:59], 0 offen
	s_nop 0
	buffer_load_ubyte v141, v145, s[56:59], 0 offen
	s_nop 0
	buffer_load_ubyte v145, v146, s[56:59], 0 offen
	buffer_load_ubyte v142, v148, s[56:59], 0 offen
	buffer_load_ubyte v82, v149, s[56:59], 0 offen
	s_nop 0
	buffer_load_ubyte v146, v151, s[60:63], 0 offen
	buffer_load_ubyte v125, v153, s[60:63], 0 offen
	buffer_load_ubyte v124, v155, s[60:63], 0 offen
	buffer_load_ubyte v143, v156, s[60:63], 0 offen
	buffer_load_ubyte v123, v157, s[60:63], 0 offen
	s_add_u32 s10, s10, 1
	buffer_load_dword v126, s[64:67], 0 offen lds
	s_mov_b32 m0, s92
	s_addc_u32 s11, s11, 0
	buffer_load_dword v127, s[64:67], 0 offen lds
	buffer_load_ubyte v127, v158, s[60:63], 0 offen
	s_nop 0
	buffer_load_ubyte v126, v159, s[60:63], 0 offen
	s_mov_b32 m0, s93
	s_add_i32 s82, s82, s88
	buffer_load_dword v128, s[64:67], 0 offen lds
	s_mov_b32 m0, s94
	s_add_i32 s81, s81, s86
	buffer_load_dword v129, s[64:67], 0 offen lds
	s_mov_b32 m0, s95
	s_mov_b64 s[8:9], 0x800
	buffer_load_dword v130, s[64:67], 0 offen lds
	s_mov_b32 m0, s96
	s_add_u32 s76, s76, 0x100
	buffer_load_dword v131, s[64:67], 0 offen lds
	s_mov_b32 m0, s97
	v_cmp_lt_i64_e32 vcc, s[10:11], v[98:99]
	buffer_load_dword v132, s[64:67], 0 offen lds
	s_mov_b32 m0, s98
	v_lshl_add_u64 v[96:97], v[96:97], 0, s[8:9]
	buffer_load_dword v133, s[64:67], 0 offen lds
	s_mov_b32 m0, s99
	s_addc_u32 s77, s77, 0
	buffer_load_dword v134, s[64:67], 0 offen lds
	s_mov_b32 m0, s80
	s_and_b64 vcc, exec, vcc
	buffer_load_dword v135, s[64:67], 0 offen lds
	s_waitcnt vmcnt(0)
	ds_read_b128 v[128:131], v118
	ds_read_b128 v[132:135], v118 offset:2048
	ds_read_b128 v[152:155], v118 offset:4096
	ds_read_b128 v[168:171], v118 offset:6144
	ds_read_b128 v[156:159], v118 offset:8192
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[160:163], v[128:131], v[60:63], v136, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[160:163], v[132:135], v[56:59], v136, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[160:163], v[152:155], v[52:55], v136, v139 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[160:163], v[168:171], v[48:51], v136, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[160:163], v[156:159], v[44:47], v136, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[164:167], v[128:131], v[40:43], v140, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[164:167], v[132:135], v[36:39], v140, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[164:167], v[152:155], v[32:35], v140, v139 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[164:167], v[168:171], v[28:31], v140, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[164:167], v[156:159], v[24:27], v140, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[160:163], v121 offset:4096
	ds_read_b128 v[164:167], v121 offset:6144
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[160:163], v[128:131], v[20:23], v141, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[160:163], v[132:135], v[16:19], v141, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[160:163], v[152:155], v[12:15], v141, v139 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[160:163], v[168:171], v[8:11], v141, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[160:163], v[156:159], v[4:7], v141, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[164:167], v[128:131], v[0:3], v142, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[164:167], v[132:135], v[76:79], v142, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[132:135], v120 offset:6144
	ds_read_b128 v[128:131], v120 offset:8192
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[164:167], v[152:155], v[72:75], v142, v139 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[152:155], v122
	ds_read_b128 v[160:163], v122 offset:2048
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[164:167], v[168:171], v[68:71], v142, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[164:167], v[156:159], v[64:67], v142, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[156:159], v120
	ds_read_b128 v[140:143], v120 offset:2048
	ds_read_b128 v[136:139], v120 offset:4096
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[152:155], v[156:159], v[60:63], v150, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[152:155], v[140:143], v[56:59], v150, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[152:155], v[136:139], v[52:55], v150, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[152:155], v[132:135], v[48:51], v150, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[152:155], v[128:131], v[44:47], v150, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[152:155], v122 offset:4096
	ds_read_b128 v[148:151], v122 offset:6144
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[160:163], v[156:159], v[40:43], v144, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[160:163], v[140:143], v[36:39], v144, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[160:163], v[136:139], v[32:35], v144, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[160:163], v[132:135], v[28:31], v144, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[160:163], v[128:131], v[24:27], v144, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[152:155], v[156:159], v[20:23], v145, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[152:155], v[140:143], v[16:19], v145, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[152:155], v[136:139], v[12:15], v145, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[152:155], v[132:135], v[8:11], v145, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[152:155], v[128:131], v[4:7], v145, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[148:151], v[156:159], v[0:3], v82, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[148:151], v[140:143], v[76:79], v82, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[148:151], v[136:139], v[72:75], v82, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[148:151], v[132:135], v[68:71], v82, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[148:151], v[128:131], v[64:67], v82, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cbranch_vccnz .LBB0_20
	v_readlane_b32 s12, v255, 16
	v_readlane_b32 s82, v255, 14
	v_readlane_b32 s13, v255, 17
	v_readlane_b32 s83, v255, 15
	s_mov_b64 s[62:63], s[22:23]
.LBB0_22:
	v_bfe_u32 v80, v63, 16, 1
	s_movk_i32 s4, 0x7fff
	v_bfe_u32 v81, v62, 16, 1
	v_add3_u32 v80, v63, v80, s4
	s_waitcnt vmcnt(2)
	v_bfe_u32 v82, v61, 16, 1
	v_add3_u32 v81, v62, v81, s4
	v_lshrrev_b32_e32 v80, 16, v80
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v84, 0x7fc0
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v83, v60, 16, 1
	v_add3_u32 v82, v61, v82, s4
	v_lshrrev_b32_e32 v81, 16, v81
	v_cndmask_b32_e32 v63, v84, v80, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v83, v60, v83, s4
	v_lshrrev_b32_e32 v82, 16, v82
	v_cndmask_b32_e32 v62, v84, v81, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v80, v59, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v61, v84, v82, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v81, v58, 16, 1
	v_add3_u32 v80, v59, v80, s4
	v_cndmask_b32_e32 v60, v84, v83, vcc
	v_bfe_u32 v82, v57, 16, 1
	v_add3_u32 v81, v58, v81, s4
	v_lshrrev_b32_e32 v80, 16, v80
	v_cmp_o_f32_e32 vcc, v59, v59
	v_add3_u32 v82, v57, v82, s4
	v_lshrrev_b32_e32 v81, 16, v81
	v_cndmask_b32_e32 v59, v84, v80, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_bfe_u32 v83, v56, 16, 1
	v_lshrrev_b32_e32 v82, 16, v82
	v_cndmask_b32_e32 v58, v84, v81, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_add3_u32 v83, v56, v83, s4
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v80, v84, v82, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v56, v55, 16, 1
	v_bfe_u32 v57, v54, 16, 1
	v_add3_u32 v56, v55, v56, s4
	v_cndmask_b32_e32 v81, v84, v83, vcc
	v_bfe_u32 v82, v53, 16, 1
	v_add3_u32 v57, v54, v57, s4
	v_lshrrev_b32_e32 v56, 16, v56
	v_cmp_o_f32_e32 vcc, v55, v55
	v_add3_u32 v82, v53, v82, s4
	v_lshrrev_b32_e32 v57, 16, v57
	v_cndmask_b32_e32 v85, v84, v56, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_bfe_u32 v83, v52, 16, 1
	v_lshrrev_b32_e32 v82, 16, v82
	s_waitcnt vmcnt(0)
	v_cndmask_b32_e32 v86, v84, v57, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_add3_u32 v83, v52, v83, s4
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v82, v84, v82, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v52, v51, 16, 1
	v_bfe_u32 v53, v50, 16, 1
	v_add3_u32 v52, v51, v52, s4
	v_cndmask_b32_e32 v83, v84, v83, vcc
	v_bfe_u32 v54, v49, 16, 1
	v_add3_u32 v53, v50, v53, s4
	v_lshrrev_b32_e32 v52, 16, v52
	v_cmp_o_f32_e32 vcc, v51, v51
	v_add3_u32 v54, v49, v54, s4
	v_lshrrev_b32_e32 v53, 16, v53
	v_cndmask_b32_e32 v87, v84, v52, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_bfe_u32 v55, v48, 16, 1
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v88, v84, v53, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_add3_u32 v55, v48, v55, s4
	v_lshrrev_b32_e32 v55, 16, v55
	v_cndmask_b32_e32 v89, v84, v54, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v48, v47, 16, 1
	v_bfe_u32 v49, v46, 16, 1
	v_add3_u32 v48, v47, v48, s4
	v_cndmask_b32_e32 v90, v84, v55, vcc
	v_bfe_u32 v50, v45, 16, 1
	v_add3_u32 v49, v46, v49, s4
	v_lshrrev_b32_e32 v48, 16, v48
	v_cmp_o_f32_e32 vcc, v47, v47
	v_add3_u32 v50, v45, v50, s4
	v_lshrrev_b32_e32 v49, 16, v49
	v_cndmask_b32_e32 v91, v84, v48, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_bfe_u32 v51, v44, 16, 1
	v_lshrrev_b32_e32 v50, 16, v50
	v_cndmask_b32_e32 v92, v84, v49, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_add3_u32 v51, v44, v51, s4
	v_lshrrev_b32_e32 v51, 16, v51
	v_cndmask_b32_e32 v93, v84, v50, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v44, v43, 16, 1
	v_bfe_u32 v45, v42, 16, 1
	v_add3_u32 v44, v43, v44, s4
	v_cndmask_b32_e32 v94, v84, v51, vcc
	v_bfe_u32 v46, v41, 16, 1
	v_add3_u32 v45, v42, v45, s4
	v_lshrrev_b32_e32 v44, 16, v44
	v_cmp_o_f32_e32 vcc, v43, v43
	v_add3_u32 v46, v41, v46, s4
	v_lshrrev_b32_e32 v45, 16, v45
	v_cndmask_b32_e32 v49, v84, v44, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_bfe_u32 v47, v40, 16, 1
	v_lshrrev_b32_e32 v46, 16, v46
	v_cndmask_b32_e32 v95, v84, v45, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_add3_u32 v47, v40, v47, s4
	v_lshrrev_b32_e32 v47, 16, v47
	v_cndmask_b32_e32 v96, v84, v46, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v40, v39, 16, 1
	v_bfe_u32 v41, v38, 16, 1
	v_add3_u32 v40, v39, v40, s4
	v_cndmask_b32_e32 v97, v84, v47, vcc
	v_bfe_u32 v42, v37, 16, 1
	v_add3_u32 v41, v38, v41, s4
	v_lshrrev_b32_e32 v40, 16, v40
	v_cmp_o_f32_e32 vcc, v39, v39
	v_add3_u32 v42, v37, v42, s4
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v46, v84, v40, vcc
	v_cmp_o_f32_e32 vcc, v38, v38
	v_bfe_u32 v43, v36, 16, 1
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v52, v84, v41, vcc
	v_cmp_o_f32_e32 vcc, v37, v37
	v_add3_u32 v43, v36, v43, s4
	v_lshrrev_b32_e32 v43, 16, v43
	v_cndmask_b32_e32 v98, v84, v42, vcc
	v_cmp_o_f32_e32 vcc, v36, v36
	v_bfe_u32 v36, v35, 16, 1
	v_bfe_u32 v37, v34, 16, 1
	v_add3_u32 v36, v35, v36, s4
	v_cndmask_b32_e32 v99, v84, v43, vcc
	v_bfe_u32 v38, v33, 16, 1
	v_add3_u32 v37, v34, v37, s4
	v_lshrrev_b32_e32 v36, 16, v36
	v_cmp_o_f32_e32 vcc, v35, v35
	v_add3_u32 v38, v33, v38, s4
	v_lshrrev_b32_e32 v37, 16, v37
	v_cndmask_b32_e32 v44, v84, v36, vcc
	v_cmp_o_f32_e32 vcc, v34, v34
	v_bfe_u32 v39, v32, 16, 1
	v_lshrrev_b32_e32 v38, 16, v38
	v_cndmask_b32_e32 v50, v84, v37, vcc
	v_cmp_o_f32_e32 vcc, v33, v33
	v_add3_u32 v39, v32, v39, s4
	v_lshrrev_b32_e32 v39, 16, v39
	v_cndmask_b32_e32 v54, v84, v38, vcc
	v_cmp_o_f32_e32 vcc, v32, v32
	v_bfe_u32 v32, v31, 16, 1
	v_bfe_u32 v33, v30, 16, 1
	v_add3_u32 v32, v31, v32, s4
	v_cndmask_b32_e32 v101, v84, v39, vcc
	v_bfe_u32 v34, v29, 16, 1
	v_add3_u32 v33, v30, v33, s4
	v_lshrrev_b32_e32 v32, 16, v32
	v_cmp_o_f32_e32 vcc, v31, v31
	v_add3_u32 v34, v29, v34, s4
	v_lshrrev_b32_e32 v33, 16, v33
	v_cndmask_b32_e32 v43, v84, v32, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_bfe_u32 v35, v28, 16, 1
	v_lshrrev_b32_e32 v34, 16, v34
	v_cndmask_b32_e32 v47, v84, v33, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_add3_u32 v35, v28, v35, s4
	v_lshrrev_b32_e32 v35, 16, v35
	v_cndmask_b32_e32 v53, v84, v34, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v28, v27, 16, 1
	v_bfe_u32 v29, v26, 16, 1
	v_add3_u32 v28, v27, v28, s4
	v_cndmask_b32_e32 v102, v84, v35, vcc
	v_bfe_u32 v30, v25, 16, 1
	v_add3_u32 v29, v26, v29, s4
	v_lshrrev_b32_e32 v28, 16, v28
	v_cmp_o_f32_e32 vcc, v27, v27
	v_add3_u32 v30, v25, v30, s4
	v_lshrrev_b32_e32 v29, 16, v29
	v_cndmask_b32_e32 v41, v84, v28, vcc
	v_cmp_o_f32_e32 vcc, v26, v26
	v_bfe_u32 v31, v24, 16, 1
	v_lshrrev_b32_e32 v30, 16, v30
	v_cndmask_b32_e32 v45, v84, v29, vcc
	v_cmp_o_f32_e32 vcc, v25, v25
	v_add3_u32 v31, v24, v31, s4
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v51, v84, v30, vcc
	v_cmp_o_f32_e32 vcc, v24, v24
	v_bfe_u32 v24, v23, 16, 1
	v_bfe_u32 v25, v22, 16, 1
	v_add3_u32 v24, v23, v24, s4
	v_cndmask_b32_e32 v55, v84, v31, vcc
	v_bfe_u32 v26, v21, 16, 1
	v_add3_u32 v25, v22, v25, s4
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v23, v23
	v_add3_u32 v26, v21, v26, s4
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v29, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_bfe_u32 v27, v20, 16, 1
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v38, v84, v25, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_add3_u32 v27, v20, v27, s4
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v42, v84, v26, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v20, v19, 16, 1
	v_bfe_u32 v21, v18, 16, 1
	v_add3_u32 v20, v19, v20, s4
	v_cndmask_b32_e32 v48, v84, v27, vcc
	v_bfe_u32 v22, v17, 16, 1
	v_add3_u32 v21, v18, v21, s4
	v_lshrrev_b32_e32 v20, 16, v20
	v_cmp_o_f32_e32 vcc, v19, v19
	v_add3_u32 v22, v17, v22, s4
	v_lshrrev_b32_e32 v21, 16, v21
	v_cndmask_b32_e32 v26, v84, v20, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_bfe_u32 v23, v16, 16, 1
	v_lshrrev_b32_e32 v22, 16, v22
	v_cndmask_b32_e32 v32, v84, v21, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_add3_u32 v23, v16, v23, s4
	v_lshrrev_b32_e32 v23, 16, v23
	v_cndmask_b32_e32 v36, v84, v22, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v16, v15, 16, 1
	v_bfe_u32 v17, v14, 16, 1
	v_add3_u32 v16, v15, v16, s4
	v_cndmask_b32_e32 v40, v84, v23, vcc
	v_bfe_u32 v18, v13, 16, 1
	v_add3_u32 v17, v14, v17, s4
	v_lshrrev_b32_e32 v16, 16, v16
	v_cmp_o_f32_e32 vcc, v15, v15
	v_add3_u32 v18, v13, v18, s4
	v_lshrrev_b32_e32 v17, 16, v17
	v_cndmask_b32_e32 v24, v84, v16, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_bfe_u32 v19, v12, 16, 1
	v_lshrrev_b32_e32 v18, 16, v18
	v_cndmask_b32_e32 v30, v84, v17, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_add3_u32 v19, v12, v19, s4
	v_lshrrev_b32_e32 v19, 16, v19
	v_cndmask_b32_e32 v34, v84, v18, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v12, v11, 16, 1
	v_bfe_u32 v13, v10, 16, 1
	v_add3_u32 v12, v11, v12, s4
	v_cndmask_b32_e32 v39, v84, v19, vcc
	v_bfe_u32 v14, v9, 16, 1
	v_add3_u32 v13, v10, v13, s4
	v_lshrrev_b32_e32 v12, 16, v12
	v_cmp_o_f32_e32 vcc, v11, v11
	v_add3_u32 v14, v9, v14, s4
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v23, v84, v12, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_bfe_u32 v15, v8, 16, 1
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v27, v84, v13, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_add3_u32 v15, v8, v15, s4
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v33, v84, v14, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v8, v7, 16, 1
	v_bfe_u32 v9, v6, 16, 1
	v_add3_u32 v8, v7, v8, s4
	v_cndmask_b32_e32 v37, v84, v15, vcc
	v_bfe_u32 v10, v5, 16, 1
	v_add3_u32 v9, v6, v9, s4
	v_lshrrev_b32_e32 v8, 16, v8
	v_cmp_o_f32_e32 vcc, v7, v7
	v_add3_u32 v10, v5, v10, s4
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v21, v84, v8, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_bfe_u32 v11, v4, 16, 1
	v_lshrrev_b32_e32 v10, 16, v10
	v_cndmask_b32_e32 v25, v84, v9, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_add3_u32 v11, v4, v11, s4
	v_lshrrev_b32_e32 v11, 16, v11
	v_cndmask_b32_e32 v31, v84, v10, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v4, v3, 16, 1
	v_bfe_u32 v5, v2, 16, 1
	v_add3_u32 v4, v3, v4, s4
	v_cndmask_b32_e32 v35, v84, v11, vcc
	v_bfe_u32 v6, v1, 16, 1
	v_add3_u32 v5, v2, v5, s4
	v_lshrrev_b32_e32 v4, 16, v4
	v_cmp_o_f32_e32 vcc, v3, v3
	v_add3_u32 v6, v1, v6, s4
	v_lshrrev_b32_e32 v5, 16, v5
	v_cndmask_b32_e32 v9, v84, v4, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_bfe_u32 v7, v0, 16, 1
	v_lshrrev_b32_e32 v6, 16, v6
	v_cndmask_b32_e32 v18, v84, v5, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_add3_u32 v7, v0, v7, s4
	v_lshrrev_b32_e32 v7, 16, v7
	v_cndmask_b32_e32 v22, v84, v6, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_bfe_u32 v0, v79, 16, 1
	v_bfe_u32 v1, v78, 16, 1
	v_add3_u32 v0, v79, v0, s4
	v_cndmask_b32_e32 v28, v84, v7, vcc
	v_bfe_u32 v2, v77, 16, 1
	v_add3_u32 v1, v78, v1, s4
	v_lshrrev_b32_e32 v0, 16, v0
	v_cmp_o_f32_e32 vcc, v79, v79
	v_bfe_u32 v3, v76, 16, 1
	v_add3_u32 v2, v77, v2, s4
	v_lshrrev_b32_e32 v1, 16, v1
	v_cndmask_b32_e32 v7, v84, v0, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_add3_u32 v3, v76, v3, s4
	v_lshrrev_b32_e32 v2, 16, v2
	v_cndmask_b32_e32 v12, v84, v1, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_bfe_u32 v0, v75, 16, 1
	v_lshrrev_b32_e32 v3, 16, v3
	v_cndmask_b32_e32 v16, v84, v2, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_bfe_u32 v1, v74, 16, 1
	v_add3_u32 v0, v75, v0, s4
	v_cndmask_b32_e32 v20, v84, v3, vcc
	v_bfe_u32 v2, v73, 16, 1
	v_add3_u32 v1, v74, v1, s4
	v_lshrrev_b32_e32 v0, 16, v0
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v3, v72, 16, 1
	v_add3_u32 v2, v73, v2, s4
	v_lshrrev_b32_e32 v1, 16, v1
	v_cndmask_b32_e32 v5, v84, v0, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v3, v72, v3, s4
	v_lshrrev_b32_e32 v2, 16, v2
	v_cndmask_b32_e32 v10, v84, v1, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v0, v71, 16, 1
	v_lshrrev_b32_e32 v3, 16, v3
	v_cndmask_b32_e32 v14, v84, v2, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v1, v70, 16, 1
	v_add3_u32 v0, v71, v0, s4
	v_cndmask_b32_e32 v19, v84, v3, vcc
	v_bfe_u32 v2, v69, 16, 1
	v_add3_u32 v1, v70, v1, s4
	v_lshrrev_b32_e32 v0, 16, v0
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v3, v68, 16, 1
	v_add3_u32 v2, v69, v2, s4
	v_lshrrev_b32_e32 v1, 16, v1
	v_cndmask_b32_e32 v4, v84, v0, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v3, v68, v3, s4
	v_lshrrev_b32_e32 v2, 16, v2
	v_cndmask_b32_e32 v8, v84, v1, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_lshrrev_b32_e32 v3, 16, v3
	v_bfe_u32 v0, v67, 16, 1
	v_cndmask_b32_e32 v13, v84, v2, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v1, v66, 16, 1
	v_add3_u32 v0, v67, v0, s4
	v_cndmask_b32_e32 v17, v84, v3, vcc
	v_bfe_u32 v3, v64, 16, 1
	v_bfe_u32 v2, v65, 16, 1
	v_add3_u32 v3, v64, v3, s4
	v_add3_u32 v1, v66, v1, s4
	v_lshrrev_b32_e32 v0, 16, v0
	v_cmp_o_f32_e32 vcc, v67, v67
	v_readlane_b32 s24, v254, 1
	v_add3_u32 v2, v65, v2, s4
	v_lshrrev_b32_e32 v1, 16, v1
	v_lshrrev_b32_e32 v15, 16, v3
	v_cndmask_b32_e32 v3, v84, v0, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_readlane_b32 s25, v254, 2
	v_lshrrev_b32_e32 v2, 16, v2
	v_cndmask_b32_e32 v6, v84, v1, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s8, s72, s25
	s_mul_hi_u32 s9, s72, s24
	v_cndmask_b32_e32 v11, v84, v2, vcc
	v_lshl_or_b32 v2, v147, 2, v117
	v_cmp_o_f32_e32 vcc, v64, v64
	v_or_b32_e32 v0, s24, v2
	v_mov_b32_e32 v1, s25
	s_add_i32 s8, s9, s8
	s_mul_i32 s9, s73, s24
	v_cndmask_b32_e32 v15, v84, v15, vcc
	v_cmp_gt_i64_e32 vcc, s[12:13], v[0:1]
	s_add_i32 s9, s8, s9
	s_mul_i32 s8, s72, s24
	s_and_b64 s[4:5], vcc, s[18:19]
	s_lshl_b64 s[8:9], s[8:9], 1
	s_add_u32 s10, s82, s8
	s_addc_u32 s11, s83, s9
	s_lshl_b64 s[8:9], s[62:63], 1
	s_add_u32 s20, s10, s8
	v_mul_lo_u32 v2, s72, v2
	s_addc_u32 s8, s11, s9
	s_and_b32 s9, s72, 0x3fff
	v_add_u32_e32 v64, v2, v115
	v_not_b32_e32 v1, -2.0
	s_lshl_b32 s9, s9, 16
	s_and_b32 s8, s8, 0xffff
	s_or_b32 s8, s9, s8
	v_cndmask_b32_e64 v56, v1, v64, s[4:5]
	s_or_b32 s21, s8, 2.0
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffd
	v_lshlrev_b32_e32 v56, 1, v56
	buffer_store_short v60, v56, s[20:23], 0 offen
	v_or_b32_e32 v56, 1, v0
	v_mov_b32_e32 v57, s25
	v_cmp_gt_i64_e64 s[4:5], s[12:13], v[56:57]
	v_add_u32_e32 v56, s72, v64
	s_and_b64 s[8:9], s[4:5], s[18:19]
	v_cndmask_b32_e64 v56, v1, v56, s[8:9]
	v_lshlrev_b32_e32 v56, 1, v56
	buffer_store_short v61, v56, s[20:23], 0 offen
	v_or_b32_e32 v56, 2, v0
	v_cmp_gt_i64_e64 s[8:9], s[12:13], v[56:57]
	v_lshl_add_u32 v60, s72, 1, v2
	v_add_u32_e32 v56, v60, v115
	s_and_b64 s[10:11], s[8:9], s[18:19]
	v_cndmask_b32_e64 v56, v1, v56, s[10:11]
	v_lshlrev_b32_e32 v56, 1, v56
	buffer_store_short v62, v56, s[20:23], 0 offen
	v_or_b32_e32 v56, 3, v0
	v_cmp_gt_i64_e64 s[10:11], s[12:13], v[56:57]
	v_mad_u64_u32 v[56:57], s[14:15], s72, 3, v[2:3]
	v_add_u32_e32 v57, v56, v115
	s_and_b64 s[14:15], s[10:11], s[18:19]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v63, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v2, v119
	s_and_b64 s[14:15], vcc, s[16:17]
	v_cndmask_b32_e64 v61, v1, v57, s[14:15]
	v_add_u32_e32 v57, s72, v57
	s_and_b64 s[14:15], s[4:5], s[16:17]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v61, 1, v61
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v81, v61, s[20:23], 0 offen
	buffer_store_short v80, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v60, v119
	s_and_b64 s[14:15], s[8:9], s[16:17]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v58, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v56, v119
	s_and_b64 s[14:15], s[10:11], s[16:17]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v59, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, 32, v64
	s_and_b64 s[14:15], vcc, s[6:7]
	v_cndmask_b32_e64 v58, v1, v57, s[14:15]
	v_add_u32_e32 v57, s72, v57
	s_and_b64 s[14:15], s[4:5], s[6:7]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_add_u32_e32 v62, 32, v115
	v_lshlrev_b32_e32 v58, 1, v58
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v83, v58, s[20:23], 0 offen
	buffer_store_short v82, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v60, v62
	s_and_b64 s[14:15], s[8:9], s[6:7]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v86, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v56, v62
	s_and_b64 s[14:15], s[10:11], s[6:7]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v85, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, 48, v64
	s_and_b64 s[14:15], vcc, s[2:3]
	v_cndmask_b32_e64 v58, v1, v57, s[14:15]
	v_add_u32_e32 v57, s72, v57
	s_and_b64 s[14:15], s[4:5], s[2:3]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v58, 1, v58
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v90, v58, s[20:23], 0 offen
	buffer_store_short v89, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v60, v100
	s_and_b64 s[14:15], s[8:9], s[2:3]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v88, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v56, v100
	s_and_b64 s[14:15], s[10:11], s[2:3]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v87, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, 64, v64
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v58, v1, v57, vcc
	v_add_u32_e32 v57, s72, v57
	s_and_b64 vcc, s[4:5], s[0:1]
	v_cndmask_b32_e32 v57, v1, v57, vcc
	v_add_u32_e32 v63, 64, v115
	v_lshlrev_b32_e32 v58, 1, v58
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v94, v58, s[20:23], 0 offen
	buffer_store_short v93, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v60, v63
	s_and_b64 vcc, s[8:9], s[0:1]
	v_cndmask_b32_e32 v57, v1, v57, vcc
	v_add_u32_e32 v56, v56, v63
	s_and_b64 vcc, s[10:11], s[0:1]
	v_cndmask_b32_e32 v56, v1, v56, vcc
	v_lshlrev_b32_e32 v57, 1, v57
	v_lshlrev_b32_e32 v56, 1, v56
	buffer_store_short v92, v57, s[20:23], 0 offen
	buffer_store_short v91, v56, s[20:23], 0 offen
	v_or_b32_e32 v56, 16, v0
	v_mov_b32_e32 v57, s25
	v_cmp_gt_i64_e32 vcc, s[12:13], v[56:57]
	v_lshl_add_u32 v64, s72, 4, v2
	v_add_u32_e32 v56, v64, v115
	s_and_b64 s[4:5], vcc, s[18:19]
	v_cndmask_b32_e64 v56, v1, v56, s[4:5]
	v_lshlrev_b32_e32 v56, 1, v56
	buffer_store_short v97, v56, s[20:23], 0 offen
	v_or_b32_e32 v56, 17, v0
	v_cmp_gt_i64_e64 s[4:5], s[12:13], v[56:57]
	v_mad_u64_u32 v[56:57], s[8:9], s72, 17, v[2:3]
	v_add_u32_e32 v57, v56, v115
	s_and_b64 s[8:9], s[4:5], s[18:19]
	v_cndmask_b32_e64 v57, v1, v57, s[8:9]
	v_or_b32_e32 v58, 18, v0
	v_mov_b32_e32 v59, s25
	v_lshlrev_b32_e32 v57, 1, v57
	v_cmp_gt_i64_e64 s[8:9], s[12:13], v[58:59]
	v_mad_u64_u32 v[58:59], s[10:11], s72, 18, v[2:3]
	buffer_store_short v96, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v58, v115
	s_and_b64 s[10:11], s[8:9], s[18:19]
	v_cndmask_b32_e64 v57, v1, v57, s[10:11]
	v_or_b32_e32 v60, 19, v0
	v_mov_b32_e32 v61, s25
	v_lshlrev_b32_e32 v57, 1, v57
	v_cmp_gt_i64_e64 s[10:11], s[12:13], v[60:61]
	v_mad_u64_u32 v[60:61], s[14:15], s72, 19, v[2:3]
	buffer_store_short v95, v57, s[20:23], 0 offen
	v_add_u32_e32 v57, v60, v115
	s_and_b64 s[14:15], s[10:11], s[18:19]
	v_cndmask_b32_e64 v57, v1, v57, s[14:15]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v49, v57, s[20:23], 0 offen
	v_add_u32_e32 v49, v64, v119
	s_and_b64 s[14:15], vcc, s[16:17]
	v_cndmask_b32_e64 v49, v1, v49, s[14:15]
	v_lshlrev_b32_e32 v49, 1, v49
	buffer_store_short v99, v49, s[20:23], 0 offen
	v_add_u32_e32 v49, v56, v119
	s_and_b64 s[14:15], s[4:5], s[16:17]
	v_cndmask_b32_e64 v49, v1, v49, s[14:15]
	v_lshlrev_b32_e32 v49, 1, v49
	buffer_store_short v98, v49, s[20:23], 0 offen
	v_add_u32_e32 v49, v58, v119
	s_and_b64 s[14:15], s[8:9], s[16:17]
	v_cndmask_b32_e64 v49, v1, v49, s[14:15]
	v_lshlrev_b32_e32 v49, 1, v49
	buffer_store_short v52, v49, s[20:23], 0 offen
	v_add_u32_e32 v49, v60, v119
	s_and_b64 s[14:15], s[10:11], s[16:17]
	v_cndmask_b32_e64 v49, v1, v49, s[14:15]
	v_lshlrev_b32_e32 v49, 1, v49
	buffer_store_short v46, v49, s[20:23], 0 offen
	v_add_u32_e32 v46, v64, v62
	s_and_b64 s[14:15], vcc, s[6:7]
	v_cndmask_b32_e64 v46, v1, v46, s[14:15]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v101, v46, s[20:23], 0 offen
	v_add_u32_e32 v46, v56, v62
	s_and_b64 s[14:15], s[4:5], s[6:7]
	v_cndmask_b32_e64 v46, v1, v46, s[14:15]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v54, v46, s[20:23], 0 offen
	v_add_u32_e32 v46, v58, v62
	s_and_b64 s[14:15], s[8:9], s[6:7]
	v_cndmask_b32_e64 v46, v1, v46, s[14:15]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v50, v46, s[20:23], 0 offen
	v_add_u32_e32 v46, v60, v62
	s_and_b64 s[14:15], s[10:11], s[6:7]
	v_cndmask_b32_e64 v46, v1, v46, s[14:15]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v44, v46, s[20:23], 0 offen
	v_add_u32_e32 v44, v64, v100
	s_and_b64 s[14:15], vcc, s[2:3]
	v_cndmask_b32_e64 v44, v1, v44, s[14:15]
	v_lshlrev_b32_e32 v44, 1, v44
	buffer_store_short v102, v44, s[20:23], 0 offen
	v_add_u32_e32 v44, v56, v100
	s_and_b64 s[14:15], s[4:5], s[2:3]
	v_cndmask_b32_e64 v44, v1, v44, s[14:15]
	v_lshlrev_b32_e32 v44, 1, v44
	buffer_store_short v53, v44, s[20:23], 0 offen
	v_add_u32_e32 v44, v58, v100
	s_and_b64 s[14:15], s[8:9], s[2:3]
	v_cndmask_b32_e64 v44, v1, v44, s[14:15]
	v_lshlrev_b32_e32 v44, 1, v44
	buffer_store_short v47, v44, s[20:23], 0 offen
	v_add_u32_e32 v44, v60, v100
	s_and_b64 s[14:15], s[10:11], s[2:3]
	v_cndmask_b32_e64 v44, v1, v44, s[14:15]
	v_lshlrev_b32_e32 v44, 1, v44
	buffer_store_short v43, v44, s[20:23], 0 offen
	v_add_u32_e32 v43, v64, v63
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v43, v1, v43, vcc
	v_lshlrev_b32_e32 v43, 1, v43
	buffer_store_short v55, v43, s[20:23], 0 offen
	v_add_u32_e32 v43, v56, v63
	s_and_b64 vcc, s[4:5], s[0:1]
	v_cndmask_b32_e32 v43, v1, v43, vcc
	v_lshlrev_b32_e32 v43, 1, v43
	buffer_store_short v51, v43, s[20:23], 0 offen
	v_add_u32_e32 v43, v58, v63
	s_and_b64 vcc, s[8:9], s[0:1]
	v_cndmask_b32_e32 v43, v1, v43, vcc
	v_lshlrev_b32_e32 v43, 1, v43
	buffer_store_short v45, v43, s[20:23], 0 offen
	v_add_u32_e32 v43, v60, v63
	s_and_b64 vcc, s[10:11], s[0:1]
	v_cndmask_b32_e32 v43, v1, v43, vcc
	v_lshlrev_b32_e32 v43, 1, v43
	v_or_b32_e32 v44, 32, v0
	v_mov_b32_e32 v45, s25
	buffer_store_short v41, v43, s[20:23], 0 offen
	v_cmp_gt_i64_e32 vcc, s[12:13], v[44:45]
	v_lshl_add_u32 v41, s72, 5, v2
	v_add_u32_e32 v43, v41, v115
	s_and_b64 s[4:5], vcc, s[18:19]
	v_cndmask_b32_e64 v43, v1, v43, s[4:5]
	v_or_b32_e32 v44, 33, v0
	v_lshlrev_b32_e32 v43, 1, v43
	v_cmp_gt_i64_e64 s[4:5], s[12:13], v[44:45]
	v_mad_u64_u32 v[44:45], s[8:9], s72, 33, v[2:3]
	buffer_store_short v48, v43, s[20:23], 0 offen
	v_add_u32_e32 v43, v44, v115
	s_and_b64 s[8:9], s[4:5], s[18:19]
	v_cndmask_b32_e64 v43, v1, v43, s[8:9]
	v_lshlrev_b32_e32 v43, 1, v43
	buffer_store_short v42, v43, s[20:23], 0 offen
	v_or_b32_e32 v42, 34, v0
	v_mov_b32_e32 v43, s25
	v_cmp_gt_i64_e64 s[8:9], s[12:13], v[42:43]
	v_mad_u64_u32 v[42:43], s[10:11], s72, 34, v[2:3]
	v_add_u32_e32 v43, v42, v115
	s_and_b64 s[10:11], s[8:9], s[18:19]
	v_cndmask_b32_e64 v43, v1, v43, s[10:11]
	v_or_b32_e32 v46, 35, v0
	v_mov_b32_e32 v47, s25
	v_lshlrev_b32_e32 v43, 1, v43
	v_cmp_gt_i64_e64 s[10:11], s[12:13], v[46:47]
	v_mad_u64_u32 v[46:47], s[14:15], s72, 35, v[2:3]
	buffer_store_short v38, v43, s[20:23], 0 offen
	v_add_u32_e32 v38, v46, v115
	s_and_b64 s[14:15], s[10:11], s[18:19]
	v_cndmask_b32_e64 v38, v1, v38, s[14:15]
	v_lshlrev_b32_e32 v38, 1, v38
	buffer_store_short v29, v38, s[20:23], 0 offen
	v_add_u32_e32 v29, v41, v119
	s_and_b64 s[14:15], vcc, s[16:17]
	v_cndmask_b32_e64 v29, v1, v29, s[14:15]
	v_lshlrev_b32_e32 v29, 1, v29
	buffer_store_short v40, v29, s[20:23], 0 offen
	v_add_u32_e32 v29, v44, v119
	s_and_b64 s[14:15], s[4:5], s[16:17]
	v_cndmask_b32_e64 v29, v1, v29, s[14:15]
	v_lshlrev_b32_e32 v29, 1, v29
	buffer_store_short v36, v29, s[20:23], 0 offen
	v_add_u32_e32 v29, v42, v119
	s_and_b64 s[14:15], s[8:9], s[16:17]
	v_cndmask_b32_e64 v29, v1, v29, s[14:15]
	v_lshlrev_b32_e32 v29, 1, v29
	buffer_store_short v32, v29, s[20:23], 0 offen
	v_add_u32_e32 v29, v46, v119
	s_and_b64 s[14:15], s[10:11], s[16:17]
	v_cndmask_b32_e64 v29, v1, v29, s[14:15]
	v_lshlrev_b32_e32 v29, 1, v29
	buffer_store_short v26, v29, s[20:23], 0 offen
	v_add_u32_e32 v26, v41, v62
	s_and_b64 s[14:15], vcc, s[6:7]
	v_cndmask_b32_e64 v26, v1, v26, s[14:15]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v39, v26, s[20:23], 0 offen
	v_add_u32_e32 v26, v44, v62
	s_and_b64 s[14:15], s[4:5], s[6:7]
	v_cndmask_b32_e64 v26, v1, v26, s[14:15]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v34, v26, s[20:23], 0 offen
	v_add_u32_e32 v26, v42, v62
	s_and_b64 s[14:15], s[8:9], s[6:7]
	v_cndmask_b32_e64 v26, v1, v26, s[14:15]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v30, v26, s[20:23], 0 offen
	v_add_u32_e32 v26, v46, v62
	s_and_b64 s[14:15], s[10:11], s[6:7]
	v_cndmask_b32_e64 v26, v1, v26, s[14:15]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v24, v26, s[20:23], 0 offen
	v_add_u32_e32 v24, v41, v100
	s_and_b64 s[14:15], vcc, s[2:3]
	v_cndmask_b32_e64 v24, v1, v24, s[14:15]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v37, v24, s[20:23], 0 offen
	v_add_u32_e32 v24, v44, v100
	s_and_b64 s[14:15], s[4:5], s[2:3]
	v_cndmask_b32_e64 v24, v1, v24, s[14:15]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v33, v24, s[20:23], 0 offen
	v_add_u32_e32 v24, v42, v100
	s_and_b64 s[14:15], s[8:9], s[2:3]
	v_cndmask_b32_e64 v24, v1, v24, s[14:15]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v27, v24, s[20:23], 0 offen
	v_add_u32_e32 v24, v46, v100
	s_and_b64 s[14:15], s[10:11], s[2:3]
	v_cndmask_b32_e64 v24, v1, v24, s[14:15]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v23, v24, s[20:23], 0 offen
	v_add_u32_e32 v23, v41, v63
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v23, v1, v23, vcc
	v_lshlrev_b32_e32 v23, 1, v23
	buffer_store_short v35, v23, s[20:23], 0 offen
	v_add_u32_e32 v23, v44, v63
	s_and_b64 vcc, s[4:5], s[0:1]
	v_cndmask_b32_e32 v23, v1, v23, vcc
	v_lshlrev_b32_e32 v23, 1, v23
	buffer_store_short v31, v23, s[20:23], 0 offen
	v_add_u32_e32 v23, v42, v63
	s_and_b64 vcc, s[8:9], s[0:1]
	v_cndmask_b32_e32 v23, v1, v23, vcc
	v_lshlrev_b32_e32 v23, 1, v23
	buffer_store_short v25, v23, s[20:23], 0 offen
	v_add_u32_e32 v23, v46, v63
	s_and_b64 vcc, s[10:11], s[0:1]
	v_cndmask_b32_e32 v23, v1, v23, vcc
	v_or_b32_e32 v24, 48, v0
	v_mov_b32_e32 v25, s25
	v_lshlrev_b32_e32 v23, 1, v23
	v_cmp_gt_i64_e32 vcc, s[12:13], v[24:25]
	v_mad_u64_u32 v[24:25], s[4:5], s72, 48, v[2:3]
	buffer_store_short v21, v23, s[20:23], 0 offen
	v_add_u32_e32 v21, v24, v115
	s_and_b64 s[4:5], vcc, s[18:19]
	v_cndmask_b32_e64 v21, v1, v21, s[4:5]
	v_or_b32_e32 v26, 49, v0
	v_mov_b32_e32 v27, s25
	v_lshlrev_b32_e32 v21, 1, v21
	v_cmp_gt_i64_e64 s[4:5], s[12:13], v[26:27]
	v_mad_u64_u32 v[26:27], s[8:9], s72, 49, v[2:3]
	buffer_store_short v28, v21, s[20:23], 0 offen
	v_add_u32_e32 v21, v26, v115
	s_and_b64 s[8:9], s[4:5], s[18:19]
	v_cndmask_b32_e64 v21, v1, v21, s[8:9]
	v_lshlrev_b32_e32 v21, 1, v21
	buffer_store_short v22, v21, s[20:23], 0 offen
	v_or_b32_e32 v22, 50, v0
	v_mov_b32_e32 v23, s25
	v_cmp_gt_i64_e64 s[8:9], s[12:13], v[22:23]
	v_mad_u64_u32 v[22:23], s[10:11], s72, 50, v[2:3]
	v_add_u32_e32 v21, v22, v115
	s_and_b64 s[10:11], s[8:9], s[18:19]
	v_or_b32_e32 v28, 51, v0
	v_mov_b32_e32 v29, s25
	v_cndmask_b32_e64 v21, v1, v21, s[10:11]
	v_cmp_gt_i64_e64 s[10:11], s[12:13], v[28:29]
	v_mad_u64_u32 v[28:29], s[12:13], s72, 51, v[2:3]
	v_add_u32_e32 v0, v28, v115
	s_and_b64 s[12:13], s[10:11], s[18:19]
	v_cndmask_b32_e64 v0, v1, v0, s[12:13]
	v_lshlrev_b32_e32 v21, 1, v21
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v18, v21, s[20:23], 0 offen
	buffer_store_short v9, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v24, v119
	s_and_b64 s[12:13], vcc, s[16:17]
	v_cndmask_b32_e64 v0, v1, v0, s[12:13]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v20, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v26, v119
	s_and_b64 s[12:13], s[4:5], s[16:17]
	v_cndmask_b32_e64 v0, v1, v0, s[12:13]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v16, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v22, v119
	s_and_b64 s[12:13], s[8:9], s[16:17]
	v_cndmask_b32_e64 v0, v1, v0, s[12:13]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v12, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v28, v119
	s_and_b64 s[12:13], s[10:11], s[16:17]
	v_cndmask_b32_e64 v0, v1, v0, s[12:13]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v7, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v24, v62
	s_and_b64 s[12:13], vcc, s[6:7]
	v_cndmask_b32_e64 v0, v1, v0, s[12:13]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v19, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v26, v62
	s_and_b64 s[12:13], s[4:5], s[6:7]
	v_cndmask_b32_e64 v0, v1, v0, s[12:13]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v14, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v22, v62
	s_and_b64 s[12:13], s[8:9], s[6:7]
	v_cndmask_b32_e64 v0, v1, v0, s[12:13]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v10, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v28, v62
	s_and_b64 s[6:7], s[10:11], s[6:7]
	v_cndmask_b32_e64 v0, v1, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v5, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v24, v100
	s_and_b64 s[6:7], vcc, s[2:3]
	v_cndmask_b32_e64 v0, v1, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v17, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v26, v100
	s_and_b64 s[6:7], s[4:5], s[2:3]
	v_cndmask_b32_e64 v0, v1, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v13, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v22, v100
	s_and_b64 s[6:7], s[8:9], s[2:3]
	v_cndmask_b32_e64 v0, v1, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v8, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v28, v100
	s_and_b64 s[2:3], s[10:11], s[2:3]
	v_cndmask_b32_e64 v0, v1, v0, s[2:3]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v4, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v24, v63
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v0, v1, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v15, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v26, v63
	s_and_b64 vcc, s[4:5], s[0:1]
	v_cndmask_b32_e32 v0, v1, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v11, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v22, v63
	s_and_b64 vcc, s[8:9], s[0:1]
	v_cndmask_b32_e32 v0, v1, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v6, v0, s[20:23], 0 offen
	v_add_u32_e32 v0, v28, v63
	s_and_b64 vcc, s[10:11], s[0:1]
	v_cndmask_b32_e32 v0, v1, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v3, v0, s[20:23], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_dynamic_gemm_256x160x256
		.amdhsa_group_segment_fixed_size 159744
		.amdhsa_private_segment_fixed_size 68
		.amdhsa_kernarg_size 104
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	wave_mxfp4_dynamic_gemm_256x160x256, .Lfunc_end0-wave_mxfp4_dynamic_gemm_256x160x256

	.set wave_mxfp4_dynamic_gemm_256x160x256.num_vgpr, 256
	.set wave_mxfp4_dynamic_gemm_256x160x256.num_agpr, 0
	.set wave_mxfp4_dynamic_gemm_256x160x256.numbered_sgpr, 100
	.set wave_mxfp4_dynamic_gemm_256x160x256.num_named_barrier, 0
	.set wave_mxfp4_dynamic_gemm_256x160x256.private_seg_size, 68
	.set wave_mxfp4_dynamic_gemm_256x160x256.uses_vcc, 1
	.set wave_mxfp4_dynamic_gemm_256x160x256.uses_flat_scratch, 0
	.set wave_mxfp4_dynamic_gemm_256x160x256.has_dyn_sized_stack, 0
	.set wave_mxfp4_dynamic_gemm_256x160x256.has_recursion, 0
	.set wave_mxfp4_dynamic_gemm_256x160x256.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  generic
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 159744
    .kernarg_segment_align: 8
    .kernarg_segment_size: 104
    .max_flat_workgroup_size: 512
    .name:           wave_mxfp4_dynamic_gemm_256x160x256
    .private_segment_fixed_size: 68
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     106
    .sgpr_spill_count: 99
    .symbol:         wave_mxfp4_dynamic_gemm_256x160x256.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 16
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
