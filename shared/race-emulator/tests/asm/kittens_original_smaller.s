	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE ; -- Begin function _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE
	.globl	_Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE
	.p2align	8
	.type	_Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE,@function
_Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE: ; @_Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE
; %bb.0:
	s_cmp_lg_u32 0, -1
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dwordx2 s[10:11], s[0:1], 0x10
	s_load_dwordx2 s[8:9], s[0:1], 0x20
	s_cselect_b32 s0, 0, 0
	s_and_b32 s1, s0, -16
	s_mov_b32 s7, 0
	s_and_b32 s6, s0, 15
	s_add_i32 s1, s1, 16
	s_cmp_eq_u64 s[6:7], 0
	s_cselect_b32 s3, s0, s1
	s_add_i32 s14, s3, 0x8000
	s_and_b32 s0, s14, -16
	s_add_i32 s15, s0, 16
	s_lshl_b32 s0, s2, 15
	s_ashr_i32 s1, s0, 31
	v_lshlrev_b32_e32 v2, 3, v0
	v_lshlrev_b32_e32 v18, 4, v0
	s_and_b32 s6, s14, 15
	s_lshl_b64 s[0:1], s[0:1], 1
	v_and_b32_e32 v19, 56, v2
	v_and_b32_e32 v28, 0x1f80, v18
	s_waitcnt lgkmcnt(0)
	s_add_u32 s12, s4, s0
	v_or_b32_e32 v2, v28, v19
	s_addc_u32 s13, s5, s1
	v_lshlrev_b32_e32 v20, 1, v2
	v_mov_b32_e32 v21, 0
	v_lshl_add_u64 v[2:3], s[12:13], 0, v[20:21]
	v_or_b32_e32 v22, 0x4000, v20
	v_mov_b32_e32 v23, v21
	v_lshl_or_b32 v19, v19, 1, v28
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[6:7], s[12:13], 0, v[22:23]
	v_or_b32_e32 v24, 0x8000, v20
	v_mov_b32_e32 v25, v21
	v_add_u32_e32 v28, s3, v19
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	v_lshl_add_u64 v[10:11], s[12:13], 0, v[24:25]
	v_or_b32_e32 v26, 0xc000, v20
	v_mov_b32_e32 v27, v21
	v_or_b32_e32 v29, 8, v19
	v_lshrrev_b32_e32 v31, 4, v28
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[10:11], off

	;;#ASMEND
	v_lshl_add_u64 v[14:15], s[12:13], 0, v[26:27]
	v_add_u32_e32 v30, s3, v29
	v_and_b32_e32 v31, 0x78, v31
	;;#ASMSTART
	global_load_dwordx4 v[14:17], v[14:15], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v32, v31, v28
	;;#ASMSTART
	ds_write_b64 v32, v[2:3]

	;;#ASMEND
	v_lshrrev_b32_e32 v2, 4, v30
	v_and_b32_e32 v2, 0x78, v2
	v_xor_b32_e32 v3, v2, v30
	;;#ASMSTART
	ds_write_b64 v3, v[4:5]

	;;#ASMEND
	v_add_u32_e32 v3, 0x2000, v28
	v_xor_b32_e32 v3, v31, v3
	;;#ASMSTART
	ds_write_b64 v3, v[6:7]

	;;#ASMEND
	v_add_u32_e32 v3, 0x2000, v30
	v_xor_b32_e32 v3, v2, v3
	;;#ASMSTART
	ds_write_b64 v3, v[8:9]

	;;#ASMEND
	v_add_u32_e32 v3, 0x4000, v28
	v_xor_b32_e32 v3, v31, v3
	;;#ASMSTART
	ds_write_b64 v3, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v3, 0x4000, v30
	v_xor_b32_e32 v3, v2, v3
	;;#ASMSTART
	ds_write_b64 v3, v[12:13]

	;;#ASMEND
	v_add_u32_e32 v3, 0x6000, v28
	v_xor_b32_e32 v3, v31, v3
	;;#ASMSTART
	ds_write_b64 v3, v[14:15]

	;;#ASMEND
	v_add_u32_e32 v3, 0x6000, v30
	s_cmp_eq_u64 s[6:7], 0
	v_xor_b32_e32 v2, v2, v3
	;;#ASMSTART
	ds_write_b64 v2, v[16:17]

	;;#ASMEND
	v_lshl_add_u64 v[2:3], s[10:11], 0, v[20:21]
	s_cselect_b32 s12, s14, s15
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[6:7], s[10:11], 0, v[22:23]
	v_add_u32_e32 v19, s12, v19
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	v_lshl_add_u64 v[10:11], s[10:11], 0, v[24:25]
	v_lshrrev_b32_e32 v21, 4, v19
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[10:11], off

	;;#ASMEND
	v_lshl_add_u64 v[14:15], s[10:11], 0, v[26:27]
	v_add_u32_e32 v20, s12, v29
	v_and_b32_e32 v21, 0x78, v21
	;;#ASMSTART
	global_load_dwordx4 v[14:17], v[14:15], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v22, v21, v19
	;;#ASMSTART
	ds_write_b64 v22, v[2:3]

	;;#ASMEND
	v_lshrrev_b32_e32 v2, 4, v20
	v_and_b32_e32 v2, 0x78, v2
	v_xor_b32_e32 v3, v2, v20
	;;#ASMSTART
	ds_write_b64 v3, v[4:5]

	;;#ASMEND
	v_add_u32_e32 v3, 0x2000, v19
	v_xor_b32_e32 v3, v21, v3
	;;#ASMSTART
	ds_write_b64 v3, v[6:7]

	;;#ASMEND
	v_add_u32_e32 v3, 0x2000, v20
	v_xor_b32_e32 v3, v2, v3
	;;#ASMSTART
	ds_write_b64 v3, v[8:9]

	;;#ASMEND
	v_add_u32_e32 v3, 0x4000, v19
	v_xor_b32_e32 v3, v21, v3
	;;#ASMSTART
	ds_write_b64 v3, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v3, 0x4000, v20
	v_xor_b32_e32 v3, v2, v3
	;;#ASMSTART
	ds_write_b64 v3, v[12:13]

	;;#ASMEND
	v_add_u32_e32 v3, 0x6000, v19
	v_xor_b32_e32 v3, v21, v3
	;;#ASMSTART
	ds_write_b64 v3, v[14:15]

	;;#ASMEND
	v_add_u32_e32 v3, 0x6000, v20
	v_xor_b32_e32 v2, v2, v3
	;;#ASMSTART
	ds_write_b64 v2, v[16:17]

	;;#ASMEND
	v_lshrrev_b32_e32 v1, 8, v0
	v_cmp_eq_u32_e32 vcc, 1, v1
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:                                ; %.critedge
	s_or_b64 exec, exec, s[6:7]
	s_add_u32 s0, s4, s0
	s_addc_u32 s1, s5, s1
	s_add_u32 s4, s0, 0x80
	s_addc_u32 s0, s1, 0
	s_or_b32 s5, s0, 0xc1000000
	v_and_b32_e32 v2, 0x70, v18
	v_lshlrev_b32_e32 v3, 5, v0
	s_movk_i32 s0, 0x3f00
	v_and_or_b32 v19, v3, s0, v2
	s_mov_b32 s7, 0x110000
	s_mov_b32 s6, 0x10000
	v_or_b32_e32 v190, 0x8000, v19
	v_or_b32_e32 v135, 0x4000, v19
	buffer_load_dwordx4 v[6:9], v19, s[4:7], 0 offen
	buffer_load_dwordx4 v[2:5], v135, s[4:7], 0 offen
	v_or_b32_e32 v191, 0xc000, v19
	buffer_load_dwordx4 v[14:17], v190, s[4:7], 0 offen
	buffer_load_dwordx4 v[10:13], v191, s[4:7], 0 offen
	v_and_b32_e32 v134, 15, v0
	v_lshlrev_b32_e32 v20, 7, v134
	v_lshl_or_b32 v20, v1, 13, v20
	v_lshrrev_b32_e32 v21, 1, v0
	v_and_b32_e32 v192, 24, v21
	v_add_u32_e32 v198, s3, v20
	v_add_u32_e32 v20, v198, v192
	v_lshrrev_b32_e32 v21, 4, v20
	v_and_b32_e32 v21, 0x78, v21
	v_add_u32_e32 v206, 0x4000, v198
	v_xor_b32_e32 v214, v21, v20
	v_add_u32_e32 v20, v206, v192
	;;#ASMSTART
	ds_read_b64 v[32:33], v214 offset:0

	;;#ASMEND
	v_lshrrev_b32_e32 v21, 4, v20
	;;#ASMSTART
	ds_read_b64 v[48:49], v214 offset:0x800

	;;#ASMEND
	v_and_b32_e32 v21, 0x78, v21
	;;#ASMSTART
	ds_read_b64 v[64:65], v214 offset:0x1000

	;;#ASMEND
	v_xor_b32_e32 v215, v21, v20
	v_lshlrev_b32_e32 v20, 7, v0
	;;#ASMSTART
	ds_read_b64 v[80:81], v214 offset:0x1800

	;;#ASMEND
	v_and_b32_e32 v20, 0x6780, v20
	;;#ASMSTART
	ds_read_b64 v[96:97], v215 offset:0

	;;#ASMEND
	v_add_u32_e32 v193, s12, v20
	;;#ASMSTART
	ds_read_b64 v[112:113], v215 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v20, v193, v192
	;;#ASMSTART
	ds_read_b64 v[128:129], v215 offset:0x1000

	;;#ASMEND
	v_lshrrev_b32_e32 v21, 4, v20
	;;#ASMSTART
	ds_read_b64 v[132:133], v215 offset:0x1800

	;;#ASMEND
	v_and_b32_e32 v21, 0x78, v21
	v_xor_b32_e32 v216, v21, v20
	;;#ASMSTART
	ds_read_b64 v[136:137], v216 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v216 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v216 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v216 offset:0x1800

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[20:23], v[32:33], v[136:137], 0
	v_mfma_f32_16x16x16_bf16 v[24:27], v[32:33], v[140:141], 0
	v_mfma_f32_16x16x16_bf16 v[28:31], v[32:33], v[144:145], 0
	v_mfma_f32_16x16x16_bf16 v[32:35], v[32:33], v[148:149], 0
	v_mfma_f32_16x16x16_bf16 v[36:39], v[48:49], v[136:137], 0
	v_mfma_f32_16x16x16_bf16 v[40:43], v[48:49], v[140:141], 0
	v_mfma_f32_16x16x16_bf16 v[44:47], v[48:49], v[144:145], 0
	v_mfma_f32_16x16x16_bf16 v[48:51], v[48:49], v[148:149], 0
	v_mfma_f32_16x16x16_bf16 v[52:55], v[64:65], v[136:137], 0
	v_mfma_f32_16x16x16_bf16 v[56:59], v[64:65], v[140:141], 0
	v_mfma_f32_16x16x16_bf16 v[60:63], v[64:65], v[144:145], 0
	v_mfma_f32_16x16x16_bf16 v[64:67], v[64:65], v[148:149], 0
	v_mfma_f32_16x16x16_bf16 v[68:71], v[80:81], v[136:137], 0
	v_mfma_f32_16x16x16_bf16 v[72:75], v[80:81], v[140:141], 0
	v_mfma_f32_16x16x16_bf16 v[76:79], v[80:81], v[144:145], 0
	v_mfma_f32_16x16x16_bf16 v[80:83], v[80:81], v[148:149], 0
	v_mfma_f32_16x16x16_bf16 v[84:87], v[96:97], v[136:137], 0
	v_mfma_f32_16x16x16_bf16 v[88:91], v[96:97], v[140:141], 0
	v_mfma_f32_16x16x16_bf16 v[92:95], v[96:97], v[144:145], 0
	v_mfma_f32_16x16x16_bf16 v[96:99], v[96:97], v[148:149], 0
	v_mfma_f32_16x16x16_bf16 v[100:103], v[112:113], v[136:137], 0
	v_mfma_f32_16x16x16_bf16 v[104:107], v[112:113], v[140:141], 0
	v_mfma_f32_16x16x16_bf16 v[108:111], v[112:113], v[144:145], 0
	v_mfma_f32_16x16x16_bf16 v[112:115], v[112:113], v[148:149], 0
	v_mfma_f32_16x16x16_bf16 v[116:119], v[128:129], v[136:137], 0
	v_mfma_f32_16x16x16_bf16 v[120:123], v[128:129], v[140:141], 0
	v_mfma_f32_16x16x16_bf16 v[124:127], v[128:129], v[144:145], 0
	v_mfma_f32_16x16x16_bf16 v[128:131], v[128:129], v[148:149], 0
	v_mfma_f32_16x16x16_bf16 v[136:139], v[132:133], v[136:137], 0
	v_mfma_f32_16x16x16_bf16 v[140:143], v[132:133], v[140:141], 0
	v_mfma_f32_16x16x16_bf16 v[144:147], v[132:133], v[144:145], 0
	v_mfma_f32_16x16x16_bf16 v[148:151], v[132:133], v[148:149], 0
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v166, 32, v192
	v_add_u32_e32 v132, v193, v166
	v_lshrrev_b32_e32 v133, 4, v132
	v_and_b32_e32 v133, 0x78, v133
	v_xor_b32_e32 v217, v133, v132
	;;#ASMSTART
	ds_read_b64 v[132:133], v217 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v217 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v158, v198, v166
	;;#ASMSTART
	ds_read_b64 v[154:155], v217 offset:0x1000

	;;#ASMEND
	v_lshrrev_b32_e32 v159, 4, v158
	;;#ASMSTART
	ds_read_b64 v[156:157], v217 offset:0x1800

	;;#ASMEND
	v_and_b32_e32 v159, 0x78, v159
	v_xor_b32_e32 v218, v159, v158
	;;#ASMSTART
	ds_read_b64 v[158:159], v218 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v218 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v166, v206, v166
	;;#ASMSTART
	ds_read_b64 v[162:163], v218 offset:0x1000

	;;#ASMEND
	v_lshrrev_b32_e32 v167, 4, v166
	;;#ASMSTART
	ds_read_b64 v[164:165], v218 offset:0x1800

	;;#ASMEND
	v_and_b32_e32 v167, 0x78, v167
	v_xor_b32_e32 v219, v167, v166
	;;#ASMSTART
	ds_read_b64 v[166:167], v219 offset:0

	;;#ASMEND
	v_or_b32_e32 v194, 64, v192
	;;#ASMSTART
	ds_read_b64 v[168:169], v219 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v174, v193, v194
	;;#ASMSTART
	ds_read_b64 v[170:171], v219 offset:0x1000

	;;#ASMEND
	v_lshrrev_b32_e32 v175, 4, v174
	;;#ASMSTART
	ds_read_b64 v[172:173], v219 offset:0x1800

	;;#ASMEND
	v_and_b32_e32 v175, 0x78, v175
	v_xor_b32_e32 v220, v175, v174
	;;#ASMSTART
	ds_read_b64 v[174:175], v220 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[176:177], v220 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v182, v198, v194
	;;#ASMSTART
	ds_read_b64 v[178:179], v220 offset:0x1000

	;;#ASMEND
	v_lshrrev_b32_e32 v183, 4, v182
	;;#ASMSTART
	ds_read_b64 v[180:181], v220 offset:0x1800

	;;#ASMEND
	v_and_b32_e32 v183, 0x78, v183
	v_xor_b32_e32 v221, v183, v182
	;;#ASMSTART
	ds_read_b64 v[182:183], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[184:185], v221 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[186:187], v221 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[188:189], v221 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[20:23], v[158:159], v[132:133], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[158:159], v[152:153], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[158:159], v[154:155], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[158:159], v[156:157], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[160:161], v[132:133], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[160:161], v[152:153], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[160:161], v[154:155], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[160:161], v[156:157], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[162:163], v[132:133], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[162:163], v[152:153], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[162:163], v[154:155], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[162:163], v[156:157], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[164:165], v[132:133], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[164:165], v[152:153], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[164:165], v[154:155], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[164:165], v[156:157], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[166:167], v[132:133], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[166:167], v[152:153], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[166:167], v[154:155], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[166:167], v[156:157], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[168:169], v[132:133], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[168:169], v[152:153], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[168:169], v[154:155], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[168:169], v[156:157], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[170:171], v[132:133], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[170:171], v[152:153], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[124:127], v[170:171], v[154:155], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[128:131], v[170:171], v[156:157], v[128:131]
	v_mfma_f32_16x16x16_bf16 v[136:139], v[172:173], v[132:133], v[136:139]
	v_mfma_f32_16x16x16_bf16 v[140:143], v[172:173], v[152:153], v[140:143]
	v_mfma_f32_16x16x16_bf16 v[144:147], v[172:173], v[154:155], v[144:147]
	v_mfma_f32_16x16x16_bf16 v[148:151], v[172:173], v[156:157], v[148:151]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s4, s10, 0x80
	s_addc_u32 s0, s11, 0
	s_or_b32 s5, s0, 0xc1000000
	buffer_load_dwordx4 v[152:155], v19, s[4:7], 0 offen
	buffer_load_dwordx4 v[156:159], v135, s[4:7], 0 offen
	buffer_load_dwordx4 v[160:163], v190, s[4:7], 0 offen
	buffer_load_dwordx4 v[164:167], v191, s[4:7], 0 offen
	v_add_u32_e32 v19, v206, v194
	v_lshrrev_b32_e32 v132, 4, v19
	v_and_b32_e32 v132, 0x78, v132
	v_xor_b32_e32 v135, v132, v19
	;;#ASMSTART
	ds_read_b64 v[132:133], v135 offset:0

	;;#ASMEND
	v_or_b32_e32 v19, 0x60, v192
	;;#ASMSTART
	ds_read_b64 v[168:169], v135 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v190, v193, v19
	;;#ASMSTART
	ds_read_b64 v[170:171], v135 offset:0x1000

	;;#ASMEND
	v_lshrrev_b32_e32 v191, 4, v190
	;;#ASMSTART
	ds_read_b64 v[172:173], v135 offset:0x1800

	;;#ASMEND
	v_and_b32_e32 v191, 0x78, v191
	v_xor_b32_e32 v222, v191, v190
	;;#ASMSTART
	ds_read_b64 v[190:191], v222 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[192:193], v222 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v198, v198, v19
	;;#ASMSTART
	ds_read_b64 v[194:195], v222 offset:0x1000

	;;#ASMEND
	v_lshrrev_b32_e32 v199, 4, v198
	;;#ASMSTART
	ds_read_b64 v[196:197], v222 offset:0x1800

	;;#ASMEND
	v_and_b32_e32 v199, 0x78, v199
	v_xor_b32_e32 v223, v199, v198
	;;#ASMSTART
	ds_read_b64 v[198:199], v223 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[200:201], v223 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v19, v206, v19
	;;#ASMSTART
	ds_read_b64 v[202:203], v223 offset:0x1000

	;;#ASMEND
	v_lshrrev_b32_e32 v206, 4, v19
	;;#ASMSTART
	ds_read_b64 v[204:205], v223 offset:0x1800

	;;#ASMEND
	v_and_b32_e32 v206, 0x78, v206
	v_xor_b32_e32 v224, v206, v19
	;;#ASMSTART
	ds_read_b64 v[206:207], v224 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v224 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[210:211], v224 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[212:213], v224 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[20:23], v[182:183], v[174:175], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[182:183], v[176:177], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[182:183], v[178:179], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[182:183], v[180:181], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[184:185], v[174:175], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[184:185], v[176:177], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[184:185], v[178:179], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[184:185], v[180:181], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[186:187], v[174:175], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[186:187], v[176:177], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[186:187], v[178:179], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[186:187], v[180:181], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[188:189], v[174:175], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[188:189], v[176:177], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[188:189], v[178:179], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[188:189], v[180:181], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[132:133], v[174:175], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[132:133], v[176:177], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[132:133], v[178:179], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[132:133], v[180:181], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[168:169], v[174:175], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[168:169], v[176:177], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[168:169], v[178:179], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[168:169], v[180:181], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[170:171], v[174:175], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[170:171], v[176:177], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[124:127], v[170:171], v[178:179], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[128:131], v[170:171], v[180:181], v[128:131]
	v_mfma_f32_16x16x16_bf16 v[136:139], v[172:173], v[174:175], v[136:139]
	v_mfma_f32_16x16x16_bf16 v[140:143], v[172:173], v[176:177], v[140:143]
	v_mfma_f32_16x16x16_bf16 v[144:147], v[172:173], v[178:179], v[144:147]
	v_mfma_f32_16x16x16_bf16 v[148:151], v[172:173], v[180:181], v[148:151]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	v_and_b32_e32 v18, 0x1ff0, v18
	v_add_u32_e32 v19, s3, v18
	v_or_b32_e32 v132, 8, v18
	v_lshrrev_b32_e32 v168, 4, v19
	v_add_u32_e32 v133, s3, v132
	v_and_b32_e32 v168, 0x78, v168
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v169, v168, v19
	s_waitcnt vmcnt(7)
	;;#ASMSTART
	ds_write_b64 v169, v[6:7]

	;;#ASMEND
	v_lshrrev_b32_e32 v6, 4, v133
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v7, v6, v133
	;;#ASMSTART
	ds_write_b64 v7, v[8:9]

	;;#ASMEND
	v_add_u32_e32 v7, 0x2000, v19
	v_xor_b32_e32 v7, v168, v7
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	ds_write_b64 v7, v[2:3]

	;;#ASMEND
	v_add_u32_e32 v2, 0x2000, v133
	v_xor_b32_e32 v2, v6, v2
	;;#ASMSTART
	ds_write_b64 v2, v[4:5]

	;;#ASMEND
	v_add_u32_e32 v2, 0x4000, v19
	v_xor_b32_e32 v2, v168, v2
	s_waitcnt vmcnt(5)
	;;#ASMSTART
	ds_write_b64 v2, v[14:15]

	;;#ASMEND
	v_add_u32_e32 v2, 0x4000, v133
	v_xor_b32_e32 v2, v6, v2
	;;#ASMSTART
	ds_write_b64 v2, v[16:17]

	;;#ASMEND
	v_add_u32_e32 v2, 0x6000, v19
	v_xor_b32_e32 v2, v168, v2
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	ds_write_b64 v2, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v2, 0x6000, v133
	v_xor_b32_e32 v2, v6, v2
	;;#ASMSTART
	ds_write_b64 v2, v[12:13]

	;;#ASMEND
	v_add_u32_e32 v2, s12, v18
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_add_u32_e32 v3, s12, v132
	v_xor_b32_e32 v5, v4, v2
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	ds_write_b64 v5, v[152:153]

	;;#ASMEND
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v6, v5, v3
	;;#ASMSTART
	ds_write_b64 v6, v[154:155]

	;;#ASMEND
	v_add_u32_e32 v6, 0x2000, v2
	v_xor_b32_e32 v6, v4, v6
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	ds_write_b64 v6, v[156:157]

	;;#ASMEND
	v_add_u32_e32 v6, 0x2000, v3
	v_xor_b32_e32 v6, v5, v6
	;;#ASMSTART
	ds_write_b64 v6, v[158:159]

	;;#ASMEND
	v_add_u32_e32 v6, 0x4000, v2
	v_xor_b32_e32 v6, v4, v6
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v6, v[160:161]

	;;#ASMEND
	v_add_u32_e32 v6, 0x4000, v3
	v_add_u32_e32 v2, 0x6000, v2
	v_xor_b32_e32 v6, v5, v6
	;;#ASMSTART
	ds_write_b64 v6, v[162:163]

	;;#ASMEND
	v_xor_b32_e32 v2, v4, v2
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v2, v[164:165]

	;;#ASMEND
	v_add_u32_e32 v2, 0x6000, v3
	v_xor_b32_e32 v2, v5, v2
	;;#ASMSTART
	ds_write_b64 v2, v[166:167]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[2:5], v[198:199], v[190:191], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[198:199], v[192:193], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[198:199], v[194:195], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[198:199], v[196:197], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[200:201], v[190:191], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[200:201], v[192:193], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[200:201], v[194:195], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[200:201], v[196:197], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[202:203], v[190:191], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[202:203], v[192:193], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[202:203], v[194:195], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[202:203], v[196:197], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[204:205], v[190:191], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[204:205], v[192:193], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[204:205], v[194:195], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[204:205], v[196:197], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[206:207], v[190:191], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[206:207], v[192:193], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[206:207], v[194:195], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[206:207], v[196:197], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[208:209], v[190:191], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[208:209], v[192:193], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[208:209], v[194:195], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[208:209], v[196:197], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[210:211], v[190:191], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[210:211], v[192:193], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[210:211], v[194:195], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[210:211], v[196:197], v[128:131]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[212:213], v[190:191], v[136:139]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[212:213], v[192:193], v[140:143]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[212:213], v[194:195], v[144:147]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[212:213], v[196:197], v[148:151]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[130:131], v216 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v216 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v216 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v216 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v214 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v214 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v214 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v214 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v215 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v215 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v215 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[154:155], v215 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[2:5], v[140:141], v[130:131], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[140:141], v[132:133], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[140:141], v[136:137], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[140:141], v[138:139], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[142:143], v[130:131], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[142:143], v[132:133], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[142:143], v[136:137], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[142:143], v[138:139], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[144:145], v[130:131], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[144:145], v[132:133], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[144:145], v[136:137], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[144:145], v[138:139], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[146:147], v[130:131], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[146:147], v[132:133], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[146:147], v[136:137], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[146:147], v[138:139], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[148:149], v[130:131], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[148:149], v[132:133], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[148:149], v[136:137], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[148:149], v[138:139], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[150:151], v[130:131], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[150:151], v[132:133], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[150:151], v[136:137], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[150:151], v[138:139], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[152:153], v[130:131], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[152:153], v[132:133], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[152:153], v[136:137], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[152:153], v[138:139], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[154:155], v[130:131], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[154:155], v[132:133], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[154:155], v[136:137], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[154:155], v[138:139], v[126:129]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[130:131], v217 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v217 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v217 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v217 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v218 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v218 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v218 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v218 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v219 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v219 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v219 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[154:155], v219 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[2:5], v[140:141], v[130:131], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[140:141], v[132:133], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[140:141], v[136:137], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[140:141], v[138:139], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[142:143], v[130:131], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[142:143], v[132:133], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[142:143], v[136:137], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[142:143], v[138:139], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[144:145], v[130:131], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[144:145], v[132:133], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[144:145], v[136:137], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[144:145], v[138:139], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[146:147], v[130:131], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[146:147], v[132:133], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[146:147], v[136:137], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[146:147], v[138:139], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[148:149], v[130:131], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[148:149], v[132:133], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[148:149], v[136:137], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[148:149], v[138:139], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[150:151], v[130:131], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[150:151], v[132:133], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[150:151], v[136:137], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[150:151], v[138:139], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[152:153], v[130:131], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[152:153], v[132:133], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[152:153], v[136:137], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[152:153], v[138:139], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[154:155], v[130:131], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[154:155], v[132:133], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[154:155], v[136:137], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[154:155], v[138:139], v[126:129]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[180:181], v220 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[184:185], v220 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[188:189], v220 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[192:193], v220 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[130:131], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v221 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v221 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v221 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v135 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v135 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[176:177], v135 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[194:195], v135 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[196:197], v222 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[198:199], v222 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[200:201], v222 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[202:203], v222 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v223 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v223 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v223 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[210:211], v223 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[212:213], v224 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[214:215], v224 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[216:217], v224 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[218:219], v224 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[2:5], v[130:131], v[180:181], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[130:131], v[184:185], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[130:131], v[188:189], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[130:131], v[192:193], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[132:133], v[180:181], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[132:133], v[184:185], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[132:133], v[188:189], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[132:133], v[192:193], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[136:137], v[180:181], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[136:137], v[184:185], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[136:137], v[188:189], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[136:137], v[192:193], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[138:139], v[180:181], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[138:139], v[184:185], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[138:139], v[188:189], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[138:139], v[192:193], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[130:133], v[144:145], v[180:181], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[136:139], v[144:145], v[184:185], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[140:143], v[144:145], v[188:189], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[144:147], v[144:145], v[192:193], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[148:151], v[160:161], v[180:181], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[152:155], v[160:161], v[184:185], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[156:159], v[160:161], v[188:189], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[160:163], v[160:161], v[192:193], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[164:167], v[176:177], v[180:181], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[168:171], v[176:177], v[184:185], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[172:175], v[176:177], v[188:189], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[176:179], v[176:177], v[192:193], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[180:183], v[194:195], v[180:181], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[184:187], v[194:195], v[184:185], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[188:191], v[194:195], v[188:189], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[192:195], v[194:195], v[192:193], v[126:129]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[126:129], v[204:205], v[196:197], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[204:205], v[198:199], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[204:205], v[200:201], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[204:205], v[202:203], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[206:207], v[196:197], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[206:207], v[198:199], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[206:207], v[200:201], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[206:207], v[202:203], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[208:209], v[196:197], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[208:209], v[198:199], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[208:209], v[200:201], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[208:209], v[202:203], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[210:211], v[196:197], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[210:211], v[198:199], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[210:211], v[200:201], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[210:211], v[202:203], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[212:213], v[196:197], v[130:133]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[212:213], v[198:199], v[136:139]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[212:213], v[200:201], v[140:143]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[212:213], v[202:203], v[144:147]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[214:215], v[196:197], v[148:151]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[214:215], v[198:199], v[152:155]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[214:215], v[200:201], v[156:159]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[214:215], v[202:203], v[160:163]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[216:217], v[196:197], v[164:167]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[216:217], v[198:199], v[168:171]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[216:217], v[200:201], v[172:175]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[216:217], v[202:203], v[176:179]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[218:219], v[196:197], v[180:183]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[218:219], v[198:199], v[184:187]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[218:219], v[200:201], v[188:191]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[218:219], v[202:203], v[192:195]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	s_barrier
.LBB0_4:
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v130, 0xc0, v0
	s_lshl_b32 s0, s2, 16
	v_lshlrev_b32_e32 v1, 14, v1
	v_or3_b32 v130, s0, v1, v130
	v_lshlrev_b32_e32 v0, 6, v0
	s_movk_i32 s0, 0xc00
	v_ashrrev_i32_e32 v131, 31, v130
	v_and_or_b32 v0, v0, s0, v134
	v_lshl_add_u64 v[132:133], v[130:131], 1, s[8:9]
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[134:135], v[132:133], 0, v[0:1]
	global_store_short_d16_hi v[134:135], v126, off
	global_store_short_d16_hi v[134:135], v127, off offset:512
	global_store_short_d16_hi v[134:135], v128, off offset:1024
	global_store_short_d16_hi v[134:135], v129, off offset:1536
	global_store_short_d16_hi v[134:135], v122, off offset:32
	global_store_short_d16_hi v[134:135], v123, off offset:544
	global_store_short_d16_hi v[134:135], v124, off offset:1056
	global_store_short_d16_hi v[134:135], v125, off offset:1568
	global_store_short_d16_hi v[134:135], v118, off offset:64
	global_store_short_d16_hi v[134:135], v119, off offset:576
	global_store_short_d16_hi v[134:135], v120, off offset:1088
	global_store_short_d16_hi v[134:135], v121, off offset:1600
	global_store_short_d16_hi v[134:135], v114, off offset:96
	global_store_short_d16_hi v[134:135], v115, off offset:608
	global_store_short_d16_hi v[134:135], v116, off offset:1120
	global_store_short_d16_hi v[134:135], v117, off offset:1632
	v_or_b32_e32 v114, 0x2000, v0
	v_mov_b32_e32 v115, v1
	v_lshl_add_u64 v[116:117], v[132:133], 0, v[114:115]
	global_store_short_d16_hi v[116:117], v110, off
	v_or_b32_e32 v116, 0x2200, v0
	v_mov_b32_e32 v117, v1
	v_lshl_add_u64 v[118:119], v[132:133], 0, v[116:117]
	global_store_short_d16_hi v[118:119], v111, off
	v_or_b32_e32 v110, 0x2400, v0
	v_mov_b32_e32 v111, v1
	v_lshl_add_u64 v[118:119], v[132:133], 0, v[110:111]
	global_store_short_d16_hi v[118:119], v112, off
	v_or_b32_e32 v118, 0x2600, v0
	v_mov_b32_e32 v119, v1
	v_lshl_add_u64 v[120:121], v[132:133], 0, v[118:119]
	global_store_short_d16_hi v[120:121], v113, off
	v_or_b32_e32 v112, 0x2020, v0
	v_mov_b32_e32 v113, v1
	v_lshl_add_u64 v[120:121], v[132:133], 0, v[112:113]
	global_store_short_d16_hi v[120:121], v106, off
	v_or_b32_e32 v120, 0x2220, v0
	v_mov_b32_e32 v121, v1
	v_lshl_add_u64 v[122:123], v[132:133], 0, v[120:121]
	global_store_short_d16_hi v[122:123], v107, off
	v_or_b32_e32 v106, 0x2420, v0
	v_mov_b32_e32 v107, v1
	v_lshl_add_u64 v[122:123], v[132:133], 0, v[106:107]
	global_store_short_d16_hi v[122:123], v108, off
	v_or_b32_e32 v122, 0x2620, v0
	v_mov_b32_e32 v123, v1
	v_lshl_add_u64 v[124:125], v[132:133], 0, v[122:123]
	global_store_short_d16_hi v[124:125], v109, off
	v_or_b32_e32 v108, 0x2040, v0
	v_mov_b32_e32 v109, v1
	v_lshl_add_u64 v[124:125], v[132:133], 0, v[108:109]
	global_store_short_d16_hi v[124:125], v102, off
	v_or_b32_e32 v124, 0x2240, v0
	v_mov_b32_e32 v125, v1
	v_lshl_add_u64 v[126:127], v[132:133], 0, v[124:125]
	global_store_short_d16_hi v[126:127], v103, off
	v_or_b32_e32 v102, 0x2440, v0
	v_mov_b32_e32 v103, v1
	v_lshl_add_u64 v[126:127], v[132:133], 0, v[102:103]
	global_store_short_d16_hi v[126:127], v104, off
	v_or_b32_e32 v126, 0x2640, v0
	v_mov_b32_e32 v127, v1
	v_lshl_add_u64 v[128:129], v[132:133], 0, v[126:127]
	global_store_short_d16_hi v[128:129], v105, off
	v_or_b32_e32 v104, 0x2060, v0
	v_mov_b32_e32 v105, v1
	v_lshl_add_u64 v[128:129], v[132:133], 0, v[104:105]
	global_store_short_d16_hi v[128:129], v98, off
	v_or_b32_e32 v128, 0x2260, v0
	v_mov_b32_e32 v129, v1
	v_lshl_add_u64 v[134:135], v[132:133], 0, v[128:129]
	global_store_short_d16_hi v[134:135], v99, off
	v_or_b32_e32 v98, 0x2460, v0
	v_mov_b32_e32 v99, v1
	v_lshl_add_u64 v[134:135], v[132:133], 0, v[98:99]
	global_store_short_d16_hi v[134:135], v100, off
	v_or_b32_e32 v134, 0x2660, v0
	v_mov_b32_e32 v135, v1
	v_lshl_add_u64 v[136:137], v[132:133], 0, v[134:135]
	global_store_short_d16_hi v[136:137], v101, off
	v_or_b32_e32 v100, 0x4000, v0
	v_mov_b32_e32 v101, v1
	v_lshl_add_u64 v[136:137], v[132:133], 0, v[100:101]
	global_store_short_d16_hi v[136:137], v94, off
	v_or_b32_e32 v136, 0x4200, v0
	v_mov_b32_e32 v137, v1
	v_lshl_add_u64 v[138:139], v[132:133], 0, v[136:137]
	global_store_short_d16_hi v[138:139], v95, off
	v_or_b32_e32 v94, 0x4400, v0
	v_mov_b32_e32 v95, v1
	v_lshl_add_u64 v[138:139], v[132:133], 0, v[94:95]
	global_store_short_d16_hi v[138:139], v96, off
	v_or_b32_e32 v138, 0x4600, v0
	v_mov_b32_e32 v139, v1
	v_lshl_add_u64 v[140:141], v[132:133], 0, v[138:139]
	global_store_short_d16_hi v[140:141], v97, off
	v_or_b32_e32 v96, 0x4020, v0
	v_mov_b32_e32 v97, v1
	v_lshl_add_u64 v[140:141], v[132:133], 0, v[96:97]
	global_store_short_d16_hi v[140:141], v90, off
	v_or_b32_e32 v140, 0x4220, v0
	v_mov_b32_e32 v141, v1
	v_lshl_add_u64 v[142:143], v[132:133], 0, v[140:141]
	global_store_short_d16_hi v[142:143], v91, off
	v_or_b32_e32 v90, 0x4420, v0
	v_mov_b32_e32 v91, v1
	v_lshl_add_u64 v[142:143], v[132:133], 0, v[90:91]
	global_store_short_d16_hi v[142:143], v92, off
	v_or_b32_e32 v142, 0x4620, v0
	v_mov_b32_e32 v143, v1
	v_lshl_add_u64 v[144:145], v[132:133], 0, v[142:143]
	global_store_short_d16_hi v[144:145], v93, off
	v_or_b32_e32 v92, 0x4040, v0
	v_mov_b32_e32 v93, v1
	v_lshl_add_u64 v[144:145], v[132:133], 0, v[92:93]
	global_store_short_d16_hi v[144:145], v86, off
	v_or_b32_e32 v144, 0x4240, v0
	v_mov_b32_e32 v145, v1
	v_lshl_add_u64 v[146:147], v[132:133], 0, v[144:145]
	global_store_short_d16_hi v[146:147], v87, off
	v_or_b32_e32 v86, 0x4440, v0
	v_mov_b32_e32 v87, v1
	v_lshl_add_u64 v[146:147], v[132:133], 0, v[86:87]
	global_store_short_d16_hi v[146:147], v88, off
	v_or_b32_e32 v146, 0x4640, v0
	v_mov_b32_e32 v147, v1
	v_lshl_add_u64 v[148:149], v[132:133], 0, v[146:147]
	global_store_short_d16_hi v[148:149], v89, off
	v_or_b32_e32 v88, 0x4060, v0
	v_mov_b32_e32 v89, v1
	v_lshl_add_u64 v[148:149], v[132:133], 0, v[88:89]
	global_store_short_d16_hi v[148:149], v82, off
	v_or_b32_e32 v148, 0x4260, v0
	v_mov_b32_e32 v149, v1
	v_lshl_add_u64 v[150:151], v[132:133], 0, v[148:149]
	global_store_short_d16_hi v[150:151], v83, off
	v_or_b32_e32 v82, 0x4460, v0
	v_mov_b32_e32 v83, v1
	v_lshl_add_u64 v[150:151], v[132:133], 0, v[82:83]
	global_store_short_d16_hi v[150:151], v84, off
	v_or_b32_e32 v150, 0x4660, v0
	v_mov_b32_e32 v151, v1
	v_lshl_add_u64 v[152:153], v[132:133], 0, v[150:151]
	global_store_short_d16_hi v[152:153], v85, off
	v_or_b32_e32 v84, 0x6000, v0
	v_mov_b32_e32 v85, v1
	v_lshl_add_u64 v[152:153], v[132:133], 0, v[84:85]
	global_store_short_d16_hi v[152:153], v78, off
	v_or_b32_e32 v152, 0x6200, v0
	v_mov_b32_e32 v153, v1
	v_lshl_add_u64 v[154:155], v[132:133], 0, v[152:153]
	global_store_short_d16_hi v[154:155], v79, off
	v_or_b32_e32 v78, 0x6400, v0
	v_mov_b32_e32 v79, v1
	v_lshl_add_u64 v[154:155], v[132:133], 0, v[78:79]
	global_store_short_d16_hi v[154:155], v80, off
	v_or_b32_e32 v154, 0x6600, v0
	v_mov_b32_e32 v155, v1
	v_lshl_add_u64 v[156:157], v[132:133], 0, v[154:155]
	global_store_short_d16_hi v[156:157], v81, off
	v_or_b32_e32 v80, 0x6020, v0
	v_mov_b32_e32 v81, v1
	v_lshl_add_u64 v[156:157], v[132:133], 0, v[80:81]
	global_store_short_d16_hi v[156:157], v74, off
	v_or_b32_e32 v156, 0x6220, v0
	v_mov_b32_e32 v157, v1
	v_lshl_add_u64 v[158:159], v[132:133], 0, v[156:157]
	global_store_short_d16_hi v[158:159], v75, off
	v_or_b32_e32 v74, 0x6420, v0
	v_mov_b32_e32 v75, v1
	v_lshl_add_u64 v[158:159], v[132:133], 0, v[74:75]
	global_store_short_d16_hi v[158:159], v76, off
	v_or_b32_e32 v158, 0x6620, v0
	v_mov_b32_e32 v159, v1
	v_lshl_add_u64 v[160:161], v[132:133], 0, v[158:159]
	global_store_short_d16_hi v[160:161], v77, off
	v_or_b32_e32 v76, 0x6040, v0
	v_mov_b32_e32 v77, v1
	v_lshl_add_u64 v[160:161], v[132:133], 0, v[76:77]
	global_store_short_d16_hi v[160:161], v70, off
	v_or_b32_e32 v160, 0x6240, v0
	v_mov_b32_e32 v161, v1
	v_lshl_add_u64 v[162:163], v[132:133], 0, v[160:161]
	global_store_short_d16_hi v[162:163], v71, off
	v_or_b32_e32 v70, 0x6440, v0
	v_mov_b32_e32 v71, v1
	v_lshl_add_u64 v[162:163], v[132:133], 0, v[70:71]
	global_store_short_d16_hi v[162:163], v72, off
	v_or_b32_e32 v162, 0x6640, v0
	v_mov_b32_e32 v163, v1
	v_lshl_add_u64 v[164:165], v[132:133], 0, v[162:163]
	global_store_short_d16_hi v[164:165], v73, off
	v_or_b32_e32 v72, 0x6060, v0
	v_mov_b32_e32 v73, v1
	v_lshl_add_u64 v[164:165], v[132:133], 0, v[72:73]
	global_store_short_d16_hi v[164:165], v66, off
	v_or_b32_e32 v164, 0x6260, v0
	v_mov_b32_e32 v165, v1
	v_lshl_add_u64 v[166:167], v[132:133], 0, v[164:165]
	global_store_short_d16_hi v[166:167], v67, off
	v_or_b32_e32 v66, 0x6460, v0
	v_mov_b32_e32 v67, v1
	v_lshl_add_u64 v[166:167], v[132:133], 0, v[66:67]
	global_store_short_d16_hi v[166:167], v68, off
	v_or_b32_e32 v166, 0x6660, v0
	v_mov_b32_e32 v167, v1
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[166:167]
	v_or_b32_e32 v68, 0x8000, v130
	global_store_short_d16_hi v[132:133], v69, off
	v_ashrrev_i32_e32 v69, 31, v68
	v_lshl_add_u64 v[68:69], v[68:69], 1, s[8:9]
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[0:1]
	global_store_short_d16_hi v[0:1], v62, off
	global_store_short_d16_hi v[0:1], v63, off offset:512
	global_store_short_d16_hi v[0:1], v64, off offset:1024
	global_store_short_d16_hi v[0:1], v65, off offset:1536
	global_store_short_d16_hi v[0:1], v58, off offset:32
	global_store_short_d16_hi v[0:1], v59, off offset:544
	global_store_short_d16_hi v[0:1], v60, off offset:1056
	global_store_short_d16_hi v[0:1], v61, off offset:1568
	global_store_short_d16_hi v[0:1], v54, off offset:64
	global_store_short_d16_hi v[0:1], v55, off offset:576
	global_store_short_d16_hi v[0:1], v56, off offset:1088
	global_store_short_d16_hi v[0:1], v57, off offset:1600
	global_store_short_d16_hi v[0:1], v50, off offset:96
	global_store_short_d16_hi v[0:1], v51, off offset:608
	global_store_short_d16_hi v[0:1], v52, off offset:1120
	global_store_short_d16_hi v[0:1], v53, off offset:1632
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[114:115]
	global_store_short_d16_hi v[0:1], v46, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[116:117]
	global_store_short_d16_hi v[0:1], v47, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[110:111]
	global_store_short_d16_hi v[0:1], v48, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[118:119]
	global_store_short_d16_hi v[0:1], v49, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[112:113]
	global_store_short_d16_hi v[0:1], v42, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[120:121]
	global_store_short_d16_hi v[0:1], v43, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[106:107]
	global_store_short_d16_hi v[0:1], v44, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[122:123]
	global_store_short_d16_hi v[0:1], v45, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[108:109]
	global_store_short_d16_hi v[0:1], v38, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[124:125]
	global_store_short_d16_hi v[0:1], v39, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[102:103]
	global_store_short_d16_hi v[0:1], v40, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[126:127]
	global_store_short_d16_hi v[0:1], v41, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[104:105]
	global_store_short_d16_hi v[0:1], v34, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[128:129]
	global_store_short_d16_hi v[0:1], v35, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[98:99]
	global_store_short_d16_hi v[0:1], v36, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[134:135]
	global_store_short_d16_hi v[0:1], v37, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[100:101]
	global_store_short_d16_hi v[0:1], v30, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[136:137]
	global_store_short_d16_hi v[0:1], v31, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[94:95]
	global_store_short_d16_hi v[0:1], v32, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[138:139]
	global_store_short_d16_hi v[0:1], v33, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[96:97]
	global_store_short_d16_hi v[0:1], v26, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[140:141]
	global_store_short_d16_hi v[0:1], v27, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[90:91]
	global_store_short_d16_hi v[0:1], v28, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[142:143]
	global_store_short_d16_hi v[0:1], v29, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[92:93]
	global_store_short_d16_hi v[0:1], v22, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[144:145]
	global_store_short_d16_hi v[0:1], v23, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[86:87]
	global_store_short_d16_hi v[0:1], v24, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[146:147]
	global_store_short_d16_hi v[0:1], v25, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[88:89]
	global_store_short_d16_hi v[0:1], v18, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[148:149]
	global_store_short_d16_hi v[0:1], v19, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[82:83]
	global_store_short_d16_hi v[0:1], v20, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[150:151]
	global_store_short_d16_hi v[0:1], v21, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[84:85]
	global_store_short_d16_hi v[0:1], v14, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[152:153]
	global_store_short_d16_hi v[0:1], v15, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[78:79]
	global_store_short_d16_hi v[0:1], v16, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[154:155]
	global_store_short_d16_hi v[0:1], v17, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[80:81]
	global_store_short_d16_hi v[0:1], v10, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[156:157]
	global_store_short_d16_hi v[0:1], v11, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[74:75]
	global_store_short_d16_hi v[0:1], v12, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[158:159]
	global_store_short_d16_hi v[0:1], v13, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[76:77]
	global_store_short_d16_hi v[0:1], v6, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[160:161]
	global_store_short_d16_hi v[0:1], v7, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[70:71]
	global_store_short_d16_hi v[0:1], v8, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[162:163]
	global_store_short_d16_hi v[0:1], v9, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[72:73]
	global_store_short_d16_hi v[0:1], v2, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[164:165]
	global_store_short_d16_hi v[0:1], v3, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[66:67]
	global_store_short_d16_hi v[0:1], v4, off
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[166:167]
	global_store_short_d16_hi v[0:1], v5, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 48
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 225
		.amdhsa_next_free_sgpr 16
		.amdhsa_accum_offset 228
		.amdhsa_reserve_vcc 1
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
	.size	_Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE, .Lfunc_end0-_Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE
                                        ; -- End function
	.set _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.num_vgpr, 225
	.set _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.num_agpr, 0
	.set _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.numbered_sgpr, 16
	.set _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.private_seg_size, 0
	.set _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.uses_vcc, 1
	.set _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.uses_flat_scratch, 0
	.set _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.has_dyn_sized_stack, 0
	.set _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.has_recursion, 0
	.set _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7168
; TotalNumSgprs: 22
; NumVgprs: 225
; NumAgprs: 0
; TotalNumVgprs: 225
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 28
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 225
; AccumOffset: 228
; Occupancy: 2
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 56
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_2638120885d6af87,@object ; @__hip_cuid_2638120885d6af87
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_2638120885d6af87
__hip_cuid_2638120885d6af87:
	.byte	0                               ; 0x0
	.size	__hip_cuid_2638120885d6af87, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_2638120885d6af87
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           16
        .value_kind:     by_value
      - .offset:         16
        .size:           16
        .value_kind:     by_value
      - .offset:         32
        .size:           16
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tkN7kittens2glI14__hip_bfloat16Li1ELi1ELi256ELi128EJEEES2_NS0_IS1_Li1ELi1ELi256ELi256EJEEE.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     225
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
