; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 -mattr='+wavefrontsize32' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32
	.p2align	8
	.type	matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32,@function
matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32:
	s_load_b128 s[12:15], s[0:1], 0x0
	v_or_b32_e32 v1, 32, v0
	v_lshrrev_b32_e32 v73, 4, v0
	v_and_b32_e32 v74, 15, v0
	v_or_b32_e32 v75, 64, v0
	v_or_b32_e32 v76, 0x60, v0
	v_lshrrev_b32_e32 v81, 4, v1
	v_lshlrev_b32_e32 v131, 10, v73
	v_lshlrev_b32_e32 v99, 4, v74
	v_or_b32_e32 v77, 0x80, v0
	v_or_b32_e32 v78, 0xa0, v0
	v_lshlrev_b32_e32 v139, 10, v81
	v_lshrrev_b32_e32 v82, 4, v75
	v_lshrrev_b32_e32 v88, 1, v0
	v_lshrrev_b32_e32 v89, 1, v1
	v_or_b32_e32 v1, v99, v131
	v_lshlrev_b32_e32 v108, 4, v0
	v_lshrrev_b32_e32 v83, 4, v76
	v_or_b32_e32 v2, v99, v139
	v_lshrrev_b32_e32 v84, 4, v77
	s_waitcnt lgkmcnt(0)
	s_and_b32 s13, s13, 0xffff
	s_mov_b32 s7, 0x31027000
	s_movk_i32 s6, 0x4000
	s_mov_b32 s4, s12
	s_mov_b32 s5, s13
	v_lshrrev_b32_e32 v85, 4, v78
	s_clause 0x1
	buffer_load_b128 v[9:12], v1, s[4:7], 0 offen
	buffer_load_b128 v[13:16], v2, s[4:7], 0 offen
	v_lshl_or_b32 v1, v82, 10, v99
	v_lshlrev_b32_e32 v124, 5, v88
	v_and_b32_e32 v117, 16, v108
	v_lshl_or_b32 v2, v83, 10, v99
	v_lshl_or_b32 v3, v84, 10, v99
	v_lshrrev_b32_e32 v111, 1, v75
	v_lshl_or_b32 v4, v85, 10, v99
	v_or_b32_e32 v79, 0xc0, v0
	v_lshrrev_b32_e32 v112, 1, v76
	v_lshrrev_b32_e32 v113, 1, v77
	s_clause 0x3
	buffer_load_b128 v[17:20], v1, s[4:7], 0 offen
	buffer_load_b128 v[21:24], v2, s[4:7], 0 offen
	buffer_load_b128 v[25:28], v3, s[4:7], 0 offen
	buffer_load_b128 v[29:32], v4, s[4:7], 0 offen
	v_or_b32_e32 v1, v124, v117
	v_lshl_or_b32 v3, v89, 5, v117
	v_or_b32_e32 v80, 0xe0, v0
	s_and_b32 s15, s15, 0xffff
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_mov_b32 s8, s14
	s_mov_b32 s9, s15
	v_lshrrev_b32_e32 v114, 1, v78
	v_lshl_or_b32 v5, v111, 5, v117
	v_lshrrev_b32_e32 v86, 4, v79
	v_lshrrev_b32_e32 v115, 1, v79
	s_clause 0x1
	buffer_load_b128 v[33:36], v1, s[8:11], 0 offen
	buffer_load_b128 v[37:40], v3, s[8:11], 0 offen
	v_lshl_or_b32 v1, v112, 5, v117
	v_lshl_or_b32 v3, v113, 5, v117
	v_lshrrev_b32_e32 v87, 4, v80
	v_lshrrev_b32_e32 v116, 1, v80
	buffer_load_b128 v[41:44], v5, s[8:11], 0 offen
	v_lshl_or_b32 v5, v114, 5, v117
	v_lshl_or_b32 v2, v86, 10, v99
	s_clause 0x1
	buffer_load_b128 v[45:48], v1, s[8:11], 0 offen
	buffer_load_b128 v[49:52], v3, s[8:11], 0 offen
	v_lshl_or_b32 v1, v115, 5, v117
	v_lshl_or_b32 v4, v87, 10, v99
	v_lshl_or_b32 v3, v116, 5, v117
	buffer_load_b128 v[53:56], v5, s[8:11], 0 offen
	buffer_load_b128 v[57:60], v2, s[4:7], 0 offen
	buffer_load_b128 v[61:64], v1, s[8:11], 0 offen
	buffer_load_b128 v[65:68], v4, s[4:7], 0 offen
	buffer_load_b128 v[69:72], v3, s[8:11], 0 offen
	s_load_b64 s[0:1], s[0:1], 0x10
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v90, 1, v0
	v_and_b32_e32 v107, 1, v0
	v_mul_u32_u24_e32 v109, 0x108, v73
	v_mul_u32_u24_e32 v132, 0x108, v74
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v3, v1 :: v_dual_and_b32 v138, 0xf0, v108
	v_lshlrev_b32_e32 v73, 1, v74
	v_lshlrev_b32_e32 v74, 4, v80
	v_lshlrev_b32_e32 v118, 4, v79
	v_lshlrev_b32_e32 v119, 4, v78
	v_lshlrev_b32_e32 v120, 4, v77
	v_lshlrev_b32_e32 v121, 4, v76
	v_lshlrev_b32_e32 v122, 4, v75
	v_or_b32_e32 v123, 0x200, v108
	v_lshlrev_b32_e32 v80, 6, v80
	v_lshlrev_b32_e32 v79, 6, v79
	v_lshlrev_b32_e32 v78, 6, v78
	v_lshlrev_b32_e32 v77, 6, v77
	v_lshlrev_b32_e32 v76, 6, v76
	v_lshlrev_b32_e32 v75, 6, v75
	v_mul_u32_u24_e32 v133, 0x108, v81
	v_mul_u32_u24_e32 v134, 0x108, v82
	v_mul_u32_u24_e32 v135, 0x108, v83
	v_mul_u32_u24_e32 v136, 0x108, v84
	v_mul_u32_u24_e32 v137, 0x108, v85
	v_mul_u32_u24_e32 v140, 0x108, v86
	v_mul_u32_u24_e32 v141, 0x108, v87
	s_movk_i32 s3, 0x1000
	s_movk_i32 s4, 0xfe0
	s_movk_i32 s5, 0xde0
	s_movk_i32 s8, 0x9e0
	s_movk_i32 s9, 0x7e0
	s_movk_i32 s16, 0x5e0
	s_movk_i32 s17, 0x3e0
	s_movk_i32 s18, 0x100
	s_movk_i32 s19, 0x3c00
	s_movk_i32 s20, 0x3400
	s_movk_i32 s21, 0x2c00
	s_movk_i32 s22, 0x2400
	s_movk_i32 s23, 0x1c00
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v8, v1
	v_or_b32_e32 v106, 0xa0, v90
	v_or_b32_e32 v105, 0x1e0, v90
	v_or_b32_e32 v103, 0x320, v90
	v_or_b32_e32 v104, 0x460, v90
	v_or_b32_e32 v102, 0x5a0, v90
	v_or_b32_e32 v101, 0x6e0, v90
	v_or_b32_e32 v100, 0x820, v90
	v_or_b32_e32 v98, 0x960, v90
	v_or_b32_e32 v97, 0xaa0, v90
	v_or_b32_e32 v96, 0xbe0, v90
	v_or_b32_e32 v95, 0xd20, v90
	v_or_b32_e32 v94, 0xe60, v90
	v_or_b32_e32 v93, 0xfa0, v90
	v_or_b32_e32 v92, 0x10e0, v90
	v_or_b32_e32 v91, 0x1220, v90
	v_or_b32_e32 v90, 0x1360, v90
	v_lshlrev_b32_e32 v107, 4, v107
	v_add3_u32 v108, v99, v109, 0x1400
	v_mad_u32_u24 v109, v88, 40, v117
	v_mad_u32_u24 v110, v89, 40, v117
	v_mad_u32_u24 v111, v111, 40, v117
	v_mad_u32_u24 v112, v112, 40, v117
	v_mad_u32_u24 v113, v113, 40, v117
	v_mad_u32_u24 v114, v114, 40, v117
	v_mad_u32_u24 v115, v115, 40, v117
	v_mad_u32_u24 v116, v116, 40, v117
	v_and_or_b32 v117, v74, s4, 0x1000
	v_and_or_b32 v118, v118, s5, 0x1000
	v_and_or_b32 v119, 0xbe0, v119, s3
	v_and_or_b32 v120, v120, s8, 0x1000
	v_and_or_b32 v121, v121, s9, 0x1000
	v_and_or_b32 v122, v122, s16, 0x1000
	v_and_or_b32 v123, v123, s17, 0x1000
	v_or_b32_e32 v124, 0x1000, v124
	v_and_or_b32 v125, v80, s19, 0x100
	v_and_or_b32 v126, v79, s20, 0x100
	v_and_or_b32 v127, v78, s21, 0x100
	v_and_or_b32 v128, v77, s22, 0x100
	v_and_or_b32 v129, v76, s23, 0x100
	v_and_or_b32 v130, 0x1400, v75, s18
	v_or_b32_e32 v131, 0x100, v131
	v_add_nc_u32_e32 v74, 0x1400, v132
	v_add_nc_u32_e32 v75, 0x1410, v132
	v_add_nc_u32_e32 v76, 0x1420, v132
	v_add_nc_u32_e32 v83, 0x1430, v132
	v_add_nc_u32_e32 v77, 0x1440, v132
	v_add_nc_u32_e32 v84, 0x1450, v132
	v_add_nc_u32_e32 v78, 0x1460, v132
	v_add_nc_u32_e32 v85, 0x1470, v132
	v_add_nc_u32_e32 v79, 0x1480, v132
	v_add_nc_u32_e32 v86, 0x1490, v132
	v_add_nc_u32_e32 v80, 0x14a0, v132
	v_add_nc_u32_e32 v87, 0x14b0, v132
	v_add_nc_u32_e32 v81, 0x14c0, v132
	v_add_nc_u32_e32 v88, 0x14d0, v132
	v_add_nc_u32_e32 v82, 0x14e0, v132
	v_add_nc_u32_e32 v89, 0x14f0, v132
	v_add3_u32 v132, v133, v138, 0x1400
	v_add3_u32 v133, v134, v138, 0x1400
	v_add3_u32 v134, v135, v138, 0x1400
	v_add3_u32 v135, v136, v138, 0x1400
	v_add3_u32 v136, v137, v138, 0x1400
	v_add3_u32 v137, v140, v138, 0x1400
	v_add3_u32 v138, v141, v138, 0x1400
	v_or_b32_e32 v139, 0x100, v139
	s_mov_b32 s2, -8
	s_mov_b32 s8, s14
	s_mov_b32 s9, s15
	s_mov_b32 s4, s12
	s_mov_b32 s5, s13
	s_waitcnt vmcnt(15)
	ds_store_2addr_b64 v108, v[9:10], v[11:12] offset1:1
	s_waitcnt vmcnt(9)
	ds_store_2addr_b64 v109, v[33:34], v[35:36] offset1:1
	ds_store_2addr_b64 v132, v[13:14], v[15:16] offset1:1
	s_waitcnt vmcnt(8)
	ds_store_2addr_b64 v110, v[37:38], v[39:40] offset1:1
	ds_store_2addr_b64 v133, v[17:18], v[19:20] offset1:1
	s_waitcnt vmcnt(7)
	ds_store_2addr_b64 v111, v[41:42], v[43:44] offset1:1
	ds_store_2addr_b64 v134, v[21:22], v[23:24] offset1:1
	s_waitcnt vmcnt(6)
	ds_store_2addr_b64 v112, v[45:46], v[47:48] offset1:1
	ds_store_2addr_b64 v135, v[25:26], v[27:28] offset1:1
	s_waitcnt vmcnt(5)
	ds_store_2addr_b64 v113, v[49:50], v[51:52] offset1:1
	ds_store_2addr_b64 v136, v[29:30], v[31:32] offset1:1
	s_waitcnt vmcnt(4)
	ds_store_2addr_b64 v114, v[53:54], v[55:56] offset1:1
	s_waitcnt vmcnt(3)
	ds_store_2addr_b64 v137, v[57:58], v[59:60] offset1:1
	s_waitcnt vmcnt(2)
	ds_store_2addr_b64 v115, v[61:62], v[63:64] offset1:1
	s_waitcnt vmcnt(1)
	ds_store_2addr_b64 v138, v[65:66], v[67:68] offset1:1
	s_waitcnt vmcnt(0)
	ds_store_2addr_b64 v116, v[69:70], v[71:72] offset1:1
.LBB0_1:
	v_add_nc_u32_e32 v9, v131, v99
	v_add_nc_u32_e32 v10, v139, v99
	v_add_nc_u32_e32 v11, v130, v99
	v_add_nc_u32_e32 v12, v129, v99
	v_add_nc_u32_e32 v13, v128, v99
	v_add_nc_u32_e32 v14, v127, v99
	v_add_nc_u32_e32 v15, v126, v99
	v_add_nc_u32_e32 v17, v125, v99
	v_add_nc_u32_e32 v18, v124, v107
	v_add_nc_u32_e32 v19, v123, v107
	v_add_nc_u32_e32 v20, v122, v107
	v_add_nc_u32_e32 v21, v121, v107
	v_add_nc_u32_e32 v22, v120, v107
	v_add_nc_u32_e32 v23, v119, v107
	v_add_nc_u32_e32 v24, v118, v107
	v_add_nc_u32_e32 v140, v117, v107
	s_clause 0x7
	buffer_load_b128 v[65:68], v9, s[4:7], 0 offen
	buffer_load_b128 v[41:44], v10, s[4:7], 0 offen
	buffer_load_b128 v[37:40], v11, s[4:7], 0 offen
	buffer_load_b128 v[33:36], v12, s[4:7], 0 offen
	buffer_load_b128 v[29:32], v13, s[4:7], 0 offen
	buffer_load_b128 v[25:28], v14, s[4:7], 0 offen
	buffer_load_b128 v[13:16], v15, s[4:7], 0 offen
	buffer_load_b128 v[9:12], v17, s[4:7], 0 offen
	s_clause 0x7
	buffer_load_b128 v[69:72], v18, s[8:11], 0 offen
	buffer_load_b128 v[61:64], v19, s[8:11], 0 offen
	buffer_load_b128 v[57:60], v20, s[8:11], 0 offen
	buffer_load_b128 v[53:56], v21, s[8:11], 0 offen
	buffer_load_b128 v[49:52], v22, s[8:11], 0 offen
	buffer_load_b128 v[45:48], v23, s[8:11], 0 offen
	buffer_load_b128 v[21:24], v24, s[8:11], 0 offen
	buffer_load_b128 v[17:20], v140, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	ds_load_u16_d16 v142, v106
	ds_load_u16_d16 v146, v105
	ds_load_u16_d16 v140, v73
	ds_load_u16_d16 v141, v73 offset:80
	ds_load_u16_d16 v144, v73 offset:320
	ds_load_u16_d16 v143, v73 offset:240
	ds_load_u16_d16 v147, v73 offset:560
	ds_load_u16_d16 v145, v73 offset:400
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v142, v73 offset:200
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v146, v73 offset:520
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v140, v73 offset:40
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v141, v73 offset:120
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v144, v73 offset:360
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v143, v73 offset:280
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v147, v73 offset:600
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v145, v73 offset:440
	ds_load_2addr_b64 v[152:155], v75 offset1:1
	ds_load_2addr_b64 v[148:151], v74 offset1:1
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[148:155], v[140:147], v[1:8]
	ds_load_u16_d16 v141, v73 offset:720
	ds_load_u16_d16 v144, v73 offset:960
	ds_load_u16_d16 v147, v73 offset:1200
	ds_load_u16_d16 v142, v103
	ds_load_u16_d16 v145, v73 offset:1040
	ds_load_u16_d16 v140, v73 offset:640
	ds_load_u16_d16 v143, v73 offset:880
	ds_load_u16_d16 v146, v104
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v141, v73 offset:760
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v144, v73 offset:1000
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v147, v73 offset:1240
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v142, v73 offset:840
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v145, v73 offset:1080
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v140, v73 offset:680
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v143, v73 offset:920
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v146, v73 offset:1160
	ds_load_2addr_b64 v[152:155], v83 offset1:1
	ds_load_2addr_b64 v[148:151], v76 offset1:1
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[148:155], v[140:147], v[1:8]
	ds_load_u16_d16 v142, v102
	ds_load_u16_d16 v140, v73 offset:1280
	ds_load_u16_d16 v145, v73 offset:1680
	ds_load_u16_d16 v143, v73 offset:1520
	ds_load_u16_d16 v146, v101
	ds_load_u16_d16 v141, v73 offset:1360
	ds_load_u16_d16 v144, v73 offset:1600
	ds_load_u16_d16 v147, v73 offset:1840
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v142, v73 offset:1480
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v140, v73 offset:1320
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v145, v73 offset:1720
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v143, v73 offset:1560
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v146, v73 offset:1800
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v141, v73 offset:1400
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v144, v73 offset:1640
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v147, v73 offset:1880
	ds_load_2addr_b64 v[152:155], v84 offset1:1
	ds_load_2addr_b64 v[148:151], v77 offset1:1
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[148:155], v[140:147], v[1:8]
	ds_load_u16_d16 v140, v73 offset:1920
	ds_load_u16_d16 v142, v100
	ds_load_u16_d16 v146, v98
	ds_load_u16_d16 v141, v73 offset:2000
	ds_load_u16_d16 v143, v73 offset:2160
	ds_load_u16_d16 v144, v73 offset:2240
	ds_load_u16_d16 v145, v73 offset:2320
	ds_load_u16_d16 v147, v73 offset:2480
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v140, v73 offset:1960
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v142, v73 offset:2120
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v146, v73 offset:2440
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v141, v73 offset:2040
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v143, v73 offset:2200
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v144, v73 offset:2280
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v145, v73 offset:2360
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v147, v73 offset:2520
	ds_load_2addr_b64 v[152:155], v85 offset1:1
	ds_load_2addr_b64 v[148:151], v78 offset1:1
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[148:155], v[140:147], v[1:8]
	ds_load_u16_d16 v140, v73 offset:2560
	ds_load_u16_d16 v142, v97
	ds_load_u16_d16 v146, v96
	ds_load_u16_d16 v141, v73 offset:2640
	ds_load_u16_d16 v144, v73 offset:2880
	ds_load_u16_d16 v143, v73 offset:2800
	ds_load_u16_d16 v147, v73 offset:3120
	ds_load_u16_d16 v145, v73 offset:2960
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v140, v73 offset:2600
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v142, v73 offset:2760
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v146, v73 offset:3080
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v141, v73 offset:2680
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v144, v73 offset:2920
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v143, v73 offset:2840
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v147, v73 offset:3160
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v145, v73 offset:3000
	ds_load_2addr_b64 v[152:155], v86 offset1:1
	ds_load_2addr_b64 v[148:151], v79 offset1:1
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[148:155], v[140:147], v[1:8]
	ds_load_u16_d16 v141, v73 offset:3280
	ds_load_u16_d16 v144, v73 offset:3520
	ds_load_u16_d16 v142, v95
	ds_load_u16_d16 v147, v73 offset:3760
	ds_load_u16_d16 v145, v73 offset:3600
	ds_load_u16_d16 v140, v73 offset:3200
	ds_load_u16_d16 v143, v73 offset:3440
	ds_load_u16_d16 v146, v94
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v141, v73 offset:3320
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v144, v73 offset:3560
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v142, v73 offset:3400
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v147, v73 offset:3800
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v145, v73 offset:3640
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v140, v73 offset:3240
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v143, v73 offset:3480
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v146, v73 offset:3720
	ds_load_2addr_b64 v[152:155], v87 offset1:1
	ds_load_2addr_b64 v[148:151], v80 offset1:1
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[148:155], v[140:147], v[1:8]
	ds_load_u16_d16 v141, v73 offset:3920
	ds_load_u16_d16 v142, v93
	ds_load_u16_d16 v144, v73 offset:4160
	ds_load_u16_d16 v146, v92
	ds_load_u16_d16 v140, v73 offset:3840
	ds_load_u16_d16 v143, v73 offset:4080
	ds_load_u16_d16 v145, v73 offset:4240
	ds_load_u16_d16 v147, v73 offset:4400
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v141, v73 offset:3960
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v142, v73 offset:4040
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v144, v73 offset:4200
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v146, v73 offset:4360
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v140, v73 offset:3880
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v143, v73 offset:4120
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v145, v73 offset:4280
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v147, v73 offset:4440
	ds_load_2addr_b64 v[152:155], v88 offset1:1
	ds_load_2addr_b64 v[148:151], v81 offset1:1
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[148:155], v[140:147], v[1:8]
	ds_load_u16_d16 v140, v73 offset:4480
	ds_load_u16_d16 v142, v91
	ds_load_u16_d16 v146, v90
	ds_load_u16_d16 v141, v73 offset:4560
	ds_load_u16_d16 v144, v73 offset:4800
	ds_load_u16_d16 v143, v73 offset:4720
	ds_load_u16_d16 v147, v73 offset:5040
	ds_load_u16_d16 v145, v73 offset:4880
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v140, v73 offset:4520
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v142, v73 offset:4680
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v146, v73 offset:5000
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v141, v73 offset:4600
	s_waitcnt lgkmcnt(7)
	ds_load_u16_d16_hi v144, v73 offset:4840
	ds_load_2addr_b64 v[152:155], v89 offset1:1
	ds_load_2addr_b64 v[148:151], v82 offset1:1
	s_waitcnt lgkmcnt(9)
	ds_load_u16_d16_hi v143, v73 offset:4760
	s_waitcnt lgkmcnt(9)
	ds_load_u16_d16_hi v147, v73 offset:5080
	s_waitcnt lgkmcnt(9)
	ds_load_u16_d16_hi v145, v73 offset:4920
	s_waitcnt lgkmcnt(0)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[148:155], v[140:147], v[1:8]
	v_add_nc_u32_e32 v117, 0x1000, v117
	v_add_nc_u32_e32 v118, 0x1000, v118
	v_add_nc_u32_e32 v119, 0x1000, v119
	v_add_nc_u32_e32 v120, 0x1000, v120
	v_add_nc_u32_e32 v121, 0x1000, v121
	v_add_nc_u32_e32 v122, 0x1000, v122
	v_add_nc_u32_e32 v123, 0x1000, v123
	v_add_nc_u32_e32 v124, 0x1000, v124
	v_add_nc_u32_e32 v125, 0x100, v125
	v_add_nc_u32_e32 v126, 0x100, v126
	v_add_nc_u32_e32 v127, 0x100, v127
	v_add_nc_u32_e32 v128, 0x100, v128
	v_add_nc_u32_e32 v129, 0x100, v129
	v_add_nc_u32_e32 v130, 0x100, v130
	v_add_nc_u32_e32 v139, 0x100, v139
	v_add_nc_u32_e32 v131, 0x100, v131
	s_add_i32 s2, s2, 8
	s_waitcnt vmcnt(15)
	ds_store_2addr_b64 v108, v[65:66], v[67:68] offset1:1
	s_waitcnt vmcnt(7)
	ds_store_2addr_b64 v109, v[69:70], v[71:72] offset1:1
	ds_store_2addr_b64 v132, v[41:42], v[43:44] offset1:1
	s_waitcnt vmcnt(6)
	ds_store_2addr_b64 v110, v[61:62], v[63:64] offset1:1
	ds_store_2addr_b64 v133, v[37:38], v[39:40] offset1:1
	s_waitcnt vmcnt(5)
	ds_store_2addr_b64 v111, v[57:58], v[59:60] offset1:1
	ds_store_2addr_b64 v134, v[33:34], v[35:36] offset1:1
	s_waitcnt vmcnt(4)
	ds_store_2addr_b64 v112, v[53:54], v[55:56] offset1:1
	ds_store_2addr_b64 v135, v[29:30], v[31:32] offset1:1
	s_waitcnt vmcnt(3)
	ds_store_2addr_b64 v113, v[49:50], v[51:52] offset1:1
	ds_store_2addr_b64 v136, v[25:26], v[27:28] offset1:1
	s_waitcnt vmcnt(2)
	ds_store_2addr_b64 v114, v[45:46], v[47:48] offset1:1
	s_cmp_lt_u32 s2, 16
	ds_store_2addr_b64 v137, v[13:14], v[15:16] offset1:1
	s_waitcnt vmcnt(1)
	ds_store_2addr_b64 v115, v[21:22], v[23:24] offset1:1
	ds_store_2addr_b64 v138, v[9:10], v[11:12] offset1:1
	s_waitcnt vmcnt(0)
	ds_store_2addr_b64 v116, v[17:18], v[19:20] offset1:1
	s_cbranch_scc1 .LBB0_1
	s_waitcnt lgkmcnt(0)
	ds_load_u16_d16 v11, v106
	ds_load_u16_d16 v15, v105
	ds_load_u16_d16 v9, v73
	ds_load_u16_d16 v10, v73 offset:80
	ds_load_u16_d16 v13, v73 offset:320
	ds_load_u16_d16 v18, v73 offset:720
	ds_load_u16_d16 v21, v73 offset:960
	ds_load_u16_d16 v24, v73 offset:1200
	ds_load_u16_d16 v12, v73 offset:240
	ds_load_u16_d16 v16, v73 offset:560
	ds_load_u16_d16 v19, v103
	ds_load_u16_d16 v23, v104
	ds_load_u16_d16 v27, v102
	ds_load_u16_d16 v31, v101
	ds_load_u16_d16 v35, v100
	ds_load_u16_d16 v39, v98
	v_lshlrev_b32_e32 v0, 2, v0
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x31027000
	s_movk_i32 s2, 0x400
	s_waitcnt lgkmcnt(15)
	ds_load_u16_d16_hi v11, v73 offset:200
	s_waitcnt lgkmcnt(15)
	ds_load_u16_d16_hi v15, v73 offset:520
	ds_load_u16_d16 v14, v73 offset:400
	ds_load_u16_d16 v22, v73 offset:1040
	ds_load_u16_d16 v25, v73 offset:1280
	ds_load_u16_d16 v30, v73 offset:1680
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v9, v73 offset:40
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v10, v73 offset:120
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v13, v73 offset:360
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v18, v73 offset:760
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v21, v73 offset:1000
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v24, v73 offset:1240
	ds_load_u16_d16 v17, v73 offset:640
	ds_load_u16_d16 v20, v73 offset:880
	ds_load_u16_d16 v28, v73 offset:1520
	ds_load_u16_d16 v33, v73 offset:1920
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v12, v73 offset:280
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v16, v73 offset:600
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v19, v73 offset:840
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v27, v73 offset:1480
	ds_load_u16_d16 v26, v73 offset:1360
	ds_load_u16_d16 v29, v73 offset:1600
	ds_load_u16_d16 v32, v73 offset:1840
	ds_load_u16_d16 v34, v73 offset:2000
	ds_load_u16_d16 v36, v73 offset:2160
	ds_load_u16_d16 v37, v73 offset:2240
	ds_load_u16_d16 v38, v73 offset:2320
	s_waitcnt lgkmcnt(24)
	ds_load_u16_d16_hi v14, v73 offset:440
	s_waitcnt lgkmcnt(24)
	ds_load_u16_d16_hi v22, v73 offset:1080
	s_waitcnt lgkmcnt(24)
	ds_load_u16_d16_hi v25, v73 offset:1320
	s_waitcnt lgkmcnt(24)
	ds_load_u16_d16_hi v30, v73 offset:1720
	ds_load_u16_d16 v43, v97
	ds_load_u16_d16 v47, v96
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v17, v73 offset:680
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v20, v73 offset:920
	ds_load_u16_d16_hi v23, v73 offset:1160
	s_waitcnt lgkmcnt(21)
	ds_load_u16_d16_hi v28, v73 offset:1560
	ds_load_u16_d16_hi v31, v73 offset:1800
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v33, v73 offset:1960
	ds_load_u16_d16_hi v35, v73 offset:2120
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v26, v73 offset:1400
	ds_load_u16_d16_hi v39, v73 offset:2440
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v29, v73 offset:1640
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v32, v73 offset:1880
	ds_load_u16_d16 v40, v73 offset:2480
	ds_load_u16_d16 v41, v73 offset:2560
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v34, v73 offset:2040
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v36, v73 offset:2200
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v37, v73 offset:2280
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v38, v73 offset:2360
	ds_load_u16_d16 v42, v73 offset:2640
	ds_load_u16_d16 v45, v73 offset:2880
	ds_load_u16_d16 v50, v73 offset:3280
	ds_load_u16_d16 v53, v73 offset:3520
	ds_load_u16_d16 v51, v95
	ds_load_u16_d16 v55, v94
	ds_load_u16_d16 v59, v93
	ds_load_u16_d16 v63, v92
	ds_load_u16_d16 v67, v91
	ds_load_u16_d16 v71, v90
	ds_load_u16_d16 v44, v73 offset:2800
	ds_load_u16_d16 v48, v73 offset:3120
	ds_load_u16_d16 v56, v73 offset:3760
	ds_load_u16_d16 v58, v73 offset:3920
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v40, v73 offset:2520
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v41, v73 offset:2600
	ds_load_u16_d16_hi v43, v73 offset:2760
	ds_load_u16_d16_hi v47, v73 offset:3080
	ds_load_u16_d16 v46, v73 offset:2960
	ds_load_u16_d16 v54, v73 offset:3600
	ds_load_u16_d16 v61, v73 offset:4160
	ds_load_u16_d16 v49, v73 offset:3200
	ds_load_u16_d16 v52, v73 offset:3440
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v42, v73 offset:2680
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v45, v73 offset:2920
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v50, v73 offset:3320
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v53, v73 offset:3560
	ds_load_u16_d16 v57, v73 offset:3840
	ds_load_u16_d16 v60, v73 offset:4080
	ds_load_u16_d16 v62, v73 offset:4240
	ds_load_u16_d16 v64, v73 offset:4400
	ds_load_u16_d16 v65, v73 offset:4480
	s_waitcnt lgkmcnt(21)
	ds_load_u16_d16_hi v44, v73 offset:2840
	s_waitcnt lgkmcnt(21)
	ds_load_u16_d16_hi v48, v73 offset:3160
	ds_load_u16_d16_hi v51, v73 offset:3400
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v56, v73 offset:3800
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v58, v73 offset:3960
	ds_load_u16_d16 v66, v73 offset:4560
	ds_load_u16_d16 v69, v73 offset:4800
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v46, v73 offset:3000
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v54, v73 offset:3640
	ds_load_u16_d16_hi v59, v73 offset:4040
	s_waitcnt lgkmcnt(21)
	ds_load_u16_d16_hi v61, v73 offset:4200
	ds_load_u16_d16_hi v63, v73 offset:4360
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v49, v73 offset:3240
	s_waitcnt lgkmcnt(22)
	ds_load_u16_d16_hi v52, v73 offset:3480
	ds_load_u16_d16_hi v55, v73 offset:3720
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v57, v73 offset:3880
	ds_load_u16_d16 v68, v73 offset:4720
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v60, v73 offset:4120
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v62, v73 offset:4280
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v64, v73 offset:4440
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v65, v73 offset:4520
	ds_load_u16_d16_hi v67, v73 offset:4680
	ds_load_u16_d16_hi v71, v73 offset:5000
	ds_load_u16_d16 v72, v73 offset:5040
	ds_load_u16_d16 v70, v73 offset:4880
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v66, v73 offset:4600
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v69, v73 offset:4840
	ds_load_2addr_b64 v[94:97], v83 offset1:1
	ds_load_2addr_b64 v[90:93], v76 offset1:1
	ds_load_2addr_b64 v[102:105], v84 offset1:1
	ds_load_2addr_b64 v[98:101], v77 offset1:1
	ds_load_2addr_b64 v[110:113], v85 offset1:1
	ds_load_2addr_b64 v[106:109], v78 offset1:1
	ds_load_2addr_b64 v[118:121], v86 offset1:1
	ds_load_2addr_b64 v[114:117], v79 offset1:1
	ds_load_2addr_b64 v[126:129], v87 offset1:1
	ds_load_2addr_b64 v[122:125], v80 offset1:1
	ds_load_2addr_b64 v[134:137], v88 offset1:1
	ds_load_2addr_b64 v[130:133], v81 offset1:1
	ds_load_2addr_b64 v[83:86], v89 offset1:1
	ds_load_2addr_b64 v[79:82], v82 offset1:1
	s_waitcnt lgkmcnt(24)
	ds_load_u16_d16_hi v68, v73 offset:4760
	ds_load_2addr_b64 v[142:145], v75 offset1:1
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v72, v73 offset:5080
	ds_load_2addr_b64 v[138:141], v74 offset1:1
	s_waitcnt lgkmcnt(20)
	ds_load_u16_d16_hi v70, v73 offset:4920
	s_waitcnt lgkmcnt(1)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[138:145], v[9:16], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[90:97], v[17:24], v[1:8]
	v_wmma_f32_16x16x16_bf16 v[1:8], v[98:105], v[25:32], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[106:113], v[33:40], v[1:8]
	v_wmma_f32_16x16x16_bf16 v[1:8], v[114:121], v[41:48], v[1:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[122:129], v[49:56], v[1:8]
	v_wmma_f32_16x16x16_bf16 v[1:8], v[130:137], v[57:64], v[1:8]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_bf16 v[1:8], v[79:86], v[65:72], v[1:8]
	s_clause 0x7
	buffer_store_b32 v1, v0, s[0:3], 0 offen
	buffer_store_b32 v2, v0, s[0:3], 0 offen offset:128
	buffer_store_b32 v3, v0, s[0:3], 0 offen offset:256
	buffer_store_b32 v4, v0, s[0:3], 0 offen offset:384
	buffer_store_b32 v5, v0, s[0:3], 0 offen offset:512
	buffer_store_b32 v6, v0, s[0:3], 0 offen offset:640
	buffer_store_b32 v7, v0, s[0:3], 0 offen offset:768
	buffer_store_b32 v8, v0, s[0:3], 0 offen offset:896
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32
		.amdhsa_group_segment_fixed_size 9344
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 241
		.amdhsa_next_free_sgpr 24
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 40
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
	.size	matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32, .Lfunc_end0-matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32

	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.num_vgpr, 156
	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.num_agpr, 0
	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.numbered_sgpr, 24
	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.num_named_barrier, 0
	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.private_seg_size, 0
	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.uses_vcc, 0
	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.uses_flat_scratch, 0
	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.has_dyn_sized_stack, 0
	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.has_recursion, 0
	.set matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.has_indirect_call, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
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
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 9344
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 32
    .name:           matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         matmul_dispatch_0_matmul_16x16x512_bf16xbf16xf32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     156
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
