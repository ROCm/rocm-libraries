; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 -mattr='+wavefrontsize32' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_dispatch_0_matmul_64x128x512_f16xf16xf32
	.p2align	8
	.type	matmul_dispatch_0_matmul_64x128x512_f16xf16xf32,@function
matmul_dispatch_0_matmul_64x128x512_f16xf16xf32:
	s_load_b128 s[12:15], s[0:1], 0x0
	v_or_b32_e32 v1, 0x80, v0
	v_lshrrev_b32_e32 v44, 4, v0
	v_lshlrev_b32_e32 v4, 4, v0
	v_lshlrev_b32_e32 v5, 3, v0
	s_lshl_b32 s3, s2, 3
	v_lshrrev_b32_e32 v45, 4, v1
	s_and_b32 s3, s3, 32
	v_or_b32_e32 v2, 0x100, v0
	v_or_b32_e32 v3, 0x180, v0
	v_or_b32_e32 v6, s3, v44
	v_and_b32_e32 v49, 0xf0, v4
	v_and_b32_e32 v52, 24, v5
	v_lshrrev_b32_e32 v50, 2, v1
	v_or_b32_e32 v1, s3, v45
	s_lshl_b32 s8, s2, 5
	v_lshrrev_b32_e32 v46, 4, v2
	s_and_b32 s20, s8, 0x60
	v_lshrrev_b32_e32 v47, 4, v3
	v_lshrrev_b32_e32 v51, 2, v2
	v_lshl_or_b32 v2, v6, 10, v49
	v_lshrrev_b32_e32 v53, 2, v3
	v_or_b32_e32 v3, s20, v52
	s_waitcnt lgkmcnt(0)
	s_and_b32 s13, s13, 0xffff
	v_lshrrev_b32_e32 v48, 2, v0
	v_lshl_or_b32 v1, v1, 10, v49
	s_mov_b32 s7, 0x31027000
	s_mov_b32 s6, 0x10000
	s_mov_b32 s4, s12
	s_mov_b32 s5, s13
	s_clause 0x1
	buffer_load_b128 v[9:12], v2, s[4:7], 0 offen
	buffer_load_b128 v[13:16], v1, s[4:7], 0 offen
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v57, 8, v50
	v_or_b32_e32 v4, s3, v46
	v_lshlrev_b32_e32 v54, 8, v48
	v_lshlrev_b32_e32 v59, 8, v51
	v_or_b32_e32 v3, s3, v47
	v_lshlrev_b32_e32 v61, 8, v53
	v_or_b32_e32 v5, v57, v1
	s_mov_b32 s10, 0x20000
	s_and_b32 s15, s15, 0xffff
	v_lshl_or_b32 v2, v4, 10, v49
	v_or_b32_e32 v4, v54, v1
	v_or_b32_e32 v6, v59, v1
	s_mov_b32 s18, s10
	s_mov_b32 s19, s7
	s_mov_b32 s16, s14
	s_mov_b32 s17, s15
	v_lshl_or_b32 v3, v3, 10, v49
	v_or_b32_e32 v1, v61, v1
	buffer_load_b128 v[21:24], v5, s[16:19], 0 offen
	buffer_load_b128 v[25:28], v2, s[4:7], 0 offen
	buffer_load_b128 v[29:32], v6, s[16:19], 0 offen
	buffer_load_b128 v[33:36], v3, s[4:7], 0 offen
	s_clause 0x1
	buffer_load_b128 v[37:40], v1, s[16:19], 0 offen
	buffer_load_b128 v[17:20], v4, s[16:19], 0 offen
	s_load_b64 s[0:1], s[0:1], 0x10
	v_mov_b32_e32 v1, 0
	v_and_b32_e32 v41, 15, v0
	v_lshrrev_b32_e32 v42, 1, v0
	v_and_b32_e32 v43, 16, v48
	v_and_b32_e32 v55, 3, v0
	v_mov_b32_e32 v3, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v7, v1 :: v_dual_and_b32 v42, 16, v42
	v_lshlrev_b32_e32 v52, 1, v52
	v_or_b32_e32 v68, v43, v41
	v_mul_u32_u24_e32 v58, 0x108, v44
	v_lshlrev_b32_e32 v44, 10, v44
	v_mul_u32_u24_e32 v62, 0x108, v45
	v_mul_u32_u24_e32 v65, 0x108, v46
	v_mul_u32_u24_e32 v66, 0x108, v47
	v_or_b32_e32 v71, v42, v41
	v_lshlrev_b32_e32 v47, 10, v47
	v_lshlrev_b32_e32 v46, 10, v46
	v_lshlrev_b32_e32 v45, 10, v45
	v_mad_u32_u24 v67, 0x48, v48, v52
	v_mul_u32_u24_e32 v48, 0x108, v68
	s_and_b32 s4, s2, 3
	s_lshl_b32 s2, s2, 13
	s_lshl_b32 s4, s4, 6
	s_and_b32 s2, s2, 0x8000
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v6, v1
	v_dual_mov_b32 v8, v1 :: v_dual_lshlrev_b32 v55, 4, v55
	v_lshlrev_b32_e32 v56, 4, v41
	v_add3_u32 v58, v58, v49, 0x2400
	v_or3_b32 v60, v54, s4, 0x8000
	v_or3_b32 v63, s2, v44, 0x100
	v_add3_u32 v64, v62, v49, 0x2400
	v_add3_u32 v65, v65, v49, 0x2400
	v_add3_u32 v66, v66, v49, 0x2400
	v_mad_u32_u24 v69, 0x48, v50, v52
	v_mad_u32_u24 v70, 0x48, v51, v52
	v_mad_u32_u24 v72, 0x48, v53, v52
	v_lshlrev_b32_e32 v44, 1, v71
	v_or3_b32 v73, v61, s4, 0x8000
	v_or3_b32 v74, v59, s4, 0x8000
	v_or3_b32 v75, v57, s4, 0x8000
	v_or3_b32 v76, s2, v47, 0x100
	v_or3_b32 v77, s2, v46, 0x100
	v_or3_b32 v78, s2, v45, 0x100
	v_add_nc_u32_e32 v68, 0x2400, v48
	v_add_nc_u32_e32 v71, 0x2410, v48
	v_add_nc_u32_e32 v61, 0x2420, v48
	v_add_nc_u32_e32 v62, 0x2430, v48
	v_add_nc_u32_e32 v57, 0x2440, v48
	v_add_nc_u32_e32 v59, 0x2450, v48
	v_add_nc_u32_e32 v53, 0x2460, v48
	v_add_nc_u32_e32 v54, 0x2470, v48
	v_add_nc_u32_e32 v51, 0x2480, v48
	v_add_nc_u32_e32 v52, 0x2490, v48
	v_add_nc_u32_e32 v49, 0x24a0, v48
	v_add_nc_u32_e32 v50, 0x24b0, v48
	v_add_nc_u32_e32 v45, 0x24c0, v48
	v_add_nc_u32_e32 v47, 0x24d0, v48
	v_add_nc_u32_e32 v46, 0x24e0, v48
	v_add_nc_u32_e32 v48, 0x24f0, v48
	s_mov_b32 s16, -8
	s_mov_b32 s11, s7
	s_mov_b32 s8, s14
	s_mov_b32 s9, s15
	s_mov_b32 s4, s12
	s_waitcnt vmcnt(7)
	ds_store_2addr_b64 v58, v[9:10], v[11:12] offset1:1
	s_waitcnt vmcnt(0)
	ds_store_2addr_b64 v67, v[17:18], v[19:20] offset1:1
	ds_store_2addr_b64 v64, v[13:14], v[15:16] offset1:1
	ds_store_2addr_b64 v69, v[21:22], v[23:24] offset1:1
	ds_store_2addr_b64 v65, v[25:26], v[27:28] offset1:1
	ds_store_2addr_b64 v70, v[29:30], v[31:32] offset1:1
	ds_store_2addr_b64 v66, v[33:34], v[35:36] offset1:1
	ds_store_2addr_b64 v72, v[37:38], v[39:40] offset1:1
.LBB0_1:
	v_add_nc_u32_e32 v9, v63, v56
	v_add_nc_u32_e32 v11, v77, v56
	v_add_nc_u32_e32 v10, v78, v56
	v_add_nc_u32_e32 v12, v76, v56
	v_add_nc_u32_e32 v17, v60, v55
	v_add_nc_u32_e32 v18, v75, v55
	v_add_nc_u32_e32 v19, v74, v55
	v_add_nc_u32_e32 v20, v73, v55
	s_clause 0x3
	buffer_load_b128 v[29:32], v9, s[4:7], 0 offen
	buffer_load_b128 v[21:24], v10, s[4:7], 0 offen
	buffer_load_b128 v[13:16], v11, s[4:7], 0 offen
	buffer_load_b128 v[9:12], v12, s[4:7], 0 offen
	s_clause 0x3
	buffer_load_b128 v[37:40], v17, s[8:11], 0 offen
	buffer_load_b128 v[33:36], v18, s[8:11], 0 offen
	buffer_load_b128 v[25:28], v19, s[8:11], 0 offen
	buffer_load_b128 v[17:20], v20, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_load_2addr_b64 v[83:86], v71 offset1:1
	ds_load_2addr_b64 v[79:82], v68 offset1:1
	ds_load_2addr_b64 v[91:94], v62 offset1:1
	ds_load_2addr_b64 v[87:90], v61 offset1:1
	ds_load_2addr_b64 v[99:102], v59 offset1:1
	ds_load_2addr_b64 v[95:98], v57 offset1:1
	ds_load_2addr_b64 v[107:110], v54 offset1:1
	ds_load_2addr_b64 v[103:106], v53 offset1:1
	ds_load_2addr_b64 v[115:118], v52 offset1:1
	ds_load_2addr_b64 v[111:114], v51 offset1:1
	ds_load_2addr_b64 v[123:126], v50 offset1:1
	ds_load_2addr_b64 v[119:122], v49 offset1:1
	ds_load_2addr_b64 v[131:134], v47 offset1:1
	ds_load_2addr_b64 v[127:130], v45 offset1:1
	ds_load_2addr_b64 v[139:142], v48 offset1:1
	ds_load_2addr_b64 v[135:138], v46 offset1:1
	ds_load_u16_d16 v143, v44
	ds_load_u16_d16 v144, v44 offset:144
	ds_load_u16_d16 v145, v44 offset:288
	ds_load_u16_d16 v146, v44 offset:432
	ds_load_u16_d16 v147, v44 offset:576
	ds_load_u16_d16 v148, v44 offset:720
	ds_load_u16_d16 v149, v44 offset:864
	ds_load_u16_d16 v150, v44 offset:1008
	ds_load_u16_d16 v151, v44 offset:1152
	ds_load_u16_d16 v152, v44 offset:1296
	ds_load_u16_d16 v153, v44 offset:1440
	ds_load_u16_d16 v154, v44 offset:1584
	ds_load_u16_d16 v155, v44 offset:1728
	ds_load_u16_d16 v156, v44 offset:1872
	ds_load_u16_d16 v157, v44 offset:2016
	ds_load_u16_d16 v158, v44 offset:2160
	ds_load_u16_d16 v159, v44 offset:2304
	ds_load_u16_d16 v160, v44 offset:2448
	ds_load_u16_d16 v161, v44 offset:2592
	ds_load_u16_d16 v162, v44 offset:2736
	ds_load_u16_d16 v163, v44 offset:2880
	ds_load_u16_d16 v164, v44 offset:3024
	ds_load_u16_d16 v165, v44 offset:3168
	ds_load_u16_d16 v166, v44 offset:3312
	ds_load_u16_d16 v167, v44 offset:3456
	ds_load_u16_d16 v168, v44 offset:3600
	ds_load_u16_d16 v169, v44 offset:3744
	ds_load_u16_d16 v170, v44 offset:3888
	ds_load_u16_d16 v171, v44 offset:4032
	ds_load_u16_d16 v172, v44 offset:4176
	ds_load_u16_d16 v173, v44 offset:4320
	ds_load_u16_d16 v174, v44 offset:4464
	ds_load_u16_d16 v175, v44 offset:4608
	ds_load_u16_d16 v176, v44 offset:4752
	ds_load_u16_d16 v177, v44 offset:4896
	ds_load_u16_d16 v178, v44 offset:5040
	ds_load_u16_d16 v179, v44 offset:5184
	ds_load_u16_d16 v180, v44 offset:5328
	ds_load_u16_d16 v181, v44 offset:5472
	ds_load_u16_d16 v182, v44 offset:5616
	ds_load_u16_d16 v183, v44 offset:5760
	ds_load_u16_d16 v184, v44 offset:5904
	ds_load_u16_d16 v185, v44 offset:6048
	ds_load_u16_d16 v186, v44 offset:6192
	ds_load_u16_d16 v187, v44 offset:6336
	ds_load_u16_d16 v188, v44 offset:6480
	ds_load_u16_d16 v189, v44 offset:6624
	ds_load_u16_d16 v190, v44 offset:6768
	ds_load_u16_d16 v191, v44 offset:6912
	ds_load_u16_d16 v192, v44 offset:7056
	ds_load_u16_d16 v193, v44 offset:7200
	ds_load_u16_d16 v194, v44 offset:7344
	ds_load_u16_d16 v195, v44 offset:7488
	ds_load_u16_d16 v196, v44 offset:7632
	ds_load_u16_d16 v197, v44 offset:7776
	ds_load_u16_d16 v198, v44 offset:7920
	ds_load_u16_d16 v199, v44 offset:8064
	ds_load_u16_d16 v200, v44 offset:8208
	ds_load_u16_d16 v201, v44 offset:8352
	ds_load_u16_d16 v202, v44 offset:8496
	ds_load_u16_d16 v203, v44 offset:8640
	ds_load_u16_d16 v204, v44 offset:8784
	ds_load_u16_d16 v205, v44 offset:8928
	ds_load_u16_d16 v206, v44 offset:9072
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v143, v44 offset:72
	ds_load_u16_d16_hi v144, v44 offset:216
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v145, v44 offset:360
	ds_load_u16_d16_hi v146, v44 offset:504
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v147, v44 offset:648
	ds_load_u16_d16_hi v148, v44 offset:792
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v149, v44 offset:936
	ds_load_u16_d16_hi v150, v44 offset:1080
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v151, v44 offset:1224
	ds_load_u16_d16_hi v152, v44 offset:1368
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v153, v44 offset:1512
	ds_load_u16_d16_hi v154, v44 offset:1656
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v155, v44 offset:1800
	ds_load_u16_d16_hi v156, v44 offset:1944
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v157, v44 offset:2088
	ds_load_u16_d16_hi v158, v44 offset:2232
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v159, v44 offset:2376
	ds_load_u16_d16_hi v160, v44 offset:2520
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v161, v44 offset:2664
	ds_load_u16_d16_hi v162, v44 offset:2808
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v163, v44 offset:2952
	ds_load_u16_d16_hi v164, v44 offset:3096
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v165, v44 offset:3240
	ds_load_u16_d16_hi v166, v44 offset:3384
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v167, v44 offset:3528
	ds_load_u16_d16_hi v168, v44 offset:3672
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v169, v44 offset:3816
	ds_load_u16_d16_hi v170, v44 offset:3960
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v171, v44 offset:4104
	ds_load_u16_d16_hi v172, v44 offset:4248
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v173, v44 offset:4392
	ds_load_u16_d16_hi v174, v44 offset:4536
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v175, v44 offset:4680
	ds_load_u16_d16_hi v176, v44 offset:4824
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v177, v44 offset:4968
	ds_load_u16_d16_hi v178, v44 offset:5112
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v179, v44 offset:5256
	ds_load_u16_d16_hi v180, v44 offset:5400
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v181, v44 offset:5544
	ds_load_u16_d16_hi v182, v44 offset:5688
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v183, v44 offset:5832
	ds_load_u16_d16_hi v184, v44 offset:5976
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v185, v44 offset:6120
	ds_load_u16_d16_hi v186, v44 offset:6264
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v187, v44 offset:6408
	ds_load_u16_d16_hi v188, v44 offset:6552
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v189, v44 offset:6696
	ds_load_u16_d16_hi v190, v44 offset:6840
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v191, v44 offset:6984
	ds_load_u16_d16_hi v192, v44 offset:7128
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v193, v44 offset:7272
	ds_load_u16_d16_hi v194, v44 offset:7416
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v195, v44 offset:7560
	ds_load_u16_d16_hi v196, v44 offset:7704
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v197, v44 offset:7848
	ds_load_u16_d16_hi v198, v44 offset:7992
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v199, v44 offset:8136
	ds_load_u16_d16_hi v200, v44 offset:8280
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v201, v44 offset:8424
	ds_load_u16_d16_hi v202, v44 offset:8568
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v203, v44 offset:8712
	ds_load_u16_d16_hi v204, v44 offset:8856
	s_waitcnt lgkmcnt(62)
	ds_load_u16_d16_hi v205, v44 offset:9000
	ds_load_u16_d16_hi v206, v44 offset:9144
	s_waitcnt lgkmcnt(56)
	v_wmma_f32_16x16x16_f16 v[1:8], v[79:86], v[143:150], v[1:8]
	s_waitcnt lgkmcnt(48)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[87:94], v[151:158], v[1:8]
	s_waitcnt lgkmcnt(40)
	v_wmma_f32_16x16x16_f16 v[1:8], v[95:102], v[159:166], v[1:8]
	s_waitcnt lgkmcnt(32)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[103:110], v[167:174], v[1:8]
	s_waitcnt lgkmcnt(24)
	v_wmma_f32_16x16x16_f16 v[1:8], v[111:118], v[175:182], v[1:8]
	s_waitcnt lgkmcnt(16)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[119:126], v[183:190], v[1:8]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[1:8], v[127:134], v[191:198], v[1:8]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[135:142], v[199:206], v[1:8]
	s_barrier
	v_add_nc_u32_e32 v73, 0x8000, v73
	v_add_nc_u32_e32 v74, 0x8000, v74
	v_add_nc_u32_e32 v75, 0x8000, v75
	v_add_nc_u32_e32 v60, 0x8000, v60
	v_add_nc_u32_e32 v76, 0x100, v76
	v_add_nc_u32_e32 v77, 0x100, v77
	v_add_nc_u32_e32 v78, 0x100, v78
	v_add_nc_u32_e32 v63, 0x100, v63
	s_add_i32 s16, s16, 8
	s_waitcnt vmcnt(7)
	ds_store_2addr_b64 v58, v[29:30], v[31:32] offset1:1
	s_waitcnt vmcnt(3)
	ds_store_2addr_b64 v67, v[37:38], v[39:40] offset1:1
	ds_store_2addr_b64 v64, v[21:22], v[23:24] offset1:1
	s_waitcnt vmcnt(2)
	ds_store_2addr_b64 v69, v[33:34], v[35:36] offset1:1
	s_cmp_lt_u32 s16, 16
	ds_store_2addr_b64 v65, v[13:14], v[15:16] offset1:1
	s_waitcnt vmcnt(1)
	ds_store_2addr_b64 v70, v[25:26], v[27:28] offset1:1
	ds_store_2addr_b64 v66, v[9:10], v[11:12] offset1:1
	s_waitcnt vmcnt(0)
	ds_store_2addr_b64 v72, v[17:18], v[19:20] offset1:1
	s_cbranch_scc1 .LBB0_1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_load_2addr_b64 v[13:16], v71 offset1:1
	ds_load_2addr_b64 v[9:12], v68 offset1:1
	ds_load_2addr_b64 v[21:24], v62 offset1:1
	ds_load_2addr_b64 v[17:20], v61 offset1:1
	ds_load_2addr_b64 v[29:32], v59 offset1:1
	ds_load_2addr_b64 v[25:28], v57 offset1:1
	ds_load_2addr_b64 v[37:40], v54 offset1:1
	ds_load_2addr_b64 v[33:36], v53 offset1:1
	ds_load_2addr_b64 v[55:58], v52 offset1:1
	ds_load_2addr_b64 v[51:54], v51 offset1:1
	ds_load_2addr_b64 v[63:66], v50 offset1:1
	ds_load_2addr_b64 v[59:62], v49 offset1:1
	ds_load_u16_d16 v67, v44
	ds_load_u16_d16 v68, v44 offset:144
	ds_load_u16_d16 v69, v44 offset:288
	ds_load_u16_d16 v70, v44 offset:432
	ds_load_u16_d16 v71, v44 offset:576
	ds_load_u16_d16 v72, v44 offset:720
	ds_load_u16_d16 v73, v44 offset:864
	ds_load_u16_d16 v74, v44 offset:1008
	ds_load_u16_d16 v75, v44 offset:1152
	ds_load_u16_d16 v76, v44 offset:1296
	ds_load_u16_d16 v77, v44 offset:1440
	ds_load_u16_d16 v78, v44 offset:1584
	ds_load_u16_d16 v79, v44 offset:1728
	ds_load_u16_d16 v80, v44 offset:1872
	ds_load_u16_d16 v81, v44 offset:2016
	ds_load_u16_d16 v82, v44 offset:2160
	ds_load_2addr_b64 v[87:90], v47 offset1:1
	ds_load_2addr_b64 v[83:86], v45 offset1:1
	ds_load_2addr_b64 v[95:98], v48 offset1:1
	ds_load_2addr_b64 v[91:94], v46 offset1:1
	v_bfe_u32 v0, v0, 4, 1
	s_lshl_b32 s2, s20, 2
	s_and_b32 s1, s1, 0xffff
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v67, v44 offset:72
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v68, v44 offset:216
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v69, v44 offset:360
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v70, v44 offset:504
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v71, v44 offset:648
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v72, v44 offset:792
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v73, v44 offset:936
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v74, v44 offset:1080
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v75, v44 offset:1224
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v76, v44 offset:1368
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v77, v44 offset:1512
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v78, v44 offset:1656
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v79, v44 offset:1800
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v80, v44 offset:1944
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v81, v44 offset:2088
	s_waitcnt lgkmcnt(19)
	ds_load_u16_d16_hi v82, v44 offset:2232
	ds_load_u16_d16 v99, v44 offset:2304
	ds_load_u16_d16 v100, v44 offset:2448
	ds_load_u16_d16 v101, v44 offset:2592
	ds_load_u16_d16 v102, v44 offset:2736
	ds_load_u16_d16 v103, v44 offset:2880
	ds_load_u16_d16 v104, v44 offset:3024
	ds_load_u16_d16 v105, v44 offset:3168
	ds_load_u16_d16 v106, v44 offset:3312
	ds_load_u16_d16 v107, v44 offset:3456
	ds_load_u16_d16 v108, v44 offset:3600
	ds_load_u16_d16 v109, v44 offset:3744
	ds_load_u16_d16 v110, v44 offset:3888
	ds_load_u16_d16 v111, v44 offset:4032
	ds_load_u16_d16 v112, v44 offset:4176
	ds_load_u16_d16 v113, v44 offset:4320
	ds_load_u16_d16 v114, v44 offset:4464
	ds_load_u16_d16 v115, v44 offset:4608
	ds_load_u16_d16 v116, v44 offset:4752
	ds_load_u16_d16 v117, v44 offset:4896
	ds_load_u16_d16 v118, v44 offset:5040
	ds_load_u16_d16 v119, v44 offset:5184
	ds_load_u16_d16 v120, v44 offset:5328
	ds_load_u16_d16 v121, v44 offset:5472
	ds_load_u16_d16 v122, v44 offset:5616
	s_waitcnt lgkmcnt(32)
	v_wmma_f32_16x16x16_f16 v[1:8], v[9:16], v[67:74], v[1:8]
	v_or3_b32 v0, v43, v0, s3
	s_mov_b32 s3, 0x31027000
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v99, v44 offset:2376
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v100, v44 offset:2520
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v101, v44 offset:2664
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v102, v44 offset:2808
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v103, v44 offset:2952
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v104, v44 offset:3096
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v105, v44 offset:3240
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v106, v44 offset:3384
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v107, v44 offset:3528
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v108, v44 offset:3672
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v109, v44 offset:3816
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v110, v44 offset:3960
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v111, v44 offset:4104
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v112, v44 offset:4248
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v113, v44 offset:4392
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v114, v44 offset:4536
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v115, v44 offset:4680
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v116, v44 offset:4824
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v117, v44 offset:4968
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v118, v44 offset:5112
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v119, v44 offset:5256
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v120, v44 offset:5400
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v121, v44 offset:5544
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v122, v44 offset:5688
	ds_load_u16_d16 v123, v44 offset:5760
	ds_load_u16_d16 v124, v44 offset:5904
	ds_load_u16_d16 v125, v44 offset:6048
	ds_load_u16_d16 v126, v44 offset:6192
	ds_load_u16_d16 v127, v44 offset:6336
	ds_load_u16_d16 v128, v44 offset:6480
	ds_load_u16_d16 v129, v44 offset:6624
	ds_load_u16_d16 v130, v44 offset:6768
	ds_load_u16_d16 v131, v44 offset:6912
	ds_load_u16_d16 v132, v44 offset:7056
	ds_load_u16_d16 v133, v44 offset:7200
	ds_load_u16_d16 v134, v44 offset:7344
	ds_load_u16_d16 v135, v44 offset:7488
	ds_load_u16_d16 v136, v44 offset:7632
	ds_load_u16_d16 v137, v44 offset:7776
	ds_load_u16_d16 v138, v44 offset:7920
	ds_load_u16_d16 v139, v44 offset:8064
	ds_load_u16_d16 v140, v44 offset:8208
	ds_load_u16_d16 v141, v44 offset:8352
	ds_load_u16_d16 v142, v44 offset:8496
	ds_load_u16_d16 v143, v44 offset:8640
	ds_load_u16_d16 v144, v44 offset:8784
	ds_load_u16_d16 v145, v44 offset:8928
	ds_load_u16_d16 v146, v44 offset:9072
	v_wmma_f32_16x16x16_f16 v[1:8], v[17:24], v[75:82], v[1:8]
	v_lshl_or_b32 v0, v0, 9, s2
	s_mov_b32 s2, 0x8000
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v123, v44 offset:5832
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v124, v44 offset:5976
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v125, v44 offset:6120
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v126, v44 offset:6264
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v127, v44 offset:6408
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v128, v44 offset:6552
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v129, v44 offset:6696
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v130, v44 offset:6840
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v131, v44 offset:6984
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v132, v44 offset:7128
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v133, v44 offset:7272
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v134, v44 offset:7416
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v135, v44 offset:7560
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v136, v44 offset:7704
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v137, v44 offset:7848
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v138, v44 offset:7992
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v139, v44 offset:8136
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v140, v44 offset:8280
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v141, v44 offset:8424
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v142, v44 offset:8568
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v143, v44 offset:8712
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v144, v44 offset:8856
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v145, v44 offset:9000
	s_waitcnt lgkmcnt(23)
	ds_load_u16_d16_hi v146, v44 offset:9144
	v_wmma_f32_16x16x16_f16 v[1:8], v[25:32], v[99:106], v[1:8]
	v_lshl_add_u32 v0, v42, 2, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[33:40], v[107:114], v[1:8]
	v_lshl_add_u32 v0, v41, 2, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[51:58], v[115:122], v[1:8]
	v_add_nc_u32_e32 v9, 0x1000, v0
	s_waitcnt lgkmcnt(16)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[59:66], v[123:130], v[1:8]
	s_waitcnt lgkmcnt(8)
	v_wmma_f32_16x16x16_f16 v[1:8], v[83:90], v[131:138], v[1:8]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[91:98], v[139:146], v[1:8]
	s_clause 0x7
	buffer_store_b32 v1, v0, s[0:3], 0 offen
	buffer_store_b32 v2, v0, s[0:3], 0 offen offset:1024
	buffer_store_b32 v3, v0, s[0:3], 0 offen offset:2048
	buffer_store_b32 v4, v0, s[0:3], 0 offen offset:3072
	buffer_store_b32 v5, v9, s[0:3], 0 offen
	buffer_store_b32 v6, v9, s[0:3], 0 offen offset:1024
	buffer_store_b32 v7, v9, s[0:3], 0 offen offset:2048
	buffer_store_b32 v8, v9, s[0:3], 0 offen offset:3072
	s_barrier
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_dispatch_0_matmul_64x128x512_f16xf16xf32
		.amdhsa_group_segment_fixed_size 17664
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
		.amdhsa_next_free_vgpr 207
		.amdhsa_next_free_sgpr 21
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
		.amdhsa_inst_pref_size 34
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
	.size	matmul_dispatch_0_matmul_64x128x512_f16xf16xf32, .Lfunc_end0-matmul_dispatch_0_matmul_64x128x512_f16xf16xf32

	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.num_vgpr, 207
	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.num_agpr, 0
	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.numbered_sgpr, 21
	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.num_named_barrier, 0
	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.private_seg_size, 0
	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.uses_vcc, 0
	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.uses_flat_scratch, 0
	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.has_dyn_sized_stack, 0
	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.has_recursion, 0
	.set matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.has_indirect_call, 0
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
    .group_segment_fixed_size: 17664
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 128
    .name:           matmul_dispatch_0_matmul_64x128x512_f16xf16xf32
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     21
    .sgpr_spill_count: 0
    .symbol:         matmul_dispatch_0_matmul_64x128x512_f16xf16xf32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     207
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
