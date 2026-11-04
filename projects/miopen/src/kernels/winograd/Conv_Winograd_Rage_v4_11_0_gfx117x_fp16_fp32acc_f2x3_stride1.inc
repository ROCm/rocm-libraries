// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.macro _vop1 op:req, vdst:req, src:req
    .long (0b0111111 << 25) | (\vdst << 17) | (\op << 9) | \src
.endm

.macro _v_rcp_f16__vop1 vdst:req, vsrc:req
    _vop1 84, \vdst, (\vsrc + /*VGPR*/ 256)
.endm

.macro _v_exp_f16__vop1 vdst:req, vsrc:req
    _vop1 88, \vdst, (\vsrc + /*VGPR*/ 256)
.endm

.macro _vop3 op:req, vdst:req, src0:req, src1:req, src2:req, opsel:req, abs:req, neg:req
    .long (0b110101 << 26) | (\op << 16) | (\opsel << 11) | (\abs << 8) | \vdst
    .long (\neg << 29) | (\src2 << 18) | (\src1 << 9) | \src0
.endm

.macro _v_cndmask_b16__vop3 vdst:req, vsrc0:req, vsrc1:req, src2:req, opsel:req
    _vop3 605, \vdst, (\vsrc0 + /*VGPR*/ 256), (\vsrc1 + /*VGPR*/ 256), \src2, \opsel, 0, 0
.endm

.macro _v_cmp_lt_f16__vop3 sdst:req, vsrc0:req, src1:req, opsel:req
    _vop3 1, \sdst, (\vsrc0 + /*VGPR*/ 256), \src1, 128, \opsel, 0, 0
.endm

.macro _v_cmp_lt_u16__vop3 sdst:req, vsrc0:req, ssrc1:req, opsel:req
    _vop3 57, \sdst, (\vsrc0 + /*VGPR*/ 256), \ssrc1, 128, \opsel, 0, 0
.endm

.macro _v_cmp_gt_f16_vccahvhv  src0, src1
    .long  0xd404196a
    .long  0x02020100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _vop3p op:req, vdst:req, src0:req, src1:req, src2:req, opsel:req, opsel_hi:req, opsel_hi2:req, neg:req, neg_hi:req
    .long (0b11001100 << 24) | (\op << 16) | (\opsel_hi2 << 14) | (\opsel << 11) | (\neg_hi << 8) | \vdst
    .long (\neg << 29) | (\opsel_hi << 27) | ((\src2 | 0x80) << 18) | (\src1 << 9) | \src0
.endm

.macro _vop3p_lit op:req, vdst:req, src0:req, src1:req, src2:req, lit:req, opsel:req, opsel_hi:req, opsel_hi2:req, neg:req, neg_hi:req
    .long (0b11001100 << 24) | (\op << 16) | (\opsel_hi2 << 14) | (\opsel << 11) | (\neg_hi << 8) | \vdst
    .long (\neg << 29) | (\opsel_hi << 27) | ((\src2 | 0x80) << 18) | (\src1 << 9) | \src0
    .long \lit
.endm

.macro _v_pk_ashrrev_i16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 6, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_add_u16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 10, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_sub_u16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 11, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_min_u16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 13, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_add_f16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 15, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_mul_f16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 16, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

.macro _v_pk_mul_f16__vop3p_lit vdst:req, lit:req, src1:req, opsel:req, opsel_hi:req
    _vop3p_lit 16, \vdst, 255, \src1, 0, \lit, \opsel, \opsel_hi, 0, 0, 0
.endm

.macro _v_pk_max_num_f16__vop3p vdst:req, src0:req, src1:req, opsel:req, opsel_hi:req, neg:req, neg_hi:req
    _vop3p 18, \vdst, \src0, \src1, 0, \opsel, \opsel_hi, 0, \neg, \neg_hi
.endm

s_version 0x2015
s_set_inst_prefetch_distance 0x3
v_lshlrev_b32 v1, 7, v0
s_getpc_b64 s[8:9]
s_mov_b32 s10, 0xca44
s_mov_b32 s11, 0x31014000
buffer_load_b32 v2, v1, s[8:11], 0 offen
s_waitcnt vmcnt(0)
s_getpc_b64 s[6:7]
s_load_b512 s[16:31], s[2:3], null
s_load_b256 s[32:39], s[2:3], 0x40
s_load_b64 s[40:41], s[2:3], 0x60
v_readfirstlane_b32 s15, v0
s_lshr_b32 s15, s15, 6
v_and_b32 v7, 0xff, v0
v_and_b32 v8, 1, v0
v_mul_u32_u24 v2, 0x1140, v8
v_bfe_u32 v12, v7, 1, 5
v_and_b32 v1, 15, v7
v_and_b32 v4, 15, v7
v_lshl_add_u32 v2, v12, 2, v2
v_lshlrev_b32 v1, 2, v1
v_lshlrev_b32 v4, 3, v4
v_mul_u32_u24 v3, 0x1040, v8
v_bfe_u32 v12, v7, 6, 1
v_bfe_u32 v11, v7, 4, 2
v_bfe_u32 v10, v7, 4, 2
v_bfe_u32 v9, v7, 1, 5
v_lshl_add_u32 v2, v12, 9, v2
v_mad_u32_u24 v1, v11, 0x440, v1
v_lshl_add_u32 v4, v10, 10, v4
v_lshl_add_u32 v3, v9, 3, v3
v_bfe_u32 v12, v7, 7, 1
v_bfe_u32 v11, v7, 6, 2
v_bfe_u32 v10, v7, 6, 2
v_bfe_u32 v9, v7, 6, 2
v_mad_u32_u24 v2, v12, 0x440, v2
v_lshl_add_u32 v1, v11, 7, v1
v_lshl_add_u32 v4, v10, 8, v4
v_lshl_add_u32 v3, v9, 10, v3
s_and_b32 s42, 1, s15
s_mul_i32 s8, s42, 0x800
s_mul_i32 s9, s42, 0x800
s_mul_i32 s10, s42, 0x400
s_mul_i32 s13, s42, 4
s_mul_i32 s14, s42, 16
s_and_b32 s42, 3, s15
s_mul_i32 s11, s42, 0x800
s_mul_i32 s12, s42, 0x800
s_waitcnt lgkmcnt(0)
s_and_b32 s22, s22, 0x3ffff
s_bitcmp1_b32 s22, 6
s_cbranch_scc0 18
s_and_b32 s25, 0xffff, s25
s_and_b32 s27, 0xffff, s27
s_and_b32 s29, 0xffff, s29
s_and_b32 s39, 0xffff, s39
s_load_b64 s[24:25], s[24:25], null
s_load_b64 s[26:27], s[26:27], null
s_load_b64 s[28:29], s[28:29], null
s_bitcmp1_b32 s22, 7
s_cbranch_scc0 2
s_load_b64 s[38:39], s[38:39], null
s_bitcmp1_b32 s22, 13
s_cbranch_scc0 6
s_load_b64 s[42:43], s[2:3], 0x68
s_load_b128 s[44:47], s[2:3], 0x70
s_load_b64 s[48:49], s[2:3], 0x80
s_bitcmp1_b32 s22, 9
s_cbranch_scc0 6
s_load_b64 s[50:51], s[2:3], 0x88
s_load_b256 s[52:59], s[2:3], 0x90
s_load_b64 s[60:61], s[2:3], 0xb0
s_bitcmp1_b32 s22, 10
s_cbranch_scc0 4
s_load_b64 s[62:63], s[2:3], 0xb8
s_load_b64 s[64:65], s[2:3], 0xc0
s_bitcmp1_b32 s22, 14
s_cbranch_scc0 2
s_load_b32 s66, s[2:3], 0xc8
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 9, 1), 0x0
s_bitcmp1_b32 s22, 3
s_cbranch_scc0 2
s_denorm_mode 0xf
s_round_mode 0x0
s_bitcmp1_b32 s22, 10
s_cbranch_scc0 0
s_waitcnt lgkmcnt(0)
s_bitcmp1_b32 s22, 13
s_cbranch_scc0 8
s_add_u32 s24, s24, s42
s_addc_u32 s25, s25, s43
s_add_u32 s28, s28, s46
s_addc_u32 s29, s29, s47
s_add_u32 s26, s26, s44
s_addc_u32 s27, s27, s45
s_add_u32 s38, s38, s48
s_addc_u32 s39, s39, s49
s_bitcmp1_b32 s22, 9
s_cbranch_scc1 14
s_mov_b32 s52, s19
s_mul_i32 s51, s18, s19
s_mul_i32 s50, s51, s17
s_mov_b32 s60, s37
s_mul_i32 s59, s36, s37
s_mul_i32 s58, s59, s20
s_mul_i32 s67, s32, s33
s_bitcmp1_b32 s22, 2
s_cselect_b32 s68, s20, s17
s_mul_i32 s68, s67, s68
s_mov_b32 s56, s33
s_bitcmp1_b32 s22, 2
s_cselect_b32 s55, s68, s67
s_cselect_b32 s54, s67, s68
s_bitcmp1_b32 s22, 10
s_cbranch_scc0 60
v_clz_i32_u32 v10, s21
v_lshlrev_b32 v11, v10, s21
v_and_b32 v9, 0xffffff00, v11
v_cmp_eq_u32 vcc, 0x80000000, v11
v_cvt_f32_u32 v9, v9
v_rcp_f32 v7, v9
v_sub_co_ci_u32 v8, vcc, 32, v10, vcc
v_cvt_f32_ubyte0 v10, v11
v_fma_f32 v9, v9, v7, -1.0
v_fma_f32 v9, v10, v7, v9
v_fmaak_f32 v9, v9, v7, 0x9f000000
v_mul_f32 v9, 0x5f800000, v9
v_mov_b32 v10, 0
v_cvt_floor_i32_f32 v9, -v9
v_lshl_add_u32 v7, v7, 9, v9
v_mad_u64_u32 v[10:11], vcc, v11, v7, v[10:11]
v_sub_co_ci_u32 v7, vcc, v7, -1, vcc
v_mul_hi_u32 v9, s4, v7
v_add_co_u32 v7, vcc, v9, s4
v_add_co_ci_u32 v9, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v8
v_cndmask_b32 v7, v7, v9, vcc
v_alignbit_b32 v7, v9, v7, v8
v_readfirstlane_b32 s67, v7
s_mul_i32 s68, s67, s21
s_sub_u32 s4, s4, s68
s_mul_i32 s68, s63, s67
s_mul_hi_u32 s69, s63, s67
s_lshl_b64 s[68:69], s[68:69], 1
s_add_u32 s24, s24, s68
s_addc_u32 s25, s25, s69
s_mul_i32 s68, s64, s67
s_mul_hi_u32 s69, s64, s67
s_lshl_b64 s[68:69], s[68:69], 1
s_add_u32 s26, s26, s68
s_addc_u32 s27, s27, s69
s_mul_i32 s68, s65, s67
s_mul_hi_u32 s69, s65, s67
s_lshl_b64 s[68:69], s[68:69], 1
s_add_u32 s28, s28, s68
s_addc_u32 s29, s29, s69
s_mul_i32 s68, s20, s67
s_lshl_b32 s68, s68, 1
s_add_u32 s38, s38, s68
s_addc_u32 s39, s39, 0
s_and_b32 s66, s66, 0xff
s_bitcmp1_b32 s22, 8
s_cselect_b32 s67, 1, 0
s_bitcmp1_b32 s22, 14
s_cselect_b32 s66, s66, s67
s_add_u32 s69, s20, 31
s_lshr_b32 s69, s69, 5
v_clz_i32_u32 v10, s69
v_lshlrev_b32 v11, v10, s69
v_and_b32 v12, 0xffffff00, v11
v_cmp_eq_u32 vcc, 0x80000000, v11
v_cvt_f32_u32 v12, v12
v_rcp_f32 v7, v12
v_sub_co_ci_u32 v9, vcc, 32, v10, vcc
v_cvt_f32_ubyte0 v10, v11
v_fma_f32 v12, v12, v7, -1.0
v_fma_f32 v12, v10, v7, v12
v_fmaak_f32 v12, v12, v7, 0x9f000000
v_mul_f32 v12, 0x5f800000, v12
v_mov_b32 v10, 0
v_cvt_floor_i32_f32 v12, -v12
v_lshl_add_u32 v7, v7, 9, v12
v_mad_u64_u32 v[10:11], vcc, v11, v7, v[10:11]
v_sub_co_ci_u32 v7, vcc, v7, -1, vcc
v_mul_hi_u32 v10, s21, v7
v_add_co_u32 v7, vcc, v10, s21
v_add_co_ci_u32 v10, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v9
v_cndmask_b32 v7, v7, v10, vcc
v_alignbit_b32 v7, v10, v7, v9
v_clz_i32_u32 v10, v7
v_lshlrev_b32 v11, v10, v7
v_and_b32 v12, 0xffffff00, v11
v_cmp_eq_u32 vcc, 0x80000000, v11
v_cvt_f32_u32 v12, v12
v_rcp_f32 v8, v12
v_sub_co_ci_u32 v9, vcc, 32, v10, vcc
v_cvt_f32_ubyte0 v10, v11
v_fma_f32 v12, v12, v8, -1.0
v_fma_f32 v12, v10, v8, v12
v_fmaak_f32 v12, v12, v8, 0x9f000000
v_mul_f32 v12, 0x5f800000, v12
v_mov_b32 v10, 0
v_cvt_floor_i32_f32 v12, -v12
v_lshl_add_u32 v8, v8, 9, v12
v_mad_u64_u32 v[10:11], vcc, v11, v8, v[10:11]
v_sub_co_ci_u32 v8, vcc, v8, -1, vcc
v_mul_hi_u32 v10, s4, v8
v_add_co_u32 v8, vcc, v10, s4
v_add_co_ci_u32 v10, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v9
v_cndmask_b32 v8, v8, v10, vcc
v_alignbit_b32 v8, v10, v8, v9
v_readfirstlane_b32 s67, v7
v_readfirstlane_b32 s70, v8
s_cmp_ge_u32 s70, s69
s_cbranch_scc1 12633
s_mul_i32 s21, s67, s69
s_mul_i32 s68, s70, s67
s_sub_u32 s68, s4, s68
s_mul_i32 s71, s70, 32
s_sub_u32 s20, s20, s71
s_min_u32 s20, s20, 32
s_lshl_b32 s71, s71, 1
s_mul_i32 s72, s71, s54
s_mul_hi_u32 s73, s71, s54
s_add_u32 s26, s26, s72
s_addc_u32 s27, s27, s73
s_mul_i32 s72, s71, s59
s_mul_hi_u32 s73, s71, s59
s_add_u32 s28, s28, s72
s_addc_u32 s29, s29, s73
s_add_u32 s38, s38, s71
s_addc_u32 s39, s39, 0
s_mov_b32 s21, s67
s_mov_b32 s4, s68
s_lshl_b32 s92, s51, 1
s_lshl_b32 s93, s55, 1
s_lshl_b32 s85, s59, 1
v_readfirstlane_b32 s95, v0
s_lshr_b32 s95, s95, 4
s_lshr_b32 s94, 0x3210, s95
s_and_b32 s94, 15, s94
s_lshr_b32 s84, 0x4400, s95
s_and_b32 s84, 15, s84
s_sub_u32 s82, s17, s94
s_cmov_b32 s82, 0
s_mul_i32 s95, s94, s92
s_add_u32 s24, s24, s95
s_addc_u32 s25, s25, 0
s_mul_i32 s95, s84, s85
s_add_u32 s28, s28, s95
s_addc_u32 s29, s29, 0
s_lshl_b32 s80, s92, 2
s_lshl_b32 s81, s93, 2
s_and_b32 s39, s39, 0xffff
s_add_u32 s39, s39, 0x20000
s_mov_b64 s[92:93], s[38:39]
s_mov_b32 s94, s20
s_mov_b32 s95, 0x11014000
s_bitcmp0_b32 s22, 7
s_cselect_b32 s94, 0, s94
v_and_b32 v32, 1, v0
v_lshlrev_b32 v32, 1, v32
v_bfe_u32 v33, v0, 1, 2
v_mad_u32_u16 v32, v33, 8, v32 op_sel:[0,0,0,0]
v_add_nc_u32 v32, s84, v32
v_add_nc_u32 v33, 1, v32
buffer_load_d16_b16 v31, v32, s[92:95], 0 idxen
buffer_load_d16_hi_b16 v31, v33, s[92:95], 0 idxen
v_cvt_f16_f32 v32, s40
v_cvt_f16_f32 v33, s41
v_readfirstlane_b32 s40, v32
v_readfirstlane_b32 s41, v33
s_pack_ll_b32_b16 s40, s40, s40
s_pack_ll_b32_b16 s41, s41, s41
s_mov_b32 s92, 0x43f0
s_mov_b32 s93, 0x6170
s_mov_b32 s94, 0x7af0
s_mov_b32 s95, 0x9a70
s_cmp_ge_u32 s15, 2
s_cmov_b32 s92, 0x52b0
s_cmov_b32 s93, 0x6e30
s_cmov_b32 s94, 0x8ab0
s_cmov_b32 s95, 0xb130
s_cmp_eq_u32 s66, 0
s_cmov_b32 s91, s92
s_cmov_b32 s40, 0x3c003c00
s_cmp_eq_u32 s66, 4
s_cmov_b32 s91, s93
s_cmp_eq_u32 s66, 1
s_cmov_b32 s91, s92
s_cmp_eq_u32 s66, 2
s_cmov_b32 s91, s94
s_cmp_eq_u32 s66, 3
s_cmov_b32 s91, s95
v_and_b32 v34, 0xff, v0
s_bitcmp1_b32 s22, 31
s_cselect_b32 s92, 2, 0
s_cselect_b32 s93, 0, 2
v_bfe_u32 v32, v34, 2, s92
v_bfe_u32 v35, v34, 6, s93
v_add_nc_u32 v35, v32, v35
v_lshrrev_b32 v36, 2, v34
v_lshrrev_b32 v36, s92, v36
v_and_b32 v32, 3, v36
v_bfe_u32 v6, v36, 2, 1
v_lshl_add_u32 v32, v6, 3, v32
v_bfe_u32 v6, v36, 3, 1
v_and_b32 v33, 15, v34
v_lshl_add_u32 v32, v6, 2, v32
v_bfe_u32 v5, v34, 6, 2
v_mul_u32_u24 v6, 24, v32
v_lshl_add_u32 v33, v5, 4, v33
v_xor_b32 v32, v34, v34 quad_perm:[0,0,0,1]
v_mul_u32_u24 v5, 24, v33
v_lshl_add_u32 v6, v32, 3, v6
v_bfe_u32 v33, v34, 4, 2
v_mad_u32_u16 v6, v35, 0x620, v6 op_sel:[0,0,0,0]
v_mad_u32_u16 v5, v33, 0x620, v5 op_sel:[0,0,0,0]
s_sub_u32 s95, s17, 1
s_and_b32 s95, s95, 15
s_add_u32 s95, s95, 1
s_bitcmp1_b32 s22, 1
s_cselect_b32 s96, s33, 1
s_cselect_b32 s64, -1, 1
s_sub_u32 s94, s96, 1
s_bitcmp1_b32 s22, 0
s_cselect_b32 s96, s32, 1
s_cselect_b32 s61, -1, 1
s_sub_u32 s96, s96, 1
s_mul_i32 s61, s61, s56
s_mul_i32 s96, s96, s56
s_add_u32 s94, s94, s96
s_bitcmp1_b32 s22, 31
s_cselect_b32 s93, 0, 1
s_cselect_b32 s92, 4, 0
s_cselect_b32 s96, 2, 0
s_cselect_b32 s97, 2, 4
v_bfe_u32 v33, v0, 2, 6
v_and_b32 v34, s15, 3
v_bfe_u32 v32, v33, 0, s96
v_mad_u32_u24 v32, v34, s93, v32
v_mov_b32 v30, s97
v_bfe_u32 v30, v33, s96, v30
v_mad_u32_u24 v30, v34, s92, v30
v_mov_b32 v33, s94
v_mad_u32_u24 v11, s55, v32, v33
s_add_u32 s93, s36, 1
s_lshr_b32 s93, s93, 1
s_lshl_b32 s94, s93, 1
s_add_u32 s95, s37, 1
s_lshr_b32 s95, s95, 1
s_lshl1_add_u32 s95, s95, 2
s_pack_ll_b32_b16 s30, s95, s93
s_pack_ll_b32_b16 s42, s19, s18
s_sub_u32 s43, s94, s34
s_sub_u32 s92, s95, s35
s_pack_ll_b32_b16 s43, s92, s43
s_pack_ll_b32_b16 s45, s37, s36
s_sub_u32 s92, s95, 1
s_pack_ll_b32_b16 s46, s92, s94
v_and_b32 v33, 63, v0
v_bfe_u32 v34, v0, 6, 1
v_and_b32 v36, 1, v0
v_and_b32 v37, 63, v0
v_mad_u32_u16 v35, s93, s95, 0 op_sel:[0,0,0,0]
v_mad_u32_u16 v34, 2, s4, v34 op_sel:[0,0,0,0]
v_mad_u32_u16 v34, 62, v34, v33 op_sel:[0,0,0,0]
v_mad_u32_u16 v37, 0x7c, s21, 0 op_sel:[0,0,0,0]
v_cmp_eq_u32 vcc, 0, v36
v_cndmask_b32 v42, v35, s95, vcc
v_cndmask_b32 v34, v37, v34, vcc
v_clz_i32_u32 v48, v42
v_lshlrev_b32 v49, v48, v42
v_and_b32 v47, 0xffffff00, v49
v_cmp_eq_u32 vcc, 0x80000000, v49
v_cvt_f32_u32 v47, v47
v_rcp_f32 v43, v47
v_sub_co_ci_u32 v44, vcc, 32, v48, vcc
v_cvt_f32_ubyte0 v48, v49
v_fma_f32 v47, v47, v43, -1.0
v_fma_f32 v47, v48, v43, v47
v_fmaak_f32 v47, v47, v43, 0x9f000000
v_mul_f32 v47, 0x5f800000, v47
v_mov_b32 v48, 0
v_cvt_floor_i32_f32 v47, -v47
v_lshl_add_u32 v43, v43, 9, v47
v_mad_u64_u32 v[48:49], vcc, v49, v43, v[48:49]
v_sub_co_ci_u32 v43, vcc, v43, -1, vcc
v_mov_b32 v46, v44 quad_perm:[1,1,1,1]
v_mov_b32 v44, v44 quad_perm:[0,0,0,0]
v_mov_b32 v45, v43 quad_perm:[1,1,1,1]
v_mov_b32 v43, v43 quad_perm:[0,0,0,0]
s_mov_b32 s96, s96
v_mul_hi_u32 v47, v34, v43
v_mul_hi_u32 v48, v34, v45
v_add_co_u32 v32, vcc, v47, v34
v_add_co_u32 v28, vcc, v48, v34
v_add_co_ci_u32 v47, vcc, 0, 0, vcc
v_add_co_ci_u32 v48, vcc, 0, 0, vcc
v_cmp_eq_u32 vcc, 32, v44
v_cmp_eq_u32 vcc, 32, v46
v_cndmask_b32 v32, v32, v47, vcc
v_cndmask_b32 v28, v28, v48, vcc
v_alignbit_b32 v32, v47, v32, v44
v_alignbit_b32 v28, v48, v28, v46
v_mad_u32_u16 v41, v32, s95, 0 op_sel:[0,0,0,0]
v_mad_u32_u16 v40, v28, s93, 0 op_sel:[0,0,0,0]
v_sub_nc_u32 v41, v34, v41
v_sub_nc_u32 v40, v32, v40
v_readlane_b32 s96, v41, 1
v_readlane_b32 s31, v40, 1
v_sub_nc_u32 v41, v41, s95
v_sub_nc_u32 v40, v40, s93
v_readlane_b32 s23, v28, 1
v_sub_nc_u32 v28, v28, s16
v_mov_b32 v41, v41 quad_perm:[0,0,2,2]
v_mov_b32 v40, v40 quad_perm:[0,0,2,2]
v_mov_b32 v28, v28 quad_perm:[0,0,2,2]
v_add_co_u32 v41, vcc, v41, v36
v_cndmask_b32 v39, 0, s95, vcc
v_add_co_ci_u32 v40, vcc, v40, 0, vcc
v_cndmask_b32 v38, 0, s93, vcc
v_add_co_ci_u32 v28, vcc, v28, 0, vcc
v_sub_nc_u32 v41, v41, v39
v_sub_nc_u32 v40, v40, v38
s_pack_ll_b32_b16 s31, s96, s31
v_lshlrev_b32 v29, 16, v40
v_bfi_b32 v29, 0xffff, v41, v29
v_min_u32 v36, v33, 63
v_cmp_eq_u32 vcc, v33, v36
v_cndmask_b32 v10, 0x80000000, v10, vcc
v_add_nc_u32 v47, v28, s16
v_mul_lo_u32 v10, v47, s50
v_mul_lo_u32 v7, v47, s58
s_mul_i32 s44, s23, s50
s_mul_i32 s47, s23, s58
v_med3_u32 v36, v33, 1, 62
v_cmp_eq_u32 vcc, v33, v36
v_cndmask_b32 v7, 0x80000000, v7, vcc
v_cmp_ge_u32 s[62:63], v47, s16
v_cndmask_b32 v10, v10, 0x80000000, s[62:63]
v_cndmask_b32 v7, v7, 0x80000000, s[62:63]
s_mov_b32 s48, 0x10000
s_mov_b32 s49, 0x30002
s_mov_b32 s53, 0x10000
s_add_u32 s65, s32, 2
s_add_u32 s67, s33, 2
s_mul_hi_u32 s93, s65, 0x55555556
s_add_u32 s65, s93, s65
s_addc_u32 s93, 0, 0
s_mov_b32 s92, s65
s_lshr_b64 s[92:93], s[92:93], 2
s_mov_b32 s65, s92
s_mul_hi_u32 s93, s67, 0x55555556
s_add_u32 s67, s93, s67
s_addc_u32 s93, 0, 0
s_mov_b32 s92, s67
s_lshr_b64 s[92:93], s[92:93], 2
s_mov_b32 s67, s92
s_sub_u32 s65, 0, s65
s_sub_u32 s67, 0, s67
s_mov_b32 s68, -1
s_mov_b32 s69, -1
s_and_b32 s25, s25, 0xffff
s_add_u32 s25, s25, 0x20000
s_and_b32 s27, s27, 0xffff
s_add_u32 s27, s27, 0x20000
s_and_b32 s29, s29, 0xffff
s_add_u32 s29, s29, 0x20000
s_mov_b64 s[72:73], s[24:25]
s_mov_b32 s74, 0
s_mov_b32 s75, 0x11014000
s_mov_b64 s[76:77], s[26:27]
s_mov_b32 s78, 0
s_mov_b32 s79, 0x11014000
s_bitset1_b32 s22, 29
s_mov_b32 s83, 0x80000000
s_mov_b32 s1, 0
s_mov_b32 s5, 0
s_mov_b32 s88, 0
s_mov_b32 s89, 0
s_mov_b32 s90, 3
s_mov_b32 s87, 0xef0
s_cmp_ge_u32 s15, 2
s_cmov_b32 s87, 0x2930
s_branch 1
s_nop 0
s_bitset1_b32 s22, 30
s_sub_u32 s88, s88, 1
s_cbranch_scc0 217
s_bitcmp1_b32 s22, 29
s_cbranch_scc0 16
s_mov_b32 s57, s62
v_and_b32 v248, 63, v0
v_lshl_add_u32 v248, v248, 3, s8
v_lshl_add_u32 v248, s1, 9, v248
v_lshl_add_u32 v249, s1, 2, s14
v_mov_b32 v250, s57
ds_store_b64 v248, v[8:9] offset:57088
ds_store_b32 v249, v250 offset:61192
s_sub_u32 s1, s1, 1
s_cselect_b32 s1, 3, s1
s_bitset0_b32 s22, 29
s_add_u32 s69, s69, 1
s_cselect_b32 s69, s67, s69
s_addc_u32 s68, s68, 0
s_cselect_b32 s68, s65, s68
s_sub_u32 s92, -1, s68
s_sub_u32 s93, -1, s69
s_mul_i32 s92, 3, s92
s_mul_i32 s93, 3, s93
s_pack_ll_b32_b16 s95, s93, s92
s_cmp_eq_u32 s95, 0
s_cbranch_scc1 40
v_pk_mad_u16 v251, v29, 0x20001, s43
_v_pk_add_u16__vop3p 251, 95, 507, 0x0, 0x3, 0x0, 0x0
v_mad_u32_u16 v20, v251, 1, v10 op_sel:[0,0,0,0]
v_cmp_lt_u16 vcc, v251, s42
v_cndmask_b32 v20, 0x80000000, v20, vcc
_v_pk_add_u16__vop3p 249, 507, 49, 0x1, 0x3, 0x0, 0x0
_v_pk_add_u16__vop3p 248, 507, 48, 0x1, 0x3, 0x0, 0x0
v_mad_u32_u16 v23, v249, s52, v20 op_sel:[1,0,0,0]
v_mad_u32_u16 v21, v248, s52, v20 op_sel:[1,0,0,0]
_v_cmp_lt_u16__vop3 106, 249, 42, 0x3
v_cndmask_b32 v23, 0x80000000, v23, vcc
_v_cmp_lt_u16__vop3 106, 248, 42, 0x3
v_cndmask_b32 v21, 0x80000000, v21, vcc
v_mad_u32_u16 v22, v249, s52, v20 op_sel:[0,0,0,0]
v_mad_u32_u16 v20, v248, s52, v20 op_sel:[0,0,0,0]
_v_cmp_lt_u16__vop3 106, 249, 42, 0x2
v_cndmask_b32 v22, 0x80000000, v22, vcc
_v_cmp_lt_u16__vop3 106, 248, 42, 0x2
v_cndmask_b32 v20, 0x80000000, v20, vcc
s_branch 87
s_bitset1_b32 s22, 29
v_pk_mad_u16 v27, v29, 0x20001, s43
v_pk_mad_u16 v24, v29, 0x20001, s46
v_mad_u32_u16 v20, v27, 1, v10 op_sel:[0,0,0,0]
v_mad_u32_u16 v8, v24, 1, v7 op_sel:[0,0,0,0]
v_cmp_lt_u16 vcc, v27, s42
v_cndmask_b32 v20, 0x80000000, v20, vcc
v_cmp_lt_u16 vcc, v24, s45
v_cndmask_b32 v8, 0x80000000, v8, vcc
v_add_co_u32 v25, vcc, v29, s31
v_cndmask_b32 v251, 0, s50, vcc
v_cndmask_b32 v252, 0, s58, vcc
_v_pk_min_u16__vop3p 26, 281, 285, 0x0, 0x3, 0x0, 0x0
v_add3_u32 v10, v10, s44, v251
v_add3_u32 v7, v7, s47, v252
_v_pk_sub_u16__vop3p 26, 285, 282, 0x0, 0x3, 0x0, 0x0
v_add_co_ci_u32 v28, s[62:63], v28, s23, vcc
v_cndmask_b32 v10, v10, 0x80000000, s[62:63]
v_cndmask_b32 v7, v7, 0x80000000, s[62:63]
_v_pk_ashrrev_i16__vop3p 26, 143, 282, 0x0, 0x2, 0x0, 0x0
_v_pk_add_u16__vop3p 249, 283, 49, 0x1, 0x3, 0x0, 0x0
_v_pk_add_u16__vop3p 248, 283, 48, 0x1, 0x3, 0x0, 0x0
v_mad_u32_u16 v23, v249, s52, v20 op_sel:[1,0,0,0]
v_mad_u32_u16 v21, v248, s52, v20 op_sel:[1,0,0,0]
_v_pk_add_u16__vop3p 250, 280, 53, 0x1, 0x3, 0x0, 0x0
_v_cmp_lt_u16__vop3 106, 249, 42, 0x3
v_cndmask_b32 v23, 0x80000000, v23, vcc
_v_cmp_lt_u16__vop3 106, 248, 42, 0x3
v_cndmask_b32 v21, 0x80000000, v21, vcc
v_mad_u32_u16 v9, v250, s60, v8 op_sel:[1,0,0,0]
v_mad_u32_u16 v22, v249, s52, v20 op_sel:[0,0,0,0]
v_mad_u32_u16 v20, v248, s52, v20 op_sel:[0,0,0,0]
_v_cmp_lt_u16__vop3 106, 250, 45, 0x3
v_cndmask_b32 v9, 0x80000000, v9, vcc
_v_cmp_lt_u16__vop3 106, 249, 42, 0x2
v_cndmask_b32 v22, 0x80000000, v22, vcc
_v_cmp_lt_u16__vop3 106, 248, 42, 0x2
v_cndmask_b32 v20, 0x80000000, v20, vcc
v_mad_u32_u16 v8, v250, s60, v8 op_sel:[0,0,0,0]
v_pk_mad_u16 v29, v26, s30, v25
_v_cmp_lt_u16__vop3 106, 250, 45, 0x2
v_cndmask_b32 v8, 0x80000000, v8, vcc
v_xor_b32 v27, v0, v0 quad_perm:[0,0,0,1]
v_add_nc_u32 v27, s93, v27
v_cmp_lt_u32 s[96:97], v27, s33
v_mul_i32_i24 v27, s64, v27
s_mul_i32 s94, s61, s92
s_sub_u32 s95, s32, 1
s_sub_u32 s95, s95, s92
s_cselect_b64 s[96:97], 0, s[96:97]
s_sub_u32 s95, s95, 1
s_cselect_b64 s[98:99], 0, s[96:97]
s_sub_u32 s95, s95, 1
s_cselect_b64 s[100:101], 0, s[98:99]
v_add3_u32 v251, v27, s94, v11
v_mad_u32_u24 v248, s54, v30, v251
v_cmp_lt_u32 vcc, v30, s20
v_cndmask_b32 v248, -2.0, v248, vcc
v_add_nc_u32 v249, s61, v248
v_add_nc_u32 v250, s61, v249
v_cndmask_b32 v248, -2.0, v248, s[96:97]
v_cndmask_b32 v249, -2.0, v249, s[98:99]
v_cndmask_b32 v250, -2.0, v250, s[100:101]
v_add_nc_u32 v252, v30, 16
v_mad_u32_u24 v24, s54, v252, v251
v_cmp_lt_u32 vcc, v252, s20
v_cndmask_b32 v24, -2.0, v24, vcc
v_add_nc_u32 v25, s61, v24
v_add_nc_u32 v26, s61, v25
v_cndmask_b32 v24, -2.0, v24, s[96:97]
v_cndmask_b32 v25, -2.0, v25, s[98:99]
v_cndmask_b32 v26, -2.0, v26, s[100:101]
v_and_b32 v252, 63, v0
v_lshl_add_u32 v27, v252, 4, s10
v_lshl_add_u32 v251, v252, 4, s11
v_mov_b32 v252, s13
v_mov_b32 v253, s57
ds_store_b128 v27, v[20:23] offset:55040
ds_store_b96 v251, v[248:250] offset:46848
ds_store_b96 v251, v[24:26] offset:47872
ds_store_b32 v252, v253 offset:61184
s_sub_u32 s88, s17, 1
s_lshr_b32 s88, s88, 4
s_branch 3403
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s89, s89, 1
s_cbranch_scc0 28
v_and_b32 v252, 63, v0
v_lshl_add_u32 v249, v252, 4, 0
v_lshl_add_u32 v250, v252, 4, s12
v_mov_b32 v251, 0
ds_load_b32 v248, v251 offset:61184
ds_load_b128 v[12:15], v249 offset:55040
ds_load_b128 v[16:19], v249 offset:56064
ds_load_b96 v[20:22], v250 offset:46848
ds_load_b96 v[24:26], v250 offset:47872
s_add_u32 s70, s82, 3
s_add_u32 s71, s82, 3
s_mov_b64 s[72:73], s[24:25]
s_mov_b64 s[76:77], s[26:27]
s_waitcnt lgkmcnt(4)
v_readfirstlane_b32 s92, v248
s_bitcmp1_b32 s92, 0
s_cmov_b32 s83, 0
s_mov_b32 s74, s83
s_mov_b32 s78, s83
s_sub_u32 s89, s17, 1
s_lshr_b32 s89, s89, 4
s_bitset0_b32 s22, 30
s_min_u32 s86, s88, s90
s_min_u32 s86, s89, s86
s_sub_u32 s88, s88, s86
s_sub_u32 s89, s89, s86
s_sub_u32 s90, s90, s86
s_add_u32 s92, s6, s87
s_addc_u32 s93, s7, 0
s_setpc_b64 s[92:93]
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
s_barrier
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v133, 1.0, v67 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v133, 1.0, v67 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v185, 1.0, v195 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v185, 1.0, v195 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:38], v[181:182], v[32:33], v[35:38]
v_wmma_f32_16x16x16_f16 v[39:42], v[181:182], v[166:167], v[39:42]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v34, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v34, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v190, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v190, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[134:137], v[196:197], v[177:178], v[134:137]
v_wmma_f32_16x16x16_f16 v[138:141], v[196:197], v[179:180], v[138:141]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[101:104], v[191:192], v[173:174], v[101:104]
v_wmma_f32_16x16x16_f16 v[105:108], v[191:192], v[175:176], v[105:108]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v34 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v34 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v190 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v190 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[68:71], v[186:187], v[168:169], v[68:71]
v_wmma_f32_16x16x16_f16 v[72:75], v[186:187], v[170:171], v[72:75]
v_mov_b32 v240, v133
v_mov_b32 v242, v67
v_mov_b32 v241, v185
v_mov_b32 v243, v195
s_setprio 2
ds_load_b64 v[183:184], v5 offset:40576
ds_store_b64 v6, v[240:241] offset:34304
ds_load_b64 v[188:189], v5 offset:40584
ds_store_b64 v6, v[242:243] offset:35456
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v133, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v185, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v100, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v172, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v34, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v190, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v67, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v195, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[198:199], v5 offset:40592
ds_store_b64 v6, v[244:245] offset:34688
ds_store_b64 v6, v[246:247] offset:35072
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v133, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v185, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v100, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v172, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v34, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v190, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v67, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v195, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_branch 1
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v248, v183, 1.0, v198 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v183, 1.0, v198 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v184, 1.0, v199 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v184, 1.0, v199 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v201, 1.0, v203 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v201, 1.0, v203 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v205, 1.0, v207 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v205, 1.0, v207 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:54], v[183:184], v[32:33], v[51:54]
v_wmma_f32_16x16x16_f16 v[55:58], v[183:184], v[166:167], v[55:58]
v_fma_mix_f32 v248, v188, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v188, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v189, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v189, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v240, v202, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v202, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v206, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v206, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[150:153], v[198:199], v[177:178], v[150:153]
v_wmma_f32_16x16x16_f16 v[154:157], v[198:199], v[179:180], v[154:157]
v_fma_mixlo_f16 v193, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v193, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v194, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v194, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[117:120], v[193:194], v[173:174], v[117:120]
v_wmma_f32_16x16x16_f16 v[121:124], v[193:194], v[175:176], v[121:124]
v_fma_mixlo_f16 v188, v248, 0.5, v188 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v188, v250, 0.5, v188 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v189, v249, 0.5, v189 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v189, v251, 0.5, v189 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v202 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v202 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v206 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v206 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[84:87], v[188:189], v[168:169], v[84:87]
v_wmma_f32_16x16x16_f16 v[88:91], v[188:189], v[170:171], v[88:91]
v_mov_b32 v240, v201
v_mov_b32 v242, v203
v_mov_b32 v241, v205
v_mov_b32 v243, v207
s_setprio 2
ds_load_b64 v[32:33], v4 offset:26048
ds_load_b64 v[166:167], v4 offset:26176
ds_store_b64 v6, v[240:241] offset:40576
ds_load_b64 v[168:169], v4 offset:30208
ds_load_b64 v[170:171], v4 offset:30336
ds_store_b64 v6, v[242:243] offset:41728
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v201, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v205, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v200, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v204, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v202, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v206, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v203, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v207, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[173:174], v4 offset:26056
ds_load_b64 v[175:176], v4 offset:26184
ds_store_b64 v6, v[244:245] offset:40960
ds_load_b64 v[177:178], v4 offset:30216
ds_load_b64 v[179:180], v4 offset:30344
ds_store_b64 v6, v[246:247] offset:41344
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v201, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v205, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v200, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v204, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v202, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v206, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v203, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v207, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_bitcmp1_b32 s22, 30
s_cbranch_scc0 6
s_setprio 1
s_mov_b32 s87, 0x13b0
s_branch 65194
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
s_barrier
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v209, -1.0, v210 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v213, -1.0, v214 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:46], v[181:182], v[32:33], v[43:46]
v_wmma_f32_16x16x16_f16 v[47:50], v[181:182], v[166:167], v[47:50]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v242, v208, -1.0, v211 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v212, -1.0, v215 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[142:145], v[196:197], v[177:178], v[142:145]
v_wmma_f32_16x16x16_f16 v[146:149], v[196:197], v[179:180], v[146:149]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v244, v208, 1.0, v209 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v212, 1.0, v213 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[109:112], v[191:192], v[173:174], v[109:112]
v_wmma_f32_16x16x16_f16 v[113:116], v[191:192], v[175:176], v[113:116]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v246, v208, -1.0, v209 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v212, -1.0, v213 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[76:79], v[186:187], v[168:169], v[76:79]
v_wmma_f32_16x16x16_f16 v[80:83], v[186:187], v[170:171], v[80:83]
s_setprio 2
ds_load_b64 v[181:182], v5 offset:34304
ds_store_b64 v3, v[240:241] offset:17792
ds_load_b64 v[186:187], v5 offset:34312
ds_store_b64 v3, v[242:243] offset:18560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v210, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v214, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v208, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v212, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v209, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v213, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[196:197], v5 offset:34320
ds_store_b64 v3, v[244:245] offset:18048
ds_store_b64 v3, v[246:247] offset:18304
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v210, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v214, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v208, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v212, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v209, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v213, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_branch 5
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v248, v181, 1.0, v196 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v181, 1.0, v196 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v182, 1.0, v197 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v182, 1.0, v197 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v217, -1.0, v218 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v221, -1.0, v222 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[59:62], v[183:184], v[32:33], v[59:62]
v_wmma_f32_16x16x16_f16 v[63:66], v[183:184], v[166:167], v[63:66]
v_fma_mix_f32 v248, v186, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v186, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v187, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v187, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v242, v216, -1.0, v219 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v220, -1.0, v223 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[158:161], v[198:199], v[177:178], v[158:161]
v_wmma_f32_16x16x16_f16 v[162:165], v[198:199], v[179:180], v[162:165]
v_fma_mixlo_f16 v191, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v191, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v192, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v192, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_pk_fma_f16 v244, v216, 1.0, v217 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v220, 1.0, v221 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[125:128], v[193:194], v[173:174], v[125:128]
v_wmma_f32_16x16x16_f16 v[129:132], v[193:194], v[175:176], v[129:132]
v_fma_mixlo_f16 v186, v248, 0.5, v186 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v250, 0.5, v186 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v187, v249, 0.5, v187 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v187, v251, 0.5, v187 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v246, v216, -1.0, v217 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v220, -1.0, v221 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[92:95], v[188:189], v[168:169], v[92:95]
v_wmma_f32_16x16x16_f16 v[96:99], v[188:189], v[170:171], v[96:99]
s_setprio 2
ds_load_b64 v[32:33], v4 offset:17792
ds_load_b64 v[166:167], v4 offset:17920
ds_store_b64 v3, v[240:241] offset:26048
ds_load_b64 v[168:169], v4 offset:21952
ds_load_b64 v[170:171], v4 offset:22080
ds_store_b64 v3, v[242:243] offset:26816
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v218, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v222, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v216, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v220, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v217, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v221, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[173:174], v4 offset:17800
ds_load_b64 v[175:176], v4 offset:17928
ds_store_b64 v3, v[244:245] offset:26304
ds_load_b64 v[177:178], v4 offset:21960
ds_load_b64 v[179:180], v4 offset:22088
ds_store_b64 v3, v[246:247] offset:26560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v218, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v222, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v216, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v220, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v217, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v221, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_sub_u32 s86, s86, 1
s_cbranch_scc0 10
s_setprio 1
s_mov_b32 s87, 0x17b0
s_branch 64718
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
s_barrier
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v225, 1.0, v227 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v225, 1.0, v227 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v229, 1.0, v231 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v229, 1.0, v231 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:38], v[181:182], v[32:33], v[35:38]
v_wmma_f32_16x16x16_f16 v[39:42], v[181:182], v[166:167], v[39:42]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v226, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v226, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v230, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v230, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[134:137], v[196:197], v[177:178], v[134:137]
v_wmma_f32_16x16x16_f16 v[138:141], v[196:197], v[179:180], v[138:141]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[101:104], v[191:192], v[173:174], v[101:104]
v_wmma_f32_16x16x16_f16 v[105:108], v[191:192], v[175:176], v[105:108]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v226 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v226 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v230 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v230 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[68:71], v[186:187], v[168:169], v[68:71]
v_wmma_f32_16x16x16_f16 v[72:75], v[186:187], v[170:171], v[72:75]
v_mov_b32 v240, v225
v_mov_b32 v242, v227
v_mov_b32 v241, v229
v_mov_b32 v243, v231
s_setprio 2
ds_load_b64 v[183:184], v5 offset:40576
ds_store_b64 v6, v[240:241] offset:34304
ds_load_b64 v[188:189], v5 offset:40584
ds_store_b64 v6, v[242:243] offset:35456
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v225, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v229, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v224, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v228, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v226, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v230, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v227, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v231, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[198:199], v5 offset:40592
ds_store_b64 v6, v[244:245] offset:34688
ds_store_b64 v6, v[246:247] offset:35072
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v225, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v229, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v224, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v228, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v226, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v230, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v227, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v231, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_branch 1
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v248, v183, 1.0, v198 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v183, 1.0, v198 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v184, 1.0, v199 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v184, 1.0, v199 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v233, 1.0, v235 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v233, 1.0, v235 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v237, 1.0, v239 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v237, 1.0, v239 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:54], v[183:184], v[32:33], v[51:54]
v_wmma_f32_16x16x16_f16 v[55:58], v[183:184], v[166:167], v[55:58]
v_fma_mix_f32 v248, v188, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v188, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v189, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v189, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v240, v234, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v234, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v238, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v238, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[150:153], v[198:199], v[177:178], v[150:153]
v_wmma_f32_16x16x16_f16 v[154:157], v[198:199], v[179:180], v[154:157]
v_fma_mixlo_f16 v193, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v193, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v194, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v194, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[117:120], v[193:194], v[173:174], v[117:120]
v_wmma_f32_16x16x16_f16 v[121:124], v[193:194], v[175:176], v[121:124]
v_fma_mixlo_f16 v188, v248, 0.5, v188 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v188, v250, 0.5, v188 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v189, v249, 0.5, v189 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v189, v251, 0.5, v189 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v234 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v234 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v238 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v238 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[84:87], v[188:189], v[168:169], v[84:87]
v_wmma_f32_16x16x16_f16 v[88:91], v[188:189], v[170:171], v[88:91]
v_mov_b32 v240, v233
v_mov_b32 v242, v235
v_mov_b32 v241, v237
v_mov_b32 v243, v239
s_setprio 2
ds_load_b64 v[32:33], v4 offset:26048
ds_load_b64 v[166:167], v4 offset:26176
ds_store_b64 v6, v[240:241] offset:40576
ds_load_b64 v[168:169], v4 offset:30208
ds_load_b64 v[170:171], v4 offset:30336
ds_store_b64 v6, v[242:243] offset:41728
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v233, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v237, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v232, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v236, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v234, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v238, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v235, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v239, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[173:174], v4 offset:26056
ds_load_b64 v[175:176], v4 offset:26184
ds_store_b64 v6, v[244:245] offset:40960
ds_load_b64 v[177:178], v4 offset:30216
ds_load_b64 v[179:180], v4 offset:30344
ds_store_b64 v6, v[246:247] offset:41344
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v233, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v237, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v232, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v236, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v234, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v238, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v235, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v239, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_bitcmp1_b32 s22, 30
s_cbranch_scc0 6
s_setprio 1
s_mov_b32 s87, 0x1c70
s_branch 64634
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
s_barrier
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v67, -1.0, v133 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v203, -1.0, v201 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:46], v[181:182], v[32:33], v[43:46]
v_wmma_f32_16x16x16_f16 v[47:50], v[181:182], v[166:167], v[47:50]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v242, v34, -1.0, v100 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v202, -1.0, v200 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[142:145], v[196:197], v[177:178], v[142:145]
v_wmma_f32_16x16x16_f16 v[146:149], v[196:197], v[179:180], v[146:149]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v244, v34, 1.0, v67 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v202, 1.0, v203 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[109:112], v[191:192], v[173:174], v[109:112]
v_wmma_f32_16x16x16_f16 v[113:116], v[191:192], v[175:176], v[113:116]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v246, v34, -1.0, v67 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v202, -1.0, v203 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[76:79], v[186:187], v[168:169], v[76:79]
v_wmma_f32_16x16x16_f16 v[80:83], v[186:187], v[170:171], v[80:83]
s_setprio 2
ds_load_b64 v[181:182], v5 offset:34304
ds_store_b64 v3, v[240:241] offset:17792
ds_load_b64 v[186:187], v5 offset:34312
ds_store_b64 v3, v[242:243] offset:18560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v133, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v201, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v34, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v202, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v67, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v203, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[196:197], v5 offset:34320
ds_store_b64 v3, v[244:245] offset:18048
ds_store_b64 v3, v[246:247] offset:18304
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v133, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v201, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v34, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v202, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v67, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v203, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_branch 5
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v248, v181, 1.0, v196 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v181, 1.0, v196 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v182, 1.0, v197 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v182, 1.0, v197 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v195, -1.0, v185 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v207, -1.0, v205 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[59:62], v[183:184], v[32:33], v[59:62]
v_wmma_f32_16x16x16_f16 v[63:66], v[183:184], v[166:167], v[63:66]
v_fma_mix_f32 v248, v186, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v186, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v187, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v187, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v242, v190, -1.0, v172 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v206, -1.0, v204 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[158:161], v[198:199], v[177:178], v[158:161]
v_wmma_f32_16x16x16_f16 v[162:165], v[198:199], v[179:180], v[162:165]
v_fma_mixlo_f16 v191, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v191, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v192, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v192, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_pk_fma_f16 v244, v190, 1.0, v195 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v206, 1.0, v207 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[125:128], v[193:194], v[173:174], v[125:128]
v_wmma_f32_16x16x16_f16 v[129:132], v[193:194], v[175:176], v[129:132]
v_fma_mixlo_f16 v186, v248, 0.5, v186 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v250, 0.5, v186 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v187, v249, 0.5, v187 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v187, v251, 0.5, v187 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v246, v190, -1.0, v195 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v206, -1.0, v207 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[92:95], v[188:189], v[168:169], v[92:95]
v_wmma_f32_16x16x16_f16 v[96:99], v[188:189], v[170:171], v[96:99]
s_setprio 2
ds_load_b64 v[32:33], v4 offset:17792
ds_load_b64 v[166:167], v4 offset:17920
ds_store_b64 v3, v[240:241] offset:26048
ds_load_b64 v[168:169], v4 offset:21952
ds_load_b64 v[170:171], v4 offset:22080
ds_store_b64 v3, v[242:243] offset:26816
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v185, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v205, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v190, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v206, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v195, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v207, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[173:174], v4 offset:17800
ds_load_b64 v[175:176], v4 offset:17928
ds_store_b64 v3, v[244:245] offset:26304
ds_load_b64 v[177:178], v4 offset:21960
ds_load_b64 v[179:180], v4 offset:22088
ds_store_b64 v3, v[246:247] offset:26560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v185, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v205, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v190, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v206, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v195, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v207, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_sub_u32 s86, s86, 1
s_cbranch_scc0 10
s_setprio 1
s_mov_b32 s87, 0x2070
s_branch 64158
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
s_barrier
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v210, 1.0, v209 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v210, 1.0, v209 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v218, 1.0, v217 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v218, 1.0, v217 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:38], v[181:182], v[32:33], v[35:38]
v_wmma_f32_16x16x16_f16 v[39:42], v[181:182], v[166:167], v[39:42]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v208, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v208, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v216, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v216, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[134:137], v[196:197], v[177:178], v[134:137]
v_wmma_f32_16x16x16_f16 v[138:141], v[196:197], v[179:180], v[138:141]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[101:104], v[191:192], v[173:174], v[101:104]
v_wmma_f32_16x16x16_f16 v[105:108], v[191:192], v[175:176], v[105:108]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v208 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v208 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v216 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v216 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[68:71], v[186:187], v[168:169], v[68:71]
v_wmma_f32_16x16x16_f16 v[72:75], v[186:187], v[170:171], v[72:75]
v_mov_b32 v240, v210
v_mov_b32 v242, v209
v_mov_b32 v241, v218
v_mov_b32 v243, v217
s_setprio 2
ds_load_b64 v[183:184], v5 offset:40576
ds_store_b64 v6, v[240:241] offset:34304
ds_load_b64 v[188:189], v5 offset:40584
ds_store_b64 v6, v[242:243] offset:35456
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v210, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v218, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v211, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v219, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v208, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v216, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v209, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v217, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[198:199], v5 offset:40592
ds_store_b64 v6, v[244:245] offset:34688
ds_store_b64 v6, v[246:247] offset:35072
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v210, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v218, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v211, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v219, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v208, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v216, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v209, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v217, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_branch 1
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v248, v183, 1.0, v198 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v183, 1.0, v198 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v184, 1.0, v199 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v184, 1.0, v199 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v214, 1.0, v213 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v214, 1.0, v213 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v222, 1.0, v221 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v222, 1.0, v221 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:54], v[183:184], v[32:33], v[51:54]
v_wmma_f32_16x16x16_f16 v[55:58], v[183:184], v[166:167], v[55:58]
v_fma_mix_f32 v248, v188, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v188, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v189, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v189, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v240, v212, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v212, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v220, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v220, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[150:153], v[198:199], v[177:178], v[150:153]
v_wmma_f32_16x16x16_f16 v[154:157], v[198:199], v[179:180], v[154:157]
v_fma_mixlo_f16 v193, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v193, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v194, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v194, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[117:120], v[193:194], v[173:174], v[117:120]
v_wmma_f32_16x16x16_f16 v[121:124], v[193:194], v[175:176], v[121:124]
v_fma_mixlo_f16 v188, v248, 0.5, v188 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v188, v250, 0.5, v188 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v189, v249, 0.5, v189 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v189, v251, 0.5, v189 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v212 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v212 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v220 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v220 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[84:87], v[188:189], v[168:169], v[84:87]
v_wmma_f32_16x16x16_f16 v[88:91], v[188:189], v[170:171], v[88:91]
v_mov_b32 v240, v214
v_mov_b32 v242, v213
v_mov_b32 v241, v222
v_mov_b32 v243, v221
s_setprio 2
ds_load_b64 v[32:33], v4 offset:26048
ds_load_b64 v[166:167], v4 offset:26176
ds_store_b64 v6, v[240:241] offset:40576
ds_load_b64 v[168:169], v4 offset:30208
ds_load_b64 v[170:171], v4 offset:30336
ds_store_b64 v6, v[242:243] offset:41728
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v214, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v222, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v215, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v223, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v212, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v220, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v213, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v221, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[173:174], v4 offset:26056
ds_load_b64 v[175:176], v4 offset:26184
ds_store_b64 v6, v[244:245] offset:40960
ds_load_b64 v[177:178], v4 offset:30216
ds_load_b64 v[179:180], v4 offset:30344
ds_store_b64 v6, v[246:247] offset:41344
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v214, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v222, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v215, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v223, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v212, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v220, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v213, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v221, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_bitcmp1_b32 s22, 30
s_cbranch_scc0 6
s_setprio 1
s_mov_b32 s87, 0x2530
s_branch 64074
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
s_barrier
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v227, -1.0, v225 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v235, -1.0, v233 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:46], v[181:182], v[32:33], v[43:46]
v_wmma_f32_16x16x16_f16 v[47:50], v[181:182], v[166:167], v[47:50]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v242, v226, -1.0, v224 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v234, -1.0, v232 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[142:145], v[196:197], v[177:178], v[142:145]
v_wmma_f32_16x16x16_f16 v[146:149], v[196:197], v[179:180], v[146:149]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v244, v226, 1.0, v227 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v234, 1.0, v235 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[109:112], v[191:192], v[173:174], v[109:112]
v_wmma_f32_16x16x16_f16 v[113:116], v[191:192], v[175:176], v[113:116]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v246, v226, -1.0, v227 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v234, -1.0, v235 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[76:79], v[186:187], v[168:169], v[76:79]
v_wmma_f32_16x16x16_f16 v[80:83], v[186:187], v[170:171], v[80:83]
s_setprio 2
ds_load_b64 v[181:182], v5 offset:34304
ds_store_b64 v3, v[240:241] offset:17792
ds_load_b64 v[186:187], v5 offset:34312
ds_store_b64 v3, v[242:243] offset:18560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v225, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v233, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v226, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v234, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v227, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v235, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[196:197], v5 offset:34320
ds_store_b64 v3, v[244:245] offset:18048
ds_store_b64 v3, v[246:247] offset:18304
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v225, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v233, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v226, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v234, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v227, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v235, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_branch 5
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v248, v181, 1.0, v196 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v181, 1.0, v196 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v182, 1.0, v197 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v182, 1.0, v197 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v231, -1.0, v229 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v239, -1.0, v237 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[59:62], v[183:184], v[32:33], v[59:62]
v_wmma_f32_16x16x16_f16 v[63:66], v[183:184], v[166:167], v[63:66]
v_fma_mix_f32 v248, v186, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v186, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v187, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v187, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v242, v230, -1.0, v228 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v238, -1.0, v236 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[158:161], v[198:199], v[177:178], v[158:161]
v_wmma_f32_16x16x16_f16 v[162:165], v[198:199], v[179:180], v[162:165]
v_fma_mixlo_f16 v191, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v191, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v192, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v192, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_pk_fma_f16 v244, v230, 1.0, v231 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v238, 1.0, v239 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[125:128], v[193:194], v[173:174], v[125:128]
v_wmma_f32_16x16x16_f16 v[129:132], v[193:194], v[175:176], v[129:132]
v_fma_mixlo_f16 v186, v248, 0.5, v186 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v250, 0.5, v186 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v187, v249, 0.5, v187 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v187, v251, 0.5, v187 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v246, v230, -1.0, v231 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v238, -1.0, v239 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[92:95], v[188:189], v[168:169], v[92:95]
v_wmma_f32_16x16x16_f16 v[96:99], v[188:189], v[170:171], v[96:99]
s_setprio 2
ds_load_b64 v[32:33], v4 offset:17792
ds_load_b64 v[166:167], v4 offset:17920
ds_store_b64 v3, v[240:241] offset:26048
ds_load_b64 v[168:169], v4 offset:21952
ds_load_b64 v[170:171], v4 offset:22080
ds_store_b64 v3, v[242:243] offset:26816
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v229, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v237, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v230, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v238, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v231, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v239, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[173:174], v4 offset:17800
ds_load_b64 v[175:176], v4 offset:17928
ds_store_b64 v3, v[244:245] offset:26304
ds_load_b64 v[177:178], v4 offset:21960
ds_load_b64 v[179:180], v4 offset:22088
ds_store_b64 v3, v[246:247] offset:26560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v229, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v237, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v230, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v238, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v231, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v239, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_sub_u32 s86, s86, 1
s_cbranch_scc0 63866
s_setprio 1
s_mov_b32 s87, 0xef0
s_branch 63598
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v133, 1.0, v67 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v133, 1.0, v67 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v185, 1.0, v195 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v185, 1.0, v195 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:38], v[181:182], v[32:33], v[35:38]
v_wmma_f32_16x16x16_f16 v[39:42], v[181:182], v[166:167], v[39:42]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v34, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v34, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v190, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v190, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[134:137], v[196:197], v[177:178], v[134:137]
v_wmma_f32_16x16x16_f16 v[138:141], v[196:197], v[179:180], v[138:141]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[101:104], v[191:192], v[173:174], v[101:104]
v_wmma_f32_16x16x16_f16 v[105:108], v[191:192], v[175:176], v[105:108]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v34 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v34 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v190 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v190 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[68:71], v[186:187], v[168:169], v[68:71]
v_wmma_f32_16x16x16_f16 v[72:75], v[186:187], v[170:171], v[72:75]
v_mov_b32 v240, v133
v_mov_b32 v242, v67
v_mov_b32 v241, v185
v_mov_b32 v243, v195
s_barrier
s_setprio 2
ds_load_b64 v[183:184], v5 offset:40576
ds_store_b64 v6, v[240:241] offset:34304
ds_load_b64 v[188:189], v5 offset:40584
ds_store_b64 v6, v[242:243] offset:35456
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v133, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v185, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v100, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v172, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v34, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v190, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v67, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v195, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[198:199], v5 offset:40592
ds_store_b64 v6, v[244:245] offset:34688
ds_store_b64 v6, v[246:247] offset:35072
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v133, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v185, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v100, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v172, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v34, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v190, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v67, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v195, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_branch 1
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
v_fma_mix_f32 v248, v183, 1.0, v198 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v183, 1.0, v198 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v184, 1.0, v199 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v184, 1.0, v199 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v201, 1.0, v203 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v201, 1.0, v203 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v205, 1.0, v207 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v205, 1.0, v207 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:54], v[183:184], v[32:33], v[51:54]
v_wmma_f32_16x16x16_f16 v[55:58], v[183:184], v[166:167], v[55:58]
v_fma_mix_f32 v248, v188, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v188, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v189, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v189, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v240, v202, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v202, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v206, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v206, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[150:153], v[198:199], v[177:178], v[150:153]
v_wmma_f32_16x16x16_f16 v[154:157], v[198:199], v[179:180], v[154:157]
v_fma_mixlo_f16 v193, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v193, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v194, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v194, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[117:120], v[193:194], v[173:174], v[117:120]
v_wmma_f32_16x16x16_f16 v[121:124], v[193:194], v[175:176], v[121:124]
v_fma_mixlo_f16 v188, v248, 0.5, v188 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v188, v250, 0.5, v188 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v189, v249, 0.5, v189 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v189, v251, 0.5, v189 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v202 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v202 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v206 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v206 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[84:87], v[188:189], v[168:169], v[84:87]
v_wmma_f32_16x16x16_f16 v[88:91], v[188:189], v[170:171], v[88:91]
v_mov_b32 v240, v201
v_mov_b32 v242, v203
v_mov_b32 v241, v205
v_mov_b32 v243, v207
s_barrier
s_setprio 2
ds_load_b64 v[32:33], v4 offset:26048
ds_load_b64 v[166:167], v4 offset:26176
ds_store_b64 v6, v[240:241] offset:40576
ds_load_b64 v[168:169], v4 offset:30208
ds_load_b64 v[170:171], v4 offset:30336
ds_store_b64 v6, v[242:243] offset:41728
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v201, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v205, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v200, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v204, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v202, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v206, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v203, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v207, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[173:174], v4 offset:26056
ds_load_b64 v[175:176], v4 offset:26184
ds_store_b64 v6, v[244:245] offset:40960
ds_load_b64 v[177:178], v4 offset:30216
ds_load_b64 v[179:180], v4 offset:30344
ds_store_b64 v6, v[246:247] offset:41344
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v201, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v205, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v200, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v204, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v202, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v206, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v203, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v207, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_bitcmp1_b32 s22, 30
s_cbranch_scc0 6
s_setprio 2
s_mov_b32 s87, 0x2df0
s_branch 63514
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v209, -1.0, v210 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v213, -1.0, v214 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:46], v[181:182], v[32:33], v[43:46]
v_wmma_f32_16x16x16_f16 v[47:50], v[181:182], v[166:167], v[47:50]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v242, v208, -1.0, v211 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v212, -1.0, v215 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[142:145], v[196:197], v[177:178], v[142:145]
v_wmma_f32_16x16x16_f16 v[146:149], v[196:197], v[179:180], v[146:149]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v244, v208, 1.0, v209 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v212, 1.0, v213 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[109:112], v[191:192], v[173:174], v[109:112]
v_wmma_f32_16x16x16_f16 v[113:116], v[191:192], v[175:176], v[113:116]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v246, v208, -1.0, v209 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v212, -1.0, v213 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[76:79], v[186:187], v[168:169], v[76:79]
v_wmma_f32_16x16x16_f16 v[80:83], v[186:187], v[170:171], v[80:83]
s_barrier
s_setprio 2
ds_load_b64 v[181:182], v5 offset:34304
ds_store_b64 v3, v[240:241] offset:17792
ds_load_b64 v[186:187], v5 offset:34312
ds_store_b64 v3, v[242:243] offset:18560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v210, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v214, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v208, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v212, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v209, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v213, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[196:197], v5 offset:34320
ds_store_b64 v3, v[244:245] offset:18048
ds_store_b64 v3, v[246:247] offset:18304
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v210, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v214, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v208, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v212, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v209, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v213, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_branch 5
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
v_fma_mix_f32 v248, v181, 1.0, v196 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v181, 1.0, v196 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v182, 1.0, v197 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v182, 1.0, v197 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v217, -1.0, v218 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v221, -1.0, v222 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[59:62], v[183:184], v[32:33], v[59:62]
v_wmma_f32_16x16x16_f16 v[63:66], v[183:184], v[166:167], v[63:66]
v_fma_mix_f32 v248, v186, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v186, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v187, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v187, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v242, v216, -1.0, v219 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v220, -1.0, v223 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[158:161], v[198:199], v[177:178], v[158:161]
v_wmma_f32_16x16x16_f16 v[162:165], v[198:199], v[179:180], v[162:165]
v_fma_mixlo_f16 v191, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v191, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v192, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v192, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_pk_fma_f16 v244, v216, 1.0, v217 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v220, 1.0, v221 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[125:128], v[193:194], v[173:174], v[125:128]
v_wmma_f32_16x16x16_f16 v[129:132], v[193:194], v[175:176], v[129:132]
v_fma_mixlo_f16 v186, v248, 0.5, v186 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v250, 0.5, v186 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v187, v249, 0.5, v187 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v187, v251, 0.5, v187 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v246, v216, -1.0, v217 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v220, -1.0, v221 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[92:95], v[188:189], v[168:169], v[92:95]
v_wmma_f32_16x16x16_f16 v[96:99], v[188:189], v[170:171], v[96:99]
s_barrier
s_setprio 2
ds_load_b64 v[32:33], v4 offset:17792
ds_load_b64 v[166:167], v4 offset:17920
ds_store_b64 v3, v[240:241] offset:26048
ds_load_b64 v[168:169], v4 offset:21952
ds_load_b64 v[170:171], v4 offset:22080
ds_store_b64 v3, v[242:243] offset:26816
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v218, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v222, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v216, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v220, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v217, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v221, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[173:174], v4 offset:17800
ds_load_b64 v[175:176], v4 offset:17928
ds_store_b64 v3, v[244:245] offset:26304
ds_load_b64 v[177:178], v4 offset:21960
ds_load_b64 v[179:180], v4 offset:22088
ds_store_b64 v3, v[246:247] offset:26560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v218, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v222, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v216, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v220, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v217, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v221, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_sub_u32 s86, s86, 1
s_cbranch_scc0 10
s_setprio 2
s_mov_b32 s87, 0x31f0
s_branch 63038
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v225, 1.0, v227 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v225, 1.0, v227 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v229, 1.0, v231 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v229, 1.0, v231 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:38], v[181:182], v[32:33], v[35:38]
v_wmma_f32_16x16x16_f16 v[39:42], v[181:182], v[166:167], v[39:42]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v226, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v226, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v230, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v230, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[134:137], v[196:197], v[177:178], v[134:137]
v_wmma_f32_16x16x16_f16 v[138:141], v[196:197], v[179:180], v[138:141]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[101:104], v[191:192], v[173:174], v[101:104]
v_wmma_f32_16x16x16_f16 v[105:108], v[191:192], v[175:176], v[105:108]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v226 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v226 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v230 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v230 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[68:71], v[186:187], v[168:169], v[68:71]
v_wmma_f32_16x16x16_f16 v[72:75], v[186:187], v[170:171], v[72:75]
v_mov_b32 v240, v225
v_mov_b32 v242, v227
v_mov_b32 v241, v229
v_mov_b32 v243, v231
s_barrier
s_setprio 2
ds_load_b64 v[183:184], v5 offset:40576
ds_store_b64 v6, v[240:241] offset:34304
ds_load_b64 v[188:189], v5 offset:40584
ds_store_b64 v6, v[242:243] offset:35456
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v225, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v229, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v224, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v228, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v226, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v230, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v227, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v231, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[198:199], v5 offset:40592
ds_store_b64 v6, v[244:245] offset:34688
ds_store_b64 v6, v[246:247] offset:35072
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v225, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v229, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v224, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v228, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v226, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v230, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v227, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v231, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_branch 1
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
v_fma_mix_f32 v248, v183, 1.0, v198 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v183, 1.0, v198 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v184, 1.0, v199 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v184, 1.0, v199 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v233, 1.0, v235 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v233, 1.0, v235 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v237, 1.0, v239 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v237, 1.0, v239 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:54], v[183:184], v[32:33], v[51:54]
v_wmma_f32_16x16x16_f16 v[55:58], v[183:184], v[166:167], v[55:58]
v_fma_mix_f32 v248, v188, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v188, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v189, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v189, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v240, v234, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v234, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v238, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v238, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[150:153], v[198:199], v[177:178], v[150:153]
v_wmma_f32_16x16x16_f16 v[154:157], v[198:199], v[179:180], v[154:157]
v_fma_mixlo_f16 v193, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v193, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v194, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v194, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[117:120], v[193:194], v[173:174], v[117:120]
v_wmma_f32_16x16x16_f16 v[121:124], v[193:194], v[175:176], v[121:124]
v_fma_mixlo_f16 v188, v248, 0.5, v188 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v188, v250, 0.5, v188 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v189, v249, 0.5, v189 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v189, v251, 0.5, v189 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v234 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v234 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v238 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v238 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[84:87], v[188:189], v[168:169], v[84:87]
v_wmma_f32_16x16x16_f16 v[88:91], v[188:189], v[170:171], v[88:91]
v_mov_b32 v240, v233
v_mov_b32 v242, v235
v_mov_b32 v241, v237
v_mov_b32 v243, v239
s_barrier
s_setprio 2
ds_load_b64 v[32:33], v4 offset:26048
ds_load_b64 v[166:167], v4 offset:26176
ds_store_b64 v6, v[240:241] offset:40576
ds_load_b64 v[168:169], v4 offset:30208
ds_load_b64 v[170:171], v4 offset:30336
ds_store_b64 v6, v[242:243] offset:41728
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v233, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v237, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v232, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v236, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v234, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v238, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v235, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v239, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[173:174], v4 offset:26056
ds_load_b64 v[175:176], v4 offset:26184
ds_store_b64 v6, v[244:245] offset:40960
ds_load_b64 v[177:178], v4 offset:30216
ds_load_b64 v[179:180], v4 offset:30344
ds_store_b64 v6, v[246:247] offset:41344
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v233, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v237, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v232, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v236, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v234, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v238, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v235, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v239, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_bitcmp1_b32 s22, 30
s_cbranch_scc0 6
s_setprio 2
s_mov_b32 s87, 0x36b0
s_branch 62954
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v67, -1.0, v133 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v203, -1.0, v201 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:46], v[181:182], v[32:33], v[43:46]
v_wmma_f32_16x16x16_f16 v[47:50], v[181:182], v[166:167], v[47:50]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v242, v34, -1.0, v100 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v202, -1.0, v200 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[142:145], v[196:197], v[177:178], v[142:145]
v_wmma_f32_16x16x16_f16 v[146:149], v[196:197], v[179:180], v[146:149]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v244, v34, 1.0, v67 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v202, 1.0, v203 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[109:112], v[191:192], v[173:174], v[109:112]
v_wmma_f32_16x16x16_f16 v[113:116], v[191:192], v[175:176], v[113:116]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v246, v34, -1.0, v67 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v202, -1.0, v203 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[76:79], v[186:187], v[168:169], v[76:79]
v_wmma_f32_16x16x16_f16 v[80:83], v[186:187], v[170:171], v[80:83]
s_barrier
s_setprio 2
ds_load_b64 v[181:182], v5 offset:34304
ds_store_b64 v3, v[240:241] offset:17792
ds_load_b64 v[186:187], v5 offset:34312
ds_store_b64 v3, v[242:243] offset:18560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v133, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v201, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v34, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v202, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v67, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v203, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[196:197], v5 offset:34320
ds_store_b64 v3, v[244:245] offset:18048
ds_store_b64 v3, v[246:247] offset:18304
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v133, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v201, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v34, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v202, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v67, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v203, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_branch 5
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
v_fma_mix_f32 v248, v181, 1.0, v196 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v181, 1.0, v196 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v182, 1.0, v197 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v182, 1.0, v197 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v195, -1.0, v185 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v207, -1.0, v205 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[59:62], v[183:184], v[32:33], v[59:62]
v_wmma_f32_16x16x16_f16 v[63:66], v[183:184], v[166:167], v[63:66]
v_fma_mix_f32 v248, v186, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v186, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v187, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v187, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v242, v190, -1.0, v172 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v206, -1.0, v204 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[158:161], v[198:199], v[177:178], v[158:161]
v_wmma_f32_16x16x16_f16 v[162:165], v[198:199], v[179:180], v[162:165]
v_fma_mixlo_f16 v191, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v191, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v192, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v192, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_pk_fma_f16 v244, v190, 1.0, v195 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v206, 1.0, v207 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[125:128], v[193:194], v[173:174], v[125:128]
v_wmma_f32_16x16x16_f16 v[129:132], v[193:194], v[175:176], v[129:132]
v_fma_mixlo_f16 v186, v248, 0.5, v186 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v250, 0.5, v186 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v187, v249, 0.5, v187 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v187, v251, 0.5, v187 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v246, v190, -1.0, v195 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v206, -1.0, v207 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[92:95], v[188:189], v[168:169], v[92:95]
v_wmma_f32_16x16x16_f16 v[96:99], v[188:189], v[170:171], v[96:99]
s_barrier
s_setprio 2
ds_load_b64 v[32:33], v4 offset:17792
ds_load_b64 v[166:167], v4 offset:17920
ds_store_b64 v3, v[240:241] offset:26048
ds_load_b64 v[168:169], v4 offset:21952
ds_load_b64 v[170:171], v4 offset:22080
ds_store_b64 v3, v[242:243] offset:26816
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v185, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v205, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v190, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v206, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v195, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v207, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[173:174], v4 offset:17800
ds_load_b64 v[175:176], v4 offset:17928
ds_store_b64 v3, v[244:245] offset:26304
ds_load_b64 v[177:178], v4 offset:21960
ds_load_b64 v[179:180], v4 offset:22088
ds_store_b64 v3, v[246:247] offset:26560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v185, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v205, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v190, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v206, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v195, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v207, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_sub_u32 s86, s86, 1
s_cbranch_scc0 10
s_setprio 2
s_mov_b32 s87, 0x3ab0
s_branch 62478
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v210, 1.0, v209 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v210, 1.0, v209 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v218, 1.0, v217 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v218, 1.0, v217 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:38], v[181:182], v[32:33], v[35:38]
v_wmma_f32_16x16x16_f16 v[39:42], v[181:182], v[166:167], v[39:42]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v208, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v208, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v216, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v216, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[134:137], v[196:197], v[177:178], v[134:137]
v_wmma_f32_16x16x16_f16 v[138:141], v[196:197], v[179:180], v[138:141]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[101:104], v[191:192], v[173:174], v[101:104]
v_wmma_f32_16x16x16_f16 v[105:108], v[191:192], v[175:176], v[105:108]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v208 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v208 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v216 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v216 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[68:71], v[186:187], v[168:169], v[68:71]
v_wmma_f32_16x16x16_f16 v[72:75], v[186:187], v[170:171], v[72:75]
v_mov_b32 v240, v210
v_mov_b32 v242, v209
v_mov_b32 v241, v218
v_mov_b32 v243, v217
s_barrier
s_setprio 2
ds_load_b64 v[183:184], v5 offset:40576
ds_store_b64 v6, v[240:241] offset:34304
ds_load_b64 v[188:189], v5 offset:40584
ds_store_b64 v6, v[242:243] offset:35456
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v210, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v218, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v211, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v219, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v208, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v216, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v209, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v217, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[198:199], v5 offset:40592
ds_store_b64 v6, v[244:245] offset:34688
ds_store_b64 v6, v[246:247] offset:35072
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v210, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v218, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v211, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v219, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v208, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v216, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v209, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v217, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_branch 1
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
v_fma_mix_f32 v248, v183, 1.0, v198 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v183, 1.0, v198 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v184, 1.0, v199 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v184, 1.0, v199 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v240, v214, 1.0, v213 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v214, 1.0, v213 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v241, v222, 1.0, v221 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v222, 1.0, v221 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:54], v[183:184], v[32:33], v[51:54]
v_wmma_f32_16x16x16_f16 v[55:58], v[183:184], v[166:167], v[55:58]
v_fma_mix_f32 v248, v188, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v188, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v189, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v189, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v240, v212, -1.0, v240 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v212, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v241, v220, -1.0, v241 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v220, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[150:153], v[198:199], v[177:178], v[150:153]
v_wmma_f32_16x16x16_f16 v[154:157], v[198:199], v[179:180], v[154:157]
v_fma_mixlo_f16 v193, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v193, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v194, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v194, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v240, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v242, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v241, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v243, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[117:120], v[193:194], v[173:174], v[117:120]
v_wmma_f32_16x16x16_f16 v[121:124], v[193:194], v[175:176], v[121:124]
v_fma_mixlo_f16 v188, v248, 0.5, v188 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v188, v250, 0.5, v188 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v189, v249, 0.5, v189 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v189, v251, 0.5, v189 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v244, v240, 0.5, v212 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v242, 0.5, v212 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v241, 0.5, v220 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v243, 0.5, v220 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[84:87], v[188:189], v[168:169], v[84:87]
v_wmma_f32_16x16x16_f16 v[88:91], v[188:189], v[170:171], v[88:91]
v_mov_b32 v240, v214
v_mov_b32 v242, v213
v_mov_b32 v241, v222
v_mov_b32 v243, v221
s_barrier
s_setprio 2
ds_load_b64 v[32:33], v4 offset:26048
ds_load_b64 v[166:167], v4 offset:26176
ds_store_b64 v6, v[240:241] offset:40576
ds_load_b64 v[168:169], v4 offset:30208
ds_load_b64 v[170:171], v4 offset:30336
ds_store_b64 v6, v[242:243] offset:41728
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_b16 v214, v12, s[72:75], 0 idxen
buffer_load_d16_b16 v222, v16, s[72:75], 0 idxen
buffer_load_d16_b16 v215, v15, s[72:75], 0 idxen
buffer_load_d16_b16 v223, v19, s[72:75], 0 idxen
buffer_load_d16_b16 v212, v13, s[72:75], 0 idxen
buffer_load_d16_b16 v220, v17, s[72:75], 0 idxen
buffer_load_d16_b16 v213, v14, s[72:75], 0 idxen
buffer_load_d16_b16 v221, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
ds_load_b64 v[173:174], v4 offset:26056
ds_load_b64 v[175:176], v4 offset:26184
ds_store_b64 v6, v[244:245] offset:40960
ds_load_b64 v[177:178], v4 offset:30216
ds_load_b64 v[179:180], v4 offset:30344
ds_store_b64 v6, v[246:247] offset:41344
s_sub_u32 s70, s70, 4
s_cselect_b32 s74, 0, s74
buffer_load_d16_hi_b16 v214, v12, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v222, v16, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v215, v15, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v223, v19, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v212, v13, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v220, v17, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v213, v14, s[72:75], 0 idxen
buffer_load_d16_hi_b16 v221, v18, s[72:75], 0 idxen
s_add_u32 s72, s72, s80
s_addc_u32 s73, s73, 0
s_bitcmp1_b32 s22, 30
s_cbranch_scc0 6
s_setprio 2
s_mov_b32 s87, 0x3f70
s_branch 62394
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0) vmcnt(56)
v_pk_fma_f16 v32, v173, -1.0, v32 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v33, v174, -1.0, v33 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v166, v175, -1.0, v166 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v176, -1.0, v167 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v227, -1.0, v225 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v235, -1.0, v233 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:46], v[181:182], v[32:33], v[43:46]
v_wmma_f32_16x16x16_f16 v[47:50], v[181:182], v[166:167], v[47:50]
v_pk_fma_f16 v177, v168, -1.0, v177 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v178, v169, -1.0, v178 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v179, v170, -1.0, v179 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v180, v171, -1.0, v180 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v242, v226, -1.0, v224 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v234, -1.0, v232 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[142:145], v[196:197], v[177:178], v[142:145]
v_wmma_f32_16x16x16_f16 v[146:149], v[196:197], v[179:180], v[146:149]
v_pk_fma_f16 v173, v168, -1.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v174, v169, -1.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v175, v170, -1.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v176, v171, -1.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v244, v226, 1.0, v227 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v234, 1.0, v235 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[109:112], v[191:192], v[173:174], v[109:112]
v_wmma_f32_16x16x16_f16 v[113:116], v[191:192], v[175:176], v[113:116]
v_pk_fma_f16 v168, v168, 2.0, v173 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v169, v169, 2.0, v174 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v170, 2.0, v175 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v171, 2.0, v176 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v246, v226, -1.0, v227 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v234, -1.0, v235 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[76:79], v[186:187], v[168:169], v[76:79]
v_wmma_f32_16x16x16_f16 v[80:83], v[186:187], v[170:171], v[80:83]
s_barrier
s_setprio 2
ds_load_b64 v[181:182], v5 offset:34304
ds_store_b64 v3, v[240:241] offset:17792
ds_load_b64 v[186:187], v5 offset:34312
ds_store_b64 v3, v[242:243] offset:18560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v225, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v233, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v226, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v234, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v227, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v235, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[196:197], v5 offset:34320
ds_store_b64 v3, v[244:245] offset:18048
ds_store_b64 v3, v[246:247] offset:18304
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v225, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v233, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v226, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v234, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v227, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v235, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_branch 5
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_waitcnt lgkmcnt(0)
v_fma_mix_f32 v248, v181, 1.0, v196 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v250, v181, 1.0, v196 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v249, v182, 1.0, v197 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v251, v182, 1.0, v197 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v231, -1.0, v229 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v239, -1.0, v237 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[59:62], v[183:184], v[32:33], v[59:62]
v_wmma_f32_16x16x16_f16 v[63:66], v[183:184], v[166:167], v[63:66]
v_fma_mix_f32 v248, v186, -1.0, v248 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v250, v186, -1.0, v250 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v249, v187, -1.0, v249 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v251, v187, -1.0, v251 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v242, v230, -1.0, v228 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v243, v238, -1.0, v236 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[158:161], v[198:199], v[177:178], v[158:161]
v_wmma_f32_16x16x16_f16 v[162:165], v[198:199], v[179:180], v[162:165]
v_fma_mixlo_f16 v191, v248, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v191, v250, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v192, v249, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v192, v251, 0.5, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_pk_fma_f16 v244, v230, 1.0, v231 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v245, v238, 1.0, v239 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[125:128], v[193:194], v[173:174], v[125:128]
v_wmma_f32_16x16x16_f16 v[129:132], v[193:194], v[175:176], v[129:132]
v_fma_mixlo_f16 v186, v248, 0.5, v186 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v250, 0.5, v186 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v187, v249, 0.5, v187 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v187, v251, 0.5, v187 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v246, v230, -1.0, v231 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_pk_fma_f16 v247, v238, -1.0, v239 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[92:95], v[188:189], v[168:169], v[92:95]
v_wmma_f32_16x16x16_f16 v[96:99], v[188:189], v[170:171], v[96:99]
s_barrier
s_setprio 2
ds_load_b64 v[32:33], v4 offset:17792
ds_load_b64 v[166:167], v4 offset:17920
ds_store_b64 v3, v[240:241] offset:26048
ds_load_b64 v[168:169], v4 offset:21952
ds_load_b64 v[170:171], v4 offset:22080
ds_store_b64 v3, v[242:243] offset:26816
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_b16 v229, v20, s[76:79], 0 idxen
buffer_load_d16_b16 v237, v24, s[76:79], 0 idxen
buffer_load_d16_b16 v230, v21, s[76:79], 0 idxen
buffer_load_d16_b16 v238, v25, s[76:79], 0 idxen
buffer_load_d16_b16 v231, v22, s[76:79], 0 idxen
buffer_load_d16_b16 v239, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
ds_load_b64 v[173:174], v4 offset:17800
ds_load_b64 v[175:176], v4 offset:17928
ds_store_b64 v3, v[244:245] offset:26304
ds_load_b64 v[177:178], v4 offset:21960
ds_load_b64 v[179:180], v4 offset:22088
ds_store_b64 v3, v[246:247] offset:26560
s_sub_u32 s71, s71, 4
s_cselect_b32 s78, 0, s78
buffer_load_d16_hi_b16 v229, v20, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v237, v24, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v230, v21, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v238, v25, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v231, v22, s[76:79], 0 idxen
buffer_load_d16_hi_b16 v239, v26, s[76:79], 0 idxen
s_add_u32 s76, s76, s81
s_addc_u32 s77, s77, 0
s_sub_u32 s86, s86, 1
s_cbranch_scc0 63866
s_setprio 2
s_mov_b32 s87, 0x2930
s_branch 61918
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s90, s90, 1
s_cbranch_scc0 8616
v_and_b32 v249, 63, v0
v_lshl_add_u32 v249, v249, 3, s9
v_lshl_add_u32 v249, s5, 9, v249
v_lshl_add_u32 v252, s5, 2, 0
ds_load_b32 v248, v252 offset:61192
ds_load_b64 v[250:251], v249 offset:57088
s_sub_u32 s5, s5, 1
s_cselect_b32 s5, 3, s5
s_bitcmp1_b32 s22, 28
s_cbranch_scc0 8463
s_sub_u32 s96, s20, s84
s_cmov_b32 s96, 0
s_mov_b64 s[92:93], s[28:29]
s_mov_b32 s94, 0x80000000
s_mov_b32 s95, 0x11014000
s_setprio 1
s_add_u32 s98, s6, s91
s_addc_u32 s99, s7, 0
s_setpc_b64 s[98:99]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_e32 v35, v35, v68
v_add_f32_e32 v36, v36, v69
v_add_f32 v134, v134, neg(v101)
v_add_f32 v135, v135, neg(v102)
v_add_f32_e32 v35, v35, v101
v_add_f32_e32 v36, v36, v102
v_add_f32_e32 v134, v134, v68
v_add_f32_e32 v135, v135, v69
v_add_f32_e32 v39, v39, v72
v_add_f32_e32 v40, v40, v73
v_add_f32 v138, v138, neg(v105)
v_add_f32 v139, v139, neg(v106)
v_add_f32_e32 v39, v39, v105
v_add_f32_e32 v40, v40, v106
v_add_f32_e32 v138, v138, v72
v_add_f32_e32 v139, v139, v73
v_add_f32_e32 v43, v43, v76
v_add_f32_e32 v44, v44, v77
v_add_f32 v142, v142, neg(v109)
v_add_f32 v143, v143, neg(v110)
v_add_f32_e32 v43, v43, v109
v_add_f32_e32 v44, v44, v110
v_add_f32_e32 v142, v142, v76
v_add_f32_e32 v143, v143, v77
v_add_f32_e32 v47, v47, v80
v_add_f32_e32 v48, v48, v81
v_add_f32 v146, v146, neg(v113)
v_add_f32 v147, v147, neg(v114)
v_add_f32_e32 v47, v47, v113
v_add_f32_e32 v48, v48, v114
v_add_f32_e32 v146, v146, v80
v_add_f32_e32 v147, v147, v81
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v37, v37, v70
v_add_f32_e32 v38, v38, v71
v_add_f32 v136, v136, neg(v103)
v_add_f32 v137, v137, neg(v104)
v_add_f32_e32 v37, v37, v103
v_add_f32_e32 v38, v38, v104
v_add_f32_e32 v136, v136, v70
v_add_f32_e32 v137, v137, v71
v_add_f32_e32 v41, v41, v74
v_add_f32_e32 v42, v42, v75
v_add_f32 v140, v140, neg(v107)
v_add_f32 v141, v141, neg(v108)
v_add_f32_e32 v41, v41, v107
v_add_f32_e32 v42, v42, v108
v_add_f32_e32 v140, v140, v74
v_add_f32_e32 v141, v141, v75
v_add_f32_e32 v45, v45, v78
v_add_f32_e32 v46, v46, v79
v_add_f32 v144, v144, neg(v111)
v_add_f32 v145, v145, neg(v112)
v_add_f32_e32 v45, v45, v111
v_add_f32_e32 v46, v46, v112
v_add_f32_e32 v144, v144, v78
v_add_f32_e32 v145, v145, v79
v_add_f32_e32 v49, v49, v82
v_add_f32_e32 v50, v50, v83
v_add_f32 v148, v148, neg(v115)
v_add_f32 v149, v149, neg(v116)
v_add_f32_e32 v49, v49, v115
v_add_f32_e32 v50, v50, v116
v_add_f32_e32 v148, v148, v82
v_add_f32_e32 v149, v149, v83
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
ds_store_b32 v1, v35 offset:4416
ds_store_b32 v1, v39 offset:4480
ds_store_b32 v1, v43 offset:4928
ds_store_b32 v1, v47 offset:4992
ds_store_b32 v1, v134 offset:4
ds_store_b32 v1, v138 offset:68
ds_store_b32 v1, v142 offset:516
ds_store_b32 v1, v146 offset:580
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v35, v51, v84
v_add_f32_e32 v36, v52, v85
v_add_f32 v134, v150, neg(v117)
v_add_f32 v135, v151, neg(v118)
v_add_f32_e32 v35, v35, v117
v_add_f32_e32 v36, v36, v118
v_add_f32_e32 v134, v134, v84
v_add_f32_e32 v135, v135, v85
v_add_f32_e32 v39, v55, v88
v_add_f32_e32 v40, v56, v89
v_add_f32 v138, v154, neg(v121)
v_add_f32 v139, v155, neg(v122)
v_add_f32_e32 v39, v39, v121
v_add_f32_e32 v40, v40, v122
v_add_f32_e32 v138, v138, v88
v_add_f32_e32 v139, v139, v89
v_add_f32_e32 v43, v59, v92
v_add_f32_e32 v44, v60, v93
v_add_f32 v142, v158, neg(v125)
v_add_f32 v143, v159, neg(v126)
v_add_f32_e32 v43, v43, v125
v_add_f32_e32 v44, v44, v126
v_add_f32_e32 v142, v142, v92
v_add_f32_e32 v143, v143, v93
v_add_f32_e32 v47, v63, v96
v_add_f32_e32 v48, v64, v97
v_add_f32 v146, v162, neg(v129)
v_add_f32 v147, v163, neg(v130)
v_add_f32_e32 v47, v47, v129
v_add_f32_e32 v48, v48, v130
v_add_f32_e32 v146, v146, v96
v_add_f32_e32 v147, v147, v97
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v37, v53, v86
v_add_f32_e32 v38, v54, v87
v_add_f32 v136, v152, neg(v119)
v_add_f32 v137, v153, neg(v120)
v_add_f32_e32 v37, v37, v119
v_add_f32_e32 v38, v38, v120
v_add_f32_e32 v136, v136, v86
v_add_f32_e32 v137, v137, v87
v_add_f32_e32 v41, v57, v90
v_add_f32_e32 v42, v58, v91
v_add_f32 v140, v156, neg(v123)
v_add_f32 v141, v157, neg(v124)
v_add_f32_e32 v41, v41, v123
v_add_f32_e32 v42, v42, v124
v_add_f32_e32 v140, v140, v90
v_add_f32_e32 v141, v141, v91
v_add_f32_e32 v45, v61, v94
v_add_f32_e32 v46, v62, v95
v_add_f32 v144, v160, neg(v127)
v_add_f32 v145, v161, neg(v128)
v_add_f32_e32 v45, v45, v127
v_add_f32_e32 v46, v46, v128
v_add_f32_e32 v144, v144, v94
v_add_f32_e32 v145, v145, v95
v_add_f32_e32 v49, v65, v98
v_add_f32_e32 v50, v66, v99
v_add_f32 v148, v164, neg(v131)
v_add_f32 v149, v165, neg(v132)
v_add_f32_e32 v49, v49, v131
v_add_f32_e32 v50, v50, v132
v_add_f32_e32 v148, v148, v98
v_add_f32_e32 v149, v149, v99
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 0
v_readlane_b32 s98, v31, 2
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 40, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 115, 40, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 40, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 121, 40, 331, 0x0, 0x3, 0x0, 0x0
_v_cmp_lt_f16__vop3 106, 70, 128, 0x0
v_cndmask_b16 v70, v70, v103, vcc
_v_cmp_lt_f16__vop3 106, 71, 128, 0x0
v_cndmask_b16 v71, v71, v115, vcc
_v_cmp_lt_f16__vop3 106, 74, 128, 0x0
v_cndmask_b16 v74, v74, v92, vcc
_v_cmp_lt_f16__vop3 106, 75, 128, 0x0
v_cndmask_b16 v75, v75, v121, vcc
_v_cmp_lt_f16__vop3 106, 70, 128, 0x1
_v_cndmask_b16__vop3 70, 70, 103, 106, 0xb
_v_cmp_lt_f16__vop3 106, 71, 128, 0x1
_v_cndmask_b16__vop3 71, 71, 115, 106, 0xb
_v_cmp_lt_f16__vop3 106, 74, 128, 0x1
_v_cndmask_b16__vop3 74, 74, 92, 106, 0xb
_v_cmp_lt_f16__vop3 106, 75, 128, 0x1
_v_cndmask_b16__vop3 75, 75, 121, 106, 0xb
ds_load_b32 v101, v2 offset:8896
ds_store_b32 v1, v35 offset:4416
ds_load_b32 v102, v2 offset:9024
ds_store_b32 v1, v39 offset:4480
ds_load_b32 v105, v2 offset:9152
ds_store_b32 v1, v43 offset:4928
ds_load_b32 v106, v2 offset:9280
ds_store_b32 v1, v47 offset:4992
ds_load_b32 v109, v2 offset:11072
ds_store_b32 v1, v134 offset:4
ds_load_b32 v110, v2 offset:11200
ds_store_b32 v1, v138 offset:68
ds_load_b32 v113, v2 offset:11328
ds_store_b32 v1, v142 offset:516
ds_load_b32 v114, v2 offset:11456
ds_store_b32 v1, v146 offset:580
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 1
v_readlane_b32 s98, v31, 3
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 40, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 115, 40, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 40, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 121, 40, 339, 0x0, 0x3, 0x0, 0x0
_v_cmp_lt_f16__vop3 106, 78, 128, 0x0
v_cndmask_b16 v78, v78, v103, vcc
_v_cmp_lt_f16__vop3 106, 79, 128, 0x0
v_cndmask_b16 v79, v79, v115, vcc
_v_cmp_lt_f16__vop3 106, 82, 128, 0x0
v_cndmask_b16 v82, v82, v92, vcc
_v_cmp_lt_f16__vop3 106, 83, 128, 0x0
v_cndmask_b16 v83, v83, v121, vcc
_v_cmp_lt_f16__vop3 106, 78, 128, 0x1
_v_cndmask_b16__vop3 78, 78, 103, 106, 0xb
_v_cmp_lt_f16__vop3 106, 79, 128, 0x1
_v_cndmask_b16__vop3 79, 79, 115, 106, 0xb
_v_cmp_lt_f16__vop3 106, 82, 128, 0x1
_v_cndmask_b16__vop3 82, 82, 92, 106, 0xb
_v_cmp_lt_f16__vop3 106, 83, 128, 0x1
_v_cndmask_b16__vop3 83, 83, 121, 106, 0xb
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_cmp_ge_u32 0, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 1, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 8, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 9, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 4
v_readlane_b32 s98, v31, 6
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 40, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 115, 40, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 40, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 121, 40, 331, 0x0, 0x3, 0x0, 0x0
_v_cmp_lt_f16__vop3 106, 70, 128, 0x0
v_cndmask_b16 v70, v70, v103, vcc
_v_cmp_lt_f16__vop3 106, 71, 128, 0x0
v_cndmask_b16 v71, v71, v115, vcc
_v_cmp_lt_f16__vop3 106, 74, 128, 0x0
v_cndmask_b16 v74, v74, v92, vcc
_v_cmp_lt_f16__vop3 106, 75, 128, 0x0
v_cndmask_b16 v75, v75, v121, vcc
_v_cmp_lt_f16__vop3 106, 70, 128, 0x1
_v_cndmask_b16__vop3 70, 70, 103, 106, 0xb
_v_cmp_lt_f16__vop3 106, 71, 128, 0x1
_v_cndmask_b16__vop3 71, 71, 115, 106, 0xb
_v_cmp_lt_f16__vop3 106, 74, 128, 0x1
_v_cndmask_b16__vop3 74, 74, 92, 106, 0xb
_v_cmp_lt_f16__vop3 106, 75, 128, 0x1
_v_cndmask_b16__vop3 75, 75, 121, 106, 0xb
ds_load_b32 v101, v2 offset:8896
ds_load_b32 v102, v2 offset:9024
ds_load_b32 v105, v2 offset:9152
ds_load_b32 v106, v2 offset:9280
ds_load_b32 v109, v2 offset:11072
ds_load_b32 v110, v2 offset:11200
ds_load_b32 v113, v2 offset:11328
ds_load_b32 v114, v2 offset:11456
s_cmp_ge_u32 2, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 3, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 10, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 11, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 5, s85
s_mul_hi_i32 s101, 5, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 5
v_readlane_b32 s98, v31, 7
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 40, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 115, 40, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 40, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 121, 40, 339, 0x0, 0x3, 0x0, 0x0
_v_cmp_lt_f16__vop3 106, 78, 128, 0x0
v_cndmask_b16 v78, v78, v103, vcc
_v_cmp_lt_f16__vop3 106, 79, 128, 0x0
v_cndmask_b16 v79, v79, v115, vcc
_v_cmp_lt_f16__vop3 106, 82, 128, 0x0
v_cndmask_b16 v82, v82, v92, vcc
_v_cmp_lt_f16__vop3 106, 83, 128, 0x0
v_cndmask_b16 v83, v83, v121, vcc
_v_cmp_lt_f16__vop3 106, 78, 128, 0x1
_v_cndmask_b16__vop3 78, 78, 103, 106, 0xb
_v_cmp_lt_f16__vop3 106, 79, 128, 0x1
_v_cndmask_b16__vop3 79, 79, 115, 106, 0xb
_v_cmp_lt_f16__vop3 106, 82, 128, 0x1
_v_cndmask_b16__vop3 82, 82, 92, 106, 0xb
_v_cmp_lt_f16__vop3 106, 83, 128, 0x1
_v_cndmask_b16__vop3 83, 83, 121, 106, 0xb
s_cmp_ge_u32 16, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 17, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 24, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 25, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_cmp_ge_u32 18, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 19, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 26, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 27, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 0, s85
s_mul_hi_i32 s101, 0, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_setprio 1
s_branch 7505
s_nop 0
v_add_f32_e32 v35, v35, v68
v_add_f32_e32 v36, v36, v69
v_add_f32 v134, v134, neg(v101)
v_add_f32 v135, v135, neg(v102)
v_add_f32_e32 v35, v35, v101
v_add_f32_e32 v36, v36, v102
v_add_f32_e32 v134, v134, v68
v_add_f32_e32 v135, v135, v69
v_add_f32_e32 v39, v39, v72
v_add_f32_e32 v40, v40, v73
v_add_f32 v138, v138, neg(v105)
v_add_f32 v139, v139, neg(v106)
v_add_f32_e32 v39, v39, v105
v_add_f32_e32 v40, v40, v106
v_add_f32_e32 v138, v138, v72
v_add_f32_e32 v139, v139, v73
v_add_f32_e32 v43, v43, v76
v_add_f32_e32 v44, v44, v77
v_add_f32 v142, v142, neg(v109)
v_add_f32 v143, v143, neg(v110)
v_add_f32_e32 v43, v43, v109
v_add_f32_e32 v44, v44, v110
v_add_f32_e32 v142, v142, v76
v_add_f32_e32 v143, v143, v77
v_add_f32_e32 v47, v47, v80
v_add_f32_e32 v48, v48, v81
v_add_f32 v146, v146, neg(v113)
v_add_f32 v147, v147, neg(v114)
v_add_f32_e32 v47, v47, v113
v_add_f32_e32 v48, v48, v114
v_add_f32_e32 v146, v146, v80
v_add_f32_e32 v147, v147, v81
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_store_b32 v1, v35 offset:4416
ds_store_b32 v1, v39 offset:4480
ds_store_b32 v1, v43 offset:4928
ds_store_b32 v1, v47 offset:4992
ds_store_b32 v1, v134 offset:4
ds_store_b32 v1, v138 offset:68
ds_store_b32 v1, v142 offset:516
ds_store_b32 v1, v146 offset:580
v_add_f32_e32 v37, v37, v70
v_add_f32_e32 v38, v38, v71
v_add_f32 v136, v136, neg(v103)
v_add_f32 v137, v137, neg(v104)
v_add_f32_e32 v37, v37, v103
v_add_f32_e32 v38, v38, v104
v_add_f32_e32 v136, v136, v70
v_add_f32_e32 v137, v137, v71
v_add_f32_e32 v41, v41, v74
v_add_f32_e32 v42, v42, v75
v_add_f32 v140, v140, neg(v107)
v_add_f32 v141, v141, neg(v108)
v_add_f32_e32 v41, v41, v107
v_add_f32_e32 v42, v42, v108
v_add_f32_e32 v140, v140, v74
v_add_f32_e32 v141, v141, v75
v_add_f32_e32 v45, v45, v78
v_add_f32_e32 v46, v46, v79
v_add_f32 v144, v144, neg(v111)
v_add_f32 v145, v145, neg(v112)
v_add_f32_e32 v45, v45, v111
v_add_f32_e32 v46, v46, v112
v_add_f32_e32 v144, v144, v78
v_add_f32_e32 v145, v145, v79
v_add_f32_e32 v49, v49, v82
v_add_f32_e32 v50, v50, v83
v_add_f32 v148, v148, neg(v115)
v_add_f32 v149, v149, neg(v116)
v_add_f32_e32 v49, v49, v115
v_add_f32_e32 v50, v50, v116
v_add_f32_e32 v148, v148, v82
v_add_f32_e32 v149, v149, v83
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
v_add_f32_e32 v35, v51, v84
v_add_f32_e32 v36, v52, v85
v_add_f32 v134, v150, neg(v117)
v_add_f32 v135, v151, neg(v118)
v_add_f32_e32 v35, v35, v117
v_add_f32_e32 v36, v36, v118
v_add_f32_e32 v134, v134, v84
v_add_f32_e32 v135, v135, v85
v_add_f32_e32 v39, v55, v88
v_add_f32_e32 v40, v56, v89
v_add_f32 v138, v154, neg(v121)
v_add_f32 v139, v155, neg(v122)
v_add_f32_e32 v39, v39, v121
v_add_f32_e32 v40, v40, v122
v_add_f32_e32 v138, v138, v88
v_add_f32_e32 v139, v139, v89
v_add_f32_e32 v43, v59, v92
v_add_f32_e32 v44, v60, v93
v_add_f32 v142, v158, neg(v125)
v_add_f32 v143, v159, neg(v126)
v_add_f32_e32 v43, v43, v125
v_add_f32_e32 v44, v44, v126
v_add_f32_e32 v142, v142, v92
v_add_f32_e32 v143, v143, v93
v_add_f32_e32 v47, v63, v96
v_add_f32_e32 v48, v64, v97
v_add_f32 v146, v162, neg(v129)
v_add_f32 v147, v163, neg(v130)
v_add_f32_e32 v47, v47, v129
v_add_f32_e32 v48, v48, v130
v_add_f32_e32 v146, v146, v96
v_add_f32_e32 v147, v147, v97
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v101, v2 offset:8896
ds_store_b32 v1, v35 offset:4416
ds_load_b32 v102, v2 offset:9024
ds_store_b32 v1, v39 offset:4480
ds_load_b32 v105, v2 offset:9152
ds_store_b32 v1, v43 offset:4928
ds_load_b32 v106, v2 offset:9280
ds_store_b32 v1, v47 offset:4992
ds_load_b32 v109, v2 offset:11072
ds_store_b32 v1, v134 offset:4
ds_load_b32 v110, v2 offset:11200
ds_store_b32 v1, v138 offset:68
ds_load_b32 v113, v2 offset:11328
ds_store_b32 v1, v142 offset:516
ds_load_b32 v114, v2 offset:11456
ds_store_b32 v1, v146 offset:580
v_add_f32_e32 v37, v53, v86
v_add_f32_e32 v38, v54, v87
v_add_f32 v136, v152, neg(v119)
v_add_f32 v137, v153, neg(v120)
v_add_f32_e32 v37, v37, v119
v_add_f32_e32 v38, v38, v120
v_add_f32_e32 v136, v136, v86
v_add_f32_e32 v137, v137, v87
v_add_f32_e32 v41, v57, v90
v_add_f32_e32 v42, v58, v91
v_add_f32 v140, v156, neg(v123)
v_add_f32 v141, v157, neg(v124)
v_add_f32_e32 v41, v41, v123
v_add_f32_e32 v42, v42, v124
v_add_f32_e32 v140, v140, v90
v_add_f32_e32 v141, v141, v91
v_add_f32_e32 v45, v61, v94
v_add_f32_e32 v46, v62, v95
v_add_f32 v144, v160, neg(v127)
v_add_f32 v145, v161, neg(v128)
v_add_f32_e32 v45, v45, v127
v_add_f32_e32 v46, v46, v128
v_add_f32_e32 v144, v144, v94
v_add_f32_e32 v145, v145, v95
v_add_f32_e32 v49, v65, v98
v_add_f32_e32 v50, v66, v99
v_add_f32 v148, v164, neg(v131)
v_add_f32 v149, v165, neg(v132)
v_add_f32_e32 v49, v49, v131
v_add_f32_e32 v50, v50, v132
v_add_f32_e32 v148, v148, v98
v_add_f32_e32 v149, v149, v99
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 0
v_readlane_b32 s98, v31, 2
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 40, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 115, 40, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 40, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 121, 40, 331, 0x0, 0x3, 0x0, 0x0
_v_cmp_lt_f16__vop3 106, 70, 128, 0x0
v_cndmask_b16 v70, v70, v103, vcc
_v_cmp_lt_f16__vop3 106, 71, 128, 0x0
v_cndmask_b16 v71, v71, v115, vcc
_v_cmp_lt_f16__vop3 106, 74, 128, 0x0
v_cndmask_b16 v74, v74, v92, vcc
_v_cmp_lt_f16__vop3 106, 75, 128, 0x0
v_cndmask_b16 v75, v75, v121, vcc
_v_cmp_lt_f16__vop3 106, 70, 128, 0x1
_v_cndmask_b16__vop3 70, 70, 103, 106, 0xb
_v_cmp_lt_f16__vop3 106, 71, 128, 0x1
_v_cndmask_b16__vop3 71, 71, 115, 106, 0xb
_v_cmp_lt_f16__vop3 106, 74, 128, 0x1
_v_cndmask_b16__vop3 74, 74, 92, 106, 0xb
_v_cmp_lt_f16__vop3 106, 75, 128, 0x1
_v_cndmask_b16__vop3 75, 75, 121, 106, 0xb
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_cmp_ge_u32 0, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 1, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 8, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 9, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 1
v_readlane_b32 s98, v31, 3
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 40, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 115, 40, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 40, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 121, 40, 339, 0x0, 0x3, 0x0, 0x0
_v_cmp_lt_f16__vop3 106, 78, 128, 0x0
v_cndmask_b16 v78, v78, v103, vcc
_v_cmp_lt_f16__vop3 106, 79, 128, 0x0
v_cndmask_b16 v79, v79, v115, vcc
_v_cmp_lt_f16__vop3 106, 82, 128, 0x0
v_cndmask_b16 v82, v82, v92, vcc
_v_cmp_lt_f16__vop3 106, 83, 128, 0x0
v_cndmask_b16 v83, v83, v121, vcc
_v_cmp_lt_f16__vop3 106, 78, 128, 0x1
_v_cndmask_b16__vop3 78, 78, 103, 106, 0xb
_v_cmp_lt_f16__vop3 106, 79, 128, 0x1
_v_cndmask_b16__vop3 79, 79, 115, 106, 0xb
_v_cmp_lt_f16__vop3 106, 82, 128, 0x1
_v_cndmask_b16__vop3 82, 82, 92, 106, 0xb
_v_cmp_lt_f16__vop3 106, 83, 128, 0x1
_v_cndmask_b16__vop3 83, 83, 121, 106, 0xb
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v101, v2 offset:8896
ds_load_b32 v102, v2 offset:9024
ds_load_b32 v105, v2 offset:9152
ds_load_b32 v106, v2 offset:9280
ds_load_b32 v109, v2 offset:11072
ds_load_b32 v110, v2 offset:11200
ds_load_b32 v113, v2 offset:11328
ds_load_b32 v114, v2 offset:11456
s_cmp_ge_u32 2, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 3, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 10, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 11, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 5, s85
s_mul_hi_i32 s101, 5, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 4
v_readlane_b32 s98, v31, 6
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 40, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 115, 40, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 40, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 121, 40, 331, 0x0, 0x3, 0x0, 0x0
_v_cmp_lt_f16__vop3 106, 70, 128, 0x0
v_cndmask_b16 v70, v70, v103, vcc
_v_cmp_lt_f16__vop3 106, 71, 128, 0x0
v_cndmask_b16 v71, v71, v115, vcc
_v_cmp_lt_f16__vop3 106, 74, 128, 0x0
v_cndmask_b16 v74, v74, v92, vcc
_v_cmp_lt_f16__vop3 106, 75, 128, 0x0
v_cndmask_b16 v75, v75, v121, vcc
_v_cmp_lt_f16__vop3 106, 70, 128, 0x1
_v_cndmask_b16__vop3 70, 70, 103, 106, 0xb
_v_cmp_lt_f16__vop3 106, 71, 128, 0x1
_v_cndmask_b16__vop3 71, 71, 115, 106, 0xb
_v_cmp_lt_f16__vop3 106, 74, 128, 0x1
_v_cndmask_b16__vop3 74, 74, 92, 106, 0xb
_v_cmp_lt_f16__vop3 106, 75, 128, 0x1
_v_cndmask_b16__vop3 75, 75, 121, 106, 0xb
s_waitcnt lgkmcnt(0)
s_barrier
s_cmp_ge_u32 16, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 17, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 24, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 25, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 5
v_readlane_b32 s98, v31, 7
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 103, 40, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 115, 40, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 92, 40, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 121, 40, 339, 0x0, 0x3, 0x0, 0x0
_v_cmp_lt_f16__vop3 106, 78, 128, 0x0
v_cndmask_b16 v78, v78, v103, vcc
_v_cmp_lt_f16__vop3 106, 79, 128, 0x0
v_cndmask_b16 v79, v79, v115, vcc
_v_cmp_lt_f16__vop3 106, 82, 128, 0x0
v_cndmask_b16 v82, v82, v92, vcc
_v_cmp_lt_f16__vop3 106, 83, 128, 0x0
v_cndmask_b16 v83, v83, v121, vcc
_v_cmp_lt_f16__vop3 106, 78, 128, 0x1
_v_cndmask_b16__vop3 78, 78, 103, 106, 0xb
_v_cmp_lt_f16__vop3 106, 79, 128, 0x1
_v_cndmask_b16__vop3 79, 79, 115, 106, 0xb
_v_cmp_lt_f16__vop3 106, 82, 128, 0x1
_v_cndmask_b16__vop3 82, 82, 92, 106, 0xb
_v_cmp_lt_f16__vop3 106, 83, 128, 0x1
_v_cndmask_b16__vop3 83, 83, 121, 106, 0xb
s_waitcnt lgkmcnt(0)
s_cmp_ge_u32 18, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 19, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 26, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 27, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 0, s85
s_mul_hi_i32 s101, 0, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_setprio 2
s_branch 6561
s_nop 0
v_add_f32_e32 v35, v35, v68
v_add_f32_e32 v36, v36, v69
v_add_f32 v134, v134, neg(v101)
v_add_f32 v135, v135, neg(v102)
v_add_f32_e32 v35, v35, v101
v_add_f32_e32 v36, v36, v102
v_add_f32_e32 v134, v134, v68
v_add_f32_e32 v135, v135, v69
v_add_f32_e32 v39, v39, v72
v_add_f32_e32 v40, v40, v73
v_add_f32 v138, v138, neg(v105)
v_add_f32 v139, v139, neg(v106)
v_add_f32_e32 v39, v39, v105
v_add_f32_e32 v40, v40, v106
v_add_f32_e32 v138, v138, v72
v_add_f32_e32 v139, v139, v73
v_add_f32_e32 v43, v43, v76
v_add_f32_e32 v44, v44, v77
v_add_f32 v142, v142, neg(v109)
v_add_f32 v143, v143, neg(v110)
v_add_f32_e32 v43, v43, v109
v_add_f32_e32 v44, v44, v110
v_add_f32_e32 v142, v142, v76
v_add_f32_e32 v143, v143, v77
v_add_f32_e32 v47, v47, v80
v_add_f32_e32 v48, v48, v81
v_add_f32 v146, v146, neg(v113)
v_add_f32 v147, v147, neg(v114)
v_add_f32_e32 v47, v47, v113
v_add_f32_e32 v48, v48, v114
v_add_f32_e32 v146, v146, v80
v_add_f32_e32 v147, v147, v81
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v37, v37, v70
v_add_f32_e32 v38, v38, v71
v_add_f32 v136, v136, neg(v103)
v_add_f32 v137, v137, neg(v104)
v_add_f32_e32 v37, v37, v103
v_add_f32_e32 v38, v38, v104
v_add_f32_e32 v136, v136, v70
v_add_f32_e32 v137, v137, v71
v_add_f32_e32 v41, v41, v74
v_add_f32_e32 v42, v42, v75
v_add_f32 v140, v140, neg(v107)
v_add_f32 v141, v141, neg(v108)
v_add_f32_e32 v41, v41, v107
v_add_f32_e32 v42, v42, v108
v_add_f32_e32 v140, v140, v74
v_add_f32_e32 v141, v141, v75
v_add_f32_e32 v45, v45, v78
v_add_f32_e32 v46, v46, v79
v_add_f32 v144, v144, neg(v111)
v_add_f32 v145, v145, neg(v112)
v_add_f32_e32 v45, v45, v111
v_add_f32_e32 v46, v46, v112
v_add_f32_e32 v144, v144, v78
v_add_f32_e32 v145, v145, v79
v_add_f32_e32 v49, v49, v82
v_add_f32_e32 v50, v50, v83
v_add_f32 v148, v148, neg(v115)
v_add_f32 v149, v149, neg(v116)
v_add_f32_e32 v49, v49, v115
v_add_f32_e32 v50, v50, v116
v_add_f32_e32 v148, v148, v82
v_add_f32_e32 v149, v149, v83
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
ds_store_b32 v1, v35 offset:4416
ds_store_b32 v1, v39 offset:4480
ds_store_b32 v1, v43 offset:4928
ds_store_b32 v1, v47 offset:4992
ds_store_b32 v1, v134 offset:4
ds_store_b32 v1, v138 offset:68
ds_store_b32 v1, v142 offset:516
ds_store_b32 v1, v146 offset:580
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v35, v51, v84
v_add_f32_e32 v36, v52, v85
v_add_f32 v134, v150, neg(v117)
v_add_f32 v135, v151, neg(v118)
v_add_f32_e32 v35, v35, v117
v_add_f32_e32 v36, v36, v118
v_add_f32_e32 v134, v134, v84
v_add_f32_e32 v135, v135, v85
v_add_f32_e32 v39, v55, v88
v_add_f32_e32 v40, v56, v89
v_add_f32 v138, v154, neg(v121)
v_add_f32 v139, v155, neg(v122)
v_add_f32_e32 v39, v39, v121
v_add_f32_e32 v40, v40, v122
v_add_f32_e32 v138, v138, v88
v_add_f32_e32 v139, v139, v89
v_add_f32_e32 v43, v59, v92
v_add_f32_e32 v44, v60, v93
v_add_f32 v142, v158, neg(v125)
v_add_f32 v143, v159, neg(v126)
v_add_f32_e32 v43, v43, v125
v_add_f32_e32 v44, v44, v126
v_add_f32_e32 v142, v142, v92
v_add_f32_e32 v143, v143, v93
v_add_f32_e32 v47, v63, v96
v_add_f32_e32 v48, v64, v97
v_add_f32 v146, v162, neg(v129)
v_add_f32 v147, v163, neg(v130)
v_add_f32_e32 v47, v47, v129
v_add_f32_e32 v48, v48, v130
v_add_f32_e32 v146, v146, v96
v_add_f32_e32 v147, v147, v97
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v37, v53, v86
v_add_f32_e32 v38, v54, v87
v_add_f32 v136, v152, neg(v119)
v_add_f32 v137, v153, neg(v120)
v_add_f32_e32 v37, v37, v119
v_add_f32_e32 v38, v38, v120
v_add_f32_e32 v136, v136, v86
v_add_f32_e32 v137, v137, v87
v_add_f32_e32 v41, v57, v90
v_add_f32_e32 v42, v58, v91
v_add_f32 v140, v156, neg(v123)
v_add_f32 v141, v157, neg(v124)
v_add_f32_e32 v41, v41, v123
v_add_f32_e32 v42, v42, v124
v_add_f32_e32 v140, v140, v90
v_add_f32_e32 v141, v141, v91
v_add_f32_e32 v45, v61, v94
v_add_f32_e32 v46, v62, v95
v_add_f32 v144, v160, neg(v127)
v_add_f32 v145, v161, neg(v128)
v_add_f32_e32 v45, v45, v127
v_add_f32_e32 v46, v46, v128
v_add_f32_e32 v144, v144, v94
v_add_f32_e32 v145, v145, v95
v_add_f32_e32 v49, v65, v98
v_add_f32_e32 v50, v66, v99
v_add_f32 v148, v164, neg(v131)
v_add_f32 v149, v165, neg(v132)
v_add_f32_e32 v49, v49, v131
v_add_f32_e32 v50, v50, v132
v_add_f32_e32 v148, v148, v98
v_add_f32_e32 v149, v149, v99
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 0
v_readlane_b32 s98, v31, 2
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 70, 128, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 71, 128, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 74, 128, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 75, 128, 331, 0x0, 0x3, 0x0, 0x0
ds_load_b32 v101, v2 offset:8896
ds_store_b32 v1, v35 offset:4416
ds_load_b32 v102, v2 offset:9024
ds_store_b32 v1, v39 offset:4480
ds_load_b32 v105, v2 offset:9152
ds_store_b32 v1, v43 offset:4928
ds_load_b32 v106, v2 offset:9280
ds_store_b32 v1, v47 offset:4992
ds_load_b32 v109, v2 offset:11072
ds_store_b32 v1, v134 offset:4
ds_load_b32 v110, v2 offset:11200
ds_store_b32 v1, v138 offset:68
ds_load_b32 v113, v2 offset:11328
ds_store_b32 v1, v142 offset:516
ds_load_b32 v114, v2 offset:11456
ds_store_b32 v1, v146 offset:580
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 1
v_readlane_b32 s98, v31, 3
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 78, 128, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 79, 128, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 82, 128, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 83, 128, 339, 0x0, 0x3, 0x0, 0x0
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_cmp_ge_u32 0, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 1, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 8, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 9, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 4
v_readlane_b32 s98, v31, 6
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 70, 128, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 71, 128, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 74, 128, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 75, 128, 331, 0x0, 0x3, 0x0, 0x0
ds_load_b32 v101, v2 offset:8896
ds_load_b32 v102, v2 offset:9024
ds_load_b32 v105, v2 offset:9152
ds_load_b32 v106, v2 offset:9280
ds_load_b32 v109, v2 offset:11072
ds_load_b32 v110, v2 offset:11200
ds_load_b32 v113, v2 offset:11328
ds_load_b32 v114, v2 offset:11456
s_cmp_ge_u32 2, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 3, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 10, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 11, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 5, s85
s_mul_hi_i32 s101, 5, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 5
v_readlane_b32 s98, v31, 7
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 78, 128, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 79, 128, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 82, 128, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 83, 128, 339, 0x0, 0x3, 0x0, 0x0
s_cmp_ge_u32 16, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 17, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 24, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 25, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_cmp_ge_u32 18, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 19, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 26, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 27, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 0, s85
s_mul_hi_i32 s101, 0, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_setprio 1
s_branch 5745
s_nop 0
v_add_f32_e32 v35, v35, v68
v_add_f32_e32 v36, v36, v69
v_add_f32 v134, v134, neg(v101)
v_add_f32 v135, v135, neg(v102)
v_add_f32_e32 v35, v35, v101
v_add_f32_e32 v36, v36, v102
v_add_f32_e32 v134, v134, v68
v_add_f32_e32 v135, v135, v69
v_add_f32_e32 v39, v39, v72
v_add_f32_e32 v40, v40, v73
v_add_f32 v138, v138, neg(v105)
v_add_f32 v139, v139, neg(v106)
v_add_f32_e32 v39, v39, v105
v_add_f32_e32 v40, v40, v106
v_add_f32_e32 v138, v138, v72
v_add_f32_e32 v139, v139, v73
v_add_f32_e32 v43, v43, v76
v_add_f32_e32 v44, v44, v77
v_add_f32 v142, v142, neg(v109)
v_add_f32 v143, v143, neg(v110)
v_add_f32_e32 v43, v43, v109
v_add_f32_e32 v44, v44, v110
v_add_f32_e32 v142, v142, v76
v_add_f32_e32 v143, v143, v77
v_add_f32_e32 v47, v47, v80
v_add_f32_e32 v48, v48, v81
v_add_f32 v146, v146, neg(v113)
v_add_f32 v147, v147, neg(v114)
v_add_f32_e32 v47, v47, v113
v_add_f32_e32 v48, v48, v114
v_add_f32_e32 v146, v146, v80
v_add_f32_e32 v147, v147, v81
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_store_b32 v1, v35 offset:4416
ds_store_b32 v1, v39 offset:4480
ds_store_b32 v1, v43 offset:4928
ds_store_b32 v1, v47 offset:4992
ds_store_b32 v1, v134 offset:4
ds_store_b32 v1, v138 offset:68
ds_store_b32 v1, v142 offset:516
ds_store_b32 v1, v146 offset:580
v_add_f32_e32 v37, v37, v70
v_add_f32_e32 v38, v38, v71
v_add_f32 v136, v136, neg(v103)
v_add_f32 v137, v137, neg(v104)
v_add_f32_e32 v37, v37, v103
v_add_f32_e32 v38, v38, v104
v_add_f32_e32 v136, v136, v70
v_add_f32_e32 v137, v137, v71
v_add_f32_e32 v41, v41, v74
v_add_f32_e32 v42, v42, v75
v_add_f32 v140, v140, neg(v107)
v_add_f32 v141, v141, neg(v108)
v_add_f32_e32 v41, v41, v107
v_add_f32_e32 v42, v42, v108
v_add_f32_e32 v140, v140, v74
v_add_f32_e32 v141, v141, v75
v_add_f32_e32 v45, v45, v78
v_add_f32_e32 v46, v46, v79
v_add_f32 v144, v144, neg(v111)
v_add_f32 v145, v145, neg(v112)
v_add_f32_e32 v45, v45, v111
v_add_f32_e32 v46, v46, v112
v_add_f32_e32 v144, v144, v78
v_add_f32_e32 v145, v145, v79
v_add_f32_e32 v49, v49, v82
v_add_f32_e32 v50, v50, v83
v_add_f32 v148, v148, neg(v115)
v_add_f32 v149, v149, neg(v116)
v_add_f32_e32 v49, v49, v115
v_add_f32_e32 v50, v50, v116
v_add_f32_e32 v148, v148, v82
v_add_f32_e32 v149, v149, v83
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
v_add_f32_e32 v35, v51, v84
v_add_f32_e32 v36, v52, v85
v_add_f32 v134, v150, neg(v117)
v_add_f32 v135, v151, neg(v118)
v_add_f32_e32 v35, v35, v117
v_add_f32_e32 v36, v36, v118
v_add_f32_e32 v134, v134, v84
v_add_f32_e32 v135, v135, v85
v_add_f32_e32 v39, v55, v88
v_add_f32_e32 v40, v56, v89
v_add_f32 v138, v154, neg(v121)
v_add_f32 v139, v155, neg(v122)
v_add_f32_e32 v39, v39, v121
v_add_f32_e32 v40, v40, v122
v_add_f32_e32 v138, v138, v88
v_add_f32_e32 v139, v139, v89
v_add_f32_e32 v43, v59, v92
v_add_f32_e32 v44, v60, v93
v_add_f32 v142, v158, neg(v125)
v_add_f32 v143, v159, neg(v126)
v_add_f32_e32 v43, v43, v125
v_add_f32_e32 v44, v44, v126
v_add_f32_e32 v142, v142, v92
v_add_f32_e32 v143, v143, v93
v_add_f32_e32 v47, v63, v96
v_add_f32_e32 v48, v64, v97
v_add_f32 v146, v162, neg(v129)
v_add_f32 v147, v163, neg(v130)
v_add_f32_e32 v47, v47, v129
v_add_f32_e32 v48, v48, v130
v_add_f32_e32 v146, v146, v96
v_add_f32_e32 v147, v147, v97
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v101, v2 offset:8896
ds_store_b32 v1, v35 offset:4416
ds_load_b32 v102, v2 offset:9024
ds_store_b32 v1, v39 offset:4480
ds_load_b32 v105, v2 offset:9152
ds_store_b32 v1, v43 offset:4928
ds_load_b32 v106, v2 offset:9280
ds_store_b32 v1, v47 offset:4992
ds_load_b32 v109, v2 offset:11072
ds_store_b32 v1, v134 offset:4
ds_load_b32 v110, v2 offset:11200
ds_store_b32 v1, v138 offset:68
ds_load_b32 v113, v2 offset:11328
ds_store_b32 v1, v142 offset:516
ds_load_b32 v114, v2 offset:11456
ds_store_b32 v1, v146 offset:580
v_add_f32_e32 v37, v53, v86
v_add_f32_e32 v38, v54, v87
v_add_f32 v136, v152, neg(v119)
v_add_f32 v137, v153, neg(v120)
v_add_f32_e32 v37, v37, v119
v_add_f32_e32 v38, v38, v120
v_add_f32_e32 v136, v136, v86
v_add_f32_e32 v137, v137, v87
v_add_f32_e32 v41, v57, v90
v_add_f32_e32 v42, v58, v91
v_add_f32 v140, v156, neg(v123)
v_add_f32 v141, v157, neg(v124)
v_add_f32_e32 v41, v41, v123
v_add_f32_e32 v42, v42, v124
v_add_f32_e32 v140, v140, v90
v_add_f32_e32 v141, v141, v91
v_add_f32_e32 v45, v61, v94
v_add_f32_e32 v46, v62, v95
v_add_f32 v144, v160, neg(v127)
v_add_f32 v145, v161, neg(v128)
v_add_f32_e32 v45, v45, v127
v_add_f32_e32 v46, v46, v128
v_add_f32_e32 v144, v144, v94
v_add_f32_e32 v145, v145, v95
v_add_f32_e32 v49, v65, v98
v_add_f32_e32 v50, v66, v99
v_add_f32 v148, v164, neg(v131)
v_add_f32 v149, v165, neg(v132)
v_add_f32_e32 v49, v49, v131
v_add_f32_e32 v50, v50, v132
v_add_f32_e32 v148, v148, v98
v_add_f32_e32 v149, v149, v99
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 0
v_readlane_b32 s98, v31, 2
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 70, 128, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 71, 128, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 74, 128, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 75, 128, 331, 0x0, 0x3, 0x0, 0x0
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_cmp_ge_u32 0, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 1, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 8, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 9, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 1
v_readlane_b32 s98, v31, 3
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 78, 128, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 79, 128, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 82, 128, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 83, 128, 339, 0x0, 0x3, 0x0, 0x0
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v101, v2 offset:8896
ds_load_b32 v102, v2 offset:9024
ds_load_b32 v105, v2 offset:9152
ds_load_b32 v106, v2 offset:9280
ds_load_b32 v109, v2 offset:11072
ds_load_b32 v110, v2 offset:11200
ds_load_b32 v113, v2 offset:11328
ds_load_b32 v114, v2 offset:11456
s_cmp_ge_u32 2, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 3, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 10, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 11, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 5, s85
s_mul_hi_i32 s101, 5, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 4
v_readlane_b32 s98, v31, 6
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 70, 128, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 71, 128, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 74, 128, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 75, 128, 331, 0x0, 0x3, 0x0, 0x0
s_waitcnt lgkmcnt(0)
s_barrier
s_cmp_ge_u32 16, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 17, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 24, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 25, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 5
v_readlane_b32 s98, v31, 7
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 78, 128, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 79, 128, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 82, 128, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_max_num_f16__vop3p 83, 128, 339, 0x0, 0x3, 0x0, 0x0
s_waitcnt lgkmcnt(0)
s_cmp_ge_u32 18, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 19, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 26, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 27, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 0, s85
s_mul_hi_i32 s101, 0, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_setprio 2
s_branch 4929
s_nop 0
v_add_f32_e32 v35, v35, v68
v_add_f32_e32 v36, v36, v69
v_add_f32 v134, v134, neg(v101)
v_add_f32 v135, v135, neg(v102)
v_add_f32_e32 v35, v35, v101
v_add_f32_e32 v36, v36, v102
v_add_f32_e32 v134, v134, v68
v_add_f32_e32 v135, v135, v69
v_add_f32_e32 v39, v39, v72
v_add_f32_e32 v40, v40, v73
v_add_f32 v138, v138, neg(v105)
v_add_f32 v139, v139, neg(v106)
v_add_f32_e32 v39, v39, v105
v_add_f32_e32 v40, v40, v106
v_add_f32_e32 v138, v138, v72
v_add_f32_e32 v139, v139, v73
v_add_f32_e32 v43, v43, v76
v_add_f32_e32 v44, v44, v77
v_add_f32 v142, v142, neg(v109)
v_add_f32 v143, v143, neg(v110)
v_add_f32_e32 v43, v43, v109
v_add_f32_e32 v44, v44, v110
v_add_f32_e32 v142, v142, v76
v_add_f32_e32 v143, v143, v77
v_add_f32_e32 v47, v47, v80
v_add_f32_e32 v48, v48, v81
v_add_f32 v146, v146, neg(v113)
v_add_f32 v147, v147, neg(v114)
v_add_f32_e32 v47, v47, v113
v_add_f32_e32 v48, v48, v114
v_add_f32_e32 v146, v146, v80
v_add_f32_e32 v147, v147, v81
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v37, v37, v70
v_add_f32_e32 v38, v38, v71
v_add_f32 v136, v136, neg(v103)
v_add_f32 v137, v137, neg(v104)
v_add_f32_e32 v37, v37, v103
v_add_f32_e32 v38, v38, v104
v_add_f32_e32 v136, v136, v70
v_add_f32_e32 v137, v137, v71
v_add_f32_e32 v41, v41, v74
v_add_f32_e32 v42, v42, v75
v_add_f32 v140, v140, neg(v107)
v_add_f32 v141, v141, neg(v108)
v_add_f32_e32 v41, v41, v107
v_add_f32_e32 v42, v42, v108
v_add_f32_e32 v140, v140, v74
v_add_f32_e32 v141, v141, v75
v_add_f32_e32 v45, v45, v78
v_add_f32_e32 v46, v46, v79
v_add_f32 v144, v144, neg(v111)
v_add_f32 v145, v145, neg(v112)
v_add_f32_e32 v45, v45, v111
v_add_f32_e32 v46, v46, v112
v_add_f32_e32 v144, v144, v78
v_add_f32_e32 v145, v145, v79
v_add_f32_e32 v49, v49, v82
v_add_f32_e32 v50, v50, v83
v_add_f32 v148, v148, neg(v115)
v_add_f32 v149, v149, neg(v116)
v_add_f32_e32 v49, v49, v115
v_add_f32_e32 v50, v50, v116
v_add_f32_e32 v148, v148, v82
v_add_f32_e32 v149, v149, v83
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
ds_store_b32 v1, v35 offset:4416
ds_store_b32 v1, v39 offset:4480
ds_store_b32 v1, v43 offset:4928
ds_store_b32 v1, v47 offset:4992
ds_store_b32 v1, v134 offset:4
ds_store_b32 v1, v138 offset:68
ds_store_b32 v1, v142 offset:516
ds_store_b32 v1, v146 offset:580
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v35, v51, v84
v_add_f32_e32 v36, v52, v85
v_add_f32 v134, v150, neg(v117)
v_add_f32 v135, v151, neg(v118)
v_add_f32_e32 v35, v35, v117
v_add_f32_e32 v36, v36, v118
v_add_f32_e32 v134, v134, v84
v_add_f32_e32 v135, v135, v85
v_add_f32_e32 v39, v55, v88
v_add_f32_e32 v40, v56, v89
v_add_f32 v138, v154, neg(v121)
v_add_f32 v139, v155, neg(v122)
v_add_f32_e32 v39, v39, v121
v_add_f32_e32 v40, v40, v122
v_add_f32_e32 v138, v138, v88
v_add_f32_e32 v139, v139, v89
v_add_f32_e32 v43, v59, v92
v_add_f32_e32 v44, v60, v93
v_add_f32 v142, v158, neg(v125)
v_add_f32 v143, v159, neg(v126)
v_add_f32_e32 v43, v43, v125
v_add_f32_e32 v44, v44, v126
v_add_f32_e32 v142, v142, v92
v_add_f32_e32 v143, v143, v93
v_add_f32_e32 v47, v63, v96
v_add_f32_e32 v48, v64, v97
v_add_f32 v146, v162, neg(v129)
v_add_f32 v147, v163, neg(v130)
v_add_f32_e32 v47, v47, v129
v_add_f32_e32 v48, v48, v130
v_add_f32_e32 v146, v146, v96
v_add_f32_e32 v147, v147, v97
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v37, v53, v86
v_add_f32_e32 v38, v54, v87
v_add_f32 v136, v152, neg(v119)
v_add_f32 v137, v153, neg(v120)
v_add_f32_e32 v37, v37, v119
v_add_f32_e32 v38, v38, v120
v_add_f32_e32 v136, v136, v86
v_add_f32_e32 v137, v137, v87
v_add_f32_e32 v41, v57, v90
v_add_f32_e32 v42, v58, v91
v_add_f32 v140, v156, neg(v123)
v_add_f32 v141, v157, neg(v124)
v_add_f32_e32 v41, v41, v123
v_add_f32_e32 v42, v42, v124
v_add_f32_e32 v140, v140, v90
v_add_f32_e32 v141, v141, v91
v_add_f32_e32 v45, v61, v94
v_add_f32_e32 v46, v62, v95
v_add_f32 v144, v160, neg(v127)
v_add_f32 v145, v161, neg(v128)
v_add_f32_e32 v45, v45, v127
v_add_f32_e32 v46, v46, v128
v_add_f32_e32 v144, v144, v94
v_add_f32_e32 v145, v145, v95
v_add_f32_e32 v49, v65, v98
v_add_f32_e32 v50, v66, v99
v_add_f32 v148, v164, neg(v131)
v_add_f32 v149, v165, neg(v132)
v_add_f32_e32 v49, v49, v131
v_add_f32_e32 v50, v50, v132
v_add_f32_e32 v148, v148, v98
v_add_f32_e32 v149, v149, v99
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 0
v_readlane_b32 s98, v31, 2
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v104, 0xbfb8aa3b
v_mov_b32 v116, 0xbfb8aa3b
v_mov_b32 v93, 0xbfb8aa3b
v_mov_b32 v122, 0xbfb8aa3b
v_fma_mix_f32 v103, v104, v70, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v115, v116, v71, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v92, v93, v74, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v121, v122, v75, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v104, v104, v70, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v116, v116, v71, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v93, v93, v74, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v122, v122, v75, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v103, v103
v_exp_f32_e32 v115, v115
v_exp_f32_e32 v92, v92
v_exp_f32_e32 v121, v121
v_exp_f32_e32 v104, v104
v_exp_f32_e32 v116, v116
v_exp_f32_e32 v93, v93
v_exp_f32_e32 v122, v122
v_add_f32_e32 v103, 1.0, v103
v_add_f32_e32 v115, 1.0, v115
v_add_f32_e32 v92, 1.0, v92
v_add_f32_e32 v121, 1.0, v121
v_add_f32_e32 v104, 1.0, v104
v_add_f32_e32 v116, 1.0, v116
v_add_f32_e32 v93, 1.0, v93
v_add_f32_e32 v122, 1.0, v122
v_rcp_f32 v103, v103
v_rcp_f32 v115, v115
v_rcp_f32 v92, v92
v_rcp_f32 v121, v121
v_rcp_f32 v104, v104
v_rcp_f32 v116, v116
v_rcp_f32 v93, v93
v_rcp_f32 v122, v122
v_cvt_f16_f32 v70, v103
v_cvt_f16_f32 v71, v115
v_cvt_f16_f32 v74, v92
v_cvt_f16_f32 v75, v121
v_cvt_f16_f32 v70.h, v104
v_cvt_f16_f32 v71.h, v116
v_cvt_f16_f32 v74.h, v93
v_cvt_f16_f32 v75.h, v122
ds_load_b32 v101, v2 offset:8896
ds_store_b32 v1, v35 offset:4416
ds_load_b32 v102, v2 offset:9024
ds_store_b32 v1, v39 offset:4480
ds_load_b32 v105, v2 offset:9152
ds_store_b32 v1, v43 offset:4928
ds_load_b32 v106, v2 offset:9280
ds_store_b32 v1, v47 offset:4992
ds_load_b32 v109, v2 offset:11072
ds_store_b32 v1, v134 offset:4
ds_load_b32 v110, v2 offset:11200
ds_store_b32 v1, v138 offset:68
ds_load_b32 v113, v2 offset:11328
ds_store_b32 v1, v142 offset:516
ds_load_b32 v114, v2 offset:11456
ds_store_b32 v1, v146 offset:580
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 1
v_readlane_b32 s98, v31, 3
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v104, 0xbfb8aa3b
v_mov_b32 v116, 0xbfb8aa3b
v_mov_b32 v93, 0xbfb8aa3b
v_mov_b32 v122, 0xbfb8aa3b
v_fma_mix_f32 v103, v104, v78, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v115, v116, v79, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v92, v93, v82, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v121, v122, v83, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v104, v104, v78, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v116, v116, v79, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v93, v93, v82, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v122, v122, v83, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v103, v103
v_exp_f32_e32 v115, v115
v_exp_f32_e32 v92, v92
v_exp_f32_e32 v121, v121
v_exp_f32_e32 v104, v104
v_exp_f32_e32 v116, v116
v_exp_f32_e32 v93, v93
v_exp_f32_e32 v122, v122
v_add_f32_e32 v103, 1.0, v103
v_add_f32_e32 v115, 1.0, v115
v_add_f32_e32 v92, 1.0, v92
v_add_f32_e32 v121, 1.0, v121
v_add_f32_e32 v104, 1.0, v104
v_add_f32_e32 v116, 1.0, v116
v_add_f32_e32 v93, 1.0, v93
v_add_f32_e32 v122, 1.0, v122
v_rcp_f32 v103, v103
v_rcp_f32 v115, v115
v_rcp_f32 v92, v92
v_rcp_f32 v121, v121
v_rcp_f32 v104, v104
v_rcp_f32 v116, v116
v_rcp_f32 v93, v93
v_rcp_f32 v122, v122
v_cvt_f16_f32 v78, v103
v_cvt_f16_f32 v79, v115
v_cvt_f16_f32 v82, v92
v_cvt_f16_f32 v83, v121
v_cvt_f16_f32 v78.h, v104
v_cvt_f16_f32 v79.h, v116
v_cvt_f16_f32 v82.h, v93
v_cvt_f16_f32 v83.h, v122
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_cmp_ge_u32 0, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 1, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 8, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 9, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 4
v_readlane_b32 s98, v31, 6
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v104, 0xbfb8aa3b
v_mov_b32 v116, 0xbfb8aa3b
v_mov_b32 v93, 0xbfb8aa3b
v_mov_b32 v122, 0xbfb8aa3b
v_fma_mix_f32 v103, v104, v70, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v115, v116, v71, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v92, v93, v74, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v121, v122, v75, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v104, v104, v70, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v116, v116, v71, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v93, v93, v74, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v122, v122, v75, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v103, v103
v_exp_f32_e32 v115, v115
v_exp_f32_e32 v92, v92
v_exp_f32_e32 v121, v121
v_exp_f32_e32 v104, v104
v_exp_f32_e32 v116, v116
v_exp_f32_e32 v93, v93
v_exp_f32_e32 v122, v122
v_add_f32_e32 v103, 1.0, v103
v_add_f32_e32 v115, 1.0, v115
v_add_f32_e32 v92, 1.0, v92
v_add_f32_e32 v121, 1.0, v121
v_add_f32_e32 v104, 1.0, v104
v_add_f32_e32 v116, 1.0, v116
v_add_f32_e32 v93, 1.0, v93
v_add_f32_e32 v122, 1.0, v122
v_rcp_f32 v103, v103
v_rcp_f32 v115, v115
v_rcp_f32 v92, v92
v_rcp_f32 v121, v121
v_rcp_f32 v104, v104
v_rcp_f32 v116, v116
v_rcp_f32 v93, v93
v_rcp_f32 v122, v122
v_cvt_f16_f32 v70, v103
v_cvt_f16_f32 v71, v115
v_cvt_f16_f32 v74, v92
v_cvt_f16_f32 v75, v121
v_cvt_f16_f32 v70.h, v104
v_cvt_f16_f32 v71.h, v116
v_cvt_f16_f32 v74.h, v93
v_cvt_f16_f32 v75.h, v122
ds_load_b32 v101, v2 offset:8896
ds_load_b32 v102, v2 offset:9024
ds_load_b32 v105, v2 offset:9152
ds_load_b32 v106, v2 offset:9280
ds_load_b32 v109, v2 offset:11072
ds_load_b32 v110, v2 offset:11200
ds_load_b32 v113, v2 offset:11328
ds_load_b32 v114, v2 offset:11456
s_cmp_ge_u32 2, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 3, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 10, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 11, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 5, s85
s_mul_hi_i32 s101, 5, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 5
v_readlane_b32 s98, v31, 7
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v104, 0xbfb8aa3b
v_mov_b32 v116, 0xbfb8aa3b
v_mov_b32 v93, 0xbfb8aa3b
v_mov_b32 v122, 0xbfb8aa3b
v_fma_mix_f32 v103, v104, v78, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v115, v116, v79, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v92, v93, v82, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v121, v122, v83, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v104, v104, v78, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v116, v116, v79, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v93, v93, v82, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v122, v122, v83, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v103, v103
v_exp_f32_e32 v115, v115
v_exp_f32_e32 v92, v92
v_exp_f32_e32 v121, v121
v_exp_f32_e32 v104, v104
v_exp_f32_e32 v116, v116
v_exp_f32_e32 v93, v93
v_exp_f32_e32 v122, v122
v_add_f32_e32 v103, 1.0, v103
v_add_f32_e32 v115, 1.0, v115
v_add_f32_e32 v92, 1.0, v92
v_add_f32_e32 v121, 1.0, v121
v_add_f32_e32 v104, 1.0, v104
v_add_f32_e32 v116, 1.0, v116
v_add_f32_e32 v93, 1.0, v93
v_add_f32_e32 v122, 1.0, v122
v_rcp_f32 v103, v103
v_rcp_f32 v115, v115
v_rcp_f32 v92, v92
v_rcp_f32 v121, v121
v_rcp_f32 v104, v104
v_rcp_f32 v116, v116
v_rcp_f32 v93, v93
v_rcp_f32 v122, v122
v_cvt_f16_f32 v78, v103
v_cvt_f16_f32 v79, v115
v_cvt_f16_f32 v82, v92
v_cvt_f16_f32 v83, v121
v_cvt_f16_f32 v78.h, v104
v_cvt_f16_f32 v79.h, v116
v_cvt_f16_f32 v82.h, v93
v_cvt_f16_f32 v83.h, v122
s_cmp_ge_u32 16, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 17, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 24, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 25, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_cmp_ge_u32 18, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 19, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 26, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 27, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 0, s85
s_mul_hi_i32 s101, 0, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_setprio 1
s_branch 3921
s_nop 0
v_add_f32_e32 v35, v35, v68
v_add_f32_e32 v36, v36, v69
v_add_f32 v134, v134, neg(v101)
v_add_f32 v135, v135, neg(v102)
v_add_f32_e32 v35, v35, v101
v_add_f32_e32 v36, v36, v102
v_add_f32_e32 v134, v134, v68
v_add_f32_e32 v135, v135, v69
v_add_f32_e32 v39, v39, v72
v_add_f32_e32 v40, v40, v73
v_add_f32 v138, v138, neg(v105)
v_add_f32 v139, v139, neg(v106)
v_add_f32_e32 v39, v39, v105
v_add_f32_e32 v40, v40, v106
v_add_f32_e32 v138, v138, v72
v_add_f32_e32 v139, v139, v73
v_add_f32_e32 v43, v43, v76
v_add_f32_e32 v44, v44, v77
v_add_f32 v142, v142, neg(v109)
v_add_f32 v143, v143, neg(v110)
v_add_f32_e32 v43, v43, v109
v_add_f32_e32 v44, v44, v110
v_add_f32_e32 v142, v142, v76
v_add_f32_e32 v143, v143, v77
v_add_f32_e32 v47, v47, v80
v_add_f32_e32 v48, v48, v81
v_add_f32 v146, v146, neg(v113)
v_add_f32 v147, v147, neg(v114)
v_add_f32_e32 v47, v47, v113
v_add_f32_e32 v48, v48, v114
v_add_f32_e32 v146, v146, v80
v_add_f32_e32 v147, v147, v81
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_store_b32 v1, v35 offset:4416
ds_store_b32 v1, v39 offset:4480
ds_store_b32 v1, v43 offset:4928
ds_store_b32 v1, v47 offset:4992
ds_store_b32 v1, v134 offset:4
ds_store_b32 v1, v138 offset:68
ds_store_b32 v1, v142 offset:516
ds_store_b32 v1, v146 offset:580
v_add_f32_e32 v37, v37, v70
v_add_f32_e32 v38, v38, v71
v_add_f32 v136, v136, neg(v103)
v_add_f32 v137, v137, neg(v104)
v_add_f32_e32 v37, v37, v103
v_add_f32_e32 v38, v38, v104
v_add_f32_e32 v136, v136, v70
v_add_f32_e32 v137, v137, v71
v_add_f32_e32 v41, v41, v74
v_add_f32_e32 v42, v42, v75
v_add_f32 v140, v140, neg(v107)
v_add_f32 v141, v141, neg(v108)
v_add_f32_e32 v41, v41, v107
v_add_f32_e32 v42, v42, v108
v_add_f32_e32 v140, v140, v74
v_add_f32_e32 v141, v141, v75
v_add_f32_e32 v45, v45, v78
v_add_f32_e32 v46, v46, v79
v_add_f32 v144, v144, neg(v111)
v_add_f32 v145, v145, neg(v112)
v_add_f32_e32 v45, v45, v111
v_add_f32_e32 v46, v46, v112
v_add_f32_e32 v144, v144, v78
v_add_f32_e32 v145, v145, v79
v_add_f32_e32 v49, v49, v82
v_add_f32_e32 v50, v50, v83
v_add_f32 v148, v148, neg(v115)
v_add_f32 v149, v149, neg(v116)
v_add_f32_e32 v49, v49, v115
v_add_f32_e32 v50, v50, v116
v_add_f32_e32 v148, v148, v82
v_add_f32_e32 v149, v149, v83
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
v_add_f32_e32 v35, v51, v84
v_add_f32_e32 v36, v52, v85
v_add_f32 v134, v150, neg(v117)
v_add_f32 v135, v151, neg(v118)
v_add_f32_e32 v35, v35, v117
v_add_f32_e32 v36, v36, v118
v_add_f32_e32 v134, v134, v84
v_add_f32_e32 v135, v135, v85
v_add_f32_e32 v39, v55, v88
v_add_f32_e32 v40, v56, v89
v_add_f32 v138, v154, neg(v121)
v_add_f32 v139, v155, neg(v122)
v_add_f32_e32 v39, v39, v121
v_add_f32_e32 v40, v40, v122
v_add_f32_e32 v138, v138, v88
v_add_f32_e32 v139, v139, v89
v_add_f32_e32 v43, v59, v92
v_add_f32_e32 v44, v60, v93
v_add_f32 v142, v158, neg(v125)
v_add_f32 v143, v159, neg(v126)
v_add_f32_e32 v43, v43, v125
v_add_f32_e32 v44, v44, v126
v_add_f32_e32 v142, v142, v92
v_add_f32_e32 v143, v143, v93
v_add_f32_e32 v47, v63, v96
v_add_f32_e32 v48, v64, v97
v_add_f32 v146, v162, neg(v129)
v_add_f32 v147, v163, neg(v130)
v_add_f32_e32 v47, v47, v129
v_add_f32_e32 v48, v48, v130
v_add_f32_e32 v146, v146, v96
v_add_f32_e32 v147, v147, v97
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v101, v2 offset:8896
ds_store_b32 v1, v35 offset:4416
ds_load_b32 v102, v2 offset:9024
ds_store_b32 v1, v39 offset:4480
ds_load_b32 v105, v2 offset:9152
ds_store_b32 v1, v43 offset:4928
ds_load_b32 v106, v2 offset:9280
ds_store_b32 v1, v47 offset:4992
ds_load_b32 v109, v2 offset:11072
ds_store_b32 v1, v134 offset:4
ds_load_b32 v110, v2 offset:11200
ds_store_b32 v1, v138 offset:68
ds_load_b32 v113, v2 offset:11328
ds_store_b32 v1, v142 offset:516
ds_load_b32 v114, v2 offset:11456
ds_store_b32 v1, v146 offset:580
v_add_f32_e32 v37, v53, v86
v_add_f32_e32 v38, v54, v87
v_add_f32 v136, v152, neg(v119)
v_add_f32 v137, v153, neg(v120)
v_add_f32_e32 v37, v37, v119
v_add_f32_e32 v38, v38, v120
v_add_f32_e32 v136, v136, v86
v_add_f32_e32 v137, v137, v87
v_add_f32_e32 v41, v57, v90
v_add_f32_e32 v42, v58, v91
v_add_f32 v140, v156, neg(v123)
v_add_f32 v141, v157, neg(v124)
v_add_f32_e32 v41, v41, v123
v_add_f32_e32 v42, v42, v124
v_add_f32_e32 v140, v140, v90
v_add_f32_e32 v141, v141, v91
v_add_f32_e32 v45, v61, v94
v_add_f32_e32 v46, v62, v95
v_add_f32 v144, v160, neg(v127)
v_add_f32 v145, v161, neg(v128)
v_add_f32_e32 v45, v45, v127
v_add_f32_e32 v46, v46, v128
v_add_f32_e32 v144, v144, v94
v_add_f32_e32 v145, v145, v95
v_add_f32_e32 v49, v65, v98
v_add_f32_e32 v50, v66, v99
v_add_f32 v148, v164, neg(v131)
v_add_f32 v149, v165, neg(v132)
v_add_f32_e32 v49, v49, v131
v_add_f32_e32 v50, v50, v132
v_add_f32_e32 v148, v148, v98
v_add_f32_e32 v149, v149, v99
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 0
v_readlane_b32 s98, v31, 2
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v104, 0xbfb8aa3b
v_mov_b32 v116, 0xbfb8aa3b
v_mov_b32 v93, 0xbfb8aa3b
v_mov_b32 v122, 0xbfb8aa3b
v_fma_mix_f32 v103, v104, v70, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v115, v116, v71, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v92, v93, v74, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v121, v122, v75, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v104, v104, v70, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v116, v116, v71, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v93, v93, v74, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v122, v122, v75, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v103, v103
v_exp_f32_e32 v115, v115
v_exp_f32_e32 v92, v92
v_exp_f32_e32 v121, v121
v_exp_f32_e32 v104, v104
v_exp_f32_e32 v116, v116
v_exp_f32_e32 v93, v93
v_exp_f32_e32 v122, v122
v_add_f32_e32 v103, 1.0, v103
v_add_f32_e32 v115, 1.0, v115
v_add_f32_e32 v92, 1.0, v92
v_add_f32_e32 v121, 1.0, v121
v_add_f32_e32 v104, 1.0, v104
v_add_f32_e32 v116, 1.0, v116
v_add_f32_e32 v93, 1.0, v93
v_add_f32_e32 v122, 1.0, v122
v_rcp_f32 v103, v103
v_rcp_f32 v115, v115
v_rcp_f32 v92, v92
v_rcp_f32 v121, v121
v_rcp_f32 v104, v104
v_rcp_f32 v116, v116
v_rcp_f32 v93, v93
v_rcp_f32 v122, v122
v_cvt_f16_f32 v70, v103
v_cvt_f16_f32 v71, v115
v_cvt_f16_f32 v74, v92
v_cvt_f16_f32 v75, v121
v_cvt_f16_f32 v70.h, v104
v_cvt_f16_f32 v71.h, v116
v_cvt_f16_f32 v74.h, v93
v_cvt_f16_f32 v75.h, v122
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_cmp_ge_u32 0, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 1, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 8, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 9, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 1
v_readlane_b32 s98, v31, 3
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v104, 0xbfb8aa3b
v_mov_b32 v116, 0xbfb8aa3b
v_mov_b32 v93, 0xbfb8aa3b
v_mov_b32 v122, 0xbfb8aa3b
v_fma_mix_f32 v103, v104, v78, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v115, v116, v79, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v92, v93, v82, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v121, v122, v83, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v104, v104, v78, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v116, v116, v79, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v93, v93, v82, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v122, v122, v83, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v103, v103
v_exp_f32_e32 v115, v115
v_exp_f32_e32 v92, v92
v_exp_f32_e32 v121, v121
v_exp_f32_e32 v104, v104
v_exp_f32_e32 v116, v116
v_exp_f32_e32 v93, v93
v_exp_f32_e32 v122, v122
v_add_f32_e32 v103, 1.0, v103
v_add_f32_e32 v115, 1.0, v115
v_add_f32_e32 v92, 1.0, v92
v_add_f32_e32 v121, 1.0, v121
v_add_f32_e32 v104, 1.0, v104
v_add_f32_e32 v116, 1.0, v116
v_add_f32_e32 v93, 1.0, v93
v_add_f32_e32 v122, 1.0, v122
v_rcp_f32 v103, v103
v_rcp_f32 v115, v115
v_rcp_f32 v92, v92
v_rcp_f32 v121, v121
v_rcp_f32 v104, v104
v_rcp_f32 v116, v116
v_rcp_f32 v93, v93
v_rcp_f32 v122, v122
v_cvt_f16_f32 v78, v103
v_cvt_f16_f32 v79, v115
v_cvt_f16_f32 v82, v92
v_cvt_f16_f32 v83, v121
v_cvt_f16_f32 v78.h, v104
v_cvt_f16_f32 v79.h, v116
v_cvt_f16_f32 v82.h, v93
v_cvt_f16_f32 v83.h, v122
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v101, v2 offset:8896
ds_load_b32 v102, v2 offset:9024
ds_load_b32 v105, v2 offset:9152
ds_load_b32 v106, v2 offset:9280
ds_load_b32 v109, v2 offset:11072
ds_load_b32 v110, v2 offset:11200
ds_load_b32 v113, v2 offset:11328
ds_load_b32 v114, v2 offset:11456
s_cmp_ge_u32 2, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 3, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 10, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 11, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 5, s85
s_mul_hi_i32 s101, 5, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 4
v_readlane_b32 s98, v31, 6
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v104, 0xbfb8aa3b
v_mov_b32 v116, 0xbfb8aa3b
v_mov_b32 v93, 0xbfb8aa3b
v_mov_b32 v122, 0xbfb8aa3b
v_fma_mix_f32 v103, v104, v70, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v115, v116, v71, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v92, v93, v74, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v121, v122, v75, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v104, v104, v70, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v116, v116, v71, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v93, v93, v74, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v122, v122, v75, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v103, v103
v_exp_f32_e32 v115, v115
v_exp_f32_e32 v92, v92
v_exp_f32_e32 v121, v121
v_exp_f32_e32 v104, v104
v_exp_f32_e32 v116, v116
v_exp_f32_e32 v93, v93
v_exp_f32_e32 v122, v122
v_add_f32_e32 v103, 1.0, v103
v_add_f32_e32 v115, 1.0, v115
v_add_f32_e32 v92, 1.0, v92
v_add_f32_e32 v121, 1.0, v121
v_add_f32_e32 v104, 1.0, v104
v_add_f32_e32 v116, 1.0, v116
v_add_f32_e32 v93, 1.0, v93
v_add_f32_e32 v122, 1.0, v122
v_rcp_f32 v103, v103
v_rcp_f32 v115, v115
v_rcp_f32 v92, v92
v_rcp_f32 v121, v121
v_rcp_f32 v104, v104
v_rcp_f32 v116, v116
v_rcp_f32 v93, v93
v_rcp_f32 v122, v122
v_cvt_f16_f32 v70, v103
v_cvt_f16_f32 v71, v115
v_cvt_f16_f32 v74, v92
v_cvt_f16_f32 v75, v121
v_cvt_f16_f32 v70.h, v104
v_cvt_f16_f32 v71.h, v116
v_cvt_f16_f32 v74.h, v93
v_cvt_f16_f32 v75.h, v122
s_waitcnt lgkmcnt(0)
s_barrier
s_cmp_ge_u32 16, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 17, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 24, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 25, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 5
v_readlane_b32 s98, v31, 7
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
v_mov_b32 v104, 0xbfb8aa3b
v_mov_b32 v116, 0xbfb8aa3b
v_mov_b32 v93, 0xbfb8aa3b
v_mov_b32 v122, 0xbfb8aa3b
v_fma_mix_f32 v103, v104, v78, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v115, v116, v79, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v92, v93, v82, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v121, v122, v83, 0 op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v104, v104, v78, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v116, v116, v79, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v93, v93, v82, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v122, v122, v83, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v103, v103
v_exp_f32_e32 v115, v115
v_exp_f32_e32 v92, v92
v_exp_f32_e32 v121, v121
v_exp_f32_e32 v104, v104
v_exp_f32_e32 v116, v116
v_exp_f32_e32 v93, v93
v_exp_f32_e32 v122, v122
v_add_f32_e32 v103, 1.0, v103
v_add_f32_e32 v115, 1.0, v115
v_add_f32_e32 v92, 1.0, v92
v_add_f32_e32 v121, 1.0, v121
v_add_f32_e32 v104, 1.0, v104
v_add_f32_e32 v116, 1.0, v116
v_add_f32_e32 v93, 1.0, v93
v_add_f32_e32 v122, 1.0, v122
v_rcp_f32 v103, v103
v_rcp_f32 v115, v115
v_rcp_f32 v92, v92
v_rcp_f32 v121, v121
v_rcp_f32 v104, v104
v_rcp_f32 v116, v116
v_rcp_f32 v93, v93
v_rcp_f32 v122, v122
v_cvt_f16_f32 v78, v103
v_cvt_f16_f32 v79, v115
v_cvt_f16_f32 v82, v92
v_cvt_f16_f32 v83, v121
v_cvt_f16_f32 v78.h, v104
v_cvt_f16_f32 v79.h, v116
v_cvt_f16_f32 v82.h, v93
v_cvt_f16_f32 v83.h, v122
s_waitcnt lgkmcnt(0)
s_cmp_ge_u32 18, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 19, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 26, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 27, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 0, s85
s_mul_hi_i32 s101, 0, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_setprio 2
s_branch 2913
s_nop 0
v_add_f32_e32 v35, v35, v68
v_add_f32_e32 v36, v36, v69
v_add_f32 v134, v134, neg(v101)
v_add_f32 v135, v135, neg(v102)
v_add_f32_e32 v35, v35, v101
v_add_f32_e32 v36, v36, v102
v_add_f32_e32 v134, v134, v68
v_add_f32_e32 v135, v135, v69
v_add_f32_e32 v39, v39, v72
v_add_f32_e32 v40, v40, v73
v_add_f32 v138, v138, neg(v105)
v_add_f32 v139, v139, neg(v106)
v_add_f32_e32 v39, v39, v105
v_add_f32_e32 v40, v40, v106
v_add_f32_e32 v138, v138, v72
v_add_f32_e32 v139, v139, v73
v_add_f32_e32 v43, v43, v76
v_add_f32_e32 v44, v44, v77
v_add_f32 v142, v142, neg(v109)
v_add_f32 v143, v143, neg(v110)
v_add_f32_e32 v43, v43, v109
v_add_f32_e32 v44, v44, v110
v_add_f32_e32 v142, v142, v76
v_add_f32_e32 v143, v143, v77
v_add_f32_e32 v47, v47, v80
v_add_f32_e32 v48, v48, v81
v_add_f32 v146, v146, neg(v113)
v_add_f32 v147, v147, neg(v114)
v_add_f32_e32 v47, v47, v113
v_add_f32_e32 v48, v48, v114
v_add_f32_e32 v146, v146, v80
v_add_f32_e32 v147, v147, v81
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v37, v37, v70
v_add_f32_e32 v38, v38, v71
v_add_f32 v136, v136, neg(v103)
v_add_f32 v137, v137, neg(v104)
v_add_f32_e32 v37, v37, v103
v_add_f32_e32 v38, v38, v104
v_add_f32_e32 v136, v136, v70
v_add_f32_e32 v137, v137, v71
v_add_f32_e32 v41, v41, v74
v_add_f32_e32 v42, v42, v75
v_add_f32 v140, v140, neg(v107)
v_add_f32 v141, v141, neg(v108)
v_add_f32_e32 v41, v41, v107
v_add_f32_e32 v42, v42, v108
v_add_f32_e32 v140, v140, v74
v_add_f32_e32 v141, v141, v75
v_add_f32_e32 v45, v45, v78
v_add_f32_e32 v46, v46, v79
v_add_f32 v144, v144, neg(v111)
v_add_f32 v145, v145, neg(v112)
v_add_f32_e32 v45, v45, v111
v_add_f32_e32 v46, v46, v112
v_add_f32_e32 v144, v144, v78
v_add_f32_e32 v145, v145, v79
v_add_f32_e32 v49, v49, v82
v_add_f32_e32 v50, v50, v83
v_add_f32 v148, v148, neg(v115)
v_add_f32 v149, v149, neg(v116)
v_add_f32_e32 v49, v49, v115
v_add_f32_e32 v50, v50, v116
v_add_f32_e32 v148, v148, v82
v_add_f32_e32 v149, v149, v83
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
ds_store_b32 v1, v35 offset:4416
ds_store_b32 v1, v39 offset:4480
ds_store_b32 v1, v43 offset:4928
ds_store_b32 v1, v47 offset:4992
ds_store_b32 v1, v134 offset:4
ds_store_b32 v1, v138 offset:68
ds_store_b32 v1, v142 offset:516
ds_store_b32 v1, v146 offset:580
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v35, v51, v84
v_add_f32_e32 v36, v52, v85
v_add_f32 v134, v150, neg(v117)
v_add_f32 v135, v151, neg(v118)
v_add_f32_e32 v35, v35, v117
v_add_f32_e32 v36, v36, v118
v_add_f32_e32 v134, v134, v84
v_add_f32_e32 v135, v135, v85
v_add_f32_e32 v39, v55, v88
v_add_f32_e32 v40, v56, v89
v_add_f32 v138, v154, neg(v121)
v_add_f32 v139, v155, neg(v122)
v_add_f32_e32 v39, v39, v121
v_add_f32_e32 v40, v40, v122
v_add_f32_e32 v138, v138, v88
v_add_f32_e32 v139, v139, v89
v_add_f32_e32 v43, v59, v92
v_add_f32_e32 v44, v60, v93
v_add_f32 v142, v158, neg(v125)
v_add_f32 v143, v159, neg(v126)
v_add_f32_e32 v43, v43, v125
v_add_f32_e32 v44, v44, v126
v_add_f32_e32 v142, v142, v92
v_add_f32_e32 v143, v143, v93
v_add_f32_e32 v47, v63, v96
v_add_f32_e32 v48, v64, v97
v_add_f32 v146, v162, neg(v129)
v_add_f32 v147, v163, neg(v130)
v_add_f32_e32 v47, v47, v129
v_add_f32_e32 v48, v48, v130
v_add_f32_e32 v146, v146, v96
v_add_f32_e32 v147, v147, v97
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_waitcnt lgkmcnt(0)
s_barrier
v_add_f32_e32 v37, v53, v86
v_add_f32_e32 v38, v54, v87
v_add_f32 v136, v152, neg(v119)
v_add_f32 v137, v153, neg(v120)
v_add_f32_e32 v37, v37, v119
v_add_f32_e32 v38, v38, v120
v_add_f32_e32 v136, v136, v86
v_add_f32_e32 v137, v137, v87
v_add_f32_e32 v41, v57, v90
v_add_f32_e32 v42, v58, v91
v_add_f32 v140, v156, neg(v123)
v_add_f32 v141, v157, neg(v124)
v_add_f32_e32 v41, v41, v123
v_add_f32_e32 v42, v42, v124
v_add_f32_e32 v140, v140, v90
v_add_f32_e32 v141, v141, v91
v_add_f32_e32 v45, v61, v94
v_add_f32_e32 v46, v62, v95
v_add_f32 v144, v160, neg(v127)
v_add_f32 v145, v161, neg(v128)
v_add_f32_e32 v45, v45, v127
v_add_f32_e32 v46, v46, v128
v_add_f32_e32 v144, v144, v94
v_add_f32_e32 v145, v145, v95
v_add_f32_e32 v49, v65, v98
v_add_f32_e32 v50, v66, v99
v_add_f32 v148, v164, neg(v131)
v_add_f32 v149, v165, neg(v132)
v_add_f32_e32 v49, v49, v131
v_add_f32_e32 v50, v50, v132
v_add_f32_e32 v148, v148, v98
v_add_f32_e32 v149, v149, v99
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 0
v_readlane_b32 s98, v31, 2
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 70, 326, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 71, 327, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 74, 330, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 75, 331, 41, 0x0, 0x3, 0x0, 0x0
v_and_b32 v103, 0x7fff7fff, v70
v_and_b32 v115, 0x7fff7fff, v71
v_and_b32 v92, 0x7fff7fff, v74
v_and_b32 v121, 0x7fff7fff, v75
v_mov_b32 v104, 0xb62eb62e
v_mov_b32 v116, 0xb62eb62e
v_mov_b32 v93, 0xb62eb62e
v_mov_b32 v122, 0xb62eb62e
v_pk_fma_f16 v104, v103, 0x30883088, v104
v_pk_fma_f16 v116, v115, 0x30883088, v116
v_pk_fma_f16 v93, v92, 0x30883088, v93
v_pk_fma_f16 v122, v121, 0x30883088, v122
v_pk_fma_f16 v104, v103, v104, 0x1e331e33
v_pk_fma_f16 v116, v115, v116, 0x1e331e33
v_pk_fma_f16 v93, v92, v93, 0x1e331e33
v_pk_fma_f16 v122, v121, v122, 0x1e331e33
_v_pk_mul_f16__vop3p 104, 359, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 116, 371, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 348, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 122, 377, 378, 0x0, 0x3, 0x0, 0x0
v_pk_fma_f16 v104, v103, v104, v103 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v116, v115, v116, v115 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v93, v92, v93, v92 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v122, v121, v122, v121 op_sel:[0,0,0] op_sel_hi:[1,1,1]
_v_pk_mul_f16__vop3p_lit 103, 0x41c541c5, 359, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 115, 0x41c541c5, 371, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 92, 0x41c541c5, 348, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 121, 0x41c541c5, 377, 0x0, 0x3
v_exp_f16 v103, v103
v_exp_f16 v115, v115
v_exp_f16 v92, v92
v_exp_f16 v121, v121
_v_exp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 103, 242, 359, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 115, 242, 371, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 242, 348, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 121, 242, 377, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v103, v103
v_rcp_f16 v115, v115
v_rcp_f16 v92, v92
v_rcp_f16 v121, v121
_v_rcp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
v_mov_b32 v107, 0x3bc23bc2
v_mov_b32 v84, 0x3bc23bc2
v_mov_b32 v96, 0x3bc23bc2
v_mov_b32 v125, 0x3bc23bc2
v_pk_fma_f16 v103, v103, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v115, v115, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v92, v92, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v121, v121, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_cmp_gt_f16 vcc, |v70|, v107
v_cndmask_b16 v104, v104, v103, vcc
v_cmp_gt_f16 vcc, |v71|, v84
v_cndmask_b16 v116, v116, v115, vcc
v_cmp_gt_f16 vcc, |v74|, v96
v_cndmask_b16 v93, v93, v92, vcc
v_cmp_gt_f16 vcc, |v75|, v125
v_cndmask_b16 v122, v122, v121, vcc
_v_cmp_gt_f16_vccahvhv 70, 107
_v_cndmask_b16__vop3 104, 104, 103, 106, 0xb
_v_cmp_gt_f16_vccahvhv 71, 84
_v_cndmask_b16__vop3 116, 116, 115, 106, 0xb
_v_cmp_gt_f16_vccahvhv 74, 96
_v_cndmask_b16__vop3 93, 93, 92, 106, 0xb
_v_cmp_gt_f16_vccahvhv 75, 125
_v_cndmask_b16__vop3 122, 122, 121, 106, 0xb
v_bfi_b32 v70, 0x7fff7fff, v104, v70
v_bfi_b32 v71, 0x7fff7fff, v116, v71
v_bfi_b32 v74, 0x7fff7fff, v93, v74
v_bfi_b32 v75, 0x7fff7fff, v122, v75
_v_pk_mul_f16__vop3p 70, 326, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 71, 327, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 74, 330, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 75, 331, 40, 0x0, 0x3, 0x0, 0x0
ds_load_b32 v101, v2 offset:8896
ds_store_b32 v1, v35 offset:4416
ds_load_b32 v102, v2 offset:9024
ds_store_b32 v1, v39 offset:4480
ds_load_b32 v105, v2 offset:9152
ds_store_b32 v1, v43 offset:4928
ds_load_b32 v106, v2 offset:9280
ds_store_b32 v1, v47 offset:4992
ds_load_b32 v109, v2 offset:11072
ds_store_b32 v1, v134 offset:4
ds_load_b32 v110, v2 offset:11200
ds_store_b32 v1, v138 offset:68
ds_load_b32 v113, v2 offset:11328
ds_store_b32 v1, v142 offset:516
ds_load_b32 v114, v2 offset:11456
ds_store_b32 v1, v146 offset:580
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 1
v_readlane_b32 s98, v31, 3
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 78, 334, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 79, 335, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 82, 338, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 83, 339, 41, 0x0, 0x3, 0x0, 0x0
v_and_b32 v103, 0x7fff7fff, v78
v_and_b32 v115, 0x7fff7fff, v79
v_and_b32 v92, 0x7fff7fff, v82
v_and_b32 v121, 0x7fff7fff, v83
v_mov_b32 v104, 0xb62eb62e
v_mov_b32 v116, 0xb62eb62e
v_mov_b32 v93, 0xb62eb62e
v_mov_b32 v122, 0xb62eb62e
v_pk_fma_f16 v104, v103, 0x30883088, v104
v_pk_fma_f16 v116, v115, 0x30883088, v116
v_pk_fma_f16 v93, v92, 0x30883088, v93
v_pk_fma_f16 v122, v121, 0x30883088, v122
v_pk_fma_f16 v104, v103, v104, 0x1e331e33
v_pk_fma_f16 v116, v115, v116, 0x1e331e33
v_pk_fma_f16 v93, v92, v93, 0x1e331e33
v_pk_fma_f16 v122, v121, v122, 0x1e331e33
_v_pk_mul_f16__vop3p 104, 359, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 116, 371, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 348, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 122, 377, 378, 0x0, 0x3, 0x0, 0x0
v_pk_fma_f16 v104, v103, v104, v103 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v116, v115, v116, v115 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v93, v92, v93, v92 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v122, v121, v122, v121 op_sel:[0,0,0] op_sel_hi:[1,1,1]
_v_pk_mul_f16__vop3p_lit 103, 0x41c541c5, 359, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 115, 0x41c541c5, 371, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 92, 0x41c541c5, 348, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 121, 0x41c541c5, 377, 0x0, 0x3
v_exp_f16 v103, v103
v_exp_f16 v115, v115
v_exp_f16 v92, v92
v_exp_f16 v121, v121
_v_exp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 103, 242, 359, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 115, 242, 371, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 242, 348, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 121, 242, 377, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v103, v103
v_rcp_f16 v115, v115
v_rcp_f16 v92, v92
v_rcp_f16 v121, v121
_v_rcp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
v_mov_b32 v107, 0x3bc23bc2
v_mov_b32 v84, 0x3bc23bc2
v_mov_b32 v96, 0x3bc23bc2
v_mov_b32 v125, 0x3bc23bc2
v_pk_fma_f16 v103, v103, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v115, v115, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v92, v92, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v121, v121, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_cmp_gt_f16 vcc, |v78|, v107
v_cndmask_b16 v104, v104, v103, vcc
v_cmp_gt_f16 vcc, |v79|, v84
v_cndmask_b16 v116, v116, v115, vcc
v_cmp_gt_f16 vcc, |v82|, v96
v_cndmask_b16 v93, v93, v92, vcc
v_cmp_gt_f16 vcc, |v83|, v125
v_cndmask_b16 v122, v122, v121, vcc
_v_cmp_gt_f16_vccahvhv 78, 107
_v_cndmask_b16__vop3 104, 104, 103, 106, 0xb
_v_cmp_gt_f16_vccahvhv 79, 84
_v_cndmask_b16__vop3 116, 116, 115, 106, 0xb
_v_cmp_gt_f16_vccahvhv 82, 96
_v_cndmask_b16__vop3 93, 93, 92, 106, 0xb
_v_cmp_gt_f16_vccahvhv 83, 125
_v_cndmask_b16__vop3 122, 122, 121, 106, 0xb
v_bfi_b32 v78, 0x7fff7fff, v104, v78
v_bfi_b32 v79, 0x7fff7fff, v116, v79
v_bfi_b32 v82, 0x7fff7fff, v93, v82
v_bfi_b32 v83, 0x7fff7fff, v122, v83
_v_pk_mul_f16__vop3p 78, 334, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 79, 335, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 82, 338, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 83, 339, 40, 0x0, 0x3, 0x0, 0x0
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_cmp_ge_u32 0, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 1, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 8, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 9, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 4
v_readlane_b32 s98, v31, 6
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 70, 326, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 71, 327, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 74, 330, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 75, 331, 41, 0x0, 0x3, 0x0, 0x0
v_and_b32 v103, 0x7fff7fff, v70
v_and_b32 v115, 0x7fff7fff, v71
v_and_b32 v92, 0x7fff7fff, v74
v_and_b32 v121, 0x7fff7fff, v75
v_mov_b32 v104, 0xb62eb62e
v_mov_b32 v116, 0xb62eb62e
v_mov_b32 v93, 0xb62eb62e
v_mov_b32 v122, 0xb62eb62e
v_pk_fma_f16 v104, v103, 0x30883088, v104
v_pk_fma_f16 v116, v115, 0x30883088, v116
v_pk_fma_f16 v93, v92, 0x30883088, v93
v_pk_fma_f16 v122, v121, 0x30883088, v122
v_pk_fma_f16 v104, v103, v104, 0x1e331e33
v_pk_fma_f16 v116, v115, v116, 0x1e331e33
v_pk_fma_f16 v93, v92, v93, 0x1e331e33
v_pk_fma_f16 v122, v121, v122, 0x1e331e33
_v_pk_mul_f16__vop3p 104, 359, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 116, 371, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 348, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 122, 377, 378, 0x0, 0x3, 0x0, 0x0
v_pk_fma_f16 v104, v103, v104, v103 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v116, v115, v116, v115 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v93, v92, v93, v92 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v122, v121, v122, v121 op_sel:[0,0,0] op_sel_hi:[1,1,1]
_v_pk_mul_f16__vop3p_lit 103, 0x41c541c5, 359, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 115, 0x41c541c5, 371, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 92, 0x41c541c5, 348, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 121, 0x41c541c5, 377, 0x0, 0x3
v_exp_f16 v103, v103
v_exp_f16 v115, v115
v_exp_f16 v92, v92
v_exp_f16 v121, v121
_v_exp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 103, 242, 359, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 115, 242, 371, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 242, 348, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 121, 242, 377, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v103, v103
v_rcp_f16 v115, v115
v_rcp_f16 v92, v92
v_rcp_f16 v121, v121
_v_rcp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
v_mov_b32 v107, 0x3bc23bc2
v_mov_b32 v84, 0x3bc23bc2
v_mov_b32 v96, 0x3bc23bc2
v_mov_b32 v125, 0x3bc23bc2
v_pk_fma_f16 v103, v103, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v115, v115, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v92, v92, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v121, v121, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_cmp_gt_f16 vcc, |v70|, v107
v_cndmask_b16 v104, v104, v103, vcc
v_cmp_gt_f16 vcc, |v71|, v84
v_cndmask_b16 v116, v116, v115, vcc
v_cmp_gt_f16 vcc, |v74|, v96
v_cndmask_b16 v93, v93, v92, vcc
v_cmp_gt_f16 vcc, |v75|, v125
v_cndmask_b16 v122, v122, v121, vcc
_v_cmp_gt_f16_vccahvhv 70, 107
_v_cndmask_b16__vop3 104, 104, 103, 106, 0xb
_v_cmp_gt_f16_vccahvhv 71, 84
_v_cndmask_b16__vop3 116, 116, 115, 106, 0xb
_v_cmp_gt_f16_vccahvhv 74, 96
_v_cndmask_b16__vop3 93, 93, 92, 106, 0xb
_v_cmp_gt_f16_vccahvhv 75, 125
_v_cndmask_b16__vop3 122, 122, 121, 106, 0xb
v_bfi_b32 v70, 0x7fff7fff, v104, v70
v_bfi_b32 v71, 0x7fff7fff, v116, v71
v_bfi_b32 v74, 0x7fff7fff, v93, v74
v_bfi_b32 v75, 0x7fff7fff, v122, v75
_v_pk_mul_f16__vop3p 70, 326, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 71, 327, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 74, 330, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 75, 331, 40, 0x0, 0x3, 0x0, 0x0
ds_load_b32 v101, v2 offset:8896
ds_load_b32 v102, v2 offset:9024
ds_load_b32 v105, v2 offset:9152
ds_load_b32 v106, v2 offset:9280
ds_load_b32 v109, v2 offset:11072
ds_load_b32 v110, v2 offset:11200
ds_load_b32 v113, v2 offset:11328
ds_load_b32 v114, v2 offset:11456
s_cmp_ge_u32 2, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 3, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 10, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 11, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 5, s85
s_mul_hi_i32 s101, 5, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_barrier
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 5
v_readlane_b32 s98, v31, 7
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 78, 334, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 79, 335, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 82, 338, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 83, 339, 41, 0x0, 0x3, 0x0, 0x0
v_and_b32 v103, 0x7fff7fff, v78
v_and_b32 v115, 0x7fff7fff, v79
v_and_b32 v92, 0x7fff7fff, v82
v_and_b32 v121, 0x7fff7fff, v83
v_mov_b32 v104, 0xb62eb62e
v_mov_b32 v116, 0xb62eb62e
v_mov_b32 v93, 0xb62eb62e
v_mov_b32 v122, 0xb62eb62e
v_pk_fma_f16 v104, v103, 0x30883088, v104
v_pk_fma_f16 v116, v115, 0x30883088, v116
v_pk_fma_f16 v93, v92, 0x30883088, v93
v_pk_fma_f16 v122, v121, 0x30883088, v122
v_pk_fma_f16 v104, v103, v104, 0x1e331e33
v_pk_fma_f16 v116, v115, v116, 0x1e331e33
v_pk_fma_f16 v93, v92, v93, 0x1e331e33
v_pk_fma_f16 v122, v121, v122, 0x1e331e33
_v_pk_mul_f16__vop3p 104, 359, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 116, 371, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 348, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 122, 377, 378, 0x0, 0x3, 0x0, 0x0
v_pk_fma_f16 v104, v103, v104, v103 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v116, v115, v116, v115 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v93, v92, v93, v92 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v122, v121, v122, v121 op_sel:[0,0,0] op_sel_hi:[1,1,1]
_v_pk_mul_f16__vop3p_lit 103, 0x41c541c5, 359, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 115, 0x41c541c5, 371, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 92, 0x41c541c5, 348, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 121, 0x41c541c5, 377, 0x0, 0x3
v_exp_f16 v103, v103
v_exp_f16 v115, v115
v_exp_f16 v92, v92
v_exp_f16 v121, v121
_v_exp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 103, 242, 359, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 115, 242, 371, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 242, 348, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 121, 242, 377, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v103, v103
v_rcp_f16 v115, v115
v_rcp_f16 v92, v92
v_rcp_f16 v121, v121
_v_rcp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
v_mov_b32 v107, 0x3bc23bc2
v_mov_b32 v84, 0x3bc23bc2
v_mov_b32 v96, 0x3bc23bc2
v_mov_b32 v125, 0x3bc23bc2
v_pk_fma_f16 v103, v103, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v115, v115, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v92, v92, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v121, v121, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_cmp_gt_f16 vcc, |v78|, v107
v_cndmask_b16 v104, v104, v103, vcc
v_cmp_gt_f16 vcc, |v79|, v84
v_cndmask_b16 v116, v116, v115, vcc
v_cmp_gt_f16 vcc, |v82|, v96
v_cndmask_b16 v93, v93, v92, vcc
v_cmp_gt_f16 vcc, |v83|, v125
v_cndmask_b16 v122, v122, v121, vcc
_v_cmp_gt_f16_vccahvhv 78, 107
_v_cndmask_b16__vop3 104, 104, 103, 106, 0xb
_v_cmp_gt_f16_vccahvhv 79, 84
_v_cndmask_b16__vop3 116, 116, 115, 106, 0xb
_v_cmp_gt_f16_vccahvhv 82, 96
_v_cndmask_b16__vop3 93, 93, 92, 106, 0xb
_v_cmp_gt_f16_vccahvhv 83, 125
_v_cndmask_b16__vop3 122, 122, 121, 106, 0xb
v_bfi_b32 v78, 0x7fff7fff, v104, v78
v_bfi_b32 v79, 0x7fff7fff, v116, v79
v_bfi_b32 v82, 0x7fff7fff, v93, v82
v_bfi_b32 v83, 0x7fff7fff, v122, v83
_v_pk_mul_f16__vop3p 78, 334, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 79, 335, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 82, 338, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 83, 339, 40, 0x0, 0x3, 0x0, 0x0
s_cmp_ge_u32 16, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 17, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 24, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 25, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_waitcnt lgkmcnt(0)
s_cmp_ge_u32 18, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 19, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 26, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 27, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 0, s85
s_mul_hi_i32 s101, 0, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_setprio 1
s_branch 1457
s_nop 0
v_add_f32_e32 v35, v35, v68
v_add_f32_e32 v36, v36, v69
v_add_f32 v134, v134, neg(v101)
v_add_f32 v135, v135, neg(v102)
v_add_f32_e32 v35, v35, v101
v_add_f32_e32 v36, v36, v102
v_add_f32_e32 v134, v134, v68
v_add_f32_e32 v135, v135, v69
v_add_f32_e32 v39, v39, v72
v_add_f32_e32 v40, v40, v73
v_add_f32 v138, v138, neg(v105)
v_add_f32 v139, v139, neg(v106)
v_add_f32_e32 v39, v39, v105
v_add_f32_e32 v40, v40, v106
v_add_f32_e32 v138, v138, v72
v_add_f32_e32 v139, v139, v73
v_add_f32_e32 v43, v43, v76
v_add_f32_e32 v44, v44, v77
v_add_f32 v142, v142, neg(v109)
v_add_f32 v143, v143, neg(v110)
v_add_f32_e32 v43, v43, v109
v_add_f32_e32 v44, v44, v110
v_add_f32_e32 v142, v142, v76
v_add_f32_e32 v143, v143, v77
v_add_f32_e32 v47, v47, v80
v_add_f32_e32 v48, v48, v81
v_add_f32 v146, v146, neg(v113)
v_add_f32 v147, v147, neg(v114)
v_add_f32_e32 v47, v47, v113
v_add_f32_e32 v48, v48, v114
v_add_f32_e32 v146, v146, v80
v_add_f32_e32 v147, v147, v81
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_store_b32 v1, v35 offset:4416
ds_store_b32 v1, v39 offset:4480
ds_store_b32 v1, v43 offset:4928
ds_store_b32 v1, v47 offset:4992
ds_store_b32 v1, v134 offset:4
ds_store_b32 v1, v138 offset:68
ds_store_b32 v1, v142 offset:516
ds_store_b32 v1, v146 offset:580
v_add_f32_e32 v37, v37, v70
v_add_f32_e32 v38, v38, v71
v_add_f32 v136, v136, neg(v103)
v_add_f32 v137, v137, neg(v104)
v_add_f32_e32 v37, v37, v103
v_add_f32_e32 v38, v38, v104
v_add_f32_e32 v136, v136, v70
v_add_f32_e32 v137, v137, v71
v_add_f32_e32 v41, v41, v74
v_add_f32_e32 v42, v42, v75
v_add_f32 v140, v140, neg(v107)
v_add_f32 v141, v141, neg(v108)
v_add_f32_e32 v41, v41, v107
v_add_f32_e32 v42, v42, v108
v_add_f32_e32 v140, v140, v74
v_add_f32_e32 v141, v141, v75
v_add_f32_e32 v45, v45, v78
v_add_f32_e32 v46, v46, v79
v_add_f32 v144, v144, neg(v111)
v_add_f32 v145, v145, neg(v112)
v_add_f32_e32 v45, v45, v111
v_add_f32_e32 v46, v46, v112
v_add_f32_e32 v144, v144, v78
v_add_f32_e32 v145, v145, v79
v_add_f32_e32 v49, v49, v82
v_add_f32_e32 v50, v50, v83
v_add_f32 v148, v148, neg(v115)
v_add_f32 v149, v149, neg(v116)
v_add_f32_e32 v49, v49, v115
v_add_f32_e32 v50, v50, v116
v_add_f32_e32 v148, v148, v82
v_add_f32_e32 v149, v149, v83
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
v_add_f32_e32 v35, v51, v84
v_add_f32_e32 v36, v52, v85
v_add_f32 v134, v150, neg(v117)
v_add_f32 v135, v151, neg(v118)
v_add_f32_e32 v35, v35, v117
v_add_f32_e32 v36, v36, v118
v_add_f32_e32 v134, v134, v84
v_add_f32_e32 v135, v135, v85
v_add_f32_e32 v39, v55, v88
v_add_f32_e32 v40, v56, v89
v_add_f32 v138, v154, neg(v121)
v_add_f32 v139, v155, neg(v122)
v_add_f32_e32 v39, v39, v121
v_add_f32_e32 v40, v40, v122
v_add_f32_e32 v138, v138, v88
v_add_f32_e32 v139, v139, v89
v_add_f32_e32 v43, v59, v92
v_add_f32_e32 v44, v60, v93
v_add_f32 v142, v158, neg(v125)
v_add_f32 v143, v159, neg(v126)
v_add_f32_e32 v43, v43, v125
v_add_f32_e32 v44, v44, v126
v_add_f32_e32 v142, v142, v92
v_add_f32_e32 v143, v143, v93
v_add_f32_e32 v47, v63, v96
v_add_f32_e32 v48, v64, v97
v_add_f32 v146, v162, neg(v129)
v_add_f32 v147, v163, neg(v130)
v_add_f32_e32 v47, v47, v129
v_add_f32_e32 v48, v48, v130
v_add_f32_e32 v146, v146, v96
v_add_f32_e32 v147, v147, v97
v_fma_mixlo_f16 v35, v35, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v35, v36, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v39, v39, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v39, v40, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v43, v43, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v43, v44, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v47, v47, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v47, v48, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v134, v134, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v134, v135, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v138, v138, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v138, v139, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v142, v142, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v142, v143, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v146, v146, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v146, v147, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v101, v2 offset:8896
ds_store_b32 v1, v35 offset:4416
ds_load_b32 v102, v2 offset:9024
ds_store_b32 v1, v39 offset:4480
ds_load_b32 v105, v2 offset:9152
ds_store_b32 v1, v43 offset:4928
ds_load_b32 v106, v2 offset:9280
ds_store_b32 v1, v47 offset:4992
ds_load_b32 v109, v2 offset:11072
ds_store_b32 v1, v134 offset:4
ds_load_b32 v110, v2 offset:11200
ds_store_b32 v1, v138 offset:68
ds_load_b32 v113, v2 offset:11328
ds_store_b32 v1, v142 offset:516
ds_load_b32 v114, v2 offset:11456
ds_store_b32 v1, v146 offset:580
v_add_f32_e32 v37, v53, v86
v_add_f32_e32 v38, v54, v87
v_add_f32 v136, v152, neg(v119)
v_add_f32 v137, v153, neg(v120)
v_add_f32_e32 v37, v37, v119
v_add_f32_e32 v38, v38, v120
v_add_f32_e32 v136, v136, v86
v_add_f32_e32 v137, v137, v87
v_add_f32_e32 v41, v57, v90
v_add_f32_e32 v42, v58, v91
v_add_f32 v140, v156, neg(v123)
v_add_f32 v141, v157, neg(v124)
v_add_f32_e32 v41, v41, v123
v_add_f32_e32 v42, v42, v124
v_add_f32_e32 v140, v140, v90
v_add_f32_e32 v141, v141, v91
v_add_f32_e32 v45, v61, v94
v_add_f32_e32 v46, v62, v95
v_add_f32 v144, v160, neg(v127)
v_add_f32 v145, v161, neg(v128)
v_add_f32_e32 v45, v45, v127
v_add_f32_e32 v46, v46, v128
v_add_f32_e32 v144, v144, v94
v_add_f32_e32 v145, v145, v95
v_add_f32_e32 v49, v65, v98
v_add_f32_e32 v50, v66, v99
v_add_f32 v148, v164, neg(v131)
v_add_f32 v149, v165, neg(v132)
v_add_f32_e32 v49, v49, v131
v_add_f32_e32 v50, v50, v132
v_add_f32_e32 v148, v148, v98
v_add_f32_e32 v149, v149, v99
v_fma_mixlo_f16 v37, v37, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v37, v38, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v41, v41, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v41, v42, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v45, v45, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v45, v46, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v49, v49, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v49, v50, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v136, v136, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v136, v137, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v140, v140, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v140, v141, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v144, v144, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v144, v145, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v148, v148, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v148, v149, 1.0, 0 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 0
v_readlane_b32 s98, v31, 2
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 70, 326, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 71, 327, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 74, 330, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 75, 331, 41, 0x0, 0x3, 0x0, 0x0
v_and_b32 v103, 0x7fff7fff, v70
v_and_b32 v115, 0x7fff7fff, v71
v_and_b32 v92, 0x7fff7fff, v74
v_and_b32 v121, 0x7fff7fff, v75
v_mov_b32 v104, 0xb62eb62e
v_mov_b32 v116, 0xb62eb62e
v_mov_b32 v93, 0xb62eb62e
v_mov_b32 v122, 0xb62eb62e
v_pk_fma_f16 v104, v103, 0x30883088, v104
v_pk_fma_f16 v116, v115, 0x30883088, v116
v_pk_fma_f16 v93, v92, 0x30883088, v93
v_pk_fma_f16 v122, v121, 0x30883088, v122
v_pk_fma_f16 v104, v103, v104, 0x1e331e33
v_pk_fma_f16 v116, v115, v116, 0x1e331e33
v_pk_fma_f16 v93, v92, v93, 0x1e331e33
v_pk_fma_f16 v122, v121, v122, 0x1e331e33
_v_pk_mul_f16__vop3p 104, 359, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 116, 371, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 348, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 122, 377, 378, 0x0, 0x3, 0x0, 0x0
v_pk_fma_f16 v104, v103, v104, v103 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v116, v115, v116, v115 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v93, v92, v93, v92 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v122, v121, v122, v121 op_sel:[0,0,0] op_sel_hi:[1,1,1]
_v_pk_mul_f16__vop3p_lit 103, 0x41c541c5, 359, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 115, 0x41c541c5, 371, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 92, 0x41c541c5, 348, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 121, 0x41c541c5, 377, 0x0, 0x3
v_exp_f16 v103, v103
v_exp_f16 v115, v115
v_exp_f16 v92, v92
v_exp_f16 v121, v121
_v_exp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 103, 242, 359, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 115, 242, 371, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 242, 348, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 121, 242, 377, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v103, v103
v_rcp_f16 v115, v115
v_rcp_f16 v92, v92
v_rcp_f16 v121, v121
_v_rcp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
v_mov_b32 v107, 0x3bc23bc2
v_mov_b32 v84, 0x3bc23bc2
v_mov_b32 v96, 0x3bc23bc2
v_mov_b32 v125, 0x3bc23bc2
v_pk_fma_f16 v103, v103, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v115, v115, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v92, v92, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v121, v121, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_cmp_gt_f16 vcc, |v70|, v107
v_cndmask_b16 v104, v104, v103, vcc
v_cmp_gt_f16 vcc, |v71|, v84
v_cndmask_b16 v116, v116, v115, vcc
v_cmp_gt_f16 vcc, |v74|, v96
v_cndmask_b16 v93, v93, v92, vcc
v_cmp_gt_f16 vcc, |v75|, v125
v_cndmask_b16 v122, v122, v121, vcc
_v_cmp_gt_f16_vccahvhv 70, 107
_v_cndmask_b16__vop3 104, 104, 103, 106, 0xb
_v_cmp_gt_f16_vccahvhv 71, 84
_v_cndmask_b16__vop3 116, 116, 115, 106, 0xb
_v_cmp_gt_f16_vccahvhv 74, 96
_v_cndmask_b16__vop3 93, 93, 92, 106, 0xb
_v_cmp_gt_f16_vccahvhv 75, 125
_v_cndmask_b16__vop3 122, 122, 121, 106, 0xb
v_bfi_b32 v70, 0x7fff7fff, v104, v70
v_bfi_b32 v71, 0x7fff7fff, v116, v71
v_bfi_b32 v74, 0x7fff7fff, v93, v74
v_bfi_b32 v75, 0x7fff7fff, v122, v75
_v_pk_mul_f16__vop3p 70, 326, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 71, 327, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 74, 330, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 75, 331, 40, 0x0, 0x3, 0x0, 0x0
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v68, v2
ds_store_b32 v1, v37 offset:13312
ds_load_b32 v69, v2 offset:128
ds_store_b32 v1, v41 offset:13376
ds_load_b32 v72, v2 offset:256
ds_store_b32 v1, v45 offset:13824
ds_load_b32 v73, v2 offset:384
ds_store_b32 v1, v49 offset:13888
ds_load_b32 v76, v2 offset:2176
ds_store_b32 v1, v136 offset:8900
ds_load_b32 v77, v2 offset:2304
ds_store_b32 v1, v140 offset:8964
ds_load_b32 v80, v2 offset:2432
ds_store_b32 v1, v144 offset:9412
ds_load_b32 v81, v2 offset:2560
ds_store_b32 v1, v148 offset:9476
s_cmp_ge_u32 0, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 1, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 8, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 9, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 1
v_readlane_b32 s98, v31, 3
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 78, 334, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 79, 335, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 82, 338, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 83, 339, 41, 0x0, 0x3, 0x0, 0x0
v_and_b32 v103, 0x7fff7fff, v78
v_and_b32 v115, 0x7fff7fff, v79
v_and_b32 v92, 0x7fff7fff, v82
v_and_b32 v121, 0x7fff7fff, v83
v_mov_b32 v104, 0xb62eb62e
v_mov_b32 v116, 0xb62eb62e
v_mov_b32 v93, 0xb62eb62e
v_mov_b32 v122, 0xb62eb62e
v_pk_fma_f16 v104, v103, 0x30883088, v104
v_pk_fma_f16 v116, v115, 0x30883088, v116
v_pk_fma_f16 v93, v92, 0x30883088, v93
v_pk_fma_f16 v122, v121, 0x30883088, v122
v_pk_fma_f16 v104, v103, v104, 0x1e331e33
v_pk_fma_f16 v116, v115, v116, 0x1e331e33
v_pk_fma_f16 v93, v92, v93, 0x1e331e33
v_pk_fma_f16 v122, v121, v122, 0x1e331e33
_v_pk_mul_f16__vop3p 104, 359, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 116, 371, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 348, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 122, 377, 378, 0x0, 0x3, 0x0, 0x0
v_pk_fma_f16 v104, v103, v104, v103 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v116, v115, v116, v115 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v93, v92, v93, v92 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v122, v121, v122, v121 op_sel:[0,0,0] op_sel_hi:[1,1,1]
_v_pk_mul_f16__vop3p_lit 103, 0x41c541c5, 359, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 115, 0x41c541c5, 371, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 92, 0x41c541c5, 348, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 121, 0x41c541c5, 377, 0x0, 0x3
v_exp_f16 v103, v103
v_exp_f16 v115, v115
v_exp_f16 v92, v92
v_exp_f16 v121, v121
_v_exp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 103, 242, 359, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 115, 242, 371, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 242, 348, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 121, 242, 377, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v103, v103
v_rcp_f16 v115, v115
v_rcp_f16 v92, v92
v_rcp_f16 v121, v121
_v_rcp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
v_mov_b32 v107, 0x3bc23bc2
v_mov_b32 v84, 0x3bc23bc2
v_mov_b32 v96, 0x3bc23bc2
v_mov_b32 v125, 0x3bc23bc2
v_pk_fma_f16 v103, v103, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v115, v115, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v92, v92, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v121, v121, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_cmp_gt_f16 vcc, |v78|, v107
v_cndmask_b16 v104, v104, v103, vcc
v_cmp_gt_f16 vcc, |v79|, v84
v_cndmask_b16 v116, v116, v115, vcc
v_cmp_gt_f16 vcc, |v82|, v96
v_cndmask_b16 v93, v93, v92, vcc
v_cmp_gt_f16 vcc, |v83|, v125
v_cndmask_b16 v122, v122, v121, vcc
_v_cmp_gt_f16_vccahvhv 78, 107
_v_cndmask_b16__vop3 104, 104, 103, 106, 0xb
_v_cmp_gt_f16_vccahvhv 79, 84
_v_cndmask_b16__vop3 116, 116, 115, 106, 0xb
_v_cmp_gt_f16_vccahvhv 82, 96
_v_cndmask_b16__vop3 93, 93, 92, 106, 0xb
_v_cmp_gt_f16_vccahvhv 83, 125
_v_cndmask_b16__vop3 122, 122, 121, 106, 0xb
v_bfi_b32 v78, 0x7fff7fff, v104, v78
v_bfi_b32 v79, 0x7fff7fff, v116, v79
v_bfi_b32 v82, 0x7fff7fff, v93, v82
v_bfi_b32 v83, 0x7fff7fff, v122, v83
_v_pk_mul_f16__vop3p 78, 334, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 79, 335, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 82, 338, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 83, 339, 40, 0x0, 0x3, 0x0, 0x0
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b32 v101, v2 offset:8896
ds_load_b32 v102, v2 offset:9024
ds_load_b32 v105, v2 offset:9152
ds_load_b32 v106, v2 offset:9280
ds_load_b32 v109, v2 offset:11072
ds_load_b32 v110, v2 offset:11200
ds_load_b32 v113, v2 offset:11328
ds_load_b32 v114, v2 offset:11456
s_cmp_ge_u32 2, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 3, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 10, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 11, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 5, s85
s_mul_hi_i32 s101, 5, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v70, v68, 1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, 1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v70, v70, 1.0, v72 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v70, v68, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v71, v72, -1.0, v69 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v69 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v71, v71, 1.0, v73 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v71, v72, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v74, v76, 1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v74, v74, 1.0, v80 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v74, v76, 1.0, v80 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v75, v80, -1.0, v77 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, -1.0, v77 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v75, v75, 1.0, v81 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v75, v80, 1.0, v81 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 4
v_readlane_b32 s98, v31, 6
_v_pk_add_f16__vop3p 70, 97, 326, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 74, 98, 330, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 71, 97, 327, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 75, 98, 331, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 70, 326, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 71, 327, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 74, 330, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 75, 331, 41, 0x0, 0x3, 0x0, 0x0
v_and_b32 v103, 0x7fff7fff, v70
v_and_b32 v115, 0x7fff7fff, v71
v_and_b32 v92, 0x7fff7fff, v74
v_and_b32 v121, 0x7fff7fff, v75
v_mov_b32 v104, 0xb62eb62e
v_mov_b32 v116, 0xb62eb62e
v_mov_b32 v93, 0xb62eb62e
v_mov_b32 v122, 0xb62eb62e
v_pk_fma_f16 v104, v103, 0x30883088, v104
v_pk_fma_f16 v116, v115, 0x30883088, v116
v_pk_fma_f16 v93, v92, 0x30883088, v93
v_pk_fma_f16 v122, v121, 0x30883088, v122
v_pk_fma_f16 v104, v103, v104, 0x1e331e33
v_pk_fma_f16 v116, v115, v116, 0x1e331e33
v_pk_fma_f16 v93, v92, v93, 0x1e331e33
v_pk_fma_f16 v122, v121, v122, 0x1e331e33
_v_pk_mul_f16__vop3p 104, 359, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 116, 371, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 348, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 122, 377, 378, 0x0, 0x3, 0x0, 0x0
v_pk_fma_f16 v104, v103, v104, v103 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v116, v115, v116, v115 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v93, v92, v93, v92 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v122, v121, v122, v121 op_sel:[0,0,0] op_sel_hi:[1,1,1]
_v_pk_mul_f16__vop3p_lit 103, 0x41c541c5, 359, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 115, 0x41c541c5, 371, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 92, 0x41c541c5, 348, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 121, 0x41c541c5, 377, 0x0, 0x3
v_exp_f16 v103, v103
v_exp_f16 v115, v115
v_exp_f16 v92, v92
v_exp_f16 v121, v121
_v_exp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 103, 242, 359, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 115, 242, 371, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 242, 348, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 121, 242, 377, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v103, v103
v_rcp_f16 v115, v115
v_rcp_f16 v92, v92
v_rcp_f16 v121, v121
_v_rcp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
v_mov_b32 v107, 0x3bc23bc2
v_mov_b32 v84, 0x3bc23bc2
v_mov_b32 v96, 0x3bc23bc2
v_mov_b32 v125, 0x3bc23bc2
v_pk_fma_f16 v103, v103, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v115, v115, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v92, v92, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v121, v121, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_cmp_gt_f16 vcc, |v70|, v107
v_cndmask_b16 v104, v104, v103, vcc
v_cmp_gt_f16 vcc, |v71|, v84
v_cndmask_b16 v116, v116, v115, vcc
v_cmp_gt_f16 vcc, |v74|, v96
v_cndmask_b16 v93, v93, v92, vcc
v_cmp_gt_f16 vcc, |v75|, v125
v_cndmask_b16 v122, v122, v121, vcc
_v_cmp_gt_f16_vccahvhv 70, 107
_v_cndmask_b16__vop3 104, 104, 103, 106, 0xb
_v_cmp_gt_f16_vccahvhv 71, 84
_v_cndmask_b16__vop3 116, 116, 115, 106, 0xb
_v_cmp_gt_f16_vccahvhv 74, 96
_v_cndmask_b16__vop3 93, 93, 92, 106, 0xb
_v_cmp_gt_f16_vccahvhv 75, 125
_v_cndmask_b16__vop3 122, 122, 121, 106, 0xb
v_bfi_b32 v70, 0x7fff7fff, v104, v70
v_bfi_b32 v71, 0x7fff7fff, v116, v71
v_bfi_b32 v74, 0x7fff7fff, v93, v74
v_bfi_b32 v75, 0x7fff7fff, v122, v75
_v_pk_mul_f16__vop3p 70, 326, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 71, 327, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 74, 330, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 75, 331, 40, 0x0, 0x3, 0x0, 0x0
s_waitcnt lgkmcnt(0)
s_barrier
s_cmp_ge_u32 16, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v70, v250, s[92:95], 0 idxen
buffer_store_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 17, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v70, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v71, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 24, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v74, v250, s[92:95], 0 idxen
buffer_store_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 25, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v74, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v75, v251, s[92:95], 0 idxen
s_mul_i32 s100, -7, s85
s_mul_hi_i32 s101, -7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
v_fma_mix_f32 v78, v101, 1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v101, v101, 1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v78, v78, 1.0, v105 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v78, v101, 1.0, v105 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v79, v105, -1.0, v102 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v105, v105, -1.0, v102 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v79, v79, 1.0, v106 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v79, v105, 1.0, v106 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v82, v109, 1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v109, v109, 1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v82, v82, 1.0, v113 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v82, v109, 1.0, v113 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v83, v113, -1.0, v110 op_sel:[0,0,0] op_sel_hi:[1,0,1]
v_fma_mix_f32 v113, v113, -1.0, v110 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v83, v83, 1.0, v114 op_sel:[0,0,0] op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v83, v113, 1.0, v114 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_readlane_b32 s97, v31, 5
v_readlane_b32 s98, v31, 7
_v_pk_add_f16__vop3p 78, 97, 334, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 82, 98, 338, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 79, 97, 335, 0x0, 0x3, 0x0, 0x0
_v_pk_add_f16__vop3p 83, 98, 339, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 78, 334, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 79, 335, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 82, 338, 41, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 83, 339, 41, 0x0, 0x3, 0x0, 0x0
v_and_b32 v103, 0x7fff7fff, v78
v_and_b32 v115, 0x7fff7fff, v79
v_and_b32 v92, 0x7fff7fff, v82
v_and_b32 v121, 0x7fff7fff, v83
v_mov_b32 v104, 0xb62eb62e
v_mov_b32 v116, 0xb62eb62e
v_mov_b32 v93, 0xb62eb62e
v_mov_b32 v122, 0xb62eb62e
v_pk_fma_f16 v104, v103, 0x30883088, v104
v_pk_fma_f16 v116, v115, 0x30883088, v116
v_pk_fma_f16 v93, v92, 0x30883088, v93
v_pk_fma_f16 v122, v121, 0x30883088, v122
v_pk_fma_f16 v104, v103, v104, 0x1e331e33
v_pk_fma_f16 v116, v115, v116, 0x1e331e33
v_pk_fma_f16 v93, v92, v93, 0x1e331e33
v_pk_fma_f16 v122, v121, v122, 0x1e331e33
_v_pk_mul_f16__vop3p 104, 359, 360, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 116, 371, 372, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 93, 348, 349, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 122, 377, 378, 0x0, 0x3, 0x0, 0x0
v_pk_fma_f16 v104, v103, v104, v103 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v116, v115, v116, v115 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v93, v92, v93, v92 op_sel:[0,0,0] op_sel_hi:[1,1,1]
v_pk_fma_f16 v122, v121, v122, v121 op_sel:[0,0,0] op_sel_hi:[1,1,1]
_v_pk_mul_f16__vop3p_lit 103, 0x41c541c5, 359, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 115, 0x41c541c5, 371, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 92, 0x41c541c5, 348, 0x0, 0x3
_v_pk_mul_f16__vop3p_lit 121, 0x41c541c5, 377, 0x0, 0x3
v_exp_f16 v103, v103
v_exp_f16 v115, v115
v_exp_f16 v92, v92
v_exp_f16 v121, v121
_v_exp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_exp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
_v_pk_add_f16__vop3p 103, 242, 359, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 115, 242, 371, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 92, 242, 348, 0x0, 0x2, 0x0, 0x0
_v_pk_add_f16__vop3p 121, 242, 377, 0x0, 0x2, 0x0, 0x0
v_rcp_f16 v103, v103
v_rcp_f16 v115, v115
v_rcp_f16 v92, v92
v_rcp_f16 v121, v121
_v_rcp_f16__vop1 (103 | /*op_sel*/ 0x80), (103 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (115 | /*op_sel*/ 0x80), (115 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (92 | /*op_sel*/ 0x80), (92 | /*op_sel*/ 0x80)
_v_rcp_f16__vop1 (121 | /*op_sel*/ 0x80), (121 | /*op_sel*/ 0x80)
v_mov_b32 v107, 0x3bc23bc2
v_mov_b32 v84, 0x3bc23bc2
v_mov_b32 v96, 0x3bc23bc2
v_mov_b32 v125, 0x3bc23bc2
v_pk_fma_f16 v103, v103, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v115, v115, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v92, v92, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v121, v121, -2.0, 1.0 op_sel:[0,0,0] op_sel_hi:[1,0,0]
v_cmp_gt_f16 vcc, |v78|, v107
v_cndmask_b16 v104, v104, v103, vcc
v_cmp_gt_f16 vcc, |v79|, v84
v_cndmask_b16 v116, v116, v115, vcc
v_cmp_gt_f16 vcc, |v82|, v96
v_cndmask_b16 v93, v93, v92, vcc
v_cmp_gt_f16 vcc, |v83|, v125
v_cndmask_b16 v122, v122, v121, vcc
_v_cmp_gt_f16_vccahvhv 78, 107
_v_cndmask_b16__vop3 104, 104, 103, 106, 0xb
_v_cmp_gt_f16_vccahvhv 79, 84
_v_cndmask_b16__vop3 116, 116, 115, 106, 0xb
_v_cmp_gt_f16_vccahvhv 82, 96
_v_cndmask_b16__vop3 93, 93, 92, 106, 0xb
_v_cmp_gt_f16_vccahvhv 83, 125
_v_cndmask_b16__vop3 122, 122, 121, 106, 0xb
v_bfi_b32 v78, 0x7fff7fff, v104, v78
v_bfi_b32 v79, 0x7fff7fff, v116, v79
v_bfi_b32 v82, 0x7fff7fff, v93, v82
v_bfi_b32 v83, 0x7fff7fff, v122, v83
_v_pk_mul_f16__vop3p 78, 334, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 79, 335, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 82, 338, 40, 0x0, 0x3, 0x0, 0x0
_v_pk_mul_f16__vop3p 83, 339, 40, 0x0, 0x3, 0x0, 0x0
s_waitcnt lgkmcnt(0)
s_cmp_ge_u32 18, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v78, v250, s[92:95], 0 idxen
buffer_store_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 19, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v78, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v79, v251, s[92:95], 0 idxen
s_mul_i32 s100, 7, s85
s_mul_hi_i32 s101, 7, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 26, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_b16 v82, v250, s[92:95], 0 idxen
buffer_store_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 1, s85
s_mul_hi_i32 s101, 1, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_cmp_ge_u32 27, s96
s_cselect_b32 s94, 0, 0x80000000
buffer_store_d16_hi_b16 v82, v250, s[92:95], 0 idxen
buffer_store_d16_hi_b16 v83, v251, s[92:95], 0 idxen
s_mul_i32 s100, 0, s85
s_mul_hi_i32 s101, 0, s85
s_add_u32 s92, s92, s100
s_addc_u32 s93, s93, s101
s_setprio 2
s_branch 1
s_nop 0
v_mov_b32 v35, 0
v_mov_b32 v36, 0
v_mov_b32 v37, 0
v_mov_b32 v38, 0
v_mov_b32 v39, 0
v_mov_b32 v40, 0
v_mov_b32 v41, 0
v_mov_b32 v42, 0
v_mov_b32 v43, 0
v_mov_b32 v44, 0
v_mov_b32 v45, 0
v_mov_b32 v46, 0
v_mov_b32 v47, 0
v_mov_b32 v48, 0
v_mov_b32 v49, 0
v_mov_b32 v50, 0
v_mov_b32 v51, 0
v_mov_b32 v52, 0
v_mov_b32 v53, 0
v_mov_b32 v54, 0
v_mov_b32 v55, 0
v_mov_b32 v56, 0
v_mov_b32 v57, 0
v_mov_b32 v58, 0
v_mov_b32 v59, 0
v_mov_b32 v60, 0
v_mov_b32 v61, 0
v_mov_b32 v62, 0
v_mov_b32 v63, 0
v_mov_b32 v64, 0
v_mov_b32 v65, 0
v_mov_b32 v66, 0
v_mov_b32 v68, 0
v_mov_b32 v69, 0
v_mov_b32 v70, 0
v_mov_b32 v71, 0
v_mov_b32 v72, 0
v_mov_b32 v73, 0
v_mov_b32 v74, 0
v_mov_b32 v75, 0
v_mov_b32 v76, 0
v_mov_b32 v77, 0
v_mov_b32 v78, 0
v_mov_b32 v79, 0
v_mov_b32 v80, 0
v_mov_b32 v81, 0
v_mov_b32 v82, 0
v_mov_b32 v83, 0
v_mov_b32 v84, 0
v_mov_b32 v85, 0
v_mov_b32 v86, 0
v_mov_b32 v87, 0
v_mov_b32 v88, 0
v_mov_b32 v89, 0
v_mov_b32 v90, 0
v_mov_b32 v91, 0
v_mov_b32 v92, 0
v_mov_b32 v93, 0
v_mov_b32 v94, 0
v_mov_b32 v95, 0
v_mov_b32 v96, 0
v_mov_b32 v97, 0
v_mov_b32 v98, 0
v_mov_b32 v99, 0
v_mov_b32 v101, 0
v_mov_b32 v102, 0
v_mov_b32 v103, 0
v_mov_b32 v104, 0
v_mov_b32 v105, 0
v_mov_b32 v106, 0
v_mov_b32 v107, 0
v_mov_b32 v108, 0
v_mov_b32 v109, 0
v_mov_b32 v110, 0
v_mov_b32 v111, 0
v_mov_b32 v112, 0
v_mov_b32 v113, 0
v_mov_b32 v114, 0
v_mov_b32 v115, 0
v_mov_b32 v116, 0
v_mov_b32 v117, 0
v_mov_b32 v118, 0
v_mov_b32 v119, 0
v_mov_b32 v120, 0
v_mov_b32 v121, 0
v_mov_b32 v122, 0
v_mov_b32 v123, 0
v_mov_b32 v124, 0
v_mov_b32 v125, 0
v_mov_b32 v126, 0
v_mov_b32 v127, 0
v_mov_b32 v128, 0
v_mov_b32 v129, 0
v_mov_b32 v130, 0
v_mov_b32 v131, 0
v_mov_b32 v132, 0
v_mov_b32 v134, 0
v_mov_b32 v135, 0
v_mov_b32 v136, 0
v_mov_b32 v137, 0
v_mov_b32 v138, 0
v_mov_b32 v139, 0
v_mov_b32 v140, 0
v_mov_b32 v141, 0
v_mov_b32 v142, 0
v_mov_b32 v143, 0
v_mov_b32 v144, 0
v_mov_b32 v145, 0
v_mov_b32 v146, 0
v_mov_b32 v147, 0
v_mov_b32 v148, 0
v_mov_b32 v149, 0
v_mov_b32 v150, 0
v_mov_b32 v151, 0
v_mov_b32 v152, 0
v_mov_b32 v153, 0
v_mov_b32 v154, 0
v_mov_b32 v155, 0
v_mov_b32 v156, 0
v_mov_b32 v157, 0
v_mov_b32 v158, 0
v_mov_b32 v159, 0
v_mov_b32 v160, 0
v_mov_b32 v161, 0
v_mov_b32 v162, 0
v_mov_b32 v163, 0
v_mov_b32 v164, 0
v_mov_b32 v165, 0
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s92, v248
s_bitcmp1_b32 s92, 0
s_cbranch_scc1 9
s_bitset1_b32 s22, 28
s_add_u32 s90, s17, 15
s_lshr_b32 s90, s90, 4
s_mul_i32 s90, s90, s65
s_mul_i32 s90, s90, s67
s_sub_u32 s90, s90, 1
s_add_u32 s92, s6, s87
s_addc_u32 s93, s7, 0
s_setpc_b64 s[92:93]
s_endpgm
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end

