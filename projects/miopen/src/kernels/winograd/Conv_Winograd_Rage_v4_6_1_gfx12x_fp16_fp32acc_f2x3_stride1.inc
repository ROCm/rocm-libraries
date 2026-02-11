// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

.macro _get_float_const value ret_v
    .if(\value == 1.0)
        \ret_v = 0xf2
    .elseif (\value == 0.5)
        \ret_v = 0xf0
    .elseif (\value == -2)
        \ret_v = 0xc2
    .elseif (\value == 0)
        \ret_v = 0x80
    .endif
.endm

.macro _v_pk_mul_f16_vlitv dst, src0, src1
    .long  0xCC100000 + (\dst)
    .long  0x180200ff + ((\src1) << (41-32))
    .long  \src0
.endm

.macro _v_pk_mul_f16_vsv dst, src0, src1
    .long  0xCC100000 + (\dst)
    .long  0x18020000 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_mul_f16_vvs dst, src0, src1
    .long  0xCC100000 + (\dst)
    .long  0x18000100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_mul_f16_vvv dst, src0, src1
    .long  0xCC100000 + (\dst)
    .long  0x18020100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_add_f16_vvs dst, src0, src1
    .long  0xCC0F0000 + (\dst)
    .long  0x18000100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_add_u16_vsv dst, src0, src1
    .long  0xCC0A0000 + (\dst)
    .long  0x18020000 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_add_u16_vvs dst, src0, src1
    .long  0xCC0A0800 + (\dst)
    .long  0x18000100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_add_f16_vconstv dst, src0, src1
    constant_code = 0
    _get_float_const \src0, constant_code
    .long  0xCC0F0000 + (\dst)
    .long  0x10020000 + ((constant_code) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_max_num_f16_vconstv dst, src0, src1
    constant_code = 0
    _get_float_const \src0, constant_code

    .long  0xCC1C0000 + (\dst)
    .long  0x18020000 + ((constant_code) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_max_num_f16_vvv dst, src0, src1
    .long  0xCC1C0000 + (\dst)
    .long  0x18020100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_min_num_f16_vvv dst, src0, src1
    .long  0xCC1B0000 + (\dst)
    .long  0x18020100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_min_u16_vvv dst, src0, src1
    .long  0xCC0D0000 + (\dst)
    .long  0x18020100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_sub_u16_vvv dst, src0, src1
    .long  0xCC0B0000 + (\dst)
    .long  0x18020100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_pk_ashrrev_i16_vnumv dst, src0, src1
    .long  0xCC060000 + (\dst)
    .long  0x10020080 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_rcp_f16_hvhv dst, src0
    .if(\dst >= 128 || \src0 >= 128)
        .long  0xd5d44800 + (\dst)
        .long  0x00000100 + (\src0)
    .else
        .long  0x7f00a980 + ((\dst) << (17)) + ((\src1) << 0)

    .endif
.endm

.macro _v_exp_f16_hvhv dst, src0
    .if(\dst >= 128 || \src0 >= 128)
        .long  0xd5d84800 + (\dst)
        .long  0x00000100 + (\src0)
    .else
        .long  0x7f00b180 + ((\dst) << (17)) + ((\src1) << 0)

    .endif
.endm

.macro _v_cmp_lt_u16_vcchvhs  src0, src1
    .long  0xd439186a
    .long  0x00000100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_cmp_lt_u16_vccvhs  src0, src1
    .long  0xd439106a
    .long  0x00000100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_cmp_gt_f16_vccahvhv  src0, src1
    .long  0xd404196a
    .long  0x00020100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_cndmask_b16_hvhvhvvcc  dst, src0, src1
    .long  0xd65d5800 + (\dst)
    .long  0x01aa0100 + ((\src0) << (32-32)) + ((\src1) << (41-32))
.endm

.macro _v_cvt_f16_f32_hvv  dst, src0
    .long  0x7f001500 + ((\dst) << (17)) + ((\src0) << 0)
.endm

s_version UC_VERSION_GFX12|UC_VERSION_W64_BIT
s_mov_b32 s1, ttmp9
s_prefetch_inst_pc_rel 0x1f80, null, 31
s_prefetch_inst_pc_rel 0x2f80, null, 31
s_prefetch_inst_pc_rel 0x3f80, null, 31
s_prefetch_inst_pc_rel 0x4f80, null, 31
s_prefetch_inst_pc_rel 0x5f80, null, 31
s_prefetch_inst_pc_rel 0x6f80, null, 31
s_getpc_b64 s[4:5]
s_load_b64 s[6:7], s[2:3], 0x0
s_load_b512 s[8:23], s[2:3], 0x8
s_load_b256 s[24:31], s[2:3], 0x48
v_readfirstlane_b32 s64, v0
v_and_b32_e32 v6, 0xff, v0
v_lshrrev_b32_e32 v7, 1, v6
v_and_b32_e32 v8, 1, v0
v_and_b32_e32 v1, 15, v6
v_mul_u32_u24_e32 v3, 0x1040, v8
v_lshlrev_b32_e32 v1, 3, v1
v_and_b32_e32 v4, 31, v7
v_bfe_u32 v2, v6, 4, 2
v_lshl_add_u32 v3, v4, 3, v3
v_lshl_add_u32 v1, v2, 10, v1
v_bfe_u32 v4, v6, 6, 2
v_bfe_u32 v2, v6, 6, 2
v_bfe_u32 v5, v0, 8, 1
v_lshl_add_u32 v1, v2, 8, v1
v_lshlrev_b32_e32 v4, v5, v4
v_bfe_u32 v2, v0, 8, 1
v_lshl_add_u32 v3, v4, 10, v3
v_lshl_add_u32 v1, v2, 7, v1
s_lshr_b32 s64, s64, 6
s_cmp_ge_u32 s64, 4
s_cbranch_scc0 1
s_setprio 3
s_wait_kmcnt 0x0
s_and_b32 s12, s12, 0x3ffff
s_and_b32 s65, s12, 0x2000
s_cbranch_scc0 2
s_load_b256 s[32:39], s[2:3], 0x68
s_and_b32 s65, s12, 0x200
s_cbranch_scc0 4
s_load_b256 s[40:47], s[2:3], 0x88
s_load_b128 s[48:51], s[2:3], 0xa8
s_and_b32 s65, s12, 0x400
s_cbranch_scc0 2
s_load_b128 s[52:55], s[2:3], 0xb8
s_and_b32 s65, s12, 0x14000
s_cbranch_scc0 2
s_load_b32 s56, s[2:3], 0xc8
s_and_b32 s65, s12, 0x10000
s_cbranch_scc0 4
s_load_b32 s57, s[2:3], 0xcc
s_load_b64 s[58:59], s[2:3], 0xd0
s_and_b32 s65, s12, 0x20000
s_cbranch_scc0 2
s_load_b128 s[60:63], s[2:3], 0xd8
s_bitcmp1_b32 s12, 3
s_cbranch_scc0 2
s_denorm_mode 15
s_round_mode 0x0
s_wait_kmcnt 0x0
s_bitcmp1_b32 s12, 6
s_cbranch_scc0 30
s_and_b32 s15, s15, 0xffff
s_and_b32 s17, s17, 0xffff
s_and_b32 s19, s19, 0xffff
s_and_b32 s29, s29, 0xffff
s_and_b32 s61, s61, 0xffff
s_and_b32 s59, s59, 0xffff
s_load_b64 s[14:15], s[14:15], 0x0
s_load_b64 s[16:17], s[16:17], 0x0
s_load_b64 s[18:19], s[18:19], 0x0
s_bitcmp1_b32 s12, 7
s_cbranch_scc0 2
s_load_b64 s[28:29], s[28:29], 0x0
s_bitcmp1_b32 s12, 17
s_cbranch_scc0 2
s_load_b64 s[60:61], s[60:61], 0x0
s_bitcmp1_b32 s12, 16
s_cbranch_scc0 2
s_load_b64 s[58:59], s[58:59], 0x0
s_bitcmp1_b32 s12, 9
s_cbranch_scc1 14
s_mov_b32 s42, s9
s_mul_i32 s41, s8, s9
s_mul_i32 s40, s41, s7
s_mov_b32 s50, s27
s_mul_i32 s49, s26, s27
s_mul_i32 s48, s49, s10
s_mul_i32 s65, s22, s23
s_bitcmp1_b32 s12, 2
s_cselect_b32 s66, s10, s7
s_mul_i32 s66, s65, s66
s_mov_b32 s46, s23
s_bitcmp1_b32 s12, 2
s_cselect_b32 s45, s66, s65
s_cselect_b32 s44, s65, s66
s_and_b32 s65, s56, 0xffffff00
s_bitcmp1_b32 s12, 8
s_cselect_b32 s66, 1, 0
s_bitcmp1_b32 s12, 14
s_cselect_b32 s56, s56, s66
s_or_b32 s56, s56, s65
s_wait_kmcnt 0x0
s_bitcmp1_b32 s12, 13
s_cbranch_scc0 5
s_add_nc_u64 s[14:15], s[14:15], s[32:33]
s_add_nc_u64 s[18:19], s[18:19], s[36:37]
s_add_nc_u64 s[16:17], s[16:17], s[34:35]
s_add_nc_u64 s[28:29], s[28:29], s[38:39]
s_add_nc_u64 s[60:61], s[60:61], s[62:63]
s_bitcmp1_b32 s12, 10
s_cbranch_scc0 57
v_clz_i32_u32_e32 v6, s11
v_lshlrev_b32_e64 v7, v6, s11
v_and_b32_e32 v5, 0xffffff00, v7
v_cmp_eq_u32_e32 vcc_lo, 0x80000000, v7
v_cvt_f32_u32_e32 v5, v5
v_rcp_f32_e32 v2, v5
v_sub_co_ci_u32_e32 v4, vcc_lo, 32, v6, vcc_lo
v_cvt_f32_ubyte0_e32 v6, v7
v_fma_f32 v5, v5, v2, -1.0
v_fma_f32 v5, v6, v2, v5
v_fmaak_f32 v5, v5, v2, 0x9f000000
v_mul_f32_e32 v5, 0x5f800000, v5
v_mov_b32_e32 v6, 0
v_cvt_floor_i32_f32_e64 v5, -v5
v_lshl_add_u32 v2, v2, 9, v5
v_mad_co_u64_u32 v[6:7], vcc_lo, v7, v2, v[6:7]
v_sub_co_ci_u32_e64 v2, vcc_lo, v2, -1, vcc_lo
v_mul_hi_u32 v5, s1, v2
v_add_co_u32 v2, vcc_lo, v5, s1
v_add_co_ci_u32_e64 v5, vcc_lo, 0, 0, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 32, v4
v_cndmask_b32_e32 v2, v2, v5, vcc_lo
v_alignbit_b32 v2, v5, v2, v4
v_readfirstlane_b32 s65, v2
s_mul_i32 s66, s65, s11
s_sub_co_u32 s1, s1, s66
s_mul_i32 s66, s53, s65
s_mul_hi_u32 s67, s53, s65
s_lshl_b64 s[66:67], s[66:67], 1
s_add_nc_u64 s[14:15], s[14:15], s[66:67]
s_mul_i32 s66, s54, s65
s_mul_hi_u32 s67, s54, s65
s_lshl_b64 s[66:67], s[66:67], 1
s_add_nc_u64 s[16:17], s[16:17], s[66:67]
s_mul_i32 s66, s55, s65
s_mul_hi_u32 s67, s55, s65
s_lshl_b64 s[66:67], s[66:67], 1
s_add_nc_u64 s[18:19], s[18:19], s[66:67]
s_mul_i32 s66, s10, s65
s_lshl_b32 s66, s66, 1
s_add_co_u32 s28, s28, s66
s_add_co_ci_u32 s29, s29, 0
s_cmp_eq_u32 s11, 64
s_cbranch_scc0 11
s_and_b32 s65, s1, 3
s_bfe_u32 s67, s1, 0x20002
s_bfe_u32 s66, s1, 0x10004
s_bfe_u32 s68, s1, 0x10005
s_mov_b32 s1, s65
s_lshl1_add_u32 s1, s1, s66
s_lshl2_add_u32 s1, s1, s67
s_lshl1_add_u32 s1, s1, s68
s_add_co_u32 s67, s10, 31
s_lshr_b32 s67, s67, 5
v_clz_i32_u32_e32 v6, s67
v_lshlrev_b32_e64 v7, v6, s67
v_and_b32_e32 v8, 0xffffff00, v7
v_cmp_eq_u32_e32 vcc_lo, 0x80000000, v7
v_cvt_f32_u32_e32 v8, v8
v_rcp_f32_e32 v2, v8
v_sub_co_ci_u32_e32 v5, vcc_lo, 32, v6, vcc_lo
v_cvt_f32_ubyte0_e32 v6, v7
v_fma_f32 v8, v8, v2, -1.0
v_fma_f32 v8, v6, v2, v8
v_fmaak_f32 v8, v8, v2, 0x9f000000
v_mul_f32_e32 v8, 0x5f800000, v8
v_mov_b32_e32 v6, 0
v_cvt_floor_i32_f32_e64 v8, -v8
v_lshl_add_u32 v2, v2, 9, v8
v_mad_co_u64_u32 v[6:7], vcc_lo, v7, v2, v[6:7]
v_sub_co_ci_u32_e64 v2, vcc_lo, v2, -1, vcc_lo
v_mul_hi_u32 v6, s11, v2
v_add_co_u32 v2, vcc_lo, v6, s11
v_add_co_ci_u32_e64 v6, vcc_lo, 0, 0, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 32, v5
v_cndmask_b32_e32 v2, v2, v6, vcc_lo
v_alignbit_b32 v2, v6, v2, v5
v_clz_i32_u32_e32 v6, v2
v_lshlrev_b32_e32 v7, v6, v2
v_and_b32_e32 v8, 0xffffff00, v7
v_cmp_eq_u32_e32 vcc_lo, 0x80000000, v7
v_cvt_f32_u32_e32 v8, v8
v_rcp_f32_e32 v4, v8
v_sub_co_ci_u32_e32 v5, vcc_lo, 32, v6, vcc_lo
v_cvt_f32_ubyte0_e32 v6, v7
v_fma_f32 v8, v8, v4, -1.0
v_fma_f32 v8, v6, v4, v8
v_fmaak_f32 v8, v8, v4, 0x9f000000
v_mul_f32_e32 v8, 0x5f800000, v8
v_mov_b32_e32 v6, 0
v_cvt_floor_i32_f32_e64 v8, -v8
v_lshl_add_u32 v4, v4, 9, v8
v_mad_co_u64_u32 v[6:7], vcc_lo, v7, v4, v[6:7]
v_sub_co_ci_u32_e64 v4, vcc_lo, v4, -1, vcc_lo
v_mul_hi_u32 v6, s1, v4
v_add_co_u32 v4, vcc_lo, v6, s1
v_add_co_ci_u32_e64 v6, vcc_lo, 0, 0, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 32, v5
v_cndmask_b32_e32 v4, v4, v6, vcc_lo
v_alignbit_b32 v4, v6, v4, v5
v_readfirstlane_b32 s65, v2
v_readfirstlane_b32 s68, v4
s_cmp_ge_u32 s68, s67
s_cbranch_scc1 9443
s_mul_i32 s11, s65, s67
s_mul_i32 s66, s68, s65
s_sub_co_u32 s66, s1, s66
s_mul_i32 s69, s68, 32
s_sub_co_u32 s10, s10, s69
s_min_u32 s10, s10, 32
s_lshl_b32 s69, s69, 1
s_mul_i32 s70, s69, s44
s_mul_hi_u32 s71, s69, s44
s_add_nc_u64 s[16:17], s[16:17], s[70:71]
s_mul_i32 s70, s69, s49
s_mul_hi_u32 s71, s69, s49
s_add_nc_u64 s[18:19], s[18:19], s[70:71]
s_add_co_u32 s28, s28, s69
s_add_co_ci_u32 s29, s29, 0
s_cmp_ge_u32 s64, 4
s_cbranch_scc1 4395
v_and_b32_e32 v27, 0xff, v0
s_bitcmp1_b32 s12, 31
s_cselect_b32 s54, 2, 0
s_cselect_b32 s55, 0, 2
v_bfe_u32 v25, v27, 2, s54
v_bfe_u32 v28, v27, 6, s55
v_add_nc_u32_e32 v28, v25, v28
v_lshrrev_b32_e32 v29, 2, v27
v_lshrrev_b32_e32 v29, s54, v29
v_and_b32_e32 v25, 3, v29
v_bfe_u32 v4, v29, 2, 1
v_lshl_add_u32 v25, v4, 3, v25
v_bfe_u32 v4, v29, 3, 1
v_and_b32_e32 v26, 15, v27
v_lshl_add_u32 v25, v4, 2, v25
v_bfe_u32 v2, v27, 6, 2
v_mul_u32_u24_e32 v4, 24, v25
v_lshl_add_u32 v26, v2, 4, v26
v_xor_b32_dpp v25, v27, v27 quad_perm:[0,0,0,1] row_mask:0xf bank_mask:0xf
v_mul_u32_u24_e32 v2, 24, v26
v_lshl_add_u32 v4, v25, 3, v4
v_bfe_u32 v26, v27, 4, 2
v_mad_u32_u16 v4, v28, 0x620, v4
v_mad_u32_u16 v2, v26, 0x620, v2
s_cmp_eq_u32 s64, 5
s_cselect_b32 s57, 0x400, 0
s_cselect_b32 s58, 4, 0
s_and_b32 s60, s64, 3
s_mul_i32 s59, s60, 0x800
s_lshl_b32 s60, s59, 1
s_cmp_ge_u32 s64, 4
s_cselect_b32 s59, s60, s59
v_and_b32_e32 v25, 63, v0
v_lshl_add_u32 v5, v25, 4, s57
v_mov_b32_e32 v7, s58
v_lshl_add_u32 v6, v25, 4, s59
s_barrier_signal -1
s_barrier_wait 0xffff
ds_load_b32 v24, v7 offset:55808
s_mov_b32 s21, 0
s_mov_b32 s43, 0
s_lshl_b32 s54, s41, 1
s_lshl_b32 s55, s45, 1
s_and_b32 s57, s64, 3
s_mul_i32 s58, 1, s57
s_sub_co_u32 s13, s7, s58
s_add_co_ci_u32 s13, s13, 3
s_lshl_b32 s58, s54, 0
s_mul_i32 s58, s58, s57
s_add_co_u32 s14, s14, s58
s_add_co_ci_u32 s15, s15, 0
s_lshl_b32 s54, s54, 2
s_lshl_b32 s55, s55, 2
s_and_b32 s15, s15, 0xffff
s_add_co_u32 s15, s15, 0x20000
s_and_b32 s17, s17, 0xffff
s_add_co_u32 s17, s17, 0x20000
s_mov_b64 s[32:33], s[14:15]
s_mov_b32 s34, 0
s_mov_b32 s35, 0x10014000
s_mov_b64 s[36:37], s[14:15]
s_mov_b32 s38, 0
s_mov_b32 s39, 0x10014000
s_mov_b32 s20, 0x80000000
s_cmp_ge_u32 s64, 2
s_cbranch_scc0 31
s_branch 1926
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_readfirstlane_b32 s47, v24
ds_load_b32 v24, v7 offset:55808
ds_load_b128 v[8:11], v5 offset:53760
ds_load_b128 v[12:15], v5 offset:54784
ds_load_b128 v[16:19], v6 offset:45568
ds_load_b128 v[20:23], v6 offset:46592
s_bitcmp1_b32 s47, 0
s_cmov_b32 s20, 0
s_add_co_ci_u32 s43, s43, 2
s_sub_co_u32 s21, s7, 1
s_add_co_u32 s52, s4, s51
s_add_co_ci_u32 s53, s5, 0
s_mov_b32 s2, s13
s_mov_b64 s[32:33], s[14:15]
s_mov_b32 s34, s20
s_mov_b32 s3, s13
s_mov_b64 s[36:37], s[16:17]
s_mov_b32 s38, s20
s_setpc_b64 s[52:53]
s_lshl_b32 s43, s43, 2
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v125, 1.0, v59 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v125, 1.0, v59 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v177, 1.0, v187 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v177, 1.0, v187 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[27:34], v[173:176], v[158:161], v[27:34]
v_wmma_f32_16x16x16_f16 v[31:38], v[173:176], v[160:163], v[31:38]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v166, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v166, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v182, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v182, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[126:133], v[188:191], v[169:172], v[126:133]
v_wmma_f32_16x16x16_f16 v[130:137], v[188:191], v[171:174], v[130:137]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[93:100], v[183:186], v[25:28], v[93:100]
v_wmma_f32_16x16x16_f16 v[97:104], v[183:186], v[167:170], v[97:104]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v166 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v166 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v182 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v182 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[60:67], v[178:181], v[162:165], v[60:67]
v_wmma_f32_16x16x16_f16 v[64:71], v[178:181], v[164:167], v[64:71]
v_mov_b32_e32 v234, v125
v_mov_b32_e32 v236, v59
v_mov_b32_e32 v235, v177
v_mov_b32_e32 v237, v187
s_setprio 2
s_mov_b32 s51, 0x860
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 3697
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[175:176], v2 offset:39296
ds_store_b64 v4, v[234:235] offset:33024
ds_load_b64 v[180:181], v2 offset:39304
ds_store_b64 v4, v[236:237] offset:34176
buffer_load_d16_b16 v125, v8, s[32:35], null idxen
buffer_load_d16_b16 v177, v12, s[32:35], null idxen
buffer_load_d16_b16 v92, v10, s[32:35], null idxen
buffer_load_d16_b16 v192, v14, s[32:35], null idxen
buffer_load_d16_b16 v166, v9, s[32:35], null idxen
buffer_load_d16_b16 v182, v13, s[32:35], null idxen
buffer_load_d16_b16 v59, v11, s[32:35], null idxen
buffer_load_d16_b16 v187, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[190:191], v2 offset:39312
ds_store_b64 v4, v[238:239] offset:33408
ds_store_b64 v4, v[240:241] offset:33792
buffer_load_d16_hi_b16 v125, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v177, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v92, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v192, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v166, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v182, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v59, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v187, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_fma_mix_f32 v244, v175, 1.0, v190 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v175, 1.0, v190 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v176, 1.0, v191 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v176, 1.0, v191 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v193, 1.0, v195 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v193, 1.0, v195 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v197, 1.0, v199 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v197, 1.0, v199 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:50], v[175:178], v[158:161], v[43:50]
v_wmma_f32_16x16x16_f16 v[47:54], v[175:178], v[160:163], v[47:54]
v_fma_mix_f32 v244, v180, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v180, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v181, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v181, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v234, v194, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v194, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v198, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v198, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[142:149], v[190:193], v[169:172], v[142:149]
v_wmma_f32_16x16x16_f16 v[146:153], v[190:193], v[171:174], v[146:153]
v_fma_mixlo_f16 v185, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v185, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v186, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[109:116], v[185:188], v[25:28], v[109:116]
v_wmma_f32_16x16x16_f16 v[113:120], v[185:188], v[167:170], v[113:120]
v_fma_mixlo_f16 v180, v244, 0.5, v180 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v180, v242, 0.5, v180 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v181, v245, 0.5, v181 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v181, v243, 0.5, v181 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v194 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v194 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v198 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v198 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[76:83], v[180:183], v[162:165], v[76:83]
v_wmma_f32_16x16x16_f16 v[80:87], v[180:183], v[164:167], v[80:87]
v_mov_b32_e32 v234, v193
v_mov_b32_e32 v236, v195
v_mov_b32_e32 v235, v197
v_mov_b32_e32 v237, v199
s_setprio 2
s_mov_b32 s51, 0xaf8
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 3651
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[158:159], v1 offset:24768
ds_load_b64 v[160:161], v1 offset:24896
ds_store_b64 v4, v[234:235] offset:39296
ds_load_b64 v[162:163], v1 offset:28928
ds_load_b64 v[164:165], v1 offset:29056
ds_store_b64 v4, v[236:237] offset:40448
buffer_load_d16_b16 v193, v8, s[32:35], null idxen
buffer_load_d16_b16 v197, v12, s[32:35], null idxen
buffer_load_d16_b16 v196, v10, s[32:35], null idxen
buffer_load_d16_b16 v200, v14, s[32:35], null idxen
buffer_load_d16_b16 v194, v9, s[32:35], null idxen
buffer_load_d16_b16 v198, v13, s[32:35], null idxen
buffer_load_d16_b16 v195, v11, s[32:35], null idxen
buffer_load_d16_b16 v199, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[25:26], v1 offset:24776
ds_load_b64 v[167:168], v1 offset:24904
ds_store_b64 v4, v[238:239] offset:39680
ds_load_b64 v[169:170], v1 offset:28936
ds_load_b64 v[171:172], v1 offset:29064
ds_store_b64 v4, v[240:241] offset:40064
buffer_load_d16_hi_b16 v193, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v197, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v196, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v200, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v194, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v198, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v195, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v199, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_mov_b32 s51, 0xc54
s_sub_co_u32 s21, s21, 16
s_cbranch_scc1 65164
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v203, -1.0, v202 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v207, -1.0, v206 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:42], v[173:176], v[158:161], v[35:42]
v_wmma_f32_16x16x16_f16 v[39:46], v[173:176], v[160:163], v[39:46]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_pk_fma_f16 v236, v203, 1.0, v204 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v207, 1.0, v208 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[134:141], v[188:191], v[169:172], v[134:141]
v_wmma_f32_16x16x16_f16 v[138:145], v[188:191], v[171:174], v[138:145]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v238, v204, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v208, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[101:108], v[183:186], v[25:28], v[101:108]
v_wmma_f32_16x16x16_f16 v[105:112], v[183:186], v[167:170], v[105:112]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v204, -1.0, v201 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v208, -1.0, v205 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[68:75], v[178:181], v[162:165], v[68:75]
v_wmma_f32_16x16x16_f16 v[72:79], v[178:181], v[164:167], v[72:79]
s_setprio 2
s_mov_b32 s51, 0xd7c
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 3610
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[173:174], v2 offset:33024
ds_store_b64 v3, v[234:235] offset:16512
ds_load_b64 v[178:179], v2 offset:33032
ds_store_b64 v3, v[236:237] offset:16768
buffer_load_d16_b16 v202, v16, s[36:39], null idxen
buffer_load_d16_b16 v206, v20, s[36:39], null idxen
buffer_load_d16_b16 v204, v17, s[36:39], null idxen
buffer_load_d16_b16 v208, v21, s[36:39], null idxen
buffer_load_d16_b16 v201, v18, s[36:39], null idxen
buffer_load_d16_b16 v205, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[188:189], v2 offset:33040
ds_store_b64 v3, v[238:239] offset:17024
ds_store_b64 v3, v[240:241] offset:17280
buffer_load_d16_hi_b16 v202, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v206, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v204, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v208, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v201, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v205, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 7
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_fma_mix_f32 v244, v173, 1.0, v188 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v173, 1.0, v188 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v174, 1.0, v189 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v174, 1.0, v189 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v211, -1.0, v210 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v215, -1.0, v214 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:58], v[175:178], v[158:161], v[51:58]
v_wmma_f32_16x16x16_f16 v[55:62], v[175:178], v[160:163], v[55:62]
v_fma_mix_f32 v244, v178, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v178, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v179, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v179, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v236, v211, 1.0, v212 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v215, 1.0, v216 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[150:157], v[190:193], v[169:172], v[150:157]
v_wmma_f32_16x16x16_f16 v[154:161], v[190:193], v[171:174], v[154:161]
v_fma_mixlo_f16 v183, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v183, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v184, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v184, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_pk_fma_f16 v238, v212, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v216, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[117:124], v[185:188], v[25:28], v[117:124]
v_wmma_f32_16x16x16_f16 v[121:128], v[185:188], v[167:170], v[121:128]
v_fma_mixlo_f16 v178, v244, 0.5, v178 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v178, v242, 0.5, v178 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v179, v245, 0.5, v179 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v179, v243, 0.5, v179 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v240, v212, -1.0, v209 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v216, -1.0, v213 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[84:91], v[180:183], v[162:165], v[84:91]
v_wmma_f32_16x16x16_f16 v[88:95], v[180:183], v[164:167], v[88:95]
s_setprio 2
s_mov_b32 s51, 0xfa8
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 3591
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[158:159], v1 offset:16512
ds_load_b64 v[160:161], v1 offset:16640
ds_store_b64 v3, v[234:235] offset:24768
ds_load_b64 v[162:163], v1 offset:20672
ds_load_b64 v[164:165], v1 offset:20800
ds_store_b64 v3, v[236:237] offset:25024
buffer_load_d16_b16 v210, v16, s[36:39], null idxen
buffer_load_d16_b16 v214, v20, s[36:39], null idxen
buffer_load_d16_b16 v212, v17, s[36:39], null idxen
buffer_load_d16_b16 v216, v21, s[36:39], null idxen
buffer_load_d16_b16 v209, v18, s[36:39], null idxen
buffer_load_d16_b16 v213, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[25:26], v1 offset:16520
ds_load_b64 v[167:168], v1 offset:16648
ds_store_b64 v3, v[238:239] offset:25280
ds_load_b64 v[169:170], v1 offset:20680
ds_load_b64 v[171:172], v1 offset:20808
ds_store_b64 v3, v[240:241] offset:25536
buffer_load_d16_hi_b16 v210, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v214, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v212, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v216, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v209, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v213, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 2
s_nop 0
s_nop 0
s_lshl_b32 s43, s43, 2
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v217, 1.0, v219 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v217, 1.0, v219 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v221, 1.0, v223 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v221, 1.0, v223 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[27:34], v[173:176], v[158:161], v[27:34]
v_wmma_f32_16x16x16_f16 v[31:38], v[173:176], v[160:163], v[31:38]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v218, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v218, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v222, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v222, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[126:133], v[188:191], v[169:172], v[126:133]
v_wmma_f32_16x16x16_f16 v[130:137], v[188:191], v[171:174], v[130:137]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[93:100], v[183:186], v[25:28], v[93:100]
v_wmma_f32_16x16x16_f16 v[97:104], v[183:186], v[167:170], v[97:104]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v218 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v218 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v222 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v222 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[60:67], v[178:181], v[162:165], v[60:67]
v_wmma_f32_16x16x16_f16 v[64:71], v[178:181], v[164:167], v[64:71]
v_mov_b32_e32 v234, v217
v_mov_b32_e32 v236, v219
v_mov_b32_e32 v235, v221
v_mov_b32_e32 v237, v223
s_setprio 2
s_mov_b32 s51, 0x1240
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 3065
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[175:176], v2 offset:39296
ds_store_b64 v4, v[234:235] offset:33024
ds_load_b64 v[180:181], v2 offset:39304
ds_store_b64 v4, v[236:237] offset:34176
buffer_load_d16_b16 v217, v8, s[32:35], null idxen
buffer_load_d16_b16 v221, v12, s[32:35], null idxen
buffer_load_d16_b16 v220, v10, s[32:35], null idxen
buffer_load_d16_b16 v224, v14, s[32:35], null idxen
buffer_load_d16_b16 v218, v9, s[32:35], null idxen
buffer_load_d16_b16 v222, v13, s[32:35], null idxen
buffer_load_d16_b16 v219, v11, s[32:35], null idxen
buffer_load_d16_b16 v223, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[190:191], v2 offset:39312
ds_store_b64 v4, v[238:239] offset:33408
ds_store_b64 v4, v[240:241] offset:33792
buffer_load_d16_hi_b16 v217, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v221, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v220, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v224, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v218, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v222, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v219, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v223, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_fma_mix_f32 v244, v175, 1.0, v190 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v175, 1.0, v190 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v176, 1.0, v191 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v176, 1.0, v191 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v225, 1.0, v227 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v225, 1.0, v227 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v229, 1.0, v231 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v229, 1.0, v231 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:50], v[175:178], v[158:161], v[43:50]
v_wmma_f32_16x16x16_f16 v[47:54], v[175:178], v[160:163], v[47:54]
v_fma_mix_f32 v244, v180, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v180, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v181, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v181, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v234, v226, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v226, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v230, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v230, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[142:149], v[190:193], v[169:172], v[142:149]
v_wmma_f32_16x16x16_f16 v[146:153], v[190:193], v[171:174], v[146:153]
v_fma_mixlo_f16 v185, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v185, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v186, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[109:116], v[185:188], v[25:28], v[109:116]
v_wmma_f32_16x16x16_f16 v[113:120], v[185:188], v[167:170], v[113:120]
v_fma_mixlo_f16 v180, v244, 0.5, v180 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v180, v242, 0.5, v180 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v181, v245, 0.5, v181 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v181, v243, 0.5, v181 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v226 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v226 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v230 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v230 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[76:83], v[180:183], v[162:165], v[76:83]
v_wmma_f32_16x16x16_f16 v[80:87], v[180:183], v[164:167], v[80:87]
v_mov_b32_e32 v234, v225
v_mov_b32_e32 v236, v227
v_mov_b32_e32 v235, v229
v_mov_b32_e32 v237, v231
s_setprio 2
s_mov_b32 s51, 0x14d8
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 3019
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[158:159], v1 offset:24768
ds_load_b64 v[160:161], v1 offset:24896
ds_store_b64 v4, v[234:235] offset:39296
ds_load_b64 v[162:163], v1 offset:28928
ds_load_b64 v[164:165], v1 offset:29056
ds_store_b64 v4, v[236:237] offset:40448
buffer_load_d16_b16 v225, v8, s[32:35], null idxen
buffer_load_d16_b16 v229, v12, s[32:35], null idxen
buffer_load_d16_b16 v228, v10, s[32:35], null idxen
buffer_load_d16_b16 v232, v14, s[32:35], null idxen
buffer_load_d16_b16 v226, v9, s[32:35], null idxen
buffer_load_d16_b16 v230, v13, s[32:35], null idxen
buffer_load_d16_b16 v227, v11, s[32:35], null idxen
buffer_load_d16_b16 v231, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[25:26], v1 offset:24776
ds_load_b64 v[167:168], v1 offset:24904
ds_store_b64 v4, v[238:239] offset:39680
ds_load_b64 v[169:170], v1 offset:28936
ds_load_b64 v[171:172], v1 offset:29064
ds_store_b64 v4, v[240:241] offset:40064
buffer_load_d16_hi_b16 v225, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v229, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v228, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v232, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v226, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v230, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v227, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v231, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_mov_b32 s51, 0x1634
s_sub_co_u32 s21, s21, 16
s_cbranch_scc1 64532
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v92, -1.0, v125 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v196, -1.0, v193 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:42], v[173:176], v[158:161], v[35:42]
v_wmma_f32_16x16x16_f16 v[39:46], v[173:176], v[160:163], v[39:46]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_pk_fma_f16 v236, v92, 1.0, v166 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v196, 1.0, v194 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[134:141], v[188:191], v[169:172], v[134:141]
v_wmma_f32_16x16x16_f16 v[138:145], v[188:191], v[171:174], v[138:145]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v238, v166, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v194, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[101:108], v[183:186], v[25:28], v[101:108]
v_wmma_f32_16x16x16_f16 v[105:112], v[183:186], v[167:170], v[105:112]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v166, -1.0, v59 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v194, -1.0, v195 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[68:75], v[178:181], v[162:165], v[68:75]
v_wmma_f32_16x16x16_f16 v[72:79], v[178:181], v[164:167], v[72:79]
s_setprio 2
s_mov_b32 s51, 0x175c
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 2978
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[173:174], v2 offset:33024
ds_store_b64 v3, v[234:235] offset:16512
ds_load_b64 v[178:179], v2 offset:33032
ds_store_b64 v3, v[236:237] offset:16768
buffer_load_d16_b16 v125, v16, s[36:39], null idxen
buffer_load_d16_b16 v193, v20, s[36:39], null idxen
buffer_load_d16_b16 v166, v17, s[36:39], null idxen
buffer_load_d16_b16 v194, v21, s[36:39], null idxen
buffer_load_d16_b16 v59, v18, s[36:39], null idxen
buffer_load_d16_b16 v195, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[188:189], v2 offset:33040
ds_store_b64 v3, v[238:239] offset:17024
ds_store_b64 v3, v[240:241] offset:17280
buffer_load_d16_hi_b16 v125, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v193, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v166, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v194, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v59, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v195, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 7
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_fma_mix_f32 v244, v173, 1.0, v188 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v173, 1.0, v188 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v174, 1.0, v189 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v174, 1.0, v189 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v192, -1.0, v177 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v200, -1.0, v197 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:58], v[175:178], v[158:161], v[51:58]
v_wmma_f32_16x16x16_f16 v[55:62], v[175:178], v[160:163], v[55:62]
v_fma_mix_f32 v244, v178, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v178, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v179, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v179, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v236, v192, 1.0, v182 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v200, 1.0, v198 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[150:157], v[190:193], v[169:172], v[150:157]
v_wmma_f32_16x16x16_f16 v[154:161], v[190:193], v[171:174], v[154:161]
v_fma_mixlo_f16 v183, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v183, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v184, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v184, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_pk_fma_f16 v238, v182, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v198, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[117:124], v[185:188], v[25:28], v[117:124]
v_wmma_f32_16x16x16_f16 v[121:128], v[185:188], v[167:170], v[121:128]
v_fma_mixlo_f16 v178, v244, 0.5, v178 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v178, v242, 0.5, v178 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v179, v245, 0.5, v179 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v179, v243, 0.5, v179 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v240, v182, -1.0, v187 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v198, -1.0, v199 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[84:91], v[180:183], v[162:165], v[84:91]
v_wmma_f32_16x16x16_f16 v[88:95], v[180:183], v[164:167], v[88:95]
s_setprio 2
s_mov_b32 s51, 0x1988
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 2959
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[158:159], v1 offset:16512
ds_load_b64 v[160:161], v1 offset:16640
ds_store_b64 v3, v[234:235] offset:24768
ds_load_b64 v[162:163], v1 offset:20672
ds_load_b64 v[164:165], v1 offset:20800
ds_store_b64 v3, v[236:237] offset:25024
buffer_load_d16_b16 v177, v16, s[36:39], null idxen
buffer_load_d16_b16 v197, v20, s[36:39], null idxen
buffer_load_d16_b16 v182, v17, s[36:39], null idxen
buffer_load_d16_b16 v198, v21, s[36:39], null idxen
buffer_load_d16_b16 v187, v18, s[36:39], null idxen
buffer_load_d16_b16 v199, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[25:26], v1 offset:16520
ds_load_b64 v[167:168], v1 offset:16648
ds_store_b64 v3, v[238:239] offset:25280
ds_load_b64 v[169:170], v1 offset:20680
ds_load_b64 v[171:172], v1 offset:20808
ds_store_b64 v3, v[240:241] offset:25536
buffer_load_d16_hi_b16 v177, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v197, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v182, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v198, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v187, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v199, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 2
s_nop 0
s_nop 0
s_lshl_b32 s43, s43, 2
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v202, 1.0, v201 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v202, 1.0, v201 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v210, 1.0, v209 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v210, 1.0, v209 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[27:34], v[173:176], v[158:161], v[27:34]
v_wmma_f32_16x16x16_f16 v[31:38], v[173:176], v[160:163], v[31:38]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v204, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v204, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v212, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v212, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[126:133], v[188:191], v[169:172], v[126:133]
v_wmma_f32_16x16x16_f16 v[130:137], v[188:191], v[171:174], v[130:137]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[93:100], v[183:186], v[25:28], v[93:100]
v_wmma_f32_16x16x16_f16 v[97:104], v[183:186], v[167:170], v[97:104]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v204 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v204 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v212 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v212 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[60:67], v[178:181], v[162:165], v[60:67]
v_wmma_f32_16x16x16_f16 v[64:71], v[178:181], v[164:167], v[64:71]
v_mov_b32_e32 v234, v202
v_mov_b32_e32 v236, v201
v_mov_b32_e32 v235, v210
v_mov_b32_e32 v237, v209
s_setprio 2
s_mov_b32 s51, 0x1c20
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 2433
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[175:176], v2 offset:39296
ds_store_b64 v4, v[234:235] offset:33024
ds_load_b64 v[180:181], v2 offset:39304
ds_store_b64 v4, v[236:237] offset:34176
buffer_load_d16_b16 v202, v8, s[32:35], null idxen
buffer_load_d16_b16 v210, v12, s[32:35], null idxen
buffer_load_d16_b16 v203, v10, s[32:35], null idxen
buffer_load_d16_b16 v211, v14, s[32:35], null idxen
buffer_load_d16_b16 v204, v9, s[32:35], null idxen
buffer_load_d16_b16 v212, v13, s[32:35], null idxen
buffer_load_d16_b16 v201, v11, s[32:35], null idxen
buffer_load_d16_b16 v209, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[190:191], v2 offset:39312
ds_store_b64 v4, v[238:239] offset:33408
ds_store_b64 v4, v[240:241] offset:33792
buffer_load_d16_hi_b16 v202, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v210, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v203, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v211, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v204, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v212, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v201, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v209, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_fma_mix_f32 v244, v175, 1.0, v190 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v175, 1.0, v190 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v176, 1.0, v191 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v176, 1.0, v191 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v206, 1.0, v205 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v206, 1.0, v205 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v214, 1.0, v213 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v214, 1.0, v213 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:50], v[175:178], v[158:161], v[43:50]
v_wmma_f32_16x16x16_f16 v[47:54], v[175:178], v[160:163], v[47:54]
v_fma_mix_f32 v244, v180, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v180, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v181, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v181, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v234, v208, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v208, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v216, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v216, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[142:149], v[190:193], v[169:172], v[142:149]
v_wmma_f32_16x16x16_f16 v[146:153], v[190:193], v[171:174], v[146:153]
v_fma_mixlo_f16 v185, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v185, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v186, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[109:116], v[185:188], v[25:28], v[109:116]
v_wmma_f32_16x16x16_f16 v[113:120], v[185:188], v[167:170], v[113:120]
v_fma_mixlo_f16 v180, v244, 0.5, v180 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v180, v242, 0.5, v180 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v181, v245, 0.5, v181 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v181, v243, 0.5, v181 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v208 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v208 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v216 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v216 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[76:83], v[180:183], v[162:165], v[76:83]
v_wmma_f32_16x16x16_f16 v[80:87], v[180:183], v[164:167], v[80:87]
v_mov_b32_e32 v234, v206
v_mov_b32_e32 v236, v205
v_mov_b32_e32 v235, v214
v_mov_b32_e32 v237, v213
s_setprio 2
s_mov_b32 s51, 0x1eb8
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 2387
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[158:159], v1 offset:24768
ds_load_b64 v[160:161], v1 offset:24896
ds_store_b64 v4, v[234:235] offset:39296
ds_load_b64 v[162:163], v1 offset:28928
ds_load_b64 v[164:165], v1 offset:29056
ds_store_b64 v4, v[236:237] offset:40448
buffer_load_d16_b16 v206, v8, s[32:35], null idxen
buffer_load_d16_b16 v214, v12, s[32:35], null idxen
buffer_load_d16_b16 v207, v10, s[32:35], null idxen
buffer_load_d16_b16 v215, v14, s[32:35], null idxen
buffer_load_d16_b16 v208, v9, s[32:35], null idxen
buffer_load_d16_b16 v216, v13, s[32:35], null idxen
buffer_load_d16_b16 v205, v11, s[32:35], null idxen
buffer_load_d16_b16 v213, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[25:26], v1 offset:24776
ds_load_b64 v[167:168], v1 offset:24904
ds_store_b64 v4, v[238:239] offset:39680
ds_load_b64 v[169:170], v1 offset:28936
ds_load_b64 v[171:172], v1 offset:29064
ds_store_b64 v4, v[240:241] offset:40064
buffer_load_d16_hi_b16 v206, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v214, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v207, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v215, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v208, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v216, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v205, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v213, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_mov_b32 s51, 0x2014
s_sub_co_u32 s21, s21, 16
s_cbranch_scc1 63900
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v220, -1.0, v217 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v228, -1.0, v225 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:42], v[173:176], v[158:161], v[35:42]
v_wmma_f32_16x16x16_f16 v[39:46], v[173:176], v[160:163], v[39:46]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_pk_fma_f16 v236, v220, 1.0, v218 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v228, 1.0, v226 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[134:141], v[188:191], v[169:172], v[134:141]
v_wmma_f32_16x16x16_f16 v[138:145], v[188:191], v[171:174], v[138:145]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v238, v218, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v226, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[101:108], v[183:186], v[25:28], v[101:108]
v_wmma_f32_16x16x16_f16 v[105:112], v[183:186], v[167:170], v[105:112]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v218, -1.0, v219 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v226, -1.0, v227 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[68:75], v[178:181], v[162:165], v[68:75]
v_wmma_f32_16x16x16_f16 v[72:79], v[178:181], v[164:167], v[72:79]
s_setprio 2
s_mov_b32 s51, 0x213c
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 2346
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[173:174], v2 offset:33024
ds_store_b64 v3, v[234:235] offset:16512
ds_load_b64 v[178:179], v2 offset:33032
ds_store_b64 v3, v[236:237] offset:16768
buffer_load_d16_b16 v217, v16, s[36:39], null idxen
buffer_load_d16_b16 v225, v20, s[36:39], null idxen
buffer_load_d16_b16 v218, v17, s[36:39], null idxen
buffer_load_d16_b16 v226, v21, s[36:39], null idxen
buffer_load_d16_b16 v219, v18, s[36:39], null idxen
buffer_load_d16_b16 v227, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[188:189], v2 offset:33040
ds_store_b64 v3, v[238:239] offset:17024
ds_store_b64 v3, v[240:241] offset:17280
buffer_load_d16_hi_b16 v217, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v225, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v218, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v226, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v219, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v227, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 7
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
v_fma_mix_f32 v244, v173, 1.0, v188 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v173, 1.0, v188 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v174, 1.0, v189 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v174, 1.0, v189 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v224, -1.0, v221 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v232, -1.0, v229 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:58], v[175:178], v[158:161], v[51:58]
v_wmma_f32_16x16x16_f16 v[55:62], v[175:178], v[160:163], v[55:62]
v_fma_mix_f32 v244, v178, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v178, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v179, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v179, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v236, v224, 1.0, v222 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v232, 1.0, v230 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[150:157], v[190:193], v[169:172], v[150:157]
v_wmma_f32_16x16x16_f16 v[154:161], v[190:193], v[171:174], v[154:161]
v_fma_mixlo_f16 v183, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v183, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v184, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v184, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_pk_fma_f16 v238, v222, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v230, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[117:124], v[185:188], v[25:28], v[117:124]
v_wmma_f32_16x16x16_f16 v[121:128], v[185:188], v[167:170], v[121:128]
v_fma_mixlo_f16 v178, v244, 0.5, v178 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v178, v242, 0.5, v178 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v179, v245, 0.5, v179 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v179, v243, 0.5, v179 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v240, v222, -1.0, v223 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v230, -1.0, v231 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[84:91], v[180:183], v[162:165], v[84:91]
v_wmma_f32_16x16x16_f16 v[88:95], v[180:183], v[164:167], v[88:95]
s_setprio 2
s_mov_b32 s51, 0x2368
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 2327
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[158:159], v1 offset:16512
ds_load_b64 v[160:161], v1 offset:16640
ds_store_b64 v3, v[234:235] offset:24768
ds_load_b64 v[162:163], v1 offset:20672
ds_load_b64 v[164:165], v1 offset:20800
ds_store_b64 v3, v[236:237] offset:25024
buffer_load_d16_b16 v221, v16, s[36:39], null idxen
buffer_load_d16_b16 v229, v20, s[36:39], null idxen
buffer_load_d16_b16 v222, v17, s[36:39], null idxen
buffer_load_d16_b16 v230, v21, s[36:39], null idxen
buffer_load_d16_b16 v223, v18, s[36:39], null idxen
buffer_load_d16_b16 v231, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[25:26], v1 offset:16520
ds_load_b64 v[167:168], v1 offset:16648
ds_store_b64 v3, v[238:239] offset:25280
ds_load_b64 v[169:170], v1 offset:20680
ds_load_b64 v[171:172], v1 offset:20808
ds_store_b64 v3, v[240:241] offset:25536
buffer_load_d16_hi_b16 v221, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v229, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v222, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v230, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v223, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v231, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 63642
s_nop 0
s_nop 0
s_lshl_b32 s43, s43, 2
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v125, 1.0, v59 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v125, 1.0, v59 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v177, 1.0, v187 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v177, 1.0, v187 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[27:34], v[173:176], v[158:161], v[27:34]
v_wmma_f32_16x16x16_f16 v[31:38], v[173:176], v[160:163], v[31:38]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v166, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v166, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v182, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v182, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[126:133], v[188:191], v[169:172], v[126:133]
v_wmma_f32_16x16x16_f16 v[130:137], v[188:191], v[171:174], v[130:137]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[93:100], v[183:186], v[25:28], v[93:100]
v_wmma_f32_16x16x16_f16 v[97:104], v[183:186], v[167:170], v[97:104]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v166 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v166 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v182 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v182 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[60:67], v[178:181], v[162:165], v[60:67]
v_wmma_f32_16x16x16_f16 v[64:71], v[178:181], v[164:167], v[64:71]
v_mov_b32_e32 v234, v125
v_mov_b32_e32 v236, v59
v_mov_b32_e32 v235, v177
v_mov_b32_e32 v237, v187
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x2600
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 1801
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[175:176], v2 offset:39296
ds_store_b64 v4, v[234:235] offset:33024
ds_load_b64 v[180:181], v2 offset:39304
ds_store_b64 v4, v[236:237] offset:34176
buffer_load_d16_b16 v125, v8, s[32:35], null idxen
buffer_load_d16_b16 v177, v12, s[32:35], null idxen
buffer_load_d16_b16 v92, v10, s[32:35], null idxen
buffer_load_d16_b16 v192, v14, s[32:35], null idxen
buffer_load_d16_b16 v166, v9, s[32:35], null idxen
buffer_load_d16_b16 v182, v13, s[32:35], null idxen
buffer_load_d16_b16 v59, v11, s[32:35], null idxen
buffer_load_d16_b16 v187, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[190:191], v2 offset:39312
ds_store_b64 v4, v[238:239] offset:33408
ds_store_b64 v4, v[240:241] offset:33792
buffer_load_d16_hi_b16 v125, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v177, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v92, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v192, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v166, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v182, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v59, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v187, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
v_fma_mix_f32 v244, v175, 1.0, v190 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v175, 1.0, v190 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v176, 1.0, v191 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v176, 1.0, v191 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v193, 1.0, v195 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v193, 1.0, v195 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v197, 1.0, v199 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v197, 1.0, v199 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:50], v[175:178], v[158:161], v[43:50]
v_wmma_f32_16x16x16_f16 v[47:54], v[175:178], v[160:163], v[47:54]
v_fma_mix_f32 v244, v180, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v180, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v181, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v181, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v234, v194, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v194, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v198, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v198, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[142:149], v[190:193], v[169:172], v[142:149]
v_wmma_f32_16x16x16_f16 v[146:153], v[190:193], v[171:174], v[146:153]
v_fma_mixlo_f16 v185, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v185, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v186, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[109:116], v[185:188], v[25:28], v[109:116]
v_wmma_f32_16x16x16_f16 v[113:120], v[185:188], v[167:170], v[113:120]
v_fma_mixlo_f16 v180, v244, 0.5, v180 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v180, v242, 0.5, v180 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v181, v245, 0.5, v181 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v181, v243, 0.5, v181 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v194 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v194 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v198 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v198 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[76:83], v[180:183], v[162:165], v[76:83]
v_wmma_f32_16x16x16_f16 v[80:87], v[180:183], v[164:167], v[80:87]
v_mov_b32_e32 v234, v193
v_mov_b32_e32 v236, v195
v_mov_b32_e32 v235, v197
v_mov_b32_e32 v237, v199
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x2898
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 1755
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[158:159], v1 offset:24768
ds_load_b64 v[160:161], v1 offset:24896
ds_store_b64 v4, v[234:235] offset:39296
ds_load_b64 v[162:163], v1 offset:28928
ds_load_b64 v[164:165], v1 offset:29056
ds_store_b64 v4, v[236:237] offset:40448
buffer_load_d16_b16 v193, v8, s[32:35], null idxen
buffer_load_d16_b16 v197, v12, s[32:35], null idxen
buffer_load_d16_b16 v196, v10, s[32:35], null idxen
buffer_load_d16_b16 v200, v14, s[32:35], null idxen
buffer_load_d16_b16 v194, v9, s[32:35], null idxen
buffer_load_d16_b16 v198, v13, s[32:35], null idxen
buffer_load_d16_b16 v195, v11, s[32:35], null idxen
buffer_load_d16_b16 v199, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[25:26], v1 offset:24776
ds_load_b64 v[167:168], v1 offset:24904
ds_store_b64 v4, v[238:239] offset:39680
ds_load_b64 v[169:170], v1 offset:28936
ds_load_b64 v[171:172], v1 offset:29064
ds_store_b64 v4, v[240:241] offset:40064
buffer_load_d16_hi_b16 v193, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v197, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v196, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v200, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v194, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v198, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v195, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v199, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_mov_b32 s51, 0x29f4
s_sub_co_u32 s21, s21, 16
s_cbranch_scc1 63268
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v203, -1.0, v202 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v207, -1.0, v206 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:42], v[173:176], v[158:161], v[35:42]
v_wmma_f32_16x16x16_f16 v[39:46], v[173:176], v[160:163], v[39:46]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_pk_fma_f16 v236, v203, 1.0, v204 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v207, 1.0, v208 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[134:141], v[188:191], v[169:172], v[134:141]
v_wmma_f32_16x16x16_f16 v[138:145], v[188:191], v[171:174], v[138:145]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v238, v204, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v208, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[101:108], v[183:186], v[25:28], v[101:108]
v_wmma_f32_16x16x16_f16 v[105:112], v[183:186], v[167:170], v[105:112]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v204, -1.0, v201 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v208, -1.0, v205 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[68:75], v[178:181], v[162:165], v[68:75]
v_wmma_f32_16x16x16_f16 v[72:79], v[178:181], v[164:167], v[72:79]
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x2b1c
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 1714
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[173:174], v2 offset:33024
ds_store_b64 v3, v[234:235] offset:16512
ds_load_b64 v[178:179], v2 offset:33032
ds_store_b64 v3, v[236:237] offset:16768
buffer_load_d16_b16 v202, v16, s[36:39], null idxen
buffer_load_d16_b16 v206, v20, s[36:39], null idxen
buffer_load_d16_b16 v204, v17, s[36:39], null idxen
buffer_load_d16_b16 v208, v21, s[36:39], null idxen
buffer_load_d16_b16 v201, v18, s[36:39], null idxen
buffer_load_d16_b16 v205, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[188:189], v2 offset:33040
ds_store_b64 v3, v[238:239] offset:17024
ds_store_b64 v3, v[240:241] offset:17280
buffer_load_d16_hi_b16 v202, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v206, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v204, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v208, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v201, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v205, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 7
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
v_fma_mix_f32 v244, v173, 1.0, v188 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v173, 1.0, v188 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v174, 1.0, v189 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v174, 1.0, v189 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v211, -1.0, v210 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v215, -1.0, v214 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:58], v[175:178], v[158:161], v[51:58]
v_wmma_f32_16x16x16_f16 v[55:62], v[175:178], v[160:163], v[55:62]
v_fma_mix_f32 v244, v178, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v178, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v179, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v179, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v236, v211, 1.0, v212 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v215, 1.0, v216 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[150:157], v[190:193], v[169:172], v[150:157]
v_wmma_f32_16x16x16_f16 v[154:161], v[190:193], v[171:174], v[154:161]
v_fma_mixlo_f16 v183, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v183, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v184, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v184, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_pk_fma_f16 v238, v212, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v216, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[117:124], v[185:188], v[25:28], v[117:124]
v_wmma_f32_16x16x16_f16 v[121:128], v[185:188], v[167:170], v[121:128]
v_fma_mixlo_f16 v178, v244, 0.5, v178 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v178, v242, 0.5, v178 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v179, v245, 0.5, v179 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v179, v243, 0.5, v179 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v240, v212, -1.0, v209 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v216, -1.0, v213 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[84:91], v[180:183], v[162:165], v[84:91]
v_wmma_f32_16x16x16_f16 v[88:95], v[180:183], v[164:167], v[88:95]
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x2d48
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 1695
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[158:159], v1 offset:16512
ds_load_b64 v[160:161], v1 offset:16640
ds_store_b64 v3, v[234:235] offset:24768
ds_load_b64 v[162:163], v1 offset:20672
ds_load_b64 v[164:165], v1 offset:20800
ds_store_b64 v3, v[236:237] offset:25024
buffer_load_d16_b16 v210, v16, s[36:39], null idxen
buffer_load_d16_b16 v214, v20, s[36:39], null idxen
buffer_load_d16_b16 v212, v17, s[36:39], null idxen
buffer_load_d16_b16 v216, v21, s[36:39], null idxen
buffer_load_d16_b16 v209, v18, s[36:39], null idxen
buffer_load_d16_b16 v213, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[25:26], v1 offset:16520
ds_load_b64 v[167:168], v1 offset:16648
ds_store_b64 v3, v[238:239] offset:25280
ds_load_b64 v[169:170], v1 offset:20680
ds_load_b64 v[171:172], v1 offset:20808
ds_store_b64 v3, v[240:241] offset:25536
buffer_load_d16_hi_b16 v210, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v214, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v212, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v216, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v209, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v213, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 2
s_nop 0
s_nop 0
s_lshl_b32 s43, s43, 2
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v217, 1.0, v219 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v217, 1.0, v219 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v221, 1.0, v223 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v221, 1.0, v223 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[27:34], v[173:176], v[158:161], v[27:34]
v_wmma_f32_16x16x16_f16 v[31:38], v[173:176], v[160:163], v[31:38]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v218, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v218, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v222, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v222, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[126:133], v[188:191], v[169:172], v[126:133]
v_wmma_f32_16x16x16_f16 v[130:137], v[188:191], v[171:174], v[130:137]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[93:100], v[183:186], v[25:28], v[93:100]
v_wmma_f32_16x16x16_f16 v[97:104], v[183:186], v[167:170], v[97:104]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v218 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v218 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v222 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v222 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[60:67], v[178:181], v[162:165], v[60:67]
v_wmma_f32_16x16x16_f16 v[64:71], v[178:181], v[164:167], v[64:71]
v_mov_b32_e32 v234, v217
v_mov_b32_e32 v236, v219
v_mov_b32_e32 v235, v221
v_mov_b32_e32 v237, v223
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x2fe0
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 1169
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[175:176], v2 offset:39296
ds_store_b64 v4, v[234:235] offset:33024
ds_load_b64 v[180:181], v2 offset:39304
ds_store_b64 v4, v[236:237] offset:34176
buffer_load_d16_b16 v217, v8, s[32:35], null idxen
buffer_load_d16_b16 v221, v12, s[32:35], null idxen
buffer_load_d16_b16 v220, v10, s[32:35], null idxen
buffer_load_d16_b16 v224, v14, s[32:35], null idxen
buffer_load_d16_b16 v218, v9, s[32:35], null idxen
buffer_load_d16_b16 v222, v13, s[32:35], null idxen
buffer_load_d16_b16 v219, v11, s[32:35], null idxen
buffer_load_d16_b16 v223, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[190:191], v2 offset:39312
ds_store_b64 v4, v[238:239] offset:33408
ds_store_b64 v4, v[240:241] offset:33792
buffer_load_d16_hi_b16 v217, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v221, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v220, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v224, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v218, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v222, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v219, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v223, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
v_fma_mix_f32 v244, v175, 1.0, v190 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v175, 1.0, v190 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v176, 1.0, v191 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v176, 1.0, v191 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v225, 1.0, v227 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v225, 1.0, v227 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v229, 1.0, v231 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v229, 1.0, v231 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:50], v[175:178], v[158:161], v[43:50]
v_wmma_f32_16x16x16_f16 v[47:54], v[175:178], v[160:163], v[47:54]
v_fma_mix_f32 v244, v180, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v180, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v181, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v181, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v234, v226, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v226, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v230, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v230, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[142:149], v[190:193], v[169:172], v[142:149]
v_wmma_f32_16x16x16_f16 v[146:153], v[190:193], v[171:174], v[146:153]
v_fma_mixlo_f16 v185, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v185, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v186, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[109:116], v[185:188], v[25:28], v[109:116]
v_wmma_f32_16x16x16_f16 v[113:120], v[185:188], v[167:170], v[113:120]
v_fma_mixlo_f16 v180, v244, 0.5, v180 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v180, v242, 0.5, v180 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v181, v245, 0.5, v181 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v181, v243, 0.5, v181 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v226 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v226 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v230 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v230 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[76:83], v[180:183], v[162:165], v[76:83]
v_wmma_f32_16x16x16_f16 v[80:87], v[180:183], v[164:167], v[80:87]
v_mov_b32_e32 v234, v225
v_mov_b32_e32 v236, v227
v_mov_b32_e32 v235, v229
v_mov_b32_e32 v237, v231
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x3278
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 1123
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[158:159], v1 offset:24768
ds_load_b64 v[160:161], v1 offset:24896
ds_store_b64 v4, v[234:235] offset:39296
ds_load_b64 v[162:163], v1 offset:28928
ds_load_b64 v[164:165], v1 offset:29056
ds_store_b64 v4, v[236:237] offset:40448
buffer_load_d16_b16 v225, v8, s[32:35], null idxen
buffer_load_d16_b16 v229, v12, s[32:35], null idxen
buffer_load_d16_b16 v228, v10, s[32:35], null idxen
buffer_load_d16_b16 v232, v14, s[32:35], null idxen
buffer_load_d16_b16 v226, v9, s[32:35], null idxen
buffer_load_d16_b16 v230, v13, s[32:35], null idxen
buffer_load_d16_b16 v227, v11, s[32:35], null idxen
buffer_load_d16_b16 v231, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[25:26], v1 offset:24776
ds_load_b64 v[167:168], v1 offset:24904
ds_store_b64 v4, v[238:239] offset:39680
ds_load_b64 v[169:170], v1 offset:28936
ds_load_b64 v[171:172], v1 offset:29064
ds_store_b64 v4, v[240:241] offset:40064
buffer_load_d16_hi_b16 v225, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v229, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v228, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v232, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v226, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v230, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v227, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v231, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_mov_b32 s51, 0x33d4
s_sub_co_u32 s21, s21, 16
s_cbranch_scc1 62636
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v92, -1.0, v125 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v196, -1.0, v193 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:42], v[173:176], v[158:161], v[35:42]
v_wmma_f32_16x16x16_f16 v[39:46], v[173:176], v[160:163], v[39:46]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_pk_fma_f16 v236, v92, 1.0, v166 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v196, 1.0, v194 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[134:141], v[188:191], v[169:172], v[134:141]
v_wmma_f32_16x16x16_f16 v[138:145], v[188:191], v[171:174], v[138:145]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v238, v166, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v194, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[101:108], v[183:186], v[25:28], v[101:108]
v_wmma_f32_16x16x16_f16 v[105:112], v[183:186], v[167:170], v[105:112]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v166, -1.0, v59 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v194, -1.0, v195 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[68:75], v[178:181], v[162:165], v[68:75]
v_wmma_f32_16x16x16_f16 v[72:79], v[178:181], v[164:167], v[72:79]
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x34fc
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 1082
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[173:174], v2 offset:33024
ds_store_b64 v3, v[234:235] offset:16512
ds_load_b64 v[178:179], v2 offset:33032
ds_store_b64 v3, v[236:237] offset:16768
buffer_load_d16_b16 v125, v16, s[36:39], null idxen
buffer_load_d16_b16 v193, v20, s[36:39], null idxen
buffer_load_d16_b16 v166, v17, s[36:39], null idxen
buffer_load_d16_b16 v194, v21, s[36:39], null idxen
buffer_load_d16_b16 v59, v18, s[36:39], null idxen
buffer_load_d16_b16 v195, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[188:189], v2 offset:33040
ds_store_b64 v3, v[238:239] offset:17024
ds_store_b64 v3, v[240:241] offset:17280
buffer_load_d16_hi_b16 v125, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v193, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v166, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v194, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v59, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v195, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 7
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
v_fma_mix_f32 v244, v173, 1.0, v188 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v173, 1.0, v188 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v174, 1.0, v189 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v174, 1.0, v189 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v192, -1.0, v177 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v200, -1.0, v197 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:58], v[175:178], v[158:161], v[51:58]
v_wmma_f32_16x16x16_f16 v[55:62], v[175:178], v[160:163], v[55:62]
v_fma_mix_f32 v244, v178, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v178, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v179, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v179, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v236, v192, 1.0, v182 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v200, 1.0, v198 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[150:157], v[190:193], v[169:172], v[150:157]
v_wmma_f32_16x16x16_f16 v[154:161], v[190:193], v[171:174], v[154:161]
v_fma_mixlo_f16 v183, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v183, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v184, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v184, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_pk_fma_f16 v238, v182, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v198, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[117:124], v[185:188], v[25:28], v[117:124]
v_wmma_f32_16x16x16_f16 v[121:128], v[185:188], v[167:170], v[121:128]
v_fma_mixlo_f16 v178, v244, 0.5, v178 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v178, v242, 0.5, v178 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v179, v245, 0.5, v179 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v179, v243, 0.5, v179 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v240, v182, -1.0, v187 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v198, -1.0, v199 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[84:91], v[180:183], v[162:165], v[84:91]
v_wmma_f32_16x16x16_f16 v[88:95], v[180:183], v[164:167], v[88:95]
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x3728
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 1063
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[158:159], v1 offset:16512
ds_load_b64 v[160:161], v1 offset:16640
ds_store_b64 v3, v[234:235] offset:24768
ds_load_b64 v[162:163], v1 offset:20672
ds_load_b64 v[164:165], v1 offset:20800
ds_store_b64 v3, v[236:237] offset:25024
buffer_load_d16_b16 v177, v16, s[36:39], null idxen
buffer_load_d16_b16 v197, v20, s[36:39], null idxen
buffer_load_d16_b16 v182, v17, s[36:39], null idxen
buffer_load_d16_b16 v198, v21, s[36:39], null idxen
buffer_load_d16_b16 v187, v18, s[36:39], null idxen
buffer_load_d16_b16 v199, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[25:26], v1 offset:16520
ds_load_b64 v[167:168], v1 offset:16648
ds_store_b64 v3, v[238:239] offset:25280
ds_load_b64 v[169:170], v1 offset:20680
ds_load_b64 v[171:172], v1 offset:20808
ds_store_b64 v3, v[240:241] offset:25536
buffer_load_d16_hi_b16 v177, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v197, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v182, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v198, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v187, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v199, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 2
s_nop 0
s_nop 0
s_lshl_b32 s43, s43, 2
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v202, 1.0, v201 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v202, 1.0, v201 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v210, 1.0, v209 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v210, 1.0, v209 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[27:34], v[173:176], v[158:161], v[27:34]
v_wmma_f32_16x16x16_f16 v[31:38], v[173:176], v[160:163], v[31:38]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v204, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v204, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v212, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v212, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[126:133], v[188:191], v[169:172], v[126:133]
v_wmma_f32_16x16x16_f16 v[130:137], v[188:191], v[171:174], v[130:137]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[93:100], v[183:186], v[25:28], v[93:100]
v_wmma_f32_16x16x16_f16 v[97:104], v[183:186], v[167:170], v[97:104]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v204 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v204 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v212 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v212 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[60:67], v[178:181], v[162:165], v[60:67]
v_wmma_f32_16x16x16_f16 v[64:71], v[178:181], v[164:167], v[64:71]
v_mov_b32_e32 v234, v202
v_mov_b32_e32 v236, v201
v_mov_b32_e32 v235, v210
v_mov_b32_e32 v237, v209
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x39c0
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 537
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[175:176], v2 offset:39296
ds_store_b64 v4, v[234:235] offset:33024
ds_load_b64 v[180:181], v2 offset:39304
ds_store_b64 v4, v[236:237] offset:34176
buffer_load_d16_b16 v202, v8, s[32:35], null idxen
buffer_load_d16_b16 v210, v12, s[32:35], null idxen
buffer_load_d16_b16 v203, v10, s[32:35], null idxen
buffer_load_d16_b16 v211, v14, s[32:35], null idxen
buffer_load_d16_b16 v204, v9, s[32:35], null idxen
buffer_load_d16_b16 v212, v13, s[32:35], null idxen
buffer_load_d16_b16 v201, v11, s[32:35], null idxen
buffer_load_d16_b16 v209, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[190:191], v2 offset:39312
ds_store_b64 v4, v[238:239] offset:33408
ds_store_b64 v4, v[240:241] offset:33792
buffer_load_d16_hi_b16 v202, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v210, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v203, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v211, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v204, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v212, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v201, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v209, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
v_fma_mix_f32 v244, v175, 1.0, v190 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v175, 1.0, v190 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v176, 1.0, v191 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v176, 1.0, v191 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v234, v206, 1.0, v205 op_sel_hi:[1,0,1]
v_fma_mix_f32 v236, v206, 1.0, v205 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v235, v214, 1.0, v213 op_sel_hi:[1,0,1]
v_fma_mix_f32 v237, v214, 1.0, v213 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[43:50], v[175:178], v[158:161], v[43:50]
v_wmma_f32_16x16x16_f16 v[47:54], v[175:178], v[160:163], v[47:54]
v_fma_mix_f32 v244, v180, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v180, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v181, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v181, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v234, v208, -1.0, v234 op_sel_hi:[1,0,0]
v_fma_mix_f32 v236, v208, -1.0, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v235, v216, -1.0, v235 op_sel_hi:[1,0,0]
v_fma_mix_f32 v237, v216, -1.0, v237 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_wmma_f32_16x16x16_f16 v[142:149], v[190:193], v[169:172], v[142:149]
v_wmma_f32_16x16x16_f16 v[146:153], v[190:193], v[171:174], v[146:153]
v_fma_mixlo_f16 v185, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v185, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v186, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v186, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v240, v234, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v240, v236, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v241, v235, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v241, v237, 0.5, 0 op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[109:116], v[185:188], v[25:28], v[109:116]
v_wmma_f32_16x16x16_f16 v[113:120], v[185:188], v[167:170], v[113:120]
v_fma_mixlo_f16 v180, v244, 0.5, v180 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v180, v242, 0.5, v180 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v181, v245, 0.5, v181 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v181, v243, 0.5, v181 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v238, v234, 0.5, v208 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v238, v236, 0.5, v208 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v239, v235, 0.5, v216 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v239, v237, 0.5, v216 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_wmma_f32_16x16x16_f16 v[76:83], v[180:183], v[162:165], v[76:83]
v_wmma_f32_16x16x16_f16 v[80:87], v[180:183], v[164:167], v[80:87]
v_mov_b32_e32 v234, v206
v_mov_b32_e32 v236, v205
v_mov_b32_e32 v235, v214
v_mov_b32_e32 v237, v213
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x3c58
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 491
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[158:159], v1 offset:24768
ds_load_b64 v[160:161], v1 offset:24896
ds_store_b64 v4, v[234:235] offset:39296
ds_load_b64 v[162:163], v1 offset:28928
ds_load_b64 v[164:165], v1 offset:29056
ds_store_b64 v4, v[236:237] offset:40448
buffer_load_d16_b16 v206, v8, s[32:35], null idxen
buffer_load_d16_b16 v214, v12, s[32:35], null idxen
buffer_load_d16_b16 v207, v10, s[32:35], null idxen
buffer_load_d16_b16 v215, v14, s[32:35], null idxen
buffer_load_d16_b16 v208, v9, s[32:35], null idxen
buffer_load_d16_b16 v216, v13, s[32:35], null idxen
buffer_load_d16_b16 v205, v11, s[32:35], null idxen
buffer_load_d16_b16 v213, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_sub_co_u32 s2, s2, 4
s_cselect_b32 s34, 0, s34
ds_load_b64 v[25:26], v1 offset:24776
ds_load_b64 v[167:168], v1 offset:24904
ds_store_b64 v4, v[238:239] offset:39680
ds_load_b64 v[169:170], v1 offset:28936
ds_load_b64 v[171:172], v1 offset:29064
ds_store_b64 v4, v[240:241] offset:40064
buffer_load_d16_hi_b16 v206, v8, s[32:35], null idxen
buffer_load_d16_hi_b16 v214, v12, s[32:35], null idxen
buffer_load_d16_hi_b16 v207, v10, s[32:35], null idxen
buffer_load_d16_hi_b16 v215, v14, s[32:35], null idxen
buffer_load_d16_hi_b16 v208, v9, s[32:35], null idxen
buffer_load_d16_hi_b16 v216, v13, s[32:35], null idxen
buffer_load_d16_hi_b16 v205, v11, s[32:35], null idxen
buffer_load_d16_hi_b16 v213, v15, s[32:35], null idxen
s_add_co_u32 s32, s32, s54
s_add_co_ci_u32 s33, s33, 0
s_branch 2
s_nop 0
s_nop 0
s_mov_b32 s51, 0x3db4
s_sub_co_u32 s21, s21, 16
s_cbranch_scc1 62004
s_setprio 1
s_wait_loadcnt 0x38
s_wait_dscnt 0x0
v_pk_fma_f16 v158, v25, -1.0, v158 op_sel_hi:[1,0,1]
v_pk_fma_f16 v159, v26, -1.0, v159 op_sel_hi:[1,0,1]
v_pk_fma_f16 v160, v167, -1.0, v160 op_sel_hi:[1,0,1]
v_pk_fma_f16 v161, v168, -1.0, v161 op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v220, -1.0, v217 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v228, -1.0, v225 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[35:42], v[173:176], v[158:161], v[35:42]
v_wmma_f32_16x16x16_f16 v[39:46], v[173:176], v[160:163], v[39:46]
v_pk_fma_f16 v169, v162, -1.0, v169 op_sel_hi:[1,0,1]
v_pk_fma_f16 v170, v163, -1.0, v170 op_sel_hi:[1,0,1]
v_pk_fma_f16 v171, v164, -1.0, v171 op_sel_hi:[1,0,1]
v_pk_fma_f16 v172, v165, -1.0, v172 op_sel_hi:[1,0,1]
v_pk_fma_f16 v236, v220, 1.0, v218 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v228, 1.0, v226 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[134:141], v[188:191], v[169:172], v[134:141]
v_wmma_f32_16x16x16_f16 v[138:145], v[188:191], v[171:174], v[138:145]
v_pk_fma_f16 v25, v162, -1.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v26, v163, -1.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v167, v164, -1.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v168, v165, -1.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v238, v218, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v226, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[101:108], v[183:186], v[25:28], v[101:108]
v_wmma_f32_16x16x16_f16 v[105:112], v[183:186], v[167:170], v[105:112]
v_pk_fma_f16 v162, v162, 2.0, v25 op_sel_hi:[1,0,1]
v_pk_fma_f16 v163, v163, 2.0, v26 op_sel_hi:[1,0,1]
v_pk_fma_f16 v164, v164, 2.0, v167 op_sel_hi:[1,0,1]
v_pk_fma_f16 v165, v165, 2.0, v168 op_sel_hi:[1,0,1]
v_pk_fma_f16 v240, v218, -1.0, v219 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v226, -1.0, v227 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[68:75], v[178:181], v[162:165], v[68:75]
v_wmma_f32_16x16x16_f16 v[72:79], v[178:181], v[164:167], v[72:79]
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x3edc
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 450
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[173:174], v2 offset:33024
ds_store_b64 v3, v[234:235] offset:16512
ds_load_b64 v[178:179], v2 offset:33032
ds_store_b64 v3, v[236:237] offset:16768
buffer_load_d16_b16 v217, v16, s[36:39], null idxen
buffer_load_d16_b16 v225, v20, s[36:39], null idxen
buffer_load_d16_b16 v218, v17, s[36:39], null idxen
buffer_load_d16_b16 v226, v21, s[36:39], null idxen
buffer_load_d16_b16 v219, v18, s[36:39], null idxen
buffer_load_d16_b16 v227, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[188:189], v2 offset:33040
ds_store_b64 v3, v[238:239] offset:17024
ds_store_b64 v3, v[240:241] offset:17280
buffer_load_d16_hi_b16 v217, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v225, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v218, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v226, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v219, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v227, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 7
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_setprio 1
s_wait_dscnt 0x0
v_fma_mix_f32 v244, v173, 1.0, v188 op_sel_hi:[1,0,1]
v_fma_mix_f32 v242, v173, 1.0, v188 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v245, v174, 1.0, v189 op_sel_hi:[1,0,1]
v_fma_mix_f32 v243, v174, 1.0, v189 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_pk_fma_f16 v234, v224, -1.0, v221 op_sel_hi:[1,0,1]
v_pk_fma_f16 v235, v232, -1.0, v229 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[51:58], v[175:178], v[158:161], v[51:58]
v_wmma_f32_16x16x16_f16 v[55:62], v[175:178], v[160:163], v[55:62]
v_fma_mix_f32 v244, v178, -1.0, v244 op_sel_hi:[1,0,0]
v_fma_mix_f32 v242, v178, -1.0, v242 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_fma_mix_f32 v245, v179, -1.0, v245 op_sel_hi:[1,0,0]
v_fma_mix_f32 v243, v179, -1.0, v243 op_sel:[1,0,0] op_sel_hi:[1,0,0]
v_pk_fma_f16 v236, v224, 1.0, v222 op_sel_hi:[1,0,1]
v_pk_fma_f16 v237, v232, 1.0, v230 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[150:157], v[190:193], v[169:172], v[150:157]
v_wmma_f32_16x16x16_f16 v[154:161], v[190:193], v[171:174], v[154:161]
v_fma_mixlo_f16 v183, v244, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v183, v242, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v184, v245, 0.5, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v184, v243, 0.5, 0 op_sel_hi:[0,0,1]
v_pk_fma_f16 v238, v222, -2.0, v236 op_sel_hi:[1,0,1]
v_pk_fma_f16 v239, v230, -2.0, v237 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[117:124], v[185:188], v[25:28], v[117:124]
v_wmma_f32_16x16x16_f16 v[121:128], v[185:188], v[167:170], v[121:128]
v_fma_mixlo_f16 v178, v244, 0.5, v178 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v178, v242, 0.5, v178 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v179, v245, 0.5, v179 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v179, v243, 0.5, v179 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_pk_fma_f16 v240, v222, -1.0, v223 op_sel_hi:[1,0,1]
v_pk_fma_f16 v241, v230, -1.0, v231 op_sel_hi:[1,0,1]
v_wmma_f32_16x16x16_f16 v[84:91], v[180:183], v[162:165], v[84:91]
v_wmma_f32_16x16x16_f16 v[88:95], v[180:183], v[164:167], v[88:95]
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 2
s_mov_b32 s51, 0x4108
s_bitcmp1_b32 s43, 5
s_cbranch_scc1 431
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[158:159], v1 offset:16512
ds_load_b64 v[160:161], v1 offset:16640
ds_store_b64 v3, v[234:235] offset:24768
ds_load_b64 v[162:163], v1 offset:20672
ds_load_b64 v[164:165], v1 offset:20800
ds_store_b64 v3, v[236:237] offset:25024
buffer_load_d16_b16 v221, v16, s[36:39], null idxen
buffer_load_d16_b16 v229, v20, s[36:39], null idxen
buffer_load_d16_b16 v222, v17, s[36:39], null idxen
buffer_load_d16_b16 v230, v21, s[36:39], null idxen
buffer_load_d16_b16 v223, v18, s[36:39], null idxen
buffer_load_d16_b16 v231, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_sub_co_u32 s3, s3, 4
s_cselect_b32 s38, 0, s38
ds_load_b64 v[25:26], v1 offset:16520
ds_load_b64 v[167:168], v1 offset:16648
ds_store_b64 v3, v[238:239] offset:25280
ds_load_b64 v[169:170], v1 offset:20680
ds_load_b64 v[171:172], v1 offset:20808
ds_store_b64 v3, v[240:241] offset:25536
buffer_load_d16_hi_b16 v221, v16, s[36:39], null idxen
buffer_load_d16_hi_b16 v229, v20, s[36:39], null idxen
buffer_load_d16_hi_b16 v222, v17, s[36:39], null idxen
buffer_load_d16_hi_b16 v230, v21, s[36:39], null idxen
buffer_load_d16_hi_b16 v223, v18, s[36:39], null idxen
buffer_load_d16_hi_b16 v231, v22, s[36:39], null idxen
s_add_co_u32 s36, s36, s55
s_add_co_ci_u32 s37, s37, 0
s_branch 63642
s_nop 0
s_nop 0
v_add_f32_e32 v27, v27, v60
v_add_f32_e32 v28, v28, v61
v_add_f32_e32 v29, v29, v62
v_add_f32_e32 v30, v30, v63
v_add_f32_e32 v31, v31, v64
v_add_f32_e32 v32, v32, v65
v_add_f32_e32 v33, v33, v66
v_add_f32_e32 v34, v34, v67
v_add_f32_e64 v126, v126, -v93
v_add_f32_e64 v127, v127, -v94
v_add_f32_e64 v128, v128, -v95
v_add_f32_e64 v129, v129, -v96
v_add_f32_e64 v130, v130, -v97
v_add_f32_e64 v131, v131, -v98
v_add_f32_e64 v132, v132, -v99
v_add_f32_e64 v133, v133, -v100
v_add_f32_e32 v27, v27, v93
v_add_f32_e32 v28, v28, v94
v_add_f32_e32 v29, v29, v95
v_add_f32_e32 v30, v30, v96
v_add_f32_e32 v31, v31, v97
v_add_f32_e32 v32, v32, v98
v_add_f32_e32 v33, v33, v99
v_add_f32_e32 v34, v34, v100
v_add_f32_e32 v126, v126, v60
v_add_f32_e32 v127, v127, v61
v_add_f32_e32 v128, v128, v62
v_add_f32_e32 v129, v129, v63
v_add_f32_e32 v130, v130, v64
v_add_f32_e32 v131, v131, v65
v_add_f32_e32 v132, v132, v66
v_add_f32_e32 v133, v133, v67
v_fma_mixlo_f16 v244, v27, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v28, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v29, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v30, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v248, v31, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v248, v32, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v249, v33, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v249, v34, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v242, v126, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v242, v127, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v243, v128, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v243, v129, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v130, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v131, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v132, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v133, 1.0, 0 op_sel_hi:[0,0,1]
v_mov_b32_e32 v27, 0
v_mov_b32_e32 v28, 0
v_mov_b32_e32 v29, 0
v_mov_b32_e32 v30, 0
v_mov_b32_e32 v31, 0
v_mov_b32_e32 v32, 0
v_mov_b32_e32 v33, 0
v_mov_b32_e32 v34, 0
v_mov_b32_e32 v60, 0
v_mov_b32_e32 v61, 0
v_mov_b32_e32 v62, 0
v_mov_b32_e32 v63, 0
v_mov_b32_e32 v64, 0
v_mov_b32_e32 v65, 0
v_mov_b32_e32 v66, 0
v_mov_b32_e32 v67, 0
v_mov_b32_e32 v93, 0
v_mov_b32_e32 v94, 0
v_mov_b32_e32 v95, 0
v_mov_b32_e32 v96, 0
v_mov_b32_e32 v97, 0
v_mov_b32_e32 v98, 0
v_mov_b32_e32 v99, 0
v_mov_b32_e32 v100, 0
v_mov_b32_e32 v126, 0
v_mov_b32_e32 v127, 0
v_mov_b32_e32 v128, 0
v_mov_b32_e32 v129, 0
v_mov_b32_e32 v130, 0
v_mov_b32_e32 v131, 0
v_mov_b32_e32 v132, 0
v_mov_b32_e32 v133, 0
ds_store_b64 v1, v[244:245] offset:4160
ds_store_b64 v1, v[248:249] offset:4288
ds_store_b64 v1, v[242:243] offset:8
ds_store_b64 v1, v[246:247] offset:136
s_add_co_u32 s52, s4, s51
s_add_co_ci_u32 s53, s5, 0
s_setpc_b64 s[52:53]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_e32 v43, v43, v76
v_add_f32_e32 v44, v44, v77
v_add_f32_e32 v45, v45, v78
v_add_f32_e32 v46, v46, v79
v_add_f32_e32 v47, v47, v80
v_add_f32_e32 v48, v48, v81
v_add_f32_e32 v49, v49, v82
v_add_f32_e32 v50, v50, v83
v_add_f32_e64 v142, v142, -v109
v_add_f32_e64 v143, v143, -v110
v_add_f32_e64 v144, v144, -v111
v_add_f32_e64 v145, v145, -v112
v_add_f32_e64 v146, v146, -v113
v_add_f32_e64 v147, v147, -v114
v_add_f32_e64 v148, v148, -v115
v_add_f32_e64 v149, v149, -v116
v_add_f32_e32 v43, v43, v109
v_add_f32_e32 v44, v44, v110
v_add_f32_e32 v45, v45, v111
v_add_f32_e32 v46, v46, v112
v_add_f32_e32 v47, v47, v113
v_add_f32_e32 v48, v48, v114
v_add_f32_e32 v49, v49, v115
v_add_f32_e32 v50, v50, v116
v_add_f32_e32 v142, v142, v76
v_add_f32_e32 v143, v143, v77
v_add_f32_e32 v144, v144, v78
v_add_f32_e32 v145, v145, v79
v_add_f32_e32 v146, v146, v80
v_add_f32_e32 v147, v147, v81
v_add_f32_e32 v148, v148, v82
v_add_f32_e32 v149, v149, v83
v_fma_mixlo_f16 v244, v43, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v44, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v45, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v46, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v248, v47, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v248, v48, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v249, v49, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v249, v50, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v242, v142, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v242, v143, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v243, v144, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v243, v145, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v146, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v147, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v148, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v149, 1.0, 0 op_sel_hi:[0,0,1]
v_mov_b32_e32 v43, 0
v_mov_b32_e32 v44, 0
v_mov_b32_e32 v45, 0
v_mov_b32_e32 v46, 0
v_mov_b32_e32 v47, 0
v_mov_b32_e32 v48, 0
v_mov_b32_e32 v49, 0
v_mov_b32_e32 v50, 0
v_mov_b32_e32 v76, 0
v_mov_b32_e32 v77, 0
v_mov_b32_e32 v78, 0
v_mov_b32_e32 v79, 0
v_mov_b32_e32 v80, 0
v_mov_b32_e32 v81, 0
v_mov_b32_e32 v82, 0
v_mov_b32_e32 v83, 0
v_mov_b32_e32 v109, 0
v_mov_b32_e32 v110, 0
v_mov_b32_e32 v111, 0
v_mov_b32_e32 v112, 0
v_mov_b32_e32 v113, 0
v_mov_b32_e32 v114, 0
v_mov_b32_e32 v115, 0
v_mov_b32_e32 v116, 0
v_mov_b32_e32 v142, 0
v_mov_b32_e32 v143, 0
v_mov_b32_e32 v144, 0
v_mov_b32_e32 v145, 0
v_mov_b32_e32 v146, 0
v_mov_b32_e32 v147, 0
v_mov_b32_e32 v148, 0
v_mov_b32_e32 v149, 0
ds_store_b64 v1, v[244:245] offset:12416
ds_store_b64 v1, v[248:249] offset:12544
ds_store_b64 v1, v[242:243] offset:8264
ds_store_b64 v1, v[246:247] offset:8392
s_add_co_u32 s52, s4, s51
s_add_co_ci_u32 s53, s5, 0
s_setpc_b64 s[52:53]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_e32 v35, v35, v68
v_add_f32_e32 v36, v36, v69
v_add_f32_e32 v37, v37, v70
v_add_f32_e32 v38, v38, v71
v_add_f32_e32 v39, v39, v72
v_add_f32_e32 v40, v40, v73
v_add_f32_e32 v41, v41, v74
v_add_f32_e32 v42, v42, v75
v_add_f32_e64 v134, v134, -v101
v_add_f32_e64 v135, v135, -v102
v_add_f32_e64 v136, v136, -v103
v_add_f32_e64 v137, v137, -v104
v_add_f32_e64 v138, v138, -v105
v_add_f32_e64 v139, v139, -v106
v_add_f32_e64 v140, v140, -v107
v_add_f32_e64 v141, v141, -v108
v_add_f32_e32 v35, v35, v101
v_add_f32_e32 v36, v36, v102
v_add_f32_e32 v37, v37, v103
v_add_f32_e32 v38, v38, v104
v_add_f32_e32 v39, v39, v105
v_add_f32_e32 v40, v40, v106
v_add_f32_e32 v41, v41, v107
v_add_f32_e32 v42, v42, v108
v_add_f32_e32 v134, v134, v68
v_add_f32_e32 v135, v135, v69
v_add_f32_e32 v136, v136, v70
v_add_f32_e32 v137, v137, v71
v_add_f32_e32 v138, v138, v72
v_add_f32_e32 v139, v139, v73
v_add_f32_e32 v140, v140, v74
v_add_f32_e32 v141, v141, v75
v_fma_mixlo_f16 v244, v35, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v36, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v37, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v38, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v248, v39, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v248, v40, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v249, v41, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v249, v42, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v242, v134, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v242, v135, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v243, v136, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v243, v137, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v138, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v139, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v140, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v141, 1.0, 0 op_sel_hi:[0,0,1]
v_mov_b32_e32 v35, 0
v_mov_b32_e32 v36, 0
v_mov_b32_e32 v37, 0
v_mov_b32_e32 v38, 0
v_mov_b32_e32 v39, 0
v_mov_b32_e32 v40, 0
v_mov_b32_e32 v41, 0
v_mov_b32_e32 v42, 0
v_mov_b32_e32 v68, 0
v_mov_b32_e32 v69, 0
v_mov_b32_e32 v70, 0
v_mov_b32_e32 v71, 0
v_mov_b32_e32 v72, 0
v_mov_b32_e32 v73, 0
v_mov_b32_e32 v74, 0
v_mov_b32_e32 v75, 0
v_mov_b32_e32 v101, 0
v_mov_b32_e32 v102, 0
v_mov_b32_e32 v103, 0
v_mov_b32_e32 v104, 0
v_mov_b32_e32 v105, 0
v_mov_b32_e32 v106, 0
v_mov_b32_e32 v107, 0
v_mov_b32_e32 v108, 0
v_mov_b32_e32 v134, 0
v_mov_b32_e32 v135, 0
v_mov_b32_e32 v136, 0
v_mov_b32_e32 v137, 0
v_mov_b32_e32 v138, 0
v_mov_b32_e32 v139, 0
v_mov_b32_e32 v140, 0
v_mov_b32_e32 v141, 0
ds_store_b64 v1, v[244:245] offset:4160
ds_store_b64 v1, v[248:249] offset:4288
ds_store_b64 v1, v[242:243] offset:8
ds_store_b64 v1, v[246:247] offset:136
s_add_co_u32 s52, s4, s51
s_add_co_ci_u32 s53, s5, 0
s_setpc_b64 s[52:53]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_e32 v51, v51, v84
v_add_f32_e32 v52, v52, v85
v_add_f32_e32 v53, v53, v86
v_add_f32_e32 v54, v54, v87
v_add_f32_e32 v55, v55, v88
v_add_f32_e32 v56, v56, v89
v_add_f32_e32 v57, v57, v90
v_add_f32_e32 v58, v58, v91
v_add_f32_e64 v150, v150, -v117
v_add_f32_e64 v151, v151, -v118
v_add_f32_e64 v152, v152, -v119
v_add_f32_e64 v153, v153, -v120
v_add_f32_e64 v154, v154, -v121
v_add_f32_e64 v155, v155, -v122
v_add_f32_e64 v156, v156, -v123
v_add_f32_e64 v157, v157, -v124
v_add_f32_e32 v51, v51, v117
v_add_f32_e32 v52, v52, v118
v_add_f32_e32 v53, v53, v119
v_add_f32_e32 v54, v54, v120
v_add_f32_e32 v55, v55, v121
v_add_f32_e32 v56, v56, v122
v_add_f32_e32 v57, v57, v123
v_add_f32_e32 v58, v58, v124
v_add_f32_e32 v150, v150, v84
v_add_f32_e32 v151, v151, v85
v_add_f32_e32 v152, v152, v86
v_add_f32_e32 v153, v153, v87
v_add_f32_e32 v154, v154, v88
v_add_f32_e32 v155, v155, v89
v_add_f32_e32 v156, v156, v90
v_add_f32_e32 v157, v157, v91
v_fma_mixlo_f16 v244, v51, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v244, v52, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v245, v53, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v245, v54, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v248, v55, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v248, v56, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v249, v57, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v249, v58, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v242, v150, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v242, v151, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v243, v152, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v243, v153, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v246, v154, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v246, v155, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v247, v156, 1.0, 0 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v247, v157, 1.0, 0 op_sel_hi:[0,0,1]
v_mov_b32_e32 v51, 0
v_mov_b32_e32 v52, 0
v_mov_b32_e32 v53, 0
v_mov_b32_e32 v54, 0
v_mov_b32_e32 v55, 0
v_mov_b32_e32 v56, 0
v_mov_b32_e32 v57, 0
v_mov_b32_e32 v58, 0
v_mov_b32_e32 v84, 0
v_mov_b32_e32 v85, 0
v_mov_b32_e32 v86, 0
v_mov_b32_e32 v87, 0
v_mov_b32_e32 v88, 0
v_mov_b32_e32 v89, 0
v_mov_b32_e32 v90, 0
v_mov_b32_e32 v91, 0
v_mov_b32_e32 v117, 0
v_mov_b32_e32 v118, 0
v_mov_b32_e32 v119, 0
v_mov_b32_e32 v120, 0
v_mov_b32_e32 v121, 0
v_mov_b32_e32 v122, 0
v_mov_b32_e32 v123, 0
v_mov_b32_e32 v124, 0
v_mov_b32_e32 v150, 0
v_mov_b32_e32 v151, 0
v_mov_b32_e32 v152, 0
v_mov_b32_e32 v153, 0
v_mov_b32_e32 v154, 0
v_mov_b32_e32 v155, 0
v_mov_b32_e32 v156, 0
v_mov_b32_e32 v157, 0
ds_store_b64 v1, v[244:245] offset:12416
ds_store_b64 v1, v[248:249] offset:12544
ds_store_b64 v1, v[242:243] offset:8264
ds_store_b64 v1, v[246:247] offset:8392
s_bitcmp1_b32 s43, 4
s_cmov_b32 s51, 0x9838
s_add_co_u32 s52, s4, s51
s_add_co_ci_u32 s53, s5, 0
s_setpc_b64 s[52:53]
s_and_b32 s29, s29, 0xffff
s_add_co_u32 s29, s29, 0x20000
s_mov_b64 s[80:81], s[28:29]
s_mov_b32 s82, 0x80000000
s_mov_b32 s83, 0x10014000
s_bitcmp0_b32 s12, 7
s_cselect_b32 s82, 0, 0x80000000
s_and_b32 s77, s64, 3
v_and_b32_e64 v140, v0, 63
v_lshrrev_b32_e32 v137, 2, v140
v_and_b32_e64 v133, v140, 3
v_mad_u32_u24 v133, v137, 8, v133
v_lshlrev_b32_e32 v133, 1, v133
v_mad_u32_u24 v133, 8, s77, v133
v_add_nc_u32_e64 v137, v133, 1
v_cmp_lt_u32_e64 vcc_lo, v133, s10
v_cndmask_b32_e32 v133, -1, v133, vcc_lo
v_cmp_lt_u32_e64 vcc_lo, v137, s10
v_cndmask_b32_e32 v137, -1, v137, vcc_lo
buffer_load_d16_b16 v131, v133, s[80:83], null idxen
buffer_load_d16_hi_b16 v131, v137, s[80:83], null idxen
s_cmp_eq_u32 s64, 5
s_cselect_b32 s28, 0x600, 0
s_cselect_b32 s29, 12, 0
s_cmp_eq_u32 s64, 5
s_cselect_b32 s77, 0x400, 0
s_cselect_b32 s78, 4, 0
s_and_b32 s80, s64, 3
s_mul_i32 s79, s80, 0x800
s_lshl_b32 s80, s79, 1
s_cmp_ge_u32 s64, 4
s_cselect_b32 s79, s80, s79
v_and_b32_e32 v133, 63, v0
v_lshl_add_u32 v92, v133, 3, s28
v_lshl_add_u32 v125, v133, 3, 0
v_mov_b32_e32 v126, s29
v_mov_b32_e32 v127, 0
v_lshl_add_u32 v128, v133, 4, s77
v_mov_b32_e32 v130, s78
v_lshl_add_u32 v129, v133, 4, s79
s_add_co_u32 s29, s26, 1
s_lshr_b32 s29, s29, 1
s_lshl_b32 s77, s29, 1
s_add_co_u32 s78, s27, 1
s_lshr_b32 s78, s78, 1
s_lshl1_add_u32 s78, s78, 2
s_pack_ll_b32_b16 s3, s78, s29
s_pack_ll_b32_b16 s14, s9, s8
s_sub_co_u32 s15, s77, s24
s_sub_co_u32 s28, s78, s25
s_pack_ll_b32_b16 s15, s28, s15
s_pack_ll_b32_b16 s17, s27, s26
s_sub_co_u32 s28, s78, 1
s_pack_ll_b32_b16 s20, s28, s77
v_lshrrev_b32_e64 v141, 16, s3
v_bfi_b32 v142, 0xffff, s3, 0
v_and_b32_e32 v144, 1, v0
v_bfe_u32 v150, v0, 6, 1
v_and_b32_e32 v137, 63, v0
v_mad_u32_u16 v145, 0x7c, s65, 0
v_mad_u32_u16 v150, 2, s66, v150
v_mad_u32_u16 v143, v141, v142, 0
v_cmp_eq_u32_e32 vcc_lo, 0, v144
v_cndmask_b32_e32 v151, v143, v142, vcc_lo
v_mad_u32_u16 v140, 62, v150, v137
v_cndmask_b32_e32 v140, v145, v140, vcc_lo
v_clz_i32_u32_e32 v158, v151
v_lshlrev_b32_e32 v159, v158, v151
v_and_b32_e32 v157, 0xffffff00, v159
v_cmp_eq_u32_e32 vcc_lo, 0x80000000, v159
v_cvt_f32_u32_e32 v157, v157
v_rcp_f32_e32 v152, v157
v_sub_co_ci_u32_e32 v153, vcc_lo, 32, v158, vcc_lo
v_cvt_f32_ubyte0_e32 v158, v159
v_fma_f32 v157, v157, v152, -1.0
v_fma_f32 v157, v158, v152, v157
v_fmaak_f32 v157, v157, v152, 0x9f000000
v_mul_f32_e32 v157, 0x5f800000, v157
v_mov_b32_e32 v158, 0
v_cvt_floor_i32_f32_e64 v157, -v157
v_lshl_add_u32 v152, v152, 9, v157
v_mad_co_u64_u32 v[158:159], vcc_lo, v159, v152, v[158:159]
v_sub_co_ci_u32_e64 v152, vcc_lo, v152, -1, vcc_lo
v_mov_b32_dpp v155, v153 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v153, v153 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v154, v152 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v152, v152 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mul_hi_u32 v157, v140, v152
v_add_co_u32 v133, vcc_lo, v157, v140
v_add_co_ci_u32_e64 v157, vcc_lo, 0, 0, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 32, v153
v_cndmask_b32_e32 v133, v133, v157, vcc_lo
v_alignbit_b32 v133, v157, v133, v153
v_mul_hi_u32 v157, v140, v154
v_add_co_u32 v4, vcc_lo, v157, v140
v_add_co_ci_u32_e64 v157, vcc_lo, 0, 0, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 32, v155
v_cndmask_b32_e32 v4, v4, v157, vcc_lo
v_alignbit_b32 v4, v157, v4, v155
v_mad_u32_u16 v149, v133, v142, 0
v_mad_u32_u16 v148, v4, v141, 0
v_sub_nc_u32_e32 v149, v140, v149
v_sub_nc_u32_e32 v148, v133, v148
v_readlane_b32 s79, v149, 1
v_sub_nc_u32_e32 v149, v149, v142
v_readlane_b32 s13, v148, 1
v_sub_nc_u32_e32 v148, v148, v141
v_readlane_b32 s2, v4, 1
v_sub_nc_u32_e64 v4, v4, s6
s_lshl_b32 s13, s13, 16
s_and_b32 s79, s79, 0xffff
s_add_co_u32 s13, s13, s79
v_mov_b32_dpp v149, v149 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v148, v148 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v4, v4 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32 v149, vcc_lo, v149, v144
v_cndmask_b32_e32 v147, 0, v142, vcc_lo
v_add_co_ci_u32_e64 v148, vcc_lo, v148, 0, vcc_lo
v_cndmask_b32_e32 v146, 0, v141, vcc_lo
v_add_co_ci_u32_e64 v4, vcc_lo, v4, 0, vcc_lo
v_min_u32_e64 v144, v137, 63
v_sub_nc_u32_e32 v149, v149, v147
v_sub_nc_u32_e32 v148, v148, v146
v_cmp_eq_u32_e32 vcc_lo, v137, v144
v_lshlrev_b32_e32 v1, 16, v148
v_bfi_b32 v1, 0xffff, v149, v1
v_add_nc_u32_e64 v160, v4, s6
v_med3_u32 v144, v137, 1, 62
v_mul_lo_u32 v2, v160, s40
v_mul_lo_u32 v11, v160, s48
s_mul_i32 s16, s2, s40
s_mul_i32 s21, s2, s48
v_cndmask_b32_e32 v2, 0x80000000, v2, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, v137, v144
v_cndmask_b32_e32 v11, 0x80000000, v11, vcc_lo
v_cmp_ge_u32_e64 s36, v160, s6
v_cndmask_b32_e64 v2, v2, 0x80000000, s36
v_cndmask_b32_e64 v11, v11, 0x80000000, s36
v_mov_b32_e32 v136, s36
ds_store_b32 v130, v136 offset:55808
s_and_b32 s8, s56, 0xff
s_cmp_eq_u32 s8, 0
s_cmov_b32 s30, 0x3c003c00
s_cbranch_scc1 18
s_cmp_eq_u32 s8, 4
s_cbranch_scc1 22
s_cmp_eq_u32 s8, 2
s_cbranch_scc1 23
v_cvt_f16_f32_e64 v133, s30
s_nop 0
v_readfirstlane_b32 s30, v133
v_cvt_f16_f32_e64 v133, s31
v_cmp_gt_f16_e64 vcc_lo, s30, 1.0
v_readfirstlane_b32 s31, v133
s_pack_ll_b32_b16 s30, s30, s30
s_pack_ll_b32_b16 s31, s31, s31
s_cmp_eq_u32 s8, 3
s_cbranch_scc1 13
s_cbranch_vccnz 3
s_mov_b32 s76, 0x5990
s_branch 11
s_mov_b32 s76, 0x6118
s_branch 8
s_mov_b32 s76, 0x68a0
s_branch 5
s_mov_b32 s76, 0x6f28
s_branch 2
s_mov_b32 s76, 0x7bb0
s_lshl_b32 s59, s49, 1
s_mul_i32 s68, s59, 16
s_mul_hi_u32 s69, s59, 16
s_and_b32 s8, s64, 3
s_lshl_b32 s9, s59, 3
s_mul_i32 s9, s9, s8
s_add_co_u32 s18, s18, s9
s_add_co_ci_u32 s19, s19, 0
s_lshl_b32 s70, s8, 3
s_sub_co_u32 s70, s10, s70
s_cmov_b32 s70, 0
s_and_b32 s19, s19, 0xffff
s_add_co_u32 s19, s19, 0x20000
s_mov_b64 s[60:61], s[18:19]
s_mov_b32 s62, 0
s_mov_b32 s63, 0x10014000
s_mov_b32 s32, 0x10000
s_mov_b32 s33, 0x30002
s_mov_b32 s34, 0x10000
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
s_sub_co_u32 s9, s23, 1
s_bitcmp1_b32 s12, 1
s_cselect_b32 s9, s9, 0
s_cselect_b32 s8, -1, 1
s_sub_co_u32 s25, s22, 1
s_bitcmp1_b32 s12, 0
s_cselect_b32 s25, s25, 0
s_cselect_b32 s24, -1, 1
s_bitcmp1_b32 s12, 31
s_cselect_b32 s27, 0, 1
s_cselect_b32 s26, 4, 0
s_cselect_b32 s29, 2, 0
s_cselect_b32 s77, 2, 4
s_and_b32 s28, s64, 3
v_bfe_u32 v140, v0, 2, 6
v_lshlrev_b32_e64 v141, 1, s28
v_bfe_u32 v137, v140, 0, s29
v_mad_u32_u24 v137, v141, s27, v137
v_mov_b32_e32 v133, s77
v_bfe_u32 v133, v140, s29, v133
v_mad_u32_u24 v133, v141, s26, v133
v_mov_b32_e32 v140, s25
v_add_nc_u32_e32 v141, s24, v140
v_add_nc_u32_e32 v142, s24, v141
v_xor_b32_dpp v143, v0, v0 quad_perm:[0,0,0,1] row_mask:0xf bank_mask:0xf
v_mov_b32_e32 v144, s9
v_mad_i32_i24 v144, s8, v143, v144
v_mad_u32_u24 v144, s45, v137, v144
v_cmp_lt_u32_e64 vcc_lo, v143, s23
v_cmp_lt_u32_e64 s28, v140, s22
v_cmp_lt_u32_e64 s78, v141, s22
v_cmp_lt_u32_e64 s80, v142, s22
s_and_b64 s[28:29], vcc, s[28:29]
s_and_b64 s[78:79], vcc, s[78:79]
s_and_b64 s[80:81], vcc, s[80:81]
s_add_co_u32 s77, s7, 15
s_and_b32 s77, s77, -16
s_sub_co_u32 s77, s77, 16
s_sub_co_u32 s77, s7, s77
v_sub_nc_u32_e32 v19, s77, v137
v_add_nc_u32_e64 v15, v19, 3
v_add_nc_u32_e64 v19, v19, 3
v_mad_u32_u24 v12, s44, v133, v144
v_mad_u32_u24 v14, s46, v142, v12
v_mad_u32_u24 v13, s46, v141, v12
v_mad_u32_u24 v12, s46, v140, v12
v_cmp_lt_u32_e64 vcc_lo, v133, s10
v_cndmask_b32_e32 v12, -1, v12, vcc_lo
v_cndmask_b32_e32 v13, -1, v13, vcc_lo
v_cndmask_b32_e32 v14, -1, v14, vcc_lo
v_cndmask_b32_e64 v12, -1, v12, s28
v_cndmask_b32_e64 v13, -1, v13, s78
v_cndmask_b32_e64 v14, -1, v14, s80
v_add_nc_u32_e64 v145, v133, 16
v_mad_u32_u24 v16, s44, v145, v144
v_mad_u32_u24 v18, s46, v142, v16
v_mad_u32_u24 v17, s46, v141, v16
v_mad_u32_u24 v16, s46, v140, v16
v_cmp_lt_u32_e64 vcc_lo, v145, s10
v_cndmask_b32_e32 v16, -1, v16, vcc_lo
v_cndmask_b32_e32 v17, -1, v17, vcc_lo
v_cndmask_b32_e32 v18, -1, v18, vcc_lo
v_cndmask_b32_e64 v16, -1, v16, s28
v_cndmask_b32_e64 v17, -1, v17, s78
v_cndmask_b32_e64 v18, -1, v18, s80
v_add_nc_u32_e64 v133, v133, s26
v_add_nc_u32_e64 v137, v137, s27
v_mad_u32_u24 v144, s45, s27, v144
v_sub_nc_u32_e32 v27, s77, v137
v_add_nc_u32_e64 v23, v27, 3
v_add_nc_u32_e64 v27, v27, 3
v_mad_u32_u24 v20, s44, v133, v144
v_mad_u32_u24 v22, s46, v142, v20
v_mad_u32_u24 v21, s46, v141, v20
v_mad_u32_u24 v20, s46, v140, v20
v_cmp_lt_u32_e64 vcc_lo, v133, s10
v_cndmask_b32_e32 v20, -1, v20, vcc_lo
v_cndmask_b32_e32 v21, -1, v21, vcc_lo
v_cndmask_b32_e32 v22, -1, v22, vcc_lo
v_cndmask_b32_e64 v20, -1, v20, s28
v_cndmask_b32_e64 v21, -1, v21, s78
v_cndmask_b32_e64 v22, -1, v22, s80
v_add_nc_u32_e64 v145, v133, 16
v_mad_u32_u24 v24, s44, v145, v144
v_mad_u32_u24 v26, s46, v142, v24
v_mad_u32_u24 v25, s46, v141, v24
v_mad_u32_u24 v24, s46, v140, v24
v_cmp_lt_u32_e64 vcc_lo, v145, s10
v_cndmask_b32_e32 v24, -1, v24, vcc_lo
v_cndmask_b32_e32 v25, -1, v25, vcc_lo
v_cndmask_b32_e32 v26, -1, v26, vcc_lo
v_cndmask_b32_e64 v24, -1, v24, s28
v_cndmask_b32_e64 v25, -1, v25, s78
v_cndmask_b32_e64 v26, -1, v26, s80
ds_store_b128 v129, v[12:15] offset:45568
ds_store_b128 v129, v[16:19] offset:46592
ds_store_b128 v129, v[20:23] offset:47616
ds_store_b128 v129, v[24:27] offset:48640
s_wait_loadcnt 0x0
v_readlane_b32 s35, v131, 0
v_readlane_b32 s38, v131, 1
v_readlane_b32 s39, v131, 2
v_readlane_b32 s43, v131, 3
v_readlane_b32 s47, v131, 4
v_readlane_b32 s51, v131, 5
v_readlane_b32 s52, v131, 6
v_readlane_b32 s53, v131, 7
s_mov_b32 s71, 0
s_mov_b32 s72, 0
s_mov_b32 s54, 2
s_lshl_b32 s55, s54, 9
s_lshl_b32 s57, s54, 2
s_branch 114
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_bitcmp1_b32 s36, 0
s_add_co_ci_u32 s71, s71, 2
s_sub_co_u32 s72, s7, 1
s_add_co_u32 s74, s4, s73
s_add_co_ci_u32 s75, s5, 0
s_setprio 3
v_mov_b32_e32 v140, v5
v_mov_b32_e32 v141, v6
v_mov_b32_e32 v142, v136
v_add_co_u32 v137, vcc_lo, v1, s13
v_pk_mad_u16 v143, v1, 0x20001, s15
v_pk_mad_u16 v144, v1, 0x20001, s20
_v_pk_min_u16_vvv 146, 137, 1
v_cndmask_b32_e64 v147, 0, s40, vcc_lo
v_cndmask_b32_e64 v149, 0, s48, vcc_lo
v_mad_u32_u16 v7, v143, 1, v2
v_mad_u32_u16 v5, v144, 1, v11
v_add3_u32 v2, v2, s16, v147
v_add3_u32 v11, v11, s21, v149
_v_pk_sub_u16_vvv 146, 1, 146
v_add_co_ci_u32_e64 v4, s36, v4, s2, vcc_lo
v_cndmask_b32_e64 v2, v2, 0x80000000, s36
v_cndmask_b32_e64 v11, v11, 0x80000000, s36
v_cmp_lt_u16_e64 vcc_lo, v143, s14
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc_lo
v_cmp_lt_u16_e64 vcc_lo, v144, s17
v_cndmask_b32_e32 v5, 0x80000000, v5, vcc_lo
_v_pk_ashrrev_i16_vnumv 146, 15, 146
_v_pk_add_u16_vvs 145, 143, 33
_v_pk_add_u16_vvs 148, 143, 32
v_mad_u32_u16 v10, v145, s42, v7 op_sel:[1,0,0,0]
v_mad_u32_u16 v8, v148, s42, v7 op_sel:[1,0,0,0]
_v_pk_add_u16_vvs 150, 144, 34
_v_cmp_lt_u16_vcchvhs 145, 14
v_cndmask_b32_e32 v10, 0x80000000, v10, vcc_lo
_v_cmp_lt_u16_vcchvhs 148, 14
v_cndmask_b32_e32 v8, 0x80000000, v8, vcc_lo
v_mad_u32_u16 v6, v150, s50, v5 op_sel:[1,0,0,0]
v_mad_u32_u16 v9, v145, s42, v7
v_mad_u32_u16 v7, v148, s42, v7
_v_cmp_lt_u16_vcchvhs 150, 17
v_cndmask_b32_e32 v6, 0x80000000, v6, vcc_lo
_v_cmp_lt_u16_vccvhs 145, 14
v_cndmask_b32_e32 v9, 0x80000000, v9, vcc_lo
_v_cmp_lt_u16_vccvhs 148, 14
v_cndmask_b32_e32 v7, 0x80000000, v7, vcc_lo
v_mad_u32_u16 v5, v150, s50, v5
v_pk_mad_u16 v1, v146, s3, v137
_v_cmp_lt_u16_vccvhs 150, 17
v_cndmask_b32_e32 v5, 0x80000000, v5, vcc_lo
v_mov_b32_e32 v136, s36
v_add_nc_u32_e32 v131, s55, v92
v_add_nc_u32_e32 v133, s57, v126
ds_store_b64 v131, v[140:141] offset:55816
ds_store_b32 v133, v142 offset:58888
ds_store_b128 v128, v[7:10] offset:53760
ds_store_b32 v130, v136 offset:55808
s_setpc_b64 s[74:75]
s_barrier_signal -1
s_barrier_wait 0xffff
s_lshl_b32 s71, s71, 2
s_mov_b32 s73, 0x52cc
s_sub_co_u32 s72, s72, 16
s_cbranch_scc1 65422
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s71, 7
s_cselect_b32 s74, s76, 0x5300
s_add_co_u32 s74, s4, s74
s_add_co_ci_u32 s75, s5, 0
s_sub_co_u32 s54, s54, 1
s_cselect_b32 s54, 2, s54
s_lshl_b32 s55, s54, 9
s_lshl_b32 s57, s54, 2
s_wait_dscnt 0x0
s_swappc_b64 s[74:75], s[74:75]
s_bitcmp1_b32 s71, 5
s_cbranch_scc1 136
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s71, 7
s_cselect_b32 s74, s76, 0x5334
s_add_co_u32 s74, s4, s74
s_add_co_ci_u32 s75, s5, 0
s_wait_dscnt 0x0
s_swappc_b64 s[74:75], s[74:75]
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s71, 7
s_cselect_b32 s74, s76, 0x5360
s_add_co_u32 s74, s4, s74
s_add_co_ci_u32 s75, s5, 0
s_wait_dscnt 0x0
s_swappc_b64 s[74:75], s[74:75]
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s71, 5
s_cselect_b32 s74, s76, 0x538c
s_add_co_u32 s74, s4, s74
s_add_co_ci_u32 s75, s5, 0
s_wait_dscnt 0x0
s_swappc_b64 s[74:75], s[74:75]
s_branch 65480
s_barrier_signal -1
s_barrier_wait 0xffff
s_lshl_b32 s71, s71, 2
s_mov_b32 s73, 0x53ac
s_sub_co_u32 s72, s72, 16
s_cbranch_scc1 65366
s_setprio 0
ds_load_b64 v[64:65], v3 offset:8256
ds_load_b64 v[66:67], v3 offset:8512
ds_load_b64 v[72:73], v3 offset:8768
ds_load_b64 v[74:75], v3 offset:9024
ds_load_b64 v[80:81], v3 offset:9280
ds_load_b64 v[82:83], v3 offset:9536
ds_load_b64 v[88:89], v3 offset:9792
ds_load_b64 v[90:91], v3 offset:10048
s_setprio 2
s_bitcmp1_b32 s71, 7
s_cselect_b32 s74, s76, 0x5520
s_add_co_u32 s74, s4, s74
s_add_co_ci_u32 s75, s5, 0
s_sub_co_u32 s54, s54, 1
s_cselect_b32 s54, 2, s54
s_lshl_b32 s55, s54, 9
s_lshl_b32 s57, s54, 2
s_wait_dscnt 0x0
v_fma_mix_f32 v115, v64, 1.0, v66 op_sel_hi:[1,0,1]
v_fma_mix_f32 v64, v64, 1.0, v66 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v116, v65, 1.0, v67 op_sel_hi:[1,0,1]
v_fma_mix_f32 v65, v65, 1.0, v67 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v115, v115, 1.0, v72 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v115, v64, 1.0, v72 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v116, v116, 1.0, v73 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v116, v65, 1.0, v73 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v113, v72, -1.0, v66 op_sel_hi:[1,0,1]
v_fma_mix_f32 v72, v72, -1.0, v66 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v114, v73, -1.0, v67 op_sel_hi:[1,0,1]
v_fma_mix_f32 v73, v73, -1.0, v67 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v113, v113, 1.0, v74 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v113, v72, 1.0, v74 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v114, v114, 1.0, v75 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v114, v73, 1.0, v75 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v123, v80, 1.0, v82 op_sel_hi:[1,0,1]
v_fma_mix_f32 v80, v80, 1.0, v82 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v124, v81, 1.0, v83 op_sel_hi:[1,0,1]
v_fma_mix_f32 v81, v81, 1.0, v83 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v123, v123, 1.0, v88 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v123, v80, 1.0, v88 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v124, v124, 1.0, v89 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v124, v81, 1.0, v89 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v121, v88, -1.0, v82 op_sel_hi:[1,0,1]
v_fma_mix_f32 v88, v88, -1.0, v82 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v122, v89, -1.0, v83 op_sel_hi:[1,0,1]
v_fma_mix_f32 v89, v89, -1.0, v83 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v121, v121, 1.0, v90 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v121, v88, 1.0, v90 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v122, v122, 1.0, v91 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v122, v89, 1.0, v91 op_sel:[0,0,1] op_sel_hi:[0,0,1]
s_swappc_b64 s[74:75], s[74:75]
s_bitcmp1_b32 s71, 5
s_cbranch_scc0 65400
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 0
ds_load_b64 v[28:29], v3
ds_load_b64 v[30:31], v3 offset:256
ds_load_b64 v[36:37], v3 offset:512
ds_load_b64 v[38:39], v3 offset:768
ds_load_b64 v[44:45], v3 offset:1024
ds_load_b64 v[46:47], v3 offset:1280
ds_load_b64 v[52:53], v3 offset:1536
ds_load_b64 v[54:55], v3 offset:1792
s_setprio 2
s_bitcmp1_b32 s71, 7
s_cselect_b32 s74, s76, 0x5594
s_add_co_u32 s74, s4, s74
s_add_co_ci_u32 s75, s5, 0
s_wait_dscnt 0x0
s_swappc_b64 s[74:75], s[74:75]
v_fma_mix_f32 v95, v28, 1.0, v30 op_sel_hi:[1,0,1]
v_fma_mix_f32 v28, v28, 1.0, v30 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v96, v29, 1.0, v31 op_sel_hi:[1,0,1]
v_fma_mix_f32 v29, v29, 1.0, v31 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v95, v95, 1.0, v36 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v95, v28, 1.0, v36 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v96, v96, 1.0, v37 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v96, v29, 1.0, v37 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v93, v36, -1.0, v30 op_sel_hi:[1,0,1]
v_fma_mix_f32 v36, v36, -1.0, v30 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v94, v37, -1.0, v31 op_sel_hi:[1,0,1]
v_fma_mix_f32 v37, v37, -1.0, v31 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v93, v93, 1.0, v38 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v93, v36, 1.0, v38 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v94, v94, 1.0, v39 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v94, v37, 1.0, v39 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v103, v44, 1.0, v46 op_sel_hi:[1,0,1]
v_fma_mix_f32 v44, v44, 1.0, v46 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v104, v45, 1.0, v47 op_sel_hi:[1,0,1]
v_fma_mix_f32 v45, v45, 1.0, v47 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v103, v103, 1.0, v52 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v103, v44, 1.0, v52 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v104, v104, 1.0, v53 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v104, v45, 1.0, v53 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v101, v52, -1.0, v46 op_sel_hi:[1,0,1]
v_fma_mix_f32 v52, v52, -1.0, v46 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v102, v53, -1.0, v47 op_sel_hi:[1,0,1]
v_fma_mix_f32 v53, v53, -1.0, v47 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v101, v101, 1.0, v54 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v101, v52, 1.0, v54 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v102, v102, 1.0, v55 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v102, v53, 1.0, v55 op_sel:[0,0,1] op_sel_hi:[0,0,1]
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 0
ds_load_b64 v[60:61], v3 offset:8256
ds_load_b64 v[62:63], v3 offset:8512
ds_load_b64 v[68:69], v3 offset:8768
ds_load_b64 v[70:71], v3 offset:9024
ds_load_b64 v[76:77], v3 offset:9280
ds_load_b64 v[78:79], v3 offset:9536
ds_load_b64 v[84:85], v3 offset:9792
ds_load_b64 v[86:87], v3 offset:10048
s_setprio 2
s_bitcmp1_b32 s71, 7
s_cselect_b32 s74, s76, 0x5700
s_add_co_u32 s74, s4, s74
s_add_co_ci_u32 s75, s5, 0
s_wait_dscnt 0x0
s_swappc_b64 s[74:75], s[74:75]
v_fma_mix_f32 v111, v60, 1.0, v62 op_sel_hi:[1,0,1]
v_fma_mix_f32 v60, v60, 1.0, v62 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v112, v61, 1.0, v63 op_sel_hi:[1,0,1]
v_fma_mix_f32 v61, v61, 1.0, v63 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v111, v111, 1.0, v68 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v111, v60, 1.0, v68 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v112, v112, 1.0, v69 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v112, v61, 1.0, v69 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v109, v68, -1.0, v62 op_sel_hi:[1,0,1]
v_fma_mix_f32 v68, v68, -1.0, v62 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v110, v69, -1.0, v63 op_sel_hi:[1,0,1]
v_fma_mix_f32 v69, v69, -1.0, v63 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v109, v109, 1.0, v70 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v109, v68, 1.0, v70 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v110, v110, 1.0, v71 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v110, v69, 1.0, v71 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v119, v76, 1.0, v78 op_sel_hi:[1,0,1]
v_fma_mix_f32 v76, v76, 1.0, v78 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v120, v77, 1.0, v79 op_sel_hi:[1,0,1]
v_fma_mix_f32 v77, v77, 1.0, v79 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v119, v119, 1.0, v84 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v119, v76, 1.0, v84 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v120, v120, 1.0, v85 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v120, v77, 1.0, v85 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v117, v84, -1.0, v78 op_sel_hi:[1,0,1]
v_fma_mix_f32 v84, v84, -1.0, v78 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v118, v85, -1.0, v79 op_sel_hi:[1,0,1]
v_fma_mix_f32 v85, v85, -1.0, v79 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v117, v117, 1.0, v86 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v117, v84, 1.0, v86 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v118, v118, 1.0, v87 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v118, v85, 1.0, v87 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_add_nc_u32_e32 v139, s55, v125
v_add_nc_u32_e32 v132, s57, v127
s_barrier_signal -1
s_barrier_wait 0xffff
s_setprio 0
ds_load_b64 v[134:135], v139 offset:55816
ds_load_b64 v[138:139], v139 offset:57352
ds_load_b32 v132, v132 offset:58888
ds_load_b64 v[32:33], v3
ds_load_b64 v[34:35], v3 offset:256
ds_load_b64 v[40:41], v3 offset:512
ds_load_b64 v[42:43], v3 offset:768
ds_load_b64 v[48:49], v3 offset:1024
ds_load_b64 v[50:51], v3 offset:1280
ds_load_b64 v[56:57], v3 offset:1536
ds_load_b64 v[58:59], v3 offset:1792
s_setprio 2
s_bitcmp1_b32 s71, 5
s_cselect_b32 s74, s76, 0x598c
s_add_co_u32 s74, s4, s74
s_add_co_ci_u32 s75, s5, 0
s_wait_dscnt 0x0
v_fma_mix_f32 v99, v32, 1.0, v34 op_sel_hi:[1,0,1]
v_fma_mix_f32 v32, v32, 1.0, v34 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v100, v33, 1.0, v35 op_sel_hi:[1,0,1]
v_fma_mix_f32 v33, v33, 1.0, v35 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v99, v99, 1.0, v40 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v99, v32, 1.0, v40 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v100, v100, 1.0, v41 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v100, v33, 1.0, v41 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v97, v40, -1.0, v34 op_sel_hi:[1,0,1]
v_fma_mix_f32 v40, v40, -1.0, v34 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v98, v41, -1.0, v35 op_sel_hi:[1,0,1]
v_fma_mix_f32 v41, v41, -1.0, v35 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v97, v97, 1.0, v42 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v97, v40, 1.0, v42 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v98, v98, 1.0, v43 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v98, v41, 1.0, v43 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v107, v48, 1.0, v50 op_sel_hi:[1,0,1]
v_fma_mix_f32 v48, v48, 1.0, v50 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v108, v49, 1.0, v51 op_sel_hi:[1,0,1]
v_fma_mix_f32 v49, v49, 1.0, v51 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v107, v107, 1.0, v56 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v107, v48, 1.0, v56 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v108, v108, 1.0, v57 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v108, v49, 1.0, v57 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mix_f32 v105, v56, -1.0, v50 op_sel_hi:[1,0,1]
v_fma_mix_f32 v56, v56, -1.0, v50 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mix_f32 v106, v57, -1.0, v51 op_sel_hi:[1,0,1]
v_fma_mix_f32 v57, v57, -1.0, v51 op_sel:[1,0,1] op_sel_hi:[1,0,1]
v_fma_mixlo_f16 v105, v105, 1.0, v58 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v105, v56, 1.0, v58 op_sel:[0,0,1] op_sel_hi:[0,0,1]
v_fma_mixlo_f16 v106, v106, 1.0, v59 op_sel_hi:[0,0,1]
v_fma_mixhi_f16 v106, v57, 1.0, v59 op_sel:[0,0,1] op_sel_hi:[0,0,1]
s_swappc_b64 s[74:75], s[74:75]
s_branch 65152
s_setprio 3
v_readfirstlane_b32 s58, v132
_v_pk_add_f16_vvs 95, 95, 35
_v_pk_add_f16_vvs 96, 96, 38
_v_pk_add_f16_vvs 99, 99, 35
_v_pk_add_f16_vvs 100, 100, 38
_v_pk_add_f16_vvs 93, 93, 35
_v_pk_add_f16_vvs 94, 94, 38
_v_pk_add_f16_vvs 97, 97, 35
_v_pk_add_f16_vvs 98, 98, 38
_v_pk_mul_f16_vsv 131, 30, 95
_v_pk_mul_f16_vsv 140, 30, 96
_v_pk_mul_f16_vsv 143, 30, 99
_v_pk_mul_f16_vsv 146, 30, 100
_v_pk_max_num_f16_vvv 95, 95, 131
_v_pk_max_num_f16_vvv 96, 96, 140
_v_pk_max_num_f16_vvv 99, 99, 143
_v_pk_max_num_f16_vvv 100, 100, 146
_v_pk_mul_f16_vsv 131, 30, 93
_v_pk_mul_f16_vsv 140, 30, 94
_v_pk_mul_f16_vsv 143, 30, 97
_v_pk_mul_f16_vsv 146, 30, 98
_v_pk_max_num_f16_vvv 93, 93, 131
_v_pk_max_num_f16_vvv 94, 94, 140
_v_pk_max_num_f16_vvv 97, 97, 143
_v_pk_max_num_f16_vvv 98, 98, 146
s_setprio 0
buffer_store_b16 v95, v134, s[60:63], null idxen
buffer_store_b16 v99, v138, s[60:63], null idxen
buffer_store_b16 v93, v135, s[60:63], null idxen
buffer_store_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v95, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v99, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v93, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v96, v134, s[60:63], null idxen
buffer_store_b16 v100, v138, s[60:63], null idxen
buffer_store_b16 v94, v135, s[60:63], null idxen
buffer_store_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v96, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v100, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v94, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x5b6c
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 103, 103, 39
_v_pk_add_f16_vvs 104, 104, 43
_v_pk_add_f16_vvs 107, 107, 39
_v_pk_add_f16_vvs 108, 108, 43
_v_pk_add_f16_vvs 101, 101, 39
_v_pk_add_f16_vvs 102, 102, 43
_v_pk_add_f16_vvs 105, 105, 39
_v_pk_add_f16_vvs 106, 106, 43
_v_pk_mul_f16_vsv 131, 30, 103
_v_pk_mul_f16_vsv 140, 30, 104
_v_pk_mul_f16_vsv 143, 30, 107
_v_pk_mul_f16_vsv 146, 30, 108
_v_pk_max_num_f16_vvv 103, 103, 131
_v_pk_max_num_f16_vvv 104, 104, 140
_v_pk_max_num_f16_vvv 107, 107, 143
_v_pk_max_num_f16_vvv 108, 108, 146
_v_pk_mul_f16_vsv 131, 30, 101
_v_pk_mul_f16_vsv 140, 30, 102
_v_pk_mul_f16_vsv 143, 30, 105
_v_pk_mul_f16_vsv 146, 30, 106
_v_pk_max_num_f16_vvv 101, 101, 131
_v_pk_max_num_f16_vvv 102, 102, 140
_v_pk_max_num_f16_vvv 105, 105, 143
_v_pk_max_num_f16_vvv 106, 106, 146
s_setprio 0
buffer_store_b16 v103, v134, s[60:63], null idxen
buffer_store_b16 v107, v138, s[60:63], null idxen
buffer_store_b16 v101, v135, s[60:63], null idxen
buffer_store_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v103, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v107, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v101, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v104, v134, s[60:63], null idxen
buffer_store_b16 v108, v138, s[60:63], null idxen
buffer_store_b16 v102, v135, s[60:63], null idxen
buffer_store_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v104, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v108, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v102, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x5d50
s_sub_co_u32 s67, s70, 17
s_cselect_b32 s62, 0, s62
s_add_nc_u64 s[60:61], s[18:19], s[68:69]
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 111, 111, 47
_v_pk_add_f16_vvs 112, 112, 51
_v_pk_add_f16_vvs 115, 115, 47
_v_pk_add_f16_vvs 116, 116, 51
_v_pk_add_f16_vvs 109, 109, 47
_v_pk_add_f16_vvs 110, 110, 51
_v_pk_add_f16_vvs 113, 113, 47
_v_pk_add_f16_vvs 114, 114, 51
_v_pk_mul_f16_vsv 131, 30, 111
_v_pk_mul_f16_vsv 140, 30, 112
_v_pk_mul_f16_vsv 143, 30, 115
_v_pk_mul_f16_vsv 146, 30, 116
_v_pk_max_num_f16_vvv 111, 111, 131
_v_pk_max_num_f16_vvv 112, 112, 140
_v_pk_max_num_f16_vvv 115, 115, 143
_v_pk_max_num_f16_vvv 116, 116, 146
_v_pk_mul_f16_vsv 131, 30, 109
_v_pk_mul_f16_vsv 140, 30, 110
_v_pk_mul_f16_vsv 143, 30, 113
_v_pk_mul_f16_vsv 146, 30, 114
_v_pk_max_num_f16_vvv 109, 109, 131
_v_pk_max_num_f16_vvv 110, 110, 140
_v_pk_max_num_f16_vvv 113, 113, 143
_v_pk_max_num_f16_vvv 114, 114, 146
s_setprio 0
buffer_store_b16 v111, v134, s[60:63], null idxen
buffer_store_b16 v115, v138, s[60:63], null idxen
buffer_store_b16 v109, v135, s[60:63], null idxen
buffer_store_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v111, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v115, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v109, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v112, v134, s[60:63], null idxen
buffer_store_b16 v116, v138, s[60:63], null idxen
buffer_store_b16 v110, v135, s[60:63], null idxen
buffer_store_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v112, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v116, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v110, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x5f28
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 119, 119, 52
_v_pk_add_f16_vvs 120, 120, 53
_v_pk_add_f16_vvs 123, 123, 52
_v_pk_add_f16_vvs 124, 124, 53
_v_pk_add_f16_vvs 117, 117, 52
_v_pk_add_f16_vvs 118, 118, 53
_v_pk_add_f16_vvs 121, 121, 52
_v_pk_add_f16_vvs 122, 122, 53
_v_pk_mul_f16_vsv 131, 30, 119
_v_pk_mul_f16_vsv 140, 30, 120
_v_pk_mul_f16_vsv 143, 30, 123
_v_pk_mul_f16_vsv 146, 30, 124
_v_pk_max_num_f16_vvv 119, 119, 131
_v_pk_max_num_f16_vvv 120, 120, 140
_v_pk_max_num_f16_vvv 123, 123, 143
_v_pk_max_num_f16_vvv 124, 124, 146
_v_pk_mul_f16_vsv 131, 30, 117
_v_pk_mul_f16_vsv 140, 30, 118
_v_pk_mul_f16_vsv 143, 30, 121
_v_pk_mul_f16_vsv 146, 30, 122
_v_pk_max_num_f16_vvv 117, 117, 131
_v_pk_max_num_f16_vvv 118, 118, 140
_v_pk_max_num_f16_vvv 121, 121, 143
_v_pk_max_num_f16_vvv 122, 122, 146
s_setprio 0
buffer_store_b16 v119, v134, s[60:63], null idxen
buffer_store_b16 v123, v138, s[60:63], null idxen
buffer_store_b16 v117, v135, s[60:63], null idxen
buffer_store_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v119, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v123, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v117, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v120, v134, s[60:63], null idxen
buffer_store_b16 v124, v138, s[60:63], null idxen
buffer_store_b16 v118, v135, s[60:63], null idxen
buffer_store_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v120, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v124, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v118, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x5990
s_bitcmp1_b32 s58, 0
s_cbranch_scc1 3533
s_sub_co_u32 s67, s70, 1
s_cselect_b32 s62, 0, 0x80000000
s_mov_b64 s[60:61], s[18:19]
s_setpc_b64 s[74:75]
s_setprio 3
v_readfirstlane_b32 s58, v132
_v_pk_add_f16_vvs 95, 95, 35
_v_pk_add_f16_vvs 96, 96, 38
_v_pk_add_f16_vvs 99, 99, 35
_v_pk_add_f16_vvs 100, 100, 38
_v_pk_add_f16_vvs 93, 93, 35
_v_pk_add_f16_vvs 94, 94, 38
_v_pk_add_f16_vvs 97, 97, 35
_v_pk_add_f16_vvs 98, 98, 38
_v_pk_mul_f16_vsv 131, 30, 95
_v_pk_mul_f16_vsv 140, 30, 96
_v_pk_mul_f16_vsv 143, 30, 99
_v_pk_mul_f16_vsv 146, 30, 100
_v_pk_min_num_f16_vvv 95, 95, 131
_v_pk_min_num_f16_vvv 96, 96, 140
_v_pk_min_num_f16_vvv 99, 99, 143
_v_pk_min_num_f16_vvv 100, 100, 146
_v_pk_mul_f16_vsv 131, 30, 93
_v_pk_mul_f16_vsv 140, 30, 94
_v_pk_mul_f16_vsv 143, 30, 97
_v_pk_mul_f16_vsv 146, 30, 98
_v_pk_min_num_f16_vvv 93, 93, 131
_v_pk_min_num_f16_vvv 94, 94, 140
_v_pk_min_num_f16_vvv 97, 97, 143
_v_pk_min_num_f16_vvv 98, 98, 146
s_setprio 0
buffer_store_b16 v95, v134, s[60:63], null idxen
buffer_store_b16 v99, v138, s[60:63], null idxen
buffer_store_b16 v93, v135, s[60:63], null idxen
buffer_store_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v95, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v99, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v93, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v96, v134, s[60:63], null idxen
buffer_store_b16 v100, v138, s[60:63], null idxen
buffer_store_b16 v94, v135, s[60:63], null idxen
buffer_store_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v96, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v100, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v94, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x62f4
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 103, 103, 39
_v_pk_add_f16_vvs 104, 104, 43
_v_pk_add_f16_vvs 107, 107, 39
_v_pk_add_f16_vvs 108, 108, 43
_v_pk_add_f16_vvs 101, 101, 39
_v_pk_add_f16_vvs 102, 102, 43
_v_pk_add_f16_vvs 105, 105, 39
_v_pk_add_f16_vvs 106, 106, 43
_v_pk_mul_f16_vsv 131, 30, 103
_v_pk_mul_f16_vsv 140, 30, 104
_v_pk_mul_f16_vsv 143, 30, 107
_v_pk_mul_f16_vsv 146, 30, 108
_v_pk_min_num_f16_vvv 103, 103, 131
_v_pk_min_num_f16_vvv 104, 104, 140
_v_pk_min_num_f16_vvv 107, 107, 143
_v_pk_min_num_f16_vvv 108, 108, 146
_v_pk_mul_f16_vsv 131, 30, 101
_v_pk_mul_f16_vsv 140, 30, 102
_v_pk_mul_f16_vsv 143, 30, 105
_v_pk_mul_f16_vsv 146, 30, 106
_v_pk_min_num_f16_vvv 101, 101, 131
_v_pk_min_num_f16_vvv 102, 102, 140
_v_pk_min_num_f16_vvv 105, 105, 143
_v_pk_min_num_f16_vvv 106, 106, 146
s_setprio 0
buffer_store_b16 v103, v134, s[60:63], null idxen
buffer_store_b16 v107, v138, s[60:63], null idxen
buffer_store_b16 v101, v135, s[60:63], null idxen
buffer_store_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v103, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v107, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v101, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v104, v134, s[60:63], null idxen
buffer_store_b16 v108, v138, s[60:63], null idxen
buffer_store_b16 v102, v135, s[60:63], null idxen
buffer_store_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v104, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v108, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v102, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x64d8
s_sub_co_u32 s67, s70, 17
s_cselect_b32 s62, 0, s62
s_add_nc_u64 s[60:61], s[18:19], s[68:69]
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 111, 111, 47
_v_pk_add_f16_vvs 112, 112, 51
_v_pk_add_f16_vvs 115, 115, 47
_v_pk_add_f16_vvs 116, 116, 51
_v_pk_add_f16_vvs 109, 109, 47
_v_pk_add_f16_vvs 110, 110, 51
_v_pk_add_f16_vvs 113, 113, 47
_v_pk_add_f16_vvs 114, 114, 51
_v_pk_mul_f16_vsv 131, 30, 111
_v_pk_mul_f16_vsv 140, 30, 112
_v_pk_mul_f16_vsv 143, 30, 115
_v_pk_mul_f16_vsv 146, 30, 116
_v_pk_min_num_f16_vvv 111, 111, 131
_v_pk_min_num_f16_vvv 112, 112, 140
_v_pk_min_num_f16_vvv 115, 115, 143
_v_pk_min_num_f16_vvv 116, 116, 146
_v_pk_mul_f16_vsv 131, 30, 109
_v_pk_mul_f16_vsv 140, 30, 110
_v_pk_mul_f16_vsv 143, 30, 113
_v_pk_mul_f16_vsv 146, 30, 114
_v_pk_min_num_f16_vvv 109, 109, 131
_v_pk_min_num_f16_vvv 110, 110, 140
_v_pk_min_num_f16_vvv 113, 113, 143
_v_pk_min_num_f16_vvv 114, 114, 146
s_setprio 0
buffer_store_b16 v111, v134, s[60:63], null idxen
buffer_store_b16 v115, v138, s[60:63], null idxen
buffer_store_b16 v109, v135, s[60:63], null idxen
buffer_store_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v111, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v115, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v109, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v112, v134, s[60:63], null idxen
buffer_store_b16 v116, v138, s[60:63], null idxen
buffer_store_b16 v110, v135, s[60:63], null idxen
buffer_store_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v112, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v116, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v110, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x66b0
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 119, 119, 52
_v_pk_add_f16_vvs 120, 120, 53
_v_pk_add_f16_vvs 123, 123, 52
_v_pk_add_f16_vvs 124, 124, 53
_v_pk_add_f16_vvs 117, 117, 52
_v_pk_add_f16_vvs 118, 118, 53
_v_pk_add_f16_vvs 121, 121, 52
_v_pk_add_f16_vvs 122, 122, 53
_v_pk_mul_f16_vsv 131, 30, 119
_v_pk_mul_f16_vsv 140, 30, 120
_v_pk_mul_f16_vsv 143, 30, 123
_v_pk_mul_f16_vsv 146, 30, 124
_v_pk_min_num_f16_vvv 119, 119, 131
_v_pk_min_num_f16_vvv 120, 120, 140
_v_pk_min_num_f16_vvv 123, 123, 143
_v_pk_min_num_f16_vvv 124, 124, 146
_v_pk_mul_f16_vsv 131, 30, 117
_v_pk_mul_f16_vsv 140, 30, 118
_v_pk_mul_f16_vsv 143, 30, 121
_v_pk_mul_f16_vsv 146, 30, 122
_v_pk_min_num_f16_vvv 117, 117, 131
_v_pk_min_num_f16_vvv 118, 118, 140
_v_pk_min_num_f16_vvv 121, 121, 143
_v_pk_min_num_f16_vvv 122, 122, 146
s_setprio 0
buffer_store_b16 v119, v134, s[60:63], null idxen
buffer_store_b16 v123, v138, s[60:63], null idxen
buffer_store_b16 v117, v135, s[60:63], null idxen
buffer_store_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v119, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v123, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v117, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v120, v134, s[60:63], null idxen
buffer_store_b16 v124, v138, s[60:63], null idxen
buffer_store_b16 v118, v135, s[60:63], null idxen
buffer_store_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v120, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v124, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v118, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x6118
s_bitcmp1_b32 s58, 0
s_cbranch_scc1 3051
s_sub_co_u32 s67, s70, 1
s_cselect_b32 s62, 0, 0x80000000
s_mov_b64 s[60:61], s[18:19]
s_setpc_b64 s[74:75]
s_setprio 3
v_readfirstlane_b32 s58, v132
_v_pk_add_f16_vvs 95, 95, 35
_v_pk_add_f16_vvs 96, 96, 38
_v_pk_add_f16_vvs 99, 99, 35
_v_pk_add_f16_vvs 100, 100, 38
_v_pk_add_f16_vvs 93, 93, 35
_v_pk_add_f16_vvs 94, 94, 38
_v_pk_add_f16_vvs 97, 97, 35
_v_pk_add_f16_vvs 98, 98, 38
_v_pk_max_num_f16_vconstv 95, 0, 95
_v_pk_max_num_f16_vconstv 96, 0, 96
_v_pk_max_num_f16_vconstv 99, 0, 99
_v_pk_max_num_f16_vconstv 100, 0, 100
_v_pk_max_num_f16_vconstv 93, 0, 93
_v_pk_max_num_f16_vconstv 94, 0, 94
_v_pk_max_num_f16_vconstv 97, 0, 97
_v_pk_max_num_f16_vconstv 98, 0, 98
s_setprio 0
buffer_store_b16 v95, v134, s[60:63], null idxen
buffer_store_b16 v99, v138, s[60:63], null idxen
buffer_store_b16 v93, v135, s[60:63], null idxen
buffer_store_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v95, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v99, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v93, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v96, v134, s[60:63], null idxen
buffer_store_b16 v100, v138, s[60:63], null idxen
buffer_store_b16 v94, v135, s[60:63], null idxen
buffer_store_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v96, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v100, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v94, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x6a3c
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 103, 103, 39
_v_pk_add_f16_vvs 104, 104, 43
_v_pk_add_f16_vvs 107, 107, 39
_v_pk_add_f16_vvs 108, 108, 43
_v_pk_add_f16_vvs 101, 101, 39
_v_pk_add_f16_vvs 102, 102, 43
_v_pk_add_f16_vvs 105, 105, 39
_v_pk_add_f16_vvs 106, 106, 43
_v_pk_max_num_f16_vconstv 103, 0, 103
_v_pk_max_num_f16_vconstv 104, 0, 104
_v_pk_max_num_f16_vconstv 107, 0, 107
_v_pk_max_num_f16_vconstv 108, 0, 108
_v_pk_max_num_f16_vconstv 101, 0, 101
_v_pk_max_num_f16_vconstv 102, 0, 102
_v_pk_max_num_f16_vconstv 105, 0, 105
_v_pk_max_num_f16_vconstv 106, 0, 106
s_setprio 0
buffer_store_b16 v103, v134, s[60:63], null idxen
buffer_store_b16 v107, v138, s[60:63], null idxen
buffer_store_b16 v101, v135, s[60:63], null idxen
buffer_store_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v103, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v107, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v101, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v104, v134, s[60:63], null idxen
buffer_store_b16 v108, v138, s[60:63], null idxen
buffer_store_b16 v102, v135, s[60:63], null idxen
buffer_store_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v104, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v108, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v102, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x6be0
s_sub_co_u32 s67, s70, 17
s_cselect_b32 s62, 0, s62
s_add_nc_u64 s[60:61], s[18:19], s[68:69]
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 111, 111, 47
_v_pk_add_f16_vvs 112, 112, 51
_v_pk_add_f16_vvs 115, 115, 47
_v_pk_add_f16_vvs 116, 116, 51
_v_pk_add_f16_vvs 109, 109, 47
_v_pk_add_f16_vvs 110, 110, 51
_v_pk_add_f16_vvs 113, 113, 47
_v_pk_add_f16_vvs 114, 114, 51
_v_pk_max_num_f16_vconstv 111, 0, 111
_v_pk_max_num_f16_vconstv 112, 0, 112
_v_pk_max_num_f16_vconstv 115, 0, 115
_v_pk_max_num_f16_vconstv 116, 0, 116
_v_pk_max_num_f16_vconstv 109, 0, 109
_v_pk_max_num_f16_vconstv 110, 0, 110
_v_pk_max_num_f16_vconstv 113, 0, 113
_v_pk_max_num_f16_vconstv 114, 0, 114
s_setprio 0
buffer_store_b16 v111, v134, s[60:63], null idxen
buffer_store_b16 v115, v138, s[60:63], null idxen
buffer_store_b16 v109, v135, s[60:63], null idxen
buffer_store_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v111, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v115, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v109, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v112, v134, s[60:63], null idxen
buffer_store_b16 v116, v138, s[60:63], null idxen
buffer_store_b16 v110, v135, s[60:63], null idxen
buffer_store_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v112, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v116, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v110, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x6d78
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 119, 119, 52
_v_pk_add_f16_vvs 120, 120, 53
_v_pk_add_f16_vvs 123, 123, 52
_v_pk_add_f16_vvs 124, 124, 53
_v_pk_add_f16_vvs 117, 117, 52
_v_pk_add_f16_vvs 118, 118, 53
_v_pk_add_f16_vvs 121, 121, 52
_v_pk_add_f16_vvs 122, 122, 53
_v_pk_max_num_f16_vconstv 119, 0, 119
_v_pk_max_num_f16_vconstv 120, 0, 120
_v_pk_max_num_f16_vconstv 123, 0, 123
_v_pk_max_num_f16_vconstv 124, 0, 124
_v_pk_max_num_f16_vconstv 117, 0, 117
_v_pk_max_num_f16_vconstv 118, 0, 118
_v_pk_max_num_f16_vconstv 121, 0, 121
_v_pk_max_num_f16_vconstv 122, 0, 122
s_setprio 0
buffer_store_b16 v119, v134, s[60:63], null idxen
buffer_store_b16 v123, v138, s[60:63], null idxen
buffer_store_b16 v117, v135, s[60:63], null idxen
buffer_store_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v119, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v123, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v117, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v120, v134, s[60:63], null idxen
buffer_store_b16 v124, v138, s[60:63], null idxen
buffer_store_b16 v118, v135, s[60:63], null idxen
buffer_store_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v120, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v124, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v118, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x68a0
s_bitcmp1_b32 s58, 0
s_cbranch_scc1 2633
s_sub_co_u32 s67, s70, 1
s_cselect_b32 s62, 0, 0x80000000
s_mov_b64 s[60:61], s[18:19]
s_setpc_b64 s[74:75]
s_setprio 3
v_readfirstlane_b32 s58, v132
_v_pk_add_f16_vvs 95, 95, 35
_v_pk_add_f16_vvs 96, 96, 38
_v_pk_add_f16_vvs 99, 99, 35
_v_pk_add_f16_vvs 100, 100, 38
_v_pk_add_f16_vvs 93, 93, 35
_v_pk_add_f16_vvs 94, 94, 38
_v_pk_add_f16_vvs 97, 97, 35
_v_pk_add_f16_vvs 98, 98, 38
v_mov_b32_e32 v133, 0xbfb8aa3b
v_mov_b32_e32 v141, 0xbfb8aa3b
v_mov_b32_e32 v144, 0xbfb8aa3b
v_mov_b32_e32 v147, 0xbfb8aa3b
v_fma_mix_f32 v131, v133, v95, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v140, v141, v96, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v143, v144, v99, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v146, v147, v100, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v133, v133, v95, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v141, v141, v96, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v144, v144, v99, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v147, v147, v100, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v131, v131
v_exp_f32_e32 v140, v140
v_exp_f32_e32 v143, v143
v_exp_f32_e32 v146, v146
v_exp_f32_e32 v133, v133
v_exp_f32_e32 v141, v141
v_exp_f32_e32 v144, v144
v_exp_f32_e32 v147, v147
v_add_f32_e32 v131, 1.0, v131
v_add_f32_e32 v140, 1.0, v140
v_add_f32_e32 v143, 1.0, v143
v_add_f32_e32 v146, 1.0, v146
v_add_f32_e32 v133, 1.0, v133
v_add_f32_e32 v141, 1.0, v141
v_add_f32_e32 v144, 1.0, v144
v_add_f32_e32 v147, 1.0, v147
v_rcp_f32_e32 v131, v131
v_rcp_f32_e32 v140, v140
v_rcp_f32_e32 v143, v143
v_rcp_f32_e32 v146, v146
v_rcp_f32_e32 v133, v133
v_rcp_f32_e32 v141, v141
v_rcp_f32_e32 v144, v144
v_rcp_f32_e32 v147, v147
v_cvt_f16_f32_e32 v95, v131
v_cvt_f16_f32_e32 v96, v140
v_cvt_f16_f32_e32 v99, v143
v_cvt_f16_f32_e32 v100, v146
_v_cvt_f16_f32_hvv 95, 133
_v_cvt_f16_f32_hvv 96, 141
_v_cvt_f16_f32_hvv 99, 144
_v_cvt_f16_f32_hvv 100, 147
v_mov_b32_e32 v133, 0xbfb8aa3b
v_mov_b32_e32 v141, 0xbfb8aa3b
v_mov_b32_e32 v144, 0xbfb8aa3b
v_mov_b32_e32 v147, 0xbfb8aa3b
v_fma_mix_f32 v131, v133, v93, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v140, v141, v94, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v143, v144, v97, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v146, v147, v98, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v133, v133, v93, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v141, v141, v94, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v144, v144, v97, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v147, v147, v98, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v131, v131
v_exp_f32_e32 v140, v140
v_exp_f32_e32 v143, v143
v_exp_f32_e32 v146, v146
v_exp_f32_e32 v133, v133
v_exp_f32_e32 v141, v141
v_exp_f32_e32 v144, v144
v_exp_f32_e32 v147, v147
v_add_f32_e32 v131, 1.0, v131
v_add_f32_e32 v140, 1.0, v140
v_add_f32_e32 v143, 1.0, v143
v_add_f32_e32 v146, 1.0, v146
v_add_f32_e32 v133, 1.0, v133
v_add_f32_e32 v141, 1.0, v141
v_add_f32_e32 v144, 1.0, v144
v_add_f32_e32 v147, 1.0, v147
v_rcp_f32_e32 v131, v131
v_rcp_f32_e32 v140, v140
v_rcp_f32_e32 v143, v143
v_rcp_f32_e32 v146, v146
v_rcp_f32_e32 v133, v133
v_rcp_f32_e32 v141, v141
v_rcp_f32_e32 v144, v144
v_rcp_f32_e32 v147, v147
v_cvt_f16_f32_e32 v93, v131
v_cvt_f16_f32_e32 v94, v140
v_cvt_f16_f32_e32 v97, v143
v_cvt_f16_f32_e32 v98, v146
_v_cvt_f16_f32_hvv 93, 133
_v_cvt_f16_f32_hvv 94, 141
_v_cvt_f16_f32_hvv 97, 144
_v_cvt_f16_f32_hvv 98, 147
s_setprio 0
buffer_store_b16 v95, v134, s[60:63], null idxen
buffer_store_b16 v99, v138, s[60:63], null idxen
buffer_store_b16 v93, v135, s[60:63], null idxen
buffer_store_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v95, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v99, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v93, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v96, v134, s[60:63], null idxen
buffer_store_b16 v100, v138, s[60:63], null idxen
buffer_store_b16 v94, v135, s[60:63], null idxen
buffer_store_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v96, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v100, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v94, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x7244
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 103, 103, 39
_v_pk_add_f16_vvs 104, 104, 43
_v_pk_add_f16_vvs 107, 107, 39
_v_pk_add_f16_vvs 108, 108, 43
_v_pk_add_f16_vvs 101, 101, 39
_v_pk_add_f16_vvs 102, 102, 43
_v_pk_add_f16_vvs 105, 105, 39
_v_pk_add_f16_vvs 106, 106, 43
v_mov_b32_e32 v133, 0xbfb8aa3b
v_mov_b32_e32 v141, 0xbfb8aa3b
v_mov_b32_e32 v144, 0xbfb8aa3b
v_mov_b32_e32 v147, 0xbfb8aa3b
v_fma_mix_f32 v131, v133, v103, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v140, v141, v104, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v143, v144, v107, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v146, v147, v108, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v133, v133, v103, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v141, v141, v104, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v144, v144, v107, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v147, v147, v108, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v131, v131
v_exp_f32_e32 v140, v140
v_exp_f32_e32 v143, v143
v_exp_f32_e32 v146, v146
v_exp_f32_e32 v133, v133
v_exp_f32_e32 v141, v141
v_exp_f32_e32 v144, v144
v_exp_f32_e32 v147, v147
v_add_f32_e32 v131, 1.0, v131
v_add_f32_e32 v140, 1.0, v140
v_add_f32_e32 v143, 1.0, v143
v_add_f32_e32 v146, 1.0, v146
v_add_f32_e32 v133, 1.0, v133
v_add_f32_e32 v141, 1.0, v141
v_add_f32_e32 v144, 1.0, v144
v_add_f32_e32 v147, 1.0, v147
v_rcp_f32_e32 v131, v131
v_rcp_f32_e32 v140, v140
v_rcp_f32_e32 v143, v143
v_rcp_f32_e32 v146, v146
v_rcp_f32_e32 v133, v133
v_rcp_f32_e32 v141, v141
v_rcp_f32_e32 v144, v144
v_rcp_f32_e32 v147, v147
v_cvt_f16_f32_e32 v103, v131
v_cvt_f16_f32_e32 v104, v140
v_cvt_f16_f32_e32 v107, v143
v_cvt_f16_f32_e32 v108, v146
_v_cvt_f16_f32_hvv 103, 133
_v_cvt_f16_f32_hvv 104, 141
_v_cvt_f16_f32_hvv 107, 144
_v_cvt_f16_f32_hvv 108, 147
v_mov_b32_e32 v133, 0xbfb8aa3b
v_mov_b32_e32 v141, 0xbfb8aa3b
v_mov_b32_e32 v144, 0xbfb8aa3b
v_mov_b32_e32 v147, 0xbfb8aa3b
v_fma_mix_f32 v131, v133, v101, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v140, v141, v102, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v143, v144, v105, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v146, v147, v106, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v133, v133, v101, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v141, v141, v102, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v144, v144, v105, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v147, v147, v106, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v131, v131
v_exp_f32_e32 v140, v140
v_exp_f32_e32 v143, v143
v_exp_f32_e32 v146, v146
v_exp_f32_e32 v133, v133
v_exp_f32_e32 v141, v141
v_exp_f32_e32 v144, v144
v_exp_f32_e32 v147, v147
v_add_f32_e32 v131, 1.0, v131
v_add_f32_e32 v140, 1.0, v140
v_add_f32_e32 v143, 1.0, v143
v_add_f32_e32 v146, 1.0, v146
v_add_f32_e32 v133, 1.0, v133
v_add_f32_e32 v141, 1.0, v141
v_add_f32_e32 v144, 1.0, v144
v_add_f32_e32 v147, 1.0, v147
v_rcp_f32_e32 v131, v131
v_rcp_f32_e32 v140, v140
v_rcp_f32_e32 v143, v143
v_rcp_f32_e32 v146, v146
v_rcp_f32_e32 v133, v133
v_rcp_f32_e32 v141, v141
v_rcp_f32_e32 v144, v144
v_rcp_f32_e32 v147, v147
v_cvt_f16_f32_e32 v101, v131
v_cvt_f16_f32_e32 v102, v140
v_cvt_f16_f32_e32 v105, v143
v_cvt_f16_f32_e32 v106, v146
_v_cvt_f16_f32_hvv 101, 133
_v_cvt_f16_f32_hvv 102, 141
_v_cvt_f16_f32_hvv 105, 144
_v_cvt_f16_f32_hvv 106, 147
s_setprio 0
buffer_store_b16 v103, v134, s[60:63], null idxen
buffer_store_b16 v107, v138, s[60:63], null idxen
buffer_store_b16 v101, v135, s[60:63], null idxen
buffer_store_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v103, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v107, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v101, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v104, v134, s[60:63], null idxen
buffer_store_b16 v108, v138, s[60:63], null idxen
buffer_store_b16 v102, v135, s[60:63], null idxen
buffer_store_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v104, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v108, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v102, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x7568
s_sub_co_u32 s67, s70, 17
s_cselect_b32 s62, 0, s62
s_add_nc_u64 s[60:61], s[18:19], s[68:69]
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 111, 111, 47
_v_pk_add_f16_vvs 112, 112, 51
_v_pk_add_f16_vvs 115, 115, 47
_v_pk_add_f16_vvs 116, 116, 51
_v_pk_add_f16_vvs 109, 109, 47
_v_pk_add_f16_vvs 110, 110, 51
_v_pk_add_f16_vvs 113, 113, 47
_v_pk_add_f16_vvs 114, 114, 51
v_mov_b32_e32 v133, 0xbfb8aa3b
v_mov_b32_e32 v141, 0xbfb8aa3b
v_mov_b32_e32 v144, 0xbfb8aa3b
v_mov_b32_e32 v147, 0xbfb8aa3b
v_fma_mix_f32 v131, v133, v111, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v140, v141, v112, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v143, v144, v115, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v146, v147, v116, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v133, v133, v111, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v141, v141, v112, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v144, v144, v115, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v147, v147, v116, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v131, v131
v_exp_f32_e32 v140, v140
v_exp_f32_e32 v143, v143
v_exp_f32_e32 v146, v146
v_exp_f32_e32 v133, v133
v_exp_f32_e32 v141, v141
v_exp_f32_e32 v144, v144
v_exp_f32_e32 v147, v147
v_add_f32_e32 v131, 1.0, v131
v_add_f32_e32 v140, 1.0, v140
v_add_f32_e32 v143, 1.0, v143
v_add_f32_e32 v146, 1.0, v146
v_add_f32_e32 v133, 1.0, v133
v_add_f32_e32 v141, 1.0, v141
v_add_f32_e32 v144, 1.0, v144
v_add_f32_e32 v147, 1.0, v147
v_rcp_f32_e32 v131, v131
v_rcp_f32_e32 v140, v140
v_rcp_f32_e32 v143, v143
v_rcp_f32_e32 v146, v146
v_rcp_f32_e32 v133, v133
v_rcp_f32_e32 v141, v141
v_rcp_f32_e32 v144, v144
v_rcp_f32_e32 v147, v147
v_cvt_f16_f32_e32 v111, v131
v_cvt_f16_f32_e32 v112, v140
v_cvt_f16_f32_e32 v115, v143
v_cvt_f16_f32_e32 v116, v146
_v_cvt_f16_f32_hvv 111, 133
_v_cvt_f16_f32_hvv 112, 141
_v_cvt_f16_f32_hvv 115, 144
_v_cvt_f16_f32_hvv 116, 147
v_mov_b32_e32 v133, 0xbfb8aa3b
v_mov_b32_e32 v141, 0xbfb8aa3b
v_mov_b32_e32 v144, 0xbfb8aa3b
v_mov_b32_e32 v147, 0xbfb8aa3b
v_fma_mix_f32 v131, v133, v109, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v140, v141, v110, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v143, v144, v113, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v146, v147, v114, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v133, v133, v109, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v141, v141, v110, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v144, v144, v113, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v147, v147, v114, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v131, v131
v_exp_f32_e32 v140, v140
v_exp_f32_e32 v143, v143
v_exp_f32_e32 v146, v146
v_exp_f32_e32 v133, v133
v_exp_f32_e32 v141, v141
v_exp_f32_e32 v144, v144
v_exp_f32_e32 v147, v147
v_add_f32_e32 v131, 1.0, v131
v_add_f32_e32 v140, 1.0, v140
v_add_f32_e32 v143, 1.0, v143
v_add_f32_e32 v146, 1.0, v146
v_add_f32_e32 v133, 1.0, v133
v_add_f32_e32 v141, 1.0, v141
v_add_f32_e32 v144, 1.0, v144
v_add_f32_e32 v147, 1.0, v147
v_rcp_f32_e32 v131, v131
v_rcp_f32_e32 v140, v140
v_rcp_f32_e32 v143, v143
v_rcp_f32_e32 v146, v146
v_rcp_f32_e32 v133, v133
v_rcp_f32_e32 v141, v141
v_rcp_f32_e32 v144, v144
v_rcp_f32_e32 v147, v147
v_cvt_f16_f32_e32 v109, v131
v_cvt_f16_f32_e32 v110, v140
v_cvt_f16_f32_e32 v113, v143
v_cvt_f16_f32_e32 v114, v146
_v_cvt_f16_f32_hvv 109, 133
_v_cvt_f16_f32_hvv 110, 141
_v_cvt_f16_f32_hvv 113, 144
_v_cvt_f16_f32_hvv 114, 147
s_setprio 0
buffer_store_b16 v111, v134, s[60:63], null idxen
buffer_store_b16 v115, v138, s[60:63], null idxen
buffer_store_b16 v109, v135, s[60:63], null idxen
buffer_store_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v111, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v115, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v109, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v112, v134, s[60:63], null idxen
buffer_store_b16 v116, v138, s[60:63], null idxen
buffer_store_b16 v110, v135, s[60:63], null idxen
buffer_store_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v112, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v116, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v110, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x7880
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 119, 119, 52
_v_pk_add_f16_vvs 120, 120, 53
_v_pk_add_f16_vvs 123, 123, 52
_v_pk_add_f16_vvs 124, 124, 53
_v_pk_add_f16_vvs 117, 117, 52
_v_pk_add_f16_vvs 118, 118, 53
_v_pk_add_f16_vvs 121, 121, 52
_v_pk_add_f16_vvs 122, 122, 53
v_mov_b32_e32 v133, 0xbfb8aa3b
v_mov_b32_e32 v141, 0xbfb8aa3b
v_mov_b32_e32 v144, 0xbfb8aa3b
v_mov_b32_e32 v147, 0xbfb8aa3b
v_fma_mix_f32 v131, v133, v119, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v140, v141, v120, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v143, v144, v123, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v146, v147, v124, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v133, v133, v119, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v141, v141, v120, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v144, v144, v123, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v147, v147, v124, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v131, v131
v_exp_f32_e32 v140, v140
v_exp_f32_e32 v143, v143
v_exp_f32_e32 v146, v146
v_exp_f32_e32 v133, v133
v_exp_f32_e32 v141, v141
v_exp_f32_e32 v144, v144
v_exp_f32_e32 v147, v147
v_add_f32_e32 v131, 1.0, v131
v_add_f32_e32 v140, 1.0, v140
v_add_f32_e32 v143, 1.0, v143
v_add_f32_e32 v146, 1.0, v146
v_add_f32_e32 v133, 1.0, v133
v_add_f32_e32 v141, 1.0, v141
v_add_f32_e32 v144, 1.0, v144
v_add_f32_e32 v147, 1.0, v147
v_rcp_f32_e32 v131, v131
v_rcp_f32_e32 v140, v140
v_rcp_f32_e32 v143, v143
v_rcp_f32_e32 v146, v146
v_rcp_f32_e32 v133, v133
v_rcp_f32_e32 v141, v141
v_rcp_f32_e32 v144, v144
v_rcp_f32_e32 v147, v147
v_cvt_f16_f32_e32 v119, v131
v_cvt_f16_f32_e32 v120, v140
v_cvt_f16_f32_e32 v123, v143
v_cvt_f16_f32_e32 v124, v146
_v_cvt_f16_f32_hvv 119, 133
_v_cvt_f16_f32_hvv 120, 141
_v_cvt_f16_f32_hvv 123, 144
_v_cvt_f16_f32_hvv 124, 147
v_mov_b32_e32 v133, 0xbfb8aa3b
v_mov_b32_e32 v141, 0xbfb8aa3b
v_mov_b32_e32 v144, 0xbfb8aa3b
v_mov_b32_e32 v147, 0xbfb8aa3b
v_fma_mix_f32 v131, v133, v117, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v140, v141, v118, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v143, v144, v121, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v146, v147, v122, 0 op_sel_hi:[0,1,0]
v_fma_mix_f32 v133, v133, v117, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v141, v141, v118, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v144, v144, v121, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_fma_mix_f32 v147, v147, v122, 0 op_sel:[0,1,0] op_sel_hi:[0,1,0]
v_exp_f32_e32 v131, v131
v_exp_f32_e32 v140, v140
v_exp_f32_e32 v143, v143
v_exp_f32_e32 v146, v146
v_exp_f32_e32 v133, v133
v_exp_f32_e32 v141, v141
v_exp_f32_e32 v144, v144
v_exp_f32_e32 v147, v147
v_add_f32_e32 v131, 1.0, v131
v_add_f32_e32 v140, 1.0, v140
v_add_f32_e32 v143, 1.0, v143
v_add_f32_e32 v146, 1.0, v146
v_add_f32_e32 v133, 1.0, v133
v_add_f32_e32 v141, 1.0, v141
v_add_f32_e32 v144, 1.0, v144
v_add_f32_e32 v147, 1.0, v147
v_rcp_f32_e32 v131, v131
v_rcp_f32_e32 v140, v140
v_rcp_f32_e32 v143, v143
v_rcp_f32_e32 v146, v146
v_rcp_f32_e32 v133, v133
v_rcp_f32_e32 v141, v141
v_rcp_f32_e32 v144, v144
v_rcp_f32_e32 v147, v147
v_cvt_f16_f32_e32 v117, v131
v_cvt_f16_f32_e32 v118, v140
v_cvt_f16_f32_e32 v121, v143
v_cvt_f16_f32_e32 v122, v146
_v_cvt_f16_f32_hvv 117, 133
_v_cvt_f16_f32_hvv 118, 141
_v_cvt_f16_f32_hvv 121, 144
_v_cvt_f16_f32_hvv 122, 147
s_setprio 0
buffer_store_b16 v119, v134, s[60:63], null idxen
buffer_store_b16 v123, v138, s[60:63], null idxen
buffer_store_b16 v117, v135, s[60:63], null idxen
buffer_store_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v119, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v123, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v117, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v120, v134, s[60:63], null idxen
buffer_store_b16 v124, v138, s[60:63], null idxen
buffer_store_b16 v118, v135, s[60:63], null idxen
buffer_store_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v120, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v124, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v118, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x6f28
s_bitcmp1_b32 s58, 0
s_cbranch_scc1 1831
s_sub_co_u32 s67, s70, 1
s_cselect_b32 s62, 0, 0x80000000
s_mov_b64 s[60:61], s[18:19]
s_setpc_b64 s[74:75]
s_setprio 3
v_readfirstlane_b32 s58, v132
_v_pk_add_f16_vvs 95, 95, 35
_v_pk_add_f16_vvs 96, 96, 38
_v_pk_add_f16_vvs 99, 99, 35
_v_pk_add_f16_vvs 100, 100, 38
_v_pk_add_f16_vvs 93, 93, 35
_v_pk_add_f16_vvs 94, 94, 38
_v_pk_add_f16_vvs 97, 97, 35
_v_pk_add_f16_vvs 98, 98, 38
_v_pk_mul_f16_vvs 95, 95, 31
_v_pk_mul_f16_vvs 96, 96, 31
_v_pk_mul_f16_vvs 99, 99, 31
_v_pk_mul_f16_vvs 100, 100, 31
v_and_b32_e32 v131, 0x7fff7fff, v95
v_and_b32_e32 v140, 0x7fff7fff, v96
v_and_b32_e32 v143, 0x7fff7fff, v99
v_and_b32_e32 v146, 0x7fff7fff, v100
v_mov_b32_e32 v133, 0xb62eb62e
v_mov_b32_e32 v141, 0xb62eb62e
v_mov_b32_e32 v144, 0xb62eb62e
v_mov_b32_e32 v147, 0xb62eb62e
v_pk_fma_f16 v133, v131, 0x30883088, v133
v_pk_fma_f16 v141, v140, 0x30883088, v141
v_pk_fma_f16 v144, v143, 0x30883088, v144
v_pk_fma_f16 v147, v146, 0x30883088, v147
v_pk_fma_f16 v133, v131, v133, 0x1e331e33
v_pk_fma_f16 v141, v140, v141, 0x1e331e33
v_pk_fma_f16 v144, v143, v144, 0x1e331e33
v_pk_fma_f16 v147, v146, v147, 0x1e331e33
_v_pk_mul_f16_vvv 133, 131, 133
_v_pk_mul_f16_vvv 141, 140, 141
_v_pk_mul_f16_vvv 144, 143, 144
_v_pk_mul_f16_vvv 147, 146, 147
v_pk_fma_f16 v133, v131, v133, v131
v_pk_fma_f16 v141, v140, v141, v140
v_pk_fma_f16 v144, v143, v144, v143
v_pk_fma_f16 v147, v146, v147, v146
_v_pk_mul_f16_vlitv 131, 0x41c541c5, 131
_v_pk_mul_f16_vlitv 140, 0x41c541c5, 140
_v_pk_mul_f16_vlitv 143, 0x41c541c5, 143
_v_pk_mul_f16_vlitv 146, 0x41c541c5, 146
v_exp_f16_e64 v131, v131
v_exp_f16_e64 v140, v140
v_exp_f16_e64 v143, v143
v_exp_f16_e64 v146, v146
_v_exp_f16_hvhv 131, 131
_v_exp_f16_hvhv 140, 140
_v_exp_f16_hvhv 143, 143
_v_exp_f16_hvhv 146, 146
_v_pk_add_f16_vconstv 131, 1.0, 131
_v_pk_add_f16_vconstv 140, 1.0, 140
_v_pk_add_f16_vconstv 143, 1.0, 143
_v_pk_add_f16_vconstv 146, 1.0, 146
v_rcp_f16_e64 v131, v131
v_rcp_f16_e64 v140, v140
v_rcp_f16_e64 v143, v143
v_rcp_f16_e64 v146, v146
_v_rcp_f16_hvhv 131, 131
_v_rcp_f16_hvhv 140, 140
_v_rcp_f16_hvhv 143, 143
_v_rcp_f16_hvhv 146, 146
v_mov_b32_e32 v137, 0x3bc23bc2
v_mov_b32_e32 v142, 0x3bc23bc2
v_mov_b32_e32 v145, 0x3bc23bc2
v_mov_b32_e32 v148, 0x3bc23bc2
v_pk_fma_f16 v131, v131, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v140, v140, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v143, v143, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v146, v146, -2.0, 1.0 op_sel_hi:[1,0,0]
v_cmp_gt_f16_e64 vcc_lo, |v95|, v137
v_cndmask_b16 v133, v133, v131, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v96|, v142
v_cndmask_b16 v141, v141, v140, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v99|, v145
v_cndmask_b16 v144, v144, v143, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v100|, v148
v_cndmask_b16 v147, v147, v146, vcc_lo
_v_cmp_gt_f16_vccahvhv 95, 137
_v_cndmask_b16_hvhvhvvcc 133, 133, 131
_v_cmp_gt_f16_vccahvhv 96, 142
_v_cndmask_b16_hvhvhvvcc 141, 141, 140
_v_cmp_gt_f16_vccahvhv 99, 145
_v_cndmask_b16_hvhvhvvcc 144, 144, 143
_v_cmp_gt_f16_vccahvhv 100, 148
_v_cndmask_b16_hvhvhvvcc 147, 147, 146
v_bfi_b32 v95, 0x7fff7fff, v133, v95
v_bfi_b32 v96, 0x7fff7fff, v141, v96
v_bfi_b32 v99, 0x7fff7fff, v144, v99
v_bfi_b32 v100, 0x7fff7fff, v147, v100
_v_pk_mul_f16_vvs 95, 95, 30
_v_pk_mul_f16_vvs 96, 96, 30
_v_pk_mul_f16_vvs 99, 99, 30
_v_pk_mul_f16_vvs 100, 100, 30
_v_pk_mul_f16_vvs 93, 93, 31
_v_pk_mul_f16_vvs 94, 94, 31
_v_pk_mul_f16_vvs 97, 97, 31
_v_pk_mul_f16_vvs 98, 98, 31
v_and_b32_e32 v131, 0x7fff7fff, v93
v_and_b32_e32 v140, 0x7fff7fff, v94
v_and_b32_e32 v143, 0x7fff7fff, v97
v_and_b32_e32 v146, 0x7fff7fff, v98
v_mov_b32_e32 v133, 0xb62eb62e
v_mov_b32_e32 v141, 0xb62eb62e
v_mov_b32_e32 v144, 0xb62eb62e
v_mov_b32_e32 v147, 0xb62eb62e
v_pk_fma_f16 v133, v131, 0x30883088, v133
v_pk_fma_f16 v141, v140, 0x30883088, v141
v_pk_fma_f16 v144, v143, 0x30883088, v144
v_pk_fma_f16 v147, v146, 0x30883088, v147
v_pk_fma_f16 v133, v131, v133, 0x1e331e33
v_pk_fma_f16 v141, v140, v141, 0x1e331e33
v_pk_fma_f16 v144, v143, v144, 0x1e331e33
v_pk_fma_f16 v147, v146, v147, 0x1e331e33
_v_pk_mul_f16_vvv 133, 131, 133
_v_pk_mul_f16_vvv 141, 140, 141
_v_pk_mul_f16_vvv 144, 143, 144
_v_pk_mul_f16_vvv 147, 146, 147
v_pk_fma_f16 v133, v131, v133, v131
v_pk_fma_f16 v141, v140, v141, v140
v_pk_fma_f16 v144, v143, v144, v143
v_pk_fma_f16 v147, v146, v147, v146
_v_pk_mul_f16_vlitv 131, 0x41c541c5, 131
_v_pk_mul_f16_vlitv 140, 0x41c541c5, 140
_v_pk_mul_f16_vlitv 143, 0x41c541c5, 143
_v_pk_mul_f16_vlitv 146, 0x41c541c5, 146
v_exp_f16_e64 v131, v131
v_exp_f16_e64 v140, v140
v_exp_f16_e64 v143, v143
v_exp_f16_e64 v146, v146
_v_exp_f16_hvhv 131, 131
_v_exp_f16_hvhv 140, 140
_v_exp_f16_hvhv 143, 143
_v_exp_f16_hvhv 146, 146
_v_pk_add_f16_vconstv 131, 1.0, 131
_v_pk_add_f16_vconstv 140, 1.0, 140
_v_pk_add_f16_vconstv 143, 1.0, 143
_v_pk_add_f16_vconstv 146, 1.0, 146
v_rcp_f16_e64 v131, v131
v_rcp_f16_e64 v140, v140
v_rcp_f16_e64 v143, v143
v_rcp_f16_e64 v146, v146
_v_rcp_f16_hvhv 131, 131
_v_rcp_f16_hvhv 140, 140
_v_rcp_f16_hvhv 143, 143
_v_rcp_f16_hvhv 146, 146
v_mov_b32_e32 v137, 0x3bc23bc2
v_mov_b32_e32 v142, 0x3bc23bc2
v_mov_b32_e32 v145, 0x3bc23bc2
v_mov_b32_e32 v148, 0x3bc23bc2
v_pk_fma_f16 v131, v131, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v140, v140, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v143, v143, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v146, v146, -2.0, 1.0 op_sel_hi:[1,0,0]
v_cmp_gt_f16_e64 vcc_lo, |v93|, v137
v_cndmask_b16 v133, v133, v131, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v94|, v142
v_cndmask_b16 v141, v141, v140, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v97|, v145
v_cndmask_b16 v144, v144, v143, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v98|, v148
v_cndmask_b16 v147, v147, v146, vcc_lo
_v_cmp_gt_f16_vccahvhv 93, 137
_v_cndmask_b16_hvhvhvvcc 133, 133, 131
_v_cmp_gt_f16_vccahvhv 94, 142
_v_cndmask_b16_hvhvhvvcc 141, 141, 140
_v_cmp_gt_f16_vccahvhv 97, 145
_v_cndmask_b16_hvhvhvvcc 144, 144, 143
_v_cmp_gt_f16_vccahvhv 98, 148
_v_cndmask_b16_hvhvhvvcc 147, 147, 146
v_bfi_b32 v93, 0x7fff7fff, v133, v93
v_bfi_b32 v94, 0x7fff7fff, v141, v94
v_bfi_b32 v97, 0x7fff7fff, v144, v97
v_bfi_b32 v98, 0x7fff7fff, v147, v98
_v_pk_mul_f16_vvs 93, 93, 30
_v_pk_mul_f16_vvs 94, 94, 30
_v_pk_mul_f16_vvs 97, 97, 30
_v_pk_mul_f16_vvs 98, 98, 30
s_setprio 0
buffer_store_b16 v95, v134, s[60:63], null idxen
buffer_store_b16 v99, v138, s[60:63], null idxen
buffer_store_b16 v93, v135, s[60:63], null idxen
buffer_store_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v95, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v99, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v93, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v97, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v96, v134, s[60:63], null idxen
buffer_store_b16 v100, v138, s[60:63], null idxen
buffer_store_b16 v94, v135, s[60:63], null idxen
buffer_store_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v96, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v100, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v94, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v98, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x82cc
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 103, 103, 39
_v_pk_add_f16_vvs 104, 104, 43
_v_pk_add_f16_vvs 107, 107, 39
_v_pk_add_f16_vvs 108, 108, 43
_v_pk_add_f16_vvs 101, 101, 39
_v_pk_add_f16_vvs 102, 102, 43
_v_pk_add_f16_vvs 105, 105, 39
_v_pk_add_f16_vvs 106, 106, 43
_v_pk_mul_f16_vvs 103, 103, 31
_v_pk_mul_f16_vvs 104, 104, 31
_v_pk_mul_f16_vvs 107, 107, 31
_v_pk_mul_f16_vvs 108, 108, 31
v_and_b32_e32 v131, 0x7fff7fff, v103
v_and_b32_e32 v140, 0x7fff7fff, v104
v_and_b32_e32 v143, 0x7fff7fff, v107
v_and_b32_e32 v146, 0x7fff7fff, v108
v_mov_b32_e32 v133, 0xb62eb62e
v_mov_b32_e32 v141, 0xb62eb62e
v_mov_b32_e32 v144, 0xb62eb62e
v_mov_b32_e32 v147, 0xb62eb62e
v_pk_fma_f16 v133, v131, 0x30883088, v133
v_pk_fma_f16 v141, v140, 0x30883088, v141
v_pk_fma_f16 v144, v143, 0x30883088, v144
v_pk_fma_f16 v147, v146, 0x30883088, v147
v_pk_fma_f16 v133, v131, v133, 0x1e331e33
v_pk_fma_f16 v141, v140, v141, 0x1e331e33
v_pk_fma_f16 v144, v143, v144, 0x1e331e33
v_pk_fma_f16 v147, v146, v147, 0x1e331e33
_v_pk_mul_f16_vvv 133, 131, 133
_v_pk_mul_f16_vvv 141, 140, 141
_v_pk_mul_f16_vvv 144, 143, 144
_v_pk_mul_f16_vvv 147, 146, 147
v_pk_fma_f16 v133, v131, v133, v131
v_pk_fma_f16 v141, v140, v141, v140
v_pk_fma_f16 v144, v143, v144, v143
v_pk_fma_f16 v147, v146, v147, v146
_v_pk_mul_f16_vlitv 131, 0x41c541c5, 131
_v_pk_mul_f16_vlitv 140, 0x41c541c5, 140
_v_pk_mul_f16_vlitv 143, 0x41c541c5, 143
_v_pk_mul_f16_vlitv 146, 0x41c541c5, 146
v_exp_f16_e64 v131, v131
v_exp_f16_e64 v140, v140
v_exp_f16_e64 v143, v143
v_exp_f16_e64 v146, v146
_v_exp_f16_hvhv 131, 131
_v_exp_f16_hvhv 140, 140
_v_exp_f16_hvhv 143, 143
_v_exp_f16_hvhv 146, 146
_v_pk_add_f16_vconstv 131, 1.0, 131
_v_pk_add_f16_vconstv 140, 1.0, 140
_v_pk_add_f16_vconstv 143, 1.0, 143
_v_pk_add_f16_vconstv 146, 1.0, 146
v_rcp_f16_e64 v131, v131
v_rcp_f16_e64 v140, v140
v_rcp_f16_e64 v143, v143
v_rcp_f16_e64 v146, v146
_v_rcp_f16_hvhv 131, 131
_v_rcp_f16_hvhv 140, 140
_v_rcp_f16_hvhv 143, 143
_v_rcp_f16_hvhv 146, 146
v_mov_b32_e32 v137, 0x3bc23bc2
v_mov_b32_e32 v142, 0x3bc23bc2
v_mov_b32_e32 v145, 0x3bc23bc2
v_mov_b32_e32 v148, 0x3bc23bc2
v_pk_fma_f16 v131, v131, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v140, v140, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v143, v143, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v146, v146, -2.0, 1.0 op_sel_hi:[1,0,0]
v_cmp_gt_f16_e64 vcc_lo, |v103|, v137
v_cndmask_b16 v133, v133, v131, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v104|, v142
v_cndmask_b16 v141, v141, v140, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v107|, v145
v_cndmask_b16 v144, v144, v143, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v108|, v148
v_cndmask_b16 v147, v147, v146, vcc_lo
_v_cmp_gt_f16_vccahvhv 103, 137
_v_cndmask_b16_hvhvhvvcc 133, 133, 131
_v_cmp_gt_f16_vccahvhv 104, 142
_v_cndmask_b16_hvhvhvvcc 141, 141, 140
_v_cmp_gt_f16_vccahvhv 107, 145
_v_cndmask_b16_hvhvhvvcc 144, 144, 143
_v_cmp_gt_f16_vccahvhv 108, 148
_v_cndmask_b16_hvhvhvvcc 147, 147, 146
v_bfi_b32 v103, 0x7fff7fff, v133, v103
v_bfi_b32 v104, 0x7fff7fff, v141, v104
v_bfi_b32 v107, 0x7fff7fff, v144, v107
v_bfi_b32 v108, 0x7fff7fff, v147, v108
_v_pk_mul_f16_vvs 103, 103, 30
_v_pk_mul_f16_vvs 104, 104, 30
_v_pk_mul_f16_vvs 107, 107, 30
_v_pk_mul_f16_vvs 108, 108, 30
_v_pk_mul_f16_vvs 101, 101, 31
_v_pk_mul_f16_vvs 102, 102, 31
_v_pk_mul_f16_vvs 105, 105, 31
_v_pk_mul_f16_vvs 106, 106, 31
v_and_b32_e32 v131, 0x7fff7fff, v101
v_and_b32_e32 v140, 0x7fff7fff, v102
v_and_b32_e32 v143, 0x7fff7fff, v105
v_and_b32_e32 v146, 0x7fff7fff, v106
v_mov_b32_e32 v133, 0xb62eb62e
v_mov_b32_e32 v141, 0xb62eb62e
v_mov_b32_e32 v144, 0xb62eb62e
v_mov_b32_e32 v147, 0xb62eb62e
v_pk_fma_f16 v133, v131, 0x30883088, v133
v_pk_fma_f16 v141, v140, 0x30883088, v141
v_pk_fma_f16 v144, v143, 0x30883088, v144
v_pk_fma_f16 v147, v146, 0x30883088, v147
v_pk_fma_f16 v133, v131, v133, 0x1e331e33
v_pk_fma_f16 v141, v140, v141, 0x1e331e33
v_pk_fma_f16 v144, v143, v144, 0x1e331e33
v_pk_fma_f16 v147, v146, v147, 0x1e331e33
_v_pk_mul_f16_vvv 133, 131, 133
_v_pk_mul_f16_vvv 141, 140, 141
_v_pk_mul_f16_vvv 144, 143, 144
_v_pk_mul_f16_vvv 147, 146, 147
v_pk_fma_f16 v133, v131, v133, v131
v_pk_fma_f16 v141, v140, v141, v140
v_pk_fma_f16 v144, v143, v144, v143
v_pk_fma_f16 v147, v146, v147, v146
_v_pk_mul_f16_vlitv 131, 0x41c541c5, 131
_v_pk_mul_f16_vlitv 140, 0x41c541c5, 140
_v_pk_mul_f16_vlitv 143, 0x41c541c5, 143
_v_pk_mul_f16_vlitv 146, 0x41c541c5, 146
v_exp_f16_e64 v131, v131
v_exp_f16_e64 v140, v140
v_exp_f16_e64 v143, v143
v_exp_f16_e64 v146, v146
_v_exp_f16_hvhv 131, 131
_v_exp_f16_hvhv 140, 140
_v_exp_f16_hvhv 143, 143
_v_exp_f16_hvhv 146, 146
_v_pk_add_f16_vconstv 131, 1.0, 131
_v_pk_add_f16_vconstv 140, 1.0, 140
_v_pk_add_f16_vconstv 143, 1.0, 143
_v_pk_add_f16_vconstv 146, 1.0, 146
v_rcp_f16_e64 v131, v131
v_rcp_f16_e64 v140, v140
v_rcp_f16_e64 v143, v143
v_rcp_f16_e64 v146, v146
_v_rcp_f16_hvhv 131, 131
_v_rcp_f16_hvhv 140, 140
_v_rcp_f16_hvhv 143, 143
_v_rcp_f16_hvhv 146, 146
v_mov_b32_e32 v137, 0x3bc23bc2
v_mov_b32_e32 v142, 0x3bc23bc2
v_mov_b32_e32 v145, 0x3bc23bc2
v_mov_b32_e32 v148, 0x3bc23bc2
v_pk_fma_f16 v131, v131, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v140, v140, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v143, v143, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v146, v146, -2.0, 1.0 op_sel_hi:[1,0,0]
v_cmp_gt_f16_e64 vcc_lo, |v101|, v137
v_cndmask_b16 v133, v133, v131, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v102|, v142
v_cndmask_b16 v141, v141, v140, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v105|, v145
v_cndmask_b16 v144, v144, v143, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v106|, v148
v_cndmask_b16 v147, v147, v146, vcc_lo
_v_cmp_gt_f16_vccahvhv 101, 137
_v_cndmask_b16_hvhvhvvcc 133, 133, 131
_v_cmp_gt_f16_vccahvhv 102, 142
_v_cndmask_b16_hvhvhvvcc 141, 141, 140
_v_cmp_gt_f16_vccahvhv 105, 145
_v_cndmask_b16_hvhvhvvcc 144, 144, 143
_v_cmp_gt_f16_vccahvhv 106, 148
_v_cndmask_b16_hvhvhvvcc 147, 147, 146
v_bfi_b32 v101, 0x7fff7fff, v133, v101
v_bfi_b32 v102, 0x7fff7fff, v141, v102
v_bfi_b32 v105, 0x7fff7fff, v144, v105
v_bfi_b32 v106, 0x7fff7fff, v147, v106
_v_pk_mul_f16_vvs 101, 101, 30
_v_pk_mul_f16_vvs 102, 102, 30
_v_pk_mul_f16_vvs 105, 105, 30
_v_pk_mul_f16_vvs 106, 106, 30
s_setprio 0
buffer_store_b16 v103, v134, s[60:63], null idxen
buffer_store_b16 v107, v138, s[60:63], null idxen
buffer_store_b16 v101, v135, s[60:63], null idxen
buffer_store_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v103, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v107, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v101, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v105, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v104, v134, s[60:63], null idxen
buffer_store_b16 v108, v138, s[60:63], null idxen
buffer_store_b16 v102, v135, s[60:63], null idxen
buffer_store_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v104, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v108, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v102, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v106, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x89f0
s_sub_co_u32 s67, s70, 17
s_cselect_b32 s62, 0, s62
s_add_nc_u64 s[60:61], s[18:19], s[68:69]
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 111, 111, 47
_v_pk_add_f16_vvs 112, 112, 51
_v_pk_add_f16_vvs 115, 115, 47
_v_pk_add_f16_vvs 116, 116, 51
_v_pk_add_f16_vvs 109, 109, 47
_v_pk_add_f16_vvs 110, 110, 51
_v_pk_add_f16_vvs 113, 113, 47
_v_pk_add_f16_vvs 114, 114, 51
_v_pk_mul_f16_vvs 111, 111, 31
_v_pk_mul_f16_vvs 112, 112, 31
_v_pk_mul_f16_vvs 115, 115, 31
_v_pk_mul_f16_vvs 116, 116, 31
v_and_b32_e32 v131, 0x7fff7fff, v111
v_and_b32_e32 v140, 0x7fff7fff, v112
v_and_b32_e32 v143, 0x7fff7fff, v115
v_and_b32_e32 v146, 0x7fff7fff, v116
v_mov_b32_e32 v133, 0xb62eb62e
v_mov_b32_e32 v141, 0xb62eb62e
v_mov_b32_e32 v144, 0xb62eb62e
v_mov_b32_e32 v147, 0xb62eb62e
v_pk_fma_f16 v133, v131, 0x30883088, v133
v_pk_fma_f16 v141, v140, 0x30883088, v141
v_pk_fma_f16 v144, v143, 0x30883088, v144
v_pk_fma_f16 v147, v146, 0x30883088, v147
v_pk_fma_f16 v133, v131, v133, 0x1e331e33
v_pk_fma_f16 v141, v140, v141, 0x1e331e33
v_pk_fma_f16 v144, v143, v144, 0x1e331e33
v_pk_fma_f16 v147, v146, v147, 0x1e331e33
_v_pk_mul_f16_vvv 133, 131, 133
_v_pk_mul_f16_vvv 141, 140, 141
_v_pk_mul_f16_vvv 144, 143, 144
_v_pk_mul_f16_vvv 147, 146, 147
v_pk_fma_f16 v133, v131, v133, v131
v_pk_fma_f16 v141, v140, v141, v140
v_pk_fma_f16 v144, v143, v144, v143
v_pk_fma_f16 v147, v146, v147, v146
_v_pk_mul_f16_vlitv 131, 0x41c541c5, 131
_v_pk_mul_f16_vlitv 140, 0x41c541c5, 140
_v_pk_mul_f16_vlitv 143, 0x41c541c5, 143
_v_pk_mul_f16_vlitv 146, 0x41c541c5, 146
v_exp_f16_e64 v131, v131
v_exp_f16_e64 v140, v140
v_exp_f16_e64 v143, v143
v_exp_f16_e64 v146, v146
_v_exp_f16_hvhv 131, 131
_v_exp_f16_hvhv 140, 140
_v_exp_f16_hvhv 143, 143
_v_exp_f16_hvhv 146, 146
_v_pk_add_f16_vconstv 131, 1.0, 131
_v_pk_add_f16_vconstv 140, 1.0, 140
_v_pk_add_f16_vconstv 143, 1.0, 143
_v_pk_add_f16_vconstv 146, 1.0, 146
v_rcp_f16_e64 v131, v131
v_rcp_f16_e64 v140, v140
v_rcp_f16_e64 v143, v143
v_rcp_f16_e64 v146, v146
_v_rcp_f16_hvhv 131, 131
_v_rcp_f16_hvhv 140, 140
_v_rcp_f16_hvhv 143, 143
_v_rcp_f16_hvhv 146, 146
v_mov_b32_e32 v137, 0x3bc23bc2
v_mov_b32_e32 v142, 0x3bc23bc2
v_mov_b32_e32 v145, 0x3bc23bc2
v_mov_b32_e32 v148, 0x3bc23bc2
v_pk_fma_f16 v131, v131, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v140, v140, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v143, v143, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v146, v146, -2.0, 1.0 op_sel_hi:[1,0,0]
v_cmp_gt_f16_e64 vcc_lo, |v111|, v137
v_cndmask_b16 v133, v133, v131, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v112|, v142
v_cndmask_b16 v141, v141, v140, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v115|, v145
v_cndmask_b16 v144, v144, v143, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v116|, v148
v_cndmask_b16 v147, v147, v146, vcc_lo
_v_cmp_gt_f16_vccahvhv 111, 137
_v_cndmask_b16_hvhvhvvcc 133, 133, 131
_v_cmp_gt_f16_vccahvhv 112, 142
_v_cndmask_b16_hvhvhvvcc 141, 141, 140
_v_cmp_gt_f16_vccahvhv 115, 145
_v_cndmask_b16_hvhvhvvcc 144, 144, 143
_v_cmp_gt_f16_vccahvhv 116, 148
_v_cndmask_b16_hvhvhvvcc 147, 147, 146
v_bfi_b32 v111, 0x7fff7fff, v133, v111
v_bfi_b32 v112, 0x7fff7fff, v141, v112
v_bfi_b32 v115, 0x7fff7fff, v144, v115
v_bfi_b32 v116, 0x7fff7fff, v147, v116
_v_pk_mul_f16_vvs 111, 111, 30
_v_pk_mul_f16_vvs 112, 112, 30
_v_pk_mul_f16_vvs 115, 115, 30
_v_pk_mul_f16_vvs 116, 116, 30
_v_pk_mul_f16_vvs 109, 109, 31
_v_pk_mul_f16_vvs 110, 110, 31
_v_pk_mul_f16_vvs 113, 113, 31
_v_pk_mul_f16_vvs 114, 114, 31
v_and_b32_e32 v131, 0x7fff7fff, v109
v_and_b32_e32 v140, 0x7fff7fff, v110
v_and_b32_e32 v143, 0x7fff7fff, v113
v_and_b32_e32 v146, 0x7fff7fff, v114
v_mov_b32_e32 v133, 0xb62eb62e
v_mov_b32_e32 v141, 0xb62eb62e
v_mov_b32_e32 v144, 0xb62eb62e
v_mov_b32_e32 v147, 0xb62eb62e
v_pk_fma_f16 v133, v131, 0x30883088, v133
v_pk_fma_f16 v141, v140, 0x30883088, v141
v_pk_fma_f16 v144, v143, 0x30883088, v144
v_pk_fma_f16 v147, v146, 0x30883088, v147
v_pk_fma_f16 v133, v131, v133, 0x1e331e33
v_pk_fma_f16 v141, v140, v141, 0x1e331e33
v_pk_fma_f16 v144, v143, v144, 0x1e331e33
v_pk_fma_f16 v147, v146, v147, 0x1e331e33
_v_pk_mul_f16_vvv 133, 131, 133
_v_pk_mul_f16_vvv 141, 140, 141
_v_pk_mul_f16_vvv 144, 143, 144
_v_pk_mul_f16_vvv 147, 146, 147
v_pk_fma_f16 v133, v131, v133, v131
v_pk_fma_f16 v141, v140, v141, v140
v_pk_fma_f16 v144, v143, v144, v143
v_pk_fma_f16 v147, v146, v147, v146
_v_pk_mul_f16_vlitv 131, 0x41c541c5, 131
_v_pk_mul_f16_vlitv 140, 0x41c541c5, 140
_v_pk_mul_f16_vlitv 143, 0x41c541c5, 143
_v_pk_mul_f16_vlitv 146, 0x41c541c5, 146
v_exp_f16_e64 v131, v131
v_exp_f16_e64 v140, v140
v_exp_f16_e64 v143, v143
v_exp_f16_e64 v146, v146
_v_exp_f16_hvhv 131, 131
_v_exp_f16_hvhv 140, 140
_v_exp_f16_hvhv 143, 143
_v_exp_f16_hvhv 146, 146
_v_pk_add_f16_vconstv 131, 1.0, 131
_v_pk_add_f16_vconstv 140, 1.0, 140
_v_pk_add_f16_vconstv 143, 1.0, 143
_v_pk_add_f16_vconstv 146, 1.0, 146
v_rcp_f16_e64 v131, v131
v_rcp_f16_e64 v140, v140
v_rcp_f16_e64 v143, v143
v_rcp_f16_e64 v146, v146
_v_rcp_f16_hvhv 131, 131
_v_rcp_f16_hvhv 140, 140
_v_rcp_f16_hvhv 143, 143
_v_rcp_f16_hvhv 146, 146
v_mov_b32_e32 v137, 0x3bc23bc2
v_mov_b32_e32 v142, 0x3bc23bc2
v_mov_b32_e32 v145, 0x3bc23bc2
v_mov_b32_e32 v148, 0x3bc23bc2
v_pk_fma_f16 v131, v131, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v140, v140, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v143, v143, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v146, v146, -2.0, 1.0 op_sel_hi:[1,0,0]
v_cmp_gt_f16_e64 vcc_lo, |v109|, v137
v_cndmask_b16 v133, v133, v131, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v110|, v142
v_cndmask_b16 v141, v141, v140, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v113|, v145
v_cndmask_b16 v144, v144, v143, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v114|, v148
v_cndmask_b16 v147, v147, v146, vcc_lo
_v_cmp_gt_f16_vccahvhv 109, 137
_v_cndmask_b16_hvhvhvvcc 133, 133, 131
_v_cmp_gt_f16_vccahvhv 110, 142
_v_cndmask_b16_hvhvhvvcc 141, 141, 140
_v_cmp_gt_f16_vccahvhv 113, 145
_v_cndmask_b16_hvhvhvvcc 144, 144, 143
_v_cmp_gt_f16_vccahvhv 114, 148
_v_cndmask_b16_hvhvhvvcc 147, 147, 146
v_bfi_b32 v109, 0x7fff7fff, v133, v109
v_bfi_b32 v110, 0x7fff7fff, v141, v110
v_bfi_b32 v113, 0x7fff7fff, v144, v113
v_bfi_b32 v114, 0x7fff7fff, v147, v114
_v_pk_mul_f16_vvs 109, 109, 30
_v_pk_mul_f16_vvs 110, 110, 30
_v_pk_mul_f16_vvs 113, 113, 30
_v_pk_mul_f16_vvs 114, 114, 30
s_setprio 0
buffer_store_b16 v111, v134, s[60:63], null idxen
buffer_store_b16 v115, v138, s[60:63], null idxen
buffer_store_b16 v109, v135, s[60:63], null idxen
buffer_store_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v111, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v115, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v109, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v113, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v112, v134, s[60:63], null idxen
buffer_store_b16 v116, v138, s[60:63], null idxen
buffer_store_b16 v110, v135, s[60:63], null idxen
buffer_store_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v112, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v116, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v110, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v114, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x9108
s_setpc_b64 s[74:75]
s_setprio 3
_v_pk_add_f16_vvs 119, 119, 52
_v_pk_add_f16_vvs 120, 120, 53
_v_pk_add_f16_vvs 123, 123, 52
_v_pk_add_f16_vvs 124, 124, 53
_v_pk_add_f16_vvs 117, 117, 52
_v_pk_add_f16_vvs 118, 118, 53
_v_pk_add_f16_vvs 121, 121, 52
_v_pk_add_f16_vvs 122, 122, 53
_v_pk_mul_f16_vvs 119, 119, 31
_v_pk_mul_f16_vvs 120, 120, 31
_v_pk_mul_f16_vvs 123, 123, 31
_v_pk_mul_f16_vvs 124, 124, 31
v_and_b32_e32 v131, 0x7fff7fff, v119
v_and_b32_e32 v140, 0x7fff7fff, v120
v_and_b32_e32 v143, 0x7fff7fff, v123
v_and_b32_e32 v146, 0x7fff7fff, v124
v_mov_b32_e32 v133, 0xb62eb62e
v_mov_b32_e32 v141, 0xb62eb62e
v_mov_b32_e32 v144, 0xb62eb62e
v_mov_b32_e32 v147, 0xb62eb62e
v_pk_fma_f16 v133, v131, 0x30883088, v133
v_pk_fma_f16 v141, v140, 0x30883088, v141
v_pk_fma_f16 v144, v143, 0x30883088, v144
v_pk_fma_f16 v147, v146, 0x30883088, v147
v_pk_fma_f16 v133, v131, v133, 0x1e331e33
v_pk_fma_f16 v141, v140, v141, 0x1e331e33
v_pk_fma_f16 v144, v143, v144, 0x1e331e33
v_pk_fma_f16 v147, v146, v147, 0x1e331e33
_v_pk_mul_f16_vvv 133, 131, 133
_v_pk_mul_f16_vvv 141, 140, 141
_v_pk_mul_f16_vvv 144, 143, 144
_v_pk_mul_f16_vvv 147, 146, 147
v_pk_fma_f16 v133, v131, v133, v131
v_pk_fma_f16 v141, v140, v141, v140
v_pk_fma_f16 v144, v143, v144, v143
v_pk_fma_f16 v147, v146, v147, v146
_v_pk_mul_f16_vlitv 131, 0x41c541c5, 131
_v_pk_mul_f16_vlitv 140, 0x41c541c5, 140
_v_pk_mul_f16_vlitv 143, 0x41c541c5, 143
_v_pk_mul_f16_vlitv 146, 0x41c541c5, 146
v_exp_f16_e64 v131, v131
v_exp_f16_e64 v140, v140
v_exp_f16_e64 v143, v143
v_exp_f16_e64 v146, v146
_v_exp_f16_hvhv 131, 131
_v_exp_f16_hvhv 140, 140
_v_exp_f16_hvhv 143, 143
_v_exp_f16_hvhv 146, 146
_v_pk_add_f16_vconstv 131, 1.0, 131
_v_pk_add_f16_vconstv 140, 1.0, 140
_v_pk_add_f16_vconstv 143, 1.0, 143
_v_pk_add_f16_vconstv 146, 1.0, 146
v_rcp_f16_e64 v131, v131
v_rcp_f16_e64 v140, v140
v_rcp_f16_e64 v143, v143
v_rcp_f16_e64 v146, v146
_v_rcp_f16_hvhv 131, 131
_v_rcp_f16_hvhv 140, 140
_v_rcp_f16_hvhv 143, 143
_v_rcp_f16_hvhv 146, 146
v_mov_b32_e32 v137, 0x3bc23bc2
v_mov_b32_e32 v142, 0x3bc23bc2
v_mov_b32_e32 v145, 0x3bc23bc2
v_mov_b32_e32 v148, 0x3bc23bc2
v_pk_fma_f16 v131, v131, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v140, v140, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v143, v143, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v146, v146, -2.0, 1.0 op_sel_hi:[1,0,0]
v_cmp_gt_f16_e64 vcc_lo, |v119|, v137
v_cndmask_b16 v133, v133, v131, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v120|, v142
v_cndmask_b16 v141, v141, v140, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v123|, v145
v_cndmask_b16 v144, v144, v143, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v124|, v148
v_cndmask_b16 v147, v147, v146, vcc_lo
_v_cmp_gt_f16_vccahvhv 119, 137
_v_cndmask_b16_hvhvhvvcc 133, 133, 131
_v_cmp_gt_f16_vccahvhv 120, 142
_v_cndmask_b16_hvhvhvvcc 141, 141, 140
_v_cmp_gt_f16_vccahvhv 123, 145
_v_cndmask_b16_hvhvhvvcc 144, 144, 143
_v_cmp_gt_f16_vccahvhv 124, 148
_v_cndmask_b16_hvhvhvvcc 147, 147, 146
v_bfi_b32 v119, 0x7fff7fff, v133, v119
v_bfi_b32 v120, 0x7fff7fff, v141, v120
v_bfi_b32 v123, 0x7fff7fff, v144, v123
v_bfi_b32 v124, 0x7fff7fff, v147, v124
_v_pk_mul_f16_vvs 119, 119, 30
_v_pk_mul_f16_vvs 120, 120, 30
_v_pk_mul_f16_vvs 123, 123, 30
_v_pk_mul_f16_vvs 124, 124, 30
_v_pk_mul_f16_vvs 117, 117, 31
_v_pk_mul_f16_vvs 118, 118, 31
_v_pk_mul_f16_vvs 121, 121, 31
_v_pk_mul_f16_vvs 122, 122, 31
v_and_b32_e32 v131, 0x7fff7fff, v117
v_and_b32_e32 v140, 0x7fff7fff, v118
v_and_b32_e32 v143, 0x7fff7fff, v121
v_and_b32_e32 v146, 0x7fff7fff, v122
v_mov_b32_e32 v133, 0xb62eb62e
v_mov_b32_e32 v141, 0xb62eb62e
v_mov_b32_e32 v144, 0xb62eb62e
v_mov_b32_e32 v147, 0xb62eb62e
v_pk_fma_f16 v133, v131, 0x30883088, v133
v_pk_fma_f16 v141, v140, 0x30883088, v141
v_pk_fma_f16 v144, v143, 0x30883088, v144
v_pk_fma_f16 v147, v146, 0x30883088, v147
v_pk_fma_f16 v133, v131, v133, 0x1e331e33
v_pk_fma_f16 v141, v140, v141, 0x1e331e33
v_pk_fma_f16 v144, v143, v144, 0x1e331e33
v_pk_fma_f16 v147, v146, v147, 0x1e331e33
_v_pk_mul_f16_vvv 133, 131, 133
_v_pk_mul_f16_vvv 141, 140, 141
_v_pk_mul_f16_vvv 144, 143, 144
_v_pk_mul_f16_vvv 147, 146, 147
v_pk_fma_f16 v133, v131, v133, v131
v_pk_fma_f16 v141, v140, v141, v140
v_pk_fma_f16 v144, v143, v144, v143
v_pk_fma_f16 v147, v146, v147, v146
_v_pk_mul_f16_vlitv 131, 0x41c541c5, 131
_v_pk_mul_f16_vlitv 140, 0x41c541c5, 140
_v_pk_mul_f16_vlitv 143, 0x41c541c5, 143
_v_pk_mul_f16_vlitv 146, 0x41c541c5, 146
v_exp_f16_e64 v131, v131
v_exp_f16_e64 v140, v140
v_exp_f16_e64 v143, v143
v_exp_f16_e64 v146, v146
_v_exp_f16_hvhv 131, 131
_v_exp_f16_hvhv 140, 140
_v_exp_f16_hvhv 143, 143
_v_exp_f16_hvhv 146, 146
_v_pk_add_f16_vconstv 131, 1.0, 131
_v_pk_add_f16_vconstv 140, 1.0, 140
_v_pk_add_f16_vconstv 143, 1.0, 143
_v_pk_add_f16_vconstv 146, 1.0, 146
v_rcp_f16_e64 v131, v131
v_rcp_f16_e64 v140, v140
v_rcp_f16_e64 v143, v143
v_rcp_f16_e64 v146, v146
_v_rcp_f16_hvhv 131, 131
_v_rcp_f16_hvhv 140, 140
_v_rcp_f16_hvhv 143, 143
_v_rcp_f16_hvhv 146, 146
v_mov_b32_e32 v137, 0x3bc23bc2
v_mov_b32_e32 v142, 0x3bc23bc2
v_mov_b32_e32 v145, 0x3bc23bc2
v_mov_b32_e32 v148, 0x3bc23bc2
v_pk_fma_f16 v131, v131, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v140, v140, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v143, v143, -2.0, 1.0 op_sel_hi:[1,0,0]
v_pk_fma_f16 v146, v146, -2.0, 1.0 op_sel_hi:[1,0,0]
v_cmp_gt_f16_e64 vcc_lo, |v117|, v137
v_cndmask_b16 v133, v133, v131, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v118|, v142
v_cndmask_b16 v141, v141, v140, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v121|, v145
v_cndmask_b16 v144, v144, v143, vcc_lo
v_cmp_gt_f16_e64 vcc_lo, |v122|, v148
v_cndmask_b16 v147, v147, v146, vcc_lo
_v_cmp_gt_f16_vccahvhv 117, 137
_v_cndmask_b16_hvhvhvvcc 133, 133, 131
_v_cmp_gt_f16_vccahvhv 118, 142
_v_cndmask_b16_hvhvhvvcc 141, 141, 140
_v_cmp_gt_f16_vccahvhv 121, 145
_v_cndmask_b16_hvhvhvvcc 144, 144, 143
_v_cmp_gt_f16_vccahvhv 122, 148
_v_cndmask_b16_hvhvhvvcc 147, 147, 146
v_bfi_b32 v117, 0x7fff7fff, v133, v117
v_bfi_b32 v118, 0x7fff7fff, v141, v118
v_bfi_b32 v121, 0x7fff7fff, v144, v121
v_bfi_b32 v122, 0x7fff7fff, v147, v122
_v_pk_mul_f16_vvs 117, 117, 30
_v_pk_mul_f16_vvs 118, 118, 30
_v_pk_mul_f16_vvs 121, 121, 30
_v_pk_mul_f16_vvs 122, 122, 30
s_setprio 0
buffer_store_b16 v119, v134, s[60:63], null idxen
buffer_store_b16 v123, v138, s[60:63], null idxen
buffer_store_b16 v117, v135, s[60:63], null idxen
buffer_store_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v119, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v123, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v117, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v121, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_b16 v120, v134, s[60:63], null idxen
buffer_store_b16 v124, v138, s[60:63], null idxen
buffer_store_b16 v118, v135, s[60:63], null idxen
buffer_store_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
buffer_store_d16_hi_b16 v120, v134, s[60:63], null idxen
buffer_store_d16_hi_b16 v124, v138, s[60:63], null idxen
buffer_store_d16_hi_b16 v118, v135, s[60:63], null idxen
buffer_store_d16_hi_b16 v122, v139, s[60:63], null idxen
s_add_co_u32 s60, s60, s59
s_add_co_ci_u32 s61, s61, 0
s_sub_co_u32 s67, s67, 1
s_cselect_b32 s62, 0, s62
s_setprio 2
s_mov_b32 s76, 0x7bb0
s_bitcmp1_b32 s58, 0
s_cbranch_scc1 5
s_sub_co_u32 s67, s70, 1
s_cselect_b32 s62, 0, 0x80000000
s_mov_b64 s[60:61], s[18:19]
s_setpc_b64 s[74:75]
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
