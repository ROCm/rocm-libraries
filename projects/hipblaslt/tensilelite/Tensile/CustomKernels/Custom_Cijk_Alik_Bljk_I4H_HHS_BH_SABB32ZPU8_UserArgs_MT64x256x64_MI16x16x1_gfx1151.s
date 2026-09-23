// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Generated with W4A16 generator revision 108edb6c5f0.
// Q27B unsigned_bias8: same tile and scheduling parameters as the ExLlama variant.
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_UserArgs_MT64x256x64_MI16x16x1_gfx1151
.globl Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_UserArgs_MT64x256x64_MI16x16x1_gfx1151
.p2align 8
.type Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_UserArgs_MT64x256x64_MI16x16x1_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_UserArgs_MT64x256x64_MI16x16x1_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 184 // vgprs
  .amdhsa_next_free_sgpr 86 // sgprs
  .amdhsa_group_segment_fixed_size 46080 // lds bytes
  .amdhsa_wavefront_size32 1 // 32-thread wavefronts
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
/* Num VGPR   =184 */
/* Num AccVGPR=0 */
/* Num SGPR   =86 */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 3
  StaggerU: 0
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_UserArgs_MT64x256x64_MI16x16x1_gfx1151
    .symbol: 'Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_UserArgs_MT64x256x64_MI16x16x1_gfx1151.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            Gemm info
        .size:            4
        .offset:          0
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info0
        .size:            4
        .offset:          4
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info1
        .size:            4
        .offset:          8
        .value_kind:      by_value
        .value_type:      u32
      - .name:            numWG
        .size:            4
        .offset:          12
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          16
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          20
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          24
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          28
        .value_kind:      by_value
        .value_type:      u32
      - .name:            A
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      i4
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideA0
        .size:            4
        .offset:          48
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          52
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          56
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          60
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      f32
      - .name:            D
        .size:            8
        .offset:          72
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          80
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      u32
      - .name:            AddressScaleA
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleB
        .size:            8
        .offset:          112
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleZeroA
        .size:            8
        .offset:          120
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            batchOffsetD
        .size:            8
        .offset:          128
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetC
        .size:            8
        .offset:          136
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetA
        .size:            8
        .offset:          144
        .value_kind:      by_value
        .value_type:      u64
      - .name:            batchOffsetB
        .size:            8
        .offset:          152
        .value_kind:      by_value
        .value_type:      u64
    .group_segment_fixed_size:   46080
    .kernarg_segment_align:      8
    .kernarg_segment_size:       160
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 86
    .sgpr_spill_count:           0
    .vgpr_count:                 184
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_UserArgs_MT64x256x64_MI16x16x1_gfx1151:
label_ASM_Start:
.set vgprMXSBase, 0
.set vgprValuC, 0
.set vgprBase, 88
.set vgprLocalWriteAddrA, 82
.set vgprLocalWriteAddrB, 83
.set vgprGlobalReadOffsetA, 64
.set vgprGlobalReadOffsetB, 66
.set vgprGlobalReadOffsetScaleA, 74
.set vgprG2LScaleA, 76
.set vgprG2LScaleZeroA, 78
.set vgprGlobalReadOffsetScaleZeroA, 80
.set vgprLocalReadAddrA, 84
.set vgprLocalReadAddrB, 85
.set vgprSerial, 178
.set vgprValuA_X0_I0_BASE, vgprBase+0
.set vgprValuB_X0_I0_BASE, vgprBase+17
.set vgprG2LA_BASE, vgprBase+50
.set vgprG2LB_BASE, vgprBase+58
.set vgprValuA_X0_I0, vgprValuA_X0_I0_BASE+0
.set vgprValuB_X0_I0, vgprValuB_X0_I0_BASE+0
.set vgprG2LA, vgprG2LA_BASE+0
.set vgprG2LB, vgprG2LB_BASE+0
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprArgType, 5
.set sgprGSUSumIdx, 6
.set sgprGSULog2BpeC, 8
.set sgprGSULog2BpeD, 9
.set sgprStaggerU, 10
.set sgprWGM, 11
.set sgprLoopCounterL, 12
.set sgprOrigLoopCounter, 13
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprNumWorkGroups0, 14
.set sgprNumWorkGroups1, 15
.set sgprSizesFree, 24
.set sgprSizesSum, 27
.set sgprAddressA, 28
.set sgprAddressB, 30
.set sgprStridesA, 32
.set sgprStridesB, 34
.set sgprAlpha, 36
.set sgprBeta, 37
.set sgprAddressD, 38
.set sgprAddressC, 40
.set sgprStridesD, 42
.set sgprStridesC, 44
.set sgprGSU, 46
.set sgprAddressScaleA, 48
.set sgprAddressScaleB, 50
.set sgprStrideScaleA, 47
.set sgprSrdScaleA, 52
.set sgprAddressScaleZeroA, 56
.set sgprSrdScaleZeroA, 60
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1
.set MT0, 64
.set MT1, 256
.set DepthU, 64
.set SrdShiftLeftA, 8
.set SrdShiftLeftB, 8
.set BufferLimit, 0xffffffff
.set BufferOOB, 0xfffff000
.set Srd127_96, 0x31004000
s_load_b32 s20, s[sgprKernArgAddress:sgprKernArgAddress+1], 0
s_load_b32 s22, s[sgprKernArgAddress:sgprKernArgAddress+1], 4
s_load_b32 s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 8
s_load_b32 s23, s[sgprKernArgAddress:sgprKernArgAddress+1], 12
s_waitcnt lgkmcnt(0)
s_lshr_b32 s21, s20, 0x1e
s_and_b32 s20, 0x3fffffff, s20
s_cmp_eq_u32 s21, 3
s_cbranch_scc1 label_Bypass_ArgType3_to_ArgType0_Instance1
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_HBMArgs
label_Bypass_ArgType3_to_ArgType0_Instance1:
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b512 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0
s_load_b128 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 64
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_branch label_LoadArgsEnd
label_HBMArgs:
s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_waitcnt lgkmcnt(0)
label_LoadArgsEnd:
s_and_b32 s[sgprStaggerU], s22, 0xffff0000
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s22, 0xffff
s_mov_b32 s[sgprArgType], s21
s_mov_b32 m0, 0xb400
v_mov_b32 v[vgprSerial], v0
s_mov_b32 vcc_hi, 0
s_lshr_b32 s68, s[sgprWGM], 0x10
s_ff1_i32_b32 s68, s68
s_lshr_b32 s69, s[sgprWGM], 0x16
s_cmp_gt_i32 s68, 0
s_cbranch_scc0 label_skip_WGMXCC
s_lshr_b32 s65, s23, s68
s_lshl_b32 s65, s65, s68
s_cmp_ge_u32 s[sgprWorkGroup0], s65
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s69, 0
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s65, s[sgprWorkGroup0], s68
s_bfm_b32 s66, s68, 0
s_and_b32 s66, s[sgprWorkGroup0], s66
s_lshr_b32 s67, s23, s68
s_mul_i32 s66, s66, s67
s_add_u32 s[sgprWorkGroup0], s65, s66
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
v_cvt_f64_u32 v[6:7], s69
v_rcp_f64 v[6:7], v[6:7]
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]
v_mul_f64 v[6:7], v[6:7], v[8:9]
v_cvt_u32_f64 v6, v[6:7]
v_mul_lo_u32 v7, v6, s69
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7
v_cmp_ge_u32 vcc_lo, v8, s69
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_mul_lo_u32 v7, v6, s69
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7
v_readfirstlane_b32 s65, v6
v_readfirstlane_b32 s66, v8
s_mul_i32 s65, s65, s69
s_lshr_b32 s66, s66, s68
s_add_u32 s65, s65, s66
v_cvt_f64_u32 v[6:7], s69
v_rcp_f64 v[6:7], v[6:7]
v_cvt_f64_u32 v[8:9], s23
v_mul_f64 v[6:7], v[6:7], v[8:9]
v_cvt_u32_f64 v6, v[6:7]
v_mul_lo_u32 v7, v6, s69
v_sub_nc_u32 v8, s23, v7
v_cmp_ge_u32 vcc_lo, v8, s69
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s66, v6
s_mul_i32 s66, s66, s69
s_sub_u32 s67, s23, s66
s_cmp_gt_u32 s[sgprWorkGroup0], s66
s_cselect_b32 s66, s67, s69
s_lshr_b32 s66, s66, s68
s_bfm_b32 s67, s68, 0
s_and_b32 s67, s[sgprWorkGroup0], s67
s_mul_i32 s66, s66, s67
s_add_u32 s[sgprWorkGroup0], s65, s66
label_skip_WGMXCC:
s_cmp_eq_u32 s21, 3
s_cbranch_scc1 label_ArgType3_Routed_To_ArgType0
s_cmp_eq_u32 s21, 0
s_cbranch_scc0 label_MultiGemm
label_ArgType3_Routed_To_ArgType0:
v_and_b32 v1, 31, v[vgprSerial]
v_and_b32 v0, 15, v1
v_lshlrev_b32 v0, 6, v0
v_lshrrev_b32 v4, 5, v[vgprSerial]
v_and_b32 v4, 1, v4
v_lshl_add_u32 v0, v4, 10, v0
v_and_b32 v2, 31, v[vgprSerial]
v_and_b32 v1, 15, v2
v_lshlrev_b32 v1, 6, v1
v_lshrrev_b32 v3, 6, v[vgprSerial]
v_and_b32 v3, 3, v3
v_lshl_add_u32 v1, v3, 10, v1
v_lshrrev_b32 v2, 5, v[vgprSerial]
v_lshrrev_b32 v2, 3, v2
s_mov_b32 s16, 64
v_mul_lo_u32 v2, s16, v2
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA]
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA]
v_lshrrev_b32 v0, 5, v[vgprSerial]
v_lshrrev_b32 v0, 3, v0
v_mul_lo_u32 v0, s16, v0
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB]
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB]
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x2400, v[vgprLocalReadAddrB+0]
v_lshrrev_b32 v0, 3, v[vgprSerial]
v_and_b32 v1, 7, v[vgprSerial]
v_lshlrev_b32 v1, 3, v1
v_mov_b32 v4, v1
v_lshrrev_b32 v2, 3, v[vgprSerial]
v_and_b32 v3, 7, v[vgprSerial]
v_lshlrev_b32 v3, 3, v3
v_mov_b32 v5, v3
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v0
v_add_nc_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA]
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA]
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrA]
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 4, v[vgprLocalWriteAddrA]
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x40, v2
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB]
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB]
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrB]
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 4, v[vgprLocalWriteAddrB]
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x2400, v[vgprLocalWriteAddrB]
s_waitcnt lgkmcnt(0)
v_mov_b32 v8, MT0
v_mov_b32 v7, s[sgprSizesFree+0]
v_cvt_f32_u32 v6, v8
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v9, v7
v_mul_f32 v6, v6, v9
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v9, v6, v8
v_sub_nc_u32 v9, v7, v9
v_cmp_ne_u32 vcc_lo, v9, 0
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo
v_mov_b32 v8, MT1
v_mov_b32 v7, s[sgprSizesFree+1]
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6
v_cvt_f32_u32 v6, v8
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v9, v7
v_mul_f32 v6, v6, v9
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v9, v6, v8
v_sub_nc_u32 v9, v7, v9
v_cmp_ne_u32 vcc_lo, v9, 0
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0xfff
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v6, s16
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v7, s[sgprWorkGroup0]
v_mul_f32 v6, v6, v7
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v7, v6, s16
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7
v_cmp_eq_u32 vcc_lo, v7, s16
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, 1, v6
s_mov_b32 exec_lo, -1
v_cmp_gt_u32 vcc_lo, v7, s16
s_mov_b32 exec_lo, vcc_lo
v_sub_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s16, v6
s_mov_b32 s[sgprWorkGroup2], s16
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v7, s[sgprWorkGroup0]
v_mul_f32 v6, v6, v7
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, 1, v6
s_mov_b32 exec_lo, -1
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo
v_sub_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s16, v6
s_mov_b32 s[sgprWorkGroup1], s16
s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
s_branch label_MultiGemmEnd
label_MultiGemm:
s_and_b32 s16, s[sgprArgType], 0xff
s_cmp_eq_u32 s16, 2
s_cbranch_scc1 label_IsExternalValid
s_mov_b32 s15, 112
s_mul_i32 s70, s20, 4
s_mov_b64 s[64:65], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 228
s_mov_b32 s70, 0
s_mov_b64 s[64:65], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:
s_mov_b32 s14, 1
s_mov_b32 s71, 0
s_load_b128 s[24:27], s[64:65], s70
s_cmpk_eq_u32 s20, 1
s_cbranch_scc1 label_wgTable_noLoadLoop
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s68, s24, 6
s_and_b32 s66, 63, s24
s_addc_u32 s68, s68, 0
s_lshr_b32 s69, s25, 8
s_and_b32 s66, 255, s25
s_addc_u32 s69, s69, 0
s_mul_i32 s68, s68, s69
s_mul_i32 s68, s68, s26
s_and_b32 s69, s[sgprGSU], 0xfff
s_mul_i32 s68, s68, s69
s_add_u32 s71, s71, s68
s_cmp_lt_u32 s[sgprWorkGroup0], s71
s_cbranch_scc1 label_FOUND
s_add_u32 s70, s70, s15
s_load_b128 s[24:27], s[64:65], s70
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s20
s_cbranch_scc1 label_Loop_GemmCount
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s68, s24, 6
s_and_b32 s66, 63, s24
s_addc_u32 s68, s68, 0
s_lshr_b32 s69, s25, 8
s_and_b32 s66, 255, s25
s_addc_u32 s69, s69, 0
s_mul_i32 s68, s68, s69
s_mul_i32 s68, s68, s26
s_and_b32 s64, s[sgprGSU], 0xfff
s_mul_i32 s68, s68, s64
s_add_u32 s71, s71, s68
label_FOUND:
s_sub_u32 s65, s14, 1
s_sub_u32 s64, s71, s68
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s64
s_and_b32 s16, s[sgprArgType], 0xff
s_cmp_eq_u32 s16, 2
s_cbranch_scc1 label_LoadExternalStruct
s_lshl2_add_u32 s[sgprKernArgAddress], s20, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_mul_i32 s65, s65, 112
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s65
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b512 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 16
s_load_b64 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 80
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
s_mul_i32 s65, s65, 228
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s65
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0
s_load_b64 s[sgprAddressD:sgprAddressD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_b64 s[sgprAddressC:sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18
s_load_b64 s[sgprAddressA:sgprAddressA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20
s_load_b64 s[sgprAddressB:sgprAddressB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28
s_load_b64 s[sgprStridesD:sgprStridesD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30
s_load_b64 s[sgprStridesC:sgprStridesC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x38
s_load_b64 s[sgprStridesA:sgprStridesA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_b64 s[sgprStridesB:sgprStridesB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48
s_load_b32 s[sgprAlpha], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_load_b32 s37, s[sgprKernArgAddress:sgprKernArgAddress+1], 96
s_load_b64 s[sgprAddressScaleA:sgprAddressScaleA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x64
s_load_b64 s[sgprAddressScaleB:sgprAddressScaleB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x6c
s_load_b64 s[sgprAddressScaleZeroA:sgprAddressScaleZeroA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x74
label_LoadExternalStructEnd:
v_and_b32 v1, 31, v[vgprSerial]
v_and_b32 v0, 15, v1
v_lshlrev_b32 v0, 6, v0
v_lshrrev_b32 v4, 5, v[vgprSerial]
v_and_b32 v4, 1, v4
v_lshl_add_u32 v0, v4, 10, v0
v_and_b32 v2, 31, v[vgprSerial]
v_and_b32 v1, 15, v2
v_lshlrev_b32 v1, 6, v1
v_lshrrev_b32 v3, 6, v[vgprSerial]
v_and_b32 v3, 3, v3
v_lshl_add_u32 v1, v3, 10, v1
v_lshrrev_b32 v2, 5, v[vgprSerial]
v_lshrrev_b32 v2, 3, v2
s_mov_b32 s16, 64
v_mul_lo_u32 v2, s16, v2
v_add_nc_u32 v[vgprLocalReadAddrA], v2, v0
v_lshlrev_b32 v[vgprLocalReadAddrA], 1, v[vgprLocalReadAddrA]
v_lshrrev_b32 v3, 7, v[vgprLocalReadAddrA]
v_lshl_add_u32 v[vgprLocalReadAddrA], v3, 4, v[vgprLocalReadAddrA]
v_lshrrev_b32 v0, 5, v[vgprSerial]
v_lshrrev_b32 v0, 3, v0
v_mul_lo_u32 v0, s16, v0
v_add_nc_u32 v[vgprLocalReadAddrB], v0, v1
v_lshlrev_b32 v[vgprLocalReadAddrB], 1, v[vgprLocalReadAddrB]
v_lshrrev_b32 v2, 7, v[vgprLocalReadAddrB]
v_lshl_add_u32 v[vgprLocalReadAddrB], v2, 4, v[vgprLocalReadAddrB]
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc_lo, 0x2400, v[vgprLocalReadAddrB+0]
v_lshrrev_b32 v0, 3, v[vgprSerial]
v_and_b32 v1, 7, v[vgprSerial]
v_lshlrev_b32 v1, 3, v1
v_mov_b32 v4, v1
v_lshrrev_b32 v2, 3, v[vgprSerial]
v_and_b32 v3, 7, v[vgprSerial]
v_lshlrev_b32 v3, 3, v3
v_mov_b32 v5, v3
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v0
v_add_nc_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA]
v_lshlrev_b32 v[vgprLocalWriteAddrA], 1, v[vgprLocalWriteAddrA]
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrA]
v_lshl_add_u32 v[vgprLocalWriteAddrA], v6, 4, v[vgprLocalWriteAddrA]
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x40, v2
v_add_nc_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB]
v_lshlrev_b32 v[vgprLocalWriteAddrB], 1, v[vgprLocalWriteAddrB]
v_lshrrev_b32 v6, 7, v[vgprLocalWriteAddrB]
v_lshl_add_u32 v[vgprLocalWriteAddrB], v6, 4, v[vgprLocalWriteAddrB]
v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0x2400, v[vgprLocalWriteAddrB]
s_waitcnt lgkmcnt(0)
v_mov_b32 v8, MT0
v_mov_b32 v7, s[sgprSizesFree+0]
v_cvt_f32_u32 v6, v8
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v9, v7
v_mul_f32 v6, v6, v9
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v9, v6, v8
v_sub_nc_u32 v9, v7, v9
v_cmp_ne_u32 vcc_lo, v9, 0
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo
v_mov_b32 v8, MT1
v_mov_b32 v7, s[sgprSizesFree+1]
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6
v_cvt_f32_u32 v6, v8
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v9, v7
v_mul_f32 v6, v6, v9
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v9, v6, v8
v_sub_nc_u32 v9, v7, v9
v_cmp_ne_u32 vcc_lo, v9, 0
v_add_co_ci_u32 v6, vcc_lo, v6, 0, vcc_lo
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6
s_cmp_eq_u32 s[sgprSizeJ], 0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:
s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s17, s[sgprGSU], 0xfff
s_mul_i32 s16, s16, s17
v_cvt_f32_u32 v6, s16
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v7, s[sgprWorkGroup0]
v_mul_f32 v6, v6, v7
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v7, v6, s16
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7
v_cmp_eq_u32 vcc_lo, v7, s16
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, 1, v6
s_mov_b32 exec_lo, -1
v_cmp_gt_u32 vcc_lo, v7, s16
s_mov_b32 exec_lo, vcc_lo
v_sub_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s16, v6
s_mov_b32 s[sgprWorkGroup2], s16
s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s16, s16, s[sgprWorkGroup2]
s_mul_i32 s16, s16, s17
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v7, s[sgprWorkGroup0]
v_mul_f32 v6, v6, v7
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]
v_sub_nc_u32 v7, s[sgprWorkGroup0], v7
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, 1, v6
s_mov_b32 exec_lo, -1
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups0]
s_mov_b32 exec_lo, vcc_lo
v_sub_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s16, v6
s_mov_b32 s[sgprWorkGroup1], s16
s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:
label_MultiGemmEnd:
.set sgprSrdA, 64
.set sgprSrdB, 68
.set sgprShadowLimitA, 58
.set sgprShadowLimitB, 72
.set sgprStaggerUIter, 74
.set sgprWrapUA, 75
.set sgprWrapUB, 77
.set sgprGlobalReadIncsA, 79
.set sgprGlobalReadIncsB, 80
s_and_b32 s16, s[sgprArgType], 0xff
s_cmp_eq_u32 s16, 3
s_cbranch_scc1 label_Skip_Address_Prepad_For_Pointer_Array
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 4
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0
label_Skip_Address_Prepad_For_Pointer_Array:
v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0
s_cbranch_vccz label_AlphaNonZero
s_mov_b32 s[sgprSizesSum+0], 0
label_AlphaNonZero:
s_and_b32 s16, s[sgprGSU], 0xfff
s_cmp_eq_u32 s16, 1
s_cbranch_scc1 label_GSU
s_and_b32 s16, s[sgprGSU], 0x4000
s_cbranch_scc1 label_GSUWGMRR
s_and_b32 s16, s[sgprGSU], 0xfff
v_cvt_f32_u32 v6, s16
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v7, s[sgprWorkGroup1]
v_mul_f32 v6, v6, v7
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v7, v6, s16
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7
v_cmp_eq_u32 vcc_lo, v7, s16
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, 1, v6
v_mov_b32 v7, 0
s_mov_b32 exec_lo, -1
v_cmp_gt_u32 vcc_lo, v7, s16
s_mov_b32 exec_lo, vcc_lo
v_sub_nc_u32 v6, v6, 1
v_mul_u32_u24 v7, v6, s16
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s[sgprWorkGroup1], v6
v_readfirstlane_b32 s[sgprGSUSumIdx], v7
s_branch label_GSUWGMRR_End
label_GSUWGMRR:
v_cvt_f32_u32 v6, s[sgprNumWorkGroups1]
v_rcp_iflag_f32 v6, v6
v_cvt_f32_u32 v7, s[sgprWorkGroup1]
v_mul_f32 v6, v6, v7
v_cvt_u32_f32 v6, v6
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups1]
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7
v_cmp_eq_u32 vcc_lo, v7, s[sgprNumWorkGroups1]
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, 1, v6
v_mov_b32 v7, 0
s_mov_b32 exec_lo, -1
v_cmp_gt_u32 vcc_lo, v7, s[sgprNumWorkGroups1]
s_mov_b32 exec_lo, vcc_lo
v_sub_nc_u32 v6, v6, 1
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups1]
v_sub_nc_u32 v7, s[sgprWorkGroup1], v7
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s[sgprGSUSumIdx], v6
v_readfirstlane_b32 s[sgprWorkGroup1], v7
label_GSUWGMRR_End:
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 1
label_GSU_End:
s_mov_b32 s16, s[sgprWGM]
s_sext_i32_i16 s16, s16
s_cmp_gt_i32 s16, 1
s_cbranch_scc1 label_WGMPositive
s_cmp_ge_i32 s16, 0
s_cbranch_scc1 label_WGM
s_abs_i32 s16, s16
v_cvt_f64_u32 v[6:7], s16
v_rcp_f64 v[6:7], v[6:7]
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup0]
v_mul_f64 v[6:7], v[6:7], v[8:9]
v_cvt_u32_f64 v6, v[6:7]
v_mul_lo_u32 v7, v6, s16
v_sub_nc_u32 v8, s[sgprWorkGroup0], v7
v_cmp_ge_u32 vcc_lo, v8, s16
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s17, v6
s_mul_i32 s20, s17, s16
s_sub_u32 s20, s[sgprWorkGroup0], s20
s_mul_i32 s20, s20, s[sgprNumWorkGroups1]
s_add_u32 s20, s20, s[sgprWorkGroup1]
v_cvt_f64_u32 v[6:7], s16
v_rcp_f64 v[6:7], v[6:7]
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups0]
v_mul_f64 v[6:7], v[6:7], v[8:9]
v_cvt_u32_f64 v6, v[6:7]
v_mul_lo_u32 v7, v6, s16
v_sub_nc_u32 v8, s[sgprNumWorkGroups0], v7
v_cmp_ge_u32 vcc_lo, v8, s16
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s18, v6
s_mul_i32 s19, s16, s18
s_sub_u32 s19, s[sgprNumWorkGroups0], s19
s_cmp_eq_u32 s19, 0
s_cmov_b32 s19, s16
s_cmp_ge_u32 s17, s18
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[6:7], s18
v_rcp_f64 v[6:7], v[6:7]
v_cvt_f64_u32 v[8:9], s20
v_mul_f64 v[6:7], v[6:7], v[8:9]
v_cvt_u32_f64 v6, v[6:7]
v_mul_lo_u32 v7, v6, s18
v_sub_nc_u32 v8, s20, v7
v_cmp_ge_u32 vcc_lo, v8, s18
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_mul_lo_u32 v7, v6, s18
v_sub_nc_u32 v8, s20, v7
v_readfirstlane_b32 s[sgprWorkGroup1], v6
v_readfirstlane_b32 s[sgprWorkGroup0], v8
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18
s_sub_u32 s[sgprWorkGroup0], s20, s[sgprWorkGroup0]
s_mul_i32 s17, s17, s16
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17
s_branch label_WGM
label_WGMPositive:
s_mov_b32 s16, s16
v_cvt_f64_u32 v[6:7], s16
v_rcp_f64 v[6:7], v[6:7]
v_cvt_f64_u32 v[8:9], s[sgprWorkGroup1]
v_mul_f64 v[6:7], v[6:7], v[8:9]
v_cvt_u32_f64 v6, v[6:7]
v_mul_lo_u32 v7, v6, s16
v_sub_nc_u32 v8, s[sgprWorkGroup1], v7
v_cmp_ge_u32 vcc_lo, v8, s16
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s17, v6
s_mul_i32 s20, s17, s16
s_sub_u32 s20, s[sgprWorkGroup1], s20
s_mul_i32 s20, s20, s[sgprNumWorkGroups0]
s_add_u32 s20, s20, s[sgprWorkGroup0]
v_cvt_f64_u32 v[6:7], s16
v_rcp_f64 v[6:7], v[6:7]
v_cvt_f64_u32 v[8:9], s[sgprNumWorkGroups1]
v_mul_f64 v[6:7], v[6:7], v[8:9]
v_cvt_u32_f64 v6, v[6:7]
v_mul_lo_u32 v7, v6, s16
v_sub_nc_u32 v8, s[sgprNumWorkGroups1], v7
v_cmp_ge_u32 vcc_lo, v8, s16
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s18, v6
s_mul_i32 s19, s16, s18
s_sub_u32 s19, s[sgprNumWorkGroups1], s19
s_cmp_eq_u32 s19, 0
s_cmov_b32 s19, s16
s_cmp_ge_u32 s17, s18
s_cselect_b32 s18, s19, s16
v_cvt_f64_u32 v[6:7], s18
v_rcp_f64 v[6:7], v[6:7]
v_cvt_f64_u32 v[8:9], s20
v_mul_f64 v[6:7], v[6:7], v[8:9]
v_cvt_u32_f64 v6, v[6:7]
v_mul_lo_u32 v7, v6, s18
v_sub_nc_u32 v8, s20, v7
v_cmp_ge_u32 vcc_lo, v8, s18
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v6, v6, 1
s_mov_b32 exec_lo, -1
v_mul_lo_u32 v7, v6, s18
v_sub_nc_u32 v8, s20, v7
v_readfirstlane_b32 s[sgprWorkGroup0], v6
v_readfirstlane_b32 s[sgprWorkGroup1], v8
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s18
s_sub_u32 s[sgprWorkGroup1], s20, s[sgprWorkGroup1]
s_mul_i32 s17, s17, s16
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s17
label_WGM:
v_mov_b32 v6, v0
v_add_co_u32 v7, vcc_lo, 32, v6
v_mov_b32 v8, v2
v_add_co_u32 v9, vcc_lo, 32, v8
v_add_co_u32 v10, vcc_lo, 32, v9
v_add_co_u32 v11, vcc_lo, 32, v10
v_add_co_u32 v12, vcc_lo, 32, v11
v_add_co_u32 v13, vcc_lo, 32, v12
v_add_co_u32 v14, vcc_lo, 32, v13
v_add_co_u32 v15, vcc_lo, 32, v14
v_mov_b32 v16, v1
v_mov_b32 v17, v3
s_mul_hi_u32 s19, s[sgprWorkGroup0], 64
s_mul_i32 s18, s[sgprWorkGroup0], 64
s_mul_hi_u32 s19, s18, s[sgprStrideA0I]
s_mul_i32 s18, s18, s[sgprStrideA0I]
s_and_b32 s16, s[sgprGSU], 0x8000
s_cbranch_scc1 label_GSUC_A
s_mul_hi_u32 s17, 64, s[sgprGSUSumIdx]
s_mul_i32 s16, 64, s[sgprGSUSumIdx]
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 6
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff
v_cvt_f32_u32 v18, s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v18, v18
v_cvt_f32_u32 v19, s[sgprLoopCounterL]
v_mul_f32 v18, v18, v19
v_cvt_u32_f32 v18, v18
v_mul_u32_u24 v19, v18, s[sgprGSUSumIdx+1]
v_sub_nc_u32 v19, s[sgprLoopCounterL], v19
v_cmp_eq_u32 vcc_lo, v19, s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v18, 1, v18
v_mov_b32 v19, 0
s_mov_b32 exec_lo, -1
v_cmp_gt_u32 vcc_lo, v19, s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo
v_sub_nc_u32 v18, v18, 1
v_mul_u32_u24 v19, v18, s[sgprGSUSumIdx+1]
v_sub_nc_u32 v19, s[sgprLoopCounterL], v19
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s[sgprLoopCounterL], v18
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v19
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx]
s_add_u32 s16, 1, s[sgprLoopCounterL]
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]
s_mul_i32 s16, s16, s[sgprGSUSumIdx]
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]
s_cselect_b32 s16, s16, s17
s_mul_hi_u32 s17, s16, 64
s_mul_i32 s16, s16, 64
label_GSUC_A_End:
s_add_u32 s18, s18, s16
s_addc_u32 s19, s19, s17
s_mov_b64 s[sgprShadowLimitA+0:sgprShadowLimitA+0+1], 1
s_sub_u32 s16, s[sgprSizeL], 1
s_mul_hi_u32 s17, constStrideAL, s16
s_mul_i32 s16, constStrideAL, s16
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17
s_sub_u32 s16, s[sgprSizeI], 1
s_mul_hi_u32 s17, s[sgprStrideA0I], s16
s_mul_i32 s16, s[sgprStrideA0I], s16
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s18
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s19
s_lshr_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 1
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 4
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit
s_and_b32 s20, s[sgprArgType], 0xff
s_cmp_eq_u32 s20, 3
s_cbranch_scc0 label_StridedBatchedGemmLoadA
s_mul_i32 s16, 8, s[sgprWorkGroup2]
s_cmp_eq_u32 s[sgprSizesSum], 0x0
s_cbranch_scc1 label_StridedBatchedGemmLoadA_End
s_add_u32 s16, s16, s[sgprAddressA+0]
s_addc_u32 s17, s[sgprAddressA+1], 0
s_load_b64 s[sgprSrdA:sgprSrdA+1], s[16:17], 0
s_waitcnt lgkmcnt(0)
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80
s_waitcnt lgkmcnt(0)
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 4
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0
s_lshr_b64 s[18:19], s[18:19], 1
s_add_u32 s[sgprSrdA+0], s18, s[sgprSrdA+0]
s_addc_u32 s[sgprSrdA+1], s19, s[sgprSrdA+1]
s_branch label_StridedBatchedGemmLoadA_End
label_StridedBatchedGemmLoadA:
s_mul_hi_u32 s17, s[sgprStrideAK], s[sgprWorkGroup2]
s_mul_i32 s16, s[sgprStrideAK], s[sgprWorkGroup2]
s_add_u32 s18, s18, s16
s_addc_u32 s19, s19, s17
s_lshr_b64 s[18:19], s[18:19], 1
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s18
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s19
label_StridedBatchedGemmLoadA_End:
s_mov_b32 s[sgprSrdA+3], Srd127_96
s_add_u32 s[sgprStrideScaleA], s[sgprSizeL], 0x1f
s_lshr_b32 s[sgprStrideScaleA], s[sgprStrideScaleA], 5
s_mul_i32 s16, s[sgprWorkGroup0], 64
s_mul_i32 s16, s16, s[sgprStrideScaleA]
s_mul_i32 s16, s16, 2
s_mul_i32 s17, s[sgprSizeI], s[sgprStrideScaleA]
s_mul_i32 s17, s17, 2
s_sub_u32 s[sgprSrdScaleA+2], s17, s16
s_add_u32 s[sgprSrdScaleA+0], s[sgprAddressScaleA+0], s16
s_addc_u32 s[sgprSrdScaleA+1], s[sgprAddressScaleA+1], 0
s_mov_b32 s[sgprSrdScaleA+3], Srd127_96
s_mul_i32 s16, s[sgprWorkGroup0], 64
s_lshr_b32 s16, s16, 1
s_mul_i32 s16, s16, s[sgprStrideScaleA]
s_add_u32 s17, s[sgprSizeI], 1
s_lshr_b32 s17, s17, 1
s_mul_i32 s17, s17, s[sgprStrideScaleA]
s_sub_u32 s[sgprSrdScaleZeroA+2], s17, s16
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprAddressScaleZeroA+0], s16
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprAddressScaleZeroA+1], 0
s_mov_b32 s[sgprSrdScaleZeroA+3], Srd127_96
s_mul_hi_u32 s19, s[sgprWorkGroup1], 256
s_mul_i32 s18, s[sgprWorkGroup1], 256
s_mul_hi_u32 s19, s18, s[sgprStrideB1J]
s_mul_i32 s18, s18, s[sgprStrideB1J]
s_and_b32 s16, s[sgprGSU], 0x8000
s_cbranch_scc1 label_GSUC_B
s_mul_hi_u32 s17, 64, s[sgprGSUSumIdx]
s_mul_i32 s16, 64, s[sgprGSUSumIdx]
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 6
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff
v_cvt_f32_u32 v18, s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v18, v18
v_cvt_f32_u32 v19, s[sgprLoopCounterL]
v_mul_f32 v18, v18, v19
v_cvt_u32_f32 v18, v18
v_mul_u32_u24 v19, v18, s[sgprGSUSumIdx+1]
v_sub_nc_u32 v19, s[sgprLoopCounterL], v19
v_cmp_eq_u32 vcc_lo, v19, s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v18, 1, v18
v_mov_b32 v19, 0
s_mov_b32 exec_lo, -1
v_cmp_gt_u32 vcc_lo, v19, s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo
v_sub_nc_u32 v18, v18, 1
v_mul_u32_u24 v19, v18, s[sgprGSUSumIdx+1]
v_sub_nc_u32 v19, s[sgprLoopCounterL], v19
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s[sgprLoopCounterL], v18
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v19
s_mul_i32 s17, s[sgprLoopCounterL], s[sgprGSUSumIdx]
s_add_u32 s16, 1, s[sgprLoopCounterL]
s_add_u32 s17, s17, s[sgprGSUSumIdx+1]
s_mul_i32 s16, s16, s[sgprGSUSumIdx]
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]
s_cselect_b32 s16, s16, s17
s_mul_hi_u32 s17, s16, 64
s_mul_i32 s16, s16, 64
label_GSUC_B_End:
s_add_u32 s18, s18, s16
s_addc_u32 s19, s19, s17
s_mov_b64 s[sgprShadowLimitB+0:sgprShadowLimitB+0+1], 1
s_sub_u32 s16, s[sgprSizeL], 1
s_mul_hi_u32 s17, constStrideBL, s16
s_mul_i32 s16, constStrideBL, s16
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17
s_sub_u32 s16, s[sgprSizeJ], 1
s_mul_hi_u32 s17, s[sgprStrideB1J], s16
s_mul_i32 s16, s[sgprStrideB1J], s16
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s18
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s19
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 1
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit
s_and_b32 s20, s[sgprArgType], 0xff
s_cmp_eq_u32 s20, 3
s_cbranch_scc0 label_StridedBatchedGemmLoadB
s_mul_i32 s16, 8, s[sgprWorkGroup2]
s_cmp_eq_u32 s[sgprSizesSum], 0x0
s_cbranch_scc1 label_StridedBatchedGemmLoadB_End
s_add_u32 s16, s16, s[sgprAddressB+0]
s_addc_u32 s17, s[sgprAddressB+1], 0
s_load_b64 s[sgprSrdB:sgprSrdB+1], s[16:17], 0
s_waitcnt lgkmcnt(0)
s_load_b64 s[16:17], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88
s_waitcnt lgkmcnt(0)
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 16
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0
s_lshl_b64 s[18:19], s[18:19], 1
s_add_u32 s[sgprSrdB+0], s18, s[sgprSrdB+0]
s_addc_u32 s[sgprSrdB+1], s19, s[sgprSrdB+1]
s_branch label_StridedBatchedGemmLoadB_End
label_StridedBatchedGemmLoadB:
s_mul_hi_u32 s17, s[sgprStrideBK], s[sgprWorkGroup2]
s_mul_i32 s16, s[sgprStrideBK], s[sgprWorkGroup2]
s_add_u32 s18, s18, s16
s_addc_u32 s19, s19, s17
s_lshl_b64 s[18:19], s[18:19], 1
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s18
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s19
label_StridedBatchedGemmLoadB_End:
s_mov_b32 s[sgprSrdB+3], Srd127_96
v_mul_lo_u32 v18, s[sgprStrideA0I], v[6]
v_add_co_u32 v[vgprGlobalReadOffsetA+0+0], vcc_lo, v[16], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetA+0+0], 0x8, v[vgprGlobalReadOffsetA+0+0]
v_lshrrev_b32 v18, 5, v16
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+0], s[sgprStrideScaleA], v6
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+0], v18, v[vgprGlobalReadOffsetScaleA+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+0], 1, v[vgprGlobalReadOffsetScaleA+0]
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 1, v6
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+0], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+0]
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+0], v18, v[vgprGlobalReadOffsetScaleZeroA+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+0], 1, v[vgprGlobalReadOffsetScaleZeroA+0]
v_and_b32 v18, 1, v6
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+0], v18, v[vgprGlobalReadOffsetScaleZeroA+0]
v_lshrrev_b32 v[vgprGlobalReadOffsetA+0], 1, v[vgprGlobalReadOffsetA+0]
v_mul_lo_u32 v18, s[sgprStrideA0I], v[7]
v_add_co_u32 v[vgprGlobalReadOffsetA+1+0], vcc_lo, v[16], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetA+1+0], 0x8, v[vgprGlobalReadOffsetA+1+0]
v_lshrrev_b32 v18, 5, v16
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleA+1], s[sgprStrideScaleA], v7
v_add_nc_u32 v[vgprGlobalReadOffsetScaleA+1], v18, v[vgprGlobalReadOffsetScaleA+1]
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleA+1], 1, v[vgprGlobalReadOffsetScaleA+1]
v_lshrrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 1, v7
v_mul_lo_u32 v[vgprGlobalReadOffsetScaleZeroA+1], s[sgprStrideScaleA], v[vgprGlobalReadOffsetScaleZeroA+1]
v_add_nc_u32 v[vgprGlobalReadOffsetScaleZeroA+1], v18, v[vgprGlobalReadOffsetScaleZeroA+1]
v_lshlrev_b32 v[vgprGlobalReadOffsetScaleZeroA+1], 1, v[vgprGlobalReadOffsetScaleZeroA+1]
v_and_b32 v18, 1, v7
v_or_b32 v[vgprGlobalReadOffsetScaleZeroA+1], v18, v[vgprGlobalReadOffsetScaleZeroA+1]
v_lshrrev_b32 v[vgprGlobalReadOffsetA+1], 1, v[vgprGlobalReadOffsetA+1]
v_mul_lo_u32 v18, s[sgprStrideB1J], v[8]
v_add_co_u32 v[vgprGlobalReadOffsetB+0+0], vcc_lo, v[17], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetB+0+0], 0x8, v[vgprGlobalReadOffsetB+0+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+0], 1, v[vgprGlobalReadOffsetB+0]
v_mul_lo_u32 v18, s[sgprStrideB1J], v[9]
v_add_co_u32 v[vgprGlobalReadOffsetB+1+0], vcc_lo, v[17], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetB+1+0], 0x8, v[vgprGlobalReadOffsetB+1+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+1], 1, v[vgprGlobalReadOffsetB+1]
v_mul_lo_u32 v18, s[sgprStrideB1J], v[10]
v_add_co_u32 v[vgprGlobalReadOffsetB+2+0], vcc_lo, v[17], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetB+2+0], 0x8, v[vgprGlobalReadOffsetB+2+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+2], 1, v[vgprGlobalReadOffsetB+2]
v_mul_lo_u32 v18, s[sgprStrideB1J], v[11]
v_add_co_u32 v[vgprGlobalReadOffsetB+3+0], vcc_lo, v[17], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetB+3+0], 0x8, v[vgprGlobalReadOffsetB+3+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+3], 1, v[vgprGlobalReadOffsetB+3]
v_mul_lo_u32 v18, s[sgprStrideB1J], v[12]
v_add_co_u32 v[vgprGlobalReadOffsetB+4+0], vcc_lo, v[17], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetB+4+0], 0x8, v[vgprGlobalReadOffsetB+4+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+4], 1, v[vgprGlobalReadOffsetB+4]
v_mul_lo_u32 v18, s[sgprStrideB1J], v[13]
v_add_co_u32 v[vgprGlobalReadOffsetB+5+0], vcc_lo, v[17], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetB+5+0], 0x8, v[vgprGlobalReadOffsetB+5+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+5], 1, v[vgprGlobalReadOffsetB+5]
v_mul_lo_u32 v18, s[sgprStrideB1J], v[14]
v_add_co_u32 v[vgprGlobalReadOffsetB+6+0], vcc_lo, v[17], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetB+6+0], 0x8, v[vgprGlobalReadOffsetB+6+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+6], 1, v[vgprGlobalReadOffsetB+6]
v_mul_lo_u32 v18, s[sgprStrideB1J], v[15]
v_add_co_u32 v[vgprGlobalReadOffsetB+7+0], vcc_lo, v[17], v[18+0]
v_add_nc_u32 v[vgprGlobalReadOffsetB+7+0], 0x8, v[vgprGlobalReadOffsetB+7+0]
v_lshlrev_b32 v[vgprGlobalReadOffsetB+7], 1, v[vgprGlobalReadOffsetB+7]
s_and_b32 s17, s[sgprGSU], 0xfff
s_mov_b32 s[sgprGlobalReadIncsA+0], 32
s_mul_i32 s17, s17, s[sgprGlobalReadIncsA+0]
s_and_b32 s16, s[sgprGSU], 0x8000
s_cselect_b32 s[sgprGlobalReadIncsA+0], s[sgprGlobalReadIncsA+0], s17
s_and_b32 s17, s[sgprGSU], 0xfff
s_mov_b32 s[sgprGlobalReadIncsB+0], 128
s_mul_i32 s17, s17, s[sgprGlobalReadIncsB+0]
s_and_b32 s16, s[sgprGSU], 0x8000
s_cselect_b32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], s17
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 6
s_and_b32 s16, s[sgprGSU], 0xfff
s_cmp_eq_u32 s16, 1
s_cbranch_scc1 label_GSU_1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0xfff
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v0, v0
v_cvt_f32_u32 v1, s[sgprLoopCounterL]
v_mul_f32 v0, v0, v1
v_cvt_u32_f32 v0, v0
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1
v_cmp_eq_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo
v_add_nc_u32 v0, 1, v0
v_mov_b32 v1, 0
s_mov_b32 exec_lo, -1
v_cmp_gt_u32 vcc_lo, v1, s[sgprGSUSumIdx+1]
s_mov_b32 exec_lo, vcc_lo
v_sub_nc_u32 v0, v0, 1
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]
v_sub_nc_u32 v1, s[sgprLoopCounterL], v1
s_mov_b32 exec_lo, -1
v_readfirstlane_b32 s[sgprLoopCounterL], v0
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1
s_add_u32 s16, 1, s[sgprLoopCounterL]
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]
s_cmov_b32 s[sgprLoopCounterL], s16
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL]
s_and_b32 s18, s[sgprStaggerU], 0x1f00
s_lshr_b32 s18, s18, 0x8
s_and_b32 s19, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s16, s[sgprStaggerU]
label_beginStaggerUIter:
s_lshl_b32 s17, s16, s18
s_cmp_ge_u32 s[sgprOrigLoopCounter], s17
s_cbranch_scc1 label_endStaggerUIter
s_lshr_b32 s16, s16, 1
s_branch label_beginStaggerUIter
label_endStaggerUIter:
s_sub_u32 s17, s16, 1
s_cmp_ge_u32 s16, 1
s_cselect_b32 s[sgprStaggerUIter], s17, 0
s_cmp_eq_u32 s19, 0x0
s_cbranch_scc0 label_StaggerUMapping
s_mov_b32 s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping:
s_cmp_eq_u32 s19, 0x2000
s_cbranch_scc0 label_StaggerUMapping_1
s_mov_b32 s16, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s19, 0x4000
s_cbranch_scc0 label_StaggerUMapping_2
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s19, 0x6000
s_cbranch_scc0 label_StaggerUMapping_3
s_mul_i32 s17, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s16, s16, s17
s_add_u32 s16, s16, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s19, 0x8000
s_cbranch_scc0 label_staggerInputEnd
s_mov_b32 s16, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s16
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s18
s_mul_hi_i32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0]
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0]
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0]
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0]
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0]
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit
s_mul_hi_i32 s17, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0]
s_mul_i32 s16, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0]
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0]
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0]
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0]
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2
s_cmp_eq_u32 s[sgprLoopCounterL], 0
s_cbranch_scc1 label_ShadowInitStart
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0
v_lshrrev_b32 v0, 1, v[vgprGlobalReadOffsetScaleZeroA+0]
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v0, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0
v_lshrrev_b32 v0, 1, v[vgprGlobalReadOffsetScaleZeroA+1]
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v0, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
s_add_u32 s18, s[sgprLoopCounterL], 1
s_cmp_eq_u32 s[sgprStaggerUIter], s18
s_cselect_b32 s16, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0]
s_cselect_b32 s17, s[sgprWrapUA+1], 0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s16
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s17
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s16
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s17
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], 0x4
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], 0x4
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], 0x2
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], 0x2
s_add_u32 s18, s[sgprLoopCounterL], 1
s_cmp_eq_u32 s[sgprStaggerUIter], s18
s_cselect_b32 s16, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0]
s_cselect_b32 s17, s[sgprWrapUB+1], 0
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s16
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s17
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s16
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s17
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit
label_ShadowInitStart:
s_and_b32 s81, s[sgprGSU], 0x3fff
s_cmp_eq_u32 s81, 1
s_cbranch_scc1 label_ArgTypeCheckD
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1]
s_branch label_GeneralBatchedGemmSrdInitiationD_End
label_ArgTypeCheckD:
s_and_b32 s81, s[sgprArgType], 0xff
s_cmp_eq_u32 s81, 3
s_cbranch_scc0 label_RegularSrdInitializationD
s_branch label_GeneralBatchedGemmSrdInitiationD
label_RegularSrdInitializationD:
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], s[sgprAddressD+0:sgprAddressD+0+1]
s_branch label_GeneralBatchedGemmSrdInitiationD_End
label_GeneralBatchedGemmSrdInitiationD:
s_mov_b64 s[sgprSrdD+0:sgprSrdD+0+1], 0
label_GeneralBatchedGemmSrdInitiationD_End:
s_mov_b32 s[sgprSrdD+2], BufferOOB
s_mov_b32 s[sgprSrdD+3], Srd127_96
s_and_b32 s81, s[sgprGSU], 0x3fff
s_cmp_eq_u32 s81, 1
s_cbranch_scc1 label_ArgTypeCheckC
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1]
s_branch label_GeneralBatchedGemmSrdInitiationC_End
label_ArgTypeCheckC:
s_and_b32 s81, s[sgprArgType], 0xff
s_cmp_eq_u32 s81, 3
s_cbranch_scc0 label_RegularSrdInitializationC
s_branch label_GeneralBatchedGemmSrdInitiationC
label_RegularSrdInitializationC:
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], s[sgprAddressC+0:sgprAddressC+0+1]
s_branch label_GeneralBatchedGemmSrdInitiationC_End
label_GeneralBatchedGemmSrdInitiationC:
s_mov_b64 s[sgprSrdC+0:sgprSrdC+0+1], 0
label_GeneralBatchedGemmSrdInitiationC_End:
s_mov_b32 s[sgprSrdC+2], BufferOOB
s_mov_b32 s[sgprSrdC+3], Srd127_96
s_mul_i32 s84, MT1, s[sgprWorkGroup1]
s_and_b32 s83, s[sgprGSU], 0xfff
s_mul_hi_u32 s83, s84, s[sgprStrideC1J]
s_mul_i32 s82, s84, s[sgprStrideC1J]
s_lshl_b64 s[82:83], s[82:83], s[sgprGSULog2BpeC]
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83
s_and_b32 s83, s[sgprGSU], 0xfff
s_mul_hi_u32 s83, s84, s[sgprStrideD1J]
s_mul_i32 s82, s84, s[sgprStrideD1J]
s_lshl_b64 s[82:83], s[82:83], s[sgprGSULog2BpeD]
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83
s_and_b32 s83, s[sgprGSU], 0xfff
s_cmp_eq_u32 s83, 1
s_cbranch_scc0 label_StridedBatchedGemmLoadC
s_and_b32 s81, s[sgprArgType], 0xff
s_cmp_eq_u32 s81, 3
s_cbranch_scc1 label_GeneralBatchedGemmLoadC
label_StridedBatchedGemmLoadC:
s_mul_hi_u32 s83, s[sgprWorkGroup2], s[sgprStrideCK]
s_mul_i32 s82, s[sgprWorkGroup2], s[sgprStrideCK]
s_lshl_b64 s[82:83], s[82:83], s[sgprGSULog2BpeC]
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83
s_branch label_GeneralBatchedGemmLoadC_End
label_GeneralBatchedGemmLoadC:
s_mul_i32 s82, 8, s[sgprWorkGroup2]
s_add_u32 s82, s82, s[sgprAddressC+0]
s_addc_u32 s83, s[sgprAddressC+1], 0
s_load_b64 s[82:83], s[82:83], 0
s_waitcnt lgkmcnt(0)
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83
s_load_b64 s[82:83], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78
s_waitcnt lgkmcnt(0)
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s82
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s83
label_GeneralBatchedGemmLoadC_End:
s_and_b32 s83, s[sgprGSU], 0xfff
s_cmp_eq_u32 s83, 1
s_cbranch_scc0 label_StridedBatchedGemmLoadD
s_and_b32 s81, s[sgprArgType], 0xff
s_cmp_eq_u32 s81, 3
s_cbranch_scc1 label_GeneralBatchedGemmLoadD
label_StridedBatchedGemmLoadD:
s_mul_hi_u32 s83, s[sgprWorkGroup2], s[sgprStrideDK]
s_mul_i32 s82, s[sgprWorkGroup2], s[sgprStrideDK]
s_lshl_b64 s[82:83], s[82:83], s[sgprGSULog2BpeD]
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83
s_branch label_GeneralBatchedGemmLoadD_End
label_GeneralBatchedGemmLoadD:
s_mul_i32 s82, 8, s[sgprWorkGroup2]
s_add_u32 s82, s82, s[sgprAddressD+0]
s_addc_u32 s83, s[sgprAddressD+1], 0
s_load_b64 s[82:83], s[82:83], 0
s_waitcnt lgkmcnt(0)
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83
s_load_b64 s[82:83], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x70
s_waitcnt lgkmcnt(0)
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83
label_GeneralBatchedGemmLoadD_End:
s_and_b32 s81, s[sgprGSU], 0xfff
s_cmp_eq_u32 s81, 1
s_cbranch_scc1 label_GSU_2
s_mul_hi_u32 s83, s[sgprSizesFree+0], s[sgprGSUSumIdx]
s_mul_i32 s82, s[sgprSizesFree+0], s[sgprGSUSumIdx]
s_sub_u32 s81, s[sgprSizesFree+1], 1
s_mul_i32 s81, s81, s[sgprGSUSumIdx]
s_mul_hi_u32 s84, s81, s[sgprStrideC1J]
s_mul_i32 s81, s81, s[sgprStrideC1J]
s_add_u32 s82, s82, s81
s_addc_u32 s83, s83, s84
s_sub_u32 s81, s[sgprSizesFree+2], 1
s_mul_i32 s81, s81, s[sgprGSUSumIdx]
s_mul_hi_u32 s84, s81, s[sgprStrideCK]
s_mul_i32 s81, s81, s[sgprStrideCK]
s_add_u32 s82, s82, s81
s_addc_u32 s83, s83, s84
s_lshl_b64 s[82:83], s[82:83], 2
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s82
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s83
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF
v_mov_b32 v[vgprValuC+0], 0
v_mov_b32 v[vgprValuC+1], 0
v_mov_b32 v[vgprValuC+2], 0
v_mov_b32 v[vgprValuC+3], 0
v_mov_b32 v[vgprValuC+4], 0
v_mov_b32 v[vgprValuC+5], 0
v_mov_b32 v[vgprValuC+6], 0
v_mov_b32 v[vgprValuC+7], 0
v_mov_b32 v[vgprValuC+8], 0
v_mov_b32 v[vgprValuC+9], 0
v_mov_b32 v[vgprValuC+10], 0
v_mov_b32 v[vgprValuC+11], 0
v_mov_b32 v[vgprValuC+12], 0
v_mov_b32 v[vgprValuC+13], 0
v_mov_b32 v[vgprValuC+14], 0
v_mov_b32 v[vgprValuC+15], 0
v_mov_b32 v[vgprValuC+16], 0
v_mov_b32 v[vgprValuC+17], 0
v_mov_b32 v[vgprValuC+18], 0
v_mov_b32 v[vgprValuC+19], 0
v_mov_b32 v[vgprValuC+20], 0
v_mov_b32 v[vgprValuC+21], 0
v_mov_b32 v[vgprValuC+22], 0
v_mov_b32 v[vgprValuC+23], 0
v_mov_b32 v[vgprValuC+24], 0
v_mov_b32 v[vgprValuC+25], 0
v_mov_b32 v[vgprValuC+26], 0
v_mov_b32 v[vgprValuC+27], 0
v_mov_b32 v[vgprValuC+28], 0
v_mov_b32 v[vgprValuC+29], 0
v_mov_b32 v[vgprValuC+30], 0
v_mov_b32 v[vgprValuC+31], 0
v_mov_b32 v[vgprValuC+32], 0
v_mov_b32 v[vgprValuC+33], 0
v_mov_b32 v[vgprValuC+34], 0
v_mov_b32 v[vgprValuC+35], 0
v_mov_b32 v[vgprValuC+36], 0
v_mov_b32 v[vgprValuC+37], 0
v_mov_b32 v[vgprValuC+38], 0
v_mov_b32 v[vgprValuC+39], 0
v_mov_b32 v[vgprValuC+40], 0
v_mov_b32 v[vgprValuC+41], 0
v_mov_b32 v[vgprValuC+42], 0
v_mov_b32 v[vgprValuC+43], 0
v_mov_b32 v[vgprValuC+44], 0
v_mov_b32 v[vgprValuC+45], 0
v_mov_b32 v[vgprValuC+46], 0
v_mov_b32 v[vgprValuC+47], 0
v_mov_b32 v[vgprValuC+48], 0
v_mov_b32 v[vgprValuC+49], 0
v_mov_b32 v[vgprValuC+50], 0
v_mov_b32 v[vgprValuC+51], 0
v_mov_b32 v[vgprValuC+52], 0
v_mov_b32 v[vgprValuC+53], 0
v_mov_b32 v[vgprValuC+54], 0
v_mov_b32 v[vgprValuC+55], 0
v_mov_b32 v[vgprValuC+56], 0
v_mov_b32 v[vgprValuC+57], 0
v_mov_b32 v[vgprValuC+58], 0
v_mov_b32 v[vgprValuC+59], 0
v_mov_b32 v[vgprValuC+60], 0
v_mov_b32 v[vgprValuC+61], 0
v_mov_b32 v[vgprValuC+62], 0
v_mov_b32 v[vgprValuC+63], 0
s_cmp_eq_u32 s[sgprLoopCounterL], 0
s_cbranch_scc0 label_NoBranch_0
s_getpc_b64 s[82:83]
s_add_i32 s84, label_PrefetchGlobalLastIterEnd, 4
s_add_u32 s82, s82, s84
s_addc_u32 s83, s83, 0
s_setpc_b64 s[82:83]
label_NoBranch_0:
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v182, v[vgprG2LScaleA+0]
v_and_b32 v183, 1, v[vgprGlobalReadOffsetScaleZeroA+0]
v_lshlrev_b32 v183, 2, v183
v_bfe_u32 v179, v[vgprG2LScaleZeroA+0], v183, 0x4
v_cvt_f32_i32 v179, v179
v_mul_f32 v183, v182, v179
v_xor_b32 v183, 0x80000000, v183
v_mov_b32 v181, v[vgprG2LA+2+0]
v_bfe_u32 v179, v181, 0x0, 0x4
v_bfe_u32 v180, v181, 0x4, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+0+0], v179, v180
v_bfe_u32 v179, v181, 0x8, 0x4
v_bfe_u32 v180, v181, 0xc, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+0+1], v179, v180
v_bfe_u32 v179, v181, 0x10, 0x4
v_bfe_u32 v180, v181, 0x14, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+0+2], v179, v180
v_bfe_u32 v179, v181, 0x18, 0x4
v_bfe_u32 v180, v181, 0x1c, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+0+3], v179, v180
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0
v_cvt_f32_f16 v182, v[vgprG2LScaleA+1]
v_and_b32 v183, 1, v[vgprGlobalReadOffsetScaleZeroA+1]
v_lshlrev_b32 v183, 2, v183
v_bfe_u32 v179, v[vgprG2LScaleZeroA+1], v183, 0x4
v_cvt_f32_i32 v179, v179
v_mul_f32 v183, v182, v179
v_xor_b32 v183, 0x80000000, v183
v_mov_b32 v181, v[vgprG2LA+6+0]
v_bfe_u32 v179, v181, 0x0, 0x4
v_bfe_u32 v180, v181, 0x4, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+4+0], v179, v180
v_bfe_u32 v179, v181, 0x8, 0x4
v_bfe_u32 v180, v181, 0xc, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+4+1], v179, v180
v_bfe_u32 v179, v181, 0x10, 0x4
v_bfe_u32 v180, v181, 0x14, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+4+2], v179, v180
v_bfe_u32 v179, v181, 0x18, 0x4
v_bfe_u32 v180, v181, 0x1c, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+4+3], v179, v180
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:4608
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:27648
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:32256
label_openLoopL:
s_cmp_le_u32 s[sgprLoopCounterL], 0x1
s_cbranch_scc1 label_LoopEndL
.align 16
label_LoopBeginL:
s_waitcnt lgkmcnt(0)
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9216
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9232
buffer_load_b32 v[vgprG2LA+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0
s_waitcnt lgkmcnt(4)
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7]
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18432
buffer_load_b32 v[vgprG2LA+6], v[vgprGlobalReadOffsetA+1], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0
s_waitcnt lgkmcnt(3)
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7]
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18448
buffer_load_d16_b16 v[vgprG2LScaleA+0], v[vgprGlobalReadOffsetScaleA+0], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0
buffer_load_d16_b16 v[vgprG2LScaleA+1], v[vgprGlobalReadOffsetScaleA+1], s[sgprSrdScaleA:sgprSrdScaleA+3], 0 offen offset:0
v_lshrrev_b32 v179, 1, v[vgprGlobalReadOffsetScaleZeroA+0]
buffer_load_d16_u8 v[vgprG2LScaleZeroA+0], v179, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0
v_lshrrev_b32 v179, 1, v[vgprGlobalReadOffsetScaleZeroA+1]
buffer_load_d16_u8 v[vgprG2LScaleZeroA+1], v179, s[sgprSrdScaleZeroA:sgprSrdScaleZeroA+3], 0 offen offset:0
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7]
buffer_load_b128 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7]
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27648
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27664
buffer_load_b128 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7]
buffer_load_b128 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+2], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7]
buffer_load_b128 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
s_waitcnt lgkmcnt(0)
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7]
buffer_load_b128 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+4], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7]
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9248
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9264
buffer_load_b128 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
s_waitcnt lgkmcnt(4)
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7]
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18464
buffer_load_b128 v[vgprG2LB+24:vgprG2LB+24+3], v[vgprGlobalReadOffsetB+6], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
s_waitcnt lgkmcnt(3)
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7]
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18480
buffer_load_b128 v[vgprG2LB+28:vgprG2LB+28+3], v[vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7]
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]
s_cselect_b32 s82, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0]
s_cselect_b32 s83, s[sgprWrapUA+1], 0
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7]
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27680
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27696
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s82
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s83
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s82
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7]
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s83
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7]
s_add_u32 s[sgprSrdScaleA+0], s[sgprSrdScaleA+0], 0x4
s_addc_u32 s[sgprSrdScaleA+1], s[sgprSrdScaleA+1], 0
s_sub_u32 s[sgprSrdScaleA+2], s[sgprSrdScaleA+2], 0x4
s_waitcnt lgkmcnt(0)
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7]
s_add_u32 s[sgprSrdScaleZeroA+0], s[sgprSrdScaleZeroA+0], 0x2
s_addc_u32 s[sgprSrdScaleZeroA+1], s[sgprSrdScaleZeroA+1], 0
s_sub_u32 s[sgprSrdScaleZeroA+2], s[sgprSrdScaleZeroA+2], 0x2
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7]
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9280
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9296
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]
s_cselect_b32 s82, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0]
s_cselect_b32 s83, s[sgprWrapUB+1], 0
s_waitcnt lgkmcnt(4)
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7]
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18496
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s82
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s83
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s82
s_waitcnt lgkmcnt(3)
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7]
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18512
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s83
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7]
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27712
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27728
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7]
s_waitcnt lgkmcnt(0)
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7]
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9312
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9328
s_waitcnt lgkmcnt(4)
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7]
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18528
s_waitcnt lgkmcnt(3)
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7]
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18544
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7]
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27744
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27760
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7]
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7]
s_waitcnt vmcnt(9)
v_cvt_f32_f16 v182, v[vgprG2LScaleA+0]
v_and_b32 v183, 1, v[vgprGlobalReadOffsetScaleZeroA+0]
v_lshlrev_b32 v183, 2, v183
v_bfe_u32 v179, v[vgprG2LScaleZeroA+0], v183, 0x4
v_cvt_f32_i32 v179, v179
v_mul_f32 v183, v182, v179
v_xor_b32 v183, 0x80000000, v183
v_mov_b32 v181, v[vgprG2LA+2+0]
v_bfe_u32 v179, v181, 0x0, 0x4
v_bfe_u32 v180, v181, 0x4, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+0+0], v179, v180
v_bfe_u32 v179, v181, 0x8, 0x4
v_bfe_u32 v180, v181, 0xc, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+0+1], v179, v180
v_bfe_u32 v179, v181, 0x10, 0x4
v_bfe_u32 v180, v181, 0x14, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+0+2], v179, v180
v_bfe_u32 v179, v181, 0x18, 0x4
v_bfe_u32 v180, v181, 0x1c, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+0+3], v179, v180
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+0+3] offset:0
s_waitcnt vmcnt(8)
v_cvt_f32_f16 v182, v[vgprG2LScaleA+1]
v_and_b32 v183, 1, v[vgprGlobalReadOffsetScaleZeroA+1]
v_lshlrev_b32 v183, 2, v183
v_bfe_u32 v179, v[vgprG2LScaleZeroA+1], v183, 0x4
v_cvt_f32_i32 v179, v179
v_mul_f32 v183, v182, v179
v_xor_b32 v183, 0x80000000, v183
v_mov_b32 v181, v[vgprG2LA+6+0]
v_bfe_u32 v179, v181, 0x0, 0x4
v_bfe_u32 v180, v181, 0x4, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+4+0], v179, v180
v_bfe_u32 v179, v181, 0x8, 0x4
v_bfe_u32 v180, v181, 0xc, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+4+1], v179, v180
v_bfe_u32 v179, v181, 0x10, 0x4
v_bfe_u32 v180, v181, 0x14, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+4+2], v179, v180
v_bfe_u32 v179, v181, 0x18, 0x4
v_bfe_u32 v180, v181, 0x1c, 0x4
v_cvt_f32_i32 v179, v179
v_cvt_f32_i32 v180, v180
v_fma_f32 v179, v179, v182, v183
v_fma_f32 v180, v180, v182, v183
v_cvt_f16_f32 v179, v179
v_cvt_f16_f32 v180, v180
v_pack_b32_f16 v[vgprG2LA+4+3], v179, v180
ds_store_b128 v[vgprLocalWriteAddrA+0], v[vgprG2LA+4:vgprG2LA+4+3] offset:4608
s_waitcnt vmcnt(7)
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+0:vgprG2LB+0+3] offset:0
s_waitcnt vmcnt(6)
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608
s_waitcnt vmcnt(5)
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216
s_waitcnt vmcnt(4)
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824
s_waitcnt vmcnt(3)
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432
s_waitcnt vmcnt(2)
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040
s_waitcnt vmcnt(1)
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+24:vgprG2LB+24+3] offset:27648
s_waitcnt vmcnt(0)
ds_store_b128 v[vgprLocalWriteAddrB+0], v[vgprG2LB+28:vgprG2LB+28+3] offset:32256
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7]
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1
s_cbranch_scc0 label_LoopBeginL
label_LoopEndL:
s_and_b32 s8, s[sgprGSU], 0xfff
s_cmp_eq_u32 s8, 1
s_cbranch_scc0 label_GSU_3
label_GSU_3:
s_waitcnt lgkmcnt(0)
s_waitcnt lgkmcnt(0)
s_barrier
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:16
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:0
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:16
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4608
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4624
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9216
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9232
s_waitcnt lgkmcnt(4)
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7]
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18432
s_waitcnt lgkmcnt(3)
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7]
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18448
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7]
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27648
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27664
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7]
s_waitcnt lgkmcnt(0)
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7]
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:32
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:48
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:32
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:48
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4640
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4656
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9248
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9264
s_waitcnt lgkmcnt(4)
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7]
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18464
s_waitcnt lgkmcnt(3)
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7]
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18480
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7]
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27680
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27696
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7]
s_waitcnt lgkmcnt(0)
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7]
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:64
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:80
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:64
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:80
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4672
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4688
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9280
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9296
s_waitcnt lgkmcnt(4)
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7]
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18496
s_waitcnt lgkmcnt(3)
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7]
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18512
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7]
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27712
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27728
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7]
s_waitcnt lgkmcnt(0)
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7]
ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:96
ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], v[vgprLocalReadAddrA+0] offset:112
ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB+0] offset:96
ds_load_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB+0] offset:112
ds_load_b128 v[vgprValuA_X0_I0+8:vgprValuA_X0_I0+8+3], v[vgprLocalReadAddrA+0] offset:4704
ds_load_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:4720
ds_load_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB+0] offset:9312
ds_load_b128 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+12+3], v[vgprLocalReadAddrB+0] offset:9328
s_waitcnt lgkmcnt(4)
v_wmma_f32_16x16x16_f16 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+0:vgprValuC+0+7]
ds_load_b128 v[vgprValuB_X0_I0+16:vgprValuB_X0_I0+16+3], v[vgprLocalReadAddrB+0] offset:18528
s_waitcnt lgkmcnt(3)
v_wmma_f32_16x16x16_f16 v[vgprValuC+8:vgprValuC+8+7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+8:vgprValuC+8+7]
ds_load_b128 v[vgprValuB_X0_I0+20:vgprValuB_X0_I0+20+3], v[vgprLocalReadAddrB+0] offset:18544
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+16:vgprValuC+16+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+16:vgprValuC+16+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+24:vgprValuC+24+7], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+24:vgprValuC+24+7]
ds_load_b128 v[vgprValuB_X0_I0+24:vgprValuB_X0_I0+24+3], v[vgprLocalReadAddrB+0] offset:27744
ds_load_b128 v[vgprValuB_X0_I0+28:vgprValuB_X0_I0+28+3], v[vgprLocalReadAddrB+0] offset:27760
s_waitcnt lgkmcnt(2)
v_wmma_f32_16x16x16_f16 v[vgprValuC+32:vgprValuC+32+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+32:vgprValuC+32+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+40:vgprValuC+40+7], v[vgprValuB_X0_I0+16+0+0:vgprValuB_X0_I0+16+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+40:vgprValuC+40+7]
s_waitcnt lgkmcnt(0)
s_barrier
v_wmma_f32_16x16x16_f16 v[vgprValuC+48:vgprValuC+48+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+7], v[vgprValuC+48:vgprValuC+48+7]
v_wmma_f32_16x16x16_f16 v[vgprValuC+56:vgprValuC+56+7], v[vgprValuB_X0_I0+24+0+0:vgprValuB_X0_I0+24+0+0+7], v[vgprValuA_X0_I0+8+0+0:vgprValuA_X0_I0+8+0+0+7], v[vgprValuC+56:vgprValuC+56+7]
label_toPGR1end_OrdNLL:
label_PrefetchGlobalLastIterEnd:
label_Summation_End_2:
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprStrideScaleA, UNDEF
.set sgprAddressScaleA, UNDEF
.set sgprAddressScaleB, UNDEF
.set sgprSrdScaleA, UNDEF
.set sgprAddressScaleZeroA, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprSrdScaleZeroA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
v_lshrrev_b32 v92, 5, v[vgprSerial]
v_lshrrev_b32 v93, 1, v92
v_mul_lo_u32 v93, 0x10, v93
v_and_b32 v89, 31, v[vgprSerial]
v_lshrrev_b32 v89, 4, v89
v_add_lshl_u32 v89, v93, v89, 0
v_mul_lo_u32 v90, v89, s[sgprStrideC1J]
v_mul_lo_u32 v91, v89, s[sgprStrideD1J]
v_and_b32 v88, 1, v92
v_mul_lo_u32 v88, 0x10, v88
v_and_b32 v93, 15, v[vgprSerial]
v_add_lshl_u32 v88, v93, v88, 0
s_mul_i32 s8, 64, s[sgprWorkGroup0]
v_add_nc_u32 v88, s8, v88
s_mul_i32 s8, 256, s[sgprWorkGroup1]
v_add_nc_u32 v89, s8, v89
s_and_b32 s8, s[sgprGSU], 0xfff
s_cmp_eq_u32 s8, 1
s_cbranch_scc1 label_GSU_4
label_GW_B0_MB:
label_GW_B0_FD0_MB:
s_and_b32 s28, 63, s[sgprSizeI]
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29
s_cselect_b32 s28, s28, 0
s_cmpk_gt_u32 s28, 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Else
s_and_b32 s28, 255, s[sgprSizeJ]
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29
s_cselect_b32 s28, s28, 0
s_cmpk_gt_u32 s28, 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_MB_Then
label_GW_B0_FD0_VW1_MB_NonEdge:
v_add_lshl_u32 v99, v91, v88, 2
v_mov_b32 v[vgprValuC+101], v[vgprValuC+0]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+8]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+1]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+9]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+2]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+10]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+3]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+11]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+4]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+12]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+5]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+13]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+6]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+14]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+7]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+15]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+16]
s_mul_i32 s8, s[sgprStrideD1J], 200
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+24]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+17]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+25]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+18]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+26]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+19]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+27]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+20]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+28]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+21]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+29]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+22]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+30]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+23]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+31]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+32]
s_mul_i32 s8, s[sgprStrideD1J], 200
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+40]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+33]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+41]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+34]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+42]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+35]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+43]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+36]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+44]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+37]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+45]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+38]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+46]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+39]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+47]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+48]
s_mul_i32 s8, s[sgprStrideD1J], 200
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+56]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+49]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+57]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+50]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+58]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+51]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+59]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+52]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+60]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+53]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+61]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+54]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+62]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+55]
s_mul_i32 s8, s[sgprStrideD1J], 8
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v[vgprValuC+101], v[vgprValuC+63]
buffer_store_b32 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:128
s_nop 0
s_branch label_GW_End
label_GW_B0_FD0_VW1_MB_NonEdgeEnd:
label_GW_B0_FD0_VW1_MB_Else:
label_GW_B0_FD0_VW1_MB_Then:
v_mov_b32 v94, BufferOOB
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+0]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+8]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+1]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+9]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+2]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+10]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+3]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+11]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+4]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+12]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+5]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+13]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+6]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+14]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+7]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+15]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 50
s_mul_i32 s28, s[sgprStrideC1J], 50
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 50
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+16]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+24]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+17]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+25]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+18]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+26]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+19]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+27]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+20]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+28]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+21]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+29]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+22]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+30]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+23]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+31]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 50
s_mul_i32 s28, s[sgprStrideC1J], 50
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 50
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+32]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+40]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+33]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+41]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+34]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+42]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+35]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+43]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+36]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+44]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+37]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+45]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+38]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+46]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+39]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+47]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 50
s_mul_i32 s28, s[sgprStrideC1J], 50
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 50
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+48]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+56]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+49]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+57]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+50]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+58]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+51]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+59]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+52]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+60]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+53]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+61]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+54]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+62]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+55]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 2
v_cndmask_b32 v100, v94, v100, s30
v_mov_b32 v[vgprValuC+99], v[vgprValuC+63]
buffer_store_b32 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
s_branch label_GW_End
label_GW_End:
s_getpc_b64 s[28:29]
s_add_i32 s30, label_KernelEnd, 4
s_add_u32 s28, s28, s30
s_addc_u32 s29, s29, 0
s_setpc_b64 s[28:29]
label_GSU_4:
s_cmpk_eq_u32 s[sgprBeta], 0
s_cbranch_scc0 label_GW_B1_GSU1
label_GW_B0_GSU1:
label_GW_B0_FD0_GSU1:
s_and_b32 s28, 63, s[sgprSizeI]
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29
s_cselect_b32 s28, s28, 0
s_cmpk_gt_u32 s28, 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Else
s_and_b32 s28, 255, s[sgprSizeJ]
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29
s_cselect_b32 s28, s28, 0
s_cmpk_gt_u32 s28, 0
s_cbranch_scc1 label_GW_B0_FD0_VW1_GSU1_Then
label_GW_B0_FD0_VW1_GSU1_NonEdge:
v_add_lshl_u32 v99, v91, v88, 1
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+8]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+1]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+9]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+2]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+10]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+3]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+11]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+4]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+12]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+5]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+13]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+6]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+14]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+7]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+15]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+16]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 100
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+24]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+17]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+25]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+18]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+26]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+19]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+27]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+20]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+28]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+21]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+29]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+22]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+30]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+23]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+31]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+32]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 100
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+40]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+33]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+41]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+34]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+42]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+35]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+43]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+36]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+44]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+37]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+45]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+38]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+46]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+39]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+47]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+48]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 100
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+56]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+49]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+57]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+50]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+58]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+51]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+59]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+52]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+60]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+53]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+61]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+54]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+62]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+55]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+63]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_branch label_GW_End_1
label_GW_B0_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B0_FD0_VW1_GSU1_Else:
label_GW_B0_FD0_VW1_GSU1_Then:
v_mov_b32 v94, BufferOOB
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+8]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+1]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+9]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+2]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+10]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+3]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+11]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+4]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+12]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+5]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+13]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+6]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+14]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+7]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+15]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 50
s_mul_i32 s28, s[sgprStrideC1J], 50
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 50
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+16]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+24]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+17]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+25]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+18]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+26]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+19]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+27]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+20]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+28]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+21]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+29]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+22]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+30]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+23]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+31]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 50
s_mul_i32 s28, s[sgprStrideC1J], 50
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 50
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+32]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+40]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+33]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+41]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+34]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+42]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+35]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+43]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+36]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+44]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+37]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+45]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+38]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+46]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+39]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+47]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 50
s_mul_i32 s28, s[sgprStrideC1J], 50
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 50
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+48]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+56]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+49]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+57]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+50]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+58]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+51]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+59]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+52]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+60]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+53]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+61]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+54]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+62]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v88, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+55]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v100, v91, v92, 1
v_cndmask_b32 v100, v94, v100, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+63]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v100, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
s_branch label_GW_End_1
label_GW_B1_GSU1:
label_GW_B1_FD0_GSU1:
s_and_b32 s28, 63, s[sgprSizeI]
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29
s_cselect_b32 s28, s28, 0
s_cmpk_gt_u32 s28, 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Else
s_and_b32 s28, 255, s[sgprSizeJ]
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29
s_cselect_b32 s28, s28, 0
s_cmpk_gt_u32 s28, 0
s_cbranch_scc1 label_GW_B1_FD0_VW1_GSU1_Then
label_GW_B1_FD0_VW1_GSU1_NonEdge:
v_add_lshl_u32 v100, v90, v88, 1
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v99, v91, v88, 1
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+0]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+8]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+1]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+9]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+2]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+10]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+3]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+11]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+4]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+12]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+5]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+13]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+6]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+14]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+7]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+15]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 100
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+16]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 100
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+24]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+17]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+25]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+18]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+26]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+19]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+27]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+20]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+28]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+21]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+29]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+22]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+30]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+23]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+31]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 100
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+32]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 100
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+40]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+33]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+41]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+34]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+42]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+35]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+43]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+36]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+44]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+37]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+45]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+38]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+46]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+39]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+47]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 100
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+48]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 100
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+56]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+49]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+57]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+50]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+58]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+51]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+59]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+52]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+60]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+53]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+61]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+54]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+62]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_mul_i32 s8, s[sgprStrideC1J], 4
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s8
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+55]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
s_mul_i32 s8, s[sgprStrideD1J], 4
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s8
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
buffer_load_d16_b16 v102, v100, s[sgprSrdC:sgprSrdC+3], 0 offen offset:64
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+63]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+101], s[sgprBeta], v102, v[vgprValuC+101] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v101, v[vgprValuC+101]
buffer_store_b16 v101, v99, s[sgprSrdD:sgprSrdD+3], 0 offen offset:64
s_nop 0
s_branch label_GW_End_1
label_GW_B1_FD0_VW1_GSU1_NonEdgeEnd:
label_GW_B1_FD0_VW1_GSU1_Else:
label_GW_B1_FD0_VW1_GSU1_Then:
v_mov_b32 v94, BufferOOB
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+0]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+8]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+1]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+9]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+2]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+10]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+3]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+11]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+4]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+12]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+5]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+13]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+6]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+14]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+7]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+15]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 50
s_mul_i32 s28, s[sgprStrideC1J], 50
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 50
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+16]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+24]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+17]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+25]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+18]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+26]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+19]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+27]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+20]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+28]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+21]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+29]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+22]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+30]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+23]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+31]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 50
s_mul_i32 s28, s[sgprStrideC1J], 50
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 50
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+32]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+40]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+33]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+41]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+34]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+42]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+35]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+43]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+36]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+44]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+37]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+45]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+38]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+46]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+39]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+47]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 50
s_mul_i32 s28, s[sgprStrideC1J], 50
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 50
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+48]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+56]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+49]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+57]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+50]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+58]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+51]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+59]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+52]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+60]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+53]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+61]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+54]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+62]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v89, vcc_lo, v89, 2
s_mul_i32 s28, s[sgprStrideC1J], 2
v_add_nc_i32 v90, v90, s28
s_mul_i32 s28, s[sgprStrideD1J], 2
v_add_nc_i32 v91, v91, s28
v_cmp_lt_u32 s28, v88, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v88, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v88, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+55]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
v_mov_b32 v94, BufferOOB
v_add_co_u32 v92, vcc_lo, v88, 32
v_cmp_lt_u32 s28, v92, s[sgprSizeI]
v_cmp_lt_u32 s30, v89, s[sgprSizeJ]
s_and_b32 s30, s28, s30
v_add_lshl_u32 v101, v90, v92, 1
v_cndmask_b32 v101, v94, v101, s30
buffer_load_d16_b16 v100, v101, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0
v_add_lshl_u32 v101, v91, v92, 1
v_cndmask_b32 v101, v94, v101, s30
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+63]
s_waitcnt vmcnt(0)
v_fma_mix_f32 v[vgprValuC+99], s[sgprBeta], v100, v[vgprValuC+99] op_sel:[0,0,0] op_sel_hi:[0,1,0]
v_cvt_f16_f32 v99, v[vgprValuC+99]
buffer_store_b16 v99, v101, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0
s_nop 0
s_branch label_GW_End_1
label_GW_End_1:
label_KernelEnd:
s_endpgm
label_ASM_End:
