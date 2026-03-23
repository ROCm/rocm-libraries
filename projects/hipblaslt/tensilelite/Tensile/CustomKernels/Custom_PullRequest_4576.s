; ===========================================================================
; Source: https://github.com/ROCm/rocm-libraries/pull/4576
; This kernel is NOT authored by the race-emulator team. It is included here
; solely to demonstrate race-emulator's ability to check custom kernels for
; synchronization errors. It will be removed before (and if) race-emulator
; lands on develop.
; ===========================================================================

/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.text
.protected Custom_Cijk_Alik_Bljk_HHS_BH_Bias_SAV_UserArgs_MT16x64x64_gfx1151
.globl Custom_Cijk_Alik_Bljk_HHS_BH_Bias_SAV_UserArgs_MT16x64x64_gfx1151
.p2align 8
.type Custom_Cijk_Alik_Bljk_HHS_BH_Bias_SAV_UserArgs_MT16x64x64_gfx1151,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Alik_Bljk_HHS_BH_Bias_SAV_UserArgs_MT16x64x64_gfx1151
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 102 // sgprs
  .amdhsa_group_segment_fixed_size 29184 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.text
/* Num VGPR   =256 */
/* Num SGPR   =102 */
/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 1 */
/* SubGroup= 2 x 64 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=1 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
  ProblemType:
      OperationType: GEMM
      DataType: h
      DestDataType: h
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: True
      TransposeB: False
      UseBeta: True
      Batched: True
      UseBias: 1
      UseScaleAlphaVec: 1
      Activation: False
      SupportUserArgs: True
  EnableMatrixInstruction: True
  MIBlock: [16, 16, 16, 1, 1, 1]
  MIInputPerThread: 16
  MIInputPerThreadA: 16
  MIInputPerThreadB: 16
  MIInputPerThreadMetadata: 16
  MIWaveGroup: [4, 1]
  MIWaveTile: [2, 6]
  MatrixInstB: 1
  MatrixInstBM: 1
  MatrixInstBN: 1
  MatrixInstK: 16
  MatrixInstM: 16
  MatrixInstN: 16
  MatrixInstruction: [16, 16, 16, 1]
  Sparse: 0
  ThreadTile: [1, 1]
  WavefrontSize: 32
  WorkGroup: [16, 8, 1]
  1LDSBuffer: 0
  ScheduleIterAlg: 3
  DepthU: 64
  GlobalReadVectorWidthA: 8
  GlobalReadVectorWidthB: 1
  GlobalSplitU: 1
  GlobalSplitUAlgorithm: MultipleBuffer
  AssertFree0ElementMultiple: 1
  AssertFree1ElementMultiple: 1
  AssertSummationElementMultiple: 1
  NoReject: True
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Custom_Cijk_Alik_Bljk_HHS_BH_Bias_SAV_UserArgs_MT16x64x64_gfx1151
    .symbol: 'Custom_Cijk_Alik_Bljk_HHS_BH_Bias_SAV_UserArgs_MT16x64x64_gfx1151.kd'
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
      - .name:            D
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      f32
      - .name:            AddressScaleAlphaVec
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            bias
        .size:            8
        .offset:          112
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            biasType
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
      - .name:            StrideBias
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            activationAlpha
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationBeta
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationType
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   29184
    .kernarg_segment_align:      8
    .kernarg_segment_size:       144
    .max_flat_workgroup_size:    128
    .private_segment_fixed_size: 0
    .sgpr_count:                 102
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             32
...
.end_amdgpu_metadata
Custom_Cijk_Alik_Bljk_HHS_BH_Bias_SAV_UserArgs_MT16x64x64_gfx1151:
label_ASM_Start:  /// Main body of the asm kernel

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprArgType, 5
.set sgprStaggerUIter, 6
.set sgprStaggerUStride, 7
.set sgprLoopCounterL, 8
.set sgprWGM, 10
.set sgprKernelInfo1, 11
.set sgprNumWorkGroups0, 14
.set sgprNumWorkGroups1, 15
.set sgprGemmInfo, 20
.set sgprArgType2, 21
.set sgprKernelInfo0, 22
.set sgprNumWG, 23
.set sgprSizesFree, 24
.set sgprSizesFree0, 24
.set sgprSizesFree1, 25
.set sgprSizesFree2, 26
.set sgprSizesSum0, 27
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprStrideD0, 36
.set sgprStrideD1, 37
.set sgprStrideC0, 38
.set sgprStrideC1, 39
.set sgprStrideA0, 40
.set sgprStrideA1, 41
.set sgprStrideB0, 42
.set sgprStrideB1, 43
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprWGMInfo, 46
.set sgprSrdA, 48
.set sgprSrdB, 52

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* v[0:7] = accumulators (ValuC)          */
.set vgprValuC, 0
/* v8  = globalReadOffsetA                */
.set vgprGlobalReadOffsetA, 8
/* v9  = globalReadOffsetB                */
.set vgprGlobalReadOffsetB, 9
/* v10 = globalReadAddrA (flat)           */
/* v11 = globalReadAddrB (flat)           */
/* v12 = localWriteAddrA                  */
.set vgprLocalWriteAddrA, 12
/* v13 = localWriteAddrB                  */
.set vgprLocalWriteAddrB, 13
/* v[82:101] = global read A/B values     */
.set vgprSerial, 102

.set MT0, 16
.set MT1, 64
.set DepthU, 64
.set LdsNumBytesA, 0xa00


/******************************************/
/* Allocate Resources                      */
/******************************************/


/* Load GemmInfo */
	s_load_b32 s[sgprGemmInfo], s[sgprKernArgAddress:sgprKernArgAddress+1], null // load GemmInfo

/* Load packed kernel args (StaggerU/GSU) */
	s_load_b32 s[sgprKernelInfo0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4 // load KernelInfo0

/* Load WGM data */
	s_load_b32 s[sgprKernelInfo1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8 // load KernelInfo1

/* Load num of WGs */
	s_load_b32 s[sgprNumWG], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc // load NumWG
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_lshr_b32 s[sgprArgType2], s[sgprGemmInfo], 30        // Get arg type
	s_and_b32 s[sgprGemmInfo], 0x3fffffff, s[sgprGemmInfo] // Get nums of gemm
	s_cmp_eq_u32 s[sgprArgType2], 0                        // Is kernel args
	s_cbranch_scc0 label_HBMArgs                           // branch to HBMArgs
	s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 16 // Shift common args
	s_addc_u32 s1, s1, 0                                   // carry from above

/* Load sizes, strides, alpha */
	s_load_b512 s[sgprSizesFree0:sgprSizesFree0+15], s[sgprKernArgAddress:sgprKernArgAddress+1], null // load sizes
	s_load_b128 s[sgprStrideA0:sgprStrideA0+3], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40 // load strides
	s_load_b64 s[sgprAlpha:sgprAlpha+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50 // load alpha/beta
	s_branch label_LoadArgsEnd                             // jump to LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
	s_load_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10 // load kernel arg ptr
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
label_LoadArgsEnd:
	s_and_b32 s[sgprWGM], s[sgprKernelInfo0], 0xffff0000   // Get WGM
	s_lshr_b32 s[sgprWGM], s[sgprWGM], 16
	s_and_b32 s[sgprWGMInfo], s[sgprKernelInfo0], 0xffff   // Get WGMInfo
	s_mov_b32 s[sgprArgType], s[sgprArgType2]
	s_mov_b32 m0, 0x7200                                   // LDS clamp at 29184 bytes
	v_mov_b32_e32 v[vgprSerial], v[vgprValuC]              // thread serial id
	s_mov_b32 vcc_hi, 0                                    // clear vcc_hi

/******************************************/
/* StaggerU / XCC Remap Calculation        */
/******************************************/


/* remap workgroup to XCCs */
	s_lshr_b32 s[sgprSrdB], s[sgprKernelInfo1], 16         // Get WGMXCC
	s_ctz_i32_b32 s[sgprSrdB], s[sgprSrdB]                 // Get log(WGMXCC)
	s_lshr_b32 s53, s[sgprKernelInfo1], 22                 // Get CU_Count
	s_cmp_gt_i32 s[sgprSrdB], 0
	s_cbranch_scc0 label_skip_WGMXCC                       // branch to skip_WGMXCC
	s_lshr_b32 s49, s[sgprNumWG], s[sgprSrdB]
	s_lshl_b32 s49, s49, s[sgprSrdB]
	s_cmp_ge_u32 s[sgprWorkGroup0], s49
	s_cbranch_scc1 label_skip_WGMXCC                       // branch to skip_WGMXCC
	s_cmp_eq_u32 s53, 0
	s_cbranch_scc0 label_XCCG_nonzero                      // branch to XCCG_nonzero
	s_lshr_b32 s49, s[sgprWorkGroup0], s[sgprSrdB]
	s_bfm_b32 s50, s[sgprSrdB], 0                          // bit field mask
	s_and_b32 s50, s[sgprWorkGroup0], s50
	s_lshr_b32 s51, s[sgprNumWG], s[sgprSrdB]
	s_mul_i32 s50, s50, s51
	s_add_u32 s[sgprWorkGroup0], s49, s50
	s_branch label_skip_WGMXCC                             // jump to skip_WGMXCC
label_XCCG_nonzero:
	v_cvt_f64_u32_e32 v[16:17], s53                        // convert to f64
	v_rcp_f64_e32 v[16:17], v[16:17]                       // 1/x (for integer div)
	v_cvt_f64_u32_e32 v[18:19], s[sgprWorkGroup0]          // convert to f64
	v_mul_f64 v[16:17], v[16:17], v[18:19]
	v_cvt_u32_f64_e32 v16, v[16:17]                        // convert to u32
	v_mul_lo_u32 v17, v16, s53
	v_sub_nc_u32_e32 v18, s[sgprWorkGroup0], v17
	v_cmp_ge_u32_e64 vcc_lo, v18, s53
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e64 v16, v16, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_mul_lo_u32 v17, v16, s53
	v_sub_nc_u32_e32 v18, s[sgprWorkGroup0], v17
	v_readfirstlane_b32 s49, v16                           // quotient
	v_readfirstlane_b32 s50, v18                           // remainder
	s_mul_i32 s49, s49, s53
	s_lshr_b32 s50, s50, s[sgprSrdB]
	s_add_u32 s49, s49, s50
	v_cvt_f64_u32_e32 v[16:17], s53                        // convert to f64
	v_rcp_f64_e32 v[16:17], v[16:17]                       // 1/x (for integer div)
	v_cvt_f64_u32_e32 v[18:19], s[sgprNumWG]               // convert to f64
	v_mul_f64 v[16:17], v[16:17], v[18:19]
	v_cvt_u32_f64_e32 v16, v[16:17]                        // convert to u32
	v_mul_lo_u32 v17, v16, s53
	v_sub_nc_u32_e32 v18, s[sgprNumWG], v17
	v_cmp_ge_u32_e64 vcc_lo, v18, s53
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e64 v16, v16, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s50, v16                           // quotient
	s_mul_i32 s50, s50, s53
	s_sub_u32 s51, s[sgprNumWG], s50
	s_cmp_gt_u32 s[sgprWorkGroup0], s50
	s_cselect_b32 s50, s51, s53                            // select XCC group
	s_lshr_b32 s50, s50, s[sgprSrdB]
	s_bfm_b32 s51, s[sgprSrdB], 0                          // bit field mask
	s_and_b32 s51, s[sgprWorkGroup0], s51
	s_mul_i32 s50, s50, s51
	s_add_u32 s[sgprWorkGroup0], s49, s50
/* XCC remap done, check arg type for single vs multi gemm */
label_skip_WGMXCC:
	s_cmp_eq_u32 s[sgprArgType2], 0                        // Is kernel args
	s_cbranch_scc0 label_MultiGemm                         // branch to MultiGemm

/******************************************/
/* Local Read/Write Address Calculation    */
/******************************************/

	v_and_b32_e32 v1, 31, v[vgprSerial]
	v_and_b32_e32 v[vgprValuC], 15, v1
	v_lshlrev_b32_e32 v[vgprValuC], 6, v[vgprValuC]
	v_and_b32_e32 v2, 31, v[vgprSerial]
	v_and_b32_e32 v1, 15, v2
	v_lshlrev_b32_e32 v1, 6, v1
	v_lshrrev_b32_e32 v3, 5, v[vgprSerial]
	v_and_b32_e32 v3, 3, v3
	v_lshl_add_u32 v1, v3, 10, v1
	v_lshrrev_b32_e32 v2, 5, v[vgprSerial]
	v_lshrrev_b32_e32 v2, 2, v2
	s_mov_b32 s16, 64
	v_mul_lo_u32 v2, s16, v2
	v_add_lshl_u32 v[vgprLocalWriteAddrA], v2, v[vgprValuC], 1
	v_lshrrev_b32_e32 v3, 7, v[vgprLocalWriteAddrA]
	v_lshl_add_u32 v[vgprLocalWriteAddrA], v3, 5, v[vgprLocalWriteAddrA]
	v_lshrrev_b32_e32 v[vgprValuC], 5, v[vgprSerial]
	v_lshrrev_b32_e32 v[vgprValuC], 2, v[vgprValuC]
	v_mul_lo_u32 v[vgprValuC], s16, v[vgprValuC]
	v_add_lshl_u32 v[vgprLocalWriteAddrB], v[vgprValuC], v1, 1
	v_lshrrev_b32_e32 v2, 7, v[vgprLocalWriteAddrB]
	v_lshl_add_u32 v[vgprLocalWriteAddrB], v2, 5, v[vgprLocalWriteAddrB]
	v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0xa00, v[vgprLocalWriteAddrB] // += LDS offset B
	v_lshrrev_b32_e32 v[vgprValuC], 3, v[vgprSerial]
	v_and_b32_e32 v1, 7, v[vgprSerial]
	v_lshlrev_b32_e32 v1, 3, v1
	v_mov_b32_e32 v4, v1
	v_lshrrev_b32_e32 v2, 6, v[vgprSerial]
	v_and_b32_e32 v3, 63, v[vgprSerial]
	v_mov_b32_e32 v5, v3
	v_mul_u32_u24_e32 v10, 64, v[vgprValuC]
	v_add_lshl_u32 v10, v4, v10, 1
	v_lshrrev_b32_e32 v6, 7, v10
	v_lshl_add_u32 v10, v6, 5, v10
	v_mul_u32_u24_e32 v11, 64, v2
	v_add_lshl_u32 v11, v5, v11, 1
	v_lshrrev_b32_e32 v6, 7, v11
	v_lshl_add_u32 v11, v6, 5, v11
	v_add_co_u32 v11, vcc_lo, 0xa00, v11                   // += LDS offset B

/******************************************/
/* Workgroup Mapping (Single Gemm)         */
/******************************************/

	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	v_mov_b32_e32 v18, 16
	v_mov_b32_e32 v17, s[sgprSizesFree0]                   // set Free0 size
	v_cvt_f32_u32_e32 v16, v18                             // convert to f32
	v_rcp_iflag_f32_e32 v16, v16                           // 1/x (for integer div)
	v_cvt_f32_u32_e32 v19, v17                             // convert to f32
	v_mul_f32_e32 v16, v16, v19
	v_cvt_u32_f32_e32 v16, v16                             // convert to u32
	v_mul_u32_u24_e32 v19, v16, v18
	v_sub_nc_u32_e32 v19, v17, v19
	v_cmp_ne_u32_e64 vcc_lo, v19, 0
	v_add_co_ci_u32_e64 v16, vcc_lo, v16, 0, vcc_lo        // ceil div fixup
	v_mov_b32_e32 v18, 64
	v_mov_b32_e32 v17, s[sgprSizesFree1]                   // set Free1 size
	v_readfirstlane_b32 s[sgprNumWorkGroups0], v16         // numWorkGroups0
	v_cvt_f32_u32_e32 v16, v18                             // convert to f32
	v_rcp_iflag_f32_e32 v16, v16                           // 1/x (for integer div)
	v_cvt_f32_u32_e32 v19, v17                             // convert to f32
	v_mul_f32_e32 v16, v16, v19
	v_cvt_u32_f32_e32 v16, v16                             // convert to u32
	v_mul_u32_u24_e32 v19, v16, v18
	v_sub_nc_u32_e32 v19, v17, v19
	v_cmp_ne_u32_e64 vcc_lo, v19, 0
	v_add_co_ci_u32_e64 v16, vcc_lo, v16, 0, vcc_lo        // ceil div fixup
	v_readfirstlane_b32 s[sgprNumWorkGroups1], v16         // numWorkGroups1
	s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // total workgroups
	s_and_b32 s17, s[sgprWGMInfo], 0x3fff                  // get GSU
	s_mul_i32 s16, s16, s17
	v_cvt_f32_u32_e32 v6, s16                              // convert to f32
	v_rcp_iflag_f32_e32 v6, v6                             // 1/x (for integer div)
	v_cvt_f32_u32_e32 v7, s[sgprWorkGroup0]                // convert to f32
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6                               // convert to u32
	v_mul_u32_u24_e64 v7, v6, s16
	v_sub_nc_u32_e32 v7, s[sgprWorkGroup0], v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v6, 1, v6
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s16, v6                            // quotient
	s_mov_b32 s[sgprWorkGroup2], s16
	s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0] // total workgroups
	s_mul_i32 s16, s16, s[sgprWorkGroup2]                  // batch offset
	s_mul_i32 s16, s16, s17
	s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
	v_cvt_f32_u32_e32 v6, s[sgprNumWorkGroups0]            // convert to f32
	v_rcp_iflag_f32_e32 v6, v6                             // 1/x (for integer div)
	v_cvt_f32_u32_e32 v7, s[sgprWorkGroup0]                // convert to f32
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6                               // convert to u32
	v_mul_u32_u24_e64 v7, v6, s[sgprNumWorkGroups0]
	v_sub_nc_u32_e32 v7, s[sgprWorkGroup0], v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s[sgprNumWorkGroups0]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v6, 1, v6
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v7, s[sgprNumWorkGroups0]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s16, v6                            // quotient
	s_mov_b32 s[sgprWorkGroup1], s16
	s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0] // tile offset
	s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
	s_branch label_MultiGemmEnd                            // jump to MultiGemmEnd

/******************************************/
/* Multi-Gemm Dispatch                     */
/******************************************/

label_MultiGemm:
	s_cmp_eq_u32 s[sgprArgType], 2                         // ArgType == 2 ?
	s_cbranch_scc1 label_IsExternalValid                   // branch to IsExternalValid
	s_mov_b32 s[sgprNumWorkGroups1], 0x7c
	s_mul_i32 s54, s[sgprGemmInfo], 4
	s_mov_b64 s[sgprSrdA:sgprSrdA+1], s[sgprKernArgAddress:sgprKernArgAddress+1]
	s_branch label_IsExternalValidEnd                      // jump to IsExternalValidEnd
label_IsExternalValid:
	s_mov_b32 s[sgprNumWorkGroups1], 0xc4
	s_mov_b32 s54, 0
	s_mov_b64 s[sgprSrdA:sgprSrdA+1], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:
	s_mov_b32 s[sgprNumWorkGroups0], 1
	s_mov_b32 s55, 0
	s_load_b128 s[sgprSizesFree0:sgprSizesFree0+3], s[sgprSrdA:sgprSrdA+1], s54 // load sizes
	s_cmpk_eq_u32 s[sgprGemmInfo], 0x1                     // single gemm ?
	s_cbranch_scc1 label_GemmSearchDone                    // branch to GemmSearchDone
label_GemmSearchLoop:
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_lshr_b32 s[sgprSrdB], s[sgprSizesFree0], 4
	s_and_b32 s50, 15, s[sgprSizesFree0]                   // SizesFree0 mod MT0
	s_addc_u32 s[sgprSrdB], s[sgprSrdB], 0                 // carry from above
	s_lshr_b32 s53, s[sgprSizesFree1], 6
	s_and_b32 s50, 63, s[sgprSizesFree1]                   // SizesFree1 mod MT1
	s_addc_u32 s53, s53, 0                                 // carry from above
	s_mul_i32 s[sgprSrdB], s[sgprSrdB], s53
	s_mul_i32 s[sgprSrdB], s[sgprSrdB], s[sgprSizesFree2]
	s_and_b32 s53, s[sgprWGMInfo], 0x3fff                  // get GSU
	s_mul_i32 s[sgprSrdB], s[sgprSrdB], s53
	s_add_u32 s55, s55, s[sgprSrdB]
	s_cmp_lt_u32 s[sgprWorkGroup0], s55
	s_cbranch_scc1 label_GemmFound                         // branch to GemmFound
	s_add_u32 s54, s54, s[sgprNumWorkGroups1]
	s_load_b128 s[sgprSizesFree0:sgprSizesFree0+3], s[sgprSrdA:sgprSrdA+1], s54 // load sizes
	s_add_u32 s[sgprNumWorkGroups0], s[sgprNumWorkGroups0], 1
	s_cmp_lt_u32 s[sgprNumWorkGroups0], s[sgprGemmInfo]
	s_cbranch_scc1 label_GemmSearchLoop                    // branch to GemmSearchLoop
label_GemmSearchDone:
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_lshr_b32 s[sgprSrdB], s[sgprSizesFree0], 4
	s_and_b32 s50, 15, s[sgprSizesFree0]                   // SizesFree0 mod MT0
	s_addc_u32 s[sgprSrdB], s[sgprSrdB], 0                 // carry from above
	s_lshr_b32 s53, s[sgprSizesFree1], 6
	s_and_b32 s50, 63, s[sgprSizesFree1]                   // SizesFree1 mod MT1
	s_addc_u32 s53, s53, 0                                 // carry from above
	s_mul_i32 s[sgprSrdB], s[sgprSrdB], s53
	s_mul_i32 s[sgprSrdB], s[sgprSrdB], s[sgprSizesFree2]
	s_and_b32 s[sgprSrdA], s[sgprWGMInfo], 0x3fff          // get GSU
	s_mul_i32 s[sgprSrdB], s[sgprSrdB], s[sgprSrdA]
	s_add_u32 s55, s55, s[sgprSrdB]
label_GemmFound:
	s_sub_u32 s49, s[sgprNumWorkGroups0], 1
	s_sub_u32 s[sgprSrdA], s55, s[sgprSrdB]
	s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s[sgprSrdA]
	s_cmp_eq_u32 s[sgprArgType], 2                         // ArgType == 2 ?
	s_cbranch_scc1 label_LoadArgsExternal                  // branch to LoadArgsExternal
	s_lshl2_add_u32 s[sgprKernArgAddress], s[sgprGemmInfo], s[sgprKernArgAddress] // offset << 2 + base
	s_addc_u32 s1, s1, 0                                   // carry from above
	s_mul_i32 s49, s49, 0x7c
	s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49 // advance kernel arg ptr
	s_addc_u32 s1, s1, 0                                   // carry from above

/* Load address of kernel arguments */
	s_load_b512 s[sgprAddressD:sgprAddressD+15], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10 // load addresses
	s_load_b64 s[sgprAlpha:sgprAlpha+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50 // load alpha/beta
	s_branch label_MultiGemm_LocalRWAddr                   // jump to MultiGemm_LocalRWAddr
label_LoadArgsExternal:
	s_mul_i32 s49, s49, 0xc4
	s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49 // advance kernel arg ptr
	s_addc_u32 s1, s1, 0                                   // carry from above

/* Load address of kernel arguments */
	s_load_b512 s[sgprAddressD:sgprAddressD+15], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10 // load addresses
	s_load_b32 s[sgprAlpha], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50 // load alpha/beta
	s_load_b32 s[sgprBeta], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60 // load beta

/******************************************/
/* Local Read/Write Addr (Multi-Gemm path) */
/******************************************/

label_MultiGemm_LocalRWAddr:
	v_and_b32_e32 v1, 31, v[vgprSerial]
	v_and_b32_e32 v[vgprValuC], 15, v1
	v_lshlrev_b32_e32 v[vgprValuC], 6, v[vgprValuC]
	v_and_b32_e32 v2, 31, v[vgprSerial]
	v_and_b32_e32 v1, 15, v2
	v_lshlrev_b32_e32 v1, 6, v1
	v_lshrrev_b32_e32 v3, 5, v[vgprSerial]
	v_and_b32_e32 v3, 3, v3
	v_lshl_add_u32 v1, v3, 10, v1
	v_lshrrev_b32_e32 v2, 5, v[vgprSerial]
	v_lshrrev_b32_e32 v2, 2, v2
	s_mov_b32 s16, 64
	v_mul_lo_u32 v2, s16, v2
	v_add_lshl_u32 v[vgprLocalWriteAddrA], v2, v[vgprValuC], 1
	v_lshrrev_b32_e32 v3, 7, v[vgprLocalWriteAddrA]
	v_lshl_add_u32 v[vgprLocalWriteAddrA], v3, 5, v[vgprLocalWriteAddrA]
	v_lshrrev_b32_e32 v[vgprValuC], 5, v[vgprSerial]
	v_lshrrev_b32_e32 v[vgprValuC], 2, v[vgprValuC]
	v_mul_lo_u32 v[vgprValuC], s16, v[vgprValuC]
	v_add_lshl_u32 v[vgprLocalWriteAddrB], v[vgprValuC], v1, 1
	v_lshrrev_b32_e32 v2, 7, v[vgprLocalWriteAddrB]
	v_lshl_add_u32 v[vgprLocalWriteAddrB], v2, 5, v[vgprLocalWriteAddrB]
	v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, 0xa00, v[vgprLocalWriteAddrB] // += LDS offset B
	v_lshrrev_b32_e32 v[vgprValuC], 3, v[vgprSerial]
	v_and_b32_e32 v1, 7, v[vgprSerial]
	v_lshlrev_b32_e32 v1, 3, v1
	v_mov_b32_e32 v4, v1
	v_lshrrev_b32_e32 v2, 6, v[vgprSerial]
	v_and_b32_e32 v3, 63, v[vgprSerial]
	v_mov_b32_e32 v5, v3
	v_mul_u32_u24_e32 v10, 64, v[vgprValuC]
	v_add_lshl_u32 v10, v4, v10, 1
	v_lshrrev_b32_e32 v6, 7, v10
	v_lshl_add_u32 v10, v6, 5, v10
	v_mul_u32_u24_e32 v11, 64, v2
	v_add_lshl_u32 v11, v5, v11, 1
	v_lshrrev_b32_e32 v6, 7, v11
	v_lshl_add_u32 v11, v6, 5, v11
	v_add_co_u32 v11, vcc_lo, 0xa00, v11                   // += LDS offset B
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	v_mov_b32_e32 v18, 16
	v_mov_b32_e32 v17, s[sgprSizesFree0]                   // set Free0 size
	v_cvt_f32_u32_e32 v16, v18                             // convert to f32
	v_rcp_iflag_f32_e32 v16, v16                           // 1/x (for integer div)
	v_cvt_f32_u32_e32 v19, v17                             // convert to f32
	v_mul_f32_e32 v16, v16, v19
	v_cvt_u32_f32_e32 v16, v16                             // convert to u32
	v_mul_u32_u24_e32 v19, v16, v18
	v_sub_nc_u32_e32 v19, v17, v19
	v_cmp_ne_u32_e64 vcc_lo, v19, 0
	v_add_co_ci_u32_e64 v16, vcc_lo, v16, 0, vcc_lo        // ceil div fixup
	v_mov_b32_e32 v18, 64
	v_mov_b32_e32 v17, s[sgprSizesFree1]                   // set Free1 size
	v_readfirstlane_b32 s[sgprNumWorkGroups0], v16         // numWorkGroups0
	v_cvt_f32_u32_e32 v16, v18                             // convert to f32
	v_rcp_iflag_f32_e32 v16, v16                           // 1/x (for integer div)
	v_cvt_f32_u32_e32 v19, v17                             // convert to f32
	v_mul_f32_e32 v16, v16, v19
	v_cvt_u32_f32_e32 v16, v16                             // convert to u32
	v_mul_u32_u24_e32 v19, v16, v18
	v_sub_nc_u32_e32 v19, v17, v19
	v_cmp_ne_u32_e64 vcc_lo, v19, 0
	v_add_co_ci_u32_e64 v16, vcc_lo, v16, 0, vcc_lo        // ceil div fixup
	v_readfirstlane_b32 s[sgprNumWorkGroups1], v16         // numWorkGroups1
	s_cmp_eq_u32 s[sgprSizesFree1], 0
	s_cbranch_scc0 label_MultiGemm_AfterEndCheck           // branch to MultiGemm_AfterEndCheck
	s_endpgm                                               // Kernel End
label_MultiGemm_AfterEndCheck:
	s_mul_i32 s16, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1] // total workgroups
	s_and_b32 s17, s[sgprWGMInfo], 0x3fff                  // get GSU
	s_mul_i32 s16, s16, s17
	v_cvt_f32_u32_e32 v6, s16                              // convert to f32
	v_rcp_iflag_f32_e32 v6, v6                             // 1/x (for integer div)
	v_cvt_f32_u32_e32 v7, s[sgprWorkGroup0]                // convert to f32
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6                               // convert to u32
	v_mul_u32_u24_e64 v7, v6, s16
	v_sub_nc_u32_e32 v7, s[sgprWorkGroup0], v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v6, 1, v6
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s16, v6                            // quotient
	s_mov_b32 s[sgprWorkGroup2], s16
	s_mul_i32 s16, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0] // total workgroups
	s_mul_i32 s16, s16, s[sgprWorkGroup2]                  // batch offset
	s_mul_i32 s16, s16, s17
	s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
	v_cvt_f32_u32_e32 v6, s[sgprNumWorkGroups0]            // convert to f32
	v_rcp_iflag_f32_e32 v6, v6                             // 1/x (for integer div)
	v_cvt_f32_u32_e32 v7, s[sgprWorkGroup0]                // convert to f32
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6                               // convert to u32
	v_mul_u32_u24_e64 v7, v6, s[sgprNumWorkGroups0]
	v_sub_nc_u32_e32 v7, s[sgprWorkGroup0], v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s[sgprNumWorkGroups0]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v6, 1, v6
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v7, s[sgprNumWorkGroups0]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s16, v6                            // quotient
	s_mov_b32 s[sgprWorkGroup1], s16
	s_mul_i32 s16, s[sgprWorkGroup1], s[sgprNumWorkGroups0] // tile offset
	s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s16
	s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree2]
	s_cbranch_scc0 label_MultiGemmEnd                      // branch to MultiGemmEnd
	s_endpgm                                               // Kernel End

/******************************************/
/* Pre-pad Pointers and WGM Dispatch       */
/******************************************/

label_MultiGemmEnd:
	s_sub_u32 s[sgprAddressA], s[sgprAddressA], 16         // pre-pad to make room for possible pointer shift
	s_subb_u32 s33, s33, 0                                 // borrow from above
	s_sub_u32 s[sgprAddressB], s[sgprAddressB], 2          // pre-pad to make room for possible pointer shift
	s_subb_u32 s35, s35, 0                                 // borrow from above
	v_cmp_eq_f32_e64 vcc_lo, s[sgprAlpha], 0               // alpha == 0 ?
	s_cbranch_vccz label_AfterBetaZeroCheck                // branch if vcc==0 to AfterBetaZeroCheck
	s_mov_b32 s[sgprSizesSum0], 0
label_AfterBetaZeroCheck:
	s_and_b32 s16, s[sgprWGMInfo], 0x3fff                  // get GSU
	s_cmp_eq_u32 s16, 1
	s_cbranch_scc1 label_WGM_eq_1                          // branch to WGM_eq_1
	s_and_b32 s16, s[sgprWGMInfo], 0x4000                  // get WGM flag
	s_cbranch_scc1 label_WGM_HasFlag                       // branch to WGM_HasFlag
	s_and_b32 s16, s[sgprWGMInfo], 0x3fff                  // get GSU
	v_cvt_f32_u32_e32 v6, s16                              // convert to f32
	v_rcp_iflag_f32_e32 v6, v6                             // 1/x (for integer div)
	v_cvt_f32_u32_e32 v7, s[sgprWorkGroup1]                // convert to f32
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6                               // convert to u32
	v_mul_u32_u24_e64 v7, v6, s16
	v_sub_nc_u32_e32 v7, s[sgprWorkGroup1], v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v6, 1, v6
	v_mov_b32_e32 v7, 0
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v6, v6, 1
	v_mul_u32_u24_e64 v7, v6, s16
	v_sub_nc_u32_e32 v7, s[sgprWorkGroup1], v7
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s[sgprWorkGroup1], v6              // quotient
	v_readfirstlane_b32 s[sgprStaggerUIter], v7            // remainder
	s_branch label_WGM_DivDone                             // jump to WGM_DivDone
label_WGM_HasFlag:
	v_cvt_f32_u32_e32 v6, s[sgprNumWorkGroups1]            // convert to f32
	v_rcp_iflag_f32_e32 v6, v6                             // 1/x (for integer div)
	v_cvt_f32_u32_e32 v7, s[sgprWorkGroup1]                // convert to f32
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6                               // convert to u32
	v_mul_u32_u24_e64 v7, v6, s[sgprNumWorkGroups1]
	v_sub_nc_u32_e32 v7, s[sgprWorkGroup1], v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s[sgprNumWorkGroups1]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v6, 1, v6
	v_mov_b32_e32 v7, 0
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v7, s[sgprNumWorkGroups1]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v6, v6, 1
	v_mul_u32_u24_e64 v7, v6, s[sgprNumWorkGroups1]
	v_sub_nc_u32_e32 v7, s[sgprWorkGroup1], v7
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s[sgprStaggerUIter], v6            // quotient
	v_readfirstlane_b32 s[sgprWorkGroup1], v7              // remainder
label_WGM_DivDone:
	s_mov_b32 s[sgprLoopCounterL], 1
	s_mov_b32 s9, 2
	s_branch label_WGM_Common                              // jump to WGM_Common
label_WGM_eq_1:
	s_mov_b64 s[sgprStaggerUIter:sgprStaggerUIter+1], 0
	s_mov_b32 s[sgprLoopCounterL], 1
	s_mov_b32 s9, 1

/******************************************/
/* Workgroup Mapping (Common Path)          */
/******************************************/

label_WGM_Common:
	s_mov_b32 s16, s[sgprKernelInfo1]
	s_sext_i32_i16 s16, s16                                // sign extend i16 to i32
	s_cmp_gt_i32 s16, 1
	s_cbranch_scc1 label_WGM_Positive                      // branch to WGM_Positive
	s_cmp_ge_i32 s16, 0
	s_cbranch_scc1 label_WGM_SkipNeg                       // branch to WGM_SkipNeg
	s_abs_i32 s16, s16                                     // absolute value
	v_cvt_f64_u32_e32 v[16:17], s16                        // convert to f64
	v_rcp_f64_e32 v[16:17], v[16:17]                       // 1/x (for integer div)
	v_cvt_f64_u32_e32 v[18:19], s[sgprWorkGroup0]          // convert to f64
	v_mul_f64 v[16:17], v[16:17], v[18:19]
	v_cvt_u32_f64_e32 v16, v[16:17]                        // convert to u32
	v_mul_lo_u32 v17, v16, s16
	v_sub_nc_u32_e32 v18, s[sgprWorkGroup0], v17
	v_cmp_ge_u32_e64 vcc_lo, v18, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e64 v16, v16, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s17, v16                           // quotient
	s_mul_i32 s[sgprGemmInfo], s17, s16
	s_sub_u32 s[sgprGemmInfo], s[sgprWorkGroup0], s[sgprGemmInfo]
	s_mul_i32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprNumWorkGroups1] // linearize WG index
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprWorkGroup1]
	v_cvt_f64_u32_e32 v[16:17], s16                        // convert to f64
	v_rcp_f64_e32 v[16:17], v[16:17]                       // 1/x (for integer div)
	v_cvt_f64_u32_e32 v[18:19], s[sgprNumWorkGroups0]      // convert to f64
	v_mul_f64 v[16:17], v[16:17], v[18:19]
	v_cvt_u32_f64_e32 v16, v[16:17]                        // convert to u32
	v_mul_lo_u32 v17, v16, s16
	v_sub_nc_u32_e32 v18, s[sgprNumWorkGroups0], v17
	v_cmp_ge_u32_e64 vcc_lo, v18, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e64 v16, v16, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s18, v16                           // quotient
	s_mul_i32 s19, s16, s18
	s_sub_u32 s19, s[sgprNumWorkGroups0], s19
	s_cmp_eq_u32 s19, 0
	s_cmov_b32 s19, s16                                    // set full stripe if no remainder
	s_cmp_ge_u32 s17, s18
	s_cselect_b32 s18, s19, s16                            // select stripe width
	v_cvt_f64_u32_e32 v[16:17], s18                        // convert to f64
	v_rcp_f64_e32 v[16:17], v[16:17]                       // 1/x (for integer div)
	v_cvt_f64_u32_e32 v[18:19], s[sgprGemmInfo]            // convert to f64
	v_mul_f64 v[16:17], v[16:17], v[18:19]
	v_cvt_u32_f64_e32 v16, v[16:17]                        // convert to u32
	v_mul_lo_u32 v17, v16, s18
	v_sub_nc_u32_e32 v18, s[sgprGemmInfo], v17
	v_cmp_ge_u32_e64 vcc_lo, v18, s18
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e64 v16, v16, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_mul_lo_u32 v17, v16, s18
	v_sub_nc_u32_e32 v18, s[sgprGemmInfo], v17
	v_readfirstlane_b32 s[sgprWorkGroup1], v16             // quotient
	v_readfirstlane_b32 s[sgprWorkGroup0], v18             // remainder
	s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s18    // tile offset
	s_sub_u32 s[sgprWorkGroup0], s[sgprGemmInfo], s[sgprWorkGroup0]
	s_mul_i32 s17, s17, s16
	s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s17
	s_branch label_WGM_SkipNeg                             // jump to WGM_SkipNeg
label_WGM_Positive:
	s_mov_b32 s16, s16
	v_cvt_f64_u32_e32 v[16:17], s16                        // convert to f64
	v_rcp_f64_e32 v[16:17], v[16:17]                       // 1/x (for integer div)
	v_cvt_f64_u32_e32 v[18:19], s[sgprWorkGroup1]          // convert to f64
	v_mul_f64 v[16:17], v[16:17], v[18:19]
	v_cvt_u32_f64_e32 v16, v[16:17]                        // convert to u32
	v_mul_lo_u32 v17, v16, s16
	v_sub_nc_u32_e32 v18, s[sgprWorkGroup1], v17
	v_cmp_ge_u32_e64 vcc_lo, v18, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e64 v16, v16, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s17, v16                           // quotient
	s_mul_i32 s[sgprGemmInfo], s17, s16
	s_sub_u32 s[sgprGemmInfo], s[sgprWorkGroup1], s[sgprGemmInfo]
	s_mul_i32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprNumWorkGroups0] // linearize WG index
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprWorkGroup0]
	v_cvt_f64_u32_e32 v[16:17], s16                        // convert to f64
	v_rcp_f64_e32 v[16:17], v[16:17]                       // 1/x (for integer div)
	v_cvt_f64_u32_e32 v[18:19], s[sgprNumWorkGroups1]      // convert to f64
	v_mul_f64 v[16:17], v[16:17], v[18:19]
	v_cvt_u32_f64_e32 v16, v[16:17]                        // convert to u32
	v_mul_lo_u32 v17, v16, s16
	v_sub_nc_u32_e32 v18, s[sgprNumWorkGroups1], v17
	v_cmp_ge_u32_e64 vcc_lo, v18, s16
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e64 v16, v16, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s18, v16                           // quotient
	s_mul_i32 s19, s16, s18
	s_sub_u32 s19, s[sgprNumWorkGroups1], s19
	s_cmp_eq_u32 s19, 0
	s_cmov_b32 s19, s16                                    // set full stripe if no remainder
	s_cmp_ge_u32 s17, s18
	s_cselect_b32 s18, s19, s16                            // select stripe width
	v_cvt_f64_u32_e32 v[16:17], s18                        // convert to f64
	v_rcp_f64_e32 v[16:17], v[16:17]                       // 1/x (for integer div)
	v_cvt_f64_u32_e32 v[18:19], s[sgprGemmInfo]            // convert to f64
	v_mul_f64 v[16:17], v[16:17], v[18:19]
	v_cvt_u32_f64_e32 v16, v[16:17]                        // convert to u32
	v_mul_lo_u32 v17, v16, s18
	v_sub_nc_u32_e32 v18, s[sgprGemmInfo], v17
	v_cmp_ge_u32_e64 vcc_lo, v18, s18
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e64 v16, v16, 1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_mul_lo_u32 v17, v16, s18
	v_sub_nc_u32_e32 v18, s[sgprGemmInfo], v17
	v_readfirstlane_b32 s[sgprWorkGroup0], v16             // quotient
	v_readfirstlane_b32 s[sgprWorkGroup1], v18             // remainder
	s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s18    // tile offset
	s_sub_u32 s[sgprWorkGroup1], s[sgprGemmInfo], s[sgprWorkGroup1]
	s_mul_i32 s17, s17, s16
	s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s17

/******************************************/
/* SRD Address Computation                 */
/******************************************/

label_WGM_SkipNeg:
	s_mul_hi_u32 s19, s[sgprWorkGroup0], 16                // high 32 bits of multiply
	s_mul_i32 s18, s[sgprWorkGroup0], 16                   // tile offset
	s_mul_hi_u32 s19, s18, s[sgprStrideA0]                 // high 32 bits of multiply
	s_mul_i32 s18, s18, s[sgprStrideA0]                    // A stride offset
	s_and_b32 s16, s[sgprWGMInfo], 0x8000                  // get stagger flag
	s_cbranch_scc1 label_SrdA_StaggerCalc                  // branch to SrdA_StaggerCalc
	s_mul_hi_u32 s17, 64, s[sgprStaggerUIter]              // high 32 bits of multiply
	s_mul_i32 s16, 64, s[sgprStaggerUIter]                 // stagger offset
	s_branch label_SrdA_StaggerDone                        // jump to SrdA_StaggerDone
label_SrdA_StaggerCalc:
	s_lshr_b32 s12, s[sgprSizesSum0], 6
	s_and_b32 s[sgprStaggerUStride], s[sgprWGMInfo], 0x3fff // get GSU
	v_cvt_f32_u32_e32 v6, s[sgprStaggerUStride]            // convert to f32
	v_rcp_iflag_f32_e32 v6, v6                             // 1/x (for integer div)
	v_cvt_f32_u32_e32 v7, s12                              // convert to f32
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6                               // convert to u32
	v_mul_u32_u24_e64 v7, v6, s[sgprStaggerUStride]
	v_sub_nc_u32_e32 v7, s12, v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s[sgprStaggerUStride]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v6, 1, v6
	v_mov_b32_e32 v7, 0
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v7, s[sgprStaggerUStride]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v6, v6, 1
	v_mul_u32_u24_e64 v7, v6, s[sgprStaggerUStride]
	v_sub_nc_u32_e32 v7, s12, v7
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s12, v6                            // quotient
	v_readfirstlane_b32 s[sgprStaggerUStride], v7          // remainder
	s_mul_i32 s17, s12, s[sgprStaggerUIter]                // stagger offset
	s_add_u32 s16, 1, s12
	s_add_u32 s17, s17, s[sgprStaggerUStride]
	s_mul_i32 s16, s16, s[sgprStaggerUIter]                // stagger offset
	s_cmp_lt_u32 s[sgprStaggerUIter], s[sgprStaggerUStride]
	s_cselect_b32 s16, s16, s17                            // select stagger offset
	s_mul_hi_u32 s17, s16, 64                              // high 32 bits of multiply
	s_mul_i32 s16, s16, 64
label_SrdA_StaggerDone:
	s_add_u32 s18, s18, s16
	s_addc_u32 s19, s19, s17                               // add with carry
	s_mov_b64 s[56:57], 1
	s_sub_u32 s16, s[sgprSizesSum0], 1
	s_mul_hi_u32 s17, 1, s16                               // high 32 bits of multiply
	s_mul_i32 s16, 1, s16
	s_add_u32 s56, s56, s16
	s_addc_u32 s57, s57, s17                               // add with carry
	s_sub_u32 s16, s[sgprSizesFree0], 1
	s_mul_hi_u32 s17, s[sgprStrideA0], s16                 // high 32 bits of multiply
	s_mul_i32 s16, s[sgprStrideA0], s16                    // A stride offset
	s_add_u32 s56, s56, s16
	s_addc_u32 s57, s57, s17                               // add with carry
	s_sub_u32 s56, s56, s18
	s_subb_u32 s57, s57, s19                               // sub with borrow
	s_lshl_b64 s[56:57], s[56:57], 1                       // scale by BPE
	s_add_u32 s56, s56, 16
	s_addc_u32 s57, s57, 0                                 // carry from above
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1                             // SRD A num records
	s_mul_hi_u32 s17, s[sgprStrideA1], s[sgprWorkGroup2]   // high 32 bits of multiply
	s_mul_i32 s16, s[sgprStrideA1], s[sgprWorkGroup2]      // batch stride
	s_add_u32 s18, s18, s16
	s_addc_u32 s19, s19, s17                               // add with carry
	s_lshl_b64 s[18:19], s[18:19], 1                       // scale by BPE
	s_add_u32 s[sgprSrdA], s[sgprAddressA], s18            // adjust SRD A
	s_addc_u32 s49, s33, s19                               // add with carry
	s_mov_b32 s51, 0x31004000                              // SRD Dword3
	s_mul_hi_u32 s19, s[sgprWorkGroup1], 64                // high 32 bits of multiply
	s_mul_i32 s18, s[sgprWorkGroup1], 64                   // tile offset
	s_mul_hi_u32 s19, s18, s[sgprStrideB0]                 // high 32 bits of multiply
	s_mul_i32 s18, s18, s[sgprStrideB0]                    // B stride offset
	s_and_b32 s16, s[sgprWGMInfo], 0x8000                  // get stagger flag
	s_cbranch_scc1 label_SrdB_StaggerCalc                  // branch to SrdB_StaggerCalc
	s_mul_hi_u32 s17, 64, s[sgprStaggerUIter]              // high 32 bits of multiply
	s_mul_i32 s16, 64, s[sgprStaggerUIter]                 // stagger offset
	s_branch label_SrdB_StaggerDone                        // jump to SrdB_StaggerDone
label_SrdB_StaggerCalc:
	s_lshr_b32 s12, s[sgprSizesSum0], 6
	s_and_b32 s[sgprStaggerUStride], s[sgprWGMInfo], 0x3fff // get GSU
	v_cvt_f32_u32_e32 v6, s[sgprStaggerUStride]            // convert to f32
	v_rcp_iflag_f32_e32 v6, v6                             // 1/x (for integer div)
	v_cvt_f32_u32_e32 v7, s12                              // convert to f32
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6                               // convert to u32
	v_mul_u32_u24_e64 v7, v6, s[sgprStaggerUStride]
	v_sub_nc_u32_e32 v7, s12, v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s[sgprStaggerUStride]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v6, 1, v6
	v_mov_b32_e32 v7, 0
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v7, s[sgprStaggerUStride]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v6, v6, 1
	v_mul_u32_u24_e64 v7, v6, s[sgprStaggerUStride]
	v_sub_nc_u32_e32 v7, s12, v7
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s12, v6                            // quotient
	v_readfirstlane_b32 s[sgprStaggerUStride], v7          // remainder
	s_mul_i32 s17, s12, s[sgprStaggerUIter]                // stagger offset
	s_add_u32 s16, 1, s12
	s_add_u32 s17, s17, s[sgprStaggerUStride]
	s_mul_i32 s16, s16, s[sgprStaggerUIter]                // stagger offset
	s_cmp_lt_u32 s[sgprStaggerUIter], s[sgprStaggerUStride]
	s_cselect_b32 s16, s16, s17                            // select stagger offset
	s_mul_hi_u32 s17, s16, 64                              // high 32 bits of multiply
	s_mul_i32 s16, s16, 64
label_SrdB_StaggerDone:
	s_add_u32 s18, s18, s16
	s_addc_u32 s19, s19, s17                               // add with carry
	s_mov_b64 s[58:59], 1
	s_sub_u32 s16, s[sgprSizesSum0], 1
	s_mul_hi_u32 s17, 1, s16                               // high 32 bits of multiply
	s_mul_i32 s16, 1, s16
	s_add_u32 s58, s58, s16
	s_addc_u32 s59, s59, s17                               // add with carry
	s_sub_u32 s16, s[sgprSizesFree1], 1
	s_mul_hi_u32 s17, s[sgprStrideB0], s16                 // high 32 bits of multiply
	s_mul_i32 s16, s[sgprStrideB0], s16                    // B stride offset
	s_add_u32 s58, s58, s16
	s_addc_u32 s59, s59, s17                               // add with carry
	s_sub_u32 s58, s58, s18
	s_subb_u32 s59, s59, s19                               // sub with borrow
	s_lshl_b64 s[58:59], s[58:59], 1                       // scale by BPE
	s_add_u32 s58, s58, 2
	s_addc_u32 s59, s59, 0                                 // carry from above
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1                             // SRD B num records
	s_mul_hi_u32 s17, s[sgprStrideB1], s[sgprWorkGroup2]   // high 32 bits of multiply
	s_mul_i32 s16, s[sgprStrideB1], s[sgprWorkGroup2]      // batch stride
	s_add_u32 s18, s18, s16
	s_addc_u32 s19, s19, s17                               // add with carry
	s_lshl_b64 s[18:19], s[18:19], 1                       // scale by BPE
	s_add_u32 s[sgprSrdB], s[sgprAddressB], s18            // adjust SRD B
	s_addc_u32 s53, s35, s19                               // add with carry
	s_mov_b32 s55, 0x31004000                              // SRD Dword3
	v_mul_lo_u32 v16, s[sgprStrideA0], v[vgprValuC]        // offset * strideA0
	v_add_co_u32 v[vgprGlobalReadOffsetA], vcc_lo, v1, v16
	v_add_nc_u32_e32 v[vgprGlobalReadOffsetA], 8, v[vgprGlobalReadOffsetA]
	v_lshlrev_b32_e32 v[vgprGlobalReadOffsetA], 1, v[vgprGlobalReadOffsetA]
	v_mul_lo_u32 v16, s[sgprStrideB0], v2                  // offset * strideB0
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v3, v16
	v_add_nc_u32_e32 v[vgprGlobalReadOffsetB], 1, v[vgprGlobalReadOffsetB]
	v_lshlrev_b32_e32 v[vgprGlobalReadOffsetB], 1, v[vgprGlobalReadOffsetB]
	s_mul_i32 s66, s[sgprStrideB0], 2                      // B stride offset
	s_lshl_b32 s66, s66, 1                                 // scale by BPE
	s_mul_i32 s67, s[sgprStrideB0], 4                      // B stride offset
	s_lshl_b32 s67, s67, 1                                 // scale by BPE
	s_mul_i32 s68, s[sgprStrideB0], 6                      // B stride offset
	s_lshl_b32 s68, s68, 1                                 // scale by BPE
	s_mul_i32 s69, s[sgprStrideB0], 8                      // B stride offset
	s_lshl_b32 s69, s69, 1                                 // scale by BPE
	s_mul_i32 s70, s[sgprStrideB0], 10                     // B stride offset
	s_lshl_b32 s70, s70, 1                                 // scale by BPE
	s_mul_i32 s71, s[sgprStrideB0], 12                     // B stride offset
	s_lshl_b32 s71, s71, 1                                 // scale by BPE
	s_mul_i32 s72, s[sgprStrideB0], 14                     // B stride offset
	s_lshl_b32 s72, s72, 1                                 // scale by BPE
	s_mul_i32 s73, s[sgprStrideB0], 16                     // B stride offset
	s_lshl_b32 s73, s73, 1                                 // scale by BPE
	s_mul_i32 s74, s[sgprStrideB0], 18                     // B stride offset
	s_lshl_b32 s74, s74, 1                                 // scale by BPE
	s_mul_i32 s75, s[sgprStrideB0], 20                     // B stride offset
	s_lshl_b32 s75, s75, 1                                 // scale by BPE
	s_mul_i32 s76, s[sgprStrideB0], 22                     // B stride offset
	s_lshl_b32 s76, s76, 1                                 // scale by BPE
	s_mul_i32 s77, s[sgprStrideB0], 24                     // B stride offset
	s_lshl_b32 s77, s77, 1                                 // scale by BPE
	s_mul_i32 s78, s[sgprStrideB0], 26                     // B stride offset
	s_lshl_b32 s78, s78, 1                                 // scale by BPE
	s_mul_i32 s79, s[sgprStrideB0], 28                     // B stride offset
	s_lshl_b32 s79, s79, 1                                 // scale by BPE
	s_mul_i32 s80, s[sgprStrideB0], 30                     // B stride offset
	s_lshl_b32 s80, s80, 1                                 // scale by BPE
	s_mul_i32 s81, s[sgprStrideB0], 32                     // B stride offset
	s_lshl_b32 s81, s81, 1                                 // scale by BPE
	s_mul_i32 s82, s[sgprStrideB0], 34                     // B stride offset
	s_lshl_b32 s82, s82, 1                                 // scale by BPE
	s_mul_i32 s83, s[sgprStrideB0], 36                     // B stride offset
	s_lshl_b32 s83, s83, 1                                 // scale by BPE
	s_mul_i32 s84, s[sgprStrideB0], 38                     // B stride offset
	s_lshl_b32 s84, s84, 1                                 // scale by BPE
	s_mul_i32 s85, s[sgprStrideB0], 40                     // B stride offset
	s_lshl_b32 s85, s85, 1                                 // scale by BPE
	s_mul_i32 s86, s[sgprStrideB0], 42                     // B stride offset
	s_lshl_b32 s86, s86, 1                                 // scale by BPE
	s_mul_i32 s87, s[sgprStrideB0], 44                     // B stride offset
	s_lshl_b32 s87, s87, 1                                 // scale by BPE
	s_mul_i32 s88, s[sgprStrideB0], 46                     // B stride offset
	s_lshl_b32 s88, s88, 1                                 // scale by BPE
	s_mul_i32 s89, s[sgprStrideB0], 48                     // B stride offset
	s_lshl_b32 s89, s89, 1                                 // scale by BPE
	s_mul_i32 s90, s[sgprStrideB0], 50                     // B stride offset
	s_lshl_b32 s90, s90, 1                                 // scale by BPE
	s_mul_i32 s91, s[sgprStrideB0], 52                     // B stride offset
	s_lshl_b32 s91, s91, 1                                 // scale by BPE
	s_mul_i32 s92, s[sgprStrideB0], 54                     // B stride offset
	s_lshl_b32 s92, s92, 1                                 // scale by BPE
	s_mul_i32 s93, s[sgprStrideB0], 56                     // B stride offset
	s_lshl_b32 s93, s93, 1                                 // scale by BPE
	s_mul_i32 s94, s[sgprStrideB0], 58                     // B stride offset
	s_lshl_b32 s94, s94, 1                                 // scale by BPE
	s_mul_i32 s95, s[sgprStrideB0], 60                     // B stride offset
	s_lshl_b32 s95, s95, 1                                 // scale by BPE
	s_mul_i32 s96, s[sgprStrideB0], 62                     // B stride offset
	s_lshl_b32 s96, s96, 1                                 // scale by BPE
	s_and_b32 s17, s[sgprWGMInfo], 0x3fff                  // get GSU
	s_mul_i32 s17, s17, 0x80
	s_and_b32 s16, s[sgprWGMInfo], 0x8000                  // get stagger flag
	s_cselect_b32 s64, 0x80, s17                           // StaggerU stride A
	s_and_b32 s17, s[sgprWGMInfo], 0x3fff                  // get GSU
	s_mul_i32 s17, s17, 0x80
	s_and_b32 s16, s[sgprWGMInfo], 0x8000                  // get stagger flag
	s_cselect_b32 s65, 0x80, s17                           // StaggerU stride B
	s_lshr_b32 s12, s[sgprSizesSum0], 6
	s_and_b32 s16, s[sgprWGMInfo], 0x3fff                  // get GSU
	s_cmp_eq_u32 s16, 1
	s_cbranch_scc1 label_StaggerU_WGM1                     // branch to StaggerU_WGM1
	s_and_b32 s[sgprStaggerUStride], s[sgprWGMInfo], 0x3fff // get GSU
	v_cvt_f32_u32_e32 v[vgprValuC], s[sgprStaggerUStride]  // convert to f32
	v_rcp_iflag_f32_e32 v[vgprValuC], v[vgprValuC]         // 1/x (for integer div)
	v_cvt_f32_u32_e32 v1, s12                              // convert to f32
	v_mul_f32_e32 v[vgprValuC], v[vgprValuC], v1
	v_cvt_u32_f32_e32 v[vgprValuC], v[vgprValuC]           // convert to u32
	v_mul_u32_u24_e64 v1, v[vgprValuC], s[sgprStaggerUStride]
	v_sub_nc_u32_e32 v1, s12, v1
	v_cmp_eq_u32_e64 vcc_lo, v1, s[sgprStaggerUStride]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v[vgprValuC], 1, v[vgprValuC]
	v_mov_b32_e32 v1, 0
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v1, s[sgprStaggerUStride]
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v[vgprValuC], v[vgprValuC], 1
	v_mul_u32_u24_e64 v1, v[vgprValuC], s[sgprStaggerUStride]
	v_sub_nc_u32_e32 v1, s12, v1
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s12, v[vgprValuC]                  // quotient
	v_readfirstlane_b32 s[sgprStaggerUStride], v1          // remainder
	s_add_u32 s16, 1, s12
	s_cmp_lt_u32 s[sgprStaggerUIter], s[sgprStaggerUStride]
	s_cmov_b32 s12, s16                                    // round up quotient

/******************************************/
/* StaggerU Shift Calculation              */
/******************************************/

label_StaggerU_WGM1:
	s_mov_b32 s13, s12
	s_and_b32 s18, s[sgprWGM], 0x1f00                      // get StaggerU shift
	s_lshr_b32 s18, s18, 8
	s_and_b32 s19, s[sgprWGM], 0xe000                      // get StaggerU dim
	s_and_b32 s[sgprWGM], s[sgprWGM], 0xff                 // mask WGM value
	s_mov_b32 s16, s[sgprWGM]
label_StaggerU_ShiftLoop:
	s_lshl_b32 s17, s16, s18
	s_cmp_ge_u32 s13, s17
	s_cbranch_scc1 label_StaggerU_ShiftDone                // branch to StaggerU_ShiftDone
	s_lshr_b32 s16, s16, 1
	s_branch label_StaggerU_ShiftLoop                      // jump to StaggerU_ShiftLoop
label_StaggerU_ShiftDone:
	s_sub_u32 s17, s16, 1
	s_cmp_ge_u32 s16, 1
	s_cselect_b32 s47, s17, 0                              // StaggerU mask
	s_cmp_eq_u32 s19, 0
	s_cbranch_scc1 label_StaggerU_Dim1                     // branch to StaggerU_Dim1
	s_mov_b32 s16, s[sgprWorkGroup0]
	s_branch label_StaggerU_DimDone                        // jump to StaggerU_DimDone
label_StaggerU_Dim1:
	s_cmp_eq_u32 s19, 0x2000
	s_cbranch_scc1 label_StaggerU_Dim2                     // branch to StaggerU_Dim2
	s_mov_b32 s16, s[sgprWorkGroup1]
	s_branch label_StaggerU_DimDone                        // jump to StaggerU_DimDone
label_StaggerU_Dim2:
	s_cmp_eq_u32 s19, 0x4000
	s_cbranch_scc1 label_StaggerU_Dim3                     // branch to StaggerU_Dim3
	s_mov_b32 s16, -1
	s_branch label_StaggerU_DimDone                        // jump to StaggerU_DimDone
label_StaggerU_Dim3:
	s_cmp_eq_u32 s19, 0x6000
	s_cbranch_scc1 label_StaggerU_Dim4                     // branch to StaggerU_Dim4
	s_mul_i32 s17, s[sgprNumWorkGroups0], s[sgprWorkGroup1] // linearize WG index
	s_add_u32 s16, s16, s17
	s_add_u32 s16, s16, s[sgprWorkGroup0]
	s_branch label_StaggerU_DimDone                        // jump to StaggerU_DimDone
label_StaggerU_Dim4:
	s_cmp_eq_u32 s19, 0x8000
	s_cbranch_scc1 label_StaggerU_DimDone                  // branch to StaggerU_DimDone
	s_mov_b32 s16, -1
	s_branch label_StaggerU_DimDone                        // jump to StaggerU_DimDone
label_StaggerU_DimDone:
	s_and_b32 s47, s47, s16
	s_lshl_b32 s47, s47, s18
	s_mul_hi_i32 s17, s47, s64                             // high 32 bits of multiply
	s_mul_i32 s16, s47, s64
	s_mul_hi_i32 s61, s12, s64                             // high 32 bits of multiply
	s_mul_i32 s60, s12, s64
	s_sub_u32 s60, s64, s60
	s_subb_u32 s61, 0, s61                                 // sub with borrow
	s_add_u32 s[sgprSrdA], s[sgprSrdA], s16                // adjust SRD A
	s_addc_u32 s49, s49, s17                               // add with carry
	s_sub_u32 s56, s56, s16
	s_subb_u32 s57, s57, s17                               // sub with borrow
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1                             // SRD A num records
	s_mul_hi_i32 s17, s47, s65                             // high 32 bits of multiply
	s_mul_i32 s16, s47, s65
	s_mul_hi_i32 s63, s12, s65                             // high 32 bits of multiply
	s_mul_i32 s62, s12, s65
	s_sub_u32 s62, s65, s62
	s_subb_u32 s63, 0, s63                                 // sub with borrow
	s_add_u32 s[sgprSrdB], s[sgprSrdB], s16                // adjust SRD B
	s_addc_u32 s53, s53, s17                               // add with carry
	s_sub_u32 s58, s58, s16
	s_subb_u32 s59, s59, s17                               // sub with borrow
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1                             // SRD B num records
	s_add_u32 s47, s47, 2
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc1 label_InitAccZero                       // branch to InitAccZero

/******************************************/
/* Global Read: Initial Load               */
/******************************************/

	buffer_load_b128 v[82:85], v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen // G -> Reg A
	buffer_load_d16_b16 v86, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], 0 offen // G -> Reg B
	buffer_load_d16_hi_b16 v86, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s66 offen // G -> Reg B
	buffer_load_d16_b16 v87, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s67 offen // G -> Reg B
	buffer_load_d16_hi_b16 v87, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s68 offen // G -> Reg B
	buffer_load_d16_b16 v88, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s69 offen // G -> Reg B
	buffer_load_d16_hi_b16 v88, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s70 offen // G -> Reg B
	buffer_load_d16_b16 v89, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s71 offen // G -> Reg B
	buffer_load_d16_hi_b16 v89, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s72 offen // G -> Reg B
	buffer_load_d16_b16 v90, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s73 offen // G -> Reg B
	buffer_load_d16_hi_b16 v90, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s74 offen // G -> Reg B
	buffer_load_d16_b16 v91, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s75 offen // G -> Reg B
	buffer_load_d16_hi_b16 v91, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s76 offen // G -> Reg B
	buffer_load_d16_b16 v92, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s77 offen // G -> Reg B
	buffer_load_d16_hi_b16 v92, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s78 offen // G -> Reg B
	buffer_load_d16_b16 v93, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s79 offen // G -> Reg B
	buffer_load_d16_hi_b16 v93, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s80 offen // G -> Reg B
	buffer_load_d16_b16 v94, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s81 offen // G -> Reg B
	buffer_load_d16_hi_b16 v94, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s82 offen // G -> Reg B
	buffer_load_d16_b16 v95, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s83 offen // G -> Reg B
	buffer_load_d16_hi_b16 v95, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s84 offen // G -> Reg B
	buffer_load_d16_b16 v96, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s85 offen // G -> Reg B
	buffer_load_d16_hi_b16 v96, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s86 offen // G -> Reg B
	buffer_load_d16_b16 v97, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s87 offen // G -> Reg B
	buffer_load_d16_hi_b16 v97, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s88 offen // G -> Reg B
	buffer_load_d16_b16 v98, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s89 offen // G -> Reg B
	buffer_load_d16_hi_b16 v98, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s90 offen // G -> Reg B
	buffer_load_d16_b16 v99, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s91 offen // G -> Reg B
	buffer_load_d16_hi_b16 v99, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s92 offen // G -> Reg B
	buffer_load_d16_b16 v100, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s93 offen // G -> Reg B
	buffer_load_d16_hi_b16 v100, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s94 offen // G -> Reg B
	buffer_load_d16_b16 v101, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s95 offen // G -> Reg B
	buffer_load_d16_hi_b16 v101, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s96 offen // G -> Reg B
	s_add_u32 s18, s12, 1
	s_cmp_eq_u32 s47, s18
	s_cselect_b32 s16, s60, s64                            // select A stride
	s_cselect_b32 s17, s61, 0                              // select A stride high
	s_add_u32 s[sgprSrdA], s[sgprSrdA], s16                // adjust SRD A
	s_addc_u32 s49, s49, s17                               // add with carry
	s_sub_u32 s56, s56, s16
	s_subb_u32 s57, s57, s17                               // sub with borrow
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1                             // SRD A num records
	s_add_u32 s18, s12, 1
	s_cmp_eq_u32 s47, s18
	s_cselect_b32 s16, s62, s65                            // select B stride
	s_cselect_b32 s17, s63, 0                              // select B stride high
	s_add_u32 s[sgprSrdB], s[sgprSrdB], s16                // adjust SRD B
	s_addc_u32 s53, s53, s17                               // add with carry
	s_sub_u32 s58, s58, s16
	s_subb_u32 s59, s59, s17                               // sub with borrow
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1                             // SRD B num records

/******************************************/
/* Setup SRD C/D and Output Offsets         */
/******************************************/

label_InitAccZero:
	s_mov_b64 s[16:17], s[sgprAddressD:sgprAddressD+1]
	s_mov_b32 s18, 0x80000000                              // buffer size = max
	s_mov_b32 s19, 0x31004000                              // SRD Dword3
	s_mov_b64 s[sgprGemmInfo:sgprGemmInfo+1], s[sgprAddressC:sgprAddressC+1]
	s_mov_b32 s[sgprKernelInfo0], 0x80000000               // buffer size = max
	s_mov_b32 s[sgprNumWG], 0x31004000                     // SRD Dword3
	s_mul_i32 s100, 64, s[sgprWorkGroup1]                  // tile offset
	s_mul_hi_u32 s99, s100, s[sgprStrideC0]                // high 32 bits of multiply
	s_mul_i32 s98, s100, s[sgprStrideC0]                   // C stride offset
	s_lshl_b64 s[98:99], s[98:99], s[sgprLoopCounterL]     // scale by BPE
	s_add_u32 s[sgprGemmInfo], s[sgprAddressC], s98
	s_addc_u32 s[sgprArgType2], s31, s99                   // add with carry
	s_mul_hi_u32 s99, s100, s[sgprStrideD0]                // high 32 bits of multiply
	s_mul_i32 s98, s100, s[sgprStrideD0]                   // D stride offset
	s_lshl_b64 s[98:99], s[98:99], s9                      // scale by BPE
	s_add_u32 s16, s[sgprAddressD], s98
	s_addc_u32 s17, s29, s99                               // add with carry
	s_mul_hi_u32 s99, s[sgprWorkGroup2], s[sgprStrideC1]   // high 32 bits of multiply
	s_mul_i32 s98, s[sgprWorkGroup2], s[sgprStrideC1]      // batch stride
	s_lshl_b64 s[98:99], s[98:99], s[sgprLoopCounterL]     // scale by BPE
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s98
	s_addc_u32 s[sgprArgType2], s[sgprArgType2], s99       // add with carry
	s_mul_hi_u32 s99, s[sgprWorkGroup2], s[sgprStrideD1]   // high 32 bits of multiply
	s_mul_i32 s98, s[sgprWorkGroup2], s[sgprStrideD1]      // batch stride
	s_lshl_b64 s[98:99], s[98:99], s9                      // scale by BPE
	s_add_u32 s16, s16, s98
	s_addc_u32 s17, s17, s99                               // add with carry
	s_and_b32 s97, s[sgprWGMInfo], 0x3fff                  // get GSU
	s_cmp_eq_u32 s97, 1
	s_cbranch_scc1 label_SkipTailLoopCheck                 // branch to SkipTailLoopCheck

/******************************************/
/* GSU Tail Loop Offset                    */
/******************************************/

	s_mul_hi_u32 s99, s[sgprSizesFree0], s[sgprStaggerUIter] // high 32 bits of multiply
	s_mul_i32 s98, s[sgprSizesFree0], s[sgprStaggerUIter]  // stagger offset
	s_sub_u32 s97, s[sgprSizesFree1], 1
	s_mul_i32 s97, s97, s[sgprStaggerUIter]                // stagger offset
	s_mul_hi_u32 s100, s97, s[sgprStrideC0]                // high 32 bits of multiply
	s_mul_i32 s97, s97, s[sgprStrideC0]                    // C stride offset
	s_add_u32 s98, s98, s97
	s_addc_u32 s99, s99, s100                              // add with carry
	s_sub_u32 s97, s[sgprSizesFree2], 1
	s_mul_i32 s97, s97, s[sgprStaggerUIter]                // stagger offset
	s_mul_hi_u32 s100, s97, s[sgprStrideC1]                // high 32 bits of multiply
	s_mul_i32 s97, s97, s[sgprStrideC1]                    // batch stride
	s_add_u32 s98, s98, s97
	s_addc_u32 s99, s99, s100                              // add with carry
	s_lshl_b64 s[98:99], s[98:99], 2                       // scale by BPE
	s_add_u32 s16, s16, s98
	s_addc_u32 s17, s17, s99                               // add with carry
label_SkipTailLoopCheck:
	v_mov_b32_e32 v[vgprValuC], 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc0 label_SetPC_End                         // branch to SetPC_End
	s_getpc_b64 s[98:99]                                   // addr of next instr
	s_add_i32 s100, 0xe84, 4
	s_add_u32 s98, s98, s100
	s_addc_u32 s99, s99, 0                                 // carry from above
	s_setpc_b64 s[98:99]                                   // early exit jump
label_SetPC_End:
	s_waitcnt vmcnt(0)                                     // wait for global read
	ds_store_b128 v10, v[82:85]                            // local write
	ds_store_b16 v11, v86                                  // local write
	ds_store_b16_d16_hi v11, v86 offset:320                // local write
	ds_store_b16 v11, v87 offset:640                       // local write
	ds_store_b16_d16_hi v11, v87 offset:960                // local write
	ds_store_b16 v11, v88 offset:1280                      // local write
	ds_store_b16_d16_hi v11, v88 offset:1600               // local write
	ds_store_b16 v11, v89 offset:1920                      // local write
	ds_store_b16_d16_hi v11, v89 offset:2240               // local write
	ds_store_b16 v11, v90 offset:2560                      // local write
	ds_store_b16_d16_hi v11, v90 offset:2880               // local write
	ds_store_b16 v11, v91 offset:3200                      // local write
	ds_store_b16_d16_hi v11, v91 offset:3520               // local write
	ds_store_b16 v11, v92 offset:3840                      // local write
	ds_store_b16_d16_hi v11, v92 offset:4160               // local write
	ds_store_b16 v11, v93 offset:4480                      // local write
	ds_store_b16_d16_hi v11, v93 offset:4800               // local write
	ds_store_b16 v11, v94 offset:5120                      // local write
	ds_store_b16_d16_hi v11, v94 offset:5440               // local write
	ds_store_b16 v11, v95 offset:5760                      // local write
	ds_store_b16_d16_hi v11, v95 offset:6080               // local write
	ds_store_b16 v11, v96 offset:6400                      // local write
	ds_store_b16_d16_hi v11, v96 offset:6720               // local write
	ds_store_b16 v11, v97 offset:7040                      // local write
	ds_store_b16_d16_hi v11, v97 offset:7360               // local write
	ds_store_b16 v11, v98 offset:7680                      // local write
	ds_store_b16_d16_hi v11, v98 offset:8000               // local write
	ds_store_b16 v11, v99 offset:8320                      // local write
	ds_store_b16_d16_hi v11, v99 offset:8640               // local write
	ds_store_b16 v11, v100 offset:8960                     // local write
	ds_store_b16_d16_hi v11, v100 offset:9280              // local write
	ds_store_b16 v11, v101 offset:9600                     // local write
	ds_store_b16_d16_hi v11, v101 offset:9920              // local write
	v_xor_b32_e32 v10, 0x4000, v10                         // LDS double-buffer swap
	v_xor_b32_e32 v11, 0x4000, v11                         // LDS double-buffer swap
	s_cmp_eq_u32 s12, 1
	s_cbranch_scc1 label_MainLoop_Entry                    // branch to MainLoop_Entry
	buffer_load_b128 v[82:85], v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen // G -> Reg A
	buffer_load_d16_b16 v86, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], 0 offen // G -> Reg B
	buffer_load_d16_hi_b16 v86, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s66 offen // G -> Reg B
	buffer_load_d16_b16 v87, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s67 offen // G -> Reg B
	buffer_load_d16_hi_b16 v87, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s68 offen // G -> Reg B
	buffer_load_d16_b16 v88, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s69 offen // G -> Reg B
	buffer_load_d16_hi_b16 v88, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s70 offen // G -> Reg B
	buffer_load_d16_b16 v89, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s71 offen // G -> Reg B
	buffer_load_d16_hi_b16 v89, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s72 offen // G -> Reg B
	buffer_load_d16_b16 v90, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s73 offen // G -> Reg B
	buffer_load_d16_hi_b16 v90, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s74 offen // G -> Reg B
	buffer_load_d16_b16 v91, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s75 offen // G -> Reg B
	buffer_load_d16_hi_b16 v91, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s76 offen // G -> Reg B
	buffer_load_d16_b16 v92, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s77 offen // G -> Reg B
	buffer_load_d16_hi_b16 v92, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s78 offen // G -> Reg B
	buffer_load_d16_b16 v93, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s79 offen // G -> Reg B
	buffer_load_d16_hi_b16 v93, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s80 offen // G -> Reg B
	buffer_load_d16_b16 v94, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s81 offen // G -> Reg B
	buffer_load_d16_hi_b16 v94, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s82 offen // G -> Reg B
	buffer_load_d16_b16 v95, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s83 offen // G -> Reg B
	buffer_load_d16_hi_b16 v95, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s84 offen // G -> Reg B
	buffer_load_d16_b16 v96, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s85 offen // G -> Reg B
	buffer_load_d16_hi_b16 v96, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s86 offen // G -> Reg B
	buffer_load_d16_b16 v97, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s87 offen // G -> Reg B
	buffer_load_d16_hi_b16 v97, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s88 offen // G -> Reg B
	buffer_load_d16_b16 v98, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s89 offen // G -> Reg B
	buffer_load_d16_hi_b16 v98, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s90 offen // G -> Reg B
	buffer_load_d16_b16 v99, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s91 offen // G -> Reg B
	buffer_load_d16_hi_b16 v99, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s92 offen // G -> Reg B
	buffer_load_d16_b16 v100, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s93 offen // G -> Reg B
	buffer_load_d16_hi_b16 v100, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s94 offen // G -> Reg B
	buffer_load_d16_b16 v101, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s95 offen // G -> Reg B
	buffer_load_d16_hi_b16 v101, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s96 offen // G -> Reg B

/******************************************/
/* Main Loop                               */
/******************************************/

label_MainLoop_Entry:
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	ds_load_b128 v[16:19], v[vgprLocalWriteAddrA]          // L -> Reg
	ds_load_b128 v[20:23], v[vgprLocalWriteAddrA] offset:16 // L -> Reg
	ds_load_b128 v[49:52], v[vgprLocalWriteAddrB]          // L -> Reg
	ds_load_b128 v[53:56], v[vgprLocalWriteAddrB] offset:16 // L -> Reg
	s_cmp_eq_u32 s12, 1
	s_cbranch_scc1 label_MainLoop_TailEnd                  // branch to MainLoop_TailEnd
	s_cmp_le_u32 s12, 2
	s_cbranch_scc1 label_MainLoop_Tail                     // branch to MainLoop_Tail
/* Main Loop body start */
label_MainLoop_Start:
	ds_load_b128 v[24:27], v[vgprLocalWriteAddrA] offset:32 // L -> Reg
	ds_load_b128 v[28:31], v[vgprLocalWriteAddrA] offset:48 // L -> Reg
	ds_load_b128 v[57:60], v[vgprLocalWriteAddrB] offset:32 // L -> Reg
	ds_load_b128 v[61:64], v[vgprLocalWriteAddrB] offset:48 // L -> Reg
	s_cmp_eq_u32 s12, s47
	s_cselect_b32 s98, s60, s64                            // select A stride
	s_cselect_b32 s99, s61, 0                              // select A stride high
	s_add_u32 s[sgprSrdA], s[sgprSrdA], s98                // adjust SRD A
	s_addc_u32 s49, s49, s99                               // add with carry
	s_sub_u32 s56, s56, s98
	s_subb_u32 s57, s57, s99                               // sub with borrow
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1                             // SRD A num records
	s_cmp_eq_u32 s12, s47
	s_cselect_b32 s98, s62, s65                            // select B stride
	s_cselect_b32 s99, s63, 0                              // select B stride high
	s_add_u32 s[sgprSrdB], s[sgprSrdB], s98                // adjust SRD B
	s_addc_u32 s53, s53, s99                               // add with carry
	s_sub_u32 s58, s58, s98
	s_subb_u32 s59, s59, s99                               // sub with borrow
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1                             // SRD B num records
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b128 v10, v[82:85]                            // local write
	buffer_load_b128 v[82:85], v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen // G -> Reg A
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v86                                  // local write
	buffer_load_d16_b16 v86, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], 0 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v86 offset:320                // local write
	buffer_load_d16_hi_b16 v86, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s66 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v87 offset:640                       // local write
	buffer_load_d16_b16 v87, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s67 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v87 offset:960                // local write
	buffer_load_d16_hi_b16 v87, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s68 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v88 offset:1280                      // local write
	buffer_load_d16_b16 v88, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s69 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v88 offset:1600               // local write
	buffer_load_d16_hi_b16 v88, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s70 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v89 offset:1920                      // local write
	buffer_load_d16_b16 v89, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s71 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v89 offset:2240               // local write
	buffer_load_d16_hi_b16 v89, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s72 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v90 offset:2560                      // local write
	buffer_load_d16_b16 v90, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s73 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v90 offset:2880               // local write
	buffer_load_d16_hi_b16 v90, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s74 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v91 offset:3200                      // local write
	buffer_load_d16_b16 v91, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s75 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v91 offset:3520               // local write
	buffer_load_d16_hi_b16 v91, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s76 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v92 offset:3840                      // local write
	buffer_load_d16_b16 v92, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s77 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v92 offset:4160               // local write
	buffer_load_d16_hi_b16 v92, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s78 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v93 offset:4480                      // local write
	buffer_load_d16_b16 v93, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s79 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v93 offset:4800               // local write
	buffer_load_d16_hi_b16 v93, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s80 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v94 offset:5120                      // local write
	buffer_load_d16_b16 v94, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s81 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v94 offset:5440               // local write
	buffer_load_d16_hi_b16 v94, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s82 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v95 offset:5760                      // local write
	buffer_load_d16_b16 v95, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s83 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v95 offset:6080               // local write
	buffer_load_d16_hi_b16 v95, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s84 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v96 offset:6400                      // local write
	buffer_load_d16_b16 v96, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s85 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v96 offset:6720               // local write
	buffer_load_d16_hi_b16 v96, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s86 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v97 offset:7040                      // local write
	buffer_load_d16_b16 v97, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s87 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v97 offset:7360               // local write
	buffer_load_d16_hi_b16 v97, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s88 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v98 offset:7680                      // local write
	buffer_load_d16_b16 v98, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s89 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v98 offset:8000               // local write
	buffer_load_d16_hi_b16 v98, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s90 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v99 offset:8320                      // local write
	buffer_load_d16_b16 v99, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s91 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v99 offset:8640               // local write
	buffer_load_d16_hi_b16 v99, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s92 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v100 offset:8960                     // local write
	buffer_load_d16_b16 v100, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s93 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v100 offset:9280              // local write
	buffer_load_d16_hi_b16 v100, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s94 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16 v11, v101 offset:9600                     // local write
	buffer_load_d16_b16 v101, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s95 offen // G -> Reg B
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b16_d16_hi v11, v101 offset:9920              // local write
	buffer_load_d16_hi_b16 v101, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s96 offen // G -> Reg B
	v_xor_b32_e32 v10, 0x4000, v10                         // LDS double-buffer swap
	v_xor_b32_e32 v11, 0x4000, v11                         // LDS double-buffer swap
	s_waitcnt lgkmcnt(37)                                  // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[49:56], v[16:23], v[vgprValuC:vgprValuC+7]
	ds_load_b128 v[32:35], v[vgprLocalWriteAddrA] offset:64 // L -> Reg
	ds_load_b128 v[36:39], v[vgprLocalWriteAddrA] offset:80 // L -> Reg
	ds_load_b128 v[65:68], v[vgprLocalWriteAddrB] offset:64 // L -> Reg
	ds_load_b128 v[69:72], v[vgprLocalWriteAddrB] offset:80 // L -> Reg
	s_waitcnt lgkmcnt(37)                                  // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[57:64], v[24:31], v[vgprValuC:vgprValuC+7]
	ds_load_b128 v[40:43], v[vgprLocalWriteAddrA] offset:96 // L -> Reg
	ds_load_b128 v[44:47], v[vgprLocalWriteAddrA] offset:112 // L -> Reg
	ds_load_b128 v[73:76], v[vgprLocalWriteAddrB] offset:96 // L -> Reg
	ds_load_b128 v[77:80], v[vgprLocalWriteAddrB] offset:112 // L -> Reg
	v_xor_b32_e32 v[vgprLocalWriteAddrA], 0x4000, v[vgprLocalWriteAddrA] // LDS double-buffer swap
	v_xor_b32_e32 v[vgprLocalWriteAddrB], 0x4000, v[vgprLocalWriteAddrB] // LDS double-buffer swap
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[65:72], v[32:39], v[vgprValuC:vgprValuC+7]
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	ds_load_b128 v[16:19], v[vgprLocalWriteAddrA]          // L -> Reg
	ds_load_b128 v[20:23], v[vgprLocalWriteAddrA] offset:16 // L -> Reg
	ds_load_b128 v[49:52], v[vgprLocalWriteAddrB]          // L -> Reg
	ds_load_b128 v[53:56], v[vgprLocalWriteAddrB] offset:16 // L -> Reg
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[73:80], v[40:47], v[vgprValuC:vgprValuC+7]
	s_sub_u32 s12, s12, 1
	s_cmp_eq_i32 s12, 2
	s_cbranch_scc0 label_MainLoop_Start                    // branch to MainLoop_Start

/******************************************/
/* Main Loop: Tail Iterations              */
/******************************************/

label_MainLoop_Tail:
	ds_load_b128 v[24:27], v[vgprLocalWriteAddrA] offset:32 // L -> Reg
	ds_load_b128 v[28:31], v[vgprLocalWriteAddrA] offset:48 // L -> Reg
	ds_load_b128 v[57:60], v[vgprLocalWriteAddrB] offset:32 // L -> Reg
	ds_load_b128 v[61:64], v[vgprLocalWriteAddrB] offset:48 // L -> Reg
	s_cmp_eq_u32 s12, s47
	s_cselect_b32 s98, s60, s64                            // select A stride
	s_cselect_b32 s99, s61, 0                              // select A stride high
	s_add_u32 s[sgprSrdA], s[sgprSrdA], s98                // adjust SRD A
	s_addc_u32 s49, s49, s99                               // add with carry
	s_sub_u32 s56, s56, s98
	s_subb_u32 s57, s57, s99                               // sub with borrow
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1                             // SRD A num records
	s_cmp_eq_u32 s12, s47
	s_cselect_b32 s98, s62, s65                            // select B stride
	s_cselect_b32 s99, s63, 0                              // select B stride high
	s_add_u32 s[sgprSrdB], s[sgprSrdB], s98                // adjust SRD B
	s_addc_u32 s53, s53, s99                               // add with carry
	s_sub_u32 s58, s58, s98
	s_subb_u32 s59, s59, s99                               // sub with borrow
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1                             // SRD B num records
	s_waitcnt vmcnt(32)                                    // wait for global read
	ds_store_b128 v10, v[82:85]                            // local write
	s_waitcnt vmcnt(31)                                    // wait for global read
	ds_store_b16 v11, v86                                  // local write
	s_waitcnt vmcnt(30)                                    // wait for global read
	ds_store_b16_d16_hi v11, v86 offset:320                // local write
	s_waitcnt vmcnt(29)                                    // wait for global read
	ds_store_b16 v11, v87 offset:640                       // local write
	s_waitcnt vmcnt(28)                                    // wait for global read
	ds_store_b16_d16_hi v11, v87 offset:960                // local write
	s_waitcnt vmcnt(27)                                    // wait for global read
	ds_store_b16 v11, v88 offset:1280                      // local write
	s_waitcnt vmcnt(26)                                    // wait for global read
	ds_store_b16_d16_hi v11, v88 offset:1600               // local write
	s_waitcnt vmcnt(25)                                    // wait for global read
	ds_store_b16 v11, v89 offset:1920                      // local write
	s_waitcnt vmcnt(24)                                    // wait for global read
	ds_store_b16_d16_hi v11, v89 offset:2240               // local write
	s_waitcnt vmcnt(23)                                    // wait for global read
	ds_store_b16 v11, v90 offset:2560                      // local write
	s_waitcnt vmcnt(22)                                    // wait for global read
	ds_store_b16_d16_hi v11, v90 offset:2880               // local write
	s_waitcnt vmcnt(21)                                    // wait for global read
	ds_store_b16 v11, v91 offset:3200                      // local write
	s_waitcnt vmcnt(20)                                    // wait for global read
	ds_store_b16_d16_hi v11, v91 offset:3520               // local write
	s_waitcnt vmcnt(19)                                    // wait for global read
	ds_store_b16 v11, v92 offset:3840                      // local write
	s_waitcnt vmcnt(18)                                    // wait for global read
	ds_store_b16_d16_hi v11, v92 offset:4160               // local write
	s_waitcnt vmcnt(17)                                    // wait for global read
	ds_store_b16 v11, v93 offset:4480                      // local write
	s_waitcnt vmcnt(16)                                    // wait for global read
	ds_store_b16_d16_hi v11, v93 offset:4800               // local write
	s_waitcnt vmcnt(15)                                    // wait for global read
	ds_store_b16 v11, v94 offset:5120                      // local write
	s_waitcnt vmcnt(14)                                    // wait for global read
	ds_store_b16_d16_hi v11, v94 offset:5440               // local write
	s_waitcnt vmcnt(13)                                    // wait for global read
	ds_store_b16 v11, v95 offset:5760                      // local write
	s_waitcnt vmcnt(12)                                    // wait for global read
	ds_store_b16_d16_hi v11, v95 offset:6080               // local write
	s_waitcnt vmcnt(11)                                    // wait for global read
	ds_store_b16 v11, v96 offset:6400                      // local write
	s_waitcnt vmcnt(10)                                    // wait for global read
	ds_store_b16_d16_hi v11, v96 offset:6720               // local write
	s_waitcnt vmcnt(9)                                     // wait for global read
	ds_store_b16 v11, v97 offset:7040                      // local write
	s_waitcnt vmcnt(8)                                     // wait for global read
	ds_store_b16_d16_hi v11, v97 offset:7360               // local write
	s_waitcnt vmcnt(7)                                     // wait for global read
	ds_store_b16 v11, v98 offset:7680                      // local write
	s_waitcnt vmcnt(6)                                     // wait for global read
	ds_store_b16_d16_hi v11, v98 offset:8000               // local write
	s_waitcnt vmcnt(5)                                     // wait for global read
	ds_store_b16 v11, v99 offset:8320                      // local write
	s_waitcnt vmcnt(4)                                     // wait for global read
	ds_store_b16_d16_hi v11, v99 offset:8640               // local write
	s_waitcnt vmcnt(3)                                     // wait for global read
	ds_store_b16 v11, v100 offset:8960                     // local write
	s_waitcnt vmcnt(2)                                     // wait for global read
	ds_store_b16_d16_hi v11, v100 offset:9280              // local write
	s_waitcnt vmcnt(1)                                     // wait for global read
	ds_store_b16 v11, v101 offset:9600                     // local write
	s_waitcnt vmcnt(0)                                     // wait for global read
	ds_store_b16_d16_hi v11, v101 offset:9920              // local write
	v_xor_b32_e32 v10, 0x4000, v10                         // LDS double-buffer swap
	v_xor_b32_e32 v11, 0x4000, v11                         // LDS double-buffer swap
	s_waitcnt lgkmcnt(37)                                  // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[49:56], v[16:23], v[vgprValuC:vgprValuC+7]
	ds_load_b128 v[32:35], v[vgprLocalWriteAddrA] offset:64 // L -> Reg
	ds_load_b128 v[36:39], v[vgprLocalWriteAddrA] offset:80 // L -> Reg
	ds_load_b128 v[65:68], v[vgprLocalWriteAddrB] offset:64 // L -> Reg
	ds_load_b128 v[69:72], v[vgprLocalWriteAddrB] offset:80 // L -> Reg
	s_waitcnt lgkmcnt(37)                                  // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[57:64], v[24:31], v[vgprValuC:vgprValuC+7]
	ds_load_b128 v[40:43], v[vgprLocalWriteAddrA] offset:96 // L -> Reg
	ds_load_b128 v[44:47], v[vgprLocalWriteAddrA] offset:112 // L -> Reg
	ds_load_b128 v[73:76], v[vgprLocalWriteAddrB] offset:96 // L -> Reg
	ds_load_b128 v[77:80], v[vgprLocalWriteAddrB] offset:112 // L -> Reg
	v_xor_b32_e32 v[vgprLocalWriteAddrA], 0x4000, v[vgprLocalWriteAddrA] // LDS double-buffer swap
	v_xor_b32_e32 v[vgprLocalWriteAddrB], 0x4000, v[vgprLocalWriteAddrB] // LDS double-buffer swap
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[65:72], v[32:39], v[vgprValuC:vgprValuC+7]
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	ds_load_b128 v[16:19], v[vgprLocalWriteAddrA]          // L -> Reg
	ds_load_b128 v[20:23], v[vgprLocalWriteAddrA] offset:16 // L -> Reg
	ds_load_b128 v[49:52], v[vgprLocalWriteAddrB]          // L -> Reg
	ds_load_b128 v[53:56], v[vgprLocalWriteAddrB] offset:16 // L -> Reg
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[73:80], v[40:47], v[vgprValuC:vgprValuC+7]

/******************************************/
/* Global Store                            */
/******************************************/

label_MainLoop_TailEnd:
	s_and_b32 s[sgprLoopCounterL], s[sgprWGMInfo], 0x3fff  // get GSU
	s_cmp_eq_u32 s[sgprLoopCounterL], 1
	s_cbranch_scc0 label_EndPgm                            // branch to EndPgm
	s_cmpk_eq_u32 s[sgprBeta], 0x0                         // beta == 0 ?
	s_cbranch_scc0 label_EndPgm                            // branch to EndPgm
	s_cmp_eq_u32 s[sgprAlpha], 1.0                         // alpha == 1.0 ?
	s_cbranch_scc0 label_EndPgm                            // branch to EndPgm
	s_and_b32 s98, 15, s[sgprSizesFree0]                   // SizesFree0 mod MT0
	s_add_u32 s99, -1, s[sgprNumWorkGroups0]
	s_cmp_ge_u32 s[sgprWorkGroup0], s99
	s_cselect_b32 s98, s98, 0                              // zero if not edge WG
	s_cmpk_gt_u32 s98, 0x0                                 // compare greater
	s_cbranch_scc1 label_EndPgm                            // branch to EndPgm
	s_and_b32 s98, 63, s[sgprSizesFree1]                   // SizesFree1 mod MT1
	s_add_u32 s99, -1, s[sgprNumWorkGroups1]
	s_cmp_ge_u32 s[sgprWorkGroup1], s99
	s_cselect_b32 s98, s98, 0                              // zero if not edge WG
	s_cmpk_gt_u32 s98, 0x0                                 // compare greater
	s_cbranch_scc1 label_EndPgm                            // branch to EndPgm
	s_and_b32 s99, 63, s[sgprSizesSum0]                    // SizesSum0 mod DepthU
	s_cmp_eq_u32 s99, 0
	s_cbranch_scc0 label_EndPgm                            // branch to EndPgm
	ds_load_b128 v[24:27], v[vgprLocalWriteAddrA] offset:32 // L -> Reg
	ds_load_b128 v[28:31], v[vgprLocalWriteAddrA] offset:48 // L -> Reg
	ds_load_b128 v[57:60], v[vgprLocalWriteAddrB] offset:32 // L -> Reg
	ds_load_b128 v[61:64], v[vgprLocalWriteAddrB] offset:48 // L -> Reg
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[49:56], v[16:23], v[vgprValuC:vgprValuC+7]
	ds_load_b128 v[32:35], v[vgprLocalWriteAddrA] offset:64 // L -> Reg
	ds_load_b128 v[36:39], v[vgprLocalWriteAddrA] offset:80 // L -> Reg
	ds_load_b128 v[65:68], v[vgprLocalWriteAddrB] offset:64 // L -> Reg
	ds_load_b128 v[69:72], v[vgprLocalWriteAddrB] offset:80 // L -> Reg
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[57:64], v[24:31], v[vgprValuC:vgprValuC+7]
	ds_load_b128 v[40:43], v[vgprLocalWriteAddrA] offset:96 // L -> Reg
	ds_load_b128 v[44:47], v[vgprLocalWriteAddrA] offset:112 // L -> Reg
	ds_load_b128 v[73:76], v[vgprLocalWriteAddrB] offset:96 // L -> Reg
	ds_load_b128 v[77:80], v[vgprLocalWriteAddrB] offset:112 // L -> Reg
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[65:72], v[32:39], v[vgprValuC:vgprValuC+7]
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[73:80], v[40:47], v[vgprValuC:vgprValuC+7]
	s_cmp_eq_u32 s[sgprArgType], 2                         // ArgType == 2 ?
	s_cbranch_scc1 label_GS_BetaCheck                      // branch to GS_BetaCheck
	s_load_b256 s[sgprSrdA:sgprSrdA+7], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 // load store args
	s_load_b32 s56, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78 // load store args
	s_branch label_GS_Start                                // jump to GS_Start
label_GS_BetaCheck:
	s_load_b128 s[sgprSrdA:sgprSrdA+3], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // load store args
	s_load_b64 s[sgprSrdB:sgprSrdB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa0 // load store args
	s_load_b64 s[54:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb8 // load store args
	s_load_b32 s56, s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc0 // load store args
label_GS_Start:
	v_lshrrev_b32_e32 v[vgprLocalWriteAddrA], 5, v[vgprSerial]
	v_lshrrev_b32_e32 v[vgprLocalWriteAddrB], 0, v[vgprLocalWriteAddrA]
	v_mul_lo_u32 v[vgprLocalWriteAddrB], 16, v[vgprLocalWriteAddrB]
	v_and_b32_e32 v[vgprGlobalReadOffsetB], 31, v[vgprSerial]
	v_lshrrev_b32_e32 v[vgprGlobalReadOffsetB], 4, v[vgprGlobalReadOffsetB]
	v_add_lshl_u32 v[vgprGlobalReadOffsetB], v[vgprLocalWriteAddrB], v[vgprGlobalReadOffsetB], 0
	v_mul_lo_u32 v10, v[vgprGlobalReadOffsetB], s[sgprStrideC0] // offset * strideC0
	v_mul_lo_u32 v11, v[vgprGlobalReadOffsetB], s[sgprStrideD0] // offset * strideD0
	v_and_b32_e32 v[vgprGlobalReadOffsetA], 0, v[vgprLocalWriteAddrA]
	v_mul_lo_u32 v[vgprGlobalReadOffsetA], 16, v[vgprGlobalReadOffsetA]
	v_and_b32_e32 v[vgprLocalWriteAddrB], 15, v[vgprSerial]
	v_add_lshl_u32 v[vgprGlobalReadOffsetA], v[vgprLocalWriteAddrB], v[vgprGlobalReadOffsetA], 0
	s_mul_i32 s[sgprLoopCounterL], 16, s[sgprWorkGroup0]   // tile offset
	v_add_nc_u32_e32 v[vgprGlobalReadOffsetA], s[sgprLoopCounterL], v[vgprGlobalReadOffsetA]
	s_mul_i32 s[sgprLoopCounterL], 64, s[sgprWorkGroup1]   // tile offset
	v_add_nc_u32_e32 v[vgprGlobalReadOffsetB], s[sgprLoopCounterL], v[vgprGlobalReadOffsetB]
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_mov_b64 s[sgprAddressA:sgprAddressA+1], s[sgprSrdA:sgprSrdA+1]
	s_mov_b32 s35, 0x31004000                              // SRD Dword3
	s_cmp_eq_u64 s[sgprSrdA:sgprSrdA+1], 0                 // compare 64-bit
	s_cbranch_scc0 label_GS_StrideD_NonUnit                // branch to GS_StrideD_NonUnit
	s_mov_b32 s[sgprAddressB], 0
	s_branch label_GS_StrideD_Done                         // jump to GS_StrideD_Done
label_GS_StrideD_NonUnit:
	s_mov_b32 s[sgprAddressB], s[sgprSizesFree0]
label_GS_StrideD_Done:
	s_mul_i32 s[sgprAddressB], 4, s[sgprAddressB]
	s_add_u32 s[sgprLoopCounterL], s[sgprWorkGroup2], 1
	s_mul_i32 s[sgprLoopCounterL], s53, s[sgprLoopCounterL]
	s_cmp_eq_u32 s[sgprLoopCounterL], 0
	s_cselect_b32 s[sgprLoopCounterL], s[sgprSizesFree0], s[sgprLoopCounterL] // select full tile if partial
	s_mov_b64 s[sgprStrideA0:sgprStrideA0+1], s[50:51]
	s_mov_b32 s[sgprStrideB1], 0x31004000                  // SRD Dword3
	s_cmp_eq_u64 s[50:51], 0                               // compare 64-bit
	s_cbranch_scc0 label_GS_StrideC_NonUnit                // branch to GS_StrideC_NonUnit
	s_mov_b32 s[sgprStrideB0], 0
	s_branch label_GS_StrideC_Done                         // jump to GS_StrideC_Done
label_GS_StrideC_NonUnit:
	s_mov_b32 s[sgprStrideB0], s[sgprLoopCounterL]
label_GS_StrideC_Done:
	s_cmpk_lg_u32 s[sgprSrdB], 0x0                         // compare not equal
	s_cbranch_scc1 label_GS_NoBias                         // branch to GS_NoBias
	s_mul_i32 s[sgprLoopCounterL], 16, s[sgprWorkGroup0]   // tile offset
	v_add_nc_u32_e32 v16, s[sgprLoopCounterL], v[vgprSerial]
	s_mul_i32 s[sgprStrideB0], 4, s[sgprStrideB0]          // Bias num_records *= sizeof(f32)
	s_mul_i32 s[sgprLoopCounterL], s53, s[sgprWorkGroup2]  // tile offset
	v_add_nc_u32_e32 v14, s[sgprLoopCounterL], v16
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v15, 2, v16
	s_mul_i32 s[sgprLoopCounterL], 64, s[sgprWorkGroup1]   // tile offset
	v_add_nc_u32_e32 v16, s[sgprLoopCounterL], v[vgprSerial]
	buffer_load_b32 v[vgprLocalWriteAddrA], v14, s[sgprStrideA0:sgprStrideA0+3], 0 offen // load bias
	buffer_load_b32 v[vgprLocalWriteAddrB], v15, s[sgprAddressA:sgprAddressA+3], 0 offen // load ScaleAlphaVec
	v_lshlrev_b32_e32 v16, 2, v[vgprSerial]
	s_barrier                                              // LDS barrier
	s_waitcnt vmcnt(1)                                     // wait for global read
	ds_store_b32 v16, v[vgprLocalWriteAddrA]               // local write
	v_cmp_gt_u32_e64 s[sgprSrdA], s[sgprAddressB], 0
	s_waitcnt vmcnt(0)                                     // wait for global read
	v_cndmask_b32_e64 v[vgprLocalWriteAddrB], 1.0, v[vgprLocalWriteAddrB], s[sgprSrdA] // use 1.0 if no ScaleAlphaVec
	ds_store_b32 v16, v[vgprLocalWriteAddrB] offset:512    // local write
	s_branch label_GS_BiasEnd                              // jump to GS_BiasEnd
label_GS_NoBias:
	s_cmpk_lg_u32 s[sgprSrdB], 0x4                         // compare not equal
	s_cbranch_scc1 label_GS_BiasEnd                        // branch to GS_BiasEnd
	s_mul_i32 s[sgprLoopCounterL], 16, s[sgprWorkGroup0]   // tile offset
	v_add_nc_u32_e32 v16, s[sgprLoopCounterL], v[vgprSerial]
	s_mul_i32 s[sgprStrideB0], 2, s[sgprStrideB0]          // Bias num_records *= sizeof(f16)
	s_mul_i32 s[sgprLoopCounterL], s53, s[sgprWorkGroup2]  // tile offset
	v_add_nc_u32_e32 v14, s[sgprLoopCounterL], v16
	v_lshlrev_b32_e32 v14, 1, v14
	v_lshlrev_b32_e32 v15, 2, v16
	s_mul_i32 s[sgprLoopCounterL], 64, s[sgprWorkGroup1]   // tile offset
	v_add_nc_u32_e32 v16, s[sgprLoopCounterL], v[vgprSerial]
	buffer_load_d16_b16 v[vgprLocalWriteAddrA], v14, s[sgprStrideA0:sgprStrideA0+3], 0 offen // load bias (f16)
	buffer_load_b32 v[vgprLocalWriteAddrB], v15, s[sgprAddressA:sgprAddressA+3], 0 offen // load ScaleAlphaVec
	v_lshlrev_b32_e32 v16, 2, v[vgprSerial]
	s_barrier                                              // LDS barrier
	s_waitcnt vmcnt(1)                                     // wait for global read
	v_cvt_f32_f16_e32 v[vgprLocalWriteAddrA], v12.l        // convert to f32
	ds_store_b32 v16, v[vgprLocalWriteAddrA]               // local write
	v_cmp_gt_u32_e64 s[sgprSrdA], s[sgprAddressB], 0
	s_waitcnt vmcnt(0)                                     // wait for global read
	v_cndmask_b32_e64 v[vgprLocalWriteAddrB], 1.0, v[vgprLocalWriteAddrB], s[sgprSrdA] // use 1.0 if no ScaleAlphaVec
	ds_store_b32 v16, v[vgprLocalWriteAddrB] offset:512    // local write
	s_branch label_GS_BiasEnd                              // jump to GS_BiasEnd
label_GS_BiasEnd:
	s_cmpk_eq_u32 s56, 0x3                                 // check activation type
	s_cbranch_scc1 label_ActGELU                           // branch to ActGELU
	s_cmpk_eq_u32 s56, 0x5                                 // check activation type
	s_cbranch_scc1 label_ActRELU                           // branch to ActRELU
	s_cmpk_eq_u32 s56, 0x6                                 // check activation type
	s_cbranch_scc1 label_ActSigmoid                        // branch to ActSigmoid
	s_cmpk_eq_u32 s56, 0xa                                 // check activation type
	s_cbranch_scc1 label_ActSilu                           // branch to ActSilu
	s_cmpk_eq_u32 s56, 0xc                                 // check activation type
	s_cbranch_scc1 label_ActClip                           // branch to ActClip
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0x1b40, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_ActSet                                  // jump to ActSet
label_ActGELU:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0x1b2c, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_ActSet                                  // jump to ActSet
label_ActRELU:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0x1b50, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_ActSet                                  // jump to ActSet
label_ActSigmoid:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0x1b44, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_ActSet                                  // jump to ActSet
label_ActSilu:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0x1b44, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_ActSet                                  // jump to ActSet
label_ActClip:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0x1b48, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_ActSet                                  // jump to ActSet
label_ActSet:
	s_mul_i32 s[sgprLoopCounterL], 16, s[sgprWorkGroup0]   // tile offset
	v_sub_nc_u32_e64 v17, v[vgprGlobalReadOffsetA], s[sgprLoopCounterL]
	v_lshlrev_b32_e32 v17, 2, v17
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	ds_load_b32 v26, v17                                   // L -> Reg
	ds_load_b32 v27, v17 offset:512                        // L -> Reg
	v_add_lshl_u32 v15, v11, v[vgprGlobalReadOffsetA], 1
	v_mov_b32_e32 v18, v[vgprValuC]
	v_mov_b32_e32 v19, v1
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v3
	v_mov_b32_e32 v22, v4
	v_mov_b32_e32 v23, v5
	v_mov_b32_e32 v24, v6
	v_mov_b32_e32 v25, v7
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	v_mul_f32_e32 v18, v27, v18                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v18         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v18, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v18.l, v18                           // convert to f16
	buffer_store_b16 v18, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v19, v27, v19                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v19         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v19, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v19.l, v19                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v19, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v20, v27, v20                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v20         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v20, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v20.l, v20                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v20, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v21, v27, v21                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v21         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v21, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v21.l, v21                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v21, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v22, v27, v22                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v22         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v22, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v22.l, v22                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v22, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v23, v27, v23                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v23         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v23, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v23.l, v23                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v23, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v24, v27, v24                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v24         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v24, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v24.l, v24                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v24, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v25, v27, v25                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v25         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v25, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v25.l, v25                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v25, v15, s[16:19], 0 offen           // store D
	s_nop 0                                                // 1 wait states
	s_branch label_EndPgm_SG                               // jump to EndPgm_SG
label_EndPgm_SG:
	s_endpgm                                               // Kernel End
label_EndPgm:
	ds_load_b128 v[24:27], v[vgprLocalWriteAddrA] offset:32 // L -> Reg
	ds_load_b128 v[28:31], v[vgprLocalWriteAddrA] offset:48 // L -> Reg
	ds_load_b128 v[57:60], v[vgprLocalWriteAddrB] offset:32 // L -> Reg
	ds_load_b128 v[61:64], v[vgprLocalWriteAddrB] offset:48 // L -> Reg
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[49:56], v[16:23], v[vgprValuC:vgprValuC+7]
	ds_load_b128 v[32:35], v[vgprLocalWriteAddrA] offset:64 // L -> Reg
	ds_load_b128 v[36:39], v[vgprLocalWriteAddrA] offset:80 // L -> Reg
	ds_load_b128 v[65:68], v[vgprLocalWriteAddrB] offset:64 // L -> Reg
	ds_load_b128 v[69:72], v[vgprLocalWriteAddrB] offset:80 // L -> Reg
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[57:64], v[24:31], v[vgprValuC:vgprValuC+7]
	ds_load_b128 v[40:43], v[vgprLocalWriteAddrA] offset:96 // L -> Reg
	ds_load_b128 v[44:47], v[vgprLocalWriteAddrA] offset:112 // L -> Reg
	ds_load_b128 v[73:76], v[vgprLocalWriteAddrB] offset:96 // L -> Reg
	ds_load_b128 v[77:80], v[vgprLocalWriteAddrB] offset:112 // L -> Reg
	s_waitcnt lgkmcnt(4)                                   // wait for LDS read
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[65:72], v[32:39], v[vgprValuC:vgprValuC+7]
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[73:80], v[40:47], v[vgprValuC:vgprValuC+7]
	v_and_b32_e32 v10, 0xf03fff, v10                       // reset LDS double-buffer
	v_and_b32_e32 v11, 0xf03fff, v11                       // reset LDS double-buffer
	s_and_b32 s12, 63, s[sgprSizesSum0]                    // SizesSum0 mod DepthU
	s_and_b32 s98, s[sgprWGMInfo], 0x8000                  // get stagger flag
	s_cbranch_scc1 label_GS_MG_BetaNonZero                 // branch to GS_MG_BetaNonZero
	s_cmp_lg_u32 s[sgprStaggerUIter], s[sgprStaggerUStride]
	s_cmov_b32 s12, 0                                      // clear if stagger done
	s_branch label_GS_MG_Start                             // jump to GS_MG_Start
label_GS_MG_BetaNonZero:
	s_lshr_b32 s99, s[sgprSizesSum0], 6
	s_and_b32 s100, s[sgprWGMInfo], 0x3fff                 // get GSU
	v_cvt_f32_u32_e32 v36, s100                            // convert to f32
	v_rcp_iflag_f32_e32 v36, v36                           // 1/x (for integer div)
	v_cvt_f32_u32_e32 v37, s99                             // convert to f32
	v_mul_f32_e32 v36, v36, v37
	v_cvt_u32_f32_e32 v36, v36                             // convert to u32
	v_mul_u32_u24_e64 v37, v36, s100
	v_sub_nc_u32_e32 v37, s99, v37
	v_cmp_eq_u32_e64 vcc_lo, v37, s100
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_add_nc_u32_e32 v36, 1, v36
	v_mov_b32_e32 v37, 0
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_cmp_gt_u32_e64 vcc_lo, v37, s100
	s_mov_b32 exec_lo, vcc_lo                              // mask active lanes
	v_sub_nc_u32_e64 v36, v36, 1
	v_mul_u32_u24_e64 v37, v36, s100
	v_sub_nc_u32_e32 v37, s99, v37
	s_mov_b32 exec_lo, -1                                  // Reset exec
	v_readfirstlane_b32 s98, v36                           // quotient
	v_readfirstlane_b32 s[sgprStaggerUStride], v37         // remainder
	s_sub_u32 s99, s100, 1
	s_cmp_eq_u32 s98, 0
	s_cselect_b32 s98, s[sgprStaggerUStride], s99          // select stagger bound
	s_cmp_lg_u32 s[sgprStaggerUIter], s98
	s_cmov_b32 s12, 0                                      // clear if stagger done
label_GS_MG_Start:
	s_cmp_eq_u32 s12, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 label_EndPgm_MG                         // branch to EndPgm_MG
	s_sub_i32 s98, 3, s47
	s_cmp_ge_i32 s98, 0
	s_cbranch_scc0 label_GS_MG_StrideD_Calc                // branch to GS_MG_StrideD_Calc
	s_mul_hi_u32 s99, s98, s64                             // high 32 bits of multiply
	s_mul_i32 s98, s98, s64
	s_branch label_GS_MG_StrideD_Done                      // jump to GS_MG_StrideD_Done
label_GS_MG_StrideD_Calc:
	s_abs_i32 s98, s98                                     // absolute value
	s_mul_hi_u32 s99, s98, s64                             // high 32 bits of multiply
	s_mul_i32 s98, s98, s64
	s_xor_b32 s98, s98, -1                                 // bitwise negate
	s_xor_b32 s99, s99, -1                                 // bitwise negate
	s_add_u32 s98, s98, 1
	s_addc_u32 s99, s99, 0                                 // carry from above
label_GS_MG_StrideD_Done:
	s_sub_u32 s98, s98, s60
	s_subb_u32 s99, s99, s61                               // sub with borrow
	s_add_u32 s[sgprSrdA], s[sgprSrdA], s98                // adjust SRD A
	s_addc_u32 s49, s49, s99                               // add with carry
	s_sub_u32 s56, s56, s98
	s_subb_u32 s57, s57, s99                               // sub with borrow
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1                             // SRD A num records
	s_sub_i32 s98, 3, s47
	s_cmp_ge_i32 s98, 0
	s_cbranch_scc0 label_GS_MG_StrideC_Calc                // branch to GS_MG_StrideC_Calc
	s_mul_hi_u32 s99, s98, s65                             // high 32 bits of multiply
	s_mul_i32 s98, s98, s65
	s_branch label_GS_MG_StrideC_Done                      // jump to GS_MG_StrideC_Done
label_GS_MG_StrideC_Calc:
	s_abs_i32 s98, s98                                     // absolute value
	s_mul_hi_u32 s99, s98, s65                             // high 32 bits of multiply
	s_mul_i32 s98, s98, s65
	s_xor_b32 s98, s98, -1                                 // bitwise negate
	s_xor_b32 s99, s99, -1                                 // bitwise negate
	s_add_u32 s98, s98, 1
	s_addc_u32 s99, s99, 0                                 // carry from above
label_GS_MG_StrideC_Done:
	s_sub_u32 s98, s98, s62
	s_subb_u32 s99, s99, s63                               // sub with borrow
	s_add_u32 s[sgprSrdB], s[sgprSrdB], s98                // adjust SRD B
	s_addc_u32 s53, s53, s99                               // add with carry
	s_sub_u32 s58, s58, s98
	s_subb_u32 s59, s59, s99                               // sub with borrow
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1                             // SRD B num records
	buffer_load_d16_b16 v16, v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen // G -> Reg A
	buffer_load_d16_hi_b16 v16, v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // G -> Reg A
	buffer_load_d16_b16 v17, v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // G -> Reg A
	buffer_load_d16_hi_b16 v17, v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen offset:6 // G -> Reg A
	buffer_load_d16_b16 v18, v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // G -> Reg A
	buffer_load_d16_hi_b16 v18, v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen offset:10 // G -> Reg A
	buffer_load_d16_b16 v19, v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // G -> Reg A
	buffer_load_d16_hi_b16 v19, v[vgprGlobalReadOffsetA], s[sgprSrdA:sgprSrdA+3], 0 offen offset:14 // G -> Reg A
	buffer_load_d16_b16 v20, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], 0 offen // G -> Reg B
	buffer_load_d16_hi_b16 v20, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s66 offen // G -> Reg B
	buffer_load_d16_b16 v21, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s67 offen // G -> Reg B
	buffer_load_d16_hi_b16 v21, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s68 offen // G -> Reg B
	buffer_load_d16_b16 v22, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s69 offen // G -> Reg B
	buffer_load_d16_hi_b16 v22, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s70 offen // G -> Reg B
	buffer_load_d16_b16 v23, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s71 offen // G -> Reg B
	buffer_load_d16_hi_b16 v23, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s72 offen // G -> Reg B
	buffer_load_d16_b16 v24, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s73 offen // G -> Reg B
	buffer_load_d16_hi_b16 v24, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s74 offen // G -> Reg B
	buffer_load_d16_b16 v25, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s75 offen // G -> Reg B
	buffer_load_d16_hi_b16 v25, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s76 offen // G -> Reg B
	buffer_load_d16_b16 v26, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s77 offen // G -> Reg B
	buffer_load_d16_hi_b16 v26, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s78 offen // G -> Reg B
	buffer_load_d16_b16 v27, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s79 offen // G -> Reg B
	buffer_load_d16_hi_b16 v27, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s80 offen // G -> Reg B
	buffer_load_d16_b16 v28, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s81 offen // G -> Reg B
	buffer_load_d16_hi_b16 v28, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s82 offen // G -> Reg B
	buffer_load_d16_b16 v29, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s83 offen // G -> Reg B
	buffer_load_d16_hi_b16 v29, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s84 offen // G -> Reg B
	buffer_load_d16_b16 v30, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s85 offen // G -> Reg B
	buffer_load_d16_hi_b16 v30, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s86 offen // G -> Reg B
	buffer_load_d16_b16 v31, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s87 offen // G -> Reg B
	buffer_load_d16_hi_b16 v31, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s88 offen // G -> Reg B
	buffer_load_d16_b16 v32, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s89 offen // G -> Reg B
	buffer_load_d16_hi_b16 v32, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s90 offen // G -> Reg B
	buffer_load_d16_b16 v33, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s91 offen // G -> Reg B
	buffer_load_d16_hi_b16 v33, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s92 offen // G -> Reg B
	buffer_load_d16_b16 v34, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s93 offen // G -> Reg B
	buffer_load_d16_hi_b16 v34, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s94 offen // G -> Reg B
	buffer_load_d16_b16 v35, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s95 offen // G -> Reg B
	buffer_load_d16_hi_b16 v35, v[vgprGlobalReadOffsetB], s[sgprSrdB:sgprSrdB+3], s96 offen // G -> Reg B
	s_waitcnt vmcnt(0)                                     // wait for global read
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	ds_store_b128 v10, v[16:19]                            // local write
	ds_store_b16 v11, v20                                  // local write
	ds_store_b16_d16_hi v11, v20 offset:320                // local write
	ds_store_b16 v11, v21 offset:640                       // local write
	ds_store_b16_d16_hi v11, v21 offset:960                // local write
	ds_store_b16 v11, v22 offset:1280                      // local write
	ds_store_b16_d16_hi v11, v22 offset:1600               // local write
	ds_store_b16 v11, v23 offset:1920                      // local write
	ds_store_b16_d16_hi v11, v23 offset:2240               // local write
	ds_store_b16 v11, v24 offset:2560                      // local write
	ds_store_b16_d16_hi v11, v24 offset:2880               // local write
	ds_store_b16 v11, v25 offset:3200                      // local write
	ds_store_b16_d16_hi v11, v25 offset:3520               // local write
	ds_store_b16 v11, v26 offset:3840                      // local write
	ds_store_b16_d16_hi v11, v26 offset:4160               // local write
	ds_store_b16 v11, v27 offset:4480                      // local write
	ds_store_b16_d16_hi v11, v27 offset:4800               // local write
	ds_store_b16 v11, v28 offset:5120                      // local write
	ds_store_b16_d16_hi v11, v28 offset:5440               // local write
	ds_store_b16 v11, v29 offset:5760                      // local write
	ds_store_b16_d16_hi v11, v29 offset:6080               // local write
	ds_store_b16 v11, v30 offset:6400                      // local write
	ds_store_b16_d16_hi v11, v30 offset:6720               // local write
	ds_store_b16 v11, v31 offset:7040                      // local write
	ds_store_b16_d16_hi v11, v31 offset:7360               // local write
	ds_store_b16 v11, v32 offset:7680                      // local write
	ds_store_b16_d16_hi v11, v32 offset:8000               // local write
	ds_store_b16 v11, v33 offset:8320                      // local write
	ds_store_b16_d16_hi v11, v33 offset:8640               // local write
	ds_store_b16 v11, v34 offset:8960                      // local write
	ds_store_b16_d16_hi v11, v34 offset:9280               // local write
	ds_store_b16 v11, v35 offset:9600                      // local write
	ds_store_b16_d16_hi v11, v35 offset:9920               // local write
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	v_and_b32_e32 v[vgprLocalWriteAddrA], 0x3fff, v[vgprLocalWriteAddrA] // mask to single LDS buffer
	v_and_b32_e32 v[vgprLocalWriteAddrB], 0x3fff, v[vgprLocalWriteAddrB] // mask to single LDS buffer
label_GS_MG_LoopBody:
	ds_load_b128 v[16:19], v[vgprLocalWriteAddrA]          // L -> Reg
	ds_load_b128 v[20:23], v[vgprLocalWriteAddrA] offset:16 // L -> Reg
	ds_load_b128 v[48:51], v[vgprLocalWriteAddrB]          // L -> Reg
	ds_load_b128 v[52:55], v[vgprLocalWriteAddrB] offset:16 // L -> Reg
	s_mov_b32 s[sgprLoopCounterL], 32
	v_add_co_u32 v[vgprLocalWriteAddrA], vcc_lo, s[sgprLoopCounterL], v[vgprLocalWriteAddrA]
	v_add_co_u32 v[vgprLocalWriteAddrB], vcc_lo, s[sgprLoopCounterL], v[vgprLocalWriteAddrB]
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_sub_i32 s98, s12, 1
	s_lshr_b32 s99, s98, 2
	s_and_b32 s98, s98, 3
	s_sub_i32 s98, 3, s98
	s_lshl_b32 s98, s98, 4
	v_cmp_eq_i32_e64 s100, s99, 0
	v_lshlrev_b64 v[80:81], s98, v[16:17]
	v_cndmask_b32_e64 v16, v16, v80, s100                  // tail mask
	v_cndmask_b32_e64 v17, v17, v81, s100                  // tail mask
	v_lshlrev_b64 v[80:81], s98, v[48:49]
	v_cndmask_b32_e64 v48, v48, v80, s100                  // tail mask
	v_cndmask_b32_e64 v49, v49, v81, s100                  // tail mask
	v_cmp_eq_i32_e64 s100, s99, 1
	v_lshlrev_b64 v[80:81], s98, v[18:19]
	v_cndmask_b32_e64 v18, v18, v80, s100                  // tail mask
	v_cndmask_b32_e64 v19, v19, v81, s100                  // tail mask
	v_lshlrev_b64 v[80:81], s98, v[50:51]
	v_cndmask_b32_e64 v50, v50, v80, s100                  // tail mask
	v_cndmask_b32_e64 v51, v51, v81, s100                  // tail mask
	v_cmp_lt_i32_e64 s100, s99, 1
	v_cndmask_b32_e64 v18, v18, 0, s100                    // zero if out of bounds
	v_cndmask_b32_e64 v19, v19, 0, s100                    // zero if out of bounds
	v_cndmask_b32_e64 v50, v50, 0, s100                    // zero if out of bounds
	v_cndmask_b32_e64 v51, v51, 0, s100                    // zero if out of bounds
	v_cmp_eq_i32_e64 s100, s99, 2
	v_lshlrev_b64 v[80:81], s98, v[20:21]
	v_cndmask_b32_e64 v20, v20, v80, s100                  // tail mask
	v_cndmask_b32_e64 v21, v21, v81, s100                  // tail mask
	v_lshlrev_b64 v[80:81], s98, v[52:53]
	v_cndmask_b32_e64 v52, v52, v80, s100                  // tail mask
	v_cndmask_b32_e64 v53, v53, v81, s100                  // tail mask
	v_cmp_lt_i32_e64 s100, s99, 2
	v_cndmask_b32_e64 v20, v20, 0, s100                    // zero if out of bounds
	v_cndmask_b32_e64 v21, v21, 0, s100                    // zero if out of bounds
	v_cndmask_b32_e64 v52, v52, 0, s100                    // zero if out of bounds
	v_cndmask_b32_e64 v53, v53, 0, s100                    // zero if out of bounds
	v_cmp_eq_i32_e64 s100, s99, 3
	v_lshlrev_b64 v[80:81], s98, v[22:23]
	v_cndmask_b32_e64 v22, v22, v80, s100                  // tail mask
	v_cndmask_b32_e64 v23, v23, v81, s100                  // tail mask
	v_lshlrev_b64 v[80:81], s98, v[54:55]
	v_cndmask_b32_e64 v54, v54, v80, s100                  // tail mask
	v_cndmask_b32_e64 v55, v55, v81, s100                  // tail mask
	v_cmp_lt_i32_e64 s100, s99, 3
	v_cndmask_b32_e64 v22, v22, 0, s100                    // zero if out of bounds
	v_cndmask_b32_e64 v23, v23, 0, s100                    // zero if out of bounds
	v_cndmask_b32_e64 v54, v54, 0, s100                    // zero if out of bounds
	v_cndmask_b32_e64 v55, v55, 0, s100                    // zero if out of bounds
	s_nop 1                                                // 2 wait states
	v_wmma_f32_16x16x16_f16 v[vgprValuC:vgprValuC+7], v[48:55], v[16:23], v[vgprValuC:vgprValuC+7]
	s_sub_i32 s12, s12, 16
	s_add_u32 s13, s13, 16
	s_cmp_le_i32 s12, 0
	s_cbranch_scc0 label_GS_MG_LoopBody                    // branch to GS_MG_LoopBody
label_EndPgm_MG:
	s_and_b32 s[sgprLoopCounterL], s[sgprWGMInfo], 0x3fff  // get GSU
	s_cmp_eq_u32 s[sgprLoopCounterL], 1
	s_cbranch_scc0 label_GS_MG_AfterSAV                    // branch to GS_MG_AfterSAV
	s_cmp_eq_u32 s[sgprArgType], 2                         // ArgType == 2 ?
	s_cbranch_scc1 label_GS_MG_SAV_BetaCheck               // branch to GS_MG_SAV_BetaCheck
	s_load_b256 s[sgprSrdA:sgprSrdA+7], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 // load store args
	s_load_b32 s56, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78 // load store args
	s_branch label_GS_MG_AfterSAV                          // jump to GS_MG_AfterSAV
label_GS_MG_SAV_BetaCheck:
	s_load_b128 s[sgprSrdA:sgprSrdA+3], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90 // load store args
	s_load_b64 s[sgprSrdB:sgprSrdB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xa0 // load store args
	s_load_b64 s[54:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb8 // load store args
	s_load_b32 s56, s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc0 // load store args
label_GS_MG_AfterSAV:
	v_lshrrev_b32_e32 v[vgprLocalWriteAddrA], 5, v[vgprSerial]
	v_lshrrev_b32_e32 v[vgprLocalWriteAddrB], 0, v[vgprLocalWriteAddrA]
	v_mul_lo_u32 v[vgprLocalWriteAddrB], 16, v[vgprLocalWriteAddrB]
	v_and_b32_e32 v[vgprGlobalReadOffsetB], 31, v[vgprSerial]
	v_lshrrev_b32_e32 v[vgprGlobalReadOffsetB], 4, v[vgprGlobalReadOffsetB]
	v_add_lshl_u32 v[vgprGlobalReadOffsetB], v[vgprLocalWriteAddrB], v[vgprGlobalReadOffsetB], 0
	v_mul_lo_u32 v10, v[vgprGlobalReadOffsetB], s[sgprStrideC0] // offset * strideC0
	v_mul_lo_u32 v11, v[vgprGlobalReadOffsetB], s[sgprStrideD0] // offset * strideD0
	v_and_b32_e32 v[vgprGlobalReadOffsetA], 0, v[vgprLocalWriteAddrA]
	v_mul_lo_u32 v[vgprGlobalReadOffsetA], 16, v[vgprGlobalReadOffsetA]
	v_and_b32_e32 v[vgprLocalWriteAddrB], 15, v[vgprSerial]
	v_add_lshl_u32 v[vgprGlobalReadOffsetA], v[vgprLocalWriteAddrB], v[vgprGlobalReadOffsetA], 0
	s_mul_i32 s[sgprLoopCounterL], 16, s[sgprWorkGroup0]   // tile offset
	v_add_nc_u32_e32 v[vgprGlobalReadOffsetA], s[sgprLoopCounterL], v[vgprGlobalReadOffsetA]
	s_mul_i32 s[sgprLoopCounterL], 64, s[sgprWorkGroup1]   // tile offset
	v_add_nc_u32_e32 v[vgprGlobalReadOffsetB], s[sgprLoopCounterL], v[vgprGlobalReadOffsetB]
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_and_b32 s[sgprLoopCounterL], s[sgprWGMInfo], 0x3fff  // get GSU
	s_cmp_eq_u32 s[sgprLoopCounterL], 1
	s_cbranch_scc1 label_GS_MG_ActEnd                      // branch to GS_MG_ActEnd
	s_and_b32 s[sgprAddressC], 15, s[sgprSizesFree0]       // SizesFree0 mod MT0
	s_add_u32 s31, -1, s[sgprNumWorkGroups0]
	s_cmp_ge_u32 s[sgprWorkGroup0], s31
	s_cselect_b32 s[sgprAddressC], s[sgprAddressC], 0      // zero if not edge WG
	s_cmpk_gt_u32 s[sgprAddressC], 0x0                     // compare greater
	s_cbranch_scc1 label_GS_MG_NoBias                      // branch to GS_MG_NoBias
	s_and_b32 s[sgprAddressC], 63, s[sgprSizesFree1]       // SizesFree1 mod MT1
	s_add_u32 s31, -1, s[sgprNumWorkGroups1]
	s_cmp_ge_u32 s[sgprWorkGroup1], s31
	s_cselect_b32 s[sgprAddressC], s[sgprAddressC], 0      // zero if not edge WG
	s_cmpk_gt_u32 s[sgprAddressC], 0x0                     // compare greater
	s_cbranch_scc1 label_GS_MG_NoBias                      // branch to GS_MG_NoBias
	v_add_lshl_u32 v15, v11, v[vgprGlobalReadOffsetA], 2
	v_mov_b32_e32 v17, v[vgprValuC]
	v_mov_b32_e32 v18, v1
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v3
	v_mov_b32_e32 v21, v4
	v_mov_b32_e32 v22, v5
	v_mov_b32_e32 v23, v6
	v_mov_b32_e32 v24, v7
	buffer_store_b32 v17, v15, s[16:19], 0 offen           // store D
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 8      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b32 v18, v15, s[16:19], 0 offen           // store D
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 8      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b32 v19, v15, s[16:19], 0 offen           // store D
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 8      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b32 v20, v15, s[16:19], 0 offen           // store D
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 8      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b32 v21, v15, s[16:19], 0 offen           // store D
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 8      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b32 v22, v15, s[16:19], 0 offen           // store D
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 8      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b32 v23, v15, s[16:19], 0 offen           // store D
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 8      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b32 v24, v15, s[16:19], 0 offen           // store D
	s_nop 0                                                // 1 wait states
	s_branch label_GS_MG_BiasActEnd                        // jump to GS_MG_BiasActEnd
label_GS_MG_NoBias:
	v_mov_b32_e32 v14, 0x80000000                          // OOB sentinel
	v_cmp_lt_u32_e64 s[sgprAddressC], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressA], s[sgprAddressC], s[sgprAddressA] // combine OOB masks
	v_add_lshl_u32 v23, v11, v[vgprGlobalReadOffsetA], 2
	v_cndmask_b32_e64 v23, v14, v23, s[sgprAddressA]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressC], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressC]                 // advance C offset
	s_mul_i32 s[sgprAddressC], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressC]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressC], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressA], s[sgprAddressC], s[sgprAddressA] // combine OOB masks
	v_add_lshl_u32 v24, v11, v[vgprGlobalReadOffsetA], 2
	v_cndmask_b32_e64 v24, v14, v24, s[sgprAddressA]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressC], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressC]                 // advance C offset
	s_mul_i32 s[sgprAddressC], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressC]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressC], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressA], s[sgprAddressC], s[sgprAddressA] // combine OOB masks
	v_add_lshl_u32 v25, v11, v[vgprGlobalReadOffsetA], 2
	v_cndmask_b32_e64 v25, v14, v25, s[sgprAddressA]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressC], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressC]                 // advance C offset
	s_mul_i32 s[sgprAddressC], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressC]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressC], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressA], s[sgprAddressC], s[sgprAddressA] // combine OOB masks
	v_add_lshl_u32 v26, v11, v[vgprGlobalReadOffsetA], 2
	v_cndmask_b32_e64 v26, v14, v26, s[sgprAddressA]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressC], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressC]                 // advance C offset
	s_mul_i32 s[sgprAddressC], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressC]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressC], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressA], s[sgprAddressC], s[sgprAddressA] // combine OOB masks
	v_add_lshl_u32 v27, v11, v[vgprGlobalReadOffsetA], 2
	v_cndmask_b32_e64 v27, v14, v27, s[sgprAddressA]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressC], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressC]                 // advance C offset
	s_mul_i32 s[sgprAddressC], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressC]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressC], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressA], s[sgprAddressC], s[sgprAddressA] // combine OOB masks
	v_add_lshl_u32 v28, v11, v[vgprGlobalReadOffsetA], 2
	v_cndmask_b32_e64 v28, v14, v28, s[sgprAddressA]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressC], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressC]                 // advance C offset
	s_mul_i32 s[sgprAddressC], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressC]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressC], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressA], s[sgprAddressC], s[sgprAddressA] // combine OOB masks
	v_add_lshl_u32 v29, v11, v[vgprGlobalReadOffsetA], 2
	v_cndmask_b32_e64 v29, v14, v29, s[sgprAddressA]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressC], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressC]                 // advance C offset
	s_mul_i32 s[sgprAddressC], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressC]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressC], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressA], s[sgprAddressC], s[sgprAddressA] // combine OOB masks
	v_add_lshl_u32 v30, v11, v[vgprGlobalReadOffsetA], 2
	v_cndmask_b32_e64 v30, v14, v30, s[sgprAddressA]       // OOB offset if mask false
	v_mov_b32_e32 v15, v[vgprValuC]
	v_mov_b32_e32 v16, v1
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v19, v4
	v_mov_b32_e32 v20, v5
	v_mov_b32_e32 v21, v6
	v_mov_b32_e32 v22, v7
	buffer_store_b32 v15, v23, s[16:19], 0 offen           // store D
	buffer_store_b32 v16, v24, s[16:19], 0 offen           // store D
	buffer_store_b32 v17, v25, s[16:19], 0 offen           // store D
	buffer_store_b32 v18, v26, s[16:19], 0 offen           // store D
	buffer_store_b32 v19, v27, s[16:19], 0 offen           // store D
	buffer_store_b32 v20, v28, s[16:19], 0 offen           // store D
	buffer_store_b32 v21, v29, s[16:19], 0 offen           // store D
	buffer_store_b32 v22, v30, s[16:19], 0 offen           // store D
	s_nop 0                                                // 1 wait states
	s_branch label_GS_MG_BiasActEnd                        // jump to GS_MG_BiasActEnd
label_GS_MG_BiasActEnd:
	s_getpc_b64 s[sgprAddressC:sgprAddressC+1]             // addr of next instr
	s_add_i32 s[sgprAddressA], 0xed8, 4
	s_add_u32 s[sgprAddressC], s[sgprAddressC], s[sgprAddressA]
	s_addc_u32 s31, s31, 0                                 // carry from above
	s_setpc_b64 s[sgprAddressC:sgprAddressC+1]             // long branch forward
label_GS_MG_ActEnd:
	s_mov_b64 s[sgprAddressA:sgprAddressA+1], s[sgprSrdA:sgprSrdA+1]
	s_mov_b32 s35, 0x31004000                              // SRD Dword3
	s_cmp_eq_u64 s[sgprSrdA:sgprSrdA+1], 0                 // compare 64-bit
	s_cbranch_scc0 label_GS_MG2_StrideD_NonUnit            // branch to GS_MG2_StrideD_NonUnit
	s_mov_b32 s[sgprAddressB], 0
	s_branch label_GS_MG2_StrideD_Done                     // jump to GS_MG2_StrideD_Done
label_GS_MG2_StrideD_NonUnit:
	s_mov_b32 s[sgprAddressB], s[sgprSizesFree0]
label_GS_MG2_StrideD_Done:
	s_mul_i32 s[sgprAddressB], 4, s[sgprAddressB]
	s_add_u32 s[sgprLoopCounterL], s[sgprWorkGroup2], 1
	s_mul_i32 s[sgprLoopCounterL], s53, s[sgprLoopCounterL]
	s_cmp_eq_u32 s[sgprLoopCounterL], 0
	s_cselect_b32 s[sgprLoopCounterL], s[sgprSizesFree0], s[sgprLoopCounterL] // select full tile if partial
	s_mov_b64 s[sgprStrideA0:sgprStrideA0+1], s[50:51]
	s_mov_b32 s[sgprStrideB1], 0x31004000                  // SRD Dword3
	s_cmp_eq_u64 s[50:51], 0                               // compare 64-bit
	s_cbranch_scc0 label_GS_MG2_StrideC_NonUnit            // branch to GS_MG2_StrideC_NonUnit
	s_mov_b32 s[sgprStrideB0], 0
	s_branch label_GS_MG2_StrideC_Done                     // jump to GS_MG2_StrideC_Done
label_GS_MG2_StrideC_NonUnit:
	s_mov_b32 s[sgprStrideB0], s[sgprLoopCounterL]
label_GS_MG2_StrideC_Done:
	s_cmpk_lg_u32 s[sgprSrdB], 0x0                         // compare not equal
	s_cbranch_scc1 label_GS_MG2_NoBias                     // branch to GS_MG2_NoBias
	s_mul_i32 s[sgprLoopCounterL], 16, s[sgprWorkGroup0]   // tile offset
	v_add_nc_u32_e32 v16, s[sgprLoopCounterL], v[vgprSerial]
	s_mul_i32 s[sgprStrideB0], 4, s[sgprStrideB0]          // Bias num_records *= sizeof(f32)
	s_mul_i32 s[sgprLoopCounterL], s53, s[sgprWorkGroup2]  // tile offset
	v_add_nc_u32_e32 v14, s[sgprLoopCounterL], v16
	v_lshlrev_b32_e32 v14, 2, v14
	v_lshlrev_b32_e32 v15, 2, v16
	s_mul_i32 s[sgprLoopCounterL], 64, s[sgprWorkGroup1]   // tile offset
	v_add_nc_u32_e32 v16, s[sgprLoopCounterL], v[vgprSerial]
	buffer_load_b32 v[vgprLocalWriteAddrA], v14, s[sgprStrideA0:sgprStrideA0+3], 0 offen // load bias
	buffer_load_b32 v[vgprLocalWriteAddrB], v15, s[sgprAddressA:sgprAddressA+3], 0 offen // load ScaleAlphaVec
	v_lshlrev_b32_e32 v16, 2, v[vgprSerial]
	s_barrier                                              // LDS barrier
	s_waitcnt vmcnt(1)                                     // wait for global read
	ds_store_b32 v16, v[vgprLocalWriteAddrA]               // local write
	v_cmp_gt_u32_e64 s[sgprSrdA], s[sgprAddressB], 0
	s_waitcnt vmcnt(0)                                     // wait for global read
	v_cndmask_b32_e64 v[vgprLocalWriteAddrB], 1.0, v[vgprLocalWriteAddrB], s[sgprSrdA] // use 1.0 if no ScaleAlphaVec
	ds_store_b32 v16, v[vgprLocalWriteAddrB] offset:512    // local write
	s_branch label_GS_MG2_BiasEnd                          // jump to GS_MG2_BiasEnd
label_GS_MG2_NoBias:
	s_cmpk_lg_u32 s[sgprSrdB], 0x4                         // compare not equal
	s_cbranch_scc1 label_GS_MG2_BiasEnd                    // branch to GS_MG2_BiasEnd
	s_mul_i32 s[sgprLoopCounterL], 16, s[sgprWorkGroup0]   // tile offset
	v_add_nc_u32_e32 v16, s[sgprLoopCounterL], v[vgprSerial]
	s_mul_i32 s[sgprStrideB0], 2, s[sgprStrideB0]          // Bias num_records *= sizeof(f16)
	s_mul_i32 s[sgprLoopCounterL], s53, s[sgprWorkGroup2]  // tile offset
	v_add_nc_u32_e32 v14, s[sgprLoopCounterL], v16
	v_lshlrev_b32_e32 v14, 1, v14
	v_lshlrev_b32_e32 v15, 2, v16
	s_mul_i32 s[sgprLoopCounterL], 64, s[sgprWorkGroup1]   // tile offset
	v_add_nc_u32_e32 v16, s[sgprLoopCounterL], v[vgprSerial]
	buffer_load_d16_b16 v[vgprLocalWriteAddrA], v14, s[sgprStrideA0:sgprStrideA0+3], 0 offen // load bias (f16)
	buffer_load_b32 v[vgprLocalWriteAddrB], v15, s[sgprAddressA:sgprAddressA+3], 0 offen // load ScaleAlphaVec
	v_lshlrev_b32_e32 v16, 2, v[vgprSerial]
	s_barrier                                              // LDS barrier
	s_waitcnt vmcnt(1)                                     // wait for global read
	v_cvt_f32_f16_e32 v[vgprLocalWriteAddrA], v12.l        // convert to f32
	ds_store_b32 v16, v[vgprLocalWriteAddrA]               // local write
	v_cmp_gt_u32_e64 s[sgprSrdA], s[sgprAddressB], 0
	s_waitcnt vmcnt(0)                                     // wait for global read
	v_cndmask_b32_e64 v[vgprLocalWriteAddrB], 1.0, v[vgprLocalWriteAddrB], s[sgprSrdA] // use 1.0 if no ScaleAlphaVec
	ds_store_b32 v16, v[vgprLocalWriteAddrB] offset:512    // local write
	s_branch label_GS_MG2_BiasEnd                          // jump to GS_MG2_BiasEnd
label_GS_MG2_BiasEnd:
	s_cmpk_eq_u32 s56, 0x3                                 // check activation type
	s_cbranch_scc1 label_GS_MG2_ActGELU                    // branch to GS_MG2_ActGELU
	s_cmpk_eq_u32 s56, 0x5                                 // check activation type
	s_cbranch_scc1 label_GS_MG2_ActRELU                    // branch to GS_MG2_ActRELU
	s_cmpk_eq_u32 s56, 0x6                                 // check activation type
	s_cbranch_scc1 label_GS_MG2_ActSigmoid                 // branch to GS_MG2_ActSigmoid
	s_cmpk_eq_u32 s56, 0xa                                 // check activation type
	s_cbranch_scc1 label_GS_MG2_ActSilu                    // branch to GS_MG2_ActSilu
	s_cmpk_eq_u32 s56, 0xc                                 // check activation type
	s_cbranch_scc1 label_GS_MG2_ActClip                    // branch to GS_MG2_ActClip
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0xd64, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_GS_MG2_ActSet                           // jump to GS_MG2_ActSet
label_GS_MG2_ActGELU:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0xd50, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_GS_MG2_ActSet                           // jump to GS_MG2_ActSet
label_GS_MG2_ActRELU:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0xd74, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_GS_MG2_ActSet                           // jump to GS_MG2_ActSet
label_GS_MG2_ActSigmoid:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0xd68, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_GS_MG2_ActSet                           // jump to GS_MG2_ActSet
label_GS_MG2_ActSilu:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0xd68, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_GS_MG2_ActSet                           // jump to GS_MG2_ActSet
label_GS_MG2_ActClip:
	s_getpc_b64 s[12:13]                                   // addr of next instr
	s_add_i32 s[sgprLoopCounterL], 0xd6c, 4
	s_add_u32 s12, s12, s[sgprLoopCounterL]
	s_addc_u32 s13, s13, 0                                 // carry from above
	s_branch label_GS_MG2_ActSet                           // jump to GS_MG2_ActSet
label_GS_MG2_ActSet:
	s_cmpk_eq_u32 s[sgprBeta], 0x0                         // beta == 0 ?
	s_cbranch_scc0 label_EndPgm_MG2                        // branch to EndPgm_MG2
	s_and_b32 s[sgprAddressA], 15, s[sgprSizesFree0]       // SizesFree0 mod MT0
	s_add_u32 s33, -1, s[sgprNumWorkGroups0]
	s_cmp_ge_u32 s[sgprWorkGroup0], s33
	s_cselect_b32 s[sgprAddressA], s[sgprAddressA], 0      // zero if not edge WG
	s_cmpk_gt_u32 s[sgprAddressA], 0x0                     // compare greater
	s_cbranch_scc1 label_GS_MG2_NoBiasActEnd               // branch to GS_MG2_NoBiasActEnd
	s_and_b32 s[sgprAddressA], 63, s[sgprSizesFree1]       // SizesFree1 mod MT1
	s_add_u32 s33, -1, s[sgprNumWorkGroups1]
	s_cmp_ge_u32 s[sgprWorkGroup1], s33
	s_cselect_b32 s[sgprAddressA], s[sgprAddressA], 0      // zero if not edge WG
	s_cmpk_gt_u32 s[sgprAddressA], 0x0                     // compare greater
	s_cbranch_scc1 label_GS_MG2_NoBiasActEnd               // branch to GS_MG2_NoBiasActEnd
	s_mul_i32 s[sgprLoopCounterL], 16, s[sgprWorkGroup0]   // tile offset
	v_sub_nc_u32_e64 v17, v[vgprGlobalReadOffsetA], s[sgprLoopCounterL]
	v_lshlrev_b32_e32 v17, 2, v17
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	ds_load_b32 v26, v17                                   // L -> Reg
	ds_load_b32 v27, v17 offset:512                        // L -> Reg
	v_add_lshl_u32 v15, v11, v[vgprGlobalReadOffsetA], 1
	v_mul_f32_e32 v18, s[sgprAlpha], v[vgprValuC]          // *= alpha
	v_mul_f32_e32 v19, s[sgprAlpha], v1                    // *= alpha
	v_mul_f32_e32 v20, s[sgprAlpha], v2                    // *= alpha
	v_mul_f32_e32 v21, s[sgprAlpha], v3                    // *= alpha
	v_mul_f32_e32 v22, s[sgprAlpha], v4                    // *= alpha
	v_mul_f32_e32 v23, s[sgprAlpha], v5                    // *= alpha
	v_mul_f32_e32 v24, s[sgprAlpha], v6                    // *= alpha
	v_mul_f32_e32 v25, s[sgprAlpha], v7                    // *= alpha
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	v_mul_f32_e32 v18, v27, v18                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v18         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v18, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v18.l, v18                           // convert to f16
	buffer_store_b16 v18, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v19, v27, v19                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v19         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v19, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v19.l, v19                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v19, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v20, v27, v20                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v20         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v20, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v20.l, v20                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v20, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v21, v27, v21                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v21         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v21, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v21.l, v21                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v21, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v22, v27, v22                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v22         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v22, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v22.l, v22                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v22, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v23, v27, v23                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v23         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v23, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v23.l, v23                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v23, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v24, v27, v24                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v24         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v24, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v24.l, v24                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v24, v15, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v25, v27, v25                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v26, v25         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v25, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v25.l, v25                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v25, v15, s[16:19], 0 offen           // store D
	s_nop 0                                                // 1 wait states
	s_branch label_Final_EndPgm                            // jump to Final_EndPgm
label_GS_MG2_NoBiasActEnd:
	v_mov_b32_e32 v14, 0x80000000                          // OOB sentinel
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v26, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v26, 2, v26
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	ds_load_b32 v23, v26                                   // L -> Reg
	ds_load_b32 v24, v26 offset:512                        // L -> Reg
	v_add_lshl_u32 v25, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v25, v14, v25, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v28, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v28, 2, v28
	v_add_lshl_u32 v27, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v27, v14, v27, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v30, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v30, 2, v30
	v_add_lshl_u32 v29, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v29, v14, v29, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v32, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v32, 2, v32
	v_add_lshl_u32 v31, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v31, v14, v31, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v34, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v34, 2, v34
	v_add_lshl_u32 v33, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v33, v14, v33, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v36, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v36, 2, v36
	v_add_lshl_u32 v35, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v35, v14, v35, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v38, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v38, 2, v38
	v_add_lshl_u32 v37, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v37, v14, v37, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v40, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v40, 2, v40
	v_add_lshl_u32 v39, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v39, v14, v39, s[sgprAddressB]       // OOB offset if mask false
	v_mul_f32_e32 v15, s[sgprAlpha], v[vgprValuC]          // *= alpha
	v_mul_f32_e32 v16, s[sgprAlpha], v1                    // *= alpha
	v_mul_f32_e32 v17, s[sgprAlpha], v2                    // *= alpha
	v_mul_f32_e32 v18, s[sgprAlpha], v3                    // *= alpha
	v_mul_f32_e32 v19, s[sgprAlpha], v4                    // *= alpha
	v_mul_f32_e32 v20, s[sgprAlpha], v5                    // *= alpha
	v_mul_f32_e32 v21, s[sgprAlpha], v6                    // *= alpha
	v_mul_f32_e32 v22, s[sgprAlpha], v7                    // *= alpha
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	v_mul_f32_e32 v15, v24, v15                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v23, v15         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v15, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v15.l, v15                           // convert to f16
	buffer_store_b16 v15, v25, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v16, v24, v16                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v23, v16         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v16, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v16.l, v16                           // convert to f16
	buffer_store_b16 v16, v27, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v17, v24, v17                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v23, v17         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v17, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v17.l, v17                           // convert to f16
	buffer_store_b16 v17, v29, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v18, v24, v18                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v23, v18         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v18, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v18.l, v18                           // convert to f16
	buffer_store_b16 v18, v31, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v19, v24, v19                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v23, v19         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v19, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v19.l, v19                           // convert to f16
	buffer_store_b16 v19, v33, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v20, v24, v20                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v23, v20         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v20, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v20.l, v20                           // convert to f16
	buffer_store_b16 v20, v35, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v21, v24, v21                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v23, v21         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v21, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v21.l, v21                           // convert to f16
	buffer_store_b16 v21, v37, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v22, v24, v22                            // *= ScaleAlphaVec
	v_add_f32_e32 v[vgprLocalWriteAddrA], v23, v22         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v22, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v22.l, v22                           // convert to f16
	buffer_store_b16 v22, v39, s[16:19], 0 offen           // store D
	s_nop 0                                                // 1 wait states
	s_branch label_Final_EndPgm                            // jump to Final_EndPgm
label_EndPgm_MG2:
	s_and_b32 s[sgprAddressA], 15, s[sgprSizesFree0]       // SizesFree0 mod MT0
	s_add_u32 s33, -1, s[sgprNumWorkGroups0]
	s_cmp_ge_u32 s[sgprWorkGroup0], s33
	s_cselect_b32 s[sgprAddressA], s[sgprAddressA], 0      // zero if not edge WG
	s_cmpk_gt_u32 s[sgprAddressA], 0x0                     // compare greater
	s_cbranch_scc1 label_GS_MG2_BiasAct_NoBias             // branch to GS_MG2_BiasAct_NoBias
	s_and_b32 s[sgprAddressA], 63, s[sgprSizesFree1]       // SizesFree1 mod MT1
	s_add_u32 s33, -1, s[sgprNumWorkGroups1]
	s_cmp_ge_u32 s[sgprWorkGroup1], s33
	s_cselect_b32 s[sgprAddressA], s[sgprAddressA], 0      // zero if not edge WG
	s_cmpk_gt_u32 s[sgprAddressA], 0x0                     // compare greater
	s_cbranch_scc1 label_GS_MG2_BiasAct_NoBias             // branch to GS_MG2_BiasAct_NoBias
	v_add_lshl_u32 v16, v10, v[vgprGlobalReadOffsetA], 1
	buffer_load_d16_b16 v26, v16, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprLoopCounterL], 16, s[sgprWorkGroup0]   // tile offset
	v_sub_nc_u32_e64 v17, v[vgprGlobalReadOffsetA], s[sgprLoopCounterL]
	v_lshlrev_b32_e32 v17, 2, v17
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	ds_load_b32 v27, v17                                   // L -> Reg
	ds_load_b32 v28, v17 offset:512                        // L -> Reg
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideC0], 4      // C stride offset
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprLoopCounterL] // advance C SRD
	s_addc_u32 s[sgprArgType2], s[sgprArgType2], 0         // carry from above
	buffer_load_d16_b16 v29, v16, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideC0], 4      // C stride offset
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprLoopCounterL] // advance C SRD
	s_addc_u32 s[sgprArgType2], s[sgprArgType2], 0         // carry from above
	buffer_load_d16_b16 v30, v16, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideC0], 4      // C stride offset
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprLoopCounterL] // advance C SRD
	s_addc_u32 s[sgprArgType2], s[sgprArgType2], 0         // carry from above
	buffer_load_d16_b16 v31, v16, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideC0], 4      // C stride offset
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprLoopCounterL] // advance C SRD
	s_addc_u32 s[sgprArgType2], s[sgprArgType2], 0         // carry from above
	buffer_load_d16_b16 v32, v16, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideC0], 4      // C stride offset
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprLoopCounterL] // advance C SRD
	s_addc_u32 s[sgprArgType2], s[sgprArgType2], 0         // carry from above
	buffer_load_d16_b16 v33, v16, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideC0], 4      // C stride offset
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprLoopCounterL] // advance C SRD
	s_addc_u32 s[sgprArgType2], s[sgprArgType2], 0         // carry from above
	buffer_load_d16_b16 v34, v16, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideC0], 4      // C stride offset
	s_add_u32 s[sgprGemmInfo], s[sgprGemmInfo], s[sgprLoopCounterL] // advance C SRD
	s_addc_u32 s[sgprArgType2], s[sgprArgType2], 0         // carry from above
	buffer_load_d16_b16 v35, v16, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	v_add_lshl_u32 v15, v11, v[vgprGlobalReadOffsetA], 1
	v_mul_f32_e32 v18, s[sgprAlpha], v[vgprValuC]          // *= alpha
	v_mul_f32_e32 v19, s[sgprAlpha], v1                    // *= alpha
	v_mul_f32_e32 v20, s[sgprAlpha], v2                    // *= alpha
	v_mul_f32_e32 v21, s[sgprAlpha], v3                    // *= alpha
	v_mul_f32_e32 v22, s[sgprAlpha], v4                    // *= alpha
	v_mul_f32_e32 v23, s[sgprAlpha], v5                    // *= alpha
	v_mul_f32_e32 v24, s[sgprAlpha], v6                    // *= alpha
	v_mul_f32_e32 v25, s[sgprAlpha], v7                    // *= alpha
	s_waitcnt vmcnt(7) lgkmcnt(0)                          // wait for prior ops
	v_mul_f32_e32 v18, v28, v18                            // *= ScaleAlphaVec
	v_fma_mix_f32 v18, s[sgprBeta], v26, v18 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v27, v18         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v18, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v18.l, v18                           // convert to f16
	buffer_store_b16 v18, v15, s[16:19], 0 offen           // store D
	s_waitcnt vmcnt(6)                                     // wait for global read
	v_mul_f32_e32 v19, v28, v19                            // *= ScaleAlphaVec
	v_fma_mix_f32 v19, s[sgprBeta], v29, v19 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v27, v19         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v19, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v19.l, v19                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v19, v15, s[16:19], 0 offen           // store D
	s_waitcnt vmcnt(5)                                     // wait for global read
	v_mul_f32_e32 v20, v28, v20                            // *= ScaleAlphaVec
	v_fma_mix_f32 v20, s[sgprBeta], v30, v20 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v27, v20         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v20, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v20.l, v20                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v20, v15, s[16:19], 0 offen           // store D
	s_waitcnt vmcnt(4)                                     // wait for global read
	v_mul_f32_e32 v21, v28, v21                            // *= ScaleAlphaVec
	v_fma_mix_f32 v21, s[sgprBeta], v31, v21 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v27, v21         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v21, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v21.l, v21                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v21, v15, s[16:19], 0 offen           // store D
	s_waitcnt vmcnt(3)                                     // wait for global read
	v_mul_f32_e32 v22, v28, v22                            // *= ScaleAlphaVec
	v_fma_mix_f32 v22, s[sgprBeta], v32, v22 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v27, v22         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v22, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v22.l, v22                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v22, v15, s[16:19], 0 offen           // store D
	s_waitcnt vmcnt(2)                                     // wait for global read
	v_mul_f32_e32 v23, v28, v23                            // *= ScaleAlphaVec
	v_fma_mix_f32 v23, s[sgprBeta], v33, v23 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v27, v23         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v23, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v23.l, v23                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v23, v15, s[16:19], 0 offen           // store D
	s_waitcnt vmcnt(1)                                     // wait for global read
	v_mul_f32_e32 v24, v28, v24                            // *= ScaleAlphaVec
	v_fma_mix_f32 v24, s[sgprBeta], v34, v24 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v27, v24         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v24, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v24.l, v24                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v24, v15, s[16:19], 0 offen           // store D
	s_waitcnt vmcnt(0)                                     // wait for global read
	v_mul_f32_e32 v25, v28, v25                            // *= ScaleAlphaVec
	v_fma_mix_f32 v25, s[sgprBeta], v35, v25 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v27, v25         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v25, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v25.l, v25                           // convert to f16
	s_mul_i32 s[sgprLoopCounterL], s[sgprStrideD0], 4      // D stride offset
	s_add_u32 s16, s16, s[sgprLoopCounterL]
	s_addc_u32 s17, s17, 0                                 // carry from above
	buffer_store_b16 v25, v15, s[16:19], 0 offen           // store D
	s_nop 0                                                // 1 wait states
	s_branch label_Final_EndPgm                            // jump to Final_EndPgm
label_GS_MG2_BiasAct_NoBias:
	v_mov_b32_e32 v14, 0x80000000                          // OOB sentinel
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	v_add_lshl_u32 v26, v10, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v26, v14, v26, s[sgprAddressB]       // OOB offset if mask false
	buffer_load_d16_b16 v23, v26, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v27, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v27, 2, v27
	s_waitcnt lgkmcnt(0)                                   // wait for all LDS/SMEM
	s_barrier                                              // LDS barrier
	ds_load_b32 v24, v27                                   // L -> Reg
	ds_load_b32 v25, v27 offset:512                        // L -> Reg
	v_add_lshl_u32 v26, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v26, v14, v26, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	v_add_lshl_u32 v29, v10, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v29, v14, v29, s[sgprAddressB]       // OOB offset if mask false
	buffer_load_d16_b16 v28, v29, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v30, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v30, 2, v30
	v_add_lshl_u32 v29, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v29, v14, v29, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	v_add_lshl_u32 v32, v10, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v32, v14, v32, s[sgprAddressB]       // OOB offset if mask false
	buffer_load_d16_b16 v31, v32, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v33, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v33, 2, v33
	v_add_lshl_u32 v32, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v32, v14, v32, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	v_add_lshl_u32 v35, v10, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v35, v14, v35, s[sgprAddressB]       // OOB offset if mask false
	buffer_load_d16_b16 v34, v35, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v36, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v36, 2, v36
	v_add_lshl_u32 v35, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v35, v14, v35, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	v_add_lshl_u32 v38, v10, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v38, v14, v38, s[sgprAddressB]       // OOB offset if mask false
	buffer_load_d16_b16 v37, v38, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v39, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v39, 2, v39
	v_add_lshl_u32 v38, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v38, v14, v38, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	v_add_lshl_u32 v41, v10, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v41, v14, v41, s[sgprAddressB]       // OOB offset if mask false
	buffer_load_d16_b16 v40, v41, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v42, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v42, 2, v42
	v_add_lshl_u32 v41, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v41, v14, v41, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	v_add_lshl_u32 v44, v10, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v44, v14, v44, s[sgprAddressB]       // OOB offset if mask false
	buffer_load_d16_b16 v43, v44, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v45, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v45, 2, v45
	v_add_lshl_u32 v44, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v44, v14, v44, s[sgprAddressB]       // OOB offset if mask false
	v_add_co_u32 v[vgprGlobalReadOffsetB], vcc_lo, v[vgprGlobalReadOffsetB], 2 // advance coord by 2
	s_mul_i32 s[sgprAddressA], s[sgprStrideC0], 2          // C stride offset
	v_add_nc_i32 v10, v10, s[sgprAddressA]                 // advance C offset
	s_mul_i32 s[sgprAddressA], s[sgprStrideD0], 2          // D stride offset
	v_add_nc_i32 v11, v11, s[sgprAddressA]                 // advance D offset
	v_cmp_lt_u32_e64 s[sgprAddressA], v[vgprGlobalReadOffsetA], s[sgprSizesFree0] // coord0 < SizesFree0 ?
	v_cmp_lt_u32_e64 s[sgprAddressB], v[vgprGlobalReadOffsetB], s[sgprSizesFree1] // coord1 < SizesFree1 ?
	s_and_b32 s[sgprAddressB], s[sgprAddressA], s[sgprAddressB] // combine OOB masks
	v_add_lshl_u32 v47, v10, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v47, v14, v47, s[sgprAddressB]       // OOB offset if mask false
	buffer_load_d16_b16 v46, v47, s[sgprGemmInfo:sgprGemmInfo+3], 0 offen // load C (f16)
	s_mul_i32 s[sgprAddressA], 16, s[sgprWorkGroup0]       // tile offset
	v_sub_nc_u32_e64 v48, v[vgprGlobalReadOffsetA], s[sgprAddressA]
	v_lshlrev_b32_e32 v48, 2, v48
	v_add_lshl_u32 v47, v11, v[vgprGlobalReadOffsetA], 1
	v_cndmask_b32_e64 v47, v14, v47, s[sgprAddressB]       // OOB offset if mask false
	v_mul_f32_e32 v15, s[sgprAlpha], v[vgprValuC]          // *= alpha
	v_mul_f32_e32 v16, s[sgprAlpha], v1                    // *= alpha
	v_mul_f32_e32 v17, s[sgprAlpha], v2                    // *= alpha
	v_mul_f32_e32 v18, s[sgprAlpha], v3                    // *= alpha
	v_mul_f32_e32 v19, s[sgprAlpha], v4                    // *= alpha
	v_mul_f32_e32 v20, s[sgprAlpha], v5                    // *= alpha
	v_mul_f32_e32 v21, s[sgprAlpha], v6                    // *= alpha
	v_mul_f32_e32 v22, s[sgprAlpha], v7                    // *= alpha
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                // wait for all
	v_mul_f32_e32 v15, v25, v15                            // *= ScaleAlphaVec
	v_fma_mix_f32 v15, s[sgprBeta], v23, v15 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v24, v15         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v15, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v15.l, v15                           // convert to f16
	buffer_store_b16 v15, v26, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v16, v25, v16                            // *= ScaleAlphaVec
	v_fma_mix_f32 v16, s[sgprBeta], v28, v16 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v24, v16         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v16, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v16.l, v16                           // convert to f16
	buffer_store_b16 v16, v29, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v17, v25, v17                            // *= ScaleAlphaVec
	v_fma_mix_f32 v17, s[sgprBeta], v31, v17 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v24, v17         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v17, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v17.l, v17                           // convert to f16
	buffer_store_b16 v17, v32, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v18, v25, v18                            // *= ScaleAlphaVec
	v_fma_mix_f32 v18, s[sgprBeta], v34, v18 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v24, v18         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v18, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v18.l, v18                           // convert to f16
	buffer_store_b16 v18, v35, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v19, v25, v19                            // *= ScaleAlphaVec
	v_fma_mix_f32 v19, s[sgprBeta], v37, v19 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v24, v19         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v19, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v19.l, v19                           // convert to f16
	buffer_store_b16 v19, v38, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v20, v25, v20                            // *= ScaleAlphaVec
	v_fma_mix_f32 v20, s[sgprBeta], v40, v20 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v24, v20         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v20, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v20.l, v20                           // convert to f16
	buffer_store_b16 v20, v41, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v21, v25, v21                            // *= ScaleAlphaVec
	v_fma_mix_f32 v21, s[sgprBeta], v43, v21 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v24, v21         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v21, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v21.l, v21                           // convert to f16
	buffer_store_b16 v21, v44, s[16:19], 0 offen           // store D
	v_mul_f32_e32 v22, v25, v22                            // *= ScaleAlphaVec
	v_fma_mix_f32 v22, s[sgprBeta], v46, v22 op_sel_hi:[0,1,0] // D = beta*C + result
	v_add_f32_e32 v[vgprLocalWriteAddrA], v24, v22         // add bias
	s_swappc_b64 s[sgprAddressC:sgprAddressC+1], s[12:13]  // call activation func
	v_mov_b32_e32 v22, v[vgprLocalWriteAddrA]
	v_cvt_f16_f32_e32 v22.l, v22                           // convert to f16
	buffer_store_b16 v22, v47, s[16:19], 0 offen           // store D
	s_nop 0                                                // 1 wait states
	s_branch label_Final_EndPgm                            // jump to Final_EndPgm
label_Final_EndPgm:
	s_endpgm                                               // Kernel End

/******************************************/
/* Activation Functions                    */
/******************************************/

	s_setpc_b64 s[sgprAddressC:sgprAddressC+1]             // return
	v_mul_f32_e32 v[vgprLocalWriteAddrB], 0x3d372713, v[vgprLocalWriteAddrA]
	v_fma_f32 v[vgprLocalWriteAddrB], v[vgprLocalWriteAddrA], v[vgprLocalWriteAddrB], 1.0
	v_mul_f32_e32 v[vgprLocalWriteAddrB], v[vgprLocalWriteAddrA], v[vgprLocalWriteAddrB]
	v_mul_f32_e32 v[vgprLocalWriteAddrB], 0x40135761, v[vgprLocalWriteAddrB]
	v_exp_f32_e32 v[vgprLocalWriteAddrB], v[vgprLocalWriteAddrB] // exp(x)
	v_add_f32_e32 v[vgprLocalWriteAddrB], 1.0, v[vgprLocalWriteAddrB]
	v_rcp_f32_e32 v[vgprLocalWriteAddrB], v[vgprLocalWriteAddrB] // 1/x
	v_fma_f32 v[vgprLocalWriteAddrB], -2.0, v[vgprLocalWriteAddrB], 2.0
	v_mul_f32_e32 v[vgprLocalWriteAddrB], v[vgprLocalWriteAddrA], v[vgprLocalWriteAddrB]
	v_mul_f32_e32 v[vgprLocalWriteAddrA], 0.5, v[vgprLocalWriteAddrB]
	s_setpc_b64 s[sgprAddressC:sgprAddressC+1]             // return
	v_max_f32_e64 v[vgprLocalWriteAddrA], v[vgprLocalWriteAddrA], 0 // ReLU
	s_setpc_b64 s[sgprAddressC:sgprAddressC+1]             // return
	v_mul_f32_e32 v[vgprLocalWriteAddrA], 0xbfb8aa3b, v[vgprLocalWriteAddrA]
	v_exp_f32_e32 v[vgprLocalWriteAddrA], v[vgprLocalWriteAddrA] // exp(x)
	v_add_f32_e32 v[vgprLocalWriteAddrA], 1.0, v[vgprLocalWriteAddrA]
	v_rcp_f32_e32 v[vgprLocalWriteAddrA], v[vgprLocalWriteAddrA] // 1/x
	s_setpc_b64 s[sgprAddressC:sgprAddressC+1]             // return
	v_mul_f32_e32 v[vgprLocalWriteAddrB], 0xbfb8aa3b, v[vgprLocalWriteAddrA]
	v_exp_f32_e32 v[vgprLocalWriteAddrB], v[vgprLocalWriteAddrB] // exp(x)
	v_add_f32_e32 v[vgprLocalWriteAddrB], 1.0, v[vgprLocalWriteAddrB]
	v_rcp_f32_e32 v[vgprLocalWriteAddrB], v[vgprLocalWriteAddrB] // 1/x
	v_mul_f32_e32 v[vgprLocalWriteAddrA], v[vgprLocalWriteAddrA], v[vgprLocalWriteAddrB]
	s_setpc_b64 s[sgprAddressC:sgprAddressC+1]             // return
	v_min_f32_e32 v[vgprLocalWriteAddrA], s55, v[vgprLocalWriteAddrA] // clip max
	v_max_f32_e32 v[vgprLocalWriteAddrA], s54, v[vgprLocalWriteAddrA] // clip min
	s_setpc_b64 s[sgprAddressC:sgprAddressC+1]             // return
	s_endpgm                                               // Kernel End
.long 0x00000000, 0x00000000
.long 0x00000000, 0x00000000
